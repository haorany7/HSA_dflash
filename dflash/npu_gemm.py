#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array matrix multiply for skinny activations (DFlash draft on the NPU).

Vendored from mlir-aie ``programming_examples/basic/matrix_multiplication/
whole_array/whole_array.py`` (Apache-2.0 WITH LLVM-exception, header above).
Local change: the C drain pattern no longer assumes two (m * 4)-row blocks, so
``M = m * 4`` works and the weights (B) are streamed from DDR exactly once per
call. The upstream design needs ``M % (2 * m * 4) == 0``; with the draft's
16-32 activation rows that forced M=128 and read every weight twice.

Original description:
Whole-array matrix multiply — IRON API design with ``@iron.jit`` compilation.

A 4xN_cols AIE array computes ``C = A @ B`` (optionally with ``B`` column-major
or ``C`` column-major).  Each compute tile owns one (m, n) output sub-tile and
streams (m, k) x (k, n) inputs through three layers of ObjectFifos.

The script has two modes:

* ``--xclbin-path=... --insts-path=...`` — compile the design ahead-of-time
  and write artifacts to the given paths (bypasses the JIT cache).  Used by
  ``makefile-common`` so ``test.cpp`` + ``sweep.sh`` can drive the design
  via ``make``.
* default — JIT-compile + run on the attached NPU + verify against numpy.
"""


import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D  # noqa: F401
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    StreamDims,
    TaskGroup,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, from_name


def _device_for(dev_str, n_aie_cols):
    # On NPU1 pick the matching ColN variant (or NPU1 itself when
    # n_aie_cols == max = 4).  On NPU2 use the unrestricted device
    # regardless of n_aie_cols so the placer has the full 8-column array.
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def _build_design(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    c_col_maj,
    emulate_bf16_mmul_with_bfp16,
    use_chess,
    scalar,
    *,
    generate_taps=False,
):
    """Build the whole-array matmul IRON design and resolve to MLIR."""
    dev_str = "npu2" if isinstance(dev, NPU2) else "npu"

    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        use_chess=use_chess,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
        vectorized=not scalar,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    if dev_str == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev_str == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    assert (
        M % (m * n_aie_rows) == 0
    ), "A must be tileable into (m * n_aie_rows, k)-sized blocks"
    assert K % k == 0
    assert (
        N % (n * n_aie_cols) == 0
    ), "B must be tileable into (k, n * n_aie_cols)-sized blocks"
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_taps = []
    B_taps = []
    C_taps = []

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos: list[ObjectFifo] = []
    A_l2l1_fifos: list[ObjectFifo] = []
    B_l3l2_fifos: list[ObjectFifo] = []
    B_l2l1_fifos: list[ObjectFifo] = []
    C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos: list[ObjectFifo] = []

    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        a_dims: list[StreamDims] = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = a_l3l2.cons().split(
            of_offsets,
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp_fifos)

    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims: StreamDims = (
            [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
            if b_col_maj
            else [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        )
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=b_dims,
            )
        )

        c_dims: StreamDims = (
            [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
            if not c_col_maj
            else [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
        )
        c_l2l3 = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=c_dims,
        )
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_aie_rows)]

        c_tmp_fifos = c_l2l3.prod().join(
            of_offsets,
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp_fifos[j])

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1)  # Workaround for issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)

            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )

    tb_max_n_rows = 4 if not c_col_maj else 2
    tb_n_rows = min(tb_max_n_rows // 2, M // m // n_aie_rows)

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    if b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (N, K),
            (n, k),
            tile_group_repeats=(N // n // n_aie_cols, K // k),
            tile_group_steps=(n_aie_cols, 1),
            prune_step=False,
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),
            (k, n),
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True,
            prune_step=False,
        )
    if c_col_maj:
        # Splitting n_aie_rows out of the tile dim is what lets TensorTiler emit
        # the (col-fast, row_block-slow) DMA pattern; iter_col_major matches it.
        C_tiles = TensorTiler2D.step_tiler(
            (N, M),
            (n, m),
            tile_group_repeats=(N // n // n_aie_cols, n_aie_rows),
            tile_group_steps=(n_aie_cols, 1),
            iter_col_major=True,
            prune_step=False,
        )
    else:
        C_tiles = TensorTiler2D.step_tiler(
            (M, N),
            (m * n_aie_rows, n),
            tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            prune_step=False,
        )
    flat_workers = [w for row in workers for w in row]

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        c_index = 0
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):
                    C_taps.append(C_tiles[c_index])
                    C_hs[col].drain(
                        C,
                        tap=C_tiles[c_index],
                        wait=True,
                        group=tg,
                    )
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)
                        if col < n_aie_rows:
                            A_hs[col].fill(
                                A,
                                tap=A_tiles[tile_offset],
                                group=tg,
                            )
                        B_hs[col].fill(
                            B,
                            tap=B_tiles[col],
                            group=tg,
                        )
                        A_taps.append(A_tiles[tile_offset])
                        B_taps.append(B_tiles[col])

                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses],
    )

    # resolve_program() runs the sequence body, which populates the closure
    # A_taps/B_taps/C_taps lists as a side effect. The generate_taps path needs
    # that side effect, so resolve first and discard the module for that mode.
    program = Program(dev, rt, workers=flat_workers)
    module = program.resolve_program()

    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    return module


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 0,
    c_col_maj: CompileTime[int] = 0,
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    use_chess: CompileTime[bool] = False,
    scalar: CompileTime[bool] = False,
):
    return _build_design(
        iron.get_current_device(),
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        dtype_in_str,
        dtype_out_str,
        b_col_maj,
        c_col_maj,
        emulate_bf16_mmul_with_bfp16,
        use_chess,
        scalar,
    )
