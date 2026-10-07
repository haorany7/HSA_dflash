"""Run DFlash draft-model linear layers on an AMD Ryzen AI NPU (AIE2/AIE2P).

Every ``nn.Linear`` in the draft model is replaced by :class:`NpuLinear`, which
dispatches ``x @ W.T`` to the mlir-aie whole-array bf16 GEMM design
(``programming_examples/basic/matrix_multiplication/whole_array``) compiled
through ``@iron.jit``.  Weights are uploaded to NPU-visible buffers once;
activations are padded up to a bucketed row count so only a handful of GEMM
shapes ever get compiled.  Everything else (norms, RoPE, attention, the target
model and the LM head) stays on the torch device.

Requires the ``aie`` Python package (mlir-aie / IRON) and an mlir-aie source
checkout for the GEMM design; point ``MLIR_AIE_SRC`` at it if it is not
``/notebooks/mlir-aie``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_WHOLE_ARRAY_DIR = "programming_examples/basic/matrix_multiplication/whole_array"

# Per-core tile sizes (m x k x n) and array width.  The design needs
# M % (2 * m * 4) == 0, K % k == 0 and N % (n * n_cols) == 0.
_TILE_M, _TILE_K, _TILE_N, _N_COLS = 16, 64, 64, 8
_M_ALIGN = 2 * _TILE_M * 4
# Larger K overflows the design's B-side DMA strides (limit 2**20), so wider
# Linears are split along K and the f32 partial products are summed on the host.
_MAX_K = 4096

_gemm = None


def _whole_array():
    global _gemm
    if _gemm is None:
        root = Path(os.environ.get("MLIR_AIE_SRC", "/notebooks/mlir-aie"))
        sys.path.insert(0, str(root / _WHOLE_ARRAY_DIR))
        import aie.iron as iron
        import whole_array

        iron.set_current_device(whole_array._device_for("npu2", _N_COLS))
        _gemm = whole_array.whole_array
    return _gemm


def _bucket(rows: int) -> int:
    """Round rows up to a power-of-two multiple of the M alignment."""
    padded = _M_ALIGN
    while padded < rows:
        padded *= 2
    return padded


def _to_numpy_bf16(x: torch.Tensor) -> np.ndarray:
    from ml_dtypes import bfloat16

    return x.detach().to("cpu", torch.bfloat16).contiguous().view(torch.uint16).numpy().view(bfloat16)


class NpuLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        import aie.iron as iron
        from ml_dtypes import bfloat16

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if self.in_features % _TILE_K or self.out_features % (_TILE_N * _N_COLS):
            raise ValueError(
                f"NPU GEMM cannot tile Linear({self.in_features}, {self.out_features})"
            )
        self.bias = linear.bias
        # nn.Linear stores W as (out, in) row-major == (in, out) column-major,
        # which is exactly the whole_array design's b_col_maj layout.
        w = _to_numpy_bf16(linear.weight)
        self.k_chunk = min(self.in_features, _MAX_K)
        if self.in_features % self.k_chunk:
            raise ValueError(f"NPU GEMM cannot split K={self.in_features}")
        self.weights = [
            iron.tensor(np.ascontiguousarray(w[:, k0 : k0 + self.k_chunk]), dtype=bfloat16, device="npu")
            for k0 in range(0, self.in_features, self.k_chunk)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import aie.iron as iron
        from ml_dtypes import bfloat16

        shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        rows = x2.shape[0]
        padded = _bucket(rows)
        x_np = _to_numpy_bf16(x2)
        acc = None
        for i, weight in enumerate(self.weights):
            k0 = i * self.k_chunk
            a = np.zeros((padded, self.k_chunk), dtype=bfloat16)
            a[:rows] = x_np[:, k0 : k0 + self.k_chunk]
            a = iron.tensor(a, dtype=bfloat16, device="npu")
            # bf16 output loses accuracy inside the design at K=4096, so take f32.
            c = iron.zeros((padded, self.out_features), dtype=np.float32, device="npu")
            _whole_array()(
                a, weight, c,
                M=padded, K=self.k_chunk, N=self.out_features,
                m=_TILE_M, k=_TILE_K, n=_TILE_N, n_aie_cols=_N_COLS,
                dtype_in_str="bf16", dtype_out_str="f32", b_col_maj=1,
            )
            part = c.numpy().reshape(padded, self.out_features)[:rows]
            acc = part.copy() if acc is None else acc + part
        out = torch.from_numpy(acc)
        out = out.to(x.device, x.dtype).view(*shape, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


def offload_linears(module: nn.Module) -> int:
    """Replace every tileable nn.Linear under ``module`` with an NpuLinear."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            try:
                setattr(module, name, NpuLinear(child))
                count += 1
            except ValueError:
                pass
        else:
            count += offload_linears(child)
    return count
