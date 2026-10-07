"""Run DFlash draft-model projections on an AMD Ryzen AI NPU (AIE2P / Strix).

The draft's GEMMs are skinny (16 noise rows plus a few context rows against
4096-24576 wide weights), so they are bound by streaming weights from DDR, not
by compute. Everything here is arranged to read each weight byte once per
draft step:

* :mod:`dflash.npu_gemm` streams B once when ``M == 64`` (one 4 x m row block).
* Attention q/k/v are fused into one GEMM over ``[context; noise]`` rows (the
  unfused path calls k_proj and v_proj twice each), via ``qkv_proj`` on
  ``Qwen3DFlashAttention``. MLP gate/up are fused the same way.
* Every call has the same shape (64 x 4096 x 4096): weights are pre-split into
  4096 x 4096 blocks (zero-padded along N where needed). Each GEMM shape is
  its own xclbin, and switching between them costs ~2.6 ms per call on Strix
  (measured; a same-shape call is ~1.4 ms), so one shape means no switching.
* Activation and output buffers are allocated once and reused by every call.

Norms, RoPE, attention and the LM head stay on the torch device. Requires the
``aie`` Python package (mlir-aie / IRON) and ``pyxrt``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Per-core tile sizes (m x k x n) and array width. The design needs
# M % (m * 4) == 0, K % k == 0 and N % (n * n_cols) == 0.
_TILE_M, _TILE_K, _TILE_N, _N_COLS = 16, 128, 64, 8
_ROW_BLOCK = _TILE_M * 4
# Block size along K and N. Larger K overflows the design's B-side DMA strides
# (limit 2**20); K blocks' f32 partial products are summed on the host.
_BLOCK = 4096

_gemm = None
_buffers: dict = {}


def _whole_array():
    global _gemm
    if _gemm is None:
        import aie.iron as iron

        from . import npu_gemm

        iron.set_current_device(npu_gemm._device_for("npu2", _N_COLS))
        _gemm = npu_gemm.whole_array
    return _gemm


def _bucket(rows: int) -> int:
    """Round rows up to a power-of-two multiple of one 64-row block."""
    padded = _ROW_BLOCK
    while padded < rows:
        padded *= 2
    return padded


def _buffer(shape: tuple[int, int], dtype, slot: int = 0):
    """Shared NPU buffer per (shape, dtype, slot); safe because calls are sequential."""
    import aie.iron as iron

    key = (shape, np.dtype(dtype), slot)
    if key not in _buffers:
        _buffers[key] = iron.zeros(shape, dtype=dtype, device="npu")
    return _buffers[key]


def _as_torch(array: np.ndarray) -> torch.Tensor:
    from aie.utils.hostruntime.torch_interop import _array_to_torch

    return _array_to_torch(array)


class NpuLinear(nn.Module):
    """``x @ W.T (+ b)`` on the NPU for a weight stored as torch (out, in)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        import aie.iron as iron
        from ml_dtypes import bfloat16

        self.out_features, self.in_features = weight.shape
        if self.in_features % _BLOCK:
            raise ValueError(f"NPU GEMM needs in_features % {_BLOCK} == 0, got {self.in_features}")
        self.bias = bias
        w = weight.detach().to("cpu", torch.bfloat16)
        n_pad = -self.out_features % _BLOCK
        if n_pad:
            w = torch.cat([w, w.new_zeros(n_pad, self.in_features)])
        # blocks[i][j] holds W[i-th N block, j-th K block]. torch (out, in)
        # row-major == (in, out) column-major, the design's b_col_maj layout.
        self.blocks = []
        for n0 in range(0, w.shape[0], _BLOCK):
            row = []
            for k0 in range(0, self.in_features, _BLOCK):
                t = iron.zeros((_BLOCK, _BLOCK), dtype=bfloat16, device="npu")
                with t.overwrite() as buf:
                    _as_torch(buf).copy_(w[n0 : n0 + _BLOCK, k0 : k0 + _BLOCK])
                row.append(t)
            self.blocks.append(row)

    @classmethod
    def fused(cls, linears: list[nn.Linear]) -> NpuLinear:
        """One NPU GEMM for several Linears that share an input (outputs concatenated)."""
        bias = None
        if any(lin.bias is not None for lin in linears):
            bias = torch.cat([
                lin.bias if lin.bias is not None else lin.weight.new_zeros(lin.out_features)
                for lin in linears
            ])
        module = cls(torch.cat([lin.weight for lin in linears]), bias)
        module.split_sizes = [lin.out_features for lin in linears]
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from ml_dtypes import bfloat16

        shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).to("cpu", torch.bfloat16)
        rows = x2.shape[0]
        padded = _bucket(rows)
        n_k = len(self.blocks[0])
        # One A buffer per K block, filled once and reused across N blocks.
        a = [_buffer((padded, _BLOCK), bfloat16, slot=j) for j in range(n_k)]
        for j in range(n_k):
            with a[j].overwrite() as buf:
                _as_torch(buf)[:rows].copy_(x2[:, j * _BLOCK : (j + 1) * _BLOCK])
        c = _buffer((padded, _BLOCK), np.float32)
        out = torch.empty(rows, len(self.blocks) * _BLOCK)
        for i, row in enumerate(self.blocks):
            dst = out[:, i * _BLOCK : (i + 1) * _BLOCK]
            for j, weight in enumerate(row):
                _whole_array()(
                    a[j], weight, c,
                    M=padded, K=_BLOCK, N=_BLOCK,
                    m=_TILE_M, k=_TILE_K, n=_TILE_N, n_aie_cols=_N_COLS,
                    dtype_in_str="bf16", dtype_out_str="f32", b_col_maj=1,
                )
                part = _as_torch(c.numpy())[:rows]
                if j == 0:
                    dst.copy_(part)
                else:
                    dst.add_(part)
        out = out[:, : self.out_features].to(device=x.device, dtype=x.dtype)
        out = out.view(*shape, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


class NpuMLP(nn.Module):
    """SwiGLU MLP with gate/up fused into one NPU GEMM."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.gate_up = NpuLinear.fused([mlp.gate_proj, mlp.up_proj])
        self.down = NpuLinear(mlp.down_proj.weight, mlp.down_proj.bias)
        self.act_fn = mlp.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).split(self.gate_up.split_sizes, dim=-1)
        return self.down(self.act_fn(gate) * up)


def offload_draft(module: nn.Module) -> int:
    """Move the draft's projections to the NPU in place; returns NPU modules created."""
    count = 0
    for name, child in list(module.named_children()):
        if type(child).__name__ == "Qwen3DFlashAttention":
            child.qkv_proj = NpuLinear.fused([child.q_proj, child.k_proj, child.v_proj])
            del child.q_proj, child.k_proj, child.v_proj
            child.o_proj = NpuLinear(child.o_proj.weight, child.o_proj.bias)
            count += 2
        elif all(hasattr(child, p) for p in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            setattr(module, name, NpuMLP(child))
            count += 1
        elif isinstance(child, nn.Linear):
            try:
                setattr(module, name, NpuLinear(child.weight, child.bias))
                count += 1
            except ValueError:
                pass
        else:
            count += offload_draft(child)
    return count
