"""Break one NPU whole_array GEMM call into stages to find where the time goes.

Usage: dflash_env; .venv/bin/python scripts/hetero/bench_npu_gemm.py [--K 4096 --N 12288 --M 128]
"""
import argparse
import time

import numpy as np
from ml_dtypes import bfloat16

from dflash.npu import _N_COLS, _TILE_K, _TILE_M, _TILE_N, _whole_array

p = argparse.ArgumentParser()
p.add_argument("--M", type=int, default=128)
p.add_argument("--K", type=int, default=4096)
p.add_argument("--N", type=int, default=12288)
p.add_argument("--m", type=int, default=_TILE_M)
p.add_argument("--k", type=int, default=_TILE_K)
p.add_argument("--n", type=int, default=_TILE_N)
p.add_argument("--dtype", choices=("bf16", "i8"), default="bf16")
p.add_argument("--iters", type=int, default=20)
args = p.parse_args()

gemm = _whole_array()
import aie.iron as iron  # noqa: E402  (after _whole_array sets the device)

M, K, N = args.M, args.K, args.N
i8 = args.dtype == "i8"
in_t, out_t = (np.int8, np.int32) if i8 else (bfloat16, np.float32)
kw = dict(M=M, K=K, N=N, m=args.m, k=args.k, n=args.n, n_aie_cols=_N_COLS,
          dtype_in_str=args.dtype, dtype_out_str="i32" if i8 else "f32", b_col_maj=1)
rng = np.random.default_rng(0)
if i8:
    a_np = rng.integers(-127, 128, (M, K), dtype=np.int8)
    w_np = rng.integers(-127, 128, (N, K), dtype=np.int8)
else:
    a_np = rng.standard_normal((M, K), dtype=np.float32).astype(bfloat16)
    w_np = rng.standard_normal((N, K), dtype=np.float32).astype(bfloat16)

a = iron.tensor(a_np, dtype=in_t, device="npu")
w = iron.tensor(w_np, dtype=in_t, device="npu")
c = iron.zeros((M, N), dtype=out_t, device="npu")
gemm(a, w, c, **kw)  # compile / warm
ref = a_np.astype(np.float64) @ w_np.astype(np.float64).T
err = np.abs(c.numpy().reshape(M, N) - ref).max() / np.abs(ref).max()


def bench(fn, n=args.iters):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


t_call = bench(lambda: gemm(a, w, c, **kw))


def touch_a_call():
    a.numpy()[:] = a_np  # host write -> forces A re-sync before the call
    gemm(a, w, c, **kw)


t_touch = bench(touch_a_call)
t_read = bench(lambda: c.numpy().reshape(M, N)[:16].copy())
t_alloc = bench(lambda: (iron.tensor(a_np, dtype=in_t, device="npu"),
                         iron.zeros((M, N), dtype=out_t, device="npu")))
wbytes = N * K * np.dtype(in_t).itemsize
print(f"M={M} K={K} N={N} m,k,n={args.m},{args.k},{args.n} relerr={err:.1e}")
print(f"  call, buffers reused      {t_call:7.2f} ms   -> weight stream {wbytes / t_call / 1e6:6.1f} GB/s")
print(f"  call + host write of A    {t_touch:7.2f} ms")
print(f"  read back C (16 rows)     {t_read:7.2f} ms")
print(f"  alloc A + C               {t_alloc:7.2f} ms")
