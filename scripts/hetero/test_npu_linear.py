import time, torch
from torch import nn
from dflash.npu import NpuLinear
torch.manual_seed(0)
for (k, n, rows) in [(4096, 4096, 16), (4096, 1024, 16), (4096, 12288, 16), (12288, 4096, 16), (4096, 4096, 100), (20480, 4096, 16)]:
    lin = nn.Linear(k, n, bias=False).to(torch.bfloat16)
    x = torch.randn(rows, k, dtype=torch.bfloat16)
    ref = lin(x).float()
    t = time.time(); m = NpuLinear(lin); t_up = time.time() - t
    t = time.time(); y = m(x); t_first = time.time() - t
    ts = []
    for _ in range(5):
        t = time.time(); y = m(x); ts.append(time.time() - t)
    err = ((y.float() - ref).norm() / ref.norm()).item()
    tc = time.time(); [lin(x) for _ in range(5)]; tcpu = (time.time() - tc) / 5
    print(f"K={k} N={n} rows={rows}: rel_err={err:.2e} first={t_first:.1f}s steady={min(ts)*1e3:.1f}ms cpu={tcpu*1e3:.1f}ms", flush=True)
