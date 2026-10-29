# TPU Pareto sweep: status and context

Working notes for the `tpu-pareto` branch, written so a fresh agent/session can pick up
where the last one stopped. Update the "Status" section when something changes.

## Goal

Measure the throughput vs per-user TPS Pareto frontier of DFlash speculative decoding
(Qwen3-8B target + `z-lab/Qwen3-8B-DFlash-b16` draft, bf16) on a Cloud TPU, and compare
it with plain AR decoding and the earlier Ryzen GPU / GPU+NPU results
(`docs/data/pareto.jsonl`, `docs/pareto.png`).

Protocol (same as `scripts/hetero/pareto.py`): first B GSM8K questions, 128 greedy new
tokens per row, EOS ignored, one identical warmup run (also compiles every graph), decode
time only. `tps_per_user = 128 / decode_s`, `throughput = B * 128 / decode_s`.

## Code map

- `dflash/batched_xla.py`: static-shape batched greedy DFlash/AR for XLA (per-row slot KV
  cache, host-side bookkeeping, one graph boundary per step). `block_size=1` is AR.
- `dflash/batched.py`: reference batched implementation (GPU path).
- `scripts/hetero/test_batched_xla.py [cpu|xla]`: checks `batched_xla` against
  `batched` on tiny random models with an oracle draft; must print `PASS`.
- `scripts/tpu/pareto_tpu.py`: one sweep (AR or DFlash), appends rows to
  `docs/data/pareto_tpu.jsonl`.
- `scripts/tpu/plot_pareto_tpu.py`: plot, optional Ryzen overlay.
- `scripts/tpu/run_sweep.sh`: setup + test + AR + DFlash + plot.

## Environment (Colab TPU v6e-1, Python 3.13 system image)

The setup step in `run_sweep.sh` (`pip install torch_xla[tpu]`) does **not** work on this
image: it installs torch_xla 2.8.1 against the preinstalled torch 2.14 (ABI mismatch,
`undefined symbol` on import) and downgrades the system libtpu, breaking the
preinstalled jax / torch_tpu. libtpu 0.0.21 (needed by torch_xla 2.9) has no cp313
wheel. So torch_xla lives in a separate Python 3.12 venv:

```bash
# UV_SYSTEM_PYTHON=true is set in this image, so always pass --python explicitly
uv python install 3.12
uv venv -p 3.12 --managed-python /content/xla_venv
P=/content/xla_venv/bin/python
uv pip install --python $P "torch==2.9.0" "torch_xla[tpu]==2.9.0" "transformers==5.15.0" \
  "tokenizers==0.22.2" datasets matplotlib accelerate \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -f https://storage.googleapis.com/libtpu-releases/index.html
uv pip install --python $P -e . --no-deps
```

Run everything with that interpreter, e.g.

```bash
cd /content/HSA_dflash
SKIP_SETUP=1 PATH=/content/xla_venv/bin:$PATH bash scripts/tpu/run_sweep.sh
# or a single point:
PYTHONPATH=. /content/xla_venv/bin/python scripts/tpu/pareto_tpu.py --system dflash --block 16 --batch 1
```

Leave the system Python alone (it keeps torch 2.14 / torch_tpu / jax / libtpu 0.0.50 /
transformers 5.20.2).

## Fixes made on this branch (commit 5b5e11c)

1. **Per-step recompilation in `batched_xla.py`.** `.cpu()` on the step's token ids only
   executes that subgraph; the in-place KV-cache `scatter_` writes stayed pending, so
   step k's graph contained all k previous cache updates and every step compiled a new,
   larger graph (the tiny XLA test ran >13 min without finishing). Fixed by calling
   `_mark_step(device)` before each host download. Now 2 compiles per shape (prefill +
   decode step), independent of the number of steps; the test passes on TPU in ~20 s.
   Check with `torch_xla.debug.metrics.metric_data("CompileTime")` or
   `PT_XLA_DEBUG_LEVEL=1` if this ever regresses; `recompiles_in_timed_run` in the
   JSONL must be 0.
2. `dflash/batched.py`: `t_valid.cumsum` -> `t_valid.long().cumsum` (bool cumsum stays
   bool on XLA; identical on CPU/CUDA).
3. `test_batched_xla.py`: reference always runs on CPU (the GPU reference path does not
   run under XLA: inference-tensor errors); only `batched_xla` runs on the device.

## Status (2026-10-29)

- Equivalence test: PASS on CPU and TPU.
- Full sweep (`run_sweep.sh`, AR + DFlash blocks 4/8/16, B = 1..128) finished: 29 points,
  all with 0 recompiles in the timed run. Results in `docs/data/pareto_tpu.jsonl`, log
  `docs/data/sweep_tpu.log`, plot `docs/pareto_tpu.png` (log axes, Ryzen overlay).
- Every DFlash block size OOMed at B=128 (AR B=128 fits).

### Results (v6e-1, bf16, GSM8K, 128 new tokens)

TPS/user / throughput (tok/s); best throughput per B in bold.

| B | AR | DFlash blk 4 | DFlash blk 8 | DFlash blk 16 |
|---|---|---|---|---|
| 1 | 16.6 / 16.6 | 40.5 / 40.5 | 45.9 / 45.9 | 52.7 / **52.6** |
| 2 | 16.2 / 32.3 | 39.7 / 79.5 | 46.0 / 92.0 | 49.2 / **98.5** |
| 4 | 15.5 / 62.0 | 37.1 / 148.4 | 43.0 / 172.1 | 53.6 / **214.4** |
| 8 | 14.9 / 118.8 | 30.8 / 246.0 | 35.9 / 287.3 | 38.8 / **310.4** |
| 16 | 12.5 / 200.0 | 28.5 / 455.9 | 26.4 / 423.0 | 29.5 / **472.4** |
| 32 | 10.8 / 346.2 | 23.0 / 734.4 | 28.8 / **921.5** | 24.0 / 766.3 |
| 64 | 8.1 / 515.0 | 17.4 / **1115.1** | 16.8 / 1076.8 | 14.3 / 915.5 |
| 128 | 5.1 / 647.9 | OOM | OOM | OOM |

Mean accept length: blk 4 ~2.8-3.1, blk 8 ~3.4-4.0, blk 16 ~3.9-5.1.

Takeaways:
- DFlash beats AR on both axes at every B tested. Peak throughput 1115 tok/s (blk 4,
  B=64) vs AR's 648 (B=128), while each user still gets 17.4 vs 5.1 tok/s.
- At AR's best per-user speed (~16.6 tok/s, B=1), DFlash serves B=64 at ~17 tok/s per
  user, about 65x the throughput.
- The best block shrinks as B grows (16 for B<=16, 8 at B=32, 4 at B=64): at large B
  the verify cost of long blocks outweighs the extra accepted tokens.
- Step time is flat (~60-90 ms) up to B~8, i.e. host/launch overhead, not compute, sets
  small-batch speed (see Caveats).
- Some noise: blk 8 B=16 is slower per user than B=32, and the B=1 blk 16 smoke run
  gave 59.5 TPS/user vs 52.7 in the sweep. Accept length varies with which questions
  are in the batch; single runs, no repeats.

### Next steps (not started)

- Cut host overhead (`torch_xla.compile` / dynamo `openxla`), then re-sweep.
- Retry DFlash B=128 in a fresh process; reduce memory if it still OOMs.
- Repeat runs for error bars.

## Caveats / open questions

- **Host overhead dominates small batches.** AR B=1 is ~61 ms/step, ~6x the HBM
  bandwidth floor for 16 GB of bf16 weights on v6e (~10 ms). Most likely lazy-tensor
  tracing of the HF model on the host every step. Fixing it (e.g. `torch_xla.compile` /
  dynamo `openxla` backend, or caching the traced step) would move the whole frontier.
- `peak_hbm_gib` is the process-wide peak (`xm.get_memory_info`), not per point, so it
  only increases within one `pareto_tpu.py` run.
- DFlash B=128 hit OOM (32 GiB HBM) for every block after B=64 succeeded, in the same process
  as smaller points; it may fit in a fresh process (fragmentation / earlier buffers).
  Untested.
- The persistent compile cache (`xr.initialize_cache`) logs "Deserializing serialized
  executable not supported" on this torch_xla/libtpu; harmless, it just recompiles in
  every process (warmup ~20-80 s per point).
- `mean_accept` averages over rows still decoding; it varies with B partly because
  different GSM8K questions are included.
