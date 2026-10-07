"""Throughput vs per-user TPS sweep (static batching) for AR / DFlash on GPU / DFlash GPU+NPU.

Each (system, block_size, batch) point decodes ``--max-new-tokens`` greedy tokens
for the first B GSM8K questions (EOS ignored), after one identical warmup run.
Times are decode only (prefill is reported separately as TTFT):

    per-user TPS = max_new_tokens / decode_time
    throughput   = batch * max_new_tokens / decode_time

Results are appended to ``--out`` as JSON lines; plot with ``plot_pareto.py``.

Usage (one system per process, so GPU memory starts clean):
    dflash_env; export PYTHONPATH=scripts/hetero:$PYTHONPATH
    .venv/bin/python scripts/hetero/pareto.py --system ar --batch 1 2 4 8 16 32
    .venv/bin/python scripts/hetero/pareto.py --system gpu --block 4 8 16 --batch 1 2 4 8 16
    .venv/bin/python scripts/hetero/pareto.py --system hetero --block 16 --batch 1 2 4 8 16
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from dflash.batched import batched_generate
from dflash.benchmark import apply_chat_template, load_and_process_dataset

p = argparse.ArgumentParser()
p.add_argument("--system", choices=("ar", "gpu", "hetero"), required=True)
p.add_argument("--block", type=int, nargs="+", default=[16])
p.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16])
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--dataset", default="gsm8k")
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
p.add_argument("--out", default="docs/data/pareto.jsonl")
args = p.parse_args()

os.environ.setdefault("W8A8_COMPILE", "1")
# Fewer fragmentation OOMs as the KV cache grows by torch.cat every step.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
# Each (M, K, N) of the W8A8 GEMM is a dynamo guard; the default limit (8) would
# silently drop later shapes to eager and skew points against each other.
for _name in ("cache_size_limit", "recompile_limit"):
    if hasattr(torch._dynamo.config, _name):
        setattr(torch._dynamo.config, _name, 256)
from gpu_w8a8 import load_draft_w8a8, load_target_w8a8  # noqa: E402  (reads W8A8_COMPILE)

dev = torch.device("cuda")
target = load_target_w8a8(args.model, dev, quantize_lm_head=True, embed_on_host=True)
draft = None
if args.system == "gpu":
    draft = load_draft_w8a8(args.draft, dev)
elif args.system == "hetero":
    from dflash.model import DFlashDraftModel
    from dflash.npu import offload_draft

    draft = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).eval()
    offload_draft(draft)
    draft = draft.to(dev)
blocks = [1] if args.system == "ar" else args.block
if draft is None:  # AR still needs the draft config object for nothing; pass a stub
    from types import SimpleNamespace

    draft = SimpleNamespace()

# Split decode time into target vs draft (synchronizing is negligible next to a forward).
spent = {"target": 0.0, "draft": 0.0}


def timed(key, fn):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize()
            spent[key] += time.perf_counter() - t
    return wrap


target.forward = timed("target", target.forward)
if args.system != "ar":
    draft.forward = timed("draft", draft.forward)

tok = AutoTokenizer.from_pretrained(args.model)
questions = load_and_process_dataset(args.dataset)
os.makedirs(os.path.dirname(args.out), exist_ok=True)

for block in blocks:
    for bsz in args.batch:
        prompts = [
            tok.encode(apply_chat_template(tok, [{"role": "user", "content": q["turns"][0]}], "off"),
                       return_tensors="pt", add_special_tokens=False)[0]
            for q in questions[:bsz]
        ]
        try:
            batched_generate(draft, target, prompts, args.max_new_tokens, block, tok.pad_token_id)  # warmup
            spent.update(target=0.0, draft=0.0)
            r = batched_generate(draft, target, prompts, args.max_new_tokens, block, tok.pad_token_id)
        except torch.OutOfMemoryError:
            print(f"{args.system} block={block} B={bsz}: OOM, skipping larger batches")
            torch.cuda.empty_cache()
            break
        # spent includes the prefill forward; subtract it so the split matches decode_time
        target_decode = spent["target"] - r.ttft
        accept = sum(map(sum, r.accept_lengths)) / (r.steps * bsz)
        rec = dict(
            system=args.system, block=block, batch=bsz, max_new_tokens=args.max_new_tokens,
            dataset=args.dataset, ttft_s=r.ttft, decode_s=r.decode_time, steps=r.steps,
            mean_accept=accept,
            tps_per_user=args.max_new_tokens / r.decode_time,
            throughput=bsz * args.max_new_tokens / r.decode_time,
            target_ms_per_step=1e3 * target_decode / r.steps,
            draft_ms_per_step=1e3 * spent["draft"] / r.steps,
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        print(f"{args.system:6s} block={block:2d} B={bsz:2d}  TPS/user={rec['tps_per_user']:6.2f}  "
              f"throughput={rec['throughput']:7.2f} tok/s  accept={accept:5.2f}  "
              f"target={rec['target_ms_per_step']:6.1f} ms  draft={rec['draft_ms_per_step']:6.1f} ms/step", flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
