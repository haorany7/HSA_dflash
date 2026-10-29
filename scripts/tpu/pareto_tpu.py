"""Throughput vs per-user TPS sweep on TPU (torch_xla): AR and DFlash, bf16, static batching.

Same protocol and JSONL schema as ``scripts/hetero/pareto.py`` (first B GSM8K
questions, ``--max-new-tokens`` greedy tokens per row, EOS ignored, one identical
warmup run that also compiles every graph, decode time only):

    per-user TPS = max_new_tokens / decode_time
    throughput   = batch * max_new_tokens / decode_time

Differences from the GPU sweep: the target and draft are plain bf16 (no W8A8);
decoding uses ``dflash.batched_xla`` (static per-row KV cache, see its
docstring); ``mean_accept`` averages over rows that are still decoding.
Per-forward target/draft times are not split (lazy tensors run at the step's
graph boundary); ``ms_per_step`` is the whole step.

Usage (on a TPU VM / Colab TPU runtime, from the repo root):
    PYTHONPATH=. python scripts/tpu/pareto_tpu.py --system ar --batch 1 2 4 8 16 32 64 128
    PYTHONPATH=. python scripts/tpu/pareto_tpu.py --system dflash --block 4 8 16 --batch 1 2 4 8 16 32 64 128
"""
import argparse
import json
import os
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.batched_xla import batched_generate_xla
from dflash.benchmark import apply_chat_template, load_and_process_dataset

p = argparse.ArgumentParser()
p.add_argument("--system", choices=("ar", "dflash"), required=True)
p.add_argument("--block", type=int, nargs="+", default=[16])
p.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--dataset", default="gsm8k")
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
p.add_argument("--out", default="docs/data/pareto_tpu.jsonl")
p.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/xla_compile"))
args = p.parse_args()

# Persist compiled graphs so re-runs (and the AR/DFlash processes) skip recompiling.
xr.initialize_cache(args.cache_dir, readonly=False)
dev = torch_xla.device()
chip = os.environ.get("TPU_ACCELERATOR_TYPE") or xr.global_runtime_device_attributes()[0].get("name", "tpu")

t = time.perf_counter()
target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(dev)
draft = None
if args.system == "dflash":
    from dflash.model import DFlashDraftModel

    draft = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).eval().to(dev)
xm.wait_device_ops()
print(f"loaded on {chip} in {time.perf_counter() - t:.0f}s", flush=True)
blocks = [1] if args.system == "ar" else args.block

tok = AutoTokenizer.from_pretrained(args.model)
questions = load_and_process_dataset(args.dataset)
os.makedirs(os.path.dirname(args.out), exist_ok=True)


def compiles():
    return met.metric_data("CompileTime")[0] if met.metric_data("CompileTime") else 0


for block in blocks:
    for bsz in args.batch:
        prompts = [
            tok.encode(apply_chat_template(tok, [{"role": "user", "content": q["turns"][0]}], "off"),
                       return_tensors="pt", add_special_tokens=False)[0]
            for q in questions[:bsz]
        ]
        try:
            t = time.perf_counter()
            batched_generate_xla(draft, target, prompts, args.max_new_tokens, block, tok.pad_token_id)  # warmup + compile
            warmup = time.perf_counter() - t
            before = compiles()
            r = batched_generate_xla(draft, target, prompts, args.max_new_tokens, block, tok.pad_token_id)
            recompiles = compiles() - before
        except RuntimeError as e:  # XLA reports HBM OOM as RESOURCE_EXHAUSTED
            if "RESOURCE_EXHAUSTED" not in str(e) and "memory" not in str(e).lower():
                raise
            print(f"{args.system} block={block} B={bsz}: OOM, skipping larger batches", flush=True)
            break
        steps = [[a for a in s if a] for s in r.accept_lengths]
        accept = sum(map(sum, steps)) / sum(map(len, steps))
        mem = xm.get_memory_info(dev)
        rec = dict(
            system=f"tpu_{args.system}", chip=chip, block=block, batch=bsz, max_new_tokens=args.max_new_tokens,
            dataset=args.dataset, ttft_s=r.ttft, decode_s=r.decode_time, steps=r.steps,
            mean_accept=accept,
            tps_per_user=args.max_new_tokens / r.decode_time,
            throughput=bsz * args.max_new_tokens / r.decode_time,
            ms_per_step=1e3 * r.decode_time / r.steps,
            warmup_s=warmup, recompiles_in_timed_run=recompiles,
            peak_hbm_gib=mem.get("peak_bytes_used", mem.get("bytes_used", 0)) / 2**30,
        )
        print(f"tpu_{args.system:6s} block={block:2d} B={bsz:3d}  TPS/user={rec['tps_per_user']:7.2f}  "
              f"throughput={rec['throughput']:8.2f} tok/s  accept={accept:5.2f}  "
              f"step={rec['ms_per_step']:6.1f} ms  warmup={warmup:5.0f}s  recompiles={recompiles}", flush=True)
        if recompiles:
            print("  WARNING: graphs recompiled during the timed run; the point is skewed", flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
