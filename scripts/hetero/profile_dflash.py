"""Per-stage wall-time breakdown of one DFlash greedy generation (CPU target)."""
import argparse
import os
import time
from collections import defaultdict

import torch

from dflash.benchmark import apply_chat_template, load_transformers_models, stop_token_ids
from dflash.model import dflash_generate

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
p.add_argument("--device", default="cpu")
p.add_argument("--draft-device", default="same")
p.add_argument("--w8a8", action="store_true")
p.add_argument("--block-size", type=int, default=None, help="1 = plain autoregressive with the same target")
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--prompt", default="How many positive whole-number divisors does 196 have? Explain briefly.")
args = p.parse_args()

dev = torch.device(args.device)
sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
if args.w8a8:
    from gpu_w8a8 import load_draft_w8a8, load_target_w8a8
    from transformers import AutoTokenizer
    t_load = time.perf_counter()
    target = load_target_w8a8(args.model, dev, quantize_lm_head=os.environ.get("W8A8_HEAD") == "1")
    if args.draft_device == "npu":
        from dflash.model import DFlashDraftModel
        from dflash.npu import offload_draft

        draft = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).eval()
        offload_draft(draft)
        draft = draft.to(dev)
    else:
        draft = load_draft_w8a8(args.draft, dev)
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"w8a8 load {time.perf_counter() - t_load:.1f}s  gpu mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
else:
    target, draft, tok = load_transformers_models(args.model, args.draft, dev, args.draft_device)
head = target.lm_head

T = defaultdict(float)
N = defaultdict(int)
stack = []  # which top-level module is running, so lm_head time is attributed correctly


def timed(key, fn):
    def wrap(*a, **k):
        sync()
        t = time.perf_counter()
        stack.append(key)
        try:
            return fn(*a, **k)
        finally:
            stack.pop()
            sync()
            T[key] += time.perf_counter() - t
            N[key] += 1
    return wrap


target.forward = timed("target_fwd", target.forward)
draft.forward = timed("draft_body", draft.forward)
draft.compute_logits = timed("draft_lm_head", draft.compute_logits)
head_fwd = head.forward


def head_wrap(*a, **k):
    if stack and stack[-1] == "target_fwd":
        return timed("  (target lm_head)", head_fwd)(*a, **k)
    return head_fwd(*a, **k)


head.forward = head_wrap

prompt = apply_chat_template(tok, [{"role": "user", "content": args.prompt}], "off")
ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
stops = stop_token_ids(target, tok)

with torch.inference_mode():
    dflash_generate(draft, target, ids, args.max_new_tokens, stops, block_size=args.block_size, return_stats=True)  # warmup (NPU compile, allocs)
    T.clear(); N.clear()
    t0 = time.perf_counter()
    r = dflash_generate(draft, target, ids, args.max_new_tokens, stops, block_size=args.block_size, return_stats=True)
    total = time.perf_counter() - t0
print("TEXT:", repr(tok.decode(r.output_ids[0, ids.shape[1]:], skip_special_tokens=True)[:300]))
if dev.type == "cuda":
    print(f"peak gpu mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

acc = sum(r.acceptance_lengths) / len(r.acceptance_lengths)
steps = len(r.acceptance_lengths)
print(f"device={args.device} draft={args.draft_device} threads={torch.get_num_threads()}")
print(f"out={r.num_output_tokens} steps={steps} mean_accept={acc:.2f} TTFT={r.time_to_first_token:.2f}s "
      f"decode={1 / r.time_per_output_token:.2f} tok/s total={total:.2f}s")
print(f"(target_fwd count includes 1 prefill = {r.time_to_first_token:.2f}s)")
for k in ["target_fwd", "  (target lm_head)", "draft_body", "draft_lm_head"]:
    print(f"{k:22s} n={N[k]:3d}  total={T[k]:7.2f}s  {100 * T[k] / total:5.1f}%  per-call={1000 * T[k] / max(N[k], 1):8.1f} ms")
other = total - T["target_fwd"] - T["draft_body"] - T["draft_lm_head"]
print(f"{'other (host glue)':22s}        total={other:7.2f}s  {100 * other / total:5.1f}%")
