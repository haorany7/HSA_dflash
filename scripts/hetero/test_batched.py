"""Check dflash.batched against dflash_generate (greedy, no EOS) on the GPU W8A8 setup.

Usage: dflash_env; PYTHONPATH=scripts/hetero:$PYTHONPATH .venv/bin/python scripts/hetero/test_batched.py
"""
import torch
from transformers import AutoTokenizer

from dflash.batched import batched_generate
from dflash.benchmark import apply_chat_template
from dflash.model import dflash_generate
from gpu_w8a8 import load_draft_w8a8, load_target_w8a8

import os
SAME = int(os.environ["SAME"]) if "SAME" in os.environ else None  # repeat one prompt: no padding
MODEL, DRAFT, N = "Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", 64
QUESTIONS = [
    "How many positive whole-number divisors does 196 have? Explain briefly.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
    "Explain the difference between TCP and UDP in three sentences.",
]

dev = torch.device("cuda")
target = load_target_w8a8(MODEL, dev, quantize_lm_head=True)
draft = load_draft_w8a8(DRAFT, dev)
tok = AutoTokenizer.from_pretrained(MODEL)
prompts = [
    tok.encode(apply_chat_template(tok, [{"role": "user", "content": q}], "off"),
               return_tensors="pt", add_special_tokens=False)[0]
    for q in QUESTIONS
]


def agree(a, b):
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return n


for block in (16, 1):
    ref = []
    for p in prompts:
        r = dflash_generate(draft, target, p[None].to(dev), N, None, block_size=block)
        ref.append(r[0, len(p):].tolist())
    one = batched_generate(draft, target, prompts[:1], N, block_size=block, pad_token_id=tok.pad_token_id)
    many = batched_generate(draft, target, prompts if SAME is None else [prompts[SAME]] * 4, N, block_size=block, pad_token_id=tok.pad_token_id)
    print(f"block={block}: B=1 agrees with reference for {agree(one.tokens[0], ref[0])}/{N} tokens")
    for b in range(len(prompts)):
        print(f"  B=4 row {b}: agrees for {agree(many.tokens[b], ref[b if SAME is None else SAME])}/{N} tokens")
    if block > 1:
        mean = sum(sum(s) for s in many.accept_lengths) / (len(many.accept_lengths) * len(prompts))
        print(f"  B=4 steps={many.steps} mean accept/step={mean:.2f}")
print("row0 text:", repr(tok.decode(many.tokens[0])[:120]))

# Teacher-forcing check: a correct greedy decoder emits the target's argmax at
# every position; under bf16/W8A8 noise it may differ only where the top two
# logits are nearly tied.
many = batched_generate(draft, target, prompts, N, block_size=16, pad_token_id=tok.pad_token_id)
for b, p in enumerate(prompts):
    seq = torch.cat([p, torch.tensor(many.tokens[b])]).to(dev)[None]
    with torch.inference_mode():
        logits = target(seq, use_cache=False).logits[0, len(p) - 1 : -1].float()
    gen = torch.tensor(many.tokens[b], device=dev)
    margin = logits.max(-1).values - logits.gather(1, gen[:, None])[:, 0]
    bad = (margin > 0).nonzero().flatten().tolist()
    print(f"teacher-forced row {b}: {len(bad)} non-argmax tokens, margins {[round(margin[i].item(), 2) for i in bad]}")
