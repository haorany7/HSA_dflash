"""DFlash Qwen3-8B smoke test: DFlash vs. autoregressive (block_size=1), greedy."""
import argparse
import time

import torch

from dflash.benchmark import apply_chat_template, load_transformers_models, stop_token_ids
from dflash.model import dflash_generate

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
p.add_argument("--device", default="cpu")
p.add_argument("--draft-device", default="same")
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--skip-ar", action="store_true")
p.add_argument("--prompt", default="How many positive whole-number divisors does 196 have? Explain briefly.")
args = p.parse_args()

torch.manual_seed(0)
t = time.time()
target, draft, tok = load_transformers_models(args.model, args.draft, torch.device(args.device), args.draft_device)
print(f"loaded in {time.time() - t:.1f}s; draft target_layer_ids={draft.target_layer_ids} block={draft.block_size}")

prompt = apply_chat_template(tok, [{"role": "user", "content": args.prompt}], "off")
ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
stops = stop_token_ids(target, tok)

runs = [("dflash", None)] + ([] if args.skip_ar else [("autoregressive", 1)])
outs = {}
with torch.inference_mode():
    for name, bs in runs:
        r = dflash_generate(draft, target, ids, args.max_new_tokens, stops, block_size=bs, return_stats=True)
        outs[name] = r.output_ids[0, ids.shape[1]:]
        acc = sum(r.acceptance_lengths) / len(r.acceptance_lengths)
        print(f"[{name}] prompt={r.num_input_tokens} out={r.num_output_tokens} TTFT={r.time_to_first_token:.2f}s "
              f"decode={1 / r.time_per_output_token:.2f} tok/s  mean_accept_len={acc:.2f}")
print("---\n" + tok.decode(outs["dflash"], skip_special_tokens=True))
if "autoregressive" in outs:
    same = torch.equal(outs["dflash"], outs["autoregressive"])
    print(f"--- greedy outputs identical to AR: {same}")
    if not same:
        a, b = outs["dflash"], outs["autoregressive"]
        n = min(len(a), len(b)); i = next((j for j in range(n) if a[j] != b[j]), n)
        print(f"first divergence at output token {i}: dflash={tok.decode(a[i:i+8])!r} ar={tok.decode(b[i:i+8])!r}")
        print("AR context:", repr(tok.decode(b[max(0, i - 20):i])))
