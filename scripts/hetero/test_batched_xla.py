"""Check dflash.batched_xla against dflash.batched on tiny random models (CPU, fp32).

The reference (``batched_generate``) always runs on CPU; ``batched_generate_xla``
runs on ``device``. A random draft is almost never accepted, so the draft's real forward (and its
cache) still runs but its proposals are replaced by an oracle: the target's AR
continuation, with ~10% of positions (irregularly) deterministically corrupted. Greedy
speculative decoding is lossless, so both implementations must reproduce the
AR tokens exactly while accepting 1..block tokens per step.

Usage: PYTHONPATH=. python scripts/hetero/test_batched_xla.py [device]
"""
import sys

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dflash.batched import batched_generate
from dflash.batched_xla import batched_generate_xla
from dflash.model import DFlashDraftModel

torch.manual_seed(0)
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
V = 512
tcfg = Qwen3Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
target = Qwen3ForCausalLM(tcfg).eval()
dcfg = Qwen3Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
dcfg.num_target_layers = 6
dcfg.dflash_config = {"target_layer_ids": [1, 3], "mask_token_id": V - 1, "block_size": 8}
draft = DFlashDraftModel(dcfg).eval()

prompts = [torch.randint(0, V - 1, (n,)) for n in (5, 17, 9, 30)]
N = 40
ar = batched_generate(draft, target, prompts, N, 1, pad_token_id=0).tokens
seen = {}


def capture(module, args, kwargs):
    seen["pos"] = kwargs["position_ids"].cpu()


def oracle_logits(hidden, head, rows):
    pos = seen["pos"][:, -hidden.shape[1]:]  # positions of the proposed tokens
    toks = torch.zeros(pos.shape, dtype=torch.long)
    for i, b in enumerate(rows):
        for j, q in enumerate(pos[i].tolist()):
            k = q - len(prompts[b])
            good = 0 <= k < N and (b * 7 + q * q * 3) % 11 != 0
            toks[i, j] = ar[b][k] if good else (b + q) % V
    return torch.nn.functional.one_hot(toks, V).float().to(hidden.device)


draft.register_forward_pre_hook(capture, with_kwargs=True)
ok = True
refs = {}
for block in (1, 4, 8):
    draft.compute_logits = lambda h, head: oracle_logits(h, head, range(len(prompts)))
    refs[block] = batched_generate(draft, target, prompts, N, block, pad_token_id=0)
target.to(dev); draft.to(dev)
for block in (1, 4, 8):
    ref = refs[block]
    draft.compute_logits = lambda h, head: oracle_logits(h, head, range(len(prompts)))
    new = batched_generate_xla(draft, target, prompts, N, block, pad_token_id=0, prompt_bucket=16)
    for b in range(len(prompts)):
        same = ref.tokens[b] == new.tokens[b] == ar[b]
        ok &= same
        n = next((i for i, (x, y) in enumerate(zip(ref.tokens[b], new.tokens[b])) if x != y), N)
        print(f"block={block} row={b}: agree {n}/{N}{'' if same else '  MISMATCH'}")
    print(f"  steps ref={ref.steps} new={new.steps}  "
          f"max accept ref={max(map(max, ref.accept_lengths))} new={max(max(a for a in s if a) for s in new.accept_lengths)}")
    # single-row consistency
    draft.compute_logits = lambda h, head: oracle_logits(h, head, [1])
    one = batched_generate_xla(draft, target, prompts[1:2], N, block, prompt_bucket=16)
    ok &= one.tokens[0] == new.tokens[1]
print("PASS" if ok else "FAIL")
