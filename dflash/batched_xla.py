"""Static-shape batched greedy DFlash / AR decoding for XLA devices (TPU).

Same algorithm and outputs as :func:`dflash.batched.batched_generate`, but every
device tensor has a fixed shape so XLA compiles each graph once:

* KV caches are preallocated (``SlotCache``) and each row writes token ``p`` at
  slot ``p`` (scatter, per-row). A rejected token's slot is overwritten by the
  next step's block, so attention is a plain per-row causal mask over slots and
  the cache needs ``prompt + max_new_tokens + 2 * block`` slots instead of the
  lockstep ``prompt + max_new_tokens * block``.
* Left-padding tokens, and draft context rows of rejected tokens, are written
  to a trash slot (the last one) that no query ever sees.
* Bookkeeping (positions, accepted counts, finished rows) lives on the host;
  each step uploads a few small index tensors and downloads ``B * 2 * block``
  token ids, which is also the step's single graph boundary.
* Rows that already have ``max_new_tokens`` keep recomputing their last block
  in place (as in lockstep, finished rows still cost compute), so positions
  stay bounded.

Works on CPU/CUDA too (no XLA calls are made there), which is how it is tested
against ``batched_generate``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from .model import _draft_value, _output_head, _raw_input_embeddings, extract_context_feature


def _is_xla(device: torch.device) -> bool:
    return device.type == "xla"


def _sync(device: torch.device) -> float:
    if _is_xla(device):
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        xm.wait_device_ops()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def _mark_step(device: torch.device) -> None:
    if _is_xla(device):
        import torch_xla.core.xla_model as xm

        xm.mark_step()


def _bucket(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


class SlotCache:
    """Preallocated KV cache; ``update`` scatters row ``b``'s token ``i`` to ``slots[b, i]``.

    Duck-types the one method HF Qwen3 and the DFlash draft attention call
    (``update(k, v, layer_idx, ...)``). Set ``slots`` (B, L) before each forward.
    """

    def __init__(self, config, bsz: int, max_len: int, dtype, device):
        heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        shape = (bsz, heads, max_len, head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.slots = None

    def update(self, key, value, layer_idx, *args, **kwargs):
        idx = self.slots[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
        self.keys[layer_idx].scatter_(2, idx, key.to(self.keys[layer_idx].dtype))
        self.values[layer_idx].scatter_(2, idx, value.to(self.values[layer_idx].dtype))
        return self.keys[layer_idx], self.values[layer_idx]


def _causal_mask(query_pos: torch.Tensor, max_len: int, extra: int = 0) -> torch.Tensor:
    """(B, 1, Q, max_len) bool: query at position p sees slots <= p + extra (the trash slot never)."""
    slots = torch.arange(max_len, device=query_pos.device)
    return (slots[None, None, :] <= (query_pos[:, :, None] + extra))[:, None]


@torch.no_grad()
def batched_generate_xla(
    draft,
    target,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    block_size: int = 16,
    pad_token_id: int = 0,
    prompt_bucket: int = 64,
) -> SimpleNamespace:
    """Decode ``max_new_tokens`` greedy tokens for every prompt; ``block_size=1`` is plain AR."""
    head = _output_head(target)
    device = next(iter([*head.parameters(), *head.buffers()])).device
    dtype = next(head.parameters()).dtype
    bsz = len(prompts)
    speculative = block_size > 1
    prompt_len = _bucket(max(len(p) for p in prompts), prompt_bucket)
    max_len = prompt_len + max_new_tokens + 2 * block_size + 1
    trash = max_len - 1

    ids = torch.full((bsz, prompt_len), pad_token_id, dtype=torch.long)
    valid = torch.zeros(bsz, prompt_len, dtype=torch.bool)
    for b, p in enumerate(prompts):
        ids[b, prompt_len - len(p):] = p
        valid[b, prompt_len - len(p):] = True
    positions = (valid.cumsum(1) - 1).clamp(min=0)
    prompt_slots = torch.where(valid, positions, trash)
    start = valid.sum(1)  # host: position of each row's last_tok, which the target has not seen yet
    arange = torch.arange(block_size)

    t0 = _sync(device)
    t_cache = SlotCache(target.config, bsz, max_len, dtype, device)
    t_cache.slots = prompt_slots.to(device)
    pos_dev = positions.to(device)
    out = target(
        ids.to(device),
        position_ids=pos_dev,
        attention_mask=_causal_mask(pos_dev, max_len),
        past_key_values=t_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=speculative,
    )
    last_tok = out.logits[:, -1].argmax(-1)
    if speculative:
        d_cache = SlotCache(draft.config, bsz, max_len, dtype, device)
        ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)
        ctx_slots, ctx_pos = prompt_slots, positions
        emb_scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
        mask_ids = torch.full((bsz, block_size - 1), draft.mask_token_id, dtype=torch.long, device=device)
    del out  # drop every layer's hidden states before the graph is cut
    first = last_tok.cpu()  # graph boundary
    ttft = _sync(device) - t0

    generated = [[tok] for tok in first.tolist()]
    accept_lengths = []
    t1 = _sync(device)
    while min(len(g) for g in generated) < max_new_tokens:
        block_pos = start[:, None] + arange
        block_pos_dev = block_pos.to(device)
        if speculative:
            noise = _raw_input_embeddings(target, torch.cat([last_tok[:, None], mask_ids], dim=1), emb_scale)
            d_cache.slots = torch.cat([ctx_slots, block_pos], dim=1).to(device)
            # Noise sees all committed context (slots < start) and its whole block.
            hidden = draft(
                target_hidden=ctx,
                noise_embedding=noise,
                position_ids=torch.cat([ctx_pos, block_pos], dim=1).to(device),
                attention_mask=_causal_mask(block_pos_dev[:, :1].expand(-1, block_size), max_len, block_size - 1),
                past_key_values=d_cache,
                use_cache=True,
            )[:, 1 - block_size:, :]
            proposal = draft.compute_logits(hidden, head).argmax(-1)
            block_ids = torch.cat([last_tok[:, None], proposal], dim=1)
        else:
            block_ids = last_tok[:, None]

        t_cache.slots = block_pos_dev
        out = target(
            block_ids,
            position_ids=block_pos_dev,
            attention_mask=_causal_mask(block_pos_dev, max_len),
            past_key_values=t_cache,
            use_cache=True,
            output_hidden_states=speculative,
        )
        posterior = out.logits.argmax(-1)
        if speculative:
            ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)
        del out
        # One transfer per step: block tokens and the target's argmax.
        both = torch.cat([block_ids, posterior], dim=1).cpu()
        block_host, post_host = both[:, :block_size], both[:, block_size:]
        if speculative:
            accepted = (block_host[:, 1:] == post_host[:, :-1]).long().cumprod(1).sum(1)
        else:
            accepted = torch.zeros(bsz, dtype=torch.long)
        done = torch.tensor([len(g) >= max_new_tokens for g in generated])
        new_last = post_host.gather(1, accepted[:, None])[:, 0]
        last_tok = torch.where(done, block_host[:, 0], new_last).to(device)
        if speculative:
            fed = arange[None] <= accepted[:, None]
            ctx_slots, ctx_pos = torch.where(fed, block_pos, trash), block_pos

        acc = accepted.tolist()
        new_tokens = torch.cat([block_host[:, 1:], post_host], dim=1).tolist()
        for b in range(bsz):
            if not done[b]:
                # accepted drafts, then the bonus token (posterior at the first mismatch)
                generated[b].extend(new_tokens[b][: acc[b]] + [new_tokens[b][block_size - 1 + acc[b]]])
        accept_lengths.append([a + 1 for a in acc])
        start = torch.where(done, start, start + accepted + 1)
    decode = _sync(device) - t1

    return SimpleNamespace(
        tokens=[g[:max_new_tokens] for g in generated],
        ttft=ttft,
        decode_time=decode,
        steps=len(accept_lengths),
        accept_lengths=accept_lengths,
    )
