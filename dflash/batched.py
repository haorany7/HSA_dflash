"""Static-batch greedy DFlash (and plain autoregressive) decoding for benchmarks.

Rows accept different numbers of draft tokens per step, so instead of cropping
each row's KV cache, every row appends a full block each step and the slots of
rejected tokens are masked out of later attention ("lockstep + masks"):

* target cache: ``t_valid[b, slot]`` marks the slots attention may read.
* draft cache: context features are padded to ``block_size`` rows per step and
  the padding is masked the same way, so draft (and NPU GEMM) shapes stay fixed.

Greedy only, and EOS is ignored: every row decodes ``max_new_tokens`` tokens,
which is what a throughput benchmark wants. With batch size 1 the output
matches :func:`dflash.model.dflash_generate` up to bf16 batching noise.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from .model import (
    _draft_value,
    _make_cache,
    _output_head,
    _raw_input_embeddings,
    extract_context_feature,
)


def _sync(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def _prefill_mask(valid: torch.Tensor) -> torch.Tensor:
    """Causal mask over left-padded prompts; pad queries see only themselves (avoids NaN rows)."""
    n = valid.shape[1]
    causal = torch.ones(n, n, dtype=torch.bool, device=valid.device).tril()
    eye = torch.eye(n, dtype=torch.bool, device=valid.device)
    return ((causal[None] & valid[:, None, :]) | eye[None])[:, None]


def _block_mask(past_valid: torch.Tensor, new_visible: torch.Tensor) -> torch.Tensor:
    """(B, 1, Q, past + new) mask: every query sees the valid past plus ``new_visible`` (Q, new)."""
    bsz, q = past_valid.shape[0], new_visible.shape[0]
    past = past_valid[:, None, :].expand(bsz, q, -1)
    return torch.cat([past, new_visible[None].expand(bsz, -1, -1)], dim=-1)[:, None]


@torch.inference_mode()
def batched_generate(
    draft,
    target,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    block_size: int = 16,
    pad_token_id: int = 0,
) -> SimpleNamespace:
    """Decode ``max_new_tokens`` greedy tokens for every prompt; ``block_size=1`` is plain AR."""
    # The output head's device, not target.device: HF reports the first
    # parameter's device, which is the host when the embedding lives there.
    head_module = _output_head(target)
    device = next(iter([*head_module.parameters(), *head_module.buffers()])).device
    bsz = len(prompts)
    prompt_len = max(len(p) for p in prompts)
    ids = torch.full((bsz, prompt_len), pad_token_id, dtype=torch.long, device=device)
    t_valid = torch.zeros(bsz, prompt_len, dtype=torch.bool, device=device)
    for b, p in enumerate(prompts):
        ids[b, prompt_len - len(p):] = p.to(device)
        t_valid[b, prompt_len - len(p):] = True
    positions = (t_valid.long().cumsum(1) - 1).clamp(min=0)
    speculative = block_size > 1
    arange = torch.arange(block_size, device=device)

    t0 = _sync(device)
    t_cache = _make_cache(target.config)
    out = target(
        ids,
        position_ids=positions,
        attention_mask=_prefill_mask(t_valid),
        past_key_values=t_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=speculative,
    )
    last_tok = out.logits[:, -1].argmax(-1)
    start = t_valid.sum(1)  # position of last_tok, which the target has not seen yet
    if speculative:
        d_cache = _make_cache(draft.config)
        d_valid = torch.zeros(bsz, 0, dtype=torch.bool, device=device)
        ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)
        ctx_valid, ctx_pos = t_valid, positions
        emb_scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
        head = _output_head(target)
    ttft = _sync(device) - t0

    generated = [[tok] for tok in last_tok.tolist()]
    accept_lengths = []
    causal_block = torch.ones(block_size, block_size, dtype=torch.bool, device=device).tril()
    t1 = _sync(device)
    while min(len(g) for g in generated) < max_new_tokens:
        block_pos = start[:, None] + arange
        if speculative:
            noise_ids = torch.full((bsz, block_size), draft.mask_token_id, dtype=torch.long, device=device)
            noise_ids[:, 0] = last_tok
            noise = _raw_input_embeddings(target, noise_ids, emb_scale)
            past_len = d_cache.get_seq_length()
            # Noise attends to all valid context (past + this step's) and the whole block.
            d_mask = torch.cat([
                d_valid[:, None, :].expand(bsz, block_size, -1),
                ctx_valid[:, None, :].expand(bsz, block_size, -1),
                torch.ones(bsz, block_size, block_size, dtype=torch.bool, device=device),
            ], dim=-1)[:, None]
            hidden = draft(
                target_hidden=ctx,
                noise_embedding=noise,
                position_ids=torch.cat([ctx_pos, block_pos], dim=1),
                attention_mask=d_mask,
                past_key_values=d_cache,
                use_cache=True,
            )[:, 1 - block_size:, :]
            d_cache.crop(past_len + ctx.shape[1])  # keep context, drop noise
            d_valid = torch.cat([d_valid, ctx_valid], dim=1)
            proposal = draft.compute_logits(hidden, head).argmax(-1)
            block_ids = torch.cat([last_tok[:, None], proposal], dim=1)
        else:
            block_ids = last_tok[:, None]

        out = target(
            block_ids,
            position_ids=block_pos,
            attention_mask=_block_mask(t_valid, causal_block),
            past_key_values=t_cache,
            use_cache=True,
            output_hidden_states=speculative,
        )
        posterior = out.logits.argmax(-1)
        if speculative:
            accepted = (block_ids[:, 1:] == posterior[:, :-1]).cumprod(1).sum(1)
        else:
            accepted = torch.zeros(bsz, dtype=torch.long, device=device)
        fed = arange[None] <= accepted[:, None]  # block tokens now committed to the cache
        t_valid = torch.cat([t_valid, fed], dim=1)
        last_tok = posterior.gather(1, accepted[:, None])[:, 0]
        new_tokens = torch.cat([block_ids[:, 1:], posterior], dim=1)
        if speculative:
            ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            ctx_valid, ctx_pos = fed, block_pos

        acc = accepted.tolist()
        rows = new_tokens.tolist()
        for b in range(bsz):
            # accepted drafts, then the bonus token (posterior at the first mismatch)
            generated[b].extend(rows[b][: acc[b]] + [rows[b][block_size - 1 + acc[b]]])
        accept_lengths.append([a + 1 for a in acc])
        start = start + accepted + 1
    decode = _sync(device) - t1

    return SimpleNamespace(
        tokens=[g[:max_new_tokens] for g in generated],
        ttft=ttft,
        decode_time=decode,
        steps=len(accept_lengths),
        accept_lengths=accept_lengths,
    )
