"""Stream-load an HF causal LM (and DFlash draft) onto a small-VRAM GPU with W8A8 Linears.

Weights: int8, symmetric per-output-channel. Activations: int8, dynamic per-token.
GEMM via torch._int_mm (int8 x int8 -> int32), rescaled to bf16.
"""
import glob
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn


class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias, device):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight_t", torch.empty(in_features, out_features, dtype=torch.int8, device=device))
        self.register_buffer("wscale", torch.empty(out_features, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False) if bias else None

    @torch.no_grad()
    def load_weight(self, w):  # w: (out, in) on GPU; chunked so a 150k-row lm_head fits
        q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
        s = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
        for r0 in range(0, w.shape[0], 8192):
            c = w[r0 : r0 + 8192].float()
            s[r0 : r0 + 8192] = c.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            q[r0 : r0 + 8192] = torch.round(c / s[r0 : r0 + 8192, None]).clamp(-127, 127).to(torch.int8)
        # (in, out) view of (out, in) row-major == the column-major B that _int_mm wants
        self.qweight_t = q.t()
        self.wscale = s

    def forward(self, x):
        return _w8a8_forward(self, x)

    def _forward(self, x):
        shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        rows = x2.shape[0]
        xs = x2.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        xq = torch.round(x2.float() / xs).clamp(-127, 127).to(torch.int8)
        padded = max(32, (rows + 7) // 8 * 8)  # _int_mm needs M > 16 and M % 8 == 0
        if padded != rows:
            xq = torch.cat([xq, xq.new_zeros(padded - rows, self.in_features)])
        y = torch._int_mm(xq, self.qweight_t)[:rows]
        y = (y.float() * xs * self.wscale).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape, self.out_features)


def _swap_linears(module, device, skip=()):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name not in skip:
            setattr(module, name, W8A8Linear(child.in_features, child.out_features, child.bias is not None, device))
        else:
            _swap_linears(child, device, skip)


def _stream_load(model, repo_dir, device):
    """Copy every safetensors tensor onto `device`, quantizing W8A8Linear weights on the GPU."""
    modules = dict(model.named_modules())
    params = {n for n, _ in model.named_parameters()}
    buffers = {n for n, _ in model.named_buffers()}
    loaded = set()
    for f in sorted(glob.glob(os.path.join(repo_dir, "*.safetensors"))):
        with safe_open(f, "pt") as sf:
            for key in sf.keys():
                mod_name, _, leaf = key.rpartition(".")
                mod = modules.get(mod_name)
                if mod is None and key not in params:
                    continue
                t = sf.get_tensor(key).to(device)
                if isinstance(mod, W8A8Linear) and leaf == "weight":
                    mod.load_weight(t)
                elif key in params:
                    setattr(modules[mod_name], leaf, nn.Parameter(t.to(torch.bfloat16), requires_grad=False))
                elif key in buffers:
                    modules[mod_name].register_buffer(leaf, t)
                loaded.add(key)
                del t
    return loaded


def _rematerialize_meta(model, device):
    """Non-persistent buffers (e.g. RoPE inv_freq) are not in the checkpoint: rebuild those modules."""
    for name, mod in list(model.named_modules()):
        if any(b.is_meta for b in mod.buffers(recurse=False)) and hasattr(mod, "config"):
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, type(mod)(config=mod.config, device=device))


def _check(model):
    bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if bad:
        raise RuntimeError(f"tensors left on meta: {bad[:10]} (+{max(0, len(bad) - 10)})")


class HostEmbedding(nn.Module):
    """Token embedding kept in host memory; only the looked-up rows move to the GPU.

    The bf16 table is 1.2 GB for Qwen3 (151936 x 4096) but decode reads a few
    dozen rows per step, so keeping it off the GPU frees room for KV cache.
    """

    def __init__(self, embedding: nn.Embedding, device):
        super().__init__()
        self.weight = nn.Parameter(embedding.weight.detach().to("cpu"), requires_grad=False)
        self.device = torch.device(device)

    def forward(self, input_ids):
        return nn.functional.embedding(input_ids.cpu(), self.weight).to(self.device, non_blocking=True)


def load_target_w8a8(model_id, device="cuda", quantize_lm_head=False, embed_on_host=False):
    from transformers import AutoConfig, AutoModelForCausalLM

    repo = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"])
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    _swap_linears(model.model, "meta")
    if quantize_lm_head:
        _swap_linears(model, "meta")
    _stream_load(model, repo, device)
    if config.tie_word_embeddings:
        model.tie_weights()
    _rematerialize_meta(model, device)
    _check(model)
    if embed_on_host and not config.tie_word_embeddings:
        model.model.embed_tokens = HostEmbedding(model.model.embed_tokens, device)
        torch.cuda.empty_cache()
    return model.eval()


def load_draft_w8a8(draft_id, device="cuda"):
    from transformers import AutoConfig

    from dflash.model import DFlash2DraftModel, DFlashDraftModel

    config = AutoConfig.from_pretrained(draft_id)
    config._attn_implementation = "sdpa"
    cls = DFlash2DraftModel if "DFlash2DraftModel" in (config.architectures or []) else DFlashDraftModel
    repo = snapshot_download(draft_id, allow_patterns=["*.json", "*.safetensors"])
    with torch.device("meta"):
        model = cls(config).to(torch.bfloat16)
    _swap_linears(model, "meta")
    _stream_load(model, repo, device)
    _rematerialize_meta(model, device)
    _check(model)
    return model.eval()


_w8a8_forward = W8A8Linear._forward
if os.environ.get("W8A8_COMPILE", "0") == "1":
    _w8a8_forward = torch.compile(W8A8Linear._forward, dynamic=True)
