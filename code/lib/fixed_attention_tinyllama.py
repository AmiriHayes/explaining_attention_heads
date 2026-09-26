"""Head-replacement module for TinyLlama (Llama-architecture): substituting
attention-weights, mirroring fixed_attention_gpt2.py's approach.

Same fix pattern as GPT-2: monkey-patch each layer's self_attn.forward to
recompute the real attention-weight matrix (post-softmax, pre-value-mixing)
and substitute the program's pattern directly into it for assigned heads,
instead of hooking the whole module's output (which is post-o_proj and
mixes all heads together -- see fixed_attention_gpt2.py's docstring / the
session notes on the original GPT-2 bug for the full explanation).

Two things GPT-2 didn't need that this does:
  1. Rotary position embeddings (RoPE) applied to Q/K before the dot product.
     We don't recompute cos/sin ourselves -- LlamaDecoderLayer already
     computes them once per forward pass and passes them into self_attn via
     the `position_embeddings` kwarg, so our patched forward just accepts
     and reuses the same (cos, sin) the real model would have used.
  2. Grouped-query attention: TinyLlama has 32 query heads but only 4 KV
     heads (module.num_key_value_groups == 8). We reuse HF's own `repeat_kv`
     utility (imported straight from transformers, not reimplemented) so the
     GQA expansion is guaranteed to match what the real model does, and the
     substituted head index `hi` in [0, 32) unambiguously refers to a single
     query head's attention row -- exactly the granularity the programs are
     defined at.

verify_identity() confirms: with an empty assignment, this patched forward
reproduces the original model's logits (mechanism correctness check, same
idea as the GPT-2 module).
"""

import numpy as np
import torch

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def normalize_pattern(mat: np.ndarray) -> torch.Tensor:
    """Causal-mask, clip negatives, row-normalize; degenerate rows -> one-hot
    on the first position (the only always-causal fallback)."""
    t = torch.tensor(np.asarray(mat), dtype=torch.float32)
    n = t.shape[0]
    t = t.clamp(min=0.0) * torch.tril(torch.ones(n, n))
    rs = t.sum(dim=-1, keepdim=True)
    fallback = torch.zeros(n, n)
    fallback[:, 0] = 1.0
    t = torch.where(rs > 1e-9, t / rs.clamp(min=1e-9), fallback)
    return t


def wrap_llama_attention(module, layer_idx: int, state: dict):
    """Replace module.forward (a LlamaAttention instance) with a substituting
    eager implementation.

    state: {"assignment": {(layer, head): prog_name}, "pattern": callable
    (prog_name, sent) -> np.ndarray | None, "sentence": [str]}.
    """
    orig_forward   = module.forward
    num_kv_groups  = module.num_key_value_groups
    scaling        = module.scaling
    head_dim       = module.head_dim

    def forward(hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None,
                **kwargs):
        input_shape  = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)
        s = hidden_states.shape[1]

        q = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        key_states   = repeat_kv(k, num_kv_groups)
        value_states = repeat_kv(v, num_kv_groups)

        attn = torch.matmul(q, key_states.transpose(2, 3)) * scaling
        causal = torch.tril(torch.ones(s, s, dtype=torch.bool, device=attn.device))
        attn = attn.masked_fill(~causal, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        for (li, hi), prog_name in state["assignment"].items():
            if li != layer_idx:
                continue
            mat = state["pattern"](prog_name, state["sentence"][0])
            if mat is None or mat.shape[0] != s:
                continue  # uncovered: keep the head's own attention (repo convention)
            attn[:, hi] = normalize_pattern(mat).to(attn.dtype).to(attn.device)

        ctx = torch.matmul(attn, value_states)
        ctx = ctx.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        out = module.o_proj(ctx)
        return out, attn

    module.forward = forward
    return orig_forward


def install(model, state: dict):
    """Wrap every layer's self_attn; returns restore()."""
    originals = []
    for li, layer in enumerate(model.model.layers):
        originals.append((layer.self_attn, wrap_llama_attention(layer.self_attn, li, state)))
    def restore():
        for mod, orig in originals:
            mod.forward = orig
    return restore


def verify_identity(model, tok, sentence: str, device) -> float:
    """Max |logit diff| between wrapped-empty and original forward."""
    toks = tok(sentence, return_tensors="pt").to(device)
    state = {"assignment": {}, "pattern": lambda *_: None, "sentence": [sentence]}
    with torch.no_grad():
        ref = model(**toks).logits.clone()
    restore = install(model, state)
    with torch.no_grad():
        got = model(**toks).logits
    restore()
    return float((ref - got).abs().max())
