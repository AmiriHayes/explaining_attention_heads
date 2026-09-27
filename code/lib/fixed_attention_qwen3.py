"""Head-replacement module for Qwen3 (Qwen3ForCausalLM / Qwen3Model).

Qwen3 is Llama-shaped with ONE addition that matters here: it applies per-head
RMSNorm to the queries and keys before RoPE. From
transformers/models/qwen3/modeling_qwen3.py, Qwen3Attention.forward:

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

fixed_attention_tinyllama.py omits those norms (its lines 69-70 project and
transpose directly), so it CANNOT be re-exported for Qwen3 the way
fixed_attention_llama3b.py is -- an unnormed wrapper is not the identity and
fails verify_identity outright. That two-line difference is the whole module.

Everything else is shared with the Llama path and imported rather than copied:
normalize_pattern, and the causal / fp32-softmax / GQA handling. apply_rotary_pos_emb
and repeat_kv are textually identical between the llama and qwen3 modules in
transformers 5.16.x, but they are imported from qwen3 here so this file tracks
Qwen3 if they ever diverge.

Note on GQA: output attention is one matrix per QUERY head (32 for Qwen3-8B),
not per KV head (8) -- repeat_kv expands K/V before the score matmul, so
`attn[:, hi]` indexes query heads and a program-per-query-head substitution is
correct.
"""

import numpy as np
import torch

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from fixed_attention_tinyllama import normalize_pattern  # identical semantics


def wrap_qwen3_attention(module, layer_idx: int, state: dict):
    """Replace module.forward (a Qwen3Attention instance) with a substituting
    eager implementation.

    state: {"assignment": {(layer, head): prog_name}, "pattern": callable
    (prog_name, sent) -> np.ndarray | None, "sentence": [str],
    "zero_heads": optional iterable of (layer, head) to zero-ablate}.
    """
    orig_forward  = module.forward
    num_kv_groups = module.num_key_value_groups
    scaling       = module.scaling
    head_dim      = module.head_dim

    def forward(hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None,
                **kwargs):
        input_shape  = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)
        s = hidden_states.shape[1]

        # THE QWEN3 DELTA: q_norm / k_norm applied per head, before RoPE.
        # v is NOT normed, matching the reference implementation.
        q = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
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

        for (li, hi) in state.get("zero_heads", ()):
            if li == layer_idx:
                attn[:, hi] = 0.0

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
        originals.append((layer.self_attn, wrap_qwen3_attention(layer.self_attn, li, state)))
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
