"""Correct head-replacement semantics for GPT-2: attention-weight substitution.

FIX for the head-replacement hooks in this repo (plot_random_baseline_figures.py
and all_experiments.ipynb "Figure 4.3" cell): those hooks run on the attention
module's forward-hook OUTPUT, which in HuggingFace GPT-2 is post-c_proj (head
channels already mixed), and they left-multiply the program matrix into that
output — composing the pattern on top of the attention the model already
applied instead of replacing it. This module substitutes the program's
(causally-masked, row-normalized) pattern for the head's softmaxed attention
probabilities BEFORE value mixing, upstream of c_proj.

NOTE: GPT-2 only. BERT needs the same treatment WITHOUT the causal mask
(bidirectional); TinyLlama/Llama-3B need it with q_proj/k_proj/v_proj + GQA
key-value head repetition instead of c_attn. Same ~30-line pattern per
architecture.

Replaces a head's softmaxed attention probabilities with the program's
(causally-masked, row-normalized) pattern BEFORE value mixing — upstream of
c_proj — instead of the repo's post-projection output composition.

The wrapper recomputes the eager attention path from the module's own
weights. Identity contract: with an empty assignment it must reproduce the
module's original output (asserted by verify_identity at setup).
"""

import math

import numpy as np
import torch


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


def wrap_gpt2_attention(module, layer_idx: int, state: dict):
    """Replace module.forward with a substituting eager implementation.

    state: {"assignment": {(layer, head): prog_name}, "pattern": callable
    (prog_name, sent) -> np.ndarray | None, "sentence": [str]}.
    """
    orig_forward = module.forward
    n_head = module.num_heads
    head_dim = module.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    def forward(hidden_states, *args, **kwargs):
        b, s, _ = hidden_states.shape
        qkv = module.c_attn(hidden_states)
        q, k, v = qkv.split(module.split_size, dim=2)

        def heads(x):
            return x.view(b, s, n_head, head_dim).permute(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        causal = torch.tril(torch.ones(s, s, dtype=torch.bool,
                                       device=attn.device))
        attn = attn.masked_fill(~causal, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)

        for (li, hi), prog_name in state["assignment"].items():
            if li != layer_idx:
                continue
            mat = state["pattern"](prog_name, state["sentence"][0])
            if mat is None or mat.shape[0] != s:
                continue  # uncovered: keep the head's own attention (repo convention)
            attn[:, hi] = normalize_pattern(mat).to(attn.dtype).to(attn.device)

        ctx = torch.matmul(attn, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(b, s, n_head * head_dim)
        out = module.c_proj(ctx)
        out = module.resid_dropout(out)
        return (out, None)

    module.forward = forward
    return orig_forward


def install(model, state: dict):
    """Wrap every layer's attention; returns restore()."""
    originals = []
    for li, block in enumerate(model.transformer.h):
        originals.append((block.attn, wrap_gpt2_attention(block.attn, li, state)))
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
