"""Head-replacement module for Llama-3.2-3B.

Llama-3.2-3B and TinyLlama are both plain `LlamaAttention` / `LlamaForCausalLM`
architecture (same RoPE application, same grouped-query-attention structure --
only the numbers differ: 24 query heads / 8 KV heads / head_dim=128 / 28 layers
here, vs. 32/4/64/22 for TinyLlama). fixed_attention_tinyllama.py's
wrap_llama_attention/install/verify_identity already read all of those
(head_dim, num_key_value_groups, scaling, n_layers via model.model.layers)
dynamically off the actual module/model instance rather than hardcoding
TinyLlama's numbers -- so the identical implementation applies here unchanged.
This file re-exports it rather than duplicating the logic (duplicating would
just risk the two copies silently drifting apart).

The one model-specific wrinkle is dtype: Llama-3.2-3B is loaded in float16
(torch_dtype=torch.float16, device_map="auto") for memory reasons, unlike
TinyLlama/GPT-2 which run in float32. wrap_llama_attention already handles
this correctly without changes -- the softmax is upcast to float32 and cast
back to q.dtype (matching HF's own eager_attention_forward numerics exactly),
and the substituted pattern is cast to attn.dtype before insertion -- so no
float16-specific branch is needed.

See fixed_attention_tinyllama.py for the full explanation of the underlying
GPT-2-style bug this replaces (hooking the parent attention module captures
its output *after* o_proj has already mixed all heads together; this instead
monkey-patches forward() to substitute the real per-head attention-weight
matrix before value-mixing).
"""

from fixed_attention_tinyllama import (  # noqa: F401
    normalize_pattern,
    wrap_llama_attention,
    install,
    verify_identity,
)
