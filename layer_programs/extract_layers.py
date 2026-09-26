"""Extract per-layer residual-stream activations for a TinyStories sample.

For each sentence, this runs GPT-2-small once with `output_hidden_states=True`
and saves every layer's [seq_len, 768] residual-stream state to disk. This is
the raw data for the layer-level program-synthesis experiments (sentence-level,
axis-level/SAE-feature, and token-level studies). It deliberately does NOT
truncate sentences to 20 tokens or touch SAEs yet -- full sentences are kept
so this data can be reused as-is for the sentence-level and axis-level
studies; a 20-token slice for the token-level study should be built by
subsampling positions from this same full-length data (not by truncating
every sentence), since GPT-2's causal masking means truncating the tail is
lossless but truncating every example to a fixed length biases the sample
toward sentence-openings.

Usage:
    python extract_layers.py --n_sentences 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
ACTIVATIONS_DIR = DATA_DIR / "activations"
SENTENCE_CACHE = DATA_DIR / "tinystories_sentences.json"
INDEX_FILE = DATA_DIR / "index.json"

MODEL_NAME = "gpt2"
SEED = 42
MIN_WORDS, MAX_WORDS = 15, 60  # same filter as code/all_experiments_v2.ipynb's load_tinystories
MAX_TOKENS = 128  # safety cap only, not the row-level 20-token subsample -- headroom
# above the true max (76 tokens for a 60-word-filtered TinyStories sample; BPE
# inflates word count by ~1.3x, so a naive cap near 60-64 truncates most examples)


def load_tinystories_sentences(n: int, seed: int = SEED) -> list[str]:
    """Load n TinyStories sentences.

    Mirrors the filtering/caching convention already used by
    `load_tinystories` in code/all_experiments_v2.ipynb, so the same pool
    logic (15-60 words, seed-42 shuffle) is reused rather than reinvented.
    """
    if SENTENCE_CACHE.exists():
        cached = json.loads(SENTENCE_CACHE.read_text())
        if len(cached) >= n:
            return cached[:n]

    from datasets import load_dataset

    print(f"Streaming TinyStories (need {n})...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    pool: list[str] = []
    for item in ds:
        text = item["text"].strip()
        nw = len(text.split())
        if MIN_WORDS <= nw <= MAX_WORDS:
            pool.append(text)
        if len(pool) >= n * 3:
            break

    rng = random.Random(seed)
    rng.shuffle(pool)
    sentences = pool[:n]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SENTENCE_CACHE.write_text(json.dumps(sentences, indent=2))
    return sentences


def extract_hidden_states(model, tokenizer, sentence: str, device: str) -> dict:
    """Run one sentence through GPT-2 and return every layer's residual-stream
    state.

    `hidden_states[L]` is the residual stream *entering* layer L, for
    L = 0..11 (L=0 is the embedding output), and `hidden_states[12]` is the
    stream *leaving* the final layer. Because the residual stream is a
    running sum, `hidden_states[L+1]` is mathematically identical to "the
    output of layer L" -- so `hidden_states[L]` and `hidden_states[L+1]` are
    the exact pre/post pair for layer L with no extra computation, and they
    line up with Neuronpedia's `{0..12}-res-jb` SAE checkpoints for
    gpt2-small.

    IMPORTANT: `output_hidden_states=True`'s last tuple entry is NOT the raw
    residual stream after the final block -- HuggingFace applies the final
    LayerNorm (`ln_f`) before appending it (verified directly with forward
    hooks: true raw post-block-11 output differs from
    `out.hidden_states[-1]` by up to ~365 on one test sentence, while
    `ln_f(raw post-block-11 output)` matches it exactly). LayerNorm divides
    every token's vector by a per-token std inflated by the residual
    stream's outlier dims, so trusting that entry silently flattens exactly
    the "massive activation" structure we care about. Entries 0..11 were
    checked the same way and are unaffected (they're captured before their
    block runs, so ln_f never touches them). Fix: grab the true raw value
    with a forward hook on the last block instead of trusting
    `output_hidden_states` for that one entry.
    """
    enc = tokenizer(sentence, return_tensors="pt")
    input_ids = enc["input_ids"][:, :MAX_TOKENS].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    captured = {}
    def grab_last_block_raw_output(module, inp, out):
        captured["raw"] = out[0] if isinstance(out, tuple) else out
    handle = model.transformer.h[-1].register_forward_hook(grab_last_block_raw_output)
    try:
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
    finally:
        handle.remove()

    # out.hidden_states: tuple of 13 tensors, each (1, seq_len, 768). Entries
    # 0..11 are correct as-is; entry 12 is post-ln_f, so swap in the raw
    # hook-captured value instead.
    hidden_states = list(out.hidden_states)
    hidden_states[-1] = captured["raw"]
    layers = torch.stack(hidden_states, dim=0)  # (13, 1, seq_len, 768)
    layers = layers.squeeze(1).cpu().numpy().astype(np.float32)  # (13, seq_len, 768)

    return {"sentence": sentence, "tokens": tokens, "hidden_states": layers}


def main(n_sentences: int = 100) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device).eval()

    sentences = load_tinystories_sentences(n_sentences)
    print(f"Loaded {len(sentences)} TinyStories sentences.")

    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    index = {
        "model": MODEL_NAME,
        "n_layers": model.config.n_layer + 1,  # +1: hidden_states includes the embedding layer
        "hidden_size": model.config.n_embd,
        "max_tokens_cap": MAX_TOKENS,
        "sentences": [],
    }

    for i, sentence in enumerate(sentences):
        out_path = ACTIVATIONS_DIR / f"sentence_{i:04d}.npz"
        if not out_path.exists():
            data = extract_hidden_states(model, tokenizer, sentence, device)
            np.savez_compressed(
                out_path,
                sentence=data["sentence"],
                tokens=np.array(data["tokens"]),
                hidden_states=data["hidden_states"],
            )
            n_tokens = len(data["tokens"])
        else:
            n_tokens = int(np.load(out_path, allow_pickle=True)["tokens"].shape[0])

        index["sentences"].append(
            {"id": i, "file": out_path.name, "n_tokens": n_tokens, "sentence": sentence}
        )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sentences)} done")

    INDEX_FILE.write_text(json.dumps(index, indent=2))
    print(f"Saved {len(sentences)} sentences' activations to {ACTIVATIONS_DIR}")
    print(f"Index written to {INDEX_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sentences", type=int, default=100)
    args = parser.parse_args()
    main(args.n_sentences)
