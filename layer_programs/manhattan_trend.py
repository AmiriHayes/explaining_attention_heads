"""Three-way normalized-Manhattan comparison across all 13 layers:
null (global constant) vs. token-level ceiling (exact-identity lookup, no
context) vs. sentence-level (primitives -> ridge, full 768-dim), now with the
lexicon widened to the full 571-word train vocabulary instead of top-40.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from primitive_library import PrimitiveBuilder, build_lexicon

THIS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = THIS_DIR / "outputs"  # findings: this script's results

idx = json.load(open("data/index.json"))
split = json.load(open("data/split.json"))

FULL_LEXICON = build_lexicon(idx, split["train"], top_k=100000)  # all 571 unique train tokens
builder = PrimitiveBuilder(FULL_LEXICON)
print(f"lexicon size: {len(FULL_LEXICON)}, primitive dims: {builder.n_dims}")


def load_layer(sid, layer):
    entry = idx["sentences"][sid]
    npz = np.load(f"data/activations/{entry['file']}", allow_pickle=True)
    return npz["hidden_states"][layer], list(npz["tokens"]), str(npz["sentence"])


def evaluate(layer):
    train_tokens, train_X, train_Y = [], [], []
    for sid in split["train"]:
        h, toks, sent = load_layer(sid, layer)
        train_tokens.extend(toks)
        train_X.append(builder(toks, sent))
        train_Y.append(h)
    train_X, train_Y = np.concatenate(train_X), np.concatenate(train_Y)

    test_tokens, test_X, test_Y = [], [], []
    for sid in split["test"]:
        h, toks, sent = load_layer(sid, layer)
        test_tokens.extend(toks)
        test_X.append(builder(toks, sent))
        test_Y.append(h)
    test_X, test_Y = np.concatenate(test_X), np.concatenate(test_Y)

    dim_std = train_Y.std(axis=0)

    def normalized_manhattan(pred):
        return (np.abs(test_Y - pred) / dim_std).mean()

    null_pred = np.tile(train_Y.mean(axis=0).astype(float), (len(test_tokens), 1))
    null_score = normalized_manhattan(null_pred)

    buckets = defaultdict(list)
    for tok, row in zip(train_tokens, train_Y):
        buckets[tok].append(row)
    token_means = {tok: np.mean(rows, axis=0) for tok, rows in buckets.items()}
    global_mean = train_Y.mean(axis=0)
    token_pred = np.array([token_means.get(t, global_mean) for t in test_tokens])
    token_score = normalized_manhattan(token_pred)

    reg = Ridge(alpha=1.0).fit(train_X, train_Y)
    sentence_pred = reg.predict(test_X)
    sentence_score = normalized_manhattan(sentence_pred)

    return null_score, token_score, sentence_score


print(f"\n{'layer':>6} {'null':>8} {'token_ceiling':>14} {'sentence_wide':>14}")
results = []
for layer in range(13):
    null_s, token_s, sent_s = evaluate(layer)
    results.append((layer, null_s, token_s, sent_s))
    print(f"{layer:>6} {null_s:>8.4f} {token_s:>14.4f} {sent_s:>14.4f}")

OUTPUTS_DIR.mkdir(exist_ok=True)
json.dump(
    [{"layer": l, "null": float(n), "token_ceiling": float(t), "sentence_wide": float(s)} for l, n, t, s in results],
    open(OUTPUTS_DIR / "manhattan_trend_results.json", "w"), indent=2,
)
