"""Sweep the primitive-library baseline across every layer, L0-L12.

This is the "before ever calling the synthesis model" gate: for each layer,
pick the top-3 deduped SAE features (by firing frequency, dropping near-
duplicates), fit one L2-regularized logistic regression per feature over the
generic primitive set (lexicon + position + POS tag), and report held-out
precision/recall/F1. Whatever's still below a threshold after this is what
should actually go to an LLM.

spaCy parsing is the expensive part and doesn't depend on layer or feature, so
it's cached once per sentence up front rather than recomputed per (layer,
feature) pair -- 100 parses total instead of ~thousands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from primitive_library import PrimitiveBuilder, build_lexicon, dedup_top_features, encode, load_sae

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"  # inputs: activations, index, split, sae weights
OUTPUTS_DIR = THIS_DIR / "outputs"  # findings: this script's fitted results


def main(k_features: int = 3, f1_escalate_threshold: float = 0.6):
    idx = json.loads((DATA_DIR / "index.json").read_text())
    split = json.loads((DATA_DIR / "split.json").read_text())

    lexicon = build_lexicon(idx, split["train"], top_k=40)
    builder = PrimitiveBuilder(lexicon)
    print(f"lexicon size: {len(lexicon)}, primitive dims: {builder.n_dims}")

    # cache primitives + tokens once per sentence (independent of layer/feature)
    print("Caching primitive matrices for all 100 sentences (one-time spaCy pass)...")
    prim_cache, sentence_cache = {}, {}
    for entry in idx["sentences"]:
        npz = np.load(DATA_DIR / "activations" / entry["file"], allow_pickle=True)
        tokens = list(npz["tokens"])
        sentence = str(npz["sentence"])
        prim_cache[entry["id"]] = builder(tokens, sentence)
        sentence_cache[entry["id"]] = npz["hidden_states"]

    X_train = np.concatenate([prim_cache[sid] for sid in split["train"]])
    X_test = np.concatenate([prim_cache[sid] for sid in split["test"]])

    results = []
    escalate = []
    for layer in range(13):
        sae = load_sae(layer)
        feats = dedup_top_features(idx, layer, sae, k=k_features)
        print(f"\n=== L{layer} -- deduped top features: {feats} ===")

        for feat in feats:
            y_train = np.concatenate(
                [encode(sae, sentence_cache[sid][layer])[:, feat] > 0 for sid in split["train"]]
            )
            y_test = np.concatenate(
                [encode(sae, sentence_cache[sid][layer])[:, feat] > 0 for sid in split["test"]]
            )
            if y_train.sum() < 2 or y_train.sum() == len(y_train):
                print(f"  feature {feat}: skipped (degenerate train labels)")
                continue

            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            row = {"layer": layer, "feature": int(feat), "precision": precision, "recall": recall, "f1": f1}
            results.append(row)
            flag = " -> ESCALATE to synthesis model" if f1 < f1_escalate_threshold else ""
            print(f"  feature {feat}: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}{flag}")
            if f1 < f1_escalate_threshold:
                escalate.append(row)

    print("\n=== Summary: mean F1 by layer ===")
    by_layer = {}
    for r in results:
        by_layer.setdefault(r["layer"], []).append(r["f1"])
    for layer in sorted(by_layer):
        f1s = by_layer[layer]
        print(f"  L{layer}: mean_f1={np.mean(f1s):.3f}  ({len(f1s)} features)")

    print(f"\n{len(escalate)}/{len(results)} features fall below F1={f1_escalate_threshold} "
          f"and would be candidates for LLM synthesis:")
    for r in escalate:
        print(f"  L{r['layer']} feature {r['feature']}: f1={r['f1']:.3f}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / "primitive_fit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
