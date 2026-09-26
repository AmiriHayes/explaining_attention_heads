"""Primitive-library baseline for SAE-feature prediction, all layers.

Idea (from the original repo's linear-interpolation refinement pipeline,
interpretability.ipynb cell 79): before ever calling a synthesis model, fit a
regularized linear combination of a small set of simple, generic, hand-written
primitives per token. Only features that still fit poorly after this should be
escalated to LLM synthesis.

Primitives (kept deliberately simple -- this is a baseline, not a final program):
  - lexical:  one-hot over the ~40 most frequent train tokens + an OTHER bucket
  - position: fractional position in the sentence, is_first, is_last
  - POS tag:  coarse spaCy POS tag, one-hot (alignment mirrors the original
              codebase's _align_to_spacy: character-span overlap between BPE
              tokens and spaCy tokens)

Each (layer, feature) gets one L2-regularized logistic regression over these
primitives -- a linear combination, not a bespoke program, deliberately.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"

POS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "PUNCT", "PROPN",
    "NUM", "CCONJ", "SCONJ", "AUX", "PART", "INTJ", "SYM", "X", "SPACE",
]

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp


def align_to_spacy(tokens: list[str], doc) -> list[list[int]]:
    """Character-span overlap alignment between BPE tokens and spaCy tokens
    (same method as the original codebase's data/gpt2_programs.py)."""
    spans, pos = [], 0
    for t in tokens:
        clean = t.lstrip("Ġ").lstrip("Ċ")  # Ġ, Ċ
        span_len = max(len(clean), 1)
        spans.append((pos, pos + span_len))
        pos += span_len
    alignment = []
    for gs, ge in spans:
        overlapping = [si for si, st in enumerate(doc) if st.idx < ge and st.idx + len(st.text) > gs]
        alignment.append(overlapping)
    return alignment


def load_sae(layer: int) -> dict:
    d = DATA_DIR / "sae" / f"L{layer}"
    return {k: np.load(d / f"{k}.npy") for k in ["W_enc", "b_enc", "W_dec", "b_dec"]}


def encode(sae: dict, x: np.ndarray) -> np.ndarray:
    return np.maximum(0, (x - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"])


def build_lexicon(idx, sentence_ids, top_k=40) -> list[str]:
    counts = Counter()
    for sid in sentence_ids:
        entry = idx["sentences"][sid]
        npz = np.load(DATA_DIR / "activations" / entry["file"], allow_pickle=True)
        counts.update(list(npz["tokens"]))
    return [t for t, _ in counts.most_common(top_k)]


class PrimitiveBuilder:
    """toks = tokens; len_seq = len(toks); out = np.zeros((len_seq, n_primitives))
    -- same rigid opening as the original codebase's program contract, just
    producing a primitive feature matrix instead of an attention matrix."""

    def __init__(self, lexicon: list[str]):
        self.lexicon = lexicon
        self.lex_index = {t: i for i, t in enumerate(lexicon)}
        self.n_lex = len(lexicon) + 1
        self.n_pos = len(POS_TAGS) + 1
        self.n_dims = self.n_lex + self.n_pos + 3

    def __call__(self, tokens: list[str], sentence: str) -> np.ndarray:
        toks = tokens
        len_seq = len(toks)
        out = np.zeros((len_seq, self.n_dims))

        doc = get_nlp()(sentence)
        alignment = align_to_spacy(toks, doc)

        for i, tok in enumerate(toks):
            lex_idx = self.lex_index.get(tok, len(self.lexicon))
            out[i, lex_idx] = 1.0

            pos_tag = doc[alignment[i][0]].pos_ if alignment[i] else None
            pos_col = self.n_lex + (POS_TAGS.index(pos_tag) if pos_tag in POS_TAGS else len(POS_TAGS))
            out[i, pos_col] = 1.0

            out[i, -3] = i / max(len_seq - 1, 1)
            out[i, -2] = 1.0 if i == 0 else 0.0
            out[i, -1] = 1.0 if i == len_seq - 1 else 0.0

        return out


def dedup_top_features(idx, layer: int, sae: dict, k: int = 3, pool: int = 40,
                        jaccard_thresh: float = 0.7) -> list[int]:
    """Rank features by firing frequency across all 100 sentences, then greedily
    keep the top-frequency feature from each non-overlapping cluster (same
    Jaccard-overlap dedup used by hand for L0/L12 last message, generalized)."""
    n_features = sae["W_enc"].shape[1]
    fire_count = np.zeros(n_features, dtype=np.int64)
    fired_by_sentence = []
    for entry in idx["sentences"]:
        npz = np.load(DATA_DIR / "activations" / entry["file"], allow_pickle=True)
        x = npz["hidden_states"][layer]
        f = encode(sae, x) > 0
        fire_count += f.sum(axis=0)
        fired_by_sentence.append(f)

    candidates = list(np.argsort(fire_count)[::-1][:pool])
    fired = {int(f): np.concatenate([fs[:, f] for fs in fired_by_sentence]) for f in candidates}

    chosen: list[int] = []
    for f in candidates:
        f = int(f)
        if fire_count[f] == 0:
            continue
        is_dup = False
        for c in chosen:
            a, b = fired[f], fired[c]
            union = (a | b).sum()
            jac = (a & b).sum() / union if union > 0 else 0
            if jac > jaccard_thresh:
                is_dup = True
                break
        if not is_dup:
            chosen.append(f)
        if len(chosen) >= k:
            break
    return chosen


def fit_and_evaluate(idx, split, layer: int, feature: int, sae: dict, builder: PrimitiveBuilder):
    def gather(sentence_ids):
        Xs, ys = [], []
        for sid in sentence_ids:
            entry = idx["sentences"][sid]
            npz = np.load(DATA_DIR / "activations" / entry["file"], allow_pickle=True)
            x = npz["hidden_states"][layer]
            tokens = list(npz["tokens"])
            sentence = str(npz["sentence"])
            y = encode(sae, x)[:, feature] > 0
            Xs.append(builder(tokens, sentence))
            ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = gather(split["train"])
    X_test, y_test = gather(split["test"])

    if y_train.sum() < 2 or y_train.sum() == len(y_train):
        return None  # degenerate: never/always fires in train, logistic regression undefined

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1, "n_test_fires": int(y_test.sum())}
