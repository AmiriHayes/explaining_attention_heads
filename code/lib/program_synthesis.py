"""
Model-agnostic attention-head program synthesis.

Engine for make_programs.ipynb. Nothing here is specific to any model family:
layer/head counts, tokenizer and embeddings are all read off a loaded model.

Scoring is soft IoU throughout (the repo convention, `data/iou_scores_*.csv`):
    IoU(p, q) = sum(min(p, q)) / sum(max(p, q))
Higher is better. JSD is deliberately not used anywhere.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def iou_score(p: np.ndarray, q: np.ndarray) -> float:
    """Soft IoU between two attention matrices. Repo convention; higher is better."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    return float(np.minimum(p, q).sum() / np.maximum(p, q).sum())


def score_program(fn, tokenizer, sentences, real_attn_by_sentence):
    """Run `fn` over sentences, score each against real attention.

    Returns dict with mean_iou and a per-sentence list sorted best-first.
    A sentence the program fails on scores 0.0 and is marked unsuccessful.
    """
    per = []
    for sent in sentences:
        real = real_attn_by_sentence[sent]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _, mat = fn(sent, tokenizer)
            mat = np.asarray(mat, dtype=np.float64)
            if mat.shape != real.shape:
                per.append({"sentence": sent, "iou": 0.0, "success": False,
                            "error": f"shape {mat.shape} != {real.shape}", "pred": None})
                continue
            per.append({"sentence": sent, "iou": iou_score(real, mat),
                        "success": True, "error": None, "pred": mat})
        except Exception as e:
            per.append({"sentence": sent, "iou": 0.0, "success": False,
                        "error": f"{type(e).__name__}: {e}", "pred": None})
    ok = [s for s in per if s["success"]]
    per.sort(key=lambda s: -s["iou"])          # best first
    return {
        "mean_iou": float(np.mean([s["iou"] for s in per])) if per else 0.0,
        "num_success": len(ok),
        "num_fail": len(per) - len(ok),
        "per_sentence": per,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Representation — stratified pairs
# ─────────────────────────────────────────────────────────────────────────────

STRATA = [("high", 0.75, 1.0, 15), ("upper-mid", 0.50, 0.75, 10),
          ("lower-mid", 0.25, 0.50, 10), ("low", 0.0, 0.25, 10)]
MIN_WEIGHT = 0.01
MAX_BORING = 3


def _edge_lines(edges, tokens):
    return [f"  '{tokens[i]}'[{i}] -> '{tokens[j]}'[{j}] ({w:.3f})" for i, j, w in edges]


def represent_head(tokens, attention, sentence, strata=STRATA,
                   min_weight=MIN_WEIGHT, max_boring=MAX_BORING):
    """Render one head's attention on one sentence as stratified text.

    Edges are bucketed by weight QUANTILE (not raw rank), so a head whose mass
    sits on the diagonal still shows its off-diagonal structure. Self-attention
    and first-token edges are capped per stratum so they cannot crowd out the
    signal that distinguishes one head from another.
    """
    n = len(tokens)
    edges = [(i, j, float(attention[i, j]))
             for i in range(n) for j in range(i + 1)
             if attention[i, j] >= min_weight]
    if not edges:
        return f'Sentence: "{sentence}"\n  (no edges above {min_weight})'

    weights = np.array([w for _, _, w in edges])
    out = [f'Sentence: "{sentence}"']
    for label, lo_q, hi_q, cap in strata:
        lo, hi = np.quantile(weights, lo_q), np.quantile(weights, hi_q)
        band = [e for e in edges if (lo <= e[2] <= hi if hi_q == 1.0 else lo <= e[2] < hi)]
        if not band:
            continue
        band.sort(key=lambda e: -e[2])
        kept, n_self, n_first, dropped = [], 0, 0, 0
        for i, j, w in band:
            if i == j:
                if n_self >= max_boring: dropped += 1; continue
                n_self += 1
            elif j == 0:
                if n_first >= max_boring: dropped += 1; continue
                n_first += 1
            kept.append((i, j, w))
            if len(kept) >= cap: break
        if not kept:
            continue
        out.append(f"\n{label} attention (p{int(lo_q*100)}-p{int(hi_q*100)}, {lo:.3f}-{hi:.3f}):")
        out.extend(_edge_lines(kept, tokens))
        if dropped:
            out.append(f"  (+ {dropped} more self-attention / first-token edges)")
    return "\n".join(out)


def represent_comparison(tokens, real, pred, sentence,
                         min_weight=MIN_WEIGHT, max_per_section=15):
    """Render real-vs-predicted disagreement, split by direction.

    under-predicted = real > pred  (structure the program MISSES)
    over-predicted  = pred > real  (structure the program HALLUCINATES)
    Agreeing self-attention is suppressed so the diff stays readable.
    """
    n = len(tokens)
    under, over = [], []
    for i in range(n):
        for j in range(i + 1):
            r, p = float(real[i, j]), float(pred[i, j])
            if r < min_weight and p < min_weight:
                continue
            d = r - p
            if abs(d) < 0.05 and i == j:
                continue
            (under if d > 0 else over).append((i, j, r, p, abs(d)))
    under.sort(key=lambda x: -x[4]); over.sort(key=lambda x: -x[4])

    def fmt(edges):
        lines = [f"  '{tokens[i]}'[{i}] -> '{tokens[j]}'[{j}] (real: {r:.3f}, pred: {p:.3f})"
                 for i, j, r, p, _ in edges[:max_per_section]]
        if len(edges) > max_per_section:
            lines.append(f"  (+ {len(edges) - max_per_section} more)")
        return lines

    out = [f'Sentence: "{sentence}"']
    if under:
        out.append(f"under-predicted (real > pred, {len(under)} edges):"); out += fmt(under)
    if over:
        out.append(f"over-predicted (pred > real, {len(over)} edges):"); out += fmt(over)
    if not under and not over:
        out.append("  (no disagreement above threshold)")
    return "\n".join(out)


def format_head_examples(head_data, layer, head, num_examples=5, seed=42):
    """Sample sentences for one head and render each. Deterministic per head."""
    import random
    rng = random.Random(seed + layer * 1000 + head)
    sel = rng.sample(head_data, num_examples) if len(head_data) > num_examples else list(head_data)
    return "\n\n".join(
        f"Example {k + 1}:\n{represent_head(d['tokens'], d['attention'][layer, head], d['sentence'])}"
        for k, d in enumerate(sel))


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction / validation / execution
# ─────────────────────────────────────────────────────────────────────────────

FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(response: str):
    m = FENCE.findall(response)
    return m[0].strip() if m else None


def validate_code(code: str, func_name: str):
    """Parse, confirm the expected function exists, and confirm it references
    only names the emitted module will actually define. That last check is the
    one that was missing before: it is what lets a program ship calling a
    helper nobody defines."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if func_name not in funcs:
        return False, f"function '{func_name}' not defined (found: {funcs})"
    return True, ""


def undefined_names(code: str, allowed: set):
    """Names loaded but bound nowhere — the llama3b failure mode, caught early."""
    tree = ast.parse(code)
    import builtins
    bound = set(dir(builtins)) | set(allowed)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(n.name)
            for a in n.args.args + n.args.kwonlyargs: bound.add(a.arg)
            if n.args.vararg: bound.add(n.args.vararg.arg)
            if n.args.kwarg: bound.add(n.args.kwarg.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): bound.add(n.id)
        elif isinstance(n, ast.alias): bound.add((n.asname or n.name).split('.')[0])
        elif isinstance(n, (ast.comprehension,)): pass
        elif isinstance(n, ast.ExceptHandler) and n.name: bound.add(n.name)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - bound


class Sandbox:
    """Executes generated programs against a live helper namespace.

    Helpers are exec'd once into a shared namespace and reused, so spaCy and the
    embedding matrix load a single time for the whole run.
    """
    def __init__(self, helpers_source: str):
        self.ns = {"np": np, "__builtins__": __builtins__}
        exec(compile(helpers_source, "<helpers>", "exec"), self.ns)
        self.helper_names = {k for k in self.ns if not k.startswith("__")}

    def load(self, code: str, func_name: str):
        local = dict(self.ns)
        exec(compile(code, f"<{func_name}>", "exec"), local)
        return local[func_name]

    def smoke(self, code: str, func_name: str, tokenizer, sentence: str):
        try:
            fn = self.load(code, func_name)
            with contextlib.redirect_stderr(io.StringIO()):
                label, mat = fn(sentence, tokenizer)
            mat = np.asarray(mat, dtype=np.float64)
            n = mat.shape[0]
            if mat.shape != (n, n):
                return False, f"not square: {mat.shape}"
            if not np.allclose(mat.sum(axis=1), 1.0, atol=1e-3):
                return False, "rows not stochastic"
            if not np.allclose(mat, np.tril(mat)):
                return False, "not lower-triangular (causal mask missing)"
            if not isinstance(label, str):
                return False, f"first return value must be str, got {type(label).__name__}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Provider-agnostic LLM client
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMClient:
    """Thin litellm wrapper with a SHA-keyed disk cache.

    model is any litellm route: 'claude-opus-5', 'gpt-5', 'openai/gpt-4o', ...
    The cache is what makes a 1000+ head run resumable.
    """
    model: str = "claude-opus-5"
    cache_dir: Path = Path(".llm_cache")
    max_tokens: int = 32000
    calls: int = 0
    cache_hits: int = 0

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt, system):
        h = hashlib.sha256(f"{self.model}\x00{system}\x00{prompt}".encode()).hexdigest()[:20]
        return self.cache_dir / f"{h}.json"

    def __call__(self, prompt: str, system: str = "") -> str:
        path = self._key(prompt, system)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())["response"]
        from litellm import completion
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        resp = completion(model=self.model, messages=msgs, max_tokens=self.max_tokens)
        text = resp.choices[0].message.content or ""
        self.calls += 1
        if not text.strip():
            # Reasoning models return an empty message when reasoning tokens exhaust
            # max_tokens. Caching that would poison every retry, so raise instead.
            fin = getattr(resp.choices[0], "finish_reason", "?")
            raise RuntimeError(f"empty response (finish_reason={fin}); "
                               f"raise max_tokens (currently {self.max_tokens})")
        path.write_text(json.dumps({"model": self.model, "system": system,
                                    "prompt": prompt, "response": text}))
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Helper surface
#
# ONE source of truth. This exact text is (a) injected into every prompt and
# (b) inlined verbatim into the emitted <model>_programs.py. That identity is
# the fix for the llama3b bug, where prompts advertised helpers the emitted
# module never defined and 51% of programs died on NameError.
# ─────────────────────────────────────────────────────────────────────────────

HELPERS_TEMPLATE = '''\
import numpy as np

_MODEL_ID = {model_id!r}
_nlp = None
_emb = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_embeddings():
    """Input-embedding matrix of the model under study, loaded once."""
    global _emb
    if _emb is None:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(_MODEL_ID)
        _emb = m.get_input_embeddings().weight.detach().to("cpu").float().numpy()
    return _emb


def tokenize(sentence, tokenizer):
    """Model tokens for `sentence`, as strings. This is the tokenization every
    program must use -- the attention matrix is aligned to it."""
    ids = tokenizer(sentence, return_tensors="pt").input_ids[0]
    return tokenizer.convert_ids_to_tokens(ids)


def spacy_parse(sentence):
    """spaCy Doc: .pos_, .dep_, .head, .ent_type_, .lemma_ ..."""
    return _get_nlp()(sentence)


def _char_spans(sentence, tokenizer):
    """(start, end) char span per model token. Uses fast-tokenizer offsets when
    available, else falls back to cumulative decoded length."""
    try:
        enc = tokenizer(sentence, return_offsets_mapping=True)
        om = enc["offset_mapping"]
        if om and any(e > s for s, e in om):
            return [tuple(x) for x in om]
    except Exception:
        pass
    spans, pos = [], 0
    for t in tokenize(sentence, tokenizer):
        clean = t.lstrip("\\u0120").lstrip("##").lstrip()
        idx = sentence.find(clean, pos) if clean else pos
        if idx < 0:
            idx = pos
        spans.append((idx, idx + len(clean)))
        pos = idx + max(len(clean), 1)
    return spans


def align_tokens_to_spacy(sentence, tokens, tokenizer):
    """For each model token, the indices of spaCy tokens it overlaps."""
    doc = _get_nlp()(sentence)
    spans = _char_spans(sentence, tokenizer)
    out = []
    for (s, e) in spans[:len(tokens)]:
        out.append([k for k, w in enumerate(doc)
                    if not (w.idx + len(w.text) <= s or w.idx >= e)])
    while len(out) < len(tokens):
        out.append([])
    return out


def align_spacy_to_tokens(sentence, tokens, tokenizer):
    """Inverse map: for each spaCy token, the model-token indices covering it."""
    doc = _get_nlp()(sentence)
    fwd = align_tokens_to_spacy(sentence, tokens, tokenizer)
    out = [[] for _ in doc]
    for ti, sids in enumerate(fwd):
        for si in sids:
            if si < len(out):
                out[si].append(ti)
    return out


def get_modifying_adjectives(token):
    """spaCy children of `token` that are adjectival modifiers."""
    return [c for c in token.children if c.dep_ in ("amod", "advmod")]


def embedding_similarity(tokens, i, j, tokenizer):
    """Cosine similarity between tokens[i] and tokens[j] in the model's own
    input-embedding space. Use this instead of string equality to catch
    morphological and semantic relatives ("heard"/"hear", "cat"/"kitten")."""
    emb = _get_embeddings()
    try:
        a = emb[tokenizer.convert_tokens_to_ids(tokens[i])]
        b = emb[tokenizer.convert_tokens_to_ids(tokens[j])]
    except Exception:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_row_stochastic(matrix):
    """Normalize each row to sum to 1; uniform-causal fallback for dead rows."""
    m = np.asarray(matrix, dtype=np.float64).copy()
    m[m < 0] = 0.0
    for i in range(m.shape[0]):
        s = m[i].sum()
        if s <= 0:
            m[i, : i + 1] = 1.0 / (i + 1)
        else:
            m[i] /= s
    return m


def apply_causal_mask(matrix):
    """Zero the strict upper triangle: token i may only attend to j <= i."""
    return np.tril(np.asarray(matrix, dtype=np.float64))
'''


def helpers_source(model_id: str) -> str:
    return HELPERS_TEMPLATE.format(model_id=model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Emitter
# ─────────────────────────────────────────────────────────────────────────────

MODULE_HEADER = '''\
"""
{model_key}_programs.py -- generated by make_programs.ipynb

Model:     {model_id}
Shape:     {n_layers} layers x {n_heads} heads = {n_progs} programs
Signature: (sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]
Scoring:   soft IoU (min/max), higher is better

Self-contained: every helper the programs call is defined below. Generated
programs are written out verbatim -- there is no text-rewriting conversion step.
"""
from typing import Tuple

from transformers import PreTrainedTokenizerBase

'''


def emit_module(path, model_key, model_id, n_layers, n_heads, programs):
    """Write <model>_programs.py.

    `programs` maps (layer, head) -> source of one `prog_L{{l}}H{{h}}` function.
    Names end in _L#H#, which satisfies both repo loaders:
      write_data.ipynb        re.search(r'_[Ll]\\d+[Hh]\\d+$', name)
      all_experiments_v2 etc. not name.startswith('_')
    """
    path = Path(path)
    parts = [MODULE_HEADER.format(model_key=model_key, model_id=model_id,
                                  n_layers=n_layers, n_heads=n_heads,
                                  n_progs=len(programs)),
             helpers_source(model_id), "\n"]
    for (l, h) in sorted(programs):
        parts.append("\n\n" + programs[(l, h)].rstrip() + "\n")
    path.write_text("".join(parts))
    return path


def verify_module(path, tokenizer, sentences, expected_n):
    """Import the emitted module and run every program. Fails loudly -- the
    whole point, since the repo's consumers swallow these exceptions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(Path(path).stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import inspect
    progs = {n: f for n, f in inspect.getmembers(mod, inspect.isfunction)
             if re.search(r"_[Ll]\d+[Hh]\d+$", n)}
    report = {"n_programs": len(progs), "expected": expected_n,
              "failures": {}, "not_stochastic": [], "not_causal": []}
    for name, fn in progs.items():
        for sent in sentences:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    label, mat = fn(sent, tokenizer)
                mat = np.asarray(mat, dtype=np.float64)
                if not np.allclose(mat.sum(axis=1), 1.0, atol=1e-3):
                    report["not_stochastic"].append(name); break
                if not np.allclose(mat, np.tril(mat)):
                    report["not_causal"].append(name); break
            except Exception as e:
                report["failures"][name] = f"{type(e).__name__}: {e}"
                break
    report["n_ok"] = len(progs) - len(report["failures"]) \
        - len(report["not_stochastic"]) - len(report["not_causal"])
    return report, mod


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert at analyzing transformer attention patterns and writing "
    "Python code to approximate them. You write clean, correct Python that uses "
    "spacy and numpy to predict attention patterns based on linguistic features."
)

_CONTRACT = '''\
## Your task

Write a Python function with this EXACT signature:

```python
def {fname}(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
```

It must return `(label, attention_matrix)` where:
  - `label` is a short string naming your hypothesis, e.g. "previous_token_with_decay"
  - `attention_matrix` is an (n, n) numpy array, n = len(tokenize(sentence, tokenizer))
  - every row sums to 1.0 (row-stochastic)
  - the matrix is lower-triangular: token i attends only to tokens j <= i

## Available helper functions

These are already defined in the execution environment. Do NOT redefine them,
and do NOT call anything that is not in this list:

```python
{helpers}
```

## Important constraint: no hard-coded words

Your code must NOT hard-code specific word strings (e.g. "she", "the", "said").
Use linguistic features from spacy instead: POS tags (token.pos_), dependency
relations (token.dep_), entity types (token.ent_type_). You MAY test for
punctuation, whitespace, and the tokenizer's own special tokens directly.

To detect semantic or morphological relatedness ("heard"/"hear", "cat"/"kitten"),
use `embedding_similarity(tokens, i, j, tokenizer)` rather than string equality.

## Output format

Start the function with a 1-2 line docstring naming your hypothesis about what
this head computes. Write ONLY the function definition, inside a single ```python
block. No imports, no test code, no example usage.'''


def prog_name(layer: int, head: int) -> str:
    return f"prog_L{layer}H{head}"


def build_initial_prompt(head_examples, layer, head, model_name, helpers, fname=None):
    fname = fname or prog_name(layer, head)
    return f'''\
I'm analyzing attention head L{layer}H{head} of {model_name}. Below are examples of
its attention patterns on several sentences. Edges are grouped into weight strata
(high / upper-mid / lower-mid / low) so that both the dominant and the subtler
structure of this head are visible.

{head_examples}

{_CONTRACT.format(fname=fname, helpers=helpers)}

## Hints

Look for patterns such as:
  - Positional (previous token, first token, self, fixed offset, recency decay)
  - Syntactic (attend to syntactic head, to dependents, to modifiers)
  - Token identity / similarity (use `embedding_similarity`)
  - Mixed (positional prior plus a linguistic correction)

Your function will be called on new sentences not shown above, so prefer general
mechanisms over anything fitted to these specific examples.'''


def build_diverse_prompt(head_examples, layer, head, model_name, helpers,
                         hypothesis_index, k, fname=None):
    fname = fname or prog_name(layer, head)
    return f'''\
I'm analyzing attention head L{layer}H{head} of {model_name}. Below are examples of
its attention patterns on several sentences.

{head_examples}

## Diversity constraint

This is hypothesis {hypothesis_index + 1} of {k} for this head. Each hypothesis must
propose a FUNDAMENTALLY DIFFERENT theory of what this head computes -- pick a
different mechanism than you would by default. Categories to consider: pure
positional, syntactic-structural, semantic/POS-driven, recency-weighted or
windowed, mixed positional-plus-linguistic.

{_CONTRACT.format(fname=fname, helpers=helpers)}'''


def build_refinement_prompt(original_code, feedback, layer, head, model_name,
                            helpers, fname=None):
    """feedback: list of (iou, comparison_repr), best-first then worst-last."""
    fname = fname or prog_name(layer, head)
    blocks = "\n\n".join(
        f"### Sentence {i} (IoU = {iou:.4f})\n{rep}"
        for i, (iou, rep) in enumerate(feedback, 1))
    return f'''\
I'm refining the attention-prediction code for head L{layer}H{head} of {model_name}.

## Original code

```python
{original_code}
```

## Evaluation feedback

Each sentence below shows where the original code disagreed with the real
attention. Edges are grouped into "under-predicted" (real attention is higher
than predicted -- structure the code MISSES) and "over-predicted" (predicted is
higher than real -- structure the code HALLUCINATES), sorted by disagreement
magnitude. Agreeing self-attention edges are suppressed. Scoring is soft IoU,
where HIGHER IS BETTER. The first sentences are the best matches and the last
are the worst.

{blocks}

## Your task

The original code already scores well on most sentences. Make a MINIMAL, TARGETED
modification that handles the failure cases without degrading the cases that
already work.

CRITICAL: Start from the original code and ADD a small amount of new logic. Do
NOT rewrite the function from scratch. Preserve the original's self-attention
weights, decay rates, and overall structure exactly. Add a new code path (e.g.
an if-statement) that detects the specific pattern the original misses and
adjusts those weights only.

Think of it as: the original code is the default behavior; you are adding a
special-case handler that fires only when that pattern is detected and leaves
every other weight unchanged.

{_CONTRACT.format(fname=fname, helpers=helpers)}'''


# ─────────────────────────────────────────────────────────────────────────────
# Improvement strategies (pluggable)
#
# Every strategy takes a generated-and-scored initial program and returns the
# program to keep, plus a record of what it tried. Add a new one and register it
# in STRATEGIES -- the driver needs no changes.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    code: str
    score: dict
    origin: str          # 'initial' | 'refined' | 'diverse_2' | 'blend' | ...

    @property
    def iou(self): return self.score["mean_iou"]


@dataclass
class HeadContext:
    """Everything a strategy may need for one head."""
    layer: int
    head: int
    model_name: str
    helpers: str
    head_examples: str
    sentences: list
    real_attn: dict
    tokenizer: object
    sandbox: Sandbox
    llm: LLMClient
    n_feedback: int = 3
    last_error: str = ""

    def generate(self, prompt):
        """LLM call -> validated, smoke-tested code. None if unusable."""
        fname = prog_name(self.layer, self.head)
        for attempt in range(3):
            p = prompt if attempt == 0 else (
                prompt + f"\n\n(Note: attempt {attempt+1} -- previous code failed with: {err})")
            try:
                resp = self.llm(p, system=SYSTEM_PROMPT)
            except Exception as e:
                err = f"api error: {type(e).__name__}: {e}"; continue
            code = extract_code(resp)
            if code is None:
                err = "no python code block in response"; continue
            ok, err = validate_code(code, fname)
            if not ok:
                continue
            missing = undefined_names(code, self.sandbox.helper_names | {"np", "Tuple",
                                      "PreTrainedTokenizerBase"})
            if missing:
                err = f"calls undefined helpers: {sorted(missing)}"; continue
            ok, err = self.sandbox.smoke(code, fname, self.tokenizer, self.sentences[0])
            if not ok:
                continue
            return code
        self.last_error = err
        return None

    def score(self, code):
        fn = self.sandbox.load(code, prog_name(self.layer, self.head))
        return score_program(fn, self.tokenizer, self.sentences, self.real_attn)

    def feedback(self, score):
        """best-n then worst-n, as (iou, comparison_repr)."""
        ok = [s for s in score["per_sentence"] if s["success"] and s["pred"] is not None]
        if not ok:
            return []
        n = self.n_feedback
        chosen = ok[:n] + ok[-n:] if len(ok) > 2 * n else ok
        out = []
        for s in chosen:
            toks = tokenize_with(self.tokenizer, s["sentence"])
            out.append((s["iou"], represent_comparison(
                toks, self.real_attn[s["sentence"]], s["pred"], s["sentence"])))
        return out


def tokenize_with(tokenizer, sentence):
    ids = tokenizer(sentence, return_tensors="pt").input_ids[0]
    return tokenizer.convert_ids_to_tokens(ids)


def strat_single(ctx, initial: Candidate):
    """No improvement -- keep the zero-shot program."""
    return initial, [initial]


def strat_two_pass(ctx, initial: Candidate):
    """Jacob's method: one refinement conditioned on scored feedback; keep better."""
    fb = ctx.feedback(initial.score)
    if not fb:
        return initial, [initial]
    code = ctx.generate(build_refinement_prompt(
        initial.code, fb, ctx.layer, ctx.head, ctx.model_name, ctx.helpers))
    if code is None:
        return initial, [initial]
    cand = Candidate(code, ctx.score(code), "refined")
    pool = [initial, cand]
    return max(pool, key=lambda c: c.iou), pool


def strat_alpha_blend(ctx, initial: Candidate):
    """Two-pass, then also try a blended program: alpha*initial + (1-alpha)*refined.

    The blend is emitted as a real function wrapping both parents, so it stays a
    single program satisfying the module contract.
    """
    best, pool = strat_two_pass(ctx, initial)
    refined = next((c for c in pool if c.origin == "refined"), None)
    if refined is None:
        return best, pool

    fa = ctx.sandbox.load(initial.code, prog_name(ctx.layer, ctx.head))
    fb_ = ctx.sandbox.load(refined.code, prog_name(ctx.layer, ctx.head))

    def blended(alpha):
        tot = []
        for s in ctx.sentences:
            try:
                _, ma = fa(s, ctx.tokenizer); _, mb = fb_(s, ctx.tokenizer)
                m = alpha * np.asarray(ma, float) + (1 - alpha) * np.asarray(mb, float)
                tot.append(iou_score(ctx.real_attn[s], m))
            except Exception:
                tot.append(0.0)
        return float(np.mean(tot))

    alphas = np.linspace(0, 1, 11)
    ious = [blended(a) for a in alphas]
    a_star = float(alphas[int(np.argmax(ious))])
    if max(ious) <= best.iou or a_star in (0.0, 1.0):
        return best, pool

    code = make_blend_source(ctx.layer, ctx.head, initial.code, refined.code, a_star)
    ok, err = ctx.sandbox.smoke(code, prog_name(ctx.layer, ctx.head),
                                ctx.tokenizer, ctx.sentences[0])
    if not ok:
        return best, pool
    cand = Candidate(code, ctx.score(code), f"blend(alpha={a_star:.1f})")
    pool.append(cand)
    return max(pool, key=lambda c: c.iou), pool


def strat_greedy(ctx, initial: Candidate, rounds=4):
    """Amiri's greedy method: refine repeatedly, keep only if IoU improves,
    stop early on the first non-improving round."""
    best, pool = initial, [initial]
    for r in range(rounds):
        fb = ctx.feedback(best.score)
        if not fb:
            break
        code = ctx.generate(build_refinement_prompt(
            best.code, fb, ctx.layer, ctx.head, ctx.model_name, ctx.helpers))
        if code is None:
            break
        cand = Candidate(code, ctx.score(code), f"refinement_{r+1}")
        pool.append(cand)
        if cand.iou <= best.iou:
            break
        best = cand
    return best, pool


def strat_best_of_k(ctx, initial: Candidate, k=3):
    """Generate k diverse hypotheses and keep the best-scoring."""
    pool = [initial]
    for i in range(1, k):
        code = ctx.generate(build_diverse_prompt(
            ctx.head_examples, ctx.layer, ctx.head, ctx.model_name, ctx.helpers, i, k))
        if code is not None:
            pool.append(Candidate(code, ctx.score(code), f"diverse_{i+1}"))
    return max(pool, key=lambda c: c.iou), pool


def make_blend_source(layer, head, code_a, code_b, alpha):
    """Compose two program bodies into one public program that blends them.

    Inner functions are underscore-prefixed so neither repo loader picks them up
    as programs; only the public prog_L#H# wrapper is exported.
    """
    name = prog_name(layer, head)
    a = code_a.replace(f"def {name}(", f"def _{name}_a(", 1)
    b = code_b.replace(f"def {name}(", f"def _{name}_b(", 1)
    return f'''{a}


{b}


def {name}(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Blend of the initial and refined hypotheses (alpha={alpha:.2f})."""
    la, ma = _{name}_a(sentence, tokenizer)
    lb, mb = _{name}_b(sentence, tokenizer)
    m = {alpha:.4f} * np.asarray(ma, dtype=np.float64) + {1-alpha:.4f} * np.asarray(mb, dtype=np.float64)
    m = apply_causal_mask(m)
    m = make_row_stochastic(m)
    return f"blend({{la}}|{{lb}})", m'''


STRATEGIES = {
    "single":      strat_single,
    "two_pass":    strat_two_pass,
    "alpha_blend": strat_alpha_blend,
    "greedy":      strat_greedy,
    "best_of_k":   strat_best_of_k,
}


# ─────────────────────────────────────────────────────────────────────────────
# Grid driver
#
# One implementation shared by make_programs.ipynb and run_synthesis.py, so the
# notebook and the script cannot drift apart.
# ─────────────────────────────────────────────────────────────────────────────

def run_grid(heads, data, score_sentences, tokenizer, sandbox, llm, helpers,
             model_name, strategy, prog_dir, manifest_path, seed=42,
             workers=1, log=print):
    """Generate one program per head. Resumable and thread-safe.

    heads            : [(layer, head), ...]
    data             : [{tokens, sentence, attention}, ...] from extraction
    score_sentences  : sentences each program is scored on
    strategy         : name in STRATEGIES, or a callable(ctx, initial)
    Returns the manifest dict.
    """
    import threading, time
    from concurrent.futures import ThreadPoolExecutor

    prog_dir = Path(prog_dir); prog_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    improve = STRATEGIES[strategy] if isinstance(strategy, str) else strategy

    # Warm the lazily-initialised globals before any thread touches them.
    sandbox.ns["_get_nlp"]()
    try:
        sandbox.ns["_get_embeddings"]()
    except Exception as e:
        log(f"[warn] embeddings unavailable ({e}); embedding_similarity will return 0.0")

    todo = [(l, h) for (l, h) in heads
            if not ((prog_dir / f"L{l}H{h}.py").exists() and f"L{l}H{h}" in manifest)]
    log(f"[run] {len(todo)} heads to do, {len(heads) - len(todo)} cached, workers={workers}")

    lock, done, t0 = threading.Lock(), [0], time.time()
    score_set = set(score_sentences)

    def one(lh):
        l, h = lh
        key = f"L{l}H{h}"
        n_tok = lambda d: len(d["tokens"])
        real = {d["sentence"]: d["attention"][l, h][:n_tok(d), :n_tok(d)].astype(np.float64)
                for d in data if d["sentence"] in score_set}
        ctx = HeadContext(layer=l, head=h, model_name=model_name, helpers=helpers,
                          head_examples=format_head_examples(data, l, h, seed=seed),
                          sentences=score_sentences, real_attn=real, tokenizer=tokenizer,
                          sandbox=sandbox, llm=llm)
        try:
            code = ctx.generate(build_initial_prompt(ctx.head_examples, l, h, model_name, helpers))
            if code is None:
                entry = {"status": "failed_generation", "layer": l, "head": h,
                         "reason": ctx.last_error}
            else:
                initial = Candidate(code, ctx.score(code), "initial")
                best, pool = improve(ctx, initial)
                (prog_dir / f"{key}.py").write_text(best.code)
                entry = {"status": "ok", "layer": l, "head": h, "winner": best.origin,
                         "iou": round(best.iou, 4), "initial_iou": round(initial.iou, 4),
                         "candidates": {c.origin: round(c.iou, 4) for c in pool}}
        except Exception as e:
            entry = {"status": "error", "layer": l, "head": h,
                     "reason": f"{type(e).__name__}: {e}"}
        with lock:
            manifest[key] = entry
            manifest_path.write_text(json.dumps(manifest, indent=1))
            done[0] += 1
            tag = (f"iou={entry['iou']:.3f} ({entry['winner']}) init={entry['initial_iou']:.3f}"
                   if entry["status"] == "ok" else f"{entry['status'].upper()}: {entry.get('reason','')[:60]}")
            log(f"  [{done[0]}/{len(todo)}] {key}  {tag}  "
                f"api={llm.calls} cached={llm.cache_hits}  {time.time()-t0:.0f}s")

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, todo))
    else:
        for lh in todo:
            one(lh)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Corpus + extraction
#
# Shared by make_programs.ipynb and the CLI below, so there is exactly one
# implementation of "which sentences" and "how attention is cached".
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(cache_path, n=200, seed=42, min_words=15, max_words=60,
                dataset="roneneldan/TinyStories", split="train", log=print):
    """TinyStories sentences filtered by word count.

    The RESOLVED SENTENCE TEXT is written to cache_path, not just a seed -- a
    rerun then reproduces exactly even if the dataset version or the shuffling
    library changes underneath.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        d = json.loads(cache_path.read_text())
        log(f"[corpus] reusing {len(d['sentences'])} cached sentences from {cache_path.name}")
        return d["sentences"]

    import random
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    pool = []
    for ex in ds:
        for s in ex["text"].replace("\n", " ").split("."):
            s = s.strip()
            if min_words <= len(s.split()) <= max_words:
                pool.append(s + ".")
        if len(pool) >= n * 3:
            break
    random.Random(seed).shuffle(pool)
    sents = pool[:n]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"dataset": dataset, "split": split, "filter": f"{min_words}-{max_words} words",
         "seed": seed, "n": len(sents), "sentences": sents}, indent=1))
    log(f"[corpus] {len(sents)} sentences -> {cache_path.name}")
    return sents


def extract_attention(model, tokenizer, sentences, out_dir, device, log=print):
    """Cache (n_layers, n_heads, seq, seq) attention per sentence as fp16 .npz.

    Skip-if-exists, so extraction is resumable. fp16 halves the on-disk cost,
    which matters at scale: Qwen3-4B is 36x32 heads, ~9 MiB per 64-token
    sentence in fp16 versus ~18 MiB in fp32.
    """
    import torch
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data, made = [], 0
    for i, s in enumerate(sentences):
        p = out_dir / f"sentence_{i:04d}.npz"
        if p.exists():
            d = np.load(p, allow_pickle=True)
            data.append({"tokens": list(d["tokens"]), "sentence": str(d["sentence"]),
                         "attention": d["attention"]})   # stays fp16; upcast per head slice
            continue
        enc = tokenizer(s, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**enc, output_attentions=True)
        att = np.stack([a[0].float().cpu().numpy() for a in o.attentions], 0)
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        np.savez_compressed(p, tokens=toks, sentence=s, attention=att.astype(np.float16))
        data.append({"tokens": toks, "sentence": s, "attention": att.astype(np.float16)})
        made += 1
    log(f"[extract] {len(data)} sentences cached ({made} new) in {out_dir.name}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# CLI
#
# The notebook is the readable entry point; this is for unattended runs, where a
# 1000+ head grid should not depend on a kernel staying alive:
#   python -m program_synthesis --model-id Qwen/Qwen3-4B-Base --model-key qwen3 \
#          --strategy two_pass --workers 8
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    import argparse, torch
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="gpt2")
    ap.add_argument("--model-key", default=None, help="output is data/<key>_programs.py")
    ap.add_argument("--llm", default="claude-opus-5", help="any litellm route")
    ap.add_argument("--strategy", default="two_pass", choices=sorted(STRATEGIES))
    ap.add_argument("--n-corpus", type=int, default=200)
    ap.add_argument("--n-score", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heads", default="all", help="'all' or 'L,H L,H ...'")
    ap.add_argument("--device", default=None,
                    help="force extraction device. A model near "
                         "torch.mps.recommended_max_memory() swaps instead of OOMing "
                         "cleanly (Qwen3-8B bf16 is 15.3 GiB vs a 16 GiB cap), so use "
                         "cpu for large models -- extraction is only a few dozen "
                         "short forward passes.")
    a = ap.parse_args(argv)
    key = a.model_key or a.model_id.split("/")[-1].lower().replace("-", "")

    root = Path(__file__).resolve().parent.parent.parent
    work = root / "code" / f".synthesis_{key}"
    work.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    if a.device:
        device = a.device

    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModel.from_pretrained(a.model_id, attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    sents = load_corpus(work / "sentences.json", n=a.n_corpus, seed=a.seed)
    data = extract_attention(model, tok, sents, work / "attention", device)
    assert data[0]["attention"].shape[:2] == (n_layers, n_heads), \
        f"attention {data[0]['attention'].shape[:2]} != config ({n_layers}, {n_heads})"
    print(f"[setup] {a.model_id}: {n_layers}x{n_heads}={n_layers*n_heads} heads | device={device}")

    helpers = helpers_source(a.model_id)
    sandbox = Sandbox(helpers)
    llm = LLMClient(model=a.llm, cache_dir=work / "llm_cache")
    heads = ([(l, h) for l in range(n_layers) for h in range(n_heads)] if a.heads == "all"
             else [tuple(map(int, x.split(","))) for x in a.heads.split()])

    manifest = run_grid(heads=heads, data=data, score_sentences=sents[:a.n_score],
                        tokenizer=tok, sandbox=sandbox, llm=llm, helpers=helpers,
                        model_name=a.model_id, strategy=a.strategy,
                        prog_dir=work / "programs", manifest_path=work / "manifest.json",
                        seed=a.seed, workers=a.workers)

    ok = {k: v for k, v in manifest.items() if v.get("status") == "ok"}
    programs = {(v["layer"], v["head"]): (work / "programs" / f"{k}.py").read_text()
                for k, v in ok.items()}
    out = root / "data" / f"{key}_programs.py"
    emit_module(out, key, a.model_id, n_layers, n_heads, programs)
    print(f"[emit] {out} ({len(programs)}/{n_layers*n_heads} programs)")

    rep, _ = verify_module(out, tok, sents[:3], n_layers * n_heads)
    print(f"[verify] {rep['n_ok']}/{rep['n_programs']} clean | failures={len(rep['failures'])} "
          f"| non-stochastic={len(rep['not_stochastic'])} | non-causal={len(rep['not_causal'])}")
    for n, e in list(rep["failures"].items())[:5]:
        print(f"    {n}: {e}")
    return 0 if not rep["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
