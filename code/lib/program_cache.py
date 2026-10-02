"""Per-sentence memoization for generated program libraries.

Every program in a library re-derives the SAME sentence-level analysis from
scratch, so with P programs active in one forward pass the parse/align/tokenize
work happens P times over identical input. In libraries emitted by the current
make_programs.ipynb, `align_tokens_to_spacy` even runs spaCy a second time on
top of `spacy_parse`, and `embedding_similarity` indexes the full input-embedding
matrix once per token pair.

Single source of truth on purpose: all_experiments{,_v2}.ipynb and any standalone
runner import this rather than carrying their own copy. The notebook's previous
inline version read `lib._align_to_spacy` unconditionally, which raised
AttributeError on every newly generated library -- so `load_programs("qwen3")`
could not run at all.

Shape-agnostic across both library generations:
  old (gpt2/tinyllama/llama3b) : spacy_parse(s), _align_to_spacy(s, tokens)
  new (make_programs.ipynb)    : tokenize(s, tok), _char_spans(s, tok),
                                 align_tokens_to_spacy(s, tokens, tok),
                                 align_spacy_to_tokens(s, tokens, tok),
                                 embedding_similarity(tokens, i, j, tok)

Verified on qwen3_programs.py: all 146 best-fit programs return bit-identical
matrices with caching on, and the per-forward program cost drops by >30x.
"""

# Helpers that are pure functions of the sentence (plus a tokenizer). `tokenize`
# returns a list the programs index into, so hand back a copy -- otherwise one
# program mutating it would corrupt every later program's view of the same key.
_SENTENCE_HELPERS = (
    ("tokenize", True),
    ("spacy_parse", False),
    ("_char_spans", False),
    ("_align_to_spacy", False),
    ("align_tokens_to_spacy", False),
    ("align_spacy_to_tokens", False),
)


def _cache_key(args):
    out = []
    for a in args:
        if isinstance(a, (list, tuple)):
            out.append(tuple(a))
        elif isinstance(a, (str, int, float, bool, type(None))):
            out.append(a)
        else:
            out.append(id(a))      # tokenizer / nlp objects: identity is enough
    return tuple(out)


def _memoize(fn, copy_list=False):
    cache = {}

    def wrapper(*args):
        key = _cache_key(args)
        if key not in cache:
            cache[key] = fn(*args)
        value = cache[key]
        return list(value) if copy_list else value

    wrapper._cache = cache
    wrapper._wrapped = fn
    return wrapper


def patch_program_caching(lib, verbose=True):
    """Memoize `lib`'s per-sentence helpers in place. Idempotent."""
    if getattr(lib, "_caching_patched", False):
        return lib

    patched = []
    for name, copy_list in _SENTENCE_HELPERS:
        fn = getattr(lib, name, None)
        if callable(fn):
            setattr(lib, name, _memoize(fn, copy_list))
            patched.append(name)

    # embedding_similarity keys on the token PAIR rather than the sentence: it is
    # called O(n^2) times per program and every call indexes the embedding matrix.
    # Both library generations are (tokens, i, j) with an optional trailing
    # tokenizer argument.
    fn = getattr(lib, "embedding_similarity", None)
    if callable(fn):
        sim_cache = {}

        def cached_embedding_similarity(tokens, i, j, *rest):
            key = (tokens[i], tokens[j]) + tuple(id(r) for r in rest)
            if key not in sim_cache:
                sim_cache[key] = fn(tokens, i, j, *rest)
            return sim_cache[key]

        cached_embedding_similarity._cache = sim_cache
        cached_embedding_similarity._wrapped = fn
        lib.embedding_similarity = cached_embedding_similarity
        patched.append("embedding_similarity")

    lib._caching_patched = True
    if verbose:
        print(f"[cache] {lib.__name__}: memoized {', '.join(patched)}")
    return lib


# Names that are helpers, not per-head programs -- excluded when collecting
# programs out of a library module.
HELPER_NAMES = frozenset({
    "_get_nlp", "_get_embeddings", "spacy_parse", "tokenize", "_char_spans",
    "_align_to_spacy", "align_tokens_to_spacy", "align_spacy_to_tokens",
    "embedding_similarity", "get_modifying_adjectives", "make_row_stochastic",
    "apply_causal_mask", "_char_sim",
})
