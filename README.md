# Replacing Attention Heads

Lightweight workspace for testing symbolic hypothesis programs against attention heads in BERT, GPT-2, and TinyLlama.

<!-- Original repository: https://github.com/AmiriHayes/LLM-Interpretability -->

<!-- Preprint: https://www.overleaf.com/6482759765tsfvtgdxygym#4ec445 -->

## What This Contains

- `code/make_prorams.ipynb`: Generates symbolic hypothesis programs for attention heads. (makes programs)
- `code/write_data.ipynb`: Generates IoU and interpolation CSV files. (scores programs)
- `code/all_experiments.ipynb`: Produces figures, best-fit mappings, and replacement experiments. (tests programs)

- `data/`: Input assets and generated score tables.
- `results/`: Best fits, plots, and replacement run outputs.

## Quick Start

1. Run `code/write_data.ipynb` to generate/update data CSVs. (takes hours, data is included in repo)
2. Run `code/all_experiments.ipynb` to generate figures and experiment outputs.
3. Check outputs in `results/plots` and `results/replacement_run`.

## Notes

- Paths in notebooks are set relative to `code/` (for example, `../data`, `../results`).
- The notebooks attempt to use consistent logging tags: `[INFO]`, `[WARN]`, `[DONE]`.

## Assignment-control baselines

`code/baseline_conditions.py` runs six head↔program assignment conditions
through one harness, using attention-weight-level replacement
(`code/fixed_attention_gpt2.py` — the program's causally-masked,
row-normalized pattern is substituted for the head's softmaxed attention
before value mixing; identity-checked at start of every run):

| condition | assignment |
|---|---|
| `correct` | each head's own best-fit program (deterministic — run once) |
| `within_category` | heads exchange programs within their functional category (`results/{model}_program_categories.json`) |
| `permute_layer` | heads exchange programs within their layer |
| `permute_global` | cyclic permutation of the assignment vector across all heads — every program is some head's genuine best-fit; heads that share a best-fit program can still receive their own |
| `random` | uniform library draw, excluding the head's own program |
| `cross_category` | a genuine best-fit program from a *different* functional category |

Smoke test (25 heads, one condition):
```
python code/baseline_conditions.py --modes permute_global --seeds 0 --smoke
```
Full suite (results land in `results/replacement_run/permutations_fixed/`):
```
python code/baseline_conditions.py --modes correct within_category \
    permute_layer permute_global random cross_category --seeds 0 1 2 3 4
```
Summary table + three-panel figure (cost curves, fidelity-vs-cost,
trajectory areas):
```
python code/analyze_baselines.py
```
Why these controls matter: "replacing heads with their own programs is cheap"
is evidence the programs capture head-specific function only if wrong
assignments are more expensive; the graded conditions separate head-level,
layer-level, and category-level specificity, and the fidelity panel checks
that pattern-IoU predicts replacement cost with the right sign.
