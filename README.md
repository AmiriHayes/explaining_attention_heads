# Replacing Attention Heads

Lightweight workspace for testing symbolic hypothesis programs against attention heads in BERT, GPT-2, and TinyLlama.

<!-- Original repository: https://github.com/AmiriHayes/LLM-Interpretability -->

<!-- Preprint: https://www.overleaf.com/6482759765tsfvtgdxygym#4ec445 -->

## What This Contains

- `code/write_data.ipynb`: Generates IoU and interpolation CSV files.
- `code/all_experiments_results.ipynb`: Produces figures, best-fit mappings, and replacement experiments.
- `data/`: Input assets and generated score tables.
- `results/`: Best fits, plots, and replacement run outputs.

## Quick Start

1. Run `code/write_data.ipynb` to generate/update data CSVs. (takes hours, data is included in repo)
2. Run `code/all_experiments_results.ipynb` to generate figures and experiment outputs.
3. Check outputs in `results/plots` and `results/replacement_run`.

## Notes

- Paths in notebooks are set relative to `code/` (for example, `../data`, `../results`).
- The notebooks use consistent logging tags: `[INFO]`, `[WARN]`, `[DONE]`.
