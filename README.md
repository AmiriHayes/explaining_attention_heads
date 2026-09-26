# Replacing Attention Heads

Lightweight workspace for testing symbolic hypothesis programs against attention heads in BERT, GPT-2, and TinyLlama.

<!-- Original repository: https://github.com/AmiriHayes/LLM-Interpretability -->

<!-- Preprint: https://www.overleaf.com/6482759765tsfvtgdxygym#4ec445 -->

## What This Contains

Three notebooks, one job each:

- `code/make_programs.ipynb`: Generates symbolic hypothesis programs for attention heads. (makes programs)
- `code/write_data.ipynb`: Generates IoU and interpolation CSV files. (scores programs)
- `code/all_experiments.ipynb`: Produces figures, best-fit mappings, and replacement experiments. (tests programs)

Supporting directories:

- `code/lib/`: Mechanism imported by the notebooks, not run directly — the synthesis
  engine (`program_synthesis.py`), the per-head attention substitution modules
  (`fixed_attention_*.py`), and the GPT-2 tokenizer shim.
- `code/archive/`: Superseded scripts, kept readable rather than deleted. Nothing
  imports them and none of the committed results depend on them.
- `data/`: Input assets, program libraries (`<model>_programs.py`), and score tables.
- `results/`: Best fits, plots, and replacement run outputs. `results/run_logs/`
  holds executed notebooks kept as provenance for committed results.

## Generating programs for a new model

`make_programs.ipynb` is model-agnostic — set `MODEL_ID` and it reads the layer and
head counts off the loaded model. For an unattended run of a large head grid:

```bash
python -m program_synthesis --model-id Qwen/Qwen3-4B-Base --model-key qwen3 \
       --strategy two_pass --workers 8        # run from code/lib/
```

Output is `data/<model_key>_programs.py`: one `prog_L{layer}H{head}` function per
attention head, self-contained, verified to execute before it is written.

## Quick Start

1. Run `code/write_data.ipynb` to generate/update data CSVs. (takes hours, data is included in repo)
2. Run `code/all_experiments.ipynb` to generate figures and experiment outputs.
3. Check outputs in `results/plots` and `results/replacement_run`.

## Notes

- Paths in notebooks are set relative to `code/` (for example, `../data`, `../results`).
- The notebooks attempt to use consistent logging tags: `[INFO]`, `[WARN]`, `[DONE]`.
