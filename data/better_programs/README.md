# better_programs

One attention program per head, for GPT-2, TinyLlama, Llama-3.2-3B and Qwen3-8B.
A program takes a sentence and a tokenizer and returns a predicted attention
matrix for that head.

These were written by the same synthesis pipeline as the paper's libraries, with
two changes:

- the synthesis model is **gpt-6.1-sol** at **medium** reasoning effort, and
- each prompt shows **20 example sentences**.

Everything else — the prompt structure, the two-pass refine-and-select protocol,
the helper functions the programs may call, and the held-out splits — is unchanged.

## How these compare to the paper's libraries

| model | heads | median test IoU, these | median test IoU, paper |
|---|---|---|---|
| GPT-2 | 144 | **0.716** | 0.504 |
| TinyLlama | 704 | **0.759** | 0.419 |
| Llama-3.2-3B | 672 | **0.801** | 0.523 |
| Qwen3-8B | 1152 | **0.674** | 0.417 |

IoU is soft IoU (elementwise min over elementwise max) between the predicted and
the real attention matrix, averaged over 200 held-out TinyStories sentences that
appear in neither the prompts nor the refinement feedback.

See `program_quality_four_models.png` for the full distributions.

## Evidence format

On GPT-2 we also tried showing the model the full attention matrix instead of
only the strongest 2.5% of links. It raised mean test IoU by about 0.027, mostly
on heads that don't attend to the first token, at roughly 2.6x the cost. That
wasn't worth it, so these libraries use the 2.5% format.

## Files

- `<model>.py` — the programs, one function per head, named `*_L<layer>H<head>`.
- `<model>.iou.json` — per-head scores from the synthesis run: test IoU on the
  200 held-out sentences, validation IoU, same-data IoU, and which of the two
  passes (`initial` or `refined`) was selected.
- `<model>.provenance.json` — the exact settings that produced the file: model
  id and revision, synthesis model, reasoning effort, seed, sentence count,
  evidence format, splits fingerprint, program count, date and sha256.
- `prompt_sentences.txt` — the 20 example sentences shown in every synthesis
  prompt (seed 1), one per line, verbatim.
- `program_quality_four_models.png` — the boxplot behind the table above.

## Known issues

**These programs raise and need fixing.** The list below comes from a partial
scan — every program in the four files was run on all 500 sentences in
`splits.json` (200 prompt pool + 100 validation + 200 test), but the scan was
stopped before the replacements were made. These will be fixed later.

| model | program | sentences that fail | exception |
|---|---|---|---|
| GPT-2 | `prog_L3H3` | 500/500 | `ValueError: could not broadcast input array from shape (2,) into shape (19,)` |
| TinyLlama | `prog_L17H28` | 500/500 | `ValueError: could not broadcast input array from shape (2,) into shape (20,)` |
| TinyLlama | `prog_L18H26` | 500/500 | `ValueError: could not broadcast input array from shape (2,) into shape (19,)` |
| Llama-3.2-3B | `prog_L12H6` | 500/500 | `ValueError: operands could not be broadcast together with shapes (19,) (2,) (19,)` |
| Qwen3-8B | `prog_L12H7` | 500/500 | `ValueError: could not broadcast input array from shape (3,) into shape (19,)` |
| Qwen3-8B | `prog_L14H10` | 500/500 | `ValueError: could not broadcast input array from shape (2,) into shape (19,)` |
| Qwen3-8B | `prog_L17H3` | 500/500 | `ValueError: could not broadcast input array from shape (2,) into shape (19,)` |
| Qwen3-8B | `prog_L22H18` | 500/500 | `ValueError: could not broadcast input array from shape (2,) into shape (19,)` |
| Qwen3-8B | `prog_L32H21` | 470/500 | `ZeroDivisionError: division by zero` |

A program that raises is scored 0, so these also drag down the medians above.

Two more programs run on all 500 sentences but still score 0.000 on the test
set: Llama-3.2-3B `prog_L13H22` and Qwen3-8B `prog_L27H15`.

Separately, Llama-3.2-3B's **paper** library calls two helpers it never defines,
so 346 of its 672 programs crash; the paper column above scores it with
reconstructed helpers. That does not affect the Llama-3.2-3B programs here.

## Interactive viewer

https://huggingface.co/spaces/AmiriHayes/attention-head-atlas

## Provenance

All four were synthesized with `gpt-6.1-sol`, `reasoning_effort=medium`,
`max_tokens=32000`, seed 1, 20 prompt sentences, `top_pct` evidence at 2.5%.

| file | origin | generated | synthesis target | sha256 |
|---|---|---|---|---|
| `gpt2.py` | copied from `llama_programs_v3/ab_compare/attn/gpt2_intended_programs.py` | 2026-10-17 | `gpt2` | `fedf4d8f4660f9972de145051d5fa00c8f3d10587e14a18aef940e8627768f2c` |
| `tinyllama.py` | written here by the cluster run | 2026-10-24 | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `d013b65390d1bf50d70670eed3e6da8b748255f2a42f4f89ff69bf1cb83f97dd` |
| `llama3b.py` | written here by the cluster run | 2026-10-24 | `meta-llama/Llama-3.2-3B` | `6c34121a2da97a996cb541b52ead78e3e8bd885bacefefeae18e9fe427cc2601` |
| `qwen3.py` | written here by the cluster run | 2026-10-24 | `Qwen/Qwen3-8B-Base` | `940f59ae791c404ec35b614d6837acd47d54503e446601a89ad3443e316b293d` |
| `program_quality_four_models.png` | copied from `results/multimodel/multimodel_program_quality.png` | 2026-10-24 | — | `269a8d058a94b2e8ac7611263035480af657d382f9e0dbdc234963c6d61db2e7` |

Each `.py` hash matches the one recorded in its `.provenance.json`. `gpt2.py`
differs from the file it was copied from in one place: the header docstring
originally said it was generated by `make_programs.ipynb`, which is wrong. Every
program body below that docstring is byte-identical. `gpt2.py` has no
`.provenance.json`, because it predates the cluster runs that wrote those files.
