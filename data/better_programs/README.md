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

Both the table and the figure predate the substitutions below, so the nine
substituted heads are still counted at 0. The two that could be rescored
locally went from 0.000 to 0.546 (GPT-2 L3H3) and 0.582 (Llama-3.2-3B L12H6),
so these medians understate the libraries slightly.

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

## Substituted programs

Nine programs raised on at least one of the 500 sentences in `splits.json`
(200 prompt pool + 100 validation + 200 test). Each has been replaced by the
paper library's program for the same head, copied verbatim. **Every program in
this folder now runs on all 500 sentences.**

The preferred replacement was the other candidate from the same synthesis run
(initial vs refined), which keeps the provenance uniform. It was not usable for
any of these: the GPT-2 run produced no second candidate for L3H3
(`program_refined` is null), and the per-head records for the other three models
live on the cluster, which was unreachable. All nine therefore fall back to the
paper library.

| model | head | replaced program | failed on | replaced by |
|---|---|---|---|---|
| GPT-2 | L3H3 | `prog_L3H3` | 500/500 | `first_token_bias_content_focus_punctuation_L3H3` |
| TinyLlama | L17H28 | `prog_L17H28` | 500/500 | `prog_L17H28` |
| TinyLlama | L18H26 | `prog_L18H26` | 500/500 | `prog_L18H26` |
| Llama-3.2-3B | L12H6 | `prog_L12H6` | 500/500 | `prog_L12H6` |
| Qwen3-8B | L12H7 | `prog_L12H7` | 500/500 | `prog_L12H7` |
| Qwen3-8B | L14H10 | `prog_L14H10` | 500/500 | `prog_L14H10` |
| Qwen3-8B | L17H3 | `prog_L17H3` | 500/500 | `prog_L17H3` |
| Qwen3-8B | L22H18 | `prog_L22H18` | 500/500 | `prog_L22H18` |
| Qwen3-8B | L32H21 | `prog_L32H21` | 470/500 | `prog_L32H21` |

Each substituted function carries a comment saying what it replaced and why.
No program's logic was edited. Three supporting changes were needed:

- `gpt2.py`, `tinyllama.py`, `llama3b.py`: the helpers `_align_to_spacy` (all
  three) and `_char_sim` (TinyLlama) are inlined verbatim from the paper
  libraries, because the substituted programs call them.
- `llama3b.py`: `embedding_similarity`'s `tokenizer` argument is now optional,
  falling back to a lazily-loaded tokenizer for the model under study. The
  substituted program calls it with the paper library's three-argument
  signature. Every program written by the synthesis run passes the argument
  explicitly, so their behaviour is unchanged.

## Known issues

Two programs run on all 500 sentences but still score 0.000 on the test set:
Llama-3.2-3B `prog_L13H22` and Qwen3-8B `prog_L27H15`. They do not raise, so
they were left alone.

Llama-3.2-3B's **paper** library calls two helpers it never defines, so 346 of
its 672 programs crash; the paper column above scores it with reconstructed
helpers. That does not affect the Llama-3.2-3B programs here.

## Interactive viewer

https://huggingface.co/spaces/AmiriHayes/attention-head-atlas

## Provenance

All four were synthesized with `gpt-6.1-sol`, `reasoning_effort=medium`,
`max_tokens=32000`, seed 1, 20 prompt sentences, `top_pct` evidence at 2.5%.

| file | origin | generated | synthesis target | sha256 |
|---|---|---|---|---|
| `gpt2.py` | copied from `llama_programs_v3/ab_compare/attn/gpt2_intended_programs.py` | 2026-10-17 | `gpt2` | `1ef8763dcee119fcadf02ac054e54000ba9645afb409874cac8d8f8a52bb7a56` |
| `tinyllama.py` | written here by the cluster run | 2026-10-24 | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `0a7fe7f2bb5b8e378eb83900fcbe869d96681a91dccf05872960a3b57a7a9036` |
| `llama3b.py` | written here by the cluster run | 2026-10-24 | `meta-llama/Llama-3.2-3B` | `80f3149bc940c31c59bde2a8c1c4ca65047f1aecd2453096eb7af6d12bb2ac91` |
| `qwen3.py` | written here by the cluster run | 2026-10-24 | `Qwen/Qwen3-8B-Base` | `705bc2ed4b6d9fbb4af6055e125dff328848db5cb5d091ca0846e2cbe3b814db` |
| `program_quality_four_models.png` | copied from `results/multimodel/multimodel_program_quality.png` | 2026-10-24 | — | `269a8d058a94b2e8ac7611263035480af657d382f9e0dbdc234963c6d61db2e7` |

The sha256 above is of the file as it now stands, after the substitutions. It
therefore no longer matches the hash recorded in each `.provenance.json`, which
describes the file as the synthesis run wrote it. The `.iou.json` scores are
likewise from that run, so the nine substituted heads are scored there at 0,
under the program that was replaced. `gpt2.py`
differs from the file it was copied from in one place: the header docstring
originally said it was generated by `make_programs.ipynb`, which is wrong. Every
program body below that docstring is byte-identical. `gpt2.py` has no
`.provenance.json`, because it predates the cluster runs that wrote those files.
