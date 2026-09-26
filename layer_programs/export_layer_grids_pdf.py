"""Export the "Grid across layers, one sentence" plot (raw values, L0-L12 stacked,
shared per-sentence percentile-clipped scale, same as visualize_layers.ipynb) for
every sentence in the dataset, one page per sentence, into a single multi-page PDF.

Usage:
    python export_layer_grids_pdf.py [--out outputs/layer_grids_all_sentences.pdf]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"  # inputs
ACTIVATIONS_DIR = DATA_DIR / "activations"
OUTPUTS_DIR = THIS_DIR / "outputs"  # findings: this script's rendered PDF


def plot_layer_heatmap(data_2d, title, ax, vmin, vmax):
    """Same normalization/style as plot_layer_heatmap in visualize_layers.ipynb:
    gamma=0.5 PowerNorm, Blues, blank axes, bold left-aligned title."""
    norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    sns.heatmap(
        data_2d,
        ax=ax,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        norm=norm,
    )
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12, loc="left")


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads((DATA_DIR / "index.json").read_text())
    sentences = index["sentences"]
    layers_to_show = list(range(13))

    with PdfPages(out_path) as pdf:
        for i, entry in enumerate(sentences):
            npz = np.load(ACTIVATIONS_DIR / entry["file"], allow_pickle=True)
            hidden_states = npz["hidden_states"]  # (13, seq_len, 768)
            sentence_preview = str(npz["sentence"])[:70]

            # per-sentence shared scale across its own 13 layers, matching the notebook
            pooled = np.concatenate([hidden_states[L].ravel() for L in layers_to_show])
            vmin, vmax = np.percentile(pooled, 1), np.percentile(pooled, 99)

            fig, axes = plt.subplots(len(layers_to_show), 1, figsize=(20, 3 * len(layers_to_show)))
            for ax, L in zip(axes, layers_to_show):
                plot_layer_heatmap(
                    hidden_states[L],
                    title=f'S{entry["id"]} L{L}: "{sentence_preview}"',
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(sentences)} pages done")

    print(f"Saved {len(sentences)}-page PDF to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "layer_grids_all_sentences.pdf"))
    args = parser.parse_args()
    main(Path(args.out))
