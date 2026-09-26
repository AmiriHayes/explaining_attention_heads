"""Download Joseph Bloom's gpt2-small residual-stream SAE suite from
jbloom/GPT2-Small-SAEs-Reformatted on HuggingFace Hub.

Not using the `sae_lens` package -- it requires Python >=3.10 and this
environment has 3.9. The architecture is a plain ReLU SAE (two linear layers),
simple enough to reimplement directly against the raw safetensors weights:

    f = ReLU((x - b_dec) @ W_enc + b_enc)      # encode: (768,) -> (24576,)
    x_hat = f @ W_dec + b_dec                  # decode: (24576,) -> (768,)

Checkpoints: blocks.{L}.hook_resid_pre for L=0..11 (SAE input = residual stream
entering layer L, i.e. our "L{L}"), plus blocks.11.hook_resid_post (residual
stream leaving the last block, i.e. our "L12"). 13 total, matching the 13
hidden_states checkpoints saved by extract_layers.py.

Usage:
    python download_sae_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
THIS_DIR = Path(__file__).resolve().parent
SAE_DIR = THIS_DIR / "data" / "sae"

# Our layer index L -> the checkpoint name in the repo. L=0..11 are the
# resid_pre entering each block; L=12 is resid_post of the last block.
CHECKPOINTS = {L: f"blocks.{L}.hook_resid_pre" for L in range(12)}
CHECKPOINTS[12] = "blocks.11.hook_resid_post"


def download_one(layer: int, checkpoint: str) -> None:
    out_dir = SAE_DIR / f"L{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = hf_hub_download(REPO, f"{checkpoint}/cfg.json")
    weights_path = hf_hub_download(REPO, f"{checkpoint}/sae_weights.safetensors")

    cfg = json.loads(Path(cfg_path).read_text())
    weights = load_file(weights_path)

    (out_dir / "cfg.json").write_text(json.dumps(cfg, indent=2))
    for name, arr in weights.items():
        # re-save into our own dir so this doesn't depend on the HF cache layout
        import numpy as np

        np.save(out_dir / f"{name}.npy", arr)

    print(f"  L{layer} ({checkpoint}): d_in={cfg['d_in']} d_sae={cfg['d_sae']} -> {out_dir}")


def main() -> None:
    SAE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(CHECKPOINTS)} SAE checkpoints from {REPO}...")
    for layer, checkpoint in sorted(CHECKPOINTS.items()):
        download_one(layer, checkpoint)
    print("Done.")


if __name__ == "__main__":
    main()
