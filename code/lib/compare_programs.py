"""Score two program libraries against real attention and plot the comparison.

Mirrors write_data.ipynb's scoring (soft IoU, best-fit per head) so the numbers
are directly comparable to the committed iou_scores_*.csv / best_fits_*.csv.
"""
import argparse, importlib.util, inspect, io, json, re, contextlib
from pathlib import Path
import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def iou(p, q):
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, np.float64), 1e-12, 1.0)
    return float(np.minimum(p, q).sum() / np.maximum(p, q).sum())


def load_lib(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return {n: f for n, f in inspect.getmembers(m, inspect.isfunction)
            if re.search(r"_[Ll]\d+[Hh]\d+$", n)}


def real_attention(model, tok, sents, device):
    out = []
    for s in sents:
        enc = tok(s, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**enc, output_attentions=True)
        out.append((s, np.stack([a[0].float().cpu().numpy() for a in o.attentions], 0)))
    return out


def baselines(n, rng):
    lower = np.tril(np.ones((n, n))); lower /= lower.sum(1, keepdims=True)
    col = np.zeros((n, n)); col[:, 0] = 1.0
    rnd = np.tril(rng.random((n, n)) + 1e-9); rnd /= rnd.sum(1, keepdims=True)
    return {"lower_diagonal": lower, "first_column": col, "random_causal": rnd}


def score_library(progs, tok, real, n_layers, n_heads):
    """Per head: the best IoU any program in the library achieves. Also tracks
    how many programs actually execute, which the repo's bare excepts hide."""
    mats, broken = {}, {}
    for name, fn in progs.items():
        per = []
        for s, _ in real:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    _, m = fn(s, tok)
                per.append(np.asarray(m, np.float64))
            except Exception as e:
                broken[name] = f"{type(e).__name__}: {e}"; per = None; break
        if per is not None:
            mats[name] = per
    rows = []
    for l in range(n_layers):
        for h in range(n_heads):
            best, bestname = -1.0, None
            for name, per in mats.items():
                sc = np.mean([iou(real[i][1][l, h][:m.shape[0], :m.shape[0]], m)
                              for i, m in enumerate(per)])
                if sc > best: best, bestname = sc, name
            rows.append({"layer": l, "head": h, "program": bestname, "iou": best})
    return pd.DataFrame(rows), len(mats), broken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="gpt2")
    ap.add_argument("--old", default=str(ROOT / "data" / "gpt2_programs.py"))
    ap.add_argument("--new", default=str(ROOT / "data" / "gpt2new_programs.py"))
    ap.add_argument("--n-sent", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else ROOT / "code" / ".synthesis_gpt2new" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModel.from_pretrained(a.model_id, attn_implementation="eager").to(dev).eval()
    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads

    sents = json.loads((ROOT / "code" / ".synthesis_gpt2new" / "sentences.json").read_text())["sentences"]
    sents = sents[-a.n_sent:]          # held out from the 12 used for synthesis scoring
    real = real_attention(model, tok, sents, dev)
    print(f"{a.model_id}: {nL}x{nH} heads | {len(sents)} HELD-OUT sentences")

    rng = np.random.default_rng(0)
    base_rows = []
    for l in range(nL):
        for h in range(nH):
            for bname in ("lower_diagonal", "first_column", "random_causal"):
                sc = []
                for s, att in real:
                    n = att.shape[-1]
                    sc.append(iou(att[l, h], baselines(n, rng)[bname]))
                base_rows.append({"layer": l, "head": h, "baseline": bname, "iou": float(np.mean(sc))})
    base = pd.DataFrame(base_rows)

    results = {}
    for tag, path in (("old", a.old), ("new", a.new)):
        if not Path(path).exists():
            print(f"[skip] {tag}: {path} not found"); continue
        progs = load_lib(path)
        df, n_ok, broken = score_library(progs, tok, real, nL, nH)
        df.to_csv(out / f"best_fits_{tag}.csv", index=False)
        results[tag] = {"df": df, "n_progs": len(progs), "n_ok": n_ok, "broken": broken}
        print(f"[{tag:3s}] {len(progs)} programs, {n_ok} execute "
              f"({n_ok/max(len(progs),1):.0%}) | mean best-fit IoU = {df.iou.mean():.4f} "
              f"| median = {df.iou.median():.4f}")

    if len(results) < 2:
        print("need both libraries for the comparison figures"); return

    # ── Figure 3 style: distribution of per-head best-fit IoU ──────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [base[base.baseline == "random_causal"].iou.values,
              base[base.baseline == "first_column"].iou.values,
              base[base.baseline == "lower_diagonal"].iou.values,
              results["old"]["df"].iou.values, results["new"]["df"].iou.values]
    labels = ["random\ncausal", "first\ncolumn", "lower\ndiagonal",
              f"OLD\n({results['old']['n_ok']} progs)", f"NEW\n({results['new']['n_ok']} progs)"]
    bp = ax.boxplot(series, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#cccccc"]*3 + ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(c)
    ax.set_ylabel("best-fit IoU per head"); ax.set_ylim(0, 1)
    ax.set_title(f"{a.model_id}: per-head best-fit IoU — old vs new programs\n"
                 f"({len(sents)} held-out sentences)")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig3_boxplot.png", dpi=150); plt.close(fig)

    # ── Figure 4 style: per-head heatmaps + delta ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    grids = {t: results[t]["df"].pivot(index="layer", columns="head", values="iou").values
             for t in ("old", "new")}
    for ax, (t, title) in zip(axes[:2], [("old", "OLD gpt2_programs.py"),
                                         ("new", "NEW gpt2new_programs.py")]):
        im = ax.imshow(grids[t], cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{title}\nmean={grids[t].mean():.3f}")
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=.046)
    d = grids["new"] - grids["old"]
    lim = max(abs(d.min()), abs(d.max()))
    im = axes[2].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    axes[2].set_title(f"NEW - OLD\nmean Δ={d.mean():+.3f}  better on {(d>0).sum()}/{d.size} heads")
    axes[2].set_xlabel("head"); axes[2].set_ylabel("layer")
    fig.colorbar(im, ax=axes[2], fraction=.046)
    fig.tight_layout(); fig.savefig(out / "fig4_heatmaps.png", dpi=150); plt.close(fig)

    summary = {
        "model": a.model_id, "n_held_out_sentences": len(sents),
        "libraries": {t: {"n_programs": r["n_progs"], "n_executable": r["n_ok"],
                          "mean_best_fit_iou": round(float(r["df"].iou.mean()), 4),
                          "median_best_fit_iou": round(float(r["df"].iou.median()), 4),
                          "n_broken": len(r["broken"])} for t, r in results.items()},
        "baselines": {b: round(float(base[base.baseline == b].iou.mean()), 4)
                      for b in base.baseline.unique()},
        "new_better_on_heads": int((d > 0).sum()), "total_heads": int(d.size),
        "mean_delta": round(float(d.mean()), 4),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps(summary, indent=1))
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
