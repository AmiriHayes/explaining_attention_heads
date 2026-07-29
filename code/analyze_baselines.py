"""Analysis + figure for the assignment-control baseline conditions.

Reads the per-condition result CSVs written by baseline_conditions.py, joins
pattern-fidelity from data/iou_scores_{model}.csv (per head x program IoU on
held-out sentences), and produces:
  - a summary table: perplexity increase at 25%/50%/100% of heads, area under
    the log-scale trajectory, and mean assigned-program IoU per condition
  - a three-panel figure: cost curves (log), fidelity-vs-cost, trajectory areas

Usage:
  python code/analyze_baselines.py            # after runs exist
  python code/analyze_baselines.py --out results/plots/baselines.png
"""

import argparse
import ast
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REPO = HERE.parent
RESULTS = REPO / "results" / "replacement_run" / "permutations_fixed"

CONDS = {
    "correct":         ("#1f77b4", "own best-fit program (deterministic)"),
    "within_category": ("#8e44ad", "another head's program, same category"),
    "permute_layer":   ("#e67e22", "another head's program, same layer"),
    "permute_global":  ("#c0392b", "another head's program, any"),
    "random":          ("#7f8c8d", "random library program"),
    "cross_category":  ("#27ae60", "genuine program, DIFFERENT category"),
}


def iou_lookup(model="gpt2"):
    d = pd.read_csv(REPO / "data" / f"iou_scores_{model}.csv")
    agg = d.groupby(["layer", "head", "program_idx"])["iou_score"].mean()
    import importlib, sys
    sys.path.insert(0, str(REPO / "data"))
    progs = importlib.import_module(f"{model}_programs").all_programs
    names = {i: p.__name__ for i, p in enumerate(progs)}
    out = {}
    for (layer, head, pi), v in agg.items():
        out[((int(layer), int(head)), names.get(int(pi), ""))] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default=str(REPO / "results" / "plots" / "baselines.png"))
    args = ap.parse_args()

    iou = iou_lookup(args.model)
    x = None
    rows, data = [], {}
    for cond, (color, label) in CONDS.items():
        files = sorted(RESULTS.glob(f"{args.model}_{cond}_seed*.csv"))
        if not files:
            continue
        runs = [pd.read_csv(f) for f in files]
        arr = np.vstack([r["increase"].values for r in runs])
        n = arr.shape[1]
        x = np.arange(1, n + 1) / n
        la = [float(np.trapezoid(np.log10(1 + np.maximum(r, -0.99)), x))
              for r in arr]
        mean_iou = float(np.mean([
            iou.get((ast.literal_eval(str(rr.head)), rr.program), np.nan)
            for r in runs for rr in r.itertuples()]))
        data[cond] = (arr, np.mean(la), np.std(la))
        q = lambda i: f"{arr[:, i].mean():.1f}±{arr[:, i].std():.1f}" \
            if len(runs) > 1 else f"{arr[0, i]:.1f}"
        rows.append({"condition": label, "25%": q(n // 4 - 1),
                     "50%": q(n // 2 - 1), "100%": q(n - 1),
                     "log_AUC": f"{np.mean(la):.2f}±{np.std(la):.2f}",
                     "mean_IoU": f"{mean_iou:.3f}", "seeds": len(runs)})
    print(pd.DataFrame(rows).to_string(index=False))

    fig = plt.figure(figsize=(15.5, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1])
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for cond, (color, label) in CONDS.items():
        if cond not in data:
            continue
        arr, _, _ = data[cond]
        mu, sd = arr.mean(0), arr.std(0)
        ax1.plot(x * 100, np.maximum(mu, 1e-2), color=color, lw=1.9, label=label)
        if arr.shape[0] > 1:
            ax1.fill_between(x * 100, np.maximum(mu - sd, 1e-2), mu + sd,
                             color=color, alpha=0.15)
    ax1.set_yscale("log"); ax1.grid(alpha=0.25)
    ax1.set_xlabel("% of heads replaced (fit-quality order)")
    ax1.set_ylabel("perplexity increase (%, log)")
    ax1.legend(fontsize=7.2, loc="lower right")
    for cond, (color, label) in CONDS.items():
        if cond not in data:
            continue
        arr, _, _ = data[cond]
        n = arr.shape[1]
        c25 = arr[:, n // 4 - 1]
        mi = np.nanmean([iou.get((ast.literal_eval(str(rr.head)), rr.program), np.nan)
                         for f in sorted(RESULTS.glob(f"{args.model}_{cond}_seed*.csv"))
                         for rr in pd.read_csv(f).itertuples()])
        ax2.errorbar(mi, max(c25.mean(), 1e-1), yerr=c25.std(), fmt="o",
                     color=color, ms=8, capsize=3)
    ax2.set_yscale("log"); ax2.grid(alpha=0.25)
    ax2.set_xlabel("mean IoU of applied pattern vs head's actual attention")
    ax2.set_ylabel("increase at 25% replaced (%, log)")
    order = sorted(data, key=lambda c: data[c][1])
    for i, cond in enumerate(order):
        _, m, s = data[cond]
        ax3.barh(i, m, xerr=s if s > 0 else None, color=CONDS[cond][0],
                 alpha=1.0 if cond == "correct" else 0.75,
                 edgecolor="k" if cond == "correct" else "none",
                 linewidth=1.6 if cond == "correct" else 0, capsize=3)
    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels([CONDS[c][1] for c in order], fontsize=7.4)
    ax3.set_xlabel("mean of log10(1 + % increase) over trajectory")
    ax3.grid(alpha=0.25, axis="x")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
