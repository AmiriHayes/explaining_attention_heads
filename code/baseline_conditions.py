"""Assignment-control baselines for the head-replacement experiment.

Six conditions, all sharing one harness and the attention-weight-level
replacement semantics from fixed_attention_gpt2.py (apply that patch first):

  correct          each head gets its own best-fit program (deterministic)
  within_category  heads exchange programs within their functional category
  permute_layer    heads exchange programs within their layer
  permute_global   cyclic permutation of the assignment vector across heads
                   (heads sharing a best-fit program can still get their own)
  random           uniform draw from the library, excluding the head's own
  cross_category   genuine best-fit program from a DIFFERENT category

Usage:
  python code/baseline_conditions.py --modes permute_global --seeds 0 --smoke
  python code/baseline_conditions.py --modes correct within_category \
      permute_layer permute_global random cross_category --seeds 0 1 2 3 4
"""

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent          # code/
REPO = HERE.parent                     # repo root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "data"))

import numpy as np
import pandas as pd
import torch

import plot_random_baseline_figures as base  # the repo's harness module
import fixed_attention_gpt2 as fa  # corrected attention-weight substitution (patch 0001)

# The repo module hardcodes a (revoked) HF token via os.environ.setdefault —
# sending it turns anonymous public downloads into 401s. Remove it.
import os

os.environ.pop("HF_TOKEN", None)


def sattolo(items: list, rng: random.Random) -> list:
    """Cyclic permutation with no fixed points (guaranteed derangement)."""
    a = list(items)
    for i in range(len(a) - 1, 0, -1):
        j = rng.randrange(i)
        a[i], a[j] = a[j], a[i]
    return a


def build_assignment(smart_df, mode: str, rng: random.Random) -> tuple[dict, dict]:
    """head_tuple -> program name under the given permutation mode."""
    heads = [base.parse_head(h) for h in smart_df["head"]]
    intended = list(smart_df["program"])
    notes = {}
    if mode == "permute_global":
        permuted = sattolo(intended, rng)
        return dict(zip(heads, permuted)), notes
    if mode == "cross_category":
        import json
        cats = json.loads((REPO / "results" / "gpt2_program_categories.json").read_text())
        prog2cat = {pn: c for c, plist in cats.items() for pn in plist}
        bestfit_pool = sorted(set(intended))
        assignment = {}
        for idx, ht in enumerate(heads):
            own_cat = prog2cat[intended[idx]]
            pool = [pn for pn in bestfit_pool if prog2cat[pn] != own_cat]
            assignment[ht] = rng.choice(pool)
        notes["taxonomy"] = {c: len(v) for c, v in cats.items()}
        return assignment, notes
    if mode == "within_category":
        import json
        cats = json.loads((REPO / "results" / "gpt2_program_categories.json").read_text())
        prog2cat = {pn: c for c, plist in cats.items() for pn in plist}
        by_cat: dict[str, list[int]] = {}
        for idx in range(len(heads)):
            by_cat.setdefault(prog2cat[intended[idx]], []).append(idx)
        assignment = {}
        for cat, idxs in by_cat.items():
            perm = sattolo([intended[i] for i in idxs], rng) if len(idxs) >= 2 else [intended[i] for i in idxs]
            for i, pnew in zip(idxs, perm):
                assignment[heads[i]] = pnew
        notes["group_sizes"] = {c: len(v) for c, v in by_cat.items()}
        return assignment, notes
    if mode == "permute_layer":
        by_layer: dict[int, list[int]] = {}
        for idx, ht in enumerate(heads):
            by_layer.setdefault(ht[0], []).append(idx)
        assignment = {}
        singletons = []
        for layer, idxs in by_layer.items():
            if len(idxs) >= 2:
                perm = sattolo([intended[i] for i in idxs], rng)
                for i, pnew in zip(idxs, perm):
                    assignment[heads[i]] = pnew
            else:
                singletons.extend(idxs)
        if len(singletons) >= 2:
            perm = sattolo([intended[i] for i in singletons], rng)
            for i, pnew in zip(singletons, perm):
                assignment[heads[i]] = pnew
            notes["singleton_pool"] = [str(heads[i]) for i in singletons]
        elif len(singletons) == 1:
            i = singletons[0]
            others = [p for p in intended if p != intended[i]]
            assignment[heads[i]] = rng.choice(others)
            notes["singleton_random"] = str(heads[i])
        return assignment, notes
    raise ValueError(mode)


def run_gpt2(seed: int, mode: str, eval_sents, smart_df, programs,
             semantics: str = "compose") -> pd.DataFrame:
    """Copy of the repo's run_gpt2_random_seed with the assignment injected."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    prog_lookup = {p.__name__: p for p in programs}
    all_names = list(prog_lookup.keys())
    rng = random.Random(seed)

    if mode in ("permute_global", "permute_layer", "cross_category", "within_category"):
        fixed_assignment, notes = build_assignment(smart_df, mode, rng)
    elif mode == "smart_protocol":
        fixed_assignment = {base.parse_head(r.head): r.program
                            for r in smart_df.itertuples()}
        notes = {"protocol": "greedy skip if marginal>50pp; last-word-truncated sentences"}
    elif mode == "correct":
        fixed_assignment = {base.parse_head(r.head): r.program
                            for r in smart_df.itertuples()}
        notes = {}
    else:
        fixed_assignment, notes = None, {}

    DEVICE = base.DEVICE
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    N_HEADS = 12
    N_LAYERS = 12
    shared_smart = {}
    active_sent = [None]
    mat_cache = {}

    def get_mat(prog_name, sentence):
        key = (prog_name, sentence)
        if key not in mat_cache:
            p = prog_lookup.get(prog_name)
            try:
                mat_cache[key] = p(sentence, tok)[1] if p else None
            except Exception:  # noqa: BLE001 — repo programs raise arbitrarily; null = skip head (their convention)
                mat_cache[key] = None
        return mat_cache[key]

    def make_hook(li):
        def hook(module, inp, out):
            ctx = out[0]
            b, s, hd = ctx.shape
            active = [(li, hi) for hi in range(N_HEADS) if (li, hi) in shared_smart]
            if not active:
                return out
            mod = ctx.view(b, s, N_HEADS, hd // N_HEADS).clone()
            cur = active_sent[0]
            for _, hi in active:
                mat = get_mat(shared_smart[(li, hi)], cur)
                if mat is not None:
                    t = torch.tensor(mat, device=DEVICE, dtype=mod.dtype)
                    if t.shape[0] == s:
                        mod[:, :, hi, :] = torch.matmul(t, mod[:, :, hi, :])
            return (mod.view(b, s, hd),) + out[1:]
        return hook

    if semantics == "substitute":
        def raw_mat(prog_name, sentence):
            key = ("raw", prog_name, sentence)
            if key not in mat_cache:
                pr = prog_lookup.get(prog_name)
                try:
                    mat_cache[key] = pr(sentence, tok)[1] if pr else None
                except Exception:  # noqa: BLE001 — repo programs raise arbitrarily; None = keep own attention
                    mat_cache[key] = None
            m = mat_cache[key]
            import numpy as _np
            return None if m is None else _np.asarray(m)

        state = {"assignment": shared_smart, "pattern": raw_mat,
                 "sentence": active_sent}
        ident = fa.verify_identity(model, tok, eval_sents[0], DEVICE)
        assert ident < 1e-3, f"identity check failed: {ident}"
        restore = fa.install(model, state)
        handles = []
    else:
        handles = [model.transformer.h[li].attn.register_forward_hook(make_hook(li))
                   for li in range(N_LAYERS)]

    def ppl(sent):
        active_sent[0] = sent
        toks = tok(sent, return_tensors="pt", truncation=True,
                   max_length=base.MAX_SEQ_LEN).to(DEVICE)
        with torch.no_grad():
            return torch.exp(model(**toks, labels=toks["input_ids"]).loss).item()

    baseline_ppls = [ppl(s) for s in eval_sents]

    def eval_pct():
        ppls = [ppl(s) for s in eval_sents]
        return float(np.mean([(r - b) / b * 100 for r, b in zip(ppls, baseline_ppls)]))

    if mode == "smart_protocol":
        eval_sents = [" ".join(s.split()[:-1]) if len(s.split()) > 3 else s
                      for s in eval_sents]
        baseline_ppls[:] = [ppl(s) for s in eval_sents]

    results = []
    prev_pct = 0.0
    for step, row in enumerate(smart_df.itertuples(), start=1):
        ht = base.parse_head(row.head)
        intended = row.program
        if fixed_assignment is not None:
            chosen = fixed_assignment[ht]
        else:
            chosen = rng.choice([n for n in all_names if n != intended])
        shared_smart[ht] = chosen
        pct = eval_pct()
        if mode == "smart_protocol" and (pct - prev_pct) > 50.0:
            del shared_smart[ht]  # their greedy skip: revert this head
            continue
        prev_pct = pct
        results.append({"k": step, "head": row.head, "program": chosen,
                        "intended": intended, "increase": pct, "mode": mode,
                        "seed": seed})
        if step % 20 == 0 or step == len(smart_df):
            print(f"  [{mode} seed={seed}] k={step}/{len(smart_df)} "
                  f"pct={pct:.1f}%", flush=True)

    for h in handles:
        h.remove()
    if semantics == "substitute":
        restore()
    del model
    df = pd.DataFrame(results)
    if notes:
        df.attrs["notes"] = notes
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=["gpt2"])
    ap.add_argument("--modes", nargs="+",
                    default=["permute_global", "permute_layer"],
                    choices=["random", "permute_global", "permute_layer",
                             "cross_category", "correct", "smart_protocol", "within_category"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--semantics", default="substitute",
                    choices=["compose", "substitute"])
    ap.add_argument("--smoke", action="store_true",
                    help="25-head, 1-condition quick check (~4 min GPU / ~15 min CPU)")
    args = ap.parse_args()

    outdir = REPO / "results" / "replacement_run" / (
        "permutations_fixed" if args.semantics == "substitute" else "permutations")
    outdir.mkdir(parents=True, exist_ok=True)

    eval_sents = base.load_eval_sentences()
    programs = base.load_programs(args.model)
    smart_df = pd.read_csv(REPO / "results" / "replacement_run"
                           / f"{args.model}_smart.csv")
    if args.smoke:
        smart_df = smart_df.head(25)
    print(f"{args.model}: {len(smart_df)} heads, {len(eval_sents)} eval "
          f"sentences, {len(programs)} programs, device={base.DEVICE}")

    for mode in args.modes:
        for seed in args.seeds:
            out = outdir / f"{args.model}_{mode}_seed{seed}.csv"
            if out.exists():
                print(f"skip existing {out.name}")
                continue
            df = run_gpt2(seed, mode, eval_sents, smart_df, programs,
                          semantics=args.semantics)
            df.to_csv(out, index=False)
            print(f"wrote {out}")
    print("ALL_RUNS_DONE")


if __name__ == "__main__":
    main()
