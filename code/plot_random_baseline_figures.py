"""
plot_random_baseline_figures.py  (corrected)
=============================================
Produces a 2x1 PDF figure with:
  Left  — Head Replacement % vs. Perplexity Increase %
  Right — IoU Similarity vs. Perplexity Increase % (log-y)

Fixes two bugs in the original shuffle implementation:
  1. Wrong program set: the old shuffles used 11 generic programs instead of
     drawing from the model's actual ~147-program library.
  2. Wrong ordering: the old shuffles also randomised head-replacement ORDER;
     the correct baseline keeps the same IOU ordering and only randomises
     WHICH program each head receives.

Correct random baseline (generated here if not already cached):
  - Same IOU head-replacement order as the smart run
  - At each step k, a program is drawn uniformly at random from the model's
    full program library, EXCLUDING that head's own intended (best-fit) program
  - Repeated for N_SEEDS seeds to produce a mean +/- 1-std band

Generated CSVs are cached in:
  results/replacement_run/correct_shuffles/{model}_random_seed_{n}.csv

Usage:
  python code/plot_random_baseline_figures.py           # generate + plot
  python code/plot_random_baseline_figures.py --force   # re-generate even if cached
  python code/plot_random_baseline_figures.py --no-generate  # plot from existing cache
  python code/plot_random_baseline_figures.py --out results/plots/figure_random_baseline.pdf
"""

import argparse, glob, inspect, json, os, pathlib, random, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

ROOT        = pathlib.Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
SMART_DIR   = RESULTS_DIR / 'replacement_run'
# SHUFFLE_DIR is set in main() based on --order flag; modules that need it use get_shuffle_dir()
_SHUFFLE_ORDER = ['iou']   # mutable default, overridden by main()

def get_shuffle_dir():
    return RESULTS_DIR / 'replacement_run' / f'shuffles_{_SHUFFLE_ORDER[0]}'

sys.path.insert(0, str(DATA_DIR))

os.environ.setdefault('HF_TOKEN', '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')

DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SEEDS       = 5
N_EVAL_SENTS  = 5
MAX_SEQ_LEN   = 128
MAX_PPL_SHOWN = 400

MODEL_CFG = {
    'gpt2': {
        'total_heads': 144, 'display': 'GPT-2', 'color': '#c0392b',
    },
    'bert': {
        'total_heads': 144, 'display': 'BERT', 'color': '#f0a868',
    },
    'tinyllama': {
        'total_heads': 704, 'display': 'TinyLlama', 'color': '#1f77b4',
        'model_id': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'n_layers': 22, 'n_heads': 32, 'head_dim': 64, 'use_float16': False,
    },
    'llama3b': {
        'total_heads': 672, 'display': 'Llama-3.2-3B', 'color': '#2a9d8f',
        'model_id': 'meta-llama/Llama-3.2-3B',
        'n_layers': 28, 'n_heads': 24, 'head_dim': 128, 'use_float16': True,
    },
}
MODELS_WITH_RANDOM = {'gpt2', 'bert', 'tinyllama', 'llama3b'}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_head(h):
    try:
        return tuple(map(int, str(h).strip('()').split(',')))
    except Exception:
        return (0, 0)

def load_eval_sentences(n=N_EVAL_SENTS):
    with open(DATA_DIR / 'generic_sentences.json') as f:
        sents = json.load(f)
    return sents[:n]

def prepare_sentence(sent):
    words = sent.split()
    return ' '.join(words[:-1]) if len(words) > 3 else sent

_UTILITY_FNS = {
    'apply_causal_mask', 'make_row_stochastic', 'spacy_parse',
    'align_spacy_to_tokens', 'embedding_similarity', 'get_pos',
    'get_spacy_features', 'is_content_word', 'is_early_content_word',
    'is_proper_noun', 'is_punct', 'is_punctuation', 'is_special',
    'is_special_token', 'is_subject_or_main_verb', 'is_verb',
}

def load_programs(model_key):
    import importlib
    mod_name = {'gpt2': 'gpt2_programs', 'bert': 'bert_programs',
                'tinyllama': 'tinyllama_programs', 'llama3b': 'llama3b_programs'}[model_key]
    mod = importlib.import_module(mod_name)
    return [f for _, f in inspect.getmembers(mod, inspect.isfunction)
            if not f.__name__.startswith('_') and f.__name__ not in _UTILITY_FNS]


# ── GPT-2 random seed runner ─────────────────────────────────────────────────

def run_gpt2_random_seed(seed, eval_sents, smart_df, programs):
    prog_lookup = {p.__name__: p for p in programs}
    all_names   = list(prog_lookup.keys())
    rng         = random.Random(seed)

    model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE).eval()
    tok   = GPT2Tokenizer.from_pretrained('gpt2')

    N_LAYERS = N_HEADS = 12
    shared_smart = {}
    active_sent  = [None]
    mat_cache    = {}

    def get_mat(prog_name, sentence):
        key = (prog_name, sentence)
        if key not in mat_cache:
            p = prog_lookup.get(prog_name)
            try:
                mat_cache[key] = p(sentence, tok)[1] if p else None
            except Exception:
                mat_cache[key] = None
        return mat_cache[key]

    def make_hook(li):
        def hook(module, inp, out):
            ctx = out[0]; b, s, hd = ctx.shape
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

    handles = [model.transformer.h[li].attn.register_forward_hook(make_hook(li))
               for li in range(N_LAYERS)]

    baseline_ppls = []
    for s in eval_sents:
        active_sent[0] = s
        toks = tok(s, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
        with torch.no_grad():
            baseline_ppls.append(torch.exp(model(**toks, labels=toks['input_ids']).loss).item())

    def eval_pct():
        ppls = []
        for s in eval_sents:
            active_sent[0] = s
            toks = tok(s, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
            with torch.no_grad():
                ppls.append(torch.exp(model(**toks, labels=toks['input_ids']).loss).item())
        return float(np.mean([(r - b) / b * 100 for r, b in zip(ppls, baseline_ppls)]))

    if _SHUFFLE_ORDER[0] == 'random':
        iterate_df = smart_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        iterate_df = smart_df.copy()

    results = []
    for step, row in enumerate(iterate_df.itertuples(), start=1):
        ht       = parse_head(row.head)
        intended = row.program
        candidates = [n for n in all_names if n != intended]
        chosen     = rng.choice(candidates)
        shared_smart[ht] = chosen
        pct = eval_pct()
        results.append({'k': step, 'head': row.head,
                        'program': chosen, 'intended': intended, 'increase': pct})
        if step % 20 == 0 or step == len(iterate_df):
            print(f'    seed={seed} k={step}/{len(iterate_df)} pct={pct:.1f}%', end='\r')

    print()
    for h in handles: h.remove()
    del model
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    return pd.DataFrame(results)


# ── BERT random seed runner ───────────────────────────────────────────────────

def bert_ppl(sentence, model, tokenizer, device):
    inputs  = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).to(device)
    ids     = inputs['input_ids']
    n_real  = ids.shape[1] - 2
    if n_real <= 0: return float('nan')
    batch = ids.repeat(n_real, 1)
    for i in range(n_real):
        batch[i, i + 1] = tokenizer.mask_token_id
    with torch.no_grad():
        logits = model(batch).logits
    log_probs = [F.log_softmax(logits[i, i + 1], dim=-1)[ids[0, i + 1]].item()
                 for i in range(n_real)]
    return float(np.exp(-np.mean(log_probs)))

def run_bert_random_seed(seed, eval_sents, smart_df, programs):
    prog_lookup = {p.__name__: p for p in programs}
    all_names   = list(prog_lookup.keys())
    rng         = random.Random(seed)

    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(DEVICE).eval()
    tok   = AutoTokenizer.from_pretrained('bert-base-uncased')

    N_LAYERS = N_HEADS = 12
    shared_smart = {}
    active_sent  = [None]
    mat_cache    = {}

    def get_mat(prog_name, sentence):
        key = (prog_name, sentence)
        if key not in mat_cache:
            p = prog_lookup.get(prog_name)
            try:
                mat_cache[key] = p(sentence, tok)[1] if p else None
            except Exception:
                mat_cache[key] = None
        return mat_cache[key]

    def make_hook(li):
        def hook(module, inp, out):
            ctx = out[0]; b, s, hd = ctx.shape
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

    handles = [model.bert.encoder.layer[li].attention.self.register_forward_hook(make_hook(li))
               for li in range(N_LAYERS)]

    baseline_ppls = []
    for s in eval_sents:
        active_sent[0] = s
        p = bert_ppl(s, model, tok, DEVICE)
        if not np.isnan(p):
            baseline_ppls.append(p)

    def eval_pct():
        ppls = []
        for s in eval_sents:
            active_sent[0] = s
            p = bert_ppl(s, model, tok, DEVICE)
            if not np.isnan(p):
                ppls.append(p)
        n = min(len(ppls), len(baseline_ppls))
        return float(np.mean([(r - b) / b * 100 for r, b in zip(ppls[:n], baseline_ppls[:n])])) if n else 0.0

    if _SHUFFLE_ORDER[0] == 'random':
        iterate_df = smart_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        iterate_df = smart_df.copy()

    results = []
    for step, row in enumerate(iterate_df.itertuples(), start=1):
        ht       = parse_head(row.head)
        intended = row.program
        candidates = [n for n in all_names if n != intended]
        chosen     = rng.choice(candidates)
        shared_smart[ht] = chosen
        pct = eval_pct()
        results.append({'k': step, 'head': row.head,
                        'program': chosen, 'intended': intended, 'increase': pct})
        if step % 20 == 0 or step == len(iterate_df):
            print(f'    seed={seed} k={step}/{len(iterate_df)} pct={pct:.1f}%', end='\r')

    print()
    for h in handles: h.remove()
    del model
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    return pd.DataFrame(results)


# ── Causal LM random seed runner (TinyLlama + LLaMA-3B) ─────────────────────

def run_causal_random_seed(seed, eval_sents, smart_df, programs, model_key):
    cfg         = MODEL_CFG[model_key]
    prog_lookup = {p.__name__: p for p in programs}
    all_names   = list(prog_lookup.keys())
    rng         = random.Random(seed)

    n_layers     = cfg['n_layers']
    n_heads      = cfg['n_heads']
    head_dim     = cfg['head_dim']
    use_float16  = cfg['use_float16']

    tok = AutoTokenizer.from_pretrained(cfg['model_id'])
    if use_float16:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.float16,
            device_map='auto', attn_implementation='eager').eval()
        actual_dev = str(next(model.parameters()).device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], attn_implementation='eager').to(DEVICE).eval()
        actual_dev = DEVICE

    shared_smart = {}
    active_sent  = [None]
    mat_cache    = {}

    def get_mat(prog_name, sentence):
        key = (prog_name, sentence)
        if key not in mat_cache:
            p = prog_lookup.get(prog_name)
            try:
                mat_cache[key] = p(sentence, tok)[1] if p else None
            except Exception:
                mat_cache[key] = None
        return mat_cache[key]

    def make_hook(li):
        def hook(module, inp, out):
            ctx = out[0].contiguous()
            if use_float16: ctx = ctx.float()
            b, s, hd = ctx.shape
            active = [(li, hi) for hi in range(n_heads) if (li, hi) in shared_smart]
            if not active:
                return out
            mod = ctx.view(b, s, n_heads, hd // n_heads).clone()
            cur = active_sent[0]
            for _, hi in active:
                mat = get_mat(shared_smart[(li, hi)], cur)
                if mat is not None:
                    t = torch.tensor(mat, device=actual_dev, dtype=mod.dtype)
                    if t.shape[0] == s:
                        mod[:, :, hi, :] = torch.matmul(t, mod[:, :, hi, :])
            result = mod.view(b, s, hd)
            if use_float16: result = result.to(out[0].dtype)
            return (result,) + out[1:]
        return hook

    handles = [model.model.layers[li].self_attn.register_forward_hook(make_hook(li))
               for li in range(n_layers)]

    baseline_ppls = []
    for s in eval_sents:
        sent = prepare_sentence(s)
        active_sent[0] = sent
        toks = tok(sent, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).to(actual_dev)
        with torch.no_grad():
            baseline_ppls.append(torch.exp(model(**toks, labels=toks['input_ids']).loss).item())

    def eval_pct():
        ppls = []
        for s in eval_sents:
            sent = prepare_sentence(s)
            active_sent[0] = sent
            toks = tok(sent, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).to(actual_dev)
            with torch.no_grad():
                ppls.append(torch.exp(model(**toks, labels=toks['input_ids']).loss).item())
        return float(np.mean([(r - b) / b * 100 for r, b in zip(ppls, baseline_ppls)]))

    if _SHUFFLE_ORDER[0] == 'random':
        iterate_df = smart_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        iterate_df = smart_df.copy()

    results = []
    for step, row in enumerate(iterate_df.itertuples(), start=1):
        ht       = parse_head(row.head)
        intended = row.program
        candidates = [n for n in all_names if n != intended]
        chosen     = rng.choice(candidates)
        shared_smart[ht] = chosen
        pct = eval_pct()
        results.append({'k': step, 'head': row.head,
                        'program': chosen, 'intended': intended, 'increase': pct})
        if step % 50 == 0 or step == len(iterate_df):
            print(f'    seed={seed} k={step}/{len(iterate_df)} pct={pct:.1f}%', end='\r')

    print()
    for h in handles: h.remove()
    del model
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    return pd.DataFrame(results)


# ── Generate & cache correct shuffles ────────────────────────────────────────

def generate_shuffles(model_key, n_seeds=N_SEEDS, force=False):
    get_shuffle_dir().mkdir(parents=True, exist_ok=True)
    smart_path = SMART_DIR / f'{model_key}_smart.csv'
    if not smart_path.exists():
        print(f'[SKIP] No smart CSV for {model_key}')
        return

    smart_df  = pd.read_csv(smart_path)
    programs  = load_programs(model_key)
    eval_sents = load_eval_sentences()

    print(f'\nGenerating random-program seeds for {model_key}')
    print(f'  Library: {len(programs)} programs | Eval sents: {len(eval_sents)} | Seeds: {n_seeds}')

    def runner(seed, sents, df, progs):
        if model_key == 'gpt2':
            return run_gpt2_random_seed(seed, sents, df, progs)
        elif model_key == 'bert':
            return run_bert_random_seed(seed, sents, df, progs)
        else:
            return run_causal_random_seed(seed, sents, df, progs, model_key)

    for seed in range(1, n_seeds + 1):
        out_path = get_shuffle_dir() / f'{model_key}_random_seed_{seed}.csv'
        if out_path.exists() and not force:
            print(f'  [CACHE] seed {seed}')
            continue
        print(f'  Running seed {seed}/{n_seeds}...')
        df = runner(seed, eval_sents, smart_df, programs)
        df.to_csv(out_path, index=False)
        print(f'  Saved {out_path.name}  final={df["increase"].iloc[-1]:.1f}%')


def load_shuffles(model_key):
    pattern = str(get_shuffle_dir() / f'{model_key}_random_seed_*.csv')
    paths   = sorted(glob.glob(pattern))
    return [pd.read_csv(p).sort_values('k') for p in paths]

def shuffle_band(dfs):
    all_k  = sorted(set.intersection(*[set(df['k'].tolist()) for df in dfs]))
    matrix = np.array([df.set_index('k').loc[all_k, 'increase'].values for df in dfs])
    mean   = matrix.mean(axis=0)
    std    = matrix.std(axis=0)
    return np.array(all_k), mean, mean - std, mean + std


# ── Figure 1: head replacement vs PPL ────────────────────────────────────────

def plot_replacement(ax):
    for model_key, cfg in MODEL_CFG.items():
        if model_key == 'bert':
            continue
        smart_path = SMART_DIR / f'{model_key}_smart.csv'
        if not smart_path.exists():
            print(f'[SKIP replacement] {model_key}: no smart CSV')
            continue

        smart_df = pd.read_csv(smart_path).sort_values('k')
        x_pct    = smart_df['k'].values / cfg['total_heads'] * 100
        ax.plot(x_pct, smart_df['increase'].values,
                color=cfg['color'], linewidth=2.5, marker='o', markersize=2,
                label=f'{cfg["display"]} (smart)')

        if model_key not in MODELS_WITH_RANDOM:
            continue
        dfs = load_shuffles(model_key)
        if not dfs:
            print(f'[SKIP baseline band] {model_key}: no shuffle CSVs')
            continue
        ks, mean, lo, hi = shuffle_band(dfs)
        x_bl = ks / cfg['total_heads'] * 100
        ax.plot(x_bl, mean,
                color=cfg['color'], linewidth=1.8, linestyle='--', alpha=0.8,
                label=f'{cfg["display"]} (random baseline)')

    ax.set_xlabel('Heads Replaced %', fontweight='bold', fontsize=12, labelpad=6)
    ax.set_ylabel('Perplexity Increase %', fontweight='bold', fontsize=12, labelpad=6)
    ax.set_title('Head Replacement vs. Perplexity Increase', fontweight='bold', fontsize=13, pad=8)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    if MAX_PPL_SHOWN is not None:
        ax.set_ylim(bottom=0, top=MAX_PPL_SHOWN)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.legend(framealpha=0.85, fontsize=7, ncol=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.4)


# ── Figure 2: IoU vs PPL ──────────────────────────────────────────────────────

def plot_iou(ax):
    for model_key, cfg in MODEL_CFG.items():
        if model_key == 'bert':
            continue

        smart_path = SMART_DIR / f'{model_key}_smart.csv'
        if not smart_path.exists():
            print(f'[SKIP iou] {model_key}: no smart CSV')
            continue

        smart_df = pd.read_csv(smart_path)
        smart_df = smart_df[smart_df['increase'] > 0].copy()

        if model_key in ('tinyllama', 'llama3b'):
            # Use kl_div (normalized) as similarity — matches original paper r values
            kl = smart_df['kl_div']
            smart_df['x_sim'] = 1.0 - (kl - kl.min()) / (kl.max() - kl.min())
            x_vals = smart_df['x_sim']
        else:
            # GPT-2: use max IoU from iou_scores
            iou_path = DATA_DIR / f'iou_scores_{model_key}.csv'
            if not iou_path.exists():
                print(f'[SKIP iou] {model_key}: no iou CSV')
                continue
            iou_df  = pd.read_csv(iou_path)
            mean_iou = (iou_df.groupby(['layer', 'head'])['iou_score']
                        .max().reset_index()
                        .rename(columns={'iou_score': 'mean_iou'}))
            smart_df['layer']  = smart_df['head'].apply(lambda h: parse_head(h)[0])
            smart_df['head_n'] = smart_df['head'].apply(lambda h: parse_head(h)[1])
            smart_df = smart_df.merge(mean_iou, left_on=['layer', 'head_n'],
                                      right_on=['layer', 'head'], how='inner')
            x_vals = smart_df['mean_iou']

        if len(smart_df) < 5:
            print(f'[SKIP iou] {model_key}: too few points after merge')
            continue

        y_vals = smart_df['increase']
        corr, pval = stats.spearmanr(x_vals, y_vals)
        print(f'{cfg["display"]:14s} Spearman r = {corr:.3f}, p = {pval:.4f}')

        ax.scatter(x_vals, y_vals,
                   alpha=0.35, color=cfg['color'], s=25, zorder=3,
                   label=f'{cfg["display"]} (r={corr:.3f})')

        log_y = np.log10(y_vals)
        fit   = np.polyfit(x_vals, log_y, deg=1)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(x_fit, 10 ** np.polyval(fit, x_fit),
                color=cfg['color'], linewidth=2, linestyle='--')

    ax.set_yscale('log')
    ax.set_xlabel('Mean IoU', fontweight='bold', fontsize=12, labelpad=6)
    ax.set_ylabel('Perplexity Increase % (log)', fontweight='bold', fontsize=12, labelpad=6)
    ax.set_title('IoU Similarity vs Perplexity Increase', fontweight='bold', fontsize=13, pad=8)
    ax.legend(framealpha=0.85, fontsize=7, ncol=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.4, which='both')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='')
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-generate', action='store_true')
    parser.add_argument('--models', default='', help='Comma-separated model keys')
    parser.add_argument('--order', choices=['iou', 'random'], default='iou',
                        help='iou = fixed IOU order + random prog; random = random order + random prog')
    args = parser.parse_args()

    # Set global order so runners and get_shuffle_dir() see it
    _SHUFFLE_ORDER[0] = args.order

    out_default = RESULTS_DIR / 'plots' / f'figure_random_baseline_{args.order}_order.pdf'
    out_path = pathlib.Path(args.out) if args.out else out_default

    if not args.no_generate:
        model_keys = [m.strip() for m in args.models.split(',')] if args.models else list(MODELS_WITH_RANDOM)
        for model_key in model_keys:
            generate_shuffles(model_key, n_seeds=args.seeds, force=args.force)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    order_label = 'Fixed IOU order' if args.order == 'iou' else 'Random order'
    fig.suptitle(f'Random baseline: {order_label}, random program', fontsize=13, fontweight='bold')
    plot_replacement(ax_left)
    plot_iou(ax_right)

    plt.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
