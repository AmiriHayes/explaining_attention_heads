"""
port_old_plots.py
==================
Re-renders the paper's original figures (from code/write_data.ipynb) using the
corrected v2 data (fault-tolerant replacement, 1000-sentence TinyStories,
centroid categorization, 5-baseline perplexity, 3-seed downstream) instead of
the old buggy/hardcoded data. Plotting code is ported near-verbatim from
write_data.ipynb; only the data-loading is swapped to point at
results/replacement_run_v2/*.

Outputs -> results/plots/*_v2.pdf (+.png where the original saved both),
so the original paper PDFs are never overwritten.
"""
import inspect, json, os, pathlib, random, re, sys, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = pathlib.Path('/Users/amirihayes/Downloads/explaining_attention_heads-main')
DATA_DIR = ROOT / 'data'
RES_DIR = ROOT / 'results'
V2_DIR = RES_DIR / 'replacement_run_v2'
PLOTS_DIR = RES_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(ROOT / 'code'))

MODELS = ['gpt2', 'tinyllama', 'llama3b']  # bert was never in this run's scope

MODEL_CFG = {
    'gpt2':      {'model_id': 'gpt2', 'dtype': torch.float32,
                   'device': 'cpu'},
    'tinyllama': {'model_id': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                   'dtype': torch.float32,
                   'device': 'mps' if torch.backends.mps.is_available() else 'cpu'},
    'llama3b':   {'model_id': 'meta-llama/Llama-3.2-3B', 'dtype': torch.float16,
                   'device': 'mps' if torch.backends.mps.is_available() else 'cpu'},
}
DISPLAY = {'gpt2': 'GPT-2', 'tinyllama': 'TinyLlama', 'llama3b': 'Llama-3.2-3B'}


def iou_score(p, q):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    return float(np.minimum(p, q).sum() / np.maximum(p, q).sum())


# ─── program loading (mirrors write_data.ipynb cell 4) ─────────────────────
def load_programs_for_model(model_key):
    prog_file = DATA_DIR / f'{model_key}_programs.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location(f'{model_key}_programs', prog_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    progs = [obj for _, obj in inspect.getmembers(mod, inspect.isfunction)
             if re.search(r'_[Ll]\d+[Hh]\d+$', obj.__name__)]
    return progs, mod


def parse_intended_head(name):
    m = re.search(r'_[Ll](\d+)[Hh](\d+)$', name)
    return (int(m.group(1)), int(m.group(2))) if m else None


MODEL_PROGRAMS = {}
INTENDED_MAPS = {}
for mk in MODELS:
    progs, mod = load_programs_for_model(mk)
    MODEL_PROGRAMS[mk] = progs
    imap = {}
    for f in progs:
        lh = parse_intended_head(f.__name__)
        if lh is not None:
            imap[lh] = f
    INTENDED_MAPS[mk] = imap
    print(f'[programs] {mk:12s}: {len(progs)} programs, {len(imap)} intended-head mappings')


# ─── caching patch (same as build_notebook.py) so program calls don't re-parse ──
def _patch_program_caching(lib):
    if getattr(lib, '_caching_patched', False):
        return
    _spacy_cache, _align_cache, _emb_cache = {}, {}, {}
    if hasattr(lib, 'spacy_parse'):
        orig_spacy_parse = lib.spacy_parse
        def cached_spacy_parse(s):
            if s not in _spacy_cache:
                _spacy_cache[s] = orig_spacy_parse(s)
            return _spacy_cache[s]
        lib.spacy_parse = cached_spacy_parse
    if hasattr(lib, '_align_to_spacy'):
        orig_align = lib._align_to_spacy
        def cached_align(s, t):
            key = (s, tuple(t))
            if key not in _align_cache:
                _align_cache[key] = orig_align(s, t)
            return _align_cache[key]
        lib._align_to_spacy = cached_align
    if hasattr(lib, 'embedding_similarity'):
        orig_emb_sim = lib.embedding_similarity
        def cached_emb_sim(t, i, j):
            key = (t[i], t[j])
            if key not in _emb_cache:
                _emb_cache[key] = orig_emb_sim(t, i, j)
            return _emb_cache[key]
        lib.embedding_similarity = cached_emb_sim
    lib._caching_patched = True


for mk in MODELS:
    _, mod = load_programs_for_model(mk)
    _patch_program_caching(mod)


# ─── sentences: reuse the validated 1000-sentence TinyStories pool, take 100 ──
with open(RES_DIR / '_cache_tinystories_1000.json') as f:
    _pool = json.load(f)
_rng = random.Random(42)
_shuf = _pool[:]
_rng.shuffle(_shuf)
BOX_SENTENCES = _shuf[:100]
print(f'[sentences] Using {len(BOX_SENTENCES)} TinyStories sentences for box-plot baselines '
      f'(vs. original paper\'s 5 generic sentences -- same pool source as the corrected v2 run)')


# ─── tokenizer/model loaders (mirrors fixed_attention_*.py conventions) ─────
def load_tok(mk):
    if mk == 'gpt2':
        from gpt2_tok_shim import GPT2TokShim
        return GPT2TokShim()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_CFG[mk]['model_id'])


def load_attn_model(mk):
    cfg = MODEL_CFG[mk]
    if mk == 'gpt2':
        from transformers import GPT2Model
        return GPT2Model.from_pretrained(cfg['model_id'], output_attentions=True,
                                          attn_implementation='eager').to(cfg['device']).eval()
    from transformers import AutoModel
    return AutoModel.from_pretrained(cfg['model_id'], output_attentions=True,
                                      attn_implementation='eager',
                                      dtype=cfg['dtype']).to(cfg['device']).eval()


def get_attention_cache(model, tok, sentences, device):
    cache = []
    for sent in sentences:
        inputs = tok(sent, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**inputs)
        sent_cache = {}
        for l, attn in enumerate(out.attentions):
            attn_np = attn[0].to(torch.float32).cpu().numpy()
            for h in range(attn_np.shape[0]):
                sent_cache[(l, h)] = attn_np[h]
        cache.append(sent_cache)
    return cache


def get_program_matrix(prog, sent, tok, target_n):
    try:
        result = prog(sent, tok)
        if isinstance(result, tuple):
            result = result[1]
        result = np.array(result, dtype=np.float64)
        if result.ndim != 2 or result.shape[0] != result.shape[1] or result.shape[0] != target_n:
            return None
        result = np.clip(result, 0, None)
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return result / row_sums
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════
# STEP 1: compute baselines_{mk}_v2.npz + intended_iou (cheap: real attention
# extraction + ONE program per head, not the full program x head matrix)
# ═════════════════════════════════════════════════════════════════════════
print('\n=== Step 1: baselines + intended-program IoU (per model) ===')
baselines = {}
intended_iou_by_model = {}

for mk in MODELS:
    npz_path = RES_DIR / f'baselines_{mk}_v2.npz'
    int_path = RES_DIR / f'intended_iou_{mk}_v2.csv'

    have_baselines = npz_path.exists()
    have_intended = int_path.exists()
    if have_baselines and have_intended:
        d = np.load(npz_path)
        baselines[mk] = (d['token'], d['col'], d['diag'])
        intended_iou_by_model[mk] = pd.read_csv(int_path)
        print(f'[{mk}] SKIPPED (fully cached) -- loaded {npz_path.name}, {int_path.name}', flush=True)
        continue
    if have_baselines:
        d = np.load(npz_path)
        baselines[mk] = (d['token'], d['col'], d['diag'])
        print(f'[{mk}] baselines cached, only intended-program IoU still needed', flush=True)

    print(f'[{mk}] loading model + extracting attention over {len(BOX_SENTENCES)} sentences...', flush=True)
    tok = load_tok(mk)
    model = load_attn_model(mk)
    device = MODEL_CFG[mk]['device']
    cache = get_attention_cache(model, tok, BOX_SENTENCES, device)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    del model
    print(f'[{mk}] attention cache built ({n_layers} layers x {n_heads} heads x {len(BOX_SENTENCES)} sentences)', flush=True)

    if not have_baselines:
        token_ious, col_ious, diag_ious = [], [], []
        for l in range(n_layers):
            for h in range(n_heads):
                for si in range(len(BOX_SENTENCES)):
                    a = cache[si].get((l, h))
                    if a is None:
                        continue
                    n = a.shape[0]
                    rt = np.zeros_like(a)
                    cols = np.random.randint(0, n, size=n)
                    rt[np.arange(n), cols] = 1.0
                    token_ious.append(iou_score(a, rt))

                    rc = np.zeros_like(a)
                    c = np.random.randint(0, n)
                    rc[:, c] = 1.0 / n
                    col_ious.append(iou_score(a, rc))

                    ld = np.tril(np.ones((n, n)))
                    ld = ld / (ld.sum(axis=1, keepdims=True) + 1e-12)
                    diag_ious.append(iou_score(a, ld))
            print(f'  [{mk}] baseline layer {l+1}/{n_layers} done', flush=True)

        t, c, d = np.array(token_ious) * 100, np.array(col_ious) * 100, np.array(diag_ious) * 100
        baselines[mk] = (t, c, d)
        np.savez(npz_path, token=t, col=c, diag=d)
        print(f'  saved {npz_path.name}  (n={len(t)} samples)', flush=True)

    # intended-program IoU: one program per head. NOTE: still O(heads x sentences)
    # program executions, so use a smaller sub-slice than the baseline sentence
    # count to keep this tractable (704/672 heads x 100 sentences = ~70k spacy
    # program calls each -- that's what made this hang for over an hour).
    INTENDED_SENTENCES_N = 10
    imap = INTENDED_MAPS[mk]
    rows = []
    n_imap = len(imap)
    for pi, ((l, h), prog) in enumerate(imap.items()):
        scores = []
        for si in range(min(INTENDED_SENTENCES_N, len(BOX_SENTENCES))):
            sent = BOX_SENTENCES[si]
            a = cache[si].get((l, h))
            if a is None:
                continue
            n = a.shape[0]
            pmat = get_program_matrix(prog, sent, tok, n)
            if pmat is None:
                continue
            scores.append(iou_score(a, pmat))
        if scores:
            rows.append({'layer': l, 'head': h, 'intended_program': prog.__name__,
                          'intended_iou': float(np.mean(scores)) * 100})
        if (pi + 1) % 100 == 0 or (pi + 1) == n_imap:
            print(f'  [{mk}] intended-IoU {pi+1}/{n_imap} heads done', flush=True)
    df_int = pd.DataFrame(rows)
    df_int.to_csv(int_path, index=False)
    intended_iou_by_model[mk] = df_int
    print(f'  saved {int_path.name}  ({len(df_int)} heads)', flush=True)
    del cache

print('[DONE] Step 1.')


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3 / figure_4_1 — Program IoU Similarity box plots (ported cell 11)
# ═════════════════════════════════════════════════════════════════════════
print('\n=== Figure 3 (box plots) ===')

ROW_COLORS = {
    'bert':      {'base': '#fdebd0', 'box': '#f0a868'},
    'gpt2':      {'base': '#f5caca', 'box': '#e07070'},
    'tinyllama': {'base': '#d6eaf8', 'box': '#7fb3d5'},
    'llama3b':   {'base': '#d4f5ec', 'box': '#b5ead7'},
}
LABELS_5 = ['Random\nToken', 'Random\nColumn', 'Lower\nDiagonal', 'Intended\nProgram', 'Best\nProgram']


def styled_boxplot(ax, data_list, labels, colors):
    valid = [(d, l, c, i + 1) for i, (d, l, c) in
             enumerate(zip(data_list, labels, colors)) if len(d) > 0]
    if not valid:
        return
    vdata, vlabels, vcolors, vpos = zip(*valid)
    bp = ax.boxplot(list(vdata), positions=list(vpos), patch_artist=True, widths=0.55,
                     medianprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='o', markersize=2, alpha=0.3),
                     whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0))
    for patch, color in zip(bp['boxes'], vcolors):
        patch.set_facecolor(color); patch.set_edgecolor('black'); patch.set_linewidth(0.7)
    ax.set_xticks(list(vpos))
    ax.set_xticklabels(list(vlabels), fontsize=8, ha='center')


keys = ['bert', 'gpt2', 'tinyllama', 'llama3b']
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

for idx, model_key in enumerate(keys):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    c = ROW_COLORS[model_key]
    disp = DISPLAY.get(model_key, model_key.upper())

    if model_key not in MODELS:
        ax.set_title(f'Program IoU Similarity: {disp}', fontweight='bold', pad=8)
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(LABELS_5, fontsize=8)
        ax.text(0.5, 0.5, 'not run this session\n(BERT out of scope)', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray', style='italic')
        ax.set_ylim(0, 100)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        continue

    bf = pd.read_csv(V2_DIR / f'best_fits_{model_key}_v2.csv')
    best_ious = (bf['best_iou'] * 100).tolist()
    intended_ious = intended_iou_by_model[model_key]['intended_iou'].tolist()
    t_bl, c_bl, d_bl = baselines[model_key]

    data_list = [t_bl.tolist(), c_bl.tolist(), d_bl.tolist(), intended_ious, best_ious]
    colors = [c['base'], c['base'], c['base'], c['box'], c['box']]

    styled_boxplot(ax, data_list, LABELS_5, colors)
    ax.set_title(f'Program IoU Similarity: {disp}', fontweight='bold', pad=8)
    ax.set_ylabel('Similarity (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.savefig(PLOTS_DIR / 'figure_4_1_v2.pdf', bbox_inches='tight', dpi=150)
plt.savefig(PLOTS_DIR / 'figure_3_boxplots_v2.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('[DONE] Figure 3 saved -> figure_4_1_v2.pdf')


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4 — per-model heatmaps (ported cell 13, reusing existing semantic
# category JSONs -- static, program libraries unchanged)
# ═════════════════════════════════════════════════════════════════════════
print('\n=== Figure 4 (heatmaps) ===')

CATEGORY_COLORS = {
    'initiation_and_anchoring':      '#4285F4',
    'relational_and_semantic':       '#EA4335',
    'special_tokens_and_boundaries': '#FBBC05',
    'linguistic_and_syntactic':      '#FF6D00',
    'sequential_and_induction':      '#34A853',
    'uniformity_and_global':         '#A142F4',
}
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'custom_gray', list(zip([0.0, 0.4, 1.0], [(1, 1, 1), (1, 1, 1), (0.15, 0.15, 0.15)])))

PROG_TO_CAT_BY_MODEL = {}
for mk in MODELS:
    cat_path = RES_DIR / f'{mk}_program_categories.json'
    with open(cat_path) as f:
        cats = json.load(f)
    PROG_TO_CAT_BY_MODEL[mk] = {name: cat for cat, names in cats.items() for name in names}


def get_cat_for_model(model_key, prog_name):
    return PROG_TO_CAT_BY_MODEL.get(model_key, {}).get(prog_name, 'uniformity_and_global')


def get_prog_index(prog_name, model_key):
    for i, p in enumerate(MODEL_PROGRAMS.get(model_key, [])):
        if p.__name__ == prog_name:
            return i + 1
    return 0


def plot_heatmap(model_key):
    bf = pd.read_csv(V2_DIR / f'best_fits_{model_key}_v2.csv')
    n_layers = int(bf['layer'].max()) + 1
    n_heads = int(bf['head'].max()) + 1

    score_mat = np.zeros((n_layers, n_heads))
    name_mat = np.full((n_layers, n_heads), '', dtype=object)
    for _, row in bf.iterrows():
        l, h = int(row['layer']), int(row['head'])
        score_mat[l, h] = row['best_iou'] if not pd.isna(row['best_iou']) else 0.0
        name_mat[l, h] = str(row['best_program'])

    cell = 0.5
    fig_w = n_heads * cell + 2.5
    fig_h = n_layers * cell + 4.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 1, left=0.15, right=0.85, top=0.9, bottom=0.4)
    ax = fig.add_subplot(gs[0])

    sns.heatmap(score_mat, ax=ax, cmap=HEATMAP_CMAP, cbar=False,
                linewidths=0.4, linecolor='#EEEEEE', vmin=0, vmax=1, square=True)

    label_fontsize = max(4, 7 - n_heads // 16)
    for l in range(n_layers):
        for h in range(n_heads):
            name = name_mat[l, h]
            if not name:
                continue
            cat = get_cat_for_model(model_key, name)
            color = CATEGORY_COLORS.get(cat, '#888888')
            idx = get_prog_index(name, model_key)
            ax.add_patch(plt.Circle((h + 0.5, l + 0.5), radius=0.38, facecolor='none',
                                     edgecolor=color, linewidth=2.0, zorder=3))
            bg_val = score_mat[l, h]
            tc = 'white' if bg_val > 0.65 else 'black'
            ax.text(h + 0.5, l + 0.5, str(idx), ha='center', va='center',
                    fontsize=label_fontsize, color=tc, fontweight='bold', zorder=4)

    ax.invert_yaxis()
    ax.set_ylim(0, n_layers)
    ax.set_xlabel('Head', fontsize=14, labelpad=14)
    ax.set_ylabel('Layer', fontsize=14, labelpad=14)

    plt.draw()
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.09, pos.width, 0.025])
    sm = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=mcolors.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label('IoU Similarity', fontsize=12, labelpad=8)
    cb.ax.tick_params(labelsize=7)

    leg_ax = fig.add_axes([pos.x0, pos.y0 - 0.23, pos.width, 0.09])
    leg_ax.axis('off')
    cat_patches = [mpatches.Patch(color=v, label=k.replace('_', ' ').title())
                   for k, v in CATEGORY_COLORS.items()]
    leg = leg_ax.legend(handles=cat_patches, loc='upper center', ncol=3, fontsize=16,
                         frameon=False, title_fontproperties={'weight': 'bold', 'size': 10},
                         handlelength=1.5, columnspacing=0.8, labelspacing=0.6, borderaxespad=0.0)
    leg._legend_box.sep = 10

    out = PLOTS_DIR / f'figure_4_{model_key}_heatmap_v2'
    plt.savefig(str(out) + '.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'[{model_key}] saved -> {out}.pdf')


for mk in MODELS:
    plot_heatmap(mk)
print('[DONE] Figure 4 heatmaps saved.')


# ═════════════════════════════════════════════════════════════════════════
# PERPLEXITY FIGURE — Heads Replaced % vs Perplexity Increase % (ported cell 18)
# chosen (solid) vs wrong_category (dashed) per model -- matches the original's
# 1 smart-line + 1 baseline-line-per-model density; wrong_category is the
# reviewer-motivated baseline this whole run exists to report.
# ═════════════════════════════════════════════════════════════════════════
print('\n=== Perplexity figure (Heads Replaced % vs PPL Increase %) ===')

TOTAL_HEADS = {'gpt2': 144, 'tinyllama': 704, 'llama3b': 672}
MODEL_PLOT_CONFIG = {
    'gpt2':      {'display': 'GPT-2',        'color': '#c0392b'},
    'tinyllama': {'display': 'TinyLlama',    'color': '#1f77b4'},
    'llama3b':   {'display': 'Llama-3.2-3B', 'color': '#2a9d8f'},
}
MAX_INCREASE_SHOWN = 400

fig, ax = plt.subplots(figsize=(7, 5))
for mk, cfg in MODEL_PLOT_CONFIG.items():
    smart_df = pd.read_csv(V2_DIR / f'ppl_sweep_{mk}_chosen_v2.csv').sort_values('k')
    x_pct = smart_df['k'].values / TOTAL_HEADS[mk] * 100
    ax.plot(x_pct, smart_df['increase'].values, color=cfg['color'], linewidth=2.5,
            marker='o', markersize=2, label=f'{cfg["display"]} — Chosen Program')

    bl_df = pd.read_csv(V2_DIR / f'ppl_sweep_{mk}_wrong_category_v2.csv').sort_values('k')
    bl_x = bl_df['k'].values / TOTAL_HEADS[mk] * 100
    ax.plot(bl_x, bl_df['increase'].values, color=cfg['color'], linewidth=1.8,
            linestyle='--', alpha=0.7, label=f'{cfg["display"]} — Wrong-Category')

ax.set_xlabel('Heads Replaced %', fontweight='bold', fontsize=14, labelpad=7)
ax.set_ylabel('Perplexity Increase %', fontweight='bold', fontsize=14, labelpad=7)
ax.set_title('Head Replacement vs. Perplexity Increase\n', fontweight='bold', fontsize=16)
ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 5))
if MAX_INCREASE_SHOWN is not None:
    ax.set_ylim(bottom=0, top=MAX_INCREASE_SHOWN)
ax.legend(framealpha=0.85, fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'figure_4_3_v2.pdf', bbox_inches='tight', dpi=600)
plt.savefig(PLOTS_DIR / 'figure_5_perplexity_v2.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('[DONE] Perplexity figure saved -> figure_4_3_v2.pdf')

# secondary: all 5 baselines per model, small multiples (extra, not in original)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
COND_STYLE = {
    'chosen':             {'color': '#2c3e50', 'ls': '-',  'lw': 2.5, 'label': 'Chosen Program'},
    'wrong_category':     {'color': '#c0392b', 'ls': '--', 'lw': 1.8, 'label': 'Wrong-Category'},
    'random_program':     {'color': '#e67e22', 'ls': '--', 'lw': 1.5, 'label': 'Random Program'},
    'structural_uniform': {'color': '#8e44ad', 'ls': ':',  'lw': 1.5, 'label': 'Structural Uniform'},
    'true_random':        {'color': '#7f8c8d', 'ls': ':',  'lw': 1.5, 'label': 'True Random Attn'},
}
for ax, mk in zip(axes, MODELS):
    for cond, style in COND_STYLE.items():
        p = V2_DIR / f'ppl_sweep_{mk}_{cond}_v2.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p).sort_values('k')
        x = df['k'].values / TOTAL_HEADS[mk] * 100
        ax.plot(x, df['increase'].values, color=style['color'], linestyle=style['ls'],
                linewidth=style['lw'], label=style['label'])
    ax.set_title(DISPLAY[mk], fontweight='bold', fontsize=13)
    ax.set_xlabel('Heads Replaced %', fontsize=11)
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axes[0].set_ylabel('Perplexity Increase %', fontsize=11)
axes[0].legend(fontsize=8, framealpha=0.85)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'figure_4_3_all_baselines_v2.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)
print('[DONE] Extended (all-baselines) perplexity figure saved -> figure_4_3_all_baselines_v2.pdf')


# ═════════════════════════════════════════════════════════════════════════
# IoU vs PERPLEXITY scatter (ported cell 19) -- reconstruct per-step head
# identity from the IoU descending order (SWEEP_STRIDE=1 in the real run)
# ═════════════════════════════════════════════════════════════════════════
print('\n=== IoU vs Perplexity scatter ===')

fig, ax = plt.subplots(figsize=(7, 5))
for mk, cfg in MODEL_PLOT_CONFIG.items():
    bf = pd.read_csv(V2_DIR / f'best_fits_{mk}_v2.csv').sort_values('best_iou', ascending=False).reset_index(drop=True)
    smart_df = pd.read_csv(V2_DIR / f'ppl_sweep_{mk}_chosen_v2.csv').sort_values('k').reset_index(drop=True)
    # step k (1-indexed) added head = bf.iloc[k-1]
    smart_df['mean_iou'] = smart_df['k'].apply(lambda k: bf.iloc[int(k) - 1]['best_iou'] if int(k) - 1 < len(bf) else np.nan)
    merged = smart_df.dropna(subset=['mean_iou'])
    merged = merged[merged['increase'] > 0].copy()
    if len(merged) < 5:
        print(f'[SKIP] {mk}: too few points')
        continue
    merged['log_increase'] = np.log10(merged['increase'])
    corr, pval = stats.spearmanr(merged['mean_iou'], merged['increase'])
    print(f"{cfg['display']:12s} Spearman r = {corr:.3f}, p = {pval:.4g}")

    ax.scatter(merged['mean_iou'], merged['increase'], alpha=0.35, color=cfg['color'],
               s=30, zorder=3, label=f'{cfg["display"]} (r={corr:.3f})')
    fit = np.polyfit(merged['mean_iou'], merged['log_increase'], deg=1)
    x_fit = np.linspace(merged['mean_iou'].min(), merged['mean_iou'].max(), 200)
    y_fit = 10 ** np.polyval(fit, x_fit)
    ax.plot(x_fit, y_fit, color=cfg['color'], linewidth=2, linestyle='--')

ax.set_yscale('log')
ax.set_xlabel('Best-Fit IoU', fontweight='bold', fontsize=14, labelpad=7)
ax.set_ylabel('Perplexity Increase % (Logged)', fontweight='bold', fontsize=14, labelpad=7)
ax.set_title('IoU Similarity vs Perplexity Increase\n', fontweight='bold', fontsize=16)
ax.legend(framealpha=0.85, fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'figure_iou_vs_perplexity_v2.pdf', bbox_inches='tight', dpi=600)
plt.savefig(PLOTS_DIR / 'figure_6_iou_vs_perplexity_v2.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('[DONE] IoU-vs-perplexity scatter saved -> figure_iou_vs_perplexity_v2.pdf')


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Downstream task accuracy vs heads replaced (ported cell 27,
# replacing the ORIGINAL's hardcoded-fake GPT-2-only data with real,
# computed results for all 3 models, chosen condition)
# ═════════════════════════════════════════════════════════════════════════
print('\n=== Figure 7 (downstream) ===')

plt.rcParams['font.family'] = 'serif'
META = {
    'HellaSwag':  {'desc': 'Common Sense Inference', 'n': '10k'},
    'PIQA':       {'desc': 'Physical Reasoning',      'n': '1.8k'},
    'SciQ':       {'desc': 'Science Understanding',   'n': '1k'},
    'ARC-Easy':   {'desc': 'Elementary Science',      'n': '2.3k'},
    'Social IQA': {'desc': 'Social Interaction',       'n': '1.9k'},
    'COPA':       {'desc': 'Causal Reasoning',         'n': '500'},
}
RANDOM_BASELINES = {'HellaSwag': 0.25, 'PIQA': 0.50, 'SciQ': 0.25,
                     'ARC-Easy': 0.25, 'Social IQA': 0.33, 'COPA': 0.50}
MODEL_COLOR = {'gpt2': {'line': '#c0392b', 'box': '#e07070'},
               'tinyllama': {'line': '#1f77b4', 'box': '#7fb3d5'},
               'llama3b': {'line': '#2a9d8f', 'box': '#b5ead7'}}

downstream = pd.read_csv(V2_DIR / 'downstream_results_v2.csv')
chosen = downstream[downstream['condition'] == 'chosen'].sort_values('replacement_pct')

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for ax, (metric, meta) in zip(axes.flatten(), META.items()):
    baseline = RANDOM_BASELINES[metric]
    ax.axhline(baseline, color='black', linestyle='--', dashes=(8, 4), linewidth=1.2,
               label='Random Guessing')
    for mk in MODELS:
        sub = chosen[chosen['model'] == mk]
        if metric not in sub.columns or sub.empty:
            continue
        ax.plot(sub['replacement_pct'], sub[metric], marker='o', linewidth=1.8, markersize=5,
                color=MODEL_COLOR[mk]['line'], markerfacecolor=MODEL_COLOR[mk]['box'],
                label=DISPLAY[mk])
    ax.set_title(metric, fontweight='bold', fontsize=11, pad=20)
    ax.text(0.5, 1.01, f'{meta["desc"]} ($N={meta["n"]}$)', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8.5, color='#333333')
    ax.set_ylim(0, 1.0); ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 20))
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel('% Attention Heads Replaced', fontsize=10)
    ax.set_ylabel('Accuracy Score', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.suptitle('Effect of Replacing Attention Heads on Downstream Model Evaluations\n',
             fontsize=12, fontweight='bold')
plt.savefig(PLOTS_DIR / 'figure_7_downstream_v2.pdf', bbox_inches='tight', dpi=150)
plt.savefig(PLOTS_DIR / 'figure_7_downstream_v2.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('[DONE] Figure 7 saved -> figure_7_downstream_v2.pdf (all 3 models, real data)')

print('\n[ALL DONE] All figures ported to', PLOTS_DIR)
