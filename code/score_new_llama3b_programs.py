"""
Score Jacob's newly-generated llama3b programs (llama_programs.zip) against
real Llama-3.2-3B attention, using the synthesis pipeline's own helpers.py
(which has per-sentence caching for spacy/alignment/embedding-similarity).

Correctly wires up init_lm() with Llama's real tokenizer + embeddings so
tokenize()/embedding_similarity() don't silently default to GPT-2.
"""
import argparse, importlib.util, json, pathlib, sys, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT      = pathlib.Path(__file__).parent.parent
LIB_DIR   = ROOT / 'data' / 'llama3b_programs_v2'
HELPERS_PATH = LIB_DIR / 'helpers.py'
PROGRAMS_DIR = LIB_DIR / 'generated_code'
MODEL_ID  = 'meta-llama/Llama-3.2-3B'

def iou_score(p, q):
    p = np.clip(p.astype(np.float64), 1e-12, 1.0)
    q = np.clip(q.astype(np.float64), 1e-12, 1.0)
    return float(np.minimum(p, q).sum() / np.maximum(p, q).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-sents', type=int, default=5)
    args = ap.parse_args()

    # ── load helpers.py as a module and expose its functions as globals for exec'd programs ──
    spec = importlib.util.spec_from_file_location('helpers', HELPERS_PATH)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    # ── load Llama-3.2-3B for real attention + embeddings ──
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID, output_attentions=True, attn_implementation='eager', torch_dtype=torch.float16
    ).to('mps' if torch.backends.mps.is_available() else 'cpu').eval()
    device = next(model.parameters()).device
    print(f'[model] loaded on {device}')

    embedding_matrix = model.get_input_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    helpers.init_lm(tok, embedding_matrix, add_special_tokens=False)
    print(f'[init_lm] wired to real Llama-3.2-3B tokenizer + embeddings ({embedding_matrix.shape})')

    # ── load all generated programs, exec'd with helpers' functions as globals ──
    prog_files = sorted(PROGRAMS_DIR.glob('*.py'))
    exec_globals_template = {
        'np': np,
        'tokenize': helpers.tokenize,
        'gpt2_tokenize': helpers.gpt2_tokenize,
        'spacy_parse': helpers.spacy_parse,
        'align_spacy_to_tokens': helpers.align_spacy_to_tokens,
        'align_tokens_to_spacy': helpers.align_tokens_to_spacy,
        'align_spacy_to_gpt2': helpers.align_spacy_to_gpt2,
        'align_gpt2_to_spacy': helpers.align_gpt2_to_spacy,
        'make_row_stochastic': helpers.make_row_stochastic,
        'apply_causal_mask': helpers.apply_causal_mask,
        'get_modifying_adjectives': helpers.get_modifying_adjectives,
        'embedding_similarity': helpers.embedding_similarity,
    }
    programs = {}
    load_errors = 0
    for f in prog_files:
        ns = dict(exec_globals_template)
        try:
            exec(f.read_text(), ns)
            fn = ns.get('predict_attention_map')
            if fn is not None:
                programs[f.stem] = fn
            else:
                load_errors += 1
        except Exception as e:
            load_errors += 1
    print(f'[programs] loaded {len(programs)}/{len(prog_files)} (load errors: {load_errors})')

    with open(ROOT / 'data' / 'generic_sentences.json') as f:
        sentences = json.load(f)[:args.n_sents]

    t0 = time.time()
    # head_key = (layer,head) -> {prog_name: [iou per sentence]}
    all_head_scores = {(l, h): {} for l in range(28) for h in range(24)}
    n_layers = n_heads = None

    for si, sent in enumerate(sentences):
        t_sent = time.time()
        inputs = tok(sent, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs)
        n = inputs['input_ids'].shape[1]
        real = {}
        for l, attn in enumerate(out.attentions):
            attn_np = attn[0].to(torch.float32).cpu().numpy()
            if n_heads is None:
                n_layers, n_heads = len(out.attentions), attn_np.shape[0]
            for h in range(attn_np.shape[0]):
                real[(l, h)] = attn_np[h]

        prog_errors = 0
        for pname, fn in programs.items():
            try:
                ptokens, pmat = fn(sent)
                pmat = np.asarray(pmat, dtype=np.float64)
                if pmat.shape != (n, n):
                    prog_errors += 1
                    continue
            except Exception:
                prog_errors += 1
                continue
            for ht, rmat in real.items():
                score = iou_score(rmat, pmat)
                all_head_scores[ht].setdefault(pname, []).append(score)

        print(f'  sentence {si+1}/{len(sentences)} (n={n} tokens): '
              f'{prog_errors} program errors, {time.time()-t_sent:.1f}s', flush=True)

    print(f'\nTotal scoring time: {time.time()-t0:.1f}s for {len(sentences)} sentence(s), '
          f'{len(programs)} programs, {n_layers*n_heads} heads')

    # ── best fit per head (mean IoU across sentences, require score on ALL sentences) ──
    best_rows = []
    for ht, prog_dict in all_head_scores.items():
        complete = {p: np.mean(v) for p, v in prog_dict.items() if len(v) == len(sentences)}
        if not complete:
            continue
        best_prog = max(complete, key=complete.get)
        best_rows.append({'layer': ht[0], 'head': ht[1], 'program': best_prog, 'best_iou': complete[best_prog]})

    import pandas as pd
    df = pd.DataFrame(best_rows).sort_values('best_iou', ascending=False).reset_index(drop=True)
    df['k'] = range(1, len(df) + 1)
    out_path = ROOT / 'results' / f'llama3b_new_programs_best_fits_n{args.n_sents}.csv'
    df.to_csv(out_path, index=False)

    print(f'\nSaved: {out_path}  ({len(df)} heads scored)')
    print(df['best_iou'].describe())
    print()
    print('distinct programs used as best-fit:', df['program'].nunique(), '/', len(programs))
    print(df['program'].value_counts().head(10))

if __name__ == '__main__':
    main()
