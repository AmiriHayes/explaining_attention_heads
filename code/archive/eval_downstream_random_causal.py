"""
eval_downstream_random_causal.py
=================================
Same as eval_downstream_random.py (single random program applied to every
replaced head, multi-seed) but generalized to TinyLlama and Llama-3.2-3B via
--model. See eval_downstream_fixed_causal.py for the shared model-loading
approach.

Usage:
  python code/eval_downstream_random_causal.py --model tinyllama --levels 0.30 0.50 --seeds 1 2 3 --n 200
  python code/eval_downstream_random_causal.py --model llama3b   --levels 0.30 0.50 --seeds 1 2 3 --n 200
"""
import argparse, importlib, inspect, json, pathlib, random, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
RES_DIR  = ROOT / 'results'
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

MAX_SEQ_LEN = 512

MODEL_CFG = {
    'tinyllama': {
        'model_id': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'total_heads': 704,
        'use_float16': False,
        'fixed_module': 'fixed_attention_tinyllama',
        'programs_module': 'tinyllama_programs',
        'smart_csv': 'tinyllama_smart.csv',
    },
    'llama3b': {
        'model_id': 'meta-llama/Llama-3.2-3B',
        'total_heads': 672,
        'use_float16': True,
        'fixed_module': 'fixed_attention_llama3b',
        'programs_module': 'llama3b_programs',
        'smart_csv': 'llama3b_smart.csv',
    },
}

def load_programs(programs_module):
    mod = importlib.import_module(programs_module)
    return {f.__name__: f for _, f in inspect.getmembers(mod, inspect.isfunction)
            if not f.__name__.startswith('_')}

def parse_head(h):
    return tuple(map(int, str(h).strip('()').split(',')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_CFG.keys()))
    ap.add_argument('--levels', type=float, nargs='+', default=[0.30, 0.50])
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    args = ap.parse_args()
    cfg = MODEL_CFG[args.model]

    fixed_mod = importlib.import_module(cfg['fixed_module'])
    install, verify_identity = fixed_mod.install, fixed_mod.verify_identity

    prog_lookup = load_programs(cfg['programs_module'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg['use_float16']:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.float16, device_map='auto', attn_implementation='eager'
        ).eval()
        device = next(model.parameters()).device
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], attn_implementation='eager'
        ).to(device).eval()
    print(f'[{args.model}] loaded on {device}, dtype={next(model.parameters()).dtype}')

    active_sent = ['']
    mat_cache = {}

    def get_mat(prog_name, sentence):
        key = (prog_name, sentence)
        if key not in mat_cache:
            p = prog_lookup.get(prog_name)
            try:
                mat_cache[key] = np.asarray(p(sentence, tokenizer)[1]) if p else None
            except Exception:
                mat_cache[key] = None
        return mat_cache[key]

    def score_completion(context, completion):
        sent = context + ' ' + completion
        active_sent[0] = sent
        enc   = tokenizer(sent, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).to(device)
        ctx_l = tokenizer(context, return_tensors='pt', truncation=True,
                           max_length=MAX_SEQ_LEN)['input_ids'].shape[1]
        with torch.no_grad():
            logits = model(**enc).logits
        lp  = F.log_softmax(logits[0].float(), dim=-1)
        ids = enc['input_ids'][0]
        scores = [lp[i - 1, ids[i]].item() for i in range(ctx_l, len(ids))]
        return float(np.mean(scores)) if scores else -np.inf

    def evaluate_hellaswag(n):
        ds = load_dataset('Rowan/hellaswag', split='validation')
        total = min(n, len(ds))
        outcomes = []
        for ex in ds.select(range(total)):
            pred = int(np.argmax([score_completion(ex['ctx'], e) for e in ex['endings']]))
            outcomes.append(int(pred == int(ex['label'])))
        return float(np.mean(outcomes)), outcomes

    def evaluate_piqa(n):
        ds = []
        with open(DATA_DIR / 'evaluations' / 'piqa_val.jsonl') as f:
            for line in f:
                ds.append(json.loads(line))
                if len(ds) >= n:
                    break
        outcomes = []
        for ex in ds:
            pred = int(np.argmax([score_completion(ex['goal'], ex[f'sol{i+1}']) for i in range(2)]))
            outcomes.append(int(pred == int(ex['label'])))
        return float(np.mean(outcomes)), outcomes

    def evaluate_sciq(n):
        ds = load_dataset('allenai/sciq', split='validation')
        total = min(n, len(ds))
        outcomes = []
        for ex in ds.select(range(total)):
            choices = [ex['distractor1'], ex['distractor2'], ex['distractor3'], ex['correct_answer']]
            pred = int(np.argmax([score_completion(ex['question'], c) for c in choices]))
            outcomes.append(int(pred == 3))
        return float(np.mean(outcomes)), outcomes

    def evaluate_arc_easy(n):
        ds = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='validation')
        outcomes = []
        for ex in ds.select(range(min(n, len(ds)))):
            labels = ex['choices']['label']
            if ex['answerKey'] not in labels:
                continue
            idx = labels.index(ex['answerKey'])
            pred = int(np.argmax([score_completion(ex['question'], c) for c in ex['choices']['text']]))
            outcomes.append(int(pred == idx))
        return (float(np.mean(outcomes)) if outcomes else 0.0), outcomes

    def evaluate_social_iqa(n):
        ds = load_dataset('lighteval/siqa', split='validation')
        outcomes = []
        for ex in ds.select(range(min(n, len(ds)))):
            lbl = int(ex['label']) - 1
            prompt  = ex['context'].strip() + ' ' + ex['question'].strip()
            choices = [ex['answerA'], ex['answerB'], ex['answerC']]
            pred = int(np.argmax([score_completion(prompt, c) for c in choices]))
            outcomes.append(int(pred == lbl))
        return (float(np.mean(outcomes)) if outcomes else 0.0), outcomes

    def evaluate_copa(n):
        ds = load_dataset('aps/super_glue', 'copa', split='validation')
        outcomes = []
        for ex in ds.select(range(min(n, len(ds)))):
            conn = 'because' if ex['question'] == 'cause' else 'so'
            ctx  = ex['premise'].rstrip('.') + f' {conn}'
            pred = int(np.argmax([score_completion(ctx, ex[f'choice{i+1}']) for i in range(2)]))
            outcomes.append(int(pred == ex['label']))
        return (float(np.mean(outcomes)) if outcomes else 0.0), outcomes

    BENCHMARKS = [
        ('HellaSwag',  evaluate_hellaswag),
        ('PIQA',       evaluate_piqa),
        ('SciQ',       evaluate_sciq),
        ('ARC-Easy',   evaluate_arc_easy),
        ('Social IQA', evaluate_social_iqa),
        ('COPA',       evaluate_copa),
    ]

    smart_df = pd.read_csv(RES_DIR / 'replacement_run' / cfg['smart_csv']).sort_values('k').reset_index(drop=True)
    unique_progs = sorted(smart_df['program'].unique())

    all_results = []
    for seed in args.seeds:
        rng = random.Random(seed)
        random_prog = rng.choice(unique_progs)
        print(f'--- [{args.model}] seed {seed}: chosen random program (of {len(unique_progs)} unique): {random_prog} ---\n')

        results = []
        individual = {}
        for pct in args.levels:
            target_heads = int(cfg['total_heads'] * pct)
            df = smart_df.head(target_heads)
            assignment = {parse_head(row.head): random_prog for row in df.itertuples()}

            print(f'=== [{args.model}] seed {seed}: {int(pct*100)}% replacement: '
                  f'{target_heads}/{cfg["total_heads"]} heads, ALL assigned "{random_prog}" ===')
            state   = {'assignment': assignment, 'pattern': get_mat, 'sentence': active_sent}
            restore = install(model, state)

            row = {'model': args.model, 'seed': seed, 'replacement_pct': int(pct * 100),
                   'n_heads': target_heads, 'random_program': random_prog}
            pct_key = int(pct * 100)
            individual[pct_key] = {}
            for name, fn in BENCHMARKS:
                acc, outcomes = fn(args.n)
                std = float(np.std(outcomes, ddof=1)) if len(outcomes) > 1 else 0.0
                se  = std / np.sqrt(len(outcomes)) if outcomes else 0.0
                row[name] = acc
                row[f'{name}_std'] = std
                row[f'{name}_se']  = se
                row[f'{name}_n']   = len(outcomes)
                individual[pct_key][name] = outcomes
                print(f'  {name:12s}: {acc:.4f}  (empirical std={std:.4f}, se={se:.4f}, n={len(outcomes)})')
            restore()
            results.append(row)
            all_results.append(row)
            print()

        seed_df   = pd.DataFrame(results)
        seed_path = RES_DIR / 'replacement_run' / f'{args.model}_downstream_random_30_50_seed{seed}.csv'
        if seed_path.exists():
            prior = pd.read_csv(seed_path)
            seed_df = pd.concat([prior[~prior['replacement_pct'].isin(seed_df['replacement_pct'])], seed_df],
                                 ignore_index=True).sort_values('replacement_pct').reset_index(drop=True)
        seed_df.to_csv(seed_path, index=False)

        indiv_path = RES_DIR / 'replacement_run' / f'{args.model}_downstream_random_30_50_seed{seed}_individual.json'
        if indiv_path.exists():
            with open(indiv_path) as f:
                prior_indiv = json.load(f)
            prior_indiv.update(individual)
            individual = prior_indiv
        with open(indiv_path, 'w') as f:
            json.dump(individual, f)
        print(f'Saved: {seed_path}\nSaved individual responses: {indiv_path}\n')

    out_df   = pd.DataFrame(all_results)
    out_path = RES_DIR / 'replacement_run' / f'{args.model}_downstream_random_30_50.csv'
    if out_path.exists():
        prior = pd.read_csv(out_path)
        key = list(zip(prior['seed'], prior['replacement_pct']))
        new_key = set(zip(out_df['seed'], out_df['replacement_pct']))
        keep_prior = prior[[k not in new_key for k in key]]
        out_df = pd.concat([keep_prior, out_df], ignore_index=True) \
                   .sort_values(['seed', 'replacement_pct']).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f'Saved combined per-seed results: {out_path}')
    print(out_df.to_string(index=False))

    bench_names = [n for n, _ in BENCHMARKS]
    agg_rows = []
    for pct in sorted(out_df['replacement_pct'].unique()):
        sub = out_df[out_df['replacement_pct'] == pct]
        agg = {'replacement_pct': int(pct), 'n_seeds': len(sub)}
        for name in bench_names:
            vals = sub[name].values
            agg[f'{name}_mean'] = float(np.mean(vals))
            agg[f'{name}_cross_seed_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
    agg_df   = pd.DataFrame(agg_rows)
    agg_path = RES_DIR / 'replacement_run' / f'{args.model}_downstream_random_30_50_seed_aggregate.csv'
    agg_df.to_csv(agg_path, index=False)
    print(f'\nSaved cross-seed aggregate: {agg_path}')
    print(agg_df.to_string(index=False))


if __name__ == '__main__':
    main()
