
import argparse, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import importlib.metadata as _md
_orig_version = _md.version

def _spoof_version(name):
    return '2025.10.22' if name == 'regex' else _orig_version(name)
_md.version = _spoof_version
DEFAULT_TASKS = ['lambada_openai', 'hellaswag', 'piqa_parquet', 'arc_easy', 'arc_challenge', 'winogrande', 'openbookqa']
TOKENIZER = 'meta-llama/Llama-3.1-8B'

def _tokenizer_path():
    d = os.environ.get('LM_TOKENIZER_DIR')
    if d and os.path.isfile(os.path.join(d, 'tokenizer.json')):
        return d
    hub = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub', 'models--meta-llama--Llama-3.1-8B', 'snapshots')
    if os.path.isdir(hub):
        snaps = sorted((os.path.join(hub, x) for x in os.listdir(hub)), key=os.path.getmtime)
        snaps = [x for x in snaps if os.path.isfile(os.path.join(x, 'tokenizer.json'))]
        if snaps:
            return snaps[-1]
    return TOKENIZER

def _load_model(run: Path, device, ckpt_glob: str='ckpt_step*.pt'):
    from experiments.lm.train.eval_ckpt import build_from_cfg_dict
    cks = sorted(run.glob(ckpt_glob), key=lambda q: int(q.stem.split('step')[-1]))
    if not cks:
        raise FileNotFoundError(f'no {ckpt_glob} in {run}')
    blob = torch.load(cks[-1], map_location='cpu', weights_only=False)
    model = build_from_cfg_dict(blob['model_cfg'], blob['model']).to(device)
    model.load_state_dict(blob['model'])
    model.eval()
    return (model, blob.get('step', -1), blob.get('tokens_seen', 0), cks[-1].name)

def _window(ctx_enc, cont_enc, max_length):
    seq = (ctx_enc + cont_enc)[-(max_length + 1):][:-1]
    return (seq, min(len(cont_enc), len(seq)))

def _ceil_to(n, q):
    return -(-n // q) * q

def _batches(requests, order, batch_size, max_length, fixed_rows, len_quantum):
    if not fixed_rows:
        return [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    out, cur, cur_b = ([], [], None)
    for i in order:
        n = len(_window(requests[i][1], requests[i][2], max_length)[0])
        b = _ceil_to(max(n, 1), len_quantum)
        if cur and (b != cur_b or len(cur) == batch_size):
            out.append(cur)
            cur = []
        cur.append(i)
        cur_b = b
    if cur:
        out.append(cur)
    return out

def score_loglikelihood(model_call, requests, *, batch_size, max_length, pad_id=0, disable_tqdm=True, fixed_rows=0, len_quantum=64):
    from tqdm import tqdm
    if fixed_rows and batch_size > fixed_rows:
        raise ValueError(f'batch_size {batch_size} > fixed_rows {fixed_rows}')
    res = []
    order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
    pbar = tqdm(total=len(requests), disable=disable_tqdm, desc='loglikelihood')
    for chunk in _batches(requests, order, batch_size, max_length, fixed_rows, len_quantum):
        inps, cont_lens = ([], [])
        for i in chunk:
            _, ctx_enc, cont_enc = requests[i]
            seq, n = _window(ctx_enc, cont_enc, max_length)
            inps.append(torch.tensor(seq, dtype=torch.long))
            cont_lens.append(n)
        maxlen = max((len(t) for t in inps))
        rows = len(inps)
        if fixed_rows:
            maxlen = _ceil_to(max(maxlen, 1), len_quantum)
            rows = fixed_rows
        batch = torch.full((rows, maxlen), pad_id, dtype=torch.long)
        for r, t in enumerate(inps):
            batch[r, :len(t)] = t
        logits = model_call(batch)
        logprobs = F.log_softmax(logits, dim=-1)
        for r, i in enumerate(chunk):
            _, ctx_enc, cont_enc = requests[i]
            n = cont_lens[r]
            end = len(inps[r])
            lp = logprobs[r, end - n:end, :]
            tgt = torch.tensor(cont_enc[len(cont_enc) - n:], dtype=torch.long, device=lp.device)
            greedy = (lp.argmax(dim=-1) == tgt).all().item()
            score = lp.gather(1, tgt.unsqueeze(-1)).sum().item()
            res.append((i, (score, bool(greedy))))
        pbar.update(len(chunk))
    pbar.close()
    res.sort(key=lambda p: p[0])
    return [r for _, r in res]

def reference_loglikelihood(model_call, ctx_enc, cont_enc, max_length):
    w = (ctx_enc + cont_enc)[-(max_length + 1):]
    first_cont = len(w) - len(cont_enc)
    with torch.no_grad():
        logits = model_call(torch.tensor([w[:-1]], dtype=torch.long))[0]
    logp = F.log_softmax(logits.double(), dim=-1)
    return sum((float(logp[k - 1, w[k]]) for k in range(max(1, first_cont), len(w))))

def _make_lm_class():
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.instance import Instance

    class MambaLMEvalWrapper(TemplateLM):

        def __init__(self, model, tokenizer, device, batch_size=8, max_length=2048, fixed_rows=0, len_quantum=64):
            super().__init__()
            self.fixed_rows = fixed_rows
            self.len_quantum = len_quantum
            self._model = model
            self.tokenizer = tokenizer
            self._device = device
            self.batch_size = batch_size
            self._max_length = max_length

        @property
        def eot_token_id(self):
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            return self._max_length

        @property
        def max_gen_toks(self):
            return 256

        def tok_encode(self, string, **kwargs):
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens, **kwargs):
            return self.tokenizer.decode(tokens)

        def _model_call(self, inps):
            cuda = self._device.type == 'cuda'
            inps = inps.to(self._device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=cuda):
                    return self._model(inps)['logits'].float()

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError('only loglikelihood tasks are supported by this wrapper')

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError('rolling loglikelihood (e.g. wikitext) is not supported')

        def _loglikelihood_tokens(self, requests, disable_tqdm=False):
            return score_loglikelihood(self._model_call, requests, batch_size=self.batch_size, max_length=self.max_length, pad_id=self.eot_token_id or 0, disable_tqdm=disable_tqdm, fixed_rows=self.fixed_rows, len_quantum=self.len_quantum)
    return MambaLMEvalWrapper
TASK_OVERRIDE_DIR = str(Path(__file__).resolve().parent / 'lmeval_tasks')

def _write_samples(out, n, path, run_name):
    lines = [f'# zero-shot samples: {run_name}  ({n} per task; ll = summed log-likelihood of the candidate)']
    for task, samples in (out.get('samples') or {}).items():
        lines.append(f"\n{'=' * 100}\n## {task}\n{'=' * 100}")
        for i, smp in enumerate(samples[:n]):
            args_ = smp.get('arguments') or []
            ctx = args_[0][0] if args_ and isinstance(args_[0], (list, tuple)) else args_[0].get('context', '') if args_ and isinstance(args_[0], dict) else ''
            conts = [a[1] if isinstance(a, (list, tuple)) else a.get('continuation', '') for a in args_]
            lls = [r[0] if isinstance(r, (list, tuple)) else r for r in smp.get('filtered_resps') or smp.get('resps') or []]
            lls = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in lls]
            pick = max(range(len(lls)), key=lambda k: lls[k]) if lls else None
            gold = smp.get('target')
            metric = {k: v for k, v in smp.items() if k in ('acc', 'acc_norm')}
            lines.append(f'\n--- {task} #{i}  gold={gold!r}  {metric}')
            lines.append('PROMPT: ' + ctx.strip().replace('\n', '\n        ')[-1200:])
            for k, (c, ll) in enumerate(zip(conts, lls)):
                mark = '<- model' if k == pick else ''
                lines.append(f'  [{k}] ll={ll:9.3f}  {c.strip()!r} {mark}')
    path.write_text('\n'.join(lines))
    print(f'wrote {path}')

def _task_manager():
    from lm_eval.tasks import TaskManager
    return TaskManager(include_path=TASK_OVERRIDE_DIR)

def download(tasks):
    import lm_eval
    tm = _task_manager()
    print(f"fetching datasets for: {', '.join(tasks)}")
    ok, bad = ([], [])
    for t in tasks:
        try:
            d = lm_eval.tasks.get_task_dict([t], tm)
            for name, obj in d.items():
                n = len(obj.eval_docs) if hasattr(obj, 'eval_docs') else '?'
                print(f'  OK   {name}: {n} eval docs')
            ok.append(t)
        except Exception as e:
            print(f'  FAIL {t}: {type(e).__name__}: {e}')
            bad.append(t)
    print(f'\n{len(ok)} ok, {len(bad)} failed' + (f" -> {', '.join(bad)}" if bad else ''))
    return bad

def self_check(lm, device, tol=0.02):
    import random
    rng = random.Random(0)
    vocab = lm._model.embedding.weight.shape[0]
    L = lm.max_length
    reqs = []
    shapes = [(ln, rng.randint(1, 5)) for ln in (7, 23, 60, 131, 264, 517, 33, 9) if ln < L]
    shapes += [(L + 37, 3), (L, 2), (5, L + 4)]
    for ln, nc in shapes:
        ctx = [rng.randrange(vocab) for _ in range(ln)]
        cont = [rng.randrange(vocab) for _ in range(nc)]
        reqs.append((('', ''), ctx, cont))
    old_bs = lm.batch_size
    pad_id = getattr(lm, 'eot_token_id', 0) or 0
    if getattr(lm, 'fixed_rows', 0) and lm.fixed_rows < 8:
        raise SystemExit(f'[self-check] --fixed-geometry {lm.fixed_rows} < 8: the check scores at batch 8')
    try:
        lm.batch_size = 1
        single = lm._loglikelihood_tokens(reqs, disable_tqdm=True)
        lm.batch_size = 8
        batched = lm._loglikelihood_tokens(reqs, disable_tqdm=True)
    finally:
        lm.batch_size = old_bs
    worst = max((abs(a[0] - b[0]) for a, b in zip(single, batched)))
    print(f'[self-check] batch 1 vs 8: max |dlogp| = {worst:.5f} over {len(reqs)} ragged/truncated requests (tol {tol})')
    ref_call = lm._model_call
    rows, q = (getattr(lm, 'fixed_rows', 0), getattr(lm, 'len_quantum', 64))
    if rows:

        def ref_call(b, _call=lm._model_call):
            x = torch.full((rows, -(-b.shape[1] // q) * q), pad_id, dtype=b.dtype)
            x[:1, :b.shape[1]] = b
            return _call(x)[:1, :b.shape[1]]
    ref = [reference_loglikelihood(ref_call, r[1], r[2], L) for r in reqs]
    worst_ref = max((abs(a[0] - b) for a, b in zip(single, ref)))
    print(f'[self-check] batch 1 vs unbatched reference: max |dlogp| = {worst_ref:.5f}')
    if worst > tol or worst_ref > tol:
        raise SystemExit(f'[self-check] FAILED: batch 1 vs 8 differs by {worst:.5f}, batch 1 vs the token-by-token reference by {worst_ref:.5f} (tol {tol}). Padding/indexing is wrong -- results would be silently corrupt. Refusing to score.')
    print('[self-check] OK')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='*', default=[])
    p.add_argument('--ckpt-glob', default='ckpt_step*.pt', help="which checkpoint file in each run dir to score (default: latest rolling; 'PRESERVED_1B_*.pt' / 'PRESERVED_2B_*.pt' / 'PRESERVED_10B_*.pt' for milestones)")
    p.add_argument('--tasks', default=','.join(DEFAULT_TASKS))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--limit', type=int, default=None, help='docs per task; smoke-testing only, do NOT report')
    p.add_argument('--out', default=None)
    p.add_argument('--samples', type=int, default=8, help='write this many per-task examples (prompt, candidates with log-likelihoods, model pick, gold) to <out>.samples.txt; 0 disables')
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--fixed-geometry', type=int, default=0, metavar='ROWS', help="batch-invariant scoring: every model call is [ROWS, len rounded up to --len-quantum] (ROWS >= --batch-size; dummy rows pad the batch), so a request's score does not depend on its batch mates or on --batch-size. 0 = off (dynamic shapes; bf16 scores then move by rounding with the batch, see score_loglikelihood)")
    p.add_argument('--len-quantum', type=int, default=64, help='length bucket of --fixed-geometry (pad tokens per row < this)')
    p.add_argument('--no-self-check', action='store_true', help='skip the batch-size invariance check (not recommended)')
    args = p.parse_args()
    tasks = [t for t in args.tasks.split(',') if t]
    if args.download_only:
        bad = download(tasks)
        sys.exit(1 if bad else 0)
    if not args.runs:
        p.error('--runs is required unless --download-only')
    import lm_eval
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_tokenizer_path())
    print(f'[tokenizer] {_tokenizer_path()}  vocab={len(tok)}', flush=True)
    Wrapper = _make_lm_class()
    dev = torch.device('cuda')
    all_res = {}
    for run in args.runs:
        run = Path(run)
        model, step, ntok, ckname = _load_model(run, dev, args.ckpt_glob)
        print(f'\n=== {run.name}  ({ckname}, step {step}, {ntok / 1000000.0:.0f}M tok) ===', flush=True)
        lm = Wrapper(model, tok, dev, batch_size=args.batch_size, max_length=args.max_length, fixed_rows=args.fixed_geometry, len_quantum=args.len_quantum)
        if not args.no_self_check:
            self_check(lm, dev)
        out = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=args.limit, task_manager=_task_manager(), bootstrap_iters=1000, log_samples=bool(args.samples and args.out))
        if args.samples and args.out:
            _write_samples(out, args.samples, Path(args.out).with_suffix('.samples.txt'), run.name)
        row = {}
        for tname, m in out['results'].items():
            acc, acc_se = (m.get('acc,none'), m.get('acc_stderr,none'))
            accn, accn_se = (m.get('acc_norm,none'), m.get('acc_norm_stderr,none'))
            row[tname] = {'acc': acc, 'acc_stderr': acc_se, 'acc_norm': accn, 'acc_norm_stderr': accn_se, 'metric': 'acc_norm' if tname in ('hellaswag', 'arc_challenge') and accn is not None else 'acc', 'value': accn if tname in ('hellaswag', 'arc_challenge') and accn is not None else acc, 'stderr': accn_se if tname in ('hellaswag', 'arc_challenge') and accn is not None else acc_se}
            v = row[tname]
            print(f"  {tname:<18} acc {(acc if acc is None else round(acc, 4))}  acc_norm {(accn if accn is None else round(accn, 4))}  -> table: {v['metric']} {v['value']:.4f} +/- {v['stderr']:.4f}" if v['value'] is not None else f'  {tname:<18} (no acc metric)')
        vals = [v['value'] for v in row.values() if v['value'] is not None]
        row['_mean'] = sum(vals) / len(vals) if vals else None
        row['_step'] = step
        row['_tokens'] = ntok
        all_res[run.name] = row
        print(f"  {'MEAN':<18} {'':<9} {row['_mean']:.4f}")
        del model, lm
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(all_res, indent=2))
        print(f'\nwrote {args.out}')
if __name__ == '__main__':
    main()
