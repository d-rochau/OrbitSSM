
import argparse
import glob
import itertools
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq
DEFAULT_TOKENIZER = 'meta-llama/Llama-3.1-8B'
TEXT_COLUMN = 'text'
READ_BATCH = 1024
_TOK = None

def _init_worker(tokenizer_name):
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_name)
    assert _TOK.bos_token_id is not None, 'tokenizer has no BOS; expected Llama-3.1'

def _tokenize_one(args):
    src_path, out_dir = args
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_name = f'tok_{stem}.bin'
    out_path = os.path.join(out_dir, out_name)
    tmp_path = out_path + '.tmp'
    ntokens = 0
    pf = pq.ParquetFile(src_path)
    with open(tmp_path, 'wb') as f:
        for batch in pf.iter_batches(batch_size=READ_BATCH, columns=[TEXT_COLUMN]):
            texts = batch.column(0).to_pylist()
            enc = _TOK(texts, add_special_tokens=True)['input_ids']
            flat = list(itertools.chain.from_iterable(enc))
            if not flat:
                continue
            arr = np.asarray(flat, dtype=np.uint32)
            arr.tofile(f)
            ntokens += arr.size
    os.replace(tmp_path, out_path)
    return (out_name, ntokens)

def discover_parquet(src):
    files = sorted(glob.glob(os.path.join(src, '**', '*.parquet'), recursive=True))
    if not files:
        sys.exit(f'no .parquet found under {src}')
    return files

def merge_manifests(out_dir):
    parts = sorted(glob.glob(os.path.join(out_dir, 'manifest_w*.json')))
    merged = []
    for p in parts:
        merged.extend(json.load(open(p)))
    merged.sort(key=lambda r: r['file'])
    total = sum((r['ntokens'] for r in merged))
    json.dump({'shards': merged, 'total_tokens': total, 'dtype': 'uint32'}, open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=2)
    print(f'merged {len(parts)} partial manifests: {len(merged)} shards, {total:,} tokens -> manifest.json')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.expandvars('./data/fineweb-edu/raw/sample/100BT'))
    ap.add_argument('--out', default=os.path.expandvars('./data/fineweb-edu/llama3_tokens'))
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    ap.add_argument('--procs', type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument('--worker', type=int, default=0, help='SLURM rank (0-indexed)')
    ap.add_argument('--num-workers', type=int, default=1, help='total SLURM ranks')
    ap.add_argument('--limit-files', type=int, default=0, help='smoke: only first N parquet files')
    ap.add_argument('--merge-manifests', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.merge_manifests:
        return merge_manifests(args.out)
    files = discover_parquet(args.src)
    if args.limit_files:
        files = files[:args.limit_files]
    my_files = files[args.worker::args.num_workers]
    todo = [f for f in my_files if not os.path.exists(os.path.join(args.out, f'tok_{os.path.splitext(os.path.basename(f))[0]}.bin'))]
    print(f'[worker {args.worker}/{args.num_workers}] {len(my_files)} files assigned, {len(todo)} to do, procs={args.procs}, tokenizer={args.tokenizer}', flush=True)
    results = []
    if todo:
        tasks = [(f, args.out) for f in todo]
        with Pool(args.procs, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
            for i, (name, n) in enumerate(pool.imap_unordered(_tokenize_one, tasks), 1):
                results.append({'file': name, 'ntokens': n})
                print(f'[worker {args.worker}] {i}/{len(todo)} {name} {n:,} tok', flush=True)
    for f in my_files:
        name = f'tok_{os.path.splitext(os.path.basename(f))[0]}.bin'
        p = os.path.join(args.out, name)
        if os.path.exists(p) and (not any((r['file'] == name for r in results))):
            results.append({'file': name, 'ntokens': os.path.getsize(p) // 4})
    man = os.path.join(args.out, f'manifest_w{args.worker}.json')
    json.dump(results, open(man, 'w'), indent=2)
    total = sum((r['ntokens'] for r in results))
    print(f'[worker {args.worker}] wrote {man}: {len(results)} shards, {total:,} tokens', flush=True)
    if args.num_workers == 1:
        merge_manifests(args.out)
if __name__ == '__main__':
    main()
