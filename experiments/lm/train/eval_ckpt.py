
import argparse, math, os, sys
from pathlib import Path
import torch
import importlib.metadata as _md
_orig_version = _md.version

def _spoof_version(name):
    if name == 'regex':
        return '2025.10.22'
    return _orig_version(name)
_md.version = _spoof_version
from dataclasses import fields
from experiments.lm.model.lm import MambaLMConfig, MambaLM
from experiments.lm.data.fineweb_dataset import FineWebTokens

@torch.no_grad()
def evaluate(model, val, device, micro_batch, n_batches, autocast_enabled):
    model.eval()
    losses = []
    g = torch.Generator().manual_seed(0)
    for _ in range(n_batches):
        x, y = val.get_batch(micro_batch, device=device, generator=g)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(x, targets=y)
        losses.append(out['loss'].detach().float())
    model.train()
    return torch.stack(losses).mean().item()

def build_from_cfg_dict(cfg_dict: dict, state_dict: dict) -> MambaLM:
    del state_dict
    valid = {f.name for f in fields(MambaLMConfig)}
    cfg_kw = {k: v for k, v in cfg_dict.items() if k in valid}
    return MambaLM(MambaLMConfig(**cfg_kw))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--tokens-dir', required=True)
    p.add_argument('--val-batches', type=int, default=200)
    p.add_argument('--micro-batch', type=int, default=6)
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--val-shards', type=int, default=2)
    p.add_argument('--split', choices=['val', 'test'], default='test', help='held-out shard set to evaluate: test = last held-out shard (the reported milestone number), val = the others')
    args = p.parse_args()
    dev = torch.device('cuda')
    val_ds = FineWebTokens(args.tokens_dir, seq_len=args.seq_len, split=args.split, val_shards=args.val_shards)
    print(f"{'run':<34}{'step':>6}{'tok':>8}{args.split + '_loss':>10}{args.split + '_ppl':>9}")
    print('-' * 67)
    for run in args.runs:
        run = Path(run)
        cks = sorted(run.glob('ckpt_step*.pt'), key=lambda q: int(q.stem.split('step')[-1]))
        if not cks:
            print(f'{run.name:<34}  (no ckpt)')
            continue
        ck = cks[-1]
        blob = torch.load(ck, map_location='cpu', weights_only=False)
        model = build_from_cfg_dict(blob['model_cfg'], blob['model']).to(dev)
        model.load_state_dict(blob['model'])
        vl = evaluate(model, val_ds, dev, args.micro_batch, args.val_batches, autocast_enabled=True)
        step = blob.get('step', -1)
        tok = blob.get('tokens_seen', 0)
        print(f'{run.name:<34}{step:>6}{tok / 1000000.0:>7.0f}M{vl:>10.4f}{math.exp(vl):>9.2f}')
        del model
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
