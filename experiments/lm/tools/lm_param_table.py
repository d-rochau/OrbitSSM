
from __future__ import annotations
import argparse
import sys
from dataclasses import replace
from pathlib import Path
import torch
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT) + '/src')
import mamba_ssm.modules.mamba_simple
import mamba_ssm.modules.mamba2
import mamba_ssm
from experiments.lm.model.lm import MambaLM, MambaLMConfig, NAMED_SSM, REGISTERED_MIXERS
LLAMA31_VOCAB = 128256
ARMS = ('official_mamba1', 'official_mamba2', 'official_mamba3', 'official_mamba3_mimo', 'llama3_attention')
DEFAULT_ATTENTION_MLP = 2864

def count(mixer: str, *, d_model: int, n_layers: int, mlp_dim: int, d_state: int, vocab: int) -> tuple[int, int]:
    cfg = MambaLMConfig(vocab_size=vocab, n_layers=n_layers, d_model=d_model, mlp_dim=mlp_dim, d_state=d_state, ssm=NAMED_SSM.get(mixer, mixer))
    with torch.device('meta'):
        model = MambaLM(cfg)
    return (model.num_params(), model.num_params(non_embedding=True))

def _mlp_for(mixer: str, mlp: int, attention_mlp: int) -> int:
    return attention_mlp if mixer == 'llama3_attention' else mlp

def match(mixer: str, *, anchor: int, depths, d_model: int, mlp_dim: int, d_state: int, vocab: int, band: float, max_trim: float) -> dict:
    depths = sorted(depths)
    counts = {d: count(mixer, d_model=d_model, n_layers=d, mlp_dim=mlp_dim, d_state=d_state, vocab=vocab)[0] for d in depths}
    under = [d for d in depths if counts[d] <= anchor]
    if not under:
        return {'ok': False, 'why': 'every depth overshoots the anchor', 'depth': depths[0], 'mlp_dim': mlp_dim, 'params': counts[depths[0]]}
    best = max(under)
    if counts[best] >= (1.0 - band) * anchor:
        return {'ok': True, 'depth': best, 'mlp_dim': mlp_dim, 'params': counts[best], 'trim': 0.0}
    over = [d for d in depths if counts[d] > anchor]
    if not over:
        return {'ok': False, 'why': f'no depth within {band:.0%} of the anchor', 'depth': best, 'mlp_dim': mlp_dim, 'params': counts[best]}
    depth = min(over)
    floor_mlp = int(round(mlp_dim * (1.0 - max_trim)))
    fits = None
    for trial in range(mlp_dim, floor_mlp - 1, -1):
        p = count(mixer, d_model=d_model, n_layers=depth, mlp_dim=trial, d_state=d_state, vocab=vocab)[0]
        if p <= anchor:
            fits = (trial, p)
            break
    if fits is None:
        return {'ok': False, 'why': f'depth {depth} still overshoots after a {max_trim:.0%} MLP trim', 'depth': best, 'mlp_dim': mlp_dim, 'params': counts[best]}
    trial, p = fits
    return {'ok': p >= (1.0 - band) * anchor, 'depth': depth, 'mlp_dim': trial, 'params': p, 'trim': 1.0 - trial / mlp_dim, 'why': None if p >= (1.0 - band) * anchor else f'lands {100 * (p - anchor) / anchor:+.2f}% even after the trim'}

def run_width(d_model: int, args) -> None:
    anchor_mlp = _mlp_for(args.anchor, args.mlp, args.attention_mlp)
    anchor_total, anchor_nonemb = count(args.anchor, d_model=d_model, n_layers=args.anchor_layers, mlp_dim=anchor_mlp, d_state=args.d_state, vocab=args.vocab)
    print()
    print('=' * 78)
    print(f'd_model {d_model}   d_state {args.d_state}   vocab {args.vocab}')
    print(f'ANCHOR: {args.anchor} @ {args.anchor_layers} layers, mlp {anchor_mlp} -> {anchor_total / 1000000.0:.2f}M total ({anchor_nonemb / 1000000.0:.2f}M non-emb)')
    print(f'band: at most {args.band:.0%} BELOW the anchor, never above')
    print('=' * 78)
    for mixer in args.mixers:
        mlp = _mlp_for(mixer, args.mlp, args.attention_mlp)
        print()
        print(f'--- {mixer}  (mlp {mlp}) ' + '-' * max(0, 48 - len(mixer)))
        print(f"{'depth':>6}{'total':>12}{'non-emb':>12}{'vs anchor':>12}   note")
        for depth in args.depths:
            total, nonemb = count(mixer, d_model=d_model, n_layers=depth, mlp_dim=mlp, d_state=args.d_state, vocab=args.vocab)
            delta = (total - anchor_total) / anchor_total
            if total > anchor_total:
                note = 'over'
            elif delta >= -args.band:
                note = 'IN BAND'
            else:
                note = ''
            print(f'{depth:>6}{total / 1000000.0:>11.2f}M{nonemb / 1000000.0:>11.2f}M{100 * delta:>11.2f}%   {note}')
    print()
    print(f'LAUNCH TABLE (d_model {d_model}, anchor {anchor_total / 1000000.0:.2f}M)')
    print(f"{'arm':<22}{'--n-layers':>11}{'--mlp-dim':>11}{'total':>12}{'vs anchor':>11}   disclosure")
    for mixer in args.mixers:
        mlp = _mlp_for(mixer, args.mlp, args.attention_mlp)
        res = match(mixer, anchor=anchor_total, depths=args.depths, d_model=d_model, mlp_dim=mlp, d_state=args.d_state, vocab=args.vocab, band=args.band, max_trim=args.max_trim)
        delta = 100 * (res['params'] - anchor_total) / anchor_total
        if not res['ok']:
            disclosure = f"NO MATCH: {res['why']}"
        elif res.get('trim'):
            disclosure = f"MLP trimmed {100 * res['trim']:.1f}% (disclose)"
        else:
            disclosure = ''
        print(f"{mixer:<22}{res['depth']:>11}{res['mlp_dim']:>11}{res['params'] / 1000000.0:>11.2f}M{delta:>10.2f}%   {disclosure}")

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--d-model', type=int, nargs='+', default=[1024, 1536])
    p.add_argument('--depths', type=int, nargs='+', default=list(range(22, 31)))
    p.add_argument('--mlp', type=int, default=2048, help='d_intermediate for every arm but attention')
    p.add_argument('--attention-mlp', type=int, default=DEFAULT_ATTENTION_MLP, help='d_intermediate for the llama3_attention reference arm')
    p.add_argument('--d-state', type=int, default=128)
    p.add_argument('--vocab', type=int, default=LLAMA31_VOCAB)
    p.add_argument('--anchor', default='official_mamba3', choices=ARMS, help='the arm every other arm is matched to')
    p.add_argument('--anchor-layers', type=int, default=29, help='depth of the anchor arm (500M class: 29; 1B class: 28)')
    p.add_argument('--mixers', nargs='+', default=list(ARMS), choices=list(ARMS) + list(NAMED_SSM), help='registered reference arms and/or the named HybridSSM arms (experiments.lm.model.lm.NAMED_SSM: C6-C9)')
    p.add_argument('--band', type=float, default=0.01, help='how far BELOW the anchor an arm may land (default 1%%)')
    p.add_argument('--max-trim', type=float, default=0.05, help='largest d_intermediate trim allowed (default 5%%)')
    args = p.parse_args()
    missing = [m for m in args.mixers if m not in REGISTERED_MIXERS and m not in NAMED_SSM]
    if missing:
        raise SystemExit(f'not registered on CausalLM: {missing}')
    for d_model in args.d_model:
        run_width(d_model, args)
if __name__ == '__main__':
    main()
