
import argparse, glob, json, os
from pathlib import Path
import numpy as np
ZS = ['lambada_openai', 'hellaswag', 'piqa_parquet', 'arc_easy', 'arc_challenge', 'winogrande', 'openbookqa']
NORM = {'hellaswag', 'arc_challenge'}
RECALL3 = ['swde', 'fda', 'squad_completion']
RECALL = RECALL3 + ['based_triviaqa', 'based_nq_1024', 'based_drop']
CHANCE = dict(lambada_openai=0.0, hellaswag=0.25, piqa_parquet=0.5, arc_easy=0.25, arc_challenge=0.25, winogrande=0.5, openbookqa=0.25)

def load(eval_dir, milestone, niah_variants=None):
    d = Path(eval_dir)
    x = {}
    for fname, tasks in ((f'lmeval_zeroshot_{milestone}.persample.json', ZS), (f'lmeval_recall_{milestone}.persample.json', RECALL), (f'lmeval_recall6_{milestone}.persample.json', RECALL)):
        p = d / fname
        if not p.exists():
            continue
        per = json.loads(p.read_text())
        for t in tasks:
            if t not in per:
                continue
            key = 'contains' if t in RECALL else 'acc_norm' if t in NORM else 'acc'
            ids = sorted(per[t], key=int)
            x[t] = (ids, np.array([per[t][i][key] for i in ids], float))
    for p in sorted(glob.glob(str(d / f'niah_{milestone}_*.json'))):
        row = next(iter(json.loads(Path(p).read_text()).values()))
        for cell, sc in row['_scores'].items():
            if niah_variants is None or cell.split('@')[0] in niah_variants:
                x['niah:' + cell] = (list(range(len(sc))), np.array(sc, float))
    return x

def paired(data, a, b, tasks, B, rng):
    obs = float(np.mean([data[a][t][1].mean() - data[b][t][1].mean() for t in tasks]))
    acc = np.zeros(B)
    for t in tasks:
        if data[a][t][0] != data[b][t][0]:
            raise ValueError(f'{t}: item ids differ between {a} and {b}')
        d = data[a][t][1] - data[b][t][1]
        acc += d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    acc /= len(tasks)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    p = min(1.0, 2 * min(float((acc <= 0).mean()), float((acc >= 0).mean())))
    return (obs, float(lo), float(hi), p)

def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='LABEL=EVAL_DIR (repeatable)')
    ap.add_argument('--pair', action='append', default=[], help='A:B comparison A - B (repeatable)')
    ap.add_argument('--milestone', default='1B')
    ap.add_argument('--niah-variants', default=None, help='comma list to restrict NIAH cells (default: all present)')
    ap.add_argument('--resamples', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', required=True, help='output directory (table.md, bootstrap.json)')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    nv = a.niah_variants.split(',') if a.niah_variants else None
    runs = dict((r.split('=', 1) for r in a.run))
    data = {k: load(v, a.milestone, nv) for k, v in runs.items()}
    common = sorted(set.intersection(*[set(v) for v in data.values()]), key=lambda t: (t.startswith('niah:'), t))
    zs = [t for t in ZS if t in common]
    rc = [t for t in RECALL if t in common]
    cells = sorted([t for t in common if t.startswith('niah:')], key=lambda c: (int(c.split('@')[1]), c))
    rc3 = [t for t in RECALL3 if t in common]
    meas = {t: [t] for t in zs + rc} | ({'ZS mean': zs} if zs else {}) | ({'recall mean': rc} if rc else {})
    if rc3 and len(rc) > len(rc3):
        meas['recall3 mean'] = rc3
    for L in sorted({int(c.split('@')[1]) for c in cells}):
        meas[f'NIAH@{L}'] = [c for c in cells if c.endswith(f'@{L}')]
    if cells:
        meas['NIAH mean'] = cells
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f'# Downstream scores and paired bootstrap ({a.milestone}; B={a.resamples}, seed {a.seed})', '', '| model | ' + ' | '.join(zs + rc + [c[5:] for c in cells]) + ' |', '|---|' + '---|' * len(zs + rc + cells)]
    for m in data:
        lines.append(f'| {m} | ' + ' | '.join((f'{100 * data[m][t][1].mean():.1f}' for t in zs + rc + cells)) + ' |')
    flags = [t for t in zs + rc + cells if all((abs(data[m][t][1].mean() - CHANCE.get(t, 0.0)) <= 2 * data[m][t][1].std(ddof=1) / np.sqrt(len(data[m][t][1])) for m in data))]
    lines += ['', 'Within 2 SE of chance for every model: ' + (', '.join(flags) or 'none'), '']
    pairs = [tuple(p.split(':')) for p in a.pair]
    res_json = {'n_items': {t: len(next(iter(data.values()))[t][1]) for t in common}, 'chance_flags': flags, 'paired': {}}
    if pairs:
        lines += ['| measure | ' + ' | '.join((f'{x}−{y}' for x, y in pairs)) + ' |', '|---|' + '---|' * len(pairs)]
        for name, ts in meas.items():
            r = [paired(data, x, y, ts, a.resamples, rng) for x, y in pairs]
            adj = holm(np.array([q[3] for q in r]))
            lines.append(f'| {name} | ' + ' | '.join((f"{100 * q[0]:+.1f} [{100 * q[1]:+.1f},{100 * q[2]:+.1f}]{('*' if p < 0.05 else '')}" for q, p in zip(r, adj))) + ' |')
            res_json['paired'][name] = {f'{x}-{y}': dict(mean=q[0], ci_lo=q[1], ci_hi=q[2], p=q[3], p_holm=float(p)) for (x, y), q, p in zip(pairs, r, adj)}
        lines += ['', 'Points (percent). * = Holm-adjusted p < 0.05 over the comparisons of that measure.']
    (out / 'table.md').write_text('\n'.join(lines) + '\n')
    (out / 'bootstrap.json').write_text(json.dumps(res_json, indent=2))
    print('\n'.join(lines))
if __name__ == '__main__':
    main()
