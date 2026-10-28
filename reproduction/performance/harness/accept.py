
import json
import math
import sys
GATE = 1.15

def load_keys(path):
    return [line.rstrip('\n') for line in open(path) if line.strip()]

def check(recs, keys):
    why = []
    if not recs:
        return ['no records']
    cell = max((r.get('canary_spread') or math.inf for r in recs))
    if not cell <= GATE:
        why.append(f'per-cell canary spread {cell:.3f} > {GATE}')
    got = [r.get('key') for r in recs]
    if len(got) != len(set(got)):
        why.append('duplicate cell keys')
    if keys is not None and sorted(got) != sorted(keys):
        miss, extra = (set(keys) - set(got), set(got) - set(keys))
        why.append(f'cell identities differ: {len(miss)} missing, {len(extra)} unexpected' + (f' (e.g. {sorted(miss)[:2]})' if miss else ''))
    shas = {r.get('git_sha') for r in recs}
    if len(shas) != 1 or None in shas:
        why.append(f'git_sha not unique/present: {sorted(map(str, shas))}')
    for r in recs:
        tag = f"{r.get('arm')}/{r.get('pass')}/{(r.get('tags') or {}).get('shape')}"
        if r.get('error'):
            why.append(f"{tag}: error {str(r['error'])[:160]}")
            continue
        ms = r.get('ms_median')
        if not (isinstance(ms, (int, float)) and math.isfinite(ms) and (ms > 0)):
            why.append(f'{tag}: non-finite timing {ms}')
        ex = (r.get('tags') or {}).get('execution') or {}
        if r.get('family') == 'scope' and (not ex.get('validation')):
            why.append(f'{tag}: no validation record')
        if str(r.get('arm', '')).split('@')[0] in ('SM-R1:m2exact', 'SM-R1:m3exact', 'HMM:m1exact') and r.get('family') == 'ssm' and (not ex.get('parity_gate')):
            why.append(f'{tag}: no parity gate record')
        pv = r.get('provenance')
        if not pv:
            why.append(f'{tag}: no provenance')
        elif pv.get('production_problem'):
            why.append(f"{tag}: production_problem {pv['production_problem']}")
    return why

def _groups(recs):
    out = {}
    for r in recs:
        out.setdefault(r.get('group'), []).append(r)
    return out

def select_cells(attempts, keys):
    key_group = {}
    for _, rs in attempts:
        for r in rs:
            key_group.setdefault(r.get('key'), r.get('group'))
    want = {}
    for k in keys:
        want.setdefault(key_group.get(k, f'<unknown group of {k}>'), set()).add(k)
    chosen, failed = ([], {})
    for g, gkeys in want.items():
        last = ['never produced']
        for name, rs in attempts:
            grs = [r for r in rs if r.get('group') == g]
            if not grs:
                continue
            why = check(grs, sorted(gkeys))
            why = [w for w in why if not w.startswith('git_sha')]
            if not why:
                for r in grs:
                    r['_attempt'] = name
                chosen += grs
                break
            last = why
        else:
            failed[g] = last
    shas = {r.get('git_sha') for r in chosen}
    if len(shas) > 1 or None in shas:
        failed['<all>'] = [f'git_sha not unique/present: {sorted(map(str, shas))}']
    return (chosen, failed)

def _load(f):
    try:
        return [json.loads(line) for line in open(f) if line.strip()]
    except FileNotFoundError:
        return []
if __name__ == '__main__' and sys.argv[1] in ('--cells', '--failed-groups'):
    kf, fs = (sys.argv[2], sys.argv[3:])
    fs = sorted(fs, key=lambda f: int(f.rsplit('_att', 1)[1].split('.')[0]))
    chosen, failed = select_cells([(f, _load(f)) for f in fs], load_keys(kf))
    if sys.argv[1] == '--failed-groups':
        print('\n'.join((g for g in failed if g != '<all>')))
        sys.exit(0)
    print(f'  cell selection over {len(fs)} attempt(s): {len(chosen)} records selected, {len(failed)} cell(s) never passed')
    for g, w in list(failed.items())[:10]:
        print('   !', g, '--', '; '.join(w[:2]))
    sys.exit(0 if not failed else 1)
if __name__ == '__main__':
    f, kf = (sys.argv[1], sys.argv[2])
    try:
        recs = [json.loads(line) for line in open(f) if line.strip()]
    except FileNotFoundError:
        print('  REJECT: no output')
        sys.exit(1)
    why = check(recs, load_keys(kf))
    run = max((r.get('canary_spread_run') or math.inf for r in recs)) if recs else math.inf
    cell = max((r.get('canary_spread') or math.inf for r in recs)) if recs else math.inf
    print(f'  records {len(recs)}  per-cell canary max {cell:.3f}  whole-run (drift, recorded) {run:.3f}  ' + ('ACCEPTED' if not why else f'REJECTED ({len(why)} reasons)'))
    for w in why[:10]:
        print('   !', w)
    sys.exit(0 if not why else 1)
