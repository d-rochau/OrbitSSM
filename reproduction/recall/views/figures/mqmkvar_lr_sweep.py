
import csv
import json
import math
import statistics
from mqmkvar_presentation import TIER_NAMES
from pathlib import Path
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'records/lr_sweep_table.json'
SEEDS = ('101', '202', '303')
LRS = ('1e-4', '5e-4', '1e-3', '2e-3')
GRID = ('5e-4', '1e-3', '2e-3')
LR_TEX = {'1e-4': '$10^{-4}$', '5e-4': '$5\\!\\times\\!10^{-4}$', '1e-3': '$10^{-3}$', '2e-3': '$2\\!\\times\\!10^{-3}$'}
LABELS = {'Attn': 'Attention', 'HM-Adyn': 'HM $A$+dynamic', 'HM-c25a': 'HM complex-$A$ mix', 'HM-c25d': 'HM complex-dynamic mix', 'HM-A': 'HM all-$A$ real', 'HM-Ac': 'HM all-$A$ complex', 'R2O2-c-ω/h': 'R2O2-c, head frequency', 'R2O2-r-nogate': 'R2O2-r, gates off', 'R2O2-c-nogate': 'R2O2-c, gates off'}

def load_sweep():
    rows = json.loads(SOURCE.read_text())['rows']
    assert len(rows) == 18
    for row in rows:
        for lr, cell in row['cells'].items():
            assert set(cell['seeds']) == set(SEEDS)
            values = [v['acc'] for v in cell['seeds'].values() if v['status'] == 'complete']
            assert len(values) == cell['n'] == cell['completed']
            assert cell['attempted'] == 3
            assert all((math.isfinite(v) and 0 <= v <= 100 for v in values))
            for key, val in [('mean', statistics.mean(values)), ('sd', statistics.stdev(values)), ('min', min(values)), ('max', max(values))]:
                assert math.isclose(cell[key], val, rel_tol=1e-10, abs_tol=1e-10), (row['abbr'], lr, key)
    with SOURCE.with_name('seeds.csv').open() as f:
        exported = {(r['arm'], r['lr'], r['seed']): r for r in csv.DictReader(f)}
    assert len(exported) == sum((3 * len(r['cells']) for r in rows))
    for row in rows:
        for lr, cell in row['cells'].items():
            for seed, value in cell['seeds'].items():
                other = exported[row['abbr'], lr, seed]
                assert other['status'] == value['status']
                assert not other['accuracy'] if value['acc'] is None else float(other['accuracy']) == float(f"{value['acc']:.3f}")
    return rows

def select_best(rows):
    selected = {}
    for row in rows:
        eligible = [(lr, c) for lr, c in row['cells'].items() if lr in GRID and c['completed'] == 3 and (c['n'] == 3) and all((c['seeds'][s]['status'] == 'complete' for s in SEEDS))]
        lr, cell = max(eligible, key=lambda x: (x[1]['mean'], x[0] == '1e-3', -float(x[0])))
        selected[row['abbr']] = (lr, cell)
    return selected

def render():
    rows = load_sweep()
    best = select_best(rows)
    head = ['\\toprule', 'Configuration & Peak LR & 101 & 202 & 303 & Mean $\\pm$ SD & Min--max & C/A \\\\', '\\midrule']
    lines = ['\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}', '\\begin{longtable}{@{}l c r r r r c c@{}}', '\\caption{Full insane-tier LR sweep (18 configurations): held-out token accuracy (percent, rounded to one decimal) per seed, mean $\\pm$ sample SD and range over completed seeds, and completed/attempted (C/A). Bold LR is the best eligible three-seed mean per configuration over the common grid $\\{5\\times10^{-4},10^{-3},2\\times10^{-3}\\}$; the $10^{-4}$ rows of attention and M3 are pre-registered extra cells, reported but not selected. --- denotes an unscheduled rate; D = no plateau escape (chance-level loss), then a dead-network collapse (final-RMSNorm overflow, zero gradient), reproduced from the checkpoint; fresh reruns of this seed escape nondeterministically and are reproduction evidence only.}\\label{tab:mq_lr_sweep}\\\\']
    lines[-1] = lines[-1].replace('insane-tier', TIER_NAMES['insane'] + '-tier')
    lines += head + ['\\endfirsthead'] + head + ['\\endhead', '\\bottomrule\\endfoot']
    records = []
    for row in rows:
        for lr in LRS:
            name = LABELS.get(row['abbr'], row['abbr'])
            rate = LR_TEX[lr]
            if lr == best[row['abbr']][0]:
                rate = '\\boldmath ' + rate
            c = row['cells'].get(lr)
            if c is None:
                values = ['---'] * 6
            else:
                values = []
                for seed in SEEDS:
                    v = c['seeds'][seed]
                    is_d = row['abbr'] == 'M1' and lr == '2e-3' and (seed == '303')
                    if is_d:
                        assert v['status'] in ('failed', 'diverged') and v['acc'] is None
                    values.append(f"{v['acc']:.1f}" if v['status'] == 'complete' else 'D' if is_d else v['status'])
                    records.append({'arm': row['abbr'], 'lr': lr, 'seed': seed, 'accuracy': v['acc'], 'source_status': v['status'], 'owner_divergence_note': is_d, 'root': v['origin'], 'trial': v['name'], 'gpu': v['gpu']})
                values += [f"${c['mean']:.1f}\\pm{c['sd']:.1f}$", f"{c['min']:.1f}--{c['max']:.1f}", f"{c['completed']}/{c['attempted']}"]
            lines.append(name + ' & ' + rate + ' & ' + ' & '.join(values) + ' \\\\')
            if row['abbr'] == 'M1' and lr == '2e-3':
                completed = [v['acc'] for v in c['seeds'].values() if v['status'] == 'complete']
                chance_mean = sum(completed) / 3
                assert round(chance_mean, 1) == round((sum(completed) + 100 / 4096) / 3, 1) == 48.9
                note = 'M1 at $2\\times10^{-3}$: ' + f'{chance_mean:.1f}' + ' (diverged seed counted as $0$) / ' + f"{c['mean']:.1f}" + ' (completed seeds, $2/3$). ' + 'Seed 303 did not escape (D); the rate is ineligible.'
                lines[-1] = name + '\\footnote{' + note + '}' + ' & ' + rate + ' & ' + ' & '.join(values) + ' \\\\'
        lines.append('\\addlinespace[2pt]')
    lines += ['\\end{longtable}', '\\endgroup']
    (HERE.parent / 'tables/mqmkvar_lr_sweep.tex').write_text('\n'.join(lines) + '\n')
    with (HERE / 'data/lr_sweep_seeds.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    print(f"Validated {sum((len(r['cells']) for r in rows))} LR cells / {len(records)} seed records; M1 2e-3 excluded from selection.")
    for arm, (lr, cell) in best.items():
        print(arm, lr, f"{cell['mean']:.6f}")
if __name__ == '__main__':
    render()
