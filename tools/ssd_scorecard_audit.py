"""Audit explicit defaults-only exports; never select the fastest attempt."""
import argparse
import csv
import json
from pathlib import Path
import statistics

SHAPES = {(6, 2048, 2048), (8, 2048, 2048), (6, 2048, 4096),
          (32, 2048, 2048), (2, 16384, 2048), (1, 32768, 4096), (8, 8192, 512)}


def legacy_filtered(row):
    return sum(v['ms'] for k, v in row['cuda_kernels'].items()
               if not k.startswith('void at::native::'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+', help='explicit independent defaults JSONLs')
    ap.add_argument('--out', required=True, help='TSV output; JSON verdict beside it')
    ap.add_argument('--gpus', type=int, default=2)
    a = ap.parse_args()
    rows, problems, seen, states = [], [], set(), set()
    for filename in a.inputs:
        raw = [json.loads(line) for line in Path(filename).read_text().splitlines()]
        for phase in ('fwd', 'fwd_bwd'):
            pair = [r for r in raw if r['pass'] == phase]
            official = [r for r in pair if r['arm'] == 'official']
            native = [r for r in pair if r['arm'] == 'calibrated_defaults']
            if len(official) != 1 or len(native) != 1:
                problems.append(f'{filename}/{phase}: expected one official/default pair')
                continue
            x, y = native[0], official[0]
            s = x['shape']
            geometry = (s['B'], s['L'], s['D'])
            key = (x['gpu'], x['precision'], phase, geometry)
            if key in seen:
                problems.append(f'duplicate attempt: {key}')
            seen.add(key)
            if s.get('N') != 128 or s.get('P') != 64 or s.get('R') != 1 or s.get('O') != 1:
                problems.append(f'wrong function geometry: {key}')
            if x.get('shape') != y.get('shape') or x.get('precision') != y.get('precision'):
                problems.append(f'unmatched pair: {key}')
            states.add((x.get('source_digest'), x.get('adapter_sha256')))
            if not x.get("source_digest") or not x.get("adapter_sha256"):
                problems.append(f'missing source/adapter digest: {filename}')
            route = x.get('dispatch_tail', [{}])[-1].get('measured_plan')
            if not route or not route.get('measurement'):
                problems.append(f'no calibrated dispatch evidence: {key}')
            if x.get('calibration') or y.get('calibration'):
                problems.append(f'calibration data is not independent scoring: {key}')
            if any(r.get('error') or r.get('canary_flagged') for r in (x, y)):
                problems.append(f'error/contended: {key}')
                continue
            if any(len(r.get('cuda_self_round_totals', [])) < 5 for r in (x, y)):
                problems.append(f'fewer than five complete GPU rounds: {key}')
                continue
            primary_x = statistics.median(x['cuda_self_round_totals'])
            primary_y = statistics.median(y['cuda_self_round_totals'])
            ratio = primary_x / primary_y
            elapsed_ratio = x['ms_median'] / y['ms_median']
            verdict = 'loss' if ratio > 1.02 else 'win' if ratio < .98 else 'tie'
            if ratio > 1.02 or elapsed_ratio > 1.02:
                problems.append(f'primary/elapsed loss: {key}')
            rows.append(dict(level='kernel+features', metric='median whole-round CUDA self including ATen',
                gpu=x['gpu'], job=x['slurm_job_id'], sha=x['git_sha'], precision=x['precision'],
                phase=phase, **s, ours_ms=primary_x, official_ms=primary_y, primary_ratio=ratio,
                elapsed_ours_ms=x['ms_median'], elapsed_official_ms=y['ms_median'],
                elapsed_ratio=elapsed_ratio, legacy_all_gpu_min_ratio=x['ms_cuda_self']/y['ms_cuda_self'],
                legacy_filtered_min_ratio=legacy_filtered(x)/legacy_filtered(y), primary_verdict=verdict,
                elapsed_verdict='loss' if elapsed_ratio > 1.02 else 'win' if elapsed_ratio < .98 else 'tie',
                plan_key=route.get('key') if route else '', raw=str(Path(filename).resolve())))
    devices = {k[0] for k in seen}
    if len(devices) != a.gpus:
        problems.append(f'expected {a.gpus} GPUs, found {len(devices)}')
    for gpu in devices:
        expected = {(gpu, precision, phase, geometry) for precision in ('bf16', 'ieee')
                    for phase in ('fwd', 'fwd_bwd') for geometry in SHAPES}
        if {k for k in seen if k[0] == gpu} != expected:
            problems.append(f'incomplete/extra geometries: {gpu}')
    if len(states) != 1:
        problems.append('scorecard mixes source/adapter states')
    out = Path(a.out)
    if rows:
        with out.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader(); writer.writerows(rows)
    verdict = dict(passed=not problems, rows=len(rows), source_states=sorted(states, key=str),
                   problems=problems, inputs=a.inputs)
    out.with_suffix('.json').write_text(json.dumps(verdict, indent=2)+'\n')
    print(json.dumps(verdict, indent=2))
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
