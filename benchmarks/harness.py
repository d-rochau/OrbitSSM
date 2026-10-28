
from __future__ import annotations
import argparse
import contextlib
import dataclasses
import importlib
import json
import os
import platform
import socket
import statistics
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import torch
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = Path(os.environ.get('HYBRID_SSM_RESULTS', Path.home() / 'hybrid-ssm-results'))
from .measure import BUDGETS, CANARY_DIM, CANARY_ITERS, CANARY_SPREAD_FLAG, DEEP_BUDGET, FULL_BUDGET, GROUP_MODES, IO_DTYPE, KERNEL_PRECISION, MODULE_PARAM_DTYPE, PAPER_BUDGET, PAPER_LABEL, PRECISIONS, SMOKE_BUDGET, Budget, _build_cell, _drop_canary_operands, _memory_and_extras, _peak_memory, _resolve_group_mode, _run_group, _run_group_rebuild, _run_group_resident, _time_events, _time_one, amp_ctx, canary_ms, precision_ctx, warm_cells
PAPER_PRECISIONS_DEFAULT: dict[str, str] = {'ssm_mqmkvar': 'bf16,tf32', 'ssm_lm_spine': 'bf16,tf32'}

@dataclass
class Cell:
    family: str
    arm: str
    entry_point: str
    shape: dict
    precision: str
    pass_: str
    build: Callable[[], Callable[[], Any]]
    grid: str = ''
    tier: str = 'auto'
    tags: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    group: str | None = None
    autocast: bool = False

    @property
    def resolved_tier(self) -> str:
        if self.tier != 'auto':
            return self.tier
        return 'do_bench' if self.pass_ == 'fwd' else 'events'

    @property
    def shape_key(self) -> str:
        return ','.join((f'{k}{v}' for k, v in sorted(self.shape.items()) if v is not None))

    @property
    def resolved_group(self) -> str:
        return self.group if self.group is not None else f'{self.grid}|{self.shape_key}'

    def key(self) -> str:
        tg = ','.join((f'{k}={v}' for k, v in sorted(self.tags.items())))
        return f'{self.family}/{self.arm}/{self.shape_key}/{self.precision}/{self.pass_}' + (f'/{tg}' if tg else '')

def group_key(grid: str, work: str, precision: str, pass_: str) -> str:
    return f'{grid}|{work}|{precision}|{pass_}'

def _first(y):
    return y[0] if isinstance(y, (tuple, list)) else y

def make_pass(call: Callable[[], Any], leaves: Sequence[torch.Tensor], pass_: str, *, amp: Callable[[], Any] | None=None) -> Callable[[], Any]:
    ctx = amp if amp is not None else contextlib.nullcontext
    if pass_ == 'fwd':

        def run_fwd():
            with torch.no_grad(), ctx():
                return call()
        return run_fwd
    if pass_ != 'fwd_bwd':
        raise ValueError(f'pass must be fwd|fwd_bwd, got {pass_!r}')
    with ctx():
        y = _first(call())
    if not y.requires_grad:
        raise RuntimeError('fwd_bwd cell produced a graph-less output — no leaf requires grad, so the backward would be free and the row a lie')
    _g = torch.Generator(device='cpu').manual_seed(24301)
    dy = torch.randn(y.shape, generator=_g, dtype=torch.float32).to(device=y.device, dtype=y.dtype)
    y.backward(dy)
    for t in leaves:
        if t is not None:
            t.grad = None
    del y
    leaves = [t for t in leaves if t is not None and t.requires_grad]

    def run_fb():
        for t in leaves:
            t.grad = None
        with ctx():
            result = call()
        _first(result).backward(dy)
        return result
    run_fb.validation_tensors = lambda: [t.grad for t in leaves if t.grad is not None]
    return run_fb

def cuda_buckets(fn: Callable[[], Any], iters: int=3) -> dict:
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    acc = {'fwd': 0.0, 'bwd': 0.0, 'state_passing': 0.0, 'glue': 0.0, 'total': 0.0}
    for ev in prof.key_averages():
        cu = float(getattr(ev, 'self_device_time_total', 0.0) or 0.0)
        if cu <= 0:
            continue
        key = ev.key.lower()
        if 'bwd' in key or 'backward' in key:
            bucket = 'bwd'
        elif 'state_passing' in key:
            bucket = 'state_passing'
        elif 'fwd' in key or 'forward' in key:
            bucket = 'fwd'
        else:
            bucket = 'glue'
        acc[bucket] += cu
        acc['total'] += cu
    return {k: v / iters / 1000.0 for k, v in acc.items()}

def git_meta() -> dict:
    return dict(git_sha='anonymous-release', git_branch='', git_dirty=False)

def _version(mod: str) -> str:
    try:
        return getattr(importlib.import_module(mod), '__version__', 'present')
    except Exception:
        return ''

def env_meta(device: str='cuda') -> dict:
    meta = dict(schema_version=SCHEMA_VERSION, run_id=uuid.uuid4().hex[:12], timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), host='anonymous', repo_path='.', torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(), python=platform.python_version(), torch=torch.__version__, triton=_version('triton'), cuda=torch.version.cuda, mamba_ssm=_version('mamba_ssm'), tilelang=_version('tilelang'), **git_meta(), slurm_job_id='')
    meta['device'] = str(device)
    if torch.device(device).type == 'cuda' and torch.cuda.is_available():
        index = torch.device(device).index
        props = torch.cuda.get_device_properties(torch.cuda.current_device() if index is None else index)
        meta.update(gpu=props.name, sm_count=props.multi_processor_count, capability=f'{props.major}.{props.minor}', gpu_mem_bytes=props.total_memory)
    else:
        meta.update(gpu=None, sm_count=None, capability=None, gpu_mem_bytes=None)
    return meta
_ENV_PIN_PREFIXES = ('SSD_MIMO_', 'UNIFIED_SSD_', 'HYBRID_MH_', 'TRIGGERED_MM_', 'MAMBA_M1_', 'TRITON_', 'PYTORCH_CUDA_ALLOC_CONF')

def env_pins() -> dict:
    return {k: ('<local-path>' if '/' in v or chr(92) in v else v) for k, v in sorted(os.environ.items()) if k.startswith(_ENV_PIN_PREFIXES)}

def autotune_snapshot() -> dict:
    snap: dict = {}
    try:
        from orbit_ssm.kernels.hybrid_mamba_mh import autotune as mh_at
        snap['mh_launcher'] = {str(k): str(v) for k, v in list(mh_at._CACHE.items())[:64]}
        snap['mh_autotune_enabled'] = bool(mh_at.enabled())
    except Exception:
        pass
    for modname in ('fused_fwd_kernel', 'fused_bwd_kernel', 'fwd_kernel', 'bwd_kernel'):
        try:
            mod = importlib.import_module(f'orbit_ssm.kernels.ssd_mimo.{modname}')
        except Exception:
            continue
        for name, obj in vars(mod).items():
            cache = getattr(obj, 'cache', None)
            if isinstance(cache, dict) and cache:
                snap[f'ssd_mimo.{modname}.{name}'] = {str(k): str(v) for k, v in list(cache.items())[:16]}
    return snap

def run_cells(cells: Sequence[Cell], budget: Budget, *, device: str='cuda', grid: str='', extra_meta: dict | None=None, post_cell: Callable[[Cell, Callable[[], Any]], dict] | None=None, group_mode: str='auto', progress: bool=True) -> list[dict]:
    meta = env_meta(device)
    meta['budget'] = dataclasses.asdict(budget)
    meta['grid'] = grid
    meta['env_pins'] = env_pins()
    if extra_meta:
        meta.update(extra_meta)
    meta['group_mode'] = group_mode
    meta['arm_order'] = 'counterbalanced: forward on even rounds, reversed on odd (ABBA)'
    meta['statistic'] = 'ms_median = median over rounds of the per-round value; per round: do_bench median (fwd) or CUDA-event mean over ev_iters (fwd_bwd)'
    groups: dict[str, list[Cell]] = {}
    for cell in cells:
        groups.setdefault(cell.resolved_group, []).append(cell)
    if progress:
        biggest = max((len(g) for g in groups.values()), default=0)
        print(f'  {len(cells)} cells in {len(groups)} comparison group(s), largest {biggest} cells', flush=True)
    results: dict[str, dict] = {}
    group_canaries: dict[str, list[float]] = {}
    all_canaries: list[float] = []
    for gname, gcells in groups.items():
        res, cans = _run_group(gname, gcells, budget, device, post_cell, progress, group_mode=group_mode)
        results[gname] = res
        group_canaries[gname] = cans
        all_canaries += cans
    run_spread = max(all_canaries) / min(all_canaries) if all_canaries else None
    if progress and run_spread is not None:
        note = '  <-- CONTENDED, discard' if run_spread > CANARY_SPREAD_FLAG else ''
        print(f'  canary median {statistics.median(all_canaries):.3f} ms  run spread {run_spread:.3f}x{note}', flush=True)
    autotune = autotune_snapshot()
    records: list[dict] = []
    for gname, gcells in groups.items():
        res = results[gname]
        cans = group_canaries[gname]
        spread = max(cans) / min(cans) if cans else None
        flagged = bool(spread is not None and spread > CANARY_SPREAD_FLAG)
        for cell in gcells:
            key = cell.key()
            vals = sorted(res['samples'].get(key, []))
            peak, base = res['peaks'].get(key) or (None, None)
            rec = dict(meta)
            rec.update(family=cell.family, arm=cell.arm, entry_point=cell.entry_point, cell_grid=cell.grid or grid, shape=dict(cell.shape), group=gname, precision=cell.precision, dtype=str(IO_DTYPE[cell.precision]).replace('torch.', ''), autocast=bool(cell.autocast), param_dtype=str(MODULE_PARAM_DTYPE).replace('torch.', '') if cell.autocast else None, kernel_precision=KERNEL_PRECISION[cell.precision], **{'pass': cell.pass_}, tier=cell.resolved_tier, tags=dict(cell.tags, execution=dict(cell.runtime)) if cell.runtime else dict(cell.tags), key=key, ms_median=statistics.median(vals) if vals else None, ms_min=vals[0] if vals else None, ms_max=vals[-1] if vals else None, samples=vals or None, rounds=len(vals), iters=budget.ev_iters if cell.resolved_tier in ('events', 'cpu_wall') else None, canary_ms_median=statistics.median(cans) if cans else None, canary_spread=spread, canary_spread_run=run_spread, canary_flagged=flagged, mem_delta_bytes=None if peak is None else peak - base, peak_mem_bytes=peak, mem_baseline_bytes=base, resolved_autotune=autotune, error=res['errors'].get(key))
            rec.update(res['extras'].get(key, {}))
            records.append(rec)
    return records

def write_jsonl(records: Iterable[dict], path: str | os.PathLike, append: bool=False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a' if append else 'w') as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + '\n')
    return out

def read_jsonl(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    files = sorted(p.glob('*.jsonl')) if p.is_dir() else [p]
    out: list[dict] = []
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out

def print_table(records: Sequence[dict]) -> None:
    print(f"\n{'family':<10}{'arm':<26}{'prec':>6} {'pass':>11}{'ms':>10}{'d-mem MiB':>11}  shape", flush=True)
    for r in records:
        ms = 'FAIL' if r['ms_median'] is None else f"{r['ms_median']:.3f}"
        mem = '-' if r.get('mem_delta_bytes') is None else f"{r['mem_delta_bytes'] / 2 ** 20:.1f}"
        sh = ' '.join((f'{k}{v}' for k, v in r['shape'].items() if v is not None))
        print(f"{r['family']:<10}{r['arm']:<26}{r['precision']:>6} {r['pass']:>11}{ms:>10}{mem:>11}  {sh}", flush=True)
        if r['error']:
            print(f"{'':>61}  ! {r['error'][:120]}", flush=True)

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--grid', default='smoke', help='named grid to run (--list-grids to enumerate)')
    parser.add_argument('--out', default=None, help='JSONL output path (default: $HYBRID_SSM_RESULTS/<grid>_<run_id>.jsonl)')
    parser.add_argument('--budget', default=None, choices=sorted(BUDGETS), help='timing budget; default smoke for the smoke grid, full otherwise')
    parser.add_argument('--rounds', type=int, default=None, help="override the budget's round count")
    parser.add_argument('--precisions', default=None, help='comma-separated subset of ieee,tf32,bf16')
    parser.add_argument('--passes', default=None, help='comma-separated subset of fwd,fwd_bwd')
    parser.add_argument('--arms', default=None, help='comma-separated exact `arm` names to keep (e.g. the real pair `ours_fair,m2_fair` of kernel_shared_set, or `SM-R1:m2exact,official_mamba2` of ssm_shared_set); default: every arm')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--group-mode', default='auto', choices=GROUP_MODES, help='how a comparison group is held: resident (all arms built up front), rebuild (one arm at a time, rebuilt every round — bounded memory, same interleaving), or auto (rebuild for model rows, resident otherwise)')
    parser.add_argument('--list-grids', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='enumerate the cells and exit without timing')
    parser.add_argument('--warm-only', action='store_true', help="B8: for every surviving cell, build() it and run its closure ONCE (one forward, or one forward+backward for a fwd_bwd cell) -- enough to trigger Triton's autotune/compile for that cell's kernel config and populate TRITON_CACHE_DIR -- then exit WITHOUT any do_bench/events timing loop, canary round, or JSONL write.  `sbatch_bench_paper.sh`'s serial warm phase uses this, on one GPU, before the 4-way split, so all 4 shards start from an already-populated shared cache instead of each independently paying first-touch compile for shapes another shard already built (B7 implementation point 4; implementationexperiments/B8_COMPACT_PROTOCOL_20260906.md).  A cell whose build/run raises is reported and skipped -- warming is best-effort, not a correctness gate.")
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--cells', default='full', choices=('full', 'paper'), help="'paper' restricts the grid to the named paper-quotable subset a module's PAPER_SETS defines for it (see benchmarks/README.md, 'paper set'); 'full' (default) runs every cell the grid builder produces.  Sugar: --grid paper:<grid> is the same as --grid <grid> --cells paper.")
    return parser

def resolve_budget(args) -> Budget:
    if args.budget:
        name = args.budget
    elif args.grid == 'smoke':
        name = 'smoke'
    elif getattr(args, 'cells', 'full') == 'paper':
        name = 'paper'
    else:
        name = 'full'
    budget = BUDGETS[name]
    if args.rounds:
        budget = dataclasses.replace(budget, rounds=args.rounds)
    return budget

def default_out(grid: str) -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return DEFAULT_RESULTS / f'{grid}_{stamp}.jsonl'

def filter_cells(cells: Sequence[Cell], args) -> list[Cell]:
    keep = list(cells)
    if args.precisions:
        want = {p.strip() for p in args.precisions.split(',') if p.strip()}
        keep = [c for c in keep if c.precision in want]
    if args.passes:
        want = {p.strip() for p in args.passes.split(',') if p.strip()}
        keep = [c for c in keep if c.pass_ in want]
    if getattr(args, 'arms', None):
        want = {p.strip() for p in args.arms.split(',') if p.strip()}
        keep = [c for c in keep if c.arm in want]
    return keep

def main_for(build_grid: Callable[[str, argparse.Namespace], list[Cell]], grids: Sequence[str], prog: str, add_args: Callable[[argparse.ArgumentParser], None] | None=None, extra_meta: Callable[[argparse.Namespace], dict] | None=None, post_cell: Callable[[argparse.Namespace], Callable[[Cell, Callable[[], Any]], dict] | None] | None=None, paper_filter: Callable[[str], Callable[[Cell], bool] | None] | None=None) -> None:
    parser = add_common_args(argparse.ArgumentParser(prog=prog))
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()
    if args.list_grids:
        print('\n'.join(grids))
        return
    if args.grid.startswith('paper:'):
        args.grid = args.grid[len('paper:'):]
        args.cells = 'paper'
    if args.grid not in grids:
        raise SystemExit(f"unknown grid {args.grid!r}; known: {', '.join(grids)}")
    if getattr(args, 'cells', 'full') == 'paper' and (not args.precisions):
        default_prec = PAPER_PRECISIONS_DEFAULT.get(args.grid)
        if default_prec:
            args.precisions = default_prec
    cells = filter_cells(build_grid(args.grid, args), args)
    if getattr(args, 'cells', 'full') == 'paper':
        predicate = paper_filter(args.grid) if paper_filter is not None else None
        if predicate is None:
            raise SystemExit(f"--cells paper: grid {args.grid!r} has no paper set defined in this module's PAPER_SETS -- refusing to silently run the full grid under a paper-sized budget")
        before = len(cells)
        cells = [c for c in cells if predicate(c)]
        print(f'{prog}: --cells paper narrowed grid={args.grid!r} {before} -> {len(cells)} cells', flush=True)
    if args.dry_run:
        groups: dict[str, list[Cell]] = {}
        for c in cells:
            groups.setdefault(c.resolved_group, []).append(c)
        for gname, gcells in groups.items():
            arms = sorted({c.arm for c in gcells})
            print(f"[{gname}]  {len(gcells)} cells, {len(arms)} arms: {', '.join(arms)}")
            for c in gcells:
                print(f'    {c.key()}')
        biggest = max((len(g) for g in groups.values()), default=0)
        fewest_arms = min((len({c.arm for c in g}) for g in groups.values()), default=0)
        print(f'\n{len(cells)} cells in {len(groups)} group(s); largest {biggest} cells; fewest arms in a group {fewest_arms}')
        return
    if not torch.cuda.is_available():
        raise SystemExit('benchmarks need a CUDA device')
    if args.warm_only:
        t0 = time.time()
        warmed, failed = warm_cells(cells)
        print(f'{prog}: --warm-only grid={args.grid} cells={len(cells)} warmed={warmed} failed={failed} in {time.time() - t0:.0f}s', flush=True)
        return
    budget = resolve_budget(args)
    out = Path(args.out) if args.out else default_out(args.grid)
    print(f'{prog}: grid={args.grid} cells={len(cells)} budget={budget.name} out={out}', flush=True)
    t0 = time.time()
    records = run_cells(cells, budget, device=args.device, grid=args.grid, extra_meta=extra_meta(args) if extra_meta else None, post_cell=post_cell(args) if post_cell else None, group_mode=args.group_mode)
    write_jsonl(records, out, append=args.append)
    print_table(records)
    ok = sum((1 for r in records if r['ms_median'] is not None))
    print(f'\nwrote {len(records)} records ({ok} measured, {len(records) - ok} skipped) to {out} in {time.time() - t0:.0f}s', flush=True)
