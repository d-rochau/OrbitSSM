
from __future__ import annotations
import contextlib
import dataclasses
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import torch
CANARY_DIM = 4096
CANARY_ITERS = 20
CANARY_SPREAD_FLAG = 1.15
_CANARY_OPERANDS: dict[str, torch.Tensor] = {}

def canary_ms(device: str='cuda') -> float:
    if device not in _CANARY_OPERANDS:
        a = torch.randn(CANARY_DIM, CANARY_DIM, device=device)
        b = torch.randn(CANARY_DIM, CANARY_DIM, device=device)
        _CANARY_OPERANDS[device] = (a, b)
    a, b = _CANARY_OPERANDS[device]
    with precision_ctx('ieee'):
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(CANARY_ITERS):
            a @ b
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / CANARY_ITERS * 1000.0

@dataclass(frozen=True)
class Budget:
    name: str
    rounds: int
    do_bench_warmup: int
    do_bench_rep: int
    ev_warmup: int
    ev_iters: int
    cuda_self: bool = True
SMOKE_BUDGET = Budget('smoke', rounds=3, do_bench_warmup=10, do_bench_rep=50, ev_warmup=3, ev_iters=10)
FULL_BUDGET = Budget('full', rounds=3, do_bench_warmup=50, do_bench_rep=500, ev_warmup=8, ev_iters=30)
DEEP_BUDGET = Budget('deep', rounds=5, do_bench_warmup=50, do_bench_rep=500, ev_warmup=10, ev_iters=50)
PAPER_BUDGET = Budget('paper', rounds=3, do_bench_warmup=50, do_bench_rep=150, ev_warmup=8, ev_iters=10, cuda_self=True)
BUDGETS = {b.name: b for b in (SMOKE_BUDGET, FULL_BUDGET, DEEP_BUDGET, PAPER_BUDGET)}
PRECISIONS = ('ieee', 'tf32', 'bf16')
IO_DTYPE = {'ieee': torch.float32, 'tf32': torch.float32, 'bf16': torch.bfloat16}
KERNEL_PRECISION = {'ieee': 'ieee', 'tf32': 'tf32', 'bf16': 'io'}
PAPER_LABEL = {'ieee': 'fp32', 'tf32': 'tf32', 'bf16': 'bf16'}
MODULE_PARAM_DTYPE = torch.float32

def amp_ctx(precision: str, device_type: str='cuda') -> Callable[[], Any]:
    if precision not in PRECISIONS:
        raise ValueError(f'precision must be one of {PRECISIONS}, got {precision!r}')
    if precision != 'bf16':
        return contextlib.nullcontext
    return lambda: torch.autocast(device_type=device_type, dtype=torch.bfloat16)

@contextlib.contextmanager
def precision_ctx(precision: str):
    if precision not in PRECISIONS:
        raise ValueError(f'precision must be one of {PRECISIONS}, got {precision!r}')
    allow = precision != 'ieee'
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cudnn

def _time_do_bench(fn, budget: Budget) -> float:
    import triton.testing
    return float(triton.testing.do_bench(fn, warmup=budget.do_bench_warmup, rep=budget.do_bench_rep, return_mode='median'))

def round_order(cells: Sequence, r: int) -> list:
    cells = list(cells)
    return cells if r % 2 == 0 else cells[::-1]

def _time_events(fn, budget: Budget) -> float:
    for _ in range(budget.ev_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(budget.ev_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / budget.ev_iters

def _time_one(fn, tier: str, budget: Budget) -> float:
    return _time_do_bench(fn, budget) if tier == 'do_bench' else _time_events(fn, budget)

def _drop_canary_operands() -> None:
    _CANARY_OPERANDS.clear()
    torch.cuda.empty_cache()

def _peak_memory(fn, device: str) -> tuple[int, int]:
    pre = getattr(fn, 'pre_call', None)
    if pre is not None:
        pre()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (int(torch.cuda.max_memory_allocated()), int(base))

def _build_cell(cell: Cell, errors: dict, progress: bool, quiet: bool=False):
    key = cell.key()
    err = None
    try:
        with precision_ctx(cell.precision):
            fn = cell.build()
            output = fn()

            def tensors(value):
                if isinstance(value, dict):
                    for child in value.values():
                        yield from tensors(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        yield from tensors(child)
                elif torch.is_tensor(value):
                    yield value
            values = list(tensors(output))
            values += list(getattr(fn, 'validation_tensors', lambda: [])())
            for value in values:
                if torch.is_tensor(value) and value.is_floating_point() and (not torch.isfinite(value).all().item()):
                    raise FloatingPointError('nonfinite output/state/gradient during warmup')
            if cell.tags.get('device', 'cuda').split(':')[0] == 'cuda':
                torch.cuda.synchronize()
    except Exception as exc:
        err = f'{type(exc).__name__}: {exc}'
    if err is not None:
        if cell.tags.get('device', 'cuda').split(':')[0] == 'cuda':
            torch.cuda.empty_cache()
        errors.setdefault(key, err)
        if progress and (not quiet):
            print(f'  SKIP {key}\n       {errors[key][:200]}', flush=True)
        return None
    if progress and (not quiet):
        print(f'  built {key}', flush=True)
    return fn

def warm_cells(cells: Sequence[Cell]) -> tuple[int, int]:
    errors: dict[str, str] = {}
    warmed = 0
    for cell in cells:
        fn = _build_cell(cell, errors, progress=True)
        if fn is not None:
            warmed += 1
        del fn
        torch.cuda.empty_cache()
    return (warmed, len(errors))

def _memory_and_extras(cells, get_fn, post_cell, device, progress):
    _drop_canary_operands()
    peaks: dict[str, tuple[int, int]] = {}
    extras: dict[str, dict] = {}
    for cell in cells:
        key = cell.key()
        fn = get_fn(cell)
        if fn is None:
            continue
        try:
            with precision_ctx(cell.precision):
                peaks[key] = _peak_memory(fn, device)
                if post_cell is not None:
                    extras[key] = post_cell(cell, fn) or {}
        except Exception as exc:
            extras[key] = dict(extras.get(key) or {}, error=f'post-timing pass failed: {type(exc).__name__}: {exc}'[:500])
        finally:
            del fn
            torch.cuda.empty_cache()
    return (peaks, extras)

def _run_group_resident(group: str, cells: Sequence[Cell], budget: Budget, device: str, post_cell, progress: bool, warm_start: bool=False) -> tuple[dict, list[float]]:
    built: dict[str, Callable[[], Any]] = {}
    errors: dict[str, str] = {}
    for cell in cells:
        fn = _build_cell(cell, errors, progress)
        if fn is not None:
            built[cell.key()] = fn
    if warm_start:
        canary_ms(device)
        for cell in cells:
            fn = built.get(cell.key())
            if fn is None:
                continue
            try:
                with precision_ctx(cell.precision):
                    fn()
            except Exception as exc:
                errors.setdefault(cell.key(), f'{type(exc).__name__}: {exc}')
                built.pop(cell.key(), None)
        torch.cuda.synchronize()
    samples: dict[str, list[float]] = {k: [] for k in built}
    canaries: list[float] = []
    for r in range(budget.rounds):
        canaries.append(canary_ms(device))
        for cell in round_order(cells, r):
            key = cell.key()
            fn = built.get(key)
            if fn is None:
                continue
            try:
                with precision_ctx(cell.precision):
                    samples[key].append(_time_one(fn, cell.resolved_tier, budget))
            except Exception as exc:
                errors.setdefault(key, f'{type(exc).__name__}: {exc}')
                samples.pop(key, None)
                built.pop(key, None)
        if progress:
            print(f'  [{group}] round {r + 1}/{budget.rounds} canary {canaries[-1]:.3f} ms', flush=True)
    peaks, extras = _memory_and_extras(cells, lambda c: built.get(c.key()), post_cell, device, progress)
    built.clear()
    torch.cuda.empty_cache()
    return (dict(samples=samples, errors=errors, peaks=peaks, extras=extras), canaries)

def _run_group_rebuild(group: str, cells: Sequence[Cell], budget: Budget, device: str, post_cell, progress: bool, warm_start: bool=False) -> tuple[dict, list[float]]:
    samples: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    dead: set[str] = set()
    canaries: list[float] = []
    if warm_start:
        canary_ms(device)
    for r in range(budget.rounds):
        canaries.append(canary_ms(device))
        for cell in round_order(cells, r):
            key = cell.key()
            if key in dead:
                continue
            fn = _build_cell(cell, errors, progress, quiet=r > 0)
            if fn is None:
                dead.add(key)
                continue
            try:
                with precision_ctx(cell.precision):
                    samples.setdefault(key, []).append(_time_one(fn, cell.resolved_tier, budget))
            except Exception as exc:
                errors.setdefault(key, f'{type(exc).__name__}: {exc}')
                samples.pop(key, None)
                dead.add(key)
            finally:
                del fn
                torch.cuda.empty_cache()
        if progress:
            print(f'  [{group}] round {r + 1}/{budget.rounds} canary {canaries[-1]:.3f} ms (rebuild mode)', flush=True)

    def _one(cell: Cell):
        if cell.key() in dead:
            return None
        return _build_cell(cell, errors, progress, quiet=True)
    peaks, extras = _memory_and_extras(cells, _one, post_cell, device, progress)
    torch.cuda.empty_cache()
    return (dict(samples=samples, errors=errors, peaks=peaks, extras=extras), canaries)
GROUP_MODES = ('auto', 'resident', 'rebuild')

def _resolve_group_mode(mode: str, cells: Sequence[Cell]) -> str:
    if mode not in GROUP_MODES:
        raise ValueError(f'group mode must be one of {GROUP_MODES}, got {mode!r}')
    if mode != 'auto':
        return mode
    return 'rebuild' if any((c.family == 'model' for c in cells)) else 'resident'

def _run_group_cpu(cells, budget, progress, group_mode='auto'):
    errors, samples, built = ({}, {}, {})
    rebuild = _resolve_group_mode(group_mode, cells) == 'rebuild'
    if not rebuild:
        for cell in cells:
            fn = _build_cell(cell, errors, progress)
            if fn is not None:
                built[cell.key()] = fn
    for _ in range(budget.rounds):
        for cell in cells:
            key = cell.key()
            if key in errors:
                continue
            fn = _build_cell(cell, errors, progress) if rebuild else built.get(key)
            if fn is None:
                continue
            try:
                with precision_ctx(cell.precision):
                    for _ in range(budget.ev_warmup):
                        fn()
                    start = time.perf_counter()
                    for _ in range(budget.ev_iters):
                        fn()
                    samples.setdefault(key, []).append((time.perf_counter() - start) * 1000 / budget.ev_iters)
            except Exception as exc:
                errors[key] = f'{type(exc).__name__}: {exc}'
                built.pop(key, None)
                samples.pop(key, None)
            finally:
                del fn
    return (dict(samples=samples, errors=errors, peaks={}, extras={}), [])

def _run_group(group: str, cells: Sequence[Cell], budget: Budget, device: str, post_cell, progress: bool, group_mode: str='auto', warm_start: bool=False) -> tuple[dict, list[float]]:
    if torch.device(device).type == 'cpu':
        return _run_group_cpu(cells, budget, progress, group_mode)
    mode = _resolve_group_mode(group_mode, cells)
    fn = _run_group_rebuild if mode == 'rebuild' else _run_group_resident
    return fn(group, cells, budget, device, post_cell, progress, warm_start)
time_one = _time_one
build_cell = _build_cell
run_group = _run_group
peak_memory = _peak_memory
drop_canary_operands = _drop_canary_operands
METRICS = ('wall', 'cuda_self', 'launches', 'host', 'memory', 'canary', 'roofline', 'launch_meta', 'kernel_resources')
ALWAYS_ON = ('wall', 'canary')

def _profile_cuda(fn: Callable[[], Any], iters: int):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per_kernel: dict[str, dict] = {}
    launches: dict[str, float] = defaultdict(float)
    for ev in prof.key_averages():
        cu = float(getattr(ev, 'self_device_time_total', 0.0) or 0.0)
        if cu <= 0:
            continue
        low = ev.key.lower()
        kind = 'memcpy' if 'memcpy' in low else 'memset' if 'memset' in low else 'kernel'
        launches[kind] += ev.count / iters
        row = per_kernel.setdefault(ev.key, dict(ms=0.0, calls=0.0))
        row['ms'] += cu / iters / 1000.0
        row['calls'] += ev.count / iters
    return (per_kernel, dict(launches))

def merge_min_over_rounds(rounds: Sequence[dict]) -> dict:
    if not rounds:
        return dict(ms_cuda_self=None, kernels={}, round_totals=[], calls_vary=[])
    best: dict[str, dict] = {}
    seen_calls: dict[str, set] = defaultdict(set)
    totals: list[float] = []
    for rnd in rounds:
        totals.append(sum((v['ms'] for v in rnd.values())))
        for name, row in rnd.items():
            seen_calls[name].add(round(row['calls'], 3))
            if name not in best or row['ms'] < best[name]['ms']:
                best[name] = dict(row)
    return dict(ms_cuda_self=sum((v['ms'] for v in best.values())), kernels={k: v for k, v in sorted(best.items(), key=lambda kv: -kv[1]['ms'])}, round_totals=totals, calls_vary=sorted((k for k, v in seen_calls.items() if len(v) > 1)))

def cuda_self(fn: Callable[[], Any], *, rounds: int=3, iters: int=1) -> dict:
    per_round = []
    launches: dict[str, float] = {}
    for _ in range(max(1, rounds)):
        pk, la = _profile_cuda(fn, iters)
        per_round.append(pk)
        launches = la
    out = merge_min_over_rounds(per_round)
    out['launches'] = launches
    out['n_launches'] = sum(launches.values())
    return out

def host_enqueue_ms(fn: Callable[[], Any], *, warmup: int=6, iters: int=10) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = (torch.cuda.Event(True), torch.cuda.Event(True))
    t0 = time.perf_counter()
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    t_host = time.perf_counter() - t0
    torch.cuda.synchronize()
    return (st.elapsed_time(en) / iters, t_host * 1000.0 / iters)
HOST_BOUND_RATIO = 1.05

def host_bound(ms_wall: float | None, ms_host: float | None) -> bool | None:
    if ms_wall is None or ms_host is None or ms_host <= 0:
        return None
    return bool(ms_wall <= ms_host * HOST_BOUND_RATIO)

def launch_meta(fn: Callable[[], Any], iters: int=2, out_dir: str | None=None) -> dict:
    import json as _json
    import tempfile
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix='.json', dir=out_dir)
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as fh:
            trace = _json.load(fh)
    finally:
        os.unlink(path)
    out: dict[str, dict] = {}
    for e in trace.get('traceEvents', []):
        if e.get('cat') not in ('kernel', 'Kernel'):
            continue
        a = e.get('args', {}) or {}
        rec = out.setdefault(e.get('name', '?'), dict(n=0.0, dur_us=0.0, launches={}))
        rec['n'] += 1
        rec['dur_us'] += float(e.get('dur', 0.0))
        key = (str(a.get('grid')), a.get('registers per thread'), a.get('shared memory'))
        sig = rec['launches'].setdefault('|'.join((str(k) for k in key)), dict(grid=a.get('grid'), block=a.get('block'), regs=a.get('registers per thread'), smem=a.get('shared memory'), occupancy_pct=a.get('est. achieved occupancy %'), blocks_per_sm=a.get('blocks per SM'), warps_per_sm=a.get('warps per SM'), count=0.0, dur_us=0.0))
        sig['count'] += 1
        sig['dur_us'] += float(e.get('dur', 0.0))
    for rec in out.values():
        rec['n'] /= iters
        rec['dur_us'] /= iters
        for sig in rec['launches'].values():
            sig['count'] /= iters
            sig['dur_us'] /= iters
    return out
_RESOURCE_KERNELS = (('orbit_ssm.kernels.ssd_mimo.fused_fwd_kernel', ('_fused_chunk_fwd_kernel',)), ('orbit_ssm.kernels.ssd_mimo.fused_bwd_kernel', ('_bwd_ka_kernel', '_bwd_kbc_hb_kernel', '_bwd_kb1_kernel', '_bwd_kb2_kernel', '_bwd_k2_kernel')))

def kernel_resources(kernels=_RESOURCE_KERNELS) -> list[dict]:
    import importlib as _il
    rows: list[dict] = []
    for modname, names in kernels:
        try:
            mod = _il.import_module(modname)
        except Exception:
            continue
        for name in names:
            k = getattr(mod, name, None)
            if k is None:
                continue
            k = getattr(k, 'fn', k)
            caches = getattr(k, 'device_caches', None)
            if not caches:
                continue
            merged: dict = {}
            for dev in list(caches):
                merged.update(caches[dev][0])
            for compiled in merged.values():
                meta = compiled.metadata
                rows.append(dict(kernel=name, warps=meta.num_warps, stages=meta.num_stages, regs=compiled.n_regs, spills=compiled.n_spills, smem=meta.shared))
    return rows
_MB = 1024.0 ** 2

def roofline_terms(B: int, H: int, P: int, N: int, L: int, T: int) -> dict:
    nc = L // T
    return dict(B=B, H=H, P=P, N=N, L=L, T=T, nc=nc, x=B * L * H * P * 2, bc=B * L * N * 2, bc_h=B * L * H * N * 2, vec=B * H * L * 4, theta=B * L * H * (N // 2) * 4, sall=B * H * nc * N * P * 4, sall_bf16=B * H * nc * N * P * 2, y=B * L * H * P * 2)

def roofline_model(B: int, H: int, P: int, N: int, L: int, T: int) -> dict:
    c = roofline_terms(B, H, P, N, L, T)
    ncta, nc = (B * H, c['nc'])
    f_chunk = 2 * (T * N * P + N * T * P + T * N * T + T * T * P)
    return dict(cell=c, bytes=c['x'] + c['y'] + 2 * c['bc'] + c['vec'], flops=f_chunk * ncta * nc)

def device_peaks(device: str='cuda') -> dict:
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    src = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)

    def _ev(fn, iters):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = (torch.cuda.Event(True), torch.cuda.Event(True))
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters
    gemm_ms = _ev(lambda: a @ b, 20)
    copy_ms = _ev(lambda: dst.copy_(src), 10)
    del a, b, src, dst
    torch.cuda.empty_cache()
    return dict(gemm4096_bf16_tflops=2 * 4096 ** 3 / (gemm_ms * 0.001) / 1000000000000.0, copy_gbs=2 * 256 * _MB / (copy_ms * 0.001) / 1000000000.0, gemm_ms=gemm_ms, copy_ms=copy_ms)

def roofline(shape: dict, ms: float | None, peaks: dict | None=None) -> dict:
    need = ('B', 'L', 'N')
    if any((shape.get(k) is None for k in need)) or not ms:
        return {}
    H = shape.get('H') or shape.get('D') or 0
    P = shape.get('P') or shape.get('G') or 0
    T = shape.get('chunk') or 64
    if not H or not P:
        return {}
    m = roofline_model(int(shape['B']), int(H), int(P), int(shape['N']), int(shape['L']), int(T))
    gbs = m['bytes'] / (ms * 0.001) / 1000000000.0
    tfs = m['flops'] / (ms * 0.001) / 1000000000000.0
    out = dict(roofline_bytes=m['bytes'], roofline_flops=m['flops'], achieved_gbs=gbs, achieved_tflops=tfs)
    if peaks:
        if peaks.get('copy_gbs'):
            out['pct_dram_peak'] = 100.0 * gbs / peaks['copy_gbs']
        if peaks.get('gemm4096_bf16_tflops'):
            out['pct_gemm_peak'] = 100.0 * tfs / peaks['gemm4096_bf16_tflops']
    return out
CAPTURE_TAG = 'cuda_graph'
GRAPH_ARM_SUFFIX = '@graph'

def graphs_available() -> bool:
    try:
        import orbit_ssm.harness.graphs
    except Exception:
        return False
    return True

def captured_closure(module, leaves, pass_: str, precision: str, *, warmup: int=11):
    from orbit_ssm.harness import graphs as gph
    x = leaves[0]
    gs = gph.graphed_step(module, (x,), fwd_bwd=pass_ == 'fwd_bwd', amp=lambda: gph.autocast(precision), warmup=warmup)

    def run():
        gs.graph.replay()
        return gs.static_output
    run.graphed_step = gs
    return run
COMPILE_ARM_SUFFIX = '@compiled'
COMPILE_COMBINED_ARM_SUFFIX = '@compiled-combined'

def compiled_closure(mod, leaves, pass_: str, *, amp: Callable[[], Any] | None=None, compile_kwargs: dict | None=None, call=None):
    from .harness import make_pass
    kw = dict(compile_kwargs or {})
    kw.setdefault('dynamic', False)
    target = call if call is not None else mod
    compiled_mod = torch.compile(target, **kw)
    invoke = compiled_mod if call is not None else lambda: compiled_mod(leaves[0])
    run = make_pass(invoke, leaves, pass_, amp=amp)
    run.compiled_module = compiled_mod
    return run

def metric_hook(metrics: Sequence[str], *, budget: Budget, peaks: dict | None=None, cuda_self_rounds: int=3, grids: frozenset[str] | None=None):
    want = set(metrics)

    def hook(cell, closure) -> dict:
        out: dict = {}
        if 'cuda_self' in want and budget.cuda_self and (grids is None or getattr(cell, 'grid', '') in grids):
            try:
                res = cuda_self(closure, rounds=cuda_self_rounds)
                out['ms_cuda_self'] = res['ms_cuda_self']
                out['cuda_self_round_totals'] = res['round_totals']
                out['cuda_kernels'] = res['kernels']
                out['cuda_calls_vary'] = res['calls_vary']
                if 'launches' in want:
                    out['launches'] = res['launches']
                    out['n_launches'] = res['n_launches']
            except Exception as exc:
                out['ms_cuda_self'] = None
                out['ms_cuda_self_error'] = f'{type(exc).__name__}: {exc}'
        elif 'launches' in want:
            try:
                _, la = _profile_cuda(closure, 1)
                out['launches'] = la
                out['n_launches'] = sum(la.values())
            except Exception as exc:
                out['launches_error'] = f'{type(exc).__name__}: {exc}'
        if 'host' in want:
            try:
                w, h = host_enqueue_ms(closure, iters=budget.ev_iters)
                out['ms_wall_events'] = w
                out['ms_host_enqueue'] = h
                out['host_bound'] = host_bound(w, h)
            except Exception as exc:
                out['ms_host_enqueue_error'] = f'{type(exc).__name__}: {exc}'
        if 'roofline' in want:
            try:
                base = out.get('ms_cuda_self') or out.get('ms_wall_events')
                out.update(roofline(getattr(cell, 'shape', {}) or {}, base, peaks))
            except Exception as exc:
                out['roofline_error'] = f'{type(exc).__name__}: {exc}'
        if 'launch_meta' in want:
            try:
                out['launch_meta'] = launch_meta(closure)
            except Exception as exc:
                out['launch_meta_error'] = f'{type(exc).__name__}: {exc}'
        if 'kernel_resources' in want:
            try:
                out['kernel_resources'] = kernel_resources()
            except Exception as exc:
                out['kernel_resources_error'] = f'{type(exc).__name__}: {exc}'
        if hasattr(closure, 'training_summary'):
            out.update(closure.training_summary())
        gs = getattr(closure, 'graphed_step', None)
        if gs is not None:
            out['capture'] = CAPTURE_TAG
            out['capture_memory_bytes'] = gs.memory
            out['capture_notes'] = list(gs.notes)
        return out
    return hook

class RatioRefused(ValueError):
    pass

def ratio(numer: dict, denom: dict, *, field: str='ms_median') -> float:
    for k, why in (('device', 'use different devices'), ('group', 'were not interleaved (different group)'), ('precision', 'are different precision columns'), ('pass', 'are different passes')):
        if numer.get(k) != denom.get(k):
            raise RatioRefused(f"refusing a ratio: {numer.get('arm')} and {denom.get('arm')} {why}: {numer.get(k)!r} vs {denom.get(k)!r}")
    if (numer.get('tags') or {}).get('capture') != (denom.get('tags') or {}).get('capture') or numer.get('capture') != denom.get('capture'):
        raise RatioRefused('refusing a ratio: one side is CUDA-graph captured and the other is eager (E2b/G34 — captured vs captured only)')
    if (numer.get('shape') or {}).get('N') != (denom.get('shape') or {}).get('N'):
        raise RatioRefused(f"refusing a ratio: the two sides ran at different state sizes N={(numer.get('shape') or {}).get('N')} vs {(denom.get('shape') or {}).get('N')} (I13b)")
    a, b = (numer.get(field), denom.get(field))
    if a is None or not b:
        raise RatioRefused(f'refusing a ratio: {field} missing on one side')
    return float(a) / float(b)
