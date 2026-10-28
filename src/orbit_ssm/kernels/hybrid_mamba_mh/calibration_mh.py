
from __future__ import annotations
import itertools
import math
import os
import time
from .. import calibration as cal
FAMILY = 'mh'
_HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = ('mh_fwd', 'mh_bwd')
FWD_AXES = ('BLOCK_P', 'BLOCK_L', 'PF')
BWD_AXES = ('TP_P', 'HP', 'MAXNREG')
TS_CANDIDATES = (1, 8, 16, 32)
SHAPES = {'lm_real': dict(B=8, H=128, P=16, N=128, L=2048), 'lm_rot': dict(B=8, H=128, P=16, N=128, L=2048, rot=True), 'c15w_rot': dict(B=8, H=32, P=64, N=128, L=2048, rot=True), 'c15w_rot_b6': dict(B=6, H=32, P=64, N=128, L=2048, rot=True), 'c15w_rot_l8k': dict(B=2, H=32, P=64, N=128, L=8192, rot=True), 'c15w_rot_ieee': dict(B=8, H=32, P=64, N=128, L=2048, rot=True, precision='ieee'), 'p32_rot': dict(B=8, H=64, P=32, N=128, L=2048, rot=True)}
for _g in SHAPES.values():
    _g.setdefault('precision', 'bf16')
    _g.setdefault('rot', False)
_CODE = ('fwd_gluon.py', 'bwd_gluon.py', 'common_gluon.py', 'ts_forward.py', 'ts_adjoint.py', 'autotune.py', 'layout.py', 'functions.py', 'slp.py', 'half_fold.py')

def cfg_key(c: dict) -> str:
    return ','.join((f'{k}={int(c[k])}' for k in sorted(c)))

def cfg_parse(key: str) -> dict:
    out = {}
    for item in filter(None, key.split(',')):
        k, _, v = item.partition('=')
        out[k] = int(v)
    return out
cal.register_family(FAMILY, env='HYBRID_MH_CALIBRATION', code_files=[os.path.join(_HERE, f) for f in _CODE], kernels=KERNELS, shapes=SHAPES, axes=FWD_AXES + BWD_AXES, group_key=lambda **g: 'complex' if g.get('rot') else 'real', cfg_key=cfg_key, cfg_parse=cfg_parse, cfg_class=lambda c: (), make_config=dict, config_key=lambda c: cfg_key(c))

def fwd_grid() -> list[dict]:
    return [dict(BLOCK_P=bp, BLOCK_L=bl, PF=pf) for bp, bl, pf in itertools.product((1, 2, 4), (256, 512, 1024), (0, 1))]

def bwd_grid() -> list[dict]:
    return [dict(TP_P=tp, HP=hp, MAXNREG=r, TS=1) for tp, hp, r in itertools.product((2, 4), (4, 8), (0, 168))]
P1_NS = (16, 128)
P1_PRECISIONS = ('bf16', 'fp32')
P1_FEATS = {'bare': 'sp1.db0.ds0.z0.h0', 'm1': 'sp1.db1.ds1.z0.h0'}
P1_SIDES = ('fwd', 'fwd_trn', 'bwd')
_DT = {'torch.bfloat16': 'bf16', 'torch.float16': 'fp16', 'torch.float32': 'fp32'}

def p1_shapes() -> dict:
    from ..shape_sets import SET
    out = {}
    for tag, (B, d, L) in SET.items():
        for N in P1_NS:
            for prec in P1_PRECISIONS:
                for cls in P1_FEATS:
                    out['%s_n%d_%s_%s' % (tag, N, prec, cls)] = dict(B=B, H=d, P=1, N=N, L=L, precision=prec, cls=cls)
    return out
P1_SHAPES = p1_shapes()

def p1_feat(*, delta_softplus, delta_bias, d_skip, z, h_init) -> str:
    return 'sp%d.db%d.ds%d.z%d.h%d' % (bool(delta_softplus), delta_bias is not None, d_skip is not None, z is not None, h_init is not None)

def p1_cell_key(B, H, N, L, dtype, feat) -> str:
    return 'B%d|H%d|N%d|L%d|%s|%s' % (B, H, N, L, _DT.get(str(dtype), str(dtype)), feat)

def p1_fwd_axes(H, L, ckpt_l, ts_ok, infer) -> dict:
    import triton
    top = triton.next_power_of_2(L)
    axes = dict(BLOCK_L=[b for b in (128, 256, 512, 1024, 2048) if b <= top and (not ckpt_l or b % ckpt_l == 0)], PF=[0, 1], NUM_WARPS=[1, 2, 4])
    if ts_ok:
        axes['TS'] = [1, 2, 4, 8, 16]
    if infer:
        axes['CKV'] = [0, 1]
    return axes

def p1_fwd_fpack_corners(H, L, ckpt_l) -> list:
    from . import autotune
    base = dict(BLOCK_P=1, BLOCK_L=512, NUM_WARPS=1, PF=0)
    return [dict(c, FPACK=1) for c in autotune.t_grid(base, H, L, ckpt_l) if c['BLOCK_P'] > 1]

def p1_bwd_axes(H, N, nblk) -> dict:
    return dict(HP=[h for h in (8, 4, 16, 32, 1) if H % h == 0], TP_P=[2, 4, 8, 16, 32], NUM_WARPS=[1, 2, 4], MAXNREG=[0, 168, 128], NS=[n for n in (1, 2, 4, 8, 16, 32) if N % n == 0], TS=[1, 2, 4, 8, 16, 32] if nblk >= 8 else [1], AGX=[0, 1], BCS=[1, 8, 16, 64, 0])

def p1_envelope(*, P, ROT, pmap, bc_head, LAW_MASK, endpoint_own, n_off, nr, N, absent) -> bool:
    return P == 1 and (not ROT) and (not pmap) and (not bc_head) and (LAW_MASK == 1) and (endpoint_own is None) and (n_off == 0) and (nr == N) and all((t is None for t in absent))
_P1_INJECT: dict = {}
_P1_PARSED: dict = {}
P1_STATS = dict(hit=0, miss=0)

def p1_config(side, B, H, N, L, dtype, feat):
    if not cal.enabled(FAMILY):
        return None
    from . import autotune
    if autotune.tier() == 'full':
        return None
    key = p1_cell_key(B, H, N, L, dtype, feat)
    cells = _P1_INJECT
    if key not in cells:
        data, _ = cal.load(FAMILY)
        cells = (data or {}).get('p1_cells') or {}
    ent = cells.get(key)
    txt = ent.get(side) if ent else None
    if not txt:
        P1_STATS['miss'] += 1
        return None
    P1_STATS['hit'] += 1
    c = _P1_PARSED.get(txt)
    if c is None:
        c = _P1_PARSED[txt] = cfg_parse(txt)
    return dict(c)

def nblk(geometry) -> int:
    from .bwd_gluon import auto_block_l
    return math.ceil(geometry['L'] / auto_block_l(geometry['P'], geometry['L']))

def _events(fn, sync_ms=25.0):
    import torch
    fn()
    torch.cuda.synchronize()
    a, b = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    a.record()
    fn()
    b.record()
    torch.cuda.synchronize()
    n = max(3, min(200, math.ceil(sync_ms / max(a.elapsed_time(b), 0.001))))
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n

def child_run(shape: str, rounds: int) -> dict:
    os.environ.update(HYBRID_MH_CALIBRATION='off', HYBRID_MH_AUTOTUNE='pinned', HYBRID_MH_TUNE_CACHE='off')
    for k in ('HYBRID_MH_FWD_CONFIG', 'HYBRID_MH_BWD_CONFIG', 'MH_BWD_MAXNREG'):
        os.environ.pop(k, None)
    import statistics
    import torch
    from benchmarks import measure
    from benchmarks.bench_kernels import build_mh
    g = SHAPES[shape]
    torch.manual_seed(20261003)
    with measure.precision_ctx(g['precision']):
        call, leaves = build_mh(g['B'], g['H'], g['P'], g['N'], g['L'], dtype=measure.IO_DTYPE[g['precision']], rot=g['rot'])
        y0 = call()
        y0 = y0[0] if isinstance(y0, tuple) else y0
        dy = torch.randn_like(y0)

        def fwd():
            with torch.no_grad():
                call()

        def fwd_bwd():
            for t in leaves:
                t.grad = None
            y = call()
            (y[0] if isinstance(y, tuple) else y).backward(dy)

        def pinned(side, c, fn):
            os.environ[f'HYBRID_MH_{side.upper()}_CONFIG'] = cfg_key(c)
            try:
                return _events(fn)
            except Exception:
                return None
            finally:
                os.environ.pop(f'HYBRID_MH_{side.upper()}_CONFIG', None)
        t0 = time.perf_counter()
        fwd_bwd()
        torch.cuda.synchronize()
        first = time.perf_counter() - t0
        F, Bk, T = ({}, {}, {})
        ts_base = None
        for r in range(rounds):
            for c in fwd_grid():
                ms = pinned('fwd', c, fwd)
                if ms is not None:
                    F.setdefault(cfg_key(c), []).append(ms)
            f_ref = _events(fwd)
            for c in bwd_grid():
                ms = pinned('bwd', c, fwd_bwd)
                if ms is not None:
                    Bk.setdefault(cfg_key({k: v for k, v in c.items() if k != 'TS'}), []).append(ms - f_ref)
            if ts_base is None:
                ts_base = min(Bk, key=lambda k: Bk[k][0])
            for ts in TS_CANDIDATES:
                ms = pinned('bwd', dict(cfg_parse(ts_base), TS=ts), fwd_bwd)
                if ms is not None:
                    T.setdefault(str(ts), []).append(ms - f_ref)
    med = lambda d: {k: statistics.median(v) for k, v in d.items()}
    return dict(shape=shape, geometry=g, first_call_s=first, rounds=rounds, timings_ms=dict(mh_fwd=med(F), mh_bwd=med(Bk)), ts_ms=med(T), ts_base=ts_base, nblk=nblk(g), level='mh_scan op, CUDA events per call, median of rounds; mh_bwd = fwd+bwd minus same-round fwd')

def derive_thresholds(res: dict) -> dict:
    rows = [(r['nblk'], r['ts_ms']) for r in res.values() if r.get('ts_ms')]
    if not rows:
        return {}
    best = None
    for ts in TS_CANDIDATES[1:]:
        for m in sorted({n for n, _ in rows}) + [math.inf]:
            sd = []
            for n, t in rows:
                pick = str(ts) if n >= m else '1'
                sd.append(t.get(pick, math.inf) / min(t.values()))
            score = (max(sd), sum(sd) / len(sd))
            if best is None or score < best[0]:
                best = (score, ts, m)
    (worst, mean), ts, m = best
    return {'mh_bwd_ts': dict(value=dict(ts=ts, min_nblk=None if m == math.inf else m), evidence=dict(worst_slowdown=worst, mean_slowdown=mean, per_shape={s: dict(nblk=r['nblk'], ts_ms=r['ts_ms']) for s, r in res.items()}))}

def describe_served() -> dict:
    out = dict(calibration_enabled=cal.enabled(FAMILY), code_hash=cal.code_hash(FAMILY))
    for k in KERNELS:
        hit = cal.lookup(FAMILY, k) if cal.enabled(FAMILY) else None
        out[k] = dict(source=None if hit is None else hit['source'], fixed=None if hit is None else hit['default']['fixed'])
    out['mh_bwd_ts'] = cal.threshold(FAMILY, 'mh_bwd_ts')
    return out
