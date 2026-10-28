
import os
import re
from .. import calibration as cal
FAMILY = 'ssd_mimo_fused'
_HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = ('fused_ka', 'fused_kb1', 'fused_kbc_hb', 'fused_kb2', 'fused_kb2_hb', 'fused_kinv', 'fused_kc')
from .. import shape_sets as _cshapes
SHAPES = {}
for _tag in _cshapes.SET:
    _g = _cshapes.geometry(_tag)
    for _rot in (True, False):
        _p = 'cx' if _rot else 'real'
        SHAPES['%s_r1o1_%s' % (_p, _tag)] = dict(G=64, N=128, R=1, O=1, rot=_rot, **_g)
        SHAPES['%s_r2o2_%s' % (_p, _tag)] = dict(G=64, N=64, R=2, O=2, rot=_rot, **_g)
HI_TAGS = ('large_balanced', 'large_batch', 'large_wide', 'lm_b32_d2048_l2048', 'lm_b6_d2048_l2048', 'lm_b6_d4096_l2048', 'medium_balanced', 'medium_batch', 'medium_wide')
HI_EXTRA = {'x_b8_d2048_l4096': (8, 2048, 4096), 'x_b2_d8192_l4096': (2, 8192, 4096), 'x_b4_d512_l4096': (4, 512, 4096)}
for _tag in HI_TAGS:
    SHAPES['real_hi_r1o1_' + _tag] = dict(G=64, N=128, R=1, O=1, rot=False, hi=True, **_cshapes.geometry(_tag))
for _tag, (_b, _d, _l) in HI_EXTRA.items():
    SHAPES['real_hi_r1o1_' + _tag] = dict(G=64, N=128, R=1, O=1, rot=False, hi=True, B=_b, L=_l, H=_d // _cshapes.P, P=_cshapes.P)
for _tag in _cshapes.LM_TAGS:
    SHAPES['cx_r4o4_' + _tag] = dict(G=64, N=64, R=4, O=4, rot=True, **_cshapes.geometry(_tag))
FP32_CHUNKS = (32, 64)
for _tag in _cshapes.SET:
    _g = _cshapes.geometry(_tag)
    for _T in FP32_CHUNKS:
        SHAPES['real_r1o1f32t%d_%s' % (_T, _tag)] = dict(G=64, N=128, R=1, O=1, rot=False, dtype='fp32', T=_T, **_g)
        SHAPES['cx_r1o1f32t%d_%s' % (_T, _tag)] = dict(G=64, N=128, R=1, O=1, rot=True, dtype='fp32', T=_T, **_g)
RANK_CHUNKS = {2: (32, 64), 4: (16, 32, 64)}
for _R, _Ts in RANK_CHUNKS.items():
    for _tag in _cshapes.SET if _R == 2 else _cshapes.LM_TAGS:
        _g = _cshapes.geometry(_tag)
        for _T in _Ts:
            for _rot in (True, False):
                for _dt in ('bf16', 'fp32'):
                    SHAPES['%s_r%do%d%st%d_%s' % ('cx' if _rot else 'real', _R, _R, 'f32' if _dt == 'fp32' else '', _T, _tag)] = dict(G=64, N=64, R=_R, O=_R, rot=_rot, dtype=_dt, T=_T, rk=True, **_g)

def group_key(rot, R, O, hi=False, dtype='bf16', T=None, rk=False, **_):
    if rk:
        return '%s_R%dO%d%s_t%d' % ('complex' if rot else 'real', R, O, '_fp32' if dtype in ('fp32', 'float32') else '', int(T))
    if dtype in ('fp32', 'float32'):
        return '%s_%s_fp32_t%d' % ('complex' if rot else 'real', 'R1O1' if R == 1 and O == 1 else 'ROgt1', int(T) if T else 32)
    if hi and (not rot) and (R == 1) and (O == 1):
        return 'real_R1O1_hi'
    return '%s_%s' % ('complex' if rot else 'real', 'R1O1' if R == 1 and O == 1 else 'ROgt1')

def group_for(rot, R, O, T=None, dtype='bf16'):
    if R > 1 and R == O and (T is not None) and (os.environ.get('SSD_MIMO_RANK_GROUPS', '1') != '0'):
        rg = group_key(rot, R, O, dtype=dtype, T=T, rk=True)
        data, _ = cal.load(FAMILY)
        if rg in ((data or {}).get('chunk') or {}):
            return rg
    g = group_key(rot, R, O, dtype=dtype, T=T)
    if g == 'real_R1O1' and T is not None:
        from .calibration_chunkpar import real_rule
        r = real_rule()
        if r is not None and int(T) == r[0]:
            return 'real_R1O1_hi'
    return g

def _key(c):
    hb = c.get('HB')
    return ('hb%d' % hb if hb is not None else '') + cal.cfg_key(c['num_warps'], c['num_stages'], c['maxnreg'])

def _parse(k):
    hb = None
    if k.startswith('hb'):
        i = k.index('w')
        hb, k = (int(k[2:i]), k[i:])
    c = cal.cfg_parse(k)
    if hb is not None:
        c['HB'] = hb
    return c

def _class(c):
    return (c['num_stages'], c.get('HB'))

def _make(c):
    import triton
    kw = dict(num_warps=c['num_warps'], num_stages=c['num_stages'])
    if c['maxnreg']:
        kw['maxnreg'] = c['maxnreg']
    return triton.Config({'HB': c['HB']} if c.get('HB') is not None else {}, **kw)

def _obj_key(cfg):
    d = dict(num_warps=cfg.num_warps, num_stages=cfg.num_stages, maxnreg=cfg.maxnreg)
    if 'HB' in cfg.kwargs:
        d['HB'] = cfg.kwargs['HB']
    return _key(d)
cal.register_family(FAMILY, env='SSD_MIMO_CALIBRATION', code_files=[os.path.join(_HERE, 'fused_bwd_kernel.py')], kernels=KERNELS, shapes=SHAPES, group_key=group_key, occupancy=lambda B, H, **_: B * H, axes=('num_warps', 'num_stages', 'maxnreg', 'HB'), cfg_key=_key, cfg_parse=_parse, cfg_class=_class, make_config=_make, config_key=_obj_key, chunk_of=lambda g: _chunk_of(g), notes=dict(fp32='real_r1o1f32t<T>_* / cx_r1o1f32t<T>_* cells (all 18 x T in {32, 64}, fp32 IO, tf32 operands, R = O = 1): groups real_R1O1_fp32_t<T> / complex_R1O1_fp32_t<T>, measured at T; they replace the six LM-only real_r1o1fp32_* kb1 cells.', shape_set='src/orbit_ssm/kernels/shape_sets.py: 12 grid cells (3 size classes x balanced/long/wide/batch at equal work) + 6 LM cells, (B, d_inner, L), P = 64', r4o4_deviation='DEVIATION from the shared set: R4O4 (complex, N = 64) is measured on the 6 LM cells only (measurement-time bound); R = O = 1 (N = 128) and R2O2 (N = 64), real and complex, are measured on all 18 cells.  Cells that ran out of memory are listed in dropped_shapes.'))

def _real_r1o1_chunks(G=64, dtype='bf16'):
    from ...ssm.ssd_mimo import _scan_chunk, _scan_chunk_bwd
    tf = _scan_chunk(1, G, 1, True, real=True, dtype=dtype)
    tb = _scan_chunk_bwd(1, G, 1, tf, real=True, dtype=dtype)
    return (tf, tf if tb is None else tb)

def _chunk_of(group):
    if '_fp32_t' in group or re.search('_R\\dO\\d_t\\d+$', group):
        return int(group.rsplit('_t', 1)[1])
    if group == 'real_R1O1':
        return _real_r1o1_chunks()[1]
    if group == 'real_R1O1_hi':
        from .calibration_chunkpar import real_rule
        r = real_rule()
        return 64 if r is None else r[0]
    from ...ssm.ssd_mimo import _scan_chunk
    cells = [g for g in SHAPES.values() if group_key(g['rot'], g['R'], g['O'], **{k: v for k, v in g.items() if k == 'hi'}) == group]
    ts = {_scan_chunk(g['R'], g['G'], g['O']) for g in cells}
    return ts.pop() if len(ts) == 1 else None

def grid(kernel, tier, *, full, light, generic_pin):
    return cal.calibrated_grid(FAMILY, kernel, tier, full=full, light=light, generic_pin=generic_pin)
R8_GROUP = 'R8O8_safe'
R8_SAFE = dict(T=16, num_warps=4, num_stages=1, maxnreg=None)

def is_r8(R, O):
    return int(R) >= 8 and int(O) >= 8

def r8_safe_configs(configs):

    def rank(c):
        return (c.num_stages, c.kwargs.get('HB', 0), abs(c.num_warps - R8_SAFE['num_warps']), c.maxnreg or 0)
    exact = [c for c in configs if c.num_warps == R8_SAFE['num_warps'] and c.num_stages == R8_SAFE['num_stages'] and (not c.maxnreg)]
    pool = exact or list(configs)
    if not pool:
        return configs
    best = min(pool, key=rank)
    return [best]

def activate(rot, R, O, B, H, T=None, dtype='bf16'):
    if cal.enabled(FAMILY):
        if is_r8(R, O):
            cal.activate(FAMILY, R8_GROUP, autotuners(), occupancy=None, T=T)
            return
        cal.activate(FAMILY, group_for(rot, R, O, T, dtype), autotuners(), occupancy=B * H / cal.device_info()['sms'], T=T)

def prune(configs, kernel):
    if cal.enabled(FAMILY) and cal.active_group(FAMILY) == R8_GROUP:
        return r8_safe_configs(configs)
    return cal.prune(configs, FAMILY, kernel)

def autotuners():
    from . import fused_bwd_kernel as fb
    return {'fused_ka': fb._bwd_ka_kernel, 'fused_kb1': fb._bwd_kb1_kernel, 'fused_kbc_hb': fb._bwd_kbc_hb_kernel, 'fused_kb2': fb._bwd_kb2_kernel, 'fused_kb2_hb': fb._bwd_kb2_hb_kernel, 'fused_kinv': fb._bwd_kinv_kernel, 'fused_kc': fb._bwd_k2_kernel}

def workload(geom, seed=20261003):
    import torch
    from benchmarks.bench_kernels import build_ssd
    from benchmarks.harness import make_pass
    from ...ssm.ssd_mimo import _scan_chunk
    R, O, G, N = (geom['R'], geom['O'], geom['G'], geom['N'])
    T = 64 if geom.get('hi') else _scan_chunk(R, G, O)
    T_bwd = None
    if not geom['rot'] and R == 1 and (O == 1):
        T, T_bwd = _real_r1o1_chunks(G)
        T_bwd = None if T_bwd == T else T_bwd
    torch.manual_seed(seed)
    if geom.get('rk'):
        from benchmarks.bench_kernels import build_rank_mixer
        call, leaves = build_rank_mixer(geom['rot'], R, geom['H'] * geom['P'], geom['B'], geom['L'], N=N, T=geom['T'], fp32=geom.get('dtype') == 'fp32', seed=seed)
        return make_pass(call, leaves, 'fwd_bwd')
    if geom.get('dtype') == 'fp32':
        if not geom['rot'] and R == 1 and (O == 1):
            from benchmarks.baselines import build_real_fair
            call, leaves = build_real_fair(geom['B'], geom['H'], G, N, geom['L'], arm='ours', chunk=geom['T'], seed=seed, dtype=torch.float32)
            return make_pass(call, leaves, 'fwd_bwd')
        call, leaves = build_ssd(geom['B'], geom['H'], G, R, O, N, geom['L'], chunk=geom['T'], dtype=torch.float32, precision='tf32', feature='rotation', seed=seed)
        return make_pass(call, leaves, 'fwd_bwd')
    if not geom['rot'] and R == 1 and (O == 1):
        from benchmarks.baselines import build_real_fair
        call, leaves = build_real_fair(geom['B'], geom['H'], G, N, geom['L'], arm='ours', chunk=T, seed=seed, chunk_bwd=T_bwd)
        return make_pass(call, leaves, 'fwd_bwd')
    if geom['rot'] and R == 1 and (O == 1):
        from benchmarks.baselines import build_m3_eq
        from benchmarks.harness import KERNEL_PRECISION
        call, leaves = build_m3_eq(geom['B'], geom['H'], geom['P'], geom['N'], geom['L'], arm='ours', chunk=T, angles='m3', trap_off=False, dtype=torch.bfloat16, precision=KERNEL_PRECISION['bf16'], seed=seed)
        return make_pass(call, leaves, 'fwd_bwd')
    feature = 'rotation' if geom['rot'] else 'bc_per_head'
    call, leaves = build_ssd(geom['B'], geom['H'], G, R, O, N, geom['L'], chunk=T, dtype=torch.bfloat16, precision='io', feature=feature, seed=seed)
    return make_pass(call, leaves, 'fwd_bwd')

def describe_served() -> dict:
    out = {'calibration_enabled': cal.enabled(FAMILY), 'code_hash': cal.code_hash(FAMILY)}
    for name, k in autotuners().items():
        n = len(k.configs)
        best = getattr(k, 'best_config', None)
        c = best if n > 1 and best is not None else k.configs[0] if n == 1 else None
        out[name] = dict(candidates=n, grid_source=cal.source(FAMILY, name), served='search-chose' if n > 1 and best is not None else 'untuned' if n > 1 else 'fixed', config=None if c is None else dict(num_warps=c.num_warps, num_stages=c.num_stages, maxnreg=c.maxnreg, **dict(c.kwargs)))
    return out

def smem_demote(T) -> bool:
    return cal.smem_demote(FAMILY, T)
