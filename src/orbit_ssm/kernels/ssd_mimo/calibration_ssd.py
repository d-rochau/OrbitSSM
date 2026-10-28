
import os
from .. import calibration as cal
FAMILY = 'ssd_mimo'
_HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = ('siso_fwd', 'siso_bwd', 'siso_adj')
from .. import shape_sets as _cshapes
FP32_CHUNKS = (32, 64)
SHAPES = {}
for _tag in _cshapes.SET:
    SHAPES['cx_' + _tag] = dict(N=128, **_cshapes.geometry(_tag))
    SHAPES['real_' + _tag] = dict(N=128, rot=False, **_cshapes.geometry(_tag))
    for _T in FP32_CHUNKS:
        SHAPES['cxf32t%d_%s' % (_T, _tag)] = dict(N=128, dtype='fp32', T=_T, **_cshapes.geometry(_tag))

def io_group(io) -> str:
    return 'fp32' if io in ('fp32', 'float32') else ''

def group_key(rot=True, R=1, O=1, dtype='bf16', T=None, **_):
    g = '%s_%s' % ('complex' if rot else 'real', 'R1O1' if R == 1 and O == 1 else 'ROgt1')
    if io_group(dtype):
        return '%s_%s_t%d' % (g, io_group(dtype), int(T) if T else 64)
    return g
cal.register_family(FAMILY, env='SSD_MIMO_CALIBRATION', group_key=group_key, occupancy=lambda B, H, **_: B * H, code_files=[os.path.join(_HERE, 'siso_fwd_kernel.py'), os.path.join(_HERE, 'siso_bwd_kernel.py')], kernels=KERNELS, shapes=SHAPES, chunk_of=lambda g: _chunk_of(g), notes=dict(shape_set='src/orbit_ssm/kernels/shape_sets.py: 12 grid cells + 6 LM cells (B, d_inner, L), P = 64; rotating R = O = 1 at N = 128 (complex_R1O1) and real R = O = 1 forward (real_R1O1) on all 18; OOM cells are listed in dropped_shapes.', fp32='cxf32t<T>_* cells: rotating R = O = 1 at fp32 IO / tf32 operands, one group per chunk (complex_R1O1_fp32_t32 / _t64), measured at that chunk'))

def _chunk_of(group):
    if '_fp32_t' in group:
        return int(group.rsplit('_t', 1)[1])
    if group == 'real_R1O1':
        from ...ssm.ssd_mimo import _scan_chunk
        return _scan_chunk(1, 64, 1, True, real=True)
    return 64
_LAST_CALL = {}

def note_call(kind, **info):
    _LAST_CALL[kind] = info

def real_workload(g, seed=20261003):
    import torch
    from .fused_fwd_kernel import seg_prefix_host
    from .siso_fwd_kernel import siso_chunk_fwd
    B, L, H, P, N = (g[k] for k in 'B L H P N'.split())
    T = _chunk_of('real_R1O1')
    nc = L // T
    dev, dt = ('cuda', torch.bfloat16)
    gen = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, device=dev).to(dt)
    ct, bt, x = (r(B, nc, T, N), r(B, nc, T, N), r(B, H, P, 1, L))
    a = torch.rand(H, generator=gen, device=dev) * 2 + 0.5
    dc = seg_prefix_host(torch.rand(B, H, nc, T, generator=gen, device=dev) * 0.1, a)

    def run():
        siso_chunk_fwd(ct, bt, x, dc, a, precision='tf32', store_sall=True, needs_bwd=True)
    return run

def activate(has_rot, B, H, T=None, io='bf16'):
    if cal.enabled(FAMILY):
        from . import siso_fwd_kernel as sf, siso_bwd_kernel as sb
        cal.activate(FAMILY, group_key(rot=bool(has_rot), dtype=io, T=T), {'siso_fwd': sf._siso_fwd_kernel, 'siso_bwd': sb._siso_bwd_scan_kernel, 'siso_adj': sb._siso_rot_adjoint_kernel}, occupancy=B * H / cal.device_info()['sms'], T=T)

def prune(configs, kernel):
    return cal.prune(configs, FAMILY, kernel)
_TIER_ENV = {'siso_fwd': 'SISO_AUTOTUNE', 'siso_bwd': 'SISO_AUTOTUNE', 'siso_adj': 'SISO_ADJ_AUTOTUNE'}

def grid(kernel, tier, *, full, light, generic_pin):
    from .fused_fwd_kernel import _env
    if cal.enabled(FAMILY) and (not (_env(_TIER_ENV[kernel], '') or _env('AUTOTUNE', ''))):
        tier = 'light'
    return cal.calibrated_grid(FAMILY, kernel, tier, full=full, light=light, generic_pin=generic_pin)

def describe_served() -> dict:
    from . import siso_fwd_kernel as sf, siso_bwd_kernel as sb
    out = {'device': cal.device_info() if cal.enabled(FAMILY) else None, 'calibration_enabled': cal.enabled(FAMILY), 'code_hash': cal.code_hash(FAMILY)}
    try:
        from . import gluon_sm90_siso_bwd as g
        out['gluon_bwd_route'] = dict(g._LAST_ROUTE)
    except Exception as e:
        out['gluon_bwd_route'] = 'unavailable: %r' % (e,)
    for kind, k in (('siso_fwd', sf._siso_fwd_kernel), ('siso_bwd', sb._siso_bwd_scan_kernel), ('siso_adj', sb._siso_rot_adjoint_kernel)):
        n = len(k.configs)
        best = getattr(k, 'best_config', None)
        c = best if n > 1 and best is not None else k.configs[0] if n == 1 else None
        out[kind] = dict(candidates=n, grid_source=cal.source(FAMILY, kind), served='search-chose' if n > 1 and best is not None else 'untuned' if n > 1 else 'fixed', config=None if c is None else dict(num_warps=c.num_warps, num_stages=c.num_stages, maxnreg=c.maxnreg), last_call=_LAST_CALL.get(kind))
    return out

def smem_demote(T) -> bool:
    return cal.smem_demote(FAMILY, T)
