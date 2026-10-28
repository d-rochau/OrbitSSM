
from __future__ import annotations
import os
_NON_FORCING = frozenset({'SSD_MIMO_M3_EQ_PHASE_LAW', 'SSD_MIMO_M3_EQ_THETA_H', 'SSD_MIMO_CP_TRACE', 'SSD_MIMO_SMEM_STRATEGY_CACHE', 'SSD_MIMO_CALIBRATION'})
_PRODUCTION_SOURCES = ('shipped', 'local', 'light(shipped)', 'light(local)')

def forcing_env() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith('SSD_MIMO_') and k not in _NON_FORCING}

def _reset():
    from orbit_ssm.kernels.ssd_mimo import calibration_ssd as cs, chunkpar, fused_bwd_kernel as fb
    cs._LAST_CALL.clear()
    chunkpar._LAST_FWD_ROUTE.clear()
    fb._CP_LAST_ROUTE.clear()
    try:
        from orbit_ssm.kernels.ssd_mimo import gluon_sm90_siso_bwd as g90
        g90._LAST_ROUTE.clear()
        g90._LAST_ROUTE.update(used=False, why='never consulted')
    except Exception:
        pass

def served(closure) -> dict:
    import torch
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import autograd, calibration_fused as cf, calibration_ssd as cs, chunkpar, fused_bwd_kernel as fb
    _reset()
    n0 = len(autograd.dispatch_log())
    nb0 = len(autograd.chunk_bwd_log())
    closure()
    torch.cuda.synchronize()
    log = autograd.dispatch_log()
    disp = dict(log[-1]) if log else {}
    blog = autograd.chunk_bwd_log()
    bwd_chunk = dict(blog[-1]) if len(blog) > nb0 or (blog and len(blog) == 32) else None
    siso_fwd, siso_bwd = ('siso_fwd' in cs._LAST_CALL, 'siso_bwd' in cs._LAST_CALL)
    gluon = False
    try:
        from orbit_ssm.kernels.ssd_mimo import gluon_sm90_siso_bwd as g90
        gluon = bool(g90._LAST_ROUTE.get('used'))
    except Exception:
        pass
    cp_f, cp_b = (dict(chunkpar._LAST_FWD_ROUTE), dict(fb._CP_LAST_ROUTE))
    fwd = 'siso' if siso_fwd else 'chunk_parallel' if cp_f else 'general'
    bwd = 'siso_gluon' if gluon else 'siso_triton' if siso_bwd else 'chunk_parallel' if cp_b else 'general'
    sources = {}
    for fam, mod in (('ssd_mimo', cs), ('ssd_mimo_fused', cf)):
        for kern, info in mod.describe_served().items():
            if isinstance(info, dict) and info.get('served') in ('search-chose', 'fixed') and ('grid_source' in info):
                sources[f'{fam}/{kern}'] = info['grid_source']
    return dict(route=dict(fwd=fwd, bwd=bwd, chunk_used=disp.get('chunk_used'), chunk_mode=disp.get('chunk_mode'), chunk_bwd=(bwd_chunk or {}).get('T_bwd'), chunk_bwd_record=bwd_chunk, cp_fwd=cp_f or None, cp_bwd=cp_b or None, dispatch_new=len(log) - n0), calibration=dict(mode=cal.mode('ssd_mimo'), sources=sources, forcing_env=forcing_env()))
_MH_NON_FORCING = frozenset({'HYBRID_MH_TUNE_CACHE'})

def mh_forcing_env() -> dict:
    return {k: v for k, v in os.environ.items() if (k.startswith('HYBRID_MH_') or k.startswith('HYBRID_SSM_MH_') or k.startswith('MH_')) and k not in _MH_NON_FORCING}

def served_mh(closure, ssm=None) -> dict:
    import torch
    from orbit_ssm.kernels.hybrid_mamba_mh import autotune, bwd_gluon, fwd_gluon
    from orbit_ssm.kernels.hybrid_mamba_mh import calibration_mh as cmh
    from orbit_ssm.kernels.ssd_mimo import autograd
    fwd_gluon.LAST_CONFIG = None
    bwd_gluon.LAST_CONFIG = None
    n0 = len(autograd.dispatch_log())
    _reset()
    closure()
    torch.cuda.synchronize()
    jsonable = lambda d: None if d is None else {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}
    routes = list(ssm.last_routes()) if ssm is not None else None
    rec = dict(route=dict(mh=dict(fwd_config=jsonable(fwd_gluon.LAST_CONFIG), bwd_config=jsonable(bwd_gluon.LAST_CONFIG), blocks=[list(r) if isinstance(r, tuple) else r for r in routes] if routes is not None else None, refusal=ssm.last_refusal() if ssm is not None else None, autotune_tier=autotune.tier()), ssd_dispatch_new=len(autograd.dispatch_log()) - n0), calibration=dict(mode=autotune.tier(), sources={k: v.get('source') for k, v in cmh.describe_served().items() if isinstance(v, dict) and 'source' in v}, mh_bwd_ts=cmh.describe_served().get('mh_bwd_ts'), forcing_env=mh_forcing_env()))
    if fwd_gluon.LAST_CONFIG is None:
        rec['route']['mh']['fwd_note'] = 'no mh_scan fwd launch observed'
    return rec

def production_problem(rec: dict | None) -> str | None:
    if not rec or 'calibration' not in rec:
        return 'no route record'
    c = rec['calibration']
    if c.get('forcing_env'):
        return 'route-forcing environment: ' + ','.join(sorted(c['forcing_env']))
    if c.get('mode') == 'off':
        return 'calibration off'
    bad = {k: v for k, v in c.get('sources', {}).items() if not v.startswith(_PRODUCTION_SOURCES)}
    if bad:
        return 'uncalibrated launch configs: ' + ', '.join((f'{k}={v}' for k, v in sorted(bad.items())))
    return None
