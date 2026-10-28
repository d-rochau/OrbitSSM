
import os
from .. import calibration as cal
FAMILY = 'ssd_mimo_route'
_HERE = os.path.dirname(os.path.abspath(__file__))
cal.register_family(FAMILY, env='SSD_MIMO_CALIBRATION', code_files=[os.path.join(_HERE, 'rot_route.py')], kernels=(), shapes={})
ALTS = (('split', 32), ('split', 64), ('inkernel', 32), ('inkernel', 64))
ALT_NAMES = tuple(('%s%d' % a for a in ALTS))
SISO = ('siso', 64)
INF = 10 ** 9

def key(phase, group, field):
    return key_(phase, group, field)

def key_(phase, group, field):
    return 'route|%s|%s|%s' % (phase, group, field)

def siso_region(bh_per_sm, L, thr):
    occ, lmax, alt = (thr('siso_occ'), thr('siso_L'), thr('alt'))
    if occ is None or lmax is None or alt is None:
        return None
    o_in = bh_per_sm < occ / 100.0 if not thr('siso_occ_dir') else bh_per_sm >= occ / 100.0
    l_in = L <= lmax if not thr('siso_L_dir') else L > lmax
    return o_in and l_in

def _table_route(bh_per_sm, L, table):
    import math
    lo, ll = (math.log2(max(bh_per_sm, 0.001)), math.log2(max(L, 1)))
    best = min(table, key=lambda r: (r[0] - lo) ** 2 + (r[1] - ll) ** 2)
    return SISO if int(best[2]) < 0 else ALTS[int(best[2])]

def decide(bh_per_sm, L, thr):
    table = thr('table')
    if table:
        return _table_route(bh_per_sm, L, table)
    r = siso_region(bh_per_sm, L, thr)
    if r is None:
        return None
    return SISO if r else ALTS[int(thr('alt'))]
_MEMO: dict = {}

def phase_of(needs_bwd, dtype=None):
    return ('grad' if needs_bwd else 'nograd') + ('_fp32' if dtype in ('fp32', 'float32') else '')

def pick(bh, L, device, needs_bwd, thh=False, dtype=None):
    ax = 'fp32' if dtype in ('fp32', 'float32') else ''
    key = (bh, L, needs_bwd, thh, device, os.environ.get('SSD_MIMO_CALIBRATION'), ax)
    hit = _MEMO.get(key, _MEMO)
    if hit is not _MEMO:
        return hit
    r = _pick(bh, L, device, needs_bwd, thh, ax)
    _MEMO[key] = r
    return r

def _pick(bh, L, device, needs_bwd, thh, ax=''):
    if not cal.enabled(FAMILY):
        return None
    from .fused_fwd_kernel import _sm_count
    phase = phase_of(needs_bwd, ax or None)
    grp = 'thh' if thh else 'noth'
    thr = lambda f: cal.threshold(FAMILY, key_(phase, grp, f), None)[0]
    r = decide(bh / _sm_count(device), L, thr)
    if r is not None and r[0] == 'siso' and ax:
        t = thr('siso_T')
        r = ('siso', int(t) if t else 64)
    return r

def explicit_chunk():
    return bool(os.environ.get('SSD_MIMO_CHUNK'))

def explicit_siso():
    from .fused_fwd_kernel import _env
    return _env('SISO_FAST', 'auto').strip().lower() not in ('', 'auto')

def explicit_rot_bwd():
    from .fused_fwd_kernel import _env
    return bool(_env('ROT_BWD', ''))
