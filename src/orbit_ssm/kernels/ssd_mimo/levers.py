
import contextlib
import os
import threading
from .. import calibration as cal
FAMILY = 'ssd_mimo_levers'
_HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = {'phase_law_fwd_active_k': ('PHASE_LAW_FWD_ACTIVE_K', 'fwd'), 'phase_law_active_k': ('PHASE_LAW_ACTIVE_K', 'bwd'), 'phase_law_headpar': ('PHASE_LAW_HEADPAR', 'bwd'), 'phase_law_suffix': ('PHASE_LAW_SUFFIX_KERNEL', 'bwd'), 'rot_ka_fwd': ('SISO_ROT_KA', 'fwd'), 'rot_ka_bwd': ('SISO_ROT_KA', 'bwd'), 'adj_static_real': ('SISO_ADJ_STATIC_REAL', 'bwd'), 'gl_packed_x': ('GLUON90_SISO_PACKED_X', 'bwd'), 'gl_dy_tmajor': ('GLUON90_SISO_DY_TMAJOR', 'bwd'), 'gl_dth_store_vec16': ('GLUON90_SISO_DTH_STORE_VEC', 'bwd'), 'gl_narrow_dth_off': ('GLUON90_SISO_NARROW_DTH', 'bwd'), 'gl_ws_producer': ('GLUON90_SISO_WS_PRODUCER', 'bwd'), 'gl_ws_prep': ('GLUON90_SISO_WS_PREP', 'bwd'), 'gl_split2': ('GLUON90_SISO_SPLIT2', 'bwd'), 'real_gluon_kbc': ('GLUON_KBC', 'bwd'), 'real_gate_dy_io': ('GATE_DY_IO', 'bwd'), 'real_kc_shared_cb': ('BWD_KC_SHARED_CB', 'bwd'), 'real_dg_lo': ('BWD_DG_LO', 'bwd'), 'real_cp_g_lo': ('CP_G_LO', 'fwd')}
REAL_UNITS = ('real_gluon_kbc', 'real_gate_dy_io', 'real_kc_shared_cb', 'real_dg_lo', 'real_cp_g_lo')
REAL_GROUP = 'real_R1O1'
REAL_GROUP_HI = 'real_R1O1_hi'
REAL_GROUP_FP32 = 'real_R1O1_fp32'

@cal._const
def _fp32_group_calibrated():
    data, _ = cal.load(FAMILY)
    th = (data or {}).get('thresholds', {})
    return any((k.endswith('|' + REAL_GROUP_FP32) for k in th))

def real_group(T=None, dtype=None):
    if dtype is not None and cal.enabled(FAMILY):
        from .calibration_chunkpar import dtype_axis
        if dtype_axis(dtype) == 'fp32' and _fp32_group_calibrated():
            return REAL_GROUP_FP32
    if T is None:
        return REAL_GROUP
    from .calibration_chunkpar import real_rule
    r = real_rule()
    return REAL_GROUP_HI if r is not None and int(T) == r[0] else REAL_GROUP
_ON_VALUE = {'gl_dth_store_vec16': '16', 'gl_narrow_dth_off': '0'}
REQUIRES = {'gl_ws_prep': 'gl_ws_producer', 'adj_static_real': 'rot_ka_bwd'}
cal.register_family(FAMILY, env='SSD_MIMO_CALIBRATION', code_files=[os.path.join(_HERE, f) for f in ('m3_epilogue.py', 'rotation_active.py', 'siso_fwd_kernel.py', 'siso_bwd_kernel.py', 'gluon_sm90_siso_bwd.py', 'gluon_sm90_siso_ws.py')], kernels=(), shapes={}, notes=dict(real_R1O1='units `real_*` (levers.REAL_UNITS) are measured by tools/calibrate_real.py (leave-one-out at the calibrated chunk structure, fp64 oracle eligibility); tools/calibrate_levers.py carries them over unchanged when it rewrites the file; their kernels (fused_bwd_kernel, chunkpar, gate_epilogue, gluon_kbc_hb) are hashed by the ssd_mimo_fused / ssd_mimo_chunkpar files, so re-run calibrate_real.py whenever those are recalibrated'))

@cal._const
def explicit(unit):
    from .fused_fwd_kernel import _env
    v = _env(UNITS[unit][0], '')
    return v if v != '' else None

def key(unit, group):
    return 'lever|%s|%s' % (unit, group)

@cal._const
def calibrated(unit, group):
    v, _ = cal.threshold(FAMILY, key(unit, group), None)
    return None if v is None else bool(v)
_FORCE = {}
_SUPPRESS = set()
GLUON_UNITS = ('gl_packed_x', 'gl_dy_tmajor', 'gl_dth_store_vec16', 'gl_narrow_dth_off', 'gl_ws_producer', 'gl_ws_prep', 'gl_split2')
RETRY_ON_OOR = True

@contextlib.contextmanager
def suppressed(units):
    old = set(_SUPPRESS)
    _SUPPRESS.update(units)
    try:
        yield
    finally:
        _SUPPRESS.clear()
        _SUPPRESS.update(old)

@contextlib.contextmanager
def forced(choice):
    old = dict(_FORCE)
    _FORCE.clear()
    _FORCE.update(choice)
    try:
        yield
    finally:
        _FORCE.clear()
        _FORCE.update(old)

@cal._const
def on(unit, group, guard=True):
    e = explicit(unit)
    if e is not None:
        return e == _ON_VALUE.get(unit, '1')
    if not guard:
        return False
    req = REQUIRES.get(unit)
    if req is not None and (not on(req, group, True)):
        return False
    if unit in _SUPPRESS:
        return False
    if _FORCE:
        return bool(_FORCE.get(unit, False))
    return bool(calibrated(unit, group))
GL_CFG = ('wn', 'stages', 'preg_prep', 'preg', 'maxnreg')

@cal._const
def cfg_value(name, group):
    v, _ = cal.threshold(FAMILY, 'lever|gl_cfg_%s|%s' % (name, group), None)
    return None if v is None else int(v)
RANK_GROUPS = {4: 'complex_R4O4'}

@cal._const
def _rank_group_calibrated(grp):
    data, _ = cal.load(FAMILY)
    th = (data or {}).get('thresholds', {})
    return any((k.endswith('|' + grp) for k in th))

@cal._const
def group_of(R, O, rot=True):
    if rot and R in RANK_GROUPS and (O in (1, R)) and cal.enabled(FAMILY) and _rank_group_calibrated(RANK_GROUPS[R]):
        return RANK_GROUPS[R]
    return '%s_%s' % ('complex' if rot else 'real', 'R1O1' if R == 1 and O == 1 else 'ROgt1')
_DECL = threading.local()

@contextlib.contextmanager
def declare_active_pairs(ka):
    old = getattr(_DECL, 'ka', None)
    _DECL.ka = ka
    try:
        yield
    finally:
        _DECL.ka = old

def declared_active_pairs():
    return getattr(_DECL, 'ka', None)
