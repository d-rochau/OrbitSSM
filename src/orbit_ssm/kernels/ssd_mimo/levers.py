"""Calibrated speed-only LEVERS of the SSD-MIMO kernels (family
`ssd_mimo_levers` in the calibration files).

A lever is a numerically-equivalent speed option (it moves the same
arithmetic, or skips terms that are exactly zero by structure) that is faster
on some GPUs / call groups and slower on others, so no single global default
is right.  Each lever UNIT is one (lever, phase) pair -- `phase` is `fwd`
(forward kernels) or `bwd` (backward kernels) -- because a lever can win on
one pass and lose on the other (bwd-on / fwd-off).

Dispatch order of `on(unit, group, guard=...)`:

  1. EXPLICIT environment (`SSD_MIMO_<ENV>` / an active `kernel_options`
     value, the A/B and test channel): "1" (the unit's on-value) -> on,
     any other value -> off.  Explicit settings never consult calibration.
  2. APPLICABILITY GUARD (`guard`, computed by the call site from the call:
     shapes, strides, dtype, static rotations, the Gluon route's own
     envelope ...).  A false guard turns the unit OFF for this call --
     no error, nothing silently wrong.  (An explicit env "1" with a false
     guard is the site's own business: it may raise.)
  3. CALIBRATED CHOICE: `thresholds["lever|<unit>|<group>"]["value"]` of this
     GPU's `ssd_mimo_levers` file (valid provenance only), when the family
     switch `SSD_MIMO_CALIBRATION` is on.
  4. DEFAULT: off.

A unit that fails the fp64 numerics stress on a GPU is INELIGIBLE there: the
calibration tool stores value 0 with `evidence.eligible = false`, never a
timing-based choice.

`SISO_ROT_KA` is the one lever whose applicability is a DECLARATION about
the call's structure (the angle tail is exactly zero: `rope_fraction` < 1
zero-pads the angles), not about its shapes.  The caller declares the active
pair count with `declare_active_pairs(ka)`; without a declaration the unit
is never on (an explicit integer env value is taken as given).
"""
import contextlib
import os
import threading

from .. import calibration as cal

FAMILY = "ssd_mimo_levers"
_HERE = os.path.dirname(os.path.abspath(__file__))

# unit -> (env suffix, phase, what "on" means)
UNITS = {
    "phase_law_fwd_active_k": ("PHASE_LAW_FWD_ACTIVE_K", "fwd"),
    "phase_law_active_k": ("PHASE_LAW_ACTIVE_K", "bwd"),
    "phase_law_headpar": ("PHASE_LAW_HEADPAR", "bwd"),
    "phase_law_suffix": ("PHASE_LAW_SUFFIX_KERNEL", "bwd"),
    "rot_ka_fwd": ("SISO_ROT_KA", "fwd"),
    "rot_ka_bwd": ("SISO_ROT_KA", "bwd"),
    "adj_static_real": ("SISO_ADJ_STATIC_REAL", "bwd"),
    "gl_packed_x": ("GLUON90_SISO_PACKED_X", "bwd"),
    "gl_dy_tmajor": ("GLUON90_SISO_DY_TMAJOR", "bwd"),
    "gl_dth_store_vec16": ("GLUON90_SISO_DTH_STORE_VEC", "bwd"),
    "gl_narrow_dth_off": ("GLUON90_SISO_NARROW_DTH", "bwd"),
    "gl_ws_producer": ("GLUON90_SISO_WS_PRODUCER", "bwd"),
    "gl_ws_prep": ("GLUON90_SISO_WS_PREP", "bwd"),
    "gl_split2": ("GLUON90_SISO_SPLIT2", "bwd"),
    # group real_R1O1 (Mamba-2 class: rotation-free R = O = 1); every one of
    # them is applicable ONLY to a real call (the call sites pass the guard)
    "real_gluon_kbc": ("GLUON_KBC", "bwd"),
    "real_gate_dy_io": ("GATE_DY_IO", "bwd"),
    "real_kc_shared_cb": ("BWD_KC_SHARED_CB", "bwd"),
    "real_dg_lo": ("BWD_DG_LO", "bwd"),
    "real_cp_g_lo": ("CP_G_LO", "fwd"),
}
#: the units of group real_R1O1
REAL_UNITS = ("real_gluon_kbc", "real_gate_dy_io", "real_kc_shared_cb",
              "real_dg_lo", "real_cp_g_lo")
REAL_GROUP = "real_R1O1"
#: the same group's lever decisions AT the structural rule's `T_hi` (the calls the rule sends to the longer chunk: its own ablation)
REAL_GROUP_HI = "real_R1O1_hi"


#: the fp32-IO lever group (tf32 operands): its own decisions when the GPU's levers file stores any `lever|*|real_R1O1_fp32` key;
#: otherwise an fp32 call reads the 16-bit group's decisions.  Of the five real units only
#: `real_kc_shared_cb` is reachable at fp32 (the other four are 16-bit-only by their own guards).
REAL_GROUP_FP32 = "real_R1O1_fp32"


@cal._const
def _fp32_group_calibrated():
    data, _ = cal.load(FAMILY)
    th = (data or {}).get("thresholds", {})
    return any(k.endswith("|" + REAL_GROUP_FP32) for k in th)


def real_group(T=None, dtype=None):
    """The lever group of a real R = O = 1 call at chunk length `T`: `real_R1O1_hi` for the calls the calibrated structural T rule
    (`calibration_chunkpar.real_rule`) sends to T_hi, else `real_R1O1`; an fp32-IO call (`dtype`) `real_R1O1_fp32` where calibrated."""
    if dtype is not None and cal.enabled(FAMILY):
        from .calibration_chunkpar import dtype_axis
        if dtype_axis(dtype) == "fp32" and _fp32_group_calibrated():
            return REAL_GROUP_FP32
    if T is None:
        return REAL_GROUP
    from .calibration_chunkpar import real_rule
    r = real_rule()
    return REAL_GROUP_HI if (r is not None and int(T) == r[0]) else REAL_GROUP
# env value that means ON for the units whose "on" is not "1"
_ON_VALUE = {"gl_dth_store_vec16": "16", "gl_narrow_dth_off": "0"}

# Calibration keys of the units that need a prerequisite unit ON for the
# call (a sub-lever of another).  The tool only ablates a unit whose
# prerequisites are in the all-on set, and the runtime requires them too.
REQUIRES = {"gl_ws_prep": "gl_ws_producer",
            "adj_static_real": "rot_ka_bwd"}

cal.register_family(
    FAMILY, env="SSD_MIMO_CALIBRATION",
    code_files=[os.path.join(_HERE, f) for f in (
        "m3_epilogue.py", "rotation_active.py",
        "siso_fwd_kernel.py", "siso_bwd_kernel.py",
        "gluon_sm90_siso_bwd.py", "gluon_sm90_siso_ws.py")],
    kernels=(), shapes={},
    notes=dict(real_R1O1="units `real_*` (levers.REAL_UNITS) are measured by the real-decay calibration "
                         "(leave-one-out at the calibrated chunk structure, fp64 oracle eligibility); "
                         "tools/calibrate_levers.py carries them over unchanged when it rewrites the file; "
                         "their kernels (fused_bwd_kernel, chunkpar, gate_epilogue, gluon_kbc_hb) are "
                         "hashed by the ssd_mimo_fused / ssd_mimo_chunkpar files, so re-run "
                         "the real-decay calibration whenever those are recalibrated"))


@cal._const
def explicit(unit):
    """The explicit setting of the unit's env lever (None when unset)."""
    from .fused_fwd_kernel import _env
    v = _env(UNITS[unit][0], "")
    return v if v != "" else None


def key(unit, group):
    return "lever|%s|%s" % (unit, group)


@cal._const
def calibrated(unit, group):
    """The calibrated value (True / False) or None (switch off, no valid
    file for this GPU, or the unit is not calibrated for the group)."""
    v, _ = cal.threshold(FAMILY, key(unit, group), None)
    return None if v is None else bool(v)


_FORCE = {}
_SUPPRESS = set()
#: Gluon-launch units: when a calibrated combination does not fit the device
#: (shared memory) the launcher retries the call without them
#: (`gluon_sm90_siso_bwd.launch`).  The calibration tool switches the retry
#: off to find the offending units.
GLUON_UNITS = ("gl_packed_x", "gl_dy_tmajor", "gl_dth_store_vec16",
               "gl_narrow_dth_off", "gl_ws_producer", "gl_ws_prep", "gl_split2")
RETRY_ON_OOR = True


@contextlib.contextmanager
def suppressed(units):
    """Calibrated / forced choices of `units` read as OFF inside (explicit
    env settings still win)."""
    old = set(_SUPPRESS)
    _SUPPRESS.update(units)
    try:
        yield
    finally:
        _SUPPRESS.clear()
        _SUPPRESS.update(old)


@contextlib.contextmanager
def forced(choice):
    """In-process override `{unit: bool}` used by the calibration tool to
    measure a unit set through the REAL dispatch (guards and prerequisites
    still apply; explicit env still wins).  Never used at runtime."""
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
        return e == _ON_VALUE.get(unit, "1")
    if not guard:
        return False
    req = REQUIRES.get(unit)
    if req is not None and not on(req, group, True):
        return False
    if unit in _SUPPRESS:
        return False
    if _FORCE:
        return bool(_FORCE.get(unit, False))
    return bool(calibrated(unit, group))


#: numeric launch parameters of the Gluon sm_90 SISO backward that are CALIBRATED per GPU (entries `lever|gl_cfg_<name>|<group>`,
#: value = the integer): the consumer warp multiple `wn`, the TMA pipeline depth `stages`, the producer register request `preg`
#: (with the preparation phase / without), and `maxnreg` (0 = no cap).  Absent = the kernel's own default.
GL_CFG = ("wn", "stages", "preg_prep", "preg", "maxnreg")


@cal._const
def cfg_value(name, group):
    """The calibrated integer of `GL_CFG[name]` for `group`, or None (switch off, no valid file, not calibrated)."""
    v, _ = cal.threshold(FAMILY, "lever|gl_cfg_%s|%s" % (name, group), None)
    return None if v is None else int(v)


#: the PER-RANK rotating lever group (`tools/calibrate_levers.py --rank-micro`): the phase-law units decided at R = 4 on their own cells.  R = 2
#: stays on `complex_ROgt1` (that group IS measured at R = 2); a GPU without `lever|*|complex_R4O4` keys reads `complex_ROgt1` at R = 4.
RANK_GROUPS = {4: "complex_R4O4"}


@cal._const
def _rank_group_calibrated(grp):
    data, _ = cal.load(FAMILY)
    th = (data or {}).get("thresholds", {})
    return any(k.endswith("|" + grp) for k in th)


@cal._const
def group_of(R, O, rot=True):
    """The calibration group of a call (the framework's four groups; plus the per-rank group of R = 4 where calibrated).  The phase-law
    call sites pass O = 1 (the angle chain does not depend on the readout rank), so a rank group answers for O in (1, R)."""
    if rot and R in RANK_GROUPS and O in (1, R) and cal.enabled(FAMILY) and _rank_group_calibrated(RANK_GROUPS[R]):
        return RANK_GROUPS[R]
    return "%s_%s" % ("complex" if rot else "real",
                      "R1O1" if (R == 1 and O == 1) else "ROgt1")


# ------------------------------------------------- SISO_ROT_KA declaration --
_DECL = threading.local()


@contextlib.contextmanager
def declare_active_pairs(ka):
    """Declare, for the SISO scan calls made inside, that only the first
    `ka` of the N/2 phase pairs can be non-zero (the tail angles are exactly
    zero by construction).  `ka` = None declares nothing."""
    old = getattr(_DECL, "ka", None)
    _DECL.ka = ka
    try:
        yield
    finally:
        _DECL.ka = old


def declared_active_pairs():
    return getattr(_DECL, "ka", None)
