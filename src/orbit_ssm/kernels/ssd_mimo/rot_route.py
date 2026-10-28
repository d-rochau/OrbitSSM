"""CALIBRATED structural route of a rotating R = O = 1 bf16 call (family `ssd_mimo_route`).

Three routes compute the same function:

  siso        SISO forward + SISO scan adjoint, chunk T = 64 (the dedicated one-CTA-per-(batch, head) schedule)
  split<T>    ROTSPLIT: rot_prologue -> real-body fused forward -> real-body 5-kernel backward -> rot_adjoint (grad only)
  inkernel<T> the general fused route with the rotation inside the kernels (also the no-grad general route)

Which one is fastest depends on the GPU and on the call: `siso` serialises the whole sequence of a (b, h) pair in ONE
CTA, so it wins where launches dominate (few CTAs, short sequences) and loses where the device is under-filled
by a long walk or where the general route's chunk-parallel pieces fill it.  The choice is therefore a calibrated per-GPU
structural rule (like the chunk-parallel crossover), measured by `tools/calibrate_routes.py` on the shared shape set
(`src/orbit_ssm/kernels/shape_sets.py`) through the production entry:

    use `siso` iff  (B*H / SMs  <|>=  siso_occ)  and  (L  <=|>  siso_L)        (else the stored alternative `alt`)

with the DIRECTION of each half chosen by the fit (`*_dir`: 0 = siso below the threshold, 1 = siso at/above it): one rule
per pass (`grad` = a call with a backward, `nograd`) and per group (`thh` = the call has a static read offset theta_h, `noth`),
thresholds `route|<pass>|<group>|siso_occ` (x100), `|siso_occ_dir`, `|siso_L`, `|siso_L_dir`, `|alt` (index into `ALTS`).  A route is ELIGIBLE on a GPU only if it passes there (fp64 stress and C10
+-theta_h under the current per-token / kappa / t-test rules); the tool never stores an ineligible alternative, and
the eligibility verdicts are recorded as `route|<pass>|eligible|<route>` thresholds.

Dispatch order: an EXPLICIT lever of the consumer (`SSD_MIMO_SISO_FAST`, `SSD_MIMO_ROT_BWD`, `SSD_MIMO_CHUNK`) is never
overridden; otherwise the calibrated rule; with no valid calibration, `pick` returns None and the call keeps the
default (siso).
"""
import os

from .. import calibration as cal

FAMILY = "ssd_mimo_route"
_HERE = os.path.dirname(os.path.abspath(__file__))

cal.register_family(FAMILY, env="SSD_MIMO_CALIBRATION",
                    code_files=[os.path.join(_HERE, "rot_route.py")], kernels=(), shapes={})

#: alternatives to `siso`, by stored index: (kind, chunk T)
ALTS = (("split", 32), ("split", 64), ("inkernel", 32), ("inkernel", 64))
ALT_NAMES = tuple("%s%d" % a for a in ALTS)
SISO = ("siso", 64)
INF = 10 ** 9


def key(phase, group, field):
    return key_(phase, group, field)


def key_(phase, group, field):
    return "route|%s|%s|%s" % (phase, group, field)


def siso_region(bh_per_sm, L, thr):
    """True iff (bh_per_sm, L) is inside the stored SISO region; None when not calibrated."""
    occ, lmax, alt = thr("siso_occ"), thr("siso_L"), thr("alt")
    if occ is None or lmax is None or alt is None:
        return None
    o_in = (bh_per_sm < occ / 100.0) if not thr("siso_occ_dir") else (bh_per_sm >= occ / 100.0)
    l_in = (L <= lmax) if not thr("siso_L_dir") else (L > lmax)
    return o_in and l_in


def _table_route(bh_per_sm, L, table):
    """Nearest measured cell in (log2 occupancy, log2 L): `table` = [[log2 occ, log2 L, route], ...], route -1 = siso, else an index
    into `ALTS`.  The fallback form of the rule when no two-threshold region reaches the 1.02 gate (`tools/calibrate_routes.py`)."""
    import math
    lo, ll = math.log2(max(bh_per_sm, 1e-3)), math.log2(max(L, 1))
    best = min(table, key=lambda r: (r[0] - lo) ** 2 + (r[1] - ll) ** 2)
    return SISO if int(best[2]) < 0 else ALTS[int(best[2])]


def decide(bh_per_sm, L, thr):
    """Pure rule: `thr(field) -> value|None` -> (kind, T) or None when nothing is calibrated.  A stored `table` (the 2D fallback)
    takes precedence over the two-threshold region."""
    table = thr("table")
    if table:
        return _table_route(bh_per_sm, L, table)
    r = siso_region(bh_per_sm, L, thr)
    if r is None:
        return None
    return SISO if r else ALTS[int(thr("alt"))]


_MEMO: dict = {}      # (bh, L, needs_bwd, thh, device, mode switch, dtype axis) -> decision: the lookup is a dict hit on the host path

#: THE DTYPE AXIS: an fp32-IO call (tf32 operands, torch's default dtype) has its own rule, phases `grad_fp32` /
#: `nograd_fp32` (keys `route|grad_fp32|noth|...`), and its SISO chunk is a calibrated field (`siso_T`: the fp32 SISO schedule's
#: operand tiles are twice as wide, so its T is chosen per GPU, 32 or 64).  No fp32 keys: `pick` returns None for fp32 and the
#: call keeps the general route.


def phase_of(needs_bwd, dtype=None):
    """The rule's PASS key: grad / nograd, with the `_fp32` suffix for fp32 IO."""
    return ("grad" if needs_bwd else "nograd") + ("_fp32" if dtype in ("fp32", "float32") else "")


def pick(bh, L, device, needs_bwd, thh=False, dtype=None):
    """(kind, T) of the calibrated route for a call with `bh` = B*H serial CTAs and sequence length `L`, or None
    (calibration off / no valid file / pass not calibrated): the caller keeps the default route.  `dtype`: "fp32" for
    an fp32-IO call (its own rule), else the 16-bit rule."""
    ax = "fp32" if dtype in ("fp32", "float32") else ""
    key = (bh, L, needs_bwd, thh, device, os.environ.get("SSD_MIMO_CALIBRATION"), ax)
    hit = _MEMO.get(key, _MEMO)
    if hit is not _MEMO:
        return hit
    r = _pick(bh, L, device, needs_bwd, thh, ax)
    _MEMO[key] = r
    return r


def _pick(bh, L, device, needs_bwd, thh, ax=""):
    if not cal.enabled(FAMILY):
        return None
    from .fused_fwd_kernel import _sm_count
    phase = phase_of(needs_bwd, ax or None)
    grp = "thh" if thh else "noth"
    thr = lambda f: cal.threshold(FAMILY, key_(phase, grp, f), None)[0]
    r = decide(bh / _sm_count(device), L, thr)
    if r is not None and r[0] == "siso" and ax:
        t = thr("siso_T")
        r = ("siso", int(t) if t else 64)
    return r


def explicit_chunk():
    return bool(os.environ.get("SSD_MIMO_CHUNK"))


def explicit_siso():
    from .fused_fwd_kernel import _env
    return _env("SISO_FAST", "auto").strip().lower() not in ("", "auto")


def explicit_rot_bwd():
    from .fused_fwd_kernel import _env
    return bool(_env("ROT_BWD", ""))
