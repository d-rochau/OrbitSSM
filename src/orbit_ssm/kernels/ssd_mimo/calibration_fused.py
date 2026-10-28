"""SSD-MIMO GENERAL FUSED route adapter of the launch-calibration framework
(`src/orbit_ssm/kernels/calibration.py`; reference adapter `calibration_ssd.py`).

Calibrated kernels (the seven `@triton.autotune` backward kernels of
`fused_bwd_kernel.py`, used at every (R, O), the real and the complex route
whenever the SISO route does not serve):

  fused_ka, fused_kb1, fused_kbc_hb (also holds HB), fused_kb2,
  fused_kb2_hb, fused_kinv, fused_kc

ONE fixed config per kernel per GPU (min worst-case slowdown over the shapes
below, R = O = 1 real / complex and R2O2 / R4O4); light = 2-4 configs at the
fixed config's stages (and HB).  Switch `SSD_MIMO_CALIBRATION=on` (default
OFF, shipped defaults unchanged).

Not covered: the fused FORWARD config / chunk length / strategy verdicts --
they keep per-device measured verdict caches (`smem.fwd_cfg_*`,
`smem.chunk_cache_*`, `smem.strategy_*`, written by `SSD_MIMO_FWD_CFG=measure`
etc.) and are structural thresholds rather than Triton launch grids.
Gluon launch parameters (WN / STAGES) are fixed manual knobs.
"""
import os
import re

from .. import calibration as cal

FAMILY = "ssd_mimo_fused"
_HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = ("fused_ka", "fused_kb1", "fused_kbc_hb", "fused_kb2",
           "fused_kb2_hb", "fused_kinv", "fused_kc")

# The shared shape set (`src/orbit_ssm/kernels/shape_sets.py`), filled in per
# group (bf16, the production chunk rule, 'io' operands): R = O = 1 at N = 128
# real / complex, and R2O2 at N = 64 real / complex for every shape; R4O4
# (N = 64, complex) on the LM cells only, to keep the measurement time bounded.
# A cell that does not fit the device is dropped by name at calibration time.
from .. import shape_sets as _cshapes

SHAPES = {}
for _tag in _cshapes.SET:
    _g = _cshapes.geometry(_tag)
    for _rot in (True, False):
        _p = "cx" if _rot else "real"
        SHAPES["%s_r1o1_%s" % (_p, _tag)] = dict(G=64, N=128, R=1, O=1, rot=_rot, **_g)
        SHAPES["%s_r2o2_%s" % (_p, _tag)] = dict(G=64, N=64, R=2, O=2, rot=_rot, **_g)
# `real_R1O1_hi` cells: the real R = O = 1 shapes the H100 structural T rule sends to T_hi = 64 (rule fit over the screen: L <= 4096 and
# B*H*L >= 393216) -- the nine shape-set cells inside it plus the three boundary points of the real-decay calibration's EXTRA shapes that lie inside.
# Measured at T = 64 (workload: `geom["hi"]`); a GPU without a rule never reads this group.
HI_TAGS = ("large_balanced", "large_batch", "large_wide", "lm_b32_d2048_l2048", "lm_b6_d2048_l2048", "lm_b6_d4096_l2048",
           "medium_balanced", "medium_batch", "medium_wide")
HI_EXTRA = {"x_b8_d2048_l4096": (8, 2048, 4096), "x_b2_d8192_l4096": (2, 8192, 4096), "x_b4_d512_l4096": (4, 512, 4096)}
for _tag in HI_TAGS:
    SHAPES["real_hi_r1o1_" + _tag] = dict(G=64, N=128, R=1, O=1, rot=False, hi=True, **_cshapes.geometry(_tag))
for _tag, (_b, _d, _l) in HI_EXTRA.items():
    SHAPES["real_hi_r1o1_" + _tag] = dict(G=64, N=128, R=1, O=1, rot=False, hi=True, B=_b, L=_l, H=_d // _cshapes.P, P=_cshapes.P)
for _tag in _cshapes.LM_TAGS:
    SHAPES["cx_r4o4_" + _tag] = dict(G=64, N=64, R=4, O=4, rot=True,
                                     **_cshapes.geometry(_tag))
# fp32 IO (tf32 operands, torch's default dtype) R = O = 1 cells on the WHOLE shared set, real and complex, every kernel the
# general route launches at fp32: groups `real_R1O1_fp32_t<T>` / `complex_R1O1_fp32_t<T>`, measured at T
# (fp32 real calls launch `fused_kb1`: the head-blocked `kbc_hb` body is 16-bit only).
#: one group per chunk length (`<group>_fp32_t<T>`): the fp32 structure (real T rule, rotating route rule) chooses among them
FP32_CHUNKS = (32, 64)
for _tag in _cshapes.SET:
    _g = _cshapes.geometry(_tag)
    for _T in FP32_CHUNKS:
        SHAPES["real_r1o1f32t%d_%s" % (_T, _tag)] = dict(G=64, N=128, R=1, O=1, rot=False, dtype="fp32", T=_T, **_g)
        SHAPES["cx_r1o1f32t%d_%s" % (_T, _tag)] = dict(G=64, N=128, R=1, O=1, rot=True, dtype="fp32", T=_T, **_g)

# PER-RANK launch groups (the per-rank calibration): R = O = 2 and 4, real and rotating, bf16 and fp32 IO, at the chunk lengths the per-rank
# T screen can choose (`calibration_chunkpar.rank_chunk`): group `<route>_R<R>O<O>[_fp32]_t<T>`.  A call at a rank / T / dtype without such a group
# reads the shared `*_ROgt1` group (recorded at T = 32).  Cells measured at T (= the backward's chunk); R = 4 on the LM cells only.
RANK_CHUNKS = {2: (32, 64), 4: (16, 32, 64)}
for _R, _Ts in RANK_CHUNKS.items():
    for _tag in (_cshapes.SET if _R == 2 else _cshapes.LM_TAGS):
        _g = _cshapes.geometry(_tag)
        for _T in _Ts:
            for _rot in (True, False):
                for _dt in ("bf16", "fp32"):
                    SHAPES["%s_r%do%d%st%d_%s" % ("cx" if _rot else "real", _R, _R, "f32" if _dt == "fp32" else "", _T, _tag)] = dict(
                        G=64, N=64, R=_R, O=_R, rot=_rot, dtype=_dt, T=_T, rk=True, **_g)


def group_key(rot, R, O, hi=False, dtype="bf16", T=None, rk=False, **_):
    """The calibration groups: {real, complex} x {R=O=1, R/O>1}, plus `real_R1O1_hi`: the real R = O = 1 calls the GPU's structural T rule
    (`calibration_chunkpar.real_rule`) sends to the longer chunk T_hi -- their own fixed / light configs, measured at T_hi.
    fp32 IO is a separate group (suffix `_fp32`, the dtype axis): every MMA operand is twice as wide.
    `rk` (a per-rank cell, R = O > 1): the group `<route>_R<R>O<O>[_fp32]_t<T>` (see `RANK_CHUNKS`)."""
    if rk:
        return "%s_R%dO%d%s_t%d" % ("complex" if rot else "real", R, O, "_fp32" if dtype in ("fp32", "float32") else "", int(T))
    if dtype in ("fp32", "float32"):
        return "%s_%s_fp32_t%d" % ("complex" if rot else "real", "R1O1" if (R == 1 and O == 1) else "ROgt1",
                                   int(T) if T else 32)
    if hi and not rot and R == 1 and O == 1:
        return "real_R1O1_hi"
    return "%s_%s" % ("complex" if rot else "real",
                      "R1O1" if (R == 1 and O == 1) else "ROgt1")


def group_for(rot, R, O, T=None, dtype="bf16"):
    """The calibration group of a call at chunk length `T` (the structural rule's T_hi selects `real_R1O1_hi`); `dtype` = the
    call's IO dtype tag (fp32 IO selects the `_fp32` groups)."""
    if R > 1 and R == O and T is not None and os.environ.get("SSD_MIMO_RANK_GROUPS", "1") != "0":   # =0: the shared ROgt1 groups (the pre-rank behaviour; the per-rank `verify`'s old arm)
        rg = group_key(rot, R, O, dtype=dtype, T=T, rk=True)
        data, _ = cal.load(FAMILY)
        if rg in ((data or {}).get("chunk") or {}):
            return rg                        # this rank's own calibrated launch group at this chunk
    g = group_key(rot, R, O, dtype=dtype, T=T)
    if g == "real_R1O1" and T is not None:
        from .calibration_chunkpar import real_rule
        r = real_rule()
        if r is not None and int(T) == r[0]:
            return "real_R1O1_hi"
    return g


# ---- codec: Triton warps/stages/maxnreg (+ HB for Kbc-hb) -----------------
def _key(c):
    hb = c.get("HB")
    return ("hb%d" % hb if hb is not None else "") + cal.cfg_key(
        c["num_warps"], c["num_stages"], c["maxnreg"])


def _parse(k):
    hb = None
    if k.startswith("hb"):
        i = k.index("w")
        hb, k = int(k[2:i]), k[i:]
    c = cal.cfg_parse(k)
    if hb is not None:
        c["HB"] = hb
    return c


def _class(c):
    return (c["num_stages"], c.get("HB"))


def _make(c):
    import triton
    kw = dict(num_warps=c["num_warps"], num_stages=c["num_stages"])
    if c["maxnreg"]:
        kw["maxnreg"] = c["maxnreg"]
    return triton.Config({"HB": c["HB"]} if c.get("HB") is not None else {},
                         **kw)


def _obj_key(cfg):
    d = dict(num_warps=cfg.num_warps, num_stages=cfg.num_stages,
             maxnreg=cfg.maxnreg)
    if "HB" in cfg.kwargs:
        d["HB"] = cfg.kwargs["HB"]
    return _key(d)


cal.register_family(
    FAMILY, env="SSD_MIMO_CALIBRATION",
    code_files=[os.path.join(_HERE, "fused_bwd_kernel.py")],
    kernels=KERNELS, shapes=SHAPES, group_key=group_key,
    occupancy=lambda B, H, **_: B * H,
    axes=("num_warps", "num_stages", "maxnreg", "HB"),
    cfg_key=_key, cfg_parse=_parse, cfg_class=_class, make_config=_make,
    config_key=_obj_key, chunk_of=lambda g: _chunk_of(g),
    notes=dict(
        fp32="real_r1o1f32t<T>_* / cx_r1o1f32t<T>_* cells (all 18 x T in {32, 64}, fp32 IO, tf32 operands, R = O = 1): groups "
             "real_R1O1_fp32_t<T> / complex_R1O1_fp32_t<T>, measured at T; they replace the six LM-only real_r1o1fp32_* kb1 cells.",
        shape_set="src/orbit_ssm/kernels/shape_sets.py: 12 grid cells (3 size classes x "
                  "balanced/long/wide/batch at equal work) + 6 LM cells, (B, d_inner, L), P = 64",
        r4o4_deviation="DEVIATION from the shared set: R4O4 (complex, N = 64) is "
                       "measured on the 6 LM cells only (measurement-time bound); "
                       "R = O = 1 (N = 128) and R2O2 (N = 64), real and complex, are "
                       "measured on all 18 cells.  Cells that ran out of memory are "
                       "listed in dropped_shapes."))


def _real_r1o1_chunks(G=64, dtype="bf16"):
    """(T_fwd, T_bwd) of the real R = O = 1 route of a TRAINING call: the per-GPU calibrated
    structural choice (`calibration_chunkpar.real_chunk`) or the chunk rule; env
    `SSD_MIMO_CHUNK` / `SSD_MIMO_CHUNK_BWD` pin them (the measurement children).  `dtype` = the IO dtype tag
    (fp32 has its own calibrated structure when the GPU's file stores one)."""
    from ...ssm.ssd_mimo import _scan_chunk, _scan_chunk_bwd
    tf = _scan_chunk(1, G, 1, True, real=True, dtype=dtype)
    tb = _scan_chunk_bwd(1, G, 1, tf, real=True, dtype=dtype)
    return tf, (tf if tb is None else tb)


def _chunk_of(group):
    """The chunk length the fused cells are measured at: the production
    chunk RULE (`_scan_chunk`) of the group's cells (see the offline driver
    `workload` of `tools/calibration_drivers/ssd_fused.py`); for the
    real R = O = 1 group the BACKWARD's calibrated chunk (these are backward
    kernels); the fp32 groups at their own fp32 chunk (`_fp32_chunk`)."""
    if "_fp32_t" in group or re.search(r"_R\dO\d_t\d+$", group):
        return int(group.rsplit("_t", 1)[1])
    if group == "real_R1O1":
        return _real_r1o1_chunks()[1]
    if group == "real_R1O1_hi":
        from .calibration_chunkpar import real_rule
        r = real_rule()
        return 64 if r is None else r[0]
    from ...ssm.ssd_mimo import _scan_chunk
    cells = [g for g in SHAPES.values()
             if group_key(g["rot"], g["R"], g["O"], **{k: v for k, v in g.items() if k == "hi"}) == group]
    ts = {_scan_chunk(g["R"], g["G"], g["O"]) for g in cells}
    return ts.pop() if len(ts) == 1 else None


def grid(kernel, tier, *, full, light, generic_pin):
    return cal.calibrated_grid(FAMILY, kernel, tier, full=full, light=light,
                               generic_pin=generic_pin)


#: R = O = 8 is NOT tuned: its launch configs are the SMALLEST that compile and run on every arch (sm_80 / sm_90 / sm_120), with no search
#: and no per-GPU file entry: T = 16 (the chunk rule's R = 8 row, `ssm.ssd_mimo._CHUNK_RANK_TABLE`), Triton warps = 4, stages = 1, no register cap, the
#: smallest head block `HB` of the head-blocked kernels.  The rank-calibration tests pin the selection and run the route on the GPU.
R8_GROUP = "R8O8_safe"
R8_SAFE = dict(T=16, num_warps=4, num_stages=1, maxnreg=None)


def is_r8(R, O):
    return int(R) >= 8 and int(O) >= 8


def r8_safe_configs(configs):
    """The one config of `configs` (a kernel's candidate grid) the R = O = 8 route launches: warps = `R8_SAFE` warps and stages = 1 with no register cap
    (the smallest head block when the kernel has one); where the grid has no such config, the smallest (stages, HB, |warps - 4|, maxnreg) of it."""
    def rank(c):
        return (c.num_stages, c.kwargs.get("HB", 0), abs(c.num_warps - R8_SAFE["num_warps"]), c.maxnreg or 0)
    exact = [c for c in configs if c.num_warps == R8_SAFE["num_warps"] and c.num_stages == R8_SAFE["num_stages"] and not c.maxnreg]
    pool = exact or list(configs)
    if not pool:
        return configs
    best = min(pool, key=rank)
    return [best]


def activate(rot, R, O, B, H, T=None, dtype="bf16"):
    """fused_chunk_bwd_5kernel entry: select the call's shape group (incl. the dtype axis) and
    occupancy (blocks per SM = B*H / SMs)."""
    if cal.enabled(FAMILY):
        if is_r8(R, O):
            cal.activate(FAMILY, R8_GROUP, autotuners(), occupancy=None, T=T)
            return
        cal.activate(FAMILY, group_for(rot, R, O, T, dtype), autotuners(),
                     occupancy=B * H / cal.device_info()["sms"], T=T)


def prune(configs, kernel):
    if cal.enabled(FAMILY) and cal.active_group(FAMILY) == R8_GROUP:
        return r8_safe_configs(configs)
    return cal.prune(configs, FAMILY, kernel)


def autotuners():
    from . import fused_bwd_kernel as fb
    return {"fused_ka": fb._bwd_ka_kernel, "fused_kb1": fb._bwd_kb1_kernel,
            "fused_kbc_hb": fb._bwd_kbc_hb_kernel,
            "fused_kb2": fb._bwd_kb2_kernel,
            "fused_kb2_hb": fb._bwd_kb2_hb_kernel,
            "fused_kinv": fb._bwd_kinv_kernel, "fused_kc": fb._bwd_k2_kernel}


def describe_served() -> dict:
    out = {"calibration_enabled": cal.enabled(FAMILY),
           "code_hash": cal.code_hash(FAMILY)}
    for name, k in autotuners().items():
        n = len(k.configs)
        best = getattr(k, "best_config", None)
        c = best if (n > 1 and best is not None) else (
            k.configs[0] if n == 1 else None)
        out[name] = dict(
            candidates=n, grid_source=cal.source(FAMILY, name),
            served=("search-chose" if n > 1 and best is not None else
                    "untuned" if n > 1 else "fixed"),
            config=None if c is None else dict(
                num_warps=c.num_warps, num_stages=c.num_stages,
                maxnreg=c.maxnreg, **dict(c.kwargs)))
    return out


def smem_demote(T) -> bool:
    """See `calibration.smem_demote`: the backward repair loop calls this on `OutOfResources` before it halves the chunk."""
    return cal.smem_demote(FAMILY, T)
