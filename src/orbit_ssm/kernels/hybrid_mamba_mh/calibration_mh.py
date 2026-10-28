"""MH adapter of the launch-calibration framework
(`src/orbit_ssm/kernels/calibration.py`, tool `tools/calibrate_kernels.py`).

Calibrated kernels: `mh_fwd` (forward launch: BLOCK_P, BLOCK_L, PF) and
`mh_bwd` (backward launch: TP_P, HP, MAXNREG): per GPU ONE fixed config each
for real and ONE for complex (rotating) launches, chosen
by best worst-case slowdown over that group's `SHAPES`, plus a light set.  Configs are MH pin strings (`kernel_options`
`fwd_config` / `bwd_config`), replayed through the pins, so every measured
config is exactly what a pinned launch runs; legality still clamps a pin
(BLOCK_P to a divisor of P, HP to a legal pack, BLOCK_L under checkpoints).

The backward time split is NOT a config axis here: it is the one backward
choice that moves with the shape (1 at short L, 32 at long L), i.e. a
structural length threshold.  It is stored as the threshold
``mh_bwd_ts`` = {'ts', 'min_nblk'} (split into `ts` super-chunks when the
walk has at least `min_nblk` chunks), measured per shape at the backward's
best untimed-split config.

Level of every number: the `mh_scan` op (kernel level, inputs from
`benchmarks.bench_kernels`), CUDA-event time per call, median over rounds
(the measurement child is the offline driver
`child_run` of `tools/calibration_drivers/mh.py`, not part of the package);
the backward column is (forward+backward) minus the forward of the same
round.  Switch: `HYBRID_MH_CALIBRATION=on` (default OFF).
"""
from __future__ import annotations

import itertools
import math
import os
import time

from .. import calibration as cal

FAMILY = "mh"
_HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = ("mh_fwd", "mh_bwd")
FWD_AXES = ("BLOCK_P", "BLOCK_L", "PF")
BWD_AXES = ("TP_P", "HP", "MAXNREG")
TS_CANDIDATES = (1, 8, 16, 32)

# Representative op cells of the per-group calibration (P > 1): name ->
# dict(B, H, P, N, L, precision, rot).  The P = 1 cells are `P1_SHAPES` below.
SHAPES = {
    "lm_real": dict(B=8, H=128, P=16, N=128, L=2048),
    "lm_rot": dict(B=8, H=128, P=16, N=128, L=2048, rot=True),
    "c15w_rot": dict(B=8, H=32, P=64, N=128, L=2048, rot=True),
    "c15w_rot_b6": dict(B=6, H=32, P=64, N=128, L=2048, rot=True),
    "c15w_rot_l8k": dict(B=2, H=32, P=64, N=128, L=8192, rot=True),
    "c15w_rot_ieee": dict(B=8, H=32, P=64, N=128, L=2048, rot=True, precision="ieee"),
    "p32_rot": dict(B=8, H=64, P=32, N=128, L=2048, rot=True),
}
for _g in SHAPES.values():
    _g.setdefault("precision", "bf16")
    _g.setdefault("rot", False)

_CODE = ("fwd_gluon.py", "bwd_gluon.py", "common_gluon.py", "ts_forward.py",
         "ts_adjoint.py", "autotune.py", "layout.py", "functions.py",
         "slp.py", "half_fold.py")


def cfg_key(c: dict) -> str:
    """'K=v,...' over the config's own keys, sorted -- also the pin string."""
    return ",".join(f"{k}={int(c[k])}" for k in sorted(c))


def cfg_parse(key: str) -> dict:
    out = {}
    for item in filter(None, key.split(",")):
        k, _, v = item.partition("=")
        out[k] = int(v)
    return out


cal.register_family(
    FAMILY, env="HYBRID_MH_CALIBRATION",
    code_files=[os.path.join(_HERE, f) for f in _CODE],
    kernels=KERNELS, shapes=SHAPES, axes=FWD_AXES + BWD_AXES,
    # one fixed config per kernel for real and one for complex launches
    group_key=lambda **g: "complex" if g.get("rot") else "real",
    cfg_key=cfg_key, cfg_parse=cfg_parse,
    # nothing is measured-then-held: the light set may vary every axis
    cfg_class=lambda c: (),
    make_config=dict, config_key=lambda c: cfg_key(c))


def fwd_grid() -> list[dict]:
    return [dict(BLOCK_P=bp, BLOCK_L=bl, PF=pf)
            for bp, bl, pf in itertools.product((1, 2, 4), (256, 512, 1024), (0, 1))]


def bwd_grid() -> list[dict]:
    # MAXNREG 0 = no cap (the pin's spelling of `MH_BWD_MAXNREG<=0`)
    return [dict(TP_P=tp, HP=hp, MAXNREG=r, TS=1)
            for tp, hp, r in itertools.product((2, 4), (4, 8), (0, 168))]


# ------------------------------------------------- P = 1 per-cell table --
P1_NS = (16, 128)
P1_PRECISIONS = ("bf16", "fp32")
#: feature classes of the P = 1 launch that are calibrated.  The key is the
#: set of constexpr-moving features the call carries (softplus on the raw dt,
#: a dt bias, the D skip, a z gate, an initial state): "bare" is the plain
#: scan (`bench_kernels.build_mh`), "m1" the Mamba-1 mixer's call (dt softplus
#: + bias + D skip inside the kernel; `HM m1exact`).  Any other feature set
#: is not in the table and runs on the tuner.
P1_FEATS = {"bare": "sp1.db0.ds0.z0.h0", "m1": "sp1.db1.ds1.z0.h0"}
P1_SIDES = ("fwd", "fwd_trn", "bwd")        # inference fwd, ckpt-writing fwd, bwd
_DT = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}


def p1_shapes() -> dict:
    """{cell: geometry}: the shared shape set (`shape_sets.SET`, H = d_inner,
    P = 1) x N x precision x feature class."""
    from ..shape_sets import SET
    out = {}
    for tag, (B, d, L) in SET.items():
        for N in P1_NS:
            for prec in P1_PRECISIONS:
                for cls in P1_FEATS:
                    out["%s_n%d_%s_%s" % (tag, N, prec, cls)] = dict(
                        B=B, H=d, P=1, N=N, L=L, precision=prec, cls=cls)
    return out


P1_SHAPES = p1_shapes()


def p1_feat(*, delta_softplus, delta_bias, d_skip, z, h_init) -> str:
    return "sp%d.db%d.ds%d.z%d.h%d" % (bool(delta_softplus), delta_bias is not None,
                                        d_skip is not None, z is not None,
                                        h_init is not None)


def p1_cell_key(B, H, N, L, dtype, feat) -> str:
    return "B%d|H%d|N%d|L%d|%s|%s" % (B, H, N, L, _DT.get(str(dtype), str(dtype)), feat)


def p1_fwd_axes(H, L, ckpt_l, ts_ok, infer) -> dict:
    """The forward's candidate space at P = 1 (non-FPACK): every axis the
    runtime tuner descends (`autotune.t_axes` tier `full`, plus its coupled
    corners `t_grid`) over its whole legal range.  BLOCK_L is a multiple of the
    checkpoint stride when the forward writes checkpoints; TS where the time
    split is served; CKV (the inference checkpoint variant) for inference."""
    import triton
    top = triton.next_power_of_2(L)
    axes = dict(BLOCK_L=[b for b in (128, 256, 512, 1024, 2048)
                         if b <= top and (not ckpt_l or b % ckpt_l == 0)],
                PF=[0, 1], NUM_WARPS=[1, 2, 4])
    if ts_ok:
        axes["TS"] = [1, 2, 4, 8, 16]
    if infer:
        axes["CKV"] = [0, 1]
    return axes


def p1_fwd_fpack_corners(H, L, ckpt_l) -> list:
    """The opt-in head-packed forward (`MH_FWD_FPACK`): the tuner's own coupled (BLOCK_P, BLOCK_L, PF)
    corners over the head extent (`autotune.t_grid` at PW = H), BLOCK_P > 1,
    kept in the space so the verdict is per GPU and per cell, not an
    inherited rule."""
    from . import autotune
    base = dict(BLOCK_P=1, BLOCK_L=512, NUM_WARPS=1, PF=0)
    return [dict(c, FPACK=1) for c in autotune.t_grid(base, H, L, ckpt_l)
            if c["BLOCK_P"] > 1]


def p1_bwd_axes(H, N, nblk) -> dict:
    """The backward's candidate space at P = 1 (`autotune.v3_axes` tier `full`
    and `v3_grid`): head pack HP, p-slice TP_P, warps, register cap (0 = none),
    n shards NS, time split TS (long walks), the x_s fold AGX and the BC slab
    count BCS.  BLOCK_L is the forward's checkpoint stride in training and is
    not an axis here."""
    return dict(HP=[h for h in (8, 4, 16, 32, 1) if H % h == 0],
                TP_P=[2, 4, 8, 16, 32], NUM_WARPS=[1, 2, 4], MAXNREG=[0, 168, 128],
                NS=[n for n in (1, 2, 4, 8, 16, 32) if N % n == 0],
                TS=[1, 2, 4, 8, 16, 32] if nblk >= 8 else [1],
                AGX=[0, 1], BCS=[1, 8, 16, 64, 0])


def p1_envelope(*, P, ROT, pmap, bc_head, LAW_MASK, endpoint_own, n_off, nr, N,
                absent) -> bool:
    """The Mamba-1 function: exactly what the table was measured on.  `absent`
    are the optional tensors that must all be None (delta_w, g_B, g_C, beta_*,
    a_dyn, mu, theta...)."""
    return (P == 1 and not ROT and not pmap and not bc_head and LAW_MASK == 1
            and endpoint_own is None and n_off == 0 and nr == N
            and all(t is None for t in absent))


_P1_INJECT: dict = {}        # tests / the calibration tool: {key: {side: cfg str}}
_P1_PARSED: dict = {}
P1_STATS = dict(hit=0, miss=0)


def p1_config(side, B, H, N, L, dtype, feat):
    """The calibrated launch config (a fresh dict, in pin keys) of this P = 1
    cell and side, or None -> the runtime tuner serves it.  None as well when
    calibration is off, when the shipped file is stale for this GPU / Triton /
    kernel code, or when the tier is an explicit `full` search."""
    if not cal.enabled(FAMILY):
        return None
    from . import autotune
    if autotune.tier() == "full":
        return None
    key = p1_cell_key(B, H, N, L, dtype, feat)
    cells = _P1_INJECT
    if key not in cells:
        data, _ = cal.load(FAMILY)
        cells = (data or {}).get("p1_cells") or {}
    ent = cells.get(key)
    txt = ent.get(side) if ent else None
    if not txt:
        P1_STATS["miss"] += 1
        return None
    P1_STATS["hit"] += 1
    c = _P1_PARSED.get(txt)
    if c is None:
        c = _P1_PARSED[txt] = cfg_parse(txt)
    return dict(c)


def nblk(geometry) -> int:
    """Chunks the backward walks at this geometry (checkpointed BLOCK_L)."""
    from .bwd_gluon import auto_block_l
    return math.ceil(geometry["L"] / auto_block_l(geometry["P"], geometry["L"]))


def derive_thresholds(res: dict) -> dict:
    """`mh_bwd_ts`: the (ts, min_nblk) rule with the best worst-case slowdown
    over the shapes, against each shape's best measured split."""
    rows = [(r["nblk"], r["ts_ms"]) for r in res.values() if r.get("ts_ms")]
    if not rows:
        return {}
    best = None
    for ts in TS_CANDIDATES[1:]:
        for m in sorted({n for n, _ in rows}) + [math.inf]:
            sd = []
            for n, t in rows:
                pick = str(ts) if n >= m else "1"
                sd.append(t.get(pick, math.inf) / min(t.values()))
            score = (max(sd), sum(sd) / len(sd))
            if best is None or score < best[0]:
                best = (score, ts, m)
    (worst, mean), ts, m = best
    return {"mh_bwd_ts": dict(
        value=dict(ts=ts, min_nblk=None if m == math.inf else m),
        evidence=dict(worst_slowdown=worst, mean_slowdown=mean,
                      per_shape={s: dict(nblk=r["nblk"], ts_ms=r["ts_ms"])
                                 for s, r in res.items()}))}


def describe_served() -> dict:
    """Calibration state per kernel: enabled, source, fixed config."""
    out = dict(calibration_enabled=cal.enabled(FAMILY), code_hash=cal.code_hash(FAMILY))
    for k in KERNELS:
        hit = cal.lookup(FAMILY, k) if cal.enabled(FAMILY) else None
        out[k] = dict(source=None if hit is None else hit["source"],
                      fixed=None if hit is None else hit["default"]["fixed"])
    out["mh_bwd_ts"] = cal.threshold(FAMILY, "mh_bwd_ts")
    return out
