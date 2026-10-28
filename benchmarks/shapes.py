"""Named shape grids, each cross-referenced to its source of truth.

The point of this module is that a re-measurement is PROVABLY the same
grid as the one it replaces.  Every constant below carries a comment
naming the file and line range it came from; if you change a grid, change
the comment, and if the source of truth moved, say so.

Vocabulary (kernel level):

    B  batch                       L  sequence length
    H  kernel heads                N  state width
    G  head width (SSD-MIMO: G_SIZE, the per-head channel block)
    P  head width (mh_scan: x is (B, H, P, L); D = H * P)
    R  input  (write) rank          O  output (read) rank
    T  chunk size
"""

from __future__ import annotations

from dataclasses import dataclass


def ssd_rows(rotate: bool, law: str, complex_frac: float = 1.0) -> list[dict]:
    """SSD-MIMO `row_blocks`: one decay `law` for the whole
    state ("head_dynamic" = the pre-spec a_dyn=True, "head_static" =
    a_dyn=False); `rotate` puts the leading `complex_frac` of the rows in a
    complex block (the pre-spec rot_mode="m3", rope_fraction=complex_frac).
    """
    if not rotate:
        return [{"frac": 1.0, "decay": law, "complex": False}]
    # The complex block is gated explicitly
    # (its default is gate=False), so the measured shapes keep their model
    rows = [{"frac": complex_frac, "decay": law, "complex": True, "gate": True}]
    if complex_frac < 1.0:
        rows.append({"frac": 1.0 - complex_frac, "decay": law,
                     "complex": False})
    return rows


# ═════════════════════════════════════════════════════════════════════
# Paper Appendix C.3 — the three tables the re-measurement must refill
# ═════════════════════════════════════════════════════════════════════

# Table 10 (tab:kernel_timings): three CONSTANT-WORK shapes (B, D, L),
# N = 64, ms per call, each cell an fp32/bf16 pair.
# Source: the paper's kernel-timing table (Appendix C.3, Table 10).
PAPER_C3_SHAPES = (
    dict(B=8, D=2048, L=2048),
    dict(B=2, D=4096, L=4096),
    dict(B=4, D=512, L=16384),
)
PAPER_C3_N = 64

# The "paper" bench's one-shape subset of `paper_c3`:
# "the P ladder (P=1/2/4/8/16) fwd and fwd+bwd at the spine
# shape only".  The first of the three constant-work shapes above IS the
# LM spine (matches `LM_SPINE`'s B=8, D=2048, L=2048).
PAPER_C3_SHAPES_PAPER = (PAPER_C3_SHAPES[0],)

# The retired warp-scan family's `R = 1..16` rows had a matmul/cadence
# axis mh_scan does not have: its rank axis IS the head width P in
# x:(B, H, P, L) (functions.py:279-289 has no matmul argument at all).
# The honest re-spelling of the R rows is therefore P at fixed D = H*P.
# OPEN: whether that mapping is accepted, and whether the `s`-cadence block
# leaves the paper.
PAPER_C3_HEAD_WIDTHS = (1, 2, 4, 8, 16)

# Table 11 (tab:kernel_timings_nscale): B = 6, D = H*P = 2048, L = 2048,
# gates off, N sweep, rows = reference / P=1 / P=4 / P=16, each with an
# fp32 and a bf16 row, columns fwd and +bwd.
# Source: the paper's kernel-timing table (Appendix C.3, Table 11).
PAPER_C3_NSWEEP = dict(B=6, D=2048, L=2048,
                       N=(16, 32, 64, 128, 256),
                       head_widths=(1, 4, 16))

# Figure 4 (fig:kernel_timings_lscale): fp32 forward+backward vs L, log-log.
# DISCREPANCY, unresolved: the caption says "fixed batch B = 1 (so B*D =
# 2048 keeps every SM saturated), D = H*R = 2048"
# (the paper's Figure 4 caption) while the generator that produced the
# shipped PDF sets LSCALE_BSZ, LSCALE_D = 1, 512
# (the earlier timing script).
# We default to the CAPTION's shape (that is what the paper claims) and
# expose the generator's as `PAPER_C3_LSWEEP_D512` so the two can be run
# side by side.  GH200 has 132 SMs, so B*D = 512 is
# under one wave, not "4x saturation".
PAPER_C3_LSWEEP = dict(B=1, D=2048, N=64,
                       L=(2048, 4096, 8192, 16384, 32768))
PAPER_C3_LSWEEP_D512 = dict(B=1, D=512, N=64,
                            L=(2048, 4096, 8192, 16384, 32768))

# ═════════════════════════════════════════════════════════════════════
# SSD-MIMO — the signed contract's own grids
# ═════════════════════════════════════════════════════════════════════

# The 26-cell Mamba-2 parity sweep that produced the contract's
# results table.  All R = O = 1, N = 64, T = 64, bf16 IO, real random dY.
# The runnable ancestor is the retired parity-sweep bench (git history).
SSD_PARITY_N = 64
# State size: every kernel-
# parity grid ran ONLY at N=64, but official Mamba-2/Mamba-3 both
# default to d_state=128 (`mamba_ssm.modules.mamba2.Mamba2`/`mamba3.
# Mamba3`'s own constructor defaults, confirmed by introspection) -- so
# "1.0x at Mamba-2's config" was established at N=64 only.  B9 adds this
# as the second column of `ssd_parity26`/`ssd_parity_m3`/
# `ssd_parity_m3_eq` (K15's own "Grid fix" line): N=64 stays (the
# ORIGINAL contract shape, kept as the secondary/full-grid column),
# N=128 becomes the HEADLINE (the officials' own default, what a
# like-for-like claim must be measured at).  `SSD_PARITY_N_VALUES` is
# the pair every one of those three grids now loops over; `PAPER_SETS`
# selects the N=128 column only (`--cells full` still gets both).
SSD_PARITY_N128 = 128
SSD_PARITY_N_VALUES = (SSD_PARITY_N, SSD_PARITY_N128)
SSD_PARITY_T = 64
SSD_PARITY_REF = dict(H=32, G=64)     # "ref" configuration
SSD_PARITY_PROD = dict(H=13, G=128)   # "prod" (campaign) configuration

SSD_PARITY26 = tuple(
    [dict(sweep="batch_ref", B=b, **SSD_PARITY_REF, L=2048) for b in (1, 2, 4, 8, 16)]
    + [dict(sweep="batch_prod", B=b, **SSD_PARITY_PROD, L=2048) for b in (1, 2, 4, 8, 16)]
    + [dict(sweep="len_ref", B=4, **SSD_PARITY_REF, L=l) for l in (512, 2048, 8192)]
    + [dict(sweep="len_prod", B=4, **SSD_PARITY_PROD, L=l) for l in (512, 2048, 8192)]
    + [dict(sweep="hg_grid", B=4, H=h, G=g, L=2048) for (h, g) in
       ((8, 64), (16, 128), (32, 32), (32, 64), (32, 128),
        (48, 64), (64, 64), (64, 128), (8, 256), (13, 128))]
)
assert len(SSD_PARITY26) == 26, "the 26-cell grid must have 26 cells"

# The "paper" bench's 6-cell representative subset of the 26: "the two
# batch extremes, two lengths, two head-width extremes of the 26".  Keyed
# on (sweep, B, H, G, L) so `bench_kernels.py`'s PAPER_SETS predicates for
# `ssd_parity26`/`ssd_parity_m3`/`ssd_parity_m3_eq` can match a cell's own
# `shape` dict without re-deriving the selection:
#   batch extremes  -- batch_ref B=1 and B=16 (SSD_PARITY_REF, L=2048)
#   two lengths     -- len_ref L=512 and L=8192 (B=4, SSD_PARITY_REF)
#   head-width      -- hg_grid's min G (32) and max G (256) pairs
#                      extremes    (B=4, L=2048)
SSD_PARITY26_PAPER_KEYS = frozenset((
    ("batch_ref", 1, 32, 64, 2048),
    ("batch_ref", 16, 32, 64, 2048),
    ("len_ref", 4, 32, 64, 512),
    ("len_ref", 4, 32, 64, 8192),
    ("hg_grid", 4, 32, 32, 2048),
    ("hg_grid", 4, 8, 256, 2048),
))
SSD_PARITY26_PAPER = tuple(
    c for c in SSD_PARITY26
    if (c["sweep"], c["B"], c["H"], c["G"], c["L"]) in SSD_PARITY26_PAPER_KEYS
)
assert len(SSD_PARITY26_PAPER) == 6, \
    "the paper subset of ssd_parity26 must have exactly 6 cells"

# Long-L envelope: the regime the segmented forward closed
# (B1 L32k 1.42 -> 0.84, B1 L64k 1.41 -> 0.75).
SSD_LONG_L = tuple(
    dict(sweep="long_l", B=b, **SSD_PARITY_REF, L=l)
    for b in (1, 2, 4, 8) for l in (8192, 32768, 65536)
)

# The "paper" bench's subset: only the two headline lengths the contract
# doc actually numbers (B1 L32k 0.84x, B1 L64k 0.75x, see the comment
# above) -- the mid lengths (8192) and the smaller batches (2, 4, 8) are
# not individually quoted anywhere.
SSD_LONG_L_PAPER = tuple(
    dict(sweep="long_l", B=1, **SSD_PARITY_REF, L=l) for l in (32768, 65536)
)

# MIMO gate cells: what (R, O) and G actually cost, against the R=O=1
# base and against Mamba-2 run R*O times.  d_inner 2048 = H * G * R.
# Envelope: R/O in {1,2,4,8},
# N pow2 in [16,128], G zero-padded host-side to a multiple of 16.
MIMO_GATES = tuple(
    dict(B=2, L=2048, d_inner=2048, R=r, O=o, G=g, N=n)
    for (r, o) in ((1, 1), (2, 2), (4, 4))
    for g in (16, 32, 64, 128)
    for n in (64, 128)
)

# The R4O4 gate cell: the contract says <= 18 ms; MET at 10.55 ms
# bf16 / 12.87 ms fp32 on GH200.  Kept as its own named cell
# because the sbatch script gives it a dedicated shard.
SSD_R4O4_GATE = dict(B=2, L=2048, d_inner=2048, R=4, O=4, G=128, N=128)

# Feature cells: each SSD-MIMO feature priced against its own absence at
# one fixed shape.  Source: the contract feature list + Section 3's
# per-feature parity requirement.
SSD_FEATURE_BASE = dict(B=4, H=32, G=64, R=1, O=1, N=64, L=2048, T=64)
SSD_FEATURES = ("none", "theta", "theta_h", "rot_bias", "beta", "gamma",
                "gate_z", "delta_w", "x0", "bc_per_head")

# Chunk sizes the contract admits.  T = 128 has never been benched.
SSD_CHUNKS = (64, 128)

# ═════════════════════════════════════════════════════════════════════
# mh_scan (HybridMamba-MH) — the family's own contract shapes
# ═════════════════════════════════════════════════════════════════════

# The four shapes every mh feature bench already uses, so a new number is
# comparable with the recorded ones.
# Source: the mh-vs-structure-A bench, measured 2026-08 (script retired
# with the in-package test tree).
MH_CONTRACT_SHAPES = (
    dict(name="B8 H64 P32 N128 L2048", B=8, H=64, P=32, N=128, L=2048),
    dict(name="B8 H16 P32 N64 L2048", B=8, H=16, P=32, N=64, L=2048),
    dict(name="B4 H32 P32 N128 L1024", B=4, H=32, P=32, N=128, L=1024),
    dict(name="B2 H32 P64 N64 L4096", B=2, H=32, P=64, N=64, L=4096),
)

# The "paper" bench's one-shape subset of `mh_features`:
# "one contract shape at P=1 and one at P>=4".  `mh_features`
# already builds BOTH the shape's native (H, P) row and its P=1 twin
# (D = H*P held fixed) for every one of the four MH_CONTRACT_SHAPES, so
# "one shape at P=1 and one at P>=4" is read as one shape's worth of BOTH
# twins -- the first entry above (native P=32, i.e. P>=4).
MH_CONTRACT_SHAPES_PAPER = (MH_CONTRACT_SHAPES[0],)

# The SISO row: H = D, P = 1, the corner where mh_scan's head axis
# collapses onto the channel axis and the two kernels are the same
# function of the same tensors.  This row has NEVER been run — every
# other mh number in the repo is a ratio against the retired structure A.
# Source: the mh SISO bench, measured 2026-08 (script retired).
MH_SISO_SHAPES = (
    dict(B=8, D=2048, N=64, L=2048),    # the contract's spine
    dict(B=8, D=2048, N=16, L=2048),    # Mamba-1's native N
    dict(B=4, D=1024, N=16, L=2048),
    dict(B=2, D=2048, N=64, L=4096),
)

# The paper's per-feature cost ledger is still to be filled.
# Each variant is priced against the "real" row at the same shape.
# Source: the mh feature bench (gates / biases at both group extremes /
# delta_w at both resolutions) and the decay-law bench (the four laws),
# measured 2026-08 (scripts retired).
MH_FEATURES = (
    ("real", dict()),
    ("gates", dict(gates=True)),
    ("bias_g1", dict(biases=True, G=1)),
    ("bias_gh", dict(biases=True, G="H")),
    ("delta_w_head", dict(delta_w="head")),
    ("delta_w_chan", dict(delta_w="chan")),
    ("rot", dict(rot=True)),
    ("gates_bias", dict(gates=True, biases=True, G=1)),
    ("all", dict(gates=True, biases=True, G=1, delta_w="chan")),
    ("all_rot", dict(gates=True, biases=True, G=1, delta_w="chan", rot=True)),
)

# Raw mh_scan decay laws, priced against A_FIX.  ``m3`` is the module-level
# HybridMamba name whose flat rows dispatch to SSD; this raw MH ABI calls the
# corresponding time-constant law ``const``.
MH_LAWS = ("A", "dyn", "mu")

# ═════════════════════════════════════════════════════════════════════
# Campaign / module shapes — the contract's rows
# ═════════════════════════════════════════════════════════════════════

# "Timing shapes: MQMKVAR (B32, L2048, d256-class, N=64 mhr)
#  and LM spine (B8, D2048, L2048, N=64)."
MQMKVAR_SHAPE = dict(B=32, L=2048, d_model=256, N_mhr=64)
LM_SPINE = dict(B=8, D=2048, L=2048, N=64)

# State-size split: the SAME N=64-only defect applies to the exact-
# config MIXER rows (`SM-R1:m2exact`/`SM-R1:m3exact` + their
# `official_mamba2`/`official_mamba3` counterparts, both grids) --
# `presets.mamba2_config`/`mamba3_config` used to default `d_state=64`
# (fixed by this same pass, `src/orbit_ssm/presets.py`), and
# `ssm_mqmkvar`/`ssm_lm_spine` always called those exact-config cells at
# `N_mhr`/`LM_SPINE["N"]` (64), never at the officials' own default.
# `EXACT_D_STATE_M23` (128) is the new HEADLINE state size for the
# `m2exact`/`m3exact` rows and their officials; `EXACT_D_STATE_M23_N64`
# (64) keeps the ORIGINAL cell as a labelled secondary row (arm name
# suffixed `@N64`, `benchmarks/bench_ssms.py`).  `HMM:m1exact`/
# `official_mamba1` do NOT move: Mamba-1's own default is d_state=16
# (`mamba_ssm.modules.mamba_simple.Mamba`'s own constructor default,
# and `presets.mamba1_config`'s own `d_state=16` default already) --
# the pre-B9 code ran them at N_mhr/64 instead, which this pass also
# fixes (see `bench_ssms._exact_arm_param_audit`'s `d_state_m1` param).
EXACT_D_STATE_M23 = 128
EXACT_D_STATE_M23_N64 = 64
EXACT_D_STATE_M1 = 16

# The 440M FineWeb-Edu spine bench_models prices, per the standing
# param-match rule (at most the baseline's params, within 1% below).
# mamba3 at 439.4M is the smallest strong baseline.
LM_440M = dict(vocab_size=50288, d_model=1024, n_layers=24, B=8, L=2048,
               d_state=64)

# The MQMKVAR S-arm widths, verbatim from the spec, so bench_ssms
# prices the arms that actually run.  G is passed EXPLICITLY (never the
# default): a G below the fused kernels' 16-lane tile is zero-padded
# host-side at a cost of 16/G, and relying on the default is what made
# a pilot's R>1 arms crawl.
#
# `width` IS d_model, NOT d_inner.  The spec's widths (S1 d264, S2/S3/S5
# d200, S4 d152 — the drive-wiring column) are the LM's d_model and the
# mixer sits behind the standard expand=2
# (`integration._lm_from_mixer_keys`: `expand = 2.0` for SSD/HM), so the module the campaign builds is d_inner = 2*width.  Feeding
# the width straight in as d_inner is what made a benchmark run skip
# all 30 S cells with `d_inner=264 not divisible by P=G*R=16` (264, 200
# and 152 are not multiples of 16; 528, 400 and 304 all are) and left
# the contract with no row for our own family.  Checked against the
# spec's own parameter ledger: built as `d_model=width, n_layers=2,
# expand=2, proj_source="main_branch"`, all five arms reproduce the spec's
# non-embedding counts EXACTLY (S1 1,022,464 / S2 1,017,256 / S3 988,456
# / S4 1,041,000 / S5 1,037,106 — delta 0 on every arm).  The residual
# wiring gives 956,094 / 772,066 / 756,066 / 684,398 / 772,066 at the
# same widths, so the table's widths are the DRIVE column, which is what
# `bench_ssms._config` builds (`proj_source="main_branch"`).
#
# `row_blocks` is the arm's state layout: S2/S4/S5 rotate the whole state (the
# pre-spec rot_mode="m3" at its default rope_fraction=1.0), S1/S3 are real;
# S5 is the only data-dependent-rate arm (pre-spec a_dyn=True).
MQMKVAR_S_ARMS = (
    dict(id="S1", width=264, d_inner=528, d_ssm_in=1, d_ssm_out=1, n_groups=16,
         row_blocks=ssd_rows(False, "head_static")),
    dict(id="S2", width=200, d_inner=400, d_ssm_in=2, d_ssm_out=2, n_groups=8,
         row_blocks=ssd_rows(True, "head_static")),
    dict(id="S3", width=200, d_inner=400, d_ssm_in=2, d_ssm_out=2, n_groups=8,
         row_blocks=ssd_rows(False, "head_static")),
    dict(id="S4", width=152, d_inner=304, d_ssm_in=4, d_ssm_out=4, n_groups=4,
         row_blocks=ssd_rows(True, "head_static")),
    dict(id="S5", width=200, d_inner=400, d_ssm_in=2, d_ssm_out=2, n_groups=8,
         row_blocks=ssd_rows(True, "head_dynamic")),
)
# The other MQMKVAR mixer rows, on the SAME width convention as the
# officials.  `ssm_mqmkvar`'s comparison group is keyed `B32L2048d256` and
# the officials in it are built from d_model=256 through expand=2, i.e.
# d_inner=512; HM was built with d_inner=256, so the group
# whose whole purpose is a like-for-like ratio held its arms at HALF the
# officials' inner width.  Holding them at half is a bug, not a
# choice — every arm in the group builds at the officials' convention
# unless the spec gives that arm its own matched width.
#
# It does not, for these:
#   * §H DOES list widths (H1/H3/H5/H7 d264, H2 d256, H4 d248) but those
#     are the RESIDUAL-wiring numbers, and `bench_ssms._config` builds
#     every family with proj_source="main_branch".  Measured against the
#     insane-256 anchor (1,023,744 non-emb, n_layers=2, head_dim=16):
#     H1 at d264 residual = 1,006,188 (-1.71%, the maximal width inside
#     the spec's 2% band, which is how d264 was derived), but the SAME
#     width under drive = 1,089,916 (+6.46%) and breaks the param recipe.
#     Under drive the compliant width is d256: 1,031,296 (+0.74%) with
#     H1's norms, 1,022,848 (-0.09%) with the bench row's norm-free
#     defaults — the closest any width/wiring gets to the anchor.
# So these arms take d_model=256 / d_inner=512 and SAY SO in the record
# (`width_convention="officials"` in tags), rather than silently claiming
# a spec width they do not have.
MQMKVAR_OTHER_ARMS = (
    # head_dim 16 => n_heads = d_inner//16 = 32, so P = 16, a power of
    # two — the MH CUDA backward refuses anything else
    # (`HybridMambaSSM`).
    dict(id="HM", family="hybrid_mamba", d_model=256, d_inner=512,
         d_head=16, width_convention="officials"),
)


def _check_mqmkvar_arms() -> None:
    """Width invariants for the `ssm_mqmkvar` group, at import.

    S arms: P = G*R = 16 (the spec's head-dim-16 matching rule), so every
    d_inner must be a multiple of 16.  Other arms: the officials'
    convention, d_inner = 2 * d_model = 2 * MQMKVAR_SHAPE["d_model"].
    Checked here rather than discovered as 30 skipped cells on an
    expensive cluster run — `raise`, not `assert`, so `python -O`
    cannot drop it."""
    for arm in MQMKVAR_S_ARMS:
        if arm["d_inner"] != 2 * arm["width"]:
            raise ValueError(f"{arm['id']}: d_inner must be 2*width "
                             f"(expand=2), got {arm}")
        P = arm["n_groups"] * arm["d_ssm_in"]
        if P != 16:
            raise ValueError(f"{arm['id']}: G*R = {P}, the spec's "
                             f"head-dim-16 rule says 16")
        if arm["d_inner"] % P:
            raise ValueError(
                f"{arm['id']}: d_inner {arm['d_inner']} is not a multiple "
                f"of P=G*R={P} — the SSD core refuses it")
    official = 2 * MQMKVAR_SHAPE["d_model"]
    for arm in MQMKVAR_OTHER_ARMS:
        if arm["d_inner"] != 2 * arm["d_model"]:
            raise ValueError(f"{arm['id']}: d_inner must be 2*d_model "
                             f"(expand=2), got {arm}")
        if arm["d_inner"] != official:
            raise ValueError(
                f"{arm['id']}: d_inner {arm['d_inner']} is not the "
                f"officials' {official} — the ssm_mqmkvar group compares "
                f"these arms against official_mamba1/2/3 built from "
                f"d_model={MQMKVAR_SHAPE['d_model']} through expand=2")
        hd = arm.get("d_head")
        if hd is not None and (hd & (hd - 1)):
            raise ValueError(f"{arm['id']}: head width {hd} is not a power "
                             f"of two — the MH CUDA backward refuses it")


_check_mqmkvar_arms()

# ═════════════════════════════════════════════════════════════════════
# The smoke grid — the gate
# ═════════════════════════════════════════════════════════════════════
# Small enough that `--grid smoke` covers BOTH kernel families and EVERY
# available baseline on the shared login GH200 in under five minutes,
# including first-call compilation (mh bwd ~6 s, SSD fused bwd ~25 s,
# tilelang M3-MIMO ~30 s).  It is a plumbing check, not a measurement:
# never quote a smoke number.
SMOKE_MH = dict(B=2, D=256, N=16, L=512)
SMOKE_MH_MIMO = dict(B=2, H=4, P=8, N=16, L=512)
SMOKE_SSD = dict(B=2, H=4, G=64, R=1, O=1, N=64, L=512, T=64)
SMOKE_SSD_MIMO = dict(B=2, H=4, G=16, R=2, O=2, N=64, L=512, T=64)
SMOKE_SSM = dict(B=2, L=256, d_model=128, d_state=32)
SMOKE_MODEL = dict(vocab_size=1024, d_model=128, n_layers=2, B=2, L=256,
                   d_state=32)


@dataclass(frozen=True)
class GridInfo:
    """What a named grid is and where its shape came from."""

    name: str
    module: str
    source: str


# The ONE fixed shape set shared by every calibration and the main benchmark
# files: size classes x axes + the LM cells.
from orbit_ssm.kernels import shape_sets as _shape_sets  # noqa: E402
SHARED_GRID, SHARED_LM, SHARED_SET = (_shape_sets.GRID, _shape_sets.LM,
                                      _shape_sets.SET)

GRID_SOURCES = (
    GridInfo("kernel_shared_set", "bench_kernels",
             "src/orbit_ssm/kernels/shape_sets.py: kernel-level "
             "ours vs Mamba-2 chunk scan / Mamba-3 SISO (a4_trapon) on the one shared set"),
    GridInfo("ssm_shared_set", "bench_ssms",
             "src/orbit_ssm/kernels/shape_sets.py: the "
             "one set every calibration and benchmark uses"),
    GridInfo("ssm_shared_small", "bench_ssms",
             "src/orbit_ssm/kernels/shape_sets.py: the small cells of ssm_shared_set (a shard-planner unit; its cells carry grid = ssm_shared_set)"),
    GridInfo("ssm_shared_medium", "bench_ssms",
             "src/orbit_ssm/kernels/shape_sets.py: the medium cells of ssm_shared_set (a shard-planner unit; its cells carry grid = ssm_shared_set)"),
    GridInfo("ssm_shared_large", "bench_ssms",
             "src/orbit_ssm/kernels/shape_sets.py: the large cells of ssm_shared_set (a shard-planner unit; its cells carry grid = ssm_shared_set)"),
    GridInfo("ssm_shared_lm", "bench_ssms",
             "src/orbit_ssm/kernels/shape_sets.py: the lm cells of ssm_shared_set (a shard-planner unit; its cells carry grid = ssm_shared_set)"),
    GridInfo("smoke", "bench_kernels", "this file (plumbing gate only)"),
    GridInfo("mh_siso", "bench_kernels",
             "the retired mh SISO bench (never run)"),
    GridInfo("mh_features", "bench_kernels",
             "the retired mh feature + decay-law benches at the "
             "mh-vs-A shapes"),
    GridInfo("paper_c3", "bench_kernels",
             "paper Appendix C.3, Table 10"),
    GridInfo("paper_c3_nscale", "bench_kernels",
             "paper Appendix C.3, Table 11"),
    GridInfo("paper_c3_lscale", "bench_kernels",
             "paper Appendix C.3, Figure 4 — caption vs "
             "generator shape discrepancy, see PAPER_C3_LSWEEP"),
    GridInfo("ssd_parity26", "bench_kernels",
             "the 26-cell Mamba-2 parity sweep"),
    GridInfo("ssd_long_l", "bench_kernels",
             "the long-L envelope"),
    GridInfo("ssd_mimo_gates", "bench_kernels",
             "the contract envelope and the R4O4 gate"),
    GridInfo("ssd_features", "bench_kernels",
             "the contract feature list"),
    GridInfo("ssm_mqmkvar", "bench_ssms",
             "the MQMKVAR S-arm widths and the shared timing shapes"),
    GridInfo("ssm_lm_spine", "bench_ssms",
             "the shared timing shapes (LM spine)"),
    GridInfo("model_440m", "bench_models",
             "the FineWeb-Edu 440M spine + the param-match standing rule"),
)
