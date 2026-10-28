"""Fused SSD-MIMO backward: 5-kernel decomposition.

KERNEL CONTRACT:
    The BWD pass uses at most 7 kernel launches (currently 3-5 Triton
    kernels + host-side Python for da_log and dGamma post-processing).

    Execution order:
      Ka   — v recurrence (sequential over chunks, reverse) → dG, dalpha
      Kb2  — aggregate-path (chunk-parallel)    → partial dB, partial dDelta
             ABSORBED INTO Kc (FUSE_KB2) when O == 1 and HAS_DW — Kc
             already holds B/dG/x and the BdG GEMM.  A/B lever:
             SSD_MIMO_FUSE_KB2=0 restores the separate launch.
      Kb1  — score-path (chunk-parallel)        → dC_rd, partial dB, partial dDelta
             REPLACED by Kbc-hb (owner-accumulate, head-batched) on the
             O = R = 1 ungrouped fused path, which also absorbs Kc's
             dB_agg AND the whole rotation
             mode: theta_h / rot_bias fold into the GEMM operands, so
             dB/dC accumulate UNROTATED and Kinv is not launched at all.
             A/B lever: SSD_MIMO_BWD_HB=0 restores Kb1 (+ Kinv).
      Kinv — inverse rotation + dThetaH         → dC (from dC_rd)
             GENERIC BRANCH ONLY (O > 1, R > 1, grouped B/C, or
             SSD_MIMO_BWD_HB=0).
      Kc   — dx (+ Kb2 work under FUSE_KB2)     → dX, dDW
      NOTE (dalpha/R_TILE): the dalpha dDelta term is per-(b, h, c, gt);
      in R-tiled grids only the r==0 CTA adds it.

    Ka carries v[N, GT] in registers (reverse chunk loop).
    Kb1/Kb2/Kc are chunk-parallel and use atomic_add for dB, dDelta, dDW.
    Kinv runs only when HAS_THETA_H **and the generic branch is taken**.
    Launch ledger at R = O = 1 in rotation mode: fwd-replay, Ka, Kbc-hb,
    Kc(NO_DBAGG) — 4 Triton launches.
    AUTOTUNE: every autotuned kernel with atomic-accumulating outputs
    declares them in restore_value — without it the first call per
    config-key returns gradients inflated by the bench replays.
    HAS_DW (production default): kernels receive RAW x + dw [B, H, L],
    scale x on load; Kc stores dx_raw and accumulates dDW in-kernel.
    dY is consumed STRIDED (native [B, H, (O,) G, L] layout or expanded
    ones) — no host .contiguous() materialisation.

Rotation modes (all angles are ADDITIVE):
    theta_h   [H, N//2]          Static per-head offset on C (readout).
                                  Kinv applies inverse R(-theta_h) to get
                                  dC from dC_rd.  dThetaH accumulated via
                                  atomic_add across chunks.
    rot_bias  [H, R, N//2]       Static per-(head,R) offset on B (write).
              (passed as [Hk, N//2])  ADDITIVE angle.  Kb1 and Kb2 apply it
                                  when computing dB gradients.  dRotBias
                                  accumulated atomically.
    theta     [B, L, N//2]       Per-timestep rotation (head-shared) —
                                  Consumed by the differentiable host FOLD in
                                  autograd.py (R(theta)^T applied to B/C
                                  before chunking); dTheta comes from the
                                  fold's autograd, so the kernels see
                                  already-rotated B~/C~ on that path.

Biases (both ADDITIVE, init zeros):
    beta      [H, N, R]          B_bias.  dBeta accumulated in Kb1 + Kb2,
              (passed as [Hk, N, R])  atomic_add across chunks.
    gamma     [H, N, O]          C_bias.  dGamma from Kb1 scratch buffer,
              (passed as [Hk, N, O])  post-processed HOST-SIDE with inverse
                                  theta_h rotation when HAS_THETA_H.

Output gradients:
    dX           [B, H, G_SIZE, R, L]  -- from Kc (tl.store, not atomic)
    dCtil        [B, nc, T, N]         -- from Kb1 + Kinv, or Kbc-hb
    dBtil        [B, nc, T, N]         -- from Kb1 + Kb2 (atomic_add),
                                          or one owner write from Kbc-hb
    dDelta       [B, H, nc, T]         -- from Kb1 + Kb2 (atomic_add)
    dThetaH      [H, N//2]            -- from Kinv (atomic_add) + host dGamma,
                                          or from Kbc-hb (atomic_add) alone
    dBeta        [H, N, R]            -- from Kb1 + Kb2 (atomic_add)
    dGamma       [H, N, O]            -- host-side post-processing
    dRotBias     [H, N//2]            -- from Kb1 + Kb2 (atomic_add)
    da_log is computed HOST-SIDE as (delta_cumsum * dDelta).sum(0,2,3).
"""

import os

import torch
import triton
import triton.language as tl
from .autotune_policy import autotune_tier
from .common import pdot  # FP32DOT operand rounding
from ..fp32_class import ieee_matmul

from triton.runtime.errors import OutOfResources

from .common import nan_debug_fill
from .. import portable as _portable
from .common import reraise_fatal
from .common import ACC_SEM, DTH_OPROUND, _cos_approx, _sin_approx, _opv
from .common import _span_sum, _excl_prefix   # SPANDELTA
from .fused_fwd_kernel import occ_bucket as _occ_bucket
from .fused_fwd_kernel import (_env, _pair_swap_2d, _pair_swap_1d,
                               _pair_bcast_2d, _dcs_tile, _dcs_end,
                               _dcs_seg, _same_seg,
                               autotune_off, dot_precision, mma_operand_dtype,
                               SM90_PTXAS_FIXED, sm90_ptxas_version)

# ── _SATBWD_NOTE: dDelta without O(1) cancellation ─────────────────────
# dL/ddelta is formed directly from the pairs that span each token
# (`_SPANDELTA_NOTE`).  Reductions over dL/dD_i (D_i the chunk-local decay
# prefix) would let the diagonal M_ii (decay exactly 1) and Kb2's chunk-end
# q_{T-1} sit in both sums and cancel only in exact arithmetic: u * O(1)
# where the truth is O(alpha) once the decay saturates.  The score diagonal
# rides the MMAs (strict only under the trapezoid, `_TRAPFOLD_NOTE` in
# fused_fwd_kernel.py).


# ── _SPANDELTA_NOTE: dL/ddelta in SPANNING-PAIR form ────────────────────
# The chunk-local decays are exp(a (D_i - D_j)) with D the in-chunk prefix
# of delta, so d/ddelta_s touches exactly the pairs that SPAN s (j < s <= i):
#   dL/ddelta_s = a [ sum_{j < s <= i} M_ij          (score, M_ij = (dY_i.x_j)
#                                                      w_j (C~_i.B~_j) e^{a(D_i-D_j)})
#                   + sum_{j < s} qd_j                (write: qd_j = dec_agg_j w_j
#                                                      x_j.(B~_j dG), pair (j, end))
#                   + sum_{i >= s} qf_i               (read: qf_i = f_i dY_i.(C~_i S_in))
#                   + alpha dalpha ]                  (carry, every s)
# Every term carries the decay ACROSS s as a factor, so nothing cancels: the
# alternative (return dL/dD_i, host takes sum_{i >= s}) lets pairs with both
# ends >= s cancel only in exact arithmetic, leaving u * O(1) where the truth
# behind a reset token (c = |A| delta >> 1) is O(e^-c).  The kernels return
# dL/ddelta directly (no host reverse cumsum): Kc owns
# the score term (it holds Score = C~B~^T o decay; one [T,GT]x[GT,T] MMA
# gives dY x^T), the write term rides Kc's tile as its last row (pair
# (j, T-1) spans s iff j < s <= T-1) or `_excl_prefix` where Kb2 / the
# Kb1 fold own it, the read term is a reverse cumsum in Kb1 (each qf_i
# carries f_i), and the carry term is a broadcast.


# ── THE ONE AUTOTUNE SWITCH, backward half ───────────────────────────
# Read ONCE, here, because every grid below is a `@triton.autotune`
# decorator argument and therefore import-time.  Under "off" each grid
# collapses to a SINGLE candidate, which is what makes triton skip
# benchmarking entirely (it only benches when len(configs) > 1).
# Each collapsed value is the file's own documented pick, or a stated
# heuristic where no measurement exists; the per-axis levers
# (SSD_MIMO_BWD_WARPS / BWD_STAGES / BWD_MAXNREG / BWD_HB_CANDS) still
# override, because they are read with these as their DEFAULTS.

# ── BWD autotuning config space ──────────────────────────────────────────
# Same candidate grid as FWD (warps × stages).  Each of the 5 kernels
# (Ka, Kb1, Kb2, Kinv, Kc) tunes independently — they have very different
# register pressure profiles.
# ── AUTOTUNE TIERS ─────────────────────────────────────────────────────
# `SSD_MIMO_AUTOTUNE` selects how wide the backward's config search is:
#
#   "pinned"  1 config   -- the best measured config on every GPU tested
#   "light"   4 configs  -- DEFAULT: tune num_warps only
#   "full"   16 configs  -- also tune stages and maxnreg
#
# The env var must be set BEFORE import (setting it afterwards silently
# measures the same config).  Axes:
#   num_warps  matters most (1 -> 8 is ~22-24 %) and A100 / Blackwell
#              disagree on the winner, so it cannot be frozen.  1 is kept
#              for small consumer GPUs, where a smaller CTA may be the only
#              shape that FITS (OutOfResources).
#   stages     spread at or below the ~5 % run-to-run noise; st=1 best on
#              both.  Frozen, tuned only in "full".
#   maxnreg    0 (= do not pass the flag, compiler's own choice) wins on
#              both; 256 is the only other value that ever placed, 128 and
#              168 lose and are dropped from every tier.
_AUTOTUNE_TIER = autotune_tier(_env("AUTOTUNE", "light"))

# Part B's flag and the tier ladder must agree: "pinned" means pinned
# everywhere, including the HB kernels this flag gates (_HB_CFGS,
# BWD_HB_CANDS), which the tier lists do not reach.
_AUTOTUNE_OFF = autotune_off() or _AUTOTUNE_TIER == "pinned"

_BWD_PINNED = (8, 1, 0)                      # (warps, stages, maxnreg)

_BWD_WARPS = tuple(int(v) for v in
                   _env("BWD_WARPS",
                        {"pinned": "8", "light": "1,2,4,8",
                         "full": "1,2,4,8"}[_AUTOTUNE_TIER]).split(","))
_BWD_STAGES = tuple(int(v) for v in
                    _env("BWD_STAGES",
                         {"pinned": "1", "light": "1",
                          "full": "1,3"}[_AUTOTUNE_TIER]).split(","))
# maxnreg axis: "0" means do not pass maxnreg (the compiler's own choice);
# SSD_MIMO_BWD_MAXNREG widens or narrows the list.  The axis matters for
# register-bound kernels (_bwd_kbc_hb_kernel at N = 128).  "off" -> 0.
_BWD_MAXNREG = tuple(int(v) for v in
                     _env("BWD_MAXNREG",
                          {"pinned": "0", "light": "0",
                           "full": "0,256"}[_AUTOTUNE_TIER]).split(","))


#: CHUNKPAR route trace (which Ka schedule served the last call)
_CP_LAST_ROUTE: dict = {}


def _bwd_cfg(w, st, mnr):
    kw = dict(num_warps=w, num_stages=st)
    if mnr:
        kw["maxnreg"] = mnr
    return triton.Config({}, **kw)


_BWD_CFGS = [_bwd_cfg(w, st, m)
             for w in _BWD_WARPS for st in _BWD_STAGES
             for m in _BWD_MAXNREG]


# ── LAUNCH CALIBRATION (src/orbit_ssm/kernels/calibration.py) ─────────────
# The tier grids as BUILDERS, so the calibration framework can ask for a
# tier's list without a second import: `_bwd_grid(_AUTOTUNE_TIER)` is
# `_BWD_CFGS` exactly.  Per kernel, `_calib_grid` returns the very same
# list unless `SSD_MIMO_CALIBRATION=on`, in which case `pinned` = that
# kernel's ONE calibrated fixed config and `light` = its 2-4 calibrated
# configs (no valid calibration for this GPU -> the tier list as today).
_TIER_AXES = {"pinned": ("8", "1", "0"), "light": ("1,2,4,8", "1", "0"),
              "full": ("1,2,4,8", "1,3", "0,256")}
_CALIB_TIER = "pinned" if _AUTOTUNE_OFF else _AUTOTUNE_TIER


def _bwd_grid(tier):
    w, st, m = _TIER_AXES[tier]
    return [_bwd_cfg(a, b, c)
            for a in (int(v) for v in _env("BWD_WARPS", w).split(","))
            for b in (int(v) for v in _env("BWD_STAGES", st).split(","))
            for c in (int(v) for v in _env("BWD_MAXNREG", m).split(","))]


def _cprune(kernel):
    """early_config_prune of a calibrated kernel: narrow to the active
    shape group's configs (no-op unless calibration is on)."""
    def prune(configs, named_args, **kwargs):
        from .calibration_fused import prune as _p
        return _p(configs, kernel)
    return prune


def _calib_grid(kernel, builder):
    from .calibration_fused import grid
    return grid(kernel, _CALIB_TIER, full=lambda: builder("full"),
                light=lambda: builder("light"),
                generic_pin=lambda: builder("pinned"))


# No `early_config_prune` on num_warps: the 8-warp MIMO fault was a real
# construct in Kc (see the DOT-SEEDED ACCUMULATOR note in `_bwd_k2_kernel`),
# fixed in the kernel, so the candidate set below is unpruned.

# ── CAST-BEFORE-TRANSPOSE (8 warps under IO-dtype operands) ─────────────
# A tile used as a tl.dot operand in BOTH orientations must be cast to MMA_DT
# ONCE, before the transpose:  X_dt = X.to(MMA_DT); pdot(X_dt, ...);
# pdot(.., tl.trans(X_dt), ..).  Casting each use separately makes Triton keep
# an fp32 `ttg.local_alloc` staging copy (truncated inside the dot_op layout)
# next to a register-transposed bf16 one; at warpsPerCTA = [8, 1] over a
# 64-row T tile that pair mis-lowers (wrong dx_r, or reads past the shared
# arena).  Casting once stages the tile as nvmma_shared<elementBitWidth = 16>
# and the transpose is a `ttg.memdesc_trans` flip; 4 and 8 warps then agree
# bit-for-bit.  Under fp32 operands the cast is a no-op.  Grep
# "K24 CAST-BEFORE-TRANSPOSE" for the sites.

# ── sm_90 wgmma faults at chunk T = 64 under ptxas < 12.9.86 ─────────────
# A ptxas older than SM90_PTXAS_FIXED (12.9.86) miscompiles wgmma descriptors
# (the root cause shared with `ro_stack_wgmma_fault` in fused_fwd_kernel.py;
# Triton 3.8's bundled ptxas is fixed).  Three backward cells are affected at
# T = 64; the dispatcher caps the chunk at 32 for them and records
# `chunk_guard`:
#   rot_io_bwd_t64_fault    HAS_THETA_H, IO-dtype MMA, R = O = 1: the O = 1
#                           per-r `S0_r` dot of `_bwd_k2_kernel` takes an
#                           illegal shared read ("k24_rot_io_bwd_t64").
#   pairhalf_r1_t64_fault   rotate-half layout, R = O = 1, tensor-core MMA
#                           ("sm90_pairhalf_r1_t64").
#   pairhalf_io_t64_fault   rotate-half layout, 16-bit MMA operands, any R, O:
#                           wrong y / non-finite / illegal access
#                           ("sm90_pairhalf_io_t64").
# The fault does not depend on num_warps / num_stages / maxnreg, so the axis
# that moves is the chunk, not the launch config.  Which (R, O) break moves
# with register allocation, so no narrower predicate holds.  "ieee" never
# reaches a wgmma and is unaffected.


def rot_io_bwd_t64_fault(capability, mma_io: bool, has_theta_h: bool,
                         R: int, O: int, ptxas_version="resolve") -> bool:
    """True on the (sm_90, HAS_THETA_H, IO-dtype MMA, R = O = 1) cell where
    `_bwd_k2_kernel`'s O = 1 per-r `S0_r` dot takes an illegal shared-memory
    read at chunk T = 64 under a ptxas OLDER than `SM90_PTXAS_FIXED`.
    `ptxas_version` as in `ro_stack_wgmma_fault` ("resolve" = the one Triton
    will use; None = unknown, judged affected).  Callers (`_ssd_dispatch`)
    cap the chunk at 32 and log it.  See the sm_90 wgmma note above.
    """
    if not (tuple(capability) == (9, 0) and bool(has_theta_h)
            and bool(mma_io) and R == 1 and O == 1):
        return False
    if isinstance(ptxas_version, str) and ptxas_version == "resolve":
        ptxas_version = sm90_ptxas_version()
    return ptxas_version is None or tuple(ptxas_version) < SM90_PTXAS_FIXED




def pairhalf_io_t64_fault(capability, mma_io: bool, pair_half: bool,
                          ptxas_version="resolve") -> bool:
    """True on (sm_90, pair_layout="half", 16-bit MMA operands) at an
    affected ptxas (< `SM90_PTXAS_FIXED`), any R, O: chunk T = 64 returns
    a wrong `y` / non-finite values / an illegal access there.  `mma_io`
    is `mma_io_enabled(precision)`; `ptxas_version` as in
    `ro_stack_wgmma_fault`.  Callers cap the chunk at 32.  See the
    sm_90 wgmma note above."""
    if (int(tuple(capability)[0]) != 9 or not bool(mma_io)
            or not bool(pair_half)):
        return False
    if isinstance(ptxas_version, str) and ptxas_version == "resolve":
        ptxas_version = sm90_ptxas_version()
    return ptxas_version is None or tuple(ptxas_version) < SM90_PTXAS_FIXED


def pairhalf_r1_t64_fault(capability, tensor_cores: bool, pair_half: bool,
                          R: int, O: int, ptxas_version="resolve") -> bool:
    """True on the (sm_90, pair_layout="half", R = O = 1, tensor-core MMA)
    cell at an affected ptxas (< `SM90_PTXAS_FIXED`), where chunk T = 64
    takes an illegal access.  `tensor_cores` is `precision != "ieee"`;
    `ptxas_version` as in `ro_stack_wgmma_fault` ("resolve" = the one
    Triton will use; None = unknown, judged affected).  Callers cap the
    chunk at 32.  See the sm_90 wgmma note above."""
    if (int(tuple(capability)[0]) != 9 or not bool(tensor_cores)
            or not bool(pair_half) or R != 1 or O != 1):
        return False
    if isinstance(ptxas_version, str) and ptxas_version == "resolve":
        ptxas_version = sm90_ptxas_version()
    return ptxas_version is None or tuple(ptxas_version) < SM90_PTXAS_FIXED


@triton.jit
def _ph_read_consistent(Ch1, Ch2, ch, sh, Ct1, Ct2, MMA_DT: tl.constexpr,
                        MMA_IO: tl.constexpr, HAS_THETA_H: tl.constexpr,
                        PREC: tl.constexpr = "ieee"):
    """THETAFREQ, rotate-half layout: the rotated read tile C~ whose
    R(nu) image the MMAs consumed.  Under MMA_IO with theta_h that is
    R(nu)^T round(C_h) (C_h = R(nu) C~: Ch1 = c Ct1 + s Ct2,
    Ch2 = c Ct2 - s Ct1); without theta_h, round(C~).  Returns (Ct1, Ct2)."""
    if (MMA_IO or PREC == "tf32") and DTH_OPROUND:
        if HAS_THETA_H:
            R1 = _opv(Ch1, MMA_DT, MMA_IO, PREC)
            R2 = _opv(Ch2, MMA_DT, MMA_IO, PREC)
            Ct1 = ch[None, :] * R1 - sh[None, :] * R2
            Ct2 = ch[None, :] * R2 + sh[None, :] * R1
        else:
            Ct1 = _opv(Ct1, MMA_DT, MMA_IO, PREC)
            Ct2 = _opv(Ct2, MMA_DT, MMA_IO, PREC)
    return Ct1, Ct2


def ka_wave_bucket(n_ctas: int, device) -> int:
    """The serial-Ka AUTOTUNE KEY's occupancy bucket: floor(log2(CTAs per
    SM)) + 4, clamped at 0.  Ka's best launch config moves with the grid, so
    the key must carry it."""
    global _KA_LAST_WB
    _KA_LAST_WB = _occ_bucket(n_ctas, device)
    return _KA_LAST_WB


_KA_LAST_WB = None     # the bucket of the last Ka launch (cp_calibrate reads it)


def _ka_prune_inner(configs, named_args, **kwargs):
    """Serial Ka's launch config from the per-device MEASURED verdict for
    this occupancy bucket (tools/cp_calibrate.py ->
    chunk:cp:<dev>|ka_serial|warps|wb<k>), or `SSD_MIMO_KA_WARPS`; the
    full autotune space (keyed per bucket) when neither exists."""
    w = _env("KA_WARPS", "")
    if not w:
        # A chunk length demoted after a shared-memory overflow (`calibration.smem_demote`):
        # the calibrated warps pin is a TIMING verdict and may leave only configs that do
        # not fit, so the generic grid is searched unpinned.
        from .. import calibration as _cal
        if str(_cal.active_group("ssd_mimo_fused") or "").startswith("~smem"):
            return configs
        # The verdict was measured with 16-bit MMA operands; an fp32-operand call
        # (MMA_IO = 0) keeps its own calibrated group's configs (`*_fp32_t<T>`) -- a
        # 16-bit warps pin at fp32 can land on a register-collapsed body.
        if not int(named_args.get("MMA_IO", kwargs.get("MMA_IO", 1)) or 0):
            return configs
        wb = named_args.get("KA_WB", kwargs.get("KA_WB", None))
        try:
            from .chunkpar import cp_calibration_key
            from .smem import chunk_cache_lookup
            v = chunk_cache_lookup(cp_calibration_key(
                torch.cuda.current_device(), "ka_serial") + f"|warps|wb{wb}")
            w = str(v) if v else ""
        except Exception:                               # noqa: BLE001
            w = ""
    if w:
        keep = [c for c in configs if c.num_warps == int(w)]
        if keep:
            # the calibrated warps pins the grid-sensitive axis; the stage
            # count (software pipelining of the chunk walk) stays a
            # per-bucket autotune choice
            return keep
    return configs


def _ka_prune(configs, named_args, **kwargs):
    from .calibration_fused import prune as _p
    return _p(_ka_prune_inner(configs, named_args, **kwargs), "fused_ka")


# Serial Ka also autotunes the STAGE count: its chunk loop's loads (C, dY, S)
# do not depend on the carried v, so Triton can prefetch them.  The best count
# varies per occupancy bucket.  `SSD_MIMO_KA_STAGES`.
_KA_STAGES = tuple(int(v) for v in _env(
    "KA_STAGES", "1" if _AUTOTUNE_TIER == "pinned" else "1,2,3").split(","))
_KA_CFGS = [_bwd_cfg(w, st, m) for w in _BWD_WARPS for st in _KA_STAGES
            for m in _BWD_MAXNREG]


def _ka_grid(tier):
    """Serial Ka's tier grid as a builder (calibration framework): the
    warps / maxnreg of the tier and the stage axis `_KA_STAGES` (1 at
    pinned, 1,2,3 otherwise)."""
    w, _, m = _TIER_AXES[tier]
    st = _env("KA_STAGES", "1" if tier == "pinned" else "1,2,3")
    return [_bwd_cfg(a_, b_, c_)
            for a_ in (int(v) for v in _env("BWD_WARPS", w).split(","))
            for b_ in (int(v) for v in st.split(","))
            for c_ in (int(v) for v in _env("BWD_MAXNREG", m).split(","))]


@triton.autotune(configs=_calib_grid("fused_ka", _ka_grid),
                 key=["T", "N", "G_SIZE", "GT", "PREC", "MMA_IO",
                      "HAS_THETA_H", "HAS_GAMMA", "O", "IDX64",
                      "SEG_SUMMARY", "PAIR_HALF", "NO_DALPHA", "KA_WB"],
                 prune_configs_by={"early_config_prune": _ka_prune})
@triton.jit
def _bwd_ka_kernel(
    C_ptr, Delta_ptr, A_ptr, ThetaH_ptr, SAll_ptr,
    Gamma_ptr,
    dY_ptr, dSF_ptr,
    dG_ptr, dAlpha_ptr, dX0_ptr,
    H,
    s_cb, s_cc, s_ct, s_cn,
    s_db, s_dh, s_dc, s_dt,
    s_yb, s_yh, s_yc, s_yt, s_yg, s_yo,
    s_sb, s_sh, s_sc, s_sn, s_sg,
    s_fb, s_fh, s_fg, s_fn,
    s_gb, s_gh, s_gc, s_gn, s_gg,
    s_dab, s_dah, s_dag, s_dac,
    s_ch,
    NC,
    NC_SEG,          # chunks per segment (== NC when unsegmented)
    s_dsfs,          # dSF SEGMENT stride (phase C carry states; 0 = plain)
    s_x0ss,          # dX0 SEGMENT stride (phase A per-segment v store)
    T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr,
    GT: tl.constexpr, PREC: tl.constexpr,
    # `tl.dot` operand dtype: fp32 = identity cast; bf16 / fp16 = IO-dtype MMA
    # with fp32 accumulate.  See `mma_operand_dtype` in fused_fwd_kernel.py.
    MMA_DT: tl.constexpr = tl.float32,
    # Autotune KEY stand-in for MMA_DT (a tl.dtype is not serialisable into
    # Triton's disk cache key): 1 iff MMA_DT is 16-bit.
    MMA_IO: tl.constexpr = 0,
    HAS_THETA_H: tl.constexpr = False, HAS_DSF: tl.constexpr = False,
    HAS_GAMMA: tl.constexpr = False,
    HAS_X0: tl.constexpr = False,
    SEG_SUMMARY: tl.constexpr = False,
    s_x0b: tl.constexpr = 0, s_x0h: tl.constexpr = 0,
    s_x0g: tl.constexpr = 0, s_x0n: tl.constexpr = 0,
    HPG_C: tl.constexpr = 1,
    O: tl.constexpr = 1,
    s_co: tl.constexpr = 0,
    IDX64: tl.constexpr = False,
    # HAS_ROT: C is loaded RAW and R(theta_t)^T is applied on load from the
    # [B, nc, T, N//2] phase.  The xor-lane partner comes from the REGISTER
    # pair swap, never a second load.
    Theta_ptr=None,
    s_thb=0, s_thc=0, s_tht=0, s_thh=0, s_thk=0,
    HAS_ROT: tl.constexpr = False,
    BIAS_PRE_ROT: tl.constexpr = False,
    # PHASE_CUMSUM: `Theta_ptr` holds per-token angle INCREMENTS and
    # `ThOff_ptr` [B, H, nc, N//2] the phase entering each chunk, so any
    # chunk's phase is reachable in O(1).
    ThOff_ptr=None,
    s_ob=0, s_oh=0, s_oc=0, s_ok=0,
    PHASE_CUMSUM: tl.constexpr = False,
    # PHASE_DEFER: the kernel stores the RAW dL/dtheta_cum and one
    # chunk-parallel pass (`m3_epilogue.phase_prefix_bwd`) does the in-chunk
    # reverse cumsum, the chunk total and the cross-chunk term (an in-kernel
    # cross-thread scan per head is too costly in this register-starved body).
    PHASE_DEFER: tl.constexpr = False,
    # PAIRHALF: the rotate-half pair layout (see `fused_fwd_kernel.py`).
    PAIR_HALF: tl.constexpr = False,
    # `Delta_ptr` holds the RAW chunked delta; this kernel forms the
    # chunk-local prefix D_i itself (`_dcs_tile` / `_dcs_end`).
    DCS_INK: tl.constexpr = False,
    # NO_DALPHA: dalpha_c = sum(v_c * S_c) is a chunk-LOCAL reduction of tiles
    # Kbc already holds, so with KBC_DALPHA it moves there and this sequential
    # kernel stops loading S_all.
    NO_DALPHA: tl.constexpr = False,
    KA_WB=0,         # autotune-key occupancy bucket only (`ka_wave_bucket`)
):
    """Ka: v recurrence only. Stores dG_all (= v_old per chunk) and dalpha."""
    tl.static_assert(not NO_DALPHA or (not PAIR_HALF and not SEG_SUMMARY),
                     "Ka NO_DALPHA: main body, unsegmented only")
    tl.static_assert(not BIAS_PRE_ROT or HAS_ROT,
                     "Ka: bias_pre_rot needs the in-kernel phase")
    b = tl.program_id(0)
    h = tl.program_id(1)
    NGT_C = G_SIZE // GT
    seg = tl.program_id(2) // NGT_C
    gt_idx = tl.program_id(2) % NGT_C
    c_lo = seg * NC_SEG
    c_hi = tl.minimum(NC, c_lo + NC_SEG)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)

    # Every decay exponent below is formed DIFFERENCE-FIRST, as
    # a_val * (a non-negative difference of chunk-local cumsums); there is
    # no clamp -- delta >= 0 is a caller obligation and a_log past exp's
    # range is refused at the entry point (K2c) — see the "Stability by construction" block at the top of
    # fused_fwd_kernel.py.  The backward must use the SAME association as
    # the forward: dec_mask / dec_agg are recomputed here, not read back,
    # so a different association would give gradients of a slightly
    # different model.
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    cg = h // HPG_C

    if PAIR_HALF:
        # ═══ PAIRHALF: the ROTATE-HALF pair layout ═════════════════════
        # Ka carries the readout recurrence v.  The state axis is an OUTPUT
        # axis of its one dot (C~^T (f dY)) and of v / dG / S_all, so it
        # becomes two contiguous [N/2, GT] accumulators; the rotation is four
        # FMAs on two [T, N/2] tiles with no cross-lane partner.  The phase
        # and read offset act on the N axis of the tile this CTA's own head
        # loads and the o loop applies them per output slot, so O > 1 needs no
        # new algebra; the split costs two half-width MMAs per o.
        KH: tl.constexpr = N // 2
        offs_h1 = tl.arange(0, KH)
        offs_h2 = KH + tl.arange(0, KH)
        if HAS_THETA_H:
            th_h = tl.load(ThetaH_ptr + h * KH + offs_h1)
            ch_h = _cos_approx(th_h)
            sh_h = _sin_approx(th_h)
        if HAS_DSF:
            f_base = (dSF_ptr + b * s_fb + h * s_fh + seg * s_dsfs
                      + offs_gt[None, :] * s_fg)
            v1 = tl.load(f_base + offs_h1[:, None] * s_fn).to(tl.float32)
            v2 = tl.load(f_base + offs_h2[:, None] * s_fn).to(tl.float32)
        else:
            v1 = tl.zeros([KH, GT], dtype=tl.float32)
            v2 = tl.zeros([KH, GT], dtype=tl.float32)

        for ci in range(c_hi - c_lo):
            c = c_hi - 1 - ci
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            sb = (SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                  + offs_gt[None, :] * s_sg)
            S1 = tl.load(sb + offs_h1[:, None] * s_sn).to(tl.float32)
            S2 = tl.load(sb + offs_h2[:, None] * s_sn).to(tl.float32)
            dalpha_c = tl.sum(v1 * S1) + tl.sum(v2 * S2)
            if HAS_ROT:
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk,
                ).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + offs_h1 * s_ok,
                    ).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
            if not SEG_SUMMARY:
                tl.store(
                    dAlpha_ptr + b * s_dab + h * s_dah
                    + gt_idx * s_dag + c * s_dac, dalpha_c)
                gb = (dG_ptr + b * s_gb + h * s_gh + c * s_gc
                      + offs_gt[None, :] * s_gg)
                tl.store(gb + offs_h1[:, None] * s_gn, v1)
                tl.store(gb + offs_h2[:, None] * s_gn, v2)

            dS1 = tl.zeros([KH, GT], dtype=tl.float32)
            dS2 = tl.zeros([KH, GT], dtype=tl.float32)
            for o in tl.static_range(O):
                c_base = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                C1 = tl.load(c_base + offs_t[:, None] * s_ct
                             + offs_h1[None, :] * s_cn).to(tl.float32)
                C2 = tl.load(c_base + offs_t[:, None] * s_ct
                             + offs_h2[None, :] * s_cn).to(tl.float32)
                if HAS_GAMMA:
                    ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O + o
                                  ).to(tl.float32)
                    ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O + o
                                  ).to(tl.float32)
                    if BIAS_PRE_ROT:
                        C1 = C1 + ga1[None, :]
                        C2 = C2 + ga2[None, :]
                if HAS_ROT:
                    Ct1 = ph_cos * C1 + ph_sin * C2
                    Ct2 = ph_cos * C2 - ph_sin * C1
                else:
                    Ct1 = C1
                    Ct2 = C2
                if HAS_THETA_H:
                    Cr1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                    Cr2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                else:
                    Cr1 = Ct1
                    Cr2 = Ct2
                if HAS_GAMMA and not BIAS_PRE_ROT:
                    if HAS_THETA_H:
                        g1 = ch_h * ga1 + sh_h * ga2
                        g2 = ch_h * ga2 - sh_h * ga1
                    else:
                        g1 = ga1
                        g2 = ga2
                    Cr1 = Cr1 + g1[None, :]
                    Cr2 = Cr2 + g2[None, :]
                dY_o = tl.load(
                    dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo
                    + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                ).to(tl.float32)
                f_dY_o = (f[:, None] * dY_o).to(MMA_DT)
                dS1 += pdot(tl.trans(Cr1).to(MMA_DT), f_dY_o,
                              input_precision=PREC)
                dS2 += pdot(tl.trans(Cr2).to(MMA_DT), f_dY_o,
                              input_precision=PREC)
            v1 = alpha * v1 + dS1
            v2 = alpha * v2 + dS2

        if SEG_SUMMARY:
            xb = (dX0_ptr + b * s_x0b + h * s_x0h + seg * s_x0ss
                  + offs_gt[None, :] * s_x0g)
            tl.store(xb + offs_h1[:, None] * s_x0n, v1)
            tl.store(xb + offs_h2[:, None] * s_x0n, v2)
        else:
            if HAS_X0:
                if c_lo == 0:
                    xb = (dX0_ptr + b * s_x0b + h * s_x0h
                          + offs_gt[None, :] * s_x0g)
                    tl.store(xb + offs_h1[:, None] * s_x0n, v1)
                    tl.store(xb + offs_h2[:, None] * s_x0n, v2)
    else:
        ROT_ANY: tl.constexpr = HAS_THETA_H or HAS_ROT
        if ROT_ANY:
            pair_idx = offs_n // 2
            fwd_sign = tl.where((offs_n % 2) == 0, 1.0, -1.0)
        if HAS_THETA_H:
            th_full = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)
            ch_full = _cos_approx(th_full)
            sh_full = _sin_approx(th_full)
            fwd_coeff = fwd_sign * sh_full

        if HAS_DSF:
            v = tl.load(
                dSF_ptr + b * s_fb + h * s_fh + seg * s_dsfs
                + offs_gt[None, :] * s_fg + offs_n[:, None] * s_fn,
            ).to(tl.float32)
        else:
            v = tl.zeros([N, GT], dtype=tl.float32)

        for ci in range(c_hi - c_lo):
            c = c_hi - 1 - ci

            # Decay
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)

            # dalpha = sum(v * S_c) — computed before GEMM to free S_c early
            if not NO_DALPHA:
                S_c = tl.load(
                    SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                    + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg,
                ).to(tl.float32)
                dalpha_c = tl.sum(v * S_c)

            if HAS_ROT:
                # loaded on its own PAIR axis: `pair_idx = offs_n // 2` is not
                # affine in the lane id, so the [T, N] form is a scalar
                # gather for a tensor with T x N/2 distinct values.
                # s_thh: 0 for the head-SHARED phase, theta.stride(3) for the
                # per-head [B, nc, T, H, K] one.  This CTA owns one head,
                # so the per-head form is the same load at a shifted base.
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht
                    + tl.arange(0, N // 2)[None, :] * s_thk,
                ).to(tl.float32)                                    # [T, N//2]
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + tl.arange(0, N // 2) * s_ok,
                    ).to(tl.float32)[None, :]
                cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, N)
                sn_rot = fwd_sign[None, :] * _pair_bcast_2d(
                    _sin_approx(th_rot), T, N)

            if not SEG_SUMMARY:
                if not NO_DALPHA:
                    tl.store(
                        dAlpha_ptr + b * s_dab + h * s_dah
                        + gt_idx * s_dag + c * s_dac,
                        dalpha_c)

                # Store dG = v (pre-update value = v_old for this chunk)
                tl.store(
                    dG_ptr + b * s_gb + h * s_gh + c * s_gc
                    + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg,
                    v)

            # v recurrence: v = alpha * v + sum_o C_o_rd^T @ (f * dY_o)
            dS_readout = tl.zeros([N, GT], dtype=tl.float32)
            for o in tl.range(O, loop_unroll_factor=1):
                c_base = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                Co = tl.load(
                    c_base + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn,
                ).to(tl.float32)
                if BIAS_PRE_ROT and HAS_GAMMA:
                    # Mamba-3's fold ORDER: the read bias joins the RAW C and
                    # rotates WITH it.
                    Co = Co + tl.load(
                        Gamma_ptr + h * N * O + offs_n * O + o
                    ).to(tl.float32)[None, :]
                if HAS_ROT:
                    # C~ = R(theta_t)^T C, pairwise on the N axis
                    Co = cs_rot * Co + sn_rot * _pair_swap_2d(Co, T, N)
                if HAS_THETA_H:
                    if HAS_ROT:
                        Co_xor = _pair_swap_2d(Co, T, N)
                    else:
                        Co_xor = tl.load(
                            c_base + offs_t[:, None] * s_ct
                            + (offs_n ^ 1)[None, :] * s_cn,
                        ).to(tl.float32)
                    C_rd_o = ch_full[None, :] * Co + fwd_coeff[None, :] * Co_xor
                else:
                    C_rd_o = Co
                if HAS_GAMMA and not BIAS_PRE_ROT:
                    gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                    if HAS_THETA_H:
                        gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                        gam = ch_full * gam + fwd_coeff * gam_xor
                    C_rd_o = C_rd_o + gam[None, :]
                dY_o = tl.load(
                    dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo
                    + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                ).to(tl.float32)
                f_dY_o = f[:, None] * dY_o
                dS_readout += pdot(tl.trans(C_rd_o).to(MMA_DT), f_dY_o.to(MMA_DT), input_precision=PREC)
            v = alpha * v + dS_readout

        if SEG_SUMMARY:
            # phase A: per-segment zero-seeded final v (the stitch input V_s)
            tl.store(
                dX0_ptr + b * s_x0b + h * s_x0h + seg * s_x0ss
                + offs_gt[None, :] * s_x0g + offs_n[:, None] * s_x0n,
                v)
        else:
            if HAS_X0:
                if c_lo == 0:
                    tl.store(
                        dX0_ptr + b * s_x0b + h * s_x0h
                        + offs_gt[None, :] * s_x0g + offs_n[:, None] * s_x0n,
                        v)






@triton.autotune(configs=_calib_grid("fused_kb1", _bwd_grid),
                 prune_configs_by={"early_config_prune": _cprune("fused_kb1")},
                 key=["T", "N", "G_SIZE", "GT", "PREC", "MMA_IO",
                      "HAS_THETA_H", "HAS_BETA", "HAS_GAMMA",
                      "HAS_ROT_BIAS", "HAS_ROT", "BIAS_PRE_ROT",
                      "PHASE_CUMSUM",
                      "R", "O", "HAS_DW", "IDX64",
                      "HBLK", "DB_STORE", "DC_STORE", "ROT_ABL",
                      "PAIR_HALF", "KB2_FOLD", "DTH_STORE", "ROT_COMB",
                      # the N split is a different body: without it here a tuned
                      # pick made at one split is replayed (in process and from
                      # the disk cache) at another.
                      "NSPLIT",
                      "OCC"],
                 # Atomic-accumulating outputs MUST be restored between
                 # autotune trials, else the first call per config-key
                 # returns gradients inflated by every bench replay.
                 restore_value=["dCtil_ptr", "dBtil_ptr", "dDelta_ptr",
                                "dThetaH_ptr", "dBeta_ptr", "dGamma_ptr",
                                "dRotBias_ptr", "dTheta_ptr", "dThOff_ptr",
                                "Scratch_ptr"])
@triton.jit
def _bwd_kb1_kernel(
    C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, ThetaH_ptr, SAll_ptr,
    Beta_ptr, Gamma_ptr, RotBias_ptr,
    dY_ptr,
    dCtil_ptr, dBtil_ptr, dDelta_ptr, dThetaH_ptr, dBeta_ptr, dGamma_ptr,
    dRotBias_ptr,
    Scratch_ptr,
    H,
    s_cb, s_cc, s_ct, s_cn,
    s_bb, s_bc, s_bt, s_bn,
    s_xb, s_xh, s_xg, s_xr, s_xl,
    s_db, s_dh, s_dc, s_dt,
    s_yb, s_yh, s_yc, s_yt, s_yg, s_yo,
    s_sb, s_sh, s_sc, s_sn, s_sg,
    s_dcb, s_dcc, s_dct, s_dcn,
    s_dbb, s_dbc, s_dbt, s_dbn,
    s_bh, s_ch,
    NC, NGT_val,
    T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr,
    GT: tl.constexpr, PREC: tl.constexpr,
    # `tl.dot` operand dtype: fp32 = identity cast; bf16 / fp16 = IO-dtype MMA
    # with fp32 accumulate.  See `mma_operand_dtype` in fused_fwd_kernel.py.
    MMA_DT: tl.constexpr = tl.float32,
    # Autotune KEY stand-in for MMA_DT (a tl.dtype is not serialisable into
    # Triton's disk cache key): 1 iff MMA_DT is 16-bit.
    MMA_IO: tl.constexpr = 0,
    HAS_THETA_H: tl.constexpr = False,
    NEED_DTHH: tl.constexpr = False,
    DEFERRED_ROTATION: tl.constexpr = False,
    HAS_BETA: tl.constexpr = False,
    HAS_GAMMA: tl.constexpr = False,
    HAS_ROT_BIAS: tl.constexpr = False,
    HPG_B: tl.constexpr = 1, HPG_C: tl.constexpr = 1,
    R: tl.constexpr = 1, O: tl.constexpr = 1,
    s_br: tl.constexpr = 0, s_co: tl.constexpr = 0,
    s_dbr: tl.constexpr = 0, s_dco: tl.constexpr = 0,
    CHUNK_PARALLEL: tl.constexpr = False,
    DW_ptr=None, s_wb=0, s_wh=0, s_wl=0,
    HAS_DW: tl.constexpr = False,
    # ── SLOTS: head-sliced atomic accumulation.  dBtil/dCtil are shared
    # across all H heads, so H CTAs collide on every atomic tile.  With
    # SLOTS > 1 the buffers carry a leading slot axis (slot = h % SLOTS,
    # strides s_slb/s_slc); the host sums the slots afterwards.
    s_slb=0, s_slc=0,
    SLOTS: tl.constexpr = 1,
    # group offsets for GROUPED dBtil/dCtil ([B, nc, Gx, T, N])
    s_dbh=0, s_dch=0,
    IDX64: tl.constexpr = False,
    # In-kernel per-(kernel head, token) phase in the GENERIC branch: B/C
    # arrive RAW, this kernel forms B~/C~ on load, un-rotates its two
    # accumulators before the atomics, and accumulates dTheta itself.
    Theta_ptr=None,
    s_thb=0, s_thc=0, s_tht=0, s_thh=0, s_thk=0,
    dTheta_ptr=None,
    s_dtb=0, s_dth=0, s_dtc=0, s_dtt=0, s_dtk=0,
    HAS_ROT: tl.constexpr = False,
    BIAS_PRE_ROT: tl.constexpr = False,
    ThOff_ptr=None,
    s_ob=0, s_oh=0, s_oc=0, s_ok=0,
    dThOff_ptr=None,
    s_dob=0, s_doh=0, s_doc=0, s_dok=0,
    PHASE_CUMSUM: tl.constexpr = False,
    # PHASE_DEFER: the kernel stores the RAW dL/dtheta_cum and one
    # chunk-parallel pass (`m3_epilogue.phase_prefix_bwd`) does the in-chunk
    # reverse cumsum, the chunk total and the cross-chunk term (an in-kernel
    # cross-thread scan per head is too costly in this register-starved body).
    PHASE_DEFER: tl.constexpr = False,
    # ── N-SPLIT of the score-path backward ──────────────────────────
    # Grid dim-2 carries an extra N-slice axis, each CTA owns
    # `NS = N // NSPLIT` state columns, and every N-DEPENDENT tile
    # ([T, N] C/B/dC/dB, [N, GT] S_c) shrinks with it while the
    # N-INDEPENDENT work (the [T, T] score product dY @ x^T and the
    # dY / x / dw loads) is repeated per slice.  At R = O = 1 the
    # launch rule pins NSPLIT = 1, so that route compiles the identical
    # code.  See `_kb1_nsplit`.
    NSPLIT: tl.constexpr = 1,
    # ── OWNER-ACCUMULATE head block ──────────────────────────────
    # HBLK = 0 compiles the one-head-per-CTA body (the `else` branch below);
    # HBLK > 0 gives the CTA a block of HBLK
    # heads, sums dBtil/dCtil in registers across it (and across the o
    # loop, for dB) and writes each tile ONCE.  DB_STORE / DC_STORE say
    # the CTA owns the tile's WHOLE fan-in, so the write is a plain
    # `tl.store`; the host decides both (see `_kb1_hb`).
    HBLK: tl.constexpr = 0,
    # ── ROTBWD: MEASUREMENT-ONLY rotation ablation bitmask
    # (`SSD_MIMO_ROT_ABL`).  Every bit but 0 computes the WRONG
    # gradient; 0 compiles this kernel exactly as before.
    #   1  ABL_NO_DTH   -- no phase-gradient accumulation / store
    #   2  ABL_PREROT   -- B~/C~ taken as already rotated on load (the
    #                      cost of lever (i), without its extra traffic)
    ROT_ABL: tl.constexpr = 0,
    DB_STORE: tl.constexpr = False,
    DC_STORE: tl.constexpr = False,
    # DTH_STORE: the per-head phase gradient as a plain STORE.
    # Under the in-kernel phase the `dTheta` tile [T, N/2] of (b, c, h) has
    # ONE writer when Kb1 absorbed Kb2 (KB2_FOLD), Kc owns no aggregate dB
    # (not FUSE_KB2) and one CTA covers the whole G axis (NGT == 1): the
    # head-blocked CTA that owns head h.  The host then allocates the
    # buffer with `torch.empty` instead of a zero-fill and this body writes
    # every (t, k) of every valid head exactly once.  Same values as the
    # atomic into zeros (0 + x == x in IEEE fp32).  See `_kb1_dth_store`.
    DTH_STORE: tl.constexpr = False,
    # ROT_COMB: the READ side's two rotations as ONE.
    # C_h = R(theta_h)^T R(theta_t)^T C' = R(theta_t + theta_h)^T C' (pair
    # rotations compose additively), so the head-blocked body builds C_h
    # with one pair swap of the LOADED tile, and its adjoint
    # R(theta_t + theta_h) gC is ONE swap of the MMA-output gradient
    # instead of two.  The two angle gradients of the read side are then
    # the same product: d/dtheta_t (through C) and d/dtheta_h are both
    # <gC, J C_h> per (t, pair) (a 2x2 rotation commutes with J, so the old
    # <R(theta_h) gC, J C~> is the same number), i.e. `dth_acc` and the
    # dThetaH column sum share one [T, N] product.  cos/sin of the sum come
    # from the angle-addition identities on tiles the body already holds
    # (cs_rot / sn_rot and the per-head ch / fwd_coeff vectors), formed at
    # each use.  Same mathematics as ROT_COMB = False, different fp32
    # rounding (the two rotations are fused).  Served where the read map
    # has no post-rotation gamma (BIAS_PRE_ROT or no gamma) and no rot_bias.
    ROT_COMB: tl.constexpr = False,
    # PAIRHALF: the rotate-half pair layout (see `fused_fwd_kernel.py`).  Kb1
    # is a GENERAL-RANK kernel, so its half branch serves every (R, O).
    PAIR_HALF: tl.constexpr = False,
    # `Delta_ptr` holds the RAW chunked delta; this kernel forms the
    # chunk-local prefix D_i itself (`_dcs_tile` / `_dcs_end`).
    DCS_INK: tl.constexpr = False,
    # ── KB2_FOLD: Kb2 ABSORBED into the head block
    # Kb2's aggregate term writes the SAME `dBtil` tile this body already
    # owns, so it is folded into `dBp[r]` BEFORE the o loop -- see the
    # `KB2_FOLD` block in the `HBLK > 0` branch and `_kb2_fold()` for the
    # algebra (Kb2's two GEMMs become one, and its whole `dDelta`
    # contribution falls out of Kb1's existing `Bc . dB` reduction).
    # Refused by name at R*O == 1 (no Kb1 launch).
    #
    # `KB2_FOLD and HAS_ROT` is a
    # SERVED combination.  The seed `dB_agg_r = (x_r dec_agg) dG^T` is
    # the gradient w.r.t. the SAME tile the score dot reads (`Bc_l[r]`:
    # R(theta) B + beta), so it enters `dBp[r]` BEFORE the rotating
    # write-side chain and rides it unchanged: the dBeta row sum, the
    # phase gradient `dth_acc += gB_ * swap(B~)` (Kb2's aggregate-path
    # dTheta term joins Kb1's `dth_acc` -- ONE dTheta atomic per head
    # instead of two), the un-rotation `cs gB - sn swap(gB)` and the
    # `dD -= a sum(Bc dB)` reduction (which is dec_agg q_r on the seed,
    # Kb2's first dDelta term).  No new [T, N] tile: the rotated B~ per
    # kernel head is the one this body already builds, and the fold's
    # only operand of its own is the [NS, GT] `dG` slab.  Still refused:
    # `HAS_ROT_BIAS`, `PAIR_HALF`, deferred rotation, the ablation
    # ladder, the slot layouts (the static assert below pins the set).
    KB2_FOLD: tl.constexpr = False,
    dG_ptr=None, s_gb=0, s_gh=0, s_gc=0, s_gn=0, s_gg=0,
    dAlpha_ptr=None, s_dab=0, s_dah=0, s_dag=0, s_dac=0,
    # TRAPFOLD (fused_fwd_kernel `_TRAPFOLD_NOTE`): under the two-tap
    # trapezoid the score diagonal's dB / dC (weight own_t, not the
    # collapsed w_t this kernel sees) is carried by `trap_adjoint`
    # (dd_t = own_t s_t); this kernel's diagonal is then zero.
    TRAP_DIAG: tl.constexpr = False,
    OCC=0,           # autotune-key occupancy bucket only (`occ_bucket`)
):
    """Kb1: Score-path gradients → dC + dThetaH + partial dB + partial dDelta.

    4 GEMMs/chunk.  GEMM 4 (dB_score) is scheduled before GEMM 3 (dC_score)
    to shorten C_rd's MMA live range and reduce dual-layout register pressure.
    dDelta reformulated: dD = a*(sum(C_rd*dC_rd,1) - sum(Bc*dBscore,1))

    DEFERRED_ROTATION: when True, stores dC_rd to Scratch_ptr (used as
    dCrd_buf) instead of doing inverse rotation inline.  A separate kinv
    kernel handles the inverse rotation and dThetaH.

    CHUNK_PARALLEL: when True, grid dim-3 encodes chunk index (grid is
    [B, H, nc*NGT] or [B, H, nc, NGT] flattened).  Each CTA processes one
    chunk instead of looping over all chunks.
    """
    tl.static_assert(not BIAS_PRE_ROT or (HAS_ROT and not HAS_ROT_BIAS),
                     "Kb1: bias_pre_rot is Mamba-3's fold order -- it "
                     "needs the in-kernel phase and no write-side angle "
                     "(the READ offset theta_h composes after the phase)")
    tl.static_assert(not HAS_ROT or not DEFERRED_ROTATION,
                     "Kb1: with the in-kernel phase the theta_h "
                     "un-rotation is done in registers, not deferred to "
                     "Kinv (which knows nothing about theta_t)")
    tl.static_assert(not ROT_COMB or (HAS_ROT and HAS_THETA_H
                                      and not HAS_ROT_BIAS and ROT_ABL == 0
                                      and not PAIR_HALF
                                      and (BIAS_PRE_ROT or not HAS_GAMMA)),
                     "Kb1: ROT_COMB fuses the phase and the static read "
                     "offset; it needs both, no rot_bias, no ablation and "
                     "no post-rotation gamma")
    tl.static_assert(not DTH_STORE or (HBLK > 0 and KB2_FOLD and HAS_ROT
                                       and not PAIR_HALF),
                     "Kb1: DTH_STORE is the sole-owner dTheta write of the "
                     "head-blocked interleaved body with Kb2 folded in")
    tl.static_assert(not KB2_FOLD or (HBLK > 0 and not PAIR_HALF
                                      and not HAS_ROT_BIAS
                                      and not DEFERRED_ROTATION
                                      and ROT_ABL == 0 and SLOTS == 1),
                     "Kb1: KB2_FOLD lives in the head-blocked interleaved "
                     "body only (real mode and the in-kernel phase HAS_ROT); "
                     "see `_kb2_fold`")
    ABL_NO_DTH: tl.constexpr = (ROT_ABL & 1) != 0
    ABL_PREROT: tl.constexpr = (ROT_ABL & 2) != 0
    if PAIR_HALF:
        # ═══ PAIRHALF: Kb1 on the ROTATE-HALF layout ═════════════════
        # Pair k lives at state coordinates (k, k + N/2), so every
        # rotated tile is TWO CONTIGUOUS [T, PS] tiles whose pair partner
        # IS THE OTHER TILE.  This branch therefore contains no
        # `_pair_swap_2d`, no `_pair_bcast_2d`, no `offs_n ^ 1` load --
        # and no ROTBWD ALGEBRAIC partner either: `swap(B~) = cs
        # swap(B) - sn B` existed only to avoid the cross-lane shuffle,
        # and under this layout the partner is a tile the build already
        # holds.  The four maps are
        #   B~_1 = c B_1 + s B_2        B~_2 = c B_2 - s B_1
        #   dB_1 = c dB~_1 - s dB~_2    dB_2 = c dB~_2 + s dB~_1
        # and the phase gradient (otherwise a cross-lane split / join) is
        #   dTheta_k = sum_r (dB~_1 B~_2 - dB~_2 B~_1)
        #            + sum_o (dC~_1 C~_2 - dC~_2 C~_1)
        # with no split / join at all.
        #
        # Register diet (the split doubles the live TILE COUNT at the same
        # element count):
        #   * the [T, T] decay mask is built AT its consumer (`DM`),
        #     not at the top of the head body;
        #   * `x` is cast to MMA_DT at the build, `dY` loaded at first
        #     use, `S_all` read as two [PS, GT] halves once per head;
        #   * nothing UN-ROTATED is kept live that the body does not
        #     read: the `B~` halves survive only because `dTheta` needs
        #     them (they are the o-loop's own operands otherwise), and
        #     the post-rotation `gamma` fold is materialised only where
        #     `BIAS_PRE_ROT` is off.
        #
        # The N SPLIT slices the PAIR axis (`PS = N/(2*NSPLIT)`), since a
        # contiguous n range is not a set of whole pairs here.
        tl.static_assert(not HAS_ROT_BIAS,
                         "PAIR_HALF does not serve rot_bias")
        tl.static_assert(ROT_ABL == 0,
                         "PAIR_HALF carries no ablation ladder")
        tl.static_assert(not DEFERRED_ROTATION,
                         "PAIR_HALF un-rotates theta_h in registers")
        tl.static_assert(N >= 32 * NSPLIT,
                         "PAIR_HALF needs N/(2*NSPLIT) >= 16 (MMA K floor)")
        PS: tl.constexpr = N // (2 * NSPLIT)
        KH: tl.constexpr = N // 2
        neg_inf = float("-inf")
        offs_t = tl.arange(0, T)
        if HBLK > 0:
            b = tl.program_id(0)
            hblk = tl.program_id(1)
            pid2 = tl.program_id(2)
            ns_idx = pid2 % NSPLIT
            rest2 = pid2 // NSPLIT
            if CHUNK_PARALLEL:
                c = rest2 // NGT_val
                gt_idx = rest2 % NGT_val
            else:
                gt_idx = rest2
            if IDX64:
                b = b.to(tl.int64)
            offs_p = ns_idx * PS + tl.arange(0, PS)
            offs_h1 = offs_p
            offs_h2 = KH + offs_p
            offs_gt = gt_idx * GT + tl.arange(0, GT)
            # The score diagonal rides the MMAs (weight w_t); under the
            # trapezoid it is STRICT (`trap_adjoint` carries own_t)
            if TRAP_DIAG:
                causal = offs_t[:, None] > offs_t[None, :]
            else:
                causal = offs_t[:, None] >= offs_t[None, :]
            h_start = hblk * HBLK
            if IDX64:
                h_start = h_start.to(tl.int64)
            bg = h_start // HPG_B
            cg = h_start // HPG_C

            nc_end = 1 if CHUNK_PARALLEL else NC
            for ci in range(nc_end):
                if CHUNK_PARALLEL:
                    pass
                else:
                    c = NC - 1 - ci

                dB1_l = ()
                dB2_l = ()
                for r in tl.static_range(R):
                    dB1_l = dB1_l + (tl.zeros([T, PS], dtype=tl.float32),)
                    dB2_l = dB2_l + (tl.zeros([T, PS], dtype=tl.float32),)
                dC1_l = ()
                dC2_l = ()
                for o in tl.static_range(O):
                    dC1_l = dC1_l + (tl.zeros([T, PS], dtype=tl.float32),)
                    dC2_l = dC2_l + (tl.zeros([T, PS], dtype=tl.float32),)

                for hh in range(HBLK):
                    h = h_start + hh
                    hv = h < H
                    h = tl.minimum(h, H - 1)
                    a_val = -tl.exp(tl.load(A_ptr + h))
                    if HAS_THETA_H:
                        th_h = tl.load(ThetaH_ptr + h * KH + offs_p)
                        ch_h = _cos_approx(th_h)
                        sh_h = _sin_approx(th_h)
                    dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
                    D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
                    seg_end = tl.max(seg_h, axis=0)
                    f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
                    if HAS_DW:
                        dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                                     + (c * T + offs_t) * s_wl)
                    if HAS_ROT:
                        th_rot = tl.load(
                            Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                            + offs_t[:, None] * s_tht
                            + offs_p[None, :] * s_thk,
                        ).to(tl.float32)                       # [T, PS]
                        if PHASE_CUMSUM:
                            th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                                ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                                + offs_p * s_ok,
                            ).to(tl.float32)[None, :]
                        ph_cos = _cos_approx(th_rot)
                        ph_sin = _sin_approx(th_rot)
                        dth_acc = tl.zeros([T, PS], dtype=tl.float32)

                    dD = tl.zeros([T], dtype=tl.float32)

                    Bt1_l = ()
                    Bt2_l = ()
                    Be1_l = ()
                    Be2_l = ()
                    x_l = ()
                    dBp1 = ()
                    dBp2 = ()
                    for r in tl.static_range(R):
                        b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                    + r * s_br + offs_t[:, None] * s_bt)
                        B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn
                                     ).to(tl.float32)
                        B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn
                                     ).to(tl.float32)
                        if HAS_BETA:
                            be1 = tl.load(Beta_ptr + h * N * R
                                          + offs_h1 * R + r).to(tl.float32)
                            be2 = tl.load(Beta_ptr + h * N * R
                                          + offs_h2 * R + r).to(tl.float32)
                            if BIAS_PRE_ROT:
                                B1 = B1 + be1[None, :]
                                B2 = B2 + be2[None, :]
                        if HAS_ROT:
                            Bt1 = ph_cos * B1 + ph_sin * B2
                            Bt2 = ph_cos * B2 - ph_sin * B1
                        else:
                            Bt1 = B1
                            Bt2 = B2
                        if HAS_BETA and not BIAS_PRE_ROT:
                            Bf1 = Bt1 + be1[None, :]
                            Bf2 = Bt2 + be2[None, :]
                        else:
                            Bf1 = Bt1
                            Bf2 = Bt2
                        x_r = tl.load(
                            X_ptr + b * s_xb + h * s_xh + r * s_xr
                            + c * T * s_xl + offs_gt[None, :] * s_xg
                            + offs_t[:, None] * s_xl,
                        ).to(tl.float32)
                        if HAS_DW:
                            x_r = x_r * dw[:, None]
                        Bt1_l = Bt1_l + (Bt1,)
                        Bt2_l = Bt2_l + (Bt2,)
                        Be1_l = Be1_l + (Bf1,)
                        Be2_l = Be2_l + (Bf2,)
                        # the score dot is x's ONLY consumer: cast at the
                        # build, one value, rounded once.
                        x_l = x_l + (tl.trans(x_r).to(MMA_DT),)
                        dBp1 = dBp1 + (tl.zeros([T, PS], dtype=tl.float32),)
                        dBp2 = dBp2 + (tl.zeros([T, PS], dtype=tl.float32),)

                    s_base = (SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                              + offs_gt[None, :] * s_sg)
                    S1 = tl.load(s_base + offs_h1[:, None] * s_sn
                                 ).to(tl.float32)
                    S2 = tl.load(s_base + offs_h2[:, None] * s_sn
                                 ).to(tl.float32)

                    dCp1 = ()
                    dCp2 = ()
                    for o in tl.static_range(O):
                        c_base_o = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                                    + o * s_co + offs_t[:, None] * s_ct)
                        C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn
                                     ).to(tl.float32)
                        C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn
                                     ).to(tl.float32)
                        if HAS_GAMMA:
                            ga1 = tl.load(Gamma_ptr + h * N * O
                                          + offs_h1 * O + o).to(tl.float32)
                            ga2 = tl.load(Gamma_ptr + h * N * O
                                          + offs_h2 * O + o).to(tl.float32)
                            if BIAS_PRE_ROT:
                                C1 = C1 + ga1[None, :]
                                C2 = C2 + ga2[None, :]
                        if HAS_ROT:
                            Ct1 = ph_cos * C1 + ph_sin * C2
                            Ct2 = ph_cos * C2 - ph_sin * C1
                        else:
                            Ct1 = C1
                            Ct2 = C2
                        if HAS_THETA_H:
                            Ch1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                            Ch2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                        else:
                            Ch1 = Ct1
                            Ch2 = Ct2
                        Cr1 = Ch1
                        Cr2 = Ch2
                        if HAS_GAMMA and not BIAS_PRE_ROT:
                            if HAS_THETA_H:
                                gr1 = ch_h * ga1 + sh_h * ga2
                                gr2 = ch_h * ga2 - sh_h * ga1
                            else:
                                gr1 = ga1
                                gr2 = ga2
                            Cr1 = Cr1 + gr1[None, :]
                            Cr2 = Cr2 + gr2[None, :]

                        dY_o = tl.load(
                            dY_ptr + b * s_yb + h * s_yh + c * s_yc
                            + o * s_yo + offs_t[:, None] * s_yt
                            + offs_gt[None, :] * s_yg,
                        ).to(tl.float32)

                        dCs1 = tl.zeros([T, PS], dtype=tl.float32)
                        dCs2 = tl.zeros([T, PS], dtype=tl.float32)
                        dBp1_n = ()
                        dBp2_n = ()
                        for r in tl.static_range(R):
                            dScore_or = pdot(dY_o.to(MMA_DT), x_l[r],
                                               input_precision=PREC)
                            # The [T, T] mask is built at its consumer (same
                            # difference-first order as the hoisted form).
                            dSb_dt = (dScore_or * tl.exp(
                                tl.where((causal) & _same_seg(seg_h),
                                         a_val * (D_h[:, None] - D_h[None, :]),
                                         neg_inf))).to(MMA_DT)
                            dCs1 += pdot(dSb_dt, Be1_l[r].to(MMA_DT),
                                           input_precision=PREC)
                            dCs2 += pdot(dSb_dt, Be2_l[r].to(MMA_DT),
                                           input_precision=PREC)
                            dSb_tr = tl.trans(dSb_dt)
                            tB1 = pdot(dSb_tr, Cr1.to(MMA_DT),
                                         input_precision=PREC)
                            tB2 = pdot(dSb_tr, Cr2.to(MMA_DT),
                                         input_precision=PREC)
                            dBp1_n = dBp1_n + (
                                dBp1[r] + tB1,)
                            dBp2_n = dBp2_n + (
                                dBp2[r] + tB2,)
                        dBp1 = dBp1_n
                        dBp2 = dBp2_n

                        dY_dt = dY_o.to(MMA_DT)
                        dCst1 = f[:, None] * pdot(
                            dY_dt, tl.trans(S1).to(MMA_DT),
                            input_precision=PREC)
                        dCst2 = f[:, None] * pdot(
                            dY_dt, tl.trans(S2).to(MMA_DT),
                            input_precision=PREC)
                        # SPANDELTA: Kb1 owns only the READ term of
                        # dL/ddelta (Kc owns the score pairs)
                        dD += a_val * (tl.sum(Cr1 * dCst1, axis=1)
                                       + tl.sum(Cr2 * dCst2, axis=1))
                        dC1 = dCs1 + dCst1
                        dC2 = dCs2 + dCst2

                        gC1 = dC1
                        gC2 = dC2
                        if HAS_GAMMA and not BIAS_PRE_ROT:
                            sg1 = tl.sum(gC1, axis=0)
                            sg2 = tl.sum(gC2, axis=0)
                            if HAS_THETA_H:
                                if NEED_DTHH:
                                    # d(R(nu)^T gamma)/d nu contracted
                                    # with the column sums.
                                    tl.atomic_add(
                                        dThetaH_ptr + h * KH + offs_p,
                                        sg1 * gr2 - sg2 * gr1, mask=hv,
                                        sem=ACC_SEM)
                                sn1 = ch_h * sg1 - sh_h * sg2
                                sg2 = ch_h * sg2 + sh_h * sg1
                                sg1 = sn1
                            tl.atomic_add(dGamma_ptr + h * N * O
                                          + offs_h1 * O + o, sg1, mask=hv,
                                          sem=ACC_SEM)
                            tl.atomic_add(dGamma_ptr + h * N * O
                                          + offs_h2 * O + o, sg2, mask=hv,
                                          sem=ACC_SEM)
                        if HAS_THETA_H:
                            if NEED_DTHH:
                                # d C_h_1 / d nu = C_h_2,
                                # d C_h_2 / d nu = -C_h_1.
                                tl.atomic_add(
                                    dThetaH_ptr + h * KH + offs_p,
                                    tl.sum(gC1 * _opv(Ch2, MMA_DT, MMA_IO, PREC)
                                       - gC2 * _opv(Ch1, MMA_DT, MMA_IO, PREC), axis=0),
                                    mask=hv, sem=ACC_SEM)
                            gn1 = ch_h[None, :] * gC1 - sh_h[None, :] * gC2
                            gC2 = ch_h[None, :] * gC2 + sh_h[None, :] * gC1
                            gC1 = gn1
                        if HAS_ROT:
                            if HAS_THETA_H:
                                Cq1, Cq2 = _ph_read_consistent(
                                    Ch1, Ch2, ch_h, sh_h, Ct1, Ct2, MMA_DT,
                                    MMA_IO, HAS_THETA_H, PREC)     # THETAFREQ
                            else:      # no theta_h: ch_h / sh_h / Ch* do not exist in this specialisation
                                Cq1 = _opv(Ct1, MMA_DT, MMA_IO, PREC)
                                Cq2 = _opv(Ct2, MMA_DT, MMA_IO, PREC)
                            dth_acc += gC1 * Cq2 - gC2 * Cq1
                            gn1 = ph_cos * gC1 - ph_sin * gC2
                            gC2 = ph_cos * gC2 + ph_sin * gC1
                            gC1 = gn1
                        if HAS_GAMMA and BIAS_PRE_ROT:
                            tl.atomic_add(dGamma_ptr + h * N * O
                                          + offs_h1 * O + o,
                                          tl.sum(gC1, axis=0), mask=hv,
                                          sem=ACC_SEM)
                            tl.atomic_add(dGamma_ptr + h * N * O
                                          + offs_h2 * O + o,
                                          tl.sum(gC2, axis=0), mask=hv,
                                          sem=ACC_SEM)
                        dCp1 = dCp1 + (gC1,)
                        dCp2 = dCp2 + (gC2,)

                    dBw1 = ()
                    dBw2 = ()
                    for r in tl.static_range(R):
                        gB1 = dBp1[r]
                        gB2 = dBp2[r]
                        if HAS_BETA and not BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R
                                          + offs_h1 * R + r,
                                          tl.sum(gB1, axis=0), mask=hv,
                                          sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + h * N * R
                                          + offs_h2 * R + r,
                                          tl.sum(gB2, axis=0), mask=hv,
                                          sem=ACC_SEM)
                        if HAS_ROT:
                            dth_acc += (gB1 * _opv(Bt2_l[r], MMA_DT, MMA_IO, PREC)
                                        - gB2 * _opv(Bt1_l[r], MMA_DT, MMA_IO, PREC))
                            gn1 = ph_cos * gB1 - ph_sin * gB2
                            gB2 = ph_cos * gB2 + ph_sin * gB1
                            gB1 = gn1
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R
                                          + offs_h1 * R + r,
                                          tl.sum(gB1, axis=0), mask=hv,
                                          sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + h * N * R
                                          + offs_h2 * R + r,
                                          tl.sum(gB2, axis=0), mask=hv,
                                          sem=ACC_SEM)
                        dBw1 = dBw1 + (gB1,)
                        dBw2 = dBw2 + (gB2,)

                    dB1_n = ()
                    dB2_n = ()
                    for r in tl.static_range(R):
                        dB1_n = dB1_n + (dB1_l[r]
                                         + tl.where(hv, dBw1[r], 0.0),)
                        dB2_n = dB2_n + (dB2_l[r]
                                         + tl.where(hv, dBw2[r], 0.0),)
                    dB1_l = dB1_n
                    dB2_l = dB2_n
                    dC1_n = ()
                    dC2_n = ()
                    for o in tl.static_range(O):
                        dC1_n = dC1_n + (dC1_l[o]
                                         + tl.where(hv, dCp1[o], 0.0),)
                        dC2_n = dC2_n + (dC2_l[o]
                                         + tl.where(hv, dCp2[o], 0.0),)
                    dC1_l = dC1_n
                    dC2_l = dC2_n

                    if HAS_ROT:
                        dth_ph = dth_acc
                        if PHASE_CUMSUM and not PHASE_DEFER:
                            dth_off = tl.sum(dth_ph, axis=0)
                            dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                            tl.atomic_add(
                                dThOff_ptr + b * s_dob + h * s_doh
                                + c * s_doc + offs_p * s_dok,
                                dth_off, mask=hv, sem=ACC_SEM)
                        tl.atomic_add(
                            dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                            + offs_t[:, None] * s_dtt
                            + offs_p[None, :] * s_dtk, dth_ph, mask=hv,
                            sem=ACC_SEM)

                    # SPANDELTA: the read term reaches every s <= i
                    tl.atomic_add(
                        dDelta_ptr + b * s_db + h * s_dh + c * s_dc
                        + offs_t * s_dt, tl.cumsum(dD, axis=0, reverse=True),
                        mask=hv, sem=ACC_SEM)

                for r in tl.static_range(R):
                    db_p = (dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                            + offs_t[:, None] * s_dbt + r * s_dbr)
                    if DB_STORE:
                        tl.store(db_p + offs_h1[None, :] * s_dbn, dB1_l[r])
                        tl.store(db_p + offs_h2[None, :] * s_dbn, dB2_l[r])
                    else:
                        tl.atomic_add(db_p + offs_h1[None, :] * s_dbn,
                                      dB1_l[r], sem=ACC_SEM)
                        tl.atomic_add(db_p + offs_h2[None, :] * s_dbn,
                                      dB2_l[r], sem=ACC_SEM)
                for o in tl.static_range(O):
                    dc_p = (dCtil_ptr + cg * s_dch + b * s_dcb + c * s_dcc
                            + o * s_dco + offs_t[:, None] * s_dct)
                    if DC_STORE:
                        tl.store(dc_p + offs_h1[None, :] * s_dcn, dC1_l[o])
                        tl.store(dc_p + offs_h2[None, :] * s_dcn, dC2_l[o])
                    else:
                        tl.atomic_add(dc_p + offs_h1[None, :] * s_dcn,
                                      dC1_l[o], sem=ACC_SEM)
                        tl.atomic_add(dc_p + offs_h2[None, :] * s_dcn,
                                      dC2_l[o], sem=ACC_SEM)
        else:
            # ── the historical ONE-HEAD-PER-CTA body, on halves ───────
            # Same algebra; the difference is only WHERE the partials
            # land: every dB / dC tile leaves the CTA as its own atomic
            # (into the head SLOT plane), and dTheta / dDelta are the
            # per-head atomics they already were.
            b = tl.program_id(0)
            h = tl.program_id(1)
            pid2 = tl.program_id(2)
            ns_idx = pid2 % NSPLIT
            rest2 = pid2 // NSPLIT
            if CHUNK_PARALLEL:
                c = rest2 // NGT_val
                gt_idx = rest2 % NGT_val
            else:
                gt_idx = rest2
            if IDX64:
                b = b.to(tl.int64)
                h = h.to(tl.int64)
            a_val = -tl.exp(tl.load(A_ptr + h))
            offs_p = ns_idx * PS + tl.arange(0, PS)
            offs_h1 = offs_p
            offs_h2 = KH + offs_p
            offs_gt = gt_idx * GT + tl.arange(0, GT)
            # The score diagonal rides the MMAs (weight w_t); under the
            # trapezoid it is STRICT (`trap_adjoint` carries own_t)
            if TRAP_DIAG:
                causal = offs_t[:, None] > offs_t[None, :]
            else:
                causal = offs_t[:, None] >= offs_t[None, :]
            bg = h // HPG_B
            cg = h // HPG_C
            slot_b = (h % SLOTS) * s_slb
            slot_c = (h % SLOTS) * s_slc
            if HAS_THETA_H:
                th_h = tl.load(ThetaH_ptr + h * KH + offs_p)
                ch_h = _cos_approx(th_h)
                sh_h = _sin_approx(th_h)

            nc_end = 1 if CHUNK_PARALLEL else NC
            for ci in range(nc_end):
                if CHUNK_PARALLEL:
                    pass
                else:
                    c = NC - 1 - ci

                dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
                D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
                seg_end = tl.max(seg_h, axis=0)
                f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
                if HAS_DW:
                    dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                                 + (c * T + offs_t) * s_wl)
                if HAS_ROT:
                    th_rot = tl.load(
                        Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                        + offs_t[:, None] * s_tht + offs_p[None, :] * s_thk,
                    ).to(tl.float32)
                    if PHASE_CUMSUM:
                        th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                            ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                            + offs_p * s_ok,
                        ).to(tl.float32)[None, :]
                    ph_cos = _cos_approx(th_rot)
                    ph_sin = _sin_approx(th_rot)
                    dth_acc = tl.zeros([T, PS], dtype=tl.float32)

                dD = tl.zeros([T], dtype=tl.float32)

                for o in tl.range(O, loop_unroll_factor=1):
                    c_base_o = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                                + o * s_co + offs_t[:, None] * s_ct)
                    C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn
                                 ).to(tl.float32)
                    C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn
                                 ).to(tl.float32)
                    if HAS_GAMMA:
                        ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O
                                      + o).to(tl.float32)
                        ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O
                                      + o).to(tl.float32)
                        if BIAS_PRE_ROT:
                            C1 = C1 + ga1[None, :]
                            C2 = C2 + ga2[None, :]
                    if HAS_ROT:
                        Ct1 = ph_cos * C1 + ph_sin * C2
                        Ct2 = ph_cos * C2 - ph_sin * C1
                    else:
                        Ct1 = C1
                        Ct2 = C2
                    if HAS_THETA_H:
                        Ch1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                        Ch2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                    else:
                        Ch1 = Ct1
                        Ch2 = Ct2
                    Cr1 = Ch1
                    Cr2 = Ch2
                    if HAS_GAMMA and not BIAS_PRE_ROT:
                        if HAS_THETA_H:
                            gr1 = ch_h * ga1 + sh_h * ga2
                            gr2 = ch_h * ga2 - sh_h * ga1
                        else:
                            gr1 = ga1
                            gr2 = ga2
                        Cr1 = Cr1 + gr1[None, :]
                        Cr2 = Cr2 + gr2[None, :]

                    dY_o = tl.load(
                        dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo
                        + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                    ).to(tl.float32)
                    dCs1 = tl.zeros([T, PS], dtype=tl.float32)
                    dCs2 = tl.zeros([T, PS], dtype=tl.float32)

                    for r in tl.range(R, loop_unroll_factor=1):
                        b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                    + r * s_br + offs_t[:, None] * s_bt)
                        B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn
                                     ).to(tl.float32)
                        B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn
                                     ).to(tl.float32)
                        if HAS_BETA:
                            be1 = tl.load(Beta_ptr + h * N * R
                                          + offs_h1 * R + r).to(tl.float32)
                            be2 = tl.load(Beta_ptr + h * N * R
                                          + offs_h2 * R + r).to(tl.float32)
                            if BIAS_PRE_ROT:
                                B1 = B1 + be1[None, :]
                                B2 = B2 + be2[None, :]
                        if HAS_ROT:
                            Bt1 = ph_cos * B1 + ph_sin * B2
                            Bt2 = ph_cos * B2 - ph_sin * B1
                        else:
                            Bt1 = B1
                            Bt2 = B2
                        if HAS_BETA and not BIAS_PRE_ROT:
                            Bf1 = Bt1 + be1[None, :]
                            Bf2 = Bt2 + be2[None, :]
                        else:
                            Bf1 = Bt1
                            Bf2 = Bt2
                        x_r = tl.load(
                            X_ptr + b * s_xb + h * s_xh + r * s_xr
                            + c * T * s_xl + offs_gt[None, :] * s_xg
                            + offs_t[:, None] * s_xl,
                        ).to(tl.float32)
                        if HAS_DW:
                            x_r = x_r * dw[:, None]
                        dScore_or = pdot(dY_o.to(MMA_DT),
                                           tl.trans(x_r).to(MMA_DT),
                                           input_precision=PREC)
                        dSb_dt = (dScore_or * tl.exp(
                            tl.where((causal) & _same_seg(seg_h),
                                     a_val * (D_h[:, None] - D_h[None, :]),
                                     neg_inf))).to(MMA_DT)
                        dCs1 += pdot(dSb_dt, Bf1.to(MMA_DT),
                                       input_precision=PREC)
                        dCs2 += pdot(dSb_dt, Bf2.to(MMA_DT),
                                       input_precision=PREC)
                        dSb_tr = tl.trans(dSb_dt)
                        gB1 = pdot(dSb_tr, Cr1.to(MMA_DT),
                                     input_precision=PREC)
                        gB2 = pdot(dSb_tr, Cr2.to(MMA_DT),
                                     input_precision=PREC)
                        if HAS_BETA and not BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R
                                          + offs_h1 * R + r,
                                          tl.sum(gB1, axis=0), sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + h * N * R
                                          + offs_h2 * R + r,
                                          tl.sum(gB2, axis=0), sem=ACC_SEM)
                        if HAS_ROT:
                            dth_acc += (gB1 * _opv(Bt2, MMA_DT, MMA_IO, PREC) - gB2 * _opv(Bt1, MMA_DT, MMA_IO, PREC))
                            gn1 = ph_cos * gB1 - ph_sin * gB2
                            gB2 = ph_cos * gB2 + ph_sin * gB1
                            gB1 = gn1
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R
                                          + offs_h1 * R + r,
                                          tl.sum(gB1, axis=0), sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + h * N * R
                                          + offs_h2 * R + r,
                                          tl.sum(gB2, axis=0), sem=ACC_SEM)
                        db_base_r = (dBtil_ptr + slot_b + bg * s_dbh
                                     + b * s_dbb + c * s_dbc
                                     + offs_t[:, None] * s_dbt + r * s_dbr)
                        tl.atomic_add(db_base_r + offs_h1[None, :] * s_dbn,
                                      gB1, sem=ACC_SEM)
                        tl.atomic_add(db_base_r + offs_h2[None, :] * s_dbn,
                                      gB2, sem=ACC_SEM)

                    s_base = (SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                              + offs_gt[None, :] * s_sg)
                    S1 = tl.load(s_base + offs_h1[:, None] * s_sn
                                 ).to(tl.float32)
                    S2 = tl.load(s_base + offs_h2[:, None] * s_sn
                                 ).to(tl.float32)
                    dY_dt = dY_o.to(MMA_DT)
                    dCst1 = f[:, None] * pdot(
                        dY_dt, tl.trans(S1).to(MMA_DT), input_precision=PREC)
                    dCst2 = f[:, None] * pdot(
                        dY_dt, tl.trans(S2).to(MMA_DT), input_precision=PREC)
                    # SPANDELTA: Kb1 owns only the READ term of dL/ddelta
                    dD += a_val * (tl.sum(Cr1 * dCst1, axis=1)
                                   + tl.sum(Cr2 * dCst2, axis=1))
                    dC1 = dCs1 + dCst1
                    dC2 = dCs2 + dCst2

                    gC1 = dC1
                    gC2 = dC2
                    if HAS_GAMMA and not BIAS_PRE_ROT:
                        sg1 = tl.sum(gC1, axis=0)
                        sg2 = tl.sum(gC2, axis=0)
                        if HAS_THETA_H:
                            if NEED_DTHH:
                                tl.atomic_add(dThetaH_ptr + h * KH + offs_p,
                                              sg1 * gr2 - sg2 * gr1,
                                              sem=ACC_SEM)
                            sn1 = ch_h * sg1 - sh_h * sg2
                            sg2 = ch_h * sg2 + sh_h * sg1
                            sg1 = sn1
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_h1 * O
                                      + o, sg1, sem=ACC_SEM)
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_h2 * O
                                      + o, sg2, sem=ACC_SEM)
                    if HAS_THETA_H:
                        if NEED_DTHH:
                            tl.atomic_add(
                                dThetaH_ptr + h * KH + offs_p,
                                tl.sum(gC1 * _opv(Ch2, MMA_DT, MMA_IO, PREC)
                                       - gC2 * _opv(Ch1, MMA_DT, MMA_IO, PREC), axis=0),
                                sem=ACC_SEM)
                        gn1 = ch_h[None, :] * gC1 - sh_h[None, :] * gC2
                        gC2 = ch_h[None, :] * gC2 + sh_h[None, :] * gC1
                        gC1 = gn1
                    if HAS_ROT:
                        if HAS_THETA_H:
                            Cq1, Cq2 = _ph_read_consistent(
                                Ch1, Ch2, ch_h, sh_h, Ct1, Ct2, MMA_DT, MMA_IO,
                                HAS_THETA_H, PREC)                  # THETAFREQ
                        else:      # no theta_h: ch_h / sh_h / Ch* do not exist in this specialisation
                            Cq1 = _opv(Ct1, MMA_DT, MMA_IO, PREC)
                            Cq2 = _opv(Ct2, MMA_DT, MMA_IO, PREC)
                        dth_acc += gC1 * Cq2 - gC2 * Cq1
                        gn1 = ph_cos * gC1 - ph_sin * gC2
                        gC2 = ph_cos * gC2 + ph_sin * gC1
                        gC1 = gn1
                    if HAS_GAMMA and BIAS_PRE_ROT:
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_h1 * O
                                      + o, tl.sum(gC1, axis=0), sem=ACC_SEM)
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_h2 * O
                                      + o, tl.sum(gC2, axis=0), sem=ACC_SEM)
                    dc_base_o = (dCtil_ptr + slot_c + cg * s_dch
                                 + b * s_dcb + c * s_dcc + o * s_dco
                                 + offs_t[:, None] * s_dct)
                    tl.atomic_add(dc_base_o + offs_h1[None, :] * s_dcn, gC1, sem=ACC_SEM)
                    tl.atomic_add(dc_base_o + offs_h2[None, :] * s_dcn, gC2, sem=ACC_SEM)

                if HAS_ROT:
                    dth_ph = dth_acc
                    if PHASE_CUMSUM and not PHASE_DEFER:
                        dth_off = tl.sum(dth_ph, axis=0)
                        dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        tl.atomic_add(
                            dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                            + offs_p * s_dok, dth_off, sem=ACC_SEM)
                    tl.atomic_add(
                        dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                        + offs_t[:, None] * s_dtt + offs_p[None, :] * s_dtk,
                        dth_ph, sem=ACC_SEM)

                # SPANDELTA: the read term reaches every s <= i
                tl.atomic_add(
                    dDelta_ptr + b * s_db + h * s_dh + c * s_dc
                    + offs_t * s_dt, tl.cumsum(dD, axis=0, reverse=True),
                    sem=ACC_SEM)
    elif HBLK > 0:
        # ══ OWNER-ACCUMULATE Kb1 at R, O > 1 ═══════════════════════════
        # `_bwd_kbc_hb_kernel`'s construction, generalised to R, O > 1.
        # The CTA owns a HEAD BLOCK of HBLK heads instead of one head:
        # dBtil[..., r] (R tiles) and dCtil[..., o] (O tiles) are summed
        # in REGISTERS across the block -- and, for dB, across the o loop
        # too, since every map between `dB_r_part` and the buffer is
        # linear -- and each tile leaves the CTA with ONE write: a plain
        # `tl.store` when the CTA is the sole owner of the tile's whole
        # fan-in (DB_STORE / DC_STORE, decided on the host), an
        # `atomic_add` otherwise, which still cuts the fan-in by HBLK.
        # Everything that is PER HEAD (dDelta, dTheta/dThOff, dBeta,
        # dGamma, dThetaH, dRotBias) keeps its existing per-head atomic,
        # taken under the head-validity mask.
        b = tl.program_id(0)
        hblk = tl.program_id(1)
        NS: tl.constexpr = N // NSPLIT
        pid2 = tl.program_id(2)
        ns_idx = pid2 % NSPLIT
        rest2 = pid2 // NSPLIT
        if CHUNK_PARALLEL:
            c = rest2 // NGT_val
            gt_idx = rest2 % NGT_val
        else:
            gt_idx = rest2
        if IDX64:
            b = b.to(tl.int64)
        offs_t = tl.arange(0, T)
        offs_n = ns_idx * NS + tl.arange(0, NS)
        offs_k = ns_idx * (NS // 2) + tl.arange(0, NS // 2)
        offs_gt = gt_idx * GT + tl.arange(0, GT)
        # The score diagonal rides the MMAs (weight w_t); under the
        # trapezoid it is STRICT (`trap_adjoint` carries own_t)
        if TRAP_DIAG:
            causal = offs_t[:, None] > offs_t[None, :]
        else:
            causal = offs_t[:, None] >= offs_t[None, :]
        neg_inf = float("-inf")
        pair_idx_kb1 = offs_n // 2
        fwd_sign_kb1 = tl.where((offs_n % 2) == 0, 1.0, -1.0)
        # SSD_SPEED constexprs of this body (defined once: a constexpr
        # cannot be rebound inside the unrolled r / o loops)
        # (THH_ACC: the o-summed dThetaH register accumulator; slower on the
        # default chain, so only the opt-in ROT_COMB chain uses it.)
        THH_ACC: tl.constexpr = ROT_COMB and NEED_DTHH
        if KB2_FOLD:
            # Kb2's chunk-boundary terms land on the LAST token only, and
            # its `dalpha` term is per (b, h, c, gt) -- NOT per r and not
            # per N-slice, so only the ns_idx == 0 CTA may add it (the
            # same accounting `_bwd_kb2_kernel`'s R_TILE gate does for r).
            is_last_kb1 = tl.where(offs_t == T - 1, 1.0, 0.0)
            if NSPLIT == 1:
                da_gate_kb1 = 1.0
            else:
                da_gate_kb1 = tl.where(ns_idx == 0, 1.0, 0.0)
        h_start = hblk * HBLK
        if IDX64:
            h_start = h_start.to(tl.int64)
        # The host only ever blocks heads INSIDE one B/C group
        # (`_kb1_hb`), so the group indices are the block's, not the
        # head's, and the accumulated tile has one address.
        bg = h_start // HPG_B
        cg = h_start // HPG_C

        nc_end = 1 if CHUNK_PARALLEL else NC
        for ci in range(nc_end):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci

            dB_l = ()
            for r in tl.static_range(R):
                dB_l = dB_l + (tl.zeros([T, NS], dtype=tl.float32),)
            dC_l = ()
            for o in tl.static_range(O):
                dC_l = dC_l + (tl.zeros([T, NS], dtype=tl.float32),)

            for hh in range(HBLK):
                h = h_start + hh
                # A block may overhang H (HBLK need not divide H); the
                # overhanging lanes read head H-1 and contribute NOTHING.
                hv = h < H
                h = tl.minimum(h, H - 1)
                a_val = -tl.exp(tl.load(A_ptr + h))
                if HAS_THETA_H:
                    th_full_kb1 = tl.load(
                        ThetaH_ptr + h * (N // 2) + pair_idx_kb1)
                    ch_full_kb1 = _cos_approx(th_full_kb1)
                    fwd_coeff_kb1 = fwd_sign_kb1 * _sin_approx(th_full_kb1)
                if HAS_ROT_BIAS:
                    rb_full_kb1 = tl.load(
                        RotBias_ptr + h * (N // 2) + pair_idx_kb1)
                    crb_kb1 = _cos_approx(rb_full_kb1)
                    srb_fwd_kb1 = fwd_sign_kb1 * _sin_approx(rb_full_kb1)

                dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
                D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
                seg_end = tl.max(seg_h, axis=0)
                f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
                dec_mask = tl.exp(
                    tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                             neg_inf))
                if HAS_DW:
                    dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                                 + (c * T + offs_t) * s_wl)
                if KB2_FOLD:
                    # ── KB2_FOLD: Kb2's AGGREGATE term, here ──
                    # `dG` is the only operand this body did not already
                    # hold; `dec_agg` is one extra [T] exp off the same
                    # D_i tile the decay mask is built from.
                    D_end_kb1 = _dcs_end(dbase, offs_t, s_dt, D_h, T,
                                         DCS_INK)
                    dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_kb1 - D_h)), 0.0)
                    dG_agg = tl.load(
                        dG_ptr + b * s_gb + h * s_gh + c * s_gc
                        + offs_n[:, None] * s_gn
                        + offs_gt[None, :] * s_gg,
                    ).to(MMA_DT)                                # [NS, GT]
                    dend_kb2 = tl.zeros([T], dtype=tl.float32)
                if HAS_ROT:
                    th_rot = tl.load(
                        Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                        + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk,
                    ).to(tl.float32)                            # [T, NS//2]
                    if PHASE_CUMSUM:
                        th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                            ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                            + offs_k * s_ok,
                        ).to(tl.float32)[None, :]
                    cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, NS)
                    sn_rot = fwd_sign_kb1[None, :] * _pair_bcast_2d(
                        _sin_approx(th_rot), T, NS)
                    dth_acc = tl.zeros([T, NS], dtype=tl.float32)

                dD = tl.zeros([T], dtype=tl.float32)

                # ── the B/x tiles are o-INVARIANT, so they are built
                # once per head instead of once per (o, r), and the o
                # contributions to dBtil[..., r] are summed in registers.
                B_til_l = ()
                Bc_l = ()
                Bx_l = ()
                x_l = ()
                dBp = ()
                for r in tl.static_range(R):
                    b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + r * s_br)
                    Bc_r = tl.load(
                        b_base_r + offs_t[:, None] * s_bt
                        + offs_n[None, :] * s_bn,
                    ).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r
                        ).to(tl.float32)[None, :]
                    # ROTBWD: the XOR-lane partner of B~ is
                    # ALGEBRAIC, not a second shuffle.  With
                    #   B~ = cs B + sn swap(B),  swap(cs) = cs,
                    #   swap(sn) = -sn  (cs is pair-broadcast, sn carries
                    #   the pair sign), so
                    #   swap(B~) = cs swap(B) - sn B
                    # from the SAME two tiles the build already has live.
                    # `_pair_swap_2d` is a cross-lane layout op ([T, N]
                    # fp32 through shared memory); the phase gradient used
                    # to pay one per (head, r) for a value that is two
                    # FMAs away.  Bit-identical to the swapped form (a
                    # permutation commutes with the elementwise FMA).
                    if HAS_ROT and not ABL_PREROT:
                        B_sw = _pair_swap_2d(Bc_r, T, NS)
                        B_til = cs_rot * Bc_r + sn_rot * B_sw
                        B_tilx = cs_rot * B_sw - sn_rot * Bc_r
                    else:
                        B_til = Bc_r
                        if HAS_ROT:
                            B_tilx = _pair_swap_2d(Bc_r, T, NS)
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            B_unrot_xor = _pair_swap_2d(B_til, T, NS)
                        else:
                            B_unrot_xor = tl.load(
                                b_base_r + offs_t[:, None] * s_bt
                                + (offs_n ^ 1)[None, :] * s_bn,
                            ).to(tl.float32)
                        Bx_l = Bx_l + (B_unrot_xor,)
                        Bc_r = (crb_kb1[None, :] * B_til
                                + srb_fwd_kb1[None, :] * B_unrot_xor)
                    else:
                        Bc_r = B_til
                    if HAS_BETA and not BIAS_PRE_ROT:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r
                        ).to(tl.float32)[None, :]
                    x_r = tl.load(
                        X_ptr + b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                        + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                    ).to(tl.float32)
                    if HAS_DW:
                        x_r = x_r * dw[:, None]
                    # the tuple carries whichever of the two the body
                    # below still reads: B~ for the rot_bias adjoint,
                    # otherwise ONLY its XOR partner (the dTheta factor).
                    if HAS_ROT and not HAS_ROT_BIAS:
                        B_til_l = B_til_l + (B_tilx,)
                    else:
                        B_til_l = B_til_l + (B_til,)
                    Bc_l = Bc_l + (Bc_r,)
                    # the score dot is x's ONLY consumer, so the operand
                    # cast rides with the hoist: one value, rounded once.
                    x_l = x_l + (tl.trans(x_r).to(MMA_DT),)
                    if KB2_FOLD:
                        # ── KB2_FOLD ──────────────────────────────────
                        # `dBp[r]` is SEEDED with Kb2's aggregate tile
                        # instead of with zeros.  Everything downstream
                        # of it is linear in dB, so the o loop, the
                        # write-side chain, the dBeta row sum and the
                        # single dBtil write all see the TOTAL and need
                        # no second pass -- which is the whole point.
                        #
                        # It also makes Kb2's own `dDelta` terms free.
                        # Kb2 forms q_r[t] = sum_gt (B~_r dG)[t,gt]
                        # x_r[t,gt] with a SECOND GEMM and subtracts
                        # a dec_agg q_r; but
                        #   sum_n B~_r[t,n] dB_agg_r[t,n]
                        #     = sum_gt xw_r[t,gt] (B~_r dG)[t,gt]
                        #     = dec_agg[t] q_r[t],
                        # and `dD -= a_val * sum(Bc_l[r] * dB_r_part, 1)`
                        # below already reduces exactly that product on
                        # the seeded tile.  So Kb2's first dDelta term
                        # falls out of a line this kernel already runs,
                        # its second (the chunk-end correction) is the
                        # t-sum of the same [T] vector, and the second
                        # GEMM disappears: 2 GEMMs/chunk -> 1.
                        xw_r = (x_r * dec_agg[:, None]).to(MMA_DT)
                        dB_agg_r = pdot(xw_r, tl.trans(dG_agg),
                                          input_precision=PREC)
                        dend_kb2 += tl.sum(Bc_r * dB_agg_r, axis=1)
                        dBp = dBp + (dB_agg_r,)
                    else:
                        dBp = dBp + (tl.zeros([T, NS], dtype=tl.float32),)

                S_c = tl.load(
                    SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                    + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg,
                ).to(tl.float32)

                dCp = ()
                # SSD_SPEED: the per-head dThetaH partial of the read side,
                # summed over o in registers -- ONE pair-reduced atomic per
                # head below instead of O [NS]-wide atomics whose lane pairs
                # (2k, 2k+1) collide on the same word.
                if THH_ACC:
                    dthh_c = tl.zeros([NS], dtype=tl.float32)
                for o in tl.static_range(O):
                    c_base_o = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                                + o * s_co)
                    Co = tl.load(
                        c_base_o + offs_t[:, None] * s_ct
                        + offs_n[None, :] * s_cn,
                    ).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_GAMMA:
                        Co = Co + tl.load(
                            Gamma_ptr + h * N * O + offs_n * O + o
                        ).to(tl.float32)[None, :]
                    # ROTBWD: as on the B side -- swap(C~) = cs swap(C)
                    # - sn C.  It is BOTH the theta_h operand (`Co_xor`)
                    # and the dTheta factor, so no shuffle is issued per (head, o).
                    if ROT_COMB:
                        # ONE rotation by theta_t + theta_h (see ROT_COMB)
                        C_sw = _pair_swap_2d(Co, T, NS)
                        cph = (cs_rot * ch_full_kb1[None, :]
                               - sn_rot * fwd_coeff_kb1[None, :])
                        sph = (sn_rot * ch_full_kb1[None, :]
                               + cs_rot * fwd_coeff_kb1[None, :])
                        C_h = cph * Co + sph * C_sw
                        C_hx = cph * C_sw - sph * Co            # swap(C_h)
                    else:
                        if HAS_ROT and not ABL_PREROT:
                            C_sw = _pair_swap_2d(Co, T, NS)
                            C_til = cs_rot * Co + sn_rot * C_sw
                            C_tilx = cs_rot * C_sw - sn_rot * Co
                        else:
                            C_til = Co
                            if HAS_ROT:
                                C_tilx = _pair_swap_2d(Co, T, NS)
                        if HAS_THETA_H:
                            if HAS_ROT:
                                Co_xor = C_tilx
                            else:
                                Co_xor = tl.load(
                                    c_base_o + offs_t[:, None] * s_ct
                                    + (offs_n ^ 1)[None, :] * s_cn,
                                ).to(tl.float32)
                            C_h = (ch_full_kb1[None, :] * C_til
                                   + fwd_coeff_kb1[None, :] * Co_xor)
                        else:
                            C_h = C_til
                    C_o_rd = C_h
                    if HAS_GAMMA and not BIAS_PRE_ROT:
                        gam = tl.load(
                            Gamma_ptr + h * N * O + offs_n * O + o
                        ).to(tl.float32)
                        if HAS_THETA_H:
                            gam_xor = tl.load(
                                Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o
                            ).to(tl.float32)
                            gam = (ch_full_kb1 * gam
                                   + fwd_coeff_kb1 * gam_xor)
                        C_o_rd = C_o_rd + gam[None, :]

                    dY_o = tl.load(
                        dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo
                        + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                    ).to(tl.float32)

                    dC_score_o = tl.zeros([T, NS], dtype=tl.float32)
                    dY_o_dt = dY_o.to(MMA_DT)
                    C_o_dt = C_o_rd.to(MMA_DT)
                    dBp_next = ()
                    for r in tl.static_range(R):
                        dScore_or = pdot(dY_o_dt, x_l[r],
                                           input_precision=PREC)
                        # one cast here, transposed below.
                        dSb_or_dt = (dScore_or * dec_mask).to(MMA_DT)
                        dC_score_o += pdot(dSb_or_dt, Bc_l[r].to(MMA_DT),
                                             input_precision=PREC)
                        dB_or = pdot(tl.trans(dSb_or_dt), C_o_dt,
                                       input_precision=PREC)
                        dBp_next = dBp_next + (dBp[r] + dB_or,)
                    dBp = dBp_next

                    dC_state_o = f[:, None] * pdot(
                        dY_o_dt, tl.trans(S_c).to(MMA_DT),
                        input_precision=PREC)
                    dC_o_rd = dC_score_o + dC_state_o
                    # SPANDELTA: Kb1 owns only the READ term of dL/ddelta
                    dD += a_val * tl.sum(C_o_rd * dC_state_o, axis=1)

                    if HAS_GAMMA and not HAS_ROT:
                        tl.atomic_add(
                            dGamma_ptr + h * N * O + offs_n * O + o,
                            tl.sum(dC_o_rd, axis=0), mask=hv, sem=ACC_SEM)

                    # ── the read-side adjoint chain ──────────────────
                    if ROT_COMB:
                        gC_ = dC_o_rd
                        prod_c = gC_ * _opv(C_hx, MMA_DT, MMA_IO, PREC)  # <gC, J C_h>
                        if NEED_DTHH:
                            dthh_c += tl.sum(fwd_sign_kb1[None, :] * prod_c,
                                             axis=0)
                        dth_acc += prod_c
                        cph = (cs_rot * ch_full_kb1[None, :]
                               - sn_rot * fwd_coeff_kb1[None, :])
                        sph = (sn_rot * ch_full_kb1[None, :]
                               + cs_rot * fwd_coeff_kb1[None, :])
                        gC_ = cph * gC_ - sph * _pair_swap_2d(gC_, T, NS)
                        if HAS_GAMMA:
                            tl.atomic_add(
                                dGamma_ptr + h * N * O + offs_n * O + o,
                                tl.sum(gC_, axis=0), mask=hv, sem=ACC_SEM)
                        dCp = dCp + (gC_,)
                    elif HAS_ROT:
                        gC_ = dC_o_rd
                        if HAS_GAMMA and not BIAS_PRE_ROT:
                            sg = tl.sum(gC_, axis=0)             # [NS]
                            if HAS_THETA_H:
                                gam_l = tl.load(
                                    Gamma_ptr + h * N * O + offs_n * O + o
                                ).to(tl.float32)
                                gam_x = tl.load(
                                    Gamma_ptr + h * N * O
                                    + (offs_n ^ 1) * O + o
                                ).to(tl.float32)
                                gam_rot = (ch_full_kb1 * gam_l
                                           + fwd_coeff_kb1 * gam_x)
                                if NEED_DTHH:
                                    tl.atomic_add(
                                        dThetaH_ptr + h * (N // 2)
                                        + pair_idx_kb1,
                                        fwd_sign_kb1 * sg
                                        * _pair_swap_1d(gam_rot, NS),
                                        mask=hv, sem=ACC_SEM)
                                sg = (ch_full_kb1 * sg
                                      - fwd_coeff_kb1
                                      * _pair_swap_1d(sg, NS))
                            tl.atomic_add(
                                dGamma_ptr + h * N * O + offs_n * O + o,
                                sg, mask=hv, sem=ACC_SEM)
                        if HAS_THETA_H:
                            if NEED_DTHH:
                                tl.atomic_add(
                                    dThetaH_ptr + h * (N // 2)
                                    + pair_idx_kb1,
                                    tl.sum(fwd_sign_kb1[None, :] * gC_
                                           # ROTBWD: swap(C_h) =
                                           # ch swap(C~) - fwd_coeff C~
                                           # (ch pair-broadcast,
                                           # fwd_coeff pair-odd).
                                           * _opv(ch_full_kb1[None, :] * C_tilx
                                                  - fwd_coeff_kb1[None, :]
                                                  * C_til, MMA_DT, MMA_IO, PREC),
                                           axis=0),
                                    mask=hv, sem=ACC_SEM)
                            gC_ = (ch_full_kb1[None, :] * gC_
                                   - fwd_coeff_kb1[None, :]
                                   * _pair_swap_2d(gC_, T, NS))
                        if not ABL_NO_DTH:
                            if (MMA_IO or PREC == "tf32") and DTH_OPROUND and HAS_THETA_H:
                                # THETAFREQ: the MMAs consumed round(C_h),
                                # C_h = R(nu) C~; the consistent factor is
                                # J R(nu)^T round(C_h) = ch round(swap C_h)
                                # + fc round(C_h) (fc pair-odd), FMAs only
                                C_hr = _opv(C_h, MMA_DT, MMA_IO, PREC)
                                C_hxr = _opv(ch_full_kb1[None, :] * C_tilx
                                             - fwd_coeff_kb1[None, :] * C_til,
                                             MMA_DT, MMA_IO, PREC)
                                dth_acc += gC_ * (ch_full_kb1[None, :] * C_hxr
                                                  + fwd_coeff_kb1[None, :]
                                                  * C_hr)
                            else:
                                dth_acc += gC_ * _opv(C_tilx, MMA_DT, MMA_IO, PREC)
                        gC_ = cs_rot * gC_ - sn_rot * _pair_swap_2d(
                            gC_, T, NS)
                        if HAS_GAMMA and BIAS_PRE_ROT:
                            tl.atomic_add(
                                dGamma_ptr + h * N * O + offs_n * O + o,
                                tl.sum(gC_, axis=0), mask=hv, sem=ACC_SEM)
                        dCp = dCp + (gC_,)
                    elif HAS_THETA_H:
                        # the two-write un-rotation of the generic
                        # theta_h path, as ONE register expression: the
                        # lane-n total is ch[n] dC[n] + fwd[n^1] dC[n^1].
                        if NEED_DTHH:
                            Co2_xor_o = tl.load(
                                c_base_o + offs_t[:, None] * s_ct
                                + (offs_n ^ 1)[None, :] * s_cn,
                            ).to(tl.float32)
                            Co2_o = tl.load(
                                c_base_o + offs_t[:, None] * s_ct
                                + offs_n[None, :] * s_cn,
                            ).to(tl.float32)
                            Crd_p_o = (ch_full_kb1[None, :] * Co2_xor_o
                                       - fwd_coeff_kb1[None, :] * Co2_o)
                            tl.atomic_add(
                                dThetaH_ptr + h * (N // 2) + pair_idx_kb1,
                                tl.sum(fwd_sign_kb1[None, :] * dC_o_rd
                                       * Crd_p_o, axis=0), mask=hv,
                                sem=ACC_SEM)
                        dCp = dCp + (
                            ch_full_kb1[None, :] * dC_o_rd
                            + _pair_swap_2d(
                                fwd_coeff_kb1[None, :] * dC_o_rd, T, NS),)
                    else:
                        dCp = dCp + (dC_o_rd,)

                if THH_ACC:
                    # pair lanes (2k, 2k+1) -> pair k in registers, so the
                    # atomic is one affine [NS/2] vector with no duplicate
                    # addresses (the per-o form issued O [NS]-wide atomics
                    # whose lane pairs collided on the same word)
                    thh_e, thh_o = tl.split(tl.reshape(dthh_c, (NS // 2, 2)))
                    tl.atomic_add(dThetaH_ptr + h * (N // 2) + offs_k,
                                  thh_e + thh_o, mask=hv, sem=ACC_SEM)
                # ── the write-side adjoint chain, once per r on the
                # o-folded total (every map in it is LINEAR in dB).
                dBw = ()
                for r in tl.static_range(R):
                    dB_r_part = dBp[r]
                    B_til = B_til_l[r]
                    if HAS_ROT:
                        gB_ = dB_r_part
                        if HAS_BETA and not BIAS_PRE_ROT:
                            tl.atomic_add(
                                dBeta_ptr + h * N * R + offs_n * R + r,
                                tl.sum(gB_, axis=0), mask=hv, sem=ACC_SEM)
                        if HAS_ROT_BIAS:
                            Brd_p = (crb_kb1[None, :] * Bx_l[r]
                                     - srb_fwd_kb1[None, :] * B_til)
                            tl.atomic_add(
                                dRotBias_ptr + h * (N // 2) + pair_idx_kb1,
                                tl.sum(fwd_sign_kb1[None, :] * gB_ * Brd_p,
                                       axis=0), mask=hv, sem=ACC_SEM)
                            gB_ = (crb_kb1[None, :] * gB_
                                   - srb_fwd_kb1[None, :]
                                   * _pair_swap_2d(gB_, T, NS))
                        if not ABL_NO_DTH:
                            # B_til_l already holds swap(B~) here (the
                            # rot_bias branch below is the only reader of
                            # B~ itself, and it keeps the old tuple).
                            # THETAFREQ: the partner of the tile the MMAs
                            # consumed (rounded under MMA_IO), see `_opv`
                            dth_acc += gB_ * _opv(
                                _pair_swap_2d(B_til, T, NS)
                                if HAS_ROT_BIAS else B_til, MMA_DT, MMA_IO, PREC)
                        gB_ = cs_rot * gB_ - sn_rot * _pair_swap_2d(
                            gB_, T, NS)
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(
                                dBeta_ptr + h * N * R + offs_n * R + r,
                                tl.sum(gB_, axis=0), mask=hv, sem=ACC_SEM)
                        dBw = dBw + (gB_,)
                    elif HAS_ROT_BIAS:
                        Brd_p = (crb_kb1[None, :] * Bx_l[r]
                                 - srb_fwd_kb1[None, :] * B_til)
                        tl.atomic_add(
                            dRotBias_ptr + h * (N // 2) + pair_idx_kb1,
                            tl.sum(fwd_sign_kb1[None, :] * dB_r_part
                                   * Brd_p, axis=0), mask=hv, sem=ACC_SEM)
                        dBw = dBw + (
                            crb_kb1[None, :] * dB_r_part
                            + _pair_swap_2d(
                                srb_fwd_kb1[None, :] * dB_r_part, T, NS),)
                    else:
                        dBw = dBw + (dB_r_part,)
                    if HAS_BETA and not HAS_ROT:
                        tl.atomic_add(
                            dBeta_ptr + h * N * R + offs_n * R + r,
                            tl.sum(dB_r_part, axis=0), mask=hv, sem=ACC_SEM)

                # ── the head's contribution joins the block total ──
                dB_next = ()
                for r in tl.static_range(R):
                    dB_next = dB_next + (
                        dB_l[r] + tl.where(hv, dBw[r], 0.0),)
                dB_l = dB_next
                dC_next = ()
                for o in tl.static_range(O):
                    dC_next = dC_next + (
                        dC_l[o] + tl.where(hv, dCp[o], 0.0),)
                dC_l = dC_next

                # ── per-head epilogue: dTheta and dDelta keep their own
                # atomics (Kb2 / Kc write the same buffers).
                if HAS_ROT and not ABL_NO_DTH:
                    dth_e, dth_o = tl.split(
                        tl.reshape(dth_acc, (T, NS // 2, 2)))
                    # ROTBWD: the pair sign is applied HERE (see the
                    # dth_acc sites above), not per term.
                    dth_ph = dth_e - dth_o                    # [T, NS//2]
                    if PHASE_CUMSUM and not PHASE_DEFER:
                        dth_off = tl.sum(dth_ph, axis=0)
                        dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        tl.atomic_add(
                            dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                            + offs_k * s_dok, dth_off, mask=hv, sem=ACC_SEM)
                    dth_p = (dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                             + offs_t[:, None] * s_dtt
                             + offs_k[None, :] * s_dtk)
                    if DTH_STORE:
                        tl.store(dth_p, dth_ph, mask=hv)
                    else:
                        tl.atomic_add(dth_p, dth_ph, mask=hv, sem=ACC_SEM)

                # SPANDELTA (`_SPANDELTA_NOTE`): the read term reaches
                # every s <= i (reverse cumsum); the folded Kb2 write term
                # every s > j (`_excl_prefix`, the pair (j, T-1)); the carry
                # term every s.  Per (b, h, c, gt) and NOT per N-slice for
                # the carry term, hence `da_gate_kb1`.
                dD = tl.cumsum(dD, axis=0, reverse=True)
                if KB2_FOLD:
                    dalpha_c = tl.load(
                        dAlpha_ptr + b * s_dab + h * s_dah
                        + gt_idx * s_dag + c * s_dac)
                    dD += a_val * (_excl_prefix(dend_kb2, offs_t)
                                   + tl.where(seg_end == 0, tl.exp(a_val * D_end_kb1), 0.0) * dalpha_c
                                   * da_gate_kb1)

                tl.atomic_add(
                    dDelta_ptr + b * s_db + h * s_dh + c * s_dc
                    + offs_t * s_dt, dD, mask=hv, sem=ACC_SEM)

            # ── ONE write per tile per CTA ─────────────────────────────
            for r in tl.static_range(R):
                db_p = (dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                        + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
                        + r * s_dbr)
                if DB_STORE:
                    tl.store(db_p, dB_l[r])
                else:
                    tl.atomic_add(db_p, dB_l[r], sem=ACC_SEM)
            for o in tl.static_range(O):
                dc_p = (dCtil_ptr + cg * s_dch + b * s_dcb + c * s_dcc
                        + o * s_dco + offs_t[:, None] * s_dct
                        + offs_n[None, :] * s_dcn)
                if DC_STORE:
                    tl.store(dc_p, dC_l[o])
                else:
                    tl.atomic_add(dc_p, dC_l[o], sem=ACC_SEM)
    else:
        b = tl.program_id(0)
        h = tl.program_id(1)
        NS: tl.constexpr = N // NSPLIT
        pid2 = tl.program_id(2)
        ns_idx = pid2 % NSPLIT
        rest2 = pid2 // NSPLIT
        if CHUNK_PARALLEL:
            c = rest2 // NGT_val
            gt_idx = rest2 % NGT_val
        else:
            gt_idx = rest2

        if IDX64:
            b = b.to(tl.int64)
            h = h.to(tl.int64)

        a_val = -tl.exp(tl.load(A_ptr + h))
        offs_t = tl.arange(0, T)
        offs_n = ns_idx * NS + tl.arange(0, NS)
        offs_gt = gt_idx * GT + tl.arange(0, GT)
        # The score diagonal rides the MMAs (weight w_t); under the
        # trapezoid it is STRICT (`trap_adjoint` carries own_t)
        if TRAP_DIAG:
            causal = offs_t[:, None] > offs_t[None, :]
        else:
            causal = offs_t[:, None] >= offs_t[None, :]
        neg_inf = float("-inf")
        bg = h // HPG_B
        cg = h // HPG_C

        slot_b = (h % SLOTS) * s_slb
        slot_c = (h % SLOTS) * s_slc

        pair_idx_kb1 = offs_n // 2
        fwd_sign_kb1 = tl.where((offs_n % 2) == 0, 1.0, -1.0)

        if HAS_THETA_H:
            th_full_kb1 = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx_kb1)
            ch_full_kb1 = _cos_approx(th_full_kb1)
            sh_full_kb1 = _sin_approx(th_full_kb1)
            fwd_coeff_kb1 = fwd_sign_kb1 * sh_full_kb1

        if HAS_ROT_BIAS:
            rb_full_kb1 = tl.load(RotBias_ptr + h * (N // 2) + pair_idx_kb1)
            crb_kb1 = _cos_approx(rb_full_kb1)
            srb_fwd_kb1 = fwd_sign_kb1 * _sin_approx(rb_full_kb1)

        nc_end = 1 if CHUNK_PARALLEL else NC
        for ci in range(nc_end):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci

            # Decay factors
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            dec_mask = tl.exp(
                tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                         neg_inf))

            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)

            # ── this chunk's phase, on its own PAIR axis ─────────────────
            # s_thh = 0 for a head-SHARED phase, theta.stride(3) for the
            # per-(kernel head) one.  `h` is a program id, so the base stays
            # affine and the tile is one wide vector load per row.
            if HAS_ROT:
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht
                    + (ns_idx * (NS // 2)
                       + tl.arange(0, NS // 2))[None, :] * s_thk,
                ).to(tl.float32)                                    # [T, N//2]
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + (ns_idx * (NS // 2)
                           + tl.arange(0, NS // 2)) * s_ok,
                    ).to(tl.float32)[None, :]
                cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, NS)
                sn_rot = fwd_sign_kb1[None, :] * _pair_bcast_2d(
                    _sin_approx(th_rot), T, NS)
                dth_acc = tl.zeros([T, NS], dtype=tl.float32)

            # `S_c` is loaded at its USE, below the r loop, not hoisted: hoisted it
            # is an [N, GT] fp32 tile live across every r iteration for a value
            # the r loop never reads (the O re-loads are L2-hot).  Pure code
            # motion.

            # Score-path with O×R loops
            dD = tl.zeros([T], dtype=tl.float32)

            for o in tl.range(O, loop_unroll_factor=1):
                # Load C_o and apply rotation
                c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                Co = tl.load(
                    c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn,
                ).to(tl.float32)
                if BIAS_PRE_ROT and HAS_GAMMA:
                    # Mamba-3's fold ORDER: gamma joins the RAW tile and
                    # rotates WITH it.
                    Co = Co + tl.load(
                        Gamma_ptr + h * N * O + offs_n * O + o
                    ).to(tl.float32)[None, :]
                if HAS_ROT:
                    C_til = cs_rot * Co + sn_rot * _pair_swap_2d(Co, T, NS)
                else:
                    C_til = Co
                if HAS_THETA_H:
                    if HAS_ROT:
                        Co_xor = _pair_swap_2d(C_til, T, NS)
                    else:
                        Co_xor = tl.load(
                            c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn,
                        ).to(tl.float32)
                    C_h = ch_full_kb1[None, :] * C_til + fwd_coeff_kb1[None, :] * Co_xor
                else:
                    C_h = C_til
                C_o_rd = C_h
                if HAS_GAMMA and not BIAS_PRE_ROT:
                    gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                    if HAS_THETA_H:
                        gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                        gam = ch_full_kb1 * gam + fwd_coeff_kb1 * gam_xor
                    C_o_rd = C_o_rd + gam[None, :]

                # dC_state_o = f * (dY_o @ S_c^T)
                dY_o = tl.load(
                    dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo
                    + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                ).to(tl.float32)
                # The state-path dot is taken AFTER the r loop: here it would keep
                # a whole [T, NS] accumulator live across every r iteration for a
                # value first used at `dC_o_rd` below.  Pure code motion (dY_o,
                # S_c and f are loop-invariant in r).
                dC_score_o = tl.zeros([T, NS], dtype=tl.float32)

                for r in tl.range(R, loop_unroll_factor=1):
                    Bc_r = tl.load(
                        B_ptr + b * s_bb + c * s_bc + bg * s_bh
                        + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                        + r * s_br,
                    ).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r
                        ).to(tl.float32)[None, :]
                    if HAS_ROT:
                        B_til = cs_rot * Bc_r + sn_rot * _pair_swap_2d(Bc_r, T, NS)
                    else:
                        B_til = Bc_r
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            # after the phase the tile is a REGISTER value, so
                            # the partner must come from it
                            Bc_r_xor_rb = _pair_swap_2d(B_til, T, NS)
                        else:
                            Bc_r_xor_rb = tl.load(
                                B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn
                                + r * s_br,
                            ).to(tl.float32)
                        B_unrot_r = B_til
                        B_unrot_xor = Bc_r_xor_rb
                        Bc_r = crb_kb1[None, :] * B_til + srb_fwd_kb1[None, :] * Bc_r_xor_rb
                    else:
                        Bc_r = B_til
                    if HAS_BETA and not BIAS_PRE_ROT:
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(
                        X_ptr + x_base_r
                        + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                    ).to(tl.float32)
                    if HAS_DW:
                        x_r = x_r * dw[:, None]

                    dScore_or = pdot(dY_o.to(MMA_DT), tl.trans(x_r).to(MMA_DT), input_precision=PREC)
                    dSb_or = dScore_or * dec_mask

                    # Accumulate dC_score_o
                    # dSb_or straight here, transposed below -- one cast.
                    dSb_or_dt = dSb_or.to(MMA_DT)
                    dC_score_o += pdot(dSb_or_dt, Bc_r.to(MMA_DT),
                                         input_precision=PREC)

                    # dB_r contribution from this o
                    dB_r_part = pdot(tl.trans(dSb_or_dt), C_o_rd.to(MMA_DT),
                                       input_precision=PREC)

                    # ── the write-side adjoint chain, outermost map first ──
                    # forward: B -> (+beta, pre) -> R(theta_t) -> R(rho)
                    #            -> (+beta, post).  Every atomic defined on a
                    # rotated frame is taken BEFORE the corresponding
                    # un-rotation, exactly as Kbc-hb does at R = O = 1.
                    if HAS_ROT:
                        gB_ = dB_r_part
                        if HAS_BETA and not BIAS_PRE_ROT:
                            tl.atomic_add(
                                dBeta_ptr + h * N * R + offs_n * R + r,
                                tl.sum(gB_, axis=0), sem=ACC_SEM)
                        if HAS_ROT_BIAS:
                            Brd_p = (crb_kb1[None, :] * B_unrot_xor
                                     - srb_fwd_kb1[None, :] * B_unrot_r)
                            tl.atomic_add(
                                dRotBias_ptr + h * (N // 2) + pair_idx_kb1,
                                tl.sum(fwd_sign_kb1[None, :] * gB_ * Brd_p,
                                       axis=0), sem=ACC_SEM)
                            gB_ = (crb_kb1[None, :] * gB_
                                   - srb_fwd_kb1[None, :] * _pair_swap_2d(gB_, T, NS))
                        # d/dtheta of (e' = c e + s o, o' = -s e + c o) is
                        # (o', -e'), i.e. the pair contribution is
                        # dB~_e B~_o - dB~_o B~_e; the pair_idx atomic below
                        # sums the two lanes.
                        dth_acc += fwd_sign_kb1[None, :] * gB_ * _pair_swap_2d(
                            _opv(B_til, MMA_DT, MMA_IO, PREC), T, NS)      # THETAFREQ
                        gB_ = cs_rot * gB_ - sn_rot * _pair_swap_2d(gB_, T, NS)
                        if HAS_BETA and BIAS_PRE_ROT:
                            # beta sits INSIDE the rotation, so its gradient
                            # is the sum of the UN-rotated dB
                            tl.atomic_add(
                                dBeta_ptr + h * N * R + offs_n * R + r,
                                tl.sum(gB_, axis=0), sem=ACC_SEM)
                        tl.atomic_add(
                            dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                            + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
                            + r * s_dbr,
                            gB_, sem=ACC_SEM)
                    elif HAS_ROT_BIAS:
                        Brd_p = crb_kb1[None, :] * B_unrot_xor - srb_fwd_kb1[None, :] * B_unrot_r
                        cross_rb = tl.sum(fwd_sign_kb1[None, :] * dB_r_part * Brd_p, axis=0)
                        tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb1, cross_rb, sem=ACC_SEM)

                        # Un-rotate dB: two-write pattern (transpose of rotation)
                        db_base_r = (dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                                     + offs_t[:, None] * s_dbt + r * s_dbr)
                        tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn,
                                      crb_kb1[None, :] * dB_r_part,
                                      sem=ACC_SEM)
                        tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn,
                                      srb_fwd_kb1[None, :] * dB_r_part,
                                      sem=ACC_SEM)
                    else:
                        tl.atomic_add(
                            dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                            + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
                            + r * s_dbr,
                            dB_r_part, sem=ACC_SEM)
                    if HAS_BETA and not HAS_ROT:
                        tl.atomic_add(
                            dBeta_ptr + h * N * R + offs_n * R + r,
                            tl.sum(dB_r_part, axis=0), sem=ACC_SEM)

                S_c = tl.load(
                    SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                    + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg,
                ).to(tl.float32)
                dC_state_o = f[:, None] * pdot(
                    dY_o.to(MMA_DT), tl.trans(S_c).to(MMA_DT), input_precision=PREC)
                dC_o_rd = dC_score_o + dC_state_o
                # SPANDELTA: Kb1 owns only the READ term of dL/ddelta
                dD += a_val * tl.sum(C_o_rd * dC_state_o, axis=1)

                if HAS_GAMMA and not DEFERRED_ROTATION and not HAS_ROT:
                    tl.atomic_add(
                        dGamma_ptr + h * N * O + offs_n * O + o,
                        tl.sum(dC_o_rd, axis=0), sem=ACC_SEM)

                # ── the read-side adjoint chain ──────────────────────────
                # forward: C -> (+gamma, pre) -> R(theta_t) = C~ ->
                #          R(theta_h) = C_h -> (+ R(theta_h) gamma, post).
                if HAS_ROT:
                    gC_ = dC_o_rd
                    if HAS_GAMMA and not BIAS_PRE_ROT:
                        sg = tl.sum(gC_, axis=0)                     # [N]
                        if HAS_THETA_H:
                            gam_l = tl.load(
                                Gamma_ptr + h * N * O + offs_n * O + o
                            ).to(tl.float32)
                            gam_x = tl.load(
                                Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o
                            ).to(tl.float32)
                            gam_rot = ch_full_kb1 * gam_l + fwd_coeff_kb1 * gam_x
                            if NEED_DTHH:
                                tl.atomic_add(
                                    dThetaH_ptr + h * (N // 2) + pair_idx_kb1,
                                    fwd_sign_kb1 * sg
                                    * _pair_swap_1d(gam_rot, NS), sem=ACC_SEM)
                            sg = (ch_full_kb1 * sg
                                  - fwd_coeff_kb1 * _pair_swap_1d(sg, NS))
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o, sg, sem=ACC_SEM)
                    if HAS_THETA_H:
                        if NEED_DTHH:
                            tl.atomic_add(
                                dThetaH_ptr + h * (N // 2) + pair_idx_kb1,
                                tl.sum(fwd_sign_kb1[None, :] * gC_
                                       * _pair_swap_2d(_opv(C_h, MMA_DT, MMA_IO, PREC),
                                                       T, NS), axis=0),
                                sem=ACC_SEM)
                        gC_ = (ch_full_kb1[None, :] * gC_
                               - fwd_coeff_kb1[None, :] * _pair_swap_2d(gC_, T, NS))
                    if (MMA_IO or PREC == "tf32") and DTH_OPROUND and HAS_THETA_H:
                        # THETAFREQ: C~' = R(nu)^T round(C_h), the value the
                        # MMAs consumed; swap(C~') = ch swap(R) + fc R
                        C_hr = _opv(C_h, MMA_DT, MMA_IO, PREC)
                        dth_acc += fwd_sign_kb1[None, :] * gC_ * (
                            ch_full_kb1[None, :] * _pair_swap_2d(C_hr, T, NS)
                            + fwd_coeff_kb1[None, :] * C_hr)
                    else:
                        dth_acc += fwd_sign_kb1[None, :] * gC_ * _pair_swap_2d(
                            _opv(C_til, MMA_DT, MMA_IO, PREC), T, NS)
                    gC_ = cs_rot * gC_ - sn_rot * _pair_swap_2d(gC_, T, NS)
                    if HAS_GAMMA and BIAS_PRE_ROT:
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o,
                                      tl.sum(gC_, axis=0), sem=ACC_SEM)
                    tl.atomic_add(
                        dCtil_ptr + slot_c + cg * s_dch + b * s_dcb + c * s_dcc
                        + o * s_dco
                        + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn,
                        gC_, sem=ACC_SEM)
                # Store dC_o_rd: deferred rotation or inline
                elif HAS_THETA_H and DEFERRED_ROTATION:
                    scratch = (Scratch_ptr + (b * H + h) * (NC * O * T * N)
                        + c * (O * T * N) + o * (T * N)
                        + offs_t[:, None] * N + offs_n[None, :])
                    if G_SIZE == GT:
                        tl.store(scratch, dC_o_rd)
                    else:
                        # each G tile has only a PARTIAL dC. The
                        # deferred buffer has no G axis, so it must reduce
                        # these partials before Kinv and the gamma reduction.
                        tl.atomic_add(scratch, dC_o_rd, sem=ACC_SEM)
                elif HAS_THETA_H:
                    if NEED_DTHH:
                        Co2_xor_o = tl.load(
                            c_base_o + offs_t[:, None] * s_ct
                            + (offs_n ^ 1)[None, :] * s_cn,
                        ).to(tl.float32)
                        Co2_o = tl.load(
                            c_base_o + offs_t[:, None] * s_ct
                            + offs_n[None, :] * s_cn,
                        ).to(tl.float32)
                        Crd_p_o = ch_full_kb1[None, :] * Co2_xor_o - fwd_coeff_kb1[None, :] * Co2_o
                        cross_o = tl.sum(fwd_sign_kb1[None, :] * dC_o_rd * Crd_p_o, axis=0)
                        tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx_kb1, cross_o, sem=ACC_SEM)
                    dc_base_o = (dCtil_ptr + slot_c + cg * s_dch + b * s_dcb + c * s_dcc
                                 + offs_t[:, None] * s_dct + o * s_dco)
                    tl.atomic_add(dc_base_o + offs_n[None, :] * s_dcn,
                                  ch_full_kb1[None, :] * dC_o_rd, sem=ACC_SEM)
                    tl.atomic_add(dc_base_o + (offs_n ^ 1)[None, :] * s_dcn,
                                  fwd_coeff_kb1[None, :] * dC_o_rd,
                                  sem=ACC_SEM)
                else:
                    tl.atomic_add(
                        dCtil_ptr + slot_c + cg * s_dch + b * s_dcb + c * s_dcc + o * s_dco
                        + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn,
                        dC_o_rd, sem=ACC_SEM)

            # ── this CTA's dTheta partial ───────────────────────────────
            # The fan-in is the G-tile axis AND the aggregate-path kernel
            # (Kb2, or Kc under FUSE_KB2), so this is always an atomic add
            # into a zero-initialised buffer -- never the sole-owner store
            # Kbc-hb can take.
            if HAS_ROT:
                dth_e, dth_o = tl.split(tl.reshape(dth_acc, (T, NS // 2, 2)))
                dth_ph = dth_e + dth_o                              # [T, N//2]
                if PHASE_CUMSUM and not PHASE_DEFER:
                    # theta_t = off_c + sum_{t' <= t} incr_t', so
                    #   dL/d incr_t = sum_{t' >= t} dL/d theta_t'  (in-chunk)
                    #   dL/d off_c  = sum_t dL/d theta_t
                    # both LINEAR, so a per-CTA partial still sums correctly.
                    dth_off = tl.sum(dth_ph, axis=0)                # [N//2]
                    dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                    tl.atomic_add(
                        dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                        + (ns_idx * (NS // 2)
                           + tl.arange(0, NS // 2)) * s_dok,
                        dth_off, sem=ACC_SEM)
                tl.atomic_add(
                    dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                    + offs_t[:, None] * s_dtt
                    + (ns_idx * (NS // 2)
                       + tl.arange(0, NS // 2))[None, :] * s_dtk,
                    dth_ph, sem=ACC_SEM)

            # Store partial dDelta -- SPANDELTA: the read term reaches
            # every s <= i
            tl.atomic_add(
                dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt,
                tl.cumsum(dD, axis=0, reverse=True), sem=ACC_SEM)



# ── Owner-accumulate score+aggregate backward (sm_90 atomic fix) ────────
# The dBtil/dCtil [T, N] tile atomics are the dominant backward cost on
# sm_90: every (b, c) tile is RMW-contended by H * NGT CTAs from Kb1 PLUS
# H * NGT from Kc's fused-Kb2 part; the fix is fan-in reduction, not
# slicing (DB_SLOTS is counterproductive there too).
# This kernel replaces Kb1 on the O = 1, R = 1, ungrouped, rotation-free,
# fuse_kb2 path: grid (B, ceil(H/HB), nc*NGT), each CTA loops over an
# HB-head block, accumulates dB/dC in registers, and issues ONE
# atomic_add per tile (a plain store when it is the sole owner:
# HB >= H and NGT == 1).  It also absorbs Kc's dB_agg product (Kc runs
# NO_DBAGG), so this kernel is the only dBtil writer.  The head-shared
# C/B tiles are loaded once per CTA instead of once per head.
# HB is autotuned (meta-dependent grid); SSD_MIMO_BWD_HB=0 restores Kb1.
# ── The Kbc-hb LIVE-RANGE schedule mask ──────────────────────────────
#
# Every bit only moves an expression earlier or later in the HEAD loop, or
# RE-READS a value instead of holding it.  No arithmetic expression, operand
# dtype, reduction order or `tl.dot` changes, so every mask gives bitwise
# identical results.  Every arm stays reachable.
#
#   1  `DM`  the [T, T] decay mask is built at its consumer
#            (`dSb = dScore * dec_mask`) instead of at the top of the
#            head body.  It is the LARGEST single tile in the loop --
#            [64, 64] fp32 is 32 registers per thread at 4 warps --
#            and its producer and consumer are eleven statements and
#            three global loads apart.
#   2  `SF`  state-FIRST: `dC_state = f * (dY @ S^T)` is taken
#            immediately after the `S_all` load, before the `x` load,
#            so the [NS, GT] state tile dies before the [T, GT] `x`
#            tile is born instead of living across it.
#   8  `XR`  the aggregate path RE-READS `x` (`volatile`) instead of
#            holding the tile from the score dot; `volatile` stops Triton
#            from CSE-ing the second load back into the first.
#  16  `AG`  `dec_agg` (and the `D_end_val` load it needs) is built in
#            the aggregate path rather than at the top.
#
# `SSD_MIMO_BWD_KBC_SCHED` overrides `_KBC_SCHED_DEFAULT`.  The mask is in
# the autotune KEY: a lever whose purpose is to move the chosen config must
# be in the key, or the first arm's winner is silently reused for every
# other arm.
# THE DEFAULT IS CONDITIONAL ON THE OPERAND MODE.  `XR` re-reads the
# [T, GT] `x` tile rather than holding it: it pays at fp32 operands (a
# spilling body; -2 to -5 % self time) and loses at bf16 operands, which
# roughly halve the spill before it (+9 to +148 %).  The rule is "XR at
# fp32 operands only"; `SSD_MIMO_BWD_KBC_SCHED` still forces any mask on
# either mode.
_KBC_SCHED_DEFAULT = 8


def kbc_sched(mma_io: bool = False) -> int:
    """The Kbc-hb mask for this call.  `mma_io` is D1's operand mode."""
    env = _env("BWD_KBC_SCHED", "")
    if env != "":
        return int(env)
    if mma_io:
        # The 16-bit-operand body's schedule is a MEASURED per-device verdict
        # (tools/cp_calibrate.py --kbc-sched), 0 where none was measured.
        try:
            from .chunkpar import cp_calibration_key
            from .smem import chunk_cache_lookup
            v = chunk_cache_lookup(cp_calibration_key(
                torch.cuda.current_device(), "kbc") + "|sched")
            if v is not None:
                return int(v)
        except Exception:                               # noqa: BLE001
            pass
        return 0
    return _KBC_SCHED_DEFAULT


# "off" -> HB 32, which IS this file's measured Kbc-hb winner (see the
# note just below: HB 32 / 8 warps / 1 stage / no maxnreg at both N).
_HB_VALS = tuple(int(v) for v in
                 _env("BWD_HB_CANDS", "32" if _AUTOTUNE_OFF
                      else "8,32").split(","))
# Kbc-hb does NOT get the maxnreg axis by default: measured over
# {0, 128, 256} x {8, 32} HB x {2, 4, 8} warps x {1, 2, 3} stages it picks
# `HB 32, 8 warps, 1 stage, maxnreg None` at both N = 64 and N = 128 and its
# time does not move, i.e. it is NOT register-bound (its cost is the
# per-head rotation work), so widening the space only costs autotune time.
# SSD_MIMO_BWD_HB_MAXNREG widens it back.
_HB_MAXNREG = tuple(int(v) for v in
                    _env("BWD_HB_MAXNREG", "0").split(","))


# ── the wgmma shared-swizzle floor, in ELEMENTS ─────────────────────────
# A `tl.dot` operand tile is staged in an `#ttg.nvmma_shared` buffer whose
# swizzle width is its contiguous byte width capped at 128.  Triton 3.6
# computes the wgmma descriptor's K stride for the 128-byte mode, so a
# 16-bit tile only 64 bytes wide (32 elements) reads past its buffer --
# see `_hb_nsplit`'s docstring for the sanitizer/nvdisasm evidence.  The
# floor is therefore 128 BYTES: 32 elements at fp32 operands (the rule
# that shipped) and 64 at bf16/fp16 ones.
_SWIZZLE_FLOOR_BYTES = 128


def _hb_ns_floor(mma_io: bool) -> int:
    """Smallest legal `NS = N // NSPLIT`, in elements, for this operand."""
    return _SWIZZLE_FLOOR_BYTES // (2 if mma_io else 4)


def _hb_nsplit(N: int, rot_per_head: bool = False,
               mma_io: bool = False) -> int:
    """How many N-slices `_bwd_kbc_hb_kernel` splits its CTA into.

    A STRUCTURAL choice, not an autotune axis: it changes the GRID, and a
    speed-only search over a grid-changing meta interacts with the
    owner/atomic rule.

    The trade is: each slice repeats the N-INDEPENDENT work (the [T, T]
    score product dY @ x^T, the dY / x / dw loads, the decay tiles) and
    does 1/NSPLIT of the N-DEPENDENT work.  So the optimum is wherever the
    N-dependent half is heavy, and that is TWO different places: under a
    per-head phase (ROT_PER_HEAD: the whole rotation runs inside the HB head
    loop, so the per-head work is proportional to the tile width) split down
    to the 32-wide floor; otherwise split only at N >= 128, where the
    un-split CTA is over the 255-register cap.

    `NS = N // NSPLIT` must stay >= 32; below that the [T, NS] operands
    fall under the MMA tile minimum (the same floor the forward's GT
    guard has).  SSD_MIMO_BWD_NSPLIT forces the value for the A/B.

    ── the floor is in BYTES, not elements ───────────────────────────
    The real floor is `NS * operand_itemsize >= 128`, i.e. the operand
    tile's contiguous dimension must be at least ONE 128-byte swizzle row.
    At 64 bytes Triton 3.6 stages the `[T, NS]` operand as
    `#ttg.nvmma_shared<swizzlingByteWidth = 64, elementBitWidth = 16>` and
    then advances that descriptor's K steps with the 128-byte-swizzle
    stride, so the second `HGMMA` of `dC_h = dot(dSb, Bop)` reads past the
    kernel's whole shared arena (illegal access, compute-sanitizer).

    In ELEMENTS that floor is 32 under fp32 operands and **64 under
    bf16/fp16 operands** (m3 route: bf16 NS = 32 / 16 fault, bf16 NS = 64 /
    128 are exact).  `tf32` at `NS = 16` is a 64-byte row too and is exact,
    because its shared layout is `elementBitWidth = 32`: the axis is the
    tile's BYTE width against the swizzle, not the element count alone.
    """
    # the legality floor, in ELEMENTS, for this operand width.
    ns_floor = _hb_ns_floor(mma_io)
    env = _env("BWD_NSPLIT", "")
    if env:
        ns = int(env)
        # An env-forced split below the floor is REFUSED, not silently
        # served: the config is an illegal access, and the A/B lever must
        # not be able to hand a wrong gradient back quietly.
        if ns > 1 and N // ns < ns_floor:
            raise ValueError(
                "SSD_MIMO_BWD_NSPLIT=%d gives NS=%d at N=%d, below the "
                "%d-element (128-byte) wgmma swizzle floor for these MMA "
                "operands -- Triton 3.6 mis-lowers the "
                "nvmma_shared<swizzlingByteWidth=64> descriptor there "
                "(illegal __shared__ read).  See _hb_nsplit's docstring."
                % (ns, N // ns, N, ns_floor))
    elif rot_per_head:
        ns = max(1, min(4, N // ns_floor))
    else:
        ns = 2 if N >= 128 else 1
    while ns > 1 and (N % ns or N // ns < ns_floor):
        ns //= 2
    return max(1, ns)


def _kb1_nsplit(N: int, R: int, O: int, mma_io: bool = False) -> int:
    """How many N-slices `_bwd_kb1_kernel` splits its CTA into.

    `_hb_nsplit`'s trade, for the kernel that serves R > 1 / O > 1.  Kb1's
    r loop keeps FIVE [T, N] fp32 tiles plus the [N, GT] state live at once
    (C_o_rd, dC_score_o, dC_state_o, Bc_r, dB_r_part); at large N that
    serialises every `wgmma` for want of registers (255 regs, spills).  The
    N-split shrinks every one of those tiles at once.

    Shipped rule: `NSPLIT = 1` at every rank.  Each slice repeats the
    [T, T] score product, which Kb1 already runs O*R times per CTA, so the
    split regresses; the live-range moves below the r loop clear the
    register pressure instead.  The split remains an experimental env
    lever.  `NS = N // NSPLIT` must satisfy `NS * operand_itemsize >= 128`
    -- the same wgmma swizzle floor `_hb_nsplit` documents, refused rather
    than served when forced.  `SSD_MIMO_BWD_KB1_NSPLIT` forces the value
    for the A/B.
    """
    ns_floor = _hb_ns_floor(mma_io)
    env = _env("BWD_KB1_NSPLIT", "")
    if env:
        ns = int(env)
        if ns > 1 and N // ns < ns_floor:
            raise ValueError(
                "SSD_MIMO_BWD_KB1_NSPLIT=%d gives NS=%d at N=%d, below "
                "the %d-element (128-byte) wgmma swizzle floor for these "
                "MMA operands -- Triton 3.6 mis-lowers the "
                "nvmma_shared<swizzlingByteWidth=64> descriptor there.  "
                "See _hb_nsplit's docstring."
                % (ns, N // ns, N, ns_floor))
    else:
        # The split works and is exact, but it LOSES at R, O > 1: each slice
        # repeats the [T, T] score product, and Kb1 already runs that product
        # O*R times per CTA (NSPLIT = 4 makes it 16).  The register pressure
        # is real; the lever that pays for it is the LIVE-RANGE mask
        # (dC_state_o / S_c taken below the r loop), which is unconditional.
        # Kept as `SSD_MIMO_BWD_KB1_NSPLIT` for the A/B.
        ns = 1
    while ns > 1 and (N % ns or N // ns < ns_floor or (N // ns) % 2):
        ns //= 2
    return max(1, ns)


# Instantiations whose head-blocked body does not FIT (shared memory),
# keyed on the launch shape + feature set (see the retry at the Kb1
# launch).  Maps key -> the (NSPLIT, forced launch config) the ladder had
# to reach, so a repeat call starts at the rung that fit.
_KB1_HB_OOR: dict = {}


# ── Shared-memory launch ladder for Ka and Kbc-hb ──────────────────
# Both kernels are atomic / chunk-serial and are launched through the
# autotuner.  Under the `pinned` tier (and under a calibrated one-config grid)
# that is ONE config, and on a 101,376 B part (sm_120) an fp32 / rotating
# Ka at T = 64, N = 128 asks ~104 KB at 8 warps while the SAME body at 4 warps
# fits; Kbc-hb at fp32 asks 131 KB at N = 64, T = 128.  The launch then raised
# a bare `OutOfResources` that the backward's chunk-halving repair cannot take
# (rotation and the host decay prefix are chunk-shaped), after the forward had
# already succeeded.  Neither warps, stages nor an exact N split change any
# arithmetic, so the ladder bypasses the autotuner and launches the JIT
# function at an explicit config, in the order below, until one fits.
# `OutOfResources` is raised at launch BEFORE any CTA runs, so a failed rung
# has written nothing.  The first rung that fits is remembered per
# instantiation.  Configs that fit today never reach this: the autotuned
# launch is rung 0 and is the call that was made before.
_KA_FORCED_LADDER: tuple = ((4, 1), (8, 1), (2, 1), (16, 1), (1, 1))
_KBC_FORCED_LADDER: tuple = ((4, 1), (2, 1), (8, 1))
_KBC_LAST_RESORT: tuple = (1, 1)     # one warp: fits an fp32 N = 64, T = 128 tile at 1 split, but is the slowest body
_KBC_FORCED_HB: int = 32
_OOR_MEMO: dict = {}          # instantiation key -> rung that fit
_OOR_LOG: list = []           # (kernel, key, rung, "ok" | message)


def _oor_key(name, kw, skip=()):
    flat = tuple(sorted((k, v) for k, v in kw.items()
                        if isinstance(v, (bool, int, str))
                        and k != "KA_WB" and k not in skip
                        and not k.startswith("s_")))
    return (name, torch.cuda.current_device()) + flat


def _ka_launch(grid, *args, **kw):
    """`_bwd_ka_kernel[grid](*args, **kw)` with the shared-memory ladder."""
    key = rung = None
    if _OOR_MEMO:                  # happy path: no key built, one truthiness test
        key = _oor_key("ka", kw)
        rung = _OOR_MEMO.get(key)
    if rung is not None:
        try:
            _bwd_ka_kernel.fn[grid](*args, num_warps=rung[0],
                                    num_stages=rung[1], **kw)
            return None
        except OutOfResources:
            _OOR_MEMO.pop(key, None)
    try:
        return _bwd_ka_kernel[grid](*args, **kw)
    except OutOfResources as exc:
        first = exc
    key = key or _oor_key("ka", kw)
    _OOR_LOG.append(("ka", key, "autotuned", str(first)[:100]))
    for w, st in _KA_FORCED_LADDER:
        try:
            _bwd_ka_kernel.fn[grid](*args, num_warps=w, num_stages=st, **kw)
        except OutOfResources as exc:
            _OOR_LOG.append(("ka", key, (w, st), str(exc)[:100]))
            continue
        _OOR_MEMO[key] = (w, st)
        _OOR_LOG.append(("ka", key, (w, st), "ok"))
        return None
    raise first


def kbc_ns_ladder(N, ns0, mma_io, pair_half):
    """`ns0` and every wider EXACT N split (each slice keeps the wgmma
    swizzle floor `_hb_ns_floor`; rotate-half keeps 16 pair lanes)."""
    out = [ns0]
    ns = ns0 * 2
    floor = _hb_ns_floor(mma_io)
    while (N % ns == 0 and N // ns >= floor and (N // ns) % 2 == 0
           and not (pair_half and N // ns < 32)):
        out.append(ns)
        ns *= 2
    return out


def _kbc_launch(grid_of, ns_ladder, *args, **kw):
    """`_bwd_kbc_hb_kernel[grid_of(ns)](*args, NSPLIT=ns, **kw)` with the
    ladder: per N split (narrowest-change first) the autotuned launch, then
    the explicit configs; one warp at the widest split is the last resort."""
    first = rung = key = None
    if _OOR_MEMO:                  # happy path: no key built, one truthiness test
        key = _oor_key("kbc", kw, skip=("KBC_PART",)) + (ns_ladder[0],)
        rung = _OOR_MEMO.get(key)
    skip_plain = False
    if rung is None:
        try:
            _bwd_kbc_hb_kernel[grid_of(ns_ladder[0])](
                *args, NSPLIT=ns_ladder[0], **kw)
            return None
        except OutOfResources as exc:
            first, skip_plain = exc, True
        key = key or (_oor_key("kbc", kw, skip=("KBC_PART",))
                      + (ns_ladder[0],))
        _OOR_LOG.append(("kbc", key, (ns_ladder[0], None), str(first)[:100]))
    rungs = [(ns, f) for ns in ns_ladder for f in (None,) + _KBC_FORCED_LADDER]
    rungs.append((ns_ladder[-1], _KBC_LAST_RESORT))
    if skip_plain:
        rungs = rungs[1:]
    elif rung in rungs:
        rungs = [rung]
    for ns, f in rungs:
        try:
            if f is None:
                _bwd_kbc_hb_kernel[grid_of(ns)](*args, NSPLIT=ns, **kw)
            else:
                _bwd_kbc_hb_kernel.fn[grid_of(ns)](
                    *args, NSPLIT=ns, HB=_KBC_FORCED_HB, num_warps=f[0],
                    num_stages=f[1], **kw)
        except OutOfResources as exc:
            first = first or exc
            _OOR_LOG.append(("kbc", key, (ns, f), str(exc)[:100]))
            continue
        _OOR_MEMO[key] = (ns, f)
        _OOR_LOG.append(("kbc", key, (ns, f), "ok"))
        return None
    raise first


# (Annotated assignments on purpose: these are HOST launch-ladder constants, not
# the tuned config space, so they stay outside `calibration.kernel_source`.)

# Dispatch trace for the Kb1 owner-accumulate schedule: every
# `fused_chunk_bwd_5kernel` call that reaches the Kb1 launch appends ONE
# record here, so `kb1_dispatch_log()` shows WHY `SSD_MIMO_BWD_KB1_HB=auto`
# did or did not engage.  `reason` is the FIRST true refusal term in source
# order, or "oor" if the upfront gate passed but the runtime shared-memory
# retry fell back to HBLK=0, or "" when it ran.
_KB1_DISPATCH_LOG: list = []
_KB1_DISPATCH_LOG_MAX = 32


def kb1_dispatch_log() -> list:
    """The last 32 Kb1-HB dispatch records, newest last.

    Keys: R, O, T, N, G_SIZE, bc_grouped, fuse_kb2, use_hb, deferred_rot,
    use_slots, use_slots_c, RO, TN, ro_max, tn_max, gate_refused, hb
    (the HBLK the upfront gate picked, 0 if refused), oor (bool: the
    runtime shared-mem retry fell back), hb_final (the HBLK actually
    launched, 0 if none), reason (str).
    """
    return list(_KB1_DISPATCH_LOG)


def _note_kb1_dispatch(record: dict) -> None:
    _KB1_DISPATCH_LOG.append(record)
    del _KB1_DISPATCH_LOG[:-_KB1_DISPATCH_LOG_MAX]


# ── The Kb1 REGISTER-CLIFF rule ───────────────────────────────────────
# WHAT.  ptxas can compile `_bwd_kb1_kernel` to a COLLAPSED body: it caps
# the register count at an occupancy boundary instead of the 255 the
# body needs, and pays for it in local-memory spills (sm_80: every
# candidate 32 regs, 5-26 KB stack/thread; sm_90: an 8-warp body of 128
# regs / 2.9 KB stack).  `SSD_MIMO_BWD_KB1_NSPLIT=2` takes the same body
# to 255 regs on every arch, but the split loses where there is NO cliff,
# so it must be conditional.
#
# RULE (compiled facts, not an arch list).  Before the FIRST launch of a
# specialisation, every candidate config the launch could run with is
# COMPILED (no launch; the autotuner reuses these very kernels, so a cell
# with no cliff compiles nothing extra) and read back:
#     collapsed  :=  n_regs <= SSD_MIMO_BWD_KB1_CLIFF_REGS (128)
#                    and stack >= SSD_MIMO_BWD_KB1_CLIFF_STACK (1024 B)
# The remedy engages only when EVERY candidate that fits is collapsed --
# with one un-collapsed candidate the autotuner can pick it, and if it
# picks a collapsed one instead that one MEASURED faster.  The remedy's
# own candidates are compiled and checked the same way; a remedy that
# is still collapsed is not taken (the next lever in the ladder is
# tried, then nothing).  One decision per specialisation per process
# (`_KB1_CLIFF`), recorded in `kb1_dispatch_log()["cliff"]`.
#
# LEVER `SSD_MIMO_BWD_KB1_CLIFF` (every remedy is a launch / schedule
# choice with the same mathematics; NSPLIT is exact -- distinct column
# sets -- and DTH_STORE / maxnreg change no arithmetic):
#   nsplit   (DEFAULT) the next legal N-split rung (NSPLIT 2, then 4).
#   maxnreg  bypass the autotuner at the pinned (8 warps, 1 stage) with
#            maxnreg=255 (compile-only evidence).
#   dth0     the dTheta sole-owner store off (zero-fill + atomics) --
#            compile-only; clears sm_90 (255 regs) but NOT sm_80 (32).
#   off      the rule never engages.
# A non-`nsplit` lever that does not clear the cliff escalates to
# `nsplit`.  `SSD_MIMO_BWD_KB1_NSPLIT` set explicitly also turns the
# rule off (the caller pinned the split).
_KB1_CLIFF: dict = {}
_KB1_CLIFF_LEVERS = ("nsplit", "maxnreg", "dth0", "off")
_KB1_CLIFF_REGS = int(_env("BWD_KB1_CLIFF_REGS", "128"))
_KB1_CLIFF_STACK = int(_env("BWD_KB1_CLIFF_STACK", "1024"))


def _kb1_cliff_lever() -> str:
    if _env("BWD_KB1_NSPLIT", ""):
        return "off"
    v = _env("BWD_KB1_CLIFF", "nsplit").strip().lower()
    if v not in _KB1_CLIFF_LEVERS:
        raise ValueError(
            "SSD_MIMO_BWD_KB1_CLIFF=%r: expected one of %s"
            % (v, "/".join(_KB1_CLIFF_LEVERS)))
    return v


def _kb1_collapsed(n_regs: int, stack_bytes: int) -> bool:
    """A body ptxas capped at an occupancy boundary AND made spill."""
    return (n_regs <= _KB1_CLIFF_REGS
            and stack_bytes >= _KB1_CLIFF_STACK)


def _kb1_probe(compile_one, cfg_kwargs: list) -> list:
    """Compile (never launch) each config and read its register facts.

    `compile_one(kw)` returns the CompiledKernel the launch with the
    extra launch options `kw` would run.  A config that does not fit
    (shared memory, ptxas) is recorded `fit=False`: the autotuner scores
    it inf and a forced launch falls to the OOR ladder, so it can never
    be the body that runs.
    """
    from triton.runtime.errors import PTXASError
    try:
        from triton.compiler.errors import CompileTimeAssertionFailure
    except ImportError:                                   # pragma: no cover
        CompileTimeAssertionFailure = OutOfResources
    out = []
    for kw in cfg_kwargs:
        f = dict(w=kw.get("num_warps"), s=kw.get("num_stages"),
                 mnr=kw.get("maxnreg"))
        try:
            k = compile_one(kw)
            k._init_handles()
            f.update(fit=True, regs=int(k.n_regs),
                     stack=4 * int(k.n_spills))
            f["collapsed"] = _kb1_collapsed(f["regs"], f["stack"])
        except (OutOfResources, PTXASError, CompileTimeAssertionFailure) as e:
            f.update(fit=False, err=type(e).__name__)
        out.append(f)
    return out


def _kb1_all_collapsed(facts: list) -> bool:
    """At least one candidate fits, and every one that fits collapsed."""
    fit = [f for f in facts if f.get("fit")]
    return bool(fit) and all(f["collapsed"] for f in fit)


def _kb1_clears(facts: list) -> bool:
    """A remedy is taken only if one of its candidates fits un-collapsed."""
    return any(f.get("fit") and not f["collapsed"] for f in facts)


def _kb1_spec_key(args, kw) -> tuple:
    """Every non-tensor launch value, plus each tensor's dtype: the
    specialisation (and more -- strides are included raw, so a new shape
    re-reads the facts, which is a compile-cache hit)."""
    def one(v):
        if isinstance(v, torch.Tensor):
            return ("T", str(v.dtype))
        if isinstance(v, (int, float, bool, str, type(None))):
            return v
        return str(v)
    return (tuple(one(v) for v in args),
            tuple(sorted((k, one(v)) for k, v in kw.items())))


_SM_COUNT = {}


def _sm_count(device) -> int:
    idx = getattr(device, "index", None)
    if idx is None:
        idx = torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(
            idx).multi_processor_count
    return _SM_COUNT[idx]


# P1 COMPILE CAP.  The register accumulators are a TUPLE indexed by the
# r / o loop variable, so those two loops must be `tl.static_range` --
# i.e. the (o, r) body is fully unrolled, R*O copies of three `tl.dot`s
# plus (under rotation) two pair-rotation chains.  R = O = 8 (64 bodies) does
# not finish a ROTATING compile in practice.  Above this product the head
# block is refused and the one-head-per-CTA body (HBLK = 0) runs.
_KB1_HB_RO_MAX = int(_env("BWD_KB1_HB_RO_MAX", "16"))

# The same cap for the NO-TENSOR-CORE route.  Under `precision="ieee"` every
# `tl.dot` lowers to an FMA expansion, so the head-blocked body's R*O
# unrolled (o, r) copies become one straight-line block of FMA dots that
# ptxas allocates as a whole: 32-40 registers, heavy local-memory spill and
# 10-97 s per config, while the HBLK = 0 body (runtime o / r loops, one body
# instance) compiles in seconds at 194-255 registers.  So the ieee route
# takes HBLK = 0 (and Kb2 unfolded) whenever R*O > this cap.  A schedule
# choice: same mathematics; tf32 / io (tensor-core) routes are unaffected.
# SSD_MIMO_BWD_KB1_HB_RO_MAX_IEEE=16 restores the old dispatch.
_KB1_HB_RO_MAX_IEEE = int(_env("BWD_KB1_HB_RO_MAX_IEEE", "1"))

# P1 TILE CAP.  Same reason from the other side: at T = N = 128 the
# head-blocked body's live ranges need ~200 KB of SHARED memory in every
# `_BWD_CFGS` config against the A100's 164 KB, so EVERY config is
# `OutOfResources` and the whole autotune is wasted (~4 min per
# instantiation) before the retry below falls back.  The retry stays as
# the safety net; this cap is what keeps the common case from paying for
# it.  T * N = 8192 admits N = 128 at chunk 64 and N = 64 at chunk 128.
_KB1_HB_TN_MAX = int(_env("BWD_KB1_HB_TN_MAX", "8192"))

# ── ROTBWD: MEASUREMENT-ONLY rotation ablation for the MIMO
# backward (`SSD_MIMO_BWD_ROT_ABL_MIMO`, default 0 = off = today's code,
# bit-identically).  Bits: 1 = no phase gradient dTheta, 2 = B~/C~ taken
# as pre-rotated on load, 4 = adjoint compiled with HAS_THH off.  Every
# non-zero value computes a WRONG gradient and exists only to price the
# rotation terms.
_ROT_ABL_MIMO = int(_env("BWD_ROT_ABL_MIMO", "0"))


def _kb1_hb(H, hpg_b, hpg_c, Bsz, nc, NGT, nsplit, refused, device):
    """The OWNER-ACCUMULATE head block of `_bwd_kb1_kernel`.

    Returns `(HBLK, DB_STORE, DC_STORE)`; `HBLK = 0` is the one-head-per-CTA
    body.

    WHY.  Kb1 at R = O = 2 is ATOMIC-bound, not compute- or spill-bound:
    every CTA issues ~27k fp32 atomic RMWs for the `dBtil` / `dCtil` /
    `dTheta` tiles, and the cost is LINEAR in atomic tiles issued (L2 RMW
    serialisation, not bandwidth).

    `_bwd_kbc_hb_kernel` removed exactly this at R = O = 1 by giving the
    CTA a head BLOCK; this is that construction at R, O > 1, where the
    accumulator set is R + O tiles instead of 2.

    The block must lie INSIDE one B/C group, or the register accumulator
    would mix two tiles that live at different addresses: HBLK is chosen
    among the divisors of `min(hpg_b, hpg_c)`, which also makes it a
    divisor of H.  The largest such block that still fills the machine
    (>= 2 CTAs per SM) wins, because the fan-in falls as 1/HBLK and the
    sole-owner store needs HBLK >= hpg.

    DEFAULT "auto": `SSD_MIMO_BWD_KB1_HB` is "auto" unless set; "auto" takes
    the heuristic, an integer forces the block size.
    """
    if refused:
        return 0, False, False
    hpg = max(1, min(hpg_b, hpg_c))
    divs = [d for d in range(hpg, 0, -1) if hpg % d == 0]
    # SSD_MIMO_BWD_KB1_HB: "auto" (default) takes the heuristic below; an
    # integer forces the block size; "0" is the one-head-per-CTA body.
    env = _env("BWD_KB1_HB", "auto")
    if env != "auto":
        hb = int(env)
        if hb <= 0:
            return 0, False, False
        if hpg % hb:
            raise ValueError(
                "SSD_MIMO_BWD_KB1_HB=%d does not divide the heads per B/C "
                "group (%d): a head block that straddles a group would "
                "accumulate two different dBtil/dCtil tiles into one "
                "register file." % (hb, hpg))
    else:
        target = 2 * _sm_count(device)
        hb = 1
        for d in divs:
            if Bsz * ((H + d - 1) // d) * nc * NGT * nsplit >= target:
                hb = d
                break
    # The tile's remaining fan-in is the heads of its group that this CTA
    # does NOT own, times the G-tile axis.  (The N-split slices columns,
    # so it never shares a tile.)
    db_store = bool(hb >= hpg_b and NGT == 1)
    dc_store = bool(hb >= hpg_c and NGT == 1)
    return hb, db_store, dc_store


def _hb_cfg(hb, w, st, mnr):
    kw = dict(num_warps=w, num_stages=st)
    if mnr:
        kw["maxnreg"] = mnr
    return triton.Config({"HB": hb}, **kw)


# Under "off" Kbc-hb takes its OWN measured winner (HB 32, 8 warps,
# 1 stage, no maxnreg) rather than the generic _BWD_* heuristic.
_HB_CFGS = ([_hb_cfg(32, 8, 1, 0)] if _AUTOTUNE_OFF else
            [_hb_cfg(hb, w, st, m)
             for hb in _HB_VALS for w in _BWD_WARPS for st in _BWD_STAGES
             for m in _HB_MAXNREG])


def _hb_grid(tier):
    if tier == "pinned":
        return [_hb_cfg(32, 8, 1, 0)]
    w, st, _ = _TIER_AXES[tier]
    return [_hb_cfg(hb, a, b, m)
            for hb in (int(v) for v in _env("BWD_HB_CANDS", "8,32").split(","))
            for a in (int(v) for v in _env("BWD_WARPS", w).split(","))
            for b in (int(v) for v in _env("BWD_STAGES", st).split(","))
            for m in _HB_MAXNREG]


@triton.autotune(configs=_calib_grid("fused_kbc_hb", _hb_grid),
                 prune_configs_by={"early_config_prune": _cprune("fused_kbc_hb")},
                 # BSZ is in the key so a small batch is not stuck with the
                 # HB block tuned at a large one (HB trades CTA count
                 # against atomic fan-in; the optimum moves with B).
                 key=["T", "N", "G_SIZE", "GT", "PREC", "MMA_IO",
                      "HAS_BETA", "HAS_GAMMA", "HAS_DW",
                      "HAS_THETA_H", "HAS_ROT_BIAS", "HAS_ROT",
                      "ROT_PER_HEAD", "BIAS_PRE_ROT", "PAIR_HALF",
                      # A lever whose purpose is to move the chosen config
                      # must be in the KEY, or the first arm's winner is
                      # silently reused.
                      "SCHED",
                      # These levers change the body, so a config
                      # tuned without them must not be reused with them
                      "SUMDS", "KBC_DALPHA", "DD_PAIRS", "KBC_PART", "TS_ABL",
                      "H", "NC", "BSZ", "IDX64"],
                 restore_value=["dCtil_ptr", "dBtil_ptr", "dDelta_ptr",
                                "dBeta_ptr", "dGamma_ptr",
                                # KBC_DALPHA: atomically accumulated, so an
                                # autotune replay would add it twice
                                "dAlpha_ptr",
                                "dThetaH_ptr", "dRotBias_ptr", "dTheta_ptr",
                                # the per-chunk phase-offset gradient
                                # is atomically accumulated whenever this
                                # CTA is not the sole owner, so an
                                # autotune REPLAY would add it twice.
                                "dThOff_ptr"])
@triton.jit
def _bwd_kbc_hb_kernel(
    C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, SAll_ptr, dG_ptr, DW_ptr,
    Beta_ptr, Gamma_ptr, ThetaH_ptr, RotBias_ptr, Theta_ptr,
    dY_ptr,
    dCtil_ptr, dBtil_ptr, dDelta_ptr, dBeta_ptr, dGamma_ptr,
    dThetaH_ptr, dRotBias_ptr, dTheta_ptr,
    H, BSZ,
    s_cb, s_cc, s_ct, s_cn,
    s_bb, s_bc, s_bt, s_bn,
    s_xb, s_xh, s_xg, s_xl,
    s_db, s_dh, s_dc, s_dt,
    s_yb, s_yh, s_yc, s_yt, s_yg,
    s_sb, s_sh, s_sc, s_sn, s_sg,
    s_gb, s_gh, s_gc, s_gn, s_gg,
    s_wb, s_wh, s_wl,
    s_dcb, s_dcc, s_dct, s_dcn,
    s_dbb, s_dbc, s_dbt, s_dbn,
    s_thb, s_thc, s_tht, s_thh, s_thk,
    s_dtb, s_dtc, s_dtt, s_dth, s_dtk,
    NC, NGT_val,
    NSPLIT: tl.constexpr,
    T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr,
    GT: tl.constexpr, PREC: tl.constexpr,
    # `tl.dot` operand dtype: fp32 = identity cast; bf16 / fp16 = IO-dtype MMA
    # with fp32 accumulate.  See `mma_operand_dtype` in fused_fwd_kernel.py.
    MMA_DT: tl.constexpr = tl.float32,
    # Autotune KEY stand-in for MMA_DT (a tl.dtype is not serialisable into
    # Triton's disk cache key): 1 iff MMA_DT is 16-bit.
    MMA_IO: tl.constexpr = 0,
    HAS_BETA: tl.constexpr = False,
    HAS_GAMMA: tl.constexpr = False,
    HAS_DW: tl.constexpr = False,
    HAS_THETA_H: tl.constexpr = False,
    HAS_ROT_BIAS: tl.constexpr = False,
    HAS_ROT: tl.constexpr = False,
    ROT_PER_HEAD: tl.constexpr = False,
    BIAS_PRE_ROT: tl.constexpr = False,
    HB: tl.constexpr = 8,
    IDX64: tl.constexpr = False,
    # ── ROT_PER_HEAD register diet (A/B lever) ────────────────────────
    # SSD_MIMO_BWD_ROT_DIET=0 restores the undieted body.  The two forms are
    # BIT-IDENTICAL (pure code motion plus a layout op recomputed instead of
    # held live), so this is a measurement lever only.
    ROT_DIET: tl.constexpr = True,
    # ── An ATTRIBUTION ladder inside the ROT_PER_HEAD body ────────────
    # SSD_MIMO_BWD_ROT_ABL, MEASUREMENT ONLY -- every bit but 0 computes the
    # WRONG gradient.  It is a BITMASK so the terms compose and can be priced
    # independently:
    #    1 no forward phase cumsum        2 no trig
    #    4 no dTheta CONTRACTION+store    8 no dBeta/dGamma atomics
    #   16 no rotation build/un-rotation 32 axis-0 (pre-A5) reductions
    #   64 dTheta contracted but NOT stored
    #  128 variant (b): C~ per head from a stored map
    #  256 variant (c)/B0 kernel side: BOTH maps per head, bias folded,
    #      no rotation and no dTheta/dBeta/dGamma, dB~/dC~ stored out
    #  512 no reverse cumsum (only meaningful without PHASE_DEFER)
    # 128/256 read and WRITE real per-head buffers, so the H-fold
    # traffic B0 pays is in the measurement.
    ROT_ABL: tl.constexpr = 0,
    BRot_ptr=None, CRot_ptr=None,
    s_rb=0, s_rc=0, s_rt=0, s_rh=0, s_rn=0,
    # ── SLOTTED bias-gradient atomics ────────────────────────────────
    # Under BIAS_PRE_ROT dBeta / dGamma are a per-HEAD [N] atomic, so
    # every one of the B * nc * NSPLIT CTAs adds into the SAME [H, N]
    # buffer once per head: 32768 vector atomics onto 4096 floats at the
    # spine cell.  BSLOT > 1 gives the CTA a private plane keyed on its
    # chunk, and the planes are summed on the host.  Same numbers up to
    # fp32 addition order (which the atomics never fixed either).
    BSLOT: tl.constexpr = 1,
    s_bes=0, s_gas=0,
    # ── The phase cumsum, in-kernel ───────────────────────────────────
    # `Theta_ptr` holds per-token INCREMENTS, `ThOff_ptr` the phase
    # entering each chunk.  This is the kernel that also produces the
    # phase gradient, so it carries the ADJOINT of that reparameterisation
    # (see the dTheta blocks below): the increment gradient is the
    # intra-chunk REVERSE cumsum of the phase gradient, and the offset
    # gradient is its chunk total.  Both are LINEAR in the phase
    # gradient, so a per-CTA partial still sums correctly under the same
    # owner/atomic rule the unsummed form used.
    ThOff_ptr=None, dThOff_ptr=None,
    s_ob=0, s_oh=0, s_oc=0, s_ok=0,
    s_dob=0, s_doh=0, s_doc=0, s_dok=0,
    PHASE_CUMSUM: tl.constexpr = False,
    # ── PAIRHALF: the ROTATE-HALF pair layout ─────────────────────────
    # See `fused_fwd_kernel.py`'s constexpr of the same name.  True puts
    # rotation pair k at (k, k + N/2) and replaces every cross-lane
    # partner with the other contiguous half tile.  Stage 1: R = O = 1
    # (this kernel IS the R = O = 1 backward), per-head phase, no
    # rot_bias, no ablation ladder.
    PAIR_HALF: tl.constexpr = False,
    # PHASE_DEFER: the kernel stores the RAW dL/dtheta_cum and one
    # chunk-parallel pass (`m3_epilogue.phase_prefix_bwd`) does the in-chunk
    # reverse cumsum, the chunk total and the cross-chunk term (an in-kernel
    # cross-thread scan per head is too costly in this register-starved body).
    PHASE_DEFER: tl.constexpr = False,
    # ── The LIVE-RANGE schedule mask (`SSD_MIMO_BWD_KBC_SCHED`) ──────
    # Numerics-neutral code motion only; see `kbc_sched` above.
    SCHED: tl.constexpr = 0,
    # `Delta_ptr` holds the RAW chunked delta; this kernel forms the
    # chunk-local prefix D_i itself (`_dcs_tile` / `_dcs_end`).
    DCS_INK: tl.constexpr = False,
    # TRAPFOLD (fused_fwd_kernel `_TRAPFOLD_NOTE`): under the two-tap
    # trapezoid the score diagonal's dB / dC (weight own_t, not the
    # collapsed w_t this kernel sees) is carried by `trap_adjoint`
    # (dd_t = own_t s_t); this kernel's diagonal is then zero.
    TRAP_DIAG: tl.constexpr = False,
    # ── SUMDS: the HEAD SUM before
    # the state contraction.  With head-SHARED B/C and no rotation, the
    # two score GEMMs of every head, dSb_h @ B and dSb_h^T @ C (T x T x N,
    # the largest two of the five per head), are linear in dSb_h with the
    # SAME right operand for every head of the block, so
    #     sum_h dSb_h @ (B + 1 beta_h^T) = (sum_h dSb_h) @ B + sum_h rs_h beta_h^T
    # (rs_h = dSb_h 1, cs_h = dSb_h^T 1; the per-head B/C biases become
    # exact rank-1 fp32 corrections).  The block accumulates the fp32
    # [T, T] sum and runs the two contractions ONCE per CTA -- Mamba-2's
    # own ngroups = 1 structure (`_chunk_scan_bwd_dcb` sums dCB over the
    # heads of a group before `bmm`).  Only the rounding changes: one
    # bf16 operand rounding of the head sum instead of one per head.
    SUMDS: tl.constexpr = False,
    # ── KBC_DALPHA (with Ka NO_DALPHA): dalpha[b, h, gt, c] += sum(dG * S)
    # over this CTA's (N-slice, G-tile) -- the same two tiles the head
    # loop already loads; fp32 atomic into a zeroed buffer.
    KBC_DALPHA: tl.constexpr = False,
    dAlpha_ptr=None, s_dab=0, s_dah=0, s_dag=0, s_dac=0,
    # ── DD_PAIRS (lever `SSD_MIMO_BWD_DD_IN_KBC`):
    # the SCORE PAIRS of dL/ddelta, M = S0 o dSb (S0 = the head-shared raw
    # score C B^T, `CB_ptr` [B, nc, T, T] fp32 -- the SHARED_CB buffer),
    # span-summed HERE, where dSb = (dY x_w^T) o mask already lives, instead
    # of in Kc, which formed dY x_w^T a second time for them.  Owned by the
    # N-slice 0 CTA only (every slice holds the same dSb).
    DD_PAIRS: tl.constexpr = False,
    CB_ptr=None, s_cbb=0, s_cbc=0,
    # ── KBC_PART (lever `SSD_MIMO_BWD_KBC_SPLIT`):
    # 0 = the fused body; 1 = the dC half only (dC_state, dC_score, the
    # READ term of dDelta, dGamma); 2 = the dB half only (dB_score,
    # dB_agg, dBeta).  Launched as two kernels, each keeps ONE [T, NS]
    # accumulator and only its own operand tiles live (Mamba-2's own
    # dc / db split); dScore = dY x_w^T is formed in both.
    KBC_PART: tl.constexpr = 0,
    # ── TS_ABL (`SSD_MIMO_BWD_TS_ABL`), MEASUREMENT
    # ONLY -- every bit but 0 computes the WRONG gradient.  It prices a
    # T_s = 128 backward state chunk (states / dG stored every 2 chunks)
    # before building it: 1 = the half state READ (S skipped on odd
    # chunks, dG on even ones -- masked loads, no bytes), 2 = the
    # mid-chunk reconstruction dot that replaces it ([NS, T] x [T, GT]
    # from the neighbour chunk's x / dY tile, loaded here).
    TS_ABL: tl.constexpr = 0,
):
    """Head-batched score+aggregate backward for the O=1, R=1, ungrouped
    fused path -- rotation mode included.

    Produces the same dCtil/dBtil/dDelta/dBeta/dGamma (and, in rotation
    mode, dThetaH/dRotBias) values as Kb1 + Kinv + Kc's dB_agg part; the
    per-head arithmetic is the same real number and only the cross-head
    summation order differs, which the atomics never fixed either.

    Per head: 5 GEMMs -- dC_state (dY @ S^T), dScore (dY @ x_s^T),
    dC_score (dSb @ B), dB_score (dSb^T @ C), dB_agg (xw @ dG^T).

    ROTATION (HAS_THETA_H / HAS_ROT_BIAS): the pair
    rotations are folded into the GEMM *operands* rather than undone on
    the GEMM *results*, so dB_acc/dC_acc stay in the UNROTATED frame and
    are written once, coalesced -- no permuted atomic scatter and no
    per-head dC_rd scratch buffer.  With M_th = R(theta_h) and
    M_rho = R(rot_bias) (block-diagonal 2x2 rotations on the adjacent
    pairs, hence orthogonal), the semantics reference.py:336-337 fixes are
    B_eff_rd = M_rho @ B + beta (beta NOT rotated) and
    C_eff_rd = M_th @ (C + gamma) (gamma IS rotated), and

        dC = dSb @ (B_eff_rd M_th)  +  f * (dY @ (M_th^T S)^T)
        dB = dSb^T @ (C_eff_rd M_rho)  +  xw @ (M_rho^T dG)^T

    with every rotated operand built from the tile at offs_n and the tile
    at offs_n ^ 1 -- both LOADS, never a register lane permutation.  The
    dDelta score terms use orthogonality:
    sum(C_eff_rd * dC_rd) = sum((C + gamma) * dC) exactly, and the beta
    part of the B term moves to M_rho^T beta.  dTheta_h and dRotBias use
    M^T J M = J (J = the pair generator), which turns the rotated-frame
    cross terms into dC_e * Ceff_o - dC_o * Ceff_e and
    dB_e * B_o - dB_o * B_e.

    IN-KERNEL PER-TOKEN PHASE (HAS_ROT): B and C
    arrive RAW and B~ = R(theta_t)^T B / C~ = R(theta_t)^T C are formed
    ONCE per CTA, outside the head loop (theta is head-shared), with the
    xor-lane partner taken from the register pair swap.  Everything
    downstream is the SAME arithmetic on B~/C~ that the host fold used to
    hand it.  The two accumulators come out in the B~/C~ frame, so the
    epilogue does the two things the host fold's autograd used to do:

        dTheta  = sum_pair (dB~_e B~_o - dB~_o B~_e)  + the same for C
        dB      = R(theta_t) dB~        (per-token, pairwise)

    dTheta is bilinear in (dB~, B~) with B~ shared across heads, so a
    per-CTA partial sums correctly under the same owner/atomic rule the
    dB/dC tiles use.  No exponent is touched.

    PER-HEAD PHASE (ROT_PER_HEAD): the phase is
    [B, nc, T, H, K] and s_thh is its head stride, so B~/C~ are no longer
    the same tile for every head in the block and the hoist above cannot
    stand.  Under this flag B and C are held RAW and the whole rotation
    -- forming B~/C~, the dTheta pair contraction and the un-rotation of
    (dB_h, dC_h) back to the raw frame -- moves INSIDE the head loop, in
    that order, so every atomic that reads the rotated frame (dThetaH,
    dRotBias, dDelta) still sees it and the two accumulators still leave
    the CTA unrotated and coalesced.  dTheta is then per (b, c, t, h, k)
    and its only fan-in is the G-tile axis, so the owner rule is
    NGT_val == 1 rather than the cross-head one.  Same arithmetic, one
    head at a time; no exponent is touched here either.

    No exponent is touched: dec_mask / dec_agg keep the unchanged
    difference-first, causal-masked-to--inf form (contract 7.2.1).
    """
    # BIAS_PRE_ROT rides ROT_PER_HEAD: with beta/gamma INSIDE the
    # rotation the rotated tile depends on the head through the bias, so
    # the hoist out of the head loop is invalid whatever the phase's own
    # head structure is -- and Mamba-3, the law this exists to reproduce,
    # has a per-head phase anyway.  Refused rather than half-served.
    tl.static_assert(
        not BIAS_PRE_ROT
        or (ROT_PER_HEAD and not HAS_ROT_BIAS),
        "Kbc-hb: bias_pre_rot needs a per-head in-kernel phase and is "
        "refused with rot_bias (theta_h, a per-head READ angle, rides the "
        "per-head tile like the phase does)")
    SCHED_DM: tl.constexpr = (SCHED.value & 1) != 0
    SCHED_SF: tl.constexpr = (SCHED.value & 2) != 0
    SCHED_XR: tl.constexpr = (SCHED.value & 8) != 0
    SCHED_AG: tl.constexpr = (SCHED.value & 16) != 0
    b = tl.program_id(0)
    hblk = tl.program_id(1)
    # ── the N SPLIT ───────────────────────────────────────────────────
    # Every quantity this kernel produces is indexed by the STATE
    # coordinate n (dB, dC, dBeta, dGamma, dThetaH, dRotBias, dTheta) or
    # is already an atomic reduction over it (dDelta), so an N-slice per
    # CTA needs NO cross-CTA reduction that was not there before.  What
    # IS duplicated is the N-independent work: the [T, T] score product
    # dY @ x^T, the dY / x / dw loads and the decay tiles.  At N = 128
    # the un-split CTA carries four [T, N] fp32 tiles plus [N, GT] state
    # and dG, hits the 255-register cap and SPILLS; halving the tile is
    # what buys the spills back.
    NS: tl.constexpr = N // NSPLIT
    pid2 = tl.program_id(2)
    ns_idx = pid2 % NSPLIT
    rest = pid2 // NSPLIT
    c = rest // NGT_val
    gt_idx = rest % NGT_val
    if IDX64:
        b = b.to(tl.int64)

    offs_t = tl.arange(0, T)
    offs_n = ns_idx * NS + tl.arange(0, NS)
    offs_k = ns_idx * (NS // 2) + tl.arange(0, NS // 2)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    # The score diagonal rides the MMAs (weight w_t); under the
    # trapezoid it is STRICT (`trap_adjoint` carries own_t)
    if TRAP_DIAG:
        causal = offs_t[:, None] > offs_t[None, :]
    else:
        causal = offs_t[:, None] >= offs_t[None, :]
    neg_inf = float("-inf")

    if PAIR_HALF:
        # ═══ PAIRHALF: the ROTATE-HALF pair layout ═════════════════════
        # Pair k lives at state coordinates (k, k + N/2), so every
        # rotated tile is TWO CONTIGUOUS [T, PS] tiles whose pair
        # partner is THE OTHER TILE.  This branch therefore contains no
        # `_pair_swap_2d`, no `_pair_bcast_2d` and no `offs_n ^ 1` load:
        #   B~_1 = c B_1 + s B_2,      B~_2 = c B_2 - s B_1
        #   swap(B~)_1 = B~_2                (free)
        #   dB_1 = c dB~_1 - s dB~_2,  dB_2 = c dB~_2 + s dB~_1
        # and the phase gradient, which would otherwise pay the cross-lane
        # partner, is
        #   dTheta = dB~_1 B~_2 - dB~_2 B~_1 + dC~_1 C~_2 - dC~_2 C~_1
        # with no split/join at all.
        #
        # The state axis is a CONTRACTION in dD and in the `tl.sum`
        # reductions (sum of the two halves) and an OUTPUT axis in
        # dB/dC/dBeta/dGamma/dC_state (two half tiles), so the N SPLIT
        # slices the PAIR axis, not the raw n axis: a contiguous n range
        # is not a set of whole pairs under this layout.
        tl.static_assert(not HAS_ROT_BIAS,
                         "PAIR_HALF does not serve rot_bias")
        tl.static_assert(not HAS_ROT or ROT_PER_HEAD,
                         "PAIR_HALF serves the per-head phase (the SSD "
                         "rotation law); a head-shared phase keeps the "
                         "interleaved branch")
        tl.static_assert(ROT_ABL == 0,
                         "PAIR_HALF carries no ablation ladder")
        tl.static_assert(N >= 32 * NSPLIT,
                         "PAIR_HALF needs N/(2*NSPLIT) >= 16 (MMA K floor)")
        PS: tl.constexpr = N // (2 * NSPLIT)
        KH: tl.constexpr = N // 2
        offs_p = ns_idx * PS + tl.arange(0, PS)      # PAIR index
        offs_h1 = offs_p                              # first-half rows
        offs_h2 = KH + offs_p                         # second-half rows

        c_base = C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct
        b_base = B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt
        Co1_base = tl.load(c_base + offs_h1[None, :] * s_cn).to(tl.float32)
        Co2_base = tl.load(c_base + offs_h2[None, :] * s_cn).to(tl.float32)
        Bc1_base = tl.load(b_base + offs_h1[None, :] * s_bn).to(tl.float32)
        Bc2_base = tl.load(b_base + offs_h2[None, :] * s_bn).to(tl.float32)

        dth_sole = NGT_val == 1

        dB1_acc = tl.zeros([T, PS], dtype=tl.float32)
        dB2_acc = tl.zeros([T, PS], dtype=tl.float32)
        dC1_acc = tl.zeros([T, PS], dtype=tl.float32)
        dC2_acc = tl.zeros([T, PS], dtype=tl.float32)

        h_start = hblk * HB
        hb_count = tl.minimum(HB, H - h_start)
        for hh in range(hb_count):
            h = h_start + hh
            if IDX64:
                h = h.to(tl.int64)
            a_val = -tl.exp(tl.load(A_ptr + h))

            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            dec_mask = tl.exp(
                tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                         neg_inf))
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)

            Bs1 = Bc1_base
            Bs2 = Bc2_base
            Cs1 = Co1_base
            Cs2 = Co2_base
            if HAS_BETA:
                be1 = tl.load(Beta_ptr + h * N + offs_h1).to(tl.float32)
                be2 = tl.load(Beta_ptr + h * N + offs_h2).to(tl.float32)
            if HAS_GAMMA:
                ga1 = tl.load(Gamma_ptr + h * N + offs_h1).to(tl.float32)
                ga2 = tl.load(Gamma_ptr + h * N + offs_h2).to(tl.float32)
            if BIAS_PRE_ROT:
                if HAS_BETA:
                    Bs1 = Bs1 + be1[None, :]
                    Bs2 = Bs2 + be2[None, :]
                if HAS_GAMMA:
                    Cs1 = Cs1 + ga1[None, :]
                    Cs2 = Cs2 + ga2[None, :]
            if HAS_ROT:
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht + offs_p[None, :] * s_thk,
                ).to(tl.float32)                                 # [T, PS]
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + offs_p * s_ok,
                    ).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
                Bt1 = ph_cos * Bs1 + ph_sin * Bs2
                Bt2 = ph_cos * Bs2 - ph_sin * Bs1
                Ct1 = ph_cos * Cs1 + ph_sin * Cs2
                Ct2 = ph_cos * Cs2 - ph_sin * Cs1
            else:
                Bt1 = Bs1
                Bt2 = Bs2
                Ct1 = Cs1
                Ct2 = Cs2

            if HAS_THETA_H:
                th_h = tl.load(ThetaH_ptr + h * KH + offs_p)
                ch_h = _cos_approx(th_h)
                sh_h = _sin_approx(th_h)

            # ── write map (the operand the dC score dot contracts) ──
            if HAS_THETA_H:
                Bop1 = ch_h[None, :] * Bt1 - sh_h[None, :] * Bt2
                Bop2 = ch_h[None, :] * Bt2 + sh_h[None, :] * Bt1
            else:
                Bop1 = Bt1
                Bop2 = Bt2
            Bdd1 = Bt1
            Bdd2 = Bt2
            if HAS_BETA and not BIAS_PRE_ROT:
                if HAS_THETA_H:
                    Bop1 = Bop1 + (ch_h * be1 - sh_h * be2)[None, :]
                    Bop2 = Bop2 + (ch_h * be2 + sh_h * be1)[None, :]
                else:
                    Bop1 = Bop1 + be1[None, :]
                    Bop2 = Bop2 + be2[None, :]
                Bdd1 = Bdd1 + be1[None, :]
                Bdd2 = Bdd2 + be2[None, :]

            # ── read map ─────────────────────────────────────────────
            Ceff1 = Ct1
            Ceff2 = Ct2
            if HAS_GAMMA and not BIAS_PRE_ROT:
                Ceff1 = Ceff1 + ga1[None, :]
                Ceff2 = Ceff2 + ga2[None, :]
            if HAS_THETA_H:
                Cop1 = ch_h[None, :] * Ceff1 + sh_h[None, :] * Ceff2
                Cop2 = ch_h[None, :] * Ceff2 - sh_h[None, :] * Ceff1
            else:
                Cop1 = Ceff1
                Cop2 = Ceff2

            dY_h = tl.load(
                dY_ptr + b * s_yb + h * s_yh + c * s_yc
                + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
            ).to(tl.float32)
            s_base = (SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                      + offs_gt[None, :] * s_sg)
            S1 = tl.load(s_base + offs_h1[:, None] * s_sn).to(tl.float32)
            S2 = tl.load(s_base + offs_h2[:, None] * s_sn).to(tl.float32)
            if HAS_THETA_H:
                S_op1 = ch_h[:, None] * S1 - sh_h[:, None] * S2
                S_op2 = ch_h[:, None] * S2 + sh_h[:, None] * S1
            else:
                S_op1 = S1
                S_op2 = S2

            dC_state1 = f[:, None] * pdot(
                dY_h.to(MMA_DT), tl.trans(S_op1).to(MMA_DT),
                input_precision=PREC)
            dC_state2 = f[:, None] * pdot(
                dY_h.to(MMA_DT), tl.trans(S_op2).to(MMA_DT),
                input_precision=PREC)

            x_base = X_ptr + b * s_xb + h * s_xh + c * T * s_xl
            x_s = tl.load(
                x_base + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
            ).to(tl.float32)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)
                x_s = x_s * dw[:, None]

            dScore = pdot(dY_h.to(MMA_DT), tl.trans(x_s).to(MMA_DT),
                            input_precision=PREC)
            dSb = dScore * dec_mask
            dSb_dt = dSb.to(MMA_DT)
            dSb_tr = tl.trans(dSb_dt)
            dC_h1 = pdot(dSb_dt, Bop1.to(MMA_DT),
                           input_precision=PREC) + dC_state1
            dC_h2 = pdot(dSb_dt, Bop2.to(MMA_DT),
                           input_precision=PREC) + dC_state2
            dB_h1 = pdot(dSb_tr, Cop1.to(MMA_DT), input_precision=PREC)
            dB_h2 = pdot(dSb_tr, Cop2.to(MMA_DT), input_precision=PREC)

            # SPANDELTA (`_SPANDELTA_NOTE`): this kernel owns only the READ
            # term of dL/ddelta (Kc owns the score pairs and the write
            # term); it reaches every s <= i -- the reverse cumsum below
            if HAS_THETA_H:
                dD = a_val * (tl.sum(Ceff1 * dC_state1, axis=1)
                              + tl.sum(Ceff2 * dC_state2, axis=1))
            else:
                dD = a_val * (tl.sum(Cop1 * dC_state1, axis=1)
                              + tl.sum(Cop2 * dC_state2, axis=1))
            dD = tl.cumsum(dD, axis=0, reverse=True)

            g_base = (dG_ptr + b * s_gb + h * s_gh + c * s_gc
                      + offs_gt[None, :] * s_gg)
            dG1 = tl.load(g_base + offs_h1[:, None] * s_gn).to(tl.float32)
            dG2 = tl.load(g_base + offs_h2[:, None] * s_gn).to(tl.float32)
            xw = x_s * dec_agg[:, None]
            dB_h1 += pdot(xw.to(MMA_DT), tl.trans(dG1).to(MMA_DT),
                            input_precision=PREC)
            dB_h2 += pdot(xw.to(MMA_DT), tl.trans(dG2).to(MMA_DT),
                            input_precision=PREC)

            if HAS_BETA and not BIAS_PRE_ROT:
                tl.atomic_add(dBeta_ptr + h * N + offs_h1,
                              tl.sum(dB_h1, axis=0), sem=ACC_SEM)
                tl.atomic_add(dBeta_ptr + h * N + offs_h2,
                              tl.sum(dB_h2, axis=0), sem=ACC_SEM)
            if HAS_GAMMA and not BIAS_PRE_ROT:
                tl.atomic_add(dGamma_ptr + h * N + offs_h1,
                              tl.sum(dC_h1, axis=0), sem=ACC_SEM)
                tl.atomic_add(dGamma_ptr + h * N + offs_h2,
                              tl.sum(dC_h2, axis=0), sem=ACC_SEM)
            if HAS_THETA_H:
                # d/dtheta_h of (C_1' = c C_1 + s C_2, C_2' = c C_2 - s C_1)
                # contracted with dC: C_2 dC_1 - C_1 dC_2, the half-layout
                # form of the interleaved sum_pair(sgn * dC * C_xor).
                tl.atomic_add(dThetaH_ptr + h * KH + offs_p,
                              tl.sum(dC_h1 * Ceff2 - dC_h2 * Ceff1, axis=0),
                              sem=ACC_SEM)
            tl.atomic_add(
                dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt,
                dD, sem=ACC_SEM)

            if HAS_ROT:
                # THETAFREQ: contract the FORWARD's MMA operands --
                # round(B~) and R(nu)^T round(C_h) (see `_opv`)
                Cq1 = Ct1
                Cq2 = Ct2
                if HAS_THETA_H:
                    Cq1, Cq2 = _ph_read_consistent(
                        ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2,
                        ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1,
                        ch_h, sh_h, Ct1, Ct2, MMA_DT, MMA_IO, HAS_THETA_H, PREC)
                else:
                    Cq1 = _opv(Ct1, MMA_DT, MMA_IO, PREC)
                    Cq2 = _opv(Ct2, MMA_DT, MMA_IO, PREC)
                dth_ph = (dB_h1 * _opv(Bt2, MMA_DT, MMA_IO, PREC)
                          - dB_h2 * _opv(Bt1, MMA_DT, MMA_IO, PREC)
                          + dC_h1 * Cq2 - dC_h2 * Cq1)          # [T, PS]
                dtp = (dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                       + offs_t[:, None] * s_dtt
                       + offs_p[None, :] * s_dtk)
                if PHASE_CUMSUM and not PHASE_DEFER:
                    dth_off = tl.sum(dth_ph, axis=0)
                    dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                    dop = (dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                           + offs_p * s_dok)
                    if dth_sole:
                        tl.store(dop, dth_off)
                    else:
                        tl.atomic_add(dop, dth_off, sem=ACC_SEM)
                if dth_sole:
                    tl.store(dtp, dth_ph)
                else:
                    tl.atomic_add(dtp, dth_ph, sem=ACC_SEM)
                # back to the RAW (un-rotated) frame
                nB1 = ph_cos * dB_h1 - ph_sin * dB_h2
                dB_h2 = ph_cos * dB_h2 + ph_sin * dB_h1
                dB_h1 = nB1
                nC1 = ph_cos * dC_h1 - ph_sin * dC_h2
                dC_h2 = ph_cos * dC_h2 + ph_sin * dC_h1
                dC_h1 = nC1
            if BIAS_PRE_ROT:
                bslot = c % BSLOT
                if HAS_BETA:
                    tl.atomic_add(dBeta_ptr + bslot * s_bes + h * N + offs_h1,
                                  tl.sum(dB_h1, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + bslot * s_bes + h * N + offs_h2,
                                  tl.sum(dB_h2, axis=0), sem=ACC_SEM)
                if HAS_GAMMA:
                    tl.atomic_add(dGamma_ptr + bslot * s_gas + h * N + offs_h1,
                                  tl.sum(dC_h1, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dGamma_ptr + bslot * s_gas + h * N + offs_h2,
                                  tl.sum(dC_h2, axis=0), sem=ACC_SEM)

            dB1_acc += dB_h1
            dB2_acc += dB_h2
            dC1_acc += dC_h1
            dC2_acc += dC_h2

        sole_owner = (NGT_val == 1) & (h_start == 0) & (hb_count >= H)
        db_p = (dBtil_ptr + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt)
        dc_p = (dCtil_ptr + b * s_dcb + c * s_dcc + offs_t[:, None] * s_dct)
        if sole_owner:
            tl.store(db_p + offs_h1[None, :] * s_dbn, dB1_acc)
            tl.store(db_p + offs_h2[None, :] * s_dbn, dB2_acc)
            tl.store(dc_p + offs_h1[None, :] * s_dcn, dC1_acc)
            tl.store(dc_p + offs_h2[None, :] * s_dcn, dC2_acc)
        else:
            tl.atomic_add(db_p + offs_h1[None, :] * s_dbn, dB1_acc, sem=ACC_SEM)
            tl.atomic_add(db_p + offs_h2[None, :] * s_dbn, dB2_acc, sem=ACC_SEM)
            tl.atomic_add(dc_p + offs_h1[None, :] * s_dcn, dC1_acc, sem=ACC_SEM)
            tl.atomic_add(dc_p + offs_h2[None, :] * s_dcn, dC2_acc, sem=ACC_SEM)
    else:
        ROT: tl.constexpr = HAS_THETA_H or HAS_ROT_BIAS
        ROT_ANY: tl.constexpr = ROT or HAS_ROT
        if ROT_ANY:
            pair_idx = offs_n // 2
            fwd_sign = tl.where((offs_n % 2) == 0, 1.0, -1.0)

        # Head-shared tiles (ungrouped: B/C carry no head axis).  The xor-lane
        # partners are head-shared too, so they are hoisted with them --
        # EXCEPT under ROT_PER_HEAD, where the ROTATED tile differs per head
        # and only the RAW load can be hoisted.
        c_base = C_ptr + b * s_cb + c * s_cc
        b_base = B_ptr + b * s_bb + c * s_bc
        Co_base = tl.load(
            c_base + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn,
        ).to(tl.float32)
        Bc_base = tl.load(
            b_base + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn,
        ).to(tl.float32)
        if HAS_ROT and not ROT_PER_HEAD:
            # B~/C~ built HERE instead of by the host fold; theta is
            # head-shared, so this is once per CTA, not once per head.
            th_rot = tl.load(
                Theta_ptr + b * s_thb + c * s_thc
                + offs_t[:, None] * s_tht
                + offs_k[None, :] * s_thk,
            ).to(tl.float32)                                        # [T, N//2]
            if PHASE_CUMSUM:
                th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                    ThOff_ptr + b * s_ob + c * s_oc
                    + offs_k * s_ok,
                ).to(tl.float32)[None, :]
            cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, NS)
            sn_rot = fwd_sign[None, :] * _pair_bcast_2d(
                _sin_approx(th_rot), T, NS)
            Co_base = cs_rot * Co_base + sn_rot * _pair_swap_2d(Co_base, T, NS)
            Bc_base = cs_rot * Bc_base + sn_rot * _pair_swap_2d(Bc_base, T, NS)
        if ROT and not ROT_PER_HEAD:
            if HAS_ROT:
                # the partner of the ROTATED tile, which is in registers
                Co_xor = _pair_swap_2d(Co_base, T, NS)
                Bc_xor = _pair_swap_2d(Bc_base, T, NS)
            else:
                Co_xor = tl.load(
                    c_base + offs_t[:, None] * s_ct
                    + (offs_n ^ 1)[None, :] * s_cn,
                ).to(tl.float32)
                Bc_xor = tl.load(
                    b_base + offs_t[:, None] * s_bt
                    + (offs_n ^ 1)[None, :] * s_bn,
                ).to(tl.float32)

        # dTheta's only fan-in under a PER-HEAD phase is the G-tile axis: the
        # head block owns each h exactly once, and (b, c) is this CTA's.
        dth_sole = NGT_val == 1

        dB_acc = tl.zeros([T, NS], dtype=tl.float32)
        dC_acc = tl.zeros([T, NS], dtype=tl.float32)
        if KBC_PART != 0:
            tl.static_assert(not ROT_ANY and not ROT_PER_HEAD and not SUMDS
                             and not BIAS_PRE_ROT and not KBC_DALPHA
                             and not DD_PAIRS and ROT_ABL == 0,
                             "KBC_PART: the unrotated fused body only")
        if SUMDS:
            tl.static_assert(
                not ROT_ANY and not ROT_PER_HEAD and not BIAS_PRE_ROT
                and ROT_ABL == 0,
                "SUMDS: head-shared unrotated B/C only (the right operand of "
                "the score contractions must be the same for every head)")
            dS_sum = tl.zeros([T, T], dtype=tl.float32)

        h_start = hblk * HB
        hb_count = tl.minimum(HB, H - h_start)
        for hh in range(hb_count):
            h = h_start + hh
            if IDX64:
                h = h.to(tl.int64)
            a_val = -tl.exp(tl.load(A_ptr + h))

            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            if not SCHED_DM:
                dec_mask = tl.exp(
                    tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                             neg_inf))
            if not SCHED_AG:
                D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
                dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)

            # PER-HEAD phase: B~/C~ formed here, one head at a time.
            if ROT_PER_HEAD and (ROT_ABL & 384) != 0:
                rbase = (b * s_rb + c * s_rc + h * s_rh
                         + offs_t[:, None] * s_rt + offs_n[None, :] * s_rn)
                Co_cur = tl.load(CRot_ptr + rbase).to(tl.float32)
                if (ROT_ABL & 256) != 0:
                    Bc_cur = tl.load(BRot_ptr + rbase).to(tl.float32)
            if ROT_PER_HEAD and (ROT_ABL & 256) == 0:
                if False:
                    th_rot = tl.zeros([T, NS // 2], dtype=tl.float32)
                else:
                    th_rot = tl.load(
                        Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                        + offs_t[:, None] * s_tht
                        + offs_k[None, :] * s_thk,
                    ).to(tl.float32)                                # [T, N//2]
                if PHASE_CUMSUM and (ROT_ABL & 1) == 0:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + offs_k * s_ok,
                    ).to(tl.float32)[None, :]
                # ── the ROT_PER_HEAD REGISTER DIET ────────────────────
                # The per-head phase is kept in its PAIR form [T, NS//2] and
                # broadcast at each use; the broadcast is a pure layout op
                # (`tl.join(v, v)`), so this is BIT-IDENTICAL and halves what
                # the phase costs in live registers across the five GEMMs.
                if (ROT_ABL & 2) != 0:
                    ph_cos = th_rot * 0.0 + 1.0
                    ph_sin = th_rot * 0.0
                else:
                    ph_cos = _cos_approx(th_rot)                    # [T, NS//2]
                    ph_sin = _sin_approx(th_rot)                    # [T, NS//2]
                cs_rot = _pair_bcast_2d(ph_cos, T, NS)
                sn_rot = fwd_sign[None, :] * _pair_bcast_2d(ph_sin, T, NS)
                Bc_src = Bc_base
                Co_src = Co_base
                if BIAS_PRE_ROT:
                    # Mamba-3's fold ORDER: the bias joins the RAW tile and
                    # rotates WITH it, so B~ = R(theta)^T (B + beta).
                    if HAS_BETA:
                        Bc_src = Bc_src + tl.load(
                            Beta_ptr + h * N + offs_n).to(tl.float32)[None, :]
                    if HAS_GAMMA:
                        Co_src = Co_src + tl.load(
                            Gamma_ptr + h * N + offs_n).to(tl.float32)[None, :]
                # ADJTHH, mirroring ROTBWD in Kb1: the
                # XOR-lane partner of the ROTATED tile is ALGEBRAIC.  With
                #   B~ = cs B + sn swap(B),  swap(cs) = cs (pair-broadcast),
                #   swap(sn) = -sn (sn carries the pair sign), so
                #   swap(B~) = cs swap(B) - sn B
                # from the two tiles the build already holds live.  That
                # removes the SECOND `_pair_swap_2d` per tile per head -- a
                # cross-lane layout op ([T, N] fp32 through shared memory),
                # which ROTBWD measured as the dTheta path's real
                # cost -- for two FMAs.  Bit-identical: a permutation
                # commutes with an elementwise FMA and the operand order is
                # preserved.
                if (ROT_ABL & 16) != 0:
                    Bc_cur = Bc_src
                    Co_cur = Co_src
                    if ROT or not ROT_DIET:
                        Bc_xor = _pair_swap_2d(Bc_cur, T, NS)
                        Co_xor = _pair_swap_2d(Co_cur, T, NS)
                else:
                    B_sw = _pair_swap_2d(Bc_src, T, NS)
                    Bc_cur = cs_rot * Bc_src + sn_rot * B_sw
                    if ROT or not ROT_DIET:
                        Bc_xor = cs_rot * B_sw - sn_rot * Bc_src
                    if (ROT_ABL & 128) == 0:
                        C_sw = _pair_swap_2d(Co_src, T, NS)
                        Co_cur = cs_rot * Co_src + sn_rot * C_sw
                        if ROT or not ROT_DIET:
                            Co_xor = cs_rot * C_sw - sn_rot * Co_src
                    elif ROT or not ROT_DIET:
                        # Co_cur is the UNROTATED tile on this ablation arm
                        Co_xor = _pair_swap_2d(Co_cur, T, NS)
            else:
                Bc_cur = Bc_base
                Co_cur = Co_base

            # ── per-head rotation coefficients ───────────────────────────
            # p1/p2 combine BOTH rotations into one 2-tile blend:
            #   (B_eff_rd M_th)  = p1 * B    + p2 * B_xor  (+ M_th applied to beta)
            #   (C_eff_rd M_rho) = p1 * Ceff - p2 * Ceff_xor
            if HAS_THETA_H:
                th_h = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)
                ch_h = _cos_approx(th_h)
                fc_h = fwd_sign * _sin_approx(th_h)
            if HAS_ROT_BIAS:
                rb_h = tl.load(RotBias_ptr + h * (N // 2) + pair_idx)
                crb_h = _cos_approx(rb_h)
                sfb_h = fwd_sign * _sin_approx(rb_h)
            if HAS_THETA_H and HAS_ROT_BIAS:
                p1 = ch_h * crb_h + fc_h * sfb_h
                p2 = ch_h * sfb_h - fc_h * crb_h
            elif HAS_THETA_H:
                p1 = ch_h
                p2 = -fc_h
            elif HAS_ROT_BIAS:
                p1 = crb_h
                p2 = sfb_h

            # ── write map ────────────────────────────────────────────────
            if ROT:
                Bop = p1[None, :] * Bc_cur + p2[None, :] * Bc_xor
            else:
                Bop = Bc_cur
            Bdd = Bc_cur
            if HAS_BETA and not BIAS_PRE_ROT:
                beta_v = tl.load(Beta_ptr + h * N + offs_n).to(tl.float32)
                if ROT:
                    beta_x = tl.load(
                        Beta_ptr + h * N + (offs_n ^ 1)).to(tl.float32)
                if HAS_THETA_H:
                    Bop = Bop + (ch_h * beta_v - fc_h * beta_x)[None, :]
                else:
                    Bop = Bop + beta_v[None, :]
                if HAS_ROT_BIAS:
                    # dDelta wants sum(B_eff_rd * dB_rd), and beta does NOT
                    # rotate: the unrotated form carries M_rho^T beta.
                    Bdd = Bdd + (crb_h * beta_v - sfb_h * beta_x)[None, :]
                else:
                    Bdd = Bdd + beta_v[None, :]

            # ── read map ─────────────────────────────────────────────────
            Ceff = Co_cur
            if ROT:
                Ceff_x = Co_xor
            if HAS_GAMMA and not BIAS_PRE_ROT:
                gam = tl.load(Gamma_ptr + h * N + offs_n).to(tl.float32)
                Ceff = Ceff + gam[None, :]
                if ROT:
                    gam_x = tl.load(
                        Gamma_ptr + h * N + (offs_n ^ 1)).to(tl.float32)
                    Ceff_x = Ceff_x + gam_x[None, :]
            if ROT:
                Cop = p1[None, :] * Ceff - p2[None, :] * Ceff_x
            else:
                Cop = Ceff

            dY_h = tl.load(
                dY_ptr + b * s_yb + h * s_yh + c * s_yc
                + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
            ).to(tl.float32)
            s_base = SAll_ptr + b * s_sb + h * s_sh + c * s_sc
            if TS_ABL & 1:
                S_c = tl.load(
                    s_base + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg,
                    mask=(offs_n[:, None] >= 0) & (c % 2 == 0),
                    other=0.0).to(tl.float32)
            else:
                S_c = tl.load(
                    s_base + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg,
                ).to(tl.float32)
            if TS_ABL & 2:
                if c % 2 == 1:
                    x_pv = tl.load(
                        X_ptr + b * s_xb + h * s_xh + (c - 1) * T * s_xl
                        + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                    ).to(MMA_DT)
                    S_c = S_c * 0.5 + pdot(tl.trans(Bop.to(MMA_DT)), x_pv,
                                              input_precision=PREC)
            if HAS_THETA_H:
                S_x = tl.load(
                    s_base + (offs_n ^ 1)[:, None] * s_sn
                    + offs_gt[None, :] * s_sg,
                ).to(tl.float32)
                S_op = ch_h[:, None] * S_c - fc_h[:, None] * S_x
            else:
                S_op = S_c

            # ── score path (accumulators in the UNROTATED frame) ──
            # `SF`: the state dot is taken HERE, before the `x` tile is
            # loaded, so `S_op` dies before `x_s` is born.  Same dot, same
            # operands, same order.
            if SCHED_SF:
                dC_state = f[:, None] * pdot(
                    dY_h.to(MMA_DT), tl.trans(S_op).to(MMA_DT),
                    input_precision=PREC)
            x_base = X_ptr + b * s_xb + h * s_xh + c * T * s_xl
            x_s = tl.load(
                x_base + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
            ).to(tl.float32)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)
                x_s = x_s * dw[:, None]

            if not SCHED_SF:
                dC_state = f[:, None] * pdot(
                    dY_h.to(MMA_DT), tl.trans(S_op).to(MMA_DT),
                    input_precision=PREC)
            dScore = pdot(dY_h.to(MMA_DT), tl.trans(x_s).to(MMA_DT), input_precision=PREC)
            if SCHED_DM:
                # `DM`: built at its ONLY consumer.  Same exponent, same
                # difference-first order (contract 7.2.1) -- bitwise the same
                # tile, thirty registers later in the body.
                dec_mask = tl.exp(
                    tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                             neg_inf))
            dSb = dScore * dec_mask
            if DD_PAIRS:
                tl.static_assert(not ROT and not ROT_PER_HEAD
                                 and not HAS_BETA and not HAS_GAMMA,
                                 "DD_PAIRS: unrotated, bias-free (S0 = C B^T)")
                S0_dd = tl.load(CB_ptr + b * s_cbb + c * s_cbc
                                + offs_t[:, None] * T + offs_t[None, :])
                dd_pairs = _span_sum(S0_dd * dSb, offs_t) * tl.where(
                    ns_idx == 0, 1.0, 0.0)
            if SUMDS:
                # the score contractions move after the head loop; this
                # head keeps only its exact rank-1 bias corrections
                dS_sum += dSb
                dC_h = dC_state
                dB_h = tl.zeros([T, NS], dtype=tl.float32)
                if HAS_BETA or HAS_GAMMA:
                    rs_h = tl.sum(dSb, axis=1)                      # dSb 1
                    cs_h = tl.sum(dSb, axis=0)                      # dSb^T 1
                if HAS_BETA:
                    dC_h += rs_h[:, None] * beta_v[None, :]
                if HAS_GAMMA:
                    dB_h += cs_h[:, None] * gam[None, :]
            else:
                # dSb feeds one dot STRAIGHT and one TRANSPOSED -- ONE
                # cast, before the transpose (CAST-BEFORE-TRANSPOSE).
                dSb_dt = dSb.to(MMA_DT)
                dC_h = pdot(dSb_dt, Bop.to(MMA_DT), input_precision=PREC) + dC_state
                dB_h = pdot(tl.trans(dSb_dt), Cop.to(MMA_DT), input_precision=PREC)

            # SPANDELTA (`_SPANDELTA_NOTE`): this kernel owns only the READ
            # term of dL/ddelta (Kc owns the score pairs and the write
            # term); it reaches every s <= i
            if ROT:
                dD = a_val * tl.sum(Ceff * dC_state, axis=1)
            else:
                dD = a_val * tl.sum(Cop * dC_state, axis=1)
            dD = tl.cumsum(dD, axis=0, reverse=True)
            if DD_PAIRS:
                dD += a_val * dd_pairs

            # ── aggregate path (moved from Kc's FUSE_KB2 dB_agg) ──
            g_base = dG_ptr + b * s_gb + h * s_gh + c * s_gc
            if TS_ABL & 1:
                dG = tl.load(
                    g_base + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg,
                    mask=(offs_n[:, None] >= 0) & (c % 2 == 1),
                    other=0.0).to(tl.float32)
            else:
                dG = tl.load(
                    g_base + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg,
                ).to(tl.float32)
            if TS_ABL & 2:
                if c % 2 == 0:
                    dY_nx = tl.load(
                        dY_ptr + b * s_yb + h * s_yh
                        + tl.minimum(c + 1, NC - 1) * s_yc
                        + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                    ).to(MMA_DT)
                    dG = dG * 0.5 + pdot(tl.trans(Cop.to(MMA_DT)), dY_nx,
                                            input_precision=PREC)
            if KBC_DALPHA:
                tl.static_assert(not ROT_ANY and not ROT_PER_HEAD,
                                 "KBC_DALPHA: unrotated state tiles only")
                tl.atomic_add(dAlpha_ptr + b * s_dab + h * s_dah
                              + gt_idx * s_dag + c * s_dac,
                              tl.sum(dG * S_c), sem=ACC_SEM)
            if HAS_ROT_BIAS:
                dG_x = tl.load(
                    g_base + (offs_n ^ 1)[:, None] * s_gn
                    + offs_gt[None, :] * s_gg,
                ).to(tl.float32)
                dG_op = crb_h[:, None] * dG - sfb_h[:, None] * dG_x
            else:
                dG_op = dG
            if SCHED_AG:
                D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
                dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            if SCHED_XR:
                # `XR`: the aggregate path RE-READS `x` rather than holding the
                # [T, GT] fp32 tile across the two score MMAs.
                # `volatile` is what stops Triton folding the second load
                # back into the first; the bytes and the scale are the same,
                # so `xw` is bitwise what the held tile would have given.
                x_a = tl.load(
                    x_base + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                    volatile=True,
                ).to(tl.float32)
                if HAS_DW:
                    x_a = x_a * dw[:, None]
                xw = x_a * dec_agg[:, None]
            else:
                xw = x_s * dec_agg[:, None]
            dB_h += pdot(xw.to(MMA_DT), tl.trans(dG_op).to(MMA_DT), input_precision=PREC)

            if HAS_BETA and not BIAS_PRE_ROT and KBC_PART != 1:
                v_beta = tl.sum(dB_h, axis=0)
                if SUMDS:
                    # the head's share of colsum(dSb^T @ C) = rs_h^T C
                    v_beta += tl.sum(rs_h[:, None] * Co_base, axis=0)
                if HAS_ROT_BIAS:
                    # dBeta = M_rho @ (sum_t dB): the two-address write lands
                    # on the [N] VECTOR, not on the [T, N] tile.
                    tl.atomic_add(dBeta_ptr + h * N + offs_n, crb_h * v_beta, sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + h * N + (offs_n ^ 1),
                                  -sfb_h * v_beta, sem=ACC_SEM)
                else:
                    tl.atomic_add(dBeta_ptr + h * N + offs_n, v_beta, sem=ACC_SEM)
            if HAS_GAMMA and not BIAS_PRE_ROT and KBC_PART != 2:
                # gamma co-rotates with C, so the UNROTATED dC sum IS dGamma.
                v_gam = tl.sum(dC_h, axis=0)
                if SUMDS:
                    # the head's share of colsum(dSb @ B) = cs_h^T B
                    v_gam += tl.sum(cs_h[:, None] * Bc_base, axis=0)
                tl.atomic_add(dGamma_ptr + h * N + offs_n, v_gam, sem=ACC_SEM)
            if HAS_THETA_H:
                tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx,
                              tl.sum(fwd_sign[None, :] * dC_h * Ceff_x, axis=0), sem=ACC_SEM)
            if HAS_ROT_BIAS:
                # B WITHOUT beta: beta is added after the write rotation.
                tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx,
                              tl.sum(fwd_sign[None, :] * dB_h * Bc_xor, axis=0), sem=ACC_SEM)
            if KBC_PART != 2:
                tl.atomic_add(
                    dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt,
                    dD, sem=ACC_SEM)

            if ROT_PER_HEAD and (ROT_ABL & 384) != 0:
                # what B0 leaves in the loop: the per-head gradient tiles go
                # OUT, in the rotated frame, for the un-rotation kernel.
                tl.store(CRot_ptr + rbase, dC_h.to(CRot_ptr.dtype.element_ty))
                if (ROT_ABL & 256) != 0:
                    tl.store(BRot_ptr + rbase,
                             dB_h.to(BRot_ptr.dtype.element_ty))
            if ROT_PER_HEAD and (ROT_ABL & 260) == 0:
                # dTheta for THIS head, in the rotated frame, then the two
                # accumulators back to the raw frame -- AFTER every atomic
                # above, each of which is defined on the rotated tiles.
                if ROT_DIET and not ROT:
                    # formed HERE, not at the top of the body: Bc_cur /
                    # Co_cur are GEMM operands and are live anyway, and a
                    # pair swap is a register layout op.  Same values.
                    Bc_xor = _pair_swap_2d(Bc_cur, T, NS)
                    Co_xor = _pair_swap_2d(Co_cur, T, NS)
                # ADJTHH: the pair sign moves OUT of the [T, N] product and
                # into the pair split (the last row).  `fwd_sign`
                # is +1 on the even lane and -1 on the odd one, so
                # sum_pair(sign * P) = P_even - P_odd: one [T, N] multiply
                # per head less, bit-identical (a per-lane constant sign).
                # THETAFREQ: contract the FORWARD's MMA operands -- round(B~)
                # and, under theta_h, R(nu)^T round(C_h), C_h = ch C~ + fc
                # swap(C~) (fc pair-odd): partner = ch round(swap C_h)
                # + fc round(C_h).  FMAs only (see `_opv`).
                Cq_xor = Co_xor
                if (MMA_IO or PREC == "tf32") and DTH_OPROUND:
                    if HAS_THETA_H and not HAS_ROT_BIAS:
                        Cq_xor = (ch_h[None, :] * _opv(
                            ch_h[None, :] * Co_xor - fc_h[None, :] * Co_cur,
                            MMA_DT, MMA_IO, PREC)
                            + fc_h[None, :] * _opv(
                                ch_h[None, :] * Co_cur + fc_h[None, :] * Co_xor,
                                MMA_DT, MMA_IO, PREC))
                    else:
                        Cq_xor = _opv(Co_xor, MMA_DT, MMA_IO, PREC)
                dth_lane = dB_h * _opv(Bc_xor, MMA_DT, MMA_IO, PREC) + dC_h * Cq_xor
                dth_e, dth_o = tl.split(tl.reshape(dth_lane, (T, NS // 2, 2)))
                dth_ph = dth_e - dth_o                              # [T, N//2]
                dtp = (dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                       + offs_t[:, None] * s_dtt
                       + offs_k[None, :] * s_dtk)
                if PHASE_CUMSUM:
                    # theta_t = off_c + sum_{t' <= t} incr_t', so
                    #   dL/d incr_t = sum_{t' >= t} dL/d theta_t'   (in-chunk)
                    #   dL/d off_c  = sum_t dL/d theta_t
                    # -- a reverse prefix sum on the tile already in registers
                    # and one row reduction; both LINEAR, so the owner/atomic
                    # rule below is unchanged.
                    if not PHASE_DEFER:
                        dth_off = tl.sum(dth_ph, axis=0)             # [N//2]
                        if (ROT_ABL & 512) == 0:
                            dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        dop = (dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                               + offs_k * s_dok)
                        if dth_sole:
                            tl.store(dop, dth_off)
                        else:
                            tl.atomic_add(dop, dth_off, sem=ACC_SEM)
                if (ROT_ABL & 64) == 0:
                    if dth_sole:
                        tl.store(dtp, dth_ph)
                    else:
                        tl.atomic_add(dtp, dth_ph, sem=ACC_SEM)
            if ROT_PER_HEAD and (ROT_ABL & 256) == 0:
                if (ROT_ABL & 16) == 0:
                    if ROT_DIET:
                        # rebuilt from the [T, NS//2] pair form rather than
                        # kept live across the GEMMs -- identical values.
                        cs_rot = _pair_bcast_2d(ph_cos, T, NS)
                        sn_rot = (fwd_sign[None, :]
                                  * _pair_bcast_2d(ph_sin, T, NS))
                    dB_h = cs_rot * dB_h - sn_rot * _pair_swap_2d(dB_h, T, NS)
                    dC_h = cs_rot * dC_h - sn_rot * _pair_swap_2d(dC_h, T, NS)
                if BIAS_PRE_ROT and (ROT_ABL & 8) == 0:
                    # beta/gamma sit INSIDE the rotation, so their gradient is
                    # the sum of the UN-rotated dB / dC -- which is why these
                    # two atomics wait until after the line above.
                    bslot = c % BSLOT
                    if HAS_BETA:
                        vb = tl.sum(dB_h, axis=0)
                        tl.atomic_add(dBeta_ptr + bslot * s_bes
                                      + h * N + offs_n, vb, sem=ACC_SEM)
                    if HAS_GAMMA:
                        vg = tl.sum(dC_h, axis=0)
                        tl.atomic_add(dGamma_ptr + bslot * s_gas
                                      + h * N + offs_n, vg, sem=ACC_SEM)

            if KBC_PART != 1:
                dB_acc += dB_h
            if (ROT_ABL & 384) == 0 and KBC_PART != 2:
                dC_acc += dC_h

        if SUMDS:
            # the block's two score contractions, ONCE (see SUMDS above)
            dS_dt = dS_sum.to(MMA_DT)
            dC_acc += pdot(dS_dt, Bc_base.to(MMA_DT), input_precision=PREC)
            dB_acc += pdot(tl.trans(dS_dt), Co_base.to(MMA_DT),
                             input_precision=PREC)

        sole_owner = (NGT_val == 1) & (h_start == 0) & (hb_count >= H)

        if HAS_ROT and not ROT_PER_HEAD:
            # dTheta in the ROTATED frame: d/dtheta of (e' = c e + s o,
            # o' = -s e + c o) is (o', -e'), so the pair contribution is
            # dB~_e B~_o - dB~_o B~_e -- the two lanes of a pair summed.
            # Bc_xor / Co_xor are the B~ / C~ partners (no beta, no gamma:
            # both are added AFTER this rotation, so neither sees theta).
            if ROT:
                Bx_r = Bc_xor
                Cx_r = Co_xor
            else:
                Bx_r = _pair_swap_2d(Bc_base, T, NS)
                Cx_r = _pair_swap_2d(Co_base, T, NS)
            if not HAS_THETA_H:
                Cx_r = _opv(Cx_r, MMA_DT, MMA_IO, PREC)                 # THETAFREQ
            dth_lane = fwd_sign[None, :] * (dB_acc * _opv(Bx_r, MMA_DT, MMA_IO, PREC)
                                            + dC_acc * Cx_r)
            dth_e, dth_o = tl.split(tl.reshape(dth_lane, (T, NS // 2, 2)))
            dth = dth_e + dth_o                                   # [T, N//2]
            # and the accumulators back to the UNROTATED frame
            dB_acc = cs_rot * dB_acc - sn_rot * _pair_swap_2d(dB_acc, T, NS)
            dC_acc = cs_rot * dC_acc - sn_rot * _pair_swap_2d(dC_acc, T, NS)
            dt_ptr = (dTheta_ptr + b * s_dtb + c * s_dtc
                      + offs_t[:, None] * s_dtt
                      + offs_k[None, :] * s_dtk)
            if PHASE_CUMSUM:
                # same reparameterisation adjoint as the per-head branch
                if not PHASE_DEFER:
                    dth_off = tl.sum(dth, axis=0)                    # [N//2]
                    dth = tl.cumsum(dth, axis=0, reverse=True)
                    dop = (dThOff_ptr + b * s_dob + c * s_doc
                           + offs_k * s_dok)
                    if sole_owner:
                        tl.store(dop, dth_off)
                    else:
                        tl.atomic_add(dop, dth_off, sem=ACC_SEM)
            if sole_owner:
                tl.store(dt_ptr, dth)
            else:
                tl.atomic_add(dt_ptr, dth, sem=ACC_SEM)

        if sole_owner:
            if KBC_PART != 1:
                tl.store(
                    dBtil_ptr + b * s_dbb + c * s_dbc
                    + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn,
                    dB_acc)
            if KBC_PART != 2:
                tl.store(
                    dCtil_ptr + b * s_dcb + c * s_dcc
                    + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn,
                    dC_acc)
        else:
            if KBC_PART != 1:
                tl.atomic_add(
                    dBtil_ptr + b * s_dbb + c * s_dbc
                    + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn,
                    dB_acc, sem=ACC_SEM)
            if KBC_PART != 2:
                tl.atomic_add(
                    dCtil_ptr + b * s_dcb + c * s_dcc
                    + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn,
                    dC_acc, sem=ACC_SEM)


@triton.autotune(configs=_calib_grid("fused_kb2", _bwd_grid),
                 prune_configs_by={"early_config_prune": _cprune("fused_kb2")},
                 key=["T", "N", "G_SIZE", "GT", "PREC", "MMA_IO",
                      "HAS_BETA", "HAS_ROT_BIAS", "HAS_ROT", "BIAS_PRE_ROT",
                      "PHASE_CUMSUM", "R", "HAS_DW", "IDX64", "ROT_ABL",
                      "PAIR_HALF",
                      "OCC"],
                 restore_value=["dBtil_ptr", "dDelta_ptr", "dBeta_ptr",
                                "dRotBias_ptr", "dTheta_ptr", "dThOff_ptr"])
@triton.jit
def _bwd_kb2_kernel(
    B_ptr, X_ptr, Delta_ptr, A_ptr,
    Beta_ptr, RotBias_ptr,
    dG_ptr, dAlpha_ptr,
    dBtil_ptr, dDelta_ptr, dBeta_ptr, dRotBias_ptr,
    H,
    s_bb, s_bc, s_bt, s_bn,
    s_xb, s_xh, s_xg, s_xr, s_xl,
    s_db, s_dh, s_dc, s_dt,
    s_gb, s_gh, s_gc, s_gn, s_gg,
    s_dab, s_dah, s_dag, s_dac,
    s_dbb, s_dbc, s_dbt, s_dbn,
    s_bh,
    NC, NGT_val,
    T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr,
    GT: tl.constexpr, PREC: tl.constexpr,
    # `tl.dot` operand dtype: fp32 = identity cast; bf16 / fp16 = IO-dtype MMA
    # with fp32 accumulate.  See `mma_operand_dtype` in fused_fwd_kernel.py.
    MMA_DT: tl.constexpr = tl.float32,
    # Autotune KEY stand-in for MMA_DT (a tl.dtype is not serialisable into
    # Triton's disk cache key): 1 iff MMA_DT is 16-bit.
    MMA_IO: tl.constexpr = 0,
    HAS_BETA: tl.constexpr = False,
    HAS_ROT_BIAS: tl.constexpr = False,
    HPG_B: tl.constexpr = 1,
    R: tl.constexpr = 1,
    s_br: tl.constexpr = 0,
    s_dbr: tl.constexpr = 0,
    R_TILE: tl.constexpr = False,
    CHUNK_PARALLEL: tl.constexpr = False,
    DW_ptr=None, s_wb=0, s_wh=0, s_wl=0,
    HAS_DW: tl.constexpr = False,
    s_dbh=0,
    IDX64: tl.constexpr = False,
    # ── the in-kernel phase on the AGGREGATE write path ─────────
    Theta_ptr=None,
    s_thb=0, s_thc=0, s_tht=0, s_thh=0, s_thk=0,
    dTheta_ptr=None,
    s_dtb=0, s_dth=0, s_dtc=0, s_dtt=0, s_dtk=0,
    HAS_ROT: tl.constexpr = False,
    BIAS_PRE_ROT: tl.constexpr = False,
    ThOff_ptr=None,
    s_ob=0, s_oh=0, s_oc=0, s_ok=0,
    dThOff_ptr=None,
    s_dob=0, s_doh=0, s_doc=0, s_dok=0,
    PHASE_CUMSUM: tl.constexpr = False,
    # PHASE_DEFER: the kernel stores the RAW dL/dtheta_cum and one
    # chunk-parallel pass (`m3_epilogue.phase_prefix_bwd`) does the in-chunk
    # reverse cumsum, the chunk total and the cross-chunk term (an in-kernel
    # cross-thread scan per head is too costly in this register-starved body).
    PHASE_DEFER: tl.constexpr = False,
    # ROTBWD: see `_bwd_kb1_kernel`'s ROT_ABL (measurement only, 0 = off).
    ROT_ABL: tl.constexpr = 0,
    # PAIRHALF: the rotate-half pair layout (see `fused_fwd_kernel.py`).
    PAIR_HALF: tl.constexpr = False,
    # `Delta_ptr` holds the RAW chunked delta; this kernel forms the
    # chunk-local prefix D_i itself (`_dcs_tile` / `_dcs_end`).
    DCS_INK: tl.constexpr = False,
    OCC=0,           # autotune-key occupancy bucket only (`occ_bucket`)
):
    """Kb2: aggregate-path gradients → partial dB + partial dDelta + dalpha.

    2 GEMMs/chunk, peak 4 live tiles.  No C_rd, no dY, no S_c, no theta.
    Reads dG_all and dalpha_all from Ka.

    R_TILE: when True, grid dim-2 is NGT*R; each CTA handles one R stream.
    CHUNK_PARALLEL: when True, grid dim-2 includes nc; each CTA does 1 chunk.
    """
    ABL_NO_DTH: tl.constexpr = (ROT_ABL & 1) != 0
    ABL_PREROT: tl.constexpr = (ROT_ABL & 2) != 0
    b = tl.program_id(0)
    h = tl.program_id(1)
    if CHUNK_PARALLEL:
        if R_TILE:
            c = tl.program_id(2) // (NGT_val * R)
            remainder = tl.program_id(2) % (NGT_val * R)
            r_fixed = remainder // NGT_val
            gt_idx = remainder % NGT_val
        else:
            c = tl.program_id(2) // NGT_val
            gt_idx = tl.program_id(2) % NGT_val
    else:
        if R_TILE:
            r_fixed = tl.program_id(2) // NGT_val
            gt_idx = tl.program_id(2) % NGT_val
        else:
            gt_idx = tl.program_id(2)

    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)

    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    bg = h // HPG_B

    ROT_ANY_KB2: tl.constexpr = HAS_ROT_BIAS or HAS_ROT
    if ROT_ANY_KB2:
        pair_idx_kb2 = offs_n // 2
        rot_sign_kb2 = tl.where((offs_n % 2) == 0, 1.0, -1.0)
    if HAS_ROT_BIAS:
        rb_full_kb2 = tl.load(RotBias_ptr + h * (N // 2) + pair_idx_kb2)
        crb_kb2 = _cos_approx(rb_full_kb2)
        srb_fwd_kb2 = rot_sign_kb2 * _sin_approx(rb_full_kb2)

    nc_end_kb2 = 1 if CHUNK_PARALLEL else NC
    if PAIR_HALF:
        # ═══ PAIRHALF: Kb2 on the ROTATE-HALF layout ═══════════════════
        # Kb2 is Kb1's aggregate-path twin: 2 GEMMs per chunk, ONE
        # `dTheta` pair term per r instead of Kb1's two, one atomic.  The
        # per-token phase gradient dominates its rotation cost, and this
        # layout makes it free:
        #   dTheta_k = sum_r (dB~_1 B~_2 - dB~_2 B~_1)
        # The aggregate `x w dG^T` has the state as an MMA OUTPUT axis
        # (two half-width MMAs per r); the `B~ dG` product that feeds
        # `q` has it as a CONTRACTION (the sum of the two halves' dots).
        # No `_pair_swap_2d`, no `_pair_bcast_2d`, no `offs_n ^ 1`.
        tl.static_assert(not HAS_ROT_BIAS,
                         "PAIR_HALF does not serve rot_bias")
        tl.static_assert(ROT_ABL == 0,
                         "PAIR_HALF carries no ablation ladder")
        tl.static_assert(N >= 32,
                         "PAIR_HALF needs N/2 >= 16 (the MMA K floor)")
        KH: tl.constexpr = N // 2
        offs_h1 = tl.arange(0, KH)
        offs_h2 = KH + tl.arange(0, KH)
        R_ITERS: tl.constexpr = 1 if R_TILE else R

        for ci in range(nc_end_kb2):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci

            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)
            if HAS_ROT:
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk,
                ).to(tl.float32)                                # [T, N/2]
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + offs_h1 * s_ok,
                    ).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
                dth_acc = tl.zeros([T, KH], dtype=tl.float32)

            dg_base = (dG_ptr + b * s_gb + h * s_gh + c * s_gc
                       + offs_gt[None, :] * s_gg)
            dG1_dt = tl.load(dg_base + offs_h1[:, None] * s_gn
                             ).to(tl.float32).to(MMA_DT)
            dG2_dt = tl.load(dg_base + offs_h2[:, None] * s_gn
                             ).to(tl.float32).to(MMA_DT)

            q = tl.zeros([T], dtype=tl.float32)
            for rr in tl.range(R_ITERS, loop_unroll_factor=1):
                if R_TILE:
                    r = r_fixed
                else:
                    r = rr
                b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                            + r * s_br + offs_t[:, None] * s_bt)
                B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn
                             ).to(tl.float32)
                B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn
                             ).to(tl.float32)
                if HAS_BETA:
                    be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R + r
                                  ).to(tl.float32)
                    be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R + r
                                  ).to(tl.float32)
                    if BIAS_PRE_ROT:
                        B1 = B1 + be1[None, :]
                        B2 = B2 + be2[None, :]
                if HAS_ROT:
                    Bt1 = ph_cos * B1 + ph_sin * B2
                    Bt2 = ph_cos * B2 - ph_sin * B1
                else:
                    Bt1 = B1
                    Bt2 = B2
                if HAS_BETA and not BIAS_PRE_ROT:
                    Bf1 = Bt1 + be1[None, :]
                    Bf2 = Bt2 + be2[None, :]
                else:
                    Bf1 = Bt1
                    Bf2 = Bt2
                x_r = tl.load(
                    X_ptr + b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                ).to(tl.float32)
                if HAS_DW:
                    x_r = x_r * dw[:, None]
                xw_dt = (x_r * dec_agg[:, None]).to(MMA_DT)
                gB1 = pdot(xw_dt, tl.trans(dG1_dt), input_precision=PREC)
                gB2 = pdot(xw_dt, tl.trans(dG2_dt), input_precision=PREC)

                if HAS_BETA and not BIAS_PRE_ROT:
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r,
                                  tl.sum(gB1, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r,
                                  tl.sum(gB2, axis=0), sem=ACC_SEM)
                if HAS_ROT:
                    dth_acc += (gB1 * _opv(Bt2, MMA_DT, MMA_IO, PREC) - gB2 * _opv(Bt1, MMA_DT, MMA_IO, PREC))
                    gn1 = ph_cos * gB1 - ph_sin * gB2
                    gB2 = ph_cos * gB2 + ph_sin * gB1
                    gB1 = gn1
                if HAS_BETA and BIAS_PRE_ROT:
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r,
                                  tl.sum(gB1, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r,
                                  tl.sum(gB2, axis=0), sem=ACC_SEM)
                db_base_r = (dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                             + offs_t[:, None] * s_dbt + r * s_dbr)
                tl.atomic_add(db_base_r + offs_h1[None, :] * s_dbn, gB1, sem=ACC_SEM)
                tl.atomic_add(db_base_r + offs_h2[None, :] * s_dbn, gB2, sem=ACC_SEM)

                # `B~ dG` -- the state axis is a CONTRACTION here, so the
                # sum splits exactly over the two halves.
                BdG_r = (pdot(Bf1.to(MMA_DT), dG1_dt,
                                input_precision=PREC)
                         + pdot(Bf2.to(MMA_DT), dG2_dt,
                                  input_precision=PREC))
                q += tl.sum(BdG_r * x_r, axis=1)

            if HAS_ROT:
                dth_ph = dth_acc
                if PHASE_CUMSUM and not PHASE_DEFER:
                    dth_off = tl.sum(dth_ph, axis=0)
                    dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                    tl.atomic_add(
                        dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                        + offs_h1 * s_dok, dth_off, sem=ACC_SEM)
                tl.atomic_add(
                    dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                    + offs_t[:, None] * s_dtt + offs_h1[None, :] * s_dtk,
                    dth_ph, sem=ACC_SEM)

            # SPANDELTA (`_SPANDELTA_NOTE`): the write term reaches every
            # s > j (the pair (j, T-1)), the carry term every s
            dalpha_c = tl.load(
                dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag
                + c * s_dac)
            if R_TILE:
                dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
            else:
                dalpha_gate = 1.0
            dD = a_val * (_excl_prefix(dec_agg * q, offs_t)
                          + alpha * dalpha_c * dalpha_gate)
            tl.atomic_add(
                dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt,
                dD, sem=ACC_SEM)
    else:
        for ci in range(nc_end_kb2):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci

            # Decay factors (only dec_agg needed, no dec_mask)
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)

            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)

            if HAS_ROT:
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht
                    + tl.arange(0, N // 2)[None, :] * s_thk,
                ).to(tl.float32)                                    # [T, N//2]
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + tl.arange(0, N // 2) * s_ok,
                    ).to(tl.float32)[None, :]
                cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, N)
                sn_rot = rot_sign_kb2[None, :] * _pair_bcast_2d(
                    _sin_approx(th_rot), T, N)
                dth_acc = tl.zeros([T, N], dtype=tl.float32)

            # Load dG[N,GT]
            dG = tl.load(
                dG_ptr + b * s_gb + h * s_gh + c * s_gc
                + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg,
            ).to(tl.float32)

            # Aggregate path
            if R_TILE:
                Bc_r = tl.load(
                    B_ptr + b * s_bb + c * s_bc + bg * s_bh
                    + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                    + r_fixed * s_br,
                ).to(tl.float32)
                if BIAS_PRE_ROT and HAS_BETA:
                    Bc_r = Bc_r + tl.load(
                        Beta_ptr + h * N * R + offs_n * R + r_fixed
                    ).to(tl.float32)[None, :]
                # Kb2 does NOT take `_bwd_kb1_kernel`'s algebraic swap(B~): it would keep
                # a SECOND [T, N] fp32 tile live across the aggregate MMA in a body with
                # no spill room.  The shuffle stays at the use site.
                if HAS_ROT and not ABL_PREROT:
                    B_til = cs_rot * Bc_r + sn_rot * _pair_swap_2d(Bc_r, T, N)
                else:
                    B_til = Bc_r
                if HAS_ROT_BIAS:
                    if HAS_ROT:
                        Bc_r_xor_rb = _pair_swap_2d(B_til, T, N)
                    else:
                        Bc_r_xor_rb = tl.load(
                            B_ptr + b * s_bb + c * s_bc + bg * s_bh
                            + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn
                            + r_fixed * s_br,
                        ).to(tl.float32)
                    B_unrot_rt = B_til
                    B_unrot_xor_rt = Bc_r_xor_rb
                    Bc_r = crb_kb2[None, :] * B_til + srb_fwd_kb2[None, :] * Bc_r_xor_rb
                else:
                    Bc_r = B_til
                if HAS_BETA and not BIAS_PRE_ROT:
                    beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)
                    Bc_r = Bc_r + beta_r[None, :]
                x_base_r = b * s_xb + h * s_xh + r_fixed * s_xr + c * T * s_xl
                x_r = tl.load(
                    X_ptr + x_base_r
                    + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                ).to(tl.float32)
                if HAS_DW:
                    x_r = x_r * dw[:, None]
                xw_r = x_r * dec_agg[:, None]
                # dG straight below, transposed here -- one cast.
                dG_dt = dG.to(MMA_DT)
                dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt),
                                  input_precision=PREC)
                if HAS_ROT:
                    gB_ = dB_r_agg
                    if HAS_BETA and not BIAS_PRE_ROT:
                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed,
                                      tl.sum(gB_, axis=0), sem=ACC_SEM)
                    if HAS_ROT_BIAS:
                        Brd_p_rt = (crb_kb2[None, :] * B_unrot_xor_rt
                                     - srb_fwd_kb2[None, :] * B_unrot_rt)
                        tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb2,
                                      tl.sum(rot_sign_kb2[None, :] * gB_ * Brd_p_rt, axis=0), sem=ACC_SEM)
                        gB_ = (crb_kb2[None, :] * gB_
                               - srb_fwd_kb2[None, :] * _pair_swap_2d(gB_, T, N))
                    if not ABL_NO_DTH:
                        dth_acc += gB_ * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, N)
                    gB_ = cs_rot * gB_ - sn_rot * _pair_swap_2d(gB_, T, N)
                    if HAS_BETA and BIAS_PRE_ROT:
                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed,
                                      tl.sum(gB_, axis=0), sem=ACC_SEM)
                    tl.atomic_add(
                        dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                        + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
                        + r_fixed * s_dbr,
                        gB_, sem=ACC_SEM)
                elif HAS_ROT_BIAS:
                    db_base_r = (dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                                 + offs_t[:, None] * s_dbt + r_fixed * s_dbr)
                    tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn,
                                  crb_kb2[None, :] * dB_r_agg, sem=ACC_SEM)
                    tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn,
                                  srb_fwd_kb2[None, :] * dB_r_agg, sem=ACC_SEM)
                    Brd_p_rt = crb_kb2[None, :] * B_unrot_xor_rt - srb_fwd_kb2[None, :] * B_unrot_rt
                    cross_rb_rt = tl.sum(rot_sign_kb2[None, :] * dB_r_agg * Brd_p_rt, axis=0)
                    tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb2, cross_rb_rt, sem=ACC_SEM)
                else:
                    tl.atomic_add(
                        dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                        + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
                        + r_fixed * s_dbr,
                        dB_r_agg, sem=ACC_SEM)
                if HAS_BETA and not HAS_ROT:
                    tl.atomic_add(
                        dBeta_ptr + h * N * R + offs_n * R + r_fixed,
                        tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                BdG_r = pdot(Bc_r.to(MMA_DT), dG_dt, input_precision=PREC)
                q = tl.sum(BdG_r * x_r, axis=1)
            else:
                q = tl.zeros([T], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    Bc_r = tl.load(
                        B_ptr + b * s_bb + c * s_bc + bg * s_bh
                        + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                        + r * s_br,
                    ).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r
                        ).to(tl.float32)[None, :]
                    # Kb2 does NOT take `_bwd_kb1_kernel`'s algebraic swap(B~): it would keep
                    # a SECOND [T, N] fp32 tile live across the aggregate MMA in a body with
                    # no spill room.  The shuffle stays at the use site.
                    if HAS_ROT and not ABL_PREROT:
                        B_til = cs_rot * Bc_r + sn_rot * _pair_swap_2d(Bc_r, T, N)
                    else:
                        B_til = Bc_r
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            Bc_r_xor_rb = _pair_swap_2d(B_til, T, N)
                        else:
                            Bc_r_xor_rb = tl.load(
                                B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn
                                + r * s_br,
                            ).to(tl.float32)
                        B_unrot_rl = B_til
                        B_unrot_xor_rl = Bc_r_xor_rb
                        Bc_r = crb_kb2[None, :] * B_til + srb_fwd_kb2[None, :] * Bc_r_xor_rb
                    else:
                        Bc_r = B_til
                    if HAS_BETA and not BIAS_PRE_ROT:
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(
                        X_ptr + x_base_r
                        + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                    ).to(tl.float32)
                    if HAS_DW:
                        x_r = x_r * dw[:, None]
                    xw_r = x_r * dec_agg[:, None]
                    # dG straight below, transposed here -- one cast.
                    dG_dt = dG.to(MMA_DT)
                    dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt),
                                      input_precision=PREC)
                    if HAS_ROT:
                        gB_ = dB_r_agg
                        if HAS_BETA and not BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r,
                                          tl.sum(gB_, axis=0), sem=ACC_SEM)
                        if HAS_ROT_BIAS:
                            Brd_p_rl = (crb_kb2[None, :] * B_unrot_xor_rl
                                         - srb_fwd_kb2[None, :] * B_unrot_rl)
                            tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb2,
                                          tl.sum(rot_sign_kb2[None, :] * gB_ * Brd_p_rl, axis=0), sem=ACC_SEM)
                            gB_ = (crb_kb2[None, :] * gB_
                                   - srb_fwd_kb2[None, :] * _pair_swap_2d(gB_, T, N))
                        if not ABL_NO_DTH:
                            dth_acc += gB_ * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, N)
                        gB_ = cs_rot * gB_ - sn_rot * _pair_swap_2d(gB_, T, N)
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r,
                                          tl.sum(gB_, axis=0), sem=ACC_SEM)
                        tl.atomic_add(
                            dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                            + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
                            + r * s_dbr,
                            gB_, sem=ACC_SEM)
                    elif HAS_ROT_BIAS:
                        db_base_r = (dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                                     + offs_t[:, None] * s_dbt + r * s_dbr)
                        tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn,
                                      crb_kb2[None, :] * dB_r_agg, sem=ACC_SEM)
                        tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn,
                                      srb_fwd_kb2[None, :] * dB_r_agg,
                                      sem=ACC_SEM)
                        Brd_p_rl = crb_kb2[None, :] * B_unrot_xor_rl - srb_fwd_kb2[None, :] * B_unrot_rl
                        cross_rb_rl = tl.sum(rot_sign_kb2[None, :] * dB_r_agg * Brd_p_rl, axis=0)
                        tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb2, cross_rb_rl, sem=ACC_SEM)
                    else:
                        tl.atomic_add(
                            dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
                            + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
                            + r * s_dbr,
                            dB_r_agg, sem=ACC_SEM)
                    if HAS_BETA and not HAS_ROT:
                        tl.atomic_add(
                            dBeta_ptr + h * N * R + offs_n * R + r,
                            tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                    BdG_r = pdot(Bc_r.to(MMA_DT), dG_dt, input_precision=PREC)
                    q += tl.sum(BdG_r * x_r, axis=1)

            if HAS_ROT and not ABL_NO_DTH:
                dth_e, dth_o = tl.split(tl.reshape(dth_acc, (T, N // 2, 2)))
                # ROTBWD: the pair sign is applied HERE, not per term.
                dth_ph = dth_e - dth_o                              # [T, N//2]
                if PHASE_CUMSUM and not PHASE_DEFER:
                    dth_off = tl.sum(dth_ph, axis=0)
                    dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                    tl.atomic_add(
                        dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                        + tl.arange(0, N // 2) * s_dok,
                        dth_off, sem=ACC_SEM)
                tl.atomic_add(
                    dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                    + offs_t[:, None] * s_dtt
                    + tl.arange(0, N // 2)[None, :] * s_dtk,
                    dth_ph, sem=ACC_SEM)

            # SPANDELTA (`_SPANDELTA_NOTE`): the write term reaches every
            # s > j (the pair (j, T-1)), the carry term every s.
            # dalpha term is per-(b, h, c, gt) — NOT per r.  In R_TILE mode
            # only the r==0 CTA may add it, otherwise it is counted R times.
            dalpha_c = tl.load(
                dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
            if R_TILE:
                dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
            else:
                dalpha_gate = 1.0
            dD = a_val * (_excl_prefix(dec_agg * q, offs_t)
                          + alpha * dalpha_c * dalpha_gate)

            tl.atomic_add(
                dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt,
                dD, sem=ACC_SEM)


# ═══ Kb2's owner-accumulate head block (`_bwd_kb2_hb_kernel`) ═══════════
#
# WHY.  The historical `_bwd_kb2_kernel` runs a (B, H, nc*NGT*R) grid and
# EVERY CTA issues one T*N fp32 `atomic_add` into `dBtil`, whose fan-in is
# HPG_B heads (the tile is indexed by the B/C GROUP, not by the head).  It
# is bound by L2 read-modify-write serialisation on sm_80 (the same bound
# as Kb1's tile atomics), not by MMAs or spills.
#
# THE FIX is Kb1's construction applied to Kb2: grid dim-1 becomes a head
# BLOCK of `HB2` heads, the CTA loops over them, and `dB_agg` -- the SAME
# [T, N] tile for every head of a group -- is summed in REGISTERS and
# written with ONE atomic per CTA instead of HB2.  The per-head quantities
# (`dDelta`, `dBeta`) keep their own atomics (a head block does not reduce
# their fan-in, and they are small against the tile).  `B_c` is loaded ONCE
# per block instead of once per head.
#
# ENVELOPE.  No rotation of any kind (`HAS_ROT`, `HAS_ROT_BIAS`), no
# `PAIR_HALF`, no ablation ladder, `CHUNK_PARALLEL` only, and `R*O > 1`.
# Every other path keeps the historical `_bwd_kb2_kernel` bit for bit.
# `_kb2_hb()` is the host rule; `SSD_MIMO_BWD_KB2_HB` ("auto" | <int> |
# "0") is the lever.
#
# ── Kb2-HB's own candidate grid ──────────────────────────────────────
# The shared grid cannot reach two candidates that win Kb2: `maxnreg=128`
# on A100 and `num_warps=1` on Blackwell.  `num_stages` moves nothing
# (<= 0.3 %), so this grid trades the stage axis for them.
#
# `num_warps=8` is nevertheless KEPT.  `test_ssd_mimo_backward_gradients
# _are_warp_count_invariant` filters EVERY autotuner in this module to one
# `num_warps`, taking the warp set from the first tuner it finds
# (`_BWD_CFGS`); a per-kernel grid that omits one of those leaves that
# tuner with an EMPTY config list (`IndexError` in `Autotuner.run`).  Any
# per-kernel grid in this file must be a SUPERSET of `_BWD_WARPS` on the
# warp axis.
_KB2_HB_CFGS = ([_bwd_cfg(4, 2, 0)] if _AUTOTUNE_OFF else
                [_bwd_cfg(w, st, m)
                 for w in (1, 2, 4, 8) for st in (1, 2) for m in (0, 128)])


def _kb2_hb_grid(tier):
    if tier == "pinned":
        return [_bwd_cfg(4, 2, 0)]
    return [_bwd_cfg(w, st, m)
            for w in (1, 2, 4, 8) for st in (1, 2) for m in (0, 128)]


@triton.autotune(configs=_calib_grid("fused_kb2_hb", _kb2_hb_grid),
                 prune_configs_by={"early_config_prune": _cprune("fused_kb2_hb")},
                 key=["T", "N", "G_SIZE", "GT", "PREC", "MMA_IO",
                      "HAS_BETA", "R", "HAS_DW", "IDX64", "HB2",
                      "OCC"],
                 restore_value=["dBtil_ptr", "dDelta_ptr", "dBeta_ptr"])
@triton.jit
def _bwd_kb2_hb_kernel(
    B_ptr, X_ptr, Delta_ptr, A_ptr,
    Beta_ptr,
    dG_ptr, dAlpha_ptr,
    dBtil_ptr, dDelta_ptr, dBeta_ptr,
    H,
    s_bb, s_bc, s_bt, s_bn,
    s_xb, s_xh, s_xg, s_xr, s_xl,
    s_db, s_dh, s_dc, s_dt,
    s_gb, s_gh, s_gc, s_gn, s_gg,
    s_dab, s_dah, s_dag, s_dac,
    s_dbb, s_dbc, s_dbt, s_dbn,
    s_bh,
    NC, NGT_val,
    T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr,
    GT: tl.constexpr, PREC: tl.constexpr,
    MMA_DT: tl.constexpr = tl.float32,
    # Autotune KEY stand-in for MMA_DT (a tl.dtype is not serialisable into
    # Triton's disk cache key): 1 iff MMA_DT is 16-bit.
    MMA_IO: tl.constexpr = 0,
    HAS_BETA: tl.constexpr = False,
    HPG_B: tl.constexpr = 1,
    R: tl.constexpr = 1,
    s_br: tl.constexpr = 0,
    s_dbr: tl.constexpr = 0,
    DW_ptr=None, s_wb=0, s_wh=0, s_wl=0,
    HAS_DW: tl.constexpr = False,
    s_dbh=0,
    IDX64: tl.constexpr = False,
    DCS_INK: tl.constexpr = False,
    HB2: tl.constexpr = 1,
    OCC=0,           # autotune-key occupancy bucket only (`occ_bucket`)
):
    """Kb2 with an owner-accumulate head block.

    grid = (B, H // HB2, nc * NGT * R).  Identical arithmetic to
    `_bwd_kb2_kernel`'s no-rotation `R_TILE` + `CHUNK_PARALLEL` body; the
    ONLY difference is that the `dBtil` tile is accumulated over the HB2
    heads of the block in registers and written once.
    """
    b = tl.program_id(0)
    hb_id = tl.program_id(1)
    c = tl.program_id(2) // (NGT_val * R)
    remainder = tl.program_id(2) % (NGT_val * R)
    r_fixed = remainder // NGT_val
    gt_idx = remainder % NGT_val
    if IDX64:
        b = b.to(tl.int64)
        hb_id = hb_id.to(tl.int64)
    h0 = hb_id * HB2
    # The block lies inside ONE B/C group (`_kb2_hb` picks HB2 among the
    # divisors of HPG_B), so the whole block shares `bg` and `B_c`.
    bg = h0 // HPG_B

    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    is_last = tl.where(offs_t == T - 1, 1.0, 0.0)

    B_raw = tl.load(
        B_ptr + b * s_bb + c * s_bc + bg * s_bh
        + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
        + r_fixed * s_br,
    ).to(tl.float32)

    dB_acc = tl.zeros([T, N], dtype=tl.float32)
    # A RUNTIME head loop: as `tl.static_range` it emitted HB2 (up to 32)
    # copies of the body per config and the grid took minutes to compile; the
    # runtime loop is also faster.
    for hh in tl.range(HB2, loop_unroll_factor=1):
        h = h0 + hh
        a_val = -tl.exp(tl.load(A_ptr + h))
        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
        seg_end = tl.max(seg_h, axis=0)
        D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
        alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
        dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)

        dG = tl.load(
            dG_ptr + b * s_gb + h * s_gh + c * s_gc
            + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg,
        ).to(tl.float32)
        x_r = tl.load(
            X_ptr + b * s_xb + h * s_xh + r_fixed * s_xr + c * T * s_xl
            + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
        ).to(tl.float32)
        if HAS_DW:
            x_r = x_r * tl.load(DW_ptr + b * s_wb + h * s_wh
                                + (c * T + offs_t) * s_wl)[:, None]
        xw_r = x_r * dec_agg[:, None]
        # K24 CAST-BEFORE-TRANSPOSE: dG is used in both orientations.
        dG_dt = dG.to(MMA_DT)
        dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt),
                          input_precision=PREC)
        dB_acc += dB_r_agg
        if HAS_BETA:
            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed,
                          tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
            Bc_r = B_raw + tl.load(
                Beta_ptr + h * N * R + offs_n * R + r_fixed,
            ).to(tl.float32)[None, :]
        else:
            Bc_r = B_raw
        BdG_r = pdot(Bc_r.to(MMA_DT), dG_dt, input_precision=PREC)
        q = tl.sum(BdG_r * x_r, axis=1)

        # SPANDELTA (`_SPANDELTA_NOTE`): the write term reaches every
        # s > j, the carry term every s.  The dalpha term is per
        # (b, h, c, gt) and NOT per r -- only the r == 0 CTA may add it
        # (see `_bwd_kb2_kernel`'s R_TILE gate).
        dalpha_c = tl.load(
            dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
        dD = a_val * (_excl_prefix(dec_agg * q, offs_t)
                      + alpha * dalpha_c * tl.where(r_fixed == 0, 1.0, 0.0))
        tl.atomic_add(
            dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt,
            dD, sem=ACC_SEM)

    tl.atomic_add(
        dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc
        + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
        + r_fixed * s_dbr,
        dB_acc, sem=ACC_SEM)


def _kb2_hb(H, hpg_b, Bsz, nc, NGT, R, refused, device):
    """Pick Kb2's head block.  0 = the old kernel.

    Mirrors `_kb1_hb`: the block must divide the heads per B group (the
    register accumulator has ONE address), and the largest such block
    that still puts >= 2 CTAs on every SM wins, because the atomic tile
    count falls as 1/HB2.  Unlike Kb1 there is no sole-owner store to
    reach for -- Kb1 (or Kc) writes `dBtil` too -- so the whole win is
    the fan-in division.
    """
    if refused:
        return 0
    hpg = max(1, min(hpg_b, H))
    env = _env("BWD_KB2_HB", "auto")
    if env != "auto":
        hb = int(env)
        if hb <= 0:
            return 0
        if hpg % hb:
            raise ValueError(
                "SSD_MIMO_BWD_KB2_HB=%d does not divide the heads per B "
                "group (%d): a head block that straddles a group would "
                "accumulate two different dBtil tiles into one register "
                "file." % (hb, hpg))
        return hb
    target = 2 * _sm_count(device)
    hb = 1
    for d in (d for d in range(hpg, 0, -1) if hpg % d == 0):
        if Bsz * ((H + d - 1) // d) * nc * NGT * R >= target:
            hb = d
            break
    return hb


# ═══ Kb2 ABSORBED INTO Kb1 (KB2_FOLD) ══════════════════════════════════
#
# Both Kb1 and Kb2 are head-blocked and both write `dBtil`; Kb1's grid
# `(B, ceil(H/HBLK), nc*NGT*NSPLIT)` carries NO G-tile axis, so folding
# the aggregate term into it costs none of the G-fold redundancy that
# makes the Kc route lose.  What the fold removes:
#
#   * the whole `_bwd_kb2_hb_kernel` LAUNCH;
#   * its re-read of `B~`, `x` and `delta` -- Kb1 holds all three live;
#   * its `dBtil` atomic pass, and with it the ORDERING constraint that
#     forced it to run AFTER Kb1 whenever `DB_STORE` is on;
#   * its `dBeta` and `dDelta` atomics (Kb1's own now carry the total);
#   * ONE OF ITS TWO GEMMS.  Kb2 needs q_r[t] = sum_gt (B~_r dG)[t,gt]
#     x_r[t,gt] for `dDelta`, and computes it with a second
#     [T,N]x[N,GT] product.  But the aggregate tile it already formed
#     satisfies
#         sum_n B~_r[t,n] dB_agg_r[t,n]
#           = sum_n B~_r[t,n] sum_gt xw_r[t,gt] dG[n,gt]
#           = sum_gt xw_r[t,gt] (B~_r dG)[t,gt]
#           = dec_agg[t] q_r[t],
#     i.e. the ROW REDUCTION of `B~_r * dB_agg_r` -- and Kb1 already
#     runs `dD -= a_val * sum(Bc_l[r] * dB_r_part, axis=1)` on the tile
#     the seed makes the total.  Kb2's `-a dec_agg q` term is therefore
#     produced by a line that was already there, and its chunk-end
#     correction is the t-sum of the same vector.
#
# What the fold ADDS to Kb1, per (head, r): one [T,GT]x[GT,NS] MMA and
# one [T,NS] row reduction; per head: one [NS,GT] `dG` load, one [T]
# exp and one scalar `dalpha` load.  Nothing new stays live across the
# o loop -- the aggregate tile IS `dBp[r]`'s initial value.
#
# ENVELOPE (refusals by name, so every path it does not serve compiles
# the historical two-kernel schedule): no `HAS_ROT_BIAS`, no
# `PAIR_HALF`, no deferred rotation, no ablation ladder, no slot layout,
# `R*O > 1`, `fuse_kb2` off (there Kc owns the aggregate), the R = O = 1
# `use_hb` route off, and HBLK > 0 (the fold lives in the head-blocked
# body only, so an `OutOfResources` fallback to HBLK = 0 also falls back
# to launching Kb2).
#
# ═══ The fold UNDER ROTATION ═══════════════════════════════════════════
# The in-kernel phase (`rot_in_kernel`, Kb1's HAS_ROT) is served too: Kb1's
# rotating body already builds R(theta) B~ per kernel head for the score
# dot, the seed is the gradient w.r.t. exactly that tile, and everything
# Kb2's rotating branch does to `dB_r_agg` afterwards (dBeta, dTheta, the
# un-rotation, the `dec_agg q` reduction) is the write-side chain Kb1 runs
# on the o-folded `dBp[r]` anyway.  So the seed enters BEFORE the chain and
# the fold is the same three lines it is in real mode; the aggregate path's
# dTheta joins Kb1's `dth_acc` (one atomic per head instead of two), and no
# second live [T, N] tile appears.  `SSD_MIMO_BWD_KB2_FOLD_ROT` ("auto" |
# "0") refuses the rotating fold by name for A/Bs; real mode is not
# reachable by it.


def _kb2_fold_rot():
    """Stage 3 lever: may the fold engage under the in-kernel phase?

    "auto" (the default) folds; "0" keeps the two-kernel schedule on the
    rotating route only (real mode folds either way)."""
    return _env("BWD_KB2_FOLD_ROT", "auto") != "0"


def _kb2_fold(kb1_hb, refused):
    """May Kb1 absorb Kb2?  `SSD_MIMO_BWD_KB2_FOLD`.

    "auto" (the default) folds wherever the envelope allows; "0" keeps
    the two-kernel schedule; "1" forces the fold and RAISES if the
    envelope refuses it, so an A/B cannot silently compare a body
    against itself.
    """
    env = _env("BWD_KB2_FOLD", "auto")
    if env == "0":
        return False
    if refused or not kb1_hb:
        if env == "1":
            raise ValueError(
                "SSD_MIMO_BWD_KB2_FOLD=1 but this cell is outside the "
                "fold's envelope (rot_bias / PAIR_HALF / slots / "
                "deferred rotation / ablation / R*O == 1 / fuse_kb2 / "
                "use_hb / HBLK == 0, or the in-kernel phase with "
                "SSD_MIMO_BWD_KB2_FOLD_ROT=0); see `_kb2_fold`'s "
                "comment block.")
        return False
    return True

@triton.autotune(configs=_calib_grid("fused_kinv", _bwd_grid),
                 prune_configs_by={"early_config_prune": _cprune("fused_kinv")},
                 key=["T", "N", "O",
                      "OCC"],
                 restore_value=["dCtil_ptr", "dThetaH_ptr"])
@triton.jit
def _bwd_kinv_kernel(
    dCrd_buf_ptr,    # [B*H, nc*O*T*N] fp32 — rotated-domain dC (flat)
    C_ptr,           # [B, nc, (Gc,) T, N(, O)]
    ThetaH_ptr,      # [H, N//2]
    dCtil_ptr,       # [B, nc, T, N(, O)] output
    dThetaH_ptr,     # [H, N//2] output
    H, NC,
    s_cb, s_cc, s_ct, s_cn,
    s_dcb, s_dcc, s_dct, s_dcn,
    s_ch,
    T: tl.constexpr, N: tl.constexpr,
    NEED_DTHH: tl.constexpr = True,
    HPG_C: tl.constexpr = 1,
    O: tl.constexpr = 1,
    s_co: tl.constexpr = 0,
    s_dco: tl.constexpr = 0,
    s_dch=0,
    IDX64: tl.constexpr = False,
    OCC=0,           # autotune-key occupancy bucket only (`occ_bucket`)
):
    """Kinv: inverse theta_h rotation + dThetaH.

    Reads dC_rd from temp buffer, loads partner via XOR index (no scratch
    needed), computes dC and atomic_adds to dCtil.  Optionally computes
    dThetaH from C and dC_rd.

    Grid: (B, nc, H).
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
    cg = h // HPG_C

    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)

    pair_idx = offs_n // 2
    th_full = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)
    ch_full = _cos_approx(th_full)
    sh_full = _sin_approx(th_full)
    fwd_sign = tl.where((offs_n % 2) == 0, 1.0, -1.0)
    fwd_coeff = fwd_sign * sh_full

    inv_coeff = -fwd_coeff

    for o in tl.range(O, loop_unroll_factor=1):
        # dCrd_buf layout: [B, H, nc, O, T, N] — stored by Kb1
        dr_off = (b * H + h) * (NC * O * T * N) + c * (O * T * N) + o * (T * N)
        dCrd = tl.load(
            dCrd_buf_ptr + dr_off
            + offs_t[:, None] * N + offs_n[None, :],
        ).to(tl.float32)
        dCrd_xor = tl.load(
            dCrd_buf_ptr + dr_off
            + offs_t[:, None] * N + (offs_n ^ 1)[None, :],
        ).to(tl.float32)

        dC = ch_full[None, :] * dCrd + inv_coeff[None, :] * dCrd_xor
        tl.atomic_add(
            dCtil_ptr + cg * s_dch + b * s_dcb + c * s_dcc + o * s_dco
            + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn,
            dC, sem=ACC_SEM)

        if NEED_DTHH:
            c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
            Co_xor = tl.load(
                c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn,
            ).to(tl.float32)
            Co = tl.load(
                c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn,
            ).to(tl.float32)
            Crd_partner = ch_full[None, :] * Co_xor - fwd_coeff[None, :] * Co
            cross_sum = tl.sum(fwd_sign[None, :] * dCrd * Crd_partner, axis=0)
            tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx, cross_sum, sem=ACC_SEM)




def _rotate_tile_fwd(tile, theta):
    """Apply R(theta)^T pairwise to even/odd N columns. tile: [..., N], theta: [..., N//2]."""
    even = tile[..., 0::2]
    odd = tile[..., 1::2]
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    out = torch.empty_like(tile)
    out[..., 0::2] = cos_th * even + sin_th * odd
    out[..., 1::2] = -sin_th * even + cos_th * odd
    return out


def _rotate_tile_inv(tile, theta):
    """Apply R(theta) (inverse of R^T) pairwise. tile: [..., N], theta: [..., N//2]."""
    even = tile[..., 0::2]
    odd = tile[..., 1::2]
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    out = torch.empty_like(tile)
    out[..., 0::2] = cos_th * even - sin_th * odd
    out[..., 1::2] = sin_th * even + cos_th * odd
    return out


def _dtheta_from_rotation(grad_rd, tile_rd, n_dim=-1):
    """dTheta = grad_rd[:, 2k] * tile_rd[:, 2k+1] - grad_rd[:, 2k+1] * tile_rd[:, 2k].
    Returns shape [..., N//2]."""
    return (grad_rd[..., 0::2] * tile_rd[..., 1::2]
            - grad_rd[..., 1::2] * tile_rd[..., 0::2])


def kbc_sumds(*, rot_in_kernel, rot_per_head, has_theta_h, has_rot_bias,
              pre_rot, pair_half, rot_abl):
    """SUMDS for the head-batched Kbc (see the constexpr's note): the head
    sum of the masked score gradient before its two state contractions.
    Served only where B/C are head-shared AND unrotated (the contractions'
    right operand is then the same tile for every head of the block);
    lever `SSD_MIMO_BWD_KBC_SUMDS` ("0" = off, the shipped body; "1" = on
    where it applies)."""
    if _env("BWD_KBC_SUMDS", "0") != "1":
        return False
    return not (rot_in_kernel or rot_per_head or has_theta_h or has_rot_bias
                or pre_rot or pair_half or rot_abl)


def kc_shared_cb(*, rot_in_kernel, has_theta_h, has_rot_bias, pre_rot,
                 pair_half, R, O, bc_grouped, use_r_tile, T=None, dtype=None):
    """SHARED_CB for Kc (see the constexpr's note): the head-independent raw
    score C_c B_c^T formed once per (b, c).  Head-shared, unrotated,
    R = O = 1 only; lever `SSD_MIMO_BWD_KC_SHARED_CB` ("1" on)."""
    ok = not (rot_in_kernel or has_theta_h or has_rot_bias or pre_rot
              or pair_half or bc_grouped or use_r_tile or R != 1 or O != 1)
    if not ok:
        return False
    from . import levers
    return levers.on("real_kc_shared_cb", levers.real_group(T, dtype), guard=ok)


def _mma_torch_dtype(mma_dt):
    return {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}.get(
        mma_dt, torch.float32)


def fused_chunk_bwd_5kernel(
    Ctil, Btil, x_scaled, delta_cumsum, a_log,
    S_all, dY,
    theta_h=None, theta=None, dS_final=None, precision="tf32",
    bc_grouped=False, Gb=1, Gc=1,
    beta=None, gamma=None, x0=None, rot_bias=None,
    bias_pre_rot=False,
    dw=None,
    trap_nd=None,
    theta_off=None,
    # ── the trapezoid's adjoint and the phase's cross-chunk term ──
    trap_dsdot=None, trap_lam=None, trap_dot=None, trap_nxt=None,
    trap_dt=None, need_dlam=False, fuse_trap=False,
    phase_internal=False,
    # ── the D-skip's adjoint inside Kc's dX store ────────────────
    # `d_skip` [H, G_SIZE, R] (fp32, contiguous, already G-padded) turns
    # on the in-kernel form; the extra trailing output is dD.  `None`
    # leaves the gate epilogue's own dXR/dDsk route in place.
    d_skip=None, dskip_dd=False,
    # PAIRHALF: the rotate-half pair layout.  Stage 1 wires
    # the two kernels the R = O = 1 rotating backward runs -- the
    # head-batched Kbc and Kc -- plus the trapezoid adjoint; every other
    # schedule is refused at the top of this function.
    pair_half=False,
    # `delta_cumsum` carries the RAW
    # chunked delta; every kernel forms D_i with its own `tl.cumsum`.
    # The caller is then responsible for turning the returned dDelta
    # (which is still dL/dD_i) into dL/ddelta with a reverse prefix sum.
    dcs_ink=False,
    # the LOCAL chart cotangent : a dict with the
    # geometry (mode, k, Mf, Mp, L, G); Kch1 / Kch2 run after Kc on the
    # tensors this backward holds (S_all, dG_all, dY, x, dw, the decay,
    # B~ / C~) and the packed tangent lands in chart["F"].  None: nothing
    # of it runs (every other route bit for bit as before).
    chart=None,
):
    """5-kernel backward: Ka + Kb2 + Kb1 + Kinv + Kc.

    dw [B, H, L_padded]: when given, x_scaled is RAW x (scaled by dw
    in-kernel), dX comes back as dx_raw, and dDW (the write-scale
    gradient) is returned as an extra trailing output.

    Supports R,O >= 1 (MIMO) and per-timestep theta rotation, the phase
    head-SHARED ([B, nc, T, K]) or PER HEAD ([B, nc, T, H, K]).

    When theta is given, B and C are pre-rotated by R(theta)^T before
    passing to the kernels.  Post-kernel, dB and dC are inverse-rotated,
    and dTheta is computed from the chain rule.

    Execution order: Ka → Kb2 → Kb1 → Kc → Python post-rotate
    """
    Bsz, H, nc, T = delta_cumsum.shape
    G_SIZE = x_scaled.shape[2]
    R = x_scaled.shape[3]
    device = x_scaled.device
    f32 = torch.float32
    dcs_kw = dict(DCS_INK=bool(dcs_ink))

    # ── D1: the MMA operand dtype, resolved ONCE for all five kernels ──
    # tl.float32 in both fp32-operand modes (and at fp32 IO in the third
    # one), where every `.to(MMA_DT)` below is Triton's cast identity.
    # `precision` is then
    # narrowed to the `tl.dot` spelling so every `PREC=precision` launch
    # argument stays as it was.  The rotated tiles are built in fp32 and
    # rounded AT the dot, which is Mamba-3's own rotary convention.
    mma_dt = mma_operand_dtype(precision, x_scaled.dtype)
    mma_io = mma_dt is not tl.float32
    # the wgmma shared-swizzle floor is a BYTE rule, so the operand
    # width is what the launch rules below are keyed on.
    dt_bytes = 2 if mma_io else 4
    precision = dot_precision(precision)

    # Detect O and N from Ctil shape.
    # grouped:   [B, nc, Gc, T, N] or [B, nc, Gc, T, N, O]
    # ungrouped: [B, nc, T, N]     or [B, nc, T, N, O]
    if bc_grouped:
        N = Ctil.shape[4]
        O = Ctil.shape[5] if Ctil.dim() == 6 else 1
    elif Ctil.dim() == 5:
        N = Ctil.shape[3]
        O = Ctil.shape[4]
    else:
        N = Ctil.shape[-1]
        O = 1

    # calibration shape group of this call ({real, complex} x {R=O=1, R/O>1})
    from .calibration_fused import activate as _calib_activate
    _calib_activate(theta is not None or theta_h is not None, R, O,
                    Bsz, H, T, dtype=("fp32" if x_scaled.dtype == torch.float32 else "bf16"))

    # N = 128 doubles every [T, N]/[N, GT] tile; GT = 64 then exceeds the
    # ~100KB shared-memory budget of CONSUMER parts in the BWD kernels
    # (the contract's known N=128 backward limit).  Datacenter sm_90/100
    # (~228KB) keeps the full 64-wide tile.  _pick_gt supports any
    # G = 16*n (odd multiples -> 16-wide tiles).
    from .fused_fwd_kernel import _pick_gt, _big_smem
    _gt_cap = 64 if (N <= 64 or _big_smem(device)) else 32
    # CHUNKPAR register-cliff lever `SSD_MIMO_BWD_GT_CAP` (16/32): a
    # NARROWER G tile halves every [.., GT] tile of Kbc-hb / Kc / Ka (dY, x,
    # S_all, dG) at the price of more CTAs re-forming the G-independent work
    _gt_env = _env("BWD_GT_CAP", "")
    if _gt_env:
        _gt_cap = min(_gt_cap, int(_gt_env))
    elif mma_dt is tl.float32 and x_scaled.dtype == torch.float32:
        # The G tile of an fp32-IO call is a per-GPU CALIBRATED structural choice
        # (threshold `fp32|bwd_gt_cap` of this GPU's ssd_mimo_fused file; its fp32
        # groups are measured at that tile): fp32 operand tiles are twice the 16-bit
        # footprint, so a narrower G tile can win (A100, N = 128, T = 32).
        from .. import calibration as _cal
        _v, _src = _cal.threshold("ssd_mimo_fused", "fp32|bwd_gt_cap", None)
        if _v:
            _gt_cap = min(_gt_cap, int(_v))
    GT = _pick_gt(G_SIZE, cap=_gt_cap)
    NGT = G_SIZE // GT

    # ── the 128-byte swizzle floor on the STATE tile ────────────────────
    # The N-split floor below (`_hb_ns_floor`) is the same law applied to
    # `NS = N // NSPLIT`, and at `N * itemsize < 128` under 16-bit operands NO
    # split meets it -- the `[T, N]` operand is already under the floor at
    # NSPLIT = 1.  It only BITES when the CTA also carries a full-width
    # `[*, GT]` tile: with bf16 IO, N = 16 / GT = 64 and N = 32 / GT = 64 take
    # an ILLEGAL ACCESS in both backward branches (`_bwd_kbc_hb_kernel`,
    # `_bwd_kb1_kernel`), while N = 16 / GT <= 32, N = 32 / GT = 32 and
    # N = 64 / GT = 64 are exact, as are the identical cells at
    # `precision="tf32"`.  The config cannot be lowered and its siblings are
    # silent at half the shapes, so it is REFUSED rather than served.  The
    # fp32-operand default is untouched.
    if (mma_io and N * dt_bytes < _SWIZZLE_FLOOR_BYTES
            and GT * dt_bytes >= _SWIZZLE_FLOOR_BYTES):
        # SERVED, not refused (a ValueError INSIDE the backward is a broken contract): the fault
        # needs a G tile at least as wide as the swizzle floor, and G is tiled -- a G tile one step narrower is exact (the same
        # math in more CTAs) and sits in the table's `ok` rows (N16 GT32, N32 GT32).
        GT = _pick_gt(G_SIZE, cap=max(16, (_SWIZZLE_FLOOR_BYTES // dt_bytes) // 2))
        NGT = G_SIZE // GT
    if (mma_io and N * dt_bytes < _SWIZZLE_FLOOR_BYTES
            and GT * dt_bytes >= _SWIZZLE_FLOOR_BYTES):
        raise ValueError(
            "SSD-MIMO backward: N=%d with %d-byte MMA operands gives a "
            "%d-byte [T, N] tile against a %d-byte G tile (GT=%d) -- "
            "Triton 3.6 strides the "
            "nvmma_shared<swizzlingByteWidth=64> descriptor of the "
            "narrow one as if it were the 128-byte mode, and "
            "_bwd_kbc_hb_kernel / _bwd_kb1_kernel then read past their "
            "shared arena.  Use precision='tf32' (the default) at this "
            "state size, or a state size of at least %d."
            % (N, dt_bytes, N * dt_bytes, GT * dt_bytes, GT,
               _SWIZZLE_FLOOR_BYTES // dt_bytes))

    has_rot = theta is not None
    has_theta_h = theta_h is not None
    need_dthh = has_theta_h
    has_dsf = dS_final is not None
    has_beta = beta is not None
    has_gamma = gamma is not None
    has_x0 = x0 is not None
    has_rot_bias = rot_bias is not None
    has_dw_early = dw is not None
    # Experimental MIMO schedule: Kc also accumulates Kb2's write-map
    # and decay adjoints when the write map has no rotation. Read-side
    # theta_h is independent of that aggregate and remains supported.
    # Keep the existing schedule until parity and measured cost justify
    # promotion; rotated write maps still use the separate Kb2 launch.
    fuse_kb2_mimo = (O > 1 and not has_rot and not has_rot_bias
                    and _env("FUSE_KB2_MIMO", "0") == "1")
    fuse_kb2 = (has_dw_early and (O == 1 or fuse_kb2_mimo)
                and _env("FUSE_KB2", "1") == "1")
    # Owner-accumulate path (see _bwd_kbc_hb_kernel): replaces Kb1,
    # Kinv AND Kc's dB_agg on the O=1, R=1, ungrouped fused path.
    # rotation mode (theta_h / rot_bias) is served
    # here too -- the pair rotations fold into the GEMM operands, so
    # the dB/dC accumulators stay unrotated and neither the permuted
    # atomic scatter nor the per-head dCrd_buf + Kinv launch is
    # needed.  Contract DRAFT 7.7.  SSD_MIMO_BWD_HB=0 restores the
    # generic Kb1 (+ Kinv) branch.
    # PAIRHALF: R, O > 1 is served -- `_bwd_kb1_kernel`,
    # `_bwd_kb2_kernel` and the GEN_VEC trapezoid adjoint all carry the
    # half branch now, next to Ka / Kbc-hb / Kc.  The remaining refusals
    # are by NAME, below: a GROUPED C (read) map, and any schedule that
    # would route a half-layout tile through a kernel that has no half
    # branch (Kinv, the host fold, the deferred rotation).
    #
    # A grouped B (WRITE) map IS served: the production S-row cell runs
    # the sibling gate at Gb = R, and every half body walks it exactly
    # as the interleaved one does -- `bg`/`cg` in Kb1 / Kb2 / the
    # forward, the `GB` group walk in the trapezoid adjoint.  A grouped
    # C map is not: it gives the adjoint a per-head C TILE, which is not
    # a factor diagonal in n, so it leaves the head-vectorised branch
    # (`GEN_VEC`) for the per-head one, which has no half form.
    if pair_half and Gc > 1:
        raise NotImplementedError(
            "pair_layout='half' does not serve a GROUPED C (read) map "
            f"(Gc={Gc}): the trapezoid adjoint's head-vectorised branch "
            "needs the per-head factors diagonal in the state axis")
    use_hb = (fuse_kb2 and O == 1 and R == 1 and not bc_grouped
              and _env("BWD_HB", "1") == "1")

    # ── where the per-token phase is applied ────────
    # "kernel": B/C reach Ka / Kbc-hb / Kc RAW and each forms
    # B~ = R(theta_t)^T B on load; Kbc-hb un-rotates its two accumulators
    # and accumulates dTheta itself, so the host does no rotation at all.
    # "host": the 4-tensor pre/post rotation below, which is what the
    # generic (Kb1 + Kinv) branch needs and what R > 1 / O > 1 / grouped
    # B/C take.  Both compute the same function.
    # SSD_MIMO_BWD_ROT_INKERNEL=0 forces the host form (the A/B lever).
    # The GENERIC branch (Kb1 + Kb2 / Kc) carries
    # the phase too -- each of those kernels owns ONE head per CTA, so the
    # per-(kernel head) slot is the same load at a shifted base, and each
    # un-rotates its own accumulator before its atomics.  That is what
    # lifts the envelope to R, O in {1, 2, 4, 8} for BOTH laws; use_hb
    # (R = O = 1) keeps the owner-accumulate fast path unchanged.
    rot_in_kernel = (has_rot
                     and _env("BWD_ROT_INKERNEL", "1") == "1")
    # bias_pre_rot (Mamba-3's fold order) lives ONLY on the in-kernel
    # per-head route: with beta/gamma inside the rotation the rotated tile
    # is per head whatever the phase is, so the hoist Kbc-hb depends on is
    # invalid, and the host fold cannot express it at all.
    pre_rot = bool(bias_pre_rot)
    # bias_pre_rot=True with NO rotation
    # reaching the scan (no theta, no theta_h, no rot_bias) is a NO-OP,
    # not a refusal -- the identity rotation makes "before"/"after" it
    # the same map.  Still refused: rot_bias (retired under this fold
    # order), a rotation present but outside the in-kernel per-head
    # route, or theta_h alone with no theta (a real, non-identity
    # READ-side rotation with no defined bias_pre_rot fold order).
    if pre_rot and (has_rot_bias or (has_rot and not rot_in_kernel)
                    or (not has_rot and has_theta_h)):
        raise NotImplementedError(
            "ssd bwd: bias_pre_rot needs the in-kernel per-head phase, or "
            "no rotation at all (theta=None, theta_h=None, rot_bias=None) "
            "(rot_in_kernel=%r, theta=%r, theta_h=%r, rot_bias=%r)"
            % (rot_in_kernel, has_rot, has_theta_h, has_rot_bias))

    if has_rot and theta.dim() == 5 and not rot_in_kernel:
        # A per-head phase has no host form: the fold below rotates the
        # head-SHARED B/C tensors.  Refuse rather than broadcast head 0.
        raise NotImplementedError(
            "ssd bwd: a PER-HEAD phase needs the in-kernel backward "
            "(use_hb=%r, SSD_MIMO_BWD_ROT_INKERNEL=%r); the host fold "
            "acts on head-shared B/C." % (use_hb, _env("BWD_ROT_INKERNEL", "1")))

    # Pre-rotate B, C by R(theta)^T so the kernels see rotated tensors
    if has_rot and not rot_in_kernel:
        theta_f = theta.float()  # [B, nc, T, N//2]
        if R > 1:
            Bt = Btil.float().transpose(-1, -2).contiguous()
            th_b = theta_f.unsqueeze(3).expand(-1, -1, -1, R, -1)
            Btil = _rotate_tile_fwd(Bt, th_b).transpose(-1, -2).contiguous().to(Btil.dtype)
        else:
            Btil = _rotate_tile_fwd(Btil.float(), theta_f).to(Btil.dtype).contiguous()
        if O > 1:
            Ct = Ctil.float().transpose(-1, -2).contiguous()
            th_c = theta_f.unsqueeze(3).expand(-1, -1, -1, O, -1)
            Ctil = _rotate_tile_fwd(Ct, th_c).transpose(-1, -2).contiguous().to(Ctil.dtype)
        else:
            Ctil = _rotate_tile_fwd(Ctil.float(), theta_f).to(Ctil.dtype).contiguous()

    # dX: Kc fully covers every (b, h, g, r, l) element with tl.store
    # (CHUNK_PARALLEL grid spans all chunks incl. padding), so no zero-fill
    # is needed; stored in the IO dtype (Kc's fp32 tile downcasts on store).
    #
    # LAYOUT.  Kc addresses `X_ptr` and `dX_ptr` through
    # the SAME stride set, so dX must be allocated in x's layout, not in
    # the dense one `torch.empty(shape)` would give.  That is also the
    # layout the CALLER wants back: `ssd_mimo_scan` now accepts a
    # PERMUTED VIEW of x, so a caller whose natural
    # order is [B, L, H, P] hands in `x.permute(0, 2, 3, 1)` and gets a
    # gradient its own permute-backward reads as a view -- no
    # `.contiguous()` on either pass.  `empty_strided` is exact here
    # because the entry point admits only non-overlapping dense layouts.
    dX = torch.empty_strided(x_scaled.shape, x_scaled.stride(),
                             device=device, dtype=x_scaled.dtype)
    dDelta_out = torch.zeros_like(delta_cumsum)

    # ── Head-sliced atomic accumulation (SLOTS) ──────────────────────
    # dBtil/dCtil are head-SHARED, so every atomic tile is contended by
    # all H heads' CTAs.  Slot the buffers by h % SLOTS (contention / S)
    # and sum the slots host-side after the kernels.  Requires fuse_kb2
    # (otherwise Kb2 would write the unslotted layout).
    # DEFAULT OFF (=1): L2 handles the 32-way fp32 atomic contention fine,
    # and the slot zero-fill + host sum cost more than the contention.  Kept
    # as an A/B lever for other architectures.
    n_slots = int(_env("DB_SLOTS", "1"))
    use_slots = ((not bc_grouped) and fuse_kb2 and not use_hb
                 and n_slots > 1 and H > n_slots)
    if use_slots:
        dBtil_acc = torch.zeros((n_slots,) + tuple(Btil.shape),
                                device=device, dtype=f32)
        s_slb_v = dBtil_acc.stride(0)
        dBtil_out = dBtil_acc[0]  # stride template; real value after sum
    else:
        # layout-preserving: keeps R/O-major memory (coalesced atomics)
        dBtil_acc = dBtil_out = torch.zeros_like(Btil, dtype=f32)
        s_slb_v = 0
    use_slots_c = use_slots and not has_theta_h
    if use_slots_c:
        dCtil_acc = torch.zeros((n_slots,) + tuple(Ctil.shape),
                                device=device, dtype=f32)
        s_slc_v = dCtil_acc.stride(0)
        dCtil_out = dCtil_acc[0]
    else:
        dCtil_acc = dCtil_out = torch.zeros_like(Ctil, dtype=f32)
        s_slc_v = 0

    dummy = torch.empty(1, device=device, dtype=f32)
    thh_c = theta_h.float().contiguous() if has_theta_h else dummy
    dsf_c = dS_final.float().contiguous() if has_dsf else dummy
    beta_c = beta.float().contiguous() if has_beta else dummy
    gamma_c = gamma.float().contiguous() if has_gamma else dummy
    rb_c = rot_bias.float().contiguous() if has_rot_bias else dummy
    dThetaH_out = (torch.zeros(H, N // 2, device=device, dtype=f32)
                   if need_dthh else dummy)
    dBeta_out = (torch.zeros(H, N, R, device=device, dtype=f32)
                 if has_beta else dummy)
    dGamma_buf = (torch.zeros(H, N, O, device=device, dtype=f32)
                  if has_gamma else dummy)
    dRotBias_out = (torch.zeros(H, N // 2, device=device, dtype=f32)
                    if has_rot_bias else dummy)
    # in-kernel phase: the [B, nc, T, N//2] operand and its gradient
    # M3_STRUCTURE stage 2: the saved phase is bf16 on the
    # 16-bit IO routes (`_SSDMimoFn`, `_theta_bf16_ok`) and is read AS IS
    # -- every `Theta_ptr` load in Kb1 / Kbc-hb / Kc / K2 upcasts in
    # registers (`.to(tl.float32)`), so the pointer is dtype-agnostic and
    # the fp32 upcast pass over [B, nc, T, H, K] is gone.  Its gradient
    # buffer stays fp32 regardless: the kernels accumulate dTheta with
    # `atomic_add`, which needs the fp32 zero-fill.
    th_c_k = ((theta if theta.dtype == torch.bfloat16 else theta.float())
              .contiguous() if rot_in_kernel else dummy)
    # (b, chunk, t, HEAD, pair).  A 4-d phase is head-SHARED (head stride
    # 0); a 5-d [B, nc, T, H, K] one is PER HEAD and switches
    # Kbc-hb to its ROT_PER_HEAD branch, where the rotation moves inside
    # the head loop because B~/C~ are no longer one tile for the block.
    rot_per_head = rot_in_kernel and th_c_k.dim() == 5
    # SSD_SPEED: when Kb1 CAN be the sole dTheta writer
    # (`_kb1_dth_store`'s static terms), the buffer starts UNINITIALISED and
    # is zero-filled only if the launch that finally runs does not take the
    # store (head block refused, fold refused, OOR fallback) -- see
    # `_dth_zero` below.  Every other route keeps the zero-fill here.
    # The store needs a PER-HEAD phase: a head-SHARED one ([B, nc, T, K],
    # head stride 0) makes every head's CTA write the SAME tile, so a plain
    # store keeps one head instead of the head sum.
    dth_store_possible = bool(
        rot_in_kernel and rot_per_head
        and not pair_half and not fuse_kb2 and R * O > 1
        and NGT == 1 and not has_rot_bias and _ROT_ABL_MIMO == 0
        and _env("BWD_DTH_STORE", "1") == "1")
    # SSD bwdwins: the owner-accumulate Kbc-hb (`use_hb`) is the SOLE dTheta writer of the R = O = 1
    # route (Kc runs NO_DBAGG: `KC_DTH` is False; no Kb1 / Kb2 / Kinv launch) and, under a PER-HEAD phase with
    # one G tile (`dth_sole`, kernel-side `NGT == 1`), it stores every [b, c, t, h, k < N/2] element exactly once
    # (its grid is (b, c, head block, N slice)).  The 100 MB (LLM cell) zero-fill is then dead.  The store was
    # already the one taken into the zeroed buffer, so the result is bit-identical.
    dth_hb_sole = bool(
        rot_in_kernel and rot_per_head and use_hb and NGT == 1
        and not has_rot_bias and _ROT_ABL_MIMO == 0
        and int(_env("BWD_ROT_ABL", "0")) == 0
        and _env("BWD_DTH_HB_EMPTY", "1") == "1")
    # NGT > 1 (the backward's G tile splits G: Blackwell's smem rule picks GT = 32 at G = 64): the G-tile CTAs of one
    # (b, c, head) each add their partial, so the buffer must start at zero.
    dTheta_buf = ((torch.empty if (dth_store_possible or dth_hb_sole) else torch.zeros)(
                      th_c_k.shape, device=device, dtype=f32)
                  if rot_in_kernel else dummy)
    if rot_in_kernel and (dth_store_possible or dth_hb_sole):
        nan_debug_fill(dTheta_buf)
    _dth_state = {"zeroed": not (dth_store_possible or dth_hb_sole)}

    def _dth_zero():
        if not _dth_state["zeroed"]:
            dTheta_buf.zero_()
            _dth_state["zeroed"] = True

    def _th5(t):
        if not rot_in_kernel:
            return [0, 0, 0, 0, 0]
        st = list(t.stride())
        return st if len(st) == 5 else [st[0], st[1], st[2], 0, st[3]]

    th_st_k = _th5(th_c_k)
    dth_st_k = _th5(dTheta_buf)
    # Kb1 / Kb2 / Kc take the dTheta buffer by NAME (it is in their
    # autotune `restore_value`, which the tuner resolves out of the call's
    # arguments, so it is passed on EVERY call, rotation or not).
    dth_kw = dict(dTheta_ptr=dTheta_buf,
                  s_dtb=dth_st_k[0], s_dtc=dth_st_k[1],
                  s_dtt=dth_st_k[2], s_dth=dth_st_k[3],
                  s_dtk=dth_st_k[4])
    # ── the phase cumsum, in-kernel ──────────────────────────────
    # With `theta_off` the `theta` operand is the per-token INCREMENT and
    # the kernels rebuild the phase as off[c] + intra-chunk cumsum.  The
    # gradient splits the same way: `dTheta_buf` comes back as the
    # increment gradient (the intra-chunk REVERSE cumsum, formed in
    # Kbc-hb) and `dThOff_buf` as the per-chunk offset gradient.  The
    # host graph that built `theta_off` from the increments carries the
    # cross-chunk term, on a tensor 64x smaller than the per-token one.
    phase_cumsum = rot_in_kernel and theta_off is not None
    if theta_off is not None and not rot_in_kernel:
        raise NotImplementedError(
            "ssd bwd: theta_off (the in-kernel phase cumsum) needs the "
            "in-kernel rotation route (rot_in_kernel=%r)" % (rot_in_kernel,))
    off_c = theta_off.float().contiguous() if phase_cumsum else dummy
    # ── DEFER the reparameterisation adjoint ─────────
    # The in-chunk reverse cumsum and the chunk total are LINEAR in the
    # phase gradient, so they can run once over the finished buffer
    # instead of once per HEAD inside five register-starved kernels.
    # `phase_prefix_bwd` then does all THREE terms of the adjoint in the
    # single streaming pass `phase_offset_bwd_add` already cost.
    # SSD_MIMO_BWD_PHASE_DEFER=0 restores the in-kernel form.
    phase_defer = phase_cumsum and _env("BWD_PHASE_DEFER", "1") == "1"
    dThOff_buf = (torch.zeros_like(off_c)
                  if (phase_cumsum and not phase_defer) else dummy)

    def _off4(t):
        st = list(t.stride())
        return [st[0], 0 if t.shape[1] == 1 else st[1], st[2], st[3]]

    # Passed ONLY when the feature is on -- an unconditional kwargs dict
    # on three launches costs the launch-bound native configs 1-8% of
    # wall for a branch that compiles out (see the fused forward's own
    # note, and `trap_kw`).
    phase_kw = {}
    # `dThOff_ptr` is ALWAYS passed: it is in Kbc-hb's `restore_value`,
    # which the autotuner resolves by NAME out of the call's arguments,
    # so omitting it raises `KeyError` during tuning.  One extra argument
    # on one of the five launches; its four strides still ride the
    # conditional dict.
    doff_kw = dict(dThOff_ptr=dThOff_buf)
    if phase_cumsum:
        off_st = _off4(off_c)
        doff_st = ([0, 0, 0, 0] if phase_defer else _off4(dThOff_buf))
        phase_kw = dict(
            ThOff_ptr=off_c,
            s_ob=off_st[0], s_oh=off_st[1], s_oc=off_st[2], s_ok=off_st[3],
            PHASE_CUMSUM=True,
            PHASE_DEFER=phase_defer,
        )
        doff_kw.update(
            s_dob=doff_st[0], s_doh=doff_st[1],
            s_doc=doff_st[2], s_dok=doff_st[3],
        )
    dX0_out = (torch.empty(Bsz, H, G_SIZE, N, device=device, dtype=f32)
               if has_x0 else dummy)
    x0_st = list(dX0_out.stride()) if has_x0 else [0, 0, 0, 0]

    has_dw = dw is not None
    if has_dw:
        dw_c = dw
        dDW_out = torch.zeros(Bsz, H, nc * T, device=device, dtype=f32)
        dw_st = list(dw_c.stride())
        ddw_st = list(dDW_out.stride())
    else:
        dw_c = dummy
        dDW_out = dummy
        dw_st = [0, 0, 0]
        ddw_st = [0, 0, 0]
    dw_kwargs = dict(DW_ptr=dw_c, s_wb=dw_st[0], s_wh=dw_st[1],
                     s_wl=dw_st[2], HAS_DW=has_dw)

    # B strides
    if bc_grouped:
        # grouped tensors carry the group axis at dim 2 and an OPTIONAL
        # trailing R/O axis (dim 5).  Dropping that trailing stride made
        # every r/o stream read stream 0 (silently wrong grouped MIMO).
        s_bb, s_bc, s_bh_v, s_bt, s_bn = Btil.stride()[:5]
        s_cb, s_cc, s_ch_v, s_ct, s_cn = Ctil.stride()[:5]
        s_br_v = Btil.stride(5) if Btil.dim() == 6 else 0
        s_co_v = Ctil.stride(5) if Ctil.dim() == 6 else 0
    else:
        b_has_r = Btil.dim() >= 5
        if b_has_r:
            s_bb, s_bc, s_bt, s_bn, s_br_v = Btil.stride()[:5]
        else:
            s_bb, s_bc, s_bt, s_bn = Btil.stride()[:4]
            s_br_v = 0
        s_bh_v = 0
        c_has_o = Ctil.dim() >= 5
        if c_has_o:
            s_cb, s_cc, s_ct, s_cn, s_co_v = Ctil.stride()[:5]
        else:
            s_cb, s_cc, s_ct, s_cn = Ctil.stride()[:4]
            s_co_v = 0
        s_ch_v = 0

    # dBtil / dCtil strides.  GROUPED tensors are [B, nc, Gx, T, N]
    # (5-dim = GROUP axis, fused implies R = O = axis-free); ungrouped
    # 5-dim tensors are [B, nc, T, N, R/O].  Confusing dim-2 (group) with
    # dim-3/4 (T/N/R) scatters grouped dB/dC gradients.
    if bc_grouped:
        s_dbb, s_dbc, s_dbh_v, s_dbt, s_dbn = dBtil_out.stride()[:5]
        s_dbr_v = dBtil_out.stride(5) if dBtil_out.dim() == 6 else 0
        s_dcb, s_dcc, s_dch_v, s_dct, s_dcn = dCtil_out.stride()[:5]
        s_dco_v = dCtil_out.stride(5) if dCtil_out.dim() == 6 else 0
    else:
        s_dbh_v = s_dch_v = 0
        if dBtil_out.dim() >= 5:
            s_dbb, s_dbc, s_dbt, s_dbn, s_dbr_v = dBtil_out.stride()[:5]
        else:
            s_dbb, s_dbc, s_dbt, s_dbn = dBtil_out.stride()[:4]
            s_dbr_v = 0
        if dCtil_out.dim() >= 5:
            s_dcb, s_dcc, s_dct, s_dcn, s_dco_v = dCtil_out.stride()[:5]
        else:
            s_dcb, s_dcc, s_dct, s_dcn = dCtil_out.stride()[:4]
            s_dco_v = 0

    hpg_b = H // Gb
    hpg_c = H // Gc
    dsf_st = list(dsf_c.stride()) if has_dsf else [0, 0, 0, 0]

    grid = (Bsz, H, NGT)
    use_r_tile = R > 1

    # Intermediate buffers
    # ── DG_LO: Ka's per-chunk state
    # adjoint dG_all [B, H, nc, N, G] is read ONLY as an MMA operand
    # (`dG.to(MMA_DT)` in Kbc's aggregate dot and in Kc's B dG / aggregate
    # dots) -- except under rot_bias, whose Kbc rotates it in fp32 first.
    # Storing it in the MMA operand dtype is then the SAME rounding (RNE on
    # store == RNE at the cast) done once instead of per consumer: bitwise
    # identical gradients, half the traffic of the backward's largest
    # scratch (268 MB fp32 at B8 L2048 H32 N128 G64 T64).  Lever
    # `SSD_MIMO_BWD_DG_LO` ("1" on, "0"/unset = the fp32 buffer).
    dg_dtype = f32
    # R = O = 1 only: the opt-in Gluon Kb1 (R*O > 1) TMA-loads dG_all as
    # fp32 and is not taught the narrow buffer.
    # KA_NODALPHA reads dG_all in Kbc for the fp32 dalpha reduction
    # sum(dG * S); a bf16 dG_all rounds that reduction's operand and
    # measurably grows the fp64 error of dt_proj.bias (m2exact), so the two
    # levers do not stack: NODALPHA keeps the fp32 buffer.
    from . import levers as _lv
    _dg_ok = (mma_dt is not tl.float32 and _env("BWD_KA_NODALPHA", "0") != "1"
              and not has_rot_bias and R == 1 and O == 1)
    if _dg_ok and _lv.on("real_dg_lo", _lv.real_group(T),
                         guard=(not has_theta_h and not pair_half)):
        dg_dtype = (torch.bfloat16 if mma_dt is tl.bfloat16
                    else torch.float16)
    dG_all = torch.empty(Bsz, H, nc, N, G_SIZE, device=device, dtype=dg_dtype)
    dalpha_all = torch.empty(Bsz, H, NGT, nc, device=device, dtype=f32)
    # dCrd_buf: flat buffer [Bsz*H*nc*O*T*N] for deferred rotation.
    # Allocated HERE, before the IDX64 decision, because it is the LARGEST
    # scratch the backward addresses (B*H*nc*O*T*N = B*L*H*O*N elements)
    # and it is indexed through a b/h product — exactly the product IDX64
    # exists to promote.  IDX64 must be decided with it, or a buffer that can
    # cross 2^31 is addressed in int32 while every caller operand stays under
    # it.
    # with the in-kernel phase Kb1 un-rotates theta_h in REGISTERS
    # (Kinv knows nothing about theta_t and would un-rotate only half the
    # composition), so the deferred buffer is not allocated there either.
    deferred_rot = has_theta_h and not use_hb and not rot_in_kernel
    if pair_half:
        # PAIRHALF admits exactly the schedules whose EVERY kernel has a
        # half branch:
        #   R = O = 1 : Ka + head-batched Kbc + chunk-parallel Kc
        #               (stage 1);
        #   R, O > 1  : Ka + Kb1 + Kb2 + Kc + the GEN_VEC trapezoid
        #               adjoint.
        # Both need the IN-KERNEL per-token phase: the host fold rotates
        # ADJACENT pairs, and `_bwd_kinv_kernel` (the deferred theta_h
        # un-rotation) has no half branch.  Anything else is refused BY
        # NAME rather than run on the wrong pairing.
        if not rot_in_kernel or deferred_rot:
            raise NotImplementedError(
                "pair_layout='half' needs the in-kernel rotation and no "
                f"deferred theta_h un-rotation (rot_in_kernel="
                f"{rot_in_kernel}, deferred_rot={deferred_rot})")
        if (O == 1 and R == 1) and not use_hb:
            raise NotImplementedError(
                "pair_layout='half' at R = O = 1 needs the head-batched "
                "Kbc schedule (use_hb=False)")
        if (O > 1 or R > 1) and use_hb:
            raise NotImplementedError(
                "pair_layout='half': the head-batched Kbc schedule is "
                "the R = O = 1 route only")
    if deferred_rot:
        # Preserve the sole-owner store path at NGT=1. At NGT>1 Kb1
        # atomically sums channel-tile partials; the tuner restores this
        # zero base between trials just like its other atomic outputs.
        alloc_dcrd = torch.empty if NGT == 1 else torch.zeros
        dCrd_buf = alloc_dcrd(Bsz * H * nc * O * T * N, device=device,
                              dtype=f32)
    else:
        dCrd_buf = dummy
    from .fused_fwd_kernel import _needs_i64
    # EVERY tensor the five kernels address, internally allocated scratch
    # included — see _needs_i64's docstring.
    idx64 = _needs_i64(x_scaled, dY, S_all, dG_all, dX,
                       dBtil_out, dCtil_out, dalpha_all, dCrd_buf,
                       Btil, Ctil, dDelta_out, dDW_out, dX0_out, dw_c,
                       delta_cumsum, dsf_c, th_c_k, dTheta_buf,
                       off_c if phase_cumsum else None,
                       dThOff_buf if phase_cumsum else None)

    # Ka: v recurrence → dG_all, dalpha_all (sequential, 1 GEMM/chunk).
    # Segmented on under-occupied long-nc shapes exactly like the FWD
    # (see _seg_plan): phase A computes per-segment zero-seeded final v
    # (V_s), a host stitch builds the carry entering each segment
    # (linearity of v = alpha*v + r, reverse direction), and phase C
    # reruns Ka per segment seeded through the dSF path.
    from .fused_fwd_kernel import _seg_plan
    nseg_ka, nc_seg_ka = _seg_plan(nc, Bsz * H * NGT, device)
    # CHUNKPAR (chunkpar.py): Ka as a chunk-parallel readout
    # plus a reverse state pass parallel over (b, h, G tile, n-block) --
    # replaces BOTH the serial walk and the segmented plan when the serial
    # grid would be under-occupied.
    from .chunkpar import chunkpar_wanted, cp_ka_refusal, cp_ka
    cp_ka_why = cp_ka_refusal(
        rot_in_kernel=rot_in_kernel, pair_half=bool(pair_half),
        has_theta_h=has_theta_h, O=O, dcs_ink=dcs_ink, seg_ka=False)
    use_cp_ka = (cp_ka_why is None
                 and _env("BWD_KA_NODALPHA", "0") != "1"
                 and chunkpar_wanted(Bsz * H, nc, device, part="ka", dtype=x_scaled.dtype))
    _CP_LAST_ROUTE["ka"] = "chunkpar" if use_cp_ka else (cp_ka_why or "serial")
    if _env("CP_TRACE", "0") == "1":
        print("CP_TRACE ka=" + _CP_LAST_ROUTE["ka"], flush=True)
    if use_cp_ka:
        nseg_ka = 1
    # GLUON_KBC_DALPHA (kernel option `gluon_kbc_dalpha`, default off: see below): when
    # the Gluon Kbc serves, it forms dalpha_c = sum(dG_c * S_c) from the
    # S / dG tiles it stages anyway, and Ka skips its S_all re-read.  The
    # Gluon envelope decision is made ONCE, HERE, before Ka, from exactly
    # the inputs the Kbc launch uses (the same expressions, hoisted); the
    # launch reuses it.  If the Gluon Kbc will not serve, Ka forms dalpha
    # as before.
    if _portable.gluon_refusal("ssd_mimo") is None:       # kernel_target="portable": no Gluon (not even its import)
        from . import gluon_kbc_hb as _gkbc
    else:
        _gkbc = None
    _ka_nod_x = bool(
        not use_cp_ka and nseg_ka == 1
        and _env("BWD_KA_NODALPHA", "0") == "1" and use_hb and not pair_half
        and not rot_in_kernel and not rot_per_head and not has_theta_h
        and not has_rot_bias and R == 1 and O == 1)
    _cb_ok_x = kc_shared_cb(rot_in_kernel=rot_in_kernel,
                            has_theta_h=has_theta_h,
                            has_rot_bias=has_rot_bias, pre_rot=pre_rot,
                            pair_half=pair_half, R=R, O=O,
                            bc_grouped=bc_grouped, use_r_tile=use_r_tile, T=T, dtype=x_scaled.dtype)
    _dd_x = bool(_env("BWD_DD_IN_KBC", "0") == "1" and use_hb and _cb_ok_x
                 and fuse_kb2 and not has_beta and not has_gamma
                 and not rot_in_kernel and not rot_per_head
                 and not has_theta_h and not has_rot_bias and not pair_half)
    _rot_abl_x = int(_env("BWD_ROT_ABL", "0"))
    _sumds_x = kbc_sumds(rot_in_kernel=rot_in_kernel, rot_per_head=rot_per_head,
                         has_theta_h=has_theta_h, has_rot_bias=has_rot_bias,
                         pre_rot=pre_rot, pair_half=pair_half,
                         rot_abl=_rot_abl_x)
    _split_x = bool(_env("BWD_KBC_SPLIT", "0") == "1"
                    and not rot_in_kernel and not rot_per_head
                    and not has_theta_h and not has_rot_bias
                    and not pre_rot and not pair_half and not _rot_abl_x
                    and not _ka_nod_x and not _dd_x and not _sumds_x)
    _ns_x = _hb_nsplit(N, rot_per_head, mma_io)
    if pair_half:
        _ns_x = max(1, min(_ns_x, N // 32))
    _gl_why_x = "lever off"
    _gl_guard = (R == 1 and O == 1 and T == 64 and not has_theta_h
                 and not rot_in_kernel and not rot_per_head
                 and not has_rot_bias and not pair_half and mma_io)
    # EXPERIMENT lever (default off; SSD_MIMO_CHART_FUSED_SM90=1): the fused chart emitter's Gluon CHART body also
    # where the plain Gluon Kbc is not the route (sm_90's mma.sync, the calibrated lever off) -- its dB~ / dC~ then
    # come from the Gluon body instead of the Triton Kbc (same arithmetic class, not the same rounding)
    _chart_sm90 = bool(chart is not None and chart.get("emitter", "fused") == "fused"
                       and _env("CHART_FUSED_SM90", "0") == "1" and _gl_guard)
    if _gkbc is not None and use_hb and (_chart_sm90 or _gkbc.gluon_kbc_wanted(guard=_gl_guard, T=T)):
        _gl_why_x = _gkbc.refusal(allow_sm90=_chart_sm90,
            device=device, mma_io=mma_io, rot_in_kernel=rot_in_kernel,
            rot_per_head=rot_per_head, has_theta_h=has_theta_h,
            has_rot_bias=has_rot_bias, has_beta=has_beta,
            has_gamma=has_gamma, pre_rot=pre_rot, pair_half=pair_half,
            sumds=_sumds_x, dd_in_kbc=_dd_x, ka_nodalpha=_ka_nod_x,
            kbc_split=_split_x, rot_abl=_rot_abl_x,
            ts_abl=int(_env("BWD_TS_ABL", "0")), T=T, N=N, GT=GT,
            nsplit=_ns_x, trap_diag=bool(fuse_trap), dcs_ink=bool(dcs_ink))
        if _gl_why_x is None and not (
                mma_dt is tl.bfloat16
                and Ctil.dtype == Btil.dtype == torch.bfloat16
                and x_scaled.dtype == torch.bfloat16):
            _gl_why_x = (f"dtypes (C {Ctil.dtype}, B {Btil.dtype}, "
                         f"x {x_scaled.dtype}, mma {mma_dt}): bf16 only")
    # The dalpha HAND-OFF (Gluon forms sum(dG * S) from its bf16-staged tiles) FAILED the mixer-level fp64
    # oracle on A100 and Blackwell (dX, dt, a_log gradients wrong by 1e1-1e3 x the bound); root cause found
    # (the Gluon Kbc's dG slot overwrote the loop-carried S slot from the second head on: `sDG`),
    # and with the fix it passes on sm_120 (the mixer-level fp64 oracle pins both).  Still OFF by
    # default: only an explicit `gluon_kbc_dalpha` kernel option turns it on.
    _dal_on = (_env("GLUON_KBC_DALPHA", "") == "1")
    _gl_dalpha = bool(_gl_why_x is None and nseg_ka == 1 and _dal_on)
    if _gl_dalpha:
        dalpha_all.zero_()           # the Gluon Kbc accumulates into it
    from .chunkpar import cp_resource_refused, cp_resource_refuse
    cp_key = (str(device), str(mma_dt), str(dY.dtype), T, N, G_SIZE, GT)
    _cp_ka_done = False
    ka_common = dict(
        KA_WB=ka_wave_bucket(Bsz * H * NGT * nseg_ka, device),
        T=T, N=N, G_SIZE=G_SIZE, GT=GT,
        PREC=precision,
        MMA_DT=mma_dt,
        MMA_IO=int(mma_dt is not tl.float32),
        HAS_THETA_H=has_theta_h,
        HAS_GAMMA=has_gamma,
        HPG_C=hpg_c,
        O=O, s_co=s_co_v,
        IDX64=idx64,
        Theta_ptr=th_c_k,
        s_thb=th_st_k[0], s_thc=th_st_k[1],
        s_tht=th_st_k[2], s_thh=th_st_k[3], s_thk=th_st_k[4],
        HAS_ROT=rot_in_kernel,
        BIAS_PRE_ROT=pre_rot,
        PAIR_HALF=bool(pair_half),
        **phase_kw,
    )
    ka_args = (Ctil, delta_cumsum, a_log,
               thh_c, S_all,
               gamma_c,
               dY)
    ka_strides = (H,
                  s_cb, s_cc, s_ct, s_cn,
                  *delta_cumsum.stride(),
                  *dY.stride()[:4], dY.stride(5), dY.stride(4),
                  *S_all.stride())
    if nseg_ka > 1:
        # phase A: per-segment zero-seeded final v
        v_seg = torch.empty(Bsz, H, nseg_ka, G_SIZE, N,
                            device=device, dtype=f32)
        _ka_launch((Bsz, H, NGT * nseg_ka),
            *ka_args, dummy,
            dG_all, dalpha_all, v_seg,
            *ka_strides,
            0, 0, 0, 0,
            *dG_all.stride(),
            *dalpha_all.stride(),
            s_ch_v,
            nc, nc_seg_ka,
            0,                    # s_dsfs
            v_seg.stride(2),      # s_x0ss
            HAS_DSF=False,
            HAS_X0=False,
            SEG_SUMMARY=True,
            s_x0b=v_seg.stride(0), s_x0h=v_seg.stride(1),
            s_x0g=v_seg.stride(3), s_x0n=v_seg.stride(4),
            **ka_common,
            **dcs_kw,
        )
        # one-launch reverse stitch: carry[last] = dS_final (or 0);
        # carry[s] = alpha_seg[s+1] * carry[s+1] + V[s+1]
        from .fused_fwd_kernel import _seg_alpha, _seg_stitch
        alpha_seg = _seg_alpha(a_log, delta_cumsum, nseg_ka, nc_seg_ka,
                               dcs_ink=dcs_ink)
        carry = _seg_stitch(v_seg, alpha_seg,
                            dsf_c.contiguous() if has_dsf else None,
                            reverse=True)
        # phase C: real stores, seeded from carry via the dSF path
        _ka_launch((Bsz, H, NGT * nseg_ka),
            *ka_args, carry,
            dG_all, dalpha_all, dX0_out,
            *ka_strides,
            carry.stride(0), carry.stride(1),
            carry.stride(3), carry.stride(4),
            *dG_all.stride(),
            *dalpha_all.stride(),
            s_ch_v,
            nc, nc_seg_ka,
            carry.stride(2),      # s_dsfs
            0,                    # s_x0ss
            HAS_DSF=True,
            HAS_X0=has_x0,
            SEG_SUMMARY=False,
            s_x0b=x0_st[0], s_x0h=x0_st[1],
            s_x0g=x0_st[2], s_x0n=x0_st[3],
            **ka_common,
            **dcs_kw,
        )
        del v_seg, carry
    elif use_cp_ka and not cp_resource_refused("ka", cp_key):
        try:
            cp_ka(Ctil=Ctil, s_c=(s_cb, s_cc, s_ct, s_cn), s_ch=s_ch_v,
                  delta_cumsum=delta_cumsum, a_log=a_log, gamma_c=gamma_c,
                  has_gamma=has_gamma, dY=dY,
                  s_y=(*dY.stride()[:4], dY.stride(5)), S_all=S_all,
                  dsf_c=dsf_c, dsf_st=dsf_st, has_dsf=has_dsf,
                  dG_all=dG_all, dalpha_all=dalpha_all, dX0_out=dX0_out,
                  x0_st=x0_st, has_x0=has_x0, H=H, nc=nc, T=T, N=N,
                  G_SIZE=G_SIZE, GT=GT, NGT=NGT, HPG_C=hpg_c,
                  precision=precision, mma_dt=mma_dt, idx64=idx64,
                  no_dalpha=_gl_dalpha)
            _cp_ka_done = True
        except triton.runtime.errors.OutOfResources as _oor:
            # a configuration the chunk-parallel kernels cannot hold (fp32
            # tiles on a 100 KB part): refused BY NAME for this key from now
            # on, and the serial Ka below serves this call (it overwrites
            # every dG / dalpha / dX0 entry the partial attempt touched)
            cp_resource_refuse("ka", cp_key, _oor)
            _CP_LAST_ROUTE["ka"] = "serial (chunkpar refused: " + str(_oor)[:80] + ")"
    # (a plain default, NOT `locals().get(...)`: under CPython <= 3.12 `locals()` caches a snapshot of every local on
    # the frame, which kept dG_all / dalpha_all / ... alive past their `del` until this function returned -- +0.1 GiB of
    # peak at the 124M cell)
    ka_nodalpha = False
    if not (nseg_ka > 1 or _cp_ka_done):
        # KA_NODALPHA: dalpha moves to Kbc (see the constexprs' notes)
        ka_nodalpha = bool(
            _env("BWD_KA_NODALPHA", "0") == "1" and use_hb and not pair_half
            and not rot_in_kernel and not rot_per_head and not has_theta_h
            and not has_rot_bias and R == 1 and O == 1)
        if ka_nodalpha and not _gl_dalpha:
            dalpha_all.zero_()
        # KA_GT (with KA_NODALPHA only: Ka then writes no per-G-tile
        # dalpha, and dG_all is tile-agnostic): a NARROWER G tile for this
        # one sequential kernel -- more CTAs over the same chunk walk,
        # i.e. more latency hiding for the kernel that cannot be made
        # chunk-parallel.  Lever `SSD_MIMO_BWD_KA_GT` (16 / 32; unset =
        # the backward's own GT).
        ka_grid, ka_kw = grid, dict(ka_common)
        _gt_ka = int(_env("BWD_KA_GT", "0") or 0)
        if (ka_nodalpha and 16 <= _gt_ka < GT and G_SIZE % _gt_ka == 0
                and GT % _gt_ka == 0):
            ka_grid = (Bsz, H, G_SIZE // _gt_ka)
            ka_kw["GT"] = _gt_ka
        ka_kw["KA_WB"] = ka_wave_bucket(
            ka_grid[0] * ka_grid[1] * ka_grid[2], device)
        _ka_launch(
            ka_grid,
            *ka_args, dsf_c,
            dG_all, dalpha_all, dX0_out,
            *ka_strides,
            *dsf_st,
            *dG_all.stride(),
            *dalpha_all.stride(),
            s_ch_v,
            nc, nc,
            0, 0,
            HAS_DSF=has_dsf,
            HAS_X0=has_x0,
            SEG_SUMMARY=False,
            s_x0b=x0_st[0], s_x0h=x0_st[1],
            s_x0g=x0_st[2], s_x0n=x0_st[3],
            NO_DALPHA=bool(ka_nodalpha or _gl_dalpha),
            **ka_kw,
            **dcs_kw,
        )

    # ── P1: Kb1's owner-accumulate head block (decided here because it
    # decides whether Kb2 may run BEFORE Kb1: a sole-owner `tl.store`
    # into dBtil overwrites whatever Kb2 put there, so when Kb1 stores,
    # Kb2's atomic pass moves AFTER it.  Kc's fused dB_agg already runs
    # after Kb1 and stays an atomic_add.)
    ka_nodalpha = bool(ka_nodalpha)
    kb1_nsplit = _kb1_nsplit(N, R, O, mma_io)
    if pair_half:
        # PAIRHALF: Kb1's N split slices the PAIR axis, so each slice is
        # N/(2*NSPLIT) wide and must stay at or above the 16-lane MMA
        # floor (the same clamp `_hb_nsplit` gets on the Kbc route).
        kb1_nsplit = max(1, min(kb1_nsplit, N // 32))
    _kb1_ro, _kb1_tn = R * O, T * N
    _kb1_ieee_cap = precision == "ieee" and _kb1_ro > _KB1_HB_RO_MAX_IEEE
    _kb1_gate_refused = (use_hb or deferred_rot or use_slots or use_slots_c
                          or _kb1_ro > _KB1_HB_RO_MAX
                          or _kb1_ieee_cap
                          or _kb1_tn > _KB1_HB_TN_MAX)
    # first true term in source order, for a human-readable reason
    _kb1_reason = next(
        (name for name, val in (
            ("use_hb", use_hb), ("deferred_rot", deferred_rot),
            ("use_slots", use_slots), ("use_slots_c", use_slots_c),
            ("R*O>RO_MAX(%d>%d)" % (_kb1_ro, _KB1_HB_RO_MAX),
             _kb1_ro > _KB1_HB_RO_MAX),
            ("ieee:R*O>RO_MAX_IEEE(%d>%d)" % (_kb1_ro, _KB1_HB_RO_MAX_IEEE),
             _kb1_ieee_cap),
            ("T*N>TN_MAX(%d>%d)" % (_kb1_tn, _KB1_HB_TN_MAX),
             _kb1_tn > _KB1_HB_TN_MAX),
        ) if val), "")
    kb1_hb, kb1_db_store, kb1_dc_store = _kb1_hb(
        H, hpg_b, hpg_c, Bsz, nc, NGT, kb1_nsplit,
        refused=_kb1_gate_refused,
        device=device)
    _kb1_dispatch_rec = dict(
        R=R, O=O, T=T, N=N, G_SIZE=G_SIZE, bc_grouped=bc_grouped,
        fuse_kb2=fuse_kb2, use_hb=use_hb, deferred_rot=deferred_rot,
        use_slots=use_slots, use_slots_c=use_slots_c,
        RO=_kb1_ro, TN=_kb1_tn,
        ro_max=_KB1_HB_RO_MAX, tn_max=_KB1_HB_TN_MAX,
        gate_refused=_kb1_gate_refused, hb=kb1_hb, oor=False,
        hb_final=kb1_hb, reason=_kb1_reason,
        # NGT>1 (multiple G-tiles) demotes the sole-owner store back to
        # atomic_add regardless of HBLK -- see the DB_STORE/DC_STORE
        # definition just above (`hb >= hpg_? and NGT == 1`).  A cell
        # that engages HBLK but keeps db_store=dc_store=False still pays
        # the atomic cost the schedule exists to remove; it only wins
        # from the fewer, larger CTAs (contention / HBLK, not atomics
        # eliminated).
        NGT=NGT, hpg_b=hpg_b, hpg_c=hpg_c,
        db_store=kb1_db_store, dc_store=kb1_dc_store)

    # Kb2: aggregate-path → partial dB + partial dDelta (chunk-parallel)
    # (absorbed into Kc when fuse_kb2 — decided above with the slot layout)
    kb2_mult = nc * NGT * R if use_r_tile else nc * NGT
    grid_kb2 = (Bsz, H, kb2_mult)

    # ── Kb2's owner-accumulate head block ──────────
    # Refused on every rotating path, on PAIR_HALF, on the ablation
    # ladder and at R*O == 1, so all of those keep `_bwd_kb2_kernel`
    # bit for bit.  See `_bwd_kb2_hb_kernel` for the diagnosis.
    _kb2_hb_refused = bool(has_rot_bias or rot_in_kernel or pair_half
                           or _ROT_ABL_MIMO != 0 or R * O <= 1
                           or deferred_rot or use_slots or use_slots_c)
    kb2_hb = _kb2_hb(H, hpg_b, Bsz, nc, NGT, R,
                     refused=_kb2_hb_refused, device=device)
    if kb2_hb and (H % kb2_hb or hpg_b % kb2_hb):
        kb2_hb = 0
    grid_kb2_hb = (Bsz, H // kb2_hb, nc * NGT * R) if kb2_hb else None

    # ── Kb2 absorbed into Kb1's head block ─────────
    # Kb2-HB's envelope MINUS the in-kernel phase (`rot_in_kernel` is served,
    # gated by `_kb2_fold_rot()`), plus
    # `fuse_kb2` (there the aggregate already lives in Kc), `use_hb`
    # (Kb1 is not launched at R = O = 1) and HBLK > 0.  When it engages,
    # `_launch_kb2` never runs and Kb1's sole-owner `tl.store` has no
    # second writer at all.
    _kb2_fold_rot_refused = bool(rot_in_kernel and not _kb2_fold_rot())
    _kb2_fold_refused = bool(has_rot_bias or pair_half
                             or _ROT_ABL_MIMO != 0 or R * O <= 1
                             or deferred_rot or use_slots or use_slots_c
                             or fuse_kb2 or use_hb or _kb2_fold_rot_refused)
    kb2_fold = _kb2_fold(kb1_hb, refused=_kb2_fold_refused)
    kb2_fold_kw_on = dict(
        KB2_FOLD=True,
        dG_ptr=dG_all, s_gb=dG_all.stride(0), s_gh=dG_all.stride(1),
        s_gc=dG_all.stride(2), s_gn=dG_all.stride(3),
        s_gg=dG_all.stride(4),
        dAlpha_ptr=dalpha_all, s_dab=dalpha_all.stride(0),
        s_dah=dalpha_all.stride(1), s_dag=dalpha_all.stride(2),
        s_dac=dalpha_all.stride(3),
    ) if kb2_fold else None

    def _launch_kb2():
        if kb2_hb:
            _bwd_kb2_hb_kernel[grid_kb2_hb](
                Btil, x_scaled, delta_cumsum, a_log,
                beta_c,
                dG_all, dalpha_all,
                dBtil_out, dDelta_out, dBeta_out,
                H,
                s_bb, s_bc, s_bt, s_bn,
                *x_scaled.stride(),
                *delta_cumsum.stride(),
                *dG_all.stride(),
                *dalpha_all.stride(),
                s_dbb, s_dbc, s_dbt, s_dbn,
                s_bh_v,
                nc, NGT,
                T=T, N=N, G_SIZE=G_SIZE, GT=GT,
                PREC=precision,
                MMA_DT=mma_dt,
                MMA_IO=int(mma_dt is not tl.float32),
                HAS_BETA=has_beta,
                HPG_B=hpg_b,
                R=R, s_br=s_br_v, s_dbr=s_dbr_v,
                s_dbh=s_dbh_v,
                IDX64=idx64,
                HB2=kb2_hb,
                **dw_kwargs,
                **dcs_kw,
                OCC=_occ_bucket(grid_kb2_hb[0] * grid_kb2_hb[1] * grid_kb2_hb[2], device))
            return
        _bwd_kb2_kernel[grid_kb2](
            Btil, x_scaled, delta_cumsum, a_log,
            beta_c, rb_c,
            dG_all, dalpha_all,
            dBtil_out, dDelta_out, dBeta_out, dRotBias_out,
            H,
            s_bb, s_bc, s_bt, s_bn,
            *x_scaled.stride(),
            *delta_cumsum.stride(),
            *dG_all.stride(),
            *dalpha_all.stride(),
            s_dbb, s_dbc, s_dbt, s_dbn,
            s_bh_v,
            nc, NGT,
            T=T, N=N, G_SIZE=G_SIZE, GT=GT,
            PREC=precision,
            MMA_DT=mma_dt,
            MMA_IO=int(mma_dt is not tl.float32),
            HAS_BETA=has_beta,
            HAS_ROT_BIAS=has_rot_bias,
            HPG_B=hpg_b,
            R=R, s_br=s_br_v, s_dbr=s_dbr_v,
            R_TILE=use_r_tile,
            CHUNK_PARALLEL=True,
            s_dbh=s_dbh_v,
            IDX64=idx64,
            Theta_ptr=th_c_k,
            s_thb=th_st_k[0], s_thc=th_st_k[1],
            s_tht=th_st_k[2], s_thh=th_st_k[3], s_thk=th_st_k[4],
            HAS_ROT=rot_in_kernel,
            BIAS_PRE_ROT=pre_rot,
            ROT_ABL=_ROT_ABL_MIMO,
            PAIR_HALF=bool(pair_half),
            **dth_kw,
            **doff_kw,
            **phase_kw,
            **dw_kwargs,
            **dcs_kw,
                OCC=_occ_bucket(grid_kb2[0] * grid_kb2[1] * grid_kb2[2], device))

    # SSD_SPEED: Kb1's fused read-side rotation (ROT_COMB); the head-
    # blocked body only, so the HBLK = 0 fallback keeps its own chain.
    # OPT-IN (SSD_MIMO_BWD_KB1_ROT_COMB=1): slower on Blackwell -- the one
    # MMA-output swap per (head, o) it saves is paid back by the
    # angle-addition tiles.
    kb1_rot_comb = bool(
        rot_in_kernel and has_theta_h and not has_rot_bias and not pair_half
        and _ROT_ABL_MIMO == 0 and (pre_rot or not has_gamma)
        and _env("BWD_KB1_ROT_COMB", "0") == "1")
    _kb2_launched = False
    if not fuse_kb2 and not kb1_db_store and not kb2_fold:
        _dth_zero()          # Kb2 accumulates dTheta: it needs the zeros
        _launch_kb2()
        del dalpha_all
        _kb2_launched = True

    # (dCrd_buf is allocated above, before the IDX64 decision)

    # ── CB buffer (SHARED_CB) built once, before Kbc-hb, so DD_PAIRS can
    # read it there as well as Kc ──
    _cb_cache = {}

    def _shared_cb():
        if "cb" not in _cb_cache:
            _c4 = torch.as_strided(Ctil, (Bsz, nc, T, N),
                                   (s_cb, s_cc, s_ct, s_cn))
            _b4 = torch.as_strided(Btil, (Bsz, nc, T, N),
                                   (s_bb, s_bc, s_bt, s_bn))
            _mt = _mma_torch_dtype(mma_dt)
            # IEEE fp32 through the torch >= 2.9 precision API: the legacy
            # `allow_tf32` read raises once the caller has set
            # `fp32_precision`
            with ieee_matmul():
                _cb_cache["cb"] = torch.matmul(
                    _c4.to(_mt).float(),
                    _b4.to(_mt).float().transpose(-1, -2)).contiguous()
        return _cb_cache["cb"]

    _cb_ok = kc_shared_cb(rot_in_kernel=rot_in_kernel, has_theta_h=has_theta_h,
                          has_rot_bias=has_rot_bias, pre_rot=pre_rot,
                          pair_half=pair_half, R=R, O=O,
                          bc_grouped=bc_grouped, use_r_tile=use_r_tile, T=T, dtype=x_scaled.dtype)
    # DD_PAIRS (lever SSD_MIMO_BWD_DD_IN_KBC): see the Kbc-hb constexpr
    dd_in_kbc = bool(_env("BWD_DD_IN_KBC", "0") == "1" and use_hb and _cb_ok
                     and fuse_kb2 and not has_beta and not has_gamma
                     and not rot_in_kernel and not rot_per_head
                     and not has_theta_h and not has_rot_bias
                     and not pair_half)
    dd_kw = {}
    if dd_in_kbc:
        _cbt0 = _shared_cb()
        dd_kw = dict(DD_PAIRS=True, CB_ptr=_cbt0, s_cbb=_cbt0.stride(0),
                     s_cbc=_cbt0.stride(1))

    # Kb1: Score-path → dC_rd to temp buf + partial dB + partial dDelta (chunk-parallel)
    _chart_sums = None             # the fused chart emitter's head sums (set in the Gluon Kbc branch)
    if use_hb:
        # Owner-accumulate replacement for Kb1 (+ Kc's dB_agg): head-batched
        # grid, register accumulation, one dBtil/dCtil write per CTA.
        nsplit = _hb_nsplit(N, rot_per_head, mma_io)
        if pair_half:
            # PAIRHALF: the split slices the PAIR axis, so each slice is
            # N/(2*NSPLIT) wide and must stay at or above the 16-lane MMA
            # floor.
            nsplit = max(1, min(nsplit, N // 32))
        # MEASUREMENT ONLY: the per-head stored-map buffers the
        # (b) and (c) ablations read and write.  Contents are garbage --
        # these levels compute the wrong gradient by construction; what
        # they measure is the H-fold traffic and the loop body B0 leaves.
        # A4: the slotted bias-gradient planes (see BSLOT in the kernel)
        bslot_n = int(_env("BWD_BIAS_SLOTS", "8")) if pre_rot else 1
        bslot_n = max(1, min(bslot_n, nc))
        hb_dBeta = hb_dGamma = None
        s_bes_v = s_gas_v = 0
        if bslot_n > 1 and has_beta:
            hb_dBeta = torch.zeros(bslot_n, H, N, device=device, dtype=f32)
            s_bes_v = hb_dBeta.stride(0)
        if bslot_n > 1 and has_gamma:
            hb_dGamma = torch.zeros(bslot_n, H, N, device=device, dtype=f32)
            s_gas_v = hb_dGamma.stride(0)
        rot_abl = int(_env("BWD_ROT_ABL", "0"))
        rot_abl_kw = {}
        if rot_abl & 384:
            _rb = torch.zeros(Bsz, nc, T, H, N, device=device,
                              dtype=Btil.dtype)
            _rc = torch.zeros_like(_rb)
            _st = _rb.stride()
            rot_abl_kw = dict(BRot_ptr=_rb, CRot_ptr=_rc,
                              s_rb=_st[0], s_rc=_st[1], s_rt=_st[2],
                              s_rh=_st[3], s_rn=_st[4])
        grid_hb = lambda ns: (lambda META: (Bsz, triton.cdiv(H, META["HB"]),
                                            nc * NGT * ns))
        _kbc_ns_ladder = kbc_ns_ladder(N, nsplit, mma_io, pair_half)
        kbc_split = bool(_env("BWD_KBC_SPLIT", "0") == "1"
                         and not rot_in_kernel and not rot_per_head
                         and not has_theta_h and not has_rot_bias
                         and not pre_rot and not pair_half and not rot_abl
                         and not ka_nodalpha and not dd_in_kbc
                         and not kbc_sumds(rot_in_kernel=rot_in_kernel,
                                           rot_per_head=rot_per_head,
                                           has_theta_h=has_theta_h,
                                           has_rot_bias=has_rot_bias,
                                           pre_rot=pre_rot,
                                           pair_half=pair_half,
                                           rot_abl=rot_abl))
        # GLUON_KBC (lever `SSD_MIMO_GLUON_KBC=1`): the
        # explicit-layout twin of this body (gluon_kbc_hb.py); refused by
        # name outside its envelope, then the Triton kernel runs unchanged.
        # the envelope decision made before Ka (see GLUON_KBC_DALPHA); the
        # launch inputs recomputed above must be the ones it used
        _gl_why = _gl_why_x
        if (nsplit, dd_in_kbc, kbc_split, ka_nodalpha, rot_abl) != (
                _ns_x, _dd_x, _split_x, _ka_nod_x, _rot_abl_x):
            raise AssertionError(
                "internal: Gluon Kbc envelope inputs changed between the "
                "pre-Ka decision and the launch")
        _CP_LAST_ROUTE["kbc"] = "gluon" if _gl_why is None else (
            "triton: " + _gl_why)
        if _env("CP_TRACE", "0") == "1":
            print("CP_TRACE kbc=" + _CP_LAST_ROUTE["kbc"], flush=True)
        # the local chart cotangent, FUSED into the Gluon Kbc (chart["emitter"] "fused", the default): the
        # CHART body forms Kch1's head sums from the tiles it stages anyway; Kch2 combines them after Kc.  Any
        # refusal is by name (chart["route"]) and the two-pass emitter (Kch1 + Kch2) serves instead.
        if chart is not None:
            from .chart_emit import _kr as _chart_kr, chart_buffers as _chart_buffers
            _cks = max(16, _chart_kr(chart["mode"], int(chart["k"])))
            if chart.get("emitter", "fused") != "fused":
                chart["route"] = "kch: chart_emitter=%r" % chart.get("emitter")
            elif _gl_why is not None:
                chart["route"] = "kch: the Gluon Kbc does not serve (" + _gl_why + ")"
            else:
                _cwhy = _gkbc.chart_refusal_fused(GT=GT, NGT=NGT, N=N, nsplit=nsplit, KS=_cks,
                                                  dalpha=dalpha_all if _gl_dalpha else None)
                if _cwhy is not None:
                    chart["route"] = "kch: " + _cwhy
                else:
                    _chart_sums = _chart_buffers(Bsz, nc, 1, T, N, _cks, device)
                    chart["route"] = "fused"
        if _gl_why is None:
            _gkbc.kbc_hb_gluon(
                chart=None if _chart_sums is None else dict(
                    dCs=_chart_sums[0], dBa=_chart_sums[1], MT=_chart_sums[2], T1=_chart_sums[3], KS=_cks),
                Ctil=Ctil, Btil=Btil, x_scaled=x_scaled,
                delta_cumsum=delta_cumsum, a_log=a_log, S_all=S_all,
                dG_all=dG_all, dw_c=dw_c, dY=dY, dCtil_out=dCtil_out,
                dBtil_out=dBtil_out, dDelta_out=dDelta_out, H=H, Bsz=Bsz,
                nc=nc, NGT=NGT, nsplit=nsplit, T=T, N=N, GT=GT,
                strides=(s_cb, s_cc, s_ct, s_cn,
                         s_bb, s_bc, s_bt, s_bn,
                         x_scaled.stride(0), x_scaled.stride(1),
                         x_scaled.stride(2), x_scaled.stride(4),
                         *delta_cumsum.stride(),
                         *dY.stride()[:4], dY.stride(5),
                         *S_all.stride(),
                         *dG_all.stride(),
                         dw_st[0], dw_st[1], dw_st[2],
                         s_dcb, s_dcc, s_dct, s_dcn,
                         s_dbb, s_dbc, s_dbt, s_dbn),
                has_dw=has_dw, dcs_ink=bool(dcs_ink),
                trap_diag=bool(fuse_trap), idx64=idx64,
                beta=beta_c if has_beta else None,
                gamma=gamma_c if has_gamma else None,
                dBeta=dBeta_out if has_beta else None,
                dGamma=dGamma_buf if has_gamma else None,
                dalpha=dalpha_all if _gl_dalpha else None, gt_bwd=GT)
            _CP_LAST_ROUTE["kbc"] = _gkbc._LAST.get("route", "gluon")
            if _env("CP_TRACE", "0") == "1":
                print("CP_TRACE kbc_served=" + _CP_LAST_ROUTE["kbc"], flush=True)
        for _kpart in (() if _gl_why is None else
                       (1, 2) if kbc_split else (0,)):
          _kbc_launch(
            grid_hb, _kbc_ns_ladder,
            Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dG_all,
            dw_c,
            beta_c, gamma_c, thh_c, rb_c, th_c_k,
            dY,
            dCtil_out, dBtil_out, dDelta_out,
            dBeta_out if hb_dBeta is None else hb_dBeta,
            dGamma_buf if hb_dGamma is None else hb_dGamma,
            dThetaH_out, dRotBias_out, dTheta_buf,
            H, Bsz,
            s_cb, s_cc, s_ct, s_cn,
            s_bb, s_bc, s_bt, s_bn,
            x_scaled.stride(0), x_scaled.stride(1),
            x_scaled.stride(2), x_scaled.stride(4),
            *delta_cumsum.stride(),
            *dY.stride()[:4], dY.stride(5),
            *S_all.stride(),
            *dG_all.stride(),
            dw_st[0], dw_st[1], dw_st[2],
            s_dcb, s_dcc, s_dct, s_dcn,
            s_dbb, s_dbc, s_dbt, s_dbn,
            *th_st_k, *dth_st_k,
            nc, NGT,
            T=T, N=N, G_SIZE=G_SIZE, GT=GT,
            PREC=precision,
            MMA_DT=mma_dt,
            MMA_IO=int(mma_dt is not tl.float32),
            HAS_BETA=has_beta,
            HAS_GAMMA=has_gamma,
            HAS_DW=has_dw,
            HAS_THETA_H=has_theta_h,
            HAS_ROT_BIAS=has_rot_bias,
            HAS_ROT=rot_in_kernel,
            ROT_PER_HEAD=rot_per_head,
            BIAS_PRE_ROT=pre_rot,
            IDX64=idx64,
            ROT_DIET=_env("BWD_ROT_DIET", "1") == "1",
            ROT_ABL=rot_abl,
            SCHED=kbc_sched(mma_io),
            PAIR_HALF=bool(pair_half),
            TRAP_DIAG=bool(fuse_trap),
            BSLOT=bslot_n, s_bes=s_bes_v, s_gas=s_gas_v,
            KBC_DALPHA=ka_nodalpha,
            dAlpha_ptr=dalpha_all, s_dab=dalpha_all.stride(0),
            s_dah=dalpha_all.stride(1), s_dag=dalpha_all.stride(2),
            s_dac=dalpha_all.stride(3),
            SUMDS=kbc_sumds(rot_in_kernel=rot_in_kernel,
                            rot_per_head=rot_per_head,
                            has_theta_h=has_theta_h,
                            has_rot_bias=has_rot_bias, pre_rot=pre_rot,
                            pair_half=pair_half, rot_abl=rot_abl),
            **rot_abl_kw,
            **doff_kw,
            **phase_kw,
            **dcs_kw,
            **dd_kw,
            KBC_PART=_kpart,
            TS_ABL=int(_env("BWD_TS_ABL", "0")),
          )
        if hb_dBeta is not None:
            dBeta_out += hb_dBeta.sum(0).view(dBeta_out.shape)
        if hb_dGamma is not None:
            dGamma_buf += hb_dGamma.sum(0).view(dGamma_buf.shape)
    else:
      # The head block is an OPTIMISATION, never a requirement.
      # The register accumulators and the o-folded B/x tuples lengthen
      # the live ranges the Triton pipeliner stages in SHARED memory, and
      # at the biggest tile (T = N = 128, fp32 operands) EVERY config in
      # `_BWD_CFGS` then asks for ~200 KB against the A100's 164 KB.  That
      # corner falls back to the one-head-per-CTA body: the launch is
      # retried once with HBLK = 0, and the refusal is remembered per
      # instantiation so the next call does not pay the failed autotune
      # again.  Kb2's ordering is NOT revisited -- it is an atomic pass
      # either way, and with `kb1_db_store` it has not run yet, so it still
      # runs after Kb1.
      # If the HBLK = 0 body ALSO does not fit (T = N = 128, fp32 operands
      # asks ~200 KB there too), the N SPLIT slices EVERY N-dependent tile
      # ([T, N] C/B/dC/dB and the [N, GT] state) by NSPLIT while leaving the
      # [T, T] score product alone, and it is EXACT (distinct column sets,
      # no shared tile).  The retry ladder is
      #     (HBLK, NSPLIT) -> (0, NSPLIT) -> (0, 2*NSPLIT) -> ...
      # up to the 128-byte wgmma swizzle floor `_hb_ns_floor` documents.  A
      # cell that fits at the first rung never sees any of this, and the
      # refusal is remembered per instantiation.  `SSD_MIMO_BWD_KB1_NSPLIT`
      # still overrides the START of the ladder, for the A/B.
      _ns_floor_kb1 = _hb_ns_floor(mma_io)
      _ns_ladder = [kb1_nsplit]
      _ns = kb1_nsplit * 2
      while (N % _ns == 0 and N // _ns >= _ns_floor_kb1
             and (N // _ns) % 2 == 0
             and not (pair_half and N // _ns < 32)):
          _ns_ladder.append(_ns)
          _ns = _ns * 2
      # LAST RUNG.  The N split shrinks the N-DEPENDENT tiles only; at
      # T = 64, N = 128, fp32 operands the sm_120 budget (101,376 B) is
      # still ~5 % short at the widest legal split (the [T, T] score tile
      # and the staged [T, GT] dY / x loads are untouched by N).  The one
      # knob that reaches them is the PIPELINE DEPTH, so the ladder ends by
      # bypassing the autotuner and launching the JIT function itself at
      # `num_stages = 1`: a LAUNCH plan, taken only after every autotuned
      # config has been proved not to fit, and exact (staging depth changes
      # no arithmetic).
      _kb1_forced = [None] + [dict(num_warps=w, num_stages=1)
                              for w in (4, 8)]
      hb_now, dbs_now, dcs_now = kb1_hb, kb1_db_store, kb1_dc_store
      fold_now = kb2_fold
      ns_now = kb1_nsplit
      forced_now = None
      hb_key = (T, N, G_SIZE, GT, R, O, kb1_hb, kb1_nsplit, precision,
                str(mma_dt), rot_in_kernel, has_theta_h, has_rot_bias,
                pre_rot, has_beta, has_gamma, has_dw, bc_grouped, idx64,
                kb2_fold)
      _oor_cached = _KB1_HB_OOR.get(hb_key)
      if _oor_cached is not None:
          hb_now, dbs_now, dcs_now, fold_now = 0, False, False, False
          ns_now, forced_now = _oor_cached
      # ── PROTOTYPE, opt-in only: the Hopper Gluon Kb1 (GLUON_SM90) ─────
      # `SSD_MIMO_GLUON90_KB1=1` replaces THIS launch (same buffers, same
      # head block, the Kb2 fold included) on an in-envelope sm_90 cell;
      # every refusal is by name and the Triton ladder below runs
      # unchanged.  Unset, this block costs one env read.
      _g90kb1_used = False
      if (_env("GLUON90_KB1", "0").lower() in ("1", "on", "true")
              and _oor_cached is None and _portable.gluon_refusal("ssd_mimo") is None):
          from . import gluon_sm90_kb1 as _g90k
          _why_k = _g90k.refusal(
              device=device, T=T, N=N, G_SIZE=G_SIZE, GT=GT, NGT=NGT,
              R=R, O=O, mma_io=mma_io, io_dtype=x_scaled.dtype,
              hblk=hb_now, nsplit=ns_now, kb2_fold=fold_now,
              has_theta_h=has_theta_h, rot_in_kernel=rot_in_kernel,
              has_rot_bias=has_rot_bias, pair_half=bool(pair_half),
              deferred_rot=deferred_rot,
              use_slots=bool(use_slots or use_slots_c), idx64=idx64,
              has_dw=has_dw, pre_rot=pre_rot, rot_abl=_ROT_ABL_MIMO,
              Btil=Btil, Ctil=Ctil, x=x_scaled, S_all=S_all, dY=dY,
              s_bt=s_bt, s_bn=s_bn, s_ct=s_ct, s_cn=s_cn,
              has_beta=has_beta, has_gamma=has_gamma,
              phase_cumsum=phase_cumsum,
              theta=th_c_k if rot_in_kernel else None)
          from . import gluon_sm90_fwd as _g90f
          _g90k_key = ("kb1", bool(rot_in_kernel), bool(has_theta_h))
          if _why_k is None:
              _why_k = _g90f.failed_refusal(_g90k_key)
          _g90k._LAST_ROUTE.clear()
          _g90k._LAST_ROUTE.update(used=_why_k is None, why=_why_k or "")
          if _why_k is None:
            # D3 (GLUON_VERIFY): a compile / launch error falls back to the
            # Triton Kb1 ladder below with a named reason (warned once).
            try:
                _g90_dth_store = bool(rot_in_kernel and dth_store_possible
                                      and not _dth_state["zeroed"])
                if rot_in_kernel and not _g90_dth_store:
                    _dth_zero()
                _g90k.launch(
                    Ctil=Ctil, Btil=Btil, x=x_scaled, delta=delta_cumsum,
                    a_log=a_log, S_all=S_all, dY=dY, dG_all=dG_all,
                    dalpha_all=dalpha_all, beta_c=beta_c, gamma_c=gamma_c,
                    has_beta=has_beta, has_gamma=has_gamma, dw=dw_c,
                    dCtil=dCtil_acc, dBtil=dBtil_acc, dDelta=dDelta_out,
                    dBeta=dBeta_out, dGamma=dGamma_buf, H=H, T=T, N=N,
                    G_SIZE=G_SIZE, GT=GT, R=R, O=O, hpg_b=hpg_b,
                    hpg_c=hpg_c, hblk=hb_now, nc=nc, db_store=dbs_now,
                    dc_store=dcs_now, dcs_ink=dcs_ink,
                    b_strides=(s_bb, s_bc, s_bh_v, s_bt, s_bn, s_br_v),
                    c_strides=(s_cb, s_cc, s_ch_v, s_ct, s_cn, s_co_v),
                    db_strides=(s_dbb, s_dbc, s_dbh_v, s_dbt, s_dbn, s_dbr_v),
                    dc_strides=(s_dcb, s_dcc, s_dch_v, s_dct, s_dcn, s_dco_v),
                    theta=th_c_k if rot_in_kernel else None,
                    th_strides=th_st_k,
                    theta_h=thh_c if has_theta_h else None,
                    dTheta=dTheta_buf if rot_in_kernel else None,
                    dth_strides=dth_st_k,
                    dThetaH=dThetaH_out if need_dthh else None,
                    dth_store=_g90_dth_store,
                    trap_diag=bool(fuse_trap), idx64=bool(idx64))
                _g90kb1_used = True
                _kb1_dispatch_rec["gluon90"] = True
            except Exception as _e:                       # noqa: BLE001
              reraise_fatal(_e)        # OOM / CUDA error: propagate, never cached
              _g90kb1_used = False
              _g90k._LAST_ROUTE.update(
                  used=False, why=_g90f.note_failure(_g90k_key, _e))
              _kb1_dispatch_rec["gluon90"] = False
              _kb1_dispatch_rec["gluon90_why"] = _g90k._LAST_ROUTE["why"]

      def _kb1_grid(hb_, ns_):
          return (Bsz, (H + hb_ - 1) // hb_ if hb_ else H, nc * NGT * ns_)

      def _kb1_call(entry, hb_, dbs_, dcs_, fold_, ns_, dth_, extra):
          """THE Kb1 launch (or, with `.fn.warmup` as `entry`, its
          compile-only twin for the cliff probe -- same arguments, so the
          same specialisation and the same cached kernel)."""
          return entry(
        Ctil, Btil, x_scaled, delta_cumsum, a_log,
        thh_c, S_all,
        beta_c, gamma_c, rb_c,
        dY,
        dCtil_acc, dBtil_acc, dDelta_out, dThetaH_out, dBeta_out, dGamma_buf,
        dRotBias_out,
        dCrd_buf,
        H,
        s_cb, s_cc, s_ct, s_cn,
        s_bb, s_bc, s_bt, s_bn,
        *x_scaled.stride(),
        *delta_cumsum.stride(),
        *dY.stride()[:4], dY.stride(5), dY.stride(4),
        *S_all.stride(),
        s_dcb, s_dcc, s_dct, s_dcn,
        s_dbb, s_dbc, s_dbt, s_dbn,
        s_bh_v, s_ch_v,
        nc, NGT,
        T=T, N=N, G_SIZE=G_SIZE, GT=GT,
        PREC=precision,
        MMA_DT=mma_dt,
        MMA_IO=int(mma_dt is not tl.float32),
        HAS_THETA_H=has_theta_h,
        NEED_DTHH=bool(need_dthh and not deferred_rot),
        DEFERRED_ROTATION=deferred_rot,
        HAS_BETA=has_beta,
        HAS_GAMMA=has_gamma,
        HAS_ROT_BIAS=has_rot_bias,
        HPG_B=hpg_b, HPG_C=hpg_c,
        R=R, O=O,
        s_br=s_br_v, s_co=s_co_v,
        s_dbr=s_dbr_v, s_dco=s_dco_v,
        CHUNK_PARALLEL=True,
        s_slb=s_slb_v, s_slc=s_slc_v,
        SLOTS=n_slots if use_slots else 1,
        s_dbh=s_dbh_v, s_dch=s_dch_v,
        IDX64=idx64,
        Theta_ptr=th_c_k,
        s_thb=th_st_k[0], s_thc=th_st_k[1],
        s_tht=th_st_k[2], s_thh=th_st_k[3], s_thk=th_st_k[4],
        HAS_ROT=rot_in_kernel,
        BIAS_PRE_ROT=pre_rot,
        NSPLIT=ns_,
        HBLK=hb_,
        ROT_ABL=_ROT_ABL_MIMO,
        DB_STORE=dbs_,
        DC_STORE=dcs_,
        DTH_STORE=dth_,
        ROT_COMB=bool(hb_ and kb1_rot_comb),
        PAIR_HALF=bool(pair_half),
        TRAP_DIAG=bool(fuse_trap),
        OCC=_occ_bucket(_kb1_grid(hb_, ns_)[0] * _kb1_grid(hb_, ns_)[1]
                        * _kb1_grid(hb_, ns_)[2], device),
        **dth_kw,
        **doff_kw,
        **phase_kw,
        **dw_kwargs,
              **dcs_kw,
              **(kb2_fold_kw_on if fold_ else {}),
              **extra,
          )

      # ── SSD speed2 item B: never land on a collapsed Kb1 body ──────
      # (the rule and its evidence: `_KB1_CLIFF`'s comment block).  The
      # plan is made for the FIRST rung only; the OOR ladder below still
      # owns every later rung, starting from the remedy.  The remedy keeps
      # this call's head block / store / fold decisions: they are valid at
      # any split (the N split slices columns, it never shares a tile).
      _cliff_lever = _kb1_cliff_lever()
      _cliff_rec = dict(lever=_cliff_lever, remedy="", engaged=False)
      cliff_dth_off = False
      cliff_forced = None
      if (_cliff_lever != "off" and _oor_cached is None
              and not _g90kb1_used):
          _dth_first = bool(dth_store_possible and hb_now and fold_now
                            and not _dth_state["zeroed"])
          _spec = []
          _kb1_call(lambda *a_, **k_: _spec.append(_kb1_spec_key(a_, k_)),
                    hb_now, dbs_now, dcs_now, fold_now, ns_now,
                    _dth_first, {})
          _ckey = (torch.cuda.get_device_capability(device), _cliff_lever,
                   _KB1_CLIFF_REGS, _KB1_CLIFF_STACK,
                   tuple(str(c) for c in _bwd_kb1_kernel.configs), _spec[0])
          _plan = _KB1_CLIFF.get(_ckey)
          if _plan is None:
              def _facts(ns_, dth_, forced_):
                  g_ = _kb1_grid(hb_now, ns_)

                  def one(kw):
                      return _kb1_call(
                          lambda *a_, **k_: _bwd_kb1_kernel.fn.warmup(
                              *a_, grid=g_, **k_),
                          hb_now, dbs_now, dcs_now, fold_now, ns_, dth_, kw)
                  return _kb1_probe(one, [forced_] if forced_ is not None
                                    else [c.all_kwargs()
                                          for c in _bwd_kb1_kernel.configs])
              _plan = dict(remedy="", base=_facts(ns_now, _dth_first, None))
              if _kb1_all_collapsed(_plan["base"]):
                  _tried = []
                  for _lv in [_cliff_lever] + (
                          ["nsplit"] if _cliff_lever != "nsplit" else []):
                      if _lv == "nsplit":
                          for _ns in [v for v in _ns_ladder if v > ns_now][:2]:
                              _f = _facts(_ns, _dth_first, None)
                              _tried.append(dict(lever=_lv, ns=_ns, facts=_f))
                              if _kb1_clears(_f):
                                  _plan.update(remedy=_lv, ns=_ns)
                                  break
                      elif _lv == "maxnreg":
                          _fc = dict(num_warps=_BWD_PINNED[0],
                                     num_stages=_BWD_PINNED[1], maxnreg=255)
                          _f = _facts(ns_now, _dth_first, _fc)
                          _tried.append(dict(lever=_lv, facts=_f))
                          if _kb1_clears(_f):
                              _plan.update(remedy=_lv, forced=_fc)
                      elif _lv == "dth0" and _dth_first:
                          _f = _facts(ns_now, False, None)
                          _tried.append(dict(lever=_lv, facts=_f))
                          if _kb1_clears(_f):
                              _plan.update(remedy=_lv)
                      if _plan["remedy"]:
                          break
                  _plan["tried"] = _tried
                  if not _plan["remedy"]:
                      _plan["remedy"] = "none-cleared"
              _KB1_CLIFF[_ckey] = _plan
          _cliff_rec.update(_plan)
          if _plan["remedy"] == "nsplit":
              ns_now = _plan["ns"]
          elif _plan["remedy"] == "maxnreg":
              cliff_forced = forced_now = dict(_plan["forced"])
          elif _plan["remedy"] == "dth0":
              cliff_dth_off = True
          _cliff_rec["engaged"] = _plan["remedy"] in ("nsplit", "maxnreg",
                                                      "dth0")
      _kb1_dispatch_rec["cliff"] = _cliff_rec

      for _kb1_try in range(0 if _g90kb1_used
                            else 2 + len(_ns_ladder) + len(_kb1_forced)):
        grid_kb1 = _kb1_grid(hb_now, ns_now)
        _kb1_entry = (_bwd_kb1_kernel if forced_now is None
                      else _bwd_kb1_kernel.fn)
        dth_store_now = bool(dth_store_possible and hb_now and fold_now
                             and not _dth_state["zeroed"]
                             and not cliff_dth_off)
        if not dth_store_now:
            _dth_zero()
        try:
          _kb1_call(_kb1_entry[grid_kb1], hb_now, dbs_now, dcs_now,
                    fold_now, ns_now, dth_store_now, forced_now or {})
        except OutOfResources:
          if cliff_forced is not None and forced_now is cliff_forced:
              # the cliff's maxnreg launch does not fit: back to the
              # autotuned launch at this rung (its own ladder follows).
              forced_now = None
              _cliff_rec["oor_fallback"] = True
              continue
          if hb_now:
              # rung 1: drop the head block, keep the N split.
              hb_now, dbs_now, dcs_now, fold_now = 0, False, False, False
              _kb1_dispatch_rec["oor"] = True
              _kb1_dispatch_rec["reason"] = "oor"
              _KB1_HB_OOR[hb_key] = (ns_now, None)
              continue
          # rungs 2..: widen the N split (REALMODE3 lever 3).
          _nxt = [v for v in _ns_ladder if v > ns_now]
          if _nxt and forced_now is None:
              ns_now = _nxt[0]
              _KB1_HB_OOR[hb_key] = (ns_now, None)
              _kb1_dispatch_rec["oor"] = True
              _kb1_dispatch_rec["reason"] = "oor(nsplit=%d)" % ns_now
              continue
          # last rungs: bypass the autotuner at num_stages = 1
          _fi = _kb1_forced.index(forced_now) + 1
          if _fi >= len(_kb1_forced):
              raise
          forced_now = _kb1_forced[_fi]
          _KB1_HB_OOR[hb_key] = (ns_now, forced_now)
          _kb1_dispatch_rec["oor"] = True
          _kb1_dispatch_rec["reason"] = (
              "oor(nsplit=%d, forced=%r)" % (ns_now, forced_now))
          continue
        break
      _kb1_dispatch_rec["hb_final"] = hb_now
      _kb1_dispatch_rec["db_store"] = dbs_now
      _kb1_dispatch_rec["dc_store"] = dcs_now
      _kb1_dispatch_rec["kb2_fold_final"] = fold_now
      _kb1_dispatch_rec["nsplit"] = kb1_nsplit
      _kb1_dispatch_rec["nsplit_final"] = ns_now
      _kb1_dispatch_rec["forced_cfg"] = forced_now
      kb2_fold = fold_now
      if hb_key in _KB1_HB_OOR and not _kb1_dispatch_rec["oor"]:
          # cached from an earlier call this process -- same reason.
          _kb1_dispatch_rec["oor"] = True
          _kb1_dispatch_rec["reason"] = "oor(cached)"

    _kb1_dispatch_rec["kb2_hb"] = kb2_hb
    _kb1_dispatch_rec["kb2_hb_refused"] = _kb2_hb_refused
    _kb1_dispatch_rec["kb2_fold"] = bool(kb2_fold)
    _kb1_dispatch_rec["kb2_fold_refused"] = _kb2_fold_refused
    # stage 3: the fold engaged UNDER the in-kernel phase (Kb2's
    # rotating launch absorbed), and whether the rotating lever refused.
    _kb1_dispatch_rec["kb2_fold_rot"] = bool(kb2_fold and rot_in_kernel)
    _kb1_dispatch_rec["kb2_fold_rot_refused"] = _kb2_fold_rot_refused
    _note_kb1_dispatch(_kb1_dispatch_rec)

    # P1: Kb1 STORED its dBtil tiles, so Kb2's atomic pass runs after it.
    # when Kb1 ABSORBED it there is no pass to run --
    # unless the shared-memory retry fell back to HBLK = 0, which also
    # turns the fold off, and then the (now atomic) Kb1 is happy either
    # way about the order.
    if not fuse_kb2 and not _kb2_launched and not kb2_fold:
        _launch_kb2()
        del dalpha_all

    # dGamma from scratch buffer (per-head dC_rd, before Kinv mixes heads)
    dGamma_out = None
    if has_gamma and deferred_rot:
        dCrd_view = dCrd_buf.view(Bsz, H, nc, O, T, N)
        dGamma_rd = dCrd_view.sum(dim=(0, 2, 4))  # [H, O, N]
        thh_f = theta_h.float()
        pair_idx = torch.arange(N, device=device) // 2
        cos_th = torch.cos(thh_f[:, pair_idx])  # [H, N]
        sin_th = torch.sin(thh_f[:, pair_idx])  # [H, N]
        fwd_sign = torch.ones(N, device=device)
        fwd_sign[1::2] = -1.0
        inv_coeff = -fwd_sign * sin_th  # [H, N] — inverse rotation
        fwd_coeff = fwd_sign * sin_th  # [H, N] — forward rotation
        dGamma_out = torch.empty(H, N, O, device=device, dtype=f32)
        xor_idx = torch.arange(N, device=device) ^ 1
        for o_idx in range(O):
            dc = dGamma_rd[:, o_idx, :]  # [H, N]
            dc_xor = dc[:, xor_idx]
            dGamma_out[:, :, o_idx] = cos_th * dc + inv_coeff * dc_xor
        # dThetaH contribution from gamma rotation
        if need_dthh:
            for o_idx in range(O):
                dc = dGamma_rd[:, o_idx, :]  # [H, N]
                gam = gamma[:, :, o_idx].float()  # [H, N]
                gam_xor = gam[:, xor_idx]
                gamma_partner = cos_th * gam_xor - fwd_coeff * gam
                contrib = fwd_sign * dc * gamma_partner  # [H, N]
                dThetaH_out += contrib[:, 0::2] + contrib[:, 1::2]

    # Kinv: inverse rotation + dThetaH (generic branch only -- Kbc-hb produces
    # dCtil/dThetaH unrotated, and so does Kb1 under the in-kernel phase)
    if deferred_rot:
        grid_inv = (Bsz, nc, H)
        _bwd_kinv_kernel[grid_inv](
            dCrd_buf, Ctil, thh_c,
            dCtil_out, dThetaH_out,
            H, nc,
            s_cb, s_cc, s_ct, s_cn,
            s_dcb, s_dcc, s_dct, s_dcn,
            s_ch_v,
            T=T, N=N,
            NEED_DTHH=need_dthh,
            HPG_C=hpg_c,
            O=O, s_co=s_co_v, s_dco=s_dco_v,
            s_dch=s_dch_v,
            IDX64=idx64,
                OCC=_occ_bucket(grid_inv[0] * grid_inv[1] * grid_inv[2], device))
        del dCrd_buf

    # Kc: dx (chunk-parallel, reads dG from Ka output)
    if os.environ.get("SSD_MIMO_DEBUG") == "1":
        print("KC-LAUNCH", dict(T=T, N=N, GT=GT, G_SIZE=G_SIZE, R=R, O=O,
              s_br=s_br_v, s_co=s_co_v, s_bh=s_bh_v, s_ch=s_ch_v,
              HPG_B=hpg_b, HPG_C=hpg_c, fuse=fuse_kb2, slots=use_slots,
              s_dbr=s_dbr_v, s_dbh=s_dbh_v, bcg=bc_grouped,
              s_bb=s_bb, s_bc=s_bc, s_bt=s_bt, s_bn=s_bn), flush=True)
    kc_mult = nc * NGT * R if use_r_tile else nc * NGT
    grid_kc = (Bsz, H, kc_mult)
    has_dskip_kc = d_skip is not None
    if has_dskip_kc and not (O == 1 and has_dw):
        raise NotImplementedError(
            "the in-kernel D-skip adjoint rides Kc's O == 1 raw-x branch "
            f"(O={O}, dw={has_dw})")
    # dD's accumulation shape is an A/B:
    # `atomic_add` onto the [H, G, R] vector, or one private slot per
    # (b, chunk) plus a host sum of a 2 MB buffer.
    dskip_slot = (_env("DSKIP_SLOT", "0") == "1") and has_dskip_kc \
        and dskip_dd
    if has_dskip_kc and dskip_dd:
        dDsk_out = (torch.empty(Bsz * nc * H * G_SIZE * R, device=device,
                                dtype=f32) if dskip_slot
                    else torch.zeros(H * G_SIZE * R, device=device,
                                     dtype=f32))
    else:
        dDsk_out = torch.empty(1, device=device, dtype=f32)
    dg_st = list(dG_all.stride())
    has_trap_adj = trap_nd is not None
    if has_trap_adj and not has_dw_early:
        raise NotImplementedError(
            "the trapezoid adjoint rides Kc's raw-x branch (it needs the "
            f"RAW x tile and `dw`); R={R}, O={O}, dw={has_dw_early}")
    # the prologue already zeroed this buffer in the pass that built
    # `trap_nd`, so the `zeros_like` memset is one launch that is gone.
    dSdot = (trap_dsdot if trap_dsdot is not None
             else (torch.zeros_like(trap_nd) if has_trap_adj
                   else torch.empty(1, device=device, dtype=f32)))
    # `trap_nd` / `dSdot` are [B, H, L] at R = O = 1 (and ride
    # `dw`'s strides, as before) and [B, H, L, O, R] above it.
    if has_trap_adj and trap_nd.dim() == 5:
        s_nd = trap_nd.stride()
    else:
        s_nd = (ddw_st[0], ddw_st[1], ddw_st[2], 0, 0) if has_trap_adj \
            else (0, 0, 0, 0, 0)
    cb_kw = {}
    if kc_shared_cb(rot_in_kernel=rot_in_kernel, has_theta_h=has_theta_h,
                    has_rot_bias=has_rot_bias, pre_rot=pre_rot,
                    pair_half=pair_half, R=R, O=O, bc_grouped=bc_grouped,
                    use_r_tile=use_r_tile, T=T, dtype=x_scaled.dtype):
        # the SAME operands the in-kernel dot reads (MMA_DT-rounded), in
        # fp32 with exact products -- the dot's arithmetic, once per (b, c)
        _cbt = _shared_cb()
        cb_kw = dict(SHARED_CB=True, CB_ptr=_cbt,
                     s_cbb=_cbt.stride(0), s_cbc=_cbt.stride(1))
    if _env("REALR1_TRACE", "0") == "1":
        print("REALR1_TRACE", dict(
            T=T, NGT=NGT, use_hb=bool(use_hb), dg_dtype=str(dG_all.dtype),
            ka_nodalpha=bool(ka_nodalpha),
            shared_cb=bool(cb_kw), sumds_env=_env("BWD_KBC_SUMDS", "0")),
            flush=True)
    # GLUON_K2 (lever `SSD_MIMO_GLUON_K2=1`): the
    # explicit-layout twin of this launch (gluon_k2.py); refused by name
    # outside its envelope, then the Triton Kc runs unchanged.
    _k2_why = "lever off"
    if _portable.gluon_refusal("ssd_mimo") is None:       # kernel_target="portable": no Gluon (not even its import)
        from . import gluon_k2 as _gk2
    else:
        _gk2 = None
    if _gk2 is not None and _gk2.gluon_k2_wanted():
        _k2_why = _gk2.refusal(
            device=device, mma_dt_is_bf16=mma_dt is tl.bfloat16,
            dtypes_bf16=(Ctil.dtype == Btil.dtype == x_scaled.dtype
                         == torch.bfloat16),
            shared_cb=bool(cb_kw), fuse_kb2=fuse_kb2, no_dbagg=bool(use_hb),
            has_dw=has_dw, O=O, R=R, use_r_tile=use_r_tile,
            rot_in_kernel=rot_in_kernel, has_theta_h=has_theta_h,
            has_rot_bias=has_rot_bias, pre_rot=pre_rot, pair_half=pair_half,
            dd_in_kbc=dd_in_kbc, ts_abl=int(_env("BWD_TS_ABL", "0")),
            dcs_ink=bool(dcs_ink), dskip_slot=dskip_slot, use_slots=use_slots,
            T=T, N=N, GT=GT)
    _CP_LAST_ROUTE["k2"] = "gluon" if _k2_why is None else "triton: " + _k2_why
    if _k2_why is None and _gk2 is not None and _gk2.k2_pipe_wanted():
        _CP_LAST_ROUTE["k2"] = "gluon_pipe"
    if _env("CP_TRACE", "0") == "1":
        print("CP_TRACE k2=" + _CP_LAST_ROUTE["k2"], flush=True)
    if _k2_why is None:
        _xs = x_scaled.stride()
        _k2_fn = _gk2.k2_gluon
        _k2_extra = {}
        if _gk2.k2_pipe_wanted():
            _k2_fn = _gk2.k2_gluon_pipe
            _k2_extra = dict(hpg_b=hpg_b, hpg_c=hpg_c)
            _CP_LAST_ROUTE["k2"] = "gluon_pipe"
        _k2_fn(**_k2_extra,
            Ctil=Ctil, Btil=Btil, x_scaled=x_scaled,
            delta_cumsum=delta_cumsum, a_log=a_log, dG_all=dG_all, dY=dY,
            dw_c=dw_c, cb=cb_kw["CB_ptr"],
            beta=beta_c if has_beta else None,
            gamma=gamma_c if has_gamma else None,
            d_skip=d_skip if has_dskip_kc else None,
            trap_nd=trap_nd if has_trap_adj else None,
            dalpha_all=dalpha_all, dX=dX, dDW_out=dDW_out,
            dDelta_out=dDelta_out, dDsk_out=dDsk_out, dSdot=dSdot,
            Bsz=Bsz, H=H, nc=nc, NGT=NGT, T=T, N=N, GT=GT, G_SIZE=G_SIZE,
            strides=(s_cb, s_cc, s_ct, s_cn,
                     s_bb, s_bc, s_bt, s_bn,
                     _xs[0], _xs[1], _xs[2], _xs[4],
                     *delta_cumsum.stride(),
                     *dY.stride()[:4], dY.stride(5),
                     *dg_st,
                     dw_st[0], dw_st[1], dw_st[2],
                     ddw_st[0], ddw_st[1], ddw_st[2],
                     cb_kw["s_cbb"], cb_kw["s_cbc"],
                     *dalpha_all.stride(),
                     s_bh_v, s_ch_v, hpg_b, hpg_c),
            idx64=idx64, dskip_dd=bool(dskip_dd))
    k2_warps = int(_env("BWD_K2_WARPS", "0"))
    k2_stages = int(_env("BWD_K2_STAGES", "2"))
    if k2_warps not in (0, 4, 8, 16) or k2_stages not in (1, 2, 3, 4):
        raise ValueError("unsupported measured K2 launch configuration")
    k2_launch_options = {}
    k2_kernel = _bwd_k2_kernel
    if k2_warps:
        k2_kernel = _bwd_k2_kernel.fn
        k2_launch_options = dict(num_warps=k2_warps, num_stages=k2_stages)
    _k2_launch = (k2_kernel[grid_kc] if _k2_why is not None
                  else (lambda *a, **k: None))
    _k2_launch(
        Ctil, Btil, delta_cumsum, a_log,
        thh_c,
        beta_c, gamma_c, rb_c,
        dY, dsf_c, dG_all,
        dX,
        H,
        s_cb, s_cc, s_ct, s_cn,
        s_bb, s_bc, s_bt, s_bn,
        *x_scaled.stride(),
        *delta_cumsum.stride(),
        *dY.stride()[:4], dY.stride(5), dY.stride(4),
        *dsf_st,
        s_bh_v, s_ch_v,
        nc, NGT,
        T=T, N=N, G_SIZE=G_SIZE, GT=GT,
        PREC=precision,
        MMA_DT=mma_dt,
        MMA_IO=int(mma_dt is not tl.float32),
        HAS_THETA_H=has_theta_h,
        HAS_DSF=has_dsf,
        HAS_BETA=has_beta, HAS_GAMMA=has_gamma,
        HAS_ROT_BIAS=has_rot_bias,
        HPG_B=hpg_b, HPG_C=hpg_c,
        R=R, O=O, s_br=s_br_v, s_co=s_co_v,
        R_TILE=use_r_tile,
        CHUNK_PARALLEL=True,
        PAIR_HALF=bool(pair_half),
        s_dgb=dg_st[0], s_dgh=dg_st[1],
        s_dgc=dg_st[2], s_dgn=dg_st[3],
        s_dgg=dg_st[4],
        X_ptr=x_scaled,   # SPANDELTA: Kc reads x on every route
        dDW_ptr=dDW_out,
        s_dwb=ddw_st[0], s_dwh=ddw_st[1], s_dwl=ddw_st[2],
        dBtil_ptr=dBtil_acc, dDelta_ptr=dDelta_out,
        dBeta_ptr=dBeta_out, dRotBias_ptr=dRotBias_out,
        dAlpha_ptr=dalpha_all if fuse_kb2 else dummy,
        s_dbb=s_dbb, s_dbc=s_dbc, s_dbt=s_dbt, s_dbn=s_dbn,
        s_dab=dalpha_all.stride(0) if fuse_kb2 else 0,
        s_dah=dalpha_all.stride(1) if fuse_kb2 else 0,
        s_dag=dalpha_all.stride(2) if fuse_kb2 else 0,
        s_dac=dalpha_all.stride(3) if fuse_kb2 else 0,
        s_dbr=s_dbr_v,
        FUSE_KB2=fuse_kb2,
        s_slb=s_slb_v,
        SLOTS=n_slots if use_slots else 1,
        s_dbh=s_dbh_v,
        NO_DBAGG=use_hb,
        NO_DD_PAIRS=dd_in_kbc,
        TS_ABL=int(_env("BWD_TS_ABL", "0")),
        IDX64=idx64,
        Theta_ptr=th_c_k,
        s_thb=th_st_k[0], s_thc=th_st_k[1],
        s_tht=th_st_k[2], s_thh=th_st_k[3], s_thk=th_st_k[4],
        HAS_ROT=rot_in_kernel,
        BIAS_PRE_ROT=pre_rot,
        **dth_kw,
        **doff_kw,
        TrapND_ptr=trap_nd if has_trap_adj else dummy,
        dSdot_ptr=dSdot,
        s_ndb=s_nd[0], s_ndh=s_nd[1], s_ndl=s_nd[2],
        s_ndo=s_nd[3], s_ndr=s_nd[4],
        HAS_TRAP_ADJ=has_trap_adj,
        Dsk_ptr=d_skip if has_dskip_kc else dummy,
        dDsk_ptr=dDsk_out,
        HAS_DSKIP=has_dskip_kc,
        DSKIP_DD=bool(has_dskip_kc and dskip_dd),
        DSKIP_SLOT=dskip_slot,
        **cb_kw,
        **phase_kw,
        **dw_kwargs,
        **dcs_kw,
        **k2_launch_options,
        OCC=_occ_bucket(grid_kc[0] * grid_kc[1] * grid_kc[2], device),
    )
    # ── collapse the slot axis (one small reduction each) ────────────
    if use_slots:
        dBtil_out = dBtil_acc.sum(dim=0)
    if use_slots_c:
        dCtil_out = dCtil_acc.sum(dim=0)
    if fuse_kb2 or kb2_fold:
        del dalpha_all
    if chart is not None:
        from .chart_emit import chart_emit
        if has_rot or has_theta_h or has_rot_bias or has_beta or has_gamma or R != 1 or O != 1:
            raise NotImplementedError(
                "the local chart cotangent: a rotation / bias / rank reached the "
                "backward (the folded maps are not the scan's B~ / C~)")
        if _chart_sums is not None:
            pass                     # the fused emitter combines after dG_all is freed (below)
        else:
            chart.setdefault("route", "kch: the head-batched Kbc does not run (use_hb False)")
            _CP_LAST_ROUTE["chart_combine"] = "kch2 (the two-pass emitter)"
            chart["F"] = chart_emit(
                Btil=Btil, Ctil=Ctil, x_scaled=x_scaled, dw=dw if has_dw else None,
                dY=dY, S_all=S_all, dG_all=dG_all, delta_cumsum=delta_cumsum,
                a_log=a_log, L=chart["L"], mode=chart["mode"], k=chart["k"],
                Mf=chart["Mf"], Mp=chart["Mp"], bc_grouped=bc_grouped, Gb=Gb, Gc=Gc,
                mma_dt=mma_dt, precision=precision, dcs_ink=dcs_ink,
                prescale=chart.get("prescale"))
        _CP_LAST_ROUTE["chart"] = chart["route"]
        from . import chart_emit as _chart_mod
        if _chart_mod._CAPTURE is not None:
            # tests: the emitter's inputs as this backward held them (kept alive by the capture only)
            _chart_mod._CAPTURE.append(dict(
                Btil=Btil, Ctil=Ctil, x_scaled=x_scaled, dw=dw if has_dw else None, dY=dY,
                S_all=S_all, dG_all=dG_all, delta_cumsum=delta_cumsum, a_log=a_log,
                route=chart["route"], mma_dt=mma_dt, precision=precision, dcs_ink=dcs_ink,
                bc_grouped=bc_grouped, Gb=Gb, Gc=Gc, F=chart.get("F")))
    del dG_all
    if _chart_sums is not None:
        # the fused emitter's combine (Kch2 on the CHART Kbc's head sums): it needs no dG_all, so F is allocated
        # after dG_all is freed (a smaller peak than the two-pass emitter's)
        from .chart_emit import chart_combine as _chart_combine, _kr as _chart_kr2
        _gwhy = _portable.gluon_refusal("ssd_mimo")
        if _gwhy is None:
            from .chart_combine_gluon import kch2g_refusal, chart_combine_gluon
            _gwhy = kch2g_refusal(T=T, kr=_chart_kr2(chart["mode"], int(chart["k"])), maps_dtype=Btil.dtype,
                                  G=Gb if bc_grouped else 1, HS=1)
        if _gwhy is None:
            chart["combine"] = "kch2g"
            chart["F"] = chart_combine_gluon(
                Btil=Btil, Ctil=Ctil, sums=_chart_sums, L=chart["L"], mode=chart["mode"],
                k=int(chart["k"]), Mf=int(chart["Mf"]), Mp=int(chart["Mp"]), T=T)
        else:
            chart["combine"] = "kch2 split2: " + _gwhy
            chart["F"] = _chart_combine(
                Btil=Btil, Ctil=Ctil, sums=_chart_sums, L=chart["L"], mode=chart["mode"],
                k=chart["k"], Mf=chart["Mf"], Mp=chart["Mp"], bc_grouped=bc_grouped, Gb=Gb,
                Gc=Gc, mma_dt=mma_dt, precision=precision, T=T, split2=True)
        _CP_LAST_ROUTE["chart_combine"] = chart["combine"]
        del _chart_sums

    # ── Post-kernel: inverse theta rotation + dTheta ────────────────────
    dTheta_out = None
    if rot_in_kernel:
        # Kbc-hb already returned dB/dC in the UNROTATED frame and
        # accumulated dTheta itself -- nothing left to undo host-side.
        dTheta_out = dTheta_buf
    elif has_rot:
        theta_f = theta.float()  # [B, nc, T, N//2]
        # dTheta = sum of chain-rule contributions from B and C rotation
        # For B: dTheta += sum_r dB_rd[..., 2k] * B_rd[..., 2k+1] - dB_rd[..., 2k+1] * B_rd[..., 2k]
        # Btil is already the rotated version; dBtil_out is gradient w.r.t. rotated B
        if R > 1:
            Bt_rd = Btil.float().transpose(-1, -2).contiguous()
            dBt_rd = dBtil_out.float().transpose(-1, -2).contiguous()
            dTheta_out = _dtheta_from_rotation(dBt_rd, Bt_rd).sum(dim=3)
        else:
            dTheta_out = _dtheta_from_rotation(dBtil_out.float(), Btil.float())
        # C contribution
        if O > 1:
            Ct_rd = Ctil.float().transpose(-1, -2).contiguous()
            dCt_rd = dCtil_out.float().transpose(-1, -2).contiguous()
            dTheta_out = dTheta_out + _dtheta_from_rotation(dCt_rd, Ct_rd).sum(dim=3)
        else:
            dTheta_out = dTheta_out + _dtheta_from_rotation(dCtil_out.float(), Ctil.float())

        # Inverse-rotate dB and dC back to unrotated domain
        if R > 1:
            dBt = dBtil_out.float().transpose(-1, -2).contiguous()
            th_b = theta_f.unsqueeze(3).expand(-1, -1, -1, R, -1)
            dBtil_out = _rotate_tile_inv(dBt, th_b).transpose(-1, -2).contiguous()
        else:
            dBtil_out = _rotate_tile_inv(dBtil_out.float(), theta_f).contiguous()
        if O > 1:
            dCt = dCtil_out.float().transpose(-1, -2).contiguous()
            th_c = theta_f.unsqueeze(3).expand(-1, -1, -1, O, -1)
            dCtil_out = _rotate_tile_inv(dCt, th_c).transpose(-1, -2).contiguous()
        else:
            dCtil_out = _rotate_tile_inv(dCtil_out.float(), theta_f).contiguous()

    # dBeta: accumulated via kernel atomics in Kb2 + Kb1
    if not has_beta:
        dBeta_out = None

    # dGamma: kernel-accumulated on the fast path and whenever there is
    # no theta_h; dCrd_buf-based only on the generic theta_h branch.
    if has_gamma and not deferred_rot:
        dGamma_out = dGamma_buf
    elif not has_gamma:
        dGamma_out = None

    # ── the phase reparameterisation's CROSS-CHUNK term ───────────────
    # `dThOff` is complete once Kbc-hb has run; an increment in chunk c
    # moves every LATER chunk's offset, so what the increment gradient
    # still owes is the reverse exclusive prefix over chunks, broadcast
    # over the chunk's tokens.  One kernel does it in place (instead of a
    # cumsum-backward over [B, nc, (H,) K] on the host autograd graph plus
    # an ADD over the T-times-larger increment tensor).
    dThOff_ret = dThOff_buf if phase_cumsum else None
    if phase_defer:
        # dTheta_out currently holds the RAW dL/dtheta_cum; ONE pass
        # turns it into the increment gradient (and, for a caller that
        # supplied its own offsets, hands back the chunk total).
        from .m3_epilogue import phase_prefix_bwd
        if phase_internal:
            phase_prefix_bwd(dTheta_out, None, cross=True)
            dThOff_ret = None
        else:
            dThOff_ret = torch.empty_like(off_c)
            phase_prefix_bwd(dTheta_out, dThOff_ret, cross=False)
    elif phase_cumsum and phase_internal:
        from .m3_epilogue import phase_offset_bwd_add
        phase_offset_bwd_add(dThOff_buf, dTheta_out)
        dThOff_ret = None

    # ── the trapezoid's adjoint, at the END of the backward ─────
    # It runs here, and not at one of the five kernels' loads, because
    # BOTH of its inputs are produced by Kc as reductions ACROSS CTAs:
    # `A_t = dL/dw_t` (dDW) over the G tiles and `s_t = sum_g dY v`
    # (dSdot) over the same.  A kernel boundary is the only place they
    # are complete.  Running it after the post-kernel inverse rotation
    # also means dB / dC are in the UNROTATED frame in every branch,
    # which is the frame `d_t = (B + beta).(C + gamma)` is defined in.
    dLam_out = None
    if fuse_trap:
        from .m3_epilogue import trap_adjoint
        dDW_new, dLam_out = trap_adjoint(
            Btil, Ctil, beta_c if has_beta else None,
            gamma_c if has_gamma else None,
            trap_dt, trap_lam, trap_dot, trap_nxt,
            dDW_out, dSdot, dBtil_out, dCtil_out,
            dBeta_out if has_beta else None,
            dGamma_buf if has_gamma else None,
            nc, T, N, precision, bool(need_dlam),
            bc_grouped=bc_grouped, Gb=Gb, Gc=Gc,
            theta_h=thh_c if has_theta_h else None,
            dThetaH=dThetaH_out if has_theta_h else None,
            pair_half=pair_half)
        dDW_out = dDW_new

    return (dX, dCtil_out, dBtil_out, dDelta_out,
            dThetaH_out if need_dthh else None,
            dTheta_out,
            dBeta_out, dGamma_out,
            dX0_out if has_x0 else None,
            dRotBias_out if has_rot_bias else None,
            dDW_out.view(Bsz, H, nc * T) if has_dw else None,
            dSdot if trap_nd is not None else None,
            dThOff_ret,
            dLam_out,
            (dDsk_out.view(Bsz * nc, H * G_SIZE * R).sum(0) if dskip_slot
             else dDsk_out) if (has_dskip_kc and dskip_dd) else None)


# ===========================================================================
# Kc: `_bwd_k2_kernel` (dx; chunk-parallel)
# ===========================================================================



@triton.autotune(configs=_calib_grid("fused_kc", _bwd_grid),
                 prune_configs_by={"early_config_prune": _cprune("fused_kc")},
                 key=["T", "N", "G_SIZE", "GT", "PREC", "MMA_IO",
                      "HAS_THETA_H", "HAS_BETA", "HAS_GAMMA",
                      "HAS_ROT_BIAS", "HAS_ROT", "BIAS_PRE_ROT",
                      "PHASE_CUMSUM",
                      "R", "O", "HAS_DW", "FUSE_KB2",
                      # HAS_TRAP_ADJ belongs in the key: it selects a materially
                      # different epilogue (the `dv -= n_t d_t dY_t` fold and the
                      # `dSdot` atomic reduction, and at R, O > 1 both carry the
                      # saved dot's (O, R) tail), and nothing else in this list
                      # discriminates it (`HAS_DW` is True in BOTH write rules
                      # whenever `a_dyn` is on).  Without it, two calls of the
                      # same shape that differ ONLY in the write rule share one
                      # tuned launch config.
                      "HAS_TRAP_ADJ",
                      "NO_DBAGG", "IDX64", "HAS_DSKIP", "PAIR_HALF", "SHARED_CB",
                      "NO_DD_PAIRS", "TS_ABL", "OCC"],
                 # dX is a pure store (replay-safe); the rest are atomic.
                 restore_value=["dDW_ptr", "dBtil_ptr", "dDelta_ptr",
                                "dBeta_ptr", "dRotBias_ptr", "dSdot_ptr",
                                "dDsk_ptr", "dTheta_ptr", "dThOff_ptr"])
@triton.jit
def _bwd_k2_kernel(
    C_ptr, B_ptr, Delta_ptr, A_ptr, ThetaH_ptr,
    Beta_ptr, Gamma_ptr, RotBias_ptr,
    dY_ptr, dSF_ptr, dG_ptr,
    dX_ptr,
    H,
    s_cb, s_cc, s_ct, s_cn,
    s_bb, s_bc, s_bt, s_bn,
    s_xb, s_xh, s_xg, s_xr, s_xl,
    s_db, s_dh, s_dc, s_dt,
    s_yb, s_yh, s_yc, s_yt, s_yg, s_yo,
    s_fb, s_fh, s_fg, s_fn,
    s_bh, s_ch,
    NC, NGT_val,
    T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr,
    GT: tl.constexpr, PREC: tl.constexpr,
    # `tl.dot` operand dtype: fp32 = identity cast; bf16 / fp16 = IO-dtype MMA
    # with fp32 accumulate.  See `mma_operand_dtype` in fused_fwd_kernel.py.
    MMA_DT: tl.constexpr = tl.float32,
    # Autotune KEY stand-in for MMA_DT (a tl.dtype is not serialisable into
    # Triton's disk cache key): 1 iff MMA_DT is 16-bit.
    MMA_IO: tl.constexpr = 0,
    HAS_THETA_H: tl.constexpr = False, HAS_DSF: tl.constexpr = False,
    HAS_BETA: tl.constexpr = False, HAS_GAMMA: tl.constexpr = False,
    HAS_ROT_BIAS: tl.constexpr = False,
    HPG_B: tl.constexpr = 1, HPG_C: tl.constexpr = 1,
    R: tl.constexpr = 1, O: tl.constexpr = 1,
    s_br: tl.constexpr = 0, s_co: tl.constexpr = 0,
    R_TILE: tl.constexpr = False,
    CHUNK_PARALLEL: tl.constexpr = False,
    # PAIRHALF: the rotate-half pair layout; see
    # `fused_fwd_kernel.py`'s constexpr of the same name.  Kc's stage-1
    # branch is R = O = 1, chunk-parallel, NO_DBAGG (the head-batched
    # Kbc owns dB~ / dTheta).
    PAIR_HALF: tl.constexpr = False,
    s_dgb: tl.constexpr = 0, s_dgh: tl.constexpr = 0,
    s_dgc: tl.constexpr = 0, s_dgn: tl.constexpr = 0,
    s_dgg: tl.constexpr = 0,
    # ── HAS_DW: X_ptr holds RAW x, dX gets dx_raw = dx_scaled * dw, and
    # dDW accumulates ddw[t] = sum_g dx_scaled * x_raw (write-scale grad).
    X_ptr=None, DW_ptr=None, dDW_ptr=None,
    s_wb=0, s_wh=0, s_wl=0,
    s_dwb=0, s_dwh=0, s_dwl=0,
    HAS_DW: tl.constexpr = False,
    # ── FUSE_KB2 (requires HAS_DW): absorb the aggregate-path
    # backward (Kb2) — dB_agg, dBeta, dRotBias, aggregate dDelta and the
    # dalpha term — reusing the already-loaded B/dG/x tiles and the BdG
    # GEMM.  Saves the whole Kb2 launch + its dG/B re-reads.
    dBtil_ptr=None, dDelta_ptr=None, dBeta_ptr=None, dRotBias_ptr=None,
    dAlpha_ptr=None,
    s_dbb=0, s_dbc=0, s_dbt=0, s_dbn=0,
    s_dab=0, s_dah=0, s_dag=0, s_dac=0,
    s_dbr: tl.constexpr = 0,
    FUSE_KB2: tl.constexpr = False,
    # head-sliced dBtil accumulation (see Kb1 SLOTS)
    s_slb=0,
    SLOTS: tl.constexpr = 1,
    s_dbh=0,
    # NO_DBAGG: the owner-accumulate kernel (_bwd_kbc_hb_kernel) owns the
    # dB_agg product and the dBtil/dBeta writes; keep only q for the
    # dDelta/dalpha epilogue.  Only reachable on the R = 1 (non-R_TILE),
    # O = 1, rotation-free path.
    NO_DBAGG: tl.constexpr = False,
    IDX64: tl.constexpr = False,
    # ── In-kernel per-token phase: B/C arrive RAW and
    # B~ = R(theta_t)^T B / C~ = R(theta_t)^T C are formed on load.
    # Kc produces no dB/dC on this route (NO_DBAGG), so it needs no
    # dTheta -- it consumes the rotated operands only.
    Theta_ptr=None,
    s_thb=0, s_thc=0, s_tht=0, s_thh=0, s_thk=0,
    HAS_ROT: tl.constexpr = False,
    BIAS_PRE_ROT: tl.constexpr = False,
    # ── when Kc owns the aggregate dB product (FUSE_KB2 and not
    # NO_DBAGG -- the R > 1, O = 1 route) it also owns that product's
    # share of dTheta and its un-rotation.
    dTheta_ptr=None,
    s_dtb=0, s_dth=0, s_dtc=0, s_dtt=0, s_dtk=0,
    dThOff_ptr=None,
    s_dob=0, s_doh=0, s_doc=0, s_dok=0,
    # ── the two-tap trapezoid's adjoint, AT THE dY LOAD ──────────
    # Forward:  y_t = scan_t - n_t d_t v_t,  n_t = dt_{t+1}(1-l_{t+1}),
    # d_t = k_t . q_t (saved by the forward).  Two of its three adjoint
    # terms belong exactly here, where dY and the RAW x tile are already
    # in registers and dX is about to be stored:
    #     dv_t   += own_t d_t dY_t        (folded into dx_r, free)
    # TRAPFOLD: the forward's Score diagonal carries the OWN
    # tap own_t = lam_t dt_t, so the score tile here is STRICT and the
    # diagonal is this exact term (a w_t-weighted MMA diagonal minus
    # n_t d_t dY_t would be an O(n_t) cancelling pair).
    #     s_t     = sum_g dY_t,g v_t,g    (the [B, H, L] reduction the
    #                                      other two terms are built from)
    # `TrapND_ptr` carries the PRODUCT own_t d_t, so the kernel loads one
    # [T] vector rather than two.  `dSdot_ptr` is accumulated with
    # atomics because the G tiles of one (b, h, chunk) live in different
    # CTAs; it is [B, H, L_padded], i.e. 2 MB at the LM cell.
    TrapND_ptr=None, dSdot_ptr=None,
    # at R, O > 1 the correction is
    #     dx_t[g, r] += own_t sum_o dY_t[o, g] d_t[o, r]
    #     dSdot_t[o, r] = sum_g dY_t[o, g] x_t[g, r]
    # so BOTH buffers grow the saved dot's (O, R) tail and are addressed
    # through their OWN strides instead of riding `dw`'s.  At R = O = 1
    # the branch below keeps the `s_w*` addressing byte for byte and
    # these five are never read.
    s_ndb=0, s_ndh=0, s_ndl=0, s_ndo=0, s_ndr=0,
    HAS_TRAP_ADJ: tl.constexpr = False,
    # ── the D-skip's own adjoint, AT THE dX STORE ─────────────────────
    # Forward:  y_pre = scan(x * dw) + D . x_raw.  dL/dx_raw is therefore
    # dw * dx_scaled  PLUS  D * dY -- and both operands of the second
    # term are already in this kernel's registers at the store (`dY_o`
    # IS the gate epilogue's dY_pre; `x_raw` is loaded for dDW), so the
    # term costs nothing here.  dD is the same reduction over the same
    # tile, with the same atomics.
    Dsk_ptr=None, dDsk_ptr=None,
    HAS_DSKIP: tl.constexpr = False,
    # dD itself is a SEPARATE decision.  Its reduction runs over the tile's
    # T axis -- the axis Kc's MMA warps are split along -- so the cross-warp
    # reduce is costly here and ~free in the epilogue's plain elementwise
    # [T, G] tile.  Default therefore: the dX term rides Kc (free: both its
    # operands are already in registers) and dD stays in the epilogue, which
    # is equally in-kernel.
    DSKIP_DD: tl.constexpr = False,
    # dD's accumulation shape: `atomic_add` onto the
    # [H, G, R] vector (256-way contention here) or one private SLOT per
    # (b, chunk) plus a 2 MB host sum.
    DSKIP_SLOT: tl.constexpr = False,
    # ── The phase cumsum, in-kernel ─────────────────
    # Kc produces no dTheta (NO_DBAGG), so it only READS the phase: the
    # increments plus the chunk's entry offset.  Being chunk-PARALLEL is
    # exactly why the offset exists.
    ThOff_ptr=None,
    s_ob=0, s_oh=0, s_oc=0, s_ok=0,
    PHASE_CUMSUM: tl.constexpr = False,
    # PHASE_DEFER: the kernel stores the RAW dL/dtheta_cum and one
    # chunk-parallel pass (`m3_epilogue.phase_prefix_bwd`) does the in-chunk
    # reverse cumsum, the chunk total and the cross-chunk term (an in-kernel
    # cross-thread scan per head is too costly in this register-starved body).
    PHASE_DEFER: tl.constexpr = False,
    # `Delta_ptr` holds the RAW chunked delta; this kernel forms the
    # chunk-local prefix D_i itself (`_dcs_tile` / `_dcs_end`).
    DCS_INK: tl.constexpr = False,
    # ── SHARED_CB: with head-SHARED,
    # unrotated B/C the raw score C_c B_c^T is the same [T, T] tile for
    # every head (and every G tile); the host forms it ONCE per (b, c)
    # in fp32 from the SAME MMA-dtype operands (`CB_ptr` [B, nc, T, T])
    # and this kernel loads it instead of its T x N x T dot -- Mamba-2's
    # own `_bmm_chunk` structure for ngroups = 1.  Per-head B/C biases
    # enter as the exact rank-1 terms of (C + 1 g^T)(B + 1 b^T)^T.
    SHARED_CB: tl.constexpr = False,
    # DD_PAIRS: Kbc-hb owns the score pairs of dL/ddelta; Kc keeps
    # the write term (the pair (j, T-1) -> an exclusive prefix) and the
    # carry, and no longer forms dY x_w^T for them.
    NO_DD_PAIRS: tl.constexpr = False,
    CB_ptr=None, s_cbb=0, s_cbc=0,
    # ── TS_ABL (`SSD_MIMO_BWD_TS_ABL`), MEASUREMENT
    # ONLY -- every bit but 0 computes the WRONG gradient.  It prices a
    # T_s = 128 backward state chunk (states / dG stored every 2 chunks)
    # before building it: 1 = the half state READ (S skipped on odd
    # chunks, dG on even ones -- masked loads, no bytes), 2 = the
    # mid-chunk reconstruction dot that replaces it ([NS, T] x [T, GT]
    # from the neighbour chunk's x / dY tile, loaded here).
    TS_ABL: tl.constexpr = 0,
    OCC=0,           # autotune-key occupancy bucket only (`occ_bucket`)
):
    """K2: recompute v recurrence + Score + BdG, then dx.  No S_all needed.

    R_TILE: when True, grid dim-2 is NGT*R and each CTA handles one R stream.
    CHUNK_PARALLEL: when True, grid includes nc; reads dG from dG_ptr (Ka output).
    """
    # The phase acts on the N axis of the tile this CTA's head loads, so
    # every r / o slot takes the same [T, N] blend; R > 1 / O > 1 additionally
    # need the un-rotation of the aggregate dB product (below) and the O > 1
    # tile loads.
    tl.static_assert(
        not HAS_TRAP_ADJ or HAS_DW,
        "Kc: the trapezoid adjoint is the raw-x branch (it needs `dw` and "
        "the RAW x tile); served at any R, O >= 1")
    tl.static_assert(
        not BIAS_PRE_ROT or (HAS_ROT and not HAS_ROT_BIAS),
        "Kc: bias_pre_rot needs the in-kernel phase and is refused with "
        "rot_bias")
    tl.static_assert(
        not HAS_DSKIP or (O == 1 and HAS_DW),
        "Kc: the in-kernel D-skip adjoint is the O == 1 raw-x branch only")
    tl.static_assert(
        not FUSE_KB2 or (HAS_DW and (O == 1 or
                         (not HAS_ROT and not HAS_ROT_BIAS and not NO_DBAGG))),
        "Kc: MIMO Kb2 fusion requires an unrotated write map and owns dB_agg")
    b = tl.program_id(0)
    h = tl.program_id(1)
    if CHUNK_PARALLEL:
        if R_TILE:
            c = tl.program_id(2) // (NGT_val * R)
            remainder = tl.program_id(2) % (NGT_val * R)
            r_fixed = remainder // NGT_val
            gt_idx = remainder % NGT_val
        else:
            c = tl.program_id(2) // NGT_val
            gt_idx = tl.program_id(2) % NGT_val
    else:
        if R_TILE:
            r_fixed = tl.program_id(2) // NGT_val
            gt_idx = tl.program_id(2) % NGT_val
        else:
            gt_idx = tl.program_id(2)

    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)

    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    if HAS_TRAP_ADJ:
        # TRAPFOLD (fused_fwd_kernel `_TRAPFOLD_NOTE`): the score tile is
        # STRICTLY causal; the diagonal (weight own_t, not w_t) is added
        # exactly below as (own_t d_t) dY_t, so dDW is the strict-tile A.
        causal = offs_t[:, None] > offs_t[None, :]
    else:
        causal = offs_t[:, None] >= offs_t[None, :]
    neg_inf = float("-inf")
    bg = h // HPG_B
    cg = h // HPG_C

    slot_b = (h % SLOTS) * s_slb

    if PAIR_HALF:
        # ═══ PAIRHALF: the ROTATE-HALF pair layout ════════════════════
        # Two bodies: the R = O = 1 one and the general (R, O) one.  `KC_S1`
        # is the exact R = O = 1 envelope.
        KH: tl.constexpr = N // 2
        offs_h1 = tl.arange(0, KH)
        offs_h2 = KH + tl.arange(0, KH)
        # Kc contributes to dTheta only where it owns the aggregate dB
        # product; everywhere else it merely CONSUMES the operands.
        KC_DTH: tl.constexpr = HAS_ROT and FUSE_KB2 and (not NO_DBAGG)
        KC_S1: tl.constexpr = (R == 1 and O == 1 and CHUNK_PARALLEL
                               and NO_DBAGG and FUSE_KB2 and not R_TILE)
        tl.static_assert(not HAS_ROT_BIAS,
                         "PAIR_HALF does not serve rot_bias")
        tl.static_assert(N >= 32,
                         "PAIR_HALF needs N/2 >= 16 (the MMA K floor)")
        if KC_S1:
            # ═══ PAIRHALF: the ROTATE-HALF pair layout ═══════
            # Kc's share at R = O = 1 with the head-batched Kbc (NO_DBAGG):
            # it CONSUMES the rotated operands and owns dX / dDW / dSdot /
            # dDsk / dDelta, but no dB~, no dBeta and no dTheta (KC_DTH is
            # False under NO_DBAGG).  So the only layout-sensitive step is
            # building B~ and C~, and under this layout that is four FMAs on
            # two contiguous [T, N/2] tiles with no cross-lane partner.  The
            # two dots whose contraction axis is the state (B~ dG and
            # C~ B~^T) split exactly over the halves.
            #
            # ── THE LIVE-RANGE DIET ──────────────────────────────────────
            # A straightforward half body spills 146-156 registers in EVERY
            # autotune config (against 0 for the interleaved body).  Eight
            # [T, N/2] tiles hold the same elements as four [T, N] ones, so
            # the fix is not fewer values but shorter LIVE RANGES.  Four
            # changes, none of which moves a number by more than fp
            # re-association:
            #
            #   1. theta_h is folded INTO the phase:
            #      R(nu_h)^T R(Theta_t)^T = R(Theta_t + nu_h)^T -- the angles
            #      add.  Kc needs NO un-rotated C~ (that is Kbc's operand for
            #      dTheta, and KC_DTH is False here), so the read map is
            #      built in ONE rotation from cos/sin(Theta + nu) and the
            #      intermediate C~_1 / C~_2 never exist.
            #   2. The read map is built FIRST and cast at the build, so the
            #      C-side trig and the raw C halves are dead before the B
            #      side is loaded; only the two MMA-dtype operands survive.
            #   3. `dec_mask` ([T, T] fp32 = 32 registers per thread at 4
            #      warps, the single largest tile in the body) is built AT
            #      its only consumer instead of at the top -- K27's `DM`,
            #      applied here.  Same exponent, same difference-first order
            #      (contract 7.2.1), bitwise the same tile.
            #   4. `dG` and `dY` are loaded at their first use rather than
            #      hoisted, and `f` (a SEQUENTIAL-branch quantity) is not
            #      computed at all.
            tl.static_assert(R == 1 and O == 1,
                             "PAIR_HALF stage 1 is the R = O = 1 route")
            tl.static_assert(CHUNK_PARALLEL,
                             "PAIR_HALF serves the chunk-parallel Kc")
            tl.static_assert(NO_DBAGG and FUSE_KB2,
                             "PAIR_HALF serves Kc next to the head-batched "
                             "Kbc (use_hb), which owns dB~/dTheta")
            tl.static_assert(not HAS_ROT_BIAS,
                             "PAIR_HALF does not serve rot_bias")
            tl.static_assert(not R_TILE, "PAIR_HALF: R = 1 takes no R tile")

            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)

            if HAS_ROT:
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk,
                ).to(tl.float32)                                   # [T, N/2]
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + offs_h1 * s_ok,
                    ).to(tl.float32)[None, :]
            if HAS_THETA_H:
                th_h = tl.load(ThetaH_ptr + h * KH + offs_h1)      # [N/2]

            # ── the READ map, in ONE rotation by Theta + nu (diet item 1) ──
            c_base_o = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                        + offs_t[:, None] * s_ct)
            C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn).to(tl.float32)
            C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn).to(tl.float32)
            if HAS_GAMMA:
                ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O).to(tl.float32)
                ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O).to(tl.float32)
                if BIAS_PRE_ROT:
                    C1 = C1 + ga1[None, :]
                    C2 = C2 + ga2[None, :]
            if HAS_ROT:
                if HAS_THETA_H:
                    th_c = th_rot + th_h[None, :]
                else:
                    th_c = th_rot
                cc = _cos_approx(th_c)
                sc = _sin_approx(th_c)
                Cr1 = cc * C1 + sc * C2
                Cr2 = cc * C2 - sc * C1
            elif HAS_THETA_H:
                cc = _cos_approx(th_h)
                sc = _sin_approx(th_h)
                Cr1 = cc[None, :] * C1 + sc[None, :] * C2
                Cr2 = cc[None, :] * C2 - sc[None, :] * C1
            else:
                Cr1 = C1
                Cr2 = C2
            if HAS_GAMMA and not BIAS_PRE_ROT:
                # gamma is added AFTER the rotation and carries nu_h ALONE
                if HAS_THETA_H:
                    ch_h = _cos_approx(th_h)
                    sh_h = _sin_approx(th_h)
                    Cr1 = Cr1 + (ch_h * ga1 + sh_h * ga2)[None, :]
                    Cr2 = Cr2 + (ch_h * ga2 - sh_h * ga1)[None, :]
                else:
                    Cr1 = Cr1 + ga1[None, :]
                    Cr2 = Cr2 + ga2[None, :]
            # cast AT the build: the fp32 read tiles die here (diet item 2)
            Cr1_dt = Cr1.to(MMA_DT)
            Cr2_dt = Cr2.to(MMA_DT)

            # ── the WRITE map ────────────────────────────────────────────
            b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                        + offs_t[:, None] * s_bt)
            B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn).to(tl.float32)
            B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn).to(tl.float32)
            if HAS_BETA:
                be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R).to(tl.float32)
                be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R).to(tl.float32)
                if BIAS_PRE_ROT:
                    B1 = B1 + be1[None, :]
                    B2 = B2 + be2[None, :]
            if HAS_ROT:
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
                Bt1 = ph_cos * B1 + ph_sin * B2
                Bt2 = ph_cos * B2 - ph_sin * B1
            else:
                Bt1 = B1
                Bt2 = B2
            if HAS_BETA and not BIAS_PRE_ROT:
                Bt1 = Bt1 + be1[None, :]
                Bt2 = Bt2 + be2[None, :]
            # K24 CAST-BEFORE-TRANSPOSE: each half feeds one dot STRAIGHT and
            # one TRANSPOSED, so it is ONE cast, before the transpose.
            Bt1_dt = Bt1.to(MMA_DT)
            Bt2_dt = Bt2.to(MMA_DT)

            # ── the two state contractions, each split over the halves ───
            dg_base = (dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc
                       + offs_gt[None, :] * s_dgg)
            BdG_r = (pdot(Bt1_dt,
                            tl.load(dg_base + offs_h1[:, None] * s_dgn
                                    ).to(MMA_DT), input_precision=PREC)
                     + pdot(Bt2_dt,
                              tl.load(dg_base + offs_h2[:, None] * s_dgn
                                      ).to(MMA_DT), input_precision=PREC))
            dx_r = dec_agg[:, None] * BdG_r
            S0_r = (pdot(Cr1_dt, tl.trans(Bt1_dt), input_precision=PREC)
                    + pdot(Cr2_dt, tl.trans(Bt2_dt), input_precision=PREC))
            # `DM`, diet item 3: the [T, T] decay mask at its ONLY
            # consumer -- same exponent, same order, bitwise the same tile.
            Score_r = S0_r * tl.exp(
                tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                         neg_inf))
            dY_o = tl.load(
                dY_ptr + b * s_yb + h * s_yh + c * s_yc
                + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
            ).to(tl.float32)
            dx_r += pdot(tl.trans(Score_r).to(MMA_DT), dY_o.to(MMA_DT),
                           input_precision=PREC)

            x_base_r = b * s_xb + h * s_xh + c * T * s_xl
            q = tl.zeros([T], dtype=tl.float32)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)
                x_raw = tl.load(
                    X_ptr + x_base_r
                    + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                ).to(tl.float32)
                q += tl.sum(BdG_r * (x_raw * dw[:, None]), axis=1)
                tl.atomic_add(
                    dDW_ptr + b * s_dwb + h * s_dwh
                    + (c * T + offs_t) * s_dwl,
                    tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                dx_r = dx_r * dw[:, None]
                if HAS_DSKIP:
                    dsk_r = tl.load(Dsk_ptr + h * G_SIZE * R + offs_gt * R)
                    dx_r = dx_r + dsk_r[None, :] * dY_o
                    if DSKIP_DD:
                        dd_v = tl.sum(dY_o * x_raw, axis=0)
                        if DSKIP_SLOT:
                            tl.store(dDsk_ptr
                                     + ((b * NC + c) * H + h) * G_SIZE * R
                                     + offs_gt * R, dd_v)
                        else:
                            tl.atomic_add(dDsk_ptr + h * G_SIZE * R
                                          + offs_gt * R, dd_v, sem=ACC_SEM)
                if HAS_TRAP_ADJ:
                    nd = tl.load(TrapND_ptr + b * s_wb + h * s_wh
                                 + (c * T + offs_t) * s_wl)
                    dx_r = dx_r + nd[:, None] * dY_o
                    tl.atomic_add(
                        dSdot_ptr + b * s_wb + h * s_wh
                        + (c * T + offs_t) * s_wl,
                        tl.sum(dY_o * x_raw, axis=1), sem=ACC_SEM)
            tl.store(
                dX_ptr + x_base_r
                + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                dx_r)

            # SPANDELTA (`_SPANDELTA_NOTE`): dL/ddelta from the pairs that
            # span each token.  M = Score o (dY x_w^T) (x_w = the written
            # x), the write term q as the pair (j, T-1), the carry term a
            # broadcast -- no O(1) term enters any token's sum.
            if HAS_DW:
                xm = x_raw * dw[:, None]
            else:
                xm = tl.load(
                    X_ptr + x_base_r
                    + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                ).to(tl.float32)
                q += tl.sum(BdG_r * xm, axis=1)
            Msp = Score_r * pdot(dY_o.to(MMA_DT), tl.trans(xm.to(MMA_DT)),
                                   input_precision=PREC)
            Msp = tl.where((offs_t == T - 1)[:, None],
                           Msp + (dec_agg * q)[None, :], Msp)
            dalpha_c = tl.load(
                dAlpha_ptr + b * s_dab + h * s_dah
                + gt_idx * s_dag + c * s_dac)
            dD = a_val * (_span_sum(Msp, offs_t) + alpha * dalpha_c)
            tl.atomic_add(
                dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt,
                dD, sem=ACC_SEM)
        else:
            # ══ PAIRHALF: the GENERAL (R, O) half Kc ══
            # Same two identities, on the interleaved kernel's own loop
            # structure: `B~ dG` and `C~ B~^T` have the state as a
            # CONTRACTION (the sum of the two halves' dots), while the
            # aggregate `x w dG^T`, the readout carry `v` and `dG` /
            # `S_all` have it as an OUTPUT axis (two [PS, GT] tiles at
            # contiguous row ranges).  The stage-1 R = O = 1 body above
            # is kept verbatim -- it carries §2's live-range diet and is
            # the shipped route.
            if not CHUNK_PARALLEL:
                if HAS_DSF:
                    fb = (dSF_ptr + b * s_fb + h * s_fh
                          + offs_gt[None, :] * s_fg)
                    v1 = tl.load(fb + offs_h1[:, None] * s_fn
                                 ).to(tl.float32)
                    v2 = tl.load(fb + offs_h2[:, None] * s_fn
                                 ).to(tl.float32)
                else:
                    v1 = tl.zeros([KH, GT], dtype=tl.float32)
                    v2 = tl.zeros([KH, GT], dtype=tl.float32)

            R_ITERS: tl.constexpr = 1 if R_TILE else R
            nc_end_kc = 1 if CHUNK_PARALLEL else NC
            for ci in range(nc_end_kc):
                if CHUNK_PARALLEL:
                    pass
                else:
                    c = NC - 1 - ci

                dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
                D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
                seg_end = tl.max(seg_h, axis=0)
                f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
                D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
                alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
                dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
                dec_mask = tl.exp(
                    tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                             neg_inf))
                if HAS_ROT:
                    th_rot = tl.load(
                        Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                        + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk,
                    ).to(tl.float32)
                    if PHASE_CUMSUM:
                        th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                            ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                            + offs_h1 * s_ok,
                        ).to(tl.float32)[None, :]
                    ph_cos = _cos_approx(th_rot)
                    ph_sin = _sin_approx(th_rot)
                    if KC_DTH:
                        dth_acc = tl.zeros([T, KH], dtype=tl.float32)
                if HAS_THETA_H:
                    th_h = tl.load(ThetaH_ptr + h * KH + offs_h1)
                    ch_h = _cos_approx(th_h)
                    sh_h = _sin_approx(th_h)
                if HAS_DW:
                    dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                                 + (c * T + offs_t) * s_wl)

                if CHUNK_PARALLEL:
                    dgb = (dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc
                           + offs_gt[None, :] * s_dgg)
                    dG1 = tl.load(dgb + offs_h1[:, None] * s_dgn
                                  ).to(tl.float32)
                    dG2 = tl.load(dgb + offs_h2[:, None] * s_dgn
                                  ).to(tl.float32)
                else:
                    v_old1 = v1
                    v_old2 = v2
                    dG1 = v_old1
                    dG2 = v_old2
                dG1_dt = dG1.to(MMA_DT)
                dG2_dt = dG2.to(MMA_DT)

                if not CHUNK_PARALLEL:
                    dS1 = tl.zeros([KH, GT], dtype=tl.float32)
                    dS2 = tl.zeros([KH, GT], dtype=tl.float32)
                    # The PAIR_HALF Kc rank loops are RUNTIME loops (a static_range
                    # unroll roughly doubles Kc time and cold compile at the half
                    # R2O2 LLM cell).  The half FORWARD and Ka keep static loops:
                    # runtime is slower there.
                    for o in tl.range(O, loop_unroll_factor=1):
                        c_base_o = (C_ptr + b * s_cb + c * s_cc
                                    + cg * s_ch + o * s_co
                                    + offs_t[:, None] * s_ct)
                        C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn
                                     ).to(tl.float32)
                        C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn
                                     ).to(tl.float32)
                        if HAS_GAMMA:
                            ga1 = tl.load(Gamma_ptr + h * N * O
                                          + offs_h1 * O + o).to(tl.float32)
                            ga2 = tl.load(Gamma_ptr + h * N * O
                                          + offs_h2 * O + o).to(tl.float32)
                            if BIAS_PRE_ROT:
                                C1 = C1 + ga1[None, :]
                                C2 = C2 + ga2[None, :]
                        if HAS_ROT:
                            Ct1 = ph_cos * C1 + ph_sin * C2
                            Ct2 = ph_cos * C2 - ph_sin * C1
                        else:
                            Ct1 = C1
                            Ct2 = C2
                        if HAS_THETA_H:
                            Cr1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                            Cr2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                        else:
                            Cr1 = Ct1
                            Cr2 = Ct2
                        if HAS_GAMMA and not BIAS_PRE_ROT:
                            if HAS_THETA_H:
                                gr1 = ch_h * ga1 + sh_h * ga2
                                gr2 = ch_h * ga2 - sh_h * ga1
                            else:
                                gr1 = ga1
                                gr2 = ga2
                            Cr1 = Cr1 + gr1[None, :]
                            Cr2 = Cr2 + gr2[None, :]
                        dY_o = tl.load(
                            dY_ptr + b * s_yb + h * s_yh + c * s_yc
                            + o * s_yo + offs_t[:, None] * s_yt
                            + offs_gt[None, :] * s_yg,
                        ).to(tl.float32)
                        f_dY = (f[:, None] * dY_o).to(MMA_DT)
                        dS1 += pdot(tl.trans(Cr1).to(MMA_DT), f_dY,
                                      input_precision=PREC)
                        dS2 += pdot(tl.trans(Cr2).to(MMA_DT), f_dY,
                                      input_precision=PREC)
                    v1 = alpha * v_old1 + dS1
                    v2 = alpha * v_old2 + dS2

                if FUSE_KB2:
                    q = tl.zeros([T], dtype=tl.float32)
                # SPANDELTA (`_SPANDELTA_NOTE`): M = sum_{o,r} Score o (dY x_w^T)
                Msp = tl.zeros([T, T], dtype=tl.float32)
                for rr in tl.range(R_ITERS, loop_unroll_factor=1):
                    if R_TILE:
                        r = r_fixed
                    else:
                        r = rr
                    b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + r * s_br + offs_t[:, None] * s_bt)
                    B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn
                                 ).to(tl.float32)
                    B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn
                                 ).to(tl.float32)
                    if HAS_BETA:
                        be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R
                                      + r).to(tl.float32)
                        be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R
                                      + r).to(tl.float32)
                        if BIAS_PRE_ROT:
                            B1 = B1 + be1[None, :]
                            B2 = B2 + be2[None, :]
                    if HAS_ROT:
                        Bt1 = ph_cos * B1 + ph_sin * B2
                        Bt2 = ph_cos * B2 - ph_sin * B1
                    else:
                        Bt1 = B1
                        Bt2 = B2
                    if HAS_BETA and not BIAS_PRE_ROT:
                        Bf1 = Bt1 + be1[None, :]
                        Bf2 = Bt2 + be2[None, :]
                    else:
                        Bf1 = Bt1
                        Bf2 = Bt2
                    Bf1_dt = Bf1.to(MMA_DT)
                    Bf2_dt = Bf2.to(MMA_DT)
                    BdG_r = (pdot(Bf1_dt, dG1_dt, input_precision=PREC)
                             + pdot(Bf2_dt, dG2_dt, input_precision=PREC))
                    agg_r = dec_agg[:, None] * BdG_r
                    x_base_r = (b * s_xb + h * s_xh + r * s_xr
                                + c * T * s_xl)
                    x_raw = tl.load(
                        X_ptr + x_base_r + offs_gt[None, :] * s_xg
                        + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DW:
                        xm_dt = (x_raw * dw[:, None]).to(MMA_DT)
                    else:
                        xm_dt = x_raw.to(MMA_DT)
                    if HAS_TRAP_ADJ:
                        trap_corr = tl.zeros([T, GT], dtype=tl.float32)
                    dx_r = tl.zeros([T, GT], dtype=tl.float32)
                    for o in tl.range(O, loop_unroll_factor=1):
                        c_base_o = (C_ptr + b * s_cb + c * s_cc
                                    + cg * s_ch + o * s_co
                                    + offs_t[:, None] * s_ct)
                        C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn
                                     ).to(tl.float32)
                        C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn
                                     ).to(tl.float32)
                        if HAS_GAMMA:
                            ga1 = tl.load(Gamma_ptr + h * N * O
                                          + offs_h1 * O + o).to(tl.float32)
                            ga2 = tl.load(Gamma_ptr + h * N * O
                                          + offs_h2 * O + o).to(tl.float32)
                            if BIAS_PRE_ROT:
                                C1 = C1 + ga1[None, :]
                                C2 = C2 + ga2[None, :]
                        if HAS_ROT:
                            Ct1 = ph_cos * C1 + ph_sin * C2
                            Ct2 = ph_cos * C2 - ph_sin * C1
                        else:
                            Ct1 = C1
                            Ct2 = C2
                        if HAS_THETA_H:
                            Cr1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                            Cr2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                        else:
                            Cr1 = Ct1
                            Cr2 = Ct2
                        if HAS_GAMMA and not BIAS_PRE_ROT:
                            if HAS_THETA_H:
                                gr1 = ch_h * ga1 + sh_h * ga2
                                gr2 = ch_h * ga2 - sh_h * ga1
                            else:
                                gr1 = ga1
                                gr2 = ga2
                            Cr1 = Cr1 + gr1[None, :]
                            Cr2 = Cr2 + gr2[None, :]
                        S0_or = (pdot(Cr1.to(MMA_DT), tl.trans(Bf1_dt),
                                        input_precision=PREC)
                                 + pdot(Cr2.to(MMA_DT), tl.trans(Bf2_dt),
                                          input_precision=PREC))
                        Score_or = S0_or * dec_mask
                        dY_o2 = tl.load(
                            dY_ptr + b * s_yb + h * s_yh + c * s_yc
                            + o * s_yo + offs_t[:, None] * s_yt
                            + offs_gt[None, :] * s_yg,
                        ).to(tl.float32)
                        dY_o2_dt = dY_o2.to(MMA_DT)
                        dx_r += pdot(tl.trans(Score_or).to(MMA_DT),
                                       dY_o2_dt,
                                       input_precision=PREC)
                        Msp += Score_or * pdot(dY_o2_dt, tl.trans(xm_dt),
                                                 input_precision=PREC)
                        if HAS_TRAP_ADJ:
                            nd_or = tl.load(
                                TrapND_ptr + b * s_ndb + h * s_ndh
                                + (c * T + offs_t) * s_ndl
                                + o * s_ndo + r * s_ndr)
                            trap_corr += nd_or[:, None] * dY_o2
                            tl.atomic_add(
                                dSdot_ptr + b * s_ndb + h * s_ndh
                                + (c * T + offs_t) * s_ndl
                                + o * s_ndo + r * s_ndr,
                                tl.sum(dY_o2 * x_raw, axis=1), sem=ACC_SEM)
                        if HAS_DSKIP:
                            dY_dsk = dY_o2
                    dx_r += agg_r
                    if HAS_DW:
                        if FUSE_KB2:
                            x_s = x_raw * dw[:, None]
                            xw_dt = (x_s * dec_agg[:, None]).to(MMA_DT)
                            gB1 = pdot(xw_dt, tl.trans(dG1_dt),
                                         input_precision=PREC)
                            gB2 = pdot(xw_dt, tl.trans(dG2_dt),
                                         input_precision=PREC)
                            if HAS_BETA and not BIAS_PRE_ROT:
                                tl.atomic_add(dBeta_ptr + h * N * R
                                              + offs_h1 * R + r,
                                              tl.sum(gB1, axis=0), sem=ACC_SEM)
                                tl.atomic_add(dBeta_ptr + h * N * R
                                              + offs_h2 * R + r,
                                              tl.sum(gB2, axis=0), sem=ACC_SEM)
                            if KC_DTH:
                                dth_acc += (gB1 * _opv(Bt2, MMA_DT, MMA_IO, PREC) - gB2 * _opv(Bt1, MMA_DT, MMA_IO, PREC))
                            if HAS_ROT:
                                gn1 = ph_cos * gB1 - ph_sin * gB2
                                gB2 = ph_cos * gB2 + ph_sin * gB1
                                gB1 = gn1
                            if HAS_BETA and BIAS_PRE_ROT:
                                tl.atomic_add(dBeta_ptr + h * N * R
                                              + offs_h1 * R + r,
                                              tl.sum(gB1, axis=0), sem=ACC_SEM)
                                tl.atomic_add(dBeta_ptr + h * N * R
                                              + offs_h2 * R + r,
                                              tl.sum(gB2, axis=0), sem=ACC_SEM)
                            db_base_r = (dBtil_ptr + slot_b + bg * s_dbh
                                         + b * s_dbb + c * s_dbc
                                         + offs_t[:, None] * s_dbt
                                         + r * s_dbr)
                            tl.atomic_add(db_base_r
                                          + offs_h1[None, :] * s_dbn, gB1,
                                          sem=ACC_SEM)
                            tl.atomic_add(db_base_r
                                          + offs_h2[None, :] * s_dbn, gB2,
                                          sem=ACC_SEM)
                            q += tl.sum(BdG_r * x_s, axis=1)
                        tl.atomic_add(
                            dDW_ptr + b * s_dwb + h * s_dwh
                            + (c * T + offs_t) * s_dwl,
                            tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                        dx_r = dx_r * dw[:, None]
                        if HAS_DSKIP:
                            dsk_r = tl.load(Dsk_ptr + h * G_SIZE * R
                                            + offs_gt * R + r)
                            dx_r = dx_r + dsk_r[None, :] * dY_dsk
                            if DSKIP_DD:
                                dd_v = tl.sum(dY_dsk * x_raw, axis=0)
                                if DSKIP_SLOT:
                                    tl.store(
                                        dDsk_ptr
                                        + ((b * NC + c) * H + h)
                                        * G_SIZE * R + offs_gt * R + r,
                                        dd_v)
                                else:
                                    tl.atomic_add(
                                        dDsk_ptr + h * G_SIZE * R
                                        + offs_gt * R + r, dd_v, sem=ACC_SEM)
                        if HAS_TRAP_ADJ:
                            dx_r = dx_r + trap_corr
                    tl.store(
                        dX_ptr + x_base_r + offs_gt[None, :] * s_xg
                        + offs_t[:, None] * s_xl, dx_r)

                if KC_DTH:
                    dth_ph = dth_acc
                    if PHASE_CUMSUM and not PHASE_DEFER:
                        dth_off = tl.sum(dth_ph, axis=0)
                        dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        tl.atomic_add(
                            dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                            + offs_h1 * s_dok, dth_off, sem=ACC_SEM)
                    tl.atomic_add(
                        dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                        + offs_t[:, None] * s_dtt
                        + offs_h1[None, :] * s_dtk, dth_ph, sem=ACC_SEM)

                # SPANDELTA: score pairs + the write term as the pair
                # (j, T-1) + the carry term, all in spanning form
                if FUSE_KB2:
                    Msp = tl.where((offs_t == T - 1)[:, None],
                                   Msp + (dec_agg * q)[None, :], Msp)
                    dalpha_c = tl.load(
                        dAlpha_ptr + b * s_dab + h * s_dah
                        + gt_idx * s_dag + c * s_dac)
                    if R_TILE:
                        dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
                    else:
                        dalpha_gate = 1.0
                    dD = a_val * (_span_sum(Msp, offs_t)
                                  + alpha * dalpha_c * dalpha_gate)
                else:
                    dD = a_val * _span_sum(Msp, offs_t)
                tl.atomic_add(
                    dDelta_ptr + b * s_db + h * s_dh + c * s_dc
                    + offs_t * s_dt, dD, sem=ACC_SEM)

    else:
        if HAS_THETA_H:
            pair_idx_k2 = offs_n // 2
            th_full_k2 = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx_k2)
            ch_full_k2 = _cos_approx(th_full_k2)
            sh_full_k2 = _sin_approx(th_full_k2)
            fwd_sign_k2 = tl.where((offs_n % 2) == 0, 1.0, -1.0)
            fwd_coeff_k2 = fwd_sign_k2 * sh_full_k2

        # Kc contributes to dTheta only where it owns the aggregate dB
        # product; everywhere else it merely CONSUMES the rotated operands.
        KC_DTH: tl.constexpr = HAS_ROT and FUSE_KB2 and (not NO_DBAGG)
        if HAS_ROT:
            rot_sign_k2_r = tl.where((offs_n % 2) == 0, 1.0, -1.0)
            pair_idx_k2_r = offs_n // 2
        if HAS_ROT_BIAS:
            pair_idx_k2_rb = offs_n // 2
            rot_sign_k2 = tl.where((offs_n % 2) == 0, 1.0, -1.0)
            rb_full_k2 = tl.load(RotBias_ptr + h * (N // 2) + pair_idx_k2_rb)
            crb_k2 = _cos_approx(rb_full_k2)
            srb_fwd_k2 = rot_sign_k2 * _sin_approx(rb_full_k2)

        if not CHUNK_PARALLEL:
            if HAS_DSF:
                v = tl.load(
                    dSF_ptr + b * s_fb + h * s_fh
                    + offs_gt[None, :] * s_fg + offs_n[:, None] * s_fn,
                ).to(tl.float32)
            else:
                v = tl.zeros([N, GT], dtype=tl.float32)

        nc_end_kc = 1 if CHUNK_PARALLEL else NC
        for ci in range(nc_end_kc):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci

            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            dec_mask = tl.exp(
                tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                         neg_inf))
            # SPANDELTA (`_SPANDELTA_NOTE`): M = sum_{o,r} Score o (dY x_w^T)
            Msp = tl.zeros([T, T], dtype=tl.float32)

            if HAS_ROT:
                th_rot_k2 = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht
                    + tl.arange(0, N // 2)[None, :] * s_thk,
                ).to(tl.float32)                                    # [T, N//2]
                if PHASE_CUMSUM:
                    th_rot_k2 = tl.cumsum(th_rot_k2, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + tl.arange(0, N // 2) * s_ok,
                    ).to(tl.float32)[None, :]
                cs_k2 = _pair_bcast_2d(_cos_approx(th_rot_k2), T, N)
                sn_k2 = rot_sign_k2_r[None, :] * _pair_bcast_2d(
                    _sin_approx(th_rot_k2), T, N)
                if KC_DTH:
                    dth_acc = tl.zeros([T, N], dtype=tl.float32)

            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)

            if CHUNK_PARALLEL:
                if TS_ABL & 1:
                    dG = tl.load(
                        dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc
                        + offs_n[:, None] * s_dgn + offs_gt[None, :] * s_dgg,
                        mask=(offs_n[:, None] >= 0) & (c % 2 == 1),
                        other=0.0).to(tl.float32)
                else:
                    dG = tl.load(
                        dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc
                        + offs_n[:, None] * s_dgn + offs_gt[None, :] * s_dgg,
                    ).to(tl.float32)
                if TS_ABL & 2:
                    if c % 2 == 0:
                        Cn = tl.load(
                            C_ptr + b * s_cb + tl.minimum(c + 1, NC - 1) * s_cc
                            + cg * s_ch + offs_t[:, None] * s_ct
                            + offs_n[None, :] * s_cn).to(MMA_DT)
                        dY_nx = tl.load(
                            dY_ptr + b * s_yb + h * s_yh
                            + tl.minimum(c + 1, NC - 1) * s_yc
                            + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                        ).to(MMA_DT)
                        dG = dG * 0.5 + pdot(tl.trans(Cn), dY_nx,
                                               input_precision=PREC)
            else:
                v_old = v
                dG = v_old

            if O == 1:
                # O=1 fused: load C, dY once, reuse across dS_readout + R
                c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch
                Co = tl.load(
                    c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn,
                ).to(tl.float32)
                if BIAS_PRE_ROT and HAS_GAMMA:
                    Co = Co + tl.load(
                        Gamma_ptr + h * N * O + offs_n * O).to(tl.float32)[None, :]
                if HAS_ROT:
                    Co = cs_k2 * Co + sn_k2 * _pair_swap_2d(Co, T, N)
                if HAS_THETA_H:
                    if HAS_ROT:
                        Co_xor = _pair_swap_2d(Co, T, N)
                    else:
                        Co_xor = tl.load(
                            c_base_o + offs_t[:, None] * s_ct
                            + (offs_n ^ 1)[None, :] * s_cn,
                        ).to(tl.float32)
                    C_rd = ch_full_k2[None, :] * Co + fwd_coeff_k2[None, :] * Co_xor
                else:
                    C_rd = Co
                if HAS_GAMMA and not BIAS_PRE_ROT:
                    gam = tl.load(Gamma_ptr + h * N * O + offs_n * O).to(tl.float32)
                    if HAS_THETA_H:
                        gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O).to(tl.float32)
                        gam = ch_full_k2 * gam + fwd_coeff_k2 * gam_xor
                    C_rd = C_rd + gam[None, :]
                dY_o = tl.load(
                    dY_ptr + b * s_yb + h * s_yh + c * s_yc
                    + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                ).to(tl.float32)
                # K24 CAST-BEFORE-TRANSPOSE: on the SEQUENTIAL branch C_rd
                # feeds `dS_readout` TRANSPOSED and `S0_r` straight, so it is
                # the same two-orientation reuse Bc_r is -- one cast.
                C_rd_dt = C_rd.to(MMA_DT)
                if not CHUNK_PARALLEL:
                    f_dY = f[:, None] * dY_o
                    dS_readout = pdot(tl.trans(C_rd_dt), f_dY.to(MMA_DT),
                                        input_precision=PREC)
                    v = alpha * v_old + dS_readout

                if R_TILE:
                    Bc_r = tl.load(
                        B_ptr + b * s_bb + c * s_bc + bg * s_bh
                        + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                        + r_fixed * s_br,
                    ).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r_fixed
                        ).to(tl.float32)[None, :]
                    if HAS_ROT:
                        B_til = cs_k2 * Bc_r + sn_k2 * _pair_swap_2d(Bc_r, T, N)
                    else:
                        B_til = Bc_r
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            Bc_r_xor_rb = _pair_swap_2d(B_til, T, N)
                        else:
                            Bc_r_xor_rb = tl.load(
                                B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn
                                + r_fixed * s_br,
                            ).to(tl.float32)
                        B_unrot = B_til
                        B_unrot_xor = Bc_r_xor_rb
                        Bc_r = crb_k2[None, :] * B_til + srb_fwd_k2[None, :] * Bc_r_xor_rb
                    else:
                        Bc_r = B_til
                    if HAS_BETA and not BIAS_PRE_ROT:
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    # ── K24 CAST-BEFORE-TRANSPOSE ─────────────────────────────────────
                    # See the note at the O == 1 non-R_TILE site below.
                    Bc_dt = Bc_r.to(MMA_DT)
                    dG_dt = dG.to(MMA_DT)
                    BdG_r = pdot(Bc_dt, dG_dt, input_precision=PREC)
                    dx_r = dec_agg[:, None] * BdG_r
                    S0_r = pdot(C_rd_dt, tl.trans(Bc_dt),
                                  input_precision=PREC)
                    Score_r = S0_r * dec_mask
                    dx_r += pdot(tl.trans(Score_r).to(MMA_DT), dY_o.to(MMA_DT), input_precision=PREC)
                    x_base_r = b * s_xb + h * s_xh + r_fixed * s_xr + c * T * s_xl
                    x_sp = tl.load(
                        X_ptr + x_base_r
                        + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                    ).to(tl.float32)
                    if HAS_DW:
                        x_sp = x_sp * dw[:, None]
                    xm_dt = x_sp.to(MMA_DT)
                    if not NO_DD_PAIRS:
                        Msp += Score_r * pdot(dY_o.to(MMA_DT), tl.trans(xm_dt),
                                                input_precision=PREC)
                    if HAS_DW:
                        x_raw = tl.load(
                            X_ptr + x_base_r
                            + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                        ).to(tl.float32)
                        if FUSE_KB2:
                            x_s = x_raw * dw[:, None]
                            xw_r = x_s * dec_agg[:, None]
                            dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt),
                                              input_precision=PREC)
                            if HAS_ROT:
                                gB_ = dB_r_agg
                                if HAS_BETA and not BIAS_PRE_ROT:
                                    tl.atomic_add(
                                        dBeta_ptr + h * N * R + offs_n * R + r_fixed,
                                        tl.sum(gB_, axis=0), sem=ACC_SEM)
                                if HAS_ROT_BIAS:
                                    Brd_p = (crb_k2[None, :] * B_unrot_xor
                                             - srb_fwd_k2[None, :] * B_unrot)
                                    tl.atomic_add(
                                        dRotBias_ptr + h * (N // 2) + pair_idx_k2_rb,
                                        tl.sum(rot_sign_k2[None, :] * gB_ * Brd_p,
                                               axis=0), sem=ACC_SEM)
                                    gB_ = (crb_k2[None, :] * gB_
                                           - srb_fwd_k2[None, :] * _pair_swap_2d(gB_, T, N))
                                dth_acc += (rot_sign_k2_r[None, :] * gB_
                                            * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, N))
                                gB_ = cs_k2 * gB_ - sn_k2 * _pair_swap_2d(gB_, T, N)
                                if HAS_BETA and BIAS_PRE_ROT:
                                    tl.atomic_add(
                                        dBeta_ptr + h * N * R + offs_n * R + r_fixed,
                                        tl.sum(gB_, axis=0), sem=ACC_SEM)
                                tl.atomic_add(
                                    dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                                    + offs_t[:, None] * s_dbt
                                    + offs_n[None, :] * s_dbn + r_fixed * s_dbr,
                                    gB_, sem=ACC_SEM)
                            elif HAS_ROT_BIAS:
                                db_base_r = (dBtil_ptr + slot_b + bg * s_dbh
                                             + b * s_dbb + c * s_dbc
                                             + offs_t[:, None] * s_dbt
                                             + r_fixed * s_dbr)
                                tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn,
                                              crb_k2[None, :] * dB_r_agg,
                                              sem=ACC_SEM)
                                tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn,
                                              srb_fwd_k2[None, :] * dB_r_agg,
                                              sem=ACC_SEM)
                                Brd_p = (crb_k2[None, :] * B_unrot_xor
                                         - srb_fwd_k2[None, :] * B_unrot)
                                tl.atomic_add(
                                    dRotBias_ptr + h * (N // 2) + pair_idx_k2_rb,
                                    tl.sum(rot_sign_k2[None, :] * dB_r_agg * Brd_p,
                                           axis=0), sem=ACC_SEM)
                            else:
                                tl.atomic_add(
                                    dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                                    + offs_t[:, None] * s_dbt
                                    + offs_n[None, :] * s_dbn + r_fixed * s_dbr,
                                    dB_r_agg, sem=ACC_SEM)
                            if HAS_BETA and not HAS_ROT:
                                tl.atomic_add(
                                    dBeta_ptr + h * N * R + offs_n * R + r_fixed,
                                    tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                            q = tl.sum(BdG_r * x_s, axis=1)
                        tl.atomic_add(
                            dDW_ptr + b * s_dwb + h * s_dwh
                            + (c * T + offs_t) * s_dwl,
                            tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                        dx_r = dx_r * dw[:, None]
                        if HAS_DSKIP:
                            dsk_r = tl.load(Dsk_ptr + h * G_SIZE * R
                                            + offs_gt * R + r_fixed)
                            dx_r = dx_r + dsk_r[None, :] * dY_o
                            if DSKIP_DD:
                                dd_v = tl.sum(dY_o * x_raw, axis=0)
                                if DSKIP_SLOT:
                                    tl.store(
                                        dDsk_ptr
                                        + ((b * NC + c) * H + h) * G_SIZE * R
                                        + offs_gt * R + r_fixed, dd_v)
                                else:
                                    tl.atomic_add(
                                        dDsk_ptr + h * G_SIZE * R
                                        + offs_gt * R + r_fixed, dd_v,
                                        sem=ACC_SEM)
                        if HAS_TRAP_ADJ:
                            # the same two terms as the non-R_TILE
                            # branch below, on the ONE r stream this CTA
                            # owns (O = 1, so no o offset).
                            nd = tl.load(TrapND_ptr + b * s_ndb + h * s_ndh
                                         + (c * T + offs_t) * s_ndl
                                         + r_fixed * s_ndr)              # [T]
                            dx_r = dx_r + nd[:, None] * dY_o
                            tl.atomic_add(
                                dSdot_ptr + b * s_ndb + h * s_ndh
                                + (c * T + offs_t) * s_ndl
                                + r_fixed * s_ndr,
                                tl.sum(dY_o * x_raw, axis=1), sem=ACC_SEM)
                    tl.store(
                        dX_ptr + x_base_r
                        + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                        dx_r)
                else:
                    if FUSE_KB2:
                        q = tl.zeros([T], dtype=tl.float32)
                    for r in tl.range(R, loop_unroll_factor=1):
                        Bc_r = tl.load(
                            B_ptr + b * s_bb + c * s_bc + bg * s_bh
                            + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                            + r * s_br,
                        ).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_BETA:
                            Bc_r = Bc_r + tl.load(
                                Beta_ptr + h * N * R + offs_n * R + r
                            ).to(tl.float32)[None, :]
                        if HAS_ROT:
                            Bc_r = cs_k2 * Bc_r + sn_k2 * _pair_swap_2d(Bc_r, T, N)
                        B_til = Bc_r          # after R(theta_t), before R(rho)
                        if HAS_ROT_BIAS:
                            if HAS_ROT:
                                Bc_r_xor_rb = _pair_swap_2d(Bc_r, T, N)
                            else:
                                Bc_r_xor_rb = tl.load(
                                    B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                    + offs_t[:, None] * s_bt
                                    + (offs_n ^ 1)[None, :] * s_bn
                                    + r * s_br,
                                ).to(tl.float32)
                            B_unrot = Bc_r
                            B_unrot_xor = Bc_r_xor_rb
                            Bc_r = crb_k2[None, :] * Bc_r + srb_fwd_k2[None, :] * Bc_r_xor_rb
                        if SHARED_CB:
                            B_raw_cb = Bc_r
                        if HAS_BETA and not BIAS_PRE_ROT:
                            beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                            Bc_r = Bc_r + beta_r[None, :]
                        # ── K24 CAST-BEFORE-TRANSPOSE ─────────────────────────────────────
                        # Bc_r and dG each feed one dot STRAIGHT and one
                        # TRANSPOSED.  Casting the TRANSPOSE separately leaves
                        # Triton an fp32 shared staging copy that it truncates
                        # inside the `dot_op` layout, and that mis-lowers at
                        # warpsPerCTA = [8, 1] over the 64-row T tile.  ONE cast,
                        # BEFORE the transpose, stages the tile at MMA_DT and
                        # makes the transpose a `ttg.memdesc_trans`.
                        Bc_dt = Bc_r.to(MMA_DT)
                        dG_dt = dG.to(MMA_DT)
                        BdG_r = pdot(Bc_dt, dG_dt, input_precision=PREC)
                        dx_r = dec_agg[:, None] * BdG_r
                        if SHARED_CB:
                            S0_r = tl.load(
                                CB_ptr + b * s_cbb + c * s_cbc
                                + offs_t[:, None] * T + offs_t[None, :])
                            if HAS_BETA and not BIAS_PRE_ROT:
                                # (C + 1 g^T) b 1^T : C_rd already carries g
                                S0_r += tl.sum(C_rd * beta_r[None, :],
                                               axis=1)[:, None]
                            if HAS_GAMMA and not BIAS_PRE_ROT:
                                # 1 g^T B^T
                                S0_r += tl.sum(B_raw_cb * gam[None, :],
                                               axis=1)[None, :]
                        else:
                            S0_r = pdot(C_rd_dt, tl.trans(Bc_dt),
                                          input_precision=PREC)
                        Score_r = S0_r * dec_mask
                        dx_r += pdot(tl.trans(Score_r).to(MMA_DT), dY_o.to(MMA_DT), input_precision=PREC)
                        x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                        x_sp = tl.load(
                            X_ptr + x_base_r
                            + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                        ).to(tl.float32)
                        if HAS_DW:
                            x_sp = x_sp * dw[:, None]
                        xm_dt = x_sp.to(MMA_DT)
                        if not NO_DD_PAIRS:
                            Msp += Score_r * pdot(dY_o.to(MMA_DT), tl.trans(xm_dt),
                                                    input_precision=PREC)
                        if HAS_DW:
                            x_raw = tl.load(
                                X_ptr + x_base_r
                                + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                            ).to(tl.float32)
                            if FUSE_KB2:
                                x_s = x_raw * dw[:, None]
                                if not NO_DBAGG:
                                    xw_r = x_s * dec_agg[:, None]
                                    dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt),
                                                      input_precision=PREC)
                                    if HAS_ROT:
                                        gB_ = dB_r_agg
                                        if HAS_BETA and not BIAS_PRE_ROT:
                                            tl.atomic_add(
                                                dBeta_ptr + h * N * R + offs_n * R + r,
                                                tl.sum(gB_, axis=0),
                                                sem=ACC_SEM)
                                        if HAS_ROT_BIAS:
                                            Brd_p = (crb_k2[None, :] * B_unrot_xor
                                                     - srb_fwd_k2[None, :] * B_unrot)
                                            tl.atomic_add(
                                                dRotBias_ptr + h * (N // 2) + pair_idx_k2_rb,
                                                tl.sum(rot_sign_k2[None, :] * gB_ * Brd_p,
                                                       axis=0), sem=ACC_SEM)
                                            gB_ = (crb_k2[None, :] * gB_
                                                   - srb_fwd_k2[None, :]
                                                   * _pair_swap_2d(gB_, T, N))
                                        dth_acc += (rot_sign_k2_r[None, :] * gB_
                                                    * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, N))
                                        gB_ = (cs_k2 * gB_
                                               - sn_k2 * _pair_swap_2d(gB_, T, N))
                                        if HAS_BETA and BIAS_PRE_ROT:
                                            tl.atomic_add(
                                                dBeta_ptr + h * N * R + offs_n * R + r,
                                                tl.sum(gB_, axis=0),
                                                sem=ACC_SEM)
                                        tl.atomic_add(
                                            dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                                            + offs_t[:, None] * s_dbt
                                            + offs_n[None, :] * s_dbn + r * s_dbr,
                                            gB_, sem=ACC_SEM)
                                    elif HAS_ROT_BIAS:
                                        db_base_r = (dBtil_ptr + slot_b + bg * s_dbh
                                                     + b * s_dbb + c * s_dbc
                                                     + offs_t[:, None] * s_dbt
                                                     + r * s_dbr)
                                        tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn,
                                                      crb_k2[None, :] * dB_r_agg, sem=ACC_SEM)
                                        tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn,
                                                      srb_fwd_k2[None, :] * dB_r_agg, sem=ACC_SEM)
                                        Brd_p = (crb_k2[None, :] * B_unrot_xor
                                                 - srb_fwd_k2[None, :] * B_unrot)
                                        tl.atomic_add(
                                            dRotBias_ptr + h * (N // 2) + pair_idx_k2_rb,
                                            tl.sum(rot_sign_k2[None, :] * dB_r_agg * Brd_p,
                                                   axis=0), sem=ACC_SEM)
                                    else:
                                        tl.atomic_add(
                                            dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                                            + offs_t[:, None] * s_dbt
                                            + offs_n[None, :] * s_dbn + r * s_dbr,
                                            dB_r_agg, sem=ACC_SEM)
                                    if HAS_BETA and not HAS_ROT:
                                        tl.atomic_add(
                                            dBeta_ptr + h * N * R + offs_n * R + r,
                                            tl.sum(dB_r_agg, axis=0),
                                            sem=ACC_SEM)
                                q += tl.sum(BdG_r * x_s, axis=1)
                            tl.atomic_add(
                                dDW_ptr + b * s_dwb + h * s_dwh
                                + (c * T + offs_t) * s_dwl,
                                tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                            dx_r = dx_r * dw[:, None]
                            if HAS_DSKIP:
                                dsk_r = tl.load(Dsk_ptr + h * G_SIZE * R
                                                + offs_gt * R + r)
                                dx_r = dx_r + dsk_r[None, :] * dY_o
                                if DSKIP_DD:
                                    dd_v = tl.sum(dY_o * x_raw, axis=0)
                                    if DSKIP_SLOT:
                                        tl.store(
                                            dDsk_ptr
                                            + ((b * NC + c) * H + h) * G_SIZE * R
                                            + offs_gt * R + r, dd_v)
                                    else:
                                        tl.atomic_add(
                                            dDsk_ptr + h * G_SIZE * R
                                            + offs_gt * R + r, dd_v,
                                            sem=ACC_SEM)
                            if HAS_TRAP_ADJ:
                                # the correction's own v path, and the [B, H, L]
                                # reduction its other two adjoint terms are built
                                # from -- both here, where dY and the RAW x tile are
                                # already in registers and dX is about to be stored.
                                nd = tl.load(TrapND_ptr + b * s_wb + h * s_wh
                                             + (c * T + offs_t) * s_wl)      # [T]
                                dx_r = dx_r + nd[:, None] * dY_o
                                tl.atomic_add(
                                    dSdot_ptr + b * s_wb + h * s_wh
                                    + (c * T + offs_t) * s_wl,
                                    tl.sum(dY_o * x_raw, axis=1), sem=ACC_SEM)
                        tl.store(
                            dX_ptr + x_base_r
                            + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                            dx_r)

                # ── Kc's share of dTheta (the aggregate dB product) ──
                if KC_DTH:
                    dth_e, dth_o = tl.split(tl.reshape(dth_acc, (T, N // 2, 2)))
                    dth_ph = dth_e + dth_o                          # [T, N//2]
                    if PHASE_CUMSUM and not PHASE_DEFER:
                        dth_off = tl.sum(dth_ph, axis=0)
                        dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        tl.atomic_add(
                            dThOff_ptr + b * s_dob + h * s_doh + c * s_doc
                            + tl.arange(0, N // 2) * s_dok,
                            dth_off, sem=ACC_SEM)
                    tl.atomic_add(
                        dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc
                        + offs_t[:, None] * s_dtt
                        + tl.arange(0, N // 2)[None, :] * s_dtk,
                        dth_ph, sem=ACC_SEM)

                # SPANDELTA: score pairs + the write term as the pair
                # (j, T-1) + the carry term, all in spanning form
                if FUSE_KB2:
                    if not NO_DD_PAIRS:
                        Msp = tl.where((offs_t == T - 1)[:, None],
                                       Msp + (dec_agg * q)[None, :], Msp)
                    dalpha_c = tl.load(
                        dAlpha_ptr + b * s_dab + h * s_dah
                        + gt_idx * s_dag + c * s_dac)
                    if R_TILE:
                        dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
                    else:
                        dalpha_gate = 1.0
                    if NO_DD_PAIRS:
                        # the write row alone spans s iff j < s
                        dD = a_val * (_excl_prefix(dec_agg * q, offs_t)
                                      + alpha * dalpha_c * dalpha_gate)
                    else:
                        dD = a_val * (_span_sum(Msp, offs_t)
                                      + alpha * dalpha_c * dalpha_gate)
                else:
                    dD = a_val * _span_sum(Msp, offs_t)
                tl.atomic_add(
                    dDelta_ptr + b * s_db + h * s_dh + c * s_dc
                    + offs_t * s_dt, dD, sem=ACC_SEM)
            else:
                # O>1: separate dS_readout loop, then R with nested O
                #
                # ── DOT-SEEDED ACCUMULATOR ──────────────────────────────────
                # `dx_r` must NOT be seeded with `dec_agg * BdG_r` and then
                # accumulated into by the o-loop's `tl.dot`.  A [T, GT] tile
                # that is BOTH a tl.dot result and a tl.dot accumulator makes
                # Triton give the two dots different Hopper MMAv3 layouts —
                # measured at T=64, GT=16, num_warps=8: the seed dot gets
                # warpsPerCTA=[8, 1] (8 warps over 64 rows, i.e. TWICE the
                # 16-rows-per-warp MMAv3 granularity, so the second warp
                # group addresses rows 64..127 of a 64-row tile) and the
                # loop accumulator gets [4, 2], with a ttg.convert_layout
                # between them.  That pair mis-compiles: an invalid
                # __shared__ read of 2048 B (= 64 rows x k8 x f32, one whole
                # surplus A-operand tile) inside the HGMMA.64x32x8, which
                # CRASHES at N=128 and silently returns ~100%-wrong dx and
                # dDelta at N in {32, 64}.  Keeping the aggregate term in its
                # own value and folding it in after the loop leaves ONE MMA
                # layout per tile and is correct at every candidate config.
                # Reproducer + upstream evidence:
                # the MMAv3 seeded-accumulator test at 8 warps
                # (kernel test suite).
                if not CHUNK_PARALLEL:
                    dS_readout = tl.zeros([N, GT], dtype=tl.float32)
                    for o in tl.range(O, loop_unroll_factor=1):
                        c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                        Co = tl.load(
                            c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn,
                        ).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_GAMMA:
                            Co = Co + tl.load(
                                Gamma_ptr + h * N * O + offs_n * O + o
                            ).to(tl.float32)[None, :]
                        if HAS_ROT:
                            Co = cs_k2 * Co + sn_k2 * _pair_swap_2d(Co, T, N)
                        if HAS_THETA_H:
                            if HAS_ROT:
                                Co_xor = _pair_swap_2d(Co, T, N)
                            else:
                                Co_xor = tl.load(
                                    c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn,
                                ).to(tl.float32)
                            C_rd_o = ch_full_k2[None, :] * Co + fwd_coeff_k2[None, :] * Co_xor
                        else:
                            C_rd_o = Co
                        if HAS_GAMMA and not BIAS_PRE_ROT:
                            gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                            if HAS_THETA_H:
                                gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                                gam = ch_full_k2 * gam + fwd_coeff_k2 * gam_xor
                            C_rd_o = C_rd_o + gam[None, :]
                        dY_o = tl.load(
                            dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo
                            + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                        ).to(tl.float32)
                        f_dY_o = f[:, None] * dY_o
                        dS_readout += pdot(tl.trans(C_rd_o).to(MMA_DT), f_dY_o.to(MMA_DT), input_precision=PREC)
                    v = alpha * v_old + dS_readout

                if R_TILE:
                    Bc_r = tl.load(
                        B_ptr + b * s_bb + c * s_bc + bg * s_bh
                        + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                        + r_fixed * s_br,
                    ).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r_fixed
                        ).to(tl.float32)[None, :]
                    if HAS_ROT:
                        Bc_r = cs_k2 * Bc_r + sn_k2 * _pair_swap_2d(Bc_r, T, N)
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            Bc_r_xor_rb = _pair_swap_2d(Bc_r, T, N)
                        else:
                            Bc_r_xor_rb = tl.load(
                                B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn
                                + r_fixed * s_br,
                            ).to(tl.float32)
                        Bc_r = crb_k2[None, :] * Bc_r + srb_fwd_k2[None, :] * Bc_r_xor_rb
                    if HAS_BETA and not BIAS_PRE_ROT:
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    # Bc_r is used STRAIGHT here and TRANSPOSED in the
                    # o-loop's S0 dot -- ONE cast, before the transpose.
                    Bc_dt = Bc_r.to(MMA_DT)
                    dG_dt = dG.to(MMA_DT)
                    BdG_r = pdot(Bc_dt, dG_dt, input_precision=PREC)
                    # ── DOT-SEEDED ACCUMULATOR: see the note above the O > 1
                    # branch.  The aggregate term is held OUT of the score
                    # accumulator and folded in after the o-loop.
                    agg_r = dec_agg[:, None] * BdG_r
                    x_base_r = b * s_xb + h * s_xh + r_fixed * s_xr + c * T * s_xl
                    x_sp = tl.load(
                        X_ptr + x_base_r
                        + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                    ).to(tl.float32)
                    if HAS_DW:
                        x_sp = x_sp * dw[:, None]
                    xm_dt = x_sp.to(MMA_DT)
                    if HAS_TRAP_ADJ:
                        # hoisted out of the o loop: `dSdot[o, r]` contracts
                        # the SAME raw x tile against every o.
                        x_raw = tl.load(
                            X_ptr + x_base_r
                            + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                        ).to(tl.float32)
                        trap_corr = tl.zeros([T, GT], dtype=tl.float32)
                    dx_r = tl.zeros([T, GT], dtype=tl.float32)
                    for o in tl.range(O, loop_unroll_factor=1):
                        c_base_o2 = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                        Co2 = tl.load(
                            c_base_o2 + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn,
                        ).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_GAMMA:
                            Co2 = Co2 + tl.load(
                                Gamma_ptr + h * N * O + offs_n * O + o
                            ).to(tl.float32)[None, :]
                        if HAS_ROT:
                            Co2 = cs_k2 * Co2 + sn_k2 * _pair_swap_2d(Co2, T, N)
                        if HAS_THETA_H:
                            if HAS_ROT:
                                Co2_xor = _pair_swap_2d(Co2, T, N)
                            else:
                                Co2_xor = tl.load(
                                    c_base_o2 + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn,
                                ).to(tl.float32)
                            C_rd_o2 = ch_full_k2[None, :] * Co2 + fwd_coeff_k2[None, :] * Co2_xor
                        else:
                            C_rd_o2 = Co2
                        if HAS_GAMMA and not BIAS_PRE_ROT:
                            gam2 = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                            if HAS_THETA_H:
                                gam2_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                                gam2 = ch_full_k2 * gam2 + fwd_coeff_k2 * gam2_xor
                            C_rd_o2 = C_rd_o2 + gam2[None, :]
                        S0_or = pdot(C_rd_o2.to(MMA_DT), tl.trans(Bc_dt),
                                       input_precision=PREC)
                        Score_or = S0_or * dec_mask
                        dY_o2 = tl.load(
                            dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo
                            + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                        ).to(tl.float32)
                        Msp += Score_or * pdot(dY_o2.to(MMA_DT), tl.trans(xm_dt),
                                                 input_precision=PREC)
                        dx_r += pdot(tl.trans(Score_or).to(MMA_DT), dY_o2.to(MMA_DT), input_precision=PREC)
                        if HAS_TRAP_ADJ:
                            # the two activation-sized adjoint terms,
                            # here, where dY and the RAW x tile are both in
                            # registers.  `trap_corr[t, g] = sum_o d_t[o, r]
                            # n_t dY_t[o, g]` is folded into dX after the dw
                            # scaling (the correction multiplies the RAW x,
                            # not the scaled one); `dSdot[o, r]` is the
                            # [B, H, L, O, R] reduction the adjoint kernel
                            # builds dd / dB / dC / dn from.
                            nd_or = tl.load(
                                TrapND_ptr + b * s_ndb + h * s_ndh
                                + (c * T + offs_t) * s_ndl
                                + o * s_ndo + r_fixed * s_ndr)
                            trap_corr += nd_or[:, None] * dY_o2
                            tl.atomic_add(
                                dSdot_ptr + b * s_ndb + h * s_ndh
                                + (c * T + offs_t) * s_ndl
                                + o * s_ndo + r_fixed * s_ndr,
                                tl.sum(dY_o2 * x_raw, axis=1), sem=ACC_SEM)
                    dx_r += agg_r
                    if HAS_DW:
                        if not HAS_TRAP_ADJ:
                            x_raw = tl.load(
                                X_ptr + x_base_r
                                + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                            ).to(tl.float32)
                        if FUSE_KB2:
                            x_s = x_raw * dw[:, None]
                            xw_r = x_s * dec_agg[:, None]
                            dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt),
                                              input_precision=PREC)
                            tl.atomic_add(
                                dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                                + offs_t[:, None] * s_dbt
                                + offs_n[None, :] * s_dbn + r_fixed * s_dbr,
                                dB_r_agg, sem=ACC_SEM)
                            if HAS_BETA:
                                tl.atomic_add(
                                    dBeta_ptr + h * N * R + offs_n * R + r_fixed,
                                    tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                            q = tl.sum(BdG_r * x_s, axis=1)
                        tl.atomic_add(
                            dDW_ptr + b * s_dwb + h * s_dwh
                            + (c * T + offs_t) * s_dwl,
                            tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                        dx_r = dx_r * dw[:, None]
                        if HAS_TRAP_ADJ:
                            dx_r = dx_r + trap_corr
                    tl.store(
                        dX_ptr + x_base_r
                        + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                        dx_r)
                else:
                    if FUSE_KB2:
                        q = tl.zeros([T], dtype=tl.float32)
                    for r in tl.range(R, loop_unroll_factor=1):
                        Bc_r = tl.load(
                            B_ptr + b * s_bb + c * s_bc + bg * s_bh
                            + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                            + r * s_br,
                        ).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_BETA:
                            Bc_r = Bc_r + tl.load(
                                Beta_ptr + h * N * R + offs_n * R + r
                            ).to(tl.float32)[None, :]
                        if HAS_ROT:
                            Bc_r = cs_k2 * Bc_r + sn_k2 * _pair_swap_2d(Bc_r, T, N)
                        if HAS_ROT_BIAS:
                            if HAS_ROT:
                                Bc_r_xor_rb = _pair_swap_2d(Bc_r, T, N)
                            else:
                                Bc_r_xor_rb = tl.load(
                                    B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                    + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn
                                    + r * s_br,
                                ).to(tl.float32)
                            Bc_r = crb_k2[None, :] * Bc_r + srb_fwd_k2[None, :] * Bc_r_xor_rb
                        if HAS_BETA and not BIAS_PRE_ROT:
                            beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                            Bc_r = Bc_r + beta_r[None, :]
                        # Bc_r is used STRAIGHT here and TRANSPOSED in the
                        # o-loop's S0 dot -- ONE cast, before the transpose.
                        Bc_dt = Bc_r.to(MMA_DT)
                        dG_dt = dG.to(MMA_DT)
                        BdG_r = pdot(Bc_dt, dG_dt, input_precision=PREC)
                        # ── DOT-SEEDED ACCUMULATOR (see the note above) ──
                        agg_r = dec_agg[:, None] * BdG_r
                        x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                        x_sp = tl.load(
                            X_ptr + x_base_r
                            + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                        ).to(tl.float32)
                        if HAS_DW:
                            x_sp = x_sp * dw[:, None]
                        xm_dt = x_sp.to(MMA_DT)
                        if HAS_TRAP_ADJ:
                            x_raw = tl.load(
                                X_ptr + x_base_r
                                + offs_gt[None, :] * s_xg
                                + offs_t[:, None] * s_xl,
                            ).to(tl.float32)
                            trap_corr = tl.zeros([T, GT], dtype=tl.float32)
                        dx_r = tl.zeros([T, GT], dtype=tl.float32)
                        for o in tl.range(O, loop_unroll_factor=1):
                            c_base_o2 = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                            Co2 = tl.load(
                                c_base_o2 + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn,
                            ).to(tl.float32)
                            if BIAS_PRE_ROT and HAS_GAMMA:
                                Co2 = Co2 + tl.load(
                                    Gamma_ptr + h * N * O + offs_n * O + o
                                ).to(tl.float32)[None, :]
                            if HAS_ROT:
                                Co2 = cs_k2 * Co2 + sn_k2 * _pair_swap_2d(Co2, T, N)
                            if HAS_THETA_H:
                                if HAS_ROT:
                                    Co2_xor = _pair_swap_2d(Co2, T, N)
                                else:
                                    Co2_xor = tl.load(
                                        c_base_o2 + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn,
                                    ).to(tl.float32)
                                C_rd_o2 = ch_full_k2[None, :] * Co2 + fwd_coeff_k2[None, :] * Co2_xor
                            else:
                                C_rd_o2 = Co2
                            if HAS_GAMMA and not BIAS_PRE_ROT:
                                gam2 = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                                if HAS_THETA_H:
                                    gam2_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                                    gam2 = ch_full_k2 * gam2 + fwd_coeff_k2 * gam2_xor
                                C_rd_o2 = C_rd_o2 + gam2[None, :]
                            S0_or = pdot(C_rd_o2.to(MMA_DT), tl.trans(Bc_dt),
                                           input_precision=PREC)
                            Score_or = S0_or * dec_mask
                            dY_o2 = tl.load(
                                dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo
                                + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg,
                            ).to(tl.float32)
                            Msp += Score_or * pdot(dY_o2.to(MMA_DT), tl.trans(xm_dt),
                                                     input_precision=PREC)
                            dx_r += pdot(tl.trans(Score_or).to(MMA_DT), dY_o2.to(MMA_DT), input_precision=PREC)
                            if HAS_TRAP_ADJ:
                                # the two activation-sized adjoint terms,
                                # here, where dY and the RAW x tile are both in
                                # registers.  `trap_corr[t, g] = sum_o d_t[o, r]
                                # n_t dY_t[o, g]` is folded into dX after the dw
                                # scaling (the correction multiplies the RAW x,
                                # not the scaled one); `dSdot[o, r]` is the
                                # [B, H, L, O, R] reduction the adjoint kernel
                                # builds dd / dB / dC / dn from.
                                nd_or = tl.load(
                                    TrapND_ptr + b * s_ndb + h * s_ndh
                                    + (c * T + offs_t) * s_ndl
                                    + o * s_ndo + r * s_ndr)
                                trap_corr += nd_or[:, None] * dY_o2
                                tl.atomic_add(
                                    dSdot_ptr + b * s_ndb + h * s_ndh
                                    + (c * T + offs_t) * s_ndl
                                    + o * s_ndo + r * s_ndr,
                                    tl.sum(dY_o2 * x_raw, axis=1), sem=ACC_SEM)
                        dx_r += agg_r
                        if HAS_DW:
                            if not HAS_TRAP_ADJ:
                                x_raw = tl.load(
                                    X_ptr + x_base_r
                                    + offs_gt[None, :] * s_xg
                                    + offs_t[:, None] * s_xl,
                                ).to(tl.float32)
                            if FUSE_KB2:
                                x_s = x_raw * dw[:, None]
                                xw_r = x_s * dec_agg[:, None]
                                dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt),
                                                  input_precision=PREC)
                                tl.atomic_add(
                                    dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc
                                    + offs_t[:, None] * s_dbt
                                    + offs_n[None, :] * s_dbn + r * s_dbr,
                                    dB_r_agg, sem=ACC_SEM)
                                if HAS_BETA:
                                    tl.atomic_add(
                                        dBeta_ptr + h * N * R + offs_n * R + r,
                                        tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                                q += tl.sum(BdG_r * x_s, axis=1)
                            tl.atomic_add(
                                dDW_ptr + b * s_dwb + h * s_dwh
                                + (c * T + offs_t) * s_dwl,
                                tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                            dx_r = dx_r * dw[:, None]
                            if HAS_TRAP_ADJ:
                                dx_r = dx_r + trap_corr
                        tl.store(
                            dX_ptr + x_base_r
                            + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl,
                            dx_r)

                # SPANDELTA: score pairs + the write term as the pair
                # (j, T-1) + the carry term, all in spanning form
                if FUSE_KB2:
                    Msp = tl.where((offs_t == T - 1)[:, None],
                                   Msp + (dec_agg * q)[None, :], Msp)
                    dalpha_c = tl.load(
                        dAlpha_ptr + b * s_dab + h * s_dah
                        + gt_idx * s_dag + c * s_dac)
                    if R_TILE:
                        dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
                    else:
                        dalpha_gate = 1.0
                    dD = a_val * (_span_sum(Msp, offs_t)
                                  + alpha * dalpha_c * dalpha_gate)
                else:
                    dD = a_val * _span_sum(Msp, offs_t)
                tl.atomic_add(
                    dDelta_ptr + b * s_db + h * s_dh + c * s_dc
                    + offs_t * s_dt, dD, sem=ACC_SEM)


# ---------------------------------------------------------------------------
# The delta adjoint, in ONE Triton pass
# ---------------------------------------------------------------------------
# With `DCS_INK` the backward owes
#
#   d_delta = _rev_cumsum(dDelta)          # flip + aten::cumsum + flip
#   da_log  = (D * dDelta).sum(0, 2, 3)    # = (delta * d_delta).sum(0, 2, 3)
#
# over a [B, H, nc, T] fp32 tensor whose innermost extent is the CHUNK, which
# is exactly the geometry ATen's scan is worst at.  The chunk-local reverse
# prefix is a per-ROW operation on a [T] tile, so one Triton kernel reads
# dDelta and delta ONCE, writes d_delta ONCE, and reduces the per-head
# `sum(delta * d_delta)` in registers with one atomic per (row block, head).
#
# NUMERICS: `tl.cumsum` is a parallel scan where ATen's flip-scan was
# sequential -- same value to a few ulp.  It runs only under DCS_INK.


@triton.jit
def _delta_adjoint_kernel(
    dD_ptr,          # [B, H, nc, T] fp32  IN: dL/dD_i   OUT: dL/ddelta_i
    Delta_ptr,       # [B, H, nc, T] fp32  the raw chunked delta
    dALog_ptr,       # [H] fp32            += sum_i D_i dL/dD_i
    n_rows,          # B * nc   (rows per head)
    s_db, s_dh, s_dc, s_dt,
    H: tl.constexpr,
    NC: tl.constexpr,
    T: tl.constexpr,
    ROWS: tl.constexpr,
    NEED_DALOG: tl.constexpr,
):
    h = tl.program_id(0)
    row0 = tl.program_id(1) * ROWS
    rows = row0 + tl.arange(0, ROWS)
    m = rows < n_rows
    bb = rows // NC
    cc = rows % NC
    offs_t = tl.arange(0, T)
    # the REVERSED token index: loading/storing through it turns the
    # forward `tl.cumsum` into the reverse-inclusive prefix sum without a
    # materialised flip.
    rev = T - 1 - offs_t
    base = (dD_ptr + bb[:, None] * s_db + h * s_dh + cc[:, None] * s_dc)
    g = tl.load(base + rev[None, :] * s_dt, mask=m[:, None], other=0.0)
    g = tl.cumsum(g, axis=1)
    tl.store(base + rev[None, :] * s_dt, g, mask=m[:, None])
    if NEED_DALOG:
        dbase = (Delta_ptr + bb[:, None] * s_db + h * s_dh
                 + cc[:, None] * s_dc)
        d = tl.load(dbase + rev[None, :] * s_dt, mask=m[:, None], other=0.0)
        tl.atomic_add(dALog_ptr + h, tl.sum(d * g), sem=ACC_SEM)


def delta_adjoint(dDelta, delta_raw, need_dalog=True):
    """dL/dD_i -> dL/ddelta_i (in place), and da_log, in one pass.

    `delta_raw` is the RAW chunked delta (the `DCS_INK` operand).  Returns
    `(dDelta, da_log)` with `dDelta` the SAME tensor, overwritten.
    """
    Bsz, H, nc, T = dDelta.shape
    if not dDelta.is_contiguous():
        raise ValueError("delta_adjoint needs a contiguous dDelta")
    if delta_raw.stride() != dDelta.stride():
        delta_raw = delta_raw.contiguous()
    n_rows = Bsz * nc
    da_log = (torch.zeros(H, device=dDelta.device, dtype=torch.float32)
              if need_dalog else None)
    rows = max(1, min(64, triton.next_power_of_2(n_rows)))
    while rows * T > 4096 and rows > 1:
        rows //= 2
    grid = (H, triton.cdiv(n_rows, rows))
    _delta_adjoint_kernel[grid](
        dDelta, delta_raw, da_log if need_dalog else dDelta,
        n_rows,
        *dDelta.stride(),
        H=H, NC=nc, T=T, ROWS=rows,
        NEED_DALOG=bool(need_dalog),
        num_warps=4,
    )
    return dDelta, da_log
