"""SISO FAST PATH (R = O = 1) backward -- rotation-free scan + ONE adjoint.

The backward half of `siso_fwd_kernel.py`, and the
whole point of storing the rotated maps in the forward.

  * `_siso_bwd_scan_kernel` is the `dqkv` analogue: a REVERSE chunk loop,
    one CTA per (batch, head), carrying `dS[N, G]` fp32 in registers.  It
    reads the ALREADY-ROTATED per-head maps `Bs` / `Cs` and NEVER touches
    a transcendental, a pair shuffle or a phase tile.  It writes `dB~` / `dC~`
    IN PLACE over `Bs` / `Cs`, which avoids a second set of map-sized
    buffers.
  * `_siso_rot_adjoint_kernel` is the `rotary_bias_angles` analogue: ONE
    chunk-parallel POINTWISE kernel, grid `(nc, B)`, no `tl.dot` on a
    real tile and no state, that maps `dB~` / `dC~` back to the raw frame,
    REDUCES over the head group so `dB` / `dC` are written once, and
    produces `dTheta`, `dThetaH`, `dRotBias`, `dBeta`, `dGamma` and the
    trapezoid's two scalar adjoints.

WHY THE ADJOINT CAN RUN AFTER THE MAPS ARE OVERWRITTEN.  The pair
rotation `M(θ)` has `dM/dθ = M J` with `J = [[0, 1], [-1, 0]]`, so

    dθ = g^T (dM/dθ) u = (M^T g)^T J u = dB_e u_o - dB_o u_e

with `u` the tile ENTERING the rotation and `dB = M^T g` the gradient
LEAVING it -- i.e. the angle adjoint can be taken entirely in the RAW
frame, from the head-SHARED `B + β` (8 MB, L2-resident) and the
un-rotated gradient.  Nothing in this kernel needs the rotated map that
the scan just overwrote.  That identity is `M(θ)^T J M(θ) = J`, the same
one the SSD-MIMO contract uses for `dθ_h` / `dρ`.

STABILITY BY CONSTRUCTION is inherited: every exponent below is the
unchanged difference-first, causal-masked `a * (D_i - D_j)` /
`a * (D_end - D)` form, no clamp is added, and the phase is a rotation
that never reaches an exponent.
"""

import os

import torch
import triton
from .fused_fwd_kernel import occ_bucket, len_bucket
import triton.language as tl
from .autotune_policy import siso_autotune_tier
from .common import pdot  # FP32DOT operand rounding
from .. import portable as _portable
from .common import _opv   # THETAFREQ: the value an MMA operand carried

from .fused_fwd_kernel import (_dcs_seg, _same_seg,
                               _env, _mmav3_eight_warp_ok, _needs_i64,
                               _pair_bcast_2d, _pair_swap_1d,
                               _pair_swap_2d)
from .common import ACC_SEM, _cos_approx, _sin_approx
from .common import _span_sum, _span_sum_T, _excl_prefix   # SPANDELTA
from .siso_fwd_kernel import _to_opd
from .rotation_active import prefix_half, join_halves, rotate_block
from . import levers as _lv


def _keep_graph():
    """True while the running backward belongs to a graph task that keeps its graph
    (`retain_graph=True` / `create_graph=True`): saved tensors must not be clobbered."""
    try:
        return bool(torch._C._autograd._get_current_graph_task_keep_graph())
    except Exception:
        return True            # unknown engine: the safe (non-aliasing) answer


# ── SPANMMA: the SPANDELTA intra term on the tensor cores ──
# `_span_sum(P)` = E_s = sum_{i >= s} sum_{j < s} P[i, j] -- the score
# pairs that SPAN s (`_SPANDELTA_NOTE`).  The tile form takes it as a
# reverse cumsum along i (a [T, T] scan whose axis crosses warps in the
# MMA layout: shared-memory round trips on every step) and a masked row
# sum.  The SAME sum is  E_s = sum_{j < s} W[s, j]  with
#     W = Lge @ P,   Lge[s, i] = [i >= s]          (0/1, exact in bf16),
# i.e. one [T, T] x [T, T] MMA plus a row sum along the accumulator's
# own row (intra-warp).  Every summand is still one spanning pair
# P[i, j] times 1: NO total is formed and nothing is subtracted, so the
# cancellation the SPANDELTA fix removed (a total minus a prefix, both
# O(1), whose difference is O(e^-c) behind a reset token) cannot come
# back -- each term keeps its own decay factor and its RELATIVE rounding.
# P is fp32; the MMA reads it as an unevaluated sum of NSPLIT bf16 parts
#     P = p1 + p2 (+ p3),  p1 = bf16(P),  p2 = bf16(P - p1), ...
# where each residual `P - p1` is the EXACT rounding error of p1 (a
# Sterbenz-exact fp32 difference of a number and its own rounding, not a
# difference of two computed totals).  NSPLIT = 2 carries ~16 mantissa
# bits of every summand, 3 carries ~24 (fp32); the small parts are
# accumulated first.  Accumulation is fp32 in the MMA.
@triton.jit
def _excl_combine(a1, l1, a2, l2):
    """Exclusive-scan monoid on (sum of all but the last, last): only
    ADDITIONS of original terms, never total - prefix (`_SPANDELTA_NOTE`)."""
    return a1 + l1 + a2, l2


@triton.jit
def _excl_prefix_scan(v):
    """sum_{j < s} v[j]: `_excl_prefix` as ONE 1-D scan instead of a
    masked [T, T] broadcast-and-reduce.  Same summands, no subtraction."""
    a, _ = tl.associative_scan((tl.zeros_like(v), v), 0, _excl_combine)
    return a


@triton.jit
def _span_sum_ax1(M, offs_t):
    """`_span_sum` with the scan on the AXIS A WARP OWNS.

    S_s = sum_{i >= s} sum_{j < s} M[i, j]: Q[i, s] = sum_{j < s} M[i, j]
    is an exclusive scan ALONG A ROW (axis 1 -- quad shuffles + registers
    in the MMA accumulator layout), then ONE masked column sum (axis 0,
    the reduction Mamba-3's dADT also pays).  The tile form scans along
    axis 0, i.e. across warps through shared memory.  Every term is still
    a pair that spans s; nothing is subtracted (SPANAX1)."""
    q, _ = tl.associative_scan((tl.zeros_like(M), M), 1, _excl_combine)
    return tl.sum(tl.where(offs_t[:, None] >= offs_t[None, :], q, 0.0),
                  axis=0)


@triton.jit
def _span_sum_mma(P, Lge, offs_t, NSPLIT: tl.constexpr):
    p1 = P.to(tl.bfloat16)
    r1 = P - p1.to(tl.float32)
    p2 = r1.to(tl.bfloat16)
    if NSPLIT >= 3:
        p3 = (r1 - p2.to(tl.float32)).to(tl.bfloat16)
        W = pdot(Lge, p3)
        W = pdot(Lge, p2, W)
    elif NSPLIT == 2:
        W = pdot(Lge, p2)
    if NSPLIT >= 2:
        W = pdot(Lge, p1, W)
    else:
        W = pdot(Lge, p1)
    return tl.sum(tl.where(offs_t[None, :] < offs_t[:, None], W, 0.0),
                  axis=1)


@triton.jit
def _dth_factor(v, IO_DT: tl.constexpr, PREC: tl.constexpr):
    """DTH_ROT's map factor as the MMAs CONSUMED it (THETAFREQ, `common._opv`).

    16-bit IO: the stored map IS the operand (`_to_opd` of a bf16 / fp16 map is
    exact), so the fp32 view is returned unchanged -- bitwise the pre-fp32 body.
    fp32 IO: the stored map is fp32 but a "tf32" MMA reads it rounded (`pdot`'s
    rna, or the tensor core's truncation under precision="tf32"); the
    angle adjoint must contract THAT value, or the exact shift identity
    sum_t dtheta_t = 0 breaks at 2^-11 and the per-increment suffix sums carry
    it coherently (the fp32 large_phase DT / angle excess).  The general route's fused backward
    does the same with `_opv` on its rotated tiles.  Identity at "ieee"."""
    if IO_DT == "fp32":
        v = _opv(v, tl.float32, False, PREC)
    return v


# ---------------------------------------------------------------------------
# 1. the scan adjoint -- Mamba-3's `dqkv`, our decay algebra
# ---------------------------------------------------------------------------

def _bwd_configs():
    from .fused_fwd_kernel import autotune_off
    mode = siso_autotune_tier(
        _env("SISO_AUTOTUNE", "0" if autotune_off() else "1"))
    one = _env("SISO_BWD_ONECFG", "")
    if one:
        # MEASUREMENT / TRIAGE ONLY: "warps,stages,maxnreg" pins the space
        # to a SINGLE config (a faulting launch poisons the CUDA context,
        # so triage needs one process per entry).
        w, st, r = one.split(",")
        return [triton.Config({}, num_stages=int(st), num_warps=int(w),
                              maxnreg=(None if r in ("", "None")
                                       else int(r)))]
    _light = lambda: [triton.Config({}, num_stages=s, num_warps=w)
                      for s in (1, 2, 3) for w in (4, 8)]
    _full = lambda: [triton.Config({}, num_stages=s, num_warps=w, maxnreg=r)
                     for s in (1, 2, 3) for w in (2, 4, 8)
                     for r in (None, 128, 256)]
    from .calibration_ssd import grid
    return grid("siso_bwd", mode, full=_full, light=_light,
                generic_pin=lambda: [triton.Config({}, num_stages=2,
                                                   num_warps=8)])


# The 128-BYTE TILE LAW, on this kernel's narrowest 16-bit operand.
#
# On sm_90 a `tl.dot` operand tile that is 16-bit and under 128 bytes
# wide gets a 64-byte-swizzle nvmma staging buffer whose wgmma descriptor
# Triton 3.6 strides as if it were the 128-byte mode -- an ILLEGAL
# ACCESS.  It is a BYTE rule, so shape-exact rather than mode-wide.
#
# The narrow operand here is `x` / `dY` (tile `[T, G_SIZE]`): at bf16 IO
# a head width of 16 is 32 bytes.  The trigger is the narrow tile AND
# `(4 warps, 1 stage)`, and the rule below removes exactly that
# intersection.  It is keyed on BYTES, not on the dtype or `G_SIZE`
# alone: fp32-operand builds never reach the 16-bit swizzle and are
# untouched.
_MMA16 = ("bf16", "fp16")


def _prune_bwd(configs, named_args, **kwargs):
    """The sm_90 eight-warp rule and the 128-byte tile law, applied before anything is
    timed -- an autotuner that TIMES a mis-compiled config either dies
    or selects it."""
    gt = kwargs.get("G_SIZE", named_args.get("G_SIZE", 64))
    io = kwargs.get("IO_DT", named_args.get("IO_DT", "fp32"))
    try:
        cc = torch.cuda.get_device_capability()
    except Exception:                       # noqa: BLE001
        cc = (9, 0)
    if cc != (9, 0):
        return configs
    if not _mmav3_eight_warp_ok(gt):
        configs = [c for c in configs if c.num_warps < 8]
    if io in _MMA16 and gt * 2 < 128:
        configs = [c for c in configs
                   if not (c.num_warps == 4 and c.num_stages == 1)]
    return configs


# ── the tuning launch may not alias ──────────────────────────────────
#
# `_siso_bwd_scan_kernel` is autotuned, and `triton.autotune` picks a
# config by REPLAYING the launch (warmup + timed reps, once per key).
# With `SSD_MIMO_SISO_INPLACE=1` (the default) `dBs`/`dCs` are the SAME
# memory as `Bs`/`Cs`, so every replay reads the previous replay's
# output: the adjoint is applied to itself dozens of times, overflows,
# and every gradient comes back NaN on the first call per autotune key.
#
# The ONE launch the autotuner benchmarks therefore runs on its own
# `dBs`/`dCs` buffers (idempotent: every other output is a plain store);
# aliasing starts from the second, cached launch.
# Guard decisions use the actual autotuner cache key, including its ordered
# operand dtype suffix. Cache resets and failed tuning therefore stay safe.
# Retained for existing test fixtures that clear this former guard registry;
# its contents no longer decide whether a launch may alias its inputs.
_BWD_TUNED: set = set()

# Default for `_siso_bwd_scan_kernel`'s SCHED bitmask: ES + EE + IL = 35.
# The three bits are NOT additive: ptxas only stops serializing the wgmma
# once EVERY long live range across the middle of the body is gone.  Bits
# 4 (SR), 8 (RC) and 16 (EV) were measured negatives on top of 35 and
# stay off; they are reachable through `SSD_MIMO_SISO_BWD_SCHED`.
_SCHED_DEFAULT = 35


# ── TMA, and the four conditions that let it run ──────────────────────
#
# `tl.make_tensor_descriptor` needs a global scratch buffer; Triton asks
# for it through the process-wide allocator, which Mamba-3 installs the
# same way.  Installing ours would OVERWRITE theirs (and theirs ours) if
# both ran in one process, so we install exactly the same thing: a plain
# `torch.empty(size, int8)` on the current device.  Idempotent either way.
_TMA_ALLOC = False


def _install_tma_allocator():
    global _TMA_ALLOC
    if _TMA_ALLOC:
        return
    def _alloc(size, alignment, stream):        # noqa: ANN001
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(_alloc)
    _TMA_ALLOC = True


def _tma_refusal(*, tensors, N, G_SIZE):
    """None when the scan adjoint may stage through descriptors, else why
    not -- BY NAME, as the route's other refusals are.

    A TMA descriptor requires (i) an innermost stride of exactly 1 ON THE
    AXIS THE DESCRIPTOR MAKES INNERMOST, (ii) a row stride that is a
    multiple of 16 bytes, (iii) a block row that is a multiple of 16
    bytes and (iv) an element size the copy engine serves (1, 2 or 4 --
    fp64 leaves are out).  Condition (i) for `x` / `dx` is EXACTLY what
    the token-major layout buys: the family's `[B, H, G, R, L]`
    puts `L` innermost, so its `s_xg` is `L` and no descriptor over a
    `[T, G]` block can be built at all.

    `tensors` is a list of `(name, tensor, row_stride, inner_stride,
    block_row)` in ELEMENTS, taken from the stride set the kernel is
    actually launched with -- not from `.stride()`, which for `x` says
    nothing about which axis the descriptor uses.
    """
    if _portable.nvidia_capability()[0] < 9:
        return "TMA needs sm_90 or later"
    for nm, t, row, inner, blk in tensors:
        it = t.element_size()
        if inner != 1:
            return ("%s: the descriptor's innermost stride is %d, not 1 "
                    "(x / dx need K25's token-major layout, "
                    "SSD_MIMO_SISO_XTM != 0)" % (nm, inner))
        if it > 4:
            return "%s is %d-byte (TMA serves 1, 2 and 4)" % (nm, it)
        if (row * it) % 16:
            return "%s row stride %d B is not a multiple of 16" % (
                nm, row * it)
        if (blk * it) % 16:
            return "%s block row %d B is not a multiple of 16" % (
                nm, blk * it)
    return None


def _autotune_cache_key(kernel, args, kwargs):
    """Match Triton's Autotuner.run key, including dtype suffix ordering."""
    all_args = {**dict(zip(kernel.arg_names, args)), **kwargs}
    named = {k: v for k, v in all_args.items() if k in kernel.arg_names}
    key = [named[k] for k in kernel.keys if k in named]
    key.extend(str(arg.dtype) for arg in named.values()
               if hasattr(arg, "dtype"))
    return tuple(key)


def _bwd_tuning_launch(sig, *, fitted=False) -> bool:
    """An uncached multi-config launch must use non-aliased map outputs.

    A cached resource-fit plan calls the JIT function once, bypassing the
    autotuner, so it can keep the normal in-place memory policy.
    """
    return (not fitted and len(_siso_bwd_scan_kernel.configs) > 1
            and sig not in _siso_bwd_scan_kernel.cache)


# ── a ONE-config space must still FIT the part ──────────────────────
#
# With two or more configs `triton.autotune` scores an over-budget one
# `inf`.  A SINGLE config (the `pinned` tier, `SSD_MIMO_SISO_BWD_ONECFG`,
# a test's pin) is launched unmeasured and can raise OutOfResources on
# ~100 KB-smem parts.  `_bwd_fit_single` lowers the pipeline DEPTH (no
# arithmetic change; same construction as the fused forward's
# `_fwd_fit_stages`) and, if no depth fits, the warp count, until an
# exact dry compile fits.  Where the config already fits the answer is
# None and the launch goes through the autotuner.
_BWD_FIT: dict = {}
_BWD_FIT_TRAIL: dict = {}


_BWD_OOR_FIT: dict = {}


def _bwd_refit_oor(grid, args, kw, device, sig):
    """The launch options of the first config of the pruned autotune space,
    widest first, whose MEASURED shared memory fits `device_smem_limit`
    (warmup compile, no launch).  Cached per signature, so the autotuner's
    over-budget choice is never launched again for it."""
    from .smem import device_smem_limit
    limit = device_smem_limit(device)
    cand = _prune_bwd(list(_siso_bwd_scan_kernel.configs), {},
                      G_SIZE=kw["G_SIZE"], IO_DT=kw["IO_DT"])
    cand = sorted(cand, key=lambda c: (-c.num_warps, -c.num_stages))
    for c in cand:
        opts = dict(num_warps=c.num_warps, num_stages=c.num_stages)
        if c.maxnreg is not None:
            opts["maxnreg"] = c.maxnreg
        try:
            k = _siso_bwd_scan_kernel.fn.warmup(*args, grid=grid, **kw, **opts)
        except Exception:      # noqa: BLE001
            continue
        if int(k.metadata.shared) <= limit:
            _BWD_OOR_FIT[sig] = opts
            return opts
    return None


def _bwd_fit_single(grid, args, kw, device, sig):
    """None = launch through the autotuner as before; else the launch
    options of the single config refitted to `device_smem_limit`: first
    the same warps at a shallower pipeline (depth changes no arithmetic),
    then fewer warps (a lowering change, rounding-level only, A7) -- each
    candidate passed through `_prune_bwd` first, so the refit can never
    land on a config `_prune_bwd` excludes.  `_BWD_FIT_TRAIL` keeps
    the measured bytes per key for the tests."""
    cfgs = _siso_bwd_scan_kernel.configs
    if len(cfgs) != 1:
        return None
    c = cfgs[0]
    key = (sig, c.num_warps, c.num_stages, c.maxnreg)
    if key in _BWD_FIT:
        return _BWD_FIT[key]
    from .smem import device_smem_limit, smem_clamp_enabled
    fit, trail = None, []
    if smem_clamp_enabled():
        limit = device_smem_limit(device)

        def shared(w, st):
            opts = dict(num_warps=w, num_stages=st)
            if c.maxnreg is not None:
                opts["maxnreg"] = c.maxnreg
            try:
                k = _siso_bwd_scan_kernel.fn.warmup(*args, grid=grid,
                                                    **kw, **opts)
                r = int(k.metadata.shared)
            except Exception:      # noqa: BLE001 -- never break a launch
                r = None
            trail.append((w, st, r))
            return r, opts
        req, _ = shared(c.num_warps, c.num_stages)
        if req is not None and req > limit:
            cand = [triton.Config({}, num_warps=w, num_stages=st)
                    for w in (c.num_warps, c.num_warps // 2,
                              c.num_warps // 4) if w >= 2
                    for st in range(c.num_stages, 0, -1)
                    if (w, st) != (c.num_warps, c.num_stages)]
            cand = _prune_bwd(cand, {}, G_SIZE=kw["G_SIZE"],
                              IO_DT=kw["IO_DT"])
            for cc in cand:
                r, opts = shared(cc.num_warps, cc.num_stages)
                if r is not None and r <= limit:
                    fit = opts
                    break
    _BWD_FIT[key] = fit
    _BWD_FIT_TRAIL[key] = trail
    return fit


# The autotune KEY carries every constexpr that changes the generated
# code, not only the shape ones.  `triton.autotune` caches `best_config`
# under this key alone (the COMPILE cache is separate and does see the
# constexprs), so any lever that moves the chosen config must be in the
# key or the first arm's winner is silently reused by every later arm.
@triton.autotune(configs=_bwd_configs(),
                 key=["T", "N", "G_SIZE", "HAS_TRAP", "HAS_DSKIP",
                      "HAS_X0", "HAS_DSF", "IO_DT", "PREC",
                      "RELOAD", "NOTRANS", "GRID_HB",
                      "TMA_M", "TMA_X", "TMA_Y", "TMA_S", "SCHED", "SPAN_MMA", "DD_SPLIT",
                      "SPAN_AX1", "EXCL1D", "DTH_ROT", "ROT_KA", "PFX_OUT", "NO_DD", "OCC",
                      "LB"],
                 prune_configs_by={"early_config_prune": _prune_bwd})
@triton.jit
def _siso_bwd_scan_kernel(
    Bs_ptr, Cs_ptr,          # [B, nc, T, H, N]  the rotated maps (in)
    dBs_ptr, dCs_ptr,        # dB~ / dC~ (out; ALIASED to Bs/Cs when INPLACE)
    X_ptr, dY_ptr, SAll_ptr, Delta_ptr, A_ptr, DSF_ptr, Dsk_ptr, QK_ptr,
    dX_ptr,                  # x / dx through ONE stride set: the family's
                             # [B, H, G, 1, L] or token-major [B, H, L, G]
    dDelta_ptr,              # [B, H, nc, T]
    dQK_ptr,                 # [B, H, L]   (HAS_TRAP)
    dX0_ptr,                 # [B, H, G, N]
    dDsk_ptr,                # [B, H, G]   per-CTA partial (no atomics)
    Emap_ptr,                # [B, H, nc, T] DDSPLIT map-side E (read only)
    Dth_ptr,                 # DTH_ROT: per-head dL/dtheta [.., T, H, N/2] fp32 (out)
    Tail_ptr,                # PFX_OUT: [B, H, nc] carry scalar (out)
    s_mb, s_mc, s_mt, s_mh, s_mn,
    s_xb, s_xh, s_xg, s_xl,
    s_gb, s_gh, s_gc, s_gt, s_gg,      # dY strides (b, h, chunk, t, g)
    s_sb, s_sh, s_sc, s_sn, s_sg,
    s_db, s_dh, s_dc, s_dt,
    s_wb, s_wh, s_wl,
    s_fb, s_fh, s_fg, s_fn,            # dS_final / dX0 strides
    s_kb, s_kh, s_kg,                  # dDsk partial strides
    s_hb, s_hc, s_ht, s_hh, s_hk,      # DTH_ROT buffer strides
    NC, H_heads,
    T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr,
    PREC: tl.constexpr, IO_DT: tl.constexpr,
    HAS_TRAP: tl.constexpr, HAS_DSKIP: tl.constexpr,
    DSKIP_DD: tl.constexpr,
    HAS_X0: tl.constexpr, HAS_DSF: tl.constexpr,
    IDX64: tl.constexpr,
    RELOAD: tl.constexpr = False,
    NOTRANS: tl.constexpr = False,
    GRID_HB: tl.constexpr = False,
    TMA_M: tl.constexpr = False,     # Bs / Cs / dB~ / dC~
    TMA_X: tl.constexpr = False,     # x / dx
    TMA_Y: tl.constexpr = False,     # dY
    TMA_S: tl.constexpr = False,     # S_all
    SCHED: tl.constexpr = 0,         # the LIVE-RANGE bitmask
    SPAN_MMA: tl.constexpr = 0,      # SPANMMA: 0 tile form, k = NSPLIT
    SPAN_AX1: tl.constexpr = False,  # SPANAX1: row-axis exclusive scans
    EXCL1D: tl.constexpr = False,    # write-term prefix as ONE 1-D scan
    ROT_KA: tl.constexpr = 0,
    DTH_ROT: tl.constexpr = False,   # angle adjoint in the ROTATED frame, fp32
    PFX_OUT: tl.constexpr = False,   # write prefix + carry combined on the host
    NO_DD: tl.constexpr = False,     # dL/ddelta not requested: the block is skipped
    DD_SPLIT: tl.constexpr = False,  # DDSPLIT: map-side dDelta terms precomputed
    ABL: tl.constexpr = 0,
    OCC=0,           # autotune-key occupancy bucket only (`occ_bucket`)
    LB=0,            # autotune-key sequence-length bucket only (`len_bucket`)
):
    """Reverse chunk loop, one CTA per (batch, head).  NO trig, NO pair
    shuffle, NO phase tile -- the rotation was consumed in the forward.

    `SCHED` is the LIVE-RANGE bitmask and is NUMERICS-NEUTRAL: every
    bit only moves an expression earlier or later in the body, or
    RE-READS a value instead of holding it, so the arithmetic and its
    order per output are unchanged.  It exists because ptxas reports:

        ptxas info : (C7512) Potential Performance Loss: wgmma.mma_async
        instructions are serialized due to insufficient register
        resources for the function `_siso_bwd_scan_kernel`

    -- a diagnostic it does NOT emit for `mamba3_siso_bwd_kernel_dqkv`.
    The bits:

        1  ES  store `dB~` / `dC~` where they are COMPUTED, not after the
               dDelta block (they are [T, N] fp32 accumulators: 128
               registers held across the whole state side for nothing)
        2  EE  take the intra-chunk part of `E` right after `Mm` / `SB`,
               so the two [T, T] fp32 products die at once
               (Mamba-3: "this is register-heavy so we compute it early
               before spilling")
        4  SR  RE-READ `S_all` (volatile) for the `alpha * sum(S . dS)`
               tail instead of holding `S_in` live to the end of the body
               (Mamba-3's `SSM_States_reloaded`)
        8  RC  rebuild `dec_mask` at its second use instead of holding the
               [T, T] tile (Mamba-3's `RECOMPUTE_MASK`)
       16  EV  take `Vj` right after `BdS`'s last MMA use, so `BdS` and
               the fp32 `xt` die before the dDelta block
       32  IL  issue each state-side dot immediately before its consumer,
               so at most ONE [T, N] fp32 accumulator is live at a time
       64  RM  (spill lever) do not hold the masked [T, T]
               operands `Mop` / `Score` across the body: REBUILD each at
               its consumer from a second `dY x^T` / `C B^T` dot (two
               extra [T, T] MMAs, same operands, same mask, same
               rounding) -- trades tensor-core work for live registers
      128  PH2 (spill lever) two phases per chunk: the MAP phase (dC~,
               dB~, dx, dQK, E) is finished and STORED before any
               state-side operand (`S_all`) is loaded; only the dS
               update and the carry/read terms run in the STATE phase

    `ABL` is an ATTRIBUTION bitmask and MEASUREMENT ONLY: every bit but
    0 computes the wrong gradient and none of them is reachable from the
    entry point (`SSD_MIMO_SISO_BWD_ABL`)."""
    # Which axis is program_id(0) decides WHICH CTAs land on neighbouring
    # SMs; the stored maps are `[B, nc, T, H, N]`, so adjacent HEADS are
    # adjacent in memory and adjacent BATCHES are far apart.  `GRID_HB`
    # selects the order (Mamba-3's `dqkv` grids `(nheads, batch)`).
    if GRID_HB:
        h = tl.program_id(0)
        b = tl.program_id(1)
    else:
        b = tl.program_id(0)
        h = tl.program_id(1)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)

    # the SCHED live-range bits, unpacked once (constexpr folds them away)
    SCHED_ES: tl.constexpr = (SCHED.value & 1) != 0
    SCHED_EE: tl.constexpr = (SCHED.value & 2) != 0
    SCHED_SR: tl.constexpr = (SCHED.value & 4) != 0
    SCHED_RC: tl.constexpr = (SCHED.value & 8) != 0
    SCHED_EV: tl.constexpr = (SCHED.value & 16) != 0
    SCHED_IL: tl.constexpr = (SCHED.value & 32) != 0
    SCHED_RM: tl.constexpr = (SCHED.value & 64) != 0
    SCHED_P2: tl.constexpr = (SCHED.value & 128) != 0

    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_g = tl.arange(0, G_SIZE)
    causal = offs_t[:, None] >= offs_t[None, :]
    strict = offs_t[:, None] > offs_t[None, :]
    causalT = offs_t[None, :] >= offs_t[:, None]
    strictT = offs_t[None, :] > offs_t[:, None]
    if SPAN_MMA > 0:
        # SPANMMA: Lge[s, i] = [i >= s], the A operand of W = Lge @ P
        Lge = tl.where(causalT, 1.0, 0.0).to(tl.bfloat16)
    neg_inf = float("-inf")

    if HAS_DSF:
        dS = tl.load(DSF_ptr + b * s_fb + h * s_fh
                     + offs_g[None, :] * s_fg
                     + offs_n[:, None] * s_fn).to(tl.float32)
    else:
        dS = tl.zeros([N, G_SIZE], dtype=tl.float32)
    if HAS_DSKIP:
        dv = tl.load(Dsk_ptr + h * G_SIZE + offs_g)
    if DSKIP_DD:
        dsk_acc = tl.zeros([G_SIZE], dtype=tl.float32)

    if TMA_M or TMA_X or TMA_Y or TMA_S:
        # Deep staging: every operand arrives through
        # `tl.make_tensor_descriptor`, so a TMA block lands straight in
        # shared memory and costs NO address registers (a `tl.load` over a
        # computed pointer tile pays both).
        #
        # Every descriptor below needs an innermost stride of 1 and a
        # 16-byte-aligned outer stride; `_siso_tma_ok` refuses the launch
        # by name otherwise, which is why the TOKEN-MAJOR x is a
        # PRE-CONDITION for the x / dx descriptors (the family layout has
        # `s_xg == L`).
        if TMA_M:
            b_desc = tl.make_tensor_descriptor(
                Bs_ptr + b * s_mb + h * s_mh, shape=[NC * T, N],
                strides=[s_mt, 1], block_shape=[T, N])
            c_desc = tl.make_tensor_descriptor(
                Cs_ptr + b * s_mb + h * s_mh, shape=[NC * T, N],
                strides=[s_mt, 1], block_shape=[T, N])
            db_desc = tl.make_tensor_descriptor(
                dBs_ptr + b * s_mb + h * s_mh, shape=[NC * T, N],
                strides=[s_mt, 1], block_shape=[T, N])
            dc_desc = tl.make_tensor_descriptor(
                dCs_ptr + b * s_mb + h * s_mh, shape=[NC * T, N],
                strides=[s_mt, 1], block_shape=[T, N])
        if TMA_X:
            x_desc = tl.make_tensor_descriptor(
                X_ptr + b * s_xb + h * s_xh, shape=[NC * T, G_SIZE],
                strides=[s_xl, 1], block_shape=[T, G_SIZE])
            dx_desc = tl.make_tensor_descriptor(
                dX_ptr + b * s_xb + h * s_xh, shape=[NC * T, G_SIZE],
                strides=[s_xl, 1], block_shape=[T, G_SIZE])
        if TMA_Y:
            dy_desc = tl.make_tensor_descriptor(
                dY_ptr + b * s_gb + h * s_gh, shape=[NC * T, G_SIZE],
                strides=[s_gt, 1], block_shape=[T, G_SIZE])
        if TMA_S:
            s_desc = tl.make_tensor_descriptor(
                SAll_ptr + b * s_sb + h * s_sh, shape=[NC * N, G_SIZE],
                strides=[s_sn, 1], block_shape=[N, G_SIZE])

    if DTH_ROT:
        rot_sign_n = tl.where((tl.arange(0, N) % 2) == 0, 1.0, -1.0)
    for cr in range(NC):
        c = NC - 1 - cr

        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        # SEGPREFIX (fused_fwd_kernel `_SEGPREFIX_NOTE`): the host prefix
        # restarts at reset tokens (sentinel -1); decoded here
        D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, False)
        seg_end = tl.max(seg_h, axis=0)
        D_end = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
        f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
        alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end), 0.0)
        dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end - D_h)), 0.0)
        dec_mask = tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h),
                                   a_val * (D_h[:, None] - D_h[None, :]),
                                   neg_inf))
        if NOTRANS:
            # the same mask with its two index axes swapped -- REBUILT,
            # never transposed (see the NOTRANS block below)
            dec_maskT = tl.exp(tl.where((strictT if HAS_TRAP else causalT) & _same_seg(seg_h),
                                        a_val * (D_h[None, :]
                                                 - D_h[:, None]),
                                        neg_inf))
        if SCHED_RC:
            # ── RC (SCHED bit 8), Mamba-3's `RECOMPUTE_MASK`.  The [T, T]
            # decay mask has TWO consumers (`Mm` and `Score`); holding it
            # across both costs 32 fp32 registers, rebuilding it costs
            # one more `MUFU.EX2` per element.
            dec_mask_2 = tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h),
                                         a_val * (D_h[:, None]
                                                  - D_h[None, :]),
                                         neg_inf))
            if NOTRANS:
                dec_maskT_2 = tl.exp(tl.where((
                    strictT if HAS_TRAP else causalT) & _same_seg(seg_h),
                    a_val * (D_h[None, :] - D_h[:, None]), neg_inf))

        mo = (b * s_mb + c * s_mc + h * s_mh
              + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn)
        # MEASUREMENT ONLY (ABL 512): read BOTH maps from chunk 0 on every
        # iteration.  The arithmetic is then wrong, but the map WORKING SET
        # collapses from [B, nc, T, H, N] (DRAM) to [B, T, H, N]
        # (L2-resident), isolating the cost of the per-head stored map.
        mo_ld = (b * s_mb + h * s_mh
                 + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn
                 ) if (ABL & 512) else mo
        xo = (X_ptr + b * s_xb + h * s_xh + c * T * s_xl
              + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl)
        go = (dY_ptr + b * s_gb + h * s_gh + c * s_gc
              + offs_t[:, None] * s_gt + offs_g[None, :] * s_gg)
        if TMA_M:
            Bop = _to_opd(b_desc.load([c * T, 0]), IO_DT)
            Cop = _to_opd(c_desc.load([c * T, 0]), IO_DT)
        else:
            Bop = _to_opd(tl.load(Bs_ptr + mo_ld), IO_DT)
            Cop = _to_opd(tl.load(Cs_ptr + mo_ld), IO_DT)
        if TMA_X:
            xop = _to_opd(x_desc.load([c * T, 0]), IO_DT)
        elif RELOAD:
            xop = _to_opd(tl.load(xo), IO_DT)
        if TMA_Y:
            dYop = _to_opd(dy_desc.load([c * T, 0]), IO_DT)
        elif RELOAD:
            dYop = _to_opd(tl.load(go), IO_DT)
        if TMA_X or RELOAD:
            # RELOAD, Mamba-3's `dO_reloaded` / `v_block_reloaded`.
            # `x` and `dY` are BOTH an MMA operand (a [T, G] tile the
            # wgmma reads out of shared memory) and the operand of four
            # elementwise reductions.  Fusing the two uses pins the tile
            # in fp32 REGISTERS (never contract a dot operand
            # elementwise).  A second, `volatile` load for the reductions
            # stops Triton CSE-ing the two back into one value.  Under TMA
            # it is not optional: contracting a descriptor tile
            # elementwise would pull it back into registers.
            #
            # Not a numerics change: `_to_opd(u)` and `_to_opd(u.to(fp32))`
            # differ by an fp32 round trip that is exact for every (IO
            # dtype, operand dtype) pair this route admits.
            xt = tl.load(xo, volatile=True).to(tl.float32)
        else:
            xt = tl.load(xo).to(tl.float32)
            xop = _to_opd(xt, IO_DT)
        if TMA_Y or RELOAD:
            dY = tl.load(go, volatile=True).to(tl.float32)
        else:
            dY = tl.load(go).to(tl.float32)
            dYop = _to_opd(dY, IO_DT)
        if SCHED_P2:
            pass            # PH2: S_all is loaded in the state phase below
        elif ABL & 64:
            S_in = tl.zeros([N, G_SIZE], dtype=SAll_ptr.dtype.element_ty)
        elif TMA_S:
            S_in = s_desc.load([c * N, 0])
        else:
            S_in = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                           + offs_n[:, None] * s_sn
                           + offs_g[None, :] * s_sg)

        dSop = _to_opd(dS, IO_DT)

        # ── never TRANSPOSE A DOT RESULT ──────────────────────────────
        # `tl.trans` of a `[T, T]` MMA accumulator is a LAYOUT CONVERSION
        # (shared-memory round trip with a barrier on each side, inside a
        # loop that is already latency-bound), whereas `tl.trans` of a dot
        # OPERAND is free: a wgmma smem descriptor transposes its B operand
        # in hardware.
        #
        # So the transposed product is obtained by SWAPPING THE OPERANDS of
        # a second dot (`dY @ x^T` and `x @ dY^T`), as Mamba-3's `dqkv`
        # does: pay one extra dot rather than move a tile.
        #
        # `dec_mask` transposed is rebuilt from `D_h` with the two index
        # axes swapped, never transposed as a tile.
        if ABL & 128:
            # MEASUREMENT ONLY: what the two [T, T] transcendental masks
            # cost, with every dot and every store left in place
            dec_mask = tl.where(strict if HAS_TRAP else causal, 1.0, 0.0)
            dec_maskT = tl.where(strictT if HAS_TRAP else causalT,
                                 1.0, 0.0)
        if NOTRANS:
            dScT = pdot(xop, tl.trans(dYop), input_precision=PREC)
            SBT = pdot(Bop, tl.trans(Cop), input_precision=PREC)
            MmT = dScT * dec_maskT
            if SCHED_RC:
                ScoreT = SBT * dec_maskT_2
            else:
                ScoreT = SBT * dec_maskT
            # `dC~`'s map term still wants the UN-transposed `M`; it is
            # the only one, so it is the only dot whose operands stay in
            # the natural order.
            dSc = pdot(dYop, tl.trans(xop), input_precision=PREC)
            Mm = dSc * dec_mask
            Mop = _to_opd(Mm, IO_DT)
        else:
            dSc = pdot(dYop, tl.trans(xop), input_precision=PREC)  # [T,T]
            SB = pdot(Cop, tl.trans(Bop), input_precision=PREC)    # [T,T]
            Mm = dSc * dec_mask
            Mop = _to_opd(Mm, IO_DT)
            if SCHED_RC:
                Score = SB * dec_mask_2
            else:
                Score = SB * dec_mask

        # ── EE (SCHED bit 2).  `Mm` and `SB` are [T, T] fp32 tiles (32
        # registers each at T = 64) kept alive only by `Atil = Mm * SB` in
        # the dDelta block at the BOTTOM of the body.  Taking that product
        # here lets both die into their bf16 operands.  The arithmetic is
        # unchanged: `E_intra` is the same two reductions of the same
        # tile, as a [T] vector.
        if SCHED_EE and not (ABL & 1) and not NO_DD and not (ABL & 1024) and not DD_SPLIT:
            # SPANDELTA (fused_bwd_kernel `_SPANDELTA_NOTE`): the score
            # pairs that span each token, not a row sum minus a column sum
            if NOTRANS:
                E_intra = _span_sum_T(MmT * SBT, offs_t)
            elif SPAN_MMA > 0:
                E_intra = _span_sum_mma(Mm * SB, Lge, offs_t, SPAN_MMA)
            elif SPAN_AX1:
                E_intra = _span_sum_ax1(Mm * SB, offs_t)
            else:
                E_intra = _span_sum(Mm * SB, offs_t)

        if SCHED_P2:
            # PH2 map phase: dYS (the only state-side operand of dC~) is
            # formed from a load issued right here and dies into dCt
            S_in = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                           + offs_n[:, None] * s_sn
                           + offs_g[None, :] * s_sg)
        Sop = _to_opd(S_in, IO_DT)
        if ABL & 256:
            # MEASUREMENT ONLY: the four STATE-side dots removed, every
            # store still written (so nothing cascades into DCE) -- the
            # split of the kernel's cost between its map side and its
            # state side
            dYS = tl.zeros([T, N], dtype=tl.float32)
            xdS = tl.zeros([T, N], dtype=tl.float32)
            BdS = tl.zeros([T, G_SIZE], dtype=tl.float32)
        elif not SCHED_IL:
            dYS = pdot(dYop, tl.trans(Sop), input_precision=PREC)  # [T,N]
            xdS = pdot(xop, tl.trans(dSop), input_precision=PREC)  # [T,N]
            BdS = pdot(Bop, dSop, input_precision=PREC)            # [T,G]

        # ── IL (SCHED bit 32).  `dYS` and `xdS` are [T, N] fp32
        # accumulators (64 registers EACH at N = 128); issuing all three
        # state dots together keeps 160 registers live across three
        # consumers that use them one at a time.  Issued next to its
        # consumer, at most ONE is live.  Same dots, same order, same
        # operands.
        # ── ES (SCHED bit 1).  `dB~` / `dC~` are finished here but stored
        # at the BOTTOM of the body, which holds two more [T, N] fp32 tiles
        # across the whole dDelta block.  Nothing between here and that
        # store reads `Bs` / `Cs` (the operands are register values loaded
        # at the top of the iteration, and the next iteration reads chunk
        # c - 1), so the store may move up unchanged -- including under
        # INPLACE.
        if SCHED_IL and not (ABL & 256):
            dYS = pdot(dYop, tl.trans(Sop), input_precision=PREC)  # [T,N]
        dCt = f[:, None] * dYS + pdot(Mop, Bop, input_precision=PREC)
        if DTH_ROT:
            # ── DTH_ROT: dL/dtheta_t in the ROTATED frame.
            # Every stage of the read/write maps is a per-pair rotation, so
            # they commute with J and (M^T g)^T J u = g^T J (M u): the angle
            # adjoint is pair_contract(dC~, C~) + pair_contract(dB~_s, B~_s)
            # (the write scale cancels: (w dBs)^T J (Bs / w)).  Taken HERE
            # from the fp32 MMA accumulators and the exact bf16 maps the
            # forward used -- not from the bf16-ROUNDED dB~ / dC~ the
            # pointwise adjoint reads back, whose 2^-9 relative rounding
            # the difference g_e u_o - g_o u_e amplifies (the SISO route's
            # theta_proj.weight excess).  The trapezoid's qk adjoint adds
            # exactly zero (qk is invariant to the common angle) and is not
            # taken.  Needs pre-rotation biases (BIAS_PRE_ROT, host-checked).
            Cre = _dth_factor(tl.load(Cs_ptr + mo, volatile=True).to(tl.float32), IO_DT, PREC)
            if ROT_KA == N // 4:
                ga, _ = prefix_half(dCt, T, N)
                xa, _ = prefix_half(Cre, T, N)
                rs = tl.where((tl.arange(0, N // 2) % 2) == 0, 1.0, -1.0)
                da = _pair_contract(ga, xa, rs, T, N // 2)
                dth_acc = join_halves(da, tl.full((T, N // 4), 0, tl.float32), T, N // 2)
            else:
                dth_acc = _pair_contract(dCt, Cre, rot_sign_n, T, N)
        if SCHED_ES and not (ABL & 4):
            if TMA_M:
                dc_desc.store([c * T, 0], dCt.to(dCs_ptr.dtype.element_ty))
            else:
                tl.store(dCs_ptr + mo, dCt.to(dCs_ptr.dtype.element_ty))
        if SCHED_IL and not (ABL & 256):
            xdS = pdot(xop, tl.trans(dSop), input_precision=PREC)  # [T,N]
        if NOTRANS:
            dBt = (pdot(_to_opd(MmT, IO_DT), Cop, input_precision=PREC)
                   + dec_agg[:, None] * xdS)
        elif SCHED_RM:
            # RM: the masked tile rebuilt at its consumer (same dot, same
            # mask, same bf16 rounding as `Mop`)
            Mop_r = _to_opd(pdot(dYop, tl.trans(xop), input_precision=PREC)
                            * tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h),
                                    a_val * (D_h[:, None] - D_h[None, :]), neg_inf)), IO_DT)
            dBt = (pdot(tl.trans(Mop_r), Cop, input_precision=PREC)
                   + dec_agg[:, None] * xdS)
        else:
            dBt = (pdot(tl.trans(Mop), Cop, input_precision=PREC)
                   + dec_agg[:, None] * xdS)
        if DTH_ROT:
            Bre = _dth_factor(tl.load(Bs_ptr + mo, volatile=True).to(tl.float32), IO_DT, PREC)
            if ROT_KA == N // 4:
                ga, _ = prefix_half(dBt, T, N)
                xa, _ = prefix_half(Bre, T, N)
                rs = tl.where((tl.arange(0, N // 2) % 2) == 0, 1.0, -1.0)
                da = _pair_contract(ga, xa, rs, T, N // 2)
                dth_acc += join_halves(da, tl.full((T, N // 4), 0, tl.float32), T, N // 2)
            else:
                dth_acc += _pair_contract(dBt, Bre, rot_sign_n, T, N)
            tl.store(Dth_ptr + b * s_hb + c * s_hc + h * s_hh
                     + offs_t[:, None] * s_ht
                     + tl.arange(0, N // 2)[None, :] * s_hk, dth_acc)
        if SCHED_ES and not (ABL & 2):
            if TMA_M:
                db_desc.store([c * T, 0], dBt.to(dBs_ptr.dtype.element_ty))
            else:
                tl.store(dBs_ptr + mo, dBt.to(dBs_ptr.dtype.element_ty))
        if SCHED_IL and not (ABL & 256):
            BdS = pdot(Bop, dSop, input_precision=PREC)            # [T,G]
        if NOTRANS:
            dx = (pdot(_to_opd(ScoreT, IO_DT), dYop,
                         input_precision=PREC)
                  + dec_agg[:, None] * BdS)
        elif SCHED_RM:
            Sc_r = (pdot(Cop, tl.trans(Bop), input_precision=PREC)
                    * tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h),
                                    a_val * (D_h[:, None] - D_h[None, :]), neg_inf)))
            dx = (pdot(tl.trans(_to_opd(Sc_r, IO_DT)), dYop,
                         input_precision=PREC)
                  + dec_agg[:, None] * BdS)
        else:
            dx = (pdot(tl.trans(_to_opd(Score, IO_DT)), dYop,
                         input_precision=PREC)
                  + dec_agg[:, None] * BdS)
        tbase = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
        if HAS_TRAP:
            # the forward's rank-1 term  y_t += qk_t x_t
            dx += tl.load(QK_ptr + tbase)[:, None] * dY
            tl.store(dQK_ptr + tbase, tl.sum(dY * xt, axis=1))
        if DSKIP_DD:
            # `SSD_MIMO_DSKIP_IN_KC=all` arm: the reduction over T
            # lands here rather than in the gate epilogue
            dsk_acc += tl.sum(dY * xt, axis=0)
        if HAS_DSKIP:
            dx += dY * dv[None, :]
        if not (ABL & 16):
            if TMA_X:
                dx_desc.store([c * T, 0], dx.to(dX_ptr.dtype.element_ty))
            else:
                tl.store(dX_ptr + b * s_xb + h * s_xh + c * T * s_xl
                         + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl,
                         dx.to(dX_ptr.dtype.element_ty))

        # ── EV (SCHED bit 16).  `Vj` is the LAST consumer of `BdS` ([T, G]
        # fp32) and of the fp32 `xt`; taken here they both die before the
        # dDelta block instead of after it.  A [T] vector survives.
        if SCHED_EV and not (ABL & 1) and not NO_DD:
            Vj_e = dec_agg * tl.sum(BdS * xt, axis=1)

        # ── dDelta.  E_t = dL/d(a D_t); dDelta = a E (so that the host's
        # `da_log = (delta_cumsum * dDelta).sum()` is dL/d a_log) ──────
        if not (ABL & 1) and not NO_DD:
            # SPANDELTA (fused_bwd_kernel `_SPANDELTA_NOTE`): E_s is
            # dL/d(a delta_s) directly -- the score pairs that span s, the
            # read term for every i >= s, the write term for every j < s,
            # the carry term for every s.  No reverse prefix sum follows.
            if DD_SPLIT and PFX_OUT:
                E = tl.zeros([T], dtype=tl.float32)
            elif DD_SPLIT:
                # DDSPLIT: span + read terms from `_siso_ddelta_map_kernel`
                # (a SEPARATE buffer: the autotuner replays this launch)
                E = tl.load(Emap_ptr + b * s_db + h * s_dh + c * s_dc
                            + offs_t * s_dt)
            elif ABL & 1024:
                # MEASUREMENT ONLY: the spanning score-pair term dropped
                E = tl.zeros([T], dtype=tl.float32)
            elif SCHED_EE:
                E = E_intra
            elif NOTRANS:
                E = _span_sum_T(MmT * SBT, offs_t)
            elif SPAN_MMA > 0:
                E = _span_sum_mma(Mm * SB, Lge, offs_t, SPAN_MMA)
            elif SPAN_AX1:
                E = _span_sum_ax1(Mm * SB, offs_t)
            else:
                E = _span_sum(Mm * SB, offs_t)
            # The f term is Mamba-3's `tl.sum(QS * dO, axis=1)` and NOT
            # `sum(C~ * dYS)`: the same number, but the second CONTRACTS A
            # DOT OPERAND ELEMENTWISE, dragging `Cop` -- a [T, N] tile the
            # MMA otherwise reads straight out of shared memory -- into
            # fp32 registers (32 KB at N = 128).  One extra [T,N]x[N,G] dot
            # keeps the reduction on a [T, G] tile of register values, the
            # trade Mamba-3's `dqkv` makes.
            if not (ABL & 2048) and not DD_SPLIT:
                # (MEASUREMENT ONLY bit 2048 drops the read term)
                if ABL & 256:
                    QS = tl.zeros([T, G_SIZE], dtype=tl.float32)
                else:
                    QS = pdot(Cop, Sop, input_precision=PREC)      # [T,G]
                E += tl.cumsum(f * tl.sum(QS * dY, axis=1), axis=0,
                               reverse=True)
            if SCHED_EV:
                Vj = Vj_e
            else:
                Vj = dec_agg * tl.sum(BdS * xt, axis=1)
            if PFX_OUT:
                # PFX_OUT: the scan keeps only the two
                # state-dependent terms as a [T] vector and a scalar; the
                # exclusive prefix of the write term and the map-side E
                # are combined chunk-parallel on the host (shifted cumsum,
                # no subtraction).  `dDelta` carries the raw write vector.
                pass
            elif not (ABL & 4096):
                # (MEASUREMENT ONLY bit 4096 drops the write term)
                if EXCL1D:
                    E += _excl_prefix_scan(Vj)
                else:
                    E += _excl_prefix(Vj, offs_t)
            tail = 0.0
            if not (ABL & 8):
                if SCHED_SR:
                    # ── SR (SCHED bit 4), Mamba-3's `SSM_States_reloaded`.
                    # `S_in` is a [N, G] tile -- 64 registers at N = 128 --
                    # whose only consumer past `Sop` is this one scalar
                    # reduction at the very bottom of the body.  A
                    # `volatile` RE-READ (the volatile is what stops
                    # Triton CSE-ing it back into the first load) costs
                    # one L2 hit and frees the whole live range.
                    S_re = tl.load(SAll_ptr + b * s_sb + h * s_sh
                                   + c * s_sc
                                   + offs_n[:, None] * s_sn
                                   + offs_g[None, :] * s_sg,
                                   volatile=True).to(tl.float32)
                    tail += alpha * tl.sum(S_re * dS)
                else:
                    tail += alpha * tl.sum(S_in.to(tl.float32) * dS)
            if PFX_OUT:
                tl.store(dDelta_ptr + b * s_db + h * s_dh + c * s_dc
                         + offs_t * s_dt, Vj)
                tl.store(Tail_ptr + (b * H_heads + h) * NC + c, tail)
            else:
                E += tail
                tl.store(dDelta_ptr + b * s_db + h * s_dh + c * s_dc
                         + offs_t * s_dt, a_val * E)

        # ── dB~ / dC~, in place over the stored maps by default (avoids a
        # second set of map-sized buffers).  `SSD_MIMO_SISO_INPLACE=0`
        # gives them their own buffers -- the A/B for whether the
        # load/store aliasing costs Triton's software pipeliner its
        # prefetch.
        if not SCHED_ES and not (ABL & 2):
            if TMA_M:
                db_desc.store([c * T, 0], dBt.to(dBs_ptr.dtype.element_ty))
            else:
                tl.store(dBs_ptr + mo, dBt.to(dBs_ptr.dtype.element_ty))
        if not SCHED_ES and not (ABL & 4):
            if TMA_M:
                dc_desc.store([c * T, 0], dCt.to(dCs_ptr.dtype.element_ty))
            else:
                tl.store(dCs_ptr + mo, dCt.to(dCs_ptr.dtype.element_ty))

        if not (ABL & 32):
            # the same rule: scale the [T, G] dY tile by f, never the
            # [T, N] read map (Mamba-3 scales `dO_reloaded`, not `q`).
            dS = alpha * dS + pdot(
                tl.trans(Cop), _to_opd(dY * f[:, None], IO_DT),
                input_precision=PREC)

    if HAS_X0:
        tl.store(dX0_ptr + b * s_fb + h * s_fh
                 + offs_g[None, :] * s_fg + offs_n[:, None] * s_fn, dS)
    if DSKIP_DD:
        tl.store(dDsk_ptr + b * s_kb + h * s_kh + offs_g * s_kg, dsk_acc)


# ── DDSPLIT: the MAP-SIDE dDelta terms, chunk-parallel ─────────────────
# `_siso_bwd_scan_kernel`'s dDelta block is a register-pressure cliff.
# Two of its four terms need NO carried state -- the spanning score pairs
# (dY x^T, C B^T, the decay mask) and the read term (C S_in, dY, f) -- so
# they can run ONE CTA PER CHUNK, fully parallel, BEFORE the scan (the
# maps are overwritten in place by the scan).  This kernel writes their
# sum E_map into `dDelta`; the scan then only adds the write term
# (`_excl_prefix` of its `Vj`) and the carry term and multiplies by a.
# Same summands, same order within each term, no subtraction anywhere
# (SPANDELTA's invariant).
@triton.jit
def _siso_ddelta_map_kernel(
    Bs_ptr, Cs_ptr, X_ptr, dY_ptr, SAll_ptr, Delta_ptr, A_ptr, E_ptr,
    s_mb, s_mc, s_mt, s_mh, s_mn,
    s_xb, s_xh, s_xg, s_xl,
    s_gb, s_gh, s_gc, s_gt, s_gg,
    s_sb, s_sh, s_sc, s_sn, s_sg,
    s_db, s_dh, s_dc, s_dt,
    H,
    T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr,
    PREC: tl.constexpr, IO_DT: tl.constexpr, HAS_TRAP: tl.constexpr,
    IDX64: tl.constexpr, SPAN_MMA: tl.constexpr,
):
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
        c = c.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_g = tl.arange(0, G_SIZE)
    causal = offs_t[:, None] >= offs_t[None, :]
    strict = offs_t[:, None] > offs_t[None, :]
    dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
    D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, False)
    f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
    dec_mask = tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h),
                               a_val * (D_h[:, None] - D_h[None, :]),
                               float("-inf")))
    mo = (b * s_mb + c * s_mc + h * s_mh
          + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn)
    Bop = _to_opd(tl.load(Bs_ptr + mo), IO_DT)
    Cop = _to_opd(tl.load(Cs_ptr + mo), IO_DT)
    xop = _to_opd(tl.load(X_ptr + b * s_xb + h * s_xh + c * T * s_xl
                          + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl),
                  IO_DT)
    dY = tl.load(dY_ptr + b * s_gb + h * s_gh + c * s_gc
                 + offs_t[:, None] * s_gt + offs_g[None, :] * s_gg
                 ).to(tl.float32)
    dYop = _to_opd(dY, IO_DT)
    S_in = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                   + offs_n[:, None] * s_sn + offs_g[None, :] * s_sg)
    Sop = _to_opd(S_in, IO_DT)
    # the SAME rounded operands as the scan: Mop = bf16(dY x^T o M), and
    # the span term reads the fp32 product Mm * SB exactly as the scan does
    dSc = pdot(dYop, tl.trans(xop), input_precision=PREC)
    SB = pdot(Cop, tl.trans(Bop), input_precision=PREC)
    P = (dSc * dec_mask) * SB
    if SPAN_MMA > 0:
        Lge = tl.where(offs_t[None, :] >= offs_t[:, None], 1.0, 0.0).to(tl.bfloat16)
        E = _span_sum_mma(P, Lge, offs_t, SPAN_MMA)
    else:
        E = _span_sum(P, offs_t)
    QS = pdot(Cop, Sop, input_precision=PREC)
    E += tl.cumsum(f * tl.sum(QS * dY, axis=1), axis=0, reverse=True)
    tl.store(E_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, E)


# ---------------------------------------------------------------------------
# 2. the rotation adjoint -- Mamba-3's `rotary_bias_angles`, our chain
# ---------------------------------------------------------------------------
#
# ONE chunk-parallel POINTWISE kernel, grid (nc, B), that
#   * un-scales the write map (`dw_t = sum_n dBs . B~`, `dB~ = w_t dBs`),
#   * puts the trapezoid's `qk_t` adjoint back on both maps,
#   * un-rotates through `M(rho)` / `M(theta_h)` and then `M(theta_t)`,
#     taking each ANGLE adjoint from the tile ENTERING its own stage,
#   * REDUCES over the head group so `dB` / `dC` (head-shared) are
#     written ONCE, and
#   * emits `dTheta`, `dThetaH`, `dRotBias`, `dBeta`, `dGamma`.
#
# It is N-SPLIT BY CONSTRUCTION (`NS = min(N, 64)`), exactly as
# Mamba-3's `BLOCK_HEADDIM_QK = min(HEADDIM_QK, 64)`, because the head
# loop keeps several [T, NS] tiles live at once; the N-INDEPENDENT
# outputs (`dw`, `dg`) therefore carry a block axis that one `torch.sum`
# reduces, which is Mamba-3's `dScale` / `dGamma` shape.
#
# Autotuned; the launch the tuner benchmarks runs on private buffers for
# the atomic accumulators (`dRotBias` / `dThetaH` / `dBeta` / `dGamma`).


@triton.jit
def _pair_contract(g, u, rot_sign, T: tl.constexpr, NS: tl.constexpr):
    """dtheta_k = g_e u_o - g_o u_e, summed over the two lanes of a pair.

    `g` is the gradient LEAVING the rotation stage (already un-rotated)
    and `u` the tile ENTERING it -- the raw-frame form of the angle
    adjoint, which is what lets this kernel run after the rotated map
    has been overwritten (see the module docstring)."""
    q = rot_sign[None, :] * g * _pair_swap_2d(u, T, NS)
    e, o = tl.split(tl.reshape(q, (T, NS // 2, 2)))
    return e + o


# ── the rotation adjoint is autotunable ──────────────────────────────
#
# It accumulates `dRotBias` / `dThetaH` / `dBeta` / `dGamma` with atomics,
# so a tuner replay is not idempotent.  The ONE launch the tuner
# benchmarks therefore runs on PRIVATE zeroed buffers for those four
# accumulators (the plain stores -- `dB`, `dC`, `dTheta`, `dW`, `dG` --
# are idempotent and need none); a second, cached launch runs on the
# real buffers.
def _adj_configs():
    # DEFAULT "0": the pinned (8 warps, 2 stages) config -- the tuner picks
    # that same config and costs one extra launch per key.  "=1" runs it
    # anyway (warps x stages), "=full" adds the maxnreg axis.
    mode = siso_autotune_tier(_env("SISO_ADJ_AUTOTUNE", "0"),
                              adjoint=True)
    _generic = lambda: [triton.Config(
        {}, num_stages=int(_env("SISO_ADJ_STAGES", "2")),
        num_warps=int(_env("SISO_ADJ_WARPS", "8")))]
    # DEFAULT light: warps x stages only.  This kernel is POINTWISE (no
    # `tl.dot` on a real tile, 8 KB of shared memory), so the register cap
    # has nothing to trade against; `full` restores the three-axis space.
    _light = lambda: [triton.Config({}, num_stages=st, num_warps=w)
                      for st in (1, 2, 3) for w in (2, 4, 8)]
    _full = lambda: [triton.Config({}, num_stages=st, num_warps=w, maxnreg=r)
                     for st in (1, 2, 3) for w in (2, 4, 8)
                     for r in (None, 128, 256)]
    if mode == "pinned" and (_env("SISO_ADJ_WARPS", "")
                             or _env("SISO_ADJ_STAGES", "")):
        return _generic()          # an explicit pin beats the calibration
    from .calibration_ssd import grid
    return grid("siso_adj", mode, full=_full, light=_light,
                generic_pin=_generic)


# Compatibility for existing test fixtures; cache state now owns the guard.
_ADJ_TUNED: set = set()


def _adj_tuning_launch(sig) -> bool:
    """An uncached multi-config launch must use disposable atomic outputs."""
    return (len(_siso_rot_adjoint_kernel.configs) > 1
            and sig not in _siso_rot_adjoint_kernel.cache)


@triton.autotune(configs=_adj_configs(),
                 key=["T", "N", "NS", "HAS_THETA_H", "HAS_ROT_BIAS",
                      "HAS_BETA", "HAS_GAMMA", "BIAS_PRE_ROT", "HAS_DW",
                      "HAS_TRAP", "THETA_PER_HEAD", "NEED_DTHETA", "RAW_COMMON",
                      "TRAP_DTH_EXCL", "NEED_DBETA", "NEED_DGAMMA", "NEED_DTHH",
                      "NEED_DRB", "ROT_KA", "ROT_STATIC", "OCC"])
@triton.jit
def _siso_rot_adjoint_kernel(
    B_ptr, C_ptr,            # [B, nc, T, N]     RAW head-shared maps
    dBs_ptr, dCs_ptr,        # [B, nc, T, H, N]  dB~ / dC~ (written in place)
    Theta_ptr, ThetaH_ptr, RotBias_ptr, Beta_ptr, Gamma_ptr,
    W_ptr, Gam_ptr, dQK_ptr,   # [B, H, L]  w_t, g_t, dL/d qk_t
    dB_ptr, dC_ptr,          # [B, nc, T, N]     head-REDUCED, written once
    dTheta_ptr,              # [B, nc, T, (H,) N//2]
    dThetaH_ptr, dRotBias_ptr, dBeta_ptr, dGamma_ptr,
    dW_ptr, dG_ptr,          # [B, H, NB, L]  N-block partials
    s_cb, s_cc, s_ct, s_cn,
    s_bb, s_bc, s_bt, s_bn,
    s_qb, s_qc, s_qt, s_qn,          # dB / dC strides
    s_mb, s_mc, s_mt, s_mh, s_mn,
    s_thb, s_thc, s_tht, s_thh, s_thk,
    s_wb, s_wh, s_wl,
    s_pb, s_ph, s_pk, s_pl,          # dW / dG partial strides
    H,
    T: tl.constexpr, N: tl.constexpr, NS: tl.constexpr,
    HAS_THETA_H: tl.constexpr, HAS_ROT_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr, HAS_GAMMA: tl.constexpr,
    BIAS_PRE_ROT: tl.constexpr, HAS_DW: tl.constexpr,
    HAS_TRAP: tl.constexpr, THETA_PER_HEAD: tl.constexpr,
    NEED_DTHETA: tl.constexpr,
    RAW_COMMON: tl.constexpr,
    IDX64: tl.constexpr,
    TRAP_DTH_EXCL: tl.constexpr = False,
    NEED_DBETA: tl.constexpr = True, NEED_DGAMMA: tl.constexpr = True,
    NEED_DTHH: tl.constexpr = True, NEED_DRB: tl.constexpr = True,
    ROT_KA: tl.constexpr = 0,
    ROT_STATIC: tl.constexpr = False,
    OCC=0,           # autotune-key occupancy bucket only (`occ_bucket`)
):
    """ONE pointwise, chunk-parallel launch: the whole rotation adjoint.

    TRAP_DTH_EXCL: with pre-rotation biases the trapezoid dot
    qk_t = g_t <B~_t, C~_t> = g_t <uB, M(rho)^T M(theta_h) uC> does not
    depend on the common per-token angle theta_t, so its adjoint adds
    EXACTLY ZERO to dtheta -- as the difference ddot (C~^T J B~ + B~^T J
    C~) of two O(|ddot||B~||C~|) terms that cancel only in exact
    arithmetic.  The lever takes dtheta from the trapezoid-free gradients
    (the same stage chain, run on dB~ / dC~ before the ddot add), which
    removes that rounding pair; dB / dC / dtheta_h / drho are unchanged
    (they do depend on qk)."""
    tl.static_assert(not RAW_COMMON or (BIAS_PRE_ROT and not HAS_THETA_H
                                        and not HAS_ROT_BIAS))
    c = tl.program_id(0)
    b = tl.program_id(1)
    if IDX64:
        b = b.to(tl.int64)
        c = c.to(tl.int64)

    offs_t = tl.arange(0, T)
    NB: tl.constexpr = N // NS
    rot_sign = tl.where((tl.arange(0, NS) % 2) == 0, 1.0, -1.0)

    # Expose the fixed real/rotated tile distinction to constant folding.
    # Default is the original dynamic loop; both orders visit nb identically.
    for nb_static in tl.static_range(NB if ROT_STATIC else 1):
        for nb_loop in range(1 if ROT_STATIC else NB):
            nb = nb_static if ROT_STATIC else nb_loop
            offs_n = nb * NS + tl.arange(0, NS)
            offs_k = nb * (NS // 2) + tl.arange(0, NS // 2)
            pidx = offs_n // 2

            Braw = tl.load(B_ptr + b * s_bb + c * s_bc
                           + offs_t[:, None] * s_bt
                           + offs_n[None, :] * s_bn).to(tl.float32)
            Craw = tl.load(C_ptr + b * s_cb + c * s_cc
                           + offs_t[:, None] * s_ct
                           + offs_n[None, :] * s_cn).to(tl.float32)

            dB_acc = tl.zeros([T, NS], dtype=tl.float32)
            dC_acc = tl.zeros([T, NS], dtype=tl.float32)
            if NEED_DTHETA and not THETA_PER_HEAD:
                dth_sh = tl.zeros([T, NS // 2], dtype=tl.float32)

            for h in range(H):
                active = (ROT_KA == 0) | (nb * NS < 2 * ROT_KA)
                if active:
                    th = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                                 + offs_t[:, None] * s_tht
                                 + offs_k[None, :] * s_thk).to(tl.float32)
                    cs = _pair_bcast_2d(_cos_approx(th), T, NS)
                    sn = rot_sign[None, :] * _pair_bcast_2d(_sin_approx(th), T, NS)
                else:
                    cs = tl.full((T, NS), 1, tl.float32)
                    sn = tl.full((T, NS), 0, tl.float32)

                uB = Braw
                uC = Craw
                if HAS_BETA:
                    beta_v = tl.load(Beta_ptr + h * N + offs_n).to(tl.float32)
                if HAS_GAMMA:
                    gam_v = tl.load(Gamma_ptr + h * N + offs_n).to(tl.float32)
                if BIAS_PRE_ROT:
                    if HAS_BETA:
                        uB = uB + beta_v[None, :]
                    if HAS_GAMMA:
                        uC = uC + gam_v[None, :]

                if RAW_COMMON:
                    # With pre-rotation biases and no distinct static rotations,
                    # both maps see the SAME orthogonal transform M. Contract
                    # in the raw frame: <dBs, M uB> = <M^T dBs, uB> and
                    # M^T M = rho I, rho = cs^2 + sn^2 for each pair. Retain
                    # rho because approximate trig is not exactly unit norm.
                    # This avoids reconstructing the forward maps and rotating
                    # the trapezoid contributions.
                    mo = (b * s_mb + c * s_mc + h * s_mh
                          + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn)
                    dBt = tl.load(dBs_ptr + mo).to(tl.float32)
                    dCt = tl.load(dCs_ptr + mo).to(tl.float32)
                    if ROT_KA == N // 4:
                        dBraw = rotate_block(dBt, cs, sn, active, T, NS, True)
                    else:
                        dBraw = cs * dBt - sn * _pair_swap_2d(dBt, T, NS)
                    if ROT_KA == N // 4:
                        dCraw = rotate_block(dCt, cs, sn, active, T, NS, True)
                    else:
                        dCraw = cs * dCt - sn * _pair_swap_2d(dCt, T, NS)
                    tb = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
                    if HAS_DW:
                        tl.store(dW_ptr + b * s_pb + h * s_ph + nb * s_pk
                                 + (c * T + offs_t) * s_pl,
                                 tl.sum(dBraw * uB, axis=1))
                        dBraw = dBraw * tl.load(W_ptr + tb)[:, None]
                    if NEED_DTHETA:
                        # The trapezoid dot is invariant to the common angle.
                        dth = (_pair_contract(dBraw, uB, rot_sign, T, NS)
                               + _pair_contract(dCraw, uC, rot_sign, T, NS))
                    if HAS_TRAP:
                        dqk = tl.load(dQK_ptr + tb)
                        rho = cs * cs + sn * sn
                        tl.store(dG_ptr + b * s_pb + h * s_ph + nb * s_pk
                                 + (c * T + offs_t) * s_pl,
                                 dqk * tl.sum(rho * uB * uC, axis=1))
                        ddot = (dqk * tl.load(Gam_ptr + tb))[:, None] * rho
                        dBraw = dBraw + ddot * uC
                        dCraw = dCraw + ddot * uB
                else:
                    if ROT_KA == N // 4:
                        vB = rotate_block(uB, cs, sn, active, T, NS, False)
                    else:
                        vB = cs * uB + sn * _pair_swap_2d(uB, T, NS)
                    if ROT_KA == N // 4:
                        vC = rotate_block(uC, cs, sn, active, T, NS, False)
                    else:
                        vC = cs * uC + sn * _pair_swap_2d(uC, T, NS)

                    # ── forward re-build of B~ / C~ (the maps were overwritten) ─
                    if HAS_ROT_BIAS:
                        rbv = tl.load(RotBias_ptr + h * (N // 2) + pidx)
                        crb = _cos_approx(rbv)
                        srb = rot_sign * _sin_approx(rbv)
                        qB = crb[None, :] * vB + srb[None, :] * _pair_swap_2d(
                            vB, T, NS)
                    else:
                        qB = vB
                    Btil = qB + beta_v[None, :] if (HAS_BETA and not BIAS_PRE_ROT) \
                        else qB

                    pC = vC + gam_v[None, :] if (HAS_GAMMA and not BIAS_PRE_ROT) \
                        else vC
                    if HAS_THETA_H:
                        thhv = tl.load(ThetaH_ptr + h * (N // 2) + pidx)
                        chh = _cos_approx(thhv)
                        shh = rot_sign * _sin_approx(thhv)
                        Ctil = chh[None, :] * pC + shh[None, :] * _pair_swap_2d(
                            pC, T, NS)
                    else:
                        Ctil = pC

                    # ── the stored gradients ───────────────────────────────────
                    mo = (b * s_mb + c * s_mc + h * s_mh
                          + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn)
                    dBt = tl.load(dBs_ptr + mo).to(tl.float32)
                    dCt = tl.load(dCs_ptr + mo).to(tl.float32)
                    tb = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
                    pbase = (dW_ptr + b * s_pb + h * s_ph + nb * s_pk
                             + (c * T + offs_t) * s_pl)

                    # the write scale is a per-token SCALAR on the write map, so
                    # its adjoint is ONE reduction over N (Mamba-3's dScale) and
                    # it carries the N-block axis a torch.sum removes
                    if HAS_DW:
                        tl.store(pbase, tl.sum(dBt * Btil, axis=1))
                        dBt = dBt * tl.load(W_ptr + tb)[:, None]

                    if TRAP_DTH_EXCL and HAS_TRAP and BIAS_PRE_ROT:
                        dBt0 = dBt
                        dCt0 = dCt
                    if HAS_TRAP:
                        # qk_t = g_t sum_n B~ C~ : two per-token scalars, both
                        # adjoints pointwise
                        dqk = tl.load(dQK_ptr + tb)
                        tl.store(dG_ptr + b * s_pb + h * s_ph + nb * s_pk
                                 + (c * T + offs_t) * s_pl,
                                 dqk * tl.sum(Btil * Ctil, axis=1))
                        ddot = (dqk * tl.load(Gam_ptr + tb))[:, None]
                        dBt = dBt + ddot * Ctil
                        dCt = dCt + ddot * Btil

                    # ── un-rotate, stage by stage; each ANGLE adjoint is taken
                    #    from the tile ENTERING its own stage ───────────────────
                    if HAS_BETA and not BIAS_PRE_ROT and NEED_DBETA:
                        tl.atomic_add(dBeta_ptr + h * N + offs_n,
                                      tl.sum(dBt, axis=0), sem=ACC_SEM)
                    if HAS_ROT_BIAS:
                        dvB = crb[None, :] * dBt - srb[None, :] * _pair_swap_2d(
                            dBt, T, NS)
                        if NEED_DRB:
                            tl.atomic_add(dRotBias_ptr + h * (N // 2) + offs_k,
                                          tl.sum(_pair_contract(dvB, vB, rot_sign,
                                                                T, NS), axis=0),
                                          sem=ACC_SEM)
                    else:
                        dvB = dBt

                    if HAS_THETA_H:
                        dpC = chh[None, :] * dCt - shh[None, :] * _pair_swap_2d(
                            dCt, T, NS)
                        if NEED_DTHH:
                            tl.atomic_add(dThetaH_ptr + h * (N // 2) + offs_k,
                                          tl.sum(_pair_contract(dpC, pC, rot_sign,
                                                                T, NS), axis=0),
                                          sem=ACC_SEM)
                    else:
                        dpC = dCt
                    if HAS_GAMMA and not BIAS_PRE_ROT and NEED_DGAMMA:
                        tl.atomic_add(dGamma_ptr + h * N + offs_n,
                                      tl.sum(dpC, axis=0), sem=ACC_SEM)
                    dvC = dpC

                    if ROT_KA == N // 4:
                        dBraw = rotate_block(dvB, cs, sn, active, T, NS, True)
                    else:
                        dBraw = cs * dvB - sn * _pair_swap_2d(dvB, T, NS)
                    if ROT_KA == N // 4:
                        dCraw = rotate_block(dvC, cs, sn, active, T, NS, True)
                    else:
                        dCraw = cs * dvC - sn * _pair_swap_2d(dvC, T, NS)
                    if NEED_DTHETA and TRAP_DTH_EXCL and HAS_TRAP and BIAS_PRE_ROT:
                        # the trapezoid-free chain, for dtheta only
                        if HAS_ROT_BIAS:
                            dvB0 = crb[None, :] * dBt0 - srb[None, :] * \
                                _pair_swap_2d(dBt0, T, NS)
                        else:
                            dvB0 = dBt0
                        if HAS_THETA_H:
                            dvC0 = chh[None, :] * dCt0 - shh[None, :] * \
                                _pair_swap_2d(dCt0, T, NS)
                        else:
                            dvC0 = dCt0
                        if ROT_KA == N // 4:
                            dBraw0 = rotate_block(dvB0, cs, sn, active, T, NS, True)
                        else:
                            dBraw0 = cs * dvB0 - sn * _pair_swap_2d(dvB0, T, NS)
                        if ROT_KA == N // 4:
                            dCraw0 = rotate_block(dvC0, cs, sn, active, T, NS, True)
                        else:
                            dCraw0 = cs * dvC0 - sn * _pair_swap_2d(dvC0, T, NS)
                if NEED_DTHETA:
                    # the angle adjoint of the M(theta_t) stage is taken from
                    # the RAW-frame gradient (dBraw = M^T dvB) against the
                    # tile ENTERING the stage (uB) -- the identity in the
                    # module docstring.  Using dvB here instead would give
                    # the gradient of the stage OUTPUT, a different number.
                    if not RAW_COMMON:
                        if TRAP_DTH_EXCL and HAS_TRAP and BIAS_PRE_ROT:
                            dth = (_pair_contract(dBraw0, uB, rot_sign, T, NS)
                                   + _pair_contract(dCraw0, uC, rot_sign, T, NS))
                        else:
                            dth = (_pair_contract(dBraw, uB, rot_sign, T, NS)
                                   + _pair_contract(dCraw, uC, rot_sign, T, NS))
                    if THETA_PER_HEAD:
                        tl.store(dTheta_ptr + b * s_thb + h * s_thh + c * s_thc
                                 + offs_t[:, None] * s_tht
                                 + offs_k[None, :] * s_thk, dth)
                    else:
                        dth_sh += dth
                if BIAS_PRE_ROT:
                    if HAS_BETA and NEED_DBETA:
                        tl.atomic_add(dBeta_ptr + h * N + offs_n,
                                      tl.sum(dBraw, axis=0), sem=ACC_SEM)
                    if HAS_GAMMA and NEED_DGAMMA:
                        tl.atomic_add(dGamma_ptr + h * N + offs_n,
                                      tl.sum(dCraw, axis=0), sem=ACC_SEM)
                dB_acc += dBraw
                dC_acc += dCraw

            if NEED_DTHETA and not THETA_PER_HEAD:
                tl.store(dTheta_ptr + b * s_thb + c * s_thc
                         + offs_t[:, None] * s_tht
                         + offs_k[None, :] * s_thk, dth_sh)
            tl.store(dB_ptr + b * s_qb + c * s_qc + offs_t[:, None] * s_qt
                     + offs_n[None, :] * s_qn, dB_acc)
            tl.store(dC_ptr + b * s_qb + c * s_qc + offs_t[:, None] * s_qt
                     + offs_n[None, :] * s_qn, dC_acc)


# ---------------------------------------------------------------------------
# Scalar write/trapezoid adjoints after the pointwise map adjoint.
# ---------------------------------------------------------------------------

@triton.jit
def _siso_scalar_reduce_kernel(
    WP, GP, DT, LAM, DDT, DLAM,
    swb: tl.constexpr, swh: tl.constexpr, swk: tl.constexpr, swt: tl.constexpr,
    sgb: tl.constexpr, sgh: tl.constexpr, sgk: tl.constexpr, sgt: tl.constexpr,
    sdb: tl.constexpr, sdh: tl.constexpr, sdt: tl.constexpr,
    slb: tl.constexpr, slh: tl.constexpr, slt: tl.constexpr,
    H: tl.constexpr, L: tl.constexpr, NB: tl.constexpr,
    HAS_TRAP: tl.constexpr, BLOCK: tl.constexpr,
):
    bh = tl.program_id(1).to(tl.int64)
    b, h = bh // H, bh % H
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = t < L
    w = tl.full((BLOCK,), 0.0, tl.float32)
    if HAS_TRAP:
        g = tl.full((BLOCK,), 0.0, tl.float32)
        previous = tl.full((BLOCK,), 0.0, tl.float32)
    for k in tl.static_range(NB):
        w += tl.load(WP + b * swb + h * swh + k * swk + t * swt,
                     mask, other=0).to(tl.float32)
        if HAS_TRAP:
            g += tl.load(GP + b * sgb + h * sgh + k * sgk + t * sgt,
                         mask, other=0).to(tl.float32)
            # Global sequence predecessor, including across BLOCK boundaries.
            previous += tl.load(
                WP + b * swb + h * swh + k * swk + (t - 1) * swt,
                mask & (t > 0), other=0).to(tl.float32)
    out = bh * L + t
    if HAS_TRAP:
        dt = tl.load(DT + b * sdb + h * sdh + t * sdt, mask, other=0)
        lam = tl.load(LAM + b * slb + h * slh + t * slt, mask, other=0)
        total = w + g
        tl.store(DDT + out, lam * total + (1.0 - lam) * previous, mask)
        tl.store(DLAM + out, dt * (total - previous), mask)
    else:
        tl.store(DDT + out, w, mask)


def _siso_scalar_reduce(dW_p, dG_p=None, dw=None, trap_lam=None):
    """Reduce N-block partials and apply the scalar trapezoid chain rule."""
    B, H, NB, L = dW_p.shape
    has_trap = trap_lam is not None
    if has_trap and (dG_p is None or dw is None):
        raise ValueError("trapezoid scalar adjoint needs dG_p and dw")
    dDW = torch.empty((B, H, L), dtype=torch.float32, device=dW_p.device)
    dLam = torch.empty_like(dDW) if has_trap else None
    _siso_scalar_reduce_kernel[(triton.cdiv(L, 256), B * H)](
        dW_p, dG_p if has_trap else dW_p,
        dw if has_trap else dW_p, trap_lam if has_trap else dW_p,
        dDW, dLam if has_trap else dDW,
        *dW_p.stride(), *(dG_p.stride() if has_trap else (0, 0, 0, 0)),
        *(dw.stride() if has_trap else (0, 0, 0)),
        *(trap_lam.stride() if has_trap else (0, 0, 0)),
        H=H, L=L, NB=NB, HAS_TRAP=has_trap, BLOCK=256,
        num_warps=4, enable_fp_fusion=False,
    )
    return dDW, dLam


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def siso_chunk_bwd(
    aux, Ctil, Btil, x_raw, delta_cumsum, a_log, S_all, dY, *,
    theta, theta_h=None, rot_bias=None, beta=None, gamma=None,
    bias_pre_rot=False, dw=None, trap_lam=None, x0=None, d_skip=None,
    dS_final=None, precision="tf32", dskip_dd=False, needs=None,
):
    """The SISO fast path's backward.  Returns a dict of gradients.

    Two launches: the rotation-free scan adjoint (`dqkv`) and the ONE
    pointwise rotation adjoint (`rotary_bias_angles`), plus the small
    `[B, H, L]` trapezoid chain rule on the host.
    """
    # Public operand-mode names are distinct from Triton's dot flag.
    # SISO already rounds operands to IO_DT; io/bf16 therefore share
    # tf32's dot flag (which only affects fp32 operands).
    from .fused_fwd_kernel import dot_precision
    precision = dot_precision(precision)
    Bsz, H, nc, T = delta_cumsum.shape
    G_SIZE = x_raw.shape[2]
    N = Btil.shape[3]
    L = nc * T
    dev = x_raw.device
    f32 = torch.float32
    rot_ka = aux.get("rot_ka_bwd", aux.get("rot_ka", N // 2))
    Bs, Cs = aux["Bs"], aux["Cs"]
    # the forward's own stride set for the maps: it, not `Bs.stride()`,
    # is what the three kernels index them with (`siso_map_head_major`)
    m_st = aux.get("m_st") or (list(Bs.stride()) if Bs is not None
                               else [0, 0, 0, 0, 0])
    io_tag = aux["io_tag"]
    has_trap = trap_lam is not None
    from . import calibration_ssd as _pins
    _pins.note_call("siso_bwd", T=T, N=N, io=io_tag)
    _pins.activate(True, Bsz, H, T, io=io_tag)
    _pins.note_call("siso_adj", T=T, N=N, io=io_tag)
    has_dw = dw is not None
    has_dskip = d_skip is not None
    # ── NEEDS: the gradients the caller
    # asked for (`ctx.needs_input_grad`).  An unrequested output is not
    # computed and comes back None; every requested one is bitwise the
    # full computation's (each skip removes a DEAD branch -- a store or
    # an atomic whose result feeds nothing else).  `needs=None` = all.
    nd = dict(ddelta=True, dtheta=True, dthh=True, drb=True, dbeta=True,
              dgamma=True, dx0=True, ddw=True, dlam=True)
    if needs is not None:
        nd.update(needs)
    if not nd["dx0"]:
        x0 = None

    # ── the route's own TOKEN-MAJOR x / dx ────────────────────────────
    # `siso_route.siso_xtm()` says why.  The forward left its own
    # `[B, H, L, G]` copy in `aux["xt"]`; `dx` is written into a buffer
    # of the same shape, so the store's fast axis is G (stride 1) rather
    # than L.  The kernel body is untouched: x and dx are addressed
    # through ONE stride set, `(s_xb, s_xh, s_xg, s_xl)`.
    from .siso_route import siso_xtm
    xtm_mode = siso_xtm()
    x_k = aux.get("xt")
    if xtm_mode == "0" or x_k is None:
        x_k = x_raw
        x_st = (x_raw.stride(0), x_raw.stride(1), x_raw.stride(2),
                x_raw.stride(4))
        dXt = torch.empty_like(x_raw)
        dX = dXt
    else:
        x_st = (x_k.stride(0), x_k.stride(1), x_k.stride(3), x_k.stride(2))
        dXt = torch.empty_like(x_k)                 # [B, H, L, G]
        dX = None                                   # built after the launch
    dDelta = torch.empty(Bsz, H, nc, T, device=dev, dtype=f32)
    dQK = (torch.empty(Bsz, H, L, device=dev, dtype=f32) if has_trap
           else torch.empty(1, device=dev, dtype=f32))
    dX0 = (torch.empty(Bsz, H, G_SIZE, N, device=dev, dtype=f32)
           if x0 is not None else torch.empty(1, device=dev, dtype=f32))
    dskip_dd = bool(dskip_dd and has_dskip)
    dDsk_p = (torch.empty(Bsz, H, G_SIZE, device=dev, dtype=f32)
              if dskip_dd else torch.empty(1, device=dev, dtype=f32))
    dsk_c = (d_skip.float().contiguous().reshape(H, G_SIZE) if has_dskip
             else torch.empty(1, device=dev, dtype=f32))
    dsf = (dS_final if dS_final is not None
           else torch.empty(1, device=dev, dtype=f32))
    qk_f = aux["dot"] if has_trap else torch.empty(1, device=dev, dtype=f32)
    w_st = dw.stride() if has_dw else (0, 0, 0)
    f_st = (dS_final.stride() if dS_final is not None
            else (dX0.stride() if x0 is not None else (0, 0, 0, 0)))
    k_st = dDsk_p.stride() if dskip_dd else (0, 0, 0)

    idx64 = _needs_i64(x_k, Bs, Cs, S_all, dY, dXt, dDelta,
                       qk_f if has_trap else None)

    abl = int(_env("SISO_BWD_ABL", "0"))
    # ── the LIVE-RANGE schedule bitmask ────────────────────────────────
    # ptxas names the residue itself -- "(C7512) Potential Performance
    # Loss: wgmma.mma_async instructions are serialized due to
    # insufficient register resources" on this kernel and not on
    # `mamba3_siso_bwd_kernel_dqkv`.  Each bit shortens ONE live range
    # and none of them changes an arithmetic expression.
    # `SSD_MIMO_SISO_BWD_SCHED` overrides the default (an integer, or "0"
    # for the unscheduled order).
    sched = int(_env("SISO_BWD_SCHED", str(_SCHED_DEFAULT)), 0)
    # split the x / dY MMA operand from the fp32 register
    # tile the elementwise reductions contract (see RELOAD in the kernel)
    reload_xdy = _env("SISO_BWD_RELOAD", "0") == "1"
    # obtain the transposed [T, T] products by swapping the
    # operands of a second dot instead of transposing an MMA result
    notrans = _env("SISO_BWD_NOTRANS", "0") == "1"
    grid_hb = _env("SISO_BWD_GRID_HB", "0") == "1"
    # SPANMMA: `SSD_MIMO_SPANDELTA_MMA` = NSPLIT (1, 2, 3)
    # takes the SPANDELTA intra term as W = Lge @ P on the tensor cores
    # (`_span_sum_mma`); "0" (default) keeps the [T, T] scan form.  It
    # needs 16-bit IO (the bf16 parts) and the un-transposed tile.
    span_mma = int(_env("SPANDELTA_MMA", "0"))
    if span_mma not in (0, 1, 2, 3):
        raise ValueError("SSD_MIMO_SPANDELTA_MMA must be 0, 1, 2 or 3")
    if span_mma and (io_tag not in ("bf16", "fp16") or notrans):
        span_mma = 0
    # DDSPLIT: `SSD_MIMO_SISO_BWD_DDSPLIT=1` precomputes the
    # map-side dDelta terms chunk-parallel (`_siso_ddelta_map_kernel`)
    dd_split = (_env("SISO_BWD_DDSPLIT", "0") == "1" and not notrans
                and nd["ddelta"]
                and not abl)
    e_map = (torch.empty(Bsz, H, nc, T, device=dev, dtype=f32)
             if dd_split else dDelta)
    # dB~ / dC~ are written IN PLACE over the saved stored maps (Bs / Cs) to save
    # 2 x [B, L, H, N]; that destroys the saved tensors, so a SECOND backward through
    # a retained graph (retain_graph=True, create_graph=True) would read the
    # overwritten maps (contract suite test_retain_graph: grad:dx off by ~1e3).  The
    # engine knows whether this backward keeps the graph: then the maps stay intact.
    inplace = (_env("SISO_INPLACE", "1") == "1"
               and not _keep_graph())
    span_ax1 = _env("SISO_SPAN_AX1", "0") == "1" and not span_mma
    excl1d = _env("SISO_EXCL1D", "0") == "1"
    # DTH_ROT: the angle adjoint in the rotated frame inside the scan
    # (see the kernel).  Served for pre-rotation biases (or none), the
    # cumulated [.., N/2] phase, and the Triton scan (the Gluon twin keeps
    # the pointwise adjoint's dtheta).
    # M3STRUCT (opt-in): the scan in Mamba-3's dqkv structure
    # (`siso_bwd_m3s`); it always takes the DDSPLIT map terms and the
    # host-side prefix combine
    from . import siso_bwd_m3s as _m3s
    m3s_why = _m3s.refusal(
        Bs=Bs, Cs=Cs, x=x_k, dY=dY, S_all=S_all, delta_cumsum=delta_cumsum,
        T=T, N=N, G=G_SIZE, io_tag=io_tag, m_st=m_st, xtm=xtm_mode, abl=abl,
        x0=x0, dskip_dd=dskip_dd) if _m3s.enabled() else "off"
    _m3s._LAST.update(used=False, why=m3s_why)
    if _env("M3S_TRACE", "0") == "1":
        import sys as _sys
        print("M3S_TRACE route=%s" % ("m3s" if m3s_why is None else m3s_why),
              file=_sys.stderr, flush=True)
    if m3s_why is None and nd["ddelta"]:
        dd_split = True
        e_map = torch.empty(Bsz, H, nc, T, device=dev, dtype=f32)
    pfx_out = dd_split and (_env("SISO_PFX_OUT", "0") == "1"
                            or m3s_why is None)
    m3s_nodd = m3s_why is None and not nd["ddelta"]
    tail_buf = (torch.empty(Bsz, H, nc, device=dev, dtype=f32) if pfx_out
                else torch.empty(1, device=dev, dtype=f32))
    # DTH_ROT: a REQUIREMENT, not an opt-in env choice, whenever this call
    # needs dL/dtheta and the Triton scan (not the Hopper Gluon twin,
    # handled separately below) is the one computing it.
    # `siso_route.dth_rot_can_serve` is the single refusal-by-name
    # predicate.  When it is needed but cannot serve (a post-rotation
    # bias, a non-cumulated phase; fp32 IO is served, `_dth_factor`) and
    # the Gluon body does not serve either, `autograd` refuses SISO BY
    # NAME at forward time and the call takes the general route
    # (`siso_route.dth_rot_refusal`); this backward is never reached then.
    from .siso_route import dth_rot_can_serve
    dth_ok = dth_rot_can_serve(io_dtype=io_tag, bias_pre_rot=bias_pre_rot,
                                beta=beta, gamma=gamma, theta=theta, N=N)
    # mandatory when needed and serviceable
    dth_rot = bool(nd["dtheta"]) and dth_ok
    if dth_rot:
        dth_buf = torch.empty(Bsz, nc, T, H, N // 2, device=dev, dtype=f32)
        dth_st = list(dth_buf.stride())
    else:
        dth_buf = torch.empty(1, device=dev, dtype=f32)
        dth_st = [0, 0, 0, 0, 0]

    # stage through TMA descriptors when the layout allows it
    # `SSD_MIMO_SISO_BWD_TMA` selects the operand GROUPS that stage
    # through descriptors: "0" none, "1"/"all" every group whose layout
    # qualifies, "require" the same but RAISES the refusal instead of
    # falling back, or an explicit subset of the letters m (Bs/Cs/dB~/
    # dC~), x (x/dx), y (dY), s (S_all).  Per group, because the four
    # arrive in four different layouts and only some of them qualify:
    # `dY` is the SISO forward's own `[B, H, 1, G, L]` native buffer
    # whose G axis strides by L exactly as the family's x does, so a
    # `[T, G]` descriptor cannot be built over it at all.
    # ── GLUON90_SISO (opt-in, `gluon_sm90_siso_bwd`): the Hopper Gluon
    # twin of the scan adjoint, into the SAME output buffers.  Every
    # refusal is by name and falls through to the Triton kernel below.
    if _portable.gluon_refusal("ssd_mimo") is None:       # kernel_target="portable": no Gluon (not even its import)
        from . import gluon_sm90_siso_bwd as _g90s
    else:
        _g90s = None
    g90_served = False
    g90_adj = None
    g90_dth = False
    if _g90s is not None and _g90s.enabled(dev):
        # a non-family x (e.g. a core called on a [B, L, d] view) is
        # served through ONE contiguous copy in and one copy of dx out
        x_g, dX_g = x_k, dXt
        if (xtm_mode == "0" and x_k.dim() == 5 and x_k.shape[3] == 1
                and not x_k.is_contiguous() and not _g90s.packed_x(x_k)):
            x_g = x_k.contiguous()
            dX_g = torch.empty_like(x_g)
        g90_why = _g90s.refusal(
            Bs=Bs, Cs=Cs, dBs=Bs, dCs=Cs, x=x_g, dX=dX_g, S_all=S_all,
            dY=dY, T=T, N=N, G=G_SIZE, io_tag=io_tag, m_st=m_st,
            dskip_dd=dskip_dd, xtm=xtm_mode, abl=abl, dS_final=dS_final,
            x0=x0)
        if g90_why is None and not inplace:
            dBs_g, dCs_g = torch.empty_like(Bs), torch.empty_like(Cs)
        else:
            dBs_g, dCs_g = Bs, Cs
        _g90s._LAST_ROUTE.update(used=False, why=g90_why)
        if _env("GLUON90_SISO_TRACE", "0") == "1":
            import sys as _sys
            print("G90S_TRACE route=%s" % ("gluon" if g90_why is None
                                             else g90_why),
                  file=_sys.stderr, flush=True)
        g90_adj = None
        if g90_why is None and _g90s.adj_enabled():
            # FUSE_ADJ: the rotation adjoint's per-head work in the Gluon
            # epilogue; the host then only reduces dB / dC over heads
            raw_common_req = (_env("SISO_ADJ_RAW_COMMON", "0") == "1"
                              and bool(bias_pre_rot) and theta_h is None
                              and rot_bias is None)
            need_dth = bool(nd["dtheta"])
            adj_why = _g90s.adj_refusal(
                theta=theta, theta_h=theta_h, rot_bias=rot_bias,
                bias_pre_rot=bias_pre_rot, beta=beta, gamma=gamma, dw=dw,
                raw_common=raw_common_req, need_dtheta=need_dth, Btil=Btil,
                Ctil=Ctil, aux=aux, qk_f=qk_f, has_trap=has_trap, N=N)
            _g90s._LAST_ROUTE["adj_why"] = adj_why
            if adj_why is None:
                g90_adj = dict(
                    theta=theta, Btil=Btil, Ctil=Ctil,
                    dTheta=torch.empty_like(theta, dtype=f32),
                    w=aux["scale"] if has_trap else dw,
                    dW_p=torch.empty(Bsz, H, 1, L, device=dev, dtype=f32))
                if has_trap:
                    g90_adj["gam"] = aux["gam"]
                    g90_adj["dG_p"] = torch.empty(Bsz, H, 1, L, device=dev,
                                                  dtype=f32)
                if beta is not None:
                    g90_adj["beta"] = beta.float().contiguous().reshape(H, N)
                    g90_adj["dBeta"] = torch.zeros(H, N, device=dev, dtype=f32)
                if gamma is not None:
                    g90_adj["gamma"] = gamma.float().contiguous().reshape(H, N)
                    g90_adj["dGamma"] = torch.zeros(H, N, device=dev,
                                                    dtype=f32)
        # DTH_ROT inside the Gluon body (default ON when it serves: the
        # maps sit in smem, so the rotated-frame contraction is nearly
        # free; SSD_MIMO_GLUON90_SISO_DTH_ROT=0 restores the pointwise
        # adjoint's bf16 read-back form, which misses the fp64 oracle on
        # theta_proj.weight)
        g90_dth = (g90_why is None and g90_adj is None
                   and _env("GLUON90_SISO_DTH_ROT", "1") == "1"
                   and bool(nd["dtheta"])
                   and (bool(bias_pre_rot) or (beta is None and gamma is None))
                   and theta.shape[-1] == N // 2)
        if g90_dth and not dth_rot:
            dth_buf = torch.empty(Bsz, nc, T, H, N // 2, device=dev,
                                  dtype=f32)
        if g90_why is None:
            _g90s._LAST_ROUTE["x_copy"] = x_g is not x_k
            _g90s.launch(
                Bs=Bs, Cs=Cs, dBs=dBs_g, dCs=dCs_g, x=x_g, dX=dX_g,
                S_all=S_all, dY=dY, delta_cumsum=delta_cumsum, a_log=a_log,
                dsf=dsf, dsk_c=dsk_c, qk_f=qk_f, dDelta=dDelta, dQK=dQK,
                dX0=dX0, has_trap=has_trap, has_dskip=has_dskip,
                has_dsf=dS_final is not None, has_x0=x0 is not None,
                f_st=f_st, w_st=w_st, idx64=idx64, adj=g90_adj,
                dth=dth_buf if g90_dth else None,
                no_dd=not nd["ddelta"], rot_ka=aux.get("rot_ka_bwd", aux.get("rot_ka", N // 2)))
            if dX_g is not dXt:
                dXt.copy_(dX_g)
            dBs, dCs = dBs_g, dCs_g
            g90_served = True

    # DTH_ROT ENFORCEMENT: the Triton scan below is the one that will
    # compute dL/dtheta whenever the Gluon body did not serve this call at
    # all (`not g90_served`).  If it needs that gradient and cannot serve
    # it with DTH_ROT on, refuse by name instead of silently running the
    # pointwise bf16 read-back adjoint that fails the fp64 large_phase
    # oracle.  `dth_rot` above is forced True whenever `dth_ok` holds, so
    # reaching here with `dth_rot` False means `dth_ok` was False.
    # `dth_rot` False means `dth_ok` was False.
    if nd["dtheta"] and not g90_served and not dth_rot and theta is not None:
        if io_tag not in ("bf16", "fp16", "fp32"):
            why = "%s IO: DTH_ROT serves bf16 / fp16 / fp32 IO" % io_tag
        elif not (bool(bias_pre_rot) or (beta is None and gamma is None)):
            why = "a post-rotation bias (beta/gamma without bias_pre_rot)"
        else:
            why = ("theta is not the cumulated per-pair phase "
                   "(theta.shape[-1]=%d != N//2=%d)" % (theta.shape[-1], N // 2))
        raise NotImplementedError(
            "SISO backward refuses this call by name: it needs dL/dtheta, "
            "the Hopper Gluon body (gluon_sm90_siso_bwd) does not serve it, "
            "and the rotated-frame angle adjoint DTH_ROT cannot serve it "
            "either (%s). DTH_ROT is a REQUIREMENT here, not an option "
            "-- re-run this call with "
            "SSD_MIMO_SISO_FAST=0 to use the general fused route instead."
            % why)

    tma_mode = _env("SISO_BWD_TMA", "0")
    want = ("mxys" if tma_mode in ("1", "all", "require") else
            ("" if tma_mode == "0" else tma_mode))
    groups = dict(
        m=[("Bs", Bs, Bs.stride(2), Bs.stride(4), N),
           ("Cs", Cs, Cs.stride(2), Cs.stride(4), N)],
        x=[("x", x_k, x_st[3], x_st[2], G_SIZE),
           ("dx", dXt, x_st[3], x_st[2], G_SIZE)],
        y=[("dY", dY, dY.stride(3), dY.stride(5), G_SIZE)],
        s=[("S_all", S_all, S_all.stride(3), S_all.stride(4), G_SIZE)],
    )
    tma_why = {}
    on = {}
    for g, ts in groups.items():
        if g not in want:
            on[g] = False
            tma_why[g] = "not requested"
            continue
        why = _tma_refusal(N=N, G_SIZE=G_SIZE, tensors=ts)
        on[g] = why is None
        tma_why[g] = why
        if why is not None and tma_mode == "require":
            raise NotImplementedError(
                "SSD_MIMO_SISO_BWD_TMA=require, group %r: %s" % (g, why))
    tma_m, tma_x, tma_y, tma_s = on["m"], on["x"], on["y"], on["s"]
    if tma_m or tma_x or tma_y or tma_s:
        _install_tma_allocator()
    # the tuning launch runs non-aliased so the autotuner's replay cannot
    # eat its own input -- see `_bwd_tuning_launch` above
    sig = (T, N, G_SIZE, has_trap, has_dskip, dskip_dd, x0 is not None,
           dS_final is not None, io_tag, precision, idx64, abl,
           Bs.dtype, Cs.dtype, x_raw.dtype, dY.dtype, S_all.dtype,
           dXt.dtype, dDelta.dtype, dsf.dtype, dsk_c.dtype, qk_f.dtype,
           dQK.dtype, dX0.dtype, dDsk_p.dtype, delta_cumsum.dtype,
           a_log.dtype, dev, reload_xdy, notrans, grid_hb,
           tma_m, tma_x, tma_y, tma_s, sched, span_mma, dd_split, span_ax1, excl1d, dth_rot, pfx_out, nd["ddelta"], rot_ka,
           occ_bucket(Bsz * H, x_k.device), len_bucket(L))
    if not g90_served:
        dBs = Bs if inplace else torch.empty_like(Bs)
        dCs = Cs if inplace else torch.empty_like(Cs)
    grid = (H, Bsz) if grid_hb else (Bsz, H)
    args = (
        Bs, Cs, dBs, dCs,
        x_k, dY, S_all, delta_cumsum, a_log, dsf, dsk_c, qk_f,
        dXt, dDelta, dQK, dX0, dDsk_p, e_map, dth_buf, tail_buf,
        *m_st,
        *x_st,
        dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3),
        dY.stride(5),
        *S_all.stride(),
        *delta_cumsum.stride(),
        *w_st,
        *f_st,
        *k_st,
        dth_st[0], dth_st[1], dth_st[2], dth_st[3], dth_st[4],
        nc, H,
    )
    kw = dict(
        OCC=occ_bucket(Bsz * H, x_k.device), LB=len_bucket(L),
        T=T, N=N, G_SIZE=G_SIZE, PREC=precision, IO_DT=io_tag,
        HAS_TRAP=has_trap, HAS_DSKIP=has_dskip, DSKIP_DD=dskip_dd,
        HAS_X0=x0 is not None, HAS_DSF=dS_final is not None,
        IDX64=idx64,
        SCHED=sched, SPAN_MMA=span_mma, DD_SPLIT=dd_split, SPAN_AX1=span_ax1, EXCL1D=excl1d, DTH_ROT=dth_rot, ROT_KA=rot_ka, PFX_OUT=pfx_out,
        NO_DD=not nd["ddelta"],
        RELOAD=reload_xdy, NOTRANS=notrans, GRID_HB=grid_hb,
        TMA_M=tma_m, TMA_X=tma_x, TMA_Y=tma_y, TMA_S=tma_s,
        ABL=abl,
    )
    if (not g90_served and m3s_why is not None and inplace
            and _bwd_tuning_launch(_autotune_cache_key(
                _siso_bwd_scan_kernel, args, kw),
                fitted=_BWD_OOR_FIT.get(sig) is not None)):
        # Only the first uncached autotuner call needs separate outputs.
        # The dtype suffix is unchanged because these match the input maps.
        dBs, dCs = torch.empty_like(Bs), torch.empty_like(Cs)
        args = (*args[:2], dBs, dCs, *args[4:])
        inplace = False
    if not g90_served and dd_split:
        # the map-side terms read the rotated maps BEFORE the scan
        # overwrites them in place
        _siso_ddelta_map_kernel[(nc, Bsz * H)](
            Bs, Cs, x_k, dY, S_all, delta_cumsum, a_log, e_map,
            *m_st, *x_st,
            dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3),
            dY.stride(5),
            *S_all.stride(),
            *delta_cumsum.stride(),
            H,
            T=T, N=N, G_SIZE=G_SIZE, PREC=precision, IO_DT=io_tag,
            HAS_TRAP=has_trap, IDX64=idx64, SPAN_MMA=span_mma,
            num_warps=int(_env("SISO_DD_WARPS", "4")), num_stages=1)
    if not g90_served:
        if m3s_why is None:
            # a non-contiguous family x (e.g. a core called on a view) is
            # served through ONE contiguous copy in and one copy of dx out
            x_m, dX_m = x_k, dXt
            if not x_k.is_contiguous():
                x_m = x_k.contiguous()
                dX_m = torch.empty_like(x_m)
            _m3s.launch(
                Bs=Bs, Cs=Cs, dBs=dBs, dCs=dCs, x=x_m, dX=dX_m, dY=dY,
                S_all=S_all, delta_cumsum=delta_cumsum, a_log=a_log,
                dsf=dsf, dsk_c=dsk_c, qk_f=qk_f, dQK=dQK, vj=dDelta,
                tail=tail_buf, dth=dth_buf, dth_st=dth_st, f_st=f_st,
                w_st=w_st,
                has_trap=has_trap, has_dskip=has_dskip,
                has_dsf=dS_final is not None, dth_rot=dth_rot,
                no_dd=m3s_nodd)
            if dX_m is not dXt:
                dXt.copy_(dX_m)
        else:
            fit = _bwd_fit_single(grid, args, kw, dev, sig)
            if fit is None:
                fit = _BWD_OOR_FIT.get(sig)
            if fit is None:
                try:
                    _siso_bwd_scan_kernel[grid](*args, **kw)
                except triton.runtime.errors.OutOfResources:
                    # the autotuned choice does not fit THIS device's shared
                    # memory.  Refit over the config space by measured bytes
                    # (OutOfResources is raised at launch, before any
                    # write); when NO config fits, the original error is
                    # re-raised unchanged, so a caller's own chunk ladder
                    # (and the by-name refusal in `autograd`) still apply.
                    fit = _bwd_refit_oor(grid, args, kw, dev, sig)
                    if fit is None:
                        raise
                    _siso_bwd_scan_kernel.fn[grid](*args, **kw, **fit)
            else:
                _siso_bwd_scan_kernel.fn[grid](*args, **kw, **fit)
        if pfx_out:
            # E = map terms + sum_{j < s} Vj + tail; the exclusive prefix
            # is a cumsum of the SHIFTED vector (only additions)
            vj = dDelta
            excl = torch.nn.functional.pad(vj[..., :-1], (1, 0)).cumsum(-1)
            a_h = -torch.exp(a_log.float()).reshape(1, H, 1, 1)
            dDelta = a_h * (e_map + excl + tail_buf[..., None])

    if dX is None:
        # [B, H, L, G] -> the family's [B, H, G, 1, L].  A pure VIEW under
        # `SSD_MIMO_SISO_XTM=view`; otherwise ONE transpose copy -- the
        # honest cost of the layout, paid once per pass and measured.
        dX = dXt.transpose(2, 3).unsqueeze(3)
        if xtm_mode != "view":
            dX = dX.contiguous()

    # ── the ONE pointwise rotation adjoint ────────────────────────────
    NS = min(N, int(_env("SISO_ADJ_NS", "64")))
    if rot_ka < N // 2:
        NS = min(NS, 2 * rot_ka)
    NB = N // NS
    dB = torch.empty(Bsz, nc, T, N, device=dev, dtype=f32)
    dC = torch.empty(Bsz, nc, T, N, device=dev, dtype=f32)
    th_c = theta
    per_head = th_c.dim() == 5
    # NEEDS: the ctx-recorded request (the saved `theta` is a rebound
    # tensor whose own requires_grad is not the caller's)
    need_dtheta = bool(nd["dtheta"])
    dTheta = (torch.empty_like(th_c, dtype=f32) if need_dtheta
              else torch.empty(1, device=dev, dtype=f32))
    if need_dtheta and not per_head:
        dTheta.zero_()
    st = th_c.stride()
    th_st = ([st[0], st[1], st[2], st[3], st[4]] if per_head
             else [st[0], st[1], st[2], 0, st[3]])
    dth_st = th_st
    # DTH_ROT: the scan already produced dtheta (fp32, rotated frame);
    # the pointwise adjoint skips its own contraction
    dth_from_scan = (dth_rot and not g90_served) or g90_dth
    need_dth_adj = need_dtheta and not dth_from_scan
    thh_c = theta_h if theta_h is not None else torch.empty(
        1, device=dev, dtype=f32)
    rb_c = rot_bias if rot_bias is not None else torch.empty(
        1, device=dev, dtype=f32)
    beta_c = (beta.float().contiguous().reshape(H, N) if beta is not None
              else torch.empty(1, device=dev, dtype=f32))
    gam_c = (gamma.float().contiguous().reshape(H, N) if gamma is not None
             else torch.empty(1, device=dev, dtype=f32))
    dThH = (torch.zeros(H, N // 2, device=dev, dtype=f32)
            if theta_h is not None else torch.empty(1, device=dev,
                                                    dtype=f32))
    dRB = (torch.zeros(H, N // 2, device=dev, dtype=f32)
           if rot_bias is not None else torch.empty(1, device=dev,
                                                    dtype=f32))
    dBeta = (torch.zeros(H, N, device=dev, dtype=f32) if beta is not None
             else torch.empty(1, device=dev, dtype=f32))
    dGamma = (torch.zeros(H, N, device=dev, dtype=f32) if gamma is not None
              else torch.empty(1, device=dev, dtype=f32))
    dW_p = (torch.empty(Bsz, H, NB, L, device=dev, dtype=f32) if has_dw
            else torch.empty(1, device=dev, dtype=f32))
    dG_p = (torch.empty(Bsz, H, NB, L, device=dev, dtype=f32) if has_trap
            else torch.empty(1, device=dev, dtype=f32))
    p_st = dW_p.stride() if has_dw else (dG_p.stride() if has_trap
                                         else (0, 0, 0, 0))
    # Experimental raw-frame contraction; the distinct-angle and post-bias
    # cases retain their stage-by-stage adjoints.
    raw_common = (_env("SISO_ADJ_RAW_COMMON", "0") == "1"
                  and bool(bias_pre_rot) and theta_h is None
                  and rot_bias is None)
    # ── the tuning launch runs on PRIVATE atomic buffers ──────────────
    # `dThH` / `dRB` / `dBeta` / `dGamma` are ACCUMULATED with
    # `tl.atomic_add`, so a tuner replay would add every benchmarked
    # config's contribution into them.  The one launch the tuner
    # benchmarks therefore gets zeroed scratch; the real launch follows
    # from the cache.  The five plain stores (`dB`, `dC`, `dTheta`, `dW`,
    # `dG`) are idempotent and are passed straight through.
    adj_args = (
        Btil, Ctil, dBs, dCs,
        th_c, thh_c, rb_c, beta_c, gam_c,
        aux["scale"] if has_trap else (dw if has_dw else dW_p),
        aux["gam"] if has_trap else dG_p,
        dQK,
        dB, dC, dTheta, dThH, dRB, dBeta, dGamma, dW_p, dG_p,
        *Ctil.stride()[:4],
        *Btil.stride()[:4],
        *dB.stride(),
        *m_st,
        *dth_st,
        *w_st,
        *p_st,
        H,
    )
    adj_kw = dict(
        T=T, N=N, NS=NS, ROT_KA=(rot_ka if rot_ka < N // 2 else 0),
        ROT_STATIC=(rot_ka < N // 2 and _lv.on("adj_static_real", "complex_R1O1")),
        HAS_THETA_H=theta_h is not None,
        HAS_ROT_BIAS=rot_bias is not None,
        HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None,
        BIAS_PRE_ROT=bool(bias_pre_rot), HAS_DW=has_dw,
        HAS_TRAP=has_trap,
        THETA_PER_HEAD=per_head, NEED_DTHETA=need_dth_adj,
        RAW_COMMON=raw_common,
        IDX64=idx64,
        TRAP_DTH_EXCL=False,
        NEED_DBETA=bool(nd["dbeta"]), NEED_DGAMMA=bool(nd["dgamma"]),
        NEED_DTHH=bool(nd["dthh"]), NEED_DRB=bool(nd["drb"]),
        OCC=occ_bucket(nc * Bsz, dev),
    )

    def _adj_launch(dThH_, dRB_, dBeta_, dGamma_):
        launch_args = (*adj_args[:15], dThH_, dRB_, dBeta_, dGamma_,
                       *adj_args[19:])
        _siso_rot_adjoint_kernel[(nc, Bsz)](*launch_args, **adj_kw)

    if g90_adj is not None:
        # FUSE_ADJ: the per-head RAW-frame dB / dC sit in the map buffers
        # (bf16, [B, nc, T, H, N]); reduce over heads in fp32
        dB = dBs.float().sum(3)
        dC = dCs.float().sum(3)
        dTheta = g90_adj["dTheta"]
        dW_p = g90_adj["dW_p"]
        if has_trap:
            dG_p = g90_adj["dG_p"]
        if beta is not None:
            dBeta = g90_adj["dBeta"]
        if gamma is not None:
            dGamma = g90_adj["dGamma"]
    elif _adj_tuning_launch(_autotune_cache_key(
            _siso_rot_adjoint_kernel, adj_args, adj_kw)):
        _adj_launch(torch.zeros_like(dThH), torch.zeros_like(dRB),
                    torch.zeros_like(dBeta), torch.zeros_like(dGamma))
    if g90_adj is None:
        _adj_launch(dThH, dRB, dBeta, dGamma)

    if need_dtheta and dth_from_scan:
        dTheta = (dth_buf.reshape(th_c.shape) if per_head
                  else dth_buf.sum(3))
    out = dict(dX=dX, dDelta=(dDelta if nd["ddelta"] else None), dB=dB, dC=dC,
               dTheta=dTheta if need_dtheta else None,
               dThetaH=dThH if theta_h is not None and nd["dthh"] else None,
               dRotBias=dRB if rot_bias is not None and nd["drb"] else None,
               dBeta=dBeta if beta is not None and nd["dbeta"] else None,
               dGamma=dGamma if gamma is not None and nd["dgamma"] else None,
               dX0=dX0 if x0 is not None else None,
               dDsk=dDsk_p.sum(0) if dskip_dd else None,
               dDW=None, dLam=None)
    if not (nd["ddw"] or nd["dlam"]):
        pass                     # NEEDS: the [B, H, L] write-scale chain
    elif has_dw and _env("SISO_SCALAR_FUSED", "0") == "1":
        out["dDW"], out["dLam"] = _siso_scalar_reduce(
            dW_p, dG_p if has_trap else None, dw, trap_lam)
    elif has_dw:
        dwv = dW_p.sum(2)                       # [B, H, L]
        if has_trap:
            # w_t = g_t + n_t,  g_t = lam_t dt_t,  n_t = dt_{t+1}(1-lam_{t+1})
            #   dL/dg_total = dwv + dgv          (g enters w and the diagonal)
            #   dL/dn       = dwv
            #   dlam_t      = dt_t (A_t - P_t),  ddt_t = lam_t A_t + (1-lam)P_t
            # with A = dL/dg_total and P_t = dwv_{t-1}.  Four elementwise
            # launches on [B, H, L].
            A = dwv + dG_p.sum(2)
            P = torch.nn.functional.pad(dwv[..., :-1], (1, 0))
            out["dDW"] = trap_lam * A + (1.0 - trap_lam) * P
            out["dLam"] = dw * (A - P)
        else:
            out["dDW"] = dwv
    return out
