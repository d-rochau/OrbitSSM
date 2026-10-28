"""Fused SSD-MIMO forward kernel: aggregate + state passing + chunk output.

STABILITY BY CONSTRUCTION.

    THE GUARANTEE.  For every input in the admitted range -- a =
    -exp(A_log) < 0 with |A_log| <= A_LOG_MAX (= 88, so the rate is
    finite in fp32; larger values are refused at the entry point),
    delta >= 0 finite, and x, B, C finite -- this family returns finite
    outputs, finite final states and finite gradients whenever the TRUE
    result is representable.  Concretely:
      * the fused ssd_mimo kernels contain NO exp whose argument can
        exceed 0.  Every exponent is a * (a non-negative CHUNK-LOCAL
        difference), <= 0 by the arithmetic itself -- there is no clamp;
      * there is no division and no log in the scan (the optional SwiGLU
        epilogue's only division is inside tl.sigmoid, whose denominator
        is >= 1 by construction);
      * the state recursion is a CONTRACTION -- every per-token and
        per-chunk decay factor lies in [0, 1] -- so rounding can never
        AMPLIFY, and the accuracy of the whole scan is a property of one
        chunk.
    What is NOT claimed: that states stay bounded.  The state is bounded
    by the accumulated forcing sum_s |b_s|, which grows with L whenever a
    channel does not forget (a ~ 0), and a large h_init or large inputs
    produce large states BY STRUCTURE -- that is the model, not a fault.
    The kernel is exactly LINEAR in x, so a 16-bit-range-edge input
    (|x| ~ 6e4) cannot overflow the fp32 accumulation early: measured
    linear to 1.3e-7 over |x| in [1, 1e36], and the first overflow is
    the one where the true output itself leaves fp32, reported as inf.

    Everything below is the argument for that guarantee.  Every decay
    factor this family computes is <= 1 EXACTLY, in floating point and
    not merely in exact arithmetic, and its relative error is bounded by
    a constant that does not grow with the sequence length.

    (The OTHER family, mh_scan, carries the same guarantee for its A /
    dyn / const laws -- contractive per token -- and its complex arm --
    unit modulus, with the angle carry wrapped mod 2pi per chunk so the
    angle does not grow with L either.  Its mu-split arm is the one
    place a per-step factor > 1 is allowed by design; it is bounded
    CUMULATIVELY, and not by L, by exp(|mu_gain| * range(mu)) because
    the mu driver telescopes.  See.)

    SETUP.  Per head there is one rate a = -exp(a_log) < 0 and a per-token
    step delta >= 0 (a CALLER obligation, contract 1.2; the modules meet
    it with softplus).  The host builds `delta_cumsum` CHUNK-LOCALLY --
    `delta.view(B, H, nc, T).cumsum(-1)` in autograd.py -- so the kernel
    only ever sees D_t in [0, T*delta_max].  Exponent magnitudes are
    bounded by the CHUNK length T, never by L.  There is no global cumsum
    anywhere in the fused pair.

    THE FOUR FORMS, and there are only four:
        f          = exp(a * D_t)                   chunk start -> t
        alpha      = exp(a * D_{T-1})               whole chunk
        dec_agg_t  = exp(a * (D_{T-1} - D_t))       t -> chunk end
        dec_mask_ij= exp(a * (D_i - D_j)),  i >= j  pairwise, causal
    The backward recomputes exactly these; the segmented path adds one
    more, `_seg_alpha` (below).

    NO OVERFLOW, structurally.  D is a cumsum of non-negative values, so
    as STORED it is non-negative and non-decreasing (fp addition of a
    non-negative value is monotone).  fp subtraction is monotone, so
    D_i - D_j >= 0 exactly for i >= j, and D_{T-1} - D_t >= 0 exactly.  fp
    multiplication never flips a sign, so a * (non-negative) <= 0 exactly
    (or -0.0, and exp(-0.0) = 1).  Every exponent is therefore <= 0
    EXACTLY: every factor is in [0, 1] and `exp` can only underflow.

    This is why forms 3 and 4 are written DIFFERENCE-FIRST, a*(D_i - D_j),
    and never as a difference of two rounded products, (a*D_i) - (a*D_j).
    The latter is algebraically identical and numerically not: the
    compiler may contract it into fma(a, D_i, -fl(a*D_j)), whose value on
    the diagonal is the rounding residual of fl(a*D_j) and can be
    POSITIVE, up to ULP(|a*D|)/2.  At a_log = 20 with delta = 0.5 and
    T = 64 that residual reaches +512, and exp(+512) = inf.  Clamping the
    exponent at 0 removes the inf but not the error: the same residual
    with the other rounding sign gives exp(-512) = 0 and silently DELETES
    the self-term (100% relative error on the only surviving term at
    a_log >= 20, 13% at a_log = 15).  Difference-first removes both.

    NO CLAMP.  Exponents are <= 0 by the arithmetic above, never by
    force: the kernels contain no `tl.minimum(., 0.0)`.  The two cases a
    clamp would cover are handled where they belong: (1) delta >= 0 is a
    caller obligation (contract 1.2; the modules meet it with softplus) --
    a negative delta makes the cumsum non-monotone and is a wrong INPUT,
    not a kernel fault; (2) a_log past exp's own overflow point (88.7,
    where a = -exp(a_log) is -inf and the diagonal would be (-inf)*0 =
    NaN) is outside the admitted range and is REFUSED at the entry point
    (`autograd.check_a_log_range`, bound A_LOG_MAX = 88, under the
    validate flag because the check is a host sync).

    PRECISION, and why it is independent of L.  Write u = 2^-24 ~ 6e-8
    for the fp32 unit roundoff.
      (i) The prefix carries its own rounding, and it is ABSOLUTE:
          fl(D_t) = D_t + e_t with |e_t| <= T*u*D_t -- u times the WHOLE
          prefix, not u times a pair's own difference: after a large
          decay step both D_i and D_j carry it and their difference loses
          it.  SEGPREFIX (`_SEGPREFIX_NOTE`)
          restarts the prefix at every token with c = |a| delta >=
          SEG_RESET_C (pairs across one are exactly 0 in fp32), so D'
          only accumulates the decay of ITS OWN segment.
      (ii) Given the stored prefixes, each exponent costs one subtraction
          and one multiplication, so with x = a (D'_i - D'_j) and
          c_j = |a| D'_j the segment's decay mass before the pair,
              |x_hat - x| <= (T + 2) u (|x| + c_j),
          c_j = 0 right after a restart and < SEG_RESET_C per token in
          between.  Difference-first is what keeps the |x| part RELATIVE;
          the other association (difference of rounded products) makes it
          ABSOLUTE, ~ULP(|a*D_end|), unbounded relative to a small true
          exponent.
      (iii) Hence the relative error of a decay factor is
              |e^x_hat - e^x| / e^x <= e^((T+2) u (|x| + c_j)) - 1.
          A factor that has not underflowed has |x| <= 88 (below that,
          exp(x) is 0 in fp32 and the true value is 0 too), so a factor
          with no sub-threshold decay before it in its segment (c_j = 0:
          every pair of a chunk without large steps, every pair right
          after a reset) is within e^((T+2) u 88) - 1 <= 6.9e-4 at
          T = 128, <= 1.6e-5 for the |x| <= 2 factors that dominate a
          trained model; each earlier in-segment step of decay c adds at
          most (T+2) u c.
      (iv) The diagonal is EXACT: D_i - D_i = 0 exactly, so
          dec_mask_ii = 1.0 bit-exactly at every rate.

    THE ONE QUANTITY THAT GROWS WITH L is `_seg_alpha` on the segmented
    long-L path: exp(a * sum_c D_end[c]) over a whole segment.  It is a
    PRODUCT of a < 0 with a non-negative sum, so the sign argument above
    applies unchanged -- it can only underflow -- and its absolute error
    is bounded by max_x x*e^-x*u = u/e ~ 2.2e-8, because the regime where
    the exponent is large is exactly the regime where the factor is 0.

    COROLLARY (the question this answers).  For any a_log in the
    admitted range (|a_log| <= A_LOG_MAX) and any finite delta >= 0, no
    decay factor in the fused forward or backward can be inf or NaN, and
    none can exceed 1: the transition is a CONTRACTION and rounding can
    never amplify.  Extreme rates (a_log = 20 -> |a| = 4.9e8, or
    delta = 1e3) make the off-diagonal factors underflow to exactly 0 --
    total forgetting, the correct limit -- while the self-term stays
    exactly 1.  What this does NOT say, and must not: the STATE is
    bounded by the accumulated forcing sum_s |b_s|, which legitimately
    grows with L when a channel does not forget (a ~ 0), and a large
    h_init or large inputs produce large states BY STRUCTURE.  Outputs
    overflow only when that true accumulated value leaves fp32 -- the
    range-edge cells assert linearity in x and honest overflow.  Pinned
    by the kernel tests "K2: stability by construction".

KERNEL CONTRACT:
    The FWD pass uses at most 3 kernel launches (currently 1 main kernel).
    The main kernel loops over chunks SEQUENTIALLY, computing inline per chunk:
      1. Chunk aggregate G = B_rot^T @ (x_scaled * decay)        [N, GT]
      2. State update    S = alpha * S + G                        [N, GT]
      3. Score           S0 = C_rd @ B_rot^T * decay_mask         [T, T]
      4. Within-chunk    y_intra = Score @ x_scaled               [T, GT]
      5. Cross-chunk     y_cross = f * C_rd @ S_entering          [T, GT]

    R=O=1 speed must reach approximate parity with Mamba2's
    mamba_chunk_scan_combined.  R,O>1 speed should scale roughly R*O or
    sublinearly.

Rotation modes (all angles are ADDITIVE):
    theta_h   [H, N//2]          Static per-head offset on C (readout).
                                  Applied as R(theta_h) 2x2 rotation to
                                  complex pairs of C.  When HAS_ROT is also
                                  on, theta_h is added to theta for the C
                                  rotation angle.
    rot_bias  [H, R, N//2]       Static per-(head,R) offset on B (write).
              (passed as [Hk, N//2])  ADDITIVE angle added to theta (or used
                                  alone) for the B rotation.  Kernel flag
                                  HAS_ROT_BIAS.
    theta     [B, nc, T, N//2]   Per-timestep rotation, HEAD-SHARED, or
              or                  [B, nc, T, H, N//2] PER HEAD -- the same
              [B,nc,T,H,N//2]     kernel code with
                                  the head stride s_thh set instead of 0, so
                                  the CTA (which owns exactly one head) reads
                                  its own [T, N//2] phase tile and nothing
                                  changes downstream.  Kernel flag HAS_ROT.
                                  B rotated by R(theta + rot_bias),
                                  C rotated by R(theta + theta_h).

Biases (both ADDITIVE, init zeros):
    bias_pre_rot (flag)          BIAS_PRE_ROT: beta/gamma are added to the
                                  RAW B/C tile and rotate WITH it
                                  (B_eff = R(theta_t)^T (B + beta)), which
                                  is Mamba-3's fold ORDER.  Default False =
                                  the shipped order, bias AFTER the
                                  rotations.

    beta      [H, N, R]          B_bias, added to B AFTER rotation.
              (passed as [Hk, N, R])  Kernel flag HAS_BETA.
    gamma     [H, N, O]          C_bias, added to C AFTER rotation.
              (passed as [Hk, N, O])  When HAS_THETA_H, gamma itself is
                                  rotated by R(theta_h) before adding to C.
                                  Kernel flag HAS_GAMMA.

Discretisation:
    Endpoint forcing uses c_t = dt_(t+1) * (1-lambda_(t+1)), final c=0:
    delta_wk = lambda_t * dt_t + c_t, followed by y_t -= c_t C_eff^T B_eff x_t.
    Native SISO M3 folds this into the scan; general module callers use the
    tiled Triton endpoint prologue/epilogue. Averaging clocks alone is not
    endpoint forcing. Euler uses raw dt as its write scale (not exact ZOH).
    The kernel receives RAW x plus dw [B, H, L] (HAS_DW) and applies the
    scale in its tiles — x_scaled is never materialised host-side.

LAUNCH RULE, sm_90 -- the one rule this kernel cannot
    express in its own source.  `num_warps = 8` REQUIRES a G tile of at
    least 32 columns.  At `GT == 16` Triton 3.6 emits
    `wgmma.mma_async.m64n8k8` for the `[T, GT]` output dots, and the
    score dot in the same chain then reads its B operand past the tile
    (an invalid __shared__ read of 1024 B by all eight warp leaders) --
    an illegal access at some (N, dtype, route) and a SILENT
    ~100 %-wrong `y` at the others.  `_mmav3_eight_warp_ok` enforces it,
    and an env-forced 8 warps at GT 16 RAISES rather than being
    downgraded.

Grid: (Bsz, H, NGT) -- one CTA per (batch, head, G-tile).
    Each CTA loops over all nc chunks, carrying state S[N, GT] in registers.
    State MUST stay in registers -- splitting the FWD kernel forces S to
    global memory and is a proven regression.
    B~ and C~ are loaded per chunk from global memory; shared B/C leverage
    L2 cache when multiple heads belong to the same group.

    Key structural properties (do not regress):
      - Y is stored NATIVE-LAYOUT [B, H, (O,) G, L] in the IO dtype by the
        kernel (strided view) — no host permute-copy, dY read strided.
      - x is passed RAW with dw scaled on load (HAS_DW); dX comes back as
        dx_raw and dDW in-kernel — no x_scaled materialisation.
      - S_all store is compiled out under no-grad (STORE_SALL=False).
    (There is no 3-dispatch fallback.)
"""

import functools
import os
import re
import subprocess
import warnings

import torch
import triton
import triton.language as tl
from .common import pdot  # FP32DOT operand rounding
from triton.language.extra import libdevice

from ... import families as _families
from ... import kernel_options as _kernel_options
from .autotune_policy import autotune_tier
from .common import _cos_approx, _sin_approx
from .. import portable as _portable
from .common import reraise_fatal


def _env(suffix: str, default: str) -> str:
    """Env lever with the canonical SSD_MIMO_ prefix.

    An ACTIVE `kernel_options` value for this lever wins over the
    environment (`src/orbit_ssm/kernel_options.py`): the config is
    the MODEL's channel and the environment is the SWEEP's, so a recipe
    that names a route reproduces it in a shell tuned for something
    else.  `env_override` returns None whenever nothing is configured,
    which leaves every default unchanged.  A lever that is a recorded,
    numerics- or semantics-changing OPTION (`kernel_options.option_only`) is never read from
    the environment: it answers from the option scope or `default`.  This is the single
    choke point for the whole SSD family -- `smem.py`, `autograd.py`, the SISO and
    epilogue modules all import THIS `_env`."""
    ko = _kernel_options.env_override("ssd_mimo", suffix)
    if ko is not None:
        return ko
    # NO memo dict here: `_env` runs inside torch.compile-traced code, and a global-dict
    # store under tracing makes Dynamo's own guard on that dict fail on the frame it was
    # created in.  Two string concatenations are free.
    if not _kernel_options.option_only("ssd_mimo", suffix):
        v = os.environ.get("SSD_MIMO_" + suffix)
        if v is not None:
            return v
    from .measured_plan import override as _measured_override   # no global store under tracing
    measured = _measured_override(suffix)
    return default if measured is None else measured


_ENV_KEYS: dict = {}
_MEASURED_OVERRIDE = None


# ── THE ONE AUTOTUNE SWITCH ───────────────────────────────────────────
# `SSD_MIMO_AUTOTUNE` / `kernel_options["autotune"]` -- tiers (see the
# AUTOTUNE TIERS block in fused_bwd_kernel):
#   "light" (default; alias "auto") -- every measure-based search runs;
#                    the fused backward tunes num_warps only.
#   "full"           -- as "light", backward also tunes stages/maxnreg.
#   "pinned" (alias "off") -- NO TIMED SEARCH ANYWHERE, both directions:
#       * `measuring_possible()` (smem.py) returns False, which is the
#         single gate on the chunk-size trial, the shared-memory repair
#         trial and the forward's measured config;
#       * `fwd_cfg_mode()` returns "off", so the forward takes its
#         device heuristic and neither reads nor writes the verdict
#         cache -- the same config every process, every time;
#       * the fused BACKWARD's `@triton.autotune` grids collapse to ONE
#         candidate each (triton only benchmarks when len(configs) > 1);
#       * `SISO_AUTOTUNE` defaults to "0", its own measured single pin.
# WHAT "pinned" SELECTS is documented at each grid below.
# The per-kernel levers still OVERRIDE it,
# because they are read with these as their defaults.
#
# LIMITATION: the backward's candidate
# grids are built AT IMPORT (they are `@triton.autotune` decorator
# arguments), so only the ENV spelling reaches them -- it must be set
# before `orbit_ssm.kernels` is imported.  A `kernel_options` value is
# resolved per call and therefore covers the measure-based halves only.
# The kernel_options schema documents that import-time limitation,
# including the SISO forward/backward and rotation-adjoint grids.
def autotune_off() -> bool:
    """True when the ONE switch says: no timed search, either direction."""
    # ONE key, one vocabulary.  The tier names are the contract surface
    # (`pinned` / `light` / `full`, see fused_bwd_kernel); `auto` and
    # `off` are aliases of `light` and `pinned`.
    return autotune_tier(_env("AUTOTUNE", "light")) == "pinned"


# ── The MMA OPERAND dtype ─────────────────────────────────────────────
# Every tile is loaded `.to(tl.float32)`, so by default every MMA runs at
# TF32 (`wgmma...m64n32k8.f32.tf32.tf32`), while Mamba-2 and Mamba-3 cast
# each `tl.dot` operand to the IO dtype and keep ONLY the accumulators and
# the decay/cumsum algebra in fp32.  The peak rate of bf16 MMA is 2x tf32.
#
# THREE operand modes, selected by the SAME `precision` argument:
#
#   "ieee"  fp32 operands, no tensor cores           (numerics oracle)
#   "tf32"  fp32 operands, TF32 tensor cores
#   "io"    IO-dtype operands, fp32 ACCUMULATE       (the DEFAULT)
#
# "bf16" is NOT an alias of "io": it names an IO DTYPE where the
# argument names an OPERAND MODE, and "io" under fp16 IO is not bf16.
# Passing it falls through to the env default, so the module refuses it
# by name before it reaches here.
#
# A "tf32" call stays fp32-operand: only `precision="io"` rounds the operands.
#
# Under fp32 IO the third mode is a NO-OP by construction: the operand
# dtype resolves back to fp32 and `tensor.to(tl.float32)` on an fp32
# tile is `semantic.cast`'s `src_sca_ty == dst_sca_ty` early return,
# i.e. not one IR instruction.
_MMA_IO_DT = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def _is_split_dot(precision: str) -> bool:
    """A registered scan operand mode that is a Triton split `input_precision` spelling (fp32 operands)."""
    sp = _families.SCAN_PRECISIONS.get(precision)
    return sp is not None and sp.split_dot


def mma_io_enabled(precision: str) -> bool:
    """True when the `tl.dot` operands are rounded to the IO dtype."""
    return precision == "io"


def mma_operand_dtype(precision: str, io_dtype):
    """The `tl.dtype` every `tl.dot` operand is cast to.

    `tl.float32` in the two fp32-operand modes AND at fp32/fp64 IO in
    the third one -- so `MMA_DT` is the single constexpr that decides
    whether a kernel emits `k8` tf32 or `k16` bf16 wgmma.
    """
    if mma_io_enabled(precision):
        return _MMA_IO_DT.get(io_dtype, tl.float32)
    return tl.float32


def dot_precision(precision: str) -> str:
    """The `input_precision` string the kernels get.

    `tl.dot` accepts only the Triton spellings, so the third mode rides
    "tf32" (which the backend ignores once the operands are bf16/fp16 --
    `input_precision` applies to fp32 operands alone).  Identity on
    "ieee" and "tf32".

    kernel_target="portable" (`kernels.portable`): the string stays the internal TAG of the mode ("tf32" = fp32
    operands on tensor cores, which the host-side routing and shared-memory rules key on); `common.pdot` turns the
    tag into `input_precision="ieee"` at the dot itself, so no tf32 literal reaches `tl.dot`.  The Triton split
    modes (the NVIDIA split-tf32 passes) are refused by name.
    """
    portable = _portable.active("ssd_mimo")
    if precision == "ieee":
        return "ieee"
    if _is_split_dot(precision):
        if portable:
            raise _portable.PortableRefusal(
                f"ssd_mimo_scan(precision={precision!r}) is Triton's NVIDIA split-tf32 dot; kernel_target='portable' "
                "serves precision='ieee' (fp32 operands) instead")
        return precision
    return "tf32"


_SM_COUNT_CACHE: dict = {}


def _sm_count(device) -> int:
    idx = torch.cuda.current_device() if device is None else \
        torch.device(device).index
    if idx not in _SM_COUNT_CACHE:
        _SM_COUNT_CACHE[idx] = torch.cuda.get_device_properties(
            idx if idx is not None else 0).multi_processor_count
    return _SM_COUNT_CACHE[idx]


def len_bucket(L: int) -> int:
    """AUTOTUNE-KEY sequence-length bucket: floor(log2(L / 1024)), >= 0.  The
    SISO scan runs one CTA per (batch, head) over the WHOLE sequence, so two
    cells with equal B*H (equal `occ_bucket`) but a 8x different L (long-L vs
    narrow cells) are different workloads that want different configs."""
    import math
    return max(0, int(math.floor(math.log2(max(L, 1) / 1024.0))))


def occ_bucket(n_ctas: int, device=None) -> int:
    """AUTOTUNE-KEY occupancy bucket: floor(log2(CTAs per SM)) + 4, >= 0.

    A key made only of tile/feature constexprs tunes ONCE per process, at
    the first shape it meets, and reuses that pick at every other grid.
    Every grid-sensitive autotuned kernel carries this bucket
    as a plain (unused) `OCC` argument in its key, so a pick is reused only
    within a 2x occupancy band."""
    import math
    r = n_ctas / max(_sm_count(device), 1)
    return max(0, int(math.floor(math.log2(max(r, 1e-6)))) + 4)


# ── The DEVICE, read from the device ──────────────────────────────────
# Everything the forward's launch-config search needs to prune its
# candidate grid, taken from `torch.cuda.get_device_properties` rather
# than from a compute-capability literal, so that an unmeasured GPU can
# earn a good config.
#
# `maxnreg_ok` is the ONE capability prune: `maxnreg` is an ptxas /
# NVIDIA notion and Triton's HIP backend does not accept it, so on a HIP
# build the register-cap AXIS is dropped entirely rather than passed and
# ignored.  See `_fwd_cfg_candidates`.
_DEV_CAPS: dict = {}


def _dev_caps(device) -> dict:
    """Occupancy / register / shared-memory facts about `device`.

    Keys: `sm` (SM count), `warp` (warp size -- 32 on NVIDIA, 64 on
    CDNA), `regs_per_sm`, `threads_per_sm`, `regs_per_thread` (the
    per-thread register budget at FULL occupancy, which is the number the
    warp-count rule below is measured against), `smem` (opt-in shared
    bytes per block) and `maxnreg_ok`.

    Every lookup is defaulted: a property torch does not expose on some
    backend degrades the rule to a default instead of raising.
    """
    idx = (torch.cuda.current_device() if device is None
           else torch.device(device).index)
    if idx is None:
        idx = torch.cuda.current_device()
    if idx not in _DEV_CAPS:
        props = torch.cuda.get_device_properties(idx)
        regs = int(getattr(props, "regs_per_multiprocessor", 0) or 65536)
        thr = int(getattr(props, "max_threads_per_multi_processor", 0)
                  or 2048)
        from .smem import device_smem_limit
        _DEV_CAPS[idx] = dict(
            sm=int(props.multi_processor_count),
            warp=int(getattr(props, "warp_size", 0) or 32),
            regs_per_sm=regs,
            threads_per_sm=thr,
            regs_per_thread=max(1, regs // max(1, thr)),
            smem=int(device_smem_limit(idx)),
            # HIP: Triton's AMD backend has no `maxnreg` launch field.
            maxnreg_ok=not bool(getattr(torch.version, "hip", None)),
        )
    return _DEV_CAPS[idx]


def _fwd_maxnreg() -> dict:
    """`{'maxnreg': n}` or `{}` -- the `SSD_MIMO_FWD_MAXNREG` pin alone.

    The default is the compiler's own choice.  The launch site does not
    call this: the register cap is the autotune's third axis and travels
    with the rest of the config through `_fwd_launch_cfg_full` (which
    honours the SAME env pin).  This is the one-line reader of the lever.
    """
    v = _env("FWD_MAXNREG", "")
    return {"maxnreg": int(v)} if v and int(v) else {}


def _mmav3_eight_warp_ok(gt: int, mma_io: bool = False) -> bool:
    """Can a G-tile of `gt` columns be lowered at 8 warps on sm_90?

    This is a COMPILER-LOWERING rule, not a
    preference, and it is not expressible inside the kernel: nothing a
    `tl.*` construct can say picks the layout that goes wrong.

    At 8 warps Triton 3.6 tiles a `[T, GT]` MMAv3 output over TWO
    warp-group columns (`warpsPerCTA = [4, 2]`) when `T == 64`, so the
    wgmma N tile is `GT / 2`.  At `GT == 16` that is 8, and the module
    is emitted with `wgmma.mma_async.m64n8k8` -- the one instruction
    shape that mis-compiles here.  In the same dot chain the
    `[T, N] x [N, T]` score dot carries `warpsPerCTA = [8, 1]` over a
    64-row tile, i.e. a SURPLUS warp group (8 x 16 = 128 rows of layout
    over 64 rows of data), and its B-operand shared descriptor is
    advanced past the tile: an invalid __shared__ read of 1024 B (one
    k8 x n32 fp32 B-operand tile) by all eight warp leaders.  Where that
    read lands inside the CTA's own arena the launch SURVIVES and `y`
    comes back ~100 % wrong; where it leaves the arena it is an illegal
    access.  Both need a REGISTER A operand, which is what the m3
    rotation makes of the B/C tiles -- the rotation-free route is
    memory-safe at more of the grid and wrong at the rest, so the rule
    cannot be conditioned on it.

    The predicate applies on EVERY device, not only on sm_90.  It is a
    Hopper-MMAv3 fault by origin, but it only REMOVES an 8-warp
    candidate; an untested part carries no measurement saying the wider
    tile is safe there, and the price of being conservative is one
    candidate out of six.

    `gt >= 32` is the ONLY spelling measured exact everywhere.  `T >= 128`
    also removes the n8 shape and is exact at `precision="tf32"`, but at
    the split-tf32 mode (8 warps, GT 16, T 128, N 64) it returns a 1.6 relative
    error -- the instruction mix moves with the precision, so the chunk
    width does not carry the rule and is NOT admitted.

    The rule is 128 BYTES of G tile, not 32 columns.
    Under IO-dtype (bf16/fp16) MMA operands the SAME `gt = 32` tile is
    only 64 bytes wide, and Triton 3.6 then stages it as
    `#ttg.nvmma_shared<swizzlingByteWidth = 64, elementBitWidth = 16>`
    and advances that descriptor's K steps with the 128-byte-swizzle
    stride.  At `HGMMA.64x16x16.F32.BF16 ... .tnspB` -- the
    `y_cross = dot(C_rd, S_entering)` dot, whose B operand is the
    `[N, GT]` state tile -- this is an "Invalid __shared__ read of size
    512 bytes", out of bounds of the kernel's shared arena, by all eight
    warp leaders (rotation-free route, N = 64, bf16 IO):

        operands   route   N     GT   8 warps
        bf16       plain    64   32   ILLEGAL ACCESS
        bf16       plain   128   32   ILLEGAL ACCESS
        bf16       plain    64   64   ok
        bf16       plain   128   64   ok
        bf16       m3      64/128  32/64  ok
        fp32       both    64/128  32/64  ok

    As above the rule is NOT conditioned on the route: the m3 route
    survives `gt = 32` here only because the rotation puts the A operands
    in registers, which is a property of the grid and the feature set,
    not a guarantee.  So the admitted G tile is `gt * itemsize >= 128`:
    32 columns at fp32 operands and 64 at 16-bit ones.
    """
    return gt >= (64 if mma_io else 32)


# ── The forward's launch config, MEASURED ────────────────────────────
#
# Two layers, and NO `sm_XX` literal in either:
#
#   1. a HEURISTIC (`_fwd_heuristic_cfg`) computed from the device's own
#      properties -- SM count, register file, warp size, shared-memory
#      budget -- which is what an unmeasured cell gets; and
#   2. a MEASURE-ONCE dispatcher (`_fwd_autotune`) that times a pruned
#      candidate grid with the same warm-up-then-CUDA-event method the
#      chunk `"measure"` mode uses and caches the winner per (device
#      name, cell) in the SAME on-disk file as the smem / chunk verdicts
#      (`smem.fwd_cfg_*`).  `SSD_MIMO_FWD_CFG=measure` turns it on;
#      `"auto"` (the default) spends no timed pass but DOES read a
#      verdict an earlier `measure` run left behind.
#
# CORRECTNESS-EQUIVALENCE BY CONSTRUCTION.  A speed-only search can pick a
# wrong-but-fast config, so the candidate axes are restricted to the three
# launch METADATA fields -- `num_warps`, `num_stages`, `maxnreg` -- none of
# which reaches the kernel body.  Every candidate computes the same
# function of the same inputs.  The G tile `gt` is deliberately NOT an
# axis: it sets `NGT`, i.e. the GRID and which CTA owns which output
# columns, so sweeping it would re-plan the launch rather than re-tune
# it (see `_pick_gt`).  The one
# axis that CAN be unsafe is `num_warps`, because it changes the MMA
# lowering -- and that hazard is pruned out of the candidate set before
# any timing by `_mmav3_eight_warp_ok`, never discovered by it.

#: The (warps, stages) grid.  Warps: the kernel's tiles are sized in T
#: and N, not in warps, so 4 and 8 are the two lowerings that exist here
#: (2 and 16 are different regimes, not candidates).
_FWD_WARP_CANDIDATES = (4, 8)
#: Stages: 1 (no prefetch, the smallest shared request), 2 and 3.  Deeper
#: is refused, not for speed but for residency: every extra stage is
#: another copy of the staged `[T, N]` operands in shared memory, and the
#: chunk-64 clamp at N = 128 is already close.
_FWD_STAGE_CANDIDATES = (1, 2, 3)
#: The register-cap rung, searched only around the winning (warps,
#: stages).  `0` = the compiler's own choice.  NVIDIA-only: dropped
#: wholesale on a HIP build (`_dev_caps()["maxnreg_ok"]`).
_FWD_MAXNREG_CANDIDATES = (0, 128, 168, 224)


#: The forward's LIVE PER-THREAD REGISTER ESTIMATE, in units of the
#: device's per-thread budget at full occupancy
#: (`regs_per_multiprocessor / max_threads_per_multiprocessor`), above
#: which the warp count doubles.  The estimate is
#:
#:     per_thread = N * (GT + T) / (warps * warp_size)
#:
#: -- the fp32 state accumulator `S[N, GT]` PLUS the `[T, N]` B/C
#: operand tiles, which is the pair that actually competes for the file.
#:
#: THE PRODUCT `N * GT` ALONE DOES NOT SEPARATE THE MEASURED POINTS, and
#: that is why the second term is there.  Two cells carry the SAME
#: accumulator (4096 fp32 = 32 regs/thread at 4 warps on a 32-reg
#: budget) and want OPPOSITE answers:
#:
#:   * N = 128, GT = 32 -> 8 warps (sm_90), and N = 128, GT = 64
#:     (8192 fp32 = 64 regs/thread at 4 warps) also wants 8 warps.
#:   * N = 64, GT = 64 -> 4 warps.
#:
#: The `[T, N]` term is what tells them apart: it doubles with N and is
#: FLAT in GT, so at (128, 32) the pair is 32 + 64 = 96 regs against
#: 32 + 32 = 64 at (64, 64).  In budget units on a 32-reg part that is
#: 3.00 against 2.00, and the constant sits between them.  The other
#: measured points land where they should: (128, 64) at 4.00 and
#: (64, 32) at 1.50.
#:
#: It is a PRIOR, not the answer -- `SSD_MIMO_FWD_CFG=measure` is the
#: answer.
_FWD_LIVE_REG_SHARE = 2.5

#: The chunk length the estimate assumes when the caller does not say.
#: 64 is the usual forward chunk; the real `T` is passed at both launch sites.
_FWD_EST_T_DEFAULT = 64


def _fwd_heuristic_cfg(device, n_ctas: int = 0, n_state: int = 0,
                       state_only: bool = False, gt: int = 0,
                       mma_io: bool = False, pair_half: bool = False,
                       T: int = 0) -> tuple[int, int, int]:
    """`(warps, stages, maxnreg)` for an UNMEASURED cell, from the device.

    Three clauses, each a device-agnostic rule computed from the device's
    own properties.

    * WARPS -- the REGISTER-PRESSURE clause.  The
      CTA carries the state accumulator `S[N, GT]` in fp32 REGISTERS, so
      it costs `N * GT / (warps * warp_size)` registers per thread, and
      the `[T, N]` B/C tiles cost `T * N / (warps * warp_size)` more.  The
      device's per-thread budget at full occupancy is
      `regs_per_multiprocessor / max_threads_per_multiprocessor`.
      When the pair exceeds `_FWD_LIVE_REG_SHARE` of that budget (the note
      on that constant lists the measured points that bracket it, and why
      the accumulator alone does NOT separate them) the kernel
      is register-bound rather than occupancy-bound and doubling the
      warps -- which halves the per-thread state -- is the lever.  E.g. at
      the Mamba-3 spine cell (N = 128, GT = 32) the accumulator is 32
      regs/thread at 4 warps against a 32-reg budget, and 4-warp timings
      are unstable across processes (a cliff: any co-tenant pushes it
      over) while 8-warp timings reproduce.  At N = 64, GT = 32 the
      accumulator is 16 regs, the clause does not fire, and 4 warps stays.
      Written as a budget ratio it needs no compute capability: the same
      arithmetic gives 8 warps on any part whose register file is that
      tight and 4 on a part with room.
    * `state_only` (the backward's S_all replay) is EXCLUDED: it stages
      no score / Y tile, so its register profile is the opposite one and
      8 warps is several times SLOWER.  It keeps the occupancy clause alone.
    * STAGES -- the OCCUPANCY clause at 4 warps: under one
      wave the chunk loop is latency-bound and the deeper prefetch pays
      (3 stages); on a saturated grid the extra shared-memory residency
      costs a few percent, so 1.  At 8 warps
      the binding term is shared memory instead, and the OPERAND MODE
      decides it: 16-bit MMA operands halve the staged
      tile, which is what pays for the deeper pipeline (3 stages); under
      fp32 operands the order reverses (1 stage).  An
      operand-dtype test is not a device test.
    * `pair_half`: the
      rotate-half body holds eight `[T, N/2]` tiles where the
      interleaved one holds four `[T, N]`, and at one stage ptxas
      allocates them badly -- 255 registers and 260 spill stores against
      0.  The half body also asks for far LESS shared memory (8192 B
      against 20480 B), so the third stage is free here where it is not
      for the interleaved body.  Both halves of that argument are
      properties of the BODY, not of the part, so the clause is applied
      on every device.

    `maxnreg` is 0 -- the compiler's own choice.  It is a MEASURED axis
    only (`_fwd_autotune`): capping registers is the wrong lever when the
    pressure is real (a 224/200 cap ran slower than uncapped on the
    rotate-half body), so the heuristic never guesses it.
    """
    caps = _dev_caps(device)
    lanes = max(1, caps["warp"])
    budget = caps["regs_per_thread"]
    warps = 4
    stages = 3 if (n_ctas and n_ctas <= caps["sm"]) else 1
    live_regs = (n_state * (max(int(gt), 1)
                           + (int(T) or _FWD_EST_T_DEFAULT))
                 ) / float(warps * lanes)
    if (not state_only and n_state > 0
            and live_regs > _FWD_LIVE_REG_SHARE * budget
            and _mmav3_eight_warp_ok(gt, mma_io)):
        warps = 8
        stages = 3 if mma_io else 1
    if pair_half:
        stages = max(stages, 3)
    return (warps, stages, 0)


def _fwd_fit_stages(args, kw, grid, device):
    """Lower `num_stages` until the launch FITS the device's shared budget.

    A LAUNCH PLAN, not a tuning choice -- the same construction the
    backward's Kb1 ladder ends with (`fused_bwd_kernel.py`'s
    `_kb1_forced`), and exact for the same reason: the pipeline depth
    changes no arithmetic, only how many copies of the staged operands
    live in the CTA's arena at once.

    Needed because the occupancy clause ("3 stages under one wave") is
    device-agnostic and so also runs on the ~100 KB consumer parts, where
    three stages can overflow a cell that fitted at one (e.g. sm_120 at
    N = 16, T = 32: 138,240 B against 101,376).  An ESTIMATE of the
    per-stage bytes is refused: it has to model the `[T, N]`
    operands, the `[T, GT]` Y tile, the `[T, T]` score tile (times R*O)
    and the trapezoid tile, and it gets small-N cells wrong.
    `JITFunction.warmup` gives the EXACT number and its compile
    lands in the JIT cache the real launch then hits, so the exact
    answer is also the cheap one.

    Returns `kw` unchanged when it already fits, when the requirement
    cannot be measured, or when no depth fits (the caller's
    `_check_fwd_smem` then raises `SsdSmemOverflow` and the entry point's
    chunk ladder takes over -- which is the right repair for a cell that
    does not fit at ONE stage either).
    """
    from .smem import device_smem_limit, smem_clamp_enabled
    if not smem_clamp_enabled():
        return kw
    limit = device_smem_limit(device)
    start = int(kw.get("num_stages", 1) or 1)
    if start <= 1:
        return kw
    req = _fwd_smem_required(args, kw, grid)
    if req is None or req <= limit:
        return kw
    for st in range(start - 1, 0, -1):
        k2 = dict(kw, num_stages=st)
        r = _fwd_smem_required(args, k2, grid)
        if r is not None and r <= limit:
            return k2
    return kw


def _fwd_cfg_candidates(device, n_ctas: int = 0, n_state: int = 0,
                        state_only: bool = False, gt: int = 0,
                        mma_io: bool = False, pair_half: bool = False,
                        T: int = 0):
    """`(primary, maxnreg_axis)` -- the pruned search space for this cell.

    `primary` is the `(warps, stages, 0)` grid with the heuristic's own
    answer FIRST, so a search that is cut short (a trial that raises, a
    CUDA-graph capture) still returns the shipped config.  The search is
    two-phase -- (warps, stages) first, then the register cap around the
    winner -- which keeps the trial count at <= |W|*|S| + |M| - 1 = 9
    instead of the full grid's 24.

    PRUNES, all of them from the device or from correctness, none from a
    compute-capability literal:

    * `_mmav3_eight_warp_ok(gt, mma_io)` -- CORRECTNESS.  Triton 3.6
      mis-lowers the forward's `[T, GT]` dots at 8 warps when the G tile
      is under 128 BYTES wide: a ~100 %-wrong `y` at some (N, dtype,
      route) and an illegal access at the rest.  It is a Hopper-MMAv3
      fault by origin, but it is applied on EVERY device, deliberately:
      the rule only ever REMOVES a candidate, an untested part has no
      measurement that says it is safe, and the cost of being
      conservative is one candidate.
    * the warp count must fit the device's own SM thread budget
      (`max_threads_per_multiprocessor`).
    * a register CAP must be able to house the whole CTA:
      `maxnreg * warps * warp_size <= regs_per_multiprocessor`.
    * `maxnreg` is dropped entirely where the backend has no such field
      (HIP).

    The SHARED-MEMORY budget is not pruned here: it is checked per
    candidate at measure time by dry-compiling the exact variant
    (`_fwd_smem_required`) and comparing against `device_smem_limit`,
    which is the same number Triton's launcher checks.  A candidate is
    disqualified, not clamped -- the chunk ladder is the clamp.
    """
    caps = _dev_caps(device)
    head = _fwd_heuristic_cfg(device, n_ctas, n_state, state_only, gt,
                              mma_io, pair_half, T)
    primary = [head]
    if gt > 0 and not _mmav3_eight_warp_ok(gt, mma_io):
        # ── THE NARROW-G-TILE REFUSAL ──
        # Below the 128-BYTE G-tile law (`_mmav3_eight_warp_ok`) the search
        # is refused ENTIRELY, not merely narrowed to 4 warps.
        #
        # The construction argument for searching on speed alone is that
        # `num_warps` / `num_stages` / `maxnreg` never reach the kernel
        # body.  That is true of the MATH and false of the LOWERING:
        # `num_warps` sets `warpsPerCTA`, and at a G tile under 128 bytes
        # Triton 3.6 mis-lowers this family's dot chain.  The hazard is
        # NOT confined to 8 warps: cells with GT = 8 or 16 under bf16
        # operands (a 16-32-BYTE tile) hit an illegal memory access at
        # searched configs while running fine at the heuristic's own
        # config, and `siso_bwd_kernel._prune_bwd` records the same
        # finding from the other side (it prunes
        # `(num_warps == 4, num_stages == 1)` at `gt * 2 < 128`).
        #
        # Where the lowering is known to be fragile the answer is not a
        # cleverer prune, it is NOT SEARCHING: the cell keeps the
        # heuristic config.  The wide-tile cells -- every
        # `gt * itemsize >= 128` -- are searched as designed.
        return primary, (0,)
    for w in _FWD_WARP_CANDIDATES:
        if w > 4 and not _mmav3_eight_warp_ok(gt, mma_io):
            continue
        if w * caps["warp"] > caps["threads_per_sm"]:
            continue
        for st in _FWD_STAGE_CANDIDATES:
            cfg = (w, st, 0)
            if cfg not in primary:
                primary.append(cfg)
    if not caps["maxnreg_ok"]:
        return primary, (0,)
    mnr = [m for m in _FWD_MAXNREG_CANDIDATES
           if m == 0 or m * 4 * caps["warp"] <= caps["regs_per_sm"]]
    return primary, tuple(mnr)


def _fwd_cell_key(n_ctas, n_state, T, gt, g_size, R, O, state_only,
                  mma_io, pair_half, dtype, precision) -> str:
    """The cell a measured forward config is keyed by.

    Everything that moves the kernel's register or shared-memory profile
    and nothing that does not: the grid size (the occupancy clause), the
    state width and chunk length (the tiles), the G tile and total G (the
    accumulator), R/O (the stacked score tile), the operand mode and IO
    dtype, `state_only` and `pair_half` (two different bodies).  The
    device NAME is prepended by `smem.fwd_cfg_key`.
    """
    return ("ctas%d|N%d|T%d|GT%d|G%d|R%d|O%d|%s|%s|%s|%s|%s"
            % (int(n_ctas), int(n_state), int(T), int(gt), int(g_size),
               int(R), int(O),
               "so1" if state_only else "so0",
               "io1" if mma_io else "io0",
               "ph1" if pair_half else "ph0",
               str(dtype).replace("torch.", ""), precision))


def _fwd_launch_cfg(device, n_ctas: int = 0, n_state: int = 0,
                    state_only: bool = False, gt: int = 0,
                    mma_io: bool = False,
                    pair_half: bool = False, cell: str | None = None,
                    T: int = 0) -> tuple[int, int]:
    """(num_warps, num_stages) for the fused FWD kernel -- see above.

    Order of preference, highest first:

      1. `SSD_MIMO_FWD_WARPS` / `_STAGES` -- the explicit pins, EXCEPT
         that 8 warps at a G tile under 128 bytes is REFUSED rather than
         served or quietly downgraded (the config cannot be
         lowered, and the fault it hits is silent at half the cells, so a
         warning is the wrong currency).
      2. a MEASURED verdict for this `cell` from `smem`'s cache
         (in-process or on disk), unless `SSD_MIMO_FWD_CFG=off`.
      3. `_fwd_heuristic_cfg` -- the device-agnostic default.

    `_fwd_launch_cfg_full` is the same decision with the register cap and
    the provenance attached.
    """
    w, s, _mnr, _src = _fwd_launch_cfg_full(
        device, n_ctas, n_state, state_only, gt, mma_io, pair_half, cell,
        T)
    return (w, s)


def _fwd_launch_cfg_full(device, n_ctas: int = 0, n_state: int = 0,
                         state_only: bool = False, gt: int = 0,
                         mma_io: bool = False, pair_half: bool = False,
                         cell: str | None = None, T: int = 0):
    """`(warps, stages, maxnreg, source)`; `source` names the rung."""
    from .smem import fwd_cfg_key, fwd_cfg_lookup, fwd_cfg_mode
    w = _env("FWD_WARPS", "")
    s = _env("FWD_STAGES", "")
    mnr_env = _env("FWD_MAXNREG", "")
    tw, ts, tm = _fwd_heuristic_cfg(device, n_ctas, n_state, state_only,
                                    gt, mma_io, pair_half, T)
    source = "heuristic"
    if cell is not None and fwd_cfg_mode() != "off":
        cached = fwd_cfg_lookup(fwd_cfg_key(cell, device))
        if cached is not None:
            tw, ts, tm = cached
            source = "measured"
    warps = int(w) if w else tw
    stages = int(s) if s else ts
    maxnreg = int(mnr_env) if mnr_env else tm
    if w or s or mnr_env:
        source = "env"
    if warps >= 8 and gt > 0 and not _mmav3_eight_warp_ok(gt, mma_io):
        # REFUSED, not downgraded: a silent downgrade is an answer change
        # under a lever the caller set deliberately, and the fault it covers
        # is itself silent at half the cells.  The candidate generator
        # prunes 8 warps here on every device, so reaching this point
        # means an ENV PIN (or a stale cached verdict) asked for a
        # config the generator would never have offered.
        raise ValueError(
            "SSD-MIMO forward: num_warps=%d cannot be lowered at GT=%d "
            "with %s MMA operands -- Triton 3.6 emits "
            "wgmma.m64n8k8 (fp32 operands, K20) or mis-strides the "
            "nvmma_shared<swizzlingByteWidth=64> descriptor (16-bit "
            "operands, K24) there, which returns a ~100%%-wrong y or an "
            "illegal memory access depending on N, dtype and route.  Use "
            "a G tile of at least %d (SSD_MIMO_FWD_GT) or leave "
            "SSD_MIMO_FWD_WARPS unset (legality rule: "
            "fused_fwd_kernel._mmav3_eight_warp_ok)."
            % (warps, gt, "IO-dtype" if mma_io else "fp32",
               64 if mma_io else 32))
    return (warps, stages, maxnreg, source)



#: The opt-in shared-memory budget above which `_big_smem` calls a part
#: "big": ~228 KB on the datacenter parts against ~100 KB on consumer
#: Blackwell, separated by a wide margin.
_BIG_SMEM_BYTES = 160 * 1024


def _big_smem(device) -> bool:
    """Does this part have room for N=128 to keep the 64-wide G tile?

    Read from `device_smem_limit` (the
    `shared_memory_per_block_optin` figure Triton's launcher itself
    checks, and the one `SSD_MIMO_SMEM_LIMIT` overrides) rather than from
    a compute capability, which is only a proxy for this number.
    """
    from .smem import device_smem_limit
    return device_smem_limit(device) >= _BIG_SMEM_BYTES


_I32_LIMIT = 2 ** 31


def _max_elem_offset(t) -> int:
    off = 1
    for sz, st in zip(t.shape, t.stride()):
        if sz > 1:
            off += (sz - 1) * abs(st)
    return off


def _needs_i64(*tensors) -> bool:
    """The kernels compute element offsets in int32 by default (program_id
    x stride products); a tensor spanning >= 2^31 elements mis-addresses
    that way (illegal-memory-access crash at x.numel() = 2.21e9).  When any
    operand crosses the limit the launch
    passes IDX64=True and the kernels cast the b/h grid indices to
    tl.int64, promoting every offset product that can overflow (a
    per-(b, h) slice >= 2^31 elements cannot exist on real hardware, so
    the remaining int32 products are safe).  The IDX64 variant JIT-
    compiles only when first needed — no compile cost for normal shapes.

    Pass EVERY tensor the launch addresses, INTERNALLY ALLOCATED SCRATCH
    INCLUDED: `dG_all` (B*H*nc*N*G) and the flat `dCrd_buf`
    (B*H*nc*O*T*N) can cross 2^31 while every operand stays under it.
    Both are indexed through a b/h product,
    exactly the product IDX64 exists to promote.  None and non-tensors
    are ignored so call sites can splat optional buffers."""
    return any(t is not None and torch.is_tensor(t)
               and _max_elem_offset(t) >= _I32_LIMIT for t in tensors)


def _seg_plan(nc: int, n_ctas: int, device) -> tuple[int, int]:
    """Segmented serial-chunk-loop plan (sm_90).

    The fused FWD (and Ka) loop chunks SERIALLY per CTA, so small-batch
    long-L shapes cannot fill the SMs.
    long-L shapes cannot fill the SMs (B1 x L>=32k lost 1.4x to M2).
    When the grid is under one wave and nc is long, split the loop into
    nseg segments: phase A computes per-segment final states from zero
    entry state (parallel over segments), a tiny host stitch turns them
    into true segment entry states (linearity of S = a*S + G), and
    phase C runs the normal kernel per segment seeded via the x0 path.
    Exact same per-chunk math — only the associativity regroups.
    SSD_MIMO_SEG=0 disables; SSD_MIMO_SEG_N pins nseg."""
    if _env("SEG", "1") != "1":
        return 1, nc
    if (_portable.nvidia_capability(device) != (9, 0)
            and _env("SEG_ANY_ARCH", "0") != "1"):
        # enabled on sm_90 only; other arches monolithic.
        # SSD_MIMO_SEG_ANY_ARCH=1 lifts the gate.
        return 1, nc
    sm = _sm_count(device)
    min_seg = max(1, int(_env("SEG_MIN_CHUNKS", "32")))
    # Long-nc only: at small nc the monolithic loop already wins — the
    # phase-A overhead needs a long serial loop to amortise against.
    # SEG_MIN_NC exists so tests can force segmentation at small shapes.
    min_nc = int(_env("SEG_MIN_NC", "256"))
    if n_ctas >= sm or nc < max(min_nc, 2 * min_seg):
        return 1, nc
    nseg_env = _env("SEG_N", "")
    if nseg_env:
        nseg = max(1, int(nseg_env))
    else:
        # target ~6 waves of phase-C CTAs (4-16 is flat, optimum ~8;
        # 2 is clearly short).
        nseg = min(nc // min_seg, -(-(6 * sm) // max(n_ctas, 1)), 8)
    if nseg <= 1:
        return 1, nc
    nc_seg = -(-nc // nseg)
    nseg = -(-nc // nc_seg)
    return nseg, nc_seg


@triton.jit
def _seg_stitch_kernel(
    Psi_ptr,     # [B, H, nseg, G, N]  per-segment zero-seeded finals
    Alpha_ptr,   # [B, H, nseg]        per-segment total decay
    Out_ptr,     # [B, H, nseg, G, N]  entry (fwd) / carry (reverse) states
    Init_ptr,    # [B, H, G, N]        x0 / dS_final (dummy if !HAS_INIT)
    H, NSEG, GN,
    s_pb, s_ph, s_ps, s_pg, s_pn,
    s_ab, s_ah, s_as,
    s_ib, s_ih, s_ig, s_in,
    BLK: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    """One-launch segment stitch (replaces a python loop of ~3*nseg tiny
    CUDA ops per call).
    Out and Psi share layout/strides.  Linear recurrence over segments:
      fwd:     entry[0] = init;  entry[s+1] = alpha[s] * entry[s] + psi[s]
      reverse: carry[last] = init; carry[s-1] = alpha[s] * carry[s] + psi[s]

    MASKED.  `off` runs over a BLK-wide tile of the flattened
    (g, n) axis, whose true extent is GN = G*N.  Without `off < GN` the
    stores run past this (b, h, seg) slice into the NEXT segment / head
    (and the last head past the buffer).
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    off = tl.program_id(2) * BLK + tl.arange(0, BLK)
    m = off < GN
    # flatten (g, n): strides are contiguous fp32 buffers (s_pn == 1,
    # s_pg == N), so a flat index over G*N is exact.
    base_p = Psi_ptr + b * s_pb + h * s_ph
    base_o = Out_ptr + b * s_pb + h * s_ph
    if HAS_INIT:
        acc = tl.load(Init_ptr + b * s_ib + h * s_ih + off,
                      mask=m, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros([BLK], dtype=tl.float32)
    for i in range(NSEG):
        if REVERSE:
            sidx = NSEG - 1 - i
        else:
            sidx = i
        tl.store(base_o + sidx * s_ps + off, acc, mask=m)
        a = tl.load(Alpha_ptr + b * s_ab + h * s_ah + sidx * s_as)
        v = tl.load(base_p + sidx * s_ps + off,
                    mask=m, other=0.0).to(tl.float32)
        acc = a * acc + v


# ---------------------------------------------------------------------------
# The chunk-local decay prefix D_i, IN KERNEL
# ---------------------------------------------------------------------------
# SEC.1.2 of the kernel contract defines D_i = sum_{s <= i} delta_s,
# CHUNK-LOCAL.  The host can materialise it with `delta_c.cumsum(-1)` in
# `autograd.py::_fused_ssd_impl` (one ATen scan kernel plus a [B, H, nc, T]
# fp32 buffer that every kernel then re-reads).
#
# With `DCS_INK` the SAME pointer carries the RAW chunked delta and each
# consumer forms D_i from the [T] tile it was already loading, with one
# `tl.cumsum` over the chunk axis.  The load traffic is identical (same
# tensor shape), the ATen scan and the buffer are gone.
#
# NUMERICS.  `tl.cumsum` is a parallel (Kogge-Stone) scan; ATen's is a
# sequential one, so the two agree to within a few ulp but are NOT bit
# identical.  That is why the host defaults `DCS_INK` ON only where it
# pays (R * O > 1) and leaves R = O = 1 -- and with it
# every SISO / rotating R1 path -- on the host cumsum, bit for bit.
# The file-header stability argument is UNCHANGED: D is still a prefix sum
# of non-negative values, every exponent is still formed as a difference of
# two entries OF THE SAME TILE, and `_dcs_end` returns literally the
# (T-1)-th entry of that tile (not an independently reduced total), so
# `D_end - D_i >= 0` stays structural and no clamp is introduced.


@triton.jit
def _dcs_tile(dbase, offs_t, s_dt, DCS_INK: tl.constexpr):
    """The chunk's inclusive decay prefix D_i as a [T] fp32 tile."""
    if DCS_INK:
        d_raw = tl.load(dbase + offs_t * s_dt)
        D_h = tl.cumsum(d_raw, axis=0)
    else:
        D_h = tl.load(dbase + offs_t * s_dt)
    return D_h


@triton.jit
def _dcs_end(dbase, offs_t, s_dt, D_h, T: tl.constexpr,
             DCS_INK: tl.constexpr):
    """D_{T-1}: the chunk's total decay mass, as a scalar -- the LAST ENTRY
    OF D_h, selected (not re-reduced and not re-loaded): keeps
    `D_end - D_h >= 0` exact, see the note above.  SEGPREFIX: D_h is the
    decoded segmented prefix on every route, so it is selected on the
    host-prefix route too (the stored entry may be a reset sentinel)."""
    D_end_val = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
    return D_end_val


# ── _SEGPREFIX_NOTE: the chunk prefix RESTARTS at reset tokens ─────────
# exp(a (D_i - D_j)) with D the fp32 chunk prefix carries an absolute
# exponent error ~u |a D_i| -- u times the WHOLE prefix, not u times the
# pair's own |a (D_i - D_j)|.  After a reset token (c = |A| delta >> 1)
# every later pair of the chunk is formed from two large, nearly equal
# numbers.  A token with
# c >= SEG_RESET_C makes every pair across it EXACTLY 0 in fp32 (e^-104 is
# below the smallest denormal; ex2.approx flushes long before), so the
# prefix restarts there: D' is the SEGMENTED prefix (the reset token's own
# delta is dropped, D'_reset = 0), seg the segment index, and
#   pair (j, i): same segment ? exp(a (D'_i - D'_j)) : 0
#   f_i       : seg_i == 0     ? exp(a D'_i)          : 0
#   dec_agg_j : seg_j == seg_end ? exp(a (D'_end - D'_j)) : 0
#   alpha     : seg_end == 0   ? exp(a D'_end)        : 0
# -- the same numbers in exact arithmetic, with every exponent now formed
# from the prefix of its OWN segment.  In-kernel (DCS_INK) the restart is
# detected from the raw delta; the host-prefix route ships D' with the
# reset tokens marked by the sentinel -1 (D' >= 0 otherwise).  The
# spanning-pair dL/ddelta (`_SPANDELTA_NOTE`) needs nothing more: every
# pair and every carry across a reset is an exact zero.
SEG_RESET_C = tl.constexpr(104.0)
_SEG_RESET_C_HOST = 104.0


def seg_prefix_host(delta_raw, a_log):
    """The host-prefix route's SEGMENTED chunk prefix (`_SEGPREFIX_NOTE`).

    `delta_raw` [B, H, (G,) nc, T] chunked delta, `a_log` [H].  Returns fp32
    D' with the reset tokens (|A| delta >= SEG_RESET_C) carrying the
    sentinel -1 and every other token the prefix of its own segment (fp64
    accumulation, one rounding).  No autograd: the backward returns
    dL/ddelta itself (`_SPANDELTA_NOTE`)."""
    # default ON (one fused launch instead of ~10 ATen fp64 ops: the M2
    # forward's host + long-L non-scan GPU work); it serves exactly the
    # shapes `seg_prefix_fused` accepts, everything else keeps the host route
    if (delta_raw.is_cuda and _env("SEG_PREFIX_FUSED", "1") == "1"
            and delta_raw.ndim in (4, 5) and a_log.ndim == 1
            and a_log.numel() == delta_raw.shape[1]
            and a_log.device == delta_raw.device
            and min(delta_raw.shape) > 0):
        return seg_prefix_fused(delta_raw, a_log)
    with torch.no_grad():
        shape = [1] * delta_raw.dim()
        shape[1] = -1
        c = delta_raw.double() * torch.exp(a_log.double()).view(shape)
        rst = c >= _SEG_RESET_C_HOST
        cs = torch.where(rst, 0.0, delta_raw.double()).cumsum(-1)
        T = delta_raw.shape[-1]
        pos = torch.arange(T, device=delta_raw.device).expand_as(rst)
        last = torch.where(rst, pos, -1).cummax(-1).values
        base = torch.where(last >= 0,
                           cs.gather(-1, last.clamp(min=0)), 0.0)
        out = (cs - base).float()
        return torch.where(rst, -1.0, out)


@triton.jit
def _segadd(v1, f1, v2, f2):
    """Segmented-sum combine: a set flag starts a new segment."""
    return tl.where(f2 != 0, v2, v1 + v2), f1 | f2


@triton.jit
def _seg_prefix_kernel(D, A, Out, SHAPE: tl.constexpr, STRIDES: tl.constexpr,
                       A_STRIDE: tl.constexpr, ROWS_PER_HEAD: tl.constexpr,
                       ROWS: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK: tl.constexpr):
    row = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    index = row
    base = 0
    for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
        base += (index % SHAPE[axis]) * STRIDES[axis]
        index //= SHAPE[axis]
    t = tl.arange(0, BLOCK)
    valid = (row[:, None] < ROWS) & (t[None, :] < SHAPE[-1])
    d = tl.load(D + base[:, None] + t[None, :] * STRIDES[-1], valid, 0).to(tl.float64)
    head = (row // ROWS_PER_HEAD) % SHAPE[1]
    a = tl.load(A + head * A_STRIDE, row < ROWS, 0).to(tl.float64)
    reset = valid & (d * libdevice.exp(a)[:, None] >= 104.0)
    # Direct segmented accumulation avoids both the int64 cummax/gather
    # chain and subtraction of two rounded prefixes after a reset.
    value, _ = tl.associative_scan((tl.where(reset, 0.0, d), reset.to(tl.int32)),
                                   1, _segadd)
    tl.store(Out + row[:, None] * SHAPE[-1] + t[None, :],
             tl.where(reset, -1.0, value).to(tl.float32), valid)


def seg_prefix_fused(delta_raw, a_log, *, row_tile=None):
    """Opt-in fused fp64 segmented prefix, with the host route's reset sentinel."""
    import math
    if delta_raw.ndim not in (4, 5) or a_log.ndim != 1 or a_log.numel() != delta_raw.shape[1]:
        raise ValueError("segmented prefix requires delta[B,H,(G,)nc,T] and a_log[H]")
    if not delta_raw.is_cuda or a_log.device != delta_raw.device:
        raise ValueError("segmented prefix requires one CUDA device")
    if min(delta_raw.shape) <= 0:
        raise ValueError("segmented prefix requires nonempty dimensions")
    row_tile = int(_env("SEG_PREFIX_ROWS", "16")) if row_tile is None else row_tile
    if row_tile not in (1, 4, 16, 32):
        raise ValueError("segmented prefix row_tile must be 1, 4, 16 or 32")
    with torch.no_grad():
        out = torch.empty(delta_raw.shape, device=delta_raw.device, dtype=torch.float32)
        rows = delta_raw.numel() // delta_raw.shape[-1]
        _seg_prefix_kernel[(triton.cdiv(rows, row_tile),)](
            delta_raw, a_log, out, tuple(delta_raw.shape), tuple(delta_raw.stride()),
            a_log.stride(0), math.prod(delta_raw.shape[2:-1]),
            ROWS=rows, BLOCK_R=row_tile,
            BLOCK=triton.next_power_of_2(delta_raw.shape[-1]), num_warps=4)
        return out


@triton.jit
def _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK: tl.constexpr):
    """(D'_i, seg_i): the chunk's SEGMENTED decay prefix and segment index
    (`_SEGPREFIX_NOTE`)."""
    d = tl.load(dbase + offs_t * s_dt)
    if DCS_INK:
        rst = (a_val * d) <= -SEG_RESET_C
        v = tl.where(rst, 0.0, d)
        ri = rst.to(tl.int32)
        D_h, _ = tl.associative_scan((v, ri), 0, _segadd)
    else:
        rst = d < 0.0
        ri = rst.to(tl.int32)
        D_h = tl.where(rst, 0.0, d)
    seg = tl.cumsum(ri, axis=0)
    return D_h, seg


@triton.jit
def _same_seg(seg):
    """[T, T] mask: tokens i and j in the same segment (`_SEGPREFIX_NOTE`)."""
    return seg[:, None] == seg[None, :]


def _seg_alpha(a_log, delta_cumsum, nseg, nc_seg, dcs_ink=False):
    """Per-segment total decay exp(a * sum(D_end)) — [B, H, nseg] fp32.

    `dcs_ink`: `delta_cumsum` carries the RAW chunked delta (lever 1), so
    the chunk total is its row SUM instead of its last entry.
    """
    Bsz, H, nc, _ = delta_cumsum.shape
    a_val = -torch.exp(a_log.float())
    if dcs_ink:
        d_end = delta_cumsum.float().sum(-1)
    else:
        # SEGPREFIX: the host prefix restarts at reset tokens (sentinel -1);
        # a chunk that holds one has alpha = 0 exactly
        dc = delta_cumsum.float()
        d_end = torch.where((dc < 0).any(-1), torch.full_like(dc[..., -1],
                            float("inf")), dc[..., -1])
    n_pad = nseg * nc_seg - nc
    if n_pad:
        d_end = torch.nn.functional.pad(d_end, (0, n_pad))
    seg_sum = d_end.view(Bsz, H, nseg, nc_seg).sum(-1)
    return torch.exp(a_val.view(1, -1, 1) * seg_sum)


def _seg_stitch(psi, alpha_seg, init, reverse):
    """Launch the stitch kernel; returns the entry/carry buffer."""
    Bsz, H, nseg, G, N = psi.shape
    out = torch.empty_like(psi)
    dummy = psi  # unused when !HAS_INIT
    has_init = init is not None
    ini = init if has_init else dummy
    ini_st = ((ini.stride(0), ini.stride(1), ini.stride(2), ini.stride(3))
              if has_init else (0, 0, 0, 0))
    blk = min(1024, triton.next_power_of_2(G * N))
    grid = (Bsz, H, (G * N + blk - 1) // blk)
    _seg_stitch_kernel[grid](
        psi, alpha_seg, out, ini,
        H, nseg, G * N,
        *psi.stride(),
        *alpha_seg.stride(),
        *ini_st,
        BLK=blk,
        REVERSE=reverse,
        HAS_INIT=has_init,
        num_warps=4,
    )
    return out


def _pick_gt(g_size: int, cap: int = 64) -> int:
    """Largest tile in {cap, 32, 16} dividing G — supports any G = 16*n
    (odd multiples fall back to 16-wide tiles; tl.dot needs >= 16)."""
    for gt in (cap, 32, 16):
        if gt <= cap and g_size % gt == 0:
            return gt
    raise ValueError(
        f"G={g_size} is not a multiple of 16 (tensor-core tile envelope)")



# ---------------------------------------------------------------------------
# The xor-lane partner WITHOUT a second global load
# ---------------------------------------------------------------------------
# Every pair rotation needs, next to the tile at `offs_n`, the tile at
# `offs_n ^ 1` — the partner coordinate of each adjacent (2k, 2k+1) pair.
# A second `tl.load` at the permuted index has the same address SET as the
# plain load, but `offs_n ^ 1` is not affine in the lane id, so Triton
# cannot vectorise it: the plain tile comes in as wide vector loads and the
# partner as one narrow load per element (~2x the cost of the register
# form, which is as cheap as having no partner at all).
#
# The register form is a pure layout op: view the tile as [.., N//2, 2],
# split the pair axis, join it back the other way round.  `tl.split` /
# `tl.join` on a trailing axis of exactly 2 are the interleave primitives,
# so no shared-memory round trip is needed.  Value-identical to the load
# by construction: result[.., 2k] = tile[.., 2k+1] and vice versa.
@triton.jit
def _pair_swap_2d(tile, T: tl.constexpr, N: tl.constexpr):
    """[T, N] -> the same tile with each adjacent (2k, 2k+1) pair swapped."""
    even, odd = tl.split(tl.reshape(tile, (T, N // 2, 2)))
    return tl.reshape(tl.join(odd, even), (T, N))


@triton.jit
def _pair_swap_1d(vec, N: tl.constexpr):
    """[N] -> the same vector with each adjacent (2k, 2k+1) pair swapped."""
    even, odd = tl.split(tl.reshape(vec, (N // 2, 2)))
    return tl.reshape(tl.join(odd, even), (N,))


# The PAIR-AXIS BROADCAST.  A per-pair quantity indexed
# `pair_idx = offs_n // 2` reads correctly but loads TERRIBLY: like the
# xor-lane partner, `offs_n // 2` is not affine in the lane id,
# so Triton cannot vectorise the access and the [T, N] tile arrives one
# element at a time -- for a tensor that only has T x N/2 distinct values.
# Loading it on its own PAIR axis (affine `tl.arange(0, N // 2)`, one wide
# vector load) and widening it HERE is the same values at half the traffic
# and a quarter of the trig, since cos/sin then run on [T, N//2] too.
# `tl.join(v, v)` puts a copy of each element next to itself, which IS the
# pair broadcast: result[.., 2k] = result[.., 2k+1] = v[.., k].
@triton.jit
def _pair_bcast_2d(v, T: tl.constexpr, N: tl.constexpr):
    """[T, N//2] -> [T, N], each value repeated on both lanes of its pair."""
    return tl.reshape(tl.join(v, v), (T, N))


# ── The same three primitives for the rotate-half layout, where pair k
# lives at (k, k + N/2).
#
# They let a kernel keep the INTERLEAVED body's MMA shapes under `half`
# (run the interleaved body and change only the pair PARTNER), and they
# are CORRECT, but SLOWER than the split-tile bodies.
#
# WHY.  The interleaved swap / broadcast ride `tl.split` / `tl.join` on
# a TRAILING axis of exactly 2, which is a register-level interleave.
# Under `half` the pair axis is the SLOW half of the state axis, so the
# [T, 2, N/2] view has the lane index FIRST and the same primitives are
# one `tl.permute` away -- a real shared-memory transpose, per tile,
# per (o, r).  The full-width route therefore needs the partner as a
# LOAD (`offs_n +/- N/2`, coalesced) plus the angle fold
# R(nu)^T R(Theta)^T = R(Theta + nu)^T so that no ROTATED tile is ever
# swapped, not these primitives.
@triton.jit
def _half_swap_2d(tile, T: tl.constexpr, N: tl.constexpr):
    """[T, N] -> the same tile with the two contiguous halves swapped."""
    h1, h2 = tl.split(tl.permute(tl.reshape(tile, (T, 2, N // 2)),
                                 (0, 2, 1)))
    return tl.reshape(tl.permute(tl.join(h2, h1), (0, 2, 1)), (T, N))


@triton.jit
def _half_swap_1d(vec, N: tl.constexpr):
    """[N] -> the same vector with the two contiguous halves swapped."""
    h1, h2 = tl.split(tl.permute(tl.reshape(vec, (2, N // 2)), (1, 0)))
    return tl.reshape(tl.permute(tl.join(h2, h1), (1, 0)), (N,))


@triton.jit
def _half_bcast_2d(v, T: tl.constexpr, N: tl.constexpr):
    """[T, N//2] -> [T, N], each value repeated on BOTH halves."""
    return tl.reshape(tl.permute(tl.join(v, v), (0, 2, 1)), (T, N))


@triton.jit
def _lay_swap_2d(tile, T: tl.constexpr, N: tl.constexpr,
                 PAIR_HALF: tl.constexpr):
    if PAIR_HALF:
        return _half_swap_2d(tile, T, N)
    else:
        return _pair_swap_2d(tile, T, N)


@triton.jit
def _lay_swap_1d(vec, N: tl.constexpr, PAIR_HALF: tl.constexpr):
    if PAIR_HALF:
        return _half_swap_1d(vec, N)
    else:
        return _pair_swap_1d(vec, N)


@triton.jit
def _lay_bcast_2d(v, T: tl.constexpr, N: tl.constexpr,
                  PAIR_HALF: tl.constexpr):
    if PAIR_HALF:
        return _half_bcast_2d(v, T, N)
    else:
        return _pair_bcast_2d(v, T, N)


@triton.jit
def _rot_apply_2d(V, ph_cos, ph_sin, rot_sign,
                  T: tl.constexpr, N: tl.constexpr):
    """R(theta_t)^T V built from the PAIR form at the use site.

    The expression is character for character the one the hoisted
    `cs_t` / `sn_t` compute -- `tl.join(v, v)` is a pure layout op, so
    this is code motion and nothing else and the result is bitwise the
    same tile.  What changes is that the two [T, N] fp32 broadcasts
    (64 registers per thread at 4 warps, N = 128) stop being live across
    everything between the phase block and the R loop; only the two
    [T, N//2] halves are.  The same lever inside
    `_bwd_kbc_hb_kernel`'s head loop is `ROT_DIET`.
    """
    cs = _pair_bcast_2d(ph_cos, T, N)
    sn = rot_sign[None, :] * _pair_bcast_2d(ph_sin, T, N)
    return cs * V + sn * _pair_swap_2d(V, T, N)


# The shipped mask.
#
# IT IS REFUSED UNDER THE IO-DTYPE OPERAND MODE, and that is a
# CORRECTNESS refusal, not a speed one.  With `MMA_DT = tl.bfloat16`,
# bit 1 (`YC`) makes the `state_only=True` forward the backward launches
# to rebuild `S_all` die with an illegal memory access (`FWD_SCHED=0`
# passes, `FWD_SCHED=1` faults, at otherwise identical settings).  That
# is the 128-byte-G-tile family (`_mmav3_eight_warp_ok`) -- MMAv3
# lowering that is wrong or faulting at some (warps, tile bytes)
# intersection under 16-bit operands -- and moving a `tl.dot` across a
# loop boundary is exactly the kind of edit that lands on it.  The root
# cause is not established, so the intersection is refused.
#
# The fp32-operand mode is measured clean.
FWD_SCHED_DEFAULT = 3


def fwd_sched(mma_io: bool = False) -> int:
    """`SSD_MIMO_FWD_SCHED` -- the forward's LIVE-RANGE schedule mask.

    Applied to `_fused_chunk_fwd_kernel`.  Every bit
    only moves an expression earlier or later in the chunk body, or
    REBUILDS a value at its use instead of holding it across the R loop.
    No arithmetic expression, no operand dtype, no reduction order and
    no `tl.dot` changes, so the fp64 oracle must read the same number to
    the last digit -- and the plain route is BITWISE identical.

      1  `YC`  the cross-chunk dot `f * (C_rd @ S_entering)` is taken
               AFTER the R loop instead of before it, so its [T, GT]
               fp32 result stops living across both MMAs.  Its two
               operands are live to that point anyway (`C_rd` feeds
               every `S0`, `S_entering` feeds `S = alpha S + G`).
      2  `RD`  the phase stays in its [T, N//2] PAIR form and the
               [T, N] cos/sin tiles are rebuilt at each use
               (`_rot_apply_2d`).  Halves what a rotation costs in live
               registers; a no-op without `HAS_ROT`.
      4  `DM`  the [T, T] decay mask is built at its consumer (`Score =
               S0 * dec_mask`) instead of at the top of the chunk body.
               The `dw` fold rides with it.

    Default `FWD_SCHED_DEFAULT`; every arm of the mask stays reachable
    through the variable.
    """
    env = _env("FWD_SCHED", "")
    if env != "":
        return int(env)
    return 0 if mma_io else FWD_SCHED_DEFAULT


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

# ── _TRAPFOLD_NOTE: the two-tap trapezoid without a pair ────────────────
# Mamba-3's write g_t = a_t g_{t-1} + a_t(1-lam_t)dt_t v_{t-1}k_{t-1}^T
# + lam_t dt_t v_t k_t^T is collapsed onto the one-tap scan with
# w_t = own_t + n_t (own_t = lam_t dt_t, n_t = dt_{t+1}(1-lam_{t+1})):
# n_t is the part of token t's write that the NEXT token owns.  Weighting
# the Score diagonal with w_t and taking n_t (k_t.q_t) v_t back out
# (S0_tt (MMA, rounded operands) * w_t minus n_t * trap_dot_t (fp32 row
# sum)) is a pair that cancels only in exact arithmetic and leaves
# u * n_t |k||q||v| where the truth is O(own_t) (n_t / own_t can reach 2e6).
# So the diagonal of `dec_mask` IS own_t (decay exp(0) = 1) and nothing
# is subtracted; off the diagonal and in the chunk-end carry `dec_agg`
# the weight stays w_t (the n_t part is a legitimate write that later
# tokens read).  `trap_dot` is still stored (STORE_DOT) as the backward's
# input: Kc's strict score tile + (own_t d_t) dY_t, and `trap_adjoint`
# carries the diagonal exactly (dd_t = own_t s_t, dL/dn_t = A_t,
# dL/down_t = A_t + d_t s_t).  Kb1 / Kbc-hb / Gluon Kb1 drop their score
# diagonal under the trapezoid (TRAP_DIAG).


@triton.jit
def _fused_chunk_fwd_kernel(
    # ── Input pointers ────────────────────────────────────────────────
    C_ptr,           # [B, nc, (Gc,) T, N]  C-tilde
    B_ptr,           # [B, nc, (Gb,) T, N]  B-tilde
    X_ptr,           # [B, H, G_SIZE, R, L_padded]  x_scaled = x * delta_w
    Delta_ptr,       # [B, H, nc, T]  delta_cumsum (inclusive)
    A_ptr,           # [H]  a_log (log decay rate, > 0)
    ThetaH_ptr,      # [H, N//2]  per-head rotation (dummy if !HAS_THETA_H)
    Theta_ptr,       # [B, nc, T, (H,) N//2] per-timestep rotation
                     # (dummy if !HAS_ROT; s_thh = 0 <=> head-shared)
    X0_ptr,          # [B, H, G_SIZE, N]  initial state (dummy if !HAS_X0)
    Beta_ptr,        # [H, N, R]  per-head write bias (dummy if !HAS_BETA)
    Gamma_ptr,       # [H, N, O]  per-head read bias (dummy if !HAS_GAMMA)
    RotBias_ptr,     # [H, N//2]  per-head write rotation bias (dummy if !HAS_ROT_BIAS)
    # ── Output pointers ───────────────────────────────────────────────
    Y_ptr,           # [B, H, nc, T, O, G_SIZE]  output (fp32)
    SAll_ptr,        # [B, H, nc, N, G_SIZE]  state entering each chunk
    SF_ptr,          # [B, H, G_SIZE, N]  final state
    # ── Scalar strides ────────────────────────────────────────────────
    H,
    s_cb, s_cc, s_ct, s_cn,        # C strides
    s_bb, s_bc, s_bt, s_bn,        # B strides
    s_xb, s_xh, s_xg, s_xr, s_xl, # X strides
    s_db, s_dh, s_dc, s_dt,        # delta_cumsum strides
    s_yb, s_yh, s_yc, s_yt, s_yg,  # Y strides (b, h, chunk, t, g)
    s_yo,                           # Y O-stride
    s_sb, s_sh, s_sc, s_sn, s_sg,  # SAll strides
    s_fb, s_fh, s_fg, s_fn,        # SF strides
    s_x0b, s_x0h, s_x0g, s_x0n,   # X0 strides
    s_thb, s_thc, s_tht, s_thh, s_thk,  # Theta strides (0 when !HAS_ROT;
                                   # s_thh = 0 <=> head-SHARED phase)
    s_bh,                           # B group stride (0 = shared)
    s_ch,                           # C group stride (0 = shared)
    s_br,                           # B R-stride (0 when R=1 or shared B)
    s_co,                           # C O-stride (0 when O=1 or shared C)
    # ── Dimensions (NC is runtime to avoid 32x unroll) ────────────────
    NC,
    NC_SEG,          # chunks per segment (== NC when unsegmented)
    s_x0s,           # X0 SEGMENT stride (phase C entry states; 0 = plain x0)
    s_fss,           # SF SEGMENT stride (>0: store final state PER SEGMENT)
    # ── Constexprs ────────────────────────────────────────────────────
    T: tl.constexpr,
    N: tl.constexpr,
    G_SIZE: tl.constexpr,
    GT: tl.constexpr,
    PREC: tl.constexpr,
    # The dtype every `tl.dot` operand is cast to.  tl.float32 (the
    # default) is `semantic.cast`'s identity on the fp32 tiles below;
    # tl.bfloat16 / tl.float16 make every MMA an IO-dtype one with fp32
    # accumulate.  See `mma_operand_dtype`.
    MMA_DT: tl.constexpr = tl.float32,
    HAS_THETA_H: tl.constexpr = False,
    HAS_ROT: tl.constexpr = False,
    HAS_X0: tl.constexpr = False,
    HPG_B: tl.constexpr = 1,
    HPG_C: tl.constexpr = 1,
    R: tl.constexpr = 1,
    O: tl.constexpr = 1,
    HAS_BETA: tl.constexpr = False,
    HAS_GAMMA: tl.constexpr = False,
    HAS_ROT_BIAS: tl.constexpr = False,
    # ── Bias fold ORDER ───────────────────────────────────────────────
    # False (default): beta / gamma are added AFTER the
    #   rotations -- B_eff = R(rho_h) R(theta_t)^T B + beta.
    # True  (Mamba-3's order, and mh's): they are added to the RAW tile
    #   and rotate WITH it -- B_eff = R(theta_t)^T (B + beta),
    #   C_eff = R(theta_t)^T (C + gamma).  This is a different law, not
    #   a tolerance; it is what makes Mamba-3's Q_bias / K_bias exactly
    #   representable.  Served with HAS_ROT and refused with theta_h /
    #   rot_bias at the entry point.
    BIAS_PRE_ROT: tl.constexpr = False,
    # ── Rotation operand fold ─────────────────────────────────────────
    # True: build each pair rotation's xor-lane partner from the tile
    # ALREADY IN REGISTERS (`_pair_swap_2d`) instead of a second,
    # un-vectorisable `tl.load` at `offs_n ^ 1`.  Value-identical; the
    # launcher sets it on the R == O == 1 rotation branch.
    PAIR_SWAP: tl.constexpr = False,
    # ── The ROTATE-HALF pair layout ───────────────────────────────────
    # False (default, `pair_layout="interleaved"`): rotation pair k is
    # the adjacent lane pair (2k, 2k+1) and its partner is the cross-lane
    # `_pair_swap_2d`.  True (`pair_layout="half"`, Mamba-3's GPT-NeoX
    # convention): pair k is (k, k + N/2), the two halves are contiguous
    # [T, N/2] tiles and the partner is the other tile -- no lane
    # permutation anywhere.  The two layouts are a fixed PERMUTATION of
    # the caller's state axis, so they are different parameterisations of
    # the same model, not different models; see
    PAIR_HALF: tl.constexpr = False,
    # ── Decoupled write scale (HAS_DW): X_ptr holds RAW x; each x tile is
    # scaled by dw[t] = delta_w on load.  Kills the host-side
    # x_scaled = x * delta_w materialisation (and its backward chain).
    DW_ptr=None,     # [B, H, L_padded] fp32 write scale
    s_wb=0, s_wh=0, s_wl=0,
    HAS_DW: tl.constexpr = False,
    STORE_SALL: tl.constexpr = True,  # False in inference: skip the 67MB
                                      # per-chunk state save (bwd only)
    STATE_ONLY: tl.constexpr = False,  # recompute mode: aggregates + state
                                       # scan + SAll store ONLY (no C loads,
                                       # no score, no Y) — launched from the
                                       # BACKWARD to rebuild S_all instead
                                       # of saving it (M2-style memory)
    IDX64: tl.constexpr = False,  # 64-bit b/h offset products (tensors
                                  # spanning >= 2^31 elements); the False
                                  # specialization compiles identical code
    # ── The LIVE-RANGE schedule mask (`SSD_MIMO_FWD_SCHED`) ──────────
    SCHED: tl.constexpr = 0,
    # ── Fused epilogue at the STORE ───────────────────────────────────
    # Contract: "the kernel computes x -> SSM(x_main).silu(
    # x_gate) end-to-end with fused forward and backward, saving only
    # (y_pre, z)", and "D-skip connection (optional, fused)".  Both happen
    # on the output tile while it is still in registers:
    #     y_pre = scan + sum_r D[h, g, r] * x_raw[.., r, t]
    #     out   = y_pre * silu(z)
    # Y_ptr keeps y_pre (the backward's operand -- "saving only
    # (y_pre, z)"); Yg_ptr takes the gated output.  STORE_Y=False drops
    # the y_pre store entirely when no backward will run.
    # HAS_DSKIP requires HAS_DW: X_ptr then holds RAW x, which is what the
    # skip adds; without it X_ptr is x_scaled and the skip would be the
    # WRONG tensor rather than merely slower, so it is a static assert.
    Yg_ptr=None,     # [B, H, nc, T, O, G_SIZE] gated output
    Z_ptr=None,      # [B, L_padded, H*G_SIZE] gate pre-activation
    Dsk_ptr=None,    # [H, G_SIZE, R] fp32 D-skip weights
    s_zb=0, s_zh=0, s_zt=0, s_zg=0,
    s_gb=0, s_gh=0, s_gc=0, s_gt=0, s_gg=0, s_go=0,
    HAS_GATE: tl.constexpr = False,
    HAS_DSKIP: tl.constexpr = False,
    STORE_Y: tl.constexpr = True,
    # ── Mamba-3's two-tap trapezoid, in-kernel ────────────────────────
    # Mamba-3's recurrence writes the PREVIOUS token's outer product too:
    #   S_t = e^{ADT} S_{t-1} + e^{ADT} dt_t (1-l_t) v_{t-1} k_{t-1}^T
    #                         + l_t dt_t v_t k_t^T
    # Substituting g_t = S_t + dt_(t+1)(1-l_(t+1)) v_t k_t^T collapses it onto this
    # kernel's ONE-tap write with
    #   w_t   = l_t dt_t + dt_{t+1} (1 - l_{t+1})
    # and leaves a LOCAL rank-1 output correction
    #   y_t  -= dt_{t+1}(1 - l_{t+1}) (k_t . q_t) v_t .
    # Here:
    #   * `w_t` is two extra [T] loads (dt and lambda at t+1, masked at
    #     the sequence end, which is what makes the last tap zero);
    #   * `k_t . q_t` is one row-reduction of the B/C tiles the chunk
    #     ALREADY holds in registers, after the bias fold and the
    #     rotation -- and the rotation cancels because both sit at the
    #     same t, which is why it needs no separate un-rotated copy;
    #   * `v_t` is the raw x tile the R loop already loaded.
    # (`_TRAPFOLD_NOTE` gives the diagonal handling actually used.)
    # STABILITY: none of this touches an exponent.  `w_t` multiplies the
    # WRITE (`x`), never `a_val * D`, so the decay factors are the same
    # non-negative-difference expressions -- no clamp is added
    # and none is needed.
    # The saved `Dot_ptr` [B, H, L_padded] is the backward's operand:
    # it is cheaper to store than the four [B, L, N] contractions that
    # rebuild it.
    #
    # R, O >= 1 (FORWARD ONLY): d_t
    # generalises from the scalar k_t . q_t to the O x R matrix
    # d_t[o, r] = C_rd_o[t] . Bc_r[t], and the correction from a scalar
    # y_t -= n_t d_t v_t to y_t[o] -= n_t sum_r d_t[o, r] x_r,t.  Both
    # branches below already loop over (o, r) for the state/score MMAs,
    # so the correction is the SAME diagonal-of-Score subtraction, taken
    # once per (o, r) pair inside those loops -- no separate [O, R] tile
    # or second pass.  `_trap_adjoint_kernel` (m3_epilogue.py) is NOT
    # generalised: it assumes one scalar dot per kernel head, so
    # `Dot_ptr`'s (O, R) tail is written here but has no consumer
    # in the backward.
    Lam_ptr=None,    # [B, H, L_padded] fp32 lambda_t (post-sigmoid)
    Dot_ptr=None,    # [B, H, L_padded(, O, R)] fp32 (k_t . q_t / d_t[o, r]),
                     # stored for bwd.  At R = O = 1 the trailing (O, R)
                     # axes collapse to zero-stride offsets
                     # (`+ o * s_do + r * s_dr` adds zero).
    s_do=0, s_dr=0,  # Dot_ptr's O, R strides (0 at R = O = 1)
    # Dot_ptr's OWN (b, h, l) strides: the token stride is O*R for the
    # [B, H, L, O, R] buffer, not 1 as for `dw`.
    s_dqb=0, s_dqh=0, s_dql=0,
    HAS_TRAP: tl.constexpr = False,
    STORE_DOT: tl.constexpr = False,
    # ── The phase cumsum, in-kernel ───────────────────────────────────
    # PHASE_CUMSUM: `Theta_ptr` holds the per-token angle INCREMENTS
    # (d_theta_t), not the running phase, and `ThOff_ptr` [B, H, nc, N/2]
    # holds the phase ENTERING each chunk (the exclusive prefix over the
    # per-chunk increment sums, built host-side on a tensor 64x smaller
    # than the per-token one).  The phase this kernel rotates by is then
    #     theta_t = ThOff[b, h, c] + cumsum_{t' <= t}(d_theta_{t'})
    # with the cumulative sum taken over the [T, N/2] tile the load
    # already brings in.  Carrying the phase as
    # a running per-CTA value across the chunk loop is WRONG here: the
    # forward is SEGMENTED (nseg > 1 splits this loop across CTAs) and
    # the backward's Ka / Kbc-hb / Kc are chunk-PARALLEL, so all four
    # kernels need random access to the phase at any chunk.  A per-CHUNK
    # offset is exactly the state that makes that random access O(1).
    # STABILITY: unchanged.  The phase is a ROTATION -- it never reaches
    # an exponent, so no decay factor moves and no clamp is added.
    ThOff_ptr=None,  # [B, H, nc, N//2] fp32 phase entering each chunk
    s_ob=0, s_oh=0, s_oc=0, s_ok=0,   # s_oh = 0 <=> head-shared phase
    PHASE_CUMSUM: tl.constexpr = False,
    # DCS_INK: D_i formed IN KERNEL.  True:
    # `Delta_ptr` holds the RAW chunked delta [B, H, nc, T] and the
    # chunk-local prefix D_i is a `tl.cumsum` of the tile this kernel
    # already loads.  See `_dcs_tile` / `_dcs_end` above.
    DCS_INK: tl.constexpr = False,
    # ── The STACKED (o, r) score tile ─────────────────────────────────
    # `RO_STACK` selects the stacked O > 1 body; `OS` / `RS` are the
    # stack factors, chosen by `_ro_stack()` on the host so the
    # [T*OS, T*RS] accumulator stays at or under 64x64.  OS = RS = 1 is
    # never selected (it would be the unstacked body with extra index
    # arithmetic).  See the body's own note for the design and for how
    # the trapezoid's fp32 `trap_dot` is formed in the stacked layout.
    RO_STACK: tl.constexpr = False,
    OS: tl.constexpr = 1,
    RS: tl.constexpr = 1,
):
    """Fused SSD forward for the A-scalar case, supporting R,O >= 1.

    One CTA per (batch, head, G-tile).  Loops over chunks serially,
    carrying S[N, GT] in registers.  State is shared across R (write)
    and O (read) streams.

    R>1: aggregate sums over R input streams, each with its own B_r and x_r.
    O>1: readout produces O output channels, each with its own C_o.
    Score is recomputed per (o, r) pair.

    HAS_BETA: beta[H,N,R] added to B AFTER rotation (rotated frame).
    HAS_GAMMA: gamma[H,N,O] added to C AFTER rotation; gamma itself is
               rotated by theta_h when HAS_THETA_H.
    """
    tl.static_assert(not HAS_DSKIP or HAS_DW,
                     "fused D-skip needs the raw-x (HAS_DW) load: without "
                     "it X_ptr holds x_scaled, not the skip's operand")
    tl.static_assert(not HAS_DSKIP or O == 1,
                     "fused D-skip is the O == 1 branch (its weights carry "
                     "no output axis)")
    tl.static_assert(not (HAS_GATE or HAS_DSKIP) or not STATE_ONLY,
                     "the epilogue has no place in the STATE_ONLY replay")
    # The trapezoid's diagonal correction is written
    # `Score - mask.to(fp32) * v[:, None]`, not `tl.where(mask, v[:, None],
    # 0.0)`.  Same numbers for finite v (v*1, v*0 exact); the `where` form
    # makes Triton 3.8's LLVM NVPTX backend abort on sm_120 ("Cannot BITCAST
    # between types of different sizes") whenever it sits
    # in the runtime R loop with tf32 MMAs (fp32 IO, precision tf32/io, R or
    # O > 1, trapezoid).
    tl.static_assert(not HAS_TRAP or HAS_DW,
                     "the in-kernel trapezoid IS the write scale: it needs "
                     "the raw-x (HAS_DW) form")
    tl.static_assert(
        not HAS_TRAP or not HAS_ROT_BIAS,
        "the in-kernel trapezoid is Mamba-3's mode at any R, O >= 1: the "
        "phase (plus the static READ offset theta_h, whose rotation the "
        "dot below and the adjoint kernel carry) with no write-side "
        "rot_bias")
    b = tl.program_id(0)
    h = tl.program_id(1)
    NGT_C = G_SIZE // GT
    seg = tl.program_id(2) // NGT_C
    gt_idx = tl.program_id(2) % NGT_C
    c_lo = seg * NC_SEG
    c_hi = tl.minimum(NC, c_lo + NC_SEG)
    if IDX64:
        # only b/h products can cross 2^31: a per-(b, h) slice that large
        # (>= 4.3 GB in bf16) cannot exist on this hardware, so c/o/r/tile
        # products stay safely in int32.
        b = b.to(tl.int64)
        h = h.to(tl.int64)

    a_val = -tl.exp(tl.load(A_ptr + h))

    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    causal = offs_t[:, None] >= offs_t[None, :]
    neg_inf = float("-inf")

    bg = h // HPG_B
    cg = h // HPG_C

    pair_idx = offs_n // 2
    rot_sign = tl.where((offs_n % 2) == 0, 1.0, -1.0)

    # ── Per-head rotation (constant across chunks) ────────────────────
    # (the PAIR_HALF branch builds its own [N/2] pair-axis form below)
    if HAS_THETA_H and not PAIR_HALF:
        th_h_full = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)  # [N]
        ch_full = _cos_approx(th_h_full)
        sh_full = _sin_approx(th_h_full)
        fwd_coeff = rot_sign * sh_full

    # ── Per-head write rotation bias (constant across chunks) ──────────
    if HAS_ROT_BIAS and not PAIR_HALF:
        rb_full = tl.load(RotBias_ptr + h * (N // 2) + pair_idx)  # [N]
        crb = _cos_approx(rb_full)
        srb_fwd = rot_sign * _sin_approx(rb_full)

    if PAIR_HALF:
        # ═══ The ROTATE-HALF state layout ═════════════════════════════
        # The state coordinates are permuted so that rotation pair k
        # lives at (k, k + N/2) instead of (2k, 2k+1).  Every rotated
        # tile is then TWO CONTIGUOUS [T, N/2] tiles and each one's pair
        # partner IS THE OTHER TILE -- so this branch contains no
        # `_pair_swap_2d`, no `offs_n ^ 1` load and no `_pair_bcast_2d`:
        # the trig runs natively on the [T, N/2] pair axis and the
        # rotation is four FMAs on the two halves,
        #     B~_1 = cos B_1 + sin B_2,   B~_2 = cos B_2 - sin B_1
        # which is the SAME map R(theta)^T the interleaved branch builds
        # (even lane <-> half 1, odd lane <-> half 2).
        #
        # Where the state axis is a CONTRACTION (the score C~^T B~ and
        # the cross term C~^T S) the sum splits exactly over the two
        # halves -- two [.., N/2]-K MMAs summed.  Where it is an OUTPUT
        # axis (the aggregate B~ x^T, the carried state S, SAll, the
        # final state) the two halves are separate [N/2, GT] accumulators
        # stored at contiguous row ranges.  Nothing else changes; pair k
        # still means pair k for theta, theta_h, beta and gamma, which is
        # why the layout is a pure permutation of the caller's N axis.
        #
        # The R = O = 1 code is kept VERBATIM under `R1O1`; the general
        # (R, O) body is a separate, structurally identical branch.
        R1O1: tl.constexpr = (R == 1 and O == 1)
        tl.static_assert(not HAS_ROT_BIAS,
                         "PAIR_HALF does not serve rot_bias")
        tl.static_assert(N >= 32,
                         "PAIR_HALF needs N/2 >= 16 (the MMA K floor)")
        KH: tl.constexpr = N // 2
        offs_h1 = tl.arange(0, KH)
        offs_h2 = KH + tl.arange(0, KH)

        if HAS_THETA_H:
            th_h = tl.load(ThetaH_ptr + h * KH + offs_h1)      # [N/2]
            ch_h = _cos_approx(th_h)
            sh_h = _sin_approx(th_h)

        if HAS_X0:
            x0_base = (X0_ptr + b * s_x0b + h * s_x0h + seg * s_x0s
                       + offs_gt[None, :] * s_x0g)
            S1 = tl.load(x0_base + offs_h1[:, None] * s_x0n).to(tl.float32)
            S2 = tl.load(x0_base + offs_h2[:, None] * s_x0n).to(tl.float32)
        else:
            S1 = tl.zeros([KH, GT], dtype=tl.float32)
            S2 = tl.zeros([KH, GT], dtype=tl.float32)

        for c in range(c_lo, c_hi):
            if STORE_SALL:
                s_base_st = (SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                             + offs_gt[None, :] * s_sg)
                tl.store(s_base_st + offs_h1[:, None] * s_sn, S1)
                tl.store(s_base_st + offs_h2[:, None] * s_sn, S2)

            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            dec_mask = tl.exp(
                tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                         neg_inf))
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)
            if HAS_TRAP:
                lam_t = tl.load(Lam_ptr + b * s_wb + h * s_wh
                                + (c * T + offs_t) * s_wl)
                idx_n = c * T + offs_t + 1
                m_n = idx_n < NC * T
                dt_n = tl.load(DW_ptr + b * s_wb + h * s_wh + idx_n * s_wl,
                               mask=m_n, other=0.0)
                lam_n = tl.load(Lam_ptr + b * s_wb + h * s_wh + idx_n * s_wl,
                                mask=m_n, other=0.0)
                trap_nxt = dt_n * (1.0 - lam_n)
                trap_own = lam_t * dw
                dw = lam_t * dw + trap_nxt
            if HAS_DW:
                dec_agg = dec_agg * dw
                dec_mask = dec_mask * dw[None, :]
            if HAS_TRAP:
                # TRAPFOLD (`_TRAPFOLD_NOTE`): the diagonal carries the
                # token's OWN tap lam_t dt_t (decay exp(0) = 1 exactly).
                dec_mask = tl.where(offs_t[:, None] == offs_t[None, :],
                                    trap_own[None, :], dec_mask)

            S1_ent = S1
            S2_ent = S2

            if HAS_ROT:
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht
                    + offs_h1[None, :] * s_thk,
                ).to(tl.float32)                               # [T, N/2]
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + offs_h1 * s_ok,
                    ).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)                   # [T, N/2]
                ph_sin = _sin_approx(th_rot)                   # [T, N/2]

            if R1O1:
                # ── B~, both halves ──────────────────────────────────────
                b_base = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                          + offs_t[:, None] * s_bt)
                B1 = tl.load(b_base + offs_h1[None, :] * s_bn).to(tl.float32)
                B2 = tl.load(b_base + offs_h2[None, :] * s_bn).to(tl.float32)
                if HAS_BETA:
                    be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R
                                  ).to(tl.float32)[None, :]
                    be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R
                                  ).to(tl.float32)[None, :]
                    if BIAS_PRE_ROT:
                        B1 = B1 + be1
                        B2 = B2 + be2
                if HAS_ROT:
                    Bt1 = ph_cos * B1 + ph_sin * B2
                    Bt2 = ph_cos * B2 - ph_sin * B1
                else:
                    Bt1 = B1
                    Bt2 = B2
                if HAS_BETA and not BIAS_PRE_ROT:
                    Bt1 = Bt1 + be1
                    Bt2 = Bt2 + be2

                x_base = b * s_xb + h * s_xh + c * T * s_xl
                x_r = tl.load(X_ptr + x_base
                              + offs_gt[None, :] * s_xg
                              + offs_t[:, None] * s_xl).to(tl.float32)
                xw_r = x_r * dec_agg[:, None]
                G1 = pdot(tl.trans(Bt1).to(MMA_DT), xw_r.to(MMA_DT),
                            input_precision=PREC)
                G2 = pdot(tl.trans(Bt2).to(MMA_DT), xw_r.to(MMA_DT),
                            input_precision=PREC)

                if not STATE_ONLY:
                    # ── C~ (read side), both halves ──────────────────────
                    c_base = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                              + offs_t[:, None] * s_ct)
                    C1 = tl.load(c_base + offs_h1[None, :] * s_cn).to(tl.float32)
                    C2 = tl.load(c_base + offs_h2[None, :] * s_cn).to(tl.float32)
                    if HAS_GAMMA:
                        ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O
                                      ).to(tl.float32)
                        ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O
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
                        # R(theta_h)^T on the halves: the partner of half 1
                        # is half 2 and nothing is permuted.
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

                    if STORE_DOT:
                        # k_t . q_t: the state axis is a CONTRACTION, so it
                        # is the sum of the two halves' row reductions.
                        # TRAPFOLD: the backward's input only.
                        trap_dot = (tl.sum(Bt1 * Cr1, axis=1)
                                    + tl.sum(Bt2 * Cr2, axis=1))     # [T]

                    # score: C~^T B~ splits EXACTLY over the two halves
                    S0 = (pdot(Cr1.to(MMA_DT), tl.trans(Bt1).to(MMA_DT),
                                 input_precision=PREC)
                          + pdot(Cr2.to(MMA_DT), tl.trans(Bt2).to(MMA_DT),
                                   input_precision=PREC))
                    Score = S0 * dec_mask
                    if STORE_DOT:
                        if gt_idx == 0:
                            tl.store(Dot_ptr + b * s_dqb + h * s_dqh
                                     + (c * T + offs_t) * s_dql,
                                     trap_dot)
                    y_intra = pdot(Score.to(MMA_DT), x_r.to(MMA_DT),
                                     input_precision=PREC)
                    y_cross = f[:, None] * (
                        pdot(Cr1.to(MMA_DT), S1_ent.to(MMA_DT),
                               input_precision=PREC)
                        + pdot(Cr2.to(MMA_DT), S2_ent.to(MMA_DT),
                                 input_precision=PREC))
                    Y_out = y_intra + y_cross
                    if HAS_DSKIP:
                        dv = tl.load(Dsk_ptr + h * (G_SIZE * R) + offs_gt * R)
                        Y_out = Y_out + x_r * dv[None, :]
                    y_base = b * s_yb + h * s_yh + c * s_yc
                    if STORE_Y:
                        tl.store(Y_ptr + y_base
                                 + offs_t[:, None] * s_yt
                                 + offs_gt[None, :] * s_yg,
                                 Y_out)
                    if HAS_GATE:
                        zt = tl.load(Z_ptr + b * s_zb + h * s_zh
                                     + (c * T + offs_t)[:, None] * s_zt
                                     + offs_gt[None, :] * s_zg).to(tl.float32)
                        tl.store(Yg_ptr + b * s_gb + h * s_gh + c * s_gc
                                 + offs_t[:, None] * s_gt
                                 + offs_gt[None, :] * s_gg,
                                 (Y_out * (zt * tl.sigmoid(zt))
                                  ).to(Yg_ptr.dtype.element_ty))
            else:
                # ══ The GENERAL (R, O) half forward.  The R = O = 1 body
                # above is kept verbatim; this branch is the interleaved
                # kernel's own R / O loop structure with every [T, N] tile
                # split into its two contiguous [T, N/2] halves.  The
                # aggregate `B~ (x w)^T` has the state as an MMA OUTPUT axis,
                # so it becomes two half-width MMAs per r; the score
                # `C~^T B~` and the cross term `C~^T S` have it as a
                # CONTRACTION, so each is the SUM of the two halves'
                # [., N/2]-K dots.  No `_pair_swap_2d` anywhere.
                G1 = tl.zeros([KH, GT], dtype=tl.float32)
                G2 = tl.zeros([KH, GT], dtype=tl.float32)
                for r in tl.static_range(R):
                    b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + offs_t[:, None] * s_bt + r * s_br)
                    B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn
                                 ).to(tl.float32)
                    B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn
                                 ).to(tl.float32)
                    if HAS_BETA:
                        be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R
                                      + r).to(tl.float32)[None, :]
                        be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R
                                      + r).to(tl.float32)[None, :]
                        if BIAS_PRE_ROT:
                            B1 = B1 + be1
                            B2 = B2 + be2
                    if HAS_ROT:
                        Bt1 = ph_cos * B1 + ph_sin * B2
                        Bt2 = ph_cos * B2 - ph_sin * B1
                    else:
                        Bt1 = B1
                        Bt2 = B2
                    if HAS_BETA and not BIAS_PRE_ROT:
                        Bt1 = Bt1 + be1
                        Bt2 = Bt2 + be2
                    x_r = tl.load(
                        X_ptr + b * s_xb + h * s_xh + r * s_xr
                        + c * T * s_xl + offs_gt[None, :] * s_xg
                        + offs_t[:, None] * s_xl).to(tl.float32)
                    xw_r = x_r * dec_agg[:, None]
                    G1 += pdot(tl.trans(Bt1).to(MMA_DT), xw_r.to(MMA_DT),
                                 input_precision=PREC)
                    G2 += pdot(tl.trans(Bt2).to(MMA_DT), xw_r.to(MMA_DT),
                                 input_precision=PREC)

                if not STATE_ONLY:
                    if HAS_GATE:
                        zt = tl.load(Z_ptr + b * s_zb + h * s_zh
                                     + (c * T + offs_t)[:, None] * s_zt
                                     + offs_gt[None, :] * s_zg
                                     ).to(tl.float32)
                        silu_z = zt * tl.sigmoid(zt)
                    for o in tl.static_range(O):
                        c_base_o = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                                    + offs_t[:, None] * s_ct + o * s_co)
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
                                g1 = ch_h * ga1 + sh_h * ga2
                                g2 = ch_h * ga2 - sh_h * ga1
                            else:
                                g1 = ga1
                                g2 = ga2
                            Cr1 = Cr1 + g1[None, :]
                            Cr2 = Cr2 + g2[None, :]
                        y_cross = f[:, None] * (
                            pdot(Cr1.to(MMA_DT), S1_ent.to(MMA_DT),
                                   input_precision=PREC)
                            + pdot(Cr2.to(MMA_DT), S2_ent.to(MMA_DT),
                                     input_precision=PREC))
                        y_intra = tl.zeros([T, GT], dtype=tl.float32)
                        if HAS_DSKIP:
                            dsk_acc = tl.zeros([T, GT], dtype=tl.float32)
                        for r in tl.static_range(R):
                            b_base_r2 = (B_ptr + b * s_bb + c * s_bc
                                         + bg * s_bh
                                         + offs_t[:, None] * s_bt
                                         + r * s_br)
                            B1 = tl.load(b_base_r2 + offs_h1[None, :] * s_bn
                                         ).to(tl.float32)
                            B2 = tl.load(b_base_r2 + offs_h2[None, :] * s_bn
                                         ).to(tl.float32)
                            if HAS_BETA:
                                be1 = tl.load(Beta_ptr + h * N * R
                                              + offs_h1 * R + r
                                              ).to(tl.float32)[None, :]
                                be2 = tl.load(Beta_ptr + h * N * R
                                              + offs_h2 * R + r
                                              ).to(tl.float32)[None, :]
                                if BIAS_PRE_ROT:
                                    B1 = B1 + be1
                                    B2 = B2 + be2
                            if HAS_ROT:
                                Bt1 = ph_cos * B1 + ph_sin * B2
                                Bt2 = ph_cos * B2 - ph_sin * B1
                            else:
                                Bt1 = B1
                                Bt2 = B2
                            if HAS_BETA and not BIAS_PRE_ROT:
                                Bt1 = Bt1 + be1
                                Bt2 = Bt2 + be2
                            if STORE_DOT:
                                # the state axis is a CONTRACTION here
                                # (TRAPFOLD: the backward's input only)
                                trap_dot = (tl.sum(Bt1 * Cr1, axis=1)
                                            + tl.sum(Bt2 * Cr2, axis=1))
                            S0 = (pdot(Cr1.to(MMA_DT),
                                         tl.trans(Bt1).to(MMA_DT),
                                         input_precision=PREC)
                                  + pdot(Cr2.to(MMA_DT),
                                           tl.trans(Bt2).to(MMA_DT),
                                           input_precision=PREC))
                            Score = S0 * dec_mask
                            if STORE_DOT:
                                if gt_idx == 0:
                                    tl.store(Dot_ptr + b * s_dqb
                                             + h * s_dqh
                                             + (c * T + offs_t) * s_dql
                                             + o * s_do + r * s_dr,
                                             trap_dot)
                            x_r = tl.load(
                                X_ptr + b * s_xb + h * s_xh + r * s_xr
                                + c * T * s_xl + offs_gt[None, :] * s_xg
                                + offs_t[:, None] * s_xl).to(tl.float32)
                            if HAS_DSKIP:
                                dv = tl.load(Dsk_ptr + h * (G_SIZE * R)
                                             + offs_gt * R + r)
                                dsk_acc += x_r * dv[None, :]
                            y_intra += pdot(Score.to(MMA_DT),
                                              x_r.to(MMA_DT),
                                              input_precision=PREC)
                        Y_out = y_intra + y_cross
                        if HAS_DSKIP:
                            Y_out = Y_out + dsk_acc
                        if STORE_Y:
                            tl.store(Y_ptr + b * s_yb + h * s_yh + c * s_yc
                                     + o * s_yo
                                     + offs_t[:, None] * s_yt
                                     + offs_gt[None, :] * s_yg,
                                     Y_out)
                        if HAS_GATE:
                            tl.store(Yg_ptr + b * s_gb + h * s_gh
                                     + c * s_gc + o * s_go
                                     + offs_t[:, None] * s_gt
                                     + offs_gt[None, :] * s_gg,
                                     (Y_out * silu_z
                                      ).to(Yg_ptr.dtype.element_ty))

            S1 = alpha * S1_ent + G1
            S2 = alpha * S2_ent + G2

        do_sf = (s_fss > 0) | (c_hi == NC)
        if do_sf:
            sf_base = (SF_ptr + b * s_fb + h * s_fh + seg * s_fss
                       + offs_gt[None, :] * s_fg)
            tl.store(sf_base + offs_h1[:, None] * s_fn, S1)
            tl.store(sf_base + offs_h2[:, None] * s_fn, S2)
    else:
        # ── Initialise state ──────────────────────────────────────────────
        if HAS_X0:
            S = tl.load(
                X0_ptr + b * s_x0b + h * s_x0h + seg * s_x0s
                + offs_gt[None, :] * s_x0g + offs_n[:, None] * s_x0n,
            ).to(tl.float32)
        else:
            S = tl.zeros([N, GT], dtype=tl.float32)

        # ── Main chunk loop (segment window; c_lo=0, c_hi=NC unsegmented) ──
        for c in range(c_lo, c_hi):

            # ── Store state entering this chunk (for backward) ────────────
            if STORE_SALL:
                tl.store(
                    SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                    + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg,
                    S,
                )

            # ── Load delta_cumsum and compute decay factors ───────────────
            # DIFFERENCE-FIRST, NO CLAMP: see the "Stability by construction"
            # block at the top of this file.  Every exponent is formed as
            # a_val * (a non-negative difference of CHUNK-LOCAL cumsums),
            # never as a difference of two rounded products — that ordering
            # is what makes the sign structural and the relative error of
            # each factor scale-free.  delta >= 0 is a caller obligation and
            # a_log past exp's range is refused at the entry point.
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)                          # start -> t
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)                # total chunk decay
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)                              # t -> end

            if (SCHED & 4) == 0:
                dec_mask = tl.exp(
                    tl.where((causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]),
                             neg_inf))                                  # pairwise [T, T]

            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh
                             + (c * T + offs_t) * s_wl)      # [T] write scale
            if HAS_TRAP:
                # DW_ptr holds the RAW dt in this mode; the collapsed one-tap
                # write scale is formed here.  The t+1 tap is masked at the
                # sequence end, which is exactly the zero the host's
                # `F.pad(x[..., 1:], (0, 1))` put there.
                lam_t = tl.load(Lam_ptr + b * s_wb + h * s_wh
                                + (c * T + offs_t) * s_wl)
                idx_n = c * T + offs_t + 1
                m_n = idx_n < NC * T
                dt_n = tl.load(DW_ptr + b * s_wb + h * s_wh + idx_n * s_wl,
                               mask=m_n, other=0.0)
                lam_n = tl.load(Lam_ptr + b * s_wb + h * s_wh + idx_n * s_wl,
                                mask=m_n, other=0.0)
                trap_nxt = dt_n * (1.0 - lam_n)              # [T]
                trap_own = lam_t * dw                        # [T] own tap
                dw = lam_t * dw + trap_nxt
            if HAS_DW:
                # ── ONE live x tile ──────────────────────────────────────
                # Rebinding the [T, GT] x tile with `dw` on load would force
                # the trapezoid to keep the RAW tile alive under a
                # second name for the whole chunk body -- two
                # [T, GT] fp32 tiles, i.e. +32 registers/thread at 4 warps,
                # on top of an N = 128 state tile that already fills the
                # register file.  `dw` is a SOURCE-token vector and BOTH of
                # the x tile's sinks carry a source-token axis of their own:
                #   aggregate:  xw = x * dec_agg   -> dec_agg  *= dw   (rows)
                #   intra:      Score = S0 * dec_mask, and Score's COLUMNS
                #               are x's rows       -> dec_mask *= dw[None,:]
                # so it is folded here, once per (head, chunk), and every x
                # tile below stays RAW.  Same function up to fp
                # re-association; with HAS_DW off nothing changes.
                dec_agg = dec_agg * dw
                if (SCHED & 4) == 0:
                    dec_mask = dec_mask * dw[None, :]
            if HAS_TRAP:
                if (SCHED & 4) == 0:
                    # TRAPFOLD (`_TRAPFOLD_NOTE`): the diagonal carries the
                    # token's OWN tap lam_t dt_t (its decay is exp(0) = 1).
                    dec_mask = tl.where(offs_t[:, None] == offs_t[None, :],
                                        trap_own[None, :], dec_mask)

            S_entering = S

            # ── Per-chunk rotation (HAS_ROT: theta, head-shared or per head) ──
            # TWO-STAGE: B~ = R(theta_t)^T B and C~ = R(theta_t)^T C are
            # formed here, and the per-head offsets then run on B~/C~ through
            # the SAME [N]-vector coefficients the theta-free path uses --
            # R(theta_h)^T R(theta_t)^T = R(theta_t + theta_h)^T, so the angles
            # still add, and gamma still rotates by theta_h alone.  Loading
            # theta on its own PAIR axis makes the access affine (one wide
            # load instead of N scalar ones) and halves the trig.
            if HAS_ROT:
                # s_thh is the HEAD stride: 0 for the head-shared [B, nc, T, K]
                # phase, theta.stride(3) for the per-head [B, nc, T, H, K] one.
                # `h` is a program id, so the base stays affine
                # and the tile is still ONE wide vector load per row.
                th_rot = tl.load(
                    Theta_ptr + b * s_thb + h * s_thh + c * s_thc
                    + offs_t[:, None] * s_tht
                    + tl.arange(0, N // 2)[None, :] * s_thk,
                ).to(tl.float32)  # [T, N//2]
                if PHASE_CUMSUM:
                    # the tile holds INCREMENTS: the running phase is the
                    # chunk's entry offset plus the intra-chunk prefix sum
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(
                        ThOff_ptr + b * s_ob + h * s_oh + c * s_oc
                        + tl.arange(0, N // 2) * s_ok,
                    ).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)                     # [T, N//2]
                ph_sin = _sin_approx(th_rot)                     # [T, N//2]
                if (SCHED & 2) == 0:
                    cs_t = _pair_bcast_2d(ph_cos, T, N)          # [T, N]
                    sn_t = rot_sign[None, :] * _pair_bcast_2d(ph_sin, T, N)

            if STATE_ONLY:
                # ── recompute mode: aggregate + state update only ─────────
                G = tl.zeros([N, GT], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + offs_t[:, None] * s_bt + r * s_br)
                    Bc_r = tl.load(b_base_r + offs_n[None, :] * s_bn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r
                        ).to(tl.float32)[None, :]
                    if HAS_ROT:
                        if (SCHED & 2) != 0:
                            Bc_r = _rot_apply_2d(Bc_r, ph_cos, ph_sin,
                                                rot_sign, T, N)
                        else:
                            Bc_r = cs_t * Bc_r + sn_t * _pair_swap_2d(Bc_r, T, N)
                    if HAS_ROT_BIAS:
                        if PAIR_SWAP or HAS_ROT:
                            # after the theta stage the tile is a REGISTER
                            # value, so the partner must come from it
                            Bc_r_xor = _pair_swap_2d(Bc_r, T, N)
                        else:
                            Bc_r_xor = tl.load(
                                b_base_r + (offs_n ^ 1)[None, :] * s_bn
                            ).to(tl.float32)
                        Bc_r = crb[None, :] * Bc_r + srb_fwd[None, :] * Bc_r_xor
                    if HAS_BETA and not BIAS_PRE_ROT:
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(X_ptr + x_base_r
                                  + offs_gt[None, :] * s_xg
                                  + offs_t[:, None] * s_xl).to(tl.float32)
                    xw_r = x_r * dec_agg[:, None]
                    G += pdot(tl.trans(Bc_r).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                S = alpha * S_entering + G
            elif O == 1:
                # ── O=1 fused path: aggregate + intra in single R loop ────
                c_base_o = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                            + offs_t[:, None] * s_ct)
                Co = tl.load(c_base_o + offs_n[None, :] * s_cn).to(tl.float32)
                if BIAS_PRE_ROT and HAS_GAMMA:
                    Co = Co + tl.load(
                        Gamma_ptr + h * N * O + offs_n * O).to(tl.float32)[None, :]
                if HAS_ROT:
                    if (SCHED & 2) != 0:
                        Co = _rot_apply_2d(Co, ph_cos, ph_sin,
                                            rot_sign, T, N)
                    else:
                        Co = cs_t * Co + sn_t * _pair_swap_2d(Co, T, N)
                if HAS_THETA_H:
                    if PAIR_SWAP or HAS_ROT:
                        Co_xor = _pair_swap_2d(Co, T, N)
                    else:
                        Co_xor = tl.load(
                            c_base_o + (offs_n ^ 1)[None, :] * s_cn
                        ).to(tl.float32)
                    C_rd = ch_full[None, :] * Co + fwd_coeff[None, :] * Co_xor
                else:
                    C_rd = Co
                if HAS_GAMMA and not BIAS_PRE_ROT:
                    gam = tl.load(Gamma_ptr + h * N * O + offs_n * O).to(tl.float32)
                    if HAS_THETA_H:
                        if PAIR_SWAP:
                            gam_xor = _pair_swap_1d(gam, N)
                        else:
                            gam_xor = tl.load(
                                Gamma_ptr + h * N * O + (offs_n ^ 1) * O
                            ).to(tl.float32)
                        gam = ch_full * gam + fwd_coeff * gam_xor
                    C_rd = C_rd + gam[None, :]

                if (SCHED & 1) == 0:
                    y_cross = f[:, None] * pdot(
                        C_rd.to(MMA_DT), S_entering.to(MMA_DT),
                        input_precision=PREC)
                G = tl.zeros([N, GT], dtype=tl.float32)
                y_intra = tl.zeros([T, GT], dtype=tl.float32)
                if HAS_DSKIP:
                    # D-skip on the RAW x tile the R loop already loads --
                    # the whole cost of the fused form is this accumulator.
                    dsk_acc = tl.zeros([T, GT], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + offs_t[:, None] * s_bt + r * s_br)
                    Bc_r = tl.load(b_base_r + offs_n[None, :] * s_bn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r
                        ).to(tl.float32)[None, :]
                    if HAS_ROT:
                        if (SCHED & 2) != 0:
                            Bc_r = _rot_apply_2d(Bc_r, ph_cos, ph_sin,
                                                rot_sign, T, N)
                        else:
                            Bc_r = cs_t * Bc_r + sn_t * _pair_swap_2d(Bc_r, T, N)
                    if HAS_ROT_BIAS:
                        if PAIR_SWAP or HAS_ROT:
                            # after the theta stage the tile is a REGISTER
                            # value, so the partner must come from it
                            Bc_r_xor = _pair_swap_2d(Bc_r, T, N)
                        else:
                            Bc_r_xor = tl.load(
                                b_base_r + (offs_n ^ 1)[None, :] * s_bn
                            ).to(tl.float32)
                        Bc_r = crb[None, :] * Bc_r + srb_fwd[None, :] * Bc_r_xor
                    if HAS_BETA and not BIAS_PRE_ROT:
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(X_ptr + x_base_r
                                  + offs_gt[None, :] * s_xg
                                  + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DSKIP:
                        dv = tl.load(Dsk_ptr + h * (G_SIZE * R)
                                     + offs_gt * R + r)          # [GT]
                        dsk_acc += x_r * dv[None, :]
                    if STORE_DOT:
                        # k_t . q_t on the tiles as they stand -- biased and
                        # rotated, and R(theta)^T is orthogonal at equal t.
                        # TRAPFOLD: the backward's input only.
                        trap_dot = tl.sum(Bc_r * C_rd, axis=1)   # [T]
                    xw_r = x_r * dec_agg[:, None]
                    G += pdot(tl.trans(Bc_r).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                    S0 = pdot(C_rd.to(MMA_DT), tl.trans(Bc_r).to(MMA_DT), input_precision=PREC)
                    if (SCHED & 4) != 0:
                        dec_mask = tl.exp(
                            tl.where((causal) & _same_seg(seg_h),
                                     a_val * (D_h[:, None] - D_h[None, :]),
                                     neg_inf))
                        if HAS_DW:
                            dec_mask = dec_mask * dw[None, :]
                        if HAS_TRAP:
                            dec_mask = tl.where(
                                offs_t[:, None] == offs_t[None, :],
                                trap_own[None, :], dec_mask)
                    Score = S0 * dec_mask
                    if STORE_DOT:
                        # TRAPFOLD (`_TRAPFOLD_NOTE`): the two-tap write's
                        # diagonal is the OWN tap, carried by `dec_mask`;
                        # nothing is subtracted here.  The fp32 `trap_dot`
                        # is the backward's input (`trap_prologue`).
                        if gt_idx == 0:
                            # one CTA per (b, h, chunk) owns the save; every
                            # G-tile computes the identical [T] vector.  The
                            # store is INSIDE the r loop: at R > 1 each r's
                            # own dot must reach memory, which a post-loop
                            # store (holding only the LAST r) cannot.  At
                            # O = R = 1 the offset `o * s_do + r * s_dr` is 0.
                            tl.store(Dot_ptr + b * s_dqb + h * s_dqh
                                     + (c * T + offs_t) * s_dql + r * s_dr,
                                     trap_dot)
                    y_intra += pdot(Score.to(MMA_DT), x_r.to(MMA_DT), input_precision=PREC)

                if (SCHED & 1) != 0:
                    # `YC`: taken HERE, after the R loop -- `C_rd` and
                    # `S_entering` are both live to this point regardless
                    # (the first feeds every `S0`, the second the state
                    # update on the next line), so the only thing the move
                    # changes is that the [T, GT] fp32 result no longer
                    # lives across the two MMAs.  Same dot, same operands,
                    # same order: bitwise identical.
                    y_cross = f[:, None] * pdot(
                        C_rd.to(MMA_DT), S_entering.to(MMA_DT),
                        input_precision=PREC)
                S = alpha * S_entering + G
                Y_out = y_intra + y_cross
                if HAS_DSKIP:
                    Y_out = Y_out + dsk_acc
                y_base = b * s_yb + h * s_yh + c * s_yc
                if STORE_Y:
                    tl.store(Y_ptr + y_base
                             + offs_t[:, None] * s_yt
                             + offs_gt[None, :] * s_yg,
                             Y_out)
                if HAS_GATE:
                    zt = tl.load(Z_ptr + b * s_zb + h * s_zh
                                 + (c * T + offs_t)[:, None] * s_zt
                                 + offs_gt[None, :] * s_zg).to(tl.float32)
                    tl.store(Yg_ptr + b * s_gb + h * s_gh + c * s_gc
                             + offs_t[:, None] * s_gt
                             + offs_gt[None, :] * s_gg,
                             (Y_out * (zt * tl.sigmoid(zt))
                              ).to(Yg_ptr.dtype.element_ty))
            elif RO_STACK:
                # ═══ The STACKED (o, r) score tile ═══
                #
                # The O > 1 body above issues, per (head, chunk):
                #   R aggregate dots, O carry dots, O*R score dots and
                #   O*R intra-output dots
                # at the S-row geometry (T = 32, N = 64, GT = 16) that is
                # 12 MMAs whose K is 64, 64, 64 and 32 and whose OUTPUT
                # tiles are [N, GT] = 64x16, [T, GT] = 32x16, [T, T] =
                # 32x32 and [T, GT] = 32x16 -- three of the four sit at or
                # below the tensor core's minimum legal tile in one
                # dimension.  It also REBUILDS `B~_r` (load + bias) O*R
                # times, once per (o, r) (head redundancy).
                #
                # Stacking folds `o` into the score tile's ROW axis and
                # `r` into its COLUMN axis, and folds `sum_r` into the
                # contraction of the intra-chunk output dot:
                #
                #   aggregate Psi   [N, T*RS] x [T*RS, GT]     K = T*RS
                #   score           [T*OS, N] x [N, T*RS]      K = N
                #   intra output    [T*OS, T*RS] x [T*RS, GT]  K = T*RS
                #   carry read      [T*OS, N] x [N, GT]        K = N
                #
                # -- 4 dots instead of 12 at OS = O, RS = R, every one of
                # them LARGER in both M and N.  No `tl.cat` is needed: the
                # operands are addressed through their own strides, with
                # `n` still the contiguous axis, so each stacked row is
                # the SAME coalesced segment the per-(o, r) load read.
                #
                # STACK FACTOR.  `Score` grows from [T, T] to
                # [T*OS, T*RS], so OS and RS are constexprs chosen on the
                # host (`_ro_stack`) to keep the accumulator at or under
                # 64x64 -- at T = 32 that is OS = RS = 2, i.e. R4O4 runs
                # 2x2 blocks of the stacked tile rather than one 128x128
                # one.  Mandatory: the larger bodies spill at 4 warps.
                #
                # THE TRAPEZOID.  `trap_dot[o, r][t] = sum_n C~_o[t, n]
                # B~_r[t, n]` is an fp32 row reduction at EQUAL t, and the
                # two stacked operands carry `o` and `r` on different row
                # axes, so it cannot be formed from `C_st` and `B_st`.  It
                # does NOT need the MMA diagonal (whose bf16 operands
                # would change what `_trap_adjoint_kernel` consumes): for
                # a fixed r, `B~_r` broadcast over o is ONE more strided
                # load -- the same bytes, with `t_m` in the row index and
                # `r` fixed -- and `tl.sum(C_st * B_rep, axis=1)` is the
                # per-(o, r) fp32 dot for ALL o at once.  Over the RS
                # columns of one block that is RS loads of [T*OS, N] and
                # RS fp32 reductions of T*OS*N, i.e. EXACTLY the bytes and
                # flops the per-(o, r) form paid (R*O tiles of [T, N]).
                # The correction then goes onto the stacked Score's block
                # diagonal, `t_m == t_k` AND `r_k == rr`.
                TO: tl.constexpr = T * OS
                TR: tl.constexpr = T * RS
                offs_m = tl.arange(0, TO)
                t_m = offs_m % T
                o_m = offs_m // T
                offs_k = tl.arange(0, TR)
                t_k = offs_k % T
                r_k = offs_k // T
                # Block-replication of the decay pieces: pure register
                # shuffles on tiles that are already live, NOT a second
                # exp.  `dec_mask`'s rows are destination tokens (-> t_m)
                # and its columns source tokens (-> t_k), and `HAS_DW` is
                # already folded into its columns and into `dec_agg`.
                dec_agg_k = tl.reshape(
                    tl.broadcast_to(tl.reshape(dec_agg, (1, T)), (RS, T)),
                    (TR,))
                f_m = tl.reshape(
                    tl.broadcast_to(tl.reshape(f, (1, T)), (OS, T)), (TO,))
                dec_row = tl.reshape(
                    tl.broadcast_to(tl.reshape(dec_mask, (T, 1, T)),
                                    (T, RS, T)), (T, TR))
                dec_st = tl.reshape(
                    tl.broadcast_to(tl.reshape(dec_row, (1, T, TR)),
                                    (OS, T, TR)), (TO, TR))

                G = tl.zeros([N, GT], dtype=tl.float32)
                for rb in tl.static_range(R // RS):
                    r_st = rb * RS + r_k
                    B_st = tl.load(
                        B_ptr + b * s_bb + c * s_bc + bg * s_bh
                        + t_k[:, None] * s_bt + r_st[:, None] * s_br
                        + offs_n[None, :] * s_bn).to(tl.float32)
                    if HAS_BETA:
                        B_st = B_st + tl.load(
                            Beta_ptr + h * N * R + offs_n[None, :] * R
                            + r_st[:, None]).to(tl.float32)
                    x_st = tl.load(
                        X_ptr + b * s_xb + h * s_xh + r_st[:, None] * s_xr
                        + (c * T + t_k)[:, None] * s_xl
                        + offs_gt[None, :] * s_xg).to(tl.float32)
                    G += pdot(tl.trans(B_st).to(MMA_DT),
                                (x_st * dec_agg_k[:, None]).to(MMA_DT),
                                input_precision=PREC)
                S = alpha * S_entering + G

                if HAS_GATE:
                    zt = tl.load(Z_ptr + b * s_zb + h * s_zh
                                 + (c * T + offs_t)[:, None] * s_zt
                                 + offs_gt[None, :] * s_zg).to(tl.float32)
                    silu_st = tl.reshape(
                        tl.broadcast_to(
                            tl.reshape(zt * tl.sigmoid(zt), (1, T, GT)),
                            (OS, T, GT)), (TO, GT))

                for ob in tl.static_range(O // OS):
                    o_st = ob * OS + o_m
                    C_st = tl.load(
                        C_ptr + b * s_cb + c * s_cc + cg * s_ch
                        + t_m[:, None] * s_ct + o_st[:, None] * s_co
                        + offs_n[None, :] * s_cn).to(tl.float32)
                    if HAS_GAMMA:
                        C_st = C_st + tl.load(
                            Gamma_ptr + h * N * O + offs_n[None, :] * O
                            + o_st[:, None]).to(tl.float32)
                    Y_out = f_m[:, None] * pdot(
                        C_st.to(MMA_DT), S_entering.to(MMA_DT),
                        input_precision=PREC)
                    for rb in tl.static_range(R // RS):
                        r_st = rb * RS + r_k
                        B_st = tl.load(
                            B_ptr + b * s_bb + c * s_bc + bg * s_bh
                            + t_k[:, None] * s_bt + r_st[:, None] * s_br
                            + offs_n[None, :] * s_bn).to(tl.float32)
                        if HAS_BETA:
                            B_st = B_st + tl.load(
                                Beta_ptr + h * N * R + offs_n[None, :] * R
                                + r_st[:, None]).to(tl.float32)
                        Score = pdot(C_st.to(MMA_DT),
                                       tl.trans(B_st).to(MMA_DT),
                                       input_precision=PREC) * dec_st
                        if STORE_DOT:
                            # TRAPFOLD: the own tap rides `dec_st` (built
                            # from the diagonal-carrying `dec_mask`); the
                            # per-(o, r) dot is the backward's input only.
                            for rr in tl.static_range(RS):
                                B_rep = tl.load(
                                    B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                    + t_m[:, None] * s_bt
                                    + (rb * RS + rr) * s_br
                                    + offs_n[None, :] * s_bn).to(tl.float32)
                                if HAS_BETA:
                                    B_rep = B_rep + tl.load(
                                        Beta_ptr + h * N * R + offs_n * R
                                        + (rb * RS + rr)
                                    ).to(tl.float32)[None, :]
                                td = tl.sum(C_st * B_rep, axis=1)     # [TO]
                                if gt_idx == 0:
                                    tl.store(
                                        Dot_ptr + b * s_dqb + h * s_dqh
                                        + (c * T + t_m) * s_dql
                                        + o_st * s_do
                                        + (rb * RS + rr) * s_dr, td)
                        x_st = tl.load(
                            X_ptr + b * s_xb + h * s_xh + r_st[:, None] * s_xr
                            + (c * T + t_k)[:, None] * s_xl
                            + offs_gt[None, :] * s_xg).to(tl.float32)
                        Y_out += pdot(Score.to(MMA_DT), x_st.to(MMA_DT),
                                        input_precision=PREC)
                    if STORE_Y:
                        tl.store(
                            Y_ptr + b * s_yb + h * s_yh + c * s_yc
                            + o_st[:, None] * s_yo + t_m[:, None] * s_yt
                            + offs_gt[None, :] * s_yg, Y_out)
                    if HAS_GATE:
                        tl.store(
                            Yg_ptr + b * s_gb + h * s_gh + c * s_gc
                            + o_st[:, None] * s_go + t_m[:, None] * s_gt
                            + offs_gt[None, :] * s_gg,
                            (Y_out * silu_st).to(Yg_ptr.dtype.element_ty))
            else:
                # ── O>1 path: separate aggregate, then O loop ─────────────
                G = tl.zeros([N, GT], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    b_base_r = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                + offs_t[:, None] * s_bt + r * s_br)
                    Bc_r = tl.load(b_base_r + offs_n[None, :] * s_bn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(
                            Beta_ptr + h * N * R + offs_n * R + r
                        ).to(tl.float32)[None, :]
                    if HAS_ROT:
                        if (SCHED & 2) != 0:
                            Bc_r = _rot_apply_2d(Bc_r, ph_cos, ph_sin,
                                                rot_sign, T, N)
                        else:
                            Bc_r = cs_t * Bc_r + sn_t * _pair_swap_2d(Bc_r, T, N)
                    if HAS_ROT_BIAS:
                        if PAIR_SWAP or HAS_ROT:
                            # after the theta stage the tile is a REGISTER
                            # value, so the partner must come from it
                            Bc_r_xor = _pair_swap_2d(Bc_r, T, N)
                        else:
                            Bc_r_xor = tl.load(
                                b_base_r + (offs_n ^ 1)[None, :] * s_bn
                            ).to(tl.float32)
                        Bc_r = crb[None, :] * Bc_r + srb_fwd[None, :] * Bc_r_xor
                    if HAS_BETA and not BIAS_PRE_ROT:
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(X_ptr + x_base_r
                                  + offs_gt[None, :] * s_xg
                                  + offs_t[:, None] * s_xl).to(tl.float32)
                    xw_r = x_r * dec_agg[:, None]
                    G += pdot(tl.trans(Bc_r).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)

                S = alpha * S_entering + G

                if HAS_GATE:
                    # ONE gate per SUBSYSTEM, broadcast over that subsystem's
                    # O outputs (the z branch is d_inner wide, not O*d_inner),
                    # so the tile is loaded once and reused for every o.
                    zt = tl.load(Z_ptr + b * s_zb + h * s_zh
                                 + (c * T + offs_t)[:, None] * s_zt
                                 + offs_gt[None, :] * s_zg).to(tl.float32)
                    silu_z = zt * tl.sigmoid(zt)

                for o in tl.range(O, loop_unroll_factor=1):
                    c_base_o = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                                + offs_t[:, None] * s_ct + o * s_co)
                    Co = tl.load(c_base_o + offs_n[None, :] * s_cn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_GAMMA:
                        Co = Co + tl.load(
                            Gamma_ptr + h * N * O + offs_n * O + o
                        ).to(tl.float32)[None, :]
                    if HAS_ROT:
                        if (SCHED & 2) != 0:
                            Co = _rot_apply_2d(Co, ph_cos, ph_sin,
                                                rot_sign, T, N)
                        else:
                            Co = cs_t * Co + sn_t * _pair_swap_2d(Co, T, N)
                    if HAS_THETA_H:
                        if HAS_ROT:
                            Co_xor = _pair_swap_2d(Co, T, N)
                        else:
                            Co_xor = tl.load(
                                c_base_o + (offs_n ^ 1)[None, :] * s_cn
                            ).to(tl.float32)
                        C_rd = ch_full[None, :] * Co + fwd_coeff[None, :] * Co_xor
                    else:
                        C_rd = Co
                    if HAS_GAMMA and not BIAS_PRE_ROT:
                        gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                        if HAS_THETA_H:
                            gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                            gam = ch_full * gam + fwd_coeff * gam_xor
                        C_rd = C_rd + gam[None, :]

                    y_cross = f[:, None] * pdot(C_rd.to(MMA_DT), S_entering.to(MMA_DT),
                                                   input_precision=PREC)

                    y_intra = tl.zeros([T, GT], dtype=tl.float32)
                    for r in tl.range(R, loop_unroll_factor=1):
                        b_base_r2 = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                                     + offs_t[:, None] * s_bt + r * s_br)
                        Bc_r = tl.load(b_base_r2 + offs_n[None, :] * s_bn).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_BETA:
                            Bc_r = Bc_r + tl.load(
                                Beta_ptr + h * N * R + offs_n * R + r
                            ).to(tl.float32)[None, :]
                        if HAS_ROT:
                            if (SCHED & 2) != 0:
                                Bc_r = _rot_apply_2d(Bc_r, ph_cos, ph_sin,
                                                    rot_sign, T, N)
                            else:
                                Bc_r = cs_t * Bc_r + sn_t * _pair_swap_2d(Bc_r, T, N)
                        if HAS_ROT_BIAS:
                            if HAS_ROT:
                                Bc_r_xor2 = _pair_swap_2d(Bc_r, T, N)
                            else:
                                Bc_r_xor2 = tl.load(
                                    b_base_r2 + (offs_n ^ 1)[None, :] * s_bn
                                ).to(tl.float32)
                            Bc_r = crb[None, :] * Bc_r + srb_fwd[None, :] * Bc_r_xor2
                        if HAS_BETA and not BIAS_PRE_ROT:
                            beta_r2 = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                            Bc_r = Bc_r + beta_r2[None, :]
                        if STORE_DOT:
                            # d_t[o, r] = C_rd[t] . Bc_r[t] on the tiles as they
                            # stand (biased, rotated) -- same fp32 row-reduction
                            # the O == 1 branch uses, generalised over the O
                            # loop this branch already runs in.  R(theta)^T is
                            # orthogonal at equal t so the rotation cancels here
                            # exactly as it does at O = 1.  TRAPFOLD: the
                            # backward's input only.
                            trap_dot = tl.sum(Bc_r * C_rd, axis=1)   # [T]
                        S0 = pdot(C_rd.to(MMA_DT), tl.trans(Bc_r).to(MMA_DT), input_precision=PREC)
                        if (SCHED & 4) != 0:
                            dec_mask = tl.exp(
                                tl.where((causal) & _same_seg(seg_h),
                                         a_val * (D_h[:, None] - D_h[None, :]),
                                         neg_inf))
                            if HAS_DW:
                                dec_mask = dec_mask * dw[None, :]
                            if HAS_TRAP:
                                dec_mask = tl.where(
                                    offs_t[:, None] == offs_t[None, :],
                                    trap_own[None, :], dec_mask)
                        Score = S0 * dec_mask
                        if STORE_DOT:
                            # TRAPFOLD: the own tap sits on `dec_mask`'s
                            # diagonal; nothing is subtracted here.
                            if gt_idx == 0:
                                tl.store(Dot_ptr + b * s_dqb + h * s_dqh
                                         + (c * T + offs_t) * s_dql
                                         + o * s_do + r * s_dr,
                                         trap_dot)

                        x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                        x_r = tl.load(X_ptr + x_base_r
                                      + offs_gt[None, :] * s_xg
                                      + offs_t[:, None] * s_xl).to(tl.float32)
                        y_intra += pdot(Score.to(MMA_DT), x_r.to(MMA_DT), input_precision=PREC)

                    Y_out = y_intra + y_cross
                    y_base = b * s_yb + h * s_yh + c * s_yc + o * s_yo
                    if STORE_Y:
                        tl.store(Y_ptr + y_base
                                 + offs_t[:, None] * s_yt
                                 + offs_gt[None, :] * s_yg,
                                 Y_out)
                    if HAS_GATE:
                        tl.store(Yg_ptr + b * s_gb + h * s_gh + c * s_gc
                                 + o * s_go
                                 + offs_t[:, None] * s_gt
                                 + offs_gt[None, :] * s_gg,
                                 (Y_out * silu_z
                                  ).to(Yg_ptr.dtype.element_ty))

        # ── Store final state (per-seg in phase A; last segment otherwise) ──
        do_sf = (s_fss > 0) | (c_hi == NC)
        if do_sf:
            tl.store(
                SF_ptr + b * s_fb + h * s_fh + seg * s_fss
                + offs_gt[None, :] * s_fg + offs_n[:, None] * s_fn,
                S,
            )


#: What the LAST `fused_chunk_fwd` launch resolved for the stack.  A test
#: hook: `_ro_stack()` alone says what the factors WOULD be, not whether
#: the dispatcher's envelope let them through.
_LAST_RO_STACK = (False, 1, 1)


def _ro_stack(T: int, R: int, O: int, refused: bool):
    """The (o, r) stack factors, or `(False, 1, 1)`.

    `Score` grows from `[T, T]` to `[T*OS, T*RS]`, so the factors are
    capped at `64 // T` per axis -- the accumulator then never exceeds
    64x64 fp32 (16 KB / CTA, 32 registers/thread at 4 warps).  At
    `T >= 64` the cap is 1 on both axes, i.e. the stacked body would be
    the unstacked one plus index arithmetic, so it is not selected.

    THE STATE WIDTH IS PRICED BY ptxas, NOT HERE.
    The cap above prices ONE of the two tile families the stack grows.
    The other is the N-BEARING operands -- `B_st [T*RS, N]`,
    `C_st [T*OS, N]` and, under the trapezoid, the `B_rep [T*OS, N]`
    replica, where the per-(o, r) body holds one `[T, N]` tile at a time
    -- and at N >= 128 they can take the body over the register file.
    An arithmetic cap on them is wrong on one part or the other, because
    what actually happens is a property of ptxas, not of the tile sizes:
    at the S-row R = O = 2, N = 128 cell the stacked body spills heavily
    on A100 (32 regs, >1 KB/thread; 13.9x slower than unstacked) while on
    H100 it is faster than unstacked (1.37x).  So the choice
    is made by READING ptxas (`_fwd_fix_stack`, the spill-ladder
    construction applied to this constexpr instead of to `num_stages`),
    and this function keeps the tile-shape rule.

    `SSD_MIMO_FWD_RO_STACK`: "auto" (default) | "0" | "1" (force where
    the envelope allows it) | "<OS>,<RS>" (forces the factors, including
    the blocked forms, for measurement).
    """
    env = _env("FWD_RO_STACK", "auto")
    if env == "0" or refused or O <= 1:
        return False, 1, 1
    if "," in env:
        os_, rs_ = (int(v) for v in env.split(","))
    else:
        cap = max(1, 64 // int(T))
        os_, rs_ = min(O, cap), min(R, cap)
    if os_ < 1 or rs_ < 1 or O % os_ or R % rs_:
        return False, 1, 1
    if os_ * rs_ <= 1 and env != "1":
        return False, 1, 1
    if "," not in env and env != "1" and (os_ != O or rs_ != R):
        return False, 1, 1          # blocked form: refuted, see above
    return True, os_, rs_


#: The last Hopper stacked-body refusal, for `last_fwd_cfg()` /
#: `dispatch_log()`: "" when the refusal did not fire on the last launch,
#: `"sm90_io_ro_stack"` when it moved an otherwise-stacked cell onto the
#: per-(o, r) body.
_LAST_STACK_GUARD = ""


#: The first ptxas build measured to compile the stacked body's wgmma
#: descriptors correctly (CUDA 12.9 Update 1).  See `ro_stack_wgmma_fault`.
SM90_PTXAS_FIXED = (12, 9, 86)


@functools.lru_cache(maxsize=None)
def _ptxas_version_at(path: str):
    """`(major, minor, patch)` of the ptxas binary at `path`, or None."""
    try:
        out = subprocess.check_output([path, "--version"],
                                      stderr=subprocess.STDOUT).decode()
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"\bV(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(v) for v in m.groups()) if m else None


def sm90_ptxas_version():
    """The ptxas Triton assembles sm_90 kernels with, as `(major, minor,
    patch)`, or None when it cannot be resolved.  Triton picks it from
    `TRITON_PTXAS_PATH` (falling back to its bundled binary) -- the
    `knobs.nvidia.ptxas` knob -- for every arch below sm_100."""
    try:
        path = triton.knobs.nvidia.ptxas.path
    except Exception:           # a Triton without the knob: unknown
        return None
    return _ptxas_version_at(path)


def ro_stack_wgmma_fault(capability, mma_io: bool,
                         ptxas_version="resolve") -> bool:
    """True where the STACKED (o, r) forward body returns a wrong `y`:
    Hopper (sm_90) with 16-bit (bf16 / fp16) MMA operands, assembled by a
    ptxas OLDER than `SM90_PTXAS_FIXED` (12.9.86).  `ptxas_version` is the
    `(major, minor, patch)` to judge ("resolve": the one Triton will use,
    `sm90_ptxas_version()`; None = unknown, judged affected).

    ROOT CAUSE: a ptxas MISCOMPILE, not Triton's
    lowering and not this kernel's layout.  The PTX Triton 3.6 emits is
    correct; ptxas 12.8.93 (Triton 3.6's bundled binary, also 3.7.1's)
    and 12.9.41 build some wgmma B-operand descriptors from a uniform
    register that is never written on that path, so the HGMMA reads
    shared memory at a garbage base -- wrong `y` where the read stays in
    the CTA's arena, an illegal access where it leaves it.
    ptxas 12.9.86 and 13.0.88 compile the SAME PTX correctly (and Triton
    3.8 bundles 12.9.86).  Minimal repro: two
    chained bf16 `tl.dot`s, no SSD code.  The SASS signature is
    pinned by the sm_90 ptxas canary test (it must match the
    stack refusal).

    Symptom at the LLM real cell (d_inner 2048, N 128, H 32, R = O = 2,
    trapezoid, bf16, stack (2, 2) at T 32): wrong `y` (GT 32 and 16), NaN
    (2 stages), wrong at R = O = 4 (stack (4, 4) at T 16); exact with tf32
    operands and on A100.  Which cells break moves with the register
    allocation, so under an affected ptxas no narrower predicate than
    "sm_90 + 16-bit operands" holds.

    The refusal is a SCHEDULE choice: the unstacked body computes the
    same sums.  fp32 operands keep the stack on Hopper,
    and every other part keeps it (sm_80 measured exact; sm_100
    unmeasured; sm_120 exact).
    """
    if int(tuple(capability)[0]) != 9 or not bool(mma_io):
        return False
    if isinstance(ptxas_version, str) and ptxas_version == "resolve":
        ptxas_version = sm90_ptxas_version()
    return ptxas_version is None or tuple(ptxas_version) < SM90_PTXAS_FIXED


def fused_chunk_fwd(
    Ctil,              # [B, nc, (Gc,) T, N, (O)]
    Btil,              # [B, nc, (Gb,) T, N, (R)]
    x_scaled,          # [B, H, G_SIZE, R, L_padded]
    delta_cumsum,      # [B, H, nc, T]
    a_log,             # [H]
    theta_h=None,      # [H, N//2] or None
    theta=None,        # [B, nc, T, N//2] (head-shared) or
                       # [B, nc, T, H, N//2] (per head) or None
    x0=None,           # [B, H, G_SIZE, N] initial state or None
    beta=None,         # [H, N, R] per-head write bias or None
    gamma=None,        # [H, N, O] per-head read bias or None
    rot_bias=None,     # [H, N//2] per-head write rotation bias or None
    bias_pre_rot=True,  # True (the default, Mamba-3's fold order): beta/
                       # gamma are added to the RAW B/C tile and rotate
                       # with it; False: they are added after the
                       # rotations (post-rot placement)
    precision="tf32",
    bc_grouped=False,
    Gb=1, Gc=1,
    dw=None,           # [B, H, L_padded] fp32 write scale; when given,
                       # x_scaled is treated as RAW x and scaled in-kernel
    store_sall=True,   # False: skip the per-chunk state save (inference)
    state_only=False,  # recompute mode: only rebuild S_all (no Y output)
    gate_z=None,       # [B, L_padded, H*G_SIZE] SwiGLU gate pre-activation
    d_skip=None,       # [H, G_SIZE, R] D-skip weights (O == 1 only)
    ypre_out=None,     # preallocated y_pre buffer; None with a gate means
                       # "no backward will run, do not store y_pre"
    gate_out_dtype=None,
    trap_lam=None,     # [B, H, L_padded] fp32 lambda_t: turns `dw` into
                       # Mamba-3's RAW dt and computes the two-tap write
                       # scale AND its rank-1 output correction in-kernel
    dot_out=None,      # [B, H, L_padded] fp32 buffer for the saved
                       # (k_t . q_t); None = do not save it (inference)
    theta_off=None,    # [B, H, nc, N//2] fp32 (or [B, 1, nc, N//2]): the
                       # phase entering each chunk.  When given, `theta`
                       # holds per-token angle INCREMENTS and the kernel
                       # cumsums them intra-chunk.
    pair_half=False,   # PAIRHALF: rotate-half state layout (pair k at
                       # (k, k+N/2) instead of (2k, 2k+1)).
    dcs_ink=False,     # `delta_cumsum` carries the RAW
                       # chunked delta and the kernel forms D_i itself
                       # (`_dcs_tile`).  The host cumsum and its buffer
                       # are then never built.
    y_token_major=False,  # allocate the RETURNED output TOKEN-major
                       # ([B, L, H*O*G] memory order) instead of the
                       # kernel-native channel-major one.  Shape and
                       # arithmetic are unchanged -- only the store's
                       # strides, which the kernel already takes as
                       # arguments -- so this costs the kernel no
                       # registers and no extra pass.
):
    """Launch the fused SSD forward kernel.

    Supports R,O >= 1.  B/C tensors carry an optional trailing R/O axis:
      Btil: [B, nc, T, N] (R=1) or [B, nc, T, N, R] (R>1)
      Ctil: [B, nc, T, N] (O=1) or [B, nc, T, N, O] (O>1)

    Returns (Y, S_all, S_final):
        Y:       [B, H, nc, T, O, G_SIZE]  output
        S_all:   [B, H, nc, N, G_SIZE]     states entering each chunk
        S_final: [B, H, G_SIZE, N]         state after last chunk

    With `gate_z` (and/or `d_skip`) the returned Y is the GATED output
    ``(scan + D.x) * silu(z)`` and the PRE-GATE value -- the backward's
    operand, contract SSD-the "saving only (y_pre, z)" -- is written
    into the caller's `ypre_out` buffer.  Passing `ypre_out=None` with a
    gate compiles the y_pre store out entirely (inference).  The 3-tuple
    shape of the return is deliberate: ~40 scripts under `tools/` unpack
    exactly three values from this function.
    """
    Bsz, H, nc, T = delta_cumsum.shape
    G_SIZE = x_scaled.shape[2]
    R = x_scaled.shape[3]
    device = x_scaled.device
    f32 = torch.float32

    # ── The MMA operand dtype, resolved ONCE per launch ───────────────
    # `mma_dt` is tl.float32 in both fp32-operand modes (and at fp32 IO
    # in the third one), which makes every `.to(MMA_DT)` below Triton's
    # cast identity.
    # `precision` is then narrowed to the `tl.dot` spelling.
    mma_dt = mma_operand_dtype(precision, x_scaled.dtype)
    mma_io = mma_dt is not tl.float32
    precision = dot_precision(precision)

    if bc_grouped:
        N = Btil.shape[4]
        b_has_r = Btil.dim() >= 6
        c_has_o = Ctil.dim() >= 6
        O = Ctil.shape[5] if c_has_o else 1
    else:
        N = Btil.shape[3]
        b_has_r = Btil.dim() >= 5
        c_has_o = Ctil.dim() >= 5
        O = Ctil.shape[4] if c_has_o else 1

    gt_env = _env("FWD_GT", "")
    GT_base = min(G_SIZE, int(gt_env)) if gt_env else _pick_gt(G_SIZE)
    # Segmentation plan at the BASE tile width.  Segmentation and the
    # GT shrink cure the same under-occupancy; when the serial chunk
    # loop is long enough to segment, the wide tile + segments is far
    # better than the narrow tile — so shrink only when
    # the plan stays monolithic (short nc).
    nseg, nc_seg = _seg_plan(nc, Bsz * H * (G_SIZE // GT_base), device)
    GT = GT_base
    if not gt_env and nseg == 1:
        if _portable.nvidia_capability(device) == (9, 0):
            # sm_90 deep-under-occupancy rule: when the
            # GT-64 grid fills at most half the SMs, halving the G-tile
            # doubles the CTA count and wins big (the FWD tile counts twice
            # via the SALL=recompute STATE_ONLY relaunch).  Saturated
            # grids must keep GT=64: at B>=4 a forced GT-32/16 split is
            # 1.5-3x slower.
            sm = _sm_count(device)
            # GLUON90 (opt-in, `gluon_sm90_fwd`): its body is GT = 32 only.
            # Shrinking below that at small B * H would hand exactly those
            # calls to the Triton body, so the ROUTE (and its rounding)
            # would depend on the batch size: a sample alone != the same
            # sample in a batch.  Under the lever the
            # shrink stops at the Gluon tile.
            gt_floor = (32 if (mma_io and G_SIZE % 32 == 0 and GT >= 32
                               and _env("GLUON90", "off").lower()
                               in ("1", "on", "true")) else 16)
            while (GT > gt_floor and G_SIZE % (GT // 2) == 0
                   and Bsz * H * (G_SIZE // GT) * 2 <= sm):
                GT //= 2
    NGT = G_SIZE // GT

    # Y is allocated NATIVE-LAYOUT [B, H, O, G, L_padded] in the IO dtype
    # and exposed as the logical [B, H, nc, T, O, G] via a strided view.
    # The kernel stores through the view's strides, so the caller's
    # y = Y.permute(...).reshape(B, H, (O,) G, L) epilogue is a FREE view
    # instead of a 67MB permute-copy, and the upstream dY arrives already
    # in the layout the backward kernels read (they take dY strides).
    has_gate = gate_z is not None
    has_dskip = d_skip is not None
    if has_dskip and O != 1:
        raise NotImplementedError(
            "fused D-skip is the O == 1 branch: its weights are [H, G, R] "
            f"with no output axis, and this call has O={O}")
    if has_dskip and dw is None:
        raise NotImplementedError(
            "fused D-skip needs the raw-x (dw) form: without it x_scaled "
            "is already delta-scaled and the skip would add the wrong "
            "tensor")
    if state_only and (has_gate or has_dskip):
        raise AssertionError(
            "the fused epilogue has no place in the STATE_ONLY replay "
            "(it rebuilds S_all, which the epilogue does not touch)")

    def _native_y(dtype, token_major=False):
        # Both forms return the SAME logical [B, H, nc, T, O, G] tensor;
        # they differ only in which memory order it is a view of.
        #
        # channel-major (kernel-native, the default): [B, H, O, G, L].
        # token-major: [B, L, H, O, G] -- the layout the
        # module's output RMSNorm and `out_proj` want.  The channel-major
        # store forces a deferred transpose (an `aten::copy_` per step;
        # `F.rms_norm` would gather over its normalized axis).  Storing
        # token-major here removes the transpose entirely rather than
        # moving it.
        if token_major:
            buf = torch.empty(Bsz, nc * T, H, O, G_SIZE, device=device,
                              dtype=dtype)
            return buf.view(Bsz, nc, T, H, O, G_SIZE).permute(
                0, 3, 1, 2, 4, 5)
        buf = torch.empty(Bsz, H, O, G_SIZE, nc * T, device=device,
                          dtype=dtype)
        return buf.view(Bsz, H, O, G_SIZE, nc, T).permute(0, 1, 4, 5, 2, 3)

    if state_only:
        # no output computed — the kernel's Y stores are compiled out
        Y = torch.empty(1, 1, 1, 1, 1, 1, device=device,
                        dtype=x_scaled.dtype)
    elif has_gate:
        # Yg is the RETURNED output; Y keeps y_pre for the backward.
        # Only the RETURNED tensor takes the token-major option -- y_pre
        # is read back only by the gate adjoint, which addresses it
        # through its own stride set either way.
        Yg = _native_y(gate_out_dtype or f32, y_token_major)
        Y = ypre_out if ypre_out is not None else torch.empty(
            1, 1, 1, 1, 1, 1, device=device, dtype=x_scaled.dtype)
    else:
        Y = _native_y(x_scaled.dtype, y_token_major)
    if store_sall:
        # S_all is the ONLY large backward save (the answer to "does this
        # kernel need more memory than M2?").  It rides the IO dtype: bf16
        # training halves it to 1x the x-tensor bytes; fp32 IO keeps full
        # precision.  kernel_options backward_states_fp32=True forces fp32 regardless, and so
        # does every mode whose MMAs take fp32 operands ("tf32", "ieee"):
        # the saved state is an MMA operand of the
        # backward, and storing it at the 16-bit IO dtype puts a u16 floor
        # under those modes (per-token dTheta 4e-4 at ieee).
        sall_dtype = (f32 if (_env("SALL_FP32", "0") == "1"
                              or not mma_io)
                      else x_scaled.dtype)
        S_all = torch.empty(Bsz, H, nc, N, G_SIZE, device=device,
                            dtype=sall_dtype)
    else:
        # inference: the kernel's SAll stores are compiled out
        S_all = torch.empty(1, 1, 1, 1, 1, device=device, dtype=f32)
    S_final = torch.empty(Bsz, H, G_SIZE, N, device=device, dtype=f32)

    if bc_grouped:
        s_bb, s_bc, s_bh, s_bt, s_bn = Btil.stride()[:5]
        s_cb, s_cc, s_ch, s_ct, s_cn = Ctil.stride()[:5]
    else:
        s_bb, s_bc, s_bt, s_bn = Btil.stride()[:4]
        s_bh = 0
        s_cb, s_cc, s_ct, s_cn = Ctil.stride()[:4]
        s_ch = 0

    s_br = Btil.stride()[-1] if b_has_r else 0
    s_co = Ctil.stride()[-1] if c_has_o else 0

    hpg_b = H // Gb
    hpg_c = H // Gc

    has_theta_h = theta_h is not None
    has_rot = theta is not None
    has_x0 = x0 is not None
    has_beta = beta is not None
    has_gamma = gamma is not None
    dummy = torch.empty(1, device=device, dtype=f32)
    thh_c = theta_h.float().contiguous() if has_theta_h else dummy
    # A bf16 phase is read AS IS -- every
    # `Theta_ptr` load in this file upcasts in registers (`.to(tl.float32)`),
    # so the pointer is dtype-agnostic and no fp32 copy is needed.  Any other
    # dtype takes the fp32 upcast (the ieee oracle's phase is fp32).
    th_c = ((theta if theta.dtype == torch.bfloat16 else theta.float())
            .contiguous() if has_rot else dummy)
    # (s_thb, s_thc, s_tht, s_thh, s_thk).  A 4-d phase is head-SHARED and
    # takes s_thh = 0; a 5-d [B, nc, T, H, K] phase is PER HEAD.
    if not has_rot:
        th_st = [0, 0, 0, 0, 0]
    elif th_c.dim() == 5:
        st = th_c.stride()
        th_st = [st[0], st[1], st[2], st[3], st[4]]
    else:
        st = th_c.stride()
        th_st = [st[0], st[1], st[2], 0, st[3]]
    # ── The phase cumsum, in-kernel ───────────────────────────────────
    # `theta_off` [B, (H|1), nc, N//2] turns `theta` into the per-token
    # INCREMENT tensor; the kernel adds the chunk's entry phase and
    # cumsums the tile it already loads.  A head-shared offset takes
    # s_oh = 0, exactly like the phase itself.
    # The six extra kernel arguments are passed ONLY when the feature is
    # on: at launch-bound cells an unconditional `phase_kw` costs 1-8% of
    # wall for a branch that compiles out.  Same pattern as `trap_kw`.
    phase_cumsum = theta_off is not None
    if phase_cumsum and not has_rot:
        raise ValueError("theta_off needs theta (the increments it offsets)")
    off_c = theta_off.float().contiguous() if phase_cumsum else dummy
    phase_kw = {}
    if phase_cumsum:
        st = off_c.stride()
        phase_kw = dict(
            ThOff_ptr=off_c,
            s_ob=st[0], s_oh=0 if off_c.shape[1] == 1 else st[1],
            s_oc=st[2], s_ok=st[3],
            PHASE_CUMSUM=True,
        )
    x0_c = x0.float().contiguous() if has_x0 else dummy
    x0_st = list(x0_c.stride()) if has_x0 else [0, 0, 0, 0]
    beta_c = beta.float().contiguous() if has_beta else dummy
    gamma_c = gamma.float().contiguous() if has_gamma else dummy
    has_rot_bias = rot_bias is not None
    rb_c = rot_bias.float().contiguous() if has_rot_bias else dummy

    # The register xor-lane partner, on the R == O == 1 rotation
    # branch only (SSD_MIMO_FWD_PAIRSWAP=0 restores the second load, the
    # A/B lever to compare the two forms).
    #
    # N >= 64.  The swap is VALUE-identical to the load
    # at every N — with tensor cores off (`precision="ieee"`) the two
    # forms are bit-identical at N in {16, 32, 64}.  But it is a
    # `tl.reshape`/`tl.split`/`tl.join` on the tile that the next
    # `tl.dot` consumes, and at N <= 32 that changes the register layout
    # Triton lowers the dot from, which changes the tf32 MMA association:
    # a rot_bias of EXACTLY ZERO (cos 0 = 1, sin 0 = 0, so the partner is
    # multiplied by zero) then moves the output by 3e-5 to 5e-5 relative,
    # which breaks "zero-init rotation is no rotation".  At N = 64 the
    # layouts coincide and the two forms are bit-identical at tf32 too.
    # Below 64 the load being replaced is 8 or 16 scalar loads per row
    # rather than 64, so this costs little.
    # R, O > 1 is served by the general half body, so there is no rank
    # refusal here.  What the layout refuses is the N floor and the
    # retired rot_bias, below.
    if pair_half and N < 32:
        raise ValueError(
            f"pair_layout='half' needs N >= 32 (the MMA K floor); got N={N}")
    if pair_half and rot_bias is not None:
        raise ValueError("pair_layout='half' does not serve "
                         "rot_bias")
    # PAIRHALF has no lane permutation at all, so the register swap is
    # not part of that branch.
    pair_swap = (not pair_half and R == 1 and O == 1 and N >= 64
                 and (has_rot or has_theta_h or has_rot_bias)
                 and _env("FWD_PAIRSWAP", "1") == "1")

    # IDX64 is decided over EVERY tensor the launch addresses, internally
    # allocated ones included: the STATE_ONLY replay passes a
    # 1-element dummy Y, so a decision taken over the caller's operands
    # plus Y alone could differ between the forward and the backward's
    # recompute of the SAME scan.  `dummy` is 1 element and never trips it.
    # ── Fused epilogue operands ───────────────────────────────────────
    if has_gate:
        if gate_z.shape[-1] != H * G_SIZE:
            raise ValueError(
                f"gate_z last dim must be H*G_SIZE={H * G_SIZE}, got "
                f"{gate_z.shape[-1]}")
        if not gate_z.is_contiguous():
            raise ValueError("the fused gate needs a contiguous gate_z")
    z_c = gate_z if has_gate else dummy
    dsk_c = d_skip if has_dskip else dummy
    store_y = (not has_gate) or ypre_out is not None
    has_trap = trap_lam is not None
    if has_trap:
        if dw is None:
            raise NotImplementedError(
                "the in-kernel trapezoid IS the write scale: pass dw (the "
                "raw dt) alongside trap_lam")
        if has_rot_bias:
            raise NotImplementedError(
                "the in-kernel trapezoid is refused with rot_bias (write-side "
                "angle); the READ offset theta_h is served: the "
                "dot k_t . q_t is taken on the rotated tiles and the adjoint "
                "kernel rotates by theta_h per head")
        if dot_out is not None and dot_out.dim() not in (3, 5):
            raise ValueError(
                "dot_out must be [B, H, L_padded] (R = O = 1) "
                "or [B, H, L_padded, O, R], got shape {tuple(dot_out.shape)}")
    # R, O >= 1: the diagonal-of-Score
    # correction generalises unchanged (see the HAS_TRAP
    # block); only the saved dot's shape grows an (O, R) tail.  At
    # R = O = 1 `dot_out` keeps the [B, H, L_padded] shape and s_do =
    # s_dr = 0, so the address the kernel forms is unchanged.
    has_or_tail = dot_out is not None and dot_out.dim() == 5
    trap_kw = dict(
        Lam_ptr=trap_lam if has_trap else dummy,
        Dot_ptr=dot_out if dot_out is not None else dummy,
        s_do=dot_out.stride(3) if has_or_tail else 0,
        s_dr=dot_out.stride(4) if has_or_tail else 0,
        s_dqb=dot_out.stride(0) if dot_out is not None else 0,
        s_dqh=dot_out.stride(1) if dot_out is not None else 0,
        s_dql=dot_out.stride(2) if dot_out is not None else 0,
        HAS_TRAP=has_trap,
        STORE_DOT=has_trap and dot_out is not None,
    ) if has_trap else {}
    if has_gate:
        zs = z_c.stride()
        ep_kw = dict(
            Yg_ptr=Yg, Z_ptr=z_c, Dsk_ptr=dsk_c,
            s_zb=zs[0], s_zh=G_SIZE * zs[2], s_zt=zs[1], s_zg=zs[2],
            s_gb=Yg.stride(0), s_gh=Yg.stride(1), s_gc=Yg.stride(2),
            s_gt=Yg.stride(3), s_gg=Yg.stride(5), s_go=Yg.stride(4),
            HAS_GATE=True, HAS_DSKIP=has_dskip, STORE_Y=store_y,
        )
    elif has_dskip:
        ep_kw = dict(Yg_ptr=dummy, Z_ptr=dummy, Dsk_ptr=dsk_c,
                     HAS_GATE=False, HAS_DSKIP=True, STORE_Y=True)
    else:
        ep_kw = {}

    idx64 = _needs_i64(x_scaled, Y, S_all, S_final, Btil, Ctil,
                       x0_c, dw, th_c, thh_c, beta_c, gamma_c, rb_c,
                       Yg if has_gate else None, z_c,
                       trap_lam, dot_out,
                       off_c if phase_cumsum else None)

    dw_kw = dict(
        DW_ptr=dw if dw is not None else dummy,
        s_wb=dw.stride(0) if dw is not None else 0,
        s_wh=dw.stride(1) if dw is not None else 0,
        s_wl=dw.stride(2) if dw is not None else 0,
        HAS_DW=dw is not None,
    )
    # Only SCHED bit 4 (`DM`, the [T, T] decay mask built at its
    # consumer) conflicts: the stacked body block-replicates `dec_mask`
    # from the top of the chunk body.  Bit 1 (`YC`) is a no-op for the
    # O > 1 body (it moves the cross-chunk dot in the O == 1 one) and
    # bit 2 (`RD`) is rotation-only, which this lever refuses anyway --
    # and `FWD_SCHED_DEFAULT` is 3, so testing `& 6` here silently
    # refused the lever on EVERY fp32-operand cell.
    _ro_refused = (has_rot or has_rot_bias or has_theta_h or has_dskip
                   or bool(pair_half) or (fwd_sched(mma_io) & 4) != 0)
    _ro_stack_on, _ro_os, _ro_rs = _ro_stack(T, R, O, refused=_ro_refused)
    global _LAST_RO_STACK, _LAST_STACK_GUARD
    _LAST_STACK_GUARD = ""
    # ptxas < 12.9.86 miscompiles the
    # stacked body's wgmma descriptors on Hopper under 16-bit MMA operands
    # (wrong y on H100, illegal access on GH200); a fixed ptxas keeps the
    # stack.  Refused AFTER `_ro_stack` so the record names the refusal
    # only on a cell that would otherwise have stacked; an env pin ("1" /
    # "OS,RS") is refused too -- a pinned wrong answer is still wrong.
    if _ro_stack_on and ro_stack_wgmma_fault(
            _portable.nvidia_capability(device), mma_io):
        _ro_stack_on, _ro_os, _ro_rs = False, 1, 1
        _LAST_STACK_GUARD = "sm90_io_ro_stack"
    _LAST_RO_STACK = (_ro_stack_on, _ro_os, _ro_rs)
    common_kw = dict(
        T=T, N=N, G_SIZE=G_SIZE, GT=GT,
        PREC=precision,
        MMA_DT=mma_dt,
        HAS_THETA_H=has_theta_h,
        HAS_ROT=has_rot,
        HAS_BETA=has_beta,
        HAS_GAMMA=has_gamma,
        HAS_ROT_BIAS=has_rot_bias,
        BIAS_PRE_ROT=bool(bias_pre_rot),
        PAIR_SWAP=pair_swap,
        PAIR_HALF=bool(pair_half),
        HPG_B=hpg_b,
        HPG_C=hpg_c,
        R=R, O=O,
        IDX64=idx64,
        SCHED=fwd_sched(mma_io),
        DCS_INK=bool(dcs_ink),
        RO_STACK=_ro_stack_on, OS=_ro_os, RS=_ro_rs,
        **phase_kw,
        **dw_kw,
        **trap_kw,
    )
    head_args = (Ctil, Btil, x_scaled, delta_cumsum, a_log,
                 thh_c, th_c)
    mid_args = (beta_c, gamma_c, rb_c)
    stride_args = (H,
                   s_cb, s_cc, s_ct, s_cn,
                   s_bb, s_bc, s_bt, s_bn,
                   *x_scaled.stride(),
                   *delta_cumsum.stride())
    tail_strides = (*th_st, s_bh, s_ch, s_br, s_co)

    if nseg > 1:
        # ── Phase A: per-segment final states from zero entry state ──
        # (STATE_ONLY + no S_all store = aggregates + state scan only.)
        psi = torch.empty(Bsz, H, nseg, G_SIZE, N, device=device, dtype=f32)
        dummy_y = torch.empty(1, 1, 1, 1, 1, 1, device=device, dtype=f32)
        dummy_s = torch.empty(1, 1, 1, 1, 1, device=device, dtype=f32)
        grid_a = (Bsz, H, NGT * nseg)
        wa, sa = _fwd_launch_cfg(device, Bsz * H * NGT * nseg, N,
                                 state_only=True, gt=GT, mma_io=mma_io,
                                 pair_half=pair_half,
                                 cell=_fwd_cell_key(
                                     Bsz * H * NGT * nseg, N, T, GT,
                                     G_SIZE, R, O, True, mma_io,
                                     pair_half, x_scaled.dtype,
                                     precision), T=T)
        _fused_chunk_fwd_kernel[grid_a](
            *head_args, dummy,
            *mid_args,
            dummy_y, dummy_s, psi,
            *stride_args,
            *dummy_y.stride()[:4], dummy_y.stride(5),
            dummy_y.stride(4),
            *dummy_s.stride(),
            psi.stride(0), psi.stride(1), psi.stride(3), psi.stride(4),
            0, 0, 0, 0,
            *tail_strides,
            nc, nc_seg,
            0,                # s_x0s
            psi.stride(2),    # s_fss: per-segment final-state store
            HAS_X0=False,
            STORE_SALL=False,
            STATE_ONLY=True,
            num_warps=wa, num_stages=sa,
            **common_kw,
        )
        # ── Phase B: one-launch stitch (linearity: S = alpha*S + G) ──
        alpha_seg = _seg_alpha(a_log, delta_cumsum, nseg, nc_seg,
                               dcs_ink=dcs_ink)
        s_entry = _seg_stitch(psi, alpha_seg,
                              x0_c.contiguous() if has_x0 else None,
                              reverse=False)
        # ── Phase C: normal kernel per segment, seeded via x0 path ──
        x0_c = s_entry
        x0_st = [s_entry.stride(0), s_entry.stride(1),
                 s_entry.stride(3), s_entry.stride(4)]
        x0_seg_stride = s_entry.stride(2)
        has_x0 = True
    else:
        x0_seg_stride = 0

    grid = (Bsz, H, NGT * nseg)
    _fwd_cell = _fwd_cell_key(Bsz * H * NGT * nseg, N, T, GT, G_SIZE, R, O,
                              state_only, mma_io, pair_half,
                              x_scaled.dtype, precision)
    # The two-tuple form stays the call the launch site makes: it is the
    # hook the guard tests monkeypatch to prove the rule is CONSULTED
    # (a table that stopped being consulted would pass silently), and
    # `_fwd_launch_cfg` is itself one line of `_fwd_launch_cfg_full`.
    wc, sc = _fwd_launch_cfg(device, Bsz * H * NGT * nseg, N,
                             state_only=state_only, gt=GT, mma_io=mma_io,
                             pair_half=pair_half, cell=_fwd_cell, T=T)
    _, _, mnrc, _cfg_src = _fwd_launch_cfg_full(
        device, Bsz * H * NGT * nseg, N, state_only=state_only, gt=GT,
        mma_io=mma_io, pair_half=pair_half, cell=_fwd_cell, T=T)
    main_args = (
        *head_args, x0_c,
        *mid_args,
        Y, S_all, S_final,
        *stride_args,
        *Y.stride()[:4], Y.stride(5),
        Y.stride(4),
        *S_all.stride(),
        *S_final.stride(),
        *x0_st,
        *tail_strides,
        nc, nc_seg,
        x0_seg_stride,
        0,                    # s_fss: single final state (last segment)
    )
    main_kw = dict(
        HAS_X0=has_x0,
        STORE_SALL=store_sall,
        STATE_ONLY=state_only,
        num_warps=wc, num_stages=sc,
        **({"maxnreg": mnrc} if mnrc else {}),
        **ep_kw,
        **common_kw,
    )

    # ── PROTOTYPE, opt-in only: the Hopper Gluon twin ───────────────────
    # `SSD_MIMO_GLUON90=1` (env only; see gluon_sm90_fwd's STATUS) routes the
    # main launch of an in-envelope sm_90 cell to `gluon_sm90_fwd`; every
    # refusal is by name and falls through to the Triton launch below,
    # unchanged.  Off by default: this block then costs one env read.
    if (_env("GLUON90", "off").lower() in ("1", "on", "true")
            and _portable.gluon_refusal("ssd_mimo") is None):         # kernel_target="portable": no Gluon
        from . import gluon_sm90_fwd as _g90
        _why = _g90.refusal(
            device=device, T=T, N=N, G_SIZE=G_SIZE, GT=GT, R=R, O=O,
            mma_io=mma_io, io_dtype=x_scaled.dtype,
            has_theta_h=has_theta_h, has_rot=has_rot,
            has_rot_bias=has_rot_bias, pair_half=bool(pair_half),
            has_gate=has_gate, has_dskip=has_dskip, nseg=nseg,
            idx64=idx64, Btil=Btil, Ctil=Ctil, x=x_scaled,
            s_bt=s_bt, s_bn=s_bn, s_ct=s_ct, s_cn=s_cn,
            state_only=state_only, bias_pre_rot=bias_pre_rot,
            has_beta=has_beta, has_gamma=has_gamma,
            phase_cumsum=phase_cumsum, theta=th_c if has_rot else None)
        if _why is None and dw is None:
            _why = "needs the raw-x write scale (dw)"
        _g90_key = ("fwd", bool(state_only), has_rot, has_theta_h)
        if _why is None:
            _why = _g90.failed_refusal(_g90_key)
        _g90._LAST_ROUTE.clear()
        _g90._LAST_ROUTE.update(used=_why is None, why=_why or "",
                                state_only=bool(state_only))
        # A compile / launch error falls back to the
        # Triton launch below with a named route reason (warned once)
        # instead of aborting the step.
        if _why is None:
          try:
            _g90.launch(
                Ctil=Ctil, Btil=Btil, x=x_scaled, delta=delta_cumsum,
                a_log=a_log, x0_c=x0_c, x0_st=x0_st,
                x0_seg_stride=x0_seg_stride, beta_c=beta_c,
                gamma_c=gamma_c, has_x0=has_x0, has_beta=has_beta,
                has_gamma=has_gamma, Y=Y, S_all=S_all, S_final=S_final,
                y_strides=(*Y.stride()[:4], Y.stride(5), Y.stride(4)),
                sall_strides=tuple(S_all.stride()),
                sf_strides=tuple(S_final.stride()),
                dw_kw=dw_kw, trap_kw=trap_kw, H=H, T=T, N=N,
                G_SIZE=G_SIZE, GT=GT, R=R, O=O, hpg_b=hpg_b, hpg_c=hpg_c,
                nc=nc, nc_seg=nc_seg, s_fss=0, grid=grid,
                store_sall=bool(store_sall), state_only=bool(state_only),
                store_y=bool(store_y), dcs_ink=bool(dcs_ink),
                b_strides=(s_bb, s_bc, s_bh, s_bt, s_bn, s_br),
                c_strides=(s_cb, s_cc, s_ch, s_ct, s_cn, s_co),
                theta=th_c if has_rot else None, th_strides=th_st,
                theta_h=thh_c if has_theta_h else None, idx64=bool(idx64))
            return (Yg if has_gate else Y), S_all, S_final
          except Exception as _e:                         # noqa: BLE001
            reraise_fatal(_e)          # OOM / CUDA error: propagate, never cached as a refusal
            _g90._LAST_ROUTE.update(used=False,
                                    why=_g90.note_failure(_g90_key, _e))

    # ── Shared-memory pre-check ───────────────────────────────────────
    # `warmup` COMPILES the exact variant the launch below will use and
    # returns it WITHOUT launching, so `metadata.shared` is the same number
    # Triton's launcher would compare against the device budget — and the
    # compile lands in the JIT cache the launch then hits, so the check is
    # free.  Raising here (instead of letting the launcher raise
    # OutOfResources) lets the entry point retry at a smaller chunk.  See
    # smem.py for the clamp rule.
    # The pipeline depth is lowered to fit BEFORE the refusal
    # (an exact dry compile per rung; see `_fwd_fit_stages`).
    main_kw = _fwd_fit_stages(main_args, main_kw, grid, device)

    # ── The STACK ladder.  BETWEEN the two depth fits, and both
    # fits are needed.  Before it, because a stacked body that does not
    # FIT at the heuristic's depth cannot be PROFILED there either
    # (`_init_handles` raises on an over-budget module), and an
    # unprofilable body would silently skip the ladder.  After it,
    # because the twin it
    # switches to has its own shared-memory request and must be fitted
    # in its own right.  `_fwd_fit_stages` is idempotent and its
    # compiles are cached, so the second call is free.
    main_kw = _fwd_fix_stack(main_args, main_kw, grid, device)

    main_kw = _fwd_fit_stages(main_args, main_kw, grid, device)

    _check_fwd_smem(main_args, main_kw, grid, T, N, device)

    # ── The SPILL ladder (the upward twin of `_fwd_fit_stages`)
    # After the refusal, so a cell that does not fit at all still reaches
    # the chunk ladder first; before the measured search, so a measured
    # verdict still wins.  No-op unless ptxas reports the body spilling.
    main_kw = _fwd_fix_spills(main_args, main_kw, grid, device,
                              io_fp32=x_scaled.dtype == torch.float32)

    # ── The measure-once forward launch-config autotune ───────────────
    # No-op unless `SSD_MIMO_FWD_CFG=measure` AND this cell has no cached
    # verdict yet; when it fires it times the pruned candidate grid on
    # THESE operands and returns the winner's kw.  Placed after the
    # shared-memory pre-check so the chunk clamp still gets first refusal
    # on the config the heuristic picked.
    main_kw = _fwd_autotune(main_args, main_kw, grid, device, _fwd_cell,
                            *_fwd_cfg_candidates(
                                device, Bsz * H * NGT * nseg, N,
                                state_only=state_only, gt=GT,
                                mma_io=mma_io, pair_half=pair_half, T=T),
                            source=_cfg_src,
                            outs=(Y, S_final,
                                  Yg if has_gate else None))

    _fused_chunk_fwd_kernel[grid](*main_args, **main_kw)

    return (Yg if has_gate else Y), S_all, S_final


# ── The measure-once forward launch-config autotune ──────────────────
# The last forward launch config chosen, for `dispatch_log()` and for a
# guard that wants to see WHICH config a call took (timing cannot answer
# that on a shared GPU).
_LAST_FWD_CFG: dict = {}


def last_fwd_cfg() -> dict:
    """The last fused-forward launch config: `warps`, `stages`,
    `maxnreg`, `source` ("heuristic" / "measured" / "env"), `cell`,
    `spill_fix` (the spill-ladder depth), `stack_fix` (the (o, r)-stack
    ladder) and `stack_guard` ("" or `"sm90_io_ro_stack"`,
    see `ro_stack_wgmma_fault`)."""
    return dict(_LAST_FWD_CFG)


def _fwd_out_checksum(outs):
    """A cheap fingerprint of a launch's outputs, for the autotune's
    correctness gate.

    Sum + max|.| + the count of non-finite entries, per output tensor,
    in fp64 so the reduction itself does not round two different answers
    together.  NOT a proof of bit-identity -- a full `torch.equal`
    against a clone of `S_all` would be gigabytes at large cells -- but
    it catches the failure this gate exists for: a mis-lowering
    returns a ~100 %-wrong tile, which moves both moments by
    orders of magnitude, or a NaN, which moves the third.
    """
    out = []
    for t in outs:
        if t is None or not torch.is_tensor(t) or t.numel() == 0:
            out.append((0.0, 0.0, 0))
            continue
        f = t.detach().to(torch.float64)
        out.append((float(f.sum()), float(f.abs().max()),
                    int((~torch.isfinite(f)).sum())))
    return tuple(out)


def _fwd_autotune(args, kw, grid, device, cell, primary, mnr_axis,
                  source="heuristic", outs=()):
    """Time the pruned candidate grid ONCE for `cell`; return the winner's kw.

    A no-op unless `SSD_MIMO_FWD_CFG=measure` and the cell has no cached
    verdict; `measuring_possible()` is honoured for the same reason the
    chunk and smem autotunes honour it (a CUDA-graph capture cannot take
    the event timing and the synchronize).

    METHOD -- deliberately the same as the chunk `"measure"` mode's:
    `smem._time_call` runs one WARM-UP call (which pays the Triton
    compile, so it is not attributed to the candidate) and then times
    `iters` calls between CUDA events.  Re-launching the forward on the
    caller's own operands is safe because the fused forward is
    IDEMPOTENT: every output it touches -- `Y`, `S_all`, `S_final`, the
    gate / d-skip epilogue buffers, the trapezoid `dot_out` -- is a plain
    store, never an `atomic_add` (the atomics in this family all live in
    the backward's epilogues).  A candidate therefore overwrites the
    previous candidate's answer with the same bits, and the final launch
    below re-writes them once more.

    Each candidate is DRY-COMPILED first and disqualified when its shared
    request exceeds the device budget, so an over-budget config is never
    launched and never raises where the heuristic's config fitted.  Any
    trial that raises is disqualified too, not propagated: the config
    that was going to run is still in `kw`.

    The search is two-phase -- (warps, stages), then the register cap
    around the winner -- which is <= 9 timed trials instead of 24.
    """
    from .smem import (_time_call, device_smem_limit, fwd_cfg_key,
                       fwd_cfg_lookup, fwd_cfg_mode, fwd_cfg_store,
                       measuring_possible)

    def _note(cfg, src):
        # `global` in the ENCLOSING function does not reach a nested one.
        global _LAST_FWD_CFG
        _LAST_FWD_CFG = dict(warps=int(cfg[0]), stages=int(cfg[1]),
                             maxnreg=int(cfg[2] or 0), source=src,
                             cell=cell, spill_fix=_LAST_SPILL_FIX,
                             stack_fix=_LAST_STACK_FIX,
                             stack_guard=_LAST_STACK_GUARD)

    running = (int(kw["num_warps"]), int(kw["num_stages"]),
               int(kw.get("maxnreg", 0) or 0))
    _note(running, source)
    if cell is None or fwd_cfg_mode() != "measure":
        return kw
    key = fwd_cfg_key(cell, device)
    if fwd_cfg_lookup(key) is not None or not measuring_possible():
        return kw
    limit = device_smem_limit(device)
    iters = max(1, int(_env("FWD_CFG_ITERS", "3")))

    ref = [None]        # the reference checksum, taken on the 1st config

    def _trial(cfg):
        k2 = dict(kw)
        k2["num_warps"], k2["num_stages"] = int(cfg[0]), int(cfg[1])
        k2.pop("maxnreg", None)
        if cfg[2]:
            k2["maxnreg"] = int(cfg[2])
        try:
            req = _fwd_smem_required(args, k2, grid)
            if req is not None and req > limit:
                return None, k2
            ms = _time_call(
                lambda: _fused_chunk_fwd_kernel[grid](*args, **k2),
                iters)
            # ── THE CORRECTNESS GATE ─────────────────────────────────
            # The construction argument says every candidate computes the
            # same function.  This CHECKS it, because the argument covers
            # the math and not the lowering.  The
            # first candidate timed is the heuristic's own config (the
            # generator puts it first), so its fingerprint is the
            # reference; a candidate that moves it is DISQUALIFIED, not
            # merely ranked lower.
            if outs:
                sig = _fwd_out_checksum(outs)
                if ref[0] is None:
                    ref[0] = sig
                elif sig != ref[0]:
                    warnings.warn(
                        "SSD-MIMO forward autotune: config "
                        "(warps=%d, stages=%d, maxnreg=%d) changed the "
                        "output fingerprint and was DISQUALIFIED; the "
                        "shipped config stands.  This is a Triton "
                        "lowering fault, not a tuning result -- please "
                        "report it with the cell key %r."
                        % (cfg[0], cfg[1], cfg[2] or 0, cell),
                        RuntimeWarning, stacklevel=2)
                    return None, k2
            return ms, k2
        except Exception as _e:      # noqa: BLE001 — disqualify, not fail
            reraise_fatal(_e)
            return None, k2

    timings, kws = {}, {}
    for cfg in primary:
        t, k2 = _trial(cfg)
        if t is not None:
            timings[cfg], kws[cfg] = t, k2
    if not timings:
        return kw
    best = min(timings, key=timings.get)
    for m in mnr_axis:
        cfg = (best[0], best[1], int(m))
        if cfg in timings:
            continue
        t, k2 = _trial(cfg)
        if t is not None:
            timings[cfg], kws[cfg] = t, k2
    best = min(timings, key=timings.get)
    fwd_cfg_store(key, best,
                  {"%d,%d,%d" % c: v for c, v in timings.items()})
    _note(best, "measured")
    return kws[best]


# ── The SPILL ladder, the upward twin of ─────────────────────────────
# `_fwd_fit_stages`.
#
# THE DEFECT IT REPAIRS.  At the S-row cell (R = O = 1, N = 128, G = 16 ->
# GT = 16) the ROTATION-FREE forward runs 6-14x slower than the ROTATING
# body at the SAME shape.  The cause is not the plan and not the layout
# (`dispatch_log()` is identical on both arms).  It is ptxas.  At ONE
# pipeline stage the rotation-free body compiles to **32 registers and
# ~1.4 KB of per-thread local memory**; the rotating body, same launch
# config, gets 255 registers and 162 bytes.  At THREE stages the
# rotation-free body gets 255 registers and 28 bytes.  Same shape, same
# math, same grid: a register-allocator cliff that only the compiler can
# see.
#
# WHY THE EXISTING MECHANISMS DO NOT CATCH IT.
#   * `_fwd_heuristic_cfg`'s register clause CANNOT fire here: its only
#     lever is 8 warps, and `_mmav3_eight_warp_ok` refuses 8 warps at a
#     G tile under 128 BYTES (GT = 16 at bf16 is 32).  The cell is
#     pinned at 4 warps by CORRECTNESS.
#   * its stages clause takes 3 only under ONE WAVE; this grid is 960
#     CTAs over 108 SMs, so it takes 1 -- which is the right answer for
#     shared-memory residency and the wrong one for this body.
#   * the measure-once search (`_fwd_cfg_candidates`) is REFUSED
#     ENTIRELY below the 128-byte G-tile law, so `SSD_MIMO_FWD_CFG=
#     measure` cannot find the config either.
#
# THE REPAIR is the construction `_fwd_fit_stages` already blesses, run
# the other way: no ESTIMATE of the register profile --
# `JITFunction.warmup` + `_init_handles` gives ptxas's OWN
# `n_regs`/`n_spills` for the exact variant, and that compile lands in
# the JIT cache the real launch then hits.  So: when the compiler
# reports catastrophic spilling, dry-compile the deeper pipelines, keep
# the shallowest one that cuts the spill by `_FWD_SPILL_GAIN`, and only
# ever among depths that FIT the device's shared budget (the deeper
# pipeline is paid for in shared memory, so this ladder and the
# downward one can never disagree -- a depth `_fwd_fit_stages` rejected
# is rejected here by the same comparison).
#
# It is device-agnostic in the sense of: no compute
# capability appears, the numbers come from the device and from ptxas,
# and `num_stages` is launch metadata that reaches no constexpr the
# kernel branches on -- the same correctness-equivalence argument the
# measured search runs on.
#
# `SSD_MIMO_FWD_SPILL_FIX=0` is the A/B arm; `dispatch_log()` reports
# `fwd_spill_fix` = `(stages_before, spills_before, spills_after)` on
# every launch the ladder moved, and `None` on every launch it did not.

#: Bytes of per-thread LOCAL memory (ptxas spill storage) at which the
#: ladder engages.  Healthy readings are <= ~160 B and broken ones
#: >= ~1.3 KB; a quarter of a KB per thread separates them.
_FWD_SPILL_FLOOR = 256

#: The factor by which a deeper pipeline must CUT the spill before it is
#: taken.  The cliff is not a gradient (e.g. 1372 -> 28 B, 49x), so a 4x
#: bar takes the cliff and refuses a wobble.
_FWD_SPILL_GAIN = 4.0

#: The same bar for the STACKED (o, r) body (`_fwd_fix_stack`).  A 4x bar
#: can keep a stack that LOSES (e.g. sm_120: stacked 372 B vs unstacked
#: 130 B, only 2.9x less spill, yet the unstacked body is faster), while
#: the two bodies at the cells where the stack really fails sit far apart
#: (A100 1492 vs 66 B), so 2x separates the measured parts the way their
#: timings do.
#: `SSD_MIMO_FWD_STACK_SPILL_GAIN` overrides.
_FWD_STACK_SPILL_GAIN = float(_env("FWD_STACK_SPILL_GAIN", "2.0"))

#: ptxas's (n_regs, n_spills) per compile key.
_FWD_PROFILE: dict = {}

#: The last spill-ladder verdict, for `dispatch_log()`.  Reset on every
#: launch that reaches the ladder, so a stale record cannot be reported
#: against a launch that did not move.
_LAST_SPILL_FIX = None


def _fwd_profile_key(kw, args):
    return (tuple(sorted((k, str(v)) for k, v in kw.items()
                         if isinstance(v, (int, bool, str, tl.dtype)))),
            tuple(a.dtype for a in args[:4] if torch.is_tensor(a)))


def _fwd_kernel_profile(args, kw, grid):
    """ptxas's `(n_regs, n_spills)` for the exact variant, or None.

    `n_spills` is the CUDA driver's `LOCAL_SIZE_BYTES` for the loaded
    function, i.e. bytes of per-thread local memory -- spill storage.
    `_init_handles` is what fills both fields in, and it is the same
    module load the launch performs; it raises when the variant does not
    fit shared memory, which is a "cannot measure", not a verdict.
    """
    key = _fwd_profile_key(kw, args)
    if key not in _FWD_PROFILE:
        try:
            compiled = _fused_chunk_fwd_kernel.warmup(*args, grid=grid, **kw)
            compiled._init_handles()
            _FWD_PROFILE[key] = (int(compiled.n_regs), int(compiled.n_spills))
        except Exception as _e:  # noqa: BLE001 — never break a good launch
            reraise_fatal(_e)
            _FWD_PROFILE[key] = None
    return _FWD_PROFILE[key]


#: The last stack-ladder verdict, for `dispatch_log()`.  Reset on every
#: launch that reaches the ladder.
_LAST_STACK_FIX = None


def _fwd_fix_stack(args, kw, grid, device):
    """Drop the stacked (o, r) body when ptxas SPILLS it.

    The spill-ladder construction (`_fwd_fix_spills`)
    applied to the `RO_STACK` constexpr instead of to `num_stages`, and
    for the same reason: the quantity that decides is what the compiler
    makes of the body, and no arithmetic on the tile sizes predicts it
    across parts.

    The stacked body folds `o` into the score tile's rows and `r` into
    its columns, so besides the `[T*OS, T*RS]` accumulator it holds
    `B_st [T*RS, N]`, `C_st [T*OS, N]` and (under the trapezoid) a
    `B_rep [T*OS, N]` replica where the per-(o, r) body holds one
    `[T, N]` tile at a time.  `_ro_stack`'s `64 // T` cap prices the
    accumulator and not those, and at N >= 128 they can take the body
    over the register file -- ON SOME PARTS.  At the S-row
    R = O = 2, N = 128, G = 16 cell, ptxas `(n_regs, n_spills)`:

        A100  stacked (32, 1492)   unstacked (255, 66)    94.42 vs 6.79 ms
        H100  stacked (255, 114)   unstacked (255, 64)     3.80 vs 5.20 ms

    -- the SAME shape, a 13.9x loss on sm_80 and a 1.37x win on sm_90.
    An estimate cannot separate those; `_FWD_SPILL_FLOOR` (256 B/thread)
    separates the two cleanly.

    So: when the compiler reports the STACKED body spilling at or above
    `_FWD_SPILL_FLOOR`, dry-compile the unstacked twin and take it if it
    cuts the spill by `_FWD_STACK_SPILL_GAIN` (2x).  The two bodies compute the
    same sums with `r` inside the contraction instead of outside it, so this is
    a SCHEDULE choice, not a correctness one -- which is why it is
    decided before `_check_fwd_smem` (the two bodies ask for different
    shared memory) rather than after it.

    No-op when the stack is not selected, when the caller PINNED the
    lever (`SSD_MIMO_FWD_RO_STACK` anything but "auto" wins, §1.5), when
    the profile cannot be measured, or when the clamp is off
    (`SSD_MIMO_SMEM_CLAMP=0` is the native resource probe).  A/B:
    `SSD_MIMO_FWD_SPILL_FIX=0`, the same lever as the depth ladder -- it
    is the same construction.  Reported as `dispatch_log()`'s
    `fwd_stack_fix` = `(spills_stacked, spills_unstacked)`, `None`
    otherwise.
    """
    from .smem import smem_clamp_enabled
    global _LAST_STACK_FIX, _LAST_RO_STACK
    _LAST_STACK_FIX = None
    if not kw.get("RO_STACK"):
        return kw
    if (not smem_clamp_enabled() or _env("FWD_SPILL_FIX", "1") == "0"
            or _env("FWD_RO_STACK", "auto") != "auto"):
        return kw
    base = _fwd_kernel_profile(args, kw, grid)
    if base is None or base[1] < _FWD_SPILL_FLOOR:
        return kw
    k2 = dict(kw, RO_STACK=False, OS=1, RS=1)
    prof = _fwd_kernel_profile(args, k2, grid)
    if prof is None or prof[1] * _FWD_STACK_SPILL_GAIN > base[1]:
        return kw
    _LAST_STACK_FIX = (base[1], prof[1])
    _LAST_RO_STACK = (False, 1, 1)
    return k2


def _fwd_fix_spills(args, kw, grid, device, io_fp32=False):
    """RAISE `num_stages` while ptxas is spilling the body into local memory.

    Returns `kw` unchanged when the compiler reports a healthy body, when
    no deeper pipeline fits the shared budget, when no deeper pipeline
    helps, when the profile cannot be measured, when the caller PINNED
    the depth (`SSD_MIMO_FWD_STAGES` wins over every rung, §1.13), or
    when the clamp is off (`SSD_MIMO_SMEM_CLAMP=0` is the native
    resource probe: it must see the unrepaired launch, exactly as
    `_fwd_fit_stages` leaves it alone).
    """
    from .smem import device_smem_limit, smem_clamp_enabled
    global _LAST_SPILL_FIX
    _LAST_SPILL_FIX = None
    if (not smem_clamp_enabled() or _env("FWD_SPILL_FIX", "1") == "0"
            or _env("FWD_STAGES", "")):
        return kw
    start = int(kw.get("num_stages", 1) or 1)
    fp32_ops = bool(io_fp32) and kw.get("MMA_DT") == tl.float32     # fp32 IO with fp32 operands (the collapse rung)
    if start >= max(_FWD_STAGE_CANDIDATES) and not fp32_ops:
        return kw
    base = _fwd_kernel_profile(args, kw, grid)
    if base is None or base[1] < _FWD_SPILL_FLOOR:
        return kw
    limit = device_smem_limit(device)
    best_kw, best = kw, base
    for st in range(start + 1, max(_FWD_STAGE_CANDIDATES) + 1):
        k2 = dict(kw, num_stages=st)
        req = _fwd_smem_required(args, k2, grid)
        if req is None or req > limit:
            break
        prof = _fwd_kernel_profile(args, k2, grid)
        if prof is None:
            continue
        if prof[1] * _FWD_SPILL_GAIN <= base[1] and prof[1] < best[1]:
            best_kw, best = k2, prof
            break               # shallowest depth that clears the cliff
    if best_kw is kw and fp32_ops and _fwd_collapsed(base) and not _env("FWD_WARPS", ""):
        best_kw, best = _fwd_fix_collapse(args, kw, grid, limit, base, device)
    if best_kw is kw:
        return kw
    _LAST_SPILL_FIX = (start, base[1], best[1])
    return best_kw


#: The REGISTER-COLLAPSE rung of the spill ladder, fp32 MMA operands only.  With fp32 IO (tf32 dots:
#: torch's default dtype) the fused forward's tiles are twice as wide and ptxas falls off the cliff `_FWD_SPILL_FLOOR` describes
#: -- a COLLAPSED allocation: a few dozen registers and kilobytes of local memory -- at configs the depth rung cannot repair.
#: E.g. (fp32 IO, tf32 operands, N128 G64) on A100: 8 warps -> 32 regs / ~2.2 KB at 1-3 stages, while 4 warps x 1 -> 255 / 52 B;
#: on Blackwell 8 x 2 -> 255 regs / 446 B is a 3.9x cut, just under the depth rung's 4x bar, so the depth rung keeps the collapsed body.
#: A spill COUNT does not separate the two regimes (910 B collapsed vs 446 B healthy); the REGISTER count does: every
#: collapsed body sits at <= 40 registers, every healthy one at >= 128 (`_fwd_collapsed`).  So, when the depth rung found
#: nothing and the body is collapsed, the admissible warp counts (8 only where `_mmav3_eight_warp_ok` admits it) are
#: dry-compiled at every depth that fits and the NON-collapsed variant with the least local memory is taken.  Restricted to
#: fp32 operands (`MMA_DT`): no 16-bit launch changes.  `SSD_MIMO_FWD_WARPS` pins the warp count (no rung).
_FWD_COLLAPSE_REGS = 128


def _fwd_collapsed(prof) -> bool:
    """ptxas's profile `(n_regs, n_spills)` is a collapsed allocation (see `_FWD_COLLAPSE_REGS`)."""
    return prof is not None and prof[1] >= _FWD_SPILL_FLOOR and prof[0] < _FWD_COLLAPSE_REGS


def _fwd_fix_collapse(args, kw, grid, limit, base, device):
    """(kw, profile) of the least-spilling NON-collapsed (warps, stages) among the admissible warp counts, or (kw, base)."""
    gt, mma_io = int(kw.get("GT", 0) or 0), kw.get("MMA_DT") in (tl.bfloat16, tl.float16)
    cands = [w for w in (4, 8) if w < 8 or gt == 0 or _mmav3_eight_warp_ok(gt, mma_io)]
    # the explicit register cap: ptxas without one falls into the collapse, with the hardware maximum (255) it allocates
    # normally (e.g. 255 regs / 0 B at 4 x 1 against 32 regs / ~1.2 KB uncapped at every depth)
    caps = (None, 255) if _dev_caps(device)["maxnreg_ok"] else (None,)
    best_kw, best = kw, None
    for mnr in caps:
        for w in cands:
            for st in _FWD_STAGE_CANDIDATES:
                k2 = dict(kw, num_warps=w, num_stages=st)
                if mnr:
                    k2["maxnreg"] = mnr
                req = _fwd_smem_required(args, k2, grid)
                if req is None or req > limit:
                    break
                prof = _fwd_kernel_profile(args, k2, grid)
                if prof is None or _fwd_collapsed(prof):
                    continue
                if best is None or prof[1] < best[1]:
                    best_kw, best = k2, prof
        if best is not None:
            break                       # the compiler's own allocation when one admissible config is healthy
    return (best_kw, best) if best is not None else (kw, base)


# Measured shared-memory request per compile key.  `warmup` is a full
# binder pass, so repeating it on every launch would double the fused
# forward's host overhead; the answer depends only on the constexprs +
# operand dtypes, so one measurement per key per process is enough.  The
# BYTES are cached, not the verdict, so `SSD_MIMO_SMEM_LIMIT` keeps taking
# effect live.
_SMEM_REQUIRED: dict = {}


def _fwd_smem_required(args, kw, grid) -> int | None:
    """Bytes of shared memory per CTA the fused forward compiles to.

    Measured by DRY-COMPILING the exact variant: `JITFunction.warmup`
    compiles and returns the `CompiledKernel` without launching it, and
    `metadata.shared` is the same figure Triton's launcher compares
    against the device budget (verified: it equals the "Required:" number
    in the `OutOfResources` message at every over-budget cell).  The
    compile lands in the JIT cache the real launch then hits, so the
    measurement costs one compile that was going to happen anyway.

    Returns None when the dry compile cannot be done at all — the caller
    then proceeds exactly as before and Triton's launcher stays the
    backstop.
    """
    key = (
        # `MMA_DT` is a tl.dtype, not an int/bool/str, so it is
        # stringified INTO the key -- the two operand modes compile to
        # different shared-memory requests and must not share a cache entry.
        tuple(sorted((k, str(v)) for k, v in kw.items()
                     if isinstance(v, (int, bool, str, tl.dtype)))),
        tuple(a.dtype for a in args[:4] if torch.is_tensor(a)),
    )
    if key not in _SMEM_REQUIRED:
        try:
            compiled = _fused_chunk_fwd_kernel.warmup(*args, grid=grid, **kw)
            _SMEM_REQUIRED[key] = int(compiled.metadata.shared)
        except Exception as _e:  # noqa: BLE001 — never break a good launch
            reraise_fatal(_e)
            _SMEM_REQUIRED[key] = None
    return _SMEM_REQUIRED[key]


def _check_fwd_smem(args, kw, grid, T, N, device) -> None:
    """Refuse BEFORE the launch when the forward will not fit.

    Silent no-op when the clamp is disabled (`SSD_MIMO_SMEM_CLAMP=0`) or
    when the requirement cannot be measured.
    """
    from .smem import (SsdSmemOverflow, device_smem_limit,
                       smem_clamp_enabled)
    if not smem_clamp_enabled():
        return
    required = _fwd_smem_required(args, kw, grid)
    if required is None:
        return
    limit = device_smem_limit(device)
    if required > limit:
        raise SsdSmemOverflow(required, limit, N, T)
