"""Helpers shared by the SSD-MIMO kernel implementations."""

import os

import triton
import triton.language as tl


def reraise_fatal(exc: BaseException) -> None:
    """Re-raise `exc` when it is not a "this variant does not compile / fit" verdict.

    The Gluon / fused fallback sites catch broad exceptions so that a variant that cannot
    be compiled, lowered or fitted takes the next route, and several CACHE that verdict
    (a failed-route refusal, a None profile).  An out-of-memory error or a CUDA
    (accelerator) error is neither: it says the device is full or the context is broken,
    so it must reach the caller and must never be cached as "route unavailable".  Call it
    first in such an ``except`` block."""
    import torch
    fatal = (torch.OutOfMemoryError,) + tuple(
        t for t in (getattr(torch, "AcceleratorError", None),) if t is not None)
    if isinstance(exc, fatal):
        raise exc


def nan_debug_fill(t):
    """`SSD_MIMO_DEBUG_NANFILL=1`: fill a `torch.empty` gradient buffer with NaN before the kernels run.

    Used for every buffer whose zero-fill was dropped because a kernel stores every element (sole-owner
    store, no accumulation): a NaN left in the buffer after the launch is an unwritten element, i.e. the
    proof obligation of the drop failed.  Read per call (a debug switch, not a constexpr); returns `t`.
    Floating buffers only (the ssd gradients are all fp32 / 16-bit floats)."""
    if os.environ.get("SSD_MIMO_DEBUG_NANFILL", "0") == "1" and t.is_floating_point():
        t.fill_(float("nan"))
    return t


# Memory ordering of the gradient ACCUMULATION atomics.  `tl.atomic_add` defaults to `sem="acq_rel"`,
# which is `atom.global.gpu.acq_rel` in PTX: ptxas brackets every one with
# a GPU-scope fence (sm_90: a `MEMBAR.ALL.GPU` before each atomic), i.e.
# every accumulation also orders ALL of the CTA's earlier memory traffic.
# Nothing here needs that: every atomic in this package is a pure
# accumulation -- return value unused, target never read by the launch
# that accumulates into it (the partial sums are consumed after the
# kernel boundary, which orders them anyway).  "relaxed" keeps the
# atomicity (no lost update, same set of fp32 additions, only their
# order may differ, as it already does) and drops the fences.
# SSD_MIMO_ATOMIC_SEM=acq_rel selects Triton's default.  Read ONCE at
# import: it is a constexpr global baked into every kernel that uses it.
def _acc_sem() -> str:
    v = os.environ.get("SSD_MIMO_ATOMIC_SEM", "relaxed").strip().lower()
    if v not in ("relaxed", "acq_rel"):
        raise ValueError(
            f"SSD_MIMO_ATOMIC_SEM={v!r}: expected 'relaxed' or 'acq_rel'")
    return v


ACC_SEM = tl.constexpr(_acc_sem())


def _ptx_trig_backend() -> bool:
    """True when Triton's ACTIVE target can take inline PTX (`cos.approx.f32`
    / `sin.approx.f32`) -- an NVIDIA/CUDA backend.  False on HIP/ROCm (no
    PTX, `cos.approx.f32` is not a valid AMDGCN instruction) and False when
    there is no active driver at all (CPU-only import, docs build, etc.),
    which is the "no PTX" half of the flag below.  Read ONCE at import time
    -- the backend cannot change mid-process -- and baked into `_cos_approx`
    / `_sin_approx`'s `USE_PTX` default so every existing call site (every
    rotation-bearing kernel in this package) keeps working unchanged: NONE
    of them need to pass the flag for the shipped NVIDIA path to stay
    bit-identical.
    """
    try:
        from triton.runtime import driver
        return driver.active.get_current_target().backend == "cuda"
    except Exception:                      # noqa: BLE001 — no active driver
        return False


#: Baked once at import time (see `_ptx_trig_backend`).  `USE_PTX=False`
#: (explicit, at the call site) is also how the two paths are compared to
#: each other on the SAME (NVIDIA) machine -- see
#: the trig-approximation fallback test (both backends on one NVIDIA machine).
_USE_PTX_TRIG_DEFAULT = _ptx_trig_backend()

#: The PROCESS-level choice read inside `_cos_approx` / `_sin_approx` / `pdot` / `_opv`: True = NVIDIA-only constructs are allowed
#: (PTX trig, the cvt.rna.tf32 operand rounding, the tf32 dot literal: the NVIDIA default), False = libdevice trig and ieee dots only.  A module-level constexpr: Triton hashes it into every kernel that
#: calls the helpers and raises if it changes after one compiled.  `kernels.portable.latch_portable` flips it
#: to False (kernel_target="portable"), before the first compile and never after (`trig_compiled`).
_TRIG_PTX = tl.constexpr(_USE_PTX_TRIG_DEFAULT)


def trig_compiled() -> bool:
    """True once a kernel using `_cos_approx` / `_sin_approx` has been compiled in this process (Triton then
    holds `_TRIG_PTX` fixed)."""
    return any(bool(getattr(f, "used_global_vals", None)) for f in (_cos_approx, _sin_approx, pdot))     # the interpreter compiles nothing


@triton.jit
def _wrap2pi(x):
    """x reduced to [-pi, pi] (Cody-Waite, two constants): x - 2 pi k.

    The model only ever uses cos / sin of the
    running phase, so the reduction is exact math; numerically k * 6.28125
    is exact for |k| < 2^16 (|x| < 4e5) and the k * (2 pi - 6.28125) term
    costs ~1e-7 |k| rad.  The SFU approximations are accurate only near
    [-pi, pi], and a trained phase reaches |Theta| ~ 4e5 rad."""
    k = tl.floor(x * 0.15915494309189535 + 0.5)
    return (x - k * 6.28125) - k * 0.0019353071795864769


@triton.jit
def _cos_approx(x, USE_PTX: tl.constexpr = _USE_PTX_TRIG_DEFAULT):
    """Hardware SFU cosine (PTX cos.approx.f32) on NVIDIA; `tl.cos` elsewhere.

    Triton's tl.cos lowers to __nv_cosf -- libdevice, with
    __internal_trig_reduction_kernel AND a slowpath (verified by dumping
    the PTX).  Mamba-3 hand-rolls this same inline-asm for the same
    reason.  Safe here because the phase is wrapped to [0, 2pi) upstream,
    which is where the approximation is accurate.

    `USE_PTX` (constexpr) defaults to `_USE_PTX_TRIG_DEFAULT`: True on a
    CUDA target (unchanged, bit-identical PTX path), False on HIP/ROCm or
    when there is no active PTX-capable driver -- the portability fallback,
    which routes to `tl.cos` (libdevice on every backend Triton supports,
    including HIP).  Passing it explicitly (either value) overrides the
    autodetected default, which is how the two paths are compared on one
    machine in the test.
    """
    if USE_PTX:
        if _TRIG_PTX:       # False under kernel_target="portable" (kernels/portable.py): libdevice, no inline asm
            # PHASEWRAP: the SFU is accurate near [-pi, pi] only
            return tl.inline_asm_elementwise(
                "cos.approx.f32 $0, $1;", constraints="=f,f", args=[_wrap2pi(x)],
                dtype=tl.float32, is_pure=True, pack=1)
    return tl.cos(x)


#: dTheta factors are the operand-rounded rotated maps (the value the MMAs consumed; `_opv`).
_DTH_OPROUND_DEFAULT = True
DTH_OPROUND = tl.constexpr(_DTH_OPROUND_DEFAULT)


#: How an fp32 MMA operand reaches the tensor core on the "tf32" dot path: pre-rounded with
#: cvt.rna.tf32.f32 (round to nearest, ties away) in `pdot`, one MMA, unbiased.  (The operand bits as
#: stored would be TRUNCATED by the sm_80/sm_90 tensor core, a signed -1.4e-3 on every fp64-oracle
#: tensor at fp32 IO.)  The exact and split modes ("ieee" and `families.SPLIT_DOT_MODES`) are
#: selected through the scan's `precision` argument, not here.  The dTheta factor reads the same
#: rounding (`_opv`).
FP32_DOT_MODE = "rna"
TF32_RNA = tl.constexpr(True)
_DTH_TF32 = "rna"
DTH_TF32 = tl.constexpr(_DTH_TF32)


@triton.jit
def _opv(v, MMA_DT: tl.constexpr, MMA_IO: tl.constexpr,
         PREC: tl.constexpr = "ieee", ON: tl.constexpr = _DTH_OPROUND_DEFAULT,
         TF32: tl.constexpr = _DTH_TF32):
    """The value an MMA operand ACTUALLY carried, back in fp32.

    The phase gradient
    dTheta_t = <dB~, J B~> + <dC~, J C~> must contract the rotated tile
    the MMAs consumed.  Under MMA_IO they consume `B~.to(MMA_DT)`; under
    PREC "tf32" with fp32 operands the tensor core reads a tf32 value.
    Pairing either with the fp32 `B~` breaks the exact shift identity
    sum_t dTheta_t = 0 at that operand's unit roundoff, which
    d theta_freq = sum_t Theta_t dTheta_t amplifies by |Theta|
    Identity at "ieee"."""
    if ON:
        if MMA_IO:
            v = v.to(MMA_DT).to(tl.float32)
        elif PREC == "tf32" and _TRIG_PTX:     # portable: the dot runs at ieee, nothing to round
            if TF32 == "rna":
                v = _rna_tf32(v)
            elif TF32 == "rz":
                v = (v.to(tl.int32, bitcast=True) & -8192).to(
                    tl.float32, bitcast=True)
    return v


@triton.jit
def _sin_approx(x, USE_PTX: tl.constexpr = _USE_PTX_TRIG_DEFAULT):
    """Hardware SFU sine (PTX sin.approx.f32) on NVIDIA; `tl.sin` elsewhere.
    See `_cos_approx`."""
    if USE_PTX:
        if _TRIG_PTX:
            return tl.inline_asm_elementwise(
                "sin.approx.f32 $0, $1;", constraints="=f,f", args=[_wrap2pi(x)],
                dtype=tl.float32, is_pure=True, pack=1)
    return tl.sin(x)


from typing import Optional

_STABLE_DEFAULT = False

#: widest rank the per-head-scalar-decay path serves (its slot is in the grid)
_ASCALAR_R_MAX = 8


def _stable_bc(T: int, c_per_o: bool = False) -> int:
    """Column-block size for the stable kernels.

    BC = T//2 (32 at T=64): two strips, dot shapes close to the
    classic kernel -- benchmarks at parity; within-block exponent
    <= r*dt*16, overflow envelope r*dt ~ 5.5 nats/token (4x the classic
    kernel's 1.4, beyond model-meaningful forgetting).  The kernels
    dispatch up to 4 strips, so T // BC must be 2 or 4.
    """
    return T // 2


_DOT_TILE_MIN = 16


def _resolve_stable(stable_score: Optional[bool], per_block_decay: bool,
                    has_rh: bool, T: int, R: int = 1,
                    g_size: int = 1, a_scalar: bool = False,
                    c_per_o: bool = False) -> bool:
    """Stable path covers per-block decay, uniform rank <= 4, G_SIZE <= 16
    — plus, with a_scalar=True, the per-head-SCALAR decay case (a_log [H],
    per_block_decay=False) on the dedicated A-scalar kernels (pairwise
    exp(la_t - la_s), all-negative exponents, NO rate cap).

    a_scalar is passed True only by the dynamic-B/C / grouped-BC dispatch
    sites: the plain per-head-scalar configuration stays on the classic
    kernel (already exponent-safe there).  Mixed-rank
    (HAS_RH) is not wired into the stable kernels; R > 4 would overflow
    the bwd kernel's four named per-stream score-strip slots; G_SIZE > 16
    exceeds the STRIP kernels' shared-memory budget on GH200 (would need
    G-tiling, same class as the classic kernels' G_SIZE=256 limit) — the
    A-scalar pair carries no strips, IS G-tiled and has NO G cap
    (the kernel grid tiles G across CTAs).

    T floor: the two families clear `_DOT_TILE_MIN`
    at DIFFERENT T because they tile T differently (see the module-level
    comment above `_DOT_TILE_MIN`).  a_scalar needs only T >= 16 (T=16 and
    T=32 both admitted — the ladder's smem floor of 32 is the practical
    limit, not this check).  The STRIP family needs `_stable_bc(T,
    c_per_o) >= 16`: T >= 32 (BC = T // 2).  T=16 is genuinely outside the STRIP family's envelope at any BC choice
    (T // 2 = 8 < 16) and must fall back to N-splitting instead of chunk
    halving.
    """
    if stable_score is None:
        stable_score = _STABLE_DEFAULT
    r_max = _ASCALAR_R_MAX if a_scalar else 4
    if has_rh or R > r_max or (not a_scalar and g_size > 16):
        return False
    if not per_block_decay and not a_scalar:
        return False
    if a_scalar:
        if T < _DOT_TILE_MIN:
            return False
    elif _stable_bc(T, c_per_o) < _DOT_TILE_MIN:
        return False
    return stable_score


def _mimo_axes(B_tok, C_tok):
    """Detect the dynamic-MIMO axes: B [B,L,N,R] per-stream, C [B,L,N,O]
    per-output.  3-dim tensors are the shared forms."""
    b_per_r = B_tok.dim() == 4
    c_per_o = C_tok.dim() == 4
    O = C_tok.shape[-1] if c_per_o else 1
    return b_per_r, c_per_o, O


def _check_mimo_support(b_per_r, c_per_o, O, R, use_stable, Q, r_heads,
                        otile=False, a_scalar=False):
    if not (b_per_r or c_per_o):
        return
    if not use_stable:
        raise NotImplementedError(
            "dynamic per-stream B / per-output C require the stable Score "
            "path: a_scalar decay needs T >= 16 (full [T,N]x[N,T] dot, no "
            "strip tile), the STRIP kernels (per-block decay) need "
            "_stable_bc(T, c_per_o) >= 16 (T >= 32 by default, T >= 64 "
            "under the legacy per-output-C BC=T//4 fallback), and either "
            "family also needs uniform rank and G_SIZE <= 16 (STRIP only); "
            "pass stable_score=True or fix the config")
    if Q is not None:
        raise NotImplementedError(
            "dynamic B/C with rotations (Q) not wired yet — run Q=None "
            "(complex/real modes)")
    if r_heads is not None:
        raise NotImplementedError("dynamic B/C with per-head ranks not wired")
    r_max = _ASCALAR_R_MAX if a_scalar else 2
    if b_per_r and R > r_max:
        raise NotImplementedError(
            f"dynamic per-stream B supports R <= {r_max} on this path "
            f"(a_scalar={a_scalar}); the strip kernels keep R <= 2 for "
            f"their named bwd per-stream slots, the A-scalar pair takes "
            f"{_ASCALAR_R_MAX} (its slot is in the grid)")
    # Per-output C above O=2 is served by o-tiling the chunk-kernel readout
    # (the scan is O-independent) on the differentiable path — see
    # autograd._unified_ssd_impl.  The non-tiled forward stays at O<=2.
    if c_per_o and O > 2 and not otile:
        raise NotImplementedError("dynamic per-output C supports O <= 2 "
                                  "(bwd per-output accumulator slots); the "
                                  "differentiable unified_ssd o-tiles above 2")


# ── span sums (fused_bwd_kernel `_SPANDELTA_NOTE`) ──────────────────────
@triton.jit
def _span_sum(M, offs_t):
    """S_s = sum_{i >= s} sum_{j < s} M[i, j] for a [T, T] tile (rows i,
    cols j): one reverse cumsum along i, one masked row sum.  Every term
    it adds is a pair that spans s (`_SPANDELTA_NOTE`)."""
    Rrev = tl.cumsum(M, axis=0, reverse=True)
    return tl.sum(tl.where(offs_t[None, :] < offs_t[:, None], Rrev, 0.0),
                  axis=1)


@triton.jit
def _excl_prefix(v, offs_t):
    """sum_{j < s} v[j] with no subtraction (`_SPANDELTA_NOTE`)."""
    return tl.sum(tl.where(offs_t[None, :] < offs_t[:, None], v[None, :],
                           0.0), axis=1)


@triton.jit
def _span_sum_T(MT, offs_t):
    """`_span_sum` of a tile held TRANSPOSED (rows j, cols i): one reverse
    cumsum along i (axis 1), one masked column sum."""
    Rr = tl.cumsum(MT, axis=1, reverse=True)
    return tl.sum(tl.where(offs_t[:, None] < offs_t[None, :], Rr, 0.0),
                  axis=0)


@triton.jit
def _rna_tf32(x):
    """cvt.rna.tf32.f32: fp32 -> tf32 round to nearest (ties away), kept in
    an fp32 register (low 13 mantissa bits zero)."""
    return tl.inline_asm_elementwise(
        "cvt.rna.tf32.f32 $0, $1;", constraints="=r,f", args=[x],
        dtype=tl.int32, is_pure=True, pack=1).to(tl.float32, bitcast=True)


@triton.jit
def _split_dot3(a, b, acc):
    """fp32-class product on bf16 tensor cores, plain Triton: a = a_hi + a_lo, b = b_hi + b_lo (bf16 pieces), three bf16 dots with fp32
    accumulation (the lo*lo term dropped): relative product error <= ~2^-16, far below the tf32 tag's 2^-11, tensor-core rate, bf16 shared-memory
    footprint on every backend with a bf16 matrix unit.  An operand that is already bf16 is exact and is not split."""
    a_f = a.dtype == tl.float32
    b_f = b.dtype == tl.float32
    ah = a.to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    acc = tl.dot(ah, bh, acc)
    if b_f:
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(ah, bl, acc)
    if a_f:
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(al, bh, acc)
    return acc


@triton.jit
def pdot(a, b, acc=None, input_precision: tl.constexpr = "tf32"):
    """`tl.dot` with the FP32DOT operand rounding (see `FP32_DOT_MODE`):
    under "rna" an fp32 operand on the tf32 path is rounded to nearest
    before the MMA instead of truncated by the tensor core.  Identity
    for 16-bit operands and every other mode.

    Portable (`_TRIG_PTX` False: kernel_target="portable", or a non-CUDA backend): no inline asm and no tf32 literal.  The "tf32" tag with an
    fp32 operand becomes the bf16 hi/lo split product `_split_dot3` (tensor cores, bf16 footprint; fp16 partners fall back to "ieee"); 16-bit
    operands take no input_precision; every other mode ("ieee", ...) is passed through."""
    if _TRIG_PTX:
        if TF32_RNA and input_precision == "tf32":
            if a.dtype == tl.float32:
                a = _rna_tf32(a)
            if b.dtype == tl.float32:
                b = _rna_tf32(b)
        return tl.dot(a, b, acc, input_precision=input_precision)
    if input_precision == "tf32":
        if a.dtype == tl.float32 or b.dtype == tl.float32:
            if (a.dtype == tl.float32 or a.dtype == tl.bfloat16) and (b.dtype == tl.float32 or b.dtype == tl.bfloat16):
                return _split_dot3(a, b, acc)
            return tl.dot(a, b, acc, input_precision="ieee")
        return tl.dot(a, b, acc)
    return tl.dot(a, b, acc, input_precision=input_precision)


@triton.jit
def plain_dot(a, b, acc=None, input_precision: tl.constexpr = "tf32"):
    """`tl.dot` without the operand rounding of `pdot`, with the same portable mapping of the "tf32" tag."""
    if _TRIG_PTX:
        return tl.dot(a, b, acc, input_precision=input_precision)
    if input_precision == "tf32":
        if a.dtype == tl.float32 or b.dtype == tl.float32:
            if (a.dtype == tl.float32 or a.dtype == tl.bfloat16) and (b.dtype == tl.float32 or b.dtype == tl.bfloat16):
                return _split_dot3(a, b, acc)
            return tl.dot(a, b, acc, input_precision="ieee")
        return tl.dot(a, b, acc)
    return tl.dot(a, b, acc, input_precision=input_precision)
