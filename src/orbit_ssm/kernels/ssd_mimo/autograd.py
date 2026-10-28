"""SSD-MIMO fused scan dispatch, autograd and bounded shape repairs.

Only the production scalar-decay route is supported. The forward and
multi-kernel backward retain compact scan state and optional fused endpoint,
rotation and gate adjoints.
"""

import os as _os

import contextlib as _contextlib
import threading as _threading
import warnings
from triton.runtime.errors import OutOfResources as _OutOfResources
import torch
from ..autograd_guard import no_double_backward as _no_double_backward
from ..input_checks import check_cuda_input
import torch.nn.functional as F
from typing import Optional

import math as _math

from .common import _mimo_axes, _resolve_stable, _check_mimo_support
from .fused_fwd_kernel import (fused_chunk_fwd, _env, dot_precision,
                               mma_io_enabled)
from .fused_bwd_kernel import (fused_chunk_bwd_5kernel, pairhalf_r1_t64_fault,
                               pairhalf_io_t64_fault, rot_io_bwd_t64_fault)
from .m3_epilogue import (
    trap_prologue,
)
from ...compile_compat import dynamo_opaque
from ...validation import RANKS, check_precision
from ...families import SCAN_PRECISIONS
from ...layers.proj import autocast_io
from .registered_scan import customop_enabled as _customop_enabled
from .registered_scan import scan_apply as _scan_apply
from ... import kernel_options as _ko
from .. import portable as _portable
from .smem import (
    N_SPLIT, T_HALVE, SsdSmemOverflow, chunk_ladder, measuring_possible,
    note_chunk_clamped, note_n_split, preferred_strategy, resolve_strategy,
    strategy_key, device_smem_limit, smem_clamp_enabled,
)

# ── Which compile boundary the scan entry draws ─────────────────────────
# With the registered route on (the default), `ssd_mimo_scan` must be
# TRACEABLE: its host prep is plain torch and Inductor should fuse it,
# and the opaque boundary moves down to the `orbit_ssm::ssd_scan_fwd`
# node `_scan_apply` puts in the graph (`registered_scan.py`).  With
# HYBRID_SSM_COMPILE_CUSTOMOP=0 the entry keeps a
# `torch.compiler.disable` boundary -- the A/B arm, and the fallback for
# anything the registered pair refuses.  Read once, at import, for the
# reason `compile_compat._OPAQUE` is.
_CUSTOMOP_SCAN = _customop_enabled()
from . import measured_plan as _measured_plan

_scan_boundary = (lambda f: f) if _CUSTOMOP_SCAN else dynamo_opaque


# ── Rotation operand width ───────────────────────────────────────────
# The host `theta` fold multiplies a (possibly bf16) B/C by an fp32
# cos/sin, and torch's promotion makes the rotated operand fp32.  So in
# bf16 rotation mode B~/C~ reach EVERY kernel twice as wide as the IO
# dtype asks for, while the rotation-FREE path loads them at the IO
# dtype.  This lever picks which:
#   "fp32"  keep the promotion            (DEFAULT)
#   "io"    cast B~/C~ back to B/C's dtype
# It is a numerics change in bf16 (one extra rounding of the rotated
# operand; dB/dC come back bf16 instead of fp32), so it ships OFF.  In
# fp32 IO both settings are bit-identical (the cast is a no-op).
_ROT_OPERAND_DTYPES = ("fp32", "io")


# ── WHERE the per-token phase is applied ─────────────────────────────
# "host"   — the differentiable torch fold below materialises
#            B~ = R(theta)^T B and C~ = R(theta)^T C before the launch;
#            the kernels see already-rotated operands and dTheta comes
#            from the fold's own autograd.
# "kernel" — B/C reach the fused forward and Ka / Kbc-hb / Kc RAW; each
#            forms its own B~/C~ tile on load from the [B, nc, T, N//2]
#            phase (the register pair swap) and the backward
#            accumulates dTheta in-kernel.  Same function, fewer launches
#            and no materialised B~/C~.
# The kernel form serves the R = O = 1 ungrouped fused envelope; every
# other config falls back to the host form, which is the only one the
# generic (Kb1 + Kinv) backward and the 3-dispatch route can take.
#
# PER-HEAD PHASE.  `theta` may be [B, L, N//2]
# (head-shared) or [B, L, H, N//2] (per head, matching
# Mamba-3's `Angles[B, L, H, ka]`).  The per-head form exists ONLY in the
# kernel: the host fold rotates B/C, and B/C are head-shared, so a
# per-head fold would have to materialise a [B, L, H, N] operand (a
# B*H batch re-layout that dominates the forward).
# Refusing it here is therefore the point of the
# feature, not a gap: `rot_where="host"` with a per-head phase raises.
_ROT_WHERE = ("host", "kernel")


# ── The operand layout the fused route accepts (contract 7.12) ───────
# The fused kernels read x through PASSED strides, so a permuted view is
# already served, and the backward allocates dX with x's strides.  The
# entry point therefore does not materialise a
# contiguous copy for any layout `torch.empty_strided` can mirror
# exactly, i.e. any NON-OVERLAPPING DENSE one (a pure permutation of a
# contiguous tensor, size-1 axes and their zero strides included).
# Anything else -- a broadcast expand, a strided slice -- is copied:
# dX would otherwise alias or leave holes.
def _is_dense_layout(t) -> bool:
    seen = sorted((abs(st), sz) for st, sz in zip(t.stride(), t.shape)
                  if sz > 1)
    expect = 1
    for st, sz in seen:
        if st != expect:
            return False
        expect *= sz
    return expect == t.numel() and all(
        st >= 0 for st in t.stride()) and t.numel() > 0


def _dense_or_contiguous(t):
    """`t` itself when the fused backward can mirror its layout exactly."""
    if t.is_contiguous() or not _is_dense_layout(t):
        return t.contiguous()
    return t


def _rot_bwd_split() -> bool:
    """`SSD_MIMO_ROT_BWD=split`: the Mamba-3 rotation decomposition.

    "split" materialises B~ / C~ per kernel head in a prologue and runs
    every scan kernel on its REAL body, with one elementwise
    `rot_adjoint` owning dTheta / dnu_h / dbeta / dgamma / dB / dC;
    "inkernel" (the default until the ladder says otherwise) keeps the
    shipped per-token dTheta inside Kb1 / Kb2 / Kbc-hb.  Both compute
    the same function; the A/B is reproducible by env alone.
    """
    from .m3_epilogue import rot_bwd_mode
    return rot_bwd_mode() == "split"


def _rot_route_split(x_raw, H, L, T, *, R_sp, O_sp, envelope, pair_half, thh):
    """The CALIBRATED structural route (rot_route.py) picks the ROTSPLIT backward for this call: a rotating
    R = O = 1 bf16 call inside the SISO envelope, no explicit ROT_BWD / SISO_FAST / CHUNK lever, on a GPU
    whose valid calibration selects `split` for this occupancy and length."""
    from . import rot_route as rr
    if (rr.explicit_rot_bwd() or rr.explicit_siso() or R_sp != 1 or O_sp != 1
            or x_raw.dtype not in (torch.bfloat16, torch.float32) or pair_half):
        return False
    r = rr.pick(x_raw.shape[0] * H, L, x_raw.device, True, thh,
                dtype="fp32" if x_raw.dtype == torch.float32 else None)
    if r is None or r[0] != "split" or r[1] != T:
        return False
    from .siso_route import siso_refusal
    return siso_refusal(**envelope) is None


def _rot_split_dtype(io_dtype, precision):
    """The dtype the ROTSPLIT prologue writes B~ / C~ in.

    The IO dtype by default (Mamba-3 saves `Q_rot` / `K_scaled` in the
    same bf16 its wrapper cast the inputs to), which costs ONE extra
    rounding of an operand the fused kernels round at the `tl.dot`
    anyway.  `precision="ieee"` is the fp32/fp64 numerics ORACLE, so it keeps fp32, and so does
    the `rot_split_dtype="fp32"` kernel option.
    """
    want = _env("ROT_SPLIT_DTYPE", "io").lower()
    if want == "fp32" or precision == "ieee":
        return torch.float32
    if want not in ("io", "fp32"):
        raise ValueError("kernel option rot_split_dtype must be 'io' or 'fp32'")
    return io_dtype


def _gate_out_io() -> bool:
    """Does the gate epilogue store the gated output at the IO dtype?

    DEFAULT, and the reason is a proof rather than
    a tolerance: `ssd_mimo_scan` ALREADY narrows `y` to the IO dtype
    before it returns (`_unified_ssd_impl`'s closing `y.to(x_in.dtype)`),
    so an fp32 store never escapes the entry point.  Rounding an fp32
    register once at the kernel's own store is the same round-to-nearest
    as rounding it in that later `.to()`, and in the backward the `.to()`
    node merely WIDENS the same bf16 numbers back to fp32 before the
    epilogue's adjoint loads them into fp32 registers.  The fp32 store
    costs a 128 MB read + 64 MB write and its adjoint at the LM-spine cell.

    `SSD_MIMO_GATE_OUT_IO=0` restores the fp32 store.
    """
    return _env("GATE_OUT_IO", "1") == "1"


def _dskip_in_kc() -> bool:
    """Does Kc accumulate the D-skip's gradient into its own dX store?

    DEFAULT.  The forward is
    ``y_pre = scan(x * dw) + D . x_raw``, so ``dL/dx_raw`` has two terms
    and the second one, ``D * dY_pre``, has BOTH operands already in
    Kc's registers at the store (`dY_o` IS dY_pre; `x_raw` is loaded for
    dDW).  Otherwise the gate epilogue writes a whole second
    [B, H, G, R, L] tensor and the host runs ``dX = dX + dX_skip``,
    for arithmetic that is free here.

    NUMERICS.  This is a dtype-level change, not a bitwise one, and in
    the SAFE direction: the two terms are summed in one fp32
    register and rounded ONCE at the IO-dtype store, where the epilogue
    route rounds each term separately and then adds in the IO dtype.  dD is
    the same reduction over the same tile with the same `atomic_add`
    (already order-nondeterministic either way).

    WHAT STAYS.  dD does NOT move: its reduction runs over the tile's
    T axis, and in Kc that is the axis the MMA warps are split along, so
    the cross-warp reduce costs more than the epilogue's plain
    elementwise [T, G] tile.  The epilogue therefore
    still reads x_raw and accumulates dD -- in-kernel either way -- and
    only its dXR STORE is dropped.  `SSD_MIMO_DSKIP_IN_KC=all` puts dD
    in Kc too.

    `SSD_MIMO_DSKIP_IN_KC=0` restores the epilogue route entirely.  It
    steers the UNFUSED-norm route only: with the fused output norm on, the
    epilogue that read x_raw no longer exists, so Kc owns both terms
    whatever this says.
    """
    v = _env("DSKIP_IN_KC", "1")
    return "all" if v == "all" else ("dx" if v == "1" else "off")


def _real_r1o1_call(ctx, x_raw, Y_pre) -> bool:
    """Is this backward a real (rotation-free) R = O = 1 call?  The guard of
    the calibrated group `real_R1O1` levers decided outside the scan kernels."""
    return bool(ctx.theta is None and ctx.theta_h is None
                and ctx.rot_bias is None and x_raw.shape[3] == 1
                and Y_pre.shape[4] == 1)


def _gate_epilogue_reads_strided() -> bool:
    """True when the gate runs in the dedicated epilogue kernel, which
    addresses z (and dz) through its own stride set."""
    return (_env("GATE_INKERNEL", "0") != "1"
            and _env("GATE_FUSED", "1") == "1")


def _rot_where(mode=None) -> str:
    """Resolve the rotation site; an unknown value is refused by name."""
    if mode is None:
        mode = _env("ROT_WHERE", "kernel")
    if mode not in _ROT_WHERE:
        raise ValueError(
            "rot_where must be one of %s, got %r" % (_ROT_WHERE, mode))
    return mode


def _rot_operand_io(mode=None) -> bool:
    """True when the rotated B~/C~ should be cast back to the IO dtype."""
    if mode is None:
        mode = _env("ROT_OPERAND_DTYPE", "fp32")
    if mode not in _ROT_OPERAND_DTYPES:
        raise ValueError(
            "rot_operand_dtype must be one of %s, got %r"
            % (_ROT_OPERAND_DTYPES, mode))
    return mode == "io"


# ── Fused SSD autograd (2 kernel launches: 1 fwd + 1 bwd) ────────────

# ═══════════════════════════════════════════════════════════════════
#  Driving the scan Function BY HAND (the combined mixer node)
#
#  `SSDMimoSSM(combined=True)` is ONE autograd node for the whole
#  mixer: it runs the forward under `no_grad` and owns the backward.
#  The scan's own Function already saves the minimal set (x_pad / B / C /
#  dt / the recomputed S_all) and has a hand-written backward, so the
#  combined node does NOT reimplement it -- it CAPTURES it.  Inside a
#  `capture_scan()` region `_unified_ssd_impl` calls `_SSDMimoFn.forward`
#  with a stand-in ctx instead of `.apply`, and records everything the
#  combined node needs to (a) run `_SSDMimoFn.backward` later and (b) map
#  its gradients back onto `ssd_mimo_scan`'s OWN arguments.
#
#  Why a capture rather than a second implementation: there is exactly
#  one place that knows the prep (the L padding, the chunk cumsum, the
#  R/O-major B/C copies, the `.float().contiguous()`s), and a copy of it
#  would drift.
# ═══════════════════════════════════════════════════════════════════

_CAPTURE = _threading.local()
# Give the slot its default HERE.  `getattr(
# _CAPTURE, "slot", None)` is read inside the traced region, and Dynamo
# guards a MISSING attribute as `not hasattr(...)` -- a guard the very
# next call falsifies, i.e. one recompile per thread-local per frame,
# against a `cache_size_limit` of 8.  An attribute that exists and holds
# None guards as a value instead.
_CAPTURE.slot = None


class _ShimCtx:
    """A stand-in for autograd's ctx, so a Function's forward/backward
    bodies can be driven from inside another Function."""

    __slots__ = ("saved_tensors", "__dict__")

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


# ── the ctx TENSORS ride `save_for_backward` ──────────────────────────
# A ctx attribute lives as long as the graph NODE does, and the node
# outlives its backward whenever the caller still references the output:
# the bf16 phase twin `theta` alone is ~192 MiB per layer at B6 L2048
# alive into the next step.  The
# autograd engine frees `saved_tensors` right after the node's backward
# (unless `retain_graph=True`, which then keeps them for the next pass --
# the reason a None-clear in the backward is NOT the fix).
#
# Layout: the forward's own positional saved list comes first and the
# named tensors are APPENDED after it, every name always present (None
# where absent), so every positional read (`sv[:7]`, `sv[idx_t]`, ...)
# keeps its index.  The SISO aux dict splits into its tensor members
# (saved, as `siso_aux.<key>`) and the rest (`ctx.siso_aux_meta`).
_CTX_SAVED = ("theta", "theta_h", "theta_off", "beta", "gamma", "rot_bias",
              "x0", "out_norm_w", "out_norm_rstd", "rs_theta", "rs_theta_h")
_SISO_AUX = "siso_aux."


def _save_with_ctx_tensors(ctx, saved, named, siso_aux=None):
    """`ctx.save_for_backward(*saved, <named tail>)`.

    `named` maps every name in `_CTX_SAVED` (missing ones are None).
    """
    names = list(_CTX_SAVED)
    vals = [named.get(k) for k in _CTX_SAVED]
    ctx.siso_aux_meta = None
    if siso_aux is not None:
        ctx.siso_aux_meta = {k: v for k, v in siso_aux.items()
                             if not isinstance(v, torch.Tensor)}
        for k, v in siso_aux.items():
            if isinstance(v, torch.Tensor):
                names.append(_SISO_AUX + k)
                vals.append(v)
    ctx.saved_head = len(saved)
    ctx.saved_names = tuple(names)
    ctx.save_for_backward(*saved, *vals)


class _SavedCtx:
    """The backward's view of a `_SSDMimoFn` ctx: the named tensors come
    out of `saved_tensors` (see `_save_with_ctx_tensors`), every other
    field passes through to the real ctx, and so do writes."""

    # `saved_tensors` is UNPACKED ONCE, here, and served from `_sv` for
    # the rest of the backward: under `torch.utils.checkpoint(...,
    # use_reentrant=False)` (the LM's `ckpt_every_blocks`,
    # SSDMimo `modular_checkpoint`) each saved tensor may be unpacked only
    # once -- a second `ctx.saved_tensors` raises CheckpointError -- and
    # under other saved-tensor hooks (offload) every unpack is a copy.
    __slots__ = ("_ctx", "_t", "_sv")

    def __init__(self, ctx):
        object.__setattr__(self, "_ctx", ctx)
        sv = tuple(ctx.saved_tensors)
        object.__setattr__(self, "_sv", sv)
        object.__setattr__(self, "_t", dict(zip(
            ctx.saved_names, sv[ctx.saved_head:])))

    def __getattr__(self, k):
        if k == "saved_tensors":
            return self._sv
        t = self._t
        if k in t:
            return t[k]
        if k == "siso_aux":
            meta = self._ctx.siso_aux_meta
            if meta is None:
                return None
            aux = dict(meta)
            aux.update({n[len(_SISO_AUX):]: v for n, v in t.items()
                        if n.startswith(_SISO_AUX)})
            return aux
        return getattr(self._ctx, k)

    def __setattr__(self, k, v):
        setattr(self._ctx, k, v)


def saved_view(ctx):
    """`_SavedCtx(ctx)`, idempotent."""
    return ctx if isinstance(ctx, _SavedCtx) else _SavedCtx(ctx)


@_contextlib.contextmanager
def capture_scan():
    """Capture the `_SSDMimoFn` call(s) a scan makes, for a caller that
    drives the backward by hand.  Yields the (initially empty) list of
    `ScanTape`s; exactly one is expected on the fused single-tile route,
    and the caller must check.
    """
    prev = getattr(_CAPTURE, "slot", None)
    _CAPTURE.slot = []
    try:
        yield _CAPTURE.slot
    finally:
        _CAPTURE.slot = prev


def _capture_slot():
    return getattr(_CAPTURE, "slot", None)


def _rev_cumsum(g):
    """Adjoint of `cumsum(dim=-1)`: the reverse-inclusive prefix sum."""
    return g.flip(-1).cumsum(-1).flip(-1)


# ── The chunk-local decay prefix, in kernel ──────────────────────────
# `SSD_MIMO_DCS_INK`
#   "auto" (default) -- ON exactly where it pays, i.e. R * O > 1 (the
#       host `aten::cumsum` that builds D_i is a visible cost there and
#       noise at R = O = 1).
#   "1" / "0"      -- force on / off (the A/B lever).
# OFF: the host cumsum is built and the kernels load it.  ON replaces it
# with one `tl.cumsum` per chunk
# tile inside every kernel that reads D (see `_dcs_tile` in
# `fused_fwd_kernel.py`), which is a PARALLEL scan where ATen's is a
# sequential one -- same value to a few ulp, not the same bits.  That is
# why R = O = 1 (and with it every SISO and every rotating R1 path) stays
# on the host cumsum by default.
def _dcs_ink_on(R, O, use_siso=False):
    lever = _os.environ.get("SSD_MIMO_DCS_INK", "auto").lower()
    if lever in ("0", "false", "no", "off"):
        return False
    if use_siso:
        # the SISO schedule's kernels are a separate file and are not
        # wired for the in-kernel prefix; it is an R = O = 1 route, i.e.
        # exactly where the lever is off by default anyway.
        return False
    if lever in ("1", "true", "yes", "on"):
        return True
    return (R * O) > 1


# ── GROUP-MAJOR B_c / C_c ─────────────────────────────────────────────
# `SSD_MIMO_BC_GROUP_MAJOR`  "auto" (= on wherever B/C carry a group
# axis) | "1" | "0".
#
# This is the prerequisite for a decoupled backward
# chunk.  The grouped chunk view can be materialised TOKEN-MAJOR,
#     [B, nc, Gx, (RO,) T, N]   physically,
# i.e. with the group axis BETWEEN the chunk index and the
# token-within-chunk index, so changing T changes the (nc, T) split and
# the tensor cannot be re-viewed at another chunk length -- the backward
# would need a second `.contiguous()` copy of the whole B and C
# activation.
#
# Group-major is
#     [B, Gx, (RO,) L, N]       physically,
# under exactly the SAME logical shape and with the SAME contiguous N
# axis, so every [T, N] tile load in every kernel is the same coalesced
# segment and every consumer -- all of them stride-driven -- is
# unchanged.  It costs the ONE permute copy the token-major build
# already pays.  What it buys: `L = nc*T` is a single contiguous run per
# (group, r), so a chunk view at ANY T is an `as_strided`, free.
def _bc_group_major_on(bc_grouped, rotating=False, wants_chunk_bwd=False):
    """Is the grouped chunk view built GROUP-MAJOR?

    DEFAULT "auto" = grouped B/C and NO ROTATION -- a refusal by name,
    like every other lever in this family.  The layout is
    a large WIN on the real (rotation-free) arms (-3 % to -8 % fwd+bwd) and
    a large LOSS on the rotating ones (+14 % on Blackwell, +33 % on A100,
    most of it in Kb1 and `_trap_adjoint_kernel`).

    The mechanism is L2 reuse ACROSS HEAD BLOCKS.  Token-major puts all
    `Gx` groups of ONE chunk in a contiguous `Gx*T*N` block, so CTAs at
    different `(b, head block)` but the same chunk share L2 lines;
    group-major puts them `L*N` apart.  The rotating bodies are the ones
    that re-read the same `[T, N]` tile many times per chunk (the pair
    swaps, the `o`/`r` unroll, the phase chain), so they are the ones
    that live on that reuse.

    `wants_chunk_bwd` does NOT override the refusal: a rotating call
    cannot USE a decoupled backward chunk
    (`_SSDMimoFn.backward` refuses it by name, because `dTheta` is a
    chunk-shaped output), so taking the layout there would pay its cost
    and buy nothing -- and it is observable, because the layout moves
    the cross-CTA atomic ORDER.  The argument is kept
    because it is what makes the rule readable: group-major is taken
    exactly where a free re-chunk is both possible and useful.
    `"1"` / `"0"` force it -- the A/B lever.
    """
    lever = _os.environ.get("SSD_MIMO_BC_GROUP_MAJOR", "auto").lower()
    if lever in ("0", "false", "no", "off"):
        out = False
    elif lever in ("1", "true", "yes", "on"):
        out = True
    else:
        out = bool(bc_grouped) and not bool(rotating)
    # The diagnostic ring buffers are HOST-side and
    # are skipped while TRACING -- the treatment `check_a_log_range`'s
    # device->host sync already takes.
    # `del LOG[:-MAX]` reads the
    # list's LENGTH, so a traced append installs a length guard that the next
    # call falsifies: one recompile per step until the buffer saturates,
    # against a `cache_size_limit` of 8.  With the LM path in ONE graph that
    # limit retires the whole module to eager.
    if not torch.compiler.is_compiling():
        _BC_LAYOUT_LOG.append(dict(bc_grouped=bool(bc_grouped),
                                   rotating=bool(rotating),
                                   wants_chunk_bwd=bool(wants_chunk_bwd),
                                   lever=lever, group_major=out))
        del _BC_LAYOUT_LOG[:-_BC_LAYOUT_LOG_MAX]
    return out


_BC_LAYOUT_LOG: list = []
_BC_LAYOUT_LOG_MAX = 32


def bc_layout_log() -> list:
    """The last 32 grouped-B/C layout decisions, newest last.

    Keys: `bc_grouped`, `rotating`, `wants_chunk_bwd`, `lever` and
    `group_major` (what was taken).  A caller who wants to know why
    `"auto"` did or did not take group-major reads this instead of
    re-deriving the rule -- the same idea as `kb1_dispatch_log()` and
    `chunk_bwd_log()`."""
    return list(_BC_LAYOUT_LOG)


def _bc_chunk_grouped(tok_pad, B, nc, T, Gx, N, RO, group_major):
    """The grouped chunk view of B / C; see the comment block above.

    `RO` is the trailing per-r / per-o axis width, or 0 when there is
    none.  Returns the logical [B, nc, Gx, T, N(, RO)] tensor either
    way -- only the MEMORY ORDER differs.
    """
    L = nc * T
    if RO:
        if group_major:
            return (tok_pad.view(B, L, Gx, N, RO)
                    .permute(0, 2, 4, 1, 3).contiguous()    # [B,Gx,RO,L,N]
                    .view(B, Gx, RO, nc, T, N)
                    .permute(0, 3, 1, 4, 5, 2))             # [B,nc,Gx,T,N,RO]
        return (tok_pad.view(B, nc, T, Gx, N, RO)
                .permute(0, 1, 3, 5, 2, 4).contiguous()
                .permute(0, 1, 2, 4, 5, 3))
    if group_major:
        return (tok_pad.view(B, L, Gx, N)
                .permute(0, 2, 1, 3).contiguous()           # [B, Gx, L, N]
                .view(B, Gx, nc, T, N)
                .permute(0, 2, 1, 3, 4))                    # [B, nc, Gx, T, N]
    return tok_pad.view(B, nc, T, Gx, N).transpose(2, 3).contiguous()


# ── The DECOUPLED BACKWARD CHUNK ─────────────────────────────────────
# `chunk_size_bwd` / `SSD_MIMO_CHUNK_BWD`.  The group-major layout above
# removes the blocker; what is left is one re-view per chunked
# artefact.
#
# WHAT MAKES A BACKWARD CHUNK LEGAL.  The backward reads five chunked
# things -- `Ctil`, `Btil`, `delta_cumsum`, `dY` and `S_all` -- and every
# other operand it touches (`x_raw` [B,H,G,R,L], `dw` [B,H,L], the whole
# trapezoid set [B,H,L(,O,R)], `x0`, `dS_final`) is TOKEN-indexed and
# therefore already chunk-free.
#
#   * `S_all` is REBUILT by the STATE_ONLY forward replay under the
#     default `SSD_MIMO_SALL=recompute`, at whatever chunk it is given:
#     it scans, it does not read a T-shaped artefact.
#   * `delta_cumsum` under `DCS_INK` IS the raw chunked delta, a view of
#     the caller's [B, H, L] tensor -- re-viewing it at another T is free
#     (without the lever it would be a second full ATen cumsum).
#   * `dY`, `Ctil`, `Btil` are re-viewable exactly when the chunk axis is
#     a pure regrouping of the token axis in MEMORY, i.e.
#         stride(chunk) == T * stride(token),
#     which `_rechunk` CHECKS rather than assumes.  For B / C that is the
#     group-major layout's whole point; `dY` satisfies it in all three of
#     the layouts `_native_y` builds (channel-major, token-major and the
#     plain contiguous gradient autograd hands back).
#
# The gradients the backward produces (`dBtil`, `dCtil`, `dDelta`) are
# allocated at T_bwd and re-viewed back to T_fwd for the return, over the
# same storage -- the shapes the autograd graph expects, no copy.
#
# The RESULT is the same function of the same inputs: the chunk length
# only regroups the scan's associativity, and the backward's own D_i /
# dL/dD_i pair is self-consistent at whatever T it runs (`delta_adjoint`
# is handed the SAME re-chunked `delta_cumsum`).  Numerically it moves by
# round-off, exactly as `smem.py`'s T-halving clamp does.


def _rechunk(t, ca, ta, T_new, T_old=None):
    """Re-view a [..., nc, T, ...] tensor at chunk length `T_new`.

    `ca` / `ta` are the chunk and token-within-chunk axes.  Returns
    `None` when the tensor is not a pure token regrouping in memory, so
    the caller can refuse the whole plan by name instead of copying.

    `T_old`, when given, is ASSERTED against `shape[ta]`: the axis map
    depends on whether B / C carry a group axis, and a wrong `ta` whose
    stride test happens to pass would re-view the GROUP axis as if it
    were the token axis -- silently, and with the right element count.
    Refusing on the size check turns that into a by-name refusal.
    """
    if t is None:
        return None
    sz, st = list(t.shape), list(t.stride())
    if T_old is not None and sz[ta] != T_old:
        return None
    if st[ca] != sz[ta] * st[ta]:
        return None
    L = sz[ca] * sz[ta]
    if T_new <= 0 or L % T_new:
        return None
    sz[ca], sz[ta] = L // T_new, T_new
    st[ca] = T_new * st[ta]
    return t.as_strided(sz, st, t.storage_offset())


def _chunk_bwd_env():
    """`chunk_size_bwd`: "" / "auto" (= the forward T) or an int.

    `kernel_options["chunk_size_bwd"]` wins over the `SSD_MIMO_CHUNK_BWD`
    env spelling; `_env` is the choke point that
    implements that precedence."""
    v = _env("CHUNK_BWD", "").strip().lower()
    if v in ("", "auto", "0"):
        return 0
    return int(v)


# The per-call `chunk_size_bwd` travels the way `_CAPTURE` does -- a
# thread-local set by the entry point around the dispatch -- rather than
# as a fourth chunk argument threaded through `_ssd_dispatch` and the
# three fallback impls that have no backward chunk at all.
_CHUNK_BWD = _threading.local()
# Present-and-None for the same reason as `_CAPTURE` above
# (`_chunk_bwd_asked` reads None and 0 identically).
_CHUNK_BWD.T = None


@_contextlib.contextmanager
def _chunk_bwd_request(T_bwd):
    prev = getattr(_CHUNK_BWD, "T", None)
    _CHUNK_BWD.T = int(T_bwd or 0)
    try:
        yield
    finally:
        _CHUNK_BWD.T = prev


def _chunk_bwd_asked():
    """This call's `chunk_size_bwd`, or the env lever, or 0."""
    return int(getattr(_CHUNK_BWD, "T", None) or 0) or _chunk_bwd_env()


_CHUNK_BWD_LOG: list = []
_CHUNK_BWD_LOG_MAX = 32


def chunk_bwd_log() -> list:
    """The last 32 backward-chunk decisions, newest last.

    Keys: `T_fwd`, `T_bwd` (what the backward ACTUALLY ran at), `asked`
    (what the caller requested, 0 = "the forward's") and `refused` (the
    first refusal term in source order, "" when the plan was taken).  A
    caller who wants to know why `chunk_size_bwd` did not engage reads
    this instead of re-deriving the envelope.
    """
    return list(_CHUNK_BWD_LOG)


def _note_chunk_bwd(rec: dict) -> None:
    # The diagnostic ring buffers are HOST-side and
    # are skipped while TRACING -- the treatment `check_a_log_range`'s
    # device->host sync already takes.  `del LOG[:-MAX]` reads the
    # list's LENGTH, so a traced append installs a length guard that the next
    # call falsifies: one recompile per step until the buffer saturates,
    # against a `cache_size_limit` of 8.  With the LM path in ONE graph that
    # limit retires the whole module to eager.
    if torch.compiler.is_compiling():
        return
    _CHUNK_BWD_LOG.append(rec)
    del _CHUNK_BWD_LOG[:-_CHUNK_BWD_LOG_MAX]


class SsdBackwardChunkRefused(RuntimeError):
    """The backward's score path does not fit shared memory at this
    chunk, and the repair -- re-running the BACKWARD at
    a shorter chunk -- is refused by the decoupled-chunk envelope.

    Raised INSTEAD of Triton's bare `OutOfResources`, so the cell is a
    named refusal with a named cause rather than a crash.
    """


def _cb_refuse(why, T_bwd):
    raise SsdBackwardChunkRefused(
        "SSD-MIMO backward: the score-path kernel does not fit shared "
        "memory at chunk %d, and the backward cannot be re-chunked "
        "shorter here: %s.  Repairs, in order: SSD_MIMO_DCS_INK=1 (the "
        "chunk-local decay prefix becomes a VIEW of delta, which is what "
        "makes the re-chunk exact), a shorter `chunk_size`, or "
        "SSD_MIMO_SALL=recompute if it was turned off."
        % (T_bwd, why))


class ScanTape:
    """One captured `ssd_mimo_scan` call.

    `backward(dY, dS_final)` runs `_SSDMimoFn.backward` and returns a
    dict keyed by `ssd_mimo_scan`'s own parameter names -- so the caller
    never has to know about the L padding, the chunk cumsum or the
    `.float()`s the wrapper inserted.
    """

    def __init__(self, ctx, inputs, meta):
        self.ctx = ctx
        self.inputs = inputs
        self.meta = meta

    def dY_from_y(self, dy):
        """Map a gradient of the SCAN'S RETURNED `y` -- [B, H, (O,) G, L],
        the shape `_unified_ssd_impl` hands back -- onto the Function's
        own `Y` [B, H, nc, T, O, G].  The forward chain between the two
        is squeeze / permute / reshape / narrow, i.e. pure views, so the
        adjoint is the inverse axis map; `dy` may arrive in any layout
        and the result is read through its strides by both consumers.
        """
        m = self.meta
        if m["L_padded"] != m["L"]:
            raise NotImplementedError(
                "captured scan: chunk-padded L needs the gradient "
                "zero-extended; the combined node's envelope excludes it")
        Bsz, H, G, nc, T, O = (m["B"], m["H"], m["G"], m["L"] // m["T"],
                               m["T"], m["O"])
        if O == 1:
            return dy.reshape(Bsz, H, G, nc, T).permute(0, 1, 3, 4, 2) \
                     .unsqueeze(4)
        return dy.reshape(Bsz, H, O, G, nc, T).permute(0, 1, 4, 5, 2, 3)

    # -- the gradient of `Y` is what the caller has; `S_final`'s is
    #    optional (None on every route the mixer takes).
    def backward(self, dY, dS_final=None):
        m = self.meta
        src = self.inputs
        g = _SSDMimoFn.backward(self.ctx, dY, dS_final)
        (dCtil, dBtil, dX, dDW, dDeltaCs, da_log, dThetaH, dTheta, dThOff,
         dZ, dDsk, dBeta, dGamma, dX0, dRotBias, dLam) = g[:16]
        dNormW = g[28] if len(g) > 28 else None
        L, Lp = m["L"], m["L_padded"]
        Bsz, H = m["B"], m["H"]
        out = {}

        def _back(name, grad, shape=None, trim_l=None):
            t = src[name]
            if grad is None or t is None:
                out[name] = None
                return
            if shape is not None:
                grad = grad.reshape(shape)
            if trim_l is not None and Lp > L:
                grad = grad.narrow(trim_l, 0, L)
            out[name] = grad.to(t.dtype) if grad.dtype != t.dtype else grad

        # x: [B, H, G, R, L(+pad)] in the kernel's own layout
        _back("x_in", dX, shape=(Bsz, H, m["G"], m["R"], Lp), trim_l=4)
        # delta / delta_w: the chunk cumsum's adjoint, plus the write scale
        d_delta = None
        if dDeltaCs is not None:
            # With the in-kernel prefix the Function
            # already applied the reverse prefix sum (its `delta_cumsum`
            # input IS delta), so this adjoint would double-apply it.
            # SPANDELTA: the Function's delta slot is the RAW chunked
            # delta and its gradient is dL/ddelta itself on every route
            # delta and its gradient is dL/ddelta itself on every route
            d_delta = dDeltaCs.reshape(Bsz, H, Lp)
        if m["delta_w_is_none"] and dDW is not None:
            dw_v = dDW.reshape(Bsz, H, Lp)
            d_delta = dw_v if d_delta is None else d_delta + dw_v
            out["delta_w"] = None
        else:
            _back("delta_w", dDW, shape=(Bsz, H, Lp), trim_l=2)
        _back("delta", d_delta, trim_l=2)
        # B / C: `B_c` / `C_c` are chunk views (possibly R/O-major COPIES)
        # of the padded token tensors, and the permute round trip that
        # builds them composes to the identity -- so the logical reshape
        # IS the adjoint.
        _back("B_tok", dBtil, shape=(Bsz, Lp) + tuple(m["b_tail"]), trim_l=1)
        _back("C_tok", dCtil, shape=(Bsz, Lp) + tuple(m["c_tail"]), trim_l=1)
        _back("a_log", da_log)
        _back("theta_h", dThetaH)
        if src["theta"] is not None:
            th_shape = (Bsz, Lp) + tuple(src["theta"].shape[2:])
            _back("theta", dTheta, shape=th_shape, trim_l=1)
        else:
            out["theta"] = None
        _back("gate_z", dZ, trim_l=1)
        _back("d_skip", dDsk)
        _back("beta", dBeta)
        _back("gamma", dGamma)
        _back("x0", dX0)
        _back("rot_bias", dRotBias)
        _back("trap_lam", dLam, shape=(Bsz, H, Lp), trim_l=2)
        # The fused output norm's weight gradient, when the enclosing
        # module handed its norm to the epilogue.
        _back("out_norm_w", dNormW)
        if dThOff is not None:
            # the HOST offset build is a torch graph; capture refuses
            # that route (see `_unified_ssd_impl`), so this cannot fire.
            raise AssertionError("captured scan returned a host dThetaOff")
        return out


class _SSDMimoFn(torch.autograd.Function):
    """The SSD-MIMO family autograd Function (fused fwd + bwd, a_scalar).
    Formerly _FusedSSDFn.  Supports R,O>=1, theta,
    optional SwiGLU gate (gate_z), beta, and gamma."""

    @staticmethod
    def forward(ctx, Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h,
                theta, theta_off, gate_z, d_skip, beta, gamma, x0,
                rot_bias, trap_lam,
                H, T, N, G_SIZE, precision, Gb, Gc,
                bc_grouped, needs_bwd=True, bias_pre_rot=False,
                theta_incr=False, y_token_major=False,
                out_norm_w=None, out_norm_eps=1e-5,
                pair_half=False, dcs_ink=False, chunk_bwd=0,
                theta_lo=None, chart_probe=None, chart_geometry=None):
        # x_raw is UNSCALED x; dw [B, H, L_padded] is the write scale
        # (delta_w, or delta itself when the write is not decoupled).
        # The kernels scale x on load and the backward returns dDW —
        # this removes the host-side x_scaled materialisation and its
        # ~200MB backward chain.
        #
        # S_all memory policy (SSD_MIMO_SALL, DEFAULT "recompute"):
        # do NOT save the per-chunk states
        # — the backward relaunches the FWD kernel in STATE_ONLY mode
        # (aggregates + scan, no score/Y) to rebuild S_all bit-identically.
        # SSD_MIMO_SALL=io restores the save-mode (a few % faster,
        # higher memory).
        recompute = (_env("SALL", "recompute")
                     == "recompute") and needs_bwd
        # SPANDELTA (fused_bwd_kernel `_SPANDELTA_NOTE`): the
        # `delta_cumsum` slot ALWAYS carries the RAW chunked delta and the
        # backward returns dL/ddelta itself (spanning-pair form, no reverse
        # prefix sum anywhere); without the in-kernel prefix the chunk
        # prefix D is formed HERE, outside autograd.
        delta_raw = delta_cumsum
        if not dcs_ink:
            # SEGPREFIX (fused_fwd_kernel `_SEGPREFIX_NOTE`): the prefix
            # restarts at reset tokens (sentinel -1 there)
            from .fused_fwd_kernel import seg_prefix_host
            delta_cumsum = seg_prefix_host(delta_raw, a_log)
        has_gate = gate_z is not None
        has_dskip = d_skip is not None
        # ── WHERE each epilogue runs ─────────────────────────────────
        # "Fused" means in the scan kernel OR in one dedicated epilogue
        # kernel per pass -- never a host-side pass over activation-sized
        # tensors -- and the choice between the first two is made by
        # MEASUREMENT, shipping the faster non-spilling form.  The fused
        # forward sits AT the 255-register ceiling, so every extra live
        # value spills: the gate's store costs the scan +15% .. +80%, the
        # D-skip +12%, the trapezoid +10%.
        #
        # So the GATE goes to the dedicated epilogue kernel
        # (`ascalar_epilogue_fwd` / `_bwd`), which costs ~0.1 ms and does
        # not touch the scan's register budget.  The D-SKIP stays at the
        # store even though it spills: moving it would force the epilogue
        # to WRITE a second activation-sized tensor (the post-skip y_pre
        # the gate adjoint reads) and to read x, which is more traffic
        # than the scan gives back.  The TRAPEZOID stays for the same
        # reason and by a much wider margin (it saves host launches on
        # both passes).
        #
        # SSD_MIMO_GATE_INKERNEL=1 puts the gate back in the store;
        # SSD_MIMO_GATE_FUSED=0 falls further back to a torch chain,
        # which is forbidden as a default and exists only as a before-arm.
        in_kernel = has_gate and _env("GATE_INKERNEL", "0") == "1"
        # ── The SISO fast path (R = O = 1), env-gated ─────────────────
        # `siso_route` decides by SHAPE; every refusal is by name and
        # falls through to the fused route below, unchanged.  The
        # schedule always stores S_all (the SSD_MIMO_SALL=io policy is
        # both faster and lower-peak for that pass).
        from .siso_route import (siso_bc, siso_enabled, siso_fuse,
                                 siso_refusal, siso_ytm)
        # real R = O = 1: the SISO route serves only where the device's
        # calibrated crossover selects the chunk-parallel forward
        real_cp = False
        if theta is None and theta_h is None and rot_bias is None:
            from .chunkpar import chunkpar_wanted as _cpw, cp_refusal as _cpr
            from .chunkpar import real_cp_decision as _rcd, plan_flags as _pf
            real_cp = (_cpr(theta=theta, theta_h=theta_h, rot_bias=rot_bias,
                            x0=x0, fuse=None, dcs_ink=dcs_ink, T=T, N=N,
                            G_SIZE=G_SIZE) is None
                       and _rcd(n_serial_ctas=x_raw.shape[0] * H, nc=delta_cumsum.shape[2], T=T, N=N,
                                G_SIZE=G_SIZE, x_dtype=x_raw.dtype, precision=precision,
                                device=x_raw.device, needs_bwd=needs_bwd,
                                flags=_pf(x0=x0, beta=beta, gamma=gamma, dw=dw, trap_lam=trap_lam,
                                          d_skip=d_skip, gate_z=gate_z, in_kernel=in_kernel)))
        siso_why = (None if siso_enabled(
            x_raw.device, rotating=theta is not None,
            io_bf16=x_raw.dtype == torch.bfloat16, T=T, real_cp=real_cp,
            bh=x_raw.shape[0] * H, L=delta_cumsum.shape[2] * T,
            needs_bwd=needs_bwd, thh=theta_h is not None,
            io_fp32=(x_raw.dtype == torch.float32 and precision != "ieee"))
            else "SSD_MIMO_SISO_FAST is off (or 'auto' does not apply)")
        if siso_why is None:
            siso_why = siso_refusal(
                Ctil=Ctil, Btil=Btil, x_raw=x_raw, G_SIZE=G_SIZE, N=N,
                bc_grouped=bc_grouped, Gb=Gb, Gc=Gc, theta_off=theta_off,
                in_kernel_gate=in_kernel, T=T, theta=theta,
                theta_h=theta_h, rot_bias=rot_bias)
        if siso_why is None:
            from .siso_route import siso_extra_refusal
            siso_why = siso_extra_refusal(
                x_raw=x_raw, precision=precision, T=T, N=N, G_SIZE=G_SIZE,
                pair_half=pair_half, theta=theta)
        if siso_why is None and theta is not None:
            from .siso_route import dth_rot_refusal
            siso_why = dth_rot_refusal(
                needs_angle_grad=needs_bwd, device=x_raw.device,
                io_dtype=x_raw.dtype, bias_pre_rot=bias_pre_rot,
                beta=beta, gamma=gamma, theta=theta, N=N,
                T=delta_cumsum.shape[-1], G_SIZE=G_SIZE)
        use_siso = siso_why is None
        if use_siso and dcs_ink:
            # The in-kernel decay prefix is not wired into siso_{fwd,bwd}_kernel.py;
            # it is an R = O = 1 route, where `_dcs_ink_on` is off by
            # default, so this can only be reached by forcing the env.
            raise NotImplementedError(
                "SSD_MIMO_DCS_INK=1 with the SISO fast path: the in-kernel "
                "decay prefix is wired into the fused kernels only "
                "(set SSD_MIMO_SISO_FAST=0 or SSD_MIMO_DCS_INK=0)")
        # ── SISOFUSE: the SISO SPEED without the SISO MEMORY ──────────
        # `siso_route.siso_fuse` carries the decomposition and the
        # G-tiling rule.  Both modes are ROTATING-ONLY (the
        # rotation-free route already runs SISO forward + general
        # backward and holds nothing) and both send the backward to the
        # general 5-kernel route, so `ctx.siso_bwd` is False under them.
        fuse = siso_fuse() if (use_siso and theta is not None) else "0"
        # The SISO forward stores S_all as a by-product of its own scan,
        # and the SISO ROTATING backward reads that store (siso_bwd).  The
        # rotation-free route (theta None) runs the GENERAL backward, which
        # already rebuilds S_all by the STATE_ONLY replay, so there the
        # override is only a policy: SSD_MIMO_SALL_SISO (default 1) lets the
        # SALL=recompute default apply (memory: no S_all save, no S_all
        # kept across fwd->bwd).  =0 restores the save-mode.
        if use_siso and (theta is None and _env("SALL_SISO", "1") == "1"
                         and fuse not in ("b", "c")):
            pass
        elif use_siso:
            recompute = False
        if out_norm_w is not None and not (
                has_gate and not in_kernel and _env("GATE_FUSED", "1") == "1"):
            raise NotImplementedError(
                "the fused output RMSNorm rides the DEDICATED gate "
                "epilogue kernel: it needs gate_z, "
                "SSD_MIMO_GATE_INKERNEL=0 and SSD_MIMO_GATE_FUSED=1")
        if has_dskip and not has_gate:
            raise NotImplementedError(
                "the fused D-skip rides the gate epilogue (its adjoint "
                "reads the same y_pre tile); pass gate_z, or apply the "
                "skip module-side")
        ypre = None
        if in_kernel:
            Bsz = x_raw.shape[0]
            nc = delta_cumsum.shape[2]
            O = (Ctil.shape[-1] if (Ctil.dim() == (6 if bc_grouped else 5))
                 else 1)
            if needs_bwd:
                # y_pre in the SAME native layout the kernel stores Y in,
                # so the backward's strided reads stay coalesced.
                ypre = (torch.empty(Bsz, H, O, G_SIZE, nc * T,
                                    device=x_raw.device, dtype=x_raw.dtype)
                        .view(Bsz, H, O, G_SIZE, nc, T)
                        .permute(0, 1, 4, 5, 2, 3))
            # The store's dtype -- see `_gate_out_io()` for why the IO
            # dtype is exact here and is the default, and for
            # the lever that restores the fp32 store.
            gate_dt = x_raw.dtype if _gate_out_io() else torch.float32
        else:
            gate_dt = None
        # ── Mamba-3's two-tap trapezoid, in the scan kernel ───────────
        # `dw` is then the RAW dt and `trap_lam` the (post-sigmoid)
        # lambda; the kernel forms the collapsed write scale AND
        # subtracts the rank-1 output correction at its store.  The
        # backward needs (k_t . q_t) back: saving the [B, H, L] vector the
        # forward already computed is far cheaper than four
        # [B, L, N] contractions to rebuild it.
        has_trap = trap_lam is not None
        dot_out = None
        if has_trap and needs_bwd:
            # At R, O > 1 the saved dot is the
            # O x R matrix d_t = C~_t^T B~_t per (kernel head, token),
            # not the scalar k_t . q_t -- the adjoint consumes every
            # (o, r) entry.  At R = O = 1 this is `empty_like(dw)`, the
            # [B, H, L_padded] shape.
            R_trap = x_raw.shape[3]
            O_trap = (Ctil.shape[-1]
                      if Ctil.dim() == (6 if bc_grouped else 5) else 1)
            dot_out = (torch.empty_like(dw) if (R_trap == 1 and O_trap == 1)
                       else torch.empty(dw.shape + (O_trap, R_trap),
                                        device=dw.device,
                                        dtype=torch.float32))
        # ── The per-chunk phase offset is built HERE, in one kernel ───
        # `theta_incr` says `theta` holds per-token INCREMENTS and the
        # offset has NOT been handed in.  Building it inside the Function
        # (a) makes it one launch and
        # (b) keeps it OFF the autograd graph, so the cross-chunk term of
        # d(increment) is one kernel's `+` in the backward rather than a
        # cumsum-backward chain over the [B, L, H, N/2] tensor.
        phase_internal = bool(theta_incr and theta_off is None)
        # ── Build the CUMULATED phase, not the offsets ────────────────
        # With the offsets the kernels rebuild the running phase with an
        # intra-chunk `tl.cumsum` ONCE PER HEAD -- a cross-thread scan
        # inside four register-starved bodies.  Building the cumulated
        # phase here instead is the SAME streaming pass plus its store,
        # and every kernel then takes the PHASE_CUMSUM = False path.
        # Memory is neutral: `theta_cum` replaces the
        # increments in the saved context and the [B, H, nc, K] offsets go
        # away.  SSD_MIMO_PHASE_PREFIX=0 restores the offset form.
        # ... and ONLY when there is a backward to pay for it: the
        # forward alone rebuilds the phase once per (head, chunk) inside
        # the ONE kernel that reads it, which is cheaper than a large
        # store.
        phase_prefix = bool(
            phase_internal
            and (needs_bwd or use_siso)
            and (_env("PHASE_PREFIX", "1") == "1" or use_siso))
        # the cumulated tensor is built under no_grad, so the INCREMENT
        # tensor's requires_grad has to be read before the rebind
        theta_wants_grad = theta is not None and theta.requires_grad
        if phase_prefix:
            from .m3_epilogue import phase_prefix_build
            with torch.no_grad():
                theta = phase_prefix_build(theta)
            theta_off = None
        elif phase_internal:
            from .m3_epilogue import phase_offset_build
            with torch.no_grad():
                theta_off = phase_offset_build(theta)
        # ── ROTSPLIT: Mamba-3's backward decomposition ────────────────
        # The rotated operands are materialised HERE, once, per KERNEL
        # HEAD; the scan (forward, the STATE_ONLY replay and all five
        # backward kernels) then runs its REAL-mode bodies on them, with
        # no angle, no partner tile and no per-token dTheta inside any
        # `tl.dot` kernel.  One elementwise kernel (`rot_adjoint`) owns
        # every rotation gradient.  See m3_epilogue.py section 5.
        #
        # Materialising them in a PROLOGUE (rather than storing B~/C~
        # from the fused forward kernel) is the same tensor at the same
        # dtype, and it has two advantages: (a) the forward kernel
        # becomes rotation-free too (it compiles the real body, so
        # `_pair_swap_2d` leaves the forward as well), and (b) no line of
        # any fused kernel changes, so real-mode bit-identity is by
        # construction rather than by branch discipline.  It costs one
        # extra streaming READ of the rotated tiles in the forward.
        ctx.rot_split = False
        rs_theta = rs_theta_h = None
        if (theta is not None and needs_bwd and not use_siso
                and bias_pre_rot and rot_bias is None
                and theta_off is None
                and (_rot_bwd_split() or _rot_route_split(
                    x_raw, H, delta_cumsum.shape[2] * T, T, R_sp=x_raw.shape[3],
                    O_sp=(Ctil.shape[-1] if (Ctil.dim() == (6 if bc_grouped else 5)) else 1),
                    envelope=dict(Ctil=Ctil, Btil=Btil, x_raw=x_raw, G_SIZE=G_SIZE, N=N,
                                  bc_grouped=bc_grouped, Gb=Gb, Gc=Gc, theta_off=theta_off,
                                  in_kernel_gate=in_kernel, T=T, theta=theta, theta_h=theta_h,
                                  rot_bias=rot_bias), pair_half=pair_half, thh=theta_h is not None))):
            from .m3_epilogue import rot_prologue
            ctx.rs_bshape, ctx.rs_bstride = tuple(Btil.shape), Btil.stride()
            ctx.rs_cshape, ctx.rs_cstride = tuple(Ctil.shape), Ctil.stride()
            ctx.rs_bdtype, ctx.rs_cdtype = Btil.dtype, Ctil.dtype
            with torch.no_grad():
                Ctil, Btil = rot_prologue(
                    Ctil, Btil, theta,
                    theta_h.float().contiguous()
                    if theta_h is not None else None,
                    beta.float().contiguous() if beta is not None else None,
                    gamma.float().contiguous() if gamma is not None else None,
                    H, Gb=Gb, Gc=Gc, bc_grouped=bc_grouped,
                    out_dtype=_rot_split_dtype(x_raw.dtype, precision),
                    pair_half=pair_half)
            ctx.rot_split = True
            ctx.rs_need_dthh = (theta_h is not None
                                and theta_h.requires_grad)
            ctx.rs_need_dbeta = beta is not None and beta.requires_grad
            ctx.rs_need_dgamma = gamma is not None and gamma.requires_grad
            rs_theta, rs_theta_h = theta, theta_h
            ctx.H_rs = H
            ctx.rs_beta_shape = (tuple(beta.shape) if beta is not None
                                 else None)
            ctx.rs_gamma_shape = (tuple(gamma.shape) if gamma is not None
                                  else None)
            ctx.rs_thh_shape = (tuple(theta_h.shape)
                                if theta_h is not None else None)
            ctx.rs_Gb, ctx.rs_Gc = Gb, Gc
            ctx.rs_bc_grouped = bc_grouped
            # every angle-bearing argument now leaves the scan
            theta = theta_h = beta = gamma = rot_bias = None
            bias_pre_rot = False
            bc_grouped = True
            Gb = Gc = H
            # ── the pair LAYOUT leaves the scan with them ─────────────
            # `pair_layout` names WHICH two state coordinates carry
            # rotation pair k.
            # With the rotation hoisted out, the scan's bodies are
            # layout-AGNOSTIC: the state axis is an unlabelled contraction
            # index to every one of them, so the interleaved (non-spilling)
            # bodies compute the half model exactly.  Only `rot_prologue` /
            # `rot_adjoint` need to know, and there the half form is a
            # pure win -- no MMA, so no output width to halve, and the
            # `tl.split`/`tl.join` pair partner becomes a slice.
            ctx.rs_pair_half = bool(pair_half)
            pair_half = False
        siso_aux = None
        # CHUNKPAR (chunkpar.py): the rotation-free SISO
        # forward as three chunk-parallel phases when the serial one-CTA-
        # per-(b, h) walk would under-fill the device.
        use_cp = False
        if use_siso and theta is None:
            from .chunkpar import chunkpar_wanted, cp_refusal
            cp_why = cp_refusal(theta=theta, theta_h=theta_h,
                                rot_bias=rot_bias, x0=x0, fuse=None,
                                dcs_ink=dcs_ink, T=T, N=N, G_SIZE=G_SIZE)
            use_cp = (cp_why is None and chunkpar_wanted(
                x_raw.shape[0] * H, delta_cumsum.shape[2], x_raw.device,
                part="fwd" if needs_bwd else "fwd_nograd", dtype=x_raw.dtype))
            from . import chunkpar as _cpm
            _cpm._LAST_FWD_ROUTE["fwd"] = ("chunkpar" if use_cp
                                           else (cp_why or "serial"))
            if _env("CP_TRACE", "0") == "1":
                print("CP_TRACE fwd=" + _cpm._LAST_FWD_ROUTE["fwd"], flush=True)
        if use_cp:
            from .chunkpar import (cp_chunk_fwd, cp_resource_refused,
                                   cp_resource_refuse)
            import triton as _tr
            _cpk = (str(x_raw.device), precision, str(Ctil.dtype),
                    str(x_raw.dtype), T, N, G_SIZE, _env("CP_FWD_GT", "0"),
                    _env("CP_KSUB_FWD", ""), _env("CP_SHARED_CB", "0"))
            if cp_resource_refused("fwd", _cpk):
                use_cp = False
            else:
                try:
                    Y, S_all, S_final, siso_aux = cp_chunk_fwd(
                        Ctil, Btil, x_raw, delta_cumsum, a_log,
                        beta=beta, gamma=gamma, dw=dw, trap_lam=trap_lam,
                        x0=x0, d_skip=d_skip, precision=precision,
                        store_sall=needs_bwd and not recompute,
                        needs_bwd=needs_bwd,
                        y_token_major=(siso_ytm() or (
                            y_token_major and (
                                not has_gate or out_norm_w is not None))))
                    if has_trap:
                        dot_out = siso_aux["dot"]
                except _tr.runtime.errors.OutOfResources as _oor:
                    # refused by name for this configuration; the serial
                    # SISO forward below serves (it recomputes everything)
                    cp_resource_refuse("fwd", _cpk, _oor)
                    use_cp = False
            if not use_cp:
                _cpm._LAST_FWD_ROUTE["fwd"] = (
                    "serial (chunkpar refused: out of shared memory)")
                if _env("CP_TRACE", "0") == "1":
                    print("CP_TRACE fwd=" + _cpm._LAST_FWD_ROUTE["fwd"],
                          flush=True)
        if use_cp:
            pass
        elif use_siso:
            from .siso_fwd_kernel import siso_chunk_fwd
            Y, S_all, S_final, siso_aux = siso_chunk_fwd(
                siso_bc(Ctil), siso_bc(Btil), x_raw, delta_cumsum, a_log,
                theta=theta, theta_h=theta_h, rot_bias=rot_bias,
                beta=beta, gamma=gamma, bias_pre_rot=bias_pre_rot,
                dw=dw, trap_lam=trap_lam, x0=x0, d_skip=d_skip,
                precision=precision, store_sall=needs_bwd and not recompute,
                needs_bwd=needs_bwd,
                # `siso_ytm()` asks for the token-major store on
                # THIS route only -- the backward's `dY` tile is then
                # G-contiguous and the SISO backward scan reads it faster
                # (`siso_route.siso_ytm`).  The
                # `has_gate` guard of the general route does
                # not apply here: with the dedicated epilogue the scan's
                # `Y` is `y_pre`, which only the gate adjoint reads, and
                # it reads it through its own stride set either way.
                y_token_major=(siso_ytm() or (
                    y_token_major and (
                        not has_gate or out_norm_w is not None))),
                fuse=(fuse if fuse in ("b", "c") else None))
            if has_trap:
                # mode "b" stages the maps, so its `dot` carries the
                # rank-1 term phase 2 reads back (`raw * g_t`); the
                # general backward's `trap_prologue` takes the raw one,
                # which phase 1 then stores next to it.
                dot_out = (siso_aux["dot_raw"] if fuse == "b"
                           else siso_aux["dot"])
            if fuse in ("b", "c"):
                # THE POINT OF THE LEVER: nothing the forward staged is
                # retained.  Dropping the only reference here is what
                # turns the 503 MB of `[B, L, H, N]` maps (mode "b") from
                # a forward-HELD save into a transient peak; mode "c"
                # never allocated them.  `dot_out` above is the one
                # tensor the backward still needs, and it is saved
                # through the general route's own slot.
                siso_aux = None
        else:
            Y, S_all, S_final = fused_chunk_fwd(
                Ctil, Btil, x_raw, delta_cumsum, a_log,
                theta_h=theta_h, theta=theta,
                beta=beta, gamma=gamma, rot_bias=rot_bias,
                bias_pre_rot=bias_pre_rot,
                x0=x0,
                precision=precision,
                bc_grouped=bc_grouped, Gb=Gb, Gc=Gc,
                dw=dw, store_sall=needs_bwd and not recompute,
                gate_z=gate_z if in_kernel else None,
                # the D-skip is at the STORE either way (see the block above),
                # so `Y` is post-skip and the gate adjoint's y_pre operand is
                # the right tensor whichever form the gate took.
                d_skip=d_skip,
                ypre_out=ypre,
                gate_out_dtype=gate_dt,
                trap_lam=trap_lam, dot_out=dot_out,
                theta_off=theta_off,
                # Only the RETURNED tensor takes the token-major store.
                # With the gate on the dedicated epilogue (the default) the
                # scan's own Y is y_pre, which nothing but the gate adjoint
                # reads -- and that reads it through its own stride set.
                # With the fused output norm the epilogue is gridded
                # per TOKEN ROW, so y_pre wants the token-major store too --
                # its channel row is then contiguous.  (Correctness does not
                # depend on it: every operand is addressed through its own
                # (b, t, h, g) stride set.)
                pair_half=pair_half,
                dcs_ink=dcs_ink,
                y_token_major=y_token_major and (in_kernel or not has_gate
                                                 or out_norm_w is not None))
        ctx.recompute_sall = recompute
        # the local chart cotangent: the geometry and the probe's length
        # (the backward returns F [B, L, G, NF] as the probe's gradient)
        ctx.chart = (None if chart_probe is None else
                     (tuple(chart_geometry), tuple(chart_probe.shape)))
        if ctx.chart is not None and use_siso and theta is not None:
            raise NotImplementedError(
                "the local chart cotangent is not wired into the SISO "
                "backward schedule")
        ctx.sall_cp = False
        # SALL_SISO route: the backward may re-chunk (the host delta prefix is
        # rebuilt at T_bwd from the raw delta)
        ctx.sall_rebuild_dcs = bool(recompute and use_siso and theta is None)
        if recompute and use_siso:
            from .chunkpar import cp_replay_ok
            ctx.sall_cp = bool(use_cp and cp_replay_ok(delta_cumsum.shape[2]))
        ctx.siso = use_siso
        ctx.siso_why = siso_why
        # (the aux maps ride save_for_backward -- `_save_with_ctx_tensors`)
        siso_aux_keep = siso_aux if needs_bwd else None
        ctx.siso_prec = (precision.split(":")[1] if ":" in precision
                         else precision)
        ctx.siso_fuse = fuse
        ctx.siso_bwd = bool(use_siso and theta is not None
                            and fuse not in ("b", "c"))
        if not torch.compiler.is_compiling():
            # ROUTE TELEMETRY (contract suite, tests/_contract/adapters/ssd.py):
            # the route this call took, read by the served-route check.  Host
            # side only; skipped while tracing like `_note_dispatch`.
            LAST_ROUTE.clear()
            LAST_ROUTE.update(
                route=("chunkpar" if use_cp else "siso" if use_siso
                       else "rotsplit" if ctx.rot_split else "general"),
                fwd=("chunkpar" if use_cp else "siso" if use_siso else "general"),
                bwd=("siso" if ctx.siso_bwd else "general"),
                rot_split=bool(ctx.rot_split), siso_why=siso_why, T=T,
                in_kernel_gate=bool(in_kernel))

        rstd = None
        if in_kernel:
            Yg = Y
            Y = ypre
        elif has_gate:
            # ONE fused pass in the kernel's own native layout.
            # A torch chain would cost four elementwise passes
            # and a full [B, H, nc, T, 1, G] fp32 `Z` materialisation,
            # and its DENSE output would turn the caller's permute+reshape
            # from a free view into another copy.
            # SSD_MIMO_GATE_FUSED=0 restores the torch chain (A/B lever).
            if out_norm_w is not None:
                # ── The module's output gated RMSNorm, HERE ──────────
                # Mamba-2's node fuses its output norm into the same
                # kernel as the gate (`layernorm_gated`); an ATen pass
                # over the whole activation costs extra traffic.  The
                # gate's own kernel already reads y_pre and z, so the
                # row RMS costs one in-register reduction and the
                # PRE-norm y_core never has to be written or saved.
                from .gate_epilogue import gate_norm_fwd
                Yg, rstd = gate_norm_fwd(
                    Y, gate_z, T, out_norm_w, out_norm_eps,
                    out_dtype=x_raw.dtype if _gate_out_io() else None)
                from ...layers import regen as _regen
                if (_regen.wants("out", theta is None and theta_h is None
                                 and rot_bias is None and x_raw.shape[3] == 1
                                 and Y.shape[4] == 1) and needs_bwd):
                    # SSD_MIMO_HELD_REGEN=out: out_proj need not keep this tensor
                    _od = x_raw.dtype if _gate_out_io() else None
                    _Y, _gz, _T, _w, _e = Y, gate_z, T, out_norm_w, out_norm_eps

                    def _out_regen():
                        from .gate_epilogue import gate_norm_fwd as _gnf
                        return _gnf(_Y, _gz, _T, _w, _e, out_dtype=_od)[0]
                    _regen.register(Yg, _out_regen, track=False)
            elif _env("GATE_FUSED", "1") == "1":
                from .gate_epilogue import ascalar_epilogue_fwd
                # ── SSD_MIMO_GATE_OUT_IO on the epilogue route ───────
                # An fp32 store would force the module to cast the whole
                # [B, H, G, L] buffer back to the IO dtype (read + write
                # forward, and the same again as the cast's adjoint in the
                # backward).
                #
                # `_gate_out_io()` carries the proof that this is
                # exact; default ON.
                Yg = ascalar_epilogue_fwd(
                    Y, gate_z, T, token_major=y_token_major,
                    out_dtype=x_raw.dtype if _gate_out_io() else None)
            else:
                Bsz, _H, nc, _T, O, _G = Y.shape
                Z_5d = (gate_z.view(Bsz, nc, _T, _H, _G)
                        .permute(0, 3, 1, 2, 4).unsqueeze(4).float())
                Yg = Y * (Z_5d * torch.sigmoid(Z_5d))
        else:
            Yg = Y

        saved = [Ctil, Btil, x_raw, dw, delta_raw, a_log,
                 None if recompute else S_all]
        if has_gate:
            # `ascalar_epilogue_bwd` addresses Ypre, dYpre and dOut
            # with THREE independent stride sets, so the strided
            # view of the native-layout buffer is saved AS IS (a
            # contiguous copy would cost the same bytes the view already
            # points at).
            saved.extend([Y if (in_kernel or _env("GATE_FUSED", "1") == "1")
                          else Y.contiguous(), gate_z])
        if has_dskip:
            saved.append(d_skip)
        if has_trap:
            saved.extend([trap_lam, dot_out if dot_out is not None
                          else trap_lam])
        # `saved` is handed to `_save_with_ctx_tensors` at the END of this
        # forward, with the named tensors (theta, beta, ...) appended.
        ctx.has_dskip = has_dskip
        ctx.in_kernel_gate = in_kernel
        # The fused output norm's saved state (rstd is [B, L] fp32;
        # y_core is NOT saved -- the adjoint
        # rebuilds it from y_pre and z, which it loads anyway).
        ctx.out_norm_eps = out_norm_eps
        ctx.need_dnormw = bool(out_norm_w is not None
                               and out_norm_w.requires_grad)
        ctx.has_trap = has_trap
        ctx.need_dlam = bool(has_trap and trap_lam.requires_grad)
        # This saved phase (`theta_keep`) is the only
        # place the (kernel-head-expanded) fp32 phase is retained across
        # the whole fwd->bwd window.  It cannot be dropped outright: both
        # `recompute_sall`'s fwd replay and `fused_chunk_bwd_5kernel`
        # need the actual numeric phase.  What IS available is
        # storing a bf16 rounding (halves its size): every reader
        # immediately upcasts (`fused_fwd_kernel.py`/`fused_bwd_kernel.py`
        # upcast in registers), so this costs only that rounding, not a
        # second fp32 copy.  Exclusions: the SISO
        # route's own backward reads `ctx.theta` straight into a Triton
        # pointer with no upcast at the call site; and
        # `precision == "ieee"` is the fp32/fp64 numerics ORACLE
        # whose exactness gates exist
        # to catch rounding regressions like this one.  Also only when the
        # scan's IO dtype is itself 16-bit: on an fp32 IO route a bf16 phase
        # moves grad(x) by 3e-3 against a 2e-4 gate.
        # The backward keeps the fp32 phase BY DEFAULT
        # (kernel option `theta_bwd_fp32`, default True; False restores the bf16 twin as
        # an opt-in A/B).  The bf16 twin of the running phase has no bits
        # left at the |Theta| training reaches (4e5 rad on trained complex
        # checkpoints: every scan gradient wrong at io), and even a phase
        # wrapped to [-pi, pi] carries 6e-3 rad in bf16 -- the backward
        # then differentiates a different rotation.
        _theta_bf16_ok = (theta is not None and not ctx.siso_bwd
                         and precision != "ieee"
                         and _env("THETA_BWD_FP32", "1") != "1"
                         and theta.dtype == torch.float32
                         and x_raw.dtype in (torch.bfloat16, torch.float16))
        # The rounding is not a pass of its own when the caller hands in the
        # twin the phase law's launch already stored (`theta_lo`, same
        # shape, bf16); `fused_chunk_fwd` / `fused_chunk_bwd_5kernel` then
        # read the bf16 pointer directly (their loads upcast in registers).
        # The exclusions above decide `_theta_bf16_ok`, and the twin is only
        # ever used under it.
        if _theta_bf16_ok:
            theta_keep = (theta_lo if (theta_lo is not None
                                       and theta_lo.dtype == torch.bfloat16
                                       and theta_lo.shape == theta.shape)
                          else theta.to(torch.bfloat16))
        else:
            theta_keep = theta
        # When set, `theta` above is the per-token angle INCREMENT
        # and `theta_off` (saved below) is the phase entering each chunk;
        # the kernels rebuild the running phase with an intra-chunk cumsum.
        ctx.phase_internal = phase_internal
        # `ctx.theta` is then the CUMULATED phase and the gradient
        # that comes back is dL/dtheta_cum; one pass turns it into the
        # increment gradient (`phase_prefix_bwd`).
        ctx.phase_prefix = phase_prefix
        ctx.bias_pre_rot = bias_pre_rot
        ctx.pair_half = bool(pair_half)
        # SPANDELTA: the saved `delta_cumsum` slot holds
        # the RAW chunked delta on every route; with `dcs_ink` the kernels
        # form the (segmented) prefix themselves, otherwise the backward
        # rebuilds it host-side.  The dDelta the kernels return is
        # dL/ddelta itself (`fused_bwd_kernel._SPANDELTA_NOTE`).
        ctx.dcs_ink = bool(dcs_ink)
        ctx.precision = precision
        ctx.Gb = Gb
        ctx.Gc = Gc
        ctx.bc_grouped = bc_grouped
        ctx.has_gate = has_gate
        ctx.T = T
        # The chunk the BACKWARD runs at.  0 / T
        # means "the forward's", which is the default; anything else is
        # honoured only if
        # `_rechunk` proves every chunked artefact is a pure token
        # regrouping in memory (see the comment block at `_rechunk`).
        ctx.chunk_bwd = int(chunk_bwd or 0) or _chunk_bwd_env()
        ctx.need_dthh = (theta_h is not None and theta_h.requires_grad)
        ctx.need_dtheta = theta_wants_grad
        ctx.need_dtheta_off = (theta_off is not None
                               and theta_off.requires_grad)
        ctx.need_dbeta = (beta is not None and beta.requires_grad)
        ctx.need_dgamma = (gamma is not None and gamma.requires_grad)
        ctx.need_drot_bias = (rot_bias is not None and rot_bias.requires_grad)
        if ctx.rot_split:
            # ROTSPLIT nulled the angle-bearing arguments above, so the
            # three flags they decide are read off the ORIGINALS.
            ctx.need_dthh = ctx.rs_need_dthh
            ctx.need_dbeta = ctx.rs_need_dbeta
            ctx.need_dgamma = ctx.rs_need_dgamma
        ctx.need_dx0 = (x0 is not None and x0.requires_grad)
        ctx.need_ddw = dw.requires_grad
        # The entry point's `kernel_options` scope is gone by the time
        # the backward runs, and the backward REPLAYS this forward.
        ctx._kernel_options = _ko.current("ssd_mimo")
        ctx._measured_plan = _measured_plan.current()
        # Every tensor the backward reads goes through
        # save_for_backward (the positional `saved` first, then the named
        # ones), so the engine frees them after the backward and
        # `retain_graph=True` keeps them for the next one.
        _save_with_ctx_tensors(ctx, saved, dict(
            theta=theta_keep, theta_h=theta_h, theta_off=theta_off,
            beta=beta, gamma=gamma, rot_bias=rot_bias, x0=x0,
            out_norm_w=out_norm_w,
            out_norm_rstd=rstd if out_norm_w is not None else None,
            rs_theta=rs_theta, rs_theta_h=rs_theta_h),
            siso_aux=siso_aux_keep)
        return Yg, S_final

    @staticmethod
    @_no_double_backward("SSD-MIMO scan")
    @_measured_plan.backward_scope
    @_ko.scoped_ctx("ssd_mimo")
    def backward(ctx, dY, dS_final):
        # the named tensors (theta, beta, ...) are read out of
        # `saved_tensors` by this view; see `_save_with_ctx_tensors`.
        ctx = saved_view(ctx)
        sv = list(ctx.saved_tensors)
        Ctil, Btil, x_raw, dw, delta_cumsum, a_log, S_all = sv[:7]
        # ── NEEDS: what the caller asked for.  A
        # captured / combined-node ctx has no `needs_input_grad` and gets
        # everything.  Input order: Ctil, Btil, x_raw, dw, delta_cumsum,
        # a_log, theta_h, theta, theta_off, gate_z, d_skip, beta, gamma,
        # x0, rot_bias, trap_lam.
        _nig = getattr(ctx, "needs_input_grad", None)
        if _nig is None or len(_nig) < 16:
            _nig = (True,) * 16
        _needs = dict(
            ddelta=bool(_nig[4] or _nig[5]),
            dtheta=bool(ctx.need_dtheta and (_nig[7] or _nig[8])),
            dthh=bool(_nig[6]), dbeta=bool(_nig[11]), dgamma=bool(_nig[12]),
            dx0=bool(_nig[13]), drb=bool(_nig[14]),
            ddw=bool(_nig[3]), dlam=bool(_nig[15]))
        need_ddsk = bool(_nig[10])
        # SPANDELTA: the saved slot is the RAW chunked delta (see forward)
        delta_raw0 = delta_cumsum
        if not getattr(ctx, "dcs_ink", False):
            from .fused_fwd_kernel import seg_prefix_host
            delta_cumsum = seg_prefix_host(delta_raw0, a_log)
        idx_t = 7 + (2 if ctx.has_gate else 0) + (1 if ctx.has_dskip else 0)
        trap_lam = trap_dot = trap_nxt = trap_nd = trap_dsdot = None
        dw_eff = dw
        fuse_trap = False
        siso_bwd = bool(getattr(ctx, "siso_bwd", False))
        if ctx.has_trap and not siso_bwd:
            # `dw` is the RAW dt in this mode.  The collapsed write scale
            # is rebuilt here, OUTSIDE the graph -- it is [B, H, L], so
            # recomputing it is cheap, and building it under autograd in
            # the FORWARD would add many launches.
            #
            # One Triton prologue does the whole elementwise chain
            # (including the zero-fill of Kc's dSdot buffer, which is free
            # in a pass that already writes three vectors of that shape).
            trap_lam, trap_dot = sv[idx_t], sv[idx_t + 1]
            # The fused adjoint is gated against the fp64 oracles directly.
            fuse_trap = True
            with torch.no_grad():
                trap_nxt, dw_eff, trap_nd, trap_dsdot = trap_prologue(
                    dw, trap_lam, trap_dot)

        elif ctx.has_trap:
            # ── The SISO schedule's own backward owns the whole
            # two-tap chain rule (its phase 1 stored `Scale` and
            # `Gamma`), so the host prologue is skipped -- only the
            # saved tensors are unpacked here.
            trap_lam, trap_dot = sv[idx_t], sv[idx_t + 1]

        # ── The DECOUPLED BACKWARD CHUNK ─────────────────────────────
        # Everything below this point -- the STATE_ONLY replay, the five
        # fused kernels and `delta_adjoint` -- runs at `T_bwd`; the
        # gradients are re-viewed back to the forward's chunking just
        # before the return.  The whole plan is refused BY NAME, and
        # falls back to `T_bwd = T_fwd` (the default schedule, bit for
        # bit), unless every condition below holds.
        T_fwd = delta_cumsum.shape[-1]
        _cb_asked = int(getattr(ctx, "chunk_bwd", 0) or 0)
        T_bwd, cb_reason = T_fwd, ""
        dY_cb = None
        # B / C carry the group axis at dim 2 when grouped, so the
        # token-within-chunk axis moves; delta and dY do not.
        _bcta = 3 if ctx.bc_grouped else 2
        if _cb_asked and _cb_asked != T_fwd:
            T_bwd = _cb_asked
            # DCS_INK off: `delta_cumsum` is the host segmented prefix, a
            # T_fwd artefact -- but the RAW chunked delta (SPANDELTA slot)
            # is a pure token regrouping, so the prefix is simply REBUILT at
            # T_bwd from it (one ATen prefix on a [B, H, L] tensor).  Only
            # for the SALL_SISO recompute route.
            _dcs_rebuild = (not getattr(ctx, "dcs_ink", False)
                            and getattr(ctx, "sall_rebuild_dcs", False))
            _d_re = None
            if _dcs_rebuild:
                _d_raw_T = _rechunk(delta_raw0, 2, 3, T_bwd, T_fwd)
                if _d_raw_T is not None:
                    from .fused_fwd_kernel import seg_prefix_host
                    _d_re = seg_prefix_host(_d_raw_T, a_log)
            _cands = dict(
                delta_cumsum=(_d_re if _dcs_rebuild else
                              _rechunk(delta_cumsum, 2, 3, T_bwd, T_fwd)),
                Btil=_rechunk(Btil, 1, _bcta, T_bwd, T_fwd),
                Ctil=_rechunk(Ctil, 1, _bcta, T_bwd, T_fwd),
                dY=_rechunk(dY, 2, 3, T_bwd, T_fwd),
            )
            if siso_bwd:
                cb_reason = "siso_bwd"
            elif not ctx.recompute_sall:
                cb_reason = "SALL=io (S_all is a T_fwd artefact)"
            elif not (getattr(ctx, "dcs_ink", False)
                      or getattr(ctx, "sall_rebuild_dcs", False)):
                cb_reason = "DCS_INK off (delta_cumsum is a T_fwd scan)"
            elif ctx.theta is not None or ctx.rot_bias is not None:
                cb_reason = "rotation (theta / rot_bias are chunk-shaped)"
            elif getattr(ctx, "pair_half", False):
                cb_reason = "pair_half"
            elif getattr(ctx, "rot_split", False):
                cb_reason = "rot_split"
            else:
                _bad = [k for k, v in _cands.items() if v is None]
                if _bad:
                    cb_reason = "not a token regrouping: " + ",".join(_bad)
            if cb_reason:
                T_bwd = T_fwd
            else:
                delta_cumsum = _cands["delta_cumsum"]
                Btil, Ctil = _cands["Btil"], _cands["Ctil"]
                # dY is re-viewed only AFTER the gate / norm epilogue,
                # which reads it against `Y_pre` at the FORWARD chunking.
                dY_cb = _cands["dY"]
        # The originals, so the shared-memory
        # fallback below can re-chunk from them at a SHORTER T_bwd.
        _cb0 = (Btil, Ctil, delta_cumsum, dw_eff, x_raw, a_log)
        _note_chunk_bwd(dict(T_fwd=T_fwd, T_bwd=T_bwd, asked=_cb_asked,
                             refused=cb_reason))

        def _replay_sall(C_, B_, dcs_):
            """The STATE_ONLY forward replay, at whatever chunk it is
            handed.  A closure because the shared-memory
            fallback re-runs it at a shorter `T_bwd`."""
            _, sa, _ = fused_chunk_fwd(
                C_, B_, x_raw, dcs_, a_log,
                theta_h=ctx.theta_h, theta=ctx.theta,
                beta=ctx.beta, gamma=ctx.gamma, rot_bias=ctx.rot_bias,
                bias_pre_rot=ctx.bias_pre_rot,
                x0=ctx.x0, precision=ctx.precision,
                bc_grouped=ctx.bc_grouped, Gb=ctx.Gb, Gc=ctx.Gc,
                dw=dw_eff, store_sall=True, state_only=True,
                pair_half=ctx.pair_half,
                dcs_ink=getattr(ctx, "dcs_ink", False),
                theta_off=ctx.theta_off)
            return sa

        _cp_done = False
        if ctx.recompute_sall and getattr(ctx, "sall_cp", False):
            # SISO / chunk-parallel forward: replay through the SAME state
            # phases (bit-identical S_all at T_fwd, chunk-parallel, no Y)
            from .chunkpar import cp_chunk_fwd as _cpf, cp_replay_ok as _cpok
            from .siso_route import siso_bc as _sbc
            import triton as _tr
            if _cpok(delta_cumsum.shape[2]):
                try:
                    _, S_all, _, _ = _cpf(
                        Ctil, _sbc(Btil), x_raw, delta_cumsum, a_log,
                        beta=ctx.beta, gamma=None, dw=dw, trap_lam=trap_lam,
                        x0=ctx.x0, precision=ctx.precision, store_sall=True,
                        needs_bwd=False, states_only=True)
                    _cp_done = True
                except _tr.runtime.errors.OutOfResources:
                    pass
        if _cp_done:
            pass
        elif ctx.recompute_sall:
            S_all = _replay_sall(Ctil, Btil, delta_cumsum)

        dZ = None
        dDsk = None
        dX_skip = None
        # The D-skip's adjoint rides Kc's dX store when it can (see
        # `_dskip_in_kc`); the epilogue then never touches x_raw.
        d_skip = sv[9] if ctx.has_dskip else None
        O_bwd = (Ctil.shape[-1]
                 if Ctil.dim() == (6 if ctx.bc_grouped else 5) else 1)
        dskip_mode = _dskip_in_kc() if ctx.has_dskip else "off"
        if not (O_bwd == 1 and dw is not None):
            dskip_mode = "off"
        if ctx.out_norm_w is not None and ctx.has_dskip and O_bwd == 1 \
                and dw is not None:
            # The fused-norm epilogue is gridded per TOKEN ROW and x_raw
            # is [B, H, G, R, L] with L contiguous, so a token row of it
            # would be a stride-L gather: on this route Kc owns dD too,
            # and it is NOT optional -- there is no other kernel left
            # that reads x_raw.  `SSD_MIMO_DSKIP_IN_KC` therefore steers
            # the unfused-norm route only.
            dskip_mode = "all"
        dskip_kc = dskip_mode != "off"
        dNormW = None
        if ctx.has_gate and ctx.out_norm_w is not None:
            # ── gate + output RMSNorm, one adjoint kernel ─────────────
            # The norm's backward and the gate's backward read the SAME
            # two tensors (y_pre, z) and the same upstream gradient, and
            # an ATen pair in between would write and re-read a whole
            # activation.  Here y_core is recomputed in registers, so
            # the separate norm backward is gone entirely.
            from .gate_epilogue import gate_norm_bwd
            Y_pre, gate_z_saved = sv[7], sv[8]
            dY, dZ, dNormW = gate_norm_bwd(
                Y_pre, gate_z_saved, ctx.out_norm_w, ctx.out_norm_rstd,
                dY, ctx.T, dZ_out=getattr(ctx, "dz_out", None),
                real=_real_r1o1_call(ctx, x_raw, Y_pre))
            if not ctx.need_dnormw:
                dNormW = None
            elif dNormW is not None:
                dNormW = dNormW.to(ctx.out_norm_w.dtype)
        elif ctx.has_gate:
            Y_pre, gate_z_saved = sv[7], sv[8]
            from .gate_epilogue import ascalar_epilogue_bwd
            # dY is read through its OWN strides: the upstream
            # gradient arrives in whatever layout autograd produced and is
            # not materialised contiguous.
            dY_pre, dZ, dX_skip, dDsk = ascalar_epilogue_bwd(
                Y_pre, gate_z_saved,
                # `store_dxr=False` keeps the dD reduction here (see
                # `_dskip_in_kc`) while Kc owns the dX term.
                # NEEDS: under "dx" the epilogue's only D-skip work is the
                # dD reduction (Kc owns the dX term), so an unrequested dD
                # drops the x read and the atomic
                None if (dskip_mode == "all"
                         or (dskip_mode == "dx" and not need_ddsk))
                else (x_raw if ctx.has_dskip else None),
                None if (dskip_mode == "all"
                         or (dskip_mode == "dx" and not need_ddsk))
                else d_skip,
                dY if (ctx.in_kernel_gate
                       or _env("GATE_FUSED", "1") == "1")
                else dY.contiguous(),
                ctx.T,
                # The combined mixer node hands in the buffer dz is
                # to be written into -- its own in_proj-slab gradient --
                # so the split's backward `cat` never runs.
                dZ_out=getattr(ctx, "dz_out", None),
                store_dxr=(dskip_mode != "dx"),
                real=_real_r1o1_call(ctx, x_raw, Y_pre))
            dY = dY_pre
            if dDsk is not None:
                dDsk = dDsk.view(d_skip.shape).to(d_skip.dtype)

        prec = ctx.precision
        if ":" in prec:
            prec = prec.split(":")[1]

        # `dY` is re-viewed at T_bwd INSIDE the
        # launch loop below -- the gate / norm epilogue has had its look
        # at it against `Y_pre` at the FORWARD chunking first, and the
        # shared-memory fallback re-views it again at each rung.
        # `dY_cb` proved the INCOMING gradient legal; the epilogue's
        # `dY_pre` is a fresh buffer in `Y_pre`'s own layout, which is
        # legal too -- re-proved there, not assumed.

        # dY is consumed STRIDED: the kernels take dY strides, so the
        # upstream gradient (naturally [B, H, (O,) G, L] contiguous, or an
        # expanded ones tensor from .sum()) is read in place — no 67MB
        # .contiguous() materialisation.
        if siso_bwd and getattr(ctx, "chart", None) is not None:
            raise NotImplementedError(
                "the local chart cotangent is not wired into the SISO "
                "backward schedule")
        if siso_bwd:
            from .siso_route import siso_backward
            bwd_result = siso_backward(
                ctx, Ctil, Btil, x_raw, dw, delta_cumsum, a_log, S_all,
                dY, dS_final, trap_lam,
                d_skip if dskip_kc else None,
                dskip_mode == "all", needs=_needs)
        else:
          # ── The backward's shared-memory repair ──────────────────────
          # `smem.py` clamps the FORWARD's chunk by dry-compiling it, and
          # argues that the backward is safe because its STATE_ONLY
          # replay is strictly cheaper than the forward it replays.  That
          # argument does NOT cover `_bwd_kb1_kernel`, which stages a
          # [T, T] score tile the replay never builds (e.g. T = 128 needs
          # 106,496 B against sm_120's 101,376).
          # Kb1's own N-split ladder cannot repair it either: the split
          # shrinks the [T, N] tiles and leaves the [T, T] score tile alone.
          #
          # The repair that DOES fit the shape is the one `smem.py`
          # uses for the forward -- T-halving -- made expressible in the
          # backward by the decoupled chunk: re-chunk B / C / delta / dY
          # at T/2, replay `S_all` there, and run the five kernels at the
          # shorter chunk.  Exact up to round-off, for the same reason the
          # forward's clamp is (the chunk length only regroups the scan's
          # associativity).  Where the decoupled-chunk envelope refuses the
          # re-chunk the overflow is re-raised unchanged, with the refusal
          # NAMED in `chunk_bwd_log()` -- the cell stays a refusal rather
          # than becoming a silent wrong answer.
          _dY_pre_cb = dY
          _chart = getattr(ctx, "chart", None)
          _chart_req = None
          if _chart is not None:
              _chart_req = dict(mode=_chart[0][0], k=int(_chart[0][1]),
                                Mf=int(_chart[0][2]), Mp=int(_chart[0][3]),
                                emitter=(str(_chart[0][4]) if len(_chart[0]) > 4
                                         else "fused"),
                                L=int(_chart[1][1]), G=int(_chart[1][2]))
          for _cb_try in range(6):
            if T_bwd != T_fwd:
                _dyc = _rechunk(dY, 2, 3, T_bwd, T_fwd)
                dY = _dyc if _dyc is not None else _rechunk(
                    _dY_pre_cb.contiguous(), 2, 3, T_bwd, T_fwd)
            try:
              bwd_result = fused_chunk_bwd_5kernel(
                Ctil, Btil, x_raw, delta_cumsum, a_log, S_all,
                dY, theta_h=ctx.theta_h, theta=ctx.theta,
                dS_final=dS_final,
                precision=prec,
                Gb=ctx.Gb, Gc=ctx.Gc, bc_grouped=ctx.bc_grouped,
                beta=ctx.beta, gamma=ctx.gamma, x0=ctx.x0,
                rot_bias=ctx.rot_bias, bias_pre_rot=ctx.bias_pre_rot,
                dw=dw_eff,
                trap_nd=trap_nd if ctx.has_trap else None,
                theta_off=ctx.theta_off,
                # ── trapezoid fused adjoint ─────────────────────────────
                trap_dsdot=trap_dsdot, trap_lam=trap_lam, trap_dot=trap_dot,
                trap_nxt=trap_nxt, trap_dt=dw if fuse_trap else None,
                need_dlam=ctx.need_dlam, fuse_trap=fuse_trap,
                phase_internal=ctx.phase_internal,
                d_skip=d_skip if dskip_kc else None,
                pair_half=ctx.pair_half,
                dcs_ink=getattr(ctx, "dcs_ink", False),
                dskip_dd=(dskip_mode == "all"),
                **({} if _chart_req is None else dict(chart=_chart_req)))
            except _OutOfResources:
              # a CALIBRATED config that does not fit this geometry: fall back to the generic grid at the SAME chunk first
              # (calibration.smem_demote), only then halve the chunk
              from .calibration_fused import smem_demote as _smem_demote
              if _smem_demote(T_bwd):
                  continue
              _half = T_bwd // 2
              _why = ""
              if _cb_try == 5 or _half < 16:
                  _why = "T_bwd floor (16)"
              elif not ctx.recompute_sall:
                  _why = "SALL=io (S_all is a T_fwd artefact)"
              elif not getattr(ctx, "dcs_ink", False):
                  _why = "DCS_INK off (delta_cumsum is a T_fwd scan)"
              elif ctx.theta is not None or ctx.rot_bias is not None:
                  _why = "rotation (theta / rot_bias are chunk-shaped)"
              elif getattr(ctx, "pair_half", False):
                  _why = "pair_half"
              elif getattr(ctx, "rot_split", False):
                  _why = "rot_split"
              if not _why:
                  _B0, _C0, _D0 = _cb0[0], _cb0[1], _cb0[2]
                  _nb = _rechunk(_B0, 1, _bcta, _half, T_fwd)
                  _nc2 = _rechunk(_C0, 1, _bcta, _half, T_fwd)
                  _nd = _rechunk(_D0, 2, 3, _half, T_fwd)
                  if _nb is None or _nc2 is None or _nd is None:
                      _why = "not a token regrouping"
                  else:
                      Btil, Ctil, delta_cumsum = _nb, _nc2, _nd
                      T_bwd = _half
                      dY = _dY_pre_cb
                      if ctx.recompute_sall:
                          S_all = _replay_sall(Ctil, Btil, delta_cumsum)
                      _note_chunk_bwd(dict(
                          T_fwd=T_fwd, T_bwd=T_bwd, asked=_cb_asked,
                          refused="", smem_fallback=True))
                      warnings.warn(
                          "SSD-MIMO: the backward's score-path kernel does "
                          "not fit shared memory at chunk %d; re-running "
                          "the BACKWARD at chunk %d (REALMODE3 lever 3). "
                          "This changes only the chunkwise associativity "
                          "of the scan, not its math."
                          % (T_fwd if _cb_try == 0 else T_bwd * 2, T_bwd),
                          RuntimeWarning, stacklevel=2)
                      continue
              _note_chunk_bwd(dict(T_fwd=T_fwd, T_bwd=T_bwd,
                                   asked=_cb_asked, refused=_why,
                                   smem_fallback=False))
              # A REFUSAL BY NAME, not a bare Triton
              # crash.  The repair exists and the message says which
              # clause blocked it -- at R = O = 1 that is `DCS_INK off`,
              # and `SSD_MIMO_DCS_INK=1` alone makes the cell run.
              _cb_refuse(_why, T_bwd)
            break
        dX, dCtil, dBtil, dDelta, dThetaH, dTheta, dBeta, dGamma, dX0 = bwd_result[:9]
        dRotBias = bwd_result[9]
        dDW = bwd_result[10]
        dSdot = bwd_result[11]
        dThOff = bwd_result[12]
        dLam = bwd_result[13]
        if getattr(ctx, "rot_split", False):
            # ── ROTSPLIT: the ONE kernel that sees an angle ───────────
            # `dCtil` / `dBtil` above are the gradients w.r.t. the
            # ROTATED tiles (every writer -- Kb1, Kb2, Kc and the
            # trapezoid adjoint, which ran its REAL branch because
            # theta_h is already folded into C~ -- accumulated into
            # them).  This turns them into dTheta / dnu_h / dbeta /
            # dgamma and the caller's own dB / dC.
            from .m3_epilogue import rot_adjoint
            dCtil, dBtil, dBeta, dGamma, dTheta, dThetaH = rot_adjoint(
                Ctil, Btil, dCtil, dBtil,
                ctx.rs_theta,
                ctx.rs_theta_h.float().contiguous()
                if ctx.rs_theta_h is not None else None,
                ctx.H_rs, Gb=ctx.rs_Gb, Gc=ctx.rs_Gc,
                bc_grouped=ctx.rs_bc_grouped,
                B_shape=ctx.rs_bshape, B_stride=ctx.rs_bstride,
                C_shape=ctx.rs_cshape, C_stride=ctx.rs_cstride,
                need_dtheta=ctx.need_dtheta, need_dthh=ctx.need_dthh,
                need_dbeta=ctx.need_dbeta, need_dgamma=ctx.need_dgamma,
                R_out=(ctx.rs_bshape[-1] if len(ctx.rs_bshape) ==
                       (6 if ctx.rs_bc_grouped else 5) else 1),
                O_out=(ctx.rs_cshape[-1] if len(ctx.rs_cshape) ==
                       (6 if ctx.rs_bc_grouped else 5) else 1),
                pair_half=ctx.rs_pair_half)
            if dBeta is not None:
                dBeta = dBeta.view(ctx.rs_beta_shape)
            if dGamma is not None:
                dGamma = dGamma.view(ctx.rs_gamma_shape)
            if dThetaH is not None:
                dThetaH = dThetaH.view(ctx.rs_thh_shape)
        if ctx.phase_prefix and dTheta is not None and ctx.need_dtheta:
            # The kernels produced dL/dtheta_cum; ONE pass turns it
            # into the increment gradient (in-chunk reverse cumsum plus
            # the cross-chunk reverse exclusive prefix).  In place.
            from .m3_epilogue import phase_prefix_bwd
            phase_prefix_bwd(dTheta, None, cross=True)
        if dskip_mode == "all":
            dDsk = bwd_result[14].view(d_skip.shape).to(d_skip.dtype)

        # SPANDELTA (fused_bwd_kernel `_SPANDELTA_NOTE`): the kernels
        # returned dL/ddelta itself, so there is no reverse prefix sum here
        # (it telescoped O(1) chunk-mate pairs into the O(alpha) spanning
        # sum), and d a_log = sum_s delta_s dL/ddelta_s -- taken below,
        # once dDelta is back at the forward's chunking.
        da_log = None


        # ROTSPLIT: `Ctil`/`Btil` are the ROTATED saved tiles; the
        # gradient goes back to the CALLER's tensors, whose dtype was
        # recorded before the prologue rebound them.
        _c_dt = ctx.rs_cdtype if getattr(ctx, "rot_split", False) else Ctil.dtype
        _b_dt = ctx.rs_bdtype if getattr(ctx, "rot_split", False) else Btil.dtype
        if T_bwd != T_fwd:
            # Back to the FORWARD's chunking for the
            # return -- the shapes the autograd graph recorded.  All
            # three are re-views over the same storage, no copy: dB/dC
            # inherit their strides from the (group-major / R-major)
            # tiles they were allocated like (`zeros_like` preserves the
            # stride permutation of a dense tensor), and dDelta is
            # contiguous out of `delta_adjoint`.  Taken BEFORE the dtype
            # cast so a `.to()` copy cannot re-order them first.
            _bcta = 3 if ctx.bc_grouped else 2
            _dbv = _rechunk(dBtil, 1, _bcta, T_fwd, T_bwd)
            _dcv = _rechunk(dCtil, 1, _bcta, T_fwd, T_bwd)
            if _dbv is None or _dcv is None:
                raise RuntimeError(
                    "chunk_size_bwd: dB / dC came back in a layout that "
                    "is not a token regrouping; with the group-major / "
                    "R-major B_c this can only mean a new writer "
                    "allocated them contiguous at T_bwd")
            dBtil, dCtil = _dbv, _dcv
            if dDelta is not None:
                dDelta = _rechunk(dDelta.contiguous(), 2, 3, T_fwd,
                                  T_bwd)
        if dDelta is not None:
            # SPANDELTA: d a_log = sum_s delta_s dL/ddelta_s (a_log enters
            # only through |A| delta; every term carries its own delta)
            da_log = (delta_raw0 * dDelta.reshape(delta_raw0.shape)).sum(
                dim=tuple(i for i in range(delta_raw0.dim()) if i != 1))
        dCtil = dCtil.to(_c_dt)
        dBtil = dBtil.to(_b_dt)
        # dX comes back in x_raw.dtype (Kc stores dx_raw = dx_scaled * dw)
        if dX_skip is not None:
            # the D-skip's own x path: y_pre = scan(x) + D.x, so dx gets
            # BOTH terms.  dX_skip is already in x_raw's layout.
            dX = dX + dX_skip.to(dX.dtype)

        return (dCtil, dBtil, dX,
                dDW if ctx.need_ddw else None,
                dDelta, da_log,
                dThetaH if ctx.need_dthh else None,
                dTheta if ctx.need_dtheta else None,
                dThOff if ctx.need_dtheta_off else None,
                dZ,
                dDsk,
                dBeta if ctx.need_dbeta else None,
                dGamma if ctx.need_dgamma else None,
                dX0 if ctx.need_dx0 else None,
                dRotBias if ctx.need_drot_bias else None,
                dLam if ctx.need_dlam else None,
                None, None, None, None, None, None, None, None, None,
                None, None, None,
                dNormW, None,
                None,          # `pair_half`, a constexpr flag
                None,          # `dcs_ink`, a flag
                None,          # `chunk_bwd`, an int
                None,          # `theta_lo`, data only
                (None if getattr(ctx, "chart", None) is None
                 else _chart_req["F"]),   # chart_probe: the local tangent
                None)          # chart_geometry


# ── Differentiable state passing (no in-place ops) ────────────────────

# ── Admitted rate range (stability by construction) ────────────────────────
# The fused kernels form every decay exponent as a * (non-negative
# chunk-local difference) with a = -exp(A_log): the sign is exact by the
# arithmetic, so the only way a factor can leave [0, 1] is a rate that is
# not representable at all -- exp(A_log) overflowing fp32 (A_log > 88.7),
# where a = -inf and the diagonal (-inf) * 0 is NaN.  That is an input
# outside the admitted range, refused here rather than clamped inside the
# kernel.  The check is a device->host sync, so it runs under the validate
# flag (on in the collected suite, `SSD_MIMO_VALIDATE=1`); production
# callers hold the bound where A_log enters (construction / checkpoint
# load) instead of paying a sync per call.
A_LOG_MAX = 88.0
VALIDATE_ENV = "SSD_MIMO_VALIDATE"


def _validate_default() -> bool:
    return _os.environ.get(VALIDATE_ENV, "0").lower() not in ("0", "", "false", "no")


def check_a_log_range(a_log, who: str = "ssd_mimo_scan") -> None:
    """Refuse a rate whose exp() is not finite in fp32 (|A_log| > A_LOG_MAX).

    Raises ValueError naming the bound.  Costs one reduction and a host
    sync; see the note above for who calls it when.
    """
    if a_log is None:
        return
    m = a_log.detach().abs().max().item()
    if not (m <= A_LOG_MAX):
        raise ValueError(
            f"{who}: A_log has magnitude {m:.3g} > {A_LOG_MAX:g}; the rate "
            f"a = -exp(A_log) must be finite in fp32 (stability-by-construction "
            f"invariant).  A decay this "
            f"fast is total forgetting within one token; clamp or re-parameterise "
            f"A_log at the module, the kernel refuses it.")




# ── Main differentiable entry point ───────────────────────────────────

_DEPRECATION_WARNED = False

# ── N-tiling at the entry point ───────────────────────────────────────
# The fused kernels carry the state tile S[N, GT] in registers and stage
# [T, N] operands in shared memory, so a single launch caps at N = 128
# (power of two, tensor-core tile floor 16).  The RECURRENCE, however, is
# independent per state coordinate: the decay a is a scalar shared across
# N, so
#
#     h_t[n] = a_t·h_{t-1}[n] + B̃_t[n]·(δ_t x_t)     (independent in n)
#     y_t    = Σ_n C_t[·, n]·h_t[n]                   (a plain sum over n)
#
# splits EXACTLY over any partition of the N axis: run the kernel once per
# tile with the B/C column slices (and the x0 / beta / gamma / theta
# slices), sum the y's, concatenate the final states.  Each tile's
# backward is the existing one; dx / ddelta / da_log accumulate across
# tiles through ordinary autograd, dB/dC are scattered back by the slice.
#
# ── N BELOW THE TILE FLOOR ────────────────────────────────────────────
# N need not be a multiple of 16.  Multiples of 16 are served by the tiles
# above; everything else ends in a remainder below 16, and a tile narrower
# than 16 has no `tl.dot`.  That remainder is ZERO-PADDED up to 16 at the
# entry point — exact, because a zero B column writes no state, a zero C
# column reads none, and the padded coordinates' (zero) gradients are
# dropped by the slice on the way out.  Cost is (16-w)/16 wasted lanes in
# ONE tile, never in the power-of-two ones.
#
# Every tensor `ssd_mimo_scan` can be handed travels from here on as ONE
# dict (`ten`), so the tiling, the padding and the strategy trials all move
# the same set around instead of re-listing keyword arguments at
# four call sites.
_N_TILE_MAX = 128
_N_TILE_MIN = 16


def _n_tile_split(N: int, max_tile: int = _N_TILE_MAX) -> list[int]:
    """Split the state axis into tiles the fused kernel can launch.

    Greedy largest-first.  Every tile but possibly the last is a power of
    two in [16, max_tile]; a trailing remainder BELOW 16 is returned as
    itself and served by zero-padding it to 16 at the launch:

        1   -> [1]            12  -> [12]        (one padded 16-wide launch)
        16  -> [16]           128 -> [128]       (no tiling, no padding)
        24  -> [16, 8]        48  -> [32, 16]
        100 -> [64, 32, 4]    192 -> [128, 64]
        200 -> [128, 64, 8]   256 -> [128, 128]

    `max_tile` caps the tile width, which is how the N-splitting repair for
    a shared-memory overflow asks for narrower subsystems at the SAME
    chunk: `_n_tile_split(128, 64) == [64, 64]`.
    """
    if N < 1:
        raise ValueError(f"SSD kernel: N must be >= 1, got N={N}")
    cap = min(int(max_tile), _N_TILE_MAX)
    if cap < _N_TILE_MIN or (cap & (cap - 1)):
        raise ValueError(
            f"SSD kernel: the N-tile cap must be a power of two in "
            f"[{_N_TILE_MIN}, {_N_TILE_MAX}], got {max_tile}")
    tiles, rem = [], int(N)
    while rem >= _N_TILE_MIN:
        tiles.append(min(cap, 1 << (rem.bit_length() - 1)))
        rem -= tiles[-1]
    if rem:
        tiles.append(rem)
    return tiles


def _n_split_caps(widest: int) -> list[int]:
    """Tile caps STRICTLY narrower than `widest`, widest-first.

    The N-splitting repair walks these at the requested chunk: 128 -> [64,
    32, 16].  A state axis already at the 16-lane floor has none.
    """
    return [c for c in (64, 32, 16) if c < widest]


def _slice_or_none(t, dim, start, length):
    return None if t is None else t.narrow(dim, start, length)


def _pad_dim(t, dim: int, amount: int):
    """Zero-pad `t` on the RIGHT of `dim` by `amount` (differentiable)."""
    if t is None or not amount:
        return t
    pad = [0, 0] * (t.dim() - dim - 1) + [0, amount]
    return F.pad(t, tuple(pad))


def _tile_inputs(ten: dict, off: int, width: int, n_axis: int, N: int,
                 keep_gate: bool) -> tuple[dict, int]:
    """The per-N slice of every state-indexed tensor, padded to the floor.

    Returns `(tensors, pad)`; `pad` is 0 unless the tile is narrower than
    16, in which case the kernel runs a 16-wide state whose trailing
    `pad` coordinates are identically zero and the caller trims the final
    state back to `width`.

    The rotation tensors (theta / theta_h / rot_bias) are indexed by
    PAIR — coordinates (2k, 2k+1) — so their slice is the pair range of
    this tile and their padding is `pad // 2` zero angles (an identity
    rotation of the zero columns).  A tile of odd width has no pair
    decomposition, so a rotation on such an N is refused rather than
    silently mis-paired.
    """
    pad = _N_TILE_MIN - width if width < _N_TILE_MIN else 0
    sub = dict(ten)
    if not keep_gate:
        # silu(z) and the D-skip both act on the SUMMED readout, so a
        # multi-tile plan applies neither per tile.
        sub["gate_z"] = None
        sub["d_skip"] = None
    if off == 0 and width == N and not pad:
        return sub, 0                      # the untiled, unpadded launch
    rot = [ten[k] for k in ("theta", "theta_h", "rot_bias")]
    if (width % 2 or off % 2) and any(t is not None for t in rot):
        raise ValueError(
            f"SSD kernel: a rotation phase (theta / theta_h / rot_bias) "
            f"needs an EVEN state tile — it is indexed by the coordinate "
            f"pairs (2k, 2k+1) — but this call lands on tile "
            f"[{off}, {off + width}) of N={N}")
    h0, h1 = off // 2, (off + width) // 2
    # the bf16 twin is not sliced: a tiled launch takes the
    # Function's own downcast of its `theta` slice
    sub["theta_lo"] = None
    sub["B_tok"] = _pad_dim(ten["B_tok"].narrow(n_axis, off, width),
                            n_axis, pad)
    sub["C_tok"] = _pad_dim(ten["C_tok"].narrow(n_axis, off, width),
                            n_axis, pad)
    sub["beta"] = _pad_dim(_slice_or_none(ten["beta"], 1, off, width), 1, pad)
    sub["gamma"] = _pad_dim(_slice_or_none(ten["gamma"], 1, off, width),
                            1, pad)
    sub["x0"] = _pad_dim(_slice_or_none(ten["x0"], 3, off, width), 3, pad)
    # theta's PAIR axis is its last one: dim 2 head-shared [B, L, K],
    # dim 3 per head [B, L, H, K].
    if ten["theta"] is not None:
        th_ax = ten["theta"].dim() - 1
        sub["theta"] = _pad_dim(
            _slice_or_none(ten["theta"], th_ax, h0, h1 - h0),
            th_ax, pad // 2)
    sub["theta_h"] = _pad_dim(_slice_or_none(ten["theta_h"], 1, h0, h1 - h0),
                              1, pad // 2)
    sub["rot_bias"] = _pad_dim(
        _slice_or_none(ten["rot_bias"], 1, h0, h1 - h0), 1, pad // 2)
    return sub, pad


def _tile_launch(ten: dict, off: int, width: int, chunk: int, n_axis: int,
                 N: int, precision: str, bc_per_head: bool, keep_gate: bool,
                 rot_operand_dtype=None, rot_where=None,
                 bias_pre_rot=False, theta_increment=False,
                 y_token_major=False, out_norm_w=None, out_norm_eps=1e-5,
                 pair_half=False, chart_geometry=None):
    """One fused launch over the state columns [off, off + width)."""
    # The current FP32 Kbc backward's N=128/T>=128 tile needs two
    # staged N*T FP32 operands (131072 B at T=128). Its forward can fit
    # Blackwell's 101376 B while its backward cannot, even with the full
    # production autotune set. Reject this training plan before launching;
    # the existing N/T repair ladders keep the computation in Triton.
    if (width == 128 and chunk >= 128 and _needs_bwd(ten)
            and ten["x_in"].shape[3] == 1
            and (ten["C_tok"].dim() == n_axis + 1 or
                 (ten["C_tok"].dim() == n_axis + 2 and ten["C_tok"].shape[-1] == 1))
            and not (mma_io_enabled(precision) and ten["x_in"].dtype in
                     (torch.float16, torch.bfloat16))
            and smem_clamp_enabled()):
        required = 2 * width * chunk * 4
        # The operand estimate above misses the IEEE score-path scratch.
        # A100, N=T=128, G=32, R=O=1: the pinned config fails;
        # Triton 3.8 reports 180224 B.
        # Reject this measured unsupported training plan before forward,
        # so the existing N/T ladder repairs both halves of autograd.
        # Keep the guard narrow: no extrapolation to other precisions,
        # geometries or GPU generations without compiled evidence.
        if (chunk == 128 and ten["x_in"].shape[2] == 32
                and precision == "ieee"
                and _portable.nvidia_capability(ten["x_in"].device) == (8, 0)):
            required = max(required, 180224)
        limit = device_smem_limit(ten["x_in"].device)
        if required > limit:
            raise SsdSmemOverflow(required, limit, width, chunk)
    sub, pad = _tile_inputs(ten, off, width, n_axis, N, keep_gate)
    y, state = _fused_ssd_prepared(
        sub["x_in"], sub["delta"], sub["B_tok"], sub["C_tok"], sub["a_log"],
        Q=None, chunk_size=chunk,
        beta=sub["beta"], gamma=sub["gamma"], theta=sub["theta"],
        theta_h=sub["theta_h"], rot_bias=sub["rot_bias"],
        theta_lo=sub.get("theta_lo"),
        delta_w=sub["delta_w"], gate_z=sub["gate_z"],
        d_skip=sub["d_skip"], trap_lam=sub["trap_lam"], x0=sub["x0"],
        precision=precision, per_block_decay=False, stable_score=True,
        bc_per_head=bc_per_head, rot_operand_dtype=rot_operand_dtype,
        rot_where=rot_where, bias_pre_rot=bias_pre_rot,
        theta_increment=theta_increment,
        y_token_major=y_token_major,
        out_norm_w=out_norm_w, out_norm_eps=out_norm_eps,
        pair_half=pair_half,
        _require_fused=True, _smem_ladder=False,
        **({} if sub.get("chart_probe") is None else
           dict(chart_probe=sub["chart_probe"], chart_geometry=chart_geometry)))
    if pad:
        state = state[..., :width]
    return y, state


def _detached(ten: dict) -> dict:
    """Graph-free leaves sharing the caller's storage (no copy).

    `.detach()` is a view, so a strategy trial costs launches and workspace
    but not a second copy of x/B/C; `requires_grad_` mirrors the real call
    so the trial compiles the same STORE_SALL / recompute variant.
    """
    out = {}
    for k, v in ten.items():
        out[k] = None if v is None else v.detach().requires_grad_(
            bool(v.requires_grad))
    return out


def _needs_bwd(ten: dict) -> bool:
    return torch.is_grad_enabled() and any(
        v is not None and v.requires_grad for v in ten.values())


#: OPT-IN, NEVER A DEFAULT: precision="bf16_ops" = bf16 TENSOR-CORE OPERANDS under fp32 IO, i.e. the
#: arithmetic official Mamba-3 runs for ANY input dtype (`mamba3_siso_combined` casts Q/K/V/trap/angles/z to bf16).  It exists
#: ONLY for the like-for-like timing against official M3 at fp32 inputs and is labelled as that arithmetic in the tables.
#: Implemented as CAST-IN / CAST-OUT through the bf16 production path: the activation inputs the 16-bit path takes in the IO
#: dtype (x, B, C, the gate z) are cast to bf16, the call runs at precision "io" (bf16 operands, fp32 accumulators; decay
#: clocks, phases, parameters and the state carry stay fp32 by design, exactly as under bf16 autocast), and `y` returns in the
#: caller's dtype; autograd carries the gradients back through the casts.  Under 16-bit IO it is precision "io".  Selected by
#: no rule and no default: precision "io" under fp32 IO stays tf32 operands.
_BF16_OPERAND_MODE = "bf16_ops"     # spelled apart from the "bf16" alias of "io" (refused)

# The family's ONE public entry, and therefore the one place the
# `torch.compile` boundary belongs -- every Triton launch of the SSD-MIMO
# forward and backward is reached from inside here, including the
# flag-templated `_fused_chunk_fwd_kernel` whose kwargs Dynamo's
# `triton_kernel_wrap` mis-binds (see `src/orbit_ssm/compile_compat.py`).
# Pass-through in eager.
#
# With the registered-scan route on, the boundary sits lower, at the
# autograd node itself (`registered_scan.py`): the launchers are still
# never traced, but they are behind a registered custom op rather than a
# graph break, so the host prep between here and them is compiled with
# the rest of the module.  `_scan_boundary` is the identity when that
# route is on and `dynamo_opaque` when it is off -- it occupies
# `dynamo_opaque`'s slot, so the kernel-options scope stays INSIDE it.
@_scan_boundary
@_measured_plan.scan_scope
@_ko.scoped_kwarg("ssd_mimo")
# Paper letters -> public layer names: H = n_heads, P = d_head, N = d_state, R = d_ssm_in,
# O = d_ssm_out, G = n_groups.  (SSDMimoSSM exposes them as read-only properties.)
def ssd_mimo_scan(
    x_in: torch.Tensor,                   # [B, H, G_SIZE, R, L]
    delta: torch.Tensor,                   # [B, H, L]  decay step
    B_tok: torch.Tensor,                   # [B, L, (Gb,) N(, R)]
    C_tok: torch.Tensor,                   # [B, L, (Gc,) N(, O)]
    a_log: torch.Tensor,                   # [H] per-head scalar
    *,
    chunk_size: int = 64,
    # How `chunk_size` was chosen -- "explicit"
    # (a caller-pinned int), "env" (SSD_MIMO_CHUNK), "auto" (the per-rank
    # table) or "measure" (the timed cache).  Logging only, never read by
    # the dispatcher itself; recorded in `dispatch_log()`'s "chunk_mode".
    chunk_mode: str = "explicit",
    # The chunk the BACKWARD runs at.
    # `None` / 0 / == chunk_size means "the forward's", which is the
    # default; any
    # other divisor of the padded length is honoured only where every
    # chunked artefact is a pure token regrouping in memory (see
    # `_rechunk`).  `SSD_MIMO_CHUNK_BWD` is the env form.
    chunk_size_bwd: Optional[int] = None,
    # ── the config's routing channel ─────────────
    # A validated `{key: value}` mapping from `kernel_options`, active
    # for the duration of this call AND of the backward it builds (the
    # Function re-pushes it, because `backward_states="recompute"`
    # REPLAYS this forward and must schedule the replay identically).
    # Every key selects routing or scheduling only -- never a parameter
    # name, a shape or the function computed.  `None` / `{}` is the
    # shipped behaviour, bit for bit.  Supported keys and defaults:
    # `orbit_ssm.kernel_options.SUPPORTED_KERNEL_OPTIONS["ssd_mimo"]`.
    kernel_options: Optional[dict] = None,
    beta=None, gamma=None,
    theta=None, theta_h=None, rot_bias=None,
    # `theta`'s 16-bit twin -- the same
    # [B, L, (H,) N//2] phase already rounded to bf16 by the launch that
    # built it (`m3_epilogue.phase_law(..., lo_dtype=bf16)`).  Data only:
    # `theta` stays the fp32 autograd input; the Function saves the twin
    # for the backward and the two kernel wrappers read the bf16 pointer
    # directly.
    # Ignored wherever the saved phase must be
    # fp32 -- the SISO backward, `precision="ieee"`, an fp32 IO dtype --
    # or where the scan re-slices / re-cumsums `theta` (N-tiled repair,
    # the increment form).
    theta_lo=None,
    bias_pre_rot: bool = True,
    delta_w=None, gate_z=None, d_skip=None, trap_lam=None, x0=None,
    precision: str = "tf32",
    bc_per_head: bool = False,
    rot_operand_dtype: Optional[str] = None,
    rot_where: Optional[str] = None,
    theta_increment: bool = False,
    y_token_major: bool = False,
    # ── The enclosing MODULE's output gated RMSNorm ───────────────────
    # `out_norm_w` [H*O*G] (or None) makes the gate epilogue compute
    # `rmsnorm(y_pre * silu(z), w)` in ONE kernel each way, the way
    # Mamba-2's node calls `layernorm_gated`.  Single N-tile, O == 1 and
    # the dedicated gate epilogue only; every other config is refused by
    # name.  The caller must then NOT apply its own norm.
    out_norm_w=None,
    out_norm_eps: float = 1e-5,
    # ── PAIRHALF ─────────────────────────────────────────────────────
    # WHICH state coordinates form a rotation pair.
    #   "interleaved" (the default): pair k is (2k, 2k+1)
    #   "half"       : pair k is (k, k + N/2) -- Mamba-3's GPT-NeoX
    #                  "rotate-half" convention.
    # The two are a fixed PERMUTATION of the state axis, so they are the
    # same model in different coordinates: B/C rows, beta, gamma, theta,
    # theta_h and omega keep their meaning per PAIR, and a checkpoint
    # trained in one layout loads into the other through that permutation
    # (`SSDMimoSSM` installs the state-dict hook).
    pair_layout: str = "interleaved",
    # ── The state boundary, refused BY NAME ───────────────────────────
    # The scan takes exactly ONE state argument -- `x0`, the initial
    # hidden state -- and returns exactly one state adjoint, `dX0`.
    # Every OTHER SSD-MIMO carry reaches this scan through a host fold in
    # `SSDMimoSSM._prepare_kernel_inputs` and therefore has no argument
    # here and needs none.
    # hands one in is told WHICH carry is not a scan argument and where
    # it belongs, instead of having it silently ignored (a dropped carry
    # is a wrong continuation that looks like a converged one).
    write_prev=None,
    conv_state=None,
    # ── the LOCAL chart cotangent  ───
    # `chart_probe` [B, L, G, NF] is an optional caller's probe output; its "gradient" this scan's
    # backward returns is the packed folded-frame tangent F of every token
    # (a private contract with the fold, not the probe's true gradient,
    # which is 0).  `chart_geometry` = (transition group, k, Mf, Mp[,
    # emitter]); emitter "fused" (default: the CHART Gluon Kbc + Kch2 where
    # it serves, else by name the two-pass emitter) | "kch" (Kch1 + Kch2,
    # the reference).  The forward ignores both; None (the default) leaves
    # every route bit for bit as it was.
    chart_probe=None,
    chart_geometry=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SSD-MIMO family entry point — FUSED kernels only.

    Serves exactly the signed SSD-MIMO contract (a_scalar decay, R/O in
    {1,2,4,8}, all rotation modes/biases/gates, G a multiple of 16 after the
    padding below, N a multiple of 16).
    Raises a clear error for any config outside the fused envelope
    instead of silently dispatching to the legacy 3-dispatch path — use
    the DEPRECATED `unified_ssd` dispatcher for legacy consumers
    (per-G, Q/per-block-decay).

    `d_skip` [H, G, R] is the contract's fused D-skip: the forward adds
    `sum_r D[h, g, r] * x[b, h, g, r, t]` to the readout ON THE OUTPUT
    TILE, inside the scan kernel and before the gate.  It needs
    `gate_z` and O == 1; every other config is refused rather than
    silently dropped.

    `trap_lam` [B, H, L] turns `delta_w` into Mamba-3's RAW dt and runs
    the TWO-TAP TRAPEZOID in the kernel: the collapsed write scale
    `lam_t dt_t + dt_{t+1}(1-lam_{t+1})` and the rank-1 output correction
    `y_t -= dt_{t+1}(1-lam_{t+1}) (k_t.q_t) v_t`, both on tiles the chunk
    already holds.  R = O = 1, no theta_h / rot_bias (their extra
    read-side angle breaks the k.q cancellation).

    G (channels per head) is the one shape the entry point REPAIRS
    rather than refuses: the fused kernels dot on a 16-lane tensor-core
    tile, so any G that is not a multiple of 16 is zero-padded here to
    `ceil(G/16)*16` and the padding sliced off the outputs.  This is
    EXACT — a zero channel carries zero state and zero output, and its
    (zero) gradient is dropped by the slice — at the cost of (Gp-G)/Gp
    wasted lanes.  delta / B / C / a_log / theta / beta / gamma /
    rot_bias are per-(head, R/O), never per-channel, so they need no
    padding.  (`SSDMimoSSM._forward_cuda` pads on its own way in and so
    arrives with Gp == G; a carrier whose
    G is the head width P relies on this.)

    N: one kernel launch serves N a power of two in [16, 128].  EVERY other
    N >= 1 is served EXACTLY, by two composable host-side repairs:
      * a multiple-of-16 remainder is split into power-of-two tiles
        (`_n_tile_split`) and run one launch per tile — the decay is shared
        across N, so the tiles are independent subsystems whose outputs sum
        and whose final states concatenate;
      * a trailing remainder below 16 is zero-padded up to the 16-lane
        tensor-core floor and the final state trimmed back.
    N = 128 (and every other single-tile, unpadded N) takes the untiled
    path bit-for-bit unchanged — see `dispatch_log()`.

    Shared memory: when the requested (N, chunk_size) does not fit, the
    launcher chooses between halving the chunk at full N and splitting the
    state axis at the full chunk, by a measured, cached micro-benchmark.
    See `smem.py`.
    """
    # Validate at the public entry, like the module does: an unknown
    # string would fall through and run as tf32 -- including
    # the removed "bf16" spelling -- so a typo would silently change numerics.
    check_precision(precision, "ssd_mimo_scan", "ssd_mimo_scan")
    _ext = SCAN_PRECISIONS.get(precision)
    if _ext is not None and _ext.dispatch is not None:
        # a registered scan operand mode (`families.register_scan_precision`): its dispatch runs the scan itself or
        # checks its inputs and lets the shared kernels run (NotImplemented)
        _res = _ext.dispatch(x_in, delta, B_tok, C_tok, a_log, chunk_size=chunk_size, delta_w=delta_w,
                             bc_per_head=bc_per_head, theta=theta, theta_h=theta_h, rot_bias=rot_bias, beta=beta,
                             gamma=gamma, gate_z=gate_z, d_skip=d_skip, trap_lam=trap_lam, x0=x0,
                             out_norm_w=out_norm_w, theta_lo=theta_lo)
        if _res is not NotImplemented:
            return _res
    _out_dtype = None
    if precision == _BF16_OPERAND_MODE:
        # see `_BF16_OPERAND_MODE`: cast-in here, cast-out at the return
        _out_dtype = x_in.dtype
        _lo = lambda t: (t.to(torch.bfloat16) if (isinstance(t, torch.Tensor) and t.is_floating_point()
                                                  and t.dtype in (torch.float32, torch.float64)) else t)
        x_in, B_tok, C_tok, gate_z = _lo(x_in), _lo(B_tok), _lo(C_tok), _lo(gate_z)
        precision = "io"
    # bias_pre_rot=True with NO
    # rotation reaching the scan (theta, theta_h and rot_bias all None) is
    # a NO-OP -- the identity rotation makes "before"/"after" the bias the
    # same map.  Normalise the flag HERE, ONCE, before any kernel launch,
    # so BIAS_PRE_ROT is never handed to a launch as true without HAS_ROT
    # -- every fused kernel's own `tl.static_assert(not BIAS_PRE_ROT or
    # HAS_ROT, ...)` still holds, and callers that never pass
    # `bias_pre_rot` explicitly keep the kernel's default True without
    # tripping it.  `theta_h`-alone and `rot_bias` combos are real
    # refusals, not
    # this no-op, and are left to the checks inside `_fused_ssd_impl`.
    if write_prev is not None:
        raise NotImplementedError(
            "ssd_mimo_scan: 'write_prev' (the trapezoid's previous "
            "unscaled write) is NOT a scan argument and has no scan "
            "adjoint.  The collapsed one-tap recurrence runs on "
            "g_t = S_t + dt_{t+1}(1-lam_{t+1}) W_t, so the carry folds "
            "into x0 as x0 += dt_0 (1-lam_0) W_{-1} on the host "
            "(SSDMimoSSM._prepare_kernel_inputs); its gradient then "
            "arrives as part of dX0.  Fold it there, do not pass it "
            "here.")
    if conv_state is not None:
        raise NotImplementedError(
            "ssd_mimo_scan: 'conv_state' (the B/C convolution window) is "
            "NOT a scan argument and has no scan adjoint.  The causal "
            "convolution is applied to the token maps BEFORE the scan, "
            "so its history belongs to the convolution op module-side "
            "(SSDMimoSSM._bc_source(prefix=...)); the scan only ever "
            "sees the already-convolved B/C.")
    if bias_pre_rot and theta is None and theta_h is None and rot_bias is None:
        bias_pre_rot = False
    if bias_pre_rot and beta is None and gamma is None:
        # no additive bias on either map: "before" or "after" the rotation
        # is the same map, so the flag is a no-op here as well (callers
        # that fold their biases into B/C on the host, e.g. HybridMamba's
        # SSD route with a head-shared phase, pass no beta/gamma).
        bias_pre_rot = False
    if pair_layout not in ("interleaved", "half"):
        raise ValueError(
            "pair_layout must be 'interleaved' or 'half', got "
            f"{pair_layout!r}")
    # With NO rotation reaching the scan the layout is unobservable (no
    # pair is ever mixed), so the flag is normalised away here, once,
    # exactly as `bias_pre_rot` is above.
    pair_half = (pair_layout == "half"
                 and (theta is not None or theta_h is not None))
    if pair_half:
        # Envelope, refused HERE by name so no caller gets a layout
        # half-served deep inside a launcher.  R, O > 1 is admitted: the
        # general-rank half bodies of the forward,
        # Ka, Kb1, Kb2, Kc and the GEN_VEC trapezoid adjoint all exist,
        # so what is left is the N floor,
        # the retired rot_bias, and the schedule refusals inside
        # `fused_chunk_bwd_5kernel` / `_ssd_dispatch`.
        _N = B_tok.shape[3 if bc_per_head else 2]
        if _N % 32:
            raise NotImplementedError(
                "pair_layout='half' needs N a multiple of 32 (the MMA K "
                f"floor on the half tile); got N={_N}")
        if rot_bias is not None:
            raise NotImplementedError(
                "pair_layout='half' does not serve rot_bias")
    if _validate_default() and not torch.compiler.is_compiling():
        # The range check is a device->host SYNC (`.item()`),
        # which no traced region can hold -- with the entry traceable it
        # would be a graph break per call.  The eager path (every collected
        # test, every eager step of a compiled run's warm-up) still runs it;
        # a compiled step inherits the same parameters.
        check_a_log_range(a_log)
    # AUTOCAST IO: under an active autocast region
    # the scan's IO dtype IS the autocast dtype.  The ACTIVATION inputs are
    # cast to it here, once, so an upstream elementwise promotion (a 16-bit
    # map meeting an fp32 parameter, an explicit `.float()`) cannot put the
    # scan on its fp32-IO path (2x the x / y / dX traffic, and fp32 MMA
    # operands under "io").
    #   cast:   x_in, gate_z (every precision but the "ieee" oracle);
    #           B_tok, C_tok under precision "io" only -- their MMA
    #           operands are rounded to the IO dtype in-kernel anyway,
    #           while "tf32" asks for fp32 operands and keeps fp32 maps
    #           (e.g. a host rotation fold's, `rot_operand_dtype`).
    #   never:  delta, delta_w, a_log, beta, gamma, theta, theta_h,
    #           rot_bias, d_skip, trap_lam, x0, out_norm_w -- fp32 BY
    #           DESIGN (decay / write clocks, rates, phases, parameters,
    #           the state carry) -- and theta_lo (already the 16-bit twin).
    # Outside autocast, and for fp64, this is the identity.
    if precision != "ieee":
        x_in = autocast_io(x_in)
        gate_z = autocast_io(gate_z)
        if precision == "io":
            B_tok = autocast_io(B_tok)
            C_tok = autocast_io(C_tok)
    G = x_in.shape[2]
    g_pad = (-G) % 16
    if g_pad:
        heads = x_in.shape[1]
        x_in = F.pad(x_in, (0, 0, 0, 0, 0, g_pad))          # [B,H,Gp,R,L]
        if x0 is not None:
            x0 = F.pad(x0, (0, 0, 0, g_pad))                # [B,H,Gp,N]
        if gate_z is not None:                              # [B,L,H*G]
            bsz, length = gate_z.shape[0], gate_z.shape[1]
            gate_z = F.pad(gate_z.view(bsz, length, heads, G),
                           (0, g_pad)).reshape(bsz, length, heads * (G + g_pad))
        if d_skip is not None:                              # [H, G, R]
            # the padded channels carry zero x, so a zero weight there is
            # the identity — and the slice drops its (zero) gradient.
            d_skip = F.pad(d_skip, (0, 0, 0, g_pad))
    ten = dict(x_in=x_in, delta=delta, B_tok=B_tok, C_tok=C_tok, a_log=a_log,
               beta=beta, gamma=gamma, theta=theta, theta_h=theta_h,
               rot_bias=rot_bias, delta_w=delta_w, gate_z=gate_z,
               d_skip=d_skip, trap_lam=trap_lam, x0=x0, theta_lo=theta_lo)
    if chart_probe is not None:
        from .chart_emit import chart_refusal
        if chart_geometry is None or len(chart_geometry) not in (4, 5):
            raise ValueError("ssd_mimo_scan: chart_probe needs chart_geometry = (group, k, Mf, Mp[, emitter])")
        if len(chart_geometry) == 5 and chart_geometry[4] not in ("fused", "kch"):
            raise ValueError(f"ssd_mimo_scan: chart emitter {chart_geometry[4]!r} (one of 'fused', 'kch')")
        _why = chart_refusal(mode=chart_geometry[0], k=int(chart_geometry[1]), Mf=int(chart_geometry[2]),
                             Mp=int(chart_geometry[3]), N=B_tok.shape[3 if bc_per_head else 2], T=chunk_size,
                             P=x_in.shape[2], R=x_in.shape[3], O=(C_tok.shape[-1] if C_tok.dim() ==
                                                                  (5 if bc_per_head else 4) else 1),
                             theta=theta, theta_h=theta_h, rot_bias=rot_bias, beta=beta, gamma=gamma,
                             pair_half=pair_half)
        if _why is not None:
            raise NotImplementedError(f"ssd_mimo_scan: the local chart cotangent is not served here ({_why})")
        ten["chart_probe"] = chart_probe
    if y_token_major and g_pad:
        # the G-pad trim below slices the CHANNEL axis, which a
        # token-major `y` cannot give back as a view.  `_forward_cuda`
        # pads G to the kernel tile itself, so this never fires from the
        # module; refuse rather than return the other layout silently.
        raise NotImplementedError(
            "y_token_major with G not a multiple of 16: the pad trim "
            "would copy; pad G module-side (SSDMimoSSM.G_kernel does)")
    with _chunk_bwd_request(chunk_size_bwd):
        (y, s_final), record = _ssd_dispatch(ten, chunk_size, precision,
                                             bc_per_head, rot_operand_dtype,
                                             rot_where, bias_pre_rot,
                                             theta_increment,
                                             y_token_major=y_token_major,
                                             pair_half=pair_half,
                                             out_norm_w=out_norm_w,
                                             out_norm_eps=out_norm_eps,
                                             chunk_mode=chunk_mode,
                                             chart_geometry=chart_geometry)
    record["measured_plan"] = _measured_plan.current()
    record["g_pad"] = g_pad
    _note_dispatch(record)
    if g_pad:
        # y is [B, H, G, L] (O == 1) or [B, H, O, G, L]; the channel axis
        # is -2 either way, and S_final is [B, H, G, N].
        y = y[..., :G, :]
        s_final = s_final[..., :G, :]
    if _out_dtype is not None and y.dtype != _out_dtype:
        y = y.to(_out_dtype)
    return y, s_final


# ── the dispatch trace ───────────────────────────────────────────────
# A speed guard needs to see WHICH plan a call took, and timing cannot
# answer that on a shared GPU.  Every `ssd_mimo_scan` records its decision
# here: the usual power-of-two N must show `tiles == [N]`, `n_pad == 0`,
# `chunk_used == chunk_requested`, `strategy == "native"` — i.e. exactly
# the untiled, unpadded launch the kernel had before any of this.  One
# dict append per call, which is nothing next to a kernel launch.
_DISPATCH_LOG: list = []
#: route telemetry of the last `_SSDMimoFn.forward` (see there); {} before any call
LAST_ROUTE: dict = {}
_DISPATCH_LOG_MAX = 32


def dispatch_log() -> list:
    """The last 32 `ssd_mimo_scan` dispatch records, newest last.

    Keys: N, tiles, n_pad, g_pad, chunk_requested, chunk_used, launches,
    strategy ("native" when the requested shape fitted as asked),
    chunk_mode (how `chunk_requested` was chosen -- "explicit", "env",
    "auto" or "measure"; see `SSDMimoSSM._resolve_chunk_size`),
    chunk_guard ("" normally; `"k24_rot_io_bwd_t64"` when the
    sm_90 rotation/IO-MMA chunk cap forced `chunk_used` below
    `chunk_requested` under a ptxas < 12.9.86 -- see `rot_io_bwd_t64_fault` in
    fused_bwd_kernel.py and `_ssd_dispatch`; `"sm90_pairhalf_r1_t64"`
    for the rotate-half R = O = 1 cap under a
    ptxas < 12.9.86, `pairhalf_r1_t64_fault`; `"sm90_pairhalf_io_t64"`
    for the rotate-half 16-bit-MMA cap at any R, O
    under the same ptxas, `pairhalf_io_t64_fault`; a CORRECTNESS clamp, unlike
    `strategy`'s shared-memory CAPACITY repair), and
    fwd_warps / fwd_stages / fwd_maxnreg / fwd_cfg -- the fused
    forward's chosen launch config and which rung chose it -- and
    fwd_spill_fix, `(stages_before, spills_before,
    spills_after)` when the spill ladder raised the pipeline depth,
    `None` otherwise -- and fwd_stack_fix,
    `(spills_stacked, spills_unstacked)` when the (o, r)-STACK ladder
    dropped the stacked body because ptxas spilled it, `None` otherwise.
    fwd_stack_guard: "" normally,
    `"sm90_io_ro_stack"` when the Hopper 16-bit-operand refusal
    (`fused_fwd_kernel.ro_stack_wgmma_fault`: sm_90, bf16/fp16 operands,
    ptxas older than 12.9.86) moved a cell that would have taken the
    stacked (o, r) body onto the per-(o, r) one.

    kernel_options / kernel_options_src
    -- the routing options this call was actually steered by, and where
    each came from ("config" or the deprecated "env" channel).  ONLY the
    keys whose in-force value differs from the family default appear:
    `{}` means "every option at its family default".  The defaults are
    contract text and are recoverable from the family, so recording them
    per call would add nothing except the invitation to believe the
    caller chose them -- and a dispatch record, like a benchmark number,
    must mean exactly what it says (Part B).
    """
    return list(_DISPATCH_LOG)


def _note_dispatch(record: dict) -> None:
    # The diagnostic ring buffers are HOST-side and
    # are skipped while TRACING -- the treatment `check_a_log_range`'s
    # device->host sync already takes.  `del LOG[:-MAX]` reads the
    # list's LENGTH, so a traced append installs a length guard that the next
    # call falsifies: one recompile per step until the buffer saturates,
    # against a `cache_size_limit` of 8.  With the LM path in ONE graph that
    # limit retires the whole module to eager.
    if torch.compiler.is_compiling():
        return
    _DISPATCH_LOG.append(record)
    del _DISPATCH_LOG[:-_DISPATCH_LOG_MAX]


def _plan_record(N, tiles, chunk_req, chunk_used, strategy,
                 chunk_mode="explicit", guard="") -> dict:
    """The dispatch record, built AFTER the launch so it can report it.

    The forward's LAUNCH CONFIG is part of the plan, so it
    is reported here -- `fwd_warps`, `fwd_stages`, `fwd_maxnreg` (0 = the
    compiler's own choice) and `fwd_cfg` ("heuristic" / "measured" /
    "env", the rung `_fwd_launch_cfg_full` took).  Timing cannot tell you
    which config a call took on a shared GPU, so the dispatcher says.
    Absent (all None) on the fallback impls, which run no Triton forward.

    `fwd_stack_fix` is `(spills_stacked,
    spills_unstacked)` when `fused_fwd_kernel._fwd_fix_stack` dropped the
    stacked (o, r) body because ptxas reported it spilling, and `None`
    otherwise -- same reporting rule, same reason.

    `fwd_spill_fix` is
    `(stages_before, spills_before, spills_after)` when the SPILL ladder
    (`fused_fwd_kernel._fwd_fix_spills`) raised the pipeline depth
    because ptxas reported the body spilling into local memory, and
    `None` otherwise -- the same reporting rule as `fwd_cfg`'s: a repair
    that timing cannot see has to be said out loud.

    `guard`: "" normally, or the name of a CORRECTNESS clamp that forced
    `chunk_used` below `chunk_req` (see `chunk_guard` in `dispatch_log()`
    -- `"k24_rot_io_bwd_t64"`, the sm_90 rotation/IO-MMA chunk cap, or
    `"sm90_pairhalf_r1_t64"`, the sm_90 rotate-half R = O = 1 cap, both in
    `_ssd_dispatch`).  Distinct from `strategy`, which names
    a shared-memory CAPACITY repair.
    """
    last = tiles[-1]
    try:
        from .fused_fwd_kernel import last_fwd_cfg
        cfg = last_fwd_cfg()
    except Exception:                    # noqa: BLE001 — no Triton at all
        cfg = {}
    ko_val, ko_src = _ko.record("ssd_mimo")
    return dict(N=N, tiles=list(tiles),
                n_pad=(_N_TILE_MIN - last) if last < _N_TILE_MIN else 0,
                chunk_requested=chunk_req, chunk_used=chunk_used,
                launches=len(tiles), strategy=strategy,
                chunk_mode=chunk_mode, chunk_guard=guard,
                fwd_warps=cfg.get("warps"), fwd_stages=cfg.get("stages"),
                fwd_maxnreg=cfg.get("maxnreg"),
                fwd_cfg=cfg.get("source"),
                fwd_spill_fix=cfg.get("spill_fix"),
                fwd_stack_fix=cfg.get("stack_fix"),
                fwd_stack_guard=cfg.get("stack_guard"),
                kernel_options=ko_val, kernel_options_src=ko_src)


def _run_plan(ten, tiles, chunk, n_axis, N, precision, bc_per_head,
              rot_operand_dtype=None, rot_where=None, bias_pre_rot=False,
              theta_increment=False, y_token_major=False,
              out_norm_w=None, out_norm_eps=1e-5, pair_half=False,
              chart_geometry=None):
    """Run one plan = (tiles, chunk).  Single-tile keeps the fused gate.

    `y_token_major` reaches only the SINGLE-tile plan: the N-tiled repair
    sums per-tile outputs, and that sum is what would have to carry the
    layout.  Nothing breaks if it does not -- the module detects the
    layout from the strides -- so the tiled route stays as it was.
    """
    if len(tiles) == 1:
        return _tile_launch(ten, 0, tiles[0], chunk, n_axis, N, precision,
                            bc_per_head, keep_gate=True,
                            rot_operand_dtype=rot_operand_dtype,
                            rot_where=rot_where, bias_pre_rot=bias_pre_rot,
                            theta_increment=theta_increment,
                            y_token_major=y_token_major,
                            out_norm_w=out_norm_w,
                            out_norm_eps=out_norm_eps,
                            pair_half=pair_half,
                            chart_geometry=chart_geometry)
    if ten.get("chart_probe") is not None:
        raise NotImplementedError(
            "the local chart cotangent (chart_probe) needs the single-tile "
            f"plan; this call resolved to {len(tiles)} N-tiles")
    if out_norm_w is not None:
        # The fused output norm normalises the FULL readout row, so
        # it cannot run per N-tile and then be summed.  The N-tiled
        # repair is a shared-memory fallback the module's own cells never
        # take; refuse by name rather than return a differently-normed y.
        raise NotImplementedError(
            "the fused output RMSNorm needs the single-tile plan; this "
            f"call resolved to {len(tiles)} N-tiles (a shared-memory "
            "repair).  Run with SSD_MIMO_OUT_NORM_FUSED=0.")
    if pair_half:
        # PAIRHALF: an N-TILE splits the state axis into contiguous
        # blocks, which under this layout cuts pairs in half.  Refuse
        # rather than compute a different model silently.
        raise NotImplementedError(
            "pair_layout='half' does not take the N-tiled shared-memory "
            "repair (an N tile would split rotation pairs)")
    return _ssd_mimo_scan_n_tiled(ten, tiles, chunk, n_axis, N, precision,
                                  bc_per_head,
                                  rot_operand_dtype=rot_operand_dtype,
                                  rot_where=rot_where,
                                  bias_pre_rot=bias_pre_rot,
                                  theta_increment=theta_increment)


def _ssd_dispatch(ten, chunk_size, precision, bc_per_head,
                  rot_operand_dtype=None, rot_where=None,
                  bias_pre_rot=False, theta_increment=False,
                  y_token_major=False, out_norm_w=None, out_norm_eps=1e-5,
                  pair_half=False, chunk_mode="explicit", chart_geometry=None):
    """Run the scan, repairing a shared-memory overflow by the faster route.

    Returns `((y, state), record)`.  The happy path is one `try` around the
    requested plan: no probing, no measurement, no extra launch.  Only when
    the fused forward's dry compile reports an overflow does the strategy
    machinery wake up — see the module docstring of `smem.py`.
    """
    n_axis = 3 if bc_per_head else 2
    N = ten["B_tok"].shape[n_axis]
    tiles = _n_tile_split(N)

    # ── ROT-IO sm_90 chunk cap ──────────────────────────────────────────
    # `_bwd_k2_kernel` (fused_bwd_kernel.py) takes an illegal shared-
    # memory read on sm_90 at chunk T = 64 with HAS_THETA_H (rot_mode
    # "m3") under IO-dtype (bf16) MMA operands at R = O = 1, under a
    # ptxas < 12.9.86 (the same wgmma miscompile as the two sm_90 guards
    # below; Triton 3.8's bundled ptxas runs T = 64 here).  See
    # `rot_io_bwd_t64_fault`'s docstring and the note above
    # `_bwd_k2_kernel` for the bisection (every
    # num_warps/num_stages/maxnreg faults alike, so this is not an
    # autotune-prune situation).  Caught HERE, before the shared-memory
    # halving ladder below even builds its candidate list, so that
    # ladder never offers T = 64 on this cell.  `chunk_req` keeps the
    # ORIGINAL ask for the dispatch record -- same convention as the
    # smem ladder's `chunk_requested` != `chunk_used`.
    chunk_req = chunk_size
    guard = ""
    if chunk_size > 32 and ten["x_in"].is_cuda:
        if bc_per_head:
            c_per_o = ten["C_tok"].dim() == 5
            O_ax = ten["C_tok"].shape[-1] if c_per_o else 1
        else:
            _, _, O_ax = _mimo_axes(ten["B_tok"], ten["C_tok"])
        R_ax = ten["x_in"].shape[3]
        _cc = _portable.nvidia_capability(ten["x_in"].device)
        if rot_io_bwd_t64_fault(
                _cc, mma_io_enabled(precision), ten["theta_h"] is not None,
                R_ax, O_ax):
            guard = "k24_rot_io_bwd_t64"
            chunk_size = 32
        elif pairhalf_r1_t64_fault(_cc, precision != "ieee", pair_half,
                                   R_ax, O_ax):
            # The rotate-half R = O = 1 bodies take
            # an illegal access at T = 64 on sm_90 under a ptxas < 12.9.86
            # (see `pairhalf_r1_t64_fault`); T = 32 is exact there.
            guard = "sm90_pairhalf_r1_t64"
            chunk_size = 32
        elif pairhalf_io_t64_fault(_cc, mma_io_enabled(precision), pair_half):
            # The rotate-half bodies with 16-bit MMA operands at
            # ANY (R, O) return a wrong y at T = 64 on sm_90 under a ptxas
            # < 12.9.86 (see `pairhalf_io_t64_fault`); T = 32 is exact.
            guard = "sm90_pairhalf_io_t64"
            chunk_size = 32

    def run(plan, tensors):
        return _run_plan(tensors, plan[0], plan[1], n_axis, N, precision,
                         bc_per_head, rot_operand_dtype=rot_operand_dtype,
                         rot_where=rot_where, bias_pre_rot=bias_pre_rot,
                         theta_increment=theta_increment,
                         y_token_major=y_token_major,
                         out_norm_w=out_norm_w, out_norm_eps=out_norm_eps,
                         pair_half=pair_half, chart_geometry=chart_geometry)

    try:
        out = run((tiles, chunk_size), ten)
        return out, _plan_record(N, tiles, chunk_req, chunk_size, "native",
                                 chunk_mode=chunk_mode, guard=guard)
    except SsdSmemOverflow as exc:
        overflow = exc

    # Two families of exact repairs, each an ordered ladder of plans.
    families = {
        T_HALVE: [(tiles, T) for T in chunk_ladder(chunk_size)[1:]],
        N_SPLIT: [(_n_tile_split(N, cap), chunk_size)
                  for cap in _n_split_caps(tiles[0])],
    }
    available = [name for name, plans in families.items() if plans]
    if not available:
        raise overflow

    key = strategy_key(N, chunk_size, ten["x_in"].dtype, precision,
                       "fwd_bwd" if _needs_bwd(ten) else "fwd",
                       device=ten["x_in"].device)
    winner = preferred_strategy(key, available)
    if winner is None and len(available) > 1 and measuring_possible():
        winner = _measure_strategy(key, families, available, run, ten)
    if winner is None:
        winner = T_HALVE if T_HALVE in available else available[0]

    for name in [winner] + [n for n in available if n != winner]:
        for plan in families[name]:
            try:
                out = run(plan, ten)
            except SsdSmemOverflow:
                continue
            plan_tiles, plan_chunk = plan
            if name == T_HALVE:
                note_chunk_clamped(N, chunk_size, plan_chunk,
                                   overflow.required, overflow.limit)
            else:
                note_n_split(N, chunk_size, plan_tiles,
                             overflow.required, overflow.limit)
            return out, _plan_record(N, plan_tiles, chunk_req, plan_chunk,
                                     name, chunk_mode=chunk_mode, guard=guard)
    raise overflow


def _measure_strategy(key, families, available, run, ten):
    """Time both repairs on detached views of the caller's own tensors.

    The trial callable a family contributes is its FIRST plan that fits —
    discovered by running the ladder here, which also pays the Triton
    compile so the timed iterations do not.  Anything that goes wrong —
    an OOM on the extra workspace, a driver refusal, a nested-autograd
    complaint — disqualifies that candidate rather than failing the user's
    call, and if both are disqualified the caller falls back to T-halving.
    """
    det = _detached(ten)
    train = _needs_bwd(ten)

    def trial(plan):
        def _call():
            if not train:
                with torch.no_grad():
                    run(plan, det)
                return
            for v in det.values():
                if v is not None:
                    v.grad = None
            y, state = run(plan, det)
            (y.float().sum() + state.float().sum()).backward()
        return _call

    trials = {}
    for name in available:
        for plan in families[name]:
            try:
                trials[name] = trial(plan)
                trials[name]()          # discovery + warm-up in one
                break
            except SsdSmemOverflow:
                trials.pop(name, None)
                continue
            except Exception:           # noqa: BLE001 — disqualify, not fail
                trials.pop(name, None)
                torch.cuda.empty_cache()   # an OOM trial must not linger
                break
    if len(trials) > 1:
        return resolve_strategy(key, trials)
    return next(iter(trials)) if trials else None


def _ssd_mimo_scan_n_tiled(ten, tiles, chunk_size, n_axis, N, precision,
                           bc_per_head, rot_operand_dtype=None,
                           rot_where=None, bias_pre_rot=False,
                           theta_increment=False
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact N > 128 (or off-grid N) via host-side state tiling.

    One existing fused launch per N-tile; `y` accumulates in fp32 across
    tiles and the final states concatenate.  Every per-N tensor is sliced,
    never materialised twice, so peak memory is that of ONE tile plus the
    accumulator — O(tiles) launches, not O(tiles) copies of the inputs.
    A trailing tile narrower than 16 is zero-padded to the tile floor by
    `_tile_inputs` and its state trimmed back before the concatenation.

    The SwiGLU gate is deliberately NOT handed to the per-tile kernels:
    silu(z) multiplies the SUMMED readout, so it is applied once, after
    the accumulation, through the same `silu_gate` op the unfused path
    uses (differentiable, fp32 math).  The D-skip is the same story --
    `y_pre = sum_tiles(scan) + D.x` has ONE skip term, not one per tile --
    so it too is added once, here, in plain autograd-visible torch.
    """
    from .gate_epilogue import silu_gate

    x_in, gate_z = ten["x_in"], ten["gate_z"]
    d_skip = ten.get("d_skip")
    y_acc = None
    states = []
    off = 0
    for n in tiles:
        y_t, s_t = _tile_launch(ten, off, n, chunk_size, n_axis, N,
                                precision, bc_per_head, keep_gate=False,
                                rot_operand_dtype=rot_operand_dtype,
                                rot_where=rot_where,
                                bias_pre_rot=bias_pre_rot,
                                theta_increment=theta_increment)
        yf = y_t.float()
        y_acc = yf if y_acc is None else y_acc + yf
        states.append(s_t)
        off += n
    state = torch.cat(states, dim=-1)          # [B, H, G, sum(tiles)]
    if d_skip is not None:
        # x_in [B, H, G, R, L] * D [H, G, R] summed over r
        y_acc = y_acc + torch.einsum(
            "bhgrl,hgr->bhgl", x_in.float(), d_skip.float())
    y = y_acc.to(x_in.dtype)
    if gate_z is not None:
        H, G_SIZE = x_in.shape[1], x_in.shape[2]
        if gate_z.shape[-1] != H * G_SIZE:
            raise ValueError(
                f"gate_z last dim must be H*G_SIZE={H * G_SIZE}, got "
                f"{gate_z.shape[-1]}")
        Bsz, L = gate_z.shape[0], gate_z.shape[1]
        z = gate_z.view(Bsz, L, H, G_SIZE).permute(0, 2, 3, 1)  # [B,H,G,L]
        if y.dim() == 5:                       # [B, H, O, G, L]
            z = z.unsqueeze(2).expand(-1, -1, y.shape[2], -1, -1)
        y = silu_gate(y.contiguous(), z.contiguous().to(y.dtype))
    return y, state


def _fused_ssd_prepared(
    x_in: torch.Tensor,                  # [B, H, G_SIZE, R, L]
    delta: torch.Tensor,                  # [B, H, L]
    B_tok: torch.Tensor,                  # [B, L, N]
    C_tok: torch.Tensor,                  # [B, L, N]
    a_log: torch.Tensor,                  # [M], [H], or [H, M] (mixed heads)
    Q: Optional[torch.Tensor],            # [B, L, M, k, k], or None (no rotations)
    chunk_size: int = 64,
    beta: Optional[torch.Tensor] = None,  # [H, N, R]
    omega: Optional[torch.Tensor] = None, # [N//2]  (not yet supported)
    x0: Optional[torch.Tensor] = None,    # [B, H, G_SIZE, N]
    precision: str = "tf32",
    per_block_decay: Optional[bool] = None,
    r_heads: Optional[torch.Tensor] = None,  # [H] int, per-head rank <= R
    stable_score: Optional[bool] = None,     # None -> False (classic kernels)
    bias_pre_rot: bool = False,              # beta/gamma INSIDE the phase
                                             # (Mamba-3's fold order)
    bc_per_head: bool = False,               # B/C carry a GROUP axis at dim 2
    gate_z: Optional[torch.Tensor] = None,   # [B, L, H*G_SIZE] SwiGLU gate
    d_skip: Optional[torch.Tensor] = None,   # [H, G_SIZE, R] D-skip weights
    trap_lam: Optional[torch.Tensor] = None, # [B, H, L] Mamba-3 trapezoid
                                             # lambda; makes delta_w RAW dt
    delta_w: Optional[torch.Tensor] = None,  # [B, H, L] WRITE scale; `delta`
                                             # then sets only the DECAY (A_dyn)
    theta: Optional[torch.Tensor] = None,    # [B, L, N//2] head-shared phase
                                             # or [B, L, H, N//2] per head
                                             # (complex mode; rotates in-kernel)
    theta_h: Optional[torch.Tensor] = None,  # [H, N//2] per-head rotation offset
                                             # (additive to theta; requires theta)
    gamma: Optional[torch.Tensor] = None,    # [H, N, O] per-head READ bias —
                                             # the rank-1 mirror of beta;
                                             # A-scalar kernels only
    b_per_g: bool = False,                   # B_tok carries a PER-G axis:
                                             # [B, L, Gb, G, N] (with
                                             # bc_per_head=True, R == 1) —
                                             # per-(head, g) write maps
    rot_bias: Optional[torch.Tensor] = None,  # [H, N//2] per-head write
                                             # rotation bias (additive with
                                             # theta inside kernel; fused only)
    b_bias: Optional[torch.Tensor] = None,   # [H, G, N] static ADDITIVE
                                             # per-(head, g) write bias:
                                             # score row term
                                             # u_g(t) = <C~_t, beta_g> on
                                             # the hoisted score + constant
                                             # drive in the aggregates;
                                             # A-scalar, R == 1; composes
                                             # with per-G decay/B~
    rot_operand_dtype: Optional[str] = None,  # "fp32" (default: the host
                                             # theta fold promotes B~/C~
                                             # to fp32) or "io" (cast back
                                             # to the IO dtype).  None ->
                                             # the `rot_operand_dtype` kernel option.
    rot_where: Optional[str] = None,         # "kernel" (default: the fused
                                             # kernels rotate B/C on load
                                             # and the backward
                                             # accumulates dTheta
                                             # in-kernel) or "host" (the
                                             # torch fold materialises
                                             # B~/C~).  None ->
                                             # SSD_MIMO_ROT_WHERE.  Same
                                             # function either way; a
                                             # config outside the R = O = 1
                                             # ungrouped fused envelope
                                             # takes "host" regardless.
    theta_increment: bool = False,           # `theta` holds per-token
                                             # angle INCREMENTS and the
                                             # kernels cumsum them
                                             # intra-chunk; a route
                                             # that cannot (host fold)
                                             # materialises the running
                                             # phase here instead
    y_token_major: bool = False,             # allocate the output
                                             # TOKEN-major ([B, L, H*O*G]
                                             # memory).  Shape is
                                             # unchanged -- only the
                                             # strides -- so the module's
                                             # output norm never gathers.
                                             # Fused route only.
    out_norm_w=None,                         # the enclosing
    out_norm_eps: float = 1e-5,              # module's output gated
                                             # RMSNorm, fused into the
                                             # gate epilogue kernel
    pair_half: bool = False,                 # PAIRHALF: rotate-half layout
    theta_lo=None,                           # theta's bf16 twin
                                             # (see ssd_mimo_scan)
    chart_probe=None, chart_geometry=None,   # the local chart
                                             # cotangent (ssd_mimo_scan)
    _require_fused: bool = False,            # internal (ssd_mimo_scan)
    _smem_ladder: bool = True,               # internal: walk the chunk
                                             # ladder here, or report the
                                             # overflow to the caller
                                             # (ssd_mimo_scan, which owns
                                             # the T-vs-N strategy choice)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable unified rotation + MH-MIMO.

    Same interface as unified_ssd_forward but supports backward.
    Returns y: [B, H, G_SIZE, L], state: [B, H, G_SIZE, N].

    Mixed heads: a_log [H, M] gives each head its own per-block decay row
    (a head of a scalar-decay family repeats its scalar), r_heads gives each head its own rank
    (streams r >= r_heads[h] are skipped; keep their x and beta at zero,
    and mask the beta gradient of rank-1 heads in the training loop).

    bf16-native: x_in / B_tok / C_tok may be bf16 — the kernels load them
    as-is (narrow loads) and, by default, upcast operands to fp32 and run
    tf32 MMA (bf16-MMA-layout dot results trip a Triton shared-memory
    codegen bug at the store/atomic epilogue) and y / dX / dB / dC come back in the
    input dtype.  All
    decay math, the state, and the small parameters run fp32 regardless;
    autocast is disabled internally since dtypes are managed explicitly.
    """
    check_cuda_input("ssd_mimo_scan (CUDA route)",
                     {"x": x_in, "delta": delta, "B": B_tok, "C": C_tok})
    if Q is not None or r_heads is not None or b_per_g or b_bias is not None or delta.dim() != 3:
        raise ValueError("Q/per-block/per-G/r_heads routes are not served; use the supported ssd_mimo_scan axes")
    _require_fused = True
    if omega is not None:
        raise NotImplementedError(
            "HAS_OMEGA removed — use HAS_ROT (theta parameter)")
    gk = dict(gate_z=gate_z, d_skip=d_skip, trap_lam=trap_lam,
              delta_w=delta_w,
              theta=theta, theta_h=theta_h, gamma=gamma, b_per_g=b_per_g,
              b_bias=b_bias, rot_bias=rot_bias,
              rot_operand_dtype=rot_operand_dtype,
              rot_where=rot_where, bias_pre_rot=bias_pre_rot,
              theta_increment=theta_increment,
              y_token_major=y_token_major,
              out_norm_w=out_norm_w, out_norm_eps=out_norm_eps,
              pair_half=pair_half, theta_lo=theta_lo,
              _require_fused=_require_fused)
    if chart_probe is not None:
        gk.update(chart_probe=chart_probe, chart_geometry=chart_geometry)
    if (gate_z is not None or d_skip is not None) and Q is not None:
        raise NotImplementedError(
            "the fused gate / D-skip epilogue is wired on the A-scalar "
            "kernels (Q=None, per-head-scalar decay) only")
    # ── Per-G features: delta may be [B, H, G, L] (per-(h,g)
    # decay), B_tok may carry a per-G axis (b_per_g), and b_bias is the
    # static per-(h,g) additive write bias.
    # per_block_decay=False), R == 1, no beta/gamma/omega/theta/epilogues.
    per_g_any = (b_per_g or b_bias is not None
                 or delta.dim() == 4)
    if per_g_any and Q is not None:
        raise NotImplementedError(
            "per-G decay / per-G B~ / b_bias require Q=None (A-scalar)")
    if b_per_g and not bc_per_head:
        raise ValueError("b_per_g=True requires bc_per_head=True "
                         "(B_tok [B, L, Gb, G, N])")
    def _dispatch(T):
        if bc_per_head:
            H = x_in.shape[1]
            Gb = B_tok.shape[2]
            Gc = C_tok.shape[2]
            if H % Gb != 0 or H % Gc != 0:
                raise ValueError(
                    f"bc_per_head: group counts Gb={Gb}, Gc={Gc} must each "
                    f"divide H={H}")
            if Gb == 1 and Gc == 1 and not b_per_g:
                # zero-cost compat: identical to never having passed the axis
                return _fused_ssd_impl(
                    x_in, delta, B_tok[:, :, 0], C_tok[:, :, 0], a_log, Q,
                    T, beta, omega, x0, precision, per_block_decay,
                    r_heads, stable_score, **gk)
            return _fused_ssd_impl(
                x_in, delta, B_tok, C_tok, a_log, Q, T, beta, omega,
                x0, precision, per_block_decay, r_heads, stable_score,
                bc_groups=(Gb, Gc), **gk)
        return _fused_ssd_impl(
            x_in, delta, B_tok, C_tok, a_log, Q, T, beta, omega,
            x0, precision, per_block_decay, r_heads, stable_score, **gk)

    # ── Shared-memory-aware chunk selection ──────────────────────────────
    # The fused forward's per-CTA shared memory grows with T x N; at
    # N = 128, T = 128 it overruns a GH200's 232,448 B.
    # `_check_fwd_smem` dry-compiles the exact variant and raises
    # `SsdSmemOverflow` BEFORE the launch, so
    # the chunk can be walked down its halving ladder (T, T/2, ... 32) to
    # the largest width that fits — a change of chunkwise associativity
    # only, announced once via a RuntimeWarning.  See smem.py.
    #
    # `_smem_ladder=False` (what `ssd_mimo_scan` passes) hands the overflow
    # straight back instead: the entry point owns the choice between this
    # ladder and the N-splitting repair, and cannot choose if the callee
    # has already silently halved the chunk.
    ladder = chunk_ladder(chunk_size) if _smem_ladder else [chunk_size]
    overflow = None
    with torch.amp.autocast("cuda", enabled=False):
        for i, T in enumerate(ladder):
            try:
                out = _dispatch(T)
            except SsdSmemOverflow as exc:
                overflow = overflow or exc
                if i + 1 == len(ladder):
                    raise
                continue
            if overflow is not None:
                note_chunk_clamped(overflow.N, chunk_size, T,
                                   overflow.required, overflow.limit)
            return out




def _fused_ssd_impl(x_in, delta, B_tok, C_tok, a_log, Q, chunk_size,
                      beta, omega, x0, precision, per_block_decay, r_heads,
                      stable_score=None, bc_groups=None, gate_z=None,
                      d_skip=None, trap_lam=None,
                      delta_w=None, theta=None, theta_h=None,
                      gamma=None, b_per_g=False, b_bias=None,
                      rot_bias=None, _require_fused=False,
                      rot_operand_dtype=None, rot_where=None,
                      bias_pre_rot=False, theta_increment=False,
                      y_token_major=False,
                      out_norm_w=None, out_norm_eps=1e-5,
                      pair_half=False, theta_lo=None,
                      chart_probe=None, chart_geometry=None):
    if omega is not None:
        raise NotImplementedError(
            "HAS_OMEGA removed — use HAS_ROT (theta parameter)"
        )
    # The combined mixer node drives this call's Function by hand.
    # The entry tensors are recorded HERE, before the squeeze / fold /
    # pad below rebind any of the names.
    _cap = _capture_slot()
    _cap_in = None if _cap is None else dict(
        x_in=x_in, delta=delta, B_tok=B_tok, C_tok=C_tok, a_log=a_log,
        beta=beta, gamma=gamma, theta=theta, theta_h=theta_h,
        rot_bias=rot_bias, delta_w=delta_w, gate_z=gate_z, d_skip=d_skip,
        trap_lam=trap_lam, x0=x0, out_norm_w=out_norm_w)
    B, H, G_SIZE, R, L = x_in.shape
    # ── Per-G features ─────────────────────────────────────────────────
    per_g_delta = delta.dim() == 4
    per_g = (per_g_delta or b_per_g or b_bias is not None)
    if per_g:
        if per_g_delta and tuple(delta.shape) != (B, H, G_SIZE, L):
            raise ValueError(
                f"per-G delta must be [B, H, G, L]={(B, H, G_SIZE, L)}, "
                f"got {tuple(delta.shape)}")
        if R != 1:
            raise NotImplementedError("per-G features require R == 1")
        if per_block_decay:
            raise NotImplementedError(
                "per-G features live on the A-scalar route "
                "(per_block_decay=False, a_log [H])")
        for nm, t in (("beta", beta), ("gamma", gamma), ("omega", omega),
                      ("theta", theta), ("gate_z", gate_z),
                      ("d_skip", d_skip), ("r_heads", r_heads)):
            if t is not None:
                raise NotImplementedError(
                    f"per-G features do not compose with {nm} yet")
        if b_per_g and B_tok.dim() != 5:
            raise ValueError(
                f"b_per_g: B_tok must be [B, L, Gb, G, N], got "
                f"{tuple(B_tok.shape)}")
        if b_per_g and B_tok.shape[3] != G_SIZE:
            raise ValueError(
                f"b_per_g: B_tok G axis {B_tok.shape[3]} != G_SIZE "
                f"{G_SIZE}")
        if b_bias is not None and tuple(b_bias.shape) != \
                (H, G_SIZE, B_tok.shape[-1]):
            raise ValueError(
                f"b_bias must be [H, G, N], got {tuple(b_bias.shape)}")
    rot_op_io = _rot_operand_io(rot_operand_dtype)
    if theta is not None:
        # R(theta)^T on the N axis, pairs adjacent -- the exact fold
        # C_t^T R(phi_t - phi_s) B_s = (R(phi_t)^T C_t)^T (R(phi_s)^T B_s).
        # theta [B, L, N//2] is HEAD-SHARED, so no head axis is created.
        # The KERNEL rotates its own C/B tiles on load, so B_tok/C_tok stay
        # UNROTATED for it; this builds only the operand the chunk
        # AGGREGATES need, since those run host-side and would otherwise
        # leave the cross-chunk carried state in the unrotated frame (the
        # 8.7-at-2-chunks bug).  Both paths land on the same B_tok, and
        # theta collects gradient from both -- the fold's (autograd) and
        # the kernel's (dTheta).
        # The N axis is 3 when B carries the grouped gauge axis
        # ([B,L,Gb,N(,R)]) and 2 otherwise -- guessing it from the tensor
        # rank is ambiguous, since [B,L,Gc,N] and [B,L,N,R] are both 4-dim.
        # LAUNCH COUNT.  The generic spelling below is
        # ~9 elementwise launches per fold, run twice (B and C), and its
        # backward is a comparable chain again, so the HOST cannot keep the
        # single fused forward kernel fed.  When the pair axis is LAST — R =
        # O = 1, grouped or not, which is the whole parity grid and
        # the LM carrier — the entire fold is ONE complex multiply:
        #     z = a + i b,  w = e^{-i theta} = cos - i sin
        #     z w = (a cos + b sin) + i (b cos - a sin)
        # whose real/imag parts are EXACTLY the two rows of R(theta)^T that
        # the generic form spells out.  `rot` is built once and shared by
        # B and C (the generic form computes cos/sin twice).
        # SSD_MIMO_ROT_FOLD=generic restores the elementwise chain — the
        # A/B lever to compare the two spellings.
        # torch.polar is fp32/fp64 only, so a narrower phase dtype keeps
        # the elementwise form (which promotes it the same way).
        _rot_ok = (theta.dtype in (torch.float32, torch.float64)
                   and _env("ROT_FOLD", "complex") == "complex")
        _rot_c = []          # built on FIRST use, then shared by B and C

        def _fold_complex(v, nax):
            K2 = v.shape[nax] // 2
            vp = v.float().reshape(*v.shape[:nax], K2, 2)
            if not vp.is_contiguous():
                return None
            if not _rot_c:
                _rot_c.append(torch.polar(torch.ones_like(theta), -theta))
            w = _rot_c[0].reshape(theta.shape[0], theta.shape[1],
                                  *((1,) * (nax - 2)), K2)
            out = torch.view_as_real(
                torch.view_as_complex(vp) * w).reshape(v.shape)
            return out.to(v.dtype) if rot_op_io else out

        def _fold(v, nax):
            K2 = v.shape[nax] // 2
            lead = v.shape[:nax]
            tail = v.shape[nax + 1:]
            if not tail and _rot_ok:
                fast = _fold_complex(v, nax)
                if fast is not None:
                    return fast
            vp = v.reshape(*lead, K2, 2, *tail)
            th = theta.reshape(theta.shape[0], theta.shape[1],
                               *((1,) * (nax - 2)), K2, *((1,) * len(tail)))
            c, sn = torch.cos(th), torch.sin(th)
            a_ = vp[(slice(None),) * nax + (slice(None), 0)]
            b_ = vp[(slice(None),) * nax + (slice(None), 1)]
            out = torch.stack([c * a_ + sn * b_, -sn * a_ + c * b_], nax + 1)
            out = out.reshape(v.shape)
            # ROTATION OPERAND WIDTH.  `theta` is fp32 by
            # contract, so torch promotes the product of a bf16 B/C with
            # cos/sin to fp32: on the rotation path ONLY, every B/C tile
            # load in the fwd kernel and in Ka/Kbc-hb/Kc is twice as wide
            # as the IO dtype asks for (the no-rotation path loads them at
            # the IO dtype).  "io" casts back, which costs ONE extra
            # rounding of the rotated operand and returns dB/dC in the IO
            # dtype; "fp32" (DEFAULT) keeps the promotion.
            if rot_op_io:
                out = out.to(v.dtype)
            return out
    # bc_groups=(Gb, Gc): B_tok [B,L,Gb,N(,R)], C_tok [B,L,Gc,N(,O)] with the
    # fused grouped gauge axis.  Q must be None here (folded consumers), the
    # dispatch guarantees it.
    bc_grouped = bc_groups is not None
    Gb, Gc = bc_groups if bc_grouped else (1, 1)
    if bc_grouped:
        b_per_r = (not b_per_g) and B_tok.dim() == 5
        c_per_o = C_tok.dim() == 5
        O = C_tok.shape[-1] if c_per_o else 1
        N = B_tok.shape[4] if b_per_g else B_tok.shape[3]
        hpg_b = H // Gb
        hpg_c = H // Gc
        fold_nax = 3
    else:
        b_per_r, c_per_o, O = _mimo_axes(B_tok, C_tok)
        N = B_tok.shape[2] if b_per_r else B_tok.shape[-1]
        fold_nax = 2
    T = chunk_size
    # "fwd:bwd" split precision (see _ChunkKernelFn.forward): the forward-
    # side Triton aggregates take the fwd half; the full (possibly colon)
    # string flows to the chunk Function, which splits it itself.
    # (`dot_precision` narrows the third operand mode back to "tf32"
    # for the omega/per-block-decay aggregates, which it does not cover.)
    prec_fwd_only = dot_precision(precision.partition(":")[0])

    a_hm = a_log.dim() == 2
    if a_hm:
        per_block_decay = True
    if Q is not None:
        M, k = Q.shape[-3], Q.shape[-1]
    else:
        assert a_hm or per_block_decay is not None, \
            "Q=None requires per_block_decay set explicitly (or a_log [H, M])"
        if a_hm or per_block_decay:
            M = a_log.shape[-1]
            k = N // M
        else:
            M, k = N, 1
    if per_block_decay is None:
        per_block_decay = (a_log.dim() == 1 and a_log.shape[0] == M)

    # Per-head-SCALAR decay (a_log [H], per_block_decay=False): composes with
    # dynamic B/C and the grouped-BC axis on the A-scalar kernels (D-F1c).
    a_scalar = (not per_block_decay) and (not a_hm)

    # ── host fold or in-kernel phase ─────────────────────────────────
    # Decided HERE, where every fact the fused dispatch below uses is
    # known, and applied here because the fold has to run before the L
    # padding.  The in-kernel form needs the R = O = 1 ungrouped fused
    # envelope (the branch Ka / Kbc-hb / Kc carry) and a head-shared
    # [B, L, N//2] phase; anything else takes the host fold.
    if bias_pre_rot and rot_bias is not None:
        raise NotImplementedError(
            "ssd kernel: bias_pre_rot combined with rot_bias (the write-side angle offset) is refused -- rot_bias has no "
            "bias_pre_rot fold order of its own "
            "(theta={}, rot_bias={})".format(
                theta is not None, rot_bias is not None))
    if bias_pre_rot and theta is None and theta_h is not None:
        raise NotImplementedError(
            "ssd kernel: bias_pre_rot with theta_h and no theta is a real "
            "(non-identity) READ-side rotation with no defined "
            "bias_pre_rot fold order -- pass theta too, or drop theta_h, "
            "or use bias_pre_rot=False")
    # bias_pre_rot=True with NO rotation
    # reaching the scan (theta is None, theta_h is None, rot_bias is
    # None) is a NO-OP, not a refusal -- with theta=None the rotation is
    # the identity, so "before" or "after" it names the same map.  Falls
    # straight through to the plain biased scan below, same as
    # bias_pre_rot=False with no rotation kwargs.

    rot_kernel = False
    if theta is not None and not bc_grouped:
        # A trailing R/O axis of LENGTH ONE is a view axis on a
        # head-shared tensor.  Squeezing it (a differentiable view, so the
        # gradient is reshaped straight back) is what lets an R = O = 1
        # caller that spells B as [B, L, N, 1] -- SSDMimoSSM does, always
        # -- reach the in-kernel phase, whose envelope is written for the
        # 3-dim spelling.  Nothing else moves: the kernels' r/o loops are
        # already single-trip and s_br / s_co are multiplied by zero.
        # Gated on `theta is not None`, so no rotation-free call changes.
        if b_per_r and B_tok.shape[-1] == 1:
            B_tok = B_tok.squeeze(-1)
            b_per_r = False
        if c_per_o and C_tok.shape[-1] == 1:
            C_tok = C_tok.squeeze(-1)
            c_per_o = False
            O = 1

    if theta is not None:
        # [B, L, K] head-shared, or [B, L, H, K] PER HEAD.
        th_shape_ok = (
            (theta.dim() == 3 and theta.shape[-1] == N // 2)
            or (theta.dim() == 4 and theta.shape[-1] == N // 2
                and theta.shape[2] == H))
        # ── The envelope is R, O in {1, 2, 4, 8} ──────────────────────
        # The phase acts on the N axis of the tile a CTA's OWN kernel head
        # loads, and every consumer (the fused forward, Ka, Kb1, Kb2, Kc,
        # Kbc-hb) owns one head per CTA, so a per-(kernel head) phase is
        # the same load at a shifted base whatever R and O are.  For
        # R > 1 / O > 1 the GENERIC backward branch un-rotates the two
        # gradient accumulators (Kb1 and whichever kernel owns the
        # aggregate dB product) and contracts dTheta --
        # the grouped B/C axis rides
        # it unchanged: the group index is a function of the head, so a
        # per-head un-rotation before the (grouped) atomic is exact.
        rot_kernel = (
            _rot_where(rot_where) == "kernel"
            and a_scalar and not per_g
            and b_bias is None and Q is None
            and r_heads is None
            and R in RANKS and O in RANKS
            and th_shape_ok
            and not (N & (N - 1)) and 16 <= N <= 128
            and G_SIZE % 16 == 0
            # the two remaining `_use_fused` clauses, mirrored: the
            # in-kernel form IS the fused pair, so a config that leaves
            # it must take the host fold rather than trip the
            # "rot_kernel resolved outside the fused path" assertion.
            and (d_skip is None or (O == 1 and gate_z is not None)))
        if theta.dim() == 4 and not rot_kernel:
            # A per-head phase has no host form (see the _ROT_WHERE note):
            # the fold acts on head-shared B/C.  Say which clause failed
            # rather than silently dropping to a head-shared rotation.
            raise NotImplementedError(
                "ssd kernel: a PER-HEAD phase theta [B, L, H, N//2] is an "
                "in-kernel-only feature and this call is outside its "
                f"envelope (rot_where={_rot_where(rot_where)!r}, "
                f"a_scalar={a_scalar}, per_g={per_g}, "
                f"grouped_bc={bc_grouped}, R={R}, O={O}, N={N}, "
                f"G={G_SIZE}, theta{tuple(theta.shape)}, H={H}).  Pass a "
                "head-shared [B, L, N//2] phase for those routes.")
        if bias_pre_rot and not (rot_kernel and theta.dim() == 4):
            # Mamba-3's fold ORDER (beta/gamma INSIDE the rotation) is an
            # in-kernel, PER-HEAD-phase feature: with the bias inside, the
            # rotated tile is per head whatever the phase is, so the
            # backward's head-block hoist is invalid, and the host fold --
            # which rotates head-shared B/C -- cannot express it at all.
            raise NotImplementedError(
                "ssd kernel: bias_pre_rot needs the in-kernel PER-HEAD "
                f"phase (rot_kernel={rot_kernel}, "
                f"theta.dim()={theta.dim()})")
        if theta_increment and rot_kernel and L <= T:
            # ── ONE chunk -- the reparameterisation buys nothing ───────
            # Every per-chunk offset is zero there, so all the increment
            # form leaves is cost: a build launch the kernels do not need
            # and, inside them, four `tl.cumsum`es and an offset load
            # they would not compile at all on the running-phase path.
            # Integrating here is the SAME two lines the host-fold route
            # below already takes, on a tensor of at most `T` tokens.
            theta = theta.cumsum(dim=1)
            theta_increment = False
            theta_lo = None
        if theta_increment and not rot_kernel:
            # The host fold needs the RUNNING phase, so a route that
            # does not reach the in-kernel cumsum materialises it here.
            # Correctness, never speed: the fast route below never takes
            # this branch.
            theta = theta.cumsum(dim=1)
            theta_increment = False
            theta_lo = None
        if not rot_kernel:
            if pair_half:
                # `_fold` rotates the ADJACENT-pair layout.
                # Refuse rather than fold the wrong pairs on the host.
                raise NotImplementedError(
                    "pair_layout='half' is an in-kernel-rotation feature; "
                    "the host fold (`rot_where='host'`, or a geometry "
                    "outside the in-kernel envelope) rotates adjacent "
                    "pairs only")
            B_tok = _fold(B_tok, fold_nax)
            C_tok = _fold(C_tok, fold_nax)
    if b_per_r or c_per_o:
        # fail fast, before the preprocessing pipeline runs.  otile=True: this
        # differentiable path o-tiles the readout so O>2 is supported.  The
        # guard sees the REAL per_block_decay (a hardcoded True would let
        # scalar + dynamic-B silently launch the classic kernel with
        # misread strides).
        _check_mimo_support(
            b_per_r, c_per_o, O, R,
            _resolve_stable(True if stable_score is None else stable_score,
                            per_block_decay, r_heads is not None, T, R,
                            G_SIZE, a_scalar=a_scalar, c_per_o=c_per_o),
            Q, r_heads, otile=True, a_scalar=a_scalar)
    if a_scalar and (b_per_r or c_per_o or bc_grouped or per_g) \
            and a_log.shape[0] != H:
        raise ValueError(
            f"per-head-scalar a_log must have shape [H]={H}, got "
            f"{tuple(a_log.shape)}")

    has_beta = beta is not None
    nc = (L + T - 1) // T
    L_padded = nc * T
    device = x_in.device
    f32 = torch.float32

    # ── Pad (differentiable via F.pad) ────────────────────────────────
    # bf16-native path: x / B / C keep their dtype (narrow loads; the
    # A-scalar kernels upcast operands to fp32 / tf32 MMA — see the entry
    # docstring's bf16 note).
    x_pad = x_in
    delta_pad = delta.float()
    B_tok_pad = B_tok
    C_tok_pad = C_tok
    Q_pad = Q.float() if Q is not None else None

    if L_padded > L:
        pad = L_padded - L
        x_pad = F.pad(x_pad, (0, pad))
        delta_pad = F.pad(delta_pad, (0, pad))
        if bc_grouped:
            # B/C carry a group axis at dim 2; pad the L dim (dim 1)
            bpad = ((0, 0, 0, 0, 0, 0, 0, pad) if (b_per_r or b_per_g)
                    else (0, 0, 0, 0, 0, pad))
            cpad = ((0, 0, 0, 0, 0, 0, 0, pad) if c_per_o
                    else (0, 0, 0, 0, 0, pad))
        else:
            bpad = (0, 0, 0, 0, 0, pad) if b_per_r else (0, 0, 0, pad)
            cpad = (0, 0, 0, 0, 0, pad) if c_per_o else (0, 0, 0, pad)
        B_tok_pad = F.pad(B_tok_pad, bpad)
        C_tok_pad = F.pad(C_tok_pad, cpad)
        if Q_pad is not None:
            Q_eye = torch.eye(k, device=device, dtype=f32)
            Q_eye = Q_eye.expand(B, pad, M, k, k)
            Q_pad = torch.cat([Q_pad, Q_eye], dim=1)

    # ── Chunk + cumulative delta ──────────────────────────────────────
    if per_g_delta:
        delta_c = delta_pad.view(B, H, G_SIZE, nc, T)
    else:
        delta_c = delta_pad.view(B, H, nc, T)
    # With the in-kernel prefix the SAME slot carries
    # the RAW chunked delta -- a VIEW of `delta_pad`, so neither the ATen
    # scan kernel nor its [B, H, nc, T] fp32 buffer is ever built.  See
    # `_dcs_ink_on` / `_dcs_tile`.
    dcs_ink = _dcs_ink_on(R, O)
    # SPANDELTA: the Function takes the RAW chunked delta on every route and
    # forms the prefix itself (in-kernel, or host-side outside autograd)
    delta_cumsum = delta_c
    # The grouped B/C chunk view's MEMORY ORDER.
    # `rotating` here is the PER-TOKEN PHASE: the arms that lose to
    # group-major are the `m3` ones, and the kernels that lose are
    # `_bwd_kb1_kernel` and `_trap_adjoint_kernel` on the in-kernel-phase
    # path.  A static read offset alone (`theta_h`) is a much lighter body
    # and does not regress, so it is not refused on.
    group_major = _bc_group_major_on(
        bc_grouped,
        # `rotating` is the PER-TOKEN PHASE and nothing else.
        # `pair_half` is NOT in the predicate: it implies `theta` or
        # `theta_h`, and forcing group-major on improves every non-`m3`
        # cell whatever its pair layout.
        rotating=(theta is not None),
        wants_chunk_bwd=_chunk_bwd_asked() != 0)
    if bc_grouped:
        # [B, L, Gx, N(,R/O)] -> [B, nc, Gx, T, N(,R/O)] (group before T so the
        # per-tile T/N strides match the head-shared layout inside the kernel)
        if b_per_g:
            # per-G axis rides between the group axis and T:
            # [B, L, Gb, G, N] -> [B, nc, Gb, G, T, N]
            B_c = B_tok_pad.view(B, nc, T, Gb, G_SIZE, N) \
                .permute(0, 1, 3, 4, 2, 5).contiguous()
        else:
            B_c = _bc_chunk_grouped(B_tok_pad, B, nc, T, Gb, N,
                                    R if b_per_r else 0, group_major)
        C_c = _bc_chunk_grouped(C_tok_pad, B, nc, T, Gc, N,
                                O if c_per_o else 0, group_major)
    else:
        # ungrouped MIMO: same R/O-major trick (one small copy, coalesced
        # tile loads in exchange); R = O = 1 stays a zero-copy view
        B_c = ((B_tok_pad.view(B, nc, T, N, R)
                .permute(0, 4, 1, 2, 3).contiguous()
                .permute(0, 2, 3, 4, 1)) if b_per_r
               else B_tok_pad.view(B, nc, T, N))
        C_c = ((C_tok_pad.view(B, nc, T, N, O)
                .permute(0, 4, 1, 2, 3).contiguous()
                .permute(0, 2, 3, 4, 1)) if c_per_o
               else C_tok_pad.view(B, nc, T, N))

    # DECOUPLED WRITE / A_dyn.  `delta` sets the DECAY exponent
    # (through delta_cumsum); `delta_w`, when given, sets the WRITE scale
    # instead.  Both are separate host-side tensors by the time the kernel
    # sees them -- delta_cumsum and x_scaled -- so decoupling them costs no
    # Triton at all, and it is what makes M3-style dynamic A expressible:
    # pass delta = m_t * dt (token-dependent decay increment, a_log keeps
    # its meaning as the static rate) and delta_w = dt (write unchanged).
    # Folding a dynamic rate into `delta` alone would silently rescale the
    # drive.
    dw_pad = delta_pad if delta_w is None else F.pad(
        delta_w.float(), (0, L_padded - L))

    # ── Fused 2-launch dispatch decision ─────────────────────────────
    # The fused kernels scale x by dw ON LOAD (raw x in, dx_raw + dDW
    # out), so the fused path never materialises x_scaled — deciding here
    # skips that ~200MB host-side round trip entirely.
    # r_heads is excluded: the fused kernels have no HAS_RH and the mask
    # below only reaches the fallback aggregates.
    _use_fused = (a_scalar
                  and not per_g
                  and b_bias is None
                  # the D-skip is computed at the fused scan
                  # kernel's store (O == 1, and it rides the gate's
                  # y_pre tile), so it does not force the legacy
                  # 3-dispatch route.  O > 1 still does.
                  and (d_skip is None
                       or (O == 1 and gate_z is not None))
                  and Q_pad is None
                  and r_heads is None)

    # ── Shape envelope ─────────────────────────────────────────────────
    # Fail LOUDLY for shapes NO path supports (non-pow2 N would crash deep
    # inside Triton's arange; G neither a multiple of 16 nor a small power
    # of two would silently drop channels).  Small power-of-two G (< 16,
    # e.g. G=8) is valid but below the tensor-core tile floor of the fused
    # kernels — those configs run the 3-dispatch fallback.
    if _use_fused:
        if (N & (N - 1)) or not (16 <= N <= 128):
            raise ValueError(
                f"SSD kernel: N must be a power of two in [16, 128], "
                f"got N={N}")
        if G_SIZE % 16:
            if G_SIZE & (G_SIZE - 1):
                raise ValueError(
                    f"SSD kernel: G must be a multiple of 16 or a small "
                    f"power of two, got G={G_SIZE}")
            _use_fused = False  # pow2 G < 16 -> fallback tiles handle it

    if rot_kernel and not _use_fused:
        # cannot happen: rot_kernel replicates every _use_fused clause.
        # Stated as a refusal anyway, because the failure mode of being
        # wrong here is a SILENTLY DROPPED rotation (the 3-dispatch route
        # sets theta_c = None and B/C were never folded).
        raise AssertionError(
            "ssd kernel: rot_where='kernel' resolved outside the fused "
            "path — the phase would be dropped; this is a bug in the "
            "rot_kernel predicate, not a user error")

    if d_skip is not None:
        # Refuse BY NAME, here, rather than letting the generic envelope
        # error below report `d_skip=True` among six other flags: at O > 1
        # and without a gate the D-skip is not merely off the fused route,
        # it has no implementation anywhere, and the message should say
        # WHICH clause failed.
        if O != 1:
            raise NotImplementedError(
                "the fused D-skip needs O == 1: its [H, G, R] weights "
                f"carry no output axis, and this call has O={O}")
        if gate_z is None:
            raise NotImplementedError(
                "the fused D-skip rides the gate epilogue (its adjoint "
                "reads the same y_pre tile); pass gate_z, or apply the "
                "skip module-side")

    if y_token_major and not _use_fused:
        # The token-major store lives in the fused forward's (or the gate
        # epilogue's) allocation; the 3-dispatch fallback has no such
        # store.  Refuse by name rather than silently returning the other
        # layout -- the caller would then pay the transpose it asked to
        # remove, invisibly.
        raise NotImplementedError(
            "y_token_major is a fused-route option; this config resolved "
            "to the 3-dispatch fallback")

    if _require_fused and not _use_fused:
        raise ValueError(
            "ssd_mimo_scan: config is outside the fused SSD-MIMO "
            f"envelope (a_scalar={a_scalar}, per_g={per_g}, "
            f"b_bias={b_bias is not None}, d_skip={d_skip is not None}, "
            f"Q={Q_pad is not None}, r_heads={r_heads is not None}, "
            f"G={G_SIZE}, N={N}). "
            "There is no generic fallback dispatcher.")

    Btil, Ctil = B_c, C_c

    # ── Fused 2-launch path ─────────────────────────────────────────────
    # Dispatch decision computed above.
    # The only route (there is no generic fallback).
    if _use_fused:
        thh_f = theta_h.float().contiguous() if theta_h is not None else None
        # On the host-fold route theta is consumed by the FOLD above
        # (R(theta)^T applied to B/C via differentiable torch ops) — the
        # fold's autograd provides dTheta and the fused kernel sees
        # already-rotated B~/C~.  Passing theta here too would
        # double-rotate (and the kernel wrapper expects a
        # [B, nc, T, K] tensor, not the [B, L, K] host layout).  The
        # in-kernel HAS_ROT path takes `th_f` below.
        th_f = None
        th_lo = None
        off_f = None
        if rot_kernel:
            # [B, L, (H,) N//2] -> [B, nc, T, (H,) N//2], the layout the
            # kernels index (the head axis, when present, stays INSIDE the
            # token axis: the [T, K] tile a CTA loads is then K contiguous
            # floats per row, exactly as in the head-shared form, so no
            # host permutation is needed -- only the head STRIDE changes).
            # F.pad and reshape are differentiable and free of arithmetic,
            # so `theta` collects its gradient straight from the
            # backward's in-kernel dTheta.  The padded tail rides zeros:
            # B/C are zero there, so it contributes nothing.
            th_f = theta
            th_ph = theta.dim() == 4
            if L_padded > L:
                th_f = F.pad(th_f, ((0, 0, 0, 0, 0, L_padded - L) if th_ph
                                    else (0, 0, 0, L_padded - L)))
            th_f = (th_f.reshape(B, nc, T, H, N // 2) if th_ph
                    else th_f.reshape(B, nc, T, N // 2))
            # The bf16 twin takes the same pad / view (data
            # only, no autograd node); a twin that does not match the
            # phase it claims to round is dropped, not trusted.
            if (theta_lo is not None and theta_lo.shape == theta.shape
                    and theta_lo.dtype == torch.bfloat16
                    and theta_lo.device == theta.device):
                th_lo = theta_lo
                if L_padded > L:
                    th_lo = F.pad(th_lo, ((0, 0, 0, 0, 0, L_padded - L)
                                          if th_ph
                                          else (0, 0, 0, L_padded - L)))
                th_lo = th_lo.reshape(th_f.shape)
            if theta_increment:
                # The offset build is ONE Triton launch inside `_SSDMimoFn`
                # (`phase_offset_build`, no autograd node).
                pass
        # gate_z rides the SAME L padding as x/B/C (F.pad is an autograd
        # op, so the padded dZ is trimmed on the way out); the epilogue
        # views it as [B, nc, T, H, G].
        gz_c = None
        if gate_z is not None:
            if gate_z.shape[-1] != H * G_SIZE:
                raise ValueError(
                    f"gate_z last dim must be H*G_SIZE={H * G_SIZE}, got "
                    f"{gate_z.shape[-1]}")
            gz_c = (F.pad(gate_z, (0, 0, 0, L_padded - L))
                    if L_padded > L else gate_z)
            # The DEDICATED gate epilogue reads z (and
            # writes dz) through a full stride triple, so a strided view
            # -- the mixer's slice of the fused in_proj slab -- is read
            # where it lies, saving a clone per step.  The IN-KERNEL gate
            # (SSD_MIMO_GATE_INKERNEL=1) and the torch chain
            # (SSD_MIMO_GATE_FUSED=0) still want it contiguous, so they
            # keep the copy.
            if not _gate_epilogue_reads_strided():
                gz_c = gz_c.contiguous()
        beta_f = beta.float().contiguous() if has_beta else None
        gamma_f = gamma.float().contiguous() if gamma is not None else None
        x0_f = x0.float().contiguous() if x0 is not None else None
        rb_f = rot_bias.float().contiguous() if rot_bias is not None else None
        # ── The D-skip is a fused-path feature ────────────────────────
        # [H, G, R] fp32, the layout the kernel's `Dsk_ptr + h*G*R +
        # g*R + r` addressing wants.  `.float()` is autograd-visible, so
        # the module's own parameter dtype is restored on the way back.
        dsk_f = d_skip.float().contiguous() if d_skip is not None else None
        # trap_lam rides x/B/C's L padding: dt is zero in the pad, so the
        # two-tap term there is zero whatever lambda is.
        tl_f = None
        if trap_lam is not None:
            tl_f = trap_lam.float()
            if L_padded > L:
                tl_f = F.pad(tl_f, (0, L_padded - L))
            tl_f = tl_f.contiguous()
        # inference (no grad anywhere): the FWD kernel skips the
        # per-chunk state save entirely
        needs_bwd = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad
            for t in (x_pad, dw_pad, Btil, Ctil, a_log, theta_h,
                      beta, gamma, x0, rot_bias, gate_z, th_f, off_f,
                      d_skip, trap_lam))
        # B/C are already materialised (possibly in R/O-major memory under
        # a logical view — .contiguous() would COPY BACK to interleaved);
        # every consumer is stride-driven.
        _fn_args = (
            Ctil, Btil,
            _dense_or_contiguous(x_pad), dw_pad,
            delta_cumsum.contiguous(),
            a_log.float().contiguous(), thh_f, th_f, off_f,
            gz_c, dsk_f, beta_f, gamma_f, x0_f, rb_f, tl_f,
            H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped,
            needs_bwd, bias_pre_rot,
            bool(theta_increment and rot_kernel and off_f is None),
            y_token_major, out_norm_w, out_norm_eps, bool(pair_half),
            bool(dcs_ink),
            # The backward's own chunk (0 = the
            # forward's).  See `_rechunk`'s comment block.
            _chunk_bwd_asked(),
            # theta's bf16 twin (data only, LAST so the
            # captured-scan call's positional splice is untouched)
            th_lo)
        if chart_probe is not None:
            # the local chart cotangent: two trailing arguments, only
            # when asked for (every other call's argument list is unchanged)
            if _cap is not None:
                raise NotImplementedError(
                    "captured scan: the local chart cotangent is not served")
            _fn_args = _fn_args + (chart_probe, tuple(chart_geometry))
        if _cap is None:
            if chart_probe is not None and torch.compiler.is_compiling():
                from .registered_scan import chart_scan_apply as _chart_apply
                Y, S_final = _chart_apply(*_fn_args)
            elif _CUSTOMOP_SCAN and torch.compiler.is_compiling():
                # One registered NODE instead of a graph break
                # (and the Function itself when the pair refuses the
                # call).  In EAGER this branch is not taken at all.
                Y, S_final = _scan_apply(*_fn_args)
            else:
                Y, S_final = _SSDMimoFn.apply(*_fn_args)
        else:
            # ── driven by hand (see `capture_scan`) ──────────────────
            if bc_grouped or per_g or off_f is not None:
                raise NotImplementedError(
                    "captured scan: only the ungrouped, per-head-scalar "
                    "fused route with the in-kernel phase prologue is "
                    f"supported (bc_grouped={bc_grouped}, per_g={per_g}, "
                    f"host_theta_off={off_f is not None})")
            _sctx = _ShimCtx()
            Y, S_final = _SSDMimoFn.forward(
                _sctx, *_fn_args[:24], True, *_fn_args[25:])
            # The tensors above were built under `no_grad`, so their
            # `requires_grad` flags say nothing; the combined node wants
            # every gradient the Function can produce and drops the ones
            # it does not need.
            for _nm, _t in (("need_dthh", thh_f), ("need_dtheta", th_f),
                            ("need_dtheta_off", off_f),
                            ("need_dbeta", beta_f), ("need_dgamma", gamma_f),
                            ("need_dx0", x0_f), ("need_drot_bias", rb_f),
                            ("need_ddw", dw_pad), ("need_dlam", tl_f)):
                setattr(_sctx, _nm, _t is not None)
            _cap.append(ScanTape(_sctx, _cap_in, dict(
                B=B, H=H, G=G_SIZE, R=R, N=N, T=T, L=L, L_padded=L_padded,
                delta_w_is_none=delta_w is None,
                O=O,
                b_tail=tuple(_cap_in["B_tok"].shape[2:]),
                c_tail=tuple(_cap_in["C_tok"].shape[2:]))))
        # ── The LAYOUT of `y`, not its shape ──────────────────────────
        # `y_token_major` asks the store above to write [B, L, H, O, G]
        # memory instead of the kernel-native [B, H, O, G, L].  The
        # SHAPE returned is [B, H, (O,) G, L] either way -- only the
        # strides differ -- so every consumer between here and the
        # module (`ssd_mimo_scan`'s G-pad trim, the N-tiled sum) keeps
        # working unchanged, and `SSDMimoSSM._finish_forward` DETECTS
        # the layout from the strides rather than being told.  The
        # permutes below are views in both layouts.
        if c_per_o:
            y = Y.permute(0, 1, 4, 5, 2, 3).reshape(
                B, H, O, G_SIZE, L_padded)[..., :L]
        else:
            y = Y.squeeze(4).permute(0, 1, 4, 2, 3).reshape(
                B, H, G_SIZE, L_padded)[:, :, :, :L]
        return y.to(x_in.dtype), S_final

    raise ValueError("ssd_mimo_scan: unsupported fused geometry")
