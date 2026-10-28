"""The GENERAL SSD-MIMO scan as a `torch.library` custom-op pair.

WHY.  `compile_compat.dynamo_opaque` (= `torch.compiler.disable`) keeps
Dynamo out of the Triton launchers by drawing a GRAPH BREAK at
`ssd_mimo_scan`.  A break is not free: at the boundary Dynamo materialises
the tensors that cross it, which can cost more than the eager ATen glue
Inductor fuses away.

WHAT THIS FILE DOES.  It registers the scan's autograd node — the
`_SSDMimoFn` forward and backward BODIES, unchanged — as
`orbit_ssm::ssd_scan_fwd` / `orbit_ssm::ssd_scan_bwd`, an opaque
custom-op PAIR with `register_fake` and `register_autograd`.  Dynamo
then puts a NODE in the graph instead of breaking, the host prep above
the node (`_fused_ssd_impl`'s pads, cumsum and views) is traced and
fused like any other torch code, and nothing crosses a boundary that
has to be copied first.

SHAPE OF THE PAIR (the same shape `registered.py`'s
`orbit_ssm::m2_fwd` uses, for the same reason): `ctx.save_for_backward`
does not survive a custom op, so the forward RETURNS everything its
backward needs that it alone can produce — `y_pre`, the trapezoid dot,
the output-norm `rstd`, the phase actually saved — and the backward
takes them as arguments next to the forward's own inputs.  The
forward's inputs are themselves most of the saved set, and they are
saved by `setup_context`, so nothing is copied to get it across.

LAYOUT.  `Yg` / `y_pre` are `[B, H, nc, T, O, G]` VIEWS of a contiguous
allocation in one of exactly two memory orders (`_native_y` /
`gate_epilogue`: channel-major `[B, H, O, G, L]` and token-major
`[B, L, H, O, G]`), and which one is a static function of this call's
flags.  A custom op's fake must predict its outputs' STRIDES, so the op
hands back the CONTIGUOUS BASE and the (traced) caller re-applies the
permutation — a view on both sides, and a fake that is plain
`torch.empty`.

WHAT IS SHAPE-VISIBLE AND WHAT IS NOT (the register_fake contract).
Every output shape here is a function of the input shapes and the
CONFIG INTS only.  The one runtime decision that could move a saved
shape is the chunk `T`: `_ssd_dispatch`'s shared-memory ladder and the
chunk guard can both change it.  The guard is a pure function of
(device capability, precision, theta_h, R, O) and is therefore already
applied before this op is reached, so `T` arrives as a plain int; the
LADDER is a measured, dry-compile decision that cannot be predicted
without compiling, so a call that would take it is REFUSED here by name
(`_ladder_would_fire` cannot be answered at trace time) — the op
verifies that the chunk it was handed is the chunk the kernels ran at
and raises rather than silently saving a differently-chunked tensor.

OFF SWITCH.  `HYBRID_SSM_COMPILE_CUSTOMOP=0`: `ssd_mimo_scan` keeps its
`dynamo_opaque` boundary and compile pays the graph break.

EAGER IS NOT ON THIS PATH.  The route is taken only under
`torch.compiler.is_compiling()`; in eager `_fused_ssd_impl` calls
`_SSDMimoFn.apply` directly, so eager numerics and speed are unaffected
(one bool test per scan call is the whole cost).
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from ..autograd_guard import no_double_backward as _no_double_backward
#: Read ONCE, at import -- the decorator on `ssd_mimo_scan` is chosen
#: from it at class/function definition time, so a later change of the
#: variable could only make the process inconsistent with itself.  Same
#: policy, and the same reason, as `compile_compat._OPAQUE`.
_CUSTOMOP = os.environ.get("HYBRID_SSM_COMPILE_CUSTOMOP", "1") == "1"


def customop_enabled() -> bool:
    """Whether the registered scan route is compiled in this process."""
    return _CUSTOMOP


# ── the two memory orders `Y` / `Yg` / `y_pre` are ever allocated in ──
# `fused_fwd_kernel._native_y` and `gate_epilogue.{ascalar_epilogue_fwd,
# gate_norm_fwd}` are the only allocators, and both spell exactly these:
#   token-major   base [B, nc, T, H, O, G] -> permute(0, 3, 1, 2, 4, 5)
#   channel-major base [B, H, O, G, nc, T] -> permute(0, 1, 4, 5, 2, 3)
_PERM_TOKEN = (0, 3, 1, 2, 4, 5)
_UNPERM_TOKEN = (0, 2, 3, 1, 4, 5)
_PERM_CHAN = (0, 1, 4, 5, 2, 3)
_UNPERM_CHAN = (0, 1, 4, 5, 2, 3)          # self-inverse


def _to_base(t, token_major: bool):
    """The contiguous allocation `t` is a permuted view of."""
    return t.permute(*(_UNPERM_TOKEN if token_major else _UNPERM_CHAN))


def _from_base(t, token_major: bool):
    """Undo `_to_base`: back to the logical [B, H, nc, T, O, G]."""
    return t.permute(*(_PERM_TOKEN if token_major else _PERM_CHAN))


def _base_shape(Bsz, H, nc, T, O, G, token_major: bool):
    return ((Bsz, nc, T, H, O, G) if token_major
            else (Bsz, H, O, G, nc, T))


# ── which of the two the store takes, from this call's flags alone ───
def _ypre_token_major(y_token_major, in_kernel, has_gate, has_norm):
    """`fused_chunk_fwd`'s own `y_token_major` argument."""
    return bool(y_token_major and (in_kernel or not has_gate or has_norm))


def _yg_token_major(y_token_major, in_kernel, has_gate, has_norm):
    """The layout of the tensor the Function RETURNS."""
    if not has_gate:
        return _ypre_token_major(y_token_major, in_kernel, has_gate, has_norm)
    if in_kernel:                       # the kernel's own gated store
        return _ypre_token_major(y_token_major, in_kernel, has_gate, has_norm)
    if has_norm:                        # `gate_norm_fwd`: always token
        return True
    return bool(y_token_major)          # G9 `ascalar_epilogue_fwd`


# ── env levers this route reads, resolved through the SAME helpers the
#    Function's own body reads them with (never a private copy) ────────
def _lever_state():
    from .fused_fwd_kernel import _env
    from .m3_epilogue import rot_bwd_mode
    from .siso_route import siso_enabled
    return dict(
        sall=_env("SALL", "recompute"),
        sall_fp32=_env("SALL_FP32", "0"),
        gate_inkernel=_env("GATE_INKERNEL", "0"),
        gate_fused=_env("GATE_FUSED", "1"),
        gate_out_io=_env("GATE_OUT_IO", "1"),
        phase_prefix=_env("PHASE_PREFIX", "1"),
        rot_bwd=rot_bwd_mode(),
        # the compile boundary: only an
        # explicit/rotating SISO route refuses it (a real call's default
        # SISO forward is the eager Function's own choice)
        siso=bool(siso_enabled(rotating=True)),
    )


# ═══════════════════════════════════════════════════════════════════
#  The refusal list.  Every clause names a route whose SAVED SET or
#  whose ctx state this pair does not carry; a refusal falls back to the
#  shipped `torch.compiler.disable` boundary (one graph break), never to a different answer.
# ═══════════════════════════════════════════════════════════════════
def scan_refusal(args, lev=None) -> Optional[str]:
    (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off,
     gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
     H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot,
     theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half,
     dcs_ink, chunk_bwd, theta_lo) = args
    lev = _lever_state() if lev is None else lev
    if lev["siso"]:
        return "SSD_MIMO_SISO_FAST: the SISO schedule saves its own aux set"
    # the SISO schedule is the DEFAULT for a rotating R = O = 1 bf16 call at T = 64 on every
    # GPU (`_lever_state` cannot know T / dtype): such a call stores
    # S_all and its own aux set, which this recompute-only custom op refuses -- take the
    # shipped `torch.compiler.disable` boundary instead of raising inside the compiled graph
    from .siso_route import compile_general_only, siso_enabled
    real_cp = False
    general = compile_general_only()    # the op bodies run the general route (`siso_route.general_route`)
    if (not general) and theta is None and theta_h is None and rot_bias is None and needs_bwd:
        # a REAL call takes the SISO forward exactly where the calibrated chunk-parallel
        # forward is selected (the decision `autograd` makes) -- it saves S_all too
        from .chunkpar import chunkpar_wanted as _cpw, cp_refusal as _cpr
        from .chunkpar import real_cp_decision as _rcd, plan_flags as _pf
        real_cp = (_cpr(theta=theta, theta_h=theta_h, rot_bias=rot_bias, x0=x0, fuse=None,
                        dcs_ink=dcs_ink, T=T, N=N, G_SIZE=G_SIZE) is None
                   and _rcd(n_serial_ctas=x_raw.shape[0] * H, nc=delta_cumsum.shape[2], T=T, N=N,
                            G_SIZE=G_SIZE, x_dtype=x_raw.dtype, precision=precision,
                            device=x_raw.device, needs_bwd=needs_bwd,
                            flags=_pf(x0=x0, beta=beta, gamma=gamma, dw=dw, trap_lam=trap_lam, d_skip=d_skip,
                                      gate_z=gate_z, in_kernel=lev['gate_inkernel'] == '1')))
    if (not general) and needs_bwd and siso_enabled(x_raw.device, rotating=theta is not None,
                                  io_bf16=x_raw.dtype == torch.bfloat16, T=T,
                                  real_cp=real_cp, bh=x_raw.shape[0] * H,
                                  L=delta_cumsum.shape[2] * T, needs_bwd=needs_bwd,
                                  thh=theta_h is not None):
        return ("the SISO route (default for rotating R = O = 1 bf16 T = 64; real calls "
                "where the chunk-parallel forward is selected) saves S_all and its own "
                "aux set")
    if lev["rot_bwd"] == "split":
        return "SSD_MIMO_ROT_BWD=split: the ROTSPLIT prologue's ctx state"
    if needs_bwd and theta is not None and x_raw.dtype == torch.bfloat16:
        from . import rot_route as _rr
        _r = None if (_rr.explicit_rot_bwd() or _rr.explicit_siso()) else _rr.pick(
            x_raw.shape[0] * H, delta_cumsum.shape[2] * T, x_raw.device, True, theta_h is not None)
        if _r is not None and _r[0] == "split":
            return "the calibrated structural route picks the ROTSPLIT backward (its ctx state)"
    if lev["gate_inkernel"] == "1":
        return "SSD_MIMO_GATE_INKERNEL=1: the in-kernel gate's y_pre buffer"
    if gate_z is not None and lev["gate_fused"] != "1":
        return "SSD_MIMO_GATE_FUSED=0: the torch gate chain"
    if needs_bwd and lev["sall"] != "recompute":
        return "SSD_MIMO_SALL != recompute: S_all is a chunk-shaped save"
    if theta_off is not None:
        return "host-built phase offsets (K12 theta_off)"
    if theta is not None and theta_incr and lev["phase_prefix"] != "1":
        return "SSD_MIMO_PHASE_PREFIX=0: the K13 offset form is rebuilt in fwd"
    return None


# ═══════════════════════════════════════════════════════════════════
#  ctx reconstruction
# ═══════════════════════════════════════════════════════════════════
class _OpCtx:
    """`_SSDMimoFn`'s ctx, rebuilt on the far side of the op boundary.

    Every field is either one of the op's OWN arguments, a tensor the
    forward returned, or a flag derived from the two by the same
    expression the Function's forward uses.  The custom-op test
    asserts this object equals the real ctx field for field.
    """

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


def _theta_bf16_ok_dt(theta_dtype, precision, x_dtype):
    """`_SSDMimoFn.forward`'s `_theta_bf16_ok`, off the dtypes alone.

    (`siso_bwd` is False on every route this pair serves -- the SISO
    schedule is refused by name above.)
    """
    from .fused_fwd_kernel import _env
    # the Function keeps the fp32 phase for its
    # backward unless `theta_bwd_fp32` is False -- mirrored here, or the
    # op's saved-phase prediction disagrees with what the Function did.
    return bool(precision != "ieee"
                and _env("THETA_BWD_FP32", "1") != "1"
                and theta_dtype == torch.float32
                and x_dtype in (torch.bfloat16, torch.float16))


def _theta_choice(theta, theta_lo, precision, x_dtype, phase_prefix):
    """WHICH tensor `_SSDMimoFn.forward` parks in `ctx.theta`.

    "none" / "theta" / "lo" are one of the op's OWN arguments, so the
    forward op returns a 0-element placeholder for them (a custom op may
    not return a tensor that aliases an input) and the backward picks the
    argument back up here; only "built" -- the phase-prefix build or the
    bf16 twin the forward had to make itself -- travels as an output.
    The forward op ASSERTS that this prediction is what the Function
    actually did, so the two cannot drift apart silently.
    """
    if theta is None:
        return "none"
    eff_dt = torch.float32 if phase_prefix else theta.dtype
    if _theta_bf16_ok_dt(eff_dt, precision, x_dtype):
        if (theta_lo is not None and theta_lo.dtype == torch.bfloat16
                and tuple(theta_lo.shape) == tuple(theta.shape)):
            return "lo"
        return "built"
    return "built" if phase_prefix else "theta"


def _phase_flags(theta, theta_off, theta_incr, needs_bwd, lev):
    phase_internal = bool(theta_incr and theta_off is None)
    phase_prefix = bool(phase_internal and needs_bwd
                        and lev["phase_prefix"] == "1")
    return phase_internal, phase_prefix


def extras_from_outputs(out, args):
    """The forward op's extra outputs, back in the shapes the ctx wants.

    `y_pre` travels as its contiguous BASE (see the module's LAYOUT
    note), so the permutation is re-applied here -- everything else is
    already in its own shape.  A 0-element placeholder means "absent".
    """
    ypre, dot, rstd, theta_saved = out[2:6]
    tm_p = _ypre_token_major(args[27], False, args[9] is not None,
                             args[28] is not None)
    return (_from_base(ypre, tm_p) if ypre is not None and ypre.numel()
            else None,
            dot if dot is not None and dot.numel() else None,
            rstd if rstd is not None and rstd.numel() else None,
            theta_saved if theta_saved is not None and theta_saved.numel()
            else None)


def _build_ctx(args, extra, need, lev):
    """The backward's ctx, from the op's arguments and returned tensors."""
    (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off,
     gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
     H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot,
     theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half,
     dcs_ink, chunk_bwd, theta_lo) = args
    from .autograd import _save_with_ctx_tensors
    ypre, dot_out, rstd, theta_saved = extra
    ctx = _OpCtx()
    has_gate = gate_z is not None
    has_dskip = d_skip is not None
    has_trap = trap_lam is not None
    saved = [Ctil, Btil, x_raw, dw, delta_cumsum, a_log, None]
    if has_gate:
        saved.extend([ypre, gate_z])
    if has_dskip:
        saved.append(d_skip)
    if has_trap:
        saved.extend([trap_lam, dot_out if dot_out is not None else trap_lam])
    # the levers the forward read, read the same way
    ctx.recompute_sall = True
    ctx.siso = False
    ctx.siso_why = "registered scan route"
    ctx.siso_prec = (precision.split(":")[1] if ":" in precision
                     else precision)
    ctx.siso_bwd = False
    ctx.sall_cp = False                  # (SALL_SISO fields: this route is never SISO)
    ctx.sall_rebuild_dcs = False
    # the registered route never takes the SISO schedule, so
    # the mode the Function records is always its OFF value here.
    ctx.siso_fuse = "0"
    ctx.rot_split = False
    ctx.has_dskip = has_dskip
    ctx.in_kernel_gate = False
    ctx.out_norm_eps = out_norm_eps
    ctx.need_dnormw = bool(need["dnormw"])
    ctx.has_trap = has_trap
    ctx.need_dlam = bool(has_trap and need["dlam"])
    phase_internal, phase_prefix = _phase_flags(
        theta, theta_off, theta_incr, needs_bwd, lev)
    theta_keep = {"none": None, "theta": theta, "lo": theta_lo,
                  "built": theta_saved}[
        _theta_choice(theta, theta_lo, precision, x_raw.dtype, phase_prefix)]
    ctx.phase_internal = phase_internal
    ctx.phase_prefix = phase_prefix
    ctx.bias_pre_rot = bias_pre_rot
    ctx.pair_half = bool(pair_half)
    ctx.dcs_ink = bool(dcs_ink)
    ctx.precision = precision
    ctx.Gb = Gb
    ctx.Gc = Gc
    ctx.bc_grouped = bc_grouped
    ctx.has_gate = has_gate
    ctx.T = T
    ctx.chunk_bwd = int(chunk_bwd or 0) or _chunk_bwd_env()
    ctx.need_dthh = bool(theta_h is not None and need["dthh"])
    ctx.need_dtheta = bool(theta is not None and need["dtheta"])
    ctx.need_dtheta_off = False
    ctx.need_dbeta = bool(beta is not None and need["dbeta"])
    ctx.need_dgamma = bool(gamma is not None and need["dgamma"])
    ctx.need_drot_bias = bool(rot_bias is not None and need["drot_bias"])
    ctx.need_dx0 = bool(x0 is not None and need["dx0"])
    ctx.need_ddw = bool(need["ddw"])
    # the Function's own saved layout: positional set, then the named
    # tensors (`autograd._save_with_ctx_tensors`)
    _save_with_ctx_tensors(ctx, saved, dict(
        theta=theta_keep, theta_h=theta_h, theta_off=None, beta=beta,
        gamma=gamma, rot_bias=rot_bias, x0=x0, out_norm_w=out_norm_w,
        out_norm_rstd=rstd if out_norm_w is not None else None))
    return ctx


def _chunk_bwd_env():
    from .autograd import _chunk_bwd_env as f
    return f()


_NEED_KEYS = ("ddw", "dthh", "dtheta", "dbeta", "dgamma", "drot_bias",
              "dx0", "dlam", "dnormw")


# ═══════════════════════════════════════════════════════════════════
#  the op pair
# ═══════════════════════════════════════════════════════════════════
@torch.library.custom_op("orbit_ssm::ssd_scan_fwd", mutates_args=())
def ssd_scan_fwd(
    Ctil: torch.Tensor, Btil: torch.Tensor, x_raw: torch.Tensor,
    dw: torch.Tensor, delta_cumsum: torch.Tensor, a_log: torch.Tensor,
    theta_h: Optional[torch.Tensor], theta: Optional[torch.Tensor],
    theta_off: Optional[torch.Tensor], gate_z: Optional[torch.Tensor],
    d_skip: Optional[torch.Tensor], beta: Optional[torch.Tensor],
    gamma: Optional[torch.Tensor], x0: Optional[torch.Tensor],
    rot_bias: Optional[torch.Tensor], trap_lam: Optional[torch.Tensor],
    H: int, T: int, N: int, G_SIZE: int, precision: str, Gb: int, Gc: int,
    bc_grouped: bool, needs_bwd: bool, bias_pre_rot: bool, theta_incr: bool,
    y_token_major: bool, out_norm_w: Optional[torch.Tensor],
    out_norm_eps: float, pair_half: bool, dcs_ink: bool, chunk_bwd: int,
    theta_lo: Optional[torch.Tensor],
) -> list[torch.Tensor]:
    """`_SSDMimoFn.forward`, opaque, with its saved set as outputs.

    Returns `[Yg_base, S_final, ypre_base, dot, rstd, theta_saved]`; an
    absent member is a 0-element tensor (the convention
    `registered.py`'s `endpoint_local_bwd` already uses).
    """
    args = (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta,
            theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
            H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd,
            bias_pre_rot, theta_incr, y_token_major, out_norm_w,
            out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo)
    return _scan_fwd_impl(args)


def _scan_fwd_impl(args, chart=()):
    """`ssd_scan_fwd`'s body; `chart` = (chart_probe, chart_geometry) appended to the Function's arguments (the
    local chart cotangent op, `ssd_scan_chart_fwd`), () for the plain op."""
    from .autograd import _SSDMimoFn, saved_view
    from .siso_route import general_route
    from .smem import SsdSmemOverflow
    (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta,
     theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
     H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd,
     bias_pre_rot, theta_incr, y_token_major, out_norm_w,
     out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo) = args
    lev = _lever_state()
    why = scan_refusal(args, lev)
    if why is not None:
        raise RuntimeError(
            "orbit_ssm::ssd_scan_fwd was reached on a route it refuses "
            f"({why}); this is a bug in `scan_apply` -- run with "
            "HYBRID_SSM_COMPILE_CUSTOMOP=0")
    ctx = _OpCtx()
    try:
        with torch.amp.autocast("cuda", enabled=False), torch.no_grad(), general_route():
            Yg, S_final = _SSDMimoFn.forward(ctx, *args, *chart)
    except SsdSmemOverflow as exc:
        # The ONE decision this route cannot predict at trace time: the
        # shared-memory repair is a dry-compile measurement, and taking
        # it would change the chunk -- i.e. the shape of everything
        # `register_fake` promised.  Refuse loudly; the graph-breaking
        # boundary still serves this cell.
        raise RuntimeError(
            "registered scan: this (N, chunk) needs the shared-memory "
            "repair, whose chunk cannot be known at trace time -- run "
            f"with HYBRID_SSM_COMPILE_CUSTOMOP=0 ({exc})") from exc
    def _e():
        # a FRESH 0-element tensor per slot: two returns that are the
        # same object would trip the custom-op aliasing check.
        return x_raw.new_empty(0)
    sv = list(ctx.saved_tensors)
    has_gate = gate_z is not None
    has_dskip = d_skip is not None
    has_trap = trap_lam is not None
    if needs_bwd and sv[6] is not None:
        raise RuntimeError("registered scan: the S_all save mode is refused")
    if int(delta_cumsum.shape[-1]) != int(T):
        raise RuntimeError(
            "registered scan: the launch ran at chunk "
            f"{delta_cumsum.shape[-1]} but the graph was traced for {T}")
    idx_t = 7 + (2 if has_gate else 0) + (1 if has_dskip else 0)
    ypre = sv[7] if has_gate else None
    # `dot_out` is only built when there IS a backward; without one the
    # Function parks `trap_lam` in that slot, which is not a save.
    dot = sv[idx_t + 1] if (has_trap and needs_bwd) else None
    vctx = saved_view(ctx)          # the named saved tensors (item A)
    rstd = vctx.out_norm_rstd if out_norm_w is not None else None
    _, _pp = _phase_flags(theta, theta_off, theta_incr, needs_bwd, lev)
    want = _theta_choice(theta, theta_lo, precision, x_raw.dtype, _pp)
    got = ("none" if vctx.theta is None else
           "theta" if vctx.theta is theta else
           "lo" if vctx.theta is theta_lo else "built")
    if want != got:
        raise RuntimeError(
            "registered scan: the saved phase is "
            f"{got!r} but this route predicted {want!r} -- "
            "`_theta_choice` does not mirror `_SSDMimoFn.forward`")
    theta_saved = vctx.theta if want == "built" else None
    tm_g = _yg_token_major(y_token_major, False, has_gate,
                           out_norm_w is not None)
    tm_p = _ypre_token_major(y_token_major, False, has_gate,
                             out_norm_w is not None)
    yg_base = _to_base(Yg, tm_g)
    ypre_base = _to_base(ypre, tm_p) if ypre is not None else _e()
    for nm, t in (("Yg", yg_base), ("y_pre", ypre_base)):
        if t.numel() and not t.is_contiguous():
            raise RuntimeError(
                f"registered scan: {nm} is not a view of a contiguous "
                "allocation in either known memory order (see this "
                "module's LAYOUT note)")
    return [yg_base, S_final, ypre_base,
            dot if dot is not None else _e(),
            rstd if rstd is not None else _e(),
            theta_saved if theta_saved is not None else _e()]


@ssd_scan_fwd.register_fake
def _ssd_scan_fwd_fake(
    Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off,
    gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
    H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot,
    theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half,
    dcs_ink, chunk_bwd, theta_lo,
):
    """Shapes (and strides) only -- no data is read.

    Every one is a function of the input shapes and the config ints; see
    the module docstring for the single decision that is NOT predictable
    this way (the shared-memory chunk repair, refused above).
    """
    from .fused_fwd_kernel import _env
    Bsz = x_raw.shape[0]
    nc = delta_cumsum.shape[2]
    R = x_raw.shape[3]
    c_per_o = Ctil.dim() == (6 if bc_grouped else 5)
    O = Ctil.shape[-1] if c_per_o else 1
    dev, f32 = x_raw.device, torch.float32
    has_gate = gate_z is not None
    has_norm = out_norm_w is not None

    def _e():
        return x_raw.new_empty(0)
    gate_dt = x_raw.dtype if _env("GATE_OUT_IO", "1") == "1" else f32
    tm_g = _yg_token_major(y_token_major, False, has_gate, has_norm)
    tm_p = _ypre_token_major(y_token_major, False, has_gate, has_norm)
    yg = torch.empty(_base_shape(Bsz, H, nc, T, O, G_SIZE, tm_g),
                     device=dev, dtype=gate_dt if has_gate else x_raw.dtype)
    s_final = torch.empty((Bsz, H, G_SIZE, N), device=dev, dtype=f32)
    ypre = (torch.empty(_base_shape(Bsz, H, nc, T, O, G_SIZE, tm_p),
                        device=dev, dtype=x_raw.dtype)
            if has_gate else _e())
    if trap_lam is not None and needs_bwd:
        dot = (torch.empty_like(dw) if (R == 1 and O == 1)
               else torch.empty(tuple(dw.shape) + (O, R), device=dev,
                                dtype=f32))
    else:
        dot = _e()
    rstd = (torch.empty((Bsz, nc * T), device=dev, dtype=f32)
            if has_norm else _e())
    _, phase_prefix = _phase_flags(theta, theta_off, theta_incr,
                                   needs_bwd, _lever_state())
    if _theta_choice(theta, theta_lo, precision, x_raw.dtype,
                     phase_prefix) == "built":
        bf = _theta_bf16_ok_dt(f32 if phase_prefix else theta.dtype,
                               precision, x_raw.dtype)
        theta_saved = torch.empty_like(
            theta, dtype=torch.bfloat16 if bf else f32)
    else:
        theta_saved = _e()
    return [yg, s_final, ypre, dot, rstd, theta_saved]


@torch.library.custom_op("orbit_ssm::ssd_scan_bwd", mutates_args=())
def ssd_scan_bwd(
    dYg: torch.Tensor, dS_final: Optional[torch.Tensor],
    Ctil: torch.Tensor, Btil: torch.Tensor, x_raw: torch.Tensor,
    dw: torch.Tensor, delta_cumsum: torch.Tensor, a_log: torch.Tensor,
    theta_h: Optional[torch.Tensor], theta: Optional[torch.Tensor],
    theta_off: Optional[torch.Tensor], gate_z: Optional[torch.Tensor],
    d_skip: Optional[torch.Tensor], beta: Optional[torch.Tensor],
    gamma: Optional[torch.Tensor], x0: Optional[torch.Tensor],
    rot_bias: Optional[torch.Tensor], trap_lam: Optional[torch.Tensor],
    H: int, T: int, N: int, G_SIZE: int, precision: str, Gb: int, Gc: int,
    bc_grouped: bool, needs_bwd: bool, bias_pre_rot: bool, theta_incr: bool,
    y_token_major: bool, out_norm_w: Optional[torch.Tensor],
    out_norm_eps: float, pair_half: bool, dcs_ink: bool, chunk_bwd: int,
    theta_lo: Optional[torch.Tensor],
    ypre: torch.Tensor, dot: torch.Tensor, rstd: torch.Tensor,
    theta_saved: torch.Tensor, need: list[bool],
) -> list[torch.Tensor]:
    """`_SSDMimoFn.backward`, opaque, driven from a rebuilt ctx.

    Returns the Function's 16 differentiable gradients in its own order
    with `dNormW` appended; an absent one is a 0-element tensor.
    """
    args = (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta,
            theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
            H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd,
            bias_pre_rot, theta_incr, y_token_major, out_norm_w,
            out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo)
    return _scan_bwd_impl(dYg, dS_final, args, (ypre, dot, rstd, theta_saved), need)


def _scan_bwd_impl(dYg, dS_final, args, saved, need, chart=None):
    """`ssd_scan_bwd`'s body; `chart` = (geometry, probe shape) for the local chart cotangent op (the packed tangent
    is then appended to the returned gradients)."""
    from .autograd import _SSDMimoFn
    from .siso_route import general_route
    x_raw, gate_z, y_token_major, out_norm_w = args[2], args[9], args[27], args[28]
    ypre, dot, rstd, theta_saved = saved
    extra = extras_from_outputs([None, None, ypre, dot, rstd, theta_saved],
                                args)
    ctx = _build_ctx(args, extra, dict(zip(_NEED_KEYS, list(need))),
                     _lever_state())
    ctx.chart = chart
    tm_g = _yg_token_major(y_token_major, False, gate_z is not None,
                           out_norm_w is not None)
    with torch.amp.autocast("cuda", enabled=False), torch.no_grad(), general_route():
        grads = _SSDMimoFn.backward(ctx, _from_base(dYg, tm_g), dS_final)
    out = [g if g is not None else x_raw.new_empty(0)
           for g in list(grads[:16]) + [grads[28]]]
    if chart is not None:
        out.append(grads[34])
    return out


@ssd_scan_bwd.register_fake
def _ssd_scan_bwd_fake(
    dYg, dS_final, Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h,
    theta, theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
    H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot,
    theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half,
    dcs_ink, chunk_bwd, theta_lo, ypre, dot, rstd, theta_saved, need,
):
    n = dict(zip(_NEED_KEYS, list(need)))

    def like(t, want=True):
        return (torch.empty_like(t) if (t is not None and want)
                else x_raw.new_empty(0))
    empty = x_raw.new_empty(0)
    return [like(Ctil), like(Btil), like(x_raw), like(dw, n["ddw"]),
            like(delta_cumsum), like(a_log),
            like(theta_h, n["dthh"]), like(theta, n["dtheta"]), empty,
            like(gate_z), like(d_skip), like(beta, n["dbeta"]),
            like(gamma, n["dgamma"]), like(x0, n["dx0"]),
            like(rot_bias, n["drot_bias"]), like(trap_lam, n["dlam"]),
            like(out_norm_w, n["dnormw"])]


# tensor / non-tensor argument positions of the pair's shared arg list
_T_POS = tuple(range(16)) + (28, 33)
_S_POS = tuple(range(16, 28)) + (29, 30, 31, 32)


def _setup(ctx, inputs, output):
    ctx.cfg = {i: inputs[i] for i in _S_POS}
    ctx.n_saved_in = len(_T_POS)
    ctx.save_for_backward(*[inputs[i] for i in _T_POS], *output[2:6])
    ctx.mark_non_differentiable(*output[2:])


@_no_double_backward("SSD-MIMO scan")
def _backward(ctx, *grads):
    # a `list[Tensor]` return arrives as ONE list of gradients, not as
    # one argument per member.
    if len(grads) == 1 and isinstance(grads[0], (list, tuple)):
        grads = tuple(grads[0])
    dYg, dS_final = grads[0], (grads[1] if len(grads) > 1 else None)
    sv = list(ctx.saved_tensors)
    args = [None] * 34
    for slot, t in zip(_T_POS, sv[:ctx.n_saved_in]):
        args[slot] = t
    for slot, v in ctx.cfg.items():
        args[slot] = v
    extra = sv[ctx.n_saved_in:]
    ng = getattr(ctx, "needs_input_grad", None)
    idx = dict(ddw=3, dthh=6, dtheta=7, dbeta=11, dgamma=12, drot_bias=14,
               dx0=13, dlam=15, dnormw=28)
    need = ([bool(ng[idx[k]]) for k in _NEED_KEYS] if ng is not None
            else [True] * len(_NEED_KEYS))
    grads = ssd_scan_bwd(dYg, dS_final, *args, *extra, need)
    g = [x if (x is not None and x.numel()) else None for x in grads]
    return (g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9],
            g[10], g[11], g[12], g[13], g[14], g[15],
            None, None, None, None, None, None, None, None,
            None, None, None, None,
            g[16], None, None, None, None, None)


ssd_scan_fwd.register_autograd(_backward, setup_context=_setup)


# ═══════════════════════════════════════════════════════════════════
#  the call site
# ═══════════════════════════════════════════════════════════════════
_REFUSAL_LOG: list = []
_REFUSAL_LOG_MAX = 32


def refusal_log() -> list:
    """The last 32 refusals of the registered route, newest last."""
    return list(_REFUSAL_LOG)


def _opaque_apply(why, *args):
    # the refusal is logged HERE, inside the disabled boundary (runtime, once per executed call): a mutation
    # of a module-level list in the traced `scan_apply` makes Dynamo guard on `len(_REFUSAL_LOG)`, so every
    # further scan call of the same process recompiled `scan_apply` until `recompile_limit` (8) was hit
    # (measured under HYBRID_SSM_COMPILE_SISO=boundary)
    _REFUSAL_LOG.append(why)
    del _REFUSAL_LOG[:-_REFUSAL_LOG_MAX]
    from .autograd import _SSDMimoFn
    return _SSDMimoFn.apply(*args)


_opaque_apply_disabled = torch.compiler.disable(_opaque_apply)


def scan_apply(*args):
    """`_SSDMimoFn.apply`, as a graph NODE where this route serves the call.

    Inside the envelope: one `orbit_ssm::ssd_scan_fwd` node and no graph
    break.  Outside it: the shipped `torch.compiler.disable` boundary,
    i.e. exactly the behaviour before CUSTOMOP.  Only reached under
    `torch.compile` -- eager calls the Function directly.
    """
    why = scan_refusal(args)
    if why is not None:
        return _opaque_apply_disabled(why, *args)
    out = ssd_scan_fwd(*args)
    tm_g = _yg_token_major(args[27], False, args[9] is not None,
                           args[28] is not None)
    return _from_base(out[0], tm_g), out[1]


# ═══════════════════════════════════════════════════════════════════
#  the local chart cotangent: the same pair with the chart probe .  A SEPARATE op pair, so
#  `ssd_scan_fwd` / `_bwd` (every other caller's compiled node) keep their schema; the bodies are shared.
#  The probe is an input whose "gradient" (the packed folded-frame tangent F) the backward appends.
# ═══════════════════════════════════════════════════════════════════
@torch.library.custom_op("orbit_ssm::ssd_scan_chart_fwd", mutates_args=())
def ssd_scan_chart_fwd(
    Ctil: torch.Tensor, Btil: torch.Tensor, x_raw: torch.Tensor,
    dw: torch.Tensor, delta_cumsum: torch.Tensor, a_log: torch.Tensor,
    theta_h: Optional[torch.Tensor], theta: Optional[torch.Tensor],
    theta_off: Optional[torch.Tensor], gate_z: Optional[torch.Tensor],
    d_skip: Optional[torch.Tensor], beta: Optional[torch.Tensor],
    gamma: Optional[torch.Tensor], x0: Optional[torch.Tensor],
    rot_bias: Optional[torch.Tensor], trap_lam: Optional[torch.Tensor],
    H: int, T: int, N: int, G_SIZE: int, precision: str, Gb: int, Gc: int,
    bc_grouped: bool, needs_bwd: bool, bias_pre_rot: bool, theta_incr: bool,
    y_token_major: bool, out_norm_w: Optional[torch.Tensor],
    out_norm_eps: float, pair_half: bool, dcs_ink: bool, chunk_bwd: int,
    theta_lo: Optional[torch.Tensor],
    chart_probe: torch.Tensor, chart_mode: str, chart_k: int, chart_Mf: int, chart_Mp: int,
    chart_emitter: str,
) -> list[torch.Tensor]:
    """`ssd_scan_fwd` with the chart probe (the forward ignores it)."""
    args = (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta,
            theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
            H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd,
            bias_pre_rot, theta_incr, y_token_major, out_norm_w,
            out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo)
    return _scan_fwd_impl(args, (chart_probe, (chart_mode, chart_k, chart_Mf, chart_Mp, chart_emitter)))


@ssd_scan_chart_fwd.register_fake
def _ssd_scan_chart_fwd_fake(*args):
    return _ssd_scan_fwd_fake(*args[:34])


@torch.library.custom_op("orbit_ssm::ssd_scan_chart_bwd", mutates_args=())
def ssd_scan_chart_bwd(
    dYg: torch.Tensor, dS_final: Optional[torch.Tensor],
    Ctil: torch.Tensor, Btil: torch.Tensor, x_raw: torch.Tensor,
    dw: torch.Tensor, delta_cumsum: torch.Tensor, a_log: torch.Tensor,
    theta_h: Optional[torch.Tensor], theta: Optional[torch.Tensor],
    theta_off: Optional[torch.Tensor], gate_z: Optional[torch.Tensor],
    d_skip: Optional[torch.Tensor], beta: Optional[torch.Tensor],
    gamma: Optional[torch.Tensor], x0: Optional[torch.Tensor],
    rot_bias: Optional[torch.Tensor], trap_lam: Optional[torch.Tensor],
    H: int, T: int, N: int, G_SIZE: int, precision: str, Gb: int, Gc: int,
    bc_grouped: bool, needs_bwd: bool, bias_pre_rot: bool, theta_incr: bool,
    y_token_major: bool, out_norm_w: Optional[torch.Tensor],
    out_norm_eps: float, pair_half: bool, dcs_ink: bool, chunk_bwd: int,
    theta_lo: Optional[torch.Tensor],
    ypre: torch.Tensor, dot: torch.Tensor, rstd: torch.Tensor,
    theta_saved: torch.Tensor, need: list[bool],
    chart_mode: str, chart_k: int, chart_Mf: int, chart_Mp: int, chart_emitter: str, chart_shape: list[int],
) -> list[torch.Tensor]:
    """`ssd_scan_bwd` + the packed chart tangent F [B, L, G, NF] (the last output)."""
    args = (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta,
            theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam,
            H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd,
            bias_pre_rot, theta_incr, y_token_major, out_norm_w,
            out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo)
    return _scan_bwd_impl(dYg, dS_final, args, (ypre, dot, rstd, theta_saved), need,
                          chart=((chart_mode, chart_k, chart_Mf, chart_Mp, chart_emitter), tuple(chart_shape)))


@ssd_scan_chart_bwd.register_fake
def _ssd_scan_chart_bwd_fake(*args):
    shape = args[-1]
    x_raw = args[4]
    return _ssd_scan_bwd_fake(*args[:41]) + [x_raw.new_empty(tuple(shape), dtype=torch.float32)]


def _setup_chart(ctx, inputs, output):
    _setup(ctx, inputs[:34], output)
    ctx.chart = (inputs[35], inputs[36], inputs[37], inputs[38], inputs[39], list(inputs[34].shape))


def _backward_chart(ctx, *grads):
    if len(grads) == 1 and isinstance(grads[0], (list, tuple)):
        grads = tuple(grads[0])
    dYg, dS_final = grads[0], (grads[1] if len(grads) > 1 else None)
    sv = list(ctx.saved_tensors)
    args = [None] * 34
    for slot, t in zip(_T_POS, sv[:ctx.n_saved_in]):
        args[slot] = t
    for slot, v in ctx.cfg.items():
        args[slot] = v
    extra = sv[ctx.n_saved_in:]
    ng = getattr(ctx, "needs_input_grad", None)
    idx = dict(ddw=3, dthh=6, dtheta=7, dbeta=11, dgamma=12, drot_bias=14,
               dx0=13, dlam=15, dnormw=28)
    need = ([bool(ng[idx[k]]) for k in _NEED_KEYS] if ng is not None
            else [True] * len(_NEED_KEYS))
    out = ssd_scan_chart_bwd(dYg, dS_final, *args, *extra, need, *ctx.chart)
    g = [x if (x is not None and x.numel()) else None for x in out]
    return (g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9],
            g[10], g[11], g[12], g[13], g[14], g[15],
            None, None, None, None, None, None, None, None,
            None, None, None, None,
            g[16], None, None, None, None, None,
            g[17], None, None, None, None, None)


ssd_scan_chart_fwd.register_autograd(_backward_chart, setup_context=_setup_chart)


def _chart_opaque(why, *args):
    _REFUSAL_LOG.append(why)
    del _REFUSAL_LOG[:-_REFUSAL_LOG_MAX]
    from .autograd import _SSDMimoFn
    return _SSDMimoFn.apply(*args)


_chart_opaque_disabled = torch.compiler.disable(_chart_opaque)


def chart_scan_apply(*args):
    """`_SSDMimoFn.apply(*args, chart_probe, chart_geometry)` as one `orbit_ssm::ssd_scan_chart_fwd` node under
    `torch.compile` (the plain pair's envelope, `scan_refusal`); outside it the shipped boundary."""
    base, probe, geo = args[:34], args[34], args[35]
    why = scan_refusal(base)
    if why is not None:
        return _chart_opaque_disabled(why, *args)
    out = ssd_scan_chart_fwd(*base, probe, str(geo[0]), int(geo[1]), int(geo[2]), int(geo[3]),
                             str(geo[4]) if len(geo) > 4 else "fused")
    tm_g = _yg_token_major(base[27], False, base[9] is not None, base[28] is not None)
    return _from_base(out[0], tm_g), out[1]
