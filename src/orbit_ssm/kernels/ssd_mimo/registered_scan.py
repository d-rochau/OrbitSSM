
from __future__ import annotations
import os
from typing import Optional
import torch
_CUSTOMOP = os.environ.get('HYBRID_SSM_COMPILE_CUSTOMOP', '1') == '1'

def customop_enabled() -> bool:
    return _CUSTOMOP
_PERM_TOKEN = (0, 3, 1, 2, 4, 5)
_UNPERM_TOKEN = (0, 2, 3, 1, 4, 5)
_PERM_CHAN = (0, 1, 4, 5, 2, 3)
_UNPERM_CHAN = (0, 1, 4, 5, 2, 3)

def _to_base(t, token_major: bool):
    return t.permute(*(_UNPERM_TOKEN if token_major else _UNPERM_CHAN))

def _from_base(t, token_major: bool):
    return t.permute(*(_PERM_TOKEN if token_major else _PERM_CHAN))

def _base_shape(Bsz, H, nc, T, O, G, token_major: bool):
    return (Bsz, nc, T, H, O, G) if token_major else (Bsz, H, O, G, nc, T)

def _ypre_token_major(y_token_major, in_kernel, has_gate, has_norm):
    return bool(y_token_major and (in_kernel or not has_gate or has_norm))

def _yg_token_major(y_token_major, in_kernel, has_gate, has_norm):
    if not has_gate:
        return _ypre_token_major(y_token_major, in_kernel, has_gate, has_norm)
    if in_kernel:
        return _ypre_token_major(y_token_major, in_kernel, has_gate, has_norm)
    if has_norm:
        return True
    return bool(y_token_major)

def _lever_state():
    from .fused_fwd_kernel import _env
    from .m3_epilogue import rot_bwd_mode
    from .siso_route import siso_enabled
    return dict(sall=_env('SALL', 'recompute'), sall_fp32=os.environ.get('UNIFIED_SSD_SALL_FP32', '0'), gate_inkernel=_env('GATE_INKERNEL', '0'), gate_fused=_env('GATE_FUSED', '1'), gate_out_io=_env('GATE_OUT_IO', '1'), phase_prefix=_env('PHASE_PREFIX', '1'), rot_bwd=rot_bwd_mode(), siso=bool(siso_enabled(rotating=True)))

def scan_refusal(args, lev=None) -> Optional[str]:
    Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam, H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot, theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo = args
    lev = _lever_state() if lev is None else lev
    if lev['siso']:
        return 'SSD_MIMO_SISO_FAST: the SISO schedule saves its own aux set'
    from .siso_route import siso_enabled
    real_cp = False
    if theta is None and theta_h is None and (rot_bias is None) and needs_bwd:
        from .chunkpar import chunkpar_wanted as _cpw, cp_refusal as _cpr
        from .chunkpar import real_cp_decision as _rcd, plan_flags as _pf
        real_cp = _cpr(theta=theta, theta_h=theta_h, rot_bias=rot_bias, x0=x0, fuse=None, dcs_ink=dcs_ink, T=T, N=N, G_SIZE=G_SIZE) is None and _rcd(n_serial_ctas=x_raw.shape[0] * H, nc=delta_cumsum.shape[2], T=T, N=N, G_SIZE=G_SIZE, x_dtype=x_raw.dtype, precision=precision, device=x_raw.device, needs_bwd=needs_bwd, flags=_pf(x0=x0, beta=beta, gamma=gamma, dw=dw, trap_lam=trap_lam, d_skip=d_skip, gate_z=gate_z, in_kernel=lev['gate_inkernel'] == '1'))
    if needs_bwd and siso_enabled(x_raw.device, rotating=theta is not None, io_bf16=x_raw.dtype == torch.bfloat16, T=T, real_cp=real_cp, bh=x_raw.shape[0] * H, L=delta_cumsum.shape[2] * T, needs_bwd=needs_bwd, thh=theta_h is not None):
        return 'the SISO route (default for rotating R = O = 1 bf16 T = 64; real calls where the chunk-parallel forward is selected) saves S_all and its own aux set'
    if lev['rot_bwd'] == 'split':
        return "SSD_MIMO_ROT_BWD=split: the ROTSPLIT prologue's ctx state"
    if needs_bwd and theta is not None and (x_raw.dtype == torch.bfloat16):
        from . import rot_route as _rr
        _r = None if _rr.explicit_rot_bwd() or _rr.explicit_siso() else _rr.pick(x_raw.shape[0] * H, delta_cumsum.shape[2] * T, x_raw.device, True, theta_h is not None)
        if _r is not None and _r[0] == 'split':
            return 'the calibrated structural route picks the ROTSPLIT backward (its ctx state)'
    if lev['gate_inkernel'] == '1':
        return "SSD_MIMO_GATE_INKERNEL=1: the in-kernel gate's y_pre buffer"
    if gate_z is not None and lev['gate_fused'] != '1':
        return 'SSD_MIMO_GATE_FUSED=0: the retired torch gate chain'
    if needs_bwd and lev['sall'] != 'recompute':
        return 'SSD_MIMO_SALL != recompute: S_all is a chunk-shaped save'
    if theta_off is not None:
        return 'host-built phase offsets (K12 theta_off)'
    if theta is not None and theta_incr and (lev['phase_prefix'] != '1'):
        return 'SSD_MIMO_PHASE_PREFIX=0: the K13 offset form is rebuilt in fwd'
    return None

class _OpCtx:

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

def _theta_bf16_ok_dt(theta_dtype, precision, x_dtype):
    from .fused_fwd_kernel import _env
    return bool(precision != 'ieee' and _env('THETA_BWD_FP32', '1') != '1' and (theta_dtype == torch.float32) and (x_dtype in (torch.bfloat16, torch.float16)))

def _theta_choice(theta, theta_lo, precision, x_dtype, phase_prefix):
    if theta is None:
        return 'none'
    eff_dt = torch.float32 if phase_prefix else theta.dtype
    if _theta_bf16_ok_dt(eff_dt, precision, x_dtype):
        if theta_lo is not None and theta_lo.dtype == torch.bfloat16 and (tuple(theta_lo.shape) == tuple(theta.shape)):
            return 'lo'
        return 'built'
    return 'built' if phase_prefix else 'theta'

def _phase_flags(theta, theta_off, theta_incr, needs_bwd, lev):
    phase_internal = bool(theta_incr and theta_off is None)
    phase_prefix = bool(phase_internal and needs_bwd and (lev['phase_prefix'] == '1'))
    return (phase_internal, phase_prefix)

def extras_from_outputs(out, args):
    ypre, dot, rstd, theta_saved = out[2:6]
    tm_p = _ypre_token_major(args[27], False, args[9] is not None, args[28] is not None)
    return (_from_base(ypre, tm_p) if ypre is not None and ypre.numel() else None, dot if dot is not None and dot.numel() else None, rstd if rstd is not None and rstd.numel() else None, theta_saved if theta_saved is not None and theta_saved.numel() else None)

def _build_ctx(args, extra, need, lev):
    Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam, H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot, theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo = args
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
    ctx.recompute_sall = True
    ctx.siso = False
    ctx.siso_why = 'registered scan route'
    ctx.siso_prec = precision.split(':')[1] if ':' in precision else precision
    ctx.siso_bwd = False
    ctx.sall_cp = False
    ctx.sall_rebuild_dcs = False
    ctx.siso_fuse = '0'
    ctx.rot_split = False
    ctx.has_dskip = has_dskip
    ctx.in_kernel_gate = False
    ctx.out_norm_eps = out_norm_eps
    ctx.need_dnormw = bool(need['dnormw'])
    ctx.has_trap = has_trap
    ctx.need_dlam = bool(has_trap and need['dlam'])
    phase_internal, phase_prefix = _phase_flags(theta, theta_off, theta_incr, needs_bwd, lev)
    theta_keep = {'none': None, 'theta': theta, 'lo': theta_lo, 'built': theta_saved}[_theta_choice(theta, theta_lo, precision, x_raw.dtype, phase_prefix)]
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
    ctx.need_dthh = bool(theta_h is not None and need['dthh'])
    ctx.need_dtheta = bool(theta is not None and need['dtheta'])
    ctx.need_dtheta_off = False
    ctx.need_dbeta = bool(beta is not None and need['dbeta'])
    ctx.need_dgamma = bool(gamma is not None and need['dgamma'])
    ctx.need_drot_bias = bool(rot_bias is not None and need['drot_bias'])
    ctx.need_dx0 = bool(x0 is not None and need['dx0'])
    ctx.need_ddw = bool(need['ddw'])
    _save_with_ctx_tensors(ctx, saved, dict(theta=theta_keep, theta_h=theta_h, theta_off=None, beta=beta, gamma=gamma, rot_bias=rot_bias, x0=x0, out_norm_w=out_norm_w, out_norm_rstd=rstd if out_norm_w is not None else None))
    return ctx

def _chunk_bwd_env():
    from .autograd import _chunk_bwd_env as f
    return f()
_NEED_KEYS = ('ddw', 'dthh', 'dtheta', 'dbeta', 'dgamma', 'drot_bias', 'dx0', 'dlam', 'dnormw')

@torch.library.custom_op('orbit_ssm::ssd_scan_fwd', mutates_args=())
def ssd_scan_fwd(Ctil: torch.Tensor, Btil: torch.Tensor, x_raw: torch.Tensor, dw: torch.Tensor, delta_cumsum: torch.Tensor, a_log: torch.Tensor, theta_h: Optional[torch.Tensor], theta: Optional[torch.Tensor], theta_off: Optional[torch.Tensor], gate_z: Optional[torch.Tensor], d_skip: Optional[torch.Tensor], beta: Optional[torch.Tensor], gamma: Optional[torch.Tensor], x0: Optional[torch.Tensor], rot_bias: Optional[torch.Tensor], trap_lam: Optional[torch.Tensor], H: int, T: int, N: int, G_SIZE: int, precision: str, Gb: int, Gc: int, bc_grouped: bool, needs_bwd: bool, bias_pre_rot: bool, theta_incr: bool, y_token_major: bool, out_norm_w: Optional[torch.Tensor], out_norm_eps: float, pair_half: bool, dcs_ink: bool, chunk_bwd: int, theta_lo: Optional[torch.Tensor]) -> list[torch.Tensor]:
    from .autograd import _SSDMimoFn, saved_view
    from .smem import SsdSmemOverflow
    args = (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam, H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot, theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo)
    lev = _lever_state()
    why = scan_refusal(args, lev)
    if why is not None:
        raise RuntimeError(f'orbit_ssm::ssd_scan_fwd was reached on a route it refuses ({why}); this is a bug in `scan_apply` -- run with HYBRID_SSM_COMPILE_CUSTOMOP=0')
    ctx = _OpCtx()
    try:
        with torch.amp.autocast('cuda', enabled=False), torch.no_grad():
            Yg, S_final = _SSDMimoFn.forward(ctx, *args)
    except SsdSmemOverflow as exc:
        raise RuntimeError(f'registered scan: this (N, chunk) needs the shared-memory repair, whose chunk cannot be known at trace time -- run with HYBRID_SSM_COMPILE_CUSTOMOP=0 ({exc})') from exc

    def _e():
        return x_raw.new_empty(0)
    sv = list(ctx.saved_tensors)
    has_gate = gate_z is not None
    has_dskip = d_skip is not None
    has_trap = trap_lam is not None
    if needs_bwd and sv[6] is not None:
        raise RuntimeError('registered scan: the S_all save mode is refused')
    if int(delta_cumsum.shape[-1]) != int(T):
        raise RuntimeError(f'registered scan: the launch ran at chunk {delta_cumsum.shape[-1]} but the graph was traced for {T}')
    idx_t = 7 + (2 if has_gate else 0) + (1 if has_dskip else 0)
    ypre = sv[7] if has_gate else None
    dot = sv[idx_t + 1] if has_trap and needs_bwd else None
    vctx = saved_view(ctx)
    rstd = vctx.out_norm_rstd if out_norm_w is not None else None
    _, _pp = _phase_flags(theta, theta_off, theta_incr, needs_bwd, lev)
    want = _theta_choice(theta, theta_lo, precision, x_raw.dtype, _pp)
    got = 'none' if vctx.theta is None else 'theta' if vctx.theta is theta else 'lo' if vctx.theta is theta_lo else 'built'
    if want != got:
        raise RuntimeError(f'registered scan: the saved phase is {got!r} but this route predicted {want!r} -- `_theta_choice` no longer mirrors `_SSDMimoFn.forward`')
    theta_saved = vctx.theta if want == 'built' else None
    tm_g = _yg_token_major(y_token_major, False, has_gate, out_norm_w is not None)
    tm_p = _ypre_token_major(y_token_major, False, has_gate, out_norm_w is not None)
    yg_base = _to_base(Yg, tm_g)
    ypre_base = _to_base(ypre, tm_p) if ypre is not None else _e()
    for nm, t in (('Yg', yg_base), ('y_pre', ypre_base)):
        if t.numel() and (not t.is_contiguous()):
            raise RuntimeError(f"registered scan: {nm} is not a view of a contiguous allocation in either known memory order (see this module's LAYOUT note)")
    return [yg_base, S_final, ypre_base, dot if dot is not None else _e(), rstd if rstd is not None else _e(), theta_saved if theta_saved is not None else _e()]

@ssd_scan_fwd.register_fake
def _ssd_scan_fwd_fake(Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam, H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot, theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo):
    from .fused_fwd_kernel import _env
    Bsz = x_raw.shape[0]
    nc = delta_cumsum.shape[2]
    R = x_raw.shape[3]
    c_per_o = Ctil.dim() == (6 if bc_grouped else 5)
    O = Ctil.shape[-1] if c_per_o else 1
    dev, f32 = (x_raw.device, torch.float32)
    has_gate = gate_z is not None
    has_norm = out_norm_w is not None

    def _e():
        return x_raw.new_empty(0)
    gate_dt = x_raw.dtype if _env('GATE_OUT_IO', '1') == '1' else f32
    tm_g = _yg_token_major(y_token_major, False, has_gate, has_norm)
    tm_p = _ypre_token_major(y_token_major, False, has_gate, has_norm)
    yg = torch.empty(_base_shape(Bsz, H, nc, T, O, G_SIZE, tm_g), device=dev, dtype=gate_dt if has_gate else x_raw.dtype)
    s_final = torch.empty((Bsz, H, G_SIZE, N), device=dev, dtype=f32)
    ypre = torch.empty(_base_shape(Bsz, H, nc, T, O, G_SIZE, tm_p), device=dev, dtype=x_raw.dtype) if has_gate else _e()
    if trap_lam is not None and needs_bwd:
        dot = torch.empty_like(dw) if R == 1 and O == 1 else torch.empty(tuple(dw.shape) + (O, R), device=dev, dtype=f32)
    else:
        dot = _e()
    rstd = torch.empty((Bsz, nc * T), device=dev, dtype=f32) if has_norm else _e()
    _, phase_prefix = _phase_flags(theta, theta_off, theta_incr, needs_bwd, _lever_state())
    if _theta_choice(theta, theta_lo, precision, x_raw.dtype, phase_prefix) == 'built':
        bf = _theta_bf16_ok_dt(f32 if phase_prefix else theta.dtype, precision, x_raw.dtype)
        theta_saved = torch.empty_like(theta, dtype=torch.bfloat16 if bf else f32)
    else:
        theta_saved = _e()
    return [yg, s_final, ypre, dot, rstd, theta_saved]

@torch.library.custom_op('orbit_ssm::ssd_scan_bwd', mutates_args=())
def ssd_scan_bwd(dYg: torch.Tensor, dS_final: Optional[torch.Tensor], Ctil: torch.Tensor, Btil: torch.Tensor, x_raw: torch.Tensor, dw: torch.Tensor, delta_cumsum: torch.Tensor, a_log: torch.Tensor, theta_h: Optional[torch.Tensor], theta: Optional[torch.Tensor], theta_off: Optional[torch.Tensor], gate_z: Optional[torch.Tensor], d_skip: Optional[torch.Tensor], beta: Optional[torch.Tensor], gamma: Optional[torch.Tensor], x0: Optional[torch.Tensor], rot_bias: Optional[torch.Tensor], trap_lam: Optional[torch.Tensor], H: int, T: int, N: int, G_SIZE: int, precision: str, Gb: int, Gc: int, bc_grouped: bool, needs_bwd: bool, bias_pre_rot: bool, theta_incr: bool, y_token_major: bool, out_norm_w: Optional[torch.Tensor], out_norm_eps: float, pair_half: bool, dcs_ink: bool, chunk_bwd: int, theta_lo: Optional[torch.Tensor], ypre: torch.Tensor, dot: torch.Tensor, rstd: torch.Tensor, theta_saved: torch.Tensor, need: list[bool]) -> list[torch.Tensor]:
    from .autograd import _SSDMimoFn
    args = (Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam, H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot, theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo)
    extra = extras_from_outputs([None, None, ypre, dot, rstd, theta_saved], args)
    ctx = _build_ctx(args, extra, dict(zip(_NEED_KEYS, list(need))), _lever_state())
    tm_g = _yg_token_major(y_token_major, False, gate_z is not None, out_norm_w is not None)
    with torch.amp.autocast('cuda', enabled=False), torch.no_grad():
        grads = _SSDMimoFn.backward(ctx, _from_base(dYg, tm_g), dS_final)
    return [g if g is not None else x_raw.new_empty(0) for g in list(grads[:16]) + [grads[28]]]

@ssd_scan_bwd.register_fake
def _ssd_scan_bwd_fake(dYg, dS_final, Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam, H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot, theta_incr, y_token_major, out_norm_w, out_norm_eps, pair_half, dcs_ink, chunk_bwd, theta_lo, ypre, dot, rstd, theta_saved, need):
    n = dict(zip(_NEED_KEYS, list(need)))

    def like(t, want=True):
        return torch.empty_like(t) if t is not None and want else x_raw.new_empty(0)
    empty = x_raw.new_empty(0)
    return [like(Ctil), like(Btil), like(x_raw), like(dw, n['ddw']), like(delta_cumsum), like(a_log), like(theta_h, n['dthh']), like(theta, n['dtheta']), empty, like(gate_z), like(d_skip), like(beta, n['dbeta']), like(gamma, n['dgamma']), like(x0, n['dx0']), like(rot_bias, n['drot_bias']), like(trap_lam, n['dlam']), like(out_norm_w, n['dnormw'])]
_T_POS = tuple(range(16)) + (28, 33)
_S_POS = tuple(range(16, 28)) + (29, 30, 31, 32)

def _setup(ctx, inputs, output):
    ctx.cfg = {i: inputs[i] for i in _S_POS}
    ctx.n_saved_in = len(_T_POS)
    ctx.save_for_backward(*[inputs[i] for i in _T_POS], *output[2:6])
    ctx.mark_non_differentiable(*output[2:])

def _backward(ctx, *grads):
    if len(grads) == 1 and isinstance(grads[0], (list, tuple)):
        grads = tuple(grads[0])
    dYg, dS_final = (grads[0], grads[1] if len(grads) > 1 else None)
    sv = list(ctx.saved_tensors)
    args = [None] * 34
    for slot, t in zip(_T_POS, sv[:ctx.n_saved_in]):
        args[slot] = t
    for slot, v in ctx.cfg.items():
        args[slot] = v
    extra = sv[ctx.n_saved_in:]
    ng = getattr(ctx, 'needs_input_grad', None)
    idx = dict(ddw=3, dthh=6, dtheta=7, dbeta=11, dgamma=12, drot_bias=14, dx0=13, dlam=15, dnormw=28)
    need = [bool(ng[idx[k]]) for k in _NEED_KEYS] if ng is not None else [True] * len(_NEED_KEYS)
    grads = ssd_scan_bwd(dYg, dS_final, *args, *extra, need)
    g = [x if x is not None and x.numel() else None for x in grads]
    return (g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11], g[12], g[13], g[14], g[15], None, None, None, None, None, None, None, None, None, None, None, None, g[16], None, None, None, None, None)
ssd_scan_fwd.register_autograd(_backward, setup_context=_setup)
_REFUSAL_LOG: list = []
_REFUSAL_LOG_MAX = 32

def refusal_log() -> list:
    return list(_REFUSAL_LOG)

def _opaque_apply(*args):
    from .autograd import _SSDMimoFn
    return _SSDMimoFn.apply(*args)
_opaque_apply_disabled = torch.compiler.disable(_opaque_apply)

def scan_apply(*args):
    why = scan_refusal(args)
    if why is not None:
        _REFUSAL_LOG.append(why)
        del _REFUSAL_LOG[:-_REFUSAL_LOG_MAX]
        return _opaque_apply_disabled(*args)
    out = ssd_scan_fwd(*args)
    tm_g = _yg_token_major(args[27], False, args[9] is not None, args[28] is not None)
    return (_from_base(out[0], tm_g), out[1])
