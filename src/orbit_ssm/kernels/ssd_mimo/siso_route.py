
import os
import torch
from .fused_fwd_kernel import _env

def gluon_default_route(device, rotating: bool, io_bf16: bool) -> bool:
    if not (rotating and io_bf16):
        return False
    if _env('SISO_FAST', 'auto').strip().lower() in ('0', 'off', 'false'):
        return False
    from . import gluon_sm90_siso_bwd as _g
    return _g.mode() == 'auto' and _g.default_serves(device)

def siso_enabled(device=None, rotating: bool=False, io_bf16: bool=False, T: int | None=None, real_cp: bool=False, bh: int | None=None, L: int | None=None, needs_bwd: bool=True, thh: bool=False, io_fp32: bool=False) -> bool:
    m = _env('SISO_FAST', 'auto').strip().lower()
    if m in ('1', 'on', 'true'):
        return True
    if m in ('0', 'off', 'false'):
        return False
    if not rotating:
        return bool(real_cp)
    if rotating and io_fp32 and (not io_bf16):
        if needs_bwd or bh is None or L is None:
            return False
        from .rot_route import pick
        r = pick(bh, L, device if device is not None else torch.device('cuda'), needs_bwd, thh, dtype='fp32')
        return r is not None and r[0] == 'siso' and (r[1] == T)
    if not (T == 64 and bool(rotating and io_bf16)):
        return False
    if bh is not None and L is not None:
        from .rot_route import pick
        r = pick(bh, L, device if device is not None else torch.device('cuda'), needs_bwd, thh)
        if r is not None and r[0] != 'siso' and (r[1] == T):
            return False
    return True
_FUSE_MODES = ('0', 'b', 'c')

def siso_fuse() -> str:
    v = _env('SISO_FUSE', '0')
    return v if v in _FUSE_MODES else '0'

def siso_xtm() -> str:
    return _env('SISO_XTM', '0')

def siso_bc(t):
    return t[..., 0] if t.dim() == 5 and t.shape[-1] == 1 else t

def siso_refusal(*, Ctil, Btil, x_raw, G_SIZE, N, bc_grouped, Gb, Gc, theta_off, in_kernel_gate, T, theta=None, theta_h=None, rot_bias=None):
    R = x_raw.shape[3]
    O = Ctil.shape[-1] if Ctil.dim() == (6 if bc_grouped else 5) else 1
    if R != 1 or O != 1:
        return 'R=%d O=%d (the SISO schedule is R = O = 1)' % (R, O)
    if bc_grouped or Gb != 1 or Gc != 1:
        return 'grouped B/C (Gb=%d Gc=%d)' % (Gb, Gc)
    if siso_bc(Btil).dim() != 4 or siso_bc(Ctil).dim() != 4:
        return 'B/C carry an R/O axis'
    if G_SIZE > 64 or G_SIZE % 16:
        return 'G_SIZE=%d (one G tile per head, 16 | G <= 64)' % G_SIZE
    if theta_off is not None:
        return 'a host-built phase OFFSET (K12); the SISO schedule takes the cumulated phase'
    if in_kernel_gate:
        return 'the in-kernel gate (SSD_MIMO_GATE_INKERNEL=1)'
    if theta is None and (theta_h is not None or rot_bias is not None):
        return 'theta_h / rot_bias without a per-token phase'
    if N < 16 or N & N - 1:
        return 'N=%d is not a power of two >= 16' % N
    return None

def siso_backward(ctx, Ctil, Btil, x_raw, dw, delta_cumsum, a_log, S_all, dY, dS_final, trap_lam, d_skip, dskip_dd, needs=None):
    from .siso_bwd_kernel import siso_chunk_bwd
    g = siso_chunk_bwd(ctx.siso_aux, siso_bc(Ctil), siso_bc(Btil), x_raw, delta_cumsum, a_log, S_all, dY, theta=ctx.theta, theta_h=ctx.theta_h, rot_bias=ctx.rot_bias, beta=ctx.beta, gamma=ctx.gamma, bias_pre_rot=ctx.bias_pre_rot, dw=dw, trap_lam=trap_lam, x0=ctx.x0, d_skip=d_skip, dS_final=dS_final, precision=ctx.siso_prec, dskip_dd=dskip_dd, needs=needs)

    def _as(t, like):
        return None if t is None or like is None else t.reshape(like.shape)
    return (g['dX'], g['dC'].reshape(Ctil.shape), g['dB'].reshape(Btil.shape), g['dDelta'], _as(g['dThetaH'], ctx.theta_h), _as(g['dTheta'], ctx.theta), _as(g['dBeta'], ctx.beta), _as(g['dGamma'], ctx.gamma), g['dX0'], _as(g['dRotBias'], ctx.rot_bias), g['dDW'], None, None, g['dLam'], g['dDsk'])

def dth_rot_can_serve(*, io_dtype, bias_pre_rot, beta, gamma, theta, N):
    if theta is None:
        return False
    if io_dtype not in ('bf16', 'fp16', 'fp32'):
        return False
    if not (bool(bias_pre_rot) or (beta is None and gamma is None)):
        return False
    return theta.shape[-1] == N // 2

def siso_precision_refusal(*, io_dtype, precision):
    if io_dtype in (torch.bfloat16, torch.float16) and precision in ('ieee', 'tf32'):
        return 'precision %r on 16-bit IO: the SISO scan rounds every MMA operand to the IO dtype, so it cannot honour it; the general route serves the call' % precision
    return None

def siso_extra_refusal(*, x_raw, precision, T, N, G_SIZE, pair_half, theta):
    why = siso_precision_refusal(io_dtype=x_raw.dtype, precision=precision)
    if why is not None:
        return why
    if x_raw.element_size() == 4 and precision == 'ieee' and (T >= 64) and (N * G_SIZE >= 128 * 64):
        from .smem import device_smem_limit
        if device_smem_limit(x_raw.device) < 196608:
            return "fp32 'ieee' operands at T=%d, N=%d, G=%d: the SISO scan adjoint needs 196,608 B of shared memory, the device allows %d B" % (T, N, G_SIZE, device_smem_limit(x_raw.device))
    if G_SIZE & G_SIZE - 1:
        return "G=%d is not a power of two (the SISO kernels' tl.arange over the head group)" % G_SIZE
    if pair_half and theta is not None:
        return "the rotate-half pair layout (pair_layout='half'): the SISO kernels rotate interleaved pairs only"
    return None

def dth_rot_refusal(*, needs_angle_grad, device, io_dtype, bias_pre_rot, beta, gamma, theta, N, T=64, G_SIZE=None):
    if not needs_angle_grad:
        return None
    tag = {torch.bfloat16: 'bf16', torch.float16: 'fp16'}.get(io_dtype, 'fp32')
    if tag == 'bf16':
        from . import gluon_sm90_siso_bwd as _g
        if _g.enabled(device) and T == 64 and (N in (64, 128)) and (G_SIZE is None or G_SIZE in (32, 64)):
            return None
    if dth_rot_can_serve(io_dtype=tag, bias_pre_rot=bias_pre_rot, beta=beta, gamma=gamma, theta=theta, N=N):
        return None
    return 'the angle adjoint needs the rotated-frame DTH_ROT, which cannot serve this call (io=%s, post-rotation bias=%s, phase last dim %d vs N/2=%d) and the Gluon body does not serve it' % (tag, not (bool(bias_pre_rot) or (beta is None and gamma is None)), theta.shape[-1], N // 2)

def siso_map_head_major() -> bool:
    return _env('SISO_MAP_HEAD_MAJOR', '0') == '1'

def siso_ytm() -> bool:
    return _env('SISO_YTM', '0') == '1'
