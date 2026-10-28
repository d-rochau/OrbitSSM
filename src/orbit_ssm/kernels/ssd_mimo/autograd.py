
import os as _os
import contextlib as _contextlib
import threading as _threading
import warnings
from triton.runtime.errors import OutOfResources as _OutOfResources
import torch
import torch.nn.functional as F
from typing import Optional
import math as _math
_BF16_MMA = _os.environ.get('UNIFIED_SSD_BF16_MMA', '0') == '1'
from ..cumprod.pi_rotate import pi_rotate_triton
from .common import _mimo_axes, _resolve_stable, _check_mimo_support
from .fused_fwd_kernel import fused_chunk_fwd, _env, dot_precision, mma_io_enabled
from .fused_bwd_kernel import fused_chunk_bwd_5kernel, pairhalf_r1_t64_fault, pairhalf_io_t64_fault, rot_io_bwd_t64_fault
from .m3_epilogue import trap_prologue
from ...compile_compat import dynamo_opaque
from ...layers.proj import autocast_io
from .registered_scan import customop_enabled as _customop_enabled
from .registered_scan import scan_apply as _scan_apply
from ... import kernel_options as _ko
from .smem import N_SPLIT, T_HALVE, SsdSmemOverflow, chunk_ladder, measuring_possible, note_chunk_clamped, note_n_split, preferred_strategy, resolve_strategy, strategy_key, device_smem_limit, smem_clamp_enabled
_CUSTOMOP_SCAN = _customop_enabled()
from . import measured_plan as _measured_plan
_scan_boundary = (lambda f: f) if _CUSTOMOP_SCAN else dynamo_opaque
_FUSED_SSD = _env('FUSED', '1') == '1'
_ROT_OPERAND_DTYPES = ('fp32', 'io')
_ROT_WHERE = ('host', 'kernel')

def _is_dense_layout(t) -> bool:
    seen = sorted(((abs(st), sz) for st, sz in zip(t.stride(), t.shape) if sz > 1))
    expect = 1
    for st, sz in seen:
        if st != expect:
            return False
        expect *= sz
    return expect == t.numel() and all((st >= 0 for st in t.stride())) and (t.numel() > 0)

def _dense_or_contiguous(t):
    if t.is_contiguous() or not _is_dense_layout(t):
        return t.contiguous()
    return t

def _rot_bwd_split() -> bool:
    from .m3_epilogue import rot_bwd_mode
    return rot_bwd_mode() == 'split'

def _rot_route_split(x_raw, H, L, T, *, R_sp, O_sp, envelope, pair_half, thh):
    from . import rot_route as rr
    if rr.explicit_rot_bwd() or rr.explicit_siso() or R_sp != 1 or (O_sp != 1) or (x_raw.dtype not in (torch.bfloat16, torch.float32)) or pair_half:
        return False
    r = rr.pick(x_raw.shape[0] * H, L, x_raw.device, True, thh, dtype='fp32' if x_raw.dtype == torch.float32 else None)
    if r is None or r[0] != 'split' or r[1] != T:
        return False
    from .siso_route import siso_refusal
    return siso_refusal(**envelope) is None

def _rot_split_dtype(io_dtype, precision):
    want = _env('ROT_SPLIT_DTYPE', 'io').lower()
    if want == 'fp32' or precision == 'ieee':
        return torch.float32
    if want not in ('io', 'fp32'):
        raise ValueError("SSD_MIMO_ROT_SPLIT_DTYPE must be 'io' or 'fp32'")
    return io_dtype

def _gate_out_io() -> bool:
    return _env('GATE_OUT_IO', '1') == '1'

def _dskip_in_kc() -> bool:
    v = _env('DSKIP_IN_KC', '1')
    return 'all' if v == 'all' else 'dx' if v == '1' else 'off'

def _real_r1o1_call(ctx, x_raw, Y_pre) -> bool:
    return bool(ctx.theta is None and ctx.theta_h is None and (ctx.rot_bias is None) and (x_raw.shape[3] == 1) and (Y_pre.shape[4] == 1))

def _gate_epilogue_reads_strided() -> bool:
    return _env('GATE_INKERNEL', '0') != '1' and _env('GATE_FUSED', '1') == '1'

def _rot_where(mode=None) -> str:
    if mode is None:
        mode = _env('ROT_WHERE', 'kernel')
    if mode not in _ROT_WHERE:
        raise ValueError('rot_where must be one of %s, got %r' % (_ROT_WHERE, mode))
    return mode

def _rot_operand_io(mode=None) -> bool:
    if mode is None:
        mode = _env('ROT_OPERAND_DTYPE', 'fp32')
    if mode not in _ROT_OPERAND_DTYPES:
        raise ValueError('rot_operand_dtype must be one of %s, got %r' % (_ROT_OPERAND_DTYPES, mode))
    return mode == 'io'
_CAPTURE = _threading.local()
_CAPTURE.slot = None

class _ShimCtx:
    __slots__ = ('saved_tensors', '__dict__')

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
_CTX_SAVED = ('theta', 'theta_h', 'theta_off', 'beta', 'gamma', 'rot_bias', 'x0', 'out_norm_w', 'out_norm_rstd', 'rs_theta', 'rs_theta_h')
_SISO_AUX = 'siso_aux.'

def _save_with_ctx_tensors(ctx, saved, named, siso_aux=None):
    names = list(_CTX_SAVED)
    vals = [named.get(k) for k in _CTX_SAVED]
    ctx.siso_aux_meta = None
    if siso_aux is not None:
        ctx.siso_aux_meta = {k: v for k, v in siso_aux.items() if not isinstance(v, torch.Tensor)}
        for k, v in siso_aux.items():
            if isinstance(v, torch.Tensor):
                names.append(_SISO_AUX + k)
                vals.append(v)
    ctx.saved_head = len(saved)
    ctx.saved_names = tuple(names)
    ctx.save_for_backward(*saved, *vals)

class _SavedCtx:
    __slots__ = ('_ctx', '_t', '_sv')

    def __init__(self, ctx):
        object.__setattr__(self, '_ctx', ctx)
        sv = tuple(ctx.saved_tensors)
        object.__setattr__(self, '_sv', sv)
        object.__setattr__(self, '_t', dict(zip(ctx.saved_names, sv[ctx.saved_head:])))

    def __getattr__(self, k):
        if k == 'saved_tensors':
            return self._sv
        t = self._t
        if k in t:
            return t[k]
        if k == 'siso_aux':
            meta = self._ctx.siso_aux_meta
            if meta is None:
                return None
            aux = dict(meta)
            aux.update({n[len(_SISO_AUX):]: v for n, v in t.items() if n.startswith(_SISO_AUX)})
            return aux
        return getattr(self._ctx, k)

    def __setattr__(self, k, v):
        setattr(self._ctx, k, v)

def saved_view(ctx):
    return ctx if isinstance(ctx, _SavedCtx) else _SavedCtx(ctx)

@_contextlib.contextmanager
def capture_scan():
    prev = getattr(_CAPTURE, 'slot', None)
    _CAPTURE.slot = []
    try:
        yield _CAPTURE.slot
    finally:
        _CAPTURE.slot = prev

def _capture_slot():
    return getattr(_CAPTURE, 'slot', None)

def _rev_cumsum(g):
    return g.flip(-1).cumsum(-1).flip(-1)

def _dcs_ink_on(R, O, use_siso=False):
    lever = _os.environ.get('SSD_MIMO_DCS_INK', 'auto').lower()
    if lever in ('0', 'false', 'no', 'off'):
        return False
    if use_siso:
        return False
    if lever in ('1', 'true', 'yes', 'on'):
        return True
    return R * O > 1

def _bc_group_major_on(bc_grouped, rotating=False, wants_chunk_bwd=False):
    lever = _os.environ.get('SSD_MIMO_BC_GROUP_MAJOR', 'auto').lower()
    if lever in ('0', 'false', 'no', 'off'):
        out = False
    elif lever in ('1', 'true', 'yes', 'on'):
        out = True
    else:
        out = bool(bc_grouped) and (not bool(rotating))
    if not torch.compiler.is_compiling():
        _BC_LAYOUT_LOG.append(dict(bc_grouped=bool(bc_grouped), rotating=bool(rotating), wants_chunk_bwd=bool(wants_chunk_bwd), lever=lever, group_major=out))
        del _BC_LAYOUT_LOG[:-_BC_LAYOUT_LOG_MAX]
    return out
_BC_LAYOUT_LOG: list = []
_BC_LAYOUT_LOG_MAX = 32

def bc_layout_log() -> list:
    return list(_BC_LAYOUT_LOG)

def _bc_chunk_grouped(tok_pad, B, nc, T, Gx, N, RO, group_major):
    L = nc * T
    if RO:
        if group_major:
            return tok_pad.view(B, L, Gx, N, RO).permute(0, 2, 4, 1, 3).contiguous().view(B, Gx, RO, nc, T, N).permute(0, 3, 1, 4, 5, 2)
        return tok_pad.view(B, nc, T, Gx, N, RO).permute(0, 1, 3, 5, 2, 4).contiguous().permute(0, 1, 2, 4, 5, 3)
    if group_major:
        return tok_pad.view(B, L, Gx, N).permute(0, 2, 1, 3).contiguous().view(B, Gx, nc, T, N).permute(0, 2, 1, 3, 4)
    return tok_pad.view(B, nc, T, Gx, N).transpose(2, 3).contiguous()

def _rechunk(t, ca, ta, T_new, T_old=None):
    if t is None:
        return None
    sz, st = (list(t.shape), list(t.stride()))
    if T_old is not None and sz[ta] != T_old:
        return None
    if st[ca] != sz[ta] * st[ta]:
        return None
    L = sz[ca] * sz[ta]
    if T_new <= 0 or L % T_new:
        return None
    sz[ca], sz[ta] = (L // T_new, T_new)
    st[ca] = T_new * st[ta]
    return t.as_strided(sz, st, t.storage_offset())

def _chunk_bwd_env():
    v = _env('CHUNK_BWD', '').strip().lower()
    if v in ('', 'auto', '0'):
        return 0
    return int(v)
_CHUNK_BWD = _threading.local()
_CHUNK_BWD.T = None

@_contextlib.contextmanager
def _chunk_bwd_request(T_bwd):
    prev = getattr(_CHUNK_BWD, 'T', None)
    _CHUNK_BWD.T = int(T_bwd or 0)
    try:
        yield
    finally:
        _CHUNK_BWD.T = prev

def _chunk_bwd_asked():
    return int(getattr(_CHUNK_BWD, 'T', None) or 0) or _chunk_bwd_env()
_CHUNK_BWD_LOG: list = []
_CHUNK_BWD_LOG_MAX = 32

def chunk_bwd_log() -> list:
    return list(_CHUNK_BWD_LOG)

def _note_chunk_bwd(rec: dict) -> None:
    if torch.compiler.is_compiling():
        return
    _CHUNK_BWD_LOG.append(rec)
    del _CHUNK_BWD_LOG[:-_CHUNK_BWD_LOG_MAX]

class SsdBackwardChunkRefused(RuntimeError):
    pass

def _cb_refuse(why, T_bwd):
    raise SsdBackwardChunkRefused('SSD-MIMO backward: the score-path kernel does not fit shared memory at chunk %d, and the backward cannot be re-chunked shorter here: %s.  Repairs, in order: SSD_MIMO_DCS_INK=1 (the chunk-local decay prefix becomes a VIEW of delta, which is what makes the re-chunk exact), a shorter `chunk_size`, or SSD_MIMO_SALL=recompute if it was turned off.' % (T_bwd, why))

class ScanTape:

    def __init__(self, ctx, inputs, meta):
        self.ctx = ctx
        self.inputs = inputs
        self.meta = meta

    def dY_from_y(self, dy):
        m = self.meta
        if m['L_padded'] != m['L']:
            raise NotImplementedError("captured scan: chunk-padded L needs the gradient zero-extended; the combined node's envelope excludes it")
        Bsz, H, G, nc, T, O = (m['B'], m['H'], m['G'], m['L'] // m['T'], m['T'], m['O'])
        if O == 1:
            return dy.reshape(Bsz, H, G, nc, T).permute(0, 1, 3, 4, 2).unsqueeze(4)
        return dy.reshape(Bsz, H, O, G, nc, T).permute(0, 1, 4, 5, 2, 3)

    def backward(self, dY, dS_final=None):
        m = self.meta
        src = self.inputs
        g = _SSDMimoFn.backward(self.ctx, dY, dS_final)
        dCtil, dBtil, dX, dDW, dDeltaCs, da_log, dThetaH, dTheta, dThOff, dZ, dDsk, dBeta, dGamma, dX0, dRotBias, dLam = g[:16]
        dNormW = g[28] if len(g) > 28 else None
        L, Lp = (m['L'], m['L_padded'])
        Bsz, H = (m['B'], m['H'])
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
        _back('x_in', dX, shape=(Bsz, H, m['G'], m['R'], Lp), trim_l=4)
        d_delta = None
        if dDeltaCs is not None:
            d_delta = dDeltaCs.reshape(Bsz, H, Lp)
        if m['delta_w_is_none'] and dDW is not None:
            dw_v = dDW.reshape(Bsz, H, Lp)
            d_delta = dw_v if d_delta is None else d_delta + dw_v
            out['delta_w'] = None
        else:
            _back('delta_w', dDW, shape=(Bsz, H, Lp), trim_l=2)
        _back('delta', d_delta, trim_l=2)
        _back('B_tok', dBtil, shape=(Bsz, Lp) + tuple(m['b_tail']), trim_l=1)
        _back('C_tok', dCtil, shape=(Bsz, Lp) + tuple(m['c_tail']), trim_l=1)
        _back('a_log', da_log)
        _back('theta_h', dThetaH)
        if src['theta'] is not None:
            th_shape = (Bsz, Lp) + tuple(src['theta'].shape[2:])
            _back('theta', dTheta, shape=th_shape, trim_l=1)
        else:
            out['theta'] = None
        _back('gate_z', dZ, trim_l=1)
        _back('d_skip', dDsk)
        _back('beta', dBeta)
        _back('gamma', dGamma)
        _back('x0', dX0)
        _back('rot_bias', dRotBias)
        _back('trap_lam', dLam, shape=(Bsz, H, Lp), trim_l=2)
        _back('out_norm_w', dNormW)
        if dThOff is not None:
            raise AssertionError('captured scan returned a host dThetaOff')
        return out

class _SSDMimoFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Ctil, Btil, x_raw, dw, delta_cumsum, a_log, theta_h, theta, theta_off, gate_z, d_skip, beta, gamma, x0, rot_bias, trap_lam, H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd=True, bias_pre_rot=False, theta_incr=False, y_token_major=False, out_norm_w=None, out_norm_eps=1e-05, pair_half=False, dcs_ink=False, chunk_bwd=0, theta_lo=None):
        recompute = _env('SALL', 'recompute') == 'recompute' and needs_bwd
        delta_raw = delta_cumsum
        if not dcs_ink:
            from .fused_fwd_kernel import seg_prefix_host
            delta_cumsum = seg_prefix_host(delta_raw, a_log)
        has_gate = gate_z is not None
        has_dskip = d_skip is not None
        in_kernel = has_gate and _env('GATE_INKERNEL', '0') == '1'
        from .siso_route import siso_bc, siso_enabled, siso_fuse, siso_refusal, siso_ytm
        real_cp = False
        if theta is None and theta_h is None and (rot_bias is None):
            from .chunkpar import chunkpar_wanted as _cpw, cp_refusal as _cpr
            from .chunkpar import real_cp_decision as _rcd, plan_flags as _pf
            real_cp = _cpr(theta=theta, theta_h=theta_h, rot_bias=rot_bias, x0=x0, fuse=None, dcs_ink=dcs_ink, T=T, N=N, G_SIZE=G_SIZE) is None and _rcd(n_serial_ctas=x_raw.shape[0] * H, nc=delta_cumsum.shape[2], T=T, N=N, G_SIZE=G_SIZE, x_dtype=x_raw.dtype, precision=precision, device=x_raw.device, needs_bwd=needs_bwd, flags=_pf(x0=x0, beta=beta, gamma=gamma, dw=dw, trap_lam=trap_lam, d_skip=d_skip, gate_z=gate_z, in_kernel=in_kernel))
        siso_why = None if siso_enabled(x_raw.device, rotating=theta is not None, io_bf16=x_raw.dtype == torch.bfloat16, T=T, real_cp=real_cp, bh=x_raw.shape[0] * H, L=delta_cumsum.shape[2] * T, needs_bwd=needs_bwd, thh=theta_h is not None, io_fp32=x_raw.dtype == torch.float32 and precision != 'ieee') else "SSD_MIMO_SISO_FAST is off (or 'auto' does not apply)"
        if siso_why is None:
            siso_why = siso_refusal(Ctil=Ctil, Btil=Btil, x_raw=x_raw, G_SIZE=G_SIZE, N=N, bc_grouped=bc_grouped, Gb=Gb, Gc=Gc, theta_off=theta_off, in_kernel_gate=in_kernel, T=T, theta=theta, theta_h=theta_h, rot_bias=rot_bias)
        if siso_why is None:
            from .siso_route import siso_extra_refusal
            siso_why = siso_extra_refusal(x_raw=x_raw, precision=precision, T=T, N=N, G_SIZE=G_SIZE, pair_half=pair_half, theta=theta)
        if siso_why is None and theta is not None:
            from .siso_route import dth_rot_refusal
            siso_why = dth_rot_refusal(needs_angle_grad=needs_bwd, device=x_raw.device, io_dtype=x_raw.dtype, bias_pre_rot=bias_pre_rot, beta=beta, gamma=gamma, theta=theta, N=N, T=delta_cumsum.shape[-1], G_SIZE=G_SIZE)
        use_siso = siso_why is None
        if use_siso and dcs_ink:
            raise NotImplementedError('SSD_MIMO_DCS_INK=1 with the SISO fast path: the in-kernel decay prefix is wired into the fused kernels only (set SSD_MIMO_SISO_FAST=0 or SSD_MIMO_DCS_INK=0)')
        fuse = siso_fuse() if use_siso and theta is not None else '0'
        if use_siso and (theta is None and _env('SALL_SISO', '1') == '1' and (fuse not in ('b', 'c'))):
            pass
        elif use_siso:
            recompute = False
        if out_norm_w is not None and (not (has_gate and (not in_kernel) and (_env('GATE_FUSED', '1') == '1'))):
            raise NotImplementedError('the fused output RMSNorm rides the DEDICATED gate epilogue kernel: it needs gate_z, SSD_MIMO_GATE_INKERNEL=0 and SSD_MIMO_GATE_FUSED=1')
        if has_dskip and (not has_gate):
            raise NotImplementedError('the fused D-skip rides the gate epilogue (its adjoint reads the same y_pre tile); pass gate_z, or apply the skip module-side')
        ypre = None
        if in_kernel:
            Bsz = x_raw.shape[0]
            nc = delta_cumsum.shape[2]
            O = Ctil.shape[-1] if Ctil.dim() == (6 if bc_grouped else 5) else 1
            if needs_bwd:
                ypre = torch.empty(Bsz, H, O, G_SIZE, nc * T, device=x_raw.device, dtype=x_raw.dtype).view(Bsz, H, O, G_SIZE, nc, T).permute(0, 1, 4, 5, 2, 3)
            gate_dt = x_raw.dtype if _gate_out_io() else torch.float32
        else:
            gate_dt = None
        has_trap = trap_lam is not None
        dot_out = None
        if has_trap and needs_bwd:
            R_trap = x_raw.shape[3]
            O_trap = Ctil.shape[-1] if Ctil.dim() == (6 if bc_grouped else 5) else 1
            dot_out = torch.empty_like(dw) if R_trap == 1 and O_trap == 1 else torch.empty(dw.shape + (O_trap, R_trap), device=dw.device, dtype=torch.float32)
        phase_internal = bool(theta_incr and theta_off is None)
        phase_prefix = bool(phase_internal and (needs_bwd or use_siso) and (_env('PHASE_PREFIX', '1') == '1' or use_siso))
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
        ctx.rot_split = False
        rs_theta = rs_theta_h = None
        if theta is not None and needs_bwd and (not use_siso) and bias_pre_rot and (rot_bias is None) and (theta_off is None) and (_rot_bwd_split() or _rot_route_split(x_raw, H, delta_cumsum.shape[2] * T, T, R_sp=x_raw.shape[3], O_sp=Ctil.shape[-1] if Ctil.dim() == (6 if bc_grouped else 5) else 1, envelope=dict(Ctil=Ctil, Btil=Btil, x_raw=x_raw, G_SIZE=G_SIZE, N=N, bc_grouped=bc_grouped, Gb=Gb, Gc=Gc, theta_off=theta_off, in_kernel_gate=in_kernel, T=T, theta=theta, theta_h=theta_h, rot_bias=rot_bias), pair_half=pair_half, thh=theta_h is not None)):
            from .m3_epilogue import rot_prologue
            ctx.rs_bshape, ctx.rs_bstride = (tuple(Btil.shape), Btil.stride())
            ctx.rs_cshape, ctx.rs_cstride = (tuple(Ctil.shape), Ctil.stride())
            ctx.rs_bdtype, ctx.rs_cdtype = (Btil.dtype, Ctil.dtype)
            with torch.no_grad():
                Ctil, Btil = rot_prologue(Ctil, Btil, theta, theta_h.float().contiguous() if theta_h is not None else None, beta.float().contiguous() if beta is not None else None, gamma.float().contiguous() if gamma is not None else None, H, Gb=Gb, Gc=Gc, bc_grouped=bc_grouped, out_dtype=_rot_split_dtype(x_raw.dtype, precision), pair_half=pair_half)
            ctx.rot_split = True
            ctx.rs_need_dthh = theta_h is not None and theta_h.requires_grad
            ctx.rs_need_dbeta = beta is not None and beta.requires_grad
            ctx.rs_need_dgamma = gamma is not None and gamma.requires_grad
            rs_theta, rs_theta_h = (theta, theta_h)
            ctx.H_rs = H
            ctx.rs_beta_shape = tuple(beta.shape) if beta is not None else None
            ctx.rs_gamma_shape = tuple(gamma.shape) if gamma is not None else None
            ctx.rs_thh_shape = tuple(theta_h.shape) if theta_h is not None else None
            ctx.rs_Gb, ctx.rs_Gc = (Gb, Gc)
            ctx.rs_bc_grouped = bc_grouped
            theta = theta_h = beta = gamma = rot_bias = None
            bias_pre_rot = False
            bc_grouped = True
            Gb = Gc = H
            ctx.rs_pair_half = bool(pair_half)
            pair_half = False
        siso_aux = None
        use_cp = False
        if use_siso and theta is None:
            from .chunkpar import chunkpar_wanted, cp_refusal
            cp_why = cp_refusal(theta=theta, theta_h=theta_h, rot_bias=rot_bias, x0=x0, fuse=None, dcs_ink=dcs_ink, T=T, N=N, G_SIZE=G_SIZE)
            use_cp = cp_why is None and chunkpar_wanted(x_raw.shape[0] * H, delta_cumsum.shape[2], x_raw.device, part='fwd' if needs_bwd else 'fwd_nograd', dtype=x_raw.dtype)
            from . import chunkpar as _cpm
            _cpm._LAST_FWD_ROUTE['fwd'] = 'chunkpar' if use_cp else cp_why or 'serial'
            if _env('CP_TRACE', '0') == '1':
                print('CP_TRACE fwd=' + _cpm._LAST_FWD_ROUTE['fwd'], flush=True)
        if use_cp:
            from .chunkpar import cp_chunk_fwd, cp_resource_refused, cp_resource_refuse
            import triton as _tr
            _cpk = (str(x_raw.device), precision, str(Ctil.dtype), str(x_raw.dtype), T, N, G_SIZE, _env('CP_FWD_GT', '0'), _env('CP_KSUB_FWD', ''), _env('CP_SHARED_CB', '0'))
            if cp_resource_refused('fwd', _cpk):
                use_cp = False
            else:
                try:
                    Y, S_all, S_final, siso_aux = cp_chunk_fwd(Ctil, Btil, x_raw, delta_cumsum, a_log, beta=beta, gamma=gamma, dw=dw, trap_lam=trap_lam, x0=x0, d_skip=d_skip, precision=precision, store_sall=needs_bwd and (not recompute), needs_bwd=needs_bwd, y_token_major=siso_ytm() or (y_token_major and (not has_gate or out_norm_w is not None)))
                    if has_trap:
                        dot_out = siso_aux['dot']
                except _tr.runtime.errors.OutOfResources as _oor:
                    cp_resource_refuse('fwd', _cpk, _oor)
                    use_cp = False
            if not use_cp:
                _cpm._LAST_FWD_ROUTE['fwd'] = 'serial (chunkpar refused: out of shared memory)'
                if _env('CP_TRACE', '0') == '1':
                    print('CP_TRACE fwd=' + _cpm._LAST_FWD_ROUTE['fwd'], flush=True)
        if use_cp:
            pass
        elif use_siso:
            from .siso_fwd_kernel import siso_chunk_fwd
            Y, S_all, S_final, siso_aux = siso_chunk_fwd(siso_bc(Ctil), siso_bc(Btil), x_raw, delta_cumsum, a_log, theta=theta, theta_h=theta_h, rot_bias=rot_bias, beta=beta, gamma=gamma, bias_pre_rot=bias_pre_rot, dw=dw, trap_lam=trap_lam, x0=x0, d_skip=d_skip, precision=precision, store_sall=needs_bwd and (not recompute), needs_bwd=needs_bwd, y_token_major=siso_ytm() or (y_token_major and (not has_gate or out_norm_w is not None)), fuse=fuse if fuse in ('b', 'c') else None)
            if has_trap:
                dot_out = siso_aux['dot_raw'] if fuse == 'b' else siso_aux['dot']
            if fuse in ('b', 'c'):
                siso_aux = None
        else:
            Y, S_all, S_final = fused_chunk_fwd(Ctil, Btil, x_raw, delta_cumsum, a_log, theta_h=theta_h, theta=theta, beta=beta, gamma=gamma, rot_bias=rot_bias, bias_pre_rot=bias_pre_rot, x0=x0, precision=precision, bc_grouped=bc_grouped, Gb=Gb, Gc=Gc, dw=dw, store_sall=needs_bwd and (not recompute), gate_z=gate_z if in_kernel else None, d_skip=d_skip, ypre_out=ypre, gate_out_dtype=gate_dt, trap_lam=trap_lam, dot_out=dot_out, theta_off=theta_off, pair_half=pair_half, dcs_ink=dcs_ink, y_token_major=y_token_major and (in_kernel or not has_gate or out_norm_w is not None))
        ctx.recompute_sall = recompute
        ctx.sall_cp = False
        ctx.sall_rebuild_dcs = bool(recompute and use_siso and (theta is None))
        if recompute and use_siso:
            from .chunkpar import cp_replay_ok
            ctx.sall_cp = bool(use_cp and cp_replay_ok(delta_cumsum.shape[2]))
        ctx.siso = use_siso
        ctx.siso_why = siso_why
        siso_aux_keep = siso_aux if needs_bwd else None
        ctx.siso_prec = precision.split(':')[1] if ':' in precision else precision
        ctx.siso_fuse = fuse
        ctx.siso_bwd = bool(use_siso and theta is not None and (fuse not in ('b', 'c')))
        rstd = None
        if in_kernel:
            Yg = Y
            Y = ypre
        elif has_gate:
            if out_norm_w is not None:
                from .gate_epilogue import gate_norm_fwd
                Yg, rstd = gate_norm_fwd(Y, gate_z, T, out_norm_w, out_norm_eps, out_dtype=x_raw.dtype if _gate_out_io() else None)
                from ...layers import regen as _regen
                if _regen.wants('out', theta is None and theta_h is None and (rot_bias is None) and (x_raw.shape[3] == 1) and (Y.shape[4] == 1)) and needs_bwd:
                    _od = x_raw.dtype if _gate_out_io() else None
                    _Y, _gz, _T, _w, _e = (Y, gate_z, T, out_norm_w, out_norm_eps)

                    def _out_regen():
                        from .gate_epilogue import gate_norm_fwd as _gnf
                        return _gnf(_Y, _gz, _T, _w, _e, out_dtype=_od)[0]
                    _regen.register(Yg, _out_regen, track=False)
            elif _env('GATE_FUSED', '1') == '1':
                from .gate_epilogue import ascalar_epilogue_fwd
                Yg = ascalar_epilogue_fwd(Y, gate_z, T, token_major=y_token_major, out_dtype=x_raw.dtype if _gate_out_io() else None)
            else:
                Bsz, _H, nc, _T, O, _G = Y.shape
                Z_5d = gate_z.view(Bsz, nc, _T, _H, _G).permute(0, 3, 1, 2, 4).unsqueeze(4).float()
                Yg = Y * (Z_5d * torch.sigmoid(Z_5d))
        else:
            Yg = Y
        saved = [Ctil, Btil, x_raw, dw, delta_raw, a_log, None if recompute else S_all]
        if has_gate:
            saved.extend([Y if in_kernel or _env('GATE_FUSED', '1') == '1' else Y.contiguous(), gate_z])
        if has_dskip:
            saved.append(d_skip)
        if has_trap:
            saved.extend([trap_lam, dot_out if dot_out is not None else trap_lam])
        ctx.has_dskip = has_dskip
        ctx.in_kernel_gate = in_kernel
        ctx.out_norm_eps = out_norm_eps
        ctx.need_dnormw = bool(out_norm_w is not None and out_norm_w.requires_grad)
        ctx.has_trap = has_trap
        ctx.need_dlam = bool(has_trap and trap_lam.requires_grad)
        _theta_bf16_ok = theta is not None and (not ctx.siso_bwd) and (precision != 'ieee') and (_env('THETA_BWD_FP32', '1') != '1') and (theta.dtype == torch.float32) and (x_raw.dtype in (torch.bfloat16, torch.float16))
        if _theta_bf16_ok:
            theta_keep = theta_lo if theta_lo is not None and theta_lo.dtype == torch.bfloat16 and (theta_lo.shape == theta.shape) else theta.to(torch.bfloat16)
        else:
            theta_keep = theta
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
        ctx.need_dthh = theta_h is not None and theta_h.requires_grad
        ctx.need_dtheta = theta_wants_grad
        ctx.need_dtheta_off = theta_off is not None and theta_off.requires_grad
        ctx.need_dbeta = beta is not None and beta.requires_grad
        ctx.need_dgamma = gamma is not None and gamma.requires_grad
        ctx.need_drot_bias = rot_bias is not None and rot_bias.requires_grad
        if ctx.rot_split:
            ctx.need_dthh = ctx.rs_need_dthh
            ctx.need_dbeta = ctx.rs_need_dbeta
            ctx.need_dgamma = ctx.rs_need_dgamma
        ctx.need_dx0 = x0 is not None and x0.requires_grad
        ctx.need_ddw = dw.requires_grad
        ctx._kernel_options = _ko.current('ssd_mimo')
        ctx._measured_plan = _measured_plan.current()
        _save_with_ctx_tensors(ctx, saved, dict(theta=theta_keep, theta_h=theta_h, theta_off=theta_off, beta=beta, gamma=gamma, rot_bias=rot_bias, x0=x0, out_norm_w=out_norm_w, out_norm_rstd=rstd if out_norm_w is not None else None, rs_theta=rs_theta, rs_theta_h=rs_theta_h), siso_aux=siso_aux_keep)
        return (Yg, S_final)

    @staticmethod
    @_measured_plan.backward_scope
    @_ko.scoped_ctx('ssd_mimo')
    def backward(ctx, dY, dS_final):
        ctx = saved_view(ctx)
        sv = list(ctx.saved_tensors)
        Ctil, Btil, x_raw, dw, delta_cumsum, a_log, S_all = sv[:7]
        _nig = getattr(ctx, 'needs_input_grad', None)
        if _nig is None or len(_nig) < 16:
            _nig = (True,) * 16
        _needs = dict(ddelta=bool(_nig[4] or _nig[5]), dtheta=bool(ctx.need_dtheta and (_nig[7] or _nig[8])), dthh=bool(_nig[6]), dbeta=bool(_nig[11]), dgamma=bool(_nig[12]), dx0=bool(_nig[13]), drb=bool(_nig[14]), ddw=bool(_nig[3]), dlam=bool(_nig[15]))
        need_ddsk = bool(_nig[10])
        delta_raw0 = delta_cumsum
        if not getattr(ctx, 'dcs_ink', False):
            from .fused_fwd_kernel import seg_prefix_host
            delta_cumsum = seg_prefix_host(delta_raw0, a_log)
        idx_t = 7 + (2 if ctx.has_gate else 0) + (1 if ctx.has_dskip else 0)
        trap_lam = trap_dot = trap_nxt = trap_nd = trap_dsdot = None
        dw_eff = dw
        fuse_trap = False
        siso_bwd = bool(getattr(ctx, 'siso_bwd', False))
        if ctx.has_trap and (not siso_bwd):
            trap_lam, trap_dot = (sv[idx_t], sv[idx_t + 1])
            fuse_trap = True
            with torch.no_grad():
                trap_nxt, dw_eff, trap_nd, trap_dsdot = trap_prologue(dw, trap_lam, trap_dot)
        elif ctx.has_trap:
            trap_lam, trap_dot = (sv[idx_t], sv[idx_t + 1])
        T_fwd = delta_cumsum.shape[-1]
        _cb_asked = int(getattr(ctx, 'chunk_bwd', 0) or 0)
        T_bwd, cb_reason = (T_fwd, '')
        dY_cb = None
        _bcta = 3 if ctx.bc_grouped else 2
        if _cb_asked and _cb_asked != T_fwd:
            T_bwd = _cb_asked
            _dcs_rebuild = not getattr(ctx, 'dcs_ink', False) and getattr(ctx, 'sall_rebuild_dcs', False)
            _d_re = None
            if _dcs_rebuild:
                _d_raw_T = _rechunk(delta_raw0, 2, 3, T_bwd, T_fwd)
                if _d_raw_T is not None:
                    from .fused_fwd_kernel import seg_prefix_host
                    _d_re = seg_prefix_host(_d_raw_T, a_log)
            _cands = dict(delta_cumsum=_d_re if _dcs_rebuild else _rechunk(delta_cumsum, 2, 3, T_bwd, T_fwd), Btil=_rechunk(Btil, 1, _bcta, T_bwd, T_fwd), Ctil=_rechunk(Ctil, 1, _bcta, T_bwd, T_fwd), dY=_rechunk(dY, 2, 3, T_bwd, T_fwd))
            if siso_bwd:
                cb_reason = 'siso_bwd'
            elif not ctx.recompute_sall:
                cb_reason = 'SALL=io (S_all is a T_fwd artefact)'
            elif not (getattr(ctx, 'dcs_ink', False) or getattr(ctx, 'sall_rebuild_dcs', False)):
                cb_reason = 'DCS_INK off (delta_cumsum is a T_fwd scan)'
            elif ctx.theta is not None or ctx.rot_bias is not None:
                cb_reason = 'rotation (theta / rot_bias are chunk-shaped)'
            elif getattr(ctx, 'pair_half', False):
                cb_reason = 'pair_half'
            elif getattr(ctx, 'rot_split', False):
                cb_reason = 'rot_split'
            else:
                _bad = [k for k, v in _cands.items() if v is None]
                if _bad:
                    cb_reason = 'not a token regrouping: ' + ','.join(_bad)
            if cb_reason:
                T_bwd = T_fwd
            else:
                delta_cumsum = _cands['delta_cumsum']
                Btil, Ctil = (_cands['Btil'], _cands['Ctil'])
                dY_cb = _cands['dY']
        _cb0 = (Btil, Ctil, delta_cumsum, dw_eff, x_raw, a_log)
        _note_chunk_bwd(dict(T_fwd=T_fwd, T_bwd=T_bwd, asked=_cb_asked, refused=cb_reason))

        def _replay_sall(C_, B_, dcs_):
            _, sa, _ = fused_chunk_fwd(C_, B_, x_raw, dcs_, a_log, theta_h=ctx.theta_h, theta=ctx.theta, beta=ctx.beta, gamma=ctx.gamma, rot_bias=ctx.rot_bias, bias_pre_rot=ctx.bias_pre_rot, x0=ctx.x0, precision=ctx.precision, bc_grouped=ctx.bc_grouped, Gb=ctx.Gb, Gc=ctx.Gc, dw=dw_eff, store_sall=True, state_only=True, pair_half=ctx.pair_half, dcs_ink=getattr(ctx, 'dcs_ink', False), theta_off=ctx.theta_off)
            return sa
        _cp_done = False
        if ctx.recompute_sall and getattr(ctx, 'sall_cp', False):
            from .chunkpar import cp_chunk_fwd as _cpf, cp_replay_ok as _cpok
            from .siso_route import siso_bc as _sbc
            import triton as _tr
            if _cpok(delta_cumsum.shape[2]):
                try:
                    _, S_all, _, _ = _cpf(Ctil, _sbc(Btil), x_raw, delta_cumsum, a_log, beta=ctx.beta, gamma=None, dw=dw, trap_lam=trap_lam, x0=ctx.x0, precision=ctx.precision, store_sall=True, needs_bwd=False, states_only=True)
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
        d_skip = sv[9] if ctx.has_dskip else None
        O_bwd = Ctil.shape[-1] if Ctil.dim() == (6 if ctx.bc_grouped else 5) else 1
        dskip_mode = _dskip_in_kc() if ctx.has_dskip else 'off'
        if not (O_bwd == 1 and dw is not None):
            dskip_mode = 'off'
        if ctx.out_norm_w is not None and ctx.has_dskip and (O_bwd == 1) and (dw is not None):
            dskip_mode = 'all'
        dskip_kc = dskip_mode != 'off'
        dNormW = None
        if ctx.has_gate and ctx.out_norm_w is not None:
            from .gate_epilogue import gate_norm_bwd
            Y_pre, gate_z_saved = (sv[7], sv[8])
            dY, dZ, dNormW = gate_norm_bwd(Y_pre, gate_z_saved, ctx.out_norm_w, ctx.out_norm_rstd, dY, ctx.T, dZ_out=getattr(ctx, 'dz_out', None), real=_real_r1o1_call(ctx, x_raw, Y_pre))
            if not ctx.need_dnormw:
                dNormW = None
            elif dNormW is not None:
                dNormW = dNormW.to(ctx.out_norm_w.dtype)
        elif ctx.has_gate:
            Y_pre, gate_z_saved = (sv[7], sv[8])
            from .gate_epilogue import ascalar_epilogue_bwd
            dY_pre, dZ, dX_skip, dDsk = ascalar_epilogue_bwd(Y_pre, gate_z_saved, None if dskip_mode == 'all' or (dskip_mode == 'dx' and (not need_ddsk)) else x_raw if ctx.has_dskip else None, None if dskip_mode == 'all' or (dskip_mode == 'dx' and (not need_ddsk)) else d_skip, dY if ctx.in_kernel_gate or _env('GATE_FUSED', '1') == '1' else dY.contiguous(), ctx.T, dZ_out=getattr(ctx, 'dz_out', None), store_dxr=dskip_mode != 'dx', real=_real_r1o1_call(ctx, x_raw, Y_pre))
            dY = dY_pre
            if dDsk is not None:
                dDsk = dDsk.view(d_skip.shape).to(d_skip.dtype)
        prec = ctx.precision
        if ':' in prec:
            prec = prec.split(':')[1]
        if siso_bwd:
            from .siso_route import siso_backward
            bwd_result = siso_backward(ctx, Ctil, Btil, x_raw, dw, delta_cumsum, a_log, S_all, dY, dS_final, trap_lam, d_skip if dskip_kc else None, dskip_mode == 'all', needs=_needs)
        else:
            _dY_pre_cb = dY
            for _cb_try in range(6):
                if T_bwd != T_fwd:
                    _dyc = _rechunk(dY, 2, 3, T_bwd, T_fwd)
                    dY = _dyc if _dyc is not None else _rechunk(_dY_pre_cb.contiguous(), 2, 3, T_bwd, T_fwd)
                try:
                    bwd_result = fused_chunk_bwd_5kernel(Ctil, Btil, x_raw, delta_cumsum, a_log, S_all, dY, theta_h=ctx.theta_h, theta=ctx.theta, dS_final=dS_final, precision=prec, Gb=ctx.Gb, Gc=ctx.Gc, bc_grouped=ctx.bc_grouped, beta=ctx.beta, gamma=ctx.gamma, x0=ctx.x0, rot_bias=ctx.rot_bias, bias_pre_rot=ctx.bias_pre_rot, dw=dw_eff, trap_nd=trap_nd if ctx.has_trap else None, theta_off=ctx.theta_off, trap_dsdot=trap_dsdot, trap_lam=trap_lam, trap_dot=trap_dot, trap_nxt=trap_nxt, trap_dt=dw if fuse_trap else None, need_dlam=ctx.need_dlam, fuse_trap=fuse_trap, phase_internal=ctx.phase_internal, d_skip=d_skip if dskip_kc else None, pair_half=ctx.pair_half, dcs_ink=getattr(ctx, 'dcs_ink', False), dskip_dd=dskip_mode == 'all')
                except _OutOfResources:
                    from .calibration_fused import smem_demote as _smem_demote
                    if _smem_demote(T_bwd):
                        continue
                    _half = T_bwd // 2
                    _why = ''
                    if _cb_try == 5 or _half < 16:
                        _why = 'T_bwd floor (16)'
                    elif not ctx.recompute_sall:
                        _why = 'SALL=io (S_all is a T_fwd artefact)'
                    elif not getattr(ctx, 'dcs_ink', False):
                        _why = 'DCS_INK off (delta_cumsum is a T_fwd scan)'
                    elif ctx.theta is not None or ctx.rot_bias is not None:
                        _why = 'rotation (theta / rot_bias are chunk-shaped)'
                    elif getattr(ctx, 'pair_half', False):
                        _why = 'pair_half'
                    elif getattr(ctx, 'rot_split', False):
                        _why = 'rot_split'
                    if not _why:
                        _B0, _C0, _D0 = (_cb0[0], _cb0[1], _cb0[2])
                        _nb = _rechunk(_B0, 1, _bcta, _half, T_fwd)
                        _nc2 = _rechunk(_C0, 1, _bcta, _half, T_fwd)
                        _nd = _rechunk(_D0, 2, 3, _half, T_fwd)
                        if _nb is None or _nc2 is None or _nd is None:
                            _why = 'not a token regrouping'
                        else:
                            Btil, Ctil, delta_cumsum = (_nb, _nc2, _nd)
                            T_bwd = _half
                            dY = _dY_pre_cb
                            if ctx.recompute_sall:
                                S_all = _replay_sall(Ctil, Btil, delta_cumsum)
                            _note_chunk_bwd(dict(T_fwd=T_fwd, T_bwd=T_bwd, asked=_cb_asked, refused='', smem_fallback=True))
                            warnings.warn("SSD-MIMO: the backward's score-path kernel does not fit shared memory at chunk %d; re-running the BACKWARD at chunk %d (REALMODE3 lever 3). This changes only the chunkwise associativity of the scan, not its math." % (T_fwd if _cb_try == 0 else T_bwd * 2, T_bwd), RuntimeWarning, stacklevel=2)
                            continue
                    _note_chunk_bwd(dict(T_fwd=T_fwd, T_bwd=T_bwd, asked=_cb_asked, refused=_why, smem_fallback=False))
                    _cb_refuse(_why, T_bwd)
                break
        dX, dCtil, dBtil, dDelta, dThetaH, dTheta, dBeta, dGamma, dX0 = bwd_result[:9]
        dRotBias = bwd_result[9]
        dDW = bwd_result[10]
        dSdot = bwd_result[11]
        dThOff = bwd_result[12]
        dLam = bwd_result[13]
        if getattr(ctx, 'rot_split', False):
            from .m3_epilogue import rot_adjoint
            dCtil, dBtil, dBeta, dGamma, dTheta, dThetaH = rot_adjoint(Ctil, Btil, dCtil, dBtil, ctx.rs_theta, ctx.rs_theta_h.float().contiguous() if ctx.rs_theta_h is not None else None, ctx.H_rs, Gb=ctx.rs_Gb, Gc=ctx.rs_Gc, bc_grouped=ctx.rs_bc_grouped, B_shape=ctx.rs_bshape, B_stride=ctx.rs_bstride, C_shape=ctx.rs_cshape, C_stride=ctx.rs_cstride, need_dtheta=ctx.need_dtheta, need_dthh=ctx.need_dthh, need_dbeta=ctx.need_dbeta, need_dgamma=ctx.need_dgamma, R_out=ctx.rs_bshape[-1] if len(ctx.rs_bshape) == (6 if ctx.rs_bc_grouped else 5) else 1, O_out=ctx.rs_cshape[-1] if len(ctx.rs_cshape) == (6 if ctx.rs_bc_grouped else 5) else 1, pair_half=ctx.rs_pair_half)
            if dBeta is not None:
                dBeta = dBeta.view(ctx.rs_beta_shape)
            if dGamma is not None:
                dGamma = dGamma.view(ctx.rs_gamma_shape)
            if dThetaH is not None:
                dThetaH = dThetaH.view(ctx.rs_thh_shape)
        if ctx.phase_prefix and dTheta is not None and ctx.need_dtheta:
            from .m3_epilogue import phase_prefix_bwd
            phase_prefix_bwd(dTheta, None, cross=True)
        if dskip_mode == 'all':
            dDsk = bwd_result[14].view(d_skip.shape).to(d_skip.dtype)
        da_log = None
        _c_dt = ctx.rs_cdtype if getattr(ctx, 'rot_split', False) else Ctil.dtype
        _b_dt = ctx.rs_bdtype if getattr(ctx, 'rot_split', False) else Btil.dtype
        if T_bwd != T_fwd:
            _bcta = 3 if ctx.bc_grouped else 2
            _dbv = _rechunk(dBtil, 1, _bcta, T_fwd, T_bwd)
            _dcv = _rechunk(dCtil, 1, _bcta, T_fwd, T_bwd)
            if _dbv is None or _dcv is None:
                raise RuntimeError('chunk_size_bwd: dB / dC came back in a layout that is not a token regrouping; with the group-major / R-major B_c this can only mean a new writer allocated them contiguous at T_bwd')
            dBtil, dCtil = (_dbv, _dcv)
            if dDelta is not None:
                dDelta = _rechunk(dDelta.contiguous(), 2, 3, T_fwd, T_bwd)
        if dDelta is not None:
            da_log = (delta_raw0 * dDelta.reshape(delta_raw0.shape)).sum(dim=tuple((i for i in range(delta_raw0.dim()) if i != 1)))
        dCtil = dCtil.to(_c_dt)
        dBtil = dBtil.to(_b_dt)
        if dX_skip is not None:
            dX = dX + dX_skip.to(dX.dtype)
        return (dCtil, dBtil, dX, dDW if ctx.need_ddw else None, dDelta, da_log, dThetaH if ctx.need_dthh else None, dTheta if ctx.need_dtheta else None, dThOff if ctx.need_dtheta_off else None, dZ, dDsk, dBeta if ctx.need_dbeta else None, dGamma if ctx.need_dgamma else None, dX0 if ctx.need_dx0 else None, dRotBias if ctx.need_drot_bias else None, dLam if ctx.need_dlam else None, None, None, None, None, None, None, None, None, None, None, None, None, dNormW, None, None, None, None, None)
A_LOG_MAX = 88.0
VALIDATE_ENV = 'SSD_MIMO_VALIDATE'

def _validate_default() -> bool:
    return _os.environ.get(VALIDATE_ENV, '0').lower() not in ('0', '', 'false', 'no')

def check_a_log_range(a_log, who: str='ssd_mimo_scan') -> None:
    if a_log is None:
        return
    m = a_log.detach().abs().max().item()
    if not m <= A_LOG_MAX:
        raise ValueError(f'{who}: A_log has magnitude {m:.3g} > {A_LOG_MAX:g}; the rate a = -exp(A_log) must be finite in fp32 (stability-by-construction invariant, implementation.md implementation.9).  A decay this fast is total forgetting within one token; clamp or re-parameterise A_log at the module, the kernel refuses it.')
_DEPRECATION_WARNED = False
_N_TILE_MAX = 128
_N_TILE_MIN = 16

def _n_tile_split(N: int, max_tile: int=_N_TILE_MAX) -> list[int]:
    if N < 1:
        raise ValueError(f'SSD kernel: N must be >= 1, got N={N}')
    cap = min(int(max_tile), _N_TILE_MAX)
    if cap < _N_TILE_MIN or cap & cap - 1:
        raise ValueError(f'SSD kernel: the N-tile cap must be a power of two in [{_N_TILE_MIN}, {_N_TILE_MAX}], got {max_tile}')
    tiles, rem = ([], int(N))
    while rem >= _N_TILE_MIN:
        tiles.append(min(cap, 1 << rem.bit_length() - 1))
        rem -= tiles[-1]
    if rem:
        tiles.append(rem)
    return tiles

def _n_split_caps(widest: int) -> list[int]:
    return [c for c in (64, 32, 16) if c < widest]

def _slice_or_none(t, dim, start, length):
    return None if t is None else t.narrow(dim, start, length)

def _pad_dim(t, dim: int, amount: int):
    if t is None or not amount:
        return t
    pad = [0, 0] * (t.dim() - dim - 1) + [0, amount]
    return F.pad(t, tuple(pad))

def _tile_inputs(ten: dict, off: int, width: int, n_axis: int, N: int, keep_gate: bool) -> tuple[dict, int]:
    pad = _N_TILE_MIN - width if width < _N_TILE_MIN else 0
    sub = dict(ten)
    if not keep_gate:
        sub['gate_z'] = None
        sub['d_skip'] = None
    if off == 0 and width == N and (not pad):
        return (sub, 0)
    rot = [ten[k] for k in ('theta', 'theta_h', 'rot_bias')]
    if (width % 2 or off % 2) and any((t is not None for t in rot)):
        raise ValueError(f'SSD kernel: a rotation phase (theta / theta_h / rot_bias) needs an EVEN state tile — it is indexed by the coordinate pairs (2k, 2k+1) — but this call lands on tile [{off}, {off + width}) of N={N}')
    h0, h1 = (off // 2, (off + width) // 2)
    sub['theta_lo'] = None
    sub['B_tok'] = _pad_dim(ten['B_tok'].narrow(n_axis, off, width), n_axis, pad)
    sub['C_tok'] = _pad_dim(ten['C_tok'].narrow(n_axis, off, width), n_axis, pad)
    sub['beta'] = _pad_dim(_slice_or_none(ten['beta'], 1, off, width), 1, pad)
    sub['gamma'] = _pad_dim(_slice_or_none(ten['gamma'], 1, off, width), 1, pad)
    sub['x0'] = _pad_dim(_slice_or_none(ten['x0'], 3, off, width), 3, pad)
    if ten['theta'] is not None:
        th_ax = ten['theta'].dim() - 1
        sub['theta'] = _pad_dim(_slice_or_none(ten['theta'], th_ax, h0, h1 - h0), th_ax, pad // 2)
    sub['theta_h'] = _pad_dim(_slice_or_none(ten['theta_h'], 1, h0, h1 - h0), 1, pad // 2)
    sub['rot_bias'] = _pad_dim(_slice_or_none(ten['rot_bias'], 1, h0, h1 - h0), 1, pad // 2)
    return (sub, pad)

def _tile_launch(ten: dict, off: int, width: int, chunk: int, n_axis: int, N: int, precision: str, bc_per_head: bool, keep_gate: bool, rot_operand_dtype=None, rot_where=None, bias_pre_rot=False, theta_increment=False, y_token_major=False, out_norm_w=None, out_norm_eps=1e-05, pair_half=False):
    if width == 128 and chunk >= 128 and _needs_bwd(ten) and (ten['x_in'].shape[3] == 1) and (ten['C_tok'].dim() == n_axis + 1 or (ten['C_tok'].dim() == n_axis + 2 and ten['C_tok'].shape[-1] == 1)) and (not (mma_io_enabled(precision) and ten['x_in'].dtype in (torch.float16, torch.bfloat16))) and smem_clamp_enabled():
        required = 2 * width * chunk * 4
        if chunk == 128 and ten['x_in'].shape[2] == 32 and (precision == 'ieee') and (torch.cuda.get_device_capability(ten['x_in'].device) == (8, 0)):
            required = max(required, 180224)
        limit = device_smem_limit(ten['x_in'].device)
        if required > limit:
            raise SsdSmemOverflow(required, limit, width, chunk)
    sub, pad = _tile_inputs(ten, off, width, n_axis, N, keep_gate)
    y, state = _fused_ssd_prepared(sub['x_in'], sub['delta'], sub['B_tok'], sub['C_tok'], sub['a_log'], Q=None, chunk_size=chunk, beta=sub['beta'], gamma=sub['gamma'], theta=sub['theta'], theta_h=sub['theta_h'], rot_bias=sub['rot_bias'], theta_lo=sub.get('theta_lo'), delta_w=sub['delta_w'], gate_z=sub['gate_z'], d_skip=sub['d_skip'], trap_lam=sub['trap_lam'], x0=sub['x0'], precision=precision, per_block_decay=False, stable_score=True, bc_per_head=bc_per_head, rot_operand_dtype=rot_operand_dtype, rot_where=rot_where, bias_pre_rot=bias_pre_rot, theta_increment=theta_increment, y_token_major=y_token_major, out_norm_w=out_norm_w, out_norm_eps=out_norm_eps, pair_half=pair_half, _require_fused=True, _smem_ladder=False)
    if pad:
        state = state[..., :width]
    return (y, state)

def _detached(ten: dict) -> dict:
    out = {}
    for k, v in ten.items():
        out[k] = None if v is None else v.detach().requires_grad_(bool(v.requires_grad))
    return out

def _needs_bwd(ten: dict) -> bool:
    return torch.is_grad_enabled() and any((v is not None and v.requires_grad for v in ten.values()))
_SCAN_PRECISIONS = ('ieee', 'tf32', 'io', 'bf16_ops')
_BF16_OPERAND_MODE = 'bf16_ops'

@_scan_boundary
@_measured_plan.scan_scope
@_ko.scoped_kwarg('ssd_mimo')
def ssd_mimo_scan(x_in: torch.Tensor, delta: torch.Tensor, B_tok: torch.Tensor, C_tok: torch.Tensor, a_log: torch.Tensor, *, chunk_size: int=64, chunk_mode: str='explicit', chunk_size_bwd: Optional[int]=None, kernel_options: Optional[dict]=None, beta=None, gamma=None, theta=None, theta_h=None, rot_bias=None, theta_lo=None, bias_pre_rot: bool=True, delta_w=None, gate_z=None, d_skip=None, trap_lam=None, x0=None, precision: str='tf32', bc_per_head: bool=False, rot_operand_dtype: Optional[str]=None, rot_where: Optional[str]=None, theta_increment: bool=False, y_token_major: bool=False, out_norm_w=None, out_norm_eps: float=1e-05, pair_layout: str='interleaved', write_prev=None, conv_state=None) -> tuple[torch.Tensor, torch.Tensor]:
    if precision not in _SCAN_PRECISIONS:
        raise ValueError(f'ssd_mimo_scan: precision must be one of {_SCAN_PRECISIONS}, got {precision!r}')
    _out_dtype = None
    if precision == _BF16_OPERAND_MODE:
        _out_dtype = x_in.dtype
        _lo = lambda t: t.to(torch.bfloat16) if isinstance(t, torch.Tensor) and t.is_floating_point() and (t.dtype in (torch.float32, torch.float64)) else t
        x_in, B_tok, C_tok, gate_z = (_lo(x_in), _lo(B_tok), _lo(C_tok), _lo(gate_z))
        precision = 'io'
    if write_prev is not None:
        raise NotImplementedError("ssd_mimo_scan: 'write_prev' (the trapezoid's previous unscaled write) is NOT a scan argument and has no scan adjoint.  The collapsed one-tap recurrence runs on g_t = S_t + dt_{t+1}(1-lam_{t+1}) W_t, so the carry folds into x0 as x0 += dt_0 (1-lam_0) W_{-1} on the host (SSDMimoSSM._prepare_kernel_inputs); its gradient then arrives as part of dX0.  Fold it there, do not pass it here.")
    if conv_state is not None:
        raise NotImplementedError("ssd_mimo_scan: 'conv_state' (the B/C convolution window) is NOT a scan argument and has no scan adjoint.  The causal convolution is applied to the token maps BEFORE the scan, so its history belongs to the convolution op module-side (SSDMimoSSM._bc_source(prefix=...)); the scan only ever sees the already-convolved B/C.")
    if bias_pre_rot and theta is None and (theta_h is None) and (rot_bias is None):
        bias_pre_rot = False
    if bias_pre_rot and beta is None and (gamma is None):
        bias_pre_rot = False
    if pair_layout not in ('interleaved', 'half'):
        raise ValueError(f"pair_layout must be 'interleaved' or 'half', got {pair_layout!r}")
    pair_half = pair_layout == 'half' and (theta is not None or theta_h is not None)
    if pair_half:
        _N = B_tok.shape[3 if bc_per_head else 2]
        if _N % 32:
            raise NotImplementedError(f"pair_layout='half' needs N a multiple of 32 (the MMA K floor on the half tile); got N={_N}")
        if rot_bias is not None:
            raise NotImplementedError("pair_layout='half' does not serve the retired rot_bias")
    if _validate_default() and (not torch.compiler.is_compiling()):
        check_a_log_range(a_log)
    if precision != 'ieee':
        x_in = autocast_io(x_in)
        gate_z = autocast_io(gate_z)
        if precision == 'io':
            B_tok = autocast_io(B_tok)
            C_tok = autocast_io(C_tok)
    G = x_in.shape[2]
    g_pad = -G % 16
    if g_pad:
        heads = x_in.shape[1]
        x_in = F.pad(x_in, (0, 0, 0, 0, 0, g_pad))
        if x0 is not None:
            x0 = F.pad(x0, (0, 0, 0, g_pad))
        if gate_z is not None:
            bsz, length = (gate_z.shape[0], gate_z.shape[1])
            gate_z = F.pad(gate_z.view(bsz, length, heads, G), (0, g_pad)).reshape(bsz, length, heads * (G + g_pad))
        if d_skip is not None:
            d_skip = F.pad(d_skip, (0, 0, 0, g_pad))
    ten = dict(x_in=x_in, delta=delta, B_tok=B_tok, C_tok=C_tok, a_log=a_log, beta=beta, gamma=gamma, theta=theta, theta_h=theta_h, rot_bias=rot_bias, delta_w=delta_w, gate_z=gate_z, d_skip=d_skip, trap_lam=trap_lam, x0=x0, theta_lo=theta_lo)
    if y_token_major and g_pad:
        raise NotImplementedError('y_token_major with G not a multiple of 16: the pad trim would copy; pad G module-side (SSDMimoSSM.G_kernel does)')
    with _chunk_bwd_request(chunk_size_bwd):
        (y, s_final), record = _ssd_dispatch(ten, chunk_size, precision, bc_per_head, rot_operand_dtype, rot_where, bias_pre_rot, theta_increment, y_token_major=y_token_major, pair_half=pair_half, out_norm_w=out_norm_w, out_norm_eps=out_norm_eps, chunk_mode=chunk_mode)
    record['measured_plan'] = _measured_plan.current()
    record['g_pad'] = g_pad
    _note_dispatch(record)
    if g_pad:
        y = y[..., :G, :]
        s_final = s_final[..., :G, :]
    if _out_dtype is not None and y.dtype != _out_dtype:
        y = y.to(_out_dtype)
    return (y, s_final)
_DISPATCH_LOG: list = []
_DISPATCH_LOG_MAX = 32

def dispatch_log() -> list:
    return list(_DISPATCH_LOG)

def _note_dispatch(record: dict) -> None:
    if torch.compiler.is_compiling():
        return
    _DISPATCH_LOG.append(record)
    del _DISPATCH_LOG[:-_DISPATCH_LOG_MAX]

def _plan_record(N, tiles, chunk_req, chunk_used, strategy, chunk_mode='explicit', guard='') -> dict:
    last = tiles[-1]
    try:
        from .fused_fwd_kernel import last_fwd_cfg
        cfg = last_fwd_cfg()
    except Exception:
        cfg = {}
    ko_val, ko_src = _ko.record('ssd_mimo')
    return dict(N=N, tiles=list(tiles), n_pad=_N_TILE_MIN - last if last < _N_TILE_MIN else 0, chunk_requested=chunk_req, chunk_used=chunk_used, launches=len(tiles), strategy=strategy, chunk_mode=chunk_mode, chunk_guard=guard, fwd_warps=cfg.get('warps'), fwd_stages=cfg.get('stages'), fwd_maxnreg=cfg.get('maxnreg'), fwd_cfg=cfg.get('source'), fwd_spill_fix=cfg.get('spill_fix'), fwd_stack_fix=cfg.get('stack_fix'), fwd_stack_guard=cfg.get('stack_guard'), kernel_options=ko_val, kernel_options_src=ko_src)

def _run_plan(ten, tiles, chunk, n_axis, N, precision, bc_per_head, rot_operand_dtype=None, rot_where=None, bias_pre_rot=False, theta_increment=False, y_token_major=False, out_norm_w=None, out_norm_eps=1e-05, pair_half=False):
    if len(tiles) == 1:
        return _tile_launch(ten, 0, tiles[0], chunk, n_axis, N, precision, bc_per_head, keep_gate=True, rot_operand_dtype=rot_operand_dtype, rot_where=rot_where, bias_pre_rot=bias_pre_rot, theta_increment=theta_increment, y_token_major=y_token_major, out_norm_w=out_norm_w, out_norm_eps=out_norm_eps, pair_half=pair_half)
    if out_norm_w is not None:
        raise NotImplementedError(f'the fused output RMSNorm needs the single-tile plan; this call resolved to {len(tiles)} N-tiles (a shared-memory repair).  Run with SSD_MIMO_OUT_NORM_FUSED=0.')
    if pair_half:
        raise NotImplementedError("pair_layout='half' does not take the N-tiled shared-memory repair (an N tile would split rotation pairs)")
    return _ssd_mimo_scan_n_tiled(ten, tiles, chunk, n_axis, N, precision, bc_per_head, rot_operand_dtype=rot_operand_dtype, rot_where=rot_where, bias_pre_rot=bias_pre_rot, theta_increment=theta_increment)

def _ssd_dispatch(ten, chunk_size, precision, bc_per_head, rot_operand_dtype=None, rot_where=None, bias_pre_rot=False, theta_increment=False, y_token_major=False, out_norm_w=None, out_norm_eps=1e-05, pair_half=False, chunk_mode='explicit'):
    n_axis = 3 if bc_per_head else 2
    N = ten['B_tok'].shape[n_axis]
    tiles = _n_tile_split(N)
    chunk_req = chunk_size
    guard = ''
    if chunk_size > 32 and ten['x_in'].is_cuda:
        if bc_per_head:
            c_per_o = ten['C_tok'].dim() == 5
            O_ax = ten['C_tok'].shape[-1] if c_per_o else 1
        else:
            _, _, O_ax = _mimo_axes(ten['B_tok'], ten['C_tok'])
        R_ax = ten['x_in'].shape[3]
        _cc = torch.cuda.get_device_capability(ten['x_in'].device)
        if rot_io_bwd_t64_fault(_cc, mma_io_enabled(precision), ten['theta_h'] is not None, R_ax, O_ax):
            guard = 'k24_rot_io_bwd_t64'
            chunk_size = 32
        elif pairhalf_r1_t64_fault(_cc, precision != 'ieee', pair_half, R_ax, O_ax):
            guard = 'sm90_pairhalf_r1_t64'
            chunk_size = 32
        elif pairhalf_io_t64_fault(_cc, mma_io_enabled(precision), pair_half):
            guard = 'sm90_pairhalf_io_t64'
            chunk_size = 32

    def run(plan, tensors):
        return _run_plan(tensors, plan[0], plan[1], n_axis, N, precision, bc_per_head, rot_operand_dtype=rot_operand_dtype, rot_where=rot_where, bias_pre_rot=bias_pre_rot, theta_increment=theta_increment, y_token_major=y_token_major, out_norm_w=out_norm_w, out_norm_eps=out_norm_eps, pair_half=pair_half)
    try:
        out = run((tiles, chunk_size), ten)
        return (out, _plan_record(N, tiles, chunk_req, chunk_size, 'native', chunk_mode=chunk_mode, guard=guard))
    except SsdSmemOverflow as exc:
        overflow = exc
    families = {T_HALVE: [(tiles, T) for T in chunk_ladder(chunk_size)[1:]], N_SPLIT: [(_n_tile_split(N, cap), chunk_size) for cap in _n_split_caps(tiles[0])]}
    available = [name for name, plans in families.items() if plans]
    if not available:
        raise overflow
    key = strategy_key(N, chunk_size, ten['x_in'].dtype, precision, 'fwd_bwd' if _needs_bwd(ten) else 'fwd', device=ten['x_in'].device)
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
                note_chunk_clamped(N, chunk_size, plan_chunk, overflow.required, overflow.limit)
            else:
                note_n_split(N, chunk_size, plan_tiles, overflow.required, overflow.limit)
            return (out, _plan_record(N, plan_tiles, chunk_req, plan_chunk, name, chunk_mode=chunk_mode, guard=guard))
    raise overflow

def _measure_strategy(key, families, available, run, ten):
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
                trials[name]()
                break
            except SsdSmemOverflow:
                trials.pop(name, None)
                continue
            except Exception:
                trials.pop(name, None)
                torch.cuda.empty_cache()
                break
    if len(trials) > 1:
        return resolve_strategy(key, trials)
    return next(iter(trials)) if trials else None

def _ssd_mimo_scan_n_tiled(ten, tiles, chunk_size, n_axis, N, precision, bc_per_head, rot_operand_dtype=None, rot_where=None, bias_pre_rot=False, theta_increment=False) -> tuple[torch.Tensor, torch.Tensor]:
    from .gate_epilogue import silu_gate
    x_in, gate_z = (ten['x_in'], ten['gate_z'])
    d_skip = ten.get('d_skip')
    y_acc = None
    states = []
    off = 0
    for n in tiles:
        y_t, s_t = _tile_launch(ten, off, n, chunk_size, n_axis, N, precision, bc_per_head, keep_gate=False, rot_operand_dtype=rot_operand_dtype, rot_where=rot_where, bias_pre_rot=bias_pre_rot, theta_increment=theta_increment)
        yf = y_t.float()
        y_acc = yf if y_acc is None else y_acc + yf
        states.append(s_t)
        off += n
    state = torch.cat(states, dim=-1)
    if d_skip is not None:
        y_acc = y_acc + torch.einsum('bhgrl,hgr->bhgl', x_in.float(), d_skip.float())
    y = y_acc.to(x_in.dtype)
    if gate_z is not None:
        H, G_SIZE = (x_in.shape[1], x_in.shape[2])
        if gate_z.shape[-1] != H * G_SIZE:
            raise ValueError(f'gate_z last dim must be H*G_SIZE={H * G_SIZE}, got {gate_z.shape[-1]}')
        Bsz, L = (gate_z.shape[0], gate_z.shape[1])
        z = gate_z.view(Bsz, L, H, G_SIZE).permute(0, 2, 3, 1)
        if y.dim() == 5:
            z = z.unsqueeze(2).expand(-1, -1, y.shape[2], -1, -1)
        y = silu_gate(y.contiguous(), z.contiguous().to(y.dtype))
    return (y, state)

def _fused_ssd_prepared(x_in: torch.Tensor, delta: torch.Tensor, B_tok: torch.Tensor, C_tok: torch.Tensor, a_log: torch.Tensor, Q: Optional[torch.Tensor], chunk_size: int=64, beta: Optional[torch.Tensor]=None, omega: Optional[torch.Tensor]=None, x0: Optional[torch.Tensor]=None, precision: str='tf32', per_block_decay: Optional[bool]=None, r_heads: Optional[torch.Tensor]=None, stable_score: Optional[bool]=None, bias_pre_rot: bool=False, bc_per_head: bool=False, gate_z: Optional[torch.Tensor]=None, d_skip: Optional[torch.Tensor]=None, trap_lam: Optional[torch.Tensor]=None, delta_w: Optional[torch.Tensor]=None, theta: Optional[torch.Tensor]=None, theta_h: Optional[torch.Tensor]=None, gamma: Optional[torch.Tensor]=None, b_per_g: bool=False, rot_bias: Optional[torch.Tensor]=None, b_bias: Optional[torch.Tensor]=None, rot_operand_dtype: Optional[str]=None, rot_where: Optional[str]=None, theta_increment: bool=False, y_token_major: bool=False, out_norm_w=None, out_norm_eps: float=1e-05, pair_half: bool=False, theta_lo=None, _require_fused: bool=False, _smem_ladder: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert x_in.is_cuda
    if not _FUSED_SSD:
        raise ValueError('FUSED=0 is retired; ssd_mimo_scan requires the fused production route')
    if Q is not None or r_heads is not None or b_per_g or (b_bias is not None) or (delta.dim() != 3):
        raise ValueError('Legacy Q/per-block/per-G/r_heads routes are retired; use the supported ssd_mimo_scan axes')
    _require_fused = True
    if omega is not None:
        raise NotImplementedError('HAS_OMEGA removed — use HAS_ROT (theta parameter)')
    gk = dict(gate_z=gate_z, d_skip=d_skip, trap_lam=trap_lam, delta_w=delta_w, theta=theta, theta_h=theta_h, gamma=gamma, b_per_g=b_per_g, b_bias=b_bias, rot_bias=rot_bias, rot_operand_dtype=rot_operand_dtype, rot_where=rot_where, bias_pre_rot=bias_pre_rot, theta_increment=theta_increment, y_token_major=y_token_major, out_norm_w=out_norm_w, out_norm_eps=out_norm_eps, pair_half=pair_half, theta_lo=theta_lo, _require_fused=_require_fused)
    if (gate_z is not None or d_skip is not None) and Q is not None:
        raise NotImplementedError('the fused gate / D-skip epilogue is wired on the A-scalar kernels (Q=None, per-head-scalar decay) only')
    per_g_any = b_per_g or b_bias is not None or delta.dim() == 4
    if per_g_any and Q is not None:
        raise NotImplementedError('per-G decay / per-G B~ / b_bias require Q=None (A-scalar)')
    if b_per_g and (not bc_per_head):
        raise ValueError('b_per_g=True requires bc_per_head=True (B_tok [B, L, Gb, G, N])')

    def _dispatch(T):
        if bc_per_head:
            H = x_in.shape[1]
            Gb = B_tok.shape[2]
            Gc = C_tok.shape[2]
            if H % Gb != 0 or H % Gc != 0:
                raise ValueError(f'bc_per_head: group counts Gb={Gb}, Gc={Gc} must each divide H={H}')
            if Gb == 1 and Gc == 1 and (not b_per_g):
                return _fused_ssd_impl(x_in, delta, B_tok[:, :, 0], C_tok[:, :, 0], a_log, Q, T, beta, omega, x0, precision, per_block_decay, r_heads, stable_score, **gk)
            return _fused_ssd_impl(x_in, delta, B_tok, C_tok, a_log, Q, T, beta, omega, x0, precision, per_block_decay, r_heads, stable_score, bc_groups=(Gb, Gc), **gk)
        return _fused_ssd_impl(x_in, delta, B_tok, C_tok, a_log, Q, T, beta, omega, x0, precision, per_block_decay, r_heads, stable_score, **gk)
    ladder = chunk_ladder(chunk_size) if _smem_ladder else [chunk_size]
    overflow = None
    with torch.amp.autocast('cuda', enabled=False):
        for i, T in enumerate(ladder):
            try:
                out = _dispatch(T)
            except SsdSmemOverflow as exc:
                overflow = overflow or exc
                if i + 1 == len(ladder):
                    raise
                continue
            if overflow is not None:
                note_chunk_clamped(overflow.N, chunk_size, T, overflow.required, overflow.limit)
            return out

def _fused_ssd_impl(x_in, delta, B_tok, C_tok, a_log, Q, chunk_size, beta, omega, x0, precision, per_block_decay, r_heads, stable_score=None, bc_groups=None, gate_z=None, d_skip=None, trap_lam=None, delta_w=None, theta=None, theta_h=None, gamma=None, b_per_g=False, b_bias=None, rot_bias=None, _require_fused=False, rot_operand_dtype=None, rot_where=None, bias_pre_rot=False, theta_increment=False, y_token_major=False, out_norm_w=None, out_norm_eps=1e-05, pair_half=False, theta_lo=None):
    if omega is not None:
        raise NotImplementedError('HAS_OMEGA removed — use HAS_ROT (theta parameter)')
    _cap = _capture_slot()
    _cap_in = None if _cap is None else dict(x_in=x_in, delta=delta, B_tok=B_tok, C_tok=C_tok, a_log=a_log, beta=beta, gamma=gamma, theta=theta, theta_h=theta_h, rot_bias=rot_bias, delta_w=delta_w, gate_z=gate_z, d_skip=d_skip, trap_lam=trap_lam, x0=x0, out_norm_w=out_norm_w)
    B, H, G_SIZE, R, L = x_in.shape
    per_g_delta = delta.dim() == 4
    per_g = per_g_delta or b_per_g or b_bias is not None
    if per_g:
        if per_g_delta and tuple(delta.shape) != (B, H, G_SIZE, L):
            raise ValueError(f'per-G delta must be [B, H, G, L]={(B, H, G_SIZE, L)}, got {tuple(delta.shape)}')
        if R != 1:
            raise NotImplementedError('per-G features require R == 1')
        if per_block_decay:
            raise NotImplementedError('per-G features live on the A-scalar route (per_block_decay=False, a_log [H])')
        for nm, t in (('beta', beta), ('gamma', gamma), ('omega', omega), ('theta', theta), ('gate_z', gate_z), ('d_skip', d_skip), ('r_heads', r_heads)):
            if t is not None:
                raise NotImplementedError(f'per-G features do not compose with {nm} yet')
        if b_per_g and B_tok.dim() != 5:
            raise ValueError(f'b_per_g: B_tok must be [B, L, Gb, G, N], got {tuple(B_tok.shape)}')
        if b_per_g and B_tok.shape[3] != G_SIZE:
            raise ValueError(f'b_per_g: B_tok G axis {B_tok.shape[3]} != G_SIZE {G_SIZE}')
        if b_bias is not None and tuple(b_bias.shape) != (H, G_SIZE, B_tok.shape[-1]):
            raise ValueError(f'b_bias must be [H, G, N], got {tuple(b_bias.shape)}')
    rot_op_io = _rot_operand_io(rot_operand_dtype)
    if theta is not None:
        _rot_ok = theta.dtype in (torch.float32, torch.float64) and _env('ROT_FOLD', 'complex') == 'complex'
        _rot_c = []

        def _fold_complex(v, nax):
            K2 = v.shape[nax] // 2
            vp = v.float().reshape(*v.shape[:nax], K2, 2)
            if not vp.is_contiguous():
                return None
            if not _rot_c:
                _rot_c.append(torch.polar(torch.ones_like(theta), -theta))
            w = _rot_c[0].reshape(theta.shape[0], theta.shape[1], *(1,) * (nax - 2), K2)
            out = torch.view_as_real(torch.view_as_complex(vp) * w).reshape(v.shape)
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
            th = theta.reshape(theta.shape[0], theta.shape[1], *(1,) * (nax - 2), K2, *(1,) * len(tail))
            c, sn = (torch.cos(th), torch.sin(th))
            a_ = vp[(slice(None),) * nax + (slice(None), 0)]
            b_ = vp[(slice(None),) * nax + (slice(None), 1)]
            out = torch.stack([c * a_ + sn * b_, -sn * a_ + c * b_], nax + 1)
            out = out.reshape(v.shape)
            if rot_op_io:
                out = out.to(v.dtype)
            return out
    bc_grouped = bc_groups is not None
    Gb, Gc = bc_groups if bc_grouped else (1, 1)
    if bc_grouped:
        b_per_r = not b_per_g and B_tok.dim() == 5
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
    prec_fwd_only = dot_precision(precision.partition(':')[0])
    a_hm = a_log.dim() == 2
    if a_hm:
        per_block_decay = True
    if Q is not None:
        M, k = (Q.shape[-3], Q.shape[-1])
    else:
        assert a_hm or per_block_decay is not None, 'Q=None requires per_block_decay set explicitly (or a_log [H, M])'
        if a_hm or per_block_decay:
            M = a_log.shape[-1]
            k = N // M
        else:
            M, k = (N, 1)
    if per_block_decay is None:
        per_block_decay = a_log.dim() == 1 and a_log.shape[0] == M
    a_scalar = not per_block_decay and (not a_hm)
    if bias_pre_rot and rot_bias is not None:
        raise NotImplementedError('ssd kernel: bias_pre_rot combined with rot_bias (the retired write-side angle offset) is refused -- rot_bias has no bias_pre_rot fold order of its own (theta={}, rot_bias={})'.format(theta is not None, rot_bias is not None))
    if bias_pre_rot and theta is None and (theta_h is not None):
        raise NotImplementedError('ssd kernel: bias_pre_rot with theta_h and no theta is a real (non-identity) READ-side rotation with no defined bias_pre_rot fold order -- pass theta too, or drop theta_h, or use bias_pre_rot=False')
    rot_kernel = False
    if theta is not None and (not bc_grouped):
        if b_per_r and B_tok.shape[-1] == 1:
            B_tok = B_tok.squeeze(-1)
            b_per_r = False
        if c_per_o and C_tok.shape[-1] == 1:
            C_tok = C_tok.squeeze(-1)
            c_per_o = False
            O = 1
    if theta is not None:
        th_shape_ok = theta.dim() == 3 and theta.shape[-1] == N // 2 or (theta.dim() == 4 and theta.shape[-1] == N // 2 and (theta.shape[2] == H))
        rot_kernel = _rot_where(rot_where) == 'kernel' and _FUSED_SSD and a_scalar and (not per_g) and (b_bias is None) and (Q is None) and (r_heads is None) and (R in (1, 2, 4, 8)) and (O in (1, 2, 4, 8)) and th_shape_ok and (not N & N - 1) and (16 <= N <= 128) and (G_SIZE % 16 == 0) and (d_skip is None or (O == 1 and gate_z is not None)) and (not (bc_grouped and R > 1) or _env('FUSED_GROUPED_R', '1') == '1')
        if theta.dim() == 4 and (not rot_kernel):
            raise NotImplementedError(f'ssd kernel: a PER-HEAD phase theta [B, L, H, N//2] is an in-kernel-only feature and this call is outside its envelope (rot_where={_rot_where(rot_where)!r}, fused={_FUSED_SSD}, a_scalar={a_scalar}, per_g={per_g}, grouped_bc={bc_grouped}, R={R}, O={O}, N={N}, G={G_SIZE}, theta{tuple(theta.shape)}, H={H}).  Pass a head-shared [B, L, N//2] phase for those routes.')
        if bias_pre_rot and (not (rot_kernel and theta.dim() == 4)):
            raise NotImplementedError(f'ssd kernel: bias_pre_rot needs the in-kernel PER-HEAD phase (rot_kernel={rot_kernel}, theta.dim()={theta.dim()}); see implementation.md implementation.4')
        if theta_increment and rot_kernel and (L <= T):
            theta = theta.cumsum(dim=1)
            theta_increment = False
            theta_lo = None
        if theta_increment and (not rot_kernel):
            theta = theta.cumsum(dim=1)
            theta_increment = False
            theta_lo = None
        if not rot_kernel:
            if pair_half:
                raise NotImplementedError("pair_layout='half' is an in-kernel-rotation feature; the host fold (`rot_where='host'`, or a geometry outside the in-kernel envelope) rotates adjacent pairs only")
            B_tok = _fold(B_tok, fold_nax)
            C_tok = _fold(C_tok, fold_nax)
    if b_per_r or c_per_o:
        _check_mimo_support(b_per_r, c_per_o, O, R, _resolve_stable(True if stable_score is None else stable_score, per_block_decay, r_heads is not None, T, R, G_SIZE, a_scalar=a_scalar, c_per_o=c_per_o), Q, r_heads, otile=True, a_scalar=a_scalar)
    if a_scalar and (b_per_r or c_per_o or bc_grouped or per_g) and (a_log.shape[0] != H):
        raise ValueError(f'per-head-scalar a_log must have shape [H]={H}, got {tuple(a_log.shape)}')
    has_beta = beta is not None
    nc = (L + T - 1) // T
    L_padded = nc * T
    device = x_in.device
    f32 = torch.float32
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
            bpad = (0, 0, 0, 0, 0, 0, 0, pad) if b_per_r or b_per_g else (0, 0, 0, 0, 0, pad)
            cpad = (0, 0, 0, 0, 0, 0, 0, pad) if c_per_o else (0, 0, 0, 0, 0, pad)
        else:
            bpad = (0, 0, 0, 0, 0, pad) if b_per_r else (0, 0, 0, pad)
            cpad = (0, 0, 0, 0, 0, pad) if c_per_o else (0, 0, 0, pad)
        B_tok_pad = F.pad(B_tok_pad, bpad)
        C_tok_pad = F.pad(C_tok_pad, cpad)
        if Q_pad is not None:
            Q_eye = torch.eye(k, device=device, dtype=f32)
            Q_eye = Q_eye.expand(B, pad, M, k, k)
            Q_pad = torch.cat([Q_pad, Q_eye], dim=1)
    if per_g_delta:
        delta_c = delta_pad.view(B, H, G_SIZE, nc, T)
    else:
        delta_c = delta_pad.view(B, H, nc, T)
    dcs_ink = _dcs_ink_on(R, O)
    delta_cumsum = delta_c
    group_major = _bc_group_major_on(bc_grouped, rotating=theta is not None, wants_chunk_bwd=_chunk_bwd_asked() != 0)
    if bc_grouped:
        if b_per_g:
            B_c = B_tok_pad.view(B, nc, T, Gb, G_SIZE, N).permute(0, 1, 3, 4, 2, 5).contiguous()
        else:
            B_c = _bc_chunk_grouped(B_tok_pad, B, nc, T, Gb, N, R if b_per_r else 0, group_major)
        C_c = _bc_chunk_grouped(C_tok_pad, B, nc, T, Gc, N, O if c_per_o else 0, group_major)
    else:
        B_c = B_tok_pad.view(B, nc, T, N, R).permute(0, 4, 1, 2, 3).contiguous().permute(0, 2, 3, 4, 1) if b_per_r else B_tok_pad.view(B, nc, T, N)
        C_c = C_tok_pad.view(B, nc, T, N, O).permute(0, 4, 1, 2, 3).contiguous().permute(0, 2, 3, 4, 1) if c_per_o else C_tok_pad.view(B, nc, T, N)
    dw_pad = delta_pad if delta_w is None else F.pad(delta_w.float(), (0, L_padded - L))
    _use_fused = _FUSED_SSD and a_scalar and (not per_g) and (b_bias is None) and (d_skip is None or (O == 1 and gate_z is not None)) and (Q_pad is None) and (r_heads is None) and (not (bc_grouped and R > 1) or _env('FUSED_GROUPED_R', '1') == '1')
    if _use_fused:
        if N & N - 1 or not 16 <= N <= 128:
            raise ValueError(f'SSD kernel: N must be a power of two in [16, 128], got N={N}')
        if G_SIZE % 16:
            if G_SIZE & G_SIZE - 1:
                raise ValueError(f'SSD kernel: G must be a multiple of 16 or a small power of two, got G={G_SIZE}')
            _use_fused = False
    if rot_kernel and (not _use_fused):
        raise AssertionError("ssd kernel: rot_where='kernel' resolved outside the fused path — the phase would be dropped; this is a bug in the rot_kernel predicate, not a user error")
    if d_skip is not None:
        if O != 1:
            raise NotImplementedError(f'the fused D-skip needs O == 1: its [H, G, R] weights carry no output axis, and this call has O={O}')
        if gate_z is None:
            raise NotImplementedError('the fused D-skip rides the gate epilogue (its adjoint reads the same y_pre tile); pass gate_z, or apply the skip module-side')
    if y_token_major and (not _use_fused):
        raise NotImplementedError('y_token_major is a fused-route option; this config resolved to the 3-dispatch fallback')
    if _require_fused and (not _use_fused):
        raise ValueError(f'ssd_mimo_scan: config is outside the fused SSD-MIMO envelope (a_scalar={a_scalar}, per_g={per_g}, b_bias={b_bias is not None}, d_skip={d_skip is not None}, Q={Q_pad is not None}, r_heads={r_heads is not None}, G={G_SIZE}, N={N}, fused_lever={_FUSED_SSD}). The generalized legacy dispatcher is retired.')
    Btil, Ctil = (B_c, C_c)
    if _use_fused:
        thh_f = theta_h.float().contiguous() if theta_h is not None else None
        th_f = None
        th_lo = None
        off_f = None
        if rot_kernel:
            th_f = theta
            th_ph = theta.dim() == 4
            if L_padded > L:
                th_f = F.pad(th_f, (0, 0, 0, 0, 0, L_padded - L) if th_ph else (0, 0, 0, L_padded - L))
            th_f = th_f.reshape(B, nc, T, H, N // 2) if th_ph else th_f.reshape(B, nc, T, N // 2)
            if theta_lo is not None and theta_lo.shape == theta.shape and (theta_lo.dtype == torch.bfloat16) and (theta_lo.device == theta.device):
                th_lo = theta_lo
                if L_padded > L:
                    th_lo = F.pad(th_lo, (0, 0, 0, 0, 0, L_padded - L) if th_ph else (0, 0, 0, L_padded - L))
                th_lo = th_lo.reshape(th_f.shape)
            if theta_increment:
                pass
        gz_c = None
        if gate_z is not None:
            if gate_z.shape[-1] != H * G_SIZE:
                raise ValueError(f'gate_z last dim must be H*G_SIZE={H * G_SIZE}, got {gate_z.shape[-1]}')
            gz_c = F.pad(gate_z, (0, 0, 0, L_padded - L)) if L_padded > L else gate_z
            if not _gate_epilogue_reads_strided():
                gz_c = gz_c.contiguous()
        beta_f = beta.float().contiguous() if has_beta else None
        gamma_f = gamma.float().contiguous() if gamma is not None else None
        x0_f = x0.float().contiguous() if x0 is not None else None
        rb_f = rot_bias.float().contiguous() if rot_bias is not None else None
        dsk_f = d_skip.float().contiguous() if d_skip is not None else None
        tl_f = None
        if trap_lam is not None:
            tl_f = trap_lam.float()
            if L_padded > L:
                tl_f = F.pad(tl_f, (0, L_padded - L))
            tl_f = tl_f.contiguous()
        needs_bwd = torch.is_grad_enabled() and any((t is not None and t.requires_grad for t in (x_pad, dw_pad, Btil, Ctil, a_log, theta_h, beta, gamma, x0, rot_bias, gate_z, th_f, off_f, d_skip, trap_lam)))
        _fn_args = (Ctil, Btil, _dense_or_contiguous(x_pad), dw_pad, delta_cumsum.contiguous(), a_log.float().contiguous(), thh_f, th_f, off_f, gz_c, dsk_f, beta_f, gamma_f, x0_f, rb_f, tl_f, H, T, N, G_SIZE, precision, Gb, Gc, bc_grouped, needs_bwd, bias_pre_rot, bool(theta_increment and rot_kernel and (off_f is None)), y_token_major, out_norm_w, out_norm_eps, bool(pair_half), bool(dcs_ink), _chunk_bwd_asked(), th_lo)
        if _cap is None:
            if _CUSTOMOP_SCAN and torch.compiler.is_compiling():
                Y, S_final = _scan_apply(*_fn_args)
            else:
                Y, S_final = _SSDMimoFn.apply(*_fn_args)
        else:
            if bc_grouped or per_g or off_f is not None:
                raise NotImplementedError(f'captured scan: only the ungrouped, per-head-scalar fused route with the in-kernel phase prologue is supported (bc_grouped={bc_grouped}, per_g={per_g}, host_theta_off={off_f is not None})')
            _sctx = _ShimCtx()
            Y, S_final = _SSDMimoFn.forward(_sctx, *_fn_args[:24], True, *_fn_args[25:])
            for _nm, _t in (('need_dthh', thh_f), ('need_dtheta', th_f), ('need_dtheta_off', off_f), ('need_dbeta', beta_f), ('need_dgamma', gamma_f), ('need_dx0', x0_f), ('need_drot_bias', rb_f), ('need_ddw', dw_pad), ('need_dlam', tl_f)):
                setattr(_sctx, _nm, _t is not None)
            _cap.append(ScanTape(_sctx, _cap_in, dict(B=B, H=H, G=G_SIZE, R=R, N=N, T=T, L=L, L_padded=L_padded, delta_w_is_none=delta_w is None, O=O, b_tail=tuple(_cap_in['B_tok'].shape[2:]), c_tail=tuple(_cap_in['C_tok'].shape[2:]))))
        if c_per_o:
            y = Y.permute(0, 1, 4, 5, 2, 3).reshape(B, H, O, G_SIZE, L_padded)[..., :L]
        else:
            y = Y.squeeze(4).permute(0, 1, 4, 2, 3).reshape(B, H, G_SIZE, L_padded)[:, :, :, :L]
        return (y.to(x_in.dtype), S_final)
    raise ValueError('ssd_mimo_scan: unsupported fused geometry')
