
from __future__ import annotations
import math
from typing import NamedTuple
import torch
import torch.nn.functional as F
from ..layers.state import State
__all__ = ('ssd_step', 'ssd_chunked', 'ssd_scan_seq', 'fold_pairs', 'token_law', 'DEFAULT_CHUNK')
DEFAULT_CHUNK = None

def compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32

class Law(NamedTuple):
    Bbar: torch.Tensor
    Cbar: torch.Tensor
    dt: torch.Tensor
    dA: torch.Tensor
    lam: torch.Tensor | None
    incr: torch.Tensor | None
    chi: torch.Tensor | None

def _no_autocast(device_type: str):
    return torch.autocast(device_type=device_type, enabled=False)

def token_slab(core, u: torch.Tensor, tok_raw: torch.Tensor | None):
    if tok_raw is not None:
        if tok_raw.shape[-1] != core.tok_width:
            raise ValueError(f"tok_raw last dim {tok_raw.shape[-1]} != tok_width={core.tok_width} (the fused token-map projection's width; use `tok_proj_weight()` to build it)")
        return tok_raw
    w, b = core.tok_proj_weight()
    return F.linear(u, w, b)

def split_slab(core, raw: torch.Tensor) -> dict:
    out = dict(zip(core._proj_keys, raw.split(core._proj_widths, dim=-1)))
    from ..layers.glue import dt_fp32_on
    if dt_fp32_on() and core.dt_proj.bias is not None:
        out['_dtb'] = core.dt_proj.bias
    return out

def bc_conv(core, bc_raw: torch.Tensor, prev: torch.Tensor | None, f):
    conv = core.bc_conv1d.conv
    d = core.bc_conv1d.d_conv
    xin = _c(bc_raw, f).transpose(1, 2)
    if prev is None:
        prev = xin.new_zeros(xin.shape[0], xin.shape[1], d)
    if xin.shape[-1] == 1:
        window = torch.cat([_c(prev[..., 1:], f), xin], dim=-1)
        y = F.conv1d(window, _c(conv.weight, f), _c(conv.bias, f), groups=conv.groups)
        return (F.silu(y).transpose(1, 2), window)
    full = torch.cat([_c(prev, f), xin], dim=-1)
    y = F.conv1d(full[..., 1:], _c(conv.weight, f), _c(conv.bias, f), groups=conv.groups)
    return (F.silu(y).transpose(1, 2), full[..., -d:])

def _c(t: torch.Tensor, f: torch.dtype) -> torch.Tensor:
    return t if t.dtype == f else t.to(f)

def _rms_last(x: torch.Tensor, norm, f) -> torch.Tensor:
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * _c(norm.weight, f)

def token_law(core, parts: dict, b_src: torch.Tensor, c_src: torch.Tensor, f: torch.dtype, lead: tuple) -> Law:
    from .ssd_mimo import _a_mag_tail_eager, _a_mag_softplus_eager
    H, R, O, N = (core.H, core.R, core.O, core.N)
    Bm = _c(b_src, f).view(*lead, N, R).transpose(-1, -2)
    Cm = _c(c_src, f).view(*lead, N, O).transpose(-1, -2)
    if len(lead) > 1:
        Bm, Cm = (Bm.contiguous(), Cm.contiguous())
    if core.b_norm:
        Bm = _rms_last(Bm, core.B_norm, f)
    if core.c_norm:
        Cm = _rms_last(Cm, core.C_norm, f)
    if core.bc_bias:
        if not core.head_bias:
            Bm = Bm + _c(core.B_bias, f)
            Cm = Cm + _c(core.C_bias, f)
    if core.use_g_b:
        gB = 1.0 + _c(parts['gB'], f).view(*lead, N, R).transpose(-1, -2)
        Bbar = gB[..., None, :, None, :] * Bm[..., None, None, :, :]
    else:
        Bbar = Bm[..., None, None, :, :]
    if core.use_g_c:
        Cm = Cm * (1.0 + _c(parts['gC'], f).view(*lead, N, O).transpose(-1, -2))
    Cbar = Cm[..., None, None, :, :]
    if core.head_bias:
        Bbar = Bbar + _c(core.B_bias, f).transpose(-1, -2)
        Cbar = Cbar + _c(core.C_bias, f).transpose(-1, -2)
    dt_pre = _c(parts['dt'], f)
    if parts.get('_dtb') is not None:
        dt_pre = dt_pre + _c(parts['_dtb'], f)
    dt = F.softplus(dt_pre).view(*lead, H, R)
    if core.a_dyn:
        va = _c(parts['a'], f)
        if torch.is_grad_enabled():
            tail = _a_mag_tail_eager if core.a_act == 'heavy_tail' else _a_mag_softplus_eager
            a_mag = tail(va, 0.0001)
        elif core.a_act == 'heavy_tail':
            a_mag = torch.where(va >= 0, va + 1.0, torch.reciprocal(1.0 - va)).clamp_min(0.0001)
        else:
            a_mag = F.softplus(va).clamp_min(0.0001)
        rate = a_mag.view(*lead, H, R) if core.a_dyn_per_sibling else a_mag.view(*lead, H, 1)
    else:
        rate = core.A_log.to(f).exp()
    dA = -(dt * rate)
    lam = torch.sigmoid(_c(parts['trap'], f)) if core.use_trap else None
    incr = None
    if core.rot_mode == 'm3':
        K, ka = (N // 2, core.n_rope_angles)
        if core.omega_per_head:
            ang = torch.tanh(_c(parts['theta'], f)).view(*lead, H, 1, 1).expand(*lead, H, 1, ka)
        else:
            ang = torch.tanh(_c(parts['theta'], f)).view(*lead, H if core.theta_per_head else 1, 1, ka)
        if core.learn_theta_freq and core.omega_layout != 'pair':
            omega = (math.pi * torch.tanh(_c(core.theta_freq, f))).view(H, 1, -1)
        elif core.learn_theta_freq:
            omega = math.pi * torch.tanh(_c(core.theta_freq, f))
        else:
            omega = math.pi
        incr = dt[..., None] * ang * -omega
    chi = None
    if core.r_mix_mode != 'none':
        chi = _c(core.r_mix, f)
        if core.r_mix_dyn:
            chi = chi + _c(parts['rmix'], f).view(*lead, H, R)
    return Law(Bbar, Cbar, dt, dA, lam, incr, chi)

def _pad_pairs(core, ang: torch.Tensor) -> torch.Tensor:
    pad = core.N // 2 - ang.shape[-1]
    return F.pad(ang, (0, pad)) if pad else ang

def _wrap2pi(x: torch.Tensor) -> torch.Tensor:
    from .ssd_mimo import _wrap2pi_t
    return _wrap2pi_t(x)

def fold_pairs(v: torch.Tensor, ang: torch.Tensor, half: bool) -> torch.Tensor:
    c = torch.cos(ang).unsqueeze(-2)
    s = torch.sin(ang).unsqueeze(-2)
    N = v.shape[-1]
    if half:
        e, o = (v[..., :N // 2], v[..., N // 2:])
        return torch.cat((c * e + s * o, c * o - s * e), dim=-1)
    vv = v.view(*v.shape[:-1], N // 2, 2)
    e, o = (vv[..., 0], vv[..., 1])
    return torch.stack((c * e + s * o, c * o - s * e), dim=-1).view(*torch.broadcast_shapes(v.shape[:-1], c.shape[:-1]), N)

def _skip(core, u: torch.Tensor, chi, f):
    uw = u if chi is None else u * chi.unsqueeze(-2)
    return torch.matmul(uw.unsqueeze(-2), _c(core._skip_gains(), f)).squeeze(-2)

def _step_core(core, x_t, raw, gate_z, state, f):
    Bsz = x_t.shape[0]
    H, G, R, N, O = (core.H, core.G, core.R, core.N, core.O)
    with _no_autocast(x_t.device.type):
        parts = split_slab(core, raw)
        b_src, c_src = (parts['B'], parts['C'])
        new = {}
        if core.bc_conv1d is not None:
            bc, window = bc_conv(core, torch.cat([b_src, c_src], dim=-1)[:, None], state['bc_conv_state'], f)
            b_src, c_src = bc[:, 0].split([N * R, N * O], dim=-1)
            new['bc_conv_state'] = window
        law = token_law(core, parts, b_src, c_src, f, (Bsz,))
        Btil, Ctil = (law.Bbar, law.Cbar)
        if law.incr is not None:
            theta = _wrap2pi(_wrap2pi(_c(state['rot_ang'], f)) + law.incr)
            half = core.pair_layout == 'half'
            Btil = fold_pairs(Btil, _pad_pairs(core, theta), half)
            ang_c = theta + _c(core.theta_h, f)[:, None, :] if core.use_theta_h else theta
            Ctil = fold_pairs(Ctil, _pad_pairs(core, ang_c), half)
            new['rot_ang'] = theta
        u = _c(x_t, f).reshape(Bsz, H, G, R)
        Bk = Btil.expand(Bsz, H, R, R, N).transpose(2, 3).reshape(Bsz, H, R, R * N)
        v = torch.matmul(u, Bk).view(Bsz, H, G, R, N)
        f_old = _c(state['hidden_state'], f).view(Bsz, H, G, R, N)
        a5 = law.dA.exp().view(Bsz, H, 1, R, 1)
        dt5 = law.dt.view(Bsz, H, 1, R, 1)
        if law.lam is not None:
            lam5 = law.lam.view(Bsz, H, 1, 1, 1)
            v_prev = _c(state['write_prev'], f).view(Bsz, H, G, R, N)
            f_new = torch.addcmul(dt5 * lam5 * v, a5, torch.addcmul(f_old, dt5 * (1.0 - lam5), v_prev))
            new['write_prev'] = v
        else:
            f_new = torch.addcmul(dt5 * v, a5, f_old)
        new['hidden_state'] = f_new
        Cr = Ctil if law.chi is None else Ctil * law.chi[..., None, None]
        Ck = Cr.expand(Bsz, H, R, O, N).transpose(-1, -2).reshape(Bsz, H, R * N, O)
        y = torch.matmul(f_new.view(Bsz, H, G, R * N), Ck)
        if core.d_skip:
            y = y + _skip(core, u, law.chi, f)
        y = y.reshape(Bsz, H * G * O)
        if gate_z is not None:
            y = y * F.silu(_c(gate_z, f))
    P = G * R
    out = {}
    for k in state:
        if k not in new:
            continue
        v_ = new[k]
        if k in ('hidden_state', 'write_prev'):
            v_ = v_.reshape(Bsz, H, P, N)
        out[k] = _c(v_, state[k].dtype)
    return (y, out)

def ssd_step(core, x_t: torch.Tensor, state, gate_z=None, sel_x=None, tok_raw=None):
    if gate_z is not None and gate_z.shape != (*x_t.shape[:-1], core.d_out):
        raise ValueError(f'gate_z must have shape {(*x_t.shape[:-1], core.d_out)}')
    state = core.resolve_state(state, batch_size=x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
    u_sel = core._select(x_t, sel_x)
    raw = token_slab(core, u_sel, tok_raw)
    y, new = _step_core(core, x_t, raw, gate_z, state, compute_dtype(x_t))
    new_state = State(new)
    core.h_final = new_state['hidden_state']
    core._final_state = new_state
    return (y.to(x_t.dtype), new_state)

def _fused_step_weight(mixer):
    from ..layers.glue import param_cache_key, dt_fp32_on
    core = mixer.ssm
    dtz = dt_fp32_on()
    mods = [getattr(core, a) for a in core._proj_attrs]
    params = [mixer.in_proj.weight] + [m.weight for m in mods] + [m.bias for m in mods if m.bias is not None]
    cache = getattr(mixer, '_step_w_cache', None)
    pinned = getattr(mixer, '_inference_pin', None) is True
    if pinned and cache is not None:
        key = param_cache_key(params, dtz)
        if cache[0] == key:
            return (cache[1], cache[2])
    w = torch.cat([mixer.in_proj.weight] + [core._proj_w(a) for a in core._proj_attrs], dim=0)
    b = None
    if core._proj_any_bias:
        lead = mixer.in_proj.weight.shape[0]
        b = torch.cat([core._proj_zero.new_zeros(lead)] + [m.bias if m.bias is not None and (not (dtz and a == 'dt_proj')) else core._proj_zero[:n] for a, m, n in zip(core._proj_attrs, mods, core._proj_widths)])
    if pinned:
        mixer._step_w_cache = (param_cache_key(params, dtz), w, b)
    return (w, b)

def mixer_step_fast(mixer, x_t: torch.Tensor, state):
    core = mixer.ssm
    Bsz = x_t.shape[0]
    state = mixer.resolve_state(state, batch_size=Bsz, device=x_t.device, dtype=x_t.dtype)
    di, dg = (mixer.d_inner, mixer.d_gate)
    f = compute_dtype(x_t)
    with torch.no_grad():
        if mixer.fuse_in_proj:
            w, b = _fused_step_weight(mixer)
            xb, z, raw = F.linear(x_t, w, b).split((di, dg, core.tok_width), dim=-1)
        else:
            xb, z = mixer.in_proj(x_t).split((di, dg), dim=-1)
            raw = None
        new = {}
        if mixer.conv1d is not None:
            window = torch.cat([_c(state['conv_state'][:, :, 1:], xb.dtype), xb.unsqueeze(-1)], dim=-1)
            conv = mixer.conv1d
            xb = F.silu(torch.linalg.vecdot(window, _c(conv.weight[:, 0], xb.dtype)) + _c(conv.bias, xb.dtype))
            new['conv_state'] = window
        if raw is None:
            raw = token_slab(core, xb, None)
        y, carries = _step_core(core, xb, raw, z, state, f)
        new.update(carries)
        y = mixer.out_proj(mixer.norm(_c(y, x_t.dtype)))
        if getattr(mixer, 'step_inplace', False):
            for k, v in new.items():
                state[k].copy_(v)
            out = state
        else:
            out = State({k: _c(new[k], state[k].dtype) if k in new else state[k] for k in state})
    core.h_final = out['hidden_state']
    mixer._final_state = out
    return (y, out)

def grad_mode() -> str:
    import os
    v = os.environ.get('SSD_MIMO_TORCH_GRAD', 'fn').strip().lower()
    if v not in ('fn', 'autograd', 'auto'):
        raise ValueError(f'SSD_MIMO_TORCH_GRAD must be fn|autograd, got {v!r}')
    return 'autograd' if v == 'autograd' else 'fn'

def default_chunk(N: int) -> int:
    return 16 if N <= 64 else 32
SCAN_MAX_L = 8

def torch_route(L: int) -> str:
    import os
    v = os.environ.get('SSD_MIMO_TORCH_ROUTE', 'auto').strip().lower()
    if v not in ('auto', 'chunked', 'scan'):
        raise ValueError(f'SSD_MIMO_TORCH_ROUTE must be auto|chunked|scan, got {v!r}')
    if v == 'auto':
        return 'scan' if L <= SCAN_MAX_L else 'chunked'
    return v

class SeqPrep(NamedTuple):
    Btil: torch.Tensor
    Cq: torch.Tensor
    u: torch.Tensor
    law: Law
    theta: torch.Tensor | None
    conv_state: torch.Tensor | None
    f: torch.dtype

def _prepare(core, x, u_sel, tok_raw, carry) -> SeqPrep:
    B, L, _ = x.shape
    H, G, R, N, O = (core.H, core.G, core.R, core.N, core.O)
    f = compute_dtype(x)
    raw = token_slab(core, u_sel, tok_raw)
    with _no_autocast(x.device.type):
        parts = split_slab(core, raw)
        b_src, c_src = (parts['B'], parts['C'])
        conv_state = None
        if core.bc_conv1d is not None:
            bc, conv_state = bc_conv(core, torch.cat([b_src, c_src], dim=-1), carry.get('bc_conv_state'), f)
            b_src, c_src = bc.split([N * R, N * O], dim=-1)
        law = token_law(core, parts, b_src, c_src, f, (B, L))
        theta = None
        Btil, Ctil = (law.Bbar, law.Cbar)
        if law.incr is not None:
            theta = _wrap2pi(torch.cumsum(law.incr, dim=1))
            if carry.get('rot_ang') is not None:
                theta = _wrap2pi(theta + _wrap2pi(_c(carry['rot_ang'], f))[:, None])
            half = core.pair_layout == 'half'
            Btil = fold_pairs(Btil, _pad_pairs(core, theta), half)
            ang_c = theta + _c(core.theta_h, f)[:, None, :] if core.use_theta_h else theta
            Ctil = fold_pairs(Ctil, _pad_pairs(core, ang_c), half)
        Btil = Btil.expand(B, L, H, R, R, N)
        Cq = Ctil if law.chi is None else Ctil * law.chi[..., None, None]
        Cq = Cq.expand(B, L, H, R, O, N)
        u = _c(x, f).reshape(B, L, H, G, R)
    return SeqPrep(Btil, Cq, u, law, theta, conv_state, f)

def _initial_folded(prep: SeqPrep, h0, wp, lag0, B, H, G, R, N):
    f = prep.f
    S0 = None
    if h0 is not None:
        S0 = _c(h0, f).view(B, H, G, R, N).permute(0, 1, 3, 4, 2)
    if wp is not None and lag0 is not None:
        shift = _c(wp, f).view(B, H, G, R, N).permute(0, 1, 3, 4, 2) * lag0[..., None, None]
        S0 = shift if S0 is None else S0 + shift
    return S0

def _finish(core, prep: SeqPrep, x, y_scan, gate_z, S_fin, want_state, carry, v_last, h0=None):
    B, L, _ = x.shape
    H, G, R, N, O = (core.H, core.G, core.R, core.N, core.O)
    f = prep.f
    with _no_autocast(x.device.type):
        y = y_scan
        if core.d_skip:
            sk = prep.u.transpose(-1, -2).contiguous()[..., None] * _c(core._skip_gains(), f).permute(0, 2, 1, 3)
            if prep.law.chi is not None:
                sk = sk * prep.law.chi[..., None, None]
            y = y + sk.sum(-3)
        y = y.reshape(B, L, H * G * O)
        if gate_z is not None:
            y = y * F.silu(_c(gate_z, f))
        y = y.to(x.dtype)
    P = G * R
    hidden = S_fin.reshape(B, H, P, N).to(x.dtype if h0 is None else h0.dtype)
    state = None
    if want_state:
        state = State(hidden_state=hidden)
        if prep.theta is not None:
            rot = carry.get('rot_ang')
            state['rot_ang'] = prep.theta[:, -1].to(rot.dtype if rot is not None else f).clone()
        if prep.law.lam is not None:
            wp = carry.get('write_prev')
            state['write_prev'] = v_last.reshape(B, H, P, N).to(wp.dtype if wp is not None else x.dtype)
        if prep.conv_state is not None:
            cst = carry.get('bc_conv_state')
            state['bc_conv_state'] = prep.conv_state.to(cst.dtype if cst is not None else x.dtype).contiguous()
    return (y, state, hidden)

def _segsum(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    xe = x[..., :, None].expand(*x.shape, n)
    lower = torch.ones(n, n, dtype=torch.bool, device=x.device).tril(-1)
    ss = torch.cumsum(xe.masked_fill(~lower, 0.0), dim=-2)
    keep = torch.ones(n, n, dtype=torch.bool, device=x.device).tril(0)
    return ss.masked_fill(~keep, float('-inf'))

def _chunk_fwd(Bq, Cq, U, dAz, wz, wdz, S0):
    B, nc, H, R, TRi, N = Bq.shape
    T = dAz.shape[-1]
    Ri, O, G = (TRi // T, Cq.shape[-2] // T, U.shape[-1])
    cs = torch.cumsum(dAz, dim=-1)
    E = torch.exp(_segsum(dAz))
    eye = torch.eye(T, dtype=torch.bool, device=E.device)
    W = torch.where(eye, wdz[..., :, None], E * wz[..., None, :])
    S = torch.matmul(Cq, Bq.transpose(-1, -2))
    M = S.view(B, nc, H, R, O, T, Ri, T) * W.view(B, nc, H, R, 1, T, 1, T)
    M = M.sum(3) if R > 1 else M.squeeze(3)
    Y = torch.matmul(M.reshape(B, nc, H, O * T, Ri * T), U)
    dend = E[..., -1, :] * wz
    Bw = (Bq.view(B, nc, H, R, Ri, T, N) * dend[:, :, :, :, None, :, None]).view(B, nc, H, R, TRi, N)
    Send = torch.matmul(Bw.transpose(-1, -2), U[:, :, :, None])
    Z0 = S0 if S0 is not None else Send.new_zeros(B, H, R, N, G)
    Zm = torch.cat([Z0[:, None], Send], dim=1)
    xz = F.pad(cs[..., -1].permute(0, 2, 3, 1), (1, 0))
    Dm = torch.exp(_segsum(xz))
    Sin = torch.einsum('bhrij,bjhrng->bihrng', Dm, Zm)
    P = torch.matmul(Cq, Sin[:, :nc])
    ys = P.view(B, nc, H, R, O, T, G) * torch.exp(cs)[:, :, :, :, None, :, None]
    ys = ys.sum(3) if R > 1 else ys.squeeze(3)
    return (Y + ys.reshape(B, nc, H, O * T, G), Sin)

def _chunk_scan(Bt, Cq, u, dA, w, wd, S0, T):
    B, L, H, R, Ri, N = Bt.shape
    O, G = (Cq.shape[-2], u.shape[3])
    nc = -(-L // T)
    pad = nc * T - L

    def chunked(t):
        t = _pad_len(t, pad)
        return t.reshape(B, nc, T, *t.shape[2:])
    Bq = chunked(Bt).permute(0, 1, 3, 4, 5, 2, 6).reshape(B, nc, H, R, Ri * T, N)
    Cc = chunked(Cq).permute(0, 1, 3, 4, 5, 2, 6).reshape(B, nc, H, R, O * T, N)
    Uc = chunked(u).permute(0, 1, 3, 5, 2, 4).reshape(B, nc, H, Ri * T, G)
    Bq, Cc, Uc = (Bq.contiguous(), Cc.contiguous(), Uc.contiguous())
    dAz, wz, wdz = (chunked(t).permute(0, 1, 3, 4, 2).contiguous() for t in (dA, w, wd))
    Y, Sin = _chunk_fwd(Bq, Cc, Uc, dAz, wz, wdz, S0)
    S_fin = Sin[:, -1]
    y = Y.view(B, nc, H, O, T, G).permute(0, 1, 4, 2, 5, 3).reshape(B, nc * T, H, G, O)[:, :L]
    return (y, S_fin)

def _pad_len(t: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return t
    widths = [0, 0] * (t.dim() - 2) + [0, pad]
    return F.pad(t, widths)

def ssd_chunked(core, x: torch.Tensor, u_sel: torch.Tensor, *, h0: torch.Tensor | None=None, gate_z=None, tok_raw=None, carry: dict | None=None, want_state: bool=False, chunk_size: int | None=None):
    B, L, _ = x.shape
    H, G, R, N, O = (core.H, core.G, core.R, core.N, core.O)
    carry = carry or {}
    prep = _prepare(core, x, u_sel, tok_raw, carry)
    law = prep.law
    with _no_autocast(x.device.type):
        dt, dA = (law.dt, law.dA)
        if law.lam is not None:
            lam = law.lam[..., None]
            lag = dt * (1.0 - lam)
            wd = lam * dt
            w = wd + F.pad(lag[:, 1:], (0, 0, 0, 0, 0, 1))
            lag0 = lag[:, 0]
        else:
            w = wd = dt
            lag0 = None
        wp = carry.get('write_prev')
        if wp is not None and lag0 is None:
            raise ValueError("state carry 'write_prev' with write_rule='euler': the one-tap write has no two-endpoint lag to fold")
        S0 = _initial_folded(prep, h0, wp, lag0, B, H, G, R, N)
        T = max(1, min(int(chunk_size or DEFAULT_CHUNK or default_chunk(N)), L))
        y_scan, S_fin = _chunk_scan(prep.Btil, prep.Cq, prep.u, dA, w, wd, S0, T)
        v_last = None
        if want_state and law.lam is not None:
            v_last = torch.einsum('bhrqn,bhgq->bhgrn', prep.Btil[:, -1], prep.u[:, -1])
    return _finish(core, prep, x, y_scan, gate_z, S_fin.permute(0, 1, 4, 2, 3), want_state, carry, v_last, h0=h0)

class _LinearScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, f0):
        fs = torch.empty_like(b)
        f = f0
        for t in range(b.shape[1]):
            f = torch.addcmul(b[:, t], a[:, t], f)
            fs[:, t] = f
        ctx.save_for_backward(a, fs, f0)
        return fs

    @staticmethod
    def backward(ctx, gfs):
        a, fs, f0 = ctx.saved_tensors
        L = fs.shape[1]
        Fa = torch.empty_like(fs)
        acc = gfs[:, L - 1]
        Fa[:, L - 1] = acc
        for t in range(L - 2, -1, -1):
            acc = torch.addcmul(gfs[:, t], a[:, t + 1], acc)
            Fa[:, t] = acc
        fprev = torch.cat([f0[:, None], fs[:, :-1]], dim=1)
        ga = (Fa * fprev).sum((3, 5), keepdim=True)
        return (ga, Fa, a[:, 0] * Fa[:, 0])

def _linear_scan_autograd(a, b, f0):
    fs, f = ([], f0)
    for t in range(b.shape[1]):
        f = torch.addcmul(b[:, t], a[:, t], f)
        fs.append(f)
    return torch.stack(fs, dim=1)

def ssd_scan_seq(core, x: torch.Tensor, u_sel: torch.Tensor, *, h0: torch.Tensor | None=None, gate_z=None, tok_raw=None, carry: dict | None=None, want_state: bool=False):
    B, L, _ = x.shape
    H, G, R, N, O = (core.H, core.G, core.R, core.N, core.O)
    carry = carry or {}
    prep = _prepare(core, x, u_sel, tok_raw, carry)
    law = prep.law
    f = prep.f
    with _no_autocast(x.device.type):
        Bk = prep.Btil.transpose(3, 4).reshape(B, L, H, R, R * N)
        v = torch.matmul(prep.u, Bk).view(B, L, H, G, R, N)
        a = torch.exp(law.dA).view(B, L, H, 1, R, 1)
        dt6 = law.dt.view(B, L, H, 1, R, 1)
        if law.lam is not None:
            lam6 = law.lam.view(B, L, H, 1, 1, 1)
            wp = carry.get('write_prev')
            vp0 = _c(wp, f).view(B, 1, H, G, R, N) if wp is not None else v.new_zeros(B, 1, H, G, R, N)
            v_prev = torch.cat([vp0, v[:, :-1]], dim=1)
            b = dt6 * lam6 * v + a * dt6 * (1.0 - lam6) * v_prev
        else:
            b = dt6 * v
        f0 = _c(h0, f).view(B, H, G, R, N) if h0 is not None else v.new_zeros(B, H, G, R, N)
        if grad_mode() == 'fn':
            fs = _LinearScanFn.apply(a, b, f0)
        else:
            fs = _linear_scan_autograd(a, b, f0)
        y_scan = torch.matmul(fs.view(B, L, H, G, R * N), prep.Cq.transpose(-1, -2).reshape(B, L, H, R * N, O))
    return _finish(core, prep, x, y_scan, gate_z, fs[:, -1].clone(), want_state, carry, v[:, -1].clone(), h0=h0)
