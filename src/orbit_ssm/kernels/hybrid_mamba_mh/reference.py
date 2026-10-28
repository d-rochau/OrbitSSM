
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .layout import DYN_SOFTPLUS, GATE_SIGMOID, LAW_A_DYN, LAW_A_FIX, LAW_MU, group_index
from .layout import dyn_rate as _dyn_rate
from .layout import gate_act as _gate_act
DT = torch.float64

def _f(t):
    return None if t is None else t.to(DT)

def _expand_hgroup(beta, H):
    if beta is None:
        return None
    G = beta.shape[0]
    return beta[group_index(H, G, device=beta.device)]

def _z_gate(Y, z):
    if z is None:
        return Y
    return Y * (_f(z) * torch.sigmoid(_f(z)))

def _row_broadcast(t, B, H, N, L, res=None):
    if t is None:
        return None
    if res is None:
        if N == H:
            raise ValueError("a_dyn/mu resolution is ambiguous when N == H; pass a_dyn_res=/mu_res= 'n' or 'h' explicitly (contract Sec.3.3)")
        res = 'n' if t.shape[1] == N else 'h'
    return t.unsqueeze(1).expand(B, H, N, L) if res == 'n' else t.unsqueeze(2)

def _prepare(x, delta, A_log, B_t, C_t, *, delta_w=None, delta_softplus=True, row_law=None, row_rot=None, a_dyn=None, mu=None, mu_gain=None, a_dyn_res=None, mu_res=None, dyn_act=DYN_SOFTPLUS, dyn_eps=0.0, g_B=None, g_C=None, gate_act=GATE_SIGMOID, theta=None, nu=None, theta_delta=False, theta_h=None, beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None, delta_bias=None, theta_init=None, mu_init=None):
    B, H, P, L = x.shape
    N = A_log.shape[2]
    K = N // 2
    dev = x.device
    x, delta, A_log, B_t, C_t = (_f(x), _f(delta), _f(A_log), _f(B_t), _f(C_t))
    delta_w, a_dyn, mu = (_f(delta_w), _f(a_dyn), _f(mu))
    mu_gain = _f(mu_gain)
    g_B, g_C, theta, nu = (_f(g_B), _f(g_C), _f(theta), _f(nu))
    theta_h = _f(theta_h)
    beta_a, beta_b, beta_c = (_f(beta_a), _f(beta_b), _f(beta_c))
    beta_gB, beta_gC = (_f(beta_gB), _f(beta_gC))
    delta_bias, theta_init, mu_init = (_f(delta_bias), _f(theta_init), _f(mu_init))
    if row_law is None:
        row_law = torch.full((N,), LAW_A_FIX, dtype=torch.int32, device=dev)
    if row_rot is None:
        row_rot = torch.full((N,), theta is not None or nu is not None, dtype=torch.bool, device=dev)
    ba = _expand_hgroup(beta_a, H)
    bb = _expand_hgroup(beta_b, H)
    bc = _expand_hgroup(beta_c, H)
    bgB = _expand_hgroup(beta_gB, H)
    bgC = _expand_hgroup(beta_gC, H)
    d_raw = delta if delta_bias is None else delta + delta_bias.view(1, H, 1)
    Delta_eff = F.softplus(d_raw) if delta_softplus else d_raw
    if delta_w is None:
        Delta_w = Delta_eff.unsqueeze(2)
    else:
        dw = F.softplus(delta_w) if delta_softplus else delta_w
        Delta_w = dw.unsqueeze(2) if dw.dim() == 3 else dw
    a_off = 0.0 if ba is None else ba.view(H, 1, N)
    A_nat = -torch.exp(A_log + a_off)
    is_mu = (row_law == LAW_MU).view(1, 1, N)
    is_dyn = (row_law == LAW_A_DYN).view(1, 1, N)
    c1 = A_nat
    c2 = torch.zeros_like(c1) if mu_gain is None else torch.where(is_mu, mu_gain, torch.zeros_like(mu_gain))
    De = Delta_eff.unsqueeze(2)
    s1 = De.expand(B, H, N, L).clone()
    if a_dyn is not None:
        rate = _dyn_rate(a_dyn, dyn_act, dyn_eps)
        s1 = torch.where(is_dyn.unsqueeze(-1), De * _row_broadcast(rate, B, H, N, L, a_dyn_res), s1)
    if mu is not None:
        mu_b = _row_broadcast(mu, B, H, N, L, mu_res).expand(B, H, N, L)
        if mu_init is None:
            mu_prev0 = torch.zeros_like(mu_b[..., :1])
        else:
            mi = mu_init
            mi = mi.view(B, 1, N, 1) if mi.shape[1] == N else mi.view(B, H, 1, 1)
            mu_prev0 = mi.expand(B, H, N, 1)
        s2 = mu_b - torch.cat([mu_prev0, mu_b[..., :-1]], dim=-1)
        s2 = torch.where(is_mu.unsqueeze(-1), s2, torch.zeros_like(s2))
    else:
        s2 = torch.zeros_like(s1)
    if theta is not None or nu is not None:
        inc = torch.zeros(B, H, K, L, dtype=DT, device=dev)
        if theta is not None:
            th = theta if theta.dim() == 4 else theta.unsqueeze(1)
            if theta_delta:
                inc = inc + th * Delta_eff[:, :, None, :]
            else:
                inc = inc + th
        if nu is not None:
            inc = inc + nu.view(1, H, K, 1)
        Theta = torch.cumsum(inc, dim=-1)
        if theta_init is not None:
            Theta = Theta + theta_init.unsqueeze(-1)
        cosT = torch.cos(Theta)
        sinT = torch.sin(Theta)
        if theta_h is not None:
            ThC = Theta + theta_h.view(1, H, K, 1)
            cosC, sinC = (torch.cos(ThC), torch.sin(ThC))
        else:
            cosC, sinC = (cosT, sinT)
    else:
        cosT = sinT = cosC = sinC = None
    Bh = B_t.unsqueeze(1).expand(B, H, N, L)
    Ch = C_t.unsqueeze(1).expand(B, H, N, L)
    if bb is not None:
        Bh = Bh + bb.view(1, H, N, 1)
    if bc is not None:
        Ch = Ch + bc.view(1, H, N, 1)
    if cosT is not None:
        rot = row_rot.view(1, 1, N, 1)
        Bh = torch.where(rot, _rotate_pairs(Bh, cosT, sinT), Bh)
        Ch = torch.where(rot, _rotate_pairs(Ch, cosC, sinC), Ch)
    if g_B is not None:
        gv = g_B.unsqueeze(1).expand(B, H, N, L)
        if bgB is not None:
            gv = gv + bgB.view(1, H, N, 1)
        Bh = Bh * _gate_act(gv, gate_act)
    if g_C is not None:
        gv = g_C.unsqueeze(1).expand(B, H, N, L)
        if bgC is not None:
            gv = gv + bgC.view(1, H, N, 1)
        Ch = Ch * _gate_act(gv, gate_act)
    return dict(B=B, H=H, P=P, N=N, L=L, x=x, c1=c1, c2=c2, s1=s1, s2=s2, Delta_w=Delta_w, w_n=Bh, r_n=Ch)

def _rotate_pairs(v, cosT, sinT):
    Bs, H, N, L = v.shape
    vp = v.reshape(Bs, H, N // 2, 2, L)
    u, w = (vp[:, :, :, 0], vp[:, :, :, 1])
    ru = u * cosT - w * sinT
    rw = u * sinT + w * cosT
    return torch.stack((ru, rw), dim=3).reshape(Bs, H, N, L)

def mh_reference(x, delta, A_log, B_t, C_t, *, h_init=None, n_off=0, n_rows=None, d_skip=None, z=None, trap_lam=None, write_prev=None, **kw):
    st = _prepare(x, delta, A_log, B_t, C_t, **kw)
    B, H, P, N, L = (st['B'], st['H'], st['P'], st['N'], st['L'])
    if n_rows is None:
        n_rows = N - n_off
    sl = slice(n_off, n_off + n_rows)
    xh = st['x']
    c1, c2 = (st['c1'][..., sl], st['c2'][..., sl])
    s1, s2 = (st['s1'][..., sl, :], st['s2'][..., sl, :])
    w_n, r_n = (st['w_n'][..., sl, :], st['r_n'][..., sl, :])
    Dw = st['Delta_w']
    h = torch.zeros(B, H, P, N, dtype=DT, device=x.device) if h_init is None else _f(h_init).clone()
    state = h[..., sl]
    previous = torch.zeros_like(state) if write_prev is None else _f(write_prev)[..., sl]
    ys = []
    for t in range(L):
        log_a = c1.unsqueeze(0) * s1[..., t].unsqueeze(2) + c2.unsqueeze(0) * s2[..., t].unsqueeze(2)
        a_bar = torch.exp(log_a)
        drive = (Dw[..., t] * xh[..., t]).unsqueeze(-1) * w_n[..., t].unsqueeze(2)
        if trap_lam is not None:
            lam = _f(trap_lam)[..., t, None, None]
            raw_write = xh[..., t, None] * w_n[..., t].unsqueeze(2)
            drive = Dw[..., t, None] * (lam * raw_write + (1 - lam) * a_bar * previous)
            previous = raw_write
        state = a_bar * state + drive
        ys.append((r_n[..., t].unsqueeze(2) * state).sum(-1))
    Y = torch.stack(ys, dim=-1)
    if d_skip is not None:
        Y = Y + _f(d_skip).view(1, H, 1, 1) * xh
    Y = _z_gate(Y, z)
    h_out = h.clone()
    h_out[..., sl] = state
    return (Y, h_out)

def mh_reference_cumprod(x, delta, A_log, B_t, C_t, *, h_init=None, n_off=0, n_rows=None, d_skip=None, z=None, trap_lam=None, write_prev=None, **kw):
    st = _prepare(x, delta, A_log, B_t, C_t, **kw)
    B, H, P, N, L = (st['B'], st['H'], st['P'], st['N'], st['L'])
    if n_rows is None:
        n_rows = N - n_off
    sl = slice(n_off, n_off + n_rows)
    xh = st['x']
    c1, c2 = (st['c1'][..., sl], st['c2'][..., sl])
    S1 = torch.cumsum(st['s1'][..., sl, :], dim=-1)
    S2 = torch.cumsum(st['s2'][..., sl, :], dim=-1)
    w_n, r_n = (st['w_n'][..., sl, :], st['r_n'][..., sl, :])
    Dw = st['Delta_w']
    log_A = c1.view(1, H, P, n_rows, 1) * S1.unsqueeze(2) + c2.view(1, H, P, n_rows, 1) * S2.unsqueeze(2)
    A_cum = torch.exp(log_A)
    drive = (Dw * xh).unsqueeze(3) * w_n.unsqueeze(2)
    if trap_lam is not None:
        raw_write = xh.unsqueeze(3) * w_n.unsqueeze(2)
        prior = torch.zeros_like(raw_write[..., :1]) if write_prev is None else _f(write_prev)[..., sl, None]
        previous = torch.cat((prior, raw_write[..., :-1]), -1)
        prefix_previous = torch.cat((torch.ones_like(A_cum[..., :1]), A_cum[..., :-1]), -1)
        lam = _f(trap_lam)[:, :, None, None, :]
        normalized = Dw.unsqueeze(3) * (lam * raw_write / A_cum + (1 - lam) * previous / prefix_previous)
        inner = torch.cumsum(normalized, dim=-1)
    else:
        inner = torch.cumsum(drive / A_cum, dim=-1)
    if h_init is not None:
        inner = inner + _f(h_init)[..., sl].unsqueeze(-1)
    hs = A_cum * inner
    Y = (r_n.unsqueeze(2) * hs).sum(3)
    if d_skip is not None:
        Y = Y + _f(d_skip).view(1, H, 1, 1) * st['x']
    Y = _z_gate(Y, z)
    h_out = torch.zeros(B, H, P, N, dtype=DT, device=x.device) if h_init is None else _f(h_init).clone()
    h_out[..., sl] = hs[..., -1]
    return (Y, h_out)

def physical_oracle(x, delta, A_log, B_t, C_t, *, trap_lam, h_init=None, write_prev=None, n_off=0, n_rows=None, d_skip=None, z=None, check_collapsed=True, **kw):
    st = _prepare(x, delta, A_log, B_t, C_t, **kw)
    B, H, P, N, L = (st['B'], st['H'], st['P'], st['N'], st['L'])
    if n_rows is None:
        n_rows = N - n_off
    sl = slice(n_off, n_off + n_rows)
    xh = st['x']
    c1, c2 = (st['c1'][..., sl], st['c2'][..., sl])
    s1, s2 = (st['s1'][..., sl, :], st['s2'][..., sl, :])
    w_n, r_n = (st['w_n'][..., sl, :], st['r_n'][..., sl, :])
    Dw = st['Delta_w']
    lam = _f(trap_lam)
    a = [torch.exp(c1.unsqueeze(0) * s1[..., t].unsqueeze(2) + c2.unsqueeze(0) * s2[..., t].unsqueeze(2)) for t in range(L)]
    v = [xh[..., t, None] * w_n[..., t].unsqueeze(2) for t in range(L)]
    dw = [Dw[..., t, None] for t in range(L)]
    lm = [lam[..., t, None, None] for t in range(L)]
    h = torch.zeros(B, H, P, N, dtype=DT, device=x.device) if h_init is None else _f(h_init).clone()
    state = h[..., sl].clone()
    prev = torch.zeros_like(state) if write_prev is None else _f(write_prev)[..., sl]
    ys = []
    for t in range(L):
        state = a[t] * state + dw[t] * (lm[t] * v[t] + (1.0 - lm[t]) * a[t] * prev)
        prev = v[t]
        ys.append((r_n[..., t].unsqueeze(2) * state).sum(-1))
    Y = torch.stack(ys, dim=-1)
    if check_collapsed:
        nx = [dw[t + 1] * (1.0 - lm[t + 1]) if t + 1 < L else torch.zeros_like(dw[0]) for t in range(L)]
        g = torch.zeros(B, H, P, N, dtype=DT, device=x.device) if h_init is None else _f(h_init).clone()
        gs = g[..., sl].clone()
        if write_prev is not None:
            gs = gs + dw[0] * (1.0 - lm[0]) * _f(write_prev)[..., sl]
        yc = []
        for t in range(L):
            w = lm[t] * dw[t] + nx[t]
            gs = a[t] * gs + w * v[t]
            d_t = (r_n[..., t] * w_n[..., t]).sum(-1)[..., None]
            yc.append((r_n[..., t].unsqueeze(2) * gs).sum(-1) - nx[t][..., 0] * d_t * xh[..., t])
        Yc = torch.stack(yc, dim=-1)
        err = (Yc - Y).abs().max() / Y.abs().max().clamp_min(1e-300)
        assert err < 1e-10, f'the collapsed one-tap form disagrees with the physical two-endpoint law by {float(err):.3e} -- the identity of implementation.md 1.1 does not hold as written'
        gerr = (gs - state).abs().max() / state.abs().max().clamp_min(1e-300)
        assert gerr < 1e-10, f'g_L != h_L by {float(gerr):.3e}: the boundary n_L = 0 is what makes the collapsed carry the physical final state'
    if d_skip is not None:
        Y = Y + _f(d_skip).view(1, H, 1, 1) * xh
    Y = _z_gate(Y, z)
    h_out = h.clone()
    h_out[..., sl] = state
    return (Y, h_out)

def rel_err(a, b):
    a, b = (a.double(), b.double())
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()

def m3_reference_cumprod(Bt, Ct, v, ADT, DT, Trap, K_bias, Q_bias, angles, *, S_init=None, kp_init=None, vp_init=None, phase_init=None, return_state=False):
    Bt, Ct, v = (Bt.double(), Ct.double(), v.double())
    ADT, DT = (ADT.double(), DT.double())
    lam = torch.sigmoid(Trap.double())
    alpha = torch.exp(ADT)
    dt_blh = DT.transpose(1, 2)
    phase0 = phase_init.double().unsqueeze(1) if phase_init is not None else 0.0
    theta = phase0 + torch.cumsum(torch.tanh(angles.double()) * math.pi * dt_blh.unsqueeze(-1), dim=1)
    k = Bt + K_bias.double().unsqueeze(0).unsqueeze(0)
    q = Ct + Q_bias.double().unsqueeze(0).unsqueeze(0)
    Ka = angles.shape[-1]
    cos, sin = (torch.cos(theta), torch.sin(theta))

    def rot(t):
        e, o = (t[..., 0::2], t[..., 1::2])
        er = torch.cat([e[..., :Ka] * cos - o[..., :Ka] * sin, e[..., Ka:]], dim=-1)
        og = torch.cat([e[..., :Ka] * sin + o[..., :Ka] * cos, o[..., Ka:]], dim=-1)
        return torch.stack([er, og], dim=-1).flatten(-2)
    k, q = (rot(k), rot(q))
    Bsz, L, H, P = v.shape
    n3 = k.shape[-1]
    lam_blh = lam.transpose(1, 2)
    dt_ = dt_blh
    raw_write = v.unsqueeze(-1) * k.unsqueeze(-2)
    kp0 = kp_init.double() if kp_init is not None else k.new_zeros(Bsz, H, n3)
    vp0 = vp_init.double() if vp_init is not None else v.new_zeros(Bsz, H, P)
    prev0 = (vp0.unsqueeze(-1) * kp0.unsqueeze(-2)).unsqueeze(1)
    previous = torch.cat([prev0, raw_write[:, :-1]], dim=1)
    log_alpha = torch.log(alpha).transpose(1, 2)
    A_cum = torch.exp(torch.cumsum(log_alpha, dim=1))
    drive = (alpha.transpose(1, 2) * dt_ * (1 - lam_blh))[..., None, None] * previous + (lam_blh * dt_)[..., None, None] * raw_write
    normalized = drive / A_cum[..., None, None]
    inner = torch.cumsum(normalized, dim=1)
    if S_init is not None:
        inner = inner + S_init.double().unsqueeze(1)
    S = A_cum[..., None, None] * inner
    Y = torch.einsum('blhpn,blhn->blhp', S, q)
    if not return_state:
        return Y
    state = dict(S=S[:, -1], kp=k[:, -1], vp=v[:, -1], phase=theta[:, -1])
    return (Y, state)

def flat_reference_cumprod(Bt, Ct, v, log_a, write, B_bias, C_bias, *, S_init=None, return_state=False):
    Bt, Ct, v = (Bt.double(), Ct.double(), v.double())
    a_blh = torch.exp(log_a.double()).transpose(1, 2)
    w_blh = write.double().transpose(1, 2)
    k = Bt.unsqueeze(2) + B_bias.double()
    q = Ct.unsqueeze(2) + C_bias.double()
    Bsz, L, H, P = v.shape
    n = k.shape[-1]
    raw_write = v.unsqueeze(-1) * k.unsqueeze(-2)
    log_a_blh = torch.log(a_blh)
    A_cum = torch.exp(torch.cumsum(log_a_blh, dim=1))
    drive = w_blh[..., None, None] * raw_write
    normalized = drive / A_cum[..., None, None]
    inner = torch.cumsum(normalized, dim=1)
    if S_init is not None:
        inner = inner + S_init.double().unsqueeze(1)
    S = A_cum[..., None, None] * inner
    Y = torch.einsum('blhpn,blhn->blhp', S, q)
    return (Y, S[:, -1]) if return_state else Y

def row_mixture_reference(x, delta, A_log, B_t, C_t, *, n_hm, m3=None, m2=None, d_skip=None, z=None, **hm_kw):
    B, H, P, L = x.shape
    Y = torch.zeros(B, H, P, L, dtype=DT, device=x.device)
    if n_hm > 0:
        y_hm, _ = mh_reference(x, delta, A_log, B_t, C_t, n_rows=n_hm, **hm_kw)
        Y = Y + y_hm
    if m3 is not None:
        y3 = m3_reference_cumprod(**m3)
        Y = Y + y3.permute(0, 2, 3, 1)
    if m2 is not None:
        y2 = flat_reference_cumprod(**m2)
        Y = Y + y2.permute(0, 2, 3, 1)
    if d_skip is not None:
        Y = Y + _f(d_skip).view(1, H, 1, 1) * _f(x)
    return _z_gate(Y, z)
