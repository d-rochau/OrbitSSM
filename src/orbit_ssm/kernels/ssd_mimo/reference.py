
import torch
from typing import Optional

def blockdiag_apply(Q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    *batch, N = x.shape
    M, k = (Q.shape[-3], Q.shape[-1])
    xb = x.view(*batch, M, k)
    y = torch.einsum('...mij,...mj->...mi', Q, xb)
    return y.reshape(*batch, N)

def rot_pairs(x: torch.Tensor, phi: torch.Tensor, pair_half: bool=False) -> torch.Tensor:
    cs = torch.cos(phi)
    sn = torch.sin(phi)
    if pair_half:
        K = x.shape[-1] // 2
        xe, xo = (x[..., :K], x[..., K:])
        cs, sn = (torch.cos(phi), torch.sin(phi))
        return torch.cat([cs * xe + sn * xo, -sn * xe + cs * xo], dim=-1)
    xe, xo = (x[..., 0::2], x[..., 1::2])
    out = torch.empty_like(x)
    out[..., 0::2] = cs * xe + sn * xo
    out[..., 1::2] = -sn * xe + cs * xo
    return out

def unified_reference(x_in: torch.Tensor, delta: torch.Tensor, B_tok: torch.Tensor, C_tok: torch.Tensor, a_log: torch.Tensor, Q: torch.Tensor, g: int, beta: Optional[torch.Tensor]=None, omega: Optional[torch.Tensor]=None, x0: Optional[torch.Tensor]=None, beta_in_rotated_frame: bool=True, chunk_size: Optional[int]=None, bc_per_head: bool=False, theta_rot: Optional[torch.Tensor]=None, theta_h: Optional[torch.Tensor]=None, gamma: Optional[torch.Tensor]=None, pair_layout: str='interleaved') -> tuple[torch.Tensor, torch.Tensor]:
    B, G, R, L = x_in.shape
    hax = 1 if bc_per_head else 0
    B_per_r = B_tok.dim() == 4 + hax
    C_per_o = C_tok.dim() == 4 + hax
    N = B_tok.shape[2 + hax] if B_per_r else B_tok.shape[-1]
    O = C_tok.shape[-1] if C_per_o else 1
    if B_per_r:
        assert B_tok.shape[-1] == R
    Gb = B_tok.shape[2] if bc_per_head else 1
    Gc = C_tok.shape[2] if bc_per_head else 1
    M, k = (Q.shape[-3], Q.shape[-1])
    per_block_decay = a_log.dim() == 1 and a_log.shape[0] == M
    per_head_decay = a_log.dim() == 1 and a_log.shape[0] != M or a_log.dim() > 1
    has_omega = omega is not None
    has_beta = beta is not None
    has_rot = theta_rot is not None
    _ph = pair_layout == 'half'
    has_theta_h = theta_h is not None
    has_gamma = gamma is not None
    H_actual = G
    if has_beta:
        H_actual = beta.shape[0]
        assert H_actual % G == 0
    elif has_theta_h:
        H_actual = theta_h.shape[0]
        assert H_actual % G == 0
    elif has_gamma:
        H_actual = gamma.shape[0]
        assert H_actual % G == 0
    heads_per_group = max(H_actual // G, 1)
    if bc_per_head:
        assert H_actual % Gb == 0, f'Gb={Gb} must divide H_actual={H_actual}'
        assert H_actual % Gc == 0, f'Gc={Gc} must divide H_actual={H_actual}'
        hpg_b = H_actual // Gb
        hpg_c = H_actual // Gc
    state = x0.clone() if x0 is not None else x_in.new_zeros(B, H_actual, g, N)
    ys = []
    Pi_t = None
    if has_beta and beta_in_rotated_frame:
        Pi_t = torch.eye(k, device=x_in.device, dtype=x_in.dtype).unsqueeze(0).unsqueeze(0).expand(B, M, k, k).clone()
    for t in range(L):
        d = delta[:, :, t]
        Q_t = Q[:, t]
        if Pi_t is not None:
            chunk_boundary = t == 0 or (chunk_size is not None and t % chunk_size == 0)
            if chunk_boundary:
                Pi_t = Q_t.clone()
            else:
                Pi_t = Q_t @ Pi_t
        y_t = x_in.new_zeros(B, G, O, g) if C_per_o else x_in.new_zeros(B, G, g)
        for gi in range(G):
            for hi in range(heads_per_group):
                h = gi * heads_per_group + hi
                d_h = d[:, gi]
                if per_block_decay:
                    a_vals = -torch.exp(a_log)
                    decay_n = torch.zeros(B, N, device=x_in.device, dtype=x_in.dtype)
                    for m_idx in range(M):
                        decay_n[:, m_idx * k:(m_idx + 1) * k] = torch.exp(a_vals[m_idx] * d_h).unsqueeze(-1).expand(-1, k)
                else:
                    a_val = -torch.exp(a_log[h] if a_log.numel() > 1 else a_log[0])
                    decay_n = torch.exp(a_val * d_h).unsqueeze(-1).expand(-1, N)
                for ci in range(g):
                    s = state[:, h, ci, :]
                    s_rot = blockdiag_apply(Q_t, s)
                    if has_omega:
                        N2 = N // 2
                        theta = omega * d_h.unsqueeze(-1)
                        cs = torch.cos(theta)
                        sn = torch.sin(theta)
                        s_re = s_rot[:, :N2]
                        s_im = s_rot[:, N2:]
                        s_rot_re = s_re * cs - s_im * sn
                        s_rot_im = s_re * sn + s_im * cs
                        s_rot = torch.cat([s_rot_re, s_rot_im], dim=-1)
                    s_decayed = decay_n * s_rot
                    drive = torch.zeros(B, N, device=x_in.device, dtype=x_in.dtype)
                    B_t = B_tok[:, t, h // hpg_b] if bc_per_head else B_tok[:, t]
                    phi_t = theta_rot[:, t] if has_rot else None
                    for r in range(R):
                        B_t_r = B_t[:, :, r] if B_per_r else B_t
                        if has_beta:
                            beta_r = beta[h, :, r]
                            if has_rot:
                                B_eff_r = B_t_r + beta_r.unsqueeze(0)
                            elif beta_in_rotated_frame and Pi_t is not None:
                                beta_r_natural = blockdiag_apply(Pi_t, beta_r.unsqueeze(0).expand(B, -1))
                                B_eff_r = B_t_r + beta_r_natural
                            else:
                                B_eff_r = B_t_r + beta_r.unsqueeze(0)
                        else:
                            B_eff_r = B_t_r
                        if has_rot:
                            B_eff_r = rot_pairs(B_eff_r, phi_t, _ph)
                        xr = x_in[:, gi, r, t]
                        drive = drive + B_eff_r * (d_h * xr).unsqueeze(-1)
                    s_new = s_decayed + drive
                    state[:, h, ci, :] = s_new
                    if has_theta_h:
                        phi_h = theta_h[h]
                        cs_h = torch.cos(phi_h)
                        sn_h = torch.sin(phi_h)
                        if _ph:
                            Kh = s_new.shape[-1] // 2
                            se, so = (s_new[..., :Kh], s_new[..., Kh:])
                            s_read = torch.cat([cs_h * se - sn_h * so, sn_h * se + cs_h * so], dim=-1)
                        else:
                            se, so = (s_new[..., 0::2], s_new[..., 1::2])
                            s_read = torch.empty_like(s_new)
                            s_read[..., 0::2] = cs_h * se - sn_h * so
                            s_read[..., 1::2] = sn_h * se + cs_h * so
                    else:
                        s_read = s_new
                    C_t = C_tok[:, t, h // hpg_c] if bc_per_head else C_tok[:, t]
                    if has_rot:
                        if C_per_o:
                            C_read = torch.empty_like(C_t)
                            for oi in range(O):
                                C_read[:, :, oi] = rot_pairs(C_t[:, :, oi], phi_t, _ph)
                            C_t = C_read
                        else:
                            C_t = rot_pairs(C_t, phi_t, _ph)
                    if C_per_o:
                        for oi in range(O):
                            C_oi = C_t[:, :, oi]
                            if has_gamma:
                                C_oi = C_oi + gamma[h, :, oi].unsqueeze(0)
                            y_hc = torch.sum(C_oi * s_read, dim=-1)
                            y_t[:, gi, oi, ci] += y_hc
                    else:
                        C_rd = C_t
                        if has_gamma:
                            C_rd = C_rd + gamma[h, :, 0].unsqueeze(0)
                        y_hc = torch.sum(C_rd * s_read, dim=-1)
                        y_t[:, gi, ci] += y_hc
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)
    return (y, state)

def _rot_fold_pairs(v: torch.Tensor, ang: torch.Tensor, pair_half: bool=False) -> torch.Tensor:
    c, s = (torch.cos(ang), torch.sin(ang))
    tailed = v.dim() != ang.dim()
    ax = -2 if tailed else -1
    if pair_half:
        K = v.shape[ax] // 2
        e = v.narrow(ax, 0, K)
        o = v.narrow(ax, K, K)
    else:
        e = v[..., 0::2] if not tailed else v[..., 0::2, :]
        o = v[..., 1::2] if not tailed else v[..., 1::2, :]
    if tailed:
        c, s = (c[..., None], s[..., None])
    ep, op = (c * e + s * o, -s * e + c * o)
    if pair_half:
        return torch.cat([ep, op], dim=ax)
    stack_dim = e.dim() - (0 if not tailed else 1)
    out = torch.stack([ep, op], dim=stack_dim)
    return out.reshape(v.shape)

def unified_ssd_quadratic(x_in: torch.Tensor, delta: torch.Tensor, B_tok: torch.Tensor, C_tok: torch.Tensor, a_log: torch.Tensor, *, beta=None, gamma=None, theta=None, theta_h=None, rot_bias=None, bias_pre_rot=True, delta_w=None, x0=None, gate_z=None, d_skip=None, bc_per_head: bool=False, pair_layout: str='interleaved') -> tuple[torch.Tensor, torch.Tensor]:
    Bsz, H, G, R, L = x_in.shape
    f = torch.float64 if x_in.dtype == torch.float64 else torch.float32
    hax = 1 if bc_per_head else 0
    b_per_r = B_tok.dim() == 4 + hax
    c_per_o = C_tok.dim() == 4 + hax
    N = B_tok.shape[2 + hax] if b_per_r else B_tok.shape[-1]
    O = C_tok.shape[-1] if c_per_o else 1
    Gb = B_tok.shape[2] if bc_per_head else 1
    Gc = C_tok.shape[2] if bc_per_head else 1
    hpg_b, hpg_c = (H // Gb, H // Gc)
    x_f = x_in.to(f)
    delta_f = delta.to(f)
    dw = delta_f if delta_w is None else delta_w.to(f)
    cs = delta_f.cumsum(dim=-1)
    Bv = B_tok.to(f)
    Cv = C_tok.to(f)
    if not bc_per_head:
        Bv, Cv = (Bv.unsqueeze(2), Cv.unsqueeze(2))
    if not b_per_r:
        Bv = Bv.unsqueeze(-1)
    if not c_per_o:
        Cv = Cv.unsqueeze(-1)
    if pair_layout not in ('interleaved', 'half'):
        raise ValueError(f"pair_layout must be 'interleaved' or 'half', got {pair_layout!r}")
    _ph = pair_layout == 'half'
    theta_ph = theta is not None and theta.dim() == 4
    if bias_pre_rot and rot_bias is not None:
        raise NotImplementedError('bias_pre_rot combined with rot_bias (the retired write-side angle offset) is refused -- rot_bias has no bias_pre_rot fold order of its own')
    if bias_pre_rot and theta is None and (theta_h is not None):
        raise NotImplementedError('bias_pre_rot with theta_h and no theta is a real (non-identity) READ-side rotation with no defined bias_pre_rot fold order -- pass theta too, or drop theta_h, or use bias_pre_rot=False')
    if theta is not None and (not theta_ph) and (not bias_pre_rot):
        th = theta.to(f)[:, :, None, :]
        Bv = _rot_fold_pairs(Bv, th, _ph)
        Cv = _rot_fold_pairs(Cv, th, _ph)
    causal = torch.arange(L, device=x_in.device)[:, None] >= torch.arange(L, device=x_in.device)[None, :]
    y = x_in.new_zeros((Bsz, H, O, G, L), dtype=f)
    S_fin = x_in.new_zeros((Bsz, H, G, N), dtype=f)
    for h in range(H):
        a_val = -torch.exp(a_log[h].to(f))
        Bh = Bv[:, :, h // hpg_b]
        Ch = Cv[:, :, h // hpg_c]
        if bias_pre_rot:
            if beta is not None:
                Bh = Bh + beta[h].to(f)
            if gamma is not None:
                Ch = Ch + gamma[h].to(f)
        if theta_ph:
            th_h = theta.to(f)[:, :, h]
            Bh = _rot_fold_pairs(Bh, th_h, _ph)
            Ch = _rot_fold_pairs(Ch, th_h, _ph)
        elif theta is not None and bias_pre_rot:
            th_s = theta.to(f)
            Bh = _rot_fold_pairs(Bh, th_s, _ph)
            Ch = _rot_fold_pairs(Ch, th_s, _ph)
        if rot_bias is not None:
            Bh = _rot_fold_pairs(Bh, rot_bias[h].to(f), _ph)
        if beta is not None and (not bias_pre_rot):
            Bh = Bh + beta[h].to(f)
        if theta_h is not None:
            Ch = _rot_fold_pairs(Ch, theta_h[h].to(f), _ph)
        if gamma is not None and (not bias_pre_rot):
            gam = gamma[h].to(f)
            if theta_h is not None:
                gam = _rot_fold_pairs(gam, theta_h[h].to(f), _ph)
            Ch = Ch + gam
        cs_h = cs[:, h]
        dec = torch.exp(a_val * (cs_h[:, :, None] - cs_h[:, None, :]))
        dec = dec * causal
        score = torch.einsum('btno,bsnr->botsr', Ch, Bh)
        xs = x_f[:, h] * dw[:, h][:, None, None, :]
        y_h = torch.einsum('botsr,bts,bgrs->bogt', score, dec, xs)
        f_t = torch.exp(a_val * cs_h)
        if x0 is not None:
            x0_h = x0[:, h].to(f)
            y_h = y_h + torch.einsum('btno,bt,bgn->bogt', Ch, f_t, x0_h)
        dec_end = torch.exp(a_val * (cs_h[:, -1:] - cs_h))
        S_h = torch.einsum('bsnr,bs,bgrs->bgn', Bh, dec_end, xs)
        if x0 is not None:
            S_h = S_h + f_t[:, -1, None, None] * x0_h
        y[:, h] = y_h
        S_fin[:, h] = S_h
    if d_skip is not None:
        if O != 1:
            raise NotImplementedError(f'the D-skip is the O == 1 branch (its [H, G, R] weights carry no output axis), got O={O}')
        y = y + torch.einsum('bhgrl,hgr->bhgl', x_in.to(f), d_skip.to(f))[:, :, None]
    if gate_z is not None:
        z = gate_z.to(f).view(Bsz, L, H, G).permute(0, 2, 3, 1)
        y = y * torch.nn.functional.silu(z)[:, :, None, :, :]
    if O == 1:
        return (y[:, :, 0], S_fin)
    return (y, S_fin)

def _chunk_affine_scan(alpha: torch.Tensor, gc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nc = alpha.shape[-1]
    cum_alpha = alpha
    cum_g = gc
    d = 1
    while d < nc:
        a_prev = cum_alpha[:, :nc - d]
        g_prev = cum_g[:, :, :nc - d, :]
        a_cur = cum_alpha[:, d:]
        g_cur = cum_g[:, :, d:, :]
        a_new = a_prev * a_cur
        g_new = a_cur[:, None, :, None] * g_prev + g_cur
        cum_alpha = torch.cat([cum_alpha[:, :d], a_new], dim=1)
        cum_g = torch.cat([cum_g[:, :, :d, :], g_new], dim=2)
        d *= 2
    return (cum_alpha, cum_g)

def unified_ssd_chunked(x_in: torch.Tensor, delta: torch.Tensor, B_tok: torch.Tensor, C_tok: torch.Tensor, a_log: torch.Tensor, *, beta=None, gamma=None, theta=None, theta_h=None, rot_bias=None, bias_pre_rot=True, delta_w=None, x0=None, gate_z=None, d_skip=None, bc_per_head: bool=False, pair_layout: str='interleaved', chunk_size: int=256) -> tuple[torch.Tensor, torch.Tensor]:
    Bsz, H, G, R, L = x_in.shape
    f = torch.float64 if x_in.dtype == torch.float64 else torch.float32
    hax = 1 if bc_per_head else 0
    b_per_r = B_tok.dim() == 4 + hax
    c_per_o = C_tok.dim() == 4 + hax
    N = B_tok.shape[2 + hax] if b_per_r else B_tok.shape[-1]
    O = C_tok.shape[-1] if c_per_o else 1
    Gb = B_tok.shape[2] if bc_per_head else 1
    Gc = C_tok.shape[2] if bc_per_head else 1
    hpg_b, hpg_c = (H // Gb, H // Gc)
    x_f = x_in.to(f)
    delta_f = delta.to(f)
    dw = delta_f if delta_w is None else delta_w.to(f)
    Bv = B_tok.to(f)
    Cv = C_tok.to(f)
    if not bc_per_head:
        Bv, Cv = (Bv.unsqueeze(2), Cv.unsqueeze(2))
    if not b_per_r:
        Bv = Bv.unsqueeze(-1)
    if not c_per_o:
        Cv = Cv.unsqueeze(-1)
    if pair_layout not in ('interleaved', 'half'):
        raise ValueError(f"pair_layout must be 'interleaved' or 'half', got {pair_layout!r}")
    _ph = pair_layout == 'half'
    theta_ph = theta is not None and theta.dim() == 4
    if bias_pre_rot and rot_bias is not None:
        raise NotImplementedError('bias_pre_rot combined with rot_bias (the retired write-side angle offset) is refused -- rot_bias has no bias_pre_rot fold order of its own')
    if bias_pre_rot and theta is None and (theta_h is not None):
        raise NotImplementedError('bias_pre_rot with theta_h and no theta is a real (non-identity) READ-side rotation with no defined bias_pre_rot fold order -- pass theta too, or drop theta_h, or use bias_pre_rot=False')
    if theta is not None and (not theta_ph) and (not bias_pre_rot):
        th = theta.to(f)[:, :, None, :]
        Bv = _rot_fold_pairs(Bv, th, _ph)
        Cv = _rot_fold_pairs(Cv, th, _ph)
    T = min(int(chunk_size), L) if L > 0 else 1
    nc = -(-L // T)
    L_pad = nc * T
    pad = L_pad - L
    causal_T = torch.tril(torch.ones(T, T, device=x_in.device, dtype=torch.bool))

    def _pad_len(v: torch.Tensor, dim: int) -> torch.Tensor:
        if pad == 0:
            return v
        shape = list(v.shape)
        shape[dim] = pad
        return torch.cat([v, v.new_zeros(shape)], dim=dim)
    delta_c = _pad_len(delta_f, 2).view(Bsz, H, nc, T)
    dw_c = _pad_len(dw, 2).view(Bsz, H, nc, T)
    x_c = _pad_len(x_f, 4).reshape(Bsz, H, G, R, nc, T)
    y = x_in.new_zeros((Bsz, H, O, G, L), dtype=f)
    S_fin = x_in.new_zeros((Bsz, H, G, N), dtype=f)
    for h in range(H):
        a_val = -torch.exp(a_log[h].to(f))
        Bh = Bv[:, :, h // hpg_b]
        Ch = Cv[:, :, h // hpg_c]
        if bias_pre_rot:
            if beta is not None:
                Bh = Bh + beta[h].to(f)
            if gamma is not None:
                Ch = Ch + gamma[h].to(f)
        if theta_ph:
            th_h = theta.to(f)[:, :, h]
            Bh = _rot_fold_pairs(Bh, th_h, _ph)
            Ch = _rot_fold_pairs(Ch, th_h, _ph)
        elif theta is not None and bias_pre_rot:
            th_s = theta.to(f)
            Bh = _rot_fold_pairs(Bh, th_s, _ph)
            Ch = _rot_fold_pairs(Ch, th_s, _ph)
        if rot_bias is not None:
            Bh = _rot_fold_pairs(Bh, rot_bias[h].to(f), _ph)
        if beta is not None and (not bias_pre_rot):
            Bh = Bh + beta[h].to(f)
        if theta_h is not None:
            Ch = _rot_fold_pairs(Ch, theta_h[h].to(f), _ph)
        if gamma is not None and (not bias_pre_rot):
            gam = gamma[h].to(f)
            if theta_h is not None:
                gam = _rot_fold_pairs(gam, theta_h[h].to(f), _ph)
            Ch = Ch + gam
        Bh_c = _pad_len(Bh, 1).view(Bsz, nc, T, N, R)
        Ch_c = _pad_len(Ch, 1).view(Bsz, nc, T, N, O)
        cs = delta_c[:, h].cumsum(dim=-1)
        raw = a_val * (cs[:, :, :, None] - cs[:, :, None, :])
        neg_inf = raw.new_full((), float('-inf'))
        dec_intra = torch.exp(torch.where(causal_T, raw, neg_inf))
        score = torch.einsum('bctno,bcsnr->bcotsr', Ch_c, Bh_c)
        xs_c = x_c[:, h] * dw_c[:, h][:, None, None]
        y_intra = torch.einsum('bcotsr,bcts,bgrcs->bogct', score, dec_intra, xs_c)
        dec_end = torch.exp(a_val * (cs[:, :, -1:] - cs))
        G_chunk = torch.einsum('bcsnr,bcs,bgrcs->bgcn', Bh_c, dec_end, xs_c)
        alpha_c = torch.exp(a_val * cs[:, :, -1])
        x0_h = x0[:, h].to(f) if x0 is not None else None
        cum_alpha, cum_g = _chunk_affine_scan(alpha_c, G_chunk)
        if x0_h is not None:
            s_after = cum_alpha[:, None, :, None] * x0_h[:, :, None, :] + cum_g
            s_entry0 = x0_h
        else:
            s_after = cum_g
            s_entry0 = cum_g.new_zeros(Bsz, G, N)
        s_in = torch.cat([s_entry0.unsqueeze(2), s_after[:, :, :-1, :]], dim=2)
        f_local = torch.exp(a_val * cs)
        y_state = torch.einsum('bctno,bct,bgcn->bogct', Ch_c, f_local, s_in)
        y_h = (y_intra + y_state).reshape(Bsz, O, G, L_pad)[..., :L]
        y[:, h] = y_h
        S_fin[:, h] = s_after[:, :, -1, :]
    if d_skip is not None:
        if O != 1:
            raise NotImplementedError(f'the D-skip is the O == 1 branch (its [H, G, R] weights carry no output axis), got O={O}')
        y = y + torch.einsum('bhgrl,hgr->bhgl', x_in.to(f), d_skip.to(f))[:, :, None]
    if gate_z is not None:
        z = gate_z.to(f).view(Bsz, L, H, G).permute(0, 2, 3, 1)
        y = y * torch.nn.functional.silu(z)[:, :, None, :, :]
    if O == 1:
        return (y[:, :, 0], S_fin)
    return (y, S_fin)
