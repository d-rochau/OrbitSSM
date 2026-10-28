"""Naive sequential recurrence — ground truth for unified rotation + MH-MIMO.

Unified SSM model (Score-path compatible):
  H_actual heads, each with g channels (group size).
  G = H_actual / g head-groups, each sharing delta.
  R input channels per head.  D_in = H_actual * R.
  State x in R^N per (head, channel).  N = M*k subsystem blocks.

  Per-timestep recurrence for head h, channel c (c = 0..g-1):
    x[h,c,n] = decay[h,n,t] * rotate(Q_t, x[h,c,:])_n
               + sum_r B_eff_r[t,n] * x_in[group(h), r, t]
    y[h,c,t] = <C_eff[t,:], x[h,c,:]>

  Where:
    B_eff_r[t,n] = B_tok[t,n(,r)] + beta[h, n, r]   (beta per-head, shared across g;
                                                     B_tok [B,L,N] shared or [B,L,N,R]
                                                     per-stream dynamic)
    C_eff_o[t,n] = C_tok[t,n(,o)]                    (C_tok [B,L,N] O=1, or [B,L,N,O]
                                                     per-output dynamic; no kappa)
    decay[h,n,t] = exp(a[block(n)] * delta[group(h), t])   (per-block)
                   or exp(a[h] * delta[group(h), t])        (per-head scalar)

    Decay modes (signed contract):
      DEFAULT "dyn": A_t in (B, L, H) -- the rate itself is data-dependent,
        achieved by folding a dynamic modulator m_t into delta
        (delta_eff = m_t * dt), giving decay per (batch, time, head),
        constant across N for SSD GEMM compatibility.  Mamba-3 style.
        The caller passes delta_eff as `delta` and uses the kernel's
        decoupled-write mechanism (delta_w = dt) for the drive scale.
      MODE "fix": a_log in (H,) -- fixed per-head scalar rate, time
        variation comes only from delta.  No decoupled write needed.

    rotate(Q_t, x) = blockdiag(Q_t^{1..M}) @ x      (k*k orthogonal blocks)

  Complex (omega) mode adds per-pair phase rotation to the decay.
    Pair layout is HALVES, not interleaved: pair i = (state[i], state[i+N/2])
    — first all real parts, then all imaginary parts (this is what lets the
    kernel split C/B into contiguous halves for two half-width GEMMs).
    R(omega_i * delta) is applied to each pair before magnitude decay.
"""

import torch
from typing import Optional


def blockdiag_apply(Q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply block-diagonal Q to states.

    Q: [..., M, k, k], x: [..., N] with N = M*k.
    """
    *batch, N = x.shape
    M, k = Q.shape[-3], Q.shape[-1]
    xb = x.view(*batch, M, k)
    y = torch.einsum("...mij,...mj->...mi", Q, xb)
    return y.reshape(*batch, N)


def rot_pairs(x: torch.Tensor, phi: torch.Tensor,
              pair_half: bool = False) -> torch.Tensor:
    """R(-phi) applied pairwise on adjacent pairs (2k, 2k+1).

    x: [..., N], phi: [..., N//2].  Returns [..., N].
    R(-phi) = [[cos phi, sin phi], [-sin phi, cos phi]].
    This matches _rot_t in the Triton kernel.
    """
    cs = torch.cos(phi)
    sn = torch.sin(phi)
    if pair_half:
        # PAIRHALF: pair k is (k, k + N/2), not (2k, 2k+1)
        K = x.shape[-1] // 2
        xe, xo = x[..., :K], x[..., K:]
        cs, sn = torch.cos(phi), torch.sin(phi)
        return torch.cat([cs * xe + sn * xo, -sn * xe + cs * xo], dim=-1)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = cs * xe + sn * xo
    out[..., 1::2] = -sn * xe + cs * xo
    return out


def unified_reference(
    x_in: torch.Tensor,                # [B, G, R, L]   input (G groups, R channels each)
    delta: torch.Tensor,                # [B, G, L]      discretization (per group)
    B_tok: torch.Tensor,                # [B, L, N] or [B, L, N, R]  write projection
    C_tok: torch.Tensor,                # [B, L, N] or [B, L, N, O]  read projection
    a_log: torch.Tensor,                # [M] or [H_actual] per-block or per-head decay (log-space)
    Q: torch.Tensor,                    # [B, L, M, k, k] rotation matrices
    g: int,                             # group size (channels per head-group)
    beta: Optional[torch.Tensor] = None,  # [H_actual, N, R] write bias (per-head)
    omega: Optional[torch.Tensor] = None, # [N//2]  rotation frequencies (complex mode)
    x0: Optional[torch.Tensor] = None,   # [B, H_actual, g, N] initial state
    beta_in_rotated_frame: bool = True,  # if True, beta is in Pi-rotated frame (kernel convention)
    chunk_size: Optional[int] = None,    # chunk size T — needed for per-chunk Pi reset when beta_in_rotated_frame
    bc_per_head: bool = False,           # B/C carry a GROUP axis at dim 2
    theta_rot: Optional[torch.Tensor] = None,  # [B, L, N//2] in-kernel pairwise rotation phase (HAS_ROT)
    theta_h: Optional[torch.Tensor] = None,    # [H_actual, N//2] per-head rotation offset
    gamma: Optional[torch.Tensor] = None,      # [H_actual, N, O] per-head read bias
    pair_layout: str = "interleaved",          # PAIRHALF: see `rot_pairs`
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns y: [B, G, g, L] (C_tok 3-dim) or [B, G, O, g, L] (C_tok 4-dim),
    and final state: [B, H_actual, g, N].

    Naming:
      G = number of head-groups (x_in dim 1)
      H_actual = G * num_heads_per_group  (but for this reference, each group
                 has a set of "heads" that share delta; the a_log / beta index
                 into these via h = group * heads_per_group + head_in_group)
      g = channels per group (output width per group)
      R = MIMO rank
      N = state dimension = M * k

    bc_per_head: B/C carry a GROUP axis at dim 2 — B_tok [B,L,Gb,N(,R)],
      C_tok [B,L,Gc,N(,O)] with Gb, Gc any divisor of H_actual.  Head h reads
      group  h // (H_actual // Gb)  on the B side and  h // (H_actual // Gc)
      on the C side.  Gb=1 broadcasts (the head-shared path, bit-identical);
      Gb=H_actual is fully per-head; the divisors in between are gauge GROUPS.
    """
    B, G, R, L = x_in.shape
    # With bc_per_head the GROUP axis sits at dim 2: [B,L,Gb,N(,R)].  Gb/Gc are
    # any divisor of H_actual — 1 (broadcast, must reproduce the head-shared
    # path exactly), H_actual (per-head), or a divisor in between (gauge
    # groups).  The flag is explicit because [B,L,N] and [B,L,N,R] collide.
    hax = 1 if bc_per_head else 0
    B_per_r = B_tok.dim() == 4 + hax
    C_per_o = C_tok.dim() == 4 + hax
    N = B_tok.shape[2 + hax] if B_per_r else B_tok.shape[-1]
    O = C_tok.shape[-1] if C_per_o else 1
    if B_per_r:
        assert B_tok.shape[-1] == R
    Gb = B_tok.shape[2] if bc_per_head else 1
    Gc = C_tok.shape[2] if bc_per_head else 1
    M, k = Q.shape[-3], Q.shape[-1]

    per_block_decay = (a_log.dim() == 1 and a_log.shape[0] == M)
    per_head_decay = (a_log.dim() == 1 and a_log.shape[0] != M) or a_log.dim() > 1
    has_omega = omega is not None
    has_beta = beta is not None
    has_rot = theta_rot is not None
    _ph = pair_layout == "half"
    has_theta_h = theta_h is not None
    has_gamma = gamma is not None

    H_actual = G  # one "head" per group for simplicity
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

    # Group axis: head h reads group h // (H_actual // Gb) on each side.
    if bc_per_head:
        assert H_actual % Gb == 0, f"Gb={Gb} must divide H_actual={H_actual}"
        assert H_actual % Gc == 0, f"Gc={Gc} must divide H_actual={H_actual}"
        hpg_b = H_actual // Gb   # heads per B-group
        hpg_c = H_actual // Gc   # heads per C-group

    state = x0.clone() if x0 is not None else x_in.new_zeros(B, H_actual, g, N)
    ys = []

    # Cumulative rotation Pi_t for rotated-frame beta
    Pi_t = None
    if has_beta and beta_in_rotated_frame:
        Pi_t = torch.eye(k, device=x_in.device, dtype=x_in.dtype).unsqueeze(0).unsqueeze(0).expand(B, M, k, k).clone()

    for t in range(L):
        d = delta[:, :, t]  # [B, G]
        Q_t = Q[:, t]       # [B, M, k, k]

        # Update cumulative rotation: Pi_t = Q_t @ Pi_{t-1}
        # Reset at chunk boundaries so beta stays in the per-chunk rotated frame
        if Pi_t is not None:
            chunk_boundary = (t == 0) or (chunk_size is not None and t % chunk_size == 0)
            if chunk_boundary:
                Pi_t = Q_t.clone()
            else:
                Pi_t = Q_t @ Pi_t

        y_t = x_in.new_zeros(B, G, O, g) if C_per_o else x_in.new_zeros(B, G, g)

        for gi in range(G):
            for hi in range(heads_per_group):
                h = gi * heads_per_group + hi

                d_h = d[:, gi]  # [B] — delta for this group

                # --- Decay (magnitude) ---
                if per_block_decay:
                    a_vals = -torch.exp(a_log)  # [M]
                    decay_n = torch.zeros(B, N, device=x_in.device, dtype=x_in.dtype)
                    for m_idx in range(M):
                        decay_n[:, m_idx*k:(m_idx+1)*k] = torch.exp(
                            a_vals[m_idx] * d_h
                        ).unsqueeze(-1).expand(-1, k)
                else:
                    a_val = -torch.exp(a_log[h] if a_log.numel() > 1 else a_log[0])
                    decay_n = torch.exp(a_val * d_h).unsqueeze(-1).expand(-1, N)  # [B, N]

                for ci in range(g):
                    s = state[:, h, ci, :]  # [B, N]

                    # Rotation
                    s_rot = blockdiag_apply(Q_t, s)  # [B, N]

                    # Complex mode: apply omega phase rotation BEFORE magnitude decay
                    if has_omega:
                        N2 = N // 2
                        theta = omega * d_h.unsqueeze(-1)  # [B, N//2]
                        cs = torch.cos(theta)
                        sn = torch.sin(theta)
                        s_re = s_rot[:, :N2]
                        s_im = s_rot[:, N2:]
                        s_rot_re = s_re * cs - s_im * sn
                        s_rot_im = s_re * sn + s_im * cs
                        s_rot = torch.cat([s_rot_re, s_rot_im], dim=-1)

                    # Magnitude decay
                    s_decayed = decay_n * s_rot  # [B, N]

                    # Drive: sum_r B_eff_r * x_r
                    drive = torch.zeros(B, N, device=x_in.device, dtype=x_in.dtype)
                    B_t = (B_tok[:, t, h // hpg_b] if bc_per_head
                           else B_tok[:, t])   # [B, N] or [B, N, R]

                    phi_t = theta_rot[:, t] if has_rot else None  # [B, K]

                    for r in range(R):
                        B_t_r = B_t[:, :, r] if B_per_r else B_t
                        if has_beta:
                            beta_r = beta[h, :, r]  # [N]
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

                        xr = x_in[:, gi, r, t]  # [B]
                        drive = drive + B_eff_r * (d_h * xr).unsqueeze(-1)

                    s_new = s_decayed + drive  # [B, N]
                    state[:, h, ci, :] = s_new

                    # Readout: theta_h is a per-head rotational lens on the
                    # state — R(theta_h) applied to s_new before dotting with
                    # Co.  Drive uses only phi_t (head-shared); the asymmetry
                    # is what makes theta_h non-gauge.
                    if has_theta_h:
                        phi_h = theta_h[h]  # [K]
                        cs_h = torch.cos(phi_h)
                        sn_h = torch.sin(phi_h)
                        if _ph:
                            Kh = s_new.shape[-1] // 2
                            se, so = s_new[..., :Kh], s_new[..., Kh:]
                            s_read = torch.cat(
                                [cs_h * se - sn_h * so,
                                 sn_h * se + cs_h * so], dim=-1)
                        else:
                            se, so = s_new[..., 0::2], s_new[..., 1::2]
                            s_read = torch.empty_like(s_new)
                            s_read[..., 0::2] = cs_h * se - sn_h * so
                            s_read[..., 1::2] = sn_h * se + cs_h * so
                    else:
                        s_read = s_new

                    C_t = (C_tok[:, t, h // hpg_c] if bc_per_head
                           else C_tok[:, t])   # [B, N] or [B, N, O]

                    if has_rot:
                        if C_per_o:
                            C_read = torch.empty_like(C_t)
                            for oi in range(O):
                                C_read[:, :, oi] = rot_pairs(
                                    C_t[:, :, oi], phi_t, _ph)
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

    y = torch.stack(ys, dim=-1)  # [B, G, g, L]
    return y, state


# ═══════════════════════════════════════════════════════════════════════
#  Quadratic torch reference for the A-scalar production contract
# ═══════════════════════════════════════════════════════════════════════

def _rot_fold_pairs(v: torch.Tensor, ang: torch.Tensor,
                    pair_half: bool = False) -> torch.Tensor:
    """Fold-form rotation on N pairs (matches kernel + module fold):
    e' = cos*e + sin*o ; o' = -sin*e + cos*o.

    v: [..., N] or [..., N, T]; ang broadcastable to [..., N//2] (or
    [..., N//2, 1] against the T-tailed form).

    PAIRHALF: `pair_half` selects WHICH two coordinates form
    pair k -- (2k, 2k+1) when False (the default, `pair_layout=
    "interleaved"`), (k, k + N/2) when True (`pair_layout="half"`,
    Mamba-3's rotate-half convention).  `ang` is indexed by PAIR in both,
    so the two layouts are the same map in permuted state coordinates.
    """
    c, s = torch.cos(ang), torch.sin(ang)
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
        c, s = c[..., None], s[..., None]
    ep, op = c * e + s * o, -s * e + c * o
    if pair_half:
        return torch.cat([ep, op], dim=ax)
    stack_dim = e.dim() - (0 if not tailed else 1)
    out = torch.stack([ep, op], dim=stack_dim)
    return out.reshape(v.shape)


def unified_ssd_quadratic(
    x_in: torch.Tensor,                   # [B, H, G, R, L]
    delta: torch.Tensor,                   # [B, H, L]  DECAY step
    B_tok: torch.Tensor,                   # [B, L, N(,R)] or [B, L, Gb, N(,R)]
    C_tok: torch.Tensor,                   # [B, L, N(,O)] or [B, L, Gc, N(,O)]
    a_log: torch.Tensor,                   # [H] per-head scalar
    *,
    beta=None,                             # [H, N, R]  additive write bias
    gamma=None,                            # [H, N, O]  additive read bias
    theta=None,                            # [B, L, N//2] head-shared phase
                                           # or [B, L, H, N//2] per head
    theta_h=None,                          # [H, N//2] read-side offset
    rot_bias=None,                         # [H, N//2] write-side offset
    bias_pre_rot=True,                     # beta/gamma INSIDE the phase
                                           # (Mamba-3's fold order, the
                                           # default)
    delta_w=None,                          # [B, H, L] decoupled write scale
    x0=None,                               # [B, H, G, N] initial state
    gate_z=None,                           # [B, L, H*G] SwiGLU gate
    d_skip=None,                           # [H, G, R] fused D-skip
    bc_per_head: bool = False,
    # PAIRHALF: which state coordinates form a rotation pair
    # -- "interleaved" (2k, 2k+1) or "half" (k, k + N/2).  See
    # `_rot_fold_pairs` and.
    pair_layout: str = "interleaved",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised QUADRATIC (attention-form) reference for the a_scalar
    unified_ssd contract — the independent oracle for the chunked kernel
    and the sequential step recurrence.  O(L^2) memory per head; pure
    torch, CPU-capable, differentiable.

    Semantics mirror the fused kernel exactly:
      B_eff = R(rot_bias_h)(R(theta)^T B) + beta_h
      C_eff = R(theta_h)(R(theta)^T C) + R(theta_h)(gamma_h)
      y_t   = sum_{r, s<=t} (C_eff_t^T B_eff_s) exp(a (cs_t - cs_s)) dw_s x_s
              + C_eff_t^T exp(a cs_t) x0
      S_L   = sum_{r, s} exp(a (cs_L - cs_s)) B_eff_s dw_s x_s + exp(a cs_L) x0
    with cs = INCLUSIVE cumsum(delta), dw = delta_w or delta.

    Returns y [B, H, G, L] (O == 1) or [B, H, O, G, L], state [B, H, G, N].
    """
    Bsz, H, G, R, L = x_in.shape
    f = torch.float64 if x_in.dtype == torch.float64 else torch.float32

    hax = 1 if bc_per_head else 0
    b_per_r = B_tok.dim() == 4 + hax
    c_per_o = C_tok.dim() == 4 + hax
    N = B_tok.shape[2 + hax] if b_per_r else B_tok.shape[-1]
    O = C_tok.shape[-1] if c_per_o else 1
    Gb = B_tok.shape[2] if bc_per_head else 1
    Gc = C_tok.shape[2] if bc_per_head else 1
    hpg_b, hpg_c = H // Gb, H // Gc

    x_f = x_in.to(f)
    delta_f = delta.to(f)
    dw = delta_f if delta_w is None else delta_w.to(f)
    cs = delta_f.cumsum(dim=-1)                      # [B, H, L] inclusive

    # normalise B/C to [B, L, Gx, N, R/O] fp
    Bv = B_tok.to(f)
    Cv = C_tok.to(f)
    if not bc_per_head:
        Bv, Cv = Bv.unsqueeze(2), Cv.unsqueeze(2)
    if not b_per_r:
        Bv = Bv.unsqueeze(-1)
    if not c_per_o:
        Cv = Cv.unsqueeze(-1)

    # theta [B, L, K] is HEAD-SHARED and folds once, here; theta
    # [B, L, H, K] is PER HEAD and folds inside the head loop,
    # on that head's own slice, which is the only difference between the
    # two laws.
    if pair_layout not in ("interleaved", "half"):
        raise ValueError(
            "pair_layout must be 'interleaved' or 'half', got "
            f"{pair_layout!r}")
    _ph = pair_layout == "half"
    theta_ph = theta is not None and theta.dim() == 4
    if bias_pre_rot and rot_bias is not None:
        raise NotImplementedError(
            "bias_pre_rot combined with rot_bias (the write-side "
            "angle offset) is refused -- rot_bias has no bias_pre_rot fold "
            "order of its own")
    if bias_pre_rot and theta is None and theta_h is not None:
        raise NotImplementedError(
            "bias_pre_rot with theta_h and no theta is a real "
            "(non-identity) READ-side rotation with no defined "
            "bias_pre_rot fold order -- pass theta too, or drop theta_h, "
            "or use bias_pre_rot=False")
    # bias_pre_rot=True with NO rotation
    # reaching the scan (theta is None, theta_h is None, rot_bias is
    # None) is a NO-OP -- the identity rotation makes "before"/"after"
    # the same map; falls through to the plain biased scan below.
    if theta is not None and not theta_ph and not bias_pre_rot:
        th = theta.to(f)[:, :, None, :]              # [B, L, 1, K]
        Bv = _rot_fold_pairs(Bv, th, _ph)
        Cv = _rot_fold_pairs(Cv, th, _ph)

    causal = (torch.arange(L, device=x_in.device)[:, None]
              >= torch.arange(L, device=x_in.device)[None, :])

    y = x_in.new_zeros((Bsz, H, O, G, L), dtype=f)
    S_fin = x_in.new_zeros((Bsz, H, G, N), dtype=f)

    for h in range(H):
        a_val = -torch.exp(a_log[h].to(f))
        Bh = Bv[:, :, h // hpg_b]                    # [B, L, N, R]
        Ch = Cv[:, :, h // hpg_c]                    # [B, L, N, O]
        if bias_pre_rot:
            # Mamba-3's fold ORDER: the biases join the RAW maps and
            # rotate WITH them -- B_eff = R(theta)^T (B + beta).  The two
            # orders differ at a nonzero angle.
            if beta is not None:
                Bh = Bh + beta[h].to(f)
            if gamma is not None:
                Ch = Ch + gamma[h].to(f)
        if theta_ph:                                 # PER-HEAD phase
            th_h = theta.to(f)[:, :, h]              # [B, L, K]
            Bh = _rot_fold_pairs(Bh, th_h, _ph)
            Ch = _rot_fold_pairs(Ch, th_h, _ph)
        elif theta is not None and bias_pre_rot:
            # head-SHARED phase under Mamba-3's fold order: the biases
            # (added above) must rotate WITH the maps, so the fold runs
            # here, after them
            th_s = theta.to(f)                       # [B, L, K]
            Bh = _rot_fold_pairs(Bh, th_s, _ph)
            Ch = _rot_fold_pairs(Ch, th_s, _ph)
        if rot_bias is not None:
            Bh = _rot_fold_pairs(Bh, rot_bias[h].to(f), _ph)
        if beta is not None and not bias_pre_rot:
            Bh = Bh + beta[h].to(f)
        if theta_h is not None:
            Ch = _rot_fold_pairs(Ch, theta_h[h].to(f), _ph)
        if gamma is not None and not bias_pre_rot:
            gam = gamma[h].to(f)                     # [N, O] (N-then-tail)
            if theta_h is not None:
                gam = _rot_fold_pairs(gam, theta_h[h].to(f), _ph)
            Ch = Ch + gam

        cs_h = cs[:, h]                              # [B, L]
        dec = torch.exp(a_val * (cs_h[:, :, None] - cs_h[:, None, :]))
        dec = dec * causal                           # [B, L(t), L(s)]

        score = torch.einsum("btno,bsnr->botsr", Ch, Bh)   # [B,O,T,S,R]
        xs = x_f[:, h] * dw[:, h][:, None, None, :]        # [B,G,R,L]
        y_h = torch.einsum("botsr,bts,bgrs->bogt",
                           score, dec, xs)                 # [B,O,G,L]

        f_t = torch.exp(a_val * cs_h)                      # [B, L]
        if x0 is not None:
            x0_h = x0[:, h].to(f)                          # [B, G, N]
            y_h = y_h + torch.einsum(
                "btno,bt,bgn->bogt", Ch, f_t, x0_h)

        dec_end = torch.exp(a_val * (cs_h[:, -1:] - cs_h))  # [B, L]
        S_h = torch.einsum("bsnr,bs,bgrs->bgn", Bh, dec_end, xs)
        if x0 is not None:
            S_h = S_h + f_t[:, -1, None, None] * x0_h
        y[:, h] = y_h
        S_fin[:, h] = S_h

    if d_skip is not None:
        # the D-skip: it is part of y_pre, so it lands BEFORE
        # the gate.  O == 1 only, exactly as the fused kernel (the
        # weights carry no output axis).
        if O != 1:
            raise NotImplementedError(
                "the D-skip is the O == 1 branch (its [H, G, R] weights "
                f"carry no output axis), got O={O}")
        y = y + torch.einsum("bhgrl,hgr->bhgl",
                             x_in.to(f), d_skip.to(f))[:, :, None]
    if gate_z is not None:
        z = gate_z.to(f).view(Bsz, L, H, G).permute(0, 2, 3, 1)  # [B,H,G,L]
        y = y * torch.nn.functional.silu(z)[:, :, None, :, :]

    if O == 1:
        return y[:, :, 0], S_fin
    return y, S_fin


# ═══════════════════════════════════════════════════════════════════════
#  Chunked (Score-path) torch backend for the A-scalar production contract
# ═══════════════════════════════════════════════════════════════════════
# The quadratic form above forms its decay factor as exp(a*(D_i - D_j)) with D = GLOBAL inclusive cumsum(delta) over
# the WHOLE sequence.  The exponent is difference-first so it never
# overflows, but D_i and D_j both grow with L, so their fp32 difference
# loses precision to cancellation as u*|a|*sum(delta) grows -- a documented
# long-L limitation of `backend="torch"`.  This section replaces that
# quadratic evaluation with the SAME chunked association the fused CUDA
# kernel uses (`kernels/ssd_mimo/{fwd_kernel,state_passing}.py`):
#
#   1. delta is cumsum'd PER CHUNK -- `delta.view(B, H, nc, T).cumsum(-1)`
#      resets at every chunk boundary because cumsum runs along the
#      chunk-local T axis -- so every exponent this function forms is
#      bounded by |a| * T * max(delta), independent of L.
#   2. within a chunk, the O(T^2) causal attention form (same math as the
#      quadratic form, restricted to one chunk) gives the intra-chunk
#      output and the chunk's own state contribution (`G_c`, `alpha_c`
#      below -- the amount and decay this chunk would apply starting from
#      an empty incoming state).
#   3. the state ENTERING each chunk is the composition of every earlier
#      chunk's (alpha, G) pair.  Because the A-scalar contract has no Pi
#      rotation (alpha is a plain scalar per (batch, head), constant over
#      N and the channel axis G), that composition is an O(log nc)
#      Hillis-Steele parallel scan over affine maps `v -> alpha_c*v + G_c`
#      (`_chunk_affine_scan`) -- the chunk-level analogue of a
#      pscan, and the same association `state_passing.py`'s in-kernel
#      chunk loop runs sequentially in registers.
#   4. each chunk's output ALSO reads the state entering it, decayed
#      chunk-locally to every position in the chunk.
#
# At `chunk_size >= L` there is exactly one chunk and this reduces to the
# quadratic form's math exactly (the chunk-local cumsum IS the global one).
# `unified_ssd_quadratic` is kept as a test oracle / short-L cross-check
# (three independent formulations score the CUDA kernel -- see
# the SSD-MIMO oracle notes of the kernel tests); this function is the
# only one of the two `backend="torch"` calls into.

def _chunk_affine_scan(
    alpha: torch.Tensor, gc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive O(log nc) Hillis-Steele scan of per-chunk affine maps.

    Recurrence being scanned (index c = 0..nc-1, `carry` = the value
    entering chunk 0, folded in by the caller AFTER this returns):
        S_after[c] = alpha[c] * S_after[c-1] + gc[c],   S_after[-1] := carry

    alpha : [B, nc]     (scalar per batch, chunk -- broadcasts over every
                         other axis of gc; the A-scalar contract has no Pi
                         rotation, so composing two chunks' maps needs only
                         a scalar product, not a matrix)
    gc    : [B, G, nc, N]  (G = the channel axis; N = state width)

    Returns (cum_alpha, cum_g), both INCLUSIVE-scanned along the chunk
    axis, such that for every c:
        S_after[c] = cum_alpha[c] * carry + cum_g[c]

    log2(nc) doubling rounds, each one vectorised tensor op (no python
    loop over positions) -- a chunk-level pscan: nc = L / chunk_size, so
    this is O(log(L/T)) = O(log L) up to the constant log T.
    """
    nc = alpha.shape[-1]
    cum_alpha = alpha
    cum_g = gc
    d = 1
    while d < nc:
        a_prev = cum_alpha[:, :nc - d]
        g_prev = cum_g[:, :, :nc - d, :]
        a_cur = cum_alpha[:, d:]
        g_cur = cum_g[:, :, d:, :]
        # combine(prev, cur) = (cur.A * prev.A, cur.A * prev.B + cur.B)
        # -- prev is the OLDER (earlier) segment, cur the newer one, so
        # cur's alpha decays prev's already-accumulated additive term.
        a_new = a_prev * a_cur
        g_new = a_cur[:, None, :, None] * g_prev + g_cur
        cum_alpha = torch.cat([cum_alpha[:, :d], a_new], dim=1)
        cum_g = torch.cat([cum_g[:, :, :d, :], g_new], dim=2)
        d *= 2
    return cum_alpha, cum_g


def unified_ssd_chunked(
    x_in: torch.Tensor,                   # [B, H, G, R, L]
    delta: torch.Tensor,                   # [B, H, L]  DECAY step
    B_tok: torch.Tensor,                   # [B, L, N(,R)] or [B, L, Gb, N(,R)]
    C_tok: torch.Tensor,                   # [B, L, N(,O)] or [B, L, Gc, N(,O)]
    a_log: torch.Tensor,                   # [H] per-head scalar
    *,
    beta=None,                             # [H, N, R]  additive write bias
    gamma=None,                            # [H, N, O]  additive read bias
    theta=None,                            # [B, L, N//2] head-shared phase
                                           # or [B, L, H, N//2] per head
    theta_h=None,                          # [H, N//2] read-side offset
    rot_bias=None,                         # [H, N//2] write-side offset
    bias_pre_rot=True,                     # beta/gamma INSIDE the phase
                                           # (Mamba-3's fold order, the
                                           # default)
    delta_w=None,                          # [B, H, L] decoupled write scale
    x0=None,                               # [B, H, G, N] initial state
    gate_z=None,                           # [B, L, H*G] SwiGLU gate
    d_skip=None,                           # [H, G, R] fused D-skip
    bc_per_head: bool = False,
    # PAIRHALF: see `_rot_fold_pairs`.
    pair_layout: str = "interleaved",
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked SSD (Score-path) torch backend -- the ``backend="torch"``
    production route for the a_scalar unified_ssd contract.

    Identical inputs/outputs/feature coverage to ``unified_ssd_quadratic``
    (same preprocessing: B_eff/C_eff fold theta/rot_bias/beta/theta_h/gamma
    exactly as there), but every decay exponent is formed CHUNK-LOCALLY
    (see the module section header above) instead of over the full L.
    Precision against the fp64 recurrence is therefore a property of
    ``chunk_size`` alone, not of L. CPU-capable, differentiable.

    Returns y [B, H, G, L] (O == 1) or [B, H, O, G, L], state [B, H, G, N].
    """
    Bsz, H, G, R, L = x_in.shape
    f = torch.float64 if x_in.dtype == torch.float64 else torch.float32

    hax = 1 if bc_per_head else 0
    b_per_r = B_tok.dim() == 4 + hax
    c_per_o = C_tok.dim() == 4 + hax
    N = B_tok.shape[2 + hax] if b_per_r else B_tok.shape[-1]
    O = C_tok.shape[-1] if c_per_o else 1
    Gb = B_tok.shape[2] if bc_per_head else 1
    Gc = C_tok.shape[2] if bc_per_head else 1
    hpg_b, hpg_c = H // Gb, H // Gc

    x_f = x_in.to(f)
    delta_f = delta.to(f)
    dw = delta_f if delta_w is None else delta_w.to(f)

    # normalise B/C to [B, L, Gx, N, R/O] fp (identical to the quadratic form)
    Bv = B_tok.to(f)
    Cv = C_tok.to(f)
    if not bc_per_head:
        Bv, Cv = Bv.unsqueeze(2), Cv.unsqueeze(2)
    if not b_per_r:
        Bv = Bv.unsqueeze(-1)
    if not c_per_o:
        Cv = Cv.unsqueeze(-1)

    # theta [B, L, K] is HEAD-SHARED and folds once, here; theta
    # [B, L, H, K] is PER HEAD and folds inside the head loop,
    # on that head's own slice, which is the only difference between the
    # two laws.
    if pair_layout not in ("interleaved", "half"):
        raise ValueError(
            "pair_layout must be 'interleaved' or 'half', got "
            f"{pair_layout!r}")
    _ph = pair_layout == "half"
    theta_ph = theta is not None and theta.dim() == 4
    if bias_pre_rot and rot_bias is not None:
        raise NotImplementedError(
            "bias_pre_rot combined with rot_bias (the write-side "
            "angle offset) is refused -- rot_bias has no bias_pre_rot fold "
            "order of its own")
    if bias_pre_rot and theta is None and theta_h is not None:
        raise NotImplementedError(
            "bias_pre_rot with theta_h and no theta is a real "
            "(non-identity) READ-side rotation with no defined "
            "bias_pre_rot fold order -- pass theta too, or drop theta_h, "
            "or use bias_pre_rot=False")
    # bias_pre_rot=True with NO rotation
    # reaching the scan (theta is None, theta_h is None, rot_bias is
    # None) is a NO-OP -- the identity rotation makes "before"/"after"
    # the same map; falls through to the plain biased scan below.
    if theta is not None and not theta_ph and not bias_pre_rot:
        th = theta.to(f)[:, :, None, :]              # [B, L, 1, K]
        Bv = _rot_fold_pairs(Bv, th, _ph)
        Cv = _rot_fold_pairs(Cv, th, _ph)

    # ── chunk geometry: T <= chunk_size, T == L when L is shorter (so the
    # single-chunk case is exactly the quadratic form's global cumsum) ──
    T = min(int(chunk_size), L) if L > 0 else 1
    nc = -(-L // T)                                   # ceil div
    L_pad = nc * T
    pad = L_pad - L
    causal_T = torch.tril(torch.ones(T, T, device=x_in.device, dtype=torch.bool))

    def _pad_len(v: torch.Tensor, dim: int) -> torch.Tensor:
        if pad == 0:
            return v
        shape = list(v.shape)
        shape[dim] = pad
        return torch.cat([v, v.new_zeros(shape)], dim=dim)

    delta_c = _pad_len(delta_f, 2).view(Bsz, H, nc, T)         # [B,H,nc,T]
    dw_c = _pad_len(dw, 2).view(Bsz, H, nc, T)
    x_c = _pad_len(x_f, 4).reshape(Bsz, H, G, R, nc, T)         # [B,H,G,R,nc,T]

    y = x_in.new_zeros((Bsz, H, O, G, L), dtype=f)
    S_fin = x_in.new_zeros((Bsz, H, G, N), dtype=f)

    for h in range(H):
        a_val = -torch.exp(a_log[h].to(f))
        Bh = Bv[:, :, h // hpg_b]                    # [B, L, N, R]
        Ch = Cv[:, :, h // hpg_c]                    # [B, L, N, O]
        if bias_pre_rot:
            # Mamba-3's fold ORDER: the biases join the RAW maps and
            # rotate WITH them -- B_eff = R(theta)^T (B + beta).  The two
            # orders differ at a nonzero angle.
            if beta is not None:
                Bh = Bh + beta[h].to(f)
            if gamma is not None:
                Ch = Ch + gamma[h].to(f)
        if theta_ph:                                 # PER-HEAD phase
            th_h = theta.to(f)[:, :, h]              # [B, L, K]
            Bh = _rot_fold_pairs(Bh, th_h, _ph)
            Ch = _rot_fold_pairs(Ch, th_h, _ph)
        elif theta is not None and bias_pre_rot:
            # head-SHARED phase under Mamba-3's fold order: the biases
            # (added above) must rotate WITH the maps, so the fold runs
            # here, after them
            th_s = theta.to(f)                       # [B, L, K]
            Bh = _rot_fold_pairs(Bh, th_s, _ph)
            Ch = _rot_fold_pairs(Ch, th_s, _ph)
        if rot_bias is not None:
            Bh = _rot_fold_pairs(Bh, rot_bias[h].to(f), _ph)
        if beta is not None and not bias_pre_rot:
            Bh = Bh + beta[h].to(f)
        if theta_h is not None:
            Ch = _rot_fold_pairs(Ch, theta_h[h].to(f), _ph)
        if gamma is not None and not bias_pre_rot:
            gam = gamma[h].to(f)                     # [N, O] (N-then-tail)
            if theta_h is not None:
                gam = _rot_fold_pairs(gam, theta_h[h].to(f), _ph)
            Ch = Ch + gam

        Bh_c = _pad_len(Bh, 1).view(Bsz, nc, T, N, R)  # [B,nc,T,N,R]
        Ch_c = _pad_len(Ch, 1).view(Bsz, nc, T, N, O)  # [B,nc,T,N,O]

        # chunk-local inclusive cumsum -- resets every chunk because it
        # runs along dim=-1 (T) of the (nc, T) view, not along L.
        cs = delta_c[:, h].cumsum(dim=-1)              # [B, nc, T]

        # ── intra-chunk quadratic form (difference-first exponent) ──────
        # For a CAUSAL pair (t >= s) cs[t] - cs[s] >= 0 (cs is a cumsum of
        # non-negative deltas), so a_val * diff <= 0 exactly -- the causal
        # entries are untouched, no clamp (numerical-stability-by-
        # construction contract: exponents are <= 0 by the arithmetic, not
        # by clipping).  For a masked (t < s) pair the same difference is
        # negative, so a_val * diff is POSITIVE and can overflow exp() to
        # +inf at plausible (a_log, delta, T) triples (e.g. a_log=2,
        # delta<=0.1, T=256 -- not an exotic corner); multiplying that by
        # the 0/1 causal mask would then read `inf * 0.0 = NaN` and poison
        # the whole chunk.  The kernel's own pattern avoids this by never
        # evaluating exp() on the overflowing value AT ALL: select -inf
        # for the masked entries BEFORE exp (exp(-inf) = 0 exactly), so
        # `torch.where`'s backward routes zero gradient to the masked
        # positions (its own -inf branch is a constant, not a function of
        # cs) while the causal branch's gradient passes through unchanged.
        raw = a_val * (cs[:, :, :, None] - cs[:, :, None, :])
        neg_inf = raw.new_full((), float("-inf"))
        dec_intra = torch.exp(torch.where(causal_T, raw, neg_inf))
                                                         # [B, nc, T(t), T(s)]

        score = torch.einsum("bctno,bcsnr->bcotsr", Ch_c, Bh_c)  # [B,nc,O,T,T,R]
        xs_c = x_c[:, h] * dw_c[:, h][:, None, None]     # [B,G,R,nc,T]
        y_intra = torch.einsum("bcotsr,bcts,bgrcs->bogct",
                               score, dec_intra, xs_c)   # [B,O,G,nc,T]

        # ── this chunk's own (alpha_c, G_c) pair, from an empty carry ───
        dec_end = torch.exp(a_val * (cs[:, :, -1:] - cs))  # [B,nc,T]
        G_chunk = torch.einsum("bcsnr,bcs,bgrcs->bgcn",
                               Bh_c, dec_end, xs_c)      # [B,G,nc,N]
        alpha_c = torch.exp(a_val * cs[:, :, -1])         # [B,nc]

        # ── state entering each chunk: O(log nc) parallel scan ──────────
        x0_h = x0[:, h].to(f) if x0 is not None else None  # [B,G,N]
        cum_alpha, cum_g = _chunk_affine_scan(alpha_c, G_chunk)
        if x0_h is not None:
            s_after = (cum_alpha[:, None, :, None] * x0_h[:, :, None, :]
                       + cum_g)                          # [B,G,nc,N]
            s_entry0 = x0_h
        else:
            s_after = cum_g
            s_entry0 = cum_g.new_zeros(Bsz, G, N)
        s_in = torch.cat([s_entry0.unsqueeze(2), s_after[:, :, :-1, :]],
                         dim=2)                           # [B,G,nc,N]

        # ── chunk-read of the entering state, decayed to each position ──
        f_local = torch.exp(a_val * cs)                   # [B,nc,T]
        y_state = torch.einsum("bctno,bct,bgcn->bogct", Ch_c, f_local, s_in)

        y_h = (y_intra + y_state).reshape(Bsz, O, G, L_pad)[..., :L]
        y[:, h] = y_h
        S_fin[:, h] = s_after[:, :, -1, :]

    if d_skip is not None:
        # the D-skip: it is part of y_pre, so it lands BEFORE
        # the gate.  O == 1 only, exactly as the fused kernel (the
        # weights carry no output axis).
        if O != 1:
            raise NotImplementedError(
                "the D-skip is the O == 1 branch (its [H, G, R] weights "
                f"carry no output axis), got O={O}")
        y = y + torch.einsum("bhgrl,hgr->bhgl",
                             x_in.to(f), d_skip.to(f))[:, :, None]
    if gate_z is not None:
        z = gate_z.to(f).view(Bsz, L, H, G).permute(0, 2, 3, 1)  # [B,H,G,L]
        y = y * torch.nn.functional.silu(z)[:, :, None, :, :]

    if O == 1:
        return y[:, :, 0], S_fin
    return y, S_fin
