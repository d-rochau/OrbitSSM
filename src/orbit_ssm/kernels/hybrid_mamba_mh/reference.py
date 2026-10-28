"""fp64 oracle for the HybridMamba-MH kernel (structure B).

Derived directly from the kernel contract, NOT ported
from the MISO `whm_ext_reference.py` -- that reference describes a different
family (R axis, p-resolved streaming gates, SHARE_SUBSYS grouping) and would
import the very structure this kernel exists to remove.

Two independent implementations are provided and must agree:

  mh_reference            explicit recurrence, loop over t          ("sequential")
  mh_reference_cumprod    cumulative-decay form, no time loop       ("torch")

They share no scan code, so agreement between them is real evidence rather than a
restatement. The kernel is then validated against `mh_reference`.

LAYOUTS -- time-last throughout:

    x           (B, H, P, L)      delta       (B, H, L)
    A_log       (H, P, N)         B_t, C_t    (B, N, L)
    g_B, g_C    (B, N, L)         a_dyn, mu   (B, N, L) or (B, H, L)
    theta       (B, K, L) or (B, H, K, L)   nu  (H, K)
    beta_*      (G, N), G | H     h_init      (B, H, P, N)
    Y           (B, H, P, L)      h_final      (B, H, P, N)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .layout import (
    DYN_SOFTPLUS, GATE_SIGMOID,
    LAW_A_DYN, LAW_A_FIX, LAW_MU,
    group_index,
)
# aliased: `gate_act` and `dyn_act` are also the names of mh_reference's own
# parameters, which shadow the imports inside the body.
from .layout import dyn_rate as _dyn_rate
from .layout import gate_act as _gate_act

DT = torch.float64


def _f(t):
    return None if t is None else t.to(DT)


def _expand_hgroup(beta, H):
    """(G,N) static, G | H  ->  (H,N).  Contiguous grouping g(h) = h*G/H."""
    if beta is None:
        return None
    G = beta.shape[0]
    return beta[group_index(H, G, device=beta.device)]


def _z_gate(Y, z):
    """The SwiGLU OUTPUT gate, `Y * silu(z)`.  `z` is channel-major, x-shaped.

    Applied HERE, at the end, where the module's own epilogue applied it --
    the fast path fuses it into the scan's store, and an oracle that shares
    the fast path's fusion is not an oracle (the rule K11's D skip set).
    It is on Y and on Y alone: `h_final` is the ungated carry, so a chunked
    caller resumes from a state the gate never touched.
    """
    if z is None:
        return Y
    return Y * (_f(z) * torch.sigmoid(_f(z)))


def _row_broadcast(t, B, H, N, L, res=None):
    """(B,N,L) -> (B,H,N,L) view;  (B,H,L) -> (B,H,1,L) view.

    Both permitted resolutions of a_dyn/mu, never both.
    `res` in {"n","h"} disambiguates the N == H case; otherwise inferred.
    """
    if t is None:
        return None
    if res is None:
        if N == H:
            raise ValueError(
                "a_dyn/mu resolution is ambiguous when N == H; pass "
                "a_dyn_res=/mu_res= 'n' or 'h' explicitly")
        res = "n" if t.shape[1] == N else "h"
    return t.unsqueeze(1).expand(B, H, N, L) if res == "n" else t.unsqueeze(2)


# ---------------------------------------------------------------------------
# shared front end: everything upstream of the recurrence
# ---------------------------------------------------------------------------

def _prepare(
    x, delta, A_log, B_t, C_t, *,
    delta_w=None, delta_softplus=True,
    row_law=None, row_rot=None,
    a_dyn=None, mu=None, mu_gain=None,
    a_dyn_res=None, mu_res=None,
    dyn_act=DYN_SOFTPLUS, dyn_eps=0.0,
    g_B=None, g_C=None, gate_act=GATE_SIGMOID,
    theta=None, nu=None, theta_delta=False, theta_h=None,
    beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None,
    delta_bias=None, theta_init=None, mu_init=None,
):
    B, H, P, L = x.shape
    N = A_log.shape[2]
    K = N // 2
    dev = x.device

    x, delta, A_log, B_t, C_t = _f(x), _f(delta), _f(A_log), _f(B_t), _f(C_t)
    delta_w, a_dyn, mu = _f(delta_w), _f(a_dyn), _f(mu)
    mu_gain = _f(mu_gain)
    g_B, g_C, theta, nu = _f(g_B), _f(g_C), _f(theta), _f(nu)
    theta_h = _f(theta_h)
    beta_a, beta_b, beta_c = _f(beta_a), _f(beta_b), _f(beta_c)
    beta_gB, beta_gC = _f(beta_gB), _f(beta_gC)
    delta_bias, theta_init, mu_init = _f(delta_bias), _f(theta_init), _f(mu_init)

    if row_law is None:
        row_law = torch.full((N,), LAW_A_FIX, dtype=torch.int32, device=dev)
    if row_rot is None:
        # Match the public scan: supplied angles rotate all rows unless
        # an explicit selector restricts them.
        row_rot = torch.full((N,), theta is not None or nu is not None,
                             dtype=torch.bool, device=dev)

    ba = _expand_hgroup(beta_a, H)                                  # (H,N)
    bb = _expand_hgroup(beta_b, H)
    bc = _expand_hgroup(beta_c, H)
    bgB = _expand_hgroup(beta_gB, H)
    bgC = _expand_hgroup(beta_gC, H)

    # --- Delta_eff (B,H,L) ------------------------------------------------
    d_raw = delta if delta_bias is None else delta + delta_bias.view(1, H, 1)
    Delta_eff = F.softplus(d_raw) if delta_softplus else d_raw

    # --- Delta_write ------------------------------------------------------
    if delta_w is None:
        Delta_w = Delta_eff.unsqueeze(2)                            # (B,H,1,L)
    else:
        dw = F.softplus(delta_w) if delta_softplus else delta_w
        Delta_w = dw.unsqueeze(2) if dw.dim() == 3 else dw          # (B,H,{1,P},L)

    # --- static decay coefficients (H,P,N) --------------------------------
    a_off = 0.0 if ba is None else ba.view(H, 1, N)
    A_nat = -torch.exp(A_log + a_off)                               # (H,P,N)

    # Separable log-decay:  log a_bar = c1*s1 + c2*s2
    # with c static in (h,p,n) and s streaming and p-FREE. The per-row law byte
    # therefore costs only a `where` on each of the four factors.
    is_mu = (row_law == LAW_MU).view(1, 1, N)
    is_dyn = (row_law == LAW_A_DYN).view(1, 1, N)

    c1 = A_nat                                                      # (H,P,N)
    c2 = torch.zeros_like(c1) if mu_gain is None else torch.where(
        is_mu, mu_gain, torch.zeros_like(mu_gain))

    # --- streaming p-free scalars ----------------------------------------
    De = Delta_eff.unsqueeze(2)                                     # (B,H,1,L)
    s1 = De.expand(B, H, N, L).clone()
    if a_dyn is not None:
        rate = _dyn_rate(a_dyn, dyn_act, dyn_eps)
        s1 = torch.where(is_dyn.unsqueeze(-1),
                         De * _row_broadcast(rate, B, H, N, L, a_dyn_res), s1)

    if mu is not None:
        mu_b = _row_broadcast(mu, B, H, N, L, mu_res).expand(B, H, N, L)
        if mu_init is None:
            mu_prev0 = torch.zeros_like(mu_b[..., :1])
        else:
            mi = mu_init
            mi = mi.view(B, 1, N, 1) if mi.shape[1] == N else mi.view(B, H, 1, 1)
            mu_prev0 = mi.expand(B, H, N, 1)
        s2 = mu_b - torch.cat([mu_prev0, mu_b[..., :-1]], dim=-1)   # telescoping
        s2 = torch.where(is_mu.unsqueeze(-1), s2, torch.zeros_like(s2))
    else:
        s2 = torch.zeros_like(s1)

    # --- rotation angle: additive INSIDE the cumsum -----
    if theta is not None or nu is not None:
        inc = torch.zeros(B, H, K, L, dtype=DT, device=dev)
        if theta is not None:
            # (B,K,L) head-shared, or (B,H,K,L) under theta_per_head
            th = theta if theta.dim() == 4 else theta.unsqueeze(1)
            if theta_delta:
                # SSD law: the per-token increment is the head's
                # effective step times the feature omega tanh(q)
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
            # static READ-side offset: C rotates by Theta + theta_h[h,k]
            ThC = Theta + theta_h.view(1, H, K, 1)
            cosC, sinC = torch.cos(ThC), torch.sin(ThC)
        else:
            cosC, sinC = cosT, sinT
    else:
        cosT = sinT = cosC = sinC = None

    # --- fold: norm (module-side) -> bias -> rotate -----
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

    # --- gates: applied AFTER the rotation
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

    return dict(
        B=B, H=H, P=P, N=N, L=L, x=x,
        c1=c1, c2=c2, s1=s1, s2=s2,
        Delta_w=Delta_w, w_n=Bh, r_n=Ch,
    )


def _rotate_pairs(v, cosT, sinT):
    """R(Theta) acting on rows (2k, 2k+1).  v: (B,H,N,L), cos/sin: (B,H,K,L).

    Sign convention: the kernel rotates by R(+Theta); the
    contract's R(-Theta)^T is its transpose, i.e. the same map.
    """
    Bs, H, N, L = v.shape
    vp = v.reshape(Bs, H, N // 2, 2, L)
    u, w = vp[:, :, :, 0], vp[:, :, :, 1]
    ru = u * cosT - w * sinT
    rw = u * sinT + w * cosT
    return torch.stack((ru, rw), dim=3).reshape(Bs, H, N, L)


# ---------------------------------------------------------------------------
# implementation 1: explicit recurrence  ("sequential" oracle)
# ---------------------------------------------------------------------------

def mh_reference(x, delta, A_log, B_t, C_t, *, h_init=None,
                 n_off=0, n_rows=None, d_skip=None, z=None, trap_lam=None, write_prev=None, **kw):
    """Explicit per-timestep recurrence. The definition of correct.

        h_t[p,n] = a_bar_t[p,n] * h_{t-1}[p,n] + x_t[p] * w_t[n]
        y_t[p]   = sum_n r_t[n] * h_t[p,n]
    """
    st = _prepare(x, delta, A_log, B_t, C_t, **kw)
    B, H, P, N, L = st["B"], st["H"], st["P"], st["N"], st["L"]
    if n_rows is None:
        n_rows = N - n_off
    sl = slice(n_off, n_off + n_rows)

    xh = st["x"]                                        # (B,H,P,L)
    c1, c2 = st["c1"][..., sl], st["c2"][..., sl]       # (H,P,nr)
    s1, s2 = st["s1"][..., sl, :], st["s2"][..., sl, :]  # (B,H,nr,L)
    w_n, r_n = st["w_n"][..., sl, :], st["r_n"][..., sl, :]
    Dw = st["Delta_w"]                                  # (B,H,{1,P},L)

    h = torch.zeros(B, H, P, N, dtype=DT, device=x.device) if h_init is None \
        else _f(h_init).clone()
    state = h[..., sl]                                  # (B,H,P,nr)
    previous = torch.zeros_like(state) if write_prev is None else _f(write_prev)[..., sl]
    ys = []
    for t in range(L):
        log_a = (c1.unsqueeze(0) * s1[..., t].unsqueeze(2)
                 + c2.unsqueeze(0) * s2[..., t].unsqueeze(2))
        a_bar = torch.exp(log_a)
        drive = (Dw[..., t] * xh[..., t]).unsqueeze(-1) * w_n[..., t].unsqueeze(2)
        if trap_lam is not None:
            lam = _f(trap_lam)[..., t, None, None]
            raw_write = xh[..., t, None] * w_n[..., t].unsqueeze(2)
            drive = Dw[..., t, None] * (lam * raw_write + (1 - lam) * a_bar * previous)
            previous = raw_write
        state = a_bar * state + drive
        ys.append((r_n[..., t].unsqueeze(2) * state).sum(-1))
    Y = torch.stack(ys, dim=-1)                          # (B,H,P,L)
    if d_skip is not None:
        # contract clause: y = SSM(x) + D_h x, per head and broadcast over
        # the head's P channels -- the fused kernels add it at their store
        Y = Y + _f(d_skip).view(1, H, 1, 1) * xh
    Y = _z_gate(Y, z)

    h_out = h.clone()
    h_out[..., sl] = state                               # rows outside slice kept
    return Y, h_out


# ---------------------------------------------------------------------------
# implementation 2: cumulative-decay form  ("torch" reference)
# ---------------------------------------------------------------------------

def mh_reference_cumprod(x, delta, A_log, B_t, C_t, *, h_init=None,
                         n_off=0, n_rows=None, d_skip=None, z=None, trap_lam=None, write_prev=None, **kw):
    """Closed form, no time loop:

        Abar(t0->t) = exp(c1*S1_t + c2*S2_t)
        h_t         = Abar_t * ( h_{-1} + sum_{s<=t} drive_s / Abar_s )

    Structurally independent of `mh_reference`. Uses the SEPARABILITY of the
    log-decay directly: the prefix sums S1/S2 are p-free,
    which is exactly the property the kernel exploits for reuse over P.

    Numerically this divides by a decaying quantity, so it is fp64-only and for
    short L; it exists as a cross-check on the recurrence, not as an oracle.
    """
    st = _prepare(x, delta, A_log, B_t, C_t, **kw)
    B, H, P, N, L = st["B"], st["H"], st["P"], st["N"], st["L"]
    if n_rows is None:
        n_rows = N - n_off
    sl = slice(n_off, n_off + n_rows)

    xh = st["x"]
    c1, c2 = st["c1"][..., sl], st["c2"][..., sl]
    S1 = torch.cumsum(st["s1"][..., sl, :], dim=-1)      # (B,H,nr,L)  p-free
    S2 = torch.cumsum(st["s2"][..., sl, :], dim=-1)
    w_n, r_n = st["w_n"][..., sl, :], st["r_n"][..., sl, :]
    Dw = st["Delta_w"]

    log_A = (c1.view(1, H, P, n_rows, 1) * S1.unsqueeze(2)
             + c2.view(1, H, P, n_rows, 1) * S2.unsqueeze(2))       # (B,H,P,nr,L)
    A_cum = torch.exp(log_A)

    drive = (Dw * xh).unsqueeze(3) * w_n.unsqueeze(2)               # (B,H,P,nr,L)
    if trap_lam is not None:
        # Independent global form: integrate each endpoint in the inverse
        # transition frame, using the prefix at t-1 for the previous write.
        raw_write = xh.unsqueeze(3) * w_n.unsqueeze(2)
        prior = (torch.zeros_like(raw_write[..., :1]) if write_prev is None
                 else _f(write_prev)[..., sl, None])
        previous = torch.cat((prior, raw_write[..., :-1]), -1)
        prefix_previous = torch.cat((torch.ones_like(A_cum[..., :1]), A_cum[..., :-1]), -1)
        lam = _f(trap_lam)[:, :, None, None, :]
        normalized = Dw.unsqueeze(3) * (lam * raw_write / A_cum + (1 - lam) * previous / prefix_previous)
        inner = torch.cumsum(normalized, dim=-1)
    else:
        inner = torch.cumsum(drive / A_cum, dim=-1)
    if h_init is not None:
        inner = inner + _f(h_init)[..., sl].unsqueeze(-1)
    hs = A_cum * inner                                              # (B,H,P,nr,L)

    Y = (r_n.unsqueeze(2) * hs).sum(3)                              # (B,H,P,L)
    if d_skip is not None:
        Y = Y + _f(d_skip).view(1, H, 1, 1) * st["x"]
    Y = _z_gate(Y, z)

    h_out = (torch.zeros(B, H, P, N, dtype=DT, device=x.device)
             if h_init is None else _f(h_init).clone())
    h_out[..., sl] = hs[..., -1]
    return Y, h_out


# ---------------------------------------------------------------------------

def physical_oracle(x, delta, A_log, B_t, C_t, *, trap_lam, h_init=None,
                    write_prev=None, n_off=0, n_rows=None, d_skip=None,
                    z=None, check_collapsed=True, **kw):
    """The TRAPEZOID's PHYSICAL two-endpoint law, in fp64 (the SSD family's `tests/test_ssd_endpoint.py::
    physical_oracle` is the model for this one).

    The kernel does NOT run this recursion.  It runs the COLLAPSED one-tap
    form of the physical law, and the two are related by an
    exact change of variable, not an approximation.  Written out in this
    family's symbols (per head `h`, channel `p`, state row `n`, with
    `v_t[p,n] = x_t[p] * w_n[t]` the UNSCALED write in folded coordinates
    and `Dw_t` the write clock):

    PHYSICAL (what this function evaluates, and what `h_final` means)

        h_t = a_t h_{t-1} + Dw_t [ lam_t v_t + (1 - lam_t) a_t v_{t-1} ]
        y_t[p] = sum_n r_t[n] h_t[p,n]                (+ D_h x_t[p])

    COLLAPSED (what the Gluon scan evaluates)

        n_t   = Dw_{t+1} (1 - lam_{t+1}),   n_L = 0
        w_t   = lam_t Dw_t + n_t
        g_t   = a_t g_{t-1} + w_t v_t                 (g_t = h_t + n_t v_t)
        y_t[p] = sum_n r_t[n] g_t[p,n]  -  n_t d_t x_t[p],
                 d_t = sum_n r_t[n] w_n[t] = C~_t^T B~_t   (per head)

    and the boundary `n_L = 0` is what makes `g_L = h_L`, i.e. the carry the
    kernel returns IS the physical final state.  With `check_collapsed` the
    identity is evaluated here as well and asserted, so a test that scores
    the kernel against this function is scoring it against the physical law
    AND against the rewrite the kernel is entitled to use.

    `write_prev` supplies `v_{-1}` (zero otherwise); every other keyword is
    `_prepare`'s.
    """
    st = _prepare(x, delta, A_log, B_t, C_t, **kw)
    B, H, P, N, L = st["B"], st["H"], st["P"], st["N"], st["L"]
    if n_rows is None:
        n_rows = N - n_off
    sl = slice(n_off, n_off + n_rows)

    xh = st["x"]                                         # (B,H,P,L)
    c1, c2 = st["c1"][..., sl], st["c2"][..., sl]
    s1, s2 = st["s1"][..., sl, :], st["s2"][..., sl, :]
    w_n, r_n = st["w_n"][..., sl, :], st["r_n"][..., sl, :]
    Dw = st["Delta_w"]                                   # (B,H,{1,P},L)
    lam = _f(trap_lam)                                   # (B,H,L)

    # a_t, v_t and the two clocks, all of them (B,H,P,nr) per step.
    a = [torch.exp(c1.unsqueeze(0) * s1[..., t].unsqueeze(2)
                   + c2.unsqueeze(0) * s2[..., t].unsqueeze(2))
         for t in range(L)]
    v = [xh[..., t, None] * w_n[..., t].unsqueeze(2) for t in range(L)]
    dw = [Dw[..., t, None] for t in range(L)]            # (B,H,{1,P},1)
    lm = [lam[..., t, None, None] for t in range(L)]

    # ── the physical recursion ───────────────────────────────────────
    h = torch.zeros(B, H, P, N, dtype=DT, device=x.device) if h_init is None \
        else _f(h_init).clone()
    state = h[..., sl].clone()
    prev = torch.zeros_like(state) if write_prev is None \
        else _f(write_prev)[..., sl]
    ys = []
    for t in range(L):
        state = a[t] * state + dw[t] * (lm[t] * v[t]
                                        + (1.0 - lm[t]) * a[t] * prev)
        prev = v[t]
        ys.append((r_n[..., t].unsqueeze(2) * state).sum(-1))
    Y = torch.stack(ys, dim=-1)

    if check_collapsed:
        # ── the same law through the kernel's change of variable ─────
        nx = [(dw[t + 1] * (1.0 - lm[t + 1])) if t + 1 < L
              else torch.zeros_like(dw[0]) for t in range(L)]
        g = torch.zeros(B, H, P, N, dtype=DT, device=x.device) \
            if h_init is None else _f(h_init).clone()
        gs = g[..., sl].clone()
        if write_prev is not None:
            # g_{-1} = h_{-1} + n_{-1} v_{-1}, and n_{-1} = Dw_0 (1 - lam_0)
            gs = gs + dw[0] * (1.0 - lm[0]) * _f(write_prev)[..., sl]
        yc = []
        for t in range(L):
            w = lm[t] * dw[t] + nx[t]
            gs = a[t] * gs + w * v[t]
            # d_t = C~_t^T B~_t, per head and token (p-free by construction)
            d_t = (r_n[..., t] * w_n[..., t]).sum(-1)[..., None]   # (B,H,1)
            yc.append((r_n[..., t].unsqueeze(2) * gs).sum(-1)
                      - (nx[t][..., 0] * d_t) * xh[..., t])
        Yc = torch.stack(yc, dim=-1)
        err = (Yc - Y).abs().max() / Y.abs().max().clamp_min(1e-300)
        assert err < 1e-10, (
            f"the collapsed one-tap form disagrees with the physical "
            f"two-endpoint law by {float(err):.3e} -- the identity of "
            f"the SSD-MIMO physical law does not hold as written")
        gerr = (gs - state).abs().max() / state.abs().max().clamp_min(1e-300)
        assert gerr < 1e-10, (
            f"g_L != h_L by {float(gerr):.3e}: the boundary n_L = 0 is what "
            f"makes the collapsed carry the physical final state")

    if d_skip is not None:
        Y = Y + _f(d_skip).view(1, H, 1, 1) * xh
    Y = _z_gate(Y, z)
    h_out = h.clone()
    h_out[..., sl] = state
    return Y, h_out


def rel_err(a, b):
    """Max abs error scaled by the reference magnitude, so a near-zero entry
    cannot manufacture a huge ratio."""
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


# ---------------------------------------------------------------------------
# CLOSED-FORM (cumulative-decay) oracles for the FLAT rows -- m3/m2 --
# and the top-level row-mixture sum.
#
# `src/orbit_ssm/ssm/hybrid_mamba.py::_m3_reference` and `::_flat_reference` are
# the flat rows' explicit-recurrence statements (a python loop over t) AND,
# unlike `mh_reference` above, they are also the production `backend=
# "sequential"`/`"torch"` route itself -- there is no second, structurally
# different implementation to check them against, so validating the fused
# kernel against them, or validating `HybridMambaSSM`'s row-mixture sum by
# re-running the same module, is not independent evidence (no oracle
# exists for the m3/m2 rows or the mixture sum).
#
# The functions below are that second, independent derivation: the same
# method `mh_reference_cumprod` is to `mh_reference` (divide-by-prefix-cumprod,
# no python token loop over the recurrence), applied to the flat laws. Each
# is checked for agreement with its `_m3_reference`/`_flat_reference`
# counterpart in `tests/test_mh_reference.py`; mutual agreement between two
# structurally different derivations is the same "real evidence rather than
# a restatement" argument this module's own docstring makes for the HM rows.
# fp64/short-L only (divides by a decaying quantity).
# ---------------------------------------------------------------------------

def m3_reference_cumprod(Bt, Ct, v, ADT, DT, Trap, K_bias, Q_bias, angles, *,
                         S_init=None, kp_init=None, vp_init=None, phase_init=None,
                         return_state=False):
    """Closed-form counterpart to
    `orbit_ssm.ssm.hybrid_mamba._m3_reference`: same law,

        S_t = alpha_t S_{t-1} + alpha_t dt_t (1-lam_t) v_{t-1} k_{t-1}^T
                              + lam_t dt_t v_t k_t^T           (trapezoid)
        y_t = S_t q_t

    stated without a python loop over the recurrence, by cumulative
    products/sums (the divide-by-prefix method `mh_reference_cumprod` uses
    for the HM rows' own transported trapezoid). The rotation of k/q
    (`Theta_t = cumsum(tanh(angles) pi dt)`) is already loop-free in
    `_m3_reference` and is reused verbatim here -- only the STATE recurrence
    is re-derived.

    Shapes / kwargs identical to `_m3_reference` (which this checks):
    Bt/Ct (B,L,1,n3) post-norm pre-bias, v (B,L,H,P), ADT/DT/Trap (B,H,L)
    with Trap RAW (pre-sigmoid), K/Q_bias (H,n3), angles (B,L,H,Ka).
    `S_init`/`kp_init`/`vp_init`/`phase_init` resume a carried state (chunk
    S1's two-tap form); `return_state=True` additionally returns that carry.
    Returns (B,L,H,P) fp64.
    """
    Bt, Ct, v = Bt.double(), Ct.double(), v.double()
    ADT, DT = ADT.double(), DT.double()
    lam = torch.sigmoid(Trap.double())                          # (B,H,L)
    alpha = torch.exp(ADT)                                       # (B,H,L)
    dt_blh = DT.transpose(1, 2)                                 # (B,L,H)
    phase0 = (phase_init.double().unsqueeze(1) if phase_init is not None
             else 0.0)
    theta = phase0 + torch.cumsum(
        torch.tanh(angles.double()) * math.pi * dt_blh.unsqueeze(-1), dim=1)

    k = Bt + K_bias.double().unsqueeze(0).unsqueeze(0)          # (B,L,H,n3)
    q = Ct + Q_bias.double().unsqueeze(0).unsqueeze(0)
    Ka = angles.shape[-1]
    cos, sin = torch.cos(theta), torch.sin(theta)               # (B,L,H,Ka)

    def rot(t):
        e, o = t[..., 0::2], t[..., 1::2]
        er = torch.cat([e[..., :Ka] * cos - o[..., :Ka] * sin,
                        e[..., Ka:]], dim=-1)
        og = torch.cat([e[..., :Ka] * sin + o[..., :Ka] * cos,
                        o[..., Ka:]], dim=-1)
        return torch.stack([er, og], dim=-1).flatten(-2)

    k, q = rot(k), rot(q)

    Bsz, L, H, P = v.shape
    n3 = k.shape[-1]
    lam_blh = lam.transpose(1, 2)                                # (B,L,H)
    dt_ = dt_blh                                                 # (B,L,H)

    raw_write = v.unsqueeze(-1) * k.unsqueeze(-2)                # (B,L,H,P,n3)
    kp0 = (kp_init.double() if kp_init is not None
          else k.new_zeros(Bsz, H, n3))
    vp0 = (vp_init.double() if vp_init is not None
          else v.new_zeros(Bsz, H, P))
    prev0 = (vp0.unsqueeze(-1) * kp0.unsqueeze(-2)).unsqueeze(1)  # (B,1,H,P,n3)
    previous = torch.cat([prev0, raw_write[:, :-1]], dim=1)       # v_{t-1} k_{t-1}^T

    log_alpha = torch.log(alpha).transpose(1, 2)                 # (B,L,H)
    A_cum = torch.exp(torch.cumsum(log_alpha, dim=1))             # (B,L,H) prefix incl. t

    drive = (alpha.transpose(1, 2) * dt_ * (1 - lam_blh))[..., None, None] * previous \
           + (lam_blh * dt_)[..., None, None] * raw_write         # (B,L,H,P,n3)
    normalized = drive / A_cum[..., None, None]
    inner = torch.cumsum(normalized, dim=1)
    if S_init is not None:
        inner = inner + S_init.double().unsqueeze(1)
    S = A_cum[..., None, None] * inner                            # (B,L,H,P,n3)

    Y = torch.einsum("blhpn,blhn->blhp", S, q)
    if not return_state:
        return Y
    state = dict(S=S[:, -1], kp=k[:, -1], vp=v[:, -1], phase=theta[:, -1])
    return Y, state


def flat_reference_cumprod(Bt, Ct, v, log_a, write, B_bias, C_bias, *,
                           S_init=None, return_state=False):
    """Closed-form counterpart to
    `orbit_ssm.ssm.hybrid_mamba._flat_reference`: same law,

        S_t = exp(log_a_t) S_{t-1} + write_t v_t (B_t + B_bias)^T
        y_t = S_t (C_t + C_bias)

    stated without a python loop over the recurrence, by the
    divide-by-prefix-cumprod method (contract-equivalent of
    `mh_reference_cumprod`). Shapes / kwargs identical to `_flat_reference`
    (which this checks): Bt/Ct (B,L,n) post-norm pre-bias, v (B,L,H,P),
    log_a/write (B,H,L), biases (H,n). Returns (B,L,H,P) fp64 (+ the carry
    when `return_state=True`).
    """
    Bt, Ct, v = Bt.double(), Ct.double(), v.double()
    a_blh = torch.exp(log_a.double()).transpose(1, 2)            # (B,L,H)
    w_blh = write.double().transpose(1, 2)                       # (B,L,H)
    k = Bt.unsqueeze(2) + B_bias.double()                        # (B,L,H,n)
    q = Ct.unsqueeze(2) + C_bias.double()
    Bsz, L, H, P = v.shape
    n = k.shape[-1]

    raw_write = v.unsqueeze(-1) * k.unsqueeze(-2)                # (B,L,H,P,n)
    log_a_blh = torch.log(a_blh)
    A_cum = torch.exp(torch.cumsum(log_a_blh, dim=1))            # (B,L,H)

    drive = w_blh[..., None, None] * raw_write
    normalized = drive / A_cum[..., None, None]
    inner = torch.cumsum(normalized, dim=1)
    if S_init is not None:
        inner = inner + S_init.double().unsqueeze(1)
    S = A_cum[..., None, None] * inner                           # (B,L,H,P,n)

    Y = torch.einsum("blhpn,blhn->blhp", S, q)
    return (Y, S[:, -1]) if return_state else Y


def row_mixture_reference(x, delta, A_log, B_t, C_t, *, n_hm, m3=None, m2=None,
                          d_skip=None, z=None, **hm_kw):
    """Independent restatement of `HybridMambaSSM`'s top-level equation:

        y_t = y_t^HM + y_t^m3 + y_t^m2 + D_h x_t

    Composes `mh_reference` (the HM rows' own oracle, unchanged) with the
    closed-form flat-row oracles above, in a code path that never touches
    `HybridMambaSSM.forward`'s own summation -- so a test that runs the real
    module and compares against this function is checking the SPLIT (does
    summing four independently-computed pieces reproduce the module's own
    sum), not re-running the module's arithmetic against itself.

    `x` (B,H,P,L) is the FULL drive (all rows read the same channel); HM
    rows use `x[..., :n_hm-columns]`... no -- x is shared, only B/C/A_log
    are sliced per law, matching `HybridMambaSSM`'s own row split (every
    law's SSM reads the SAME per-head x, only the state width N differs).
    `m3`/`m2`, each `None` or a dict of that law's kwargs (see
    `m3_reference_cumprod`/`flat_reference_cumprod`), keyed WITHOUT the
    shared `x`/`delta` (those come from the HM args). `d_skip`/`z` apply
    once, on the summed output, exactly as the module's own epilogue.
    """
    B, H, P, L = x.shape
    Y = torch.zeros(B, H, P, L, dtype=DT, device=x.device)
    if n_hm > 0:
        y_hm, _ = mh_reference(x, delta, A_log, B_t, C_t, n_rows=n_hm, **hm_kw)
        Y = Y + y_hm
    if m3 is not None:
        y3 = m3_reference_cumprod(**m3)
        Y = Y + y3.permute(0, 2, 3, 1)                          # (B,L,H,P)->(B,H,P,L)
    if m2 is not None:
        y2 = flat_reference_cumprod(**m2)
        Y = Y + y2.permute(0, 2, 3, 1)
    if d_skip is not None:
        Y = Y + _f(d_skip).view(1, H, 1, 1) * _f(x)
    return _z_gate(Y, z)
