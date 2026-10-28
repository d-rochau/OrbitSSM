"""[Library of tests/test_ssd_endpoint.py, moved here so the contract suite does not import a legacy test file.]

THE canonical SSD-MIMO oracle (``physical_oracle``) and its endpoint tests.

``physical_oracle`` is the "obviously correct" transcription
of the defining recurrence (physical law, skip / sibling weight / gate): one explicit python loop per batch row,
head, sibling and token, explicit 2x2-block rotation matrices, the explicit
two-endpoint write, the explicit skip chi-law, physical coordinates, no
production map/scan/step helpers.  Every optimized torch route is
REARRANGED math and is scored against it (this file and
``test_physical_law_witness.py``).  Run it in fp64 (cast module and inputs).
"""
import copy
import math
import sys

import pytest
import torch
import torch.nn.functional as F

from _tolerance import assert_close_terms, terms_abs
from orbit_ssm.layers.state import ScanState
from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig


#: the complex-block spelling of the old `rot_mode="m3"` (rope_fraction 1.0)
# The complex block is gated explicitly (its default
# is gate=False), so the gated cells keep gating the complex rows
ROT_ROWS = [{"frac": 1.0, "decay": "head_dynamic", "complex": True, "gate": True}]


def ssd_rows(mode, dynamic=True):
    """Test label -> `rows`: `mode` "m3" = the whole state is one complex
    block, None = all real; `dynamic` picks head_dynamic / head_static
    (the old a_dyn=True / False).  The block is gated explicitly
    (a complex block's default is gate=False), so
    the gated tests below keep gating the complex rows."""
    law = "head_dynamic" if dynamic else "head_static"
    return [{"frac": 1.0, "decay": law, "complex": mode == "m3", "gate": True}]


def pair_perm(N, device=None):
    """PAIRHALF: `idx` with half coordinate i = interleaved coordinate
    `idx[i]` -- [0, 2, 4, ..., 1, 3, 5, ...]."""
    return torch.cat([torch.arange(0, N, 2, device=device),
                      torch.arange(1, N, 2, device=device)])


def rotation(angle, pair_half=False):
    c, s = angle.cos(), angle.sin()
    blocks = torch.stack((c, -s, s, c), -1).reshape(*angle.shape, 2, 2)
    M = torch.block_diag(*blocks.unbind(0))
    if pair_half:
        # The SAME rotation in permuted state
        # coordinates -- pair k at (k, k + N/2) instead of (2k, 2k+1).
        # R_half = P^T R P, and P is a permutation, so this is a gather.
        idx = pair_perm(M.shape[0], M.device)
        M = M.index_select(0, idx).index_select(1, idx)
    return M


def physical_oracle(m, x, u, h0, z=None, *, carry=None, return_carries=False,
                    stats=None):
    """The ONE law, in PHYSICAL coordinates.

    Contract equations, per batch
    row b, head h, sibling r and token t:

        Bbar_t = diag(1 + gB_t[:, r]) Bhat_t + beta_{h,r}        (N x R)
        Cbar_t = (1 + gC_t) * Chat_t + gamma_{h,r}               (N x O)
        A_t    = exp(a_t Delta_t) R(dTheta_t),  dTheta_t[k] = -omega_k tanh(Ang_t[k]) Delta_t
        v_t    = Bbar_t u_t                                      (N x G, one column per group)
        h_t    = A_t h_{t-1} + Delta_t [lam_t v_t + (1 - lam_t) A_t v_{t-1}]
        q_t    = Cbar_t^T R(nu_h) h_t                            (O x G)
        y_t    = silu(z_t) * sum_r chi_{t,h,r} (q_t + D_{h,:,r,:} u_t)

    ``m`` is the module (its PARAMETERS and constructor fields define the
    maps; none of its methods are called), ``x`` the drive [B, L, H*G*R],
    ``u`` the token-map source [B, L, d_sel], ``h0`` the initial FOLDED
    hidden state [B, H, G*R, N] (the public State layout), ``z`` the output
    gate [B, L, H*G*O] or None.

    ``carry`` (optional) holds the other public State carries at t = -1:
    ``rot_ang`` Theta_{-1} [B, H, R, K_rot] (the ROTATED pairs only,
    K_rot = n_rope_angles: every rotation quantity -- Theta, theta_h -- acts
    on the complex rows only and the real pairs are exactly the identity), ``write_prev`` the folded previous
    write Q_{-1}^T v_{-1} [B, H, G*R, N], ``bc_conv_state`` the last d_conv
    raw B|C inputs [B, C, d_conv].  Absent = sequence start (zeros).

    Returns ``(y, folded final hidden)``, or with ``return_carries=True``
    ``(y, carries)`` where ``carries`` is every State key at t = L-1 in
    the public (folded) coordinates: hidden_state = Q^T h, rot_ang = Theta,
    write_prev = Q^T v, bc_conv_state = the raw window.

    ``stats`` (optional dict, plain fp64 calls only) receives the two
    magnitudes that amplify rounding in a REARRANGED route (A7 ``c``, see
    ``law_amplification``): ``decay`` = max over (b, h, r) of the total
    decay argument sum_t |a_t| Delta_t of this call, ``phase`` = max
    |Theta_t| (unwrapped, carried clock included) + max |theta_h|.
    """
    carry = carry or {}
    B, L, _ = x.shape
    H, G, R, O, N = m.H, m.G, m.R, m.O, m.N
    K = N // 2
    half = getattr(m, "pair_layout", "interleaved") == "half"

    def project(name):
        # the Linear map of the selection input: W_name sel_t + b_name
        p = getattr(m, name)
        return F.linear(u, p.weight, p.bias)

    # ── token maps ──────────────────────────────────
    b, c = project("B_proj"), project("C_proj")          # B_proj sel_t, C_proj sel_t
    window = None
    if m.bc_conv1d is not None:
        # optional causal depthwise conv + SiLU over the raw [B | C] stream,
        # continuing from the carried window of the last d_conv raw inputs
        conv, d = m.bc_conv1d.conv, m.bc_conv1d.d_conv
        raw = torch.cat((b, c), -1).transpose(1, 2)       # [B, C, L]
        prev = carry.get("bc_conv_state")
        if prev is None:
            prev = raw.new_zeros(B, raw.shape[1], d)
        full = torch.cat((prev, raw), -1)                 # [B, C, d + L]
        window = full[..., -d:]                           # carried to the next call
        bc = F.conv1d(full[..., 1:], conv.weight, conv.bias, groups=conv.groups)
        b, c = F.silu(bc).transpose(1, 2).split((N * R, N * O), -1)
    b, c = b.reshape(B, L, N, R), c.reshape(B, L, N, O)   # columns n*R + q, n*O + o
    # Bhat = B / sqrt(mean_n B^2 + eps) * w_B, per column (RMSNorm over N);
    # each map's norm is its own switch (b_norm / c_norm)
    if m.b_norm:
        epsb = m.B_norm.eps if m.B_norm.eps is not None else torch.finfo(b.dtype).eps
        b = b / (b.square().mean(-2, keepdim=True) + epsb).sqrt() * m.B_norm.weight[:, None]
    if m.c_norm:
        epsc = m.C_norm.eps if m.C_norm.eps is not None else torch.finfo(c.dtype).eps
        c = c / (c.square().mean(-2, keepdim=True) + epsc).sqrt() * m.C_norm.weight[:, None]
    if m.bc_bias:
        if not m.head_bias:
            # bc_bias="shared": the R^N bias joins inside the norm block
            b, c = b + m.B_bias[:, None], c + m.C_bias[:, None]
    # residual map gates 1 + gB, 1 + gC (unit when disabled)
    def gate(name, width):
        g = project(name)
        kr = m.n_rope_angles if m.rot_mode == "m3" else 0
        # A gate on a complex pair is a complex
        # scalar (pair k's ONE factor on both interleaved coordinates), and
        # each row block's `gate` field says whether its rows are gated at
        # all -- a gate-off block's rows have factor exactly 1 (g = 0).  The
        # compact rows: the gated complex block's K_rot pairs, then the
        # gated real block's rows.
        blocks = m.spec["row_blocks"]
        g_cx = any(b_["gate"] for b_ in blocks if b_["complex"])
        g_re = any(b_["gate"] for b_ in blocks if not b_["complex"])
        n_re = N - 2 * kr
        g = g.reshape(B, L, -1, width)
        parts, off = [], 0
        if kr:
            if g_cx:
                parts.append(g[:, :, off:off + kr].repeat_interleave(2, 2))
                off += kr
            else:
                parts.append(g.new_zeros(B, L, 2 * kr, width))
        if n_re:
            if g_re:
                parts.append(g[:, :, off:off + n_re])
                off += n_re
            else:
                parts.append(g.new_zeros(B, L, n_re, width))
        assert off == g.shape[2], "gate projection rows != gated rows"
        g = torch.cat(parts, 2)
        if kr and half:
            g = g[:, :, pair_perm(N, g.device)]
        return g.reshape(B, L, N, width)

    gb = 1 + gate("gB_proj", R) if m.use_g_b else torch.ones_like(b)
    gc = 1 + gate("gC_proj", O) if m.use_g_c else torch.ones_like(c)
    # Delta = softplus(dt_proj sel_t + dt_bias), uncapped
    raw_dt = project("dt_proj")
    dt = torch.logaddexp(raw_dt, torch.zeros_like(raw_dt)).reshape(B, L, H, R)
    # decay rate |a| (a_t = -rate)
    if m.a_dyn:
        va = project("a_dyn_proj")
        if m.a_act == "heavy_tail":
            # ht(v) = 1 + v (v >= 0), 1 / (1 - v) (v < 0), written without a
            # branch: max(v, 0) + 1 / (1 - min(v, 0)) is exactly that function
            rate = va.clamp_min(0) + (1 - va.clamp_max(0)).reciprocal()
        else:
            rate = torch.logaddexp(va, torch.zeros_like(va))     # softplus(v)
        rate = rate.clamp_min(1e-4)                                # a_floor
        rate = (rate.reshape(B, L, H, R) if m.a_dyn_per_sibling
                else rate[..., None].expand(B, L, H, R))           # shared over r
    else:
        rate = m.A_log.exp()[None, None].expand(B, L, H, R)
    # endpoint weight lambda_t = sigmoid(trap_proj sel_t); Euler = lambda 1
    lam = project("trap_proj").sigmoid() if m.use_trap else x.new_ones(B, L, H)
    # phase increment dTheta_t[h,r,k] = -omega_k tanh(Ang_t[k]) Delta_t[h,r]
    inc = x.new_zeros(B, L, H, R, K)
    if m.rot_mode == "m3":
        ang = project("theta_proj")
        layout = getattr(m, "omega_layout", "pair")
        if layout == "head":
            # ONE eigenvalue per head (ROT_RECALL): the head's single angle
            # feature and frequency on every rotated pair
            ang = ang.reshape(B, L, H, 1).expand(B, L, H, m.n_rope_angles)
            omega = (math.pi * torch.tanh(m.theta_freq) if m.learn_theta_freq
                     else x.new_full((H,), math.pi))[:, None]
        elif layout == "head_pair":
            # one frequency per (head, pair): the angle on every head first
            ang = ang.reshape(B, L, H if m.theta_per_head else 1,
                              m.n_rope_angles).expand(B, L, H, m.n_rope_angles)
            omega = math.pi * torch.tanh(m.theta_freq)            # [H, K_rot]
        else:
            ang = ang.reshape(B, L, H if m.theta_per_head else 1, m.n_rope_angles)
            omega = (math.pi * torch.tanh(m.theta_freq) if m.learn_theta_freq
                     else x.new_full((m.n_rope_angles,), math.pi))
        ang = F.pad(ang.tanh() * omega, (0, K - m.n_rope_angles))  # pairs >= K_rot: 0
        inc = -dt[..., None] * ang[:, :, :, None]
    K_rot = m.n_rope_angles if m.rot_mode == "m3" else 0
    theta0 = carry.get("rot_ang")                                  # Theta_{-1}
    if theta0 is None:
        theta0 = x.new_zeros(B, H, R, K_rot)
    theta0 = F.pad(theta0, (0, K - K_rot))                         # real pairs: 0
    phase = theta0[:, None] + inc.cumsum(1)                        # Theta_t
    if stats is not None:
        stats["decay"] = float((rate * dt).detach().sum(1).max())
        stats["phase"] = float(phase.detach().abs().max()) + (
            float(m.theta_h.detach().abs().max())
            if m.rot_mode == "m3" and m.use_theta_h else 0.0)
    eye = torch.eye(N, dtype=x.dtype, device=x.device)
    w_prev0 = carry.get("write_prev")

    outputs, final, fin_write = [], [], []
    for batch in range(B):
        heads, end_heads, end_writes = [], [], []
        for head in range(H):
            siblings, end_siblings, end_w = [], [], []
            for r in range(R):
                # Q_{-1} = R(Theta_{-1}); public carries are folded: h = Q f
                q0 = rotation(theta0[batch, head, r], half) if m.rot_mode else eye
                h = q0 @ h0[batch, head].reshape(G, R, N)[:, r].T            # h_{-1}, N x G
                previous = (q0 @ w_prev0[batch, head].reshape(G, R, N)[:, r].T
                            if w_prev0 is not None else torch.zeros_like(h))  # v_{-1}
                # R(nu_h): theta_h has one entry per ROTATED pair; real pairs 0
                read_off = (rotation(F.pad(m.theta_h[head], (0, K - K_rot)), half)
                            if m.use_theta_h else eye)
                ys = []
                for t in range(L):
                    # A_t = exp(-rate Delta) R(dTheta_t)
                    transition = (-rate[batch, t, head, r] * dt[batch, t, head, r]).exp() * (
                        rotation(inc[batch, t, head, r], half) if m.rot_mode else eye)
                    # Bbar = diag(1 + gB[:, r]) Bhat + beta_{h,r};  Cbar = (1 + gC) * Chat + gamma_{h,r}
                    bt = b[batch, t] * gb[batch, t, :, r, None]
                    ct = c[batch, t] * gc[batch, t]
                    bb = m.B_bias[head, r] if m.head_bias else torch.zeros_like(bt)
                    cb = m.C_bias[head, r] if m.head_bias else torch.zeros_like(ct)
                    bw, cbar = bt + bb, ct + cb
                    u_hg = x[batch, t].reshape(H, G, R)[head]                 # G x R inputs
                    drive = bw @ u_hg.T                                      # v_t = Bbar u_t (N x G)
                    l, d = lam[batch, t, head], dt[batch, t, head, r]
                    # h_t = A h_{t-1} + Delta [lam v_t + (1 - lam) A v_{t-1}]
                    h = transition @ h + d * (l * drive + (1 - l) * (transition @ previous))
                    previous = drive
                    y = (cbar.T @ read_off @ h).T                             # q_t^T = h^T R(nu)^T Cbar -> G x O
                    if m.skip:
                        # branch-local skip D[h, g, r, :] u[h, g, r], before the sibling sum
                        gains = (m.D[head] if m.skip_sharing == "head" else
                                 m.D[head, :, r, None] if m.skip_sharing == "channel" else
                                 m.D[head, :, r, :])
                        y = y + u_hg[:, r, None] * gains
                    if m.r_mix_mode != "none":
                        # chi_{t,h,r} = zeta_{h,r} + (W_mix sel_t)_{h,r}
                        mix = m.r_mix[head, r]
                        if m.r_mix_dyn:
                            mix = mix + project("r_mix_proj")[batch, t].reshape(H, R)[head, r]
                        y = mix * y
                    ys.append(y)
                siblings.append(torch.stack(ys))
                q_end = rotation(phase[batch, L - 1, head, r], half) if m.rot_mode else eye
                end_siblings.append((q_end.T @ h).T)                          # folded f = Q^T h
                end_w.append((q_end.T @ previous).T)                          # folded Q^T v_{L-1}
            heads.append(torch.stack(siblings).sum(0))                        # sum over siblings r
            end_heads.append(torch.stack(end_siblings, 1).reshape(G * R, N))
            end_writes.append(torch.stack(end_w, 1).reshape(G * R, N))
        outputs.append(torch.stack(heads, 1).reshape(L, H * G * O))
        final.append(torch.stack(end_heads))
        fin_write.append(torch.stack(end_writes))
    output = torch.stack(outputs)
    if z is not None:
        output = output * F.silu(z)                                           # output gate, after the sum
    if not return_carries:
        return output, torch.stack(final)
    carries = {"hidden_state": torch.stack(final)}
    if m.rot_mode == "m3":
        carries["rot_ang"] = phase[:, L - 1, ..., :K_rot]
    if m.use_trap:
        carries["write_prev"] = torch.stack(fin_write)
    if window is not None:
        carries["bc_conv_state"] = window
    return output, carries


# ── A7 bounds for a REARRANGED route against this oracle ──
# Every optimized route computes the same law in another order, so the two
# fp64 arms differ by reassociation only.  The comparison is the
# terms-scaled form, elementwise (a gradient, a gated output or a carry can
# cancel to ~0):  |a - r| <= c * u64 * sum|terms|, sum|terms| from the
# oracle run in absolute arithmetic (`terms_abs`), and
#
#   c = sqrt(2) * sqrt(depth) + decay + phase
#
# sqrt(depth): typical random walk of `depth` roundings (A7), sqrt(2): both
# arms round independently; `decay` = the largest decay argument
# sum_t |a_t| Delta_t of the call and `phase` = the largest |Theta_t| (+
# |theta_h|): a route that forms exp(-(cs_t - cs_s)) or cos/sin(Theta_t -
# Theta_s) from cumulative sums carries an argument error u * |cumsum|,
# which exp / cos / sin turn into that relative error of the factor.
def ssd_depth(m, L, B=1, extra=0):
    """(depth of an output / carry, depth of a gradient) of one call.

    Output: y_t sums the writes of <= L tokens x N states x max(R, O)
    siblings x 2 endpoints (trapezoid), each a product of <= L + 8 factors
    (L transitions, Delta, lambda, B, gB, x, C, gC, chi, gate): n + m.
    Gradient: one token's adjoint is the transposed recursion (as deep as
    the output) and a parameter / selection gradient sums B*L of them.
    ``extra``: further sums in series with the core (a mixer's layers)."""
    y = 2 * L * m.N * max(m.R, m.O) + L + 8 + extra
    return y, (B * L + 1) * y


def law_c(m, stats, L, B=1, extra=0, arms=2):
    """(c for outputs / carries, c for gradients) -- see the block above.
    ``arms=1`` when only the route under test rounds at ``u`` (an fp32
    route against the fp64 oracle)."""
    dy, dg = ssd_depth(m, L, B, extra)
    amp = stats["decay"] + stats["phase"]
    return math.sqrt(arms * dy) + amp, math.sqrt(arms * dg) + amp


def law_stats(m, x, u, h0, z=None, carry=None):
    """`stats` of the oracle on these inputs (plain fp64 run, no grad)."""
    stats = {}
    with torch.no_grad():
        d = lambda v: None if v is None else v.detach().double()  # noqa: E731
        physical_oracle(m, d(x), d(u), d(h0), d(z),
                        carry=None if carry is None else
                        {k: d(v) for k, v in carry.items()}, stats=stats)
    return stats


def cores_law_c(module, call, L, B=1, extra=0, arms=2):
    """``law_c`` of every SSD-MIMO scan ``module`` contains, on the inputs
    its layers actually feed each scan: ``call()`` (a plain no-grad run of
    the module) is observed through each layer's ``scan``; each scan call adds its
    decay / phase stats (``law_stats``) and its ``ssd_depth`` (cores in
    series: errors add), ``extra`` the depth of the non-core sums."""
    seen = []

    def spy(core):
        original = core.scan

        def scan(*args, **kwargs):
            seen.append((core, args, kwargs))
            return original(*args, **kwargs)
        return scan

    patched = [m for m in module.modules() if isinstance(m, SSDMimoSSM)]
    for m in patched:
        m.scan = spy(m)
    try:
        with torch.no_grad():
            call()
    finally:
        for m in patched:
            del m.scan
    assert seen, "no SSD-MIMO core ran"
    amp, depth = 0.0, extra
    for core, args, kwargs in seen:
        xb = args[0]
        sel = kwargs.get("sel_x")
        st = kwargs.get("state")
        carry = ({k: v for k, v in st.items()
                  if k != "hidden_state" and torch.is_tensor(v)} if st else None)
        h0 = (st["hidden_state"] if st and st.get("hidden_state") is not None
              else xb.new_zeros(xb.shape[0], core.H, core.P, core.N))
        stats = law_stats(core, xb, xb if sel is None else sel, h0,
                          kwargs.get("gate_z"), carry)
        amp += stats["decay"] + stats["phase"]
        depth += ssd_depth(core, xb.shape[1], xb.shape[0])[0]
    return (math.sqrt(arms * depth) + amp,
            math.sqrt(arms * (B * L + 1) * depth) + amp)


def mixer_law_c(mix, x, **kw):
    """``cores_law_c`` of a mixer called as ``mix(x, **kw)``, with the
    layer's own sums in series with the core in the depth: in_proj
    (d_model), the causal conv (d_conv), the output RMSNorm and out_proj
    (d_inner each)."""
    extra = mix.d_model + getattr(mix, "d_conv", 0) + 2 * mix.d_inner
    return cores_law_c(mix, lambda: mix(x.detach(), **kw), x.shape[1],
                       x.shape[0], extra)


def module_terms(mod, fn, *args, grad_outputs=None):
    """``terms_abs`` of a module route (parameters of ``mod`` tracked),
    with the fused glue nodes off (``SSD_MIMO_GLUE=0``): they are custom
    autograd Functions writing into preallocated buffers, and the plain
    ops they replace return the same values (``split_in_proj``), so the
    plain composition is the route's absolute-arithmetic twin."""
    import os
    old = os.environ.get("SSD_MIMO_GLUE")
    os.environ["SSD_MIMO_GLUE"] = "0"
    try:
        return terms_abs(fn, *args, grad_outputs=grad_outputs, params=mod)
    finally:
        if old is None:
            os.environ.pop("SSD_MIMO_GLUE")
        else:
            os.environ["SSD_MIMO_GLUE"] = old


class OracleTerms:
    """The physical oracle in fp64 absolute arithmetic (``terms_abs``).

    ``inputs``: ordered {name: tensor} of the oracle's x, u, h0, z (z may
    be None); ``carry``: the other State carries.  Gradients (with
    ``grad_outputs`` matching the oracle's return, ``None`` for untracked
    outputs) come back for every requires-grad input and every parameter
    of ``m`` (fp64), keyed by name.  ``c_y`` / ``c_g``: ``law_c``; ``u``
    and ``arms``: the arithmetic of the route under test (default two fp64
    formulations)."""

    def __init__(self, m, inputs, *, carry=None, grad_outputs=None,
                 return_carries=False, u="fp64", arms=2):
        names = list(inputs)
        ckeys = sorted(carry or {})
        self.stats = {}

        def fn(*vals):
            v = dict(zip(names + ckeys, vals))
            c = {k: v[k] for k in ckeys} or None
            return physical_oracle(m, *(v[n] for n in names), carry=c,
                                   return_carries=return_carries,
                                   stats=self.stats)

        args = [inputs[n] for n in names] + [carry[k] for k in ckeys]
        tracked = [n for n, a in zip(names + [f"carry.{k}" for k in ckeys],
                                     args)
                   if a is not None and a.requires_grad]
        res = terms_abs(fn, *args, grad_outputs=grad_outputs, params=m)
        self.out, self.terms = res.out, res.terms
        self.grads = dict(zip(tracked, res.grads or ()))
        self.grad_terms = dict(zip(tracked, res.grad_terms or ()))
        self.grads.update(res.param_grads or {})
        self.grad_terms.update(res.param_grad_terms or {})
        x = inputs[names[0]]
        self.u = u
        self.c_y, self.c_g = law_c(m, self.stats, x.shape[1], x.shape[0],
                                   arms=arms)

    def check_grad(self, name, actual, like):
        """``actual`` (None = unused) against the oracle's gradient of
        ``name`` (None = no path: exact zero, zero terms)."""
        b, t = self.grads.get(name), self.grad_terms.get(name)
        if b is None:
            b = t = torch.zeros_like(like, dtype=torch.float64)
        a = torch.zeros_like(b) if actual is None else actual
        assert_close_terms(a, b, t, self.u, self.c_g, name=f"grad {name}")


def _make_case_layer(mode, rank, conv, dynamic, read_rank):
    return SSDMimoSSM(SSDMimoConfig(d_model=3, d_state=16, d_inner=4 * rank, n_groups=2, n_heads=2,
                  d_ssm_in=rank, d_ssm_out=rank if read_rank is None else read_rank, proj_source="layer_input",
                  write_rule="trapezoid", row_blocks=ssd_rows(mode, dynamic), bc_conv=conv,
                  # EXPLICIT, not the default: the conv arm must ask for
                  # the width it exercises, and the no-conv arm must ask
                  # for the absence, so neither arm moves if the generic
                  # recipe default moves again.
                  d_conv=4 if conv else 0,
                  b_gate=True, c_gate=True), execution={"backend": "torch", "precision": "ieee"})


def make_case(mode, rank=2, length=7, conv=False, dynamic=True, device="cpu", dtype=torch.float64, read_rank=None):
    torch.manual_seed(31)
    # the full layer; the cases drive its bare recurrence, `m.scan(...)`.
    # The layer also owns in_proj / out_proj, which the cases never run but
    # whose default nn.Linear init draws from the global RNG BEFORE the scan
    # parameters -- shifting every scan-owned draw away from the bare-core
    # case this cell has always tested (dt_proj.bias grad 0.0219 vs the 0.02
    # bound on the shifted draw, within it on the original one).  Draw the
    # Linear inits from a forked stream so the scan-owned parameters, the
    # inputs and the gates below are exactly the bare-core draw
    # (`test_make_case_draws_the_bare_core_weights` pins it).
    _linear_reset = torch.nn.Linear.reset_parameters

    def _forked_linear_reset(self):
        # Linear.reset_parameters <- Linear.__init__ <- the constructor that
        # owns the Linear: fork only for the LAYER's own (SSDMimoSSM._build),
        # never for the scan core's projections, which are the case.
        owner = sys._getframe(2).f_code.co_qualname
        if not owner.startswith("SSDMimoSSM."):
            return _linear_reset(self)
        with torch.random.fork_rng(devices=[]):
            _linear_reset(self)

    torch.nn.Linear.reset_parameters = _forked_linear_reset
    try:
        m = _make_case_layer(mode, rank, conv, dynamic, read_rank)
    finally:
        torch.nn.Linear.reset_parameters = _linear_reset
    m = m.to(device=device, dtype=dtype)
    m.B_norm.eps = m.C_norm.eps = 1e-5
    # Nonzero gates, offsets and varying clocks/lambda prevent init-only parity.
    with torch.no_grad():
        for name, p in m.named_parameters():
            if any(k in name for k in ("gB_proj", "gC_proj", "trap_proj", "theta_proj", "theta_h", "B_bias", "C_bias")):
                p.uniform_(-.4, .4)
            if name == "theta_freq":
                p.uniform_(-1.0, 1.5)      # omega = pi*tanh(.) in (-.76, .90) pi
    x = torch.randn(1, length, m.d_inner, device=device, dtype=dtype, requires_grad=True)
    u = torch.randn(1, length, 3, device=device, dtype=dtype, requires_grad=True)
    h = torch.randn(1, m.H, m.P, m.N, device=device, dtype=dtype, requires_grad=True) * .1
    z = torch.randn(*x.shape[:-1], m.d_out, device=device, dtype=dtype, requires_grad=True)
    return m, x, u, h, z


























# ── the static READ offset served IN-KERNEL under Mamba-3's bias order
# C~ = R(theta_h) R(Theta)^T (C + gamma).  Before, theta_h was
# host-folded into a per-kernel-head read map (Gc = H_k), which at R2O2 cost
# 2.3x the time and 4x the memory of the real row at the MQMKVAR shape. ──





# ── the in-scan trapezoid's ROUTE in training ──
# The forward was generalised to R, O >= 1 but was
# gated on `not torch.is_grad_enabled()` because the ADJOINT was still
# the R = O = 1 scalar-dot kernel; every training step at R2O2 therefore
# paid `endpoint_local_map` + `endpoint_correct` + their backward.  The
# adjoint is generalised now, so the grad-mode gate is gone: at R = O = 1
# the in-scan route is the DEFAULT, and above it it is reachable by name
# (`SSD_MIMO_TRAP_INKERNEL_RO=1` -- the endpoint pass keeps the default
# there because it is measurably faster, see the routing comment in
# `ssd_mimo.py`).  Either way the route must be the one asked for, with
# NO host correction built beside it, and that is what this asserts.





# ═══════════════════════════════════════════════════════════════════════
# `pair_layout` at the MODULE
# ═══════════════════════════════════════════════════════════════════════
#
# The module permutes NOTHING at runtime: B_proj / C_proj simply learn in
# the permuted state basis, and every per-pair parameter (theta_h,
# theta_freq) keeps its meaning.  Two statements are scored here:
#   (1) the CUDA route in the half layout is the module's own physical
#       law in the half layout (`physical_oracle` reads `m.pair_layout`);
#   (2) a module in one layout and a module in the other, related by the
#       explicit state-dict permutation, are the SAME function.



