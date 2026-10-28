"""Shared fp64 reference helpers for the collected test suite.

Not a test module (no ``test_`` prefix, so pytest never collects it): a
helper library the three collected modules import.  Everything here is an
*oracle* — a slow, obviously-correct formulation of a family's math at
fp64 — used to score the production backends ("cuda", "torch",
"sequential").

The oracles keep the module front ends (projections, convs, norms,
epilogues) in their shipped fp32 form and replace ONLY the scan, so a
parity failure localises to the scan/fold rather than to matmul rounding
in the projections.
"""

import contextlib
import torch
from orbit_ssm.kernels.hybrid_mamba_mh.layout import (
    DYN_SOFTPLUS, GATE_SIGMOID, LAW_A_DYN, LAW_MU,
    fracs_to_rows, norm_fracs,
)
from orbit_ssm.kernels.ssd_mimo.reference import unified_reference


def relative(actual, expected):
    """Relative Frobenius error, computed in fp64."""
    return ((actual.double() - expected.double()).norm()
            / expected.double().norm().clamp_min(1e-30)).item()


def ssd_state_dim(b_tok, bc_per_head):
    """State width N from a B_tok whose rank axis is optional."""
    return b_tok.shape[3] if bc_per_head else b_tok.shape[2]


def _ssd_c_per_o(c_tok, bc_per_head):
    return c_tok.dim() == (5 if bc_per_head else 4)


def ssd_reference_fwd(x_in, delta, b_tok, c_tok, a_log, bc_per_head,
                      x0=None):
    """Forward oracle: ``unified_reference`` run once per G channel.

    ``unified_reference`` carries the G axis as its ``g`` argument and is
    exercised here at g=1 with an identity rotation, so each channel is an
    independent SISO/MIMO scan; the results are re-assembled into the
    kernel's [B, H, (O,) G, L] / [B, H, G, N] layout.
    """
    x = x_in.double()
    batch, _heads, groups, _rank, length = x.shape
    state = ssd_state_dim(b_tok, bc_per_head)
    c_per_o = _ssd_c_per_o(c_tok, bc_per_head)
    blocks, block_k = state, 1
    q = torch.ones(batch, length, blocks, block_k, block_k,
                   device=x.device, dtype=torch.float64)

    ys, states = [], []
    for channel in range(groups):
        x0_c = (x0[:, :, channel:channel + 1].double()
                if x0 is not None else None)
        y_c, s_c = unified_reference(
            x[:, :, channel], delta.double(), b_tok.double(), c_tok.double(),
            a_log.double(), q, g=1, bc_per_head=bc_per_head, x0=x0_c)
        ys.append(y_c)
        states.append(s_c)
    y = torch.cat(ys, dim=3 if c_per_o else 2)
    return y, torch.cat(states, dim=2)


def ssd_diff_reference(x_in, delta, b_tok, c_tok, a_log, bc_per_head):
    """Differentiable token-by-token oracle (per-head scalar decay).

    h_t = h_{t-1}·exp(-e^{a_h}·δ_t) + Σ_r B_r(t)·δ_t·x_{r,t},
    y_t = C(t)ᵀ h_t — written so autograd differentiates the loop itself.

    The archived original carried one python loop per (token, G channel,
    rank stream); the G and R axes are contracted with an einsum here (the
    same sum, one term per r) so a G=64/R=8 cell costs L python steps
    instead of L·G·R.  The TOKEN loop is deliberately kept: it is what
    makes this an independent formulation of the chunked kernel rather
    than a second copy of the same blocked algebra.

    Returns the kernel layout: y [B, H, G, L] (O == 1) or [B, H, O, G, L],
    state [B, H, G, N].
    """
    batch, heads, groups, rank, length = x_in.shape
    device = x_in.device
    state = ssd_state_dim(b_tok, bc_per_head)

    if bc_per_head:
        b_per_r = b_tok.dim() == 5
        c_per_o = c_tok.dim() == 5
        groups_b, groups_c = b_tok.shape[2], c_tok.shape[2]
        hpg_b, hpg_c = heads // groups_b, heads // groups_c
    else:
        b_per_r = b_tok.dim() == 4
        c_per_o = c_tok.dim() == 4

    rates = -torch.exp(a_log)                               # [H]
    hidx = torch.arange(heads, device=device)
    h = x_in.new_zeros(batch, heads, groups, state)
    ys = []

    def _per_head(tok, per_axis, hpg):
        """One token's B/C slice, broadcast to [B, H, N, tail]."""
        if bc_per_head:
            out = tok[:, hidx // hpg]                       # [B, H, N(, tail)]
        else:
            out = tok.unsqueeze(1).expand(-1, heads, *tok.shape[1:])
        return out if per_axis else out.unsqueeze(-1)

    for t in range(length):
        step = delta[:, :, t]                               # [B, H]
        decay = torch.exp(rates[None, :] * step)            # [B, H]
        bt_h = _per_head(b_tok[:, t], b_per_r, hpg_b if bc_per_head else 1)
        ct_h = _per_head(c_tok[:, t], c_per_o, hpg_c if bc_per_head else 1)
        h = h * decay[:, :, None, None]
        h = h + torch.einsum("bhnr,bhgr->bhgn", bt_h,
                             x_in[..., t] * step[:, :, None, None])
        ys.append(torch.einsum("bhgn,bhno->bhog", h, ct_h))

    y = torch.stack(ys, dim=-1)                             # [B,H,O,G,L]
    return (y if c_per_o else y[:, :, 0]), h


def _mh_pair_tie(t, axis, pair_rot):
    """Copy each rotating pair's EVEN coordinate onto its odd partner.

    ``pair_rot`` is ``row_rot[0::2] & row_rot[1::2]``, so non-rotating rows
    keep their independent draws and the untied path stays covered.
    """
    if t is None:
        return t
    v = t.movedim(axis, -1).clone()
    v[..., 1::2] = torch.where(pair_rot, v[..., 0::2], v[..., 1::2])
    return v.movedim(-1, axis).contiguous()


def mh_make_inputs(B=2, H=3, P=4, N=8, L=16, *, device="cpu",
                   dtype=torch.float32, fracs=None, rot=False, gates=False,
                   biases=False, delta_w=None, a_dyn_res="n", mu_res="n",
                   h_init=False, theta_init=False, mu_init=False,
                   seed=0, G=None, G_gate=None,
                   dact=DYN_SOFTPLUS, untie=()):
    """Returns ``(args, kwargs)`` ready to splat into ``mh_reference``,
    ``mh_forward``, ``mh_backward`` or ``mh_scan``.

    Positional args are ``(x, delta, A_log, B_t, C_t)``; everything the
    case switches on rides in the kwargs dict, exactly as the kernel ABI
    takes it.

    Every factor of a rotating pair's modulus comes out PAIR-TIED:
    ``A_log``, the decay bias ``beta_a``, ``mu_gain``
    and — at n-resolution — the ``a_dyn``/``mu``/``mu_init`` streams.  Not
    just ``A_log``: the rotation is folded onto B/C rather than applied to
    the state, and that fold is a change of frame only while
    ``diag(a_even, a_odd)`` is a scalar on the pair, which every one of
    those can break on its own.  ``untie`` names factors to leave untied on
    purpose, which is how the refusal cells reach
    ``layout.assert_pair_tied``.
    """
    untie = (untie,) if isinstance(untie, str) else tuple(untie)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, scale=1.0):
        return ((torch.randn(*shape, generator=gen) * scale)
                .to(device=device, dtype=dtype))

    pairs = N // 2
    frs = norm_fracs(fracs, N)
    row_law, row_rot = fracs_to_rows(frs, N, device=device)
    if rot and not row_rot.any():
        row_rot[:] = True
    laws = {f.law for f in frs}

    x = r(B, H, P, L)
    delta = r(B, H, L, scale=0.5)
    b_tok = r(B, N, L)
    c_tok = r(B, N, L)

    # A_log spread over a decade; pair-tied below, with the rest of the
    # modulus, wherever a row rotates
    a_log = r(H, P, N, scale=0.5).to(torch.float32)

    kw = dict(row_law=row_law, row_rot=row_rot,
              a_dyn_res=a_dyn_res, mu_res=mu_res,
              dyn_act=dact, dyn_eps=1e-4, gate_act=GATE_SIGMOID)

    groups = G if G is not None else H
    if LAW_A_DYN in laws:
        kw["a_dyn"] = r(B, N if a_dyn_res == "n" else H, L, scale=0.5).float()
    if LAW_MU in laws:
        kw["mu"] = r(B, N if mu_res == "n" else H, L, scale=0.3).float()
        kw["mu_gain"] = r(H, P, N, scale=0.3).float()
        if mu_init:
            kw["mu_init"] = r(B, N if mu_res == "n" else H, scale=0.3).float()

    if rot or row_rot.any():
        kw["theta"] = r(B, pairs, L, scale=0.4).float()
        if theta_init:
            kw["theta_init"] = r(B, H, pairs, scale=1.0).float()
    if gates:
        kw["g_B"] = r(B, N, L, scale=0.7)
        kw["g_C"] = r(B, N, L, scale=0.7)
    if biases:
        kw["beta_a"] = r(groups, N, scale=0.2).float()
        kw["beta_b"] = r(groups, N, scale=0.3).float()
        kw["beta_c"] = r(groups, N, scale=0.3).float()
        kw["delta_bias"] = r(H, scale=0.2).float()
        if gates:
            # `G_gate` is separate from `G` on purpose: every static tensor
            # declares its own group count, and tying all five to one
            # of them is what let the kernels index the gate bias through
            # beta_b's group -- an out-of-bounds read at G_gate < G.
            gate_groups = G_gate if G_gate is not None else groups
            kw["beta_gB"] = r(gate_groups, N, scale=0.3).float()
            kw["beta_gC"] = r(gate_groups, N, scale=0.3).float()
    if delta_w == "head":
        kw["delta_w"] = r(B, H, L, scale=0.5)
    elif delta_w == "chan":
        kw["delta_w"] = r(B, H, P, L, scale=0.5)
    if h_init:
        kw["h_init"] = r(B, H, P, N, scale=0.5).float()

    # --- the pair tie ----------------------------------
    # `layout.MODULUS_FACTORS` is the list this mirrors; anything added there
    # has to be tied here or the checked suite stops passing, which is the
    # point of keeping the two next to each other.
    pair_rot = row_rot[0::2] & row_rot[1::2]
    if pair_rot.any():
        if "A_log" not in untie:
            a_log = _mh_pair_tie(a_log, -1, pair_rot)
        for name in ("beta_a", "mu_gain"):
            if kw.get(name) is not None and name not in untie:
                kw[name] = _mh_pair_tie(kw[name], -1, pair_rot)
        if a_dyn_res == "n" and kw.get("a_dyn") is not None \
                and "a_dyn" not in untie:
            kw["a_dyn"] = _mh_pair_tie(kw["a_dyn"], 1, pair_rot)
        if mu_res == "n":
            for name in ("mu", "mu_init"):
                if kw.get(name) is not None and name not in untie:
                    kw[name] = _mh_pair_tie(kw[name], 1, pair_rot)

    return (x, delta, a_log, b_tok, c_tok), kw


def mh_split_shape(spec):
    """Split a case spec into ``(shape_kwargs, feature_kwargs)``.

    The CASES lists carry B/H/P/N/L inline with the feature switches;
    ``mh_make_inputs`` takes them as ordinary keywords, so this is only
    here to keep the two groups readable at the call site.
    """
    spec = dict(spec)
    shape = {k: spec.pop(k) for k in ("B", "H", "P", "N", "L") if k in spec}
    return shape, spec


U64 = 2.0 ** -53


def assert_fp64_routes_close(actual, expected, depth, what="fp64 routes"):
    """Normwise ||actual - expected|| / ||expected|| < 2 * depth * u64."""
    bound = 2 * depth * U64
    err = ((actual - expected).norm() / expected.norm()).item()
    assert err < bound, f"{what}: normwise rel {err:.2e} >= 2*{depth}*u64 = {bound:.1e}"


def parameter_grads(model):
    """Snapshot of every populated ``.grad`` keyed by parameter name."""
    return {name: (p.grad.detach().clone() if p.grad is not None else None)
            for name, p in model.named_parameters()}


def worst_parameter_grad_error(actual, expected, min_numel=2, min_norm=1e-12):
    """(error, name) of the worst relative parameter-gradient mismatch.

    Scalar parameters (``numel < min_numel``) are skipped for the *numeric*
    comparison: a single scalar accumulates the gradient of every token, so
    its relative error is dominated by catastrophic cancellation, not by the
    scan.

    A VANISHED gradient is not a skip.  When one side has a gradient and the
    other has ``None``, the backend stopped propagating into that parameter
    altogether — the loudest possible failure of a fold/prologue migration,
    and one that a "no tensor, nothing to compare" skip would score as 0.0.
    Such a parameter returns ``inf`` immediately (before the ``min_numel``
    filter, so scalars are covered too), which makes the caller's
    ``assert err < tol`` trip with the dead parameter's name.  Both sides
    ``None`` stays a legitimate skip: that is a parameter the config does
    not use.
    """
    worst = (0.0, "<none>")
    for name, ref in expected.items():
        got = actual.get(name)
        if ref is None and got is None:
            continue
        if ref is None or got is None:
            dead = "actual" if got is None else "expected"
            return (float("inf"), f"{name} (gradient is None on the {dead} side)")
        if ref.numel() < min_numel:
            continue
        if ref.double().norm() <= min_norm:
            continue
        err = relative(got, ref)
        if err > worst[0]:
            worst = (err, name)
    return worst


def _rotate_pairs_independent(t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """R(theta)^T on adjacent pairs of `t`'s N axis (dim 2), independent of
    `kernels/ssd_mimo/reference._rot_fold_pairs`.

    t: [B, L, N] or [B, L, N, R]; theta: [B, L, N//2], broadcast over the
    trailing R axis when present.  Convention (matches the kernel's own
    fold, contract-defined, not incidental): e' = cos*e + sin*o,
    o' = -sin*e + cos*o.
    """
    has_tail = t.dim() == 4
    e = t[:, :, 0::2, :] if has_tail else t[:, :, 0::2]
    o = t[:, :, 1::2, :] if has_tail else t[:, :, 1::2]
    c, s = torch.cos(theta), torch.sin(theta)
    if has_tail:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    out = torch.empty_like(t)
    if has_tail:
        out[:, :, 0::2, :] = c * e + s * o
        out[:, :, 1::2, :] = -s * e + c * o
    else:
        out[:, :, 0::2] = c * e + s * o
        out[:, :, 1::2] = -s * e + c * o
    return out


def ssd_theta_reference(x_in, delta, b_tok, c_tok, a_log, theta,
                        bc_per_head):
    """fp64 oracle for `ssd_mimo_scan`'s raw per-timestep `theta` kwarg:
    rotate B/C by the given angle, THEN run the independent token-by-token
    recurrence (`ssd_diff_reference`).  `bc_per_head=True` (a group axis on
    B/C) is not supported — `theta` is head-shared by construction, so the
    feature never combines with it in production either.
    """
    if bc_per_head:
        raise NotImplementedError(
            "ssd_theta_reference: theta is head-shared; bc_per_head is out "
            "of scope for this oracle")
    b_rot = _rotate_pairs_independent(b_tok.double(), theta.double())
    c_rot = _rotate_pairs_independent(c_tok.double(), theta.double())
    return ssd_diff_reference(x_in.double(), delta.double(), b_rot, c_rot,
                              a_log.double(), False)


def decode_by_stepping(module, x, **state_kwargs):
    """``(full_sequence_forward, token_by_token_decode)`` for any module
    carrying the unified step/init_state contract (chunk S1).

    The contract is identical for a mixer, a ``ResidualBlock`` and a
    ``BlockStack`` — ``y, state = m.step(x_t, state)`` seeded by
    ``state = m.init_state(B, ...)`` — which is exactly why the wrappers
    can be scored by the same helper as the mixers they wrap: the stack's
    per-layer state LIST is just the state object at its level.

    Runs under ``no_grad`` (decode is an inference path) and returns both
    tensors shaped ``(B, L, d)`` so the caller owns the tolerance: fp64 on
    the torch backends is exact to ~1e-8, while anything routed through a
    float32 kernel (a chart-based fold, the SSD-MIMO fused scan) is a
    float32-level promise however precise the surrounding module is.
    """
    with torch.no_grad():
        full = module(x)
        state = module.init_state(x.shape[0], **state_kwargs)
        stepped = []
        for token in x.unbind(dim=1):
            y_t, state = module.step(token, state)
            stepped.append(y_t)
    return full, torch.stack(stepped, dim=1)
