"""MH kernel under the single SSD rotation law.

Hand witness, written from the law and not from any reference module:

    Theta_t[b,h,k] = Sum_{s<=t} Delta_s[b,h] * theta_s[b,k]        (theta_delta)
    B~_t = R(Theta_t)   (B_t + beta_b[h])
    C~_t = R(Theta_t + theta_h[h,k]) (C_t + beta_c[h])              (read offset)
    h_t  = a_t h_{t-1} + Delta_t [lam_t x_t B~_t + (1-lam_t) a_t x_{t-1} B~_{t-1}]
    y_t  = <C~_t, h_t>

`Delta` is the EFFECTIVE step (softplus(delta + delta_bias) when enabled), so
the phase increment is the same head-and-token dependent clock ssd_mimo uses.
"""
import pytest
import torch
import torch.nn.functional as F

from _tolerance import DrawPool, draw_seeds, terms_abs
from _contract.mh_oracle import (amplification, assert_oracle_close, leaf_names,
                               mh_dims, oracle_pairs, oracle_terms)


def _witness(x, dt, al, bm, cm, theta, theta_h, bb, cb, lam, *, softplus,
             dbias):
    b, h, p, length = x.shape
    n = bm.shape[1]
    k = n // 2
    de = dt + dbias[None, :, None] if dbias is not None else dt
    if softplus:
        de = F.softplus(de)
    hh = torch.zeros(b, h, p, n, dtype=x.dtype, device=x.device)
    prev = torch.zeros_like(hh)
    phase = torch.zeros(b, h, k, dtype=x.dtype, device=x.device)
    outs = []

    def rotate(v, ang):                      # v (b,h,n), ang (b,h,k)
        pair = v.reshape(b, h, k, 2)
        co, si = ang.cos(), ang.sin()
        return torch.stack((co * pair[..., 0] - si * pair[..., 1],
                            si * pair[..., 0] + co * pair[..., 1]),
                           -1).reshape(b, h, n)

    for t in range(length):
        phase = phase + de[:, :, t, None] * theta[:, None, :, t]
        bv = rotate(bm[:, None, :, t] + bb[None], phase)
        cv = rotate(cm[:, None, :, t] + cb[None], phase + theta_h[None])
        drive = x[:, :, :, t, None] * bv[:, :, None, :]
        a = torch.exp(-al.exp()[None] * de[:, :, t, None, None])
        l_ = lam[:, :, t, None, None]
        hh = a * hh + de[:, :, t, None, None] * (l_ * drive + (1 - l_) * a * prev)
        outs.append((hh * cv[:, :, None, :]).sum(-1))
        prev = drive
    return torch.stack(outs, -1), hh


@pytest.mark.parametrize("p,length", [(1, 1), (1, 65), (4, 37), (4, 129)])
@pytest.mark.parametrize("softplus", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_mh_ssd_rotation_law_hand_witness(require_cuda, monkeypatch, p, length,
                                          softplus, checkpoint):
    """Kernel Y, final state and EVERY gradient against the hand-written law
    (fp64), and the torch reference against the same witness."""
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import mh_reference
    torch.manual_seed(1912)
    b, h, n = 2, 4, 8

    def rand(*shape, scale=0.2):
        return (torch.randn(*shape, device="cuda") * scale).requires_grad_()
    x = rand(b, h, p, length)
    dt = rand(b, h, length).sigmoid().detach().requires_grad_()
    al = rand(h, p, n // 2).repeat_interleave(2, -1).detach().requires_grad_()
    bm, cm = rand(b, n, length), rand(b, n, length)
    lam = rand(b, h, length).sigmoid().detach().requires_grad_()
    theta = rand(b, n // 2, length, scale=0.5)
    theta_h = rand(h, n // 2, scale=0.5)
    bb, cb = rand(h, n), rand(h, n)
    dbias = rand(h) if softplus else None
    vals = [x, dt, al, bm, cm, theta, theta_h, bb, cb, lam] + \
        ([dbias] if softplus else [])
    kw = dict(theta=theta, theta_delta=True, theta_h=theta_h, beta_b=bb,
              beta_c=cb, trap_lam=lam, delta_softplus=softplus,
              delta_bias=dbias)
    actual, final, _ = mh_scan(x, dt, al, bm, cm, return_state=True,
                               ckpt=checkpoint, **kw)
    wy, wh = torch.randn_like(actual), torch.randn_like(final)
    got = torch.autograd.grad((actual, final), vals, (wy, wh))
    # the hand witness (fp64) in absolute arithmetic: values, sum|terms|, and
    # every gradient through the witness's own graph
    wit = terms_abs(lambda *v: _witness(*v[:10], softplus=softplus,
                                        dbias=v[10] if softplus else None),
                    *vals, grad_outputs=(wy, wh))
    expected, hexp = wit.out
    names = ["x", "dt", "A_log", "B", "C", "theta", "theta_h", "beta_b",
             "beta_c", "trap_lam", "delta_bias"]
    dims, amp = mh_dims(vals[:5]), amplification(vals[:5], kw)
    # A7 class (b): the fp32 kernel vs the fp64 witness on the same inputs:
    # u=fp32, c = oracle_c (scan depth L + N + factors + a leaf's fanout,
    # typical sqrt, + the decay/angle amplification), per head / row.
    assert_oracle_close(list(zip(("y", "h_final"), (actual, final), wit.out,
                                 wit.terms))
                        + list(zip(names, got, wit.grads, wit.grad_terms)),
                        dims, amp, "fp32")
    # the torch reference implements the same law: two computed fp64
    # formulations that both round (u=fp64): terms = T_oracle + T_witness,
    # each from `terms_abs` on its own arm; the same depth / amplification
    rkw = {k_: (v.detach().double().requires_grad_() if isinstance(v, torch.Tensor)
                else v) for k_, v in kw.items()}
    ref_y, ref_h = mh_reference(*[v.detach().double() for v in (x, dt, al, bm, cm)],
                                **rkw)
    ora = oracle_terms(vals[:5], kw, (wy, wh))
    assert_oracle_close([("ref y", ref_y, expected, wit.terms[0] + ora.terms[0]),
                         ("ref h_final", ref_h, hexp, wit.terms[1] + ora.terms[1])],
                        dims, amp, "fp64")


@pytest.mark.parametrize("law", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_groups,gates", [("head", True),
                                               ("shared", True),
                                               ("shared", False)])
def test_mh_ssd_rotation_law_across_decay_laws(require_cuda, monkeypatch, law,
                                               dtype, bias_groups, gates):
    """The Delta-scaled phase and the read offset composed with every decay
    law, the gates, the per-channel write strength and z: kernel vs the torch
    reference, all gradients.

    `bias_groups="shared"` gives B/C and the gates head-shared (G=1) biases,
    which is what routes a gated rotating launch onto the pair maps
    (`functions._pair_maps`, kernel PMAP) and an ungated one onto the raw
    maps (`_pair_maps_raw`, backward PRAW); the route is asserted.  Per-head
    biases keep the in-kernel rotation (gate activation hoisted, GATE_RAW)."""
    monkeypatch.setenv('HYBRID_MH_ROTATION', 'in_scan')   # asserts the in-scan pair-map route
    import orbit_ssm.kernels.hybrid_mamba_mh.functions as fn_mod
    pair_maps, raw_maps = [], []
    real_pair_maps, real_raw = fn_mod._pair_maps, fn_mod._pair_maps_raw
    monkeypatch.setattr(fn_mod, "_pair_maps",
                        lambda *a, **k: pair_maps.append(1) or
                        real_pair_maps(*a, **k))
    monkeypatch.setattr(fn_mod, "_pair_maps_raw",
                        lambda *a, **k: raw_maps.append(1) or
                        real_raw(*a, **k))
    b, h, p, n, l = 2, 4, 4, 8, 65

    def r(*shape, scale=.1, io=False):
        return (torch.randn(*shape, device="cuda") * scale).to(
            dtype if io else torch.float32).requires_grad_()
    # A7 class (b): the kernel vs the fp64 oracle fed the same (IO-rounded)
    # inputs, per head / row (`assert_oracle_close`): fp32 terms-scaled,
    # u=fp32, c = oracle_c; bf16 the 16-bit policy (tol16, depth =
    # oracle_depth), >= 5 draws, each inside the bound, signed bias pooled.
    pool = DrawPool()
    for seed in draw_seeds(dtype, 612):
        pair_maps.clear()
        raw_maps.clear()
        _decay_law_draw(seed, r, b, h, p, n, l, law, dtype, bias_groups, gates,
                        pair_maps, raw_maps, pool)
    pool.assert_unbiased()


def _decay_law_draw(seed, r, b, h, p, n, l, law, dtype, bias_groups, gates,
                    pair_maps, raw_maps, pool):
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    torch.manual_seed(seed)
    x, dt, aa, bm, cm = r(b, h, p, l, io=True), r(b, h, l), r(h, p, n // 2), \
        r(b, n, l, io=True), r(b, n, l, io=True)
    aa = aa.repeat_interleave(2, -1).detach().requires_grad_()
    kw = dict(trap_lam=r(b, h, l).sigmoid().detach().requires_grad_(),
              delta_w=r(b, h, l), delta_bias=r(h),
              theta=r(b, n // 2, l, scale=.5), theta_delta=True,
              theta_h=r(h, n // 2, scale=.5),
              beta_b=r(h if bias_groups == "head" else 1, n),
              beta_c=r(h if bias_groups == "head" else 1, n),
              g_B=r(b, n, l) if gates else None,
              g_C=r(b, n, l) if gates else None,
              d_skip=r(h), z=r(b, h, p, l, io=True),
              row_law=torch.full((n,), law, device="cuda", dtype=torch.int32))
    if law == 1:
        kw['a_dyn'] = r(b, h, l); kw['a_dyn_res'] = 'h'
    elif law == 2:
        kw['mu'] = r(b, h, l); kw['mu_res'] = 'h'
        kw['mu_gain'] = r(h, p, n // 2).repeat_interleave(2, -1).detach().requires_grad_()
    args = [x, dt, aa, bm, cm]
    vals = args + [v for v in kw.values()
                   if isinstance(v, torch.Tensor) and v.requires_grad]
    actual = mh_scan(x, dt, aa, bm, cm, **kw)
    assert bool(pair_maps) == (bias_groups == "shared" and gates), pair_maps
    assert bool(raw_maps) == (bias_groups == "shared" and not gates), raw_maps
    weight = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, vals, weight, allow_unused=True)
    res = oracle_terms(args, kw, (weight, None))
    assert_oracle_close(oracle_pairs((actual,), ga, res, leaf_names(kw)),
                        mh_dims(args), amplification(args, kw), dtype,
                        pool=pool)


def test_mh_scan_rejects_head_offset_without_rotation(require_cuda):
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    b, h, p, n, l = 1, 2, 1, 4, 5
    x = torch.randn(b, h, p, l, device="cuda")
    dt = torch.rand(b, h, l, device="cuda")
    al = torch.zeros(h, p, n, device="cuda")
    bm = torch.randn(b, n, l, device="cuda")
    with pytest.raises(ValueError):
        mh_scan(x, dt, al, bm, bm, theta_h=torch.zeros(h, n // 2, device="cuda"))
    with pytest.raises(ValueError):
        mh_scan(x, dt, al, bm, bm, theta_delta=True)
