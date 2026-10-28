"""Closed-form witnesses for the transported-endpoint law, from the paper.

Paper sec. 2 eq. (exp_trapezoidal_recurrence) / sec. 5 eq. (hm_disc):

    h_t = A_t h_{t-1} + Delta_t [ lambda_t v_t + (1 - lambda_t) A_t v_{t-1} ],
    v_{-1} = 0,   y_t = C_t^T h_t,

with A_t the FULL physical transition of a rotating pair, a_t R(dTheta_t).
The numbers below are evaluated with scalar arithmetic on a single N = 2
pair over two tokens -- no production helper, no oracle module, no folded
coordinates -- so they anchor the module AND `tests/test_ssd_endpoint.py`'s
physical oracle to the displayed equation rather than to each other.

Conventions pinned here (the module's, documented in `SSDMimoSSM.step`):
the scan-frame fold rotates a pair by [[c, s], [-s, c]] = R(-Theta) in the
standard R(phi) = [[c, -s], [s, c]]; hence the physical transition rotates
by R(+dTheta_t) and, with neither bias nor static offset, the physical
write/read maps are the raw B_t / C_t.  With a complex row block the biases join
the raw (normalised) maps BEFORE the rotation, so the physical maps are
norm(B_t) + beta and norm(C_t) + gamma.
"""
import math

import pytest
import torch

from _tolerance import assert_close_terms, terms_abs
from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig


def _R(phi):
    c, s = math.cos(phi), math.sin(phi)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def _softplus(v):
    return math.log1p(math.exp(v))


def _sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


def _rms(vec, weight, eps):
    # nn.RMSNorm over the state axis: v / sqrt(mean(v^2) + eps) * w
    scale = 1.0 / math.sqrt(sum(x * x for x in vec) / len(vec) + eps)
    return torch.tensor([x * scale * w for x, w in zip(vec, weight)],
                        dtype=torch.float64)


#: A7 worst-case rounding depth gamma_{n+m}/u of ONE arm of the two-token,
#: N = 2 witness per output entry (the hand arm and every module/oracle arm
#: evaluate the same law; chained relative errors add):
#:   B_t / C_t: RMSNorm (2-term mean + eps, sqrt, div, weight) + bias     7
#:   per token h_t: n = 5 summands (2-term A h dot, v, 2-term A v dot),
#:     m = 13 factors (a_t, cos/sin, Delta, lam or 1-lam, B_t's 7, x_t);
#:     two tokens                                                     2*18
#:   readout: C_t (7) + R(nu) 2-dot + C^T 2-dot (the fold R(-Theta) 2-dot
#:     for h is smaller)                                                11
#:   arguments of exp / cos / sin / softplus / sigmoid / tanh that carry
#:     rounding: |a Delta| <= 2, |Theta| <= 3 over two tokens              5
_C_WITNESS = 7 + 2 * 18 + 11 + 5


def _hand_terms(steps, R_abs_read=None):
    """sum|terms| of the hand arm, by the same loop in absolute arithmetic.

    ``steps``: per token (B_abs, C_abs, x_t, delta, lam, a_t, d_theta), the
    absolute summands of B_t / C_t (|RMSNorm entry| + |bias|).  A subtraction
    counts both summands (1 - lam -> 1 + lam); |A_t| = a_t |R(dTheta_t)|.
    Returns (y terms [2], folded-h terms [2])."""
    h = torch.zeros(2, dtype=torch.float64)
    v_prev = torch.zeros(2, dtype=torch.float64)
    theta_abs, ys = 0.0, []
    for B_abs, C_abs, x_t, delta, lam, a_t, d_theta in steps:
        A_abs = a_t * _R(d_theta).abs()
        v = B_abs * abs(x_t)
        h = A_abs @ h + delta * (lam * v + (1 + lam) * (A_abs @ v_prev))
        read = h if R_abs_read is None else R_abs_read @ h
        ys.append(float(C_abs @ read))
        v_prev = v
        theta_abs += d_theta
    return (torch.tensor(ys, dtype=torch.float64).view(1, 2, 1),
            (_R(-theta_abs).abs() @ h).view(1, 1, 1, 2))


def _module_terms(m, x, u):
    """sum|terms| of the module's sequence route and its step route."""
    def seq(x_, u_):
        y, st = m.scan(x_, sel_x=u_, return_state=True)
        return y, st["hidden_state"]

    def step(x_, u_):
        state = m.init_state(1, dtype=x_.dtype)
        ys = []
        for t in range(x_.shape[1]):
            yt, state = m.scan_step(x_[:, t], state, sel_x=u_[:, t])
            ys.append(yt)
        return torch.stack(ys, 1), state["hidden_state"]

    return terms_abs(seq, x, u).terms, terms_abs(step, x, u).terms


def _check_witness(m, x, u, want_y, want_h_folded, hand_y, hand_h):
    """Every arm against the hand numbers.  A7: u=fp64 (fp64 formulations
    of one law), bound gamma(arm) + gamma(hand) = c*u*(terms_arm +
    terms_hand), c = _C_WITNESS."""
    from test_ssd_endpoint import physical_oracle
    y, h_final, y_step, h_step = _run_module(m, x, u)
    (ty, th), (tsy, tsh) = _module_terms(m, x, u)
    h0 = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    oy, oh = physical_oracle(m, x, u, h0)
    toy, toh = terms_abs(lambda x_, u_, h_: physical_oracle(m, x_, u_, h_),
                         x, u, h0).terms
    for got, want, t_arm, t_hand, name in (
            (y, want_y, ty, hand_y, "y"), (y_step, want_y, tsy, hand_y, "y step"),
            (h_final, want_h_folded, th, hand_h, "h"),
            (h_step, want_h_folded, tsh, hand_h, "h step"),
            (oy, want_y, toy, hand_y, "oracle y"),
            (oh, want_h_folded, toh, hand_h, "oracle h")):
        assert_close_terms(got, want, t_arm + t_hand, "fp64", _C_WITNESS,
                           name=name)


def _run_module(m, x, u):
    y = m.scan(x, sel_x=u)
    h_final = m.h_final
    state = m.init_state(1, dtype=x.dtype)
    ys = []
    for t in range(x.shape[1]):
        yt, state = m.scan_step(x[:, t], state, sel_x=u[:, t])
        ys.append(yt)
    return y, h_final, torch.stack(ys, 1), state["hidden_state"]


def test_m3_ladder_and_offset_two_token_witness():
    """complex rows (static rate) with the two extras ON: learned per-pair frequency
    omega (theta_freq) and the static read offset nu (theta_h), static rate:
    dTheta_t = -omega tanh(ang_t) Delta_t,  y_t = (C_t+gamma)^T R(nu) h_t."""
    m = SSDMimoSSM(SSDMimoConfig(skip=False, d_inner=1, d_conv=0, d_state=2, n_groups=1, n_heads=1, d_ssm_in=1, d_ssm_out=1,
                   d_model=1, proj_source="layer_input", bc_conv=False,
                   row_blocks=[{"frac": 1.0, "decay": "head_static", "complex": True}],
                   b_gate=False, c_gate=False), execution={"backend": "torch", "precision": "ieee"}).double()
    eps = 1e-5
    m.B_norm.eps = m.C_norm.eps = eps
    wB, wC = (0.8, -0.3), (0.5, 1.1)
    nB, nC = (1.0, 1.0), (1.0, 1.0)
    beta, gamma = (0.15, -0.25), (-0.1, 0.2)
    w_dt, b_dt = 0.4, -0.2
    w_th, omega, nu = 0.6, 2.3, 0.7
    w_lam, b_lam = -0.7, 0.3
    with torch.no_grad():
        m.B_proj.weight.copy_(torch.tensor([[wB[0]], [wB[1]]], dtype=torch.float64))
        m.C_proj.weight.copy_(torch.tensor([[wC[0]], [wC[1]]], dtype=torch.float64))
        m.B_bias.copy_(torch.tensor(beta, dtype=torch.float64).view(1, 1, 2, 1))
        m.C_bias.copy_(torch.tensor(gamma, dtype=torch.float64).view(1, 1, 2, 1))
        m.dt_proj.weight.fill_(w_dt); m.dt_proj.bias.fill_(b_dt)
        m.theta_proj.weight.fill_(w_th)
        m.theta_freq.fill_(math.atanh(omega / math.pi)); m.theta_h.fill_(nu)
        m.trap_proj.weight.fill_(w_lam); m.trap_proj.bias.fill_(b_lam)
        m.A_log.zero_()            # rate = exp(0) = 1
    u_vals, x_vals = (1.0, -0.5), (0.7, -1.3)
    u = torch.tensor(u_vals, dtype=torch.float64).view(1, 2, 1)
    x = torch.tensor(x_vals, dtype=torch.float64).view(1, 2, 1)

    h_prev = torch.zeros(2, dtype=torch.float64)
    v_prev = torch.zeros(2, dtype=torch.float64)
    theta_abs = 0.0
    want_y = []
    steps = []
    for u_t, x_t in zip(u_vals, x_vals):
        B_t = _rms((wB[0] * u_t, wB[1] * u_t), nB, eps) + torch.tensor(beta, dtype=torch.float64)
        C_t = _rms((wC[0] * u_t, wC[1] * u_t), nC, eps) + torch.tensor(gamma, dtype=torch.float64)
        delta = _softplus(w_dt * u_t + b_dt)
        lam = _sigmoid(w_lam * u_t + b_lam)
        a_t = math.exp(-1.0 * delta)
        d_theta = -omega * math.tanh(w_th * u_t) * delta
        theta_abs += d_theta
        A_t = a_t * _R(d_theta)
        v_t = B_t * x_t
        h_t = A_t @ h_prev + delta * (lam * v_t + (1 - lam) * (A_t @ v_prev))
        want_y.append(float(C_t @ (_R(nu) @ h_t)))
        h_prev, v_prev = h_t, v_t
        steps.append((
            _rms((wB[0] * u_t, wB[1] * u_t), nB, eps).abs() + torch.tensor(beta).abs(),
            _rms((wC[0] * u_t, wC[1] * u_t), nC, eps).abs() + torch.tensor(gamma).abs(),
            x_t, delta, lam, a_t, d_theta))
    want_y = torch.tensor(want_y, dtype=torch.float64).view(1, 2, 1)
    want_h_folded = (_R(-theta_abs) @ h_prev).view(1, 1, 1, 2)
    hand_y, hand_h = _hand_terms(steps, R_abs_read=_R(nu).abs())
    _check_witness(m, x, u, want_y, want_h_folded, hand_y, hand_h)


def test_m3_endpoint_two_token_witness():
    """complex rows, head_dynamic law: dynamic rate a_t = exp(-heavy_tail(a_raw) Delta_t),
    dTheta_t = -pi tanh(ang_t) Delta_t on the write step, biases folded
    BEFORE the rotation (physical maps norm(B)+beta, norm(C)+gamma)."""
    m = SSDMimoSSM(SSDMimoConfig(skip=False, d_inner=1, d_conv=0, d_state=2, n_groups=1, n_heads=1, d_ssm_in=1, d_ssm_out=1,
                   d_model=1, proj_source="layer_input", bc_conv=False,
                   write_rule="trapezoid",
                   row_blocks=[{"frac": 1.0, "decay": "head_dynamic", "complex": True}],
                   use_theta_h=False, learn_theta_freq=False,
                   b_gate=False, c_gate=False), execution={"backend": "torch", "precision": "ieee"}).double()
    eps = 1e-5
    m.B_norm.eps = m.C_norm.eps = eps
    wB, wC = (0.8, -0.3), (0.5, 1.1)
    nB, nC = (1.2, 0.9), (0.7, 1.3)          # RMSNorm weights
    beta, gamma = (0.15, -0.25), (-0.1, 0.2)  # per-head pre-rotation biases
    w_dt, b_dt = 0.4, -0.2
    w_a, b_a = 0.5, 0.1
    w_th = 0.6                                # m3 angle proj has no bias
    w_lam, b_lam = -0.7, 0.3
    with torch.no_grad():
        m.B_proj.weight.copy_(torch.tensor([[wB[0]], [wB[1]]], dtype=torch.float64))
        m.C_proj.weight.copy_(torch.tensor([[wC[0]], [wC[1]]], dtype=torch.float64))
        m.B_norm.weight.copy_(torch.tensor(nB, dtype=torch.float64)); m.C_norm.weight.copy_(torch.tensor(nC, dtype=torch.float64))
        m.B_bias.copy_(torch.tensor(beta, dtype=torch.float64).view(1, 1, 2, 1))
        m.C_bias.copy_(torch.tensor(gamma, dtype=torch.float64).view(1, 1, 2, 1))
        m.dt_proj.weight.fill_(w_dt); m.dt_proj.bias.fill_(b_dt)
        m.a_dyn_proj.weight.fill_(w_a); m.a_dyn_proj.bias.fill_(b_a)
        m.theta_proj.weight.fill_(w_th)
        m.trap_proj.weight.fill_(w_lam); m.trap_proj.bias.fill_(b_lam)
    assert m.A_log is None                    # the m3 rate is dynamic
    u_vals, x_vals = (1.0, -0.5), (0.7, -1.3)
    u = torch.tensor(u_vals, dtype=torch.float64).view(1, 2, 1)
    x = torch.tensor(x_vals, dtype=torch.float64).view(1, 2, 1)

    h_prev = torch.zeros(2, dtype=torch.float64)
    v_prev = torch.zeros(2, dtype=torch.float64)
    theta_abs = 0.0
    want_y = []
    steps = []
    for u_t, x_t in zip(u_vals, x_vals):
        B_t = _rms((wB[0] * u_t, wB[1] * u_t), nB, eps) + torch.tensor(beta, dtype=torch.float64)
        C_t = _rms((wC[0] * u_t, wC[1] * u_t), nC, eps) + torch.tensor(gamma, dtype=torch.float64)
        delta = _softplus(w_dt * u_t + b_dt)
        raw_a = w_a * u_t + b_a
        a_mag = max(1 + raw_a if raw_a >= 0 else 1 / (1 - raw_a), 1e-4)
        lam = _sigmoid(w_lam * u_t + b_lam)
        a_t = math.exp(-a_mag * delta)
        d_theta = -math.pi * math.tanh(w_th * u_t) * delta   # write-step clock
        theta_abs += d_theta
        A_t = a_t * _R(d_theta)
        v_t = B_t * x_t
        h_t = A_t @ h_prev + delta * (lam * v_t + (1 - lam) * (A_t @ v_prev))
        want_y.append(float(C_t @ h_t))
        h_prev, v_prev = h_t, v_t
        steps.append((
            _rms((wB[0] * u_t, wB[1] * u_t), nB, eps).abs() + torch.tensor(beta).abs(),
            _rms((wC[0] * u_t, wC[1] * u_t), nC, eps).abs() + torch.tensor(gamma).abs(),
            x_t, delta, lam, a_t, d_theta))
    want_y = torch.tensor(want_y, dtype=torch.float64).view(1, 2, 1)
    want_h_folded = (_R(-theta_abs) @ h_prev).view(1, 1, 1, 2)
    hand_y, hand_h = _hand_terms(steps)
    _check_witness(m, x, u, want_y, want_h_folded, hand_y, hand_h)


def test_witness_discriminates_the_old_clock_blend_law():
    """The retired law scaled the lagged tap with the PREVIOUS clock and
    did not transport it; the witness numbers must reject it."""
    m = SSDMimoSSM(SSDMimoConfig(skip=False, d_inner=1, d_conv=0, d_state=2, n_groups=1, n_heads=1, d_ssm_in=1, d_ssm_out=1,
                   d_model=1, proj_source="layer_input", bc_conv=False,
                   row_blocks=[{"frac": 1.0, "decay": "head_static", "complex": True}], use_theta_h=False,
                   learn_theta_freq=False,
                   b_gate=False, c_gate=False), execution={"backend": "torch", "precision": "ieee"}).double()
    eps = 1e-5
    m.B_norm.eps = m.C_norm.eps = eps
    with torch.no_grad():
        m.B_proj.weight.copy_(torch.tensor([[0.8], [-0.3]], dtype=torch.float64))
        m.C_proj.weight.copy_(torch.tensor([[0.5], [1.1]], dtype=torch.float64))
        m.dt_proj.weight.fill_(0.4); m.dt_proj.bias.fill_(-0.2)
        m.theta_proj.weight.fill_(0.6)
        m.trap_proj.weight.fill_(-0.7); m.trap_proj.bias.fill_(0.3)
        m.A_log.zero_()
    u = torch.tensor([1.0, -0.5], dtype=torch.float64).view(1, 2, 1)
    x = torch.tensor([0.7, -1.3], dtype=torch.float64).view(1, 2, 1)
    y = m.scan(x, sel_x=u)
    # wrong law: h_t = A_t h_{t-1} + lam_t Delta_t v_t + (1-lam_t) Delta_{t-1} v_{t-1}
    h_prev = torch.zeros(2, dtype=torch.float64); v_prev = h_prev.clone(); d_prev = 0.0
    wrong = []
    for u_t, x_t in zip((1.0, -0.5), (0.7, -1.3)):
        B_t = _rms((0.8 * u_t, -0.3 * u_t), (1.0, 1.0), eps)
        C_t = _rms((0.5 * u_t, 1.1 * u_t), (1.0, 1.0), eps)
        delta = _softplus(0.4 * u_t - 0.2); lam = _sigmoid(-0.7 * u_t + 0.3)
        A_t = math.exp(-delta) * _R(-math.pi * math.tanh(0.6 * u_t) * delta)
        v_t = B_t * x_t
        h_t = A_t @ h_prev + lam * delta * v_t + (1 - lam) * d_prev * v_prev
        wrong.append(float(C_t @ h_t)); h_prev, v_prev, d_prev = h_t, v_t, delta
    assert abs(float(y.detach()[0, 1, 0]) - wrong[1]) > 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["m3plus", "m3"])
def test_cuda_kernel_reproduces_the_hand_witness(require_cuda, mode):
    """The production CUDA route against the SAME closed-form numbers, in
    fp32 with IEEE operands (N = 2 rides the kernel's zero-padded 16-row
    tile, which the contract declares exact)."""
    import copy
    if mode == "m3plus":
        m = SSDMimoSSM(SSDMimoConfig(skip=False, d_inner=1, d_conv=0, d_state=2, n_groups=1, n_heads=1, d_ssm_in=1, d_ssm_out=1,
                       d_model=1, proj_source="layer_input", bc_conv=False,
                       row_blocks=[{"frac": 1.0, "decay": "head_static", "complex": True}],
                       b_gate=False, c_gate=False), execution={"backend": "torch", "precision": "ieee"}).double()
        m.B_norm.eps = m.C_norm.eps = 1e-5
        with torch.no_grad():
            m.B_proj.weight.copy_(torch.tensor([[0.8], [-0.3]], dtype=torch.float64))
            m.C_proj.weight.copy_(torch.tensor([[0.5], [1.1]], dtype=torch.float64))
            m.B_bias.copy_(torch.tensor((0.15, -0.25), dtype=torch.float64).view(1, 1, 2, 1))
            m.C_bias.copy_(torch.tensor((-0.1, 0.2), dtype=torch.float64).view(1, 1, 2, 1))
            m.dt_proj.weight.fill_(0.4); m.dt_proj.bias.fill_(-0.2)
            m.theta_proj.weight.fill_(0.6)
            m.theta_freq.fill_(math.atanh(2.3 / math.pi)); m.theta_h.fill_(0.7)
            m.trap_proj.weight.fill_(-0.7); m.trap_proj.bias.fill_(0.3)
            m.A_log.zero_()
    else:
        m = SSDMimoSSM(SSDMimoConfig(skip=False, d_inner=1, d_conv=0, d_state=2, n_groups=1, n_heads=1, d_ssm_in=1, d_ssm_out=1,
                       d_model=1, proj_source="layer_input", bc_conv=False,
                       write_rule="trapezoid",
                       row_blocks=[{"frac": 1.0, "decay": "head_dynamic", "complex": True}],
                       use_theta_h=False, learn_theta_freq=False,
                       b_gate=False, c_gate=False), execution={"backend": "torch", "precision": "ieee"}).double()
        m.B_norm.eps = m.C_norm.eps = 1e-5
        with torch.no_grad():
            m.B_proj.weight.copy_(torch.tensor([[0.8], [-0.3]], dtype=torch.float64))
            m.C_proj.weight.copy_(torch.tensor([[0.5], [1.1]], dtype=torch.float64))
            m.B_norm.weight.copy_(torch.tensor((1.2, 0.9), dtype=torch.float64))
            m.C_norm.weight.copy_(torch.tensor((0.7, 1.3), dtype=torch.float64))
            m.B_bias.copy_(torch.tensor((0.15, -0.25), dtype=torch.float64).view(1, 1, 2, 1))
            m.C_bias.copy_(torch.tensor((-0.1, 0.2), dtype=torch.float64).view(1, 1, 2, 1))
            m.dt_proj.weight.fill_(0.4); m.dt_proj.bias.fill_(-0.2)
            m.a_dyn_proj.weight.fill_(0.5); m.a_dyn_proj.bias.fill_(0.1)
            m.theta_proj.weight.fill_(0.6)
            m.trap_proj.weight.fill_(-0.7); m.trap_proj.bias.fill_(0.3)
    u = torch.tensor([1.0, -0.5], dtype=torch.float64).view(1, 2, 1)
    x = torch.tensor([0.7, -1.3], dtype=torch.float64).view(1, 2, 1)
    k = copy.deepcopy(m).float().cuda()  # copy before forward retains a live graph
    want_y, want_h = m.scan(x, sel_x=u, return_state=True)
    want_h = want_h.hidden_state           # fp64 torch route == hand numbers (tests above)

    def seq(x_, u_):
        y_, st = m.scan(x_, sel_x=u_, return_state=True)
        return y_, st["hidden_state"]

    t_y, t_h = terms_abs(seq, x, u).terms
    k.set_execution(dict(backend="cuda"))
    y, h = k.scan(x.float().cuda(), sel_x=u.float().cuda(), return_state=True)
    h = h.hidden_state
    # A7: u=fp32 (the fp32 IEEE kernel vs the fp64 torch route; the fp64
    # arm's own rounding is 2^-29 of this and is dropped).  c = _C_WITNESS
    # (the kernel arm's arithmetic, same law)
    #   + 12: the kernel is fed fp32-ROUNDED inputs and weights (0.7, 0.8,
    #     -0.3, ... are not fp32-exact): one rounding per input factor of a
    #     term (u, x, wB|wC, beta|gamma, w_dt, b_dt, w_lam, b_lam, w_th,
    #     omega|w_a, b_a, nu);
    #   + 36: the SFU cos.approx / sin.approx (absolute error 2^-20.9 ~ 8.6 u
    #     on the unit-sum pair (|cos| + |sin| >= 1), plus the 2pi wrap's
    #     ~2 u) in each of the <= 4 rotations applied (two write steps, the
    #     read offset, the fold), a 2-term dot each: 4 * 9.
    c_kernel = _C_WITNESS + 12 + 36
    assert_close_terms(y.double().cpu(), want_y, t_y, "fp32", c_kernel, name="y")
    assert_close_terms(h.double().cpu(), want_h, t_h, "fp32", c_kernel, name="h")
