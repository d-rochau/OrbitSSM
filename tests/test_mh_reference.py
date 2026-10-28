"""Closes the "no oracle" gap an audit flagged on the
HybridMambaSSM flat rows (m3/m2) and the top-level row-mixture sum.

`src/orbit_ssm/ssm/hybrid_mamba.py::_m3_reference`/`_flat_reference` are the
flat laws' explicit-recurrence statement AND the production
`backend="sequential"`/`"torch"` route itself -- there is no second,
structurally different implementation to check them against, so scoring the
fused kernel or the module's own row-mixture sum against them is not
independent evidence.

`src/orbit_ssm/kernels/hybrid_mamba_mh/reference.py::m3_reference_cumprod` /
`::flat_reference_cumprod` / `::row_mixture_reference` are that second
derivation: the same divide-by-prefix-cumprod method `mh_reference_cumprod`
already uses for the HM rows, applied to the flat laws and to the top-level
sum. This file:

  1. checks the closed-form flat oracles against `_m3_reference`/
     `_flat_reference` directly, on synthetic inputs (including a resumed
     carry, i.e. the same "forward in two chunks" shape `step()` needs);
  2. checks the row-mixture SPLIT against a real mixed-law `HybridMambaSSM`
     in `backend="sequential"`, by intercepting the exact inputs its own
     halves compute and re-summing them independently (never re-using the
     module's own `y = y_hm + y_m3 + ...` line);
  3. checks `step()` against one `forward()` scan call, per law and mixed,
     the "does an optimised single-token step exist and agree with the
     scan" question the audit also asked.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from _tolerance import assert_close_terms, terms_abs
from orbit_ssm.kernels.hybrid_mamba_mh.reference import (
    flat_reference_cumprod, m3_reference_cumprod)
# Bound at IMPORT time, before any test monkeypatches
# `orbit_ssm.kernels.hybrid_mamba_mh.reference.mh_reference` -- this name
# stays the true original forever, so re-using it inside a captured-call
# replay loop (below) cannot recurse into the patched module attribute.
from orbit_ssm.kernels.hybrid_mamba_mh.reference import mh_reference as _mh_reference_unpatched
from orbit_ssm.ssm.hybrid_mamba import (HybridMambaSSM, _flat_reference,
                                         _m3_reference, HybridMambaConfig)

DT = torch.float64

# A7 for the closed-form (divide-by-prefix-cumprod) oracles against the loop
# references: u=fp64 (both arms fp64, sqrt(2) as both round).  A value sums
# n = L token terms of m <= L decays + 8 other factors (x, write, B, C, their
# biases, rotation, trap weight): sqrt(2 (n + m)).  The closed form reads
# every decay product as exp(S_t - S_s) of a cumulative log-decay sum S, so
# each argument carries rounding of |S_t| + |S_s| <= 2 max|S|: + 2 max|S|.
_FACTORS = 8


def _c_cumprod(log_decay):
    L = log_decay.shape[-1]
    amp = 2 * log_decay.abs().sum(-1).max().item()
    return math.sqrt(2 * (L + L + _FACTORS)) + amp


def _relative(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------------------
# 1. closed-form flat oracles vs the loop references, on synthetic inputs
# ---------------------------------------------------------------------------

def _m3_inputs(seed, Bsz=2, L=17, H=3, P=4, n3=8, Ka=2, rotate=True):
    g = torch.Generator().manual_seed(seed)
    def r(*shape, scale=0.3):
        return torch.randn(*shape, generator=g, dtype=DT) * scale
    Bt = r(Bsz, L, 1, n3)
    Ct = r(Bsz, L, 1, n3)
    v = r(Bsz, L, H, P)
    ADT = -r(Bsz, H, L, scale=0.1).abs()          # <= 0
    DTt = r(Bsz, H, L, scale=0.1).abs() + 1e-3    # > 0
    Trap = r(Bsz, H, L)                            # raw, pre-sigmoid
    K_bias = r(H, n3)
    Q_bias = r(H, n3)
    angles = r(Bsz, L, H, Ka) if rotate else torch.zeros(Bsz, L, H, Ka, dtype=DT)
    return Bt, Ct, v, ADT, DTt, Trap, K_bias, Q_bias, angles


@pytest.mark.parametrize("rotate", [False, True])
def test_m3_cumprod_matches_loop_reference(rotate):
    args = _m3_inputs(101, rotate=rotate)
    Y_loop = _m3_reference(*args)
    Y_cp = m3_reference_cumprod(*args)
    terms = terms_abs(_m3_reference, *args).terms
    # A7: u=fp64, c=_c_cumprod(ADT) (module note above)
    assert_close_terms(Y_cp, Y_loop, terms, "fp64", _c_cumprod(args[3]), name="Y")


@pytest.mark.parametrize("rotate", [False, True])
def test_m3_cumprod_matches_loop_reference_resumed(rotate):
    """Split the sequence in two and resume from a carried state -- the
    exact continuation `step()` needs."""
    Bt, Ct, v, ADT, DTt, Trap, K_bias, Q_bias, angles = _m3_inputs(202)
    L = Bt.shape[1]
    cut = 7

    def half(t, sl):
        return t[:, sl] if t.dim() > 2 and t.shape[1] == L else \
            (t[:, :, sl] if t.dim() >= 3 and t.shape[2] == L else t)

    Y_full_loop, st_full = _m3_reference(Bt, Ct, v, ADT, DTt, Trap, K_bias,
                                         Q_bias, angles, return_state=True)
    Y_full_cp, st_full_cp = m3_reference_cumprod(
        Bt, Ct, v, ADT, DTt, Trap, K_bias, Q_bias, angles, return_state=True)
    y_terms, st_terms = terms_abs(_m3_reference, Bt, Ct, v, ADT, DTt, Trap, K_bias,
                                  Q_bias, angles, return_state=True).terms
    # A7: u=fp64, c=_c_cumprod(ADT) (module note above); the two-piece
    # resumes below sum the same terms, so the full run's terms bound them
    c = _c_cumprod(ADT)
    assert_close_terms(Y_full_cp, Y_full_loop, y_terms, "fp64", c, name="Y")
    for k in ("S", "kp", "vp", "phase"):
        assert_close_terms(st_full_cp[k], st_full[k], st_terms[k], "fp64", c, name=k)

    sl1, sl2 = slice(0, cut), slice(cut, L)
    Y1_loop, st1 = _m3_reference(Bt[:, sl1], Ct[:, sl1], v[:, sl1], ADT[:, :, sl1],
                                 DTt[:, :, sl1], Trap[:, :, sl1], K_bias, Q_bias,
                                 angles[:, sl1], return_state=True)
    Y2_loop = _m3_reference(Bt[:, sl2], Ct[:, sl2], v[:, sl2], ADT[:, :, sl2],
                            DTt[:, :, sl2], Trap[:, :, sl2], K_bias, Q_bias,
                            angles[:, sl2], S_init=st1["S"], kp_init=st1["kp"],
                            vp_init=st1["vp"], phase_init=st1["phase"])
    Y1_cp, st1_cp = m3_reference_cumprod(
        Bt[:, sl1], Ct[:, sl1], v[:, sl1], ADT[:, :, sl1], DTt[:, :, sl1],
        Trap[:, :, sl1], K_bias, Q_bias, angles[:, sl1], return_state=True)
    Y2_cp = m3_reference_cumprod(
        Bt[:, sl2], Ct[:, sl2], v[:, sl2], ADT[:, :, sl2], DTt[:, :, sl2],
        Trap[:, :, sl2], K_bias, Q_bias, angles[:, sl2], S_init=st1_cp["S"],
        kp_init=st1_cp["kp"], vp_init=st1_cp["vp"], phase_init=st1_cp["phase"])
    Y_cat_cp = torch.cat([Y1_cp, Y2_cp], dim=1)
    assert_close_terms(Y_cat_cp, torch.cat([Y1_loop, Y2_loop], dim=1), y_terms,
                       "fp64", c, name="Y resumed vs resumed loop")
    assert_close_terms(Y_cat_cp, Y_full_loop, y_terms, "fp64", c,
                       name="Y resumed vs full loop")


def _flat_inputs(seed, Bsz=2, L=17, H=3, P=4, n=8):
    g = torch.Generator().manual_seed(seed)
    def r(*shape, scale=0.3):
        return torch.randn(*shape, generator=g, dtype=DT) * scale
    Bt = r(Bsz, L, n)
    Ct = r(Bsz, L, n)
    v = r(Bsz, L, H, P)
    dt = r(Bsz, H, L, scale=0.1).abs() + 1e-3
    A_log = r(H, scale=0.5)
    log_a = -(dt * A_log.exp()[None, :, None])              # the m2 law
    write = dt
    B_bias = r(H, n)
    C_bias = r(H, n)
    return Bt, Ct, v, log_a, write, B_bias, C_bias


def test_flat_cumprod_matches_loop_reference():
    args = _flat_inputs(303)
    Y_loop = _flat_reference(*args)
    Y_cp = flat_reference_cumprod(*args)
    terms = terms_abs(_flat_reference, *args).terms
    # A7: u=fp64, c=_c_cumprod(log_a) (module note above)
    assert_close_terms(Y_cp, Y_loop, terms, "fp64", _c_cumprod(args[3]), name="Y")


def test_flat_cumprod_matches_loop_reference_resumed():
    Bt, Ct, v, log_a, write, B_bias, C_bias = _flat_inputs(404)
    L = Bt.shape[1]
    cut = 6
    Y_full, S_full = _flat_reference(Bt, Ct, v, log_a, write, B_bias, C_bias,
                                     return_state=True)
    Y1, S1 = _flat_reference(Bt[:, :cut], Ct[:, :cut], v[:, :cut],
                             log_a[:, :, :cut], write[:, :, :cut], B_bias, C_bias,
                             return_state=True)
    Y2, _ = _flat_reference(Bt[:, cut:], Ct[:, cut:], v[:, cut:],
                            log_a[:, :, cut:], write[:, :, cut:], B_bias, C_bias,
                            S_init=S1, return_state=True)
    Y1_cp, S1_cp = flat_reference_cumprod(Bt[:, :cut], Ct[:, :cut], v[:, :cut],
                                          log_a[:, :, :cut], write[:, :, :cut],
                                          B_bias, C_bias, return_state=True)
    Y2_cp = flat_reference_cumprod(Bt[:, cut:], Ct[:, cut:], v[:, cut:],
                                   log_a[:, :, cut:], write[:, :, cut:], B_bias,
                                   C_bias, S_init=S1_cp)
    Y_cat = torch.cat([Y1_cp, Y2_cp], dim=1)
    terms = terms_abs(_flat_reference, Bt, Ct, v, log_a, write, B_bias, C_bias).terms
    # A7: u=fp64, c=_c_cumprod(log_a) (module note above); the resumed pieces
    # sum the same terms as the full run
    c = _c_cumprod(log_a)
    assert_close_terms(Y_cat, torch.cat([Y1, Y2], dim=1), terms, "fp64", c,
                       name="Y resumed vs resumed loop")
    assert_close_terms(Y_cat, Y_full, terms, "fp64", c, name="Y resumed vs full loop")


# ---------------------------------------------------------------------------
# 2. row-mixture SPLIT: re-sum independently-computed pieces against a real
#    mixed-law HybridMambaSSM (backend="sequential"), never touching the
#    module's own `y = y_hm + y_m3 + ...` line.
# ---------------------------------------------------------------------------

# A flat law's row total is that law's ssd_mimo_scan launch width, and the public entry serves every width
# exactly (pad below 16, power-of-two tiles above): no row total is refused, so `d_state` varies with how many
# rows a config asks for.  `_ODD_WIDTH_FRACS` pins the widths the old rule refused (8, 24, 256).
_MIX_FRACS = [
    [{"rows": 16, "decay": "coord_static"}, {"rows": 16, "decay": "coord_static", "complex": True},
     {"rows": 16, "decay": "head_dynamic", "complex": True}, {"rows": 16, "decay": "head_static"}],
    [{"rows": 32, "decay": "coord_dynamic"}, {"rows": 32, "decay": "head_static"}],
    [{"rows": 16, "decay": "coord_potential"}, {"rows": 32, "decay": "coord_static"},
     {"rows": 16, "decay": "head_static"}],
]


_ODD_WIDTH_FRACS = [
    ([{"rows": 8, "decay": "coord_static"}, {"rows": 8, "decay": "head_dynamic"}], 16),
    ([{"rows": 8, "decay": "coord_static"}, {"rows": 24, "decay": "head_static"}], 32),
    ([{"rows": 16, "decay": "coord_static"}, {"rows": 256, "decay": "head_dynamic"}], 272),
    ([{"rows": 8, "decay": "coord_static"}, {"rows": 24, "decay": "head_dynamic", "complex": True}], 32),
]


def _mix_core(fracs, backend="sequential", d_inner=32, d_state=64, n_heads=4,
             use_trap=True):
    torch.manual_seed(11)
    return HybridMambaSSM(HybridMambaConfig(d_model=d_inner, d_state=d_state, d_inner=d_inner, d_conv=0,
                          n_heads=n_heads, proj_source="main_branch",
                          row_blocks=fracs, write_rule="trapezoid" if use_trap else "euler",
                          # the flat rows serve only the per-head B/C biases
                          # and the fixed-frequency rotation;
                          # `rotation` is global to the core
                          learn_theta_freq=False, use_theta_h=False), execution={"backend": backend})


def _capture(monkeypatch):
    """Intercept flat-law and production MH scan calls made
    by the sequential backend, recording (args, kwargs) without changing
    their behaviour."""
    import orbit_ssm.kernels.hybrid_mamba_mh.torch_scan as ref_mod
    import orbit_ssm.ssm.hybrid_mamba as hm_mod

    calls = {"m3": [], "flat": [], "hm": []}

    orig_m3 = hm_mod._m3_reference
    def spy_m3(*a, **kw):
        calls["m3"].append((a, kw))
        return orig_m3(*a, **kw)
    monkeypatch.setattr(hm_mod, "_m3_reference", spy_m3)

    orig_flat = hm_mod._flat_reference
    def spy_flat(*a, **kw):
        calls["flat"].append((a, kw))
        return orig_flat(*a, **kw)
    monkeypatch.setattr(hm_mod, "_flat_reference", spy_flat)

    orig_hm = ref_mod.mh_torch_scan
    def spy_hm(*a, **kw):
        oracle_kw = {key: value for key, value in kw.items() if key != "sequential"}
        calls["hm"].append((a, oracle_kw))
        return orig_hm(*a, **kw)
    monkeypatch.setattr(ref_mod, "mh_torch_scan", spy_hm)
    return calls


@pytest.mark.parametrize("fracs,d_state", [(f, 64) for f in _MIX_FRACS] + _ODD_WIDTH_FRACS)
def test_row_mixture_split_matches_module_sequential(monkeypatch, fracs, d_state):
    core = _mix_core(fracs, d_state=d_state)
    torch.manual_seed(7)
    x = torch.randn(2, 20, core.d_inner, dtype=torch.float32) * 0.4
    # The module's OUTPUT is its production sequential route.  Since the
    # complete-State rewrite a mixed-row `forward` on
    # backend="sequential" is the per-token `step` loop, which calls none
    # of the per-block references -- capturing around `forward` then
    # recorded nothing and the re-sum was the D skip alone (red earlier:
    # 0.25-0.32 relative).  The PIECES are captured from the
    # module's own per-block scan dispatch (`_scan_cf` without a State),
    # which feeds each block the same token maps the step loop reads.
    y_actual = core.scan(x)
    assert core.last_route() == "all: stepped(per-token recurrence)", \
        core.last_route()
    calls = _capture(monkeypatch)
    core._scan_cf(x.transpose(1, 2).contiguous(), None, None, False)

    # Re-sum the pieces the module itself computed, independently: the
    # closed-form oracle for every flat call, the (unchanged, already-an-
    # oracle) mh_reference for the HM call, plus the D skip applied exactly
    # as the module's own epilogue.
    Y = torch.zeros(2, core.H, core.P, x.shape[1], dtype=DT)
    for a, kw in list(calls["hm"]):
        # mh_reference itself is the oracle; just take its own output. Uses
        # the pre-patch binding (see import above) -- calling the module
        # attribute here would recurse into the still-active spy and grow
        # `calls["hm"]` while this very loop iterates it.
        y_hm, _ = _mh_reference_unpatched(*a, **kw)
        Y = Y + y_hm
    for a, kw in calls["m3"]:
        kw2 = dict(kw)
        kw2.pop("S_init", None), kw2.pop("kp_init", None)
        kw2.pop("vp_init", None), kw2.pop("phase_init", None)
        kw2.pop("return_state", None)
        y3 = m3_reference_cumprod(*a, **kw2)
        Y = Y + y3.permute(0, 2, 3, 1)
    for a, kw in calls["flat"]:
        kw2 = dict(kw)
        kw2.pop("S_init", None), kw2.pop("return_state", None)
        yf = flat_reference_cumprod(*a, **kw2)
        Y = Y + yf.permute(0, 2, 3, 1)
    Y = Y + core.D.double().view(1, core.H, 1, 1) * x.double().transpose(1, 2).reshape(
        2, core.H, core.P, x.shape[1])
    y_mix = Y.reshape(2, core.d_inner, x.shape[1]).transpose(1, 2)

    # every block present must have been captured -- an empty capture
    # re-sums the D skip alone, which is how this cell went red unnoticed
    assert len(calls["hm"]) == (core.d_state_hm > 0)
    assert len(calls["m3"]) == (core.d_state_m3 > 0)
    assert len(calls["flat"]) == (core.d_state_m2 > 0)
    # ~1e-7 observed: both sides are fp64, but the module's own y_actual
    # carries a few fp32-parameter reads (RMSNorm weight/eps, D, biases)
    # promoted to fp64 at a different point in the arithmetic than this
    # re-summation does, not fp64 rounding alone.
    assert _relative(y_mix, y_actual) < 1e-6


@pytest.mark.parametrize("fracs,d_state", _ODD_WIDTH_FRACS)
def test_odd_flat_widths_torch_route_matches_the_sequential_oracle_route(fracs, d_state):
    """contract: row totals the old rule refused (8, 24, 256) are built, and the parallel torch route agrees with the
    per-token recurrence (fp32 module; both arms round their own sums: rel < 1e-4)."""
    seq, par = _mix_core(fracs, "sequential", d_state=d_state), _mix_core(fracs, "torch", d_state=d_state)
    par.load_state_dict(seq.state_dict())
    torch.manual_seed(5)
    x = torch.randn(2, 24, seq.d_inner) * 0.4
    assert _relative(par.scan(x).double(), seq.scan(x).double()) < 1e-4


@pytest.mark.gpu
@pytest.mark.parametrize("fracs,d_state", [
    ([{"rows": 8, "decay": "coord_static"}, {"rows": 8, "decay": "head_dynamic"}], 16),
    ([{"rows": 8, "decay": "coord_static"}, {"rows": 24, "decay": "head_dynamic"}], 32),
    ([{"rows": 16, "decay": "coord_static"}, {"rows": 256, "decay": "head_dynamic"}], 272),
    ([{"rows": 8, "decay": "coord_static"}, {"rows": 24, "decay": "head_dynamic", "complex": True}], 32)])
def test_odd_flat_widths_cuda_matches_the_torch_route(fracs, d_state):
    """contract: head_dynamic row totals of 8, 24 and 256 run on the kernel route (`ssd_mimo_scan` pads / tiles them)
    and agree with the torch route in value and in the gradient of x (fp32; relative bound 2e-3: the arms differ in
    TF32-free fp32 reduction order over L=24 tokens and up to 256 rows)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cuda, ref = (_mix_core(fracs, b, d_state=d_state).cuda() for b in ("cuda", "torch"))
    ref.load_state_dict(cuda.state_dict())
    torch.manual_seed(5)
    x = (torch.randn(2, 24, cuda.d_inner, device="cuda") * 0.4)
    outs = []
    for core in (cuda, ref):
        xi = x.clone().requires_grad_()
        y = core.scan(xi)
        y.square().sum().backward()
        outs.append((y.detach().double(), xi.grad.double(), core.last_route()))
    assert outs[0][2] is None, f"a row block left the kernels: {outs[0][2]}"
    assert _relative(outs[0][0], outs[1][0]) < 2e-3
    assert _relative(outs[0][1], outs[1][1]) < 2e-3


# ---------------------------------------------------------------------------
# 3. step() vs one forward() scan call -- per law and mixed, every backend
#    step() itself is backend-independent (always torch-native math), so
#    this also exercises step() against `backend="cuda"` when available.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["sequential", "cuda"])
@pytest.mark.parametrize("fracs", _MIX_FRACS)
def test_step_matches_one_scan_call(request, backend, fracs):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = "cuda" if backend == "cuda" else "cpu"
    core = _mix_core(fracs, backend=backend).to(dev)
    torch.manual_seed(23)
    Bsz, L = 2, 12
    x = (torch.randn(Bsz, L, core.d_inner, device=dev) * 0.3)

    y_scan, _ = core._forward_cf(x.transpose(1, 2), want_state=False)
    y_scan = y_scan.transpose(1, 2)

    state = core.init_state(Bsz, device=dev, dtype=x.dtype)
    ys = []
    for t in range(L):
        y_t, state = core.scan_step(x[:, t], state)
        ys.append(y_t)
    y_step = torch.stack(ys, dim=1)

    assert torch.isfinite(y_step).all()
    assert _relative(y_step, y_scan) < 5e-3, \
        f"step() vs one forward() scan call diverges (backend={backend})"
