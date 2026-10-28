"""SSD-MIMO endpoint cells against THE canonical oracle (``physical_oracle``).  The oracle, ``OracleTerms``, ``law_c``
and the case builders live in tests/_contract/ssd_physical.py (the contract suite's SSD cells import them from
there); they are re-exported here for the legacy files that still import them from this module."""
import copy
import math
import sys

import pytest
import torch
import torch.nn.functional as F

from _tolerance import assert_close_terms, terms_abs
from conftest import assert_kernel_served
from orbit_ssm.layers.state import ScanState
from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig
from tools.audit_ssm_semantics import scalar_case
from _contract.ssd_physical import *  # noqa: F401,F403  (re-export for the legacy importers)
from _contract.ssd_physical import (OracleTerms, ROT_ROWS, cores_law_c, law_c, law_stats, make_case, mixer_law_c,
                                    module_terms, pair_perm, physical_oracle, rotation, ssd_depth, ssd_rows)

from _design_warnings import SILENCE_SSD_DESIGN_WARNINGS

pytestmark = SILENCE_SSD_DESIGN_WARNINGS  # builds the warned configurations on purpose


def _rot_rows(rank):
    """The rotating row block of the module cells.  A map gate exists at rank > 1 only (the config refuses `gate` where no B/C
    gate resolves), so the rank-1 block carries no gate key."""
    return ROT_ROWS if rank > 1 else [{k: v for k, v in ROT_ROWS[0].items() if k != "gate"}]


def test_make_case_draws_the_bare_core_weights():
    """PINS: `make_case` draws the scan-owned weights and the inputs of the
    BARE-core case (the one built before the full layer), not the
    layer's shifted stream.  The layer's in_proj / out_proj Linear inits
    would otherwise consume the global RNG ahead of the scan parameters.
    The sums are float64 values from the bare-core `make_case`
    (same seed 31, CPU); an exact comparison would also bind torch's RNG
    implementation: each fp32 draw may differ by one rounding between CPUs, so a sum of n draws is
    compared to within n * 2**-24 * max|draw| (a wrong stream differs by O(1)), plus 1e-9."""
    want = {None: (2.444906741381, -0.354760736227, -16.164858818054, 7.020938885003, 9.792011900142),
            "m3": (2.444906741381, 0.16402515769, -15.108450651169, -9.861331644029, -3.978901563186)}
    for mode, ref in want.items():
        m, x, u, h, z = make_case(mode)
        P = dict(m.named_parameters())
        drawn = (P["B_proj.weight"], P["dt_proj.weight"], P["dt_proj.bias"], x, z)
        for t, r in zip(drawn, ref):
            bound = 1e-9 + t.numel() * 2.0 ** -24 * float(t.abs().max().detach())
            total = float(t.sum().detach())
            assert abs(total - r) < bound, (mode, total, r)


@pytest.mark.parametrize("backend", ["torch", "sequential"])
@pytest.mark.parametrize("mode", [None, "m3"])
@pytest.mark.parametrize("rank,dynamic,conv", [(1, False, False), (2, True, True)])
def test_endpoint_physical_outputs_and_gradients(backend, mode, rank, dynamic, conv):
    m, x, u, h, z = make_case(mode, rank, conv=conv, dynamic=dynamic)
    m.set_execution(dict(backend=backend))
    actual = m.scan(x, state=ScanState(m).init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), sel_x=u, gate_z=z)
    weight = torch.randn_like(actual)
    # A7: u=fp64 (both arms fp64, the route reassociates the law); c=law_c:
    # sqrt(2 * depth) + decay + phase amplification (OracleTerms docstring)
    ref = OracleTerms(m, dict(x=x, selection=u, h_init=h, gate=z),
                      grad_outputs=(weight, None))
    assert_close_terms(actual, ref.out[0], ref.terms[0], "fp64", ref.c_y, name="y")
    params = (x, u, h, z, *m.parameters())
    got = torch.autograd.grad((actual * weight).sum(), params, allow_unused=True)
    for name, a, p in zip(("x", "selection", "h_init", "gate", *dict(m.named_parameters())), got, params):
        ref.check_grad(name, a, p)


@pytest.mark.parametrize("mode", [None, "m3"])
@pytest.mark.parametrize("grad", [False, True])
def test_endpoint_resume_and_reset(mode, grad):
    m, x, u, h, z = make_case(mode, conv=True)
    with torch.set_grad_enabled(grad):
        full = m.scan(x, state=ScanState(m).init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), sel_x=u, gate_z=z)
        left, state = m.scan(x[:, :3], state=ScanState(m).init_state((x[:, :3]).shape[0], h_init=h, device=(x[:, :3]).device, dtype=next(m.parameters()).dtype), sel_x=u[:, :3], gate_z=z[:, :3], return_state=True)
        right, state = m.scan(x[:, 3:], state=state, sel_x=u[:, 3:], gate_z=z[:, 3:], return_state=True)
        joined = torch.cat((left, right), 1)
        # A7: u=fp64 (split and whole torch route reassociate the same law);
        # c=law_c (sqrt(2 * depth) + decay + phase); sum|terms| from the
        # fp64 oracle (the same values, OracleTerms)
        ref = OracleTerms(m, dict(x=x, selection=u, h_init=h, gate=z),
                          grad_outputs=(torch.ones_like(full), None))
        assert_close_terms(joined, full, ref.terms[0], "fp64", ref.c_y, name="joined y")
        assert_close_terms(state["hidden_state"], ref.out[1], ref.terms[1], "fp64", ref.c_y,
                           name="hidden_state")
        if grad:
            params = (x, u, h, z, *m.parameters())
            names = ("x", "selection", "h_init", "gate", *dict(m.named_parameters()))
            a = torch.autograd.grad(joined.sum(), params, allow_unused=True)
            b = torch.autograd.grad(full.sum(), params, allow_unused=True)
            for name, av, bv in zip(names, a, b):
                if av is not None and bv is not None:
                    assert_close_terms(av, bv, ref.grad_terms[name], "fp64", ref.c_g,
                                       name=f"grad {name}")
        ScanState(m).reset_state(state, mask=torch.tensor([True]))
        key = "write_prev"
        assert state[key].count_nonzero() == 0


@pytest.mark.parametrize("mode", [None, "m3"])
def test_endpoint_inference_cache_owns_compact_storage(mode):
    m, x, u, h, z = make_case(mode, length=65, conv=True)
    with torch.no_grad():
        _, state = m.scan(x, state=ScanState(m).init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), sel_x=u, gate_z=z, return_state=True)
    for name, value in state.items():
        if torch.is_tensor(value):
            assert not value.requires_grad
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size(), name


def test_legacy_clock_cache_is_rejected():
    m, x, u, h, _ = make_case(None)
    legacy = {"hidden_state": h, "dt_prev": torch.ones(1, m.H, m.P)}
    with pytest.raises(ValueError, match="unknown keys.*dt_prev"):
        m.scan(x, sel_x=u, state=legacy)
    with pytest.raises(ValueError, match="unknown keys.*dt_prev"):
        m.scan_step(x[:, 0], legacy, sel_x=u[:, 0])


@pytest.mark.parametrize("checkpoint", [False, True])
def test_modular_mixer_endpoint_resume_and_checkpoint(checkpoint):
    torch.manual_seed(3)
    m = SSDMimoSSM(SSDMimoConfig(d_model=8, d_state=16, expand=1, d_conv=3,
                   d_ssm_in=2, d_ssm_out=2, n_heads=2, n_groups=2,
                   write_rule="trapezoid", row_blocks=ROT_ROWS), execution={"backend": "torch",
                              "kernel_options": {"mode": "modular",
                                                 "modular_checkpoint": checkpoint}}).double()
    x = torch.randn(1, 7, 8, dtype=torch.float64, requires_grad=True)
    full = m(x)
    left, state = m(x[:, :3], return_state=True)
    right, _ = m(x[:, 3:], state=state, return_state=True)
    joined = torch.cat((left, right), 1)
    # A7: u=fp64 (split vs whole, the same fp64 route reassociated);
    # c=mixer_law_c (the core's law_c, depth + the layer's own sums);
    # sum|terms| of the whole-sequence reference arm, absolute arithmetic
    c_y, c_g = mixer_law_c(m, x)
    ref = module_terms(m, lambda x: m(x), x, grad_outputs=torch.ones_like(full))
    assert_close_terms(joined, full, ref.terms, "fp64", c_y, name="y")
    params = (x, *m.parameters())
    names = ("x", *dict(m.named_parameters()))
    terms = dict(zip(names, (ref.grad_terms[0], *ref.param_grad_terms.values())))
    got = torch.autograd.grad(joined.sum(), params, allow_unused=True)
    want = torch.autograd.grad(full.sum(), params, allow_unused=True)
    for name, a, b in zip(names, got, want):
        assert (a is None) == (b is None)
        if a is not None:
            assert_close_terms(a, b, terms[name], "fp64", c_g, name=f"grad {name}")


@pytest.mark.parametrize("backend", ["torch", "sequential"])
def test_original_scalar_counterexample_now_passes(backend):
    assert scalar_case(trap=True, backend=backend)["passed"]


@pytest.mark.parametrize("mode", [None, "m3"])
@pytest.mark.parametrize("r,o", [(1, 2), (2, 1), (4, 8), (8, 4)])
def test_endpoint_mixed_interfaces(mode, r, o):
    m, x, u, h, z = make_case(mode, rank=r, read_rank=o, length=3)
    ref = OracleTerms(m, dict(x=x, selection=u, h_init=h, gate=z))
    actual = m.scan(x, state=ScanState(m).init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), sel_x=u, gate_z=z)
    # A7: u=fp64 (fp64 route vs fp64 oracle); c=law_c (sqrt(2 * depth) + decay + phase)
    assert_close_terms(actual, ref.out[0], ref.terms[0], "fp64", ref.c_y, name="y")


@pytest.mark.parametrize("mode", [None, "m3"])
def test_endpoint_lambda_one_recovers_the_euler_write(mode):
    """lambda -> 1 makes the "trapezoid" rule write only the current
    endpoint, which IS the "euler" rule (both truncate phi ~ I and differ only in
    the quadrature weights).  The name is `euler`, NOT `zoh` -- `zoh` is
    reserved for the exact hold, which neither rule computes, and a
    sweep briefly renamed this cell the wrong way."""
    m, x, u, h, z = make_case(mode)
    with torch.no_grad():
        m.trap_proj.weight.zero_()
        m.trap_proj.bias.fill_(1000.)
    baseline = copy.deepcopy(m)  # copy before forward creates differentiable cached tensors
    actual = m.scan(x, state=ScanState(m).init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), sel_x=u, gate_z=z)
    baseline.write_rule = "euler"   # the selector is the ONE source of truth
    # A7: u=fp64 (fp64 trapezoid route at lambda = sigmoid(1000) = 1 exactly
    # vs the fp64 Euler oracle); c=law_c (sqrt(2 * depth) + decay + phase)
    ref = OracleTerms(baseline, dict(x=x, selection=u, h_init=h, gate=z))
    assert_close_terms(actual, ref.out[0], ref.terms[0], "fp64", ref.c_y, name="y")


@pytest.mark.parametrize("mode", [None, "m3"])
@pytest.mark.parametrize("length", [1, 65])
def test_endpoint_boundaries_and_causality(mode, length):
    m, x, u, h, z = make_case(mode, length=length)
    actual = m.scan(x, state=ScanState(m).init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), sel_x=u, gate_z=z)
    first = torch.zeros_like(actual)
    first[:, 0] = 1                     # the adjoint of sum(y[:, 0])
    ref = OracleTerms(m, dict(x=x, selection=u, h_init=h, gate=z),
                      grad_outputs=(first, None))
    # A7: u=fp64 (fp64 route vs fp64 oracle); c=law_c (sqrt(2 * depth) +
    # decay + phase).  Causality: the oracle's y_0 has no path to a later
    # token, so its gradient there is 0 with sum|terms| = 0 and the terms
    # form demands an exact 0 from the route (its causal masks select / zero).
    assert_close_terms(actual, ref.out[0], ref.terms[0], "fp64", ref.c_y, name="y")
    grads = torch.autograd.grad(actual[:, 0].sum(), (x, u, z))
    for name, future in zip(("x", "selection", "gate"), grads):
        assert_close_terms(future[:, 1:], ref.grads[name][:, 1:],
                           ref.grad_terms[name][:, 1:], "fp64", ref.c_g,
                           name=f"future grad {name}")


@pytest.mark.parametrize("mode", [None, "m3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_endpoint_cuda_against_physical_oracle(require_cuda, mode, dtype, record_property):
    m, x, u, h, z = make_case(mode, length=65, device="cuda", dtype=dtype)
    m.set_execution(dict(backend="cuda"))
    ref = copy.deepcopy(m).double()
    inputs = [v.detach().double().requires_grad_() for v in (x, u, h, z)]
    actual = m.scan(x, sel_x=u, state=ScanState(m).init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), gate_z=z)
    assert_kernel_served(m, "the kernel-vs-fp64-oracle comparison")
    w = torch.randn_like(actual)
    names = ("input", "selection", "initial", "gate", *dict(m.named_parameters()))
    # the fp64 oracle on the SAME rounded inputs / parameters, in absolute
    # arithmetic: values identical to the plain oracle, plus sum|terms|.
    # A7 (fp32 cell): u=fp32 (precision="ieee": fp32 FMA; the oracle is
    # fp64, one arm rounds); c=law_c(arms=1): sqrt(depth) + decay + phase.
    oracle = OracleTerms(ref, dict(zip(names[:4], inputs)),
                         grad_outputs=(w.double(), None), u="fp32", arms=1)
    expected = oracle.out[0]
    want = tuple(oracle.grads.get(n) for n in names)
    if dtype == torch.bfloat16:
        # 16-bit class-(b) cell: left on its pre-A7 elementwise bound; the
        # >= 5-draw tol16 policy is a restructure of this GPU cell (reported)
        tol = .07
        torch.testing.assert_close(actual.double(), expected, rtol=tol, atol=tol)
    else:
        assert_close_terms(actual, expected, oracle.terms[0], "fp32",
                           oracle.c_y, name="y")
    got = torch.autograd.grad((actual * w).sum(), (x, u, h, z, *m.parameters()), allow_unused=True)
    for name, a, b in zip(names, got, want):
        if a is not None and b is not None:
            error = (a.double() - b).norm() / b.norm().clamp_min(1e-12)
            record_property(name + "_relative_gradient_error", float(error))
    if dtype == torch.bfloat16:
        # Separate quantization from a scan-adjoint defect. In the initial
        # diagnostic the unchanged M3 B-projection gradient was bit-identical
        # to eager bf16, while BOTH missed a cancellation-sensitive fp64
        # component. Require per-parameter error bounds and matched-precision
        # agreement, rather than the former loose 21% elementwise allowance.
        eager = copy.deepcopy(ref).to(dtype=dtype)
        eager.set_execution(dict(backend="torch"))
        ei = [v.detach().to(dtype).requires_grad_() for v in inputs]
        ey = eager.scan(ei[0], sel_x=ei[1], state=ScanState(eager).init_state((ei[0]).shape[0], h_init=ei[2], device=(ei[0]).device, dtype=next(eager.parameters()).dtype), gate_z=ei[3])
        eg = torch.autograd.grad((ey * w).sum(), (*ei, *eager.parameters()), allow_unused=True)
        relative_output = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-12)
        assert relative_output < .02
        for name, a, b, e in zip(names, got, want, eg):
            assert (a is None) == (b is None) == (e is None), name
            if a is not None:
                ideal_error = (a.double() - b).norm() / b.norm().clamp_min(1e-12)
                scan_error = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-12)
                record_property(name + "_matched_bf16_error", float(scan_error))
                # .06: the m3 read map is folded per kernel head on the host
                # (theta_h), which costs one extra bf16 rounding of C; the
                # matched-precision check below is the scan-adjoint bound.
                assert ideal_error < .06, (name, float(ideal_error))
                # The matched-precision arm is itself a bf16 computation and is NOT exact: for the trapezoid parameters its lambda is
                # sigmoid(bf16), rounded to bf16 before the adjoint's lambda(1 - lambda) is built from it, while the kernel route rounds
                # nothing there (fp32 sigmoid) -- the two arms differ in exactly the quantity this cell is about.  The
                # triangle inequality bounds their distance by what each arm is off the fp64 oracle:
                #     ||kernel - eager|| <= ||kernel - fp64|| + ||eager - fp64||,
                # so for `trap_proj.*` the bound is the larger of .02 and that sum (the kernel stays under .06 above); every other
                # parameter keeps the plain .02.  (A100: trap_proj.bias kernel-vs-eager 0.0234, bisected to a specific change.)
                eager_error = (e.double() - b).norm() / b.norm().clamp_min(1e-12)
                bound = max(.02, float(ideal_error + eager_error)) if name.startswith("trap_proj") else .02
                record_property(name + "_eager_bf16_error", float(eager_error))
                assert scan_error < bound, (name, float(scan_error), float(ideal_error), float(eager_error))
    else:
        for name, a, b, p in zip(names, got, want, (x, u, h, z, *m.parameters())):
            assert (a is None) == (b is None), name
            if a is not None:
                oracle.check_grad(name, a, p)


@pytest.mark.parametrize("mode", [None, "m3"])
@pytest.mark.parametrize("precision", ["tf32", "io"])
def test_endpoint_training_autocast(require_cuda, mode, precision, record_property):
    """Training convention: FP32 parameters/inputs, BF16 projected operands."""
    m, x, u, h, z = make_case(mode, length=65, device="cuda", dtype=torch.float32)
    m.set_execution(dict(backend="cuda", precision=precision))
    ref = copy.deepcopy(m).double()
    # the matched-precision torch arm, copied BEFORE the kernel forward: a
    # grad-enabled forward leaves non-leaf cached tensors on the module,
    # which `copy.deepcopy` refuses; the parameters are the same either way
    eager = copy.deepcopy(m); eager.set_execution(dict(backend="torch"))
    inputs = [v.detach().double().requires_grad_() for v in (x, u, h, z)]
    # The surrounding mixer's projection/conv emits BF16 values. Without
    # this cast a bare-core test leaves x FP32 and precision="io" is a
    # deliberate no-op, so it would never exercise BF16 scan arithmetic.
    inputs[0] = x.detach().bfloat16().double().requires_grad_()
    expected, _ = physical_oracle(ref, *inputs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = m.scan(x.to(torch.bfloat16), sel_x=u, state=ScanState(m).init_state((x.to(torch.bfloat16)).shape[0], h_init=h, device=(x.to(torch.bfloat16)).device, dtype=next(m.parameters()).dtype), gate_z=z)
    assert_kernel_served(m, f"the precision={precision!r} scan arithmetic claim")
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert (actual.double()-expected).norm()/expected.norm() < .025
    weight = torch.randn_like(actual)
    got = torch.autograd.grad(actual, (x,u,h,z,*m.parameters()), weight, allow_unused=True)
    want = torch.autograd.grad(expected, (*inputs,*ref.parameters()), weight.double(), allow_unused=True)
    # Matched-precision arm: the SAME bf16-autocast forward through the torch
    # backend.  At this toy width (d_model=3) the per-parameter distance to the
    # fp64 ideal is dominated by bf16 rounding of the projections and varies
    # 0.01-0.25 with the random draw (probe), so the ideal bound is loose and the scan-adjoint claim is the
    # matched-precision one.
    ei = [v.detach().clone().requires_grad_() for v in (x, u, h, z)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ey = eager.scan(ei[0].to(torch.bfloat16), sel_x=ei[1], state=ScanState(eager).init_state((ei[0].to(torch.bfloat16)).shape[0], h_init=ei[2], device=(ei[0].to(torch.bfloat16)).device, dtype=next(eager.parameters()).dtype), gate_z=ei[3])
    eg = torch.autograd.grad(ey, (*ei, *eager.parameters()), weight, allow_unused=True)
    names = ("input", "selection", "initial", "gate", *dict(m.named_parameters()))
    for name, a, b, e in zip(names, got, want, eg):
        assert (a is None) == (b is None) == (e is None), name
        if a is not None:
            error = (a.double()-b).norm()/b.norm().clamp_min(1e-12)
            scan_error = (a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-12)
            record_property(name+"_relative_error",float(error))
            record_property(name+"_matched_bf16_error",float(scan_error))
            assert error < .3, (name,float(error))
            # tf32 arm: the CUDA scan uses TF32 MMA operands, the torch arm fp32
            assert scan_error < (.05 if precision == "tf32" else .03), (name,float(scan_error))


@pytest.mark.gpu
@pytest.mark.parametrize("increment", [False, True], ids=["phase", "increments"])
@pytest.mark.parametrize("rank", [1, 2], ids=["R1O1", "R2O2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16],
                         ids=["fp32", "bf16"])
def test_read_offset_in_kernel_under_pre_rotation_bias(require_cuda, rank, dtype,
                                                       increment):
    """Kernel (per-head phase, theta_h, beta, gamma, bias_pre_rot, decoupled
    write) against the fp64 quadratic oracle with the same law: y, final
    state and every gradient."""
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    from orbit_ssm.kernels.ssd_mimo.reference import unified_ssd_quadratic
    torch.manual_seed(31 + rank)
    B, H, G, N, L = 2, 3, 32, 64, 192
    R = O = rank
    f64 = dict(device="cuda", dtype=torch.float64)
    x = torch.randn(B, H, G, R, L, **f64) * 0.5
    delta = torch.rand(B, H, L, **f64) * 0.1 + 0.01
    Bt = torch.randn(B, L, N, R, **f64) * 0.3
    Ct = torch.randn(B, L, N, O, **f64) * 0.3
    a_log = torch.randn(H, **f64) * 0.3
    theta = torch.randn(B, L, H, N // 2, **f64) * 0.2
    theta_h = torch.randn(H, N // 2, **f64) * 0.5
    beta = torch.randn(H, N, R, **f64) * 0.1
    gamma = torch.randn(H, N, O, **f64) * 0.1
    dw = torch.rand(B, H, L, **f64) * 0.05 + 0.01
    leaves = [x, delta, Bt, Ct, a_log, theta, theta_h, beta, gamma, dw]

    def run(fn, cast):
        vals = [t.detach().to(cast if i in (0, 2, 3) else torch.float64
                              if fn is unified_ssd_quadratic else torch.float32)
                .clone().requires_grad_(True) for i, t in enumerate(leaves)]
        xx, dd, bb, cc, aa, th, thh, be, ga, w = vals
        kw = dict(theta=th, theta_h=thh, beta=be, gamma=ga, delta_w=w,
                  bias_pre_rot=True)
        if fn is ssd_mimo_scan:
            if increment:
                # the module's route: per-token increments, cumulated in-kernel
                inc = torch.cat([th[:, :1], th[:, 1:] - th[:, :-1]], dim=1)
                kw["theta"] = inc
                kw["theta_increment"] = True
            y, s = fn(xx, dd, bb, cc, aa, chunk_size=64,
                      precision="ieee" if dtype is torch.float32 else "io",
                      **kw)
        else:
            y, s = fn(xx, dd, bb, cc, aa, **kw)
        return y, s, vals

    y_k, s_k, v_k = run(ssd_mimo_scan, dtype)
    y_r, s_r, v_r = run(unified_ssd_quadratic, torch.float64)
    torch.manual_seed(7)
    dy, ds = torch.randn_like(y_r), torch.randn_like(s_r)
    ((y_k.double() * dy).sum() + (s_k.double() * ds).sum()).backward()
    ((y_r * dy).sum() + (s_r * ds).sum()).backward()
    tol = 5e-5 if dtype is torch.float32 else 6e-2
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12)).item()
    assert rel(y_k, y_r) < tol, f"y {rel(y_k, y_r):.2e}"
    assert rel(s_k, s_r) < tol, f"state {rel(s_k, s_r):.2e}"
    names = ["x", "delta", "B", "C", "a_log", "theta", "theta_h", "beta", "gamma", "delta_w"]
    for n, a, b in zip(names, v_k, v_r):
        assert a.grad is not None and b.grad is not None, n
        assert rel(a.grad, b.grad) < tol, f"d{n} {rel(a.grad, b.grad):.2e}"


@pytest.mark.gpu
@pytest.mark.parametrize("rank", [1, 2], ids=["R1O1", "R2O2"])
def test_module_read_offset_route_matches_sequential(require_cuda, rank):
    """SSDMimoSSM at its defaults (m3 rotation, theta_freq, theta_h, trapezoid)
    on the CUDA route against the fp32 sequential backend, same weights and
    inputs: forward and every parameter gradient.  Both ranks keep
    theta_h in-kernel; R1O1 takes the in-scan trapezoid and R2O2 the
    endpoint pass (the adjoint is generalised to R, O >= 1 and gated,
    but the endpoint pass measured FASTER there, so it keeps the
    default -- see `test_module_trapezoid_stays_in_the_scan_kernel_in_training`
    for the in-scan route at both ranks)."""
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    torch.manual_seed(5)
    kw = dict(d_ssm_in=rank, d_ssm_out=rank, n_groups=16 // rank, n_heads=8,
              proj_source="main_branch", row_blocks=_rot_rows(rank), bc_bias="head")
    m = SSDMimoSSM(SSDMimoConfig(d_model=128, d_state=32, d_inner=128,
                   **kw), execution={"backend": "cuda", "precision": "ieee"}).cuda()
    ref = SSDMimoSSM(SSDMimoConfig(d_model=128, d_state=32, d_inner=128,
                     **kw), execution={"backend": "sequential"}).cuda()
    ref.load_state_dict(m.state_dict())
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ref.load_state_dict(m.state_dict())
    torch.manual_seed(2)
    x = torch.randn(2, 40, 128, device="cuda")
    outs = []
    for mod in (m, ref):
        xi = x.clone().requires_grad_(True)
        y, _ = mod.scan(xi)
        leaves = [xi] + [p for _, p in sorted(mod.named_parameters())]
        g = torch.autograd.grad(y.float().square().sum(), leaves, allow_unused=True)
        outs.append((y.detach().float(), [n for n, _ in sorted(mod.named_parameters())], g))
    # `ref` is the sequential DUAL and has no route claim; `m` is the cell's
    # subject and its forward is the one this asserts (ref's forward does
    # not touch m's note).
    assert_kernel_served(m, "the CUDA-route-vs-sequential comparison")
    (y_c, names, g_c), (y_s, _, g_s) = outs
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12)).item()
    assert rel(y_c, y_s) < 2e-4, f"forward {rel(y_c, y_s):.2e}"
    for n, a, b in zip(["x"] + names, g_c, g_s):
        if a is None or b is None or b.abs().max() == 0:
            continue
        assert rel(a, b) < 5e-3, f"d{n} {rel(a, b):.2e}"


@pytest.mark.gpu
@pytest.mark.parametrize("rank", [1, 2], ids=["R1O1", "R2O2"])
def test_module_trapezoid_stays_in_the_scan_kernel_in_training(require_cuda,
                                                               rank,
                                                               monkeypatch):
    """The ROUTE, with grad enabled: `trap_lam_k` set, no host correction.

    `m3_corr is None` and `endpoint is None` together say the endpoint
    pass (`_local` / `_endpoint_mix_fwd` / `_endpoint_mix_bwd`) never
    launches -- the two-tap write and its rank-1 output correction are
    both inside the scan kernel, and its adjoint inside the backward.
    """
    monkeypatch.setenv("SSD_MIMO_TRAP_INKERNEL_RO", "1")
    torch.manual_seed(5)
    m = SSDMimoSSM(SSDMimoConfig(d_model=128, d_state=32, d_inner=128,
                   
                   d_ssm_in=rank, d_ssm_out=rank, n_groups=16 // rank, n_heads=8,
                   proj_source="main_branch", row_blocks=_rot_rows(rank),
                   bc_bias="head"), execution={"backend": "cuda", "precision": "ieee"}).cuda()
    assert m.use_trap, "this cell needs the two-tap trapezoid on"
    seen = {}
    original = m._prepare_kernel_inputs

    def spy(*args, **kwargs):
        kw = original(*args, **kwargs)
        seen.update(kw)
        seen["in_kernel_trap"] = kwargs.get("in_kernel_trap")
        return kw

    m._prepare_kernel_inputs = spy
    x = torch.randn(2, 40, 128, device="cuda", requires_grad=True)
    with torch.enable_grad():
        y = m.scan(x)
        y.float().square().sum().backward()
    assert seen, "the CUDA route never built kernel inputs"
    # FIRST: `_prepare_kernel_inputs` is SHARED with the chunked torch route,
    # so the spy fires on both arms.  Only `last_route()` separates "the
    # kernel ran" from "the kernel refused and torch ran on the same inputs",
    # and a refused ROUTE is the root cause of the `in_kernel_trap` check
    # below failing -- reporting it first names the real reason.
    assert_kernel_served(m, "the in-scan trapezoid ROUTE claim")
    assert seen["in_kernel_trap"] is True, (
        "the router refused the in-scan trapezoid with grad enabled")
    assert seen["trap_lam_k"] is not None, (
        "the scan kernel did not get lambda: the two-tap write is on the host")
    assert seen["m3_corr"] is None, "a host trapezoid correction was built"
    assert seen["endpoint"] is None, "the fused endpoint pass still runs"
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.gpu
@pytest.mark.parametrize("rank", [1, 2], ids=["R1O1", "R2O2"])
def test_module_in_kernel_trapezoid_matches_sequential(require_cuda, rank,
                                                       monkeypatch):
    """The in-scan trapezoid's NUMBERS at the production grouped shape.

    Same module and comparison as
    `test_module_read_offset_route_matches_sequential`, but with the
    R, O > 1 in-scan route forced on (`SSD_MIMO_TRAP_INKERNEL_RO=1`).
    This is the cell that scores the adjoint through the GROUPED
    write map (`bc_bias="head"`, Gb = R) and the head-shared read
    map, which the kernel-level K13 cells (head-shared B and C) do not
    reach -- forward and every parameter gradient against the fp32
    sequential backend.
    """
    monkeypatch.setenv("SSD_MIMO_TRAP_INKERNEL_RO", "1")
    torch.manual_seed(5)
    kw = dict(d_ssm_in=rank, d_ssm_out=rank, n_groups=16 // rank, n_heads=8,
              proj_source="main_branch", row_blocks=_rot_rows(rank), bc_bias="head")
    m = SSDMimoSSM(SSDMimoConfig(d_model=128, d_state=32, d_inner=128,
                   **kw), execution={"backend": "cuda", "precision": "ieee"}).cuda()
    ref = SSDMimoSSM(SSDMimoConfig(d_model=128, d_state=32, d_inner=128,
                     **kw), execution={"backend": "sequential"}).cuda()
    with torch.no_grad():
        for p_ in m.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
    ref.load_state_dict(m.state_dict())
    torch.manual_seed(2)
    x = torch.randn(2, 40, 128, device="cuda")
    outs = []
    for mod in (m, ref):
        xi = x.clone().requires_grad_(True)
        y, _ = mod.scan(xi)
        leaves = [xi] + [p_ for _, p_ in sorted(mod.named_parameters())]
        g = torch.autograd.grad(y.float().square().sum(), leaves,
                                allow_unused=True)
        outs.append((y.detach().float(),
                     [n for n, _ in sorted(mod.named_parameters())], g))
    # only `m` claims the kernel; `ref` is the sequential dual.
    assert_kernel_served(m, "the in-scan trapezoid's NUMBERS on the CUDA route")
    (y_c, names, g_c), (y_s, _, g_s) = outs
    rel = lambda a, b: ((a.double() - b.double()).norm()
                        / b.double().norm().clamp_min(1e-12)).item()
    assert rel(y_c, y_s) < 2e-4, f"forward {rel(y_c, y_s):.2e}"
    for n, a, b in zip(["x"] + names, g_c, g_s):
        if a is None or b is None or b.abs().max() == 0:
            continue
        assert rel(a, b) < 5e-3, f"d{n} {rel(a, b):.2e}"


@pytest.mark.gpu
@pytest.mark.parametrize("rank", [1, 2], ids=["R1O1", "R2O2"])
def test_pairhalf_module_cuda_matches_the_physical_oracle(require_cuda,
                                                          rank):
    """m3, theta_h, trapezoid -- the module envelope.

    R = O = 1 is stage 1's cell; R = O = 2 is the S-row production shape,
    and it exercises Kb1 / Kb2 / the
    GEN_VEC trapezoid adjoint's half branches end to end.

    fp32 only: the bf16 half route is scored by the bf16 kernel cells
    of the contract suite, against the same fp64 quadratic oracle."""
    dtype = torch.float32
    torch.manual_seed(17 + rank)
    # H * G * R must equal d_inner (= 2 * 128): 8 heads x 16 channels at
    # R = 1, 4 heads at R = 2.
    kw = dict(d_ssm_in=rank, d_ssm_out=rank, n_groups=16, n_heads=8 // rank,
              proj_source="main_branch", row_blocks=_rot_rows(rank), bc_bias="head",
              d_state=32)
    m = SSDMimoSSM(SSDMimoConfig(d_model=128, d_inner=128,
                   
                   pair_layout="half", **kw), execution={"backend": "cuda", "precision": "ieee"}).cuda()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if any(k in name for k in ("theta_h", "B_bias", "C_bias",
                                       "theta_proj", "theta_freq")):
                p.uniform_(-0.4, 0.4)
    x = torch.randn(1, 24, 128, device="cuda")
    # copy BEFORE the forward: a grad-enabled forward leaves non-leaf cached
    # tensors on the module, which `copy.deepcopy` refuses (torch >= 2.11)
    ref = copy.deepcopy(m).double()
    y = m.scan(x.to(dtype) if dtype is not torch.float32 else x)
    assert_kernel_served(m, "the CUDA route in the half layout")
    y_ref, _ = physical_oracle(
        ref, x.double(), x.double(),
        torch.zeros(1, ref.H, ref.P, ref.N, device="cuda",
                    dtype=torch.float64))
    rel = ((y.double().reshape(y_ref.shape) - y_ref).norm()
           / y_ref.norm().clamp_min(1e-12)).item()
    tol = 2e-4 if dtype is torch.float32 else 5e-2
    assert rel < tol, f"cuda(half) vs physical oracle(half): {rel:.2e}"


@pytest.mark.gpu
@pytest.mark.parametrize("rank", [1, 2], ids=["R1O1", "R2O2"])
def test_pairhalf_module_is_the_interleaved_module_permuted(require_cuda,
                                                            rank):
    """Explicit state-coordinate conversion preserves the model.

    An interleaved checkpoint loaded into a half module through
    `convert_pair_layout` must give the same y and the same gradients --
    the layouts are two coordinate systems for one model, so this is a
    statement about the permutation being right, not a tolerance.
    """
    torch.manual_seed(23 + rank)
    kw = dict(d_ssm_in=rank, d_ssm_out=rank, n_groups=16, n_heads=8 // rank,
              proj_source="main_branch", row_blocks=_rot_rows(rank), bc_bias="head",
              d_state=32, d_inner=128)
    mi = SSDMimoSSM(SSDMimoConfig(d_model=128, pair_layout="interleaved", **kw), execution={"backend": "cuda", "precision": "ieee"}).cuda()
    with torch.no_grad():
        for name, p in mi.named_parameters():
            p.add_(0.05 * torch.randn_like(p))
            if any(k in name for k in ("theta_h", "B_bias", "C_bias")):
                p.uniform_(-0.4, 0.4)
    mh = SSDMimoSSM(SSDMimoConfig(d_model=128, pair_layout="half", **kw), execution={"backend": "cuda", "precision": "ieee"}).cuda()
    mh.load_state_dict(mh.convert_pair_layout(mi.state_dict()))
    assert int(mh.pair_layout_code) == 1
    # a round trip back into an interleaved module restores the weights
    mi2 = SSDMimoSSM(SSDMimoConfig(d_model=128, pair_layout="interleaved", **kw), execution={"backend": "cuda", "precision": "ieee"}).cuda()
    mi2.load_state_dict(mi2.convert_pair_layout(mh.state_dict()))
    for (n, a), (_, b) in zip(sorted(mi.named_parameters()),
                              sorted(mi2.named_parameters())):
        assert torch.equal(a, b), f"round trip moved {n}"

    torch.manual_seed(4)
    x = torch.randn(2, 24, 128, device="cuda")
    outs = []
    for mod in (mi, mh):
        xi = x.clone().requires_grad_(True)
        y, _ = mod.scan(xi)
        # BOTH sides are backend="cuda" here: the conversion claim is about
        # the kernel's two state layouts, so a side that demoted to torch
        # would score the permutation without ever running the half branch.
        assert_kernel_served(mod, f"pair_layout={mod.pair_layout!r}")
        names = [n for n, _ in sorted(mod.named_parameters())]
        leaves = [xi] + [p for _, p in sorted(mod.named_parameters())]
        g = torch.autograd.grad(y.float().square().sum(), leaves,
                                allow_unused=True)
        outs.append((y.detach().float(), names, g))
    (y_i, names, g_i), (y_h, _, g_h) = outs
    rel = lambda a, b: ((a.double() - b.double()).norm()
                        / b.double().norm().clamp_min(1e-12)).item()
    assert rel(y_h, y_i) < 3e-5, f"forward {rel(y_h, y_i):.2e}"
    idx = pair_perm(mi.N, device="cuda")
    # above rank 1 the sibling GATE has its own B / C
    # projections (`use_g_b` / `use_g_c`), and they carry the state axis
    # exactly as `B_proj` / `C_proj` do -- `_pair_layout_targets` lists
    # all four.  A map that forgot them would score the conversion as wrong
    # when it is the TEST that is incomplete.
    perm_axis = {"B_proj.weight": 0, "C_proj.weight": 0,
                 "gB_proj.weight": 0, "gC_proj.weight": 0,
                 "B_norm.weight": 0, "C_norm.weight": 0,
                 "B_bias": 2, "C_bias": 2}
    if mi.gate_tie_idx is not None:
        # PAIRGATES: gates on complex pairs are
        # stored once per pair / real row -- layout-independent rows, so
        # they compare WITHOUT the state-axis permutation
        perm_axis.pop("gB_proj.weight")
        perm_axis.pop("gC_proj.weight")
    for n, a, b in zip(["x"] + names, g_i, g_h):
        if a is None or b is None or b.abs().max() == 0:
            continue
        if n in perm_axis:
            ax = perm_axis[n]
            if n.endswith("proj.weight"):
                blk = a.shape[0] // mi.N
                a = (a.unflatten(0, (mi.N, blk)).index_select(0, idx)
                     .flatten(0, 1))
            else:
                a = a.index_select(ax, idx)
        assert rel(b, a) < 5e-4, f"d{n} {rel(b, a):.2e}"

