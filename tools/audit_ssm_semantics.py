#!/usr/bin/env python3
"""Equation-first CPU audit. Exit 1 means an experiment-readiness gate failed.

The endpoint oracle takes mathematical transitions and unscaled writes; it
imports no production preparation, scan, step, rotation or reference helpers.
Production imports occur only in the system-under-test adapter below.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
from orbit_ssm import SSDMimoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT) + "/src")


def endpoint_states(transitions, writes, steps, lambdas, h0):
    """h[t] = A[t]h[t-1] + dt[t](lam[t]v[t]+(1-lam[t])A[t]v[t-1]).

    All vectors are in physical state coordinates. v[-1]=0, independently
    of h0. This is transported endpoint quadrature, not implicit trapezoid
    integration of the complete differential equation or exact ZOH.
    """
    h = h0
    previous = torch.zeros_like(h0)
    result = []
    for a, v, dt, lam in zip(transitions, writes, steps, lambdas):
        h = a @ h + dt * (lam * v + (1 - lam) * (a @ previous))
        result.append(h)
        previous = v
    return torch.stack(result)


def scalar_case(*, trap, backend, initial=0.0):
    """Known scalar B=C=1, constant rate 1, dt=.05, x=[1,0,2]."""
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig

    # the layer is its own scan: `model.scan(...)` is the bare recurrence audited here
    model = SSDMimoSSM(SSDMimoConfig(
        d_model=1, d_state=1, d_inner=1, n_groups=1, n_heads=1, d_ssm_in=1, d_ssm_out=1, skip=False,
        proj_source="layer_input", bc_conv=False, b_norm=False, c_norm=False, bc_bias="none",
        b_gate=False, c_gate=False,
        row_blocks=[{"frac": 1.0, "decay": "head_static"}],
        write_rule=("trapezoid" if trap else "euler"),
        
    ), execution={"backend": backend}).double()
    with torch.no_grad():
        model.A_log.zero_()
        model.B_proj.weight.fill_(1)
        model.C_proj.weight.fill_(1)
        model.dt_proj.weight.zero_()
        model.dt_proj.bias.fill_(math.log(math.expm1(.05)))
        if trap:
            model.trap_proj.weight.zero_()
            model.trap_proj.bias.zero_()

    x = torch.tensor([1., 0., 2.], dtype=torch.float64, requires_grad=True)
    h0 = torch.tensor([initial], dtype=torch.float64, requires_grad=True)
    from orbit_ssm.layers.state import ScanState
    actual = model.scan(x.reshape(1, 3, 1), sel_x=torch.ones(1, 3, 1,
                      dtype=torch.float64), state=ScanState(model).init_state((x.reshape(1, 3, 1)).shape[0], h_init=h0.reshape(1, 1, 1, 1), device=(x.reshape(1, 3, 1)).device, dtype=next(model.parameters()).dtype))
    # Independent scalar evaluation of the documented step law. Do not call
    # _dt_tail_eager or read values back from prepared kernel arguments.
    # Delta = softplus(dt_bias) = .05 exactly; no soft cap
    dt = .05
    a = torch.full((3, 1, 1), math.exp(-dt), dtype=torch.float64)
    lam_logit = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    lam = torch.sigmoid(lam_logit).expand(3) if trap else torch.ones(3)
    expected = endpoint_states(a, x[:, None], torch.full((3,), dt,
                               dtype=torch.float64), lam, h0).flatten()
    params = (x, h0, model.trap_proj.bias) if trap else (x, h0)
    oracle_params = (x, h0, lam_logit) if trap else (x, h0)
    actual_grads = torch.autograd.grad(actual.sum(), params, allow_unused=True)
    expected_grads = torch.autograd.grad(expected.sum(), oracle_params)
    error = float((actual.flatten() - expected).abs().max().detach())
    grad_errors = {}
    for name, got, want in zip(("input", "initial_state", "lambda_logit"),
                               actual_grads, expected_grads):
        got = torch.zeros_like(want) if got is None else got.reshape(want.shape)
        grad_errors[name] = float((got - want).abs().max().detach())
    return dict(backend=backend,
                write_rule=("trapezoid" if trap else "euler"),
                initial_state=initial,
                actual=actual.detach().flatten().tolist(),
                expected=expected.detach().tolist(), max_abs_error=error,
                gradient_max_abs_errors=grad_errors,
                passed=error < 1e-10 and max(grad_errors.values()) < 1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON evidence path outside the checkout")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("write evidence outside the checkout")
    cases = [scalar_case(trap=trap, backend=backend, initial=initial)
             for backend in ("torch", "sequential")
             for trap in (False, True) for initial in (0., .7)]
    from orbit_ssm.ssm.ssd_mimo import _dt_tail_eager
    cap = _dt_tail_eager(torch.tensor([-100., -30.], dtype=torch.float64), .2, .01)
    report = dict(git_sha=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        repo_path=str(ROOT), torch_version=torch.__version__,
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        git_status=subprocess.check_output(["git", "status", "--short"],
                                           cwd=ROOT, text=True).splitlines(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__).resolve(),
                                 ROOT / "src/orbit_ssm/ssm/ssd_mimo.py")},
        scope="CPU fp64 real scalar SSD; not CUDA, MIMO, rotation or full-family certification",
        cases=cases, softcap_fp64=cap.tolist(),
        softcap_nonnegative=bool((cap >= 0).all()))
    report["endpoint_gate_passed"] = all(c["passed"] for c in cases)
    report["experiment_gate_passed"] = report["endpoint_gate_passed"] and report["softcap_nonnegative"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["experiment_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
