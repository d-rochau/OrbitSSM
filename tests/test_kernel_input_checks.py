"""Pins: a CUDA kernel entry refuses CPU / mixed / mistyped inputs with a NAMED
ValueError that points at backend='torch', never a bare assert.  CPU only: the refusal must fire before any
CUDA call.

Sites: `ssd_mimo_scan`'s fused entry (was `assert x_in.is_cuda`, an empty
AssertionError), `gate_epilogue.silu_gate` and the gate-adjoint Z check (were
asserts), `hybrid_mamba_mh.layout.validate_forward_inputs` (had no device
check; it is the CUDA (Gluon) launch's validator, its one caller is fwd_gluon).  The HM layer call site (`ssm/hybrid_mamba.py`) is B.
"""
import pytest
import torch

from orbit_ssm import HybridMambaConfig, HybridMambaSSM, SSDMimoConfig, SSDMimoSSM
from orbit_ssm.kernels.input_checks import check_cuda_input

MSG = r"is on cpu, but this is the CUDA kernel route.*backend': 'torch'"


def test_check_cuda_input_names_the_argument_and_the_cpu_route():
    """envelope: check_cuda_input refuses a CPU or non-tensor input by name, pointing at backend='torch'."""
    with pytest.raises(ValueError, match=r"who: b is on cpu.*backend': 'torch'"):
        check_cuda_input("who", {"a": None, "b": torch.zeros(2)})
    with pytest.raises(ValueError, match="must be a torch.Tensor"):
        check_cuda_input("who", {"a": 1.0})
    check_cuda_input("who", {"a": None})          # nothing given: nothing to refuse


@pytest.mark.parametrize("family", ["ssd_mimo", "hybrid_mamba"])
def test_backend_cuda_on_cpu_tensors_is_a_named_value_error(family):
    """envelope: a backend='cuda' layer on CPU tensors raises a named ValueError."""
    if family == "ssd_mimo":
        layer = SSDMimoSSM(SSDMimoConfig(d_model=32, d_state=16, expand=1, d_conv=0, d_ssm_in=1,
                                         d_ssm_out=1, n_groups=16), execution={"backend": "cuda"})
    else:
        layer = HybridMambaSSM(HybridMambaConfig(d_model=32, d_state=16),
                               execution={"backend": "cuda"})
    with pytest.raises(ValueError, match=MSG):
        layer(torch.randn(2, 8, 32))


def test_silu_gate_refuses_cpu_and_shape_mismatch_by_name(monkeypatch):
    """envelope: silu_gate refuses CPU tensors by name, and a y/z shape mismatch by name
    before any launch (the device check is stubbed for the second case, R13)."""
    from orbit_ssm.kernels.ssd_mimo import gate_epilogue
    with pytest.raises(ValueError, match=r"silu_gate: y is on cpu"):
        gate_epilogue.silu_gate(torch.zeros(4), torch.zeros(4))
    monkeypatch.setattr(gate_epilogue, "check_cuda_input", lambda who, tensors: None)
    with pytest.raises(ValueError, match=r"silu_gate: y \(4,\) and z \(5,\) must have one shape"):
        gate_epilogue.silu_gate(torch.zeros(4), torch.zeros(5))


def test_mh_validator_refuses_cpu_tensors_by_name():
    """envelope: the MH CUDA-route validator refuses CPU tensors by name."""
    from orbit_ssm.kernels.hybrid_mamba_mh.layout import validate_forward_inputs
    B, H, P, N, L = 1, 2, 4, 8, 16
    with pytest.raises(ValueError, match=r"mh_scan \(CUDA route\): x is on cpu"):
        validate_forward_inputs(torch.zeros(B, H, P, L), torch.zeros(B, H, L),
                                torch.zeros(H, P, N), torch.zeros(B, N, L), torch.zeros(B, N, L))


@pytest.mark.parametrize("cap,outcome", [((7, 5), "refuse"), ((8, 0), "ok"), ((8, 6), "warn"), ((9, 0), "ok"),
                                         ((10, 0), "warn"), ((12, 0), "ok")])
def test_cuda_arch_gate_refuses_below_sm80_and_warns_outside_the_validated_set(monkeypatch, cap, outcome):
    """contract: below sm_80 the CUDA route is refused by name (pointing at the torch
    backend); sm_80 / sm_90 / sm_120 run silently; any other architecture runs with one RuntimeWarning."""
    import warnings
    from orbit_ssm.kernels import input_checks
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: cap)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: "TestGPU")
    input_checks.check_cuda_arch.cache_clear()
    try:
        if outcome == "refuse":
            with pytest.raises(ValueError, match=r"sm_75\) is below the supported minimum sm_80.*'backend': 'torch'"):
                input_checks.check_cuda_arch(0)
            return
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            input_checks.check_cuda_arch(0)
            input_checks.check_cuda_arch(0)                 # cached: at most one warning per device
        runtime = [w for w in seen if issubclass(w.category, RuntimeWarning)]
        assert len(runtime) == (1 if outcome == "warn" else 0), [str(w.message) for w in runtime]
    finally:
        input_checks.check_cuda_arch.cache_clear()
