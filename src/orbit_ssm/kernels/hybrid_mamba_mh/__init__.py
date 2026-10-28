"""HybridMamba-MH (structure B) kernel -- self-contained copy.

Re-exports the main differentiable entry point, `mh_scan`. It is the second
production kernel family alongside SSD-MIMO.

Independent of `hybrid_mamba` and of `hybrid_mamba.kernels.mamba1` in
PRODUCTION -- nothing here imports either at module scope, which is what let
the package move over unchanged. Structure B exists because structure A's ABI
cannot express axis sharing, so reusing its code would reimport the defect.

ONE production forward and ONE production backward:
`mh_scan` over `fwd_gluon.mh_forward` and `bwd_gluon.mh_backward`.
Helpers shared by both kernels live in `common_gluon.py`. The fp64 oracles `mh_reference` and
`mh_reference_cumprod` stay: they are the independent implementations the
collected suite scores the pair against.

`assert_pair_tied` is the family's one ABI assertion. It is
NOT on the executed path by default -- `mh_scan(validate=True)` or
`HYBRID_SSM_MH_VALIDATE=1` runs it, and the collected suite runs with it on.
It is the only check of the pair tie there can be: on untied input the kernel
and both oracles agree to fp32 rounding, because they fold the rotation the
same way, so no oracle comparison can see a pair that has stopped being a
complex mode.
"""
from .layout import (  # noqa: F401
    DYN_EXP, DYN_RAW, DYN_SOFTPLUS,
    GATE_EXP, GATE_RAW, GATE_SIGMOID, GATE_SOFTPLUS, GATE_TANH,
    LAW_A_DYN, LAW_A_FIX, LAW_MU,
    Frac, Shapes,
    assert_pair_tied, fracs_to_rows, law_mask, norm_fracs,
    validate_forward_inputs,
)
from .reference import (  # noqa: F401
    mh_reference, mh_reference_cumprod, rel_err,
)
try:
    from .functions import mh_scan  # noqa: F401
except ImportError as _gluon_import_error:       # a Triton build without triton.experimental.gluon (the MH kernels are Gluon-only)
    _why = f"{type(_gluon_import_error).__name__}: {_gluon_import_error}"

    def mh_scan(*args, **kwargs):
        from ..portable import PortableRefusal
        raise PortableRefusal(
            "mh_scan: the MH kernels are Gluon kernels (triton.experimental.gluon) and this Triton build cannot import them "
            f"({_why}); use Execution(backend='torch') (`mh_torch_scan`, PyTorch ops).  Gap G-MH of "
            "the portability audit")

__all__ = [
    "LAW_A_FIX", "LAW_A_DYN", "LAW_MU",
    "DYN_RAW", "DYN_SOFTPLUS", "DYN_EXP",
    "GATE_RAW", "GATE_SIGMOID", "GATE_EXP", "GATE_TANH", "GATE_SOFTPLUS",
    "Frac", "Shapes", "norm_fracs", "fracs_to_rows", "law_mask",
    "validate_forward_inputs", "assert_pair_tied",
    "mh_reference", "mh_reference_cumprod", "rel_err",
    "mh_scan",
]
