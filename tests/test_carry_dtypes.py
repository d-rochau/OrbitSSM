"""Returned carries keep their dtypes across calls.

Pins (one rule for every family; the config defines the architecture):
  law       every floating carry a layer returns has the dtype of the same carry it was given; when no State was passed
            it has the default carry dtype, promote(parameter dtype, fp32); integer / bool leaves are untouched.
            Holds for `forward(..., return_state=True)`, `step`, `scan` and `scan_step` (grad on and off), for fp32 / bf16 / fp64 layers,
            for the default State and for a passed State with fp32 or fp64 carries.
  chain     forward -> step -> forward -> step keeps every carry dtype (a drifting State recompiles `torch.compile`
            and breaks the static buffers of CUDA-graph capture).
Cells: SSD-MIMO plain, with a B/C conv, with d_ssm_in=d_ssm_out=2; HybridMamba plain and with a `coord_potential`
row block (the `mu` carry).  CPU, torch backend; dtype only, values are pinned by the oracle tests.
`check_carry_law` / `check_chain` take any layer builder, so another family's cells can reuse them.
"""
import pytest
import torch

from orbit_ssm import HybridMambaConfig, HybridMambaSSM, SSDMimoConfig, SSDMimoSSM

B, L, D_MODEL = 2, 6, 32
EX = {"backend": "torch"}


def _ssd(**kw):
    return lambda: SSDMimoSSM(SSDMimoConfig(d_model=D_MODEL, d_state=16, **kw), execution=EX)


def _hm(**kw):
    return lambda: HybridMambaSSM(HybridMambaConfig(d_model=D_MODEL, d_state=16, **kw), execution=EX)


CELLS = {
    "ssd": _ssd(),
    "ssd_bc_conv": _ssd(bc_conv=True, d_conv=4),
    "ssd_r2": _ssd(d_ssm_in=2, d_ssm_out=2),
    "hm": _hm(),
    "hm_coord_potential": _hm(n_heads=2, d_inner=D_MODEL, row_blocks=[
        dict(rows=8, decay="coord_potential"), dict(rows=8, decay="coord_static")]),
}
MODULE_DTYPES = [torch.float32, torch.bfloat16, torch.float64]
#: the carries a call is given: None = the default State of `init_state()`
GIVEN = [None, torch.float32, torch.float64]


def dtypes(state):
    """{key: dtype} of the floating tensor leaves (nested dicts flattened with a dotted key)."""
    out = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out.update({f"{k}.{kk}": vv for kk, vv in dtypes(v).items()})
        elif torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.dtype
    return out


def make(build, dtype):
    torch.manual_seed(0)
    return build().to(dtype)


def incoming(m, given):
    return m.init_state(B) if given is None else m.init_state(B, dtype=given)


def expected(m, given):
    """The dtypes every returned carry must have."""
    want = dtypes(incoming(m, given))
    default = torch.promote_types(next(m.parameters()).dtype, torch.float32)
    if given is None:
        assert set(want.values()) == {default}, want        # the default rule itself
    return want


def scan_state(state):
    """The bare scan's carries: the layer State without the drive-conv window."""
    return None if state is None else type(state)({k: v for k, v in state.items() if k != "conv_state"})


def run(m, op, x, state, grad):
    with torch.set_grad_enabled(grad):
        if op == "forward":
            return m(x, state=state, return_state=True)[1]
        if op == "step":
            return m.step(x[:, 0], state)[1]
        drive = torch.randn(B, L, m.d_inner).to(x.dtype)       # the bare scan on projected inputs
        if op == "scan":
            return m.scan(drive, sel_x=x, state=scan_state(state), return_state=True)[1]
        return m.scan_step(drive[:, 0], scan_state(state), sel_x=x[:, 0])[1]


def check_carry_law(build, dtype, given, op, grad):
    m = make(build, dtype)
    x = torch.randn(B, L, D_MODEL).to(dtype)
    state = incoming(m, given)
    want = expected(m, given)
    if op in ("scan", "scan_step"):
        want = {k: v for k, v in want.items() if k != "conv_state"}
    assert dtypes(run(m, op, x, state, grad)) == want
    if given is None:                                       # no State passed: the default carry dtype too
        with torch.set_grad_enabled(grad):
            out = {"forward": lambda: m(x, return_state=True)[1], "step": lambda: m.step(x[:, 0], None)[1],
                   "scan": lambda: m.scan(torch.randn(B, L, m.d_inner).to(dtype), sel_x=x, return_state=True)[1],
                   "scan_step": lambda: m.scan_step(torch.randn(B, m.d_inner).to(dtype), None, sel_x=x[:, 0])[1],
                   }[op]()
        assert dtypes(out) == want


def check_chain(build, dtype, given):
    m = make(build, dtype)
    x = torch.randn(B, L, D_MODEL).to(dtype)
    want = expected(m, given)
    state = incoming(m, given)
    for grad in (False, True):
        with torch.set_grad_enabled(grad):
            _, state = m(x, state=state, return_state=True)
            assert dtypes(state) == want, ("forward", grad)
            _, state = m.step(x[:, 0], state)
            assert dtypes(state) == want, ("step", grad)
            _, state = m(x[:, :3], state=state, return_state=True)
            assert dtypes(state) == want, ("forward again", grad)
            _, state = m.step(x[:, 1], state)
            assert dtypes(state) == want, ("step again", grad)


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("dtype", MODULE_DTYPES)
@pytest.mark.parametrize("given", GIVEN)
@pytest.mark.parametrize("op", ["forward", "step", "scan", "scan_step"])
@pytest.mark.parametrize("grad", [False, True])
def test_returned_carries_keep_the_incoming_dtypes(cell, dtype, given, op, grad):
    """contract: forward(return_state), step and the bare scan / scan_step return every carry in the incoming carry's dtype (default
    promote(param, fp32) without a State), on the no-grad and the differentiable path."""
    check_carry_law(CELLS[cell], dtype, given, op, grad)


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("dtype", MODULE_DTYPES)
@pytest.mark.parametrize("given", GIVEN)
def test_forward_step_forward_chain_keeps_the_dtypes(cell, dtype, given):
    """contract: forward -> step -> forward -> step never changes a carry dtype."""
    check_chain(CELLS[cell], dtype, given)
