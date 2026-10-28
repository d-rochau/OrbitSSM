"""bf16 single-token decoding.

Pins:
  law       a bf16 layer decodes with `step()` from its default `init_state()` (fp32 carries) and from a bf16 prefill,
            on the no-grad (fast) and the differentiable step;
            the step loop and the parallel forward both track the fp64 recurrence of the SAME bf16-rounded weights
            within the derived 16-bit bound (`tol16`, 5 draws, pooled signed bias)
  regression  the step used to feed the fp32 conv carry's output into the bf16 projections
            ("expected m1 and m2 to have the same dtype")
"""

import pytest
import torch

from _tolerance import DrawPool, _rel_frob, _signed_bias, draw_seeds, escalation_seeds, tol16
from orbit_ssm import HybridMambaConfig, HybridMambaSSM, SSDMimoConfig, SSDMimoSSM


B, L, D_MODEL, N = 2, 12, 32, 16


CELLS = {
    "ssd_mimo": (SSDMimoSSM, SSDMimoConfig, {}),
    "ssd_mimo_r2": (SSDMimoSSM, SSDMimoConfig, dict(d_ssm_in=2, d_ssm_out=2)),
    "hybrid_mamba": (HybridMambaSSM, HybridMambaConfig, {}),
}


BF16 = torch.bfloat16


def _steps(m, x, state):
    ys = []
    for t in range(x.shape[1]):
        y, state = m.step(x[:, t], state)
        ys.append(y)
    return torch.stack(ys, 1)


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_bf16_step_and_forward_track_the_fp64_recurrence(cell):
    """contract: in bf16, `forward`, `step` from the default init_state, `step` after a prefill and the
    differentiable `step` each track the fp64 recurrence on the same bf16-rounded weights within tol16 (>= 5
    draws, signed-bias check); every arm keeps the bf16 output dtype."""
    layer, config, kw = CELLS[cell]
    pool = DrawPool()

    def draw(seed, check=True):
        torch.manual_seed(seed)
        m16 = layer(config(d_model=D_MODEL, d_state=N, **kw), execution={"backend": "torch"}).to(BF16)
        ref = layer(config(d_model=D_MODEL, d_state=N, **kw), execution={"backend": "torch"}).double()
        ref.load_state_dict({k: v.double() for k, v in m16.state_dict().items()})   # the same bf16-rounded weights
        x16 = torch.randn(B, L, D_MODEL, generator=torch.Generator().manual_seed(seed + 1)).to(BF16)
        with torch.no_grad():
            want = ref(x16.double())
            arms = {"forward": m16(x16),
                    "step_from_init_state": _steps(m16, x16, m16.init_state(B)),
                    "step_after_prefill": torch.cat([m16(x16[:, :4]),
                                                     _steps(m16, x16[:, 4:], m16(x16[:, :4], return_state=True)[1])], 1)}
        with torch.enable_grad():                  # the differentiable step (SSD's no-grad step is its fast route)
            arms["step_with_grad"] = _steps(m16, x16, m16.init_state(B)).detach()
        # 16-bit terms per output element: in_proj (d_model), the state contraction (N) over the L-token prefix,
        # out_proj (d_inner); one 16-bit arm against fp64
        depth = D_MODEL + N + L + m16.d_inner
        bound = tol16(BF16, depth, arms=1)
        for name, got in arms.items():
            assert got.dtype == BF16, (name, got.dtype)
            err = _rel_frob(got, want)
            assert err < bound or not check, f"{cell} {name} draw {seed}: rel err {err:.3e} > tol16(depth={depth}) {bound:.3e}"
            if check:
                pool.add(name, bound, *_signed_bias(got, want))
        return {name: (got, want, bound) for name, got in arms.items()}

    for seed in draw_seeds(BF16):
        draw(seed)
    # a pooled mean above bound/4 is redrawn on 30 draws and judged by size AND significance (tests/_tolerance.py DrawPool)
    pool.assert_unbiased(lambda seed: draw(seed, check=False), escalation_seeds(BF16, 0))


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("dtype,carry", [(torch.bfloat16, torch.float32), (torch.float32, torch.float32),
                                         (torch.float64, torch.float64)])
def test_default_carries_follow_the_parameters_never_below_fp32(cell, dtype, carry):
    """law (one rule for every family): `init_state()` without a dtype allocates every carry in
    promote(parameter dtype, fp32) -- an fp64 layer used to get fp32 carries in HM."""
    layer, config, kw = CELLS[cell]
    m = layer(config(d_model=D_MODEL, d_state=N, **kw), execution={"backend": "torch"}).to(dtype)
    state = m.init_state(B)
    assert {k: v.dtype for k, v in state.items() if v.is_floating_point()} == \
        {k: carry for k, v in state.items() if v.is_floating_point()}
