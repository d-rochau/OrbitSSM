"""A caller who has set torch's fp32 matmul-precision API (`torch.backends.cuda.matmul.fp32_precision`) can train in fp32.

torch >= 2.9 refuses to READ the legacy `allow_tf32` flag once the new API is set, so a kernel host path that touches the
legacy flag crashes the backward of every such user.  The kernel cells run with the default flags; this test runs three fp32
cells (real R = O = 1 with and without B/C biases, rotating R = O = 1) through the same forward + every-gradient oracle check with the new API set to each of its settings, and
the CPU witness that `ieee_matmul` works under a caller's setting and restores it.
"""
import pytest
import torch

import _contract
import test_contract as TC
from _contract import core
from orbit_ssm.kernels.fp32_class import ieee_matmul

pytestmark = pytest.mark.contract_cells      # sized here, not thinned by the profile

_BE = torch.backends.cuda.matmul
_CELLS = {"real_r1o1": dict(cplx="real"), "real_r1o1_unbiased": dict(cplx="real", bias="none"), "half_r1o1": dict(cplx="half")}
_RUNS = [("real_r1o1", "tf32"), ("real_r1o1_unbiased", "tf32"), ("half_r1o1", "tf32"), ("real_r1o1", "ieee")]


@pytest.fixture
def precision_api():
    if not hasattr(_BE, "fp32_precision"):
        pytest.skip("torch without the fp32_precision API")
    old = _BE.fp32_precision
    yield _BE
    _BE.fp32_precision = old


def test_ieee_matmul_under_a_caller_setting_sets_ieee_and_restores(precision_api):
    """witness: `ieee_matmul` works under the caller's fp32_precision setting (the legacy flag cannot be read there)."""
    precision_api.fp32_precision = "tf32"
    with ieee_matmul():
        assert precision_api.fp32_precision == "ieee"
        a = torch.randn(4, 3, dtype=torch.float32)
        torch.matmul(a, a.T)
    assert precision_api.fp32_precision == "tf32"


@pytest.mark.parametrize("name,setting", _RUNS)
def test_fp32_training_under_the_new_precision_api(require_cuda, precision_api, name, setting):
    """contract: forward + every gradient of an fp32 cell match the fp64 oracle with `fp32_precision` set to `setting`."""
    from _contract.adapters import ssd as S
    ad = TC.FAMILIES["ssd"]
    c = S._full(dict(_CELLS[name], dtype="fp32", prec="tf32"))
    assert ad.space.valid(c)
    precision_api.fp32_precision = setting
    cell = core.Cell("ssd", "recipe", f"fp32_{name}_{setting}", tuple(sorted(c.items())))
    # the shared C B^T product of the real backward is a calibrated speed lever (off until a GPU's calibration turns it on);
    # forcing it (and the pair-fold that reads it) makes the host path that product takes run on every GPU, calibrated or not
    env = {**ad.env(c), "SSD_MIMO_BWD_KC_SHARED_CB": "1", "SSD_MIMO_BWD_DD_IN_KBC": "1",
           "SSD_MIMO_SISO_FAST": "0"}      # the general fused backward
    pool = ad.new_pool(c)
    TC._oracle_run(ad, c, cell, env, pool=pool)
    TC._settle(ad, c, cell, env, None, pool, True, None)
    pool.assert_unbiased()
