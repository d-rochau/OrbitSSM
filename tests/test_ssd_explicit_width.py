"""Explicit Paper 1 widths at the mixer, model and harness boundaries."""

import json
import pytest
import torch
from orbit_ssm.configuration import resolve_mixer_width
from orbit_ssm.model import CausalLM
from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig
from orbit_ssm import HybridMambaConfig

from _design_warnings import SILENCE_SSD_DESIGN_WARNINGS

pytestmark = SILENCE_SSD_DESIGN_WARNINGS  # builds the warned configurations on purpose


@pytest.mark.parametrize('output_rank', [1, 2])
def test_explicit_width_preserves_expansion_weights_outputs_and_gradients(output_rank):
    common = dict(d_model=40, d_state=8, d_conv=4,
                  d_ssm_in=2, d_ssm_out=output_rank, d_head=32)
    torch.manual_seed(12)
    old = SSDMimoSSM(SSDMimoConfig(**common, expand=2.4), execution={'backend': 'torch'}).double()
    torch.manual_seed(12)
    new = SSDMimoSSM(SSDMimoConfig(**common, d_inner=96), execution={'backend': 'torch'}).double()
    assert new.d_inner == old.d_inner == 96
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 3, 40, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    y, yr = old(x), new(xr)
    torch.testing.assert_close(y, yr, rtol=0, atol=0)
    g = torch.autograd.grad(y.square().sum(), (x, *old.parameters()), allow_unused=True)
    gr = torch.autograd.grad(yr.square().sum(), (xr, *new.parameters()), allow_unused=True)
    for a, b in zip(g, gr):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    config = json.loads(json.dumps(new.to_recipe()))
    restored = SSDMimoSSM.from_recipe(config).double()
    restored.load_state_dict(new.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), yr, rtol=0, atol=0)


def test_model_per_layer_explicit_width_roundtrip():
    base = dict(type='ssd_mimo', d_state=8, d_ssm_in=2, d_head=32)
    model = CausalLM(vocab_size=32, layers=[dict(base, d_inner=64), dict(base, d_inner=96)], d_model=40, execution={'backend': 'torch'})
    assert [b.mixer.d_inner for b in model.backbone.layers] == [64, 96]
    config = json.loads(json.dumps(model.to_recipe()))
    restored = CausalLM.from_recipe(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randint(0, 32, (2, 3))
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


@pytest.mark.parametrize('expand,width', [(2,96), (True,None),
    (float('nan'),None), (float('inf'),None), (0,None), (-1,None), (0.001,None),
    (None,True), (None,0), (None,64.5)])
def test_invalid_or_conflicting_widths_refuse(expand, width):
    with pytest.raises(ValueError):
        resolve_mixer_width(32, expand, width)


def test_nested_width_and_conflicting_width_refuse():
    # the nested core `d_inner` is gone: `ssm_kwargs` is an unknown keyword,
    # and the shared width head refuses a conflicting or non-dividing width by name.
    with pytest.raises(TypeError, match='ssm_kwargs'):
        SSDMimoSSM(SSDMimoConfig(d_model=40, d_inner=96, ssm_kwargs=dict(d_inner=64)))
    with pytest.raises(ValueError, match='conflicts'):
        CausalLM(vocab_size=32, layers=dict(type='ssd_mimo', expand=2, d_inner=128), n_layers=1, d_model=40)
    with pytest.raises(ValueError, match='divisor'):
        CausalLM(vocab_size=32, layers=dict(type='ssd_mimo', d_inner=96), n_layers=1, d_model=40)
    assert CausalLM(vocab_size=32, layers=dict(type='ssd_mimo', d_inner=128), n_layers=1, d_model=40).backbone.layers[0].mixer.d_inner == 128
