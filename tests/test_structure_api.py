"""Construction and state-boundary regressions from the structure audit."""

import pytest
import torch
from orbit_ssm.model import CausalLM, BlockStack
from orbit_ssm.config import HybridMambaConfig
from orbit_ssm.ssm import CONFIGS, MIXERS, SSMS
from orbit_ssm import SSDMimoConfig


class CountingBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, x, state=None, return_state=False):
        self.calls += 1
        return (x + 1, state) if return_state else x + 1
    def step(self, x, state):
        self.calls += 1
        return x + 1, state
    def reset_state(self, state, mask=None): self.calls += 1
    def set_state(self, state, other): self.calls += 1


@pytest.mark.parametrize('length', [0, 1, 3])
@pytest.mark.parametrize('operation', ['forward', 'step', 'reset', 'set', 'set_other'])
def test_invalid_state_lengths_raise_before_any_layer_runs(length, operation):
    stack = BlockStack([CountingBlock(), CountingBlock()], 2)
    bad = [None] * length
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError, match='exactly 2'):
        if operation == 'forward': stack(x[:, None], state=bad, return_state=True)
        elif operation == 'step': stack.step(x, bad)
        elif operation == 'reset': stack.reset_state(bad)
        elif operation == 'set': stack.set_state(bad, [None] * 2)
        else: stack.set_state([None] * 2, bad)
    assert all(b.calls == 0 for b in stack.layers)


def test_hybrid_config_builds_complete_lm_mixer():
    cfg = HybridMambaConfig(d_inner=64, d_model=32, d_state=8,
                           n_heads=4, d_conv=0)
    m = CausalLM(vocab_size=16, layers=cfg, n_layers=1, execution={'backend': 'sequential'})
    mixer = m.backbone.layers[0].mixer
    assert isinstance(mixer, MIXERS['hybrid_mamba'])
    assert mixer.d_inner == 64
    y = m(torch.tensor([[1, 2, 3]]))
    assert y.shape == (1, 3, 16)
    y.square().mean().backward()
    assert mixer.out_proj.weight.grad is not None


def test_registry_keys_and_canonical_classes():
    from orbit_ssm import config
    from orbit_ssm.ssm.ssd_mimo import SSDMimoConfig
    assert set(CONFIGS) == set(MIXERS) == set(SSMS)
    assert config.SSDMimoConfig is SSDMimoConfig


def test_config_respects_requested_family():
    cfg = HybridMambaConfig(d_inner=64, d_state=8, n_heads=4, d_model=32, d_conv=0)
    m = CausalLM._build_mixer(32, cfg)
    assert isinstance(m, MIXERS["hybrid_mamba"])
    with pytest.raises(ValueError, match="unknown type"):
        CausalLM._build_mixer(32, dict(type="unknown"))


def test_public_geometry_properties_equal_constructor_values():
    """Pins: the read-only public names (d_ssm_in, n_heads, ...) return the constructor values; the
    paper-notation attributes (.R .O .P .H .N .G .k) they read stay unchanged."""
    from orbit_ssm import SSDMimoSSM, HybridMambaSSM
    s = SSDMimoSSM(SSDMimoConfig(d_model=64, d_state=16, d_inner=128, d_ssm_in=2, d_ssm_out=2, n_groups=2, d_head=4))
    assert (s.d_model, s.d_inner, s.d_state, s.d_ssm_in, s.d_ssm_out, s.n_groups, s.d_head, s.n_heads) \
        == (64, 128, 16, 2, 2, 2, 4, 32)
    assert (s.R, s.O, s.G, s.P, s.H, s.N) == (2, 2, 2, 4, 32, 16)
    h = HybridMambaSSM(HybridMambaConfig(d_model=64, d_state=16, d_inner=128, n_heads=4))
    assert (h.d_model, h.d_inner, h.d_state, h.n_heads, h.d_head, h.d_ssm_in, h.d_ssm_out) \
        == (64, 128, 16, 4, 32, 1, 1)
    for m in (s, h):
        with pytest.raises(AttributeError):
            m.n_heads = 3
