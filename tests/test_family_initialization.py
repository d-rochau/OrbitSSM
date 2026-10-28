"""Published-family initialization and fresh recipe behavior (no CUDA needed)."""
import json
import math

import pytest
import torch

from orbit_ssm.ssm.hybrid_mamba import HybridMambaConfig, HybridMambaSSM
from orbit_ssm.ssm.ssd_mimo import SSDMimoConfig, SSDMimoSSM
from orbit_ssm.model import CausalLM


pytestmark = [pytest.mark.paper1, pytest.mark.contract]


def assert_uniform_projection(weight, width):
    # rtol=.25 below is a SAMPLING tolerance on the variance of a random
    # initialiser, not a rounding bound between two arms: A7's u/c model
    # does not apply, so it stays literal (A7 scope).
    # Published M3 uses ordinary Linear rows, uniform +/- 1/sqrt(input width).
    # The variance check distinguishes the former normal(0,.02) initializer.
    assert weight.abs().max() <= 1 / math.sqrt(width)
    torch.testing.assert_close(weight.var(), weight.new_tensor(1 / (3 * width)),
                               rtol=.25, atol=0)


@pytest.mark.parametrize('per_sibling', [False, True])
def test_ssd_uses_m3_projection_bias_and_clock_initialization(per_sibling):
    torch.manual_seed(147)
    cfg = SSDMimoConfig(d_inner=256, d_model=256, d_state=64, n_heads=8,
                       d_ssm_in=2, d_ssm_out=4,
                       row_blocks=[{'frac': 1.0, 'decay': 'head_dynamic', 'complex': True, 'gate': True}],
                       a_dyn_per_sibling=per_sibling)
    core = cfg.build()
    assert core.a_dyn_proj.out_features == core.H * (core.R if per_sibling else 1)
    for name in ('B_proj', 'C_proj', 'dt_proj', 'a_dyn_proj', 'theta_proj', 'trap_proj'):
        assert_uniform_projection(getattr(core, name).weight, core.d_sel)
    for name in ('a_dyn_proj', 'trap_proj'):
        bias = getattr(core, name).bias
        torch.testing.assert_close(bias, torch.zeros_like(bias), rtol=0, atol=0)
    torch.testing.assert_close(core.D, torch.ones_like(core.D), rtol=0, atol=0)
    # B/C bias: this cell is R=2, so the `bc_bias_init` rule
    # resolves "auto" to "spread" -- sibling 0 is EXACTLY the old all-ones
    # vector and the others carry the orthogonal DCT directions that lift
    # the sibling block off rank 1.  At R == 1 (the exact presets) "spread"
    # reduces to "ones", which the LM cell below still pins.
    assert core.bc_bias_init == 'spread'
    for value in (core.B_bias, core.C_bias):
        torch.testing.assert_close(value[:, 0], torch.ones_like(value[:, 0]),
                                   rtol=0, atol=0)
        assert not torch.equal(value, torch.ones_like(value))
    dt = torch.nn.functional.softplus(core.dt_proj.bias)
    assert dt.min() >= .001 and dt.max() <= .1
    document = json.loads(json.dumps(core.to_recipe()))
    assert document['model']['dt_init_floor'] == 1e-4
    restored = SSDMimoSSM.from_recipe(document)
    restored.load_state_dict(core.state_dict(), strict=True)
    assert restored.to_recipe() == document


def test_ssd_clock_floor_survives_lm_initialization():
    model = CausalLM(vocab_size=16, layers=dict(type='ssd_mimo', d_state=16, n_heads=2, d_ssm_in=1, d_ssm_out=1, dt_min=.001, dt_max=.002, dt_init_floor=.003), n_layers=1, d_model=16, execution={'backend': 'torch'})
    core = model.backbone.layers[0].mixer
    torch.testing.assert_close(torch.nn.functional.softplus(core.dt_proj.bias),
                               torch.full_like(core.dt_proj.bias, .003))
    torch.testing.assert_close(core.B_bias, torch.ones_like(core.B_bias))
    torch.testing.assert_close(core.C_bias, torch.ones_like(core.C_bias))
    for bad in (0, -1, float('inf'), float('nan')):
        with pytest.raises(ValueError, match='dt_init_floor'):
            SSDMimoSSM(SSDMimoConfig(d_model=16, d_state=16, n_heads=2, d_ssm_in=1, proj_source='main_branch', dt_init_floor=bad, d_inner=16))


def test_ssd_mixer_projections_take_the_m3_default_init_in_an_lm():
    """Pins the init policy ("our default should be
    the m3 default for ssd_mimo"): SSDMimoSSM's in_proj keeps PyTorch's
    default nn.Linear init (Kaiming-uniform, bound 1/sqrt(fan_in), variance
    1/(3 fan_in)) -- as Mamba-3's in_proj, never re-initialised -- both
    standalone and inside CausalLM; the LM's own policy is unchanged:
    embedding N(0, 0.02), MLP gate/up default Linear, residual-out weights
    (mixer out_proj, MLP down) Kaiming-uniform / sqrt(n_residuals).  At
    d_model 256 the former trunc-normal(0.02) has variance 4.0e-4 against
    1.3e-3 here, 3.3x outside the rtol=0.25 band, so it cannot pass."""
    torch.manual_seed(20260923)
    d_model, d_mlp, n_layers = 256, 512, 2
    kw = dict(n_heads=4, d_ssm_in=2, d_ssm_out=2, d_state=16)
    mixer = SSDMimoSSM(SSDMimoConfig(d_model=d_model, **kw), execution={'backend': 'torch'})
    assert_uniform_projection(mixer.in_proj.weight, d_model)
    assert_uniform_projection(mixer.out_proj.weight, mixer.out_proj.in_features)

    lm = CausalLM(vocab_size=64, layers=dict(type='ssd_mimo', **kw), n_layers=n_layers, d_model=d_model, d_mlp=d_mlp)
    n_res = lm.backbone.n_residuals
    torch.testing.assert_close(lm.embedding.weight.std(),
                               torch.tensor(0.02), rtol=.05, atol=0)
    for block in lm.backbone.layers:
        mix = block.mixer
        assert isinstance(mix, SSDMimoSSM)
        assert_uniform_projection(mix.in_proj.weight, d_model)
        assert_uniform_projection(mix.out_proj.weight * math.sqrt(n_res),
                                  mix.out_proj.in_features)
        assert_uniform_projection(block.mlp.gate.weight, d_model)
        assert_uniform_projection(block.mlp.up.weight, d_model)
        assert_uniform_projection(block.mlp.down.weight * math.sqrt(n_res), d_mlp)


@pytest.mark.parametrize('heads', [2, 256])
def test_mh_defaults_to_m1_delta_initialization_for_dense_and_low_rank(heads):
    torch.manual_seed(113)
    cfg = HybridMambaConfig(d_inner=256, d_model=128, d_state=16, n_heads=heads,
                           proj_source='main_branch')
    core = cfg.build(execution={'backend': 'torch'})
    assert core.dt_init == 'random'
    assert (core.dt_low is None) == (heads == 2)
    rank = core.dt_proj.in_features
    assert core.dt_proj.weight.abs().max() <= rank ** -.5
    torch.testing.assert_close(core.dt_proj.weight.var(),
                               core.dt_proj.weight.new_tensor(1 / (3 * rank)),
                               rtol=.25, atol=0)
    expected = torch.arange(1, 17).float().expand(heads, core.P, 16)
    torch.testing.assert_close(core.A_log.exp(), expected)
    dt = torch.nn.functional.softplus(core.dt_bias)
    assert dt.min() >= .001 and dt.max() <= .1
    assert core.to_recipe()['model']['dt_init'] == 'random'
    mixer = HybridMambaSSM(cfg)
    assert mixer.dt_init == 'random'


@pytest.mark.parametrize('family,config_type,core_type,source,width', [
    # both families default to the block input
    ('hybrid_mamba', HybridMambaConfig, HybridMambaSSM, 'layer_input', 32),
    ('ssd_mimo', SSDMimoConfig, SSDMimoSSM, 'layer_input', 32),
])
def test_family_selection_default_agrees_across_construction_and_recipe(
        family, config_type, core_type, source, width):
    options = dict(n_heads=2)
    execution = {'backend': 'torch'}
    if family == 'hybrid_mamba':
        # bc_bias='none': the pre-spec HM default (no B/C bias)
        options.update(row_blocks=({'frac': 1., 'decay': 'coord_static'},), bc_bias='none')
    else:
        options.update(d_ssm_in=1, d_ssm_out=1)
    direct = core_type(config_type(d_model=32, d_state=16, d_inner=64, **options), execution=execution)
    typed = config_type(d_inner=64, d_state=16, d_model=32, **options).build(execution=execution)
    mixer = core_type(config_type(d_model=32, d_state=16, **options), execution=execution)
    model = CausalLM(vocab_size=16, layers=dict(type=family, d_state=16, **options), n_layers=1, d_model=32, execution=execution)
    drive, layer_input = torch.randn(1, 4, 64), torch.randn(1, 4, 32)
    for core in (direct, typed, mixer, model.backbone.layers[0].mixer):
        assert core.proj_source == source
        assert core.d_sel == width
        output = core.scan(drive, sel_x=layer_input)
        other_input = core.scan(drive, sel_x=torch.zeros_like(layer_input))
        if source == 'main_branch':
            torch.testing.assert_close(output, other_input, rtol=0, atol=0)
        else:
            assert not torch.allclose(output, other_input)
        document = json.loads(json.dumps(core.to_recipe()))
        restored = core_type.from_recipe(document)
        restored.load_state_dict(core.state_dict(), strict=True)
        assert restored.proj_source == source
        torch.testing.assert_close(restored.scan(drive, sel_x=layer_input),
                                   output)


def test_mh_a_log_ladder_restarts_per_block_and_counts_pairs():
    """Mamba-1's S4D-real ladder
    log(1..n) restarts in every A/mu block; a complex block's ladder runs over
    its PAIRS (one modulus per pair, one stored column); dyn rows store none."""
    core = HybridMambaSSM(HybridMambaConfig(d_model=32, d_state=48, n_heads=4, row_blocks=[
        dict(rows=16, decay='coord_static', complex=True), dict(rows=8, decay='coord_dynamic'),
        dict(rows=16, decay='coord_static'), dict(rows=8, decay='coord_potential', complex=True)],
        d_inner=64), execution={"backend": "torch"})
    want = torch.tensor(list(range(1, 9)) + list(range(1, 17)) + list(range(1, 5)),
                        dtype=torch.float32)
    torch.testing.assert_close(core.A_log.exp(), want.expand(4, 16, -1))


def test_mh_m1exact_init_matches_official_mamba1():
    """The Mamba-1 preset's init is official Mamba-1's: A_log exactly, D, the
    dt_proj bound (dt_rank^-1/2) and the dt bias range (inverse softplus of
    [1e-3, 0.1]); shapes 1:1."""
    mamba_simple = pytest.importorskip('mamba_ssm.modules.mamba_simple')
    from orbit_ssm.presets import mamba1_config
    d_model = 64
    torch.manual_seed(0)
    ours = HybridMambaSSM(mamba1_config(d_model), execution={'backend': 'torch'})
    off = mamba_simple.Mamba(d_model, device='cpu')
    assert tuple(ours.A_log.shape) == (off.A_log.shape[0], 1, off.A_log.shape[1])
    torch.testing.assert_close(ours.A_log[:, 0], off.A_log, rtol=0, atol=0)
    torch.testing.assert_close(ours.D, off.D, rtol=0, atol=0)
    assert ours.dt_proj.weight.shape == off.dt_proj.weight.shape
    bound = off.dt_proj.weight.shape[1] ** -.5
    assert ours.dt_proj.weight.abs().max() <= bound
    assert off.dt_proj.weight.abs().max() <= bound
    for b in (ours.dt_bias, off.dt_proj.bias):
        dt = torch.nn.functional.softplus(b.detach())
        assert dt.min() >= 1e-3 - 1e-9 and dt.max() <= .1 + 1e-9
