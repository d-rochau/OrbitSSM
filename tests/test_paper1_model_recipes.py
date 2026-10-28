"""Paper 1 model construction, fresh resume, and mixer-boundary gradients."""
import copy
import io
import json
import math

import pytest
import torch

from orbit_ssm.model import CausalLM
from orbit_ssm.ssm.hybrid_mamba import HybridMambaSSM, HybridMambaConfig
from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig

from _tolerance import assert_close_terms
from test_state_lifecycle import argument_amplification, module_terms

pytestmark = [pytest.mark.paper1, pytest.mark.contract]
MIXERS = [HybridMambaSSM, SSDMimoSSM]


def mixer_options(cls, **overrides):
    core = dict(n_heads=2, execution={'backend': 'torch'}, proj_source='main_branch',
                out_norm=False)
    # Both families' selector is the named `write_rule`.
    if cls is SSDMimoSSM:
        core.update(d_ssm_in=2, d_ssm_out=4, r_mix_dyn=True)
    core.update(write_rule='trapezoid')
    return dict(d_model=4, expand=2, d_state=32, d_conv=3,
                **core, **overrides)


def build_mixer(cls, **overrides):
    """The layer `mixer_options` describes: its family config plus `execution=`."""
    options = mixer_options(cls, **overrides)
    execution = options.pop('execution')
    return cls({SSDMimoSSM: SSDMimoConfig, HybridMambaSSM: HybridMambaConfig}[cls](**options), execution=execution)


@pytest.mark.parametrize('cls', MIXERS)
@pytest.mark.parametrize('device,dtype', [('cpu', torch.float64),
    pytest.param('cuda', torch.float32, marks=pytest.mark.gpu)])
def test_mixer_chunking_preserves_all_gradients_and_bounded_inference_cache(cls, monkeypatch, device, dtype):
    monkeypatch.setenv('HM_GLUE', 'all')
    if device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('requires supported training GPU')
    torch.manual_seed(18)
    full = build_mixer(cls).to(device=device, dtype=dtype)
    chunked = copy.deepcopy(full)
    ref = copy.deepcopy(full).double()
    x = torch.randn(2, 7, 4, device=device, dtype=dtype, requires_grad=True)
    xc = x.detach().clone().requires_grad_()
    y, final = full(x, return_state=True)
    y0, state = chunked(xc[:, :2], return_state=True)
    y1, actual = chunked(xc[:, 2:], state=state, return_state=True)

    # A7: whole vs chunked continuation of the same torch route, both in
    # `dtype` -> u = dtype (fp64 on CPU, fp32 on the GPU cell), two arms
    # rounding independently.  n = in-proj 4 + conv 3 + the scan
    # L*N*R*O = 7*32*2*4 + out-proj 8 (+ the B*L = 14 token reduction of a
    # parameter gradient), m = 8 factors: n + m ~ 1.8e3, so the typical
    # scale sqrt(2 (n + m)) (the terms bound already carries the sqrt(n)
    # pessimism, tests/_tolerance.py); plus the exp/cos argument
    # amplification.  sum|terms| from the fp64 reference of the same mixer.

    def forward(x):
        y, st = ref(x, return_state=True)
        return y, {k: v for k, v in st.items() if torch.is_tensor(v)}

    def loss_fn(x):
        y, st = ref(x, return_state=True)
        return y.square().sum() + sum(v.square().sum() for v in st.values())

    xd = x.detach().double().requires_grad_()
    fwd = module_terms(forward, xd, module=ref)
    grad = module_terms(loss_fn, xd, module=ref)
    n = 4 + 3 + 7 * 32 * 2 * 4 + 8
    amp = argument_amplification(forward, xd)
    c = math.sqrt(2 * (n + 8)) + amp
    c_grad = math.sqrt(2 * (n + 14 + 8)) + amp
    assert_close_terms(torch.cat([y0, y1], 1), y, fwd.terms[0], dtype, c,
                       name='y')
    for k in final:
        assert_close_terms(actual[k], final[k], fwd.terms[1][k], dtype, c,
                           name=k)
    loss = y.square().sum() + sum(v.square().sum() for v in final.values())
    loss_c = y0.square().sum()+y1.square().sum()+sum(v.square().sum() for v in actual.values())
    gs = torch.autograd.grad(loss, (x, *full.parameters()), allow_unused=True)
    gc = torch.autograd.grad(loss_c, (xc, *chunked.parameters()), allow_unused=True)
    terms = [grad.grad_terms[0], *(grad.param_grad_terms.get(n)
                                   for n, _ in full.named_parameters())]
    for name, g, c_, t in zip(['input', *dict(full.named_parameters())], gs, gc, terms):
        assert (g is None) == (c_ is None), name
        if g is not None:
            assert_close_terms(c_, g, t, dtype, c_grad, name=name)
    with torch.no_grad():
        _, cache = full(x.detach(), return_state=True)
    history = cache['conv_state']
    assert history.untyped_storage().nbytes() == history.numel()*history.element_size()


@pytest.mark.parametrize('cls', MIXERS)
def test_mixer_recipe_roundtrip_and_nested_settings(cls):
    # API unification: flat `d_conv` is the shared head, so the
    # core's d_conv equals the mixer's (a distinct core value no longer
    # round-trips: constructor_kwargs refuses the conflict).
    original = build_mixer(cls).double()
    recipe = json.loads(json.dumps(original.to_recipe()))
    assert recipe['model']['d_conv'] == 3
    assert 'core' not in recipe['model'] and 'core_execution' not in recipe['execution']   # one flat section
    restored = cls.from_recipe(recipe).double()
    restored.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 4, 4, dtype=torch.float64)
    torch.testing.assert_close(original(x), restored(x), rtol=0, atol=0)
    for section, key in [('model', 'out_norm'), ('execution', 'kernel_options')]:
        bad = copy.deepcopy(recipe)
        del bad[section][key]
        with pytest.raises((ValueError, KeyError)):
            cls.from_recipe(bad)
    bad = copy.deepcopy(recipe)
    del bad['model']['write_rule']
    with pytest.raises(ValueError, match='do not match the config schema'):   # a missing field is refused
        cls.from_recipe(bad)


@pytest.mark.parametrize('cls,cfg_cls', [(HybridMambaSSM, HybridMambaConfig),
                                       (SSDMimoSSM, SSDMimoConfig)])
def test_typed_and_dict_core_options_use_one_mapping(cls, cfg_cls):
    import dataclasses
    cfg = cfg_cls(d_inner=8, d_model=4, d_state=32, d_conv=2,
                  n_heads=2, proj_source='main_branch')
    ex = {'backend': 'torch'}
    # the config holds the architecture, including the one conv width; the layer takes it alone
    with pytest.raises(TypeError):
        cls(cfg, d_conv=3)
    fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    typed = cls(cfg, execution=ex).to_recipe()
    assert cls(fields, execution=ex).to_recipe() == typed                  # the strict dict is the same input
    assert cls(config=cfg, execution=ex).to_recipe() == typed
    assert cfg.build(4, execution=ex).to_recipe() == typed                 # build() at the config's own width
    assert cls(cfg.replace(d_conv=3), execution=ex).to_recipe()['model']['d_conv'] == 3
    assert typed['model']['d_conv'] == 2 == cls(cfg, execution=ex).d_conv
    with pytest.raises(ValueError, match='d_model'):
        cfg.build(8)                        # the config names its width: a conflict raises
    assert cfg.replace(d_model=None).build(8, execution=ex).d_model == 8


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('families', [('hybrid_mamba',), ('ssd_mimo',),
                                     ('hybrid_mamba', 'ssd_mimo')])
def test_full_model_json_weights_and_adamw_resume(families, tied):
    torch.manual_seed(121)
    configs = []
    for family in families:
        cls = HybridMambaSSM if family == 'hybrid_mamba' else SSDMimoSSM
        options = mixer_options(cls)
        options.pop('d_model')
        execution = options.pop('execution')
        configs.append(dict(type=family, **options))
    model = CausalLM(vocab_size=11, layers=configs, d_model=4, d_mlp=8, tie_word_embeddings=tied, execution=execution).double()
    opt = torch.optim.AdamW(model.parameters(), lr=.002, foreach=False)
    tokens = [torch.randint(0, 11, (2, 4)) for _ in range(2)]

    def update(m, optimizer, x):
        optimizer.zero_grad(set_to_none=True)
        loss = m(x).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    update(model, opt, tokens[0])
    recipe = json.loads(json.dumps(model.to_recipe()))
    buffer = io.BytesIO()
    torch.save(dict(recipe=recipe, weights=model.state_dict(), optimizer=opt.state_dict()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = CausalLM.from_recipe(saved['recipe']).double()
    restored.load_state_dict(saved['weights'], strict=True)
    assert restored.to_recipe() == recipe
    assert (restored.embedding.weight is restored.lm_head.weight) == tied
    assert list(dict(model.named_parameters())) == list(dict(restored.named_parameters()))
    resumed = torch.optim.AdamW(restored.parameters(), lr=.002, foreach=False)
    resumed.load_state_dict(saved['optimizer'])
    torch.testing.assert_close(update(model, opt, tokens[1]), update(restored, resumed, tokens[1]), rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    for key, state in opt.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed.state_dict()['state'][key][name], rtol=0, atol=0)


@pytest.mark.parametrize('layers', [[{'type': 'hybrid_mamba', 'd_sttae': 32}], [{}, {}],
                                    [{'type': 'ssd_mimo', 'gate': 'typo'}]])
def test_model_refuses_unused_or_unknown_layer_options(layers):
    # unknown names are refused by the config's strict dict form (TypeError); a dict without a type likewise
    with pytest.raises(TypeError):
        CausalLM(vocab_size=8, layers=layers, d_model=4)
