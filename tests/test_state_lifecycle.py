"""State API conformance: continuation, gradients and explicit lifecycle operations."""
import contextlib
import copy
import io
import json
import os
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from _tolerance import assert_close_terms, terms_abs

from orbit_ssm.layers.state import State
from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig
from orbit_ssm.ssm.hybrid_mamba import HybridMambaSSM, HybridMambaConfig
from orbit_ssm.model import ResidualBlock, BlockStack

from _design_warnings import SILENCE_SSD_DESIGN_WARNINGS

pytestmark = SILENCE_SSD_DESIGN_WARNINGS  # builds the warned configurations on purpose


# ---------------------------------------------------------------------------
# A7 helpers for module-level comparisons (also imported by
# other module-level test files).
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def plain_ops():
    """Run a reference through the plain torch ops (glue items off).

    The G33/H2 glue nodes are custom autograd Functions whose backward
    writes into fresh ``torch.empty`` buffers, which ``terms_abs`` cannot
    follow; with ``SSD_MIMO_GLUE=0`` / ``HM_GLUE=0`` the same function is
    computed by the plain ops the glue fuses (glue.py: "returns exactly the
    same values either way"), so sum|terms| is that of the same sums."""
    old = {k: os.environ.get(k) for k in ('SSD_MIMO_GLUE', 'HM_GLUE')}
    os.environ.update(SSD_MIMO_GLUE='0', HM_GLUE='0')
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def module_terms(fn, *args, module=None, grad_outputs=None):
    """``terms_abs`` of a module-level fp64 reference, through plain ops."""
    with plain_ops():
        return terms_abs(fn, *args, params=module, grad_outputs=grad_outputs)


_ARG_AMPLIFIED = {'exp', 'exp2', 'expm1', 'sin', 'cos', 'sigmoid', 'silu',
                  'softplus', 'tanh', 'log_sigmoid_forward'}


class _ArgumentScan(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.worst = 0.0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name.split('::')[-1].rstrip('_')
        x = args[0] if args else None
        if (name in _ARG_AMPLIFIED and torch.is_tensor(x)
                and x.is_floating_point() and x.numel()):
            v = x.detach().abs()
            if name.startswith('exp'):
                # exp(x) < u: the term is below one rounding of its sum
                v = torch.where(x.detach() >= -37.0, v, torch.zeros_like(v))
            v = v[torch.isfinite(v)]
            if v.numel():
                self.worst = max(self.worst, float(v.max()))
        return func(*args, **(kwargs or {}))


def argument_amplification(fn, *args, **kwargs):
    """A7 nonlinear amplification term: 1 + max |x| over every argument the
    reference hands to exp / sin / cos / sigmoid / silu / softplus / tanh.

    ``terms_abs`` counts a nonlinear function's OUTPUT as one term; the
    rounding its ARGUMENT carries is amplified by |x f'(x) / f(x)| <= |x| + 1
    (exp: |x|; sin/cos: an absolute |x|; the sigmoid family: <= |x| + 1),
    which belongs in c.  Measured on the reference's own arguments (the
    arithmetic), never on the observed error."""
    with _ArgumentScan() as scan:
        fn(*args, **kwargs)
    return 1.0 + scan.worst


def test_nested_state_transfers_preserve_phase_precision_and_gradients():
    hidden = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    phase = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    state = State(children=[State(hidden_state=hidden, rot_ang=phase,
                                 ticks=torch.arange(3))])
    moved = state.to(dtype=torch.bfloat16)
    child = moved['children'][0]
    assert isinstance(child, State)
    assert child.hidden_state.dtype == torch.bfloat16
    assert child['rot_ang'].dtype == torch.float32
    assert child['ticks'].dtype == torch.long
    selected = moved.clone().index_select(torch.tensor([2, 0, 2]))
    assert isinstance(selected['children'][0], State)
    loss = sum(selected['children'][0][k].sum() for k in ('hidden_state', 'rot_ang'))
    actual = torch.autograd.grad(loss, (hidden, phase))
    expected = torch.tensor([1., 0., 2.], dtype=torch.float64)[:, None].expand_as(hidden)
    for grad in actual:
        torch.testing.assert_close(grad, expected)
    detached = moved.detach()['children'][0]
    assert isinstance(detached, State)
    assert detached.hidden_state.grad_fn is None
    assert detached['rot_ang'].grad_fn is None


@pytest.mark.parametrize('operation', ['forward', 'step', 'reset'])
def test_later_invalid_model_state_refuses_before_any_layer_runs(operation, monkeypatch):
    from orbit_ssm.model import CausalLM
    model = CausalLM(vocab_size=16, layers=dict(type="ssd_mimo", d_state=8, expand=2, d_ssm_in=2, d_head=8), n_layers=2, d_model=8, execution={'backend': 'torch'})
    states = model.init_state(2)
    states[1]['hidden_state'] = states[1]['hidden_state'][:1]
    first = model.backbone.layers[0]
    def unexpected(*args, **kwargs):
        raise AssertionError('earlier layer ran before validating all caches')
    monkeypatch.setattr(first, 'reset_state' if operation == 'reset' else operation, unexpected)
    with pytest.raises(ValueError, match='shape'):
        if operation == 'reset':
            model.reset_state(states)
        elif operation == 'step':
            model.step(torch.ones(2, 1, dtype=torch.long), state=states)
        else:
            model(torch.ones(2, 2, dtype=torch.long), state=states)


def test_model_partial_state_defaults_and_batch_reselection():
    from orbit_ssm.model import CausalLM
    model = CausalLM(vocab_size=16, layers=dict(type="ssd_mimo", d_state=8, expand=2, d_ssm_in=2, d_head=8), n_layers=2, d_model=8, execution={'backend': 'torch'}).double()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    _, cache = model(ids[:, :2], return_state=True)
    indices = torch.tensor([1, 0, 1])
    selected = [s.index_select(indices) for s in cache]
    y, _ = model(ids.index_select(0, indices)[:, 2:], state=selected, return_state=True)
    reference, _ = model(ids[:, 2:], state=cache, return_state=True)
    torch.testing.assert_close(y, reference.index_select(0, indices))
    partial, _ = model(ids, state=[None, model.init_state(2, dtype=torch.float64)[1]],
                       return_state=True)
    torch.testing.assert_close(partial, model(ids))


@pytest.mark.parametrize('output_rank', [1, 2])
def test_ssd_model_split_all_gradients_and_checkpoint_continuation(output_rank):
    """Exercise rotated MIMO + conv + MLP at model level, including saved caches.

    Existing lifecycle tests cover core initial-state gradients; this checks
    all trainable model parameters and auxiliary terminal losses together.
    """
    from orbit_ssm.model import CausalLM
    torch.manual_seed(42)
    model = CausalLM(vocab_size=16, layers=dict(type="ssd_mimo", d_state=8, d_inner=16, d_conv=3, d_ssm_in=2, d_ssm_out=output_rank, d_head=8, row_blocks=[{'frac': 1.0, 'decay': 'head_dynamic', 'complex': True}]), n_layers=2, d_model=8, d_mlp=12, execution={'backend': 'torch'}).double()
    ids = torch.tensor([[1, 3, 2, 5, 7], [4, 2, 6, 1, 3]])
    states = model.init_state(2, dtype=torch.float64)
    seeds, slots = [], []
    for i, state in enumerate(states):
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                state[key] = (torch.randn_like(value) * .03).requires_grad_()
                seeds.append(state[key])
                slots.append((i, key))
    before = [s.clone() for s in states]
    full, final = model(ids, state=states, return_state=True)
    first, middle = model(ids[:, :2], state=states, return_state=True)
    rest, end = model(ids[:, 2:], state=middle, return_state=True)
    joined = torch.cat((first, rest), dim=1)

    def whole(*leaves):
        st = [State(s) for s in states]
        for (i, key), leaf in zip(slots, leaves):
            st[i][key] = leaf
        return model(ids, state=st)

    # A7: u=fp64 (whole vs split continuation, both fp64, differ in where
    # the scan is cut); worst case per arm gamma_{n+m}, x2 for two arms:
    # n = the composed sums of the two layers (in-proj 8, conv 3, scan
    # L*N*R*O = 5*8*2*O, out-proj 16, MLP 12 + 16) + the tied head 8,
    # m = 8 factors per layer; plus the exp/cos argument amplification.
    res = module_terms(whole, *seeds, module=model)
    n = 2 * (8 + 3 + 5 * 8 * 2 * output_rank + 16 + 12 + 16) + 8
    c = 2 * (n + 2 * 8 + argument_amplification(whole, *seeds))
    assert_close_terms(joined, full, res.terms, 'fp64', c, name='logits')

    def loss(y, cache):
        return y.square().sum() + sum(
            v.square().sum() for s in cache for v in s.values()
            if torch.is_tensor(v) and v.is_floating_point())

    leaves = (*seeds, *model.parameters())
    gfull = torch.autograd.grad(loss(full, final), leaves, allow_unused=True)
    gsplit = torch.autograd.grad(loss(joined, end), leaves, allow_unused=True)
    for a, b in zip(gfull, gsplit):
        assert (a is None) == (b is None)
        if a is not None:
            # A7 NOT CONVERTED (reported: terms_abs through a
            # multi-layer LM backward is vacuous -- sum|terms| / |grad|
            # reaches 1e19 (one layer) to 1e73 (these two layers): the fully
            # expanded product of every layer's sums.  c*u*sum|terms| then
            # exceeds |grad| itself and the relative guard refuses every
            # unit.  Kept literal until a first-order (layer-local) terms
            # propagation exists in tests/_tolerance.py.
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
    for original, snapshot in zip(states, before):
        for key in original:
            torch.testing.assert_close(original[key], snapshot[key], rtol=0, atol=0)

    # Serialization is an explicit graph boundary, using plain cache dicts so
    # safe weights-only loading requires no custom State allowlist.
    buffer = io.BytesIO()
    torch.save(dict(recipe=json.loads(json.dumps(model.to_recipe())),
                    weights=model.state_dict(),
                    states=[s.detach().to_dict() for s in middle]), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = CausalLM.from_recipe(saved['recipe']).double()
    restored.load_state_dict(saved['weights'], strict=True)
    replay, replay_end = restored(ids[:, 2:], state=[State.from_dict(s) for s in saved['states']],
                                  return_state=True)
    torch.testing.assert_close(replay, rest, rtol=0, atol=0)
    for actual, expected in zip(replay_end, end):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def core(family, **kw):
    # the full layer (d_model == d_inner == 8, no drive conv) whose bare
    # recurrence -- `scan` / `scan_step` -- these tests drive
    ex = {'backend': 'torch'}
    if family == 'ssd':
        return SSDMimoSSM(SSDMimoConfig(d_model=8, d_state=32, d_inner=8, d_conv=0, d_ssm_in=2, n_groups=2, n_heads=2,
                          proj_source='main_branch',
                          row_blocks=[{'frac': 1.0, 'decay': 'head_dynamic', 'complex': True}],
                          bc_conv=False), execution=ex).double()
    rows = ([{'rows': 4, 'decay': 'coord_static', 'complex': True},
             {'rows': 16, 'decay': 'head_dynamic', 'complex': True},
             {'rows': 16, 'decay': 'head_static'}]
            if family == 'mixed' else [{'rows': 8, 'decay': 'coord_static'}])
    if family == 'mixed':
        # flat rows serve only the per-head bias and the fixed-frequency
        # rotation without read offset (one global rotation setting)
        kw = dict(bc_bias='head', learn_theta_freq=False, use_theta_h=False, **kw)
    else:
        kw = dict(bc_bias='none', **kw)     # the pre-spec HM default
    return HybridMambaSSM(HybridMambaConfig(d_model=8, d_state=sum(f['rows'] for f in rows), d_inner=8, d_conv=0,
                          n_heads=2, row_blocks=rows, proj_source='main_branch', 
                          learnable_h_init=True, **kw), execution=ex).double()


@pytest.mark.parametrize('family', ['ssd', 'mh', 'mixed'])
def test_hidden_shorthand_and_complete_state_bptt(family):
    torch.manual_seed(20)
    m = core(family)
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    h = torch.randn(2, *m.hidden_state_shape(), dtype=torch.float64, requires_grad=True)
    state = m.init_state(2, dtype=torch.float64, h_init=h)
    assert state['hidden_state'] is h
    assert not {'c3_S','c2_S','cf_S'}.intersection(state)
    y, final = m.scan(x, state=m.init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), return_state=True)
    y0, state = m.scan(x[:, :2], state=state, return_state=True)
    ys = [y0]
    for t in range(2, 4):
        yt, state = m.scan_step(x[:, t], state)
        ys.append(yt[:, None])
    torch.testing.assert_close(y, torch.cat(ys, dim=1))
    for key in final:
        torch.testing.assert_close(final[key], state[key])
    seed = torch.randn_like(h)
    g = torch.autograd.grad(final['hidden_state'], (h, x), seed, retain_graph=True)
    gg = torch.autograd.grad(state['hidden_state'], (h, x), seed)
    for a,b in zip(g,gg):
        torch.testing.assert_close(a,b)
    assert g[0].abs().sum() > 0


@pytest.mark.parametrize('family', ['ssd', 'mh', 'mixed'])
def test_complete_state_leaves_keep_gradients_and_are_not_mutated(family):
    m = core(family)
    state = State({k: torch.randn_like(v, requires_grad=True) for k,v in
                   m.init_state(2, dtype=torch.float64).items()})
    before = {k:v.detach().clone() for k,v in state.items()}
    x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    y, out = m.scan_step(x, state)
    loss = y.square().sum() + sum(v.square().sum() for v in out.values())
    grads = torch.autograd.grad(loss, tuple(state.values()), allow_unused=True)
    for key, g in zip(state, grads):
        assert g is not None and torch.isfinite(g).all(), key
        torch.testing.assert_close(state[key], before[key])
    assert out is not state


@pytest.mark.parametrize('family', ['ssd', 'mh', 'mixed'])
def test_overrides_validation_and_empty_forward(family):
    m = core(family)
    h = torch.randn(*m.hidden_state_shape(), dtype=torch.float64, requires_grad=True)
    state = m.init_state(2, dtype=torch.float64, h_init=h)
    assert state['hidden_state'].shape == (2, *h.shape)
    y, final = m.scan(torch.empty(2,0,8,dtype=torch.float64), state=state, return_state=True)
    assert y.shape == (2,0,m.out_width)
    torch.testing.assert_close(final['hidden_state'], state['hidden_state'])
    with pytest.raises(TypeError, match='h_init'):
        m.scan_step(torch.zeros(2,8,dtype=torch.float64), state, h_init=h)
    with pytest.raises(ValueError, match='unknown'):
        m.init_state(2, overrides={'typo': h})
    with pytest.raises(ValueError, match='not both'):
        m.init_state(2, h_init=h, overrides={'hidden_state':h})
    with pytest.raises(ValueError, match='shape'):
        m.init_state(2, h_init=torch.zeros(3))
    bad = dict(state, typo=h)
    with pytest.raises(ValueError, match='unknown'):
        m.scan_step(torch.zeros(2,8,dtype=torch.float64), bad)


@pytest.mark.parametrize('family', ['ssd', 'mh', 'mixed'])
def test_reset_selected_keys_rows_and_full_replacement(family):
    m = core(family)
    h = torch.randn(3,*m.hidden_state_shape(),dtype=torch.float64,requires_grad=True)
    state = m.init_state(3, dtype=torch.float64, h_init=h)
    untouched = {k:v for k,v in state.items() if k != 'hidden_state'}
    mask = torch.tensor([True, False, True])
    m.reset_state(state, mask, keys=['hidden_state'])
    for k,v in untouched.items():
        assert state[k] is v
    expected = m.init_state(3, dtype=torch.float64)['hidden_state']
    torch.testing.assert_close(state['hidden_state'][mask], expected[mask])
    torch.testing.assert_close(state['hidden_state'][~mask], h[~mask])
    g, = torch.autograd.grad(state['hidden_state'].sum(),h,retain_graph=True)
    torch.testing.assert_close(g[mask],torch.zeros_like(g[mask]))
    torch.testing.assert_close(g[~mask],torch.ones_like(g[~mask]))
    if getattr(m,'h_init_param',None) is not None:
        gp, = torch.autograd.grad(state['hidden_state'].sum(),m.h_init_param)
        torch.testing.assert_close(gp, torch.full_like(gp,2))
    destination = {'stale': torch.ones(1)}
    m.set_state(destination,state)
    assert destination.keys() == state.keys()
    assert destination['hidden_state'] is not state['hidden_state']
    before = dict(destination)
    with pytest.raises(ValueError):
        m.set_state(destination,dict(state, unexpected=torch.zeros(1)))
    assert all(destination[k] is v for k,v in before.items())


def test_state_batch_selection_and_integer_dtype():
    h = torch.randn(3,2,requires_grad=True)
    state = State(hidden_state=h, ticks=torch.arange(3), global_count=torch.tensor(7))
    selected = state.index_select(torch.tensor([2,0,2]))
    selected['hidden_state'].sum().backward()
    torch.testing.assert_close(h.grad,torch.tensor([[1.,1.],[0.,0.],[2.,2.]]))
    assert selected['global_count'].ndim == 0
    assert state.to(dtype=torch.float64)['ticks'].dtype == torch.long
    assert state.detach()['hidden_state'].grad_fn is None


@pytest.mark.parametrize('family', ['ssd', 'mh'])
def test_mixer_and_block_hidden_shorthand(family):
    kw = dict(d_model=4,d_state=32,expand=2,d_conv=3,
              n_heads=2,proj_source='main_branch')
    if family == 'ssd':
        kw.update(n_groups=2, d_ssm_in=2)
    layer, config = (SSDMimoSSM, SSDMimoConfig) if family == 'ssd' else (HybridMambaSSM, HybridMambaConfig)
    m = layer(config(**kw)).double()
    block = ResidualBlock(m,4).double()
    x = torch.randn(2,4,4,dtype=torch.float64)
    h = torch.randn(2,*m.hidden_state_shape(),dtype=torch.float64,requires_grad=True)
    y, end = block(x,state=State.from_hidden(block, h),return_state=True)
    state = block.init_state(2,dtype=torch.float64,h_init=h)
    ys=[]
    for xt in x.unbind(1):
        yt,state=block.step(xt,state)
        ys.append(yt)

    def whole(h):
        return block(x, state=State.from_hidden(block, h))

    # A7: u=fp64 (block forward vs token steps, both fp64, differ in the
    # scan's association); worst case per arm gamma_{n+m}, x2 for two arms:
    # n = norm 4 + in-proj 4 + conv 3 + scan L*N = 4*32 + out-proj 8,
    # m = 8 factors; plus the exp/cos argument amplification.
    res = module_terms(whole, h, module=block)
    c = 2 * (4 + 4 + 3 + 4 * 32 + 8 + 8 + argument_amplification(whole, h))
    assert_close_terms(torch.stack(ys, 1), y, res.terms, 'fp64', c,
                       name='block y')
    torch.testing.assert_close(end['hidden_state'],state['hidden_state'])
    y0,s0=m.step(x[:,0],state=m.init_state((x[:,0]).shape[0], h_init=h, device=(x[:,0]).device, dtype=next(m.parameters()).dtype))
    y1,s1=m.step(x[:,0],m.init_state(2,dtype=torch.float64,h_init=h))
    torch.testing.assert_close(y0,y1)


@pytest.mark.parametrize('family', ['ssd_mimo', 'hybrid_mamba'])
def test_lm_hidden_shorthand_continuation_and_gradients(family):
    from orbit_ssm.model import CausalLM
    kw = dict(n_heads=2, proj_source='main_branch')
    if family == 'ssd_mimo':
        kw.update(n_groups=2, d_ssm_in=2)
    model = CausalLM(vocab_size=11, layers=dict(type=family, d_state=32, expand=2, d_conv=3, **kw), n_layers=2, d_model=4).double()
    ids = torch.tensor([[1, 2, 3]])
    hidden = [torch.randn(1, *b.mixer.hidden_state_shape(),
                          dtype=torch.float64, requires_grad=True)
              for b in model.backbone.layers]
    full, final = model(ids, state=model.init_state((ids).shape[0], h_init=hidden, device=(ids).device, dtype=next(model.parameters()).dtype), return_state=True)
    torch.testing.assert_close(model(ids, state=model.init_state((ids).shape[0], h_init=hidden, device=(ids).device, dtype=next(model.parameters()).dtype)), full)
    first, state = model.step(ids[:, :1], state=model.init_state((ids[:, :1]).shape[0], h_init=hidden, device=(ids[:, :1]).device, dtype=next(model.parameters()).dtype))
    rest, state = model(ids[:, 1:], state=state, return_state=True)
    carried = torch.cat([first[:, None], rest], dim=1)

    def whole(*h):
        return model(ids, state=model.init_state(
            1, h_init=list(h), device=ids.device, dtype=torch.float64))

    # A7: u=fp64 (whole forward vs step + continuation, both fp64); worst
    # case per arm gamma_{n+m}, x2 for two arms: n = two layers of (in-proj
    # 4, conv 3, scan L*N = 3*32, out-proj 8) + the tied head 4, m = 8
    # factors per layer; plus the exp/cos argument amplification.
    res = module_terms(whole, *hidden, module=model)
    c = 2 * (2 * (4 + 3 + 3 * 32 + 8) + 4 + 2 * 8
             + argument_amplification(whole, *hidden))
    assert_close_terms(carried, full, res.terms, 'fp64', c, name='logits')
    for actual, expected in zip(state, final):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])
    g1 = torch.autograd.grad(full.square().sum(), hidden, retain_graph=True)
    g2 = torch.autograd.grad(carried.square().sum(), hidden)
    for actual, expected in zip(g1, g2):
        # A7 NOT CONVERTED (reported: terms_abs through a
        # multi-layer LM backward is vacuous -- sum|terms| / |grad| is
        # 4.5e14 median / 2.9e17 max for layer 0's hidden (ssd_mimo): the
        # fully expanded product of every layer's sums.  c*u*sum|terms|
        # then exceeds |grad| itself and the relative guard refuses.  Kept
        # literal until a first-order (layer-local) terms propagation exists
        # in tests/_tolerance.py.
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
        assert actual.abs().sum() > 0
    with pytest.raises(ValueError, match='exactly 2'):
        model(ids, state=model.init_state((ids).shape[0], h_init=hidden[:1], device=(ids).device, dtype=next(model.parameters()).dtype))
    with pytest.raises(TypeError, match='h_init'):
        model(ids, state=state, h_init=hidden)


def test_mixed_channel_major_hidden_return_is_complete():
    m = core('mixed')
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    h = torch.randn(2,*m.hidden_state_shape(),dtype=torch.float64,requires_grad=True)
    y, final = m.scan(x,state=m.init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), return_state=True)
    final = final.hidden_state
    yc, hc = m._forward_cf(x.transpose(1,2), h_init=h, want_state=True)
    torch.testing.assert_close(yc.transpose(1,2),y)
    torch.testing.assert_close(hc,final)
    assert hc.shape == h.shape
    y_only, _ = m._forward_cf(x.transpose(1,2), h_init=h, want_state=False)
    torch.testing.assert_close(y_only.transpose(1,2),y)
    with torch.no_grad():
        m.h_init_param.normal_(std=.1)
    default = m.scan(x)
    fast, _ = m._forward_cf(x.transpose(1,2), want_state=False)
    # A7: u=fp64 (public forward vs the channel-major route, both fp64);
    # worst case per arm gamma_{n+m}, x2 for two arms: n = in-proj 8 +
    # scan L*N = 3*36, m = 8 factors; plus the argument amplification.
    res = module_terms(lambda x: m.scan(x), x, module=m)
    c = 2 * (8 + 3 * m.d_state + 8 + argument_amplification(m.scan, x))
    assert_close_terms(fast.transpose(1, 2), default, res.terms, 'fp64', c,
                       name='y')
    assert torch.autograd.grad(hc.square().sum(),h)[0] is not None


@pytest.mark.parametrize('family', ['ssd', 'mixed'])
def test_phase_precision_survives_state_transfer(family):
    m = core(family)
    state = m.init_state(2, dtype=torch.bfloat16)
    phases = set(state).intersection({'rot_ang', 'theta', 'c3_phase'})
    assert phases
    for key in phases:
        assert state[key].dtype == torch.float32
    m.validate_state(state)
    converted = state.to(dtype=torch.float64).to(dtype=torch.float16)
    assert converted['hidden_state'].dtype == torch.float16
    for key in phases:
        assert converted[key].dtype == torch.float32
    m.validate_state(converted)
    first = next(iter(phases))
    converted[first] = converted[first].half()
    with pytest.raises(TypeError, match='at least float32'):
        m.validate_state(converted)


@pytest.mark.parametrize('a_act', ['heavy_tail', 'softplus'])
def test_mixed_rows_keep_public_coordinates_and_independent_gradients(a_act):
    m = core('mixed', a_act=a_act)
    with torch.no_grad():
        for parameter in m.parameters():
            parameter.zero_()
    h = torch.arange(1, m.d_state + 1, dtype=torch.float64).expand(2, m.H, m.P, -1).clone().requires_grad_()
    x = torch.zeros(2, 8, dtype=torch.float64)
    _, state = m.scan_step(x, state=m.init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype))
    _, state = m.scan_step(x, state)
    dt = torch.log(torch.tensor(2., dtype=h.dtype))
    # m3 rows: |A| = f(0) with every weight zero -- heavy_tail(0) = 1 (the
    # default), softplus(0) = log 2 = dt -- on the clock dt
    rate0 = torch.ones((), dtype=h.dtype) if a_act == 'heavy_tail' else dt
    factors = torch.cat([torch.full((4,), .5, dtype=h.dtype),
                         torch.exp(-rate0 * dt).expand(16),
                         torch.full((16,), .5, dtype=h.dtype)])
    torch.testing.assert_close(state['hidden_state'], h * factors.square())
    grad, = torch.autograd.grad(state['hidden_state'].sum(),h)
    torch.testing.assert_close(grad, factors.square().expand_as(h))


def test_mixed_step_does_not_call_sequence_oracles(monkeypatch):
    import orbit_ssm.ssm.hybrid_mamba as hm
    def forbidden(*args, **kwargs):
        raise AssertionError('step must use its direct recurrence')
    monkeypatch.setattr(hm, '_m3_reference', forbidden)
    monkeypatch.setattr(hm, '_flat_reference', forbidden)
    m = core('mixed').float()
    y, state = m.scan_step(torch.randn(2,8))
    assert y.dtype == state['hidden_state'].dtype == torch.float32
    assert all(t.dtype != torch.float64 for t in state.values())


def test_mixed_direct_steps_match_scan_parameter_gradients():
    torch.manual_seed(47)
    m = core('mixed')
    with torch.no_grad():
        m.theta_proj.weight.normal_(std=.1)
        m.m3_angle_proj.weight.normal_(std=.1)
    x = torch.randn(2,4,8,dtype=torch.float64,requires_grad=True)
    scan,_ = m._forward_cf(x.transpose(1,2),want_state=False)
    direct = m.scan(x)
    dy=torch.randn_like(direct)
    # A7: u=fp64 (channel-major scan vs direct steps, both fp64); worst
    # case per arm gamma_{n+m}, x2 for two arms: n = in-proj 8 + scan
    # L*N = 4*36 (forward), + the B*L token reduction of a parameter
    # gradient 2*4 and its output width 8; m = 8 factors; plus the
    # exp/cos argument amplification.
    res = module_terms(lambda x: m.scan(x), x, module=m, grad_outputs=dy)
    amp = argument_amplification(m.scan, x)
    assert_close_terms(scan.transpose(1, 2), direct, res.terms, 'fp64',
                       2 * (8 + 4 * m.d_state + 8 + amp), name='y')
    c_grad = 2 * (8 + 4 * m.d_state + 2 * 4 + 8 + 8 + amp)
    leaves=(x,*m.parameters())
    expected=torch.autograd.grad((scan.transpose(1,2)*dy).sum(),leaves,allow_unused=True)
    actual=torch.autograd.grad((direct*dy).sum(),leaves,allow_unused=True)
    terms = [res.grad_terms[0], *(res.param_grad_terms.get(n)
                                  for n, _ in m.named_parameters())]
    for name,a,b,t in zip(['x',*dict(m.named_parameters())],actual,expected,terms):
        assert (a is None)==(b is None), name
        if a is not None:
            assert_close_terms(a, b, t, 'fp64', c_grad, name=name)


@pytest.mark.parametrize('family', ['ssd', 'mh', 'mixed'])
def test_state_from_hidden_preserves_network_gradient_and_complete_schema(family):
    m = core(family)
    producer = torch.nn.Linear(3, m.empty_state(1).numel()).double()
    source = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    hidden = producer(source).reshape(2, *m.hidden_state_shape())
    state = State.from_hidden(m, hidden)
    assert state.hidden_state is hidden
    assert state.keys() == m.init_state(2).keys()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    y, final = m.scan(x, state=state, return_state=True)
    assert isinstance(final, State)
    assert torch.is_tensor(m.scan(x))
    grads = torch.autograd.grad(y.square().sum() + final.hidden_state.square().sum(),
                                (source, producer.weight))
    assert all(g.isfinite().all() and g.abs().sum() > 0 for g in grads)
    with pytest.raises(TypeError, match='h_init'):
        m.scan(x, h_init=hidden)
    with pytest.raises(TypeError):
        m.scan(x, hidden)
    with pytest.raises(ValueError, match='batch_size'):
        State.from_hidden(m, hidden[0])
    shared = State.from_hidden(m, hidden[0], batch_size=3)
    assert shared.hidden_state.shape[0] == 3
    with pytest.raises(ValueError, match='shape'):
        m.scan(x, state=shared)


@pytest.mark.parametrize('mixer', [SSDMimoSSM, HybridMambaSSM])
def test_generic_mixer_output_normalization_defaults_off(mixer):
    from orbit_ssm.presets import mamba2_config
    m = mixer({SSDMimoSSM: SSDMimoConfig, HybridMambaSSM: HybridMambaConfig}[mixer](d_model=32, d_state=16, d_head=8))
    assert isinstance(m.norm, torch.nn.Identity)
    assert m.backend == 'auto'
    m2 = SSDMimoSSM(mamba2_config(64, d_state=16))
    assert not isinstance(m2.norm, torch.nn.Identity)
