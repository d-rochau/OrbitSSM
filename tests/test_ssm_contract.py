"""Regression checks for differentiable boundaries and resolved SSD options."""
import copy

import pytest
import torch

from _tolerance import assert_close_terms, terms_abs
from orbit_ssm.layers.state import ScanState
from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig
from orbit_ssm.ssm.hybrid_mamba import HybridMambaSSM
from orbit_ssm import HybridMambaConfig

from _design_warnings import SILENCE_SSD_DESIGN_WARNINGS

pytestmark = SILENCE_SSD_DESIGN_WARNINGS  # builds the warned configurations on purpose

#: the old rows=ROT (rope_fraction 1.0): the whole state is one complex block
ROT = [{'frac': 1.0, 'decay': 'head_dynamic', 'complex': True}]
#: the HM flat laws serve only the per-head bias; the per-coordinate laws
#: keep the pre-spec no-bias default
HM_FLAT = ('head_dynamic', 'head_static')


def _c_ssd(L, N, d_sel, R=1, conv=0, sfu=0):
    """A7 worst-case rounding depth gamma_{n+m}/u of ONE arm of an SSD /
    HM mixer core, per output entry, stage by stage (the relative errors of
    chained stages add; stages both arms share are counted anyway):
      token maps (B, C, dt, theta, trap, gates): d_sel-dot + 1   d_sel + 1
      optional depthwise B|C conv: taps + 1, SiLU argument <= 3  conv + 4
      B/C RMSNorm over N: N-term mean + eps, sqrt, div, weight      N + 4
      biases / residual gates (1 + g)                                  2
      softplus clock and exp decay argument (<= 2)                     3
      the scan: n = 2 (L + 1) summands (the trapezoid's two taps),
        m = 3 L + 3 factors (a 2-term pair rotation and the decay per
        token; u, B, dt), + L for the exp(cumsum) / cos,sin(cumulative
        angle) arguments (per-token log-decay and angle <= 1)      6 L + 5
      R sibling writes summed, readout N-dot + 1                 R + N + 1
      D skip + output gate                                             3
      sfu: a kernel's cos.approx/sin.approx (absolute 2^-20.9 = 8.6 u on
        a unit pair, + the 2 pi wrap) per term                       sfu
    """
    if conv:
        conv = conv + 4
    return (d_sel + 1) + conv + (N + 4) + 2 + 3 + (6 * L + 5) + (R + N + 1) + 3 + sfu


def _c_ssd_grad(c_fwd, reduction, fan_out):
    """Backward of a `_c_ssd` forward: the forward values it reuses, the
    adjoint pass through the same stages, and a parameter gradient's
    reduction over the B*L token positions (and heads) and the widest
    fan-out of a projection row."""
    return 2 * c_fwd + reduction + fan_out


def _grad_terms(res, n_inputs, names, i):
    """sum|terms| of gradient i of (inputs..., parameters...) from a
    `terms_abs(..., params=module)` result; 0 for an unused one."""
    g = res.grad_terms[i] if i < n_inputs else res.param_grad_terms.get(names[i - n_inputs])
    return 0.0 if g is None else g


def rows_for(rot):
    return [{'frac': 1.0, 'decay': 'head_dynamic', 'complex': rot == 'm3'}]


@pytest.mark.parametrize('layout', ['interleaved', 'half'])
@pytest.mark.parametrize('rank,outputs', [(1, 1), (2, 2), (2, 1)])
def test_ssd_hidden_boundary_and_step_layout(layout, rank, outputs):
    torch.manual_seed(41)
    # the full layer; the cell drives its bare recurrence, `scan` / `scan_step`
    m = SSDMimoSSM(SSDMimoConfig(d_model=4 * rank, d_state=32, d_inner=4 * rank, d_ssm_in=rank, d_ssm_out=outputs,
                   n_groups=2, n_heads=2, proj_source='main_branch', bc_conv=False,
                   row_blocks=ROT, pair_layout=layout), execution={'backend': 'torch', 'precision': 'ieee'}).double()
    with torch.no_grad():
        m.theta_proj.weight.uniform_(-.4, .4)
        m.theta_h.uniform_(-.3, .3)
    ref = copy.deepcopy(m)
    ref.set_execution(dict(backend='sequential'))
    inputs = [torch.randn(1, 5, m.d_inner, dtype=torch.float64, requires_grad=True),
              torch.randn(1, m.H, m.P, m.N, dtype=torch.float64, requires_grad=True)]
    ri = [v.detach().clone().requires_grad_() for v in inputs]
    y, h = m.scan(inputs[0], state=ScanState(m).init_state((inputs[0]).shape[0], h_init=inputs[1], device=(inputs[0]).device, dtype=next(m.parameters()).dtype), return_state=True)
    h = h.hidden_state
    yr, hr = ref.scan(ri[0], state=ScanState(ref).init_state((ri[0]).shape[0], h_init=ri[1], device=(ri[0]).device, dtype=next(ref.parameters()).dtype), return_state=True)
    hr = hr.hidden_state

    def route(mod):
        def run(x, h0):
            yy, st = mod.scan(x, state=ScanState(mod).init_state(x.shape[0], h_init=h0, device=x.device,
                                                 dtype=x.dtype), return_state=True)
            hh = st.hidden_state
            return yy, hh, hh.square().sum()
        return run

    one = torch.ones((), dtype=torch.float64)
    # sum|terms| of both backends (each rounds on its own summands); the
    # scalar hidden-only loss is the third output
    t_m = terms_abs(route(m), *ri, params=m, grad_outputs=(None, None, one))
    t_r = terms_abs(route(ref), *ri, params=ref, grad_outputs=(None, None, one))
    # A7: u=fp64 (torch vs sequential backend, both fp64), bound gamma(torch)
    # + gamma(sequential); c = _c_ssd(L=5, N=32, d_sel=d_inner, R).
    L = inputs[0].shape[1]
    c_fwd = _c_ssd(L, m.N, m.d_inner, rank)
    assert_close_terms(y, yr, t_m.terms[0] + t_r.terms[0], "fp64", c_fwd, name="y")
    assert_close_terms(h, hr, t_m.terms[1] + t_r.terms[1], "fp64", c_fwd, name="h")
    # Final-hidden-only loss must reach inputs, initial state and parameters.
    g = torch.autograd.grad(h.square().sum(), (*inputs, *m.parameters()), allow_unused=True)
    gr = torch.autograd.grad(hr.square().sum(), (*ri, *ref.parameters()), allow_unused=True)
    names = [n for n, _ in m.named_parameters()]
    for i, (a, b) in enumerate(zip(g, gr)):
        assert (a is None) == (b is None)
        if a is not None:
            # A7: u=fp64, both backends' terms; c = _c_ssd_grad: reduction
            # B*L*H = 10 token-heads, fan-out d_inner; +2 for the squared loss.
            assert_close_terms(a, b, _grad_terms(t_m, 2, names, i)
                               + _grad_terms(t_r, 2, names, i), "fp64",
                               _c_ssd_grad(c_fwd, L * m.H, m.d_inner) + 2,
                               name=f"grad {i}")
    assert inputs[1].shape == ScanState(ref).init_state(1)['hidden_state'].shape


def test_ssd_config_and_strict_loading():
    cfg = SSDMimoConfig(d_inner=8, d_state=32, n_groups=2, n_heads=2, d_ssm_in=2,
                        pair_layout='half', row_blocks=ROT, proj_source='main_branch', bc_conv=False,
                        a_dyn_per_sibling=True)
    ex = {'backend': 'torch', 'kernel_options': {'allow_chunk_autotune': False}}
    a = cfg.build(execution=ex)
    b = SSDMimoSSM(SSDMimoConfig(**{**cfg.layer_kwargs(), 'd_model': 8}), execution=ex)
    assert a.R == b.R == a.O == b.O == 2
    assert a.pair_layout == b.pair_layout == 'half'
    assert a.a_dyn_per_sibling and b.a_dyn_per_sibling
    b.load_state_dict(a.state_dict(), strict=True)
    kw = dict(d_model=4, d_state=8, d_inner=4, d_ssm_in=1, n_groups=2, n_heads=2, proj_source="main_branch")
    learned = SSDMimoSSM(SSDMimoConfig(**kw, r_mix='learned'), execution={'backend': 'torch'})
    plain = SSDMimoSSM(SSDMimoConfig(**kw, r_mix='none'), execution={'backend': 'torch'})
    with pytest.raises(RuntimeError, match='Unexpected key'):
        plain.load_state_dict(learned.state_dict(), strict=True)


def test_mh_hidden_init_with_complete_state_return():
    m = HybridMambaSSM(HybridMambaConfig(d_model=8, d_state=8, d_inner=8, n_heads=2, proj_source='main_branch',
                       row_blocks=[{'frac': 1.0, 'decay': 'coord_static'}], bc_bias='none'), execution={'backend': 'sequential'}).double()
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    state = ScanState(m).init_state(1, dtype=torch.float64)
    h = torch.randn_like(state['hidden_state'], requires_grad=True)
    state['hidden_state'] = h
    y, final = m.scan(x, state=ScanState(m).init_state((x).shape[0], h_init=h, device=(x).device, dtype=next(m.parameters()).dtype), return_state=True)
    yr, expected = m.scan(x, state=state, return_state=True)
    torch.testing.assert_close(y, yr)
    torch.testing.assert_close(final['hidden_state'], expected['hidden_state'])
    g, = torch.autograd.grad(final['hidden_state'].square().sum(), h)
    assert g.abs().sum() > 0

@pytest.mark.gpu
@pytest.mark.parametrize('layout', ['interleaved', 'half'])
def test_cuda_final_hidden_adjoint(require_cuda, layout):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("installed Triton supports NVIDIA compute capability 8.0 or newer")
    torch.manual_seed(52)
    m = SSDMimoSSM(SSDMimoConfig(d_model=128, d_state=32, d_inner=128, d_ssm_in=2, d_ssm_out=2, n_groups=32, n_heads=2,
                   proj_source='main_branch', bc_conv=False, row_blocks=ROT,
                   pair_layout=layout), execution={'backend': 'cuda', 'precision': 'ieee'}).cuda()
    ref = copy.deepcopy(m)
    ref.set_execution(dict(backend='torch'))
    # sum|terms| from the torch route run in fp64 on CPU (copied before any
    # forward leaves non-leaf caches on the module)
    ref64 = copy.deepcopy(ref).double().cpu()
    x = torch.randn(1, 35, 128, device='cuda', requires_grad=True)
    h0 = torch.randn(1, m.H, m.P, m.N, device='cuda', requires_grad=True)
    xr, h0r = [v.detach().clone().requires_grad_() for v in (x, h0)]
    _, h = m.scan(x, state=ScanState(m).init_state((x).shape[0], h_init=h0, device=(x).device, dtype=next(m.parameters()).dtype), return_state=True)
    h = h.hidden_state
    _, hr = ref.scan(xr, state=ScanState(ref).init_state((xr).shape[0], h_init=h0r, device=(xr).device, dtype=next(ref.parameters()).dtype), return_state=True)
    hr = hr.hidden_state
    covector = torch.randn_like(h)

    def hidden64(xv, hv):
        _, st = ref64.scan(xv, state=ScanState(ref64).init_state(xv.shape[0], h_init=hv, device=xv.device,
                                                 dtype=xv.dtype), return_state=True)
        return st.hidden_state

    t64 = terms_abs(hidden64, xr.detach().cpu().requires_grad_(),
                    h0r.detach().cpu().requires_grad_(), params=ref64,
                    grad_outputs=covector.cpu())
    # A7: u=fp32 (the fp32 IEEE CUDA route vs the fp32 torch route: the same
    # summands, reassociated -- chunked scan, split reductions, atomics -- so
    # the torch terms are both arms' terms and the bound is gamma(cuda) +
    # gamma(torch) = 2 c u terms).  c = _c_ssd(L=35, N=32, d_sel=128, R=2,
    # sfu=9: the kernel's SFU cos/sin per term).
    c_fwd = _c_ssd(35, m.N, 128, 2, sfu=9)
    assert_close_terms(h, hr, t64.terms, "fp32", 2 * c_fwd, name="h")
    g = torch.autograd.grad(h, (x, h0, *m.parameters()), covector, allow_unused=True)
    gr = torch.autograd.grad(hr, (xr, h0r, *ref.parameters()), covector, allow_unused=True)
    names = [n for n, _ in m.named_parameters()]
    for i, (a, b) in enumerate(zip(g, gr)):
        if a is None or b is None:
            # A joint CUDA output/state node can materialize zero adjoints for
            # readout-only inputs. The hidden-only Torch graph omits those
            # edges. Require exact zero, never ignore a nonzero gradient.
            present = b if a is None else a
            if present is not None:
                torch.testing.assert_close(present, torch.zeros_like(present), rtol=0, atol=0)
        else:
            # A7: u=fp32, both arms (2x); c = _c_ssd_grad: reduction B*L*H =
            # 70 token-heads, fan-out 128 (d_sel of a projection column).
            assert_close_terms(a, b, _grad_terms(t64, 2, names, i), "fp32",
                               2 * _c_ssd_grad(c_fwd, 35 * m.H, 128),
                               name=f"grad {i} ({(['x', 'h0'] + names)[i]})")

@pytest.mark.parametrize('init', ['random', 'constant'])
def test_mh_mamba1_clock_initialization(init):
    from orbit_ssm.ssm.hybrid_mamba import HybridMambaConfig
    cfg = HybridMambaConfig(d_inner=8, d_state=8, d_model=33, n_heads=2,
                            proj_source='main_branch', dt_rank='auto', dt_init=init,
                            dt_scale=0.7, dt_min=0.001, dt_max=0.002,
                            dt_init_floor=0.003)
    m = cfg.build(execution={'backend': 'sequential'})
    assert m.dt_rank == 3
    bound = 0.7 / 3**0.5
    if init == 'constant':
        torch.testing.assert_close(m.dt_proj.weight, torch.full_like(m.dt_proj.weight, bound))
    else:
        assert m.dt_proj.weight.abs().max() <= bound
        assert m.dt_proj.weight.std() > 0
    torch.testing.assert_close(torch.nn.functional.softplus(m.dt_bias),
                               torch.full_like(m.dt_bias, 0.003))


def test_uncapped_clock_tail_config():
    from orbit_ssm.layers.tails import TailCfg
    from orbit_ssm.ssm.ssd_mimo import _dt_tail_eager
    assert not {'dt_cap', 'dt_beta', 'clamp'}.intersection(TailCfg._fields)
    x = torch.tensor([-20., 0., 20., 100.], requires_grad=True)
    actual = _dt_tail_eager(x)
    torch.testing.assert_close(actual, torch.nn.functional.softplus(x))
    actual.sum().backward()
    assert x.grad[-1] == 1

@pytest.mark.parametrize('source', ['drive', 'residual'])
def test_retired_projection_aliases_are_rejected(source):
    with pytest.raises(ValueError, match='proj_source'):
        SSDMimoSSM(SSDMimoConfig(d_model=4, d_state=8, d_inner=8, n_groups=2, n_heads=2, d_ssm_in=2, proj_source=source))


@pytest.mark.gpu
@pytest.mark.parametrize('rank,outputs', [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize('rot', [None, 'm3'])
@pytest.mark.parametrize('sharing', ['input_output', 'channel', 'head'])
def test_input_skip_cuda_physical_law(require_cuda, rank, outputs, rot, sharing):
    from test_ssd_endpoint import physical_oracle
    torch.manual_seed(712)
    m = SSDMimoSSM(SSDMimoConfig(d_model=8, d_state=16, d_inner=32*rank, d_ssm_in=rank, d_ssm_out=outputs, n_heads=2,
                   proj_source='layer_input', bc_conv=False,
                   row_blocks=rows_for(rot),
                   r_mix='learned', r_mix_dyn=True, skip_sharing=sharing), execution={'backend': 'cuda', 'precision': 'ieee'}).cuda().float()
    with torch.no_grad():
        m.D.normal_()
        m.r_mix.normal_()
        m.r_mix_proj.weight.normal_(std=.2)
    # fp64 CPU copy of the module for the oracle's sum|terms| (before any
    # forward leaves non-leaf caches on it)
    m64 = copy.deepcopy(m).double().cpu()
    x = torch.randn(1, 7, m.d_inner, device='cuda', dtype=torch.float32, requires_grad=True)
    z = torch.randn(1, 7, m.d_out, device="cuda", requires_grad=True)
    source = torch.randn(1, 7, 8, device='cuda', dtype=torch.float32, requires_grad=True)
    h = torch.zeros(1, m.H, m.P, m.N, device='cuda', dtype=torch.float32)
    # No complete State: exercise the actual CUDA sequence path, not the step fallback.
    y = m.scan(x, sel_x=source, gate_z=z)
    expected, _ = physical_oracle(m, x, source, h, z)
    seed = torch.randn_like(y)
    cpu = [v.detach().cpu().requires_grad_() for v in (x, source, z)]
    t64 = terms_abs(lambda xv, sv, zv: physical_oracle(m64, xv, sv, h.cpu().double(), zv),
                    *cpu, params=m64, grad_outputs=(seed.cpu(), None))
    # A7: u=fp32 (the fp32 IEEE CUDA route vs the fp32 physical oracle, fed the
    # SAME fp32 inputs and weights; both round, so gamma(cuda) + gamma(oracle)
    # = 2 c u terms, the oracle's terms standing for both).  c = _c_ssd(L=7,
    # N=16, d_sel=8, R, sfu=9) + R*O + 2 for the r_mix sibling mix
    # (learned + dynamic: one product and an R-term sum).
    c_fwd = _c_ssd(7, 16, 8, rank, sfu=9) + rank * outputs + 2
    assert_close_terms(y, expected, t64.terms[0], "fp32", 2 * c_fwd, name="y")
    leaves = (x, z, source, *m.parameters())
    names = [n for n, _ in m.named_parameters()]
    # terms_abs grads follow its requires-grad arguments (x, source, z)
    t_in = {0: t64.grad_terms[0], 1: t64.grad_terms[2], 2: t64.grad_terms[1]}
    got = torch.autograd.grad(y, leaves, seed, retain_graph=True, allow_unused=True)
    ref = torch.autograd.grad(expected, leaves, seed, allow_unused=True)
    for i, (a, b) in enumerate(zip(got, ref)):
        if a is None or b is None:
            present = b if a is None else a
            if present is not None:
                torch.testing.assert_close(present, torch.zeros_like(present), rtol=0, atol=0)
        else:
            t = t_in[i] if i < 3 else t64.param_grad_terms.get(names[i - 3])
            # A7: u=fp32, both arms (2x); c = _c_ssd_grad: reduction B*L*H =
            # 14 token-heads, fan-out d_inner = 32 R (a projection column).
            assert_close_terms(a, b, 0.0 if t is None else t, "fp32",
                               2 * _c_ssd_grad(c_fwd, 7 * m.H, m.d_inner),
                               name=f"grad {i} ({(['x', 'z', 'source'] + names)[i]})")

@pytest.mark.parametrize('key,value', [('d_model_hint', 4), ('use_main_branch_for_proj', True),
                                       ('low_rank_mode', None), ('use_cuda', False)])
def test_mh_removed_arguments_are_rejected(key, value):
    with pytest.raises(TypeError, match=key):
        HybridMambaSSM(HybridMambaConfig(d_model=8, d_state=8, d_inner=8, n_heads=2, row_blocks=[{'frac': 1., 'decay': 'coord_static'}], bc_bias='none',
                        proj_source='main_branch', **{key: value}))


def test_ssd_old_shapes_are_not_upgraded():
    kw = dict(d_model=8, d_state=32, d_inner=8, n_heads=2, n_groups=2, d_ssm_in=2,
              proj_source='main_branch', a_dyn_per_sibling=True)
    model = SSDMimoSSM(SSDMimoConfig(**kw), execution={'backend': 'torch'})
    for name, old in [('B_bias', model.B_bias[:, 0]),
                      ('a_dyn_proj.weight', model.a_dyn_proj.weight[::2]),
                      ('a_dyn_proj.bias', model.a_dyn_proj.bias[::2])]:
        state = dict(model.state_dict())
        state[name] = old
        with pytest.raises(RuntimeError, match='size mismatch'):
            model.load_state_dict(state, strict=True)

@pytest.mark.parametrize('layout', ['interleaved', 'half'])
def test_ssd_layout_load_refuses_both_a_mismatched_and_an_absent_code(layout):
    """PINS: `pair_layout_code` is persistent, and BOTH failure modes refuse.

    The two layouts differ by the state-axis permutation P, so a state dict
    that does not declare its basis cannot be loaded into either layout: it
    would install a permuted basis with no exception and no NaN. This pins
    (i) the buffer reaches the state dict at all, (ii) a MISMATCHED code
    raises, (iii) an ABSENT code raises and the message names
    `convert_pair_layout`. It does NOT pin the numerical equivalence of the
    two layouts -- that is the conversion test above.
    """
    kw = dict(d_model=8, d_state=32, d_inner=8, n_heads=2, n_groups=2, d_ssm_in=2,
              proj_source='layer_input', bc_conv=False, 
              row_blocks=ROT)
    core = SSDMimoSSM(SSDMimoConfig(**kw, pair_layout=layout), execution={'backend': 'torch'}).double()
    other = SSDMimoSSM(SSDMimoConfig(**kw, pair_layout=(
        'half' if layout == 'interleaved' else 'interleaved')), execution={'backend': 'torch'}).double()

    # (i) persistent: a plain state_dict() carries the code, and it is the
    # code of THIS module.
    sd = core.state_dict()
    assert 'pair_layout_code' in sd
    assert int(sd['pair_layout_code']) == int(layout == 'half')
    core.load_state_dict(sd, strict=True)        # a labelled round trip loads

    # (ii) mismatched code
    with pytest.raises(ValueError, match='convert_pair_layout'):
        other.load_state_dict(dict(sd), strict=True)

    # (iii) absent code -- the hole this test closes. Strict and non-strict
    # both refuse: `strict=False` is not a licence to guess a basis.
    unlabelled = {k: v for k, v in sd.items() if k != 'pair_layout_code'}
    for strict in (True, False):
        with pytest.raises(ValueError, match='convert_pair_layout'):
            core.load_state_dict(dict(unlabelled), strict=strict)
        with pytest.raises(ValueError, match='pair_layout_code'):
            other.load_state_dict(dict(unlabelled), strict=strict)

    # the explicit route still works: declare the basis, then convert.
    relabelled = dict(unlabelled)
    relabelled['pair_layout_code'] = sd['pair_layout_code'].clone()
    other.load_state_dict(other.convert_pair_layout(relabelled), strict=True)

@pytest.mark.parametrize('family', ['mamba2', 'mamba3'])
def test_the_official_mappers_declare_their_state_coordinate_basis(family):
    """CONTRACT: the partial mappers DECLARE the basis, they are not exempt.

    The load hook refuses a state dict with no `pair_layout_code`, and
    `tests/_official_maps.py`'s two official-Mamba mappers load with
    `strict=False` while carrying permutation-sensitive `B_proj` /
    `C_proj` rows.  If they omitted the code they would either break the
    exactness paths outright or, if the hook were weakened for partial
    dicts, keep installing a permuted basis silently into a
    `pair_layout="half"` mixer.  Pins that they declare INTERLEAVED
    (code 0), because official's rows are copied verbatim.

    CPU and `mamba_ssm`-free on purpose: the two exactness cells that
    exercise this for real need a GPU and the package, so this is the
    cell that fails on a laptop the day someone drops the declaration.
    """
    import torch as _t
    import _official_maps as compat

    d_model, N, H, headdim, d_conv = 32, 32, 4, 16, 4
    d_inner = 2 * d_model
    if family == 'mamba2':
        rows = 2 * d_inner + 2 * N + H
        official = {
            'in_proj.weight': _t.randn(rows, d_model),
            'conv1d.weight': _t.randn(d_inner + 2 * N, 1, d_conv),
            'conv1d.bias': _t.randn(d_inner + 2 * N),
            'dt_bias': _t.randn(H), 'A_log': _t.randn(H), 'D': _t.randn(H),
            'norm.weight': _t.randn(d_inner),
            'out_proj.weight': _t.randn(d_model, d_inner),
        }
        mapped = compat.mamba2_state_dict_to_ssd_mimo_mixer(
            official, d_inner=d_inner, n_heads=H, d_state=N)
    else:
        ka = N // 2
        rows = 2 * d_inner + 2 * N + 3 * H + ka
        official = {
            'in_proj.weight': _t.randn(rows, d_model),
            'dt_bias': _t.randn(H), 'D': _t.randn(H),
            'B_bias': _t.randn(H, 1, N), 'C_bias': _t.randn(H, 1, N),
            'B_norm.weight': _t.randn(N), 'C_norm.weight': _t.randn(N),
            'out_proj.weight': _t.randn(d_model, d_inner),
        }
        mapped = compat.mamba3_state_dict_to_ssd_mimo_mixer(
            official, d_inner=d_inner, n_heads=H, d_state=N)

    assert 'pair_layout_code' in mapped, (
        'the mapper produces permutation-sensitive B/C rows and does not '
        'say which basis they are in')
    assert int(mapped['pair_layout_code']) == 0, 'interleaved, verbatim'

    # ...and the declaration is what makes a half mixer REFUSE rather
    # than silently install a permuted basis.
    core = SSDMimoSSM(SSDMimoConfig(d_model=8, d_state=32, d_inner=8, n_heads=2, n_groups=2, d_ssm_in=2,
                      proj_source='layer_input', bc_conv=False,
                      row_blocks=ROT, pair_layout='half'), execution={'backend': 'torch'})
    with pytest.raises(ValueError, match='convert_pair_layout'):
        core.load_state_dict({'pair_layout_code': mapped['pair_layout_code']},
                             strict=False)


@pytest.mark.parametrize('rank', [1, 2])
@pytest.mark.parametrize('conv', [False, True])
def test_ssd_layout_conversion_is_explicit_and_equivalent(rank, conv):
    torch.manual_seed(17)
    # d_conv is EXPLICIT, not the generic recipe default: `bc_conv=True`
    # requires d_conv > 0, so the conv arm must ask for the width it
    # exercises and the no-conv arm for its absence.  Neither arm moves
    # if the recipe default moves.
    kw = dict(d_model=4*rank, d_state=32, d_inner=4*rank, n_heads=2, n_groups=2,
              d_ssm_in=rank, proj_source='layer_input', bc_conv=conv,
              d_conv=4 if conv else 0,
              row_blocks=ROT)
    a = SSDMimoSSM(SSDMimoConfig(**kw, pair_layout='interleaved'), execution={'backend': 'torch'}).double()
    b = SSDMimoSSM(SSDMimoConfig(**kw, pair_layout='half'), execution={'backend': 'torch'}).double()
    with torch.no_grad():
        for name, p in a.named_parameters():
            if 'theta' in name or name in ('B_bias', 'C_bias'):
                p.uniform_(-.3, .3)
    source = a.state_dict()
    with pytest.raises(ValueError, match='pair_layout'):
        b.load_state_dict(source)
    b.load_state_dict(b.convert_pair_layout(source))
    assert int(source['pair_layout_code']) == 0
    for name, value in a.convert_pair_layout(b.state_dict()).items():
        torch.testing.assert_close(value, source[name], rtol=0, atol=0)
    x = torch.randn(1, 4, a.d_inner, dtype=torch.float64, requires_grad=True)
    y = a.scan(x, sel_x=x)
    yy = b.scan(x, sel_x=x)
    seed = torch.randn_like(y)
    # the layouts permute the state axis, so N-reductions reassociate: sum|terms|
    # of both layouts, each rounding on its own order
    t_a = terms_abs(lambda v: a.scan(v, sel_x=v), x, grad_outputs=seed)
    t_b = terms_abs(lambda v: b.scan(v, sel_x=v), x, grad_outputs=seed)
    # A7: u=fp64 (two fp64 layouts of one law), bound gamma(a) + gamma(b);
    # c = _c_ssd(L=4, N=32, d_sel=d_model, R, conv=4 taps when on).
    c_fwd = _c_ssd(x.shape[1], a.N, a.d_inner, rank, conv=4 if conv else 0)
    assert_close_terms(y, yy, t_a.terms + t_b.terms, "fp64", c_fwd, name="y")
    g, = torch.autograd.grad(y, x, seed)
    gg, = torch.autograd.grad(yy, x, seed)
    # A7: u=fp64, both layouts; c = _c_ssd_grad (no parameter reduction;
    # fan-out: the widest projection, 2N rows of B|C).
    assert_close_terms(g, gg, t_a.grad_terms[0] + t_b.grad_terms[0], "fp64",
                       _c_ssd_grad(c_fwd, 0, 2 * a.N), name="dx")

@pytest.mark.parametrize('source', ['main_branch', 'layer_input'])
@pytest.mark.parametrize('law', ['coord_static', 'coord_dynamic', 'coord_potential', 'head_dynamic', 'head_static'])
def test_mh_current_path_matches_step_after_cleanup(source, law):
    torch.manual_seed(11)
    m = HybridMambaSSM(HybridMambaConfig(d_model=3, d_state=16, d_inner=4, n_heads=2, row_blocks=[{'frac': 1., 'decay': law}],
                       bc_bias='head' if law in HM_FLAT else 'none',
                       proj_source=source, bc_conv=False), execution={'backend': 'torch'}).double()
    x = torch.randn(1, 4, 4, dtype=torch.float64, requires_grad=True)
    sel = (torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
           if source == 'layer_input' else None)
    y = m.scan(x, sel_x=sel)
    state = ScanState(m).init_state(1, dtype=torch.float64)
    ys = []
    for t in range(x.shape[1]):
        yt, state = m.scan_step(x[:, t], state, sel_x=sel[:, t] if sel is not None else None)
        ys.append(yt)
    ref = torch.stack(ys, 1)
    leaves = (x, *(() if sel is None else (sel,)), *m.parameters())
    seed = torch.randn_like(y)

    def parallel(xv, sv=None):
        return m.scan(xv, sel_x=sv)

    def stepped(xv, sv=None):
        st = ScanState(m).init_state(1, dtype=xv.dtype)
        out = []
        for t in range(xv.shape[1]):
            yt, st = m.scan_step(xv[:, t], st, sel_x=sv[:, t] if sv is not None else None)
            out.append(yt)
        return torch.stack(out, 1)

    ins = (x,) if sel is None else (x, sel)
    t_par = terms_abs(parallel, *ins, params=m, grad_outputs=seed)
    t_step = terms_abs(stepped, *ins, params=m, grad_outputs=seed)
    # A7: u=fp64 (the chunked route vs the step route, both fp64), bound
    # gamma(parallel) + gamma(step); c = _c_ssd(L=4, N=16, d_sel=max(d_inner,
    # d_model)=4).
    c_fwd = _c_ssd(x.shape[1], 16, 4)
    assert_close_terms(y, ref, t_par.terms + t_step.terms, "fp64", c_fwd, name="y")
    grads = torch.autograd.grad(y, leaves, seed, allow_unused=True)
    refs = torch.autograd.grad(ref, leaves, seed, allow_unused=True)
    names = [n for n, _ in m.named_parameters()]
    for i, (a, b) in enumerate(zip(grads, refs)):
        if a is None or b is None:
            assert a is None and b is None
        else:
            # A7: u=fp64, both routes' terms; c = _c_ssd_grad: reduction
            # B*L*H = 8 token-heads, fan-out N = 16 rows of a map.
            assert_close_terms(a, b, _grad_terms(t_par, len(ins), names, i)
                               + _grad_terms(t_step, len(ins), names, i), "fp64",
                               _c_ssd_grad(c_fwd, x.shape[1] * 2, 16),
                               name=f"grad {i}")


@pytest.mark.parametrize('layout', ['interleaved', 'half'])
@pytest.mark.parametrize('write_rule', ['euler', 'trapezoid'])
def test_rotation_and_discretization_resolve_independently(layout, write_rule):
    torch.manual_seed(31)
    use_trap = write_rule == 'trapezoid'
    m = SSDMimoSSM(SSDMimoConfig(d_model=8, d_state=32, d_inner=8, d_ssm_in=2, d_ssm_out=2, n_heads=2, n_groups=2,
                   row_blocks=ROT, write_rule=write_rule, pair_layout=layout,
                   proj_source='main_branch', bc_conv=False), execution={'backend': 'torch'}).double()
    assert m.use_trap is use_trap
    assert (getattr(m, "trap_proj", None) is not None) is use_trap
    with torch.no_grad():
        m.theta_proj.weight.normal_(std=.2)
    # Two INDEPENDENT arms of the same law (they share no scan code): the
    # chunked `torch` route under test, and the token-by-token `sequential`
    # backend (a deepcopy, so both read identical weights); plus the
    # module's own `step` loop for the forward and carries.
    ref = copy.deepcopy(m)
    ref.set_execution(dict(backend='sequential'))
    x = torch.randn(2,5,8,dtype=torch.float64,requires_grad=True)
    h = torch.randn(2,*m.hidden_state_shape(),dtype=torch.float64,requires_grad=True)
    xr, hr0 = [v.detach().clone().requires_grad_() for v in (x, h)]

    def route(mod):
        def run(xx, h0):
            yy, st = mod.scan(xx, state=ScanState(mod).init_state(xx.shape[0], h_init=h0, device=xx.device,
                                                  dtype=xx.dtype), return_state=True)
            carries = {k: st[k] for k in sorted(st)}
            return yy, carries, yy.square().sum() + st.hidden_state.square().sum()
        return run

    parallel, cp, _ = route(m)(x, h)
    sequential, cs, _ = route(ref)(xr, hr0)
    hp, hs = cp['hidden_state'], cs['hidden_state']
    assert ('write_prev' in cp) is use_trap
    with torch.no_grad():
        st = ScanState(ref).init_state(2, h_init=h.detach(), device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.shape[1]):
            yt, st = ref.scan_step(x.detach()[:, t], st)
            ys.append(yt)
        stepped = torch.stack(ys, 1)
    one = torch.ones((), dtype=torch.float64)
    no = {k: None for k in cp}
    t_m = terms_abs(route(m), xr, hr0, params=m, grad_outputs=(None, no, one))
    t_r = terms_abs(route(ref), xr, hr0, params=ref, grad_outputs=(None, no, one))
    # A7: u=fp64 (torch route vs sequential backend vs step loop, all
    # fp64); each arm rounds on its own summands -> the terms of both (the
    # step loop shares the sequential backend's summands: 2 t_r); c =
    # _c_ssd(L=5, N=32, d_sel=d_inner, R=2) per arm.
    c_fwd = _c_ssd(x.shape[1], m.N, m.d_inner, 2)
    assert_close_terms(parallel, sequential, t_m.terms[0] + t_r.terms[0],
                       "fp64", c_fwd, name="y")
    assert_close_terms(stepped, sequential, 2 * t_r.terms[0], "fp64", c_fwd,
                       name="step y")
    assert set(st.keys()) == set(cs)
    for key in cs:
        assert_close_terms(cp[key], cs[key], t_m.terms[1][key] + t_r.terms[1][key],
                           "fp64", c_fwd, name=key)
        assert_close_terms(st[key], cs[key], 2 * t_r.terms[1][key], "fp64",
                           c_fwd, name=f"step {key}")
    gp = torch.autograd.grad(parallel.square().sum()+hp.square().sum(), (x,h))
    gs = torch.autograd.grad(sequential.square().sum()+hs.square().sum(), (xr,hr0))
    # A7: c = _c_ssd_grad: reduction B*L*H = 20 token-heads, fan-out d_inner,
    # +2 for the squared loss.
    c_grad = _c_ssd_grad(c_fwd, 20, m.d_inner) + 2
    for name, a, b, ta, tb in zip(("dx", "dh"), gp, gs, t_m.grad_terms, t_r.grad_terms):
        assert_close_terms(a, b, ta + tb, "fp64", c_grad, name=name)


def test_official_m2_copy_declares_and_checks_pair_layout():
    """The copy helper must validate metadata too, not bypass the load hook."""
    pytest.importorskip('mamba_ssm')
    from mamba_ssm.modules.mamba2 import Mamba2
    from _official_maps import copy_official_mamba2_into_ssd_mimo_mixer
    from orbit_ssm.presets import mamba2_config, mamba3_config
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    official = Mamba2(d_model=32, d_state=32, expand=2, headdim=16,
                      ngroups=1, d_conv=4, device='cpu')
    target = SSDMimoSSM(mamba2_config(32, d_state=32, d_head=16), execution={'backend': 'torch'})
    assert copy_official_mamba2_into_ssd_mimo_mixer(official, target) == []
    assert target.pair_layout_code.item() == 0
    half = SSDMimoSSM(mamba3_config(32, d_state=32, d_head=16, pair_layout='half'),
                     execution={'backend': 'torch'})
    with pytest.raises(ValueError, match='convert_pair_layout'):
        copy_official_mamba2_into_ssd_mimo_mixer(official, half)
