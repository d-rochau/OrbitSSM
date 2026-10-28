"""SSD M3 default law, admissible geometry, and rank warnings."""
import warnings

import pytest
import torch

from _tolerance import assert_close_terms, terms_abs
from test_ssd_endpoint import law_c, law_stats, module_terms

from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig, _a_mag_tail_eager


def test_heavy_tail_values_and_upstream_clamp_adjoint():
    x = torch.tensor([-1e5, -3., -0.25, 0., 0.25, 3.], dtype=torch.float64,
                     requires_grad=True)
    y = _a_mag_tail_eager(x, 1e-4)
    torch.testing.assert_close(y, x.new_tensor([1e-4, .25, .8, 1., 1.25, 4.]))
    # Exact upstream implementation uses two clamp subgradients at zero.
    grad, = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(grad, x.new_tensor([0., 1/16, .64, 2., 1., 1.]))


@pytest.mark.parametrize('per_sibling', [False, True])
def test_default_maps_use_m3_rate_and_explicit_epsilon(per_sibling):
    m = SSDMimoSSM(SSDMimoConfig(d_model=64, d_state=8, d_inner=64, proj_source='main_branch', a_dyn_per_sibling=per_sibling)).double()
    assert m.B_norm.eps == m.C_norm.eps == 1e-5
    assert m.A_log is None
    assert not hasattr(m, 'A_bias')
    x = torch.randn(2, 3, 64, dtype=torch.float64)
    with torch.no_grad():
        m.a_dyn_proj.weight.zero_()
        m.a_dyn_proj.bias.fill_(-3.)
    _, _, _, _, dt, decay = m._token_maps(x)
    torch.testing.assert_close(decay, dt * .25)


@pytest.mark.parametrize('d_state,frac,match', [
    (3, 1.0, 'odd'),                 # a complex block holds (re, im) pairs
    (8, 1.5, r"\['frac'\]"),
    (8, float('nan'), r"\['frac'\]"),
    (8, float('inf'), r"\['frac'\]"),
    (8, 0., r"\['frac'\]"),
    (8, -0.2, r"\['frac'\]"),
    # was 'resolved to 0 rows': now a frac must give an INTEGER
    # row count (no implicit rounding), so a sub-row frac is refused by
    # that rule first; an explicit 0-row block is the next test
    (8, .01, 'must be an integer'),
], ids=['odd d_state', 'frac', 'frac', 'frac', 'frac', 'frac', 'sub-row frac'])
def test_rotation_geometry_is_rejected_at_construction(d_state, frac, match):
    """The rotated size is the complex block of `rows` (was rot_mode='m3'
    + rope_fraction); a bad one is refused at construction."""
    rows = [{'frac': frac, 'decay': 'head_dynamic', 'complex': True}]
    if 0 < frac < 1:
        rows.append({'frac': 1 - frac, 'decay': 'head_dynamic', 'complex': False})
    with pytest.raises(ValueError, match=match):
        SSDMimoSSM(SSDMimoConfig(d_model=64, d_state=d_state, d_inner=64, proj_source='main_branch', row_blocks=rows))


def test_zero_rotated_pairs_is_rejected_at_construction():
    """A complex block that rotates 0 pairs is refused, not built empty."""
    rows = [{'rows': 0, 'decay': 'head_dynamic', 'complex': True},
            {'rows': 8, 'decay': 'head_dynamic', 'complex': False}]
    with pytest.raises(ValueError, match='resolved to 0 rows'):
        SSDMimoSSM(SSDMimoConfig(d_model=64, d_state=8, d_inner=64, proj_source='main_branch', row_blocks=rows))


@pytest.mark.parametrize('r,o', [(1,1), (1,2), (2,1), (2,2), (2,4)])
def test_rectangular_rank_one_warning_only(r, o):
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter('always')
        m = SSDMimoSSM(SSDMimoConfig(d_model=64, d_state=8, d_inner=64, d_ssm_in=r, d_ssm_out=o, proj_source='main_branch'))
    rank_warnings = [w for w in found if 'local temporal block' in str(w.message)]
    assert bool(rank_warnings) == ((r == 1) != (o == 1))
    assert m.scan(torch.randn(1,2,64)).shape[-1] == m.d_out


def test_shared_write_ablation_is_valid_without_gate_or_bias():
    with pytest.warns(RuntimeWarning, match='write gating'):
        m = SSDMimoSSM(SSDMimoConfig(d_model=64, d_state=8, d_inner=64, d_ssm_in=2, b_gate=False, b_norm=False,
                      c_norm=False, bc_bias='none',
                      proj_source='main_branch'))
    assert torch.isfinite(m.scan(torch.randn(1,2,64))).all()


def test_chunk_measurement_control_and_removed_kwargs():
    m = SSDMimoConfig(d_inner=64, proj_source='main_branch').build(
        execution={'kernel_options': {'allow_chunk_autotune': False}})
    assert not m.allow_chunk_autotune
    assert not SSDMimoSSM.from_recipe(m.to_recipe()).allow_chunk_autotune
    for key in ('deterministic', 'allow_degenerate'):
        with pytest.raises(TypeError, match=key):
            SSDMimoSSM(SSDMimoConfig(d_model=64, d_inner=64, **{key: False}))


@pytest.mark.gpu
@pytest.mark.parametrize('per_sibling', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_heavy_tail_fused_forward_and_adjoint(require_cuda, per_sibling, dtype):
    from orbit_ssm.layers.tails import TailCfg, token_tails
    h, r, length = 2, 2, 5
    dt = torch.randn(1, length, h*r, device='cuda', dtype=dtype, requires_grad=True)
    a = torch.linspace(-3, 3, length*h*(r if per_sibling else 1),
                       device='cuda', dtype=dtype).reshape(1, length, -1)
    a[0,0,0] = 0  # pin upstream's clamp derivative convention too
    a.requires_grad_()
    cfg = TailCfg(N=8, R=r, O=r, H=h, L=length, bc_norm=False,
                  shared_bias=False, fold_gb=False, fold_gc=False, a_dyn=True,
                  has_dt_bias=False, has_a_bias=False, eps_b=1e-5, eps_c=1e-5,
                  a_floor=1e-4, a_per_sibling=per_sibling, a_heavy_tail=True)
    _, _, got_dt, got_decay = token_tails(
        cfg, b_src=None, c_src=None, gb_src=None, gc_src=None, dt_src=dt, a_src=a,
        w_b=None, w_c=None, bias_b=None, bias_c=None, bias_dt=None, bias_a=None,
        need_b=False, need_c=False, out_dtype_b=dtype, out_dtype_c=dtype)
    # Upstream's clamp subgradient PASSES at the boundary (grad of
    # clamp_min(x, lo) is 1 at x == lo; torch <= 2.11 clamp backward).
    # torch 2.14 made clamp's mask strict (grad 0 at x == lo), so the
    # convention is spelled out here rather than inherited from torch --
    # see a7493cd and `test_heavy_tail_values_and_upstream_clamp_adjoint`
    # (d rate / d a = 2 at a = 0, the a[0,0,0] cell above).
    def pass_min(v, lo):
        return torch.where(v >= lo, v, v.detach().clamp_min(lo))

    def pass_max(v, hi):
        return torch.where(v <= hi, v, v.detach().clamp_max(hi))

    def law(dt_f, raw):
        """The eager tail in the dtype it is given (fp32 values here, fp64
        absolute arithmetic under terms_abs)."""
        e_dt = torch.nn.functional.softplus(dt_f).reshape(1,length,h,r)
        rate = pass_min(pass_min(raw, 0.) + (1 - pass_max(raw, 0.)).reciprocal(),
                        1e-4)
        rate = rate.reshape(1,length,h,r) if per_sibling else rate[...,None]
        e_decay = (e_dt*rate).permute(0,3,2,1).reshape_as(got_decay)
        return e_dt.permute(0,3,2,1).reshape_as(got_dt), e_decay

    expected_dt, expected_decay = law(dt.float(), a.float())
    seeds = (torch.randn_like(got_dt), torch.randn_like(got_decay))
    ref = terms_abs(law, dt, a, grad_outputs=seeds)
    # A7: u=fp32 (fused Triton tail vs the eager fp32 chain on the SAME
    # rounded inputs; both call libdevice expf / log1pf).  c, worst case per
    # arm x 2 arms: dt = softplus: expf <= 2 ulp (4u) + log1pf <= 1 ulp (2u)
    # -> c = 2 * 6 = 12; decay = dt * rate adds the heavy-tail rate (a
    # reciprocal <= 2 ulp, one add, one clamp) and the product: + 6u per arm
    # -> c = 2 * 12 = 24.  sum|terms|: |dt|, |dt| * |rate| (terms_abs).
    assert_close_terms(got_dt, expected_dt, ref.terms[0], "fp32", 12.0, name="dt")
    assert_close_terms(got_decay, expected_decay, ref.terms[1], "fp32", 24.0,
                       name="decay")
    grads = torch.autograd.grad((got_dt,got_decay),(dt,a), seeds, retain_graph=True)
    expected = torch.autograd.grad((expected_dt,expected_decay),(dt,a),seeds)
    # A7 gradients: d dt = s_dt sig + s_dec rate sig, d a = s_dec dt rate':
    # sums of <= 2 products of <= 3 factors (n + m = 5) plus the sigmoid /
    # rate' chains (<= 7 roundings) per arm -> c32 = 2 * 12 = 24 at u=fp32.
    # bf16 leaves: both arms round that fp32 gradient to bf16 once more (the
    # .float() cast's backward / the kernel's store), 1 u_bf16 per arm ->
    # c = 2 + 24 u32 / u_bf16 at u=bf16.
    if dtype == torch.bfloat16:
        u_g, c_g = "bf16", 2 + 24 * 2.0 ** -24 / 2.0 ** -8
    else:
        u_g, c_g = "fp32", 24.0
    for name, g, e, t in zip(("dt", "a"), grads, expected, ref.grad_terms):
        assert_close_terms(g, e, t, u_g, c_g, name=f"grad {name}")


@pytest.mark.parametrize("a_bias,trap_bias", [(False,False),(False,True),(True,False)])
def test_optional_projection_biases_match_zero_bias_layer(a_bias,trap_bias):
    torch.manual_seed(910)
    opts = dict(d_model=8,d_state=8,d_inner=16,d_ssm_in=2,d_ssm_out=2,n_heads=2,n_groups=4,
                proj_source="layer_input")
    zero = SSDMimoSSM(SSDMimoConfig(**opts), execution={"backend": "torch"}).double()
    plain = SSDMimoSSM(SSDMimoConfig(**opts,a_proj_bias=a_bias,trap_proj_bias=trap_bias), execution={"backend": "torch"}).double()
    state = zero.state_dict()
    for name, enabled in [("a_dyn_proj",a_bias),("trap_proj",trap_bias)]:
        assert (getattr(plain,name).bias is not None) == enabled
        if not enabled:
            state.pop(name+".bias")
    plain.load_state_dict(state,strict=True)
    x = torch.randn(1,5,16,dtype=torch.float64,requires_grad=True)
    sel = torch.randn(1,5,8,dtype=torch.float64,requires_grad=True)
    y, ref = plain.scan(x,sel_x=sel), zero.scan(x,sel_x=sel)
    torch.testing.assert_close(y,ref,atol=0,rtol=0)
    # the scan's own parameters (the layer's projections do not enter the scan)
    names = {n: p for n, p in plain.named_parameters()
             if not n.startswith(("in_proj", "conv1d", "out_proj", "norm"))}
    gp = torch.autograd.grad(y.sum(),(x,sel,*names.values()))
    gz = torch.autograd.grad(ref.sum(),(x,sel,*(dict(zero.named_parameters())[k] for k in names)))
    # A7: u=fp64 (the forward is bit-identical above; the backward of a
    # biased and an unbiased Linear may take different GEMM routes, so the
    # gradients are a reassociation); c=law_c (sqrt(2 * depth) + decay +
    # phase, test_ssd_endpoint); sum|terms| of the zero-bias reference
    _, c_g = law_c(zero, law_stats(zero, x, sel, x.new_zeros(1, zero.H, zero.P, zero.N)), 5)
    t = module_terms(zero, lambda x, sel: zero.scan(x, sel_x=sel), x, sel,
                     grad_outputs=torch.ones_like(ref))
    terms = dict(zip(("x", "sel"), t.grad_terms), **t.param_grad_terms)
    for name,got,expected in zip(("x", "sel", *names),gp,gz):
        assert_close_terms(got, expected, terms[name], "fp64", c_g, name=f"grad {name}")
    restored = SSDMimoSSM.from_recipe(plain.to_recipe())
    assert (restored.a_dyn_proj.bias is not None) == a_bias
    assert (restored.trap_proj.bias is not None) == trap_bias


def test_m3_preset_omits_extra_projection_bias_parameters():
    from orbit_ssm.presets import mamba3_config
    m = SSDMimoSSM(mamba3_config(16,d_state=16,d_head=8))
    assert m.a_dyn_proj.bias is m.trap_proj.bias is None
    assert m.dt_proj.bias is not None
