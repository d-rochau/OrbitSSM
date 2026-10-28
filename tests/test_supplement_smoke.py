"""Small standalone checks for the supplementary source distribution."""
import copy
import pytest
import torch
from orbit_ssm import SSDMimoSSM, HybridMambaSSM, HybridMambaConfig, SSDMimoConfig


_CONFIG = {SSDMimoSSM: SSDMimoConfig, HybridMambaSSM: HybridMambaConfig}


@pytest.mark.parametrize('family', ['ssd', 'hm'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mixer_reference_and_gradients(family, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(31)
    cls = SSDMimoSSM if family == 'ssd' else HybridMambaSSM
    kw = dict(d_model=32, d_state=16, expand=1, d_conv=0,
              n_heads=2)
    ref = cls(_CONFIG[cls](**kw)).double().to(device)
    mod = copy.deepcopy(ref).float()
    mod.set_execution(dict(backend='cuda' if device == 'cuda' else 'torch'))
    x = torch.randn(1, 32, 32, device=device, requires_grad=True)
    xr = x.detach().double().requires_grad_()
    y, yr = mod(x), ref(xr)
    dy = torch.randn_like(y)
    g, = torch.autograd.grad(y, x, dy)
    gr, = torch.autograd.grad(yr, xr, dy.double())
    assert torch.isfinite(y).all() and torch.isfinite(g).all()
    assert (y.double()-yr).norm()/yr.norm().clamp_min(1e-20) < 0.03
    assert (g.double()-gr).norm()/gr.norm().clamp_min(1e-20) < 0.1


@pytest.mark.parametrize('cls', [SSDMimoSSM, HybridMambaSSM])
def test_state_continuation(cls):
    torch.manual_seed(32)
    m = cls(_CONFIG[cls](d_model=16, d_state=8, expand=1, d_conv=0, n_heads=2),
            execution={'backend': 'torch'}).double()
    x = torch.randn(1, 12, 16, dtype=torch.float64)
    whole = m(x)
    first, state = m(x[:, :5], return_state=True)
    second, _ = m(x[:, 5:], state=state, return_state=True)
    torch.testing.assert_close(torch.cat((first, second), 1), whole, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('p', [32, 64])
@pytest.mark.parametrize('rotating', [False, True])
def test_hm_shared_backward_memory_path(p, rotating):
    """The exported dynamic HM path agrees with fp64 at paper head widths."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(33)
    kw = dict(d_head=p, 
              row_blocks=[dict(frac=1.0, decay='coord_dynamic', complex=rotating)],
              b_norm=True, c_norm=True, bc_bias='head',
              b_gate=False, c_gate=False, write_rule="trapezoid",
              proj_source='main_branch')
    ref = HybridMambaSSM(HybridMambaConfig(d_model=2*p, d_state=16, expand=1,
                           d_conv=0, **kw), execution={'backend': 'torch', 'checkpoint': True}).double().cuda()
    mod = copy.deepcopy(ref).float()
    mod.set_execution(dict(backend='cuda'))
    x = torch.randn(1, 65, 2*p, device='cuda', requires_grad=True)
    xr = x.detach().double().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = mod(x)
    yr = ref(xr)
    dy = torch.randn_like(y)
    y.backward(dy)
    yr.backward(dy.double())
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert (y.double()-yr).norm()/yr.norm().clamp_min(1e-20) < 0.03
    assert (x.grad.double()-xr.grad).norm()/xr.grad.norm().clamp_min(1e-20) < 0.1
