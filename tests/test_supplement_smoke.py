
import copy
import pytest
import torch
from orbit_ssm import SSDMimoMixer, HybridMambaMixer

@pytest.mark.parametrize('family', ['ssd', 'hm'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mixer_reference_and_gradients(family, device):
    if device == 'cuda' and (not torch.cuda.is_available()):
        pytest.skip('CUDA unavailable')
    torch.manual_seed(31)
    cls = SSDMimoMixer if family == 'ssd' else HybridMambaMixer
    kw = dict(d_model=32, d_state=16, expand=1, d_conv=0, ssm_kwargs=dict(heads=2, backend='torch'))
    ref = cls(**kw).double().to(device)
    mod = copy.deepcopy(ref).float()
    mod.ssm.backend = 'cuda' if device == 'cuda' else 'torch'
    x = torch.randn(1, 32, 32, device=device, requires_grad=True)
    xr = x.detach().double().requires_grad_()
    y, yr = (mod(x), ref(xr))
    dy = torch.randn_like(y)
    g, = torch.autograd.grad(y, x, dy)
    gr, = torch.autograd.grad(yr, xr, dy.double())
    assert torch.isfinite(y).all() and torch.isfinite(g).all()
    assert (y.double() - yr).norm() / yr.norm().clamp_min(1e-20) < 0.03
    assert (g.double() - gr).norm() / gr.norm().clamp_min(1e-20) < 0.1

@pytest.mark.parametrize('cls', [SSDMimoMixer, HybridMambaMixer])
def test_state_continuation(cls):
    torch.manual_seed(32)
    m = cls(d_model=16, d_state=8, expand=1, d_conv=0, ssm_kwargs=dict(heads=2, backend='torch')).double()
    x = torch.randn(1, 12, 16, dtype=torch.float64)
    whole = m(x)
    first, state = m(x[:, :5], return_state=True)
    second, _ = m(x[:, 5:], state=state, return_state=True)
    torch.testing.assert_close(torch.cat((first, second), 1), whole, rtol=1e-07, atol=1e-08)

@pytest.mark.parametrize('p', [32, 64])
@pytest.mark.parametrize('rotating', [False, True])
def test_hm_shared_backward_memory_path(p, rotating):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(33)
    kw = dict(head_subsystems=p, backend='torch', rows=[dict(frac=1.0, decay='dyn', complex=rotating)], b_norm=True, c_norm=True, bias_gran='head_bc', use_g_b=False, use_g_c=False, use_trap=True, proj_source='main_branch', ckpt=True)
    ref = HybridMambaMixer(d_model=2 * p, d_state=16, expand=1, d_conv=0, ssm_kwargs=kw).double().cuda()
    mod = copy.deepcopy(ref).float()
    mod.ssm.backend = 'cuda'
    x = torch.randn(1, 65, 2 * p, device='cuda', requires_grad=True)
    xr = x.detach().double().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = mod(x)
    yr = ref(xr)
    dy = torch.randn_like(y)
    y.backward(dy)
    yr.backward(dy.double())
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert (y.double() - yr).norm() / yr.norm().clamp_min(1e-20) < 0.03
    assert (x.grad.double() - xr.grad).norm() / xr.grad.norm().clamp_min(1e-20) < 0.1
