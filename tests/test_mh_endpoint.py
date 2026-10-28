"""Physical endpoint equation, independent of MH's collapsed scan mapping."""
import math

import pytest
import torch

from _tolerance import assert_close_terms, terms_abs

# A7 depth of the MH kernel (fp32 FMA, no tensor-core operands) against an
# fp64 oracle fed the SAME fp32 inputs: only the kernel rounds (u=fp32, no
# sqrt(2)).  A value sums n = L token terms of m <= L decays + _FACTORS other
# factors (x, write clock, B + bias, rotation, gate, C + bias, trap weight,
# skip/z gate ...): sqrt(n + m).  A gradient also sums over the positions
# its parameter is shared by (`shared`, the largest such reduction among
# the compared gradients).  Amplifications of rounded arguments, both formed
# by the kernel in fp32 and passed through exp / cos,sin:
#   * decays: each a_t = exp(log a_t), a term carries a product of up to L
#     of them: + max sum_t sum|terms of log a_t|        (_decay_amp)
#   * rotation: the angle is an L-term prefix sum of the increments, rounded
#     at the typical sqrt(L + 1) u sum|inc|: + sqrt(L + 1) max sum|inc|
_FACTORS = 12


def _c_kernel(length, *, shared=0, amp=0.0):
    return math.sqrt(shared + 2 * length + _FACTORS) + amp


def _angle_amp(increments):
    """increments (..., L): sqrt(L + 1) * max over the rest of sum_t |inc_t|."""
    if increments is None:
        return 0.0
    length = increments.shape[-1]
    return math.sqrt(length + 1) * increments.detach().double().abs().sum(-1).max().item()


def _mh_log_decay(x, delta, A_log, B_t, C_t, **kw):
    """The oracle's log-decay argument log a_t = c1 s1_t + c2 s2_t, (B,H,P,N,L)."""
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import _prepare
    kw = {k: v for k, v in kw.items() if k not in (
        "h_init", "n_off", "n_rows", "d_skip", "z", "trap_lam", "write_prev")}
    st = _prepare(x, delta, A_log, B_t, C_t, **kw)
    return (st["c1"][None, ..., None] * st["s1"][:, :, None]
            + st["c2"][None, ..., None] * st["s2"][:, :, None])


def _decay_amp(args, kw):
    """max over (b,h,p,n) of sum_t sum|terms| of the log-decay argument."""
    terms = terms_abs(_mh_log_decay, *[a.detach() for a in args],
                      **{k: (v.detach() if torch.is_tensor(v) else v)
                         for k, v in kw.items()}).terms
    return terms.sum(-1).max().item()


def _zero_h(oracle):
    """`oracle` with an explicit zero h_init: h_init=None allocates the
    oracle's state as a plain constant, which the absolute-arithmetic pass of
    terms_abs cannot write into; zeros made from an input are the same value."""
    def run(x, delta, A_log, *rest, **kw):
        b, h, p = x.shape[:3]
        return oracle(x, delta, A_log, *rest, h_init=x.new_zeros(b, h, p, A_log.shape[-1]), **kw)
    return run


def _endpoint_oracle(xx, dd, aa, bbm, ccm, ll, hh, ggb, ggc, bbb, ccb, theta=None):
    """The physical endpoint recurrence, fp64, one token per step."""
    b, n = bbm.shape[0], bbm.shape[1]
    length = xx.shape[-1]
    rotated = theta is not None
    prev = torch.zeros_like(hh)
    phase = torch.zeros(b, n // 2, device=xx.device, dtype=torch.float64)
    outputs = []
    for t in range(length):
        bv, cv = bbm[:, :, t] + bbb, ccm[:, :, t] + ccb
        if rotated:
            phase = phase + theta[:, :, t]
            def rotate(v):
                pair = v.reshape(b, n // 2, 2)
                co, si = phase.cos(), phase.sin()
                return torch.stack((co * pair[:, :, 0] - si * pair[:, :, 1],
                                    si * pair[:, :, 0] + co * pair[:, :, 1]), -1).reshape(b, n)
            bv, cv = rotate(bv), rotate(cv)
        bv, cv = bv * ggb[:, :, t].sigmoid(), cv * ggc[:, :, t].sigmoid()
        drive = xx[:, :, :, t, None] * bv[:, None, None, :]
        transition = torch.exp(-aa.exp()[None, :, :, :] * dd[:, :, t, None, None])
        hh = transition * hh + dd[:, :, t, None, None] * (
            ll[:, :, t, None, None] * drive + (1 - ll[:, :, t, None, None]) * transition * prev)
        outputs.append((hh * cv[:, None, None, :]).sum(-1))
        prev = drive
    return torch.stack(outputs, -1), hh


@pytest.mark.parametrize("p,length", [(1, 1), (1, 65), (4, 37), (4, 129)])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_mh_endpoint_equation_and_all_adjoints(require_cuda, p, length, rotated, checkpoint):
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    torch.manual_seed(192)
    b, h, n = 2, 4, 4
    def rand(*shape, scale=0.2):
        return (torch.randn(*shape, device="cuda") * scale).requires_grad_()
    x = rand(b,h,p,length)
    dt = rand(b,h,length).sigmoid().detach().requires_grad_()
    al = rand(h,p,n//2).repeat_interleave(2,-1).detach().requires_grad_()
    bm, cm = rand(b,n,length), rand(b,n,length)
    la = rand(b,h,length).sigmoid().detach().requires_grad_()
    hi = rand(b,h,p,n)
    gb, gc = rand(b,n,length), rand(b,n,length)
    bb, cb = rand(1,n), rand(1,n)
    theta = rand(b,n//2,length,scale=.05) if rotated else None
    vals = [x,dt,al,bm,cm,la,hi,gb,gc,bb,cb] + ([theta] if rotated else [])
    refs = [v.detach().double().requires_grad_() for v in vals]
    expected, hh = _endpoint_oracle(*refs)
    actual, final, _=mh_scan(x,dt,al,bm,cm,trap_lam=la,h_init=hi,g_B=gb,g_C=gc,beta_b=bb,beta_c=cb,theta=theta,delta_softplus=False,return_state=True,ckpt=checkpoint)
    wy,wh=torch.randn_like(actual),torch.randn_like(final)
    ref = terms_abs(_endpoint_oracle, *vals, grad_outputs=(wy, wh))
    # A7: u=fp32 (kernel vs the fp64 oracle on the same fp32 inputs);
    # c=_c_kernel(L) + decay amplification sum_t exp(a) dt + angle amplification;
    # gradients: shared = b*h*p*L (beta_b/beta_c sum over batch, heads,
    # channels, tokens)
    amp = ((al.detach().double().exp()[None, ..., None]
            * dt.detach().double()[:, :, None, None, :]).sum(-1).max().item()
           + _angle_amp(theta))
    c_fwd = _c_kernel(length, amp=amp)
    c_grad = _c_kernel(length, shared=b * h * p * length, amp=amp)
    assert_close_terms(actual, expected, ref.terms[0], "fp32", c_fwd, name="y")
    assert_close_terms(final, hh, ref.terms[1], "fp32", c_fwd, name="h_final")
    actual_grads=torch.autograd.grad((actual,final),vals,(wy,wh))
    expected_grads=torch.autograd.grad((expected,hh),refs,(wy.double(),wh.double()))
    for i,(got,want,s) in enumerate(zip(actual_grads,expected_grads,ref.grad_terms)):
        assert_close_terms(got, want, s, "fp32", c_grad, name=f"grad{i}")


@pytest.mark.parametrize("per_channel", [False, True])
def test_mh_endpoint_reference_and_resume(require_cuda, per_channel):
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import mh_reference, mh_reference_cumprod
    torch.manual_seed(51)
    b,h,p,n,l = 2,4,4,4,67
    x=torch.randn(b,h,p,l,device="cuda",requires_grad=True)
    dt=(torch.rand(b,h,l,device="cuda")*.1+.01).requires_grad_()
    a=torch.zeros(h,p,n,device="cuda",requires_grad=True)
    bm=torch.randn(b,n,l,device="cuda",requires_grad=True)
    cm=torch.randn_like(bm,requires_grad=True)
    lam=torch.rand_like(dt,requires_grad=True)
    prev=torch.randn(b,h,p,n,device="cuda",requires_grad=True)
    dw = torch.rand_like(x, requires_grad=True) if per_channel else None
    inputs=[x,dt,a,bm,cm,lam,prev]+([dw] if per_channel else [])
    kw=dict(delta_softplus=False,trap_lam=lam,write_prev=prev,delta_w=dw)
    ref,last=mh_reference(x,dt,a,bm,cm,**kw)
    closed,clast=mh_reference_cumprod(x,dt,a,bm,cm,**kw)
    wy,wh=torch.randn_like(ref,dtype=torch.float32),torch.randn_like(last,dtype=torch.float32)
    oracle = terms_abs(_zero_h(mh_reference), x, dt, a, bm, cm, grad_outputs=(wy, wh), **kw)
    decay = _decay_amp((x, dt, a, bm, cm), kw)
    # A7: u=fp64 (both oracles fp64 on the same inputs, sqrt(2)); the closed
    # form reads decay products as exp(S_t - S_s) of the cumulative
    # log-decay: + 2 max|S| = 2 * decay (module note)
    c64 = math.sqrt(2 * (2 * l + _FACTORS)) + 2 * decay
    assert_close_terms(closed, ref, oracle.terms[0], "fp64", c64, name="closed y")
    assert_close_terms(clast, last, oracle.terms[1], "fp64", c64, name="closed h")
    actual,alast,_=mh_scan(x,dt,a,bm,cm,return_state=True,**kw)
    cut=31
    first,carry,_=mh_scan(x[...,:cut],dt[...,:cut],a,bm[...,:cut],cm[...,:cut],delta_softplus=False,trap_lam=lam[...,:cut],write_prev=prev,delta_w=dw[...,:cut] if per_channel else None,return_state=True)
    lag=x[...,cut-1,None]*bm[:,None,None,:,cut-1]
    second,carry2,_=mh_scan(x[...,cut:],dt[...,cut:],a,bm[...,cut:],cm[...,cut:],delta_softplus=False,trap_lam=lam[...,cut:],write_prev=lag,h_init=carry,delta_w=dw[...,cut:] if per_channel else None,return_state=True)
    # A7: u=fp32 (kernel vs the fp64 oracle on the same fp32 inputs; the
    # resumed carry `lag` is one more fp32 product, inside _FACTORS);
    # c=_c_kernel(L) + decay amplification; gradients: shared = max(b*L,
    # h*p, p*n) (A_log sums batch and tokens)
    c_fwd = _c_kernel(l, amp=decay)
    c_grad = _c_kernel(l, shared=max(b * l, h * p, p * n), amp=decay)
    for value in [actual,torch.cat((first,second),-1)]:
        assert_close_terms(value, ref, oracle.terms[0], "fp32", c_fwd, name="y")
    ga=torch.autograd.grad((torch.cat((first,second),-1),carry2),inputs,(wy,wh),retain_graph=True)
    gr=torch.autograd.grad((ref,last),inputs,(wy.double(),wh.double()))
    for i,(actual,expected,s) in enumerate(zip(ga,gr,oracle.grad_terms)):
        assert_close_terms(actual, expected, s, "fp32", c_grad, name=f"grad{i}")


@pytest.mark.parametrize("law", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mh_endpoint_in_kernel_rotation_laws(require_cuda, monkeypatch, law, dtype):
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import mh_reference
    torch.manual_seed(611)
    b,h,p,n,l=2,4,4,8,65
    def r(*shape, scale=.1, io=False):
        return (torch.randn(*shape,device="cuda")*scale).to(dtype if io else torch.float32).requires_grad_()
    x,dt,aa,bm,cm=r(b,h,p,l,io=True),r(b,h,l),r(h,p,n//2),r(b,n,l,io=True),r(b,n,l,io=True)
    aa=aa.repeat_interleave(2,-1).detach().requires_grad_()
    kw=dict(trap_lam=r(b,h,l).sigmoid().detach().requires_grad_(),
            delta_w=r(b,h,l),delta_bias=r(h),theta=r(b,n//2,l),
            beta_b=r(h,n),beta_c=r(h,n),g_B=r(b,n,l),g_C=r(b,n,l),
            d_skip=r(h),z=r(b,h,p,l,io=True),row_law=torch.full((n,),law,device="cuda",dtype=torch.int32))
    if law==1:
        kw['a_dyn']=r(b,h,l);kw['a_dyn_res']='h'
    elif law==2:
        kw['mu']=r(b,h,l);kw['mu_res']='h'
        kw['mu_gain']=r(h,p,n//2).repeat_interleave(2,-1).detach().requires_grad_()
    vals=[x,dt,aa,bm,cm]+[v for v in kw.values() if isinstance(v,torch.Tensor) and v.requires_grad]
    reference_kw={k:v.detach().double().requires_grad_() if isinstance(v,torch.Tensor) and v.requires_grad else v for k,v in kw.items()}
    refargs=[v.detach().double().requires_grad_() for v in [x,dt,aa,bm,cm]]
    refvals=refargs+[v for v in reference_kw.values() if isinstance(v,torch.Tensor) and v.requires_grad]
    expected,_=mh_reference(*refargs,**reference_kw)
    actual=mh_scan(x,dt,aa,bm,cm,**kw)
    weight=torch.randn_like(actual)
    ga=torch.autograd.grad(actual,vals,weight,allow_unused=True)
    ge=torch.autograd.grad(expected,refvals,weight.double(),allow_unused=True)
    if dtype==torch.float32:
        oracle = terms_abs(_zero_h(mh_reference), x, dt, aa, bm, cm,
                           grad_outputs=(weight, None), **kw)
        # A7: u=fp32 (kernel vs the fp64 oracle on the same fp32 inputs);
        # c=_c_kernel(L) + decay (A / a_dyn / mu telescoping arguments) and
        # angle amplifications; gradients: shared = b*p*n*L (delta_bias)
        amp = _decay_amp((x, dt, aa, bm, cm), kw) + _angle_amp(kw['theta'])
        c_fwd = _c_kernel(l, amp=amp)
        c_grad = _c_kernel(l, shared=b * p * n * l, amp=amp)
        assert_close_terms(actual, expected, oracle.terms[0], "fp32", c_fwd, name="y")
        grad_terms = oracle.grad_terms
    else:
        assert (actual.double()-expected).norm()/expected.norm() < .025
        grad_terms = [None] * len(ga)
    for i,(got,want,s) in enumerate(zip(ga,ge,grad_terms)):
        if want is None:
            assert got is None or got.count_nonzero()==0
        elif dtype==torch.float32:
            assert_close_terms(got, want, s, "fp32", c_grad, name=f"grad{i}")
        else:
            assert (got.double()-want).norm()/want.norm().clamp_min(1e-12)<.05


def test_mh_oracle_default_rotation_matches_explicit_selector():
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import mh_reference, mh_reference_cumprod
    # The first write and second read select different axes. Rotating
    # them through successive quarter turns gives a nonzero cross read.
    x=torch.ones(1,1,1,2,dtype=torch.float64)
    dt=torch.full((1,1,2),.5,dtype=torch.float64)
    a=torch.zeros(1,1,2,dtype=torch.float64)
    bm=torch.tensor([[[1.,0.],[0.,0.]]],dtype=torch.float64)
    cm=torch.tensor([[[0.,0.],[0.,1.]]],dtype=torch.float64)
    theta=torch.full((1,1,2),torch.pi/2,dtype=torch.float64)
    for oracle in (mh_reference,mh_reference_cumprod):
        implicit,_=oracle(x,dt,a,bm,cm,theta=theta,delta_softplus=False)
        explicit,_=oracle(x,dt,a,bm,cm,theta=theta,row_rot=torch.ones(2,dtype=torch.bool),delta_softplus=False)
        # A7 bit identity: the default selector IS an all-True row mask, so
        # both calls run the same fp operations in the same order
        torch.testing.assert_close(implicit,explicit,rtol=0,atol=0)
        terms = terms_abs(_zero_h(oracle),x,dt,a,bm,cm,theta=theta,delta_softplus=False).terms[0]
        # A7: u=fp64 (the oracle vs the closed form -exp(-1/2)/2, both
        # rounded: sqrt(2)); c = sqrt(2 (L + m)) + |Theta|: L=2 tokens of
        # m=6 factors (x, dt, B, cos/sin, decay, C), cos/sin of the rounded
        # cumulative angle |Theta| <= pi
        c = math.sqrt(2 * (2 + 6)) + math.pi
        closed = -.5*torch.exp(torch.tensor(-.5,dtype=torch.float64))
        assert_close_terms(implicit[0,0,0,1],closed,terms[0,0,0,1],"fp64",c,name=oracle.__name__)


# ── the in-scan trapezoid against the PHYSICAL fp64 oracle ────────────
# `reference.physical_oracle` evaluates
# the two-endpoint law itself -- the law the module means -- and asserts
# internally that the COLLAPSED one-tap form the Gluon scan runs is its
# exact rewrite (`g_t = h_t + n_t v_t`, `w_t = lam_t Dw_t + n_t`, readout
# `- n_t d_t u_t` with `d_t = C~_t^T B~_t`, boundary `n_L = 0` giving
# `g_L = h_L`).  The kernel is then scored against the physical side, so
# a cell here fails if EITHER the kernel or the rewrite is wrong.
@pytest.mark.gpu
@pytest.mark.parametrize("p,length", [(1, 65), (4, 37), (4, 128)])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("resumed", [False, True])
def test_mh_in_scan_trapezoid_matches_the_physical_oracle(
        require_cuda, p, length, rotated, resumed):
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import (
        physical_oracle, rel_err)

    torch.manual_seed(7 + length)
    b, h, n = 2, 4, 8

    def r(*shape, scale=0.2):
        return (torch.randn(*shape, device="cuda") * scale).requires_grad_()

    x = r(b, h, p, length)
    dt = r(b, h, length).sigmoid().detach().requires_grad_()
    al = r(h, p, n // 2).repeat_interleave(2, -1).detach().requires_grad_()
    bm, cm = r(b, n, length), r(b, n, length)
    kw = dict(trap_lam=r(b, h, length).sigmoid().detach().requires_grad_(),
              delta_w=r(b, h, length), beta_b=r(1, n), beta_c=r(1, n),
              h_init=r(b, h, p, n), delta_softplus=False)
    if rotated:
        kw.update(theta=r(b, n // 2, length, scale=.05), theta_delta=True,
                  theta_h=r(h, n // 2, scale=.3))
    if resumed:
        # the endpoint rule's OWN carry: v_{-1}.  `h_final` alone does not
        # contain it, so a resumed chunk must hand it over separately.
        kw["write_prev"] = r(b, h, p, n)

    args = [x, dt, al, bm, cm]
    leaves = args + [v for v in kw.values()
                     if isinstance(v, torch.Tensor) and v.requires_grad]
    y, h_final, _ = mh_scan(*args, **kw, return_state=True, validate=True)
    w = torch.randn_like(y)
    got = torch.autograd.grad(y, leaves, w)

    refargs = [v.detach().double().requires_grad_() for v in args]
    refkw = {k: (v.detach().double().requires_grad_()
                 if isinstance(v, torch.Tensor) and v.requires_grad else v)
             for k, v in kw.items()}
    y_ref, h_ref = physical_oracle(*refargs, **refkw)
    want = torch.autograd.grad(
        y_ref, refargs + [v for v in refkw.values()
                          if isinstance(v, torch.Tensor) and v.requires_grad],
        w.double())

    assert rel_err(y, y_ref) < 3e-4, f"Y {rel_err(y, y_ref):.2e}"
    assert rel_err(h_final, h_ref) < 3e-4, f"h_final {rel_err(h_final, h_ref):.2e}"
    bad = [f"{i}={rel_err(g, e):.1e}" for i, (g, e) in enumerate(zip(got, want))
           if not rel_err(g, e) < 3e-4]
    assert not bad, f"adjoints vs the physical oracle: {bad}"
