
import torch
import triton
import triton.language as tl

@triton.jit
def _mh_weights(D, A, W, O, GW, GO, GD, GA, SIZE, L: tl.constexpr, BLOCK: tl.constexpr, BACKWARD: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < SIZE
    dt = tl.load(D + i, mask, 0).to(tl.float32)
    lam = tl.load(A + i, mask, 0).to(tl.float32)
    if not BACKWARD:
        dn = tl.load(D + i + 1, mask & (i % L < L - 1), 0).to(tl.float32)
        ln = tl.load(A + i + 1, mask & (i % L < L - 1), 0).to(tl.float32)
        own = lam * dt
        tl.store(O + i, own, mask)
        tl.store(W + i, own + dn * (1.0 - ln), mask)
    else:
        gw = tl.load(GW + i, mask, 0)
        gt = gw + tl.load(GO + i, mask, 0)
        prev = tl.load(GW + i - 1, mask & (i % L > 0), 0)
        tl.store(GD + i, lam * gt + (1.0 - lam) * prev, mask)
        tl.store(GA + i, dt * (gt - prev), mask)

class _MHEndpointWeights(torch.autograd.Function):

    @staticmethod
    def forward(ctx, dt, lam):
        dt, lam = (dt.contiguous(), lam.contiguous())
        write, own = (torch.empty_like(dt), torch.empty_like(dt))
        _mh_weights[triton.cdiv(dt.numel(), 512),](dt, lam, write, own, dt, dt, dt, dt, dt.numel(), L=dt.shape[-1], BLOCK=512, BACKWARD=False)
        ctx.save_for_backward(dt, lam)
        return (write, own)

    @staticmethod
    def backward(ctx, gw, go):
        dt, lam = ctx.saved_tensors
        gw = torch.zeros_like(dt) if gw is None else gw.float().contiguous()
        go = torch.zeros_like(dt) if go is None else go.float().contiguous()
        dd, dl = (torch.empty_like(dt), torch.empty_like(lam))
        _mh_weights[triton.cdiv(dt.numel(), 512),](dt, lam, dt, dt, gw, go, dd, dl, dt.numel(), L=dt.shape[-1], BLOCK=512, BACKWARD=True)
        return (dd, dl)

def mh_endpoint_weights(dt, lam):
    if dt.shape != lam.shape:
        raise ValueError('endpoint dt and lambda must have matching shapes')
    if not dt.is_cuda:
        own = lam * dt
        nxt = torch.nn.functional.pad((dt * (1 - lam))[..., 1:], (0, 1))
        return (own + nxt, own)
    return _MHEndpointWeights.apply(dt.float(), lam.float())
