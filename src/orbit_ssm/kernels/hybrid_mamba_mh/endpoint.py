"""Trapezoid (transported endpoint) weights for `mh_scan`: the write and the diagonal.

The two-tap write h_t = a_t h_{t-1} + own_t B_t x_t + a_t nxt_{t-1} B_{t-1} x_{t-1},
own_t = lam_t dt_t, nxt_t = (1 - lam_{t+1}) dt_{t+1} (nxt_{L-1} = 0), is carried as
z_t = h_t + nxt_t B_t x_t, a ONE-tap scan with write w_t = own_t + nxt_t; the readout
is h_t = a_t z_{t-1} + own_t B_t x_t.  The kernels therefore take (w, own), never
nxt: `own` is formed here as the product lam*dt, not recovered as w - nxt (which
cancels to nothing on a lam -> 0 head).  Their adjoints come back split the same
way -- gw carries only the transport (the strict adjoint), gown the diagonal -- so
the backward below adds no pair of differently rounded copies of one term:

    dL/ddt_t  = lam_t (gw_t + gown_t) + (1 - lam_t) gw_{t-1}
    dL/dlam_t = dt_t (gw_t + gown_t - gw_{t-1})

(SSD's `ssd_mimo.endpoint.endpoint_weights` returns (w, nxt) for its own kernels.)
"""
import torch
import triton
import triton.language as tl

from ..autograd_guard import no_double_backward


@triton.jit
def _mh_weights(D, A, W, O, GW, GO, GD, GA, SIZE, L: tl.constexpr,
                BLOCK: tl.constexpr, BACKWARD: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < SIZE
    dt = tl.load(D + i, mask, 0).to(tl.float32)
    lam = tl.load(A + i, mask, 0).to(tl.float32)
    if not BACKWARD:
        dn = tl.load(D + i + 1, mask & (i % L < L - 1), 0).to(tl.float32)
        ln = tl.load(A + i + 1, mask & (i % L < L - 1), 0).to(tl.float32)
        own = lam * dt
        tl.store(O + i, own, mask)
        tl.store(W + i, own + dn * (1. - ln), mask)
    else:
        gw = tl.load(GW + i, mask, 0)
        gt = gw + tl.load(GO + i, mask, 0)
        prev = tl.load(GW + i - 1, mask & (i % L > 0), 0)
        tl.store(GD + i, lam * gt + (1. - lam) * prev, mask)
        tl.store(GA + i, dt * (gt - prev), mask)


class _MHEndpointWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt, lam):
        dt, lam = dt.contiguous(), lam.contiguous()
        write, own = torch.empty_like(dt), torch.empty_like(dt)
        _mh_weights[(triton.cdiv(dt.numel(), 512),)](
            dt, lam, write, own, dt, dt, dt, dt, dt.numel(), L=dt.shape[-1],
            BLOCK=512, BACKWARD=False)
        ctx.save_for_backward(dt, lam)
        return write, own

    @staticmethod
    @no_double_backward("HybridMamba endpoint weights")
    def backward(ctx, gw, go):
        dt, lam = ctx.saved_tensors
        gw = torch.zeros_like(dt) if gw is None else gw.float().contiguous()
        go = torch.zeros_like(dt) if go is None else go.float().contiguous()
        dd, dl = torch.empty_like(dt), torch.empty_like(lam)
        _mh_weights[(triton.cdiv(dt.numel(), 512),)](
            dt, lam, dt, dt, gw, go, dd, dl, dt.numel(), L=dt.shape[-1],
            BLOCK=512, BACKWARD=True)
        return dd, dl


def mh_endpoint_weights(dt, lam):
    """(write w = own + nxt, own = lam*dt), fp32, time last; see the module doc."""
    if dt.shape != lam.shape:
        raise ValueError("endpoint dt and lambda must have matching shapes")
    if not dt.is_cuda:
        own = lam * dt
        nxt = torch.nn.functional.pad((dt * (1 - lam))[..., 1:], (0, 1))
        return own + nxt, own
    return _MHEndpointWeights.apply(dt.float(), lam.float())
