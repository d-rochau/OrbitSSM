
import torch
from torch.utils.checkpoint import checkpoint
from .torch_scan import _rotate

def _chunk(h, previous, x, k, q, log_a, dt, lam, beta, gamma, phase):
    if beta is not None:
        k = k + beta[None]
    if gamma is not None:
        q = q + gamma[None]
    if phase is not None:
        k, q = (_rotate(k, phase), _rotate(q, phase))
    a = log_a.exp()[:, :, None, None]
    raw = x[..., None, :] * k[:, :, None]
    drive = dt[:, :, None, None] * raw
    if lam is not None:
        prior = torch.cat((previous[..., None], raw[..., :-1]), -1)
        w = lam[:, :, None, None]
        drive = dt[:, :, None, None] * (w * raw + (1 - w) * a * prior)
    b = drive
    stride = 1
    while stride < x.shape[-1]:
        b = torch.cat((b[..., :stride], b[..., stride:] + a[..., stride:] * b[..., :-stride]), -1)
        a = torch.cat((a[..., :stride], a[..., stride:] * a[..., :-stride]), -1)
        stride *= 2
    states = b + a * h[..., None]
    return ((states * q[:, :, None]).sum(-2), states[..., -1].contiguous(), raw[..., -1].contiguous())

def flat_torch_scan(x, k, q, log_a, dt, h, *, lam=None, previous=None, beta=None, gamma=None, phase=None, ckpt=True):
    outputs, finals = ([], [])
    for p in range(0, x.shape[2], 16):
        hp = h[:, :, p:p + 16]
        prev = torch.zeros_like(hp) if previous is None else previous[:, :, p:p + 16]
        chunks = []
        for t in range(0, x.shape[-1], 32):
            sl = slice(t, t + 32)
            args = (hp, prev, x[:, :, p:p + 16, sl], k[..., sl], q[..., sl], log_a[..., sl], dt[..., sl], None if lam is None else lam[..., sl], beta, gamma, None if phase is None else phase[..., sl])
            if ckpt and torch.is_grad_enabled() and any((a is not None and a.requires_grad for a in args)):
                y, hp, prev = checkpoint(_chunk, *args, use_reentrant=False)
            else:
                y, hp, prev = _chunk(*args)
            chunks.append(y)
        outputs.append(torch.cat(chunks, -1))
        finals.append(hp)
    return (torch.cat(outputs, 2), torch.cat(finals, 2))
