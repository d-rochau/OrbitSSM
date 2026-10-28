
from __future__ import annotations
import math
import os
import torch
__all__ = ['fold_geometry', 'wrapped_angle', 'realify', 'interpret', 'cuda_arch', 'TWO_PI']
TWO_PI = 2.0 * math.pi

def interpret() -> bool:
    return os.environ.get('TRITON_INTERPRET', '0') == '1'
_ARCH = {}

def cuda_arch(dev=None) -> int:
    if not torch.cuda.is_available():
        return 0
    idx = torch.cuda.current_device() if dev is None or getattr(dev, 'index', dev) is None else getattr(dev, 'index', dev)
    a = _ARCH.get(idx)
    if a is None:
        mj, mn = torch.cuda.get_device_capability(idx)
        a = _ARCH[idx] = 10 * mj + mn
    return a

def fold_geometry(Q, B=None, C=None, *, phase=None, paired=False, pi0=None, who='fold_geometry'):
    if Q.dim() != 6 or Q.shape[-1] != Q.shape[-2]:
        raise ValueError(f'{who}: Q must be [B, L, G, Mf, k, k], got {tuple(Q.shape)}')
    Bsz, L, G, Mf, k, _ = Q.shape
    if paired and phase is not None:
        raise ValueError(f"{who}: paired=True ('orthogonal_phase') takes no phase -- SSD applies the per-head phases in-kernel; phase= selects the 'unitary' group")
    group = 'unitary' if phase is not None else 'orthogonal_phase' if paired else 'orthogonal'
    kr = k if group == 'orthogonal' else 2 * k
    Mp = 0
    if phase is not None:
        if phase.dim() != 5 or tuple(phase.shape[:3]) != tuple(Q.shape[:3]) or phase.shape[4] != k or (phase.shape[3] < Mf):
            raise ValueError(f'{who}: phase must be [B, L, G, Mf+Mp, k] = [{Bsz}, {L}, {G}, >= {Mf}, {k}], got {tuple(phase.shape)}')
        Mp = phase.shape[3] - Mf
    if pi0 is not None:
        if group == 'unitary':
            raise ValueError(f"{who}: pi0 is the real chain's initial prefix ('orthogonal', paired); 'unitary' (phase=) takes none")
        if tuple(pi0.shape) != (Bsz, G, Mf, k, k):
            raise ValueError(f'{who}: pi0 must be [B, G, Mf, k, k] = {[Bsz, G, Mf, k, k]}, got {tuple(pi0.shape)}')
    for name, V in (('B', B), ('C', C)):
        if V is None:
            continue
        g = 'G' + name.lower()
        if V.dim() != 5 or tuple(V.shape[:2]) != (Bsz, L):
            raise ValueError(f'{who}: {name} must be [B, L, {g}, N, cols] with B, L = {(Bsz, L)}, got {tuple(V.shape)}')
        if V.shape[2] % G:
            raise ValueError(f'{who}: map groups {g}={V.shape[2]} must be a multiple of the angle groups G={G}')
    if B is not None and C is not None:
        N = B.shape[3]
        if C.shape[3] != N:
            raise ValueError(f'{who}: B and C state widths differ ({N} vs {C.shape[3]})')
        if N < (Mf + Mp) * kr:
            raise ValueError(f'{who}: N={N} < (Mf+Mp)*kk = ({Mf}+{Mp})*{kr} = {(Mf + Mp) * kr} block rows')
    return (group, Mf, Mp, k, kr)

def wrapped_angle(phi: torch.Tensor) -> torch.Tensor:
    x = phi.double().movedim(1, -1).contiguous()
    Phi = torch.cumsum(x, dim=-1)
    Phi = Phi - TWO_PI * torch.round(Phi / TWO_PI)
    return Phi.movedim(-1, 1)

def realify(Pr: torch.Tensor, Pim: torch.Tensor | None=None) -> torch.Tensor:
    if Pim is None:
        Pim = torch.zeros_like(Pr)
    k = Pr.shape[-1]
    top = torch.stack([Pr, -Pim], dim=-1)
    bot = torch.stack([Pim, Pr], dim=-1)
    return torch.stack([top, bot], dim=-3).reshape(*Pr.shape[:-2], 2 * k, 2 * k)
