
from __future__ import annotations
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

def scalar_decay_scan_seq(a: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    length = a.shape[1]
    h = drive.new_zeros(drive.shape[0], *drive.shape[2:])
    out = []
    for t in range(length):
        h = a[:, t, ..., None] * h + drive[:, t]
        out.append(h)
    return torch.stack(out, 1)

def scalar_decay_scan_chunked(a: torch.Tensor, drive: torch.Tensor, chunk: int) -> torch.Tensor:
    bsz, length, heads, pdim = a.shape
    state = drive.shape[-1]
    n_chunks, tokens = (length // chunk, chunk)
    log_a = torch.log(a.clamp_min(torch.finfo(a.dtype).tiny)).view(bsz, n_chunks, tokens, heads, pdim)
    la_cum = log_a.cumsum(dim=2)
    la_flat = la_cum.permute(0, 1, 3, 4, 2).reshape(-1, tokens)
    diff = (la_flat.unsqueeze(2) - la_flat.unsqueeze(1)).clamp(max=0)
    decay_mat = torch.exp(diff).tril()
    d_c = drive.reshape(bsz, n_chunks, tokens, heads, pdim, state)
    d_flat = d_c.permute(0, 1, 3, 4, 2, 5).reshape(-1, tokens, state)
    h_flat = torch.bmm(decay_mat, d_flat)
    h_local = h_flat.view(bsz, n_chunks, heads, pdim, tokens, state).permute(0, 1, 4, 2, 3, 5)
    total_decay = torch.exp(la_cum[:, :, -1])
    carry = drive.new_zeros(bsz, heads, pdim, state)
    carry_list = [carry]
    for c in range(n_chunks - 1):
        carry = total_decay[:, c, :, :, None] * carry + h_local[:, c, -1]
        carry_list.append(carry)
    carries = torch.stack(carry_list, 1)
    decay_prefix = torch.exp(la_cum).unsqueeze(-1)
    h = h_local + decay_prefix * carries[:, :, None, :, :, :]
    return h.reshape(bsz, length, heads, pdim, state)

def scalar_decay_scan(a: torch.Tensor, drive: torch.Tensor, chunk: int | None=None) -> torch.Tensor:
    length = a.shape[1]
    if chunk and length > chunk and (length % chunk == 0):
        return scalar_decay_scan_chunked(a, drive, chunk)
    return scalar_decay_scan_seq(a, drive)
