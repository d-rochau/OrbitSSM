"""Scalar-decay scan reference: sequential and chunked forms of h_t = a_t h_{t-1} + d_t."""
from __future__ import annotations
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def scalar_decay_scan_seq(a: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """Token-by-token reference: a [B, L, H, P], drive [B, L, H, P, N]."""
    length = a.shape[1]
    h = drive.new_zeros(drive.shape[0], *drive.shape[2:])
    out = []
    for t in range(length):
        h = a[:, t, ..., None] * h + drive[:, t]
        out.append(h)
    return torch.stack(out, 1)


def scalar_decay_scan_chunked(a: torch.Tensor, drive: torch.Tensor,
                              chunk: int) -> torch.Tensor:
    """Chunked form: intra-chunk via a decay-matrix matmul, inter-chunk by carry.

    h_t = a_t h_{t-1} + d_t equals h = L @ d with L[t,s] = exp(la_cum_t -
    la_cum_s) for t >= s; all entries lie in (0, 1], so nothing overflows.
    Requires L % chunk == 0.

    `a` is clamped to the smallest normal float before the log: a decay that
    underflows to exactly 0 would give log(0) = -inf and NaN differences
    (inf - inf) that poison the scan, where the true limit is total
    forgetting. The clamp changes nothing for healthy decays.
    """
    bsz, length, heads, pdim = a.shape
    state = drive.shape[-1]
    n_chunks, tokens = length // chunk, chunk
    log_a = torch.log(a.clamp_min(torch.finfo(a.dtype).tiny)) \
        .view(bsz, n_chunks, tokens, heads, pdim)
    la_cum = log_a.cumsum(dim=2)                         # [B, nc, T, H, P]

    # decay matrix: L[t,s] = exp(la_cum_t - la_cum_s), entries in (0, 1]
    la_flat = la_cum.permute(0, 1, 3, 4, 2).reshape(-1, tokens)
    diff = (la_flat.unsqueeze(2) - la_flat.unsqueeze(1)).clamp(max=0)
    decay_mat = torch.exp(diff).tril()                   # [BnHP, T, T]

    d_c = drive.reshape(bsz, n_chunks, tokens, heads, pdim, state)
    d_flat = d_c.permute(0, 1, 3, 4, 2, 5).reshape(-1, tokens, state)
    h_flat = torch.bmm(decay_mat, d_flat)                # [BnHP, T, N]
    h_local = h_flat.view(bsz, n_chunks, heads, pdim, tokens, state) \
        .permute(0, 1, 4, 2, 3, 5)                       # [B, nc, T, H, P, N]

    # inter-chunk carry (one step per chunk)
    total_decay = torch.exp(la_cum[:, :, -1])            # [B, nc, H, P]
    carry = drive.new_zeros(bsz, heads, pdim, state)
    carry_list = [carry]
    for c in range(n_chunks - 1):
        carry = total_decay[:, c, :, :, None] * carry + h_local[:, c, -1]
        carry_list.append(carry)
    carries = torch.stack(carry_list, 1)                 # [B, nc, H, P, N]

    # add carry: h_global = h_local + exp(la_from_chunk_start) * carry
    decay_prefix = torch.exp(la_cum).unsqueeze(-1)       # [B, nc, T, H, P, 1]
    h = h_local + decay_prefix * carries[:, :, None, :, :, :]
    return h.reshape(bsz, length, heads, pdim, state)


def scalar_decay_scan(a: torch.Tensor, drive: torch.Tensor,
                      chunk: int | None = None) -> torch.Tensor:
    """The scan, chunked where the length allows it and sequential else."""
    length = a.shape[1]
    if chunk and length > chunk and length % chunk == 0:
        return scalar_decay_scan_chunked(a, drive, chunk)
    return scalar_decay_scan_seq(a, drive)
