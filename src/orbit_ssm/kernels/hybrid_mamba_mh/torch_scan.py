"""Stable Torch MH recurrence with bounded chunk-local affine prefixes.

No inverse cumulative decay is formed. Expanded [B,H,P,N,T] intermediates
are limited by token/channel/row tiles. Training checkpoints chunk interiors,
retaining their boundary states and p-free maps instead of a full hidden tape.
This is a usable fallback, not a claim of CUDA-kernel performance parity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layout import (DYN_SOFTPLUS, GATE_SIGMOID, LAW_A_DYN, LAW_A_FIX, LAW_MU,
                     dyn_rate, gate_act as activate_gate, group_index)


def phase_bmm(clock, feature_transposed):
    """Preserve FP32/FP64 phase accumulation under model autocast."""
    with torch.autocast(device_type=clock.device.type, enabled=False):
        return torch.bmm(clock, feature_transposed)


def _group(value, heads, dtype):
    if value is None:
        return None
    if heads % value.shape[0]:
        raise ValueError("static bias groups must divide heads")
    return value.to(dtype)[group_index(heads, value.shape[0], device=value.device)]


def _rows(value, resolution, heads, rows, name):
    if resolution is None:
        if heads == rows:
            raise ValueError(f"{name}_res must be 'n' or 'h' when N == H")
        resolution = "n" if value.shape[1] == rows else "h"
    if resolution == "n" and value.shape[1] == rows:
        return value[:, None]
    if resolution == "h" and value.shape[1] == heads:
        return value[:, :, None]
    raise ValueError(f"{name}_res does not match tensor geometry")


def _rotate(value, angle):
    first, second = value[..., 0::2, :], value[..., 1::2, :]
    cosine, sine = angle.cos(), angle.sin()
    return torch.stack((first * cosine - second * sine,
                        first * sine + second * cosine), -2).flatten(-3, -2)


def _chunk(initial, previous, x, write, read, clock, c1, c2, s1, s2, lam,
           *, sequential=False):
    log_a = c1[None, ..., None] * s1[:, :, None] + c2[None, ..., None] * s2[:, :, None]
    transition = log_a.exp()
    raw = x[:, :, :, None] * write[:, :, None]
    drive = clock[:, :, :, None] * raw
    if lam is not None:
        prior = torch.cat((previous[..., None], raw[..., :-1]), -1)
        weight = lam[:, :, None, None]
        drive = clock[:, :, :, None] * (weight * raw + (1 - weight) * transition * prior)
    if sequential:
        outputs = []
        hidden = initial
        for t in range(x.shape[-1]):
            hidden = transition[..., t] * hidden + drive[..., t]
            outputs.append((hidden * read[..., t][:, :, None]).sum(-1))
        return torch.stack(outputs, -1), hidden, raw[..., -1]
    # Composition (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1).
    a, b = transition, drive
    stride = 1
    while stride < x.shape[-1]:
        b = torch.cat((b[..., :stride], b[..., stride:] + a[..., stride:] * b[..., :-stride]), -1)
        a = torch.cat((a[..., :stride], a[..., stride:] * a[..., :-stride]), -1)
        stride *= 2
    hidden = b + a * initial[..., None]
    return (hidden * read[:, :, None]).sum(-2), hidden[..., -1], raw[..., -1]


def mh_torch_scan(
    x, delta, A_log, B_t, C_t, *, h_init=None,
    delta_w=None, delta_softplus=True, delta_bias=None,
    row_law=None, row_rot=None, a_dyn=None, a_dyn_res=None,
    dyn_act=DYN_SOFTPLUS, dyn_eps=0.0, mu=None, mu_gain=None,
    mu_res=None, mu_init=None, mu_prediff=False,
    g_B=None, g_C=None, gate_act=GATE_SIGMOID,
    theta=None, nu=None, theta_delta=False, theta_h=None, theta_init=None,
    beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None,
    trap_lam=None, write_prev=None, d_skip=None, z=None,
    n_off=0, n_rows=None, chunk_size=32, row_tile=32, channel_tile=16,
    max_working_elements=1 << 20, checkpoint_chunks=True, sequential=False,
    _tile_internal=False, _return_write=False,
):
    """Return `(Y, h_final)` in the scan's folded state coordinates.

    Layout and equations match `mh_scan`; raw maps are time-last. Row windows
    preserve hidden coordinates outside the selected slice. IO uses x.dtype;
    hidden accumulation is fp32, or fp64 when x is fp64. Angles use interleaved
    pairs, as the MH kernel ABI does. `theta_init`, `mu_init` and `write_prev`
    carry their ordinary differentiable meaning. This function does not return
    a complete module cache; the module also owns convolution/phase history.
    """
    # Checkpoint preprocessing with the recurrence, not just the hidden scan:
    # head-dependent maps, rates and phases must never span the full sequence.
    options = dict(locals())
    time_tile = min(chunk_size, max(1, max_working_elements //
                    (x.shape[0] * x.shape[1] * A_log.shape[-1])))
    if not _tile_internal and x.shape[-1] > time_tile and time_tile > 0:
        time_keys = ("x", "delta", "B_t", "C_t", "delta_w", "a_dyn", "mu",
                     "g_B", "g_C", "theta", "z", "trap_lam")
        hidden, previous, phase, prior_mu = h_init, write_prev, theta_init, mu_init
        wd = torch.float64 if x.dtype == torch.float64 else torch.float32
        outputs = []
        for t in range(0, x.shape[-1], time_tile):
            stop = min(t + time_tile, x.shape[-1])
            kw = dict(options)
            for key in time_keys:
                value = options[key]
                kw[key] = None if value is None else value[..., t:stop]
            kw.update(h_init=hidden, write_prev=previous, theta_init=phase, mu_init=prior_mu,
                      _tile_internal=True, _return_write=True, checkpoint_chunks=False)
            if checkpoint_chunks and torch.is_grad_enabled():
                y, hidden, previous = checkpoint(mh_torch_scan, use_reentrant=False, **kw)
            else:
                y, hidden, previous = mh_torch_scan(**kw)
            outputs.append(y)
            if theta is not None or nu is not None:
                raw = delta[..., t:stop].to(wd)
                if delta_bias is not None:
                    raw = raw + delta_bias.to(wd)[None, :, None]
                clock = F.softplus(raw) if delta_softplus else raw
                addition = x.new_zeros(x.shape[0], x.shape[1], A_log.shape[-1] // 2, dtype=wd)
                if theta is not None:
                    feature = theta[..., t:stop].to(wd)
                    if feature.dim() == 4:          # theta_per_head
                        addition = (torch.einsum("bhl,bhkl->bhk", clock, feature) if theta_delta
                                    else feature.sum(-1))
                    else:
                        addition = (phase_bmm(clock, feature.transpose(1, 2)) if theta_delta
                                    else feature.sum(-1)[:, None].expand_as(addition))
                if nu is not None:
                    addition = addition + nu.to(wd)[None] * (stop - t)
                phase = addition if phase is None else phase + addition
            if mu is not None and not mu_prediff:
                prior_mu = mu[..., stop - 1]
        result = (torch.cat(outputs, -1), hidden)
        return (*result, previous) if _return_write else result
    batch, heads, channels, length = x.shape
    rows = A_log.shape[-1]
    n_rows = rows - n_off if n_rows is None else n_rows
    if n_off < 0 or n_rows <= 0 or n_off + n_rows > rows:
        raise ValueError("invalid n_off/n_rows window")
    for name, size in (("chunk_size", chunk_size), ("row_tile", row_tile),
                       ("channel_tile", channel_tile), ("max_working_elements", max_working_elements)):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if A_log.shape != (heads, channels, rows):
        raise ValueError("A_log must have shape [H,P,N]")
    if delta.shape != (batch, heads, length) or B_t.shape != (batch, rows, length) or C_t.shape != B_t.shape:
        raise ValueError("delta/B_t/C_t geometry does not match x and A_log")
    wd = torch.float64 if x.dtype == torch.float64 else torch.float32
    cast = lambda value: None if value is None else value.to(wd)
    initial = x.new_zeros(batch, heads, channels, rows, dtype=wd) if h_init is None else cast(h_init)
    if initial.shape != (batch, heads, channels, rows):
        raise ValueError("h_init must have shape [B,H,P,N]")
    if length == 0:
        return x.new_empty(x.shape), initial
    xf, dt = cast(x), cast(delta)
    if delta_bias is not None:
        dt = dt + cast(delta_bias)[None, :, None]
    dt = F.softplus(dt) if delta_softplus else dt
    if delta_w is None:
        clock = dt[:, :, None]
    else:
        clock = cast(delta_w)
        clock = F.softplus(clock) if delta_softplus else clock
        if clock.ndim == 3:
            clock = clock[:, :, None]
        if clock.shape not in ((batch, heads, 1, length), (batch, heads, channels, length)):
            raise ValueError("delta_w must be [B,H,L] or [B,H,P,L]")
    law = (torch.full((rows,), LAW_A_FIX, dtype=torch.int32, device=x.device)
           if row_law is None else row_law)
    static_a = _group(beta_a, heads, wd)
    c1 = -(cast(A_log) + (0 if static_a is None else static_a[:, None])).exp()
    c2 = torch.zeros_like(c1) if mu_gain is None else torch.where(
        (law == LAW_MU)[None, None], cast(mu_gain), 0)
    s1 = dt[:, :, None].expand(batch, heads, rows, length)
    if a_dyn is not None:
        rate = _rows(dyn_rate(cast(a_dyn), dyn_act, dyn_eps), a_dyn_res, heads, rows, "a_dyn")
        s1 = torch.where((law == LAW_A_DYN)[None, None, :, None], s1 * rate, s1)
    s2 = torch.zeros_like(s1)
    if mu is not None:
        value = _rows(cast(mu), mu_res, heads, rows, "mu").expand_as(s1)
        if mu_prediff:
            if mu_init is not None:
                raise ValueError("mu_prediff and mu_init are mutually exclusive")
            difference = value
        else:
            start = value[..., :1] * 0 if mu_init is None else _rows(
                cast(mu_init)[..., None], mu_res, heads, rows, "mu").expand(batch, heads, rows, 1)
            difference = value - torch.cat((start, value[..., :-1]), -1)
        s2 = torch.where((law == LAW_MU)[None, None, :, None], difference, 0)
    write, read = cast(B_t)[:, None], cast(C_t)[:, None]
    for name, bias in (("write", beta_b), ("read", beta_c)):
        bias = _group(bias, heads, wd)
        if bias is not None:
            if name == "write":
                write = write + bias[None, :, :, None]
            else:
                read = read + bias[None, :, :, None]
    if theta is not None or nu is not None:
        if rows % 2:
            raise ValueError("rotation requires even N")
        increments = xf.new_zeros(batch, heads, rows // 2, length)
        if theta is not None:
            th = cast(theta) if theta.dim() == 4 else cast(theta)[:, None]
            increments = increments + th * (dt[:, :, None] if theta_delta else 1)
        if nu is not None:
            increments = increments + cast(nu)[None, :, :, None]
        angles = increments.cumsum(-1)
        if theta_init is not None:
            angles = angles + cast(theta_init)[..., None]
        mask = (torch.ones(rows, dtype=torch.bool, device=x.device) if row_rot is None else row_rot)[None, None, :, None]
        write = torch.where(mask, _rotate(write, angles), write)
        read_angles = angles if theta_h is None else angles + cast(theta_h)[None, :, :, None]
        read = torch.where(mask, _rotate(read, read_angles), read)
    for name, stream, bias in (("write", g_B, beta_gB), ("read", g_C, beta_gC)):
        if stream is None:
            continue
        gate = cast(stream)[:, None]
        bias = _group(bias, heads, wd)
        if bias is not None:
            gate = gate + bias[None, :, :, None]
        gate = activate_gate(gate, gate_act)
        if name == "write":
            write = write * gate
        else:
            read = read * gate
    write, read = write.expand(batch, heads, rows, length), read.expand(batch, heads, rows, length)
    lam = cast(trap_lam)
    previous = torch.zeros_like(initial) if write_prev is None else cast(write_prev)
    width = min(channel_tile, max(1, max_working_elements //
                                  (batch * heads * min(n_rows, row_tile) * min(length, chunk_size))))
    outputs, final_channels = [], []
    for p in range(0, channels, width):
        pend = min(p + width, channels)
        summed, final_rows = None, []
        for n in range(n_off, n_off + n_rows, row_tile):
            end = min(n + row_tile, n_off + n_rows)
            hidden, prior = initial[:, :, p:pend, n:end], previous[:, :, p:pend, n:end]
            chunks = []
            for t in range(0, length, chunk_size):
                stop = min(t + chunk_size, length)
                inputs = (hidden, prior, xf[:, :, p:pend, t:stop], write[:, :, n:end, t:stop],
                          read[:, :, n:end, t:stop],
                          clock[..., t:stop] if clock.shape[2] == 1 else clock[:, :, p:pend, t:stop],
                          c1[:, p:pend, n:end], c2[:, p:pend, n:end],
                          s1[:, :, n:end, t:stop], s2[:, :, n:end, t:stop],
                          None if lam is None else lam[:, :, t:stop])
                if checkpoint_chunks and torch.is_grad_enabled() and any(v is not None and v.requires_grad for v in inputs):
                    out, hidden, prior = checkpoint(_chunk, *inputs, sequential=sequential, use_reentrant=False)
                else:
                    out, hidden, prior = _chunk(*inputs, sequential=sequential)
                chunks.append(out)
            part = torch.cat(chunks, -1)
            summed = part if summed is None else summed + part
            final_rows.append(hidden)
        outputs.append(summed)
        final_channels.append(torch.cat((initial[:, :, p:pend, :n_off], *final_rows,
                                         initial[:, :, p:pend, n_off + n_rows:]), -1))
    y, hidden = torch.cat(outputs, 2), torch.cat(final_channels, 2)
    if d_skip is not None:
        y = y + cast(d_skip)[None, :, None, None] * xf
    if z is not None:
        y = y * F.silu(cast(z))
    result = (y.to(x.dtype), hidden)
    if _return_write:
        last_write = xf[..., -1, None] * write[..., -1][:, :, None]
        return (*result, last_write.contiguous())
    return result
