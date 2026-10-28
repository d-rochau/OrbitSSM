"""Plain-Triton depthwise causal conv1d (+ fused SiLU), forward and backward: the portable conv.

y[b, c, t] = act( bias[c] + sum_{j<W} w[c, j] x[b, c, t - (W-1) + j] ),   x[.., s<0] = 0,   act = SiLU or identity.

Replaces the ``causal_conv1d`` package (a compiled CUDA extension) where that cannot run: ``kernel_target="portable"``,
a non-CUDA Triton backend, or ``conv_kernel="triton"`` on any backend.  Any width W (a constexpr), any dtype
float16 / bfloat16 / float32 / float64, any strides of x and dy (channel-last or channel-major, narrow views included): the
kernels take strides, so no copy is made; the memory order of dx follows x.

Single-token decode (`conv_step_triton`): shift the [B, C, W] window by one, append the new input, one dot over the window, SiLU; one launch.

Chunked continuation: the callers prepend the carried d_conv - 1 raw inputs and drop the first d_conv - 1 outputs
(`layers.conv` idiom), so the kernel needs no initial-state argument.

Arithmetic (the bounds of the conv parity test): products and sums in fp32 (fp64 for fp64 tensors), one rounding to the output
dtype; SiLU in the accumulator dtype.
Backward: dz = dy * act'(z) with z RECOMPUTED from x (nothing but x is saved);
  dx[c, l]   = sum_j w[c, j] dz[c, l + W - 1 - j]                          (one owner per element, no atomics)
  dw[c, j]   = sum_{b, t} dz[b, c, t] x[b, c, t - W + 1 + j],   db[c] = sum_{b, t} dz[b, c, t]
dw / db are a DETERMINISTIC TWO-STAGE reduction (no atomics): stage one, each program (b, channel block, token chunk of CH tokens) sums its chunk in
a fixed sequential order into a fp32 partial row; stage two, a torch sum over the partial rows in a fixed order.  Same inputs give the same bits.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

__all__ = ["conv_fwd_triton", "conv_bwd_triton", "causal_conv_triton", "conv_step_triton", "supported"]

_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_CH = 256          # tokens per dw program (the fixed order of stage one)


def supported(x: torch.Tensor) -> bool:
    return x.dtype in _FLOATS and (x.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1")


@triton.jit
def _silu_grad(z, ACC: tl.constexpr):
    s = 1.0 / (1.0 + tl.exp(-z))
    return s * (1.0 + z * (1.0 - s))


@triton.jit
def _pre(X, Wt, Bi, pb, offs_c, cm, t, C, L, sxb, sxc, sxl, W: tl.constexpr, HAS_BIAS: tl.constexpr, ACC: tl.constexpr):
    """z[c, t] = bias + sum_j w[c, j] x[c, t - W + 1 + j] for the token vector t (positions outside [0, L) read zero)."""
    z = tl.zeros((offs_c.shape[0], t.shape[0]), ACC)
    for j in tl.static_range(W):
        s = t - (W - 1) + j
        m = cm[:, None] & (s[None, :] >= 0) & (s[None, :] < L)
        xv = tl.load(X + pb * sxb + offs_c[:, None] * sxc + s[None, :] * sxl, mask=m, other=0.0).to(ACC)
        wv = tl.load(Wt + offs_c * W + j, mask=cm, other=0.0).to(ACC)
        z += xv * wv[:, None]
    if HAS_BIAS:
        z += tl.load(Bi + offs_c, mask=cm, other=0.0).to(ACC)[:, None]
    return z


@triton.jit
def _fwd_kernel(X, Wt, Bi, Y, C, L, sxb, sxc, sxl, syb, syc, syl,
                W: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr,
                BC: tl.constexpr, BL: tl.constexpr, LFAST: tl.constexpr):
    if LFAST:       # token blocks fastest: consecutive programs read adjacent rows (shared halo, L2-local)
        pl, pcb, pb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    else:
        pb, pcb, pl = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_c = pcb * BC + tl.arange(0, BC)
    offs_l = pl * BL + tl.arange(0, BL)
    cm = offs_c < C
    z = _pre(X, Wt, Bi, pb, offs_c, cm, offs_l, C, L, sxb, sxc, sxl, W, HAS_BIAS, ACC)
    if SILU:
        z = z / (1.0 + tl.exp(-z))
    tl.store(Y + pb * syb + offs_c[:, None] * syc + offs_l[None, :] * syl, z.to(Y.dtype.element_ty),
             mask=cm[:, None] & (offs_l[None, :] < L))


@triton.jit
def _dz_at(X, Wt, Bi, DY, pb, offs_c, cm, t, C, L, sxb, sxc, sxl, sdb, sdc, sdl,
           W: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr):
    """dz[c, t] = dy[c, t] act'(z[c, t]); zero outside [0, L)."""
    tm = cm[:, None] & (t[None, :] >= 0) & (t[None, :] < L)
    dy = tl.load(DY + pb * sdb + offs_c[:, None] * sdc + t[None, :] * sdl, mask=tm, other=0.0).to(ACC)
    if SILU:
        z = _pre(X, Wt, Bi, pb, offs_c, cm, t, C, L, sxb, sxc, sxl, W, HAS_BIAS, ACC)
        dy = dy * _silu_grad(z, ACC)
    return tl.where(tm, dy, 0.0)


@triton.jit
def _dx_kernel(X, Wt, Bi, DY, DX, C, L, sxb, sxc, sxl, sdb, sdc, sdl, sob, soc, sol,
               W: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr,
               BC: tl.constexpr, BL: tl.constexpr):
    pb = tl.program_id(0)
    offs_c = tl.program_id(1) * BC + tl.arange(0, BC)
    offs_l = tl.program_id(2) * BL + tl.arange(0, BL)
    cm = offs_c < C
    acc = tl.zeros((BC, BL), ACC)
    for j in tl.static_range(W):
        t = offs_l + (W - 1 - j)
        dz = _dz_at(X, Wt, Bi, DY, pb, offs_c, cm, t, C, L, sxb, sxc, sxl, sdb, sdc, sdl, W, HAS_BIAS, SILU, ACC)
        wv = tl.load(Wt + offs_c * W + j, mask=cm, other=0.0).to(ACC)
        acc += dz * wv[:, None]
    tl.store(DX + pb * sob + offs_c[:, None] * soc + offs_l[None, :] * sol, acc.to(DX.dtype.element_ty),
             mask=cm[:, None] & (offs_l[None, :] < L))


@triton.jit
def _dw_kernel(X, Wt, Bi, DY, PART, C, L, NCH, sxb, sxc, sxl, sdb, sdc, sdl,
               W: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr,
               BC: tl.constexpr, BL: tl.constexpr, CH: tl.constexpr):
    """Stage one of dw / db: PART[(b * NCH + chunk), c, 0..W-1] = dw partials, [.., W] = db partial (fp32/fp64 ACC)."""
    pb = tl.program_id(0)
    offs_c = tl.program_id(1) * BC + tl.arange(0, BC)
    pk = tl.program_id(2)
    cm = offs_c < C
    row = (pb * NCH + pk) * C
    t0 = pk * CH
    for j in tl.static_range(W):
        acc = tl.zeros((BC,), ACC)
        for l0 in range(t0, t0 + CH, BL):
            t = l0 + tl.arange(0, BL)
            dz = _dz_at(X, Wt, Bi, DY, pb, offs_c, cm, t, C, L, sxb, sxc, sxl, sdb, sdc, sdl, W, HAS_BIAS, SILU, ACC)
            s = t - (W - 1) + j
            m = cm[:, None] & (s[None, :] >= 0) & (s[None, :] < L)
            xv = tl.load(X + pb * sxb + offs_c[:, None] * sxc + s[None, :] * sxl, mask=m, other=0.0).to(ACC)
            acc += tl.sum(dz * xv, axis=1)
        tl.store(PART + (row + offs_c) * (W + 1) + j, acc, mask=cm)
    accb = tl.zeros((BC,), ACC)
    for l0 in range(t0, t0 + CH, BL):
        t = l0 + tl.arange(0, BL)
        dz = _dz_at(X, Wt, Bi, DY, pb, offs_c, cm, t, C, L, sxb, sxc, sxl, sdb, sdc, sdl, W, HAS_BIAS, SILU, ACC)
        accb += tl.sum(dz, axis=1)
    tl.store(PART + (row + offs_c) * (W + 1) + W, accb, mask=cm)


# ── the streaming ("rows") kernels: width 4, channel-last (stride_c == 1) ──────────────────────────────────────────────────────────────────
# One program owns a block of BC channels and a chunk of TL consecutive tokens and walks the tokens one ROW at a time, keeping the last W - 1 input
# rows (forward) / the next W - 1 pre-activation gradients (backward) in registers: every x and dy element is read once, every output written once,
# one exp per output (the tile kernels above re-read each element W times and recompute SiLU W times in the backward).
@triton.jit
def _row(P, base, s, offs_c, cm, sl, L):
    """Row s of a [.., L, C] channel-last tensor for the channel vector offs_c (zero outside [0, L))."""
    return tl.load(P + base + s * sl + offs_c, mask=cm & (s >= 0) & (s < L), other=0.0)


@triton.jit
def _fwd_rows(X, Wt, Bi, Y, C, L, sxb, sxl, syb, syl, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr,
              BC: tl.constexpr, TL: tl.constexpr, NS: tl.constexpr):
    pb = tl.program_id(0)
    offs_c = tl.program_id(1) * BC + tl.arange(0, BC)
    t0 = tl.program_id(2) * TL
    cm = offs_c < C
    w0 = tl.load(Wt + offs_c * 4 + 0, mask=cm, other=0.0).to(ACC)
    w1 = tl.load(Wt + offs_c * 4 + 1, mask=cm, other=0.0).to(ACC)
    w2 = tl.load(Wt + offs_c * 4 + 2, mask=cm, other=0.0).to(ACC)
    w3 = tl.load(Wt + offs_c * 4 + 3, mask=cm, other=0.0).to(ACC)
    bias = tl.zeros((BC,), ACC)
    if HAS_BIAS:
        bias = tl.load(Bi + offs_c, mask=cm, other=0.0).to(ACC)
    xb = pb * sxb
    x1 = _row(X, xb, t0 - 3, offs_c, cm, sxl, L).to(ACC)
    x2 = _row(X, xb, t0 - 2, offs_c, cm, sxl, L).to(ACC)
    x3 = _row(X, xb, t0 - 1, offs_c, cm, sxl, L).to(ACC)
    for i in tl.range(0, TL, num_stages=NS):
        t = t0 + i
        x4 = _row(X, xb, t, offs_c, cm, sxl, L).to(ACC)
        z = w0 * x1 + w1 * x2 + w2 * x3 + w3 * x4 + bias
        if SILU:
            z = z / (1.0 + tl.exp(-z))
        tl.store(Y + pb * syb + t * syl + offs_c, z.to(Y.dtype.element_ty), mask=cm & (t < L))
        x1 = x2
        x2 = x3
        x3 = x4


@triton.jit
def _dz_row(X, DY, Wt0, Wt1, Wt2, Wt3, bias, pb, s, offs_c, cm, sxb, sxl, sdb, sdl, L, SILU: tl.constexpr, ACC: tl.constexpr):
    """dz at token s from scratch (4 x rows + dy): the backward's halo."""
    xb = pb * sxb
    z = (Wt0 * _row(X, xb, s - 3, offs_c, cm, sxl, L).to(ACC) + Wt1 * _row(X, xb, s - 2, offs_c, cm, sxl, L).to(ACC)
         + Wt2 * _row(X, xb, s - 1, offs_c, cm, sxl, L).to(ACC) + Wt3 * _row(X, xb, s, offs_c, cm, sxl, L).to(ACC) + bias)
    dy = _row(DY, pb * sdb, s, offs_c, cm, sdl, L).to(ACC)
    if SILU:
        dy = dy * _silu_grad(z, ACC)
    return tl.where(cm & (s >= 0) & (s < L), dy, 0.0)


@triton.jit
def _bwd_step(DX, obase, t, t0, dy, xnew, d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db, w0, w1, w2, w3, bias, offs_c, cm, sol,
              SILU: tl.constexpr, NEED_DX: tl.constexpr, ACC: tl.constexpr):
    """One reverse token t of `_bwd_rows`: z_t = w0 x_{t-3} + w1 x_{t-2} + w2 x_{t-1} + w3 x_t + b from the register window, dz_t, dx_t, the dw / db sums, the window shift."""
    valid = (t >= t0) & (t >= 0)
    z = w0 * xd + w1 * xc + w2 * xbm + w3 * xa + bias
    if SILU:
        dy = dy * _silu_grad(z, ACC)
    dz = tl.where(cm & valid, dy, 0.0)
    if NEED_DX:
        dx = w0 * d3 + w1 * d2 + w2 * d1 + w3 * dz
        tl.store(DX + obase + t * sol + offs_c, dx.to(DX.dtype.element_ty), mask=cm & valid)
    return dz, d1, d2, xbm, xc, xd, xnew, dw0 + dz * xd, dw1 + dz * xc, dw2 + dz * xbm, dw3 + dz * xa, db + dz


@triton.jit
def _bwd_rows(X, Wt, Bi, DY, DX, PART, C, L, NCH, sxb, sxl, sdb, sdl, sob, sol, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
              ACC: tl.constexpr, BC: tl.constexpr, TL: tl.constexpr, NS: tl.constexpr, NEED_DX: tl.constexpr):
    """dx and the stage-one dw / db partials in ONE reverse pass over the chunk [t0, t0 + TL): per token one dz (SiLU' recomputed from x), one dx row."""
    pb = tl.program_id(0)
    offs_c = tl.program_id(1) * BC + tl.arange(0, BC)
    pk = tl.program_id(2)
    t0 = pk * TL
    t1 = tl.minimum(t0 + TL, L)
    cm = offs_c < C
    w0 = tl.load(Wt + offs_c * 4 + 0, mask=cm, other=0.0).to(ACC)
    w1 = tl.load(Wt + offs_c * 4 + 1, mask=cm, other=0.0).to(ACC)
    w2 = tl.load(Wt + offs_c * 4 + 2, mask=cm, other=0.0).to(ACC)
    w3 = tl.load(Wt + offs_c * 4 + 3, mask=cm, other=0.0).to(ACC)
    bias = tl.zeros((BC,), ACC)
    if HAS_BIAS:
        bias = tl.load(Bi + offs_c, mask=cm, other=0.0).to(ACC)
    # halo: dz at t1, t1 + 1, t1 + 2 (zero past L)
    d1 = _dz_row(X, DY, w0, w1, w2, w3, bias, pb, t1, offs_c, cm, sxb, sxl, sdb, sdl, L, SILU, ACC)
    d2 = _dz_row(X, DY, w0, w1, w2, w3, bias, pb, t1 + 1, offs_c, cm, sxb, sxl, sdb, sdl, L, SILU, ACC)
    d3 = _dz_row(X, DY, w0, w1, w2, w3, bias, pb, t1 + 2, offs_c, cm, sxb, sxl, sdb, sdl, L, SILU, ACC)
    xb = pb * sxb
    # x window for the first (last) token t = t1 - 1: x_t, x_{t-1}, x_{t-2}, x_{t-3}
    xa = _row(X, xb, t1 - 1, offs_c, cm, sxl, L).to(ACC)
    xbm = _row(X, xb, t1 - 2, offs_c, cm, sxl, L).to(ACC)
    xc = _row(X, xb, t1 - 3, offs_c, cm, sxl, L).to(ACC)
    xd = _row(X, xb, t1 - 4, offs_c, cm, sxl, L).to(ACC)
    dw0 = tl.zeros((BC,), ACC)
    dw1 = tl.zeros((BC,), ACC)
    dw2 = tl.zeros((BC,), ACC)
    dw3 = tl.zeros((BC,), ACC)
    db = tl.zeros((BC,), ACC)
    # four tokens per iteration: their dy and x[t - 4] rows do not depend on the recurrence, so all eight loads are issued before the first is used
    # (memory-level parallelism: the one-token loop was bound by the bytes in flight, not by bandwidth, at narrow C)
    for i in tl.range(0, TL, 4, num_stages=NS):
        ta = t1 - 1 - i
        dya = _row(DY, pb * sdb, ta, offs_c, cm, sdl, L).to(ACC)
        dyb = _row(DY, pb * sdb, ta - 1, offs_c, cm, sdl, L).to(ACC)
        dyc = _row(DY, pb * sdb, ta - 2, offs_c, cm, sdl, L).to(ACC)
        dyd = _row(DY, pb * sdb, ta - 3, offs_c, cm, sdl, L).to(ACC)
        xna = _row(X, xb, ta - 4, offs_c, cm, sxl, L).to(ACC)
        xnb = _row(X, xb, ta - 5, offs_c, cm, sxl, L).to(ACC)
        xnc = _row(X, xb, ta - 6, offs_c, cm, sxl, L).to(ACC)
        xnd = _row(X, xb, ta - 7, offs_c, cm, sxl, L).to(ACC)
        d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db = _bwd_step(
            DX, pb * sob, ta, t0, dya, xna, d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db, w0, w1, w2, w3, bias, offs_c, cm, sol,
            SILU, NEED_DX, ACC)
        d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db = _bwd_step(
            DX, pb * sob, ta - 1, t0, dyb, xnb, d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db, w0, w1, w2, w3, bias, offs_c, cm, sol,
            SILU, NEED_DX, ACC)
        d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db = _bwd_step(
            DX, pb * sob, ta - 2, t0, dyc, xnc, d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db, w0, w1, w2, w3, bias, offs_c, cm, sol,
            SILU, NEED_DX, ACC)
        d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db = _bwd_step(
            DX, pb * sob, ta - 3, t0, dyd, xnd, d1, d2, d3, xa, xbm, xc, xd, dw0, dw1, dw2, dw3, db, w0, w1, w2, w3, bias, offs_c, cm, sol,
            SILU, NEED_DX, ACC)
    row = (pb * NCH + pk) * C
    base = (row + offs_c) * 5
    tl.store(PART + base + 0, dw0, mask=cm)
    tl.store(PART + base + 1, dw1, mask=cm)
    tl.store(PART + base + 2, dw2, mask=cm)
    tl.store(PART + base + 3, dw3, mask=cm)
    tl.store(PART + base + 4, db, mask=cm)


# ── the fused tile backward: width 4, channel-MAJOR x (stride_l == 1) ───────────────────────────────────────────────────────────────────
# Tokens are the contiguous axis, so the row-streaming kernels above (channels contiguous) would read one element per channel row.  One program owns a
# [BC channels x BL tokens] tile and loops over a chunk of CH tokens.  Per tile: the 7 x tiles x[t-3 .. t+3] and the 4 dy tiles dy[t .. t+3] are read ONCE,
# the 4 pre-activations z[t+k] (k = 0..3) come from the shared x tiles, dx[t] = w0 dz[t+3] + w1 dz[t+2] + w2 dz[t+1] + w3 dz[t], and dw / db use dz[t] with
# the same x tiles (dw_j += sum_t dz[t] x[t-3+j]).  The tile pair (`_dx_kernel` + `_dw_kernel`) reads x 4x and re-evaluates every dz four times more.
@triton.jit
def _ld(P, base, offs_c, cm, s, sc, sl, L):
    return tl.load(P + base + offs_c[:, None] * sc + s[None, :] * sl, mask=cm[:, None] & (s[None, :] >= 0) & (s[None, :] < L), other=0.0)


@triton.jit
def _bwd_tile4(X, Wt, Bi, DY, DX, PART, C, L, NCH, sxb, sxc, sxl, sdb, sdc, sdl, sob, soc, sol,
               HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr, BC: tl.constexpr, BL: tl.constexpr, CH: tl.constexpr,
               NEED_DX: tl.constexpr, NEED_DW: tl.constexpr):
    pb = tl.program_id(0)
    offs_c = tl.program_id(1) * BC + tl.arange(0, BC)
    pk = tl.program_id(2)
    cm = offs_c < C
    w0 = tl.load(Wt + offs_c * 4 + 0, mask=cm, other=0.0).to(ACC)[:, None]
    w1 = tl.load(Wt + offs_c * 4 + 1, mask=cm, other=0.0).to(ACC)[:, None]
    w2 = tl.load(Wt + offs_c * 4 + 2, mask=cm, other=0.0).to(ACC)[:, None]
    w3 = tl.load(Wt + offs_c * 4 + 3, mask=cm, other=0.0).to(ACC)[:, None]
    bias = tl.zeros((BC, 1), ACC)
    if HAS_BIAS:
        bias = tl.load(Bi + offs_c, mask=cm, other=0.0).to(ACC)[:, None]
    xb = pb * sxb
    db_ = pb * sdb
    dw0 = tl.zeros((BC,), ACC)
    dw1 = tl.zeros((BC,), ACC)
    dw2 = tl.zeros((BC,), ACC)
    dw3 = tl.zeros((BC,), ACC)
    dbs = tl.zeros((BC,), ACC)
    t0 = pk * CH
    for l0 in range(t0, t0 + CH, BL):
        t = l0 + tl.arange(0, BL)
        xm3 = _ld(X, xb, offs_c, cm, t - 3, sxc, sxl, L).to(ACC)
        xm2 = _ld(X, xb, offs_c, cm, t - 2, sxc, sxl, L).to(ACC)
        xm1 = _ld(X, xb, offs_c, cm, t - 1, sxc, sxl, L).to(ACC)
        x0 = _ld(X, xb, offs_c, cm, t, sxc, sxl, L).to(ACC)
        dy0 = _ld(DY, db_, offs_c, cm, t, sdc, sdl, L).to(ACC)
        if SILU:
            z0 = w0 * xm3 + w1 * xm2 + w2 * xm1 + w3 * x0 + bias
            dy0 = dy0 * _silu_grad(z0, ACC)
        dz0 = tl.where(cm[:, None] & (t[None, :] < L), dy0, 0.0)
        if NEED_DX:
            xp1 = _ld(X, xb, offs_c, cm, t + 1, sxc, sxl, L).to(ACC)
            xp2 = _ld(X, xb, offs_c, cm, t + 2, sxc, sxl, L).to(ACC)
            xp3 = _ld(X, xb, offs_c, cm, t + 3, sxc, sxl, L).to(ACC)
            dy1 = _ld(DY, db_, offs_c, cm, t + 1, sdc, sdl, L).to(ACC)
            dy2 = _ld(DY, db_, offs_c, cm, t + 2, sdc, sdl, L).to(ACC)
            dy3 = _ld(DY, db_, offs_c, cm, t + 3, sdc, sdl, L).to(ACC)
            if SILU:
                dy1 = dy1 * _silu_grad(w0 * xm2 + w1 * xm1 + w2 * x0 + w3 * xp1 + bias, ACC)
                dy2 = dy2 * _silu_grad(w0 * xm1 + w1 * x0 + w2 * xp1 + w3 * xp2 + bias, ACC)
                dy3 = dy3 * _silu_grad(w0 * x0 + w1 * xp1 + w2 * xp2 + w3 * xp3 + bias, ACC)
            dz1 = tl.where(cm[:, None] & (t[None, :] + 1 < L), dy1, 0.0)
            dz2 = tl.where(cm[:, None] & (t[None, :] + 2 < L), dy2, 0.0)
            dz3 = tl.where(cm[:, None] & (t[None, :] + 3 < L), dy3, 0.0)
            dx = w0 * dz3 + w1 * dz2 + w2 * dz1 + w3 * dz0
            tl.store(DX + pb * sob + offs_c[:, None] * soc + t[None, :] * sol, dx.to(DX.dtype.element_ty),
                     mask=cm[:, None] & (t[None, :] < L))
        if NEED_DW:
            dw0 += tl.sum(dz0 * xm3, axis=1)
            dw1 += tl.sum(dz0 * xm2, axis=1)
            dw2 += tl.sum(dz0 * xm1, axis=1)
            dw3 += tl.sum(dz0 * x0, axis=1)
            dbs += tl.sum(dz0, axis=1)
    if NEED_DW:
        base = ((pb * NCH + pk) * C + offs_c) * 5
        tl.store(PART + base + 0, dw0, mask=cm)
        tl.store(PART + base + 1, dw1, mask=cm)
        tl.store(PART + base + 2, dw2, mask=cm)
        tl.store(PART + base + 3, dw3, mask=cm)
        tl.store(PART + base + 4, dbs, mask=cm)


# ---- channel-major, width 4, 8-token vectors (the SSD layer's call: contiguous rows of L tokens) ----
# A token shift along the contiguous axis is a MISALIGNED load (no vector loads in 16-bit), so these kernels only issue aligned 8-wide
# loads (the chunk, its neighbour) and form the shifts in registers: `_un8` splits a [N, 8] tile into its 8 columns (tl.split, thread-local),
# `_pk8` joins them back.  Program = one (b, c) row x one SPAN of tokens, N chunks per step; needs L, the row strides and the base
# pointers aligned to 8 elements (`_v8_ok`).  Same arithmetic as the tile kernels (fp32, one rounding), dw / db in the same two stages.
@triton.jit
def _un8(v, N: tl.constexpr):
    a, b = tl.split(tl.reshape(v, (N, 4, 2)))
    a0, a1 = tl.split(tl.reshape(a, (N, 2, 2)))
    b0, b1 = tl.split(tl.reshape(b, (N, 2, 2)))
    v0, v4 = tl.split(a0)
    v2, v6 = tl.split(a1)
    v1, v5 = tl.split(b0)
    v3, v7 = tl.split(b1)
    return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _pk8(v0, v1, v2, v3, v4, v5, v6, v7, N: tl.constexpr):
    a = tl.reshape(tl.join(tl.join(v0, v4), tl.join(v2, v6)), (N, 4))
    b = tl.reshape(tl.join(tl.join(v1, v5), tl.join(v3, v7)), (N, 4))
    return tl.reshape(tl.join(a, b), (N, 8))


@triton.jit
def _act8(z, SILU: tl.constexpr):
    if SILU:
        z = z / (1.0 + tl.exp(-z))
    return z


@triton.jit
def _dact8(dy, z, SILU: tl.constexpr, ACC: tl.constexpr):
    if SILU:
        dy = dy * _silu_grad(z, ACC)
    return dy


@triton.jit
def _wb8(Wt, Bi, c, HAS_BIAS: tl.constexpr, ACC: tl.constexpr):
    bb = tl.zeros((), ACC)
    if HAS_BIAS:
        bb = tl.load(Bi + c).to(ACC)
    return (tl.load(Wt + c * 4).to(ACC), tl.load(Wt + c * 4 + 1).to(ACC), tl.load(Wt + c * 4 + 2).to(ACC),
            tl.load(Wt + c * 4 + 3).to(ACC), bb)


@triton.jit
def _fwd_v8(X, Wt, Bi, Y, C, L8, SPAN, sxb, sxc, syb, syc, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr,
            N: tl.constexpr):
    r = tl.program_id(0)
    ps = tl.program_id(1)
    pb = r // C
    c = r % C
    w0, w1, w2, w3, bb = _wb8(Wt, Bi, c, HAS_BIAS, ACC)
    xo = tl.multiple_of(pb.to(tl.int64) * sxb + c.to(tl.int64) * sxc, 8)
    yo = tl.multiple_of(pb.to(tl.int64) * syb + c.to(tl.int64) * syc, 8)
    i = tl.arange(0, 8)[None, :]
    for l0 in range(ps * SPAN, ps * SPAN + SPAN, N * 8):
        k = l0 // 8 + tl.arange(0, N)[:, None]
        t = k * 8 + i
        ok = (k < L8) & (i >= 0)
        x0, x1, x2, x3, x4, x5, x6, x7 = _un8(tl.load(X + xo + t, mask=ok, other=0.0).to(ACC), N)
        p0, p1, p2, p3, p4, p5, p6, p7 = _un8(tl.load(X + xo + t - 8, mask=(k >= 1) & (k - 1 < L8) & (i >= 0), other=0.0).to(ACC), N)
        y0 = _act8(bb + w0 * p5 + w1 * p6 + w2 * p7 + w3 * x0, SILU)
        y1 = _act8(bb + w0 * p6 + w1 * p7 + w2 * x0 + w3 * x1, SILU)
        y2 = _act8(bb + w0 * p7 + w1 * x0 + w2 * x1 + w3 * x2, SILU)
        y3 = _act8(bb + w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3, SILU)
        y4 = _act8(bb + w0 * x1 + w1 * x2 + w2 * x3 + w3 * x4, SILU)
        y5 = _act8(bb + w0 * x2 + w1 * x3 + w2 * x4 + w3 * x5, SILU)
        y6 = _act8(bb + w0 * x3 + w1 * x4 + w2 * x5 + w3 * x6, SILU)
        y7 = _act8(bb + w0 * x4 + w1 * x5 + w2 * x6 + w3 * x7, SILU)
        tl.store(Y + yo + t, _pk8(y0, y1, y2, y3, y4, y5, y6, y7, N).to(Y.dtype.element_ty), mask=ok)


@triton.jit
def _bwd_v8(X, Wt, Bi, DY, DX, PART, C, L8, SPAN, NSP, sxb, sxc, sdb, sdc, sob, soc,
            HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr, N: tl.constexpr, NEED_DX: tl.constexpr,
            NEED_DW: tl.constexpr):
    """dz of chunk k and of the first 3 tokens of chunk k + 1 (dx[t] reads dz[t .. t + 3]); x of chunks k - 1, k, k + 1."""
    r = tl.program_id(0)
    ps = tl.program_id(1)
    pb = r // C
    c = r % C
    w0, w1, w2, w3, bb = _wb8(Wt, Bi, c, HAS_BIAS, ACC)
    xo = tl.multiple_of(pb.to(tl.int64) * sxb + c.to(tl.int64) * sxc, 8)
    do = tl.multiple_of(pb.to(tl.int64) * sdb + c.to(tl.int64) * sdc, 8)
    oo = tl.multiple_of(pb.to(tl.int64) * sob + c.to(tl.int64) * soc, 8)
    i = tl.arange(0, 8)[None, :]
    a0 = tl.zeros((N,), ACC)
    a1 = tl.zeros((N,), ACC)
    a2 = tl.zeros((N,), ACC)
    a3 = tl.zeros((N,), ACC)
    ab = tl.zeros((N,), ACC)
    for l0 in range(ps * SPAN, ps * SPAN + SPAN, N * 8):
        k = l0 // 8 + tl.arange(0, N)[:, None]
        t = k * 8 + i
        ok = (k < L8) & (i >= 0)
        okn = (k + 1 < L8) & (i >= 0)
        x0, x1, x2, x3, x4, x5, x6, x7 = _un8(tl.load(X + xo + t, mask=ok, other=0.0).to(ACC), N)
        p0, p1, p2, p3, p4, p5, p6, p7 = _un8(tl.load(X + xo + t - 8, mask=(k >= 1) & (k - 1 < L8) & (i >= 0), other=0.0).to(ACC), N)
        g0, g1, g2, g3, g4, g5, g6, g7 = _un8(tl.load(DY + do + t, mask=ok, other=0.0).to(ACC), N)
        d0 = _dact8(g0, bb + w0 * p5 + w1 * p6 + w2 * p7 + w3 * x0, SILU, ACC)
        d1 = _dact8(g1, bb + w0 * p6 + w1 * p7 + w2 * x0 + w3 * x1, SILU, ACC)
        d2 = _dact8(g2, bb + w0 * p7 + w1 * x0 + w2 * x1 + w3 * x2, SILU, ACC)
        d3 = _dact8(g3, bb + w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3, SILU, ACC)
        d4 = _dact8(g4, bb + w0 * x1 + w1 * x2 + w2 * x3 + w3 * x4, SILU, ACC)
        d5 = _dact8(g5, bb + w0 * x2 + w1 * x3 + w2 * x4 + w3 * x5, SILU, ACC)
        d6 = _dact8(g6, bb + w0 * x3 + w1 * x4 + w2 * x5 + w3 * x6, SILU, ACC)
        d7 = _dact8(g7, bb + w0 * x4 + w1 * x5 + w2 * x6 + w3 * x7, SILU, ACC)
        if NEED_DX:
            n0, n1, n2, n3, n4, n5, n6, n7 = _un8(tl.load(X + xo + t + 8, mask=okn, other=0.0).to(ACC), N)
            h0, h1, h2, h3, h4, h5, h6, h7 = _un8(tl.load(DY + do + t + 8, mask=okn, other=0.0).to(ACC), N)
            e0 = _dact8(h0, bb + w0 * x5 + w1 * x6 + w2 * x7 + w3 * n0, SILU, ACC)
            e1 = _dact8(h1, bb + w0 * x6 + w1 * x7 + w2 * n0 + w3 * n1, SILU, ACC)
            e2 = _dact8(h2, bb + w0 * x7 + w1 * n0 + w2 * n1 + w3 * n2, SILU, ACC)
            q0 = w3 * d0 + w2 * d1 + w1 * d2 + w0 * d3
            q1 = w3 * d1 + w2 * d2 + w1 * d3 + w0 * d4
            q2 = w3 * d2 + w2 * d3 + w1 * d4 + w0 * d5
            q3 = w3 * d3 + w2 * d4 + w1 * d5 + w0 * d6
            q4 = w3 * d4 + w2 * d5 + w1 * d6 + w0 * d7
            q5 = w3 * d5 + w2 * d6 + w1 * d7 + w0 * e0
            q6 = w3 * d6 + w2 * d7 + w1 * e0 + w0 * e1
            q7 = w3 * d7 + w2 * e0 + w1 * e1 + w0 * e2
            tl.store(DX + oo + t, _pk8(q0, q1, q2, q3, q4, q5, q6, q7, N).to(DX.dtype.element_ty), mask=ok)
        if NEED_DW:
            a3 += d0 * x0 + d1 * x1 + d2 * x2 + d3 * x3 + d4 * x4 + d5 * x5 + d6 * x6 + d7 * x7
            a2 += d0 * p7 + d1 * x0 + d2 * x1 + d3 * x2 + d4 * x3 + d5 * x4 + d6 * x5 + d7 * x6
            a1 += d0 * p6 + d1 * p7 + d2 * x0 + d3 * x1 + d4 * x2 + d5 * x3 + d6 * x4 + d7 * x5
            a0 += d0 * p5 + d1 * p6 + d2 * p7 + d3 * x0 + d4 * x1 + d5 * x2 + d6 * x3 + d7 * x4
            ab += d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
    if NEED_DW:
        base = ((pb * NSP + ps) * C + c) * 5
        tl.store(PART + base, tl.sum(a0))
        tl.store(PART + base + 1, tl.sum(a1))
        tl.store(PART + base + 2, tl.sum(a2))
        tl.store(PART + base + 3, tl.sum(a3))
        tl.store(PART + base + 4, tl.sum(ab))


def _v8_ok(*ts) -> bool:
    """The 8-vector kernels' precondition: tokens contiguous, row starts aligned to 8 elements (and 16 bytes)."""
    for t in ts:
        if t.stride(2) != 1 or t.shape[2] % 8 or t.stride(0) % 8 or t.stride(1) % 8 or t.data_ptr() % 16:
            return False
    return True


def _v8_cfg(kind, B, C, L):
    """(N chunks per step, SPAN tokens per program, warps): from the A100 sweep of the SSD layer shapes; FORCE[kind] overrides."""
    c = FORCE.get(kind)
    if c is not None:
        return c
    rows, l2 = B * C, triton.next_power_of_2(L)
    if kind == "fwd_v8":
        return dict(N=256, SPAN=min(l2, 4096), warps=4) if rows >= 8192 else dict(N=64, SPAN=512, warps=1)
    return dict(N=32, SPAN=min(l2, 512 if rows < 4096 else 1024 if rows < 16384 else 4096), warps=1)


#: LAUNCH CONFIGURATIONS are a deterministic function of (kernel, element size, B, C, L): no runtime autotune, so a run's kernels never depend on timing noise.
#: The rule is a nearest-neighbour lookup in `_TABLE`, the best configuration of each swept shape (log-distance in B*L*C, C and L).  `FORCE[kind] = cfg` overrides it
#: (tests, sweeps).  Kinds: "fwd_cl" channel-last forward (tile: BC, BL, warps, lfast = token blocks fastest in the grid; or streaming rows: BC, TL, warps, stages),
#: "bwd_cl" channel-last streaming backward (BC, TL, warps, stages), "fwd_cm" channel-major tile forward (BC, BL, warps), "bwd_cm" channel-major fused tile backward
#: (BC, BL, CH, warps).
#:
#: THE SWEEP (an RTX PRO 5000 Blackwell, sm_120; L2 COLD = a 256 MB write before every call; CUDA-event median of 12):
#: shapes (B, L, C) = (8,2048,128) (8,2048,256) (2,2048,256) (8,2048,512) [channel-last only: the narrow B|C conv of SSD-MIMO] (8,2048,1536) (8,2048,2048) (8,2048,3072) (16,4096,4096) (2,2048,2048) (4,512,1536) (1,8192,2048) (8,2048,768), bf16 and fp32, both layouts,
#: every candidate of: fwd_cl tile BC {64,128,256} x BL {16,32,64} x warps {2,4,8} x lfast, and rows BC {128,256} x TL {128,256,512} x warps {2,4} x stages {2,3,4};
#: bwd_cl rows BC {64,128,256} x TL {32..512} x warps {1,2,4,8} x stages {2,3,4}; fwd_cm BC {2,4,8,16} x BL {128..1024} x warps {2,4,8};
#: bwd_cm BC {2,4,8} x BL {64,128,256} x CH {512,1024,2048} x warps {2,4,8}.  What the sweep says: channel-last forward wants wide channel tiles (128-256, 256 B rows in bf16)
#: with few tokens (16-32: the 3-row halo is the re-read) and the tile kernel beats the streaming rows kernel; the streaming backward wants LONG chunks (TL 256-512) on
#: big problems (fewer dw partial rows) and one warp on narrow C; the channel-major kernels want the token axis long (BL 512-1024 forward, CH 1024-2048 backward).
#: The channel-last fused tile backward (the channel-major kernel on channel-last x) was swept too and never won.  The streaming backward issues four tokens' loads at
#: once (`_bwd_step`): before that, narrow / small shapes were bound by bytes in flight (0.9 us per token step) and ran 1.3-1.8x the package.
_TABLE = {
    ("fwd_cl", 2): [
        ((1, 8192, 2048), dict(BC=128, BL=16, warps=2, lfast=False)),
        ((2, 2048, 256), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((2, 2048, 2048), dict(BC=128, BL=16, warps=4, lfast=False)),
        ((4, 512, 1536), dict(BC=64, BL=16, warps=4, lfast=True)),
        ((8, 2048, 128), dict(BC=64, BL=32, warps=2, lfast=True)),
        ((8, 2048, 256), dict(BC=128, BL=32, warps=8, lfast=True)),
        ((8, 2048, 512), dict(BC=64, BL=32, warps=8, lfast=True)),
        ((8, 2048, 768), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((8, 2048, 1536), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 2048), dict(BC=64, BL=16, warps=8, lfast=False)),
        ((8, 2048, 3072), dict(BC=128, BL=32, warps=2, lfast=False)),
        ((16, 4096, 4096), dict(BC=256, BL=64, warps=8, lfast=True)),
    ],
    ("fwd_cl", 4): [
        ((1, 8192, 2048), dict(BC=64, BL=16, warps=2, lfast=False)),
        ((2, 2048, 256), dict(BC=64, BL=16, warps=4, lfast=True)),
        ((2, 2048, 2048), dict(BC=64, BL=16, warps=2, lfast=False)),
        ((4, 512, 1536), dict(BC=256, BL=32, warps=8, lfast=True)),
        ((8, 2048, 128), dict(BC=128, BL=16, warps=8, lfast=True)),
        ((8, 2048, 256), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((8, 2048, 512), dict(BC=64, BL=32, warps=2, lfast=True)),
        ((8, 2048, 768), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 1536), dict(BC=128, BL=32, warps=4, lfast=False)),
        ((8, 2048, 2048), dict(BC=256, BL=16, warps=4, lfast=False)),
        ((8, 2048, 3072), dict(BC=256, BL=16, warps=4, lfast=False)),
        ((16, 4096, 4096), dict(BC=256, BL=32, warps=8, lfast=True)),
    ],
    ("bwd_cl", 2): [
        ((1, 8192, 2048), dict(BC=64, TL=512, warps=1, stages=4)),
        ((2, 2048, 256), dict(BC=64, TL=128, warps=1, stages=4)),
        ((2, 2048, 2048), dict(BC=128, TL=256, warps=1, stages=4)),
        ((4, 512, 1536), dict(BC=128, TL=128, warps=2, stages=4)),
        ((8, 2048, 128), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 256), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 512), dict(BC=64, TL=512, warps=1, stages=4)),
        ((8, 2048, 768), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 1536), dict(BC=256, TL=512, warps=2, stages=4)),
        ((8, 2048, 2048), dict(BC=128, TL=512, warps=2, stages=3)),
        ((8, 2048, 3072), dict(BC=256, TL=512, warps=1, stages=3)),
        ((16, 4096, 4096), dict(BC=256, TL=512, warps=1, stages=4)),
    ],
    ("bwd_cl", 4): [
        ((1, 8192, 2048), dict(BC=64, TL=512, warps=1, stages=4)),
        ((2, 2048, 256), dict(BC=32, TL=64, warps=2, stages=3)),
        ((2, 2048, 2048), dict(BC=64, TL=256, warps=1, stages=3)),
        ((4, 512, 1536), dict(BC=64, TL=64, warps=2, stages=4)),
        ((8, 2048, 128), dict(BC=32, TL=128, warps=1, stages=3)),
        ((8, 2048, 256), dict(BC=32, TL=256, warps=1, stages=3)),
        ((8, 2048, 512), dict(BC=64, TL=256, warps=4, stages=3)),
        ((8, 2048, 768), dict(BC=64, TL=512, warps=1, stages=4)),
        ((8, 2048, 1536), dict(BC=64, TL=512, warps=1, stages=3)),
        ((8, 2048, 2048), dict(BC=128, TL=512, warps=1, stages=3)),
        ((8, 2048, 3072), dict(BC=256, TL=512, warps=2, stages=4)),
        ((16, 4096, 4096), dict(BC=256, TL=512, warps=2, stages=3)),
    ],
    ("fwd_cm", 2): [
        ((1, 8192, 2048), dict(BC=4, BL=1024, warps=8)),
        ((2, 2048, 2048), dict(BC=4, BL=512, warps=4)),
        ((4, 512, 1536), dict(BC=2, BL=512, warps=4)),
        ((8, 2048, 768), dict(BC=4, BL=1024, warps=8)),
        ((8, 2048, 1536), dict(BC=16, BL=256, warps=8)),
        ((8, 2048, 2048), dict(BC=4, BL=1024, warps=2)),
        ((8, 2048, 3072), dict(BC=8, BL=1024, warps=4)),
        ((16, 4096, 4096), dict(BC=4, BL=1024, warps=8)),
    ],
    ("fwd_cm", 4): [
        ((1, 8192, 2048), dict(BC=8, BL=1024, warps=4)),
        ((2, 2048, 2048), dict(BC=8, BL=1024, warps=4)),
        ((4, 512, 1536), dict(BC=2, BL=1024, warps=8)),
        ((8, 2048, 768), dict(BC=2, BL=256, warps=2)),
        ((8, 2048, 1536), dict(BC=16, BL=512, warps=4)),
        ((8, 2048, 2048), dict(BC=8, BL=1024, warps=4)),
        ((8, 2048, 3072), dict(BC=4, BL=1024, warps=2)),
        ((16, 4096, 4096), dict(BC=4, BL=1024, warps=8)),
    ],
    ("bwd_cm", 2): [
        ((1, 8192, 2048), dict(BC=2, BL=128, CH=1024, warps=2)),
        ((2, 2048, 2048), dict(BC=2, BL=128, CH=512, warps=2)),
        ((4, 512, 1536), dict(BC=2, BL=128, CH=512, warps=2)),
        ((8, 2048, 768), dict(BC=2, BL=128, CH=512, warps=2)),
        ((8, 2048, 1536), dict(BC=2, BL=128, CH=1024, warps=2)),
        ((8, 2048, 2048), dict(BC=2, BL=128, CH=1024, warps=2)),
        ((8, 2048, 3072), dict(BC=2, BL=128, CH=2048, warps=2)),
        ((16, 4096, 4096), dict(BC=2, BL=128, CH=2048, warps=2)),
    ],
    ("bwd_cm", 4): [
        ((1, 8192, 2048), dict(BC=2, BL=128, CH=1024, warps=4)),
        ((2, 2048, 2048), dict(BC=2, BL=128, CH=512, warps=2)),
        ((4, 512, 1536), dict(BC=2, BL=256, CH=512, warps=4)),
        ((8, 2048, 768), dict(BC=4, BL=128, CH=512, warps=2)),
        ((8, 2048, 1536), dict(BC=2, BL=128, CH=2048, warps=4)),
        ((8, 2048, 2048), dict(BC=2, BL=256, CH=2048, warps=4)),
        ((8, 2048, 3072), dict(BC=4, BL=128, CH=2048, warps=2)),
        ((16, 4096, 4096), dict(BC=2, BL=128, CH=2048, warps=4)),
    ],
}


#: The same sweep on an A100-SXM4-80GB, sm_80 (L2-cold sweep for the 12 table shapes; the irregular-length shapes (L 1000-3000) by a CHAINED sweep: 30 back-to-back calls / 30, inputs cycled past L2). The Blackwell picks ran bf16 forward at 1.1-1.5x the package and bf16 backward at irregular L at up to 1.2x; these run forward at ~0.93-1.0x, backward at ~0.4-0.6x. bwd_cl = the Blackwell rows plus the chained winners.
_TABLE_SM80 = {
    ('fwd_cl', 2): [
        ((1, 8192, 2048), dict(BC=256, BL=16, warps=4, lfast=False)),
        ((2, 2048, 256), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((2, 2048, 2048), dict(BC=64, BL=64, warps=8, lfast=False)),
        ((2, 4096, 1024), dict(BC=64, BL=64, warps=4, lfast=True)),
        ((4, 512, 1536), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((4, 1024, 2560), dict(BC=64, BL=64, warps=4, lfast=False)),
        ((4, 3000, 2048), dict(BC=256, BL=32, warps=8, lfast=False)),
        ((6, 3000, 1280), dict(BC=256, BL=16, warps=8, lfast=True)),
        ((8, 1500, 1280), dict(BC=64, BL=32, warps=4, lfast=False)),
        ((8, 2048, 128), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((8, 2048, 256), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 512), dict(BC=64, BL=32, warps=4, lfast=False)),
        ((8, 2048, 768), dict(BC=256, BL=16, warps=4, lfast=False)),
        ((8, 2048, 1536), dict(BC=64, BL=32, warps=4, lfast=False)),
        ((8, 2048, 2048), dict(BC=64, BL=32, warps=4, lfast=False)),
        ((8, 2048, 3072), dict(BC=64, BL=32, warps=4, lfast=False)),
        ((16, 4096, 4096), dict(BC=256, BL=32, warps=8, lfast=True)),
    ],
    ('fwd_cl', 4): [
        ((1, 8192, 2048), dict(BC=256, BL=16, warps=8, lfast=False)),
        ((2, 2048, 256), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((2, 2048, 2048), dict(BC=128, BL=16, warps=4, lfast=False)),
        ((2, 4096, 1024), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((4, 512, 1536), dict(BC=128, BL=32, warps=4, lfast=True)),
        ((4, 1024, 2560), dict(BC=256, BL=16, warps=8, lfast=False)),
        ((4, 3000, 2048), dict(BC=256, BL=16, warps=8, lfast=False)),
        ((6, 3000, 1280), dict(BC=256, BL=16, warps=8, lfast=False)),
        ((8, 1500, 1280), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 128), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((8, 2048, 256), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 512), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 768), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 1536), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 2048), dict(BC=256, BL=16, warps=8, lfast=False)),
        ((8, 2048, 3072), dict(BC=256, BL=32, warps=8, lfast=False)),
        ((16, 4096, 4096), dict(BC=256, BL=16, warps=8, lfast=False)),
    ],
    ('bwd_cl', 2): [
        ((1, 8192, 2048), dict(BC=64, TL=512, warps=1, stages=4)),
        ((2, 2048, 256), dict(BC=64, TL=128, warps=1, stages=4)),
        ((2, 2048, 2048), dict(BC=128, TL=256, warps=1, stages=4)),
        ((2, 4096, 1024), dict(BC=256, TL=64, warps=2, stages=3)),
        ((4, 512, 1536), dict(BC=128, TL=128, warps=2, stages=4)),
        ((4, 1024, 2560), dict(BC=256, TL=64, warps=4, stages=3)),
        ((4, 3000, 2048), dict(BC=64, TL=512, warps=1, stages=4)),
        ((6, 3000, 1280), dict(BC=64, TL=512, warps=1, stages=4)),
        ((8, 1500, 1280), dict(BC=64, TL=128, warps=1, stages=3)),
        ((8, 2048, 128), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 256), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 512), dict(BC=64, TL=512, warps=1, stages=4)),
        ((8, 2048, 768), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 1536), dict(BC=256, TL=512, warps=2, stages=4)),
        ((8, 2048, 2048), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 3072), dict(BC=256, TL=512, warps=1, stages=3)),
        ((16, 4096, 4096), dict(BC=256, TL=512, warps=1, stages=4)),
    ],
    ('bwd_cl', 4): [
        ((1, 8192, 2048), dict(BC=64, TL=512, warps=1, stages=4)),
        ((2, 2048, 256), dict(BC=32, TL=64, warps=2, stages=3)),
        ((2, 2048, 2048), dict(BC=64, TL=256, warps=1, stages=3)),
        ((2, 4096, 1024), dict(BC=64, TL=256, warps=2, stages=3)),
        ((4, 512, 1536), dict(BC=64, TL=64, warps=2, stages=4)),
        ((4, 1024, 2560), dict(BC=128, TL=256, warps=2, stages=3)),
        ((4, 3000, 2048), dict(BC=128, TL=512, warps=2, stages=3)),
        ((6, 3000, 1280), dict(BC=128, TL=512, warps=2, stages=3)),
        ((8, 1500, 1280), dict(BC=128, TL=256, warps=2, stages=3)),
        ((8, 2048, 128), dict(BC=32, TL=128, warps=1, stages=3)),
        ((8, 2048, 256), dict(BC=32, TL=256, warps=1, stages=3)),
        ((8, 2048, 512), dict(BC=64, TL=256, warps=4, stages=3)),
        ((8, 2048, 768), dict(BC=64, TL=512, warps=1, stages=4)),
        ((8, 2048, 1536), dict(BC=64, TL=512, warps=1, stages=3)),
        ((8, 2048, 2048), dict(BC=256, TL=256, warps=2, stages=3)),
        ((8, 2048, 3072), dict(BC=256, TL=512, warps=2, stages=4)),
        ((16, 4096, 4096), dict(BC=256, TL=512, warps=2, stages=3)),
    ],
}

#: The same sweep on an H100 PCIe, sm_90 (L2-cold for the table shapes, CHAINED for the irregular-length ones, as on sm_80). The Blackwell picks ran bf16 forward at 1.1-1.5x the package there; these run forward at ~0.95-1.0x, backward at ~0.3-0.6x. bwd_cl = the Blackwell rows plus the chained winners.
#: (These tables cover the channel-LAST path; the SSD layer's channel-major call runs `_fwd_v8` / `_bwd_v8`, whose rule is `_v8_cfg`.)
_TABLE_SM90 = {
    ('fwd_cl', 2): [
        ((1, 8192, 2048), dict(BC=128, BL=16, warps=4, lfast=False)),
        ((2, 2048, 256), dict(BC=64, BL=32, warps=4, lfast=True)),
        ((2, 2048, 2048), dict(BC=128, BL=16, warps=4, lfast=False)),
        ((2, 4096, 1024), dict(BC=128, BL=16, warps=2, lfast=False)),
        ((4, 512, 1536), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((4, 1024, 2560), dict(BC=128, BL=16, warps=4, lfast=False)),
        ((4, 3000, 2048), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((6, 3000, 1280), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 1500, 1280), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 128), dict(BC=128, BL=16, warps=2, lfast=True)),
        ((8, 2048, 256), dict(BC=64, BL=32, warps=4, lfast=True)),
        ((8, 2048, 512), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 768), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 1536), dict(BC=128, BL=16, warps=2, lfast=True)),
        ((8, 2048, 2048), dict(BC=128, BL=16, warps=4, lfast=True)),
        ((8, 2048, 3072), dict(BC=128, BL=16, warps=2, lfast=True)),
        ((16, 4096, 4096), dict(BC=128, BL=32, warps=4, lfast=True)),
    ],
    ('fwd_cl', 4): [
        ((1, 8192, 2048), dict(BC=64, BL=32, warps=4, lfast=False)),
        ((2, 2048, 256), dict(BC=64, BL=16, warps=4, lfast=True)),
        ((2, 2048, 2048), dict(BC=64, BL=32, warps=4, lfast=False)),
        ((2, 4096, 1024), dict(BC=64, BL=32, warps=4, lfast=True)),
        ((4, 512, 1536), dict(BC=64, BL=16, warps=4, lfast=True)),
        ((4, 1024, 2560), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((4, 3000, 2048), dict(BC=64, BL=32, warps=4, lfast=False)),
        ((6, 3000, 1280), dict(BC=256, BL=16, warps=4, lfast=False)),
        ((8, 1500, 1280), dict(BC=256, BL=16, warps=4, lfast=True)),
        ((8, 2048, 128), dict(BC=64, BL=32, warps=4, lfast=True)),
        ((8, 2048, 256), dict(BC=64, BL=32, warps=4, lfast=True)),
        ((8, 2048, 512), dict(BC=64, BL=32, warps=4, lfast=True)),
        ((8, 2048, 768), dict(BC=64, BL=16, warps=2, lfast=True)),
        ((8, 2048, 1536), dict(BC=64, BL=16, warps=4, lfast=True)),
        ((8, 2048, 2048), dict(BC=256, BL=16, warps=4, lfast=False)),
        ((8, 2048, 3072), dict(BC=64, BL=64, warps=8, lfast=True)),
        ((16, 4096, 4096), dict(BC=64, BL=16, warps=2, lfast=True)),
    ],
    ('bwd_cl', 2): [
        ((1, 8192, 2048), dict(BC=64, TL=512, warps=1, stages=4)),
        ((2, 2048, 256), dict(BC=64, TL=128, warps=1, stages=4)),
        ((2, 2048, 2048), dict(BC=128, TL=256, warps=1, stages=4)),
        ((2, 4096, 1024), dict(BC=128, TL=128, warps=2, stages=4)),
        ((4, 512, 1536), dict(BC=128, TL=128, warps=2, stages=4)),
        ((4, 1024, 2560), dict(BC=128, TL=128, warps=2, stages=4)),
        ((4, 3000, 2048), dict(BC=256, TL=256, warps=1, stages=3)),
        ((6, 3000, 1280), dict(BC=256, TL=256, warps=1, stages=3)),
        ((8, 1500, 1280), dict(BC=128, TL=256, warps=2, stages=4)),
        ((8, 2048, 128), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 256), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 512), dict(BC=64, TL=512, warps=1, stages=4)),
        ((8, 2048, 768), dict(BC=64, TL=256, warps=1, stages=4)),
        ((8, 2048, 1536), dict(BC=256, TL=512, warps=2, stages=4)),
        ((8, 2048, 2048), dict(BC=128, TL=256, warps=1, stages=3)),
        ((8, 2048, 3072), dict(BC=256, TL=512, warps=1, stages=3)),
        ((16, 4096, 4096), dict(BC=256, TL=512, warps=1, stages=4)),
    ],
    ('bwd_cl', 4): [
        ((1, 8192, 2048), dict(BC=64, TL=512, warps=1, stages=4)),
        ((2, 2048, 256), dict(BC=32, TL=64, warps=2, stages=3)),
        ((2, 2048, 2048), dict(BC=64, TL=256, warps=1, stages=3)),
        ((2, 4096, 1024), dict(BC=128, TL=128, warps=2, stages=3)),
        ((4, 512, 1536), dict(BC=64, TL=64, warps=2, stages=4)),
        ((4, 1024, 2560), dict(BC=64, TL=128, warps=1, stages=3)),
        ((4, 3000, 2048), dict(BC=256, TL=256, warps=4, stages=4)),
        ((6, 3000, 1280), dict(BC=256, TL=256, warps=4, stages=4)),
        ((8, 1500, 1280), dict(BC=64, TL=256, warps=1, stages=3)),
        ((8, 2048, 128), dict(BC=32, TL=128, warps=1, stages=3)),
        ((8, 2048, 256), dict(BC=32, TL=256, warps=1, stages=3)),
        ((8, 2048, 512), dict(BC=64, TL=256, warps=4, stages=3)),
        ((8, 2048, 768), dict(BC=64, TL=512, warps=1, stages=4)),
        ((8, 2048, 1536), dict(BC=64, TL=512, warps=1, stages=3)),
        ((8, 2048, 2048), dict(BC=256, TL=512, warps=4, stages=3)),
        ((8, 2048, 3072), dict(BC=256, TL=512, warps=2, stages=4)),
        ((16, 4096, 4096), dict(BC=256, TL=512, warps=2, stages=3)),
    ],
}


def _table(kind, es, device):
    """The swept table of the device's architecture (sm_80 / sm_90; the Blackwell sweep otherwise and for the channel-major kinds)."""
    key = (kind, 2 if es <= 2 else 4)
    major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 12
    arch = _TABLE_SM80 if major == 8 else _TABLE_SM90 if major == 9 else _TABLE
    return arch.get(key, _TABLE[key])


def _nearest(kind, es, B, C, L, device=torch.device("cpu")):
    import math
    best, bd = None, None
    for (b, l, c), cfg in _table(kind, es, device):
        d = abs(math.log2(B * L * C / (b * l * c))) + abs(math.log2(C / c)) + 0.5 * abs(math.log2(L / l))
        if bd is None or d < bd:
            best, bd = cfg, d
    return best


_CFG_CACHE: dict = {}


def _cfg(kind, x, B, C, L):
    c = FORCE.get(kind)
    if c is not None:
        return c
    key = (kind, x.element_size(), B, C, L, x.device)       # host cost: the lookup runs once per shape, not per launch
    c = _CFG_CACHE.get(key)
    if c is None:
        c = _CFG_CACHE[key] = _nearest(kind, x.element_size(), B, C, L, x.device)
    return c


FORCE: dict = {}


def dw_chunk(x: torch.Tensor, W: int = 4) -> int:
    """Tokens summed sequentially per dw / db partial row on the path `conv_bwd_triton` takes for ``x`` (the bound of the conv test: chunk + rows + 2 roundings)."""
    B, C, L = x.shape
    if W == 4 and x.stride(1) == 1:
        return _cfg("bwd_cl", x, B, C, L)["TL"]
    if W == 4 and x.dtype != torch.float64 and x.stride(2) == 1 and L % 8 == 0 and x.stride(0) % 8 == 0 and x.stride(1) % 8 == 0:
        return _v8_cfg("bwd_v8", B, C, L)["SPAN"]
    if W == 4 and x.stride(2) == 1:
        return _cfg("bwd_cm", x, B, C, L)["CH"]
    return _CH


def _use_rows(x, W, kind="fwd") -> bool:
    return W == 4 and x.stride(1) == 1 and kind == "bwd"



@triton.jit
def _reduce_kernel(PART, DW, DB, NP, C, W: tl.constexpr, WP: tl.constexpr, BC: tl.constexpr, HAS_DB: tl.constexpr, ACC: tl.constexpr):
    """Stage two of dw / db: sum the NP chunk rows in a FIXED order (r = 0, 1, ...), one launch."""
    offs_c = tl.program_id(0) * BC + tl.arange(0, BC)
    j = tl.arange(0, WP)
    cm = offs_c < C
    m = cm[:, None] & (j[None, :] < W + 1)
    acc = tl.zeros((BC, WP), ACC)
    for r in tl.range(0, NP, num_stages=4):
        acc += tl.load(PART + (r * C + offs_c[:, None]) * (W + 1) + j[None, :], mask=m, other=0.0)
    tl.store(DW + offs_c[:, None] * W + j[None, :], acc.to(DW.dtype.element_ty), mask=cm[:, None] & (j[None, :] < W))
    if HAS_DB:
        dbv = tl.sum(tl.where(j[None, :] == W, acc, 0.0), axis=1)
        tl.store(DB + offs_c, dbv.to(DB.dtype.element_ty), mask=cm)


def _finish(part, weight, bias, need, W, C, x):
    """Stage two on the device: (dw shaped like weight | None, db | None)."""
    dw = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype) if need[1] else None
    db = torch.empty(bias.shape, device=bias.device, dtype=bias.dtype) if (need[2] and bias is not None) else None
    if dw is None and db is None:
        return None, None
    BC = 64
    _reduce_kernel[(triton.cdiv(C, BC),)](
        part, dw if dw is not None else part, db if db is not None else part, part.shape[0], C, W=W,
        WP=triton.next_power_of_2(W + 1), BC=BC, HAS_DB=db is not None, ACC=_acc(x.dtype), num_warps=2)
    return dw, db


def _acc(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _blocks(x: torch.Tensor):
    """(BC, BL, warps): wide along the contiguous axis."""
    if x.stride(1) == 1:                 # channel-last
        return 64, 16, 4
    return 8, 128, 4


def _wmat(weight):
    w = weight.reshape(weight.shape[0], -1)
    return w if w.is_contiguous() else w.contiguous()


def _buffer(x: torch.Tensor) -> torch.Tensor:
    """An output with x's memory order ([B, C, L] logical; channel-last when x is)."""
    if x.stride(1) == 1:
        b, c, l = x.shape
        return torch.empty((b, l, c), device=x.device, dtype=x.dtype).transpose(1, 2)
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def conv_fwd_triton(x: torch.Tensor, weight: torch.Tensor, bias, silu: bool, out=None) -> torch.Tensor:
    """x [B, C, L] (any strides) -> y [B, C, L] in x's dtype and memory order (or into ``out``, [B, C, L] any strides).
    weight [C, 1, W] | [C, W]."""
    B, C, L = x.shape
    w = _wmat(weight)
    W = w.shape[1]
    y = _buffer(x) if out is None else out
    BC, BL, nw = _blocks(x)
    lfast = False
    if x.stride(1) == 1:                 # channel-last
        cfg = _cfg("fwd_cl", x, B, C, L)
        if "TL" in cfg and W == 4 and y.stride(1) == 1:       # the streaming rows forward
            _fwd_rows[(B, triton.cdiv(C, cfg["BC"]), triton.cdiv(L, cfg["TL"]))](
                x, w, bias if bias is not None else w, y, C, L, x.stride(0), x.stride(2), y.stride(0), y.stride(2),
                HAS_BIAS=bias is not None, SILU=bool(silu), ACC=_acc(x.dtype), BC=cfg["BC"], TL=cfg["TL"], NS=cfg["stages"],
                num_warps=cfg["warps"])
            return y
        if "TL" not in cfg:
            BC, BL, nw, lfast = cfg["BC"], cfg["BL"], cfg["warps"], cfg["lfast"]
    elif W == 4 and x.dtype != torch.float64 and _v8_ok(x, y):          # channel-major, aligned rows: 8-vectors
        cfg = _v8_cfg("fwd_v8", B, C, L)
        _fwd_v8[(B * C, triton.cdiv(L, cfg["SPAN"]))](
            x, w, bias if bias is not None else w, y, C, L // 8, cfg["SPAN"], x.stride(0), x.stride(1), y.stride(0), y.stride(1),
            HAS_BIAS=bias is not None, SILU=bool(silu), ACC=_acc(x.dtype), N=cfg["N"], num_warps=cfg["warps"])
        return y
    elif x.stride(2) == 1:               # channel-major
        cfg = _cfg("fwd_cm", x, B, C, L)
        BC, BL, nw = cfg["BC"], cfg["BL"], cfg["warps"]
    nl, nc = triton.cdiv(L, BL), triton.cdiv(C, BC)
    grid = (nl, nc, B) if lfast else (B, nc, nl)
    _fwd_kernel[grid](x, w, bias if bias is not None else w, y, C, L, *x.stride(), *y.stride(),
                      W=W, HAS_BIAS=bias is not None, SILU=bool(silu), ACC=_acc(x.dtype), BC=BC, BL=BL, LFAST=lfast,
                      num_warps=nw)
    return y


def conv_bwd_triton(x: torch.Tensor, weight: torch.Tensor, bias, dy: torch.Tensor, silu: bool,
                    need=(True, True, True), dx=None):
    """-> (dx [B, C, L] in x's memory order (or written into ``dx``), dw shaped like weight, db | None)."""
    B, C, L = x.shape
    w = _wmat(weight)
    W = w.shape[1]
    dxo = dx
    cl_cfg = _cfg("bwd_cl", x, B, C, L) if (W == 4 and x.stride(1) == 1) else None
    if cl_cfg is not None and "TL" in cl_cfg and dy.stride(1) == 1 and (not need[0] or (dxo if dxo is not None else x).stride(1) == 1):
        cfg = cl_cfg
        nch = triton.cdiv(L, cfg["TL"])
        part = torch.empty(B * nch, C, W + 1, device=x.device, dtype=torch.float64 if x.dtype == torch.float64 else torch.float32)
        dx = (dxo if dxo is not None else _buffer(x)) if need[0] else x
        _bwd_rows[(B, triton.cdiv(C, cfg["BC"]), nch)](
            x, w, bias if bias is not None else w, dy, dx, part, C, L, nch, x.stride(0), x.stride(2), dy.stride(0), dy.stride(2),
            dx.stride(0), dx.stride(2), HAS_BIAS=bias is not None, SILU=bool(silu), ACC=_acc(x.dtype), BC=cfg["BC"], TL=cfg["TL"],
            NS=cfg["stages"], NEED_DX=bool(need[0]), num_warps=cfg["warps"])
        dw, db = _finish(part, weight, bias, need, W, C, x)
        return (dx if need[0] else None), dw, db
    dxv = (dxo if dxo is not None else _buffer(x)) if (need[0] and W == 4 and x.stride(1) != 1) else None
    if W == 4 and x.dtype != torch.float64 and x.stride(1) != 1 and _v8_ok(x, dy, *(() if dxv is None else (dxv,))):
        cfg = _v8_cfg("bwd_v8", B, C, L)          # channel-major, aligned rows: 8-vectors
        nsp = triton.cdiv(L, cfg["SPAN"])
        need_dw = bool(need[1] or (need[2] and bias is not None))
        part = torch.empty(B * nsp if need_dw else 1, C, 5, device=x.device, dtype=torch.float32)
        dxb = dxv if dxv is not None else x
        _bwd_v8[(B * C, nsp)](
            x, w, bias if bias is not None else w, dy, dxb, part, C, L // 8, cfg["SPAN"], nsp, x.stride(0), x.stride(1),
            dy.stride(0), dy.stride(1), dxb.stride(0), dxb.stride(1), HAS_BIAS=bias is not None, SILU=bool(silu), ACC=_acc(x.dtype),
            N=cfg["N"], NEED_DX=bool(need[0]), NEED_DW=need_dw, num_warps=cfg["warps"])
        dw = db = None
        if need_dw:
            dw, db = _finish(part, weight, bias, need, W, C, x)
        return dxv, dw, db
    if W == 4 and x.stride(2) == 1 and x.stride(1) != 1:        # channel-major: the fused tile backward
        cfg = _cfg("bwd_cm", x, B, C, L)
        nch = triton.cdiv(L, cfg["CH"])
        need_dw = bool(need[1] or (need[2] and bias is not None))
        part = torch.empty(B * nch if need_dw else 1, C, 5, device=x.device,
                           dtype=torch.float64 if x.dtype == torch.float64 else torch.float32)
        dxb = dxv if need[0] else x
        _bwd_tile4[(B, triton.cdiv(C, cfg["BC"]), nch)](
            x, w, bias if bias is not None else w, dy, dxb, part, C, L, nch, *x.stride(), *dy.stride(), *dxb.stride(),
            HAS_BIAS=bias is not None, SILU=bool(silu), ACC=_acc(x.dtype), BC=cfg["BC"], BL=cfg["BL"], CH=cfg["CH"],
            NEED_DX=bool(need[0]), NEED_DW=need_dw, num_warps=cfg["warps"])
        dw = db = None
        if need_dw:
            dw, db = _finish(part, weight, bias, need, W, C, x)
        return (dxb if need[0] else None), dw, db
    BC, BL, nw = _blocks(x)
    common = dict(W=W, HAS_BIAS=bias is not None, SILU=bool(silu), ACC=_acc(x.dtype), BC=BC, BL=BL, num_warps=nw)
    bptr = bias if bias is not None else w
    dx = dw = db = None
    if need[0]:
        dx = dxo if dxo is not None else _buffer(x)
        _dx_kernel[(B, triton.cdiv(C, BC), triton.cdiv(L, BL))](
            x, w, bptr, dy, dx, C, L, *x.stride(), *dy.stride(), *dx.stride(), **common)
    if need[1] or (need[2] and bias is not None):
        nch = triton.cdiv(L, _CH)
        part = torch.empty(B * nch, C, W + 1, device=x.device, dtype=torch.float64 if x.dtype == torch.float64 else torch.float32)
        _dw_kernel[(B, triton.cdiv(C, BC), nch)](
            x, w, bptr, dy, part, C, L, nch, *x.stride(), *dy.stride(), CH=_CH, **common)
        dw, db = _finish(part, weight, bias, need, W, C, x)       # stage two: fixed order, one launch
    return dx, dw, db


@triton.jit
def _step_kernel(X, ST, Wt, Bi, Y, NST, C, sxb, ssb, ssc, ssj, nsb, nsc, nsj, syb,
                 W: tl.constexpr, WP: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, ACC: tl.constexpr, BC: tl.constexpr):
    pb = tl.program_id(0)
    offs_c = tl.program_id(1) * BC + tl.arange(0, BC)
    j = tl.arange(0, WP)
    cm = offs_c < C
    old = tl.load(ST + pb * ssb + offs_c[:, None] * ssc + (j[None, :] + 1) * ssj, mask=cm[:, None] & (j[None, :] < W - 1), other=0.0)
    xt = tl.load(X + pb * sxb + offs_c, mask=cm, other=0.0)
    new = tl.where(j[None, :] == W - 1, xt[:, None].to(NST.dtype.element_ty), old.to(NST.dtype.element_ty))
    tl.store(NST + pb * nsb + offs_c[:, None] * nsc + j[None, :] * nsj, new, mask=cm[:, None] & (j[None, :] < W))
    wv = tl.load(Wt + offs_c[:, None] * W + j[None, :], mask=cm[:, None] & (j[None, :] < W), other=0.0).to(ACC)
    z = tl.sum(new.to(ACC) * wv, axis=1)
    if HAS_BIAS:
        z += tl.load(Bi + offs_c, mask=cm, other=0.0).to(ACC)
    if SILU:
        z = z / (1.0 + tl.exp(-z))
    tl.store(Y + pb * syb + offs_c, z.to(Y.dtype.element_ty), mask=cm)


def conv_step_triton(x_t: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias, silu: bool, out_dtype=None):
    """One decode token: x_t [B, C], state [B, C, W] (oldest first, the window BEFORE this token), weight [C, 1, W] | [C, W] ->
    (y [B, C] in ``out_dtype`` (default: the promotion of state and weight dtypes, as the torch composite gives), new_state [B, C, W] in
    the promotion of the state and x_t dtypes).  The window is the W newest raw inputs; y = act(bias + sum_j w_j window_j), products and sum in fp32
    (fp64 for fp64), one rounding to the output dtype."""
    B, C = x_t.shape
    if x_t.stride(1) != 1:
        x_t = x_t.contiguous()
    w = _wmat(weight)
    W = w.shape[1]
    nst = torch.empty(B, C, W, device=x_t.device, dtype=torch.promote_types(state.dtype, x_t.dtype))
    y = torch.empty(B, C, device=x_t.device, dtype=out_dtype if out_dtype is not None else torch.promote_types(state.dtype, w.dtype))
    BC = 128
    _step_kernel[(B, triton.cdiv(C, BC))](
        x_t, state, w, bias if bias is not None else w, y, nst, C, x_t.stride(0), state.stride(0), state.stride(1), state.stride(2),
        nst.stride(0), nst.stride(1), nst.stride(2), y.stride(0), W=W, WP=triton.next_power_of_2(W), HAS_BIAS=bias is not None, SILU=bool(silu),
        ACC=_acc(torch.promote_types(x_t.dtype, state.dtype)), BC=BC, num_warps=4)
    return y, nst


class _TritonConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, silu):
        ctx.silu = bool(silu)
        ctx.save_for_backward(x, weight, bias)
        return conv_fwd_triton(x, weight, bias, silu)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        need = (ctx.needs_input_grad[0], ctx.needs_input_grad[1], bias is not None and ctx.needs_input_grad[2])
        dx, dw, db = conv_bwd_triton(x, weight, bias, dy, ctx.silu, need)
        return dx, dw, db, None


def causal_conv_triton(x: torch.Tensor, weight: torch.Tensor, bias, silu: bool) -> torch.Tensor:
    """Autograd-tracked plain-Triton depthwise causal conv: x [B, C, L] -> [B, C, L]."""
    return _TritonConv.apply(x, weight, bias, silu)
