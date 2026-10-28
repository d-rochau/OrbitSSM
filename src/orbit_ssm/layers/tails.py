"""Fused elementwise kernels for the token-map tail, for the SSD-MIMO and HM mixers.

The tail turns the fused token projection's raw slab into the maps the scan
kernel takes: B/C RMS-norm, shared bias, the gB / gC gates, the dt softplus,
the dynamic-rate (`a_dyn`) tail and the dt * rate product. `token_tails` does
all of it in one launch forward and two backward (the second is a deterministic
stage-2 reduction of the parameter gradients; atomics would make the backward
non-reproducible). It is a module-side fusion and does not touch the recurrence.

Layout: every source is read through its own (B, L, col) stride triple, so a
token-major slice and the channel-major output of the xBC conv are both read in
place. Outputs are written in the memory order the eager chain produced
(r-major (..., R, N) when the norm runs, n-major otherwise).

Numerics: the whole tail is computed in fp32 registers and rounded once on the
store, whereas the eager chain rounds to the IO dtype at every step. Under bf16
autocast the two agree to bf16 tolerance (this is the more accurate), under
fp32 to round-off. `SSD_MIMO_TAILS=0` selects the eager chain.

Envelope: CUDA fp16/bf16/fp32 sources (fp64 falls back) with the fused
token-map projection. `theta` / `trap` / `rmix` slices are not part of the tail;
they read the fp32 `dt_k` it produces. Both families use uncapped softplus;
`TailCfg.dt_dtype` selects the dt IO dtype (MH: IO dtype, SSD: fp32) and
`bc_channel_major` MH's (B, N, L) map layout.

`gate_transpose` and `dt_softplus_cf` at the end serve the HM epilogue (layout
transition with optional fused gate, and the wide channel-major dt tail).
"""

from __future__ import annotations

import os
from typing import NamedTuple

import contextlib as _contextlib
import threading as _threading
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..compile_compat import dynamo_opaque, tails_boundary

__all__ = ["tails_enabled", "TailCfg", "token_tails", "can_fuse",
           "gate_transpose", "dt_softplus_cf"]

_SP_THR = 20.0          # torch's softplus threshold (beta = 1)


def tails_enabled() -> bool:
    """`SSD_MIMO_TAILS=0` restores the eager token-map tail."""
    return os.environ.get("SSD_MIMO_TAILS", "1") == "1"


class TailCfg(NamedTuple):
    """Static configuration the kernels specialise on (plain python values).

    N, R, O, H, L: state width, B rank, C rank, heads, sequence length.
    bc_norm: RMS-norm B and C (b_norm / c_norm override per map; None follows
    bc_norm). shared_bias: B_bias / C_bias are applied here. fold_gb: gB folded
    onto B (R == 1 only). fold_gc: gC folded onto C. a_dyn: dynamic rate tail;
    a_per_sibling: `a_src` has one rate per kernel head (B, L, H*R, column
    h*R + r) instead of per head (B, L, H); a_heavy_tail: SSD M3 rate law
    instead of softplus. has_dt_bias / has_a_bias: biases present. eps_b,
    eps_c: norm epsilons; a_floor: lower clamp of the dynamic rate. dt_dtype:
    dt output dtype (None = fp32). bc_channel_major: HM's (B, N, L) B/C layout.
    """
    N: int
    R: int
    O: int
    H: int
    L: int
    bc_norm: bool
    shared_bias: bool
    fold_gb: bool
    fold_gc: bool
    a_dyn: bool
    has_dt_bias: bool
    has_a_bias: bool
    eps_b: float
    eps_c: float
    a_floor: float
    dt_dtype: "torch.dtype | None" = None
    bc_channel_major: bool = False
    a_per_sibling: bool = False
    a_heavy_tail: bool = False
    b_norm: "bool | None" = None
    c_norm: "bool | None" = None

    @property
    def norm_b(self) -> bool:
        return bool(self.bc_norm if self.b_norm is None else self.b_norm)

    @property
    def norm_c(self) -> bool:
        return bool(self.bc_norm if self.c_norm is None else self.c_norm)


# ── the tail's scalar maps, shared by both kernels ───────────────────
# log1p(exp(x)) via libdevice, matching ATen's softplus_kernel at small positive x.
@triton.jit
def _softplus(x, thr):
    return tl.where(x > thr, x, libdevice.log1p(tl.exp(tl.minimum(x, thr))))


@triton.jit
def _softplus_grad(x, thr):
    return tl.where(x > thr, 1.0, tl.sigmoid(x))


# ── forward ──────────────────────────────────────────────────────────
@triton.jit
def _rate_activation(x, thr, heavy: tl.constexpr):
    if heavy:
        return tl.maximum(x, 0.0) + 1.0 / (1.0 - tl.minimum(x, 0.0))
    return _softplus(x, thr)


@triton.jit
def _rate_grad(x, thr, heavy: tl.constexpr):
    if heavy:
        # Upstream torch clamp_min/max both pass their derivative at x=0.
        inv = 1.0 / (1.0 - tl.minimum(x, 0.0))
        return tl.where(x >= 0, 1.0, 0.0) + tl.where(x <= 0, inv * inv, 0.0)
    return _softplus_grad(x, thr)


@triton.jit
def _tails_fwd_kernel(
    PB, PC, PGB, PGC, PDT, PA,
    WB, WC, BB, CB, DTB, AB,
    BO, CO, DTK, DECK,
    T,
    bb_, bl_, bc_, cb_, cl_, cc_, gbb, gbl, gbc, gcb, gcl, gcc,
    db_, dl_, dc_, ab_, al_, ac_,
    ob_b, ob_l, ob_n, ob_r, oc_b, oc_l, oc_n, oc_o,
    N: tl.constexpr, R: tl.constexpr, O: tl.constexpr, H: tl.constexpr,
    L: tl.constexpr,
    eps_b: tl.constexpr, eps_c: tl.constexpr,
    afloor: tl.constexpr,
    BT: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr, BOO: tl.constexpr,
    BH: tl.constexpr,
    HAS_B: tl.constexpr, HAS_C: tl.constexpr, B_NORM: tl.constexpr,
    C_NORM: tl.constexpr,
    SH_BIAS: tl.constexpr, FOLD_GB: tl.constexpr, FOLD_GC: tl.constexpr,
    HAS_DTB: tl.constexpr, HAS_AB: tl.constexpr, A_DYN: tl.constexpr,
    A_R: tl.constexpr, A_HEAVY: tl.constexpr,
    THR: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    mt = t < T
    ts = tl.where(mt, t, 0)
    bi = ts // L
    li = ts - bi * L
    n = tl.arange(0, BN)
    mn = n < N
    r = tl.arange(0, BR)
    mr = r < R
    h = tl.arange(0, BH)
    mh = h < H

    # ── B: RMS-norm over N per (token, r), shared bias, gB fold ──────
    if HAS_B:
        m3 = mt[:, None, None] & mn[None, :, None] & mr[None, None, :]
        tok = (bi * bb_ + li * bl_)[:, None, None]
        col = (n[None, :, None] * R + r[None, None, :]) * bc_
        x = tl.load(PB + tok + col, mask=m3, other=0.0).to(tl.float32)
        if B_NORM:
            rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps_b)
            w = tl.load(WB + n, mask=mn, other=0.0).to(tl.float32)
            x = x * rstd[:, None, :] * w[None, :, None]
        if SH_BIAS:
            x = x + tl.load(BB + n, mask=mn, other=0.0).to(
                tl.float32)[None, :, None]
        if FOLD_GB:
            g = tl.load(PGB + (bi * gbb + li * gbl)[:, None, None]
                        + (n[None, :, None] * R + r[None, None, :]) * gbc,
                        mask=m3, other=0.0).to(tl.float32)
            x = x * (1.0 + g)
        tl.store(BO + (bi * ob_b + li * ob_l)[:, None, None]
                 + n[None, :, None] * ob_n + r[None, None, :] * ob_r,
                 x, mask=m3)

    # ── C: the same, over O, with gC folded ──────────────────────────
    if HAS_C:
        o = tl.arange(0, BOO)
        mo = o < O
        m3c = mt[:, None, None] & mn[None, :, None] & mo[None, None, :]
        tok = (bi * cb_ + li * cl_)[:, None, None]
        col = (n[None, :, None] * O + o[None, None, :]) * cc_
        y = tl.load(PC + tok + col, mask=m3c, other=0.0).to(tl.float32)
        if C_NORM:
            rstd = 1.0 / tl.sqrt(tl.sum(y * y, axis=1) / N + eps_c)
            w = tl.load(WC + n, mask=mn, other=0.0).to(tl.float32)
            y = y * rstd[:, None, :] * w[None, :, None]
        if SH_BIAS:
            y = y + tl.load(CB + n, mask=mn, other=0.0).to(
                tl.float32)[None, :, None]
        if FOLD_GC:
            g = tl.load(PGC + (bi * gcb + li * gcl)[:, None, None]
                        + (n[None, :, None] * O + o[None, None, :]) * gcc,
                        mask=m3c, other=0.0).to(tl.float32)
            y = y * (1.0 + g)
        tl.store(CO + (bi * oc_b + li * oc_l)[:, None, None]
                 + n[None, :, None] * oc_n + o[None, None, :] * oc_o,
                 y, mask=m3c)

    # ── dt / a_dyn, written STRAIGHT into the kernel's [B,Hk,L] store ──
    m3d = mt[:, None, None] & mh[None, :, None] & mr[None, None, :]
    d = tl.load(PDT + (bi * db_ + li * dl_)[:, None, None]
                + (h[None, :, None] * R + r[None, None, :]) * dc_,
                mask=m3d, other=0.0).to(tl.float32)
    if HAS_DTB:
        d = d + tl.load(DTB + h[:, None] * R + r[None, :],
                        mask=mh[:, None] & mr[None, :],
                        other=0.0).to(tl.float32)[None, :, :]
    sp = _softplus(d, THR)
    dtr = sp
    ok = (bi[:, None, None] * (R * H * L)
          + (r[None, None, :] * H + h[None, :, None]) * L
          + li[:, None, None])
    tl.store(DTK + ok, dtr, mask=m3d)
    if A_DYN:
        if A_R:
            # one dynamic rate per kernel head, column h*R + r as in `dt_src`
            a = tl.load(PA + (bi * ab_ + li * al_)[:, None, None]
                        + (h[None, :, None] * R + r[None, None, :]) * ac_,
                        mask=m3d, other=0.0).to(tl.float32)
            if HAS_AB:
                a = a + tl.load(AB + h[:, None] * R + r[None, :],
                                mask=mh[:, None] & mr[None, :],
                                other=0.0).to(tl.float32)[None, :, :]
            am = tl.maximum(_rate_activation(a, THR, A_HEAVY), afloor)
            tl.store(DECK + ok, dtr * am, mask=m3d)
        else:
            a = tl.load(PA + (bi * ab_ + li * al_)[:, None] + h[None, :] * ac_,
                        mask=mt[:, None] & mh[None, :], other=0.0).to(tl.float32)
            if HAS_AB:
                a = a + tl.load(AB + h, mask=mh, other=0.0).to(tl.float32)[None, :]
            am = tl.maximum(_rate_activation(a, THR, A_HEAVY), afloor)
            tl.store(DECK + ok, dtr * am[:, :, None], mask=m3d)


# ── backward ─────────────────────────────────────────────────────────
@triton.jit
def _tails_bwd_kernel(
    PB, PC, PGB, PGC, PDT, PA,
    WB, WC, BB, CB, DTB, AB,
    DBO, DCO, DDTK, DDECK,
    DB, DC, DGB, DGC, DDT, DA, PART,
    T,
    bb_, bl_, bc_, cb_, cl_, cc_, gbb, gbl, gbc, gcb, gcl, gcc,
    db_, dl_, dc_, ab_, al_, ac_,
    ob_b, ob_l, ob_n, ob_r, oc_b, oc_l, oc_n, oc_o,
    N: tl.constexpr, R: tl.constexpr, O: tl.constexpr, H: tl.constexpr,
    L: tl.constexpr,
    eps_b: tl.constexpr, eps_c: tl.constexpr,
    afloor: tl.constexpr,
    BT: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr, BOO: tl.constexpr,
    BH: tl.constexpr,
    HAS_B: tl.constexpr, HAS_C: tl.constexpr, B_NORM: tl.constexpr,
    C_NORM: tl.constexpr,
    SH_BIAS: tl.constexpr, FOLD_GB: tl.constexpr, FOLD_GC: tl.constexpr,
    HAS_DTB: tl.constexpr, HAS_AB: tl.constexpr, A_DYN: tl.constexpr,
    A_R: tl.constexpr, A_HEAVY: tl.constexpr,
    WRED: tl.constexpr, O_WB: tl.constexpr, O_WC: tl.constexpr,
    O_BB: tl.constexpr, O_CB: tl.constexpr, O_DTB: tl.constexpr,
    O_AB: tl.constexpr, THR: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    mt = t < T
    ts = tl.where(mt, t, 0)
    bi = ts // L
    li = ts - bi * L
    n = tl.arange(0, BN)
    mn = n < N
    r = tl.arange(0, BR)
    mr = r < R
    h = tl.arange(0, BH)
    mh = h < H

    if HAS_B:
        m3 = mt[:, None, None] & mn[None, :, None] & mr[None, None, :]
        nr = n[None, :, None] * R + r[None, None, :]
        x = tl.load(PB + (bi * bb_ + li * bl_)[:, None, None] + nr * bc_,
                    mask=m3, other=0.0).to(tl.float32)
        dy = tl.load(DBO + (bi * ob_b + li * ob_l)[:, None, None]
                     + n[None, :, None] * ob_n + r[None, None, :] * ob_r,
                     mask=m3, other=0.0).to(tl.float32)
        rstd = tl.zeros([BT, BR], dtype=tl.float32) + 1.0
        xhat = x
        w = tl.zeros([BN], dtype=tl.float32) + 1.0
        if B_NORM:
            rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps_b)
            xhat = x * rstd[:, None, :]
            w = tl.load(WB + n, mask=mn, other=0.0).to(tl.float32)
        y1 = xhat * w[None, :, None]
        if SH_BIAS:
            y1 = y1 + tl.load(BB + n, mask=mn, other=0.0).to(
                tl.float32)[None, :, None]
        if FOLD_GB:
            g = 1.0 + tl.load(PGB + (bi * gbb + li * gbl)[:, None, None]
                              + nr * gbc, mask=m3, other=0.0).to(tl.float32)
            tl.store(DGB + ts[:, None, None] * (N * R) + nr, dy * y1, mask=m3)
            dy = dy * g
        if O_BB >= 0:
            tl.store(PART + pid * WRED + O_BB + n,
                     tl.sum(tl.sum(dy, axis=2), axis=0), mask=mn)
        if B_NORM:
            if O_WB >= 0:
                tl.store(PART + pid * WRED + O_WB + n,
                         tl.sum(tl.sum(dy * xhat, axis=2), axis=0), mask=mn)
            dxh = dy * w[None, :, None]
            mean = tl.sum(dxh * xhat, axis=1) / N
            dx = rstd[:, None, :] * (dxh - xhat * mean[:, None, :])
        else:
            dx = dy
        tl.store(DB + ts[:, None, None] * (N * R) + nr, dx, mask=m3)

    if HAS_C:
        o = tl.arange(0, BOO)
        mo = o < O
        m3c = mt[:, None, None] & mn[None, :, None] & mo[None, None, :]
        no = n[None, :, None] * O + o[None, None, :]
        x = tl.load(PC + (bi * cb_ + li * cl_)[:, None, None] + no * cc_,
                    mask=m3c, other=0.0).to(tl.float32)
        dy = tl.load(DCO + (bi * oc_b + li * oc_l)[:, None, None]
                     + n[None, :, None] * oc_n + o[None, None, :] * oc_o,
                     mask=m3c, other=0.0).to(tl.float32)
        rstd = tl.zeros([BT, BOO], dtype=tl.float32) + 1.0
        xhat = x
        w = tl.zeros([BN], dtype=tl.float32) + 1.0
        if C_NORM:
            rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps_c)
            xhat = x * rstd[:, None, :]
            w = tl.load(WC + n, mask=mn, other=0.0).to(tl.float32)
        y1 = xhat * w[None, :, None]
        if SH_BIAS:
            y1 = y1 + tl.load(CB + n, mask=mn, other=0.0).to(
                tl.float32)[None, :, None]
        if FOLD_GC:
            g = 1.0 + tl.load(PGC + (bi * gcb + li * gcl)[:, None, None]
                              + no * gcc, mask=m3c, other=0.0).to(tl.float32)
            tl.store(DGC + ts[:, None, None] * (N * O) + no, dy * y1,
                     mask=m3c)
            dy = dy * g
        if O_CB >= 0:
            tl.store(PART + pid * WRED + O_CB + n,
                     tl.sum(tl.sum(dy, axis=2), axis=0), mask=mn)
        if C_NORM:
            if O_WC >= 0:
                tl.store(PART + pid * WRED + O_WC + n,
                         tl.sum(tl.sum(dy * xhat, axis=2), axis=0), mask=mn)
            dxh = dy * w[None, :, None]
            mean = tl.sum(dxh * xhat, axis=1) / N
            dx = rstd[:, None, :] * (dxh - xhat * mean[:, None, :])
        else:
            dx = dy
        tl.store(DC + ts[:, None, None] * (N * O) + no, dx, mask=m3c)

    m3d = mt[:, None, None] & mh[None, :, None] & mr[None, None, :]
    hr = h[None, :, None] * R + r[None, None, :]
    d = tl.load(PDT + (bi * db_ + li * dl_)[:, None, None] + hr * dc_,
                mask=m3d, other=0.0).to(tl.float32)
    if HAS_DTB:
        d = d + tl.load(DTB + h[:, None] * R + r[None, :],
                        mask=mh[:, None] & mr[None, :],
                        other=0.0).to(tl.float32)[None, :, :]
    sp = _softplus(d, THR)
    dtr = sp
    ok = (bi[:, None, None] * (R * H * L)
          + (r[None, None, :] * H + h[None, :, None]) * L
          + li[:, None, None])
    ddtr = tl.load(DDTK + ok, mask=m3d, other=0.0).to(tl.float32)
    if A_DYN:
        ddec = tl.load(DDECK + ok, mask=m3d, other=0.0).to(tl.float32)
        if A_R:
            a = tl.load(PA + (bi * ab_ + li * al_)[:, None, None]
                        + (h[None, :, None] * R + r[None, None, :]) * ac_,
                        mask=m3d, other=0.0).to(tl.float32)
            if HAS_AB:
                a = a + tl.load(AB + h[:, None] * R + r[None, :],
                                mask=mh[:, None] & mr[None, :],
                                other=0.0).to(tl.float32)[None, :, :]
            spa = _rate_activation(a, THR, A_HEAVY)
            am = tl.maximum(spa, afloor)
            ddtr = ddtr + ddec * am
            # no sum over r: each (h, r) has its own rate
            dam = tl.where(spa >= afloor, ddec * dtr, 0.0)
            da = dam * _rate_grad(a, THR, A_HEAVY)
            tl.store(DA + ts[:, None, None] * (H * R) + hr, da, mask=m3d)
            if O_AB >= 0:
                tl.store(PART + pid * WRED + O_AB + h[:, None] * R + r[None, :],
                         tl.sum(da, axis=0), mask=mh[:, None] & mr[None, :])
        else:
            a = tl.load(PA + (bi * ab_ + li * al_)[:, None] + h[None, :] * ac_,
                        mask=mt[:, None] & mh[None, :], other=0.0).to(tl.float32)
            if HAS_AB:
                a = a + tl.load(AB + h, mask=mh, other=0.0).to(tl.float32)[None, :]
            spa = _rate_activation(a, THR, A_HEAVY)
            am = tl.maximum(spa, afloor)
            ddtr = ddtr + ddec * am[:, :, None]
            dam = tl.where(spa >= afloor, tl.sum(ddec * dtr, axis=2), 0.0)
            da = dam * _rate_grad(a, THR, A_HEAVY)
            tl.store(DA + ts[:, None] * H + h[None, :], da,
                     mask=mt[:, None] & mh[None, :])
            if O_AB >= 0:
                tl.store(PART + pid * WRED + O_AB + h, tl.sum(da, axis=0),
                         mask=mh)
    dd = ddtr * _softplus_grad(d, THR)
    tl.store(DDT + ts[:, None, None] * (H * R) + hr, dd, mask=m3d)
    if O_DTB >= 0:
        tl.store(PART + pid * WRED + O_DTB + h[:, None] * R + r[None, :],
                 tl.sum(dd, axis=0), mask=mh[:, None] & mr[None, :])


# ── host side ────────────────────────────────────────────────────────
def _pow2(x: int) -> int:
    return (1 << (int(x) - 1).bit_length()) if x > 1 else 1


def _st3(t):
    """(batch, token, column) strides of a [B, L, W] source, or zeros."""
    return (0, 0, 0) if t is None else (t.stride(0), t.stride(1), t.stride(2))


def can_fuse(*srcs) -> bool:
    """Every source must be a CUDA, non-fp64, 3-D [B, L, W] tensor (any strides)."""
    for t in srcs:
        if t is None:
            continue
        if not t.is_cuda or t.dtype == torch.float64 or t.dim() != 3:
            return False
    return True


def _tail_grid_meta(cfg, Bsz):
    """(T, BT, BN, BR, BOO, BH): token count, token tile and padded block sizes."""
    BN, BR, BOO, BH = _pow2(cfg.N), _pow2(cfg.R), _pow2(cfg.O), _pow2(cfg.H)
    inner = max(BN * BR, BN * BOO, BH * BR)
    BT = max(1, min(16, 4096 // max(1, inner)))
    return Bsz * cfg.L, BT, BN, BR, BOO, BH


def _tail_out_alloc(cfg, Bsz, L, w_inner, w_outer, dt, normed, dev):
    """The B / C output buffer of the tail (shared with the op's fake)."""
    if cfg.bc_channel_major:
        # HM layout: (B, w_inner, L) memory viewed as (B, L, w_inner, 1)
        return torch.empty_strided(
            (Bsz, L, w_inner, w_outer),
            (w_inner * L, 1, L, L * w_inner), device=dev, dtype=dt)
    # r-major (..., R, N) memory when the norm runs, n-major otherwise: the
    # strides the eager chain produced, which the scan kernel's loads expect.
    st = ((L * w_outer * w_inner, w_outer * w_inner, 1, w_inner)
          if normed else
          (L * w_inner * w_outer, w_inner * w_outer, w_outer, 1))
    return torch.empty_strided((Bsz, L, w_inner, w_outer), st,
                               device=dev, dtype=dt)


class _TokenTails(torch.autograd.Function):
    """Autograd node of the fused tail (see `token_tails`)."""

    @staticmethod
    def forward(ctx, cfg, b_src, c_src, gb_src, gc_src, dt_src, a_src,
                w_b, w_c, bias_b, bias_c, bias_dt, bias_a,
                need_b, need_c, dt_b, dt_c):
        N, R, O, H, L = cfg.N, cfg.R, cfg.O, cfg.H, cfg.L
        Bsz = dt_src.shape[0]
        T = Bsz * L
        dev = dt_src.device

        T, BT, BN, BR, BOO, BH = _tail_grid_meta(cfg, Bsz)
        grid = (triton.cdiv(T, BT),)

        def _alloc(w_inner, w_outer, dt, normed):
            return _tail_out_alloc(cfg, Bsz, L, w_inner, w_outer, dt,
                                   normed, dev)

        b_out = _alloc(N, R, dt_b, cfg.norm_b) if need_b else None
        c_out = _alloc(N, O, dt_c, cfg.norm_c) if need_c else None
        ob = b_out.stride() if need_b else (0, 0, 0, 0)
        oc = c_out.stride() if need_c else (0, 0, 0, 0)
        dt_k = torch.empty(Bsz, R * H, L, device=dev,
                           dtype=cfg.dt_dtype or torch.float32)
        dec_k = torch.empty_like(dt_k) if cfg.a_dyn else None

        _tails_fwd_kernel[grid](
            b_src, c_src, gb_src, gc_src, dt_src, a_src,
            w_b, w_c, bias_b, bias_c, bias_dt, bias_a,
            b_out, c_out, dt_k, dec_k,
            T, *_st3(b_src), *_st3(c_src), *_st3(gb_src), *_st3(gc_src),
            *_st3(dt_src), *_st3(a_src),
            ob[0], ob[1], ob[2], ob[3], oc[0], oc[1], oc[2], oc[3],
            N=N, R=R, O=O, H=H, L=L,
            eps_b=cfg.eps_b, eps_c=cfg.eps_c, afloor=cfg.a_floor,
            BT=BT, BN=BN, BR=BR, BOO=BOO, BH=BH,
            HAS_B=need_b, HAS_C=need_c, B_NORM=cfg.norm_b,
            C_NORM=cfg.norm_c,
            SH_BIAS=cfg.shared_bias, FOLD_GB=cfg.fold_gb,
            FOLD_GC=cfg.fold_gc, HAS_DTB=cfg.has_dt_bias,
            HAS_AB=cfg.has_a_bias, A_DYN=cfg.a_dyn, A_R=cfg.a_per_sibling, A_HEAVY=cfg.a_heavy_tail,
            THR=_SP_THR, num_warps=4)

        ctx.cfg = cfg
        ctx.meta = (Bsz, T, BT, BN, BR, BOO, BH, need_b, need_c)
        ctx.save_for_backward(b_src, c_src, gb_src, gc_src, dt_src, a_src,
                              w_b, w_c, bias_b, bias_c, bias_dt, bias_a)
        # Outputs this config does not produce are empty non-differentiable
        # placeholders, not None: Dynamo's backward speculation calls
        # `.contiguous()` on every incoming gradient. `token_tails` maps them
        # back to None.
        outs = [b_out, c_out, dt_k, dec_k]
        ph = [t is None for t in outs]
        outs = [dt_k.new_empty(0) if p else t for t, p in zip(outs, ph)]
        if any(ph):
            ctx.mark_non_differentiable(*[t for t, p in zip(outs, ph) if p])
        return tuple(outs)

    @staticmethod
    def backward(ctx, d_b_out, d_c_out, d_dt_k, d_dec_k):
        cfg = ctx.cfg
        (Bsz, T, BT, BN, BR, BOO, BH, need_b, need_c) = ctx.meta
        (b_src, c_src, gb_src, gc_src, dt_src, a_src,
         w_b, w_c, bias_b, bias_c, bias_dt, bias_a) = ctx.saved_tensors
        N, R, O, H, L = cfg.N, cfg.R, cfg.O, cfg.H, cfg.L
        dev = dt_src.device
        grid = (triton.cdiv(T, BT),)

        # A missing incoming grad means that output went unused downstream.
        if need_b and d_b_out is None:
            d_b_out = torch.zeros(Bsz, L, N, R, device=dev, dtype=b_src.dtype)
        if need_c and d_c_out is None:
            d_c_out = torch.zeros(Bsz, L, N, O, device=dev, dtype=c_src.dtype)
        # The incoming gradient need not carry the forward output's strides
        # (it is plain contiguous (B, L, N, R) for the r-major store), so read
        # it through its own strides; the forward's would be wrong at R > 1.
        ob = d_b_out.stride() if need_b else (0, 0, 0, 0)
        oc = d_c_out.stride() if need_c else (0, 0, 0, 0)
        d_dt_k = (torch.zeros(Bsz, R * H, L, device=dev,
                              dtype=cfg.dt_dtype or torch.float32)
                  if d_dt_k is None else d_dt_k.contiguous())
        if cfg.a_dyn:
            d_dec_k = (torch.zeros_like(d_dt_k) if d_dec_k is None
                       else d_dec_k.contiguous())

        # Parameter gradients: per-block partials, then one deterministic
        # second-stage sum (no atomics: reproducible run to run).
        regions: list = []
        wred = 0

        def _reg(p, width, active):
            nonlocal wred
            if active and p is not None and p.requires_grad:
                off = wred
                regions.append((p, off, width))
                wred += width
                return off
            return -1

        o_wb = _reg(w_b, N, cfg.norm_b)
        o_wc = _reg(w_c, N, cfg.norm_c)
        o_bb = _reg(bias_b, N, cfg.shared_bias)
        o_cb = _reg(bias_c, N, cfg.shared_bias)
        o_dtb = _reg(bias_dt, H * R, cfg.has_dt_bias)
        o_ab = _reg(bias_a, H * R if cfg.a_per_sibling else H,
                    cfg.a_dyn and cfg.has_a_bias)
        part = torch.zeros(grid[0], max(wred, 1), device=dev,
                           dtype=torch.float32)

        def _emp(width, like):
            return torch.empty(Bsz, L, width, device=dev, dtype=like.dtype)

        d_b = _emp(N * R, b_src) if need_b else None
        d_c = _emp(N * O, c_src) if need_c else None
        d_gb = _emp(N * R, gb_src) if cfg.fold_gb else None
        d_gc = _emp(N * O, gc_src) if cfg.fold_gc else None
        d_dt = _emp(H * R, dt_src)
        d_a = (_emp(H * R if cfg.a_per_sibling else H, a_src)
               if cfg.a_dyn else None)

        _tails_bwd_kernel[grid](
            b_src, c_src, gb_src, gc_src, dt_src, a_src,
            w_b, w_c, bias_b, bias_c, bias_dt, bias_a,
            d_b_out, d_c_out, d_dt_k, d_dec_k,
            d_b, d_c, d_gb, d_gc, d_dt, d_a, part,
            T, *_st3(b_src), *_st3(c_src), *_st3(gb_src), *_st3(gc_src),
            *_st3(dt_src), *_st3(a_src),
            ob[0], ob[1], ob[2], ob[3], oc[0], oc[1], oc[2], oc[3],
            N=N, R=R, O=O, H=H, L=L,
            eps_b=cfg.eps_b, eps_c=cfg.eps_c, afloor=cfg.a_floor,
            BT=BT, BN=BN, BR=BR, BOO=BOO, BH=BH,
            HAS_B=need_b, HAS_C=need_c, B_NORM=cfg.norm_b,
            C_NORM=cfg.norm_c,
            SH_BIAS=cfg.shared_bias, FOLD_GB=cfg.fold_gb,
            FOLD_GC=cfg.fold_gc, HAS_DTB=cfg.has_dt_bias,
            HAS_AB=cfg.has_a_bias, A_DYN=cfg.a_dyn, A_R=cfg.a_per_sibling, A_HEAVY=cfg.a_heavy_tail,
            WRED=max(wred, 1), O_WB=o_wb, O_WC=o_wc, O_BB=o_bb, O_CB=o_cb,
            O_DTB=o_dtb, O_AB=o_ab, THR=_SP_THR,
            num_warps=4)

        gp: dict = {}
        if wred:
            red = part.sum(dim=0)
            for p, off, width in regions:
                gp[id(p)] = red[off:off + width].view(p.shape).to(p.dtype)

        def _g(p):
            return gp.get(id(p)) if p is not None else None

        return (None, d_b, d_c, d_gb, d_gc, d_dt, d_a,
                _g(w_b), _g(w_c), _g(bias_b), _g(bias_c),
                _g(bias_dt), _g(bias_a), None, None, None, None)


# ── driving the tail by hand (the combined mixer node) ──
# Same contract as `kernels/ssd_mimo/autograd.capture_scan`: inside a
# `capture_tails()` region the tail runs its forward with a stand-in ctx and
# records it, so the combined node can call `_TokenTails.backward` itself.
_CAPTURE = _threading.local()
# Present-and-None, not missing: a missing attribute makes Dynamo guard on
# `hasattr`, which the next call falsifies (a recompile per frame).
_CAPTURE.slot = None


class _ShimCtx:
    __slots__ = ("saved_tensors", "cfg", "meta")

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def mark_non_differentiable(self, *tensors):
        pass


@_contextlib.contextmanager
def capture_tails():
    """Record `token_tails` calls made inside the region; yields the list of tapes."""
    prev = getattr(_CAPTURE, "slot", None)
    _CAPTURE.slot = []
    try:
        yield _CAPTURE.slot
    finally:
        _CAPTURE.slot = prev


class TailTape:
    """One captured `token_tails` call.  `backward(dBt, dCt, ddt, ddec)`
    returns the gradients in `_TokenTails`' own input order, minus the
    leading `cfg`."""

    def __init__(self, ctx):
        self.ctx = ctx

    def backward(self, d_b_out, d_c_out, d_dt_k, d_dec_k):
        g = _TokenTails.backward(self.ctx, d_b_out, d_c_out, d_dt_k,
                                 d_dec_k)
        keys = ("b_src", "c_src", "gb_src", "gc_src", "dt_src", "a_src",
                "w_b", "w_c", "bias_b", "bias_c", "bias_dt", "bias_a")
        return dict(zip(keys, g[1:1 + len(keys)]))


# Compile boundaries (see `compile_compat`): the HM entry points below are
# opaque to Dynamo; `token_tails` takes `tails_boundary` instead, so the SSD-MIMO
# mixer stays one graph.
# Cfg <-> (ints, floats) packing for the registered custom ops.
_CFG_INT_FIELDS = ("N", "R", "O", "H", "L", "bc_norm", "shared_bias",
                   "fold_gb", "fold_gc", "a_dyn", "has_dt_bias", "has_a_bias",
                   "bc_channel_major", "a_per_sibling", "a_heavy_tail")
_CFG_FLOAT_FIELDS = ("eps_b", "eps_c", "a_floor")


def _cfg_pack(cfg):
    ints = [int(getattr(cfg, f)) if isinstance(getattr(cfg, f), bool)
            else getattr(cfg, f) for f in _CFG_INT_FIELDS]
    ints += [-1 if cfg.b_norm is None else int(cfg.b_norm),
             -1 if cfg.c_norm is None else int(cfg.c_norm)]
    return ints, [float(getattr(cfg, f)) for f in _CFG_FLOAT_FIELDS]


def _cfg_unpack(ints, floats, dt_dtype):
    kw = dict(zip(_CFG_INT_FIELDS, ints))
    for f in _CFG_INT_FIELDS[5:]:
        kw[f] = bool(kw[f])
    kw.update(zip(_CFG_FLOAT_FIELDS, floats))
    bn, cn = ints[len(_CFG_INT_FIELDS):]
    return TailCfg(dt_dtype=dt_dtype, b_norm=None if bn < 0 else bool(bn),
                   c_norm=None if cn < 0 else bool(cn), **kw)


@torch.library.custom_op("orbit_ssm::token_tails_fwd", mutates_args=())
def _tails_fwd_op(b_src: torch.Tensor | None, c_src: torch.Tensor | None,
                  gb_src: torch.Tensor | None, gc_src: torch.Tensor | None,
                  dt_src: torch.Tensor, a_src: torch.Tensor | None,
                  w_b: torch.Tensor | None, w_c: torch.Tensor | None,
                  bias_b: torch.Tensor | None, bias_c: torch.Tensor | None,
                  bias_dt: torch.Tensor | None, bias_a: torch.Tensor | None,
                  ints: list[int], floats: list[float],
                  dt_dtype: torch.dtype | None, need_b: bool, need_c: bool,
                  out_dtype_b: torch.dtype | None,
                  out_dtype_c: torch.dtype | None
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                             torch.Tensor]:
    cfg = _cfg_unpack(ints, floats, dt_dtype)
    return _TokenTails.forward(
        _ShimCtx(), cfg, b_src, c_src, gb_src, gc_src, dt_src, a_src, w_b,
        w_c, bias_b, bias_c, bias_dt, bias_a, need_b, need_c, out_dtype_b,
        out_dtype_c)


@_tails_fwd_op.register_fake
def _tails_fwd_fake(b_src, c_src, gb_src, gc_src, dt_src, a_src, w_b, w_c,
                    bias_b, bias_c, bias_dt, bias_a, ints, floats, dt_dtype,
                    need_b, need_c, out_dtype_b, out_dtype_c):
    cfg = _cfg_unpack(ints, floats, dt_dtype)
    Bsz, dev = dt_src.shape[0], dt_src.device
    dt_k = torch.empty((Bsz, cfg.R * cfg.H, cfg.L), device=dev,
                       dtype=cfg.dt_dtype or torch.float32)
    ph = dt_k.new_empty(0)
    b = (_tail_out_alloc(cfg, Bsz, cfg.L, cfg.N, cfg.R, out_dtype_b,
                         cfg.norm_b, dev) if need_b else ph)
    c = (_tail_out_alloc(cfg, Bsz, cfg.L, cfg.N, cfg.O, out_dtype_c,
                         cfg.norm_c, dev) if need_c else dt_k.new_empty(0))
    dec = torch.empty_like(dt_k) if cfg.a_dyn else dt_k.new_empty(0)
    return b, c, dt_k, dec


# Tail parameters in `_TokenTails.backward`'s output order; `_param_active`
# gives the cfg switch under which each one's gradient exists.
_TAIL_PARAMS = ("w_b", "w_c", "bias_b", "bias_c", "bias_dt", "bias_a")


def _param_active(cfg):
    return (cfg.norm_b, cfg.norm_c, cfg.shared_bias, cfg.shared_bias,
            cfg.has_dt_bias, cfg.a_dyn and cfg.has_a_bias)


@torch.library.custom_op("orbit_ssm::token_tails_bwd", mutates_args=())
def _tails_bwd_op(d_b_out: torch.Tensor | None, d_c_out: torch.Tensor | None,
                  d_dt_k: torch.Tensor | None, d_dec_k: torch.Tensor | None,
                  b_src: torch.Tensor | None, c_src: torch.Tensor | None,
                  gb_src: torch.Tensor | None, gc_src: torch.Tensor | None,
                  dt_src: torch.Tensor, a_src: torch.Tensor | None,
                  w_b: torch.Tensor | None, w_c: torch.Tensor | None,
                  bias_b: torch.Tensor | None, bias_c: torch.Tensor | None,
                  bias_dt: torch.Tensor | None, bias_a: torch.Tensor | None,
                  ints: list[int], floats: list[float],
                  dt_dtype: torch.dtype | None, need_b: bool, need_c: bool,
                  req: list[bool]) -> list[torch.Tensor]:
    """Gradients of the 12 tensor inputs via `_TokenTails.backward`; missing
    ones are empty placeholders. `req` carries the parameters' requires_grad
    (the op sees detached tensors)."""
    cfg = _cfg_unpack(ints, floats, dt_dtype)
    params = [None if p is None else p.detach().requires_grad_(r)
              for p, r in zip((w_b, w_c, bias_b, bias_c, bias_dt, bias_a),
                              req)]
    ctx = _ShimCtx()
    ctx.cfg = cfg
    Bsz = dt_src.shape[0]
    ctx.meta = (Bsz, *_tail_grid_meta(cfg, Bsz), need_b, need_c)
    ctx.saved_tensors = (b_src, c_src, gb_src, gc_src, dt_src, a_src,
                         *params)
    g = _TokenTails.backward(
        ctx, d_b_out if need_b else None, d_c_out if need_c else None,
        d_dt_k, d_dec_k if cfg.a_dyn else None)[1:13]
    ph = dt_src.new_empty(0)
    # parameter gradients view one reduction buffer; op outputs may not alias, so clone
    return [ph.clone() if t is None else (t.clone() if i >= 6 else t)
            for i, t in enumerate(g)]


@_tails_bwd_op.register_fake
def _tails_bwd_fake(d_b_out, d_c_out, d_dt_k, d_dec_k, b_src, c_src, gb_src,
                    gc_src, dt_src, a_src, w_b, w_c, bias_b, bias_c, bias_dt,
                    bias_a, ints, floats, dt_dtype, need_b, need_c, req):
    cfg = _cfg_unpack(ints, floats, dt_dtype)
    N, R, O, H, L = cfg.N, cfg.R, cfg.O, cfg.H, cfg.L
    Bsz, dev = dt_src.shape[0], dt_src.device

    def emp(width, like):
        return torch.empty((Bsz, L, width), device=dev, dtype=like.dtype)

    ph = lambda: dt_src.new_empty(0)
    out = [emp(N * R, b_src) if need_b else ph(),
           emp(N * O, c_src) if need_c else ph(),
           emp(N * R, gb_src) if cfg.fold_gb else ph(),
           emp(N * O, gc_src) if cfg.fold_gc else ph(),
           emp(H * R, dt_src),
           (emp(H * R if cfg.a_per_sibling else H, a_src) if cfg.a_dyn
            else ph())]
    for p, r, act in zip((w_b, w_c, bias_b, bias_c, bias_dt, bias_a), req,
                         _param_active(cfg)):
        out.append(torch.empty_like(p) if (p is not None and r and act)
                   else ph())
    return out


def _tails_setup(ctx, inputs, output):
    (b_src, c_src, gb_src, gc_src, dt_src, a_src, w_b, w_c, bias_b, bias_c,
     bias_dt, bias_a, ints, floats, dt_dtype, need_b, need_c, _odb,
     _odc) = inputs
    ctx.save_for_backward(b_src, c_src, gb_src, gc_src, dt_src, a_src, w_b,
                          w_c, bias_b, bias_c, bias_dt, bias_a)
    ctx.args = (list(ints), list(floats), dt_dtype, need_b, need_c)
    cfg = _cfg_unpack(ints, floats, dt_dtype)
    ph = [o for o, real in zip(output, (need_b, need_c, True, cfg.a_dyn))
          if not real]
    if ph:
        ctx.mark_non_differentiable(*ph)


def _tails_backward(ctx, d_b_out, d_c_out, d_dt_k, d_dec_k):
    saved = ctx.saved_tensors
    ints, floats, dt_dtype, need_b, need_c = ctx.args
    req = [bool(ctx.needs_input_grad[6 + i]) for i in range(6)]
    g = _tails_bwd_op(d_b_out, d_c_out, d_dt_k, d_dec_k, *saved, ints,
                      floats, dt_dtype, need_b, need_c, req)
    grads = [None if (inp is None or t.numel() == 0 and inp.numel() != 0)
             else t for t, inp in zip(g, saved)]
    return (*grads, None, None, None, None, None, None, None)


_tails_fwd_op.register_autograd(_tails_backward, setup_context=_tails_setup)


@tails_boundary
def token_tails(cfg: TailCfg, *, b_src, c_src, gb_src, gc_src, dt_src, a_src,
                w_b, w_c, bias_b, bias_c, bias_dt, bias_a,
                need_b: bool, need_c: bool, out_dtype_b, out_dtype_c):
    """Run the fused tail.  Returns ``(Bt, Ct, dt_k, dec_k)``.

    ``Bt`` / ``Ct`` are (B, L, N, R) / (B, L, N, O) in the dtype the eager
    chain would produce, or ``None`` when the config asks for no elementwise
    work on that map (the caller keeps its raw view). ``dt_k`` / ``dec_k``
    are contiguous (B, R*H, L) in ``cfg.dt_dtype`` (fp32 by default);
    ``dec_k`` is ``None`` unless ``cfg.a_dyn``.
    """
    if torch.compiler.is_compiling() and getattr(_CAPTURE, "slot", None) is None:
        # Under torch.compile use the registered custom-op pair (same
        # launchers): Dynamo cannot speculate `_TokenTails` under dynamic shapes.
        ints, floats = _cfg_pack(cfg)
        b, c, dt_k, dec_k = _tails_fwd_op(
            b_src, c_src, gb_src, gc_src, dt_src, a_src, w_b, w_c, bias_b,
            bias_c, bias_dt, bias_a, ints, floats, cfg.dt_dtype, need_b,
            need_c, out_dtype_b, out_dtype_c)
        return (b if need_b else None, c if need_c else None, dt_k,
                dec_k if cfg.a_dyn else None)
    args = (cfg, b_src, c_src, gb_src, gc_src, dt_src, a_src,
            w_b, w_c, bias_b, bias_c, bias_dt, bias_a,
            need_b, need_c, out_dtype_b, out_dtype_c)
    slot = getattr(_CAPTURE, "slot", None)
    if slot is None:
        out = _TokenTails.apply(*args)
    else:
        ctx = _ShimCtx()
        out = _TokenTails.forward(ctx, *args)
        slot.append(TailTape(ctx))
    b, c, dt_k, dec_k = out
    return (b if need_b else None, c if need_c else None, dt_k,
            dec_k if cfg.a_dyn else None)



# ═════════════════════════════════════════════════════════════════════
#  HM epilogue kernels
#
#  `gate_transpose(y_cf, z)`: the (B, D, L) -> (B, L, D) layout transition of
#      `mh_scan`'s output, optionally with the SwiGLU gate folded in (the
#      norm-free epilogue). Without the gate it is bitwise the `.contiguous()`
#      it replaces, at tiled-transpose bandwidth. Its backward writes `dy`
#      channel-major, the layout `mh_scan`'s backward wants.
#
#  `dt_softplus_cf(dt_lin, bias)`: the HM dt tail when nothing else is
#      needed. `_TokenTails` tiles tokens x columns, which is wrong for a dt
#      that is d_inner wide and channel-major; this tiles (H x L) with L
#      innermost, giving contiguous loads and stores. Same expression in
#      fp32 registers, so bitwise equal to the fused tail's dt.
# ═════════════════════════════════════════════════════════════════════

#: 64x64 transpose tile per program.
_TR_BL = 64
_TR_BD = 64

#: dt tile: long in L, thin in H, so rows are full 256 B sectors.
_DT_BL = 128
_DT_BH = 16


@triton.jit
def _silu(x):
    """`x * sigmoid(x)` in fp32."""
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _gate_tr_fwd(YP, ZP, OP, sy0, sy1, sy2, sz0, sz1, sz2, so0, so1, so2,
                 L, D, HAS_Z: tl.constexpr,
                 BL: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(2)
    ls = tl.program_id(0) * BL + tl.arange(0, BL)
    ds = tl.program_id(1) * BD + tl.arange(0, BD)
    m = (ds[:, None] < D) & (ls[None, :] < L)
    y = tl.load(YP + b * sy0 + ds[:, None] * sy1 + ls[None, :] * sy2,
                mask=m, other=0.0)
    if HAS_Z:
        z = tl.load(ZP + b * sz0 + ls[None, :] * sz1 + ds[:, None] * sz2,
                    mask=m, other=0.0).to(tl.float32)
        y = (y.to(tl.float32) * _silu(z)).to(OP.dtype.element_ty)
    tl.store(OP + b * so0 + ls[None, :] * so1 + ds[:, None] * so2, y, mask=m)


@triton.jit
def _gate_tr_bwd(GOP, YP, ZP, DYP, DZP,
                 sg0, sg1, sg2, sy0, sy1, sy2, sz0, sz1, sz2,
                 L, D, HAS_Z: tl.constexpr, NEED_Y: tl.constexpr,
                 NEED_Z: tl.constexpr, BL: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(2)
    ls = tl.program_id(0) * BL + tl.arange(0, BL)
    ds = tl.program_id(1) * BD + tl.arange(0, BD)
    m = (ds[:, None] < D) & (ls[None, :] < L)
    go = tl.load(GOP + b * sg0 + ls[None, :] * sg1 + ds[:, None] * sg2,
                 mask=m, other=0.0)
    if HAS_Z:
        z = tl.load(ZP + b * sz0 + ls[None, :] * sz1 + ds[:, None] * sz2,
                    mask=m, other=0.0).to(tl.float32)
        sg = 1.0 / (1.0 + tl.exp(-z))
        gof = go.to(tl.float32)
        if NEED_Y:
            tl.store(DYP + (b * D + ds[:, None]) * L + ls[None, :],
                     (gof * (z * sg)).to(DYP.dtype.element_ty), mask=m)
        if NEED_Z:
            y = tl.load(YP + b * sy0 + ds[:, None] * sy1 + ls[None, :] * sy2,
                        mask=m, other=0.0).to(tl.float32)
            dz = gof * y * (sg * (1.0 + z * (1.0 - sg)))
            tl.store(DZP + (b * L + ls[None, :]) * D + ds[:, None],
                     dz.to(DZP.dtype.element_ty), mask=m)
    elif NEED_Y:
        tl.store(DYP + (b * D + ds[:, None]) * L + ls[None, :], go, mask=m)


class _GateTranspose(torch.autograd.Function):
    """(B, D, L) [* silu(z)] -> (B, L, D); `dy` is returned channel-major."""

    @staticmethod
    def forward(ctx, y_cf, z):
        Bs, D, L = y_cf.shape
        out = torch.empty(Bs, L, D, device=y_cf.device, dtype=y_cf.dtype)
        grid = (triton.cdiv(L, _TR_BL), triton.cdiv(D, _TR_BD), Bs)
        _gate_tr_fwd[grid](y_cf, z, out, *y_cf.stride(),
                           *(z.stride() if z is not None else (0, 0, 0)),
                           *out.stride(), L, D, HAS_Z=z is not None,
                           BL=_TR_BL, BD=_TR_BD, num_warps=4)
        # `y_cf` is needed for dz only
        ctx.save_for_backward(*((y_cf, z) if z is not None else ()))
        ctx.shape = (Bs, D, L)
        ctx.has_z = z is not None
        return out

    @staticmethod
    def backward(ctx, go):
        Bs, D, L = ctx.shape
        need_y = ctx.needs_input_grad[0]
        need_z = ctx.needs_input_grad[1] and ctx.has_z
        y_cf, z = ctx.saved_tensors if ctx.has_z else (None, None)
        # channel-major contiguous: the layout `mh_scan`'s backward wants
        dy = (torch.empty(Bs, D, L, device=go.device, dtype=go.dtype)
              if need_y else None)
        dz = (torch.empty(Bs, L, D, device=go.device, dtype=go.dtype)
              if need_z else None)
        if dy is not None or dz is not None:
            grid = (triton.cdiv(L, _TR_BL), triton.cdiv(D, _TR_BD), Bs)
            _gate_tr_bwd[grid](
                go, y_cf, z, dy, dz, *go.stride(),
                *(y_cf.stride() if ctx.has_z else (0, 0, 0)),
                *(z.stride() if ctx.has_z else (0, 0, 0)),
                L, D, HAS_Z=ctx.has_z, NEED_Y=need_y, NEED_Z=need_z,
                BL=_TR_BL, BD=_TR_BD, num_warps=4)
        return dy, dz


@dynamo_opaque
def gate_transpose(y_cf: torch.Tensor, z: torch.Tensor | None = None):
    """``y_cf.transpose(1, 2) [* silu(z)]`` -> (B, L, D) contiguous.

    ``y_cf`` (B, D, L) is the scan kernel's channel-major output, read
    through its own strides; ``z`` is (B, L, D) or None.

    Falls back to the eager composition outside the Triton envelope (CPU,
    fp64, or `SSD_MIMO_TAILS=0`). With ``z=None`` the fused route is bitwise
    the eager one; with a gate, silu and the product are computed in fp32
    and rounded once (bf16 tolerance vs eager, the more accurate).
    """
    ok = (tails_enabled() and y_cf.is_cuda and y_cf.dim() == 3
          and y_cf.dtype in (torch.float32, torch.float16, torch.bfloat16)
          and (z is None or (z.is_cuda and z.dtype == y_cf.dtype
                             and tuple(z.shape) == (y_cf.shape[0],
                                                    y_cf.shape[2],
                                                    y_cf.shape[1]))))
    if not ok:
        y = y_cf.transpose(1, 2).contiguous()
        return y if z is None else y * torch.nn.functional.silu(z)
    return _GateTranspose.apply(y_cf, z)


# ── the wide dt tail, channel-major ──
@triton.jit
def _dt_cf_fwd(DP, BP, OP, sd0, sd1, sd2, L, H, HAS_B: tl.constexpr,
               THR: tl.constexpr, BL: tl.constexpr, BH: tl.constexpr):
    b = tl.program_id(2)
    hs = tl.program_id(1) * BH + tl.arange(0, BH)
    ls = tl.program_id(0) * BL + tl.arange(0, BL)
    m = (hs[:, None] < H) & (ls[None, :] < L)
    d = tl.load(DP + b * sd0 + hs[:, None] * sd1 + ls[None, :] * sd2,
                mask=m, other=0.0).to(tl.float32)
    if HAS_B:
        d = d + tl.load(BP + hs, mask=hs < H, other=0.0).to(
            tl.float32)[:, None]
    tl.store(OP + (b * H + hs[:, None]) * L + ls[None, :],
             _softplus(d, THR).to(OP.dtype.element_ty), mask=m)


@triton.jit
def _dt_cf_bwd(DP, BP, GP, GDP, GBP, sd0, sd1, sd2, sg0, sg1, sg2, L, H,
               HAS_B: tl.constexpr, NEED_D: tl.constexpr,
               NEED_B: tl.constexpr, THR: tl.constexpr,
               BL: tl.constexpr, BH: tl.constexpr):
    pl, b = tl.program_id(0), tl.program_id(2)
    nb = tl.num_programs(2)
    hs = tl.program_id(1) * BH + tl.arange(0, BH)
    ls = pl * BL + tl.arange(0, BL)
    m = (hs[:, None] < H) & (ls[None, :] < L)
    d = tl.load(DP + b * sd0 + hs[:, None] * sd1 + ls[None, :] * sd2,
                mask=m, other=0.0).to(tl.float32)
    if HAS_B:
        d = d + tl.load(BP + hs, mask=hs < H, other=0.0).to(
            tl.float32)[:, None]
    g = tl.load(GP + b * sg0 + hs[:, None] * sg1 + ls[None, :] * sg2,
                mask=m, other=0.0).to(tl.float32) * _softplus_grad(d, THR)
    if NEED_D:
        tl.store(GDP + (b * H + hs[:, None]) * L + ls[None, :],
                 g.to(GDP.dtype.element_ty), mask=m)
    if NEED_B:
        # per-(l-block, batch) partial, summed by the caller (no atomics)
        tl.store(GBP + (pl * nb + b) * H + hs,
                 tl.sum(tl.where(m, g, 0.0), axis=1), mask=hs < H)


class _DtSoftplusCf(torch.autograd.Function):
    """softplus(dt + bias) on a (B, H, L) channel-major dt."""

    @staticmethod
    def forward(ctx, dt_lin, bias, output_dtype):
        Bs, H, L = dt_lin.shape
        out = torch.empty(Bs, H, L, device=dt_lin.device, dtype=output_dtype or dt_lin.dtype)
        grid = (triton.cdiv(L, _DT_BL), triton.cdiv(H, _DT_BH), Bs)
        _dt_cf_fwd[grid](dt_lin, bias, out, *dt_lin.stride(), L, H,
                         HAS_B=bias is not None, THR=_SP_THR,
                         BL=_DT_BL, BH=_DT_BH, num_warps=4)
        ctx.save_for_backward(dt_lin, bias)
        return out

    @staticmethod
    def backward(ctx, g):
        dt_lin, bias = ctx.saved_tensors
        Bs, H, L = dt_lin.shape
        need_d = ctx.needs_input_grad[0]
        need_b = ctx.needs_input_grad[1] and bias is not None
        nl = triton.cdiv(L, _DT_BL)
        grid = (nl, triton.cdiv(H, _DT_BH), Bs)
        gd = (torch.empty(Bs, H, L, device=g.device, dtype=dt_lin.dtype)
              if need_d else None)
        part = (torch.empty(nl * Bs, H, device=g.device, dtype=torch.float32)
                if need_b else None)
        if need_d or need_b:
            gc = g.contiguous()
            _dt_cf_bwd[grid](dt_lin, bias, gc, gd, part, *dt_lin.stride(),
                             *gc.stride(), L, H, HAS_B=bias is not None,
                             NEED_D=need_d, NEED_B=need_b, THR=_SP_THR,
                             BL=_DT_BL, BH=_DT_BH, num_warps=4)
        return gd, (part.sum(0).to(bias.dtype) if need_b else None), None


@dynamo_opaque
def dt_softplus_cf(dt_lin: torch.Tensor, bias: torch.Tensor | None, *,
                   output_dtype: torch.dtype | None = None):
    """``softplus(dt_lin + bias[:, None])`` on a (B, H, L) CHANNEL-MAJOR dt.

    The HM dt tail when there is no B/C norm to fuse with. Returns (B, H, L)
    contiguous in ``dt_lin``'s dtype (what `mh_scan` takes), or in
    ``output_dtype`` (fp16/bf16/fp32) if given, e.g. fp32 for SSD clocks.
    Bitwise the expression `_tails_fwd_kernel` computes. Falls back to the
    eager chain outside the Triton envelope.
    """
    if output_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dt output_dtype must be fp16, bf16 or fp32")
    if not (tails_enabled() and dt_lin.is_cuda and dt_lin.dim() == 3
            and dt_lin.dtype in (torch.float32, torch.float16,
                                 torch.bfloat16)):
        d = dt_lin
        if output_dtype is not None and d.dtype != torch.float64:
            d = d.float()
        d = d if bias is None else d + bias.unsqueeze(-1)
        out = torch.nn.functional.softplus(d)
        return out if output_dtype is None else out.to(output_dtype).contiguous()
    return _DtSoftplusCf.apply(dt_lin, bias, output_dtype)
