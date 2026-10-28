"""The sibling-mix (`r_mix`) readout **and the `D` skip**, in ONE
kernel.

`SSDMimoSSM._finish_forward` reduces the scan's output over the WRITE
rank `r` with a per-`(b, h, r, t)` coefficient and hands the result to
the gate / norm / `out_proj` chain in TOKEN-major order:

    out[b, l, h, g, o] = sum_r y[b, r*H + h, o, g, l] * zeta[b, l, h, r]
    zeta[b, l, h, r]   = r_mix[h, r] ( + r_mix_proj(u)[b, l, h, r] )

In torch that is a broadcast multiply over the FULL `[B, R, H, O, G, L]`
activation, a strided `.sum(1)`, and a `.permute(0, 4, 1, 3, 2)` whose
`.reshape` is a copy — three passes over an activation-sized tensor in
the forward and their adjoints in the backward (at R = 1 the whole
epilogue is a view).

This module is the same arithmetic in one pass each way, writing the
token-major buffer the norm and `out_proj` want.

── THE `D` SKIP ──────────────────────────────────────────────────────
The SSD-MIMO contract states ONE law for the branch-local
skip, and states that the skip and the readout must receive *identical*
factors -- the same sibling weight, the same output gate, gradients
included:

    out[b,l,h,g,o] = sum_r chi[b,l,h,r] * ( y[b, r*H+h, o, g, l]
                     + D[h,g,r,o] * SiLU(z[b,l,h,g,r]) * u[b,l,h,g,r] )

The skip is applied HERE for every (R, O): one code path, one
accumulation order, one adjoint.

WHY THIS MODULE AND NOT `endpoint_mix`.  `endpoint_mix` refuses
`R != 2 or O != 2` outright -- it is the R = O = 2 endpoint-correction
specialisation -- so it cannot carry the SISO route.  `rmix_reduce`'s
`R`/`O` are ordinary constexprs, it is on the epilogue of every
non-endpoint arm, and it already writes the token-major buffer the skip
has to land in.  `endpoint_mix` folds the SAME law through the shared
`_skip_term` device function below, so the two kernels cannot drift.

D LAYOUT.  ONE internal layout, `[H, G, R, O]`, normalised at the
module boundary (`_skip_gains_kernel`).  §1.5's three sharing modes are
views of it and NOT branches in here:
    "head"          (H,)        -> broadcast over (G, R, O)
    "channel"       (H, G, R)   -> broadcast over O
    "input_output"  (H, G, R, O)-> itself
At R = O = 1 the "head" and "channel" shapes carry identical values and
`[H, G, 1, 1]` is both of them: the kernel never learns that three modes
exist.
"""
from __future__ import annotations

import torch
from ..autograd_guard import no_double_backward as _no_double_backward
import triton
import triton.language as tl
from .common import ACC_SEM


def _tblk(L: int, G: int, O: int) -> int:
    """Token rows per CTA: enough work to cover the launch, small enough
    that the O output accumulators stay in registers."""
    budget = max(1, 8192 // max(1, G * O))
    t = 1
    while t * 2 <= min(64, budget) and t * 2 <= L:
        t *= 2
    return t


@triton.jit
def _gated_drive(U_ptr, Z_ptr, s_ub, s_ul, s_uh, s_ug, s_ur,
                 s_zb, s_zl, s_zh, s_zg, s_zr,
                 b, h, r, offs_t, offs_g, mask_tg, HAS_GATE: tl.constexpr):
    """`SiLU(z) * u` for one `(b, h, r)` token tile -- §1.5's skip input.

    The gate is the SAME per-subsystem SwiGLU gate the scan applies to
    the readout (`gz_k`, channel order `(r, h, g)`), so the skip and the
    readout are gated by one tensor and one expression.
    """
    u = tl.load(U_ptr + b * s_ub + offs_t[:, None] * s_ul + h * s_uh
                + offs_g[None, :] * s_ug + r * s_ur,
                mask=mask_tg, other=0.0).to(tl.float32)
    if HAS_GATE:
        z = tl.load(Z_ptr + b * s_zb + offs_t[:, None] * s_zl + h * s_zh
                    + offs_g[None, :] * s_zg + r * s_zr,
                    mask=mask_tg, other=0.0).to(tl.float32)
        u = u * (z * tl.sigmoid(z))
    return u


@triton.jit
def _skip_term(xg, D_ptr, s_dh, s_dg, s_dr, s_do, h, r, o, offs_g, g_mask):
    """§1.5's `D[h,g,r,o] * SiLU(z) * u` for one `(h, r, o)` tile.

    `xg` is `_gated_drive`'s result.  Shared verbatim with
    `endpoint_mix` so the two epilogues cannot drift apart.
    """
    dv = tl.load(D_ptr + h * s_dh + offs_g * s_dg + r * s_dr + o * s_do,
                 mask=g_mask, other=0.0).to(tl.float32)
    return xg * dv[None, :]


@triton.jit
def _rmix_fwd_kernel(
    Y_ptr, C_ptr, Rm_ptr, Out_ptr, D_ptr, U_ptr, Z_ptr,
    s_yb, s_yh, s_yo, s_yg, s_yl,
    s_mb, s_ml, s_mh, s_mr,
    s_ob, s_ol, s_oh, s_og, s_oo,
    s_dh, s_dg, s_dr, s_do,
    s_ub, s_ul, s_uh, s_ug, s_ur,
    s_zb, s_zl, s_zh, s_zg, s_zr,
    L, H,
    TB: tl.constexpr, G: tl.constexpr, R: tl.constexpr, O: tl.constexpr,
    HAS_DYN: tl.constexpr, HAS_C: tl.constexpr,
    HAS_D: tl.constexpr, HAS_GATE: tl.constexpr,
):
    tb = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    offs_t = tb * TB + tl.arange(0, TB)
    m = offs_t < L
    BG: tl.constexpr = triton.next_power_of_2(G)
    offs_g = tl.arange(0, BG)
    g_mask = offs_g < G
    mask_tg = m[:, None] & g_mask[None, :]

    # o OUTER, r inner: one [TB, G] accumulator lives at a time (the
    # sibling coefficient is a [TB] vector, so recomputing it per o is
    # free beside re-tiling the activation).
    ob = Out_ptr + b * s_ob + h * s_oh
    for o in tl.static_range(O):
        acc = tl.zeros([TB, BG], dtype=tl.float32)
        for r in tl.static_range(R):
            # No mix at all (`r_mix="none"`) is the plain sum over r, i.e.
            # the coefficient is ONE, not zero.
            z = tl.full([TB], 0.0 if (HAS_C or HAS_DYN) else 1.0,
                        tl.float32)
            if HAS_C:
                z += tl.load(C_ptr + h * R + r).to(tl.float32)
            if HAS_DYN:
                z += tl.load(Rm_ptr + b * s_mb + offs_t * s_ml
                             + h * s_mh + r * s_mr, mask=m,
                             other=0.0).to(tl.float32)
            yt = tl.load(Y_ptr + b * s_yb + (r * H + h) * s_yh + o * s_yo
                         + offs_g[None, :] * s_yg + offs_t[:, None] * s_yl,
                         mask=mask_tg, other=0.0).to(tl.float32)
            if HAS_D:
                # §1.5: the skip enters the sibling sum BESIDE the
                # readout, so it picks up the same `z` below by
                # construction rather than by a second multiply.
                # The gated drive does not depend on `o`; it is re-tiled
                # per o rather than held in O live registers, which is
                # +(O-1) activation-sized L2 reads and no register cost.
                xg = _gated_drive(U_ptr, Z_ptr,
                                  s_ub, s_ul, s_uh, s_ug, s_ur,
                                  s_zb, s_zl, s_zh, s_zg, s_zr,
                                  b, h, r, offs_t, offs_g, mask_tg,
                                  HAS_GATE)
                yt += _skip_term(xg, D_ptr, s_dh, s_dg, s_dr, s_do,
                                 h, r, o, offs_g, g_mask)
            if HAS_C or HAS_DYN:
                acc += yt * z[:, None]
            else:
                acc += yt
        tl.store(ob + o * s_oo + offs_g[None, :] * s_og
                 + offs_t[:, None] * s_ol,
                 acc.to(Out_ptr.dtype.element_ty), mask=mask_tg)


@triton.jit
def _rmix_bwd_kernel(
    Y_ptr, C_ptr, Rm_ptr, dOut_ptr, dY_ptr, dC_ptr, dRm_ptr,
    D_ptr, U_ptr, Z_ptr, dD_ptr, dU_ptr, dZ_ptr,
    s_yb, s_yh, s_yo, s_yg, s_yl,
    s_mb, s_ml, s_mh, s_mr,
    s_qb, s_ql, s_qh, s_qr,
    s_ob, s_ol, s_oh, s_og, s_oo,
    s_dh, s_dg, s_dr, s_do,
    s_ub, s_ul, s_uh, s_ug, s_ur,
    s_zb, s_zl, s_zh, s_zg, s_zr,
    L, H,
    TB: tl.constexpr, G: tl.constexpr, R: tl.constexpr, O: tl.constexpr,
    HAS_DYN: tl.constexpr, HAS_C: tl.constexpr,
    HAS_D: tl.constexpr, HAS_GATE: tl.constexpr,
):
    tb = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    offs_t = tb * TB + tl.arange(0, TB)
    m = offs_t < L
    BG: tl.constexpr = triton.next_power_of_2(G)
    offs_g = tl.arange(0, BG)
    g_mask = offs_g < G
    mask_tg = m[:, None] & g_mask[None, :]

    ob = dOut_ptr + b * s_ob + h * s_oh
    for r in tl.static_range(R):
        z = tl.full([TB], 0.0 if (HAS_C or HAS_DYN) else 1.0, tl.float32)
        if HAS_C:
            z += tl.load(C_ptr + h * R + r).to(tl.float32)
        if HAS_DYN:
            z += tl.load(Rm_ptr + b * s_mb + offs_t * s_ml
                         + h * s_mh + r * s_mr, mask=m,
                         other=0.0).to(tl.float32)
        yb = Y_ptr + b * s_yb + (r * H + h) * s_yh
        db = dY_ptr + b * s_yb + (r * H + h) * s_yh
        dz = tl.zeros([TB], dtype=tl.float32)
        if HAS_D:
            # ONE load of the drive / gate per (r) here -- the backward
            # keeps r OUTER, so unlike the forward nothing is re-tiled.
            u = tl.load(U_ptr + b * s_ub + offs_t[:, None] * s_ul
                        + h * s_uh + offs_g[None, :] * s_ug + r * s_ur,
                        mask=mask_tg, other=0.0).to(tl.float32)
            if HAS_GATE:
                zg = tl.load(Z_ptr + b * s_zb + offs_t[:, None] * s_zl
                             + h * s_zh + offs_g[None, :] * s_zg
                             + r * s_zr, mask=mask_tg,
                             other=0.0).to(tl.float32)
                sig = tl.sigmoid(zg)
                silu = zg * sig
                # d/dz SiLU(z) = sigma(z) (1 + z (1 - sigma(z)))
                silu_p = sig * (1.0 + zg * (1.0 - sig))
            else:
                silu = tl.full([TB, BG], 1.0, tl.float32)
                silu_p = tl.zeros([TB, BG], tl.float32)
            xg = u * silu
            # adjoint of the gated drive, summed over o
            dxg = tl.zeros([TB, BG], dtype=tl.float32)
        for o in tl.static_range(O):
            go = tl.load(ob + o * s_oo + offs_g[None, :] * s_og
                         + offs_t[:, None] * s_ol,
                         mask=mask_tg, other=0.0).to(tl.float32)
            yt = tl.load(yb + o * s_yo + offs_g[None, :] * s_yg
                         + offs_t[:, None] * s_yl,
                         mask=mask_tg, other=0.0).to(tl.float32)
            tl.store(db + o * s_yo + offs_g[None, :] * s_yg
                     + offs_t[:, None] * s_yl,
                     (go * z[:, None]).to(dY_ptr.dtype.element_ty),
                     mask=mask_tg)
            if HAS_D:
                sk = _skip_term(xg, D_ptr, s_dh, s_dg, s_dr, s_do,
                                h, r, o, offs_g, g_mask)
                # the sibling weight sees the SKIP as well -- §1.5's
                # "identical factors, including gradients" is this line
                yt += sk
                w = go * z[:, None]
                dv = tl.load(D_ptr + h * s_dh + offs_g * s_dg
                             + r * s_dr + o * s_do,
                             mask=g_mask, other=0.0).to(tl.float32)
                dxg += w * dv[None, :]
                # dD is [H, G, R, O] -- tiny, and every CTA of the token
                # grid contributes, so this is the coalesced cross-CTA
                # reduction atomics are the default for.
                tl.atomic_add(dD_ptr + h * s_dh + offs_g * s_dg
                              + r * s_dr + o * s_do,
                              tl.sum(tl.where(mask_tg, w * xg, 0.0), axis=0),
                              mask=g_mask, sem=ACC_SEM)
            dz += tl.sum(yt * go, axis=1)
        if HAS_D:
            tl.store(dU_ptr + b * s_ub + offs_t[:, None] * s_ul
                     + h * s_uh + offs_g[None, :] * s_ug + r * s_ur,
                     (dxg * silu).to(dU_ptr.dtype.element_ty),
                     mask=mask_tg)
            if HAS_GATE:
                tl.store(dZ_ptr + b * s_zb + offs_t[:, None] * s_zl
                         + h * s_zh + offs_g[None, :] * s_zg + r * s_zr,
                         (dxg * u * silu_p).to(dZ_ptr.dtype.element_ty),
                         mask=mask_tg)
        if HAS_DYN:
            tl.store(dRm_ptr + b * s_qb + offs_t * s_ql
                     + h * s_qh + r * s_qr, dz, mask=m)
        if HAS_C:
            tl.atomic_add(dC_ptr + h * R + r,
                          tl.sum(tl.where(m, dz, 0.0), axis=0), sem=ACC_SEM)


_ZERO5 = (0, 0, 0, 0, 0)


class _RMixReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, c, rmix, H, R, O, G, out_dtype, d, drive, gate):
        Bsz = y.shape[0]
        L = y.shape[-1]
        # VIEWS only: `y` reaches here as the scan's own (possibly
        # channel-trimmed) strided output and every access below goes
        # through its full stride set, so an O = 1 output is unsqueezed
        # rather than reshaped -- a `.reshape` there would be the copy
        # this kernel exists to remove.
        yv = y if y.dim() == 5 else y.unsqueeze(2)
        assert yv.shape[1] == R * H and yv.shape[2] == O, \
            f"rmix_reduce: y {tuple(y.shape)} vs R{R} H{H} O{O}"
        if d is not None:
            if tuple(d.shape) != (H, G, R, O):
                raise ValueError(
                    "rmix_reduce: d must be the normalised [H, G, R, O] "
                    f"gain, got {tuple(d.shape)} (H={H} G={G} R={R} O={O})")
            if not d.is_contiguous():
                # `dD` is allocated DENSE and the backward addresses it
                # with `d`'s stride set, so a non-contiguous gain would
                # scatter the PARAMETER gradient to the wrong elements.
                # The module always hands over a `.contiguous()` gain
                # (`_skip_gains_kernel`); this refuses rather than trusts.
                raise ValueError(
                    "rmix_reduce: d must be contiguous (the dD adjoint is "
                    f"addressed with its stride set, got {tuple(d.stride())})")
            if drive is None:
                raise ValueError("rmix_reduce: d needs its drive operand")
            if tuple(drive.shape) != (Bsz, L, H, G, R):
                raise ValueError(
                    "rmix_reduce: drive must be [B, L, H, G, R], got "
                    f"{tuple(drive.shape)}")
            if gate is not None and tuple(gate.shape) != (Bsz, L, H, G, R):
                raise ValueError(
                    "rmix_reduce: gate must be [B, L, H, G, R], got "
                    f"{tuple(gate.shape)}")
            # The backward mirrors `drive` / `gate` with `empty_strided`,
            # so a BROADCAST operand would alias its own gradient rows
            # into one another and silently under-count.  Refuse it here
            # rather than return a wrong adjoint.
            for name, t in (("drive", drive), ("gate", gate)):
                if t is not None and 0 in t.stride():
                    raise ValueError(
                        f"rmix_reduce: {name} has a zero stride "
                        f"{tuple(t.stride())} -- expand() it into real "
                        "memory first, the adjoint cannot alias")
        elif drive is not None or gate is not None:
            raise ValueError("rmix_reduce: drive/gate without d")
        out = torch.empty(Bsz, L, H, G, O, device=y.device, dtype=out_dtype)
        TB = _tblk(L, G, O)
        ys, os_ = yv.stride(), out.stride()
        ms = rmix.stride() if rmix is not None else (0, 0, 0, 0)
        ds = d.stride() if d is not None else (0, 0, 0, 0)
        us = drive.stride() if drive is not None else _ZERO5
        zs = gate.stride() if gate is not None else _ZERO5
        empty = y.new_empty(0)
        grid = (triton.cdiv(L, TB), Bsz, H)
        _rmix_fwd_kernel[grid](
            yv, c, rmix, out,
            d if d is not None else empty,
            drive if drive is not None else empty,
            gate if gate is not None else empty,
            ys[0], ys[1], ys[2], ys[3], ys[4],
            ms[0], ms[1], ms[2], ms[3],
            os_[0], os_[1], os_[2], os_[3], os_[4],
            ds[0], ds[1], ds[2], ds[3],
            us[0], us[1], us[2], us[3], us[4],
            zs[0], zs[1], zs[2], zs[3], zs[4],
            L, H, TB=TB, G=G, R=R, O=O,
            HAS_DYN=rmix is not None, HAS_C=c is not None,
            HAS_D=d is not None, HAS_GATE=gate is not None,
        )
        ctx.save_for_backward(yv, c, rmix, d, drive, gate)
        ctx.meta = (H, R, O, G, TB, Bsz, L, y.dim())
        return out.reshape(Bsz, L, H * G * O)

    @staticmethod
    @_no_double_backward("SSD-MIMO mixer")
    def backward(ctx, dout):
        yv, c, rmix, d, drive, gate = ctx.saved_tensors
        H, R, O, G, TB, Bsz, L, ydim = ctx.meta
        dout = dout.reshape(Bsz, L, H, G, O).contiguous()
        dY = torch.empty_strided(yv.shape, yv.stride(), device=yv.device,
                                 dtype=yv.dtype)
        if L % TB:
            # the token tail is masked OFF in the kernel, so those rows
            # would come back uninitialised
            dY.zero_()
        dC = (torch.zeros_like(c, dtype=torch.float32)
              if c is not None else None)
        dRm = (torch.zeros(rmix.shape, device=rmix.device,
                           dtype=rmix.dtype)
               if rmix is not None else None)
        has_d = d is not None
        has_gate = gate is not None
        # dD is an ATOMIC accumulation over the whole token grid, so it
        # must start at zero and it must be fp32 (`atomic_add` on bf16 is
        # not a reduction we would trust for a parameter gradient).
        dD = torch.zeros(d.shape, device=d.device,
                         dtype=torch.float32) if has_d else None
        dU = (torch.empty_strided(drive.shape, drive.stride(),
                                  device=drive.device, dtype=drive.dtype)
              if has_d else None)
        dZ = (torch.empty_strided(gate.shape, gate.stride(),
                                  device=gate.device, dtype=gate.dtype)
              if has_gate else None)
        if has_d and L % TB:
            dU.zero_()
            if has_gate:
                dZ.zero_()
        ys, os_ = yv.stride(), dout.stride()
        ms = rmix.stride() if rmix is not None else (0, 0, 0, 0)
        # `dRm` is a fresh DENSE tensor while `rmix` is typically a slice
        # of the fused token-map slab, so the two stride sets differ.
        qs = dRm.stride() if dRm is not None else (0, 0, 0, 0)
        ds = d.stride() if has_d else (0, 0, 0, 0)
        us = drive.stride() if has_d else _ZERO5
        zs = gate.stride() if has_gate else _ZERO5
        empty = yv.new_empty(0)
        grid = (triton.cdiv(L, TB), Bsz, H)
        _rmix_bwd_kernel[grid](
            yv, c, rmix, dout, dY, dC, dRm,
            d if has_d else empty,
            drive if has_d else empty,
            gate if has_gate else empty,
            dD if has_d else empty,
            dU if has_d else empty,
            dZ if has_gate else empty,
            ys[0], ys[1], ys[2], ys[3], ys[4],
            ms[0], ms[1], ms[2], ms[3],
            qs[0], qs[1], qs[2], qs[3],
            os_[0], os_[1], os_[2], os_[3], os_[4],
            ds[0], ds[1], ds[2], ds[3],
            us[0], us[1], us[2], us[3], us[4],
            zs[0], zs[1], zs[2], zs[3], zs[4],
            L, H, TB=TB, G=G, R=R, O=O,
            HAS_DYN=rmix is not None, HAS_C=c is not None,
            HAS_D=has_d, HAS_GATE=has_gate,
        )
        return (dY if ydim == 5 else dY.squeeze(2),
                None if dC is None else dC.to(c.dtype), dRm,
                None, None, None, None, None,
                None if dD is None else dD.to(d.dtype), dU, dZ)


def rmix_reduce(y, c, rmix, H, R, O, G, out_dtype,
                d=None, drive=None, gate=None):
    """§1.5's sibling readout, skip included, in one pass each way.

        out[b,l,h,g,o] = sum_r chi[b,l,h,r] * ( y[b, r*H+h, o, g, l]
                         + d[h,g,r,o] * SiLU(gate[b,l,h,g,r])
                                      * drive[b,l,h,g,r] )
        chi[b,l,h,r]   = c[h,r] + rmix[b,l,h,r]

    `c` is `[H, R]` (None when the sibling mix is off — then the
    coefficient is 1 and this is the plain sum over `r`), `rmix` is
    `[B, L, H, R]` (None when the mix is static).  `d` is the NORMALISED
    `[H, G, R, O]` skip gain (None when `d_skip` is off); it requires
    `drive` `[B, L, H, G, R]` and takes `gate` `[B, L, H, G, R]`
    (pre-SiLU, None when the mixer is ungated).  Returns a DENSE
    token-major `[B, L, H*G*O]` in `out_dtype`.
    """
    return _RMixReduce.apply(y, c, rmix, H, R, O, G, out_dtype,
                             d, drive, gate)
