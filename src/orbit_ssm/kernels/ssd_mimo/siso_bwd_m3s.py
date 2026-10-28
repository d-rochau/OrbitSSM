"""SISO scan backward in Mamba-3's `dqkv` STRUCTURE (opt-in, M3STRUCT).

Design modelled on Mamba-3's `dqkv` kernel (`mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py`, mamba_ssm 2.3.2.post1, Apache-2.0, Tri Dao & Albert Gu): one CTA per (head, batch), a reverse chunk loop, TMA-staged operands, each [T, T] tile formed immediately before its one consumer, the state gradient carried in registers. The implementation and all arithmetic are our own:

  * the decay mask, f, alpha and dec_agg come from the SEGMENTED chunk
    prefix (`fused_fwd_kernel._SEGPREFIX_NOTE`),
  * under the trapezoid the score tile is STRICTLY causal
    (`_TRAPFOLD_NOTE`; the own-tap diagonal is the rank-1 `qk` term),
  * dL/ddelta is NOT Mamba-3's dADT (`colsum - rowsum` per token, then
    `total + cumsum - x`): that is exactly the retired historical form (4)
     and has no segmented prefix (5).  Here the
    scan emits only the two STATE-dependent spanning-pair pieces -- the
    write vector Vj (reaches every s > j) and the carry scalar -- and the
    map-side spanning pairs + read term come from the chunk-parallel
    `_siso_ddelta_map_kernel`; the host adds the exclusive prefix of Vj as
    a shifted cumsum.  No term is formed as a difference of totals
    (`_SPANDELTA_NOTE`).
  * the operand roundings match the Triton scan (bf16 masked tiles, bf16
    dS operand, fp32 accumulate), so the result is the same function with
    a different MMA k-order.

Optional DTH_ROT (as in the Triton scan): dL/dtheta in the rotated frame
from the fp32 dB~ / dC~ accumulators.

Reachable only through `SSD_MIMO_SISO_BWD_M3STRUCT=1` on the SISO route;
every out-of-envelope call is refused by name (`refusal`) and runs the
unchanged Triton scan.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from .. import portable as _portable
from .common import pdot  # FP32DOT operand rounding

from .fused_fwd_kernel import _dcs_seg, _same_seg, _pair_swap_2d


def enabled() -> bool:
    return os.environ.get("SSD_MIMO_SISO_BWD_M3STRUCT", "0") == "1"


_LAST = {"used": False, "why": "not run"}


def _cfgs():
    # ONE config (no benchmarking, so no replay): the kernel writes dB~ /
    # dC~ IN PLACE over the maps it reads, which an autotuner's replay
    # would corrupt.  `SSD_MIMO_M3S_CFG=warps,stages` pins another one.
    w, st = os.environ.get("SSD_MIMO_M3S_CFG", "4,2").split(",")
    return [triton.Config({}, num_warps=int(w), num_stages=int(st))]


@triton.jit
def _pc(g, u, rot_sign, T: tl.constexpr, N: tl.constexpr):
    """dtheta_k = g_e u_o - g_o u_e (see `siso_bwd_kernel._pair_contract`)."""
    q = rot_sign[None, :] * g * _pair_swap_2d(u, T, N)
    e, o = tl.split(tl.reshape(q, (T, N // 2, 2)))
    return e + o


@triton.autotune(configs=_cfgs(),
                 key=["T", "N", "G", "HAS_TRAP", "HAS_DSKIP", "HAS_DSF",
                      "DTH_ROT", "NO_DD"])
@triton.jit
def _siso_bwd_m3s_kernel(
    B_ptr, C_ptr, dB_ptr, dC_ptr,     # maps / their grads, [B*L, H*N]
    X_ptr, dX_ptr,                    # [B*H*G, L]
    dY_ptr,                           # [B*H*L, G] fp32 (the SISO fwd's token-major dY)
    S_ptr,                            # [B*H*NC*N, G]
    Delta_ptr, A_ptr, DSF_ptr, Dsk_ptr, QK_ptr,
    dQK_ptr, Vj_ptr, Tail_ptr, Dth_ptr,
    s_fb, s_fh, s_fg, s_fn,
    s_hb, s_hc, s_ht, s_hh, s_hk,
    s_wb, s_wh, s_wl,
    NC, H,
    T: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
    HAS_TRAP: tl.constexpr, HAS_DSKIP: tl.constexpr, HAS_DSF: tl.constexpr,
    DTH_ROT: tl.constexpr,
    NO_DD: tl.constexpr = False,
):
    h = tl.program_id(0)
    b = tl.program_id(1)
    L = NC * T
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_g = tl.arange(0, G)
    neg_inf = float("-inf")
    if HAS_TRAP:
        tri = offs_t[:, None] > offs_t[None, :]
        triT = offs_t[None, :] > offs_t[:, None]
    else:
        tri = offs_t[:, None] >= offs_t[None, :]
        triT = offs_t[None, :] >= offs_t[:, None]

    # ── TMA descriptors (Mamba-3's operand sourcing) ──────────────────
    b_desc = tl.make_tensor_descriptor(
        B_ptr + b * L * H * N + h * N, shape=[L, N], strides=[H * N, 1],
        block_shape=[T, N])
    c_desc = tl.make_tensor_descriptor(
        C_ptr + b * L * H * N + h * N, shape=[L, N], strides=[H * N, 1],
        block_shape=[T, N])
    db_desc = tl.make_tensor_descriptor(
        dB_ptr + b * L * H * N + h * N, shape=[L, N], strides=[H * N, 1],
        block_shape=[T, N])
    dc_desc = tl.make_tensor_descriptor(
        dC_ptr + b * L * H * N + h * N, shape=[L, N], strides=[H * N, 1],
        block_shape=[T, N])
    xrow = (b * H + h) * G
    x_desc = tl.make_tensor_descriptor(
        X_ptr + xrow * L, shape=[G, L], strides=[L, 1], block_shape=[G, T])
    dx_desc = tl.make_tensor_descriptor(
        dX_ptr + xrow * L, shape=[G, L], strides=[L, 1], block_shape=[G, T])
    ybase = (b * H + h) * L * G
    dy_desc = tl.make_tensor_descriptor(
        dY_ptr + ybase, shape=[L, G], strides=[G, 1], block_shape=[T, G])
    s_desc = tl.make_tensor_descriptor(
        S_ptr + (b * H + h) * NC * N * G, shape=[NC * N, G], strides=[G, 1],
        block_shape=[N, G])

    if HAS_DSF:
        dS = tl.load(DSF_ptr + b * s_fb + h * s_fh + offs_g[None, :] * s_fg
                     + offs_n[:, None] * s_fn).to(tl.float32)
    else:
        dS = tl.zeros([N, G], dtype=tl.float32)
    if HAS_DSKIP:
        dv = tl.load(Dsk_ptr + h * G + offs_g)
    if DTH_ROT:
        rot_sign = tl.where((offs_n % 2) == 0, 1.0, -1.0)

    for cr in range(NC):
        c = NC - 1 - cr
        # decays (segmented prefix) -- first, as Mamba-3 loads its da_cs
        # "to overlap computation with TMA loads"
        dbase = Delta_ptr + ((b * H + h) * NC + c) * T
        D_h, seg_h = _dcs_seg(dbase, offs_t, 1, a_val, False)
        seg_end = tl.max(seg_h, axis=0)
        D_end = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
        f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
        alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end), 0.0)
        dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end - D_h)),
                           0.0)
        same = _same_seg(seg_h)

        dYop = dy_desc.load([c * T, 0]).to(tl.bfloat16)   # [T, G]
        xT = x_desc.load([0, c * T])            # [G, T]
        Bt = b_desc.load([c * T, 0])            # [T, N]
        Ct = c_desc.load([c * T, 0])            # [T, N]
        dSop = dS.to(tl.bfloat16)

        # ── dB~ (Mamba-3's dK): (x dY^T o L^T) C + dec_agg x dS^T ───────
        dpT = pdot(tl.trans(xT), tl.trans(dYop))                # [Tj, Ti]
        dpT = dpT * tl.exp(tl.where(triT & same,
                                    a_val * (D_h[None, :] - D_h[:, None]),
                                    neg_inf))
        acc_dB = pdot(dpT.to(tl.bfloat16), Ct)
        acc_dB += dec_agg[:, None] * pdot(tl.trans(xT), tl.trans(dSop))
        if DTH_ROT:
            dth = _pc(acc_dB, Bt.to(tl.float32), rot_sign, T, N)
        db_desc.store([c * T, 0], acc_dB.to(tl.bfloat16))

        # ── dC~ (Mamba-3's dQ): (dY x^T o L) B + f dY S^T ──────────────
        s_blk = pdot(dYop, xT)                                  # [Ti, Tj]
        s_blk = s_blk * tl.exp(tl.where(tri & same,
                                        a_val * (D_h[:, None] - D_h[None, :]),
                                        neg_inf))
        acc_dC = pdot(s_blk.to(tl.bfloat16), Bt)
        St = s_desc.load([c * N, 0])                              # [N, G]
        acc_dC += f[:, None] * pdot(dYop, tl.trans(St))
        if DTH_ROT:
            dth += _pc(acc_dC, Ct.to(tl.float32), rot_sign, T, N)
            tl.store(Dth_ptr + b * s_hb + c * s_hc + h * s_hh
                     + offs_t[:, None] * s_ht
                     + tl.arange(0, N // 2)[None, :] * s_hk, dth)
        dc_desc.store([c * T, 0], acc_dC.to(tl.bfloat16))

        # ── dx (Mamba-3's dV): (B C^T o L^T) dY + dec_agg B dS ─────────
        pT = pdot(Bt, tl.trans(Ct))                             # [Tj, Ti]
        pT = pT * tl.exp(tl.where(triT & same,
                                  a_val * (D_h[None, :] - D_h[:, None]),
                                  neg_inf))
        acc_dx = pdot(pT.to(tl.bfloat16), dYop)
        BdS = pdot(Bt, dSop)                                     # [T, G]
        acc_dx += dec_agg[:, None] * BdS
        # the elementwise terms on RELOADED tiles (Mamba-3's volatile
        # dO_reloaded / v_block_reloaded)
        xr = tl.trans(tl.load(X_ptr + xrow * L + offs_g[:, None] * L
                              + c * T + offs_t[None, :],
                              volatile=True).to(tl.float32))       # [T, G]
        dyr = tl.load(dY_ptr + ybase + (c * T + offs_t[:, None]) * G
                      + offs_g[None, :], volatile=True).to(tl.float32)
        tb = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
        if HAS_TRAP:
            acc_dx += tl.load(QK_ptr + tb)[:, None] * dyr
            tl.store(dQK_ptr + tb, tl.sum(dyr * xr, axis=1))
        if HAS_DSKIP:
            acc_dx += dyr * dv[None, :]
        dx_desc.store([0, c * T], tl.trans(acc_dx).to(tl.bfloat16))

        # ── the two state-dependent dDelta pieces (vector + scalar) ────
        if not NO_DD:
            Vj = dec_agg * tl.sum(BdS * xr, axis=1)
            tl.store(Vj_ptr + ((b * H + h) * NC + c) * T + offs_t, Vj)
            Sre = s_desc.load([c * N, 0]).to(tl.float32)
            tl.store(Tail_ptr + (b * H + h) * NC + c,
                     alpha * tl.sum(Sre * dS))

        # ── state gradient for the previous chunk ──────────────────────
        dS = alpha * dS + pdot(tl.trans(Ct),
                                 (dyr * f[:, None]).to(tl.bfloat16))


def refusal(*, Bs, Cs, x, dY, S_all, delta_cumsum, T, N, G, io_tag, m_st,
            xtm, abl, x0, dskip_dd) -> str | None:
    if not enabled():
        return "off"
    if _portable.nvidia_capability(Bs.device)[0] < 9:
        return "TMA needs sm_90+"
    if io_tag != "bf16":
        return "bf16 IO only"
    if abl:
        return "ABL attribution not served"
    if xtm != "0":
        return "family x layout only"
    if x0 is not None:
        return "x0 not served"
    if dskip_dd:
        return "DSKIP_DD not served"
    if T not in (32, 64) or N not in (64, 128) or G not in (16, 32, 64):
        return f"tile (T={T}, N={N}, G={G}) not served"
    for nm, t in (("Bs", Bs), ("Cs", Cs), ("x", x), ("S_all", S_all)):
        if t.dtype != torch.bfloat16:
            return f"{nm} must be bf16"
        if t.data_ptr() % 16:
            return f"{nm} not 16-byte aligned"
    Bsz, nc, T_, H, N_ = Bs.shape
    want = [nc * T_ * H * N_, T_ * H * N_, H * N_, N_, 1]
    if list(m_st) != want or list(Bs.stride()) != want \
            or list(Cs.stride()) != want:
        return "maps must be contiguous [B, nc, T, H, N]"
    if x.dim() != 5 or x.shape[3] != 1:
        return "x must be the family layout [B, H, G, 1, L]"
    if dY.dtype != torch.float32 or dY.data_ptr() % 16:
        return f"dY must be fp32, 16-byte aligned (got {dY.dtype})"
    if dY.dim() != 6 or dY.shape[4] != 1 or not dY.is_contiguous():
        return "dY must be the contiguous token-major [B, H, nc, T, 1, G]"
    if not delta_cumsum.is_contiguous():
        return "delta_cumsum must be contiguous [B, H, nc, T]"
    if S_all.shape != (Bsz, H, nc, N, G) or not S_all.is_contiguous():
        return "S_all must be contiguous [B, H, nc, N, G]"
    return None


_ALLOC = False


def _alloc():
    global _ALLOC
    if not _ALLOC:
        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, device="cuda",
                                                    dtype=torch.int8))
        _ALLOC = True


def launch(*, Bs, Cs, dBs, dCs, x, dX, dY, S_all, delta_cumsum, a_log, dsf,
           dsk_c, qk_f, dQK, vj, tail, dth, dth_st, f_st, w_st, has_trap,
           has_dskip, has_dsf, dth_rot, no_dd=False):
    _alloc()
    Bsz, nc, T, H, N = Bs.shape
    G = x.shape[2]
    _LAST.update(used=True, why=None)
    _siso_bwd_m3s_kernel[(H, Bsz)](
        Bs, Cs, dBs, dCs, x, dX, dY, S_all,
        delta_cumsum, a_log.float().contiguous(), dsf, dsk_c, qk_f,
        dQK, vj, tail, dth,
        *f_st, *dth_st, *(w_st if has_trap else (0, 0, 0)),
        nc, H,
        T=T, N=N, G=G, HAS_TRAP=has_trap, HAS_DSKIP=has_dskip,
        HAS_DSF=has_dsf, DTH_ROT=dth_rot, NO_DD=no_dd)
