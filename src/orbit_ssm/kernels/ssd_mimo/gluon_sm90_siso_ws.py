"""Opt-in sm90 SISO TMA producer; consumer arithmetic matches the legacy body.

Eight consumer warps retain the fp32 recurrent state and every stability
feature. Four producer warps refill the two-slot TMA ring (24-register request).
With WS_PREP=1 they also prepare the segmented prefix and bf16 dY operands
for the next chunk; a 48-register request avoids spilling that preparation. A slot is released only after its prior TMA output stores complete;
ready/free barrier phases alternate on each reuse. No consumer ping-pong,
state downcast, additional checkpoint tape or phase-law fusion is introduced.

Enable SSD_MIMO_GLUON90_SISO_WS_PRODUCER=1 through the existing host launcher.
The initial envelope is T64/N128/G64, WN2/STAGES2, SPANMMA1 or 2, dDelta
requested and standalone rotation adjoint. Unsupported calls retain the
existing route, with the reason exposed in last_route(). Default is off.

WS_PREP is separately opt-in and keeps the fp32 recurrent state. Its four
shared vectors and two bf16 operands use the same free-slot ownership as
the TMA ring, with a separate preparation-ready barrier.

The consumer is the legacy body with only TMA scheduling moved; keep numerical changes synchronized with gluon_sm90_siso_bwd.py.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from .gluon_sm90_siso_bwd import _add_combine, _pair_cross

@gluon.jit
def _ws_consumer(
    BLK_SC: gl.constexpr, BLK_Y: gl.constexpr, DD: gl.constexpr, DSF_ptr, DTH_ROT: gl.constexpr,
    DTH_STORE_VEC: gl.constexpr, DY_TMAJOR: gl.constexpr, Delta_ptr, Dsk_ptr, Dth_ptr, G: gl.constexpr,
    HAS_DSF: gl.constexpr, HAS_X0: gl.constexpr, KP: gl.constexpr, L_tot, MMA_TG: gl.constexpr,
    MMA_TN: gl.constexpr, MMA_TT: gl.constexpr, N: gl.constexpr, NARROW_DTH: gl.constexpr, NC,
    NW: gl.constexpr, P_DSK: gl.constexpr, P_SEG: gl.constexpr, P_SPAN: gl.constexpr,
    P_TRAPF: gl.constexpr, QK_ptr, ROT_KA: gl.constexpr, SPAN_MMA: gl.constexpr, SPLIT2: gl.constexpr,
    STAGES: gl.constexpr, SYNC: gl.constexpr, T: gl.constexpr, V1: gl.constexpr, X_PACKED: gl.constexpr,
    a_val, b, bars, bo, bs, cs, dDelta_ptr, dQK_ptr, dX0_ptr, dY_ptr, dbs_desc, dcs_desc, dss, dx_desc,
    dxs, dyfs_buf, dys_buf, freebars, h, ho, lges, mms, pps, s_db, s_dc, s_dh, s_dt, s_fb, s_fg, s_fh, s_fn,
    s_gb, s_gc, s_gg, s_gh, s_gt, s_hb, s_hc, s_hh, s_hk, s_ht, s_wb, s_wh, s_wl, scs, ss, vecs_buf, xrow,
    xs,
    PREP: gl.constexpr, prepbars,
):
    offs_t = gl.arange(0, T, layout=V1)
    i_tt = gl.arange(0, T, layout=gl.SliceLayout(1, MMA_TT))
    j_tt = gl.arange(0, T, layout=gl.SliceLayout(0, MMA_TT))
    t_y = gl.arange(0, T, layout=gl.SliceLayout(1, BLK_Y))
    g_y = gl.arange(0, G, layout=gl.SliceLayout(0, BLK_Y))
    t_g = gl.arange(0, T, layout=gl.SliceLayout(1, MMA_TG))
    g_g = gl.arange(0, G, layout=gl.SliceLayout(0, MMA_TG))
    n_g = gl.arange(0, N, layout=gl.SliceLayout(1, MMA_TG))
    s_sc = gl.arange(0, T, layout=gl.SliceLayout(1, BLK_SC))
    k_sc = gl.arange(0, T, layout=gl.SliceLayout(0, BLK_SC))
    below = k_sc[None, :] < s_sc[:, None]
    if SPAN_MMA > 0 and DD:
        lges.store(gl.where(j_tt[None, :] >= i_tt[:, None], 1.0, 0.0).to(gl.bfloat16))
        fence_async_shared()
        gl.barrier()
    if HAS_DSF:
        dS = gl.load(DSF_ptr + bo * s_fb + ho * s_fh + g_g[None, :] * s_fg + n_g[:, None] * s_fn).to(gl.float32)
    else:
        dS = gl.zeros([N, G], gl.float32, MMA_TG)
    if P_DSK:
        dv = gl.load(Dsk_ptr + h * G + g_g)
    for cr in range(NC):
        c = NC - 1 - cr
        slot = cr % STAGES
        phase = cr // STAGES & 1
        tma.store_wait(0)
        gl.barrier()
        if cr > 0:
            mbarrier.arrive(freebars.index((cr - 1) % STAGES))
        if PREP:
            mbarrier.wait(prepbars.index(slot), phase)
            vecs = vecs_buf
            voff = slot*4
            dys = dys_buf.index(slot)
            dyfs = dyfs_buf.index(slot)
            # f[T-1] equals alpha: exp(a*D_end) without a reset, zero
            # otherwise. This avoids a fifth vector without changing rounding.
            alpha = gl.sum(gl.where(offs_t == T-1, vecs.index(voff+2).load(V1), 0.0), axis=0)
        else:
            vecs = vecs_buf
            voff = 0
            dys = dys_buf
            dyfs = dyfs_buf
            d_raw = gl.load(Delta_ptr + bo * s_db + ho * s_dh + c * s_dc + offs_t * s_dt)
            if P_SEG:
                rst = d_raw < 0.0
                ri = gl.where(rst, 1.0, 0.0)
                D_h = gl.where(rst, 0.0, d_raw)
                seg_h = gl.associative_scan(ri, 0, _add_combine)
                seg_end = gl.max(seg_h, axis=0)
                D_end = gl.sum(gl.where(offs_t == T - 1, D_h, 0.0), axis=0)
                alpha = gl.where(seg_end == 0.0, gl.exp(a_val * D_end), 0.0)
                f_v = gl.where(seg_h == 0.0, gl.exp(a_val * D_h), 0.0)
                dagg_v = gl.where(seg_h == seg_end, gl.exp(a_val * (D_end - D_h)), 0.0)
            else:
                D_h = d_raw
                seg_h = gl.zeros([T], gl.float32, V1)
                D_end = gl.sum(gl.where(offs_t == T - 1, D_h, 0.0), axis=0)
                alpha = gl.exp(a_val * D_end)
                f_v = gl.exp(a_val * D_h)
                dagg_v = gl.exp(a_val * (D_end - D_h))
            vecs.index(voff+0).store(D_h)
            vecs.index(voff+1).store(seg_h)
            vecs.index(voff+2).store(f_v)
            vecs.index(voff+3).store(dagg_v)
            gl.barrier()
            dy_b = gl.load(dY_ptr + bo * s_gb + ho * s_gh + c * s_gc + t_y[:, None] * s_gt + g_y[None, :] * s_gg).to(gl.float32)
            f_y = vecs.index(voff+2).load(gl.SliceLayout(1, BLK_Y))
            dys.store(dy_b.to(gl.bfloat16))
            dyfs.store((dy_b * f_y[:, None]).to(gl.bfloat16))
        dss.store(dS.to(gl.bfloat16))
        fence_async_shared()
        gl.barrier()
        mbarrier.wait(bars.index(slot), phase)
        bsl = bs.index(slot)
        csl = cs.index(slot)
        if X_PACKED:
            xsl = xs.index(slot).permute((1, 0))
        else:
            xsl = xs.index(slot)
        ssl = ss.index(slot)
        dSc = warpgroup_mma(dys, xsl, gl.zeros([T, T], gl.float32, MMA_TT), use_acc=False, is_async=True)
        if SYNC:
            dSc = warpgroup_mma_wait(0, deps=[dSc])
        SB = warpgroup_mma(csl, bsl.permute((1, 0)), gl.zeros([T, T], gl.float32, MMA_TT), use_acc=False, is_async=True)
        if SYNC:
            SB = warpgroup_mma_wait(0, deps=[SB])
        dYS = warpgroup_mma(dys, ssl.permute((1, 0)), gl.zeros([T, N], gl.float32, MMA_TN), use_acc=False, is_async=True)
        if SYNC:
            dYS = warpgroup_mma_wait(0, deps=[dYS])
        xdS = warpgroup_mma(xsl.permute((1, 0)), dss.permute((1, 0)), gl.zeros([T, N], gl.float32, MMA_TN), use_acc=False, is_async=True)
        if SYNC:
            xdS = warpgroup_mma_wait(0, deps=[xdS])
            BdS = warpgroup_mma(bsl, dss, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            BdS = warpgroup_mma_wait(0, deps=[BdS])
            QS = warpgroup_mma(csl, ssl, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            QS = warpgroup_mma_wait(0, deps=[QS])
        elif SPLIT2:
            dSc, SB = warpgroup_mma_wait(2, deps=[dSc, SB])
        else:
            BdS = warpgroup_mma(bsl, dss, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            QS = warpgroup_mma(csl, ssl, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            dSc, SB = warpgroup_mma_wait(4, deps=[dSc, SB])
        D_i = vecs.index(voff+0).load(gl.SliceLayout(1, MMA_TT))
        D_j = vecs.index(voff+0).load(gl.SliceLayout(0, MMA_TT))
        sg_i = vecs.index(voff+1).load(gl.SliceLayout(1, MMA_TT))
        sg_j = vecs.index(voff+1).load(gl.SliceLayout(0, MMA_TT))
        if P_TRAPF:
            cm = i_tt[:, None] > j_tt[None, :]
        else:
            cm = i_tt[:, None] >= j_tt[None, :]
        if P_SEG:
            cm = cm & (sg_i[:, None] == sg_j[None, :])
        dec = gl.where(cm, gl.exp(a_val * (D_i[:, None] - D_j[None, :])), 0.0)
        Mm = dSc * dec
        mms.store(Mm.to(gl.bfloat16))
        scs.store((SB * dec).to(gl.bfloat16))
        if not DD:
            pass
        elif P_SPAN and SPAN_MMA > 0:
            Pf = Mm * SB
            p1 = Pf.to(gl.bfloat16)
            r1 = Pf - p1.to(gl.float32)
            pps.index(0).store(p1)
            if SPAN_MMA >= 2:
                p2 = r1.to(gl.bfloat16)
                pps.index(1).store(p2)
            if SPAN_MMA >= 3:
                pps.index(2).store((r1 - p2.to(gl.float32)).to(gl.bfloat16))
            fence_async_shared()
            gl.barrier()
            Wsp = gl.zeros([T, T], gl.float32, MMA_TT)
            for kk in gl.static_range(SPAN_MMA):
                Wsp = warpgroup_mma(lges, pps.index(SPAN_MMA - 1 - kk), Wsp, is_async=True)
        elif P_SPAN:
            P = gl.convert_layout(Mm * SB, BLK_SC)
            Rrev = gl.associative_scan(P, 0, _add_combine, reverse=True)
            E_intra = gl.sum(gl.where(below, Rrev, 0.0), axis=1)
        else:
            PT = Mm * SB
            rq = gl.convert_layout(gl.sum(PT, axis=1), V1) - gl.convert_layout(gl.sum(PT, axis=0), V1)
            E_intra = gl.associative_scan(rq, 0, _add_combine, reverse=True)
        if SPLIT2 and (not SYNC):
            BdS = warpgroup_mma(bsl, dss, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            QS = warpgroup_mma(csl, ssl, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
        if not SYNC:
            dYS, xdS, BdS, QS = warpgroup_mma_wait(0, deps=[dYS, xdS, BdS, QS])
        if DD and P_SPAN and (SPAN_MMA > 0):
            Wsp = warpgroup_mma_wait(0, deps=[Wsp])
            E_intra = gl.sum(gl.where(j_tt[None, :] < i_tt[:, None], Wsp, 0.0), axis=1)
        f_i = vecs.index(voff+2).load(gl.SliceLayout(1, MMA_TN))
        dagg_n = vecs.index(voff+3).load(gl.SliceLayout(1, MMA_TN))
        dagg_g = vecs.index(voff+3).load(gl.SliceLayout(1, MMA_TG))
        f_g = vecs.index(voff+2).load(gl.SliceLayout(1, MMA_TG))
        if DY_TMAJOR:
            dy = gl.convert_layout(gl.load(dY_ptr + bo * s_gb + ho * s_gh + c * s_gc + t_y[:, None] * s_gt + g_y[None, :] * s_gg).to(gl.float32), MMA_TG)
        else:
            dy = gl.load(dY_ptr + bo * s_gb + ho * s_gh + c * s_gc + t_g[:, None] * s_gt + g_g[None, :] * s_gg).to(gl.float32)
        xt = xsl.permute((1, 0)).load(MMA_TG).to(gl.float32)
        if DD:
            t1 = f_g * gl.sum(QS * dy, axis=1)
            vj = dagg_g * gl.sum(BdS * xt, axis=1)
            tail = alpha * gl.sum(gl.sum(ssl.load(MMA_TG).to(gl.float32) * dS, axis=1), axis=0)
        tbase = bo * s_wb + ho * s_wh + (c * T + t_g) * s_wl
        if P_TRAPF:
            dqk_t = gl.sum(dy * xt, axis=1)
            gl.store(dQK_ptr + tbase, dqk_t)
        acc_c = dYS * f_i[:, None]
        acc_b = xdS * dagg_n[:, None]
        acc_x = BdS * dagg_g[:, None]
        acc_s = dS * alpha
        fence_async_shared()
        gl.barrier()
        dCt = warpgroup_mma(mms, bsl, acc_c, is_async=True)
        if SYNC:
            dCt = warpgroup_mma_wait(0, deps=[dCt])
        dBt = warpgroup_mma(mms.permute((1, 0)), csl, acc_b, is_async=True)
        if SYNC:
            dBt = warpgroup_mma_wait(0, deps=[dBt])
        dx = warpgroup_mma(scs.permute((1, 0)), dys, acc_x, is_async=True)
        if SYNC:
            dx = warpgroup_mma_wait(0, deps=[dx])
        dSn = warpgroup_mma(csl.permute((1, 0)), dyfs, acc_s, is_async=True)
        if SYNC:
            dSn = warpgroup_mma_wait(0, deps=[dSn])
        if DD:
            t1_s = gl.convert_layout(t1, V1)
            vj_s = gl.convert_layout(vj, gl.SliceLayout(0, BLK_SC))
            if P_SPAN:
                vx = gl.convert_layout(gl.sum(gl.where(below, vj_s[None, :], 0.0), axis=1), V1)
            else:
                vv = gl.convert_layout(vj, V1)
                vx = gl.associative_scan(vv, 0, _add_combine) - vv
            E = gl.convert_layout(E_intra, V1) + gl.associative_scan(t1_s, 0, _add_combine, reverse=True) + vx + tail
            gl.store(dDelta_ptr + bo * s_db + ho * s_dh + c * s_dc + offs_t * s_dt, a_val * E)
        if not SYNC:
            dCt, dBt, dx, dSn = warpgroup_mma_wait(0, deps=[dCt, dBt, dx, dSn])
        if DTH_ROT > 0:
            if ROT_KA == N // 4 and NARROW_DTH:
                BLK_R: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
                gcr = gl.convert_layout(dCt, BLK_R)
                gbr = gl.convert_layout(dBt, BLK_R)
                Cre = csl.load(BLK_R).to(gl.float32)
                Bre = bsl.load(BLK_R).to(gl.float32)
                gc, _ = gl.split(gl.permute(gl.reshape(gcr, [T, 2, N // 2]), (0, 2, 1)))
                gb, _ = gl.split(gl.permute(gl.reshape(gbr, [T, 2, N // 2]), (0, 2, 1)))
                xc, _ = gl.split(gl.permute(gl.reshape(Cre, [T, 2, N // 2]), (0, 2, 1)))
                xb, _ = gl.split(gl.permute(gl.reshape(Bre, [T, 2, N // 2]), (0, 2, 1)))
                active = _pair_cross(gc, xc, T, N // 2) + _pair_cross(gb, xb, T, N // 2)
                zero = gl.full([T, N // 4], 0, gl.float32, active.type.layout)
                dth = gl.reshape(gl.permute(gl.join(active, zero), (0, 2, 1)), [T, N // 2])
            else:
                Cre = csl.load(MMA_TN).to(gl.float32)
                Bre = bsl.load(MMA_TN).to(gl.float32)
                dth = _pair_cross(dCt, Cre, T, N) + _pair_cross(dBt, Bre, T, N)
            if DTH_ROT == 3:
                gl.store(Dth_ptr + bo * s_hb + c * s_hc + ho * s_hh, gl.sum(gl.sum(dth, axis=1), axis=0))
            else:
                BLK_D: gl.constexpr = gl.BlockedLayout([1, DTH_STORE_VEC], [32 // (KP // DTH_STORE_VEC), KP // DTH_STORE_VEC], [NW, 1], [1, 0])
                dthv = gl.convert_layout(dth, BLK_D)
                L_DT: gl.constexpr = dthv.type.layout
                t_d = gl.arange(0, T, layout=gl.SliceLayout(1, L_DT))
                k_d = gl.arange(0, KP, layout=gl.SliceLayout(0, L_DT))
                if ROT_KA < N // 2 and (not NARROW_DTH):
                    dthv = gl.where(k_d[None, :] < ROT_KA, dthv, 0.0)
                gl.store(Dth_ptr + bo * s_hb + c * s_hc + ho * s_hh + t_d[:, None] * s_ht + k_d[None, :] * s_hk, dthv)
            gl.barrier()
        if P_TRAPF:
            qk = gl.load(QK_ptr + tbase)
            dx = dx + qk[:, None] * dy
        if P_DSK:
            dx = dx + dy * dv[None, :]
        if X_PACKED:
            dxs.store(dx.to(gl.bfloat16))
        else:
            dxs.permute((1, 0)).store(dx.to(gl.bfloat16))
        csl.store(dCt.to(gl.bfloat16))
        bsl.store(dBt.to(gl.bfloat16))
        fence_async_shared()
        gl.barrier()
        if X_PACKED:
            tma.async_store(dx_desc, [b * L_tot + c * T, h * G], dxs)
        else:
            tma.async_store(dx_desc, [xrow, c * T], dxs)
        tma.async_store(dcs_desc, [b * L_tot + c * T, h * N], csl)
        tma.async_store(dbs_desc, [b * L_tot + c * T, h * N], bsl)
        dS = dSn
    tma.store_wait(0)
    if HAS_X0:
        gl.store(dX0_ptr + bo * s_fb + ho * s_fh + g_g[None, :] * s_fg + n_g[:, None] * s_fn, dS)

@gluon.jit
def _ws_producer(bs_desc,cs_desc,x_desc,s_desc,bs,cs,xs,ss,bars,freebars,
                 b,h,H,NC,L_tot,xrow,srow,T:gl.constexpr,N:gl.constexpr,G:gl.constexpr,
                 STAGES:gl.constexpr,CBYTES:gl.constexpr,X_PACKED:gl.constexpr,
                 PREP:gl.constexpr,prepbars,vecs_buf,dys_buf,dyfs_buf,
                 Delta_ptr,dY_ptr,bo,ho,a_val,s_db,s_dh,s_dc,s_dt,s_gb,s_gh,s_gc,s_gt,s_gg,
                 P_SEG:gl.constexpr,DY_TMAJOR:gl.constexpr):
    PV: gl.constexpr = gl.BlockedLayout([1],[32],[4],[0])
    if DY_TMAJOR:
        PY: gl.constexpr = gl.BlockedLayout([4,1],[8,4],[1,4],[0,1])
    else:
        PY: gl.constexpr = gl.BlockedLayout([1,4],[32//(G//4),G//4],[4,1],[1,0])
    offs_t=gl.arange(0,T,layout=PV)
    t_y=gl.arange(0,T,layout=gl.SliceLayout(1,PY))
    g_y=gl.arange(0,G,layout=gl.SliceLayout(0,PY))
    for cr in range(NC):
        slot=cr % STAGES
        if cr >= STAGES:
            mbarrier.wait(freebars.index(slot),((cr//STAGES)-1)&1)
        c=NC-1-cr
        mb=bars.index(slot)
        mbarrier.expect(mb,CBYTES)
        tma.async_load(bs_desc,[b*L_tot+c*T,h*N],mb,bs.index(slot))
        tma.async_load(cs_desc,[b*L_tot+c*T,h*N],mb,cs.index(slot))
        if X_PACKED:
            tma.async_load(x_desc,[b*L_tot+c*T,h*G],mb,xs.index(slot))
        else:
            tma.async_load(x_desc,[xrow,c*T],mb,xs.index(slot))
        tma.async_load(s_desc,[(srow+c)*N,0],mb,ss.index(slot))

        if PREP:
            vecs=vecs_buf
            voff=slot*4
            dys=dys_buf.index(slot)
            dyfs=dyfs_buf.index(slot)
            d_raw = gl.load(Delta_ptr + bo * s_db + ho * s_dh + c * s_dc + offs_t * s_dt)
            if P_SEG:
                rst = d_raw < 0.0
                ri = gl.where(rst, 1.0, 0.0)
                D_h = gl.where(rst, 0.0, d_raw)
                seg_h = gl.associative_scan(ri, 0, _add_combine)
                seg_end = gl.max(seg_h, axis=0)
                D_end = gl.sum(gl.where(offs_t == T - 1, D_h, 0.0), axis=0)
                alpha = gl.where(seg_end == 0.0, gl.exp(a_val * D_end), 0.0)
                f_v = gl.where(seg_h == 0.0, gl.exp(a_val * D_h), 0.0)
                dagg_v = gl.where(seg_h == seg_end, gl.exp(a_val * (D_end - D_h)), 0.0)
            else:
                D_h = d_raw
                seg_h = gl.zeros([T], gl.float32, PV)
                D_end = gl.sum(gl.where(offs_t == T - 1, D_h, 0.0), axis=0)
                alpha = gl.exp(a_val * D_end)
                f_v = gl.exp(a_val * D_h)
                dagg_v = gl.exp(a_val * (D_end - D_h))
            vecs.index(voff+0).store(D_h)
            vecs.index(voff+1).store(seg_h)
            vecs.index(voff+2).store(f_v)
            vecs.index(voff+3).store(dagg_v)
            gl.barrier()
            dy_b = gl.load(dY_ptr + bo * s_gb + ho * s_gh + c * s_gc + t_y[:, None] * s_gt + g_y[None, :] * s_gg).to(gl.float32)
            f_y = vecs.index(voff+2).load(gl.SliceLayout(1, PY))
            dys.store(dy_b.to(gl.bfloat16))
            dyfs.store((dy_b * f_y[:, None]).to(gl.bfloat16))
            fence_async_shared()
            gl.barrier()
            mbarrier.arrive(prepbars.index(slot))

@gluon.jit
def _gl90_siso_bwd_ws_kernel(
    bs_desc, cs_desc, dbs_desc, dcs_desc, x_desc, dx_desc, s_desc, dY_ptr, Delta_ptr, A_ptr, DSF_ptr,
    Dsk_ptr, QK_ptr, dDelta_ptr, dQK_ptr, dX0_ptr, H, s_gb, s_gh, s_gc, s_gt, s_gg, s_db, s_dh, s_dc,
    s_dt, s_wb, s_wh, s_wl, s_fb, s_fh, s_fg, s_fn, NC, braw_desc, craw_desc, Theta_ptr, dTheta_ptr,
    W_ptr, Gam_ptr, Beta_ptr, Gamma_ptr, dW_ptr, dG_ptr, dBeta_ptr, dGamma_ptr, s_thb, s_thc, s_tht,
    s_thh, s_thk, s_pb, s_ph, s_pl, Dth_ptr, s_hb, s_hc, s_ht, s_hh, s_hk, T: gl.constexpr,
    N: gl.constexpr, G: gl.constexpr, HAS_TRAP: gl.constexpr, HAS_DSKIP: gl.constexpr,
    HAS_DSF: gl.constexpr, HAS_X0: gl.constexpr, IDX64: gl.constexpr, FUSE_ADJ: gl.constexpr=False,
    HAS_BETA: gl.constexpr=False, HAS_GAMMA: gl.constexpr=False, X_PACKED: gl.constexpr=False,
    DY_TMAJOR: gl.constexpr=False, NARROW_DTH: gl.constexpr=True, SPLIT2: gl.constexpr=False,
    WN: gl.constexpr=2, STAGES: gl.constexpr=2, SYNC: gl.constexpr=False, PRICE: gl.constexpr=0,
    SPAN_MMA: gl.constexpr=0, DTH_ROT: gl.constexpr=0, DTH_STORE_VEC: gl.constexpr=4,
    ROT_KA: gl.constexpr=0, NO_DD: gl.constexpr=False,
    PREP: gl.constexpr=False, PREG: gl.constexpr=24,
):
    NW: gl.constexpr = 4 * WN
    gl.static_assert(gl.num_warps() == NW, 'WN warpgroups, N-split')
    gl.static_assert(T == 64, 'one wgmma M tile per chunk (T = 64)')
    P_SEG: gl.constexpr = PRICE & 1 == 0
    P_SPAN: gl.constexpr = PRICE & 2 == 0
    DD: gl.constexpr = not NO_DD
    P_TRAPF: gl.constexpr = HAS_TRAP and PRICE & 4 == 0
    P_DSK: gl.constexpr = HAS_DSKIP and PRICE & 8 == 0
    MMA_TT: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN], instr_shape=[16, T // WN, 16])
    MMA_TN: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN], instr_shape=[16, N // WN, 16])
    MMA_TG: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN], instr_shape=[16, G // WN, 16])
    V1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    if DY_TMAJOR:
        BLK_Y: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [1, NW], [0, 1])
    else:
        BLK_Y: gl.constexpr = gl.BlockedLayout([1, 4], [32 // (G // 4), G // 4], [NW, 1], [1, 0])
    if WN == 2:
        BLK_SC: gl.constexpr = gl.BlockedLayout([8, 2], [4, 8], [2, 4], [0, 1])
    else:
        BLK_SC: gl.constexpr = gl.BlockedLayout([8, 4], [4, 8], [2, 2], [0, 1])
    LAY_BN: gl.constexpr = bs_desc.layout
    LAY_X: gl.constexpr = x_desc.layout
    LAY_S: gl.constexpr = s_desc.layout
    LAY_TG: gl.constexpr = gl.NVMMASharedLayout.get_default_for([T, G], gl.bfloat16)
    LAY_TT: gl.constexpr = gl.NVMMASharedLayout.get_default_for([T, T], gl.bfloat16)
    FLAT1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    TH: gl.constexpr = 32
    KP: gl.constexpr = N // 2
    b = gl.program_id(0)
    h = gl.program_id(1)
    if IDX64:
        bo = b.to(gl.int64)
        ho = h.to(gl.int64)
    else:
        bo = b
        ho = h
    a_val = -gl.exp(gl.load(A_ptr + h))
    bs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, T, N], LAY_BN)
    cs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, T, N], LAY_BN)
    if X_PACKED:
        xs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, T, G], LAY_X)
    else:
        xs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, G, T], LAY_X)
    ss = gl.allocate_shared_memory(gl.bfloat16, [STAGES, N, G], LAY_S)
    if PREP:
        dys = gl.allocate_shared_memory(gl.bfloat16, [STAGES,T,G], LAY_TG)
        dyfs = gl.allocate_shared_memory(gl.bfloat16, [STAGES,T,G], LAY_TG)
    else:
        dys = gl.allocate_shared_memory(gl.bfloat16, [T,G], LAY_TG)
        dyfs = gl.allocate_shared_memory(gl.bfloat16, [T,G], LAY_TG)
    dss = gl.allocate_shared_memory(gl.bfloat16, [N, G], LAY_S)
    mms = gl.allocate_shared_memory(gl.bfloat16, [T, T], LAY_TT)
    scs = gl.allocate_shared_memory(gl.bfloat16, [T, T], LAY_TT)
    if X_PACKED:
        dxs = gl.allocate_shared_memory(gl.bfloat16, [T, G], LAY_X)
    else:
        dxs = gl.allocate_shared_memory(gl.bfloat16, [G, T], LAY_X)
    if SPAN_MMA > 0 and DD:
        lges = gl.allocate_shared_memory(gl.bfloat16, [T, T], LAY_TT)
        pps = gl.allocate_shared_memory(gl.bfloat16, [SPAN_MMA, T, T], LAY_TT)
    if PREP:
        vecs = gl.allocate_shared_memory(gl.float32,[STAGES*4,T],FLAT1)
    else:
        vecs = gl.allocate_shared_memory(gl.float32,[5,T],FLAT1)
    bars = gl.allocate_shared_memory(gl.int64, [STAGES + 1, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES + 1):
        mbarrier.init(bars.index(i), count=1)
    if PREP:
        prepbars = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(prepbars.index(i), count=1)
    else:
        prepbars = bars  # Unused constexpr branch; no extra allocation.
    fence_async_shared()
    CBYTES: gl.constexpr = 2 * T * N * 2 + G * T * 2 + N * G * 2
    L_tot = NC * T
    xrow = (b * H + h) * G
    srow = (b * H + h) * NC
    gl.static_assert(not FUSE_ADJ and SPAN_MMA > 0 and not NO_DD)
    freebars=gl.allocate_shared_memory(gl.int64,[STAGES,1],mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(freebars.index(i),count=1)
    fence_async_shared()
    gl.barrier()
    gl.warp_specialize(((_ws_consumer,(BLK_SC, BLK_Y, DD, DSF_ptr, DTH_ROT, DTH_STORE_VEC, DY_TMAJOR, Delta_ptr, Dsk_ptr, Dth_ptr, G, HAS_DSF, HAS_X0, KP, L_tot, MMA_TG, MMA_TN, MMA_TT, N, NARROW_DTH, NC, NW, P_DSK, P_SEG, P_SPAN, P_TRAPF, QK_ptr, ROT_KA, SPAN_MMA, SPLIT2, STAGES, SYNC, T, V1, X_PACKED, a_val, b, bars, bo, bs, cs, dDelta_ptr, dQK_ptr, dX0_ptr, dY_ptr, dbs_desc, dcs_desc, dss, dx_desc, dxs, dyfs, dys, freebars, h, ho, lges, mms, pps, s_db, s_dc, s_dh, s_dt, s_fb, s_fg, s_fh, s_fn, s_gb, s_gc, s_gg, s_gh, s_gt, s_hb, s_hc, s_hh, s_hk, s_ht, s_wb, s_wh, s_wl, scs, ss, vecs, xrow, xs, PREP, prepbars,)),
                        (_ws_producer,(bs_desc,cs_desc,x_desc,s_desc,bs,cs,xs,ss,bars,freebars,
                         b,h,H,NC,L_tot,xrow,srow,T,N,G,STAGES,CBYTES,X_PACKED,PREP,prepbars,vecs,dys,dyfs,
                         Delta_ptr,dY_ptr,bo,ho,a_val,s_db,s_dh,s_dc,s_dt,s_gb,s_gh,s_gc,s_gt,s_gg,
                         P_SEG,DY_TMAJOR))),[4],[PREG])
    for i in gl.static_range(STAGES+1):
        mbarrier.invalidate(bars.index(i))
    for i in gl.static_range(STAGES):
        mbarrier.invalidate(freebars.index(i))
        if PREP:
            mbarrier.invalidate(prepbars.index(i))
