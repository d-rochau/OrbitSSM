
from __future__ import annotations
import os
import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.extra import libdevice
from . import autotune, slp
from .common_gluon import INV_TWO_PI, K_DYN, K_MU, LOG2E, TWO_PI, _add, _cmul, _divisor_le, _f32, _gate, _rate, _row1, _silu_fast, _softplus, _ssm_combine, _ssm_combine_q, selector_window
from .layout import DYN_SOFTPLUS, GATE_SIGMOID, LAW_A_DYN, LAW_A_FIX, LAW_MU, validate_forward_inputs

@gluon.constexpr_function
def _lay(BLOCK_P, BLOCK_L, NUM_WARPS, TPW_L):
    tp_p = min(32, max(BLOCK_P // NUM_WARPS, 1))
    if TPW_L > 0:
        tp_p = min(tp_p, max(32 // TPW_L, 1))
    tp_p = 1 << tp_p.bit_length() - 1
    tp_l = max(32 // tp_p, 1)
    spt_p = max(BLOCK_P // (tp_p * NUM_WARPS), 1)
    spt_l = max(BLOCK_L // tp_l, 1)
    return gl.BlockedLayout([spt_p, spt_l], [tp_p, tp_l], [NUM_WARPS, 1], [1, 0])

@gluon.jit
def _ldt(ptr, mask, EVEN: gl.constexpr):
    if EVEN:
        v = gl.load(ptr)
    else:
        v = gl.load(ptr, mask=mask, other=0.0)
    return v

@gluon.jit
def _zt(v, mask, EVEN: gl.constexpr):
    if EVEN:
        w = v
    else:
        w = gl.where(mask, v, 0.0)
    return w

@gluon.jit
def _mh_fwd_kernel(x_ptr, delta_ptr, dw_ptr, A_log_ptr, mu_gain_ptr, B_ptr, C_ptr, gB_ptr, gC_ptr, a_dyn_ptr, mu_ptr, mu_init_ptr, beta_a_ptr, beta_b_ptr, beta_c_ptr, beta_gB_ptr, beta_gC_ptr, delta_bias_ptr, row_law_ptr, row_rot_ptr, theta_ptr, thh_ptr, Y_ptr, h_state_ptr, th_state_ptr, ckpt_ptr, d_skip_ptr, z_ptr, endpoint_ptr, H, P, N, L, K, N_OFF, N_ROWS, CKS, LAW_MASK: gl.constexpr, HAS_ENDPOINT: gl.constexpr, ENDPOINT_P: gl.constexpr, ROT: gl.constexpr, HAS_RROT: gl.constexpr, THETA_DELTA: gl.constexpr, HAS_THH: gl.constexpr, THETA_PH: gl.constexpr, DELTA_SOFTPLUS: gl.constexpr, DYN_ACT: gl.constexpr, GATE_ACT: gl.constexpr, A_DYN_N_RES: gl.constexpr, MU_N_RES: gl.constexpr, HAS_DW: gl.constexpr, DW_PER_P: gl.constexpr, HAS_GB: gl.constexpr, HAS_GC: gl.constexpr, HAS_BA: gl.constexpr, HAS_BB: gl.constexpr, HAS_BC: gl.constexpr, HAS_BGB: gl.constexpr, HAS_BGC: gl.constexpr, HAS_DBIAS: gl.constexpr, HAS_MU_INIT: gl.constexpr, MU_PREDIFF: gl.constexpr, HG_A: gl.constexpr, HG_BC: gl.constexpr, HG_G: gl.constexpr, DYN_EPS: gl.constexpr, BLOCK_L: gl.constexpr, BLOCK_P: gl.constexpr, NUM_WARPS: gl.constexpr, TPW_L: gl.constexpr, ABL: gl.constexpr=0, CKPT_L: gl.constexpr=0, CKPACK: gl.constexpr=False, CKPT_COMPACT: gl.constexpr=False, BC_HEAD: gl.constexpr=False, FAST_TRIG: gl.constexpr=False, EVEN_L: gl.constexpr=False, PF: gl.constexpr=0, HAS_DSKIP: gl.constexpr=False, HAS_Z: gl.constexpr=False, PMAP: gl.constexpr=False, SLP: gl.constexpr=False, FPACK: gl.constexpr=False, TS: gl.constexpr=1):
    HAS_DYN: gl.constexpr = LAW_MASK >> 1 & 1
    HAS_MU: gl.constexpr = LAW_MASK >> 2 & 1
    UNI: gl.constexpr = LAW_MASK & LAW_MASK - 1 == 0
    PFA: gl.constexpr = PF
    NPAIR: gl.constexpr = 2 if ROT else 1
    INV_BP: gl.constexpr = 1.0 / BLOCK_P
    tile: gl.constexpr = _lay(BLOCK_P, BLOCK_L, NUM_WARPS, TPW_L)
    lt: gl.constexpr = gl.SliceLayout(0, tile)
    pl: gl.constexpr = gl.SliceLayout(1, tile)
    row1: gl.constexpr = _row1(tile)
    if CKPT_COMPACT:
        cl: gl.constexpr = gl.BlockedLayout(tile.size_per_thread, tile.threads_per_warp, tile.warps_per_cta, [0, 1])
    pid_p = gl.program_id(0)
    h = gl.program_id(1)
    if FPACK:
        h = h * BLOCK_P
    if TS > 1:
        b = (gl.program_id(2) // TS).to(gl.int64)
        ts = gl.program_id(2) % TS
        nbs = ((L + BLOCK_L - 1) // BLOCK_L + TS - 1) // TS
        t_lo = ts * nbs * BLOCK_L
        t_hi = gl.minimum(t_lo + nbs * BLOCK_L, L)
        h_state_ptr += ts.to(gl.int64) * (gl.num_programs(2) // TS) * H * N * P
    else:
        b = gl.program_id(2).to(gl.int64)
        t_lo = 0
        t_hi = L
    bh = b * H + h
    thb = (bh if THETA_PH else b) * K
    if BC_HEAD:
        B_ptr += h.to(gl.int64) * gl.num_programs(2) * N * L
        C_ptr += h.to(gl.int64) * gl.num_programs(2) * N * L
    ga = h // HG_A
    gbc = h // HG_BC
    gg = h // HG_G
    z2 = gl.zeros([BLOCK_P, BLOCK_L], gl.float32, layout=tile)
    zl = gl.zeros([BLOCK_L], gl.int32, layout=lt)
    zp = gl.zeros([BLOCK_P], gl.int32, layout=pl)
    col = gl.arange(0, BLOCK_L, layout=lt)
    first2 = gl.expand_dims(col == 0, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tile)
    last2 = gl.expand_dims(col == BLOCK_L - 1, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tile)
    idx_prv = gl.maximum(col - 1, 0)
    pf = (pid_p * BLOCK_P + gl.arange(0, BLOCK_P, layout=pl)).to(gl.int64)
    hp = bh * P + pf
    if FPACK:
        anp = (h.to(gl.int64) + pf) * N * P
        hnp = (bh + pf) * N * P
        rowL = gl.expand_dims((bh + pf) * L, 1)
    else:
        anp = h.to(gl.int64) * N * P + pf
        hnp = bh * N * P + pf
    snc = H * P if CKPACK else P
    if CKPACK:
        ckb = b.to(gl.int64) * CKS * N * snc + h.to(gl.int64) + pf
    else:
        ckb = bh * CKS * N * P + pf
    if HAS_DBIAS:
        if FPACK:
            d_bias2 = gl.expand_dims(gl.load(delta_bias_ptr + h + pf), 1)
        else:
            d_bias = gl.load(delta_bias_ptr + h + zl)
    for s in range(t_lo, t_hi, BLOCK_L):
        jj = s // BLOCK_L
        offs_t = s + col
        mask_t = offs_t < L
        m2 = gl.expand_dims(mask_t, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tile)
        if FPACK:
            d_raw2 = _ldt(delta_ptr + rowL + gl.expand_dims(offs_t, 0), m2, EVEN_L).to(gl.float32)
            if HAS_DBIAS:
                d_raw2 = d_raw2 + d_bias2
            if DELTA_SOFTPLUS:
                de2 = _softplus(d_raw2)
            else:
                de2 = d_raw2
            de2 = _zt(de2, m2, EVEN_L)
            de = gl.sum(de2, axis=0)
        else:
            d_raw = _ldt(delta_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32)
            if HAS_DBIAS:
                d_raw = d_raw + d_bias
            if DELTA_SOFTPLUS:
                de = _softplus(d_raw)
            else:
                de = d_raw
            de = _zt(de, mask_t, EVEN_L)
        po = gl.expand_dims(hp, 1) * L + gl.expand_dims(offs_t, 0)
        xt = _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        if HAS_Z:
            zg = _silu_fast(_ldt(z_ptr + po, m2, EVEN_L).to(gl.float32))
        if HAS_DW:
            if DW_PER_P:
                dwr = _ldt(dw_ptr + po, m2, EVEN_L).to(gl.float32)
            else:
                dwr = _ldt(dw_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32)
            if DELTA_SOFTPLUS:
                dwv = _softplus(dwr)
            else:
                dwv = dwr
            if DW_PER_P:
                dw = _zt(dwv, m2, EVEN_L)
            else:
                dw = gl.expand_dims(_zt(dwv, mask_t, EVEN_L), 0)
        elif FPACK:
            dw = de2
        else:
            dw = gl.expand_dims(de, 0)
        xw = dw * xt
        if HAS_ENDPOINT:
            if ENDPOINT_P:
                eo = _ldt(endpoint_ptr + po, m2, EVEN_L).to(gl.float32)
            else:
                eo = gl.expand_dims(_ldt(endpoint_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32), 0)
        if HAS_DYN and (not A_DYN_N_RES):
            rate_h = _rate(_ldt(a_dyn_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32), DYN_ACT, DYN_EPS)
        if HAS_MU and (not MU_N_RES):
            mu_h = _ldt(mu_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32)
            if MU_PREDIFF:
                s2_h = gl.expand_dims(_zt(mu_h, mask_t, EVEN_L), 0)
            else:
                mu_ph = gl.load(mu_ptr + bh * L + offs_t - 1, mask=mask_t & (offs_t > 0), other=0.0).to(gl.float32)
                if HAS_MU_INIT:
                    mu_ph = gl.where(offs_t == 0, gl.load(mu_init_ptr + bh + zl), mu_ph)
                s2_h = gl.expand_dims(_zt(mu_h - mu_ph, mask_t, EVEN_L), 0)
        yacc = z2
        if HAS_ENDPOINT:
            bcac = gl.zeros([BLOCK_L], gl.float32, layout=lt)
        if PF and (not ROT):
            bcp = (b * N + N_OFF).to(gl.int64) * L
            Bnx = _ldt(B_ptr + bcp + offs_t, mask_t, EVEN_L).to(gl.float32)
            Cnx = _ldt(C_ptr + bcp + offs_t, mask_t, EVEN_L).to(gl.float32)
        if PFA:
            Anx = gl.load(A_log_ptr + anp + N_OFF * P)
        if ROT and PF:
            thn = _ldt(theta_ptr + (thb + N_OFF // NPAIR) * L + offs_t, mask_t, EVEN_L)
        if PMAP and PF:
            mrn = (b * 2 * N + 2 * N_OFF).to(gl.int64) * L
            Pn0 = _ldt(B_ptr + mrn + offs_t, mask_t, EVEN_L)
            Pn1 = _ldt(B_ptr + (mrn + L) + offs_t, mask_t, EVEN_L)
            Qn0 = _ldt(C_ptr + mrn + offs_t, mask_t, EVEN_L)
            Qn1 = _ldt(C_ptr + (mrn + L) + offs_t, mask_t, EVEN_L)
        for kk in range(N_OFF // NPAIR, (N_OFF + N_ROWS) // NPAIR):
            if ROT:
                thp = th_state_ptr + (bh * gl.num_programs(0) + pid_p).to(gl.int64) * K + kk
                if PF:
                    thv = thn.to(gl.float32)
                    thn = _ldt(theta_ptr + (thb + gl.minimum(kk + 1, (N_OFF + N_ROWS) // NPAIR - 1).to(gl.int64)) * L + offs_t, mask_t, EVEN_L)
                else:
                    thv = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_t, mask_t, EVEN_L).to(gl.float32)
                if THETA_DELTA:
                    thv = thv * de
                inc1 = gl.expand_dims(gl.convert_layout(_zt(thv, mask_t, EVEN_L), gl.SliceLayout(0, row1)), 0)
                if not ABL & 64:
                    inc1 = gl.associative_scan(inc1, axis=1, combine_fn=_add)
                Th = gl.convert_layout(gl.sum(inc1, axis=0), lt) + gl.load(thp)
                if ABL & 256:
                    cT, sT = (Th, Th)
                elif FAST_TRIG:
                    cT = libdevice.fast_cosf(Th)
                    sT = libdevice.fast_sinf(Th)
                else:
                    cT = gl.cos(Th)
                    sT = gl.sin(Th)
                tc = gl.sum(gl.where(col == BLOCK_L - 1, Th, 0.0), axis=0)
                if NUM_WARPS > 1:
                    gl.barrier()
                gl.store(thp, tc - TWO_PI * libdevice.floor(tc * INV_TWO_PI))
                if NUM_WARPS > 1:
                    gl.barrier()
            if ROT:
                if HAS_THH:
                    cn_h = gl.load(thh_ptr + (h * K + kk) * 2)
                    sn_h = gl.load(thh_ptr + (h * K + kk) * 2 + 1)
                    cC = cT * cn_h - sT * sn_h
                    sC = sT * cn_h + cT * sn_h
                else:
                    cC = cT
                    sC = sT
            if ROT and (not PMAP):
                p0 = (b * N + kk * 2).to(gl.int64) * L
                B0 = _ldt(B_ptr + p0 + offs_t, mask_t, EVEN_L).to(gl.float32)
                C0 = _ldt(C_ptr + p0 + offs_t, mask_t, EVEN_L).to(gl.float32)
                if ABL & 128:
                    B1, C1 = (B0, C0)
                else:
                    B1 = _ldt(B_ptr + (p0 + L) + offs_t, mask_t, EVEN_L).to(gl.float32)
                    C1 = _ldt(C_ptr + (p0 + L) + offs_t, mask_t, EVEN_L).to(gl.float32)
                if HAS_BB:
                    bb = beta_b_ptr + gbc * N + kk * 2
                    B0 = B0 + gl.load(bb + zl)
                    B1 = B1 + gl.load(bb + 1 + zl)
                if HAS_BC:
                    cb = beta_c_ptr + gbc * N + kk * 2
                    C0 = C0 + gl.load(cb + zl)
                    C1 = C1 + gl.load(cb + 1 + zl)
            for pi in gl.static_range(NPAIR):
                n = kk * NPAIR + pi
                bc = (b * N + n).to(gl.int64) * L
                if not UNI:
                    law_s = gl.load(row_law_ptr + n)
                if PMAP:
                    if PF:
                        P0, P1 = (Pn0.to(gl.float32), Pn1.to(gl.float32))
                        Q0, Q1 = (Qn0.to(gl.float32), Qn1.to(gl.float32))
                        mrn = (b * 2 * N + 2 * gl.minimum(n + 1, N_OFF + N_ROWS - 1)).to(gl.int64) * L
                        Pn0 = _ldt(B_ptr + mrn + offs_t, mask_t, EVEN_L)
                        Pn1 = _ldt(B_ptr + (mrn + L) + offs_t, mask_t, EVEN_L)
                        Qn0 = _ldt(C_ptr + mrn + offs_t, mask_t, EVEN_L)
                        Qn1 = _ldt(C_ptr + (mrn + L) + offs_t, mask_t, EVEN_L)
                    else:
                        mr = (b * 2 * N + 2 * n).to(gl.int64) * L
                        P0 = _ldt(B_ptr + mr + offs_t, mask_t, EVEN_L).to(gl.float32)
                        P1 = _ldt(B_ptr + (mr + L) + offs_t, mask_t, EVEN_L).to(gl.float32)
                        Q0 = _ldt(C_ptr + mr + offs_t, mask_t, EVEN_L).to(gl.float32)
                        Q1 = _ldt(C_ptr + (mr + L) + offs_t, mask_t, EVEN_L).to(gl.float32)
                    if pi == 0:
                        Bt, Ct = (P0 * cT - sT * P1, Q0 * cC - sC * Q1)
                    else:
                        Bt, Ct = (P1 * cT + sT * P0, Q1 * cC + sC * Q0)
                    if HAS_RROT:
                        rr = gl.load(row_rot_ptr + n + zl) != 0
                        Bt = gl.where(rr, Bt, P0 if pi == 0 else P1)
                        Ct = gl.where(rr, Ct, Q0 if pi == 0 else Q1)
                elif ROT:
                    if pi == 0:
                        Bt, Ct = (B0 * cT - sT * B1, C0 * cC - sC * C1)
                    else:
                        Bt, Ct = (B1 * cT + sT * B0, C1 * cC + sC * C0)
                    if HAS_RROT:
                        rr = gl.load(row_rot_ptr + n + zl) != 0
                        Bt = gl.where(rr, Bt, B0 if pi == 0 else B1)
                        Ct = gl.where(rr, Ct, C0 if pi == 0 else C1)
                elif ABL & 16:
                    Bt = de
                    Ct = de
                elif PF:
                    Bt, Ct = (Bnx, Cnx)
                    bcn = (b * N + gl.minimum(n + 1, N_OFF + N_ROWS - 1)).to(gl.int64) * L
                    Bnx = _ldt(B_ptr + bcn + offs_t, mask_t, EVEN_L).to(gl.float32)
                    Cnx = _ldt(C_ptr + bcn + offs_t, mask_t, EVEN_L).to(gl.float32)
                    if HAS_BB:
                        Bt = Bt + gl.load(beta_b_ptr + gbc * N + n + zl)
                    if HAS_BC:
                        Ct = Ct + gl.load(beta_c_ptr + gbc * N + n + zl)
                else:
                    Bt = _ldt(B_ptr + bc + offs_t, mask_t, EVEN_L).to(gl.float32)
                    Ct = _ldt(C_ptr + bc + offs_t, mask_t, EVEN_L).to(gl.float32)
                    if HAS_BB:
                        Bt = Bt + gl.load(beta_b_ptr + gbc * N + n + zl)
                    if HAS_BC:
                        Ct = Ct + gl.load(beta_c_ptr + gbc * N + n + zl)
                if HAS_GB:
                    gv = _ldt(gB_ptr + bc + offs_t, mask_t, EVEN_L).to(gl.float32)
                    if HAS_BGB:
                        gv = gv + gl.load(beta_gB_ptr + gg * N + n + zl)
                    Bt = Bt * _gate(gv, GATE_ACT)
                if HAS_GC:
                    gv = _ldt(gC_ptr + bc + offs_t, mask_t, EVEN_L).to(gl.float32)
                    if HAS_BGC:
                        gv = gv + gl.load(beta_gC_ptr + gg * N + n + zl)
                    Ct = Ct * _gate(gv, GATE_ACT)
                Ct = _zt(Ct, mask_t, EVEN_L)
                s1 = de
                if HAS_DYN and (UNI or law_s == K_DYN):
                    if A_DYN_N_RES:
                        s1 = de * _rate(_ldt(a_dyn_ptr + bc + offs_t, mask_t, EVEN_L).to(gl.float32), DYN_ACT, DYN_EPS)
                    else:
                        s1 = de * rate_h
                if not FPACK:
                    s1 = _zt(s1, mask_t, EVEN_L)
                if FPACK:
                    s1_2 = de2
                else:
                    s1_2 = gl.expand_dims(s1, 0)
                if HAS_BA:
                    b_a = gl.load(beta_a_ptr + ga * N + n + zp)
                else:
                    b_a = gl.zeros([BLOCK_P], gl.float32, layout=pl)
                if ABL & 512:
                    c1 = b_a - 1.0
                elif PFA:
                    c1 = -gl.exp(Anx + b_a)
                    Anx = gl.load(A_log_ptr + anp + gl.minimum(n + 1, N_OFF + N_ROWS - 1) * P)
                else:
                    c1 = -gl.exp(gl.load(A_log_ptr + anp + n * P) + b_a)
                c1e = gl.expand_dims(c1 * LOG2E, 1)
                log_a = c1e * s1_2
                if HAS_MU and (UNI or law_s == K_MU):
                    if MU_N_RES:
                        mu_t = _ldt(mu_ptr + bc + offs_t, mask_t, EVEN_L).to(gl.float32)
                        if MU_PREDIFF:
                            s2_2 = gl.expand_dims(_zt(mu_t, mask_t, EVEN_L), 0)
                        else:
                            mu_p = gl.gather(mu_t, idx_prv, 0)
                            at0 = (col == 0) & mask_t
                            mu_p = gl.where(at0, gl.load(mu_ptr + bc + offs_t - 1, mask=at0 & (offs_t > 0), other=0.0).to(gl.float32), mu_p)
                            if HAS_MU_INIT:
                                mu_p = gl.where(offs_t == 0, gl.load(mu_init_ptr + b * N + n + zl), mu_p)
                            s2_2 = gl.expand_dims(_zt(mu_t - mu_p, mask_t, EVEN_L), 0)
                    else:
                        s2_2 = s2_h
                    log_a = log_a + gl.expand_dims(gl.load(mu_gain_ptr + anp + n * P) * LOG2E, 1) * s2_2
                if ABL & 1:
                    a = log_a
                else:
                    a = libdevice.exp2(log_a)
                drive = xw * gl.expand_dims(Bt, 0)
                if not ABL & 8:
                    hprev = gl.load(h_state_ptr + hnp + n * P)
                    drive = gl.where(first2, drive + a * gl.expand_dims(hprev, 1), drive)
                if ABL & 2:
                    hs = drive + a
                    qh = a
                elif HAS_ENDPOINT:
                    qs = gl.where(first2, a, 0.0)
                    if not ABL & 8:
                        qs = qs * gl.expand_dims(hprev, 1)
                    _, hs, qh = gl.associative_scan((a, drive, qs), axis=1, combine_fn=_ssm_combine_q)
                else:
                    _, hs = gl.associative_scan((a, drive), axis=1, combine_fn=_ssm_combine)
                if HAS_ENDPOINT:
                    yacc += qh * gl.expand_dims(Ct, 0)
                    bcac += Bt * Ct
                else:
                    yacc += hs * gl.expand_dims(Ct, 0)
                if ABL & 32:
                    gl.store(h_state_ptr + hnp + n * P, c1)
                else:
                    gl.store(h_state_ptr + hnp + n * P, gl.sum(gl.where(last2, hs, 0.0), axis=1))
                if CKPT_L > 0:
                    if CKPT_COMPACT:
                        hc = gl.convert_layout(hs.to(ckpt_ptr.dtype.element_ty), cl)
                        pc = pid_p * BLOCK_P + gl.arange(0, BLOCK_P, layout=gl.SliceLayout(1, cl))
                        cc = gl.arange(0, BLOCK_L, layout=gl.SliceLayout(0, cl))
                        slot = (jj * (BLOCK_L // CKPT_L) + cc // CKPT_L + 1).to(gl.int64)
                        if CKPACK:
                            ck_base = b * CKS * N * snc + h.to(gl.int64) + pc
                        else:
                            ck_base = bh * CKS * N * P + pc
                        gl.store(ckpt_ptr + gl.expand_dims(ck_base, 1) + (gl.expand_dims(slot, 0) * N + n) * snc, hc, mask=gl.expand_dims(cc % CKPT_L == CKPT_L - 1, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=cl))
                    else:
                        slot = (jj * (BLOCK_L // CKPT_L) + col // CKPT_L + 1).to(gl.int64)
                        gl.store(ckpt_ptr + gl.expand_dims(ckb, 1) + (gl.expand_dims(slot, 0) * N + n) * snc, hs.to(ckpt_ptr.dtype.element_ty), mask=gl.expand_dims(col % CKPT_L == CKPT_L - 1, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tile))
        if HAS_ENDPOINT or HAS_DSKIP:
            xe = _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        if HAS_ENDPOINT:
            yacc += eo * xe * gl.expand_dims(bcac, 0)
        if HAS_DSKIP:
            if FPACK:
                yacc += gl.expand_dims(gl.load(d_skip_ptr + h + pf), 1) * xe
            else:
                yacc += gl.load(d_skip_ptr + h) * xe
        if HAS_Z:
            yacc = yacc * zg
        if EVEN_L:
            gl.store(Y_ptr + po, yacc)
        else:
            gl.store(Y_ptr + po, yacc, mask=m2)

def _thh_table(theta_h):
    if theta_h is None:
        return None
    t = theta_h.detach().float()
    return torch.stack((torch.cos(t), torch.sin(t)), dim=-1).contiguous()
LAST_COMPILED = None
LAST_CONFIG = None
ELEMS_PER_THREAD = 16
TILE_P = 1
MAX_BLOCK_L = 512

def _block_l(BLOCK_P, NUM_WARPS, L):
    want = 32 * NUM_WARPS * ELEMS_PER_THREAD // max(BLOCK_P, 1)
    return min(triton.next_power_of_2(L), max(32, min(256, triton.next_power_of_2(max(want, 1)))))
WAVE_CTAS = 2048
BP2_L_MAX = 2048
_SM = None

def _sm():
    global _SM
    if _SM is None:
        try:
            major, minor = torch.cuda.get_device_capability()
            _SM = major * 10 + minor
        except Exception:
            _SM = 0
    return _SM

def _pick_config(Bs, H, P, N, L, heavy=True, esz=4):
    block_p = _divisor_le(P, TILE_P)
    block_l = min(triton.next_power_of_2(L), MAX_BLOCK_L)
    tpw_l = 4
    ctas = Bs * H * (P // 2)
    if P % 2 == 0 and block_l == MAX_BLOCK_L and (ctas >= WAVE_CTAS) and (esz > 2 or (ctas < 2 * WAVE_CTAS and L <= BP2_L_MAX)):
        block_p, block_l, tpw_l = (2, MAX_BLOCK_L // 2, 32)
    num_warps = max(1, block_p * block_l // (32 * ELEMS_PER_THREAD))
    return dict(BLOCK_L=block_l, BLOCK_P=block_p, NUM_WARPS=num_warps, TPW_L=tpw_l, PF=0)

def _tpw(cfg):
    c = dict(cfg)
    c['TPW_L'] = 32 if c.get('BLOCK_P', 1) > 1 else 4
    return c

def _hnp(t, Bs, H, P, N):
    out = torch.empty(Bs, H, N, P, device=t.device, dtype=torch.float32)
    return out.copy_(t.reshape(Bs, H, P, N).permute(0, 1, 3, 2))

def mh_forward(x, delta, A_log, B_t, C_t, *, delta_w=None, delta_softplus=True, endpoint_own=None, row_law=None, row_rot=None, a_dyn=None, mu=None, mu_gain=None, a_dyn_res='n', mu_res='n', dyn_act=DYN_SOFTPLUS, dyn_eps=0.0, g_B=None, g_C=None, gate_act=GATE_SIGMOID, theta=None, theta_delta=False, theta_h=None, beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None, delta_bias=None, h_init=None, theta_init=None, mu_init=None, mu_prediff=False, d_skip=None, z=None, n_off=0, n_rows=None, config=None, ckpt_l=0, ckpt_out=None, ckpt_dt=None, bc_head=False, pmap=False):
    _argv = dict(locals())
    if mu_prediff and mu_init is not None:
        raise ValueError('mu_prediff means `mu` already holds the telescoped\nincrement, so mu_init is folded into column 0 and must\nbe None -- applying it again shifts row 0 of every b')
    if pmap == 'raw':
        pmap = False
    if bc_head:
        expected = (x.shape[1], x.shape[0], A_log.shape[-1], x.shape[-1])
        if B_t.shape != expected or C_t.shape != expected:
            raise ValueError('prepared B/C must be (H,B,N,L)')
        if any((t is not None for t in (theta, g_B, g_C, beta_b, beta_c, beta_gB, beta_gC))):
            raise ValueError('prepared maps must have all map folds applied')
    if pmap:
        if theta is None or bc_head or any((t is not None for t in (g_B, g_C, beta_b, beta_c, beta_gB, beta_gC))):
            raise ValueError('pmap: pair maps belong to a rotating launch with every B/C fold (bias, gate) applied')
        if B_t.shape[1] != 2 * A_log.shape[-1] or C_t.shape != B_t.shape:
            raise ValueError('pmap: B/C pair maps must be (B, 2N, L)')
    sh = validate_forward_inputs(x, delta, A_log, B_t[0] if bc_head else B_t[:, ::2] if pmap else B_t, C_t[0] if bc_head else C_t[:, ::2] if pmap else C_t, delta_w=delta_w, a_dyn=a_dyn, mu=mu, mu_gain=mu_gain, g_B=g_B, g_C=g_C, theta=theta, beta_a=beta_a, beta_b=beta_b, beta_c=beta_c, beta_gB=beta_gB, beta_gC=beta_gC, delta_bias=delta_bias, h_init=h_init, theta_init=theta_init, mu_init=mu_init, n_off=n_off, n_rows=n_rows)
    Bs, H, P, N, L = (sh.B, sh.H, sh.P, sh.N, sh.L)
    nr = N - n_off if n_rows is None else n_rows
    dev = x.device
    all_a_fix = row_law is None
    if all_a_fix:
        row_law = torch.zeros(N, dtype=torch.int32, device=dev)
    rot_w = selector_window(row_rot, n_off, nr)
    law_present = {LAW_A_FIX} if all_a_fix else set(selector_window(row_law, n_off, nr))
    row_law = row_law.to(torch.int32).contiguous()
    ROT = theta is not None and (rot_w is None or any(rot_w))
    ROT = ROT or bool(pmap)
    if ROT and (n_off % 2 or nr % 2):
        raise ValueError(f'ROT needs a pair-aligned n window, got n_off={n_off} n_rows={nr}')
    LAW_MASK = sum((1 << law for law in (LAW_A_FIX, LAW_A_DYN, LAW_MU) if law in law_present)) or 1
    h_np = torch.zeros(Bs, H, N, P, device=dev, dtype=torch.float32) if h_init is None else _hnp(_f32(h_init), Bs, H, P, N)
    Y = torch.empty(Bs, H, P, L, device=dev, dtype=x.dtype)

    def _t3(t):
        return None if t is None else _f32(t).reshape(H, P, N).permute(0, 2, 1).contiguous()
    _infer = not ckpt_l and ckpt_out is None and (not ROT)
    CKPACK = bool(ckpt_l) and P == 1
    ckpt_compact = bool(ckpt_l) and os.environ.get('MH_CKPT_STORE', '1') != '0'
    ckpt_zero = os.environ.get('MH_CKPT_ZERO', '0') != '0' or n_off != 0 or nr != N
    heavy = ROT or g_B is not None or g_C is not None
    _pin = autotune.pinned_config('fwd') if config is None else None
    _cal = None
    if config is None and (not _pin) and (ckpt_out is None) and (P == 1):
        from . import calibration_mh as _cm
        if _cm.p1_envelope(P=P, ROT=ROT, pmap=pmap, bc_head=bc_head, LAW_MASK=LAW_MASK, endpoint_own=endpoint_own, n_off=n_off, nr=nr, N=N, absent=(delta_w, g_B, g_C, beta_a, beta_b, beta_c, beta_gB, beta_gC, a_dyn, mu, theta, mu_init, theta_init)):
            _cal = _cm.p1_config('fwd_trn' if ckpt_l else 'fwd', Bs, H, N, L, x.dtype, _cm.p1_feat(delta_softplus=delta_softplus, delta_bias=delta_bias, d_skip=d_skip, z=z, h_init=h_init))
    _fp = (_pin or _cal or config or {}).get('FPACK')
    fpack_ok = P == 1 and H > 1 and (not ROT) and (not pmap) and (not bc_head) and (LAW_MASK == 1) and (endpoint_own is None) and all((t is None for t in (delta_w, g_B, g_C, beta_a, beta_b, beta_c, beta_gB, beta_gC, a_dyn, mu))) and (bool(_fp) if _fp is not None else os.environ.get('MH_FWD_FPACK', '0') == '1')
    PW = H if fpack_ok else P
    from .ts_forward import ts_fwd_refusal
    ts_ok = L >= int(os.environ.get('MH_FWD_TS_MIN_L', 4096)) and ts_fwd_refusal(ROT=ROT, pmap=pmap, LAW_MASK=LAW_MASK, delta_w=delta_w, beta_a=beta_a, beta_b=beta_b, beta_c=beta_c, g_B=g_B, g_C=g_C, beta_gB=beta_gB, beta_gC=beta_gC, endpoint_own=endpoint_own, bc_head=bc_head, n_off=n_off, nr=nr, N=N, fpack=fpack_ok) is None
    cfg = _pick_config(Bs, H, P, N, L, heavy, x.element_size()) if config is None else dict(config)
    _pin = _pin or _cal
    if config is None:
        cfg.update(_pin or autotune.table_fwd(_sm(), ROT, bool(pmap), x.element_size(), P, N) or {})
        if _pin or cfg.get('BLOCK_P', 1) > 1:
            cfg = _tpw(cfg)
    cfg['BLOCK_P'] = _divisor_le(PW, cfg['BLOCK_P'])
    cfg.setdefault('PF', 0)
    cfg.setdefault('TS', int(os.environ.get('MH_FWD_TS', 1)))
    _tier = autotune.tier()
    if config is None and ckpt_out is None and (not _pin) and autotune.enabled():
        key = ('fwd_t', _tier, slp.active(_sm()), bool(theta_delta) and theta is not None, theta_h is not None, bool(ROT and rot_w is not None and (not all(rot_w))), os.environ.get('MH_CKPT_BF16', '1') != '0', bool(bc_head), bool(pmap), B_t.dtype, Bs, H, P, N, L, n_off, nr, x.dtype, LAW_MASK, ROT, theta is not None, None if theta is None else theta.dim(), bool(delta_softplus), int(dyn_act), int(gate_act), float(dyn_eps), a_dyn_res, mu_res, delta_w is not None, None if delta_w is None else delta_w.dim(), g_B is not None, g_C is not None, h_init is not None, beta_a is not None, beta_b is not None, beta_c is not None, beta_gB is not None, beta_gC is not None, delta_bias is not None, mu_init is not None, bool(mu_prediff), int(ckpt_l), ckpt_dt, CKPACK, ckpt_compact, ckpt_zero, 'ckv' if _infer else None, fpack_ok, ts_ok, d_skip is not None, z is not None, endpoint_own is not None, None if endpoint_own is None else endpoint_own.ndim, *(None if t is None else t.shape[0] for t in (beta_a, beta_b, beta_c, beta_gB, beta_gC)))
        _hit = autotune.cached(key)
        if _hit is not None:
            cfg = _tpw(_hit)
        else:
            kwt = {k: v for k, v in _argv.items() if k not in ('x', 'delta', 'A_log', 'B_t', 'C_t', 'config')}

            def _run(c):
                return mh_forward(x, delta, A_log, B_t, C_t, **kwt, config=_tpw(c))
            cfg = _tpw(autotune.choose_axes(key, cfg, autotune.t_axes(cfg, PW, L, int(ckpt_l), ROT, _tier, ts_axis=ts_ok), _run, persist=_mh_fwd_kernel.cache_key, grid=[dict(g, CKV=v) for g in autotune.t_grid(cfg, PW, L, int(ckpt_l)) for v in ((0, 1) if _infer else (0,))], candidate_filter=autotune.light_fwd_candidate if _tier == 'light' else None, pre=('TS',)))
        cfg['BLOCK_P'] = _divisor_le(PW, cfg['BLOCK_P'])
    FPACK = fpack_ok and cfg['BLOCK_P'] > 1
    TS = max(1, min(int(cfg.get('TS', 1)), triton.cdiv(L, cfg['BLOCK_L']))) if ts_ok else 1
    grid = (1, H // cfg['BLOCK_P'], Bs) if FPACK else (P // cfg['BLOCK_P'], H, Bs * TS)
    K = N // 2
    gp = grid[0]
    th_np = torch.zeros(Bs, H, gp, K, device=dev, dtype=torch.float32)
    if theta_init is not None:
        th_np.copy_(theta_init.reshape(Bs, H, 1, K))
    _ckv = False
    if _infer and int(cfg.get('CKV', 0)):
        from .bwd_gluon import auto_block_l
        _s = auto_block_l(P, L)
        if _s and cfg['BLOCK_L'] % _s == 0:
            ckpt_l, _ckv = (_s, True)
            CKPACK = P == 1
            ckpt_compact = os.environ.get('MH_CKPT_STORE', '1') != '0'
    cks = 1
    if ckpt_l:
        if cfg['BLOCK_L'] % ckpt_l:
            raise ValueError(f"ckpt_l {ckpt_l} must divide BLOCK_L {cfg['BLOCK_L']}")
        cks = triton.cdiv(L, cfg['BLOCK_L']) * (cfg['BLOCK_L'] // ckpt_l) + 1
        if ckpt_out is None and CKPACK:
            if ckpt_dt is None:
                ckpt_dt = x.dtype if x.element_size() == 2 and os.environ.get('MH_CKPT_BF16', '1') != '0' else torch.float32
            ckpt_out = (torch.zeros if ckpt_zero else torch.empty)(Bs, cks, N, H * P, device=dev, dtype=ckpt_dt)
        elif ckpt_out is None:
            if ckpt_dt is None:
                ckpt_dt = x.dtype if x.element_size() == 2 and os.environ.get('MH_CKPT_BF16', '1') != '0' else torch.float32
            ckpt_out = (torch.zeros if ckpt_zero else torch.empty)(Bs, H, cks, N, P, device=dev, dtype=ckpt_dt)
        if CKPACK:
            ckpt_out[:, 0].copy_(h_np.permute(0, 2, 1, 3).reshape(Bs, N, H * P))
        else:
            ckpt_out[:, :, 0].copy_(h_np)
    h_ts = h_np
    if TS > 1:
        from .ts_forward import ts_fwd_seeds
        nbs = triton.cdiv(triton.cdiv(L, cfg['BLOCK_L']), TS)
        h_ts = ts_fwd_seeds(TS, nbs * cfg['BLOCK_L'], x, delta, A_log.reshape(H, P, N), B_t, torch.empty(TS, Bs, H, N, P, device=dev, dtype=torch.float32), delta_bias=delta_bias, delta_softplus=delta_softplus, h0=h_np if h_init is not None else None)
    global LAST_COMPILED, LAST_CONFIG
    LAST_CONFIG = dict(cfg, TS=TS, FPACK=bool(FPACK), grid=tuple(grid))
    LAST_COMPILED = _mh_fwd_kernel[grid](x.contiguous(), delta.contiguous(), None if delta_w is None else delta_w.contiguous(), _t3(A_log), _t3(mu_gain), B_t.contiguous(), C_t.contiguous(), None if g_B is None else g_B.contiguous(), None if g_C is None else g_C.contiguous(), None if a_dyn is None else a_dyn.contiguous(), None if mu is None else mu.contiguous(), _f32(mu_init), _f32(beta_a), _f32(beta_b), _f32(beta_c), _f32(beta_gB), _f32(beta_gC), _f32(delta_bias), row_law, None if row_rot is None else row_rot.to(torch.int32).contiguous(), None if theta is None else theta.contiguous(), _thh_table(theta_h), Y, h_ts, th_np, ckpt_out, None if d_skip is None else d_skip.float().contiguous(), None if z is None else z.contiguous(), endpoint_own, H, P, N, L, K, n_off, nr, cks, LAW_MASK=LAW_MASK, HAS_ENDPOINT=endpoint_own is not None, ENDPOINT_P=endpoint_own is not None and endpoint_own.ndim == 4, ROT=ROT, THETA_DELTA=bool(theta_delta) and theta is not None, THETA_PH=theta is not None and theta.dim() == 4, HAS_THH=theta_h is not None, HAS_RROT=ROT and rot_w is not None and (not all(rot_w)), PMAP=bool(pmap), SLP=slp.active(_sm()), DELTA_SOFTPLUS=bool(delta_softplus), DYN_ACT=int(dyn_act), GATE_ACT=int(gate_act), A_DYN_N_RES=a_dyn_res == 'n', MU_N_RES=mu_res == 'n', HAS_DW=delta_w is not None, DW_PER_P=delta_w is not None and delta_w.dim() == 4, HAS_GB=g_B is not None, HAS_GC=g_C is not None, HAS_BA=beta_a is not None, HAS_BB=beta_b is not None, HAS_BC=beta_c is not None, HAS_BGB=beta_gB is not None, HAS_BGC=beta_gC is not None, HAS_DBIAS=delta_bias is not None, HAS_MU_INIT=mu_init is not None, MU_PREDIFF=bool(mu_prediff), HG_A=H // beta_a.shape[0] if beta_a is not None else 1, HG_BC=H // beta_b.shape[0] if beta_b is not None else H // beta_c.shape[0] if beta_c is not None else 1, HG_G=H // beta_gB.shape[0] if beta_gB is not None else H // beta_gC.shape[0] if beta_gC is not None else 1, DYN_EPS=float(dyn_eps), BLOCK_L=cfg['BLOCK_L'], BLOCK_P=cfg['BLOCK_P'], NUM_WARPS=cfg['NUM_WARPS'], TPW_L=cfg.get('TPW_L', 4), HAS_DSKIP=d_skip is not None, HAS_Z=z is not None, ABL=int(os.environ.get('MH_FWD_ABL', cfg.get('ABL', 0))), CKPT_L=int(ckpt_l), CKPACK=CKPACK, CKPT_COMPACT=ckpt_compact, BC_HEAD=bool(bc_head), PF=int(cfg.get('PF', 0)), FAST_TRIG=bool(cfg.get('FAST_TRIG', True)), EVEN_L=L % cfg['BLOCK_L'] == 0 and bool(cfg.get('EVEN_L', True)), FPACK=FPACK, TS=TS, num_warps=cfg['NUM_WARPS'])
    if TS > 1:
        h_np = h_ts[TS - 1]
    return (Y, h_np.permute(0, 1, 3, 2), ckpt_out if ckpt_l and (not _ckv) else None, th_np[:, :, 0, :] if ROT else None)
