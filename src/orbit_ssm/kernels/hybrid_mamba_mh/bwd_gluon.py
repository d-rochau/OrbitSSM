
from __future__ import annotations
import os
from triton.language import range as loop_range
import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.extra import libdevice
from . import autotune, slp
from .common_gluon import INV_TWO_PI, K_DYN, K_MU, LOG2E, TWO_PI, _cmul, _dact, _f32, _gate, _gate_dv, _i32, _rate, _row1, _silu_fast_d, _softplus, _ssm_combine, _ssm_combine_q, selector_window
from .fwd_gluon import _ldt, _thh_table, _zt
from .layout import DYN_RAW, GATE_SIGMOID, LAW_A_DYN, LAW_A_FIX, LAW_MU, group_index
LAST_COMPILED = None
LAST_CONFIG = None
_EXTREME_A_LOG = 40.0

def _envi(name, dflt):
    v = os.environ.get(name)
    return dflt if v is None else int(v)

def _maxnreg():
    v = os.environ.get('MH_BWD_MAXNREG')
    if v is None:
        return (None, False)
    n = int(v)
    return (n if n > 0 else None, True)
MAX_BLOCK_P = 32
SMALL_P_BLOCK_L = 128

def auto_np(P):
    np_ = 1
    while P // np_ > MAX_BLOCK_P and P // np_ % 2 == 0:
        np_ *= 2
    return np_
HP_DEFAULT = 8
HPACK_BUDGET = 512

def _hp_default(H):
    hp = HP_DEFAULT
    while hp > 1 and H % hp:
        hp //= 2
    return max(1, min(hp, H))

def auto_block_l(P, L, block_l=None, budget=1024):
    if block_l:
        bl = block_l
    elif P == 1 and budget == 1024:
        bl = max(min(HPACK_BUDGET // HP_DEFAULT, SMALL_P_BLOCK_L), 8)
    else:
        bl = max(min(budget // (P // auto_np(P)), SMALL_P_BLOCK_L), 8)
    bl = 1 << bl.bit_length() - 1
    return min(bl, triton.next_power_of_2(L))

def _hsum(buf):
    return buf[0] if buf.shape[0] == 1 else buf.sum(0)

@gluon.jit
def _add(a, b):
    return a + b

@gluon.jit
def _tail_sum(v2, rev2, prev2, first2, last2):
    rev_incl = gl.associative_scan(gl.gather(v2, rev2, 1), axis=1, combine_fn=_add)
    total = gl.sum(gl.where(last2, rev_incl, 0.0), axis=1)
    tail = gl.gather(gl.where(first2, 0.0, gl.gather(rev_incl, prev2, 1)), rev2, 1)
    return (total, tail)

@gluon.constexpr_function
def _red_asm(vec, tpl, wown=0):
    wchk = '' if wown <= 0 else f'.reg .u32 wid;\n.reg .pred q;\nmov.u32 wid, %tid.x;\nshr.u32 wid, wid, 5;\nsetp.lt.u32 q, wid, {wown};\nand.pred p, p, q;\n'
    vals = ', '.join(('$%d' % (2 * vec + i) for i in range(vec)))
    zero = '\n'.join(('mov.u32 $%d, 0;' % i for i in range(vec)))
    return f'{{\n.reg .pred p;\n.reg .u32 lid;\nmov.u32 lid, %laneid;\nsetp.lt.u32 p, lid, {tpl};\nsetp.ne.and.b32 p, ${3 * vec}, 0, p;\n{wchk}@p red.global.add.v{vec}.f32 [${vec}], {{{vals}}};\n{zero}\n}}'

@gluon.constexpr_function
def _red_cons(vec):
    return ','.join(['=r'] * vec + ['l'] * vec + ['f'] * vec + ['r'] * vec)

@gluon.jit
def _red_add(ptr, val, mask, VEC: gl.constexpr, TPL: gl.constexpr, WOWN: gl.constexpr=0):
    gl.inline_asm_elementwise(_red_asm(VEC, TPL, WOWN), _red_cons(VEC), [ptr.to(gl.int64), val, mask.to(gl.int32)], dtype=gl.int32, is_pure=False, pack=VEC)

@gluon.jit
def _put(ptr, val, mask, ATOMIC: gl.constexpr, EVEN: gl.constexpr=False, VEC: gl.constexpr=0, TPL: gl.constexpr=32, WOWN: gl.constexpr=0):
    if ATOMIC:
        if VEC > 0:
            _red_add(ptr, val, mask, VEC, TPL, WOWN)
        elif EVEN:
            gl.atomic_add(ptr, val, sem='relaxed')
        else:
            gl.atomic_add(ptr, val, mask=mask, sem='relaxed')
    elif EVEN:
        gl.store(ptr, val)
    else:
        gl.store(ptr, val, mask=mask)

@gluon.jit
def _bacc(ptr, g, n, N, val, mask, ON: gl.constexpr, EVEN: gl.constexpr=False):
    if ON:
        gl.atomic_add(ptr + g * N + n, gl.sum(_zt(val, mask, EVEN)), sem='relaxed')

@gluon.jit
def _gate_bacc(ptr, g, n, N, val, mask, ON: gl.constexpr, EVEN: gl.constexpr, PARTIAL: gl.constexpr, slot):
    if ON:
        if PARTIAL:
            gl.store(ptr + slot * N + n, gl.sum(_zt(val, mask, EVEN)))
        else:
            _bacc(ptr, g, n, N, val, mask, ON, EVEN)

@gluon.jit
def _st(ptr, val, mask, EVEN: gl.constexpr):
    if EVEN:
        gl.store(ptr, val)
    else:
        gl.store(ptr, val, mask=mask)

@gluon.constexpr_function
def _lay_pv(BLOCK_L, NUM_WARPS):
    return gl.BlockedLayout([max(BLOCK_L // (32 * NUM_WARPS), 1)], [32], [NUM_WARPS], [0])

@gluon.constexpr_function
def _lay3(BLOCK_P, BLOCK_L, NUM_WARPS, TP_P, WP=1):
    tp_p = min(TP_P, BLOCK_P)
    tp_l = max(32 // tp_p, 1)
    spt_p = max(BLOCK_P // (tp_p * WP), 1)
    spt_l = max(BLOCK_L // (tp_l * (NUM_WARPS // WP)), 1)
    return gl.BlockedLayout([spt_p, spt_l], [tp_p, tp_l], [WP, NUM_WARPS // WP], [1, 0])

def _vec_tp_p(itemsize, BLOCK_L, NUM_WARPS, BLOCK_P, L):
    if itemsize >= 4:
        return None
    vec = 16 // itemsize
    if L % vec or BLOCK_L % vec:
        return None
    tp_p = vec * 32 * NUM_WARPS // BLOCK_L
    if tp_p < 1 or tp_p > min(32, BLOCK_P) or tp_p & tp_p - 1:
        return None
    return tp_p

@gluon.jit
def _de_load(delta_ptr, delta_bias_ptr, bh, h, L, offs_t, mask_t, zl, DELTA_SOFTPLUS: gl.constexpr, HAS_DBIAS: gl.constexpr, EVEN_L: gl.constexpr=False, DSIG: gl.constexpr=False):
    d = _ldt(delta_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32)
    if HAS_DBIAS:
        d = d + gl.load(delta_bias_ptr + h + zl)
    ds = d
    if DELTA_SOFTPLUS:
        if DSIG:
            ds = 1.0 / (1.0 + gl.exp(-d))
        d = _softplus(d)
    return (_zt(d, mask_t, EVEN_L), ds)

@gluon.jit
def _de_load_t(delta_ptr, delta_bias_ptr, h, pofs, po, m2, DELTA_SOFTPLUS: gl.constexpr, HAS_DBIAS: gl.constexpr, EVEN_L: gl.constexpr=False, DSIG: gl.constexpr=False):
    d = _ldt(delta_ptr + po, m2, EVEN_L).to(gl.float32)
    if HAS_DBIAS:
        d = d + gl.expand_dims(gl.load(delta_bias_ptr + h + pofs), 1)
    ds = d
    if DELTA_SOFTPLUS:
        if DSIG:
            ds = 1.0 / (1.0 + gl.exp(-d))
        d = _softplus(d)
    return (_zt(d, m2, EVEN_L), ds)

@gluon.jit
def _dw_load(dw_ptr, de2, bh, L, po, offs_t, mask_t, m2, z2, DELTA_SOFTPLUS: gl.constexpr, HAS_DW: gl.constexpr, DW_P: gl.constexpr, EVEN_L: gl.constexpr=False, HPACK: gl.constexpr=False):
    d = de2
    if HAS_DW:
        if DW_P or HPACK:
            d = _ldt(dw_ptr + po, m2, EVEN_L).to(gl.float32)
        else:
            d = gl.expand_dims(_ldt(dw_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32), 0) + z2
        if DELTA_SOFTPLUS:
            d = _softplus(d)
        d = _zt(d, m2, EVEN_L)
    return d

@gluon.jit
def _bc_load(ptr, bias_ptr, off, n, gbc, N, mask_t, zl, HAS_BIAS: gl.constexpr, EVEN_L: gl.constexpr=False):
    v = _ldt(ptr + off, mask_t, EVEN_L).to(gl.float32)
    if HAS_BIAS:
        v = v + gl.load(bias_ptr + gbc * N + n + zl)
    return v

@gluon.jit
def _gate_load(g_ptr, bias_ptr, off, n, gg, N, mask_t, zl, GATE_ACT: gl.constexpr, HAS_BIAS: gl.constexpr, EVEN_L: gl.constexpr=False):
    v = _ldt(g_ptr + off, mask_t, EVEN_L).to(gl.float32)
    if HAS_BIAS:
        v = v + gl.load(bias_ptr + gg * N + n + zl)
    return _gate(v, GATE_ACT)

@gluon.jit
def _gate_load_d(g_ptr, bias_ptr, off, n, gg, N, mask_t, zl, GATE_ACT: gl.constexpr, HAS_BIAS: gl.constexpr, EVEN_L: gl.constexpr=False):
    v = _ldt(g_ptr + off, mask_t, EVEN_L).to(gl.float32)
    if HAS_BIAS:
        v = v + gl.load(bias_ptr + gg * N + n + zl)
    return _gate_dv(v, GATE_ACT)

@gluon.jit
def _rot_cs(theta_ptr, thck_ptr, thb, h, bh, kk, K, L, NBLK, j, offs_t, col, m2, z2, zl, de, INV_BP: gl.constexpr, ABL: gl.constexpr, FAST_TRIG: gl.constexpr, EVEN_L: gl.constexpr=False, THETA_DELTA: gl.constexpr=False, ROW1: gl.constexpr=None):
    Th = gl.load(thck_ptr + (bh * NBLK + j) * K + kk + zl)
    if not ABL & 32:
        thv = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_t, offs_t < L, EVEN_L).to(gl.float32)
        if THETA_DELTA:
            thv = thv * de
        inc1 = gl.expand_dims(gl.convert_layout(_zt(thv, offs_t < L, EVEN_L), gl.SliceLayout(0, ROW1)), 0)
        Th += gl.convert_layout(gl.sum(gl.associative_scan(inc1, axis=1, combine_fn=_add), axis=0), zl.type.layout)
    if ABL & 256:
        return (Th, Th)
    if FAST_TRIG:
        return (libdevice.fast_cosf(Th), libdevice.fast_sinf(Th))
    return (gl.cos(Th), gl.sin(Th))

@gluon.jit
def _rot_sel(row_rot_ptr, kk, NPAIR: gl.constexpr):
    return gl.load(row_rot_ptr + kk * NPAIR).to(gl.float32)

@gluon.jit
def _law_s1(a_dyn_ptr, law_s, de, b, bh, n, N, L, offs_t, mask_t, zl, HAS_DYN: gl.constexpr, UNI: gl.constexpr, DYN_ACT: gl.constexpr, DYN_EPS: gl.constexpr, A_DYN_N_RES: gl.constexpr, EVEN_L: gl.constexpr=False):
    s1 = de
    fac = zl.to(gl.float32) + 1.0
    if HAS_DYN and (UNI or law_s == K_DYN):
        if A_DYN_N_RES:
            ad = _ldt(a_dyn_ptr + (b * N + n.to(gl.int64)) * L + offs_t, mask_t, EVEN_L).to(gl.float32)
        else:
            ad = _ldt(a_dyn_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32)
        fac = _rate(ad, DYN_ACT, DYN_EPS)
        s1 = de * fac
    return (_zt(s1, mask_t, EVEN_L), fac)

@gluon.jit
def _law_s2(mu_ptr, mu_init_ptr, b, bh, n, N, L, offs_t, mask_t, col, zl, MU_N_RES: gl.constexpr, HAS_MU_INIT: gl.constexpr, MU_PREDIFF: gl.constexpr, EVEN_L: gl.constexpr=False):
    if MU_N_RES:
        mo = (b * N + n.to(gl.int64)) * L + offs_t
    else:
        mo = bh * L + offs_t
    mu_t = _ldt(mu_ptr + mo, mask_t, EVEN_L).to(gl.float32)
    if MU_PREDIFF:
        s2 = mu_t
    else:
        mu_p = gl.gather(mu_t, gl.maximum(col - 1, 0), 0)
        at0 = (col == 0) & mask_t
        mu_p = gl.where(at0, gl.load(mu_ptr + mo - 1, mask=at0 & (offs_t > 0), other=0.0).to(gl.float32), mu_p)
        if HAS_MU_INIT:
            if MU_N_RES:
                mi = gl.load(mu_init_ptr + b * N + n + zl)
            else:
                mi = gl.load(mu_init_ptr + bh + zl)
            mu_p = gl.where(offs_t == 0, mi, mu_p)
        s2 = mu_t - mu_p
    return _zt(s2, mask_t, EVEN_L)

@gluon.jit
def _mh_bwd_kernel(x_ptr, delta_ptr, A_log_ptr, B_ptr, C_ptr, gy_ptr, theta_ptr, thh_ptr, thi_ptr, row_law_ptr, row_rot_ptr, a_dyn_ptr, mu_ptr, mu_gain_ptr, mu_init_ptr, delta_bias_ptr, beta_b_ptr, beta_c_ptr, dw_ptr, gB_ptr, gC_ptr, bgB_ptr, bgC_ptr, gx_ptr, gde_ptr, gc1_ptr, ghi_ptr, gBt_ptr, gCt_ptr, gth_ptr, gthh_ptr, gc2_ptr, grate_ptr, gs2o_ptr, gdw_ptr, ggB_ptr, ggC_ptr, hck_ptr, alam_ptr, thck_ptr, rcar_ptr, d_skip_ptr, gd_skip_ptr, z_ptr, gz_ptr, endpoint_ptr, gendpoint_ptr, gdbias_ptr, gbb_ptr, gbc_ptr, gbgB_ptr, gbgC_ptr, Bs, H, P, N, L, K, NBLK, CKS, BLOCK_L: gl.constexpr, BLOCK_P: gl.constexpr, NUM_WARPS: gl.constexpr, HAS_ENDPOINT: gl.constexpr=False, ENDPOINT_P: gl.constexpr=False, TP_P: gl.constexpr=8, ABL: gl.constexpr=0, WP: gl.constexpr=1, RUN_P1: gl.constexpr=True, NS: gl.constexpr=1, NP: gl.constexpr=1, ATOMIC_GX: gl.constexpr=False, ATOMIC_BC: gl.constexpr=False, BCS: gl.constexpr=1, LWS: gl.constexpr=0, LAW_VEC: gl.constexpr=0, BC_VEC: gl.constexpr=0, GX_VEC: gl.constexpr=0, TPL: gl.constexpr=32, ROT: gl.constexpr=False, RUN_TH: gl.constexpr=True, PAIR_UNROLL: gl.constexpr=False, PMAP: gl.constexpr=False, PRAW: gl.constexpr=False, SLP: gl.constexpr=False, VEC_V: gl.constexpr=0, TPL_V: gl.constexpr=32, FAST_TRIG: gl.constexpr=False, THETA_DELTA: gl.constexpr=False, HAS_THH: gl.constexpr=False, THETA_PH: gl.constexpr=False, LAW_MASK: gl.constexpr=1, DYN_ACT: gl.constexpr=0, DYN_EPS: gl.constexpr=0.0, A_DYN_N_RES: gl.constexpr=True, MU_N_RES: gl.constexpr=True, HAS_MU_INIT: gl.constexpr=False, MU_PREDIFF: gl.constexpr=False, DELTA_SOFTPLUS: gl.constexpr=True, HAS_DBIAS: gl.constexpr=False, DBACC: gl.constexpr=False, HAS_Z: gl.constexpr=False, HAS_BB: gl.constexpr=False, HAS_BC: gl.constexpr=False, HG_BC: gl.constexpr=1, HAS_GHL: gl.constexpr=False, BACC_B: gl.constexpr=False, BACC_C: gl.constexpr=False, BACC_GB: gl.constexpr=False, BGPART: gl.constexpr=False, BACC_GC: gl.constexpr=False, HAS_DW: gl.constexpr=False, DW_P: gl.constexpr=False, N_OFF: gl.constexpr=0, N_ROWS: gl.constexpr=0, ROT_MIX: gl.constexpr=False, HAS_GB: gl.constexpr=False, HAS_GC: gl.constexpr=False, GATE_ACT: gl.constexpr=1, HAS_BGB: gl.constexpr=False, HAS_BGC: gl.constexpr=False, HG_G: gl.constexpr=1, EVEN_L: gl.constexpr=False, HAS_DSKIP: gl.constexpr=False, HPACK: gl.constexpr=False, HP: gl.constexpr=1, BC_HEAD: gl.constexpr=False, DSIG: gl.constexpr=False, CKPACK: gl.constexpr=False, ATOMIC_TH: gl.constexpr=False, THS: gl.constexpr=1, TH_VEC: gl.constexpr=0, TS: gl.constexpr=1):
    tw: gl.constexpr = _lay3(BLOCK_P, BLOCK_L, NUM_WARPS, TP_P, WP)
    WOWN: gl.constexpr = NUM_WARPS // WP if WP > 1 else 0
    lt: gl.constexpr = gl.SliceLayout(0, tw)
    pl: gl.constexpr = gl.SliceLayout(1, tw)
    row1: gl.constexpr = _row1(tw)
    pv: gl.constexpr = _lay_pv(BLOCK_L, NUM_WARPS)
    HAS_DYN: gl.constexpr = LAW_MASK >> 1 & 1
    HAS_MU: gl.constexpr = LAW_MASK >> 2 & 1
    PLAIN: gl.constexpr = LAW_MASK == 1
    UNI: gl.constexpr = LAW_MASK & LAW_MASK - 1 == 0
    NPAIR: gl.constexpr = 2 if ROT or ABL & 512 else 1
    INV_BP: gl.constexpr = 1.0 / BLOCK_P
    h = gl.program_id(0) // NP * HP
    pid_p = gl.program_id(0) % NP
    b = gl.program_id(1).to(gl.int64)
    bh = b * H + h
    thb = (bh if THETA_PH else b) * K
    if BC_HEAD:
        B_ptr += h.to(gl.int64) * Bs * N * L
        C_ptr += h.to(gl.int64) * Bs * N * L
    gbc = h // HG_BC
    gg = h // HG_G
    sh = gl.program_id(2) // TS
    ts = gl.program_id(2) % TS
    n_lo = N_OFF + sh * (N_ROWS // NS)
    k_lo = n_lo // NPAIR
    k_hi = (n_lo + N_ROWS // NS) // NPAIR
    gxb = sh.to(gl.int64) * Bs * H * P * L
    gdb = (sh.to(gl.int64) * NP + pid_p) * Bs * H * L
    gdbh = ((sh.to(gl.int64) * TS + ts) * NP + pid_p) * Bs * H
    gxo = gxb * (0 if ATOMIC_GX else 1)
    col = gl.arange(0, BLOCK_L, layout=lt)
    idx_rev = BLOCK_L - 1 - col
    idx_nxt = gl.minimum(BLOCK_L - col, BLOCK_L - 1)
    zl = gl.zeros([BLOCK_L], gl.int32, layout=lt)
    z2 = gl.zeros([BLOCK_P, BLOCK_L], gl.float32, layout=tw)
    rev2 = gl.expand_dims(idx_rev, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int32, layout=tw)
    if HPACK:
        nxt2 = gl.expand_dims(idx_nxt, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int32, layout=tw)
    idx_prv = gl.maximum(col - 1, 0)
    prev2 = gl.expand_dims(idx_prv, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int32, layout=tw)
    first2 = gl.expand_dims(col == 0, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tw)
    last2 = gl.expand_dims(col == BLOCK_L - 1, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tw)
    pofs = gl.arange(0, BLOCK_P, layout=pl)
    pf = (pid_p * BLOCK_P + pofs).to(gl.int64)
    zp = gl.zeros([BLOCK_P], gl.int32, layout=pl)
    hp = bh * P + pf
    sn = H * P if HPACK else P
    snc = H * P if CKPACK else P
    if HPACK:
        anp = h.to(gl.int64) + pf
        cyp = b * N * sn + h.to(gl.int64) + pf
        cyp += ts.to(gl.int64) * Bs * N * sn
    else:
        anp = h.to(gl.int64) * N * P + pf
        cyp = bh * N * P + pf
        cyp += ts.to(gl.int64) * Bs * H * N * P
    if CKPACK:
        ckp = b * CKS * N * snc + h.to(gl.int64) + pf
    else:
        ckp = bh * CKS * N * P + pf
    rcp = (bh + pf) * K * P if HPACK else bh * K * P + pf
    hslot = gl.program_id(0).to(gl.int64)
    hbn = (hslot * Bs + b) * N * L
    lawbn = (hslot % LWS * Bs + b) * N * L if LWS else hbn
    NMR = N * 2 if PMAP and (not PRAW) else N
    if ATOMIC_BC:
        bcn = ((h.to(gl.int64) if BC_HEAD else hslot % BCS) * Bs + b) * NMR * L
    elif PMAP:
        bcn = (hslot * Bs + b) * NMR * L
    else:
        bcn = hbn
    if ATOMIC_TH:
        thn = (hslot % THS * Bs + b) * K
    else:
        thn = (hslot * Bs + b) * K
    pair_iterator: gl.constexpr = gl.static_range if not ROT or HPACK or BLOCK_P < 4 or PAIR_UNROLL else loop_range
    if ROT and RUN_TH:
        for kk in range(k_lo, k_hi):
            thc = gl.load(thi_ptr + (bh + pf if HPACK else bh + zp) * K + kk)
            for j in range(NBLK):
                offs_t = j * BLOCK_L + col
                m2 = gl.expand_dims(offs_t < L, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tw)
                gl.store(thck_ptr + ((bh + pf if HPACK else bh + zp) * NBLK + j) * K + kk, thc, mask=gl.full((BLOCK_P,), True, gl.int1, layout=pl) if HPACK else pofs == 0)
                feat = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_t, offs_t < L, EVEN_L).to(gl.float32)
                inc = gl.expand_dims(feat, 0) + z2
                if THETA_DELTA:
                    if HPACK:
                        ptime = gl.expand_dims(hp, 1) * L + gl.expand_dims(offs_t, 0)
                        de0, _ = _de_load_t(delta_ptr, delta_bias_ptr, h, pofs, ptime, m2, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L)
                    else:
                        dev, _ = _de_load(delta_ptr, delta_bias_ptr, bh, h, L, offs_t, offs_t < L, zl, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L)
                        de0 = gl.expand_dims(dev, 0) + z2
                    inc *= de0
                thc += gl.sum(_zt(inc, m2, EVEN_L), axis=1)
                thc -= TWO_PI * libdevice.floor(thc * INV_TWO_PI)
    de = zl.to(gl.float32)
    for j in range(NBLK if RUN_P1 else 0):
        offs_t = j * BLOCK_L + col
        mask_t = offs_t < L
        m2 = gl.expand_dims(mask_t, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tw)
        po = gl.expand_dims(hp, 1) * L + gl.expand_dims(offs_t, 0)
        if HPACK:
            de2, _ = _de_load_t(delta_ptr, delta_bias_ptr, h, pofs, po, m2, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L)
        else:
            de, _ = _de_load(delta_ptr, delta_bias_ptr, bh, h, L, offs_t, mask_t, zl, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L)
            de2 = gl.expand_dims(de, 0) + z2
        u = _dw_load(dw_ptr, de2, bh, L, po, offs_t, mask_t, m2, z2, DELTA_SOFTPLUS, HAS_DW, DW_P, EVEN_L, HPACK) * _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        if PLAIN:
            sL, gap2 = _tail_sum(de2, rev2, prev2, first2, last2)
        for kk in range(k_lo, k_hi):
            if ROT:
                cT, sT = _rot_cs(theta_ptr, thck_ptr, thb, h, bh, kk, K, L, NBLK, j, offs_t, col, m2, z2, zl, de, INV_BP, ABL, FAST_TRIG, EVEN_L, THETA_DELTA, row1)
                if ROT_MIX:
                    rf = _rot_sel(row_rot_ptr, kk, NPAIR)
                    cT, sT = (rf * cT + (1.0 - rf), rf * sT)
            gB0, gC0 = (z2, z2) if HPACK else (zl.to(gl.float32), zl.to(gl.float32))
            for pi in pair_iterator.value(NPAIR):
                n = kk * NPAIR + pi
                bc = b * N * L + n.to(gl.int64) * L + offs_t
                if PMAP:
                    if PRAW:
                        mr = (b * N + 2 * kk.to(gl.int64)) * L + offs_t
                    else:
                        mr = (b * 2 * N + 2 * n.to(gl.int64)) * L + offs_t
                    P0 = _ldt(B_ptr + mr, mask_t, EVEN_L).to(gl.float32)
                    P1 = _ldt(B_ptr + mr + L, mask_t, EVEN_L).to(gl.float32)
                    w = P0 * cT - sT * P1 if pi == 0 else P1 * cT + sT * P0
                else:
                    w = _bc_load(B_ptr, beta_b_ptr, bc, n, gbc, N, mask_t, zl, HAS_BB, EVEN_L)
                if ROT and (not PMAP):
                    np_ = kk * NPAIR + 1 - pi
                    pc = b * N * L + np_.to(gl.int64) * L + offs_t
                    Bp = w if ABL & 128 else _bc_load(B_ptr, beta_b_ptr, pc, np_, gbc, N, mask_t, zl, HAS_BB, EVEN_L)
                    w = w * cT - sT * Bp if pi == 0 else w * cT + sT * Bp
                if HAS_GB:
                    w = w * _gate_load(gB_ptr, bgB_ptr, bc, n, gg, N, mask_t, zl, GATE_ACT, HAS_BGB, EVEN_L)
                w2 = gl.expand_dims(w, 0)
                if PLAIN:
                    c1v = -gl.exp(gl.load(A_log_ptr + anp + n * sn))
                    ltot = c1v * sL
                    lgap = gl.expand_dims(c1v, 1) * gap2
                else:
                    law_s = gl.load(row_law_ptr + n)
                    c1v = -gl.exp(gl.load(A_log_ptr + anp + n * sn))
                    s1, _ = _law_s1(a_dyn_ptr, law_s, de, b, bh, n, N, L, offs_t, mask_t, zl, HAS_DYN, UNI, DYN_ACT, DYN_EPS, A_DYN_N_RES, EVEN_L)
                    s1_2 = gl.expand_dims(s1, 0) + z2
                    S1L, s1_gap = _tail_sum(s1_2, rev2, prev2, first2, last2)
                    ltot = c1v * S1L
                    lgap = gl.expand_dims(c1v, 1) * s1_gap
                    if HAS_MU and (UNI or law_s == K_MU):
                        c2v = gl.load(mu_gain_ptr + anp + n * sn)
                        s2_2 = gl.expand_dims(_law_s2(mu_ptr, mu_init_ptr, b, bh, n, N, L, offs_t, mask_t, col, zl, MU_N_RES, HAS_MU_INIT, MU_PREDIFF, EVEN_L), 0) + z2
                        S2L, s2_gap = _tail_sum(s2_2, rev2, prev2, first2, last2)
                        ltot = ltot + c2v * S2L
                        lgap = lgap + gl.expand_dims(c2v, 1) * s2_gap
                ck = hck_ptr + ckp + (j * N + n) * snc
                gl.store(ck + N * snc, (libdevice.exp2(ltot * LOG2E) * gl.load(ck).to(gl.float32) + gl.sum(libdevice.exp2(lgap * LOG2E) * w2 * u, axis=1)).to(hck_ptr.dtype.element_ty))
    if DBACC:
        dba = gl.zeros([BLOCK_P], gl.float32, layout=pl)
    NBS = (NBLK + TS - 1) // TS
    j_hi = gl.minimum((ts + 1) * NBS, NBLK)
    for jj in range(j_hi - ts * NBS):
        j = j_hi - 1 - jj
        offs_t = j * BLOCK_L + col
        mask_t = offs_t < L
        m2 = gl.expand_dims(mask_t, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tw)
        carry = gl.zeros([BLOCK_P], gl.int32, layout=pl) + jj > 0
        lcar = gl.zeros([BLOCK_P], gl.int32, layout=pl) + jj >= 0 if HAS_GHL or TS > 1 else carry
        at_t0 = gl.zeros([BLOCK_P], gl.int32, layout=pl) + j == 0
        po = gl.expand_dims(hp, 1) * L + gl.expand_dims(offs_t, 0)
        if HPACK:
            de2, dsig = _de_load_t(delta_ptr, delta_bias_ptr, h, pofs, po, m2, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L, DSIG)
            den2 = gl.gather(de2, nxt2, 1)
        else:
            de, dsig = _de_load(delta_ptr, delta_bias_ptr, bh, h, L, offs_t, mask_t, zl, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L, DSIG)
            de2 = gl.expand_dims(de, 0) + z2
            den2 = gl.expand_dims(gl.gather(de, idx_nxt, 0), 0) + z2
        xt = _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        gyp = _ldt(gy_ptr + po, m2, EVEN_L).to(gl.float32)
        if HAS_Z:
            zt = _ldt(z_ptr + po, m2, EVEN_L).to(gl.float32)
            zsg = 1.0 / (1.0 + libdevice.exp2(-zt * LOG2E))
            gyz = gyp
            gyp = gyp * (zt * zsg)
        gyr = gl.gather(gyp, rev2, 1)
        dw2 = _dw_load(dw_ptr, de2, bh, L, po, offs_t, mask_t, m2, z2, DELTA_SOFTPLUS, HAS_DW, DW_P, EVEN_L, HPACK)
        u = dw2 * xt
        gu = z2
        if HAS_ENDPOINT:
            if HPACK or ENDPOINT_P:
                eo = _ldt(endpoint_ptr + po, m2, EVEN_L).to(gl.float32)
            else:
                eo = gl.expand_dims(_ldt(endpoint_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32), 0)
            endpoint_dot = gl.zeros([BLOCK_L], gl.float32, layout=lt)
            local_own = gl.sum(gyp * eo * xt, axis=0)
        if HAS_Z:
            ypre = z2
        gdev = z2 if HPACK else gl.zeros([BLOCK_L], gl.float32, layout=lt)
        if HAS_DYN and (not A_DYN_N_RES):
            grt_acc = gl.zeros([BLOCK_L], gl.float32, layout=lt)
        if HAS_MU and (not MU_N_RES):
            gs2_acc = gl.zeros([BLOCK_L], gl.float32, layout=lt)
        if PMAP:
            offs_v = j * BLOCK_L + gl.arange(0, BLOCK_L, layout=pv)
            mask_v = offs_v < L
            de_v = gl.convert_layout(de, pv)
        for kk in range(k_lo, k_hi):
            if PMAP:
                thv = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_v, mask_v, EVEN_L).to(gl.float32)
                if THETA_DELTA:
                    thv = thv * de_v
                Thv = gl.associative_scan(_zt(thv, mask_v, EVEN_L), axis=0, combine_fn=_add) + gl.load(thck_ptr + (bh * NBLK + j) * K + kk)
                if FAST_TRIG:
                    cT, sT = (libdevice.fast_cosf(Thv), libdevice.fast_sinf(Thv))
                else:
                    cT, sT = (gl.cos(Thv), gl.sin(Thv))
                if HAS_THH:
                    chh = gl.load(thh_ptr + (h * K + kk) * 2)
                    shh = gl.load(thh_ptr + (h * K + kk) * 2 + 1)
                    cC = cT * chh - sT * shh
                    sC = sT * chh + cT * shh
                else:
                    cC, sC = (cT, sT)
                if ROT_MIX:
                    rf = _rot_sel(row_rot_ptr, kk, NPAIR)
                    cT, sT = (rf * cT + (1.0 - rf), rf * sT)
                    cC, sC = (rf * cC + (1.0 - rf), rf * sC)
            elif ROT:
                if HPACK:
                    feat = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_t, mask_t, EVEN_L).to(gl.float32)
                    inc = gl.expand_dims(feat, 0) + z2
                    if THETA_DELTA:
                        inc *= de2
                    ph = gl.expand_dims(gl.load(thck_ptr + ((bh + pf) * NBLK + j) * K + kk), 1)
                    ph += gl.associative_scan(_zt(inc, m2, EVEN_L), axis=1, combine_fn=_add)
                    cT = libdevice.fast_cosf(ph) if FAST_TRIG else libdevice.cos(ph)
                    sT = libdevice.fast_sinf(ph) if FAST_TRIG else libdevice.sin(ph)
                else:
                    cT, sT = _rot_cs(theta_ptr, thck_ptr, thb, h, bh, kk, K, L, NBLK, j, offs_t, col, m2, z2, zl, de, INV_BP, ABL, FAST_TRIG, EVEN_L, THETA_DELTA, row1)
                if HAS_THH:
                    if HPACK:
                        tho = ((h + pf) * K + kk) * 2
                        chh = gl.expand_dims(gl.load(thh_ptr + tho), 1)
                        shh = gl.expand_dims(gl.load(thh_ptr + tho + 1), 1)
                    else:
                        chh = gl.load(thh_ptr + (h * K + kk) * 2)
                        shh = gl.load(thh_ptr + (h * K + kk) * 2 + 1)
                    cC = cT * chh - sT * shh
                    sC = sT * chh + cT * shh
                else:
                    cC, sC = (cT, sT)
                if ROT_MIX:
                    rf = _rot_sel(row_rot_ptr, kk, NPAIR)
                    cT, sT = (rf * cT + (1.0 - rf), rf * sT)
                    cC, sC = (rf * cC + (1.0 - rf), rf * sC)
            gB0, gC0 = (z2, z2) if HPACK else (zl.to(gl.float32), zl.to(gl.float32))
            if PMAP:
                gTa = gl.zeros([BLOCK_L], gl.float32, layout=pv)
                gTc = gl.zeros([BLOCK_L], gl.float32, layout=pv)
                if PRAW:
                    mrp = (b * N + 2 * kk.to(gl.int64)) * L + offs_v
                    R0 = _ldt(B_ptr + mrp, mask_v, EVEN_L).to(gl.float32)
                    R1 = _ldt(B_ptr + mrp + L, mask_v, EVEN_L).to(gl.float32)
                    S0 = _ldt(C_ptr + mrp, mask_v, EVEN_L).to(gl.float32)
                    S1 = _ldt(C_ptr + mrp + L, mask_v, EVEN_L).to(gl.float32)
                    dB0 = gl.zeros([BLOCK_L], gl.float32, layout=pv)
                    dB1 = gl.zeros([BLOCK_L], gl.float32, layout=pv)
                    dC0 = gl.zeros([BLOCK_L], gl.float32, layout=pv)
                    dC1 = gl.zeros([BLOCK_L], gl.float32, layout=pv)
            for pi in pair_iterator.value(NPAIR):
                n = kk * NPAIR + pi
                bc = b * N * L + n.to(gl.int64) * L + offs_t
                if PMAP:
                    if PRAW:
                        P0, P1, Q0, Q1 = (R0, R1, S0, S1)
                    else:
                        mr = (b * 2 * N + 2 * n.to(gl.int64)) * L + offs_v
                        P0 = _ldt(B_ptr + mr, mask_v, EVEN_L).to(gl.float32)
                        P1 = _ldt(B_ptr + mr + L, mask_v, EVEN_L).to(gl.float32)
                        Q0 = _ldt(C_ptr + mr, mask_v, EVEN_L).to(gl.float32)
                        Q1 = _ldt(C_ptr + mr + L, mask_v, EVEN_L).to(gl.float32)
                    if pi == 0:
                        wv, dwt = (P0 * cT - sT * P1, -(sT * P0) - cT * P1)
                        rv, drt = (Q0 * cC - sC * Q1, -(sC * Q0) - cC * Q1)
                    else:
                        wv, dwt = (P1 * cT + sT * P0, cT * P0 - sT * P1)
                        rv, drt = (Q1 * cC + sC * Q0, cC * Q0 - sC * Q1)
                    w = gl.convert_layout(wv, lt)
                    r = gl.convert_layout(rv, lt)
                else:
                    w = _bc_load(B_ptr, beta_b_ptr, bc, n, gbc, N, mask_t, zl, HAS_BB, EVEN_L)
                    r = _bc_load(C_ptr, beta_c_ptr, bc, n, gbc, N, mask_t, zl, HAS_BC, EVEN_L)
                w0, r0 = (z2, z2) if HPACK else (zl.to(gl.float32), zl.to(gl.float32))
                if ROT and (not PMAP):
                    np_ = kk * NPAIR + 1 - pi
                    pc = b * N * L + np_.to(gl.int64) * L + offs_t
                    if ABL & 128:
                        Bp, Cp = (w, r)
                    else:
                        Bp = _bc_load(B_ptr, beta_b_ptr, pc, np_, gbc, N, mask_t, zl, HAS_BB, EVEN_L)
                        Cp = _bc_load(C_ptr, beta_c_ptr, pc, np_, gbc, N, mask_t, zl, HAS_BC, EVEN_L)
                    if HPACK:
                        w, r = (gl.expand_dims(w, 0) + z2, gl.expand_dims(r, 0) + z2)
                        Bp, Cp = (gl.expand_dims(Bp, 0) + z2, gl.expand_dims(Cp, 0) + z2)
                    if pi == 0:
                        w, r = (w * cT - sT * Bp, r * cC - sC * Cp)
                    else:
                        w0 = Bp * cT - sT * w
                        r0 = Cp * cC - sC * r
                        w, r = (w * cT + sT * Bp, r * cC + sC * Cp)
                r = _zt(r, m2 if HPACK and ROT else mask_t, EVEN_L)
                if HAS_GB:
                    if HPACK and ROT:
                        raw_gate = gl.expand_dims(_ldt(gB_ptr + bc, mask_t, EVEN_L).to(gl.float32), 0) + z2
                        if HAS_BGB:
                            raw_gate += gl.expand_dims(gl.load(bgB_ptr + (h + pf) // HG_G * N + n), 1)
                        gtB = _gate(raw_gate, GATE_ACT)
                    else:
                        gtB = _gate_load(gB_ptr, bgB_ptr, bc, n, gg, N, mask_t, zl, GATE_ACT, HAS_BGB, EVEN_L)
                    w2 = w * gtB if HPACK and ROT else gl.expand_dims(w * gtB, 0)
                else:
                    w2 = w if HPACK and ROT else gl.expand_dims(w, 0)
                if HAS_GC:
                    if HPACK and ROT:
                        raw_gate = gl.expand_dims(_ldt(gC_ptr + bc, mask_t, EVEN_L).to(gl.float32), 0) + z2
                        if HAS_BGC:
                            raw_gate += gl.expand_dims(gl.load(bgC_ptr + (h + pf) // HG_G * N + n), 1)
                        gtC = _gate(raw_gate, GATE_ACT)
                    else:
                        gtC = _gate_load(gC_ptr, bgC_ptr, bc, n, gg, N, mask_t, zl, GATE_ACT, HAS_BGC, EVEN_L)
                    rg = r * gtC
                else:
                    rg = r
                if PLAIN:
                    c1v = -gl.exp(gl.load(A_log_ptr + anp + n * sn))
                    s1_2, s1n_2 = (de2, den2)
                else:
                    law_s = gl.load(row_law_ptr + n)
                    c1v = -gl.exp(gl.load(A_log_ptr + anp + n * sn))
                    s1, fac = _law_s1(a_dyn_ptr, law_s, de, b, bh, n, N, L, offs_t, mask_t, zl, HAS_DYN, UNI, DYN_ACT, DYN_EPS, A_DYN_N_RES, EVEN_L)
                    s1_2 = gl.expand_dims(s1, 0)
                    s1n_2 = gl.expand_dims(gl.gather(s1, idx_nxt, 0), 0)
                c1e = gl.expand_dims(c1v, 1)
                la = c1e * LOG2E * s1_2
                lan = c1e * LOG2E * s1n_2
                if HAS_MU:
                    c2e = gl.expand_dims(gl.zeros([BLOCK_P], gl.float32, layout=pl), 1)
                    s2_2 = z2
                    if UNI or law_s == K_MU:
                        c2e = gl.expand_dims(gl.load(mu_gain_ptr + anp + n * sn), 1)
                        s2 = _law_s2(mu_ptr, mu_init_ptr, b, bh, n, N, L, offs_t, mask_t, col, zl, MU_N_RES, HAS_MU_INIT, MU_PREDIFF, EVEN_L)
                        s2_2 = gl.expand_dims(s2, 0) + z2
                        la = la + c2e * LOG2E * s2_2
                        lan = lan + c2e * LOG2E * (gl.expand_dims(gl.gather(s2, idx_nxt, 0), 0) + z2)
                a = libdevice.exp2(la)
                cy = cyp + n * sn
                drive = w2 * u
                rr2 = (rg if ABL & 4 else gl.gather(rg, rev2, 1)) if HPACK and ROT else gl.expand_dims(rg if ABL & 4 else gl.gather(rg, idx_rev, 0), 0)
                if ABL & 2:
                    lam = gyr * rr2
                    if HAS_ENDPOINT:
                        gu += lam * w2
                        gBs = lam * u if HPACK and ROT else gl.sum(lam * u, axis=0)
                else:
                    ac = gl.load(alam_ptr + cy, mask=lcar, other=0.0)
                    beta = gyr * rr2
                    if HAS_ENDPOINT:
                        acs = gl.expand_dims(ac, 1)
                        _, lam_r, lex_r = gl.associative_scan((libdevice.exp2(lan), gl.where(first2, beta + acs, beta), gl.where(first2, acs + z2, 0.0)), axis=1, combine_fn=_ssm_combine_q)
                        lex = lex_r if ABL & 4 else gl.gather(lex_r, rev2, 1)
                        gu += lex * w2
                        gBs = lex * u if HPACK and ROT else gl.sum(lex * u, axis=0)
                        lam = lex + gyp * (rg if HPACK and ROT else gl.expand_dims(rg, 0))
                    else:
                        _, lam_r = gl.associative_scan((libdevice.exp2(lan), gl.where(first2, beta + gl.expand_dims(ac, 1), beta)), axis=1, combine_fn=_ssm_combine)
                        lam = lam_r if ABL & 4 else gl.gather(lam_r, rev2, 1)
                if ABL & 8:
                    qh = a * u
                    hs = drive + qh
                else:
                    hprev = gl.load(hck_ptr + ckp + (j * N + n) * snc).to(gl.float32)
                    hseed = gl.where(first2, drive + a * gl.expand_dims(hprev, 1), drive)
                    if ABL & 4096:
                        _, hs = gl.associative_scan((a, hseed), axis=1, combine_fn=_ssm_combine)
                    else:
                        _, hs, qh = gl.associative_scan((a, hseed, gl.where(first2, a, 0.0) * gl.expand_dims(hprev, 1)), axis=1, combine_fn=_ssm_combine_q)
                D = lam * (hs - drive) if ABL & 4096 else lam * qh
                if not HAS_ENDPOINT:
                    gu += lam * w2
                if not ABL & 2:
                    al0 = gl.sum(gl.where(first2, a * lam, 0.0), axis=1)
                    gl.store(alam_ptr + cy, al0)
                    gl.store(ghi_ptr + cy, al0, mask=at_t0)
                if not ABL & 1:
                    gl.store(gc1_ptr + cy, gl.load(gc1_ptr + cy, mask=carry, other=0.0) + gl.sum(D * s1_2, axis=1))
                    if HAS_MU and (UNI or law_s == K_MU):
                        gl.store(gc2_ptr + cy, gl.load(gc2_ptr + cy, mask=carry, other=0.0) + gl.sum(D * s2_2, axis=1))
                if not ABL & 16:
                    gs1 = D * c1e if HPACK else gl.sum(D * c1e, axis=0)
                    if PLAIN:
                        gdev += gs1
                    else:
                        gdev += gs1 * fac
                    if HAS_DYN and (UNI or law_s == K_DYN):
                        if A_DYN_N_RES:
                            _put(grate_ptr + lawbn + n.to(gl.int64) * L + offs_t, gs1 * de, mask_t, LWS != 0, EVEN_L, LAW_VEC, TPL, WOWN)
                        else:
                            grt_acc += gs1 * de
                    if HAS_MU and (UNI or law_s == K_MU):
                        gs2 = gl.sum(D * c2e, axis=0)
                        if MU_N_RES:
                            _put(gs2o_ptr + lawbn + n.to(gl.int64) * L + offs_t, gs2, mask_t, LWS != 0, EVEN_L, LAW_VEC, TPL, WOWN)
                        else:
                            gs2_acc += gs2
                    sc = bcn + n.to(gl.int64) * L + offs_t
                    if HAS_ENDPOINT:
                        qr = hs - drive if ABL & 4096 else qh
                        gB = gBs
                        gC = gyp * qr if HPACK and ROT else gl.sum(gyp * qr, axis=0)
                        wb = gl.sum(w2, axis=0)
                        gB += local_own * rg
                        gC += local_own * wb
                        endpoint_dot += wb * rg
                    else:
                        gB = lam * u if HPACK and ROT else gl.sum(lam * u, axis=0)
                        gC = gyp * hs if HPACK and ROT else gl.sum(gyp * hs, axis=0)
                    if HAS_Z:
                        if HAS_ENDPOINT:
                            ypre += qr * gl.expand_dims(rg, 0)
                        else:
                            ypre += hs * gl.expand_dims(rg, 0)
                    if HAS_GB:
                        if HPACK and ROT:
                            rv = gl.expand_dims(_ldt(gB_ptr + bc, mask_t, EVEN_L).to(gl.float32), 0) + z2
                            if HAS_BGB:
                                rv += gl.expand_dims(gl.load(bgB_ptr + (h + pf) // HG_G * N + n), 1)
                            dgt = _gate_dv(rv, GATE_ACT)
                        else:
                            dgt = _gate_load_d(gB_ptr, bgB_ptr, bc, n, gg, N, mask_t, zl, GATE_ACT, HAS_BGB, EVEN_L)
                        vgb = gB * w * dgt
                        _put(ggB_ptr + sc, gl.sum(vgb, axis=0) if HPACK and ROT else vgb, mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        if HPACK and ROT:
                            if BACC_GB:
                                gl.atomic_add(gbgB_ptr + (h + pf) // HG_G * N + n, gl.sum(_zt(vgb, m2, EVEN_L), axis=1), sem='relaxed')
                        else:
                            _gate_bacc(gbgB_ptr, gg, n, N, vgb, mask_t, BACC_GB, EVEN_L, BGPART, (gl.program_id(0) * Bs + b) * NBLK + j)
                        gB = gB * gtB
                    if HAS_GC:
                        if HPACK and ROT:
                            rv = gl.expand_dims(_ldt(gC_ptr + bc, mask_t, EVEN_L).to(gl.float32), 0) + z2
                            if HAS_BGC:
                                rv += gl.expand_dims(gl.load(bgC_ptr + (h + pf) // HG_G * N + n), 1)
                            dgt = _gate_dv(rv, GATE_ACT)
                        else:
                            dgt = _gate_load_d(gC_ptr, bgC_ptr, bc, n, gg, N, mask_t, zl, GATE_ACT, HAS_BGC, EVEN_L)
                        vgc = gC * r * dgt
                        _put(ggC_ptr + sc, gl.sum(vgc, axis=0) if HPACK and ROT else vgc, mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        if HPACK and ROT:
                            if BACC_GC:
                                gl.atomic_add(gbgC_ptr + (h + pf) // HG_G * N + n, gl.sum(_zt(vgc, m2, EVEN_L), axis=1), sem='relaxed')
                        else:
                            _gate_bacc(gbgC_ptr, gg, n, N, vgc, mask_t, BACC_GC, EVEN_L, BGPART, (gl.program_id(0) * Bs + b) * NBLK + j)
                        gC = gC * gtC
                    if PMAP:
                        scm = bcn + (2 * n).to(gl.int64) * L + offs_v
                        gBv = gl.convert_layout(gB, pv)
                        gCv = gl.convert_layout(gC, pv)
                        if pi == 0:
                            aB, bB, aC, bC = (cT, -sT, cC, -sC)
                        else:
                            aB, bB, aC, bC = (sT, cT, sC, cC)
                        if PRAW:
                            dB0 += aB * gBv
                            dB1 += bB * gBv
                            dC0 += aC * gCv
                            dC1 += bC * gCv
                        else:
                            _put(gBt_ptr + scm, aB * gBv, mask_v, ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                            _put(gBt_ptr + scm + L, bB * gBv, mask_v, ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                            _put(gCt_ptr + scm, aC * gCv, mask_v, ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                            _put(gCt_ptr + scm + L, bC * gCv, mask_v, ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                        thcr = _zt(gCv * drt, mask_v, EVEN_L)
                        gTa += _zt(gBv * dwt, mask_v, EVEN_L) + thcr
                        gTc += thcr
                    elif not ROT:
                        _put(gBt_ptr + sc, gB, mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _put(gCt_ptr + sc, gC, mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _bacc(gbb_ptr, gbc, n, N, gB, mask_t, BACC_B, EVEN_L)
                        _bacc(gbc_ptr, gbc, n, N, gC, mask_t, BACC_C, EVEN_L)
                    elif pi == 0:
                        gB0, gC0 = (gB, gC)
                    else:
                        sc0 = sc - L
                        vb0 = gB0 * cT + gB * sT
                        vc0 = gC0 * cC + gC * sC
                        vb1 = gB * cT - gB0 * sT
                        vc1 = gC * cC - gC0 * sC
                        _put(gBt_ptr + sc0, gl.sum(vb0, axis=0) if HPACK else vb0, mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _put(gCt_ptr + sc0, gl.sum(vc0, axis=0) if HPACK else vc0, mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _put(gBt_ptr + sc, gl.sum(vb1, axis=0) if HPACK else vb1, mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _put(gCt_ptr + sc, gl.sum(vc1, axis=0) if HPACK else vc1, mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _bacc(gbb_ptr, gbc, n - 1, N, gl.sum(vb0, axis=0) if HPACK else vb0, mask_t, BACC_B, EVEN_L)
                        _bacc(gbc_ptr, gbc, n - 1, N, gl.sum(vc0, axis=0) if HPACK else vc0, mask_t, BACC_C, EVEN_L)
                        _bacc(gbb_ptr, gbc, n, N, gl.sum(vb1, axis=0) if HPACK else vb1, mask_t, BACC_B, EVEN_L)
                        _bacc(gbc_ptr, gbc, n, N, gl.sum(vc1, axis=0) if HPACK else vc1, mask_t, BACC_C, EVEN_L)
                        gThC = _zt(r0 * gC - r * gC0, m2 if HPACK else mask_t, EVEN_L)
                        gTh = _zt(w0 * gB - w * gB0, m2 if HPACK else mask_t, EVEN_L) + gThC
                        if HAS_THH:
                            if ROT_MIX:
                                gThC = gThC * rf
                            if HPACK:
                                gl.store(gthh_ptr + (((h + pf) * Bs + b) * K + kk.to(gl.int64)) * NBLK + j, gl.sum(gThC, axis=1))
                            else:
                                gl.store(gthh_ptr + ((hslot * Bs + b) * K + kk.to(gl.int64)) * NBLK + j + zp, gl.zeros([BLOCK_P], gl.float32, layout=pl) + gl.sum(gThC), mask=pofs == 0)
                        if ROT_MIX:
                            gTh = gTh * rf
                        if not ABL & 64:
                            if HPACK:
                                rc = rcar_ptr + rcp + kk * P
                                Rr = gl.associative_scan(gl.gather(gTh, rev2, 1), axis=1, combine_fn=_add)
                                Rr += gl.expand_dims(gl.load(rc, mask=carry, other=0.0), 1)
                                gl.store(rc, gl.sum(gl.where(last2, Rr, 0.0), axis=1))
                                gTh = gl.gather(Rr, rev2, 1)
                            else:
                                rv1 = gl.expand_dims(gl.convert_layout(idx_rev, gl.SliceLayout(0, row1)), 0)
                                R1 = gl.associative_scan(gl.gather(gl.expand_dims(gl.convert_layout(gTh, gl.SliceLayout(0, row1)), 0), rv1, 1), axis=1, combine_fn=_add)
                                rc1 = rcar_ptr + bh * K * P + pid_p * BLOCK_P + kk * P
                                R1 += gl.where(jj > 0, gl.load(rc1), 0.0)
                                gl.store(rc1, gl.sum(gl.sum(gl.where(rv1 == 0, R1, 0.0), axis=1), axis=0))
                                gTh = gl.convert_layout(gl.sum(gl.gather(R1, rv1, 1), axis=0), lt)
                        if THETA_DELTA:
                            thr = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_t, mask_t, EVEN_L).to(gl.float32)
                            gdev += gl.expand_dims(thr, 0) * gTh if HPACK else thr * gTh
                            gTh = gTh * (de2 if HPACK else de)
                        _put(gth_ptr + (thn + kk.to(gl.int64)) * L + offs_t, gl.sum(gTh, axis=0) if HPACK else gTh, mask_t, ATOMIC_TH, EVEN_L, TH_VEC, TPL, WOWN)
            if PMAP:
                if PRAW:
                    scp = bcn + (2 * kk).to(gl.int64) * L + offs_v
                    _put(gBt_ptr + scp, dB0, mask_v, ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                    _put(gBt_ptr + scp + L, dB1, mask_v, ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                    _put(gCt_ptr + scp, dC0, mask_v, ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                    _put(gCt_ptr + scp + L, dC1, mask_v, ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                if HAS_THH:
                    if ROT_MIX:
                        gTc = gTc * rf
                    gl.store(gthh_ptr + ((hslot * Bs + b) * K + kk.to(gl.int64)) * NBLK + j + zp, gl.zeros([BLOCK_P], gl.float32, layout=pl) + gl.sum(gTc), mask=pofs == 0)
                if ROT_MIX:
                    gTa = gTa * rf
                if not ABL & 64:
                    rc1 = rcar_ptr + bh * K * P + pid_p * BLOCK_P + kk * P
                    gTa = gl.associative_scan(gTa, axis=0, combine_fn=_add, reverse=True) + gl.where(jj > 0, gl.load(rc1), 0.0)
                    gl.store(rc1, gl.sum(gl.where(offs_v == j * BLOCK_L, gTa, 0.0), axis=0))
                if THETA_DELTA:
                    thr = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_v, mask_v, EVEN_L).to(gl.float32)
                    gdev += gl.convert_layout(thr * gTa, lt)
                    gTa = gTa * de_v
                _put(gth_ptr + (thn + kk.to(gl.int64)) * L + offs_v, gTa, mask_v, ATOMIC_TH, EVEN_L, VEC_V, TPL_V)
        xt = _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        gxv = gu * dw2
        if HAS_ENDPOINT:
            local_grad = gyp * gl.expand_dims(endpoint_dot, 0)
            gxv += local_grad * eo
            if HAS_Z:
                ypre += eo * xt * gl.expand_dims(endpoint_dot, 0)
            if ENDPOINT_P:
                _st(gendpoint_ptr + gxb + po, local_grad * xt, m2, EVEN_L)
            elif HPACK:
                _st(gendpoint_ptr + gdb + po, local_grad * xt, m2, EVEN_L)
            else:
                _st(gendpoint_ptr + gdb + bh * L + offs_t, gl.sum(local_grad * xt, axis=0), mask_t, EVEN_L)
        if HAS_DSKIP:
            own = 1.0 if NS == 1 else gl.where(sh == 0, 1.0, 0.0)
            if HPACK:
                dv = gl.expand_dims(gl.load(d_skip_ptr + h + pofs), 1) * own
                gxv += dv * gyp
                gl.atomic_add(gd_skip_ptr + h + pofs, own * gl.sum(gl.where(m2, gyp * xt, 0.0), axis=1), sem='relaxed')
            else:
                dv = gl.load(d_skip_ptr + h) * own
                gxv += dv * gyp
                gl.atomic_add(gd_skip_ptr + h, own * gl.sum(gl.sum(gl.where(m2, gyp * xt, 0.0), axis=1), axis=0), sem='relaxed')
        if HAS_Z:
            if HAS_DSKIP:
                ypre += dv * xt
            _st(gz_ptr + po, gyz * ypre * _silu_fast_d(zt, zsg), m2, EVEN_L)
        _put(gx_ptr + gxo + po, gxv, m2, ATOMIC_GX, EVEN_L, GX_VEC, 32)
        if HAS_DW:
            if DW_P:
                _st(gdw_ptr + gxb + po, gu * xt, m2, EVEN_L)
            elif HPACK:
                _st(gdw_ptr + gdb + po, gu * xt, m2, EVEN_L)
            else:
                _st(gdw_ptr + gdb + bh * L + offs_t, gl.sum(gu * xt, axis=0), mask_t, EVEN_L)
        else:
            gdev += gu * xt if HPACK else gl.sum(gu * xt, axis=0)
        if DELTA_SOFTPLUS and DSIG:
            gdev = gdev * dsig
        if HPACK:
            _st(gde_ptr + gdb + po, gdev, m2, EVEN_L)
            if DBACC:
                dba += gl.sum(gl.where(m2, gdev, 0.0), axis=1)
        else:
            _st(gde_ptr + gdb + bh * L + offs_t, gdev, mask_t, EVEN_L)
            if DBACC:
                dba += gl.zeros([BLOCK_P], gl.float32, layout=pl) + gl.sum(_zt(gdev, mask_t, EVEN_L))
        if HAS_DYN and (not A_DYN_N_RES):
            _st(grate_ptr + gdb + bh * L + offs_t, grt_acc, mask_t, EVEN_L)
        if HAS_MU and (not MU_N_RES):
            _st(gs2o_ptr + gdb + bh * L + offs_t, gs2_acc, mask_t, EVEN_L)
    if DBACC:
        if HPACK:
            gl.store(gdbias_ptr + gdbh + bh + pf, dba)
        else:
            gl.store(gdbias_ptr + gdbh + bh + zp, dba, mask=pofs == 0)
_PLANS: dict = {}
_PLAN_ON = os.environ.get('MH_BWD_PLAN', '1') != '0'
_ROW_SEL = ('row_law', 'row_rot')
_ENV_PFX = (b'MH_', b'HYBRID_', b'SSD_')
_PLAIN = frozenset((type(None), bool, int, float, str))

def _plan_key(argv):
    parts = []
    for k, v in argv.items():
        t = type(v)
        if t in _PLAIN:
            parts.append(v)
        elif isinstance(v, torch.Tensor):
            if k in _ROW_SEL:
                parts.append(tuple(selector_window(v, 0, v.numel())))
            else:
                parts.append((v.shape, v.stride(), v.dtype, v.device, v.data_ptr() % 16 == 0))
        elif isinstance(v, dict):
            parts.append(tuple(sorted(((a, repr(b)) for a, b in v.items()))))
        else:
            parts.append((t, repr(v)))
    from ... import kernel_options as _ko
    _raw = getattr(os.environ, '_data', None)
    env = frozenset(((a, b) for a, b in _raw.items() if a.startswith(_ENV_PFX))) if isinstance(_raw, dict) and _raw and isinstance(next(iter(_raw)), bytes) else frozenset(((a, b) for a, b in os.environ.items() if a.startswith(('MH_', 'HYBRID_', 'SSD_'))))
    return (tuple(parts), env, repr(_ko.current('hybrid_mamba')), bool(slp.ENABLED), autotune.tier())

def mh_backward(x, delta, A_log, B_t, C_t, gy, *, theta=None, theta_init=None, row_law=None, row_rot=None, a_dyn=None, mu=None, mu_gain=None, mu_init=None, mu_prediff=False, dyn_act=DYN_RAW, dyn_eps=0.0, a_dyn_res='n', mu_res='n', h_init=None, delta_bias=None, delta_softplus=True, beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None, g_B=None, g_C=None, gate_act=GATE_SIGMOID, delta_w=None, gh_final=None, d_skip=None, z=None, endpoint_own=None, n_off=0, n_rows=None, config=None, out=None, hck=None, mem=None, theta_delta=False, theta_h=None, bc_head=False, pmap=False):
    _argv = dict(locals())
    _pk = _plan_key(_argv) if config is None and out is None and _PLAN_ON else None
    _pl = _PLANS.get(_pk) if _pk is not None else None
    if _pl is not None and _pl[1] is not None and (autotune.cached(_pl[1]) is not _pl[2]):
        _pl = None
    if _pl is not None:
        A_DYN_N_RES, BACC_B, BACC_C, BACC_GB, BACC_GC, BL, BLK_P, Bs, CKPACK, DBACC, DSIG, DW_P, G_BC, G_G, H, HAS_DYN, HAS_MU, HG_BC, HG_G, HP, HPACK, HSL, K, L, LAW_MASK, MU_N_RES, N, NBLK, NBS, NP, P, ROT, ROT_MIX, TS, _cc, bc_exact, bias_part, cfg, has_bias, nr, vec_ok = _pl[0]
        cfg = dict(cfg)
        if row_rot is not None:
            row_rot = _i32(row_rot).contiguous()
        if row_law is not None:
            row_law = _i32(row_law).contiguous()
    else:
        if mu_prediff and mu_init is not None:
            raise ValueError('mu_prediff means `mu` already holds the telescoped\nincrement, so mu_init is folded into column 0 and must\nbe None -- applying it again shifts row 0 of every b')
        Bs, H, P, L = x.shape
        N = B_t.shape[-2] // (2 if pmap and pmap != 'raw' else 1)
        K = N // 2
        nr = N - n_off if n_rows is None else n_rows
        ROT, ROT_MIX = (theta is not None, False)
        if row_rot is not None:
            rot_w = selector_window(row_rot, n_off, nr)
            ROT_MIX = ROT and (not all(rot_w))
            ROT = ROT and any(rot_w)
            row_rot = _i32(row_rot).contiguous()
        if pmap:
            if theta is None or bc_head or any((t is not None for t in (g_B, g_C, beta_b, beta_c, beta_gB, beta_gC))):
                raise ValueError('pmap: pair maps belong to a rotating launch with every B/C fold (bias, gate) applied')
            if not ROT:
                ROT, ROT_MIX = (True, True)
        if ROT and (n_off % 2 or nr % 2):
            raise ValueError('a rotating partial launch must start and end on a pair boundary: got n_off=%d n_rows=%d' % (n_off, nr))
        LAW_MASK = 1
        if row_law is not None:
            window = set(selector_window(row_law, n_off, nr))
            row_law = _i32(row_law).contiguous()
            LAW_MASK = sum((1 << law for law in (LAW_A_FIX, LAW_A_DYN, LAW_MU) if law in window)) or 1
        G_BC = beta_b.shape[0] if beta_b is not None else beta_c.shape[0] if beta_c is not None else 1
        HG_BC = H // G_BC
        G_G = beta_gB.shape[0] if beta_gB is not None else beta_gC.shape[0] if beta_gC is not None else 1
        HG_G = H // G_G
        _bshared = [t is None or t.shape[0] == 1 for t in (beta_b, beta_c, beta_gB, beta_gC)]
        _bgrouped = [t is not None and t.shape[0] > 1 for t in (beta_b, beta_c, beta_gB, beta_gC)]
        has_bias = any((t is not None for t in (beta_b, beta_c, beta_gB, beta_gC)))
        _pin = autotune.pinned_config('bwd') if config is None else None
        if config is None and (not _pin) and (out is None) and (hck is not None) and (P == 1):
            from . import calibration_mh as _cm
            if _cm.p1_envelope(P=P, ROT=ROT, pmap=pmap, bc_head=bc_head, LAW_MASK=LAW_MASK, endpoint_own=endpoint_own, n_off=n_off, nr=nr, N=N, absent=(delta_w, g_B, g_C, beta_a, beta_b, beta_c, beta_gB, beta_gC, a_dyn, mu, theta, mu_init, theta_init, gh_final)):
                _pin = _cm.p1_config('bwd', Bs, H, N, L, x.dtype, _cm.p1_feat(delta_softplus=delta_softplus, delta_bias=delta_bias, d_skip=d_skip, z=z, h_init=h_init)) or None

        def _axis(name, env, dflt):
            for src in (config, mem, _pin):
                if src and name in src:
                    return src[name]
            return _envi(env, dflt)
        bacc_on = bool(_axis('BACC', 'MH_BWD_BIAS_ACC', 1)) and _axis('BCS', 'MH_BWD_BC_SLOTS', None) != 0
        if config is not None and '_resolved_bacc' in config:
            bacc_on = bool(config['_resolved_bacc'])
        _acc = [g and bacc_on for g in _bgrouped]
        BACC_B, BACC_C, BACC_GB, BACC_GC = _acc
        bias_part = autotune.gate_bias_partial(P) and (BACC_GB or BACC_GC)
        bc_exact = not bc_head and all((sh or ac for sh, ac in zip(_bshared, _acc)))
        HAS_DYN = bool(LAW_MASK >> LAW_A_DYN & 1)
        HAS_MU = bool(LAW_MASK >> LAW_MU & 1)
        A_DYN_N_RES, MU_N_RES = (a_dyn_res == 'n', mu_res == 'n')
        DW_P = delta_w is not None and delta_w.dim() == 4
        hpack_ok = P == 1 and LAW_MASK == 1 and (not pmap) and (theta is None or theta.dim() == 3) and (theta is None or (hck is not None and (not bias_part) and (beta_b is None) and (beta_c is None) and (endpoint_own is None) and (z is None))) and bc_exact and (hck is None or hck.dim() == 4)
        cfg = dict(NUM_WARPS=None, BLOCK_L=None, TP_P=None, ABL=0, HP=_hp_default(H) if hpack_ok else 1, RUN_P1=hck is None, NS=None, NP=auto_np(P), RUN_TH=True, FAST_TRIG=True, PU=0, MAXNREG=_maxnreg()[0], AGX=_envi('MH_BWD_ATOMIC_GX', 1), BCS=_envi('MH_BWD_BC_SLOTS', None), LWS=_envi('MH_BWD_LAW_SLOTS', 1), BCBF=_envi('MH_BWD_BC_BF16', 1), XSBF=_envi('MH_BWD_BF16_SLOTS', 0), DEBF=_envi('MH_BWD_DE_16', 1), THS=_envi('MH_BWD_TH_SLOTS', None), VBC=_envi('MH_BWD_BC_VEC', 1), VGX=_envi('MH_BWD_GX_VEC', 1), WP=_envi('MH_BWD_WP', 1), BACC=_envi('MH_BWD_BIAS_ACC', 1), DBACC=_envi('MH_BWD_DBIAS_ACC', 1), CKBF=_envi('MH_BWD_CK_BF16', 1), TS=_envi('MH_BWD_TS', 1))
        if _pin:
            cfg.update(_pin)
            if cfg.get('MAXNREG') is not None and cfg['MAXNREG'] <= 0:
                cfg['MAXNREG'] = None
        elif config is None:
            _tab = autotune.table_bwd(torch.cuda.get_device_capability(x.device)[0] * 10 + torch.cuda.get_device_capability(x.device)[1], theta is not None, x.element_size(), P)
            cfg.update({k: v for k, v in (_tab or {}).items() if k not in (mem or {})})
        cfg.update({k: v for k, v in (mem or {}).items() if k in cfg})
        cfg.update(config or {})
        if cfg['AGX'] == 0 and cfg['BCS'] == 0:
            cfg['LWS'] = 0
        mem_pinned = tuple(sorted((k for k in mem or {} if k in cfg)))
        if not hpack_ok:
            cfg['HP'] = 1
        _hg = HG_BC if ROT else min(HG_BC, HG_G)
        while cfg['HP'] > 1 and (H % cfg['HP'] or _hg % cfg['HP']):
            cfg['HP'] //= 2
        HPACK = cfg['HP'] > 1
        if HPACK:
            cfg['NP'] = 1
        while cfg['NP'] > 1 and (P % cfg['NP'] or P // cfg['NP'] < 8):
            cfg['NP'] //= 2
        NP = cfg['NP']
        if cfg['NS'] is None:
            cfg['NS'] = max(1, min(triton.next_power_of_2(max(4096 // (H * Bs), 1)), nr // 8, 8))
        if z is not None:
            cfg['NS'] = 1
        while cfg['NS'] > 1 and nr % cfg['NS']:
            cfg['NS'] //= 2
        if ROT:
            while cfg['NS'] > 1 and nr // cfg['NS'] % 2:
                cfg['NS'] //= 2
        if hck is not None:
            cfg['BLOCK_L'] = auto_block_l(P, L)
        BL = auto_block_l(cfg['HP'], L, cfg['BLOCK_L'], budget=HPACK_BUDGET) if HPACK and hck is None else auto_block_l(P, L, cfg['BLOCK_L'])
        if cfg['NUM_WARPS'] is None:
            cfg['NUM_WARPS'] = 1
        if pmap and cfg['NUM_WARPS'] > 1 and (BL < 32 * cfg['NUM_WARPS']):
            cfg['NUM_WARPS'] = max(1, BL // 32)
        if cfg['TP_P'] is None:
            _bp = cfg['HP'] if HPACK else P // cfg['NP']
            cfg['TP_P'] = max(2 if ROT else 4, min(_bp // 8 if ROT else _bp // 4, 32))
            if not ROT:
                v = _vec_tp_p(x.element_size(), BL, cfg['NUM_WARPS'], _bp, L)
                if v is not None:
                    cfg['TP_P'] = max(cfg['TP_P'], v)
        cfg['BLOCK_L'] = BL
        _cc = torch.cuda.get_device_capability(x.device)
        vec_ok = _cc[0] >= 9
        if cfg['BCS'] is None:
            cfg['BCS'] = 1 if bc_exact or (bc_head and (not has_bias) and (g_B is None) and (g_C is None)) else 0
        th_fold_ok = ROT and cfg['BCS'] != 0 and (not (theta is not None and theta.dim() == 4))
        if cfg['THS'] is None:
            cfg['THS'] = 1 if th_fold_ok else 0
        elif cfg['THS'] and (not th_fold_ok):
            cfg['THS'] = 0
        _tier = autotune.tier()
        if config is None and out is None and (not _pin) and autotune.enabled():
            key = ('mh_bwd', _tier, slp.active(torch.cuda.get_device_capability(x.device)[0] * 10 + torch.cuda.get_device_capability(x.device)[1]), bool(bc_head), str(pmap), B_t.dtype, Bs, H, P, N, L, nr, n_off, x.dtype, LAW_MASK, ROT, ROT_MIX, HAS_DYN, HAS_MU, A_DYN_N_RES, MU_N_RES, DW_P, g_B is not None, g_C is not None, gh_final is not None, h_init is not None, delta_w is not None, bool(delta_softplus), cfg['RUN_P1'], cfg['RUN_TH'], hck is not None, None if hck is None else hck.dtype, cfg['HP'], cfg['AGX'], cfg['BCS'], cfg['THS'], cfg['BCBF'], cfg['XSBF'], cfg['LWS'], cfg['VBC'], cfg['VGX'], cfg['CKBF'], cfg['DBACC'], cfg['DEBF'], d_skip is not None, z is not None, cfg['TS'], endpoint_own is not None, None if endpoint_own is None else endpoint_own.ndim, os.environ.get('MH_BWD_MAXNREG'), BACC_B, BACC_C, BACC_GB, BACC_GC, bias_part, theta is not None, None if theta is None else theta.dim(), bool(theta_delta) and theta is not None, theta_h is not None, int(dyn_act), float(dyn_eps), int(gate_act), mu_init is not None, bool(mu_prediff), delta_bias is not None, *(None if t is None else t.shape[0] for t in (beta_b, beta_c, beta_gB, beta_gC)), tuple(((k, cfg[k]) for k in mem_pinned)))
            _hit = autotune.cached(key)
            if _hit is not None:
                cfg = _hit
            else:
                kw = {k: v for k, v in _argv.items() if k not in ('x', 'delta', 'A_log', 'B_t', 'C_t', 'gy', 'config', 'out')}

                def trial(c, trial_alog=A_log):
                    result = mh_backward(x, delta, trial_alog, B_t, C_t, gy, **kw, config=dict(c, _resolved_bacc=bacc_on))
                    return tuple((result[name] for name, value in _argv.items() if name not in ('gy', 'hck', 'out') and isinstance(value, torch.Tensor) and value.is_floating_point() and (name in result)))
                extreme = None if hck is not None else lambda c, _al=torch.full_like(A_log, _EXTREME_A_LOG): trial(c, _al)
                xs1 = Bs * H * P * L * 4
                ck_isz = 2 if cfg['CKBF'] and x.element_size() == 2 else 4

                def ck_bytes(bl):
                    return Bs * H * (triton.cdiv(L, bl) + 1) * N * P * ck_isz
                des1 = Bs * H * L * 4
                nbc = 2 + (g_B is not None) + (g_C is not None)
                bc_rows = 2 * N if pmap and pmap != 'raw' else N

                def cfg_bytes(c):
                    ns = c['NS'] * c.get('NP', 1)
                    hnp = H // c.get('HP', 1) * c.get('NP', 1)
                    slabs = max(1, min(c['BCS'] or hnp, hnp)) if bc_exact else hnp
                    if bc_head and (not has_bias) and (g_B is None) and (g_C is None) and c['BCS']:
                        slabs = H
                    bcw = 2 if bc_exact and c['BCBF'] and (x.element_size() == 2) and (not has_bias) and (slabs == hnp) else 4
                    atom_x = c['AGX'] and c['NS'] > 1
                    xw = 2 if x.element_size() == 2 and (not atom_x) and (c['NS'] == 1 or c.get('XSBF', 0)) else 4
                    return (1 if atom_x else c['NS']) * (xs1 // 4) * xw + ns * des1 + (ck_bytes(auto_block_l(c.get('HP', 1) if c.get('HP', 1) > 1 else P, L, c['BLOCK_L'])) if hck is None else 0) + slabs * Bs * bc_rows * L * nbc * bcw + max(1, min(c['LWS'] or hnp, hnp)) * Bs * N * L * 4 * (int(HAS_DYN and A_DYN_N_RES) + int(HAS_MU and MU_N_RES))
                from .ts_adjoint import ts_refusal as _tsr
                _ts_ok = hck is not None and (not cfg['RUN_P1']) and (triton.cdiv(L, auto_block_l(P, L)) >= int(os.environ.get('MH_BWD_TS_MIN_NBLK', 64))) and (_tsr(ROT=ROT, pmap=pmap, LAW_MASK=LAW_MASK, hck=hck, delta_w=delta_w, beta_b=beta_b, beta_c=beta_c, g_B=g_B, g_C=g_C, endpoint_own=endpoint_own, bc_head=bc_head, nr=nr, N=N, NP=cfg['NP']) is None)
                cfg = autotune.choose_axes(key, cfg, {k: v for k, v in autotune.v3_axes(cfg, nr, ROT, xs_bytes1=xs1, ckpt_bytes=ck_bytes, tune_block_l=hck is None, bc_axis=bc_exact and vec_ok, ts_axis=_ts_ok).items() if k not in mem_pinned and (_tier == 'full' or k in ('NS', 'TP_P', 'BLOCK_L', 'TS'))}, trial, bytes_of=cfg_bytes, extreme=extreme, persist=_mh_bwd_kernel.cache_key, pre=('TS',), grid=autotune.v3_grid(cfg, _hg, L, (lambda hp: BL) if hck is not None else lambda hp: auto_block_l(hp, L, budget=HPACK_BUDGET) if hp > 1 else auto_block_l(P, L), caps=() if _maxnreg()[1] else (128, 168), half_block_l=hck is None) if hpack_ok else None, drop_keys=mem_pinned)
            if z is not None:
                cfg['NS'] = 1
            while cfg['NS'] > 1 and (nr % cfg['NS'] or (ROT and nr // cfg['NS'] % 2)):
                cfg['NS'] //= 2
            while cfg['HP'] > 1 and (H % cfg['HP'] or _hg % cfg['HP']):
                cfg['HP'] //= 2
            HPACK = cfg['HP'] > 1
            if hck is not None:
                cfg['BLOCK_L'] = auto_block_l(P, L)
            BL = auto_block_l(cfg['HP'], L, cfg['BLOCK_L'], budget=HPACK_BUDGET) if HPACK and hck is None else auto_block_l(P, L, cfg['BLOCK_L'])
            if pmap and cfg['NUM_WARPS'] > 1 and (BL < 32 * cfg['NUM_WARPS']):
                cfg['NUM_WARPS'] = max(1, BL // 32)
            if cfg['THS'] and (not (ROT and cfg['BCS'] != 0)):
                cfg['THS'] = 0
        NBLK = triton.cdiv(L, BL)
        HP = cfg['HP']
        TS, NBS = (1, NBLK)
        if cfg.get('TS', 1) > 1 and (not cfg['RUN_P1']):
            from .ts_adjoint import ts_refusal
            _why = ts_refusal(ROT=ROT, pmap=pmap, LAW_MASK=LAW_MASK, hck=hck, delta_w=delta_w, beta_b=beta_b, beta_c=beta_c, g_B=g_B, g_C=g_C, endpoint_own=endpoint_own, bc_head=bc_head, nr=nr, N=N, NP=NP)
            if _why is None:
                TS = min(cfg['TS'], NBLK)
                NBS = triton.cdiv(NBLK, TS)
        cfg['TS'] = TS
        DSIG = HPACK and bool(delta_softplus)
        DBACC = delta_bias is not None and bool(cfg['DBACC']) and (DSIG or not delta_softplus)
        CKPACK = hck.dim() == 4 if hck is not None else HPACK
        BLK_P = HP if HPACK else P // cfg['NP']
        HSL = H // HP * cfg['NP']
        if _pk is not None:
            _tk = locals().get('key') if 'key' in locals() else None
            _PLANS[_pk] = ((A_DYN_N_RES, BACC_B, BACC_C, BACC_GB, BACC_GC, BL, BLK_P, Bs, CKPACK, DBACC, DSIG, DW_P, G_BC, G_G, H, HAS_DYN, HAS_MU, HG_BC, HG_G, HP, HPACK, HSL, K, L, LAW_MASK, MU_N_RES, N, NBLK, NBS, NP, P, ROT, ROT_MIX, TS, _cc, bc_exact, bias_part, cfg, has_bias, nr, vec_ok), _tk, autotune.cached(_tk) if _tk is not None else None)
            _PLANS[_pk] = (tuple((dict(v) if isinstance(v, dict) else v for v in _PLANS[_pk][0])),) + _PLANS[_pk][1:]
    if out is None:
        dev, f32 = (x.device, torch.float32)
        zf = lambda *s: torch.zeros(*s, device=dev, dtype=f32)
        e = lambda *s: torch.empty(*s, device=dev, dtype=f32)
        z1 = zf(1)
        ns = cfg['NS']
        atomic_gx = ns > 1 and cfg['AGX']
        nsp = ns * NP
        law_slots = max(1, min(cfg['LWS'] or HSL, HSL))
        nsp_db = nsp * TS
        bcs = HSL
        if bc_exact:
            bcs = max(1, min(cfg['BCS'] or bcs, bcs))
        elif bc_head and (not has_bias) and (g_B is None) and (g_C is None) and cfg['BCS']:
            bcs = H
        atomic_bc = bcs < HSL
        bc_dt = x.dtype if bc_exact and cfg['BCBF'] and (not atomic_bc) and (not has_bias) and (x.element_size() == 2) else f32

        def bc(*rest):
            return (torch.zeros if atomic_bc else torch.empty)(bcs, *rest, device=dev, dtype=bc_dt)
        xs_dt = x.dtype if x.element_size() == 2 and (not atomic_gx) and (ns == 1 or cfg['XSBF']) else f32
        de_dt = delta.dtype if cfg['DEBF'] and delta.element_size() == 2 and (nsp == 1) and (DSIG or not delta_softplus) and (delta_bias is None or DBACC) else f32
        out = dict(x_s=(torch.zeros if atomic_gx else torch.empty)(1 if atomic_gx else ns, Bs, H, P, L, device=dev, dtype=xs_dt), de_s=torch.empty(nsp, Bs, H, L, device=dev, dtype=de_dt), gdb_s=torch.empty(nsp_db, Bs, H, device=dev, dtype=f32) if DBACC else z1, c1_s=(zf if nr != N or TS > 1 else e)(*((TS,) if TS > 1 else ()), Bs, N, H * P) if HPACK else (zf if nr != N or TS > 1 else e)(*((TS,) if TS > 1 else ()), Bs, H, N, P), hi_s=(zf if nr != N else e)(Bs, N, H * P) if HPACK else (zf if nr != N else e)(Bs, H, N, P), Bs_=bc(Bs, 2 * N if pmap and pmap != 'raw' else N, L), Cs_=bc(Bs, 2 * N if pmap and pmap != 'raw' else N, L), hck=torch.empty(*((Bs, NBLK + 1, N, H * P) if CKPACK else (Bs, H, NBLK + 1, N, P)), device=dev, dtype=x.dtype if cfg['CKBF'] and x.element_size() == 2 else f32) if hck is None else hck, alam=e(*((TS,) if TS > 1 else ()), Bs, N, H * P) if HPACK else e(*((TS,) if TS > 1 else ()), Bs, H, N, P), thck=e(Bs, H, NBLK, K) if ROT else z1, rcar=e(Bs, H, K, P) if ROT else z1, gth_s=zf(min(HSL, cfg['THS']) if cfg['THS'] else HSL, Bs, K, L) if ROT else z1, gthh_s=(zf if nr != N else e)(H if HPACK else HSL, Bs, K, NBLK) if ROT and theta_h is not None else z1, c2_s=zf(Bs, H, N, P) if HAS_MU else z1, grate=(zf(law_slots, Bs, N, L) if A_DYN_N_RES else zf(nsp, Bs, H, L)) if HAS_DYN else z1, gs2o=(zf(law_slots, Bs, N, L) if MU_N_RES else zf(nsp, Bs, H, L)) if HAS_MU else z1, gdw_s=(e(ns, Bs, H, P, L) if DW_P else e(nsp, Bs, H, L)) if delta_w is not None else z1, ggB_s=bc(Bs, N, L) if g_B is not None else z1, ggC_s=bc(Bs, N, L) if g_C is not None else z1, gbb_s=zf(G_BC, N) if BACC_B else z1, gbc_s=zf(G_BC, N) if BACC_C else z1, gbgB_s=zf(G_G, N) if BACC_GB else z1, gbgC_s=zf(G_G, N) if BACC_GC else z1)
    if bias_part:
        shape = (H // HP * NP * Bs * NBLK, N)
        if BACC_GB:
            out['gbgB_s'] = torch.zeros(shape, device=x.device, dtype=torch.float32)
        if BACC_GC:
            out['gbgC_s'] = torch.zeros(shape, device=x.device, dtype=torch.float32)
    cks = out['hck'].shape[1] if CKPACK else out['hck'].shape[2]
    if hck is not None and cks < NBLK + 1:
        raise ValueError("the handed-over hck carries %d chunk slots but this backward chunks L=%d at BLOCK_L=%d (%d chunks): the checkpoint stride and the backward's chunk length must be the same number" % (cks, L, BL, NBLK))
    atomic_gx = out['x_s'].shape[0] == 1 < cfg['NS']
    bcs = out['Bs_'].shape[0]
    atomic_bc = bcs < HSL
    if bc_head and atomic_bc and (bcs != H or has_bias or g_B is not None or (g_C is not None)):
        raise ValueError('prepared-map reduction needs one slab per head and no unfused B/C features')
    law_slots = out['grate'].shape[0] if HAS_DYN and A_DYN_N_RES else out['gs2o'].shape[0] if HAS_MU and MU_N_RES else HSL
    if HAS_DYN and A_DYN_N_RES and HAS_MU and MU_N_RES:
        if out['grate'].shape[0] != out['gs2o'].shape[0]:
            raise ValueError('dynamic-rate and mu adjoint slots must match')
    atomic_law = law_slots < HSL
    for name, active in (('grate', HAS_DYN and A_DYN_N_RES), ('gs2o', HAS_MU and MU_N_RES)):
        if active and atomic_law and (out[name].dtype != torch.float32):
            raise ValueError('folded law adjoint slots must be fp32')
    ths = out['gth_s'].shape[0] if ROT and theta is not None else HSL
    atomic_th = ths < HSL
    if atomic_bc and out['Bs_'].dtype != torch.float32:
        raise ValueError('a folded (atomic) BC slot buffer must be fp32')
    tp_p = min(cfg['TP_P'], BLK_P)
    tp_l = max(32 // tp_p, 1)
    WP = max(1, int(cfg.get('WP', 1) or 1))
    while WP > 1 and BLK_P < tp_p * WP:
        WP //= 2
    cfg['WP'] = WP
    NW_T = cfg['NUM_WARPS'] * WP
    spt_l = max(BL // (tp_l * cfg['NUM_WARPS']), 1)

    def _vecw(on):
        if on and vec_ok:
            for w_ in (4, 2):
                if spt_l % w_ == 0 and L % w_ == 0:
                    return w_
        return 0
    bc_vec = _vecw(atomic_bc and cfg['VBC'])
    spt_v = max(BL // (32 * NW_T), 1)
    tpl_v = min(32, max(BL // NW_T, 1))
    vec_v = 0
    if pmap and vec_ok and cfg['VBC'] and (WP == 1 or BL >= 32 * NW_T):
        for w_ in (4, 2):
            if spt_v % w_ == 0 and L % w_ == 0:
                vec_v = w_
                break
    th_vec = _vecw(atomic_th and cfg['VBC'])
    gx_vec = _vecw(atomic_gx and cfg['VGX'])

    def _pfold(t):
        return t if NP == 1 else t.reshape(-1, NP, *t.shape[1:]).sum(1)
    ck0 = out['hck'][:, 0] if CKPACK else out['hck'][:, :, 0]
    if h_init is not None and hck is None:
        ck0.copy_(_f32(h_init).reshape(Bs, H * P, N).permute(0, 2, 1) if CKPACK else _f32(h_init).reshape(Bs, H, P, N).permute(0, 1, 3, 2))
    elif hck is None:
        ck0.zero_()
    if gh_final is not None and TS == 1:
        out['alam'].copy_(_f32(gh_final).reshape(Bs, H * P, N).permute(0, 2, 1) if HPACK else _f32(gh_final).reshape(Bs, H, P, N).permute(0, 1, 3, 2))
    alog_e = _f32(A_log)
    if beta_a is not None:
        ba_h = _f32(beta_a)[group_index(H, beta_a.shape[0], x.device)].view(H, 1, N)
        alog_e = alog_e + ba_h
    gd_skip = torch.zeros(H, device=x.device, dtype=torch.float32) if d_skip is not None else torch.zeros(1, device=x.device, dtype=torch.float32)
    gz = torch.empty_like(z) if z is not None else torch.zeros(1, device=x.device, dtype=torch.float32)
    endpoint_grad = torch.zeros_like(out['gdw_s'] if endpoint_own.ndim == 4 else out['de_s'], dtype=torch.float32) if endpoint_own is not None else None
    global LAST_COMPILED, LAST_CONFIG
    LAST_CONFIG = dict(cfg, BC_HEAD=bool(bc_head), BLOCK_L=BL, BLOCK_P=BLK_P, HPACK=HPACK, CKPACK=CKPACK, DSIG=DSIG, DBACC_ON=DBACC, NBLK=NBLK, grid=(H // HP * NP, Bs, cfg['NS']))

    def _stat(t):
        if t is None:
            return None
        return t.reshape(H * P, N).permute(1, 0).contiguous() if HPACK else t.permute(0, 2, 1).contiguous()
    if TS > 1:
        from .ts_adjoint import ts_seeds
        ts_seeds(TS, NBS * BL, x, delta, alog_e, C_t, gy, delta_bias=delta_bias, delta_softplus=delta_softplus, z=z, gh_final=gh_final, out=out['alam'], packed=HPACK)
    LAST_COMPILED = _mh_bwd_kernel[H // HP * NP, Bs, cfg['NS'] * TS](x.contiguous(), delta.contiguous(), _stat(alog_e), B_t.contiguous(), C_t.contiguous(), gy.contiguous(), None if theta is None else theta.contiguous(), _thh_table(theta_h), _f32(theta_init) if theta_init is not None else torch.zeros(Bs, H, K, device=x.device, dtype=torch.float32) if ROT else out['thck'], row_law, row_rot, None if a_dyn is None else a_dyn.contiguous(), None if mu is None else mu.contiguous(), _stat(_f32(mu_gain)), _f32(mu_init), _f32(delta_bias), _f32(beta_b), _f32(beta_c), None if delta_w is None else delta_w.contiguous(), None if g_B is None else g_B.contiguous(), None if g_C is None else g_C.contiguous(), _f32(beta_gB), _f32(beta_gC), out['x_s'], out['de_s'], out['c1_s'], out['hi_s'], out['Bs_'], out['Cs_'], out['gth_s'], out['gthh_s'], out['c2_s'], out['grate'], out['gs2o'], out['gdw_s'], out['ggB_s'], out['ggC_s'], out['hck'], out['alam'], out['thck'], out['rcar'], None if d_skip is None else d_skip.float().contiguous(), gd_skip, None if z is None else z, gz, endpoint_own, endpoint_grad, out['gdb_s'], out['gbb_s'], out['gbc_s'], out['gbgB_s'], out['gbgC_s'], Bs, H, P, N, L, K, NBLK, cks, BLOCK_L=BL, BLOCK_P=BLK_P, NUM_WARPS=NW_T, WP=WP, HPACK=HPACK, HP=HP, DSIG=DSIG, CKPACK=CKPACK, BC_HEAD=bool(bc_head), TP_P=cfg['TP_P'], HAS_DSKIP=d_skip is not None, DBACC=DBACC, HAS_Z=z is not None, HAS_ENDPOINT=endpoint_own is not None, ENDPOINT_P=endpoint_own is not None and endpoint_own.ndim == 4, ABL=int(os.environ.get('MH_BWD_ABL', cfg['ABL'])), RUN_P1=cfg['RUN_P1'], NS=cfg['NS'], NP=NP, TS=TS, ATOMIC_GX=atomic_gx, ATOMIC_BC=atomic_bc, BCS=bcs, LWS=law_slots if atomic_law else 0, LAW_VEC=_vecw(atomic_law and cfg['VBC']), ATOMIC_TH=atomic_th, THS=ths, TH_VEC=th_vec, BC_VEC=bc_vec, GX_VEC=gx_vec, TPL=tp_l, ROT=ROT, PMAP=bool(pmap), PRAW=pmap == 'raw', SLP=slp.active(_cc[0] * 10 + _cc[1]), VEC_V=vec_v, TPL_V=tpl_v, THETA_DELTA=bool(theta_delta) and theta is not None, THETA_PH=theta is not None and theta.dim() == 4, HAS_THH=theta_h is not None, RUN_TH=cfg['RUN_TH'], FAST_TRIG=bool(cfg['FAST_TRIG']), PAIR_UNROLL=bool(cfg.get('PU', 0)), LAW_MASK=LAW_MASK, DYN_ACT=int(dyn_act), DYN_EPS=float(dyn_eps), A_DYN_N_RES=A_DYN_N_RES, MU_N_RES=MU_N_RES, HAS_MU_INIT=mu_init is not None, MU_PREDIFF=bool(mu_prediff), DELTA_SOFTPLUS=bool(delta_softplus), HAS_DBIAS=delta_bias is not None, HAS_BB=beta_b is not None, HAS_BC=beta_c is not None, HG_BC=HG_BC, BACC_B=BACC_B, BACC_C=BACC_C, BACC_GB=BACC_GB, BACC_GC=BACC_GC, BGPART=bias_part, HAS_GHL=gh_final is not None, HAS_DW=delta_w is not None, DW_P=DW_P, N_OFF=n_off, N_ROWS=nr, ROT_MIX=ROT_MIX, HAS_GB=g_B is not None, HAS_GC=g_C is not None, GATE_ACT=int(gate_act), HAS_BGB=beta_gB is not None, HAS_BGC=beta_gC is not None, HG_G=HG_G, EVEN_L=False, num_warps=NW_T, maxnreg=cfg['MAXNREG'])
    if nr != N:
        for buf in (out['Bs_'], out['Cs_']) + ((out['ggB_s'],) if g_B is not None else ()) + ((out['ggC_s'],) if g_C is not None else ()):
            mf = 2 if pmap and buf.shape[2] == 2 * N else 1
            buf[:, :, :mf * n_off] = 0.0
            buf[:, :, mf * (n_off + nr):] = 0.0
        if ROT:
            out['rcar'][:, :, :n_off // 2] = 0.0
            out['rcar'][:, :, (n_off + nr) // 2:] = 0.0

    def _unstat(t):
        return t.permute(0, 2, 1).unsqueeze(2) if HPACK else t.permute(0, 1, 3, 2)
    out['c1'] = _unstat(out['c1_s'].sum(0) if TS > 1 else out['c1_s'])
    out['h_init'] = _unstat(out['hi_s'])
    if h_init is not None and gh_final is not None and (nr != N):
        out['h_init'][..., :n_off] = gh_final[..., :n_off].float()
        out['h_init'][..., n_off + nr:] = gh_final[..., n_off + nr:].float()
    out['x'] = _hsum(out['x_s'])
    if d_skip is not None:
        out['d_skip'] = gd_skip.reshape(d_skip.shape)
        if mem is not None and mem.get('BCS') == 0:
            gyf = gy.float()
            if z is not None:
                zf = z.float()
                gyf = gyf * (zf * torch.sigmoid(zf))
            out['d_skip'] = (gyf * x.float()).sum(dim=(0, 2, 3)).reshape(d_skip.shape)
    if z is not None:
        out['z'] = gz
    gde = _hsum(out['de_s'])
    if delta_softplus and (not DSIG):
        d_raw = delta.float() if delta_bias is None else delta.float() + _f32(delta_bias).view(1, H, 1)
        gde = gde * torch.sigmoid(d_raw)
    out['delta'] = gde
    if delta_bias is not None:
        out['delta_bias'] = out['gdb_s'].sum((0, 1)) if DBACC else gde.sum((0, 2))
    if delta_w is not None:
        gdw = _hsum(out['gdw_s'])
        if delta_softplus:
            gdw = gdw * torch.sigmoid(delta_w.float())
        out['delta_w'] = gdw
    gc1_h = out['c1'].sum(0)
    zc = torch.zeros_like(gc1_h)
    lawv = None if row_law is None else row_law.view(1, 1, N)
    if mu_gain is not None:
        out['mu_gain'] = torch.where(lawv == LAW_MU, _unstat(out['c2_s']).sum(0), zc) if HAS_MU else zc
    if a_dyn is not None:
        gr_ = _hsum(out['grate']) if HAS_DYN else torch.zeros_like(a_dyn, dtype=torch.float32)
        out['a_dyn'] = gr_ * _dact(a_dyn.float(), dyn_act)
    if mu is not None:
        gs = _hsum(out['gs2o']) if HAS_MU else torch.zeros_like(mu, dtype=torch.float32)
        if mu_prediff:
            out['mu'] = gs
        else:
            out['mu'] = gs - torch.cat([gs[..., 1:], torch.zeros_like(gs[..., :1])], dim=-1)
            if mu_init is not None:
                out['mu_init'] = -gs[..., 0]
    gal = gc1_h * -torch.exp(alog_e)
    out['A_log'] = gal
    if beta_a is not None:
        G = beta_a.shape[0]
        gba = gal.sum(1)
        out['beta_a'] = gba.reshape(G, H // G, N).sum(1)
    if cfg.get('REDUCE', True):
        if bc_head:
            out['B_t'] = out['Bs_'].reshape(H, NP, Bs, N, L).sum(1) if bcs > H else out['Bs_']
            out['C_t'] = out['Cs_'].reshape(H, NP, Bs, N, L).sum(1) if bcs > H else out['Cs_']
        else:
            out['B_t'] = _hsum(out['Bs_'])
            out['C_t'] = _hsum(out['Cs_'])

        def _beta(slot, folded, acc, G, on_acc):
            if on_acc:
                return acc
            if slot.shape[0] == 1:
                return folded.sum(-1).sum(0).reshape(1, N)
            per_g = slot.shape[0] // NP // G
            return _pfold(slot.sum(-1).sum(1)).reshape(G, per_g, N).sum(1)
        if beta_b is not None:
            out['beta_b'] = _beta(out['Bs_'], out['B_t'], out['gbb_s'], G_BC, BACC_B)
        if beta_c is not None:
            out['beta_c'] = _beta(out['Cs_'], out['C_t'], out['gbc_s'], G_BC, BACC_C)
        for sk, gk, gt, bk, bt, ak, on in (('ggB_s', 'g_B', g_B, 'beta_gB', beta_gB, 'gbgB_s', BACC_GB), ('ggC_s', 'g_C', g_C, 'beta_gC', beta_gC, 'gbgC_s', BACC_GC)):
            if gt is None:
                if bt is not None:
                    out[bk] = torch.zeros_like(bt, dtype=torch.float32)
                continue
            out[gk] = _hsum(out[sk])
            if bt is not None:
                out[bk] = out[ak].reshape(G_G, -1, N).sum(1) if bias_part and on else _beta(out[sk], out[gk], out[ak], G_G, on)
        if ROT:
            if theta.dim() == 4:
                gs = out['gth_s']
                out['theta'] = gs.reshape(H, -1, *gs.shape[1:]).sum(1).permute(1, 0, 2, 3)
            else:
                out['theta'] = _hsum(out['gth_s'])
            if theta_h is not None:
                out['theta_h'] = _pfold(out['gthh_s'].sum((1, 3)))
            out['theta_init'] = out['rcar'][..., 0] if NP == 1 else out['rcar'][..., ::P // NP].sum(-1)
        else:
            for k_, t_ in (('theta', theta), ('theta_init', theta_init), ('theta_h', theta_h)):
                if t_ is not None:
                    out[k_] = torch.zeros_like(t_, dtype=torch.float32)
    if endpoint_own is not None:
        out['endpoint_own'] = _hsum(endpoint_grad)
    return out
