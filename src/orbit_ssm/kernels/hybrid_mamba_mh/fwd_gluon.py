"""Transposed-tile forward: p is the tile axis, n the sequential loop.

WHY THIS LAYOUT
---------------
Tiling [BLOCK_N, BLOCK_L] with the warps on N makes the readout
`y[p,t] = sum_n C[n,t]*h[p,n,t]` a cross-warp reduce run NBLK*BLOCK_P times per
CTA, and one fp32 tile is then so large (BLOCK_N=128, BLOCK_L=64, 4 warps:
64 registers per thread, ~7 live) that the kernel spills and runs at ~17%
occupancy. Here the tile is [BLOCK_P, BLOCK_L] with n sequential -- the same
transposition the backward uses:

  * `y = sum_n C*h` becomes `yacc += hs * C[n]`, a register accumulation with
    no collective at all.
  * `x` and `delta_w` are n-free, so they are loaded ONCE per chunk instead of
    once per (chunk, p).
  * live 2-D tiles are ~4 (xw, a, hs, yacc), each
    BLOCK_P*BLOCK_L/(32*NUM_WARPS) registers.

The p-free sharing is kept: B/C, the gates and s1/s2 are computed once per
(b,h,n,t) as [BLOCK_L] vectors and broadcast across the p tile axis for free.
Total p-free element count is N*L per CTA.

Chunk is the OUTER loop and n the inner one, as in the backward. n outer
would keep the h carry in registers but would force a [BLOCK_P, L] Y
accumulator; chunk outer keeps Y to one [BLOCK_P, BLOCK_L] tile flushed per
chunk and pays a [BLOCK_P] state carry per (chunk, n) through global memory.

Every per-(chunk, n) buffer is therefore laid out p-FASTEST -- A_log
and mu_gain as (H, N, P) and the h carry as (B, H, N, P). In the contract's
(.., P, N) order a [BLOCK_P] access strides by N and costs 32 sectors instead
of one. The launcher permutes on the host; h_final is returned as a view.

INSTRUCTION STREAM
------------------
  * a [BLOCK_L] vector is broadcast onto the tile by plain broadcasting, never
    by ADDING a zero tile: `x + 0.0 -> x` is not a legal fold in the presence
    of -0.0, so each such add survives to SASS.
  * when BLOCK_L divides L the tail predicates (load masks, their `other`
    selects, the `where`s that re-zero past L) select between a value and
    itself. EVEN_L removes them; `_ldt`/`_zt` keep the two branches
    type-identical.

REGISTERS VS RESIDENCY
----------------------
At ~96 registers residency is 65536/(32*96) = 20 single-warp CTAs per SM, so
every extra register costs warps. B/C prefetch (PF), unrolling n by 2 and
hoisting A_log/exp out of the chunk loop all attack the L1TEX long scoreboard
but raise the register count; they are tuned axes, not defaults. The remaining
gap on small grids is grid size (waves), not the loop body; splitting n across
CTAs does not fit in the residency that registers already cap.

ROT
---
Theta, cos/sin and the phase carry do not depend on p, and this layout is the
hostile one for that: with BLOCK_P=1 the grid is P CTAs per (b, h) and each of
them redoes the whole angle for all K pairs. The redundancy is invisible inside
one CTA. The cost of a rotating launch splits into the phase scan, the
transcendentals (the largest part) and the partner B/C loads; the partner loads
are hoisted above the `pi` loop so each pair fetches its two rows once, and
FAST_TRIG takes the SFU approximations (checked against the fp64 oracle at the
2e-5 bar).

What the nesting does cost irreducibly is the phase carry. n is sequential, so
there is no tile axis to hold a per-pair phase across chunks and it goes to
scratch -- (B, H, P/BLOCK_P, K), one copy per p-CTA because they would
otherwise race. The n window must also be pair-aligned, since the partner row
is fetched by index rather than sliced out of a tile.

ROT at BLOCK_P >= 2 (TPW_L=32)
------------------------------
There the cost is registers, not the angle: the pair loop spills at large
BLOCK_P/BLOCK_L. LLVM already merges the p-replicated phase-scan rows (the
source's [BLOCK_P, BLOCK_L] scan survives to SASS as one row), so the angle
chain is a small part of the loop. The loop is arranged as:

  * the phase prefix runs on ONE row (`_row1`): a SliceLayout scan does not
    lower, and a [1, BLOCK_L] tensor whose p lanes broadcast converts to and
    from `lt` for free -- no replicated tile, no reduce back over p;
  * theta_h's (cos, sin) come from a host table (`_thh_table`): the precise
    per-(chunk, pair) cos/sin had range-reduction branches in the loop that
    kept ptxas from hoisting the pair's loads;
  * the endpoint diagonal (own B.C x) is contracted over n once per chunk
    (`bcac`, sum_n B_n C_n) and x is re-read at the store, so no x tile is
    live across the n loop (the real arm drops from 255 to 168 registers at
    BP4/BL256);
  * PF under ROT prefetches A_log and the pair's angle one pair ahead;
  * PMAP: a GATED rotating launch reads per-row pair maps (see the kernel's
    PMAP note) -- no in-kernel activation or gate multiply, two loads per
    row, nothing live across the pair.  Together these keep the rotating
    loop at BP4/BL128 within 168 registers without spill.
"""
from __future__ import annotations

import os

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.extra import libdevice

from . import autotune, slp
from ... import kernel_options as _ko
from .common_gluon import (
    INV_TWO_PI, K_DYN, K_MU, LOG2E, TWO_PI,
    _add, _cmul, _divisor_le, _f32, _gate, _rate, _row1, _silu_fast,
    _softplus, _ssm_combine, _ssm_combine_q,
    selector_window,
)
from .layout import (
    DYN_SOFTPLUS, GATE_SIGMOID, LAW_A_DYN, LAW_A_FIX, LAW_MU,
    validate_forward_inputs,
)


@gluon.constexpr_function
def _lay(BLOCK_P, BLOCK_L, NUM_WARPS, TPW_L):
    """[BLOCK_P, BLOCK_L] with every warp on P.

    `warps_per_cta[1] == 1` is the load-bearing part: the time scan and the
    `sum_t` that extracts the chunk-end state then never leave a warp. This is
    the mirror image of _lay3 in bwd_gluon, which puts the warps on time
    because there the contraction is over p.

    The warps stay off time (wl is not derived): warps_per_cta=[1, wl] needs
    spt_l divided by wl to cover the tile, and even then the cross-warp scan
    loses on most shapes and only ties the best single-warp config at long L.

    spt_l = BLOCK_L/tp_l is the run each thread scans sequentially while the
    warp part of the scan costs a fixed log2(tp_l) shuffle rounds, so a small
    tp_l is what makes the scan cheap (this is the TPW_L knob). Small BLOCK_P forces tp_l up, so the
    launcher raises BLOCK_L to keep spt_l where it belongs.
    """
    tp_p = min(32, max(BLOCK_P // NUM_WARPS, 1))
    if TPW_L > 0:
        tp_p = min(tp_p, max(32 // TPW_L, 1))
    tp_p = 1 << (tp_p.bit_length() - 1)          # floor to a power of two
    tp_l = max(32 // tp_p, 1)
    spt_p = max(BLOCK_P // (tp_p * NUM_WARPS), 1)
    spt_l = max(BLOCK_L // tp_l, 1)
    return gl.BlockedLayout([spt_p, spt_l], [tp_p, tp_l], [NUM_WARPS, 1],
                            [1, 0])


@gluon.jit
def _ldt(ptr, mask, EVEN: gl.constexpr):
    """A global load whose tail predicate exists only when there IS a tail.

    BLOCK_L divides L on every shape this kernel ships for, and then `mask` is
    all ones: the predicate, the `other` select and the `where` that re-zeroes
    the tail are dead weight on the hottest path in the file. Structure A
    spends a template parameter (`EVEN`) on exactly this. These are functions
    rather than inline `if`s so that both constexpr branches obviously produce
    the same type and layout.
    """
    if EVEN:
        v = gl.load(ptr)
    else:
        v = gl.load(ptr, mask=mask, other=0.0)
    return v


@gluon.jit
def _zt(v, mask, EVEN: gl.constexpr):
    """Zero past the end of the sequence, or nothing if there is no end."""
    if EVEN:
        w = v
    else:
        w = gl.where(mask, v, 0.0)
    return w


@gluon.jit
def _mh_fwd_kernel(
    x_ptr, delta_ptr, dw_ptr, A_log_ptr, mu_gain_ptr,
    B_ptr, C_ptr, gB_ptr, gC_ptr, a_dyn_ptr, mu_ptr, mu_init_ptr,
    beta_a_ptr, beta_b_ptr, beta_c_ptr, beta_gB_ptr, beta_gC_ptr,
    delta_bias_ptr, row_law_ptr, row_rot_ptr, theta_ptr, thh_ptr,
    Y_ptr, h_state_ptr, th_state_ptr, ckpt_ptr,
    d_skip_ptr, z_ptr, endpoint_ptr,
    H, P, N, L, K, N_OFF, N_ROWS, CKS,
    LAW_MASK: gl.constexpr, HAS_ENDPOINT: gl.constexpr, ENDPOINT_P: gl.constexpr,
    ROT: gl.constexpr, HAS_RROT: gl.constexpr,
    # SSD law: the per-token increment is theta[k,t] * Delta[h,t]
    # (THETA_DELTA) and the READ map rotates by Theta + theta_h[h,k] (HAS_THH)
    THETA_DELTA: gl.constexpr, HAS_THH: gl.constexpr, THETA_PH: gl.constexpr,
    DELTA_SOFTPLUS: gl.constexpr, DYN_ACT: gl.constexpr, GATE_ACT: gl.constexpr,
    A_DYN_N_RES: gl.constexpr, MU_N_RES: gl.constexpr,
    HAS_DW: gl.constexpr, DW_PER_P: gl.constexpr,
    HAS_GB: gl.constexpr, HAS_GC: gl.constexpr,
    HAS_BA: gl.constexpr, HAS_BB: gl.constexpr, HAS_BC: gl.constexpr,
    HAS_BGB: gl.constexpr, HAS_BGC: gl.constexpr,
    HAS_DBIAS: gl.constexpr, HAS_MU_INIT: gl.constexpr,
    MU_PREDIFF: gl.constexpr,
    HG_A: gl.constexpr, HG_BC: gl.constexpr, HG_G: gl.constexpr,
    DYN_EPS: gl.constexpr,
    BLOCK_L: gl.constexpr, BLOCK_P: gl.constexpr,
    NUM_WARPS: gl.constexpr, TPW_L: gl.constexpr, ABL: gl.constexpr = 0,
    CKPT_L: gl.constexpr = 0, CKPACK: gl.constexpr = False,
    CKPT_COMPACT: gl.constexpr = False,
    BC_HEAD: gl.constexpr = False,
    FAST_TRIG: gl.constexpr = False,
    EVEN_L: gl.constexpr = False, PF: gl.constexpr = 0,
    # ── K11: the D skip, at the STORE ────────────────────────────────
    # `y = SSM(x) + D_h x` is fused here instead of a separate elementwise
    # pass.  The raw `xt` tile is n-free and already in registers, so
    # the skip is one fused multiply-add on `yacc` and one scalar load.
    # `mh_scan(d_skip=...)` is refused when the call is an n-SHARD
    # (`n_off`/`n_rows`), because a shard's Y is one summand of the
    # readout and the skip must be added exactly once.
    HAS_DSKIP: gl.constexpr = False,
    # ── H3 item 1: the SwiGLU OUTPUT gate, at the STORE ──────────────
    # `y = (SSM(x) + D_h x) * silu(z)`, fused so the norm-free epilogue
    # needs no separate launch.  `yacc` is already in registers
    # here, so the gate costs ONE activation-sized LOAD and two FLOPs on a
    # kernel that is nowhere near bandwidth -- the read hides under the
    # scan.  `z` is CHANNEL-major, (B, H, P, L), i.e. `po` addresses it
    # exactly as it addresses `x` and `Y`; a token-major `z` at P == 1
    # would be a stride-D gather per element and is refused by the
    # launcher rather than served slowly.
    #
    # AFTER the D skip, which is the order `selective_scan_fn` uses
    # (`out_z = (out + D u) * silu(z)`).
    HAS_Z: gl.constexpr = False,
    # ── PMAP: gated rotating launches read PAIR MAPS ─────────────────
    # B_ptr/C_ptr are (B, 2N, L): map rows 2n, 2n+1 hold row n's gate times
    # the RAW pair of its rotation partner, act(g_n) * (B_2k, B_2k+1)
    # (`functions._pair_maps`).  Then diag(g) R(Theta) B is a 2x2 on two
    # loads per row -- same load count as B + partner + gate, no activation,
    # no gate multiply, and nothing live across the pair.  Biases and the
    # gate are folded on the host, so HAS_GB/HAS_GC/HAS_BB/HAS_BC are off.
    PMAP: gl.constexpr = False,
    # compiled with LLVM SLP forced on (`slp.py`); in the signature only so
    # the two builds are different specialisations
    SLP: gl.constexpr = False,
    # FPACK: at P == 1 the tile's rows are BLOCK_P consecutive
    # HEADS (channels) instead of p, so the head-free B/C rows are loaded
    # once per BLOCK_P channels instead of once per channel -- the
    # backward's HPACK, forward side.  delta, its bias, A_log, the carry and
    # D become per-row.  Envelope (launcher `_fpack_ok`): LAW_A, real, no
    # B/C biases / gates / delta_w / beta_a / endpoint / per-head B/C.
    FPACK: gl.constexpr = False,
    # TS: the time split (`ts_forward`).  Grid dim 2 is B * TS;
    # program (b, ts) walks super-chunk ts only, seeded from carry slab ts
    # of a (TS, B, H, N, P) `h_state` that `ts_fwd_seeds` filled.
    TS: gl.constexpr = 1,
):
    HAS_DYN: gl.constexpr = (LAW_MASK >> 1) & 1
    HAS_MU: gl.constexpr = (LAW_MASK >> 2) & 1
    # one bit set = the whole tensor is on one law, and then the runtime test
    # is dead weight: the branches below fold at trace time. Python's `or`
    # short-circuits on the constexpr, so the predicate never becomes a Gluon
    # value and no branch is emitted.
    UNI: gl.constexpr = (LAW_MASK & (LAW_MASK - 1)) == 0
    # PF runs B/C a row ahead; PFA does the same for A_log.  Every law left
    # in this kernel reads A_log (the per-head-scalar `const` law routes to
    # `ssd_mimo_scan`), so the prefetch is always the load that is taken.
    PFA: gl.constexpr = PF
    # the n loop steps by pairs under ROT and by single rows otherwise, so one
    # body serves both; NPAIR=1 makes `kk` the old `n` exactly.
    NPAIR: gl.constexpr = 2 if ROT else 1
    # Under the one rotation law (the SSD law) every rotating launch
    # carries a per-token `theta`, so `ROT` implies a per-token angle and the
    # phase is always the cumulative sum below.
    INV_BP: gl.constexpr = 1.0 / BLOCK_P

    tile: gl.constexpr = _lay(BLOCK_P, BLOCK_L, NUM_WARPS, TPW_L)
    lt: gl.constexpr = gl.SliceLayout(0, tile)
    pl: gl.constexpr = gl.SliceLayout(1, tile)
    row1: gl.constexpr = _row1(tile)
    if CKPT_COMPACT:
        cl: gl.constexpr = gl.BlockedLayout(
            tile.size_per_thread, tile.threads_per_warp,
            tile.warps_per_cta, [0, 1])

    pid_p = gl.program_id(0)
    h = gl.program_id(1)
    if FPACK:
        h = h * BLOCK_P               # first head of the pack
    if TS > 1:
        b = (gl.program_id(2) // TS).to(gl.int64)
        ts = gl.program_id(2) % TS
        nbs = ((L + BLOCK_L - 1) // BLOCK_L + TS - 1) // TS   # chunks per super-chunk
        t_lo = ts * nbs * BLOCK_L
        t_hi = gl.minimum(t_lo + nbs * BLOCK_L, L)
        # this super-chunk's carry slab
        h_state_ptr += ts.to(gl.int64) * (gl.num_programs(2) // TS) * H * N * P
    else:
        b = gl.program_id(2).to(gl.int64)
        t_lo = 0
        t_hi = L
    bh = b * H + h
    # the angle feature's row base: (B,K,L) head-shared, or (B,H,K,L) per
    # head under THETA_PH (theta_per_head)
    thb = (bh if THETA_PH else b) * K
    if BC_HEAD:
        # Prepared maps are (H,B,N,L); the real recurrence is unchanged.
        B_ptr += h.to(gl.int64) * gl.num_programs(2) * N * L
        C_ptr += h.to(gl.int64) * gl.num_programs(2) * N * L
    ga = h // HG_A
    gbc = h // HG_BC
    # Each static tensor declares its OWN G|H, so the gate biases get
    # their own index: beta_gB at G=1 read through gbc runs off the end.
    gg = h // HG_G

    z2 = gl.zeros([BLOCK_P, BLOCK_L], gl.float32, layout=tile)
    zl = gl.zeros([BLOCK_L], gl.int32, layout=lt)
    zp = gl.zeros([BLOCK_P], gl.int32, layout=pl)
    col = gl.arange(0, BLOCK_L, layout=lt)
    first2 = gl.expand_dims(col == 0, 0) + gl.zeros([BLOCK_P, BLOCK_L],
                                                    gl.int1, layout=tile)
    last2 = gl.expand_dims(col == BLOCK_L - 1, 0) + gl.zeros(
        [BLOCK_P, BLOCK_L], gl.int1, layout=tile)
    idx_prv = gl.maximum(col - 1, 0)

    pf = (pid_p * BLOCK_P + gl.arange(0, BLOCK_P, layout=pl)).to(gl.int64)
    hp = bh * P + pf                # x, Y as (B, H, P, L)
    if FPACK:
        # rows are heads (P == 1): row r is head h + r
        anp = (h.to(gl.int64) + pf) * N * P
        hnp = (bh + pf) * N * P
        rowL = gl.expand_dims((bh + pf) * L, 1)        # (B,H,L) row base
    else:
        anp = h.to(gl.int64) * N * P + pf   # A_log, mu_gain as (H, N, P)
        hnp = bh * N * P + pf               # h carry as (B, H, N, P)
    # Checkpoints are (B, H, CKS, N, P) normally and (B, CKS, N, D) under
    # CKPACK -- the layout the backward's PACKED small-P schedule reads, whose
    # tile axis spans HEADS and so needs the channel fastest (K14; see
    # `bwd_gluon._mh_bwd_kernel`'s "PACKED SMALL-P SCHEDULE").  CKPACK is a
    # property of P alone (it is only ever taken at P == 1, where a head IS a
    # channel), NOT of the backward's pack width, so the two sides agree
    # without the forward having to know what the backward will choose.
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
        m2 = gl.expand_dims(mask_t, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1,
                                                  layout=tile)

        # ---------- p-free AND n-free: once per chunk --------------------
        if FPACK:
            d_raw2 = _ldt(delta_ptr + rowL + gl.expand_dims(offs_t, 0),
                          m2, EVEN_L).to(gl.float32)
            if HAS_DBIAS:
                d_raw2 = d_raw2 + d_bias2
            if DELTA_SOFTPLUS:
                de2 = _softplus(d_raw2)
            else:
                de2 = d_raw2
            de2 = _zt(de2, m2, EVEN_L)
            de = gl.sum(de2, axis=0)        # unused under the envelope; keeps `de` defined
        else:
            d_raw = _ldt(delta_ptr + bh * L + offs_t,
                         mask_t, EVEN_L).to(gl.float32)
            if HAS_DBIAS:
                d_raw = d_raw + d_bias
            if DELTA_SOFTPLUS:
                de = _softplus(d_raw)
            else:
                de = d_raw
            de = _zt(de, mask_t, EVEN_L)

        po = gl.expand_dims(hp, 1) * L + gl.expand_dims(offs_t, 0)
        # n-free, so loaded ONCE per chunk
        xt = _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        if HAS_Z:
            # Issued HERE, not at the store: this kernel runs ONE warp per
            # CTA, so a load whose only consumer is the last instruction
            # has nothing to overlap its latency with.  Issued before the
            # n loop it hides under the scan, at the cost of one live
            # [BLOCK_P, BLOCK_L] tile.
            zg = _silu_fast(_ldt(z_ptr + po, m2, EVEN_L).to(gl.float32))
        if HAS_DW:
            if DW_PER_P:
                dwr = _ldt(dw_ptr + po, m2, EVEN_L).to(gl.float32)
            else:
                dwr = _ldt(dw_ptr + bh * L + offs_t,
                           mask_t, EVEN_L).to(gl.float32)
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

        # h-resolved law drivers address by (b, h, t), so they are the same
        # vector for every row: reading them here instead of in the n loop
        # turns N loads and N transcendentals per chunk into one.
        if HAS_DYN and not A_DYN_N_RES:
            rate_h = _rate(_ldt(a_dyn_ptr + bh * L + offs_t,
                                mask_t, EVEN_L).to(gl.float32),
                           DYN_ACT, DYN_EPS)
        if HAS_MU and not MU_N_RES:
            mu_h = _ldt(mu_ptr + bh * L + offs_t,
                        mask_t, EVEN_L).to(gl.float32)
            if MU_PREDIFF:
                s2_h = gl.expand_dims(_zt(mu_h, mask_t, EVEN_L), 0)
            else:
                mu_ph = gl.load(mu_ptr + bh * L + offs_t - 1,
                                mask=mask_t & (offs_t > 0), other=0.0
                                ).to(gl.float32)
                if HAS_MU_INIT:
                    mu_ph = gl.where(offs_t == 0,
                                     gl.load(mu_init_ptr + bh + zl), mu_ph)
                s2_h = gl.expand_dims(_zt(mu_h - mu_ph, mask_t, EVEN_L), 0)

        yacc = z2
        if HAS_ENDPOINT:
            bcac = gl.zeros([BLOCK_L], gl.float32, layout=lt)

        # ---- B/C one n ahead, so the load overlaps the previous row's scan --
        # Whether the extra registers pay depends on residency, so PF is a
        # tuned axis rather than a default.
        #
        # Primed per chunk, not per kernel: the prefetched vectors are indexed
        # by `offs_t`, which moves with the chunk, so a carry across the L loop
        # would hand row n the previous chunk's columns.
        if PF and not ROT:
            bcp = (b * N + N_OFF).to(gl.int64) * L
            Bnx = _ldt(B_ptr + bcp + offs_t, mask_t, EVEN_L).to(gl.float32)
            Cnx = _ldt(C_ptr + bcp + offs_t, mask_t, EVEN_L).to(gl.float32)
        # ONE row ahead is enough: ptxas already hoists the depth-1 load as
        # far up the body as the dependences allow, so a deeper prefetch buys
        # nothing.
        # A_log gets the same treatment, and is the better candidate of the
        # two on paper: it is [BLOCK_P] rather than [BLOCK_L], so running one
        # row ahead costs a handful of registers, and it sits at the HEAD of
        # load -> exp -> mul -> exp2 -> scan, where nothing downstream can be
        # issued until it lands. It is also chunk-invariant -- only N distinct
        # values exist per CTA and the loop re-reads them once per chunk -- so
        # the cost being priced here is latency, not traffic.
        if PFA:
            Anx = gl.load(A_log_ptr + anp + N_OFF * P)
        # Under ROT, PF runs the pair's angle feature one pair ahead instead
        # of B/C (whose pair loads sit above the `pi` loop, see below): it is
        # the head of theta -> scan -> cos/sin -> rotate, and otherwise
        # consumed a few instructions after it is issued.
        if ROT and PF:
            thn = _ldt(theta_ptr + (thb + N_OFF // NPAIR) * L + offs_t,
                       mask_t, EVEN_L)
        if PMAP and PF:
            # the pair maps one ROW ahead, kept in the input dtype until use
            mrn = (b * 2 * N + 2 * N_OFF).to(gl.int64) * L
            Pn0 = _ldt(B_ptr + mrn + offs_t, mask_t, EVEN_L)
            Pn1 = _ldt(B_ptr + (mrn + L) + offs_t, mask_t, EVEN_L)
            Qn0 = _ldt(C_ptr + mrn + offs_t, mask_t, EVEN_L)
            Qn1 = _ldt(C_ptr + (mrn + L) + offs_t, mask_t, EVEN_L)

        # ---------------- n: sequential, in registers --------------------
        for kk in range(N_OFF // NPAIR, (N_OFF + N_ROWS) // NPAIR):
            # ---- rotation phase, shared by both rows of the pair ---------
            # p-free: one [BLOCK_L] vector per thread-row of `lt` (see the
            # header), cos/sin once per element.
            if ROT:
                thp = th_state_ptr \
                    + (bh * gl.num_programs(0) + pid_p).to(gl.int64) * K + kk
                if PF:
                    thv = thn.to(gl.float32)
                    thn = _ldt(theta_ptr + (thb + gl.minimum(
                        kk + 1, (N_OFF + N_ROWS) // NPAIR - 1).to(gl.int64))
                        * L + offs_t, mask_t, EVEN_L)
                else:
                    thv = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L
                               + offs_t, mask_t, EVEN_L).to(gl.float32)
                if THETA_DELTA:
                    # SSD law: inc_t[h,k] = Delta_t[h] * (omega_k tanh q_t[k])
                    thv = thv * de
                # Theta_t = carry + sum_{s<=t} theta_s: the increment is
                # additive INSIDE the cumulative sum.
                # The scan runs on ONE row: a [1, BLOCK_L] tensor in `row1`,
                # whose p lanes/warps broadcast, so its slice IS `lt` and both
                # conversions are free. Running it on the p-replicated
                # [BLOCK_P, BLOCK_L] tile would cost an extra live tile and
                # BLOCK_P-1 adds per element at the register ceiling.
                inc1 = gl.expand_dims(gl.convert_layout(
                    _zt(thv, mask_t, EVEN_L), gl.SliceLayout(0, row1)), 0)
                if not (ABL & 64):
                    inc1 = gl.associative_scan(inc1, axis=1, combine_fn=_add)
                Th = gl.convert_layout(gl.sum(inc1, axis=0), lt) \
                    + gl.load(thp)
                if ABL & 256:
                    cT, sT = Th, Th
                elif FAST_TRIG:
                    cT = libdevice.fast_cosf(Th)
                    sT = libdevice.fast_sinf(Th)
                else:
                    cT = gl.cos(Th)
                    sT = gl.sin(Th)
                # wrap before the carry leaves the chunk: cos/sin lose their
                # last bits once Theta reaches ~1e7, and a long sequence on a
                # modest rate gets there.
                tc = gl.sum(gl.where(col == BLOCK_L - 1, Th, 0.0), axis=0)
                if NUM_WARPS > 1:
                    # Every warp reads the carry above and thread 0 rewrites
                    # it below (warps sit on P, so nothing in between crosses
                    # warps).  Barrier 1: every warp's read of chunk s lands
                    # before the rewrite (else a fast warp's store reached a
                    # slow warp's load -- a run-to-run race in Y, seen by the
                    # checkpoint-store test at wider tiles, BW,
                    # BLOCK_P 4-8 x 2 warps).  Barrier 2 (below): the rewrite
                    # lands before any warp's read at chunk s+1 -- with ONE
                    # pair in the window no other barrier sits in between.
                    # Multi-warp configs only.
                    gl.barrier()
                gl.store(thp, tc - TWO_PI * libdevice.floor(tc * INV_TWO_PI))
                if NUM_WARPS > 1:
                    gl.barrier()
            if ROT:
                if HAS_THH:
                    # read map rotates by Theta + theta_h[h,k] (angle addition)
                    # (cos, sin) of the static offset come from the host's
                    # (H, K, 2) table: evaluated here they were a precise
                    # cos/sin per (chunk, pair) whose range-reduction branches
                    # kept ptxas from hoisting the pair's loads above them.
                    cn_h = gl.load(thh_ptr + (h * K + kk) * 2)
                    sn_h = gl.load(thh_ptr + (h * K + kk) * 2 + 1)
                    cC = cT * cn_h - sT * sn_h
                    sC = sT * cn_h + cT * sn_h
                else:
                    cC = cT
                    sC = sT

            # ---- the pair's raw B/C, loaded ONCE for both rows -------------
            # R(Theta) mixes rows 2k and 2k+1, so each row needs its partner.
            # Fetching the partner inside the `pi` loop reads the same two rows
            # twice, and a rotation-only diagnostic charged that 13-24% of
            # the rotation in
            # bf16 and 47-64% in fp32, where B/C are the wide loads. Hoisted,
            # ROT costs the same B/C traffic as the real path.
            if ROT and not PMAP:
                p0 = (b * N + kk * 2).to(gl.int64) * L
                B0 = _ldt(B_ptr + p0 + offs_t, mask_t, EVEN_L).to(gl.float32)
                C0 = _ldt(C_ptr + p0 + offs_t, mask_t, EVEN_L).to(gl.float32)
                if ABL & 128:
                    B1, C1 = B0, C0
                else:
                    B1 = _ldt(B_ptr + (p0 + L) + offs_t,
                              mask_t, EVEN_L).to(gl.float32)
                    C1 = _ldt(C_ptr + (p0 + L) + offs_t,
                              mask_t, EVEN_L).to(gl.float32)
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
                # SCALAR row base, and `offs_t` stays 32-bit. Folding offs_t
                # into a 64-bit `bc` tensor first made every one of B, C, gB,
                # gC, a_dyn and mu pay a per-lane 64-bit add (two IADD3s per
                # element, 16 elements per thread) on top of the widening;
                # against a uniform base ptxas contracts the whole address into
                # one IMAD.WIDE per element.
                bc = (b * N + n).to(gl.int64) * L
                if not UNI:
                    # a SCALAR, not a per-lane value: n is the sequential axis,
                    # so every lane in the CTA is on the same row and the law
                    # predicate is uniform. A row that is not MU never issues
                    # the two mu loads at all -- which is the difference
                    # between a mixture costing the sum of every law's
                    # overhead and costing their average.
                    law_s = gl.load(row_law_ptr + n)

                # ---- R(Theta), after the betas and before the gates -------
                # sgn is -1 on the even row of the
                # pair and +1 on the odd one.
                if PMAP:
                    if PF:
                        P0, P1 = Pn0.to(gl.float32), Pn1.to(gl.float32)
                        Q0, Q1 = Qn0.to(gl.float32), Qn1.to(gl.float32)
                        mrn = (b * 2 * N + 2 * gl.minimum(
                            n + 1, N_OFF + N_ROWS - 1)).to(gl.int64) * L
                        Pn0 = _ldt(B_ptr + mrn + offs_t, mask_t, EVEN_L)
                        Pn1 = _ldt(B_ptr + (mrn + L) + offs_t, mask_t, EVEN_L)
                        Qn0 = _ldt(C_ptr + mrn + offs_t, mask_t, EVEN_L)
                        Qn1 = _ldt(C_ptr + (mrn + L) + offs_t, mask_t, EVEN_L)
                    else:
                        mr = (b * 2 * N + 2 * n).to(gl.int64) * L
                        P0 = _ldt(B_ptr + mr + offs_t, mask_t,
                                  EVEN_L).to(gl.float32)
                        P1 = _ldt(B_ptr + (mr + L) + offs_t, mask_t,
                                  EVEN_L).to(gl.float32)
                        Q0 = _ldt(C_ptr + mr + offs_t, mask_t,
                                  EVEN_L).to(gl.float32)
                        Q1 = _ldt(C_ptr + (mr + L) + offs_t, mask_t,
                                  EVEN_L).to(gl.float32)
                    if pi == 0:
                        Bt, Ct = P0 * cT - sT * P1, Q0 * cC - sC * Q1
                    else:
                        Bt, Ct = P1 * cT + sT * P0, Q1 * cC + sC * Q0
                    if HAS_RROT:
                        rr = gl.load(row_rot_ptr + n + zl) != 0
                        Bt = gl.where(rr, Bt, P0 if pi == 0 else P1)
                        Ct = gl.where(rr, Ct, Q0 if pi == 0 else Q1)
                elif ROT:
                    if pi == 0:
                        Bt, Ct = B0 * cT - sT * B1, C0 * cC - sC * C1
                    else:
                        Bt, Ct = B1 * cT + sT * B0, C1 * cC + sC * C0
                    if HAS_RROT:
                        rr = gl.load(row_rot_ptr + n + zl) != 0
                        Bt = gl.where(rr, Bt, B0 if pi == 0 else B1)
                        Ct = gl.where(rr, Ct, C0 if pi == 0 else C1)
                elif ABL & 16:
                    Bt = de
                    Ct = de
                elif PF:
                    # take the row that is already in flight, then issue the
                    # next one BEFORE the exp/scan chain that consumes this
                    # one. The last row re-reads itself rather than branching:
                    # one redundant load per (chunk, n-window) against a
                    # divergent tail.
                    Bt, Ct = Bnx, Cnx
                    bcn = (b * N + gl.minimum(n + 1, N_OFF + N_ROWS - 1)
                           ).to(gl.int64) * L
                    Bnx = _ldt(B_ptr + bcn + offs_t,
                               mask_t, EVEN_L).to(gl.float32)
                    Cnx = _ldt(C_ptr + bcn + offs_t,
                               mask_t, EVEN_L).to(gl.float32)
                    if HAS_BB:
                        Bt = Bt + gl.load(beta_b_ptr + gbc * N + n + zl)
                    if HAS_BC:
                        Ct = Ct + gl.load(beta_c_ptr + gbc * N + n + zl)
                else:
                    Bt = _ldt(B_ptr + bc + offs_t,
                              mask_t, EVEN_L).to(gl.float32)
                    Ct = _ldt(C_ptr + bc + offs_t,
                              mask_t, EVEN_L).to(gl.float32)
                    if HAS_BB:
                        Bt = Bt + gl.load(beta_b_ptr + gbc * N + n + zl)
                    if HAS_BC:
                        Ct = Ct + gl.load(beta_c_ptr + gbc * N + n + zl)
                if HAS_GB:
                    gv = _ldt(gB_ptr + bc + offs_t,
                              mask_t, EVEN_L).to(gl.float32)
                    if HAS_BGB:
                        gv = gv + gl.load(beta_gB_ptr + gg * N + n + zl)
                    Bt = Bt * _gate(gv, GATE_ACT)
                if HAS_GC:
                    gv = _ldt(gC_ptr + bc + offs_t,
                              mask_t, EVEN_L).to(gl.float32)
                    if HAS_BGC:
                        gv = gv + gl.load(beta_gC_ptr + gg * N + n + zl)
                    Ct = Ct * _gate(gv, GATE_ACT)
                Ct = _zt(Ct, mask_t, EVEN_L)

                # ---- separable decay drivers, p-free ---------
                s1 = de
                if HAS_DYN and (UNI or law_s == K_DYN):
                    if A_DYN_N_RES:
                        s1 = de * _rate(
                            _ldt(a_dyn_ptr + bc + offs_t, mask_t,
                                 EVEN_L).to(gl.float32),
                            DYN_ACT, DYN_EPS)
                    else:
                        s1 = de * rate_h
                # past L: s1 = 0 -> a_bar = 1 and drive = 0, so the tail
                # columns pass the state through and `last2` still reads the
                # true chunk end
                if not FPACK:
                    s1 = _zt(s1, mask_t, EVEN_L)
                # broadcast, NOT `+ z2`: adding a zero tile is a real fadd per
                # element (x + 0.0 is not x at -0.0, so nothing folds it away),
                # and at BLOCK_P=1 these three sites were ~17% of the inner
                # loop's instructions for no arithmetic at all.
                if FPACK:
                    s1_2 = de2                # per-row decay drive, a zeroed tile
                else:
                    s1_2 = gl.expand_dims(s1, 0)

                # ---- per-channel statics: [BLOCK_P], p-contiguous --------
                if HAS_BA:
                    b_a = gl.load(beta_a_ptr + ga * N + n + zp)
                else:
                    b_a = gl.zeros([BLOCK_P], gl.float32, layout=pl)
                if ABL & 512:
                    # -A_log load: prices structure A's smem_a_static, which
                    # hoists exactly this scalar load and its exp out of the
                    # chunk loop. It sits at the head of the arithmetic chain
                    # (load -> exp -> mul -> exp2 -> scan), so its latency is
                    # not obviously hidden the way the wide B/C loads are.
                    c1 = b_a - 1.0
                elif PFA:
                    c1 = -gl.exp(Anx + b_a)
                    Anx = gl.load(A_log_ptr + anp
                                  + gl.minimum(n + 1, N_OFF + N_ROWS - 1) * P)
                else:
                    c1 = -gl.exp(gl.load(A_log_ptr + anp + n * P) + b_a)
                c1e = gl.expand_dims(c1 * LOG2E, 1)

                log_a = c1e * s1_2
                if HAS_MU and (UNI or law_s == K_MU):
                    # telescoping: sum_{s<=t}(mu_s - mu_{s-1}) = mu_t -
                    # mu_{t0-1}, so MU needs no scan of its own
                    if MU_N_RES:
                        mu_t = _ldt(mu_ptr + bc + offs_t, mask_t,
                                    EVEN_L).to(gl.float32)
                        if MU_PREDIFF:
                            s2_2 = gl.expand_dims(
                                _zt(mu_t, mask_t, EVEN_L), 0)
                        else:
                            # inside the chunk mu_{t-1} is mu_t shifted by one
                            # lane, so only column 0 comes from memory. Loading
                            # the whole shifted vector instead costs 1.3x the
                            # WHOLE forward: bc-1 is 4-byte misaligned against
                            # the 128-bit LDG and every lane falls back to its
                            # own transaction.
                            mu_p = gl.gather(mu_t, idx_prv, 0)
                            at0 = (col == 0) & mask_t
                            mu_p = gl.where(at0, gl.load(
                                mu_ptr + bc + offs_t - 1,
                                mask=at0 & (offs_t > 0),
                                other=0.0).to(gl.float32), mu_p)
                            if HAS_MU_INIT:
                                mu_p = gl.where(
                                    offs_t == 0,
                                    gl.load(mu_init_ptr + b * N + n + zl),
                                    mu_p)
                            s2_2 = gl.expand_dims(
                                _zt(mu_t - mu_p, mask_t, EVEN_L), 0)
                    else:
                        s2_2 = s2_h
                    log_a = log_a + gl.expand_dims(
                        gl.load(mu_gain_ptr + anp + n * P) * LOG2E, 1) * s2_2
                if ABL & 1:
                    a = log_a
                else:
                    a = libdevice.exp2(log_a)

                drive = xw * gl.expand_dims(Bt, 0)
                if not (ABL & 8):
                    hprev = gl.load(h_state_ptr + hnp + n * P)
                    drive = gl.where(first2,
                                     drive + a * gl.expand_dims(hprev, 1),
                                     drive)
                if ABL & 2:
                    hs = drive + a
                    qh = a
                elif HAS_ENDPOINT:
                    # the carried state is z_t = h_t + nxt_t B_t x_t (write
                    # w = own + nxt); the readout is h_t = q_t + own_t B_t x_t
                    # with q_t = a_t z_{t-1}, the scan's third component
                    # (`_ssm_combine_q`: no extra shuffles).  Reading z_t and
                    # taking nxt_t B_t x_t back out would cancel whenever
                    # own << nxt (lambda -> 0 heads of a trained trapezoid),
                    # leaving a residue u |nxt B x| independent of the true
                    # readout.
                    qs = gl.where(first2, a, 0.0)
                    if not (ABL & 8):
                        qs = qs * gl.expand_dims(hprev, 1)
                    _, hs, qh = gl.associative_scan(
                        (a, drive, qs), axis=1, combine_fn=_ssm_combine_q)
                else:
                    _, hs = gl.associative_scan((a, drive), axis=1,
                                                combine_fn=_ssm_combine)

                # free: n is the sequential axis
                if HAS_ENDPOINT:
                    yacc += qh * gl.expand_dims(Ct, 0)
                    # own's term has p-dependence own*xt only (n-free), so
                    # sum_n own xt B_n C_n = own xt sum_n B_n C_n: one
                    # [BLOCK_L] FMA per row here, one tile FMA per chunk below.
                    bcac += Bt * Ct
                else:
                    yacc += hs * gl.expand_dims(Ct, 0)
                if ABL & 32:
                    gl.store(h_state_ptr + hnp + n * P, c1)
                else:
                    gl.store(h_state_ptr + hnp + n * P,
                             gl.sum(gl.where(last2, hs, 0.0), axis=1))
                # The backward rebuilds these in a pass of its own unless the
                # forward hands them over; that pass is 16-20% of the
                # backward. Slot 0
                # is the h_init and is never written, so slot k+1 is the end
                # of sub-chunk k -- exactly what the backward's `hck` expects.
                # One masked scatter, not BLOCK_L/CKPT_L masked reduces. The
                # boundary values are already in registers; picking them out
                # costs a store mask, whereas gl.sum cross-lane reduces the
                # whole tile once per sub-chunk -- and that runs inside the n
                # loop: N * BLOCK_L/CKPT_L reductions per chunk.
                if CKPT_L > 0:
                    if CKPT_COMPACT:
                        # Keep each thread's values local, but put adjacent P
                        # values together in register order for vector stores.
                        # Storage and the backward reader remain unchanged.
                        hc = gl.convert_layout(hs.to(ckpt_ptr.dtype.element_ty), cl)
                        pc = pid_p * BLOCK_P + gl.arange(
                            0, BLOCK_P, layout=gl.SliceLayout(1, cl))
                        cc = gl.arange(0, BLOCK_L, layout=gl.SliceLayout(0, cl))
                        slot = (jj * (BLOCK_L // CKPT_L)
                                + cc // CKPT_L + 1).to(gl.int64)
                        if CKPACK:
                            ck_base = b * CKS * N * snc + h.to(gl.int64) + pc
                        else:
                            ck_base = bh * CKS * N * P + pc
                        gl.store(ckpt_ptr + gl.expand_dims(ck_base, 1)
                                 + (gl.expand_dims(slot, 0) * N + n) * snc, hc,
                                 mask=gl.expand_dims(cc % CKPT_L == CKPT_L - 1, 0)
                                 + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=cl))
                    else:
                        slot = (jj * (BLOCK_L // CKPT_L)
                                + col // CKPT_L + 1).to(gl.int64)
                        gl.store(ckpt_ptr + gl.expand_dims(ckb, 1)
                                 + (gl.expand_dims(slot, 0) * N + n) * snc,
                                 hs.to(ckpt_ptr.dtype.element_ty),
                                 mask=gl.expand_dims(col % CKPT_L == CKPT_L - 1, 0)
                                 + gl.zeros([BLOCK_P, BLOCK_L], gl.int1,
                                            layout=tile))
        # x is re-read rather than kept: its only in-loop use is `xw`, and a
        # [BLOCK_P, BLOCK_L] tile held live across the n loop is what pushed
        # this kernel into spilling (the same bf16 value, so exact).
        if HAS_ENDPOINT or HAS_DSKIP:
            xe = _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        if HAS_ENDPOINT:
            yacc += eo * xe * gl.expand_dims(bcac, 0)
        if HAS_DSKIP:
            # per HEAD, broadcast over the head's P channels -- the same
            # `D.repeat_interleave(P)` the module applied.
            if FPACK:
                yacc += gl.expand_dims(gl.load(d_skip_ptr + h + pf), 1) * xe
            else:
                yacc += gl.load(d_skip_ptr + h) * xe
        if HAS_Z:
            # the gate itself: one multiply on a tile that is already in
            # registers, against the 201 MB `_gate_tr_fwd` moved for it.
            yacc = yacc * zg
        if EVEN_L:
            gl.store(Y_ptr + po, yacc)
        else:
            gl.store(Y_ptr + po, yacc, mask=m2)


# ---------------------------------------------------------------------------
# host launcher
# ---------------------------------------------------------------------------
def _thh_table(theta_h):
    """theta_h (H, K) -> its (cos, sin) as one (H, K, 2) fp32 table.

    Both kernels read the READ-side offset only through cos/sin, and the
    offset is static per launch, so evaluating it per (chunk, pair) inside the
    loops would be H*L/BLOCK_L-fold redundant.  One table serves the forward and
    the backward, so the two differentiate the same map.
    """
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
    """Keep the tile at ~ELEMS_PER_THREAD registers per live tile.

    The whole point of the transpose is that registers stop being the binding
    constraint, so BLOCK_L must move opposite BLOCK_P rather than being a
    constant -- a fixed 64 at BLOCK_P=8 leaves 4 elements per thread, which is
    too few for the sequential part of the scan to amortise the shuffles.
    """
    want = 32 * NUM_WARPS * ELEMS_PER_THREAD // max(BLOCK_P, 1)
    return min(triton.next_power_of_2(L),
               max(32, min(256, triton.next_power_of_2(max(want, 1)))))


# One residency wave, floored to a power of two: 110 SMs x 65536/(32*96)
# warps at this kernel's 96 registers is 2200 CTAs, since every CTA is one
# warp. BLOCK_P=2 halves the grid, so it may only be taken where the halved
# grid still fills the machine. The power-of-two floor matters -- B*H*P is a
# power of two on every real shape, so 2200 and 2048 are different rules.
WAVE_CTAS = 2048
BP2_L_MAX = 2048

_SM = None


def _sm():
    """Compute capability as an int (90, 120, ...), resolved once.

    The constants in `_pick_config` are fitted for 110 SMs; other parts
    (e.g. 114 SMs) take the tuner / per-device table rather than a re-fit.
    """
    global _SM
    if _SM is None:
        try:
            major, minor = torch.cuda.get_device_capability()
            _SM = major * 10 + minor
        except Exception:                        # no CUDA: rule is unused
            _SM = 0
    return _SM


def _pick_config(Bs, H, P, N, L, heavy=True, esz=4):
    """Measured, not derived: the fallback rule when no tuned config exists.

    Every good config satisfies one relation --

        BLOCK_P * BLOCK_L / (32 * NUM_WARPS) == ELEMS_PER_THREAD  (== 16)

    -- and within that family only two members are taken: BLOCK_P=1 with the
    whole tile on time, and BLOCK_P=2 with BLOCK_L halved. Both use 96
    registers without spilling.

    BLOCK_P=2 only pays with TPW_L=32. `_lay` puts p on THREADS --
    `tp_p = BLOCK_P // NUM_WARPS` -- so at a small TPW_L raising BLOCK_P splits
    the warp across p and REPLICATES every p-free vector (B, C, delta, the law
    drivers) over the new thread groups: the amortisation is exactly zero.
    TPW_L=32 pins `tp_p=1`, which moves p into `size_per_thread`, and only then
    does a wider p tile cover more work per load.

    The trade is load width against scan width. `spt_l` halves with BLOCK_P
    (8 bf16 elements is one LDG.128, 4 is an LDG.64) and the chunk count
    doubles, so the scan's 5 shuffle rounds are paid twice as often per unit of
    work. Hence the two guards:

    * The halved grid must still fill one residency wave (B*H*P/2 >= ~2200);
      below that BLOCK_P=2 idles half the machine.
    * In bf16 it must ALSO land near exactly one wave, and L must be short.
      The saving scales with bytes loaded, which bf16 halves, while the scan
      tax does not move -- so bf16 wins only where the grid gain is real
      (two launched waves become one) and the chunk count is small. fp32 has
      no such condition.

    The ragged L=1000 shape, whose 512-tile straddles L and drops to
    per-element predicated loads, gains most.
    """
    # General tuned kernels, no capability literals: the coupled corners (BP4 x BL128/256 x PF0/1, BLOCK_L walking
    # to 512) are in `autotune.t_grid`, so the default `light` search reaches
    # them by measurement. A greedy descent seeded at BLOCK_P=1 does not walk
    # to 4, which is why the corners are tried as a grid before the descent.
    # The `pinned` tier (no timing) falls to the generic rule below or to a
    # persisted measured verdict.
    block_p = _divisor_le(P, TILE_P)
    block_l = min(triton.next_power_of_2(L), MAX_BLOCK_L)
    tpw_l = 4
    ctas = Bs * H * (P // 2)
    if (P % 2 == 0 and block_l == MAX_BLOCK_L and ctas >= WAVE_CTAS
            and (esz > 2 or (ctas < 2 * WAVE_CTAS and L <= BP2_L_MAX))):
        block_p, block_l, tpw_l = 2, MAX_BLOCK_L // 2, 32
    num_warps = max(1, block_p * block_l // (32 * ELEMS_PER_THREAD))
    # PF is spelled out even where it is off, so that every config this returns
    # carries the same key set and the tuner's axes always name something the
    # heuristic already has an opinion about.
    return dict(BLOCK_L=block_l, BLOCK_P=block_p,
                NUM_WARPS=num_warps, TPW_L=tpw_l, PF=0)


def _tpw(cfg):
    """Derive TPW_L from BLOCK_P. Exactly one value is ever right.

    `_lay` collapses `tp_p` to 1 at BLOCK_P=1 whatever TPW_L says, so a tuned
    TPW_L there is pure noise -- it varies between runs with zero timing
    effect, which is how the forward's first tuner wiring came to list it as an
    axis. Above BLOCK_P=1 it must be 32, which pins tp_p=1 so that BLOCK_P
    lands in `size_per_thread` rather than splitting the warp across p and
    replicating every p-free vector for no amortisation at all. Leaving it to
    the search would spend trials rediscovering that.
    """
    c = dict(cfg)
    c["TPW_L"] = 32 if c.get("BLOCK_P", 1) > 1 else 4
    return c


def _hnp(t, Bs, H, P, N):
    """(B,H,P,N) -> (B,H,N,P), p fastest, ALWAYS a fresh buffer.

    The kernel advances the state through this tensor in place, so it must not
    be a view of the caller's `h_init`. `.contiguous()` alone does not
    guarantee that: at P == 1 the permute is already contiguous and returns
    `t` itself.
    """
    out = torch.empty(Bs, H, N, P, device=t.device, dtype=torch.float32)
    return out.copy_(t.reshape(Bs, H, P, N).permute(0, 1, 3, 2))


def mh_forward(
    x, delta, A_log, B_t, C_t, *,
    delta_w=None, delta_softplus=True, endpoint_own=None,
    row_law=None, row_rot=None,
    a_dyn=None, mu=None, mu_gain=None,
    a_dyn_res="n", mu_res="n",
    dyn_act=DYN_SOFTPLUS, dyn_eps=0.0,
    g_B=None, g_C=None, gate_act=GATE_SIGMOID,
    theta=None, theta_delta=False, theta_h=None,
    beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None,
    delta_bias=None, h_init=None, theta_init=None, mu_init=None,
    mu_prediff=False, d_skip=None, z=None,
    n_off=0, n_rows=None, config=None, ckpt_l=0, ckpt_out=None,
    ckpt_dt=None, bc_head=False, pmap=False,
):
    """Returns (Y, h_final, ckpt, theta_final).

    `pmap=True`: B_t/C_t are the (B, 2N, L) pair maps of a gated rotating
    launch (`functions._pair_maps`, kernel constexpr PMAP); gates and B/C
    biases are already inside them.

    `mu_prediff` says `mu` already holds the telescoped increment
    `mu_t - mu_{t-1}` rather than `mu` itself, so the kernel loads it and stops.
    The difference is the caller's to take (functions.py does it on the host,
    where it is head-redundant); `mu_init` is then already folded into column 0
    and MUST be None.

    `ckpt_l > 0` additionally writes h at every ckpt_l steps into a
    (B, H, CKS, N, P) buffer, slot 0 being the h_init. That is exactly the
    `hck` the backward otherwise spends a whole pass rebuilding. `ckpt_dt`
    sets its dtype; None means the INPUT dtype when that is 16-bit (the
    checkpoint is a rounding of a state the inputs already carry at that
    precision, and the buffer is the backward's largest fixed term -- 520 MiB
    at the reference shape) and fp32 otherwise. `ckpt_bf16=False` restores
    fp32 globally.

    Under ROT the n window must be pair-aligned: with n sequential the partner
    row is fetched by index, so a window that starts or ends mid-pair has no
    partner to fetch.
    """
    _argv = dict(locals())
    if mu_prediff and mu_init is not None:
        raise ValueError("mu_prediff means `mu` already holds the telescoped\n"
                         "increment, so mu_init is folded into column 0 and must\n"
                         "be None -- applying it again shifts row 0 of every b")
    # "raw" pair maps are the plain (bias-folded) B/C: the forward's own
    # rotating body already loads each pair once, so only the backward
    # (PRAW) reads them differently
    if pmap == "raw":
        pmap = False
    if bc_head:
        expected = (x.shape[1], x.shape[0], A_log.shape[-1], x.shape[-1])
        if B_t.shape != expected or C_t.shape != expected:
            raise ValueError("prepared B/C must be (H,B,N,L)")
        if any(t is not None for t in (theta, g_B, g_C, beta_b, beta_c, beta_gB, beta_gC)):
            raise ValueError("prepared maps must have all map folds applied")
    if pmap:
        if theta is None or bc_head or any(t is not None for t in (
                g_B, g_C, beta_b, beta_c, beta_gB, beta_gC)):
            raise ValueError("pmap: pair maps belong to a rotating launch "
                             "with every B/C fold (bias, gate) applied")
        if B_t.shape[1] != 2 * A_log.shape[-1] or C_t.shape != B_t.shape:
            raise ValueError("pmap: B/C pair maps must be (B, 2N, L)")
    sh = validate_forward_inputs(
        x, delta, A_log,
        B_t[0] if bc_head else (B_t[:, ::2] if pmap else B_t),
        C_t[0] if bc_head else (C_t[:, ::2] if pmap else C_t),
        delta_w=delta_w, a_dyn=a_dyn, mu=mu,
        mu_gain=mu_gain, g_B=g_B, g_C=g_C, theta=theta,
        beta_a=beta_a, beta_b=beta_b, beta_c=beta_c, beta_gB=beta_gB,
        beta_gC=beta_gC, delta_bias=delta_bias, h_init=h_init,
        theta_init=theta_init, mu_init=mu_init, n_off=n_off, n_rows=n_rows)
    Bs, H, P, N, L = sh.B, sh.H, sh.P, sh.N, sh.L
    nr = (N - n_off) if n_rows is None else n_rows
    dev = x.device

    # `row_law=None` is LAW_A everywhere, and knowing that costs no read at
    # all -- the zeros buffer below is a fresh tensor on every call, so
    # `selector_window`'s memo could never hit it.
    all_a_fix = row_law is None
    if all_a_fix:
        row_law = torch.zeros(N, dtype=torch.int32, device=dev)
    # The windows come off the CALLER's tensors, BEFORE any dtype conversion:
    # a bool `row_rot` (what `fracs_to_rows` returns) converted first would be
    # a fresh object every call and would miss `selector_window`'s memo.
    # One device->host transfer of the (N,) selector, then plain python:
    # `.any()`/`.all()`/`== law` are each a kernel plus a SYNC: invisible in
    # GPU self-time, but host wall time on a busy card.
    rot_w = selector_window(row_rot, n_off, nr)
    law_present = ({LAW_A_FIX} if all_a_fix
                   else set(selector_window(row_law, n_off, nr)))
    row_law = row_law.to(torch.int32).contiguous()
    ROT = theta is not None and (rot_w is None or any(rot_w))
    # pair maps are only readable by the rotating body; a launch where no
    # row rotates runs it with every row on the identity (HAS_RROT below).
    ROT = ROT or bool(pmap)
    if ROT and (n_off % 2 or nr % 2):
        raise ValueError(f"ROT needs a pair-aligned n window, got "
                         f"n_off={n_off} n_rows={nr}")

    # exactly which laws this n window holds, not just whether any non-A one
    # is there: a window on a single law compiles without the runtime dispatch
    # at all (UNI in the kernel), so an n split can specialise per shard.
    LAW_MASK = sum(1 << law for law in
                   (LAW_A_FIX, LAW_A_DYN, LAW_MU)
                   if law in law_present) or 1

    # the carry lives p-fastest; h_final is handed back as a view, not a copy
    h_np = torch.zeros(Bs, H, N, P, device=dev, dtype=torch.float32) \
        if h_init is None else _hnp(_f32(h_init), Bs, H, P, N)
    Y = torch.empty(Bs, H, P, L, device=dev, dtype=x.dtype)

    def _t3(t):
        return None if t is None else \
            _f32(t).reshape(H, P, N).permute(0, 2, 1).contiguous()

    # The backward's packed small-P schedule reads its checkpoints
    # channel-fastest; taken at P == 1 for the same reason it is (a head is a
    # channel there), and keyed on P alone so the two sides agree without the
    # forward knowing the backward's pack width.
    # An INFERENCE launch (no checkpoint asked for, none handed in) may still
    # run the checkpoint-writing variant into a discarded scratch buffer:
    # config key CKV, searched with the coupled corners.  Same Y / h_final;
    # what differs is the compiled loop, and which variant is faster flips
    # with the corner (BLOCK_L, BLOCK_P, PF), so it is measured, never decided by a rule.
    # Only NON-ROTATING launches take part (the fold route, whose scan does
    # not rotate): with the rotation inside the scan the checkpoint variant's
    # forward is not bitwise the plain one, and an inference result must not
    # depend on which variant a search picked.
    _infer = not ckpt_l and ckpt_out is None and not ROT
    CKPACK = bool(ckpt_l) and P == 1
    # Register-order packing combines adjacent P values into vector stores.
    # A full-row launch overwrites every boundary, including the padded tail;
    # only partial launches need zero-filled unowned rows. Env pins retain
    # the two independent ablations without changing the checkpoint ABI.
    ckpt_compact = bool(ckpt_l) and os.environ.get("MH_CKPT_STORE", "1") != "0"
    ckpt_zero = os.environ.get("MH_CKPT_ZERO", "0") != "0" or n_off != 0 or nr != N
    heavy = ROT or g_B is not None or g_C is not None
    # An explicit `fwd_config` pin, else the calibrated P = 1 cell (the shipped
    # per-GPU table, `calibration_mh.p1_config`): both are taken as is and
    # never searched; a cell the table does not hold goes to the tuner.
    _pin = autotune.pinned_config("fwd") if config is None else None
    _cal = None
    if config is None and not _pin and ckpt_out is None and P == 1:
        from . import calibration_mh as _cm
        if _cm.p1_envelope(P=P, ROT=ROT, pmap=pmap, bc_head=bc_head,
                           LAW_MASK=LAW_MASK, endpoint_own=endpoint_own,
                           n_off=n_off, nr=nr, N=N,
                           absent=(delta_w, g_B, g_C, beta_a, beta_b, beta_c,
                                   beta_gB, beta_gC, a_dyn, mu, theta, mu_init,
                                   theta_init)):
            _cal = _cm.p1_config(
                "fwd_trn" if ckpt_l else "fwd", Bs, H, N, L, x.dtype,
                _cm.p1_feat(delta_softplus=delta_softplus, delta_bias=delta_bias,
                            d_skip=d_skip, z=z, h_init=h_init))
    _fp = (_pin or _cal or config or {}).get("FPACK")
    # FPACK envelope: the Mamba-1 function at P == 1 (see the kernel note)
    fpack_ok = (P == 1 and H > 1 and not ROT and not pmap and not bc_head
                and LAW_MASK == 1 and endpoint_own is None
                and all(t is None for t in (delta_w, g_B, g_C, beta_a, beta_b,
                                            beta_c, beta_gB, beta_gC, a_dyn, mu))
                # A loss at every P = 1 cell measured: the forward is not
                # B/C-load bound, and packing multiplies elements per thread.
                # Opt-in only.
                and (bool(_fp) if _fp is not None
                     else os.environ.get("MH_FWD_FPACK", "0") == "1"))
    PW = H if fpack_ok else P        # the extent BLOCK_P tiles
    # TS (time split, `ts_forward`): offered where the envelope serves it and
    # the walk is long enough for the seed pass's two extra launches to be
    # noise (they are host time the GPU-timed search cannot see)
    from .ts_forward import ts_fwd_refusal
    ts_ok = (L >= int(os.environ.get("MH_FWD_TS_MIN_L", 4096))
             and ts_fwd_refusal(
                 ROT=ROT, pmap=pmap, LAW_MASK=LAW_MASK, delta_w=delta_w,
                 beta_a=beta_a, beta_b=beta_b, beta_c=beta_c, g_B=g_B, g_C=g_C,
                 beta_gB=beta_gB, beta_gC=beta_gC, endpoint_own=endpoint_own,
                 bc_head=bc_head, n_off=n_off, nr=nr, N=N, fpack=fpack_ok) is None)
    cfg = (_pick_config(Bs, H, P, N, L, heavy, x.element_size())
           if config is None else dict(config))
    # `fwd_config` (kernel option / HYBRID_MH_FWD_CONFIG): an explicit launch
    # config, taken as is and never searched; otherwise the per-device
    # table of measured winners (`autotune.table_fwd`) replaces the
    # heuristic, as the `pinned` tier's config and the search's trial 0.
    _pin = _pin or _cal
    if config is None:
        cfg.update(_pin or autotune.table_fwd(
            _sm(), ROT, bool(pmap), x.element_size(), P, N) or {})
        if _pin or cfg.get("BLOCK_P", 1) > 1:
            cfg = _tpw(cfg)
    cfg["BLOCK_P"] = _divisor_le(PW, cfg["BLOCK_P"])
    cfg.setdefault("PF", 0)          # a caller's config need not mention it
    cfg.setdefault("TS", int(os.environ.get("MH_FWD_TS", 1)))

    _tier = autotune.tier()
    if config is None and ckpt_out is None and not _pin and autotune.enabled():
        # The heuristic above is trial 0 and is only displaced by a config that
        # is both faster AND agrees with it, so this can match the hand-fitted
        # rule but never regress against it. `_pick_config` carries constants
        # fitted on 110 SMs; the search covers other devices.
        #
        # `ckpt_out` given means the caller sized the buffer, and `cks` depends
        # on BLOCK_L whenever BLOCK_L does not divide L -- so a trial could
        # need a shape the caller did not allocate. Same guard as the
        # backward's `out`.
        #
        # EVERY constexpr of the launch below belongs in this key and nothing
        # else does: one left out lets two feature sets that compile to
        # DIFFERENT kernels share a tuned config, which is silent and wrong
        # rather than merely slow.
        # SLP (a constexpr the host decides) and the checkpoint dtype lever
        # change the compiled kernel without touching its source hash
        key = ("fwd_t", _tier, slp.active(_sm()),
               # rotation-law constexprs of the kernel (every constexpr belongs
               # in the key: another value is another kernel)
               bool(theta_delta) and theta is not None, theta_h is not None,
               bool(ROT and rot_w is not None and not all(rot_w)),
               _ko.value("hybrid_mamba", "ckpt_bf16"),
               bool(bc_head), bool(pmap), B_t.dtype, Bs, H, P, N, L, n_off, nr, x.dtype, LAW_MASK, ROT,
               theta is not None, None if theta is None else theta.dim(), bool(delta_softplus),
               int(dyn_act), int(gate_act), float(dyn_eps),
               a_dyn_res, mu_res, delta_w is not None,
               None if delta_w is None else delta_w.dim(),
               g_B is not None, g_C is not None, h_init is not None,
               beta_a is not None, beta_b is not None, beta_c is not None,
               beta_gB is not None, beta_gC is not None,
               delta_bias is not None, mu_init is not None,
               bool(mu_prediff), int(ckpt_l), ckpt_dt, CKPACK, ckpt_compact, ckpt_zero,
               "ckv" if _infer else None,
               fpack_ok, ts_ok,
               # the store's two epilogue constexprs (neither changes the
               # tile geometry, but the rule above admits no exceptions)
               d_skip is not None, z is not None,
               endpoint_own is not None, None if endpoint_own is None else endpoint_own.ndim,
               *(None if t is None else t.shape[0]
                 for t in (beta_a, beta_b, beta_c, beta_gB, beta_gC)))
        # host fast path: a settled key skips the closure/axes
        _hit = autotune.cached(key)
        if _hit is not None:
            cfg = _tpw(_hit)
        else:
            kwt = {k: v for k, v in _argv.items()
                   if k not in ("x", "delta", "A_log", "B_t", "C_t", "config")}

            def _run(c):
                return mh_forward(x, delta, A_log, B_t, C_t, **kwt,
                                    config=_tpw(c))

            cfg = _tpw(autotune.choose_axes(
                key, cfg, autotune.t_axes(cfg, PW, L, int(ckpt_l), ROT, _tier,
                                          ts_axis=ts_ok),
                _run, persist=_mh_fwd_kernel.cache_key,
                grid=[dict(g, CKV=v) for g in autotune.t_grid(cfg, PW, L, int(ckpt_l))
                      for v in ((0, 1) if _infer else (0,))],
                candidate_filter=autotune.light_fwd_candidate if _tier == "light" else None,
                # the split changes the parallelism regime the corners are
                # fitted in (as in the backward)
                pre=("TS",)))
        cfg["BLOCK_P"] = _divisor_le(PW, cfg["BLOCK_P"])
    FPACK = fpack_ok and cfg["BLOCK_P"] > 1
    TS = max(1, min(int(cfg.get("TS", 1)), triton.cdiv(L, cfg["BLOCK_L"]))) if ts_ok else 1
    grid = ((1, H // cfg["BLOCK_P"], Bs) if FPACK
            else (P // cfg["BLOCK_P"], H, Bs * TS))

    # The phase carry is per (b, h, pair) but every p-CTA advances its own
    # copy: with n sequential there is no tile axis to hold it, so it lives in
    # scratch, and sharing one copy across the grid's p dimension would be a
    # race. Slice 0 is the answer; the rest is P/BLOCK_P-fold redundant work on
    # a quantity that is 1/(N*L) of the kernel's traffic.
    K = N // 2
    gp = grid[0]
    # Allocate then copy: the kernel WRAPS AND REWRITES this carry in place, so
    # it must not alias `theta_init`. `.expand(...).contiguous()` looks like a
    # copy but is a no-op view at gp == 1 (BLOCK_P == P), and that silently
    # overwrote the caller's tensor with the final phase.
    th_np = torch.zeros(Bs, H, gp, K, device=dev, dtype=torch.float32)
    if theta_init is not None:
        th_np.copy_(theta_init.reshape(Bs, H, 1, K))

    _ckv = False
    if _infer and int(cfg.get("CKV", 0)):
        from .bwd_gluon import auto_block_l
        _s = auto_block_l(P, L)
        if _s and cfg["BLOCK_L"] % _s == 0:
            # the training launch's exact checkpoint geometry (stride, pack,
            # compact store), into scratch that is dropped below
            ckpt_l, _ckv = _s, True
            CKPACK = P == 1
            ckpt_compact = os.environ.get("MH_CKPT_STORE", "1") != "0"

    # Slot count is fixed by the FORWARD's chunking, not by cdiv(L, ckpt_l):
    # when L is ragged the last forward chunk still emits its full complement
    # of sub-boundaries, and the backward has to index with the same stride.
    cks = 1
    if ckpt_l:
        if cfg["BLOCK_L"] % ckpt_l:
            raise ValueError(f"ckpt_l {ckpt_l} must divide BLOCK_L "
                             f"{cfg['BLOCK_L']}")
        cks = triton.cdiv(L, cfg["BLOCK_L"]) * (cfg["BLOCK_L"] // ckpt_l) + 1
        if ckpt_out is None and CKPACK:
            if ckpt_dt is None:
                ckpt_dt = x.dtype if (
                    x.element_size() == 2
                    and _ko.value("hybrid_mamba", "ckpt_bf16")
                ) else torch.float32
            ckpt_out = (torch.zeros if ckpt_zero else torch.empty)(Bs, cks, N, H * P, device=dev,
                                   dtype=ckpt_dt)
        elif ckpt_out is None:
            if ckpt_dt is None:
                ckpt_dt = x.dtype if (
                    x.element_size() == 2
                    and _ko.value("hybrid_mamba", "ckpt_bf16")
                ) else torch.float32
            # Partial n_off/n_rows launches leave unowned rows
            # untouched; full launches write all slots after the seed below.
            ckpt_out = (torch.zeros if ckpt_zero else torch.empty)(Bs, H, cks, N, P, device=dev,
                                   dtype=ckpt_dt)
        # The kernel writes slots 1.. only, so slot 0 must be seeded here or a
        # non-zero h_init never reaches the backward, and a reused `ckpt_out`
        # keeps the previous call's h_init.  `h_np` is (B,H,N,P); under CKPACK
        # slot 0 is (B,N,D), i.e. the same elements with the head axis folded
        # into the channel one.
        if CKPACK:
            ckpt_out[:, 0].copy_(h_np.permute(0, 2, 1, 3)
                                 .reshape(Bs, N, H * P))
        else:
            ckpt_out[:, :, 0].copy_(h_np)

    h_ts = h_np
    if TS > 1:
        from .ts_forward import ts_fwd_seeds
        nbs = triton.cdiv(triton.cdiv(L, cfg["BLOCK_L"]), TS)
        h_ts = ts_fwd_seeds(
            TS, nbs * cfg["BLOCK_L"], x, delta, A_log.reshape(H, P, N), B_t,
            torch.empty(TS, Bs, H, N, P, device=dev, dtype=torch.float32),
            delta_bias=delta_bias, delta_softplus=delta_softplus,
            h0=h_np if h_init is not None else None)

    global LAST_COMPILED, LAST_CONFIG
    LAST_CONFIG = dict(cfg, TS=TS, FPACK=bool(FPACK), grid=tuple(grid))   # provenance (benchmarks.route_record.served_mh)
    LAST_COMPILED = _mh_fwd_kernel[grid](
        x.contiguous(), delta.contiguous(),
        None if delta_w is None else delta_w.contiguous(),
        _t3(A_log), _t3(mu_gain),
        B_t.contiguous(), C_t.contiguous(),
        None if g_B is None else g_B.contiguous(),
        None if g_C is None else g_C.contiguous(),
        None if a_dyn is None else a_dyn.contiguous(),
        None if mu is None else mu.contiguous(), _f32(mu_init),
        _f32(beta_a), _f32(beta_b), _f32(beta_c), _f32(beta_gB), _f32(beta_gC),
        _f32(delta_bias), row_law,
        None if row_rot is None else row_rot.to(torch.int32).contiguous(),
        None if theta is None else theta.contiguous(), _thh_table(theta_h),
        Y, h_ts, th_np, ckpt_out,
        None if d_skip is None else d_skip.float().contiguous(),
        None if z is None else z.contiguous(),
        endpoint_own,
        H, P, N, L, K, n_off, nr, cks,
        LAW_MASK=LAW_MASK, HAS_ENDPOINT=endpoint_own is not None,
        ENDPOINT_P=endpoint_own is not None and endpoint_own.ndim == 4,
        ROT=ROT,
        THETA_DELTA=bool(theta_delta) and theta is not None,
        THETA_PH=theta is not None and theta.dim() == 4,
        HAS_THH=theta_h is not None,
        HAS_RROT=ROT and rot_w is not None and not all(rot_w),
        PMAP=bool(pmap), SLP=slp.active(_sm()),
        DELTA_SOFTPLUS=bool(delta_softplus), DYN_ACT=int(dyn_act),
        GATE_ACT=int(gate_act),
        A_DYN_N_RES=(a_dyn_res == "n"), MU_N_RES=(mu_res == "n"),
        HAS_DW=delta_w is not None, DW_PER_P=(delta_w is not None
                                              and delta_w.dim() == 4),
        HAS_GB=g_B is not None, HAS_GC=g_C is not None,
        HAS_BA=beta_a is not None, HAS_BB=beta_b is not None,
        HAS_BC=beta_c is not None, HAS_BGB=beta_gB is not None,
        HAS_BGC=beta_gC is not None,
        HAS_DBIAS=delta_bias is not None, HAS_MU_INIT=mu_init is not None,
        MU_PREDIFF=bool(mu_prediff),
        HG_A=(H // beta_a.shape[0]) if beta_a is not None else 1,
        HG_BC=(H // beta_b.shape[0]) if beta_b is not None else
              ((H // beta_c.shape[0]) if beta_c is not None else 1),
        HG_G=(H // beta_gB.shape[0]) if beta_gB is not None else
             ((H // beta_gC.shape[0]) if beta_gC is not None else 1),
        DYN_EPS=float(dyn_eps),
        BLOCK_L=cfg["BLOCK_L"], BLOCK_P=cfg["BLOCK_P"],
        NUM_WARPS=cfg["NUM_WARPS"], TPW_L=cfg.get("TPW_L", 4),
        HAS_DSKIP=d_skip is not None, HAS_Z=z is not None,
        ABL=int(os.environ.get("MH_FWD_ABL", cfg.get("ABL", 0))),
        CKPT_L=int(ckpt_l), CKPACK=CKPACK, CKPT_COMPACT=ckpt_compact, BC_HEAD=bool(bc_head),
        # PF applies only where it is defined: the ROT path
        # fetches B/C by PAIR above the `pi` loop, so there is no per-row load
        # left in the loop body for it to run ahead of.
        # under ROT the B/C part of PF does not apply (the pair loads sit
        # above the `pi` loop); PF then prefetches A_log and the angle.
        PF=int(cfg.get("PF", 0)),
        FAST_TRIG=bool(cfg.get("FAST_TRIG", True)),
        # cfg can only turn it OFF: the predicate is what correctness rests on
        # when BLOCK_L does not divide L, so a config may not assert evenness.
        EVEN_L=(L % cfg["BLOCK_L"] == 0) and bool(cfg.get("EVEN_L", True)),
        FPACK=FPACK, TS=TS,
        num_warps=cfg["NUM_WARPS"],
    )
    if TS > 1:
        h_np = h_ts[TS - 1]           # the last super-chunk ends at L
    return (Y, h_np.permute(0, 1, 3, 2), ckpt_out if ckpt_l and not _ckv else None,
            th_np[:, :, 0, :] if ROT else None)
