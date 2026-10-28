"""The family's one production backward (`mh_backward`), in structure A's
data layout.

WHY THIS LAYOUT
---------------
Tiling [BLOCK_N, BLOCK_L] and distributing N across warps makes
`gu = sum_n lam*w` a cross-warp reduce. It costs log2(NUM_WARPS) rounds
INDEPENDENT of BLOCK_N and runs NBLK*BLOCK_P times per CTA; with ~20 live 2-D
tiles pinning the CTA at 255 registers there is one CTA per SM, so none of
those barriers are hidden.

Structure A never pays it. `d = (h,p)` is on the grid, `n` is a sequential
in-thread loop (m1_bwd.cu:861), and threads span TIME -- so the same
contraction is a register accumulation costing nothing. This kernel mirrors
that layout: the cost is a property of the TILING, not of the work.

THE TRADE, STATED PLAINLY
-------------------------
gu = sum_n(...) wants p live and n sequential; gw/gr/gs1 = sum_p(...) want n
live and p sequential. Both cannot be free. A chooses free-gu and pays
gpuAtomicAdd for the p-contracted gradients (m1_bwd.cu:1469-1485). This kernel
takes the same side of the trade but NOT the atomics: cross-CTA sums through
gpuAtomicAdd would be a DEPARTURE from the determinism contract, which requires
fixed-order slot buffers. `gB_t`/`gC_t` go through `(H,B,N,L)` slot buffers
summed in fixed host order, and the gradients that contract over the n shards
go through slot buffers of their own. Nothing here uses gpuAtomicAdd.

p is the OUTER loop and n the inner one, which is A's nesting and the reason gu
stays a single live vector.

Scope. `mh_backward` carries the whole ABI: the rotation with its per-row
`row_rot` selector, all four decay laws of the contract, the gates, all five
`beta_*` biases, `delta_w`, `h_init`/`gh_final`, `delta_bias`/`delta_softplus`
and the partial-row launch. `functions.py` runs it, and the kernel tests
score it against the two fp64 oracles (`mh_reference`, the explicit
recurrence, and `mh_reference_cumprod`, the closed cumulative-decay form).

Warps sit on time (WARPS_N=False): IPT*32*NUM_WARPS spans ~1-2k steps, the
regime cub::BlockScan is built for (at BLOCK_L <= 64 the cross-warp scan has
nothing to amortise against).
"""
from __future__ import annotations

import os

from triton.language import range as loop_range
import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.extra import libdevice

from . import autotune, slp
from ... import kernel_options as _ko
from .common_gluon import (
    INV_TWO_PI, K_DYN, K_MU, LOG2E, TWO_PI,
    _cmul, _dact, _f32, _gate, _gate_dv, _i32, _rate, _row1, _silu_fast_d,
    _softplus, _ssm_combine, _ssm_combine_q,
    selector_window,
)
from .fwd_gluon import _ldt, _thh_table, _zt
from .layout import (
    DYN_RAW, GATE_SIGMOID, LAW_A_DYN, LAW_A_FIX, LAW_MU,
    group_index,
)

LAST_COMPILED = None
#: The resolved schedule of the most recent `mh_backward` launch: the tuner's
#: pick plus everything the launcher derived from the INPUTS rather than from
#: the config (HPACK's legality, the checkpoint layout, where the softplus
#: adjoint was taken).  A diagnostic, and what a test asserting the launcher
#: REFUSED to pack has to read -- `CompiledKernel.src.constants` is keyed by
#: argument index, not by name.
LAST_CONFIG = None

#: The autotuner's extreme-rate probe rate, passed to `choose_axes` as
#: `extreme=`. Not a refusal boundary the way SSD-MIMO's `A_LOG_MAX` is --
#: this family has no such cap, `exp(40)` is nowhere near fp32's overflow --
#: it is a rate every BLOCK_L in {8, 16} fails at and every BLOCK_L in
#: {32, 64} passes, so a search has to probe it to keep a bad BLOCK_L from
#: ever looking like a winner.
_EXTREME_A_LOG = 40.0


def _envi(name, dflt):
    """An integer knob overridable from the environment, for sweeps only."""
    v = os.environ.get(name)
    return dflt if v is None else int(v)


def _maxnreg():
    """``(maxnreg, pinned)`` -- the cap OVERRIDE, and whether there is one.

    The cap is a TUNER AXIS under the packed schedule
    (`autotune.v3_grid`), not a constant: it helps at some shapes and hurts
    at others (e.g. `L = 4096`, `N = 64`), which is what the search is for.

    `MH_BWD_MAXNREG` pins it for an A/B, and a PIN has to bind the SEARCH
    as well as the launch: ``<= 0`` means "no cap", which is the A/B's
    `before` arm and would be undone in one trial if the grid still
    offered the capped corners.  So `pinned` travels with the value, into
    `v3_grid(caps=...)` and into the tune key -- a pinned run and an
    unpinned one at the same shape are two different searches and must not
    share a cached config.
    """
    v = os.environ.get("MH_BWD_MAXNREG")
    if v is None:
        return None, False
    n = int(v)
    return (n if n > 0 else None), True


#: Widest p tile one CTA takes before `auto_np` shards p instead. With
#: BLOCK_P = P the chunk length `1024 // P` shrinks as the head widens, and the
#: chunk length is also the forward's checkpoint stride -- a short stride makes
#: the forward's checkpoint scatter a masked store with few live lanes.
MAX_BLOCK_P = 32

#: Ceiling on the backward's chunk length, and so on the forward's checkpoint
#: stride.  See `auto_block_l`: the `1024 // BLOCK_P` element budget prices the
#: register tile and not the p-free [BLOCK_L] vectors that dominate at small
#: BLOCK_P, so it has to be capped from above.  Inert at P >= 16.
SMALL_P_BLOCK_L = 128


def auto_np(P):
    """How many CTAs share the p axis, so that BLOCK_P <= MAX_BLOCK_P.

    Kept next to `auto_block_l` because the two must agree: `BLOCK_P = P // NP`
    is what makes `auto_block_l`'s element budget come out at 32.
    """
    np_ = 1
    while P // np_ > MAX_BLOCK_P and (P // np_) % 2 == 0:
        np_ *= 2
    return np_


#: Default pack width for the packed small-P schedule.  It is a tuner axis,
#: not a constant, and is capped by H; the search sweeps it JOINTLY with
#: BLOCK_L (`v3_grid`), because the two set the register tile together.
HP_DEFAULT = 8

#: `auto_block_l`'s element budget under the packed schedule -- half the
#: ordinary one, because the packed tile's rows are different heads and
#: `delta`, its shift and the softplus adjoint become live tiles instead of
#: broadcasts.  See `auto_block_l`.
HPACK_BUDGET = 512


def _hp_default(H):
    hp = HP_DEFAULT
    while hp > 1 and H % hp:
        hp //= 2
    return max(1, min(hp, H))


def auto_block_l(P, L, block_l=None, budget=1024):
    """This kernel's chunk length. The forward's `ckpt_l` has to agree with
    this, so it
    is a function rather than an expression buried in the launcher.

    `1024 // BLOCK_P` is 32 elements per thread at one warp, and BLOCK_P is
    `P // auto_np(P)` rather than P itself: at wide P, sharding p across two
    CTAs and doubling BLOCK_L improves both the forward (its checkpoint scatter
    halves) and the backward, but sharding further needs NUM_WARPS=2 to stay
    inside the element budget, and `_lay3` spends warps on TIME, so the
    warp-local scans become block collectives. Only the JOINT move -- halve
    BLOCK_P and double BLOCK_L together, holding elements per thread and
    NUM_WARPS fixed -- is a win.

    Floored to a power of two because the forward REQUIRES `ckpt_l` to divide
    its `BLOCK_L`, which is always one.

    SMALL-P CAP.  `1024 // BLOCK_P` holds *elements per thread* fixed at 32,
    and that is the right invariant only while the [BLOCK_P, BLOCK_L] TILE is
    what fills the register file.  Below BLOCK_P = 16 it stops being: the n
    loop carries a dozen p-free [BLOCK_L] vectors (B, C, the reversed C,
    gB/gC, gdelta, the law drivers), each costing
    `BLOCK_L * tp_p / (32 * NUM_WARPS)` registers with `tp_p = 1`, i.e. the
    SAME BLOCK_L/32 per thread as the whole tile at BLOCK_P = 1.  So at small
    P the rule would size one live tile and ignore the twelve that scale with
    BLOCK_L alone, and the kernel spills.  `min(..., 128)` is the correction:
    exact at P >= 16 (where `1024 // BLOCK_P <= 64` already), and the tuner's
    own BLOCK_L axis closes the rest, because `_pow2_span` reaches 64 and 32
    from a base of 128 and could reach neither from 1024 or 512.

    `budget` is the element target the first factor holds: 1024 (32 per
    thread at one warp) for the ordinary schedule, 512 for the PACKED one,
    where the tile's rows are DIFFERENT heads and so `delta`, its shifted
    copy and the softplus adjoint stop being broadcasts and become three
    more live tiles; HP*BLOCK_L = 512 holds across HP 4/8/16.
    """
    if block_l:
        bl = block_l
    elif P == 1 and budget == 1024:
        # P == 1 is the PACKED schedule's shape, and this function is also
        # what `functions.py` calls to set the forward's `ckpt_l` -- the two
        # must name the same number or a handed-over checkpoint is read at
        # the wrong stride.  So the P == 1 answer is the packed one
        # (HP_DEFAULT channels x this chunk = HPACK_BUDGET elements), not
        # what the ordinary rule would say for a one-channel tile.
        bl = max(min(HPACK_BUDGET // HP_DEFAULT, SMALL_P_BLOCK_L), 8)
    else:
        bl = max(min(budget // (P // auto_np(P)), SMALL_P_BLOCK_L), 8)
    bl = 1 << (bl.bit_length() - 1)
    return min(bl, triton.next_power_of_2(L))


def _hsum(buf):
    """Collapse a slot buffer's leading axis, skipping the pass if it is 1.

    One slab means either H == 1 or the kernel already accumulated across
    slots atomically, and `sum(0)` on a length-1 axis is a full copy of a
    buffer that is 64 MiB at PROD -- exactly the traffic the collapse exists
    to avoid.
    """
    return buf[0] if buf.shape[0] == 1 else buf.sum(0)


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _tail_sum(v2, rev2, prev2, first2, last2):
    """Exact `(total, tail)` for a chunk driver tile, difference-free.

    `total[p] = sum_t v2[p, t]` and `tail[p, t] = sum_{s > t} v2[p, s]`, both
    off ONE reverse scan -- built as reverse / inclusive-scan / shift /
    un-reverse rather than `total - forward_cumsum(v2)`.

    That subtraction would need `total` (a `gl.sum` REDUCTION) and the forward
    scan's own last column to agree to the bit at t == BLOCK_L - 1, and at an
    extreme rate they need not, even though both compute the same
    mathematical sum: at A_log = 40, c1 ~ -2.4e17, and a residual as small as
    1 ULP of `total`, multiplied by c1, turns a t == BLOCK_L - 1 exponent
    that must be exactly 0 into a large POSITIVE one -- `exp2` overflows to
    +-inf rather than landing on 1.0, and every chunk whose `hprev` reads
    that checkpoint inherits it as `0 * inf`.

    The shift used here carries no such requirement. At t == BLOCK_L - 1 the
    tail is the literal additive identity (0.0), never a value two
    independent roundings produced; every other position is a pure
    accumulation of v2's OWN values, shifted by a gather rather than
    recovered by subtracting a neighbour's value. For v2 >= 0 (every s1 this
    kernel scans is a delta, a rate times a delta, or the constant 1 --
    LAW_A/DYN, `_law_s1`; LAW_MU's s2 is signed by design and is the
    one caller-bounded exception) that makes `tail >= 0`
    exact for EVERY t, not merely for the one position a separate reduction
    happens to agree with.
    """
    rev_incl = gl.associative_scan(gl.gather(v2, rev2, 1), axis=1,
                                   combine_fn=_add)
    total = gl.sum(gl.where(last2, rev_incl, 0.0), axis=1)
    tail = gl.gather(gl.where(first2, 0.0, gl.gather(rev_incl, prev2, 1)),
                     rev2, 1)
    return total, tail


@gluon.constexpr_function
def _red_asm(vec, tpl, wown=0):
    """PTX for a lane-deduplicated VECTOR atomic add, fp32 only.

    The cost of the BC fold's atomics is flat in contention -- it is the
    per-element instruction count of `gl.atomic_add` against the handful of
    vector transactions a store issues.  `red.global.add.v4.f32` lands `vec`
    consecutive elements in one op, so the atomic path issues about a store's
    instruction count. sm_90+ only; the launcher gates on compute capability.

    `tpl` is the owner-lane count. A [BLOCK_L] SliceLayout(0) value is
    REPLICATED across the parent layout's tp_p lane groups (order [1, 0]:
    lane = p_idx * tp_l + l_idx), and inline asm sees every replica where
    `gl.atomic_add` dedups internally -- without the `laneid < tpl` predicate
    each element would be added tp_p times. 32 means "no replication" (the
    full 2-D gx tile) and ptxas folds the always-true compare away.

    Register map with pack=vec: $0..$vec-1 dummy outputs, then vec addresses,
    vec values, vec masks. Only the FIRST address and mask of a pack are used:
    the launcher only enables a width whose packs are whole aligned runs of L
    (spt_l % vec == 0, L % vec == 0), so the addresses are consecutive and the
    mask is uniform across the pack.

    `wown` (> 0) is the owner-WARP count, for tiles that put warps on the p
    axis (`WP` > 1): a SliceLayout(0) value is then replicated across the
    WP warp rows too (order [1, 0]: warp = p_row * wown + l_col), and only
    warps below `wown` add.  0 = warps all on time, no warp replicas.
    """
    wchk = ("" if wown <= 0 else
            f".reg .u32 wid;\n.reg .pred q;\nmov.u32 wid, %tid.x;\n"
            f"shr.u32 wid, wid, 5;\nsetp.lt.u32 q, wid, {wown};\n"
            f"and.pred p, p, q;\n")
    vals = ", ".join("$%d" % (2 * vec + i) for i in range(vec))
    zero = "\n".join("mov.u32 $%d, 0;" % i for i in range(vec))
    return f"""{{
.reg .pred p;
.reg .u32 lid;
mov.u32 lid, %laneid;
setp.lt.u32 p, lid, {tpl};
setp.ne.and.b32 p, ${3 * vec}, 0, p;
{wchk}@p red.global.add.v{vec}.f32 [${vec}], {{{vals}}};
{zero}
}}"""


@gluon.constexpr_function
def _red_cons(vec):
    return ",".join(["=r"] * vec + ["l"] * vec + ["f"] * vec + ["r"] * vec)


@gluon.jit
def _red_add(ptr, val, mask, VEC: gl.constexpr, TPL: gl.constexpr,
             WOWN: gl.constexpr = 0):
    gl.inline_asm_elementwise(
        _red_asm(VEC, TPL, WOWN), _red_cons(VEC),
        [ptr.to(gl.int64), val, mask.to(gl.int32)],
        dtype=gl.int32, is_pure=False, pack=VEC)


@gluon.jit
def _put(ptr, val, mask, ATOMIC: gl.constexpr, EVEN: gl.constexpr = False,
         VEC: gl.constexpr = 0, TPL: gl.constexpr = 32, WOWN: gl.constexpr = 0):
    """Store, or accumulate in L2 -- the two ways to land a cross-CTA sum."""
    if ATOMIC:
        if VEC > 0:
            _red_add(ptr, val, mask, VEC, TPL, WOWN)
        elif EVEN:
            gl.atomic_add(ptr, val, sem="relaxed")
        else:
            gl.atomic_add(ptr, val, mask=mask, sem="relaxed")
    else:
        if EVEN:
            gl.store(ptr, val)
        else:
            gl.store(ptr, val, mask=mask)


@gluon.jit
def _bacc(ptr, g, n, N, val, mask, ON: gl.constexpr, EVEN: gl.constexpr = False):
    """K-C: accumulate a static bias' gradient into a (G, N) buffer.

    A `beta_b`/`beta_c`/`beta_gB`/`beta_gC` gradient is the sum over
    (head-in-group, b, t) of the SAME adjoint the caller is about to store,
    so it can be taken here instead of read back off a slot axis kept at
    `H*NP` for no other purpose (K-C).  One scalar atomic per
    (CTA, chunk, row) against `H*NP` slabs of `(B, N, L)` fp32.

    Only reached when the bias is GROUPED (`G > 1`): a head-SHARED bias'
    gradient is the same reduction of the already-folded `(1, B, N, L)`
    buffer, which the host takes for free, so `G == 1` keeps the inner loop
    exactly as it was.
    """
    if ON:
        gl.atomic_add(ptr + g * N + n, gl.sum(_zt(val, mask, EVEN)), sem="relaxed")


@gluon.jit
def _gate_bacc(ptr, g, n, N, val, mask, ON: gl.constexpr,
               EVEN: gl.constexpr, PARTIAL: gl.constexpr, slot):
    if ON:
        if PARTIAL:
            gl.store(ptr + slot * N + n, gl.sum(_zt(val, mask, EVEN)))
        else:
            _bacc(ptr, g, n, N, val, mask, ON, EVEN)


@gluon.jit
def _st(ptr, val, mask, EVEN: gl.constexpr):
    """A store whose tail predicate exists only when there IS a tail."""
    if EVEN:
        gl.store(ptr, val)
    else:
        gl.store(ptr, val, mask=mask)


@gluon.constexpr_function
def _lay_pv(BLOCK_L, NUM_WARPS):
    """[BLOCK_L] with every lane and warp on time, nothing replicated while
    BLOCK_L >= 32 * NUM_WARPS (below that the high lanes broadcast, and
    `TPL_V` dedups them in the atomics)."""
    return gl.BlockedLayout([max(BLOCK_L // (32 * NUM_WARPS), 1)], [32],
                            [NUM_WARPS], [0])


@gluon.constexpr_function
def _lay3(BLOCK_P, BLOCK_L, NUM_WARPS, TP_P, WP=1):
    """[BLOCK_P, BLOCK_L] with every warp on TIME.

    `warps_per_cta[0] == 1` is the load-bearing part: `sum_p` then never leaves
    a warp, so the p-contractions cost shuffles, not
    barrier round-trips. Registers per tile are BLOCK_P*BLOCK_L/(32*NUM_WARPS)
    whatever TP_P is -- TP_P only trades in-thread adds against shuffles.

    `WP` > 1 moves WP of the warps onto p (the packed head axis): the time
    scans stay inside a warp when WP == NUM_WARPS, and `sum_p` becomes a
    cross-warp reduce -- the trade that experiment prices.
    """
    tp_p = min(TP_P, BLOCK_P)
    tp_l = max(32 // tp_p, 1)
    spt_p = max(BLOCK_P // (tp_p * WP), 1)
    spt_l = max(BLOCK_L // (tp_l * (NUM_WARPS // WP)), 1)
    return gl.BlockedLayout([spt_p, spt_l], [tp_p, tp_l], [WP, NUM_WARPS // WP],
                            [1, 0])


def _vec_tp_p(itemsize, BLOCK_L, NUM_WARPS, BLOCK_P, L):
    """Smallest TP_P that fills a 16-byte transaction per thread on the L axis.

    `_lay3` puts `order=[1, 0]`, so L is the fastest axis and each thread holds
    a run of `spt_l = BLOCK_L * tp_p / (32 * NUM_WARPS)` CONTIGUOUS elements of
    it -- x, gy and delta are all (..., L) contiguous, so that run is exactly
    what one load instruction can fold into a single wide transaction.

    At 4 bytes `spt_l >= 4` already fills 16 B and there is nothing to fix, so
    fp32 returns None and keeps its fitted TP_P bit-for-bit. At 2 bytes the same
    `spt_l` is only 8 B: bf16 issues TWICE the loads for HALF the bytes, which
    is the one way a 16-bit input can come out no cheaper than a 32-bit one.

    None also when the width is unreachable or the addresses would not be
    aligned to it -- row `hp` starts at `hp * L` and chunk `j` at `j * BLOCK_L`,
    so both strides must themselves be whole transactions.

    Answers "what width would this cost?", not "should you take it": the caller
    applies it only when not ROT, where it loses to the spills.
    """
    if itemsize >= 4:
        return None
    vec = 16 // itemsize
    if L % vec or BLOCK_L % vec:
        return None
    tp_p = vec * 32 * NUM_WARPS // BLOCK_L
    if tp_p < 1 or tp_p > min(32, BLOCK_P) or tp_p & (tp_p - 1):
        return None
    return tp_p


@gluon.jit
def _de_load(delta_ptr, delta_bias_ptr, bh, h, L, offs_t, mask_t, zl,
             DELTA_SOFTPLUS: gl.constexpr, HAS_DBIAS: gl.constexpr,
             EVEN_L: gl.constexpr = False, DSIG: gl.constexpr = False):
    """The effective delta over a chunk, as [BLOCK_L].

    Both passes need it and both must agree, so the bias-and-softplus policy
    lives here rather than being spelled out twice. `delta_bias` is (H,) only
    -- an (H,P) bias would make `a` p-dependent through a streaming tensor and
    break the streaming-tensor rule; per-channel write strength is `delta_w`'s job.

    Past L this returns 0, so a = 1 and the drive vanishes: the tail columns
    pass the state through untouched and `last2` still reads the true chunk
    end. That zeroing is why the bias goes on before the mask and not after.
    """
    d = _ldt(delta_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32)
    if HAS_DBIAS:
        d = d + gl.load(delta_bias_ptr + h + zl)
    ds = d
    if DELTA_SOFTPLUS:
        if DSIG:
            # the softplus' own derivative, off the RAW sum and before the
            # mask.  It is otherwise a host pass -- `gde * sigmoid(delta +
            # bias)` on a (B,H,L) fp32 tensor, two more kernels and two more
            # round trips of a tensor this kernel already has in registers.
            # Taken here it is one vector op per chunk and EXACT: same
            # argument, same sigmoid.  See the `DSIG` constexpr for when it
            # is worth the register it keeps live.
            ds = 1.0 / (1.0 + gl.exp(-d))
        d = _softplus(d)
    return _zt(d, mask_t, EVEN_L), ds


@gluon.jit
def _de_load_t(delta_ptr, delta_bias_ptr, h, pofs, po, m2,
               DELTA_SOFTPLUS: gl.constexpr, HAS_DBIAS: gl.constexpr,
               EVEN_L: gl.constexpr = False, DSIG: gl.constexpr = False):
    """The effective delta as a [BLOCK_P, BLOCK_L] TILE, one row per HEAD.

    The packed small-P schedule (`HPACK`, see the kernel docstring) puts
    consecutive HEADS on the tile axis, and `delta` -- p-free inside a head,
    but per-head -- then stops being a broadcast vector.  `HPACK` is only
    taken at P == 1, where `delta` (B,H,L) has exactly `x` (B,H,1,L)'s
    layout, so the tile is the `po` gather the `x` load already builds and
    costs no new index arithmetic.

    Same bias-and-softplus policy and the same past-L zeroing as `_de_load`;
    the two must agree because the packed and unpacked schedules are the same
    kernel.
    """
    d = _ldt(delta_ptr + po, m2, EVEN_L).to(gl.float32)
    if HAS_DBIAS:
        d = d + gl.expand_dims(gl.load(delta_bias_ptr + h + pofs), 1)
    ds = d
    if DELTA_SOFTPLUS:
        if DSIG:
            ds = 1.0 / (1.0 + gl.exp(-d))
        d = _softplus(d)
    return _zt(d, m2, EVEN_L), ds


@gluon.jit
def _dw_load(dw_ptr, de2, bh, L, po, offs_t, mask_t, m2, z2,
             DELTA_SOFTPLUS: gl.constexpr, HAS_DW: gl.constexpr,
             DW_P: gl.constexpr, EVEN_L: gl.constexpr = False,
             HPACK: gl.constexpr = False):
    """The write strength as a [BLOCK_P, BLOCK_L] tile.

    Absent `delta_w` this IS the decay's delta, which is why the
    no-gate path can keep one tensor for both and why `gdelta` below has to
    stop collecting the write term as soon as `delta_w` appears.

    The `HAS_DW` guard has to be a block rather than an early return: Gluon
    traces the whole body regardless of a constexpr `return`, so the loads
    below would still be built against a null `dw_ptr`.
    """
    d = de2
    if HAS_DW:
        if DW_P or HPACK:
            # (B,H,P,L) always, and (B,H,L) too under HPACK: at P == 1 the
            # head-resolved tensor has the channel-major layout `po` reads,
            # and the tile axis is heads, so it is no longer a broadcast.
            d = _ldt(dw_ptr + po, m2, EVEN_L).to(gl.float32)
        else:
            d = gl.expand_dims(_ldt(dw_ptr + bh * L + offs_t, mask_t,
                                    EVEN_L).to(gl.float32), 0) + z2
        if DELTA_SOFTPLUS:
            d = _softplus(d)
        d = _zt(d, m2, EVEN_L)
    return d


@gluon.jit
def _bc_load(ptr, bias_ptr, off, n, gbc, N, mask_t, zl,
             HAS_BIAS: gl.constexpr, EVEN_L: gl.constexpr = False):
    """B or C for row `n` over a chunk, as [BLOCK_L], with its bias folded in.

    The (G,N) bias goes on BEFORE the rotation, so it has
    to ride with the load rather than sit at the use site: the partner row is
    loaded through here too, and a 2x2 that mixed a biased row with an
    unbiased one would be silently wrong only on the rotating rows.
    """
    v = _ldt(ptr + off, mask_t, EVEN_L).to(gl.float32)
    if HAS_BIAS:
        v = v + gl.load(bias_ptr + gbc * N + n + zl)
    return v


@gluon.jit
def _gate_load(g_ptr, bias_ptr, off, n, gg, N, mask_t, zl,
               GATE_ACT: gl.constexpr, HAS_BIAS: gl.constexpr,
               EVEN_L: gl.constexpr = False):
    """act(g + beta_g) for row `n` over a chunk, as [BLOCK_L].

    `g_B`/`g_C` are (B,N,L), the same shape as B_t/C_t, so the caller's `bc`
    offset serves both. Only the ACTIVATED value is returned: `_gate_load_d`
    re-reads the pre-activation for the derivative, so it need not stay live.
    """
    v = _ldt(g_ptr + off, mask_t, EVEN_L).to(gl.float32)
    if HAS_BIAS:
        v = v + gl.load(bias_ptr + gg * N + n + zl)
    return _gate(v, GATE_ACT)


@gluon.jit
def _gate_load_d(g_ptr, bias_ptr, off, n, gg, N, mask_t, zl,
                 GATE_ACT: gl.constexpr, HAS_BIAS: gl.constexpr,
                 EVEN_L: gl.constexpr = False):
    """d act/dv at the same (row, chunk) as `_gate_load`, from the pre-activation
    re-read here (an L1/L2 hit) rather than held live across the row, and exact
    where the gate saturates (`_gate_dv`)."""
    v = _ldt(g_ptr + off, mask_t, EVEN_L).to(gl.float32)
    if HAS_BIAS:
        v = v + gl.load(bias_ptr + gg * N + n + zl)
    return _gate_dv(v, GATE_ACT)


@gluon.jit
def _rot_cs(theta_ptr, thck_ptr, thb, h, bh, kk, K, L, NBLK, j, offs_t,
            col, m2, z2, zl, de, INV_BP: gl.constexpr, ABL: gl.constexpr,
            FAST_TRIG: gl.constexpr, EVEN_L: gl.constexpr = False,
            THETA_DELTA: gl.constexpr = False, ROW1: gl.constexpr = None):
    """(cos Theta, sin Theta) for pair `kk` over chunk `j`, as [BLOCK_L].

    The prefix sum runs on ONE row (`ROW1`, `common_gluon._row1`): a scan
    along a SliceLayout axis does not lower, and the alternative is the
    p-replicated [BLOCK_P, BLOCK_L] tile plus a reduce back over p. The
    transcendentals stay at one per thread either way.

    Theta reaches the chunk carry already wrapped into [0, 2pi), so the SFU
    approximations are used on the interval they are accurate on. Their error
    is checked against the fp64 oracle at the same 2e-5 bar.

    Under the ONE rotation law every rotating launch carries `theta`, so only
    the cumulative-sum form below is reachable.
    """
    Th = gl.load(thck_ptr + (bh * NBLK + j) * K + kk + zl)
    if not (ABL & 32):
        thv = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_t,
                   offs_t < L, EVEN_L).to(gl.float32)
        if THETA_DELTA:
            thv = thv * de
        inc1 = gl.expand_dims(gl.convert_layout(
            _zt(thv, offs_t < L, EVEN_L), gl.SliceLayout(0, ROW1)), 0)
        Th += gl.convert_layout(gl.sum(gl.associative_scan(
            inc1, axis=1, combine_fn=_add), axis=0), zl.type.layout)
    if ABL & 256:
        return Th, Th
    if FAST_TRIG:
        return libdevice.fast_cosf(Th), libdevice.fast_sinf(Th)
    return gl.cos(Th), gl.sin(Th)


@gluon.jit
def _rot_sel(row_rot_ptr, kk, NPAIR: gl.constexpr):
    """1.0 if pair `kk` rotates, else 0.0.

    `row_rot` is pair-tied by construction (layout.norm_fracs rejects a
    rotating block at an odd offset or with an odd row count), so the even row
    decides for both. Returned as a float rather than a predicate because it
    scales cos/sin into the identity, which leaves every rotated expression
    equal to its plain-real form without a second code path.
    """
    return gl.load(row_rot_ptr + kk * NPAIR).to(gl.float32)


@gluon.jit
def _law_s1(a_dyn_ptr, law_s, de, b, bh, n, N, L, offs_t, mask_t, zl,
            HAS_DYN: gl.constexpr, UNI: gl.constexpr,
            DYN_ACT: gl.constexpr, DYN_EPS: gl.constexpr,
            A_DYN_N_RES: gl.constexpr, EVEN_L: gl.constexpr = False):
    """(s1, d s1/d de) for row `n` as [BLOCK_L]; s1 = de on LAW_A.

    `law_s` is a SCALAR, not a per-lane value, so these are real branches. n is
    this kernel's sequential loop, so every lane in the CTA is on the same
    row and the
    predicate is uniform -- a row that is not A_DYN never issues the a_dyn load
    or the rate's transcendental at all (with n on a tile axis the rows would
    disagree and every law would be evaluated on every row).

    UNI is when the whole tensor is on one law, and then the branch is pure
    loss: it buys no skipped work and costs the scheduler the freedom to hoist
    the a_dyn load. Python's `or` short-circuits on the constexpr, so under UNI
    the predicate never becomes a Gluon value and no branch is emitted.

    The derivative comes back with s1 because the delta adjoint needs it, and
    it is 1 on LAW_A and the rate on DYN.
    """
    s1 = de
    fac = zl.to(gl.float32) + 1.0
    if HAS_DYN and (UNI or law_s == K_DYN):
        if A_DYN_N_RES:
            ad = _ldt(a_dyn_ptr + (b * N + n.to(gl.int64)) * L + offs_t,
                      mask_t, EVEN_L).to(gl.float32)
        else:
            ad = _ldt(a_dyn_ptr + bh * L + offs_t, mask_t,
                      EVEN_L).to(gl.float32)
        fac = _rate(ad, DYN_ACT, DYN_EPS)
        s1 = de * fac
    # past L: s1 = 0 -> a = 1 and drive = 0, so the tail columns pass the state
    # through and `last2` still reads the true chunk end.
    return _zt(s1, mask_t, EVEN_L), fac


@gluon.jit
def _law_s2(mu_ptr, mu_init_ptr, b, bh, n, N, L, offs_t, mask_t, col, zl,
            MU_N_RES: gl.constexpr, HAS_MU_INIT: gl.constexpr,
            MU_PREDIFF: gl.constexpr, EVEN_L: gl.constexpr = False):
    """s2 = mu_t - mu_{t-1}: the telescoped MU driver.

    Called only under a scalar `law_s == K_MU`, so there is no law predicate
    here. The driver telescopes, but the adjoint does not telescope for free:
    gs2 is emitted per step and the host takes the difference -- so under
    MU_PREDIFF, where `mu` already IS s2, the emitted gradient is already the
    one the caller wants and the launcher stops differencing it too.
    """
    if MU_N_RES:
        mo = (b * N + n.to(gl.int64)) * L + offs_t
    else:
        mo = bh * L + offs_t
    mu_t = _ldt(mu_ptr + mo, mask_t, EVEN_L).to(gl.float32)
    if MU_PREDIFF:
        s2 = mu_t
    else:
        # inside the chunk mu_{t-1} is mu_t shifted by one lane, so only column
        # 0 comes from memory. Loading the whole shifted vector instead breaks
        # the 128-bit LDG -- mo-1 is 4-byte misaligned so every lane falls back
        # to its own transaction.
        mu_p = gl.gather(mu_t, gl.maximum(col - 1, 0), 0)
        at0 = (col == 0) & mask_t
        mu_p = gl.where(at0, gl.load(mu_ptr + mo - 1, mask=at0 & (offs_t > 0),
                                     other=0.0).to(gl.float32), mu_p)
        if HAS_MU_INIT:
            if MU_N_RES:
                mi = gl.load(mu_init_ptr + b * N + n + zl)
            else:
                mi = gl.load(mu_init_ptr + bh + zl)
            mu_p = gl.where(offs_t == 0, mi, mu_p)
        s2 = mu_t - mu_p
    return _zt(s2, mask_t, EVEN_L)


@gluon.jit
def _mh_bwd_kernel(
    x_ptr, delta_ptr, A_log_ptr, B_ptr, C_ptr, gy_ptr,
    theta_ptr, thh_ptr, thi_ptr,
    row_law_ptr, row_rot_ptr,
    a_dyn_ptr, mu_ptr, mu_gain_ptr, mu_init_ptr,
    delta_bias_ptr, beta_b_ptr, beta_c_ptr, dw_ptr,
    gB_ptr, gC_ptr, bgB_ptr, bgC_ptr,
    gx_ptr, gde_ptr, gc1_ptr, ghi_ptr, gBt_ptr, gCt_ptr, gth_ptr,
    gthh_ptr, gc2_ptr, grate_ptr, gs2o_ptr, gdw_ptr, ggB_ptr, ggC_ptr,
    hck_ptr, alam_ptr, thck_ptr, rcar_ptr,
    d_skip_ptr, gd_skip_ptr, z_ptr, gz_ptr, endpoint_ptr, gendpoint_ptr,
    # H3 item 2: the `delta_bias` gradient's BLOCK PARTIALS, (nsp, B, H)
    # fp32.  See the `DBACC` constexpr.
    gdbias_ptr,
    # K-C: the (G, N) bias accumulators.  Live only where the bias is
    # GROUPED (G > 1) -- see `_bacc`.
    gbb_ptr, gbc_ptr, gbgB_ptr, gbgC_ptr,
    Bs, H, P, N, L, K, NBLK, CKS,
    BLOCK_L: gl.constexpr, BLOCK_P: gl.constexpr, NUM_WARPS: gl.constexpr,
    HAS_ENDPOINT: gl.constexpr = False, ENDPOINT_P: gl.constexpr = False,
    TP_P: gl.constexpr = 8, ABL: gl.constexpr = 0, WP: gl.constexpr = 1,
    RUN_P1: gl.constexpr = True, NS: gl.constexpr = 1,
    NP: gl.constexpr = 1,
    ATOMIC_GX: gl.constexpr = False, ATOMIC_BC: gl.constexpr = False,
    BCS: gl.constexpr = 1,
    LWS: gl.constexpr = 0, LAW_VEC: gl.constexpr = 0,
    BC_VEC: gl.constexpr = 0, GX_VEC: gl.constexpr = 0,
    TPL: gl.constexpr = 32,
    ROT: gl.constexpr = False, RUN_TH: gl.constexpr = True,
    PAIR_UNROLL: gl.constexpr = False,
    # pair maps (fwd_gluon PMAP): B/C are (B, 2N, L), gate and biases folded
    # in, and gB_t/gC_t are the adjoints of the MAPS, (slot, B, 2N, L).
    PMAP: gl.constexpr = False,
    # PRAW: an UNGATED rotating launch on the same pv path; B/C are the plain
    # (bias-folded) (B, N, L) maps, the pair's raw rows are loaded once per
    # pair and the map adjoints are summed over the pair before one put per
    # row -- the in-kernel coupled un-rotation, without the replicated layout.
    PRAW: gl.constexpr = False,
    # compiled with LLVM SLP forced on (`slp.py`); signature-only
    SLP: gl.constexpr = False,
    # PMAP's p-free work runs in `pv`, an UNREPLICATED 1-D layout over the
    # chunk (lanes on time, as `tw` has its warps): a [BLOCK_L] vector in `lt`
    # is replicated across the TP_P lane groups, so every rotation op and
    # every live rotation vector cost TP_P times its size there.  Only the
    # rotated B/C enter the tile body (convert to `lt`) and only the
    # p-contracted gB/gC come back.  VEC_V/TPL_V are `_put`'s width and
    # owner-lane count in `pv`.
    VEC_V: gl.constexpr = 0, TPL_V: gl.constexpr = 32,
    FAST_TRIG: gl.constexpr = False,
    # SSD law: per-token increment theta[k,t] * Delta[h,t]
    # (THETA_DELTA) and a static READ-side offset theta_h[h,k] (HAS_THH):
    # C rotates by Theta + theta_h, B by Theta.  Gradients: gtheta gets the
    # Delta factor, gde gets sum_k theta_k * gTh_k, gtheta_h the time sum of
    # the C-side half of gTh.
    THETA_DELTA: gl.constexpr = False, HAS_THH: gl.constexpr = False, THETA_PH: gl.constexpr = False,
    LAW_MASK: gl.constexpr = 1, DYN_ACT: gl.constexpr = 0,
    DYN_EPS: gl.constexpr = 0.0, A_DYN_N_RES: gl.constexpr = True,
    MU_N_RES: gl.constexpr = True, HAS_MU_INIT: gl.constexpr = False,
    MU_PREDIFF: gl.constexpr = False,
    DELTA_SOFTPLUS: gl.constexpr = True, HAS_DBIAS: gl.constexpr = False,
    # ── `delta_bias`'s gradient, out of THIS pass' partials ──
    # `delta_bias` is a length-H vector added to `delta` before the
    # softplus, so its gradient is `sum_{b,p,t} dL/d(delta)` -- the very
    # tile this kernel is about to store to `gde`.  Taking it here replaces a
    # full `aten::sum` over the [B, H, L] gradient on the host: one [BLOCK_P]
    # register accumulator carried across the chunk loop and ONE plain store
    # per (CTA, b) at the end.
    #
    # Plain store, NOT an atomic: the buffer keeps the same (slot, B, H)
    # shape the `gde` slot axis has, and the host sums it in fixed order.
    # So unlike K-C's `_bacc` this accumulator is deterministic by
    # construction and owes the pin nothing -- the pin covers it
    # because it covers every host slot sum already.
    #
    # Only legal where the STORED `gdev` is already the final dL/d(delta):
    # under `DSIG` the kernel applied the softplus derivative itself, and
    # with `DELTA_SOFTPLUS` off there is none to apply.  With a HOST
    # sigmoid still to come (`DELTA_SOFTPLUS and not DSIG`) the partial
    # would be the pre-sigmoid one, and the launcher refuses.
    DBACC: gl.constexpr = False,
    # ── the SwiGLU output gate's ADJOINT, in this pass ─────
    # The forward stores `y = (SSM(x) + D_h x) * silu(z)`, so this pass
    # owes two things and both are due where `gy` is read:
    #
    #   dL/dy_pre = gy * silu(z)   -- so `gyp` is SCALED at its load and
    #                                 every consumer below (gB, gC, gx,
    #                                 gdelta, gD) is unchanged;
    #   dL/dz     = gy * y_pre * silu'(z).
    #
    # `y_pre` is not stored by the forward and is not read back: it is
    # REBUILT here, `sum_n C_n h_n (+ D_h x)`, out of the same replayed
    # `hs` and the same effective `C` that `gC = sum_p gy h` already uses.
    # One extra [BLOCK_P, BLOCK_L] accumulator, no extra traffic and no
    # extra activation held across the caller's forward.  It is exactly as
    # accurate as `gC` is: both read the same replayed state, so a bf16
    # checkpoint (`CKBF`) moves them together and neither is the oracle.
    #
    # That rebuild is a sum over ALL of n, which is why the launcher pins
    # `NS = 1` under a gate (an n-shard holds one SUMMAND of `y_pre`) and
    # refuses `n_off`/`n_rows` outright -- the same ownership rule K11's
    # D skip states, one step stronger because a shard cannot own a
    # product it only has a term of.
    HAS_Z: gl.constexpr = False,
    HAS_BB: gl.constexpr = False, HAS_BC: gl.constexpr = False,
    HG_BC: gl.constexpr = 1, HAS_GHL: gl.constexpr = False,
    # K-C: accumulate this bias' gradient in-kernel instead of reading it
    # back off the `H*NP` BC slot axis (which is then free to fold).
    BACC_B: gl.constexpr = False, BACC_C: gl.constexpr = False,
    BACC_GB: gl.constexpr = False, BGPART: gl.constexpr = False, BACC_GC: gl.constexpr = False,
    HAS_DW: gl.constexpr = False, DW_P: gl.constexpr = False,
    N_OFF: gl.constexpr = 0, N_ROWS: gl.constexpr = 0,
    ROT_MIX: gl.constexpr = False,
    HAS_GB: gl.constexpr = False, HAS_GC: gl.constexpr = False,
    GATE_ACT: gl.constexpr = 1, HAS_BGB: gl.constexpr = False,
    HAS_BGC: gl.constexpr = False, HG_G: gl.constexpr = 1,
    EVEN_L: gl.constexpr = False,
    # ── K11: the D skip's adjoint, at the gy LOAD ────────────────────
    # `y = SSM(x) + D_h x` gives  gx += D_h gy  and  gD_h = sum gy x.
    # Both operands are already in registers here (`gyp` and `xt` are
    # n-free and loaded once per chunk), so the whole adjoint is one FMA
    # on the tile that is about to be stored plus one scalar atomic.
    HAS_DSKIP: gl.constexpr = False,
    # ── K14: the packed small-P schedule ─────────────────────────────
    # HP consecutive HEADS share one CTA and one tile axis.  See the
    # "PACKED SMALL-P SCHEDULE" paragraph of the docstring.
    HPACK: gl.constexpr = False, HP: gl.constexpr = 1,
    BC_HEAD: gl.constexpr = False,
    # Take the softplus' own derivative here instead of on the host.  A
    # constexpr and not unconditional: it keeps one more [BLOCK_L] value live
    # across the whole n loop, which this kernel pays for in spill, and the
    # host pass it deletes is only worth having when `gdelta` is big -- at
    # P == 1 it is (B,D,L), at P = 16 it is 1/16 of that and the
    # register costs more than the pass.
    DSIG: gl.constexpr = False,
    # `hck`'s layout, which the FORWARD fixes when it hands one over: packed
    # (B,CKS,N,D) at P == 1, (B,H,CKS,N,P) otherwise.  Separate from HPACK
    # because a packed checkpoint is legal with any pack width, HP == 1
    # included.
    CKPACK: gl.constexpr = False,
    # ── the gtheta slot axis, folded ────────────────────────────
    # `gth_s` is the ONE rotation buffer that keeps a per-token L axis, and
    # its slot axis is the grid's head axis, `H*NP/HP` slabs of (B, K, L).
    # A rotating launch cannot be packed (`hpack_ok` refuses `theta`), so at
    # P = 1 that is H slabs of (B, K, L), zeroed, written and read back once
    # each per call.  `out["theta"]` is the sum over exactly
    # that axis, so it folds the same way `BCS` folds gB_t/gC_t: THS slabs
    # with `% THS` spreading consecutive heads, and L2 doing the rest of the
    # sum.  `out["theta"]` is the ONLY reader of the axis, so the fold is
    # legal wherever the fp32 vector red is.
    ATOMIC_TH: gl.constexpr = False, THS: gl.constexpr = 1,
    TH_VEC: gl.constexpr = 0,
    # TS (`ts_adjoint.py`): grid dim 2 is NS * TS; super-chunk
    # `ts` walks chunks [ts*NBS, min(NBLK, (ts+1)*NBS)), NBS = cdiv(NBLK, TS)
    # (derived here, not an argument, so the launch ABI is unchanged),
    # seeded from `alam[ts]`.
    TS: gl.constexpr = 1,
):
    """grid (H*NP/HP, B, NS): at NP=1 the whole of P lives in one CTA.

    p is the tile axis and n the sequential loop, so `gu = sum_n lam*w` is a
    plain register accumulation into one live [BLOCK_P, BLOCK_L] tile, and the
    p-contractions it trades for stay inside a warp. `gc1`'s reduce over time
    yields all BLOCK_P results at once.

    Chunk is the OUTER loop and n the inner one. That nesting is forced: with
    chunk outer, `gu` is a single live tile flushed to `gx` per chunk, and the
    only thing that must survive the n loop is a [BLOCK_P] carry per n -- which
    goes to global scratch, since Gluon's shared-memory `.slice` needs a
    constexpr index and `n` is a runtime value. The other nesting (n outer)
    would make the carries free but `gu` would need [BLOCK_P, L].

    Every per-(chunk, n) buffer is therefore laid out p-FASTEST -- hck as
    (B, H, NBLK+1, N, P), alam/gc1/ghi as (B, H, N, P), A_log as (H, N, P).
    In the contract's (.., P, N) order a [BLOCK_P] access strides by N and
    costs 32 sectors instead of one, and there are six such accesses per
    (chunk, n). The launcher permutes gc1/ghi back on the host.

    ROT rotates rows 2k and 2k+1 into each other by Theta. The whole
    of it is p-FREE, and this layout makes that cheap:

      * n is the sequential loop, so the partner row n^1 is just a second
        [BLOCK_L] load off the same pointer (no `gl.gather` out of a tile).
      * the rotation adjoint acts on gB/gC AFTER they have been contracted over
        p, so it runs on [BLOCK_L] vectors -- a handful of FMAs.

    PACKED SMALL-P SCHEDULE (`HPACK`).  Everything above is written for
    a CTA that owns one head, and it prices the n loop's p-FREE work -- the
    B/C loads, the reversed C, the two [BLOCK_L] gB/gC stores -- once per
    (b, h).  That work is `O(N*L)` while a head's own state work is
    `O(P*N*L)`, so at P = 1 the two are equal and structure B's whole premise
    (share the p-free work across the tile) buys nothing: the backward
    re-reads all of B/C, and re-issues all of gB/gC's atomics, D times
    instead of D/P times.  `HPACK` restores the sharing by putting HP
    consecutive HEADS on the tile axis instead of P channels.  It is exact,
    not an approximation, and it is legal precisely because at P == 1 every
    per-head tensor already has the layout the p tile wants:

      * `x`, `gy`, `gx`, `delta`, `delta_w`, `gdelta` are all channel-major
        with time fastest, and at P == 1 a head IS a channel, so `po` -- the
        index the x load already builds -- addresses all of them unchanged;
      * `B_t`, `C_t` and the gates are (B,N,L), head-free by invariant (III),
        so they are shared across the tile exactly as they were across p;
      * the per-(h,n) statics and scratch (`A_log`, `mu_gain`,
        `hck`, `alam`, `gc1`, `ghi`) are re-laid out CHANNEL-FASTEST by the
        launcher -- (N,D) and (B,[CKS,]N,D) -- so a [BLOCK_P] access is one
        sector here just as it is at HP = 1.

    What changes inside: `delta` becomes a tile (`_de_load_t`) instead of a
    broadcast vector, `gdelta` stays a tile instead of being p-reduced, and
    the row stride of the four channel-fastest buffers is `D` rather than
    `P` (`sn`).  The gB/gC contraction `gl.sum(..., axis=0)` now sums over
    HEADS, which is exactly the axis those adjoints contract over anyway.

    Packing requires P == 1 and LAW_A. Real calls retain the existing bias
    group constraints. Rotating calls require saved packed checkpoints and
    no B/C bias, endpoint correction or output gate. Their clocks, phase
    carries and gate biases vary across the packed heads; inverse rotation
    happens BEFORE summing map/gate adjoints across that axis. Other feature
    combinations use the ordinary schedule.

    The cost is that the n loop must walk PAIRS, because the adjoint of row 2k
    needs the reduced gB/gC of row 2k+1. Small and packed tiles unroll
    the pair; larger rotating tiles reuse a runtime loop body. The cost of
    the pair walk is live ranges, not arithmetic, so only gB0/gC0 are carried
    across the pair; the even row's rotated B/C and its store offset are all
    recoverable at pi=1 from values already live there.
    """
    # PMAP + `pv`.  A gated rotating launch reads per-row pair
    # maps (fwd_gluon PMAP): row n = al P0 + be P1 with (al, be) = (c, -s) on
    # the even row and (s, c) on the odd one, so dL/dP0 = al gB and
    # dL/dP1 = be gB go out per ROW and nothing crosses the pair but the two
    # Theta sums; the gate is in the maps (no activation, no derivative, no
    # ggB/ggC scatter).  All of that p-free work runs in `pv`, which holds a
    # [BLOCK_L] vector unreplicated (BLOCK_L/32 per lane at one warp) where
    # `lt` repeats it across the TP_P lane groups: at TP_P=4 every rotation
    # vector cost 4x its registers and instructions.
    tw: gl.constexpr = _lay3(BLOCK_P, BLOCK_L, NUM_WARPS, TP_P, WP)
    WOWN: gl.constexpr = NUM_WARPS // WP if WP > 1 else 0
    lt: gl.constexpr = gl.SliceLayout(0, tw)
    pl: gl.constexpr = gl.SliceLayout(1, tw)
    row1: gl.constexpr = _row1(tw)
    pv: gl.constexpr = _lay_pv(BLOCK_L, NUM_WARPS)
    HAS_DYN: gl.constexpr = (LAW_MASK >> 1) & 1
    HAS_MU: gl.constexpr = (LAW_MASK >> 2) & 1
    # LAW_A alone leaves s1 = de, which is n-free, and every law branch below
    # folds away.
    PLAIN: gl.constexpr = LAW_MASK == 1
    # one bit set = every row is on the same law, so the runtime test is dead
    # weight and the branches below fold away. The mixture is the only case
    # that has anything to skip.
    UNI: gl.constexpr = (LAW_MASK & (LAW_MASK - 1)) == 0
    # the n loop steps by pairs under ROT and by single rows otherwise, so one
    # loop body serves both; NPAIR=1 makes `kk` the row index `n` itself.
    # ABL 512 walks pairs with NO rotation: same rows, same answers, only
    # the unrolled loop STRUCTURE differs, so it prices the pair walk alone.
    NPAIR: gl.constexpr = 2 if (ROT or (ABL & 512)) else 1
    INV_BP: gl.constexpr = 1.0 / BLOCK_P

    # NP shards the p AXIS across CTAs, BLOCK_P = P // NP. It is the second grid
    # knob and it rides grid dim 0 rather than dim 2 so that the shards of one
    # (h, b) launch CONSECUTIVELY: they read identical B/C/delta/theta rows, and
    # adjacent CTAs get those out of L2 where H*Bs-apart ones would not.
    # HPACK: grid dim 0 counts HEAD GROUPS, so the base head of this CTA is
    # the group index times the pack width (NP == 1 is forced with HP > 1).
    h = (gl.program_id(0) // NP) * HP
    pid_p = gl.program_id(0) % NP
    b = gl.program_id(1).to(gl.int64)
    bh = b * H + h
    # the angle feature's row base: (B,K,L) head-shared, or (B,H,K,L) per
    # head under THETA_PH (theta_per_head)
    thb = (bh if THETA_PH else b) * K
    if BC_HEAD:
        B_ptr += h.to(gl.int64) * Bs * N * L
        C_ptr += h.to(gl.int64) * Bs * N * L
    gbc = h // HG_BC            # beta_b/beta_c group of this head
    gg = h // HG_G              # beta_gB/beta_gC group, which need not match
    # NS shards the n loop over an NS-times-larger grid, which is what long L
    # is short of: at the reference shape (H, B) is only 512 CTAs on 110 SMs.
    # Everything keyed by n stays exact; gx and gde contract over n, so they
    # go to slot buffers (slot = shard) summed in fixed host order.
    sh = gl.program_id(2) // TS
    ts = gl.program_id(2) % TS
    # a partial launch shards [N_OFF, N_OFF+N_ROWS) rather than all of N; `n`
    # below stays ABSOLUTE, so every buffer offset is unchanged and the rows
    # outside the window are simply never visited.
    n_lo = N_OFF + sh * (N_ROWS // NS)
    k_lo = n_lo // NPAIR
    k_hi = (n_lo + N_ROWS // NS) // NPAIR
    gxb = sh.to(gl.int64) * Bs * H * P * L
    # Everything a p-shard CONTRACTS over p becomes a partial, so it needs a
    # slot of its own the way an n-shard does. Two slot axes carry that: `gdb`
    # for the n-shard-keyed buffers (gde/gdw/grate/gs2o, slot = sh*NP + pid_p)
    # and `hslot` for the h-keyed ones (gB_t/gC_t/ggB/ggC/gth and the n-resolved
    # law slots, slot = h*NP + pid_p). Both are summed on the host by a pass
    # that already existed -- the p axis just rides the existing slot axis, so
    # NP costs no new reduction, only NP times the slot memory. Buffers indexed
    # BY p (x_s, gdw_s under DW_P, hck, alam, gc1, ghi, rcar) keep `gxb`: each
    # shard owns its own p slice of them outright and nothing is contracted.
    gdb = (sh.to(gl.int64) * NP + pid_p) * Bs * H * L
    # the same slot, on the L-free (slot, B, H) `delta_bias` partials
    gdbh = ((sh.to(gl.int64) * TS + ts) * NP + pid_p) * Bs * H
    # ATOMIC_GX collapses the NS gx slabs onto one, so every shard targets the
    # same address and the cross-shard sum happens in L2 instead of in a host
    # pass over DRAM. gdw keeps `gxb` -- only gx changes owner.
    gxo = gxb * (0 if ATOMIC_GX else 1)

    col = gl.arange(0, BLOCK_L, layout=lt)
    idx_rev = BLOCK_L - 1 - col
    # column c holds time BLOCK_L-1-c, so a_{t+1} sits at c-1; c=0 is clamped
    # because the reverse scan never reads it.
    idx_nxt = gl.minimum(BLOCK_L - col, BLOCK_L - 1)
    zl = gl.zeros([BLOCK_L], gl.int32, layout=lt)
    z2 = gl.zeros([BLOCK_P, BLOCK_L], gl.float32, layout=tw)
    rev2 = gl.expand_dims(idx_rev, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int32,
                                                 layout=tw)
    # one-column-earlier index, clamped: Pass 1's reverse-exclusive-cumsum
    # (the `_tail_sum` note below `if PLAIN`) shifts a reversed inclusive scan
    # by this instead of subtracting, so the boundary is the literal additive
    # identity rather than a self-cancellation.
    if HPACK:
        # HPACK's `de` is a real tile, so its one-column shift is a tile
        # gather; the unpacked path shifts the [BLOCK_L] vector instead and
        # never builds this.
        nxt2 = gl.expand_dims(idx_nxt, 0) + gl.zeros(
            [BLOCK_P, BLOCK_L], gl.int32, layout=tw)
    idx_prv = gl.maximum(col - 1, 0)
    prev2 = gl.expand_dims(idx_prv, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int32,
                                                   layout=tw)
    # The adjoint scan's decay tile IS the h scan's one column over -- both are
    # exp2 of the same [BLOCK_P, 1] coefficients against the same driver, and
    # `an[p, c] == a[p, idx_nxt[c]]` holds bit for bit, c == 0 included. So the
    # second exp2 below can be deleted in favour of `gl.gather(a, nxt2, 1)` on a
    # 2-D form of idx_nxt. That is exact but slower and spills more, because
    # the index tile has to stay live: a full-tile axis-1 gather is dearer here
    # than a full-tile MUFU plus the multiply that builds its exponent.
    first2 = gl.expand_dims(col == 0, 0) + gl.zeros([BLOCK_P, BLOCK_L],
                                                    gl.int1, layout=tw)
    # in the time-reversed tile the LAST column is the chunk's first step, so
    # this is where the Theta adjoint's carry into chunk j-1 comes from.
    last2 = gl.expand_dims(col == BLOCK_L - 1, 0) + gl.zeros(
        [BLOCK_P, BLOCK_L], gl.int1, layout=tw)

    pofs = gl.arange(0, BLOCK_P, layout=pl)
    # `pofs` stays LOCAL (it is a lane predicate for thck) and only `pf` carries
    # the shard base, so every p-indexed buffer below is addressed globally and
    # none of them changes shape with NP.
    pf = (pid_p * BLOCK_P + pofs).to(gl.int64)
    zp = gl.zeros([BLOCK_P], gl.int32, layout=pl)
    hp = bh * P + pf                        # (B, H, P, L) tensors: x, gy, gx
    # Row stride of the four channel-fastest buffers.  Unpacked they are
    # (H,N,P) / (B,[CKS,]N,P) and consecutive n is P apart; packed they are
    # (N,D) / (B,[CKS,]N,D) and it is D, because the tile now spans heads.
    sn = H * P if HPACK else P
    # `hck`'s own row stride: its layout is the FORWARD's when the forward
    # handed one over, so it is CKPACK's and not HPACK's.
    snc = H * P if CKPACK else P
    if HPACK:
        # `pf` is the channel offset inside the pack (P == 1, NP == 1), so
        # `h + pf` is the absolute channel and `hp` is unchanged.
        anp = h.to(gl.int64) + pf           # A_log as (N, D)
        cyp = b * N * sn + h.to(gl.int64) + pf           # alam/gc1/ghi (B,N,D)
        cyp += ts.to(gl.int64) * Bs * N * sn             # TS slab (ts == 0 at j == 0)
    else:
        anp = h.to(gl.int64) * N * P + pf   # A_log as (H, N, P)
        cyp = bh * N * P + pf               # alam / gc1 / ghi as (B, H, N, P)
        cyp += ts.to(gl.int64) * Bs * H * N * P          # TS slab
    if CKPACK:
        ckp = b * CKS * N * snc + h.to(gl.int64) + pf    # hck (B,CKS,N,D)
    else:
        ckp = bh * CKS * N * P + pf         # hck as (B, H, CKS, N, P)
    rcp = (bh + pf) * K * P if HPACK else bh * K * P + pf                   # rcar as (B, H, K, P)
    # h-keyed slot, widened by the p shard.  Spelled as the grid index rather
    # than `h * NP + pid_p` (the same number at HP == 1) so that under HPACK
    # it counts HEAD GROUPS: `% BCS` must spread CONSECUTIVE CTAs across the
    # slabs, and packed head bases step by HP, which `% BCS` would alias.
    hslot = gl.program_id(0).to(gl.int64)
    hbn = (hslot * Bs + b) * N * L          # n-resolved slots as (H*NP,B,N,L)
    # Dynamic-rate and mu features are head-shared even when rotated B/C
    # maps are not. Reduce their adjoints without retaining one slab/head.
    lawbn = ((hslot % LWS) * Bs + b) * N * L if LWS else hbn
    # gB_t/gC_t take a CAPPED slot axis of their own: BCS slabs instead of
    # H*NP, with `% BCS` spreading consecutive heads across them so the
    # contention per address is H*NP/BCS rather than all of it. The two indices
    # coincide at BCS == H*NP, which is the default and the no-atomic path.
    # under PMAP the adjoint rows are the (B, 2N, L) map's
    NMR = N * 2 if (PMAP and not PRAW) else N
    if ATOMIC_BC:
        # Prepared maps retain a head axis. Only the P partitions of the
        # same head may share an accumulator; modulo BCS would mix heads.
        bcn = ((h.to(gl.int64) if BC_HEAD else hslot % BCS) * Bs + b) * NMR * L
    elif PMAP:
        bcn = (hslot * Bs + b) * NMR * L
    else:
        bcn = hbn
    # K18: same rule, same reason, for the gtheta slot axis (see ATOMIC_TH).
    if ATOMIC_TH:
        thn = ((hslot % THS) * Bs + b) * K
    else:
        thn = (hslot * Bs + b) * K

    # Keep the compact P=1/packed body unrolled. Larger rotating tiles
    # benefit from reusing the pair body instead of duplicating its live
    # values. `.value` below exposes the iterator class to Gluon's visitor.
    # PAIR_UNROLL pins the unrolled body at any tile (a config knob).
    pair_iterator: gl.constexpr = gl.static_range if (
        not ROT or HPACK or BLOCK_P < 4 or PAIR_UNROLL) else loop_range

    # Pass 0 (ROT): Theta at every chunk boundary. Theta is a plain prefix sum
    # of a p-free increment, so this touches `theta` and nothing else -- cheap
    # enough that the backward can rebuild it rather than have the forward hand
    # it over the way it hands over `hck`. RUN_TH=False when it does.
    if ROT and RUN_TH:
        for kk in range(k_lo, k_hi):
            thc = gl.load(thi_ptr + (bh + pf if HPACK else bh + zp) * K + kk)
            for j in range(NBLK):
                offs_t = j * BLOCK_L + col
                m2 = gl.expand_dims(offs_t < L, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1, layout=tw)
                gl.store(thck_ptr + ((bh + pf if HPACK else bh + zp) * NBLK + j) * K + kk,
                         thc, mask=gl.full((BLOCK_P,), True, gl.int1, layout=pl) if HPACK else pofs == 0)
                feat = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_t, offs_t < L, EVEN_L).to(gl.float32)
                inc = gl.expand_dims(feat, 0) + z2
                if THETA_DELTA:
                    if HPACK:
                        ptime = gl.expand_dims(hp,1) * L + gl.expand_dims(offs_t,0)
                        de0, _ = _de_load_t(delta_ptr, delta_bias_ptr, h, pofs, ptime, m2,
                                           DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L)
                    else:
                        dev, _ = _de_load(delta_ptr, delta_bias_ptr, bh, h, L, offs_t,
                                         offs_t < L, zl, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L)
                        de0 = gl.expand_dims(dev,0) + z2
                    inc *= de0
                thc += gl.sum(_zt(inc,m2,EVEN_L),axis=1)
                thc -= TWO_PI * libdevice.floor(thc * INV_TWO_PI)

    de = zl.to(gl.float32)  # packed rotation requires saved boundaries
    # Pass 1: h at every chunk boundary. Slot 0 stays the zero h_init, so only
    # 1..NBLK are written and the buffer never needs re-zeroing between calls.
    for j in range(NBLK if RUN_P1 else 0):
        offs_t = j * BLOCK_L + col
        mask_t = offs_t < L
        m2 = gl.expand_dims(mask_t, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1,
                                                  layout=tw)
        po = gl.expand_dims(hp, 1) * L + gl.expand_dims(offs_t, 0)
        if HPACK:
            de2, _ = _de_load_t(delta_ptr, delta_bias_ptr, h, pofs, po, m2,
                                DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L)
        else:
            de, _ = _de_load(delta_ptr, delta_bias_ptr, bh, h, L, offs_t,
                             mask_t, zl, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L)
            de2 = gl.expand_dims(de, 0) + z2
        u = _dw_load(dw_ptr, de2, bh, L, po, offs_t, mask_t, m2, z2,
                     DELTA_SOFTPLUS, HAS_DW, DW_P, EVEN_L, HPACK) \
            * _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)

        # Only the LAST h of the chunk is wanted, and log a is separable, so
        # with S_t = sum_{s<=t} de_s and S_L its total,
        #   h_end = exp(c1 S_L) h_prev + sum_t exp(c1 (S_L - S_t)) w_t u_t.
        # `de` is p- and n-free, so that costs ONE scan per chunk plus a plain
        # commutative reduce per n, instead of an SSM scan per (chunk, n).
        #
        # `S_L - S_t` (`gap2` below, `_tail_sum`'s second return) is
        # mathematically >= 0 for every t (c1 < 0, de >= 0).  Building it as a
        # `gl.sum` reduction for S_L minus a separate forward scan's running
        # total for S_t would use two different reduction trees over the same
        # values, free to disagree at the bit -- including in SIGN at
        # t == L-1, where the exact value is 0.0.  At A_log = 40,
        # c1 ~ -2.4e17, so a ULP-scale positive residual times c1 is a large
        # POSITIVE exponent: `exp2` gives +-inf and poisons every gradient
        # that reads this chunk's checkpoint as its `hprev` (non-finite
        # backward at BLOCK_L in {8,16}; only a multi-chunk launch ever reads
        # a checkpoint this pass wrote).
        #
        # `_tail_sum` is difference-first instead: see its docstring.  It
        # costs one extra reverse gather and one extra shift-gather, both
        # cheap next to the scan itself, and the finite-regime numerics are
        # unchanged.
        # on the 2-D tile, not the [BLOCK_L] vector: a scan along a SliceLayout
        # axis fails to lower ("PassManager::run failed" in ConvertToLLVM).
        #
        # Under any law but LAW_A the driver depends on n, so the scan moves
        # into the n loop and the trick keeps only its second half. That is the
        # right place to pay it: this pass exists as the fallback for when the
        # forward did not hand over checkpoints, and RUN_P1=False deletes it.
        if PLAIN:
            sL, gap2 = _tail_sum(de2, rev2, prev2, first2, last2)

        for kk in range(k_lo, k_hi):
            if ROT:
                cT, sT = _rot_cs(theta_ptr, thck_ptr, thb, h, bh, kk, K,
                                 L, NBLK, j, offs_t, col, m2, z2, zl, de,
                                 INV_BP, ABL, FAST_TRIG, EVEN_L,
                                 THETA_DELTA, row1)
                if ROT_MIX:
                    rf = _rot_sel(row_rot_ptr, kk, NPAIR)
                    cT, sT = rf * cT + (1.0 - rf), rf * sT
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
                    w = _bc_load(B_ptr, beta_b_ptr, bc, n, gbc, N, mask_t, zl,
                                 HAS_BB, EVEN_L)
                if ROT and not PMAP:
                    np_ = kk * NPAIR + 1 - pi
                    pc = b * N * L + np_.to(gl.int64) * L + offs_t
                    Bp = w if (ABL & 128) else _bc_load(
                        B_ptr, beta_b_ptr, pc, np_, gbc, N, mask_t, zl,
                        HAS_BB, EVEN_L)
                    w = w * cT - sT * Bp if pi == 0 else w * cT + sT * Bp
                if HAS_GB:
                    # the gate lands on the ROTATED B, so it
                    # goes here and not inside _bc_load the way the bias does.
                    w = w * _gate_load(gB_ptr, bgB_ptr, bc, n, gg, N, mask_t,
                                       zl, GATE_ACT, HAS_BGB, EVEN_L)
                # thin, not `+ z2`: its only use broadcasts. s1_2/s2_2 below
                # cannot follow -- they are SCANNED along axis 1, and a scan on
                # a SliceLayout axis does not lower.
                w2 = gl.expand_dims(w, 0)
                if PLAIN:
                    c1v = -gl.exp(gl.load(A_log_ptr + anp + n * sn))
                    ltot = c1v * sL
                    lgap = gl.expand_dims(c1v, 1) * gap2
                else:
                    law_s = gl.load(row_law_ptr + n)
                    c1v = -gl.exp(gl.load(A_log_ptr + anp + n * sn))
                    s1, _ = _law_s1(a_dyn_ptr, law_s, de, b, bh, n, N, L,
                                    offs_t, mask_t, zl, HAS_DYN,
                                    UNI, DYN_ACT, DYN_EPS, A_DYN_N_RES,
                                    EVEN_L)
                    s1_2 = gl.expand_dims(s1, 0) + z2
                    # `_tail_sum` (see its docstring, and the note above
                    # `if PLAIN`): a `total - forward_cumsum` split needs the
                    # two to agree to the bit at t == L-1, and at c1 ~ -2.4e17
                    # (A_log = 40) even a 1-ULP disagreement turns that
                    # exponent positive and `exp2` overflows to +-inf instead
                    # of landing on 1.0. s1 >= 0 always (LAW_A/DYN), so
                    # the reverse/shift construction gives lgap <= 0 for
                    # every t, not merely the one a separate reduction agrees
                    # with.
                    S1L, s1_gap = _tail_sum(s1_2, rev2, prev2, first2, last2)
                    ltot = c1v * S1L
                    lgap = gl.expand_dims(c1v, 1) * s1_gap
                    if HAS_MU and (UNI or law_s == K_MU):
                        c2v = gl.load(mu_gain_ptr + anp + n * sn)
                        s2_2 = gl.expand_dims(
                            _law_s2(mu_ptr, mu_init_ptr, b, bh, n, N, L,
                                    offs_t, mask_t, col, zl, MU_N_RES,
                                    HAS_MU_INIT, MU_PREDIFF, EVEN_L),
                            0) + z2
                        # s2 is signed by design (mu-split)
                        # so this tail is not sign-guaranteed the way s1_gap
                        # is -- but it is still exact at t == L-1 (the literal
                        # 0.0 identity, not a cancellation), which is what
                        # this pass needs.
                        S2L, s2_gap = _tail_sum(s2_2, rev2, prev2, first2,
                                                last2)
                        ltot = ltot + c2v * S2L
                        lgap = lgap + gl.expand_dims(c2v, 1) * s2_gap
                ck = hck_ptr + ckp + (j * N + n) * snc
                # the chain reads back its own rounded value, so under a bf16
                # hck the rounding compounds down the chunk chain -- but a is
                # a decay, so the recursion contracts and the error stays at
                # one rounding's relative size (the bf16 INPUTS already cost
                # more).
                gl.store(ck + N * snc,
                         (libdevice.exp2(ltot * LOG2E)
                          * gl.load(ck).to(gl.float32)
                          + gl.sum(libdevice.exp2(lgap * LOG2E) * w2 * u,
                                   axis=1)).to(hck_ptr.dtype.element_ty))

    # H3 item 2: `delta_bias`'s gradient rides the chunk loop.  One
    # [BLOCK_P] fp32 accumulator -- one entry per head under HPACK, one
    # live lane otherwise -- against a 67 MB `aten::sum` on the host.
    if DBACC:
        dba = gl.zeros([BLOCK_P], gl.float32, layout=pl)
    # Pass 2: reverse chunk order, the real work.
    NBS = (NBLK + TS - 1) // TS
    j_hi = gl.minimum((ts + 1) * NBS, NBLK)
    for jj in range(j_hi - ts * NBS):
        j = j_hi - 1 - jj
        offs_t = j * BLOCK_L + col
        mask_t = offs_t < L
        m2 = gl.expand_dims(mask_t, 0) + gl.zeros([BLOCK_P, BLOCK_L], gl.int1,
                                                  layout=tw)
        # `jj > 0` as a [BLOCK_P] predicate: the chunk processed first has no
        # incoming adjoint and no gc1 to accumulate onto, so both loads are
        # masked off rather than requiring the buffers to be pre-zeroed.
        carry = (gl.zeros([BLOCK_P], gl.int32, layout=pl) + jj) > 0
        # gh_final is delivered by pre-seeding alam, so with it that one load is
        # live from the first chunk on -- but only that one: gc1/gc2/rcar
        # accumulate and their `carry` is still "has this slot been written".
        lcar = ((gl.zeros([BLOCK_P], gl.int32, layout=pl) + jj) >= 0
                if (HAS_GHL or TS > 1) else carry)
        at_t0 = (gl.zeros([BLOCK_P], gl.int32, layout=pl) + j) == 0
        po = gl.expand_dims(hp, 1) * L + gl.expand_dims(offs_t, 0)
        if HPACK:
            de2, dsig = _de_load_t(delta_ptr, delta_bias_ptr, h, pofs, po, m2,
                                   DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L, DSIG)
            # one column over, per row: the [BLOCK_L] shift the unpacked
            # path takes on the vector has to be a tile gather here, because
            # the rows no longer hold the same value.
            den2 = gl.gather(de2, nxt2, 1)
        else:
            de, dsig = _de_load(delta_ptr, delta_bias_ptr, bh, h, L, offs_t,
                                mask_t, zl, DELTA_SOFTPLUS, HAS_DBIAS, EVEN_L,
                                DSIG)
            de2 = gl.expand_dims(de, 0) + z2
            den2 = gl.expand_dims(gl.gather(de, idx_nxt, 0), 0) + z2

        # x and gy are n-free: loaded once for the whole n loop.
        xt = _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        gyp = _ldt(gy_ptr + po, m2, EVEN_L).to(gl.float32)
        if HAS_Z:
            # sigma = 1/(1+exp(-z)), silu = z*sigma, silu' = sigma*(1 +
            # z(1-sigma)) -- `tails._gate_tr_bwd`'s expressions, through
            # the HARDWARE exponential (`common_gluon._silu_fast`).
            zt = _ldt(z_ptr + po, m2, EVEN_L).to(gl.float32)
            zsg = 1.0 / (1.0 + libdevice.exp2(-zt * LOG2E))
            # dL/d(y_gated) has to survive to the dz store at the end of
            # the chunk; `gyp` itself becomes dL/dy_pre right here so that
            # not one consumer below has to know about the gate.
            gyz = gyp
            gyp = gyp * (zt * zsg)
        gyr = gl.gather(gyp, rev2, 1)
        dw2 = _dw_load(dw_ptr, de2, bh, L, po, offs_t, mask_t, m2, z2,
                       DELTA_SOFTPLUS, HAS_DW, DW_P, EVEN_L, HPACK)
        u = dw2 * xt

        gu = z2
        if HAS_ENDPOINT:
            if HPACK or ENDPOINT_P:
                eo = _ldt(endpoint_ptr + po, m2, EVEN_L).to(gl.float32)
            else:
                eo = gl.expand_dims(_ldt(endpoint_ptr + bh * L + offs_t, mask_t, EVEN_L).to(gl.float32), 0)
            # B/C are p-free. One vector accumulator, no state-sized tape
            # and no extra live P x T accumulator in this register-heavy loop.
            endpoint_dot = gl.zeros([BLOCK_L], gl.float32, layout=lt)
            # n-free (gy, x and own are all per chunk), so it is contracted
            # over p ONCE here rather than once per row: the diagonal
            # readout own_t B_t.C_t x_t's share of gB / gC.
            local_own = gl.sum(gyp * eo * xt, axis=0)
        if HAS_Z:
            # the pre-gate output, rebuilt: `sum_n C_n h_n` here and the
            # D skip added below, exactly as the forward's store builds it.
            ypre = z2
        # HPACK: gdelta is PER HEAD, and the tile axis is heads, so the
        # p-contraction that produces it unpacked must not happen -- the
        # accumulator stays a tile and is stored channel-major.
        gdev = z2 if HPACK else gl.zeros([BLOCK_L], gl.float32, layout=lt)
        # On LAW_A the delta p-reduction in the n loop below has no other
        # consumer, so the two sums can swap: keep a third [BLOCK_P, BLOCK_L]
        # accumulator across the n loop and contract p ONCE per chunk, folded
        # into the `gl.sum(gu * xt, axis=0)` already there -- N-1 fewer
        # p-reductions for N tile adds. That is correct but slower: the extra
        # live tile adds spill, and at 255 registers a new live tile is never
        # free. Same lesson as the exp2/gather note above.
        #
        # h-resolved law gradients contract over n, so they accumulate here and
        # are stored once per chunk; the n-resolved ones go out per row.
        if HAS_DYN and not A_DYN_N_RES:
            grt_acc = gl.zeros([BLOCK_L], gl.float32, layout=lt)
        if HAS_MU and not MU_N_RES:
            gs2_acc = gl.zeros([BLOCK_L], gl.float32, layout=lt)
        if PMAP:
            offs_v = j * BLOCK_L + gl.arange(0, BLOCK_L, layout=pv)
            mask_v = offs_v < L
            de_v = gl.convert_layout(de, pv)

        for kk in range(k_lo, k_hi):
            if PMAP:
                # the pair prologue in `pv`: the in-chunk Theta prefix is a
                # plain 1-D scan there (only a SliceLayout scan fails to lower)
                thv = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_v,
                           mask_v, EVEN_L).to(gl.float32)
                if THETA_DELTA:
                    thv = thv * de_v
                Thv = gl.associative_scan(_zt(thv, mask_v, EVEN_L), axis=0,
                                          combine_fn=_add) \
                    + gl.load(thck_ptr + (bh * NBLK + j) * K + kk)
                if FAST_TRIG:
                    cT, sT = libdevice.fast_cosf(Thv), libdevice.fast_sinf(Thv)
                else:
                    cT, sT = gl.cos(Thv), gl.sin(Thv)
                if HAS_THH:
                    chh = gl.load(thh_ptr + (h * K + kk) * 2)
                    shh = gl.load(thh_ptr + (h * K + kk) * 2 + 1)
                    cC = cT * chh - sT * shh
                    sC = sT * chh + cT * shh
                else:
                    cC, sC = cT, sT
                if ROT_MIX:
                    rf = _rot_sel(row_rot_ptr, kk, NPAIR)
                    cT, sT = rf * cT + (1.0 - rf), rf * sT
                    cC, sC = rf * cC + (1.0 - rf), rf * sC
            elif ROT:
                if HPACK:
                    feat = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L + offs_t, mask_t, EVEN_L).to(gl.float32)
                    inc = gl.expand_dims(feat,0) + z2
                    if THETA_DELTA:
                        inc *= de2
                    ph = gl.expand_dims(gl.load(thck_ptr + ((bh + pf) * NBLK + j) * K + kk),1)
                    ph += gl.associative_scan(_zt(inc,m2,EVEN_L),axis=1,combine_fn=_add)
                    cT = libdevice.fast_cosf(ph) if FAST_TRIG else libdevice.cos(ph)
                    sT = libdevice.fast_sinf(ph) if FAST_TRIG else libdevice.sin(ph)
                else:
                    cT, sT = _rot_cs(theta_ptr, thck_ptr, thb, h, bh, kk, K,
                                     L, NBLK, j, offs_t, col, m2, z2, zl, de,
                                     INV_BP, ABL, FAST_TRIG, EVEN_L, THETA_DELTA,
                                     row1)
                if HAS_THH:
                    # READ map rotates by Theta + theta_h[h,k]: angle addition
                    # on the phasor, so the scan above is shared by B and C.
                    # (cos, sin) from the forward's host table
                    # (`fwd_gluon._thh_table`), so both passes read one map.
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
                    cC, sC = cT, sT
                if ROT_MIX:
                    rf = _rot_sel(row_rot_ptr, kk, NPAIR)
                    cT, sT = rf * cT + (1.0 - rf), rf * sT
                    cC, sC = rf * cC + (1.0 - rf), rf * sC
            gB0, gC0 = (z2, z2) if HPACK else (zl.to(gl.float32), zl.to(gl.float32))
            if PMAP:
                # the pair's Theta adjoint and its C half (theta_h), summed
                # over both rows and taken after the pair
                gTa = gl.zeros([BLOCK_L], gl.float32, layout=pv)
                gTc = gl.zeros([BLOCK_L], gl.float32, layout=pv)
                if PRAW:
                    # the raw pair, once per pair, and its two rows' map
                    # adjoints summed over the pair (2 registers each here)
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
                    # row n = diag(g) R(Theta) on its pair map rows: with
                    # (al, be) = (c, -s) on the even row and (s, c) on the
                    # odd one, w = al P0 + be P1 and dw/dTheta = be P0 - al P1.
                    # The derivative is taken here so P0/P1 die at once.
                    if PRAW:
                        P0, P1, Q0, Q1 = R0, R1, S0, S1
                    else:
                        mr = (b * 2 * N + 2 * n.to(gl.int64)) * L + offs_v
                        P0 = _ldt(B_ptr + mr, mask_v, EVEN_L).to(gl.float32)
                        P1 = _ldt(B_ptr + mr + L, mask_v, EVEN_L).to(gl.float32)
                        Q0 = _ldt(C_ptr + mr, mask_v, EVEN_L).to(gl.float32)
                        Q1 = _ldt(C_ptr + mr + L, mask_v, EVEN_L).to(gl.float32)
                    if pi == 0:
                        wv, dwt = P0 * cT - sT * P1, -(sT * P0) - cT * P1
                        rv, drt = Q0 * cC - sC * Q1, -(sC * Q0) - cC * Q1
                    else:
                        wv, dwt = P1 * cT + sT * P0, cT * P0 - sT * P1
                        rv, drt = Q1 * cC + sC * Q0, cC * Q0 - sC * Q1
                    w = gl.convert_layout(wv, lt)
                    r = gl.convert_layout(rv, lt)
                else:
                    w = _bc_load(B_ptr, beta_b_ptr, bc, n, gbc, N, mask_t, zl,
                                 HAS_BB, EVEN_L)
                    r = _bc_load(C_ptr, beta_c_ptr, bc, n, gbc, N, mask_t, zl,
                                 HAS_BC, EVEN_L)
                w0, r0 = (z2, z2) if HPACK else (zl.to(gl.float32), zl.to(gl.float32))
                if ROT and not PMAP:
                    # Brot[n] = B[n] cos + sgn[n] B[n^1] sin, sgn = -1 on even
                    # rows. The partner is a second contiguous load.
                    np_ = kk * NPAIR + 1 - pi
                    pc = b * N * L + np_.to(gl.int64) * L + offs_t
                    if ABL & 128:
                        Bp, Cp = w, r
                    else:
                        Bp = _bc_load(B_ptr, beta_b_ptr, pc, np_, gbc, N,
                                      mask_t, zl, HAS_BB, EVEN_L)
                        Cp = _bc_load(C_ptr, beta_c_ptr, pc, np_, gbc, N,
                                      mask_t, zl, HAS_BC, EVEN_L)
                    if HPACK:
                        w, r = gl.expand_dims(w,0) + z2, gl.expand_dims(r,0) + z2
                        Bp, Cp = gl.expand_dims(Bp,0) + z2, gl.expand_dims(Cp,0) + z2
                    if pi == 0:
                        w, r = w * cT - sT * Bp, r * cC - sC * Cp
                    else:
                        # the EVEN row's rotated B/C fall out of the same two
                        # loads, so they need not stay live across the pair --
                        # only gB0/gC0 do, and this kernel spills.
                        w0 = Bp * cT - sT * w
                        r0 = Cp * cC - sC * r
                        w, r = w * cT + sT * Bp, r * cC + sC * Cp
                r = _zt(r, m2 if HPACK and ROT else mask_t, EVEN_L)
                # w/r stay PRE-gate from here on. Both gTh and the gate's own
                # adjoint want the unscaled value while the drive and the lambda
                # recursion want the scaled one, and holding the factor apart
                # costs one [BLOCK_L] vector where holding both products costs
                # two -- which this kernel, already spilling under ROT, has not
                # got. The rotated pair (w0/r0) is pre-gate for the same reason:
                # each row carries its own gate, so it cannot be factored out.
                # `expand_dims` alone, NOT `+ z2`: a [1, BLOCK_L] operand
                # broadcasts against the tile for free in every use below,
                # while adding a zero tile is a real fadd per element -- x+0.0
                # is not x at -0.0, so nothing folds it away. Same cut the
                # forward took (see fwd_gluon), and it matters more here:
                # ncu counts 600 M NON-FUSED fp32 instructions out of 1.564 G.
                if HAS_GB:
                    if HPACK and ROT:
                        raw_gate = gl.expand_dims(_ldt(gB_ptr + bc, mask_t, EVEN_L).to(gl.float32),0) + z2
                        if HAS_BGB:
                            raw_gate += gl.expand_dims(gl.load(bgB_ptr + ((h+pf)//HG_G)*N+n),1)
                        gtB = _gate(raw_gate, GATE_ACT)
                    else:
                        gtB = _gate_load(gB_ptr, bgB_ptr, bc, n, gg, N, mask_t, zl,
                                         GATE_ACT, HAS_BGB, EVEN_L)
                    w2 = w * gtB if HPACK and ROT else gl.expand_dims(w * gtB, 0)
                else:
                    w2 = w if HPACK and ROT else gl.expand_dims(w, 0)
                if HAS_GC:
                    if HPACK and ROT:
                        raw_gate = gl.expand_dims(_ldt(gC_ptr + bc, mask_t, EVEN_L).to(gl.float32),0) + z2
                        if HAS_BGC:
                            raw_gate += gl.expand_dims(gl.load(bgC_ptr + ((h+pf)//HG_G)*N+n),1)
                        gtC = _gate(raw_gate, GATE_ACT)
                    else:
                        gtC = _gate_load(gC_ptr, bgC_ptr, bc, n, gg, N, mask_t, zl,
                                         GATE_ACT, HAS_BGC, EVEN_L)
                    rg = r * gtC
                else:
                    rg = r
                # s1/s2 are the separable drivers of log a; on
                # LAW_A alone s1 is de and every line here folds back to it.
                if PLAIN:
                    c1v = -gl.exp(gl.load(A_log_ptr + anp + n * sn))
                    s1_2, s1n_2 = de2, den2
                else:
                    law_s = gl.load(row_law_ptr + n)
                    c1v = -gl.exp(gl.load(A_log_ptr + anp + n * sn))
                    s1, fac = _law_s1(a_dyn_ptr, law_s, de, b, bh, n, N, L,
                                      offs_t, mask_t, zl, HAS_DYN,
                                      UNI, DYN_ACT, DYN_EPS, A_DYN_N_RES,
                                    EVEN_L)
                    s1_2 = gl.expand_dims(s1, 0)
                    s1n_2 = gl.expand_dims(gl.gather(s1, idx_nxt, 0), 0)
                # c1e is [BLOCK_P, 1], so la/lan below are full tiles however
                # thin s1_2/s1n_2 are, and the two scans still see real tiles.
                c1e = gl.expand_dims(c1v, 1)
                la = c1e * LOG2E * s1_2
                lan = c1e * LOG2E * s1n_2
                if HAS_MU:
                    # both are needed after the branch, for gc2 and gs2, so
                    # they are defined outside it and only overwritten inside.
                    c2e = gl.expand_dims(gl.zeros([BLOCK_P], gl.float32,
                                                  layout=pl), 1)
                    s2_2 = z2
                    if UNI or law_s == K_MU:
                        c2e = gl.expand_dims(
                            gl.load(mu_gain_ptr + anp + n * sn), 1)
                        s2 = _law_s2(mu_ptr, mu_init_ptr, b, bh, n, N, L,
                                     offs_t, mask_t, col, zl, MU_N_RES,
                                     HAS_MU_INIT, MU_PREDIFF, EVEN_L)
                        s2_2 = gl.expand_dims(s2, 0) + z2
                        la = la + c2e * LOG2E * s2_2
                        lan = lan + c2e * LOG2E * (
                            gl.expand_dims(gl.gather(s2, idx_nxt, 0), 0) + z2)
                a = libdevice.exp2(la)
                cy = cyp + n * sn

                drive = w2 * u
                # ABL ablates one cost each -- wrong answers, right shape. 1:
                # the gc1 write. 2: the adjoint scan. 4: the reverse gathers.
                # 8: the h replay scan. 16: the p-contractions and their
                # writes. ROT-only: 32: Theta's in-chunk scan. 64: gtheta's
                # reverse scan. 128: the partner B/C loads.  512: walk
                # pairs with NO rotation (the loop STRUCTURE alone).
                # 4096: the third scan component below -- D from
                # h_t - drive_t (right answers only at weak decay; the
                # 2-operand reference of the D-form static test and
                # the same-tree A/B arm).
                # D below needs q_t = a_t h_{t-1}, the pre-write state. Formed
                # as h_t - drive_t it cancels under strong decay (|a h| <<
                # |drive|, e.g. HM's init A_log = log(n+1)): h_t's own rounding
                # u|h_t| becomes a relative error ~u|drive|/|a h| in D, and
                # through D in gc1 (dA_log), gc2, gs1 (ddelta, grate) and gs2.
                # So the scan carries q as a third component (_ssm_combine_q),
                # seeded a_0 h_prev at t = 0 and 0 * h_prev elsewhere (a zero
                # that only ever meets an addition); (a, hs) are the 2-operand
                # scan's text and, as compiled, its contractions -- except HPACK
                # real on sm_100+, where losing `hs - drive`'s use of drive
                # changes which product LLVM folds (tests/test_mh_dform_static
                # .py, a strict xfail): hs moves there at the rounding level.
                # REORDER A/B arm: the adjoint scan first, then the h scan
                rr2 = (rg if (ABL & 4) else gl.gather(rg, rev2, 1)) if HPACK and ROT else gl.expand_dims(
                    rg if (ABL & 4) else gl.gather(rg, idx_rev, 0), 0)
                if ABL & 2:
                    lam = gyr * rr2
                    if HAS_ENDPOINT:
                        gu += lam * w2
                        gBs = lam * u if HPACK and ROT else gl.sum(lam * u, axis=0)
                else:
                    ac = gl.load(alam_ptr + cy, mask=lcar, other=0.0)
                    beta = gyr * rr2
                    if HAS_ENDPOINT:
                        # The trapezoid's readout is q_t + own_t B_t x_t, so
                        # nothing reads the carried z_t itself: its adjoint
                        # is the STRICT one, e_t = a_{t+1} lam_{t+1} -- the
                        # adjoint scan's own `_ssm_combine_q` component,
                        # seeded with the carry a_0 lam_0 of chunk j+1 (no
                        # extra shuffles, and the one reverse gather moves
                        # from lam to e).  lam (the adjoint of q) is e + beta.
                        # Taking the inclusive lam and removing the diagonal
                        # would cancel on lam -> 0 heads, in gdw, gx, gB and
                        # gC alike.
                        acs = gl.expand_dims(ac, 1)
                        _, lam_r, lex_r = gl.associative_scan(
                            (libdevice.exp2(lan),
                             gl.where(first2, beta + acs, beta),
                             gl.where(first2, acs + z2, 0.0)),
                            axis=1, combine_fn=_ssm_combine_q)
                        lex = lex_r if (ABL & 4) else gl.gather(lex_r, rev2, 1)
                        # e's consumers run HERE, so e dies before the replay
                        # scan: holding e and lam both across it cost ROT +38
                        # spill bytes and +14% bwd on Blackwell
                        gu += lex * w2
                        gBs = lex * u if HPACK and ROT else gl.sum(lex * u, axis=0)
                        lam = lex + gyp * (rg if HPACK and ROT else gl.expand_dims(rg, 0))
                    else:
                        _, lam_r = gl.associative_scan(
                            (libdevice.exp2(lan),
                             gl.where(first2, beta + gl.expand_dims(ac, 1), beta)),
                            axis=1, combine_fn=_ssm_combine)
                        lam = lam_r if (ABL & 4) else gl.gather(lam_r, rev2, 1)
                if ABL & 8:
                    qh = a * u
                    hs = drive + qh
                else:
                    hprev = gl.load(hck_ptr + ckp
                                    + (j * N + n) * snc).to(gl.float32)
                    # The b seed keeps the 2-operand scan's text on purpose:
                    # its product a*hprev stays SINGLE-use, so it contracts
                    # into fma(a, hprev, drive) exactly as before. The q seed
                    # takes a_0 hprev through its own product (select, then
                    # multiply) that CSE cannot merge with it: a shared 2-use
                    # product ties the DAG combiner's choice of which
                    # multiply to fold, and it folds drive's instead --
                    # fma(w, u, a*hprev), a different h_0 (seen in the PTX).
                    hseed = gl.where(first2, drive
                                     + a * gl.expand_dims(hprev, 1), drive)
                    if ABL & 4096:
                        _, hs = gl.associative_scan(
                            (a, hseed), axis=1, combine_fn=_ssm_combine)
                    else:
                        _, hs, qh = gl.associative_scan(
                            (a, hseed, gl.where(first2, a, 0.0)
                             * gl.expand_dims(hprev, 1)),
                            axis=1, combine_fn=_ssm_combine_q)
                D = lam * (hs - drive) if ABL & 4096 else lam * qh

                if not HAS_ENDPOINT:
                    gu += lam * w2          # free: n is the sequential axis
                if not (ABL & 2):
                    # a_0*lam_0 is both the carry into chunk j-1 and, at j == 0,
                    # gh_init itself.
                    al0 = gl.sum(gl.where(first2, a * lam, 0.0), axis=1)
                    gl.store(alam_ptr + cy, al0)
                    gl.store(ghi_ptr + cy, al0, mask=at_t0)
                if not (ABL & 1):
                    # one reduce for all BLOCK_P p's
                    gl.store(gc1_ptr + cy,
                             gl.load(gc1_ptr + cy, mask=carry, other=0.0)
                             + gl.sum(D * s1_2, axis=1))
                    if HAS_MU and (UNI or law_s == K_MU):
                        gl.store(gc2_ptr + cy,
                                 gl.load(gc2_ptr + cy, mask=carry, other=0.0)
                                 + gl.sum(D * s2_2, axis=1))
                if not (ABL & 16):
                    # sum over p: warps_per_cta[0] == 1, shuffles, no barrier.
                    # d s1/d de is 1 on LAW_A and the rate on DYN,
                    # so delta only picks up the rows whose driver it is. PLAIN
                    # is the constant-1 case, and it does not merely skip the
                    # multiply -- `fac` comes back from `_law_s1`, which the
                    # PLAIN branch above never calls, so referencing it here is
                    # a compile error rather than a wasted instruction.
                    gs1 = D * c1e if HPACK else gl.sum(D * c1e, axis=0)
                    if PLAIN:
                        gdev += gs1
                    else:
                        gdev += gs1 * fac
                    # a row on another law contributes nothing to grate or
                    # gs2o, so it skips the write and its slot keeps the zero
                    # it was allocated with.
                    if HAS_DYN and (UNI or law_s == K_DYN):
                        if A_DYN_N_RES:
                            _put(grate_ptr + lawbn + n.to(gl.int64) * L
                                 + offs_t, gs1 * de, mask_t, LWS != 0, EVEN_L,
                                 LAW_VEC, TPL, WOWN)
                        else:
                            grt_acc += gs1 * de
                    if HAS_MU and (UNI or law_s == K_MU):
                        gs2 = gl.sum(D * c2e, axis=0)
                        if MU_N_RES:
                            _put(gs2o_ptr + lawbn + n.to(gl.int64) * L
                                 + offs_t, gs2, mask_t, LWS != 0, EVEN_L,
                                 LAW_VEC, TPL, WOWN)
                        else:
                            gs2_acc += gs2
                    # gB_t/gC_t still contract over h, i.e. across CTAs,
                    # so they go to a slot buffer (slot = h) summed on the host -- unless
                    # ATOMIC_BC narrows the slot axis to
                    # BCS and lets L2 do the rest of that sum (see the launcher
                    # for why this one pays where the gx atomic did not).
                    sc = bcn + n.to(gl.int64) * L + offs_t
                    if HAS_ENDPOINT:
                        # strict transport adjoint + the diagonal, both exact
                        qr = (hs - drive) if ABL & 4096 else qh
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
                        # the readout the forward summed: same `hs`, same
                        # effective C (`rg` is post-rotation, post-gate),
                        # same order over n.
                        if HAS_ENDPOINT:
                            # own's diagonal is added once per chunk below
                            ypre += qr * gl.expand_dims(rg, 0)
                        else:
                            ypre += hs * gl.expand_dims(rg, 0)
                    # The gate is the LAST fold, so its adjoint
                    # is the first undone: what continues down the chain is the
                    # adjoint of the pre-gate rotated B/C, and every line below
                    # -- the rotation transpose, gTh, the plain store -- then
                    # reads exactly as it does without gates.
                    if HAS_GB:
                        # the derivative from the pre-activation, re-read
                        # (`_gate_dv`: g (1 - g) cancels where the gate saturates)
                        if HPACK and ROT:
                            rv = gl.expand_dims(_ldt(gB_ptr + bc, mask_t, EVEN_L).to(gl.float32), 0) + z2
                            if HAS_BGB:
                                rv += gl.expand_dims(gl.load(bgB_ptr + ((h+pf)//HG_G)*N+n), 1)
                            dgt = _gate_dv(rv, GATE_ACT)
                        else:
                            dgt = _gate_load_d(gB_ptr, bgB_ptr, bc, n, gg, N, mask_t, zl,
                                               GATE_ACT, HAS_BGB, EVEN_L)
                        vgb = gB * w * dgt
                        _put(ggB_ptr + sc, gl.sum(vgb,axis=0) if HPACK and ROT else vgb,
                             mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        # beta_gB is added to g_B BEFORE the activation,
                        # so its adjoint is this same
                        # vector -- K-C takes its (g, n) sum here.
                        if HPACK and ROT:
                            if BACC_GB:
                                gl.atomic_add(gbgB_ptr + ((h+pf)//HG_G)*N+n,
                                              gl.sum(_zt(vgb,m2,EVEN_L),axis=1), sem="relaxed")
                        else:
                            _gate_bacc(gbgB_ptr, gg, n, N, vgb, mask_t, BACC_GB, EVEN_L,
                                       BGPART, (gl.program_id(0) * Bs + b) * NBLK + j)
                        gB = gB * gtB
                    if HAS_GC:
                        if HPACK and ROT:
                            rv = gl.expand_dims(_ldt(gC_ptr + bc, mask_t, EVEN_L).to(gl.float32), 0) + z2
                            if HAS_BGC:
                                rv += gl.expand_dims(gl.load(bgC_ptr + ((h+pf)//HG_G)*N+n), 1)
                            dgt = _gate_dv(rv, GATE_ACT)
                        else:
                            dgt = _gate_load_d(gC_ptr, bgC_ptr, bc, n, gg, N, mask_t, zl,
                                               GATE_ACT, HAS_BGC, EVEN_L)
                        vgc = gC * r * dgt
                        _put(ggC_ptr + sc, gl.sum(vgc,axis=0) if HPACK and ROT else vgc,
                             mask_t, ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        if HPACK and ROT:
                            if BACC_GC:
                                gl.atomic_add(gbgC_ptr + ((h+pf)//HG_G)*N+n,
                                              gl.sum(_zt(vgc,m2,EVEN_L),axis=1), sem="relaxed")
                        else:
                            _gate_bacc(gbgC_ptr, gg, n, N, vgc, mask_t, BACC_GC, EVEN_L,
                                       BGPART, (gl.program_id(0) * Bs + b) * NBLK + j)
                        gC = gC * gtC
                    if PMAP:
                        # the MAP rows' adjoints: w = al P0 + be P1, so
                        # dL/dP0 = al gB and dL/dP1 = be gB, per row -- no
                        # partner adjoint needed, so nothing crosses the pair
                        # but the two Theta sums.
                        scm = bcn + (2 * n).to(gl.int64) * L + offs_v
                        gBv = gl.convert_layout(gB, pv)
                        gCv = gl.convert_layout(gC, pv)
                        if pi == 0:
                            aB, bB, aC, bC = cT, -sT, cC, -sC
                        else:
                            aB, bB, aC, bC = sT, cT, sC, cC
                        if PRAW:
                            dB0 += aB * gBv
                            dB1 += bB * gBv
                            dC0 += aC * gCv
                            dC1 += bC * gCv
                        else:
                            _put(gBt_ptr + scm, aB * gBv, mask_v, ATOMIC_BC,
                                 EVEN_L, VEC_V, TPL_V)
                            _put(gBt_ptr + scm + L, bB * gBv, mask_v,
                                 ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                            _put(gCt_ptr + scm, aC * gCv, mask_v, ATOMIC_BC,
                                 EVEN_L, VEC_V, TPL_V)
                            _put(gCt_ptr + scm + L, bC * gCv, mask_v,
                                 ATOMIC_BC, EVEN_L, VEC_V, TPL_V)
                        thcr = _zt(gCv * drt, mask_v, EVEN_L)
                        gTa += _zt(gBv * dwt, mask_v, EVEN_L) + thcr
                        gTc += thcr
                    elif not ROT:
                        # gB/gC are the adjoint of the BIASED B/C,
                        # which is exactly what
                        # beta_b/beta_c are added to.
                        _put(gBt_ptr + sc, gB, mask_t, ATOMIC_BC, EVEN_L,
                             BC_VEC, TPL, WOWN)
                        _put(gCt_ptr + sc, gC, mask_t, ATOMIC_BC, EVEN_L,
                             BC_VEC, TPL, WOWN)
                        _bacc(gbb_ptr, gbc, n, N, gB, mask_t, BACC_B, EVEN_L)
                        _bacc(gbc_ptr, gbc, n, N, gC, mask_t, BACC_C, EVEN_L)
                    elif pi == 0:
                        gB0, gC0 = gB, gC
                    else:
                        sc0 = sc - L        # the pair's rows are adjacent
                        # R(Theta)^T on the p-contracted adjoints: four FMAs on
                        # [BLOCK_L] vectors.
                        vb0 = gB0 * cT + gB * sT
                        vc0 = gC0 * cC + gC * sC
                        vb1 = gB * cT - gB0 * sT
                        vc1 = gC * cC - gC0 * sC
                        _put(gBt_ptr + sc0, gl.sum(vb0,axis=0) if HPACK else vb0, mask_t,
                             ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _put(gCt_ptr + sc0, gl.sum(vc0,axis=0) if HPACK else vc0, mask_t,
                             ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _put(gBt_ptr + sc, gl.sum(vb1,axis=0) if HPACK else vb1, mask_t,
                             ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        _put(gCt_ptr + sc, gl.sum(vc1,axis=0) if HPACK else vc1, mask_t,
                             ATOMIC_BC, EVEN_L, BC_VEC, TPL, WOWN)
                        # the rotation is applied AFTER the bias, so the
                        # un-rotated pair above is the adjoint the bias sees.
                        _bacc(gbb_ptr, gbc, n - 1, N, gl.sum(vb0,axis=0) if HPACK else vb0, mask_t, BACC_B,
                              EVEN_L)
                        _bacc(gbc_ptr, gbc, n - 1, N, gl.sum(vc0,axis=0) if HPACK else vc0, mask_t, BACC_C,
                              EVEN_L)
                        _bacc(gbb_ptr, gbc, n, N, gl.sum(vb1,axis=0) if HPACK else vb1, mask_t, BACC_B, EVEN_L)
                        _bacc(gbc_ptr, gbc, n, N, gl.sum(vc1,axis=0) if HPACK else vc1, mask_t, BACC_C, EVEN_L)
                        # d Brot[n]/dTheta == sgn[n] * Brot[n^1], and both rows
                        # of the pair drive the SAME Theta, so their two terms
                        # add. w/r here are already rotated, i.e. Brot/Crot.
                        gThC = _zt(r0 * gC - r * gC0, m2 if HPACK else mask_t, EVEN_L)
                        gTh = _zt(w0 * gB - w * gB0, m2 if HPACK else mask_t, EVEN_L) + gThC
                        if HAS_THH:
                            # theta_h enters C's angle directly (no prefix
                            # sum), so its gradient is the plain time sum of
                            # the C-side half: one partial per (chunk, pair),
                            # plain store.
                            if ROT_MIX:
                                gThC = gThC * rf
                            if HPACK:
                                gl.store(gthh_ptr + (((h+pf)*Bs+b)*K+kk.to(gl.int64))*NBLK+j,
                                         gl.sum(gThC,axis=1))
                            else:
                                gl.store(gthh_ptr + ((hslot * Bs + b) * K
                                                     + kk.to(gl.int64)) * NBLK + j
                                         + zp, gl.zeros([BLOCK_P], gl.float32,
                                                        layout=pl)
                                         + gl.sum(gThC), mask=pofs == 0)
                        if ROT_MIX:
                            # the identity rotation leaves the four B/C above
                            # right but NOT this: a non-rotating pair drives no
                            # Theta at all, and the expression does not vanish
                            # on its own.
                            gTh = gTh * rf
                        if not (ABL & 64):
                            # Theta is a forward prefix sum, so its adjoint is
                            # a reverse one -- taken as gather/scan/gather
                            # because Triton's reverse=True lowering is slow.
                            if HPACK:
                                rc = rcar_ptr + rcp + kk * P
                                Rr = gl.associative_scan(
                                    gl.gather(gTh, rev2, 1),
                                    axis=1, combine_fn=_add)
                                Rr += gl.expand_dims(
                                    gl.load(rc, mask=carry, other=0.0), 1)
                                gl.store(rc,
                                         gl.sum(gl.where(last2, Rr, 0.0), axis=1))
                                gTh = gl.gather(Rr, rev2, 1)
                            else:
                                # p-free, so ONE row (`row1`), not the
                                # p-replicated tile; the carry is the
                                # p-block's first slot of `rcar`.
                                rv1 = gl.expand_dims(gl.convert_layout(
                                    idx_rev, gl.SliceLayout(0, row1)), 0)
                                R1 = gl.associative_scan(
                                    gl.gather(gl.expand_dims(gl.convert_layout(
                                        gTh, gl.SliceLayout(0, row1)), 0),
                                        rv1, 1),
                                    axis=1, combine_fn=_add)
                                rc1 = rcar_ptr + bh * K * P \
                                    + pid_p * BLOCK_P + kk * P
                                R1 += gl.where(jj > 0, gl.load(rc1), 0.0)
                                gl.store(rc1, gl.sum(gl.sum(gl.where(
                                    rv1 == 0, R1, 0.0), axis=1), axis=0))
                                gTh = gl.convert_layout(gl.sum(gl.gather(
                                    R1, rv1, 1), axis=0), lt)
                        if THETA_DELTA:
                            # inc = Delta * theta: chain both factors.
                            thr = _ldt(theta_ptr + (thb
                                                    + kk.to(gl.int64)) * L
                                       + offs_t, mask_t,
                                       EVEN_L).to(gl.float32)
                            gdev += gl.expand_dims(thr,0) * gTh if HPACK else thr * gTh
                            gTh = gTh * (de2 if HPACK else de)
                        _put(gth_ptr
                             + (thn + kk.to(gl.int64)) * L + offs_t,
                             gl.sum(gTh,axis=0) if HPACK else gTh, mask_t, ATOMIC_TH, EVEN_L,
                             TH_VEC, TPL, WOWN)
            if PMAP:
                if PRAW:
                    scp = bcn + (2 * kk).to(gl.int64) * L + offs_v
                    _put(gBt_ptr + scp, dB0, mask_v, ATOMIC_BC, EVEN_L,
                         VEC_V, TPL_V)
                    _put(gBt_ptr + scp + L, dB1, mask_v, ATOMIC_BC, EVEN_L,
                         VEC_V, TPL_V)
                    _put(gCt_ptr + scp, dC0, mask_v, ATOMIC_BC, EVEN_L,
                         VEC_V, TPL_V)
                    _put(gCt_ptr + scp + L, dC1, mask_v, ATOMIC_BC, EVEN_L,
                         VEC_V, TPL_V)
                # the pair's Theta tail, as the in-kernel route runs it at
                # pi == 1 (theta_h partial, identity pairs, reverse prefix,
                # Delta chain), on the sums the rows accumulated
                if HAS_THH:
                    if ROT_MIX:
                        gTc = gTc * rf
                    gl.store(gthh_ptr + ((hslot * Bs + b) * K
                                         + kk.to(gl.int64)) * NBLK + j
                             + zp, gl.zeros([BLOCK_P], gl.float32, layout=pl)
                             + gl.sum(gTc), mask=pofs == 0)
                if ROT_MIX:
                    gTa = gTa * rf
                if not (ABL & 64):
                    # Theta is a forward prefix sum, so its adjoint is the
                    # reverse one, carried across chunks (processed last to
                    # first) in this p-block's first `rcar` slot
                    rc1 = rcar_ptr + bh * K * P + pid_p * BLOCK_P + kk * P
                    gTa = gl.associative_scan(gTa, axis=0, combine_fn=_add,
                                              reverse=True) \
                        + gl.where(jj > 0, gl.load(rc1), 0.0)
                    gl.store(rc1, gl.sum(gl.where(offs_v == j * BLOCK_L, gTa,
                                                  0.0), axis=0))
                if THETA_DELTA:
                    thr = _ldt(theta_ptr + (thb + kk.to(gl.int64)) * L
                               + offs_v, mask_v, EVEN_L).to(gl.float32)
                    gdev += gl.convert_layout(thr * gTa, lt)
                    gTa = gTa * de_v
                _put(gth_ptr + (thn + kk.to(gl.int64)) * L + offs_v, gTa,
                     mask_v, ATOMIC_TH, EVEN_L, VEC_V, TPL_V)

        # x is re-read rather than held: inside the n loop only `u` and the
        # hoisted `local_own` use it, and a [BLOCK_P, BLOCK_L] tile kept live
        # across that loop is spill (the same bf16 value, so exact).
        xt = _ldt(x_ptr + po, m2, EVEN_L).to(gl.float32)
        # GX_VEC needs no lane dedup (the 2-D tile has no replicas): TPL=32.
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
                _st(gendpoint_ptr + gdb + bh * L + offs_t,
                    gl.sum(local_grad * xt, axis=0), mask_t, EVEN_L)
        if HAS_DSKIP:
            # NS n-SHARDS all see the same (gy, x) tile and all accumulate
            # into gx, so shard 0 OWNS the skip -- otherwise both gx and
            # gD would be counted NS times.  NP p-blocks are disjoint in
            # p, so they each contribute their own share and need no such
            # gate.
            own = 1.0 if NS == 1 else gl.where(sh == 0, 1.0, 0.0)
            if HPACK:
                # one D per tile ROW, and one atomic per row
                dv = gl.expand_dims(gl.load(d_skip_ptr + h + pofs), 1) * own
                gxv += dv * gyp
                gl.atomic_add(gd_skip_ptr + h + pofs,
                              own * gl.sum(gl.where(m2, gyp * xt, 0.0),
                                           axis=1), sem="relaxed")
            else:
                dv = gl.load(d_skip_ptr + h) * own
                gxv += dv * gyp
                gl.atomic_add(gd_skip_ptr + h,
                              own * gl.sum(gl.sum(gl.where(m2, gyp * xt, 0.0),
                                                  axis=1), axis=0), sem="relaxed")
        if HAS_Z:
            if HAS_DSKIP:
                # the skip is part of what the gate multiplied, so it is
                # part of `y_pre`.  `NS == 1` under a gate, so `own` is 1
                # and `dv` is D_h itself.
                ypre += dv * xt
            # dL/dz = dL/dy * y_pre * silu'(z).  `gz` is CHANNEL-major and
            # in z's own dtype: one plain store, no slot axis (NS == 1),
            # so this rounds the fp32 register value ONCE -- the same
            # single rounding `tails._gate_tr_bwd` gave it.
            _st(gz_ptr + po,
                gyz * ypre * _silu_fast_d(zt, zsg), m2, EVEN_L)
        _put(gx_ptr + gxo + po, gxv, m2, ATOMIC_GX, EVEN_L, GX_VEC, 32)
        # the write path lands on whichever tensor owns the write strength;
        # with delta_w present, `delta` keeps only the decay term gdev already
        # holds.
        if HAS_DW:
            if DW_P:
                _st(gdw_ptr + gxb + po, gu * xt, m2, EVEN_L)
            elif HPACK:
                _st(gdw_ptr + gdb + po, gu * xt, m2, EVEN_L)
            else:
                _st(gdw_ptr + gdb + bh * L + offs_t,
                    gl.sum(gu * xt, axis=0), mask_t, EVEN_L)
        else:
            gdev += gu * xt if HPACK else gl.sum(gu * xt, axis=0)
        # the softplus' derivative, applied here rather than on the host
        # (`dsig` above).  `out["delta"]` is therefore already dL/d(delta).
        if DELTA_SOFTPLUS and DSIG:
            gdev = gdev * dsig
        if HPACK:
            # (nsp, B, H, L) with H == D at P == 1: `po` addresses the same
            # buffer the unpacked path writes one head-row of.
            _st(gde_ptr + gdb + po, gdev, m2, EVEN_L)
            if DBACC:
                # the tile's rows ARE heads here, so the time reduce alone
                # gives one partial per head.
                dba += gl.sum(gl.where(m2, gdev, 0.0), axis=1)
        else:
            _st(gde_ptr + gdb + bh * L + offs_t, gdev, mask_t, EVEN_L)
            if DBACC:
                # one head per CTA and `gdev` already p-reduced: the whole
                # chunk contributes ONE number, broadcast so that the store
                # below has the same shape it has under HPACK.
                dba += gl.zeros([BLOCK_P], gl.float32, layout=pl) \
                    + gl.sum(_zt(gdev, mask_t, EVEN_L))
        if HAS_DYN and not A_DYN_N_RES:
            _st(grate_ptr + gdb + bh * L + offs_t, grt_acc, mask_t,
                EVEN_L)
        if HAS_MU and not MU_N_RES:
            _st(gs2o_ptr + gdb + bh * L + offs_t, gs2_acc, mask_t,
                EVEN_L)
    if DBACC:
        # (slot, B, H) fp32.  Under HPACK the tile spans HP consecutive
        # heads and `bh + pf` is the absolute one; unpacked the CTA owns a
        # single head and only lane 0 writes.  A plain store into a slot
        # the host already sums -- see the constexpr's note on determinism.
        if HPACK:
            gl.store(gdbias_ptr + gdbh + bh + pf, dba)
        else:
            gl.store(gdbias_ptr + gdbh + bh + zp, dba, mask=pofs == 0)


#: Launch plans (see `mh_backward`'s LAUNCH PLAN note).  `MH_BWD_PLAN=0`
#: turns the cache off (every call re-derives the plan) -- the A/B arm.
_PLANS: dict = {}
_PLAN_ON = os.environ.get("MH_BWD_PLAN", "1") != "0"
_ROW_SEL = ("row_law", "row_rot")
_ENV_PFX = (b"MH_", b"HYBRID_", b"SSD_")
_PLAIN = frozenset((type(None), bool, int, float, str))


def _plan_key(argv):
    """Everything the plan is a function of, hashable.

    Tensors by (shape, stride, dtype, device, 16-byte alignment); the two row
    selectors by CONTENT (their launched windows decide the law mask and the
    rotation mix; `selector_window` caches the host copy per tensor version);
    dicts by items; every MH/HYBRID env var and the active kernel-option
    scope, because the launcher reads both; the SLP switch."""
    # positional over the launcher's own `locals()`, so the names are implied
    parts = []
    for k, v in argv.items():
        t = type(v)
        if t in _PLAIN:
            parts.append(v)
        elif isinstance(v, torch.Tensor):
            if k in _ROW_SEL:
                parts.append(tuple(selector_window(v, 0, v.numel())))
            else:
                parts.append((v.shape, v.stride(), v.dtype, v.device,
                              v.data_ptr() % 16 == 0))
        elif isinstance(v, dict):
            parts.append(tuple(sorted((a, repr(b)) for a, b in v.items())))
        else:
            parts.append((t, repr(v)))
    from ... import kernel_options as _ko
    # the raw (bytes) mapping: `os.environ.items()` decodes every variable of
    # the process, ~0.1 ms on a cluster login shell -- more than the plan saves
    _raw = getattr(os.environ, "_data", None)
    env = (frozenset((a, b) for a, b in _raw.items() if a.startswith(_ENV_PFX))
           if isinstance(_raw, dict) and _raw and isinstance(next(iter(_raw)), bytes)
           else frozenset((a, b) for a, b in os.environ.items()
                          if a.startswith(("MH_", "HYBRID_", "SSD_"))))
    return (tuple(parts), env, repr(_ko.current("hybrid_mamba")),
            bool(slp.ENABLED), autotune.tier())


def mh_backward(x, delta, A_log, B_t, C_t, gy, *, theta=None,
                theta_init=None, row_law=None, row_rot=None, a_dyn=None,
                mu=None, mu_gain=None, mu_init=None, mu_prediff=False,
                dyn_act=DYN_RAW,
                dyn_eps=0.0, a_dyn_res="n", mu_res="n", h_init=None,
                delta_bias=None, delta_softplus=True, beta_a=None,
                beta_b=None, beta_c=None, beta_gB=None, beta_gC=None,
                g_B=None, g_C=None, gate_act=GATE_SIGMOID,
                delta_w=None, gh_final=None, d_skip=None, z=None, endpoint_own=None,
                n_off=0, n_rows=None, config=None, out=None, hck=None,
                mem=None, theta_delta=False, theta_h=None, bc_head=False,
                pmap=False):
    """Transposed tile: p on the tile axis, n sequential, gu in registers.

    `theta` supplies the rotation angle and `row_rot` says which rows use
    it; absent `row_rot` every row rotates, which is the worst case and so the
    right thing to measure. A mixture costs one extra scalar load per pair --
    the non-rotating pairs take the identity rotation rather than a second code
    path -- and a launch where no row rotates compiles the path out entirely.

    `row_law` selects the per-row decay law of the contract, so that log a is
    c1*s1 + c2*s2 with s1 = delta (LAW_A) or delta*rate(a_dyn) (A_DYN),
    and s2 the telescoping mu increment (MU).

    `hck` hands over the forward's chunk checkpoints, and handing them over is
    what `RUN_P1` defaults to: there is no reason to rebuild a state the caller
    already computed, and asking for the skip through `config=` instead would
    disable the autotuner, which is a bigger loss than pass 1 is a saving (27%
    at B4H8P64N64L8192 against 20%). An explicit `config["RUN_P1"]` still wins,
    which is how the checkpoint bench timed the two sides on one config.

    The corollary is that `hck` is NOT a scratch buffer. Handing over one whose
    contents do not match these inputs gives wrong gradients with right shapes,
    silently. Pass it instead of building `out` yourself: the scratch set grows
    whenever a feature lands, and a caller-side copy of this allocator silently
    goes stale (`gth_s`/`thck`/`rcar` arrived with ROT).

    `h_init` is carried in slot 0 of `hck` rather than as its own argument:
    pass 1 chains slot j -> j+1 and pass 2 reads slot j as the chunk's incoming
    state, so seeding slot 0 is all a non-zero initial state needs. It is
    written here even when `hck` is handed over, since the two sides must agree
    and the write is idempotent.

    `delta_bias` is (H,) and folds into delta before the softplus, so
    `out["delta"]` is dL/d(delta) with the softplus derivative already applied
    -- not the raw dL/d(effective delta).

    `beta_a` costs the kernel nothing: it is a (G,N) offset on A_log inside the
    exp, and A_log is already copied on the host to permute it, so the offset
    rides along in that copy. Its adjoint then falls straight out of gc1, which
    is why it lands before the other four beta_*.

    `g_B`/`g_C` are the (B,N,L) gates of the contract, folded LAST -- after the bias
    and after the rotation. Their adjoints need slot buffers of
    their own rather than being recovered from `Bs_`/`Cs_`, since that would
    mean dividing by act(gv), which loses every digit at sigmoid saturation.
    `beta_gB`/`beta_gC` then fall out of those buffers the way `beta_b`/`beta_c`
    fall out of `Bs_`/`Cs_`, and may group heads at a different rate.

    `delta_w` splits the write strength off the decay's delta and may
    be (B,H,L) or per-channel (B,H,P,L). Where it is absent the two are one
    tensor and `out["delta"]` carries both terms; where it is present `delta`
    keeps the decay term alone and the write term goes to `out["delta_w"]`.

    `gh_final` is the adjoint of the returned final state, and rides in `alam`
    the way `h_init` rides in slot 0 of `hck`.

    `n_off`/`n_rows` restrict the launch to a window of state rows. `NS` already
    shards the n loop, so this only moves where that partition starts and ends;
    row indices stay absolute and every gradient outside the window is zero,
    since those rows never entered the recurrence.
    """
    # Exactly the arguments, captured before any of them is rebound (row_law
    # and row_rot are). The autotuner re-enters this function per candidate, and
    # spelling the keywords out by hand would silently drop the next feature
    # added to the signature from every trial.
    _argv = dict(locals())
    # LAUNCH PLAN: everything above the allocation is a pure
    # function of the call's signature (shapes, strides, dtypes, alignment,
    # None-ness, scalar options, row-selector CONTENTS, env / kernel-option
    # scope, device) and of the tuner's settled verdict.  A repeat call takes
    # the resolved plan instead of re-deriving it (host time).
    # Invalidated when the tuner's verdict it was built on is dropped.
    _pk = _plan_key(_argv) if (config is None and out is None and _PLAN_ON) else None
    _pl = _PLANS.get(_pk) if _pk is not None else None
    if _pl is not None and _pl[1] is not None and autotune.cached(_pl[1]) is not _pl[2]:
        _pl = None
    if _pl is not None:
        (A_DYN_N_RES, BACC_B, BACC_C, BACC_GB, BACC_GC, BL, BLK_P, Bs, CKPACK, DBACC, DSIG, DW_P, G_BC, G_G, H, HAS_DYN, HAS_MU, HG_BC, HG_G, HP, HPACK, HSL, K, L, LAW_MASK, MU_N_RES, N, NBLK, NBS, NP, P, ROT, ROT_MIX, TS, _cc, bc_exact, bias_part, cfg, has_bias, nr, vec_ok,) = _pl[0]
        cfg = dict(cfg)
        if row_rot is not None:
            row_rot = _i32(row_rot).contiguous()
        if row_law is not None:
            row_law = _i32(row_law).contiguous()
    else:
        if mu_prediff and mu_init is not None:
            raise ValueError("mu_prediff means `mu` already holds the telescoped\n"
                             "increment, so mu_init is folded into column 0 and must\n"
                             "be None -- applying it again shifts row 0 of every b")
        Bs, H, P, L = x.shape
        # under `pmap` B_t/C_t are the (B, 2N, L) pair maps (fwd_gluon PMAP)
        N = B_t.shape[-2] // (2 if (pmap and pmap != "raw") else 1)
        K = N // 2
        nr = (N - n_off) if n_rows is None else n_rows
        # `theta` supplies the angle; `row_rot` says which rows use it. Absent it
        # every row rotates.
        ROT, ROT_MIX = theta is not None, False
        if row_rot is not None:
            # The window is read off the CALLER's tensor, before `_i32` -- a bool
            # `row_rot` (which is what `fracs_to_rows` hands back) would otherwise
            # be a fresh object on every call and miss the memo every time.
            # ONE device->host transfer, not two `.any()`/`.all()` reductions.
            # Each of those is a kernel plus a SYNC -- invisible in GPU
            # self-time, but host wall time on a busy card.  `row_rot` is (N,),
            # so the transfer is bytes.
            rot_w = selector_window(row_rot, n_off, nr)
            ROT_MIX = ROT and not all(rot_w)
            ROT = ROT and any(rot_w)
            row_rot = _i32(row_rot).contiguous()
        if pmap:
            if theta is None or bc_head or any(t is not None for t in (
                    g_B, g_C, beta_b, beta_c, beta_gB, beta_gC)):
                raise ValueError("pmap: pair maps belong to a rotating launch "
                                 "with every B/C fold (bias, gate) applied")
            # only the rotating body reads pair maps: a launch with no rotating
            # row runs it with every pair on the identity
            if not ROT:
                ROT, ROT_MIX = True, True
        if ROT and (n_off % 2 or nr % 2):
            raise ValueError("a rotating partial launch must start and end on a "
                             "pair boundary: got n_off=%d n_rows=%d" % (n_off, nr))
        # exactly which laws are present, not just whether any is: a tensor on one
        # law compiles without the runtime dispatch at all (UNI in the kernel). Only
        # the launched window counts -- a law that appears solely outside it is dead
        # weight in the dispatch.
        LAW_MASK = 1
        if row_law is not None:
            # One transfer, then a set: see the `row_rot` note above.
            window = set(selector_window(row_law, n_off, nr))
            row_law = _i32(row_law).contiguous()
            LAW_MASK = sum(1 << law for law in
                           (LAW_A_FIX, LAW_A_DYN, LAW_MU)
                           if law in window) or 1
        # The four static B/C biases' head-GROUP counts.  Needed this
        # early because K-C makes them part of the BC slot axis' legality, which
        # `hpack_ok` and the autotune key both read.
        G_BC = (beta_b.shape[0] if beta_b is not None else
                beta_c.shape[0] if beta_c is not None else 1)
        HG_BC = H // G_BC
        G_G = (beta_gB.shape[0] if beta_gB is not None else
               beta_gC.shape[0] if beta_gC is not None else 1)
        HG_G = H // G_G
        # K-C.  A static bias' gradient is the sum over (head-in-group,
        # b, t) of the SAME adjoint the BC slot buffers hold, so the ONLY thing
        # the `H*NP` slot axis was kept for under a bias is the per-head regroup
        # `Bs_.sum(-1).sum(1).reshape(G, H/G, N).sum(1)`.  Two ways to stop
        # needing it, and the cheap one is picked per tensor:
        #
        #  * G == 1 (head-SHARED): the regroup is the sum over EVERY head, which
        #    is what the folded (1,B,N,L) buffer already holds -- so the bias
        #    costs the kernel nothing at all and the read-back is one host
        #    reduction of `out["B_t"]`.  This is the paper's row.
        #  * G > 1 (GROUPED): `_bacc` accumulates the (G, N) sum in-kernel with
        #    one scalar atomic per (CTA, chunk, row), and the slot axis folds
        #    the same way.
        #
        # Either way `bc_exact` below stops testing `beta_* is None`.
        _bshared = [(t is None or t.shape[0] == 1)
                    for t in (beta_b, beta_c, beta_gB, beta_gC)]
        _bgrouped = [(t is not None and t.shape[0] > 1)
                     for t in (beta_b, beta_c, beta_gB, beta_gC)]
        has_bias = any(t is not None
                       for t in (beta_b, beta_c, beta_gB, beta_gC))
        # `cfg` is built below, but `hpack_ok` -- which feeds it -- needs the
        # answer, so the two axes K-C reads are resolved from the same three
        # places `cfg.update` would take them from, in the same order.
        # `bwd_config` (kernel option / HYBRID_MH_BWD_CONFIG) is resolved HERE so
        # the K-C axes read it too: precedence is config > mem > pin > env.
        _pin = autotune.pinned_config("bwd") if config is None else None
        # ...else the calibrated P = 1 cell (the shipped per-GPU table,
        # `calibration_mh.p1_config`), served as a pin: a training launch
        # (the forward's checkpoints handed over) of a Mamba-1 call whose
        # exact cell was measured; every other call stays with the tuner.
        if (config is None and not _pin and out is None and hck is not None
                and P == 1):
            from . import calibration_mh as _cm
            if _cm.p1_envelope(P=P, ROT=ROT, pmap=pmap, bc_head=bc_head,
                               LAW_MASK=LAW_MASK, endpoint_own=endpoint_own,
                               n_off=n_off, nr=nr, N=N,
                               absent=(delta_w, g_B, g_C, beta_a, beta_b, beta_c,
                                       beta_gB, beta_gC, a_dyn, mu, theta,
                                       mu_init, theta_init, gh_final)):
                _pin = _cm.p1_config(
                    "bwd", Bs, H, N, L, x.dtype,
                    _cm.p1_feat(delta_softplus=delta_softplus,
                                delta_bias=delta_bias, d_skip=d_skip, z=z,
                                h_init=h_init)) or None

        def _axis(name, env, dflt):
            for src in (config, mem, _pin):
                if src and name in src:
                    return src[name]
            return _envi(env, dflt)
        # `BCS=0` is the contract determinism pin, and it asks for the
        # FULL slot axis and fixed-order host sums.  K-C's accumulator is an
        # atomic, so the pin turns it off too -- otherwise `mem=dict(AGX=0,
        # BCS=0)` would still leave dBeta non-deterministic.
        bacc_on = bool(_axis("BACC", "MH_BWD_BIAS_ACC", 1)) \
            and _axis("BCS", "MH_BWD_BC_SLOTS", None) != 0
        # Recursive tuning trials receive the caller's resolved bias policy.
        # An automatic BCS=0 means full map slabs on scalar-atomic devices;
        # it must not become an explicit determinism request inside the trial.
        # Otherwise the trial disables grouped bias accumulation/head packing,
        # and the final launch runs a different kernel from the one timed.
        if config is not None and "_resolved_bacc" in config:
            bacc_on = bool(config["_resolved_bacc"])
        _acc = [g and bacc_on for g in _bgrouped]
        BACC_B, BACC_C, BACC_GB, BACC_GC = _acc
        bias_part = autotune.gate_bias_partial(P) and (BACC_GB or BACC_GC)
        # "no bias that still needs the per-head slot axis": a head-SHARED bias
        # reads off the folded (1,B,N,L) buffer, a GROUPED one off its own (G,N)
        # accumulator, and anything else keeps the axis (and the fold is off).
        bc_exact = not bc_head and all(sh or ac for sh, ac in zip(_bshared, _acc))
        HAS_DYN = bool((LAW_MASK >> LAW_A_DYN) & 1)
        HAS_MU = bool((LAW_MASK >> LAW_MU) & 1)
        A_DYN_N_RES, MU_N_RES = (a_dyn_res == "n"), (mu_res == "n")
        DW_P = delta_w is not None and delta_w.dim() == 4
        # MAXNREG caps ptxas's allocation. It stays None: a cap trades
        # registers for resident CTAs, but every cap loses in proportion to the
        # spill it causes. The kernel is neither memory- nor compute-bound and
        # its grid (H,B,NS) is smaller than the per-SM register ceiling
        # 65536/(32*regs), so the register limit has slack and a cap buys slots
        # no CTA exists to fill. The direction is more CTAs, not fewer
        # registers.
        #
        # AGX / BCS / BCBF / XSBF are the four MEMORY axes (see the buffer block
        # below for what each trades). They live in `cfg` rather than in an env var
        # because they change what the kernel compiles to, and an env var read
        # after the autotune key is built lets one setting inherit another's tuned
        # config -- silent, and exactly the failure mode the key's docstring warns
        # about. BCS=0 means the full H*NP slot axis.
        #
        # AGX, BCS (where legal, see below) and BCBF are ON by default and XSBF
        # is off: AGX and the vectorised BCS fold trade a little time for a large
        # cut in peak memory, and XSBF is inert under AGX. BCBF only matters
        # where the fold is refused (`bc_exact` false, or no vector red): it
        # narrows the full slot buffer to the dtype the INPUTS already are.
        # VBC / VGX vectorise the atomic paths (red.global.add.v{2,4}.f32): the
        # scalar fold's price is per-element instruction count, not contention,
        # and a vector red is that count divided by the width. CKBF stores `hck`
        # in the INPUT dtype when that is 16-bit: the checkpoint is a rounding of a
        # state the bf16 inputs already carry at bf16 precision, and it is the
        # single largest fixed buffer.
        # HPACK -- HP consecutive HEADS on the tile axis (kernel docstring,
        # "PACKED SMALL-P SCHEDULE").  Legal only where every per-head tensor is
        # either channel-major already or absent, which is exactly this list;
        # anywhere else HP stays 1.
        #
        # Packing with a B/C bias is legal under one condition, which is about
        # the bias LOAD and not the gradient: `_bc_load`/`_gate_load` fetch ONE
        # `(G,N)` row for the CTA's base head and broadcast it across the tile,
        # and under HPACK the tile's rows are HP different heads.  They must
        # therefore all be in the same bias group, i.e. HP must divide the group
        # size (checked against the resolved HP below, which is why only the
        # presence test is here).
        hpack_ok = (P == 1 and LAW_MASK == 1 and not pmap
                    and (theta is None or theta.dim() == 3)
                    and (theta is None or (hck is not None and not bias_part
                        and beta_b is None and beta_c is None and endpoint_own is None and z is None)) and bc_exact
                    # a handed-over checkpoint has to be in the layout the packed
                    # schedule reads; `mh_forward` writes that one at P == 1
                    # (CKPACK), so a 5-D buffer here is an unpacked one.
                    and (hck is None or hck.dim() == 4))
        cfg = dict(NUM_WARPS=None, BLOCK_L=None, TP_P=None, ABL=0,
                   HP=(_hp_default(H) if hpack_ok else 1),
                   RUN_P1=hck is None,
                   # NP defaults with P, not to 1: `auto_block_l` computes the
                   # chunk length from `P // auto_np(P)`, and the two must agree or
                   # the forward's checkpoint stride stops matching the chunk the
                   # backward reads it for. See `auto_block_l`.
                   NS=None, NP=auto_np(P), RUN_TH=True, FAST_TRIG=True, PU=0,
                   # The register CAP is a SEARCHED axis under the packed
                   # schedule (`autotune.v3_grid`), seeded here at None.  The
                   # MAXNREG note above holds for P > 1; at P == 1 the packed
                   # schedule can profit from a cap, but not at every shape, so
                   # the search prices it per shape.  `MH_BWD_MAXNREG` pins it.
                   MAXNREG=_maxnreg()[0],
                   AGX=_envi("MH_BWD_ATOMIC_GX", 1),
                   # None = resolved below once the legality of the fold is known
                   BCS=_envi("MH_BWD_BC_SLOTS", None),
                   LWS=_envi("MH_BWD_LAW_SLOTS", 1),
                   BCBF=1,
                   XSBF=0,
                   # `gdelta`'s ONE slot in delta's own dtype.
                   # Bitwise (the store rounds the same fp32 register value
                   # `functions.py` would have rounded), so 0 is an A/B knob
                   # and not an accuracy escape.
                   DEBF=_envi("MH_BWD_DE_16", 1),
                   # slabs of the gtheta slot axis.  None = resolved below
                   # once the vector red's availability is known; 0 means the
                   # full `H*NP/HP` axis and fixed-order host sums, which is what
                   # the §1.14 determinism pin asks for.
                   THS=_envi("MH_BWD_TH_SLOTS", None),
                   VBC=_envi("MH_BWD_BC_VEC", 1),
                   VGX=_envi("MH_BWD_GX_VEC", 1),
                   # warps on the p (packed head) axis, ON TOP of NUM_WARPS on
                   # time: the CTA runs NUM_WARPS * WP warps.  1 = the layout
                   # no p-axis warps.
                   WP=_envi("MH_BWD_WP", 1),
                   # K-C: the in-kernel (G,N) bias accumulator, which is what
                   # lets a GROUPED bias keep the folded BC slot axis.  0 puts
                   # the grouped bias back on the per-head slot read-back --
                   # the determinism escape of the contract (this is the
                   # only NEW atomic K-C introduces, and it is the one dBeta
                   # rides on).
                   BACC=_envi("MH_BWD_BIAS_ACC", 1),
                   # `delta_bias`'s gradient out of this kernel's
                   # own chunk partials instead of a host `aten::sum` over the
                   # whole [B, H, L] delta gradient.  A plain store into the
                   # existing slot axis, so 0 puts back a host reduce that was
                   # never the more deterministic of the two -- it is a knob for
                   # A/B and for the oracle, not a determinism escape.
                   DBACC=_envi("MH_BWD_DBIAS_ACC", 1),
                   CKBF=int(bool(_ko.value("hybrid_mamba", "ckpt_bf16"))),
                   # TS: super-chunks walked in parallel, seeded by
                   # `ts_adjoint.ts_seeds`.  1 = the sequential walk.
                   TS=_envi("MH_BWD_TS", 1))
        # `mem` sets those three WITHOUT pinning the rest, which `config` cannot do
        # -- `config is not None` is also what turns the autotuner off, so asking
        # for a smaller buffer through it would cost the tuned BLOCK_L/NS/TP_P as
        # a side effect.
        # `bwd_config` (kernel option / HYBRID_MH_BWD_CONFIG, `_pin` above): an
        # explicit launch config, taken as is and never searched -- the
        # caller-facing form of the `pinned` tier.  A `mem=` dial and `config=`
        # (internal callers, the tuner's own trials) both win over it: `mem` is
        # applied AFTER it, so `mem=dict(AGX=0, BCS=0)` stays the determinism
        # pin under any bwd_config.
        if _pin:
            cfg.update(_pin)
            # as `MH_BWD_MAXNREG`: a pinned cap <= 0 means "no cap"
            if cfg.get("MAXNREG") is not None and cfg["MAXNREG"] <= 0:
                cfg["MAXNREG"] = None
        elif config is None:
            _tab = autotune.table_bwd(
                torch.cuda.get_device_capability(x.device)[0] * 10
                + torch.cuda.get_device_capability(x.device)[1],
                theta is not None, x.element_size(), P)
            # a `mem` pin is an answer (K-C gate below); the table fills the rest
            cfg.update({k: v for k, v in (_tab or {}).items()
                        if k not in (mem or {})})
        cfg.update({k: v for k, v in (mem or {}).items() if k in cfg})
        cfg.update(config or {})
        if cfg["AGX"] == 0 and cfg["BCS"] == 0:
            cfg["LWS"] = 0   # preserve the explicit determinism pin
        # K-C gate.  An axis the CALLER pinned through `mem` is an ANSWER, not a
        # starting point: `v3_axes` below is built around `cfg` and would
        # otherwise offer the search a way off it, and `choose_axes` takes any
        # candidate that wins on time-and-bytes.  For `AGX`/`BCS` that is a
        # CORRECTNESS question -- the contract makes
        # `mem=dict(AGX=0, BCS=0)` the determinism pin, and a search that moves
        # off it puts the cross-CTA atomics back and the pinned backward stops
        # being bit-reproducible.  `mem` is where a config axis is
        # asked for WITHOUT turning the autotuner off, and this is exactly that: the
        # pinned axes drop out of the space, the rest are still searched.  The
        # pinned set (names AND values) joins the cache key so a pinned and an
        # unpinned launch that happen to share a base cannot share a tuned
        # config.
        mem_pinned = tuple(sorted(k for k in (mem or {}) if k in cfg))
        # A `config=` may not turn HPACK on where the launch is not legal for it:
        # the packed layout is a property of the INPUTS, not a schedule knob, and
        # a caller pinning it would get wrong gradients with right shapes.
        if not hpack_ok:
            cfg["HP"] = 1
        # H, and (K-C) the smallest bias group, must both be whole packs: a tile
        # row is a head, its `(G,N)` bias row is fetched once for the whole tile.
        # Packed rotation loads gate biases per head, unlike the real schedule.
        _hg = HG_BC if ROT else min(HG_BC, HG_G)
        while cfg["HP"] > 1 and (H % cfg["HP"] or _hg % cfg["HP"]):
            cfg["HP"] //= 2
        HPACK = cfg["HP"] > 1
        if HPACK:
            # the pack IS the p tile, so the p-shard axis has nothing left to
            # shard and the chunk length is sized against HP, not against P.
            cfg["NP"] = 1
        # NP shards p across CTAs. It stays 1: sharding p works mechanically (the
        # p-contracted outputs ride the slot axes that already exist for h and for
        # the n shard, so the host reductions collapse them with no new pass, no
        # atomics and no zeroing) and cuts registers and spill, but it is SLOWER,
        # with time rising by the p-free work each shard replays. More resident
        # warps buy no throughput: the kernel is limited by the instruction
        # stream (the ABL ablations price its pieces), not by grid size.
        #
        # Legal only where it tiles P and leaves the tile wide enough for the
        # p-contraction to be worth a shuffle tree.
        while cfg["NP"] > 1 and (P % cfg["NP"] or P // cfg["NP"] < 8):
            cfg["NP"] //= 2
        NP = cfg["NP"]
        if cfg["NS"] is None:
            # Target ~4096 CTAs. The base grid is only (H, B), which at the
            # reference shape is 512 on 110 SMs -- a third-full last wave.
            #
            # Below 8 rows a shard replays more p-free work than the extra CTAs
            # buy back, so eight rows per shard is a floor (it binds at every
            # N=64 shape).
            #
            # The hard cap of 8 is a MEMORY cap, not a speed one: `x_s` is
            # (NS, B, H, P, L) fp32, i.e. NS x twice the size of `x`, so NS is the
            # one config knob that costs gigabytes.
            cfg["NS"] = max(1, min(triton.next_power_of_2(
                max(4096 // (H * Bs), 1)), nr // 8, 8))
        # The gate's adjoint needs `y_pre = sum_n C_n h_n`, and an
        # n-SHARD holds one SUMMAND of that sum, not the sum.  So a gated
        # backward runs unsharded.  It is a pin and not a refusal because NS
        # is a pure occupancy knob: at the P = 1 cells the gate is
        # used on, the (H//HP, B) grid is already thousands of CTAs and NS would
        # be 1 anyway.
        if z is not None:
            cfg["NS"] = 1
        # shards must tile the window exactly -- the kernel steps by nr // NS and
        # any remainder would be rows no CTA visits.
        while cfg["NS"] > 1 and nr % cfg["NS"]:
            cfg["NS"] //= 2
        if ROT:
            # a shard must own whole rotation pairs, or the partner row it needs is
            # in another CTA's half of the n loop
            while cfg["NS"] > 1 and (nr // cfg["NS"]) % 2:
                cfg["NS"] //= 2
        # Measured, not derived: every winner sits at ONE warp and
        #     BLOCK_P * BLOCK_L / (32 * NUM_WARPS) == 32
        # elements per thread -- BLOCK_L 32 at P=32, 16 at P=64 -- the same kind
        # of family the forward has at 16. BLOCK_P is pinned to P because
        # gB_t/gC_t contract over p, so BLOCK_L is the only lever.
        #
        # A smaller BLOCK_L (8) is not taken: bf16 loses much more than fp32
        # gains, and BLOCK_L is also the checkpoint stride, so 8 would
        # quadruple `hck` and bill the forward for writing it.
        # With a caller-supplied `hck` the chunk length is the FORWARD's stride,
        # which `functions.py` set from `auto_block_l(P, L)` -- so the P-only rule
        # is the one that has to be used here too, packed or not.
        #
        # And it is FORCED, not defaulted.  `cfg["BLOCK_L"]` is
        # reachable from three places -- `config=`, `mem=`, and the autotuner's
        # own corner grid -- and pass 2 reads `hck` slot j for chunk j with no
        # stride conversion (`hprev = load(hck_ptr + ckp + (j * N + n) * snc)`),
        # so a BLOCK_L that is not the forward's stride reads the wrong incoming
        # state at every chunk boundary: right shapes, silently wrong gradients
        # (e.g. a `v3_grid` MAXNREG corner that halves BLOCK_L).
        # `tests/test_mh_p1_rotation.py` is the gate.
        if hck is not None:
            cfg["BLOCK_L"] = auto_block_l(P, L)
        BL = (auto_block_l(cfg["HP"], L, cfg["BLOCK_L"], budget=HPACK_BUDGET)
              if (HPACK and hck is None) else auto_block_l(P, L, cfg["BLOCK_L"]))
        if cfg["NUM_WARPS"] is None:
            cfg["NUM_WARPS"] = 1
        if pmap and cfg["NUM_WARPS"] > 1 and BL < 32 * cfg["NUM_WARPS"]:
            # `pv` spreads warps along time too; a warp with no elements of its
            # own would broadcast, and `_put`'s lane dedup cannot see warps
            cfg["NUM_WARPS"] = max(1, BL // 32)
        if cfg["TP_P"] is None:
            # the p-contraction is warp-local only while TP_P divides the warp;
            # wider p wants a wider p-slice, and P//4 is the default.
            #
            # ROT halves it. A [BLOCK_L] vector on
            # SliceLayout(0) costs BLOCK_L*TP_P/(32*NUM_WARPS) registers, and the
            # pair walk keeps more of them live -- cos/sin, the partner's B/C, the
            # even row's gB0/gC0 -- so the p-slice has to give the room back.
            # Sized against the TILE width, which is HP under the packed
            # schedule and P // NP otherwise -- the p-contraction it widens does
            # not care which axis the tile holds.
            #
            # The ROT floor is 2 (not 4), and the search spans down to 2: at
            # P=16 a floor of 4 would clamp P//8 back up and cost the rotating
            # backward heavily, while the real (non-rotating) one prefers 4.
            _bp = cfg["HP"] if HPACK else P // cfg["NP"]
            cfg["TP_P"] = max(2 if ROT else 4,
                              min(_bp // 8 if ROT else _bp // 4, 32))
            # ...and then a FLOOR for 16-bit inputs, so the load width is a property
            # of the dtype rather than a coincidence of the two constants above.
            # The non-ROT configs already clear it, so this is inert on the
            # default path; it keeps a BLOCK_L refit from halving the bf16 load
            # width.
            #
            # Not under ROT, where it is not inert and loses: raising TP_P back
            # to a full transaction there costs the registers the P//8 rule
            # bought (more spill), and doubling the load width does not pay for
            # that.
            if not ROT:
                v = _vec_tp_p(x.element_size(), BL, cfg["NUM_WARPS"], _bp, L)
                if v is not None:
                    cfg["TP_P"] = max(cfg["TP_P"], v)
        cfg["BLOCK_L"] = BL

        # `bc_exact`, `BACC_*` and `has_bias` are resolved with the group counts
        # at the top of this function, because `hpack_ok` reads them and it runs
        # before `cfg` exists.  Both BC axes gate on `bc_exact`: folding the
        # slots destroys the head axis outright, and narrowing the slot DTYPE
        # leaves it but sums L rounded terms into a bias gradient, which is
        # where the precision of a length-L reduction actually goes (hence
        # `has_bias` on `bc_dt` below).  They are needed by this point because
        # the search's byte count prices the BC buffers.
        #
        # the vector red is fp32 and sm_90+; below either it falls back to the
        # scalar `gl.atomic_add`.
        _cc = torch.cuda.get_device_capability(x.device)
        vec_ok = _cc[0] >= 9
        # The fold is the DEFAULT wherever the vector red carries it.
        # It cannot be left to the search alone: in bf16 the unfolded base stores
        # BCBF slots, and the fold's fp32-accumulated gradients differ from those
        # by bf16 rounding -- more than `_agrees`' 2e-2 -- so a search STARTING
        # unfolded vetoes the fold as a numeric mismatch. Starting folded, the
        # same veto (and `_wins`' growth bar) is what keeps it.
        #
        # Scalar-atomic devices (sm_80, no red.v4) take the fold too: the MH
        # atomics are relaxed, whereas Triton's default acq_rel fence on every
        # scalar atomic would make the full slot axis win.
        if cfg["BCS"] is None:
            cfg["BCS"] = 1 if (bc_exact or (bc_head and not has_bias
                                           and g_B is None and g_C is None)) else 0
        # Same gate as BCS; `out["theta"]` is the only reader of the gtheta
        # slot axis.  Tied to the BCS pin as well: `mem=dict(AGX=0, BCS=0)` is
        # the determinism pin and it must not leave a cross-CTA atomic behind
        # on gtheta.
        # a per-head angle feature (theta_per_head) keeps its head axis: no fold
        th_fold_ok = ROT and cfg["BCS"] != 0 and not (theta is not None and theta.dim() == 4)
        if cfg["THS"] is None:
            cfg["THS"] = 1 if th_fold_ok else 0
        elif cfg["THS"] and not th_fold_ok:
            cfg["THS"] = 0

        _tier = autotune.tier()
        if config is None and out is None and not _pin and autotune.enabled():
            # Every rule above is a constant fitted offline, and a BLOCK_L refit
            # can silently invalidate the NS target. A runtime search is what
            # keeps them honest across a GPU change, and it costs one search per
            # shape, cached.
            #
            # Guards: `config` given means the caller pinned the config and must
            # get it; `out` given means the buffers are sized for THIS BLOCK_L/NS
            # and a trial would write the wrong shape. A caller-supplied `hck` pins
            # BLOCK_L for the same reason, but NS and the two thread knobs stay
            # free, so only that axis drops out.
            #
            # EVERY constexpr of the launch below belongs in this key, and nothing
            # else does. A constexpr left out lets two feature sets that compile to
            # DIFFERENT kernels share one tuned config, so whichever did not win the
            # search runs a config fitted to somebody else's kernel -- silent, and a
            # quality bug rather than a wrong answer. A non-constexpr put in splits
            # the cache for nothing and buys an extra 12-trial search. `beta_a` is
            # the example of the second: it is folded into `A_log` on the host, so
            # it never reaches a constexpr and must stay out. The head-group sizes
            # DO reach one (HG_BC/HG_G), and carry the presence flags with them.
            key = ("mh_bwd", _tier, slp.active(
                       torch.cuda.get_device_capability(x.device)[0] * 10
                       + torch.cuda.get_device_capability(x.device)[1]),
                   bool(bc_head), str(pmap), B_t.dtype, Bs, H, P, N, L, nr, n_off, x.dtype, LAW_MASK, ROT,
                   ROT_MIX, HAS_DYN, HAS_MU, A_DYN_N_RES, MU_N_RES,
                   DW_P, g_B is not None, g_C is not None, gh_final is not None,
                   h_init is not None, delta_w is not None, bool(delta_softplus),
                   cfg["RUN_P1"], cfg["RUN_TH"], hck is not None,
                   None if hck is None else hck.dtype,
                   cfg["HP"],
                   cfg["AGX"], cfg["BCS"], cfg["THS"], cfg["BCBF"], cfg["XSBF"], cfg["LWS"],
                   cfg["VBC"], cfg["VGX"], cfg["CKBF"],
                   cfg["DBACC"], cfg["DEBF"], d_skip is not None, z is not None,
                   cfg["TS"],
                   endpoint_own is not None, None if endpoint_own is None else endpoint_own.ndim,
                   os.environ.get("MH_BWD_MAXNREG"),
                   BACC_B, BACC_C, BACC_GB, BACC_GC, bias_part,
                   theta is not None, None if theta is None else theta.dim(),
                   bool(theta_delta) and theta is not None, theta_h is not None,
                   int(dyn_act), float(dyn_eps), int(gate_act),
                   mu_init is not None, bool(mu_prediff),
                   delta_bias is not None,
                   *(None if t is None else t.shape[0]
                     for t in (beta_b, beta_c, beta_gB, beta_gC)),
                   tuple((k, cfg[k]) for k in mem_pinned))
            # HOST FAST PATH: a key the search already settled in this process
            # skips building the trial closure, the axes and the corner grid
            # (`choose_axes` would return the same cached dict).
            _hit = autotune.cached(key)
            if _hit is not None:
                cfg = _hit
            else:
                kw = {k: v for k, v in _argv.items()
                      if k not in ("x", "delta", "A_log", "B_t", "C_t", "gy",
                                   "config", "out")}
                # K5: agreement at ordinary rates cannot see a BLOCK_L that overflows
                # only once the decay rate is extreme (Pass 1's chunk checkpoint,
                # `_tail_sum` above) -- two configs that both round the same finite
                # sum to the same finite digest can still differ on whether an
                # `exp2` of a huge exponent lands on 0.0 or +-inf, and the digest
                # never sees that difference happen. Every candidate this search
                # times is therefore ALSO launched once, untimed, on the caller's
                # own shape at `_EXTREME_A_LOG` (the K2 bisection's rate), and a
                # candidate that goes non-finite there is rejected outright --
                # `choose_axes`' `extreme=`, folded into the same accept/reject pass
                # so it costs one extra launch per candidate rather than a second
                # search. Only live when Pass 1 can run at all (`hck is None`; a
                # caller-handed-over checkpoint pins BLOCK_L to its own stride and
                # this branch's own tuning of it is already skipped).
                def trial(c, trial_alog=A_log):
                    result = mh_backward(
                        x, delta, trial_alog, B_t, C_t, gy, **kw,
                        config=dict(c, _resolved_bacc=bacc_on))
                    # The tuner fingerprints tuples of tensors, not the launcher's
                    # dictionary of gradients AND shape-dependent scratch buffers.
                    # Check the public adjoints without comparing scratch layouts.
                    return tuple(result[name] for name, value in _argv.items()
                                 if name not in ("gy", "hck", "out")
                                 and isinstance(value, torch.Tensor)
                                 and value.is_floating_point() and name in result)

                extreme = None if hck is not None else (
                    lambda c, _al=torch.full_like(A_log, _EXTREME_A_LOG): trial(c, _al))
                xs1 = Bs * H * P * L * 4
                ck_isz = 2 if (cfg["CKBF"] and x.element_size() == 2) else 4

                def ck_bytes(bl):
                    return Bs * H * (triton.cdiv(L, bl) + 1) * N * P * ck_isz

                des1 = Bs * H * L * 4
                # count of (H*NP, B, N, L)-shaped slot buffers a BCS fold moves
                nbc = 2 + (g_B is not None) + (g_C is not None)
                # gated pair maps: Bs_/Cs_ hold 2N rows per slab (the allocation below)
                bc_rows = 2 * N if (pmap and pmap != "raw") else N

                def cfg_bytes(c):
                    """The buffers the search can actually move, in bytes.

                    Not the peak and not meant to be: `Bs_`/`Cs_` and the inputs are
                    invariant across every candidate, so including them would only add
                    a constant to both sides of every comparison. What matters is that
                    this ranks candidates the way peak memory would, and these are the
                    only terms that differ between them.

                    `de_s` is here even though it is ~1/(N*P) of `x_s`, because it is
                    the ONLY term that still scales with NS once AGX has folded `x_s`
                    to one slab. Leave it out and the byte model is exactly flat in NS
                    on the default path, which turns `_wins` back into a pure-time rule
                    on the one axis that most needs the other.
                    """
                    ns = c["NS"] * c.get("NP", 1)
                    # the BC slot buffers joined when BCS became a searched axis;
                    # slab count and width match the allocation rule below (bf16 only
                    # off the atomic path).
                    hnp = (H // c.get("HP", 1)) * c.get("NP", 1)
                    slabs = max(1, min(c["BCS"] or hnp, hnp)) if bc_exact else hnp
                    if bc_head and not has_bias and g_B is None and g_C is None and c["BCS"]:
                        slabs = H
                    bcw = 2 if (bc_exact and c["BCBF"] and x.element_size() == 2
                                and not has_bias and slabs == hnp) else 4
                    # x_s at its allocated WIDTH: one unsharded slab of a 16-bit x is
                    # stored in x's dtype (`xs_dt` below), which counted as fp32 made
                    # NS=1 look twice its size to a bytes-aware `_wins`
                    atom_x = c["AGX"] and c["NS"] > 1
                    xw = 2 if (x.element_size() == 2 and not atom_x
                               and (c["NS"] == 1 or c.get("XSBF", 0))) else 4
                    return ((1 if atom_x else c["NS"]) * (xs1 // 4) * xw
                            + ns * des1
                            + (ck_bytes(auto_block_l(
                                c.get("HP", 1) if c.get("HP", 1) > 1 else P,
                                L, c["BLOCK_L"]))
                               if hck is None else 0)
                            + slabs * Bs * bc_rows * L * nbc * bcw
                            + max(1, min(c["LWS"] or hnp, hnp)) * Bs * N * L * 4
                            * (int(HAS_DYN and A_DYN_N_RES) + int(HAS_MU and MU_N_RES)))

                # TS (time split, `ts_adjoint.py`): offered to the search wherever the
                # envelope serves it; the trial times the seed kernel too.
                from .ts_adjoint import ts_refusal as _tsr
                # ...and only where the walk is long enough to amortise TS's two
                # extra launches (host time): the tuner times GPU work with an L2
                # flush between reps, which hides host cost, so a short walk can
                # win on GPU time and lose on wall time.
                _ts_ok = (hck is not None and not cfg["RUN_P1"]
                          and triton.cdiv(L, auto_block_l(P, L)) >= int(os.environ.get("MH_BWD_TS_MIN_NBLK", 64))
                      and _tsr(
                    ROT=ROT, pmap=pmap, LAW_MASK=LAW_MASK, hck=hck, delta_w=delta_w,
                    beta_b=beta_b, beta_c=beta_c, g_B=g_B, g_C=g_C,
                    endpoint_own=endpoint_own, bc_head=bc_head, nr=nr, N=N,
                    NP=cfg["NP"]) is None)
                cfg = autotune.choose_axes(
                    key, cfg,
                    {k: v for k, v in autotune.v3_axes(
                        cfg, nr, ROT, xs_bytes1=xs1, ckpt_bytes=ck_bytes,
                        tune_block_l=hck is None,
                        bc_axis=bc_exact and vec_ok,
                        ts_axis=_ts_ok).items()
                     if k not in mem_pinned
                     # `light` keeps the axes that usually win (NS, TP_P,
                     # BLOCK_L) and the time split (TS: long L / small batch);
                     # warps, AGX and the BC slab count are left to `full`
                     and (_tier == "full" or k in ("NS", "TP_P", "BLOCK_L", "TS"))},
                    trial,
                    bytes_of=cfg_bytes, extreme=extreme,
                    persist=_mh_bwd_kernel.cache_key,
                    # TS before the corners too: the split walk re-fits HP,
                    # TP_P and the cap, so a grid-first search can miss the
                    # joint optimum
                    pre=("TS",),
                    # HP and BLOCK_L are the coupled pair here, as BLOCK_P and
                    # BLOCK_L are in the forward: a corner grid, not a descent.
                    grid=(autotune.v3_grid(
                        # K-C: `_hg` and not `H`.  `v3_grid`'s filter is
                        # `hp <= X and X % hp == 0`, and the pack width must divide
                        # the SMALLEST bias group, not just H -- a tile row is a
                        # head and its `(G,N)` bias row is fetched once for the
                        # whole tile.  `_hg == H` with no bias, so this is
                        # inert everywhere else.  Without it the search reaches
                        # straight past the cap the launcher applied above, which
                        # is a wrong gradient with a right shape.
                        cfg, _hg, L,
                        # with a caller `hck` the chunk length is pinned to the
                        # forward's stride, so only HP is free.
                        (lambda hp: BL) if hck is not None else
                        (lambda hp: (auto_block_l(hp, L, budget=HPACK_BUDGET)
                                     if hp > 1 else auto_block_l(P, L))),
                        # a PINNED `MH_BWD_MAXNREG` binds the search too, so the
                        # A/B's `before` arm really is uncapped.
                        # The 168 cap helps on some devices and not others, so
                        # it is priced by the search like 128, never a default.
                        caps=() if _maxnreg()[1] else (128, 168),
                        half_block_l=hck is None)
                          if hpack_ok else None),
                    # a corner may not undo a `mem` pin either: a caller that
                    # pinned `HP=1` to reach the UNPACKED schedule must not be
                    # handed a packed corner -- and with it `DSIG`, i.e. a
                    # different function's gradient path than the one it asked
                    # for.
                    # (`test_h3_delta_bias_accumulator_is_refused_under_a_host_sigmoid`)
                    drop_keys=mem_pinned)
            # a tuned NS still has to tile the window, and whole rotation pairs
            # -- and a tuned NS may not undo the gate's pin above either.
            if z is not None:
                cfg["NS"] = 1
            while cfg["NS"] > 1 and (nr % cfg["NS"]
                                     or (ROT and (nr // cfg["NS"]) % 2)):
                cfg["NS"] //= 2
            # ...and a tuned HP still has to be a legal pack.  Belt and braces:
            # the grid above is already filtered, and a stale cached entry or a
            # future axis must not be able to reintroduce an illegal width.
            while cfg["HP"] > 1 and (H % cfg["HP"] or _hg % cfg["HP"]):
                cfg["HP"] //= 2
            HPACK = cfg["HP"] > 1
            # ...and a tuned BLOCK_L may not move the checkpoint stride either.
            # Same argument as the pin above, restated after the search because
            # the search is one of the three places that can set this key.
            if hck is not None:
                cfg["BLOCK_L"] = auto_block_l(P, L)
            BL = (auto_block_l(cfg["HP"], L, cfg["BLOCK_L"],
                               budget=HPACK_BUDGET)
                  if (HPACK and hck is None)
                  else auto_block_l(P, L, cfg["BLOCK_L"]))
            # ...and the pv warp clamp, which every trial applied on re-entry, so
            # the launch is the kernel the search timed (a cached verdict too)
            if pmap and cfg["NUM_WARPS"] > 1 and BL < 32 * cfg["NUM_WARPS"]:
                cfg["NUM_WARPS"] = max(1, BL // 32)
            # ...and the K18 theta fold, tied to BCS: a tuned BCS=0 (full slots)
            # makes each trial reset THS to 0 on re-entry, so the launch must too
            if cfg["THS"] and not (ROT and cfg["BCS"] != 0):
                cfg["THS"] = 0
        NBLK = triton.cdiv(L, BL)
        HP = cfg["HP"]
        # TS: resolve the super-chunk count against the envelope and NBLK.
        TS, NBS = 1, NBLK
        if cfg.get("TS", 1) > 1 and not cfg["RUN_P1"]:
            from .ts_adjoint import ts_refusal
            _why = ts_refusal(ROT=ROT, pmap=pmap, LAW_MASK=LAW_MASK, hck=hck,
                              delta_w=delta_w, beta_b=beta_b, beta_c=beta_c,
                              g_B=g_B, g_C=g_C, endpoint_own=endpoint_own,
                              bc_head=bc_head, nr=nr, N=N, NP=NP)
            if _why is None:
                TS = min(cfg["TS"], NBLK)
                # the kernel derives NBS = cdiv(NBLK, TS) itself; a trailing
                # super-chunk may be empty, which passes its seed through
                NBS = triton.cdiv(NBLK, TS)
        cfg["TS"] = TS
        # Where the softplus' derivative is taken.  In the kernel exactly where
        # the host pass is expensive relative to the register it costs: that is
        # the packed schedule, whose `gdelta` is the full (B,D,L) activation.
        DSIG = HPACK and bool(delta_softplus)
        # Legal exactly where the tile the kernel STORES is already
        # dL/d(delta): the kernel took the softplus derivative itself (DSIG), or
        # there is no softplus at all.  `delta_softplus and not DSIG` leaves a
        # host sigmoid between the store and the answer, and a partial taken
        # before it would be the gradient of the wrong function.
        DBACC = (delta_bias is not None and bool(cfg["DBACC"])
                 and (DSIG or not delta_softplus))
        # The checkpoint LAYOUT, which is the caller's when the caller brought
        # one.  It is a separate constexpr from HPACK because the two can differ:
        # `mh_forward` writes the packed layout at P == 1 whatever the backward
        # then decides its pack width to be, HP == 1 included.
        CKPACK = (hck.dim() == 4) if hck is not None else HPACK
        # channels per CTA and the CTA's own head-slot count: HP heads at P == 1
        # when packed, P channels of one head otherwise.
        BLK_P = HP if HPACK else P // cfg["NP"]
        HSL = (H // HP) * cfg["NP"]

        if _pk is not None:
            _tk = locals().get('key') if 'key' in locals() else None
            _PLANS[_pk] = ((A_DYN_N_RES, BACC_B, BACC_C, BACC_GB, BACC_GC, BL, BLK_P, Bs, CKPACK, DBACC, DSIG, DW_P, G_BC, G_G, H, HAS_DYN, HAS_MU, HG_BC, HG_G, HP, HPACK, HSL, K, L, LAW_MASK, MU_N_RES, N, NBLK, NBS, NP, P, ROT, ROT_MIX, TS, _cc, bc_exact, bias_part, cfg, has_bias, nr, vec_ok,), _tk, autotune.cached(_tk) if _tk is not None else None)
            _PLANS[_pk] = (tuple(dict(v) if isinstance(v, dict) else v for v in _PLANS[_pk][0]),) + _PLANS[_pk][1:]
    if out is None:
        dev, f32 = x.device, torch.float32
        zf = lambda *s: torch.zeros(*s, device=dev, dtype=f32)
        e = lambda *s: torch.empty(*s, device=dev, dtype=f32)
        # ONE placeholder for every absent feature's slot, instead of ten
        # separate `torch.zeros(1)`.  The kernel never dereferences them --
        # their feature's constexpr is off -- so they only have to be a valid
        # pointer, and each one otherwise costs an allocation and a memset
        # LAUNCH per call on the host's critical path.
        z1 = zf(1)
        # c1_s/hi_s are (B, H, N, P): p fastest, so the per-(chunk, n) write is
        # one sector. Slot 0 of hck is the zero h_init and is never written.
        #
        # Which of these may be `e`: this kernel has no atomics -- every
        # gradient store
        # is a plain store into a slot the CTA owns alone -- so a buffer needs
        # zeroing only where some element is left UNwritten. x_s/de_s/gdw_s
        # carry no n axis, and pass 2 stores them once per chunk over all NBLK
        # chunks unconditionally, so a partial n window still writes them
        # whole. c1_s/hi_s/grate/gs2o/gth_s are indexed by n or k and a partial
        # window does leave rows untouched, so those stay zeroed (a memset of
        # x_s and hck on every call would be pure waste).
        ns = cfg["NS"]
        # `x_s` is the largest buffer this kernel owns: (NS,B,H,P,L) fp32 is
        # NS x TWICE the size of `x`, written by the kernel and read back by the
        # host sum. Neither half can be recovered by going faster; only moving
        # fewer bytes, or not round-tripping them, helps. `AGX` and `XSBF` are
        # those two.
        #
        # XSBF halves the width. Each shard then rounds its partial before the
        # sum instead of the host rounding once at the end, so the error picks
        # up the condition number of that sum (a small flat relative error
        # against an fp32 oracle). Faster and smaller; it stays opt-in only
        # because the error is the user's to accept, and it is inert under AGX
        # (one slab, no host sum left to round).
        #
        # AGX removes the round-trip instead: all NS shards atomic_add onto ONE
        # slab, so the cross-shard sum happens in L2 and the host pass and the
        # NS-1 extra slabs both disappear. It costs a few percent of this
        # kernel's time for a third to a half of the step's memory, which is why
        # it is the DEFAULT and a config axis.
        #
        # Why the few percent and not more: the atomic must read the old value
        # where a store to a private address is fire-and-forget, and the
        # collapsed slab is far past L2 -- shards for one (h, b) sit H*Bs CTAs
        # apart in the grid, so the line is evicted before the next shard
        # reaches it and every atomic is a DRAM read-modify-write. That is a
        # wash on traffic, and it loses the tiebreak because the RMW sits on the
        # kernel's critical path while the host sum is a separate stream. It is
        # a wash precisely BECAUSE the atomic deletes traffic in proportion to
        # what it costs, which is the condition BCS below fails.
        #
        # What it is NOT is an accuracy argument: the results agree to the
        # last digits either way, so the "never atomicAdd" line buys
        # bitwise reproducibility, not correctness.
        atomic_gx = ns > 1 and cfg["AGX"]
        # The SAME question for the four h-slot buffers has a different answer.
        # Bs_/Cs_ are (H*NP,B,N,L) and at BCS=1 collapse to (B,N,L), a target
        # that trivially lives in L2, yet the full collapse costs far more time
        # than gx's. It is NOT contention: BCS keeps BCS slabs with
        # slot = (h*NP + pid_p) % BCS, so contention is H*NP/BCS, and the cost is
        # flat across that whole range.
        #
        # What is flat in contention and proportional to element count is the
        # INSTRUCTION: a [BLOCK_L] fp32 store is a handful of vector
        # transactions, while scalar atomics are per-element, so the innermost
        # loop -- entered N * NBLK times -- issues ~BLOCK_L ops where it issued a
        # few. gx survives this because its atomic DELETES proportional work
        # (NS-1 slabs of writes and a host reduction); the BC host sums are
        # small, so here the same instruction cost lands undiluted, and the
        # vector red (instruction count divided by the width) is what makes the
        # fold affordable. The rule: AN ATOMIC IS WORTH IT WHEN IT DELETES
        # TRAFFIC PROPORTIONAL TO WHAT IT COSTS.
        #
        # NP widens both slot axes: a p-shard contracts over only its own slice
        # of p, so its partial needs a slot exactly as an h or an n shard's
        # does. The host reductions below already collapse these axes, so this
        # is the whole cost of the p split on the reduction side.
        nsp = ns * NP
        law_slots = max(1, min(cfg["LWS"] or HSL, HSL))
        nsp_db = nsp * TS          # delta_bias partials: one slot per super-chunk too
        bcs = HSL
        # `bc_exact` (computed above, before the search) is the legality gate
        # for both BC axes.
        if bc_exact:
            bcs = max(1, min(cfg["BCS"] or bcs, bcs))
        elif bc_head and not has_bias and g_B is None and g_C is None and cfg["BCS"]:
            bcs = H
        atomic_bc = bcs < HSL

        # Halving the slot WIDTH instead of its count is the other axis, and on
        # this buffer it is the cheap one: a bf16 store stays a vector store,
        # so it moves half the bytes at the same instruction count -- exactly
        # what the atomic cannot do. NOT on the
        # atomic path: the vector red is fp32-only, and a bf16 fold would
        # accumulate H rounded terms where the store path rounds each term
        # once -- the folded slab is small anyway.
        bc_dt = x.dtype if (bc_exact and cfg["BCBF"] and not atomic_bc
                            and not has_bias
                            and x.element_size() == 2) else f32

        def bc(*rest):
            """The BCS-wide slot buffer gB_t/gC_t and the gates share."""
            return (torch.zeros if atomic_bc else torch.empty)(
                bcs, *rest, device=dev, dtype=bc_dt)
        # At NS == 1 there is no host sum over shards, so the input dtype is
        # not an accuracy axis at all: `functions.py` rounds `out["x"]` to
        # exactly this dtype on its way back to the caller, so storing it
        # here rounds the same value once instead of writing 2x the bytes and
        # rounding later.  XSBF stays the (opt-in, inexact) knob for NS > 1,
        # where each shard would round BEFORE the sum.
        xs_dt = x.dtype if (x.element_size() == 2 and not atomic_gx
                            and (ns == 1 or cfg["XSBF"])) else f32
        # Same argument for gdelta, and the same one-slot
        # condition -- but three consumers can still force fp32, and each
        # one is a REASON rather than a guess:
        #
        #   * a HOST softplus derivative still to be applied
        #     (`delta_softplus and not DSIG`): it READS this buffer and
        #     writes fp32, so a 16-bit slot would round twice.  Under
        #     `DSIG` the kernel already applied it, and with
        #     `delta_softplus=False` there is nothing to apply -- and that
        #     is the `m1exact` default, where the caller owns the softplus
        #     and hands `mh_scan` the activated delta;
        #   * `delta_bias`'s gradient when it is NOT reduced in the kernel
        #     -- a length-B*L reduction OF this buffer is where a 16-bit
        #     slot would really cost digits.  `DBACC` takes it
        #     off fp32 block partials instead, and then this buffer has no
        #     fp32 consumer left at all;
        #   * `delta.dtype`, not `x.dtype`: this buffer IS dL/d(delta) and
        #     `functions.py` rounds it to `delta`'s dtype.  The two agree
        #     on every shipping call, and keying off the wrong one would
        #     round a 16-bit gradient into an fp32 leaf.
        de_dt = delta.dtype if (cfg["DEBF"] and delta.element_size() == 2
                                and nsp == 1
                                and (DSIG or not delta_softplus)
                                and (delta_bias is None or DBACC)) else f32
        # an atomic target must start at zero, and only ONE slab is allocated.
        out = dict(x_s=(torch.zeros if atomic_gx else torch.empty)(
                       1 if atomic_gx else ns, Bs, H, P, L,
                       device=dev, dtype=xs_dt),
                   de_s=torch.empty(nsp, Bs, H, L, device=dev, dtype=de_dt),
                   # `delta_bias`'s block partials, one per (slot, b, head),
                   # in place of an `aten::sum` over the whole gradient.
                   gdb_s=(torch.empty(nsp_db, Bs, H, device=dev, dtype=f32)
                          if DBACC else z1),
                   # HPACK re-lays the four channel-fastest buffers as
                   # (B, [CKS,] N, D): the tile spans heads, so the axis a
                   # [BLOCK_P] access must be contiguous in is the CHANNEL
                   # one and not P.  Same element count either way.
                   # every row of a whole-N launch is written (gc1 from its
                   # first chunk on, ghi at t = 0), so only a window launch,
                   # whose unvisited rows the host still sums, needs zeros
                   # TS > 1: zeroed, because a trailing super-chunk may be
                   # empty (NBS = cdiv(NBLK, TS)) and writes no gc1 at all
                   c1_s=((zf if (nr != N or TS > 1) else e)(*((TS,) if TS > 1 else ()), Bs, N, H * P) if HPACK
                         else (zf if (nr != N or TS > 1) else e)(*((TS,) if TS > 1 else ()), Bs, H, N, P)),
                   hi_s=((zf if nr != N else e)(Bs, N, H * P) if HPACK
                         else (zf if nr != N else e)(Bs, H, N, P)),
                   Bs_=bc(Bs, 2 * N if (pmap and pmap != "raw") else N, L),
                   Cs_=bc(Bs, 2 * N if (pmap and pmap != "raw") else N, L),
                   # CKBF: the checkpoints in the INPUT dtype. Pass 1's chain
                   # and pass 2's seed read them back through a `.to(f32)`.
                   hck=torch.empty(
                       *((Bs, NBLK + 1, N, H * P) if CKPACK
                         else (Bs, H, NBLK + 1, N, P)), device=dev,
                       dtype=x.dtype if (cfg["CKBF"]
                                         and x.element_size() == 2) else f32)
                   if hck is None else hck,
                   alam=(e(*((TS,) if TS > 1 else ()), Bs, N, H * P) if HPACK
                         else e(*((TS,) if TS > 1 else ()), Bs, H, N, P)),
                   # Theta's own chunk carries. thck is p-free and rcar is
                   # p-replicated, so both are ~1/(N*P) and ~1/N of hck.
                   # pass 0 writes every slot pass 2 reads; the host never
                   # reads it
                   thck=e(Bs, H, NBLK, K) if ROT else z1,
                   rcar=e(Bs, H, K, P) if ROT else z1,
                   # The per-token angle's (HSL,B,K,L) adjoint buffer; its
                   # slot axis folds under `THS`.
                   gth_s=zf(min(HSL, cfg["THS"]) if cfg["THS"] else HSL,
                            Bs, K, L) if ROT else z1,
                   # theta_h's gradient: one partial per (slot, b, pair, chunk).
                   gthh_s=(zf if nr != N else e)(H if HPACK else HSL, Bs, K, NBLK)
                   if (ROT and theta_h is not None)
                   else z1,
                   # n-resolved law slots are exact per n up to the p split, so
                   # they take the h slot axis; the h-resolved ones contract
                   # over n and take de_s' shape.
                   c2_s=zf(Bs, H, N, P) if HAS_MU else z1,
                   grate=(zf(law_slots, Bs, N, L) if A_DYN_N_RES
                          else zf(nsp, Bs, H, L)) if HAS_DYN else z1,
                   gs2o=(zf(law_slots, Bs, N, L) if MU_N_RES
                         else zf(nsp, Bs, H, L)) if HAS_MU else z1,
                   # gdw_s mirrors whichever of x_s/de_s delta_w's own rank
                   # matches, since the store is the same shard-then-sum. Under
                   # DW_P it is p-INDEXED, so it keeps `gxb` and the ns axis.
                   gdw_s=(e(ns, Bs, H, P, L) if DW_P else e(nsp, Bs, H, L))
                   if delta_w is not None else z1,
                   # the gate adjoints contract over h exactly as gB_t/gC_t do,
                   # so they take the same slot shape and the same host sum.
                   ggB_s=bc(Bs, N, L) if g_B is not None else z1,
                   ggC_s=bc(Bs, N, L) if g_C is not None else z1,
                   # K-C: (G, N) fp32, atomic targets, so zeroed.  One
                   # element per bias entry -- 2 KiB at G = 32, N = 16 --
                   # against the `H*NP` slabs of (B, N, L) they replace.
                   gbb_s=zf(G_BC, N) if BACC_B else z1,
                   gbc_s=zf(G_BC, N) if BACC_C else z1,
                   gbgB_s=zf(G_G, N) if BACC_GB else z1,
                   gbgC_s=zf(G_G, N) if BACC_GC else z1)
    if bias_part:
        # Each CTA owns its chunk/row partial; NS shards disjoint N rows.
        # Zero initialization also defines rows outside a partial launch.
        shape = ((H // HP) * NP * Bs * NBLK, N)
        if BACC_GB:
            out["gbgB_s"] = torch.zeros(shape, device=x.device, dtype=torch.float32)
        if BACC_GC:
            out["gbgC_s"] = torch.zeros(shape, device=x.device, dtype=torch.float32)
    # A forward-supplied hck may carry more slots than NBLK+1 (its chunking is
    # its own), so the stride comes from the buffer, never from NBLK.
    cks = out["hck"].shape[1] if CKPACK else out["hck"].shape[2]
    # ...but it may not carry FEWER, and that is the one place a chunk length
    # that is not the forward's stride shows up as a shape rather than as a
    # wrong number.  Pass 2 reads slot `j` for chunk `j` (`hprev` above), so
    # NBLK + 1 slots is not an optimisation, it is the indexing contract.
    # Cheap (a host-side shape test) and it turns a silent wrong gradient
    # into a refusal.
    if hck is not None and cks < NBLK + 1:
        raise ValueError(
            "the handed-over hck carries %d chunk slots but this backward "
            "chunks L=%d at BLOCK_L=%d (%d chunks): the checkpoint stride "
            "and the backward's chunk length must be the same number"
            % (cks, L, BL, NBLK))
    # Same rule for the two slot axes: a NARROWER buffer than the shard count
    # means the missing slots fold atomically, so the BUFFER decides, not the
    # config -- a caller-supplied `out` never re-reads it, and the kernel must
    # agree with what was zeroed.
    atomic_gx = out["x_s"].shape[0] == 1 < cfg["NS"]
    bcs = out["Bs_"].shape[0]
    atomic_bc = bcs < HSL
    if bc_head and atomic_bc and (bcs != H or has_bias or g_B is not None or g_C is not None):
        raise ValueError("prepared-map reduction needs one slab per head and no unfused B/C features")
    law_slots = (out["grate"].shape[0] if HAS_DYN and A_DYN_N_RES else
                 out["gs2o"].shape[0] if HAS_MU and MU_N_RES else HSL)
    if HAS_DYN and A_DYN_N_RES and HAS_MU and MU_N_RES:
        if out["grate"].shape[0] != out["gs2o"].shape[0]:
            raise ValueError("dynamic-rate and mu adjoint slots must match")
    atomic_law = law_slots < HSL
    for name, active in (("grate", HAS_DYN and A_DYN_N_RES),
                         ("gs2o", HAS_MU and MU_N_RES)):
        if active and atomic_law and out[name].dtype != torch.float32:
            raise ValueError("folded law adjoint slots must be fp32")
    # the BUFFER decides here too -- a caller-supplied `out` never
    # re-reads `cfg`, and the kernel must agree with what was zeroed.
    ths = out["gth_s"].shape[0] if (ROT and theta is not None) else HSL
    atomic_th = ths < HSL
    # a bf16 slot buffer never reaches the atomic path (bc_dt above), but a
    # CALLER-supplied one could; the scalar fallback cannot add bf16 either,
    # so refuse rather than corrupt.
    if atomic_bc and out["Bs_"].dtype != torch.float32:
        raise ValueError("a folded (atomic) BC slot buffer must be fp32")

    # Vector width for the two atomic paths. A pack must be a whole, aligned,
    # mask-uniform run of L: per-thread runs are spt_l consecutive L elements
    # (order [1, 0]) starting at multiples of spt_l, rows start at multiples
    # of L, so spt_l % w == 0 and L % w == 0 give consecutive 16-byte-aligned
    # addresses and one mask bit per pack. fp32 red.v{2,4} is sm_90+.
    tp_p = min(cfg["TP_P"], BLK_P)
    tp_l = max(32 // tp_p, 1)
    # a p-warp needs a p-slice of its own; past that it would only replicate
    WP = max(1, int(cfg.get("WP", 1) or 1))
    while WP > 1 and BLK_P < tp_p * WP:
        WP //= 2
    cfg["WP"] = WP
    NW_T = cfg["NUM_WARPS"] * WP
    spt_l = max(BL // (tp_l * cfg["NUM_WARPS"]), 1)

    def _vecw(on):
        if on and vec_ok:
            for w_ in (4, 2):
                if spt_l % w_ == 0 and L % w_ == 0:
                    return w_
        return 0

    bc_vec = _vecw(atomic_bc and cfg["VBC"])
    # PMAP's `pv` layout: lanes (and warps) on time, BL // (32 * NW) per
    # thread; below 32 elements per warp the high lanes broadcast and the
    # owner-lane count dedups them.
    spt_v = max(BL // (32 * NW_T), 1)
    tpl_v = min(32, max(BL // NW_T, 1))
    vec_v = 0
    # (with p-warps `pv` spans NW_T warps and can broadcast across them,
    # which the owner-lane count cannot see: scalar atomics there)
    if pmap and vec_ok and cfg["VBC"] and (WP == 1 or BL >= 32 * NW_T):
        for w_ in (4, 2):
            if spt_v % w_ == 0 and L % w_ == 0:
                vec_v = w_
                break
    th_vec = _vecw(atomic_th and cfg["VBC"])
    gx_vec = _vecw(atomic_gx and cfg["VGX"])

    def _pfold(t):
        """Fold the p-shard sub-axis out of an h-slot reduction.

        Slot is h*NP + pid_p, so anything that has collapsed the trailing axes
        but kept the slot one still has to sum NP consecutive entries before it
        can be regrouped by head.
        """
        return t if NP == 1 else t.reshape(-1, NP, *t.shape[1:]).sum(1)
    ck0 = out["hck"][:, 0] if CKPACK else out["hck"][:, :, 0]
    if h_init is not None and hck is None:
        ck0.copy_(_f32(h_init).reshape(Bs, H * P, N).permute(0, 2, 1)
                  if CKPACK else
                  _f32(h_init).reshape(Bs, H, P, N).permute(0, 1, 3, 2))
    elif hck is None:
        # slot 0 is the only part pass 1 leaves alone, so it is the only part a
        # memset still has to cover. A caller-supplied hck brought its own,
        # including a nonzero h_init. Do not mutate autograd's saved tensor:
        # even an identical copy increments its version on retained graphs.
        ck0.zero_()
    # gh_final is the mirror of h_init: pass 2 walks chunks backwards reading
    # `alam` as the incoming adjoint, so seeding it is the whole feature.
    if gh_final is not None and TS == 1:
        out["alam"].copy_(
            _f32(gh_final).reshape(Bs, H * P, N).permute(0, 2, 1) if HPACK else
            _f32(gh_final).reshape(Bs, H, P, N).permute(0, 1, 3, 2))

    # beta_a offsets the exponent on every row, and that is a
    # host copy already, so the whole feature is this fold plus its adjoint below.
    # (G_BC/HG_BC/G_G/HG_G are resolved at the top -- K-C's routing needs
    # them before `hpack_ok`.)
    alog_e = _f32(A_log)
    if beta_a is not None:
        ba_h = _f32(beta_a)[group_index(H, beta_a.shape[0], x.device)] \
            .view(H, 1, N)
        alog_e = alog_e + ba_h

    gd_skip = (torch.zeros(H, device=x.device, dtype=torch.float32)
               if d_skip is not None
               else torch.zeros(1, device=x.device, dtype=torch.float32))
    # dL/dz, channel-major and in z's own dtype.  Every element
    # is written exactly once (NS is pinned to 1 under a gate), so this is
    # a plain store and needs no slot axis and no zeroing.
    gz = (torch.empty_like(z) if z is not None
          else torch.zeros(1, device=x.device, dtype=torch.float32))
    endpoint_grad = torch.zeros_like(out["gdw_s"] if endpoint_own.ndim == 4 else out["de_s"], dtype=torch.float32) if endpoint_own is not None else None
    global LAST_COMPILED, LAST_CONFIG
    LAST_CONFIG = dict(cfg, BC_HEAD=bool(bc_head), BLOCK_L=BL, BLOCK_P=BLK_P, HPACK=HPACK,
                       CKPACK=CKPACK, DSIG=DSIG, DBACC_ON=DBACC, NBLK=NBLK,
                       grid=((H // HP) * NP, Bs, cfg["NS"]))

    def _stat(t):
        """An (H,P,N) static in the layout the schedule reads it in.

        Unpacked: (H,N,P), p fastest inside a head.  Packed: (N,D), the
        CHANNEL fastest across heads -- the same permutation one level up,
        because the tile axis is one level up.
        """
        if t is None:
            return None
        return (t.reshape(H * P, N).permute(1, 0).contiguous() if HPACK
                else t.permute(0, 2, 1).contiguous())

    if TS > 1:
        from .ts_adjoint import ts_seeds
        # written straight into alam's layout ((TS,B,N,D) packed, else (TS,B,H,N,P))
        ts_seeds(TS, NBS * BL, x, delta, alog_e, C_t, gy,
                 delta_bias=delta_bias, delta_softplus=delta_softplus,
                 z=z, gh_final=gh_final, out=out["alam"], packed=HPACK)
    LAST_COMPILED = _mh_bwd_kernel[((H // HP) * NP, Bs, cfg["NS"] * TS)](
        x.contiguous(), delta.contiguous(),
        _stat(alog_e),
        B_t.contiguous(), C_t.contiguous(), gy.contiguous(),
        None if theta is None else theta.contiguous(),
        _thh_table(theta_h),
        _f32(theta_init) if theta_init is not None else
        # read only under ROT (pass 0); a placeholder otherwise -- one memset
        # launch less per call
        (torch.zeros(Bs, H, K, device=x.device, dtype=torch.float32) if ROT
         else out["thck"]),
        row_law, row_rot,
        None if a_dyn is None else a_dyn.contiguous(),
        None if mu is None else mu.contiguous(),
        _stat(_f32(mu_gain)),
        _f32(mu_init), _f32(delta_bias), _f32(beta_b), _f32(beta_c),
        None if delta_w is None else delta_w.contiguous(),
        None if g_B is None else g_B.contiguous(),
        None if g_C is None else g_C.contiguous(),
        _f32(beta_gB), _f32(beta_gC),
        out["x_s"], out["de_s"], out["c1_s"], out["hi_s"],
        out["Bs_"], out["Cs_"], out["gth_s"], out["gthh_s"],
        out["c2_s"], out["grate"], out["gs2o"], out["gdw_s"],
        out["ggB_s"], out["ggC_s"],
        out["hck"], out["alam"], out["thck"], out["rcar"],
        None if d_skip is None else d_skip.float().contiguous(), gd_skip,
        None if z is None else z, gz, endpoint_own, endpoint_grad,
        out["gdb_s"],
        out["gbb_s"], out["gbc_s"], out["gbgB_s"], out["gbgC_s"],
        Bs, H, P, N, L, K, NBLK, cks,
        BLOCK_L=BL, BLOCK_P=BLK_P, NUM_WARPS=NW_T, WP=WP,
        HPACK=HPACK, HP=HP, DSIG=DSIG, CKPACK=CKPACK, BC_HEAD=bool(bc_head),
        TP_P=cfg["TP_P"],
        HAS_DSKIP=d_skip is not None, DBACC=DBACC, HAS_Z=z is not None,
        HAS_ENDPOINT=endpoint_own is not None,
        ENDPOINT_P=endpoint_own is not None and endpoint_own.ndim == 4,
        ABL=int(os.environ.get("MH_BWD_ABL", cfg["ABL"])),
        RUN_P1=cfg["RUN_P1"], NS=cfg["NS"], NP=NP, TS=TS,
        ATOMIC_GX=atomic_gx, ATOMIC_BC=atomic_bc, BCS=bcs,
        LWS=law_slots if atomic_law else 0, LAW_VEC=_vecw(atomic_law and cfg["VBC"]),
        ATOMIC_TH=atomic_th, THS=ths, TH_VEC=th_vec,
        BC_VEC=bc_vec, GX_VEC=gx_vec, TPL=tp_l, ROT=ROT, PMAP=bool(pmap),
        PRAW=(pmap == "raw"),
        SLP=slp.active(_cc[0] * 10 + _cc[1]),
        VEC_V=vec_v, TPL_V=tpl_v,

        THETA_DELTA=bool(theta_delta) and theta is not None,
        THETA_PH=theta is not None and theta.dim() == 4,
        HAS_THH=theta_h is not None,
        RUN_TH=cfg["RUN_TH"], FAST_TRIG=bool(cfg["FAST_TRIG"]),
        PAIR_UNROLL=bool(cfg.get("PU", 0)),
        LAW_MASK=LAW_MASK, DYN_ACT=int(dyn_act), DYN_EPS=float(dyn_eps),
        A_DYN_N_RES=A_DYN_N_RES, MU_N_RES=MU_N_RES,
        HAS_MU_INIT=mu_init is not None, MU_PREDIFF=bool(mu_prediff),
        DELTA_SOFTPLUS=bool(delta_softplus),
        HAS_DBIAS=delta_bias is not None,
        HAS_BB=beta_b is not None, HAS_BC=beta_c is not None, HG_BC=HG_BC,
        BACC_B=BACC_B, BACC_C=BACC_C, BACC_GB=BACC_GB, BACC_GC=BACC_GC, BGPART=bias_part,
        HAS_GHL=gh_final is not None,
        HAS_DW=delta_w is not None, DW_P=DW_P,
        N_OFF=n_off, N_ROWS=nr, ROT_MIX=ROT_MIX,
        HAS_GB=g_B is not None, HAS_GC=g_C is not None,
        GATE_ACT=int(gate_act), HAS_BGB=beta_gB is not None,
        HAS_BGC=beta_gC is not None, HG_G=HG_G,
        # Pinned False, not a knob, on two independent grounds. It LOSES:
        # dropping the tail predicate is worth 3-9% on the forward, which
        # holds 96 of its 255 registers, and costs 0-5% here, which holds all
        # 255 and already spills -- the predicate shortens live ranges, so
        # without it ptxas keeps more values live and spills 68 -> 78 bytes.
        # And it is WRONG: at True, `ghl long` and `dw long` come back NaN
        # even though L % BLOCK_L == 0, so some mask reached by EVEN_L is not
        # purely the tail. Unpicking that buys a slowdown, so it stays off.
        EVEN_L=False,
        num_warps=NW_T, maxnreg=cfg["MAXNREG"],
    )
    if nr != N:
        # Bs_/Cs_/ggB_s/ggC_s/rcar come from torch.empty, so the rows no CTA
        # visited hold garbage rather than zero -- and every reduction below
        # reads all N. The zero-filled buffers (c1_s, hi_s, gth_s, grate, gs2o)
        # need nothing.
        for buf in (out["Bs_"], out["Cs_"]) + \
                ((out["ggB_s"],) if g_B is not None else ()) + \
                ((out["ggC_s"],) if g_C is not None else ()):
            mf = 2 if (pmap and buf.shape[2] == 2 * N) else 1
            buf[:, :, :mf * n_off] = 0.0
            buf[:, :, mf * (n_off + nr):] = 0.0
        if ROT:
            out["rcar"][:, :, :n_off // 2] = 0.0
            out["rcar"][:, :, (n_off + nr) // 2:] = 0.0
    def _unstat(t):
        """A (B,H,N,P) / (B,N,D) scratch slab back to the ABI's (B,H,P,N).

        Both are free views: unpacked it is the trailing transpose the layout
        note describes, packed it is (B,N,D) -> (B,D,N) -> (B,H,P,N) with
        P == 1, so the last step only names an axis of length one.
        """
        return (t.permute(0, 2, 1).unsqueeze(2) if HPACK
                else t.permute(0, 1, 3, 2))

    out["c1"] = _unstat(out["c1_s"].sum(0) if TS > 1 else out["c1_s"])   # (P, N) view
    out["h_init"] = _unstat(out["hi_s"])
    if h_init is not None and gh_final is not None and nr != N:
        # A row outside the scan window is an identity carry, not a zero.
        # Its final-state adjoint bypasses the recurrence entirely.
        out["h_init"][..., :n_off] = gh_final[..., :n_off].float()
        out["h_init"][..., n_off + nr:] = gh_final[..., n_off + nr:].float()
    # one slab means nothing to sum: either NS==1, or the shards already
    # summed themselves in L2 on the way in.
    out["x"] = _hsum(out["x_s"])
    if d_skip is not None:
        out["d_skip"] = gd_skip.reshape(d_skip.shape)
        if mem is not None and mem.get("BCS") == 0:
            # the contract determinism pin: the kernel accumulates dD with
            # relaxed atomics from every tile of a head (one per tile ROW under
            # HPACK), which would make grad(d_skip) vary run to run under
            # mem=dict(AGX=0, BCS=0).  Under the pin dD is the fixed-order host sum
            # dL/dD_h = sum_{b,p,t} gy_eff * x, gy_eff = gy * silu(z) when gated
            # (the skip is inside what the gate multiplies).
            gyf = gy.float()
            if z is not None:
                zf = z.float()
                gyf = gyf * (zf * torch.sigmoid(zf))
            out["d_skip"] = (gyf * x.float()).sum(dim=(0, 2, 3)).reshape(d_skip.shape)
    if z is not None:
        out["z"] = gz
    # `out["delta"]` is dL/d(delta), softplus' own derivative included, as
    # `functions.py` documents.  Under DSIG the kernel already applied it
    # (`dsig` in `_de_load`), which deletes a host sigmoid and a multiply over
    # a (B,H,L) fp32 tensor the scan holds in registers anyway.  Same sigmoid,
    # so the two routes agree.
    gde = _hsum(out["de_s"])
    if delta_softplus and not DSIG:
        d_raw = delta.float() if delta_bias is None else \
            delta.float() + _f32(delta_bias).view(1, H, 1)
        gde = gde * torch.sigmoid(d_raw)
    out["delta"] = gde
    if delta_bias is not None:
        # off the kernel's own chunk partials where it took them
        # (a (slot, B, H) fp32 buffer, fixed-order host sum), and off the
        # whole [B, H, L] gradient otherwise -- which is the route
        # `MH_BWD_DBIAS_ACC=0` and a host softplus derivative both keep.
        out["delta_bias"] = (out["gdb_s"].sum((0, 1)) if DBACC
                             else gde.sum((0, 2)))
    if delta_w is not None:
        gdw = _hsum(out["gdw_s"])
        # delta_w has no bias of its own, so unlike gde this is the softplus
        # derivative at the raw tensor.
        if delta_softplus:
            gdw = gdw * torch.sigmoid(delta_w.float())
        out["delta_w"] = gdw
    gc1_h = out["c1"].sum(0)                            # (H,P,N), unmasked
    zc = torch.zeros_like(gc1_h)
    # gc1 is dL/dc1 for every row; which parameter that IS depends on the law,
    # and the two are disjoint, so one buffer serves both.
    lawv = None if row_law is None else row_law.view(1, 1, N)
    # A law with no row in the launched window has no scratch buffer, but its
    # parameter's gradient still has to BE here -- it is exactly zero, and
    # functions.py must return one gradient per input either way.
    if mu_gain is not None:
        out["mu_gain"] = torch.where(
            lawv == LAW_MU, _unstat(out["c2_s"]).sum(0), zc) \
            if HAS_MU else zc
    if a_dyn is not None:
        gr_ = _hsum(out["grate"]) \
            if HAS_DYN else torch.zeros_like(a_dyn, dtype=torch.float32)
        out["a_dyn"] = gr_ * _dact(a_dyn.float(), dyn_act)
    if mu is not None:
        # the kernel emits dL/ds2 per step; s2 telescopes, so mu_t picks up its
        # own step minus the next one's. Under mu_prediff the input
        # IS s2, so that difference belongs to whoever took the forward one --
        # taking it here as well would apply the adjoint twice.
        gs = _hsum(out["gs2o"]) \
            if HAS_MU else torch.zeros_like(mu, dtype=torch.float32)
        if mu_prediff:
            out["mu"] = gs
        else:
            out["mu"] = gs - torch.cat(
                [gs[..., 1:], torch.zeros_like(gs[..., :1])], dim=-1)
            if mu_init is not None:
                out["mu_init"] = -gs[..., 0]
    # c1 = -exp(A_log + beta_a), so dc1/dA_log = dc1/dbeta_a
    # = c1; beta_a takes gc1 raw and
    # A_log takes nothing.
    gal = gc1_h * -torch.exp(alog_e)
    out["A_log"] = gal
    if beta_a is not None:
        G = beta_a.shape[0]
        gba = gal.sum(1)                                    # (H,N)
        out["beta_a"] = gba.reshape(G, H // G, N).sum(1)
    if cfg.get("REDUCE", True):
        # one slab means the h sum already happened in L2 (ATOMIC_BC); more
        # than one is the slot buffer, summed in fixed host order.
        if bc_head:
            # Reduce channel partitions, retaining the prepared map head.
            out["B_t"] = out["Bs_"].reshape(H, NP, Bs, N, L).sum(1) if bcs > H else out["Bs_"]
            out["C_t"] = out["Cs_"].reshape(H, NP, Bs, N, L).sum(1) if bcs > H else out["Cs_"]
        else:
            out["B_t"] = _hsum(out["Bs_"])
            out["C_t"] = _hsum(out["Cs_"])

        def _beta(slot, folded, acc, G, on_acc):
            """A static B/C bias' gradient, off whichever route served it.

            beta is the bias on the SAME quantity the slot buffer holds the
            adjoint of, so it is always a reduction of one of three things,
            and which one is exactly K-C's routing:

            * `on_acc` -- the kernel accumulated it into a (G, N) buffer
              (GROUPED bias);
            * G == 1 -- the sum over every head, batch and step, which is
              what the FOLDED (B, N, L) gradient already is;
            * otherwise -- the read-back off the per-head slot axis,
              which is what `BACC=0`/`BCS=0` keep the axis for.
            """
            if on_acc:
                return acc
            if slot.shape[0] == 1:
                # `.sum(-1)` before `.sum(0)`: L is the contiguous axis.
                return folded.sum(-1).sum(0).reshape(1, N)
            # The per-head slot route.  The slot axis counts what the GRID counts,
            # which is heads only at `HP == 1` -- under the pack (which a
            # head-shared bias can reach, §11.4) it counts head
            # GROUPS of `HP`, and the regroup has to be taken off the
            # buffer's own length rather than off `H/G`.  `HP` divides the
            # bias group by the launcher's own cap, so this is exact.
            per_g = slot.shape[0] // NP // G
            return _pfold(slot.sum(-1).sum(1)).reshape(G, per_g, N).sum(1)

        if beta_b is not None:
            out["beta_b"] = _beta(out["Bs_"], out["B_t"], out["gbb_s"],
                                  G_BC, BACC_B)
        if beta_c is not None:
            out["beta_c"] = _beta(out["Cs_"], out["C_t"], out["gbc_s"],
                                  G_BC, BACC_C)
        for sk, gk, gt, bk, bt, ak, on in (
                ("ggB_s", "g_B", g_B, "beta_gB", beta_gB, "gbgB_s", BACC_GB),
                ("ggC_s", "g_C", g_C, "beta_gC", beta_gC, "gbgC_s", BACC_GC)):
            if gt is None:
                # a bias on an absent gate reaches nothing, but functions.py
                # still owes its input a gradient.
                if bt is not None:
                    out[bk] = torch.zeros_like(bt, dtype=torch.float32)
                continue
            out[gk] = _hsum(out[sk])                # (H,B,N,L) -> (B,N,L)
            if bt is not None:
                out[bk] = (out[ak].reshape(G_G, -1, N).sum(1)
                           if bias_part and on else
                           _beta(out[sk], out[gk], out[ak], G_G, on))
        if ROT:
            if theta.dim() == 4:            # (H*NP, B, K, L) -> (B, H, K, L)
                gs = out["gth_s"]
                out["theta"] = gs.reshape(H, -1, *gs.shape[1:]).sum(1).permute(1, 0, 2, 3)
            else:
                out["theta"] = _hsum(out["gth_s"])
            if theta_h is not None:
                out["theta_h"] = _pfold(out["gthh_s"].sum((1, 3)))
            # after the last (j == 0) chunk rcar holds Rv at t=0, which is
            # exactly gtheta_init. It is replicated within each P shard,
            # but different shards carry DIFFERENT partial sums. Retain one
            # column per shard and reduce all shards (NP=1 is a free view).
            out["theta_init"] = (out["rcar"][..., 0] if NP == 1 else
                                  out["rcar"][..., ::P // NP].sum(-1))
        else:
            # an angle was supplied but no launched row rotates, so it reaches
            # nothing: the gradient is zero, not absent.
            for k_, t_ in (("theta", theta),
                           ("theta_init", theta_init), ("theta_h", theta_h)):
                if t_ is not None:
                    out[k_] = torch.zeros_like(t_, dtype=torch.float32)
    if endpoint_own is not None:
        out["endpoint_own"] = _hsum(endpoint_grad)
    return out
