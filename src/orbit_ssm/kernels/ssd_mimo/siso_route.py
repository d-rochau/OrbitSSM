"""Route selection for the SISO fast path -- `SSD_MIMO_SISO_FAST`.

`siso_enabled` decides the default.  When on, an `_SSDMimoFn` call whose SHAPE is inside the
envelope below runs `siso_fwd_kernel` / `siso_bwd_kernel` instead of
`fused_chunk_fwd` / `fused_chunk_bwd_5kernel`; everything else -- every
(R, O) above 1, grouped B/C, a wide head, the in-kernel gate, a host
phase offset -- stays on the existing fused route, unchanged and
untouched.  This is the "specialised SCHEDULE inside the same family"
clause the `mh` P = 1 kernel already uses; it is not a third kernel family and it is not reachable except
through `ssd_mimo_scan`.

THE ENVELOPE (`siso_refusal` returns the reason, by name, or None):

  R == O == 1                the schedule has no r/o loop
  head-SHARED B/C            `bc_grouped` off, Gb == Gc == 1
  G_SIZE <= 64, %16 == 0     ONE G tile per head: phase 1 stores the
                             per-head maps exactly once, which needs
                             NGT == 1 (it is also what lets the backward
                             overwrite them in place)
  the CUMULATED phase        `theta_off` must be None; a host offset
                             is refused rather than silently ignored
  no IN-KERNEL gate          `SSD_MIMO_GATE_INKERNEL=1` is refused (the
                             DEDICATED gate epilogue, the default, runs
                             outside the scan and is untouched)

MEMORY.  With a rotation the schedule stores two per-head maps
`[B, L, H, N]` in the IO dtype and always stores `S_all` (the
`SSD_MIMO_SALL=io` policy, the faster and lower-peak choice for the
pass).  The stored maps cost two `[B, L, H, N]` IO-dtype tensors; the
backward writes `dB~` / `dC~` OVER them, so the pass never holds
four.  The rotation-free schedule stores nothing extra at all.

At wide-head cells the maps are FORWARD-HELD, because `ctx.siso_bwd` sends
the rotating case to the SISO backward.  `siso_fuse` below is the lever
that takes this schedule's SPEED without that memory.
"""

import contextlib
import os

import torch

from .fused_fwd_kernel import _env
from .. import portable as _portable


def gluon_default_route(device, rotating: bool, io_bf16: bool) -> bool:
    """The Hopper default: a ROTATING
    R = O = 1 training call in bf16 IO on sm_90 takes the SISO schedule with
    the Gluon scan adjoint (`gluon_sm90_siso_bwd`) at T = 64.  It is an
    ALGORITHM/route choice that cannot run on other parts (wgmma/TMA), so it
    is keyed on the device's capability like `_seg_plan`, not a launch-config search.  Off when either lever is pinned
    to "0"; the rotation-free route never stores maps, so it stays out."""
    if not (rotating and io_bf16):
        return False
    if _env("SISO_FAST", "auto").strip().lower() in ("0", "off", "false"):
        return False
    if _portable.gluon_refusal("ssd_mimo") is not None:        # kernel_target="portable": the Triton SISO schedule
        return False
    from . import gluon_sm90_siso_bwd as _g
    return _g.mode() == "auto" and _g.default_serves(device)


#: True while a registered compile op (`registered_scan`) runs its Function body: the SISO schedule saves S_all and
#: its own aux set, which the recompute-only op pair does not carry, so the pair runs the GENERAL route.
_GENERAL_ONLY = [False]


def compile_general_only() -> bool:
    """`HYBRID_SSM_COMPILE_SISO`: "general" (default) = a compiled scan takes the general (recompute-only) route as a
    graph node; "boundary" = the shipped `torch.compiler.disable` boundary around the SISO schedule (one graph break
    per scan, SISO speed)."""
    return os.environ.get("HYBRID_SSM_COMPILE_SISO", "general") != "boundary"


@contextlib.contextmanager
def general_route():
    old, _GENERAL_ONLY[0] = _GENERAL_ONLY[0], compile_general_only()
    try:
        yield
    finally:
        _GENERAL_ONLY[0] = old


def siso_enabled(device=None, rotating: bool = False, io_bf16: bool = False,
                 T: int | None = None, real_cp: bool = False,
                 bh: int | None = None, L: int | None = None,
                 needs_bwd: bool = True, thh: bool = False,
                 io_fp32: bool = False) -> bool:
    """`SSD_MIMO_SISO_FAST`: "1" on, "0" off, unset / "auto" = on exactly
    where `gluon_default_route` applies (and at the T = 64 it needs), and
    for a real (rotation-free) call exactly when the calibrated chunk-parallel
    forward is selected (`real_cp`; envelope: `siso_refusal`)."""
    m = _env("SISO_FAST", "auto").strip().lower()
    if m in ("1", "on", "true"):
        return True
    if m in ("0", "off", "false"):
        return False
    if _GENERAL_ONLY[0]:
        return False
    if not rotating:
        # real (rotation-free) R = O = 1 takes the SISO route ONLY when the
        # device's calibrated decision selects the chunk-parallel forward
        # (`chunkpar_wanted`, decided by the caller: `real_cp`); otherwise
        # the general route.  Bare serial SISO is not a default.
        return bool(real_cp)
    # Rotating R = O = 1: the SISO schedule is the DEFAULT on every GPU.
    # sm_90 keeps its Gluon body inside the same route.
    # The calls it does NOT take (they fall back to the general fused route):
    # not rotating (the real case: `real_cp` above), io dtype != bf16,
    # chunk T != 64, SSD_MIMO_SISO_FAST=0, and everything `siso_refusal` names.
    if rotating and io_fp32 and not io_bf16:
        # fp32 IO: the SISO schedule serves an fp32 rotating call only WITHOUT a backward and only where this
        # GPU's CALIBRATED fp32 no-grad rule (rot_route, phase `nograd_fp32`, eligibility-gated) selects it, at the
        # rule's fp32 SISO chunk.  A TRAINING fp32 call never takes the SISO backward by default (its scan adjoint is
        # slower than the general route's backward kernels); SSD_MIMO_SISO_FAST=1 still forces it (DTH_ROT serves fp32).
        # No fp32 calibration -> general route.
        if needs_bwd or bh is None or L is None:
            return False
        from .rot_route import pick
        r = pick(bh, L, device if device is not None else torch.device("cuda"), needs_bwd, thh, dtype="fp32")
        return r is not None and r[0] == "siso" and r[1] == T
    if not (T == 64 and bool(rotating and io_bf16)):
        return False
    if bh is not None and L is not None:
        # the CALIBRATED per-GPU structural choice (rot_route.py): siso vs the general routes by
        # occupancy and length; no valid calibration -> siso
        from .rot_route import pick
        r = pick(bh, L, device if device is not None else torch.device("cuda"), needs_bwd, thh)
        if r is not None and r[0] != "siso" and r[1] == T:
            return False        # (a T the route did not choose -- an explicit chunk -- keeps the SISO schedule)
    return True


_FUSE_MODES = ("0", "b", "c")


def siso_fuse() -> str:
    """SISOFUSE -- the SISO SPEED at R = O = 1 without the SISO MEMORY.

    DEFAULT "0" (off).  The route's win at R = O = 1 has two separable
    parts:

      * the DEDICATED SCHEDULE (no r/o machinery, one CTA per (b, h),
        the whole sequence, `S[N, G]` in registers, an autotuned
        pipeline depth).  Rotation-free there are no stored maps at all;
        its only cost is that the route forces `SALL=io` (extra
        forward-HELD memory per layer).
      * the TWO-PHASE STAGING (phase 1 stores the per-head rotated maps,
        phase 2 scans rotation-free).  It stores two `[B, L, H, N]`
        IO-dtype maps, which dominate the forward-held memory.

    Both modes below keep the dedicated schedule and hand the BACKWARD
    to the general 5-kernel route, which re-rotates at tile load as it
    does today -- i.e. they extend the combination the rotation-FREE
    case already ships (SISO forward + general backward) to the rotating
    one.  `ctx.siso_bwd` is therefore False under both, and nothing the
    forward stages is retained across the fwd->bwd window.

      "0"   off (DEFAULT).  Rotating SISO = staged maps + the SISO
            backward, held.  Unchanged, bit for bit.
      "b"   TRANSIENT MAPS.  Phase 1 still stages `Bs`/`Cs` for phase 2,
            but they are dropped at the end of the forward instead of
            being saved, so the map memory becomes PEAK rather than HELD.
            Phase 1 also stores the trapezoid's RAW `k_t . q_t` next to
            the `g_t`-scaled one phase 2 reads back, because the general
            backward's `trap_prologue` takes the raw convention (the
            general forward's `Dot_ptr` store) and the staged schedule's
            is `raw * g_t`.
      "c"   IN-KERNEL ROTATION.  No phase 1 and no maps at all: the
            dedicated schedule rotates at TILE LOAD, exactly as
            `_fused_chunk_fwd_kernel` does, and the trapezoid returns to
            its Score-diagonal form (which already stores the raw dot).
            The G-TILING DISCIPLINE IS UNCHANGED -- `NGT == 1`, one G
            tile per head.  The envelope clause `G_SIZE <= 64, %16 == 0`
            was written for the stored maps ("phase 1 stores the
            per-head map exactly once"); with no maps that reason is
            gone, but a SECOND one that the general route does not have
            remains and is binding: this schedule carries `S[N, G]` in
            REGISTERS across the whole sequence with no state-passing
            kernel, so a G tile is not a loop the kernel may take -- a
            second G tile would either re-run the entire scan (and its
            `[T, T]` score) per tile or need the general route's
            cross-CTA state hand-off, which is the schedule this route
            exists to avoid.  The clause therefore stays, for the
            register-resident state rather than for the maps.
    """
    v = _env("SISO_FUSE", "0")
    return v if v in _FUSE_MODES else "0"


def siso_xtm() -> str:
    """The SISO route's own TOKEN-MAJOR `x` / `dx` layout.

    The family layout is `x [B, H, G, R, L]` with L contiguous, so the
    scan's `[T, G]` tile strides by L along its FAST axis, making the
    `dx` store slow.  Mamba-3's
    `V` / `dV` are `(batch, seqlen, nheads, headdim_v)` -- token-major,
    headdim contiguous -- and their `dV` store is a plain TMA block.

    This lever converts ONCE at the route boundary (`x` on the way in,
    `dx` on the way out) and hands the kernels a stride set with
    `s_xg == 1`.  Nothing inside either kernel changes: both index `x`
    through `offs_g * s_xg + offs_t * s_xl`, so the layout is entirely a
    property of the four strides passed in.  The forward therefore reads
    the SAME VALUES in the SAME ORDER -- a permute copy is exact, so the
    forward's numerics are bitwise unchanged.

      "0"     off (DEFAULT); the family layout.  It is the default
              because the coalesced store gains nothing that pays for
              the two transposes it needs.
      "1"     on; `dx` is copied back to `[B, H, G, R, L]`
      "view"  on, but `dx` is returned as a permuted VIEW of the
              token-major buffer -- the right shape, token-major
              strides, no copy.  The A/B for whether the copy back is
              worth what it saves downstream.
    """
    return _env("SISO_XTM", "0")


def siso_bc(t):
    """`[B, nc, T, N]` view of a B/C tensor that may carry a SIZE-1 R/O
    axis.  `_unified_ssd_impl` materialises the trailing axis whenever
    the module asks for a per-head bias, even at R = O = 1; it is a
    stride-preserving squeeze, not a copy, and the gradient is reshaped
    back to the caller's shape on the way out."""
    return t[..., 0] if t.dim() == 5 and t.shape[-1] == 1 else t


def siso_refusal(*, Ctil, Btil, x_raw, G_SIZE, N, bc_grouped, Gb, Gc,
                 theta_off, in_kernel_gate, T, theta=None, theta_h=None,
                 rot_bias=None):
    """None when the SISO schedule serves this call; else why not."""
    R = x_raw.shape[3]
    O = (Ctil.shape[-1] if Ctil.dim() == (6 if bc_grouped else 5) else 1)
    if R != 1 or O != 1:
        return "R=%d O=%d (the SISO schedule is R = O = 1)" % (R, O)
    if bc_grouped or Gb != 1 or Gc != 1:
        return "grouped B/C (Gb=%d Gc=%d)" % (Gb, Gc)
    if siso_bc(Btil).dim() != 4 or siso_bc(Ctil).dim() != 4:
        return "B/C carry an R/O axis"
    if G_SIZE > 64 or G_SIZE % 16:
        return "G_SIZE=%d (one G tile per head, 16 | G <= 64)" % G_SIZE
    if theta_off is not None:
        return "a host-built phase OFFSET; the SISO schedule takes "\
               "the cumulated phase"
    if in_kernel_gate:
        return "the in-kernel gate (SSD_MIMO_GATE_INKERNEL=1)"
    if theta is None and (theta_h is not None or rot_bias is not None):
        # A STATIC per-head rotation with NO per-token phase.  Phase 1
        # exists for the per-token phase and is what carries the static
        # angles with it; without one there is no map to store and the
        # scan would silently drop `theta_h` / `rot_bias`.  Serving it
        # would mean a third branch in phase 2 for two `[N]` vectors, and
        # `dyn` (the only law that uses those angles in production) always
        # carries a phase, so this corner keeps the general route.
        return "theta_h / rot_bias without a per-token phase"
    if N < 16 or (N & (N - 1)):
        return "N=%d is not a power of two >= 16" % N
    return None



def siso_backward(ctx, Ctil, Btil, x_raw, dw, delta_cumsum, a_log,
                  S_all, dY, dS_final, trap_lam, d_skip, dskip_dd,
                  needs=None):
    """The route-B backward, in `fused_chunk_bwd_5kernel`'s tuple order."""
    from .siso_bwd_kernel import siso_chunk_bwd
    g = siso_chunk_bwd(
        ctx.siso_aux, siso_bc(Ctil), siso_bc(Btil), x_raw, delta_cumsum,
        a_log, S_all, dY,
        theta=ctx.theta, theta_h=ctx.theta_h, rot_bias=ctx.rot_bias,
        beta=ctx.beta, gamma=ctx.gamma, bias_pre_rot=ctx.bias_pre_rot,
        dw=dw, trap_lam=trap_lam, x0=ctx.x0, d_skip=d_skip,
        dS_final=dS_final, precision=ctx.siso_prec, dskip_dd=dskip_dd,
        needs=needs)

    def _as(t, like):
        return None if (t is None or like is None) else t.reshape(like.shape)

    return (g["dX"],
            g["dC"].reshape(Ctil.shape),
            g["dB"].reshape(Btil.shape),
            g["dDelta"],
            _as(g["dThetaH"], ctx.theta_h),
            _as(g["dTheta"], ctx.theta),
            _as(g["dBeta"], ctx.beta),
            _as(g["dGamma"], ctx.gamma),
            g["dX0"],
            _as(g["dRotBias"], ctx.rot_bias),
            g["dDW"], None, None, g["dLam"],
            g["dDsk"])


def dth_rot_can_serve(*, io_dtype, bias_pre_rot, beta, gamma, theta, N):
    """Whether the Triton SISO backward's rotated-frame angle adjoint
    (`DTH_ROT`, `siso_bwd_kernel._siso_bwd_scan_kernel`'s `DTH_ROT`
    constexpr) can serve a call that needs `dL/dtheta`.

    DTH_ROT is a REQUIREMENT, not an
    opt-in choice, whenever the Triton SISO route (not the Hopper Gluon
    twin, `gluon_sm90_siso_bwd`, which keeps its own always-on default)
    serves a rotating call that needs the angle gradient: the pointwise
    bf16 read-back adjoint (DTH_ROT off) exceeds the fp64 normwise bound
    on DT/Angles.  This predicate is the single place that
    decides whether DTH_ROT can be switched on for a given call; `False`
    means the Triton SISO backward must refuse the angle adjoint (and the
    caller must not run it without DTH_ROT -- see `siso_bwd_kernel`).

    False cases, by name:
      * `theta is None`           -- no per-token phase, nothing to adjoint
      * a POST-rotation bias      -- `beta`/`gamma` without
                                      `bias_pre_rot`; DTH_ROT's rotated-
                                      frame contraction assumes the bias
                                      sits before the rotation
      * `theta.shape[-1] != N//2`  -- not the cumulated per-pair phase
                                      DTH_ROT's buffer layout assumes
    """
    if theta is None:
        return False
    if io_dtype not in ("bf16", "fp16", "fp32"):
        return False
    if not (bool(bias_pre_rot) or (beta is None and gamma is None)):
        return False
    return theta.shape[-1] == N // 2


def siso_precision_refusal(*, io_dtype, precision):
    """None when the SISO route honours the call's `precision`, else why, by name (the call then takes the general route).

    The SISO scan rounds EVERY MMA operand to the IO dtype (`siso_fwd_kernel._to_opd`, the Mamba-2 / Mamba-3 convention: "SISO already
    rounds operands to IO_DT; io/bf16 share"), so on 16-bit IO it runs bf16 operands whatever `precision` asks.  The general route
    honours `precision` ("ieee" = fp32 operands, "tf32" = tf32): theta_proj.weight against the fp64 reference differs
    markedly at ieee / tf32 on bf16 IO, while at precision "io" both routes agree.  A caller that asks for ieee / tf32 operands on
    16-bit IO therefore does not get them from SISO, and is routed to the route that honours the request.  Production calls run
    precision "io" and are unaffected."""
    if io_dtype in (torch.bfloat16, torch.float16) and precision in ("ieee", "tf32"):
        return ("precision %r on 16-bit IO: the SISO scan rounds every MMA operand to the IO dtype, so it cannot honour it; "
                "the general route serves the call" % precision)
    return None


def siso_extra_refusal(*, x_raw, precision, T, N, G_SIZE, pair_half, theta):
    """The refusals `_SSDMimoFn.forward` applies after `siso_refusal` (one function, so the contract suite's route prediction and the
    dispatcher cannot drift): None when none applies, else why."""
    why = siso_precision_refusal(io_dtype=x_raw.dtype, precision=precision)
    if why is not None:
        return why
    if (x_raw.element_size() == 4
            and precision == "ieee" and T >= 64
            and N * G_SIZE >= 128 * 64):
        # the SISO
        # scan adjoint with fp32 "ieee" operands at N=128, G=64, T=64
        # needs 196,608 B of shared memory at its smallest config; the
        # SISO backward has no chunk ladder (it reads the forward's
        # chunk-shaped maps), so a part below that is refused HERE, by
        # name, before the forward stores them.  Smaller (N, G) cells
        # are served; the backward refits its config if the tuned one
        # overflows (siso_bwd_kernel._bwd_refit_oor).
        from .smem import device_smem_limit
        if device_smem_limit(x_raw.device) < 196608:
            return ("fp32 'ieee' operands at T=%d, N=%d, G=%d: the "
                    "SISO scan adjoint needs 196,608 B of shared "
                    "memory, the device allows %d B" % (
                        T, N, G_SIZE,
                        device_smem_limit(x_raw.device)))
    if G_SIZE & (G_SIZE - 1):
        # the SISO kernels index the head-group axis with tl.arange(0, G_SIZE): a non-power-of-two G (48) fails to compile
        # ("arange's range must be a power of 2"); the general route pads the tile
        return "G=%d is not a power of two (the SISO kernels' tl.arange over the head group)" % G_SIZE
    if pair_half and theta is not None:
        # The SISO kernels rotate INTERLEAVED pairs only; the rotate-half
        # layout was never wired into them (siso_chunk_fwd takes no
        # pair flag), so serving it would rotate the wrong pairs
        # (the forward would be wrong against fp64).
        return ("the rotate-half pair layout (pair_layout='half'): "
                "the SISO kernels rotate interleaved pairs only")
    return None


def dth_rot_refusal(*, needs_angle_grad, device, io_dtype, bias_pre_rot,
                    beta, gamma, theta, N, T=64, G_SIZE=None):
    """None when the SISO route may serve a rotating call; else the reason,
    by name, so the call takes the GENERAL route (DTH_ROT is a
    requirement wherever the Triton scan computes the angle adjoint).

    The Hopper Gluon body keeps its own rotated-frame adjoint, so a bf16
    call on a device where it is enabled is exempt; every other call that
    needs dL/dtheta must be servable by DTH_ROT."""
    if not needs_angle_grad:
        return None
    tag = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(io_dtype, "fp32")
    if tag == "bf16" and _portable.gluon_refusal("ssd_mimo") is None:
        from . import gluon_sm90_siso_bwd as _g
        # the Gluon body is exempt only INSIDE its own static envelope (`gluon_sm90_siso_bwd.refusal`: T = 64, N in (64, 128),
        # G in (32, 64)); outside it the backward falls to the Triton scan adjoint, which then needs DTH_ROT -- and a call it cannot
        # serve must be refused HERE, at forward time, so it takes the general route (never a backward-time raise)
        if _g.enabled(device) and T == 64 and N in (64, 128) and (G_SIZE is None or G_SIZE in (32, 64)):
            return None
    if dth_rot_can_serve(io_dtype=tag, bias_pre_rot=bias_pre_rot, beta=beta,
                         gamma=gamma, theta=theta, N=N):
        return None
    return ("the angle adjoint needs the rotated-frame DTH_ROT, which cannot "
            "serve this call (io=%s, post-rotation bias=%s, phase last dim "
            "%d vs N/2=%d) and the Gluon body does not serve it"
            % (tag, not (bool(bias_pre_rot) or (beta is None and gamma is None)),
               theta.shape[-1], N // 2))


def siso_map_head_major() -> bool:
    """The STORED maps' memory order.

    By default `Bs` / `Cs` are allocated as `[B, nc, T, H, N]`, so adjacent HEADS
    are adjacent in memory: the rotation adjoint's head loop then walks
    a contiguous `[T, H, N]` block per (b, c).  The price is paid by the
    SCAN adjoint, whose `[T, N]` tile is then many separate segments
    instead of one contiguous block -- and that kernel reads BOTH maps
    and writes both gradients over them.

    Mamba-3's `Q` / `K` are `(batch, seqlen, nheads_qk, headdim)` with
    `nheads_qk == 1`, so their `dqkv` tile IS one contiguous block.
    `SSD_MIMO_SISO_MAP_HEAD_MAJOR=1` allocates `[B, H, nc, T, N]` and
    gives ours the same property.  Nothing in any kernel changes: all
    three address the maps through `(s_mb, s_mc, s_mt, s_mh, s_mn)`.
    """
    return _env("SISO_MAP_HEAD_MAJOR", "0") == "1"


def siso_ytm() -> bool:
    """Ask the SISO forward for the TOKEN-MAJOR `Y` store.

    `siso_native_y` offers two memory orders; the
    DEFAULT (`[B, H, 1, G, L]`, L contiguous) makes the backward's `dY`
    tile stride by L along its fast axis, exactly as the family's `x`
    does.  For `x` that costs nothing (a STORE is fire-and-forget).  For
    `dY` it is a LOAD, on the critical path, so the `dY` layout alone
    speeds the scan adjoint at the kernel boundary.  It is free: the same
    helper allocates either order, the values are identical, and every
    consumer reads `Y` and `dY` through their own strides.

    **AT MODULE LEVEL IT IS A NO-OP, AND THAT IS WHY IT IS DEFAULT OFF.**
    On every route with a gate -- the default, the DEDICATED gate
    epilogue -- the `dY` the scan adjoint receives is NOT the SISO
    forward's own `Y` buffer at all: it is `ascalar_epilogue_bwd`'s
    `dYpre`, a plain contiguous `[B, H, nc, T, O, G]` fp32 tensor whose
    G axis already has stride 1.  The kernel-boundary table above is
    therefore the statement for a caller that hands the gradient back in
    the forward's native order, and the lever exists for that caller.
    """
    return _env("SISO_YTM", "0") == "1"
