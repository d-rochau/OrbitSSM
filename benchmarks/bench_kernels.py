"""Kernel-level rows: the two families against the reference kernels.

The timing peer of the kernel-level correctness tests.  Two family sections, both calling the
FAMILY ENTRY POINT and nothing below it:

* `mh_scan` (`orbit_ssm.kernels.hybrid_mamba_mh.functions.mh_scan`) vs
  `mamba_ssm`'s `selective_scan_fn`.
* `ssd_mimo_scan` (`orbit_ssm.kernels.ssd_mimo.autograd.ssd_mimo_scan`)
  vs `mamba_chunk_scan_combined`, plus Mamba-3 SISO and MIMO where
  importable.

Calling the entry point is not a style preference.  `ssd_mimo_scan` sets
`_require_fused=True`, so a config outside the fused envelope raises
instead of silently falling back to a slower path: a benchmark that takes
a silent fallback measures a kernel nobody ships.

**Wall time vs GPU time.**  Every cell's `ms_median`
times the WHOLE scan autograd Function per call — for an autograd
Function wrapping a fused whole-layer op (official Mamba-2's
`mamba_split_conv1d_scan_combined`) that includes host-side torch work
around the kernel launches, invisible to a self-CUDA reading.  The
kernel parity grids (`ssd_parity26`, `ssd_parity_m3`, `ssd_parity_m3_eq`)
therefore also record `ms_cuda_self` — the summed self-CUDA time of the
same call, one extra `torch.profiler` pass per cell, via the SAME
`cuda_buckets` helper `bench_ssms.py`'s `--profile` uses (now shared
from `harness.py`) — for BOTH arms.  `ms_median` is the honest
per-call number a caller pays; `ms_cuda_self` isolates the GPU work.
A table of the parity grids prints both side by side with
a ratio for each.

Grids (`--list-grids`); every one is defined in `shapes.py` with its
source of truth:

    smoke             the plumbing gate: both families, every available
                      baseline, < 5 min on the shared login GH200
    mh_siso           mh_scan P=1 vs selective_scan_fn — NEVER RUN before;
                      the cheapest measurement that unblocks the App C.3
                      headline claim, whose current anchor is a ratio
                      against the retired structure A
    mh_features       the paper's per-feature cost ledger
    paper_c3          Table 10's three constant-work shapes
    paper_c3_nscale   Table 11's N sweep
    paper_c3_lscale   Figure 4's L sweep
    ssd_parity26      the signed contract's 26-cell Mamba-2 parity grid
    ssd_parity_m3     the same 26 shapes, rotation mode, ours vs
                      mamba3_siso_combined (kernel vs kernel), each in
                      its OWN native rotation configuration
    ssd_parity_m3_eq  the same 26 shapes with both kernels computing the
                      SAME function on the SAME inputs (K9): the proven
                      parameter map, an equality gate before timing, and
                      both angle-count / trapezoid conventions
    ssd_long_l        the long-L envelope the segmented forward closed
    ssd_mimo_gates    what (R, O) and G cost, incl. the R4O4 gate cell
    ssd_features      each SSD-MIMO feature against its own absence

Usage:
    python -m benchmarks.bench_kernels --grid smoke --out /tmp/x.jsonl
    python -m benchmarks.bench_kernels --grid ssd_parity26 --budget full
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable

import torch

from . import baselines as bl
from . import shapes as sh
from .harness import (
    IO_DTYPE, KERNEL_PRECISION, Cell, cuda_buckets, group_key, main_for,
    make_pass, resolve_budget,
)

GRIDS = ("smoke", "mh_siso", "mh_features", "paper_c3", "paper_c3_nscale",
         "paper_c3_lscale", "ssd_parity26", "ssd_parity_m3",
         "ssd_parity_m3_eq", "ssd_long_l",
         "ssd_mimo_gates", "ssd_features", "kernel_shared_set")

MH_ENTRY = "orbit_ssm.kernels.hybrid_mamba_mh.functions.mh_scan"
SSD_ENTRY = "orbit_ssm.kernels.ssd_mimo.autograd.ssd_mimo_scan"


# ═════════════════════════════════════════════════════════════════════
# mh_scan
# ═════════════════════════════════════════════════════════════════════
def _mh_inputs(B, H, P, N, L, *, dtype, device="cuda", seed=0,
               rot=False, gates=False, biases=False, delta_w=None,
               law="A", G=None, theta_delta=False, theta_h=False):
    """Self-contained mh_scan input generator.

    Deliberately self-contained: the in-package bench/diag tree this
    generator was once shared with lived inside the production package
    and was deleted, so `benchmarks/` must not depend on
    anything of its kind.  The ABI reproduced here is
    `layout.validate_forward_inputs` (x (B,H,P,L), delta (B,H,L), A_log
    (H,P,N), B_t/C_t (B,N,L), theta (B,K,L), beta_* (G,N),
    delta_bias (H,)) — that function is the contract, and it will refuse
    us loudly if this drifts.
    """
    from orbit_ssm.kernels.hybrid_mamba_mh.layout import (
        DYN_SOFTPLUS, GATE_SIGMOID, LAW_A_DYN, LAW_MU,
        fracs_to_rows, norm_fracs,
    )

    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, scale=1.0, f32=False, grad=False):
        t = (torch.randn(*shape, generator=g) * scale).to(
            device=device, dtype=torch.float32 if f32 else dtype).contiguous()
        return t.requires_grad_(grad)

    K = N // 2
    # `law` is the ledger label (law:A / law:dyn / law:mu, kept so stored rows stay comparable)
    decay = {"A": "coord_static", "dyn": "coord_dynamic", "mu": "coord_potential"}[law]
    fracs = [{"frac": 1.0, "decay": decay, "rot": rot}]
    frs = norm_fracs(fracs, N)
    row_law, row_rot = fracs_to_rows(frs, N, device=device)
    laws = {f.law for f in frs}

    x = r(B, H, P, L, grad=True)
    delta = r(B, H, L, scale=0.5, grad=True)
    B_t = r(B, N, L, grad=True)
    C_t = r(B, N, L, grad=True)

    # A_log is pair-tied wherever a row rotates
    A_log = r(H, P, N, scale=0.5, f32=True)
    if bool(row_rot.any()):
        A_log = A_log.reshape(H, P, K, 2)[..., :1].expand(H, P, K, 2)
        A_log = A_log.reshape(H, P, N).contiguous()
    A_log = A_log.detach().requires_grad_(True)

    kw: dict[str, Any] = dict(row_law=row_law, row_rot=row_rot,
                              dyn_act=DYN_SOFTPLUS, dyn_eps=1e-4,
                              gate_act=GATE_SIGMOID)
    Gg = H if G in (None, "H") else int(G)
    if LAW_A_DYN in laws:
        kw["a_dyn"] = r(B, N, L, scale=0.5, f32=True, grad=True)
    if LAW_MU in laws:
        kw["mu"] = r(B, N, L, scale=0.3, f32=True, grad=True)
        kw["mu_gain"] = r(H, P, N, scale=0.3, f32=True, grad=True)
    if rot:
        kw["theta"] = r(B, K, L, scale=0.4, f32=True, grad=True)
        kw["theta_delta"] = theta_delta
        if theta_h:
            kw["theta_h"] = r(H, K, scale=0.2, f32=True, grad=True)
    elif theta_delta or theta_h:
        raise ValueError("theta_delta/theta_h require rot=True")
    if gates:
        kw["g_B"] = r(B, N, L, scale=0.7, grad=True)
        kw["g_C"] = r(B, N, L, scale=0.7, grad=True)
    if biases:
        kw["beta_a"] = r(Gg, N, scale=0.2, f32=True, grad=True)
        kw["beta_b"] = r(Gg, N, scale=0.3, f32=True, grad=True)
        kw["beta_c"] = r(Gg, N, scale=0.3, f32=True, grad=True)
        kw["delta_bias"] = r(H, scale=0.2, f32=True, grad=True)
        if gates:
            kw["beta_gB"] = r(Gg, N, scale=0.3, f32=True, grad=True)
            kw["beta_gC"] = r(Gg, N, scale=0.3, f32=True, grad=True)
    if delta_w == "head":
        kw["delta_w"] = r(B, H, L, scale=0.5, grad=True)
    elif delta_w == "chan":
        kw["delta_w"] = r(B, H, P, L, scale=0.5, grad=True)

    leaves = [x, delta, A_log, B_t, C_t] + [
        v for v in kw.values() if isinstance(v, torch.Tensor) and v.requires_grad]
    return (x, delta, A_log, B_t, C_t), kw, leaves


def build_mh(B, H, P, N, L, *, dtype, endpoint=False, backend="cuda", **feat):
    """`mh_scan` at (B, H, P, N, L)."""
    from orbit_ssm.kernels.hybrid_mamba_mh import mh_scan

    from functools import partial
    from orbit_ssm.kernels.hybrid_mamba_mh.torch_scan import mh_torch_scan
    entries = {"cuda": mh_scan, "torch": mh_torch_scan,
               "sequential": partial(mh_torch_scan, sequential=True)}
    if backend not in entries:
        raise ValueError(f"unknown MH backend {backend!r}")
    entry = entries[backend]
    args, kw, leaves = _mh_inputs(B, H, P, N, L, dtype=dtype, **feat)
    if endpoint:
        lam = torch.full((B, H, L), .5, device=args[0].device,
                         dtype=torch.float32, requires_grad=True)
        kw["trap_lam"] = lam
        leaves.append(lam)

    def call():
        return entry(*args, **kw)

    return call, leaves


def _mh_cell(grid, arm, B, H, P, N, L, precision, pass_, *, work, tags=None,
             **feat):
    """One mh_scan row.  `work` is the group's constant-work key — see
    `harness.group_key`; it is REQUIRED so no arm can silently end up
    alone in its own group."""
    dtype = IO_DTYPE[precision]
    return Cell(
        family="mh_scan", arm=arm, entry_point=MH_ENTRY,
        shape=dict(B=B, H=H, P=P, N=N, L=L, D=H * P),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        tags=dict(tags or {}),
        build=lambda: make_pass(*build_mh(B, H, P, N, L, dtype=dtype, **feat),
                                pass_),
    )


def _m1_cell(grid, B, D, N, L, precision, pass_, *, work):
    dtype = IO_DTYPE[precision]
    return Cell(
        family="mh_scan", arm="m1_selective_scan",
        entry_point=bl.M1_SELECTIVE_SCAN.entry_point,
        shape=dict(B=B, H=D, P=1, N=N, L=L, D=D),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        build=lambda: make_pass(*bl.build_m1(B, D, N, L, dtype=dtype), pass_),
    )


# ═════════════════════════════════════════════════════════════════════
# ssd_mimo_scan
# ═════════════════════════════════════════════════════════════════════
_SSD_FEATURE_KW = {
    "none", "theta", "theta_h", "rot_bias", "beta", "gamma", "gate_z",
    "delta_w", "x0", "bc_per_head", "rotation",
}


def build_rank_mixer(rot, R, d_inner, B, L, *, N=128, T=None, T_bwd=None, fp32=False, seed=0, device="cuda"):
    """The PRODUCTION mixer of the R = O > 1 calibration cells: the LM contract's `ssd_mimo_r<R>o<R>_{real,complex}` arm
    (the language-model preset table; d_model = d_inner / 2, n_heads = d_inner / 64, headdim = 64, N = 128, B/C norms + bias, gates, trapezoid, rmix,
    D skip) -- the call whose kernel variants production launches.  The plain `build_ssd` call compiles other constexpr variants (no gates /
    d-skip / norms): the real R2O2 launch group fitted on it ran the H100 mixer at 42.9 ms against 12.8 ms uncalibrated.
    `T` / `T_bwd` pin the chunks (None = the rule).  Returns (call, leaves)."""
    try:
        from experiments.lm.model.lm import NAMED_SSM
    except ImportError:
        raise NotImplementedError(
            "build_rank_mixer builds the language-model presets of the R = O > 1 mixers, which are not part of this "
            "release; benchmark those ranks through `build_ssd` (raw kernel) instead") from None
    from orbit_ssm.model import CausalLM
    spec = NAMED_SSM["ssd_mimo_r%do%d_%s" % (R, R, "complex" if rot else "real")]
    from .cases import split_execution
    kw = dict(spec["kwargs"], n_heads=d_inner // 64)
    ex = split_execution("ssd_mimo", kw)
    if T is not None:
        ex["chunk_size"] = int(T)
    if T_bwd is not None and T_bwd != T:
        ex["kernel_options"] = dict(ex.get("kernel_options", {}), chunk_size_bwd=int(T_bwd))
    if fp32:
        ex["precision"] = "tf32"
    torch.manual_seed(seed)
    mod = CausalLM._build_mixer(d_inner // 2, dict(type="ssd_mimo", expand=2.0, d_conv=4, d_state=N, **kw), execution=ex).to(device=device)
    x = torch.randn(B, L, d_inner // 2, device=device, requires_grad=True)

    def call():
        if fp32:
            return mod(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return mod(x)
    return call, [x, *mod.parameters()]


def _ssd_auto_chunk(B, H, G, R, O, L, dtype, feature, nb, device):
    """chunk="auto": the production chunk rule at kernel level, the dispatch the mixer uses
    (`SSDMimoSSM._resolve_chunk_size_impl`): the calibrated structural rot route's T for a rotating R = O = 1 call, else
    `_scan_chunk` (real=True for the real, non-rotating decay: the per-GPU calibrated structural rule).  Returns (T, T_bwd or None)."""
    import os as _os
    from orbit_ssm.kernels.ssd_mimo import rot_route as _rr
    from orbit_ssm.ssm.ssd_mimo import _scan_chunk, _scan_chunk_bwd
    if _os.environ.get("SSD_MIMO_CHUNK"):
        return int(_os.environ["SSD_MIMO_CHUNK"]), None
    rotating = feature in ("theta", "theta_h", "rot_bias", "rotation")
    if rotating and R == 1 and O == 1:
        if _rr.explicit_siso() or _rr.explicit_rot_bwd() or _rr.explicit_chunk():
            return 64, None
        r = _rr.pick(B * H, L, device, nb, feature in ("theta_h", "rotation"),
                     **({"dtype": "fp32"} if dtype == torch.float32 else {}))
        if r is not None:
            return r[1], None
        return (64, None) if dtype != torch.float32 else (_scan_chunk(1, G, 1, nb), None)
    kw = dict(real=not rotating, L=L, work=B * H * L, dtype=dtype)
    T = _scan_chunk(R, G, O, nb, **kw)
    return T, (_scan_chunk_bwd(R, G, O, T, **kw) if nb else None)


def build_ssd(B, H, G, R, O, N, L, *, chunk="auto", dtype, precision="tf32",
              feature="none", device="cuda", seed=0, backend="cuda", chunk_bwd=None):
    """`ssd_mimo_scan` at (B, H, G, R, O, N, L).  `chunk_bwd` (int, with an int `chunk`): the backward's own chunk (`chunk_size_bwd`).  `chunk="auto"` (default) is the production chunk rule
    (`_ssd_auto_chunk`); an int pins it.

    `feature` turns exactly ONE optional input on, so the ledger prices
    each against its own absence, EXCEPT `"rotation"`, which is the
    COMPOSITE "rotation mode" (theta + theta_h + rot_bias together) used
    to price the kernel against `mamba3_siso_combined`'s own rotation path
    (ssd_parity_m3) -- Mamba-3 always carries a per-timestep phase, a
    per-head offset and per-head biases at once, so the fair comparison
    turns all three of ours on together rather than one at a time.  Shapes
    are the ones the entry point documents (autograd.py:893-940).
    """
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan

    from orbit_ssm.kernels.ssd_mimo.reference import unified_ssd_chunked, unified_ssd_quadratic
    entries = {"cuda": ssd_mimo_scan, "torch": unified_ssd_chunked, "quadratic": unified_ssd_quadratic}
    if backend not in entries:
        raise ValueError(f"SSD kernel backend must be one of {tuple(entries)}; sequential is an SSM route")
    entry = entries[backend]
    if feature not in _SSD_FEATURE_KW:
        raise ValueError(f"unknown ssd feature {feature!r}")
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, scale=1.0, f32=False, grad=False):
        t = (torch.randn(*shape, generator=g) * scale).to(
            device=device, dtype=torch.float32 if f32 else dtype).contiguous()
        return t.requires_grad_(grad)

    bc_per_head = feature == "bc_per_head"
    x = r(B, H, G, R, L, grad=True)
    delta = (torch.rand(B, H, L, generator=g).to(device) * 0.1 + 0.01
             ).requires_grad_(True)
    a_log = (torch.randn(H, generator=g).to(device) - 1.0
             ).requires_grad_(True)

    b_shape = (B, L, H, N) if bc_per_head else (B, L, N)
    c_shape = (B, L, H, N) if bc_per_head else (B, L, N)
    if R > 1:
        b_shape = b_shape + (R,)
    if O > 1:
        c_shape = c_shape + (O,)
    B_tok = r(*b_shape, grad=True)
    C_tok = r(*c_shape, grad=True)

    kw: dict[str, Any] = dict(chunk_size=chunk, precision=precision,
                              bc_per_head=bc_per_head)
    if feature in ("theta", "theta_h", "rot_bias", "rotation"):
        # `theta` here is head-shared 3-D (`[B, L, N//2]`); `bias_pre_rot`
        # (True by default) needs the in-kernel PER-HEAD (4-D)
        # phase, so pin the legal post-rotation order explicitly. Every
        # other `feature` value carries no rotation kwarg at all, so the
        # (now no-op) default rides unpinned.
        kw["bias_pre_rot"] = False
        kw["theta"] = r(B, L, N // 2, scale=0.4, f32=True, grad=True)
    if feature in ("theta_h", "rotation"):
        kw["theta_h"] = r(H, N // 2, scale=0.2, f32=True, grad=True)
    if feature in ("rot_bias", "rotation"):
        kw["rot_bias"] = r(H, N // 2, scale=0.2, f32=True, grad=True)
    if feature == "beta":
        kw["beta"] = r(H, N, R, scale=0.2, f32=True, grad=True)
    if feature == "gamma":
        kw["gamma"] = r(H, N, O, scale=0.2, f32=True, grad=True)
    if feature == "gate_z":
        kw["gate_z"] = r(B, L, H * G, scale=0.5, grad=True)
    if feature == "delta_w":
        kw["delta_w"] = r(B, H, L, scale=0.5, f32=True, grad=True)
    if feature == "x0":
        kw["x0"] = r(B, H, G, N, scale=0.2, f32=True, grad=True)

    args = (x, delta, B_tok, C_tok, a_log)
    leaves = [t for t in (*args, *kw.values())
              if isinstance(t, torch.Tensor) and t.requires_grad]
    if chunk_bwd is not None and chunk != "auto" and backend == "cuda":
        kw["chunk_size_bwd"] = int(chunk_bwd)
    if chunk == "auto":
        if backend == "cuda":
            kw["chunk_size"], tb = _ssd_auto_chunk(B, H, G, R, O, L, dtype, feature, torch.is_grad_enabled(), device)
            if tb is not None:
                kw["chunk_size_bwd"] = tb
        else:
            kw["chunk_size"] = 64

    if backend != "cuda":
        kw.pop("precision")
    if backend == "quadratic":
        kw.pop("chunk_size")

    def call():
        return entry(*args, **kw)

    return call, leaves


def _ssd_cell(grid, B, H, G, R, O, N, L, precision, pass_, *, work,
              chunk=64, feature="none", arm="ours", tags=None):
    dtype = IO_DTYPE[precision]
    kprec = KERNEL_PRECISION[precision]
    # production chunk rule by default (the mixer's dispatch); PARITY_PIN_CHUNK=1 pins the grid's `chunk`
    pinned = os.environ.get("PARITY_PIN_CHUNK") == "1"
    tags = dict(tags or {}, chunk_mode="pinned" if pinned else "auto",
                chunk_ours=("auto" if not pinned else chunk), chunk_official_m2=(M2_DEFAULT_CHUNK if not pinned else chunk))
    if not pinned:
        chunk = "auto"
    return Cell(
        family="ssd_mimo", arm=arm, entry_point=SSD_ENTRY,
        shape=dict(B=B, H=H, G=G, R=R, O=O, N=N, L=L, chunk=chunk,
                   d_inner=H * G * R),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        tags=dict(tags or {}, feature=feature),
        build=lambda: make_pass(
            *build_ssd(B, H, G, R, O, N, L, chunk=chunk, dtype=dtype,
                       precision=kprec, feature=feature), pass_),
    )


# official Mamba-2's own module default (`mamba_ssm.modules.mamba2.Mamba2(chunk_size=256)`); official Mamba-3's is 64
M2_DEFAULT_CHUNK = 256
M3_DEFAULT_CHUNK = 64


def _m2_cell(grid, B, H, P, N, L, precision, pass_, *, work, chunk=64,
             tags=None):
    """Official Mamba-2 at ITS OWN default chunk (256), not the grid's T (every arm runs its own production config);
    `PARITY_PIN_CHUNK=1` is the labelled variant that gives it the grid's `chunk`."""
    dtype = IO_DTYPE[precision]
    pinned = os.environ.get("PARITY_PIN_CHUNK") == "1"
    if not pinned:
        chunk = M2_DEFAULT_CHUNK
    tags = dict(tags or {}, chunk_mode="pinned" if pinned else "auto", chunk_ours="pinned" if pinned else "auto",
                chunk_official_m2=chunk)
    return Cell(
        family="ssd_mimo", arm="m2_chunk_scan",
        entry_point=bl.M2_CHUNK_SCAN.entry_point,
        shape=dict(B=B, H=H, G=P, R=1, O=1, N=N, L=L, chunk=chunk,
                   d_inner=H * P),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_), tags=tags,
        build=lambda: make_pass(
            *bl.build_m2(B, H, P, N, L, chunk=chunk, dtype=dtype), pass_),
    )


def _m3_siso_cell(grid, B, H, P, N, L, precision, pass_, *, work, chunk=64,
                  tags=None):
    dtype = IO_DTYPE[precision]
    pinned = os.environ.get("PARITY_PIN_CHUNK") == "1"
    if not pinned:
        chunk = M3_DEFAULT_CHUNK          # official Mamba-3's own default (SISO)
    tags = dict(tags or {}, chunk_mode="pinned" if pinned else "auto", chunk_official_m3=chunk)
    return Cell(
        family="ssd_mimo", arm="m3_siso", entry_point=bl.M3_SISO.entry_point,
        shape=dict(B=B, H=H, G=P, R=1, O=1, N=N, L=L, chunk=chunk,
                   d_inner=H * P),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        tags=tags,
        build=lambda: make_pass(
            *bl.build_m3_siso(B, H, P, N, L, chunk=chunk, dtype=dtype), pass_),
    )


def _m3_mimo_cell(grid, B, H, P, N, L, precision, pass_, *, work, R=4,
                  chunk=16):
    dtype = IO_DTYPE[precision]
    return Cell(
        family="ssd_mimo", arm="m3_mimo", entry_point=bl.M3_MIMO.entry_point,
        shape=dict(B=B, H=H, G=P, R=R, O=R, N=N, L=L, chunk=chunk,
                   d_inner=H * P),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        tags=dict(engine="tilelang"),
        build=lambda: make_pass(
            *bl.build_m3_mimo(B, H, P, N, L, R=R, chunk=chunk, dtype=dtype),
            pass_),
    )


# ═════════════════════════════════════════════════════════════════════
# grids
# ═════════════════════════════════════════════════════════════════════
def grid_smoke() -> list[Cell]:
    """Both families, every available baseline, tiny.

    A plumbing check, not a measurement: one shape per family, two
    precisions, both passes.  Never quote a smoke number — the shapes are
    far below the launch-latency floor.
    """
    g = "smoke"
    cells: list[Cell] = []
    m = sh.SMOKE_MH
    for prec in ("tf32", "bf16"):
        for p in ("fwd", "fwd_bwd"):
            # One group per family per (precision, pass): the smoke grid is
            # a plumbing check, so the point of the grouping here is to
            # exercise the round-robin over every arm, not to produce a
            # ratio (never quote a smoke number).
            mhw, ssdw = "mh", "ssd"
            # mh SISO corner (H = D, P = 1) and the reference it is measured against
            cells.append(_mh_cell(g, "ours_p1", m["B"], m["D"], 1, m["N"],
                                  m["L"], prec, p, work=mhw))
            cells.append(_m1_cell(g, m["B"], m["D"], m["N"], m["L"], prec, p,
                                  work=mhw))
            # mh at a real head width
            w = sh.SMOKE_MH_MIMO
            cells.append(_mh_cell(g, f"ours_p{w['P']}", w["B"], w["H"], w["P"],
                                  w["N"], w["L"], prec, p, work=mhw))
            # SSD-MIMO R=O=1 vs Mamba-2 and Mamba-3 SISO
            s = sh.SMOKE_SSD
            cells.append(_ssd_cell(g, s["B"], s["H"], s["G"], s["R"], s["O"],
                                   s["N"], s["L"], prec, p, chunk=s["T"],
                                   work=ssdw))
            cells.append(_m2_cell(g, s["B"], s["H"], s["G"], s["N"], s["L"],
                                  prec, p, chunk=s["T"], work=ssdw))
            cells.append(_m3_siso_cell(g, s["B"], s["H"], s["G"], s["N"],
                                       s["L"], prec, p, chunk=s["T"],
                                       work=ssdw))
            # SSD-MIMO with R, O > 1 (the fused MIMO route)
            q = sh.SMOKE_SSD_MIMO
            cells.append(_ssd_cell(g, q["B"], q["H"], q["G"], q["R"], q["O"],
                                   q["N"], q["L"], prec, p, chunk=q["T"],
                                   work=ssdw))
    # TileLang M3-MIMO: forward only in the smoke grid — its first call
    # compiles for ~30 s and its ~150 ms/call host overhead puts it in a
    # different measurement regime anyway.
    s = sh.SMOKE_SSD
    cells.append(_m3_mimo_cell(g, s["B"], s["H"], 64, 128, s["L"], "bf16",
                               "fwd", R=4, chunk=16, work="ssd"))
    return cells


def grid_mh_siso() -> list[Cell]:
    """The row that has never been run: mh_scan P=1 vs selective_scan_fn."""
    g = "mh_siso"
    cells: list[Cell] = []
    for s in sh.MH_SISO_SHAPES:
        w = f"B{s['B']}D{s['D']}N{s['N']}L{s['L']}"
        for prec in ("ieee", "tf32", "bf16"):
            for p in ("fwd", "fwd_bwd"):
                cells.append(_mh_cell(g, "ours_p1", s["B"], s["D"], 1,
                                      s["N"], s["L"], prec, p, work=w))
                cells.append(_m1_cell(g, s["B"], s["D"], s["N"], s["L"],
                                      prec, p, work=w))
    return cells


def grid_mh_features() -> list[Cell]:
    """The per-feature cost ledger, at the four mh contract shapes.

    `sh.MH_CONTRACT_SHAPES` runs the ledger at P = 32, 32, 32, 64 (H = 64,
    16, 32, 32 respectively) -- every row here predates the dedicated P = 1 backward
    schedule (Mamba-1's own per-channel dt, H = D).  The P = 1 set adds
    a P = 1 arm NEXT TO each shape's existing P for both the `feat:*` and
    `law:*` rows, so the ledger is priced on K14's schedule too, not just
    the P = 32/64 one it was built against.  `D = H*P` is the CONSTANT
    WORK invariant across P (same total (B, D, N, L) elementwise work
    however it splits between heads and head width), so both P schedules
    share ONE group per shape -- the `work` key is built from `D` rather
    than the literal (H, P) pair the original single-P version of this
    grid used (a purely internal round-robin label, per
    `harness.group_key`'s own docstring: "never appear[s]" in a record,
    so reshaping it costs nothing downstream).  Base-P arm names are
    UNCHANGED (`feat:real`, `law:A`, ...); the P = 1 twin gets a `_p1`
    suffix (`feat:real_p1`, `law:A_p1`, ...), the same "_p<N>" convention
    `grid_smoke`'s `ours_p{P}` already uses elsewhere in this module.
    """
    g = "mh_features"
    cells: list[Cell] = []
    for s in sh.MH_CONTRACT_SHAPES:
        D = s["H"] * s["P"]
        # Every feature (and every decay law), at BOTH P schedules, at
        # ONE shape is one group: the ledger's whole content is
        # `feature / featureless` (and now `/ P=1`) at the same shape, so
        # those rows must share a clock state.
        w = f"B{s['B']}D{D}N{s['N']}L{s['L']}"
        for hh, pp, suffix in ((s["H"], s["P"], ""), (D, 1, "_p1")):
            for name, feat in sh.MH_FEATURES:
                for prec in ("tf32", "bf16"):
                    for p in ("fwd", "fwd_bwd"):
                        kw = dict(feat)
                        if kw.get("G") == "H":
                            kw["G"] = "H"
                        cells.append(_mh_cell(
                            g, f"feat:{name}{suffix}", s["B"], hh, pp,
                            s["N"], s["L"], prec, p, work=w,
                            tags=dict(feature=name, P=pp), **kw))
            for law in sh.MH_LAWS:
                for p in ("fwd", "fwd_bwd"):
                    cells.append(_mh_cell(
                        g, f"law:{law}{suffix}", s["B"], hh, pp, s["N"],
                        s["L"], "tf32", p, work=w, tags=dict(law=law, P=pp),
                        law=law))
    return cells


def grid_paper_c3() -> list[Cell]:
    """Table 10: three constant-work shapes, head widths 1..16."""
    g = "paper_c3"
    cells: list[Cell] = []
    n = sh.PAPER_C3_N
    for s in sh.PAPER_C3_SHAPES:
        B, D, L = s["B"], s["D"], s["L"]
        # The constant-work shape IS the group: the reference row and all
        # ten head-width arms differ only in how D is split into (H, P),
        # and the table is nothing but their ratios.
        w = f"B{B}D{D}L{L}N{n}"
        for prec in ("ieee", "bf16"):
            for p in ("fwd", "fwd_bwd"):
                cells.append(_m1_cell(g, B, D, n, L, prec, p, work=w))
                for P in sh.PAPER_C3_HEAD_WIDTHS:
                    if D % P:
                        continue
                    # ungated ("ours", the like-for-like implementation row)
                    cells.append(_mh_cell(g, f"ours_p{P}", B, D // P, P, n, L,
                                          prec, p, work=w,
                                          tags=dict(gates=False)))
                    # gated (the production LLM configuration)
                    cells.append(_mh_cell(g, f"ours_gated_p{P}", B, D // P, P,
                                          n, L, prec, p, work=w,
                                          tags=dict(gates=True), gates=True))
    return cells


def grid_paper_c3_nscale() -> list[Cell]:
    """Table 11: state-size scaling, gates off."""
    g = "paper_c3_nscale"
    cfg = sh.PAPER_C3_NSWEEP
    cells: list[Cell] = []
    for n in cfg["N"]:
        # B, D and L are fixed across the sweep, so N is the whole
        # constant-work key; the reference row and P in {1, 4, 16}
        # interleave at each N.
        w = f"B{cfg['B']}D{cfg['D']}L{cfg['L']}N{n}"
        for prec in ("ieee", "bf16"):
            for p in ("fwd", "fwd_bwd"):
                cells.append(_m1_cell(g, cfg["B"], cfg["D"], n, cfg["L"],
                                      prec, p, work=w))
                for P in cfg["head_widths"]:
                    cells.append(_mh_cell(g, f"ours_p{P}", cfg["B"],
                                          cfg["D"] // P, P, n, cfg["L"],
                                          prec, p, work=w))
    return cells


def grid_paper_c3_lscale() -> list[Cell]:
    """Figure 4: sequence-length scaling, forward+backward.

    Runs the CAPTION's shape (B=1, D=2048).  The generator that produced
    the shipped PDF used D=512; see `shapes.PAPER_C3_LSWEEP` for the open
    discrepancy, and run `--grid paper_c3_lscale` with
    `BENCH_LSCALE_D=512` to price the other one.

    This grid used to hardcode
    `precision="ieee"` for every cell -- the figure predates a bf16
    column entirely, which, with ieee dropped from every
    benchmark submission and from the paper's quoted columns,
    left it with NO paper set at all, the one grid `--cells paper`
    refused outright.  It now also builds a `bf16` column at the SAME
    shapes/arms (`for prec in ("ieee", "bf16")`, the same loop
    `grid_paper_c3`/`grid_paper_c3_nscale` already use), so `PAPER_SETS`
    has something bf16 to select -- `ieee` stays in the FULL grid
    unconditionally (`--cells full`, the default): this only ADDS a
    column, it never removes the history the ieee rows already are.
    """
    g = "paper_c3_lscale"
    cfg = dict(sh.PAPER_C3_LSWEEP)
    if os.environ.get("BENCH_LSCALE_D"):
        cfg["D"] = int(os.environ["BENCH_LSCALE_D"])
    cells: list[Cell] = []
    for L in cfg["L"]:
        w = f"B{cfg['B']}D{cfg['D']}N{cfg['N']}L{L}"
        for prec in ("ieee", "bf16"):
            cells.append(_m1_cell(g, cfg["B"], cfg["D"], cfg["N"], L, prec,
                                  "fwd_bwd", work=w))
            for P in (1, 4):
                cells.append(_mh_cell(g, f"ours_p{P}", cfg["B"],
                                      cfg["D"] // P, P, cfg["N"], L, prec,
                                      "fwd_bwd", work=w))
    return cells


def grid_ssd_parity26() -> list[Cell]:
    """The signed contract's 26-cell Mamba-2 parity grid.

    State-size split: two N columns, `sh.SSD_PARITY_N_VALUES` = (64,
    128) -- 64 is the ORIGINAL contract shape (kept, secondary, `--cells
    full` only); 128 is official Mamba-2's own `d_state` default
    (confirmed by introspection against `mamba_ssm.modules.mamba2.
    Mamba2`), the HEADLINE `PAPER_SETS` now selects
    (`_paper_ssd_parity`).  `N` rides in `work` so the round-robin
    comparison unit never spans two state sizes -- a ratio between an
    N=64 cell and an N=128 cell would not be the like-for-like claim
    this grid exists to make.
    """
    g = "ssd_parity26"
    t = sh.SSD_PARITY_T
    cells: list[Cell] = []
    for c in sh.SSD_PARITY26:
        for n in sh.SSD_PARITY_N_VALUES:
            tags = dict(sweep=c["sweep"])
            w = f"{c['sweep']}|B{c['B']}H{c['H']}G{c['G']}L{c['L']}|N{n}"
            for prec in ("ieee", "bf16"):
                for p in ("fwd", "fwd_bwd"):
                    cells.append(_ssd_cell(g, c["B"], c["H"], c["G"], 1, 1, n,
                                           c["L"], prec, p, chunk=t, work=w,
                                           tags=tags))
                    cells.append(_m2_cell(g, c["B"], c["H"], c["G"], n, c["L"],
                                          prec, p, chunk=t, work=w, tags=tags))
    return cells


def grid_ssd_parity_m3() -> list[Cell]:
    """Kernel-vs-kernel: `ssd_mimo_scan` rotation mode vs `mamba3_siso_
    combined`, at the ssd_parity26 shapes, R = O = 1.

    We beat Mamba-2 on every ssd_parity26 cell
    -- do we also beat Mamba-3 SISO's own rotation path, or is ITS
    all-real-arithmetic route faster than ours at the same work?  This is
    NOT the `test_ssd_mimo_scan_reproduces_mamba3_siso` parity map
    (the M3 parity test), which reformulates the trapezoid as a
    host-side rank-1 correction so the two sides compute the SAME function
    bit-for-bit; that map's extra host arithmetic would bias a speed
    comparison against us.  This grid instead runs each kernel in ITS OWN
    native rotation configuration -- ours: `theta` + `theta_h` + `rot_bias`
    all set (`build_ssd(..., feature="rotation")`, the composite added
    above for this purpose); theirs: `mamba3_siso_combined` exactly as
    `ssm_hybrid.py`'s const rows call it -- Trap, Q_bias/K_bias, Angles all
    on (`baselines.build_m3_siso`, already the option set the smoke grid's
    `_m3_siso_cell` uses) -- so each side pays for what it actually
    computes, not for an equivalence proof.

    State-size split: same two-N-column fix as `grid_ssd_parity26` --
    `sh.SSD_PARITY_N_VALUES` = (64, 128).  At N=128, `build_m3_siso`'s
    own `angle_pairs` default (`max(2, (N // 4) & ~1)`) computes 32
    without any override -- exactly Mamba-3's own `rope_fraction=0.5`
    convention (half the state pairs) at this state width, so "Mamba-3
    at its real defaults" needs no change to `baselines.build_m3_siso`
    itself, only to the `N` this grid calls it at.
    """
    g = "ssd_parity_m3"
    t = sh.SSD_PARITY_T
    cells: list[Cell] = []
    for c in sh.SSD_PARITY26:
        for n in sh.SSD_PARITY_N_VALUES:
            tags = dict(sweep=c["sweep"])
            w = f"{c['sweep']}|B{c['B']}H{c['H']}G{c['G']}L{c['L']}|N{n}"
            for prec in ("ieee", "bf16"):
                for p in ("fwd", "fwd_bwd"):
                    cells.append(_ssd_cell(g, c["B"], c["H"], c["G"], 1, 1, n,
                                           c["L"], prec, p, chunk=t, work=w,
                                           feature="rotation", tags=tags))
                    cells.append(_m3_siso_cell(g, c["B"], c["H"], c["G"], n,
                                               c["L"], prec, p, chunk=t, work=w,
                                               tags=tags))
    return cells


# ── the equivalent-config Mamba-3 grid (K9, rewired K10) ────────────
# ORDER MATTERS for reading the table: the trapezoid-ON rows come first
# because `rot_mode="m3"` IS Mamba-3 with its trapezoid,
# so `a4_trapon` is the headline and the trapoff rows are
# the secondary "what does the trapezoid cost us" split.
_M3_EQ_VARIANTS = (
    # (angle convention, trapezoid off, tag)
    ("m3", False, "a4_trapon"),
    ("half", False, "a2_trapon"),
    ("m3", True, "a4_trapoff"),
    ("half", True, "a2_trapoff"),
)


def _m3_eq_cell(grid, B, H, P, N, L, precision, pass_, arm, angles,
                trap_off, variant, work, chunk=64, tags=None,
                themap="head"):
    dtype = IO_DTYPE[precision]
    kprec = KERNEL_PRECISION[precision]
    ka = bl.m3_eq_angle_pairs(N, angles)
    entry = SSD_ENTRY if arm == "ours" else bl.M3_SISO.entry_point
    # OUR arm runs the PRODUCTION chunk dispatch (chunk="auto": `rot_route.pick` / the calibrated structural rule), not a pinned T: a pinned
    # chunk=64 forces the SISO route, which production does not take everywhere (Blackwell bwd, kernel scope: pinned 3.88 ms, auto 3.56,
    # official 3.63 -- a pinned cell understated us by 8-20 %).  Official Mamba-3 keeps its own chunk 64.  `PARITY_PIN_CHUNK=1` restores the
    # pinned-T cell as a LABELLED variant (tag chunk_mode = "pinned").
    pinned = os.environ.get("PARITY_PIN_CHUNK", "0") == "1"
    if arm == "ours" and not pinned:
        chunk = "auto"
    tags = dict(tags or {})
    tags["chunk_mode"] = "pinned" if (pinned or arm != "ours") else "auto"
    if arm == "ours":
        tags["chunk_arm"] = chunk

    def build():
        # The two arms must compute the SAME function on the SAME inputs;
        # a cell whose arms disagree is a recorded FAILURE, never a timed
        # row (`_build_cell` turns the AssertionError into a SKIP).
        bl.m3_eq_check(B, H, P, N, L, chunk=chunk, angles=angles,
                       trap_off=trap_off, dtype=dtype, precision=kprec,
                       grads=(pass_ == "fwd_bwd"), themap=themap)
        return make_pass(
            *bl.build_m3_eq(B, H, P, N, L, arm=arm, chunk=chunk,
                            angles=angles, trap_off=trap_off, dtype=dtype,
                            precision=kprec, themap=themap), pass_)

    return Cell(
        family="ssd_mimo", arm=("ours_eq" if arm == "ours" else "m3_siso_eq"),
        entry_point=entry,
        shape=dict(B=B, H=H, G=P, R=1, O=1, N=N, L=L, chunk=chunk,
                   d_inner=H * P),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        tags=dict(tags, variant=variant, angle_pairs=ka,
                  trap_off=trap_off, feature="m3_equivalent",
                  themap=themap, bias=bl.m3_eq_bias_for(themap)),
        build=build,
    )


def grid_ssd_parity_m3_eq() -> list[Cell]:
    """Ours vs `mamba3_siso_combined` at an EQUIVALENT configuration.

    `ssd_parity_m3` runs each kernel in its own native rotation
    configuration, so its ratio prices two different functions: ours
    rotates all N/2 pairs and carries theta + theta_h + rot_bias, theirs
    rotates N/4 pairs and carries Trap + Q_bias + K_bias + Angles.  This
    grid instead drives BOTH arms from ONE set of Mamba-3 leaves through
    the parameter map the M3 parity test proves exact
    (`baselines.build_m3_eq`), asserts the two arms agree
    before timing, and prices the two conventions the comparison turns on:

      * ANGLE COUNT -- `a4` is Mamba-3's own default
        `max(2, (N//4) & ~1)` rotated pairs (our side zero-pads the phase
        on the rest); `a2` raises both to N/2.  Both are exactly the same
        function; only the work differs.
      * TRAPEZOID -- `trapon` (the HEADLINE, and what `rot_mode="m3"`
        means) keeps a random `Trap`: Mamba-3 pays its two-tap write
        in-kernel and we pay the exactly-equivalent collapsed write plus
        a rank-1 output correction, host-side, inside the timed closure.
        `trapoff` sets `Trap` to a constant whose sigmoid is exactly 1.0,
        which is the only way to switch Mamba-3's two-tap write off (it
        is a runtime value, not a flag), so neither side pays for it --
        the secondary row that prices the trapezoid on its own.

      * WHERE OUR PHASE AND BIAS LIVE (`K10_EQ_MAP`, K10) -- "head"
        (default) hands `Angles[B, L, H, ka]` straight to the kernel's
        per-head `theta` slot and Mamba-3's per-head `Q_bias`/`K_bias`
        straight to `gamma`/`beta` with `bias_pre_rot` (its own fold
        ORDER, bias INSIDE the rotation), so BOTH arms are KERNEL-only
        and nothing is re-laid out; "fold" is K9's head-shared phase
        slot, which forces a B*H batch fold of every per-token tensor and
        a per-head B/C to place the bias, and is kept as the BEFORE arm.
        The two compute the same function -- the equality gate checks it
        under both.

    Everything our map costs is inside our timed closure, and Mamba-3
    does all of it in-kernel -- the conservative direction for a ratio we
    have to beat.  What is left OF that map shrank twice: K11 moved the
    write scale and the rank-1 correction into the scan kernel, and K12
    moved the phase cumsum and dropped the V re-layout (a permuted view is served on both passes).  What our arm
    still pays on the host at `a4_trapon` is the `tanh`/`sigmoid`
    elementwise maps, the bias casts, and the per-chunk phase offset --
    a reduction over T plus a cumsum over nc.
    `SSD_MIMO_PHASE_INKERNEL=0` / `SSD_MIMO_STRIDED_V=0` restore the K11
    forms of those two, which is how the before/after is measured on ONE
    binary.

    State-size split: a second N column, `sh.SSD_PARITY_N_VALUES` =
    (64, 128).  N=64
    is the ORIGINAL K9/K10 shape (kept, secondary, `--cells full` only);
    N=128 is Mamba-3's own state-width default and the HEADLINE
    `PAPER_SETS` now selects (`_paper_ssd_parity_m3_eq`); the
    equal-config forward+backward gap measured 2.90x at
    N=128 (vs 2.51x at N=64), so this is not just a bookkeeping
    change.  `N` rides in `w` (the group key) for the same reason
    `grid_ssd_parity26` puts it there: an N=64/N=128 pair must never be
    round-robined as if it were one ratio.
    """
    g = "ssd_parity_m3_eq"
    t = sh.SSD_PARITY_T
    # WHERE our arm carries the per-(token, head) phase.  "head" (default,
    # K10) is the kernel's per-head `theta`; "fold" is K9's head-shared
    # slot plus the B*H batch re-layout, kept so the before/after is one
    # env var apart on the SAME binary.
    themap = os.environ.get("K10_EQ_MAP", "head")
    # 26 shapes x 2 N x 4 variants x 2 precisions x 2 passes x 2 arms
    # = 1664 cells (was 832 at N=64 only, before the second N column).  K9_EQ_VARIANTS
    # shards it by variant tag (comma-separated) so a run can be split
    # across nodes or budgets; unset means all four.
    want = os.environ.get("K9_EQ_VARIANTS", "")
    keep = [v.strip() for v in want.split(",") if v.strip()]
    variants = [v for v in _M3_EQ_VARIANTS if not keep or v[2] in keep]
    if keep and not variants:
        raise ValueError(
            "K9_EQ_VARIANTS=%r matched none of %s"
            % (want, [v[2] for v in _M3_EQ_VARIANTS]))
    cells: list[Cell] = []
    for c in sh.SSD_PARITY26:
        for n in sh.SSD_PARITY_N_VALUES:
            for angles, trap_off, variant in variants:
                tags = dict(sweep=c["sweep"])
                w = (variant + "|" + c["sweep"] + "|B" + str(c["B"]) + "H"
                     + str(c["H"]) + "G" + str(c["G"]) + "L" + str(c["L"])
                     + "|N" + str(n))
                for prec in ("tf32", "bf16"):
                    for p in ("fwd", "fwd_bwd"):
                        for arm in ("ours", "m3"):
                            cells.append(_m3_eq_cell(
                                g, c["B"], c["H"], c["G"], n, c["L"], prec, p,
                                arm, angles, trap_off, variant, w, chunk=t,
                                tags=tags, themap=themap))
    return cells


def grid_ssd_long_l() -> list[Cell]:
    """The long-L envelope; its own shard (a slow tail arm idles a node)."""
    g = "ssd_long_l"
    n, t = sh.SSD_PARITY_N, sh.SSD_PARITY_T
    cells: list[Cell] = []
    for c in sh.SSD_LONG_L:
        w = f"B{c['B']}H{c['H']}G{c['G']}L{c['L']}"
        for p in ("fwd", "fwd_bwd"):
            cells.append(_ssd_cell(g, c["B"], c["H"], c["G"], 1, 1, n, c["L"],
                                   "bf16", p, chunk=t, work=w,
                                   tags=dict(sweep="long_l")))
            cells.append(_m2_cell(g, c["B"], c["H"], c["G"], n, c["L"],
                                  "bf16", p, chunk=t, work=w,
                                  tags=dict(sweep="long_l")))
    return cells


def grid_ssd_mimo_gates() -> list[Cell]:
    """What (R, O) and G cost, plus the R4O4 gate cell and chunk 128."""
    g = "ssd_mimo_gates"
    cells: list[Cell] = []

    def work_of(c) -> str:
        # d_inner and (B, L, N) are the constant work; (R, O), G and the
        # chunk size are the ARMS whose cost this grid exists to price,
        # so they must all interleave inside one group.
        return f"B{c['B']}L{c['L']}d{c['d_inner']}N{c['N']}"

    for c in sh.MIMO_GATES:
        H = c["d_inner"] // (c["G"] * c["R"])
        if H < 1:
            continue
        for prec in ("tf32", "bf16"):
            for p in ("fwd", "fwd_bwd"):
                for chunk in sh.SSD_CHUNKS:
                    cells.append(_ssd_cell(
                        g, c["B"], H, c["G"], c["R"], c["O"], c["N"], c["L"],
                        prec, p, chunk=chunk, work=work_of(c),
                        tags=dict(gate="mimo")))
    # the contract gate cell, all three precisions.  Its work key
    # matches the MIMO_GATES cells at the same (B, L, d_inner, N), so at
    # tf32/bf16 it lands in their group and is round-robined against the
    # R1O1 base.  `ieee` has no MIMO_GATES rows at all, so the R1O1 base
    # is added there explicitly — otherwise the gate cell would be alone
    # in its group and "R4O4 costs Kx R1O1" would be a cross-group ratio.
    c = sh.SSD_R4O4_GATE
    H = c["d_inner"] // (c["G"] * c["R"])
    for prec in ("ieee", "tf32", "bf16"):
        for p in ("fwd", "fwd_bwd"):
            cells.append(_ssd_cell(g, c["B"], H, c["G"], c["R"], c["O"],
                                   c["N"], c["L"], prec, p, chunk=64,
                                   work=work_of(c),
                                   tags=dict(gate="r4o4_18ms")))
            if prec == "ieee":
                cells.append(_ssd_cell(
                    g, c["B"], c["d_inner"] // c["G"], c["G"], 1, 1, c["N"],
                    c["L"], prec, p, chunk=64, work=work_of(c),
                    tags=dict(gate="mimo")))
    return cells


def grid_ssd_features() -> list[Cell]:
    """Each SSD-MIMO feature against its own absence, one fixed shape."""
    g = "ssd_features"
    base = sh.SSD_FEATURE_BASE
    w = (f"B{base['B']}H{base['H']}G{base['G']}R{base['R']}O{base['O']}"
         f"N{base['N']}L{base['L']}")
    cells: list[Cell] = []
    for feat in sh.SSD_FEATURES:
        for prec in ("tf32", "bf16"):
            for p in ("fwd", "fwd_bwd"):
                # every feature and the `none` base share one group: the
                # ledger is exactly `feature / none` at this shape
                cells.append(_ssd_cell(
                    g, base["B"], base["H"], base["G"], base["R"], base["O"],
                    base["N"], base["L"], prec, p, chunk=base["T"],
                    feature=feat, work=w, tags=dict(feature=feat)))
    return cells


def _real_fair_cell(grid, B, H, P, N, L, pass_, arm, chunk, work, tags=None, *,
                    chunk_ours, chunk_m2=256, chunk_bwd=None):
    """One arm of the FAIR real R=O=1 pair (`baselines.build_real_fair`)."""
    ours = arm == "ours"

    def build():
        # a cell whose arms disagree is a recorded FAILURE, never a timed row
        bl.real_fair_check(B, H, P, N, L, chunk_ours=chunk_ours, chunk_m2=chunk_m2)
        return make_pass(*bl.build_real_fair(B, H, P, N, L, arm=arm, chunk=chunk,
                                                    chunk_bwd=chunk_bwd if ours else None), pass_)

    return Cell(
        family="ssd_mimo", arm=("ours_fair" if ours else "m2_fair"),
        entry_point=SSD_ENTRY if ours else bl.M2_CHUNK_SCAN.entry_point,
        shape=dict(B=B, H=H, G=P, R=1, O=1, N=N, L=L, chunk=chunk, d_inner=H * P),
        precision="bf16", pass_=pass_, grid=grid,
        group=group_key(grid, work, "bf16", pass_),
        tags=dict(tags or {}, feature="m2_block", chunk_arm=chunk,
                  features="dt_bias+softplus+D+gated_rmsnorm"),
        build=build)


def grid_kernel_shared_set() -> list[Cell]:
    """The ONE fixed shape set (`src/orbit_ssm/kernels/shape_sets.py`: 3 size
    classes x {balanced, long, wide, batch} at equal work + the LM cells) at
    kernel level, bf16, N = 128: our real R = O = 1 block vs the FAIR Mamba-2
    block (official `mamba_chunk_scan_combined` at its own chunk 256 with D, dt_bias,
    softplus and the gated RMSNorm, ours at the production chunk rule; both arms'
    chunks in the cell tags), and our rotating scan vs Mamba-3 SISO at the
    equivalent configuration (a4_trapon, M3 at its default chunk 64).  The calibrated kernel defaults are
    measured on the same cells, so the paper's kernel
    tables and the calibration rest on one set."""
    from orbit_ssm.kernels import shape_sets as ss
    g = "kernel_shared_set"
    n, t = sh.SSD_PARITY_N128, sh.SSD_PARITY_T
    cells: list[Cell] = []
    for tag, (B, d_inner, L) in ss.SET.items():
        H, P = d_inner // ss.P, ss.P
        w = f"{tag}|B{B}H{H}G{P}L{L}|N{n}"
        tags = dict(sweep=tag, shape=tag)
        for p in ("fwd", "fwd_bwd"):
            # REAL arm: the FAIR pair -- official M2 at ITS default chunk 256 with
            # D / dt_bias / softplus / gated RMSNorm, ours at the PRODUCTION chunk rule
            # (never pinned to M2's or a fixed 64); each arm's chunk is in its tags
            # (per-GPU calibrated for the real R = O = 1 route: forward T, and the
            # backward's T of a training call when it differs)
            from orbit_ssm.ssm.ssd_mimo import _scan_chunk, _scan_chunk_bwd
            t_ours = _scan_chunk(1, P, 1, needs_bwd=(p == "fwd_bwd"), real=True)
            t_bwd = (_scan_chunk_bwd(1, P, 1, t_ours, real=True)
                     if p == "fwd_bwd" else None)
            rt = dict(tags, kind="real", chunk_bwd=t_bwd or t_ours)
            cells.append(_real_fair_cell(g, B, H, P, n, L, p, "ours", t_ours,
                                         w + "|real", tags=rt, chunk_ours=t_ours,
                                         chunk_bwd=t_bwd))
            cells.append(_real_fair_cell(g, B, H, P, n, L, p, "m2", 256,
                                         w + "|real", tags=rt, chunk_ours=t_ours))
            for arm in ("ours", "m3"):
                cells.append(_m3_eq_cell(g, B, H, P, n, L, "bf16", p, arm,
                                         "m3", False, "a4_trapon", w + "|rot",
                                         chunk=t, tags=dict(tags, kind="rot", chunk_arm=t)))
    return cells


_GRID_FNS: dict[str, Callable[[], list[Cell]]] = {
    "kernel_shared_set": grid_kernel_shared_set,
    "smoke": grid_smoke,
    "mh_siso": grid_mh_siso,
    "mh_features": grid_mh_features,
    "paper_c3": grid_paper_c3,
    "paper_c3_nscale": grid_paper_c3_nscale,
    "paper_c3_lscale": grid_paper_c3_lscale,
    "ssd_parity26": grid_ssd_parity26,
    "ssd_parity_m3": grid_ssd_parity_m3,
    "ssd_parity_m3_eq": grid_ssd_parity_m3_eq,
    "ssd_long_l": grid_ssd_long_l,
    "ssd_mimo_gates": grid_ssd_mimo_gates,
    "ssd_features": grid_ssd_features,
}


def build_grid(name: str, args: argparse.Namespace) -> list[Cell]:
    return _GRID_FNS[name]()


# ═════════════════════════════════════════════════════════════════════
# "paper" cell sets: ONLY the cells a paper
# table or sentence quotes, per the paper's tables.  Each
# predicate is applied AFTER the grid's own build/precision/pass loops, so
# it narrows an already-built `Cell` list rather than re-deriving one --
# see `benchmarks/README.md`'s "paper set" section for the cell -> table
# row mapping.  `smoke` has none on purpose ("never quote a smoke
# number"), so `--grid paper:smoke` refuses (`harness.main_for`).
# ═════════════════════════════════════════════════════════════════════
def _paper_mh_features(c: Cell) -> bool:
    """One contract shape (`sh.MH_CONTRACT_SHAPES_PAPER`), both P
    schedules (native P and the P=1 twin) it already builds."""
    s = sh.MH_CONTRACT_SHAPES_PAPER[0]
    return (c.shape.get("B"), c.shape.get("D"), c.shape.get("N"),
            c.shape.get("L")) == (s["B"], s["H"] * s["P"], s["N"], s["L"])


def _paper_paper_c3(c: Cell) -> bool:
    """The P ladder, fwd and fwd+bwd, at the spine shape only, bf16 only
    (ieee is a numerics-validation mode, dropped from every
    future benchmark submission and from the paper's quoted columns;
    this grid never builds a tf32 cell at all)."""
    if c.precision == "ieee":
        return False
    s = sh.PAPER_C3_SHAPES_PAPER[0]
    return (c.shape.get("B"), c.shape.get("D"),
            c.shape.get("L")) == (s["B"], s["D"], s["L"])


def _paper_ssd_parity(c: Cell) -> bool:
    """The 6 representative shapes of the 26 (batch/length/head-width
    extremes) -- shared by `ssd_parity26` and `ssd_parity_m3` -- bf16
    only (this grid never builds a tf32 cell), N=128 only
    (official Mamba-2/Mamba-3's own `d_state` default
    is the HEADLINE state size a like-for-like claim must be measured
    at; the original N=64 shape stays under `--cells full` as the
    labelled secondary column, `emit_tables.emit_ssd_parity`'s new `$N$`
    column)."""
    if c.precision == "ieee":
        return False
    if c.shape.get("N") != sh.SSD_PARITY_N128:
        return False
    return (c.tags.get("sweep"), c.shape.get("B"), c.shape.get("H"),
            c.shape.get("G"), c.shape.get("L")) in sh.SSD_PARITY26_PAPER_KEYS


def _paper_ssd_parity_m3_eq(c: Cell) -> bool:
    """One variant (a4, trapezoid on both sides) at the 6 shapes, N=128
    only (the same headline-state-size correction as
    `_paper_ssd_parity` above).  Precision is left as this grid already had
    it (tf32+bf16, no ieee) -- the state-size split is a STATE-SIZE correction, not a
    precision one."""
    if c.shape.get("N") != sh.SSD_PARITY_N128:
        return False
    return (c.tags.get("variant") == "a4_trapon"
            and (c.tags.get("sweep"), c.shape.get("B"), c.shape.get("H"),
                c.shape.get("G"), c.shape.get("L"))
            in sh.SSD_PARITY26_PAPER_KEYS)


_SSD_LONG_L_PAPER_KEYS = frozenset(
    (c["B"], c["H"], c["G"], c["L"]) for c in sh.SSD_LONG_L_PAPER)


def _paper_ssd_long_l(c: Cell) -> bool:
    """The two headline lengths the contract doc numbers (B1 L32k/L64k)."""
    return (c.shape.get("B"), c.shape.get("H"), c.shape.get("G"),
            c.shape.get("L")) in _SSD_LONG_L_PAPER_KEYS


def _paper_ssd_mimo_gates(c: Cell) -> bool:
    """The R4O4 gate cell plus the worst-case (R=O, G=16, N=128) sweep
    the table's caption names by number (7.46 ms at G=16, chunk=128,
    N=128) -- both chunk sizes and R in {1, 2, 4} so the "no cell of any
    (R, O) exceeds it" comparison the caption makes is still
    reproducible.  tf32+bf16 only: the caption itself calls the recorded
    ieee cells "not part of the gate below", and ieee is dropped
    from the paper's quoted columns outright, so the dedicated ieee R1O1
    companion this grid also builds is excluded here."""
    if c.precision == "ieee":
        return False
    gate = c.tags.get("gate")
    if gate == "r4o4_18ms":
        return True
    if gate == "mimo":
        return c.shape.get("G") == 16 and c.shape.get("N") == 128
    return False


def _paper_accept_all(c: Cell) -> bool:
    """Grids already minimal for the paper (the sweep axis itself is the
    deliverable, or the grid is already one shape) and with no ieee cell
    to exclude: `mh_features`'s law rows aside, `ssd_features` (the
    feature ledger, tf32/bf16 only, already one shape)."""
    return True


def _paper_accept_all_no_ieee(c: Cell) -> bool:
    """Like `_paper_accept_all`, minus ieee (dropped from the
    paper's quoted columns): `mh_siso` (4 shapes, never run before,
    ieee/tf32/bf16 in the full grid), `paper_c3_nscale` (Table 11's N
    sweep, ieee/bf16 in the full grid) and `paper_c3_lscale` (Figure 4's
    L sweep, ieee/bf16 in the full grid) -- see that
    grid's own docstring for the bf16 column."""
    return c.precision != "ieee"


PAPER_SETS: dict[str, Callable[[Cell], bool]] = {
    "mh_siso": _paper_accept_all_no_ieee,
    "mh_features": _paper_mh_features,
    "paper_c3": _paper_paper_c3,
    "paper_c3_nscale": _paper_accept_all_no_ieee,
    # `paper_c3_lscale` (Figure 4) used to have NO entry
    # here -- every cell the grid built was precision="ieee"
    # (`grid_paper_c3_lscale` hardcoded it, predating a bf16 column), so
    # the rule "no ieee in the paper's quoted columns" left it with
    # nothing to quote under the compact protocol and `--cells paper`
    # refused, the same as `smoke`.  The grid now
    # also builds a bf16 column at the same shapes/arms; this predicate
    # selects it exactly like `mh_siso`/`paper_c3_nscale` do.  `ieee`
    # keeps running under `--cells full` (the default) -- unaffected,
    # nothing here deletes that history.
    "paper_c3_lscale": _paper_accept_all_no_ieee,
    "ssd_parity26": _paper_ssd_parity,
    "ssd_parity_m3": _paper_ssd_parity,
    "ssd_parity_m3_eq": _paper_ssd_parity_m3_eq,
    "ssd_long_l": _paper_ssd_long_l,
    "ssd_mimo_gates": _paper_ssd_mimo_gates,
    "ssd_features": _paper_accept_all,
    # the shared shape set: every cell is a paper cell (bf16 only by construction)
    "kernel_shared_set": _paper_accept_all,
}


# ── self-CUDA column for the kernel parity grids ─────
# The parity grids (`ssd_parity26`, `ssd_parity_m3`, `ssd_parity_m3_eq`)
# time the whole scan autograd Function per call — WALL time.  For an
# autograd Function built around a fused whole-layer op (official
# Mamba-2's `mamba_split_conv1d_scan_combined`) that includes host-side
# torch work the Function performs around its own kernel launches; two
# kernels can therefore rank differently by wall time than by the GPU
# work they actually do.  `ms_cuda_self` records the summed self-CUDA
# time of the same call (one extra `torch.profiler` pass per cell, via
# the SAME `cuda_buckets` helper `bench_ssms.py`'s `--profile` uses,
# hoisted to `harness.py` so both modules share it) for BOTH arms in
# these three grids, so a table can print
# wall and GPU time side by side with a ratio for each; see
# `benchmarks/README.md` for which is which.
_SELF_CUDA_GRIDS = frozenset({"ssd_parity26", "ssd_parity_m3",
                              "ssd_parity_m3_eq"})


def _post_cell(args: argparse.Namespace):
    # The extra profiler pass is the single largest per-cell,
    # budget-invariant cost the parity grids carry — it is dropped
    # from the compact budget specifically.  `Budget.cuda_self` is
    # that one-line flip (`harness.PAPER_BUDGET`'s own field); resolved
    # here via the SAME `resolve_budget(args)` `main_for` uses, so a
    # `--budget` override on the CLI is honored identically.
    budget = resolve_budget(args)

    def hook(cell: Cell, closure):
        out: dict = {}
        if cell.grid == "kernel_shared_set" and cell.arm in ("ours_fair", "ours_eq"):
            # the route that ACTUALLY served our arm + the calibration source of its launch configs
            from .route_record import served
            try:
                out.update(served(closure))
            except Exception as exc:               # noqa: BLE001
                out["route_error"] = f"{type(exc).__name__}: {exc}"
        if cell.grid in _SELF_CUDA_GRIDS and budget.cuda_self:
            try:
                out["ms_cuda_self"] = cuda_buckets(closure, iters=1)["total"]
            except Exception as exc:               # noqa: BLE001
                out.update(ms_cuda_self=None,
                           ms_cuda_self_error=f"{type(exc).__name__}: {exc}")
        return out or None

    return hook


if __name__ == "__main__":
    main_for(build_grid, GRIDS, "benchmarks.bench_kernels",
             post_cell=_post_cell, paper_filter=PAPER_SETS.get)
