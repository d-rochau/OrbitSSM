"""Reference-kernel and reference-module adapters.

Every adapter exposes ``available() -> (bool, reason)``.  A missing
install must produce a *recorded skip* with the reason in the JSONL, never
an exception that costs the rest of a sweep — the old scripts' per-cell
`try/except` had the right idea and threw the reason away.

All four reference kernels and all three reference modules were verified
importable with torch 2.11.0+cu130, triton 3.6.0,
mamba_ssm 2.3.2.post1 (editable install), tilelang 0.1.8.

The builders return ``(call, leaves)``: ``call()`` runs the reference once
and returns its output; ``leaves`` are the tensors whose gradients a
backward pass must produce.  `harness.make_pass` turns that pair into the
timed closure, so every arm — ours and theirs — gets the identical
no-grad / real-random-dY treatment.

Fairness notes that are load-bearing for the ratios:

* **M1 (`selective_scan_fn`)** is called with the D skip and the z gate
  OFF.  mamba_ssm fuses both and mh_scan carries neither on the SISO
  path, so including them would time work only one side does — and
  dropping them makes the REFERENCE cheaper, which is the conservative
  direction for a ratio we have to beat.  Both sides run
  `delta_softplus=True` and our `A_log` is handed over as `-exp(A_log)`.
  Source: the mh SISO bench, measured 2026-08 (script retired).
* **M2 (`mamba_chunk_scan_combined`)** is called with `D=None`, `z=None`,
  `dt_softplus=False` at ngroups = 1, matching the SSD-MIMO R=O=1 corner
  cell for cell.  Source: the retired parity-sweep bench (git history).
* **M3 SISO (`mamba3_siso_combined`)** is called exactly as the shipped
  mixer calls it: D=None, Z=None, no
  Input_States, chunk 64.
* **M3 MIMO (`mamba3_mimo`)** is TileLang, not Triton, and its exclusion
  is scoped to WALL-CLOCK metrics only (measured).
  WALL: ~900 ms-1 s per fwd+bwd call at L = 2048, which is ~40x the
  kernel's own device time and 30-120x the peer arms' whole wall time.
  This is NOT a JIT/warmup artefact -- it persists unabated through 55
  calls at a fixed shape (call 1 is a separate ~110-120 s TileLang
  compile) -- and it is NOT the "~150 ms" this note used to record, which
  undersold it ~6x.  It is TileLang's own un-jitted `kernel_cache`
  dispatch path (the wrapper is not `@tilelang.jit`), ~81 % of it inside
  `_Mamba3FunctionBackward`'s frame with no child op absorbing it.  Keep
  it out of every wall-clock ratio table.
  DEVICE (`cuda_self`, the metric the structure and compile
  comparisons already reason in): competitive, and it is our only
  STRUCTURAL peer -- its B/C are grouped `(r, g, n)` as ours are and it
  rotates in-kernel as we do, whereas `m3_siso` is rank-1 with per-head
  Q/K.  A100, matched geometry, kernel level: 25.2 vs our R2O2-rot 26.5
  (S-row) and 17.2 vs 18.4 (LM), i.e. 0.93-0.95x ours; their own
  MIMO/SISO multiplier is 5.6-7.8x, against our 8.2x.  Cite it in
  cuda_self tables, labelled device-time-only with the wall caveat.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F


# ── availability probes ──────────────────────────────────────────────
@dataclass(frozen=True)
class Adapter:
    """A reference implementation, addressed by import path."""

    name: str
    module: str
    attr: str
    what: str

    def available(self) -> tuple[bool, str]:
        try:
            mod = importlib.import_module(self.module)
        except Exception as exc:                   # noqa: BLE001
            return False, f"import {self.module}: {type(exc).__name__}: {exc}"
        if not hasattr(mod, self.attr):
            return False, f"{self.module} has no attribute {self.attr!r}"
        return True, "ok"

    def load(self):
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"{self.name} unavailable: {why}")
        return getattr(importlib.import_module(self.module), self.attr)

    @property
    def entry_point(self) -> str:
        return f"{self.module}.{self.attr}"


M1_SELECTIVE_SCAN = Adapter(
    "m1_selective_scan", "mamba_ssm.ops.selective_scan_interface",
    "selective_scan_fn", "Mamba-1 reference selective scan (CUDA)")
M2_CHUNK_SCAN = Adapter(
    "m2_chunk_scan", "mamba_ssm.ops.triton.ssd_combined",
    "mamba_chunk_scan_combined", "Mamba-2 SSD chunked scan (Triton)")
M3_SISO = Adapter(
    "m3_siso", "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
    "mamba3_siso_combined", "Mamba-3 SISO combined (Triton)")
M3_MIMO = Adapter(
    "m3_mimo", "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo",
    "mamba3_mimo", "Mamba-3 MIMO (TileLang)")

MAMBA1_LAYER = Adapter("mamba1_layer", "mamba_ssm.modules.mamba_simple",
                       "Mamba", "official Mamba-1 mixer")
MAMBA2_LAYER = Adapter("mamba2_layer", "mamba_ssm", "Mamba2",
                       "official Mamba-2 mixer")
MAMBA3_LAYER = Adapter("mamba3_layer", "mamba_ssm", "Mamba3",
                       "official Mamba-3 mixer")

KERNEL_ADAPTERS = (M1_SELECTIVE_SCAN, M2_CHUNK_SCAN, M3_SISO, M3_MIMO)
MODULE_ADAPTERS = (MAMBA1_LAYER, MAMBA2_LAYER, MAMBA3_LAYER)
ALL_ADAPTERS = KERNEL_ADAPTERS + MODULE_ADAPTERS


def availability_report() -> list[dict]:
    out = []
    for ad in ALL_ADAPTERS:
        ok, why = ad.available()
        out.append(dict(name=ad.name, entry_point=ad.entry_point,
                        available=ok, reason=why, what=ad.what))
    return out


# ── shared input helpers ─────────────────────────────────────────────
def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _r(g, *shape, scale=1.0, device="cuda", dtype=torch.float32,
       requires_grad=False) -> torch.Tensor:
    t = (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dtype)
    return t.contiguous().requires_grad_(requires_grad)


# ── kernel-level builders ────────────────────────────────────────────
def build_m1(B: int, D: int, N: int, L: int, *, dtype=torch.float32,
             device="cuda", seed=0) -> tuple[Callable[[], Any], list]:
    """Mamba-1 selective scan at (B, D, N, L), D skip and z gate OFF."""
    fn = M1_SELECTIVE_SCAN.load()
    g = _rng(seed)
    u = _r(g, B, D, L, device=device, dtype=dtype, requires_grad=True)
    delta = _r(g, B, D, L, scale=0.5, device=device, dtype=dtype,
               requires_grad=True)
    a_log = _r(g, D, N, scale=0.5, device=device)
    A = (-a_log.exp()).detach().requires_grad_(True)
    Bt = _r(g, B, N, L, device=device, dtype=dtype, requires_grad=True)
    Ct = _r(g, B, N, L, device=device, dtype=dtype, requires_grad=True)

    def call():
        return fn(u, delta, A, Bt, Ct, delta_softplus=True)

    return call, [u, delta, A, Bt, Ct]


def build_m2(B: int, H: int, P: int, N: int, L: int, *, chunk: int = 64,
             ngroups: int = 1, dtype=torch.bfloat16, device="cuda",
             seed=0) -> tuple[Callable[[], Any], list]:
    """Mamba-2 SSD chunked scan matched to the SSD-MIMO R=O=1 corner."""
    fn = M2_CHUNK_SCAN.load()
    g = _rng(seed)
    x = _r(g, B, L, H, P, device=device, dtype=dtype, requires_grad=True)
    dt = (torch.rand(B, L, H, generator=g).to(device) * 0.1 + 0.01
          ).requires_grad_(True)
    A = (-torch.rand(H, generator=g).to(device)).requires_grad_(True)
    Bm = _r(g, B, L, ngroups, N, device=device, dtype=dtype, requires_grad=True)
    Cm = _r(g, B, L, ngroups, N, device=device, dtype=dtype, requires_grad=True)

    def call():
        return fn(x, dt, A, Bm, Cm, chunk, dt_softplus=False,
                  return_final_states=False)

    return call, [x, dt, A, Bm, Cm]


def build_real_fair(B: int, H: int, P: int, N: int, L: int, *, arm: str,
                    chunk: int, dtype=torch.bfloat16, device="cuda", seed=0,
                    chunk_bwd: int | None = None
                    ) -> tuple[Callable[[], Any], list]:
    """The FAIR real R = O = 1 kernel-level pair (campaign: bench_fair.py).

    The function both arms compute -- Mamba-2's own block between the conv and
    out_proj:  dt = softplus(dt_raw + dt_bias);  h_t = exp(-exp(a_log) dt_t) h_{t-1}
    + dt_t B_t x_t^T;  y_t = C_t^T h_t + D x_t;  out = rmsnorm(y * silu(z)) * w
    (norm_before_gate = False).

    arm "m2":   official `mamba_chunk_scan_combined(x, dt_raw, A, B, C, chunk, D=D,
                dt_bias=dt_bias, dt_softplus=True)` + `rmsnorm_fn(y, w, z, ...)`, at ITS
                chunk (256, its default -- never pinned to ours), from the strided
                views of the slab its mixer produces.
    arm "ours": `ssd_mimo_scan` with gate_z / d_skip / out_norm_w, `softplus(dt +
                dt_bias)` in the timed region, at OUR production chunk rule (the
                caller passes it), from the strided views of OUR mixer's slab.
    The caller records each arm's chunk in the cell tags."""
    import torch.nn.functional as F
    d_in = H * P
    g = _rng(seed)
    rr = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc)
    base = dict(x=rr(B, L, H, P, sc=0.5), dt_raw=rr(B, L, H) - 3.0, dt_bias=rr(H, sc=0.5),
                a_log=torch.log(torch.rand(H, generator=g) * 15 + 1), Bm=rr(B, L, N, sc=0.5),
                Cm=rr(B, L, N, sc=0.5), D=torch.ones(H) + rr(H, sc=0.1),
                z=rr(B, L, H, P, sc=0.7), w=torch.ones(d_in) + rr(d_in, sc=0.1))
    params = {k: base[k].to(device).requires_grad_() for k in ("dt_bias", "a_log", "D", "w")}
    # BOTH arms take the RAW dt pre-activation and apply dt_bias + softplus inside the timed block, and both
    # differentiate dt_bias (official M2 in-kernel; ours in the fused `_tails_fwd_kernel` with `bias_dt`, the
    # production route since SSD_MIMO_DT_FP32 became the default: the token GEMM no longer carries
    # the dt bias).  Previously our arm pre-added dt_bias outside the timing and dropped its gradient.
    if arm == "m2":
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
        fn = M2_CHUNK_SCAN.load()
        zxbcdt = torch.empty(B, L, 2 * d_in + 2 * N + H, device=device, dtype=dtype)
        zxbcdt[..., :d_in] = base["z"].reshape(B, L, d_in).to(device, dtype)
        zxbcdt[..., -H:] = base["dt_raw"].to(device, dtype)
        xbc = torch.empty(B, L, d_in + 2 * N, device=device, dtype=dtype)
        xbc[..., :d_in] = base["x"].reshape(B, L, d_in).to(device, dtype)
        xbc[..., d_in:d_in + N] = base["Bm"].to(device, dtype)
        xbc[..., d_in + N:] = base["Cm"].to(device, dtype)
        zxbcdt.requires_grad_(True); xbc.requires_grad_(True)

        def call():
            z = zxbcdt[..., :d_in]
            x = xbc[..., :d_in].view(B, L, H, P)
            Bt = xbc[..., d_in:d_in + N].view(B, L, 1, N)
            Ct = xbc[..., d_in + N:].view(B, L, 1, N)
            A = -torch.exp(params["a_log"])
            y = fn(x, zxbcdt[..., -H:], A, Bt, Ct, chunk, D=params["D"], z=None,
                   dt_bias=params["dt_bias"], dt_softplus=True)
            return rmsnorm_fn(y.reshape(B, L, d_in), params["w"], None, z=z, eps=1e-5,
                              norm_before_gate=False)
        return call, [zxbcdt, xbc, *params.values()]
    if arm != "ours":
        raise ValueError(f"arm must be m2|ours, got {arm!r}")
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    x = base["x"].permute(0, 2, 3, 1).contiguous().to(device, dtype).unsqueeze(3)   # [B,H,P,1,L]
    width = d_in + H + 2 * N
    slab = torch.zeros(B, L, width, device=device, dtype=dtype)
    slab[..., :d_in] = base["z"].reshape(B, L, d_in).to(device, dtype)
    slab[..., d_in:d_in + H] = base["dt_raw"].to(device, dtype)                     # raw pre-activation (no bias)
    slab[..., d_in + H:d_in + H + N] = base["Bm"].to(device, dtype)
    slab[..., d_in + H + N:] = base["Cm"].to(device, dtype)
    x.requires_grad_(True); slab.requires_grad_(True)
    from orbit_ssm.layers import tails as _tails
    cfg = _tails.TailCfg(N=N, R=1, O=1, H=H, L=L, bc_norm=False, b_norm=False, c_norm=False,
                         shared_bias=False, fold_gb=False, fold_gc=False, a_dyn=False,
                         a_per_sibling=False, has_dt_bias=True, has_a_bias=False,
                         eps_b=0.0, eps_c=0.0, a_floor=1e-4, a_heavy_tail=False)

    def call():
        z = slab[..., :d_in]
        dt = slab[..., d_in:d_in + H]
        Bt = slab[..., d_in + H:d_in + H + N].unsqueeze(2).unsqueeze(-1)    # [B,L,1,N,1]
        Ct = slab[..., d_in + H + N:].unsqueeze(2)                          # [B,L,1,N]
        _, _, delta, _ = _tails.token_tails(                                # softplus + layout, ONE fused kernel
            cfg, b_src=None, c_src=None, gb_src=None, gc_src=None, dt_src=dt, a_src=None,
            w_b=None, w_c=None, bias_b=None, bias_c=None, bias_dt=params["dt_bias"], bias_a=None,
            need_b=False, need_c=False, out_dtype_b=None, out_dtype_c=None)
        dsk = params["D"][:, None, None].expand(H, P, 1)
        y, _ = ssd_mimo_scan(x, delta, Bt, Ct, params["a_log"], chunk_size=chunk,
                             chunk_size_bwd=chunk_bwd,
                             y_token_major=True, precision="io", bc_per_head=True,
                             gate_z=z, d_skip=dsk, out_norm_w=params["w"], out_norm_eps=1e-5)
        return y.permute(0, 3, 1, 2).reshape(B, L, d_in)    # token-major memory: a free view
    return call, [x, slab, *params.values()]


def real_fair_check(B, H, P, N, L, *, chunk_ours, chunk_m2=256, dtype=torch.bfloat16,
                    device="cuda", tol_fwd=0.05, tol_grad=0.2) -> dict:
    """Both arms must compute the same function on the same draw before anything is
    timed: the forward AND every gradient (dx, dB, dC, dz, ddt, dD, da, ddt_bias, dw),
    under one shared random dY.  Returns the per-tensor pairwise relative errors; raises
    AssertionError naming the first tensor above its tolerance (two bf16 arms that are each
    within the 16-bit bound of the oracle agree to a few percent; `tol_grad` is the loose
    end of that, a defect-catcher, not an accuracy claim)."""
    d_in = H * P
    out = {}
    for arm, ch in (("m2", chunk_m2), ("ours", chunk_ours)):
        call, leaves = build_real_fair(B, H, P, N, L, arm=arm, chunk=ch, dtype=dtype,
                                       device=device)
        y = call()
        gy = torch.randn(y.shape, generator=_rng(1), dtype=torch.float32).to(device, y.dtype)
        y.backward(gy)
        if arm == "m2":
            zxbcdt, xbc, dtb, a_log, D, w = leaves
            g = dict(dx=xbc.grad[..., :d_in], dB=xbc.grad[..., d_in:d_in + N],
                     dC=xbc.grad[..., d_in + N:], dz=zxbcdt.grad[..., :d_in],
                     ddt=zxbcdt.grad[..., -H:])
        else:
            x, slab, dtb, a_log, D, w = leaves
            g = dict(dx=x.grad.squeeze(3).permute(0, 3, 1, 2).reshape(B, L, d_in),
                     dB=slab.grad[..., d_in + H:d_in + H + N], dC=slab.grad[..., d_in + H + N:],
                     dz=slab.grad[..., :d_in], ddt=slab.grad[..., d_in:d_in + H])
        g.update(dD=D.grad, da=a_log.grad, dw=w.grad, ddtb=dtb.grad)
        out[arm] = dict(y=y.detach().float(), **{k: v.detach().float().clone() for k, v in g.items()})
        del call, leaves, y, g
        torch.cuda.empty_cache()
    err = {k: ((out["ours"][k] - out["m2"][k]).norm()
               / out["m2"][k].norm().clamp_min(1e-30)).item() for k in out["m2"] if k in out["ours"]}
    for k, e in err.items():
        lim = tol_fwd if k == "y" else tol_grad
        if not e < lim:
            raise AssertionError(f"real fair pair disagrees on {k}: rel {e:.3e} >= {lim}")
    return err


def build_m3_siso(B: int, H: int, P: int, N: int, L: int, *, chunk: int = 64,
                  angle_pairs: int | None = None, dtype=torch.float32,
                  device="cuda", seed=0) -> tuple[Callable[[], Any], list]:
    """Mamba-3 SISO exactly as the shipped Hybrid-Mamba layer calls it.

    `angle_pairs` is M3's rotated-pair count: pairs beyond it are left
    unrotated, so it is the analogue of `rotary_dim_divisor`.  Default
    N//4 (even, at least 2) mirrors the MIMO kernel's divisor-4 default.
    """
    fn = M3_SISO.load()
    g = _rng(seed)
    ka = angle_pairs if angle_pairs is not None else max(2, (N // 4) & ~1)
    v = _r(g, B, L, H, P, device=device, dtype=dtype, requires_grad=True)
    Bt = _r(g, B, L, 1, N, device=device, dtype=dtype, requires_grad=True)
    Ct = _r(g, B, L, 1, N, device=device, dtype=dtype, requires_grad=True)
    dt = F.softplus(_r(g, B, H, L, scale=0.5, device=device) - 3.0
                    ).detach().requires_grad_(True)
    ADT = (-dt.detach()).clone().requires_grad_(True)
    trap = _r(g, B, H, L, scale=0.5, device=device, requires_grad=True)
    Qb = _r(g, H, N, scale=0.1, device=device, requires_grad=True)
    Kb = _r(g, H, N, scale=0.1, device=device, requires_grad=True)
    angles = _r(g, B, L, H, ka, scale=0.3, device=device, requires_grad=True)

    def call():
        return fn(Q=Ct, K=Bt, V=v, ADT=ADT, DT=dt, Trap=trap,
                  Q_bias=Qb, K_bias=Kb, Angles=angles, D=None, Z=None,
                  chunk_size=chunk, Input_States=None)

    return call, [v, Bt, Ct, dt, ADT, trap, Qb, Kb, angles]


# ── the EQUIVALENT-CONFIG Mamba-3 pair ──────────────
# `ssd_parity_m3` runs each kernel in its OWN native rotation
# configuration, so its ratio prices two different functions.  The pair
# below prices the SAME function on the SAME inputs: one set of Mamba-3
# leaves drives both arms, ours through the four-step map that
# the M3 parity test proves exact against `mamba3_siso_combined`
# (via `_mamba3_via_ssd_mimo`) --
#
#   1. DECAY   a_log = 0 (a = -1) and delta = -ADT, so a*delta = ADT.
#   2. WRITE   the two-tap trapezoid collapses onto the one-tap
#              recurrence with delta_w = lam*DT + DT_{t+1}(1 - lam_{t+1})
#              and a LOCAL rank-1 output correction.
#   3. PHASE   theta = -cumsum(tanh(Angles) pi DT); Mamba-3's phase is
#              PER HEAD (Angles head-shared, DT not), and our `theta` slot
#              is head-shared, so H folds into the batch axis.
#   4. BIASES  Mamba-3 adds K_bias/Q_bias BEFORE the rotation, our
#              beta/gamma after it, so under a non-zero phase the biases
#              fold into B_tok/C_tok host-side.
#
# Everything step 1-4 costs on OUR side is inside the timed closure: the
# phase cumsum, the write scale, the bias fold, the batch fold and (when
# the trapezoid is on) the rank-1 correction.  Mamba-3 does all of it
# in-kernel, so charging it to us is the conservative direction.
#
# Two knobs the grid reports both settings of:
#   `angles`    "m3" = Mamba-3's own default max(2, (N//4) & ~1) pairs,
#               "half" = N//2, the count our rotation modes carry.  The
#               map zero-pads the phase on the unrotated pairs, so BOTH
#               settings are exactly equal functions and only the work
#               differs.
#   `trap_off`  True sets Trap to a constant large enough that
#               sigmoid(Trap) is EXACTLY 1.0 in fp32 and bf16, which makes
#               Mamba-3's trapezoid term identically zero (it is a runtime
#               value, not a flag, so this is the only way to turn it off)
#               and lets our side drop the rank-1 correction.  False keeps
#               a random Trap and charges the correction to us.
#   `themap`    WHERE the per-(token, head) phase is carried.  "fold"
#               (K9) is the head-SHARED `theta` slot plus a B*H batch
#               re-layout of every per-token tensor, which is what a
#               head-shared phase forces and what K9 4.5 measured at
#               52-70% of the whole equal-config forward.  "head" (K10) is
#               the per-head `theta` slot the kernels now take directly:
#               [B, L, H, N//2] straight in, B/C left head-shared, no fold.
#   `bias`      Both maps now run Mamba-3's OWN per-head Q_bias/K_bias
#               [H, N].  "fold" (K9) reaches it by materialising a
#               per-head B/C host-side; "head" (K10) hands it to the
#               kernel as beta/gamma with `bias_pre_rot`, which is
#               Mamba-3's fold ORDER (bias INSIDE the rotation).  X1 2.5
#               measures the two orders 49% apart, so before K10 the
#               per-head bias was only reachable through the [B, L, H, N]
#               materialisation the fold pays for -- which is exactly what
#               the "head" map removes.
_TRAP_OFF_LOGIT = 30.0     # sigmoid -> 1.0 exactly at fp32 and bf16


def _m3_eq_theta_h(H, N, device):
    g = torch.Generator(device="cpu").manual_seed(4242)
    return (torch.randn(H, N // 2, generator=g) * 0.3).to(device)


def m3_eq_angle_pairs(N: int, convention: str) -> int:
    """`m3` = Mamba-3's own default (N//4, even, >= 2); `half` = N//2."""
    if convention == "m3":
        return max(2, (N // 4) & ~1)
    if convention == "half":
        return N // 2
    raise ValueError(f"angle convention must be m3|half, got {convention!r}")


def _rel_eq(a, b) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def _m3_eq_leaves(B, H, P, N, L, *, angle_pairs, trap_off, dtype,
                  device="cuda", seed=0, bias="perhead"):
    """ONE set of Mamba-3 leaves; both arms of the pair consume it.

    Shapes and scales follow the leaves of the M3 parity test, which is
    what the exactness proof was measured on.  `bias="shared"` makes the
    two bias leaves [1, N] (see the knob block above); the Mamba-3 arm
    expands them into its own [H, N] slot, which is a differentiable view,
    so both arms still differentiate the SAME leaf.
    """
    g = _rng(seed)
    f32 = dict(device=device, dtype=torch.float32)
    dt = (torch.rand(B, H, L, generator=g).to(**f32) * 0.05 + 0.01
          ).requires_grad_(True)
    adt = (-F.softplus(_r(g, B, H, L, device=device)) * dt.detach()
           ).clone().requires_grad_(True)
    if trap_off:
        trap = torch.full((B, H, L), _TRAP_OFF_LOGIT, **f32
                          ).requires_grad_(True)
    else:
        trap = _r(g, B, H, L, device=device, requires_grad=True)
    return dict(
        K=_r(g, B, L, 1, N, scale=0.3, device=device, dtype=dtype,
             requires_grad=True),
        Q=_r(g, B, L, 1, N, scale=0.3, device=device, dtype=dtype,
             requires_grad=True),
        V=_r(g, B, L, H, P, scale=0.3, device=device, dtype=dtype,
             requires_grad=True),
        ADT=adt, DT=dt, Trap=trap,
        K_bias=_r(g, 1 if bias == "shared" else H, N, scale=0.2,
                  device=device, dtype=dtype, requires_grad=True),
        Q_bias=_r(g, 1 if bias == "shared" else H, N, scale=0.2,
                  device=device, dtype=dtype, requires_grad=True),
        Angles=_r(g, B, L, H, angle_pairs, scale=0.5, device=device,
                  requires_grad=True),
    )


_M3_EQ_ORDER = ("K", "Q", "V", "ADT", "DT", "Trap", "K_bias", "Q_bias",
                "Angles")


def _m3_eq_phase(t, N, cumsum=True):
    """Mamba-3's own phase Theta_t, per (token, head), zero on the pairs
    its angle convention does not rotate.  `theta = -Theta` is the map.

    `cumsum=False` returns the per-token INCREMENTS instead -- what the
    K12 kernels take, with the running phase rebuilt from a per-chunk
    offset plus an intra-chunk `tl.cumsum`.  The cumulative sum down the
    MIDDLE (token) axis is the single most expensive host op in this map:
    0.465 ms forward and 1.23 ms fwd+bwd at B4 H32 N64 L2048, against
    0.033 ms for the `tanh * pi * dt` that feeds it.

    On the INCREMENT path the `tanh * pi * dt`,
    the negate and the `F.pad` are ONE Triton launch
    (`m3_epilogue.m3_phase_increments`) and its adjoint one more.  The
    host chain they replace is 8 of the m3 forward's 15 launches and
    ~110 us of its 149 us of torch glue at the B4 H32 P64 N128 L2048
    spine cell -- Mamba-3 does the identical arithmetic inside
    `angle_dt_fwd_kernel` and pays nothing for it.  The forward is
    BIT-IDENTICAL to the chain below (measured 0.0 relative on fp32
    leaves); the two gradients agree to 1e-7, which is `libdevice.tanh`
    against `torch.tanh` plus a different reduction order for dDT.
    `SSD_MIMO_M3_PHASE_FUSED=0` restores the host chain (the A/B lever).
    """
    dt = t["DT"]
    angle_pairs = t["Angles"].shape[-1]
    if not cumsum and os.environ.get("SSD_MIMO_M3_PHASE_FUSED", "1") == "1":
        from orbit_ssm.kernels.ssd_mimo.m3_epilogue import (
            m3_phase_increments)
        return m3_phase_increments(t["Angles"], dt, N // 2, sign=-1.0)
    incr = torch.tanh(t["Angles"]) * math.pi * dt.permute(0, 2, 1).unsqueeze(-1)
    if cumsum:
        incr = torch.cumsum(incr, dim=1)
    return F.pad(-incr, (0, N // 2 - angle_pairs))            # [B,L,H,N/2]


def _m3_eq_write(t, trap_off):
    """(delta_w, the next-tap term) -- the trapezoid as a one-tap write."""
    dt, trap = t["DT"], t["Trap"]
    if trap_off:
        return dt, None                              # lam == 1, nxt == 0
    lam = torch.sigmoid(trap)
    nxt = F.pad((dt * (1 - lam))[..., 1:], (0, 1))
    return lam * dt + nxt, nxt


def _m3_eq_ours_fold(t, *, N, chunk, precision, trap_off, rot_where=None):
    """K9's map: a HEAD-SHARED `theta` slot, so H folds into the batch.

    Kept as the BEFORE arm of the K10 table.  Every per-token tensor is
    re-laid out to (B*H, 1, ...) because the phase has no head axis, which
    is what the "head" map below removes.
    """
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    k_tok, q_tok, v = t["K"], t["Q"], t["V"]
    adt = t["ADT"]
    batch, length, heads, headdim = v.shape
    write, nxt = _m3_eq_write(t, trap_off)
    phase = _m3_eq_phase(t, N)
    k = k_tok[:, :, 0].unsqueeze(2) + t["K_bias"]             # [B,L,H,N]
    q = q_tok[:, :, 0].unsqueeze(2) + t["Q_bias"]

    def fold(x, tail):
        return x.permute(0, 2, 1, 3).reshape(batch * heads, length,
                                             tail).contiguous()

    y, _ = ssd_mimo_scan(
        v.permute(0, 2, 3, 1).reshape(batch * heads, 1, headdim, 1,
                                      length).contiguous(),
        (-adt).reshape(batch * heads, 1, length).contiguous(),
        fold(k, N), fold(q, N),
        # H has been folded into B: the scan has ONE head. A per-folded-
        # batch vector would broadcast a spurious head axis into the scan.
        torch.zeros(1, device=v.device),                       # a = -1
        chunk_size=chunk, precision=precision,
        delta_w=write.reshape(batch * heads, 1, length).contiguous(),
        theta=fold(phase, N // 2), rot_where=rot_where,
        bias_pre_rot=False)  # bias already folded into k/q host-side
                             # above and theta here is head-shared
                             # (H folded into batch), not per-head, so
                             # the True default's in-kernel pre-rot path
                             # does not apply -- pin explicitly
    y = y.reshape(batch, heads, headdim, length).permute(0, 3, 1, 2)
    if not trap_off:
        dot = (k * q).sum(-1)                                  # [B,L,H]
        y = y - (nxt.permute(0, 2, 1) * dot).unsqueeze(-1) * v
    return y


def _m3_eq_ours_head(t, *, N, chunk, precision, trap_off, rot_where=None):
    """K10's map: the PER-HEAD `theta` slot, so nothing is re-laid out.

    `Angles[B, L, H, ka]` becomes `theta[B, L, H, N//2]` DIRECTLY (zeros on
    the pairs Mamba-3 does not rotate), the heads stay the kernel's heads,
    B/C stay head-shared, and the biases -- head-shared here, see the
    `bias` knob -- are one broadcast add on a [B, L, N] tensor.  What is
    left inside the timed closure is the phase cumsum, the write scale and
    the V re-layout Mamba-3 reads for free from its own argument order.
    """
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    k_tok, q_tok, v = t["K"], t["Q"], t["V"]
    adt = t["ADT"]
    batch, length, heads, headdim = v.shape
    # ── K12: the phase cumsum runs IN the kernels ────────────────────
    # `theta_increment=True` hands them the per-token angle INCREMENTS;
    # each of the four kernels that reads the phase adds the chunk's
    # entry offset (built inside `ssd_mimo_scan` by a reduction over T
    # and a cumsum on a tensor 64x smaller) to an intra-chunk
    # `tl.cumsum` of the [T, N/2] tile it already loads.
    # SSD_MIMO_PHASE_INKERNEL=0 restores the host cumsum (the A/B lever).
    phase_in_kernel = os.environ.get("SSD_MIMO_PHASE_INKERNEL", "1") == "1"
    if os.environ.get("SSD_MIMO_M3_EQ_PHASE_LAW", "0") == "1":
        # Match the mixer's fused law: already cumulative, fp32 retained.
        # Do not integrate twice or create the optional low-precision twin.
        from orbit_ssm.kernels.ssd_mimo.m3_epilogue import phase_law
        phase = phase_law(t["Angles"], t["DT"], None, heads, N // 2,
                          sign=-1.0)
        phase_in_kernel = False
    else:
        phase = _m3_eq_phase(t, N, cumsum=not phase_in_kernel)  # [B,L,H,N/2]
    # Mamba-3's PER-HEAD Q_bias / K_bias go in as beta / gamma with
    # `bias_pre_rot` -- the kernel then folds them into the RAW B/C tile
    # and rotates WITH them, which is Mamba-3's own fold order.  B and C
    # stay head-shared: nothing per-head is materialised.
    kb, qb = t["K_bias"], t["Q_bias"]
    beta = kb.float().unsqueeze(-1).contiguous()               # [H,N,1]
    gamma = qb.float().unsqueeze(-1).contiguous()              # [H,N,1]

    # ── K11: the trapezoid runs IN the kernel ────────────────────────
    # `trap_lam` makes `delta_w` Mamba-3's RAW dt and moves BOTH taps
    # into the scan: the collapsed write scale and the rank-1 output
    # correction `y_t -= dt_{t+1}(1-l_{t+1})(k_t.q_t)v_t`.  The host map
    # this replaces was 23 of the arm's 37 forward launches and 72 of its
    # 132 fwd+bwd -- the single largest term on both passes.
    # SSD_MIMO_TRAP_INKERNEL=0 restores the host form (the A/B lever).
    in_kernel = (not trap_off
                 and os.environ.get("SSD_MIMO_TRAP_INKERNEL", "1") == "1")
    lam = None
    if in_kernel:
        write, nxt = t["DT"], None
        lam = torch.sigmoid(t["Trap"])
    else:
        write, nxt = _m3_eq_write(t, trap_off)

    # ── K12: V / B / C go in as VIEWS ────────────────────────────────
    # The kernels read every operand through passed strides and the
    # backward now allocates dX in the layout the caller handed in, so Mamba-3's own [B, L, H, P] argument order
    # needs no transpose copy on either pass -- which is the one thing it
    # got for free that this arm was paying for.
    # SSD_MIMO_STRIDED_V=0 restores the `.contiguous()` (the A/B lever).
    strided_v = os.environ.get("SSD_MIMO_STRIDED_V", "1") == "1"
    v_in = v.permute(0, 2, 3, 1).unsqueeze(3)                  # [B,H,P,1,L]
    if not strided_v:
        v_in = v_in.contiguous()
    y, _ = ssd_mimo_scan(
        v_in,
        (-adt).contiguous(),                                   # [B,H,L]
        k_tok[:, :, 0].contiguous(), q_tok[:, :, 0].contiguous(),
        torch.zeros(heads, device=v.device),                   # a = -1
        chunk_size=chunk, precision=precision,
        delta_w=write.contiguous(),
        trap_lam=lam.contiguous() if lam is not None else None,
        theta=phase, theta_increment=phase_in_kernel,
        beta=beta, gamma=gamma, bias_pre_rot=True,
        rot_where=rot_where,
        # SPEED-ONLY variant (not the Mamba-3 function): the production
        # mixer's static per-head read offset, so its overhead is measured.
        **(dict(theta_h=_m3_eq_theta_h(heads, N, v.device))
           if os.environ.get("SSD_MIMO_M3_EQ_THETA_H", "0") == "1" else {}))
    y = y.permute(0, 3, 1, 2)                                  # [B,L,H,P]
    if not trap_off and not in_kernel:
        # the rank-1 trapezoid correction: k_t . q_t with the BIASED maps
        # (the rotation cancels at equal t), expanded so the [B,L,H,N]
        # per-head map is never built.
        kk, qq = k_tok[:, :, 0].float(), q_tok[:, :, 0].float()
        kbf, qbf = kb.float(), qb.float()
        dot = ((kk * qq).sum(-1)[:, :, None]
               + kk @ qbf.t() + qq @ kbf.t()
               + (kbf * qbf).sum(-1))                          # [B,L,H]
        y = y - (nxt.permute(0, 2, 1) * dot).unsqueeze(-1) * v
    return y


_M3_EQ_MAPS = {"fold": _m3_eq_ours_fold, "head": _m3_eq_ours_head}


def m3_eq_bias_for(themap: str) -> str:
    """The bias structure each map runs (see the `bias` knob block).

    BOTH maps run Mamba-3's own per-head Q_bias / K_bias since K10; the
    difference is only WHERE it is folded.
    """
    return "perhead"


def build_m3_eq(B: int, H: int, P: int, N: int, L: int, *, arm: str,
                chunk: int = 64, angles: str = "m3", trap_off: bool = True,
                dtype=torch.float32, precision: str = "tf32",
                rot_where=None, device="cuda", seed=0, themap: str = "head"
                ) -> tuple[Callable[[], Any], list]:
    """One arm of the equivalent-config pair (`arm` = "ours" | "m3").

    `themap` picks WHERE the per-(token, head) phase is carried on our
    side: "head" (default, K10) hands it to the kernel's per-head `theta`
    slot; "fold" is K9's head-shared slot plus the B*H batch re-layout.
    The Mamba-3 arm is identical under both -- only our map moves.
    """
    if themap not in _M3_EQ_MAPS:
        raise ValueError(f"themap must be head|fold, got {themap!r}")
    ka = m3_eq_angle_pairs(N, angles)
    t = _m3_eq_leaves(B, H, P, N, L, angle_pairs=ka, trap_off=trap_off,
                      dtype=dtype, device=device, seed=seed,
                      bias=m3_eq_bias_for(themap))
    leaves = [t[k] for k in _M3_EQ_ORDER]
    if arm == "m3":
        fn = M3_SISO.load()
        kb, qb = t["K_bias"], t["Q_bias"]
        m3_chunk = 64 if chunk == "auto" else chunk      # official Mamba-3's own default; "auto" is OUR production chunk rule

        def call():
            return fn(Q=t["Q"], K=t["K"], V=t["V"], ADT=t["ADT"],
                      DT=t["DT"], Trap=t["Trap"], Q_bias=qb,
                      K_bias=kb, Angles=t["Angles"], D=None,
                      Z=None, chunk_size=m3_chunk, Input_States=None)
    elif arm == "ours":
        ours = _M3_EQ_MAPS[themap]

        from orbit_ssm.kernels.ssd_mimo import levers as _levers
        # the mixer's SISO_ROT_KA declaration: the angle tail is zero-padded
        _decl = ka if 4 * ka == N else None

        def _chunk():
            # chunk="auto": the production chunk rule at kernel level -- the calibrated structural route's T for
            # this call (grad / no-grad, theta_h group), 64 (the SISO schedule) when nothing is calibrated
            if chunk != "auto":
                return chunk
            from orbit_ssm.kernels.ssd_mimo import rot_route as _rr
            if _rr.explicit_siso() or _rr.explicit_rot_bwd() or _rr.explicit_chunk():
                return int(os.environ.get("SSD_MIMO_CHUNK") or 64)
            nb = torch.is_grad_enabled() and any(x.requires_grad for x in leaves)
            if dtype == torch.float32 and precision != "bf16_ops":
                # fp32 IO: the GPU's fp32 route rule (rot_route `*_fp32`); without one, the general route's chunk rule
                # (what the module does for an fp32 rotating call: `_scan_chunk`)
                r = _rr.pick(B * H, L, device, nb, os.environ.get("SSD_MIMO_M3_EQ_THETA_H") == "1", dtype="fp32")
                if r is None:
                    from orbit_ssm.ssm.ssd_mimo import _scan_chunk
                    return _scan_chunk(1, P, 1, nb)
                return r[1]
            r = _rr.pick(B * H, L, device, nb, os.environ.get("SSD_MIMO_M3_EQ_THETA_H") == "1")
            return 64 if r is None else r[1]

        def call():
            with _levers.declare_active_pairs(_decl):
                return ours(t, N=N, chunk=_chunk(), precision=precision,
                            trap_off=trap_off, rot_where=rot_where)
    else:
        raise ValueError(f"arm must be ours|m3, got {arm!r}")
    return call, leaves


# The equality gate's tolerances are the ones the parity test already
# carries, and they are `mamba3_siso_combined`'s OWN distance to fp64
# (9.6e-3 at fp32 on a small cell, where the map sits 2.1e-7), not a
# statement about either kernel: the M3 parity test derives it in its
# tolerance block.
M3_EQ_TOL = dict()
M3_EQ_TOL[torch.float32] = (4e-2, 1.2e-1)
M3_EQ_TOL[torch.bfloat16] = (1.2e-1, 2.5e-1)


def m3_eq_check(B, H, P, N, L, chunk=64, angles="m3", trap_off=True,
                dtype=torch.float32, precision="tf32", rot_where=None,
                device="cuda", seed=0, grads=True, themap="head") -> dict:
    """Assert the two arms compute the same function, before timing.

    Returns the per-quantity relative errors so a grid can record them.
    Raises `AssertionError` when a quantity exceeds the tolerance -- the
    harness turns that into a recorded SKIP, i.e. a cell whose two arms
    disagree is a failure and is never timed.
    """
    call_o, leaves_o = build_m3_eq(
        B, H, P, N, L, arm="ours", chunk=chunk, angles=angles,
        trap_off=trap_off, dtype=dtype, precision=precision,
        rot_where=rot_where, device=device, seed=seed, themap=themap)
    call_m, leaves_m = build_m3_eq(
        B, H, P, N, L, arm="m3", chunk=chunk, angles=angles,
        trap_off=trap_off, dtype=dtype, precision=precision,
        device=device, seed=seed, themap=themap)
    y_o, y_m = call_o(), call_m()
    errs = dict(y=_rel_eq(y_o, y_m))
    if grads:
        gen = torch.Generator(device=device).manual_seed(7)
        dy = torch.randn(y_m.shape, generator=gen, device=device,
                         dtype=torch.float32)
        (y_o.float() * dy).sum().backward()
        (y_m.float() * dy).sum().backward()
        for name, a, b in zip(_M3_EQ_ORDER, leaves_o, leaves_m):
            if trap_off and name == "Trap":
                # sigmoid'(30) is 0 to fp32, so BOTH gradients are zero
                # and the ratio is meaningless; the term itself is what
                # trap_off asserts away, and `y` already pins that.
                continue
            if a.grad is None or b.grad is None:
                continue
            errs["d" + name] = _rel_eq(a.grad, b.grad)
    tol_fwd, tol_grad = M3_EQ_TOL[dtype]
    bad = sorted((k, v) for k, v in errs.items()
                 if v > (tol_fwd if k == "y" else tol_grad))
    if bad:
        detail = ", ".join("%s %.2e" % kv for kv in bad)
        raise AssertionError(
            "ssd_parity_m3_eq: the two arms disagree at B%d H%d P%d N%d "
            "L%d angles=%s trap_off=%s map=%s %s: %s"
            % (B, H, P, N, L, angles, trap_off, themap,
               str(dtype).split(".")[1], detail))
    return errs


def build_m3_mimo(B: int, H: int, P: int, N: int, L: int, *, R: int = 4,
                  chunk: int = 16, rotary_divisor: int = 4,
                  dtype=torch.bfloat16, device="cuda",
                  seed=0) -> tuple[Callable[[], Any], list]:
    """Mamba-3 MIMO (TileLang).  N is the Q/K head width (`hqk`)."""
    fn = M3_MIMO.load()
    g = _rng(seed)
    Q = _r(g, B, L, R, 1, N, device=device, dtype=dtype, requires_grad=True)
    K = _r(g, B, L, R, 1, N, device=device, dtype=dtype, requires_grad=True)
    V = _r(g, B, L, H, P, device=device, dtype=dtype, requires_grad=True)
    DT = F.softplus(_r(g, B, H, L, device=device) - 3.0
                    ).detach().requires_grad_(True)
    ADT = (-DT.detach() * math.log2(math.e)).clone().requires_grad_(True)
    Trap = (torch.rand(B, H, L, generator=g).to(device=device, dtype=dtype)
            * 0.5).requires_grad_(True)
    Qb = _r(g, H, R, N, device=device, requires_grad=True)
    Kb = _r(g, H, R, N, device=device, requires_grad=True)
    MV = _r(g, H, R, P, device=device, requires_grad=True)
    MZ = _r(g, H, R, P, scale=1.0 / R, device=device, requires_grad=True)
    MO = _r(g, H, R, P, scale=1.0 / R, device=device, requires_grad=True)
    Ang = _r(g, B, L, H, N // rotary_divisor, scale=0.3, device=device,
             requires_grad=True)
    Dsk = _r(g, H, device=device, requires_grad=True)
    Z = _r(g, B, L, H, P, device=device, dtype=dtype, requires_grad=True)

    def call():
        return fn(Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap, Q_bias=Qb,
                  K_bias=Kb, MIMO_V=MV, MIMO_Z=MZ, MIMO_Out=MO, Angles=Ang,
                  D=Dsk, Z=Z, chunk_size=chunk,
                  rotary_dim_divisor=rotary_divisor, dtype=dtype,
                  return_state=False)

    return call, [Q, K, V, ADT, DT, Trap, Qb, Kb, MV, MZ, MO, Ang, Dsk, Z]


# ── module-level builders ────────────────────────────────────────────
def build_mamba_layer(variant: str, d_model: int, *, d_state: int,
                      L: int, B: int, d_conv: int = 4, expand: int = 2,
                      headdim: int = 64, ngroups: int = 1,
                      chunk_size: int | None = None, n_layers: int = 1,
                      dtype=torch.float32, device="cuda",
                      seed=0) -> tuple[Callable[[], Any], list, torch.nn.Module]:
    """One official Mamba mixer at (B, L, d_model).

    `variant` is "mamba1" | "mamba2" | "mamba3".  Returns
    (call, leaves, module) so the caller can also record the param count
    the standing param-match rule is checked against.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if variant == "mamba1":
        cls = MAMBA1_LAYER.load()
        mod = cls(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                  use_fast_path=True, layer_idx=0)
    elif variant == "mamba2":
        cls = MAMBA2_LAYER.load()
        mod = cls(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                  headdim=headdim, ngroups=ngroups,
                  chunk_size=256 if chunk_size is None else chunk_size,
                  layer_idx=0)
    elif variant == "mamba3":
        cls = MAMBA3_LAYER.load()
        mod = cls(d_model, d_state=d_state, expand=expand, headdim=headdim,
                  ngroups=ngroups,
                  chunk_size=64 if chunk_size is None else chunk_size,
                  layer_idx=0, n_layer=n_layers)
    else:
        raise ValueError(f"unknown official variant {variant!r}")
    mod = mod.to(device=device, dtype=dtype)
    x = module_input(B, L, d_model, device=device, dtype=dtype,
                     seed=0 if seed is None else seed)

    def call():
        return mod(x)

    return call, [x, *mod.parameters()], mod


def module_input(B: int, L: int, d_model: int, *, device="cuda",
                 dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    """The module-level input x [B, L, d_model], drawn from a FIXED generator
    (seed + 1) rather than the global RNG, so the official arm and our
    copied-weight arm of a pair see the same x whatever was initialised
    before them."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    return torch.randn(B, L, d_model, generator=g).to(
        device=device, dtype=dtype).requires_grad_(True)


if __name__ == "__main__":                          # availability probe
    for row in availability_report():
        mark = "OK  " if row["available"] else "MISS"
        print(f"{mark} {row['name']:<18} {row['entry_point']}")
        if not row["available"]:
            print(f"     {row['reason']}")
