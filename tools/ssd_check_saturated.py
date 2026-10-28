#!/usr/bin/env python
"""SSD-MIMO backward check in the SATURATED decay regime (C7 finding).

`tools/ssd_check.py` runs at init weights, where Delta|A| per token is
~1e-3..0.1 and the decay alpha = exp(-Delta|A|) is ~1.  In training the
per-token decay saturates (alpha ~ 0 on most tokens of some heads); the
decay-rate gradients (dt / a_dyn paths) are then TINY in exact arithmetic
while the terms the kernels sum to form them are O(1).  This tool forces
that regime and scores every precision arm against fp64.

Regime: for the chosen heads the dt and a_dyn biases are set so that
Delta|A| per token is log-spread over [--sat-lo, --sat-hi] across those
heads (token noise from the projections spreads it further, x e^+-1);
the other heads keep their init (Delta|A| ~ 1e-3..0.1).

Two levels, same draws, same rounded inputs:

  * module: y and every parameter gradient of `SSDMimoSSM` (bf16 autocast,
    fp32 params) at precision io / tf32 / ieee vs the fp64 torch route.
  * scan  (--scan): the kernel-level inputs of ONE kernel call are captured
    (fp32 / bf16 values as the kernel saw them) together with the upstream
    gradient of the scan output; each arm's backward is re-run on them and
    scored against an independent fp64 oracle (`unified_ssd_chunked` with
    the host two-tap trapezoid, D-skip and gate) -- this separates the
    scan's own arithmetic from the module's projections.

Per tensor: normwise relative error, signed error along the reference
(<err, ref>/<ref, ref>, positive = inflated), and PER HEAD (saturated vs
unsaturated heads, worst head); for the decay-path gradients also the sign
agreement of the per-head error with the reference.  Bounds follow A7
class (b) as in ssd_check.py; a tensor passes when normwise AND the
worst saturated head are under the bound, and (>= 2 draws) the pooled
signed error is under bound/4.

    python tools/ssd_check_saturated.py --arm ssd_mimo_r2o2_real --scan
    python tools/ssd_check_saturated.py --arm ssd_mimo_r2o2_real --sat half
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from orbit_ssm import SSDMimoConfig


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="ssd_mimo_r2o2_real",
                   help="language-model preset name (not part of this release)")
    p.add_argument("--rot", choices=["arm", "none", "m3"], default="arm",
                   help="override the arm's rotation (m3 = the complex arm's law)")
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--d-inner", type=int, default=2048)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--L", type=int, default=2048)
    p.add_argument("--draws", type=int, default=1)
    p.add_argument("--seed", type=int, default=22091)
    p.add_argument("--precisions", default="io,tf32,ieee")
    p.add_argument("--sat", default="all",
                   help="all | half (even heads) | none | comma list of heads")
    p.add_argument("--sat-lo", type=float, default=1.0)
    p.add_argument("--sat-hi", type=float, default=50.0)
    p.add_argument("--scan", action="store_true",
                   help="also run the scan-level fp64 oracle comparison")
    p.add_argument("--no-module", action="store_true")
    p.add_argument("--seq-heads", type=int, default=6,
                   help="saturated kernel heads scored against the EXACT "
                        "sequential fp64 oracle (plus 2 unsaturated); 0 = off")
    p.add_argument("--ssm-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--json-out", default="")
    p.add_argument("--expect-gluon90", action="store_true",
                   help="require the opt-in Hopper Gluon forward + Kb1 "
                        "(SSD_MIMO_GLUON90=1 / SSD_MIMO_GLUON90_KB1=1) to "
                        "SERVE every precision='io' kernel call (module and "
                        "scan level); a refusal / fallback is a failure")
    p.add_argument("--perturb", default="",
                   help="comma list applied to the CAPTURED scan call before "
                        "the scan / seq levels (module level unchanged; "
                        "REQUIRES --scan): "
                        "reset (per-token reset tokens c = 30/300/3000 at "
                        "t %% 32 == 0 on every other kernel head), resethead "
                        "(large next-token write: lam -> 1e-3, delta_w x1e3 "
                        "on every 7th token of every other head, nxt/own ~1e6), "
                        "large_phase (phase increments x1e3..1e5: running "
                        "|Theta| 1e2..1e6 rad)")
    a = p.parse_args()
    if a.perturb and not a.scan:
        # the perturbation edits the CAPTURED scan call, which only the
        # scan / seq levels read: without --scan it was a silent no-op
        p.error("--perturb requires --scan (it perturbs the captured scan call)")
    return a


# ── the regime ─────────────────────────────────────────────────────────────

def _sat_heads(spec, H):
    if spec == "all":
        return list(range(H))
    if spec == "none":
        return []
    if spec == "half":
        return list(range(0, H, 2))
    return [int(v) for v in spec.split(",") if v != ""]


def force_regime(m, heads, lo, hi):
    """Set dt / a_dyn biases so Delta|A| ~ c_h, c_h log-spread in [lo, hi]."""
    import torch
    H, R = m.H, m.R
    targets = {}
    if not heads:
        return targets
    n = len(heads)
    with torch.no_grad():
        for i, h in enumerate(heads):
            c = lo * (hi / lo) ** (i / max(n - 1, 1))
            s = math.sqrt(c)                       # Delta ~ s, |A| ~ s
            a_bias = s - 1.0 if s >= 1.0 else 1.0 - 1.0 / s  # heavy tail inverse
            if m.a_act != "heavy_tail":
                a_bias = s + math.log(-math.expm1(-s))
            dt_bias = s + math.log(-math.expm1(-s))  # softplus inverse
            if not m.a_dyn:
                m.A_log[h, :] = math.log(s)
            elif m.a_dyn_proj.bias is not None:
                if m.a_dyn_per_sibling:
                    m.a_dyn_proj.bias[h * R:(h + 1) * R] = a_bias
                else:
                    m.a_dyn_proj.bias[h] = a_bias
            m.dt_proj.bias[h * R:(h + 1) * R] = dt_bias
            targets[h] = c
    return targets


# ── per-head views of each gradient ────────────────────────────────────────

def head_view(name, t, m):
    """[H, -1] view of a module gradient, or None when not per head."""
    H, R, P = m.H, m.R, m.P
    if name in ("dt_proj.weight", "dt_proj.bias", "trap_proj.weight"):
        if t.shape[0] == H * R:
            return t.reshape(H, -1)
        if t.shape[0] == H:
            return t.reshape(H, -1)
    if name in ("a_dyn_proj.weight", "a_dyn_proj.bias", "trap_proj.bias",
                "D", "r_mix", "B_bias", "C_bias", "theta_h"):
        if t.shape[0] == H:
            return t.reshape(H, -1)
        if t.shape[0] == H * R:
            return t.reshape(H, -1)
    if name in ("dx", "dz"):                      # [B, L, H*P]
        return t.reshape(t.shape[0], t.shape[1], H, -1).permute(2, 0, 1, 3) \
                .reshape(H, -1)
    return None


def score(ga, gb, hv=None, sat=None):
    """normwise, signed, per-head worst (sat / unsat) + sign agreement."""
    import torch
    ga, gb = ga.double(), gb.double()
    d = ga - gb
    out = dict(err=float(d.norm() / gb.norm().clamp_min(1e-300)),
               signed=float((d * gb).sum() / (gb * gb).sum().clamp_min(1e-300)),
               ref_norm=float(gb.norm()), finite=bool(torch.isfinite(ga).all()))
    if hv is not None:
        A, Bv = hv(ga), hv(gb)
        D = A - Bv
        per = (D.norm(dim=1) / Bv.norm(dim=1).clamp_min(1e-300)).tolist()
        sg = ((D * Bv).sum(1) / (Bv * Bv).sum(1).clamp_min(1e-300)).tolist()
        out["per_head"] = per
        out["per_head_signed"] = sg
        out["per_head_ref_norm"] = Bv.norm(dim=1).tolist()
        out["per_head_err_norm"] = D.norm(dim=1).tolist()
        sat = set(sat or [])
        hs = [h for h in range(len(per)) if h in sat]
        hu = [h for h in range(len(per)) if h not in sat]
        if hs:
            w = max(hs, key=lambda h: per[h])
            out["sat_worst"] = (w, per[w], sg[w])
            # sign bias over saturated heads: fraction whose error points
            # ALONG the reference (inflates it)
            out["sat_frac_inflating"] = sum(sg[h] > 0 for h in hs) / len(hs)
        if hu:
            w = max(hu, key=lambda h: per[h])
            out["unsat_worst"] = (w, per[w], sg[w])
    return out


# ── the scan-level fp64 oracle ─────────────────────────────────────────────

def fold_bc(kw, Bt, Ct, beta, gamma, theta, theta_h, Hk, heads=None):
    """Per-kernel-head effective maps B~ [B,L,K,N,R], C~ [B,L,K,N,O].

    The kernel's fold (reference.unified_ssd_chunked, bias_pre_rot):
    B~ = R(Theta) (B + beta), C~ = R(theta_h) R(Theta) (C + gamma), with
    Theta the RUNNING phase (cumsum of the increments when
    theta_increment).  Differentiable in every argument.
    """
    import torch
    from orbit_ssm.kernels.ssd_mimo.reference import _rot_fold_pairs
    hk = (torch.arange(Hk, device=Bt.device) if heads is None
          else torch.as_tensor(heads, device=Bt.device))
    Bv = Bt if Bt.dim() == 5 else Bt.unsqueeze(-1)                # [B,L,Gb,N,R]
    Cv = Ct if Ct.dim() == 5 else Ct.unsqueeze(-1)                # [B,L,Gc,N,O]
    Bh = Bv[:, :, hk // (Hk // Bv.shape[2])]
    Ch = Cv[:, :, hk // (Hk // Cv.shape[2])]
    pre = bool(kw.get("bias_pre_rot", True))
    ph = kw.get("pair_layout", "interleaved") == "half"
    if not pre and theta is not None:
        raise NotImplementedError("oracle: bias_pre_rot=False with a phase")
    if beta is not None:
        Bh = Bh + beta[hk][None, None]
    if gamma is not None:
        Ch = Ch + gamma[hk][None, None]
    if theta is not None:
        th = theta.cumsum(dim=1) if kw.get("theta_increment") else theta
        th = th[:, :, hk] if th.dim() == 4 else th[:, :, None].expand(
            -1, -1, len(hk), -1)                                     # [B,L,K,N/2]
        # _rot_fold_pairs folds the LAST axis: move N last per r / o
        Bh = _rot_fold_pairs(Bh.transpose(-1, -2), th[..., None, :], ph).transpose(-1, -2)
        Ch = _rot_fold_pairs(Ch.transpose(-1, -2), th[..., None, :], ph).transpose(-1, -2)
    if theta_h is not None:
        Ch = _rot_fold_pairs(Ch.transpose(-1, -2),
                             theta_h[hk][None, None, :, None, :], ph).transpose(-1, -2)
    return Bh, Ch


def scan_oracle(cap, dY):
    """fp64 y and grads of the captured scan call (host trapezoid form)."""
    import torch
    import torch.nn.functional as F
    from orbit_ssm.kernels.ssd_mimo.reference import unified_ssd_chunked
    f = torch.float64
    a = cap["args"]
    kw = cap["kw"]
    leaves = {}

    def leaf(name, t):
        if t is None:
            return None
        v = t.detach().to(f).requires_grad_(True)
        leaves[name] = v
        return v
    x = leaf("x", a[0]); delta = leaf("delta", a[1])
    Bt = leaf("B", a[2]); Ct = leaf("C", a[3]); alog = leaf("a_log", a[4])
    dt = leaf("delta_w", kw.get("delta_w"))
    lam = leaf("trap_lam", kw.get("trap_lam"))
    beta = leaf("beta", kw.get("beta")); gamma = leaf("gamma", kw.get("gamma"))
    gz = leaf("gate_z", kw.get("gate_z")); dsk = leaf("d_skip", kw.get("d_skip"))
    theta = leaf("theta", kw.get("theta"))
    theta_h = leaf("theta_h", kw.get("theta_h"))
    if kw.get("rot_bias") is not None:
        raise NotImplementedError("scan oracle: rot_bias (retired)")
    if kw.get("x0") is not None:
        raise NotImplementedError("scan oracle: x0")
    # y_token_major changes the output's MEMORY only; the returned shape is
    # [B, H, (O,) G, L] either way (autograd.py), so the oracle ignores it.
    Bsz, H, G, R, L = x.shape
    if lam is not None:
        nxt = F.pad((dt * (1 - lam))[..., 1:], (0, 1))
        write = lam * dt + nxt
    else:
        nxt, write = None, dt
    Bh, Ch = fold_bc(kw, Bt, Ct, beta, gamma, theta, theta_h, H)  # [B,L,H,N,*]
    O = Ch.shape[-1]
    y, s = unified_ssd_chunked(x, delta, Bh, Ch if O > 1 else Ch[..., 0], alog,
                               delta_w=write, bc_per_head=True, chunk_size=64)
    if O == 1 and y.dim() == 4:
        y = y.unsqueeze(2)                                    # [B,H,1,G,L]
    if nxt is not None:
        dot = torch.einsum("blhno,blhnr->bhlor", Ch, Bh)      # [B,H,L,O,R]
        corr = torch.einsum("bhlor,bhgrl->bhogl", dot, x)
        y = y - nxt[:, :, None, None, :] * corr
    if dsk is not None:
        y = y + torch.einsum("hgr,bhgrl->bhgl", dsk, x)[:, :, None]
    if gz is not None:
        z = gz.view(Bsz, L, H, G).permute(0, 2, 3, 1)[:, :, None]
        y = y * F.silu(z)
    if O == 1:
        y = y[:, :, 0]
    grads = torch.autograd.grad(y, list(leaves.values()), dY.to(f),
                                allow_unused=True)
    return y.detach(), dict(zip(leaves.keys(), grads))


def seq_oracle(cap, dY, heads):
    """EXACT fp64 oracle for the decay-path gradients of a head subset.

    The sequential recurrence g_t = alpha_t g_{t-1} + w_t sum_r x_r B_r^T
    differentiated by autograd gives d delta_t = a alpha_t <dg_t, g_{t-1}>:
    a product with alpha_t as a FACTOR, so its relative accuracy does not
    degrade as alpha_t -> 0 (the chunked form's does: it recovers the same
    number as a difference of O(1) terms).  Heads are independent in the
    scan, so a subset is exact for every per-head input gradient
    (delta, delta_w, trap_lam, a_log, x).
    """
    import torch
    import torch.nn.functional as F
    f = torch.float64
    a, kw = cap["args"], cap["kw"]
    hk = torch.tensor(heads, device=a[0].device)
    x = a[0][:, hk].detach().to(f).requires_grad_(True)           # [B,K,G,R,L]
    delta = a[1][:, hk].detach().to(f).requires_grad_(True)       # [B,K,L]
    alog = a[4][hk].detach().to(f).requires_grad_(True)           # [K]
    dt = (kw["delta_w"][:, hk].detach().to(f).requires_grad_(True)
          if kw.get("delta_w") is not None else delta)
    lam = (kw["trap_lam"][:, hk].detach().to(f).requires_grad_(True)
           if kw.get("trap_lam") is not None else None)
    Hk = a[0].shape[1]
    _f = lambda t: None if t is None else t.to(f)
    Bh, Ch = fold_bc(kw, a[2].to(f), a[3].to(f), _f(kw.get("beta")),
                     _f(kw.get("gamma")), _f(kw.get("theta")),
                     _f(kw.get("theta_h")), Hk, heads)            # [B,L,K,N,*]
    Bsz, K, G, R, L = x.shape
    O = Ch.shape[-1]
    if lam is not None:
        nxt = F.pad((dt * (1 - lam))[..., 1:], (0, 1))
        w = lam * dt + nxt
    else:
        nxt, w = None, dt
    alpha = torch.exp(-torch.exp(alog)[None, :, None] * delta)   # [B,K,L]
    g = torch.zeros(Bsz, K, G, Bh.shape[3], dtype=f, device=x.device)
    ys = []
    for t in range(L):
        g = alpha[:, :, t, None, None] * g + w[:, :, t, None, None] * torch.einsum(
            "bkgr,bknr->bkgn", x[..., t], Bh[:, t])
        ys.append(torch.einsum("bkgn,bkno->bkog", g, Ch[:, t]))
    y = torch.stack(ys, -1)                                       # [B,K,O,G,L]
    if nxt is not None:
        dot = torch.einsum("blkno,blknr->bklor", Ch, Bh)
        y = y - nxt[:, :, None, None, :] * torch.einsum("bklor,bkgrl->bkogl", dot, x)
    if kw.get("d_skip") is not None:
        y = y + torch.einsum("kgr,bkgrl->bkgl", kw["d_skip"][hk].to(f), x)[:, :, None]
    if kw.get("gate_z") is not None:
        z = kw["gate_z"].to(f).view(Bsz, L, Hk, G)[:, :, hk].permute(0, 2, 3, 1)
        y = y * F.silu(z)[:, :, None]
    dYk = dY[:, hk].to(f)
    if dYk.dim() == 4:
        dYk = dYk.unsqueeze(2)
    leaves = dict(x=x, delta=delta, a_log=alog)
    if kw.get("delta_w") is not None:
        leaves["delta_w"] = dt
    if lam is not None:
        leaves["trap_lam"] = lam
    grads = torch.autograd.grad(y, list(leaves.values()), dYk)
    out = dict(zip(leaves.keys(), grads))
    # a_log's CONDITIONING: L sees a_log and delta only through A * delta,
    # so d a_log = sum_(b,t) delta_t * d delta_t -- a signed sum.  Its
    # componentwise error scale is sum |delta * d delta| (what a per-token
    # relative error u in d delta can move it by); `a_log|cond` scores the
    # kernel's a_log error against THAT, per head.
    out["_a_log_scale"] = (delta.detach() * out["delta"]).abs().sum(dim=(0, 2))
    return out


# ── theta_freq / theta_proj conditioning (theta_freq) ─────────────
#
# The phase law Theta_t[k] = sign pi tanh(thf_k) sum_{s<=t} tanh(Ang_s[k]) dt_s
# is linear in tanh(thf_k) per pair, so
#     d thf_k = sech^2(thf_k) / tanh(thf_k) * sum_(b,t,hk) Theta_t[k] g_t[k]
# with g = dL/dTheta (the scan's per-token phase gradient).  In exact
# arithmetic sum_t g_t = 0 per (b, hk, k) (a constant phase shift per pair
# is invisible: B~ and C~ rotate together), so this sum of terms of size
# |Theta_t g_t| (|Theta| up to 1e4..1e5 rad trained) returns a number
# 1e3..1e5 times smaller than its terms.  A per-token relative error e in
# g -- which every 16-bit route has at e ~ u16 -- moves it by up to
# e * sum |Theta g|.  `theta_freq|cond` scores the error against THAT
# (per pair, the a_log|cond analogue); `theta_proj.weight|cond` against
# sum_s |d incr_s / d Ang_s| (sum_{t>=s} |g_t|) |u_s|, the same per-token
# error carried through the law adjoint's suffix sum and the projection;
# `theta|gauge` (scan level) gates the shift identity itself:
# ||sum_t g_t|| / ||sum_t |g_t|||, which a wrong term in either the write-
# or the read-side phase gradient breaks at O(1).

def law_phase64(ang, dt, theta_freq, H, K, sign):
    """fp64 UNWRAPPED running phase [B, L, H_k, K] of `m3_epilogue.phase_law`."""
    import torch
    import torch.nn.functional as F
    Bsz, L, HA, ka = ang.shape
    Hk = dt.shape[1]
    om = F.pad(torch.tanh(theta_freq.double()), (0, K - ka))
    a = F.pad(torch.tanh(ang.double()), (0, K - ka))
    hk = torch.arange(Hk, device=ang.device)
    a_k = a[:, :, (hk % H) if HA == H else torch.zeros_like(hk)]
    incr = sign * math.pi * om * a_k * dt.double().permute(0, 2, 1)[..., None]
    return incr.cumsum(1)


def theta_cond_scales(m, law, dcum, u_proj):
    """Componentwise error scales of d theta_freq [ka] and d theta_proj.weight
    [HA*ka, d_sel] for a per-token relative error in g = dcum (see above)."""
    import torch
    import torch.nn.functional as F
    ang, dt, H, K, sign = law
    Bsz, L, HA, ka = ang.shape
    Hk = dt.shape[1]
    g = dcum.double()[..., :ka].abs()                              # [B,L,Hk,ka]
    Th = law_phase64(ang, dt, m.theta_freq.detach(), H, K, sign)[..., :ka]
    om = torch.tanh(m.theta_freq.detach().double())[:ka]
    s_thf = (Th.abs() * g).sum((0, 1, 2)) * (1 - om ** 2) / om.abs().clamp_min(1e-30)
    # d incr_s / d Ang_s = sign pi om sech^2(Ang_s) dt_s ; suffix of |g|
    suf = g.flip(1).cumsum(1).flip(1)                              # [B,L,Hk,ka]
    c = (math.pi * om.abs() * (1 - torch.tanh(ang.double()) ** 2))  # [B,L,HA,ka]
    hk = torch.arange(Hk, device=ang.device)
    w = suf * dt.double().permute(0, 2, 1)[..., None]              # [B,L,Hk,ka]
    A = torch.zeros(Bsz, L, HA, ka, dtype=torch.float64, device=ang.device)
    A.index_add_(2, (hk % H) if HA == H else torch.zeros_like(hk), w)
    A = (A * c).reshape(Bsz * L, HA * ka)
    s_w = A.t() @ u_proj.double().reshape(Bsz * L, -1).abs()     # [HA*ka, d_sel]
    return s_thf, s_w


def law_adjoint64(m, law, dcum, u_proj):
    """fp64 (d theta_freq, d theta_proj.weight) of the SAME dcum the route's
    law adjoint received -- isolates the adjoint's own arithmetic (and any
    wrong term in it) from the conditioning of g: `|law` rows."""
    import torch
    ang, dt, H, K, sign = law
    Bsz, L, HA, ka = ang.shape
    a64 = ang.double().requires_grad_(True)
    thf = m.theta_freq.detach().double().requires_grad_(True)
    Th = law_phase64(a64, dt, thf, H, K, sign)
    d_ang, d_thf = torch.autograd.grad(Th, [a64, thf], dcum.double())
    d_w = d_ang.reshape(Bsz * L, HA * ka).t() @ u_proj.double().reshape(Bsz * L, -1)
    return d_thf, d_w


def gauge_score(g_k, g_ref):
    """theta|gauge: ||sum_t (g_k - g_ref)|| / ||sum_t |g_ref|||, g [B,L,Hk,K]."""
    d = (g_k.double() - g_ref.double()).sum(1)
    return dict(err=float(d.norm() / g_ref.double().abs().sum(1).norm().clamp_min(1e-300)),
                signed=0.0, finite=bool(g_k.isfinite().all()))


def perturb_capture(cap, spec, seed=0):
    """In-place regime change of a captured scan call (`--perturb`).

    Every change keeps the call inside its declared domain (delta >= 0,
    0 <= lam <= 1); heads are chosen as every other KERNEL head so the rest
    stay as captured (the unsaturated control).  Returns {kind: heads}."""
    import torch
    a, kw = cap["args"], cap["kw"]
    out = {}
    if not spec:
        return out
    x, delta, alog = a[0], a[1], a[4]
    Bsz, Hk, L = delta.shape
    heads = list(range(0, Hk, 2))
    absA = torch.exp(alog.float())
    for kind in spec.split(","):
        if kind == "reset":
            spikes = (30.0, 300.0, 3000.0)
            t = torch.arange(L, device=delta.device)
            d = delta.clone()
            for i, k in enumerate(heads):
                c = spikes[i % 3]
                pos = (t % 32) == 0
                d[:, k, pos] = c / absA[k]
            a[1] = d.to(delta.dtype)
            out[kind] = heads
        elif kind == "resethead":
            if kw.get("trap_lam") is None:
                continue
            lam = kw["trap_lam"].clone()
            dw = kw["delta_w"].clone()
            t = torch.arange(L, device=delta.device)
            pos = (t % 7) == 3
            for k in heads:
                lam[:, k] = 1e-3
                dw[:, k, pos] = dw[:, k, pos] * 1e3
            kw["trap_lam"], kw["delta_w"] = lam, dw
            out[kind] = heads
        elif kind == "large_phase":
            th = kw.get("theta")
            if th is None:
                continue
            th = th.clone()
            if th.dim() == 4:                    # [B, L, K, N/2] per head
                for i, k in enumerate(heads):
                    th[:, :, k] = th[:, :, k] * (10.0 ** (3 + 2 * (i % 2)))
            else:
                th = th * 1e3
            kw["theta"] = th
            out[kind] = heads
        else:
            raise ValueError(f"--perturb: unknown kind {kind!r}")
    return out


GLUON_ROUTES = []      # (where, precision, fwd route, Kb1 route)


def gluon_note(where, precision):
    """Record which route the opt-in Gluon sm_90 kernels took on the last
    call (only when their module was imported, i.e. a lever is set)."""
    f = sys.modules.get("orbit_ssm.kernels.ssd_mimo.gluon_sm90_fwd")
    k = sys.modules.get("orbit_ssm.kernels.ssd_mimo.gluon_sm90_kb1")
    fr = f.last_route() if f is not None else None
    kr = k.last_route() if k is not None else None
    GLUON_ROUTES.append((where, precision, fr, kr))
    if f is not None:
        f._LAST_ROUTE.clear()
        f._LAST_ROUTE.update(used=False, why="not consulted since the last note")
    if k is not None:
        k._LAST_ROUTE.clear()
        k._LAST_ROUTE.update(used=False, why="not consulted since the last note")
    return fr, kr


def scan_kernel(cap, dY, precision):
    import torch
    from orbit_ssm.kernels.ssd_mimo import autograd as _ua
    a, kw = cap["args"], dict(cap["kw"])
    kw["precision"] = precision
    leaves = {}

    def leaf(name, t):
        if t is None:
            return None
        v = t.detach().clone().requires_grad_(t.dtype.is_floating_point)
        leaves[name] = v
        return v
    args = [leaf(n, t) for n, t in zip(("x", "delta", "B", "C", "a_log"), a)]
    for n in ("delta_w", "trap_lam", "beta", "gamma", "gate_z", "d_skip",
              "theta", "theta_h"):
        if kw.get(n) is not None:
            kw[n] = leaf(n, kw[n])
    y, s = _ua._orig_scan(*args, **kw)
    grads = torch.autograd.grad(y, list(leaves.values()), dY.to(y.dtype),
                                allow_unused=True)
    gluon_note("scan", precision)
    return y.detach(), dict(zip(leaves.keys(), grads))


def main() -> int:
    a = _parse()
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    import ast
    import torch
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig
    from orbit_ssm.kernels.ssd_mimo import autograd as _ua
    import orbit_ssm.kernels.ssd_mimo.m3_epilogue as _m3e
    try:
        from experiments.lm.model.lm import NAMED_SSM
    except ImportError:
        raise SystemExit("this command needs the language-model presets, which are not part of this release; "
                         "`force_regime` and the other helpers of this module can still be imported")

    dev = torch.cuda.get_device_properties(0)
    ssm_kwargs = dict(NAMED_SSM[a.arm]["kwargs"])
    law = ssm_kwargs["row_blocks"][0]["decay"]
    if a.rot == "m3":
        # rotated half + real half (half rotated rows, half real rows)
        ssm_kwargs.update(row_blocks=[{"frac": 0.5, "decay": law, "complex": True},
                                {"frac": 0.5, "decay": law, "complex": False}],
                          use_theta_h=True, learn_theta_freq=True)
    elif a.rot == "none":
        ssm_kwargs.update(row_blocks=[{"frac": 1.0, "decay": law, "complex": False}],
                          use_theta_h=False, learn_theta_freq=False,
                          theta_per_head=False)
    for kv in a.ssm_kw:
        k, v = kv.split("=", 1)
        ssm_kwargs[k] = ast.literal_eval(v)
    precs = [p for p in a.precisions.split(",") if p]
    print("SATCHECK_CONFIG", json.dumps(dict(
        gpu=dev.name, cc=list(torch.cuda.get_device_capability(0)),
        torch=torch.__version__, arm=a.arm, rot=a.rot, B=a.B, L=a.L,
        sat=a.sat, sat_range=[a.sat_lo, a.sat_hi], precisions=precs,
        env=a.env, ssm_kw=a.ssm_kw), default=str), flush=True)

    u16 = 2.0 ** -8
    results = {}          # (level, prec, name) -> list of per-draw scores
    fails = []
    t0 = time.time()
    _ua._orig_scan = _ua.ssd_mimo_scan
    for d in range(a.draws):
        seed = a.seed + d
        torch.manual_seed(seed)
        kw = dict(ssm_kwargs)
        ex = {**kw.pop("execution", {}), "backend": "cuda"}
        # the layer is its own scan: `m.scan(...)` is the bare recurrence checked here
        m0 = SSDMimoSSM(SSDMimoConfig(d_model=a.d_model, d_state=a.N,
                        d_inner=a.d_inner,
                        d_conv=NAMED_SSM[a.arm].get("d_conv", 4),
                        **kw), execution=ex).cuda()
        scan_own = lambda mod: [(n, p) for n, p in mod.named_parameters()
                                if n.split(".")[0] not in ("in_proj", "conv1d", "norm", "out_proj")]
        heads = _sat_heads(a.sat, m0.H)
        targets = force_regime(m0, heads, a.sat_lo, a.sat_hi)
        if d == 0:
            print("SATCHECK_TARGETS", json.dumps(targets), flush=True)
        ref = copy.deepcopy(m0).double()
        ref.set_execution(dict(backend="torch"))
        io = torch.bfloat16
        x = torch.randn(a.B, a.L, a.d_inner, device="cuda").to(io)
        uu = torch.randn(a.B, a.L, a.d_model, device="cuda").to(io)
        z = torch.randn(a.B, a.L, m0.d_out, device="cuda").to(io)
        rleaves = [v.double().requires_grad_() for v in (x, uu, z)]
        y_ref = ref.scan(rleaves[0], sel_x=rleaves[1], gate_z=rleaves[2])
        w = torch.randn_like(y_ref)
        want = None
        if not a.no_module:
            want = torch.autograd.grad(y_ref, (*rleaves, *(p for _, p in scan_own(ref))),
                                       w, allow_unused=True)
        names = ("dx", "du", "dz", *(n for n, _ in scan_own(m0)))
        # realised decay per head (fp64 route's own token maps)
        with torch.no_grad():
            parts = ref._proj_split(rleaves[1])
            dtr = torch.nn.functional.softplus(parts["dt"]).view(a.B, a.L, m0.H, m0.R)
            from orbit_ssm.ssm.ssd_mimo import _a_mag_tail_eager
            if m0.a_dyn:
                am = _a_mag_tail_eager(parts["a"], 1e-4)
                dec = dtr * am.view(a.B, a.L, m0.H, -1)
            else:
                dec = dtr * ref.A_log.exp()[None, None, :, :]
            q = torch.quantile(dec.permute(2, 0, 1, 3).reshape(m0.H, -1).float(),
                               torch.tensor([0.05, 0.5, 0.95], device="cuda",
                                            dtype=torch.float32), dim=1)
            if d == 0:
                print("SATCHECK_DECAY Delta|A| per head q05/q50/q95:",
                      " ".join(f"h{h}:{q[0, h]:.2g}/{q[1, h]:.2g}/{q[2, h]:.2g}"
                               for h in range(m0.H)), flush=True)
        del y_ref
        cap_store = {}
        for prec in precs:
            m = copy.deepcopy(m0)
            m.set_execution(dict(precision=prec, backend="cuda"))
            leaves = [v.clone().requires_grad_() for v in (x, uu, z)]
            captured = []

            def _wrap(*args, **kw):
                y_, s_ = _ua._orig_scan(*args, **kw)
                if kw.get("chunk_mode") != "measure":
                    captured.clear()
                    rec = dict(args=[t.detach().clone() for t in args],
                               kw={k: (v.detach().clone()
                                       if isinstance(v, torch.Tensor) else v)
                                   for k, v in kw.items()}, dY=None)
                    captured.append(rec)
                    if y_.requires_grad:
                        def _hook(g, rec=rec):
                            rec["dY"] = g.detach().clone()
                        y_.register_hook(_hook)
                return y_, s_
            _ua.ssd_mimo_scan = _wrap
            lawrec = {}
            _o_law = _m3e.phase_law
            _o_split = m._proj_split

            def _law(ang, dt, om, H, K, sign=-1.0, lo_dtype=None, **kw):
                # forward the newer layout arguments (ka, omega_layout) unchanged
                th = _o_law(ang, dt, om, H, K, sign=sign, lo_dtype=lo_dtype, **kw)
                t0_ = th[0] if isinstance(th, tuple) else th
                lawrec["law"] = (ang.detach(), dt.detach(), H, K, sign)
                if t0_.requires_grad:
                    t0_.register_hook(lambda g_: lawrec.__setitem__("dcum", g_.detach()))
                return th

            def _split(u_, *a_, **k_):
                lawrec["u"] = u_.detach()
                return _o_split(u_, *a_, **k_)
            if getattr(m, "learn_theta_freq", False):
                _m3e.phase_law = _law
                m._proj_split = _split
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = m.scan(leaves[0], sel_x=leaves[1], gate_z=leaves[2])
                if m.last_route() is not None:
                    fails.append(f"{prec}: route fell off the kernel: {m.last_route()}")
                got = torch.autograd.grad(y, (*leaves, *(p for _, p in scan_own(m))),
                                          w.to(y.dtype), allow_unused=True)
            finally:
                _ua.ssd_mimo_scan = _ua._orig_scan
                _m3e.phase_law = _o_law
                m.__dict__.pop("_proj_split", None)
            torch.cuda.synchronize()
            if d == 0:
                try:
                    from orbit_ssm.kernels.ssd_mimo.fused_bwd_kernel import kb1_dispatch_log
                    kl = kb1_dispatch_log()
                    print(f"SATCHECK_KB1 {prec}: {kl[-1] if kl else None}", flush=True)
                except Exception as e:          # noqa: BLE001 -- report only
                    print(f"SATCHECK_KB1 {prec}: n/a ({e})", flush=True)
                g90 = sys.modules.get("orbit_ssm.kernels.ssd_mimo.gluon_sm90_kb1")
                if g90 is not None:
                    print(f"SATCHECK_GLUON90_KB1 {prec}: {g90.last_route()}",
                          flush=True)
            gluon_note("module", prec)
            if captured and prec == precs[0]:
                cap_store = captured[0]
            if (want is not None and "dcum" in lawrec and "u" in lawrec
                    and getattr(m, "learn_theta_freq", False)):
                s_thf, s_w = theta_cond_scales(m0, lawrec["law"], lawrec["dcum"],
                                               lawrec["u"])
                l_thf, l_w = law_adjoint64(m0, lawrec["law"], lawrec["dcum"],
                                           lawrec["u"])
                for name, ga in zip(names, got):
                    lref = dict(theta_freq=l_thf, **{"theta_proj.weight": l_w}).get(name)
                    if lref is not None and ga is not None and lref.shape == ga.shape:
                        results.setdefault(("module", prec, name + "|law"), []).append(
                            score(ga, lref))
                for name, ga, gb in zip(names, got, want):
                    scl = dict(theta_freq=s_thf).get(name)
                    if name == "theta_proj.weight" and s_w.shape == gb.shape:
                        scl = s_w
                    if scl is None or ga is None:
                        continue
                    dlt = ga.double() - gb.double()
                    # ||error|| / ||scale||: the a_log|cond form
                    results.setdefault(("module", prec, name + "|cond"), []).append(
                        score(scl + dlt, scl))
            del lawrec
            if want is not None:
                for name, ga, gb in zip(names, got, want):
                    if ga is None or gb is None or not gb.abs().max() > 0:
                        continue
                    hv = (lambda t, n=name: head_view(n, t, m0))
                    hv = hv if head_view(name, gb, m0) is not None else None
                    results.setdefault(("module", prec, name), []).append(
                        score(ga, gb, hv, heads))
            del m, got, y
            torch.cuda.empty_cache()
        if a.scan and cap_store and cap_store.get("dY") is not None:
            dY = cap_store["dY"]
            pk = {}
            if a.perturb:
                pk = perturb_capture(cap_store, a.perturb)
                print("SATCHECK_PERTURB", json.dumps(pk), flush=True)
            y_o, g_o = scan_oracle(cap_store, dY)
            Hk = cap_store["args"][0].shape[1]
            H = m0.H
            # kernel heads are r-major: k = r*H + h -> model head h = k % H
            sat_k = [k for k in range(Hk) if (k % H) in set(heads)]
            if a.perturb and pk:
                # the perturbed kernel heads are the ones under test
                sat_k = sorted(set(sat_k) | {k for v in pk.values() for k in v})
            seq_heads = sorted(set([k for k in sat_k][::max(1, len(sat_k) // a.seq_heads)]
                                   [:a.seq_heads]
                                   + [k for k in range(Hk) if k not in sat_k][:2]))
            g_s = seq_oracle(cap_store, dY, seq_heads) if a.seq_heads > 0 else {}
            for prec in precs:
                y_k, g_k = scan_kernel(cap_store, dY, prec)
                for name, gb in g_s.items():
                    if name == "_a_log_scale":
                        continue
                    ga = g_k.get(name)
                    if ga is None:
                        continue
                    if name == "a_log":
                        scl = g_s["_a_log_scale"]
                        dlt = ga[seq_heads].double() - gb.double()
                        sc = score(scl + dlt, scl, lambda t: t.reshape(len(seq_heads), -1),
                                   [i for i, k in enumerate(seq_heads) if k in sat_k])
                        sc["heads"] = seq_heads
                        results.setdefault(("seq", prec, "a_log|cond"), []).append(sc)
                    ga = ga[seq_heads] if name == "a_log" else ga[:, seq_heads]
                    hv = ((lambda t: t.reshape(len(seq_heads), -1)) if name == "a_log"
                          else (lambda t: t.transpose(0, 1).reshape(len(seq_heads), -1)))
                    sat_local = [i for i, k in enumerate(seq_heads) if k in sat_k]
                    sc = score(ga, gb, hv, sat_local)
                    sc["heads"] = seq_heads
                    results.setdefault(("seq", prec, name), []).append(sc)
                if g_k.get("theta") is not None and g_o.get("theta") is not None:
                    results.setdefault(("scan", prec, "theta|gauge"), []).append(
                        gauge_score(g_k["theta"], g_o["theta"]))
                results.setdefault(("scan", prec, "y"), []).append(
                    score(y_k, y_o, lambda t: t.transpose(0, 1).reshape(Hk, -1), sat_k))
                for name, gb in g_o.items():
                    ga = g_k.get(name)
                    if ga is None or gb is None or not gb.abs().max() > 0:
                        continue
                    if name in ("x", "delta", "delta_w", "trap_lam"):
                        hv = lambda t: t.transpose(0, 1).reshape(Hk, -1)
                    elif name in ("beta", "gamma", "d_skip", "a_log"):
                        hv = lambda t: t.reshape(Hk, -1)
                    else:
                        hv = None
                    results.setdefault(("scan", prec, name), []).append(
                        score(ga, gb, hv, sat_k))
                del g_k
            del g_o
        del m0, ref, cap_store
        torch.cuda.empty_cache()

    # ── report ─────────────────────────────────────────────────────────────
    rows = []
    for (lvl, prec, name), lst in sorted(results.items()):
        err = max(s["err"] for s in lst)
        sgn = sum(s["signed"] for s in lst) / len(lst)
        # the |cond / |gauge rows carry the PER-TOKEN bound (depth 64): their
        # scale already is the sum of the per-token magnitudes
        depth = (a.B * a.L if lvl == "module" and name not in ("dx", "du", "dz")
                 and "|" not in name else 64)
        tol = min(max(u16 * math.sqrt(depth) * math.sqrt(2), 0.64 * 2 * u16),
                  6.4 * 2 * u16)
        if name.endswith("|cond") or name.endswith("|gauge"):
            # THETAFREQ: the scale IS the sum of the per-token magnitudes
            # (|cond) / of |g| (|gauge), so the bound is ONE 16-bit unit
            # roundoff per token -- no depth factor.  Measured worst
            # (A100, io): 3.0e-4 |cond (trained fixture), 1.6e-4 gauge.
            tol = u16
        sw = max((s.get("sat_worst", (None, 0.0, 0.0)) for s in lst),
                 key=lambda t: t[1])
        uw = max((s.get("unsat_worst", (None, 0.0, 0.0)) for s in lst),
                 key=lambda t: t[1])
        infl = [s["sat_frac_inflating"] for s in lst if "sat_frac_inflating" in s]
        # per-head gating only against the EXACT oracle ("seq"): the
        # chunked fp64 oracle shares the cancellation (its floor is
        # 1e-16 * O(1), above the truth on the most saturated heads), and
        # the module arms differ from the fp64 route by the bf16 rounding
        # of the projections, which per-head sums with cancellation
        # (trap_proj.bias, ...) amplify.  Both are reported, not gated.
        # a_log's raw per-head error is a signed sum's (reported, gated
        # normwise only); its per-head gate is the conditioned row.
        head_bad = lvl == "seq" and name != "a_log" and ((sw[0] is not None and sw[1] >= tol)
                                     or (uw[0] is not None and uw[1] >= tol))
        # theta_freq: the raw theta_freq / theta_proj.weight rows
        # are gated.  Before the operand-consistent dTheta (common._opv) the
        # in-kernel backward contracted the fp32 rotated maps while the MMAs
        # consumed the rounded ones (0.098 here; 0.022 after).  The |cond /
        # |law / |gauge rows localise a failure.
        bad = (err >= tol or head_bad
               or (a.draws > 1 and abs(sgn) >= tol / 4)
               or not all(s["finite"] for s in lst))
        if bad:
            fails.append(f"{lvl}/{prec}/{name}")
        row = dict(level=lvl, prec=prec, name=name, err=err, signed=sgn,
                   bound=tol, sat_worst=sw, unsat_worst=uw,
                   sat_frac_inflating=(sum(infl) / len(infl)) if infl else None,
                   ok=not bad)
        rows.append(row)
        if lvl == "seq" and lst and "heads" in lst[0]:
            hmap = lst[0]["heads"]
            sw = (hmap[sw[0]] if sw[0] is not None else None,) + tuple(sw[1:])
            uw = (hmap[uw[0]] if uw[0] is not None else None,) + tuple(uw[1:])
            row["sat_worst"], row["unsat_worst"] = sw, uw
            row["per_head"] = {hmap[i]: v for i, v in enumerate(lst[0]["per_head"])}
        swt = (f"sat-worst h{sw[0]} {sw[1]:9.2e} ({sw[2]:+.1e})"
               if sw[0] is not None else "")
        uwt = (f"unsat-worst h{uw[0]} {uw[1]:9.2e}" if uw[0] is not None else "")
        inf = (f"infl {row['sat_frac_inflating']:.2f}"
               if row["sat_frac_inflating"] is not None else "")
        print(f"  {lvl:6s} {prec:5s} {name:22s} err {err:9.2e} signed {sgn:+9.2e} "
              f"bound {tol:8.2e} {swt} {uwt} {inf} {'ok' if not bad else 'FAIL'}",
              flush=True)
    # ── the Gluon sm_90 route: served where it must be ─────────────────
    if GLUON_ROUTES:
        agg = {}
        for where, prec, fr, kr in GLUON_ROUTES:
            e = agg.setdefault((where, prec), dict(n=0, fwd=0, kb1=0, why=set()))
            e["n"] += 1
            e["fwd"] += bool(fr and fr.get("used"))
            e["kb1"] += bool(kr and kr.get("used"))
            for tag, r in (("fwd", fr), ("kb1", kr)):
                if r is not None and not r.get("used"):
                    e["why"].add(f"{tag}: {r.get('why')}")
        for (where, prec), e in sorted(agg.items()):
            print(f"SATCHECK_GLUON90 {where} {prec}: calls {e['n']} fwd served "
                  f"{e['fwd']} kb1 served {e['kb1']}"
                  + (f" refusals {sorted(e['why'])}" if e["why"] else ""),
                  flush=True)
            if a.expect_gluon90 and prec == "io" and (
                    e["fwd"] < e["n"] or e["kb1"] < e["n"]):
                fails.append(f"gluon90/{where}/{prec}: not served "
                             f"(fwd {e['fwd']}/{e['n']}, kb1 {e['kb1']}/{e['n']})")
    elif a.expect_gluon90:
        fails.append("gluon90: the Gluon modules were never imported "
                     "(set SSD_MIMO_GLUON90=1 SSD_MIMO_GLUON90_KB1=1)")
    summary = dict(passed=not fails, seconds=round(time.time() - t0, 1),
                   failures=fails[:40])
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(dict(config=vars(a), rows=rows, summary=summary), fh,
                      default=str, indent=1)
    print("SATCHECK", json.dumps(summary, default=str), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
