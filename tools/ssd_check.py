#!/usr/bin/env python
"""One-configuration SSD-MIMO kernel correctness check -- seconds, not hours.

Runs the fused CUDA kernel for ONE configuration and scores y (training and
no-grad) and EVERY gradient against the chunked torch route in fp64, fed the
same rounded inputs (the fp64 evidence rule).  Bounds follow the operand-precision rule (class (b):
normwise relative, u*sqrt(depth)*sqrt(arms) at the operand precision; >=1
draw, pooled signed-error check under bound/4 when --draws > 1).  Asserts the
kernel route was actually taken (`last_route() is None`).  Exit status 0 = pass.

Use it to reproduce and fix ONE failing configuration instead of running a
suite.  Examples:

    # the LLM cell, complex arm, bf16
    python tools/ssd_check.py --R 2 --O 2 --rot m3
    # the sm_90 half-layout trapezoid-backward family
    python tools/ssd_check.py --R 2 --O 2 --rot m3 --pair-layout half --L 256
    # a nonzero static read offset (theta_h is zero-init)
    python tools/ssd_check.py --R 2 --O 2 --rot m3 --theta-h-std 0.5
    # small and fast, fp32, several draws
    python tools/ssd_check.py --d-inner 256 --N 16 --P 16 --L 128 --dtype fp32 --draws 3
    # force a kernel lever for bisection (repeatable)
    python tools/ssd_check.py --R 2 --O 2 --env SSD_MIMO_FWD_GT=16

Prints one line per checked tensor and a final JSON summary (`SSDCHECK {...}`).

PARAMETER-GRADIENT CHECKS ("tolerances follow the arithmetic").  A parameter gradient is a SUM over tokens,
dp = sum_t g_t (a bias; a weight is sum_t g_t (x) x_t).  Two gates:

 (i) PER-TOKEN: native vs fp64 per-token gradient g_t of the token-map slab
     (the gradient flowing into the slab, captured by a hook on BOTH models),
     per parameter's columns.  Bound = the per-element 16-bit bound of the
     arithmetic, u sqrt(T) sqrt(arms) (floor 0.64 eps, cap 6.4 eps; T = the
     chunk length = the contraction depth of one token's gradient path), with
     NO sum / sqrt(B L) factor.  This is the strict check: a per-token error
     above what the arithmetic predicts is the defect to chase.
(ii) SUM: err(sum_t g~_t) <= (per-token bound) * kappa, with the coherent
     worst case  |sum_t g~_t - sum_t g_t| <= c u sum_t |g_t|
     ==> ||d dp|| / ||dp|| <= c u kappa,  kappa = sum_t |g_t| / |sum_t g_t|
     (bias: ||sum_t |g_t| || / ||sum_t g_t||; weight: ||sum_t |g_t| (x) |x_t| ||_F /
     ||sum_t g_t (x) x_t||_F), computed ONLY from the INDEPENDENT fp64 reference's
     per-token gradients, never from the kernel; kappa >= 1.
When kappa > KAPPA_UNINFORMATIVE (10) the sum bound (per-token bound x kappa) is no
longer a statement about the arithmetic (a bound of 50 % or more), so the SUM check is
marked "uninformative (cancellation-dominated)" in the evidence and NOT gated: that
tensor is held by the per-token gate and the signed-bias check alone, and is listed
in the summary (`uninformative_sum`).  There is deliberately no arbitrary cap.
The older u sqrt(B L) sqrt(arms) (cap 6.4 eps) bound survives as a printed WARN
only.

SIGNED-BIAS CHECKS (the pooled signed error of a
cancellation-dominated sum is noise: C10's a_dyn_proj.bias has per-draw signed errors
of +0.058 .. -0.018 with SE ~ 9e-3 over 8 draws, t = 0.55, while its per-token signed
error is -1.9e-3):
 (1) PER-TOKEN signed error of every slab tensor's captured per-token gradient,
     pooled over the draws, gated at the existing threshold (|mean| >= per-token bound / 4)
     -- the strict detector of a SYSTEMATIC bias, at the level where the kernel
     (not the cancellation) produces it.
 (2) SUM-level signed error: kappa1 <= 2: the existing rule (|pooled mean| >= bound / 4);
     kappa1 > 2 (mean over draws) and >= 8 draws: a t-test -- flag only when
     |mean| > 3 SE (SE = std_draws / sqrt(n)) AND |mean| > |per-token signed mean| x kappa1
     (what a coherent per-token bias would predict for the sum).  With < 8 draws a
     kappa1 > 2 tensor falls back to the existing rule.  The old pooled value that
     would have flagged is always printed as a WARN.  Parameters outside
the token-map slab have no per-token decomposition here: kappa = 1 and no
per-token gate (listed in the evidence).  `--evidence-out F` writes per draw and
tensor: err, signed, per-token err and bound, kappa, sum bound, old bound, warn.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import weakref
from orbit_ssm import SSDMimoConfig



sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from _tolerance import (FLOOR_FACTOR, KAPPA_UNINFORMATIVE, U32, Scorer, accuracy_bound,  # noqa: E402,F401
                        prefix_allowance)  # the shared two-part rule (tests/_tolerance.py)


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--d-inner", type=int, default=2048)
    p.add_argument("--N", type=int, default=128, help="d_state")
    p.add_argument("--P", type=int, default=64, help="head width")
    p.add_argument("--R", type=int, default=2)
    p.add_argument("--O", type=int, default=2)
    p.add_argument("--rot", choices=["none", "m3"], default="m3")
    p.add_argument("--rope-fraction", type=float, default=0.5)
    p.add_argument("--write-rule", choices=["trapezoid", "euler"], default="trapezoid")
    p.add_argument("--pair-layout", choices=["interleaved", "half"], default="interleaved")
    p.add_argument("--proj-source", default="layer_input")
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--bc-conv", action="store_true")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                   help="bf16 = autocast bf16 with fp32 params (training); "
                        "fp32 = fp32 inputs/params")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--draws", type=int, default=1)
    p.add_argument("--seed", type=int, default=22091)
    p.add_argument("--no-grad-only", action="store_true",
                   help="check the forward only (no backward)")
    p.add_argument("--ssm-kw", action="append", default=[], metavar="K=V",
                   help="extra SSDMimoSSM kwarg (python literal), repeatable")
    p.add_argument("--theta-h-std", type=float, default=0.0,
                   help="draw the static read offset theta_h ~ N(0, std^2) "
                        "(it is zero-init, so the default leaves it inert)")
    p.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="set an env lever before import (e.g. SSD_MIMO_FWD_GT=16)")
    p.add_argument("--arm", default="",
                   help="take every SSDMimoSSM kwarg from a named language-model "
                        "preset (not part of this release; overrides "
                        "--R/--O/--rot/--write-rule/--pair-layout/--proj-source/"
                        "--bc-conv); --ssm-kw still applies on top")
    p.add_argument("--expect-gluon90", action="store_true",
                   help="fail unless the opt-in Gluon sm_90 forward served the "
                        "no-grad forward, the training forward AND the "
                        "backward's STATE_ONLY replay (GLUON_SM90_20260923); "
                        "combine with --env SSD_MIMO_GLUON90=1")
    p.add_argument("--expect-gluon90-kb1", action="store_true",
                   help="fail unless the opt-in Gluon sm_90 Kb1 served the "
                        "training backward (combine with --env "
                        "SSD_MIMO_GLUON90_KB1=1)")
    p.add_argument("--expect-gluon90-siso", action="store_true",
                   help="fail unless the opt-in Gluon sm_90 SISO scan adjoint "
                        "served the training backward (combine with --env "
                        "SSD_MIMO_GLUON90_SISO=1 SSD_MIMO_SISO_FAST=1 "
                        "SSD_MIMO_CHUNK=64)")
    p.add_argument("--evidence-out", default="",
                   help="write per-draw per-tensor (err, signed, tol, kappa, tol_eff) as JSON")
    p.add_argument("--launch-blocking", action="store_true",
                   help="CUDA_LAUNCH_BLOCKING=1 to localise illegal accesses")
    return p.parse_args()


from _contract.ssd_capture import _PrefixCapture, _SlabCapture  # noqa: E402,F401  (moved to the contract suite)


def main() -> int:
    a = _parse()
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    if a.launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    import ast
    import torch
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig
    from orbit_ssm.kernels.ssd_mimo.autograd import dispatch_log

    dev = torch.cuda.get_device_properties(0)
    extra = {}
    for kv in a.ssm_kw:
        k, v = kv.split("=", 1)
        extra[k] = ast.literal_eval(v)
    # --rot m3 --rope-fraction f -> a complex block of the leading rows
    # (the pre-spec floor-to-even rule, spelled as explicit row counts so
    # every N builds the same model); --rot none -> default real rows.
    if a.rot == "m3":
        split = int(a.N * a.rope_fraction)
        split -= split % 2
        rows = [{"rows": split, "decay": "head_dynamic", "complex": True}]
        if a.N - split > 0:
            rows.append({"rows": a.N - split, "decay": "head_dynamic",
                         "complex": False})
        rot = dict(row_blocks=rows)
    else:
        rot = {}
    ssm_kwargs = dict(d_ssm_in=a.R, d_ssm_out=a.O, d_head=a.P, proj_source=a.proj_source,
                      bc_conv=a.bc_conv, write_rule=a.write_rule,
                      pair_layout=a.pair_layout, **rot, **extra)
    if a.arm:
        try:
            from experiments.lm.model.lm import NAMED_SSM
        except ImportError:
            raise SystemExit("--arm: the language-model presets are not part of this release; "
                             "set --R/--O/--rot/--ssm-kw instead")
        ssm_kwargs = dict(NAMED_SSM[a.arm]["kwargs"])
        ssm_kwargs.update(extra)
        # the arm's own mixer conv width (C12 is conv-free) beats --d-conv
        a.d_conv = NAMED_SSM[a.arm].get("d_conv", a.d_conv)
    g90 = None
    if a.expect_gluon90 or os.environ.get("SSD_MIMO_GLUON90", "0") in ("1", "on"):
        from orbit_ssm.kernels.ssd_mimo import gluon_sm90_fwd as g90
    bf16 = a.dtype == "bf16"
    # A7 unit roundoff of the kernel's MMA operands: bf16 io operands under
    # bf16 autocast; fp32 inputs run the module's default precision (tf32 or
    # ieee) -- read it from the module below.
    header = dict(gpu=dev.name, sm=dev.multi_processor_count,
                  cc=list(torch.cuda.get_device_capability(0)),
                  torch=torch.__version__, dtype=a.dtype, B=a.B, L=a.L,
                  d_model=a.d_model, d_inner=a.d_inner, ssm_kwargs=ssm_kwargs,
                  theta_h_std=a.theta_h_std,
                  env={kv.split("=")[0]: kv.split("=", 1)[1] for kv in a.env})
    print("SSDCHECK_CONFIG", json.dumps(header, default=str), flush=True)

    fails = []
    scorer = Scorer(a.B, a.L, bf16, a.no_grad_only, fails=fails)
    evidence, uninformative = scorer.evidence, scorer.uninformative
    t0 = time.time()
    rec = {}
    for d in range(a.draws):
        print("SSDCHECK_DRAW", d, flush=True)
        seed = a.seed + d
        torch.manual_seed(seed)
        kw = dict(ssm_kwargs)
        ex = {**kw.pop("execution", {}), "backend": "cuda"}
        # the layer is its own scan: `m.scan(...)` is the bare recurrence checked here
        m = SSDMimoSSM(SSDMimoConfig(d_model=a.d_model, d_state=a.N,
                       d_inner=a.d_inner, d_conv=a.d_conv,
                       **kw), execution=ex).cuda()
        scan_own = lambda mod: [(n, p) for n, p in mod.named_parameters()
                                if n.split(".")[0] not in ("in_proj", "conv1d", "norm", "out_proj")]
        if a.theta_h_std > 0:
            if getattr(m, "theta_h", None) is None:
                fails.append("--theta-h-std: this configuration has no theta_h")
            else:
                with torch.no_grad():
                    m.theta_h.normal_(std=a.theta_h_std)
        if not bf16:
            m = m.float()
        prec = getattr(m, "precision", "tf32")
        u = (2.0 ** -8 if bf16 else
             {"ieee": 2.0 ** -24, "tf32": 2.0 ** -11}.get(prec, 2.0 ** -11))
        ref = copy.deepcopy(m).double()
        ref.set_execution(dict(backend="torch"))
        io = torch.bfloat16 if bf16 else torch.float32
        x = torch.randn(a.B, a.L, a.d_inner, device="cuda").to(io)
        uu = torch.randn(a.B, a.L, a.d_model, device="cuda").to(io)
        z = torch.randn(a.B, a.L, m.d_out, device="cuda").to(io)
        leaves = [v.clone().requires_grad_() for v in (x, uu, z)]
        rleaves = [v.double().requires_grad_() for v in (x, uu, z)]
        ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)

        def served(what, state_only=False, required=True):
            r = m.last_route()
            if r is not None:
                fails.append(f"{what}: route fell off the kernel: {r}")
            if g90 is not None:
                gr = g90.last_route()
                ok = gr.get("used") and bool(gr.get("state_only")) == state_only
                if not ok and a.expect_gluon90 and required:
                    fails.append(f"{what}: gluon90 route not taken: {gr}")
                print(f"  gluon90 {'served' if ok else 'NOT served'} {what}: {gr}",
                      flush=True)
                g90._LAST_ROUTE.clear()
            return r is None

        with torch.no_grad(), ac:
            y_inf = m.scan(x, sel_x=uu, gate_z=z)
        torch.cuda.synchronize()
        # the no-grad forward takes the no-grad chunk rule (T = 64 at R2),
        # outside the Gluon envelope BY DESIGN: reported, never required
        served("no-grad forward", required=False)
        T_inf = dispatch_log()[-1].get("chunk_used")
        # Forward-only checks need no reference tape. Retaining it made
        # the oracle, rather than the native full geometry, exhaust VRAM.
        cap_ctx = _SlabCapture(ref)
        with torch.set_grad_enabled(not a.no_grad_only), cap_ctx:
            y_ref = ref.scan(rleaves[0], sel_x=rleaves[1], gate_z=rleaves[2])
        cells = [("y_nograd", y_inf, y_ref.detach(), T_inf or 64)]
        if not a.no_grad_only:
            # Finish the independent fp64 tape before allocating native
            # training saves/scratch. Keeping both tapes live exceeds a
            # 48-GB device at the full long-sequence fp32 geometry.
            w = torch.randn_like(y_ref)
            want = torch.autograd.grad(y_ref, (*rleaves, *(p for _, p in scan_own(ref))),
                                       w, allow_unused=True)
            kappas = cap_ctx.kappas()
            with ac, cap_ctx.native(m):
                y = m.scan(leaves[0], sel_x=leaves[1], gate_z=leaves[2])
                served("training forward")
                rec = dispatch_log()[-1]
                T = rec.get("chunk_used") or 64
                got = torch.autograd.grad(y, (*leaves, *(p for _, p in scan_own(m))),
                                          w.to(y.dtype), allow_unused=True)
            tok_errs = cap_ctx.token_errors()
            torch.cuda.synchronize()
            if g90 is not None:
                served("backward STATE_ONLY replay", state_only=True)
            if a.expect_gluon90_kb1 or os.environ.get("SSD_MIMO_GLUON90_KB1", "0") in ("1", "on"):
                from orbit_ssm.kernels.ssd_mimo import gluon_sm90_kb1 as g90k
                kr = g90k.last_route()
                if not kr.get("used") and a.expect_gluon90_kb1:
                    fails.append(f"backward: gluon90 Kb1 route not taken: {kr}")
                print(f"  gluon90 Kb1 {'served' if kr.get('used') else 'NOT served'} "
                      f"the backward: {kr}", flush=True)
                g90k._LAST_ROUTE.clear()
            if a.expect_gluon90_siso or os.environ.get("SSD_MIMO_GLUON90_SISO", "0") in ("1", "on"):
                from orbit_ssm.kernels.ssd_mimo import gluon_sm90_siso_bwd as g90s
                sr = g90s.last_route()
                if not sr.get("used") and a.expect_gluon90_siso:
                    fails.append(f"backward: gluon90 SISO route not taken: {sr}")
                print(f"  gluon90 SISO {'served' if sr.get('used') else 'NOT served'} "
                      f"the backward: {sr}", flush=True)
                g90s._LAST_ROUTE.clear()
            names = ("dx", "du", "dz", *(n for n, _ in scan_own(m)))
            cells.append(("y", y.detach(), y_ref.detach(), T))
            for i, (name, ga, gb) in enumerate(zip(names, got, want)):
                if (ga is None) != (gb is None):
                    fails.append(f"{name}: one side has no gradient")
                    continue
                if ga is None or not gb.abs().max() > 0:
                    continue
                cells.append((name, ga, gb, T if i < 3 else a.B * a.L))
        scorer.draw(d, cells, T=T, u=u, kappas=kappas if not a.no_grad_only else {}, tok_errs=tok_errs if not a.no_grad_only else {})
        # Release this draw before constructing the next one. In Python
        # an assignment evaluates its RHS before releasing the old graph.
        del cells, leaves, rleaves, x, uu, z, y_inf, y_ref
        if not a.no_grad_only:
            del y, w, got, want
            ga = gb = None                       # the loop's last gradient pair
        del m, ref
        torch.cuda.empty_cache()
    scorer.finish()
    if a.evidence_out:
        with open(a.evidence_out, "w") as f:
            json.dump(dict(config=header, rows=evidence), f, indent=1, default=str)
    summary = dict(passed=not fails, seconds=round(time.time() - t0, 1),
                   chunk=rec.get("chunk_used"), fwd_stack_guard=rec.get("fwd_stack_guard"),
                   chunk_guard=rec.get("chunk_guard"),
                   arm=a.arm or None, gluon90_expected=bool(a.expect_gluon90),
                   gluon90_kb1_expected=bool(a.expect_gluon90_kb1),
                   gluon90_siso_expected=bool(a.expect_gluon90_siso),
                   uninformative_sum=sorted(uninformative),
                   failures=fails[:20])
    print("SSDCHECK", json.dumps(summary, default=str), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
