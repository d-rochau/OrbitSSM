"""Micro-benchmark of the fused gated RMSNorm ``rms_norm(y * silu(z)) * weight``: ours (Triton) vs torch
(``F.rms_norm`` on ``y * silu(z)``) vs mamba_ssm's kernel (skipped when not importable).  GPU only.

    python benchmarks/bench_gated_norm.py [--dtype bf16|fp32|fp16] [--rows 16384 65536] [--widths 1024 2048 4096]
                                          [--check] [--json OUT.json]

Reports forward and forward+backward milliseconds (median of CUDA-event timings) and the effective GB/s of
the forward (read y, z, write out).  ``--check`` first compares each route's output and gradients against
the fp64 formula (normwise relative error), so a fast-but-wrong route is visible in the same table.
Evidence level: kernel micro-benchmark, one device, warm caches; geometry is printed per row.
"""
from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

_DT = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}


def _routes():
    from orbit_ssm.kernels.gated_norm import gated_rms_norm_triton
    routes = {
        "triton": lambda y, w, z, eps: gated_rms_norm_triton(y, w, eps, z),
        "torch": lambda y, w, z, eps: F.rms_norm(y * F.silu(z), w.shape, w.to(y.dtype), eps),
    }
    from orbit_ssm.layers import norms
    fn = norms._resolve_mamba_gated()
    if fn is not None:
        routes["mamba_ssm"] = lambda y, w, z, eps: fn(
            y.reshape(-1, y.shape[-1]), w, None, z=z.reshape(-1, z.shape[-1]), eps=eps,
            norm_before_gate=False).view(y.shape)
    return routes


def _time(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


def _rel(a, r):
    return ((a.double() - r).norm() / r.norm()).item()


def _check(route, y, w, z, eps, dout):
    y, z, w = (t.detach().clone().requires_grad_() for t in (y, z, w))
    out = route(y, w, z, eps)
    out.backward(dout)
    y64, z64, w64 = (t.detach().double().requires_grad_() for t in (y, z, w))
    g = y64 * F.silu(z64)
    ref = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps) * w64
    ref.backward(dout.double())
    return {"out": _rel(out, ref.detach()), "dy": _rel(y.grad, y64.grad), "dz": _rel(z.grad, z64.grad),
            "dw": _rel(w.grad, w64.grad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=_DT)
    ap.add_argument("--rows", type=int, nargs="+", default=[16384, 65536])
    ap.add_argument("--widths", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "bench_gated_norm needs a GPU"
    dt, dev = _DT[args.dtype], "cuda"
    routes = _routes()
    print(f"device {torch.cuda.get_device_name()}  dtype {args.dtype}  routes {', '.join(routes)}")
    rows_out = []
    for M in args.rows:
        for N in args.widths:
            y = torch.randn(M, N, device=dev, dtype=dt)
            z = torch.randn(M, N, device=dev, dtype=dt)
            w = (1 + 0.1 * torch.randn(N, device=dev)).float()
            dout = torch.randn(M, N, device=dev, dtype=dt)
            rec = {"M": M, "N": N, "dtype": args.dtype}
            for name, route in routes.items():
                yy, zz, ww = (t.detach().clone().requires_grad_() for t in (y, z, w))
                with torch.no_grad():
                    fwd = _time(lambda: route(yy, ww, zz, args.eps))
                def step():
                    route(yy, ww, zz, args.eps).backward(dout)
                fb = _time(step)
                gbs = 3 * M * N * y.element_size() / (fwd * 1e-3) / 1e9
                rec[name] = {"fwd_ms": fwd, "fwd_bwd_ms": fb, "fwd_GBps": gbs}
                if args.check:
                    rec[name]["rel_err"] = _check(route, y, w, z, args.eps, dout)
            rows_out.append(rec)
            print(f"M={M:6d} N={N:5d}  " + "  ".join(
                f"{n}: fwd {r['fwd_ms']:.3f} ms ({r['fwd_GBps']:.0f} GB/s) f+b {r['fwd_bwd_ms']:.3f} ms"
                for n, r in rec.items() if isinstance(r, dict)))
            if args.check:
                for n, r in rec.items():
                    if isinstance(r, dict):
                        print("   " + n + " rel err " + " ".join(f"{k}={v:.2e}" for k, v in r["rel_err"].items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows_out, f, indent=1)


if __name__ == "__main__":
    main()
