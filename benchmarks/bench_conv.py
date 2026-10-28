"""Kernel-level microbenchmark of the depthwise causal conv (width 4, SiLU fused): conv_kernel triton vs causal_conv1d vs torch (F.conv1d).
    python -m benchmarks.bench_conv OUT.json
Channel-last x [B, L, C] viewed as [B, C, L] (the layers' layout).  ms = CUDA-event median of 30 iterations after 10 warm-ups; GB/s = the MINIMUM
HBM traffic / time (forward: read x + write y = 2 B C L s; backward: read x and dy, write dx = 3 B C L s; weights negligible), against the
device's memory peak 2 x mem_clock x bus / 8 (GDDR7).  Each cell runs under three L2 states of x (hot / cold / layer, see CTXS)."""
import json, statistics, sys, torch
import torch.nn.functional as F
from triton.runtime import driver

def peak_gbs():
    p = driver.active.utils.get_device_properties(driver.active.get_current_device())
    return 2.0 * p["mem_clock_rate"] * 1e3 * p["mem_bus_width"] / 8 / 1e9

def median_ms(fn, it=30, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)

from orbit_ssm.kernels import conv_triton as CT
from orbit_ssm.kernels.conv_triton import causal_conv_triton
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
W = 4
SHAPES = [("SSD d_inner 2048", 8, 2048, 2048), ("HM d_inner 3072", 8, 2048, 3072), ("large", 16, 4096, 4096), ("small B2 L2048 C2048", 2, 2048, 2048), ("small B4 L512 C1536", 4, 512, 1536), ("held-out B4 L1024 C2560", 4, 1024, 2560), ("held-out B2 L4096 C1024", 2, 4096, 1024), ("held-out B6 L3000 C1280", 6, 3000, 1280)]
CTXS = ("hot", "cold", "layer")      # L2 state of x: hot = the same x re-read every iteration (a 50 MB x sits in the 96 MB L2: the package reaches
                                     # > HBM peak); cold = L2 flushed (256 MB write) before each call; layer = x is the narrow of a slab a GEMM just wrote
flush_buf = torch.empty(256 * 2**20 // 4, device="cuda", dtype=torch.float32)


def timed(fn, prep, it=30, warm=10):
    ts = []
    for k in range(it + warm):
        prep()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        if k >= warm: ts.append(a.elapsed_time(b))
    return statistics.median(ts)


peak = peak_gbs(); res = {"peak_gbs": peak}
print("HBM peak GB/s", round(peak))
for name, B, L, C in SHAPES:
    for dt in (torch.bfloat16, torch.float32):
        es = torch.finfo(dt).bits // 8
        x0 = torch.randn(B, L, C, device="cuda", dtype=dt).transpose(1, 2)
        slab_in = torch.randn(B, L, 512, device="cuda", dtype=dt); slab_w = torch.randn(C + 1024, 512, device="cuda", dtype=dt) * 0.05
        w = (torch.randn(C, W, device="cuda", dtype=dt) * 0.5)
        b = torch.randn(C, device="cuda", dtype=dt)
        dy = torch.randn(B, L, C, device="cuda", dtype=dt).transpose(1, 2)
        impls = {"triton": lambda x: causal_conv_triton(x, w.unsqueeze(1), b, True),
                 "torch": lambda x: F.silu(F.conv1d(x, w.unsqueeze(1), b, padding=W - 1, groups=C)[..., :L])}
        if causal_conv1d_fn is not None:
            impls["causal_conv1d"] = lambda x: causal_conv1d_fn(x, w, b, activation="silu")
        from orbit_ssm.layers.conv import causal_depthwise_conv1d_narrow_cl as narrow
        for impl, fn0 in impls.items():
            for ctx in (CTXS if impl != "torch" else ("hot",)):
                fn = fn0
                if ctx == "layer" and impl != "torch":          # the layers' own route for a narrow of the slab (dx into the slab's buffer)
                    pk = "triton" if impl == "triton" else True
                    fn = lambda x, pk=pk: narrow(x.transpose(1, 2), w, b, "silu", package=pk).transpose(1, 2)
                st = {}
                if ctx == "hot":
                    prep = lambda: None; mk = lambda: x0.detach().requires_grad_(True)
                elif ctx == "cold":
                    prep = lambda: flush_buf.zero_(); mk = lambda: x0.detach().requires_grad_(True)
                else:
                    prep = lambda: None
                    mk = lambda: F.linear(slab_in, slab_w)[..., :C].transpose(1, 2).detach().requires_grad_(True)
                # each timed call gets its own x (layer: freshly written by the GEMM, outside the timed region)
                def run_fwd():
                    def p():
                        prep(); st["x"] = mk()
                    def f():
                        with torch.no_grad(): fn(st["x"])
                    return timed(f, p)
                def run_fb():
                    def p():
                        prep(); st["x"] = mk()
                    def f():
                        y = fn(st["x"]); y.backward(dy)
                    return timed(f, p)
                try:
                    f_, t_ = run_fwd(), run_fb()
                    byt_f, byt_fb = 2 * B * C * L * es, 5 * B * C * L * es
                    r = dict(fwd_ms=f_, fwd_bwd_ms=t_, fwd_gbs=byt_f / f_ / 1e6, fwd_bwd_gbs=byt_fb / t_ / 1e6)
                    r["fwd_pct_peak"] = 100 * r["fwd_gbs"] / peak; r["fwd_bwd_pct_peak"] = 100 * r["fwd_bwd_gbs"] / peak
                except Exception as e:
                    r = dict(error=f"{type(e).__name__}: {str(e)[:160]}")
                res[f"{name}|{str(dt).split('.')[-1]}|{impl}|{ctx}"] = r
                print(name, str(dt).split('.')[-1], impl, ctx, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
