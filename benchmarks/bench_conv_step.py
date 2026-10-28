"""Training-step benchmark of the conv choice:
    python -m benchmarks.bench_conv_step OUT.json [N]
One SSD-MIMO / HybridMamba layer, forward + backward, conv_kernel triton vs causal_conv1d,
everything else identical (same weights, same input).  N steps back to back / N (default 20), inputs cycled, median of 5 interleaved rounds."""
import statistics, sys, json, torch
from orbit_ssm import SSDMimoConfig, SSDMimoSSM, HybridMambaConfig, HybridMambaSSM
dev = "cuda" if torch.cuda.is_available() else "cpu"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
CASES = [  # (name, family, d_model, B, L)
    ("SSD d1024 B8 L2048", "ssd", 1024, 8, 2048), ("SSD d640 B6 L3000", "ssd", 640, 6, 3000), ("SSD d2048 B4 L4096", "ssd", 2048, 4, 4096),
    ("HM d1024 B8 L2048", "hm", 1024, 8, 2048), ("HM d640 B6 L3000", "hm", 640, 6, 3000),
]
if dev == "cpu": CASES = [("tiny", "ssd", 64, 2, 32), ("tiny hm", "hm", 64, 2, 32)]; N = 2
out = {}
for name, fam, d, B, L in CASES:
    for dt in (torch.bfloat16, torch.float32):
        torch.manual_seed(0)
        layers = {}
        for ck in ("triton", "causal_conv1d"):
            ex = {"kernel_options": {"conv_kernel": ck}} if dev == "cuda" else {"backend": "torch"}
            if fam == "ssd": m = SSDMimoSSM(SSDMimoConfig(d_model=d), execution=ex)
            else: m = HybridMambaSSM(HybridMambaConfig(d_model=d, n_heads=max(2, d // 64)), execution=ex)
            layers[ck] = m.to(dev, dt)
        layers["causal_conv1d"].load_state_dict(layers["triton"].state_dict())
        xs = [torch.randn(B, L, d, device=dev, dtype=dt) for _ in range(3)]
        def step(m, x):
            y = m(x); y.float().square().mean().backward()
        def chain(m):
            if dev == "cuda": torch.cuda.synchronize(); a, e = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
            for i in range(N): step(m, xs[i % 3])
            if dev == "cuda": e.record(); torch.cuda.synchronize(); return a.elapsed_time(e) / N
            return 0.0
        for m in layers.values(): chain(m)            # compile / tune
        T = {k: [] for k in layers}
        for r in range(5):
            for k, m in layers.items(): T[k].append(chain(m))
        ms = {k: statistics.median(v) for k, v in T.items()}
        used = getattr(layers["triton"], "conv_kernel_in_use", "?"), getattr(layers["causal_conv1d"], "conv_kernel_in_use", "?")
        out[f"{name}|{str(dt)[6:]}"] = ms
        print(f"{name:22s} {str(dt)[6:]:8s} step ms triton {ms['triton']:.3f}  causal_conv1d {ms['causal_conv1d']:.3f}  ratio {ms['triton']/max(ms['causal_conv1d'],1e-9):.3f}  in_use {used}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
