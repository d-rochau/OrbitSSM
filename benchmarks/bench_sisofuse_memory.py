"""`SSD_MIMO_SISO_FUSE` — what the lever costs and saves, in MiB.

This is the MEMORY evidence for the SISOFUSE lever.  It lives in
`benchmarks/` and not in `tests/` on purpose.

WHAT MOVED HERE AND WHY.  The kernel test that pins the SISO-fuse route
(it frees the staged maps and takes the general backward)
used to carry a third assertion beside its route and held-memory pins:

    out["fuse_b"]["peak"] - out["fuse_c"]["peak"] > 0.6 * maps_MB

A difference of two PEAKS is not a property of the lever.  `peak` is the
maximum of the allocator's high-water mark over a whole forward, and the
two schedules do not reach their maxima at the same instant, so the
difference is a LOWER BOUND on the saving whose tightness depends on the
allocator, the device, the free-block layout at entry, and the size of
the run.  The cell's own comment recorded exactly that: at 0.5 MiB of
maps the bound read 0.78x the map size and at 2 MiB it read ~1.00x, and
the shipped bar of 0.6x was picked to sit under both.  That is a
CALIBRATED number, i.e. evidence, and evidence goes stale on any
hardware or allocator change.  What the test keeps is the pair of claims
that cannot go stale: WHICH kernels ran, and whether the staged maps are
still HELD across the forward-to-backward window (a saved-tensor count,
not a high-water mark).

THE ARMS (`siso_route.siso_fuse`, default "0"):

    siso     `SSD_MIMO_SISO_FAST=1`, fuse "0" -- the shipped rotating
             SISO route: phase 1 stages the per-head rotated maps and
             SAVES them for its own backward.
    fuse_b   phase 1 still stages the maps, but DROPS them at the end of
             the forward; the general 5-kernel backward rebuilds.
    fuse_c   no phase 1 at all -- the dedicated schedule rotates AT TILE
             LOAD, so the maps are never allocated.

The expected ordering is `held: siso >> fuse_b == fuse_c` and
`peak: siso >= fuse_b > fuse_c`, with the gaps on the order of the map
size, `2 * B * L * H * N * 2` bytes.

    PYTHONPATH=$PWD python -m benchmarks.bench_sisofuse_memory
    PYTHONPATH=$PWD python -m benchmarks.bench_sisofuse_memory --length 2048

READING (NVIDIA RTX PRO 5000 Blackwell; defaults below: B = 2 H = 8 G = 32 N = 64 L = 512 T = 64, maps 2.00 MiB):

    arm        held MiB   peak MiB   siso bwd?   general bwd?
    siso           4.28     260.31        True          False
    fuse_b         1.72     260.28       False           True
    fuse_c         1.72     258.25       False           True

    held:  both fused arms drop 2.56 MiB = 1.28x the map size
    peak:  fuse_c sits 2.03 MiB = 1.02x the map size under fuse_b

The peak figure landing at ~1.00x at 2 MiB of maps is the calibration
the retired test bar of 0.6x was chosen to sit under.
"""
import argparse
import gc
import os

import torch

from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan

MODES = {"siso": ("1", "0"), "fuse_b": ("1", "b"), "fuse_c": ("1", "c")}


def _set_mode(mode):
    fast, fuse = MODES[mode]
    os.environ["SSD_MIMO_SISO_FAST"] = fast
    os.environ["SSD_MIMO_SISO_FUSE"] = fuse


def _inputs(batch, heads, groups, state, length, seed=8890):
    """The rotating R = O = 1 cell the lever is defined on, in the
    kernel's own layout (the layout `ssd_mimo_scan` documents)."""
    torch.manual_seed(seed)
    f32 = dict(device="cuda", dtype=torch.float32)
    bf = dict(device="cuda", dtype=torch.bfloat16)
    core = [
        (torch.randn(batch, heads, groups, 1, length, **bf) * 0.1
         ).requires_grad_(True),
        (torch.rand(batch, heads, length, **f32) * 0.1 + 0.01
         ).requires_grad_(True),
        (torch.randn(batch, length, state, **bf) * 0.1).requires_grad_(True),
        (torch.randn(batch, length, state, **bf) * 0.1).requires_grad_(True),
        (torch.randn(heads, **f32) - 1.0).requires_grad_(True),
    ]
    # the per-head phase, handed over as INCREMENTS (theta_increment=True)
    theta = torch.randn(batch, length, heads, state // 2, **f32) * 0.4
    increments = torch.cat([theta[:, :1], theta[:, 1:] - theta[:, :-1]], 1)
    feats = dict(
        theta=increments.detach().requires_grad_(True),
        theta_h=(torch.randn(heads, state // 2, **f32) * 0.3
                 ).requires_grad_(True),
        beta=(torch.randn(heads, state, 1, **f32) * 0.1).requires_grad_(True),
        gamma=(torch.randn(heads, state, 1, **f32) * 0.1).requires_grad_(True),
    )
    delta_w = (torch.rand(batch, heads, length, **f32) * 0.05 + 0.01
               ).requires_grad_(True)
    lam = torch.sigmoid(torch.randn(batch, heads, length, **f32)
                        ).requires_grad_(True)
    return core, feats, delta_w, lam


def measure(mode, batch, heads, groups, state, length, chunk=64):
    """`held` = allocated after the forward minus before, i.e. the
    per-layer forward-to-backward SAVE.  `peak` = the allocator's
    high-water mark over the same window.  `kern` = every Triton kernel
    the forward AND the backward launched."""
    from triton.compiler import CompiledKernel
    import triton.knobs as knobs

    _set_mode(mode)
    core, feats, delta_w, lam = _inputs(batch, heads, groups, state, length)

    seen, orig = set(), CompiledKernel.launch_metadata
    old_hook = knobs.runtime.launch_enter_hook

    def patched(self, grid, stream, *a):
        seen.add(self.name)
        return orig(self, grid, stream, *a)

    CompiledKernel.launch_metadata = patched
    knobs.runtime.launch_enter_hook = lambda lazy: None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        y, state_out = ssd_mimo_scan(*core, chunk_size=chunk, precision="io",
                                     bias_pre_rot=True, delta_w=delta_w,
                                     trap_lam=lam, theta_increment=True,
                                     **feats)
        torch.cuda.synchronize()
        held = (torch.cuda.memory_allocated() - before) / 2 ** 20
        peak = (torch.cuda.max_memory_allocated() - before) / 2 ** 20
        (y.float().sum() + state_out.float().sum()).backward()
        torch.cuda.synchronize()
        return dict(held=held, peak=peak, kern=set(seen))
    finally:
        CompiledKernel.launch_metadata = orig
        knobs.runtime.launch_enter_hook = old_hook
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--state", type=int, default=64)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a CUDA device")

    maps_MB = (2 * args.batch * args.length * args.heads * args.state * 2
               / 2 ** 20)
    print(f"device   {torch.cuda.get_device_name(0)}")
    print(f"cell     B={args.batch} H={args.heads} G={args.groups} "
          f"N={args.state} L={args.length} T={args.chunk}  R=O=1, rotating")
    print(f"maps     {maps_MB:.2f} MiB  (2 x [B, L, H, N] at 2 bytes)\n")

    siso_bwd = {"_siso_bwd_scan_kernel", "_siso_rot_adjoint_kernel"}
    out = {}
    for mode in MODES:
        out[mode] = measure(mode, args.batch, args.heads, args.groups,
                            args.state, args.length, args.chunk)

    print(f"{'arm':8s} {'held MiB':>10s} {'peak MiB':>10s}  "
          f"{'siso bwd?':>10s}  {'general bwd?':>13s}")
    for mode, row in out.items():
        print(f"{mode:8s} {row['held']:10.2f} {row['peak']:10.2f}  "
              f"{str(bool(row['kern'] & siso_bwd)):>10s}  "
              f"{str(any(k.startswith('_bwd_k') for k in row['kern'])):>13s}")

    print("\nthe claims this probe is the evidence for")
    for mode in ("fuse_b", "fuse_c"):
        saved = out["siso"]["held"] - out[mode]["held"]
        print(f"  {mode} drops the staged maps from the forward save: "
              f"{saved:+.2f} MiB = {saved / maps_MB:.2f}x the map size")
    gap = out["fuse_b"]["peak"] - out["fuse_c"]["peak"]
    print(f"  fuse_c never allocates them, so its forward PEAK is lower "
          f"than fuse_b's: {gap:+.2f} MiB = {gap / maps_MB:.2f}x the map "
          f"size")
    print("\n  NB the peak DIFFERENCE is a lower bound on the saving, not "
          "the saving:\n  the two schedules do not reach their maxima at "
          "the same instant.  It is\n  smaller at small map sizes -- 0.78x "
          "at 0.5 MiB, ~1.00x at 2 MiB\n  (Blackwell) -- which "
          "is why this is a benchmark and not a\n  contract bar.")


if __name__ == "__main__":
    main()
