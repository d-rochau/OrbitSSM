"""Offline driver of the `ssd_mimo_fused` calibration family: the WORKLOAD a
calibration child times (`tools/calibrate_kernels.py --family ssd_mimo_fused`).

Kept out of the installed package: it builds
its calls from the repo-only `benchmarks` package, so inside
`orbit_ssm.kernels.ssd_mimo.calibration_fused` it made the wheel import a
module the wheel does not ship.  The family's registration, shapes and
config keys stay in the package (`calibration_fused`); only the measurement
driver lives here.  Run from a checkout (repo root and `src` on sys.path).
"""
from orbit_ssm.kernels.ssd_mimo.calibration_fused import _real_r1o1_chunks


def workload(geom, seed=20261003):
    """Callable doing ONE complete fwd+bwd of the geometry through the PRODUCTION entry
    (`ssd_mimo_scan`, the call the mixers make), so the head-blocked backward kernels
    production runs (`_bwd_kbc_hb`, `_bwd_kb2_hb`, `_bwd_kinv`) are exercised.  Real
    groups use the per-head B/C layout (`bc_per_head=True`) of the mixers; rotating groups the
    composite rotation features."""
    import torch
    from benchmarks.bench_kernels import build_ssd
    from benchmarks.harness import make_pass
    # the structure: the chunk RULE (`_scan_chunk`: rank table + wide-W clause)
    from orbit_ssm.ssm.ssd_mimo import _scan_chunk
    R, O, G, N = geom["R"], geom["O"], geom["G"], geom["N"]
    T = 64 if geom.get("hi") else _scan_chunk(R, G, O)
    T_bwd = None
    if not geom["rot"] and R == 1 and O == 1:       # the real R = O = 1 route: calibrated (T_fwd, T_bwd)
        T, T_bwd = _real_r1o1_chunks(G)
        T_bwd = None if T_bwd == T else T_bwd
    torch.manual_seed(seed)
    if geom.get("rk"):
        # a per-rank cell: the PRODUCTION mixer of the LM contract (`bench_kernels.build_rank_mixer`) at the group's T and IO dtype
        from benchmarks.bench_kernels import build_rank_mixer
        call, leaves = build_rank_mixer(geom["rot"], R, geom["H"] * geom["P"], geom["B"], geom["L"], N=N, T=geom["T"],
                                        fp32=geom.get("dtype") == "fp32", seed=seed)
        return make_pass(call, leaves, "fwd_bwd")
    if geom.get("dtype") == "fp32":
        # the fp32 groups (fp32 IO, tf32 operands): the SAME calls as the 16-bit groups -- the fair Mamba-2 block for real,
        # the composite rotation features for complex -- at fp32, at the group's fp32 chunks, through the general route
        # (the SISO route's fp32 kernels are the `ssd_mimo` family's)
        if not geom["rot"] and R == 1 and O == 1:
            from benchmarks.baselines import build_real_fair
            call, leaves = build_real_fair(geom["B"], geom["H"], G, N, geom["L"], arm="ours", chunk=geom["T"],
                                           seed=seed, dtype=torch.float32)
            return make_pass(call, leaves, "fwd_bwd")
        call, leaves = build_ssd(geom["B"], geom["H"], G, R, O, N, geom["L"], chunk=geom["T"],
                                 dtype=torch.float32, precision="tf32", feature="rotation",
                                 seed=seed)
        return make_pass(call, leaves, "fwd_bwd")
    if not geom["rot"] and R == 1 and O == 1:
        # the REAL R = O = 1 group runs the call our Mamba-2-style mixer makes: gate_z, d_skip,
        # out_norm_w, softplus(dt + dt_bias) host tail, strided slab views, token-major y (the
        # same function `benchmarks.baselines.build_real_fair` times against official Mamba-2)
        from benchmarks.baselines import build_real_fair
        call, leaves = build_real_fair(geom["B"], geom["H"], G, N, geom["L"], arm="ours", chunk=T,
                                       seed=seed, chunk_bwd=T_bwd)
        return make_pass(call, leaves, "fwd_bwd")
    if geom["rot"] and R == 1 and O == 1:
        # the rotating R = O = 1 group runs the call the Mamba-3 mixer makes: PER-HEAD phase increments
        # (`rot_per_head`, the in-kernel phase cumsum), pre-rotation per-head beta / gamma, the in-kernel trapezoid, at the
        # group's chunk T through the general route.  The composite `feature="rotation"` below (head-SHARED theta + theta_h
        # + the retired rot_bias) compiles other constexpr variants of the three kernels (ROT_PER_HEAD, HAS_BETA / HAS_GAMMA,
        # HAS_ROT_BIAS) than the ones production launches, so its light sets were tuned on the wrong register budget.
        # R2O2 / R4O4 keep the composite (no production caller at those ranks).
        from benchmarks.baselines import build_m3_eq
        from benchmarks.harness import KERNEL_PRECISION
        call, leaves = build_m3_eq(geom["B"], geom["H"], geom["P"], geom["N"], geom["L"], arm="ours", chunk=T,
                                   angles="m3", trap_off=False, dtype=torch.bfloat16,
                                   precision=KERNEL_PRECISION["bf16"], seed=seed)
        return make_pass(call, leaves, "fwd_bwd")
    feature = "rotation" if geom["rot"] else "bc_per_head"
    call, leaves = build_ssd(geom["B"], geom["H"], G, R, O, N, geom["L"], chunk=T,
                             dtype=torch.bfloat16, precision="io", feature=feature,
                             seed=seed)
    return make_pass(call, leaves, "fwd_bwd")
