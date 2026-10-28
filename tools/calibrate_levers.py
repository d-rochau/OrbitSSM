#!/usr/bin/env python
"""Calibrate the speed-only LEVERS (`src/orbit_ssm/kernels/ssd_mimo/levers.py`,
family `ssd_mimo_levers`) on THIS GPU: eligibility (fp64 stress), per-unit
leave-one-out ablation per phase (fwd / bwd), choice, file.

    python tools/calibrate_levers.py --write-shipped --raw OUT.json
    python tools/calibrate_levers.py                 # LOCAL file (no-op if valid)
    python tools/calibrate_levers.py --from-raw OUT.json --write-shipped

Method
  1. ELIGIBILITY.  The fp64 stress (reset / large_phase / euler_limit x 5
     draws, the exact-function complex R1O1 bf16 check; bound
     2^-8 sqrt(P) sqrt(2)) runs for BASE (every unit off) and ALLON (every
     applicable unit on).  BASE must pass.  If ALLON fails, each unit is
     stressed alone and a failing unit is INELIGIBLE: stored 0, never
     calibratable, whatever its timing.
  2. ABLATION (group complex_R1O1; shapes = the complex SISO calibration
     shapes).  Per shape, one process, interleaved rounds (5 x 20 calls) of
     BASE, ALLON and LOO_u (ALLON without u) for every eligible unit u, on
     the forward alone and on forward+backward; the backward time is
     fwd+bwd minus fwd of the same configuration.  r_s(u, phase) =
     t_phase(LOO_u) / t_phase(ALLON) > 1 means u helps.  A unit is ON iff
     geomean_s r >= 1.015 and min_s r >= 0.97 on ITS phase.
  3. VERIFY.  FINAL (the chosen set) is timed against BASE and ALLON on the
     same shapes; if FINAL is slower than BASE (geomean) on a phase, every
     unit of that phase is turned off.
  4. group complex_ROgt1: the phase-law units are timed in isolation at R = 2
     (their cost does not depend on O); eligibility there is DIFFERENTIAL
     (unit on vs all-off output, rel. diff <= 1e-5).
"""
import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, str(ROOT) + "/src")

GROUP = "complex_R1O1"
TOL_ON, TOL_WORST = 1.015, 0.97
STRESS_PATTERNS = ("reset", "large_phase", "euler_limit")
BASE_ENV = dict(SSD_MIMO_CALIBRATION="off", SSD_MIMO_SISO_FAST="1",
                SSD_MIMO_M3_EQ_PHASE_LAW="1")
UNITS_R1O1 = ["phase_law_fwd_active_k", "phase_law_active_k",
              "phase_law_headpar", "phase_law_suffix", "rot_ka_fwd",
              "rot_ka_bwd", "adj_static_real", "gl_packed_x", "gl_dy_tmajor",
              "gl_dth_store_vec16", "gl_narrow_dth_off", "gl_ws_producer",
              "gl_ws_prep", "gl_split2"]
UNITS_ROGT1 = ["phase_law_fwd_active_k", "phase_law_active_k",
               "phase_law_headpar", "phase_law_suffix"]


def _prep_env():
    for k in [k for k in os.environ if k.startswith("SSD_MIMO_")]:
        del os.environ[k]          # explicit settings would bypass the dispatch
    os.environ.update(BASE_ENV)


def applicable_r1o1():
    from orbit_ssm.kernels.ssd_mimo import gluon_sm90_siso_bwd as g
    import torch
    gl = False
    try:
        gl = bool(g.enabled(torch.device("cuda")))
    except Exception:
        pass
    return [u for u in UNITS_R1O1 if gl or not u.startswith("gl_")]


# ------------------------------------------------------------------- stress --
def stress(units_on):
    """fp64 stress with exactly `units_on` forced on.  Returns dict(passed,
    worst={tensor: [normwise, |signed|]}, failures=[...])."""
    import torch
    from benchmarks import baselines as bl
    from orbit_ssm.kernels.ssd_mimo import levers
    from orbit_ssm.ssm.hybrid_mamba import _m3_reference
    failed, worst = [], {}
    choice = {u: True for u in units_on}
    with levers.forced(choice):
        for pattern in STRESS_PATTERNS:
            for seed in range(5):
                call, leaves = bl.build_m3_eq(
                    1, 4, 64, 128, 1024, arm="ours", chunk=64, angles="m3",
                    trap_off=False, dtype=torch.bfloat16, precision="io",
                    seed=seed)
                t = dict(zip(bl._M3_EQ_ORDER, leaves))
                with torch.no_grad():
                    if pattern == "reset":
                        t["ADT"][:, :, 64] = -40
                    elif pattern == "large_phase":
                        t["DT"].fill_(5)
                        t["ADT"].fill_(-.5)
                    else:
                        t["Trap"].fill_(-12)
                refs = [x.detach().double().requires_grad_() for x in leaves]
                y, r = call(), _m3_reference(*refs)
                dy = torch.randn((y.shape[0], y.shape[2], y.shape[3], y.shape[1]),
                                 device=y.device, dtype=y.dtype,
                                 generator=torch.Generator("cuda").manual_seed(
                                     seed + 71)).permute(0, 3, 1, 2)
                g = torch.autograd.grad(y, leaves, dy)
                rg = torch.autograd.grad(r, refs, dy.double())
                for name, a_, b_ in zip(("y", *bl._M3_EQ_ORDER), (y, *g), (r, *rg)):
                    d = a_.double() - b_
                    n = (d.norm() / b_.norm().clamp_min(1e-300)).item()
                    sg = (d.mean() / b_.square().mean().sqrt().clamp_min(1e-300)).item()
                    w = worst.setdefault(name, [0.0, 0.0])
                    w[0], w[1] = max(w[0], n), max(w[1], abs(sg))
                    if not math.isfinite(n) or n >= 2 ** -8 * math.sqrt(64) * math.sqrt(2):
                        failed.append([pattern, seed, name, n])
    return dict(passed=not failed, worst=worst, failures=failed)


def child_stress(a):
    _prep_env()
    sets = json.loads(a.sets)
    return {k: stress(v) for k, v in sets.items()}


# -------------------------------------------------------------- feasibility --
def child_feasible(a):
    """Greedy maximal set of the Gluon units whose COMBINATION launches on
    this device at every probe shape (shared memory: a combination can need
    more than any single unit).  The launcher's own retry is switched off
    here so the overflow is seen, not repaired."""
    _prep_env()
    import torch
    from benchmarks import baselines as bl
    from benchmarks.harness import make_pass
    from orbit_ssm.kernels.ssd_mimo import calibration_ssd as cs
    from orbit_ssm.kernels.ssd_mimo import levers
    from triton.runtime.errors import OutOfResources
    levers.RETRY_ON_OOR = False
    cands = json.loads(a.sets)               # ordered unit list
    fns = {}
    for s in a.shape.split(","):
        g = cs.SHAPES[s]
        call, leaves = bl.build_m3_eq(g["B"], g["H"], g["P"], g["N"], g["L"],
                                      arm="ours", chunk=64, angles="m3",
                                      trap_off=False, dtype=torch.bfloat16,
                                      precision="io")
        fns[s] = make_pass(call, leaves, "fwd_bwd")

    def fits(units):
        with levers.forced({u: True for u in units}):
            for s, fn in fns.items():
                try:
                    fn()
                    torch.cuda.synchronize()
                except OutOfResources as e:
                    return False, "%s: %s" % (s, str(e).splitlines()[0])
        return True, ""
    keep, excluded = [u for u in cands if not u.startswith("gl_")], {}
    gl = [u for u in cands if u.startswith("gl_")]
    for u in gl:
        ok, why = fits(keep + [u])
        if ok:
            keep.append(u)
        else:
            excluded[u] = why
    return dict(keep=keep, excluded=excluded)


# ----------------------------------------------------------------- ablation --
def child_ablate(a):
    _prep_env()
    import torch
    from benchmarks import baselines as bl
    from benchmarks.harness import make_pass
    from orbit_ssm.kernels.ssd_mimo import calibration_ssd as cs
    from orbit_ssm.kernels.ssd_mimo import levers
    g = cs.SHAPES[a.shape]
    configs = json.loads(a.sets)           # name -> [units on]
    from triton.runtime.errors import OutOfResources
    levers.RETRY_ON_OOR = False            # see the overflow, do not repair it
    torch.manual_seed(20261003)
    try:
        call, leaves = bl.build_m3_eq(g["B"], g["H"], g["P"], g["N"], g["L"],
                                      arm="ours", chunk=64, angles="m3",
                                      trap_off=False, dtype=torch.bfloat16,
                                      precision="io")
        make_pass(call, leaves, "fwd_bwd")()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        return dict(shape=a.shape, oom=True, error=str(e).splitlines()[0][:200])
    fns = {p: make_pass(call, leaves, p) for p in ("fwd", "fwd_bwd")}
    t0 = time.perf_counter()
    infeasible = {}
    for name, units in list(configs.items()):    # compile every variant
        try:
            with levers.forced({u: True for u in units}):
                for fn in fns.values():
                    for _ in range(3):
                        fn()
            torch.cuda.synchronize()
        except OutOfResources as e:              # this combination does not launch
            infeasible[name] = str(e).splitlines()[0]
            del configs[name]
    first = time.perf_counter() - t0
    res = {n: {p: [] for p in fns} for n in configs}
    for r in range(a.rounds):
        order = list(configs) if r % 2 == 0 else list(reversed(list(configs)))
        for name in order:
            with levers.forced({u: True for u in configs[name]}):
                for p, fn in fns.items():
                    ev0, ev1 = (torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True))
                    torch.cuda.synchronize()
                    ev0.record()
                    for _ in range(20):
                        fn()
                    ev1.record()
                    ev1.synchronize()
                    res[name][p].append(ev0.elapsed_time(ev1) / 20)
    med = {n: {p: statistics.median(v) for p, v in d.items()}
           for n, d in res.items()}
    return dict(shape=a.shape, geometry=g, first_call_s=first, rounds=a.rounds,
                median_ms=med, raw_ms=res, infeasible=infeasible)


# --------------------------------------------------------- group ROgt1 / any --
def child_micro(a):
    """Isolated phase-law units (R > 1): timing + differential check."""
    _prep_env()
    import torch
    from orbit_ssm.kernels.ssd_mimo import levers
    from orbit_ssm.kernels.ssd_mimo import m3_epilogue as me
    out = dict(timings={}, oracle={})
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(20261003)
    from orbit_ssm.kernels import shape_sets as ss
    # the shared shape set at R = 2, N = 64 (K = 32 pairs, KA = 16 active); a
    # cell that does not fit is dropped by name
    cells = {t: dict(B=g["B"], L=g["L"], H=g["H"], R=2, N=64, KA=16)
             for t, g in ((t, ss.geometry(t)) for t in ss.SET)}
    if getattr(a, "ranks", "2") != "2":
        # the PER-RANK group complex_R4O4 (R = 4: R*H angle rows): its own cells, `r4_<tag>`, on the LM cells (measurement-time bound,
        # as the fused R4O4 cells); `--ranks 4` measures ONLY those (the R = 2 cells of the shipped group stay as they are)
        cells = {"r4_" + t: dict(B=ss.geometry(t)["B"], L=ss.geometry(t)["L"], H=ss.geometry(t)["H"], R=4, N=64, KA=16) for t in ss.LM_TAGS}
    out["dropped"] = {}
    # NOTE ka*2 = N/2 -> K == 2 KA (the narrow tile's own condition)
    def med(fn, n=20, rounds=5):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        v = []
        for _ in range(rounds):
            e0, e1 = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
            torch.cuda.synchronize()
            e0.record()
            for _ in range(n):
                fn()
            e1.record()
            e1.synchronize()
            v.append(e0.elapsed_time(e1) / n)
        return statistics.median(v)
    cfgs = {"all_off": [], "all_on": UNITS_ROGT1}
    for u in UNITS_ROGT1:
        cfgs["loo_" + u] = [x for x in UNITS_ROGT1 if x != u]
    for cname, c in cells.items():
        try:
                B, L, H, R, N, KA = (c[k] for k in "B L H R N KA".split())
                K = N // 2
                ang = torch.randn(B, L, H, KA, device=dev, generator=gen) * 0.5
                dt = torch.nn.functional.softplus(
                    torch.randn(B, R * H, L, device=dev, generator=gen) - 3.0)
                dcum = torch.randn(B, L, R * H, K, device=dev, generator=gen)
                ref = {}
                for name, units in cfgs.items():
                    with levers.forced({u: True for u in units}):
                        th, _ = me.phase_law_fwd(ang, dt, None, H, K, -1.0, ka=KA)
                        bw = me.phase_law_bwd(dcum, ang, dt, None, H, K, -1.0, ka=KA)
                        tf = med(lambda: me.phase_law_fwd(ang, dt, None, H, K, -1.0, ka=KA))
                        tb = med(lambda: me.phase_law_bwd(dcum, ang, dt, None, H, K, -1.0, ka=KA))
                    out["timings"].setdefault(cname, {})[name] = dict(fwd=tf, bwd=tb)
                # ELIGIBILITY vs the fp64 ORACLE (tol16 rule: >= 5 draws + signed bias
                # check): BOTH arms of every unit are scored against the oracle; a
                # differential between two fp32 paths is not an arithmetic tolerance.
                import math
                import torch.nn.functional as F
                errs = {n: {t: [] for t in ("th", "dang", "ddt")} for n in cfgs}
                sgn = {n: {t: [] for t in ("th", "dang", "ddt")} for n in cfgs}
                for draw in range(5):
                    g2 = torch.Generator(device=dev).manual_seed(20261003 + 17 * draw)
                    ang_d = torch.randn(B, L, H, KA, device=dev, generator=g2) * 0.5
                    dt_d = torch.nn.functional.softplus(
                        torch.randn(B, R * H, L, device=dev, generator=g2) - 3.0)
                    dcum_d = torch.randn(B, L, R * H, K, device=dev, generator=g2)
                    a64 = ang_d.double().requires_grad_()
                    d64 = dt_d.double().requires_grad_()
                    pad = F.pad(torch.tanh(a64) * math.pi, (0, K - KA))        # [B,L,H,K]
                    pad = pad.unsqueeze(2).expand(B, L, R, H, K).reshape(B, L, R * H, K)
                    th64 = (-1.0 * pad * d64.permute(0, 2, 1)[..., None]).cumsum(1)
                    da64, dd64 = torch.autograd.grad(th64, (a64, d64), dcum_d.double())
                    oracle = dict(th=th64.detach(), dang=da64, ddt=dd64)
                    del a64, d64, pad, th64
                    for name, units in cfgs.items():
                        with levers.forced({u: True for u in units}):
                            th, _ = me.phase_law_fwd(ang_d, dt_d, None, H, K, -1.0, ka=KA)
                            bw = me.phase_law_bwd(dcum_d, ang_d, dt_d, None, H, K, -1.0, ka=KA)
                        for t, x in (("th", th), ("dang", bw[0]), ("ddt", bw[1])):
                            r_ = oracle[t]
                            d_ = x.double() - r_
                            if t == "th":
                                # the kernel stores the phase WRAPPED modulo 2 pi (only cos / sin
                                # of it are ever used): compare the wrapped difference against
                                # the wrapped oracle, not the raw cumulated angle
                                two_pi = 2.0 * math.pi
                                d_ = torch.remainder(d_ + math.pi, two_pi) - math.pi
                                r_ = torch.remainder(r_ + math.pi, two_pi) - math.pi
                            errs[name][t].append((d_.norm() / r_.norm().clamp_min(1e-300)).item())
                            sgn[name][t].append(((d_ * r_).sum() / (r_ * r_).sum().clamp_min(1e-300)).item())
                    del oracle
                out["oracle"][cname] = dict(err=errs, signed=sgn, L=L)
        except torch.cuda.OutOfMemoryError as e:
            out["timings"].pop(cname, None); out["oracle"].pop(cname, None)
            out["dropped"][cname] = str(e).splitlines()[0][:200]
            torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------- orchestration --
def run_child(kind, a, extra, base):
    out = os.path.join(base, "%s_%s.json" % (kind, extra.get("shape", "all")))
    cdir = os.path.join(base, "triton")
    os.makedirs(cdir, exist_ok=True)
    env = dict(os.environ, TRITON_CACHE_DIR=cdir,
               PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, "src")]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, os.path.abspath(__file__), "--child", kind,
           "--out", out, "--rounds", str(a.rounds)]
    for k, v in extra.items():
        cmd += ["--" + k, v]
    t = time.time()
    p = subprocess.run(cmd, env=env, timeout=a.timeout)
    if p.returncode:
        raise SystemExit("child %s %s failed (exit %d)" % (kind, extra, p.returncode))
    print("[levers] %s %s done in %.0fs" % (kind, extra.get("shape", ""), time.time() - t),
          flush=True)
    return json.load(open(out))


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def _oracle_eligible(oracle, cells, u):
    """A unit is eligible in a group when, on every cell and tensor, the lever
    arm (all units on) vs the base arm (the same set without `u`), both scored
    against the fp64 oracle over 5 draws: (1) the lever arm meets the 16-bit
    tolerance rule u_fp32 * sqrt(depth) (depth = L: the cumulated phase and its
    suffix-sum adjoint; floor 1e-6); (2) its pooled signed error is under
    bound/4; (3) its mean error is not worse than the base arm's beyond noise
    (max(2 sigma_base, 25 % of the base error))."""
    import statistics
    worst = 0.0
    for c in cells:
        o = oracle[c]
        bound = max(2.0 ** -24 * (o["L"] ** 0.5), 1e-6)
        for t in ("th", "dang", "ddt"):
            on, base = o["err"]["all_on"][t], o["err"]["loo_" + u][t]
            e_on, e_base = statistics.mean(on), statistics.mean(base)
            sd = statistics.pstdev(base)
            sg = statistics.mean(o["signed"]["all_on"][t])
            worst = max(worst, e_on / max(e_base, 1e-300))
            if e_on > bound or abs(sg) >= bound / 4 or e_on > e_base + max(2 * sd, 0.25 * e_base):
                return False, worst
    return True, worst


def decide_micro(mic, th, rep, only=None):
    """The isolated R > 1 phase-law groups (`complex_ROgt1`: R = 2 cells; `complex_R4O4`: the `r4_*` cells) -> thresholds into `th`."""
    if mic:
        for grp, units in (("complex_ROgt1", UNITS_ROGT1), ("complex_R4O4", UNITS_ROGT1)):
            if only and grp not in only:
                continue
            cells = sorted(c for c in mic["timings"] if c.startswith("r4_") == (grp == "complex_R4O4"))
            if not cells:
                continue
            for u in units:
                from orbit_ssm.kernels.ssd_mimo.levers import UNITS
                ph = UNITS[u][1]
                ok, worst_ratio = _oracle_eligible(mic["oracle"], cells, u)
                rs = [mic["timings"][c]["loo_" + u][ph] / mic["timings"][c]["all_on"][ph]
                      for c in cells]
                gm, mn = geomean(rs), min(rs)
                on = ok and gm >= TOL_ON and mn >= TOL_WORST
                th["lever|%s|%s" % (u, grp)] = dict(value=int(on), evidence=dict(
                    eligible=ok, phase=ph, geomean_gain=gm, worst_shape_gain=mn,
                    oracle_worst_on_over_base=worst_ratio,
                    method="isolated phase_law at R = %d: BOTH arms vs the fp64 oracle, tol16 "
                           "rule (5 draws, signed-bias check), lever not worse than base beyond noise" % (4 if grp == "complex_R4O4" else 2)))
                rep.append("  [%s] %-24s %s gain %.3f worst %.3f eligible=%s -> %s" % (
                    grp, u, ph, gm, mn, ok, "ON" if on else "off"))


def decide(raw):
    """raw: {'applicable': [...], 'eligible': {...}, 'ablate': {shape: ...},
    'micro': ...} -> thresholds dict + report lines."""
    th, rep = {}, []
    elig = raw["eligible"]
    ab = raw["ablate"]
    shapes = sorted(ab)
    phase_of = {"fwd": "fwd", "bwd": "bwd"}
    unit_phase = {u: __import__("orbit_ssm.kernels.ssd_mimo.levers", fromlist=["x"]).UNITS[u][1]
                  for u in UNITS_R1O1}

    def t_phase(cfg, shape, ph):
        m = ab[shape]["median_ms"][cfg]
        return m["fwd"] if ph == "fwd" else m["fwd_bwd"] - m["fwd"]
    chosen = {}
    for u in raw["applicable"]:
        ph = unit_phase[u]
        if u in raw.get("excluded", {}):
            chosen[u] = dict(value=0, evidence=dict(
                eligible=True, phase=ph, excluded=raw["excluded"][u],
                reason="the unit combination does not launch on this device "
                       "(shared memory)"))
            rep.append("  %-24s %s EXCLUDED (combination overflows: %s)" % (
                u, ph, raw["excluded"][u][:90]))
            continue
        if not elig[u]["eligible"]:
            chosen[u] = dict(value=0, evidence=dict(
                eligible=False, reason="fp64 stress failed", phase=ph,
                failures=elig[u].get("failures")))
            rep.append("  %-24s %s INELIGIBLE (fp64 stress)" % (u, ph))
            continue
        if any("loo_" + u in ab[s].get("infeasible", {}) for s in shapes):
            bad = [(s, ab[s]["infeasible"]["loo_" + u]) for s in shapes
                   if "loo_" + u in ab[s].get("infeasible", {})]
            chosen[u] = dict(value=1, evidence=dict(
                eligible=True, phase=ph, required=True,
                reason="the set without this unit does not launch on this "
                       "device (shared memory): %s" % (bad[0][1],)))
            rep.append("  %-24s %s REQUIRED (without it the set overflows shared memory) -> ON" % (u, ph))
            continue
        rs = [t_phase("loo_" + u, s, ph) / t_phase("allon", s, ph) for s in shapes]
        gm, mn = geomean(rs), min(rs)
        on = gm >= TOL_ON and mn >= TOL_WORST
        chosen[u] = dict(value=int(on), evidence=dict(
            eligible=True, phase=ph, geomean_gain=gm, worst_shape_gain=mn,
            per_shape={s: r for s, r in zip(shapes, rs)}))
        rep.append("  %-24s %s gain geomean %.3f worst %.3f -> %s" % (
            u, ph, gm, mn, "ON" if on else "off"))
    # verification of the chosen set per phase
    final_on = [u for u, c in chosen.items() if c["value"]]
    ver = {}
    if "final" in ab[shapes[0]]["median_ms"]:
        for ph in ("fwd", "bwd"):
            r_base = geomean([t_phase("final", s, ph) / t_phase("base", s, ph) for s in shapes])
            r_all = geomean([t_phase("final", s, ph) / t_phase("allon", s, ph) for s in shapes])
            ver[ph] = dict(final_over_base=r_base, final_over_allon=r_all)
            rep.append("  VERIFY %s: FINAL/BASE %.3f  FINAL/ALLON %.3f" % (ph, r_base, r_all))
            if r_base > 1.0:
                for u, c in chosen.items():
                    if c["evidence"].get("phase") == ph and c["value"]:
                        c["value"] = 0
                        c["evidence"]["turned_off"] = "FINAL slower than BASE on %s" % ph
                rep.append("  -> every %s unit turned off (FINAL slower than BASE)" % ph)
    for u, c in chosen.items():
        c["evidence"]["verify"] = ver
        th["lever|%s|%s" % (u, GROUP)] = c
    decide_micro(raw.get("micro"), th, rep)
    return th, rep


def measure(a):
    base = "/dev/shm/levers_%s_%d" % (os.environ.get("SLURM_JOB_ID", "x"), os.getpid())
    os.makedirs(base, exist_ok=True)
    raw = {}
    try:
        from orbit_ssm.kernels.ssd_mimo import calibration_ssd as cs
        _prep_env()
        import torch  # noqa: F401
        app = applicable_r1o1()
        raw["applicable"] = app
        # 1. eligibility
        st = run_child("stress", a, dict(sets=json.dumps(
            {"base": [], "allon": app})), base)
        if not st["base"]["passed"]:
            raise SystemExit("BASE fails the fp64 stress: %s" % st["base"]["failures"][:3])
        elig = {u: dict(eligible=st["allon"]["passed"]) for u in app}
        raw["stress"] = {"base": st["base"], "allon": st["allon"]}
        if not st["allon"]["passed"]:
            solo = run_child("stress", a, dict(sets=json.dumps(
                {u: [u] for u in app})), base)
            for u in app:
                elig[u] = dict(eligible=solo[u]["passed"],
                               failures=solo[u]["failures"][:5])
            raw["stress_solo"] = solo
        raw["eligible"] = elig
        # 1b. feasibility: the Gluon units whose combination fits the device
        ok_u = [u for u in app if elig[u]["eligible"]]
        raw["excluded"] = {}
        if any(u.startswith("gl_") for u in ok_u):
            fe = run_child("feasible", a, dict(shape="cx_small_balanced",
                                               sets=json.dumps(ok_u)), base)
            raw["feasible"] = fe
            raw["excluded"] = fe["excluded"]
            ok_u = fe["keep"]
        # 2. ablation
        el = [u for u in app if elig[u]["eligible"] and u in ok_u]
        configs = {"base": [], "allon": el}
        for u in el:
            configs["loo_" + u] = [x for x in el if x != u]
        shapes = [s for s, g in cs.SHAPES.items() if g.get("rot", True)]
        if a.shapes:
            shapes = a.shapes.split(",")
        raw["ablate"], raw["dropped"] = {}, {}
        for s in shapes:
            r = run_child("ablate", a, dict(shape=s, sets=json.dumps(configs)), base)
            if r.get("oom"):
                raw["dropped"][s] = r["error"]
                print("[levers] %s DROPPED (out of memory): %s" % (s, r["error"]),
                      flush=True)
                continue
            raw["ablate"][s] = r
        shapes = list(raw["ablate"])
        # 3. first decision, then verify FINAL
        th, _ = decide(raw)
        final = [u for u in el if th["lever|%s|%s" % (u, GROUP)]["value"]]
        raw["final_units"] = final
        for s in shapes:
            raw["ablate"][s]["final_run"] = run_child("ablate", a, dict(
                shape=s, sets=json.dumps({"base": [], "allon": el, "final": final})), base)
            if "final" in raw["ablate"][s]["final_run"]["infeasible"]:
                raise SystemExit("FINAL set %s does not launch at %s: %s" % (
                    final, s, raw["ablate"][s]["final_run"]["infeasible"]["final"]))
            raw["ablate"][s]["median_ms"]["final"] = \
                raw["ablate"][s]["final_run"]["median_ms"]["final"]
        raw["micro"] = run_child("micro", a, {}, base)
        raw["dropped"].update({"micro:" + k: v for k, v in raw["micro"].get("dropped", {}).items()})
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write-shipped", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--raw", default="")
    ap.add_argument("--from-raw", dest="from_raw", default="")
    ap.add_argument("--gpu", default="", help="with --from-raw: GPU name of the raw file")
    ap.add_argument("--sms", type=int, default=0, help="with --from-raw: SM count of that GPU")
    ap.add_argument("--redo-micro", action="store_true",
                    help="with --from-raw on the SAME GPU: re-measure only the isolated R>1 phase-law "
                         "group (child micro) and merge it into the raw file (written to --raw)")
    ap.add_argument("--ranks", default="2", help="micro child: `4` measures ONLY the per-rank group complex_R4O4 (cells r4_<LM tag>)")
    ap.add_argument("--rank-micro", action="store_true",
                    help="measure ONLY the complex_R4O4 phase-law group and MERGE its thresholds into the existing file of the target layer")
    ap.add_argument("--out-file", default="", help="write the derived file HERE (default: shipped/local path)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--child", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--shape", default="")
    ap.add_argument("--sets", default="")
    a = ap.parse_args()
    if a.child:
        r = globals()["child_" + a.child](a)
        json.dump(r, open(a.out, "w"))
        return
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import levers
    if a.gpu and a.sms:          # re-derive another GPU's raw file offline
        import triton as _t
        cal._DEV.update(gpu=a.gpu, sms=a.sms, driver=None,
                        triton=_t.__version__, torch="offline")
    dev = cal.device_info()
    gpu = dev["gpu"]
    data, layer = cal.load(levers.FAMILY, gpu, dev["triton"])
    if data is not None and layer == "local" and not (a.force or a.write_shipped):
        print("valid local calibration exists; --force to redo")
        return
    if a.rank_micro:
        a.ranks = "4"
        base = "/dev/shm/levers_rankmicro_%d" % os.getpid()
        os.makedirs(base, exist_ok=True)
        try:
            micro = run_child("micro", a, {"ranks": "4"}, base)
        finally:
            shutil.rmtree(base, ignore_errors=True)
        th, rep = {}, []
        decide_micro(micro, th, rep, only=("complex_R4O4",))
        print("\n".join(rep))
        p = a.out_file or (cal.shipped_path if a.write_shipped else cal.local_path)(levers.FAMILY, gpu)
        old = cal._read(p) or {}
        keep = {k: v for k, v in old.get("thresholds", {}).items() if not k.endswith("|complex_R4O4")}
        data = cal.build_file(levers.FAMILY, {}, a.rounds, old.get("provenance", {}).get("shapes", []),
                              thresholds=dict(keep, **th), dropped=dict(old.get("provenance", {}).get("dropped_shapes") or {},
                                                                        **{"micro:" + k: v for k, v in micro.get("dropped", {}).items()}))
        print("wrote", cal.write_json(p, data))
        return
    if a.redo_micro:
        raw = json.load(open(a.from_raw))
        base = "/dev/shm/levers_micro_%d" % os.getpid()
        os.makedirs(base, exist_ok=True)
        try:
            raw["micro"] = run_child("micro", a, {}, base)
        finally:
            shutil.rmtree(base, ignore_errors=True)
        raw["dropped"] = {k: v for k, v in raw.get("dropped", {}).items() if not k.startswith("micro:")}
        raw["dropped"].update({"micro:" + k: v for k, v in raw["micro"].get("dropped", {}).items()})
        json.dump(raw, open(a.raw or a.from_raw, "w"), indent=1)
        print("micro re-measured; derive offline with --from-raw --gpu --sms")
        return
    raw = json.load(open(a.from_raw)) if a.from_raw else measure(a)
    if a.raw:
        json.dump(raw, open(a.raw, "w"), indent=1)
    th, rep = decide(raw)
    print("lever calibration on %s (triton %s, code %s):" % (
        gpu, dev["triton"], cal.code_hash(levers.FAMILY)))
    print("\n".join(rep))
    # the real_R1O1 units are measured by the real-decay calibration: carry the file's choices over
    _p = a.out_file or (cal.shipped_path if a.write_shipped else cal.local_path)(levers.FAMILY, gpu)
    _old = cal._read(_p) or {}
    th.update({k: v for k, v in _old.get("thresholds", {}).items() if k.endswith(("|real_R1O1", "|real_R1O1_hi"))})
    fresh = cal.build_file(levers.FAMILY, {}, a.rounds, list(raw["ablate"]),
                           thresholds=th, dropped=raw.get("dropped"))
    path = a.out_file or (cal.shipped_path if a.write_shipped else cal.local_path)(
        levers.FAMILY, gpu)
    print("wrote", cal.write_json(path, fresh))


if __name__ == "__main__":
    main()
