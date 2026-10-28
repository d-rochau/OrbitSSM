#!/usr/bin/env python
"""Calibrate the rotating R=O=1 structural route (family `ssd_mimo_route`, src/orbit_ssm/kernels/ssd_mimo/rot_route.py).

Inputs (one GPU):
  --ab FILE [FILE ...]   jsonl of `benchmarks`-side route timings: per shared-set cell (B, L, H),
                         arms A_T64 / B_T32 / B_T64 / C_T32 / C_T64 (theta_h False and True) and `official_m3`, passes fwd / fwd_bwd
  --elig DIR             the route eligibility run of THIS GPU (run_elig.sh): stress_<route>.log, c10_<route>_{noth,thh}.log
A route is ELIGIBLE iff its fp64 stress passed and C10 passed with and without theta_h (8 draws, current rules).
The rule `siso iff B*H/SMs < siso_occ and L <= siso_L else alt` is fitted per pass (grad: fwd_bwd timings, alternatives split/inkernel;
nograd: fwd timings, alternatives inkernel) minimising sum log(t_chosen / t_best_eligible) over the cells and both theta_h variants;
`alt` is the best eligible alternative.  Ties go to the simpler rule (siso everywhere).
    python tools/calibrate_routes.py --ab ab.jsonl --elig elig_dir [--gpu NAME --sms N] [--write-shipped | --out-file F]"""
import argparse
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/src")

ARM = {"A_T64": "siso", "B_T32": "split32", "B_T64": "split64", "C_T32": "inkernel32", "C_T64": "inkernel64"}
#: fp32 IO (`--dtype fp32`): the SISO schedule is measured at BOTH chunks (its fp32 operand tiles are twice as wide);
#: the rule's SISO chunk `siso_T` is the eligible one with the lower total time, then the same two-threshold fit runs
ARM32 = {"A_T32": "siso32", "A_T64": "siso64", "B_T32": "split32", "B_T64": "split64", "C_T32": "inkernel32",
         "C_T64": "inkernel64"}
INF = 10 ** 9
GATE = 1.02


def load_times(paths, arms=None):
    """{(B, L, H): {(route, thh, pass): ms}} (last record wins), plus the official time per pass."""
    arms = arms or ARM
    T = collections.defaultdict(dict)
    off = collections.defaultdict(dict)
    gpu = None
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            gpu = gpu or r.get("gpu")
            k = (r["B"], r["L"], r["H"])
            if "ms" not in r:
                continue
            if r["arm"] == "official_m3":
                off[k][r["pass_"]] = r["ms"]
            elif r["arm"] in arms:
                T[k][(arms[r["arm"]], bool(r.get("theta_h")), r["pass_"])] = r["ms"]
    return T, off, gpu


def eligibility32(d):
    """fp32: {route: {...}} from an fp32 eligibility dir: stress_<A32|A64|B32|B64|C32|C64>.log (`STRESS32 <name> passed`, the
    fp32 fp64-oracle stress) and c10_<same>_{noth,thh}.log (tools/ssd_check.py --dtype fp32, 8 draws, '"passed": true')."""
    out = {}
    for arm, name in ARM32.items():
        r = arm.replace("_T", "")
        s = os.path.join(d, "stress_%s.log" % r)
        st = os.path.exists(s) and re.search(r"STRESS32 \S+ passed", open(s).read()) is not None
        c = {}
        for tag in ("noth", "thh"):
            f = os.path.join(d, "c10_%s_%s.log" % (r, tag))
            c[tag] = os.path.exists(f) and '"passed": true' in open(f).read()
        out[name] = dict(stress=bool(st), c10_noth=bool(c["noth"]), c10_thh=bool(c["thh"]),
                         eligible=bool(st and c["noth"] and c["thh"]))
    return out


def eligibility(d):
    """{route: {'stress': bool, 'c10_noth': bool, 'c10_thh': bool, 'eligible': bool}} from a run_elig.sh output dir."""
    out = {}
    for arm, name in ARM.items():
        r = arm.replace("_T", "") if arm != "A_T64" else "A"
        s = os.path.join(d, "stress_%s.log" % r)
        st = os.path.exists(s) and re.search(r"ROUTE_STRESS \S+ passed", open(s).read()) is not None
        c = {}
        for tag in ("noth", "thh"):
            f = os.path.join(d, "c10_%s_%s.log" % (r, tag))
            c[tag] = os.path.exists(f) and '"passed": true' in open(f).read()
        out[name] = dict(stress=bool(st), c10_noth=bool(c["noth"]), c10_thh=bool(c["thh"]),
                         eligible=bool(st and c["noth"] and c["thh"]))
    return out


def fit(T, sms, phase_pass, routes, thh):
    """-> dict(occ=x100, occ_dir, L, L_dir, alt, cost, worst, table) or None.  `routes`: the candidate routes of this
    pass (siso included); `thh`: the group (the call has a static theta_h).  The fit chooses the DIRECTION of each half
    of the rule; its key is (worst misroute capped at the 1.02 gate, sum of log slowdowns, siso-everywhere first)."""
    gi = 1 if thh else 0
    rows = []                                       # (occ, L, B, H, {route: ms})
    for k in sorted(T):
        B, L, H = k
        tt = {r: T[k][(r, bool(thh), phase_pass)] for r in routes if (r, bool(thh), phase_pass) in T[k]}
        if "siso" in tt:
            rows.append((B * H / sms, L, B, H, tt))
    alts = [r for r in routes if r != "siso" and all(r in tt for *_, tt in rows)]
    if not alts or not rows:
        return None

    def region(occ, L, ot, od, lt, ld):
        return ((occ * 100 < ot) if not od else (occ * 100 >= ot)) and ((L <= lt) if not ld else (L > lt))

    def cost(ot, od, lt, ld, alt):
        c, worst = 0.0, 1.0
        for occ, L, B, H, tt in rows:
            ch = "siso" if region(occ, L, ot, od, lt, ld) else alt
            x = tt[ch] / min(tt.values())
            c += math.log(x)
            worst = max(worst, x)
        return c, worst
    occ_c = sorted({int(math.floor(o * 100)) + 1 for o, *_ in rows} | {0, INF})
    l_c = sorted({r[1] for r in rows} | {0, INF})
    best = None
    for alt in alts:
        for od in (0, 1):
            for ld in (0, 1):
                for ot in occ_c:
                    for lt in l_c:
                        c, w = cost(ot, od, lt, ld, alt)
                        everywhere = (od == 0 and ld == 0 and ot == INF and lt == INF)
                        key = (round(max(w, 1.02), 6), round(c, 9), 0 if everywhere else 1)
                        if best is None or key < best[0]:
                            best = (key, dict(occ=ot, occ_dir=od, L=lt, L_dir=ld, alt=alt, cost=c, worst=w))
    res = best[1]
    res["table"] = []
    for occ, L, B, H, tt in rows:
        ch = "siso" if region(occ, L, res["occ"], res["occ_dir"], res["L"], res["L_dir"]) else res["alt"]
        bt = min(tt, key=tt.get)
        res["table"].append(dict(B=B, L=L, H=H, occ=round(occ, 3), chosen=ch, chosen_ms=round(tt[ch], 3), best=bt,
                                 best_ms=round(tt[bt], 3), slowdown=round(tt[ch] / tt[bt], 3),
                                 siso_ms=round(tt["siso"], 3), official_ms=None))
    return res


def table_from(T, sms, phase_pass, routes, thh):
    """The 2D fallback: per measured cell the best route (siso = -1, else the index into `rr.ALTS`) in (log2 B*H/SMs, log2 L)."""
    from orbit_ssm.kernels.ssd_mimo import rot_route as rr
    gi = bool(thh)
    rows, table = [], []
    for k in sorted(T):
        B, L, H = k
        tt = {r: T[k][(r, gi, phase_pass)] for r in routes if (r, gi, phase_pass) in T[k]}
        if "siso" not in tt:
            continue
        best = min(tt, key=tt.get)
        idx = -1 if best == "siso" else rr.ALT_NAMES.index(best)
        table.append([round(math.log2(max(B * H / sms, 1e-3)), 4), round(math.log2(L), 4), idx])
        rows.append(dict(B=B, L=L, H=H, occ=round(B * H / sms, 3), chosen=best, chosen_ms=round(tt[best], 3), best=best,
                         best_ms=round(tt[best], 3), slowdown=1.0, siso_ms=round(tt["siso"], 3)))
    return table, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ab", nargs="+", required=True)
    ap.add_argument("--elig", required=True)
    ap.add_argument("--gpu", default="")
    ap.add_argument("--sms", type=int, default=0)
    ap.add_argument("--write-shipped", action="store_true")
    ap.add_argument("--allow-fail", action="store_true", help="write the file although a rule misroutes a cell by more than the gate")
    ap.add_argument("--out-file", default="")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"),
                    help="fp32: fit the fp32-IO rule (phases grad_fp32 / nograd_fp32, SISO at its calibrated chunk `siso_T`) and "
                         "MERGE its keys into the GPU's existing file (the 16-bit rule is kept)")
    a = ap.parse_args()
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import rot_route as rr
    if a.gpu and a.sms:
        import triton as _t
        cal._DEV.update(gpu=a.gpu, sms=a.sms, driver=None, triton=_t.__version__, torch="offline")
    dev = cal.device_info()
    if a.dtype == "fp32":
        return main_fp32(a, cal, rr, dev)
    T, off, gpu = load_times(a.ab)
    if gpu != dev["gpu"]:
        raise SystemExit("timings are from %r, calibrating %r" % (gpu, dev["gpu"]))
    el = eligibility(a.elig)
    print("eligibility on %s:" % gpu)
    for r, v in el.items():
        print("  %-11s %s" % (r, v))
    th, report, failed = {}, {}, []
    for phase, pas, pool in (("grad", "fwd_bwd", ("siso", "split32", "split64", "inkernel32", "inkernel64")),
                             ("nograd", "fwd", ("siso", "inkernel32", "inkernel64"))):
        routes = [r for r in pool if r == "siso" or el[r]["eligible"]]
        for grp in ("noth", "thh"):
            f = fit(T, dev["sms"], pas, routes, grp == "thh")
            for r in pool:
                th[rr.key(phase, grp, "eligible|" + r)] = dict(value=int(el[r]["eligible"]), evidence=el[r])
            if f is None:
                print("%s/%s: no eligible alternative -> siso everywhere" % (phase, grp))
                for fld, v in (("siso_occ", INF), ("siso_occ_dir", 0), ("siso_L", INF), ("siso_L_dir", 0), ("alt", 0)):
                    th[rr.key(phase, grp, fld)] = dict(value=v, evidence="no eligible alternative")
                continue
            form = "regions"
            if f["worst"] > GATE:       # no two-threshold region reaches the gate: the per-cell table (nearest measured cell)
                tab, trows = table_from(T, dev["sms"], pas, routes, grp == "thh")
                th[rr.key(phase, grp, "table")] = dict(value=tab, evidence=trows)
                form = "table"
                f = dict(f, worst=1.0, table=trows)
            ev = dict(tool="tools/calibrate_routes.py", form=form, cost=f["cost"], worst_misroute=f["worst"], cells=len(f["table"]),
                      candidates=routes, gate=GATE, passed=bool(f["worst"] <= GATE))
            for fld, v in (("siso_occ", f["occ"]), ("siso_occ_dir", f["occ_dir"]), ("siso_L", f["L"]),
                           ("siso_L_dir", f["L_dir"])):
                th[rr.key(phase, grp, fld)] = dict(value=v, evidence=ev)
            th[rr.key(phase, grp, "alt")] = dict(value=rr.ALT_NAMES.index(f["alt"]), evidence=dict(ev, alt=f["alt"]))
            if form != "table":
                th[rr.key(phase, grp, "table")] = dict(value=None, evidence=f["table"])
            inf = lambda v: "inf" if v >= INF else v
            print("%s/%s: siso iff (B*H/SMs %s %s) and (L %s %s), else %s   (sum log = %.3f, worst misroute = %.3f %s)" % (
                phase, grp, ">=" if f["occ_dir"] else "<", inf(f["occ"] / 100 if f["occ"] < INF else f["occ"]),
                ">" if f["L_dir"] else "<=", inf(f["L"]), f["alt"], f["cost"], f["worst"],
                "PASS" if f["worst"] <= GATE else "FAILS the %.2f gate" % GATE))
            for row in f["table"]:
                print("   B%-4d L%-6d H%-4d occ %7.3f  chosen %-10s %8.3f ms   best %-10s %8.3f   misroute x%.3f   (siso %.3f)" % (
                    row["B"], row["L"], row["H"], row["occ"], row["chosen"], row["chosen_ms"], row["best"], row["best_ms"],
                    row["slowdown"], row["siso_ms"]))
            if f["worst"] > GATE:
                failed.append("%s/%s worst misroute %.3f" % (phase, grp, f["worst"]))
            report[(phase, grp)] = f
    if failed and not a.allow_fail:
        raise SystemExit("rule gate FAILED (worst misroute > %.2f): %s; nothing written (--allow-fail to write anyway)" % (
            GATE, "; ".join(failed)))
    from orbit_ssm.kernels import shape_sets as ss
    fresh = cal.build_file(rr.FAMILY, {}, 5, list(ss.SET), thresholds=th)
    fresh["provenance"]["notes"] = "route timings: followup/route_ab (production module path), eligibility: %s" % os.path.basename(
        os.path.normpath(a.elig))
    path = a.out_file or (cal.shipped_path if a.write_shipped else cal.local_path)(rr.FAMILY, gpu)
    print("wrote", cal.write_json(path, fresh))


def main_fp32(a, cal, rr, dev):
    """The fp32-IO rule (see `--dtype`): eligibility from the fp32 dir, `siso_T` = the eligible SISO chunk with the lower total
    time (grad: fwd_bwd), then `fit` per (pass, group) exactly as for 16-bit; keys `route|<grad|nograd>_fp32|<group>|...` merged
    into the existing file (shipped with --write-shipped, else local) next to the 16-bit keys."""
    T, off, gpu = load_times(a.ab, ARM32)
    if gpu != dev["gpu"]:
        raise SystemExit("timings are from %r, calibrating %r" % (gpu, dev["gpu"]))
    el = eligibility32(a.elig)
    print("fp32 eligibility on %s:" % gpu)
    for r, v in el.items():
        print("  %-11s %s" % (r, v))
    th, failed = {}, []
    for phase, pas, pool in (("grad", "fwd_bwd", ("split32", "split64", "inkernel32", "inkernel64")),
                             ("nograd", "fwd", ("inkernel32", "inkernel64"))):
        ph = rr.phase_of(phase == "grad", "fp32")
        # fp32 TRAINING never takes the SISO backward (siso_route.siso_enabled): the grad rule chooses among the general routes
        sisos = [s for s in ("siso32", "siso64") if el[s]["eligible"]] if phase == "nograd" else []
        cost = {}
        for s_ in sisos:                       # total log-time over every cell and both theta_h variants
            v = [math.log(T[k][(s_, thh, pas)]) for k in T for thh in (False, True) if (s_, thh, pas) in T[k]]
            n_all = sum(1 for k in T for thh in (False, True) if any((x, thh, pas) in T[k] for x in ("siso32", "siso64")))
            if v and len(v) == n_all:
                cost[s_] = sum(v)
        siso = min(cost, key=cost.get) if cost else None
        alts = [r for r in pool if el[r]["eligible"]]
        for grp in ("noth", "thh"):
            for r in ("siso32", "siso64") + pool:
                th[rr.key(ph, grp, "eligible|" + r)] = dict(value=int(el[r]["eligible"]), evidence=el[r])
            if siso is None or not alts:
                why = "no eligible fp32 SISO chunk" if siso is None else "no eligible alternative"
                if siso is None:
                    # never SISO: the region is empty; alt = the eligible alternative with the lowest total time, and when
                    # that single alternative misroutes a cell by more than the gate, the per-cell table (nearest measured
                    # cell in log2 occupancy x log2 L) over the alternatives
                    thh_ = grp == "thh"
                    cells = [(k, {r: T[k][(r, thh_, pas)] for r in alts if (r, thh_, pas) in T[k]}) for k in sorted(T)]
                    cells = [(k, tt) for k, tt in cells if len(tt) == len(alts)]
                    best = min(alts, key=lambda r: sum(math.log(tt[r]) for _, tt in cells)) if alts else "inkernel32"
                    worst = max([tt[best] / min(tt.values()) for _, tt in cells] or [1.0])
                    ev = dict(why=why, alt=best, worst_misroute_single=worst, cells=len(cells), candidates=alts)
                    for fld, v in (("siso_occ", 0), ("siso_occ_dir", 0), ("siso_L", INF), ("siso_L_dir", 0),
                                   ("alt", rr.ALT_NAMES.index(best))):
                        th[rr.key(ph, grp, fld)] = dict(value=v, evidence=ev)
                    if worst > GATE and cells:
                        tab = [[round(math.log2(max(B * H / dev["sms"], 1e-3)), 4), round(math.log2(L), 4),
                                rr.ALT_NAMES.index(min(tt, key=tt.get))] for (B, L, H), tt in cells]
                        th[rr.key(ph, grp, "table")] = dict(value=tab, evidence=[dict(B=B, L=L, H=H, best=min(tt, key=tt.get),
                                                                                       ms={r: round(v, 3) for r, v in tt.items()})
                                                                                  for (B, L, H), tt in cells])
                    print("%s/%s: no SISO; alt %s (single-alt worst misroute %.3f%s)" % (
                        ph, grp, best, worst, ", per-cell table stored" if worst > GATE else ""))
                    continue
                else:
                    for fld, v in (("siso_occ", INF), ("siso_occ_dir", 0), ("siso_L", INF), ("siso_L_dir", 0), ("alt", 0)):
                        th[rr.key(ph, grp, fld)] = dict(value=v, evidence=why)
                    th[rr.key(ph, grp, "siso_T")] = dict(value=int(siso[4:]), evidence=dict(cost=cost))
                print("%s/%s: %s" % (ph, grp, why))
                continue
            Tg = collections.defaultdict(dict)        # rename the chosen SISO chunk to "siso" for `fit`
            for k, d in T.items():
                for (r, thh, p), ms in d.items():
                    if r == siso:
                        Tg[k][("siso", thh, p)] = ms
                    elif not r.startswith("siso"):
                        Tg[k][(r, thh, p)] = ms
            f = fit(Tg, dev["sms"], pas, ["siso"] + alts, grp == "thh")
            form = "regions"
            if f["worst"] > GATE:
                tab, trows = table_from(Tg, dev["sms"], pas, ["siso"] + alts, grp == "thh")
                th[rr.key(ph, grp, "table")] = dict(value=tab, evidence=trows)
                form, f = "table", dict(f, worst=1.0, table=trows)
            ev = dict(tool="tools/calibrate_routes.py --dtype fp32", form=form, cost=f["cost"], worst_misroute=f["worst"],
                      cells=len(f["table"]), candidates=["siso(%s)" % siso] + alts, gate=GATE, passed=bool(f["worst"] <= GATE),
                      siso_cost=cost)
            for fld, v in (("siso_occ", f["occ"]), ("siso_occ_dir", f["occ_dir"]), ("siso_L", f["L"]),
                           ("siso_L_dir", f["L_dir"])):
                th[rr.key(ph, grp, fld)] = dict(value=v, evidence=ev)
            th[rr.key(ph, grp, "alt")] = dict(value=rr.ALT_NAMES.index(f["alt"]), evidence=dict(ev, alt=f["alt"]))
            th[rr.key(ph, grp, "siso_T")] = dict(value=int(siso[4:]), evidence=ev)
            if form != "table":
                th[rr.key(ph, grp, "table")] = dict(value=None, evidence=f["table"])
            inf = lambda v: "inf" if v >= INF else v
            print("%s/%s: siso(T=%s) iff (B*H/SMs %s %s) and (L %s %s), else %s   (sum log = %.3f, worst misroute = %.3f)" % (
                ph, grp, siso[4:], ">=" if f["occ_dir"] else "<", inf(f["occ"] / 100 if f["occ"] < INF else f["occ"]),
                ">" if f["L_dir"] else "<=", inf(f["L"]), f["alt"], f["cost"], f["worst"]))
            for row in f["table"]:
                print("   B%-4d L%-6d H%-4d occ %7.3f  chosen %-10s %8.3f ms   best %-10s %8.3f   misroute x%.3f   (siso %.3f)" % (
                    row["B"], row["L"], row["H"], row["occ"], row["chosen"], row["chosen_ms"], row["best"], row["best_ms"],
                    row["slowdown"], row["siso_ms"]))
            if f["worst"] > GATE:
                failed.append("%s/%s worst misroute %.3f" % (ph, grp, f["worst"]))
    path = a.out_file or (cal.shipped_path if a.write_shipped else cal.local_path)(rr.FAMILY, gpu)
    base = cal._read(path)
    if base is None:
        from orbit_ssm.kernels import shape_sets as ss
        base = cal.build_file(rr.FAMILY, {}, 5, list(ss.SET), thresholds={})
    if base["provenance"].get("code_hash") != cal.code_hash(rr.FAMILY):
        raise SystemExit("the existing %s is stale for the current rot_route.py (%s != %s): recalibrate the 16-bit rule first"
                         % (path, base["provenance"].get("code_hash"), cal.code_hash(rr.FAMILY)))
    base["thresholds"] = {k: v for k, v in base.get("thresholds", {}).items() if "_fp32|" not in k}
    base["thresholds"].update(th)
    base["provenance"].setdefault("merged", []).append(dict(what="fp32 route rule (tools/calibrate_routes.py --dtype fp32)",
                                                           ab=[os.path.basename(x) for x in a.ab],
                                                           elig=os.path.basename(os.path.normpath(a.elig)),
                                                           job_id=os.environ.get("SLURM_JOB_ID"),
                                                           date=__import__("datetime").date.today().isoformat(),
                                                           failed_gate=failed))
    print("wrote", cal.write_json(path, base))


if __name__ == "__main__":
    main()
