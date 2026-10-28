#!/usr/bin/env python
"""MH (`mh_scan`) backward in the EXTREME-DECAY regimes -- the HM counterpart
of `tools/ssd_check_saturated.py` (decay-gradient cancellation).

Chosen heads are forced into a decay limit by their `A_log` (the per-token
rate is Delta*|A|, Delta = softplus(delta) ~ 0.8 at these draws):

  * saturated: Delta|A| log-spread over [--sat-lo, --sat-hi] (alpha -> 0);
  * slow:      Delta|A| log-spread over [--slow-lo, --slow-hi] (alpha -> 1);
  * the rest keep the draw's own A_log (Delta|A| ~ 0.3..3).

Every arm runs `mh_scan` (the production entry, host folds included) forward
+ backward under the loss <gy, Y> + <gh, h_final>; the reference is the
SEQUENTIAL fp64 recurrence `mh_reference` under autograd, whose decay
adjoints carry alpha as a factor (a chunked fp64 route would share any
cancellation).  Arms: io = x/B/C in bf16 (the LLM route's operands),
ieee = fp32 (delta and A_log fp32 in both, as the module passes them).  MH has no MMA precision knob, so there is no tf32 arm.

Per decay-path tensor (A_log, delta, and the law streams a_dyn / mu /
mu_gain when present) and per head: normwise relative error, the signed
error along the reference (<err, ref>/<ref, ref>; > 0 = inflated), and the
same pooled over draws.  A7 class (b) bound as in ssd_check_saturated.py:
io u16 * sqrt(depth), ieee u32 * sqrt(depth), depth = B*L (every decay
gradient reduces over batch and time), clamped to the A7 band; for ieee the
limit is max(bound, 4 x the fp32-ACHIEVABLE floor), measured per group by the
sequential recurrence itself in fp32 (arm `seq32`) -- a scan cannot beat the
fp32 sequential recurrence, and on slow heads (alpha -> 1, depth ~ L inside
the state) that floor sits above u32*sqrt(B*L).  PASS when the worst head of
every group is under the limit and |pooled signed| < max(bound/4, 2x the
fp32 recurrence's own pooled signed error) (ieee; io: bound/4).

    python tools/hm_check_saturated.py --geom llm --draws 5
    MH_BWD_ABL=4096 python tools/hm_check_saturated.py --geom llm   # pre-fix arm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

GEOMS = {
    # HM LLM arm (HM_LLM_ARM_PROPOSAL.md): d_inner 2048, P16 -> H128, N128; L cut for the fp64 oracle
    "llm": dict(B=2, H=128, P=16, N=128, L=512),
    # MQMKVAR insane HM rows: d_inner 400, head_subsystems 16 -> H25, N128
    "mqmkvar": dict(B=2, H=25, P=16, N=128, L=512),
    "tiny": dict(B=2, H=8, P=4, N=16, L=64),
}
LAWS = {
    "A": None,
    "rot": [dict(frac=1.0, decay="coord_static", rot=True)],
    "dyn": [dict(frac=0.5, decay="coord_static"), dict(frac=0.5, decay="coord_dynamic")],
    "mu": [dict(frac=0.5, decay="coord_static"), dict(frac=0.5, decay="coord_potential")],
}


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geom", choices=sorted(GEOMS), default="llm")
    p.add_argument("--laws", default="A,rot,dyn,mu")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--arms", default="io,ieee")
    p.add_argument("--sat-lo", type=float, default=1.0)
    p.add_argument("--sat-hi", type=float, default=50.0)
    p.add_argument("--slow-lo", type=float, default=1e-4)
    p.add_argument("--slow-hi", type=float, default=1e-2)
    p.add_argument("--tag", default=os.environ.get("HMSAT_TAG", "run"))
    p.add_argument("--json", default=None)
    return p.parse_args()


def _bound(arm, depth):
    u = 2.0 ** -8 if arm == "io" else 2.0 ** -24
    eps = 2 * u
    return min(max(u * math.sqrt(depth), 0.64 * eps), 6.4 * eps)


def _force(a_log, delta, groups, a):
    """A_log per head so that the head's mean Delta|A| hits its target."""
    dmean = torch.nn.functional.softplus(delta.double()).mean().item()
    H = a_log.shape[0]
    idx = torch.arange(H)
    sat, slow = idx[: H // 2 : 2], idx[1 : H // 2 : 2]
    for heads, lo, hi, name in ((sat, a.sat_lo, a.sat_hi, "sat"),
                                (slow, a.slow_lo, a.slow_hi, "slow")):
        n = len(heads)
        for i, h in enumerate(heads.tolist()):
            tgt = lo * (hi / lo) ** (i / max(n - 1, 1))
            a_log[h] = math.log(tgt / dmean)
            groups[h] = (name, tgt)
    for h in range(H):
        groups.setdefault(h, ("ord", None))


def main():
    a = _parse()
    from _oracles import mh_make_inputs
    from test_kernels import _mh_fp64_backward
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    g = GEOMS[a.geom]
    depth = g["B"] * g["L"]
    report = []
    for law in a.laws.split(","):
        pooled = {}          # (arm, tensor, group) -> [signed per draw]
        worst = {}           # (arm, tensor, group) -> max rel err over heads, draws
        for d in range(a.draws):
            args, kw = mh_make_inputs(**g, device="cuda", fracs=LAWS[law],
                                      rot=(law == "rot"), seed=1000 + d)
            x, delta, a_log, b_t, c_t = args
            groups = {}
            a_log = a_log.clone()
            _force(a_log, delta, groups, a)
            if law == "rot":         # A_log must stay pair-tied on rotating rows
                a_log[..., 1::2] = a_log[..., 0::2]
            names = None
            for arm in a.arms.split(",") + ["seq32"]:
                dt = torch.bfloat16 if arm == "io" else torch.float32
                # the oracle sees the SAME rounded inputs as the arm (A7 class b):
                # only the arm's own arithmetic is scored
                rx, rb, rc = ((t.to(torch.bfloat16).float() if arm == "io" else t)
                              for t in (x, b_t, c_t))
                _, _, gy, gh, leaves = _mh_fp64_backward((rx, delta, a_log, rb, rc), kw)
                names = names or [n for n in ("A_log", "delta", "a_dyn", "mu", "mu_gain") if n in leaves]
                if arm == "seq32":
                    # the fp32-ACHIEVABLE floor: the sequential recurrence itself in fp32
                    from test_kernels import _mh_leaves
                    from orbit_ssm.kernels.hybrid_mamba_mh.reference import mh_reference
                    lv, rkw = _mh_leaves((x, delta, a_log, b_t, c_t), kw)
                    lv = {k: v.detach().float().requires_grad_(True) for k, v in lv.items()}
                    rkw = {k: (lv[k] if k in lv else v) for k, v in rkw.items()}
                    import orbit_ssm.kernels.hybrid_mamba_mh.reference as _R
                    _dt, _R.DT = _R.DT, torch.float32     # the reference upcasts to DT: run it IN fp32
                    try:
                        y32, h32 = mh_reference(*(lv[k] for k in ("x", "delta", "A_log", "B_t", "C_t")), **rkw)[:2]
                    finally:
                        _R.DT = _dt
                    assert y32.dtype == torch.float32, y32.dtype
                    ((y32.double() * gy).sum() + (h32.double() * gh).sum()).backward()
                    got = {k: v.grad for k, v in lv.items()}
                    _score(names, leaves, got, groups, arm, worst, pooled)
                    continue
                ins = {}
                for n, t in (("x", x), ("delta", delta), ("A_log", a_log), ("B_t", b_t), ("C_t", c_t)):
                    t2 = t.to(dt) if n in ("x", "B_t", "C_t") else t.float()
                    ins[n] = t2.detach().clone().requires_grad_(True)
                kw2 = {k: (v.detach().clone().float().requires_grad_(True)
                           if torch.is_tensor(v) and v.is_floating_point() else v)
                       for k, v in kw.items()}
                y, h = mh_scan(ins["x"], ins["delta"], ins["A_log"], ins["B_t"], ins["C_t"],
                               **kw2, return_state=True)[:2]
                ((y.double() * gy).sum() + (h.double() * gh).sum()).backward()
                got = {**{n: ins[n].grad for n in ins}, **{k: v.grad for k, v in kw2.items()
                                                         if torch.is_tensor(v) and v.grad is not None}}
                _score(names, leaves, got, groups, arm, worst, pooled)
                del y, h
            torch.cuda.empty_cache()
        _report(a, law, depth, worst, pooled, report)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1)


def _score(names, leaves, got, groups, arm, worst, pooled):
    for n in names:
        ref = leaves[n].grad.double()
        out = got[n].double().reshape(ref.shape)
        # head axis: A_log/mu_gain (H,P,N) -> 0; delta (B,H,L) -> 1; a_dyn/mu per row -> none
        hax = 0 if n in ("A_log", "mu_gain") else (1 if n == "delta" else None)
        for grp in ("sat", "slow", "ord", "all"):
            hs = [h_ for h_, (gname, _) in groups.items() if grp in (gname, "all")]
            if hax is None:
                if grp != "all":
                    continue
                r_, o_ = ref, out
            else:
                r_, o_ = ref.index_select(hax, torch.tensor(hs, device=ref.device)), \
                    out.index_select(hax, torch.tensor(hs, device=ref.device))
            err = o_ - r_
            den = r_.norm().item() or 1e-300
            rel = err.norm().item() / den
            sig = (err * r_).sum().item() / (den * den)
            key = (arm, n, grp)
            worst[key] = max(worst.get(key, 0.0), rel if hax is None else max(
                ((o_.select(hax, i) - r_.select(hax, i)).norm()
                 / (r_.select(hax, i).norm() + 1e-300)).item()
                for i in range(len(hs))))
            pooled.setdefault(key, []).append(sig)


def _report(a, law, depth, worst, pooled, report):
    for key in sorted(worst):
        arm, n, grp = key
        if arm == "seq32":
            continue
        b = _bound(arm, depth)
        floor = worst.get(("seq32", n, grp), 0.0)
        lim = max(b, 4.0 * floor) if arm == "ieee" else b
        ps = sum(pooled[key]) / len(pooled[key])
        fl = pooled.get(("seq32", n, grp), [0.0])
        fsig = abs(sum(fl) / len(fl))           # the fp32 recurrence's own pooled bias
        ok = worst[key] <= lim and abs(ps) <= max(b / 4, 2.0 * fsig if arm == "ieee" else 0.0)
        row = dict(tag=a.tag, geom=a.geom, law=law, arm=arm, tensor=n, group=grp,
                   worst_head_rel=worst[key], fp32_seq_floor=floor, pooled_signed=ps,
                   signs=[1 if x > 0 else -1 for x in pooled[key]], bound=lim, ok=ok)
        report.append(row)
        print(f"HMSAT {a.tag} {a.geom} law={law} {arm:4s} {n:7s} {grp:4s} "
              f"worst_head_rel={worst[key]:.2e} seq32_floor={floor:.2e} pooled_signed={ps:+.2e} seq32_signed={fsig:.1e} "
              f"signs={''.join('+' if x > 0 else '-' for x in pooled[key])} "
              f"limit={lim:.1e} {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
