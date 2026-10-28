"""G9 S4 — what `SSD_MIMO_GATE_OUT_IO` costs in accuracy.

The gate epilogue stores `y_pre * silu(z)` from an fp32 register.  With
the lever OFF it stores fp32 and the module casts the whole buffer to the
IO dtype afterwards; with it ON the kernel rounds at its own store.  The
FORWARD is bit-identical (one round-to-nearest of the same fp32 value
either way, which this script asserts rather than assumes).  What changes
is the BACKWARD: the node's upstream gradient is then bf16, so the
epilogue's adjoint reads a rounded `dOut`.

This scores both settings against an fp64 oracle -- the same module, same
parameters, same input, run in double precision -- so "is the ON setting
at the bf16 noise floor?" is answered by a number and not by an argument.
The floor itself is printed as the OFF column: everything upstream of the
gate is already bf16 in both.

    PYTHONPATH=$PWD python -m benchmarks.g9_gateio
"""
import argparse
import os
import statistics

import torch

from orbit_ssm import SSDMimoSSM, SSDMimoConfig

# Mamba-2's exact core: one static rate per head, no rotation, no B/C
# norms or biases, no gates (the rows/rotation/maps spelling, §5.3).
M2EXACT = dict(row_blocks=[{"frac": 1.0, "decay": "head_static", "complex": False}],
               b_norm=False, c_norm=False, bc_bias="none",
               b_gate=False, c_gate=False, skip=True,
               r_mix_dyn=False, proj_source="layer_input", bc_conv=True)


def build(ssm_kw, mixer_kw, d_model, N, ngroups, seed, dtype, backend):
    torch.manual_seed(seed)
    kw = dict(d_model=d_model, d_state=N, expand=2.0, d_conv=4)
    kw.update(mixer_kw)
    core = dict(d_ssm_in=1, d_ssm_out=1, n_groups=ngroups, backend=backend,
                precision="ieee")
    core.update(ssm_kw)
    kw.update(core)  # core options are flat mixer kwargs
    ex = {k: kw.pop(k) for k in ("backend", "precision") if k in kw}
    return SSDMimoSSM(SSDMimoConfig(**kw), execution=ex).cuda().to(dtype)


def run(m, x, dy, amp):
    for p in m.parameters():
        p.grad = None
    x.grad = None
    with (torch.autocast("cuda", dtype=torch.bfloat16) if amp
          else torch.autocast("cuda", enabled=False)):
        y = m(x)
    y.backward(dy)
    return ([y.detach().double(), x.grad.detach().double()]
            + [(p.grad.detach().double() if p.grad is not None else None)
               for _, p in sorted(m.named_parameters())])


def relerr(ref, got):
    """max |got - ref| / max |ref|, the scale-free error the K-section
    grids use."""
    if ref is None or got is None:
        return 0.0
    s = max(ref.abs().max().item(), 1e-30)
    return (got - ref).abs().max().item() / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-state", type=int, default=32)
    ap.add_argument("--ngroups", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--config", default="m2exact",
                    choices=("m2exact", "defaults"))
    a = ap.parse_args()

    ssm_kw = dict(M2EXACT) if a.config == "m2exact" else dict(
        proj_source="layer_input")
    mixer_kw = {}      # the B/C maps live in the core kwargs (M2EXACT)
    names = None
    rows = {0: [], 1: []}
    fwd_bit_equal = True
    for seed in range(a.seeds):
        # ---- fp64 oracle: the torch backend, the same parameters ------
        m64 = build(ssm_kw, mixer_kw, a.d_model, a.d_state, a.ngroups,
                    seed, torch.float64, "torch")
        m64.combined = False
        torch.manual_seed(1000 + seed)
        x64 = torch.randn(a.batch, a.len, a.d_model, device="cuda",
                          dtype=torch.float64, requires_grad=True)
        dy64 = torch.randn(a.batch, a.len, a.d_model, device="cuda",
                           dtype=torch.float64)
        ref = run(m64, x64, dy64, False)
        names = (["y", "dx"]
                 + [n for n, _ in sorted(m64.named_parameters())])
        del m64

        # ---- bf16 autocast over fp32 parameters, both settings --------
        m32 = build(ssm_kw, mixer_kw, a.d_model, a.d_state, a.ngroups,
                    seed, torch.float32, "cuda")
        x32 = x64.detach().float().requires_grad_(True)
        dy32 = dy64.detach().float()
        got = {}
        for gio in (0, 1):
            os.environ["SSD_MIMO_GATE_OUT_IO"] = str(gio)
            got[gio] = run(m32, x32, dy32, True)
            rows[gio].append([relerr(r, g) for r, g in zip(ref, got[gio])])
        # BITWISE, forward and every gradient, in BOTH graph shapes -- and
        # against a REPEAT of the same setting, because the D-skip's
        # parameter gradient is an atomic reduction and therefore not
        # run-to-run reproducible: a difference is only the lever's if it
        # exceeds the same-setting control.
        for comb in (True, False):
            m32.combined = comb
            side = {}
            for tag, gio in (("0", 0), ("1", 1), ("0b", 0)):
                os.environ["SSD_MIMO_GATE_OUT_IO"] = str(gio)
                side[tag] = run(m32, x32, dy32, True)
            for nm_i, (p, q, r) in enumerate(zip(side["0"], side["1"],
                                                 side["0b"])):
                if p is None:
                    continue
                lever = torch.equal(p, q)
                ctrl = torch.equal(p, r)
                if not lever and ctrl:
                    fwd_bit_equal = False
                    print(f"  lever CHANGES tensor {names[nm_i]} "
                          f"(seed {seed}, combined={comb}); "
                          f"same-setting control is bitwise equal")
                elif not lever:
                    print(f"  nondeterministic: {names[nm_i]} "
                          f"(seed {seed}, combined={comb}) -- the "
                          f"same-setting control differs too")
            del side
        m32.combined = True
        os.environ["SSD_MIMO_GATE_OUT_IO"] = "0"
        del m32, x32, x64, ref, got

    print(f"cell B{a.batch} L{a.len} d_model {a.d_model} N {a.d_state} "
          f"ngroups {a.ngroups}, config {a.config}, "
          f"{a.seeds} seeds, bf16 autocast over fp32 parameters")
    print("forward AND every gradient bitwise identical across the "
          f"lever, combined and multi-node: {fwd_bit_equal}")
    print(f"{'tensor':<28} {'GATE_OUT_IO=0':>16} {'GATE_OUT_IO=1':>16} "
          f"{'ratio':>8}")
    worst = (0.0, 0.0)
    for i, nm in enumerate(names):
        e0 = statistics.mean(r[i] for r in rows[0])
        e1 = statistics.mean(r[i] for r in rows[1])
        worst = (max(worst[0], e0), max(worst[1], e1))
        print(f"{nm:<28} {e0:16.3e} {e1:16.3e} "
              f"{(e1 / e0 if e0 else 1.0):8.2f}")
    print(f"{'WORST':<28} {worst[0]:16.3e} {worst[1]:16.3e} "
          f"{(worst[1] / worst[0] if worst[0] else 1.0):8.2f}")


if __name__ == "__main__":
    main()
