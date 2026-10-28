"""MH (HybridMamba `mh_scan`) adapter of the contract suite.

Feature space = the `mh_scan` options,
with the host rules the launcher enforces as constraints; plus the compile
class and the training regime as factors.  Oracle = `reference.mh_reference`
(the explicit per-token recurrence, fp64) scored per head under A7
(`_contract.mh_oracle.assert_oracle_close`: fp32 terms-scaled; 16-bit tol16
over >= 5 draws with a pooled bias check).
"""
from __future__ import annotations

import torch

from .. import register
from ..core import Factor, Space

B2 = (False, True)
DYN, MU = frozenset({2, 3, 6, 7}), frozenset({4, 5, 6, 7})
T = frozenset({True})

CLASSES = {
    # fp32 IO, small, odd L (neither a multiple of 16 nor of any BLOCK_L)
    "f32": dict(dtype=torch.float32, b=2, h=4, p=4, n=16, length=65),
    # bf16 IO at the production head width class (fold serves P >= 16)
    "bf16": dict(dtype=torch.bfloat16, b=2, h=4, p=16, n=16, length=130),
}
#: production geometry of the shipped recipes (C13/C15 LM cell), used only by
#: test_recipes_fit: the recipe must run on the device, OOM there is a red
PROD = dict(dtype=torch.bfloat16, b=8, h=32, p=64, n=128, length=2048)


def _infeasible(c, device=None):
    return None


def regime_delta(c, delta, g, b, h, length):
    """Hook: the training regime of the clock (applied after the raw-clock shift).  The base space draws the plain case."""
    return delta


def regime_theta_gain(c):
    """Hook: the scale factor of the rotation angles' regime."""
    return 1.0


def _route(c):
    """The route mh_scan must take for this configuration (functions.LAST_ROUTE)."""
    if c["rot"] is not True:
        return "real"
    if c["route"] == "in_scan":
        r = "in_scan"
    elif c["theta_ph"]:
        r = "in_scan(per-head theta (B,H,K,L))"
    else:
        r = "fold"
    # the mixed-launch split: a leading rotating window, no carries / state return
    if (c["rot_rows"] == "part" and not c["ret_state"] and c["write_prev"] is not True
            and c["mu_init"] is not True):
        return "split+" + r
    return r


SPACE = Space({
    "cls": tuple(CLASSES),
    "laws": (1, 3, 7),                                    # bitmask A | DYN | MU over row pairs
    "softplus": Factor(B2, recipe_only=True),
    "dbias": B2,
    "dyn_act": Factor((1,), ({"laws": DYN},)),            # layout SOFTPLUS
    "a_dyn_res": Factor(("n",), ({"laws": DYN},)),
    "mu_res": Factor(("n",), ({"laws": MU},)),
    "mu_init": Factor(B2, ({"laws": MU},)),
    "rot": B2,
    "rot_rows": Factor(("all", "part"), ({"rot": T},)),
    "route": Factor(("fold",), ({"rot": T},)),
    "theta_delta": Factor(B2, ({"rot": T},), recipe_only=True),
    "theta_h": Factor(B2, ({"rot": T},)),
    "theta_ph": Factor((False,), ({"rot": T},)),
    "theta_init": Factor((False,), ({"rot": T},)),
    "gB": B2,
    "gC": Factor((False,)),
    "gate_act": Factor((1, 0), ({"gB": T}, {"gC": T})),                 # sigmoid (the shipped gate), raw
    "bgB": Factor(B2, ({"gB": T},), recipe_only=True),
    "bgC": Factor((False,), ({"gC": T},)),
    "g_g": Factor(("head", "grp"), ({"bgB": T}, {"bgC": T}), recipe_only=True),
    "bB": B2,
    "bC": B2,
    "g_bc": Factor(("head", "grp"), ({"bB": T}, {"bC": T}), recipe_only=True),
    "bA": B2,
    "g_a": Factor(("head", "grp"), ({"bA": T},), recipe_only=True),
    "trap": B2,
    "dw": Factor(("none", "h"), recipe_only=True),
    "write_prev": Factor(B2, ({"trap": T},), recipe_only=True),
    "dskip": B2,
    "z": B2,
    "h_init": B2,
    "ret_state": B2,
}, groups=[
    ["laws", "softplus", "dbias", "dyn_act", "a_dyn_res", "mu_res", "mu_init"],
    ["rot", "rot_rows", "route", "theta_delta", "theta_h", "theta_ph", "theta_init"],
    ["gB", "gC", "gate_act", "bgB", "bgC", "g_g"],
    ["bB", "bC", "g_bc", "bA", "g_a"],
    ["trap", "dw", "write_prev", "softplus", "dbias"],
    ["dskip", "z", "h_init", "ret_state", "rot"],
    ["cls", "rot", "laws"],
], infeasible=_infeasible, route=_route)


class MH:
    family = "mh"
    space = SPACE

    # -- the torch route (CPU; tests/_contract/torch_route.py) -------------------
    @staticmethod
    def torch_cells():
        from . import mh_torch
        return mh_torch.cells()

    @staticmethod
    def torch_scan_check(cell):
        from . import mh_torch
        return mh_torch.scan_check(cell)

    @staticmethod
    def torch_layer(c, dtype, seed):
        from . import mh_torch
        return mh_torch.layer(c, dtype, seed)

    @staticmethod
    def torch_known_failure(c, check):
        from . import mh_torch
        return mh_torch.known_failure(c, check) if "source" in c else None

    @staticmethod
    def torch_state_dtype_exceptions(c, route):
        from . import mh_torch
        return mh_torch.state_dtype_exceptions(c, route) if c.get("kind") != "scan" and "source" in c else {}

    @staticmethod
    def recipes():
        """Kernel-level images of the shipped HM presets (the LM presets)."""
        lm = dict(cls="bf16", dbias=True, softplus=True, bB=True, bC=True, g_bc="head",
                  trap=True, dskip=True, z=True)
        return [
            ("m1exact", dict(cls="bf16", laws=1, softplus=True, dbias=True, dskip=True, z=True)),
            ("c13_a_dyn", dict(lm, laws=3, dyn_act=1, a_dyn_res="n")),
            ("c15_a_dyn_cplx25", dict(lm, laws=3, dyn_act=1, a_dyn_res="n", rot=True,
                                      rot_rows="part", route="fold", theta_delta=True,
                                      theta_h=True)),
            ("lm_real_bias", dict(lm, laws=1)),
        ]

    @staticmethod
    def env(c):
        e = {"HYBRID_MH_AUTOTUNE": "pinned"}          # the shipped fixed launch configuration: no search
        if c["rot"] is True:
            e.update(HYBRID_MH_ROTATION=c["route"], MH_FOLD_MIN_P="1")
        return e

    @staticmethod
    def draws(c):
        from _tolerance import draw_seeds
        return draw_seeds(CLASSES[c["cls"]]["dtype"], 591)

    @staticmethod
    def build(c, seed=591, device="cuda", geometry=None):
        """mh_scan (args, kw).  Every factor of the decay modulus (A_log,
        a_dyn, mu, mu_gain, gates and their biases, beta_a) is pair-tied, as
        rotating pairs require (layout.assert_pair_tied); B/C and their
        additive biases are not."""
        k_ = geometry or CLASSES[c["cls"]]
        dtype, b, h, p, n, length = (k_[x] for x in ("dtype", "b", "h", "p", "n", "length"))
        g = torch.Generator(device="cpu").manual_seed(seed)

        def raw(*shape, scale=0.15):
            return torch.randn(*shape, generator=g, dtype=torch.float32) * scale

        def leaf(v, tied=False):
            if tied:
                v = v.repeat_interleave(2, -1)
            return v.to(device=device, dtype=dtype).requires_grad_()

        def tied_rows(*shape, scale=0.15):            # (..., N, L), tied along N
            v = raw(*shape[:-2], shape[-2] // 2, shape[-1], scale=scale)
            return leaf(v.repeat_interleave(2, -2))

        groups = {"head": h, "grp": 2}
        delta = raw(b, h, length)
        if not c["softplus"]:
            delta = delta + 0.6                        # a raw clock must stay positive
        delta = regime_delta(c, delta, g, b, h, length)
        args = [leaf(raw(b, h, p, length)), leaf(delta), leaf(raw(h, p, n // 2), tied=True),
                leaf(raw(b, n, length)), leaf(raw(b, n, length))]
        kw = dict(delta_softplus=c["softplus"], return_state=c["ret_state"])
        if c["dbias"]:
            kw["delta_bias"] = leaf(raw(h, scale=0.1))
        laws = [x for x in (1, 2, 4) if c["laws"] & x]
        if laws != [1]:
            code = {1: 0, 2: 1, 4: 2}
            pairs = [code[laws[i % len(laws)]] for i in range(n // 2)]
            kw["row_law"] = torch.tensor(pairs, dtype=torch.int32, device=device).repeat_interleave(2)
        if c["laws"] & 2:
            kw.update(dyn_act=c["dyn_act"], a_dyn_res=c["a_dyn_res"])
            kw["a_dyn"] = tied_rows(b, n, length) if c["a_dyn_res"] == "n" else leaf(raw(b, h, length))
        if c["laws"] & 4:
            kw["mu_res"] = c["mu_res"]
            kw["mu"] = (tied_rows(b, n, length, scale=0.03) if c["mu_res"] == "n"
                        else leaf(raw(b, h, length, scale=0.03)))
            kw["mu_gain"] = leaf(raw(h, p, n // 2, scale=0.1), tied=True)
            if c["mu_init"]:
                kw["mu_init"] = (leaf(raw(b, n // 2, scale=0.03), tied=True) if c["mu_res"] == "n"
                                 else leaf(raw(b, h, scale=0.03)))
        if c["rot"] is True:
            k = n // 2
            th_scale = 0.03 * regime_theta_gain(c)
            kw["theta"] = leaf(raw(*((b, h, k, length) if c["theta_ph"] else (b, k, length)), scale=th_scale))
            kw["theta_delta"] = c["theta_delta"]
            if c["theta_h"]:
                kw["theta_h"] = leaf(raw(h, k))
            if c["theta_init"]:
                kw["theta_init"] = leaf(raw(b, h, k))
            if c["rot_rows"] == "part":
                rr = torch.zeros(k, dtype=torch.bool)
                rr[: max(1, k // 4)] = True             # leading 25 % (C15-like)
                kw["row_rot"] = rr.repeat_interleave(2).to(device)
        if c["gB"] or c["gC"]:
            kw["gate_act"] = c["gate_act"]
        if c["gB"]:
            kw["g_B"] = tied_rows(b, n, length)
            if c["bgB"] is True:
                kw["beta_gB"] = leaf(raw(groups[c["g_g"]], n // 2), tied=True)
        if c["gC"]:
            kw["g_C"] = tied_rows(b, n, length)
            if c["bgC"] is True:
                kw["beta_gC"] = leaf(raw(groups[c["g_g"]], n // 2), tied=True)
        if c["bB"]:
            kw["beta_b"] = leaf(raw(groups[c["g_bc"]], n))
        if c["bC"]:
            kw["beta_c"] = leaf(raw(groups[c["g_bc"]], n))
        if c["bA"]:
            kw["beta_a"] = leaf(raw(groups[c["g_a"]], n // 2, scale=0.1), tied=True)
        if c["dw"] != "none":
            dw = raw(b, h, length) if c["dw"] == "h" else raw(b, h, p, length)
            if not c["softplus"]:
                dw = dw + 0.6
            kw["delta_w"] = leaf(dw)
        if c["trap"]:
            kw["trap_lam"] = leaf(torch.rand(b, h, length, generator=g))
            if c["write_prev"] is True:
                kw["write_prev"] = leaf(raw(b, h, p, n, scale=0.03))
        if c["dskip"]:
            kw["d_skip"] = leaf(raw(h))
        if c["z"]:
            kw["z"] = leaf(raw(b, h, p, length))
        if c["h_init"]:
            kw["h_init"] = leaf(raw(b, h, p, n))
        return args, kw

    @staticmethod
    def call(args, kw, **extra):
        from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
        out = mh_scan(*args, **kw, **extra)
        return list(out[:2]) if kw.get("return_state") or extra.get("return_state") else [out]

    # -- refusal paths (tests/test_contract_refusal.py) --------------------------------------------------------------
    @staticmethod
    def inert_options(c):
        """{option: values}: options with NO meaning in this cell that a caller may still carry.  The module passes its
        gate activation setting on every call, gated or not (C15 spells it `one_plus_tanh` with both gates off), so an
        ungated call must accept every activation code, including the host-only one no kernel implements."""
        if c["gB"] or c["gC"]:
            return {}
        from orbit_ssm.kernels.hybrid_mamba_mh import layout
        return {"gate_act": sorted(set(layout.GATE_NAMES.values()))}

    @staticmethod
    def with_inert(c, args, kw, name, value):
        return args, dict(kw, **{name: value})

    @classmethod
    def entries(cls, c, args, kw):
        """[(label, thunk(args, kw) -> [tensors])]: every public entry that takes the cell's options.  `half_fold` is
        the host fold of the rotating-pair maps; it is called directly by the backward's re-fold and by tests."""
        out = [("mh_scan", cls.call)]
        if c["rot"] is True and c["theta_ph"] is not True:
            out.append(("half_fold", cls._half_fold))
        return out

    @staticmethod
    def _half_fold(args, kw):
        from orbit_ssm.kernels.hybrid_mamba_mh.half_fold import half_fold
        fold_kw = {k: kw[k] for k in ("theta_h", "theta_init", "beta_b", "beta_c", "g_B", "g_C", "beta_gB", "beta_gC",
                                      "delta_bias", "row_rot", "theta_delta", "delta_softplus", "gate_act") if k in kw}
        return list(half_fold(args[3], args[4], args[1], kw["theta"], **fold_kw))

    @staticmethod
    def refusals():
        """[(label, fixed factors, mutate(args, kw) -> (args, kw), exception type, message regex)]: requests the launcher
        must REFUSE BY NAME before it launches anything (so they run on CPU tensors).  Each is a combination of options
        that cannot be served and that the space does not generate."""
        def kw_(**add):
            return lambda a, k: (a, dict(k, **add))

        def zeros(*shape):
            return torch.zeros(*shape)

        L = CLASSES["f32"]["length"]
        b, h, p, n = (CLASSES["f32"][x] for x in ("b", "h", "p", "n"))
        plain = dict(cls="f32", rot=False, gB=False, gC=False, trap=False)
        return [
            ("write_prev without trap_lam", plain, kw_(write_prev=zeros(b, h, p, n)), ValueError, "write_prev requires trap_lam"),
            ("theta_delta without theta", plain, kw_(theta_delta=True), ValueError, "theta_delta requires theta"),
            ("theta_h without theta", plain, kw_(theta_h=zeros(h, n // 2)), ValueError, "theta_h requires theta"),
            ("theta_init without theta", plain, kw_(theta_init=zeros(b, h, n // 2)), ValueError, "needs theta"),
            ("theta of the wrong shape", dict(plain, rot=True), kw_(theta=zeros(b, n // 2 + 1, L)), ValueError,
             "theta must be"),
            ("trap_lam of the wrong shape", dict(plain, trap=True), kw_(trap_lam=zeros(b, h, L + 1)), ValueError,
             "trap_lam must have delta"),
            ("a gate activation no kernel implements",
             dict(plain, gB=True, gate_act=1), kw_(gate_act=6), ValueError, "no kernel implementation"),
        ]

    @staticmethod
    def served_route():
        from orbit_ssm.kernels.hybrid_mamba_mh import functions
        return functions.LAST_ROUTE

    @staticmethod
    def leaves(args, kw):
        return args + [v for v in kw.values() if torch.is_tensor(v) and v.requires_grad]

    @staticmethod
    def oracle(c, args, kw, outs, grads, pool=None, label=""):
        from _contract.mh_oracle import (amplification, assert_oracle_close, leaf_names,
                                       mh_dims, oracle_pairs, oracle_terms)
        okw = {k: v for k, v in kw.items() if k != "return_state"}
        res = oracle_terms(args, okw, outs.weights + [None] * (2 - len(outs.weights)))
        names = ("y", "h_final")[:len(outs.values)]
        assert_oracle_close(oracle_pairs(outs.values, grads, res, leaf_names(kw), out_names=names),
                            mh_dims(args), amplification(args, okw), CLASSES[c["cls"]]["dtype"],
                            pool=pool, label=label)

    pinned_env = {"HYBRID_MH_AUTOTUNE": "pinned"}
    #: keys that select the launch mode; unset in the production slice (B)
    mode_keys = ("HYBRID_MH_AUTOTUNE", "HYBRID_MH_FWD_CONFIG", "HYBRID_MH_BWD_CONFIG")
    #: tensors known to vary run to run, WITH the measured reason
    #: (criterion C).  Measured, not assumed (Blackwell, 24 cells x 3
    #: processes x 10 repeats in default mode): exactly the gradients of the
    #: HEAD-SHARED tensors vary (up to 30 distinct results of 30); the
    #: forward, h_final and every per-head gradient are bitwise identical.
    #: These gradients sum over heads with relaxed cross-CTA atomics (BCS /
    #: vector-red fold, the default); under the
    #: determinism pin (mem AGX=0, BCS=0) they are bitwise (test_determinism).
    nondeterministic = {
        n: "head-shared tensor: its gradient sums over heads with relaxed "
           "atomics (the default); bitwise under the determinism pin"
        for n in ("grad:B_t", "grad:C_t", "grad:g_B", "grad:g_C", "grad:beta_b",
                  "grad:beta_c", "grad:beta_gB", "grad:beta_gC", "grad:theta")}
    #: Measured second pass (Blackwell): grad:x varies (4 / 15 of 30)
    #: where the light search picks a backward that splits the state rows
    #: across CTAs and adds their x gradients atomically (ATOMIC_GX, the AGX
    #: of the determinism pin) -- config-dependent, so only some cells show it.
    #: Measured third pass (Blackwell, contract compact gate, layout / poison / retain-graph checks of the
    #: laws = 7, rotating-fold cell): grad:a_dyn varies between two identical default-mode runs; under the
    #: determinism pin (AGX = BCS = THS = 0, test_determinism) it is bitwise like the rest
    nondeterministic["grad:a_dyn"] = ("tied-row dynamic-rate gradient of the rotating fold route accumulated with the "
                                      "relaxed atomics the determinism pin switches off")
    nondeterministic["grad:x"] = ("row-split backward (ATOMIC_GX) chosen by the "
                                  "search adds x gradients atomically; bitwise "
                                  "under the determinism pin (AGX=0)")

    @staticmethod
    def tensor_names(args, kw):
        from _contract.mh_oracle import leaf_names
        outs = ["y", "h_final"] if kw.get("return_state") else ["y"]
        return outs + ["grad:" + n for n in leaf_names(kw)]

    @staticmethod
    def check_forward(c, args, kw, values, label=""):
        """Forward values only (no gradients) against the fp64 oracle."""
        from _contract.mh_oracle import amplification, assert_oracle_close, mh_dims, oracle_terms
        okw = {k: v for k, v in kw.items() if k != "return_state"}
        res = oracle_terms(args, okw, [None, None])
        names = ("y", "h_final")[:len(values)]
        pairs = list(zip(names, values, res.out, res.terms))
        assert_oracle_close(pairs, mh_dims(args), amplification(args, okw),
                            CLASSES[c["cls"]]["dtype"], label=label)

    @staticmethod
    def determinism_kw():
        """mh_scan options under which the backward is bit-reproducible."""
        return dict(mem={"AGX": 0, "BCS": 0, "THS": 0})

    @staticmethod
    def continuation(c):
        """Whether the cell has a forward state-carry check: every carry the
        call needs is returned or reconstructible (trap's write_prev is a
        decode-side quantity, not handed back -- excluded)."""
        return not c["trap"]

    @staticmethod
    def split_kw(c, args, kw, cut, first_out):
        """(args, kw) of the second half given the first half's returned
        (y, h_final, theta_final)."""
        def tail(name, v):
            if not torch.is_tensor(v) or v.dim() == 0:
                return v
            if name in ("x", "delta", "B_t", "C_t", "a_dyn", "mu", "g_B", "g_C", "theta",
                        "delta_w", "z", "trap_lam"):
                return v[..., cut:]
            return v
        names = ("x", "delta", "A_log", "B_t", "C_t")
        a2 = [tail(nm, v) for nm, v in zip(names, args)]
        k2 = {nm: tail(nm, v) for nm, v in kw.items()}
        _, h_fin, th_fin = first_out
        k2["h_init"] = h_fin
        if "theta" in kw:
            k2["theta_init"] = th_fin
        if "mu" in kw:
            mu = kw["mu"]
            k2["mu_init"] = mu[..., cut - 1] if kw.get("mu_res") == "n" else mu[..., cut - 1]
        return a2, k2

    @classmethod
    def check_continuation(cls, c, label=""):
        """mh_scan over L == two calls joined at `cut` by the returned
        (h_final, theta_final) and the raw mu carry; both halves scored
        against the fp64 oracle of the whole call, terms doubled (two arms of
        the same summands, tests/_tolerance.py)."""
        from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
        from _contract.mh_oracle import amplification, assert_oracle_close, mh_dims, oracle_terms
        args, kw = cls.build(c)
        kw = dict(kw, return_state=True)
        L = args[0].shape[-1]
        cut = L // 2 + 1
        timed = ("a_dyn", "mu", "g_B", "g_C", "theta", "delta_w", "z", "trap_lam")
        with torch.no_grad():
            a1 = [v[..., :cut] if i in (0, 1, 3, 4) else v for i, v in enumerate(args)]
            k1 = {n: (v[..., :cut] if n in timed and torch.is_tensor(v) else v) for n, v in kw.items()}
            first = mh_scan(*a1, **k1)
            a2, k2 = cls.split_kw(c, args, kw, cut, first)
            second = mh_scan(*a2, **k2)
        y = torch.cat([first[0], second[0]], dim=-1)
        okw = {k: v for k, v in kw.items() if k != "return_state"}
        res = oracle_terms(args, okw, [None, None])
        pairs = [("y", y, res.out[0], 2 * res.terms[0]),
                 ("h_final", second[1], res.out[1], 2 * res.terms[1])]
        assert_oracle_close(pairs, mh_dims(args), amplification(args, okw),
                            CLASSES[c["cls"]]["dtype"], label=label)

    @staticmethod
    def route_pairs(c):
        return []

    @classmethod
    def build_production(cls, c):
        return cls.build(c, geometry=PROD)

    # -- hooks of the generic metamorphic checks (F) -------------------------
    double_backward = "unsupported"      # MHScan has no create_graph backward
    variant_kinds = ("views", "permuted")
    #: time is the LAST axis of every time-varying MH input
    _TIMED = ("x", "delta", "B_t", "C_t", "a_dyn", "mu", "g_B", "g_C", "theta",
              "delta_w", "z", "trap_lam")
    #: leaves whose gradient is per token (anti-causal under a truncated cotangent)
    per_token_leaves = tuple("grad:" + n for n in _TIMED if n != "theta")
    _BATCHED = _TIMED + ("theta_init", "mu_init", "write_prev", "h_init")
    _ARGN = ("x", "delta", "A_log", "B_t", "C_t")

    @staticmethod
    def zero_input_underflows(c):
        return False

    @staticmethod
    def is_sixteen_bit(c):
        return CLASSES[c["cls"]]["dtype"] != torch.float32

    @staticmethod
    def length(args, kw):
        return args[0].shape[-1]

    @staticmethod
    def causal_outputs(outs):
        return [0]                       # y; a returned h_final is not causal

    @classmethod
    def _map(cls, args, kw, fn):
        a2 = [fn(n, v) for n, v in zip(cls._ARGN, args)]
        k2 = {n: (fn(n, v) if torch.is_tensor(v) and v.is_floating_point() else v)
              for n, v in kw.items()}
        return a2, k2

    @classmethod
    def perturb_after(cls, args, kw, t0):
        def f(n, v):
            if n not in cls._TIMED:
                return v
            w = v.detach().clone()
            w[..., t0:] = torch.randn_like(w[..., t0:]) * w.float().std().to(w.dtype)
            if n == "trap_lam":
                w[..., t0:] = torch.rand_like(w[..., t0:])
            if n == "delta" or n == "delta_w":
                w[..., t0:] = w[..., t0:].abs()
            # rotating pairs need a tied modulus (layout.assert_pair_tied):
            # a tensor tied along its row axis stays tied after the change
            if v.dim() >= 2 and v.shape[-2] % 2 == 0 and torch.equal(v[..., 0::2, :], v[..., 1::2, :]):
                w[..., 1::2, t0:] = w[..., 0::2, t0:]
            return w.requires_grad_(v.requires_grad)
        return cls._map(args, kw, f)

    @classmethod
    def slice_sample(cls, args, kw, i):
        def f(n, v):
            if n not in cls._BATCHED:
                return v
            return v.detach()[i:i + 1].clone().requires_grad_(v.requires_grad)
        return cls._map(args, kw, f)

    @classmethod
    def variant(cls, args, kw, kind):
        """Same values, other layout: `views` = every tensor a slice of a
        wider NaN-filled buffer; `permuted` = time-varying tensors stored
        time-major (non-unit last-axis stride); `zero_x` / `denormal_x` =
        x exactly zero / scaled into the fp32 denormal range."""
        def view(n, v):
            big = torch.full((*v.shape[:-1], v.shape[-1] + 5), float("nan"),
                             device=v.device, dtype=v.dtype)
            big[..., :v.shape[-1]] = v.detach()
            return big[..., :v.shape[-1]].requires_grad_(v.requires_grad)

        def perm(n, v):
            if n not in cls._TIMED or v.dim() < 2:
                return v
            return v.detach().movedim(-1, 0).contiguous().movedim(0, -1).requires_grad_(v.requires_grad)

        def scale(factor):
            def f(n, v):
                if n != "x":
                    return v
                return (v.detach() * factor).requires_grad_(v.requires_grad)
            return f
        fn = {"views": view, "permuted": perm, "zero_x": scale(0.0),
              "denormal_x": scale(1e-39)}[kind]
        return cls._map(args, kw, fn)

    @staticmethod
    def close(c, label, a, b):
        """Two kernel results of the same values that may differ in tiling
        (batch size, layout): normwise relative agreement at the IO dtype's
        rounding (each arm rounds its output once, plus reordered fp32 sums),
        guarded against a near-zero reference (assert_rel_close)."""
        from _tolerance import assert_rel_close, unit_roundoff
        dt = CLASSES[c["cls"]]["dtype"]
        u = "bf16" if dt == torch.bfloat16 else "fp32"
        assert a is not None and b is not None, label
        assert_rel_close(a.float(), b.float(), 16 * unit_roundoff(u), terms=b.float().abs(),
                         u=u, name=label)

    @staticmethod
    def compiled(cell):
        return cell.kind == "recipe"


register(MH())
