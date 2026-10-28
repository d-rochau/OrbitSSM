"""MH / HybridMamba on the torch route: the cells of the torch-route checks (tests/_contract/torch_route.py).

Two kinds of cell, because the module and the scan function are different torch-route surfaces:

  scan   `mh_torch_scan` (the chunked torch recurrence behind `HybridMambaSSM(backend="torch")` and the CPU fallback of
         `mh_scan`) on the kernel adapter's inputs (`MH.build`, the same option space minus the launch factors), forward
         and EVERY gradient against `reference.mh_reference` (the explicit per-token fp64 recurrence) under the family's
         terms-scaled rule (`mh_oracle.assert_oracle_close`: fp32 terms-scaled; 16-bit tol16 over >= 5 draws with the pooled
         signed bias).  The module does not route every option through this function on CPU (the D skip, the delta bias and
         the mu carry between time tiles reach it only here), so the scan is scored directly.
  layer  `HybridMambaSSM` on `backend="torch"`: the State / step / dtype checks (continuation, step loop, default carries)
         against the fp64 step recurrence of the same layer.  There is no layer-level independent oracle for HybridMamba
         (its prologue is the family's own); the layer's scan arithmetic is held by the scan cells.
"""
from __future__ import annotations

import warnings

import torch

from .. import torch_route as TR
from ..core import Cell, Factor, Space, cell_name, covering_set

B2 = (False, True)

#: scan cells: the kernel adapter's option space without the launch factors (tier, forward pin) and the kernel route
#: (fold / in_scan: both end in the same torch scan); small enough for the python-loop fp64 oracle
SCAN_GEOM = {
    "fp32": dict(dtype=torch.float32, b=2, h=4, p=4, n=16, length=65),
    "fp64": dict(dtype=torch.float64, b=2, h=4, p=4, n=16, length=65),
    "bf16": dict(dtype=torch.bfloat16, b=2, h=4, p=4, n=16, length=65),
}


#: the dtypes of the torch route cells
DTYPES = ("fp32", "bf16")
#: the kernel adapter's factors that have no meaning on the torch route (the compile class and the kernel route)
DROP = {"cls", "route"}


def _scan_space():
    from .mh import SPACE
    factors = {k: v for k, v in SPACE.factors.items() if k not in DROP}
    factors = {"dtype": DTYPES, **factors}
    groups = [[n for n in g if n not in DROP] for g in SPACE.groups]
    groups = [g for g in groups if len(g) > 1]
    return Space(factors, groups=groups)


def scan_cells():
    from ..core import file_cached
    sp = _scan_space()
    cfgs = file_cached("mh-torch-scan-cells", ["tests/_contract/adapters/mh_torch.py", "tests/_contract/adapters/mh.py"],
                       lambda: covering_set(sp, seed=20261016, triples=False).configs)
    return [Cell("mh", "scan", cell_name(sp, c), tuple(sorted(c.items()))) for c in cfgs]


def scan_check(cell):
    """`mh_torch_scan` against `mh_reference`: outputs and every gradient, per head, under the family's rule."""
    from orbit_ssm.kernels.hybrid_mamba_mh.torch_scan import mh_torch_scan
    from _tolerance import DrawPool, draw_seeds
    from .. import mh_oracle as O
    from .mh import MH

    c = cell.c
    geom = SCAN_GEOM[c["dtype"]]
    pool = DrawPool()
    for seed in draw_seeds(geom["dtype"], 591):
        args, kw = MH.build(dict(c, cls="f32"), seed=seed, device="cpu", geometry=geom)
        kw = {k: v for k, v in kw.items() if k != "return_state"}
        out = mh_torch_scan(*args, **kw)
        leaves = MH.leaves(args, kw)
        weights = [_w(v, seed + i) for i, v in enumerate(out)]
        grads = torch.autograd.grad(list(out), leaves, weights, allow_unused=True)
        for t in list(out) + [g for g in grads if g is not None]:
            assert torch.isfinite(t).all(), f"{cell.id}: non-finite"
        res = O.oracle_terms(args, kw, weights)
        O.assert_oracle_close(O.oracle_pairs(list(out), grads, res, O.leaf_names(kw), out_names=("y", "h_final")),
                              O.mh_dims(args), O.amplification(args, kw), geom["dtype"], pool=pool,
                              label=f"{cell.id} [seed {seed}] ")
    pool.assert_unbiased()


def _w(v, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(v.shape, generator=g, dtype=torch.float64).to(v.dtype)


# ---------------------------------------------------------------------------
# the layer cells
# ---------------------------------------------------------------------------
_FLAT = {"decay": frozenset({"head_dynamic", "head_static"})}
LAYER_SPACE = Space({
    "dtype": DTYPES,
    "decay": ("head_dynamic", "head_static", "coord_static"),
    "complex": Factor(B2, ({"decay": frozenset({"head_dynamic", "coord_static"})},)),
    "write": Factor(("trapezoid", "euler"), recipe_only=True),
    "source": Factor(("main_branch", "layer_input"), recipe_only=True),
    "conv": Factor((True, False), recipe_only=True),
    # the flat rows (head_dynamic / head_static) serve per-head biases and no map gates (config_spec / the constructor)
    "gates": Factor(B2, domains=((_FLAT, (False,)),)),
    "gate_act": Factor(("sigmoid", "one_plus_tanh"), ({"gates": frozenset({True})},)),
    "bias": Factor(("head", "shared", "none"), domains=((_FLAT, ("head",)),), recipe_only=True),
    "skip": Factor((True, False), recipe_only=True),
    "out_norm": Factor(B2, recipe_only=True),
}, groups=[["decay", "complex", "write", "dtype"], ["source", "conv", "gates"]])

D_MODEL, D_STATE, HEADS = 16, 16, 2


def _rows(c):
    row = {"rows": D_STATE, "decay": c["decay"], "complex": c["complex"] is True}
    if row["complex"] or c["gates"] is True:
        row["gate"] = c["gates"] is True
    return [row]


def _layer(c, dtype, seed):
    from orbit_ssm import HybridMambaConfig, HybridMambaSSM
    flat = c["decay"] in ("head_dynamic", "head_static")
    kw = dict(d_model=D_MODEL, d_state=D_STATE, d_inner=D_MODEL, n_heads=HEADS, row_blocks=_rows(c),
              proj_source=c["source"], d_conv=4 if c["conv"] else 0, bc_bias=c["bias"],
              skip=bool(c["skip"]), out_norm=bool(c["out_norm"]))
    if not flat:        # a flat law fixes its own write (head_dynamic trapezoid, head_static euler): the config refuses a write_rule
        kw["write_rule"] = c["write"]
    if c["gates"] is True:
        # the in-scan ("post") gate is the sigmoid; the one-plus-tanh spelling is the host-side pre-rotation gate
        kw.update(b_gate=True, c_gate=True, bc_gate_act=c["gate_act"],
                  bc_gate_pos="post" if c["gate_act"] == "sigmoid" else "pre")
    if flat and c["complex"] is True:
        kw.update(learn_theta_freq=False, use_theta_h=False)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = HybridMambaSSM(HybridMambaConfig(**kw), execution={"backend": "torch", "chunk_size": "auto"})
    TR.perturb(m, seed, 0.2)
    return m.to(dtype)


def layer(c, dtype, seed):
    L = 37
    return TR.TorchLayer(_layer(c, dtype, seed), d_in=D_MODEL, oracle=None, scan_depth=L + D_STATE + 8, B=2, L=L)


def layer_cells():
    from ..core import file_cached
    cfgs = file_cached("mh-torch-layer-cells", ["tests/_contract/adapters/mh_torch.py"],
                       lambda: covering_set(LAYER_SPACE, seed=20261016, triples=False).configs)
    return ([Cell("mh", "cover", cell_name(LAYER_SPACE, c), tuple(sorted(c.items()))) for c in cfgs]
            + [Cell("mh", "regression", n, tuple(sorted(c.items()))) for n, c in REGRESSION_CELLS.items()])


_REG = dict(dtype="bf16", decay="coord_static", complex=True, conv=True, gates=True, bias="head", skip=True, out_norm=False)
#: Named layer cells.  step_decay_*: the bf16 single-token step formed the clock and the per-token decay factor in the IO dtype
#: (one bf16 rounding of the whole fp32 carry per token; the parallel forward forms them in fp32).  On these three cells the
#: step loop's final hidden state then cancelled to ~0 against its terms and missed the terms form c = IO_ROUNDINGS = 6 by
#: 7-10x (5 draws); with the clock and decay in the carry's working dtype they score <= 0.011.
REGRESSION_CELLS = {
    "step_decay_trap_layer_input_one_plus_tanh": dict(_REG, write="trapezoid", source="layer_input", gate_act="one_plus_tanh"),
    "step_decay_trap_layer_input_sigmoid": dict(_REG, write="trapezoid", source="layer_input", gate_act="sigmoid"),
    "step_decay_euler_main_branch_sigmoid": dict(_REG, write="euler", source="main_branch", gate_act="sigmoid"),
}


def state_dtype_exceptions(c, route):
    """Diagnosed deviations from "a returned carry has the dtype of the supplied one" (torch_route._assert_dtypes).
    None since the returned-carry rule (StateLifecycle._conform_carries): conv_state, mu and write_prev keep
    the supplied dtype on forward and step."""
    return {}


def known_failure(c, check):
    """Declared failures of the torch-route checks: none."""
    return None


#: how many mixed-precision twins the cells carry (None: one per bf16 layer cell)
AMP_TWINS = 1
_CELLS = []


def cells():
    if not _CELLS:
        _CELLS.extend(scan_cells() + layer_cells())
        _CELLS.extend(TR.amp_twins(_CELLS, limit=AMP_TWINS))    # the mixed-precision twins of the bf16 layer cells
    return _CELLS
