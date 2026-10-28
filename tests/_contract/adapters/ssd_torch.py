"""SSD-MIMO on `backend="torch"`: the cells and the independent oracle of the torch-route checks
(tests/_contract/torch_route.py).

The layer is `SSDMimoSSM` on the chunked torch dual (chunk 16, the smallest the layer accepts: the test length 23 crosses one chunk
boundary and ends in a ragged chunk).  The oracle is `physical_oracle` (tests/_contract/ssd_physical.py: the defining recurrence written
as one explicit python loop per batch row, head, sibling and token, sharing no production map / scan / step helper),
composed with the layer's own prologue and epilogue written out in plain ops: in_proj split, the causal depthwise
x-branch conv + SiLU, the scan, the optional output RMSNorm, out_proj.

The option space is the torch-visible subset of the kernel adapter's: rank (R, O), the row layout (real / half / all
complex), the decay law, the write rule (trapezoid / euler), the x-branch and the B/C convolutions, per-head B/C biases,
map gates, map norms, the D skip and its sharing, the output norm, projection biases, the pair layout and the angle
layout.  The output gate z is always on (it is the in_proj's second half).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from .. import torch_route as TR
from ..core import Factor, Space

R_O = {"R1O1": (1, 1), "R2O2": (2, 2)}
CX = frozenset({"half", "all"})
B2 = (False, True)
L_TOK = 23                  # two chunks of 16 (one boundary) and a ragged tail of 7: the smallest length that crosses a chunk boundary
# backend="torch" has TWO sequence routes: the chunked dual for L > SCAN_MAX_L and the token scan (adjoint-recurrence backward
# `_LinearScanFn`) for shorter calls.  The route is a factor, so both are covered (the defect corpus's S02, a sign error in
# the token scan's decay adjoint, was invisible while every cell had L = 23).
from orbit_ssm.ssm.ssd_mimo_torch import SCAN_MAX_L
ROUTE_L = {"chunked": L_TOK, "token_scan": SCAN_MAX_L}
H, G, N = 2, 2, 32          # N = 32: the half pair layout needs a multiple of 32

SPACE = Space({
    "dtype": ("fp32", "bf16"),
    "route": tuple(ROUTE_L),
    "rank": ("R1O1", "R2O2"),
    "cplx": ("real", "half"),
    "decay": Factor(("head_dynamic", "head_static"), recipe_only=True),
    "write": Factor(("trapezoid", "euler"), recipe_only=True),
    "xconv": Factor((True, False), recipe_only=True),
    "bcconv": Factor(B2, ({"xconv": frozenset({True})},), recipe_only=True),
    "bias": Factor(("head", "none"), domains=(({"cplx": CX}, ("head",)),), recipe_only=True),   # a complex block needs per-head biases
    "gates": B2,
    "norms": Factor((True, False), recipe_only=True),
    "dskip": Factor((True, False), recipe_only=True),
    "skip_share": Factor(("head", "input_output"), ({"dskip": frozenset({True})},), recipe_only=True),
    "out_norm": Factor(B2, recipe_only=True),
    "proj_bias": Factor((True, False), recipe_only=True),
    "a_sib": Factor(B2, ({"decay": frozenset({"head_dynamic"}), "rank": frozenset({"R2O2"})},), recipe_only=True),
    "layout": Factor(("interleaved", "half"), ({"cplx": CX},), recipe_only=True),
    "omega": Factor(("pair", "head_pair", "head"), ({"cplx": CX},), recipe_only=True),
    "theta_h": Factor(B2, ({"cplx": CX},), recipe_only=True),
}, groups=[["rank", "cplx", "decay", "write", "dtype", "route"], ["cplx", "theta_h"],
           ["gates", "dskip", "out_norm"]])


def _rows(c):
    law = c["decay"]
    if c["cplx"] == "real":
        return [{"frac": 1.0, "decay": law, "complex": False}]
    if c["cplx"] == "half":
        return [{"frac": 0.5, "decay": law, "complex": True}, {"frac": 0.5, "decay": law, "complex": False}]
    blk = {"frac": 1.0, "decay": law, "complex": True}
    if c["gates"]:
        blk["gate"] = True
    return [blk]


def _layer(c, dtype, seed):
    from orbit_ssm.ssm.ssd_mimo import SSDMimoConfig, SSDMimoSSM
    R, O = R_O[c["rank"]]
    cx = c["cplx"] != "real"
    kw = dict(d_model=8, d_state=N, d_inner=H * G * R, d_ssm_in=R, d_ssm_out=O, n_groups=G, n_heads=H,
              row_blocks=_rows(c), write_rule=c["write"], d_conv=4 if c["xconv"] else 0, bc_conv=c["bcconv"] is True,
              bc_bias=c["bias"], b_norm=bool(c["norms"]), c_norm=bool(c["norms"]), b_gate=bool(c["gates"]),
              c_gate=bool(c["gates"]), skip=bool(c["dskip"]), out_norm=bool(c["out_norm"]),
              a_proj_bias=bool(c["proj_bias"]) or c["decay"] != "head_dynamic", trap_proj_bias=bool(c["proj_bias"]) or c["write"] != "trapezoid",
              a_dyn_per_sibling=c["a_sib"] is True, proj_source="layer_input")
    if c["dskip"]:
        kw["skip_sharing"] = c["skip_share"]
    if cx:
        kw.update(pair_layout=c["layout"], omega_layout=c["omega"], theta_per_head=c["omega"] == "head",
                  use_theta_h=bool(c["theta_h"]), learn_theta_freq=c["omega"] != "pair")
    torch.manual_seed(seed)
    with warnings.catch_warnings():            # the rank / gate advisories of the constructor
        warnings.simplefilter("ignore")
        m = SSDMimoSSM(SSDMimoConfig(**kw), execution={"backend": "torch", "precision": "ieee", "chunk_size": 16})
    TR.perturb(m, seed)
    for norm in (getattr(m, "B_norm", None), getattr(m, "C_norm", None)):
        if norm is not None:
            norm.eps = 1e-5                # the oracle's eps must not depend on the dtype
    return m.to(dtype)


def oracle(m, x, stats=None):
    """The layer's physical output from its parameters (see the module docstring)."""
    from ..ssd_physical import physical_oracle
    B, L, _ = x.shape
    proj = F.linear(x, m.in_proj.weight, m.in_proj.bias)
    xin, z = proj[..., :m.d_inner], proj[..., m.d_inner:]
    if m.conv1d is not None:
        k, C = m.conv1d.kernel_size[0], xin.shape[-1]
        v = F.pad(xin.transpose(1, 2), (k - 1, 0))
        xin = F.silu(F.conv1d(v, m.conv1d.weight, m.conv1d.bias, groups=C)).transpose(1, 2)
    st = {}
    y, _ = physical_oracle(m, xin, x, x.new_zeros(B, m.H, m.P, m.N), z, stats=st)
    if stats is not None:
        stats["amp"] = st["decay"] + st["phase"]
    if not isinstance(m.norm, torch.nn.Identity):
        y = m.norm.weight * y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + m.norm.eps)
    return m.out_proj(y)


def layer(c, dtype, seed):
    m = _layer(c, dtype, seed)
    R, O = R_O[c["rank"]]
    L = ROUTE_L[c["route"]]
    return TR.TorchLayer(m, d_in=8, oracle=oracle, scan_depth=2 * L * N * max(R, O) + L + 8, L=L,
                          circular=("rot_ang",))


#: how many mixed-precision twins the cells carry (None: one per bf16 layer cell)
AMP_TWINS = 1
_CELLS = []


def cells():
    if _CELLS:
        return _CELLS
    from ..core import Cell, cell_name, covering_set, file_cached
    cfgs = file_cached("ssd-torch-cells", ["tests/_contract/adapters/ssd_torch.py"],
                       lambda: covering_set(SPACE, seed=20261016, triples=False).configs)
    _CELLS.extend(Cell("ssd", "cover", cell_name(SPACE, c), tuple(sorted(c.items()))) for c in cfgs)
    _CELLS.extend(TR.amp_twins(_CELLS, limit=AMP_TWINS))        # the mixed-precision twins of the bf16 cells
    return _CELLS


def state_dtype_exceptions(c, route):
    """Diagnosed deviations from "a returned carry has the dtype of the supplied one" (see torch_route._assert_dtypes).
    None since the returned-carry rule (StateLifecycle._conform_carries)."""
    return {}