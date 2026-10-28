"""The shared row_blocks / rotation / B-C-map spec (pure Python, no torch).

The layer families (`SSDMimoSSM`, `HybridMambaSSM` and every
family registered through `orbit_ssm.families`) resolve their state rows, rotation and B/C
maps through `resolve`; the modules then build from the canonical dict.

One input form, flat keywords (constructors, configs and saved recipes use the same names)::

    row_blocks          list of `RowBlock` or their strict dicts ``{rows | frac, decay, complex, gate, chart}``
    theta_per_head, learn_theta_freq, use_theta_h, omega_layout, theta_init      (the rotation)
    b_norm, c_norm, bc_bias, b_gate, c_gate                                     (the B/C maps)
    bc_bias_init                                                                (SSD)
    bc_gate_act, bc_gate_bias, bc_gate_pos                                      (HybridMamba)

They resolve to ONE canonical plain dict::

    {"row_blocks": [{"rows": int, "decay": str, "complex": bool, "gate": bool}, ...],
     "rotation": None | {"theta_per_head": bool, "learn_theta_freq": bool,
                         "use_theta_h": bool,
                         "omega_layout": "pair"|"head"|"head_pair",
                         "theta_init": "kaiming"|"small"|"zero"},
     "maps": {"b_norm": bool, "c_norm": bool, "bc_bias": "head"|"shared"|"none",
              "b_gate": bool, "c_gate": bool, <family extensions>}}

A dict block is `RowBlock.from_dict` (strict: an unknown key or a mistyped value is a ``TypeError``).
An omitted ``decay`` resolves to the family default (`DEFAULT_DECAY`) and is saved resolved on every block; ``"none"`` (a_t = 1)
is not a law of these two families and is refused by name.  `check_law_levers` refuses a layer-level lever of a decay law no block uses.
Unknown decay laws fail with the valid names; there are no aliases (old spellings are translated by the
repository's separate migration tooling only).  `flat_spec` is the canonical dict in the flat form (what a saved recipe
stores); `resolve` is idempotent on it.

Per-block B/C gates: ``b_gate`` / ``c_gate`` say whether the B / C gates exist; each block's ``gate``
says which rows they act on.  ``gate=False`` rows have gate factor exactly 1 (no dead parameters).
The default is `BLOCK_GATE_DEFAULT` (real blocks gated, complex not), always saved resolved.  SSD's
d_ssm_in/d_ssm_out-driven gate default resolves off when no block is gated; an explicit b_gate /
c_gate with no gated block raises.

Decay laws.  One name is one law (formulas: `DECAY_LAWS`); the same law has the same name in every family
and different laws have different names:
  "head_dynamic"       log a_t = -|A_t| Delta_t,  |A_t| selected, Delta_t = softplus(dt_t + dt_bias) selected
                       (Mamba-3; SSD, HM flat rows; the paper's "SSD dynamic" / "HM flat dynamic")
  "head_static"        log a_t = -exp(A_log_h) Delta_t,  A static per head, Delta_t selected (Mamba-2;
                       SSD, HM flat rows; the paper's "SSD static" / "HM flat static")
  "coord_static" / "coord_dynamic" / "coord_potential"
                       HM only, the per-coordinate mh_scan laws (the paper's "HM static / dynamic / potential":
                       coord_static is a static A per (h, p, n); coord_dynamic times a selected per-row rate;
                       coord_potential plus a bounded telescoping potential)

A family registered through `orbit_ssm.families` extends the spec through its `SpecHooks` (extra block keys,
row layout rules, rotation defaults, bias granularities, B/C gates); `resolve` calls them.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any
from . import families as _families
from .validation import check_divides

#: the families this module serves itself; a registered family (`orbit_ssm.families`) is added by `_on_family`
FAMILIES = ("ssd_mimo", "hybrid_mamba")

# ── decay-law vocabulary ─────────────────────────────────────────────────
# SSD: one law for the whole state.
HEAD_DYNAMIC = "head_dynamic"
HEAD_STATIC = "head_static"
SSD_DECAYS = (HEAD_DYNAMIC, HEAD_STATIC)
HM_PER_COORD = ("coord_static", "coord_dynamic", "coord_potential")          # mh_scan rows
HM_FLAT = (HEAD_DYNAMIC, HEAD_STATIC)   # ssd_mimo_scan rows, R = O = 1
HM_DECAYS = HM_PER_COORD + HM_FLAT
HM_COMPLEX_OK = HM_PER_COORD + (HEAD_DYNAMIC,)
# Per-block B/C gate default, keyed by the block's `complex` flag; change it here only.
BLOCK_GATE_DEFAULT = {False: True, True: False}
BLOCK_KEYS = ("rows", "frac", "decay", "complex", "gate")
HM_MAX_BLOCKS = 8
SSD_MAX_BLOCKS = 2
#: the decay a block resolves to when it omits ``decay`` (recorded resolved on every block); a family registered
#: through `orbit_ssm.families` states its own via its `SpecHooks`
DEFAULT_DECAY = {"ssd_mimo": HEAD_DYNAMIC, "hybrid_mamba": "coord_static"}
#: ``decay="none"`` (a_t = 1 exactly) is not a law of SSD-MIMO or HybridMamba (their scans always apply the law): refused by name
DECAY_NONE = "none"
#: per-family block keys beyond rows/frac/decay/complex (registered families: `SpecHooks.row_ext`)
ROW_EXT: dict = {}

#: The decay laws, one entry per law: the families that serve it and its per-token log-decay.  One name is one law;
#: the same law has the same name in every family.  (Delta_t = softplus(dt_proj(x_t) + dt_bias) is the selected step,
#: |A_t| a selected rate, mu_t / z_t selected potentials; "per (h,p,n)" = one static value per head, channel, row.)
DECAY_LAWS = {
    "head_dynamic": dict(families=("ssd_mimo", "hybrid_mamba"),
                         log_a="-|A_t| Delta_t, |A_t| selected per head (heavy_tail / softplus), Delta_t selected"),
    "head_static": dict(families=("ssd_mimo", "hybrid_mamba"),
                        log_a="-exp(A_log_h) Delta_t, A static per head, Delta_t selected"),
    "coord_static": dict(families=("hybrid_mamba",), log_a="-exp(A_log[h,p,n]) Delta_t, A static per coordinate"),
    "coord_dynamic": dict(families=("hybrid_mamba",),
                log_a="-exp(A_log[h,p,n]) Delta_t rate_t[n | h], rate_t selected per row (or per head)"),
    "coord_potential": dict(families=("hybrid_mamba",),
               log_a="-exp(A_log[h,p,n]) Delta_t + c[h,p,n] (mu_t - mu_{t-1}), mu = tanh(.), c = G sigmoid(.) bounded"),
}

BIAS_GRANS = ("head", "shared", "none")
SSD_BIAS_INITS = ("auto", "ones", "spread")
# layout.GATE_NAMES minus the "identity" alias, plus SSD's (1 + g) form
HM_GATE_ACTS = ("sigmoid", "exp", "tanh", "softplus", "raw", "one_plus", "one_plus_tanh")
# Where HM's B/C gates act: "pre" = before bias and rotation (SSD's placement, default),
# "post" = the in-scan gate after the rotation.
HM_GATE_POS = ("pre", "post")

ROTATION_DEFAULTS = {"theta_per_head": False, "learn_theta_freq": True,
                     "use_theta_h": True, "omega_layout": "pair",
                     "theta_init": "kaiming"}
#   omega_layout  the shape of the learned frequency omega = pi tanh(theta_freq):
#                 "pair"       (K_rot,)    one per rotated pair (the default)
#                 "head_pair"  (H, K_rot)  one per (head, pair); needs
#                              learn_theta_freq (a fixed pi has no layout)
#                 "head"       (H,)        ONE complex eigenvalue per head:
#                              every rotated pair of a head also shares ONE
#                              angle feature (W_theta emits H values), so
#                              Theta_{t,h} += Delta omega_h tanh(q_{t,h});
#                              theta_per_head resolves True (explicit False
#                              raises)
#   theta_init    the angle projection W_theta's init: "kaiming" (nn.Linear's
#                 kaiming_uniform), "small" (that draw x 0.1 sqrt(3): std ~0.1 on
#                 unit-RMS inputs, near-zero rotation with random directions),
#                 "zero" (no rotation at init)
OMEGA_LAYOUTS = ("pair", "head", "head_pair")
THETA_INITS = ("kaiming", "small", "zero")
_ROTATION_CHOICES = {"omega_layout": OMEGA_LAYOUTS, "theta_init": THETA_INITS}
_ROTATION_BOOLS = ("theta_per_head", "learn_theta_freq", "use_theta_h")

# canonical maps keys: shared core + per-family extensions
MAPS_CORE = ("b_norm", "c_norm", "bc_bias", "b_gate", "c_gate")
MAPS_EXT = {"ssd_mimo": ("bc_bias_init",),
            "hybrid_mamba": ("bc_gate_act", "bc_gate_bias", "bc_gate_pos")}
ROTATION_FIELDS = tuple(ROTATION_DEFAULTS)
#: every flat spec keyword of any family, in storage order (`row_blocks` first)
SPEC_FIELDS = ("row_blocks",) + ROTATION_FIELDS + MAPS_CORE + tuple(
    k for ext in MAPS_EXT.values() for k in ext)
#: the `SpecHooks` of the registered families
_HOOKS: dict = {}


def _on_family(fam) -> None:
    """Take a registered family's spec hooks into the tables (`orbit_ssm.families.on_register`)."""
    global FAMILIES, SPEC_FIELDS
    hooks = fam.spec
    if hooks is None or fam.name in FAMILIES:
        return
    _HOOKS[fam.name] = hooks
    FAMILIES = FAMILIES + (fam.name,)
    ROW_EXT[fam.name] = tuple(hooks.row_ext)
    MAPS_EXT[fam.name] = tuple(hooks.maps_ext)
    SPEC_FIELDS = SPEC_FIELDS + tuple(k for k in hooks.maps_ext if k not in SPEC_FIELDS)
    for law in hooks.decays:
        DECAY_LAWS[law]["families"] += (fam.name,)


_families.on_register(_on_family)

class ConfigSpecError(ValueError):
    """A spec that cannot be resolved (conflict, removed key, bad value)."""


# ── the helper type ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class RowBlock:
    """One block of state rows (an item of ``row_blocks``).

    Parameters
    ----------
    rows, frac : int, float
        Exactly one: the number of rows, or the fraction of ``d_state`` (must give an integer).
    decay : str or None
        Decay law; the valid names depend on the family (module docstring).  None (omitted) resolves to the family's
        default law (`DEFAULT_DECAY`: SSD-MIMO ``head_dynamic``, HybridMamba ``coord_static``) and is recorded resolved.
        Blocks that share one scan with one decay per head must resolve to the same law.  Paper names:
        ``head_dynamic`` = "SSD dynamic" (HM: "HM flat dynamic"), ``head_static`` = "SSD static" ("HM flat
        static"), ``coord_static`` / ``coord_dynamic`` / ``coord_potential`` = "HM static / dynamic /
        potential" (hybrid_mamba only).
    complex : bool, default False
        The block holds (re, im) pairs (even ``rows``); it rotates.
    gate : bool or None
        B/C gates act on its rows; None = `BLOCK_GATE_DEFAULT` for its ``complex`` flag.
    chart : bool or None
        Families that extend the block spec only (`SpecHooks.row_ext`): the block is mixed by the family's chart.
    tri : object or dict or None
        Families with block settings objects only (`SpecHooks.row_objects`): the block holds a family-specific row kind with these settings (an object or its dict).
    hm : object or dict or None
        Likewise (`SpecHooks.row_objects`): a second settings-object key of a family: a block whose rows run another family's scan, with the decays
        inside the object: `SpecHooks.block_decay`).
    """
    rows: int | None = None
    frac: float | None = None
    decay: str | None = None
    complex: bool = False
    gate: bool | None = None
    chart: bool | None = None
    tri: Any = None
    hm: Any = None

    def to_dict(self) -> dict:
        """The strict dict form of this block (only the fields that are set)."""
        d = {"decay": self.decay, "complex": self.complex}
        for key in ("gate", "rows", "frac", "chart"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        for key in ("tri", "hm"):
            v = getattr(self, key)
            if v is not None:
                d[key] = v.to_dict() if hasattr(v, "to_dict") else dict(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RowBlock":
        """The strict dict form: only the field names, each of its type (``TypeError`` otherwise)."""
        from .validation import strict_from_dict
        return strict_from_dict(cls, d)


# ── helpers ──────────────────────────────────────────────────────────────
def _bool(v, name):
    if not isinstance(v, bool):
        raise ConfigSpecError(f"{name} must be a bool, got {v!r}")
    return v


def _rot_value(key, v, name):
    """Validate one rotation field (the bools, `omega_layout`, `theta_init`)."""
    if key in _ROTATION_CHOICES:
        if v not in _ROTATION_CHOICES[key]:
            raise ConfigSpecError(f"{name} must be one of "
                                  f"{_ROTATION_CHOICES[key]}, got {v!r}")
        return v
    return _bool(v, name)


def spec_kwargs(values: dict) -> dict:
    """The flat spec keywords a caller actually set (None = unset)."""
    return {k: values[k] for k in SPEC_FIELDS if k in values and values[k] is not None}


def flat_spec(spec: dict) -> dict:
    """Canonical spec in the flat form recipes store (rotation fields are None without a complex block)."""
    rot = spec["rotation"] or {}
    return {"row_blocks": copy.deepcopy(spec["row_blocks"]),
            **{k: rot.get(k) for k in ROTATION_FIELDS}, **spec["maps"]}


# ── rows ─────────────────────────────────────────────────────────────────
def _resolve_rows(family, rows, d_state):
    hooks = _HOOKS.get(family)
    if rows is None:
        if family == "hybrid_mamba" or (hooks is not None and hooks.rows_required):
            raise ConfigSpecError(f"{family}: `row_blocks` is required")
        return [{"rows": d_state, "decay": HEAD_DYNAMIC, "complex": False,
                 "gate": None}]                                  # None: resolved by `_resolve_block_gates`
    if isinstance(rows, (RowBlock, dict)):
        raise ConfigSpecError("row_blocks must be a list of blocks, not one block")
    blocks = []
    for i, b in enumerate(rows):
        if isinstance(b, RowBlock):
            b = b.to_dict()
        elif isinstance(b, dict):
            b = RowBlock.from_dict(b).to_dict()
        else:
            raise TypeError(f"row_blocks[{i}] must be a RowBlock or its dict, got {type(b).__name__}")
        objs = (hooks.row_objects or {}) if hooks is not None else {}
        extra = set(b) - set(BLOCK_KEYS) - set(ROW_EXT.get(family, ())) - set(objs)
        if extra:
            raise ConfigSpecError(f"row_blocks[{i}]: unknown keys {sorted(extra)}")
        if b.get("rows") is not None and b.get("frac") is not None:
            raise ConfigSpecError(f"row_blocks[{i}]: give 'rows' or 'frac', not both")
        if b.get("rows") is not None:
            n = b["rows"]
            if isinstance(n, bool) or not isinstance(n, Integral):
                raise ConfigSpecError(f"row_blocks[{i}]['rows'] must be an int, got {n!r}")
            n = int(n)
        elif b.get("frac") is not None:
            f = b["frac"]
            if isinstance(f, bool) or not isinstance(f, Real) or not 0 < f <= 1:
                raise ConfigSpecError(f"row_blocks[{i}]['frac'] must be in (0, 1], got {f!r}")
            x = float(f) * d_state
            n = int(round(x))
            if abs(x - n) > 1e-9:                   # no implicit tie/rounding rule
                raise ConfigSpecError(
                    f"row_blocks[{i}]['frac']={f!r} gives {x:g} rows of d_state={d_state}; "
                    "frac*d_state must be an integer (or give 'rows' explicitly)")
        else:
            raise ConfigSpecError(f"row_blocks[{i}]: needs 'rows' or 'frac'")
        if n <= 0:
            raise ConfigSpecError(f"row_blocks[{i}]: resolved to {n} rows")
        holder = next((k for k in ((hooks.block_decay or {}) if hooks is not None else {}) if b.get(k) is not None), None)
        if holder is not None:
            # a block whose object holds its own decays: the outer block names none (recorded as the object's marker)
            if b.get("decay") not in (None, hooks.block_decay[holder]):          # the marker itself is the resolved (recorded) form
                raise ConfigSpecError(f"row_blocks[{i}]: a block with a {holder!r} key has no 'decay' of its own (got {b['decay']!r}); its decays are inside {holder!r}")
            b = {**b, "decay": hooks.block_decay[holder]}
        elif b.get("decay") is None:
            default = DEFAULT_DECAY.get(family) or (hooks.default_decay if hooks is not None else None)       # else the family hook
            if default is None:
                raise ConfigSpecError(f"row_blocks[{i}]: 'decay' is required")
            b = {**b, "decay": default}                   # resolved, recorded on every block
        cx = _bool(b.get("complex", False), f"row_blocks[{i}]['complex']")
        if cx and n % 2:
            raise ConfigSpecError(
                f"row_blocks[{i}]: a complex block holds (re, im) pairs; {n} rows is odd")
        gate = b.get("gate")
        if hooks is not None and hooks.block_gate is not None:
            gate = hooks.block_gate(i, gate, cx)
        else:
            gate = None if gate is None else _bool(gate, f"row_blocks[{i}]['gate']")   # None: `_resolve_block_gates`
        block = {"rows": n, "decay": b["decay"], "complex": cx, "gate": gate}
        for key in ROW_EXT.get(family, ()):
            block[key] = _bool(b.get(key, False), f"row_blocks[{i}][{key!r}]")
        for key, check in objs.items():
            if b.get(key) is not None:
                block[key] = check(b[key], i)
        blocks.append(block)
    if not blocks:
        raise ConfigSpecError("row_blocks must be a non-empty list of blocks")
    total = sum(b["rows"] for b in blocks)
    if total != d_state:
        raise ConfigSpecError(f"row_blocks sum to {total}, need exactly d_state={d_state}")

    if hooks is not None:
        if hooks.check_rows is not None:
            hooks.check_rows(blocks)
    elif family == "ssd_mimo":
        if len(blocks) > SSD_MAX_BLOCKS:
            raise ConfigSpecError(
                "ssd_mimo: at most two row blocks (one complex part, one real part)")
        for i, b in enumerate(blocks):
            if b["decay"] == DECAY_NONE:
                raise ConfigSpecError(
                    f"row_blocks[{i}]: decay {DECAY_NONE!r} (a_t = 1) is not served by ssd_mimo (its scan always applies the "
                    f"decay law); ssd_mimo decay must be one of {SSD_DECAYS}")
        laws = {b["decay"] for b in blocks}
        if len(laws) > 1:
            named = ", ".join(f"row_blocks[{i}]={b['decay']!r}" for i, b in enumerate(blocks))
            raise ConfigSpecError(
                "ssd_mimo: one decay law for the whole state: the row blocks are rows of ONE scan with ONE decay a_t per "
                f"(head, sibling), so their decays must resolve to the same law; got {named}")
        for i, b in enumerate(blocks):
            if b["decay"] not in SSD_DECAYS:
                raise ConfigSpecError(
                    f"row_blocks[{i}]: ssd_mimo decay must be one of {SSD_DECAYS}, "
                    f"got {b['decay']!r}")
        if len(blocks) == 2:
            if blocks[0]["complex"] == blocks[1]["complex"]:
                raise ConfigSpecError(
                    "ssd_mimo: two blocks must be one complex and one real part")
            if not blocks[0]["complex"]:
                raise ConfigSpecError(
                    "ssd_mimo: the complex block comes first (the kernel "
                    "rotates the leading pairs)")
    else:
        if len(blocks) > HM_MAX_BLOCKS:
            raise ConfigSpecError(f"hybrid_mamba: at most {HM_MAX_BLOCKS} row blocks")
        for i, b in enumerate(blocks):
            if b["decay"] == DECAY_NONE:
                raise ConfigSpecError(
                    f"row_blocks[{i}]: decay {DECAY_NONE!r} (a_t = 1) is not served by hybrid_mamba (every block runs a decay "
                    f"law); hybrid_mamba decay must be one of {HM_DECAYS}")
            if b["decay"] not in HM_DECAYS:
                raise ConfigSpecError(
                    f"row_blocks[{i}]: hybrid_mamba decay must be one of {HM_DECAYS}, "
                    f"got {b['decay']!r}")
            if b["complex"] and b["decay"] not in HM_COMPLEX_OK:
                raise ConfigSpecError(
                    f"row_blocks[{i}]: complex=True is served on {HM_COMPLEX_OK} "
                    f"only, got {b['decay']!r}")
        flat = [(i, b) for i, b in enumerate(blocks) if b["decay"] == HEAD_DYNAMIC]
        for (_, a), (j, b) in zip(flat, flat[1:]):
            if b["complex"] and not a["complex"]:
                raise ConfigSpecError(
                    f"row_blocks[{j}]: a complex head_dynamic block follows a real one; the head_dynamic rows are one scan whose "
                    "rotated rows lead (as in ssd_mimo), so list the complex block first")
        # packing / power-of-two / pair-alignment rules stay in compile_rows
    return blocks


def _resolve_block_gates(family, blocks, mp, d_ssm_in, d_ssm_out):
    """The ``gate`` of every block of a built-in family, resolved to what actually runs: an omitted ``gate`` is
    `BLOCK_GATE_DEFAULT` for its ``complex`` flag when B/C gates exist and ``False`` when none does; an explicit ``gate=True``
    with no B/C gate raises (it would be ignored).  Whether gates exist: an explicit ``b_gate`` / ``c_gate``, else SSD's rank
    default (``d_ssm_in > 1`` / ``d_ssm_out > 1`` and some block gated by default), HM's ``False``."""
    default = [BLOCK_GATE_DEFAULT[b["complex"]] if b["gate"] is None else b["gate"] for b in blocks]
    if family == "ssd_mimo":
        any_def = any(default)
        exist = mp.get("b_gate", d_ssm_in > 1 and any_def) or mp.get("c_gate", d_ssm_out > 1 and any_def)
    else:
        exist = mp.get("b_gate", False) or mp.get("c_gate", False)
    out = []
    for i, (b, d) in enumerate(zip(blocks, default)):
        if b["gate"] and not exist:
            raise ConfigSpecError(
                f"row_blocks[{i}]: gate=True, but no B/C gate exists (b_gate and c_gate resolve False), so it would be ignored; "
                "drop the key (it resolves to False), or enable b_gate / c_gate")
        out.append({**b, "gate": d if exist else False})
    return out


def check_law_levers(who: str, blocks, table, levers: dict) -> None:
    """Refuse a layer-level lever of a decay law that no row block uses (it would be silently ignored).

    ``table``: ``[(laws, {lever: default}), ...]``: the levers apply only to rows with one of ``laws``.  ``levers``: the
    values the caller set (``None`` = unset).  A lever at its default is never refused (recipes record defaults).  The
    levers stay at layer level (they are shared by every block of the law); this only checks that they have a block to act on."""
    present = {b["decay"] for b in blocks}
    for laws, defaults in table:
        if present & set(laws):
            continue
        for key, default in defaults.items():
            v = levers.get(key)
            if v is not None and v != default:
                raise ConfigSpecError(
                    f"{who}: {key}={v!r} is set, but no row block has a decay in {tuple(laws)}, so it would be ignored; "
                    f"row_blocks decays are {sorted(present)} (drop {key}, or add a block with one of those laws)")


# ── the resolver ─────────────────────────────────────────────────────────
def resolve(family: str, *, d_state: int, d_ssm_in: int = 1, d_ssm_out: int | None = None,
            row_blocks=None, **flat) -> dict:
    """Resolve ``row_blocks`` and the flat spec keywords (`SPEC_FIELDS`) to the canonical dict
    ``{"row_blocks", "rotation", "maps"}``; ``d_ssm_in`` / ``d_ssm_out`` drive SSD's gate defaults."""
    if family not in FAMILIES:
        raise ConfigSpecError(f"family must be one of {FAMILIES}, got {family!r}")
    if isinstance(d_state, bool) or not isinstance(d_state, Integral) or d_state <= 0:
        raise ConfigSpecError(f"d_state must be a positive int, got {d_state!r}")
    d_ssm_out = d_ssm_in if d_ssm_out is None else d_ssm_out
    allowed_maps = set(MAPS_CORE) | set(MAPS_EXT[family])
    unknown = set(flat) - set(ROTATION_FIELDS) - set(MAPS_CORE) - {k for e in MAPS_EXT.values() for k in e}
    if unknown:
        raise ConfigSpecError(f"{family}: unknown config keys {sorted(unknown)}")
    flat = {k: v for k, v in flat.items() if v is not None}

    blocks = _resolve_rows(family, row_blocks, d_state)
    has_complex = any(b["complex"] for b in blocks)

    rot = {k: _rot_value(k, flat[k], k) for k in ROTATION_FIELDS if k in flat}
    mp = {}
    for k in flat:
        if k in ROTATION_FIELDS:
            continue
        if k not in allowed_maps:
            other = [f for f, ext in MAPS_EXT.items() if k in ext]
            raise ConfigSpecError(f"{k} is a {other[0]}-only field")
        mp[k] = flat[k]

    if has_complex:
        hooks = _HOOKS.get(family)
        defaults = (hooks.rotation_defaults if hooks is not None and hooks.rotation_defaults is not None
                    else ROTATION_DEFAULTS)
        rotation_out = {k: rot.get(k, d) for k, d in defaults.items()}
        if hooks is not None and hooks.check_rotation is not None:
            hooks.check_rotation(rotation_out)
        layout = rotation_out["omega_layout"]
        if layout == "head":
            if rot.get("theta_per_head") is False:
                raise ConfigSpecError(
                    "theta_per_head=False conflicts with omega_layout='head': one eigenvalue per "
                    "head has one angle feature per head (theta_per_head=True)")
            rotation_out["theta_per_head"] = True
        if layout == "head_pair" and not rotation_out["learn_theta_freq"]:
            raise ConfigSpecError(
                "omega_layout='head_pair' needs "
                "learn_theta_freq=True (the fixed pi has no layout)")
    else:
        on = sorted(k for k, v in rot.items()
                    if (v if k in _ROTATION_BOOLS else v != ROTATION_DEFAULTS[k]))
        if on:
            raise ConfigSpecError(
                f"rotation fields {on} requested, but `row_blocks` has no complex "
                "block (add one, or drop them)")
        rotation_out = None

    for k in ("b_norm", "c_norm", "b_gate", "c_gate", "bc_gate_bias"):
        if k in mp:
            _bool(mp[k], k)
    hooks = _HOOKS.get(family)
    grans = hooks.bias_grans if hooks is not None and hooks.bias_grans is not None else BIAS_GRANS
    if "bc_bias" in mp and mp["bc_bias"] not in grans:
        raise ConfigSpecError(f"bc_bias must be one of {grans}, got {mp['bc_bias']!r}")
    out = {"b_norm": mp.get("b_norm", True),
           "c_norm": mp.get("c_norm", True),
           "bc_bias": mp.get("bc_bias", hooks.default_bias if hooks is not None else "head")}
    if hooks is None:
        blocks = _resolve_block_gates(family, blocks, mp, d_ssm_in, d_ssm_out)
    any_gated = any(b["gate"] for b in blocks)
    if hooks is not None and hooks.resolve_maps is not None:
        out.update(hooks.resolve_maps(mp, blocks))
    elif family == "ssd_mimo":
        # default: gate when d_ssm_in/out > 1, off when no block is gated
        out["b_gate"] = mp.get("b_gate", d_ssm_in > 1 and any_gated)
        out["c_gate"] = mp.get("c_gate", d_ssm_out > 1 and any_gated)
        init = mp.get("bc_bias_init", "auto")
        if out["bc_bias"] == "none":
            if init not in ("auto", "none"):
                raise ConfigSpecError(
                    f"bc_bias_init={init!r} given, but bc_bias='none'")
            init = "none"
        elif init not in SSD_BIAS_INITS:
            raise ConfigSpecError(f"bc_bias_init must be one of "
                                  f"{SSD_BIAS_INITS}, got {init!r}")
        elif init == "auto":         # resolved, never saved as "auto"
            init = "ones" if (d_ssm_in == 1 or out["bc_bias"] == "shared") else "spread"
        out["bc_bias_init"] = init
        if has_complex and out["bc_bias"] != "head":
            raise ConfigSpecError(
                "ssd_mimo: a complex block needs bc_bias='head' (the "
                f"per-kernel-head biases act before the rotation), got {out['bc_bias']!r}")
    else:
        out["b_gate"] = mp.get("b_gate", False)
        out["c_gate"] = mp.get("c_gate", False)
        # default form is SSD's (1 + g)
        act = mp.get("bc_gate_act", "one_plus")
        if act not in HM_GATE_ACTS:
            raise ConfigSpecError(f"bc_gate_act must be one of {HM_GATE_ACTS}, got {act!r}")
        out["bc_gate_act"] = act
        out["bc_gate_bias"] = mp.get("bc_gate_bias", False)
        if out["bc_gate_bias"] and not (out["b_gate"] or out["c_gate"]):
            raise ConfigSpecError("bc_gate_bias=True needs b_gate or c_gate")
        pos = mp.get("bc_gate_pos", "pre")
        if pos not in HM_GATE_POS:
            raise ConfigSpecError(f"bc_gate_pos must be one of {HM_GATE_POS}, got {pos!r}")
        out["bc_gate_pos"] = pos
        if not (out["b_gate"] or out["c_gate"]) and (act != "one_plus" or pos != "pre"):
            raise ConfigSpecError(
                f"bc_gate_act={act!r} / bc_gate_pos={pos!r} are set, but no B/C gate exists (b_gate and c_gate are False), "
                "so they would be ignored; drop them or enable b_gate / c_gate")
    # a gate that exists must act on some rows
    for k in ("b_gate", "c_gate"):
        if out[k] and not any_gated:
            raise ConfigSpecError(
                f"{k}=True but no row block has gate=True (the "
                f"per-block default is {BLOCK_GATE_DEFAULT[False]} for real and "
                f"{BLOCK_GATE_DEFAULT[True]} for complex blocks): set "
                "gate=True on the block(s) to gate, or drop the gate")
    return {"row_blocks": blocks, "rotation": rotation_out, "maps": out}


def load_saved(family: str, spec: dict, *, d_state: int, d_ssm_in: int = 1,
               d_ssm_out: int | None = None) -> dict:
    """Validate a saved spec (flat form of `flat_spec`): it must already be canonical; returns the canonical dict."""
    again = resolve(family, d_state=d_state, d_ssm_in=d_ssm_in, d_ssm_out=d_ssm_out, **spec)
    if flat_spec(again) != spec:
        raise ConfigSpecError("saved spec is not canonical (resolve() changes it)")
    return again


def refuse_per_coordinate_gates(module, state_dict, prefix, per_coordinate_rows) -> None:
    """Load pre-hook of SSD / HybridMamba layers: gates on complex row pairs are pair-tied, so a
    state dict with per-COORDINATE gate rows on a complex block is refused, never migrated.
    ``per_coordinate_rows(name)``: the row count that refused layout has for gate parameter ``name``."""
    if not module.gate_pair_tied:
        return
    for name in ("gB_proj.weight", "gC_proj.weight", "beta_gB", "beta_gC"):
        mod = getattr(module, name.split(".")[0], None)
        t = state_dict.get(prefix + name)
        if t is None or mod is None:
            continue
        own = mod.weight if name.endswith(".weight") else mod
        axis = 0 if name.endswith(".weight") else -1
        if t.shape[axis] != own.shape[axis] and t.shape[axis] == per_coordinate_rows(name):
            raise RuntimeError(
                f"{prefix}{name}: this checkpoint gates the complex rows PER "
                f"COORDINATE ({t.shape[axis]} rows); gates on complex row pairs "
                "are pair-tied (one factor per pair, "
                f"{own.shape[axis]} rows).  There is no migration -- retrain.")
