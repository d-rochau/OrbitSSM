"""`kernel_options`: the one place a config may steer a kernel.
Set through ``Execution(kernel_options={...})``; `describe` lists the keys of a family.

The surface is a plain ``dict[str, Any]`` so levers can come and go without config
migrations.  Three rules, enforced in code:

1. Unknown keys and unsupported values raise, by name (never silently ignored).
2. An option never changes a parameter name or shape (checkpoint invariant); no key
   may collide with a config or `Execution` field (`_check_schema`).
3. Every key declares ``open_until``, the measurement whose verdict deletes it.

Precedence, highest first: explicit ``kernel_options`` > env lever > family default.
(``SSD_MIMO_CHUNK`` is the one exception: it overrides the ``chunk_size`` field.)
A key marked ``env_read=False`` (the numerics- and semantics-changing switches) has NO
environment spelling: the config is its only channel, so the recipe records every such choice.

Imports nothing from `orbit_ssm` and no torch, so configs can be built and validated
without a GPU.
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading
from typing import Any, Callable

from . import families as _families
from .validation import valid_chunk_size

__all__ = [
    "KernelOptionError",
    "FAMILIES",
    "SUPPORTED_KERNEL_OPTIONS",
    "LAYER_KEYS",
    "split",
    "schema",
    "defaults",
    "describe",
    "validate",
    "resolve",
    "non_default",
    "record",
    "push",
    "pop",
    "scope",
    "current",
    "env_override",
    "scoped_kwarg",
    "scoped_ctx",
]


class KernelOptionError(ValueError):
    """An unknown `kernel_options` key, or an unsupported value.

    A `ValueError` subclass so that callers already catching the config
    surface's refusals keep catching it.
    """


# ── the option spec ──────────────────────────────────────────────────

_JSON_SAFE = (str, int, float, bool, type(None))


def _json_safe(v) -> bool:
    """A scalar, or a flat ``{str: scalar}`` dict (the MH `mem` dial)."""
    if isinstance(v, dict):
        return all(isinstance(k, str) and isinstance(x, _JSON_SAFE) for k, x in v.items())
    return isinstance(v, _JSON_SAFE)


class _Opt:
    """One supported key: its values, its default, and what retires it."""

    __slots__ = ("key", "default", "values", "check", "env", "doc",
                 "open_until", "env_read")

    def __init__(self, key: str, *, default: Any, env: str | None,
                 doc: str, open_until: str,
                 values: tuple | None = None,
                 check: Callable[[Any], bool] | None = None,
                 env_read: bool = True):
        self.key = key
        self.default = default
        self.values = values
        self.check = check
        self.env = env
        self.doc = doc
        self.open_until = open_until
        #: False: a numerics- or semantics-changing option.  ``env`` then only names the
        #: kernels' ``_env(<env>, ...)`` reader that the option feeds; the process
        #: environment is NOT read (a recorded option has no hidden second channel).
        self.env_read = env_read

    # -- the admissible surface, spelled for an error message ---------
    def allowed(self) -> str:
        if self.values is not None:
            return " | ".join(repr(v) for v in self.values)
        return self.doc

    def accepts(self, value: Any) -> bool:
        if self.values is not None and value in self.values:
            # tuple `in` uses ==, so True == 1: re-check the type (no bool for an int option).
            return any(value == v and type(value) is type(v)
                       for v in self.values)
        if self.check is not None:
            return bool(self.check(value))
        return False

    def to_env(self, value: Any) -> str | None:
        """The string an `_env()` reader should see (None -> "" = unset)."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)


def _pow2_chunk_or_auto(v) -> bool:
    return v != "measure" and valid_chunk_size(v)


def _int_in(lo: int, hi: int) -> Callable[[Any], bool]:
    def _f(v) -> bool:
        if v is None:
            return True
        if isinstance(v, bool) or not isinstance(v, int):
            return False
        return lo <= v <= hi
    return _f


def _kernel_target_opt():
    return _Opt("kernel_target", default="auto", values=("auto", "portable"), env=None,
                doc="the kernel target: 'auto' (default; the shipped NVIDIA kernels, Gluon included; resolves to 'portable' by itself "
                    "when Triton's active backend is not CUDA) or 'portable' (plain Triton only: no Gluon, no inline asm, no tf32 "
                    "literal -- fp32 operands run at 'ieee'; a route with no plain-Triton kernel is refused by name).  "
                    "Selecting 'portable' latches libdevice trig for the process (before the first rotation kernel compiles)",
                open_until="a non-NVIDIA backend runs the portable kernels in CI (no such hardware in the budget: correctness is "
                           "checked by forcing the path on NVIDIA and by the Triton interpreter), or the Gluon kernels are deleted")


# ── the schemas ──────────────────────────────────────────────────────
#
# Per-family option sets.  An unlisted key raises (e.g. `fwd_warps` on HybridMamba).
# NOTE: `hybrid_mamba` also launches the SSD kernel for
# its flat rows, but has no key naming that kernel family, so SSD levers do not
# reach it.  A registered family (`orbit_ssm.families`) brings its own layer switches.

_SSD_MIMO_OPTS: tuple[_Opt, ...] = (
    _kernel_target_opt(),
    _Opt("autotune",
         default="auto", values=("pinned", "light", "full", "auto", "off"),
         env="AUTOTUNE",
         doc="THE ONE SWITCH, as a tier.  The env lever "
             "SSD_MIMO_AUTOTUNE takes 'pinned' | 'light' (default) | "
             "'full'; the aliases 'off' and 'auto' mean "
             "'pinned' and 'light'.  'pinned' runs no timed search in "
             "either direction -- the forward takes its device "
             "heuristic (cache neither read nor written), the "
             "chunk-size and shared-memory trials are skipped, the "
             "fused backward's autotune grids collapse to one "
             "candidate each and the SISO kernels take their measured "
             "pins.  'light' tunes "
             "the fused backward over num_warps only (4 configs); "
             "'full' also tunes stages and maxnreg (16).  As a CONFIG "
             "value this key accepts the same tiers and aliases. "
             "CAVEAT, stated "
             "rather than hidden: fused backward and SISO forward, "
             "backward and rotation-adjoint grids are built AT "
             "IMPORT, so only the ENV spelling (e.g. "
             "SSD_MIMO_AUTOTUNE=pinned) reaches them -- set it before "
             "importing orbit_ssm.  As a config value this key covers "
             "the measure-based halves.",
         open_until="the backward's grids stop being decorator "
                    "arguments, at which point the env caveat goes and "
                    "the key covers both halves on its own"),
    _Opt("fwd_cfg",
         default="auto", values=("auto", "measure", "off"),
         env="FWD_CFG",
         doc="which rung picks the fused forward's launch config",
         open_until="the REALMODE4 ladder rules whether 'measure' "
                    "becomes the default; the loser's name is deleted"),
    _Opt("fwd_cfg_iters",
         default=3, check=_int_in(1, 64),
         env="FWD_CFG_ITERS",
         doc="timed reps per candidate under fwd_cfg='measure' "
             "(int 1..64)",
         open_until="with `fwd_cfg` -- it has no meaning without it"),
    _Opt("fwd_warps",
         default=None, values=(None, 1, 2, 4, 8),
         env="FWD_WARPS",
         doc="explicit num_warps pin for the fused forward "
             "(None = let the dispatcher choose)",
         open_until="the measured dispatcher is trusted on "
                    "every device we ship on; the three pins go together"),
    _Opt("fwd_stages",
         default=None, check=_int_in(1, 8),
         env="FWD_STAGES",
         doc="explicit num_stages pin for the fused forward "
             "(None, or int 1..8)",
         open_until="with `fwd_warps`"),
    _Opt("fwd_maxnreg",
         default=None, check=_int_in(16, 255),
         env="FWD_MAXNREG",
         doc="explicit maxnreg pin for the fused forward "
             "(None, or int 16..255; dropped on a HIP build)",
         open_until="with `fwd_warps`"),
    _Opt("backward_states",
         default="recompute", values=("recompute", "io"),
         env="SALL",
         doc="the backward's per-chunk state policy: 'recompute' replays "
             "the forward in STATE_ONLY mode, 'io' stores them",
         open_until="a measured execution policy picks "
                    "one; the losing policy is deleted, not kept"),
    _Opt("chunk_size_bwd",
         default="auto", check=_pow2_chunk_or_auto,
         env="CHUNK_BWD",
         doc="the chunk T the BACKWARD runs at: 'auto' (the forward's, "
             "the shipped schedule bit for bit) or a power of two >= 16",
         open_until="`_CHUNK_BWD_TABLE` is filled by a measured (T_fwd, "
                    "T_bwd) pair, which makes 'auto' the only answer"),
    _Opt("smem_strategy",
         default="measure", values=("measure", "t_halve", "n_split"),
         env="SMEM_STRATEGY",
         doc="the shared-memory OVERFLOW repair to use when a cell does "
             "not fit: measure both, or pin one",
         open_until="a measurement rules one repair dominant at every "
                    "cell that overflows"),
)

#: MH (hybrid_mamba) launch-config keys a `fwd_config` / `bwd_config` / carry pin may name.
#: Every searchable axis is pinnable; TS, HP and BLOCK_L stay clamped to what the launch
#: allows, TPW_L is derived from BLOCK_P.
MH_FWD_CONFIG_KEYS = ("BLOCK_P", "BLOCK_L", "NUM_WARPS", "PF", "TS", "CKV", "FPACK")
MH_BWD_CONFIG_KEYS = ("TP_P", "NS", "NP", "NUM_WARPS", "PU", "BCS", "THS",
                      "AGX", "MAXNREG", "HP", "BLOCK_L", "TS", "LWS")
MH_CARRY_CONFIG_KEYS = ("BLOCK_C", "BLOCK_T", "NUM_WARPS")


def parse_mh_config(v, keys) -> dict:
    """'KEY=int,KEY=int' (or a JSON object) -> {KEY: int}; ValueError otherwise.
    A string, so the pin is a JSON scalar with one spelling in recipes and env."""
    txt = str(v).strip()
    if txt.startswith("{"):
        import json
        try:
            obj = json.loads(txt)
        except ValueError as exc:
            raise ValueError(f"bad MH config {txt!r}: {exc}") from None
        if not isinstance(obj, dict) or not all(
                isinstance(val, int) and not isinstance(val, bool)
                for val in obj.values()):
            raise ValueError(f"bad MH config {txt!r}; keys: {keys}")
        txt = ",".join(f"{k}={val}" for k, val in obj.items())
    out = {}
    for item in filter(None, (t.strip() for t in txt.split(","))):
        k, _, val = item.partition("=")
        k = k.strip().upper()
        if k not in keys or not val.strip().lstrip("-").isdigit():
            raise ValueError(f"bad MH config item {item!r}; keys: {keys}")
        out[k] = int(val)
    return out


def _mh_cfg_check(keys):
    def _f(v) -> bool:
        if v is None:
            return True
        if not isinstance(v, str):
            return False
        try:
            return bool(parse_mh_config(v, keys))
        except ValueError:
            return False
    return _f


# `hybrid_mamba` (MH per-channel rows); tier names follow `ssd_mimo`'s autotune.
_HYBRID_MAMBA_OPTS: tuple[_Opt, ...] = (
    _kernel_target_opt(),
    _Opt("autotune",
         default="light", values=("pinned", "light", "full", "auto", "off"),
         env="AUTOTUNE",
         doc="the MH launch-config search, as a tier (env HYBRID_MH_AUTOTUNE, "
             "'0' = pinned, '1' = light; config aliases 'off' = "
             "pinned, 'auto' = light).  'pinned' times nothing: the "
             "per-device table / heuristic config (or `fwd_config` / "
             "`bwd_config`) is launched as is -- deterministic across "
             "processes and ranks, two compiles per shape.  'light' "
             "searches the forward's (BLOCK_P, BLOCK_L, PF) corners and "
             "the backward's NS/TP_P axes; 'full' also warps, slab counts, "
             "p-sharding and register caps",
         open_until="a per-device pinned table covers every shipped cell "
                    "and light never beats it"),
    _Opt("fwd_config",
         default=None, check=_mh_cfg_check(MH_FWD_CONFIG_KEYS),
         env="FWD_CONFIG",
         doc="explicit forward launch config, e.g. 'BLOCK_P=4,BLOCK_L=256,"
             "PF=1' or the same as a JSON object (keys: "
             + ", ".join(MH_FWD_CONFIG_KEYS) + "); never searched, at every "
             "tier (BLOCK_P still clamped to a divisor of P)",
         open_until="the pinned table is trusted on every device"),
    _Opt("bwd_config",
         default=None, check=_mh_cfg_check(MH_BWD_CONFIG_KEYS),
         env="BWD_CONFIG",
         doc="explicit backward launch config, e.g. 'TP_P=2,NS=4' or the "
             "same as a JSON object (keys: " + ", ".join(MH_BWD_CONFIG_KEYS)
             + "); never searched, at every tier; a mem= dial wins over it "
             "(precedence config > mem > pin > env)",
         open_until="with `fwd_config`"),
    _Opt("rotation",
         default="fold", values=("fold", "in_scan"),
         env="ROTATION",
         doc="the rotation route (env HYBRID_MH_ROTATION): "
             "'fold' folds the head-dependent phase into per-head B/C maps "
             "ahead of ONE real scan and splits a mixed launch into its "
             "rotating window (folded) and its real rows (plain real scan); "
             "training within 7 % of all-real at ~1.18x its peak memory.  'in_scan' rotates inside "
             "the scan (A/B only; also the counted fallback for per-head "
             "theta and row-window launches)",
         open_until="the fold serves per-head theta, row windows and "
                    "state continuation; then in-scan rotation retires"),
    _Opt("carry_fwd_config",
         default=None, check=_mh_cfg_check(MH_CARRY_CONFIG_KEYS),
         env="CARRY_FWD_CONFIG",
         doc="explicit config of the time-split forward carry kernel, e.g. "
             "'BLOCK_C=16,BLOCK_T=128,NUM_WARPS=4'; launched alone at every "
             "tier (keys: " + ", ".join(MH_CARRY_CONFIG_KEYS) + ")",
         open_until="with `fwd_config`"),
    _Opt("carry_bwd_config",
         default=None, check=_mh_cfg_check(MH_CARRY_CONFIG_KEYS),
         env="CARRY_BWD_CONFIG",
         doc="explicit config of the time-split adjoint carry kernel; as "
             "`carry_fwd_config`",
         open_until="with `fwd_config`"),
    _Opt("gate_bias_partial",
         default="auto", values=("auto", False, True),
         env="BWD_GATE_BIAS_PARTIAL",
         doc="per-group gate-bias gradients as per-block partials summed "
             "on the host (True) or relaxed fp32 atomics into the (G, N) "
             "buffer (False); 'auto' (default) = atomics: with relaxed "
             "atomics the two tie on H100 and atomics need "
             "32 MiB less at the LLM cell and keep the P == 1 head packing",
         open_until="Blackwell/A100 confirm the tie with relaxed atomics; "
                    "then the partial path retires or stays an A/B lever"),
)


# Numerics- and semantics-changing switches.  Each was an environment lever (`SSD_MIMO_*`,
# `MH_*`); as a recorded option it travels with the recipe and the process environment is never
# consulted (`env_read=False`: `env` only names the kernels' `_env(<env>)` reader the option feeds,
# see `option_only`).  The remaining environment names are read by the kernels.
def _num_opt(key, default, env, doc, open_until, values=(False, True)):
    return _Opt(key, default=default, values=values, env=env, doc=doc,
                open_until=open_until, env_read=False)


_SSD_MIMO_NUMERIC_OPTS: tuple[_Opt, ...] = (
    _num_opt("backward_states_fp32", False, "SALL_FP32",
             "store the per-chunk states S_all the backward consumes in fp32 instead of the IO "
             "dtype (every fp32-operand mode already does; this adds it to the 16-bit-operand "
             "mode)",
             "a measured execution policy (contract section 6) settles the state dtype"),
    _num_opt("theta_bwd_fp32", True, "THETA_BWD_FP32",
             "keep the running phase the backward reads in fp32 (True, default); False saves its "
             "bf16 twin on 16-bit IO routes (a bf16 phase carries 6e-3 rad at |Theta| ~ 4e5, so "
             "this is an A/B arm, not a production setting)",
             "the bf16 twin is measured to lose everywhere and its plumbing is deleted"),
    _num_opt("rot_split_dtype", "io", "ROT_SPLIT_DTYPE",
             "dtype the rotation prologue writes B~ / C~ in: 'io' (one extra rounding of an "
             "operand the kernels round at the tl.dot anyway) or 'fp32'",
             "the fp64 oracle ranks one dtype dominant at every cell", values=("io", "fp32")),
    _num_opt("rot_operand_dtype", "fp32", "ROT_OPERAND_DTYPE",
             "whether the rotated B~ / C~ operands are cast back to the IO dtype ('io') or stay "
             "fp32 (default)",
             "the fp64 oracle ranks one dtype dominant at every cell", values=("fp32", "io")),
    _num_opt("gluon_kbc_dalpha", False, "GLUON_KBC_DALPHA",
             "Gluon Kbc takes the dalpha hand-off from Ka instead of forming it itself.  Failed "
             "the mixer-level fp64 oracle on A100 and Blackwell until the dG-slot aliasing fix "
             "(`sDG`); passes it on sm_120 now.  Stays off by default",
             "the hand-off is measured at every cell or is deleted"),
    _num_opt("siso_adj_raw_common", False, "SISO_ADJ_RAW_COMMON",
             "SISO rotation adjoint contracts in the raw frame (pre-rotation biases, common "
             "angle); the other cases keep their stage-by-stage adjoints",
             "the raw-frame contraction is measured against the stage chain at every cell"),
    _num_opt("dt_fp32", True, "DT_FP32",
             "add the dt bias in fp32 inside the tail's softplus (True, default); False folds a bf16 bias into the token "
             "GEMM (a 2-3 % signed error in Delta near a pre-activation of -9)",
             "the bf16 fold is measured to lose everywhere"),
)

_HYBRID_MAMBA_NUMERIC_OPTS: tuple[_Opt, ...] = (
    _num_opt("ckpt_bf16", True, "CKPT_BF16",
             "store the forward's state checkpoints in the 16-bit input dtype (True, default; "
             "the checkpoint is a rounding of a state the 16-bit inputs already carry) or "
             "always in fp32",
             "the fp32 checkpoint is measured to buy no accuracy at 16-bit IO"),
)


# ── layer-level switches ─────────────────────────────────────────────
# Execution-only switches read by the layer itself (`split` separates them from the
# kernel keys); same function and parameters.  `env=None`: no environment spelling.

def _bool_opt(key, default, doc, open_until):
    return _Opt(key, default=default, values=(False, True), env=None,
                doc=doc, open_until=open_until)


def _m3_kernel_opt():
    return _Opt("m3_kernel", default="native", values=("native", "upstream"), env=None,
                doc="which M3 scan serves the flat Mamba-3 rows: 'native' (ours) or "
                    "'upstream' (the mamba_ssm Mamba-3 kernels; refused at apply time "
                    "when the layer uses a feature upstream lacks, or on a backend "
                    "without the provider)",
                open_until="the upstream adapter is only a parity/timing baseline; it "
                           "retires with the Mamba-3 comparison")


def _conv_kernel_opt():
    return _Opt("conv_kernel", default="auto", values=("auto", "causal_conv1d", "torch", "triton"), env=None,
                doc="which depthwise causal conv serves the layer's convolutions: 'auto' (default) = 'triton', the plain-Triton "
                    "conv (`kernels.conv_triton`: any backend, no package, within 1.1x of the package at the LM shapes; "
                    "the layer's conv_kernel_in_use names the choice), 'causal_conv1d' (the compiled package, an OPT-IN extra; raises by "
                    "name when it is not importable) or 'torch' (F.conv1d, no package)",
                open_until="the package routes are deleted")


def _int_dict(v) -> bool:
    return v is None or (isinstance(v, dict) and all(
        isinstance(k, str) and isinstance(x, int) for k, x in v.items()))


_SSD_LAYER_OPTS: tuple[_Opt, ...] = (
    _m3_kernel_opt(),
    _conv_kernel_opt(),
    _bool_opt("compile_token_maps", False,
              "torch.compile the token-map tails (dt/a softplus chains)",
              "the compiled tails are measured to win or lose everywhere"),
    _bool_opt("allow_chunk_autotune", True,
              "allow the first-call chunk-size measurement under chunk_size='measure'",
              "chunk_size='measure' is removed or always on"),
    _Opt("mode", default="modular", values=("modular", "fused"), env=None,
         doc="'fused': the whole layer is ONE autograd node (the combined node, "
             "d_ssm_in = d_ssm_out = 1 CUDA envelope; outside it the layer silently runs modular); "
             "'modular': the ordinary multi-node graph",
         open_until="the combined node covers the envelope of the modular graph or is deleted"),
    _bool_opt("modular_checkpoint", False,
              "activation-checkpoint the modular graph (needs kernel_options mode='modular')",
              "the modular graph's memory matches the checkpointed one"),
    _bool_opt("registered_m2", False,
              "the registered Mamba-2 route: one fixed scan+epilogue custom op between torch "
              "projections and tail (CUDA, precision='io', M2 geometry; refused by name "
              "otherwise).  Internal opt-in",
              "the registered route replaces the modular graph at the M2 cell or is deleted"),
)

_HYBRID_MAMBA_LAYER_OPTS: tuple[_Opt, ...] = (
    _m3_kernel_opt(),
    _conv_kernel_opt(),
    _Opt("mem", default=None, check=_int_dict, env=None,
         doc="backward memory dial of the MH kernel, e.g. {'AGX': 0, 'BCS': 0} "
             "(the determinism pin); None = the kernel's own policy",
         open_until="the determinism pin has a named tier of its own"),
    _bool_opt("compile_gate", False,
              "torch.compile the SwiGLU gate (+ RMSNorm) epilogue",
              "the compiled gate is measured to win or lose everywhere"),
    _Opt("norm_kernel", default="triton", values=("triton", "torch", "mamba_ssm"), env=None,
         doc="which kernel evaluates the fused gated RMSNorm epilogue on CUDA: 'triton' (ours, "
             "orbit_ssm.kernels.gated_norm), 'torch' (F.rms_norm on y * silu(z)) or 'mamba_ssm' "
             "(explicit opt-in to mamba_ssm's kernel; raises by name when the package is not "
             "importable).  CPU and fp64 always use the plain formula",
         open_until="the GPU micro-benchmark (`benchmarks/bench_gated_norm.py`) confirms our kernel "
                    "is not slower than mamba_ssm's; then 'mamba_ssm' retires with the baselines"),
)

_SCHEMAS: dict[str, dict[str, _Opt]] = {
    "ssd_mimo": {o.key: o for o in _SSD_MIMO_OPTS + _SSD_MIMO_NUMERIC_OPTS + _SSD_LAYER_OPTS},
    "hybrid_mamba": {o.key: o for o in _HYBRID_MAMBA_OPTS + _HYBRID_MAMBA_NUMERIC_OPTS
                     + _HYBRID_MAMBA_LAYER_OPTS},
}

#: Per family, the keys the LAYER reads itself (not a kernel lever).
LAYER_KEYS: dict[str, frozenset[str]] = {
    "ssd_mimo": frozenset(o.key for o in _SSD_LAYER_OPTS),
    "hybrid_mamba": frozenset(o.key for o in _HYBRID_MAMBA_LAYER_OPTS),
}


def split(family: str, options: Any, who: str | None = None) -> tuple[dict, dict]:
    """Validate ``options``; return ``(kernel_keys, layer_keys)`` (kernels see the first via `push`)."""
    opts = validate(family, options, who)
    lay = LAYER_KEYS[family]
    return ({k: v for k, v in opts.items() if k not in lay},
            {k: v for k, v in opts.items() if k in lay})

FAMILIES: tuple[str, ...] = tuple(sorted(_SCHEMAS))

#: Per family, the keys `kernel_options` accepts.  An empty set is a
#: real answer: every key is then refused by name.
SUPPORTED_KERNEL_OPTIONS: dict[str, frozenset[str]] = {
    f: frozenset(s) for f, s in _SCHEMAS.items()
}


def _on_family(fam) -> None:
    """Take the layer switches of a registered family (`orbit_ssm.families`) into the tables."""
    global FAMILIES
    if fam.name in _SCHEMAS:
        return
    _SCHEMAS[fam.name] = {o.key: o for o in fam.kernel_options}
    LAYER_KEYS[fam.name] = frozenset(_SCHEMAS[fam.name])
    SUPPORTED_KERNEL_OPTIONS[fam.name] = frozenset(_SCHEMAS[fam.name])
    FAMILIES = tuple(sorted(_SCHEMAS))


_families.on_register(_on_family)

# Env prefix per family; only used to report provenance (the kernels' `_env()` reads the env).
_ENV_PREFIX = {"ssd_mimo": "SSD_MIMO_", "hybrid_mamba": "HYBRID_MH_"}


_CHECKED = False


def _ensure_checked() -> None:
    """Run `_check_schema` once, lazily (an import-time `config` import would be circular)."""
    global _CHECKED
    if not _CHECKED:
        _CHECKED = True
        _check_schema()


def _check_schema() -> None:
    """Rules 2 and 3 of the module docstring; also that defaults are JSON-safe."""
    from dataclasses import fields as _fields
    from .config import _CoreConfigBase
    from .execution import Execution
    reserved = ({f.name for f in _fields(_CoreConfigBase)}
                | {f.name for f in _fields(Execution)}) - {"kernel_options"}
    for fam, sch in _SCHEMAS.items():
        for key, opt in sch.items():
            if key in reserved:
                raise KernelOptionError(
                    f"kernel_options[{fam}]: key {key!r} collides with a "
                    "config or Execution field; a kernel option selects routing "
                    "only and may never be a second channel for a "
                    "parameter-bearing setting")
            if not opt.open_until:
                raise KernelOptionError(
                    f"kernel_options[{fam}]: key {key!r} declares no "
                    "`open_until`; a key that no measurement could "
                    "retire does not belong on this surface")
            if not _json_safe(opt.default):
                raise KernelOptionError(
                    f"kernel_options[{fam}]: default for {key!r} is not "
                    "JSON-serialisable; the surface rides the model "
                    "recipe")


# ── validation ───────────────────────────────────────────────────────

def schema(family: str) -> dict:
    """The family's `{key: _Opt}` map (read-only by convention)."""
    try:
        return _SCHEMAS[family]
    except KeyError:
        raise KernelOptionError(
            f"kernel_options: unknown family {family!r}; "
            f"supported: {', '.join(FAMILIES)}") from None


def describe(family: str) -> str:
    """The allowed ``kernel_options`` keys of a family, as a text block for layer docstrings.

    Parameters
    ----------
    family : str
        a name of `FAMILIES` (``"ssd_mimo"``, ``"hybrid_mamba"``, registered families); anything
        else raises `KernelOptionError`.

    Returns
    -------
    str
        One line per key, sorted: ``- ``key`` (default D): admissible values`` (the
        value list, or the first line of the key's description).  Pass the keys as
        ``Execution(kernel_options={...})``; unknown keys or unsupported values
        raise `KernelOptionError`.
    """
    lines = [f"``kernel_options`` keys of the {family!r} family (all optional; an unknown key or an "
             "unsupported value raises):", ""]
    for key, opt in sorted(schema(family).items()):
        values = " | ".join(repr(v) for v in opt.values) if opt.values is not None else opt.doc.split("\n")[0]
        lines.append(f"- ``{key}`` (default {opt.default!r}): {values}")
    return "\n".join(lines)


def defaults(family: str) -> dict:
    """Every supported key at its family default."""
    return {k: o.default for k, o in schema(family).items()}


def validate(family: str, options: Any, who: str | None = None) -> dict:
    """`options` as a plain validated dict (raises by name); None and {} give {}."""
    _ensure_checked()
    who = who or f"kernel_options[{family!r}]"
    sch = schema(family)
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise KernelOptionError(
            f"{who}: must be a dict (plain mapping), got "
            f"{type(options).__name__}")
    supported = ", ".join(sorted(sch)) or (
        "(none -- this family supports no kernel options; "
        "an empty set is a real answer)")
    out: dict = {}
    for key in sorted(options):
        if not isinstance(key, str):
            raise KernelOptionError(
                f"{who}: keys must be strings, got {key!r}")
        opt = sch.get(key)
        if opt is None:
            raise KernelOptionError(
                f"{who}: unknown kernel option {key!r}; "
                f"supported keys for family {family!r}: {supported}")
        value = options[key]
        if not _json_safe(value):
            raise KernelOptionError(
                f"{who}: kernel option {key!r} value {value!r} is not "
                "JSON-serialisable; the surface rides the model recipe")
        if not opt.accepts(value):
            raise KernelOptionError(
                f"{who}: kernel option {key!r} does not support value "
                f"{value!r}; allowed: {opt.allowed()}")
        out[key] = value
    if family == "ssd_mimo" and "autotune" in out:
        warn_autotune_tier_mismatch(out["autotune"])
    return out


class AutotuneTierWarning(UserWarning):
    """A ``kernel_options['autotune']`` tier the import-time backward grids do not follow."""


_WARNED_TIERS: set = set()


def imported_autotune_tier() -> str:
    """The tier the fused-backward / SISO autotune grids are built with.

    They are decorator arguments, fixed when `kernels.ssd_mimo.fused_bwd_kernel` is
    imported (``autotune_policy.autotune_tier`` of the SSD ``_env("AUTOTUNE")`` lever).
    This imports that module if needed (any `kernels.ssd_mimo` import does; ~0.4 s
    once) and returns its recorded tier, so there is no second copy of the env
    precedence or the alias table to drift (R10); an invalid ``SSD_MIMO_AUTOTUNE``
    raises the import's ValueError.

    Side effect (fix round 2): `validate` calls this only when a config names
    ``autotune``, and then the backward grids are fixed at the first validation
    (`split` at layer construction, or a scoped `push`), not at the first kernel call.  A light re-derivation is not
    possible without a second copy: the reader `_env` (kernel_options override, then
    ``SSD_MIMO_``) lives in `fused_fwd_kernel`, a calibration code
    file whose hash must not move.  Set ``SSD_MIMO_AUTOTUNE`` before building the model.
    """
    from .kernels.ssd_mimo import fused_bwd_kernel
    return fused_bwd_kernel._AUTOTUNE_TIER


def warn_autotune_tier_mismatch(requested: str) -> None:
    """Warn, by name and once per (requested, imported) pair, when a
    config ``autotune`` tier differs from the import-time grids' tier.  The config value
    steers the measure-based forward / chunk / shared-memory searches only; the fused
    backward and SISO grids keep the tier they were imported with."""
    imp = imported_autotune_tier()
    from .kernels.ssd_mimo.autotune_policy import autotune_tier
    req = autotune_tier(requested)
    if req == imp or (req, imp) in _WARNED_TIERS:
        return
    _WARNED_TIERS.add((req, imp))
    import warnings
    warnings.warn(
        f"kernel_options['autotune']={requested!r} (tier {req!r}) does not reach the "
        f"fused-backward and SISO autotune grids: they were built at import with tier "
        f"{imp!r} (SSD_MIMO_AUTOTUNE).  Set SSD_MIMO_AUTOTUNE={req} before importing "
        f"orbit_ssm for the backward to follow; the config value covers the measure-based "
        f"forward / chunk / shared-memory searches only.",
        AutotuneTierWarning, stacklevel=3)


def resolve(family: str, options: Any = None) -> dict:
    """Every supported key with its in-force value (explicit option > env lever > default)."""
    opts = validate(family, options)
    out = {}
    for key, opt in schema(family).items():
        value, _ = _effective(family, key, opt, opts)
        out[key] = value
    return out


def _env_value(family: str, opt: _Opt) -> str | None:
    """The raw env string for `opt`, or None when nothing is set (always None for an option
    without an environment spelling: ``env=None`` or ``env_read=False``)."""
    if opt.env is None or not opt.env_read:
        return None
    v = os.environ.get(_ENV_PREFIX.get(family, "") + opt.env)
    if v is None or v.strip() == "":
        return None
    return v


def _effective(family: str, key: str, opt: _Opt, opts: dict):
    """`(value, source)` with source in {"config", "env", "default"}."""
    if key in opts:
        return opts[key], "config"
    raw = _env_value(family, opt)
    if raw is not None:
        return _coerce(opt, raw), "env"
    return opt.default, "default"


def _coerce(opt: _Opt, raw: str):
    """Best-effort typed read of an env string, for reporting only; unparseable values stay verbatim."""
    raw = raw.strip()
    if isinstance(opt.default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if opt.values is not None and True in opt.values and False in opt.values:
        # a tri-state such as ('auto', False, True): bool-like spellings are
        # the bools the kernel reads, anything else is reported verbatim
        low = raw.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    if isinstance(opt.default, str) and opt.values is not None:
        return raw                     # a closed set of spellings
    try:
        return int(raw)
    except ValueError:
        return raw


def non_default(family: str, options: Any = None) -> dict:
    """Only the keys whose in-force value is NOT the family default."""
    values, _ = record(family, options)
    return values


def record(family: str, options: Any = None) -> tuple[dict, dict]:
    """`(values, sources)` for `dispatch_log()`: the keys whose in-force value is not the
    default, and their source ("config" or "env").  Defaults are not recorded."""
    if options is None:
        options = current(family)
    opts = options if isinstance(options, dict) else {}
    values: dict = {}
    sources: dict = {}
    for key, opt in schema(family).items():
        value, src = _effective(family, key, opt, opts)
        if src != "default" and value != opt.default:
            values[key] = value
            sources[key] = src
    return values, sources


# ── the active scope ─────────────────────────────────────────────────
# The kernels' `_env` readers sit deep inside a launch; instead of threading options
# through every frame, a scope is pushed around the scan entry and popped in `finally`.

_ACTIVE = threading.local()


def _stacks() -> dict:
    st = getattr(_ACTIVE, "stacks", None)
    if st is None:
        st = {}
        _ACTIVE.stacks = st
    return st


def _with_target(family: str, opts: dict) -> dict:
    """A layer running under ``kernel_target="portable"`` (`kernels.portable.use`) hands that choice to the scans it
    calls, so it travels with the scan's options (an autograd backward re-pushes them, outside the layer's scope)."""
    if "kernel_target" in opts or "kernel_target" not in _SCHEMAS.get(family, ()):
        return opts
    from .kernels.portable import scope_request
    r = scope_request()
    return opts if r is None or r == "auto" else {**opts, "kernel_target": r}


def push(family: str, options: Any) -> tuple:
    """Make `options` active for `family`; returns a token for `pop`.  ``None`` inherits the scope
    already active (a scan called without ``kernel_options`` inside a ``scope``); ``{}`` is an
    explicit empty scope."""
    opts = validate(family, options) if options else (dict(current(family)) if options is None else {})
    opts = _with_target(family, opts)
    stack = _stacks().setdefault(family, [])
    stack.append(opts)
    return (family, len(stack))


def pop(token: tuple) -> None:
    stack = _stacks().get(token[0])
    if stack:
        stack.pop()


@contextlib.contextmanager
def scope(family: str, options: Any):
    """``with scope(family, options):`` -- `options` active for the block (`push` / `pop`)."""
    token = push(family, options)
    try:
        yield
    finally:
        pop(token)


def current(family: str) -> dict:
    """Options active now for `family` ({} when none); thread-local."""
    st = getattr(_ACTIVE, "stacks", None)
    if not st:
        return {}
    stack = st.get(family)
    return stack[-1] if stack else {}


def value(family: str, key: str, options: Any = None) -> Any:
    """The in-force value of option ``key``: ``options`` (default: the active scope of `push`) if it
    names the key, else the schema default.  The reader of an option that has no environment spelling."""
    opts = current(family) if options is None else options
    if key in opts:
        return opts[key]
    return schema(family)[key].default


def option_only(family: str, env_suffix: str) -> bool:
    """True when ``env_suffix`` names a registered option that is configuration-only (``env_read=False``):
    a kernel `_env(<suffix>)` reader must then answer from the option scope or its own default,
    never from the process environment."""
    return any(o.env == env_suffix and not o.env_read for o in schema(family).values())


def env_override(family: str, env_suffix: str) -> str | None:
    """What an `_env(env_suffix, ...)` reader should return, or None (nothing configured: read the env)."""
    opts = current(family)
    if not opts:
        return None
    for key, opt in schema(family).items():
        if opt.env == env_suffix and key in opts:
            return opt.to_env(opts[key])
    return None


def scoped_kwarg(family: str, kwarg: str = "kernel_options"):
    """Decorator: activate `kwargs[kwarg]` for the duration of the call."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            token = push(family, kwargs.get(kwarg))
            try:
                return fn(*args, **kwargs)
            finally:
                pop(token)
        return wrapper
    return deco


def scoped_ctx(family: str, attr: str = "_kernel_options"):
    """Decorator for an autograd `backward(ctx, ...)`: re-activates the options saved on ``ctx``,
    since the recompute replay must be scheduled as the forward was."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            token = push(family, getattr(ctx, attr, None))
            try:
                return fn(ctx, *args, **kwargs)
            finally:
                pop(token)
        return wrapper
    return deco
