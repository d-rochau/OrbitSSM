"""SSM-level rows: each production mixer against its original Mamba layer.

The timing peer of the mixer-level correctness tests.  This is the module that fills the SIGNED
contract's Section 4 timing table — "rows = Section 2 configs + the
campaign configs (mhR2O2c/r, mhR4O4c), columns = fwd /
fwd+bwd x fp32 / tf32 / bf16 at MQMKVAR shape (B32 L2048 d256-class) and
LM spine", with the standing rule **no row, no launch**.

Three things make an SSM-level number different from a kernel-level one
and all three are recorded:

1. **Glue is not free.**  A profiled step at the audit shape put ~36% of
   the CUDA time outside the scan kernels (projections, conv, norms,
   permutes).  `--profile` adds a `cuda_buckets` tag splitting the
   self-CUDA time into fwd / bwd / state_passing / glue so a kernel-only
   ratio cannot be mistaken for a layer ratio.
2. **Parameter counts must be comparable.**  The standing rule is that an
   arm carries AT MOST the baseline's parameters, within 1% below.  Every
   record carries `params` and, where a baseline is in the same grid, the
   ratio — so an unmatched row is visibly flagged instead of silently
   compared.
3. **bf16 at this level is AUTOCAST, not a parameter cast.**  Modules keep
   fp32 parameters in every column and the bf16 column runs inside
   `torch.autocast`; `.to(bfloat16)` on a whole module is not a
   configuration anybody trains, and it makes every `residual_in_fp32`
   baseline raise rather than be measured.  Records carry
   `autocast=True` and `param_dtype="float32"`.

Rows are built through the single uniform path `<Family>Config(...).build(
d_model, execution=...)` over `orbit_ssm.ssm.CONFIGS`, never per-family
ad-hoc kwargs, so a config that a family refuses raises here exactly as it
would in a training run.

4. **Two mixer LEVELS, both visible.**  The `core:` rows are the BARE scan
   (the private `_SSDMimoScan` / `_HybridMambaScan`: no in_proj/conv1d/
   gate/norm/out_proj) for `ssd_mimo` / `hybrid_mamba`, while the rows of a family
   that already owns its projection stack (a registered case family, see
   `register_case_family`) are complete layers.  Ratioing
   the first kind's bare core against an official's full mixer is a 16x
   parameter mismatch, not the like-for-like module comparison the
   contract's table presents it as (finding B2).  Both `ssm_lm_spine` and
   `ssm_mqmkvar` therefore carry BOTH levels for `ssd_mimo` /
   `hybrid_mamba`: `core:<name>` (the bare core, `_ssm_cell`, numbers
   unchanged) and the full `orbit_ssm.SSDMimoSSM` /
   `HybridMambaSSM` layer (`_full_mixer_cell`) at the officials' own
   `(d_model, expand)` surface, tagged `mixer_level` in every record.
5. **Exact-config arms are the ONLY like-for-like
   row.**  Every arm above -- default OR bare-core-vs-full-mixer-fixed
   -- still runs OUR feature/width choices against an official's exact
   config, which is a "nonsensical comparison" once stated
   as a headline.  `SM-R1:m2exact` / `SM-R1:m3exact` / `HMM:m1exact`
   (`_exact_mixer_cell`, built from `orbit_ssm.presets`) are the actual
   like-for-like rows, in the SAME group as their `official_mamba{1,2,3}`
   counterpart at the SAME width; every arm's `tags.config` says which
   kind of row it is (`"m1exact"`/`"m2exact"`/`"m3exact"`/`"official"`/
   `"default"`).  See `benchmarks/README.md`'s exact-config section.

Grids:
    smoke          tiny, every family + every available baseline
    ssm_mqmkvar    the campaign widths (S1-S5) at B32 L2048
    ssm_lm_spine   B8 D2048 L2048 N64 — the contract's second shape
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import torch

from . import baselines as bl
from . import shapes as sh
from .harness import (
    KERNEL_PRECISION, MODULE_PARAM_DTYPE, Cell, amp_ctx, cuda_buckets,
    group_key, main_for, make_pass,
)
from .cases import _ScanOnly, split_execution
from .measure import precision_ctx
from orbit_ssm import HybridMambaConfig, SSDMimoConfig

GRIDS = ("smoke", "ssm_mqmkvar", "ssm_lm_spine", "ssm_shared_set",
         "ssm_shared_small", "ssm_shared_medium", "ssm_shared_large", "ssm_shared_lm")
# `ssm_shared_set` is the UNION; its parts (one size class / the LM cells each) are the shard-planner's units -- one 432-cell
# unit is 2.4x the others and no bin-pack of 4 shards can balance it.  Every part's cells still carry grid = "ssm_shared_set".
UNION_GRIDS = frozenset({"ssm_shared_set"})

# Per-family config factories.  Further case families (arms of a registered family) are added by
# `cases.register_case_family`; every module `benchmarks/*_cases.py` that is present registers its arms
# when `cases.load_extensions` imports it (below).
from .cases import (CASE_CONFIGS, CASE_SMOKE, FAMILIES, GRID_HOOKS, PAPER_TF32_ARMS,  # noqa: E402
                    load_extensions)
load_extensions()

# Families whose `core:` row is the BARE scan (no in_proj/conv1d/gate/norm/
# out_proj) rather than a complete layer.
# A case family that owns its full projection stack already -- a `_ssm_cell` row for
# it IS the like-for-like full-mixer row.  For these two
# families it is NOT: the finding this module fixes (a bf16 profile audit) is that "S-R1"/"S-R2O2"/"S-R4O4"
# (ssd_mimo) and "HM-LM"/"HM" (hybrid_mamba) were built and ratioed against
# FULL official mixers while being 16x-fewer-param bare cores themselves.
_BARE_CORE_FAMILIES = frozenset({"ssd_mimo", "hybrid_mamba"})


def _hm_pin_no_bias(kw: dict) -> None:
    """Pin an HM arm's B/C bias to its pre-spec default (none).

    Before the rows/rotation/maps spec
    `HybridMambaSSM` allocated no B/C bias unless asked (`bc_bias=False`);
    the spec default is `bc_bias="head"`.  Every HM arm here that did
    not ask for a bias keeps measuring the model it always measured."""
    kw.setdefault("bc_bias", "none")


def _config(family: str, d_inner: int, d_state: int, **over):
    from orbit_ssm.ssm import HybridMambaConfig, SSDMimoConfig
    common = dict(d_inner=d_inner, d_state=d_state)
    if family == "ssd_mimo":
        d_ssm_in = over.pop("d_ssm_in", 1)
        d_ssm_out = over.pop("d_ssm_out", d_ssm_in)
        n_groups = over.pop("n_groups", max(16 // d_ssm_in, 1))
        return SSDMimoConfig(**common, d_ssm_in=d_ssm_in, d_ssm_out=d_ssm_out, n_groups=n_groups,
                             proj_source="main_branch", **over)
    if family in CASE_CONFIGS:
        return CASE_CONFIGS[family](d_inner, d_state, **over)
    if family == "hybrid_mamba":
        # b_norm/c_norm pinned to their earlier value (False): the class
        # default flipped to True where there are HM rows -- this arm keeps running what it always ran
        # unless a caller overrides via `over`.
        # Likewise the B/C bias: none unless asked (`_hm_pin_no_bias`).
        _hm_pin_no_bias(over)
        return HybridMambaConfig(**common,
                                 n_heads=over.pop("n_heads",
                                                  max(d_inner // 16, 1)),
                                 b_norm=over.pop("b_norm", False),
                                 c_norm=over.pop("c_norm", False),
                                 proj_source="main_branch", **over)
    raise ValueError(f"unknown family {family!r}")


def build_mixer(family: str, d_inner: int, d_state: int, B: int, L: int, *,
                dtype=torch.float32, device="cuda", seed=0, **over):
    """Build one mixer and its input at (B, L, d_inner).

    `dtype` is the PARAMETER dtype and every caller in this package passes
    fp32: the bf16 column is autocast (see the module docstring), not a
    module cast.
    """
    torch.manual_seed(seed)
    over = dict(over)
    ex = split_execution(family, over, backend="cuda")
    cfg = _config(family, d_inner, d_state, **over)
    if family in _BARE_CORE_FAMILIES:
        # the bare scan: the layer's own `scan`, its own parameters only
        mod = _ScanOnly(cfg.build(execution=ex))
    else:
        mod = cfg.build(execution=ex)
    mod = mod.to(device=device, dtype=dtype)
    x = torch.randn(B, L, d_inner, device=device, dtype=dtype,
                    requires_grad=True)

    def call():
        return mod(x)

    return call, [x, *mod.parameters()], mod


def build_full_mixer(family: str, d_model: int, d_state: int, B: int, L: int, *,
                     dtype=torch.float32, device="cuda", seed=0,
                     expand: float = 2.0, d_conv: int = 4,
                     mixer_kwargs: dict | None = None):
    """Build the PRODUCTION mixer (in_proj -> conv1d+SiLU -> SSM core ->
    SwiGLU gate -> RMSNorm -> out_proj) around a bare-core family's SSM, at
    the officials' own (d_model, expand) surface -- the like-for-like
    counterpart to `build_mixer`'s bare-core builder.

    Only `ssd_mimo` and `hybrid_mamba` need this wrapper: a case family
    already owns its full projection stack
    (`src/orbit_ssm/model.py`'s mixer dispatch), so `build_mixer` /
    `_ssm_cell` already measures them at the full-mixer level (finding B2).
    """
    from orbit_ssm import HybridMambaSSM, SSDMimoSSM

    torch.manual_seed(seed)
    kw = dict(mixer_kwargs or {})
    ex = split_execution(family, kw)
    if family == "ssd_mimo":
        # b_norm / c_norm / bc_bias stay in the CORE kwargs; absent, the core
        # default (norms on, per-head B/C bias) is what the old mixer default
        # `bc_norm=True` built.
        mod = SSDMimoSSM(SSDMimoConfig(d_model=d_model, d_state=d_state, expand=expand,
                         d_conv=d_conv, **kw), execution=ex)
    elif family == "hybrid_mamba":
        # b_norm/c_norm pinned to their earlier value (the
        # class default has since flipped to True where there are
        # HM rows) so this arm keeps its current numbers unless the
        # caller's mixer_kwargs overrides them.
        kw.setdefault("b_norm", False)
        kw.setdefault("c_norm", False)
        # the same pin for the B/C bias: the pre-spec HM default was NO
        # bias (`bc_bias=False`); the spec default is bc_bias="head",
        # so an arm that never asked for a bias keeps none.
        _hm_pin_no_bias(kw)
        mod = HybridMambaSSM(HybridMambaConfig(d_model=d_model, d_state=d_state, expand=expand,
                             d_conv=d_conv, **kw), execution=ex)
    else:
        raise ValueError(
            f"build_full_mixer: {family!r} already owns its full projection "
            f"stack -- use build_mixer/_ssm_cell instead")
    mod = mod.to(device=device, dtype=dtype)
    x = torch.randn(B, L, d_model, device=device, dtype=dtype,
                    requires_grad=True)

    def call():
        return mod(x)

    return call, [x, *mod.parameters()], mod


def _cpu_param_count(build_module: Callable[[], Any]) -> int:
    """Cheap CPU-only parameter count for the grid-build-time param-match
    audit: no CUDA, no timing, just weight allocation -- so `--dry-run` and
    a real GPU run see identical tags and the audit costs nothing."""
    mod = build_module()
    return sum(p.numel() for p in mod.parameters())


# `cuda_buckets` (the self-CUDA fwd/bwd/state_passing/glue split) moved to
# `harness.py` so `bench_kernels.py` can share it for its
# own `ms_cuda_self` column; re-imported above so
# `benchmarks.bench_ssms.cuda_buckets` (quoted by several
# reports) still resolves.


# ── cells ────────────────────────────────────────────────────────────
def _ssm_cell(grid, family, arm, d_inner, d_state, B, L, precision, pass_, *,
              work, tags=None, **over):
    """One mixer row.  `work` is the group's constant-work key (required:
    an arm alone in its group has no trustworthy ratio); the module is
    built in fp32 and bf16 is reached by autocast."""
    amp = amp_ctx(precision)

    def build():
        call, leaves, mod = build_mixer(family, d_inner, d_state, B, L,
                                        dtype=MODULE_PARAM_DTYPE, **over)
        _PARAMS[f"{family}/{arm}/{d_inner}"] = sum(
            p.numel() for p in mod.parameters())
        return make_pass(call, leaves, pass_, amp=amp)

    return Cell(
        family="ssm", arm=arm,
        entry_point=f"orbit_ssm.ssm.CONFIGS[{family!r}].build",
        shape=dict(B=B, L=L, d_inner=d_inner, N=d_state,
                   **{k: v for k, v in over.items() if isinstance(v, int)}),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        autocast=(precision == "bf16"),
        tags=dict(tags or {}, mixer=family,
                  mixer_level=("core" if family in _BARE_CORE_FAMILIES
                              else "full")),
        build=build,
    )


def _full_mixer_cell(grid, family, arm, d_model, d_state, B, L, precision,
                     pass_, *, work, expand: float = 2.0, tags=None,
                     graphed=False, **mixer_kwargs):
    """One FULL-MIXER row: in_proj/conv1d/SwiGLU-gate/RMSNorm/out_proj
    around a bare-core family's SSM, at the officials' own (d_model,
    expand) surface -- the like-for-like counterpart to `_ssm_cell`'s
    bare-core rows for `ssd_mimo` / `hybrid_mamba` (finding B2).  `d_model` is the MIXER's input
    width (matching the officials' own `d_model`); `d_inner = d_model *
    expand` is derived, not passed."""
    amp = amp_ctx(precision)

    def build():
        call, leaves, mod = build_full_mixer(
            family, d_model, d_state, B, L, dtype=MODULE_PARAM_DTYPE,
            expand=expand, mixer_kwargs=dict(mixer_kwargs))
        _PARAMS[f"{family}/{arm}/{d_model}"] = sum(
            p.numel() for p in mod.parameters())
        return (_graphed_closure(mod, leaves, pass_, precision) if graphed
                else make_pass(call, leaves, pass_, amp=amp))

    d_inner = int(d_model * expand)
    wrapper_cls = "SSDMimoSSM" if family == "ssd_mimo" else "HybridMambaSSM"
    return Cell(
        family="ssm", arm=arm,
        entry_point=f"orbit_ssm.{wrapper_cls}",
        shape=dict(B=B, L=L, d_model=d_model, d_inner=d_inner, N=d_state,
                   **{k: v for k, v in mixer_kwargs.items()
                      if isinstance(v, int)}),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        autocast=(precision == "bf16"),
        tags=dict(tags or {}, mixer=family, mixer_level="full"), build=build,
    )


# ═════════════════════════════════════════════════════════════════════
# G34: the captured-vs-captured regime
# ═════════════════════════════════════════════════════════════════════
# Rationale: "the kernel advantage
# only shows on wall under graphs; every quoted number captured vs
# captured".  G32 measured BOTH mixers host-bound at both paper cells
# (wall == host enqueue; 1.15-1.65 ms of GPU idle each) and G33 removed
# every glue item without moving the wall, so an eager wall ratio at
# these cells prices the python/ATen enqueue path, not the kernels.
# `--graphed` adds an `<arm>@graph` row for each like-for-like pair --
# OURS AND THE OFFICIALS ALIKE -- whose timed closure is one
# `CUDAGraph.replay()` of a whole fwd(+bwd) step captured by
# `orbit_ssm.graphs.graphed_step`.  Same group, same budget, same
# canary rules as every other arm, so the eager and captured rows of one
# arm are interleaved with each other and with the other arm's.
# bf16 only and both passes: the captured regime exists to serve the
# paper's bf16 like-for-like table.
_GRAPHED_LIKE_FOR_LIKE = ("SM-R1:m2exact", "SM-R1:m3exact", "HMM:m1exact",
                          "official_mamba1", "official_mamba2",
                          "official_mamba3")
GRAPH_ARM_SUFFIX = "@graph"
COMPILE_ARM_SUFFIX = "@compile"


def _compiled_closure(mod, leaves, pass_, precision, amp, arm):
    """The timed closure of a `@compile` row: the module under `torch.compile`
    (default mode), the SAME input and `make_pass` treatment as the eager row.

    Compile time is EXCLUDED from the timed rounds: the first forward (and, for
    fwd_bwd, the backward) runs here, inside `build`, and its seconds are hung on
    the closure (`compile_seconds`) for `_post_cell` to record separately.  A
    module that fails to compile raises a NAMED error, which the harness records
    as that cell's failure -- never a silent drop."""
    import time
    import torch._dynamo
    torch._dynamo.reset()                          # per-cell cache: no recompile-limit spill
    x = leaves[0]
    try:
        cmod = torch.compile(mod)
        t0 = time.perf_counter()
        fn = make_pass(lambda: cmod(x), leaves, pass_, amp=amp)
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
    except Exception as exc:                       # noqa: BLE001
        raise RuntimeError(f"{arm}: torch.compile failed ({type(exc).__name__}): "
                           f"{str(exc).splitlines()[0][:300]}") from exc
    fn.compile_seconds = secs
    fn.compile_mode = "default"
    return fn


def _with_module(mod, leaves, amp, pass_, fn):
    """Hang the module and an EAGER call (forward, + backward for fwd_bwd) of the
    same module/input on the timed closure: provenance (`route_record`) reads
    the route from an eager call (a graph replay runs no host dispatch)."""
    x = leaves[0]

    def eager():
        if pass_ == "fwd":
            with torch.no_grad(), amp():
                return mod(x)
        with amp():
            y = mod(x)
        y.float().square().mean().backward()
        for q in leaves:
            if isinstance(q, torch.Tensor):
                q.grad = None
        return y
    fn.module = mod
    fn.eager = eager
    return fn


def _graphed_closure(mod, leaves, pass_, precision):
    """The timed closure of an `@graph` row: one replay, nothing else.

    Deliberately NOT `copy_in` + replay.  The eager arms reuse their own
    `x` from iteration to iteration (`make_pass` builds the closure over
    one fixed input), so charging the captured arm for a copy the eager
    one never makes would be a handicap, not a measurement; a real
    trainer's H2D of the next batch is paid by both regimes alike.

    The `GraphedStep` is hung on the returned function so it stays alive
    (it owns the graph, the static input/dY buffers and the gradient
    buffers) and so `_post_cell` can record its memory cost.
    """
    from orbit_ssm import graphs as gph

    x = leaves[0]
    gs = gph.graphed_step(mod, (x,), fwd_bwd=(pass_ == "fwd_bwd"),
                          amp=lambda: gph.autocast(precision), warmup=11)

    def run():
        gs.graph.replay()
        return gs.static_output

    run.graphed_step = gs
    return run


def _official_cell(grid, variant, d_model, d_state, B, L, precision, pass_, *,
                   work, headdim=64, ngroups=1, expand=2, tags=None,
                   arm=None, graphed=False, compiled=False):
    """One official Mamba mixer row.  `arm` defaults to `f"official_
    {variant}"`; the state-size split needs a SECOND `official_mamba2`/
    `official_mamba3` cell in the SAME group (the N=64 secondary row
    next to the new N=128 headline), which collide under one literal
    name in `emit_tables.emit_like_for_like`'s `by_group[group][arm]`
    dict -- pass `arm="official_mamba2@N64"` for that row."""
    amp = amp_ctx(precision)
    arm = arm or f"official_{variant}"
    if graphed:
        arm += GRAPH_ARM_SUFFIX
    if compiled:
        arm += COMPILE_ARM_SUFFIX

    def build():
        call, leaves, mod = bl.build_mamba_layer(
            variant, d_model, d_state=d_state, L=L, B=B, headdim=headdim,
            ngroups=ngroups, expand=expand, dtype=MODULE_PARAM_DTYPE)
        _PARAMS[f"official/{variant}/{d_model}"] = sum(
            p.numel() for p in mod.parameters())
        return _with_module(mod, leaves, amp, pass_, (
            _graphed_closure(mod, leaves, pass_, precision) if graphed else
            _compiled_closure(mod, leaves, pass_, precision, amp, arm) if compiled else
            make_pass(call, leaves, pass_, amp=amp)))

    return Cell(
        family="ssm", arm=arm,
        entry_point=f"mamba_ssm.{variant}",
        shape=dict(B=B, L=L, d_inner=d_model * expand, N=d_state,
                   d_model=d_model),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        autocast=(precision == "bf16"),
        tags=dict(tags or {}, mixer=variant,
                  **({"capture": "cuda_graph"} if graphed else {}),
                  **({"capture": "torch_compile"} if compiled else {})),
        build=build,
    )


# Parameter counts, filled as each cell is built.  The SAME dict object is
# handed to `run_cells` as run metadata, so every record sees the counts
# that were collected after the metadata was assembled — the records are
# serialised at write time, not at build time.  `params_by_arm` is what
# the standing param-match rule (at most the baseline's params, within 1%
# below) is checked against; `emit_tables` flags an unmatched row.
_PARAMS: dict[str, int] = {}


# ═════════════════════════════════════════════════════════════════════
# Exact-config arms ("reproduce their exact configs" headline rows)
# ═════════════════════════════════════════════════════════════════════
# Design decision:
# "If we compare them, we reproduce their exact configs; if we do
# additional computations it is clear we might take more time, but that
# is a nonsensical comparison."  Every DEFAULT arm above (`SM-R1`,
# `HMM-LM`, `core:S-R1`, ...) stays exactly as it was -- it is the
# SEPARATE, priced "our features cost this much" row also asked for --
# and the three rows built here are the actual like-for-like headline:
# our own layer classes (`SSDMimoSSM`, `HybridMambaSSM`), built at
# the OFFICIAL model's exact configuration through
# `orbit_ssm.presets.mamba{1,2,3}_kwargs` (the same presets
# the mixer exactness tests validate against copied official
# parameters), timed in the SAME group as `_official_cell`'s official
# module at the same (d_model, d_state, headdim).
_EXACT_PRESET = {"m1exact": "mamba1", "m2exact": "mamba2", "m3exact": "mamba3"}
_EXACT_FAMILY = {"m1exact": "hybrid_mamba", "m2exact": "ssd_mimo",
                 "m3exact": "ssd_mimo"}


def build_preset_mixer(config_name: str, d_model: int, d_state: int, B: int,
                       L: int, *, dtype=torch.float32, device="cuda",
                       seed=0, headdim: int = 64, **overrides):
    """Build ONE exact-config mixer (`config_name` in "m1exact" |
    "m2exact" | "m3exact") straight from `orbit_ssm.presets`, and its
    input at (B, L, d_model).  `headdim` is ignored for "m1exact"
    (Mamba-1 has no head-width knob -- `mamba1_config` fixes
    `d_head=1`, per-channel dt); `**overrides` forward into the
    preset's flat core options (e.g. a `backend` pin)."""
    from orbit_ssm import presets
    from orbit_ssm import HybridMambaSSM, SSDMimoSSM

    if config_name not in _EXACT_PRESET:
        raise ValueError(f"unknown exact config {config_name!r}, must be "
                         f"one of {sorted(_EXACT_PRESET)}")
    torch.manual_seed(seed)
    over = dict(overrides)
    # how the layer runs (backend, precision, mode "fused" = the G9 one-node
    # route / "modular" = the default now, ...) is its `Execution`
    ex = split_execution(_EXACT_FAMILY[config_name], over, backend="cuda")
    if config_name == "m1exact":
        mod = HybridMambaSSM(presets.mamba1_config(d_model, d_state=d_state, **over), execution=ex)
    else:
        preset_fn = (presets.mamba2_config if config_name == "m2exact"
                    else presets.mamba3_config)
        mod = SSDMimoSSM(preset_fn(d_model, d_state=d_state, d_head=headdim, **over), execution=ex)
    mod = mod.to(device=device, dtype=dtype)
    # the input comes from a FIXED generator (not the global RNG, whose state
    # depends on what was initialised before): a pair's two arms see the same x
    x = bl.module_input(B, L, d_model, device=device, dtype=dtype, seed=seed)

    def call():
        return mod(x)

    return call, [x, *mod.parameters()], mod


_OFFICIAL_OF = {"m1exact": "mamba1", "m2exact": "mamba2", "m3exact": "mamba3"}
# rel-L2 gates of the copied-weight pair (forward y and input gradient dx),
# per precision column.  Two bf16-autocast arms that each round differently
# agree to a few 1e-3 (measured 0.35 / 0.10 / 0.41 % for M1/M2/M3);
# the gate is the defect-catcher above that, not an accuracy claim.
PAIR_GATE = {"bf16": (2e-2, 5e-2), "tf32": (1e-2, 3e-2), "ieee": (1e-4, 1e-3)}
_PAIR_GATE_CACHE: dict = {}


def build_exact_pair(config_name: str, d_model: int, d_state: int, B: int,
                     L: int, *, dtype=torch.float32, device="cuda", seed=0,
                     headdim: int = 64, **overrides):
    """(official module, our exact-preset mixer) with EVERY official parameter
    copied into ours through `tests/_official_maps.py`, and no trainable
    parameter of ours left unmapped (refused otherwise).  Mamba-3: the target's
    |A| activation is the INSTALLED reference's (`mamba3_reference_a_act`), as
    the guarded copy helper requires.  Returns (official, ours)."""
    import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tests'))   # the official-module maps live with the tests
    import _official_maps as compat
    variant = _OFFICIAL_OF[config_name]
    _, _, official = bl.build_mamba_layer(
        variant, d_model, d_state=d_state, L=1, B=1, headdim=headdim,
        ngroups=1, expand=2, dtype=dtype, device=device, seed=seed)
    over = dict(overrides)
    if config_name == "m3exact":
        over.setdefault("a_act", compat.mamba3_reference_a_act(official))
    _, _, ours = build_preset_mixer(config_name, d_model, d_state, 1, 1,
                                    dtype=dtype, device=device, seed=seed,
                                    headdim=headdim, **over)
    if config_name == "m1exact":
        unmapped = compat.copy_official_mamba1_into_hm_mixer(official, ours)
    elif config_name == "m2exact":
        unmapped = compat.copy_official_mamba2_into_ssd_mimo_mixer(official, ours)
    else:
        unmapped = compat.copy_official_mamba3_into_ssd_mimo_mixer(official, ours)
    if unmapped:
        raise RuntimeError(f"{config_name}: trainable parameters without an "
                           f"official counterpart: {unmapped}")
    return official, ours


def _rel(a, b) -> float:
    a, b = a.detach().float(), b.detach().float()     # fp32 norms: no fp64 copies of large activations
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def exact_pair_gate(config_name, d_model, d_state, B, L, column, *,
                    seed=0, headdim=64, **overrides) -> dict:
    """Validate the copied-weight pair BEFORE anything is timed: one shared
    input x and one shared dY, forward y and input gradient dx rel-L2 against
    the official, plus the set of parameters that received a gradient on each
    side.  Raises AssertionError (=> a recorded cell FAILURE, never a timed
    row) above `PAIR_GATE[column]` (`column` = the precision COLUMN; a
    `precision=` in `overrides` is OUR kernel's precision kwarg).  Cached
    per (pair, shape, column, overrides)."""
    precision = column
    key = (config_name, d_model, d_state, B, L, precision, headdim,
           json.dumps(overrides, sort_keys=True, default=repr))
    if key in _PAIR_GATE_CACHE:
        res = _PAIR_GATE_CACHE[key]
        if isinstance(res, str):
            raise AssertionError(res)
        return res
    official, ours = build_exact_pair(config_name, d_model, d_state, B, L,
                                      dtype=MODULE_PARAM_DTYPE, seed=seed,
                                      headdim=headdim, **overrides)
    amp = amp_ctx(precision)
    x = bl.module_input(B, L, d_model, device="cuda", dtype=MODULE_PARAM_DTYPE,
                        seed=seed)
    out = {}
    dy = None
    with precision_ctx(precision):
        for name, mod in (("official", official), ("ours", ours)):
            xi = x.detach().clone().requires_grad_(True)
            with amp():
                y = mod(xi)
            if dy is None:
                g = torch.Generator(device="cuda").manual_seed(seed + 7)
                dy = torch.randn(y.shape, generator=g, device="cuda",
                                 dtype=torch.float32).to(y.dtype)
            y.backward(dy)
            out[name] = dict(y=y.detach().float(), dx=xi.grad.detach().float(),
                             with_grad=sorted(n for n, q in mod.named_parameters()
                                              if q.grad is not None),
                             n_params=sum(1 for q in mod.parameters() if q.requires_grad))
            del y, xi
    tol_y, tol_dx = PAIR_GATE[precision]
    res = dict(pair=f"{config_name}~official_{_OFFICIAL_OF[config_name]}",
               rel_y=_rel(out["ours"]["y"], out["official"]["y"]),
               rel_dx=_rel(out["ours"]["dx"], out["official"]["dx"]),
               tol_y=tol_y, tol_dx=tol_dx,
               grads_ours=f"{len(out['ours']['with_grad'])}/{out['ours']['n_params']}",
               grads_official=f"{len(out['official']['with_grad'])}/{out['official']['n_params']}",
               a_act=(ours.a_act if config_name == "m3exact" else None), weights="copied")
    del official, ours, out
    torch.cuda.empty_cache()
    bad = []
    if not res["rel_y"] < tol_y:
        bad.append(f"y rel {res['rel_y']:.3e} >= {tol_y}")
    if not res["rel_dx"] < tol_dx:
        bad.append(f"dx rel {res['rel_dx']:.3e} >= {tol_dx}")
    for side in ("ours", "official"):
        have, tot = res[f"grads_{side}"].split("/")
        if have != tot:
            bad.append(f"{side}: only {have}/{tot} parameters received a gradient")
    if bad:
        msg = f"parity gate {res['pair']} B{B} L{L} d{d_model} N{d_state} {precision}: " + "; ".join(bad)
        _PAIR_GATE_CACHE[key] = msg
        raise AssertionError(msg)
    _PAIR_GATE_CACHE[key] = res
    return res


def _exact_mixer_cell(grid, arm, config_name, d_model, d_state, B, L,
                      precision, pass_, *, work, headdim=64, tags=None,
                      config_tag=None, graphed=False, compiled=False, **overrides):
    """One EXACT-CONFIG full-mixer row.  Grouped with
    `_official_cell`'s official module at the SAME (d_model, d_state,
    headdim) -- that official IS the row this one is like-for-like
    against, and `tags.config` names which preset built it.

    `config_name` must be one of "m1exact"/"m2exact"/"m3exact" (it
    selects the family/preset), but `config_tag`
    lets the RECORDED `tags.config` differ from it -- the N=64 secondary
    row this pass adds next to the N=128 headline is still built via
    `config_name="m2exact"`/`"m3exact"` (same preset, same family) but
    tagged `config="m2exact_n64"`/`"m3exact_n64"` so `emit_like_for_like`
    (which reads `tags.config`, not the arm name, to find each exact
    row's family and the `default`-arm rows' reference) never conflates
    the two N variants in the SAME group.  Defaults to `config_name`
    (every pre-B9 call site is unaffected).

    D1 fix: `presets.mamba2_config`/`mamba3_config` hardcode
    the SSD-MIMO kernel's OWN `precision="ieee"` (no-tf32 GEMM) because
    that is the config their exactness tests validate against an fp64/
    fp32 oracle (the copy functions of `tests/_official_maps.py`) -- it has
    nothing to do with what the official Mamba-2/3 layers themselves do
    (they run no such GEMM at all).  Every OTHER arm in this grid
    (`SM-R1`, `official_mamba2`, ...) reaches the kernel at its class
    default `precision="tf32"` regardless of this cell's own precision
    column, because nothing here ever set it to anything else.  Left
    alone, that made `SM-R1:m2exact`/`SM-R1:m3exact` run a full-precision
    (no tensor-core) GEMM while `SM-R1` at the SAME (bf16 fwd+bwd) cell
    ran the fast tf32 one -- a 5-8x, config-comparison-breaking gap that
    has nothing to do with the officials being reproduced (measured:
    28.9/32.1 ms as-coded vs 6.3/12.4 ms once matched; SM-R1 4.5-4.7 ms).  `KERNEL_PRECISION`
    (`harness.py`) is the grid's own documented map from a cell's
    precision COLUMN to the kernel's `precision=` kwarg -- `bench_kernels
    .py` already applies it for the kernel-level grids; this arm was the
    one place in this module that built a real kernel without applying
    it.  An explicit `precision=` in `**overrides` still wins (kept
    overridable, never silently clobbered)."""
    amp = amp_ctx(precision)
    family = _EXACT_FAMILY[config_name]
    if graphed:
        arm += GRAPH_ARM_SUFFIX
    if compiled:
        arm += COMPILE_ARM_SUFFIX

    cell_runtime: dict = {}

    def build():
        over = dict(overrides)
        # `precision` (ieee|tf32 GEMM mode) only exists on the ssd_mimo
        # family's kernel -- HybridMambaSSM (m1exact) has no such GEMM
        # and REFUSES the kwarg outright (`reject_unknown_kwargs`); mh_scan
        # is a per-channel scalar recurrence, not chunked-GEMM SSD, so
        # there is nothing here for it to select between.
        if family == "ssd_mimo":
            over.setdefault("precision", KERNEL_PRECISION[precision])
        # the pair is validated (copied official weights, shared x and dY,
        # rel-L2 of y and dx) BEFORE this arm is timed; a failing gate raises
        # and the harness records the cell as a failure, never a timed row
        gate = exact_pair_gate(config_name, d_model, d_state, B, L, precision,
                               headdim=headdim, **over)
        official, mod = build_exact_pair(config_name, d_model, d_state, B, L,
                                         dtype=MODULE_PARAM_DTYPE,
                                         headdim=headdim, **over)
        del official
        x = bl.module_input(B, L, d_model, device="cuda",
                            dtype=MODULE_PARAM_DTYPE, seed=0)
        call, leaves = (lambda: mod(x)), [x, *mod.parameters()]
        cell_runtime["parity_gate"] = gate
        _PARAMS[f"{family}/{arm}/{d_model}"] = sum(
            p.numel() for p in mod.parameters())
        return _with_module(mod, leaves, amp, pass_, (
            _graphed_closure(mod, leaves, pass_, precision) if graphed else
            _compiled_closure(mod, leaves, pass_, precision, amp, arm) if compiled else
            make_pass(call, leaves, pass_, amp=amp)))

    wrapper_cls = "HybridMambaSSM" if family == "hybrid_mamba" else "SSDMimoSSM"
    preset_name = _EXACT_PRESET[config_name]
    return Cell(
        family="ssm", arm=arm,
        entry_point=(f"orbit_ssm.presets.{preset_name}_kwargs -> "
                    f"orbit_ssm.{wrapper_cls}"),
        shape=dict(B=B, L=L, d_model=d_model, d_inner=d_model * 2,
                   N=d_state, headdim=headdim),
        precision=precision, pass_=pass_, grid=grid,
        group=group_key(grid, work, precision, pass_),
        autocast=(precision == "bf16"),
        tags=dict(tags or {}, mixer=family, mixer_level="full",
                  config=config_tag or config_name,
                  **({"capture": "cuda_graph"} if graphed else {}),
                  **({"capture": "torch_compile"} if compiled else {})),
        build=build, runtime=cell_runtime,
    )


def _exact_arm_param_audit(d_model: int, d_state: int, headdim: int, B: int,
                           L: int, *, d_state_m1: int = 16) -> dict[str, dict]:
    """The param-match gate for the three EXACT-CONFIG arms,
    once, on CPU (grid-build time) -- the same soft-assert pattern
    `_lm_spine_param_audit` uses for the DEFAULT arms' +-5% gate, but
    tighter (+-0.5%): an exact arm is BUILT from the official's own
    preset, so a miss here means the preset and the official have
    silently drifted apart, not that a feature costs parameters.

    `m3exact` PASSES this gate: `presets.
    mamba3_config` reproduces official Mamba-3's MODULE (in_proj/conv/
    norm layout), not just the bare-core scan recipe -- measured
    ~1.0000-1.0001 at three widths, down from the ~1.27-1.39x this gate used
    to record as an expected miss.  The assert is soft (caught, printed,
    tagged), exactly like the +-5% gate, so a REGRESSION here (the
    preset and the official silently drifting apart again) is still
    only RECORDED, never fatal to the grid.

    `d_state` is `m2exact`/`m3exact`'s state size
    (128 at the headline call sites, 64 at the labelled-secondary ones);
    `d_state_m1` is `m1exact`/`official_mamba1`'s OWN state size,
    ALWAYS Mamba-1's real default (16, `presets.mamba1_config`'s own
    default) regardless of `d_state` -- the pre-B9 code passed the SAME
    `d_state` (whatever `N_mhr`/`LM_SPINE["N"]` was, 64) to all three
    variants, which was never Mamba-1's own default and is fixed here.
    """
    B = L = 1  # Parameter counts do not depend on sequence/batch dimensions.
    from orbit_ssm import presets
    from orbit_ssm import HybridMambaSSM, SSDMimoSSM

    ds = {"mamba1": d_state_m1, "mamba2": d_state, "mamba3": d_state}
    official_n: dict[str, int] = {}
    for variant in ("mamba1", "mamba2", "mamba3"):
        official_n[variant] = _cpu_param_count(
            lambda variant=variant: bl.build_mamba_layer(
                variant, d_model, d_state=ds[variant], L=L, B=B,
                headdim=headdim, ngroups=1, expand=2,
                dtype=MODULE_PARAM_DTYPE, device="cpu")[2])

    builders = {
        "m1exact": lambda: HybridMambaSSM(presets.mamba1_config(d_model, d_state=d_state_m1)),
        "m2exact": lambda: SSDMimoSSM(presets.mamba2_config(d_model, d_state=d_state, d_head=headdim)),
        "m3exact": lambda: SSDMimoSSM(presets.mamba3_config(d_model, d_state=d_state, d_head=headdim)),
    }
    tags: dict[str, dict] = {}
    for name, build_fn in builders.items():
        variant = _EXACT_PRESET[name]
        n = _cpu_param_count(build_fn)
        target = official_n[variant]
        ratio = n / target
        ok = abs(ratio - 1.0) <= 0.005
        try:
            assert ok, (f"{name}: {n} params vs official_{variant} target "
                        f"{target} (ratio {ratio:.4f}) is outside the "
                        f"+-0.5% exact-config param-match gate")
        except AssertionError as exc:
            print(f"[param-match] WARNING: {exc}", flush=True)
        tags[name] = dict(config=name, param_match_target_arm=f"official_{variant}",
                          param_match_target_n=target, param_match_n=n,
                          param_match_ratio=round(ratio, 4), param_match_ok=ok,
                          d_state=ds[variant])
    return tags


# ═════════════════════════════════════════════════════════════════════
# Feature arms ON TOP of the exact-config rows -- price the
# complex (rotation) and gated configurations AT THE MIXER LEVEL (the
# kernel-level ledger already prices these in bench_kernels.py's
# `mh_features` grid -- this is the same three combinations one level up,
# on the real production mixers).
#
# Three named combinations, matching `mh_features`'s own
# `feat:rot`/`feat:gates`/`feat:all` naming:
#   "rot"   -- the family's own rotation: a complex block in `rows`
#              covering the whole state.  HM: a per-token angle from
#              `theta_proj`, SHARED across every head of the block,
#              cumulated in-kernel -- "the head-shared dynamic angle".
#              SSD-MIMO: the single rotation law (Mamba-3's per-head
#              phase).  An equivalence check proved the two are
#              the SAME law under `theta_ssd = -Theta_mh`.
#   "gates" -- `use_g_b=True, use_g_c=True` (g_B/g_C) on both families
#              (SSD: both default off at R=O=1, see `mamba2_config`).
#   "all"   -- rotation + gates + biases: HM adds `bias_gran="head"`/
#              `gate_bias` (the decay bias `a_bias` was removed); SSD-MIMO adds `bias_gran="head"`
#              plus the B/C RMSNorms `mamba2_config` turns off (its
#              `b_norm=c_norm=False, bias_gran="none"` is the "no B/C
#              RMSNorm or bias to match M2's mixer" choice, overridden
#              here for this arm only through the preset's `**overrides`).
#
# NOT a param-match gate.  Every one of these arms is DESIGNED to cost
# more parameters than its base row (that is the point of the feature) --
# a +-X% assertion here would fail on every cell, which is not a finding.
# `tags.param_ratio` records the UNCHECKED ratio to the arm's own base
# row instead: the exact-config row for the `m1exact`/`m2exact`-based
# arms, the design-width full-mixer row (`HMM-LM`/`HMM`) for the other
# three.  `tags.feature` names which combination; `tags.base_arm` names
# the row the ratio is against.
# Spelling: the rows / rotation / maps spec,
# The HM gate defaults moved with it (gate_act "one_plus",
# gate_pos "pre" = gate before bias and rotation); every gated HM arm
# below pins the gate these rows always measured -- sigmoid, applied in
# the scan AFTER the rotation (`gate_act="sigmoid", gate_pos="post"`) --
# so the priced model is unchanged.  The HM B/C bias: `bias_gran="head"`
# is the pre-spec `bc_bias=True` (G = H), `"shared"` is `bc_bias_heads=1`.
# BLOCKGATES: a complex block's gate default is False;
# the `all` rows price gates ON the complex rows, so the block spells
# gate=True (a block gate needs a B/C gate): the priced models are unchanged.
_HM_COMPLEX_A = [{"frac": 1.0, "decay": "coord_static", "complex": True}]
_HM_COMPLEX_GATED = [{"frac": 1.0, "decay": "coord_static", "complex": True, "gate": True}]
_HM_GATES_POST = dict(b_gate=True, c_gate=True, bc_gate_act="sigmoid",
                      bc_gate_pos="post")
_HM_FEATURE_OVERRIDES: dict[str, dict] = {
    "rot": dict(row_blocks=_HM_COMPLEX_A),
    "gates": dict(_HM_GATES_POST),
    "all": dict(row_blocks=_HM_COMPLEX_GATED, **_HM_GATES_POST, bc_bias="head",
               bc_gate_bias=True),
    # Head-shared bias: `all` at P=1 (`HMM:m1exact+all`,
    # ssm_lm_spine's H = d_inner/head_dim = 2048) allocates `beta_b`/
    # `beta_c` at (G, n_hm) with G = H -- head-RESOLVED B/C bias,
    # 46 GB / 6.5x at that width.  The `all32` row (a 32-group bias,
    # `bc_bias_heads=32`) is GONE: intermediate bias groups 1 < G < H left
    # the spec (bc_bias is "head" | "shared" | "none"), and a measurement showed
    # it took the same path as `all` anyway.
    # `all1` is the head-
    # SHARED bias (G = 1), which stays on the K16 rotation fold -- +4 %
    # over `+rot+gates`.  Its gate biases are head-shared too
    # (G = 1), which the spec expresses only under bc_gate_pos="pre"
    # ("post" makes them per head), so this row pins the sigmoid gate
    # BEFORE bias and rotation (sigmoid does not factor, so the pre gate
    # bias is not redundant with the norm weight and stays legal).  NOT the
    # pre-spec row exactly: there is no decay bias (removed), and
    # the pre-spec row gated in-scan after the rotation; that row is not
    # expressible any more.
    "all1": dict(row_blocks=_HM_COMPLEX_GATED, b_gate=True, c_gate=True,
                bc_gate_act="sigmoid", bc_gate_pos="pre", bc_bias="shared",
                bc_gate_bias=True),
}
# SSD-MIMO on top of `mamba2_config` (no norms, no bias, one static rate
# per head): "rot" puts the whole state in a complex block under the SAME
# law and turns on the per-head B/C biases and the B/C RMSNorms the
# rotation needs (the pre-spec `rot_mode="m3", bias_gran="head",
# bc_norm=True` is now `bc_bias="head", b_norm=True, c_norm=True`).
_SSD_COMPLEX_STATIC = [{"frac": 1.0, "decay": "head_static", "complex": True}]
_SSD_COMPLEX_STATIC_GATED = [{"frac": 1.0, "decay": "head_static", "complex": True,
                              "gate": True}]    # BLOCKGATES pin (a gate must exist), as _HM_COMPLEX_GATED
_SSD_FEATURE_OVERRIDES: dict[str, dict] = {
    "rot": dict(row_blocks=_SSD_COMPLEX_STATIC, bc_bias="head",
               b_norm=True, c_norm=True),
    "gates": dict(b_gate=True, c_gate=True),
    "all": dict(row_blocks=_SSD_COMPLEX_STATIC_GATED, b_gate=True, c_gate=True,
               bc_bias="head", b_norm=True, c_norm=True),
}
_FEATURE_NAMES = ("rot", "gates", "all")
# HM-only: `all1` re-prices `all`'s bias allocation (see the override
# dict's comment above) and has no SSD-MIMO counterpart (`bc_bias`
# already only allocates ONE per-head bias tensor, not a per-channel
# one -- there is no analogous blow-up to re-price), so it is added only
# to the two HM feature-cell builders below, never to
# `_ssd_exact_feature_cells`, which still iterates `_FEATURE_NAMES`.
_HM_FEATURE_NAMES = _FEATURE_NAMES + ("all1",)


def _feature_tags(base_arm: str, base_n: int, feature: str,
                  build_fn: Callable[[], Any], **extra) -> dict:
    """CPU-only param count for one feature arm, recorded as an
    UNCHECKED ratio to `base_arm`'s own count (`base_n`) -- see the
    module note above for why there is no +-X% gate here."""
    n = _cpu_param_count(build_fn)
    return dict(extra, feature=feature, base_arm=base_arm,
               base_param_n=base_n, param_n=n,
               param_ratio=round(n / max(base_n, 1), 4))


def _hm_exact_feature_cells(g: str, d_model: int, d_state: int, B: int,
                            L: int, prec: str, p: str, w: str,
                            base_n: int) -> list[Cell]:
    """`HMM:m1exact+{rot,gates,all}` -- Mamba-1's exact config (P = 1,
    `orbit_ssm.presets.mamba1_config`) plus one feature combination, in
    the SAME group as `HMM:m1exact`/`official_mamba1` (built through
    `_exact_mixer_cell`, so the D1 precision remap and the fused-route
    construction apply exactly as they do for `HMM:m1exact` itself --
    `HybridMambaSSM` has no `precision=` kwarg at all, so there is
    nothing to remap for this family; see `_exact_mixer_cell`'s own
    docstring)."""
    from orbit_ssm import presets
    from orbit_ssm import HybridMambaSSM

    cells: list[Cell] = []
    for feat in _HM_FEATURE_NAMES:
        over = _HM_FEATURE_OVERRIDES[feat]
        arm = f"HMM:m1exact+{feat}"
        tags = _feature_tags(
            "HMM:m1exact", base_n, feat,
            lambda over=over: HybridMambaSSM(presets.mamba1_config(d_model, d_state=d_state, **over)))
        cells.append(_exact_mixer_cell(
            g, arm, "m1exact", d_model, d_state, B, L, prec, p,
            work=w, tags=tags, **over))
    return cells


def _hm_design_feature_cells(g: str, arm_base: str, d_model: int,
                             n_heads: int, d_state: int, B: int, L: int,
                             prec: str, p: str, w: str,
                             base_n: int,
                             feature_names: tuple[str, ...] = _FEATURE_NAMES
                             ) -> list[Cell]:
    """`{HMM-LM,HMM}+{rot,gates,all}` -- our own design width (P = 64,
    `n_heads` passed explicitly) plus one feature combination, in the
    SAME group/width as `arm_base` (`_full_mixer_cell`, the DEFAULT-arm
    route `HMM-LM`/`HMM` already use).

    `feature_names` defaults to the original three; `ssm_lm_spine`'s
    `HMM-LM` passes `_HM_FEATURE_NAMES` (adds `all1`) explicitly.  (It
    also used to add `all32`, a 32-group bias that needed n_heads % 32 ==
    0 -- the reason `ssm_mqmkvar`'s `HMM`, n_heads=8, never got the
    extra rows; `all32` left the spec with `bc_bias_heads`.)"""
    cells: list[Cell] = []
    for feat in feature_names:
        over = _HM_FEATURE_OVERRIDES[feat]
        arm = f"{arm_base}+{feat}"
        mixer_kwargs = dict(n_heads=n_heads, proj_source="main_branch", **over)
        tags = _feature_tags(
            arm_base, base_n, feat,
            lambda mixer_kwargs=mixer_kwargs: build_full_mixer(
                "hybrid_mamba", d_model, d_state, 1, 1,
                dtype=MODULE_PARAM_DTYPE, mixer_kwargs=dict(mixer_kwargs),
                device="cpu")[2],
            config="default")
        cells.append(_full_mixer_cell(
            g, "hybrid_mamba", arm, d_model, d_state, B, L, prec, p,
            work=w, tags=tags, **mixer_kwargs))
    return cells


def _ssd_exact_feature_cells(g: str, d_model: int, d_state: int, B: int,
                             L: int, prec: str, p: str, w: str,
                             headdim: int, base_n: int) -> list[Cell]:
    """`SM-R1:m2exact+{rot,gates,all}` -- Mamba-2's exact config plus one
    feature combination, in the SAME group as `SM-R1:m2exact`/
    `official_mamba2` (`_exact_mixer_cell`, so D1's `KERNEL_PRECISION`
    remap and the fused-route construction -- `ssd_mimo_scan`'s
    `_require_fused=True` -- apply exactly as they do for `SM-R1:m2exact`
    itself)."""
    from orbit_ssm import presets
    from orbit_ssm import SSDMimoSSM

    cells: list[Cell] = []
    for feat in _FEATURE_NAMES:
        over = _SSD_FEATURE_OVERRIDES[feat]
        arm = f"SM-R1:m2exact+{feat}"
        tags = _feature_tags(
            "SM-R1:m2exact", base_n, feat,
            lambda over=over: SSDMimoSSM(presets.mamba2_config(d_model, d_state=d_state,
                                        d_head=headdim, **over)))
        cells.append(_exact_mixer_cell(
            g, arm, "m2exact", d_model, d_state, B, L, prec, p,
            work=w, headdim=headdim, tags=tags, **over))
    return cells


def _graphed_like_for_like_cells(g, d_model, B, L, prec, p, w, exact_tags):
    """The six `<arm>@graph` rows of a grid (G34), or [] outside bf16.

    Exactly the exact-config like-for-like set -- our three exact-preset
    mixers and the three officials they reproduce -- in the SAME group,
    at the same widths and state sizes as their eager twins, so a group
    carries `X`, `X@graph`, `official_Y` and `official_Y@graph` and every
    one of the four ratios comes out of interleaved rounds.  bf16 only:
    the captured regime exists for the paper's bf16 table, and the
    tf32 column would double the graph pools resident in one group for a
    number nothing quotes.
    """
    if prec != "bf16":
        return []
    cells = [
        _exact_mixer_cell(
            g, "SM-R1:m2exact", "m2exact", d_model, sh.EXACT_D_STATE_M23,
            B, L, prec, p, work=w, headdim=64, tags=exact_tags["m2exact"],
            graphed=True),
        _exact_mixer_cell(
            g, "SM-R1:m3exact", "m3exact", d_model, sh.EXACT_D_STATE_M23,
            B, L, prec, p, work=w, headdim=64, tags=exact_tags["m3exact"],
            graphed=True),
        _exact_mixer_cell(
            g, "HMM:m1exact", "m1exact", d_model, sh.EXACT_D_STATE_M1,
            B, L, prec, p, work=w, tags=exact_tags["m1exact"], graphed=True),
    ]
    for variant, ds in (("mamba1", sh.EXACT_D_STATE_M1),
                        ("mamba2", sh.EXACT_D_STATE_M23),
                        ("mamba3", sh.EXACT_D_STATE_M23)):
        cells.append(_official_cell(
            g, variant, d_model, ds, B, L, prec, p, work=w, headdim=64,
            tags=dict(config="official"), graphed=True))
    return cells


# ═════════════════════════════════════════════════════════════════════
# grids
# ═════════════════════════════════════════════════════════════════════
def grid_smoke(graphed: bool = False) -> list[Cell]:
    g = "smoke"
    s = sh.SMOKE_SSM
    d, n, B, L = s["d_model"], s["d_state"], s["B"], s["L"]
    w = f"B{B}L{L}d{d}N{n}"
    cells: list[Cell] = []
    for prec in ("tf32", "bf16"):
        for p in ("fwd", "fwd_bwd"):
            for fam in FAMILIES:
                over: dict[str, Any] = {}
                if fam == "ssd_mimo":
                    over = dict(d_ssm_in=2, d_ssm_out=2, n_groups=8)
                elif fam in CASE_SMOKE:
                    over = dict(CASE_SMOKE[fam])
                elif fam == "hybrid_mamba":
                    over = dict(n_heads=8)
                cells.append(_ssm_cell(g, fam, fam, d, n, B, L, prec, p,
                                       work=w, **over))
            for variant, hd in (("mamba1", 64), ("mamba2", 32), ("mamba3", 32)):
                cells.append(_official_cell(g, variant, d, n, B, L, prec, p,
                                            work=w, headdim=hd))
    return cells


def grid_ssm_mqmkvar(graphed: bool = False) -> list[Cell]:
    """The MQMKVAR campaign widths, verbatim from the signed spec.

    `n_groups` is passed EXPLICITLY on every SSD-MIMO arm — a G below the
    fused kernels' 16-lane tile is zero-padded host-side at a cost of
    16/G, and relying on the default is what made R>1
    arms crawl past the wall-time limit in a pilot run.

    Same two mixer LEVELS as `ssm_lm_spine` (finding B2): `core:S1`..`core:S5` and `core:HM`
    are the BARE `SSDMimoSSM` / `HybridMambaSSM` cores this grid always
    measured under the plain `S1`../`HM` names; `SM1`..`SM5` and `HMM`
    are the FULL production mixers at the SAME width each spec/officials
    arm already uses (`SM{n}` at `width_convention=spec`'s own d_model,
    `HMM` at `width_convention=officials`).  The rows of a case family
    that owns its full projection stack are unchanged (always full-mixer rows).
    Unlike `ssm_lm_spine` there is no +-5% param-match assertion on the
    DEFAULT arms here — the standing gate that instruction names is
    scoped to "the spine cell"; at this narrower width even `SM1` (R=O=1)
    lands at 1.22x `official_mamba2` (measured), so a hard
    gate at this shape would need a different tolerance, a design decision
    this module does not make on its own.  Every arm's `tags.config` is
    "default" -- the SAME mixer classes, but at THIS grid's own width/
    feature choices, not the officials'.

    Three more
    rows, `SM-R1:m2exact` / `SM-R1:m3exact` / `HMM:m1exact`
    (`tags.config` = "m2exact"/"m3exact"/"m1exact"), are the actual
    like-for-like headline -- our mixer classes built at the OFFICIALS'
    exact configuration (`orbit_ssm.presets`) at the SAME (d_model,
    d_state, headdim) the `official_mamba{1,2,3}` rows below already use
    at THIS width, in the SAME group.  `_exact_arm_param_audit` gates
    each to +-0.5% of its official's parameter count, CPU-only, at
    grid-build time (a miss recorded, not fatal -- `m3exact` is expected
    to miss, see that function's docstring).

    Six more rows price the complex/gated FEATURE
    combinations at the mixer level, ON TOP of the like-for-like rows
    above -- `HMM:m1exact+{rot,gates,all}` (Mamba-1's exact config plus
    one feature), `HMM+{rot,gates,all}` (the SAME three, on this grid's
    own `HMM` design width), and `SM-R1:m2exact+{rot,gates,all}`
    (Mamba-2's exact config plus one feature).  None of these is param-
    matched -- see `_feature_tags`'s docstring -- `tags.param_ratio`
    records the unchecked cost against each arm's own base row instead.

    Head-shared bias: `HMM:m1exact+all1`
    re-prices `HMM:m1exact+all` with a head-SHARED bias instead of the
    per-head one (`_HM_FEATURE_OVERRIDES`'s own comment) -- kept NEXT TO
    the original `+all` row, not replacing it.  (The 32-group `+all32`
    row left with `bc_bias_heads`.)  No `HMM+all1` design-
    width counterpart here; `ssm_lm_spine`'s `HMM-LM` is where that
    design-width row exists.

    State-size split: every exact-config row above ran at N=64
    (`n_mhr`, this grid's OWN mhr-kernel state size), never at official
    Mamba-2/Mamba-3's own `d_state=128` default.  `SM-R1:m2exact`/
    `SM-R1:m3exact` and their `official_mamba2`/`official_mamba3`
    counterparts now run at N=128 (`sh.EXACT_D_STATE_M23`) as the
    HEADLINE like-for-like row; the ORIGINAL N=64 cell survives as an
    `@N64`-suffixed secondary row (`tags.config` = `*_n64`), excluded
    from the paper set.  `HMM:m1exact`/`official_mamba1` are fixed to
    run at Mamba-1's OWN default (`sh.EXACT_D_STATE_M1` = 16) instead of
    `n_mhr` -- they were never at 64 to begin with, and 64 was never
    Mamba-1's own default either.  The default S1-S5/HM/HMM
    arms above are UNCHANGED (still `n_mhr` = 64, the spec's own state
    size, not an "officials' default" question).
    """
    g = "ssm_mqmkvar"
    B, L = sh.MQMKVAR_SHAPE["B"], sh.MQMKVAR_SHAPE["L"]
    n_mhr = sh.MQMKVAR_SHAPE["N_mhr"]
    # ONE group per (precision, pass): the campaign arms and the three
    # officials are the comparison, and every one of them is a d256-class
    # mixer over the same (B, L) token budget.  The arms differ in width,
    # R/O, N and family — which is exactly why none of them may be
    # measured in a group of its own.
    d_model = sh.MQMKVAR_SHAPE["d_model"]      # ONE source for the width
    w = f"B{B}L{L}d{d_model}"
    other = {a["id"]: a for a in sh.MQMKVAR_OTHER_ARMS}
    hm = other["HM"]
    # State-size split: the exact-config headline audit runs at the
    # officials' own `d_state` default (128, `m2exact`/`m3exact`) and
    # Mamba-1's own default (16, `m1exact` -- `d_state_m1`), NOT at
    # `n_mhr` (this grid's own mhr-kernel state size, which the pre-B9
    # code used for all three, matching none of the officials).  A
    # second audit at the ORIGINAL n_mhr=64 secondary state size feeds
    # the `SM-R1:m2exact@N64`/`SM-R1:m3exact@N64` rows below.
    exact_tags = _exact_arm_param_audit(
        d_model, sh.EXACT_D_STATE_M23, 64, B, L,
        d_state_m1=sh.EXACT_D_STATE_M1)
    exact_tags_n64 = _exact_arm_param_audit(
        d_model, sh.EXACT_D_STATE_M23_N64, 64, B, L,
        d_state_m1=sh.EXACT_D_STATE_M1)
    # Feature arms: `HMM`'s own param count (this grid runs no +-5% audit on the
    # DEFAULT arms, see the docstring above, so there is no existing tag
    # to reuse the way `ssm_lm_spine` reuses `param_tags["HMM-LM"]`) --
    # the base row the `HMM+{rot,gates,all}` feature arms below price
    # their (unchecked) `param_ratio` against.
    hmm_n_heads = hm["d_inner"] // 64
    hmm_base_n = _cpu_param_count(lambda: build_full_mixer(
        "hybrid_mamba", hm["d_model"], n_mhr, 1, 1,
        dtype=MODULE_PARAM_DTYPE, device="cpu",
        mixer_kwargs=dict(n_heads=hmm_n_heads, proj_source="main_branch"))[2])

    def wtags(arm: dict, **extra) -> dict:
        """Say in the record WHICH width convention a row was built on."""
        return dict(extra, d_model=arm["d_model"],
                    width_convention=arm["width_convention"])

    cells: list[Cell] = []
    for prec in ("ieee", "tf32", "bf16"):
        for p in ("fwd", "fwd_bwd"):
            for arm in sh.MQMKVAR_S_ARMS:
                # `arm["width"]` is the spec's d_model; the mixer runs at
                # d_inner = 2*width (expand=2), which is also what makes
                # d_inner a multiple of P = G*R = 16.  See shapes.py.
                cells.append(_ssm_cell(
                    g, "ssd_mimo", f"core:{arm['id']}", arm["d_inner"],
                    n_mhr, B, L, prec, p, work=w,
                    tags=dict(spec_arm=arm["id"], d_model=arm["width"],
                             width_convention="spec", config="default"),
                    d_ssm_in=arm["d_ssm_in"], d_ssm_out=arm["d_ssm_out"], n_groups=arm["n_groups"],
                    row_blocks=arm["row_blocks"],
                    r_mix_dyn=(arm["d_ssm_in"] > 1)))
                # full mixer at the SAME (width, R, O, ngroups, rows) the
                # bare core above just ran -- the only variable
                # is in_proj/conv1d/gate/norm/out_proj.
                cells.append(_full_mixer_cell(
                    g, "ssd_mimo", f"SM{arm['id'][1:]}", arm["width"], n_mhr,
                    B, L, prec, p, work=w,
                    tags=dict(spec_arm=arm["id"], d_model=arm["width"],
                             width_convention="spec", config="default"),
                    d_ssm_in=arm["d_ssm_in"], d_ssm_out=arm["d_ssm_out"], n_groups=arm["n_groups"],
                    row_blocks=arm["row_blocks"],
                    r_mix_dyn=(arm["d_ssm_in"] > 1), proj_source="main_branch"))
            # The case families and HM at the OFFICIALS' width convention: the
            # group compares them against official_mamba1/2/3, which are
            # built from d_model=256 through expand=2 (d_inner=512), and
            # these used to be built at d_inner=256 — half the
            # inner width of the arms they are ratioed against.  The spec
            # gives none of them its own drive-wiring width, so the
            # convention is recorded in the row (see shapes.py).
            for hook in GRID_HOOKS["ssm_mqmkvar:officials_width"]:
                cells.extend(hook(dict(g=g, prec=prec, p=p, w=w, B=B, L=L, n_state=n_mhr,
                                       ssm_cell=_ssm_cell, wtags=wtags, arms=other)))
            # head_dim=16 => P=16, the power-of-two head width the MH CUDA
            # backward requires; passed EXPLICITLY, never left to the
            # family default.
            cells.append(_ssm_cell(g, hm["family"], "core:HM", hm["d_inner"],
                                   n_mhr, B, L, prec, p, work=w,
                                   tags=wtags(hm, config="default"),
                                   n_heads=hm["d_inner"] // hm["d_head"]))
            # full mixer at the SAME officials' width; head_dim raised to
            # 64 (still a power of two) to match the officials' own
            # headdim=64 convention, as `HMM-LM` does at the spine cell.
            cells.append(_full_mixer_cell(
                g, "hybrid_mamba", "HMM", hm["d_model"], n_mhr, B, L,
                prec, p, work=w, tags=wtags(hm, d_head=64, config="default"),
                n_heads=hm["d_inner"] // 64, proj_source="main_branch"))
            # ---- exact-config arms, same group, SAME width the
            # officials below use (d_model, headdim=64).
            # `SM-R1:m2exact`/`SM-R1:m3exact` now run at
            # N=128 (`EXACT_D_STATE_M23`, official Mamba-2/3's own
            # `d_state` default) as the HEADLINE like-for-like row; the
            # ORIGINAL N=64 cell survives as a labelled secondary row
            # (`@N64` arm suffix, `config` tagged `*_n64` --
            # `emit_tables.emit_like_for_like` keys off `tags.config`,
            # not the arm name, so the two never collide in one group).
            # `HMM:m1exact` does NOT move: Mamba-1's own default is
            # d_state=16 (`EXACT_D_STATE_M1`), not `n_mhr` -- the pre-B9
            # code ran it at n_mhr (64), which was never Mamba-1's own
            # default (fixed here, see `_exact_arm_param_audit`'s
            # `d_state_m1` param).
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m2exact", "m2exact", d_model, sh.EXACT_D_STATE_M23,
                B, L, prec, p, work=w, headdim=64, tags=exact_tags["m2exact"]))
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m2exact@N64", "m2exact", d_model,
                sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=64,
                config_tag="m2exact_n64",
                tags=dict(exact_tags_n64["m2exact"], config="m2exact_n64")))
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m3exact", "m3exact", d_model, sh.EXACT_D_STATE_M23,
                B, L, prec, p, work=w, headdim=64, tags=exact_tags["m3exact"]))
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m3exact@N64", "m3exact", d_model,
                sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=64,
                config_tag="m3exact_n64",
                tags=dict(exact_tags_n64["m3exact"], config="m3exact_n64")))
            cells.append(_exact_mixer_cell(
                g, "HMM:m1exact", "m1exact", d_model, sh.EXACT_D_STATE_M1,
                B, L, prec, p, work=w, tags=exact_tags["m1exact"]))
            # ---- Feature arms on top of the exact-config / design-
            # width rows above (same group, same width).  State-size split: the SSD
            # feature arms ride on the N=128 HEADLINE exact row.
            cells.extend(_hm_exact_feature_cells(
                g, d_model, sh.EXACT_D_STATE_M1, B, L, prec, p, w,
                exact_tags["m1exact"]["param_match_n"]))
            cells.extend(_hm_design_feature_cells(
                g, "HMM", hm["d_model"], hmm_n_heads, n_mhr, B, L, prec, p,
                w, hmm_base_n))
            cells.extend(_ssd_exact_feature_cells(
                g, d_model, sh.EXACT_D_STATE_M23, B, L, prec, p, w, 64,
                exact_tags["m2exact"]["param_match_n"]))
            # officials: mamba1 at ITS OWN default (16, unchanged, never
            # moves); mamba2/mamba3 at the N=128 headline PLUS a
            # secondary `@N64` cell  -- the same group, so
            # `SM-R1:m2exact`'s literal-name ref lookup in
            # `emit_like_for_like` (`_I6_EXACT_TO_OFFICIAL`) still finds
            # an unsuffixed `official_mamba2` built at the SAME N=128.
            for variant, hd, ds in (
                    ("mamba1", 64, sh.EXACT_D_STATE_M1),
                    ("mamba2", 64, sh.EXACT_D_STATE_M23),
                    ("mamba3", 64, sh.EXACT_D_STATE_M23)):
                cells.append(_official_cell(g, variant, d_model, ds, B,
                                            L, prec, p, work=w, headdim=hd,
                                            tags=dict(config="official")))
            for variant, hd in (("mamba2", 64), ("mamba3", 64)):
                cells.append(_official_cell(
                    g, variant, d_model, sh.EXACT_D_STATE_M23_N64, B, L,
                    prec, p, work=w, headdim=hd,
                    arm=f"official_{variant}@N64",
                    tags=dict(config="official_n64")))
            # G34 (`--graphed`): the captured twin of each like-for-like
            # row, same group, bf16 only.
            if graphed:
                cells.extend(_graphed_like_for_like_cells(
                    g, d_model, B, L, prec, p, w, exact_tags))
    return cells


def _lm_spine_param_audit(D: int, N: int, B: int, L: int) -> dict[str, dict]:
    """The full-mixer param-match audit, once, on CPU (grid-build time).

    Standing gate for a NEW full-mixer arm at the spine cell: within +-5%
    of `official_mamba2`'s parameter count (`official_mamba2` because it
    is the middle of the three officials' 6.47-6.96M spread here, not the
    smallest -- that "at most, within 1% below" rule is the separate
    whole-LM-pretraining standing rule and does not apply to a kernel/
    module speed cell).  `SM-R1` and `HMM-LM` (head_dim raised 16 -> 64 to
    close the gap) land inside it; `SM-R2O2` / `SM-R4O4` do NOT and are
    reported rather than forced -- swept `ngroups` in {8..1024} and
    `d_state` in {64,128} for R2O2 first: the
    MIMO read/write mixing matrices (`r_mix_proj` is new outright;
    `gB_proj`/`gC_proj`, present only at R,O > 1, double the
    `B_proj`/`C_proj` cost) cost real parameters that no head-width knob
    removes at a FIXED d_model matching the officials'.  The assert is
    soft (caught, printed, tagged) rather than fatal: a param-match miss
    on one arm must not cost the whole grid's measurement, the same
    recorded-skip-over-crash policy `harness.py` uses everywhere else.
    """
    B = L = 1  # Parameter counts do not depend on sequence/batch dimensions.
    d_model_off = D // 2
    m2_target = _cpu_param_count(
        lambda: bl.build_mamba_layer(
            "mamba2", d_model_off, d_state=N, L=L, B=B, headdim=64,
            ngroups=1, expand=2, dtype=MODULE_PARAM_DTYPE, device="cpu")[2])

    specs = {
        "SM-R1": dict(family="ssd_mimo", d_model=d_model_off, d_state=N,
                     mixer_kwargs=dict(d_ssm_in=1, d_ssm_out=1, n_groups=64,
                                       proj_source="main_branch")),
        "SM-R2O2": dict(family="ssd_mimo", d_model=d_model_off, d_state=128,
                       mixer_kwargs=dict(d_ssm_in=2, d_ssm_out=2, n_groups=64,
                                         proj_source="main_branch",
                                         r_mix_dyn=True)),
        "SM-R4O4": dict(family="ssd_mimo", d_model=d_model_off, d_state=128,
                       mixer_kwargs=dict(d_ssm_in=4, d_ssm_out=4, n_groups=128,
                                         proj_source="main_branch",
                                         r_mix_dyn=True)),
        # head_dim raised 64 (was core:HM-LM's 16) so the full mixer lands
        # inside the +-5% gate -- both are powers of two, so the MH CUDA
        # backward's own constraint (ssm_hybrid.py:305-311) is unaffected.
        "HMM-LM": dict(family="hybrid_mamba", d_model=d_model_off, d_state=N,
                      mixer_kwargs=dict(n_heads=D // 64,
                                        proj_source="main_branch")),
    }
    tags: dict[str, dict] = {}
    for name, spec in specs.items():
        n = _cpu_param_count(lambda spec=spec: build_full_mixer(
            spec["family"], spec["d_model"], spec["d_state"], B, L,
            dtype=MODULE_PARAM_DTYPE, mixer_kwargs=spec["mixer_kwargs"],
            device="cpu")[2])
        ratio = n / m2_target
        ok = abs(ratio - 1.0) <= 0.05
        try:
            assert ok, (f"{name}: {n} params vs official_mamba2 target "
                        f"{m2_target} (ratio {ratio:.3f}) is outside the "
                        f"spine cell's +-5% full-mixer param-match gate")
        except AssertionError as exc:
            print(f"[param-match] WARNING: {exc}", flush=True)
        tags[name] = dict(param_match_target_arm="official_mamba2",
                          param_match_target_n=m2_target,
                          param_match_n=n,
                          param_match_ratio=round(ratio, 4),
                          param_match_ok=ok, config="default")
    return tags


def grid_ssm_lm_spine(graphed: bool = False) -> list[Cell]:
    """B8 D2048 L2048 N64 — the contract's second shape.

    A gate stated as a ratio of two rows of this grid is measured in the SAME group by construction:
    it may not be assembled out of two numbers taken minutes apart on a node that drifts 1.3-2.3x mid-sweep.

    Two mixer LEVELS are visible here (finding B2): `core:S-R1` / `core:S-R2O2` /
    `core:S-R4O4` / `core:HM-LM` are the BARE `SSDMimoSSM` /
    `HybridMambaSSM` cores (no in_proj/conv1d/gate/norm/out_proj) that
    this grid always measured under the plain names before this fix;
    `SM-R1` / `SM-R2O2` / `SM-R4O4` / `HMM-LM` are the FULL production
    mixers (`orbit_ssm.SSDMimoSSM` / `HybridMambaSSM`) at
    the officials' own `d_model`/`expand` surface -- the like-for-like
    counterpart the officials were always ratioed against without one.
    The rows of a case family that owns its full projection stack are unchanged (always full-mixer
    rows).  Every one of the arms
    above carries `tags.config = "default"`: same mixer classes, but at
    the OFFICIALS' width run with THIS repository's own feature choices
    (norms, a_dyn, biases, siblings), not the officials' own config.

    `SM-R1:
    m2exact` / `SM-R1:m3exact` / `HMM:m1exact` are the actual
    like-for-like headline -- our mixer classes built at the OFFICIALS'
    EXACT configuration (`orbit_ssm.presets.mamba{1,2,3}_kwargs`) at
    this same (d_model_off, N, headdim=64), in the SAME group as the
    `official_mamba{1,2,3}` rows they are compared against.  The DEFAULT
    rows above are the separate, priced "what our features cost" story;
    `_exact_arm_param_audit` gates each exact row to +-0.5% of its
    official's parameter count, CPU-only, at grid-build time (a miss
    recorded, not fatal -- as of E3, all three of
    `m1exact`/`m2exact`/`m3exact` pass it; see that function's docstring).

    Six more rows price the complex/gated FEATURE
    combinations at the mixer level, ON TOP of the like-for-like rows
    above -- `HMM:m1exact+{rot,gates,all}` (Mamba-1's exact config plus
    one feature), `HMM-LM+{rot,gates,all}` (the SAME three, on this
    grid's own `HMM-LM` design width), and `SM-R1:m2exact+{rot,gates,
    all}` (Mamba-2's exact config plus one feature).  None of these is
    param-matched -- see `_feature_tags`'s docstring -- `tags.param_ratio`
    records the unchecked cost against each arm's own base row instead.

    Head-shared bias: `HMM:m1exact+all` at
    this spine's width is H = d_inner/head_dim = 2048 per-channel heads,
    and `bc_bias=True` defaults its bias groups to G = H -- head-resolved
    B/C bias, 46 GB / 6.5x.  `HMM:m1exact+all1` and `HMM-LM+all1`
    re-price the SAME combination with a head-shared bias
    (`_HM_FEATURE_OVERRIDES`'s own comment), NEXT TO the original `+all`
    rows, not replacing them -- which to quote is a separate decision.  (The
    32-group `+all32` rows left with `bc_bias_heads`.)

    State-size split: same N=128 headline / N=64 secondary /
    Mamba-1-default fix as `grid_ssm_mqmkvar`'s own state-size note -- see
    that docstring.  One extra wrinkle here: a case-family row stays at
    N (`sh.LM_SPINE["N"]` = 64, unaffected), so a ratio against
    official_mamba3 reads off `official_mamba3@N64` (the
    secondary row), NOT the new N=128 `official_mamba3` headline --
    the two are no longer built at the same state size.
    """
    g = "ssm_lm_spine"
    B, D, L, N = (sh.LM_SPINE["B"], sh.LM_SPINE["D"], sh.LM_SPINE["L"],
                  sh.LM_SPINE["N"])
    w = f"B{B}L{L}d{D}"
    d_model_off = D // 2
    param_tags = _lm_spine_param_audit(D, N, B, L)
    # State-size split: same headline/secondary state-size split as
    # `grid_ssm_mqmkvar` -- see that grid's own comment.  NOTE for a
    # reader computing a case-family ratio against official_mamba3 by
    # hand: the case-family rows are still built at N=64 (`sh.LM_SPINE["N"]`,
    # unaffected by this pass), so that specific ratio reads off
    # `official_mamba3@N64` (the secondary row), NOT the new N=128
    # headline `official_mamba3` -- the two are no longer the same
    # state size.
    exact_tags = _exact_arm_param_audit(
        d_model_off, sh.EXACT_D_STATE_M23, 64, B, L,
        d_state_m1=sh.EXACT_D_STATE_M1)
    exact_tags_n64 = _exact_arm_param_audit(
        d_model_off, sh.EXACT_D_STATE_M23_N64, 64, B, L,
        d_state_m1=sh.EXACT_D_STATE_M1)
    cells: list[Cell] = []
    for prec in ("ieee", "tf32", "bf16"):
        for p in ("fwd", "fwd_bwd"):
            for hook in GRID_HOOKS["ssm_lm_spine:first"]:
                cells.extend(hook(dict(g=g, prec=prec, p=p, w=w, B=B, L=L, D=D, n_state=N, ssm_cell=_ssm_cell)))
            # ---- bare cores (unchanged numbers, explicit `core:` names) ----
            cells.append(_ssm_cell(g, "ssd_mimo", "core:S-R1", D, N, B, L,
                                   prec, p, work=w, d_ssm_in=1, d_ssm_out=1, n_groups=64,
                                   tags=dict(config="default")))
            cells.append(_ssm_cell(g, "ssd_mimo", "core:S-R2O2", D, 128, B, L,
                                   prec, p, work=w, d_ssm_in=2, d_ssm_out=2, n_groups=64,
                                   r_mix_dyn=True, tags=dict(config="default")))
            cells.append(_ssm_cell(g, "ssd_mimo", "core:S-R4O4", D, 128, B, L,
                                   prec, p, work=w, d_ssm_in=4, d_ssm_out=4, n_groups=128,
                                   r_mix_dyn=True,
                                   tags=dict(gate="r4o4_18ms",
                                            config="default")))
            cells.append(_ssm_cell(g, "hybrid_mamba", "core:HM-LM", D, N, B, L,
                                   prec, p, work=w, n_heads=D // 16,
                                   tags=dict(config="default")))
            # ---- full mixers: officials' own (d_model, expand) surface ----
            cells.append(_full_mixer_cell(
                g, "ssd_mimo", "SM-R1", d_model_off, N, B, L, prec, p,
                work=w, tags=param_tags["SM-R1"],
                d_ssm_in=1, d_ssm_out=1, n_groups=64, proj_source="main_branch"))
            # PC.3: the same LM-spine mixer and kernel under the two mixer
            # graph shapes, plus modular's prologue-memory option.  These
            # explicit arms keep mode comparisons in one interleaved group.
            if prec == "bf16":
                pc3 = (
                    ("SM-R1:fused", "fused", False),
                    ("SM-R1:modular", "modular", False),
                    ("SM-R1:modular+checkpoint", "modular", True),
                )
                for arm, mode, checkpoint in pc3:
                    tags = dict(config="pc3_m3", mixer_mode=mode,
                                modular_checkpoint=checkpoint,
                                comparison="pc3_modular_mixer")
                    # m3 selection packs B/C/dt beside z in the prologue
                    # slab, so the checkpoint arm recomputes the complete
                    # boundary PC.3 intends to price.
                    over = dict(d_ssm_in=1, d_ssm_out=1, n_groups=64, proj_source="layer_input",
                                mode=mode,
                                modular_checkpoint=checkpoint)
                    cells.append(_full_mixer_cell(
                        g, "ssd_mimo", arm, d_model_off, N, B, L, prec, p,
                        work=w, tags=tags, **over))
                    if graphed:
                        cells.append(_full_mixer_cell(
                            g, "ssd_mimo", arm + GRAPH_ARM_SUFFIX,
                            d_model_off, N, B, L, prec, p, work=w,
                            tags=tags, graphed=True, **over))
            cells.append(_full_mixer_cell(
                g, "ssd_mimo", "SM-R2O2", d_model_off, 128, B, L, prec, p,
                work=w, tags=param_tags["SM-R2O2"],
                d_ssm_in=2, d_ssm_out=2, n_groups=64, r_mix_dyn=True, proj_source="main_branch"))
            cells.append(_full_mixer_cell(
                g, "ssd_mimo", "SM-R4O4", d_model_off, 128, B, L, prec, p,
                work=w, tags=dict(param_tags["SM-R4O4"], gate="r4o4_18ms"),
                d_ssm_in=4, d_ssm_out=4, n_groups=128, r_mix_dyn=True, proj_source="main_branch"))
            cells.append(_full_mixer_cell(
                g, "hybrid_mamba", "HMM-LM", d_model_off, N, B, L, prec, p,
                work=w, tags=param_tags["HMM-LM"],
                n_heads=D // 64, proj_source="main_branch"))
            # ---- already full mixers (own projection stack) ----
            # ---- exact-config arms, SAME group, SAME width as the
            # officials below (d_model_off, headdim=64).
            # `SM-R1:m2exact`/`SM-R1:m3exact` now run at
            # N=128 (`EXACT_D_STATE_M23`) as the HEADLINE row; the
            # ORIGINAL N=64 cell survives as a labelled secondary row
            # (`@N64` arm suffix, `config` tagged `*_n64`).
            # `HMM:m1exact` does NOT move (Mamba-1's own default,
            # `EXACT_D_STATE_M1` = 16, not `N`).
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m2exact", "m2exact", d_model_off,
                sh.EXACT_D_STATE_M23, B, L, prec, p, work=w, headdim=64,
                tags=exact_tags["m2exact"]))
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m2exact@N64", "m2exact", d_model_off,
                sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=64,
                config_tag="m2exact_n64",
                tags=dict(exact_tags_n64["m2exact"], config="m2exact_n64")))
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m3exact", "m3exact", d_model_off,
                sh.EXACT_D_STATE_M23, B, L, prec, p, work=w, headdim=64,
                tags=exact_tags["m3exact"]))
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m3exact@N64", "m3exact", d_model_off,
                sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=64,
                config_tag="m3exact_n64",
                tags=dict(exact_tags_n64["m3exact"], config="m3exact_n64")))
            cells.append(_exact_mixer_cell(
                g, "HMM:m1exact", "m1exact", d_model_off, sh.EXACT_D_STATE_M1,
                B, L, prec, p, work=w, tags=exact_tags["m1exact"]))
            # ---- Feature arms on top of the exact-config / design-
            # width rows above (same group, same width).  State-size split: the SSD
            # feature arms ride on the N=128 HEADLINE exact row.
            cells.extend(_hm_exact_feature_cells(
                g, d_model_off, sh.EXACT_D_STATE_M1, B, L, prec, p, w,
                exact_tags["m1exact"]["param_match_n"]))
            cells.extend(_hm_design_feature_cells(
                g, "HMM-LM", d_model_off, D // 64, N, B, L, prec, p, w,
                param_tags["HMM-LM"]["param_match_n"],
                feature_names=_HM_FEATURE_NAMES))
            cells.extend(_ssd_exact_feature_cells(
                g, d_model_off, sh.EXACT_D_STATE_M23, B, L, prec, p, w, 64,
                exact_tags["m2exact"]["param_match_n"]))
            # officials: mamba1 at ITS OWN default (16, unchanged);
            # mamba2/mamba3 at the N=128 headline PLUS a secondary
            # `@N64` cell  -- see the docstring note above re: a
            # ratio at N=64 reading off the `@N64` one, not this headline.
            for variant, hd, ds in (
                    ("mamba1", 64, sh.EXACT_D_STATE_M1),
                    ("mamba2", 64, sh.EXACT_D_STATE_M23),
                    ("mamba3", 64, sh.EXACT_D_STATE_M23)):
                cells.append(_official_cell(g, variant, D // 2, ds, B, L,
                                            prec, p, work=w, headdim=hd,
                                            tags=dict(gate_ref=True,
                                                     config="official")))
            for variant, hd in (("mamba2", 64), ("mamba3", 64)):
                cells.append(_official_cell(
                    g, variant, D // 2, sh.EXACT_D_STATE_M23_N64, B, L,
                    prec, p, work=w, headdim=hd,
                    arm=f"official_{variant}@N64",
                    tags=dict(gate_ref=True, config="official_n64")))
            # G34 (`--graphed`): the captured twin of each like-for-like
            # row, same group, bf16 only.
            if graphed:
                cells.extend(_graphed_like_for_like_cells(
                    g, d_model_off, B, L, prec, p, w, exact_tags))
    return cells


def _graphed_like_for_like_cells_shared(g, d_model, B, L, w, tag):
    cells = []
    for p in ("fwd", "fwd_bwd"):
        for arm, cfg in (("SM-R1:m2exact", "m2exact"), ("SM-R1:m3exact", "m3exact")):
            cells.append(_exact_mixer_cell(
                g, arm, cfg, d_model, sh.EXACT_D_STATE_M23, B, L, "bf16", p, work=w,
                headdim=64, tags=dict(config=cfg, shape=tag), graphed=True))
        for variant in ("mamba2", "mamba3"):
            cells.append(_official_cell(
                g, variant, d_model, sh.EXACT_D_STATE_M23, B, L, "bf16", p, work=w,
                headdim=64, tags=dict(config="official", shape=tag), graphed=True))
    return cells


def _compiled_like_for_like_cells_shared(g, d_model, B, L, w, tag):
    """The `torch.compile` (default mode) twin of each like-for-like row, ours AND
    the official modules; compile time is excluded from the timing and recorded
    separately (`compile_seconds`), a compile failure is a named cell failure."""
    cells = []
    for p in ("fwd", "fwd_bwd"):
        for arm, cfg in (("SM-R1:m2exact", "m2exact"), ("SM-R1:m3exact", "m3exact")):
            cells.append(_exact_mixer_cell(
                g, arm, cfg, d_model, sh.EXACT_D_STATE_M23, B, L, "bf16", p, work=w,
                headdim=64, tags=dict(config=cfg, shape=tag), compiled=True))
        for variant in ("mamba2", "mamba3"):
            cells.append(_official_cell(
                g, variant, d_model, sh.EXACT_D_STATE_M23, B, L, "bf16", p, work=w,
                headdim=64, tags=dict(config="official", shape=tag), compiled=True))
    return cells


def grid_ssm_shared_set(graphed: bool = False) -> list[Cell]:
    """The ONE fixed shape set (`src/orbit_ssm/kernels/shape_sets.py`: 3 size
    classes x {balanced, long, wide, batch} at equal work per class, plus the
    LM application cells) at the like-for-like pair: our exact-preset mixers
    (m2exact, m3exact) against the official Mamba-2 / Mamba-3 modules, bf16,
    N = 128.  The calibrated kernel defaults are measured on the same cells,
    so the paper tables and the calibration rest on one set."""
    from orbit_ssm.kernels import shape_sets as ss
    g = "ssm_shared_set"
    cells: list[Cell] = []
    for tag, (B, d_inner, L) in ss.SET.items():
        d_model = d_inner // 2                   # expand = 2
        w = f"{tag}:B{B}L{L}d{d_model}N{sh.EXACT_D_STATE_M23}"
        for p in ("fwd", "fwd_bwd"):
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m2exact", "m2exact", d_model, sh.EXACT_D_STATE_M23,
                B, L, "bf16", p, work=w, headdim=64,
                tags=dict(config="m2exact", shape=tag)))
            cells.append(_exact_mixer_cell(
                g, "SM-R1:m3exact", "m3exact", d_model, sh.EXACT_D_STATE_M23,
                B, L, "bf16", p, work=w, headdim=64,
                tags=dict(config="m3exact", shape=tag)))
            for variant in ("mamba2", "mamba3"):
                cells.append(_official_cell(
                    g, variant, d_model, sh.EXACT_D_STATE_M23, B, L, "bf16", p,
                    work=w, headdim=64, tags=dict(config="official", shape=tag)))
        # the CUDA-graph captured twin of each like-for-like row (`<arm>@graph`, same
        # group, same budget): the paper reports the mixer numbers both ways, so the shared set always carries eager AND graphed rows
        cells += _graphed_like_for_like_cells_shared(g, d_model, B, L, w, tag)
        cells += _compiled_like_for_like_cells_shared(g, d_model, B, L, w, tag)
    return cells


def _shared_part(prefix: str) -> Callable[..., list[Cell]]:
    def fn(graphed: bool = False) -> list[Cell]:
        return [c for c in grid_ssm_shared_set(graphed)
                if c.tags["shape"].startswith(prefix)]
    return fn


_GRID_FNS: dict[str, Callable[[], list[Cell]]] = {
    "ssm_shared_set": grid_ssm_shared_set,
    "ssm_shared_small": _shared_part("small_"), "ssm_shared_medium": _shared_part("medium_"),
    "ssm_shared_large": _shared_part("large_"), "ssm_shared_lm": _shared_part("lm_"),
    "smoke": grid_smoke,
    "ssm_mqmkvar": grid_ssm_mqmkvar,
    "ssm_lm_spine": grid_ssm_lm_spine,
}


def build_grid(name: str, args: argparse.Namespace) -> list[Cell]:
    return _GRID_FNS[name](graphed=bool(getattr(args, "graphed", False)))


# ═════════════════════════════════════════════════════════════════════
# "paper" cell sets (see benchmarks/README.md's
# "paper set" section for the full mapping).  Main-text numbers
# quote bf16 ONLY, and the fp32-operand (tf32) column stays "appendix-only
# ... for the arms the appendix table shows" -- as of the current paper
# appendix, `tab:lm_spine_modules`/
# `tab:ssm_mqmkvar_shapes` show tf32+bf16 for the OFFICIALS and the priced
# DEFAULT arms only; the exact-config rows and the feature arms
# are the newer like-for-like headline (README's `like_for_like_<tag>.tex`)
# and are not in that table yet, so they stay bf16-only here too.  `ieee`
# is dropped entirely at the module level ("dropped from every future
# benchmark submit"), so it never appears in either arm set below.
# ═════════════════════════════════════════════════════════════════════
_LM_SPINE_TF32_ARMS = frozenset((
    "core:S-R1", "core:S-R2O2", "core:S-R4O4", "core:HM-LM",
    "SM-R1", "SM-R2O2", "SM-R4O4", "HMM-LM",
    "official_mamba1", "official_mamba2", "official_mamba3",
))
# G34: the captured twins are bf16-only like their eager originals, and
# BOTH SIDES are in the set -- a captured number may only be quoted
# against another captured number.
_GRAPHED_PAPER_ARMS = frozenset(
    a + GRAPH_ARM_SUFFIX for a in _GRAPHED_LIKE_FOR_LIKE)
_LM_SPINE_BF16_ONLY_ARMS = frozenset((
    "SM-R1:m2exact", "SM-R1:m3exact", "HMM:m1exact",
    "HMM:m1exact+rot", "HMM:m1exact+gates", "HMM:m1exact+all",
    "HMM:m1exact+all1",
    "HMM-LM+rot", "HMM-LM+gates", "HMM-LM+all", "HMM-LM+all1",
    "SM-R1:m2exact+rot", "SM-R1:m2exact+gates", "SM-R1:m2exact+all",
    "SM-R1:fused", "SM-R1:modular", "SM-R1:modular+checkpoint",
    "SM-R1:fused@graph", "SM-R1:modular@graph",
    "SM-R1:modular+checkpoint@graph",
)) | _GRAPHED_PAPER_ARMS


def _paper_ssm_lm_spine(c: Cell) -> bool:
    """Officials + the three exact-config arms + the priced default arms
    + the feature arms, at bf16 only (tf32 kept for the officials/
    default arms the appendix table already shows)."""
    if c.arm in _LM_SPINE_TF32_ARMS or c.arm in PAPER_TF32_ARMS["ssm_lm_spine"]:
        return c.precision in ("tf32", "bf16")
    if c.arm in _LM_SPINE_BF16_ONLY_ARMS:
        return c.precision == "bf16"
    return False


_MQMKVAR_TF32_ARMS = frozenset((
    "core:S1", "core:S2", "core:S3", "core:S4", "core:S5",
    "SM1", "SM2", "SM3", "SM4", "SM5",
    "core:HM", "HMM",
    "official_mamba1", "official_mamba2", "official_mamba3",
))
_MQMKVAR_BF16_ONLY_ARMS = frozenset((
    "SM-R1:m2exact", "SM-R1:m3exact", "HMM:m1exact",
    "HMM:m1exact+rot", "HMM:m1exact+gates", "HMM:m1exact+all",
    # `HMM:m1exact+all1` only -- NOT `HMM+all1`: `_hm_design_feature_cells`
    # builds the extra rows only at the ssm_lm_spine-only
    # `feature_names=_HM_FEATURE_NAMES` call site.
    "HMM:m1exact+all1",
    "HMM+rot", "HMM+gates", "HMM+all",
    "SM-R1:m2exact+rot", "SM-R1:m2exact+gates", "SM-R1:m2exact+all",
)) | _GRAPHED_PAPER_ARMS


def _paper_ssm_mqmkvar(c: Cell) -> bool:
    """The same rule as `ssm_lm_spine`, at the MQMKVAR shape's own arms."""
    if c.arm in _MQMKVAR_TF32_ARMS or c.arm in PAPER_TF32_ARMS["ssm_mqmkvar"]:
        return c.precision in ("tf32", "bf16")
    if c.arm in _MQMKVAR_BF16_ONLY_ARMS:
        return c.precision == "bf16"
    return False


PAPER_SETS: dict[str, Callable[[Cell], bool]] = {
    "ssm_mqmkvar": _paper_ssm_mqmkvar,
    "ssm_lm_spine": _paper_ssm_lm_spine,
    # the shared shape set: every cell (bf16 only by construction)
    "ssm_shared_set": lambda c: True,
    "ssm_shared_small": lambda c: True, "ssm_shared_medium": lambda c: True,
    "ssm_shared_large": lambda c: True, "ssm_shared_lm": lambda c: True,
}


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", action="store_true",
                        help="also record the self-CUDA fwd/bwd/sp/glue "
                             "bucket split (adds a profiled pass per cell)")
    parser.add_argument("--graphed", action="store_true",
                        help="G34: also emit an `<arm>@graph` row for each "
                             "like-for-like arm (ours AND the officials), "
                             "whose timed closure is one CUDA-graph replay "
                             "of a whole captured fwd(+bwd) step "
                             "(orbit_ssm.graphs).  Same group, same budget "
                             "and canary rules as the eager rows; bf16 only. "
                             "Both mixers are host-bound at these cells "
                             "(G32/G33), so this is the regime in which a "
                             "wall ratio prices the kernels")


def _extra_meta(args: argparse.Namespace) -> dict:
    return dict(profile=bool(getattr(args, "profile", False)),
                graphed=bool(getattr(args, "graphed", False)),
                params_by_arm=_PARAMS)


def _post_cell(args: argparse.Namespace):
    """Per-cell extras: the profiler buckets (`--profile`) and, for every
    `@graph` row, what being captured COSTS.

    `harness._peak_memory` cannot see a captured row's real footprint --
    a replay allocates nothing, so its `mem_delta_bytes` reads ~0 while
    the static buffers and the graph's private pool sit there all along.
    The two are recorded here instead, straight off the `GraphedStep`."""
    want_profile = bool(getattr(args, "profile", False))

    def hook(cell, closure):
        out: dict = {}
        if cell.grid == "ssm_shared_set" and cell.arm.startswith("SM-R1:") and "@" not in cell.arm:
            from .route_record import served
            try:
                out.update(served(closure))
            except Exception as exc:               # noqa: BLE001
                out["route_error"] = f"{type(exc).__name__}: {exc}"
        if getattr(closure, "compile_seconds", None) is not None:
            out["compile_seconds"] = closure.compile_seconds      # excluded from the timing
            out["compile_mode"] = getattr(closure, "compile_mode", None)
        gs = getattr(closure, "graphed_step", None)
        if gs is not None:
            out["graph_memory"] = dict(gs.memory)
            out["graph_notes"] = list(gs.notes)
            out["graph_warnings"] = [w for w in gs.capture_warnings
                                     if "deprecated" not in w]
        if want_profile:
            try:
                out["cuda_buckets"] = cuda_buckets(closure)
            except Exception as exc:               # noqa: BLE001
                out["cuda_buckets"] = None
                out["cuda_buckets_error"] = f"{type(exc).__name__}: {exc}"
        return out

    return hook


if __name__ == "__main__":
    main_for(build_grid, GRIDS, "benchmarks.bench_ssms",
             add_args=_add_args, extra_meta=_extra_meta,
             post_cell=_post_cell, paper_filter=PAPER_SETS.get)
