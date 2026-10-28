"""Contract suite: generic checks (tests/test_contract.py) over per-family
adapters (tests/_contract/adapters/).  See tests/_contract/core.py.

An adapter is a module-level object with:

  family            str, the `--family` selector name
  space             core.Space: factors + constraints (+ infeasible(cfg, device), route(cfg))
  served_route()    the route telemetry of the last call (compared with space.route)
  recipes()         [(name, partial cfg)]: shipped production configs, run exactly
  env(cfg)          {env var: value} the cell pins (routes, ...)
  build(cfg, seed)  inputs for one draw
  run(inputs)       (outputs, leaves): differentiable outputs and the leaves
  oracle(inputs, outputs, grads, cfg, pool)
                    asserts outputs/grads against the INDEPENDENT oracle
                    under the family's tolerance rule
  draws(cfg)        seeds for one cell (16-bit cells need >= 5, A7)
  continuation(cfg, inputs)
                    None, or (whole, split, check) for the state-carry check
  route_pairs(cfg)  [(label, env_a, env_b, exact)] -- the same cell under two
                    routes that must agree (bitwise if exact)
  compiled(cell)    True when the cell also runs under torch.compile (`compile_check(c, label)`, when the adapter has
                    it, replaces the generic eager-vs-oracle compile check: fullgraph, no recompile, ...)
  build_production(cfg)
                    inputs at the shipped production geometry (test_recipes_fit)

Optional hooks (getattr-defaulted by the generic checks):

  prepare()         once per session, before any cell (e.g. a calibration
                    check that must fail fast)
  new_pool(cfg, primary=True)  the draw pool of a cell (every family pool takes `primary`), when the family's bias rule is
                    not `_tolerance.DrawPool`'s (add / assert_unbiased)
  bitwise_backward  False when the family has no deterministic backward: the
                    forward is compared bitwise, each gradient through
                    grad_repeatability(cfg, label, a, b)
  route_pair_close(cfg, label, a, b)
                    scores a NON-exact route pair arm against arm (else each
                    arm runs the oracle check)
  causal_outputs(outs), slice_sample(args, kw, i), variant(args, kw, kind)
                    hooks for causality / batch-independence / layout checks
  has_step(c), check_step(c, label)
                    the single-token step loop equals the parallel forward (families with a decode path)
  arm(attrs, kw)    context manager applying the '@'-keyed entries of a route-pair arm (module attributes /
                    module globals that are not environment variables) and restoring them
  double_backward   "unsupported": no create_graph through the scan (declared,
                    not xfail)
  pinned_env        env that pins the launch selection (bit-identity pairs)
  mode_keys         env keys that select the launch mode; unset in the
                    default-mode production slice (test_production_mode)
  nondeterministic_for(cfg)
                    the same map when it depends on the cell (precision, block size, architecture); the key
                    "grad:*" declares every gradient; declared tensors are held to 4x their own run-to-run
                    spread by the metamorphic checks (`_noise_floor`)
  determinism_context(cfg, args, kw), grad_repeatability(cfg, label, a, b, ctx=, name=)
                    the kappa-aware bound on a declared tensor's run-to-run spread
  nondeterministic  {tensor name: reason} -- MEASURED run-to-run variation,
                    excluded from test_reproducible_across_processes
  tensor_names(args, kw), check_forward(cfg, args, kw, values, label)

  fatal_guards      ("package.module:function", ...): the helpers the family's fallback sites call first in a broad `except`
                    (re-raise a device error); tests/test_contract_refusal.py holds each

Refusal-path hooks (tests/test_contract_refusal.py; CPU where the refusal is decided before the launch):

  construct(c)      builds the cell's op / module without launching (CPU or meta device); raises the constructor's refusal
  constructor_reasons
                    prefixes of the `space.infeasible` reasons that `construct` itself must raise (the others are decided at
                    call time and are checked by the cells)
  refusals()        [(label, fixed factors, mutate(args, kw) -> (args, kw), exception type, message regex)]: requests the
                    family must refuse BY NAME before launching anything
  inert_options(c)  {option: values}: options with no meaning in cell `c` that a caller may still carry
  with_inert(c, args, kw, name, value)
                    the call's (args, kw) with the stray option added
  entries(c, args, kw)
                    [(label, thunk(args, kw) -> [tensors])]: every public entry that takes the cell's options (default: only
                    `call`); the inert check runs the stray option through each of them

Torch-route hooks (CPU; tests/test_contract_torch.py, tests/_contract/torch_route.py):

  torch_cells()     [Cell]: a pair-covering set of the options visible on backend="torch" x dtype (fp32 / fp64 / bf16); a cell
                    of kind "scan" runs `torch_scan_check(cell)` (the family's torch scan function against its oracle), any
                    other cell is a layer cell
  torch_layer(cfg, dtype, seed)
                    a `torch_route.TorchLayer`: the layer on backend="torch" with its parameters off their initialisation,
                    its independent fp64 oracle (or None), the depth of its recurrence
  torch_scan_check(cell)                           optional, for "scan" cells
  torch_known_failure(cfg, check)                  optional: the reason of a DIAGNOSED product failure (a strict xfail)
  torch_state_dtype_exceptions(cfg, route)         optional: {carry: reason}, "forward" / "step" returns a carry in another
                                                   dtype than supplied (declared, and red when it stops occurring)
"""
FAMILIES = {}

#: the environment variable that turns the optional extension packages off (the suite then runs on its own base spaces)
NO_OVERLAY = "HYBRIDSSM_NO_OVERLAY"


def overlays():
    """The optional extension modules of this suite: the `overlay` module of each package `tests/_contract_<name>/`, each with
    `extend(adapter)`.

    The adapters in this package declare a BASE space (the options a user of the library meets); an extension package that
    sits next to this one, when the checkout has it, widens the spaces and registers further checks.  A tree without such a
    package (the exported suite) runs the base alone; `HYBRIDSSM_NO_OVERLAY=1` does the same in a tree that has one."""
    import importlib
    import os
    import pathlib
    if os.environ.get(NO_OVERLAY, "") not in ("", "0"):
        return []
    tests = pathlib.Path(__file__).resolve().parents[1]
    return [importlib.import_module(f"{d.name}.overlay") for d in sorted(tests.glob("_contract_*"))
            if any(d.glob("overlay.p*"))]


def overlay_files(family=None):
    """Repo-relative paths of the active extension files (they join the digests that key the cached covering sets)."""
    import pathlib
    tests = pathlib.Path(__file__).resolve().parents[1]
    return sorted(str(p.resolve().relative_to(tests.parent)) for m in overlays()
                  for p in pathlib.Path(m.__file__).parent.glob("*.py"))


def register(adapter):
    FAMILIES[adapter.family] = adapter
    for mod in overlays():
        mod.extend(adapter)
    return adapter


def load_all():
    import importlib
    import pkgutil
    from . import adapters
    for info in sorted(pkgutil.iter_modules(adapters.__path__), key=lambda m: m.name != "mh" and m.name != "ssd"):
        importlib.import_module(f"{__name__}.adapters.{info.name}")      # registers on import
    return FAMILIES
