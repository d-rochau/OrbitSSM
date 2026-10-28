"""Build public SSM, residual-layer and LM operations for the shared timer."""
from __future__ import annotations

import torch

PASSES = ('fwd', 'fwd_bwd', 'step', 'step_bwd', 'rollout', 'rollout_bwd')
MODULE_ROUTES = ('ssm', 'layer', 'model')


OFFICIAL_FAMILIES = ("official_mamba1", "official_mamba2", "official_mamba3")

# ── case families: arms of a registered family beyond the two shared ones ────────────────────────────────
#: the families `bench_ssms` builds rows for, in grid order
FAMILIES = ["ssd_mimo", "hybrid_mamba"]
#: case family -> config factory ``(d_inner, d_state, **over) -> config``
CASE_CONFIGS: dict = {}
#: case family -> the overrides its smoke row is built with
CASE_SMOKE: dict = {}
#: grid hook points -> callables ``(ctx: dict) -> list[Cell]`` (see the call sites in `bench_ssms`)
GRID_HOOKS: dict = {"ssm_mqmkvar:officials_width": [], "ssm_lm_spine:first": []}
#: grid -> arm names a case family adds to the paper sets (tf32 + bf16 columns)
PAPER_TF32_ARMS: dict = {"ssm_lm_spine": set(), "ssm_mqmkvar": set()}


def register_case_family(name, config, *, smoke=None, after=None):
    """Add a case family: its config factory, its smoke overrides and its place in `FAMILIES`."""
    if name in FAMILIES:
        raise ValueError(f"case family {name!r} is already registered")
    FAMILIES.insert(FAMILIES.index(after) + 1 if after else len(FAMILIES), name)
    CASE_CONFIGS[name] = config
    CASE_SMOKE[name] = dict(smoke or {})


def load_extensions():
    """Import every module ``benchmarks/*_cases.py`` that is present (each registers its arms on import)."""
    import importlib
    import os
    import pkgutil
    for module in pkgutil.iter_modules([os.path.dirname(os.path.abspath(__file__))]):
        if module.name.endswith("_cases"):
            importlib.import_module(f"{__package__}.{module.name}")


def _official_module(route, family, width, state_size, config, backend, device):
    """Keep upstream implementations and their backend restrictions explicit."""
    from .baselines import build_mamba_layer
    from orbit_ssm.model import ResidualBlock

    if backend != "cuda":
        raise ValueError(f"{family} exposes its upstream CUDA implementation; "
                         "torch/sequential are repository backends, not upstream variants")
    if route == "ssm":
        raise ValueError("official families are full mixers; use layer/model or a raw kernel route")
    if route == "layer":
        _, _, mixer = build_mamba_layer(family.removeprefix("official_"), width,
                                       d_state=state_size, B=1, L=1,
                                       device=device, seed=None, **config)
        return ResidualBlock(mixer, width)
    try:
        from experiments.models.mamba_official import OfficialMambaLMHeadModel
    except ImportError:
        raise NotImplementedError(
            f"{family} as a full language model (route {route!r}) is not part of this release; "
            "use route 'layer' or a raw kernel route") from None
    options = dict(config)
    if "expand" in options:
        options["expand_factor"] = options.pop("expand")
    module = OfficialMambaLMHeadModel(family, d_model=width,
                                    d_state=state_size, **options)
    module.vocab_size = int(config["vocab_size"])
    return module


class _ScanOnly(torch.nn.Module):
    """A layer seen as its bare recurrence: `forward` = `layer.scan`, `step` = `layer.scan_step`,
    parameters = the scan's own (no in_proj / conv1d / norm / out_proj)."""
    _LAYER_OWN = ("in_proj", "conv1d", "norm", "out_proj")

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.d_inner, self.proj_source, self.d_sel = layer.d_inner, layer.proj_source, layer.d_sel

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        for n, p in self.layer.named_parameters(remove_duplicate=remove_duplicate):
            if n.split('.')[0] not in self._LAYER_OWN:
                yield ('layer.' + n if not prefix else f'{prefix}.layer.{n}'), p

    def parameters(self, recurse=True):
        return (p for _, p in self.named_parameters())

    def forward(self, x, **kw):
        return self.layer.scan(x, **kw)

    def step(self, x_t, state, **kw):
        return self.layer.scan_step(x_t, state, **kw)

    def init_state(self, batch_size, device=None, dtype=None):
        from orbit_ssm.layers.state import ScanState
        own = getattr(self.layer, "_scan_state", None)         # the layer's carries without the conv window
        return (own() if own is not None else ScanState(self.layer)).init_state(batch_size, device=device, dtype=dtype)


_EXEC_FIELDS = {"backend": "backend", "precision": "precision", "chunk_size": "chunk_size",
                "checkpoint": "checkpoint"}


def split_execution(family, options, **defaults):
    """Pop the run-time keys out of a flat option dict (in place) and return them as an
    `Execution` dict: the `Execution` fields plus the family's layer switches
    (`kernel_options.LAYER_KEYS`) as `kernel_options`.  Constructors take only
    `execution=`; benchmark configs keep their flat spelling and this is the
    one translation."""
    from orbit_ssm.kernel_options import LAYER_KEYS
    ex = {k: v for k, v in defaults.items() if v is not None}
    base = options.pop("execution", None)       # an `execution=` entry is the base
    if base is not None:
        ex.update(dict(base.__dict__) if not isinstance(base, dict) else base)
    for key, field in _EXEC_FIELDS.items():
        if key in options:
            ex[field] = options.pop(key)
    ko = dict(options.pop("kernel_options", None) or {})
    for key in sorted(LAYER_KEYS.get(family, ())):
        if key in options:
            ko[key] = options.pop(key)
    if ko:
        ex["kernel_options"] = ko
    return ex


def build_operation(route, shape, config, *, backend, device, pass_, precision=None):
    """The benchmarked module of one cell; ``config`` uses the current option and execution names."""
    from orbit_ssm.ssm import SSMS, MIXERS, CONFIGS
    from orbit_ssm.model import ResidualBlock, CausalLM

    config = dict(config)
    family = config.pop('family', 'ssd_mimo')
    if family == 'ssd_mimo' and precision is not None:
        from .measure import KERNEL_PRECISION
        required = KERNEL_PRECISION[precision]
        selected = config.get('precision', required)
        if selected != required and not (selected == 'io' and precision != 'ieee'):
            raise ValueError(f'core precision={selected!r} conflicts with benchmark column {precision!r}')
        config['precision'] = selected
    batch, length, state_size = (int(shape[k]) for k in ('B', 'L', 'N'))
    if min(batch, length, state_size) <= 0:
        raise ValueError('B, L and N must be positive')
    if pass_ not in PASSES:
        raise ValueError(f'unknown pass {pass_!r}')
    stepping = pass_.startswith(('step', 'rollout'))
    rollout = pass_.startswith('rollout')
    steps = length if rollout else 1
    width = int(shape.get('d_model', shape.get('d_inner', 0)))
    if width <= 0:
        raise ValueError('module shape needs positive d_model or d_inner')
    if family in OFFICIAL_FAMILIES:
        if stepping:
            raise NotImplementedError(
                'upstream step/cache adapters are not yet benchmarked here; '
                'use fwd/fwd_bwd (do not replace step with a length-one forward)')
        mod = _official_module(route, family, width, state_size, config, backend, device)
    elif route == 'ssm':
        if family not in ('ssd_mimo', 'hybrid_mamba'):
            raise ValueError(f'the bare core of family {family!r} needs explicit token maps; use route=layer')
        # the bare recurrence: the layer's `scan` / `scan_step` (its own parameters only)
        config.setdefault('d_model', width)
        ex = split_execution(family, config, backend=backend)
        d_inner = int(shape.get('d_inner', width))
        layer = SSMS[family](CONFIGS[family](d_model=config.pop('d_model'), d_state=state_size, d_inner=d_inner,
                                             **config), execution=ex)
        mod = _ScanOnly(layer)
        width = mod.d_inner
    else:
        ex = split_execution(family, config, backend=backend)
        if route == 'layer':
            mixer = MIXERS[family](CONFIGS[family](d_model=width, d_state=state_size, **config), execution=ex)
            mod = ResidualBlock(mixer, width)
        elif route == 'model':
            lm_kw = {k: config.pop(k) for k in ('vocab_size', 'n_layers', 'd_mlp', 'tie_word_embeddings',
                                                'pad_vocab_size_multiple') if k in config}
            mod = CausalLM(vocab_size=lm_kw.pop('vocab_size'), layers=dict(type=family, d_state=state_size, **config),
                           n_layers=lm_kw.pop('n_layers'), d_model=width, execution=ex, **lm_kw)
        else:
            raise ValueError(f'unknown module route {route!r}')
    mod = mod.to(device)
    for child in mod.modules():
        if hasattr(child, 'set_inference'):
            child.set_inference(False)  # functional caches, identical for eager/compile
    token_count = steps if stepping else length
    if route == 'model':
        x = torch.randint(mod.vocab_size, (batch, token_count), device=device)
    else:
        x = torch.randn(batch, token_count, width, device=device, requires_grad=True)
    leaves = [x, *mod.parameters()]
    select = None
    if route == 'ssm' and mod.proj_source == 'layer_input':
        select = torch.randn(batch, token_count, mod.d_sel, device=device, requires_grad=True)
        leaves.append(select)
    if stepping:
        state = mod.init_state(batch, device=device)
        def boundary(value):
            if torch.is_tensor(value):
                result = value.detach()
                if result.is_floating_point() and pass_.endswith('bwd'):
                    result.requires_grad_(True)
                    leaves.append(result)
                return result
            if isinstance(value, dict):
                return type(value)({k: boundary(v) for k, v in value.items()})
            if isinstance(value, list):
                return [boundary(v) for v in value]
            return value
        state = boundary(state)
        # Populate validation schemas before Dynamo tracing.
        if hasattr(mod, 'validate_state'):
            mod.validate_state(state)
        def call():
            carry = state
            ys = []
            for t in range(steps):
                kw = {'sel_x': select[:, t]} if select is not None else {}
                y, carry = mod.step(x[:, t], carry, **kw)
                ys.append(y)
            return (torch.stack(ys, 1) if rollout else ys[0]), carry
    else:
        def call():
            return mod(x, **({'sel_x': select} if select is not None else {}))
    return call, leaves, mod
