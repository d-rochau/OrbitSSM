
from __future__ import annotations
import dataclasses
import functools
from dataclasses import dataclass, field, fields
from .configuration import positive_int, resolve_alias
BACKENDS = ('cuda', 'torch', 'sequential', 'auto')
_RETIRED_KWARGS = ('engine',)
WRITE_RULES = ('euler', 'trapezoid', 'zoh')
WRITE_RULES_IMPLEMENTED = ('euler', 'trapezoid')
WRITE_RULE_DEFAULT = 'trapezoid'
_WRITE_RULE_RETIRED = {'trapezoidal': 'trapezoid', 'zoh_exact': 'zoh', 'trap': 'trapezoid'}

def resolve_write_rule(write_rule: str, who: str, supported=WRITE_RULES_IMPLEMENTED, field: str='write_rule') -> str:
    if write_rule in supported:
        return write_rule
    if write_rule == 'zoh':
        raise NotImplementedError(f"{who}: {field}='zoh' denotes the EXACT zero-order hold, forcing Delta*phi(Delta A)*b(x_t) with phi(Z) = Z^-1 (e^Z - I), and is NOT IMPLEMENTED.  It is accepted by name so that a request for exact ZOH is REFUSED instead of silently resolving to the truncation phi ~ I -- which is the Euler write, {field}='euler'.  Implemented rules: {', '.join(supported)}")
    if write_rule in _WRITE_RULE_RETIRED:
        raise ValueError(f"{who}: {field}={write_rule!r} was RETIRED on 2026-09-20; use {field}={_WRITE_RULE_RETIRED[write_rule]!r}.  ZOH names the EXACT hold; the truncated rule is 'euler'.")
    raise ValueError(f"{who}: unknown {field} {write_rule!r}; supported: {', '.join(supported)} (plus 'zoh', which is named and refused because the exact hold is not implemented)")

def check_backend(backend: str, who: str, backends=BACKENDS) -> str:
    if backend not in backends:
        raise ValueError(f"{who}: backend must be one of: {', '.join(backends)}; got {backend!r}")
    return backend

def reject_retired_keys(keys, who: str, backends=BACKENDS) -> None:
    for name in keys:
        if name in _RETIRED_KWARGS:
            raise TypeError(f"{who}: unknown keyword {name!r}; use backend= with one of: {', '.join(backends)}")

def _valid_chunk_size(v) -> bool:
    if isinstance(v, str):
        return v in ('auto', 'measure')
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return v >= 16 and v & v - 1 == 0
    return False

def reject_unknown_kwargs(kwargs: dict, who: str, backends=BACKENDS) -> None:
    if not kwargs:
        return
    reject_retired_keys(kwargs, who, backends)
    raise TypeError(f'{who}: unknown kwargs {sorted(kwargs)}')

def refuse_retired_kwargs(cls):
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        from .config_spec import reject_removed
        reject_retired_keys(kwargs, cls.__name__)
        reject_removed(kwargs, cls.__name__, getattr(cls, '_REMOVED_KWARGS', None))
        orig_init(self, *args, **kwargs)
    cls.__init__ = __init__
    return cls

@refuse_retired_kwargs
@dataclass
class SSMCoreConfig:
    d_inner: int = 0
    d_state: int = 64
    R: int = 1
    O: int | None = None
    ngroups: int | None = None
    n_heads: int | None = None
    heads: int | None = None
    H: int | None = None
    head_subsystems: int | None = None
    P: int | None = None
    write_rule: str | None = None
    proj_source: str = 'layer_input'
    d_model: int | None = None
    bc_conv: bool | None = None
    d_conv: int = 4
    backend: str = 'auto'
    precision: str = 'io'
    dt_min: float = 0.001
    dt_max: float = 0.1
    chunk_size: int | str = 'auto'
    kernel_options: dict = field(default_factory=dict)

    def __post_init__(self):
        from .kernel_options import validate_for_config
        self.kernel_options = validate_for_config(self, self.kernel_options)
        positive_int(self.d_inner, 'd_inner')
        positive_int(self.d_state, 'd_state')
        resolve_alias('heads', heads=self.heads, H=self.H, n_heads=self.n_heads)
        resolve_alias('head_subsystems', head_subsystems=self.head_subsystems, P=self.P)
        check_backend(self.backend, type(self).__name__)
        if self.precision not in ('ieee', 'tf32', 'io'):
            raise ValueError(f'precision must be ieee|tf32|io, got {self.precision!r}')
        if not _valid_chunk_size(self.chunk_size):
            raise ValueError(f"chunk_size must be 'auto', 'measure' or a power-of-two >= 16, got {self.chunk_size!r}")

    def core_fields(self) -> dict:
        core_names = {f.name for f in fields(SSMCoreConfig)}
        return {k: v for k, v in dataclasses.asdict(self).items() if k in core_names}

    def replace(self, **kw) -> 'SSMCoreConfig':
        return dataclasses.replace(self, **kw)

@dataclass
class SpecFields:
    rows: list | None = None
    rotation: dict | None = None
    maps: dict | None = None
    theta_per_head: bool | None = None
    learn_theta_freq: bool | None = None
    use_theta_h: bool | None = None
    b_norm: bool | None = None
    c_norm: bool | None = None
    bc_norm: bool | None = None
    bias_gran: str | None = None
    use_g_b: bool | None = None
    use_g_c: bool | None = None

    def spec_kwargs(self) -> dict:
        from .config_spec import FLAT_KWARGS
        import copy as _copy
        names = ('rows', 'rotation', 'maps') + tuple(FLAT_KWARGS)
        return {n: _copy.deepcopy(getattr(self, n)) for n in names if hasattr(self, n) and getattr(self, n) is not None}

def _refuse(cfg: SSMCoreConfig, family: str, unsupported: dict) -> None:
    problems = [f'{name}={getattr(cfg, name)!r}' for name, refused in unsupported.items() if refused(cfg)]
    if problems:
        raise NotImplementedError(f"{family} does not support: {', '.join(problems)}. Use a family that does, or drop the field.")

def build_ssm(cfg: SSMCoreConfig):
    if type(cfg) is SSMCoreConfig:
        from .ssm.ssd_mimo import SSDMimoConfig
        cfg = SSDMimoConfig(**cfg.core_fields())
    return cfg.build()

def __getattr__(name):
    if name not in ('FAMILY_CONFIGS', 'SSDMimoConfig', 'HybridMambaConfig'):
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from . import ssm
    if name == 'FAMILY_CONFIGS':
        return ssm.CONFIGS
    if name in ('SSDMimoConfig', 'HybridMambaConfig'):
        return getattr(ssm, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
