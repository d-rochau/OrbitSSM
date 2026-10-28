
from __future__ import annotations
import copy
import inspect
import json
from numbers import Integral
CONFIG_SCHEMA = 'orbit_ssm.core/v0'
SEMANTICS_VERSION = 'unfrozen'

def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return int(value)

def resolve_dt_rank(dt_rank, *, d_sel: int, heads: int, d_model=None, force_dt_dense: bool=False) -> int | None:
    if not isinstance(force_dt_dense, bool):
        raise ValueError('force_dt_dense must be a boolean')
    if force_dt_dense:
        if dt_rank is not None and dt_rank != 'adaptive':
            raise ValueError('force_dt_dense conflicts with an explicit bottleneck dt_rank')
        return None
    if dt_rank is None:
        return None
    if dt_rank not in ('adaptive', 'auto'):
        return positive_int(dt_rank, 'dt_rank')
    if dt_rank == 'auto' and d_model is None:
        raise ValueError("dt_rank='auto' requires d_model (Mamba-1 uses ceil(d_model/16))")
    width = positive_int(d_sel if d_model is None else d_model, 'delta reference width')
    rank = (width + 15) // 16
    if dt_rank == 'auto' or rank * (d_sel + heads) < d_sel * heads:
        return rank
    return None

def resolve_alias(name: str, *, default=None, **aliases) -> int | None:
    values = {key: positive_int(value, key) for key, value in aliases.items() if value is not None}
    if len(set(values.values())) > 1:
        raise ValueError(f'conflicting {name} aliases: {values}')
    if values:
        return next(iter(values.values()))
    return None if default is None else positive_int(default, name)

def resolve_heads(d_inner: int, *, heads=None, H=None, n_heads=None, head_subsystems=None, P=None, default_subsystems=None):
    width = positive_int(d_inner, 'd_inner')
    h = resolve_alias('heads', heads=heads, H=H, n_heads=n_heads)
    p = resolve_alias('head_subsystems', head_subsystems=head_subsystems, P=P)
    if h is None and p is None:
        p = resolve_alias('head_subsystems', default=default_subsystems)
        if p is None:
            raise ValueError('specify heads/H/n_heads or head_subsystems/P')
    if h is None:
        if width % p:
            raise ValueError(f'head_subsystems={p} must divide d_inner={width}')
        h = width // p
    elif p is None:
        if width % h:
            raise ValueError(f'heads={h} must divide d_inner={width}')
        p = width // h
    if h * p != width:
        raise ValueError(f'heads*head_subsystems must equal d_inner ({h}*{p} != {width})')
    return (h, p)

def resolve_ssd_geometry(d_inner: int, *, R: int, ngroups=None, **aliases):
    R = positive_int(R, 'R')
    group_width = None if ngroups is None else positive_int(ngroups, 'ngroups') * R
    p = resolve_alias('head_subsystems', head_subsystems=aliases.pop('head_subsystems', None), P=aliases.pop('P', None), ngroups_times_R=group_width)
    h, p = resolve_heads(d_inner, head_subsystems=p, default_subsystems=max(1, 64 // R) * R, **aliases)
    if p % R:
        raise ValueError(f'SSD head_subsystems=P={p} must be divisible by R={R} (P=G*R)')
    return (h, p, p // R)
LEGACY_CORE_DEFAULTS: dict = {}

def fill_legacy_core_fields(family, model: dict) -> None:
    for key, value in LEGACY_CORE_DEFAULTS.get(family, {}).items():
        if key not in model:
            model[key] = value(model) if callable(value) else value

class Configurable:
    config_schema = CONFIG_SCHEMA

    def resolved_config(self) -> dict:
        return copy.deepcopy(self._resolved_constructor_config)

    @classmethod
    def _with_legacy_defaults(cls, document):
        if isinstance(document, dict) and isinstance(document.get('model'), dict):
            document = copy.deepcopy(document)
            fill_legacy_core_fields(document.get('family'), document['model'])
        return document

    @classmethod
    def from_config(cls, document: dict):
        document = cls._with_legacy_defaults(document)
        expected = {'schema', 'semantics_version', 'family', 'model', 'execution'}
        if set(document) != expected:
            raise ValueError(f'config fields must be exactly {sorted(expected)}')
        if document['schema'] != cls.config_schema or document['semantics_version'] != SEMANTICS_VERSION:
            raise ValueError('unsupported config schema or semantics version')
        if document['family'] != cls.config_family:
            raise ValueError(f"expected family {cls.config_family!r}, got {document['family']!r}")
        model, execution = (document['model'], document['execution'])
        if set(model) & set(execution):
            raise ValueError('model and execution config fields overlap')
        module = cls._from_config_sections(copy.deepcopy(model), copy.deepcopy(execution))
        for section in ('model', 'execution'):
            if set(document[section]) != set(module._resolved_constructor_config[section]):
                raise ValueError(f'{section} fields do not match the core config schema')
        if module.resolved_config() != document:
            raise ValueError('config must contain canonical resolved values, including nested fields')
        return module

    @classmethod
    def _from_config_sections(cls, model, execution):
        return cls(**model, **execution)

class MixerConfigurable(Configurable):
    config_schema = 'orbit_ssm.mixer/v0'

    @classmethod
    def _with_legacy_defaults(cls, document):
        if isinstance(document, dict) and isinstance(document.get('model'), dict) and isinstance(document['model'].get('core'), dict):
            document = copy.deepcopy(document)
            fill_legacy_core_fields(document.get('family'), document['model']['core'])
        return document

    @classmethod
    def constructor_kwargs(cls, document):
        if document.get('schema') != cls.config_schema or document.get('family') != cls.config_family:
            raise ValueError('mixer recipe schema/family mismatch')
        model, execution = (copy.deepcopy(document['model']), copy.deepcopy(document['execution']))
        if set(model) != {'d_model', 'expand', 'd_conv', 'out_norm', 'core'}:
            raise ValueError('mixer model fields do not match the config schema')
        if set(execution) != {'core_execution', *cls.config_execution_fields}:
            raise ValueError('mixer execution fields do not match the config schema')
        core = model.pop('core')
        core_execution = execution.pop('core_execution')
        if set(core) & set(core_execution):
            raise ValueError('core model and execution fields overlap')
        core.update(core_execution)
        core['out_norm'] = model.pop('out_norm')
        if model['expand'] is None:
            model['d_inner'] = core['d_inner']
        return dict(**model, **execution, ssm_kwargs=core)

    @classmethod
    def _from_config_sections(cls, model, execution):
        return cls(**cls.constructor_kwargs(dict(schema=cls.config_schema, family=cls.config_family, model=model, execution=execution)))

def record_mixer_config(module, *, expand, out_norm, execution):
    core = module.ssm.resolved_config()
    module._resolved_constructor_config = dict(schema=module.config_schema, semantics_version=SEMANTICS_VERSION, family=module.config_family, model=dict(d_model=module.d_model, expand=expand, d_conv=module.d_conv, out_norm=out_norm, core=core['model']), execution=dict(core_execution=core['execution'], **execution))

def mixer_core_options(options, *, config_type, d_model, d_inner, d_state, d_conv, **defaults):
    from .config import SSMCoreConfig
    if isinstance(options, SSMCoreConfig):
        if type(options) is not config_type:
            raise TypeError(f'expected {config_type.__name__}, got {type(options).__name__}')
        options = options.to_ssm_kwargs()
    kw = dict(d_model=d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv, **defaults)
    options = copy.deepcopy(options or {})
    if options.get('d_model') is None:
        options.pop('d_model', None)
    kw.update(options)
    for name, width in (('d_inner', d_inner), ('d_model', d_model)):
        if kw[name] != width:
            raise ValueError(f'core {name}={kw[name]} conflicts with mixer {name}={width}')
    return kw

def record_constructor_config(module, values, *, constructor, overrides=None, execution=(), exclude=()):
    parameters = inspect.signature(constructor).parameters
    resolved = {name: copy.deepcopy(values[name]) for name, param in parameters.items() if name != 'self' and name not in exclude and (param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD))}
    resolved.update(overrides or {})
    for name in ('n_heads', 'H', 'P'):
        resolved.pop(name, None)
    resolved.update(heads=module.H, head_subsystems=module.P)
    execution_values = {key: resolved.pop(key) for key in execution}
    document = dict(schema=CONFIG_SCHEMA, semantics_version=SEMANTICS_VERSION, family=module.config_family, model=resolved, execution=execution_values)
    module._resolved_constructor_config = json.loads(json.dumps(document, allow_nan=False))
