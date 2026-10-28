
from __future__ import annotations
import functools
import os
import threading
from typing import Any, Callable
__all__ = ['KernelOptionError', 'FAMILIES', 'SUPPORTED_KERNEL_OPTIONS', 'schema', 'defaults', 'validate', 'validate_for_config', 'resolve', 'non_default', 'record', 'family_of_config', 'push', 'pop', 'current', 'env_override', 'scoped_kwarg', 'scoped_ctx']

class KernelOptionError(ValueError):
    pass
_JSON_SAFE = (str, int, float, bool, type(None))

class _Opt:
    __slots__ = ('key', 'default', 'values', 'check', 'env', 'doc', 'open_until')

    def __init__(self, key: str, *, default: Any, env: str | None, doc: str, open_until: str, values: tuple | None=None, check: Callable[[Any], bool] | None=None):
        self.key = key
        self.default = default
        self.values = values
        self.check = check
        self.env = env
        self.doc = doc
        self.open_until = open_until

    def allowed(self) -> str:
        if self.values is not None:
            return ' | '.join((repr(v) for v in self.values))
        return self.doc

    def accepts(self, value: Any) -> bool:
        if self.values is not None and value in self.values:
            return any((value == v and type(value) is type(v) for v in self.values))
        if self.check is not None:
            return bool(self.check(value))
        return False

    def to_env(self, value: Any) -> str | None:
        if value is None:
            return ''
        if isinstance(value, bool):
            return '1' if value else '0'
        return str(value)

def _pow2_chunk_or_auto(v) -> bool:
    if v == 'auto':
        return True
    if isinstance(v, bool) or not isinstance(v, int):
        return False
    return v >= 16 and v & v - 1 == 0

def _int_in(lo: int, hi: int) -> Callable[[Any], bool]:

    def _f(v) -> bool:
        if v is None:
            return True
        if isinstance(v, bool) or not isinstance(v, int):
            return False
        return lo <= v <= hi
    return _f
_SSD_MIMO_OPTS: tuple[_Opt, ...] = (_Opt('autotune', default='auto', values=('pinned', 'light', 'full', 'auto', 'off'), env='AUTOTUNE', doc="THE ONE SWITCH, as a tier.  The env lever SSD_MIMO_AUTOTUNE takes 'pinned' | 'light' (default) | 'full'; the legacy spellings 'off' and 'auto' mean 'pinned' and 'light'.  'pinned' runs no timed search in either direction -- the forward takes its device heuristic (cache neither read nor written), the chunk-size and shared-memory trials are skipped, the fused backward's autotune grids collapse to one candidate each and the SISO kernels take their measured pins (implementation implementation.13.8).  'light' tunes the fused backward over num_warps only (4 configs); 'full' also tunes stages and maxnreg (16).  As a CONFIG value this key accepts the same tiers and legacy aliases. CAVEAT, stated rather than hidden: fused backward and SISO forward, backward and rotation-adjoint grids are built AT IMPORT, so only the ENV spelling (e.g. SSD_MIMO_AUTOTUNE=pinned) reaches them -- set it before importing orbit_ssm.  As a config value this key covers the measure-based halves.", open_until="the backward's grids stop being decorator arguments, at which point the env caveat goes and the key covers both halves on its own"), _Opt('fwd_cfg', default='auto', values=('auto', 'measure', 'off'), env='FWD_CFG', doc="which rung picks the fused forward's launch config (implementation implementation.13.2)", open_until="the REALMODE4 ladder rules whether 'measure' becomes the default; the loser's name is deleted"), _Opt('fwd_cfg_iters', default=3, check=_int_in(1, 64), env='FWD_CFG_ITERS', doc="timed reps per candidate under fwd_cfg='measure' (int 1..64)", open_until='with `fwd_cfg` -- it has no meaning without it'), _Opt('fwd_warps', default=None, values=(None, 1, 2, 4, 8), env='FWD_WARPS', doc='explicit num_warps pin for the fused forward (None = let the dispatcher choose)', open_until='the measured dispatcher (implementation.13.2) is trusted on every device we ship on; the three pins go together'), _Opt('fwd_stages', default=None, check=_int_in(1, 8), env='FWD_STAGES', doc='explicit num_stages pin for the fused forward (None, or int 1..8)', open_until='with `fwd_warps`'), _Opt('fwd_maxnreg', default=None, check=_int_in(16, 255), env='FWD_MAXNREG', doc='explicit maxnreg pin for the fused forward (None, or int 16..255; dropped on a HIP build, implementation.13.5)', open_until='with `fwd_warps`'), _Opt('backward_states', default='recompute', values=('recompute', 'io'), env='SALL', doc="the backward's per-chunk state policy: 'recompute' replays the forward in STATE_ONLY mode, 'io' stores (implementation implementation.11)", open_until='the 2026-09-20 execution policy (contract implementation) picks one; the losing policy is deleted, not kept'), _Opt('chunk_size_bwd', default='auto', check=_pow2_chunk_or_auto, env='CHUNK_BWD', doc="the chunk T the BACKWARD runs at: 'auto' (the forward's, the shipped schedule bit for bit) or a power of two >= 16 (implementation implementation.12 lever 2b)", open_until="`_CHUNK_BWD_TABLE` is filled by a measured (T_fwd, T_bwd) pair, which makes 'auto' the only answer"), _Opt('smem_strategy', default='measure', values=('measure', 't_halve', 'n_split'), env='SMEM_STRATEGY', doc='the shared-memory OVERFLOW repair to use when a cell does not fit: measure both, or pin one (implementation implementation.10)', open_until='a measurement rules one repair dominant at every cell that overflows'))
MH_FWD_CONFIG_KEYS = ('BLOCK_P', 'BLOCK_L', 'NUM_WARPS', 'PF', 'TS', 'CKV', 'FPACK')
MH_BWD_CONFIG_KEYS = ('TP_P', 'NS', 'NP', 'NUM_WARPS', 'PU', 'BCS', 'THS', 'AGX', 'MAXNREG', 'HP', 'BLOCK_L', 'TS', 'LWS')
MH_CARRY_CONFIG_KEYS = ('BLOCK_C', 'BLOCK_T', 'NUM_WARPS')

def parse_mh_config(v, keys) -> dict:
    txt = str(v).strip()
    if txt.startswith('{'):
        import json
        try:
            obj = json.loads(txt)
        except ValueError as exc:
            raise ValueError(f'bad MH config {txt!r}: {exc}') from None
        if not isinstance(obj, dict) or not all((isinstance(val, int) and (not isinstance(val, bool)) for val in obj.values())):
            raise ValueError(f'bad MH config {txt!r}; keys: {keys}')
        txt = ','.join((f'{k}={val}' for k, val in obj.items()))
    out = {}
    for item in filter(None, (t.strip() for t in txt.split(','))):
        k, _, val = item.partition('=')
        k = k.strip().upper()
        if k not in keys or not val.strip().lstrip('-').isdigit():
            raise ValueError(f'bad MH config item {item!r}; keys: {keys}')
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
_HYBRID_MAMBA_OPTS: tuple[_Opt, ...] = (_Opt('autotune', default='light', values=('pinned', 'light', 'full', 'auto', 'off'), env='AUTOTUNE', doc="the MH launch-config search, as a tier (env HYBRID_MH_AUTOTUNE, legacy '0' = pinned, '1' = light; config aliases 'off' = pinned, 'auto' = light).  'pinned' times nothing: the per-device table / heuristic config (or `fwd_config` / `bwd_config`) is launched as is -- deterministic across processes and ranks, two compiles per shape.  'light' searches the forward's (BLOCK_P, BLOCK_L, PF) corners and the backward's NS/TP_P axes; 'full' also warps, slab counts, p-sharding and register caps", open_until='a per-device pinned table covers every shipped cell and light never beats it'), _Opt('fwd_config', default=None, check=_mh_cfg_check(MH_FWD_CONFIG_KEYS), env='FWD_CONFIG', doc="explicit forward launch config, e.g. 'BLOCK_P=4,BLOCK_L=256,PF=1' or the same as a JSON object (keys: " + ', '.join(MH_FWD_CONFIG_KEYS) + '); never searched, at every tier (BLOCK_P still clamped to a divisor of P)', open_until='the pinned table is trusted on every device'), _Opt('bwd_config', default=None, check=_mh_cfg_check(MH_BWD_CONFIG_KEYS), env='BWD_CONFIG', doc="explicit backward launch config, e.g. 'TP_P=2,NS=4' or the same as a JSON object (keys: " + ', '.join(MH_BWD_CONFIG_KEYS) + '); never searched, at every tier; a mem= dial wins over it (precedence config > mem > pin > env)', open_until='with `fwd_config`'), _Opt('rotation', default='fold', values=('fold', 'in_scan'), env='ROTATION', doc="the rotation route (env HYBRID_MH_ROTATION; owner 2026-10-02): 'fold' folds the head-dependent phase into per-head B/C maps ahead of ONE real scan and splits a mixed launch into its rotating window (folded) and its real rows (plain real scan); C15 training within 7 % of all-real at ~1.18x its peak memory (A100 9353541, Blackwell 9353542).  'in_scan' rotates inside the scan (A/B only; also the counted fallback for per-head theta and row-window launches)", open_until='the fold serves per-head theta, row windows and state continuation; then in-scan rotation retires'), _Opt('carry_fwd_config', default=None, check=_mh_cfg_check(MH_CARRY_CONFIG_KEYS), env='CARRY_FWD_CONFIG', doc="explicit config of the time-split forward carry kernel, e.g. 'BLOCK_C=16,BLOCK_T=128,NUM_WARPS=4'; launched alone at every tier (keys: " + ', '.join(MH_CARRY_CONFIG_KEYS) + ')', open_until='with `fwd_config`'), _Opt('carry_bwd_config', default=None, check=_mh_cfg_check(MH_CARRY_CONFIG_KEYS), env='CARRY_BWD_CONFIG', doc='explicit config of the time-split adjoint carry kernel; as `carry_fwd_config`', open_until='with `fwd_config`'), _Opt('gate_bias_partial', default='auto', values=('auto', False, True), env='BWD_GATE_BIAS_PARTIAL', doc="per-group gate-bias gradients as per-block partials summed on the host (True) or relaxed fp32 atomics into the (G, N) buffer (False); 'auto' (default) = atomics: with relaxed atomics the two tie on H100 (measurement) and atomics need 32 MiB less at the LLM cell and keep the P == 1 head packing. Legacy env MH_BWD_GATE_BIAS_PARTIAL", open_until='Blackwell/A100 confirm the tie with relaxed atomics; then the partial path retires or stays an A/B lever'))
_SCHEMAS: dict[str, dict[str, _Opt]] = {'ssd_mimo': {o.key: o for o in _SSD_MIMO_OPTS}, 'hybrid_mamba': {o.key: o for o in _HYBRID_MAMBA_OPTS}}
FAMILIES: tuple[str, ...] = tuple(sorted(_SCHEMAS))
SUPPORTED_KERNEL_OPTIONS: dict[str, frozenset[str]] = {f: frozenset(s) for f, s in _SCHEMAS.items()}
_ENV_PREFIX = {'ssd_mimo': 'SSD_MIMO_', 'hybrid_mamba': 'HYBRID_MH_'}
_CHECKED = False

def _ensure_checked() -> None:
    global _CHECKED
    if not _CHECKED:
        _CHECKED = True
        _check_schema()

def _check_schema() -> None:
    from dataclasses import fields as _fields
    from .config import SSMCoreConfig
    reserved = {f.name for f in _fields(SSMCoreConfig)} - {'kernel_options'}
    for fam, sch in _SCHEMAS.items():
        for key, opt in sch.items():
            if key in reserved:
                raise KernelOptionError(f'kernel_options[{fam}]: key {key!r} collides with an SSMCoreConfig field; a kernel option selects routing only and may never be a second channel for a parameter-bearing setting (implementation implementation.2)')
            if not opt.open_until:
                raise KernelOptionError(f'kernel_options[{fam}]: key {key!r} declares no `open_until`; a key that no measurement could retire does not belong on this surface (implementation implementation.2)')
            if not isinstance(opt.default, _JSON_SAFE):
                raise KernelOptionError(f'kernel_options[{fam}]: default for {key!r} is not JSON-serialisable; the surface rides the model recipe (implementation implementation.2)')

def schema(family: str) -> dict:
    try:
        return _SCHEMAS[family]
    except KeyError:
        raise KernelOptionError(f"kernel_options: unknown family {family!r}; supported: {', '.join(FAMILIES)}") from None

def defaults(family: str) -> dict:
    return {k: o.default for k, o in schema(family).items()}

def validate(family: str, options: Any, who: str | None=None) -> dict:
    _ensure_checked()
    who = who or f'kernel_options[{family!r}]'
    sch = schema(family)
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise KernelOptionError(f'{who}: must be a dict (plain mapping), got {type(options).__name__}')
    supported = ', '.join(sorted(sch)) or '(none -- this family supports no kernel options in this version; an empty set is a real answer, see implementation implementation.2)'
    out: dict = {}
    for key in sorted(options):
        if not isinstance(key, str):
            raise KernelOptionError(f'{who}: keys must be strings, got {key!r}')
        opt = sch.get(key)
        if opt is None:
            raise KernelOptionError(f'{who}: unknown kernel option {key!r}; supported keys for family {family!r}: {supported}')
        value = options[key]
        if not isinstance(value, _JSON_SAFE):
            raise KernelOptionError(f'{who}: kernel option {key!r} value {value!r} is not JSON-serialisable; the surface rides the model recipe')
        if not opt.accepts(value):
            raise KernelOptionError(f'{who}: kernel option {key!r} does not support value {value!r}; allowed: {opt.allowed()}')
        out[key] = value
    return out

def family_of_config(cfg) -> str | None:
    for klass in type(cfg).__mro__:
        fam = _CONFIG_CLASS_FAMILY.get(klass.__name__)
        if fam is not None:
            return fam
    return None
_CONFIG_CLASS_FAMILY: dict[str, str] = {'SSDMimoConfig': 'ssd_mimo', 'HybridMambaConfig': 'hybrid_mamba', 'SSMCoreConfig': 'ssd_mimo'}

def validate_for_config(cfg, options: Any) -> dict:
    fam = family_of_config(cfg)
    name = type(cfg).__name__
    if fam is None:
        if options:
            raise KernelOptionError(f'{name}: kernel_options given but {name!r} is not a registered kernel-option family; register it in src/orbit_ssm/kernel_options.py')
        return {}
    return validate(fam, options, who=f'{name}.kernel_options')

def resolve(family: str, options: Any=None) -> dict:
    opts = validate(family, options)
    out = {}
    for key, opt in schema(family).items():
        value, _ = _effective(family, key, opt, opts)
        out[key] = value
    return out
_LEGACY_ENV = {('hybrid_mamba', 'BWD_GATE_BIAS_PARTIAL'): 'MH_BWD_GATE_BIAS_PARTIAL'}

def _env_value(family: str, opt: _Opt) -> str | None:
    if opt.env is None:
        return None
    prefix = _ENV_PREFIX.get(family, '')
    v = os.environ.get(prefix + opt.env)
    if v is None and prefix == 'SSD_MIMO_':
        v = os.environ.get('UNIFIED_SSD_' + opt.env)
    if (v is None or not v.strip()) and (family, opt.env) in _LEGACY_ENV:
        v = os.environ.get(_LEGACY_ENV[family, opt.env])
    if v is None or v.strip() == '':
        return None
    return v

def _effective(family: str, key: str, opt: _Opt, opts: dict):
    if key in opts:
        return (opts[key], 'config')
    raw = _env_value(family, opt)
    if raw is not None:
        return (_coerce(opt, raw), 'env')
    return (opt.default, 'default')

def _coerce(opt: _Opt, raw: str):
    raw = raw.strip()
    if isinstance(opt.default, bool):
        return raw.lower() in ('1', 'true', 'yes', 'on')
    if opt.values is not None and True in opt.values and (False in opt.values):
        low = raw.lower()
        if low in ('1', 'true', 'yes', 'on'):
            return True
        if low in ('0', 'false', 'no', 'off'):
            return False
    if isinstance(opt.default, str) and opt.values is not None:
        return raw
    try:
        return int(raw)
    except ValueError:
        return raw

def non_default(family: str, options: Any=None) -> dict:
    values, _ = record(family, options)
    return values

def record(family: str, options: Any=None) -> tuple[dict, dict]:
    if options is None:
        options = current(family)
    opts = options if isinstance(options, dict) else {}
    values: dict = {}
    sources: dict = {}
    for key, opt in schema(family).items():
        value, src = _effective(family, key, opt, opts)
        if src != 'default' and value != opt.default:
            values[key] = value
            sources[key] = src
    return (values, sources)
_ACTIVE = threading.local()

def _stacks() -> dict:
    st = getattr(_ACTIVE, 'stacks', None)
    if st is None:
        st = {}
        _ACTIVE.stacks = st
    return st

def push(family: str, options: Any) -> tuple:
    opts = validate(family, options) if options else {}
    stack = _stacks().setdefault(family, [])
    stack.append(opts)
    return (family, len(stack))

def pop(token: tuple) -> None:
    stack = _stacks().get(token[0])
    if stack:
        stack.pop()

def current(family: str) -> dict:
    st = getattr(_ACTIVE, 'stacks', None)
    if not st:
        return {}
    stack = st.get(family)
    return stack[-1] if stack else {}

def env_override(family: str, env_suffix: str) -> str | None:
    opts = current(family)
    if not opts:
        return None
    for key, opt in schema(family).items():
        if opt.env == env_suffix and key in opts:
            return opt.to_env(opts[key])
    return None

def scoped_kwarg(family: str, kwarg: str='kernel_options'):

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

def scoped_ctx(family: str, attr: str='_kernel_options'):

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
