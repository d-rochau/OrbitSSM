
from __future__ import annotations
import copy
from dataclasses import dataclass, fields
from numbers import Integral, Real
FAMILIES = ('ssd_mimo', 'hybrid_mamba')
HEAD_DYNAMIC = 'head_dynamic'
HEAD_STATIC = 'head_static'
SSD_DECAYS = (HEAD_DYNAMIC, HEAD_STATIC)
HM_PER_COORD = ('A', 'dyn', 'mu')
HM_FLAT = (HEAD_DYNAMIC, HEAD_STATIC)
HM_DECAYS = HM_PER_COORD + HM_FLAT
HM_COMPLEX_OK = HM_PER_COORD + (HEAD_DYNAMIC,)
REMOVED_DECAYS = {'m3': HEAD_DYNAMIC, 'm2': HEAD_STATIC, 'const': HEAD_DYNAMIC}
DROPPED_DECAYS = {'fixed': "the HM time-constant decay law 'fixed' (log a = -exp(fixed_A_log), no step factor) is dropped; use 'head_static' (per-head static rate, Delta-scaled) or 'head_dynamic'"}
BLOCK_GATE_DEFAULT = {False: True, True: False}
BLOCK_KEYS = ('rows', 'frac', 'decay', 'complex', 'gate')
HM_MAX_BLOCKS = 8
SSD_MAX_BLOCKS = 2
ROW_EXT = {}
BIAS_GRANS = ('head_bc', 'shared', 'none')
SSD_BIAS_INITS = ('auto', 'ones', 'spread')
HM_GATE_ACTS = ('sigmoid', 'exp', 'tanh', 'softplus', 'raw', 'one_plus', 'one_plus_tanh')
HM_GATE_POS = ('pre', 'post')
ROTATION_DEFAULTS = {'theta_per_head': False, 'learn_theta_freq': True, 'use_theta_h': True, 'omega_layout': 'pair', 'theta_init': 'kaiming'}
OMEGA_LAYOUTS = ('pair', 'head', 'head_pair')
THETA_INITS = ('kaiming', 'small', 'zero')
_ROTATION_CHOICES = {'omega_layout': OMEGA_LAYOUTS, 'theta_init': THETA_INITS}
_ROTATION_BOOLS = ('theta_per_head', 'learn_theta_freq', 'use_theta_h')
MAPS_CORE = ('b_norm', 'c_norm', 'bias', 'gate_b', 'gate_c')
MAPS_EXT = {'ssd_mimo': ('bias_init',), 'hybrid_mamba': ('gate_act', 'gate_bias', 'gate_pos')}
FLAT = {'theta_per_head': ('rotation', 'theta_per_head'), 'learn_theta_freq': ('rotation', 'learn_theta_freq'), 'use_theta_h': ('rotation', 'use_theta_h'), 'b_norm': ('maps', 'b_norm'), 'c_norm': ('maps', 'c_norm'), 'bias_gran': ('maps', 'bias'), 'use_g_b': ('maps', 'gate_b'), 'use_g_c': ('maps', 'gate_c'), 'bc_bias_init': ('maps', 'bias_init'), 'gate_act': ('maps', 'gate_act'), 'gate_bias': ('maps', 'gate_bias'), 'gate_pos': ('maps', 'gate_pos')}
FLAT_KWARGS = tuple(FLAT) + ('bc_norm',)
SHORTHANDS = ('bc_norm',)
REMOVED = {'rot_mode': 'rot_mode was removed (one rotation law): a rotation is a block with complex=True in `rows`; its settings live in `rotation`', 'rope_fraction': "rope_fraction was removed: the rotated fraction is the size of the complex block in `rows`, e.g. rows=[{'frac': 0.5, 'decay': 'head_dynamic', 'complex': True}, {'frac': 0.5, 'decay': 'head_dynamic', 'complex': False}]", 'a_dyn': "a_dyn was removed: the SSD decay law is rows[i]['decay'] ('head_dynamic' = selected rate, was a_dyn=True; 'head_static' = static rate, was a_dyn=False)", 'decay': "the core field `decay` ('dyn'/'fix') was removed: use rows[i]['decay'] ('head_dynamic'/'head_static')", 'fracs': "fracs was renamed: pass the same blocks as `rows` ({'rows'|'frac', 'decay', 'complex', 'gate'})", 'gates': 'gates was removed: use use_g_b / use_g_c (maps.gate_b / maps.gate_c)', 'bc_bias': "bc_bias was removed: use bias_gran / maps.bias ('head_bc' | 'shared' | 'none')", 'bc_bias_heads': "bc_bias_heads was removed: the bias granularity is maps.bias ('head_bc' = per head, 'shared' = one); intermediate head groups are not in the spec", 'rot': "block key 'rot' was renamed 'complex'", 'rotate': "block key 'rotate' was renamed 'complex'", 'use_rot_bias': 'use_rot_bias is retired (biases act before the rotation)', 'rot_theta_dyn': 'rot_theta_dyn is retired (the single m3 rotation law)'}
MIXER_BC_NORM_REMOVED = 'bc_norm was removed from the SSD-MIMO / HybridMamba mixer and LM surface: pass b_norm / c_norm / bias_gran (or maps={...}, or the core shorthand bc_norm) inside ssm_kwargs'
_REMOVED_ROTATION_KEYS = {'rot_mode', 'rope_fraction', 'mode', 'pair_layout'}

class ConfigSpecError(ValueError):
    pass

@dataclass(frozen=True)
class RowBlock:
    rows: int | None = None
    frac: float | None = None
    decay: str | None = None
    complex: bool = False
    gate: bool | None = None

    def to_dict(self) -> dict:
        d = {'decay': self.decay, 'complex': self.complex}
        if self.gate is not None:
            d['gate'] = self.gate
        if self.rows is not None:
            d['rows'] = self.rows
        if self.frac is not None:
            d['frac'] = self.frac
        return d

@dataclass(frozen=True)
class RotationSpec:
    theta_per_head: bool | None = None
    learn_theta_freq: bool | None = None
    use_theta_h: bool | None = None
    omega_layout: str | None = None
    theta_init: str | None = None

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

@dataclass(frozen=True)
class MapSpec:
    b_norm: bool | None = None
    c_norm: bool | None = None
    bias: str | None = None
    gate_b: bool | None = None
    gate_c: bool | None = None
    bias_init: str | None = None
    gate_act: str | None = None
    gate_bias: bool | None = None
    gate_pos: str | None = None

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

def _bool(v, name):
    if not isinstance(v, bool):
        raise ConfigSpecError(f'{name} must be a bool, got {v!r}')
    return v

def _rot_value(key, v, name):
    if key in _ROTATION_CHOICES:
        if v not in _ROTATION_CHOICES[key]:
            raise ConfigSpecError(f'{name} must be one of {_ROTATION_CHOICES[key]}, got {v!r}')
        return v
    return _bool(v, name)

def _as_dict(value, typ, name):
    if value is None:
        return {}
    if isinstance(value, typ):
        return value.to_dict()
    if isinstance(value, dict):
        return copy.deepcopy(value)
    raise ConfigSpecError(f'{name} must be a {typ.__name__} or a dict, got {type(value).__name__}')

def reject_removed(kwargs, who: str, extra: dict | None=None) -> None:
    for key in kwargs:
        if key in REMOVED:
            raise ConfigSpecError(f'{who}: {REMOVED[key]}')
        if extra and key in extra:
            raise ConfigSpecError(f'{who}: {extra[key]}')

def flat_kwargs(values: dict) -> dict:
    return {k: values[k] for k in FLAT_KWARGS if k in values and values[k] is not None}

def _removed(key, where):
    if key in REMOVED:
        raise ConfigSpecError(f'{where}: {REMOVED[key]}')

def _merge(target: dict, origin: dict, key, value, spelling):
    if key in target and target[key] != value:
        raise ConfigSpecError(f'conflicting spellings for {key!r}: {origin[key]}={target[key]!r} vs {spelling}={value!r}')
    target.setdefault(key, value)
    origin.setdefault(key, spelling)

def _resolve_rows(family, rows, d_state):
    if rows is None:
        if family == 'hybrid_mamba':
            raise ConfigSpecError(f'{family}: `rows` is required')
        return [{'rows': d_state, 'decay': HEAD_DYNAMIC, 'complex': False, 'gate': BLOCK_GATE_DEFAULT[False]}]
    if isinstance(rows, (RowBlock, dict)):
        raise ConfigSpecError('rows must be a list of blocks, not one block')
    blocks = []
    for i, b in enumerate(rows):
        b = _as_dict(b, RowBlock, f'rows[{i}]')
        extra = set(b) - set(BLOCK_KEYS) - set(ROW_EXT.get(family, ()))
        for k in extra:
            _removed(k, f'rows[{i}]')
        if extra:
            raise ConfigSpecError(f'rows[{i}]: unknown keys {sorted(extra)}')
        if b.get('rows') is not None and b.get('frac') is not None:
            raise ConfigSpecError(f"rows[{i}]: give 'rows' or 'frac', not both")
        if b.get('rows') is not None:
            n = b['rows']
            if isinstance(n, bool) or not isinstance(n, Integral):
                raise ConfigSpecError(f"rows[{i}]['rows'] must be an int, got {n!r}")
            n = int(n)
        elif b.get('frac') is not None:
            f = b['frac']
            if isinstance(f, bool) or not isinstance(f, Real) or (not 0 < f <= 1):
                raise ConfigSpecError(f"rows[{i}]['frac'] must be in (0, 1], got {f!r}")
            x = float(f) * d_state
            n = int(round(x))
            if abs(x - n) > 1e-09:
                raise ConfigSpecError(f"rows[{i}]['frac']={f!r} gives {x:g} rows of d_state={d_state}; frac*d_state must be an integer (or give 'rows' explicitly)")
        else:
            raise ConfigSpecError(f"rows[{i}]: needs 'rows' or 'frac'")
        if n <= 0:
            raise ConfigSpecError(f'rows[{i}]: resolved to {n} rows')
        if b.get('decay') is None:
            raise ConfigSpecError(f"rows[{i}]: 'decay' is required")
        if isinstance(b['decay'], str) and b['decay'] in REMOVED_DECAYS:
            raise ConfigSpecError(f"rows[{i}]: decay law {b['decay']!r} was renamed by meaning: use {REMOVED_DECAYS[b['decay']]!r} (head_dynamic = per-head selected rate, head_static = per-head static rate)")
        if isinstance(b['decay'], str) and b['decay'] in DROPPED_DECAYS:
            raise ConfigSpecError(f"rows[{i}]: decay law {b['decay']!r} was removed -- {DROPPED_DECAYS[b['decay']]}")
        cx = _bool(b.get('complex', False), f"rows[{i}]['complex']")
        if cx and n % 2:
            raise ConfigSpecError(f'rows[{i}]: a complex block holds (re, im) pairs; {n} rows is odd')
        gate = b.get('gate')
        gate = BLOCK_GATE_DEFAULT[cx] if gate is None else _bool(gate, f"rows[{i}]['gate']")
        block = {'rows': n, 'decay': b['decay'], 'complex': cx, 'gate': gate}
        for key in ROW_EXT.get(family, ()):
            block[key] = _bool(b.get(key, False), f'rows[{i}][{key!r}]')
        blocks.append(block)
    if not blocks:
        raise ConfigSpecError('rows must be a non-empty list of blocks')
    total = sum((b['rows'] for b in blocks))
    if total != d_state:
        raise ConfigSpecError(f'rows sum to {total}, need exactly d_state={d_state}')
    if family == 'ssd_mimo':
        if len(blocks) > SSD_MAX_BLOCKS:
            raise ConfigSpecError('ssd_mimo: at most two row blocks (one complex part, one real part)')
        laws = {b['decay'] for b in blocks}
        if len(laws) > 1:
            raise ConfigSpecError(f'ssd_mimo: one decay law for the whole state, got {sorted(laws)}')
        for i, b in enumerate(blocks):
            if b['decay'] not in SSD_DECAYS:
                raise ConfigSpecError(f"rows[{i}]: ssd_mimo decay must be one of {SSD_DECAYS}, got {b['decay']!r}")
        if len(blocks) == 2:
            if blocks[0]['complex'] == blocks[1]['complex']:
                raise ConfigSpecError('ssd_mimo: two blocks must be one complex and one real part')
            if not blocks[0]['complex']:
                raise ConfigSpecError('ssd_mimo: the complex block comes first (the kernel rotates the leading pairs)')
    else:
        if len(blocks) > HM_MAX_BLOCKS:
            raise ConfigSpecError(f'hybrid_mamba: at most {HM_MAX_BLOCKS} row blocks')
        for i, b in enumerate(blocks):
            if b['decay'] not in HM_DECAYS:
                raise ConfigSpecError(f"rows[{i}]: hybrid_mamba decay must be one of {HM_DECAYS}, got {b['decay']!r}")
            if b['complex'] and b['decay'] not in HM_COMPLEX_OK:
                raise ConfigSpecError(f"rows[{i}]: complex=True is served on {HM_COMPLEX_OK} only, got {b['decay']!r}")
    return blocks

def resolve(family: str, *, d_state: int, R: int=1, O: int | None=None, rows=None, rotation=None, maps=None, **flat) -> dict:
    if family not in FAMILIES:
        raise ConfigSpecError(f'family must be one of {FAMILIES}, got {family!r}')
    if isinstance(d_state, bool) or not isinstance(d_state, Integral) or d_state <= 0:
        raise ConfigSpecError(f'd_state must be a positive int, got {d_state!r}')
    O = R if O is None else O
    for k in flat:
        _removed(k, family)
    unknown = set(flat) - set(FLAT) - set(SHORTHANDS)
    if unknown:
        raise ConfigSpecError(f'{family}: unknown config keys {sorted(unknown)}')
    blocks = _resolve_rows(family, rows, d_state)
    has_complex = any((b['complex'] for b in blocks))
    rot, rot_from = ({}, {})
    rot_in = _as_dict(rotation, RotationSpec, 'rotation')
    for k, v in rot_in.items():
        if k in _REMOVED_ROTATION_KEYS:
            raise ConfigSpecError(f"rotation[{k!r}]: {REMOVED.get(k, k + ' is not a rotation field')}")
        if k not in ROTATION_DEFAULTS:
            raise ConfigSpecError(f'rotation: unknown key {k!r} (per-block rotation settings are not supported; fields: {sorted(ROTATION_DEFAULTS)})')
        _merge(rot, rot_from, k, _rot_value(k, v, f'rotation[{k!r}]'), f'rotation[{k!r}]')
    mp, mp_from = ({}, {})
    maps_in = _as_dict(maps, MapSpec, 'maps')
    shorthand = {}
    if 'bc_norm' in maps_in:
        shorthand['bc_norm'] = ("maps['bc_norm']", _bool(maps_in.pop('bc_norm'), "maps['bc_norm']"))
    allowed_maps = set(MAPS_CORE) | set(MAPS_EXT[family])
    for k, v in maps_in.items():
        _removed(k, 'maps')
        if k not in allowed_maps:
            other = [f for f, ext in MAPS_EXT.items() if k in ext]
            hint = f' (a {other[0]}-only field)' if other else ''
            raise ConfigSpecError(f'maps: unknown key {k!r} for {family}{hint}')
        _merge(mp, mp_from, k, v, f'maps[{k!r}]')
    for name, v in flat.items():
        if name in SHORTHANDS:
            v = _bool(v, name)
            if 'bc_norm' in shorthand and shorthand['bc_norm'][1] != v:
                raise ConfigSpecError(f"conflicting spellings for 'bc_norm': {shorthand['bc_norm'][0]}={shorthand['bc_norm'][1]!r} vs bc_norm={v!r}")
            shorthand.setdefault('bc_norm', ('bc_norm', v))
            continue
        section, key = FLAT[name]
        if section == 'maps' and key not in allowed_maps:
            other = [f for f, ext in MAPS_EXT.items() if key in ext]
            raise ConfigSpecError(f'{name} is a {other[0]}-only field')
        if section == 'rotation':
            _merge(rot, rot_from, key, _rot_value(key, v, name), name)
        else:
            _merge(mp, mp_from, key, v, name)
    if has_complex:
        defaults = ROTATION_DEFAULTS
        rotation_out = {k: rot.get(k, d) for k, d in defaults.items()}
        layout = rotation_out['omega_layout']
        if layout == 'head':
            if rot.get('theta_per_head') is False:
                raise ConfigSpecError(f"{rot_from['theta_per_head']}=False conflicts with {rot_from['omega_layout']}='head': one eigenvalue per head has one angle feature per head (theta_per_head=True)")
            rotation_out['theta_per_head'] = True
        if layout == 'head_pair' and (not rotation_out['learn_theta_freq']):
            raise ConfigSpecError(f"{rot_from['omega_layout']}='head_pair' needs learn_theta_freq=True (the fixed pi has no layout)")
    else:
        on = sorted((k for k, v in rot.items() if (v if k in _ROTATION_BOOLS else v != ROTATION_DEFAULTS[k])))
        if on:
            raise ConfigSpecError(f'rotation fields {on} requested, but `rows` has no complex block (add one, or drop them)')
        rotation_out = None
    bc = shorthand.get('bc_norm', (None, None))[1]
    if bc is False and mp.get('bias', 'none') != 'none':
        raise ConfigSpecError(f"conflicting spellings: {shorthand['bc_norm'][0]}=False removes the B/C biases, but {mp_from['bias']}={mp['bias']!r}")
    for k in ('b_norm', 'c_norm', 'gate_b', 'gate_c', 'gate_bias'):
        if k in mp:
            _bool(mp[k], mp_from[k])
    grans = BIAS_GRANS
    if 'bias' in mp and mp['bias'] not in grans:
        raise ConfigSpecError(f"{mp_from['bias']} must be one of {grans}, got {mp['bias']!r}")
    norm_default = True if bc is None else bc
    out = {'b_norm': mp.get('b_norm', norm_default), 'c_norm': mp.get('c_norm', norm_default), 'bias': mp.get('bias', 'none' if bc is False else 'head_bc')}
    any_gated = any((b['gate'] for b in blocks))
    if family == 'ssd_mimo':
        out['gate_b'] = mp.get('gate_b', R > 1 and any_gated)
        out['gate_c'] = mp.get('gate_c', O > 1 and any_gated)
        init = mp.get('bias_init', 'auto')
        if out['bias'] == 'none':
            if init not in ('auto', 'none'):
                raise ConfigSpecError(f"{mp_from['bias_init']}={init!r} given, but maps.bias='none'")
            init = 'none'
        elif init not in SSD_BIAS_INITS:
            raise ConfigSpecError(f"{mp_from['bias_init']} must be one of {SSD_BIAS_INITS}, got {init!r}")
        elif init == 'auto':
            init = 'ones' if R == 1 or out['bias'] == 'shared' else 'spread'
        out['bias_init'] = init
        if has_complex and out['bias'] != 'head_bc':
            raise ConfigSpecError(f"ssd_mimo: a complex block needs maps.bias='head_bc' (the per-kernel-head biases act before the rotation), got {out['bias']!r}")
    else:
        out['gate_b'] = mp.get('gate_b', False)
        out['gate_c'] = mp.get('gate_c', False)
        act = mp.get('gate_act', 'one_plus')
        if act not in HM_GATE_ACTS:
            raise ConfigSpecError(f'gate_act must be one of {HM_GATE_ACTS}, got {act!r}')
        out['gate_act'] = act
        out['gate_bias'] = mp.get('gate_bias', False)
        if out['gate_bias'] and (not (out['gate_b'] or out['gate_c'])):
            raise ConfigSpecError('gate_bias=True needs gate_b or gate_c')
        pos = mp.get('gate_pos', 'pre')
        if pos not in HM_GATE_POS:
            raise ConfigSpecError(f'gate_pos must be one of {HM_GATE_POS}, got {pos!r}')
        out['gate_pos'] = pos
    for k in ('gate_b', 'gate_c'):
        if out[k] and (not any_gated):
            raise ConfigSpecError(f"{mp_from.get(k, k)}=True but no row block has gate=True (the per-block default is {BLOCK_GATE_DEFAULT[False]} for real and {BLOCK_GATE_DEFAULT[True]} for complex blocks): set rows[i]['gate']=True on the block(s) to gate, or drop the gate")
    return {'rows': blocks, 'rotation': rotation_out, 'maps': out}

def load_saved(family: str, spec: dict, *, d_state: int, R: int=1, O: int | None=None) -> dict:
    if set(spec) != {'rows', 'rotation', 'maps'}:
        extra = set(spec) - {'rows', 'rotation', 'maps'}
        for k in extra:
            _removed(k, f'saved {family} config')
        raise ConfigSpecError(f"saved spec fields must be exactly ['maps', 'rotation', 'rows'], got {sorted(spec)}")
    again = resolve(family, d_state=d_state, R=R, O=O, rows=spec['rows'], rotation=spec['rotation'], maps=spec['maps'])
    if again != spec:
        raise ConfigSpecError('saved spec is not canonical (resolve() changes it)')
    return again
