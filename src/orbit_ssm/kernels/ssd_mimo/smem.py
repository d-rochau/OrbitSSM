
import json
import os
import tempfile
import warnings
from pathlib import Path
import torch
from .fused_fwd_kernel import _env

class SsdSmemOverflow(RuntimeError):

    def __init__(self, required: int, limit: int, N: int, T: int):
        self.required = required
        self.limit = limit
        self.N = N
        self.T = T
        super().__init__(f'fused SSD-MIMO plan needs {required} B of shared memory per CTA at N={N}, chunk T={T}; the device budget is {limit} B')
_LIMIT_CACHE: dict = {}

def device_smem_limit(device=None) -> int:
    override = _env('SMEM_LIMIT', '')
    if override:
        return int(override)
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    if idx not in _LIMIT_CACHE:
        props = torch.cuda.get_device_properties(idx)
        _LIMIT_CACHE[idx] = int(getattr(props, 'shared_memory_per_block_optin', 0) or props.shared_memory_per_block)
    return _LIMIT_CACHE[idx]

def smem_clamp_enabled() -> bool:
    return _env('SMEM_CLAMP', '1') == '1'

def chunk_ladder(chunk_size: int) -> list[int]:
    floor = int(_env('SMEM_FLOOR', '32'))
    if chunk_size & chunk_size - 1:
        return [chunk_size]
    out = []
    t = chunk_size
    while t >= floor:
        out.append(t)
        t //= 2
    return out or [chunk_size]
_WARNED: set = set()
_CLAMP_LOG: list = []
_CLAMP_LOG_MAX = 32

def clamp_log() -> list:
    return list(_CLAMP_LOG)

def _note(entry, key, message) -> None:
    _CLAMP_LOG.append(entry)
    del _CLAMP_LOG[:-_CLAMP_LOG_MAX]
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=4)

def note_chunk_clamped(N: int, requested: int, used: int, required: int, limit: int) -> None:
    _note(('t_halve', N, requested, used, (required, limit)), ('t_halve', N, requested, used), f'SSD-MIMO: chunk_size {requested} does not fit in shared memory at N={N} (the fused scan plan needs {required} B/CTA, device budget {limit} B) — running at chunk_size={used} instead.  This changes only the chunkwise associativity of the scan, not its math; set SSD_MIMO_SMEM_CLAMP=0 to get the launch failure back.')

def note_n_split(N: int, requested: int, tiles, required: int, limit: int) -> None:
    tiles = tuple(tiles)
    _note(('n_split', N, requested, tiles, (required, limit)), ('n_split', N, requested, tiles), f"SSD-MIMO: chunk_size {requested} does not fit in shared memory at N={N} (the fused scan plan needs {required} B/CTA, device budget {limit} B) — keeping the chunk and running the state axis as the independent subsystems {tiles} instead (measured faster than halving the chunk at this cell).  The decay is shared across N, so this is the same function: the tiles' outputs sum and their final states concatenate.  Set SSD_MIMO_SMEM_STRATEGY=t_halve to pin the chunk-halving route.")
T_HALVE = 't_halve'
N_SPLIT = 'n_split'
_STRATEGY: dict = {}
_STRATEGY_TIMINGS: dict = {}
_STRATEGY_LOG: list = []
_STRATEGY_LOG_MAX = 32
_DISK_LOADED = False
_DISK_WARNED = False

def strategy_key(N: int, T: int, dtype, precision: str, mode: str, device=None) -> str:
    try:
        name = torch.cuda.get_device_name(device)
    except Exception:
        name = 'unknown'
    dt = str(dtype).replace('torch.', '')
    return f'{name}|N{N}|T{T}|{dt}|{precision}|{mode}'

def strategy_cache_path():
    override = _env('SMEM_STRATEGY_CACHE', None)
    if override is not None:
        return Path(override) if override else None
    root = os.environ.get('TRITON_CACHE_DIR')
    if not root:
        root = os.path.join(os.path.expanduser('~'), '.triton', 'cache')
    return Path(root) / 'ssd_mimo_smem_strategy.json'

def _disk_read() -> dict:
    path = strategy_cache_path()
    if path is None:
        return {}
    try:
        with open(path, 'r') as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _disk_write(key: str, winner: str) -> None:
    global _DISK_WARNED
    path = strategy_cache_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _disk_read()
        data[key] = winner
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as fh:
                json.dump(data, fh, indent=0, sort_keys=True)
            os.replace(tmp, path)
        except Exception:
            os.unlink(tmp)
            raise
    except Exception as exc:
        if not _DISK_WARNED:
            _DISK_WARNED = True
            warnings.warn(f'SSD-MIMO: could not persist the shared-memory strategy verdict to {path} ({exc}); it stays in-process only.', RuntimeWarning, stacklevel=3)

def _disk_load_once() -> None:
    global _DISK_LOADED
    if _DISK_LOADED:
        return
    _DISK_LOADED = True
    for k, v in _disk_read().items():
        if k.startswith('chunk:'):
            try:
                _CHUNK_CACHE.setdefault(k, int(v))
            except (TypeError, ValueError):
                continue
        elif k.startswith(_FWD_CFG_PREFIX):
            cfg = _fwd_cfg_parse(v)
            if cfg is not None:
                _FWD_CFG_CACHE.setdefault(k, cfg)
        else:
            _STRATEGY.setdefault(k, v)

def lookup_strategy(key: str):
    if key in _STRATEGY:
        return _STRATEGY[key]
    _disk_load_once()
    return _STRATEGY.get(key)

def strategy_log() -> list:
    return list(_STRATEGY_LOG)

def _log_strategy(key, winner, timings, source) -> None:
    _STRATEGY_LOG.append((key, winner, dict(timings), source))
    del _STRATEGY_LOG[:-_STRATEGY_LOG_MAX]

def measuring_possible() -> bool:
    from .fused_fwd_kernel import autotune_off
    if autotune_off():
        return False
    if _env('SMEM_MEASURE', '1') != '1':
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        return False
    return True

def _time_call(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters

def preferred_strategy(key: str, available):
    available = list(available)
    if not available:
        return None
    forced = _env('SMEM_STRATEGY', 'measure')
    if forced in (T_HALVE, N_SPLIT):
        winner = forced if forced in available else available[0]
        _log_strategy(key, winner, {}, 'pinned')
        return winner
    if len(available) == 1:
        _log_strategy(key, available[0], {}, 'only-candidate')
        return available[0]
    cached = lookup_strategy(key)
    if cached in available:
        _log_strategy(key, cached, _STRATEGY_TIMINGS.get(key, {}), 'cached')
        return cached
    return None

def resolve_strategy(key: str, candidates: dict, fallback: str=T_HALVE):
    if not candidates:
        raise ValueError('resolve_strategy: no candidate plans')

    def _fall():
        return fallback if fallback in candidates else next(iter(candidates))
    pref = preferred_strategy(key, candidates)
    if pref is not None:
        return pref
    if not measuring_possible():
        winner = _fall()
        _log_strategy(key, winner, {}, 'unmeasurable')
        return winner
    iters = max(1, int(_env('SMEM_ITERS', '3')))
    timings = {}
    for name, trial in candidates.items():
        try:
            timings[name] = _time_call(trial, iters)
        except Exception:
            continue
    if not timings:
        winner = _fall()
        _log_strategy(key, winner, {}, 'trials-failed')
        return winner
    winner = min(timings, key=timings.get)
    _STRATEGY[key] = winner
    _STRATEGY_TIMINGS[key] = timings
    _log_strategy(key, winner, timings, 'measured')
    _disk_write(key, winner)
    return winner
_CHUNK_CACHE: dict = {}
_CHUNK_TIMINGS: dict = {}
_CHUNK_LOG: list = []
_CHUNK_LOG_MAX = 32
_CHUNK_T_CANDIDATES = (64, 32, 16)
_CHUNK_T_CEILING = 64

def chunk_admissible(ceiling: int=_CHUNK_T_CEILING, floor: int=16) -> list[int]:
    return [T for T in _CHUNK_T_CANDIDATES if floor <= T <= ceiling]

def chunk_cache_key(R: int, O: int, N: int, dtype, precision: str, mode: str, shape, device=None) -> str:
    try:
        name = torch.cuda.get_device_name(device)
    except Exception:
        name = 'unknown'
    dt = str(dtype).replace('torch.', '')
    shp = 'x'.join((str(s) for s in shape))
    return f'chunk:{name}|R{R}|O{O}|N{N}|{dt}|{precision}|{mode}|{shp}'

def chunk_cache_lookup(key: str):
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]
    _disk_load_once()
    v = _CHUNK_CACHE.get(key)
    if v is None and key.startswith('chunk:cp:'):
        from .calibration_chunkpar import lookup
        v = lookup(key)
    return v

def chunk_cache_store(key: str, T: int, timings: dict | None=None) -> None:
    _CHUNK_CACHE[key] = T
    if timings:
        _CHUNK_TIMINGS[key] = dict(timings)
    _CHUNK_LOG.append((key, T, dict(timings or {})))
    del _CHUNK_LOG[:-_CHUNK_LOG_MAX]
    _disk_write(key, T)

def chunk_measure_log() -> list:
    return list(_CHUNK_LOG)
_FWD_CFG_PREFIX = 'fwdcfg:'
_FWD_CFG_CACHE: dict = {}
_FWD_CFG_TIMINGS: dict = {}
_FWD_CFG_LOG: list = []
_FWD_CFG_LOG_MAX = 32

def _fwd_cfg_parse(v):
    try:
        w, st, mn = (int(p) for p in str(v).split(','))
    except (TypeError, ValueError):
        return None
    if w <= 0 or st <= 0 or mn < 0:
        return None
    return (w, st, mn)

def _fwd_cfg_str(cfg) -> str:
    return '%d,%d,%d' % (int(cfg[0]), int(cfg[1]), int(cfg[2] or 0))

def fwd_cfg_key(cell: str, device=None) -> str:
    try:
        name = torch.cuda.get_device_name(device)
    except Exception:
        name = 'unknown'
    return f'{_FWD_CFG_PREFIX}{name}|{cell}'

def fwd_cfg_lookup(key: str):
    if key in _FWD_CFG_CACHE:
        return _FWD_CFG_CACHE[key]
    _disk_load_once()
    return _FWD_CFG_CACHE.get(key)

def fwd_cfg_store(key: str, cfg, timings: dict | None=None) -> None:
    cfg = (int(cfg[0]), int(cfg[1]), int(cfg[2] or 0))
    _FWD_CFG_CACHE[key] = cfg
    if timings:
        _FWD_CFG_TIMINGS[key] = dict(timings)
    _FWD_CFG_LOG.append((key, cfg, dict(timings or {})))
    del _FWD_CFG_LOG[:-_FWD_CFG_LOG_MAX]
    _disk_write(key, _fwd_cfg_str(cfg))

def fwd_cfg_log() -> list:
    return list(_FWD_CFG_LOG)

def fwd_cfg_mode() -> str:
    from .fused_fwd_kernel import autotune_off
    if autotune_off():
        return 'off'
    v = _env('FWD_CFG', 'auto').strip().lower()
    return v if v in ('auto', 'measure', 'off') else 'auto'
