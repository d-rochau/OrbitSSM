
from __future__ import annotations
import os
import torch
import triton
_CACHE: dict = {}
_CARRY_LAUNCHERS: dict = {}

def carry_launcher(full_kernel, side):
    from ... import kernel_options as _ko
    if side not in ('fwd', 'bwd'):
        raise ValueError(f'unknown MH carry side: {side!r}')
    pin = _opt('CARRY_FWD_CONFIG' if side == 'fwd' else 'CARRY_BWD_CONFIG')
    mode = tier()
    if mode == 'full' and pin is None:
        return full_kernel
    key = (full_kernel, side, mode, pin)
    if key not in _CARRY_LAUNCHERS:
        if pin is not None:
            c = _ko.parse_mh_config(pin, _ko.MH_CARRY_CONFIG_KEYS)
            triples = ((c['BLOCK_C'], c['BLOCK_T'], c.get('NUM_WARPS', 4)),)
        elif side == 'fwd':
            triples = ((16, 128, 4),) if mode == 'pinned' else ((16, 64, 2), (16, 128, 2), (16, 128, 4))
        elif side == 'bwd':
            triples = ((32, 64, 4),) if mode == 'pinned' else ((32, 64, 4), (32, 128, 8), (16, 64, 4), (16, 128, 4), (16, 128, 8), (32, 32, 2), (16, 64, 8))
        configs = [triton.Config(dict(BLOCK_C=bc, BLOCK_T=bt), num_warps=w) for bc, bt, w in triples]

        def bench(fn, quantiles):
            times = triton.testing.do_bench(fn, warmup=3, rep=20, quantiles=quantiles)
            if times[0] > 4:
                times = triton.testing.do_bench(fn, warmup=1, rep=5 * times[0], quantiles=quantiles)
            return times
        _CARRY_LAUNCHERS[key] = triton.autotune(configs=configs, key=full_kernel.keys, do_bench=bench)(full_kernel.fn)
    return _CARRY_LAUNCHERS[key]

def _opt(env_suffix, legacy=None):
    from ... import kernel_options as _ko
    v = _ko.env_override('hybrid_mamba', env_suffix)
    if v is None:
        v = os.environ.get('HYBRID_MH_' + env_suffix)
        if (v is None or not v.strip()) and legacy:
            v = os.environ.get(legacy)
    if v is None or not v.strip():
        return None
    return v.strip()
_TIERS = {'0': 'pinned', 'off': 'pinned', 'false': 'pinned', 'no': 'pinned', 'pinned': 'pinned', '1': 'light', 'on': 'light', 'true': 'light', 'yes': 'light', 'auto': 'light', 'light': 'light', 'full': 'full'}

def tier():
    v = _opt('AUTOTUNE')
    if v is None:
        return 'light'
    t = _TIERS.get(v.lower())
    if t is None:
        raise ValueError(f'HYBRID_MH_AUTOTUNE / autotune={v!r}: expected pinned | light | full (legacy 0/1, off/auto)')
    return t

def enabled():
    return tier() != 'pinned'

def pinned_config(side):
    from ... import kernel_options as _ko
    v = _opt('FWD_CONFIG' if side == 'fwd' else 'BWD_CONFIG')
    if v is None:
        return None
    return _ko.parse_mh_config(v, _ko.MH_FWD_CONFIG_KEYS if side == 'fwd' else _ko.MH_BWD_CONFIG_KEYS)
_TABLE_FWD = {(120, True, 2, 16): dict(BLOCK_P=4, BLOCK_L=256, NUM_WARPS=1, PF=1), (120, False, 2, 16): dict(BLOCK_P=2, BLOCK_L=512, NUM_WARPS=1, PF=1)}
_TABLE_BWD = {(120, True, 2, 16): dict(TP_P=2, NS=4, NUM_WARPS=1), (120, False, 2, 16): dict(TP_P=4, NS=4, NUM_WARPS=1)}

def table_fwd(cc, rot, pmap, esz, P, N):
    e = _TABLE_FWD.get((cc, bool(rot), esz, P))
    return dict(e) if e else None

def table_bwd(cc, rot, esz, P):
    e = _TABLE_BWD.get((cc, bool(rot), esz, P))
    return dict(e) if e else None

def gate_bias_partial(P=None):
    v = _opt('BWD_GATE_BIAS_PARTIAL', legacy='MH_BWD_GATE_BIAS_PARTIAL')
    if v is None or v.lower() == 'auto':
        return False
    return v.lower() in ('1', 'true', 'yes', 'on')

def _bench(fn):
    t = triton.testing.do_bench(fn, warmup=5, rep=10)
    if t > 2.0:
        t = triton.testing.do_bench(fn, warmup=1, rep=5 * t, return_mode='median')
    return t
_INF = float('inf')

def _digest(out):
    vals = []
    for t in out if isinstance(out, (tuple, list)) else (out,):
        if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.numel() == 0:
            continue
        t = t.detach()
        vals.append(torch.stack((torch.linalg.vector_norm(t, ord=_INF, dtype=torch.float32), torch.linalg.vector_norm(t, dtype=torch.float32))))
    return torch.stack(vals) if vals else None

def _agrees(ref, got, rtol=0.02):
    if ref is None or got is None:
        return True
    if ref.shape != got.shape:
        return False
    return bool(torch.isfinite(got).all() and (got - ref).abs().le(rtol * ref.abs() + 1e-06).all())

def _all_finite(out):
    for t in out if isinstance(out, (tuple, list)) else (out,):
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel() and (not torch.isfinite(t).all()):
            return False
    return True

def _try(run, cfg, extreme=None):
    try:
        digest = _digest(run(cfg))
        if extreme is not None and (not _all_finite(extreme(cfg))):
            return (None, None)
        return (_bench(lambda: run(cfg)), digest)
    except Exception:
        return (None, None)
    finally:
        torch.cuda.empty_cache()

def choose(key, candidates, run):
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    best, best_ms = (None, float('inf'))
    for cfg in candidates:
        try:
            ms = _bench(lambda: run(cfg))
        except Exception:
            continue
        if ms < best_ms:
            best, best_ms = (cfg, ms)
    if best is None:
        best = candidates[0]
    _CACHE[key] = best
    return best

def _wins(ms, best_ms, cand_b, best_b, tol):
    if cand_b is None or cand_b == best_b:
        return ms < best_ms
    if cand_b < best_b:
        return ms <= best_ms * (1.0 + tol)
    return ms * (1.0 + max(tol, cand_b / best_b - 1.0)) < best_ms

def _hopeless(cand, best_b, bytes_of):
    if bytes_of is None or best_b is None:
        return False
    return bytes_of(cand) > 2 * best_b

def _persist_path():
    v = os.environ.get('HYBRID_MH_TUNE_CACHE')
    if v is not None:
        v = v.strip()
        return None if v.lower() in ('', '0', 'off', 'none') else v
    d = os.environ.get('TRITON_CACHE_DIR') or os.path.join(os.path.expanduser('~'), '.triton', 'cache')
    return os.path.join(d, 'hybrid_mh_tune.json')

def _persist_digest(key, tag):
    import hashlib
    dev = torch.cuda.current_device()
    ident = (repr(key), str(tag), _host_code_hash(), torch.cuda.get_device_name(dev), torch.cuda.get_device_capability(dev), triton.__version__, torch.__version__)
    return hashlib.sha1(repr(ident).encode()).hexdigest()
_HOST = {}

def _host_code_hash():
    h = _HOST.get('h')
    if h is None:
        import hashlib
        from . import bwd_gluon, common_gluon, fwd_gluon, layout, slp, ts_adjoint, ts_forward
        d = hashlib.sha1()
        for mod in (__import__(__name__, fromlist=['x']), fwd_gluon, bwd_gluon, common_gluon, layout, slp, ts_adjoint, ts_forward):
            with open(mod.__file__, 'rb') as fh:
                d.update(fh.read())
        h = _HOST['h'] = d.hexdigest()
    return h
_DISK = {'path': None, 'data': None}

def _persist_load(digest):
    path = _persist_path()
    if path is None:
        return None
    if _DISK['path'] != path:
        import json
        try:
            with open(path) as fh:
                _DISK['data'] = json.load(fh)
        except (OSError, ValueError):
            _DISK['data'] = {}
        _DISK['path'] = path
    hit = _DISK['data'].get(digest)
    return dict(hit) if isinstance(hit, dict) else None

def _persist_store(digest, cfg):
    import fcntl
    import json
    path = _persist_path()
    if path is None:
        return
    try:
        if any((not isinstance(v, (int, float, str, bool, type(None))) for v in cfg.values())):
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path + '.lock', 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                data = {}
            data[digest] = cfg
            tmp = f'{path}.{os.getpid()}.tmp'
            with open(tmp, 'w') as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
            _DISK['path'], _DISK['data'] = (path, data)
    except (OSError, TypeError, ValueError):
        pass

def cached(key):
    return _CACHE.get(key)

def choose_axes(key, base, axes, run, bytes_of=None, tol=0.03, grid=None, extreme=None, drop_keys=(), persist=None, pre=(), candidate_filter=None):
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    pkey = None
    if persist is not None:
        pkey = _persist_digest(key, persist)
        hit = _persist_load(pkey)
        if hit is not None:
            _CACHE[key] = hit
            return hit
    trials = {}

    def measure(cfg, probe=None):
        identity = (tuple(sorted(cfg.items())), probe is not None)
        if identity not in trials:
            trials[identity] = _try(run, cfg, probe)
        return trials[identity]
    best = dict(base)
    best_ms, ref = measure(best)
    if best_ms is None:
        _CACHE[key] = best
        return best
    best_b = None if bytes_of is None else bytes_of(best)
    base_state = (best, best_ms, best_b)

    def _search(pre, axes):
        best, best_ms, best_b = base_state

        def _descend(names):
            nonlocal best, best_ms, best_b
            for name, values in ((n, axes[n]) for n in names):
                for v in values:
                    if v == best.get(name):
                        continue
                    cand = dict(best, **{name: v})
                    if candidate_filter is not None and (not candidate_filter(cand)):
                        continue
                    if _hopeless(cand, best_b, bytes_of):
                        continue
                    ms, digest = measure(cand, extreme)
                    if ms is None or not _agrees(ref, digest):
                        continue
                    cand_b = None if bytes_of is None else bytes_of(cand)
                    if _wins(ms, best_ms, cand_b, best_b, tol):
                        best, best_ms, best_b = (cand, ms, cand_b)
        _descend([n for n in pre if n in axes])
        for over in grid or ():
            if any((k in over for k in drop_keys)):
                continue
            cand = dict(best, **over)
            if candidate_filter is not None and (not candidate_filter(cand)):
                continue
            if cand == best or _hopeless(cand, best_b, bytes_of):
                continue
            ms, digest = measure(cand, extreme)
            if ms is None or not _agrees(ref, digest):
                continue
            cand_b = None if bytes_of is None else bytes_of(cand)
            if _wins(ms, best_ms, cand_b, best_b, tol):
                best, best_ms, best_b = (cand, ms, cand_b)
        _descend([n for n in axes if n not in pre] + [n for n in pre if n in axes])
        return (best, best_ms, best_b)
    best, best_ms, best_b = _search(pre, axes)
    if any((n in axes for n in pre)):
        alt = _search((), {k: v for k, v in axes.items() if k not in pre})
        if alt[1] is not None and _wins(alt[1], best_ms, alt[2], best_b, tol):
            best, best_ms, best_b = alt
    _CACHE[key] = best
    if pkey is not None:
        _persist_store(pkey, best)
    return best

def _pow2_span(base, lo, hi):
    out = []
    for step in (2, 4):
        for cand in (base // step, base * step):
            if lo <= cand <= hi and cand > 0 and (cand != base) and (cand not in out):
                out.append(cand)
    return out

def t_axes(base, P, L, ckpt_l, rot, tier='full', ts_axis=False):
    top = triton.next_power_of_2(L)
    ok_l = [b for b in (128, 256, 512, 1024, 2048) if b <= top and (not ckpt_l or b % ckpt_l == 0)]
    axes = {}
    if ts_axis and tier == 'full':
        axes['TS'] = [2, 4, 8, 16, 1]
    axes['BLOCK_L'] = [b for b in _pow2_span(base['BLOCK_L'], 32, 2048) if b in ok_l]
    axes['PF'] = [1 - int(base.get('PF', 0))]
    if tier == 'full':
        axes['NUM_WARPS'] = [w for w in (1, 2, 4) if w != base['NUM_WARPS']]
    return {k: v for k, v in axes.items() if v}

def t_grid(base, P, L, ckpt_l):
    top = triton.next_power_of_2(L)
    out = []
    for bp in (1, 2, 4, 8):
        if P % bp:
            continue
        for bl in (max(512 // bp, 128), max(1024 // bp, 128)):
            if bl > top or (ckpt_l and bl % ckpt_l):
                continue
            for pf in (0, 1):
                o = dict(BLOCK_P=bp, BLOCK_L=bl, NUM_WARPS=1, PF=pf)
                if o not in out:
                    out.append(o)
    return out

def light_fwd_candidate(cfg):
    bp = cfg.get('BLOCK_P', 1)
    return bp == 1 or bp * cfg['BLOCK_L'] <= 2048 * cfg['NUM_WARPS']

def v3_grid(base, H, L, block_l_of, caps=(128,), half_block_l=True):
    out = []
    for hp in (8, 16, 4, 32, 1):
        if hp > H or H % hp:
            continue
        bl = block_l_of(hp)
        if bl > triton.next_power_of_2(L):
            continue
        cand = dict(HP=hp, BLOCK_L=bl)
        if cand not in out:
            out.append(cand)
        if hp in (4, 8, 16) and base.get('TP_P') not in (None, 2):
            cand = dict(HP=hp, BLOCK_L=bl, TP_P=2)
            if cand not in out:
                out.append(cand)
    if out and base.get('MAXNREG') is None:
        hp, bl = (out[0]['HP'], out[0]['BLOCK_L'])
        for cap in caps:
            for b in (bl, max(8, bl // 2)) if half_block_l else (bl,):
                cand = dict(HP=hp, BLOCK_L=b, MAXNREG=cap)
                if cand not in out:
                    out.append(cand)
    return out

def v3_axes(base, nr, rot, xs_bytes1, ckpt_bytes, budget=2 << 30, tune_block_l=True, bc_axis=False, ts_axis=False):
    axes = {}
    if ts_axis:
        axes['TS'] = [4, 8, 16, 32, 2, 1]
    if tune_block_l:
        axes['BLOCK_L'] = [b for b in _pow2_span(base['BLOCK_L'], 8, 256) if ckpt_bytes(b) <= budget]
    ns = [n for n in (1, 2, 4, 8, 16, 32) if n != base['NS'] and nr % n == 0 and (not rot or nr // n % 2 == 0) and (n * xs_bytes1 <= budget)]
    if ns:
        axes['NS'] = ns
    axes['TP_P'] = _pow2_span(base['TP_P'], 2, 32)
    axes['NUM_WARPS'] = [w for w in (1, 2, 4) if w != base['NUM_WARPS']]
    axes['AGX'] = [v for v in (0, 1) if v != base['AGX']]
    if bc_axis:
        axes['BCS'] = [v for v in (1, 8, 16, 64, 0) if v != base['BCS']]
    return {k: v for k, v in axes.items() if v}

def _dedup(cfgs):
    seen, out = (set(), [])
    for c in cfgs:
        t = tuple(sorted(c.items()))
        if t not in seen:
            seen.add(t)
            out.append(c)
    return out

def fwd_candidates(base, P, block_n, divisor_le):
    out = [dict(base)]
    for block_l in (32, 64):
        for tpw_l in (4, 8):
            if tpw_l < 32 // block_n or tpw_l > block_l:
                continue
            for warps in (4, 8):
                for div in (1, 4):
                    out.append(dict(base, BLOCK_L=block_l, TPW_L=tpw_l, NUM_WARPS=warps, BLOCK_P=divisor_le(P, max(1, base['BLOCK_P'] // div))))
    return _dedup(out)

def bwd_candidates(base, block_n, ckpt_bytes=None, budget=2 << 30):
    out = [dict(base)]
    for block_l in (16, 32, 64):
        if ckpt_bytes is not None and ckpt_bytes(block_l) > budget:
            continue
        for tpw_l in (4, 8):
            if tpw_l < 32 // block_n or tpw_l > block_l:
                continue
            for warps in (4, 8):
                out.append(dict(base, BLOCK_L=block_l, TPW_L=tpw_l, NUM_WARPS=warps))
    return _dedup(out)
