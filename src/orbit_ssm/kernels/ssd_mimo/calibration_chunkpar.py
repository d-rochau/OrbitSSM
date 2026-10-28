
import os
import time
from .. import calibration as cal
FAMILY = 'ssd_mimo_chunkpar'
_HERE = os.path.dirname(os.path.abspath(__file__))
cal.register_family(FAMILY, env='SSD_MIMO_CALIBRATION', code_files=[os.path.join(_HERE, 'chunkpar.py')], kernels=(), shapes={})

def lookup(cache_key):
    if '|' not in cache_key or not cal.enabled(FAMILY):
        return None
    suffix = cache_key.split('|', 1)[1]
    v, src = cal.threshold(FAMILY, suffix, None)
    return v
REAL_T_KEYS = {'train_fwd': 'real_r1o1|T|train_fwd', 'train_bwd': 'real_r1o1|T|train_bwd', 'fwd_only': 'real_r1o1|T|fwd_only'}
REAL_T_G = 64
REAL_RULE_KEYS = {'T_hi': 'real_r1o1|Trule|T_hi', 'L_max': 'real_r1o1|Trule|L_max', 'W_min': 'real_r1o1|Trule|W_min'}
RULE_KINDS = ('train_fwd', 'train_bwd')
_RULE_MEMO = {}
_RULE_TTL = 2.0
FP32_SUFFIX = '|fp32'

def dtype_axis(dtype) -> str:
    import torch
    return 'fp32' if dtype in (torch.float32, 'fp32', 'float32') else ''

@cal._const
def _fp32_calibrated() -> bool:
    data, _ = cal.load(FAMILY)
    th = (data or {}).get('thresholds', {})
    return any((k + FP32_SUFFIX in th for k in REAL_T_KEYS.values()))

@cal._const
def real_rule(dtype=None):
    ax = dtype_axis(dtype) if dtype is not None else ''
    memo = _RULE_MEMO.setdefault(ax, [False, None, 0.0, None])
    now = time.monotonic()
    env = os.environ.get('SSD_MIMO_CALIBRATION')
    if now - memo[2] < _RULE_TTL and env == memo[3]:
        return memo[1]
    memo[2], memo[3] = (now, env)
    if not cal.enabled(FAMILY):
        memo[0], memo[1] = (False, None)
        return None
    data, _ = cal.load(FAMILY)
    if memo[0] is not data:
        th = (data or {}).get('thresholds', {})
        suf = FP32_SUFFIX if ax == 'fp32' and _fp32_calibrated() else ''
        vals = [th.get(REAL_RULE_KEYS[k] + suf) for k in ('T_hi', 'L_max', 'W_min')]
        memo[0] = data
        memo[1] = None if any((v is None for v in vals)) else tuple((int(v['value']) for v in vals))
    return memo[1]

def real_chunk(kind, G=None, L=None, work=None, dtype=None):
    if G is not None and int(G) != REAL_T_G:
        return None
    if not cal.enabled(FAMILY):
        return None
    suf = FP32_SUFFIX if dtype is not None and dtype_axis(dtype) == 'fp32' and _fp32_calibrated() else ''
    v, _ = cal.threshold(FAMILY, REAL_T_KEYS[kind] + suf, None)
    if v is None:
        return None
    if L is not None and work is not None and (kind in RULE_KINDS):
        rule = real_rule(dtype)
        if rule is not None and int(L) <= rule[1] and (int(work) >= rule[2]):
            return rule[0]
    return int(v)
RANK_T_KINDS = ('train_fwd', 'train_bwd', 'fwd_only')

def rank_key(real, R, O, kind, fp32=False):
    return 'rank|%s|R%dO%d|T|%s%s' % ('real' if real else 'cx', int(R), int(O), kind, FP32_SUFFIX if fp32 else '')

def rank_chunk(real, R, O, kind, G=None, dtype=None):
    if int(R) < 2 or int(R) != int(O):
        return None
    if not cal.enabled(FAMILY):
        return None
    key = rank_key(real, R, O, kind, dtype is not None and dtype_axis(dtype) == 'fp32')
    data, _ = cal.load(FAMILY)
    entry = (data or {}).get('thresholds', {}).get(key, {})
    if G is not None and int(G) != int(entry.get('kernel_width', REAL_T_G)):
        return None
    v, _ = cal.threshold(FAMILY, key, None)
    return None if v is None else int(v)

def export_from_cache(chunk_cache, device_name, rounds=5, write_shipped=False, dropped=None):
    prefix = 'chunk:cp:%s|' % device_name
    th = {k[len(prefix):]: {'value': int(v), 'evidence': 'tools/cp_calibrate.py'} for k, v in chunk_cache.items() if k.startswith(prefix)}
    path = (cal.shipped_path if write_shipped else cal.local_path)(FAMILY, cal.device_info()['gpu'])
    old = cal._read(path)
    if old and cal.check_file(old, FAMILY, cal.device_info()['gpu'], cal.device_info()['triton']) is None:
        th = dict(old.get('thresholds', {}), **th)
    from .. import shape_sets as ss
    data = cal.build_file(FAMILY, {}, rounds, [t for t in ss.SET if t not in (dropped or {})], thresholds=th, dropped=dropped)
    return cal.write_json(path, data)
