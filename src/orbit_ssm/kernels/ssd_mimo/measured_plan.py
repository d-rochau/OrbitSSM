
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import torch
import triton
_ACTIVE = ContextVar('ssd_measured_plan', default=None)
_ALLOWED = {'SISO_FAST', 'BWD_DG_LO', 'BWD_GT_CAP', 'BWD_KBC_SCHED', 'BWD_KBC_SUMDS', 'BWD_KC_SHARED_CB', 'BWD_NSPLIT', 'CHUNKPAR', 'CHUNKPAR_KA', 'CP_FWD_GT', 'CP_G_LO', 'CP_KSUB_FWD', 'CP_KSUB_KA', 'CP_SHARED_CB', 'DSKIP_SLOT', 'GATE_DY_IO', 'GLUON_KBC', 'GLUON_KBC_DALPHA', 'SEG_PREFIX_FUSED', 'SEG_PREFIX_ROWS', 'BWD_K2_WARPS', 'BWD_K2_STAGES'}

def current():
    return _ACTIVE.get()

def override(suffix):
    plan = current()
    return None if plan is None else plan.get('levers', {}).get(suffix)

@contextmanager
def scope(plan):
    token = _ACTIVE.set(plan)
    try:
        yield
    finally:
        _ACTIVE.reset(token)

def backward_scope(fn):

    @wraps(fn)
    def call(ctx, *args):
        with scope(getattr(ctx, '_measured_plan', None)):
            return fn(ctx, *args)
    return call

@lru_cache(None)
def source_digest():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'kernels' / 'ssd_mimo').glob('*.py'))
    paths += [root / 'kernel_options.py', root / 'layers' / 'tails.py']
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()

@lru_cache(None)
def device_key(index):
    p = torch.cuda.get_device_properties(index)
    return (p.name, p.major, p.minor, p.multi_processor_count, p.total_memory, torch.__version__, torch.version.cuda, triton.__version__)

def request_key(arguments):
    values = []
    grad = torch.is_grad_enabled()
    for name, value in arguments.items():
        if name in ('chunk_size', 'chunk_mode'):
            continue
        if isinstance(value, torch.Tensor):
            value = (tuple(value.shape), tuple(value.stride()), str(value.dtype), grad and value.requires_grad)
        values.append((name, value))
    dev = arguments['x_in'].device.index
    body = (source_digest(), device_key(dev), grad, values)
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

@lru_cache(None)
def _read(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 1 or data.get('source_digest') != source_digest():
        raise ValueError('SSD measured plans have a stale source digest/schema')
    return data['plans']

def validate(plan):
    if plan.get('chunk') not in (16, 32, 64):
        raise ValueError('measured SSD chunk must be 16, 32 or 64')
    unknown = set(plan.get('levers', {})) - _ALLOWED
    if unknown:
        raise ValueError(f'unsupported measured SSD levers: {sorted(unknown)}')
    if not plan.get('measurement'):
        raise ValueError('measured SSD plan requires measurement provenance')
    if any((not isinstance(v, str) for v in plan.get('levers', {}).values())):
        raise ValueError('measured SSD lever values must be strings')
    return plan

def scan_scope(fn):
    signature = inspect.signature(fn)

    @torch.compiler.disable
    def calibrated_call(*args, **kwargs):
        path = os.environ.get('SSD_MIMO_MEASURED_PLANS')
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = request_key(bound.arguments)
        plan = {'key': key, 'levers': {}}
        if path:
            plans = _read(path)
            if key not in plans:
                raise ValueError(f'no measured SSD plan for request {key}')
            plan = dict(validate(plans[key]), key=key)
            if not os.environ.get('SSD_MIMO_CHUNK'):
                kwargs = dict(kwargs, chunk_size=plan['chunk'], chunk_mode='calibrated')
        with scope(plan):
            return fn(*args, **kwargs)

    @wraps(fn)
    def call(*args, **kwargs):
        if os.environ.get('SSD_MIMO_MEASURED_PLANS') or os.environ.get('SSD_MIMO_CAPTURE_PLAN_KEY') == '1':
            return calibrated_call(*args, **kwargs)
        return fn(*args, **kwargs)
    return call
