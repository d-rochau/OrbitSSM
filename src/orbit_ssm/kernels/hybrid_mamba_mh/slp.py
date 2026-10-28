
from __future__ import annotations
import os
ENABLED = os.environ.get('HYBRID_MH_SLP', '1').strip().lower() not in ('', '0', 'off', 'false', 'no')
_MH_KERNELS = ('@_mh_fwd_kernel', '@_mh_bwd_kernel')
_STATE = {'installed': None}

class _SLPOn:

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def optimize_module(self, *args, **kwargs):
        kwargs['disable_slp_vectorizer'] = False
        return self._real.optimize_module(*args, **kwargs)

def _install():
    try:
        from triton import knobs
        import triton.backends.nvidia.compiler as nvc
    except Exception:
        return False
    if not hasattr(nvc, 'llvm') or not hasattr(knobs.runtime, 'add_stages_inspection_hook'):
        return False
    if knobs.runtime.add_stages_inspection_hook is not None:
        return False

    def hook(*args):
        if not args:
            return ('', '')
        _backend, stages, _options, _language, capability = args
        if capability != 90 or 'llir' not in stages:
            return
        inner = stages['llir']

        def llir(src, metadata):
            if not _enabled():
                return inner(src, metadata)
            text = src.str_nodebug()
            if not any((k in text for k in _MH_KERNELS)):
                return inner(src, metadata)
            real = nvc.llvm
            nvc.llvm = _SLPOn(real)
            try:
                return inner(src, metadata)
            finally:
                nvc.llvm = real
        stages['llir'] = llir
    knobs.runtime.add_stages_inspection_hook = hook
    return True

def _enabled():
    return bool(ENABLED) and ENABLED not in ('0', 'off', 'False')

def active(capability):
    if capability != 90 or not _enabled():
        return False
    if _STATE['installed'] is None:
        _STATE['installed'] = _install()
    return bool(_STATE['installed'])
