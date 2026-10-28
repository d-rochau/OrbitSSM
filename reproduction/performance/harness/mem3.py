
import argparse
import gc
import json
import os
import weakref
import torch
import hmarms
import scope as S
from orbit_ssm.kernels import shape_sets as ss
GIT_SHA = None
AMP = lambda: torch.autocast('cuda', dtype=torch.bfloat16)

def tags_of(spec):
    out = set()
    for t in filter(None, spec.split(',')):
        if t == 'main':
            out |= {k for k in ss.SET if k.startswith('large_') or k in ss.LM_TAGS}
        elif t == 'scope':
            out |= {k for k in ss.SET if k.startswith('large_')} | {'lm_b6_d2048_l2048'}
        elif t == 'rest':
            out |= {k for k in ss.SET if k.startswith(('small_', 'medium_'))}
        elif t == 'all':
            out |= set(ss.SET)
        else:
            out.add(t)
    return out

class SavedRecorder:

    def __init__(self):
        self.refs = []

    def _note(self, r):
        if isinstance(r, torch.Tensor) and r.is_cuda:
            self.refs.append(weakref.ref(r))
        return r

    def __enter__(self):
        C = torch._C._autograd
        self._orig = C._push_saved_tensors_default_hooks
        rec = self

        def push(pack, unpack):
            return rec._orig(lambda t: rec._note(pack(t)), unpack)
        C._push_saved_tensors_default_hooks = push
        self._orig(lambda t: rec._note(t.detach()), lambda t: t)
        return self

    def __exit__(self, *a):
        C = torch._C._autograd
        C._pop_saved_tensors_default_hooks()
        C._push_saved_tensors_default_hooks = self._orig

    def bytes(self, exclude_ptrs):
        seen, tot = (set(), 0)
        for r in self.refs:
            t = r()
            if t is None:
                continue
            st = t.untyped_storage()
            k = st.data_ptr()
            if k in seen or k in exclude_ptrs or st.nbytes() == 0:
                continue
            seen.add(k)
            tot += st.nbytes()
        return (tot, len(seen))

def _first(y):
    return y[0] if isinstance(y, (tuple, list)) else y

def measure(call, leaves, params, amp, dy_fn, pass_):
    leaves = [t for t in leaves if t is not None and t.requires_grad]
    params = list(params)

    def free():
        for t in leaves + params:
            t.grad = None
    with torch.no_grad(), amp():
        y = _first(call())
    dy = dy_fn(y)
    del y
    for _ in range(2):
        free()
        if pass_ == 'fwd':
            with torch.no_grad(), amp():
                call()
        else:
            with amp():
                y = _first(call())
            y.backward(dy)
            del y
    free()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if pass_ == 'fwd':
        with torch.no_grad(), amp():
            y = call()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del y
        return dict(peak_bytes=peak)
    with amp():
        y = _first(call())
    y.backward(dy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    grad_bytes = sum((t.grad.untyped_storage().nbytes() for t in leaves + params if t.grad is not None))
    del y
    free()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    b2 = torch.cuda.memory_allocated()
    excl = {t.untyped_storage().data_ptr() for t in leaves + params}
    with SavedRecorder() as rec, amp():
        y = _first(call())
    torch.cuda.synchronize()
    excl.add(y.untyped_storage().data_ptr())
    hook_bytes, n_st = rec.bytes(excl)
    out_bytes = y.untyped_storage().nbytes()
    saved = torch.cuda.memory_allocated() - b2 - out_bytes
    del rec
    y.backward(dy)
    del y
    free()
    gc.collect()
    torch.cuda.synchronize()
    residual = torch.cuda.memory_allocated() - b2
    torch.cuda.empty_cache()
    if residual > 2 ** 20:
        raise RuntimeError(f'saved-for-backward not released by the backward: {residual} B stay allocated after fwd+bwd')
    return dict(saved_for_backward_bytes=saved, peak_bytes=peak, grad_bytes=grad_bytes, out_bytes=out_bytes, graph_release_residual_bytes=residual, saved_hook_bytes=hook_bytes, saved_hook_storages=n_st, hook_vs_alloc=None if saved <= 0 else round(hook_bytes / saved, 4))

def build(level, arm, d_inner, N, B, L):
    if level in ('pair', 'scope'):
        s, v = S.build_scope(arm, d_inner, N, B, L, keep_other=False)

        def dy_fn(y, s=s):
            dy = torch.empty_like(y)
            yc = s.canon(dy)
            yc.copy_(S.canonical_dy(*yc.shape, device=y.device))
            return dy
        keep = s.keep
        params = [p for p in (keep.parameters() if keep is not None else []) if p.requires_grad]
        return (s.call, s.leaf_list(), params, s.amp, dy_fn, s, dict(validation=v, geometry={**dict(B=B, L=L, d_inner=d_inner), **s.meta}))
    call, leaves, mod = hmarms.build(arm, d_inner // 2, N, B, L)

    def dy_fn(y):
        g = torch.Generator(device='cpu').manual_seed(24301)
        return torch.randn(y.shape, generator=g, dtype=torch.float32).to(device=y.device, dtype=y.dtype)
    return (call, leaves[:1], list(mod.parameters()), AMP, dy_fn, mod, dict(geometry=dict(B=B, L=L, d_inner=d_inner, N=N)))
import hashlib
_here = os.path.dirname(os.path.abspath(__file__))
SCRIPT_SHA = hashlib.sha256(b''.join((open(os.path.join(_here, f), 'rb').read() for f in ('mem3.py', 'scope.py', 'hmarms.py')))).hexdigest()[:16]

def git_sha():
    try:
        from benchmarks.harness import env_meta
        return env_meta('cuda').get('git_sha')
    except Exception:
        return None

def provenance(level, arm, keep, call, leaves, params, amp, dy_fn):
    from benchmarks import route_record as rr
    ssm = getattr(keep, 'ssm', None)
    if ssm is None:
        return dict(route=dict(entry=f'official {arm}'), chunk=dict(official=getattr(keep, 'chunk_size', None) or getattr(getattr(keep, '_n', None), 'chunk_size', None) or 'official default'), calibration='official defaults (mamba_ssm 2.3.2.post1)')

    def once():
        with amp():
            y = _first(call())
        with torch.no_grad():
            dy = dy_fn(y)
        y.backward(dy)
        for t in list(leaves) + list(params):
            t.grad = None
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    r = rr.served(once) if isinstance(ssm, SSDMimoSSM) else rr.served_mh(once, ssm)
    rt = r.get('route') or {}
    chunk = dict(fwd=rt.get('chunk_used'), bwd=rt.get('chunk_bwd')) if 'mh' not in rt else dict(fwd_BLOCK_L=(rt['mh'].get('fwd_config') or {}).get('BLOCK_L'), bwd_BLOCK_L=(rt['mh'].get('bwd_config') or {}).get('BLOCK_L'))
    out = dict(route=rt, chunk=chunk, calibration=r.get('calibration'))
    if isinstance(ssm, SSDMimoSSM):
        out['production_problem'] = rr.production_problem(r)
    return out

def specs():
    out = []
    for x in os.environ['MEM_SPECS'].split(';'):
        p = x.split(':')
        out.append((p[0], ':'.join(p[1:-1]), int(p[-1])))
    return out

def main():
    from orbit_ssm.layers import regen as _regen
    from orbit_ssm.ssm.hybrid_mamba import hm_held_regen
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='main')
    a = ap.parse_args()
    only = tags_of(a.only)
    fo = open(a.out, 'a')
    global GIT_SHA
    GIT_SHA = git_sha()
    policy = dict(HM_HELD_REGEN=os.environ.get('HM_HELD_REGEN', '<unset>=all'), hm_effective=sorted(hm_held_regen()), SSD_MIMO_HELD_REGEN=os.environ.get('SSD_MIMO_HELD_REGEN', '<unset>'), ssd_effective_real=_regen.mode(True), ssd_effective_other=_regen.mode(False), compile='off (eager)')
    for tag, (B, di, L) in ss.SET.items():
        if tag not in only:
            continue
        for level, arm, N in specs():
            for p in ('fwd', 'fwd_bwd'):
                torch.cuda.empty_cache()
                keep = None
                rec = dict(tag=tag, B=B, d_inner=di, L=L, level=level, arm=arm, N=N, pass_=p, regen_policy=policy, git_sha=GIT_SHA, script_sha=SCRIPT_SHA, gpu=torch.cuda.get_device_name(), torch=torch.__version__)
                try:
                    call, leaves, params, amp, dy_fn, keep, extra = build(level, arm, di, N, B, L)
                    rec.update(extra)
                    g = rec['geometry']
                    for k, v in (('d_model', di // 2), ('R', 1), ('O', 1)):
                        g.setdefault(k, v)
                    ssm = getattr(keep, 'ssm', None)
                    for k, attr in (('H', 'H'), ('P', 'P'), ('R', 'R'), ('O', 'O')):
                        if g.get(k) is None and ssm is not None:
                            g[k] = getattr(ssm, attr, None)
                    if level == 'mixer' and ssm is None:
                        g.setdefault('H', getattr(keep, 'nheads', di))
                        g.setdefault('P', getattr(keep, 'headdim', 1))
                    rec.update(measure(call, leaves, params, amp, dy_fn, p))
                    rec['provenance'] = provenance(level, arm, keep, call, leaves, params, amp, dy_fn)
                    if rec['provenance'].get('production_problem'):
                        raise RuntimeError(f"production_problem: {rec['provenance']['production_problem']}")
                    rec['error'] = None
                except Exception as e:
                    rec['error'] = f'{type(e).__name__}: {e}'[:300]
                fo.write(json.dumps(rec, default=str) + '\n')
                fo.flush()
                print(tag, level, arm, N, p, rec.get('saved_for_backward_bytes'), rec.get('peak_bytes'), rec.get('hook_vs_alloc'), rec['error'] or '', flush=True)
                call = leaves = params = keep = None
                gc.collect()
                torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
