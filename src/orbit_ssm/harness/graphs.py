
from __future__ import annotations
import contextlib
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence
import torch
import torch.nn as nn
__all__ = ['CaptureError', 'GraphedStep', 'autocast', 'capture_safe', 'graphed_step', 'prewarm_host_caches', 'validation_off']

class CaptureError(RuntimeError):

    def __init__(self, reason: str, detail: str, cause: BaseException | None=None):
        super().__init__(f'{reason}: {detail}')
        self.reason = reason
        self.detail = detail
        self.__cause__ = cause
_SYNC_MARKERS = ('operation not permitted when stream is capturing', 'operation would make the legacy stream depend on a capturing', 'captures_underway', 'cudaErrorStreamCaptureUnsupported', 'cudaErrorStreamCaptureInvalidated', 'cudaErrorStreamCaptureIsolation')

def autocast(dtype: torch.dtype | str | None=torch.bfloat16, device_type: str='cuda'):
    if isinstance(dtype, str):
        name = dtype.lower()
        if name in ('bf16', 'bfloat16'):
            dtype = torch.bfloat16
        elif name in ('fp16', 'float16', 'half'):
            dtype = torch.float16
        elif name in ('fp32', 'float32', 'ieee', 'tf32', 'none', ''):
            dtype = None
        else:
            raise ValueError(f'unknown autocast precision {dtype!r}')
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type, dtype=dtype, cache_enabled=False)

def _amp_factory(amp) -> Callable[[], Any]:
    if amp is None:
        return contextlib.nullcontext
    if isinstance(amp, (str, torch.dtype)):
        return lambda: autocast(amp)
    if callable(amp):
        return amp
    raise TypeError(f'amp must be None, a precision name, a torch.dtype or a zero-argument context-manager factory, got {amp!r}')
_ENV_GLUE = 'SSD_MIMO_GLUE'
_ENV_VALIDATE = ('SSD_MIMO_VALIDATE', 'HYBRID_SSM_MH_VALIDATE')
_VALIDATE_WHAT = {'SSD_MIMO_VALIDATE': "check_a_log_range's .item()", 'HYBRID_SSM_MH_VALIDATE': "assert_pair_tied's row_rot read"}

def _glue_without_wcat(value: str | None) -> str:
    v = '1' if value is None else value
    if v in ('1', '', '0'):
        return v
    if v == 'all':
        return 'all:wcat'
    if v.startswith('all:'):
        off = [s for s in v[4:].split(',') if s]
        return 'all:' + ','.join(off + ['wcat']) if 'wcat' not in off else v
    return v

def prewarm_host_caches(module: nn.Module) -> list[str]:
    warmed: list[str] = []
    try:
        from ..kernels.hybrid_mamba_mh.common_gluon import selector_window
    except Exception:
        return warmed
    for name, sub in module.named_modules():
        for buf in ('row_law', 'row_rot'):
            t = getattr(sub, buf, None)
            if isinstance(t, torch.Tensor) and t.is_cuda:
                selector_window(t, 0, int(t.numel()))
                warmed.append(f"{(name + '.' if name else '')}{buf}")
    return warmed

@contextlib.contextmanager
def validation_off():
    notes: list[str] = []
    old = {k: os.environ.get(k) for k in _ENV_VALIDATE}
    for key, was in old.items():
        if was is not None and was.lower() not in ('0', '', 'false', 'no'):
            os.environ[key] = '0'
            notes.append(f"{key}: {was!r} -> '0' for the capture only ({_VALIDATE_WHAT[key]} is a device->host sync; the warm-up iterations still run it)")
    try:
        yield notes
    finally:
        for key, was in old.items():
            if was is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = was

@contextlib.contextmanager
def capture_safe(module: nn.Module | None=None, *, validation: bool=True):
    notes: list[str] = []
    old_glue = os.environ.get(_ENV_GLUE)
    new_glue = _glue_without_wcat(old_glue)
    if new_glue != (old_glue if old_glue is not None else '1'):
        os.environ[_ENV_GLUE] = new_glue
        notes.append(f'{_ENV_GLUE}: {old_glue!r} -> {new_glue!r} (the G33 fused-in_proj weight cache would freeze the weights inside the graph)')
    if module is not None:
        for sub in module.modules():
            if getattr(sub, '_in_proj_cache', None) is not None:
                sub._in_proj_cache = None
                notes.append('cleared a stale _in_proj_cache before capture')
        warmed = prewarm_host_caches(module)
        if warmed:
            notes.append('pre-read the mh row selectors (' + ', '.join(warmed) + ") so selector_window's memo cannot miss inside the capture")
    try:
        with contextlib.ExitStack() as stack:
            if not validation:
                notes += stack.enter_context(validation_off())
            yield notes
    finally:
        if old_glue is None:
            os.environ.pop(_ENV_GLUE, None)
        else:
            os.environ[_ENV_GLUE] = old_glue
        if module is not None:
            for sub in module.modules():
                if getattr(sub, '_in_proj_cache', None) is not None:
                    sub._in_proj_cache = None

@dataclass
class GraphedStep:
    module: nn.Module
    graph: torch.cuda.CUDAGraph
    fwd_bwd: bool
    static_inputs: tuple[torch.Tensor, ...]
    static_output: torch.Tensor
    static_grad_output: torch.Tensor | None
    leaves: tuple[tuple[str, torch.Tensor], ...]
    static_grads: tuple[torch.Tensor, ...]
    pool: Any
    stream: torch.cuda.Stream
    notes: list[str] = field(default_factory=list)
    static_bytes: int = 0
    pool_bytes: int = 0
    capture_warnings: tuple[str, ...] = ()

    def copy_in(self, *inputs: torch.Tensor, grad_outputs: torch.Tensor | None=None) -> None:
        if inputs:
            if len(inputs) != len(self.static_inputs):
                raise ValueError(f'captured {len(self.static_inputs)} input(s), got {len(inputs)}')
            for dst, src in zip(self.static_inputs, inputs):
                if src is dst:
                    continue
                if tuple(src.shape) != tuple(dst.shape) or src.dtype != dst.dtype:
                    raise ValueError(f'a captured graph IS its shape: expected {tuple(dst.shape)}/{dst.dtype}, got {tuple(src.shape)}/{src.dtype}')
                with torch.no_grad():
                    dst.copy_(src)
        if grad_outputs is not None:
            if self.static_grad_output is None:
                raise ValueError('forward-only capture takes no grad_outputs')
            with torch.no_grad():
                self.static_grad_output.copy_(grad_outputs)

    def replay(self, *inputs: torch.Tensor, grad_outputs: torch.Tensor | None=None) -> torch.Tensor:
        self.copy_in(*inputs, grad_outputs=grad_outputs)
        self.graph.replay()
        return self.static_output
    __call__ = replay

    @property
    def grads(self) -> dict[str, torch.Tensor]:
        return {n: g for (n, _), g in zip(self.leaves, self.static_grads) if g is not None}

    @property
    def memory(self) -> dict[str, int]:
        return dict(static=self.static_bytes, graph_pool=self.pool_bytes, total=self.static_bytes + self.pool_bytes)

    def __repr__(self) -> str:
        return f"GraphedStep({type(self.module).__name__}, {('fwd_bwd' if self.fwd_bwd else 'fwd')}, inputs={[tuple(t.shape) for t in self.static_inputs]}, static={self.static_bytes / 2 ** 20:.1f} MiB, pool={self.pool_bytes / 2 ** 20:.1f} MiB)"

def _first(y):
    if isinstance(y, torch.Tensor):
        return y
    if isinstance(y, (tuple, list)):
        if len(y) and isinstance(y[0], torch.Tensor):
            return y[0]
    raise CaptureError('output', f'the module returned {type(y).__name__}, not a tensor (a `state=`/`return_state=` continuation is outside the capture envelope -- the carry is a python dict whose identity changes per call)')

def _nbytes(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.numel() * t.element_size()

def _check_cuda(module: nn.Module, inputs: Sequence[torch.Tensor]) -> torch.device:
    if not torch.cuda.is_available():
        raise CaptureError('no-cuda', 'CUDA graphs need a CUDA device; this process has none')
    devs = {t.device for t in inputs if isinstance(t, torch.Tensor)}
    devs |= {p.device for p in module.parameters()}
    devs |= {b.device for b in module.buffers()}
    non_cuda = sorted({str(d) for d in devs if d.type != 'cuda'})
    if non_cuda:
        raise CaptureError('not-on-cuda', f'every parameter, buffer and input must live on one CUDA device; found {non_cuda}')
    if len({d.index for d in devs}) > 1:
        raise CaptureError('multi-device', f'the captured region must run on ONE device; found {sorted((str(d) for d in devs))}')
    return next(iter(devs))

def _classify(exc: BaseException) -> CaptureError:
    msg = str(exc)
    low = msg.lower()
    if any((m.lower() in low for m in _SYNC_MARKERS)):
        return CaptureError('host-sync', 'something inside the captured region talked to the host (an `.item()`, `.tolist()`, `.cpu()`, a `torch.cuda.synchronize()` or a shape read off a device tensor).  Every such read has to be hoisted out of the per-step path or memoised before the capture -- see `orbit_ssm.harness.graphs.capture_safe`.  torch said: ' + msg, exc)
    if 'out of memory' in low:
        return CaptureError('oom', 'the capture ran out of memory; a graph pool is allocated on top of the live activations.  torch said: ' + msg, exc)
    if 'autocast' in low and 'cach' in low:
        return CaptureError('autocast-cache', msg, exc)
    return CaptureError('capture-failed', msg, exc)

def graphed_step(module: nn.Module, example_inputs: torch.Tensor | Sequence[torch.Tensor], *, fwd_bwd: bool=True, amp: Any=None, warmup: int=11, grad_outputs: torch.Tensor | None=None, params: Iterable[torch.Tensor] | None=None, pool: Any=None, stream: torch.cuda.Stream | None=None, seed: int | None=0) -> GraphedStep:
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    example_inputs = tuple(example_inputs)
    if not example_inputs:
        raise CaptureError('output', 'a capture needs at least one input tensor to copy a step into')
    dev = _check_cuda(module, example_inputs)
    amp_ctx = _amp_factory(amp)
    static_inputs = tuple((t.detach().clone().requires_grad_(t.requires_grad) for t in example_inputs))
    leaf_list: list[tuple[str, torch.Tensor]] = []
    if params is None:
        leaf_list += [(n, p) for n, p in module.named_parameters() if p.requires_grad]
        leaf_list += [(f'input[{i}]', t) for i, t in enumerate(static_inputs) if t.requires_grad]
    else:
        leaf_list += [(f'leaf[{i}]', p) for i, p in enumerate(params) if p.requires_grad]
    leaves = tuple(leaf_list)
    notes: list[str] = []
    stream = stream if stream is not None else torch.cuda.Stream(device=dev)
    with capture_safe(module) as safety_notes:
        notes += safety_notes
        stream.wait_stream(torch.cuda.current_stream(dev))
        shapes: set[tuple] = set()
        dy_ref: torch.Tensor | None = None
        try:
            with torch.cuda.stream(stream):
                for it in range(max(int(warmup), 1)):
                    if fwd_bwd:
                        for _, p in leaves:
                            p.grad = None
                        with amp_ctx():
                            _check_autocast_cache()
                            y = _first(module(*static_inputs))
                        shapes.add((tuple(y.shape), y.dtype))
                        if not y.requires_grad:
                            raise CaptureError('no-grad', 'the forward produced a graph-less output, so there is no backward to capture (no leaf requires grad)')
                        if dy_ref is None:
                            if seed is not None:
                                torch.manual_seed(seed)
                            dy_ref = torch.randn_like(y) if grad_outputs is None else grad_outputs.detach().clone().to(device=y.device, dtype=y.dtype)
                        y.backward(dy_ref)
                    else:
                        with torch.no_grad(), amp_ctx():
                            _check_autocast_cache()
                            y = _first(module(*static_inputs))
                        shapes.add((tuple(y.shape), y.dtype))
                    del y
        except CaptureError:
            raise
        except Exception as exc:
            raise _classify(exc) from exc
        torch.cuda.current_stream(dev).wait_stream(stream)
        torch.cuda.synchronize(dev)
        if len(shapes) != 1:
            raise CaptureError('dynamic-shape', f'the forward produced more than one output shape/dtype over the warm-up iterations ({sorted(map(str, shapes))}); a captured graph IS its shape')
        if fwd_bwd:
            for _, p in leaves:
                p.grad = torch.zeros_like(p)
        static_dy = dy_ref
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved(dev)
        graph = torch.cuda.CUDAGraph()
        kw = {} if pool is None else {'pool': pool}
        try:
            with warnings.catch_warnings(record=True) as caught, validation_off() as vnotes:
                warnings.simplefilter('always')
                notes += vnotes
                with torch.cuda.graph(graph, stream=stream, **kw):
                    if fwd_bwd:
                        for _, p in leaves:
                            p.grad.zero_()
                        with amp_ctx():
                            static_out = _first(module(*static_inputs))
                        static_out.backward(static_dy)
                    else:
                        with torch.no_grad(), amp_ctx():
                            static_out = _first(module(*static_inputs))
        except CaptureError:
            raise
        except Exception as exc:
            raise _classify(exc) from exc
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        pool_bytes = max(0, torch.cuda.memory_reserved(dev) - before)
        cap_warnings = tuple(sorted({str(w.message).split('\n')[0][:200] for w in caught}))
        acc = [w for w in cap_warnings if 'AccumulateGrad' in w]
        if acc:
            notes.append('torch still warned about the AccumulateGrad stream: ' + acc[0])
    static_grads: tuple[torch.Tensor, ...] = ()
    if fwd_bwd:
        ptrs = [None if p.grad is None else p.grad.data_ptr() for _, p in leaves]
        graph.replay()
        torch.cuda.synchronize(dev)
        if [None if p.grad is None else p.grad.data_ptr() for _, p in leaves] != ptrs:
            raise CaptureError('unstable-grad', 'a replay re-bound at least one `.grad` to a new address, so the gradient buffers a caller holds would go stale after one step')
        static_grads = tuple((p.grad for _, p in leaves))
    static_bytes = sum((_nbytes(t) for t in static_inputs)) + _nbytes(static_dy) + sum((_nbytes(g) for g in static_grads))
    return GraphedStep(module=module, graph=graph, fwd_bwd=fwd_bwd, static_inputs=static_inputs, static_output=static_out, static_grad_output=static_dy, leaves=leaves, static_grads=static_grads, pool=graph.pool(), stream=stream, notes=notes, static_bytes=static_bytes, pool_bytes=pool_bytes, capture_warnings=cap_warnings)

def _check_autocast_cache() -> None:
    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise CaptureError('autocast-cache', 'the autocast context must be built with `cache_enabled=False` (use `orbit_ssm.harness.graphs.autocast`): with caching on, the bf16 cast of every fp32 parameter is memoised across iterations, so the graph would record a READ of a warm-up tensor and every replay after an optimizer step would use stale weights')
