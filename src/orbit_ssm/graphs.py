"""CUDA-graph capture of a whole training step (forward, backward) of any module.

Both our mixers and the official Mamba layers are host-bound at benchmark
sizes, so kernel ratios are only meaningful graphed against graphed; this
module captures the step for any `nn.Module` and returns a `GraphedStep` to
replay. Entry point: `graphed_step`.

What is captured, one graph per step:

    zero the static .grad buffers  ->  forward under autocast  ->
    y.backward(static dY)

With ``fwd_bwd=False`` only the forward is captured, under `no_grad`. The
optimizer stays outside the graph: its `step()` reads `lr` and step counters
on the host (only `capturable=True` variants keep them on device, and
`fused=True` does not bump `_version`), and `GraphedStep.grads` gives an eager
optimizer stable, fixed-address gradient buffers.

Requirements and invariants:

1. Static buffers. A replay re-runs the recorded kernels against the recorded
   addresses, so inputs, the incoming gradient and every parameter gradient are
   allocated once before capture; a step is `copy_` into the input buffers plus
   `replay()`. Parameter gradients are zeroed inside the graph, so `.grad`
   holds this step's gradient, not an accumulation.
2. Autocast must use `cache_enabled=False` (`graphs.autocast`). Otherwise a
   bf16 weight cast made during warm-up would be read, not recomputed, inside
   the capture, and replays would train on stale weights. A caller-supplied
   factory is checked and refused if it caches.
3. Warm-up and capture run on the same stream (`GraphedStep.stream`): an
   `AccumulateGrad` node remembers the stream it was created on, and a
   mismatch warns and may break capture. Stepping the same module eagerly on
   another stream afterwards warns for the converse reason, so benchmarks that
   interleave both regimes on one module must use `.stream`.
4. Host reads and cross-step caches on the per-step path are neutralised at
   module level, see `capture_safe`.
5. The recorded kernels embody the module's execution at capture time (backend,
   precision, chunk size, kernel options).  Changing it afterwards
   (`set_execution`, the ``execution(...)`` context manager, also on a single
   layer) bumps `orbit_ssm.execution.execution_version` of the module tree; a
   `GraphedStep` stores the value it was captured at, `GraphedStep.stale`
   reports a difference and `replay` raises `CaptureError("stale-execution")`
   instead of silently running the old kernels.  Capture again under the new
   execution.

Not covered: `state=` / `return_state=` continuation (the carry is a python
dict whose identity changes per call), dynamic shapes (a graph is its shape),
and anything that syncs the device inside the captured region; all are
refused with a `CaptureError`.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn as nn

from .execution import execution_version

__all__ = ["CaptureError", "GraphedStep", "autocast", "capture_safe",
           "graphed_step", "prewarm_host_caches", "validation_off"]


# ── refusal ──────────────────────────────────────────────────────────
class CaptureError(RuntimeError):
    """A module could not be captured, with the reason attached.

    Raised, never swallowed, so a benchmark row becomes a recorded skip rather
    than an eager number quoted as a captured one.  `reason` is a short tag
    (see `graphed_step`); `str(exc)` carries the explanation and, where torch
    produced one, the original error.
    """

    def __init__(self, reason: str, detail: str, cause: BaseException | None = None):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.__cause__ = cause


#: Substrings of CUDA/torch errors meaning "the captured region talked to the
#: host"; used by `_classify`.
_SYNC_MARKERS = (
    "operation not permitted when stream is capturing",
    "operation would make the legacy stream depend on a capturing",
    "captures_underway",
    "cudaErrorStreamCaptureUnsupported",
    "cudaErrorStreamCaptureInvalidated",
    "cudaErrorStreamCaptureIsolation",
)


# ── autocast, in the one form capture accepts ────────────────────────
def autocast(dtype: torch.dtype | str | None = torch.bfloat16,
             device_type: str = "cuda"):
    """The autocast a capture may run under: ``cache_enabled=False``.

    Parameters
    ----------
    dtype : torch.dtype, str or None, default torch.bfloat16
        A dtype or a precision name: ``"bf16"``, ``"fp16"``, or
        ``"fp32"``/``"ieee"``/``"tf32"``/``None`` (no autocast).
    device_type : str, default "cuda"

    Returns a context manager (``with graphs.autocast("bf16"):``). To pass it
    to `graphed_step` as a factory use ``amp=lambda: graphs.autocast("bf16")``.
    """
    if isinstance(dtype, str):
        name = dtype.lower()
        if name in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif name in ("fp16", "float16", "half"):
            dtype = torch.float16
        elif name in ("fp32", "float32", "ieee", "tf32", "none", ""):
            dtype = None
        else:
            raise ValueError(f"unknown autocast precision {dtype!r}")
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type, dtype=dtype, cache_enabled=False)


def _amp_factory(amp) -> Callable[[], Any]:
    """Normalise `graphed_step`'s ``amp=`` into a context-manager factory."""
    if amp is None:
        return contextlib.nullcontext
    if isinstance(amp, (str, torch.dtype)):
        return lambda: autocast(amp)
    if callable(amp):
        return amp
    raise TypeError("amp must be None, a precision name, a torch.dtype or a "
                    f"zero-argument context-manager factory, got {amp!r}")


# ── the module-level capture-safety envelope ─────────────────────────
#
# Host reads abort a capture; cross-step caches are worse: they capture fine
# and silently freeze a value the replay can never refresh. Three are handled
# here, at module level, without touching `src/orbit_ssm/kernels/`:
#
#  1. The `wcat` fused in_proj weight cache (`glue_on("wcat")`, keyed on
#     `(data_ptr, _version, requires_grad)`). Opt-in, but under
#     `SSD_MIMO_GLUE=all` a warm-up populates it and the capture would hit it,
#     so replays would read the warm-up's weights. It is turned off and cleared
#     for warm-up and capture.
#  2. The per-call ABI validation passes (`SSD_MIMO_VALIDATE`:
#     `check_a_log_range`'s `.item()`; `HYBRID_SSM_MH_VALIDATE`:
#     `assert_pair_tied`'s `row_rot` read). Both are device->host syncs and
#     opt-in, but `tests/conftest.py` turns them on. Forced off for the capture
#     only; warm-up still runs them, and a replay runs no python so neither
#     condition can change.
#  3. `selector_window`'s `.tolist()` of the mh `row_law` / `row_rot`
#     selectors, memoised per tensor and `_version`; a miss inside the capture
#     is a sync. `prewarm_host_caches` reads them before the capture.

_ENV_GLUE = "SSD_MIMO_GLUE"
#: The per-call validation passes of the two families.  Both sync; both
#: are already opt-in; both are turned off for the capture ONLY (the
#: warm-up iterations run them).
_ENV_VALIDATE = ("SSD_MIMO_VALIDATE", "HYBRID_SSM_MH_VALIDATE")
_VALIDATE_WHAT = {
    "SSD_MIMO_VALIDATE": "check_a_log_range's .item()",
    "HYBRID_SSM_MH_VALIDATE": "assert_pair_tied's row_rot read",
}


def _glue_without_wcat(value: str | None) -> str:
    """`SSD_MIMO_GLUE` with the `wcat` cache forced off (grammar of `glue_on`:
    ``"1"``/``"0"``/a plain off-list already exclude it; ``"all"`` and
    ``"all:<list>"`` turn it on)."""
    v = "1" if value is None else value
    if v in ("1", "", "0"):
        return v
    if v == "all":
        return "all:wcat"
    if v.startswith("all:"):
        off = [s for s in v[4:].split(",") if s]
        return "all:" + ",".join(off + ["wcat"]) if "wcat" not in off else v
    return v                       # a plain off-list: wcat is off already


def prewarm_host_caches(module: nn.Module) -> list[str]:
    """Read the mh family's `row_law` / `row_rot` selectors once, so the kernel
    launcher's memo can only hit inside a capture (item 3 of the envelope).
    Returns the qualified buffer names it warmed."""
    warmed: list[str] = []
    try:
        from .kernels.hybrid_mamba_mh.common_gluon import selector_window
    except Exception:                                      # noqa: BLE001
        return warmed
    for name, sub in module.named_modules():
        for buf in ("row_law", "row_rot"):
            t = getattr(sub, buf, None)
            if isinstance(t, torch.Tensor) and t.is_cuda:
                selector_window(t, 0, int(t.numel()))
                warmed.append(f"{name + '.' if name else ''}{buf}")
    return warmed


@contextlib.contextmanager
def validation_off():
    """Turn the two families' per-call validation syncs off (item 2 of the
    envelope). Yields a list of notes describing what was changed.

    Held around the capture only, never the warm-up, so a bad module is still
    refused before anything is recorded.
    """
    notes: list[str] = []
    old = {k: os.environ.get(k) for k in _ENV_VALIDATE}
    for key, was in old.items():
        if was is not None and was.lower() not in ("0", "", "false", "no"):
            os.environ[key] = "0"
            notes.append(f"{key}: {was!r} -> '0' for the capture only "
                         f"({_VALIDATE_WHAT[key]} is a device->host sync; "
                         "the warm-up iterations still run it)")
    try:
        yield notes
    finally:
        for key, was in old.items():
            if was is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = was


@contextlib.contextmanager
def capture_safe(module: nn.Module | None = None, *, validation: bool = True):
    """The process state a capture needs, restored on exit.

    Turns off the cross-step caches of the envelope above, clears anything
    already cached on `module`, and pre-reads the mh row selectors.
    `validation=False` additionally enters `validation_off`. Yields the list
    of notes (what it changed), which end up in `GraphedStep.notes`.
    """
    notes: list[str] = []
    old_glue = os.environ.get(_ENV_GLUE)
    new_glue = _glue_without_wcat(old_glue)
    if new_glue != (old_glue if old_glue is not None else "1"):
        os.environ[_ENV_GLUE] = new_glue
        notes.append(f"{_ENV_GLUE}: {old_glue!r} -> {new_glue!r} "
                     "(the fused-in_proj weight cache would freeze the "
                     "weights inside the graph)")
    if module is not None:
        for sub in module.modules():
            if getattr(sub, "_in_proj_cache", None) is not None:
                sub._in_proj_cache = None
                notes.append("cleared a stale _in_proj_cache before capture")
        warmed = prewarm_host_caches(module)
        if warmed:
            notes.append("pre-read the mh row selectors (" +
                         ", ".join(warmed) + ") so selector_window's memo "
                         "cannot miss inside the capture")
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
                if getattr(sub, "_in_proj_cache", None) is not None:
                    sub._in_proj_cache = None


# ── the handle ───────────────────────────────────────────────────────
@dataclass
class GraphedStep:
    """A captured training step, replayable against static buffers.

    `replay(*inputs)` copies `inputs` into the static input buffers and
    replays the graph; it returns `static_output`, the graph's own y buffer,
    overwritten by the next replay (clone it to keep it).  After a fwd+bwd
    replay every entry of `grads` holds this step's gradient.
    """

    module: nn.Module
    graph: torch.cuda.CUDAGraph
    fwd_bwd: bool
    static_inputs: tuple[torch.Tensor, ...]
    static_output: torch.Tensor
    static_grad_output: torch.Tensor | None
    #: (name, parameter) in a fixed order
    leaves: tuple[tuple[str, torch.Tensor], ...]
    #: the graph's own gradient buffers in `leaves` order (see `grads`); held
    #: so an eager step setting `.grad = None` cannot free memory a replay
    #: still writes. Taken after the capture: a reference held during it would
    #: push `AccumulateGrad` off its in-place branch.
    static_grads: tuple[torch.Tensor, ...]
    pool: Any
    stream: torch.cuda.Stream
    #: what `capture_safe` (and this function) had to change
    notes: list[str] = field(default_factory=list)
    #: bytes of buffers this object allocated and holds (inputs, dY, grads)
    static_bytes: int = 0
    #: bytes the capture itself added to the allocator's reserved pool
    pool_bytes: int = 0
    #: warnings torch emitted during the capture (empty is the goal)
    capture_warnings: tuple[str, ...] = ()
    #: `execution_version(module)` at capture time: the recorded kernels embody that execution
    execution_version: int = 0

    @property
    def stale(self) -> bool:
        """True once the module's execution was changed after the capture (`set_execution`, the
        ``execution(...)`` context manager): the recorded graph then runs the OLD execution."""
        return execution_version(self.module) != self.execution_version

    def check_fresh(self) -> None:
        """Raise `CaptureError` (``reason="stale-execution"``) if the execution changed since the capture."""
        if self.stale:
            raise CaptureError(
                "stale-execution",
                "the module's execution (backend / precision / chunk_size / kernel_options) was changed "
                "after this graph was captured, so a replay would still run the old kernels; capture "
                "again with graphed_step(...) under the new execution")

    # -------------------------------------------------------------- api
    def copy_in(self, *inputs: torch.Tensor,
                grad_outputs: torch.Tensor | None = None) -> None:
        """Fill the static input buffers.  Shapes and dtypes must match."""
        if inputs:
            if len(inputs) != len(self.static_inputs):
                raise ValueError(
                    f"captured {len(self.static_inputs)} input(s), got "
                    f"{len(inputs)}")
            for dst, src in zip(self.static_inputs, inputs):
                if src is dst:
                    continue
                if tuple(src.shape) != tuple(dst.shape) or src.dtype != dst.dtype:
                    raise ValueError(
                        "a captured graph IS its shape: expected "
                        f"{tuple(dst.shape)}/{dst.dtype}, got "
                        f"{tuple(src.shape)}/{src.dtype}")
                with torch.no_grad():
                    dst.copy_(src)
        if grad_outputs is not None:
            if self.static_grad_output is None:
                raise ValueError("forward-only capture takes no grad_outputs")
            with torch.no_grad():
                self.static_grad_output.copy_(grad_outputs)

    def replay(self, *inputs: torch.Tensor,
               grad_outputs: torch.Tensor | None = None) -> torch.Tensor:
        """Copy in, replay, return the static output buffer.  Raises `CaptureError` if the module's
        execution changed after the capture (see `check_fresh`)."""
        self.check_fresh()
        self.copy_in(*inputs, grad_outputs=grad_outputs)
        self.graph.replay()
        return self.static_output

    __call__ = replay

    @property
    def grads(self) -> dict[str, torch.Tensor]:
        """`name -> the graph's gradient buffer` for this step.

        These are the buffers the recorded `AccumulateGrad` writes, not
        whatever `.grad` currently points at (an interleaved eager step may
        have rebound it).
        """
        return {n: g for (n, _), g in zip(self.leaves, self.static_grads)
                if g is not None}

    @property
    def memory(self) -> dict[str, int]:
        """The cost of being captured, in bytes."""
        return dict(static=self.static_bytes, graph_pool=self.pool_bytes,
                    total=self.static_bytes + self.pool_bytes)

    def __repr__(self) -> str:                             # pragma: no cover
        return (f"GraphedStep({type(self.module).__name__}, "
                f"{'fwd_bwd' if self.fwd_bwd else 'fwd'}, "
                f"inputs={[tuple(t.shape) for t in self.static_inputs]}, "
                f"static={self.static_bytes / 2**20:.1f} MiB, "
                f"pool={self.pool_bytes / 2**20:.1f} MiB)")


# ── helpers ──────────────────────────────────────────────────────────
def _first(y):
    """The tensor a mixer's forward returns (a bare tensor, or y of (y, state))."""
    if isinstance(y, torch.Tensor):
        return y
    if isinstance(y, (tuple, list)):
        if len(y) and isinstance(y[0], torch.Tensor):
            return y[0]
    raise CaptureError(
        "output", f"the module returned {type(y).__name__}, not a tensor "
        "(a `state=`/`return_state=` continuation is outside the capture "
        "envelope -- the carry is a python dict whose identity changes "
        "per call)")


def _nbytes(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.numel() * t.element_size()


def _check_cuda(module: nn.Module, inputs: Sequence[torch.Tensor]) -> torch.device:
    if not torch.cuda.is_available():
        raise CaptureError("no-cuda", "CUDA graphs need a CUDA device; this "
                                      "process has none")
    devs = {t.device for t in inputs if isinstance(t, torch.Tensor)}
    devs |= {p.device for p in module.parameters()}
    devs |= {b.device for b in module.buffers()}
    non_cuda = sorted({str(d) for d in devs if d.type != "cuda"})
    if non_cuda:
        raise CaptureError(
            "not-on-cuda",
            "every parameter, buffer and input must live on one CUDA "
            f"device; found {non_cuda}")
    if len({d.index for d in devs}) > 1:
        raise CaptureError("multi-device",
                           "the captured region must run on ONE device; found "
                           f"{sorted(str(d) for d in devs)}")
    return next(iter(devs))


def _classify(exc: BaseException) -> CaptureError:
    """Turn a raw capture failure into a reason a report can quote."""
    msg = str(exc)
    low = msg.lower()
    if any(m.lower() in low for m in _SYNC_MARKERS):
        return CaptureError(
            "host-sync",
            "something inside the captured region talked to the host (an "
            "`.item()`, `.tolist()`, `.cpu()`, a `torch.cuda.synchronize()` "
            "or a shape read off a device tensor).  Every such read has to "
            "be hoisted out of the per-step path or memoised before the "
            "capture -- see `orbit_ssm.graphs.capture_safe`.  torch said: "
            + msg, exc)
    if "out of memory" in low:
        return CaptureError("oom", "the capture ran out of memory; a graph "
                                   "pool is allocated on top of the live "
                                   "activations.  torch said: " + msg, exc)
    if "autocast" in low and "cach" in low:
        return CaptureError("autocast-cache", msg, exc)
    return CaptureError("capture-failed", msg, exc)


# ── the entry point ──────────────────────────────────────────────────
def graphed_step(module: nn.Module,
                 example_inputs: torch.Tensor | Sequence[torch.Tensor],
                 *,
                 fwd_bwd: bool = True,
                 amp: Any = None,
                 warmup: int = 11,
                 grad_outputs: torch.Tensor | None = None,
                 params: Iterable[torch.Tensor] | None = None,
                 pool: Any = None,
                 stream: torch.cuda.Stream | None = None,
                 seed: int | None = 0,
                 ) -> GraphedStep:
    """Capture one training step of `module` as a CUDA graph.

    Parameters
    ----------
    module
        Any `nn.Module` whose forward is shape-static and free of host
        syncs; `SSDMimoSSM`, `HybridMambaSSM` and the official Mamba-1/2/3
        layers are tested (capture, replay and gradients against eager).
    example_inputs
        The positional arguments of the forward, as CUDA tensors.  They
        are COPIED into static buffers; the originals are not retained
        and are never read again.
    fwd_bwd
        `True` (default) captures forward + `y.backward(dY)` with a static dY;
        `False` captures the forward alone under `no_grad`.
    amp
        `None`, a precision name (``"bf16"``/``"fp16"``/``"fp32"``), a
        `torch.dtype`, or a zero-argument context-manager factory.  A
        factory that leaves autocast caching on is refused (module docstring,
        item 2).
    warmup
        Warm-up iterations on the capture stream before recording (default 11,
        at least 1). They absorb Triton compilation and autotune, and create
        every `AccumulateGrad` node on the capture stream.
    grad_outputs
        The dY to record.  Defaults to `randn_like(y)`; pass one for a
        reproducible test.
    params
        The leaves whose gradients the step must produce.  Defaults to
        every `module.parameter()` with `requires_grad`, plus every
        input with `requires_grad`.
    pool, stream
        A graph memory pool to share (`graph.pool()` of an earlier
        capture) and the stream to warm up and capture on.  Both default to
        fresh ones owned by this call.
    seed
        Seeds `torch.manual_seed` before the dY is drawn, so two captures
        of two modules see the same dY.  `None` leaves the RNG alone.

    Returns
    -------
    GraphedStep

    Raises
    ------
    CaptureError
        With a `reason` tag -- ``no-cuda``, ``not-on-cuda``,
        ``multi-device``, ``output``, ``dynamic-shape``, ``no-grad``,
        ``autocast-cache``, ``host-sync``, ``oom``, ``unstable-grad`` or
        ``capture-failed`` -- and an explanation.  Never a silently eager
        fallback: a benchmark row must be able to record the skip.
    """
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    example_inputs = tuple(example_inputs)
    if not example_inputs:
        raise CaptureError("output", "a capture needs at least one input "
                                     "tensor to copy a step into")
    dev = _check_cuda(module, example_inputs)
    amp_ctx = _amp_factory(amp)

    # Static input buffers; requires_grad is kept so the input's gradient is
    # part of the captured step as in the eager one.
    static_inputs = tuple(
        t.detach().clone().requires_grad_(t.requires_grad)
        for t in example_inputs)

    leaf_list: list[tuple[str, torch.Tensor]] = []
    if params is None:
        leaf_list += [(n, p) for n, p in module.named_parameters()
                      if p.requires_grad]
        leaf_list += [(f"input[{i}]", t) for i, t in enumerate(static_inputs)
                      if t.requires_grad]
    else:
        leaf_list += [(f"leaf[{i}]", p) for i, p in enumerate(params)
                      if p.requires_grad]
    leaves = tuple(leaf_list)

    notes: list[str] = []
    stream = stream if stream is not None else torch.cuda.Stream(device=dev)

    with capture_safe(module) as safety_notes:
        notes += safety_notes

        # ---- warm-up, on the capture stream (off the default stream, and
        # AccumulateGrad nodes must be created on the capture stream) ----
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
                            raise CaptureError(
                                "no-grad",
                                "the forward produced a graph-less output, so "
                                "there is no backward to capture (no leaf "
                                "requires grad)")
                        if dy_ref is None:
                            if seed is not None:
                                torch.manual_seed(seed)
                            # clone: this is a static buffer and must not alias the caller's tensor
                            dy_ref = (torch.randn_like(y)
                                      if grad_outputs is None else
                                      grad_outputs.detach().clone().to(
                                          device=y.device, dtype=y.dtype))
                        y.backward(dy_ref)
                    else:
                        with torch.no_grad(), amp_ctx():
                            _check_autocast_cache()
                            y = _first(module(*static_inputs))
                        shapes.add((tuple(y.shape), y.dtype))
                    del y
        except CaptureError:
            raise
        except Exception as exc:                            # noqa: BLE001
            raise _classify(exc) from exc
        torch.cuda.current_stream(dev).wait_stream(stream)
        torch.cuda.synchronize(dev)

        if len(shapes) != 1:
            raise CaptureError(
                "dynamic-shape",
                "the forward produced more than one output shape/dtype over "
                f"the warm-up iterations ({sorted(map(str, shapes))}); a "
                "captured graph IS its shape")

        # ---- static gradient buffers, allocated OUTSIDE the graph ---
        if fwd_bwd:
            for _, p in leaves:
                p.grad = torch.zeros_like(p)
        static_dy = dy_ref

        # ---- capture -----------------------------------------------
        # pool_bytes is measured between two empty_cache() calls, so the delta
        # is the graph's private pool, not leftover warm-up cache.
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved(dev)
        graph = torch.cuda.CUDAGraph()
        kw = {} if pool is None else {"pool": pool}
        try:
            with warnings.catch_warnings(record=True) as caught, \
                    validation_off() as vnotes:
                warnings.simplefilter("always")
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
        except Exception as exc:                            # noqa: BLE001
            raise _classify(exc) from exc
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        pool_bytes = max(0, torch.cuda.memory_reserved(dev) - before)

        cap_warnings = tuple(sorted({str(w.message).split("\n")[0][:200]
                                     for w in caught}))
        acc = [w for w in cap_warnings if "AccumulateGrad" in w]
        if acc:
            notes.append("torch still warned about the AccumulateGrad "
                         "stream: " + acc[0])

    # ---- post-capture verification ---------------------------------
    # A replay must write the same addresses every time, or `.grads` goes
    # stale: replay once and compare .grad pointers.
    static_grads: tuple[torch.Tensor, ...] = ()
    if fwd_bwd:
        ptrs = [None if p.grad is None else p.grad.data_ptr()
                for _, p in leaves]
        graph.replay()
        torch.cuda.synchronize(dev)
        if [None if p.grad is None else p.grad.data_ptr()
                for _, p in leaves] != ptrs:
            raise CaptureError(
                "unstable-grad",
                "a replay re-bound at least one `.grad` to a new address, so "
                "the gradient buffers a caller holds would go stale after "
                "one step")
        static_grads = tuple(p.grad for _, p in leaves)

    static_bytes = (sum(_nbytes(t) for t in static_inputs)
                    + _nbytes(static_dy)
                    + sum(_nbytes(g) for g in static_grads))

    return GraphedStep(
        module=module, graph=graph, fwd_bwd=fwd_bwd,
        static_inputs=static_inputs, static_output=static_out,
        static_grad_output=static_dy, leaves=leaves,
        static_grads=static_grads,
        pool=graph.pool(), stream=stream, notes=notes,
        static_bytes=static_bytes, pool_bytes=pool_bytes,
        capture_warnings=cap_warnings, execution_version=execution_version(module))


def _check_autocast_cache() -> None:
    """Refuse an autocast that memoises parameter casts.

    Called inside the `amp` context. A cached cast is produced on an earlier
    iteration, so the capture would record a read no replay refreshes
    (stale weights after the first optimizer step).
    """
    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise CaptureError(
            "autocast-cache",
            "the autocast context must be built with `cache_enabled=False` "
            "(use `orbit_ssm.graphs.autocast`): with caching on, the bf16 "
            "cast of every fp32 parameter is memoised across iterations, so "
            "the graph would record a READ of a warm-up tensor and every "
            "replay after an optimizer step would use stale weights")
