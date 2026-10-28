"""THE measurement core — one implementation, every benchmark path.

Everything in `benchmarks/` that produces a number produces it here:
`bench_kernels.py` and `bench_ssms.py` (through
`harness.run_cells`, which is record assembly around this module's
group driver).  A second
timing loop anywhere else in the tree is a bug — the ad-hoc probe
scripts that each had one were folded in here and
deleted.

WHAT THIS MODULE OWNS
---------------------
1. **The protocol** — interleaved round-robin over the arms of ONE
   comparison group inside ONE process, a fixed 4096^3 sgemm canary at
   the head of every round, per-round samples reduced by median (the
   quoted number) and min, and `canary_spread` on every record.  On the
   shared login GH200 the node drifts 1.3-2.3x mid-sweep, so a ratio
   between two arms that were NOT interleaved is not evidence.
2. **The budgets** (`Budget`) — how much wall clock one cell may spend,
   and whether it pays the extra profiler pass.
3. **The precision columns** (`ieee`/`tf32`/`bf16`, `IO_DTYPE`,
   `KERNEL_PRECISION`, `precision_ctx`, `amp_ctx`).
4. **The metrics.**  Each is a named, individually switchable pass that
   runs AFTER the timed rounds so it cannot perturb them:

   ``wall``       CUDA-event (or `triton.testing.do_bench`) time per
                  call of the whole timed closure.  Always on.
   ``cuda_self``  `torch.profiler` GPU self time, summed and split per
                  kernel, with MIN-OVER-ROUNDS attribution per kernel
                  (G32: on a contended card interference can only ADD
                  time to a kernel, so the least contaminated estimate
                  of a kernel is its minimum over rounds and the sum of
                  those minima is the least contaminated estimate of the
                  step).  This is the GPU-work column G20 rules the
                  parity tables must print beside the wall column.
   ``launches``   CUDA kernel/memcpy/memset launch counts per call.
   ``host``       host-enqueue ms per call: the same loop NOT
                  synchronised, i.e. python + ATen dispatch + the launch
                  calls.  `wall <= host * 1.05` means the cell is
                  HOST-ENQUEUE BOUND and its wall ratio prices the
                  enqueue path, not the kernels (guarded by a
                  self-test of the harness).
   ``memory``     per-cell allocator delta and the process-global peak
                  (provenance only).
   ``roofline``   the analytic bytes/FLOP model for a cell,
                  divided by measured device peaks.
   ``canary``     always on; the contention gate.

5. **The capture regime** (`captured_closure`) — one CUDA graph per
   fwd(+bwd) step via `orbit_ssm.graphs.graphed_step` when that module
   is present.  A captured number may only ever be ratioed against
   another CAPTURED number; the pairing code
   refuses to mix, and a self-test of the harness guards it.

WHAT IS EXCLUDED, DELIBERATELY
------------------------------
* build/compile time (every cell is built and called once before any
  round), * the first call of every round-robin round (warmup inside
  each timer), * `torch.cuda.max_memory_allocated()` as an absolute (it
  is process-global; only the delta is per-cell), * anything measured in
  a different process than the arm it is compared to.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


# ── the canary ───────────────────────────────────────────────────────
CANARY_DIM = 4096          # fixed by the protocol; do not tune per run
CANARY_ITERS = 20
CANARY_SPREAD_FLAG = 1.15  # above this the round is contended -> flagged

_CANARY_OPERANDS: dict[str, torch.Tensor] = {}


def canary_ms(device: str = "cuda") -> float:
    """One 4096^3 sgemm block, wall ms per matmul.

    Same operands every round (allocated once) so the number measures the
    machine, not the allocator, and pinned to true fp32 (tensor cores
    off) so it is the same probe in every round regardless of which
    precision column the surrounding cells are in — a canary that moved
    with the process state would measure us, not the node.
    """
    if device not in _CANARY_OPERANDS:
        a = torch.randn(CANARY_DIM, CANARY_DIM, device=device)
        b = torch.randn(CANARY_DIM, CANARY_DIM, device=device)
        _CANARY_OPERANDS[device] = (a, b)
    a, b = _CANARY_OPERANDS[device]
    with precision_ctx("ieee"):
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(CANARY_ITERS):
            a @ b
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / CANARY_ITERS * 1e3


# ── budgets ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Budget:
    """How much wall clock one cell is allowed to consume.

    `do_bench_rep` is a TIME budget in ms (triton's semantics), not an
    iteration count.
    """

    name: str
    rounds: int
    do_bench_warmup: int
    do_bench_rep: int
    ev_warmup: int
    ev_iters: int
    # Whether this budget pays the extra `ms_cuda_self` `torch.profiler`
    # pass (`bench_kernels._post_cell`, the `_SELF_CUDA_GRIDS` kernel-
    # parity grids only) -- one BUDGET-INVARIANT extra call per cell,
    # the single largest per-cell cost the parity grids carry.  Default True everywhere (unchanged
    # behavior for every existing budget).  Dropping
    # the ms_cuda_self GPU-time column from the compact budget is a
    # one-line flip: `PAPER_BUDGET = dataclasses.replace(PAPER_BUDGET,
    # cuda_self=False)` just below -- `bench_kernels._post_cell` and
    # `paper_shard_plan`'s cost model both read this same field, so the
    # flip changes what actually RUNS and what the shard plan PREDICTS
    # together.
    cuda_self: bool = True


# Three rounds, not two: with an even round count the "median" is the mean
# of two samples, and a single contended round then silently doubles the
# quoted number instead of being outvoted.
SMOKE_BUDGET = Budget("smoke", rounds=3, do_bench_warmup=10, do_bench_rep=50,
                      ev_warmup=3, ev_iters=10)
FULL_BUDGET = Budget("full", rounds=3, do_bench_warmup=50, do_bench_rep=500,
                     ev_warmup=8, ev_iters=30)
DEEP_BUDGET = Budget("deep", rounds=5, do_bench_warmup=50, do_bench_rep=500,
                     ev_warmup=10, ev_iters=50)
# The compact paper protocol: the SAME paper numbers in
# ~10 min of node time instead of 40-110 min.  Warmup (do_bench_warmup,
# ev_warmup) is KEPT AT FULL_BUDGET's values on purpose -- it is what buys
# the autotuner its look at the kernel before anything is timed, and a
# shrunk warmup would put an untuned config into the ratio, not just a
# noisier one.  What shrinks is the MEASURED window: do_bench_rep 500ms ->
# 150ms, ev_iters 30 -> 10, rounds 3 -> 2.  Two rounds, not three, is a
# deliberate departure from the odd-round-count rule above (a single
# contended round can silently double a 2-round median instead of being
# outvoted) -- that margin is traded for wall clock, leaning on
# the UNCHANGED canary-spread gate (CANARY_SPREAD_FLAG = 1.15x) to catch
# the contention case instead of a third round.  The resulting deviation was
# measured against a full-budget run to confirm the trade is safe.
#
# Back to an ODD round count --
# THREE rounds, so `ms_median` is a true median of three per-round values
# (two rounds made it the mean of two), with the arm order counterbalanced
# across rounds (`round_order`: ABBA...), and the whole-run canary spread
# gated by the caller (job script, <= 1.15 else retry) rather than per group.
PAPER_BUDGET = Budget("paper", rounds=3, do_bench_warmup=50, do_bench_rep=150,
                      ev_warmup=8, ev_iters=10, cuda_self=True)
BUDGETS = {b.name: b for b in (SMOKE_BUDGET, FULL_BUDGET, DEEP_BUDGET,
                               PAPER_BUDGET)}

# ── precision columns ────────────────────────────────────────────────
PRECISIONS = ("ieee", "tf32", "bf16")

IO_DTYPE = {"ieee": torch.float32, "tf32": torch.float32,
            "bf16": torch.bfloat16}
# The `bf16` column maps to the kernel's `"io"` policy, NOT to `"tf32"`.
# It used to map to `"tf32"`, which gave the column bf16 OPERANDS but a
# tf32 MMA -- a configuration nothing ships, and not the bf16 path the
# shipped default (`precision="io"`, D1b) actually takes, which rounds
# the `tl.dot` operands to the IO dtype (`mma_io_enabled`,
# fused_fwd_kernel).  The value spelled here must be `"io"`.  Since bf16 MMA
# is FASTER than tf32 MMA on both architectures, the old mapping made our
# own bf16 rows understate the kernel.  Tables collected before this change
# carry the old meaning; they are not wrong as "bf16 operands, tf32 MMA"
# measurements, but they do not price what we ship and are not comparable
# with rows collected after this change.
KERNEL_PRECISION = {"ieee": "ieee", "tf32": "tf32", "bf16": "io"}
# The column label the paper's tables use for each precision level.
PAPER_LABEL = {"ieee": "fp32", "tf32": "tf32", "bf16": "bf16"}

# Module- and model-level rows keep fp32 PARAMETERS in every column and
# reach bf16 through autocast (see the module docstring): a whole-module
# `.to(bfloat16)` is not a configuration anybody trains, and it makes
# every `residual_in_fp32=True` baseline raise instead of being measured.
MODULE_PARAM_DTYPE = torch.float32


def amp_ctx(precision: str, device_type: str = "cuda"
            ) -> Callable[[], Any]:
    """Context-manager FACTORY for a module-level row's bf16 column.

    Returns a zero-argument callable so the closure can enter a fresh
    context on every timed iteration.  `ieee`/`tf32` get a null context:
    their operands are already fp32 and `precision_ctx` owns the
    tensor-core allowance.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision != "bf16":
        return contextlib.nullcontext
    return lambda: torch.autocast(device_type=device_type,
                                  dtype=torch.bfloat16)


@contextlib.contextmanager
def precision_ctx(precision: str):
    """Put the process in the global state that `precision` names.

    Only the tensor-core allowance is global; the operand dtype and the
    kernel's own `precision=` kwarg are per-cell and come from
    `IO_DTYPE` / `KERNEL_PRECISION`.  `ieee` turns tf32 OFF for the torch
    matmuls too, so a baseline built out of `torch.matmul` is priced on
    the same exactness path our kernel is.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    allow = precision != "ieee"
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cudnn


# ── the round-robin driver ───────────────────────────────────────────
def _time_do_bench(fn, budget: Budget) -> float:
    import triton.testing
    # per-round statistic = the MEDIAN of do_bench's own per-call samples
    # (its default is the mean); the record's `ms_median` is then the median
    # of these per-round values over the rounds
    return float(triton.testing.do_bench(fn, warmup=budget.do_bench_warmup,
                                         rep=budget.do_bench_rep,
                                         return_mode="median"))


def round_order(cells: Sequence, r: int) -> list:
    """Counterbalanced arm order: round r runs the group's cells forward when r
    is even and REVERSED when r is odd (ABBA over consecutive rounds), so no
    arm is systematically first after the canary / last before the next one."""
    cells = list(cells)
    return cells if r % 2 == 0 else cells[::-1]


def _time_events(fn, budget: Budget) -> float:
    for _ in range(budget.ev_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(budget.ev_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / budget.ev_iters


def _time_one(fn, tier: str, budget: Budget) -> float:
    return (_time_do_bench(fn, budget) if tier == "do_bench"
            else _time_events(fn, budget))


def _drop_canary_operands() -> None:
    """Free the canary's permanently-resident 4096^2 operands (128 MiB).

    `canary_ms` re-allocates them on its next call, so this costs one
    pair of `randn`s per group.  It is done before the memory pass
    because `torch.cuda.max_memory_allocated()` is process-global: with
    the operands resident, every cell's raw peak reads ~183 MiB whatever
    the cell actually costs.  `mem_delta_bytes` is right either way (the
    operands sit in the baseline as well as in the peak) — this only
    makes the raw provenance figure interpretable too.
    """
    _CANARY_OPERANDS.clear()
    torch.cuda.empty_cache()


def _peak_memory(fn, device: str) -> tuple[int, int]:
    """(peak bytes during one call, allocator baseline before it).

    The peak is PROCESS-GLOBAL, so only `peak - baseline` is a per-cell
    number; `run_cells` records that as `mem_delta_bytes` and keeps the
    raw peak as provenance.  A closure with a `pre_call` attribute (e.g.
    one that frees gradients an earlier cell left on shared leaves) has it
    run BEFORE the baseline is read, so bytes the call itself would free
    first do not deflate `peak - baseline`.
    """
    pre = getattr(fn, "pre_call", None)
    if pre is not None:
        pre()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated()), int(base)


def _build_cell(cell: Cell, errors: dict, progress: bool, quiet: bool = False):
    """Build one cell's closure, recording (not raising) a failure."""
    key = cell.key()
    err = None
    try:
        with precision_ctx(cell.precision):
            fn = cell.build()
            output = fn()                         # first call: compile, then validate
            def tensors(value):
                if isinstance(value, dict):
                    for child in value.values():
                        yield from tensors(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        yield from tensors(child)
                elif torch.is_tensor(value):
                    yield value
            values = list(tensors(output))
            values += list(getattr(fn, "validation_tensors", lambda: [])())
            for value in values:
                if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all().item():
                    raise FloatingPointError("nonfinite output/state/gradient during warmup")
            if cell.tags.get("device", "cuda").split(":")[0] == "cuda":
                torch.cuda.synchronize()
    except Exception as exc:                      # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    if err is not None:
        # Freed HERE, not inside the `except`: while `exc` is in scope its
        # traceback pins the frames of the half-built module, so an
        # `empty_cache()` there returns almost nothing.  This is the OOM
        # recovery path — the next cell must start from a clean allocator.
        if cell.tags.get("device", "cuda").split(":")[0] == "cuda":
            torch.cuda.empty_cache()
        errors.setdefault(key, err)
        if progress and not quiet:
            print(f"  SKIP {key}\n       {errors[key][:200]}", flush=True)
        return None
    if progress and not quiet:
        print(f"  built {key}", flush=True)
    return fn


def warm_cells(cells: Sequence[Cell]) -> tuple[int, int]:
    """B8 `--warm-only`: build every cell and call its closure ONCE, no
    timing, no JSONL.  `_build_cell` already does exactly "build + one
    call, synchronize" per cell -- the same first call any real run pays
    the Triton compile for -- so this is that helper run over a whole
    cell list with each cell's model/closure freed before the next one
    starts (a warm pass never needs more than one cell's memory
    resident at a time, unlike a real comparison group).  Returns
    `(n_warmed, n_failed)`; a failing cell is recorded and skipped, not
    raised -- warming is best-effort, not a correctness gate."""
    errors: dict[str, str] = {}
    warmed = 0
    for cell in cells:
        fn = _build_cell(cell, errors, progress=True)
        if fn is not None:
            warmed += 1
        del fn
        torch.cuda.empty_cache()
    return warmed, len(errors)


def _memory_and_extras(cells, get_fn, post_cell, device, progress):
    """The post-timing pass: per-cell memory delta and the `post_cell` hook."""
    _drop_canary_operands()
    peaks: dict[str, tuple[int, int]] = {}
    extras: dict[str, dict] = {}
    for cell in cells:
        key = cell.key()
        fn = get_fn(cell)
        if fn is None:
            continue
        try:
            with precision_ctx(cell.precision):
                peaks[key] = _peak_memory(fn, device)
                if post_cell is not None:
                    extras[key] = post_cell(cell, fn) or {}
        except Exception as exc:                   # noqa: BLE001
            # never swallowed: the cell's record carries the failure (refused downstream)
            extras[key] = dict(extras.get(key) or {},
                               error=f"post-timing pass failed: {type(exc).__name__}: {exc}"[:500])
        finally:
            del fn
            torch.cuda.empty_cache()
    return peaks, extras


def _run_group_resident(group: str, cells: Sequence[Cell], budget: Budget,
                        device: str, post_cell,
                        progress: bool, warm_start: bool = False) -> tuple[dict, list[float]]:
    """Build the whole group, round-robin it, measure, FREE it.

    The default: every arm stays resident so a round touches them
    back to back with nothing rebuilt in between.  `warm_start`: one
    untimed canary and one untimed call of every built cell right before
    round 1, so a short group's first round does not start on clocks
    that idled through the (host-side) builds.
    """
    built: dict[str, Callable[[], Any]] = {}
    errors: dict[str, str] = {}
    for cell in cells:
        fn = _build_cell(cell, errors, progress)
        if fn is not None:
            built[cell.key()] = fn
    if warm_start:
        canary_ms(device)
        for cell in cells:
            fn = built.get(cell.key())
            if fn is None:
                continue
            try:
                with precision_ctx(cell.precision):
                    fn()
            except Exception as exc:              # noqa: BLE001
                errors.setdefault(cell.key(), f"{type(exc).__name__}: {exc}")
                built.pop(cell.key(), None)
        torch.cuda.synchronize()

    samples: dict[str, list[float]] = {k: [] for k in built}
    canaries: list[float] = []
    for r in range(budget.rounds):
        canaries.append(canary_ms(device))
        for cell in round_order(cells, r):
            key = cell.key()
            fn = built.get(key)
            if fn is None:
                continue
            try:
                with precision_ctx(cell.precision):
                    samples[key].append(_time_one(fn, cell.resolved_tier,
                                                  budget))
            except Exception as exc:              # noqa: BLE001
                errors.setdefault(key, f"{type(exc).__name__}: {exc}")
                samples.pop(key, None)
                built.pop(key, None)
        if progress:
            print(f"  [{group}] round {r + 1}/{budget.rounds} canary "
                  f"{canaries[-1]:.3f} ms", flush=True)

    peaks, extras = _memory_and_extras(
        cells, lambda c: built.get(c.key()), post_cell, device, progress)

    # free the group's tensors before the next group is built
    built.clear()
    torch.cuda.empty_cache()
    return dict(samples=samples, errors=errors, peaks=peaks,
                extras=extras), canaries


def _run_group_rebuild(group: str, cells: Sequence[Cell], budget: Budget,
                       device: str, post_cell,
                       progress: bool, warm_start: bool = False) -> tuple[dict, list[float]]:
    """Round-robin with ONE cell resident at a time, rebuilt every round.

    Same interleaving — arms are still measured back to back inside a
    round, against the same canary — but residency is one cell instead of
    the whole group.  This is what makes the model tier runnable: a
    `model_440m` LM group is 8 whole language models, each with its
    gradients populated by the build-time call, and ONE of them needs
    9-31 GiB of activations at the LM spine (measured).  A build OOM is only a recorded
    skip, so the resident path would lose rows silently rather than
    crash.

    The price is a rebuild per cell per round; the kernels themselves are
    already compiled in-process after round 0, so it is module
    construction, not compilation.
    """
    samples: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    dead: set[str] = set()
    canaries: list[float] = []
    if warm_start:                                # (each cell's build call warms it per round)
        canary_ms(device)
    for r in range(budget.rounds):
        canaries.append(canary_ms(device))
        for cell in round_order(cells, r):
            key = cell.key()
            if key in dead:
                continue
            fn = _build_cell(cell, errors, progress, quiet=(r > 0))
            if fn is None:
                dead.add(key)
                continue
            try:
                with precision_ctx(cell.precision):
                    samples.setdefault(key, []).append(
                        _time_one(fn, cell.resolved_tier, budget))
            except Exception as exc:              # noqa: BLE001
                errors.setdefault(key, f"{type(exc).__name__}: {exc}")
                samples.pop(key, None)
                dead.add(key)
            finally:
                del fn
                torch.cuda.empty_cache()
        if progress:
            print(f"  [{group}] round {r + 1}/{budget.rounds} canary "
                  f"{canaries[-1]:.3f} ms (rebuild mode)", flush=True)

    # memory + post_cell: build once more, one cell at a time
    def _one(cell: Cell):
        if cell.key() in dead:
            return None
        return _build_cell(cell, errors, progress, quiet=True)

    peaks, extras = _memory_and_extras(cells, _one, post_cell, device,
                                       progress)
    torch.cuda.empty_cache()
    return dict(samples=samples, errors=errors, peaks=peaks,
                extras=extras), canaries


GROUP_MODES = ("auto", "resident", "rebuild")


def _resolve_group_mode(mode: str, cells: Sequence[Cell]) -> str:
    """`auto` = rebuild the model tier, keep everything else resident."""
    if mode not in GROUP_MODES:
        raise ValueError(f"group mode must be one of {GROUP_MODES}, got {mode!r}")
    if mode != "auto":
        return mode
    return "rebuild" if any(c.family == "model" for c in cells) else "resident"


def _run_group_cpu(cells, budget, progress, group_mode="auto"):
    """Host wall-clock round robin; GPU contention/memory metrics do not apply."""
    errors, samples, built = {}, {}, {}
    rebuild = _resolve_group_mode(group_mode, cells) == "rebuild"
    if not rebuild:
        for cell in cells:
            fn = _build_cell(cell, errors, progress)
            if fn is not None:
                built[cell.key()] = fn
    for _ in range(budget.rounds):
        for cell in cells:
            key = cell.key()
            if key in errors:
                continue
            fn = _build_cell(cell, errors, progress) if rebuild else built.get(key)
            if fn is None:
                continue
            try:
                with precision_ctx(cell.precision):
                    for _ in range(budget.ev_warmup):
                        fn()
                    start = time.perf_counter()
                    for _ in range(budget.ev_iters):
                        fn()
                    samples.setdefault(key, []).append(
                        (time.perf_counter() - start) * 1000 / budget.ev_iters)
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
                built.pop(key, None)
                samples.pop(key, None)
            finally:
                del fn
    return dict(samples=samples, errors=errors, peaks={}, extras={}), []


def _run_group(group: str, cells: Sequence[Cell], budget: Budget,
               device: str, post_cell, progress: bool,
               group_mode: str = "auto",
               warm_start: bool = False) -> tuple[dict, list[float]]:
    """Build, warm, round-robin, measure and FREE one comparison group.
    `warm_start` (GPU): an untimed canary (+ one call per resident cell)
    right before round 1."""
    if torch.device(device).type == "cpu":
        return _run_group_cpu(cells, budget, progress, group_mode)
    mode = _resolve_group_mode(group_mode, cells)
    fn = _run_group_rebuild if mode == "rebuild" else _run_group_resident
    return fn(group, cells, budget, device, post_cell, progress, warm_start)



# ═════════════════════════════════════════════════════════════════════
# public aliases — the names outside this module use
# ═════════════════════════════════════════════════════════════════════
# `_time_one` / `_build_cell` / `_run_group` / `_peak_memory` above are
# the historical (harness-private) spellings, kept so nothing that
# already imports them breaks; these are the names new code uses.
time_one = _time_one
build_cell = _build_cell
run_group = _run_group
peak_memory = _peak_memory
drop_canary_operands = _drop_canary_operands


# ═════════════════════════════════════════════════════════════════════
# the metric passes
# ═════════════════════════════════════════════════════════════════════
# Every one of these runs AFTER a group's timed rounds (`run_cells`'
# `post_cell` hook), so no metric can perturb the number it annotates.
METRICS = ("wall", "cuda_self", "launches", "host", "memory", "canary",
           "roofline", "launch_meta", "kernel_resources")
#: metrics that are not optional: `wall` IS the timed rounds and
#: `canary` IS the contention gate, so a config cannot switch them off.
ALWAYS_ON = ("wall", "canary")


def _profile_cuda(fn: Callable[[], Any], iters: int):
    """One `torch.profiler` CUDA-only pass over `iters` calls.

    Returns `{kernel name: dict(ms, calls)}` plus the launch-kind split.
    CUDA-only activities on purpose: adding `ProfilerActivity.CPU` costs
    a large multiple of the pass and changes what the GPU sees.
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per_kernel: dict[str, dict] = {}
    launches: dict[str, float] = defaultdict(float)
    for ev in prof.key_averages():
        cu = float(getattr(ev, "self_device_time_total", 0.0) or 0.0)
        if cu <= 0:
            continue
        low = ev.key.lower()
        kind = ("memcpy" if "memcpy" in low else
                "memset" if "memset" in low else "kernel")
        launches[kind] += ev.count / iters
        row = per_kernel.setdefault(ev.key, dict(ms=0.0, calls=0.0))
        row["ms"] += cu / iters / 1e3
        row["calls"] += ev.count / iters
    return per_kernel, dict(launches)


def merge_min_over_rounds(rounds: Sequence[dict]) -> dict:
    """Fold N profiler rounds into one, PER KERNEL, by the MINIMUM (G32).

    Taking the best ROUND needs a round in which nothing else ran on the
    card; on the shared login GH200 that can simply not happen.
    Interference can only ADD time to a kernel, so the least
    contaminated estimate of each kernel is its minimum over the rounds,
    and the sum of those minima is the least contaminated estimate of
    the call.  Launch counts are deterministic here (same code path
    every round); a kernel whose count differs between rounds is
    reported in `calls_vary` rather than silently averaged.

    Adapted from `profile_mixer.merge_attrib`, which this replaces for
    every path that is not the standalone mixer profiler.
    """
    if not rounds:
        return dict(ms_cuda_self=None, kernels={}, round_totals=[],
                    calls_vary=[])
    best: dict[str, dict] = {}
    seen_calls: dict[str, set] = defaultdict(set)
    totals: list[float] = []
    for rnd in rounds:
        totals.append(sum(v["ms"] for v in rnd.values()))
        for name, row in rnd.items():
            seen_calls[name].add(round(row["calls"], 3))
            if name not in best or row["ms"] < best[name]["ms"]:
                best[name] = dict(row)
    return dict(
        ms_cuda_self=sum(v["ms"] for v in best.values()),
        kernels={k: v for k, v in sorted(best.items(),
                                         key=lambda kv: -kv[1]["ms"])},
        round_totals=totals,
        calls_vary=sorted(k for k, v in seen_calls.items() if len(v) > 1),
    )


def cuda_self(fn: Callable[[], Any], *, rounds: int = 3, iters: int = 1
              ) -> dict:
    """GPU self time per call, per kernel, min over `rounds` (G32).

    `iters` calls are profiled per round.  The returned `ms_cuda_self`
    is the number G20 rules the parity tables print BESIDE the wall
    column: for an autograd Function that does host-side torch work
    around its launches (official Mamba-2's fused
    `mamba_split_conv1d_scan_combined`) the two differ by that work, and
    two kernels can rank differently under each.
    """
    per_round = []
    launches: dict[str, float] = {}
    for _ in range(max(1, rounds)):
        pk, la = _profile_cuda(fn, iters)
        per_round.append(pk)
        launches = la
    out = merge_min_over_rounds(per_round)
    out["launches"] = launches
    out["n_launches"] = sum(launches.values())
    return out


def host_enqueue_ms(fn: Callable[[], Any], *, warmup: int = 6, iters: int = 10
                    ) -> tuple[float, float]:
    """`(wall ms/call from CUDA events, host ms/call of the enqueue)`.

    The host number is the wall clock of `iters` calls WITHOUT a
    synchronise: `.backward()` returns once the autograd engine has
    walked the graph, so what it measures is python + ATen dispatch +
    the CUDA launch calls with the GPU still draining behind it.  When
    the two are equal the cell is host-enqueue bound and its wall ratio
    is a ratio of enqueue paths, not of kernels (G32; the flag is
    `host_bound` on the record).

    Lifted verbatim in method from `profile_mixer.wall_and_host_ms`.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    t0 = time.perf_counter()
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    t_host = time.perf_counter() - t0
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters, t_host * 1e3 / iters


#: the cell is HOST-ENQUEUE BOUND when its GPU span and its enqueue span
#: agree to within this factor: the GPU finished everything it was given
#: as fast as the host could give it, so the wall number prices python +
#: ATen dispatch + the launch calls (G32 measured both mixers here at
#: both paper cells).  A GPU-bound cell has wall well ABOVE host.
HOST_BOUND_RATIO = 1.05


def host_bound(ms_wall: float | None, ms_host: float | None) -> bool | None:
    """Is this cell's wall time the enqueue path rather than the kernels?"""
    if ms_wall is None or ms_host is None or ms_host <= 0:
        return None
    # One-sided on purpose.  A GPU-BOUND cell has a GPU span well ABOVE
    # its enqueue span (the launches are issued long before the kernels
    # finish: 0.170 ms vs 0.026 ms for a 4096^3 GEMM).  Anything else —
    # the two equal, or the enqueue even outrunning the GPU span because
    # the host loop also pays python around the launches — is the GPU
    # waiting on the host, which is what this flag names.
    return bool(ms_wall <= ms_host * HOST_BOUND_RATIO)


# ── per-launch metadata (K21 / D1) ───────────────────────────────────
def launch_meta(fn: Callable[[], Any], iters: int = 2,
                out_dir: str | None = None) -> dict:
    """grid / block / registers / smem / occupancy per launch, from Kineto.

    `ncu` is refused on the login node (ERR_NVGPUCTRPERM), so the launch
    geometry is read out of a chrome trace instead — the method K21 §2
    and D1 both used from their own scripts, folded in here so a config
    can ask for it as the `launch_meta` metric.  One signature is kept
    per distinct (grid, regs, smem), with its launch count and summed
    duration, so a kernel launched in several geometries is not averaged
    into one meaningless row.
    """
    import json as _json
    import tempfile

    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json", dir=out_dir)
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as fh:
            trace = _json.load(fh)
    finally:
        os.unlink(path)
    out: dict[str, dict] = {}
    for e in trace.get("traceEvents", []):
        if e.get("cat") not in ("kernel", "Kernel"):
            continue
        a = e.get("args", {}) or {}
        rec = out.setdefault(e.get("name", "?"),
                             dict(n=0.0, dur_us=0.0, launches={}))
        rec["n"] += 1
        rec["dur_us"] += float(e.get("dur", 0.0))
        key = (str(a.get("grid")), a.get("registers per thread"),
               a.get("shared memory"))
        sig = rec["launches"].setdefault(
            "|".join(str(k) for k in key),
            dict(grid=a.get("grid"), block=a.get("block"),
                 regs=a.get("registers per thread"),
                 smem=a.get("shared memory"),
                 occupancy_pct=a.get("est. achieved occupancy %"),
                 blocks_per_sm=a.get("blocks per SM"),
                 warps_per_sm=a.get("warps per SM"),
                 count=0.0, dur_us=0.0))
        sig["count"] += 1
        sig["dur_us"] += float(e.get("dur", 0.0))
    for rec in out.values():
        rec["n"] /= iters
        rec["dur_us"] /= iters
        for sig in rec["launches"].values():
            sig["count"] /= iters
            sig["dur_us"] /= iters
    return out


# ── compiled-kernel resources (D1 `regs`) ────────────────────────────
_RESOURCE_KERNELS = (
    ("orbit_ssm.kernels.ssd_mimo.fused_fwd_kernel",
     ("_fused_chunk_fwd_kernel",)),
    ("orbit_ssm.kernels.ssd_mimo.fused_bwd_kernel",
     ("_bwd_ka_kernel", "_bwd_kbc_hb_kernel", "_bwd_kb1_kernel",
      "_bwd_kb2_kernel", "_bwd_k2_kernel")),
)


def kernel_resources(kernels=_RESOURCE_KERNELS) -> list[dict]:
    """registers / spills / shared memory of every COMPILED variant.

    Read off the Triton `JITFunction`'s own device caches after a cell
    has run, so it reports what actually launched.  This is the number
    K15b's "the kernel spills at N=128" finding is made of (fused
    forward: 0 spills at N=64, 112 at N=128), and the reason a config
    can ask for it beside the timing instead of a separate probe run.

    NOTE the caches are cumulative within a process: a process that
    built two operand modes reports both modes' variants together.  Run
    one mode per process when the modes have to be told apart.
    """
    import importlib as _il

    rows: list[dict] = []
    for modname, names in kernels:
        try:
            mod = _il.import_module(modname)
        except Exception:                          # noqa: BLE001
            continue
        for name in names:
            k = getattr(mod, name, None)
            if k is None:
                continue
            k = getattr(k, "fn", k)
            caches = getattr(k, "device_caches", None)
            if not caches:
                continue
            merged: dict = {}
            for dev in list(caches):
                merged.update(caches[dev][0])
            for compiled in merged.values():
                meta = compiled.metadata
                rows.append(dict(kernel=name,
                                 warps=meta.num_warps, stages=meta.num_stages,
                                 regs=compiled.n_regs,
                                 spills=compiled.n_spills,
                                 smem=meta.shared))
    return rows


# ── roofline ────────────────────────────────────────────────
# Analytic COMPULSORY DRAM traffic and FLOPs for a chunked-SSD cell,
# read off the tile math (ncu is refused on the login node:
# ERR_NVGPUCTRPERM), divided by measured device peaks.  A head-SHARED
# [B, L, N] map read by all H CTAs of a batch is counted ONCE (the
# working set is L2-resident and every CTA of a wave sits on the same
# chunk); a per-head [B, L, H, N] map is counted in full.
_MB = 1024.0 ** 2


def roofline_terms(B: int, H: int, P: int, N: int, L: int, T: int) -> dict:
    """The per-tensor byte terms of one chunked-SSD cell (K21 `_cell`)."""
    nc = L // T
    return dict(B=B, H=H, P=P, N=N, L=L, T=T, nc=nc,
                x=B * L * H * P * 2,             # [B,L,H,P] bf16
                bc=B * L * N * 2,                # [B,L,N]   bf16 head-shared
                bc_h=B * L * H * N * 2,          # [B,L,H,N] bf16 per head
                vec=B * H * L * 4,               # [B,H,L]   fp32
                theta=B * L * H * (N // 2) * 4,  # fp32, N/2 pairs
                sall=B * H * nc * N * P * 4,     # [B,H,nc,N,P] fp32
                sall_bf16=B * H * nc * N * P * 2,
                y=B * L * H * P * 2)


def roofline_model(B: int, H: int, P: int, N: int, L: int, T: int) -> dict:
    """`bytes` and `flops` of ONE fused forward at this cell (K21).

    The four chunk dots are identical on every chunked forward path, so
    this is the model both sides of a parity pair are scored against.
    """
    c = roofline_terms(B, H, P, N, L, T)
    ncta, nc = B * H, c["nc"]
    f_chunk = 2 * (T * N * P + N * T * P + T * N * T + T * T * P)
    return dict(cell=c,
                bytes=c["x"] + c["y"] + 2 * c["bc"] + c["vec"],
                flops=f_chunk * ncta * nc)


def device_peaks(device: str = "cuda") -> dict:
    """Measured device peaks: a 4096^3 GEMM and a 256 MiB device copy.

    Measured, never quoted from a datasheet: the roofline percentages
    are only meaningful against what THIS card sustains right now.
    """
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    src = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)

    def _ev(fn, iters):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters

    gemm_ms = _ev(lambda: a @ b, 20)
    copy_ms = _ev(lambda: dst.copy_(src), 10)
    del a, b, src, dst
    torch.cuda.empty_cache()
    return dict(gemm4096_bf16_tflops=2 * 4096 ** 3 / (gemm_ms * 1e-3) / 1e12,
                copy_gbs=2 * 256 * _MB / (copy_ms * 1e-3) / 1e9,
                gemm_ms=gemm_ms, copy_ms=copy_ms)


def roofline(shape: dict, ms: float | None, peaks: dict | None = None
             ) -> dict:
    """Achieved GB/s and TFLOP/s of a measured cell against the model.

    `shape` is a `Cell.shape`; the cell is skipped (empty dict) when it
    does not carry the chunked-SSD axes the model is written for.
    """
    need = ("B", "L", "N")
    if any(shape.get(k) is None for k in need) or not ms:
        return {}
    H = shape.get("H") or shape.get("D") or 0
    P = shape.get("P") or shape.get("G") or 0
    T = shape.get("chunk") or 64
    if not H or not P:
        return {}
    m = roofline_model(int(shape["B"]), int(H), int(P), int(shape["N"]),
                       int(shape["L"]), int(T))
    gbs = m["bytes"] / (ms * 1e-3) / 1e9
    tfs = m["flops"] / (ms * 1e-3) / 1e12
    out = dict(roofline_bytes=m["bytes"], roofline_flops=m["flops"],
               achieved_gbs=gbs, achieved_tflops=tfs)
    if peaks:
        if peaks.get("copy_gbs"):
            out["pct_dram_peak"] = 100.0 * gbs / peaks["copy_gbs"]
        if peaks.get("gemm4096_bf16_tflops"):
            out["pct_gemm_peak"] = (100.0 * tfs
                                    / peaks["gemm4096_bf16_tflops"])
    return out


# ── the capture regime (G34 / E2b) ───────────────────────────────────
CAPTURE_TAG = "cuda_graph"
GRAPH_ARM_SUFFIX = "@graph"


def graphs_available() -> bool:
    """Is `orbit_ssm.graphs` (G34) importable in this checkout?"""
    try:
        import orbit_ssm.graphs  # noqa: F401
    except Exception:                              # noqa: BLE001
        return False
    return True


def captured_closure(module, leaves, pass_: str, precision: str, *,
                     warmup: int = 11):
    """The timed closure of a CAPTURED row: one graph replay, nothing else.

    Deliberately NOT `copy_in` + replay: the eager arms reuse their own
    `x` from iteration to iteration (`harness.make_pass` closes over one
    fixed input), so charging the captured arm for a copy the eager one
    never makes would be a handicap, not a measurement.

    The `GraphedStep` is hung on the returned function so it stays alive
    (it owns the graph and every static buffer) and so a metric pass can
    read its memory cost.  Raises whatever `graphed_step` raises — a
    capture refusal must be a RECORDED skip, never a silent eager
    fallback that would put an eager number in a captured column.
    """
    from orbit_ssm import graphs as gph

    x = leaves[0]
    gs = gph.graphed_step(module, (x,), fwd_bwd=(pass_ == "fwd_bwd"),
                          amp=lambda: gph.autocast(precision), warmup=warmup)

    def run():
        gs.graph.replay()
        return gs.static_output

    run.graphed_step = gs
    return run


# ── the compiled regime ─────────────────
COMPILE_ARM_SUFFIX = "@compiled"
COMPILE_COMBINED_ARM_SUFFIX = "@compiled-combined"


def compiled_closure(mod, leaves, pass_: str, *,
                     amp: Callable[[], Any] | None = None,
                     compile_kwargs: dict | None = None, call=None):
    """Compile a module or arbitrary call, then reuse the standard pass wrapper.

    Compilation and lazy backward compilation run during warmup, outside timing.
    `call` supports multiple inputs, explicit cache passing, and kernel functions.
    `fullgraph=True` can be requested to reject graph breaks; failures propagate
    to the normal per-cell error record rather than falling back to eager.
    """
    from .harness import make_pass
    kw = dict(compile_kwargs or {})
    kw.setdefault("dynamic", False)
    target = call if call is not None else mod
    compiled_mod = torch.compile(target, **kw)
    invoke = compiled_mod if call is not None else lambda: compiled_mod(leaves[0])
    run = make_pass(invoke, leaves, pass_, amp=amp)
    run.compiled_module = compiled_mod
    return run


# ── the per-cell metric hook ─────────────────────────────────────────
def metric_hook(metrics: Sequence[str], *, budget: Budget,
                peaks: dict | None = None,
                cuda_self_rounds: int = 3,
                grids: frozenset[str] | None = None):
    """Build the `post_cell` hook that records `metrics` for every cell.

    Returns a `(cell, closure) -> dict` merged into that cell's record.
    `grids`, when given, restricts the expensive `cuda_self` pass to
    those grids (the historical `bench_kernels._SELF_CUDA_GRIDS`
    behaviour); `Budget.cuda_self=False` switches it off wholesale.
    """
    want = set(metrics)

    def hook(cell, closure) -> dict:
        out: dict = {}
        if "cuda_self" in want and budget.cuda_self and (
                grids is None or getattr(cell, "grid", "") in grids):
            try:
                res = cuda_self(closure, rounds=cuda_self_rounds)
                out["ms_cuda_self"] = res["ms_cuda_self"]
                out["cuda_self_round_totals"] = res["round_totals"]
                out["cuda_kernels"] = res["kernels"]
                out["cuda_calls_vary"] = res["calls_vary"]
                if "launches" in want:
                    out["launches"] = res["launches"]
                    out["n_launches"] = res["n_launches"]
            except Exception as exc:               # noqa: BLE001
                out["ms_cuda_self"] = None
                out["ms_cuda_self_error"] = f"{type(exc).__name__}: {exc}"
        elif "launches" in want:
            try:
                _, la = _profile_cuda(closure, 1)
                out["launches"] = la
                out["n_launches"] = sum(la.values())
            except Exception as exc:               # noqa: BLE001
                out["launches_error"] = f"{type(exc).__name__}: {exc}"
        if "host" in want:
            try:
                w, h = host_enqueue_ms(closure, iters=budget.ev_iters)
                out["ms_wall_events"] = w
                out["ms_host_enqueue"] = h
                out["host_bound"] = host_bound(w, h)
            except Exception as exc:               # noqa: BLE001
                out["ms_host_enqueue_error"] = f"{type(exc).__name__}: {exc}"
        if "roofline" in want:
            try:
                base = out.get("ms_cuda_self") or out.get("ms_wall_events")
                out.update(roofline(getattr(cell, "shape", {}) or {}, base,
                                    peaks))
            except Exception as exc:               # noqa: BLE001
                out["roofline_error"] = f"{type(exc).__name__}: {exc}"
        if "launch_meta" in want:
            try:
                out["launch_meta"] = launch_meta(closure)
            except Exception as exc:               # noqa: BLE001
                out["launch_meta_error"] = f"{type(exc).__name__}: {exc}"
        if "kernel_resources" in want:
            try:
                out["kernel_resources"] = kernel_resources()
            except Exception as exc:               # noqa: BLE001
                out["kernel_resources_error"] = f"{type(exc).__name__}: {exc}"
        if hasattr(closure, "training_summary"):
            out.update(closure.training_summary())
        gs = getattr(closure, "graphed_step", None)
        if gs is not None:
            out["capture"] = CAPTURE_TAG
            out["capture_memory_bytes"] = gs.memory
            out["capture_notes"] = list(gs.notes)
        return out

    return hook


# ── ratios: the ONLY place a ratio is formed ─────────────────────────
class RatioRefused(ValueError):
    """A ratio the protocol forbids (different group, or mixed regimes)."""


def ratio(numer: dict, denom: dict, *, field: str = "ms_median") -> float:
    """`numer[field] / denom[field]`, refusing every unsound pairing.

    Refused: records from different comparison GROUPS (they were not
    interleaved, so on a drifting node their ratio is noise); a CAPTURED
    record against an EAGER one (a captured number may only be
    ratioed against another captured number); and a mismatch of the
    precision/pass/N labels that would make the two sides different
    measurements.
    """
    for k, why in (("device", "use different devices"),
                   ("group", "were not interleaved (different group)"),
                   ("precision", "are different precision columns"),
                   ("pass", "are different passes")):
        if numer.get(k) != denom.get(k):
            raise RatioRefused(
                f"refusing a ratio: {numer.get('arm')} and {denom.get('arm')} "
                f"{why}: {numer.get(k)!r} vs {denom.get(k)!r}")
    if (numer.get("tags") or {}).get("capture") != \
            (denom.get("tags") or {}).get("capture") or \
            numer.get("capture") != denom.get("capture"):
        raise RatioRefused(
            "refusing a ratio: one side is CUDA-graph captured and the "
            "other is eager (E2b/G34 — captured vs captured only)")
    if (numer.get("shape") or {}).get("N") != (denom.get("shape") or {}).get("N"):
        raise RatioRefused(
            "refusing a ratio: the two sides ran at different state sizes "
            f"N={(numer.get('shape') or {}).get('N')} vs "
            f"{(denom.get('shape') or {}).get('N')}")
    a, b = numer.get(field), denom.get(field)
    if a is None or not b:
        raise RatioRefused(f"refusing a ratio: {field} missing on one side")
    return float(a) / float(b)
