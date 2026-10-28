"""Shared-memory-aware launch policy for the fused SSD-MIMO kernels.

WHY THIS EXISTS
---------------
The fused forward stages its `[T, N]` score/B/C operands in shared memory
and pipelines them `num_stages` deep, so its per-CTA shared-memory request
grows super-linearly in the chunk size T and linearly in the state size N.
When the request crosses the device budget (e.g. N = 128 x T = 128 on sm_90's
232,448 B) Triton raises `OutOfResources` AT LAUNCH.  This module detects
that before launching and repairs the plan.

HOW THE NEED IS DETERMINED -- MEASURED, NOT MODELLED
----------------------------------------------------
`fused_fwd_kernel._fwd_smem_required` asks TRITON, via a dry compile (`JITFunction.warmup`,
which compiles and returns the `CompiledKernel` WITHOUT launching it) and
reads `kernel.metadata.shared`.  That number is exactly the "Required:" figure
in the `OutOfResources` message, and the compile lands in the same JIT cache
the real launch consults -- so the check is EXACT and costs nothing: the
compile had to happen anyway.

No fitted formula is used: the requirement is not smooth in (N, T, dtype,
precision) (e.g. fp32/"tf32" at N=128, T=64 needs MORE than fp32/"ieee"), and
cells sit within ~1 KB of the budget.

The G tile does not change the request (the `[N, GT]` state and `[T, GT]`
x/y tiles live in registers), and MIMO cells (R, O > 1) ask for LESS than
R = O = 1, which is the worst case on those axes.

The training path stores the per-chunk states (`STORE_SALL=True`) and asks
for more than a `no_grad` forward at the same (N, T, dtype, precision).
That is why the clamp asks Triton per call instead of consulting a table.

The BACKWARD's own forward replay (`STATE_ONLY=True, STORE_SALL=True`,
which rebuilds the per-chunk states under the default `SALL=recompute`) is
strictly CHEAPER than the forward it replays -- it stages no score/Y tile.
That is what makes it safe for the forward alone to pick the
plan: a plan whose forward fits cannot blow up inside the backward.

TWO WAYS OUT, AND THE CHOICE IS MEASURED
----------------------------------------
A cell that does not fit has TWO exact repairs, and they trade different
things:

  T-HALVING at full N   `T, T/2, T/4, ... 32`, take the first that fits.
                        Same grid, shorter chunks: more chunk boundaries,
                        so more serial state-passing steps per kernel and
                        a smaller intra-chunk quadratic block.
  N-SPLITTING at full T  keep the requested T and run the state axis as
                        several narrower subsystems (`_n_tile_split`).
                        The decay is a scalar shared across N, so the
                        tiles are INDEPENDENT: their outputs sum and their
                        final states concatenate.  Same chunk width, but
                        `ceil(N/tile)` launches over the same tokens -- a
                        larger grid, and B/C reloaded once per tile.

Both are exact (T only regroups the scan's associativity; the N-split is
the same function by linearity), so the choice is purely about speed --
and speed here is not predictable from the shape: which one wins depends
on the occupancy the narrower tile buys against the launches it costs, and
can differ between forward-only and forward+backward at the same cell.
So the launcher MEASURES it, once per
`(device, N, T, dtype, precision, mode)`, and caches the verdict in-process
and on disk next to the Triton cache.  (`SSD_MIMO_SMEM_ITERS` and the
strategy log make the measurement reproducible.)

Falling to a smaller T only regroups the scan's associativity (the chunked
algorithm is exact for any T; the state carried across chunk boundaries is
the same state), so the numerics move by at most fp round-off — never
silently by more.  A one-time `RuntimeWarning` names N, T, the requested
bytes, the device budget and the plan actually used.

WHEN MEASURING IS IMPOSSIBLE
---------------------------
The micro-benchmark runs the two candidate plans a few times on DETACHED
views of the caller's own tensors (`.detach()` shares storage, so nothing
is copied) — under `no_grad` when the call needs no backward, and with a
real `.backward()` when it does.  That is not always allowed:

  * inside a CUDA-graph capture, a synchronize / event timing would break
    the capture — `torch.cuda.is_current_stream_capturing()` is checked;
  * anywhere the trial itself raises (an OOM from the second plan's
    workspace, a driver error, a nested-autograd refusal), the candidate
    is simply disqualified, and if both are, the fallback is T-halving.

A no-grad-only caller is NOT one of these cases: the trial runs
forward-only there, which is exactly the work that caller does, and the
verdict is filed under its own `mode="fwd"` key — an inference process and
a training process at the same shape do not share a verdict, because they
do not have the same answer (see the bf16/ieee row, where the forward
prefers N-splitting and the forward+backward prefers T-halving).

The fallback is always T-HALVING: it needs no extra launches.  `SSD_MIMO_SMEM_STRATEGY=t_halve|n_split` pins it by hand and skips
the measurement entirely.

Env levers (`SSD_MIMO_` prefix):
  `SSD_MIMO_SMEM_CLAMP=0`      disable the check entirely (a
                               hard OutOfResources at launch)
  `SSD_MIMO_SMEM_LIMIT=<int>`  override the device budget in bytes (tests
                               use this to force the clamp on and off)
  `SSD_MIMO_SMEM_FLOOR=<int>`  smallest chunk the ladder will fall to (32)
  `SSD_MIMO_SMEM_STRATEGY=`    `measure` (default), `t_halve`, `n_split`
  `SSD_MIMO_SMEM_MEASURE=0`    never measure; take the fallback strategy
  `SSD_MIMO_SMEM_ITERS=<int>`  timed iterations per candidate (3)
  `SSD_MIMO_SMEM_STRATEGY_CACHE=<path>`
                               where the measured verdicts are persisted;
                               `""` disables the on-disk half.  The file
                               records its `cache_scope()` (GPU, capability,
                               Triton, torch, kernel code hash) and is
                               ignored under any other scope
"""

import json
import os
import tempfile
import warnings
from pathlib import Path

import torch

from .fused_fwd_kernel import _env


@torch._dynamo.assume_constant_result       # traced: a host query, evaluated once at trace time (a str is not a graph value)
def device_name(device) -> str:
    """`torch.cuda.get_device_name(device)`, "unknown" off CUDA (CPU-only import)."""
    try:
        return torch.cuda.get_device_name(device)
    except Exception:                     # noqa: BLE001 — CPU-only import
        return "unknown"



class SsdSmemOverflow(RuntimeError):
    """The fused scan plan exceeds its shared-memory legality envelope.

    Raised by the launch path BEFORE the kernel runs (the dry compile knows
    the number); caught by the entry point, which retries under whichever
    exact repair it measured faster — a smaller chunk at the full state
    width, or the full chunk on a narrower state.  Escapes to the caller
    only when BOTH ladders run out.
    """

    def __init__(self, required: int, limit: int, N: int, T: int):
        self.required = required
        self.limit = limit
        self.N = N
        self.T = T
        super().__init__(
            f"fused SSD-MIMO plan needs {required} B of shared memory "
            f"per CTA at N={N}, chunk T={T}; the device budget is "
            f"{limit} B")


_LIMIT_CACHE: dict = {}


def device_smem_limit(device=None) -> int:
    """Opt-in shared memory per block, in bytes (`SSD_MIMO_SMEM_LIMIT` wins).

    `shared_memory_per_block_optin` is the number Triton's launcher checks
    against (232,448 on GH200 / sm_90); `shared_memory_per_block` is the
    48 KB default-carveout figure and is NOT the limit here.
    """
    override = _env("SMEM_LIMIT", "")
    if override:
        return int(override)
    idx = (torch.cuda.current_device() if device is None
           else torch.device(device).index)
    if idx is None:
        idx = torch.cuda.current_device()
    if idx not in _LIMIT_CACHE:
        props = torch.cuda.get_device_properties(idx)
        _LIMIT_CACHE[idx] = int(
            getattr(props, "shared_memory_per_block_optin", 0)
            or props.shared_memory_per_block)
    return _LIMIT_CACHE[idx]


def smem_clamp_enabled() -> bool:
    return _env("SMEM_CLAMP", "1") == "1"


def chunk_ladder(chunk_size: int) -> list[int]:
    """`chunk_size` and the halvings below it, down to the floor (32).

    A non-power-of-two chunk has no halving ladder (the kernels' `T` is a
    Triton constexpr tile width and must stay a power of two), so it yields
    a single-entry ladder: no clamp is possible and the overflow is
    reported to the caller.
    """
    floor = int(_env("SMEM_FLOOR", "32"))
    if chunk_size & (chunk_size - 1):
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
    """The last 32 shared-memory RE-PLAN decisions, newest last.

    Two kinds of entry, both 5-tuples so a caller that only counts them
    (which is all the tests do) is indifferent:

        ("t_halve", N, requested_chunk, used_chunk,  (required, limit))
        ("n_split", N, requested_chunk, tuple(tiles), (required, limit))

    The WARNING is emitted once per cell per process — a training loop must
    not print it every step — so it is not a reliable observable.  This log
    records EVERY re-plan and is what the tests read.
    """
    return list(_CLAMP_LOG)


def _note(entry, key, message) -> None:
    _CLAMP_LOG.append(entry)
    del _CLAMP_LOG[:-_CLAMP_LOG_MAX]
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=4)


def note_chunk_clamped(N: int, requested: int, used: int,
                       required: int, limit: int) -> None:
    """Record a T-halving re-plan; warn once per (N, requested, used)."""
    _note(
        ("t_halve", N, requested, used, (required, limit)),
        ("t_halve", N, requested, used),
        f"SSD-MIMO: chunk_size {requested} does not fit in shared memory at "
        f"N={N} (the fused scan plan needs {required} B/CTA, device "
        f"budget {limit} B) — running at chunk_size={used} instead.  This "
        f"changes only the chunkwise associativity of the scan, not its "
        f"math; set SSD_MIMO_SMEM_CLAMP=0 to get the launch failure back.")


def note_n_split(N: int, requested: int, tiles, required: int,
                 limit: int) -> None:
    """Record an N-splitting re-plan; warn once per (N, requested, tiles)."""
    tiles = tuple(tiles)
    _note(
        ("n_split", N, requested, tiles, (required, limit)),
        ("n_split", N, requested, tiles),
        f"SSD-MIMO: chunk_size {requested} does not fit in shared memory at "
        f"N={N} (the fused scan plan needs {required} B/CTA, device "
        f"budget {limit} B) — keeping the chunk and running the state axis "
        f"as the independent subsystems {tiles} instead (measured faster "
        f"than halving the chunk at this cell).  The decay is shared across "
        f"N, so this is the same function: the tiles' outputs sum and their "
        f"final states concatenate.  Set SSD_MIMO_SMEM_STRATEGY=t_halve to "
        f"pin the chunk-halving route.")


# ══════════════════════════════════════════════════════════════════════
#  Strategy choice — measured once per cell, cached in-process and on disk
# ══════════════════════════════════════════════════════════════════════

T_HALVE = "t_halve"
N_SPLIT = "n_split"

_STRATEGY: dict = {}            # key -> winning strategy name
_STRATEGY_TIMINGS: dict = {}    # key -> {name: ms}
_STRATEGY_LOG: list = []
_STRATEGY_LOG_MAX = 32
_DISK_LOADED = False
_DISK_WARNED = False


def strategy_key(N: int, T: int, dtype, precision: str, mode: str,
                 device=None) -> str:
    """The cell a measured verdict is keyed by.

    `mode` is "fwd" or "fwd_bwd": the training path stores/rebuilds the
    per-chunk states and so has a DIFFERENT shared-memory request and a
    different balance between the two plans (the backward is the heavy
    half).  The device NAME rather than the index: two GH200s in one node
    share a verdict, a GH200 and an RTX do not.
    """
    name = device_name(device)
    dt = str(dtype).replace("torch.", "")
    return f"{name}|N{N}|T{T}|{dt}|{precision}|{mode}"


#: the kernel files whose compiled shared-memory requests and timings the
#: persisted verdicts are about (their `calibration.kernel_source`, hashed)
_SCOPE_CODE = ("fused_fwd_kernel.py", "fused_bwd_kernel.py", "chunkpar.py")
_SCOPE_KEY = "__scope__"
_SCOPE: dict = {}
#: the scope of the last file read that was IGNORED for a scope mismatch (diagnostics)
DISK_SCOPE_MISMATCH: dict = {}


def cache_scope() -> dict:
    """The scope a persisted verdict is valid in: GPU name,
    compute capability, Triton and torch versions, and the hash of the kernel source
    (`calibration.kernel_source` of `_SCOPE_CODE`).  A verdict measured under another
    scope -- a kernel edit, a Triton / torch upgrade, another GPU -- answers a different
    question, so a file whose recorded scope differs is ignored, and rewritten on the
    next store.  Computed once per process.

    Known limits (R14, accepted): the first call costs ~0.4 s (ast parse / unparse of
    the three large kernel files; measured on the login node).  The scope is that of
    the CURRENT device at that first call, while `strategy_key` is per call, so on a
    mixed-GPU node a process that switches devices keeps reading and writing the first
    device's file; give such processes one device each (CUDA_VISIBLE_DEVICES), as the
    fast-gate runner does, or a per-device `SSD_MIMO_SMEM_STRATEGY_CACHE`."""
    if not _SCOPE:
        import hashlib
        import triton
        from .. import calibration as cal
        here = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.sha256()
        for f in _SCOPE_CODE:
            h.update(f.encode())
            h.update(cal.kernel_source(os.path.join(here, f)).encode())
        try:
            cap = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        except Exception:                 # noqa: BLE001 — no usable device
            cap = None
        _SCOPE.update(gpu=device_name(None), capability=cap, triton=triton.__version__,
                      torch=torch.__version__, code_hash=h.hexdigest()[:16])
    return dict(_SCOPE)


def strategy_cache_path():
    """Where measured verdicts are persisted — next to the Triton cache.

    `SSD_MIMO_SMEM_STRATEGY_CACHE` overrides it; the empty string turns the
    on-disk half off (the in-process cache still works).  Returns None when
    persistence is disabled.

    Scoped: the file records `cache_scope()` and is ignored when the
    scope differs, so a verdict no longer outlives a kernel change, a
    Triton / torch upgrade or a GPU swap.  The default file name carries a
    short hash of the scope too, so processes of two scopes sharing one
    TRITON_CACHE_DIR keep one file each instead of overwriting each other.
    """
    override = _env("SMEM_STRATEGY_CACHE", None)
    if override is not None:
        return Path(override) if override else None
    root = os.environ.get("TRITON_CACHE_DIR")
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".triton", "cache")
    import hashlib
    tag = hashlib.sha256(json.dumps(cache_scope(), sort_keys=True).encode()).hexdigest()[:12]
    return Path(root) / f"ssd_mimo_smem_strategy-{tag}.json"


def _disk_read() -> dict:
    """The verdicts of the on-disk file, or {} when absent, corrupt or of another scope."""
    path = strategy_cache_path()
    if path is None:
        return {}
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except Exception:                     # noqa: BLE001 — absent or corrupt
        return {}
    if not isinstance(data, dict):
        return {}
    scope = data.pop(_SCOPE_KEY, None)
    if scope != cache_scope():
        DISK_SCOPE_MISMATCH.clear()
        DISK_SCOPE_MISMATCH.update(path=str(path), recorded=scope)
        return {}
    return data


def _disk_write(key: str, winner: str) -> None:
    """Merge one verdict into the on-disk cache (atomic, best-effort).

    Re-reads first so concurrent processes measuring different cells do not
    clobber each other; a failure to write is not an error — the verdict
    still stands for this process.
    """
    global _DISK_WARNED
    path = strategy_cache_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _disk_read()               # {} for a file of another scope: replaced
        data[key] = winner
        data[_SCOPE_KEY] = cache_scope()
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, indent=0, sort_keys=True)
            os.replace(tmp, path)
        except Exception:                 # noqa: BLE001
            os.unlink(tmp)
            raise
    except Exception as exc:              # noqa: BLE001 — read-only cache dir
        if not _DISK_WARNED:
            _DISK_WARNED = True
            warnings.warn(
                f"SSD-MIMO: could not persist the shared-memory strategy "
                f"verdict to {path} ({exc}); it stays in-process only.",
                RuntimeWarning, stacklevel=3)


@torch._dynamo.assume_constant_result       # traced: the one-time disk read (file IO) runs at trace time
def _disk_load_once() -> None:
    """Fan the on-disk file out into the three in-process caches ONCE.

    One file carries three kinds of verdict, told apart by key prefix:
    `"chunk:"` (a chunk T, int), `"fwdcfg:"` (a forward launch config,
    `"warps,stages,maxnreg"`) and the bare, unprefixed shared-memory
    strategy names.  Routing here rather than in each lookup is what
    keeps `_DISK_LOADED` honest: whichever cache is consulted first, all
    three end up populated from the same read.
    """
    global _DISK_LOADED
    if _DISK_LOADED:
        return
    _DISK_LOADED = True
    for k, v in _disk_read().items():
        if k.startswith("chunk:"):
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
    """The cached verdict for `key`, or None.  Disk is read once."""
    if key in _STRATEGY:
        return _STRATEGY[key]
    _disk_load_once()
    return _STRATEGY.get(key)


def strategy_log() -> list:
    """The last 32 (key, winner, timings, source) strategy resolutions."""
    return list(_STRATEGY_LOG)


def _log_strategy(key, winner, timings, source) -> None:
    _STRATEGY_LOG.append((key, winner, dict(timings), source))
    del _STRATEGY_LOG[:-_STRATEGY_LOG_MAX]


def measuring_possible() -> bool:
    """False when the micro-benchmark must not (or cannot) be run.

    A CUDA-graph capture is the hard case: the trial's event timing and
    synchronize would break the capture.  `SSD_MIMO_SMEM_MEASURE=0` is the
    manual off switch.
    """
    from .fused_fwd_kernel import autotune_off
    if autotune_off():          # the one autotune switch
        return False
    if _env("SMEM_MEASURE", "1") != "1":
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:                     # noqa: BLE001 — no CUDA at all
        return False
    return True


def _time_call(fn, iters: int) -> float:
    """Milliseconds per call, CUDA-event timed, after one warm-up.

    The warm-up matters more than usual here: the first call of a candidate
    pays Triton compilation and (outside the test pins) the backward's
    autotune sweep, which would otherwise be attributed to the plan.
    """
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
    """The verdict that needs no measurement, or None.

    Lets the caller skip the (expensive) discovery of each family's first
    fitting plan when a pin or a cached verdict already decides it.
    """
    available = list(available)
    if not available:
        return None
    forced = _env("SMEM_STRATEGY", "measure")
    if forced in (T_HALVE, N_SPLIT):
        winner = forced if forced in available else available[0]
        _log_strategy(key, winner, {}, "pinned")
        return winner
    if len(available) == 1:
        _log_strategy(key, available[0], {}, "only-candidate")
        return available[0]
    cached = lookup_strategy(key)
    if cached in available:
        _log_strategy(key, cached, _STRATEGY_TIMINGS.get(key, {}), "cached")
        return cached
    return None


def resolve_strategy(key: str, candidates: dict, fallback: str = T_HALVE):
    """Pick the strategy for `key`; `candidates` maps name -> trial callable.

    Returns the winning name (always a key of `candidates`).  Order of
    preference: an explicit `SSD_MIMO_SMEM_STRATEGY` pin, the in-process /
    on-disk cache, a fresh measurement, and finally `fallback` (T-halving)
    when measuring is impossible or every
    trial raises.
    """
    if not candidates:
        raise ValueError("resolve_strategy: no candidate plans")

    def _fall():
        return fallback if fallback in candidates else next(iter(candidates))

    pref = preferred_strategy(key, candidates)
    if pref is not None:
        return pref
    if not measuring_possible():
        winner = _fall()
        _log_strategy(key, winner, {}, "unmeasurable")
        return winner

    iters = max(1, int(_env("SMEM_ITERS", "3")))
    timings = {}
    for name, trial in candidates.items():
        try:
            timings[name] = _time_call(trial, iters)
        except Exception:                 # noqa: BLE001 — disqualify, not fail
            continue
    if not timings:
        winner = _fall()
        _log_strategy(key, winner, {}, "trials-failed")
        return winner
    winner = min(timings, key=timings.get)
    _STRATEGY[key] = winner
    _STRATEGY_TIMINGS[key] = timings
    _log_strategy(key, winner, timings, "measured")
    _disk_write(key, winner)
    return winner


# ══════════════════════════════════════════════════════════════════════
#  Chunk-T autotune cache — `SSDMimoSSM(chunk_size="measure")`
#
#  A different question from the strategy cache above (T-halve vs
#  N-split repairs an OVERFLOW; this picks the fused scan's chunk T
#  before any overflow is even in play), but the same shape of answer: a
#  small verdict keyed by the cell, cached in-process and persisted next
#  to the Triton cache.  Reuses `strategy_cache_path()` / `_disk_read()`
#  / `_disk_write()` under a `"chunk:"`-prefixed key so the two verdict
#  kinds share one file without colliding (the smem strategy keys are
#  bare `"<device>|N..|T..|..."` strings, never prefixed).
# ══════════════════════════════════════════════════════════════════════

_CHUNK_CACHE: dict = {}
_CHUNK_TIMINGS: dict = {}
_CHUNK_LOG: list = []
_CHUNK_LOG_MAX = 32

#: The admissible chunk-T ladder for the autotune: powers of two, floor
#: 16.  Unrelated to `chunk_ladder`'s SMEM_FLOOR-gated overflow
#: ladder (default floor 32, a different purpose) even though both are
#: halving sequences of the same shape.
#:
#: T = 128 is excluded: it lost at every measured cell, neither rule table
#: (`_CHUNK_RANK_TABLE`, `_CHUNK_RANK_TABLE_FWD`) can return it (both cap at
#: 64), and timing it in the `"measure"` ladder would spend a warmed, timed
#: pass per call on a candidate that never wins.  `chunk_ladder`'s OVERFLOW
#: ladder still starts from whatever chunk the caller asked for, 128
#: included: that is the shared-memory safety net, a different question.
_CHUNK_T_CANDIDATES = (64, 32, 16)

#: The largest T `chunk_admissible` will hand back (see above).
_CHUNK_T_CEILING = 64


def chunk_admissible(ceiling: int = _CHUNK_T_CEILING,
                     floor: int = 16) -> list[int]:
    """The candidate chunk-T set a `chunk_size="measure"` trial times."""
    return [T for T in _CHUNK_T_CANDIDATES if floor <= T <= ceiling]


def chunk_cache_key(R: int, O: int, N: int, dtype, precision: str,
                    mode: str, shape, device=None) -> str:
    """The cell a measured chunk-T verdict is keyed by.

    `mode` is "fwd" or "fwd_bwd" (mirrors `strategy_key`'s reasoning: a
    chunk T that wins forward-only need not win with a backward
    attached).  `shape` is the caller's own x tensor shape — same (R, O,
    N, dtype, precision) at a different (B, L) can legitimately pick a
    different T.
    """
    name = device_name(device)
    dt = str(dtype).replace("torch.", "")
    shp = "x".join(str(s) for s in shape)
    return f"chunk:{name}|R{R}|O{O}|N{N}|{dt}|{precision}|{mode}|{shp}"


def chunk_cache_lookup(key: str):
    """The cached chunk-T verdict for `key`, or None.  Disk is read once
    (shares `_DISK_LOADED`/`_disk_read` with the smem strategy cache)."""
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]
    _disk_load_once()
    v = _CHUNK_CACHE.get(key)
    if v is None and key.startswith("chunk:cp:"):
        # the shipped / local calibration file's structural thresholds
        # (`calibration_chunkpar`); inert unless SSD_MIMO_CALIBRATION is on
        from .calibration_chunkpar import lookup
        v = lookup(key)
    return v


def chunk_cache_store(key: str, T: int, timings: dict | None = None) -> None:
    """Cache a measured chunk-T verdict in-process, on disk, and in the
    chunk log (`chunk_measure_log()`)."""
    _CHUNK_CACHE[key] = T
    if timings:
        _CHUNK_TIMINGS[key] = dict(timings)
    _CHUNK_LOG.append((key, T, dict(timings or {})))
    del _CHUNK_LOG[:-_CHUNK_LOG_MAX]
    _disk_write(key, T)


def chunk_measure_log() -> list:
    """The last 32 (key, winning T, timings) chunk-autotune resolutions."""
    return list(_CHUNK_LOG)


# ══════════════════════════════════════════════════════════════════════
#  Forward LAUNCH-CONFIG autotune cache
#
#  The third verdict kind in this file, and the same shape of answer as
#  the two above: measured once per (device, cell), cached in-process and
#  persisted in the SAME on-disk file under a `"fwdcfg:"` prefix (see
#  `_disk_load_once` for the three-way routing).  The candidate grid is
#  pruned from the DEVICE's own properties and timed with the same
#  warm-up-then-CUDA-event method the chunk `"measure"` mode uses.
#
#  WHY IT IS SAFE TO SEARCH ON SPEED ALONE: every candidate differs from
#  every other ONLY in `num_warps`, `num_stages` and `maxnreg`, three launch
#  metadata fields that do not reach the kernel body.  No candidate changes
#  a `tl.constexpr` the kernel branches on, the chunk length, the G tile,
#  the operand dtype, the schedule bits or the grid -- so all of them
#  compute the same function of the same inputs, and the search cannot
#  express a wrong answer.  (`num_warps` DOES change the MMA lowering,
#  which is a correctness hazard on Hopper -- that one is pruned from the
#  candidate set before any timing, by `_mma_wide_tile_ok`.)
# ══════════════════════════════════════════════════════════════════════

_FWD_CFG_PREFIX = "fwdcfg:"
_FWD_CFG_CACHE: dict = {}
_FWD_CFG_TIMINGS: dict = {}
_FWD_CFG_LOG: list = []
_FWD_CFG_LOG_MAX = 32


def _fwd_cfg_parse(v):
    """`"warps,stages,maxnreg"` -> `(warps, stages, maxnreg)`, or None."""
    try:
        w, st, mn = (int(p) for p in str(v).split(","))
    except (TypeError, ValueError):
        return None
    if w <= 0 or st <= 0 or mn < 0:
        return None
    return (w, st, mn)


def _fwd_cfg_str(cfg) -> str:
    return "%d,%d,%d" % (int(cfg[0]), int(cfg[1]), int(cfg[2] or 0))


def fwd_cfg_key(cell: str, device=None) -> str:
    """The cell a measured forward launch config is keyed by.

    `cell` is the caller's own shape key -- `fused_fwd_kernel` builds it
    out of everything that changes the kernel's register / shared-memory
    profile (N, T, the G tile, R, O, the operand mode, `state_only`,
    `pair_half`, the grid size) -- and the DEVICE NAME is prepended here
    for the same reason `strategy_key` does it: two GH200s in one node
    share a verdict, a GH200 and an RTX do not.
    """
    name = device_name(device)
    return f"{_FWD_CFG_PREFIX}{name}|{cell}"


def fwd_cfg_lookup(key: str):
    """The cached `(warps, stages, maxnreg)` for `key`, or None."""
    if key in _FWD_CFG_CACHE:
        return _FWD_CFG_CACHE[key]
    _disk_load_once()
    return _FWD_CFG_CACHE.get(key)


def fwd_cfg_store(key: str, cfg, timings: dict | None = None) -> None:
    """Cache a measured forward launch config in-process, on disk and in
    `fwd_cfg_log()`."""
    cfg = (int(cfg[0]), int(cfg[1]), int(cfg[2] or 0))
    _FWD_CFG_CACHE[key] = cfg
    if timings:
        _FWD_CFG_TIMINGS[key] = dict(timings)
    _FWD_CFG_LOG.append((key, cfg, dict(timings or {})))
    del _FWD_CFG_LOG[:-_FWD_CFG_LOG_MAX]
    _disk_write(key, _fwd_cfg_str(cfg))


def fwd_cfg_log() -> list:
    """The last 32 (key, winning config, timings) forward-config
    resolutions, newest last."""
    return list(_FWD_CFG_LOG)


def fwd_cfg_mode() -> str:
    """`SSD_MIMO_FWD_CFG`: "auto" (default), "measure" or "off".

    * "auto"    -- take a CACHED measured verdict when one exists for
                   this cell, otherwise the device-agnostic heuristic.
                   Never spends a timed pass.  This is what makes a
                   one-off `"measure"` run permanently useful: the
                   verdict lands in the on-disk cache beside the Triton
                   cache and every later "auto" process reads it.
    * "measure" -- time the candidate grid once per cell, then cache.
    * "off"     -- the heuristic only; the cache is neither read nor
                   written (the A/B arm, and the escape hatch if a stale
                   verdict outlives a kernel change).
    """
    from .fused_fwd_kernel import autotune_off
    if autotune_off():          # the one autotune switch
        return "off"            # heuristic only, cache neither read nor written
    v = _env("FWD_CFG", "auto").strip().lower()
    return v if v in ("auto", "measure", "off") else "auto"
