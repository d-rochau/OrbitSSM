"""Launcher-level autotuning for the MH kernels.

`triton.autotune` does not fit these kernels. It picks a config *inside* the
launch, but two of the knobs have to be known before it: BLOCK_L sizes the
backward's checkpoint buffer (`hck` is `(B,H,P,cdiv(L,BLOCK_L)+1,N)`) and
BLOCK_P sets the forward's grid. Tuning one level up, at `mh_forward` /
`mh_backward`, is also what makes the trials safe -- both allocate every output
fresh on each call, so re-running one with a different config has no effect on
any earlier result and needs no `restore_value` bookkeeping.

The heuristic config is always trial 0, so autotuning can only ever match or
beat it (`fwd_gluon._pick_config` for the forward, `mh_backward`'s own
`cfg` defaults for the backward), never regress against it. Set
`HYBRID_MH_AUTOTUNE=0` to pin the heuristic and skip the search.

Tiers and pins (`kernel_options` family `hybrid_mamba`):
`autotune` = pinned | light | full (env HYBRID_MH_AUTOTUNE, 0/1),
`fwd_config` / `bwd_config` explicit configs (launched as is), and the
per-device table of measured winners (`_TABLE_FWD`/`_TABLE_BWD`) that is
the pinned config and the search's trial 0.  A search's verdict is also
written to `hybrid_mh_tune.json` next to the Triton compile cache
(HYBRID_MH_TUNE_CACHE overrides the path, 'off' disables), keyed by the
tune key, the kernel's source hash, the device and the stack: a later
process reads it and times -- and compiles -- nothing but the chosen config.
"""
from __future__ import annotations

import os

import torch
import triton

_CACHE: dict = {}
_CARRY_LAUNCHERS: dict = {}


def carry_launcher(full_kernel, side):
    """Apply the caller's MH tier to the nested time-split seed kernel.

    The full tier explores the complete inner search; light uses a shortlist
    of the measured carry winners. No device capability constants enter this
    selector.
    """
    from ... import kernel_options as _ko
    if side not in ("fwd", "bwd"):
        raise ValueError(f"unknown MH carry side: {side!r}")
    pin = _opt("CARRY_FWD_CONFIG" if side == "fwd" else "CARRY_BWD_CONFIG")
    mode = tier()
    if mode == "full" and pin is None:
        return full_kernel
    key = (full_kernel, side, mode, pin)
    if key not in _CARRY_LAUNCHERS:
        if pin is not None:
            c = _ko.parse_mh_config(pin, _ko.MH_CARRY_CONFIG_KEYS)
            triples = ((c["BLOCK_C"], c["BLOCK_T"], c.get("NUM_WARPS", 4)),)
        elif side == "fwd":
            triples = ((16, 128, 4),) if mode == "pinned" else (
                (16, 64, 2), (16, 128, 2), (16, 128, 4))
        elif side == "bwd":
            triples = ((32, 64, 4),) if mode == "pinned" else (
                (32, 64, 4), (32, 128, 8), (16, 64, 4),
                (16, 128, 4), (16, 128, 8), (32, 32, 2), (16, 64, 8))
        configs = [triton.Config(dict(BLOCK_C=bc, BLOCK_T=bt), num_warps=w)
                   for bc, bt, w in triples]

        def bench(fn, quantiles):
            # Carry kernels are sub-ms to a few ms. Avoid the nested tuner's
            # default 100-ms measurement for every candidate at every SPAN.
            times = triton.testing.do_bench(fn, warmup=3, rep=20, quantiles=quantiles)
            if times[0] > 4:
                times = triton.testing.do_bench(
                    fn, warmup=1, rep=5 * times[0], quantiles=quantiles)
            return times

        _CARRY_LAUNCHERS[key] = triton.autotune(
            configs=configs, key=full_kernel.keys, do_bench=bench)(full_kernel.fn)
    return _CARRY_LAUNCHERS[key]


def _opt(env_suffix):
    """The raw string for a `hybrid_mamba` kernel option: the active
    `kernel_options` scope first (`kernel_options.env_override`), then the
    environment (`HYBRID_MH_<suffix>`), else None."""
    from ... import kernel_options as _ko
    v = _ko.env_override("hybrid_mamba", env_suffix)
    if v is None:
        v = os.environ.get("HYBRID_MH_" + env_suffix)
    if v is None or not v.strip():
        return None
    return v.strip()


_TIERS = {"0": "pinned", "off": "pinned", "false": "pinned", "no": "pinned",
          "pinned": "pinned", "1": "light", "on": "light", "true": "light",
          "yes": "light", "auto": "light", "light": "light", "full": "full"}


def tier():
    """'pinned' | 'light' | 'full' -- the `autotune` kernel option / env
    HYBRID_MH_AUTOTUNE, spelled like SSD_MIMO_AUTOTUNE (0/1)."""
    v = _opt("AUTOTUNE")
    if v is None:
        return "light"
    t = _TIERS.get(v.lower())
    if t is None:
        raise ValueError(f"HYBRID_MH_AUTOTUNE / autotune={v!r}: expected "
                         "pinned | light | full (aliases 0/1, off/auto)")
    return t


def enabled():
    """Any timed search at all (`tier() != 'pinned'`)."""
    return tier() != "pinned"


def pinned_config(side):
    """The explicit `fwd_config` / `bwd_config` pin as a dict, or None."""
    from ... import kernel_options as _ko
    v = _opt("FWD_CONFIG" if side == "fwd" else "BWD_CONFIG")
    if v is None:
        return None
    return _ko.parse_mh_config(v, _ko.MH_FWD_CONFIG_KEYS if side == "fwd"
                               else _ko.MH_BWD_CONFIG_KEYS)


# Winners of the `light` search, per (compute capability, rotating,
# input element size, P): the `pinned` tier's configs and the search's
# trial 0.  A missing entry falls back to the launchers' heuristics.
_TABLE_FWD = {
    (120, True, 2, 16): dict(BLOCK_P=4, BLOCK_L=256, NUM_WARPS=1, PF=1),
    (120, False, 2, 16): dict(BLOCK_P=2, BLOCK_L=512, NUM_WARPS=1, PF=1),
}
_TABLE_BWD = {
    (120, True, 2, 16): dict(TP_P=2, NS=4, NUM_WARPS=1),
    (120, False, 2, 16): dict(TP_P=4, NS=4, NUM_WARPS=1),
}


def table_fwd(cc, rot, pmap, esz, P, N):
    """The forward's table entry (a copy), or None."""
    e = _TABLE_FWD.get((cc, bool(rot), esz, P))
    return dict(e) if e else None


def table_bwd(cc, rot, esz, P):
    """The backward's table entry (a copy), or None."""
    e = _TABLE_BWD.get((cc, bool(rot), esz, P))
    return dict(e) if e else None


def gate_bias_partial(P=None):
    """The `gate_bias_partial` kernel option.  Unset / 'auto': the bias ATOMICS (relaxed
    ordering); partials only tie with them while costing extra memory and
    the P == 1 head packing.  An explicit setting wins; `P` is kept for the call sites."""
    v = _opt("BWD_GATE_BIAS_PARTIAL")
    if v is None or v.lower() == "auto":
        return False
    return v.lower() in ("1", "true", "yes", "on")


def _bench(fn):
    # Gluon compiles per config, so the first call is dominated by compilation;
    # do_bench's own warmup absorbs it and the reps stay small because this
    # runs once per (shape, mode) and then hits the cache forever.
    t = triton.testing.do_bench(fn, warmup=5, rep=10)
    # A 10 ms budget is ONE timed run of a kernel that takes 10 ms or more, and
    # `_wins` compares equal-byte candidates with a strict `<`, so a single
    # sample makes the pick a coin toss between configs within noise.  Below
    # five samples, re-time on a median of about five.
    if t > 2.0:
        t = triton.testing.do_bench(fn, warmup=1, rep=5 * t,
                                    return_mode="median")
    return t


_INF = float("inf")


def _digest(out):
    """Magnitude fingerprint of a run's outputs: max-abs and L2 per tensor.

    Has to survive the one difference a legal config is allowed to make -- a
    different tiling sums the same terms in a different order -- while catching
    a wrong answer. Magnitudes move by rounding under reordering and by orders
    of magnitude when the arithmetic is actually wrong, so they separate the two
    with room to spare. Both norms accumulate in fp32 and neither materialises a
    temporary, which matters because these outputs run to hundreds of MB.
    """
    vals = []
    for t in (out if isinstance(out, (tuple, list)) else (out,)):
        if not isinstance(t, torch.Tensor) or not t.is_floating_point() \
                or t.numel() == 0:
            continue
        t = t.detach()
        vals.append(torch.stack((
            torch.linalg.vector_norm(t, ord=_INF, dtype=torch.float32),
            torch.linalg.vector_norm(t, dtype=torch.float32))))
    return torch.stack(vals) if vals else None


def _agrees(ref, got, rtol=2e-2):
    if ref is None or got is None:
        return True
    if ref.shape != got.shape:
        return False
    return bool(torch.isfinite(got).all()
                and (got - ref).abs().le(rtol * ref.abs() + 1e-6).all())


def _all_finite(out):
    """Every floating tensor in a run's output is finite.

    Same tuple/list unwrap as `_digest`, kept separate rather than folded into
    it: a digest is a MAGNITUDE fingerprint meant to be compared across
    configs with `_agrees`' tolerance, while this is a hard pass/fail with no
    reference -- an extreme-rate probe has no golden digest to agree with (a
    tiling that reorders the same terms is expected to move the CHUNK
    boundaries an extreme rate is sensitive to), so "did it stay finite" is
    the only question that makes sense to ask of it.
    """
    for t in (out if isinstance(out, (tuple, list)) else (out,)):
        if isinstance(t, torch.Tensor) and t.is_floating_point() \
                and t.numel() and not torch.isfinite(t).all():
            return False
    return True


def _try(run, cfg, extreme=None):
    """Time one config, fingerprint what it computed, and (optionally) probe
    it at an extreme rate: `(ms, digest)`, with the probe folded into the
    same accept/reject decision.

    `(None, None)` if it cannot run OR -- with `extreme` given -- if it ran
    but was not finite there. A trial allocates a full set of output buffers,
    and the backward's `x_s` is NS x twice the size of `x`, so a rejected
    trial can leave the allocator fragmented enough to OOM the *next* one.
    Dropping the cache after each keeps the search from failing for a reason
    unrelated to the config.

    `extreme`, when given, is called EXACTLY ONCE here -- not under `_bench`,
    which reps a config 10+ times to time it -- so a config this rejects
    costs one extra launch, not one extra benchmark.
    """
    try:
        digest = _digest(run(cfg))
        if extreme is not None and not _all_finite(extreme(cfg)):
            return None, None
        return _bench(lambda: run(cfg)), digest
    except Exception:  # noqa: BLE001 -- an unusable config is not an error
        return None, None
    finally:
        torch.cuda.empty_cache()


def choose(key, candidates, run):
    """Time each candidate config once, cache and return the winner.

    `run(cfg)` must fully execute the kernel with `cfg`. Candidates that fail to
    compile or launch (a tiling that does not divide the shape, a register
    request ptxas rejects) are skipped rather than raised, so the search space
    can stay coarse without the launcher having to model every constraint.
    """
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    best, best_ms = None, float("inf")
    for cfg in candidates:
        try:
            ms = _bench(lambda: run(cfg))
        except Exception:  # noqa: BLE001 -- an unusable config is not an error
            continue
        if ms < best_ms:
            best, best_ms = cfg, ms
    if best is None:                      # nothing ran; let the caller's own
        best = candidates[0]              # heuristic config raise properly
    _CACHE[key] = best
    return best


def _wins(ms, best_ms, cand_b, best_b, tol):
    """Is `cand` an improvement, counting BYTES as well as milliseconds?

    Without `bytes_of` this is a strict speed win. With it the
    comparison is two-sided and ASYMMETRIC: a config that is smaller need only
    match within `tol` to be taken, and one that is larger has to beat the
    incumbent by a margin that SCALES WITH HOW MUCH LARGER IT IS.

    A flat `tol` would be wrong because the growth side of the trade is
    unbounded: `AGX=0` is a few percent faster on some shapes and costs `NS`
    slabs of `x_s` instead of one, so a flat rule would buy a tiny speedup for
    GiBs of memory.

    So the requirement is `max(tol, growth)`: doubling the bytes demands a 50 %
    speed win, 8x demands 87 %, and neither will ever be met -- which is
    correct, because those are not trades anyone would take. `tol` survives as
    the floor, for candidates whose size barely moves.

    The smaller side keeps the flat rule deliberately. Its memory axes cost
    little time for a large peak reduction, so a pure-time search would reject
    them on what is nearly noise; `tol` is what buys them back.
    """
    if cand_b is None or cand_b == best_b:
        return ms < best_ms
    if cand_b < best_b:
        return ms <= best_ms * (1.0 + tol)
    return ms * (1.0 + max(tol, cand_b / best_b - 1.0)) < best_ms


def _hopeless(cand, best_b, bytes_of):
    """A candidate `_wins` could only accept at a >= 2x speedup.

    More than double the incumbent's bytes demands `ms * (1 + growth) <
    best_ms` with growth >= 1 -- never met by a config knob -- yet TIMING it
    allocates those bytes: the AGX=0 x NS=16 backward trial is 16 fp32 x_s
    slabs.  Skipping it cannot change the pick."""
    if bytes_of is None or best_b is None:
        return False
    return bytes_of(cand) > 2 * best_b


def _persist_path():
    """The on-disk tuned-config file: HYBRID_MH_TUNE_CACHE (a path; 'off' or
    '0' disables), else `hybrid_mh_tune.json` in TRITON_CACHE_DIR (so it is
    per node wherever the compile cache is, and shared wherever that is)."""
    v = os.environ.get("HYBRID_MH_TUNE_CACHE")
    if v is not None:
        v = v.strip()
        return None if v.lower() in ("", "0", "off", "none") else v
    d = os.environ.get("TRITON_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".triton", "cache")
    return os.path.join(d, "hybrid_mh_tune.json")


def _persist_digest(key, tag):
    """What a stored verdict is valid for: the tune key, the kernel's own
    source hash (`tag`), the device and the stack.  Any change -> new digest,
    so a verdict never outlives the code or the GPU it was measured on."""
    import hashlib
    dev = torch.cuda.current_device()
    ident = (repr(key), str(tag), _host_code_hash(),
             torch.cuda.get_device_name(dev),
             torch.cuda.get_device_capability(dev), triton.__version__,
             torch.__version__)
    return hashlib.sha1(repr(ident).encode()).hexdigest()


_HOST = {}


def _host_code_hash():
    """sha1 of the host code that DEFINES a verdict (search axes, grids,
    tables, byte model, legalisation after the search) -- this module and the
    two launchers.  The kernel's `cache_key` covers only the jitted source, so
    without this a changed axis/table/rule would keep replaying old verdicts."""
    h = _HOST.get("h")
    if h is None:
        import hashlib
        from . import bwd_gluon, common_gluon, fwd_gluon, layout, slp, ts_adjoint, ts_forward
        d = hashlib.sha1()
        for mod in (__import__(__name__, fromlist=["x"]), fwd_gluon, bwd_gluon,
                    common_gluon, layout, slp, ts_adjoint, ts_forward):
            with open(mod.__file__, "rb") as fh:
                d.update(fh.read())
        h = _HOST["h"] = d.hexdigest()
    return h


_DISK = {"path": None, "data": None}


def _persist_load(digest):
    path = _persist_path()
    if path is None:
        return None
    if _DISK["path"] != path:
        import json
        try:
            with open(path) as fh:
                _DISK["data"] = json.load(fh)
        except (OSError, ValueError):
            _DISK["data"] = {}
        _DISK["path"] = path
    hit = _DISK["data"].get(digest)
    return dict(hit) if isinstance(hit, dict) else None


def _persist_store(digest, cfg):
    """Read-merge-write under an exclusive lock, atomic rename: concurrent
    processes (ranks, pytest workers) each add their verdicts; a reader
    never sees a torn file."""
    import fcntl
    import json
    path = _persist_path()
    if path is None:
        return
    try:
        if any(not isinstance(v, (int, float, str, bool, type(None)))
               for v in cfg.values()):
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                data = {}
            data[digest] = cfg
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
            _DISK["path"], _DISK["data"] = path, data
    except (OSError, TypeError, ValueError):
        pass                       # a cache, never a failure


def cached(key):
    """The in-process verdict for `key`, or None (no search, no disk read)."""
    return _CACHE.get(key)


def choose_axes(key, base, axes, run, bytes_of=None, tol=0.03, grid=None,
                extreme=None, drop_keys=(), persist=None, pre=(), candidate_filter=None):
    """Coordinate descent around `base`: one knob at a time, greedily.

    The backward has four knobs and a full cross product is 70+ Gluon
    compiles per shape,
    which is minutes of first-call latency. Descent costs the SUM of the axis
    lengths instead of their product, and the knobs are close to separable
    because each moves a different resource: BLOCK_L moves elements per thread,
    NS moves CTAs, TP_P moves the p-contraction width, NUM_WARPS regroups
    threads. It is also how the constants were fitted by hand, so
    the search reproduces the sweep rather than second-guessing it.

    `axes` is ordered by leverage, largest first. **BLOCK_L must precede NS**: the
    two are coupled through NBLK -- the chunk loop each shard replays.  Descending in this order re-fits NS against whatever BLOCK_L just won,
    which is the coupling a hand-fitted rule misses.

    `base` is trial 0 and is only displaced by an improvement under `_wins`, so
    this can match the heuristic but never regress against it -- on whichever
    resource `bytes_of` says is being counted.

    A candidate must also AGREE with trial 0, not merely be faster. Time alone
    is not a filter: a config that runs, is quick and returns garbage is
    precisely what a search maximising speed will select, and rejecting only the
    ones that RAISE cannot see it. Such a bug is real: a `theta_init` buffer that
    aliases its own input at BLOCK_P == P.

    `extreme`, when given, is a second `run`-shaped callable -- same config
    protocol, a shape the caller already has sitting around at an extreme
    rate (e.g. its own inputs with A_log pinned to the family's admitted
    max) -- probed once per candidate for finiteness alongside the ordinary-
    rate agreement check, and folded into the SAME accept/reject decision so
    it costs one extra launch per candidate, not a second search. It exists
    because agreement alone cannot catch this class of bug: a chunk-size
    knob (BLOCK_L) can be correct AND fast on every ordinary-rate input the
    timing loop ever sees while being non-finite on a rate the caller's own
    training run will eventually produce, precisely because ordinary-rate
    agreement has no way to notice a decay exponent that only overflows once
    the rate is extreme (K5: `mh_bwd`'s BLOCK_L in {8, 16} at A_log = 40).
    Unprobed candidates (`extreme=None`, the default) skip the probe --
    callers that have no such shape handy, or whose kernel carries
    no such hazard, pay nothing for this.

    `pre` names axes that change the regime the grid's corners are fitted
    in (the time splits): they are descended once BEFORE the grid, and again
    LAST, after every other knob has been re-fitted -- the way back from a
    split that only won against the untuned base.
    """
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    # `persist` (the kernel's source hash) turns on the on-disk verdict: a
    # later process -- another rank, the next run, a test worker -- takes
    # the stored config and times nothing.
    pkey = None
    if persist is not None:
        pkey = _persist_digest(key, persist)
        hit = _persist_load(pkey)
        if hit is not None:
            _CACHE[key] = hit
            return hit
    # Split-first and unsplit searches often revisit the same complete
    # configuration. Reuse its measurement within this search only; the
    # baseline's unprobed timing must not bypass an extreme-rate check.
    # Include every field (including resource/layout pins), not just axes.
    trials = {}

    def measure(cfg, probe=None):
        identity = (tuple(sorted(cfg.items())), probe is not None)
        if identity not in trials:
            trials[identity] = _try(run, cfg, probe)
        return trials[identity]

    best = dict(base)
    best_ms, ref = measure(best)
    if best_ms is None:          # let the caller's own config raise properly
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
                    # `.get`, not `[]`: an axis may name a knob the heuristic leaves
                    # implicit (the forward's `PF` is only emitted on the one path that
                    # defaults it on), and a KeyError here fails the LAUNCH rather than
                    # the trial -- it is raised outside `_try`, so it does not read as
                    # "this config is unusable" but as a broken kernel.
                    if v == best.get(name):
                        continue
                    cand = dict(best, **{name: v})
                    if candidate_filter is not None and not candidate_filter(cand):
                        continue
                    if _hopeless(cand, best_b, bytes_of):
                        continue
                    ms, digest = measure(cand, extreme)
                    if ms is None or not _agrees(ref, digest):
                        continue
                    cand_b = None if bytes_of is None else bytes_of(cand)
                    if _wins(ms, best_ms, cand_b, best_b, tol):
                        best, best_ms, best_b = cand, ms, cand_b

        _descend([n for n in pre if n in axes])

        # A COUPLED pair gets a grid, not a descent. Descent assumes the knobs are
        # near-separable, and the forward's BLOCK_P/BLOCK_L are not: they set
        # elements-per-thread jointly, so moving one alone walks straight into the
        # register ceiling and the descent stops at the first wall rather than at
        # the optimum: no single-axis step from a poor corner may improve.
        # `grid` is a list of full
        # override dicts, tried against trial 0 before the per-axis descent runs,
        # so the descent starts from the best CORNER rather than from a guess.
        for over in (grid or ()):
            # `drop_keys` are knobs the CALLER pinned: the per-axis
            # descent below already skips them, and a grid corner that moved
            # one would be the same override by another door -- and a worse
            # one, because a corner moves several knobs at once and the pin
            # would vanish silently.  A corner that touches a pinned knob is
            # dropped whole rather than trimmed: the corners are COUPLED, so
            # half of one is not a fitted point.
            if any(k in over for k in drop_keys):
                continue
            cand = dict(best, **over)
            if candidate_filter is not None and not candidate_filter(cand):
                continue
            if cand == best or _hopeless(cand, best_b, bytes_of):
                continue
            ms, digest = measure(cand, extreme)
            if ms is None or not _agrees(ref, digest):
                continue
            cand_b = None if bytes_of is None else bytes_of(cand)
            if _wins(ms, best_ms, cand_b, best_b, tol):
                best, best_ms, best_b = cand, ms, cand_b

        _descend([n for n in axes if n not in pre] + [n for n in pre if n in axes])
        return best, best_ms, best_b

    # With split axes there are TWO basins: the split taken against the
    # untuned base (the pre pass) and the unsplit schedule tuned as usual.
    # Descent from one cannot reliably reach the other, so both are searched
    # and the faster verdict kept.  Without `pre` there is a single search.
    best, best_ms, best_b = _search(pre, axes)
    if any(n in axes for n in pre):
        alt = _search((), {k: v for k, v in axes.items() if k not in pre})
        if alt[1] is not None and _wins(alt[1], best_ms, alt[2], best_b, tol):
            best, best_ms, best_b = alt
    _CACHE[key] = best
    if pkey is not None:
        _persist_store(pkey, best)
    return best


def _pow2_span(base, lo, hi):
    """Powers of two bracketing `base` within [lo, hi], NEAREST FIRST.

    Order matters: descent takes the first strict improvement it finds on an
    axis and keeps going from there, so offering the far end first would let a
    marginal win at 4x block a larger one at 2x.
    """
    out = []
    for step in (2, 4):
        for cand in (base // step, base * step):
            if lo <= cand <= hi and cand > 0 and cand != base \
                    and cand not in out:
                out.append(cand)
    return out


def t_axes(base, P, L, ckpt_l, rot, tier="full", ts_axis=False):
    """Search space for `mh_forward`, ordered by leverage.

    The winning config family is device dependent (BLOCK_P=4 with TPW_L=32 on
    sm_90, BLOCK_P=1 on sm_120), so rather than carry a hand-fitted rule per
    part, the axes are offered to the same descent the backward uses.

    BLOCK_P goes FIRST because it is the axis with the leverage and because
    every other axis is conditioned on it: it sets the register tile with
    BLOCK_L, it divides the p-CTA count (and so the B/C traffic, which each
    p-CTA re-reads whole), and it decides whether TPW_L does anything at all.

    TPW_L is DERIVED, not searched. `_lay` collapses `tp_p` to 1 at BLOCK_P=1
    regardless, so a tuned TPW_L there has zero timing effect. Above BLOCK_P=1 exactly one value is ever right: 32, which
    pins tp_p=1 so BLOCK_P lands in size_per_thread instead of splitting the
    warp across p and replicating every p-free vector.

    PF is searched rather than fixed by a rule, because its sign genuinely depends on
    BOTH arch and geometry (on sm_120 it loses at BLOCK_P=1 and wins at
    BLOCK_P>=2; on sm_90 it wins in bf16 and loses in fp32), which a rule gets
    wrong and a search does not.
    """
    # BLOCK_L must stay a multiple of the checkpoint stride: the forward's
    # scatter indexes slot `jj * (BLOCK_L // CKPT_L) + col // CKPT_L`, so a
    # BLOCK_L that is not a multiple collapses distinct sub-chunks onto one
    # slot and the launcher rejects it outright.
    top = triton.next_power_of_2(L)
    ok_l = [b for b in (128, 256, 512, 1024, 2048)
            if b <= top and (not ckpt_l or b % ckpt_l == 0)]
    axes = {}
    if ts_axis and tier == "full":
        # The fitted unsplit forward usually wins, so split exploration is
        # full-tier only (it compiles/times the carry helpers). Explicit TS
        # configs still launch at every tier. Backward TS stays in light.
        # the forward time split (`ts_forward`), FIRST: it sets how many
        # CTAs walk how far, which the other axes are fitted against
        # unfiltered, so 1 stays the way BACK once the corners are re-fitted
        axes["TS"] = [2, 4, 8, 16, 1]
    axes["BLOCK_L"] = [b for b in _pow2_span(base["BLOCK_L"], 32, 2048)
                       if b in ok_l]
    axes["PF"] = [1 - int(base.get("PF", 0))]
    if tier == "full":
        # 1-warp scans usually win; `light` skips
        axes["NUM_WARPS"] = [w for w in (1, 2, 4) if w != base["NUM_WARPS"]]
    return {k: v for k, v in axes.items() if v}


def t_grid(base, P, L, ckpt_l):
    """The coupled (BLOCK_P, BLOCK_L) corners, for `choose_axes(grid=...)`.

    These two are the pair a coordinate descent cannot handle. They set the
    register tile TOGETHER -- `BLOCK_P * BLOCK_L / (32 * NUM_WARPS)` elements
    per thread -- so from a base at (1, 512) every single-axis step is
    downhill: raising BLOCK_P alone quadruples the tile into the register
    ceiling, and lowering BLOCK_L alone throws away the load width without
    buying the p reuse that would pay for it. Only the joint move (4, 512) with
    TPW_L pinned wins, and no descent from (1, 512) can see it.

    Kept small on purpose: one row per BLOCK_P at the two element-per-thread
    budgets that ever win (the base's, and the base's halved), which is 6-8
    trials rather than the 25 a full cross product would cost. Every trial is a
    Gluon compile and this runs once per shape before the cache takes over.
    """
    top = triton.next_power_of_2(L)
    out = []
    for bp in (1, 2, 4, 8):
        if P % bp:
            continue
        # the two element-per-thread budgets that ever win, 16 and 32 at one
        # warp (BLOCK_L = 512/BP and 1024/BP); the base itself is trial 0.
        # (base, base/BP) alone would skip 32/thread at BP4 from a BL512 base.
        for bl in (max(512 // bp, 128), max(1024 // bp, 128)):
            if bl > top or (ckpt_l and bl % ckpt_l):
                continue
            # PF rides the corners too: its sign flips with BLOCK_P, so a
            # descent that fixes BLOCK_P first and tries PF afterwards can
            # only find PF's best AT the winning BLOCK_P and misses joint
            # points such as (BP2, PF1).
            # P=1 needs both prefetch corners too: BL512/PF1 can beat
            # BL1024/PF1 although BL512/PF0 loses. Omitting that corner
            # traps the subsequent one-axis descent at BL1024.
            for pf in (0, 1):
                o = dict(BLOCK_P=bp, BLOCK_L=bl, NUM_WARPS=1, PF=pf)
                if o not in out:
                    out.append(o)
    return out


def light_fwd_candidate(cfg):
    """Keep multi-channel tiles within light's coupled-corner register budget.

    The BLOCK_L axes are independent of BLOCK_P, so at BP4 they would offer
    128/256 elements per thread against the grid's 16/32; up to 64/thread is
    allowed. Full retains the larger boundary choices. P1 keeps its full
    length range, including the long-sequence cells.
    This is a search-cost filter, not a kernel legality restriction.
    """
    bp = cfg.get("BLOCK_P", 1)
    return bp == 1 or bp * cfg["BLOCK_L"] <= 2048 * cfg["NUM_WARPS"]


def v3_grid(base, H, L, block_l_of, caps=(128,), half_block_l=True):
    """The coupled (HP, BLOCK_L) corners of the packed schedule.

    Same reason the forward has `t_grid`: HP and BLOCK_L set the register
    tile TOGETHER (`HP * BLOCK_L` elements), so from a base at (8, 64) every
    single-axis step is downhill -- doubling HP alone doubles the tile into
    the spill, and halving BLOCK_L alone throws away the load width without
    buying the head reuse that pays for it; the good corners sit near
    HP * BLOCK_L = 512, which no descent from an off-diagonal one can see.

    HP = 1 is in the list as the way OUT: a shape where packing does not pay
    must be able to reach the ordinary schedule, and `block_l_of` gives it
    the chunk length that schedule would have chosen.

    `caps` makes the REGISTER CAP part of the same coupled corner,
    for the same reason HP and BLOCK_L are: `maxnreg` and the tile size
    trade against each other and neither single-axis step from an uncapped
    corner can see the pair.  `_mh_bwd_kernel` sits at 255 registers with
    ONE warp per CTA, i.e. 8 resident warps per SM out of 64 and no shared
    memory to trade against it; capping at 128 doubles that residency and
    pays for the spill it causes -- but ONLY on some shapes, which is
    exactly the kind of shape-dependent trade a search should price and a
    constant should not.  Two corners per cap, at the incumbent pack width:
    the cap at the base chunk, and at HALF of it, because a smaller tile is
    how the kernel gets under the cap without spilling more (the cap is the
    lever and the narrower chunk is its consequence).
    """
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
        # TP_P is coupled to HP too: it sets how many lanes share a head row,
        # so the p-free [BLOCK_L] vectors cost BLOCK_L*TP_P/32 registers each
        # and HP/TP_P rows land per thread.  The narrow p-slice is offered at
        # each mid pack width because descent from the incumbent cannot reach
        # it: neither single step from (8, 4) sees the (4, 2) corner.
        if hp in (4, 8, 16) and base.get("TP_P") not in (None, 2):
            cand = dict(HP=hp, BLOCK_L=bl, TP_P=2)
            if cand not in out:
                out.append(cand)
    if out and base.get("MAXNREG") is None:
        hp, bl = out[0]["HP"], out[0]["BLOCK_L"]
        # `half_block_l=False` where BLOCK_L is not this search's to move: a
        # caller-supplied `hck` makes it the FORWARD's checkpoint stride, and
        # the backward reads slot j for chunk j with no stride conversion, so
        # the halved corner is a wrong gradient with a right shape rather than
        # a cheaper tile.  The cap is still priced -- at the pinned chunk.
        for cap in caps:
            for b in ((bl, max(8, bl // 2)) if half_block_l else (bl,)):
                cand = dict(HP=hp, BLOCK_L=b, MAXNREG=cap)
                if cand not in out:
                    out.append(cand)
    return out


def v3_axes(base, nr, rot, xs_bytes1, ckpt_bytes, budget=2 << 30,
            tune_block_l=True, bc_axis=False, ts_axis=False):
    """Search space for `mh_backward`, ordered by leverage.

    Two knobs are budget-guarded because they buy speed with gigabytes and the
    caller may not have them:

    * `NS` shards the n loop over an NS-times-larger grid, and `x_s` is
      (NS,B,H,P,L) fp32 -- NS x twice the size of `x`. `xs_bytes1` is that
      buffer at NS=1.
    * `BLOCK_L` is also the checkpoint stride, so halving it doubles `hck`.

    The `nr // 8` floor and the `NS <= 8` cap in the launcher's heuristic are
    NOT reimposed here. They are fitted on one GPU, and baking
    them into the search would reintroduce the staleness this autotuner exists
    to prevent. The budget, not a fitted constant, is what
    keeps the space honest.

    `AGX` goes LAST and is the one axis that costs time to gain bytes: it folds
    the NS `x_s` slabs into one with atomics, trading a little time for
    a large peak reduction. It is the default, so offering 0 here is offering the
    search a way OUT of it -- taken only where the cost is real, and `choose_axes`' `tol` is what
    decides "real". Last, because it changes what NS costs but not what NS is
    worth, so the NS descent above must already have settled.
    """
    axes = {}
    if ts_axis:
        # TS: the time split, FIRST -- it changes the parallelism regime the
        # other axes are fitted in.  Wins only where the chunk walk limits
        # parallelism (long L, small B*D/HP); never a default.
        # the base's own value is NOT filtered out (descent skips it anyway):
        # 1 is the way BACK when the pre-grid pass took a split that the
        # re-fitted corners no longer need
        axes["TS"] = [4, 8, 16, 32, 2, 1]
    if tune_block_l:
        axes["BLOCK_L"] = [b for b in _pow2_span(base["BLOCK_L"], 8, 256)
                           if ckpt_bytes(b) <= budget]
    ns = [n for n in (1, 2, 4, 8, 16, 32)
          if n != base["NS"] and nr % n == 0
          and (not rot or (nr // n) % 2 == 0)
          and n * xs_bytes1 <= budget]
    if ns:
        axes["NS"] = ns
    # down to 2: the rotating backward's best at P=16 (bwd_gluon TP_P note)
    axes["TP_P"] = _pow2_span(base["TP_P"], 2, 32)
    axes["NUM_WARPS"] = [w for w in (1, 2, 4) if w != base["NUM_WARPS"]]
    axes["AGX"] = [v for v in (0, 1) if v != base["AGX"]]
    if bc_axis:
        # BCS=1 folds the (H,B,N,L) BC slot buffers to one slab each --
        # gigabytes at H=128 -- and with the vector red (VBC) the fold's
        # instruction cost is stores' again, so it is a candidate the
        # byte-aware `_wins` can weigh rather than a priced-out trade.
        # Offered, like AGX, only where legal (`bc_axis` = no beta_* reads
        # the buffer back, and the vector red exists) and after it: it moves
        # the same time-for-bytes direction. 0 is the full slot axis, the way
        # OUT for a base that already folds.
        #
        # The INTERMEDIATE slab counts are offered too: at large H the ends
        # (1 and H slabs) are far apart and the middle is the only reachable
        # part of the axis. `_wins`' growth term prices it.
        #
        # Under the PACKED schedule more slabs do not help, because packing
        # already divides the number of CTAs contending for one slab by HP.
        # The axis is kept for the unpacked shapes.
        axes["BCS"] = [v for v in (1, 8, 16, 64, 0) if v != base["BCS"]]
    return {k: v for k, v in axes.items() if v}


def _dedup(cfgs):
    seen, out = set(), []
    for c in cfgs:
        t = tuple(sorted(c.items()))
        if t not in seen:
            seen.add(t)
            out.append(c)
    return out


def fwd_candidates(base, P, block_n, divisor_le):
    """BLOCK_L x TPW_L x NUM_WARPS x BLOCK_P around the heuristic.

    BLOCK_P is swept downward only: it trades p-free reuse for grid width, and
    the heuristic already sits at the largest value that keeps the grid full, so
    the open question is only ever whether giving reuse back buys parallelism.
    """
    out = [dict(base)]
    for block_l in (32, 64):
        for tpw_l in (4, 8):
            if tpw_l < 32 // block_n or tpw_l > block_l:
                continue
            for warps in (4, 8):
                for div in (1, 4):
                    out.append(dict(base, BLOCK_L=block_l, TPW_L=tpw_l,
                                    NUM_WARPS=warps,
                                    BLOCK_P=divisor_le(P, max(1, base["BLOCK_P"] // div))))
    return _dedup(out)


def bwd_candidates(base, block_n, ckpt_bytes=None, budget=2 << 30):
    """BLOCK_L x TPW_L x NUM_WARPS around the heuristic.

    Small BLOCK_L shrinks the ~20 live [BLOCK_N, BLOCK_L] tiles, but it also
    INFLATES the checkpoint buffer, which is `(B,H,P,cdiv(L,BLOCK_L)+1,N)` f32 --
    at the reference shape BLOCK_L=16 wants 10.8 GB. A trial that OOMs is
    survivable (choose() skips it) but it can fragment the allocator for the
    real call, so candidates over `budget` are never offered. The heuristic
    config is exempt: it is the caller's own choice and must stay reachable.
    """
    out = [dict(base)]
    for block_l in (16, 32, 64):
        if ckpt_bytes is not None and ckpt_bytes(block_l) > budget:
            continue
        for tpw_l in (4, 8):
            if tpw_l < 32 // block_n or tpw_l > block_l:
                continue
            for warps in (4, 8):
                out.append(dict(base, BLOCK_L=block_l, TPW_L=tpw_l,
                                NUM_WARPS=warps))
    return _dedup(out)
