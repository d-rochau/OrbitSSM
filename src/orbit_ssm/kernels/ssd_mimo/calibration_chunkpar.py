"""STRUCTURAL thresholds of the chunk-parallel real R1O1 route in the
calibration files (family `ssd_mimo_chunkpar`; no launch grids -- the CP
kernels' own launch configs stay with `ssd_mimo`'s adapters).

`tools/cp_calibrate.py` measures, per GPU, the serial-vs-chunk-parallel
crossover (waves x 100 and chunks per wave, forward and Ka), the big-chunk
sub-length `ksub`, the serial-Ka warps per occupancy bucket and the Kbc-hb
body schedule, and used to keep them only in the per-process strategy cache.
They are now ALSO exported into the shipped/local calibration file's
`thresholds` section under the cache key's suffix ("fwd", "fwd|ncpw",
"fwd|ksub", "ka_serial|warps|wb5", "kbc|sched", ...).  `smem.chunk_cache_lookup`
consults them for every `chunk:cp:<device>|<suffix>` key that the process
cache and the disk strategy cache do not answer, so `SSD_MIMO_CHUNKPAR=auto`
routes a calibrated GPU without any per-process measurement.  Active only when
the family's switch (`SSD_MIMO_CALIBRATION`) is on.
"""
import os
import time

from .. import calibration as cal

FAMILY = "ssd_mimo_chunkpar"
_HERE = os.path.dirname(os.path.abspath(__file__))

cal.register_family(FAMILY, env="SSD_MIMO_CALIBRATION",
                    code_files=[os.path.join(_HERE, "chunkpar.py")],
                    kernels=(), shapes={})


def lookup(cache_key):
    """The calibrated threshold for a `chunk:cp:<device>|<suffix>` cache key,
    or None (switch off, no valid file, or not calibrated)."""
    if "|" not in cache_key or not cal.enabled(FAMILY):
        return None
    suffix = cache_key.split("|", 1)[1]
    v, src = cal.threshold(FAMILY, suffix, None)
    return v


# ---- STRUCTURAL choice: the chunk length T of the real R = O = 1 route ------
#: thresholds (value = T) of group real_R1O1 at the calibrated head width
REAL_T_KEYS = {"train_fwd": "real_r1o1|T|train_fwd",    # forward of a training call
               "train_bwd": "real_r1o1|T|train_bwd",    # backward of a training call
               "fwd_only": "real_r1o1|T|fwd_only"}      # a call without backward
REAL_T_G = 64            # the head width the choice is measured at (P = 64)


#: the STRUCTURAL RULE over the call's size (the screen's winner changes with the call --
#: T = 64 on the balanced / batch cells, T = 32 on the long and the tiny ones -- so one T per GPU is not enough):
#:     T = T_hi  iff  L <= L_max  AND  work >= W_min   (training calls; work = B * H * L)
#:     T = the base `real_r1o1|T|<kind>` value otherwise
#: thresholds `real_r1o1|Trule|{T_hi, L_max, W_min}` of the chunkpar file (absent = no rule: the base T alone).
REAL_RULE_KEYS = {"T_hi": "real_r1o1|Trule|T_hi", "L_max": "real_r1o1|Trule|L_max", "W_min": "real_r1o1|Trule|W_min"}
RULE_KINDS = ("train_fwd", "train_bwd")


_RULE_MEMO = {}                  # dtype axis -> [loaded file dict (identity), the rule parsed from it, time of the last check, the mode env it was read under]
_RULE_TTL = 2.0                  # seconds: the file is re-checked as often as `calibration.load` revalidates it (a GPU's rule is a constant)

#: THE DTYPE AXIS.  fp32 IO (tf32 operands, torch's default dtype) has its OWN structure when the GPU's file stores it:
#: every key above with the suffix `|fp32` (`real_r1o1|T|train_fwd|fp32`, `real_r1o1|Trule|T_hi|fp32`, ...).  A file without the
#: fp32 keys keeps the 16-bit structure for fp32 calls; a file WITH fp32 T keys but no fp32
#: rule has no rule at fp32 (the 16-bit rule was fitted on 16-bit timings).
FP32_SUFFIX = "|fp32"


def dtype_axis(dtype) -> str:
    """"fp32" for fp32 IO (a torch dtype or the kernels' io tag), "" for the 16-bit IO dtypes."""
    import torch
    return "fp32" if dtype in (torch.float32, "fp32", "float32") else ""


@cal._const
def _fp32_calibrated() -> bool:
    data, _ = cal.load(FAMILY)
    th = (data or {}).get("thresholds", {})
    return any(k + FP32_SUFFIX in th for k in REAL_T_KEYS.values())


@cal._const
def real_rule(dtype=None):
    """(T_hi, L_max, W_min) of the calibrated structural rule of this GPU (at the call's dtype axis), or None.  `_const`: torch.compile evaluates it at trace time (its memo store and clock read would otherwise break the graph: the compiled mixer grew a drive-sized copy, `test_ssd_convlayout`).  Host-path cost: one clock
    read inside the TTL (every training call reads it at several lever sites)."""
    ax = dtype_axis(dtype) if dtype is not None else ""
    memo = _RULE_MEMO.setdefault(ax, [False, None, 0.0, None])
    now = time.monotonic()
    env = os.environ.get("SSD_MIMO_CALIBRATION")
    if now - memo[2] < _RULE_TTL and env == memo[3]:
        return memo[1]
    memo[2], memo[3] = now, env
    if not cal.enabled(FAMILY):
        memo[0], memo[1] = False, None
        return None
    data, _ = cal.load(FAMILY)
    if memo[0] is not data:
        th = (data or {}).get("thresholds", {})
        suf = FP32_SUFFIX if (ax == "fp32" and _fp32_calibrated()) else ""
        vals = [th.get(REAL_RULE_KEYS[k] + suf) for k in ("T_hi", "L_max", "W_min")]
        memo[0] = data
        memo[1] = None if any(v is None for v in vals) else tuple(int(v["value"]) for v in vals)
    return memo[1]


def real_chunk(kind, G=None, L=None, work=None, dtype=None):
    """The calibrated chunk length T of the real R = O = 1 route for `kind`
    in ("train_fwd", "train_bwd", "fwd_only"), or None (switch off, no valid
    file, not calibrated, or a head width other than the calibrated one).

    `L` / `work` (= B * H * L): the call's length and size; with the GPU's structural rule (`real_rule`) a TRAINING call inside it
    takes `T_hi`, any other call the base value.  `dtype`: the call's IO dtype (the fp32 axis, see `FP32_SUFFIX`)."""
    if G is not None and int(G) != REAL_T_G:
        return None
    if not cal.enabled(FAMILY):
        return None
    suf = FP32_SUFFIX if (dtype is not None and dtype_axis(dtype) == "fp32" and _fp32_calibrated()) else ""
    v, _ = cal.threshold(FAMILY, REAL_T_KEYS[kind] + suf, None)
    if v is None:
        return None
    if L is not None and work is not None and kind in RULE_KINDS:
        rule = real_rule(dtype)
        if rule is not None and int(L) <= rule[1] and int(work) >= rule[2]:
            return rule[0]
    return int(v)


# ---- STRUCTURAL choice per RANK: the chunk length T of the R = O = 2 / 4 routes, real and rotating ------------------------------
#: thresholds (value = T) `rank|<route>|R<R>O<O>|T|<kind>[|fp32]`, route in {real, cx}, written by the per-rank calibration (the R = O = 1
#: keys above are untouched).  Absent = the fixed chunk rule of `ssm.ssd_mimo._scan_chunk` (rank table + wide-W clause).  A call at
#: another head width than the entry's `kernel_width` (`REAL_T_G`), a rank without a stored key, or an uncalibrated dtype answers None.
RANK_T_KINDS = ("train_fwd", "train_bwd", "fwd_only")


def rank_key(real, R, O, kind, fp32=False):
    return "rank|%s|R%dO%d|T|%s%s" % ("real" if real else "cx", int(R), int(O), kind, FP32_SUFFIX if fp32 else "")


def rank_chunk(real, R, O, kind, G=None, dtype=None):
    """The calibrated chunk length of the R = O > 1 route (`real`: rotation-free) for `kind` in `RANK_T_KINDS`, or None (switch off, no valid
    file, rank / dtype not calibrated, other head width, or R != O)."""
    if int(R) < 2 or int(R) != int(O):
        return None
    if not cal.enabled(FAMILY):
        return None
    key = rank_key(real, R, O, kind, dtype is not None and dtype_axis(dtype) == "fp32")
    data, _ = cal.load(FAMILY)
    entry = (data or {}).get("thresholds", {}).get(key, {})
    if G is not None and int(G) != int(entry.get("kernel_width", REAL_T_G)):
        return None
    v, _ = cal.threshold(FAMILY, key, None)
    return None if v is None else int(v)


def export_from_cache(chunk_cache, device_name, rounds=5, write_shipped=False,
                      dropped=None):
    """Write every `chunk:cp:<device_name>|<suffix>` verdict of the strategy
    cache dict into this GPU's calibration file (`thresholds`), keeping the
    thresholds of an existing valid file of the same layer.  Returns the path."""
    prefix = "chunk:cp:%s|" % device_name
    th = {k[len(prefix):]: {"value": int(v), "evidence": "tools/cp_calibrate.py"}
          for k, v in chunk_cache.items() if k.startswith(prefix)}
    path = (cal.shipped_path if write_shipped else cal.local_path)(
        FAMILY, cal.device_info()["gpu"])
    old = cal._read(path)
    if old and cal.check_file(old, FAMILY, cal.device_info()["gpu"],
                              cal.device_info()["triton"]) is None:
        th = dict(old.get("thresholds", {}), **th)
    from .. import shape_sets as ss
    data = cal.build_file(FAMILY, {}, rounds,
                          [t for t in ss.SET if t not in (dropped or {})],
                          thresholds=th, dropped=dropped)
    return cal.write_json(path, data)
