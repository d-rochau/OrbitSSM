#!/usr/bin/env python
"""FAIL FAST on missing calibration (run it at the start of every benchmark and gate job).

A job whose tuned kernels silently fall back to the generic grid pays light-grid autotune compiles (~6 min/cell) and measures a
configuration production does not run.  This check runs at the START of such a job on the job's own GPU and exits 3 when, for any
family, the shipped/local file is missing or stale for this GPU / Triton / kernel code hash, or a registered kernel has no calibrated
entry (except a kernel the GPU's own route never launches: sm_90's Gluon SISO backward replaces `siso_bwd`).
    python tools/check_calibration.py            # exit 0 + one line per family when everything is calibrated
Env SSD_MIMO_CALIBRATION=off disables the requirement (that is an explicit uncalibrated measurement)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/src")


def main() -> int:
    if os.environ.get("SSD_MIMO_CALIBRATION", "").strip().lower() == "off":
        print("check_calibration: SSD_MIMO_CALIBRATION=off -> uncalibrated run requested explicitly")
        return 0
    import torch
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import (calibration_chunkpar, calibration_fused,  # noqa: F401
                                             calibration_ssd, levers, rot_route)
    dev = cal.device_info()
    gpu, tv = dev["gpu"], dev["triton"]
    gluon_siso = False
    try:
        from orbit_ssm.kernels.ssd_mimo import gluon_sm90_siso_bwd as g
        gluon_siso = bool(g.enabled(torch.device("cuda")))
    except Exception:                                   # noqa: BLE001
        pass
    bad = []
    skip = {x for x in os.environ.get("CHECK_CALIBRATION_SKIP", "").split(",") if x}   # a family being calibrated by THIS job
    for fam in ("ssd_mimo", "ssd_mimo_fused", "ssd_mimo_chunkpar", "ssd_mimo_levers", "ssd_mimo_route"):
        if fam in skip:
            print("check_calibration: %-18s SKIPPED (CHECK_CALIBRATION_SKIP: calibrated by this job)" % fam)
            continue
        data, layer = cal.load(fam, gpu, tv)
        if data is None:
            bad.append("%s: no valid file for %s (triton %s): %s" % (
                fam, gpu, tv, "; ".join(cal._STALE.get((fam, gpu)) or ["no file"])))
            continue
        unlaunched = set(data.get("unlaunched_kernels", []))        # explicit: no calibration cell launched it
        missing = [k for k in cal.family(fam)["kernels"]
                   if k not in data["kernels"] and k not in unlaunched]
        if fam == "ssd_mimo" and gluon_siso:
            missing = [k for k in missing if k != "siso_bwd"]        # the Gluon backward serves it
        if missing:
            bad.append("%s: kernels without a calibrated entry on %s: %s" % (fam, gpu, ", ".join(missing)))
        n_thr = len(data.get("thresholds", {}))
        if fam in ("ssd_mimo_chunkpar", "ssd_mimo_levers", "ssd_mimo_route") and not n_thr:
            bad.append("%s: file has no thresholds" % fam)
        print("check_calibration: %-18s %-7s %d kernels, %d thresholds, chunk=%s" % (
            fam, layer, len(data["kernels"]), n_thr, data.get("chunk")))
    if bad:
        print("CHECK_CALIBRATION FAILED on %s -- refusing to benchmark uncalibrated kernels:" % gpu)
        for b in bad:
            print("  " + b)
        return 3
    print("check_calibration: OK on %s" % gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
