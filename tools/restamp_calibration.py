#!/usr/bin/env python
"""Re-stamp shipped calibration files with the CURRENT kernel code hash, every entry unchanged (`calibration.restamp`).

ONLY for a kernel-source change that is BIT-IDENTICAL for every configuration the files hold -- e.g. a constexpr branch that none
of the files' groups compiles -- and only with the evidence named in --note (a PTX identity check of the affected kernels).  Any
other source change needs a recalibration (`tools/calibrate_kernels.py`, `tools/calibrate_levers.py`, ...).

    python tools/restamp_calibration.py --family ssd_mimo --from 00d5c5424af5e5f0 --note "..."     # every GPU's shipped file
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/src")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--from", dest="old", required=True, help="the old code hash the files must carry")
    ap.add_argument("--note", required=True, help="why the change is bit-identical for the files' groups, and the evidence")
    a = ap.parse_args()
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import (calibration_chunkpar, calibration_fused, calibration_ssd,  # noqa: F401
                                             levers, rot_route)
    n = 0
    for f in sorted(glob.glob(os.path.join(cal.shipped_dir(a.family), "*.json"))):
        d = json.load(open(f))
        if d["provenance"].get("code_hash") == cal.code_hash(a.family):
            print("current already:", f)
            continue
        cal.write_json(f, cal.restamp(d, a.family, a.old, a.note))
        print("restamped:", f)
        n += 1
    print("%d file(s) restamped to %s" % (n, cal.code_hash(a.family)))


if __name__ == "__main__":
    main()
