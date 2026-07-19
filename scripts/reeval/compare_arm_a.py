"""Arm A gate verdict: compare the reproduced final_* metrics against EMRDM's
released test-log values under the pre-registered tolerances.

Targets (their released testtube metrics.csv, full precision — §3 of
emrdm_reevaluation.md):
    PSNR 32.13542556762695   SAM 5.266563415527344
    SSIM 0.9244527220726013  MAE 0.018326831981539726
Tolerances (pre-registered, may NOT be widened after the fact):
    SAM +-0.05 deg   PSNR +-0.10 dB   SSIM +-0.005   MAE +-0.001

Usage:
    python scripts/reeval/compare_arm_a.py --metrics <reproduced metrics.csv>
"""

from __future__ import annotations

import argparse
import csv
import sys

TARGETS = {"PSNR": 32.13542556762695, "SAM": 5.266563415527344,
           "SSIM": 0.9244527220726013, "MAE": 0.018326831981539726}
TOL = {"PSNR": 0.10, "SAM": 0.05, "SSIM": 0.005, "MAE": 0.001}


def read_final_metrics(path: str) -> dict[str, float]:
    """Read the last row; prefer final_* columns (their aggregate test metrics)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"no rows in {path}")
    row = rows[-1]
    out: dict[str, float] = {}
    for metric in TARGETS:
        for key in (f"final_{metric}", metric):
            if key in row and row[key] not in ("", "nan"):
                out[metric] = float(row[key])
                break
        else:
            sys.exit(f"metric {metric} not found in {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    got = read_final_metrics(args.metrics)

    print(f"{'metric':6s} {'reproduced':>14s} {'target':>14s} {'delta':>12s} {'tol':>8s}  verdict")
    ok = True
    for metric in ("PSNR", "SSIM", "SAM", "MAE"):
        delta = got[metric] - TARGETS[metric]
        passed = abs(delta) <= TOL[metric]
        ok &= passed
        print(f"{metric:6s} {got[metric]:14.6f} {TARGETS[metric]:14.6f} "
              f"{delta:+12.6f} {TOL[metric]:8.3f}  {'PASS' if passed else 'FAIL'}")
    print("ARM_A_REPRODUCED" if ok else "ARM_A_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
