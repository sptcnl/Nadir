"""Gate 0 — reproduction determinism (emrdm_reevaluation.md §2.3.2).

EMRDM uses a fixed seed + EDM sampler and must be deterministic: the
prediction-saving inference pass (emrdm_infer_scene.py over the 9 scenes)
must aggregate to the SAME metrics as the authoritative Arm A run
(main.py test loop). This is "same code run twice", so the bar is TIGHT
(third decimal), unlike the pre-registered cross-implementation tolerances.

Compares the pooled per-patch mean of the 9 emrdm_infer metrics.csv files
against main.py's final_* values.

    python scripts/reeval/gate0_determinism.py --infer-base ~/logs/arm_a_infer \
        --armA-metrics ~/logs/arm_a/<run>/testtube/version_0/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

SCENES = [
    ("ROIs1158_spring", 31), ("ROIs1158_spring", 44), ("ROIs1158_spring", 106),
    ("ROIs1158_spring", 123), ("ROIs1158_spring", 140), ("ROIs1868_summer", 119),
    ("ROIs1970_fall", 139), ("ROIs2017_winter", 63), ("ROIs2017_winter", 108),
]
# Determinism bar: agree to the third decimal (same code, fixed seed).
DET_TOL = {"SAM": 0.001, "PSNR": 0.001, "SSIM": 0.0005, "MAE": 0.0005}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-base", required=True)
    parser.add_argument("--armA-metrics", required=True)
    args = parser.parse_args()

    # pool per-patch metrics from the prediction-saving pass
    pooled: dict[str, list[float]] = {"MAE": [], "PSNR": [], "SAM": [], "SSIM": []}
    for season, scene in SCENES:
        csv_path = Path(args.infer_base) / f"{season}_{scene}" / "metrics.csv"
        if not csv_path.exists():
            sys.exit(f"missing {csv_path}")
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                for k in pooled:
                    pooled[k].append(float(row[k]))
    n = len(pooled["SAM"])

    # authoritative Arm A (main.py) final_* values
    with open(args.armA_metrics, newline="") as fh:
        arm = list(csv.DictReader(fh))[-1]
    ref = {m: float(arm[f"final_{m}"]) for m in pooled}

    print(f"pooled patches (replica): {n}")
    print(f"{'metric':6s} {'replica':>13s} {'mainpy':>13s} {'|delta|':>11s} {'tol':>8s}  verdict")
    ok = True
    for m in ("PSNR", "SSIM", "SAM", "MAE"):
        rep = statistics.fmean(pooled[m])
        d = abs(rep - ref[m])
        passed = d <= DET_TOL[m]
        ok &= passed
        print(f"{m:6s} {rep:13.6f} {ref[m]:13.6f} {d:11.6f} {DET_TOL[m]:8.4f}  {'PASS' if passed else 'FAIL'}")
    if n != 7116:
        print(f"WARNING: replica pooled {n} patches, expected 7116")
    print("GATE0_DETERMINISTIC" if ok else "GATE0_NONDETERMINISTIC")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
