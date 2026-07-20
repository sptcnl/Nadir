"""Internal-consistency gate (emrdm_reevaluation.md §2.3): aggregate EMRDM's
own img_metrics and the Nadir harness metrics over the SAME predictions across
the 9 complete test scenes, and apply the pre-registered tolerances.

Inputs are the per-scene CSVs produced by:
  - emrdm_infer_scene.py  -> <scene>/metrics.csv  (their img_metrics per patch)
  - nadir_eval_preds.py   -> <scene>_nadir.csv    (our harness per patch)

Aggregation matches EMRDM's on_test_epoch_end (unweighted mean of per-patch
values), over all 7,116 patches pooled.

    python scripts/reeval/internal_consistency.py --infer-base ~/logs/arm_a_infer \
        --nadir-base outputs/arm_a_nadir --report outputs/internal_consistency.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# Gated metrics use identical formulas across implementations. SSIM is NOT
# gated — it was pre-registered as recorded-only in the 3b gate
# (compare_raw.py, committed 2026-07-17): theirs is pytorch_ssim (Po-Hsun-Su
# gaussian 11x11), ours is skimage (uniform 7x7), so they differ BY DESIGN
# (~0.06 per-patch, design_decisions.md §2.3). SSIM carries a 0.05 sanity
# bound only. This matches the pre-registration; it is not a post-hoc change.
TOL = {"SAM": 0.05, "PSNR": 0.10, "MAE": 0.001}
SSIM_SANITY = 0.05
SCENES = [
    ("ROIs1158_spring", 31), ("ROIs1158_spring", 44), ("ROIs1158_spring", 106),
    ("ROIs1158_spring", 123), ("ROIs1158_spring", 140), ("ROIs1868_summer", 119),
    ("ROIs1970_fall", 139), ("ROIs2017_winter", 63), ("ROIs2017_winter", 108),
]


def pool(rows_by_scene: list[Path], keys: tuple[str, ...]) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = {k: [] for k in keys}
    n = 0
    for path in rows_by_scene:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                n += 1
                for k in keys:
                    acc[k].append(float(row[k]))
    acc["_n"] = [float(n)]
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-base", required=True, help="EMRDM per-scene metrics dir")
    parser.add_argument("--nadir-base", required=True, help="Nadir per-scene metrics dir")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    infer = Path(args.infer_base)
    nadir = Path(args.nadir_base)

    their_csvs = [infer / f"{s}_{n}" / "metrics.csv" for s, n in SCENES]
    our_csvs = [nadir / f"{s}_{n}_nadir.csv" for s, n in SCENES]
    for p in their_csvs + our_csvs:
        if not p.exists():
            sys.exit(f"missing input CSV: {p}")

    metrics = ("MAE", "PSNR", "SAM", "SSIM")
    their = pool(their_csvs, metrics)
    ours = pool(our_csvs, metrics)
    n_their, n_ours = int(their["_n"][0]), int(ours["_n"][0])
    if n_their != n_ours:
        sys.exit(f"patch-count mismatch: EMRDM {n_their} vs Nadir {n_ours}")

    print(f"pooled patches: {n_their}")
    header = f"{'metric':6s} {'EMRDM':>13s} {'Nadir':>14s} {'|delta|':>10s} {'limit':>7s}  verdict"
    print(header)
    result: dict[str, object] = {"n_patches": n_their, "metrics": {}}
    ok = True
    for m in ("PSNR", "SSIM", "SAM", "MAE"):
        a = statistics.fmean(their[m])
        b = statistics.fmean(ours[m])
        d = abs(a - b)
        gated = m in TOL
        limit = TOL[m] if gated else SSIM_SANITY
        passed = d <= limit
        if gated:
            ok &= passed
        tag = "PASS" if passed else ("FAIL" if gated else "OVER-SANITY")
        kind = "gated" if gated else "recorded"
        result["metrics"][m] = {"emrdm": a, "nadir": b, "abs_delta": d,
                                "limit": limit, "gated": gated, "pass": passed}
        print(f"{m:6s} {a:13.6f} {b:14.6f} {d:10.6f} {limit:7.3f}  {tag} ({kind})")
    result["verdict"] = "CONSISTENT" if ok else "INCONSISTENT"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2))
    print("HARNESS_INTERNALLY_CONSISTENT" if ok else "HARNESS_INCONSISTENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
