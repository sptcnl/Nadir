"""Step 3b gate verdict: join the two per-patch raw-metric CSVs and apply the
pre-registered tolerances (docs/emrdm_reevaluation.md §3.1).

    python scripts/reeval/compare_raw.py --theirs raw_emrdm.csv --ours raw_nadir.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

# Pre-registered per-patch tolerances. SSIM is recorded, not gated (different
# implementations by design); its 0.05 bound is a sanity check only.
GATED = {"SAM": 0.01, "PSNR": 0.01, "MAE": 1e-5, "RMSE": 1e-5}
SSIM_SANITY = 0.05


def load(path: str) -> dict[int, dict[str, float]]:
    with open(path, newline="") as fh:
        return {
            int(row["patch"]): {k: float(v) for k, v in row.items() if k != "patch"}
            for row in csv.DictReader(fh)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theirs", required=True)
    parser.add_argument("--ours", required=True)
    args = parser.parse_args()

    theirs = load(args.theirs)
    ours = load(args.ours)
    if theirs.keys() != ours.keys():
        sys.exit(
            f"GATE FAIL: patch sets differ (theirs {len(theirs)}, ours {len(ours)}, "
            f"symmetric diff {sorted(set(theirs) ^ set(ours))[:10]}...)"
        )

    worst: dict[str, float] = {k: 0.0 for k in list(GATED) + ["SSIM"]}
    failures: list[str] = []
    for patch in sorted(theirs):
        for metric in worst:
            delta = abs(theirs[patch][metric] - ours[patch][metric])
            worst[metric] = max(worst[metric], delta)
            if metric in GATED and delta > GATED[metric]:
                failures.append(f"patch {patch}: |d{metric}|={delta:.6g} > {GATED[metric]}")

    print(f"patches compared: {len(theirs)}")
    for metric, value in worst.items():
        limit = GATED.get(metric, SSIM_SANITY)
        tag = "(gated)" if metric in GATED else "(recorded)"
        print(f"  worst |d{metric}| = {value:.6g}  limit {limit} {tag}")
    if failures:
        print(f"GATE FAIL: {len(failures)} violations, first 10:")
        for line in failures[:10]:
            print("  " + line)
        sys.exit(1)
    if worst["SSIM"] > SSIM_SANITY:
        print(f"NOTE: SSIM delta {worst['SSIM']:.4f} exceeds sanity bound {SSIM_SANITY} "
              "(recorded, not gating)")
    print("GATE PASS")


if __name__ == "__main__":
    main()
