"""H1 verdict (emrdm_reevaluation.md §5.1, pre-registered decision rule).

Reads the B1 per-scene metrics.csv tree
(<base>/seed<seed>/<vh25|vh325>/<season>_<scene>/metrics.csv), pools SAM
per (seed, condition) over the 7,116 patches and per season, computes the
paired VH effect per seed, and applies the pre-registered rule.

VHeff(s)   = SAM_agg(vh325, s) − SAM_agg(vh25, s)     (paired by seed)
mean/ sd   over seeds; per-season VHeff likewise.
Rule: |mean| > 0.527 and |mean| > sd -> SUPPORTED
      sd < |mean| <= 0.527           -> REJECTED (protocol robust)
      |mean| <= sd                   -> INCONCLUSIVE
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PAPER_GAP = 0.527  # DB-CR 4.740 vs EMRDM 5.267
SEEDS = [3407, 0, 42]
SCENES = [
    ("ROIs1158_spring", 31), ("ROIs1158_spring", 44), ("ROIs1158_spring", 106),
    ("ROIs1158_spring", 123), ("ROIs1158_spring", 140), ("ROIs1868_summer", 119),
    ("ROIs1970_fall", 139), ("ROIs2017_winter", 63), ("ROIs2017_winter", 108),
]
SEASON = {"ROIs1158_spring": "spring", "ROIs1868_summer": "summer",
          "ROIs1970_fall": "fall", "ROIs2017_winter": "winter"}


def sams(base: Path, seed: int, cond: str) -> dict[str, list[float]]:
    """Return {'all': [...], season: [...]} per-patch SAM lists."""
    out: dict[str, list[float]] = {"all": [], "spring": [], "summer": [],
                                   "fall": [], "winter": []}
    for season, scene in SCENES:
        p = base / f"seed{seed}" / cond / f"{season}_{scene}" / "metrics.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                v = float(row["SAM"])
                out["all"].append(v)
                out[SEASON[season]].append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    base = Path(args.base)

    groups = ["all", "spring", "summer", "fall", "winter"]
    vheff: dict[str, list[float]] = {g: [] for g in groups}
    per_seed_agg = {}
    for seed in SEEDS:
        s25 = sams(base, seed, "vh25")
        s325 = sams(base, seed, "vh325")
        per_seed_agg[seed] = {
            "vh25_all": statistics.fmean(s25["all"]),
            "vh325_all": statistics.fmean(s325["all"]),
        }
        for g in groups:
            vheff[g].append(statistics.fmean(s325[g]) - statistics.fmean(s25[g]))

    print(f"paper gap (DB-CR vs EMRDM) = {PAPER_GAP}°   seeds = {SEEDS}")
    for seed in SEEDS:
        a = per_seed_agg[seed]
        print(f"  seed {seed}: SAM vh25={a['vh25_all']:.4f}  vh325={a['vh325_all']:.4f}  "
              f"VHeff={a['vh325_all'] - a['vh25_all']:+.4f}")

    result: dict[str, object] = {"paper_gap": PAPER_GAP, "seeds": SEEDS, "groups": {}}
    mean_all = statistics.fmean(vheff["all"])
    sd_all = statistics.stdev(vheff["all"]) if len(vheff["all"]) > 1 else 0.0
    print(f"\n{'group':8s} {'mean VHeff':>11s} {'sd(seed)':>9s}  interpretation")
    for g in groups:
        mean = statistics.fmean(vheff[g])
        sd = statistics.stdev(vheff[g]) if len(vheff[g]) > 1 else 0.0
        result["groups"][g] = {"mean_VHeff": mean, "sd_seed": sd, "per_seed": vheff[g]}
        note = ""
        if g == "all":
            note = "<- primary"
        print(f"{g:8s} {mean:+11.4f} {sd:9.4f}  {note}")

    if abs(mean_all) > PAPER_GAP and abs(mean_all) > sd_all:
        verdict = "H1_SUPPORTED"
    elif abs(mean_all) <= sd_all:
        verdict = "H1_INCONCLUSIVE"
    else:
        verdict = "H1_REJECTED"
    result["mean_VHeff_all"] = mean_all
    result["sd_seed_all"] = sd_all
    result["verdict"] = verdict
    print(f"\n|mean VHeff (all)| = {abs(mean_all):.4f}°   sd(seed) = {sd_all:.4f}°   "
          f"gap = {PAPER_GAP}°\nVERDICT: {verdict}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
