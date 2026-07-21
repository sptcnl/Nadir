"""Per-season evaluation of a trained DSen2-CR checkpoint on the held-out
9-scene test set (protocol.md §4.2, Step 5).

The spring-6 baseline is trained on ONE season, so the split that matters is
in-domain (spring) vs out-of-domain (summer/fall/winter): in-domain shows
whether training converged at all; out-of-domain shows the single-season
generalization limit. Reporting only the pooled aggregate would conflate the
two and invite misreading domain shift as model failure.

Metrics use the Nadir suite (SAM/PSNR/SSIM/MAE), full-image region. Summer's
incomplete triplets (scene-73 clear unrecoverable) are skipped (strict=False).

    python scripts/eval_per_season.py \
        --checkpoint outputs/spring6_train/checkpoints/last.pt \
        --test-root //wsl.localhost/Ubuntu-24.04/home/sptcnl/data/sen12mscr \
        --out outputs/spring6_per_season.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nadir.data.sen12mscr import Sen12MSCRDataset
from nadir.metrics import MetricSuite
from nadir.models.dsen2cr import DSen2CR
from nadir.utils import resolve_device

SEASON_OF = {
    "ROIs1158": "spring", "ROIs1868": "summer",
    "ROIs1970": "fall", "ROIs2017": "winter",
}
TRAIN_SEASON = "spring"  # spring-6 baseline


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = resolve_device("cuda")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = state["config"]["model"]
    model = DSen2CR(
        in_channels=mcfg["in_channels"], out_channels=mcfg["out_channels"],
        features=mcfg["features"], num_blocks=mcfg["num_blocks"], res_scale=mcfg["res_scale"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    ds = Sen12MSCRDataset(Path(args.test_root), augment=False, strict=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    suite = MetricSuite(device=device, lpips_net=args.lpips_net)

    # accumulate per-season sums of per-image metrics (full region)
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    keys = ("psnr/full", "sam/full", "ssim/full", "mae/full", "lpips/full")

    with torch.no_grad():
        for batch in loader:
            s1 = batch["s1"].to(device)
            s2c = batch["s2_cloudy"].to(device)
            clear = batch["s2_clear"].to(device)
            mask = batch["mask"].to(device)
            pred = model(s2c, s1)
            # per-sample metrics so we can bucket by season
            for i in range(s1.shape[0]):
                season = SEASON_OF[batch["roi"][i].split("_")[0]]
                m = suite(denorm(pred[i : i + 1]), denorm(clear[i : i + 1]), mask[i : i + 1])
                counts[season] += 1
                for k in keys:
                    if not math.isnan(m[k]):
                        sums[season][k] += m[k]

    report: dict[str, object] = {"train_season": TRAIN_SEASON, "per_season": {}, "counts": {}}
    for season in ("spring", "summer", "fall", "winter"):
        n = counts[season]
        if n == 0:
            continue
        report["counts"][season] = n
        report["per_season"][season] = {k: sums[season][k] / n for k in keys}

    # pooled (all seasons) for reference only
    tot = sum(counts.values())
    report["counts"]["ALL"] = tot
    report["per_season"]["ALL"] = {
        k: sum(sums[s][k] for s in counts) / tot for k in keys
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"{'season':8s} {'n':>5s} {'PSNR':>8s} {'SAM':>8s} {'SSIM':>7s} {'MAE':>8s}  domain")
    for season in ("spring", "summer", "fall", "winter", "ALL"):
        if season not in report["per_season"]:
            continue
        m = report["per_season"][season]
        if season == TRAIN_SEASON:
            dom = "IN-DOMAIN"
        elif season == "ALL":
            dom = "(pooled)"
        else:
            dom = "out-of-domain"
        print(f"{season:8s} {report['counts'][season]:5d} {m['psnr/full']:8.3f} "
              f"{m['sam/full']:8.3f} {m['ssim/full']:7.4f} {m['mae/full']:8.4f}  {dom}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
