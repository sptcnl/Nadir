"""Diagnose the model-vs-no-model SAM regression before recording it as a
finding (protocol.md §4.3 follow-up).

The epoch-59 checkpoint is overfit (val SAM bottoms at epoch ~10 then climbs
back). Reporting its held-out SAM as evidence of a "PSNR-up / SAM-down" thesis
would conflate the thesis with an overfitting artifact. This script separates
the two by sweeping several checkpoints on the in-domain (spring) held-out
subset in ONE data pass, comparing each against the do-nothing baseline
(pred := cloudy). SAM/PSNR/MAE only (all GPU-vectorized); SSIM/LPIPS omitted —
SAM is the metric under scrutiny.

    python scripts/investigate_sam.py \
        --test-root //wsl.localhost/Ubuntu-24.04/home/sptcnl/data/sen12mscr \
        --ckpt-dir outputs/spring6_train/checkpoints \
        --epochs 6 10 20 59 --season spring
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nadir.data.sen12mscr import Sen12MSCRDataset
from nadir.metrics.suite import mae_map, mse_map, psnr_from_mse, region_mean, sam_map
from nadir.models.dsen2cr import DSen2CR
from nadir.utils import resolve_device

SEASON_OF = {
    "ROIs1158": "spring", "ROIs1868": "summer",
    "ROIs1970": "fall", "ROIs2017": "winter",
}


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def load_model(path: Path, device: torch.device) -> DSen2CR:
    state = torch.load(path, map_location=device, weights_only=False)
    m = state["config"]["model"]
    model = DSen2CR(
        in_channels=m["in_channels"], out_channels=m["out_channels"],
        features=m["features"], num_blocks=m["num_blocks"], res_scale=m["res_scale"],
    ).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--season", default="spring",
                    help="restrict to one season (default spring, in-domain); "
                         "'all' keeps every season")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="parallel DataLoader workers; >0 hides WSL network read "
                         "latency (this eval is I/O-bound, not GPU-bound)")
    args = ap.parse_args()

    device = resolve_device("cuda")
    ckpt_dir = Path(args.ckpt_dir)
    models = {f"ep{e}": load_model(ckpt_dir / f"epoch{e:04d}.pt", device)
              for e in args.epochs}
    arms = ["no-model", *models.keys()]

    ds = Sen12MSCRDataset(Path(args.test_root), augment=False, strict=False)
    # Prune to the target season at the triplet level so __getitem__ only READS
    # those patches (scan_triplets globs paths without reading rasters). This is
    # the fix for the earlier run reading all 7,116 patches to score 3,983.
    if args.season != "all":
        want = {r for r, s in SEASON_OF.items() if s == args.season}
        ds.triplets = [t for t in ds.triplets if t.roi.split("_")[0] in want]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    n_batches = (len(ds.triplets) + args.batch_size - 1) // args.batch_size
    print(f"season={args.season}: {len(ds.triplets)} patches, {n_batches} batches, "
          f"arms={['no-model', *models.keys()]}", flush=True)
    t0 = time.time()

    # per-arm running sums of per-image SAM/PSNR/MAE, full region. PSNR is the
    # mean of per-image PSNR (matches MetricSuite aggregation), not PSNR of the
    # mean MSE.
    sam_sum = defaultdict(float)
    psnr_sum = defaultdict(float)
    mae_sum = defaultdict(float)
    n = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi % 25 == 0 and bi:
                rate = n / (time.time() - t0)
                eta = (len(ds.triplets) - n) / rate if rate else 0
                print(f"  batch {bi}/{n_batches}  {n} patches  "
                      f"{rate:.1f} patch/s  ETA {eta:.0f}s", flush=True)
            keep = [i for i in range(len(batch["roi"]))
                    if args.season == "all"
                    or SEASON_OF[batch["roi"][i].split("_")[0]] == args.season]
            if not keep:
                continue
            idx = torch.tensor(keep, device=device)
            s1 = batch["s1"].to(device)[idx]
            s2c = batch["s2_cloudy"].to(device)[idx]
            clear = denorm(batch["s2_clear"].to(device)[idx])
            region = torch.ones(s1.shape[0], s1.shape[2], s1.shape[3],
                                dtype=torch.bool, device=device)
            n += s1.shape[0]
            for arm in arms:
                pred = denorm(s2c) if arm == "no-model" else denorm(models[arm](s2c, s1))
                sam_sum[arm] += float(region_mean(sam_map(pred, clear), region).sum())
                psnr_sum[arm] += float(psnr_from_mse(region_mean(mse_map(pred, clear), region)).sum())
                mae_sum[arm] += float(region_mean(mae_map(pred, clear), region).sum())

    print(f"season={args.season}  n={n}\n")
    print(f"{'arm':>9} {'SAM':>8} {'PSNR':>8} {'MAE':>8}   vs no-model SAM")
    base_sam = sam_sum["no-model"] / n
    for arm in arms:
        sam = sam_sum[arm] / n
        psnr = psnr_sum[arm] / n
        mae = mae_sum[arm] / n
        delta = "" if arm == "no-model" else f"  {'WORSE' if sam > base_sam else 'better'} ({sam - base_sam:+.2f})"
        print(f"{arm:>9} {sam:8.3f} {psnr:8.3f} {mae:8.4f}{delta}")


if __name__ == "__main__":
    main()
