"""Step 3d: consume EMRDM prediction files (uint16 DN .npz) with the Nadir
harness and compare per-patch metrics against EMRDM's in-run metrics CSV.

This validates the cross-venv file handoff end to end and measures the cost
of the uint16 quantization in the handoff format (their in-run metrics were
computed on float predictions; the files are round(refl*10000) uint16).

    .venv/Scripts/python scripts/reeval/nadir_eval_preds.py \
        --preds <dir with p<patch>.npz> --theirs <their metrics.csv> \
        --root <SEN12MS-CR root> --season ROIs2017_winter --scene 63 \
        --out outputs/eval_preds_winter63.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
import torch

from nadir.data.preprocess import denormalize_s2, normalize_s2
from nadir.metrics import MetricSuite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--theirs", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--season", default="ROIs2017_winter")
    parser.add_argument("--scene", type=int, default=63)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preds_dir = Path(args.preds)
    root = Path(args.root)
    with open(args.theirs, newline="") as fh:
        theirs = {int(r["patch"]): r for r in csv.DictReader(fh)}

    suite = MetricSuite(lpips_net=None)
    rows = []
    worst: dict[str, float] = {"SAM": 0.0, "PSNR": 0.0, "MAE": 0.0}
    for npz_path in sorted(preds_dir.glob("p*.npz"), key=lambda p: int(p.stem[1:])):
        patch = int(npz_path.stem[1:])
        dn = np.load(npz_path)["dn"]
        pred = torch.from_numpy(dn.astype(np.float32) / 10000.0).unsqueeze(0)
        gt_path = (
            root / f"{args.season}_s2" / f"s2_{args.scene}"
            / f"{args.season}_s2_{args.scene}_p{patch}.tif"
        )
        with rasterio.open(gt_path) as src:
            gt_dn = src.read()
        target = torch.from_numpy(denormalize_s2(normalize_s2(gt_dn))).unsqueeze(0)
        mask = torch.zeros(1, 256, 256)
        m = suite(pred, target, mask)
        ours = {"SAM": m["sam/full"], "PSNR": m["psnr/full"], "MAE": m["mae/full"]}
        rows.append({"patch": patch, **ours, "SSIM": m["ssim/full"]})
        for k in worst:
            worst[k] = max(worst[k], abs(ours[k] - float(theirs[patch][k])))
        if len(rows) % 200 == 0:
            print(f"{len(rows)} patches done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"n={len(rows)}; worst |delta| vs their in-run metrics "
          f"(includes uint16 handoff quantization):")
    for k, v in worst.items():
        print(f"  {k}: {v:.6f}")
    print("HANDOFF_EVAL_DONE")


if __name__ == "__main__":
    main()
