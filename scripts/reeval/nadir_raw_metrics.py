"""Step 3b, Nadir side: per-patch no-model ("raw") metrics for one scene.

Same quantity as emrdm_raw_metrics.py (cloudy vs clear, [0,1] reflectance)
computed with Nadir's own preprocessing and metric suite. Runs in the MAIN
venv (Windows); reads the WSL ext4 data via the \\\\wsl.localhost UNC path.

    .venv/Scripts/python scripts/reeval/nadir_raw_metrics.py \
        --root \\\\wsl.localhost\\Ubuntu-24.04\\home\\sptcnl\\data\\sen12mscr \
        --season ROIs2017_winter --scene 63 --out outputs/raw_nadir_winter63.csv

RMSE is not part of nadir.metrics (PSNR is derived from region MSE there);
for this gate it is computed ad hoc below, as declared in the 3b gate table.
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


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--season", default="ROIs2017_winter")
    parser.add_argument("--scene", type=int, default=63)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    clear_dir = root / f"{args.season}_s2" / f"s2_{args.scene}"
    cloudy_dir = root / f"{args.season}_s2_cloudy" / f"s2_cloudy_{args.scene}"
    clear_files = sorted(clear_dir.glob("*.tif"))
    if not clear_files:
        raise FileNotFoundError(f"no patches under {clear_dir}")

    suite = MetricSuite(lpips_net=None)
    rows = []
    for clear_path in clear_files:
        cloudy_path = cloudy_dir / clear_path.name.replace("_s2_", "_s2_cloudy_")
        patch = int(clear_path.stem.rsplit("_p", 1)[1])
        # Nadir pipeline: DN -> [-1,1] (normalize_s2) -> [0,1] reflectance.
        clear = torch.from_numpy(denormalize_s2(normalize_s2(_read(clear_path)))).unsqueeze(0)
        cloudy = torch.from_numpy(denormalize_s2(normalize_s2(_read(cloudy_path)))).unsqueeze(0)
        mask = torch.zeros(1, clear.shape[-2], clear.shape[-1])  # full-image only
        m = suite(cloudy, clear, mask)
        rmse = float(torch.sqrt(((cloudy - clear) ** 2).mean()))
        rows.append(
            {"patch": patch, "MAE": m["mae/full"], "RMSE": rmse,
             "PSNR": m["psnr/full"],  # suite value, derived from region MSE
             "SAM": m["sam/full"], "SSIM": m["ssim/full"]}
        )
        if len(rows) % 100 == 0:
            print(f"{len(rows)} patches done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["patch"]))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
