"""Step 3b, EMRDM side: per-patch no-model ("raw") metrics for one scene.

Replicates exactly what produces the raw_* columns in EMRDM's test log
(sgm/models/diffusion.py shared_test_step): per patch,
img_metrics(target=clear, pred=cloudy) with both images in [0,1] via their
own preprocessing (process_MS, rescale_method='default'). No model, no GPU.

Runs INSIDE the emrdm venv (WSL):
    venv/bin/python emrdm_raw_metrics.py --emrdm ~/emrdm/EMRDM \
        --root ~/data/sen12mscr --season ROIs2017_winter --scene 63 \
        --out ~/logs/raw_emrdm_winter63.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _shim_s2cloudless() -> None:
    """Stub the s2cloudless module before sentinel.py imports it.

    Their loader imports S2PixelCloudDetector at module top even when
    cloud_masks=None (the released test configuration), and lightgbm needs
    the system libgomp.so.1 which requires sudo to install. The shim only
    engages when the real import fails, and any actual use of the detector
    raises loudly. (libgomp1 is installed on this machine as of 2026-07-17,
    so the shim is normally dormant.)
    """
    try:
        import s2cloudless  # noqa: F401
        return
    except (ImportError, OSError):
        pass
    import types

    module = types.ModuleType("s2cloudless")

    class S2PixelCloudDetector:  # noqa: ANN001
        def __init__(self, *a: object, **k: object) -> None:
            raise RuntimeError("s2cloudless shim: real package unavailable (libgomp missing)")

    module.S2PixelCloudDetector = S2PixelCloudDetector
    sys.modules["s2cloudless"] = module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emrdm", required=True, help="EMRDM repo root (for sgm imports)")
    parser.add_argument("--root", required=True, help="SEN12MS-CR root directory")
    parser.add_argument("--season", default="ROIs2017_winter")
    parser.add_argument("--scene", type=int, default=63)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.emrdm).expanduser()))
    _shim_s2cloudless()
    import torch
    from sgm.data.sentinel.sentinel import process_MS, read_img, read_tif
    from sgm.modules.learning.metrics import img_metrics

    root = Path(args.root).expanduser()
    clear_dir = root / f"{args.season}_s2" / f"s2_{args.scene}"
    cloudy_dir = root / f"{args.season}_s2_cloudy" / f"s2_cloudy_{args.scene}"
    clear_files = sorted(clear_dir.glob("*.tif"))
    if not clear_files:
        raise FileNotFoundError(f"no patches under {clear_dir}")

    rows = []
    for clear_path in clear_files:
        cloudy_path = cloudy_dir / clear_path.name.replace("_s2_", "_s2_cloudy_")
        patch = int(clear_path.stem.rsplit("_p", 1)[1])
        clear = process_MS(read_img(read_tif(str(clear_path))), "default")
        cloudy = process_MS(read_img(read_tif(str(cloudy_path))), "default")
        # Their raw path: tensors already in [0,1] here (interface's *2-1 and
        # scale_01 cancel); img_metrics on batch dim 1, as in shared_test_step.
        m = img_metrics(
            target=torch.tensor(clear).unsqueeze(0).float(),
            pred=torch.tensor(cloudy).unsqueeze(0).float(),
        )
        rows.append(
            {"patch": patch, "MAE": m["MAE"], "RMSE": m["RMSE"], "PSNR": m["PSNR"],
             "SAM": m["SAM"], "SSIM": m["SSIM"]}
        )
        if len(rows) % 100 == 0:
            print(f"{len(rows)} patches done", flush=True)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["patch"]))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
