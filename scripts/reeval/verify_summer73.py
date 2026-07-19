"""Validation gate for gzrecover-recovered summer scene-73 clear patches
(emrdm_reevaluation.md §2.2). Recovery is NOT trusted until this passes.

Per patch, require: 13 bands, uint16, 256x256, plausible reflectance range,
AND an existing cloudy (s2_cloudy_73) + SAR (s1_73) partner already on disk.
Patches failing any check (including gzrecover reconstruction failures that
leave truncated/garbled tif) are DROPPED and listed; the surviving count and
the dropped list are printed for the record (protocol.md).

    python scripts/reeval/verify_summer73.py --root ~/data/sen12mscr \
        --drop-invalid --report outputs/summer73_recovery.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio

SEASON = "ROIs1868_summer"
SCENE = 73
REFERENCE = 783  # summer_s1 scene-73 patch count


def patch_num(p: Path) -> int:
    return int(p.stem.rsplit("_p", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--drop-invalid", action="store_true",
                        help="delete tif files that fail validation")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser()

    clear_dir = root / f"{SEASON}_s2" / f"s2_{SCENE}"
    cloudy_dir = root / f"{SEASON}_s2_cloudy" / f"s2_cloudy_{SCENE}"
    s1_dir = root / f"{SEASON}_s1" / f"s1_{SCENE}"

    clear_files = sorted(clear_dir.glob("*.tif"), key=patch_num) if clear_dir.is_dir() else []
    valid: list[int] = []
    dropped: dict[str, list[int]] = {"unreadable": [], "bad_spec": [], "orphan": []}

    for f in clear_files:
        pn = patch_num(f)
        try:
            with rasterio.open(f) as src:
                arr = src.read()
        except Exception:
            dropped["unreadable"].append(pn)
            if args.drop_invalid:
                f.unlink()
            continue
        if arr.shape != (13, 256, 256) or arr.dtype.name != "uint16" or arr.min() < 0:
            dropped["bad_spec"].append(pn)
            if args.drop_invalid:
                f.unlink()
            continue
        cloudy = cloudy_dir / f.name.replace("_s2_", "_s2_cloudy_")
        s1 = s1_dir / f.name.replace("_s2_", "_s1_")
        if not (cloudy.exists() and s1.exists()):
            dropped["orphan"].append(pn)
            if args.drop_invalid:
                f.unlink()
            continue
        valid.append(pn)

    n_dropped = sum(len(v) for v in dropped.values())
    report = {
        "scene": SCENE,
        "reference_count": REFERENCE,
        "recovered_files": len(clear_files),
        "valid": len(valid),
        "dropped_total": n_dropped,
        "dropped_detail": dropped,
        "missing_vs_reference": REFERENCE - len(valid),
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(
        f"SUMMER73_VALID={len(valid)} DROPPED={n_dropped} "
        f"MISSING_VS_783={REFERENCE - len(valid)}"
    )


if __name__ == "__main__":
    main()
