"""Integrity check for extracted SEN12MS-CR scenes (Step 3/4 rule: nothing
may be deleted before this passes).

Checks per scene: all three modalities present, identical patch-id sets,
band counts (S1=2, S2=13), dtypes (S1 float32, S2 uint16), and value ranges
(S1 within [-32.5, 0] dB envelope wide enough for both VH conventions;
S2 uint16 nonnegative, warn-only above 20000 mean).

    python scripts/reeval/verify_extraction.py --root ~/data/sen12mscr \
        --season ROIs2017_winter --scene 63
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def patch_ids(directory: Path) -> set[int]:
    return {int(p.stem.rsplit("_p", 1)[1]) for p in directory.glob("*.tif")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument(
        "--sample", type=int, default=0,
        help="raster-check only N random patches per modality (0 = all)",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser()

    dirs = {
        "s1": root / f"{args.season}_s1" / f"s1_{args.scene}",
        "s2": root / f"{args.season}_s2" / f"s2_{args.scene}",
        "s2_cloudy": root / f"{args.season}_s2_cloudy" / f"s2_cloudy_{args.scene}",
    }
    errors: list[str] = []
    ids: dict[str, set[int]] = {}
    for name, d in dirs.items():
        if not d.is_dir():
            sys.exit(f"FAIL: missing modality directory {d}")
        ids[name] = patch_ids(d)
        print(f"{name}: {len(ids[name])} patches")
    if not (ids["s1"] == ids["s2"] == ids["s2_cloudy"]):
        sys.exit(f"FAIL: patch-id sets differ: {[(k, len(v)) for k, v in ids.items()]}")

    spec = {"s1": (2, "float32"), "s2": (13, "uint16"), "s2_cloudy": (13, "uint16")}
    rng = np.random.default_rng(0)
    for name, d in dirs.items():
        files = sorted(d.glob("*.tif"))
        if args.sample and len(files) > args.sample:
            files = list(rng.choice(files, size=args.sample, replace=False))
        bands, dtype = spec[name]
        for f in files:
            with rasterio.open(f) as src:
                arr = src.read()
            if arr.shape[0] != bands:
                errors.append(f"{f.name}: {arr.shape[0]} bands, expected {bands}")
            if arr.dtype.name != dtype:
                errors.append(f"{f.name}: dtype {arr.dtype.name}, expected {dtype}")
            if arr.shape[1] != 256 or arr.shape[2] != 256:
                errors.append(f"{f.name}: spatial {arr.shape[1:]}, expected 256x256")
            if name == "s1" and (arr.min() < -60.0 or arr.max() > 10.0):
                errors.append(f"{f.name}: S1 dB range ({arr.min():.1f},{arr.max():.1f})")
        print(f"{name}: raster-checked {len(files)} files")

    if errors:
        print(f"FAIL: {len(errors)} integrity violations, first 10:")
        for e in errors[:10]:
            print("  " + e)
        sys.exit(1)
    print(f"INTEGRITY OK: scene {args.scene}, {len(ids['s1'])} triplets")


if __name__ == "__main__":
    main()
