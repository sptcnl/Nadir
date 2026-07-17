"""Integrity check for extracted SEN12MS-CR scenes (Step 3/4 rule: nothing
may be deleted before this passes).

Checks per scene: all three modalities present, identical patch-id sets,
band counts (S1=2, S2=13), dtypes (S1 float32, S2 uint16), spatial 256x256,
and value ranges (S1 within a [-60, 10] dB envelope wide enough for both VH
conventions; S2 uint16 nonnegative by dtype).

Single scene:
    python scripts/reeval/verify_extraction.py --root ~/data/sen12mscr \
        --season ROIs2017_winter --scene 63

Full canonical test split (all 10 scenes + total-count gate vs 7,899):
    python scripts/reeval/verify_extraction.py --root ~/data/sen12mscr --split test
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio

EXPECTED_TEST_PATCHES = 7899  # per DB-CR's statement of the UnCRtainTS split

SPEC = {"s1": (2, "float32"), "s2": (13, "uint16"), "s2_cloudy": (13, "uint16")}


def _load_download_module():  # noqa: ANN202
    path = Path(__file__).resolve().parents[1] / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_ids(directory: Path) -> set[int]:
    return {int(p.stem.rsplit("_p", 1)[1]) for p in directory.glob("*.tif")}


def verify_scene(root: Path, season: str, scene: int, sample: int) -> tuple[int, list[str]]:
    """Returns (triplet_count, errors). Missing modality dir is an error."""
    dirs = {
        "s1": root / f"{season}_s1" / f"s1_{scene}",
        "s2": root / f"{season}_s2" / f"s2_{scene}",
        "s2_cloudy": root / f"{season}_s2_cloudy" / f"s2_cloudy_{scene}",
    }
    errors: list[str] = []
    ids: dict[str, set[int]] = {}
    for name, d in dirs.items():
        if not d.is_dir():
            return 0, [f"{season}/{scene}: missing modality directory {d}"]
        ids[name] = patch_ids(d)
    if not (ids["s1"] == ids["s2"] == ids["s2_cloudy"]):
        errors.append(
            f"{season}/{scene}: patch-id sets differ: {[(k, len(v)) for k, v in ids.items()]}"
        )
    rng = np.random.default_rng(0)
    for name, d in dirs.items():
        files = sorted(d.glob("*.tif"))
        if sample and len(files) > sample:
            files = list(rng.choice(files, size=sample, replace=False))
        bands, dtype = SPEC[name]
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
    return len(ids["s1"]), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--season", default=None)
    parser.add_argument("--scene", type=int, default=None)
    parser.add_argument("--split", choices=["test"], default=None)
    parser.add_argument(
        "--sample", type=int, default=0,
        help="raster-check only N random patches per modality (0 = all)",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser()

    if args.split == "test":
        dl = _load_download_module()
        targets = [
            (dl.SEASONS[season], scene)
            for season, scenes in dl.TEST_SPLIT.items()
            for scene in scenes
        ]
    elif args.season and args.scene is not None:
        targets = [(args.season, args.scene)]
    else:
        parser.error("pass --split test, or --season and --scene")

    total = 0
    all_errors: list[str] = []
    for season, scene in targets:
        count, errors = verify_scene(root, season, scene, args.sample)
        total += count
        all_errors.extend(errors)
        print(f"{season}/s?_{scene}: {count} triplets, {len(errors)} errors")

    if all_errors:
        print(f"FAIL: {len(all_errors)} integrity violations, first 10:")
        for e in all_errors[:10]:
            print("  " + e)
        sys.exit(1)
    print(f"INTEGRITY OK: {len(targets)} scene(s), {total} triplets")
    if args.split == "test":
        if total != EXPECTED_TEST_PATCHES:
            print(
                f"FAIL: total {total} != expected {EXPECTED_TEST_PATCHES} "
                "(UnCRtainTS test split per DB-CR) — split ambiguity, investigate "
                "before deleting anything"
            )
            sys.exit(1)
        print(f"TOTAL MATCHES EXPECTED {EXPECTED_TEST_PATCHES}")


if __name__ == "__main__":
    main()
