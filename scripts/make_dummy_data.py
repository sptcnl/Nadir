"""Generate a synthetic SEN12MS-CR-compatible dummy dataset.

Usage:
    python scripts/make_dummy_data.py                       # 64 samples -> data/sen12mscr_dummy
    python scripts/make_dummy_data.py --out data/tiny --num-scenes 2 --patches-per-scene 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nadir.data.dummy import DummyConfig, generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sen12mscr_dummy"))
    parser.add_argument("--num-scenes", type=int, default=8, help="distinct ROIs (scenes)")
    parser.add_argument("--patches-per-scene", type=int, default=8)
    parser.add_argument("--size", type=int, default=256, help="patch size in pixels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = DummyConfig(
        out_dir=args.out,
        num_scenes=args.num_scenes,
        patches_per_scene=args.patches_per_scene,
        size=args.size,
        seed=args.seed,
    )
    written = generate(cfg)
    total = len(written)
    print(f"Wrote {total} triplets ({3 * total} GeoTIFFs) to {args.out}")


if __name__ == "__main__":
    main()
