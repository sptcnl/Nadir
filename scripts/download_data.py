"""Download SEN12MS-CR from the official TUM distribution.

Official source (no authentication required):
    https://mediatum.ub.tum.de/1554803  (dataset page; Ebel et al., IEEE TGRS
    2021, "Multisensor Data Fusion for Cloud Removal in Global and All-Season
    Sentinel-2 Imagery")
    https://dataserv.ub.tum.de/s/m1554803  (file server)

URL scheme (verified against the official UnCRtainTS util/dl_data.sh):
    https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=<archive>
with one archive per season and modality, e.g. ROIs1158_spring_s1.tar.gz.

The FULL dataset is >100 GB — full download is deliberately NOT the default.
Granularity:
  - download: per season (--seasons), each season = 3 modality archives;
  - extraction: per ROI/scene (--scenes) — only matching scene folders are
    unpacked, so a small geographic subset can be materialized from one
    season's archives.

Examples:
    python scripts/download_data.py --list
    python scripts/download_data.py --seasons spring --scenes 1 8 17
    python scripts/download_data.py --seasons spring summer
    python scripts/download_data.py --all          # full dataset, explicit
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files="

SEASONS: dict[str, str] = {
    "spring": "ROIs1158_spring",
    "summer": "ROIs1868_summer",
    "fall": "ROIs1970_fall",
    "winter": "ROIs2017_winter",
}
MODALITIES = ("s1", "s2", "s2_cloudy")

_CHUNK = 1 << 20  # 1 MiB


def archive_names(season: str) -> list[str]:
    """The 3 modality archives of a season; all are needed to form triplets."""
    prefix = SEASONS[season]
    return [f"{prefix}_{modality}.tar.gz" for modality in MODALITIES]


def download(archive: str, dest_dir: Path) -> Path:
    """Stream one archive with HTTP-Range resume support."""
    dest = dest_dir / archive
    dest_dir.mkdir(parents=True, exist_ok=True)
    offset = dest.stat().st_size if dest.exists() else 0
    request = urllib.request.Request(BASE_URL + archive)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as err:
        if err.code == 416:  # requested range not satisfiable: already complete
            print(f"{archive}: already complete")
            return dest
        raise
    total = response.headers.get("Content-Length")
    total_mb = f"{int(total) / 1e6:,.0f} MB" if total else "unknown size"
    mode = "ab" if offset and response.status == 206 else "wb"
    done = offset if mode == "ab" else 0
    print(f"{archive}: downloading ({total_mb}, resuming at {offset / 1e6:,.0f} MB)")
    with open(dest, mode) as fh:
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if done % (100 * _CHUNK) < _CHUNK:
                print(f"  {archive}: {done / 1e6:,.0f} MB", flush=True)
    return dest


def member_scene(name: str) -> int | None:
    """Scene id encoded in an archive member path, or None.

    Matches the '<modality>_<scene>' directory segment, e.g.
    'ROIs1158_spring_s1/s1_73/ROIs1158_spring_s1_73_p12.tif' -> 73.
    """
    m = re.search(r"(?:^|/)(?:s1|s2|s2_cloudy)_(\d+)(?:/|$)", name)
    return int(m.group(1)) if m else None


def extract(archive_path: Path, out_dir: Path, scenes: set[int] | None) -> int:
    """Unpack an archive; when `scenes` is given, only those ROIs are extracted."""
    extracted = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        members = []
        for member in tar:
            scene = member_scene(member.name)
            if scenes is not None and scene is not None and scene not in scenes:
                continue
            members.append(member)
        tar.extractall(out_dir, members=members, filter="data")
        extracted = sum(1 for m in members if m.isfile())
    print(f"{archive_path.name}: extracted {extracted} files -> {out_dir}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/sen12mscr"))
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="where .tar.gz land (default: <out>/_archives)",
    )
    parser.add_argument("--seasons", nargs="+", choices=sorted(SEASONS), default=[])
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=int,
        default=None,
        help="ROI/scene ids to extract (subset); omit to extract everything downloaded",
    )
    parser.add_argument("--all", action="store_true", help="download ALL seasons (>100 GB)")
    parser.add_argument("--list", action="store_true", help="list available archives and exit")
    parser.add_argument(
        "--skip-extract", action="store_true", help="download archives only, do not unpack"
    )
    args = parser.parse_args()

    if args.list:
        for season in SEASONS:
            for archive in archive_names(season):
                print(BASE_URL + archive)
        return

    seasons = sorted(SEASONS) if args.all else args.seasons
    if not seasons:
        parser.error("pass --seasons (subset) or --all (full >100 GB download), or --list")

    archive_dir = args.archive_dir or (args.out / "_archives")
    scenes = set(args.scenes) if args.scenes else None
    for season in seasons:
        for archive in archive_names(season):
            path = download(archive, archive_dir)
            if not args.skip_extract:
                extract(path, args.out, scenes)

    # The dataset class scans <out>/*_s1/s1_*/*.tif; fail loudly here rather
    # than late in training if the archive layout ever changes upstream.
    if not args.skip_extract and not list(args.out.glob("*_s1/s1_*/*.tif")):
        sys.exit(
            "extraction finished but no '*_s1/s1_*/*.tif' patches found under "
            f"{args.out} — archive layout may have changed; inspect {archive_dir}"
        )


if __name__ == "__main__":
    main()
