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
    python scripts/download_data.py --split test              # canonical test split, streamed
    python scripts/download_data.py --seasons spring --scenes 1 8 17
    python scripts/download_data.py --seasons spring summer --stream
    python scripts/download_data.py --all          # full dataset, explicit
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

BASE_URL = "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files="

SEASONS: dict[str, str] = {
    "spring": "ROIs1158_spring",
    "summer": "ROIs1868_summer",
    "fall": "ROIs1970_fall",
    "winter": "ROIs2017_winter",
}
MODALITIES = ("s1", "s2", "s2_cloudy")

# Canonical UnCRtainTS/EMRDM test split (geographically held out), extracted
# from the released EMRDM loader (sgm/data/sentinel/sentinel.py, identical to
# the UnCRtainTS split): 10 scenes across all four seasons. DB-CR reports
# 7,899 test patches for this split.
TEST_SPLIT: dict[str, tuple[int, ...]] = {
    "spring": (31, 44, 106, 123, 140),
    "summer": (73, 119),
    "fall": (139,),
    "winter": (63, 108),
}

_CHUNK = 1 << 20  # 1 MiB


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_manifest(path: Path) -> dict:
    """Resume manifest: records per-archive extraction status.

    Streamed gzip archives cannot resume mid-stream, so the resume unit is
    one archive: 'done' archives are skipped on rerun, anything else
    (missing / 'in_progress' after a crash) is re-streamed from zero.
    Worst-case retransfer after an interruption = one archive (<= 49 GB).
    """
    if path.exists():
        return json.loads(path.read_text())
    return {"created": _now(), "archives": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)


def archive_done(manifest: dict, archive: str) -> bool:
    return manifest["archives"].get(archive, {}).get("status") == "done"


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


def stream_extract(archive: str, out_dir: Path, scenes: set[int] | None) -> int:
    """Stream an archive over HTTPS and extract matching members WITHOUT
    storing the .tar.gz on disk.

    Rationale: the server only offers whole-season archives (8-49 GB each),
    so a scene subset still costs the full archive in *transfer* — but with
    streaming it costs only the extracted patches in *disk*. No resume:
    a failed stream restarts from zero for that archive.
    """
    response = urllib.request.urlopen(urllib.request.Request(BASE_URL + archive))
    # Large read buffer: tarfile stream mode otherwise issues small TLS reads.
    buffered = io.BufferedReader(response, buffer_size=4 * _CHUNK)
    extracted = 0
    # r|gz = non-seekable stream mode; members must be extracted while iterating.
    with tarfile.open(fileobj=buffered, mode="r|gz", bufsize=_CHUNK) as tar:
        for member in tar:
            scene = member_scene(member.name)
            if scenes is not None and scene is not None and scene not in scenes:
                continue
            tar.extract(member, out_dir, filter="data")
            if member.isfile():
                extracted += 1
                if extracted % 500 == 0:
                    print(f"  {archive}: {extracted} files extracted", flush=True)
    print(f"{archive}: streamed, extracted {extracted} files -> {out_dir}")
    return extracted


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
    parser.add_argument(
        "--split",
        choices=["test"],
        default=None,
        help="download a canonical split: 'test' = the UnCRtainTS/EMRDM held-out"
        " scenes (all seasons, per-season scene filter). Implies --stream unless"
        " archives are kept explicitly.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream archives and extract on the fly without storing .tar.gz"
        " (transfer cost unchanged; disk cost = extracted patches only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent archive downloads. The TUM server caps per-connection"
        " throughput (~2-4 MB/s measured from this host), so 3 workers"
        " roughly triples aggregate speed on a faster line",
    )
    args = parser.parse_args()

    if args.list:
        for season in SEASONS:
            for archive in archive_names(season):
                print(BASE_URL + archive)
        return

    # Per-season scene filters. --split test uses the canonical per-season map
    # (a flat scene-id set would bleed across seasons: e.g. scene 73 is a test
    # scene in summer but a training scene id may collide in another season).
    plan: dict[str, set[int] | None]
    if args.split == "test":
        plan = {season: set(scenes) for season, scenes in TEST_SPLIT.items()}
    else:
        seasons = sorted(SEASONS) if args.all else args.seasons
        if not seasons:
            parser.error(
                "pass --seasons (subset), --split test, --all (full download), or --list"
            )
        scenes = set(args.scenes) if args.scenes else None
        plan = {season: scenes for season in seasons}

    archive_dir = args.archive_dir or (args.out / "_archives")
    manifest_path = args.out / "_manifest.json"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)

    jobs: list[tuple[str, set[int] | None]] = []
    for season, season_scenes in plan.items():
        for archive in archive_names(season):
            if archive_done(manifest, archive):
                done = manifest["archives"][archive]
                print(f"{archive}: already done ({done.get('files')} files), skipping")
            else:
                jobs.append((archive, season_scenes))

    # Archives extract into disjoint top-level dirs (one season+modality each),
    # so concurrent workers never write the same path. Manifest writes are
    # serialized behind a lock.
    lock = threading.Lock()

    def run_job(job: tuple[str, set[int] | None]) -> str | None:
        """Process one archive; returns None on success, the archive name on
        failure. Transient network drops (the server cuts long streams) are
        retried in place; a stream restarted mid-archive re-extracts from
        zero, which only costs transfer time (extraction is idempotent)."""
        archive, scenes = job
        with lock:
            manifest["archives"][archive] = {
                "status": "in_progress",
                "started": _now(),
                "scenes": sorted(scenes) if scenes else None,
            }
            save_manifest(manifest_path, manifest)
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                if args.stream or args.split == "test":
                    n = stream_extract(archive, args.out, scenes)
                else:
                    path = download(archive, archive_dir)
                    n = extract(path, args.out, scenes) if not args.skip_extract else -1
                break
            except Exception as err:  # noqa: BLE001 — network/tar errors alike
                last_error = err
                print(f"{archive}: attempt {attempt} failed ({err!r}); retrying in 30s",
                      flush=True)
                time.sleep(30)
        else:
            print(f"{archive}: FAILED after 5 attempts: {last_error!r}", flush=True)
            return archive
        with lock:
            manifest["archives"][archive].update(
                {"status": "done", "files": n, "finished": _now()}
            )
            save_manifest(manifest_path, manifest)
        return None

    if args.workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            failures = [f for f in pool.map(run_job, jobs) if f]
    else:
        failures = [f for f in map(run_job, jobs) if f]
    if failures:
        sys.exit(f"{len(failures)} archive(s) failed permanently: {failures}")

    # The dataset class scans <out>/*_s1/s1_*/*.tif; fail loudly here rather
    # than late in training if the archive layout ever changes upstream.
    if not args.skip_extract and not list(args.out.glob("*_s1/s1_*/*.tif")):
        sys.exit(
            "extraction finished but no '*_s1/s1_*/*.tif' patches found under "
            f"{args.out} — archive layout may have changed; inspect {archive_dir}"
        )


if __name__ == "__main__":
    main()
