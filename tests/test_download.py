"""Offline tests for the download script's subset-extraction logic."""

from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dl = _load_script()


def test_archive_names_cover_all_modalities() -> None:
    names = dl.archive_names("spring")
    assert names == [
        "ROIs1158_spring_s1.tar.gz",
        "ROIs1158_spring_s2.tar.gz",
        "ROIs1158_spring_s2_cloudy.tar.gz",
    ]


@pytest.mark.parametrize(
    ("member", "scene"),
    [
        ("ROIs1158_spring_s1/s1_73/ROIs1158_spring_s1_73_p12.tif", 73),
        ("ROIs1868_summer_s2_cloudy/s2_cloudy_5/x_p1.tif", 5),
        ("s2_142/file.tif", 142),
        ("ROIs1158_spring_s1", None),  # top-level dir has no scene segment
        ("README.txt", None),
    ],
)
def test_member_scene(member: str, scene: int | None) -> None:
    assert dl.member_scene(member) == scene


def _make_archive(path: Path) -> None:
    src = path.parent / "src"
    for scene in (1, 2, 3):
        d = src / "ROIs1158_spring_s1" / f"s1_{scene}"
        d.mkdir(parents=True)
        (d / f"ROIs1158_spring_s1_{scene}_p1.tif").write_bytes(b"x")
    with tarfile.open(path, "w:gz") as tar:
        tar.add(src / "ROIs1158_spring_s1", arcname="ROIs1158_spring_s1")


def test_extract_scene_subset(tmp_path: Path) -> None:
    archive = tmp_path / "ROIs1158_spring_s1.tar.gz"
    _make_archive(archive)
    out = tmp_path / "out"
    n = dl.extract(archive, out, scenes={1, 3})
    assert n == 2
    assert (out / "ROIs1158_spring_s1" / "s1_1").exists()
    assert (out / "ROIs1158_spring_s1" / "s1_3").exists()
    assert not (out / "ROIs1158_spring_s1" / "s1_2").exists()


def test_extract_all_when_no_subset(tmp_path: Path) -> None:
    archive = tmp_path / "ROIs1158_spring_s1.tar.gz"
    _make_archive(archive)
    out = tmp_path / "out"
    assert dl.extract(archive, out, scenes=None) == 3
