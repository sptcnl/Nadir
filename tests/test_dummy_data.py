"""Spec-compliance tests for the dummy SEN12MS-CR generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from nadir.data.dummy import (
    S1_BANDS,
    S1_DB_MAX,
    S1_DB_MIN,
    S2_BANDS,
    S2_MAX,
    DummyConfig,
    generate,
)

SIZE = 64  # small patches keep the test fast; spec checks are size-independent


@pytest.fixture(scope="module")
def dummy_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dummy")
    generate(DummyConfig(out_dir=root, num_scenes=3, patches_per_scene=2, size=SIZE, seed=7))
    return root


def _read(path: Path) -> tuple[np.ndarray, str]:
    with rasterio.open(path) as src:
        return src.read(), src.dtypes[0]


def _triplet_paths(s1_path: Path) -> tuple[Path, Path]:
    """Derive S2/S2_cloudy paths by substitution, as the real dataloader does."""
    s2 = Path(str(s1_path).replace("_s1", "_s2").replace("s1_", "s2_"))
    s2c = Path(str(s1_path).replace("_s1", "_s2_cloudy").replace("s1_", "s2_cloudy_"))
    return s2, s2c


def test_layout_and_pairing(dummy_root: Path) -> None:
    s1_files = sorted(dummy_root.glob("*_s1/s1_*/*.tif"))
    assert len(s1_files) == 3 * 2
    for s1_path in s1_files:
        s2, s2c = _triplet_paths(s1_path)
        assert s2.exists(), f"missing clear pair for {s1_path}"
        assert s2c.exists(), f"missing cloudy pair for {s1_path}"


def test_s1_spec(dummy_root: Path) -> None:
    for path in dummy_root.glob("*_s1/s1_*/*.tif"):
        arr, dtype = _read(path)
        assert arr.shape == (S1_BANDS, SIZE, SIZE)
        assert dtype == "float32"
        assert arr.min() >= S1_DB_MIN and arr.max() <= S1_DB_MAX


def test_s2_spec(dummy_root: Path) -> None:
    paths = list(dummy_root.glob("*_s2/s2_*/*.tif"))
    paths += list(dummy_root.glob("*_s2_cloudy/s2_cloudy_*/*.tif"))
    assert paths
    for path in paths:
        arr, dtype = _read(path)
        assert arr.shape == (S2_BANDS, SIZE, SIZE)
        assert dtype == "uint16"
        assert arr.min() >= 0 and arr.max() <= S2_MAX


def test_cloudy_differs_and_is_brighter(dummy_root: Path) -> None:
    for s1_path in dummy_root.glob("*_s1/s1_*/*.tif"):
        s2, s2c = _triplet_paths(s1_path)
        clear, _ = _read(s2)
        cloudy, _ = _read(s2c)
        assert not np.array_equal(clear, cloudy)
        # Clouds are bright: mean reflectance must increase.
        assert cloudy.mean() > clear.mean()


def test_generation_is_seeded(tmp_path: Path) -> None:
    cfg_a = DummyConfig(out_dir=tmp_path / "a", num_scenes=1, patches_per_scene=1, size=SIZE)
    cfg_b = DummyConfig(out_dir=tmp_path / "b", num_scenes=1, patches_per_scene=1, size=SIZE)
    (path_a,) = generate(cfg_a)
    (path_b,) = generate(cfg_b)
    arr_a, _ = _read(path_a)
    arr_b, _ = _read(path_b)
    assert np.array_equal(arr_a, arr_b)
