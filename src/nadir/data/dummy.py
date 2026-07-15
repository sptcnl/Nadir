"""Synthetic SEN12MS-CR-compatible dummy data.

Generates triplets (S1, S2 clear, S2 cloudy) with the exact on-disk layout,
shapes, dtypes, and value ranges of the real dataset, so the whole pipeline
(dataset class -> preprocessing -> training loop) can be exercised before the
~100GB download.

Real SEN12MS-CR layout (verified against the official UnCRtainTS dataloader,
https://github.com/PatrickTUM/UnCRtainTS/blob/main/data/dataLoader.py):

    root/
      ROIs1158_spring_s1/s1_<scene>/ROIs1158_spring_s1_<scene>_p<patch>.tif
      ROIs1158_spring_s2/s2_<scene>/ROIs1158_spring_s2_<scene>_p<patch>.tif
      ROIs1158_spring_s2_cloudy/s2_cloudy_<scene>/ROIs1158_spring_s2_cloudy_<scene>_p<patch>.tif

File specs matched here:
  - S1: 2 bands (VV, VH), float32, backscatter in dB, ~[-25, 0]
  - S2: 13 bands (L1C), uint16, TOA reflectance x 10000, nominally [0, 10000]
    (real data can exceed 10000 over bright clouds; the dummy stays in range)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from skimage.transform import resize

S1_BANDS = 2
S2_BANDS = 13
S1_DB_MIN = -25.0
S1_DB_MAX = 0.0
S2_MAX = 10000

# Season roots as they appear in the real dataset.
SEASONS = ("ROIs1158_spring", "ROIs1868_summer", "ROIs1970_fall", "ROIs2017_winter")


@dataclass
class DummyConfig:
    out_dir: Path
    num_scenes: int = 8  # distinct ROIs (scenes), round-robin over seasons
    patches_per_scene: int = 8
    size: int = 256
    seed: int = 42
    seasons: tuple[str, ...] = field(default=SEASONS)


def _smooth_field(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Low-frequency random field in [0, 1]: coarse noise upsampled bilinearly."""
    coarse = rng.random((cells, cells))
    up = resize(coarse, (size, size), order=1, mode="reflect", anti_aliasing=False)
    lo, hi = float(up.min()), float(up.max())
    return (up - lo) / (hi - lo + 1e-12)


def _make_s2_clear(rng: np.random.Generator, size: int) -> np.ndarray:
    """13-band clear scene with correlated bands.

    Bands are linear mixes of a few latent "terrain" fields so that pixels have
    consistent spectral signatures — this is what makes SAM a meaningful metric
    even on dummy data.
    """
    latents = np.stack([_smooth_field(rng, size, cells) for cells in (4, 8, 16)])
    weights = rng.dirichlet(np.ones(latents.shape[0]), size=S2_BANDS)  # (13, 3)
    base = np.tensordot(weights, latents, axes=1)  # (13, H, W)
    # Per-band gain/offset: keeps reflectance in a plausible 200..6000 range.
    gain = rng.uniform(1500.0, 5000.0, size=(S2_BANDS, 1, 1))
    offset = rng.uniform(200.0, 800.0, size=(S2_BANDS, 1, 1))
    refl = base * gain + offset
    return np.clip(refl, 0, S2_MAX).astype(np.uint16)


def make_cloud_alpha(rng: np.random.Generator, size: int) -> np.ndarray:
    """Cloud opacity in [0, 1] with both thick cores and thin fringes."""
    f = _smooth_field(rng, size, cells=6)
    thr = rng.uniform(0.45, 0.65)
    # Smoothstep above the threshold: 0 outside clouds, ->1 in thick cores.
    alpha = np.clip((f - thr) / (1.0 - thr), 0.0, 1.0)
    return (alpha**0.7).astype(np.float64)  # widen thin-cloud fringes


def _make_s2_cloudy(
    rng: np.random.Generator, clear: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Blend bright, spectrally-flat cloud over the clear scene."""
    cloud_level = rng.uniform(7000.0, 9500.0)
    cloud = cloud_level + rng.normal(0.0, 150.0, size=clear.shape[1:])
    cloudy = (1.0 - alpha) * clear.astype(np.float64) + alpha * cloud
    return np.clip(cloudy, 0, S2_MAX).astype(np.uint16)


def _make_s1(rng: np.random.Generator, size: int) -> np.ndarray:
    """VV/VH backscatter in dB. VH sits ~7 dB below VV on average."""
    terrain = _smooth_field(rng, size, cells=8)
    speckle = rng.normal(0.0, 1.5, size=(size, size))  # SAR is noisy by nature
    vv = -18.0 + 12.0 * terrain + speckle
    vh = vv - 7.0 + rng.normal(0.0, 1.0, size=(size, size))
    s1 = np.stack([vv, vh]).astype(np.float32)
    return np.clip(s1, S1_DB_MIN, S1_DB_MAX)


def _write_tif(path: Path, array: np.ndarray, transform: rasterio.Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": array.dtype.name,
        "crs": "EPSG:32632",  # arbitrary but valid UTM zone; real data varies per ROI
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def generate(cfg: DummyConfig) -> list[Path]:
    """Generate the dummy dataset; returns the list of written S1 paths."""
    rng = np.random.default_rng(cfg.seed)
    written: list[Path] = []
    for scene in range(1, cfg.num_scenes + 1):
        season = cfg.seasons[(scene - 1) % len(cfg.seasons)]
        # Distinct geolocation per scene: 10m pixels, scenes tiled apart.
        origin_x, origin_y = 500_000.0 + scene * 30_000.0, 4_000_000.0
        for patch in range(1, cfg.patches_per_scene + 1):
            transform = from_origin(
                origin_x + (patch - 1) * cfg.size * 10.0, origin_y, 10.0, 10.0
            )
            clear = _make_s2_clear(rng, cfg.size)
            alpha = make_cloud_alpha(rng, cfg.size)
            cloudy = _make_s2_cloudy(rng, clear, alpha)
            s1 = _make_s1(rng, cfg.size)

            def _path(modality: str) -> Path:
                return (
                    cfg.out_dir
                    / f"{season}_{modality}"
                    / f"{modality}_{scene}"
                    / f"{season}_{modality}_{scene}_p{patch}.tif"
                )

            _write_tif(_path("s1"), s1, transform)
            _write_tif(_path("s2"), clear, transform)
            _write_tif(_path("s2_cloudy"), cloudy, transform)
            written.append(_path("s1"))
    return written
