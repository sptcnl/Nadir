"""SEN12MS-CR triplet dataset (S1, cloudy S2, clear S2) with ROI-aware splits.

Directory layout (real dataset and the dummy generator share it):

    root/ROIs1158_spring_s1/s1_<scene>/ROIs1158_spring_s1_<scene>_p<patch>.tif
    root/ROIs1158_spring_s2/s2_<scene>/...
    root/ROIs1158_spring_s2_cloudy/s2_cloudy_<scene>/...

An "ROI" is (season, scene), e.g. "ROIs1158_spring_7". Patches of one ROI are
overlapping crops of a single scene, so splits are assigned per ROI — see
nadir.data.splits.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from nadir.data.cloud_mask import CloudMaskModel, ThresholdCloudMask
from nadir.data.preprocess import normalize_s1, normalize_s2
from nadir.data.splits import assert_no_leakage, split_rois
from nadir.data.transforms import apply_geometric, sample_geometric

# e.g. "ROIs1158_spring_s1_7_p12.tif" -> season "ROIs1158_spring", scene 7, patch 12
_FILENAME_RE = re.compile(r"^(ROIs\d+_[a-z]+)_s1_(\d+)_p(\d+)\.tif$")


@dataclass(frozen=True)
class TripletPaths:
    s1: Path
    s2: Path
    s2_cloudy: Path
    roi: str    # "<season>_<scene>", the split unit
    patch: int


def _derive_pair(s1_path: Path, season: str, scene: str, patch: str) -> TripletPaths:
    root = s1_path.parents[2]
    s2 = root / f"{season}_s2" / f"s2_{scene}" / f"{season}_s2_{scene}_p{patch}.tif"
    s2c = (
        root
        / f"{season}_s2_cloudy"
        / f"s2_cloudy_{scene}"
        / f"{season}_s2_cloudy_{scene}_p{patch}.tif"
    )
    return TripletPaths(
        s1=s1_path, s2=s2, s2_cloudy=s2c, roi=f"{season}_{scene}", patch=int(patch)
    )


def scan_triplets(root: Path) -> list[TripletPaths]:
    """Discover complete triplets under root; incomplete ones raise."""
    triplets: list[TripletPaths] = []
    for s1_path in sorted(root.glob("*_s1/s1_*/*.tif")):
        m = _FILENAME_RE.match(s1_path.name)
        if m is None:
            raise ValueError(f"unexpected S1 filename: {s1_path}")
        triplet = _derive_pair(s1_path, m.group(1), m.group(2), m.group(3))
        for path in (triplet.s2, triplet.s2_cloudy):
            if not path.exists():
                raise FileNotFoundError(f"incomplete triplet, missing {path}")
        triplets.append(triplet)
    if not triplets:
        raise FileNotFoundError(f"no SEN12MS-CR patches found under {root}")
    return triplets


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


class Sen12MSCRDataset(Dataset):
    """Yields normalized tensors plus a 3-class cloud mask.

    Sample dict:
        s1        (2, H, W)  float32 in [-1, 1]
        s2_cloudy (13, H, W) float32 in [-1, 1]
        s2_clear  (13, H, W) float32 in [-1, 1]
        mask      (H, W)     uint8, CloudClass values (computed on the cloudy image)
        roi       str, patch int
    """

    def __init__(
        self,
        root: Path | str,
        rois: list[str] | None = None,
        mask_model: CloudMaskModel | None = None,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        all_triplets = scan_triplets(self.root)
        if rois is not None:
            wanted = set(rois)
            self.triplets = [t for t in all_triplets if t.roi in wanted]
            if not self.triplets:
                raise ValueError(f"no patches for requested ROIs: {sorted(wanted)}")
        else:
            self.triplets = all_triplets
        self.mask_model = mask_model if mask_model is not None else ThresholdCloudMask()
        self.augment = augment
        # Per-dataset RNG for augmentation; reseeded per worker via initial torch seed.
        self._rng = random.Random(torch.initial_seed() % 2**32)

    @property
    def rois(self) -> list[str]:
        return sorted({t.roi for t in self.triplets})

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        t = self.triplets[index]
        s1 = normalize_s1(_read(t.s1))
        s2_cloudy_dn = _read(t.s2_cloudy)
        s2_clear = normalize_s2(_read(t.s2))
        s2_cloudy = normalize_s2(s2_cloudy_dn)
        # Mask is computed on TOA reflectance in [0, 1], per the mask interface.
        mask = self.mask_model(s2_cloudy_dn.astype(np.float32) / 10000.0)

        if self.augment:
            params = sample_geometric(self._rng)
            s1 = apply_geometric(s1, params)
            s2_cloudy = apply_geometric(s2_cloudy, params)
            s2_clear = apply_geometric(s2_clear, params)
            mask = apply_geometric(mask, params)

        return {
            "s1": torch.from_numpy(s1),
            "s2_cloudy": torch.from_numpy(s2_cloudy),
            "s2_clear": torch.from_numpy(s2_clear),
            "mask": torch.from_numpy(mask),
            "roi": t.roi,
            "patch": t.patch,
        }


def build_datasets(cfg: DictConfig) -> dict[str, Sen12MSCRDataset]:
    """Construct train/val/test datasets with a leak-checked geographic split."""
    root = Path(cfg.data.root)
    all_rois = sorted({t.roi for t in scan_triplets(root)})
    splits = split_rois(
        all_rois,
        val_fraction=cfg.data.split.val_fraction,
        test_fraction=cfg.data.split.test_fraction,
        seed=cfg.data.split.split_seed,
    )
    assert_no_leakage(splits)
    mask_model = ThresholdCloudMask(
        thin=cfg.data.mask.thin, thick=cfg.data.mask.thick, shadow=cfg.data.mask.shadow
    )
    return {
        name: Sen12MSCRDataset(
            root,
            rois=splits[name],
            mask_model=mask_model,
            augment=(name == "train" and cfg.data.augment),
        )
        for name in ("train", "val", "test")
    }
