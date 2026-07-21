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
import warnings
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


def scan_triplets(root: Path, strict: bool = True) -> list[TripletPaths]:
    """Discover complete triplets under root.

    strict=True (default): an incomplete triplet (a missing S2/S2_cloudy pair)
    raises — the integrity check for data we expect to be whole. strict=False:
    incomplete triplets are skipped with a one-line warning and a count — used
    for real SEN12MS-CR data with a KNOWN gap (summer-73's clear targets are
    unrecoverable from the corrupt mirror; see emrdm_reevaluation.md §2.1), so
    the dataset yields only the complete triplets, matching EMRDM's loader.
    """
    triplets: list[TripletPaths] = []
    skipped = 0
    for s1_path in sorted(root.glob("*_s1/s1_*/*.tif")):
        m = _FILENAME_RE.match(s1_path.name)
        if m is None:
            raise ValueError(f"unexpected S1 filename: {s1_path}")
        triplet = _derive_pair(s1_path, m.group(1), m.group(2), m.group(3))
        missing = [p for p in (triplet.s2, triplet.s2_cloudy) if not p.exists()]
        if missing:
            if strict:
                raise FileNotFoundError(f"incomplete triplet, missing {missing[0]}")
            skipped += 1
            continue
        triplets.append(triplet)
    if skipped:
        warnings.warn(
            f"scan_triplets: skipped {skipped} incomplete triplet(s) under {root} "
            "(strict=False)",
            stacklevel=2,
        )
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
        strict: bool = True,
    ) -> None:
        self.root = Path(root)
        all_triplets = scan_triplets(self.root, strict=strict)
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


def _build_patch_split(cfg: DictConfig) -> dict:
    """Patch-level train/val split within `train_root`, test from `test_root`.

    Used when the training data is a single scene (spring-6 baseline): an
    ROI-level split cannot divide one scene, so train/val are split at the
    PATCH level within it. NOTE: patches of one scene overlap spatially, so
    this train/val split leaks — val metrics are optimistic and used ONLY to
    monitor convergence, never as the headline result. The headline evaluation
    is the geographically-disjoint `test_root` (per-season), run separately.
    """
    from torch.utils.data import Subset

    mask_model = ThresholdCloudMask(
        thin=cfg.data.mask.thin, thick=cfg.data.mask.thick, shadow=cfg.data.mask.shadow
    )
    train_root, test_root = Path(cfg.data.train_root), Path(cfg.data.test_root)
    full_aug = Sen12MSCRDataset(
        train_root, mask_model=mask_model, augment=cfg.data.augment
    )
    full_noaug = Sen12MSCRDataset(train_root, mask_model=mask_model, augment=False)
    n = len(full_aug)
    idx = list(range(n))
    random.Random(cfg.data.split.split_seed).shuffle(idx)
    n_val = max(1, round(n * cfg.data.split.val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    # strict=False: the 9-scene test root has a KNOWN gap (summer-73's clear
    # targets are unrecoverable, §2.1) — skip its incomplete triplets, keep the
    # 7,116 complete ones (matches EMRDM's loader).
    test_ds = Sen12MSCRDataset(
        test_root, mask_model=mask_model, augment=False, strict=False
    )
    return {
        "train": Subset(full_aug, train_idx),
        "val": Subset(full_noaug, val_idx),
        "test": test_ds,
    }


def build_datasets(cfg: DictConfig) -> dict:
    """Construct train/val/test datasets.

    strategy 'roi' (default): leak-checked geographic (per-ROI) split of one
    root. strategy 'patch': patch-level train/val within `train_root` + a
    separate `test_root` (single-scene baseline; see `_build_patch_split`).
    """
    if getattr(cfg.data.split, "strategy", "roi") == "patch":
        return _build_patch_split(cfg)
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
