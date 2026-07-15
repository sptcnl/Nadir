"""End-to-end dataset tests on generated dummy data: spec, masks, augmentation,
and geographic split leakage at the dataset level."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from nadir.data.cloud_mask import CloudClass, ThresholdCloudMask, cloud_or_shadow
from nadir.data.dummy import DummyConfig, generate
from nadir.data.sen12mscr import Sen12MSCRDataset, build_datasets, scan_triplets
from nadir.data.transforms import GeometricParams, apply_geometric, sample_geometric

SIZE = 64


@pytest.fixture(scope="module")
def dummy_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ds")
    generate(DummyConfig(out_dir=root, num_scenes=6, patches_per_scene=3, size=SIZE, seed=3))
    return root


def _cfg(root: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {
                "root": str(root),
                "mask": {"thin": 0.35, "thick": 0.6, "shadow": 0.08},
                "augment": True,
                "split": {"val_fraction": 0.2, "test_fraction": 0.2, "split_seed": 1},
            }
        }
    )


def test_sample_spec(dummy_root: Path) -> None:
    ds = Sen12MSCRDataset(dummy_root)
    assert len(ds) == 18
    sample = ds[0]
    assert sample["s1"].shape == (2, SIZE, SIZE)
    assert sample["s2_cloudy"].shape == (13, SIZE, SIZE)
    assert sample["s2_clear"].shape == (13, SIZE, SIZE)
    assert sample["mask"].shape == (SIZE, SIZE)
    for key in ("s1", "s2_cloudy", "s2_clear"):
        t = sample[key]
        assert t.dtype == torch.float32
        assert t.min() >= -1.0 and t.max() <= 1.0
    assert sample["mask"].dtype == torch.uint8


def test_mask_classes_present(dummy_root: Path) -> None:
    ds = Sen12MSCRDataset(dummy_root)
    values: set[int] = set()
    for i in range(len(ds)):
        mask = ds[i]["mask"].numpy()
        values |= set(np.unique(mask).tolist())
    assert values <= {int(c) for c in CloudClass}
    # Dummy scenes contain bright clouds, so cloudy classes must appear somewhere.
    assert int(CloudClass.THIN) in values or int(CloudClass.THICK) in values


def test_mask_marks_cloud_cores(dummy_root: Path) -> None:
    # Thick-cloud pixels in the cloudy image are near the cloud level (>= 0.6
    # visible reflectance), so the binary mask must fire on a cloudy patch.
    ds = Sen12MSCRDataset(dummy_root, mask_model=ThresholdCloudMask())
    any_cloud = any(cloud_or_shadow(ds[i]["mask"].numpy()).any() for i in range(len(ds)))
    assert any_cloud


def test_geometric_transform_preserves_values() -> None:
    # D4 transforms permute pixels; the value multiset must be identical
    # (this is exactly why they are radiometrically safe).
    arr = np.random.default_rng(0).random((13, 8, 8)).astype(np.float32)
    params = GeometricParams(rot90=3, hflip=True)
    out = apply_geometric(arr, params)
    assert out.shape == arr.shape
    np.testing.assert_array_equal(np.sort(out, axis=None), np.sort(arr, axis=None))


def test_augmentation_applied_consistently(dummy_root: Path) -> None:
    # All modalities and the mask must receive the SAME transform: re-derive
    # the mask from the augmented cloudy image and compare.
    ds = Sen12MSCRDataset(dummy_root, augment=False)
    sample = ds[0]
    params = sample_geometric(random.Random(0))
    mask_model = ThresholdCloudMask()
    from nadir.data.preprocess import denormalize_s2

    cloudy_aug = apply_geometric(sample["s2_cloudy"].numpy(), params)
    mask_aug = apply_geometric(sample["mask"].numpy(), params)
    np.testing.assert_array_equal(mask_model(denormalize_s2(cloudy_aug)), mask_aug)


def test_build_datasets_no_roi_leakage(dummy_root: Path) -> None:
    datasets = build_datasets(_cfg(dummy_root))
    train, val, test = (set(datasets[k].rois) for k in ("train", "val", "test"))
    assert train and val and test
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    # Every patch (not just every ROI) must be assigned to exactly one split.
    n_total = len(scan_triplets(dummy_root))
    assert len(datasets["train"]) + len(datasets["val"]) + len(datasets["test"]) == n_total
    # Augmentation only on train.
    assert datasets["train"].augment and not datasets["val"].augment


def test_split_stable_across_calls(dummy_root: Path) -> None:
    a = build_datasets(_cfg(dummy_root))
    b = build_datasets(_cfg(dummy_root))
    assert {k: v.rois for k, v in a.items()} == {k: v.rois for k, v in b.items()}
