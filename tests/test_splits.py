"""Tests for geographic ROI splitting: determinism and leakage."""

from __future__ import annotations

import pytest

from nadir.data.splits import assert_no_leakage, split_rois

ROIS = [f"ROIs1158_spring_{i}" for i in range(10)] + [f"ROIs1868_summer_{i}" for i in range(10)]


def test_partition_is_complete_and_disjoint() -> None:
    splits = split_rois(ROIS, val_fraction=0.1, test_fraction=0.1, seed=0)
    assert_no_leakage(splits)
    combined = splits["train"] + splits["val"] + splits["test"]
    assert sorted(combined) == sorted(ROIS)


def test_deterministic_given_seed() -> None:
    a = split_rois(ROIS, 0.2, 0.2, seed=42)
    b = split_rois(list(reversed(ROIS)), 0.2, 0.2, seed=42)  # scan order must not matter
    assert a == b
    c = split_rois(ROIS, 0.2, 0.2, seed=43)
    assert a != c


def test_small_dataset_gets_nonempty_eval_splits() -> None:
    splits = split_rois(["a", "b", "c"], 0.1, 0.1, seed=0)
    assert len(splits["val"]) == 1 and len(splits["test"]) == 1 and len(splits["train"]) == 1


def test_rejects_degenerate_fractions() -> None:
    with pytest.raises(ValueError):
        split_rois(ROIS, 0.6, 0.5, seed=0)
    with pytest.raises(ValueError):
        split_rois(["a", "b"], 0.5, 0.5, seed=0)  # would leave no train ROIs


def test_leakage_detection() -> None:
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_leakage({"train": ["a", "b"], "test": ["b"]})
