"""Geographic (ROI-level) train/val/test splitting.

SEN12MS-CR patches within one ROI are spatially overlapping crops of the same
scene. Splitting at the patch level would place near-duplicate pixels in both
train and test, inflating every metric. Therefore the split unit is the ROI
(season + scene id); a patch inherits its ROI's split, unconditionally.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def split_rois(
    rois: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    """Deterministically partition ROI ids into train/val/test.

    Fractions are rounded up to at least one ROI (when the fraction is > 0)
    so tiny dummy datasets still produce non-empty eval splits.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("fractions must be >= 0 and sum to < 1")
    unique = sorted(set(rois))  # sort first: split must not depend on scan order
    if not unique:
        raise ValueError("no ROIs to split")
    rng = random.Random(seed)
    rng.shuffle(unique)

    n = len(unique)
    n_val = max(1, round(n * val_fraction)) if val_fraction > 0 else 0
    n_test = max(1, round(n * test_fraction)) if test_fraction > 0 else 0
    if n_val + n_test >= n:
        raise ValueError(
            f"split leaves no training ROIs: {n} total, {n_val} val, {n_test} test"
        )

    splits = {
        "val": sorted(unique[:n_val]),
        "test": sorted(unique[n_val : n_val + n_test]),
        "train": sorted(unique[n_val + n_test :]),
    }
    return splits


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    """Raise if any ROI appears in more than one split."""
    seen: dict[str, str] = {}
    for name, rois in splits.items():
        for roi in rois:
            if roi in seen:
                raise AssertionError(f"ROI {roi} leaked: in both {seen[roi]} and {name}")
            seen[roi] = name
