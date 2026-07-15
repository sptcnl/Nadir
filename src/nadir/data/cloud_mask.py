"""Cloud/shadow mask models.

Masks are (H, W) uint8 with the following classes:

    0 = CLEAR, 1 = THIN cloud, 2 = THICK cloud, 3 = SHADOW

The 3-class distinction matters downstream: thin clouds retain partial ground
signal (reconstruction can lean on the optical input), thick clouds must be
reconstructed almost purely from SAR, and shadows are radiometrically dark but
geometrically tied to clouds.

TODO(Step 5/7): add an S2CloudlessMask implementation backed by the
`s2cloudless` package (LightGBM on 10 of the 13 L1C bands, input TOA
reflectance in [0, 1], returns cloud probability). Thin/thick can be derived
from two probability thresholds; shadows need a separate projection-based
step (e.g. cloud-shadow displacement from solar geometry) or a dark-pixel
heuristic. The dependency chain (lightgbm) is heavy, hence the interface +
threshold fallback below for Phase 1.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Protocol

import numpy as np


class CloudClass(IntEnum):
    CLEAR = 0
    THIN = 1
    THICK = 2
    SHADOW = 3


class CloudMaskModel(Protocol):
    """Maps a 13-band TOA reflectance patch to a 3-class cloud mask."""

    def __call__(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Args:
            reflectance: (13, H, W) float array, TOA reflectance in [0, 1].

        Returns:
            (H, W) uint8 mask with CloudClass values.
        """
        ...


class ThresholdCloudMask:
    """Brightness-threshold placeholder for s2cloudless.

    Heuristic: clouds are bright and spectrally flat in the visible bands,
    shadows are very dark. Thresholds operate on mean visible (B02/B03/B04)
    reflectance. This is NOT scientifically adequate for real evaluation —
    it exists so the pipeline (CARL loss, mask-split metrics) is exercisable
    end-to-end before s2cloudless is wired in. See module TODO.
    """

    # L1C band order: B01, B02, B03, B04, ... -> visible RGB = indices 1..3.
    VISIBLE_BANDS = (1, 2, 3)

    def __init__(self, thin: float = 0.35, thick: float = 0.6, shadow: float = 0.08) -> None:
        if not (0.0 <= shadow < thin < thick):
            raise ValueError("thresholds must satisfy 0 <= shadow < thin < thick")
        self.thin = thin
        self.thick = thick
        self.shadow = shadow

    def __call__(self, reflectance: np.ndarray) -> np.ndarray:
        visible = reflectance[list(self.VISIBLE_BANDS)].mean(axis=0)
        mask = np.full(visible.shape, CloudClass.CLEAR, dtype=np.uint8)
        mask[visible >= self.thin] = CloudClass.THIN
        mask[visible >= self.thick] = CloudClass.THICK
        mask[visible <= self.shadow] = CloudClass.SHADOW
        return mask


def cloud_or_shadow(mask: np.ndarray) -> np.ndarray:
    """Binary (H, W) bool mask: any non-clear class. Used by CARL and metrics."""
    return mask != CloudClass.CLEAR
