"""Geometric augmentation: flips and 90-degree rotations ONLY.

Color/intensity augmentations (jitter, gamma, channel shuffling, ...) are
deliberately absent: pixel values are physical quantities (TOA reflectance,
sigma0 dB) and perturbing them breaks the sensor model the network must learn.
Dihedral-group transforms are the only augmentations that leave both the
radiometry and the S1<->S2 co-registration intact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometricParams:
    """One element of the dihedral group D4: rotation then optional flip."""

    rot90: int  # number of CCW 90-degree rotations, in {0, 1, 2, 3}
    hflip: bool


def sample_geometric(rng: random.Random) -> GeometricParams:
    return GeometricParams(rot90=rng.randrange(4), hflip=rng.random() < 0.5)


def apply_geometric(array: np.ndarray, params: GeometricParams) -> np.ndarray:
    """Apply the same D4 transform to a (C, H, W) or (H, W) array."""
    axes = (-2, -1)  # spatial axes, works for both masks and multiband stacks
    out = np.rot90(array, k=params.rot90, axes=axes)
    if params.hflip:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)
