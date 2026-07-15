"""Radiometric preprocessing for Sentinel-1/-2 patches.

All normalizations map to [-1, 1]:
  - bounded inputs are required by the Phase-2 diffusion model, and using the
    same range for S1/S2 keeps the CNN baseline's input scaling uniform;
  - bounded targets make pixel metrics (PSNR/MAE) comparable across runs.
"""

from __future__ import annotations

import numpy as np

S1_DB_MIN = -25.0
S1_DB_MAX = 0.0
S2_REFLECTANCE_MAX = 10000.0


def normalize_s2(dn: np.ndarray) -> np.ndarray:
    """Sentinel-2 L1C digital numbers -> [-1, 1] float32.

    Clipping strategy (identical bounds for ALL 13 bands, on purpose):
      - L1C DNs encode TOA reflectance x 10000. Physical land reflectance is
        mostly <= 1.0, but DNs exceed 10000 over bright clouds, snow, and
        specular water (values up to ~28000 occur in SEN12MS-CR).
      - We clip to [0, 10000] uniformly rather than per-band percentile
        stretching: any band-dependent rescaling would change inter-band
        ratios and corrupt spectral signatures, invalidating SAM and every
        downstream index (NDVI, NDWI, ...). Uniform clipping only saturates
        cloud/snow pixels — precisely the pixels this model replaces.
      - Reference: DSen2-CR (Meraner et al., ISPRS 2020) also clips S2 to
        [0, 10000] before scaling; they divide by 2000 while we map to
        [-1, 1] for the reasons in the module docstring.
    """
    refl = np.clip(dn.astype(np.float32), 0.0, S2_REFLECTANCE_MAX)
    return (refl / S2_REFLECTANCE_MAX) * 2.0 - 1.0


def denormalize_s2(x: np.ndarray) -> np.ndarray:
    """[-1, 1] -> TOA reflectance in [0, 1] (float32), for metrics/plots."""
    return ((x.astype(np.float32) + 1.0) / 2.0).clip(0.0, 1.0)


def normalize_s1(db: np.ndarray) -> np.ndarray:
    """Sentinel-1 sigma0 in dB -> [-1, 1] float32.

    [-25, 0] dB covers the typical land backscatter range (same bounds as the
    SEN12MS-CR reference loaders and DSen2-CR preprocessing); values outside
    are sensor noise floor / urban corner reflectors and are safely saturated.
    """
    x = np.clip(db.astype(np.float32), S1_DB_MIN, S1_DB_MAX)
    return (x - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN) * 2.0 - 1.0
