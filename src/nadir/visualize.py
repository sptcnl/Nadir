"""Image grids for qualitative monitoring in W&B.

Display-only transforms live here and must never leak into training or
metrics: the brightness gain below exists because land reflectance rarely
exceeds ~0.4, so ungained RGB renders nearly black.
"""

from __future__ import annotations

import numpy as np
import torch

from nadir.metrics import RGB_BAND_INDICES

DISPLAY_GAIN = 2.5


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def rgb_display(reflectance: torch.Tensor) -> np.ndarray:
    """(13, H, W) reflectance in [0, 1] -> (H, W, 3) uint8, gained for display."""
    rgb = reflectance[list(RGB_BAND_INDICES)].detach().cpu().numpy().transpose(1, 2, 0)
    return _to_uint8(rgb * DISPLAY_GAIN)


def sar_display(s1_normalized: torch.Tensor) -> np.ndarray:
    """(2, H, W) normalized S1 in [-1, 1] -> (H, W, 3) uint8 grayscale of VV."""
    vv = (s1_normalized[0].detach().cpu().numpy() + 1.0) / 2.0
    return _to_uint8(np.stack([vv, vv, vv], axis=-1))


def prediction_grid(
    s1: torch.Tensor,
    s2_cloudy: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    """Rows = samples; columns = SAR (VV) | cloudy | prediction | ground truth.

    Args:
        s1: (N, 2, H, W) normalized SAR in [-1, 1].
        s2_cloudy/pred/target: (N, 13, H, W) reflectance in [0, 1].

    Returns:
        (N*H, 4*W, 3) uint8 image.
    """
    rows = []
    for i in range(s1.shape[0]):
        row = np.concatenate(
            [
                sar_display(s1[i]),
                rgb_display(s2_cloudy[i]),
                rgb_display(pred[i]),
                rgb_display(target[i]),
            ],
            axis=1,
        )
        rows.append(row)
    return np.concatenate(rows, axis=0)
