"""Cloud-Adaptive Regularized Loss (CARL).

Definition from the original DSen2-CR code
(https://github.com/ameraner/dsen2-cr, Code/tools/image_metrics.py):

    carl = mean( clear_mask * |pred - input_cloudy|
               + cloud_mask * |pred - target| )
         + 1.0 * mean( |pred - target| )

Semantics:
  - inside the cloud/shadow mask the prediction must match the clear target
    (that is the actual reconstruction task);
  - outside the mask it must match the *cloudy input* — those pixels are
    already valid observations and the network must not repaint them;
  - the global L1 term regularizes toward the target everywhere, which
    stabilizes training when the mask is noisy (thin clouds, missed shadows).
"""

from __future__ import annotations

import torch
from torch import nn


class CARLLoss(nn.Module):
    """Cloud-Adaptive Regularized L1 loss.

    Args:
        lambda_reg: weight of the global L1 regularization term. The original
            implementation hardcodes 1.0.
    """

    def __init__(self, lambda_reg: float = 1.0) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        pred: torch.Tensor,
        s2_cloudy: torch.Tensor,
        target: torch.Tensor,
        cloud_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) predicted clear image.
            s2_cloudy: (B, C, H, W) cloudy input, same normalization as pred.
            target: (B, C, H, W) clear ground truth.
            cloud_mask: (B, H, W) binary mask, nonzero = cloud or shadow
                (use nadir.data.cloud_mask.cloud_or_shadow to binarize the
                3-class mask).

        Returns:
            Scalar loss.
        """
        mask = (cloud_mask != 0).to(pred.dtype).unsqueeze(1)  # (B, 1, H, W)
        clear = 1.0 - mask
        adaptive = (clear * (pred - s2_cloudy).abs() + mask * (pred - target).abs()).mean()
        regularization = (pred - target).abs().mean()
        return adaptive + self.lambda_reg * regularization
