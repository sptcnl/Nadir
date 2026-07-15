"""DSen2-CR: SAR-guided deep residual network for cloud removal.

Re-implemented in PyTorch from the original Keras code
(Meraner et al., "Cloud removal in Sentinel-2 imagery using a deep residual
neural network and SAR-optical data fusion", ISPRS 2020;
https://github.com/ameraner/dsen2-cr, Code/dsen2cr_network.py). Architecture
facts verified against that source:

  - inputs: 13-band cloudy S2 and 2-band S1 are concatenated channel-wise;
  - head: 3x3 conv (F=256) + ReLU;
  - body: B=16 residual blocks (conv3x3 -> ReLU -> conv3x3, output scaled by
    0.1, EDSR-style, no batch norm), as configured in dsen2cr_main.py
    ("num_layers = 16  # B value in paper");
  - tail: 3x3 conv to 13 channels;
  - long skip: the tail output is added to the OPTICAL input only
    ("Add()([x, input_opt])") — the network predicts a residual correction
    to the cloudy image, never to the SAR channels.

Clear-region preservation: the architecture does NOT hard-copy input pixels
outside the cloud mask. Preservation emerges from two soft mechanisms:
(1) the long skip makes the identity mapping trivial to represent, and
(2) CARL's clear-region term (see nadir.losses.carl) explicitly pulls the
prediction toward the cloudy input outside the mask. For evaluation or
deployment where exact preservation is required, use `paste_clear_pixels`.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResBlock(nn.Module):
    """conv3x3 -> ReLU -> conv3x3, scaled residual add (no normalization)."""

    def __init__(self, features: int, res_scale: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.conv2(self.relu(self.conv1(x)))


class DSen2CR(nn.Module):
    """SAR + cloudy-S2 -> clear-S2 residual CNN.

    Args:
        in_channels: concatenated input channels (13 optical + 2 SAR = 15).
        out_channels: predicted optical channels (13).
        features: feature width F (paper: 256).
        num_blocks: residual blocks B (paper: 16).
        res_scale: residual scaling inside blocks (paper: 0.1).
        grad_checkpointing: recompute block activations in backward to trade
            compute for memory (useful at F=256 on a 16GB GPU).
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 13,
        features: int = 256,
        num_blocks: int = 16,
        res_scale: float = 0.1,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= out_channels:
            raise ValueError("in_channels must include SAR channels on top of optical")
        self.out_channels = out_channels
        self.grad_checkpointing = grad_checkpointing
        self.head = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.body = nn.ModuleList(ResBlock(features, res_scale) for _ in range(num_blocks))
        self.tail = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Keras `he_uniform` equivalent (original uses it for every conv).
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, s2_cloudy: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s2_cloudy: (B, 13, H, W) normalized cloudy optical input.
            s1: (B, 2, H, W) normalized SAR input.

        Returns:
            (B, 13, H, W) predicted clear optical image.
        """
        x = torch.cat([s2_cloudy, s1], dim=1)
        h = self.relu(self.head(x))
        for block in self.body:
            if self.grad_checkpointing and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        # Long skip over the optical input only (matches the original).
        return s2_cloudy + self.tail(h)


def paste_clear_pixels(
    pred: torch.Tensor, s2_cloudy: torch.Tensor, cloud_mask: torch.Tensor
) -> torch.Tensor:
    """Composite: keep original pixels outside the cloud/shadow mask.

    Args:
        pred: (B, 13, H, W) model output.
        s2_cloudy: (B, 13, H, W) original cloudy input (same normalization).
        cloud_mask: (B, H, W) binary mask, 1 = cloud or shadow.

    Returns:
        (B, 13, H, W) image equal to `s2_cloudy` where the mask is 0 and to
        `pred` where the mask is 1.
    """
    m = cloud_mask.to(pred.dtype).unsqueeze(1)
    return m * pred + (1.0 - m) * s2_cloudy
