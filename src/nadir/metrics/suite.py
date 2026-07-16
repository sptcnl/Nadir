"""Evaluation metrics with mask-aware region aggregation.

Every metric is computed as a dense per-pixel map, then averaged over three
regions derived from the cloud mask:

    full  — whole image,
    cloud — pixels where mask != 0 (thin/thick/shadow): the actual
            reconstruction task,
    clear — pixels where mask == 0: measures whether the model corrupts
            already-valid observations.

Reporting only "full" numbers is misleading: on a patch with 10% cloud cover
a model that copies its input gets excellent full-image scores while failing
the task completely. The cloud/clear separation is therefore mandatory.

Inputs are TOA reflectance in [0, 1] (use denormalize_s2 / the torch
equivalent on model outputs before calling). PSNR uses data_range = 1.

LPIPS operates on RGB only. Band mapping (documented per Step-5 requirement):
the 13-band L1C stack is ordered B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,
B11,B12, so RGB = (B04, B03, B02) = indices (3, 2, 1). Reflectance in [0, 1]
is mapped linearly to [-1, 1] as LPIPS expects; no brightness gain is applied
so the mapping stays invertible and model-independent.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity

# RGB band indices in the 13-band Sentinel-2 L1C stack (see module docstring).
RGB_BAND_INDICES: tuple[int, int, int] = (3, 2, 1)

_EPS = 1e-12


def region_mean(values: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Mean of per-pixel `values` (B, H, W) over `region` (B, H, W) per image.

    Images whose region is empty yield NaN (aggregated with nanmean later);
    e.g. a fully cloud-free patch has no "cloud" region.
    """
    region_f = region.to(values.dtype)
    denom = region_f.sum(dim=(-2, -1))
    num = (values * region_f).sum(dim=(-2, -1))
    out = num / denom.clamp(min=_EPS)
    out[denom == 0] = float("nan")
    return out


def mae_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> per-pixel L1 averaged over channels, (B, H, W)."""
    return (pred - target).abs().mean(dim=1)


def mse_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> per-pixel squared error averaged over channels."""
    return (pred - target).square().mean(dim=1)


def psnr_from_mse(mse: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image PSNR in dB from per-image MSE; clamped at ~120 dB via eps."""
    return 10.0 * torch.log10(data_range**2 / mse.clamp(min=_EPS))


def sam_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral Angle Mapper per pixel, in degrees. (B, C, H, W) -> (B, H, W).

    Angle between the C-dimensional spectral vectors of prediction and target.
    Invariant to per-pixel scaling (illumination), sensitive to band-ratio
    distortion — exactly the failure mode PSNR cannot see.
    """
    # float64: acos is ill-conditioned near cos=1, and float32 noise there
    # shows up as a spurious ~0.01 deg floor on identical images.
    p = pred.double()
    t = target.double()
    dot = (p * t).sum(dim=1)
    norm = p.norm(dim=1) * t.norm(dim=1)
    cos = (dot / norm.clamp(min=_EPS)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).to(pred.dtype)


def ssim_map(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-pixel SSIM averaged over channels, (B, H, W).

    Uses skimage's multichannel SSIM (7x7 uniform window) with full=True to
    obtain the dense map, so it can be aggregated per region like every other
    metric. Runs on CPU/numpy — evaluation only, not in the training step.
    """
    maps = []
    for p, t in zip(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), strict=True):
        _, smap = structural_similarity(
            t, p, channel_axis=0, data_range=data_range, full=True
        )
        maps.append(np.asarray(smap, dtype=np.float32).mean(axis=0))
    return torch.from_numpy(np.stack(maps)).to(pred.device)


class MetricSuite:
    """Computes PSNR/MAE/SAM/SSIM(/LPIPS) split by full/cloud/clear regions.

    Args:
        device: device for LPIPS inference.
        lpips_net: LPIPS backbone ("alex" per the LPIPS paper's recommendation
            for evaluation, "squeeze"/"vgg" also valid); None disables LPIPS.
    """

    def __init__(
        self, device: torch.device | str = "cpu", lpips_net: str | None = "alex"
    ) -> None:
        self.device = torch.device(device)
        self._lpips = None
        if lpips_net is not None:
            import lpips  # local import: heavy, downloads backbone weights on first use

            # spatial=True yields a per-pixel distance map for region masking.
            self._lpips = lpips.LPIPS(net=lpips_net, spatial=True).to(self.device).eval()

    def _lpips_map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert self._lpips is not None
        rgb_pred = pred[:, RGB_BAND_INDICES].to(self.device) * 2.0 - 1.0
        rgb_target = target[:, RGB_BAND_INDICES].to(self.device) * 2.0 - 1.0
        return self._lpips(rgb_pred, rgb_target).squeeze(1)  # (B, H, W)

    @torch.no_grad()
    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, float]:
        """
        Args:
            pred/target: (B, 13, H, W) TOA reflectance in [0, 1].
            mask: (B, H, W) cloud mask; nonzero = cloud or shadow (3-class ok).

        Returns:
            {"<metric>/<region>": value} with regions full/cloud/clear.
            Values are batch means; NaN if the region is empty batch-wide.
        """
        pred = pred.float()
        target = target.float()
        regions = {
            "full": torch.ones_like(mask, dtype=torch.bool),
            "cloud": mask != 0,
            "clear": mask == 0,
        }
        maps: dict[str, torch.Tensor] = {
            "mae": mae_map(pred, target),
            "sam": sam_map(pred, target),
            "ssim": ssim_map(pred, target),
        }
        if self._lpips is not None:
            maps["lpips"] = self._lpips_map(pred, target).to(pred.device)
        mse = mse_map(pred, target)

        out: dict[str, float] = {}
        for region_name, region in regions.items():
            region = region.to(pred.device)
            for metric_name, metric_map in maps.items():
                per_image = region_mean(metric_map, region)
                out[f"{metric_name}/{region_name}"] = _nanmean(per_image)
            # PSNR is log of the region MSE, not a mean of a per-pixel map.
            psnr = psnr_from_mse(region_mean(mse, region))
            out[f"psnr/{region_name}"] = _nanmean(psnr)
        return out


def _nanmean(x: torch.Tensor) -> float:
    value = torch.nanmean(x)
    return float(value) if not torch.isnan(value) else math.nan
