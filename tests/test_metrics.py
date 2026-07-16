"""Tests for the metric suite: correctness on hand-computable cases and,
critically, that the full/cloud/clear region separation behaves as claimed."""

from __future__ import annotations

import math

import pytest
import torch

from nadir.metrics import MetricSuite, sam_map

SIZE = 16  # >= 7 required by the SSIM window


def _suite() -> MetricSuite:
    return MetricSuite(lpips_net=None)  # LPIPS covered separately (weight download)


def _images() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    target = torch.rand(2, 13, SIZE, SIZE, generator=g)
    return target.clone(), target


def test_perfect_prediction() -> None:
    pred, target = _images()
    mask = torch.zeros(2, SIZE, SIZE)
    mask[:, :4] = 2  # some thick cloud so all regions are non-empty
    m = _suite()(pred, target, mask)
    for region in ("full", "cloud", "clear"):
        assert m[f"mae/{region}"] == 0.0
        assert m[f"sam/{region}"] < 1e-3
        assert m[f"ssim/{region}"] == pytest.approx(1.0, abs=1e-6)
        assert m[f"psnr/{region}"] > 100.0  # eps-clamped "infinity"


def test_constant_offset_mae_and_psnr() -> None:
    _, target = _images()
    target = torch.full_like(target, 0.5)
    pred = target + 0.1
    mask = torch.zeros(2, SIZE, SIZE)
    m = _suite()(pred, target, mask)
    assert m["mae/full"] == pytest.approx(0.1, rel=1e-5)
    # MSE = 0.01 -> PSNR = 10*log10(1/0.01) = 20 dB.
    assert m["psnr/full"] == pytest.approx(20.0, rel=1e-4)


def test_sam_is_scale_invariant_but_mae_is_not() -> None:
    _, target = _images()
    target = target + 0.1  # keep spectra away from zero
    pred = target * 0.5  # same spectral direction, half brightness
    angles = sam_map(pred, target)
    assert float(angles.max()) < 1e-3  # SAM sees no spectral distortion
    mask = torch.zeros(2, SIZE, SIZE)
    m = _suite()(pred, target, mask)
    assert m["mae/full"] > 0.05  # pixel metrics do see the error


def test_sam_detects_band_ratio_distortion() -> None:
    _, target = _images()
    target = target + 0.1
    pred = target.clone()
    pred[:, 0] *= 3.0  # corrupt one band's ratio only
    m = _suite()(pred, target, torch.zeros(2, SIZE, SIZE))
    assert m["sam/full"] > 1.0  # degrees


def test_region_separation() -> None:
    # Corrupt ONLY the masked (cloud) area: clear-region metrics must stay
    # perfect, cloud-region metrics must degrade, full sits in between.
    pred, target = _images()
    mask = torch.zeros(2, SIZE, SIZE)
    mask[:, : SIZE // 2] = 1
    noise = torch.rand_like(pred) * 0.5
    region = mask.bool().unsqueeze(1).expand_as(pred)
    pred[region] = (pred + noise)[region].clamp(0, 1)

    m = _suite()(pred, target, mask)
    assert m["mae/clear"] == 0.0
    assert m["mae/cloud"] > 0.05
    assert 0.0 < m["mae/full"] < m["mae/cloud"]
    assert m["psnr/clear"] > m["psnr/full"] > m["psnr/cloud"]
    # SSIM's sliding window bleeds across the region boundary, so "clear" is
    # near-perfect but not exactly 1; the ordering must still hold.
    assert m["ssim/clear"] > m["ssim/full"] > m["ssim/cloud"]


def test_empty_region_yields_nan() -> None:
    pred, target = _images()
    m = _suite()(pred, target, torch.zeros(2, SIZE, SIZE))  # no clouds at all
    assert math.isnan(m["mae/cloud"])
    assert math.isnan(m["psnr/cloud"])
    assert m["mae/clear"] == m["mae/full"]


def test_lpips_rgb_pathway() -> None:
    # squeeze backbone keeps the first-use download small; the pathway
    # (band extraction, [-1,1] mapping, spatial map, region masking) is
    # identical for alex.
    suite = MetricSuite(lpips_net="squeeze")
    # LPIPS backbones downsample aggressively; 64x64 is a safe minimum size.
    g = torch.Generator().manual_seed(0)
    target = torch.rand(2, 13, 64, 64, generator=g)
    mask = torch.zeros(2, 64, 64)
    mask[:, :32] = 2
    m_same = suite(target.clone(), target, mask)
    assert m_same["lpips/full"] == pytest.approx(0.0, abs=1e-4)
    pred = (target + torch.rand_like(target) * 0.8).clamp(0, 1)
    m_diff = suite(pred, target, mask)
    assert m_diff["lpips/full"] > m_same["lpips/full"]
    assert not math.isnan(m_diff["lpips/cloud"]) and not math.isnan(m_diff["lpips/clear"])
