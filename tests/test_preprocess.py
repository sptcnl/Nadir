"""Tests for radiometric normalization."""

from __future__ import annotations

import numpy as np

from nadir.data.preprocess import denormalize_s2, normalize_s1, normalize_s2


def test_s2_range_and_endpoints() -> None:
    dn = np.array([[0, 5000, 10000, 28000]], dtype=np.uint16)  # incl. >10000 cloud DN
    out = normalize_s2(dn)
    assert out.dtype == np.float32
    assert out.min() >= -1.0 and out.max() <= 1.0
    np.testing.assert_allclose(out[0], [-1.0, 0.0, 1.0, 1.0], atol=1e-6)


def test_s2_preserves_band_ratios_below_clip() -> None:
    # Uniform clipping must not distort spectra: reflectance ratios survive
    # the affine map after denormalization.
    dn = np.array([[[2000.0]], [[4000.0]]], dtype=np.float32)  # (2, 1, 1)
    refl = denormalize_s2(normalize_s2(dn))
    np.testing.assert_allclose(refl[1] / refl[0], 2.0, rtol=1e-5)


def test_s2_roundtrip() -> None:
    dn = np.linspace(0, 10000, 11, dtype=np.float32)
    refl = denormalize_s2(normalize_s2(dn))
    np.testing.assert_allclose(refl, dn / 10000.0, atol=1e-6)


def test_s1_range_and_endpoints() -> None:
    db = np.array([-30.0, -25.0, -12.5, 0.0, 5.0], dtype=np.float32)
    out = normalize_s1(db)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [-1.0, -1.0, 0.0, 1.0, 1.0], atol=1e-6)
