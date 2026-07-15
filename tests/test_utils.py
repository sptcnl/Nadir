"""Tests for reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch

from nadir.utils import resolve_device, set_seed


def test_set_seed_reproducible() -> None:
    set_seed(123)
    a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    set_seed(123)
    b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a == b


def test_set_seed_configures_cudnn() -> None:
    set_seed(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_resolve_device_cpu_fallback() -> None:
    assert resolve_device("cpu").type == "cpu"
    # "cuda" must never raise, even on CPU-only machines.
    assert resolve_device("cuda").type in ("cuda", "cpu")
