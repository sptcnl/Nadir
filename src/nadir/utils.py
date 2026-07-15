"""Reproducibility and small shared utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs (python, numpy, torch) and configure cuDNN.

    Args:
        seed: Global seed applied to every RNG source.
        deterministic: If True, force deterministic cuDNN kernels. This can
            slow training down but makes runs bit-reproducible on the same
            hardware/driver stack.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Required by some CUDA deterministic implementations (e.g. cublas).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive per-worker seeds from torch's seed.

    Prevents identical augmentation streams across workers while staying
    reproducible given a fixed global seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to CPU when CUDA is absent."""
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
