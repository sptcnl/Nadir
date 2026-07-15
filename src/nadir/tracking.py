"""Weights & Biases integration.

Kept behind a thin wrapper so the training loop never touches wandb directly:
runs must work identically in online, offline, and disabled modes.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

import wandb


def init_wandb(cfg: DictConfig) -> "wandb.sdk.wandb_run.Run | None":
    """Initialise a W&B run from the Hydra config.

    Returns None when mode is "disabled" so callers can guard logging calls
    with a simple truthiness check.
    """
    mode: str = cfg.wandb.mode
    if mode == "disabled":
        return None
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=mode,
        tags=list(cfg.wandb.tags),
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
    )
    return run


def log_metrics(run: "wandb.sdk.wandb_run.Run | None", metrics: dict[str, Any], step: int) -> None:
    """Log a metrics dict if a run is active; no-op otherwise."""
    if run is not None:
        run.log(metrics, step=step)
