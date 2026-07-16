"""Training entrypoint.

Usage:
    python -m nadir.train +experiment=dsen2cr_dummy
    python -m nadir.train train.checkpoint.resume=path/to/last.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nadir.data.sen12mscr import Sen12MSCRDataset, build_datasets
from nadir.losses.carl import CARLLoss
from nadir.metrics import MetricSuite
from nadir.models.dsen2cr import DSen2CR
from nadir.tracking import init_wandb, log_metrics
from nadir.utils import resolve_device, seed_worker, set_seed
from nadir.visualize import prediction_grid


def build_model(cfg: DictConfig) -> DSen2CR:
    return DSen2CR(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        features=cfg.model.features,
        num_blocks=cfg.model.num_blocks,
        res_scale=cfg.model.res_scale,
        grad_checkpointing=cfg.train.grad_checkpointing,
    )


def make_loader(
    dataset: Sen12MSCRDataset, cfg: DictConfig, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] model space -> [0, 1] reflectance (torch mirror of denormalize_s2)."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(
    model: DSen2CR,
    loader: DataLoader,
    suite: MetricSuite,
    device: torch.device,
    grid_samples: int,
) -> tuple[dict[str, float], np.ndarray | None]:
    """Run mask-split metrics over a loader; also render a qualitative grid
    (SAR | cloudy | prediction | ground truth) from the first batch."""
    model.eval()
    batch_metrics: list[dict[str, float]] = []
    grid: np.ndarray | None = None
    for i, batch in enumerate(loader):
        s1 = batch["s1"].to(device)
        s2_cloudy = batch["s2_cloudy"].to(device)
        s2_clear = batch["s2_clear"].to(device)
        mask = batch["mask"].to(device)
        pred = model(s2_cloudy, s1)
        batch_metrics.append(suite(denormalize(pred), denormalize(s2_clear), mask))
        if i == 0 and grid_samples > 0:
            n = min(grid_samples, s1.shape[0])
            grid = prediction_grid(
                s1[:n],
                denormalize(s2_cloudy[:n]),
                denormalize(pred[:n]),
                denormalize(s2_clear[:n]),
            )
    # nanmean over batches: NaN marks batches whose region was empty (e.g. a
    # cloud-free batch has no "cloud" pixels). Unweighted over batches — exact
    # enough for monitoring; nadir.eval reports the properly pooled numbers.
    aggregated = {
        key: float(np.nanmean([m[key] for m in batch_metrics])) for key in batch_metrics[0]
    }
    return aggregated, grid


def _checkpoint_dir(cfg: DictConfig) -> Path:
    """Checkpoints go under the Hydra run dir; fall back to CWD when the app
    is driven without a Hydra context (tests use compose())."""
    try:
        from hydra.core.hydra_config import HydraConfig

        base = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        base = Path.cwd()
    return base / cfg.train.checkpoint.dir


def _save_checkpoint(
    path: Path,
    model: DSen2CR,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    cfg: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def run_training(cfg: DictConfig) -> dict[str, float]:
    """Full training loop; returns the final validation metrics."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run = init_wandb(cfg)

    datasets = build_datasets(cfg)
    train_loader = make_loader(datasets["train"], cfg, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(datasets["val"], cfg, shuffle=False, seed=cfg.seed)

    model = build_model(cfg).to(device)
    loss_fn = CARLLoss(lambda_reg=cfg.model.loss.lambda_reg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    suite = MetricSuite(device=device, lpips_net=cfg.eval.lpips_net)
    ckpt_dir = _checkpoint_dir(cfg)

    start_epoch = 0
    global_step = 0
    if cfg.train.checkpoint.resume is not None:
        state: dict[str, Any] = torch.load(
            cfg.train.checkpoint.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        print(f"resumed from {cfg.train.checkpoint.resume} at epoch {start_epoch}")

    # bf16 autocast: RTX 4080 (Ada) has native bf16; no GradScaler needed
    # (bf16 has fp32's exponent range, so no loss-scale underflow issue).
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"

    val_metrics: dict[str, float] = {}
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        for batch in train_loader:
            s1 = batch["s1"].to(device, non_blocking=True)
            s2_cloudy = batch["s2_cloudy"].to(device, non_blocking=True)
            s2_clear = batch["s2_clear"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                pred = model(s2_cloudy, s1)
                loss = loss_fn(pred, s2_cloudy, s2_clear, mask)
            loss.backward()
            if cfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step % cfg.train.log.every_steps == 0:
                log_metrics(
                    run,
                    {"train/loss": float(loss.detach()), "train/epoch": epoch},
                    step=global_step,
                )

        val_metrics, grid = evaluate(
            model, val_loader, suite, device, cfg.train.log.image_grid_samples
        )
        payload: dict[str, Any] = {f"val/{k}": v for k, v in val_metrics.items()}
        if grid is not None and run is not None:
            payload["val/examples"] = wandb.Image(
                grid, caption="rows: samples | cols: SAR(VV), cloudy, pred, GT"
            )
        log_metrics(run, payload, step=global_step)
        pretty = {k: round(v, 4) for k, v in val_metrics.items()}
        print(f"epoch {epoch}: {pretty}")

        _save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, global_step, cfg)
        if (epoch + 1) % cfg.train.checkpoint.every_epochs == 0:
            _save_checkpoint(
                ckpt_dir / f"epoch{epoch:04d}.pt", model, optimizer, epoch, global_step, cfg
            )

    if run is not None:
        run.finish()
    return val_metrics


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
