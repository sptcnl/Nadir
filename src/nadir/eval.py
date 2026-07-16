"""Evaluation entrypoint: mask-split metrics on the (geographically held-out)
test split.

Usage:
    python -m nadir.eval +experiment=dsen2cr_dummy checkpoint=outputs/.../last.pt
"""

from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from nadir.data.sen12mscr import build_datasets
from nadir.metrics import MetricSuite
from nadir.train import build_model, evaluate, make_loader
from nadir.utils import resolve_device, set_seed


def run_eval(cfg: DictConfig) -> dict[str, float]:
    if cfg.checkpoint is None:
        raise ValueError("pass checkpoint=path/to/last.pt")
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    model = build_model(cfg).to(device)
    state = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    loader = make_loader(build_datasets(cfg)["test"], cfg, shuffle=False, seed=cfg.seed)
    suite = MetricSuite(device=device, lpips_net=cfg.eval.lpips_net)
    metrics, _ = evaluate(model, loader, suite, device, grid_samples=0)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
