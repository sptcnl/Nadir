"""Training entrypoint.

Usage:
    python -m nadir.train +experiment=dsen2cr_dummy

The full training loop lands at Step 6; for now this entrypoint validates
config composition, seeding, and device resolution end-to-end.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from nadir.utils import resolve_device, set_seed


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"device: {device}")
    # TODO(Step 6): build datamodule, model, and run the training loop.


if __name__ == "__main__":
    main()
