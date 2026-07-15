"""Evaluation entrypoint.

Usage:
    python -m nadir.eval checkpoint=path/to/ckpt.pt

Implemented at Step 5/6 together with the metrics module.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO(Step 5/6): load checkpoint, run metrics (full / in-mask / out-of-mask).
    raise NotImplementedError("Evaluation lands at Step 5/6.")


if __name__ == "__main__":
    main()
