"""End-to-end training smoke test: tiny model, tiny dummy data, CPU, one epoch.

Verifies the whole chain (dataset -> loader -> model -> CARL -> metrics ->
checkpointing -> resume) without a GPU or network access.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from nadir.data.dummy import DummyConfig, generate
from nadir.train import run_training

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.fixture(scope="module")
def dummy_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("smoke")
    generate(DummyConfig(out_dir=root, num_scenes=3, patches_per_scene=2, size=64, seed=11))
    return root


def _cfg(dummy_root: Path, work: Path, extra: list[str] | None = None) -> object:
    overrides = [
        f"data.root={dummy_root.as_posix()}",
        "data.loader.num_workers=0",
        "device=cpu",
        "train.epochs=1",
        "train.batch_size=2",
        "train.amp=false",
        "train.log.every_steps=1",
        "train.log.image_grid_samples=2",
        f"train.checkpoint.dir={ (work / 'ckpt').as_posix() }",
        "model.features=8",
        "model.num_blocks=1",
        "eval.lpips_net=null",  # no network access in tests
        "wandb.mode=disabled",
    ] + (extra or [])
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def test_one_epoch_end_to_end(dummy_root: Path, tmp_path: Path) -> None:
    metrics = run_training(_cfg(dummy_root, tmp_path))  # type: ignore[arg-type]
    # All three regions reported for every metric.
    for name in ("psnr", "mae", "sam", "ssim"):
        for region in ("full", "cloud", "clear"):
            assert f"{name}/{region}" in metrics
    assert not math.isnan(metrics["mae/full"])
    assert (tmp_path / "ckpt" / "last.pt").exists()
    assert (tmp_path / "ckpt" / "epoch0000.pt").exists()


def test_resume_from_checkpoint(dummy_root: Path, tmp_path: Path) -> None:
    run_training(_cfg(dummy_root, tmp_path))  # type: ignore[arg-type]
    ckpt = tmp_path / "ckpt" / "last.pt"
    cfg = _cfg(
        dummy_root,
        tmp_path,
        extra=[f"train.checkpoint.resume={ckpt.as_posix()}", "train.epochs=2"],
    )
    metrics = run_training(cfg)  # type: ignore[arg-type]
    assert not math.isnan(metrics["mae/full"])
