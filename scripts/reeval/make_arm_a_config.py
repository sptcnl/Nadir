"""Produce the Arm A config by patching EMRDM's released test config.

Arm A = faithful reproduction: EMRDM's own released test-project.yaml with
ONLY the machine-specific paths changed (checkpoint + data roots). Everything
that affects the numbers — network, sampler (ResidualEulerEDMSampler,
num_steps=5), EDM preconditioning, rescale, cloud_masks=null, batch_size=1 —
is left exactly as released.

Run their test loop with:
    cd ~/emrdm/EMRDM
    python main.py --base configs/arm_a_sentinel.yaml --enable_tf32 -t false \
        --logdir ~/logs/arm_a -n armA
which dispatches trainer.test() -> final_* metrics in the testtube metrics.csv.

Usage:
    python scripts/reeval/make_arm_a_config.py \
        --in  ~/emrdm/artifacts/sentinel/test-project.yaml \
        --ckpt ~/emrdm/artifacts/sentinel/last.ckpt \
        --data-root ~/data/sen12mscr \
        --out ~/emrdm/EMRDM/configs/arm_a_sentinel.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.inp).expanduser().read_text())

    # 1) checkpoint: released path -> local weights (hash-verified in §7.3)
    cfg["model"]["params"]["ckpt_path"] = str(Path(args.ckpt).expanduser())

    # 2) data roots: their cluster path -> our extracted test split. All four
    # split entries are repointed; only 'test' has data on disk, and their
    # get_paths() filters by split, so train/val/predict resolve to empty
    # without error (verified in sentinel.py).
    root = str(Path(args.data_root).expanduser())
    for split in ("train", "validation", "test", "predict"):
        if split in cfg["data"]["params"]:
            cfg["data"]["params"][split]["params"]["root"] = root

    # Assert nothing else drifted from the released numbers-affecting config.
    sampler = cfg["model"]["params"]["sampler_config"]["params"]
    assert sampler["num_steps"] == 5, "num_steps changed — not a faithful Arm A"
    assert cfg["model"]["params"]["use_ema"] is True, "EMA off — not faithful"
    assert cfg["data"]["params"]["test"]["params"]["rescale"] is True
    assert cfg["data"]["params"]["test"]["params"]["cloud_masks"] is None

    out = Path(args.out).expanduser()
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote Arm A config -> {out}")
    print(f"  ckpt   = {cfg['model']['params']['ckpt_path']}")
    print(f"  root   = {root}")
    print(f"  sampler= {sampler['target'].split('.')[-1]} num_steps={sampler['num_steps']}")


if __name__ == "__main__":
    main()
