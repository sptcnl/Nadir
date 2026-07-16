"""Step 3c/3d: EMRDM inference over one scene with explicit TF32 control.

Mirrors predict_step (sgm/models/diffusion.py) exactly — their preprocessing
(SEN12MSCRInterface semantics), EMA weights, ResidualEulerEDMSampler — while
iterating triplet files directly (their hardcoded test split needs all 10
scenes on disk; Step 3 has one). Per patch it writes:
  - their img_metrics values (CSV row), computed in [0,1] space as they do;
  - the prediction as uint16 DN (round(clip(refl,0,1)*10000)) in an .npz,
    the file-handoff format consumed by the Nadir-side harness.

TF32 (--tf32 on) replicates `main.py --enable_tf32`: cuda.matmul + cuDNN
TF32 on, natten GEMM-NA TF32 at its default (on). --tf32 off disables all
three. Same seed => identical sampler noise across the two runs.

Runs in the emrdm venv:
    venv/bin/python emrdm_infer_scene.py --emrdm ~/emrdm/EMRDM \
        --ckpt ~/emrdm/artifacts/sentinel/last.ckpt \
        --root ~/data/sen12mscr --season ROIs2017_winter --scene 63 \
        --tf32 on --out-dir ~/logs/infer_w63_tf32on
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emrdm", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--season", default="ROIs2017_winter")
    parser.add_argument("--scene", type=int, default=63)
    parser.add_argument("--tf32", choices=["on", "off"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = all patches")
    parser.add_argument("--save-preds", action="store_true", help="write uint16 .npz predictions")
    args = parser.parse_args()

    emrdm = Path(args.emrdm).expanduser()
    sys.path.insert(0, str(emrdm))
    import natten
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from sgm.data.sentinel.sentinel import process_MS, process_SAR, read_img, read_tif
    from sgm.modules.learning.metrics import img_metrics
    from sgm.util import instantiate_from_config

    if args.tf32 == "on":  # replicate main.py --enable_tf32 (+ natten default)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        natten.disable_tf32()
    print(f"tf32={args.tf32} matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn={torch.backends.cudnn.allow_tf32} "
          f"natten_gemm_tf32={natten.is_tf32_in_gemm_na_enabled()}")

    cfg = OmegaConf.load(emrdm / "configs/example_training/sentinel.yaml")
    cfg.model.params.ckpt_path = str(Path(args.ckpt).expanduser())
    engine = instantiate_from_config(cfg.model).eval().cuda()

    root = Path(args.root).expanduser()
    s1_dir = root / f"{args.season}_s1" / f"s1_{args.scene}"
    s1_files = sorted(s1_dir.glob("*.tif"), key=lambda p: int(p.stem.rsplit("_p", 1)[1]))
    if args.limit:
        s1_files = s1_files[: args.limit]

    out_dir = Path(args.out_dir).expanduser()
    pred_dir = out_dir / "pred"
    (pred_dir if args.save_preds else out_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rows = []
    t0 = time.time()
    for s1_path in s1_files:
        patch = int(s1_path.stem.rsplit("_p", 1)[1])
        s2_path = root / f"{args.season}_s2" / f"s2_{args.scene}" / s1_path.name.replace(
            "_s1_", "_s2_"
        )
        s2c_path = root / f"{args.season}_s2_cloudy" / f"s2_cloudy_{args.scene}" / (
            s1_path.name.replace("_s1_", "_s2_cloudy_")
        )
        # SEN12MSCRInterface semantics: process_* -> [0,1], then *2-1.
        s1 = torch.tensor(process_SAR(read_img(read_tif(str(s1_path))), "default")).float()
        s2c = torch.tensor(process_MS(read_img(read_tif(str(s2c_path))), "default")).float()
        target = torch.tensor(process_MS(read_img(read_tif(str(s2_path))), "default")).float()
        batch = {
            "target": (target * 2 - 1).unsqueeze(0).cuda(),
            "S1": (s1 * 2 - 1).unsqueeze(0).cuda(),
            "S2": (s2c * 2 - 1).unsqueeze(0).cuda(),
            "S1S2": (torch.cat([s1, s2c], dim=0) * 2 - 1).unsqueeze(0).cuda(),
            "image_path": [s2_path.name],
        }
        with torch.no_grad():
            mu = engine.get_input(batch, engine.mean_key)
            c, uc = engine.conditioner.get_unconditional_conditioning(
                batch, force_uc_zero_embeddings=[]
            )
            z_mu = engine.encode_first_stage(mu)
            with engine.ema_scope():
                samples, _ = engine.sample(c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=1)
                samples = engine.decode_first_stage(samples)
            target01 = engine.scale_01(batch["target"])
            samples01 = engine.scale_01(samples)
            m = img_metrics(target=target01, pred=samples01)
        rows.append({"patch": patch, **{k: m[k] for k in ("MAE", "RMSE", "PSNR", "SAM", "SSIM")}})
        if args.save_preds:
            dn = (samples01[0].clamp(0, 1).cpu().numpy() * 10000.0).round().astype(np.uint16)
            np.savez_compressed(pred_dir / f"p{patch}.npz", dn=dn)
        if len(rows) % 50 == 0:
            rate = len(rows) / (time.time() - t0)
            print(f"{len(rows)}/{len(s1_files)} patches ({rate:.2f}/s)", flush=True)

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    agg = {k: sum(r[k] for r in rows) / len(rows) for k in ("MAE", "RMSE", "PSNR", "SAM", "SSIM")}
    print(f"AGGREGATE n={len(rows)} " + " ".join(f"{k}={v:.6f}" for k, v in agg.items()))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
