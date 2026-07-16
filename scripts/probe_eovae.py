"""B-prime feasibility probe: EO-VAE roundtrip on a 13-band Sentinel-2 patch.

Measurement/verification utility (not model code). Verifies that the released
EO-VAE weights (HF `nilsleh/eo-vae`) download, accept the S2 L1C 13-band
configuration, and produce a finite encode->decode roundtrip; reports
PSNR/MAE/SAM of the roundtrip in reflectance space.

Environment: EO-VAE pins torchvision==0.16.2 — do NOT install it into the
main venv. Use an isolated venv:

    uv venv eovae-venv --python 3.11
    uv pip install --python <venv> torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python <venv> einops lightning safetensors omegaconf \
        huggingface_hub numpy focal-frequency-loss
    git clone https://github.com/nilsleh/eo-vae
    uv pip install --python <venv> --no-deps -e eo-vae

Usage:
    python scripts/probe_eovae.py --patch path/to/s2_dn.npy   # (13, H, W) raw DN
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from eo_vae.models.new_autoencoder import EOFluxVAE

# S2 L1C central wavelengths (um), B01..B12 with B8A after B08, per EO-VAE README.
WVS_S2L1C = torch.tensor(
    [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190],
    dtype=torch.float32,
)

# Input convention verified in eo_vae/datasets/terramesh.py (MultimodalNormalize):
# per-band z-score of raw DN with TerraMesh statistics — NOT [0,1] reflectance.
S2L1C_MEAN = torch.tensor(
    [2357.090, 2137.398, 2018.799, 2082.998, 2295.663, 2854.548, 3122.860,
     3040.571, 3306.491, 1473.849, 506.072, 2472.840, 1838.943]
).view(1, 13, 1, 1)
S2L1C_STD = torch.tensor(
    [1673.639, 1722.641, 1602.205, 1873.138, 1866.055, 1779.839, 1776.496,
     1724.114, 1771.041, 1079.786, 512.404, 1340.879, 1172.435]
).view(1, 13, 1, 1)


def sam_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = (a * b).sum(0)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-12
    cos = np.clip(dot / denom, -1, 1)
    return float(np.degrees(np.arccos(cos)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", required=True, help="npy file with (13, H, W) raw S2 DN")
    args = parser.parse_args()

    model = EOFluxVAE.from_pretrained(
        repo_id="nilsleh/eo-vae",
        ckpt_filename="eo-vae.ckpt",
        config_filename="model_config.yaml",
        device="cpu",
    )
    model.eval()
    print(f"loaded EOFluxVAE, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    dn = torch.from_numpy(np.load(args.patch)).unsqueeze(0).float()
    x = (dn - S2L1C_MEAN) / S2L1C_STD
    print(f"input: {tuple(x.shape)} z-scored range=({x.min():.3f},{x.max():.3f})")

    with torch.no_grad():
        z = model.encode_spatial_normalized(x, WVS_S2L1C)
        recon = model.reconstruct(x, WVS_S2L1C)
    print(f"latent: {tuple(z.shape)}")
    print(
        f"recon:  {tuple(recon.shape)} finite={bool(torch.isfinite(recon).all())} "
        f"range=({recon.min():.3f},{recon.max():.3f})"
    )

    recon_dn = recon * S2L1C_STD + S2L1C_MEAN
    a = (dn[0] / 10000.0).clamp(0, 1).numpy()
    b = (recon_dn[0] / 10000.0).clamp(0, 1).numpy()
    mse = float(((a - b) ** 2).mean())
    print(
        f"roundtrip in reflectance space: PSNR={10 * math.log10(1.0 / max(mse, 1e-12)):.2f} dB, "
        f"MAE={np.abs(a - b).mean():.4f}, SAM={sam_deg(a, b):.3f} deg"
    )
    print("PROBE OK")


if __name__ == "__main__":
    main()
