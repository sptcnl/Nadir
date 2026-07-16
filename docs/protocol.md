# Nadir Evaluation & Preprocessing Protocol

**Status: v1, 2026-07-16.** This document declares every preprocessing and
evaluation constant used by this project, each with its basis — or with an
honest "arbitrary, sensitivity unmeasured" where no basis exists. The
absence of exactly this declaration in published work (see
`design_decisions.md` §2.3) is a core critique this project makes; we hold
ourselves to the standard first.

Basis labels:
- **[physical]** — derived from sensor physics or data format.
- **[convention]** — matches a cited external convention, adopted for
  comparability.
- **[choice]** — a reasoned project decision with stated rationale.
- **[ARBITRARY]** — no evidence behind the value; sensitivity unmeasured
  unless stated. These are the honest debts.

## 1. Sentinel-2 optical (13 bands, L1C)

| Constant | Value | Basis |
|---|---|---|
| Band count/order | 13, L1C order B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12 | [physical] L1C product definition |
| DN interpretation | reflectance × 10000 | [physical] Sentinel-2 L1C spec |
| Clipping | [0, 10000], identical for all bands | [convention] DSen2-CR, UnCRtainTS, DB-CR, EMRDM all clip here. Rationale documented in `preprocess.py`: uniform clipping cannot distort band ratios; it saturates only cloud/snow/specular pixels. **Sensitivity unmeasured**: real DN reaches ~28000 over bright clouds; the clip's effect on cloud-region metrics has not been quantified. |
| Model-space normalization | [0,1] → [-1,1] affine | [choice] bounded symmetric input for the Phase-2 diffusion; invertible; documented in `preprocess.py` |
| Metric-space | reflectance [0,1], PSNR data_range = 1 | [convention] matches UnCRtainTS/EMRDM metric code |

## 2. Sentinel-1 SAR (VV, VH; sigma0 dB)

| Constant | Value | Basis |
|---|---|---|
| VV clipping | [-25, 0] dB | **[ARBITRARY]** — specified in the Phase-1 project prompt without evidence. It *coincides* with the UnCRtainTS/EMRDM `default` convention; that is coincidence, not a decision. Sensitivity unmeasured. |
| VH clipping | [-25, 0] dB | **[ARBITRARY]** — same origin. Note the competing convention: DB-CR and the UnCRtainTS/DSen2-CR `resnet` path clip VH to **[-32.5, 0]**, and there is a physical argument for it (cross-pol backscatter runs ~7 dB below co-pol, so a -25 floor truncates more of VH's informative range). **Sensitivity measurement scheduled: re-evaluation arm B1** (`emrdm_reevaluation.md`) measures exactly how much this single constant moves SAM. |
| Normalization | clip → [-1, 1] affine | [choice] symmetric bounded, as S2 |

## 3. Cloud mask

| Constant | Value | Basis |
|---|---|---|
| Classes | 0 clear / 1 thin / 2 thick / 3 shadow | [choice] thin clouds retain ground signal, thick do not, shadows are radiometrically distinct — the Phase-2 mask-aware loss needs the distinction |
| Phase-1 implementation | visible-brightness thresholds (thin 0.35, thick 0.6, shadow 0.08) | **[ARBITRARY]** placeholder, explicitly declared "not scientifically adequate" in `cloud_mask.py`. Real evaluation requires s2cloudless (+ shadow detection); threshold values were eyeballed against dummy data only. |
| Real-data plan | s2cloudless probability, two thresholds for thin/thick; shadow via UnCRtainTS-style detection | [convention] s2cloudless is the community standard for S2 L1C; thresholds to be declared before first real-data run |

## 4. Splits

| Constant | Value | Basis |
|---|---|---|
| Split unit | ROI/scene (never patch) | [choice] patches within a scene overlap spatially; patch-level splits leak (documented in `splits.py`) |
| Real-data split | **the canonical UnCRtainTS scene split** (10 test scenes: spring 31/44/106/123/140, summer 73/119, fall 139, winter 63/108) | [convention] adopted verbatim for cross-paper comparability — our seeded ROI split (Phase 1) applies to dummy data and any data without a canonical split |
| Dummy-data split | seeded deterministic ROI shuffle, val/test ≥ 1 ROI | [choice] Phase-1 infrastructure |

## 5. Augmentation

| Constant | Value | Basis |
|---|---|---|
| Allowed | D4 (flip/rot90) only | [physical] reflectance and sigma0 are physical quantities; photometric augmentations corrupt the sensor model (documented in `transforms.py`) |
| Applied to | train split only | [convention] |

## 6. Metrics

| Constant | Value | Basis |
|---|---|---|
| Suite | PSNR, MAE, SAM, SSIM, LPIPS | [choice] pixel fidelity + spectral fidelity + structure + perceptual |
| Region split | full / cloud (mask≠0) / clear (mask=0), every metric | [choice] full-image averages hide clear-pixel copying; this is the project's core evaluation claim |
| SAM | 13-band vectors, float64 acos, degrees, per-pixel then batch nanmean | [choice] float64 because acos is ill-conditioned near 1 (float32 shows a ~0.01° noise floor on identical images) |
| SSIM | skimage, 7×7 window, data_range 1, per-pixel map averaged per region | [convention] skimage defaults; window bleeds ~3 px across region boundaries (acknowledged) |
| LPIPS | RGB = bands (B04,B03,B02) = indices (3,2,1); reflectance [0,1]→[-1,1] linear, no gain; backbone `alex` (eval) / `squeeze` (smoke) | [choice] documented in `metrics/suite.py`; no-gain mapping keeps it invertible and model-independent |
| Empty regions | NaN + nanmean, never silent zeros | [choice] |

## 7. Reproducibility & dependency isolation

| Rule | Basis |
|---|---|
| Global seed 42 (configurable), cudnn deterministic, seeded DataLoader workers | [choice] Phase-1 infrastructure |
| Every hyperparameter in Hydra configs; hardcoded values are review-rejectable | [choice] project rule |
| **External repos get isolated venvs.** EO-VAE pins `torchvision==0.16.2`; EMRDM depends on flash_attn/natten/pytorch-lightning. Neither may be installed into the main venv. Cross-repo data exchange happens through files (prediction tensors on disk), never through imports. | [choice] learned from the EO-VAE probe (`design_decisions.md` §6.1); made binding here |
| Failed experiments are logged to W&B, not deleted | [choice] project rule; failure records feed Phase-3 analysis |

## 8. Known unmeasured sensitivities (standing debt list)

1. S1 VV/VH clip ranges (§2) — B1 measures the VH axis; VV untouched.
2. S2 [0,10000] clip effect on cloud-region metrics (§1).
3. Threshold cloud-mask values (§3) — retired when s2cloudless lands.
4. LPIPS no-gain RGB mapping vs gained variants (§6).
5. SSIM window bleed at mask boundaries (§6).

Each item either gets measured when it becomes load-bearing, or stays on
this list. Removing an item without a measurement is not allowed.
