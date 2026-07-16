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

### 4.1 Phase-2 real-data scene selection — FIXED (single-transfer commitment)

The TUM server distributes only whole-season archives; every scene choice
below is extracted during **one** 292 GB streaming pass. **Changing this
list later means re-transferring 292 GB.** Fixed on 2026-07-16.

Selection rule [choice]: from the canonical UnCRtainTS *train* scene lists
(155 scenes), 7 scenes per season (season-balanced 25/25/25/25), drawn with
`random.Random(42).sample(sorted(scenes), 7)` per season — seeded sampling
rather than first-N to avoid any ordering/geography correlation in scene
ids. Validation: 1 scene per season from the canonical *val* list, same
rule. Geographic separation from the test ROIs is inherited from the
canonical split (train/val/test scene sets are disjoint distinct ROIs).

| Split | Scenes | Expected patches* |
|---|---|---|
| train subset | spring 6, 15, 39, 97, 101, 119, 147 · summer 7, 36, 40, 72, 76, 87, 143 · fall 3, 22, 35, 37, 40, 119, 134 · winter 8, 25, 42, 59, 61, 104, 146 (28 scenes) | ~19.5k (28 × ~698 avg) |
| val subset | spring 17 · summer 17 · fall 65 · winter 22 (4 scenes) | ~2.8k |
| test | canonical 10 scenes (§4) | 7,899 per DB-CR |

*Patch counts per scene are unknown before extraction (dataset average
698 = 122,218/175); actual counts get recorded in the download manifest.
The ~20k target is met in expectation, not guaranteed per-scene.

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

## 9. Infrastructure (local — supersedes all earlier cloud premises)

Execution environment, fixed 2026-07-16: **local machine only.**

| Fact | Value (measured, not assumed) |
|---|---|
| GPU | RTX 4080 16 GB (Ada; bf16 native) |
| Host | Windows 11 Home 26200, single C: drive, 953 GB total |
| WSL | 2.3.26.0, kernel 5.15.167.4-1, distro Ubuntu-24.04 |
| vhdx path | `C:\Users\kimma\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu24.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx` |
| vhdx current size | 2.3 GB (fresh distro) |
| **vhdx max cap** | **1 TB** (measured: ext4 root fs = 1007 GB) — the old 256 GB WSL default does NOT apply here; no expansion needed |

Rules:

1. **All dataset/prediction IO happens inside ext4** (e.g.
   `~/data/sen12mscr`). `/mnt/c/` is 9p-mounted and is a dataloader
   bottleneck; placing data there is forbidden. The Windows-side repo can
   stay on C:; only bulk data must be ext4-resident.
2. **vhdx never shrinks by itself.** Deleting 60 GB inside ext4 returns
   nothing to C: until compacted. Reclaim procedure (Windows 11 Home — no
   Hyper-V `Optimize-VHD`):
   ```
   (inside WSL)   sudo fstrim -a
   (Windows)      wsl --shutdown
                  diskpart
                    select vdisk file="<ext4.vhdx path above>"
                    attach vdisk readonly
                    compact vdisk
                    detach vdisk
   ```
   Alternative: sparse vhdx (`wsl --manage Ubuntu-24.04 --set-sparse true`)
   reclaims automatically; requires `wsl --shutdown` first (attempted
   2026-07-16, blocked by running docker-desktop — pending user approval).
   Every bulk-deletion step in the pipelines below ends with a reclaim.
3. **C: usage must stay under 90 % (≤ 858 GB used / ≥ 95 GB free).**

Disk budget (C: at 700 GB used / 252 GB free before this project's data):

| Item (ext4-resident) | Size est. |
|---|---|
| WSL base + EMRDM env (torch, flash-attn, natten; incl. pip caches) | ~17 GB |
| EMRDM repo + SEN12MS-CR weights | ~3 GB |
| Test split rasters (10 scenes, 7,899 patches) | ~31 GB |
| Train subset rasters (28 scenes, ~19.5k patches) | ~78 GB |
| Val subset rasters (4 scenes, ~2.8k patches) | ~11 GB |
| Prediction chunk (per-scene generate→eval→delete, peak) | ~4 GB |
| Logs/manifests/headroom | ~5 GB |
| **Peak total (end of full streaming pass + Arm A inference)** | **~149 GB** |

Peak C: usage = 700 + 149 ≈ **849 GB = 89.1 %** — under the 90 % line with
~9 GB margin. This margin is thin; mitigations if Windows-side usage grows:
purge pip caches (−4 GB), enable sparse vhdx, or trim the val subset. The
peak occurs at the end of the single streaming pass (all splits extracted)
during Arm A inference; after Arm A, predictions are deleted and the vhdx
compacted.
