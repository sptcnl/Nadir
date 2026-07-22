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

The TUM server distributes only whole-season archives. **Changing the scene
lists below means re-transferring 292 GB.** Fixed on 2026-07-16.

**Execution status (updated 2026-07-17):** the Step-4 streaming pass
extracts the **test split only** (~31 GB). The train/val subset is
deliberately NOT extracted in the same pass. The reason is **decision
deferral, not disk** (the earlier disk-constraint framing is superseded):
no training data is acquired before (a) the Arm A reproduction gate has a
verdict and (b) the go/no-go on training our own model is made. This
knowingly accepts a possible second 292 GB pass later; the scene selection
itself stays fixed as declared below, so the deferral changes *when* the
transfer happens, never *what* is selected.

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
| val | **canonical UnCRtainTS val, ALL 10 scenes**: spring 17 · summer 17, 19, 80, 127 · fall 65 · winter 22, 84, 107, 130 | 7,176 per DB-CR |
| test | canonical 10 scenes (§4) | 7,899 per DB-CR |

*Patch counts per scene are unknown before extraction (dataset average
698 = 122,218/175); actual counts get recorded in the download manifest.
The ~20k train target is met in expectation, not guaranteed per-scene.

**Realized test extraction (2026-07-20):** 9 of 10 test scenes complete =
**7,116 triplets**; summer scene **73 = 783 clear patches lost to a
server-side 2 MB corruption** in `ROIs1868_summer_s2.tar.gz`
(`emrdm_reevaluation.md` §2.1/§2.2.1 — unrecoverable, no mirror). Per-scene
complete counts: spring 106=784, 123=781, 140=850, 31=784, 44=784; summer
119=782; fall 139=784; winter 63=784, 108=783. **7,116 + 783 = 7,899**
matches DB-CR's stated test count, confirming the canonical split. All
re-evaluation numbers are reported over the exact realized count (7,116),
never rounded up to 7,899.

### 4.2 DSen2-CR baseline training subset — FIXED [choice] (2026-07-20)

Purpose is explicitly **"prove the model can be trained"** (loss converges,
pipeline runs, baseline metrics emerge on a 16 GB GPU within a day) — **not**
full training or SOTA reproduction. A small subset therefore suffices;
over-fitting is acceptable and will be reported as such.

Selection: **3 scenes, one per season, spring/fall/winter** (summer dropped
— see below), from the §4.1 pre-registered train pool:

| Season | Scene | Expected patches |
|---|---|---|
| spring | 6 | ~780 |
| fall | 3 | ~780 |
| winter | 8 | ~780 |
| **total** | 3 scenes | **~2,300** |

**Geographic disjointness (verified 2026-07-20, logged):** none of {spring 6,
fall 3, winter 8} shares a (season, scene) ROI with the test 9-scene set
{spring 31/44/106/123/140, summer 119, fall 139, winter 63/108} → no
train/test leakage; baseline metrics are valid on that axis.

**Summer dropped — reason recorded.** The pre-registered summer train scenes
{7, 36, 40, 72, 76, 87, 143} are **all stored after the 15.84 GB corruption**
in `ROIs1868_summer_s2.tar.gz` (verified 2026-07-20 by decoding the readable
prefix locally: 0/7 candidates present; §2.1 of `emrdm_reevaluation.md`).
Rather than step on a known-corrupt archive, summer is excluded; the summer
archive is not touched at all (corruption risk = 0). A 3-season subset is
sufficient for the pipeline-proof goal; seasonal completeness is not required
here. If a fuller train set is ever needed, a pre-gap summer scene (e.g. one
of {115, 121, 132, 133}, which decode cleanly from the prefix) can be added —
but that is a new declared selection, not taken opportunistically now.

**Realized training set = spring-6 ONLY (further reduced, 2026-07-21).** After
spring (all 3 modalities) and fall (s1 + clear) downloaded, the TUM server
throughput collapsed to ~0.5 MB/s (10× slower; throttling after sustained
transfer — NOT corruption: range probes at 16/17/20 GB of fall_s2_cloudy all
returned 206 with data). At that rate the remaining fall_s2_cloudy + winter
(~58 GB) would take ~30 h, breaking the one-day goal. Since **spring-6 alone
(700 complete triplets) already suffices to prove the pipeline trains**, fall
(incomplete: no cloudy) and winter (not started) were set aside — resumable
later via the fixed downloader (`--reclaim-pause`, HTTP-200 truncation fix)
when the server recovers. Training therefore ran on **spring-6, 700 triplets,
patch-split 630 train / 70 val** (val leaks spatially — convergence monitor
only; headline eval is the geographically-disjoint 9-scene test).

### 4.3 DSen2-CR baseline — training result (2026-07-22)

Ran on spring-6 (700 triplets, 630/70 patch-split), 60 epochs, bf16 +
gradient checkpointing, CARL loss, W&B offline. **Goal was pipeline-proof +
convergence, NOT SOTA** — read every number below in that light.

**Convergence (val = spring-6 patch holdout; leaks spatially → optimistic):**
SAM 8.75→7.53 (epoch 9) and PSNR 28.58→29.33 by epoch ~9, SSIM 0.77→0.87 by
epoch 59. The pipeline trains and the model learns; SAM/PSNR then drift back
(epoch 59 SAM 8.56) — overfitting on 700 patches, as expected/accepted.

**Held-out evaluation (9-scene test, `eval_per_season.py`, final epoch-59
checkpoint), model vs the do-nothing baseline (pred := cloudy input):**

| Season (n) | PSNR model/none | SAM model/none | SSIM model/none | MAE model/none |
|---|---|---|---|---|
| spring (3983, in-domain) | 20.71 / 18.72 | 15.62 / 13.71 | 0.647 / 0.654 | 0.080 / 0.123 |
| summer (782, OOD) | 19.55 / 17.39 | 16.67 / 12.54 | 0.557 / 0.619 | 0.074 / 0.098 |
| fall (784, OOD) | 19.85 / 17.88 | 16.45 / 13.21 | 0.550 / 0.588 | 0.074 / 0.095 |
| winter (1567, OOD) | 20.07 / 18.28 | 16.39 / 14.55 | 0.620 / 0.610 | 0.085 / 0.120 |
| **ALL (7116)** | **20.35 / 18.38** | **16.00 / 13.71** | **0.620 / 0.633** | **0.080 / 0.117** |

**Honest reading (this is a capability proof, not a competitive model):**
1. **Pipeline + convergence: demonstrated.** End-to-end training on real data,
   metrics improve from epoch 0, checkpoints + eval run.
2. **The model beats do-nothing on PSNR (+1.96 dB) and MAE, but is WORSE on
   SAM (16.0° vs 13.7°)** — it lowers per-pixel brightness error while
   *distorting inter-band ratios more than the cloudy input does*. This is a
   concrete, in-house demonstration of the project's SAM-first thesis: PSNR
   alone calls this a success; SAM reveals spectral corruption. (Likely
   causes: the placeholder brightness cloud-mask mis-weights CARL, and hard
   overfitting to one scene — not a claim about DSen2-CR proper.)
3. **In-domain > out-of-domain** on PSNR/SAM/SSIM (spring beats summer/fall/
   winter) — the single-season generalization limit is visible; the per-season
   split was necessary to see it (pooled numbers would have hidden it).
4. **Severe overfitting:** leaky-val PSNR ~29 vs held-out test PSNR ~20.7 (the
   held-out spring scenes are different ROIs than spring-6).
5. **Far from SOTA** (SAM 16° vs EMRDM 5.27°) — expected for one scene / 60
   epochs / overfit; performance was never the goal.
   Cross-check: the do-nothing SAM (13.71°, our harness) ≈ EMRDM's logged
   `raw_SAM` 13.37° — re-validates the harness once more.

Not cherry-picked: the final (epoch-59) checkpoint is reported; earlier
checkpoints (val peaked ~epoch 9) exist if better generalization is wanted.

**Rejected alternative (recorded 2026-07-16):** a val reduction to 1 scene
per season (~11 GB instead of ~27 GB) was proposed to protect the C:-drive
90% budget line. **Rejected on protocol-integrity grounds:** letting a disk
constraint alter the experimental design creates exactly the kind of
undeclared canonical-split deviation this project audits DB-CR/EMRDM for
(§5.1 of `design_decisions.md`). The disk constraint was resolved on the
infrastructure side instead (dedicated SSD, §9). The train *subset* remains
a subset — that is a declared, seeded selection for compute budget, not a
silent split modification; val and test stay canonical and complete.

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
| SSIM | **[choice] skimage `structural_similarity`, uniform 7×7 window, data_range 1, `full=True` per-pixel map averaged per region.** Rationale: (1) the mask-split protocol requires a dense SSIM *map*, which skimage provides natively; (2) skimage is an independently maintained reference implementation with every parameter explicit. Known divergence, **measured 2026-07-17**: vs the field's dominant convention in cloud removal (Po-Hsun-Su `pytorch_ssim`, gaussian 11×11 σ1.5 — used by UnCRtainTS, DB-CR, EMRDM), per-patch SSIM differs by up to **0.0596** on real SEN12MS-CR data — larger than published inter-method SSIM margins (~0.02). Consequence: whenever Nadir SSIM is placed next to published tables, we dual-report: our convention AND a `pytorch_ssim`-equivalent (skimage with `gaussian_weights=True, sigma=1.5, use_sample_covariance=False`), labeled explicitly. No paper we audited declares its SSIM convention in the text; we do. |
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
   Primary mechanism: **sparse vhdx — ENABLED 2026-07-16** (approved;
   executed at vhdx = 2.3 GB, the cheapest possible moment). Verified by
   measurement: `Get-Item ext4.vhdx` attributes now include `SparseFile`,
   distro boots normally. With sparse on, freed ext4 blocks return to
   Windows automatically; the diskpart procedure above stays as fallback
   if reclaim ever lags. Every bulk-deletion step in the pipelines below
   still ends with `fstrim`.
3. **C: usage must stay under 90 % (≤ 858 GB used / ≥ 95 GB free).**

### 9.1 Disk budget (rev 3 — C:-resident, SSD relocation ON HOLD)

Premise change (2026-07-17): disk resolved on C: (263 GB free measured);
the SSD relocation plan in §9.2 is **on hold in its entirety** — nothing in
it executes until reactivated. Everything runs in the existing sparse vhdx
on C:. With the test-only Step 4 (see §4.1 deferral), the budget is:

| Item (ext4-resident) | Size est. |
|---|---|
| EMRDM env + repo + weights (measured) | ~10 GB |
| Test split rasters (7,899 patches) | ~31 GB |
| Prediction chunk peak + logs/manifest | ~5 GB |
| **vhdx total** | **~46 GB** |

Peak C: usage ≈ 736 GB ≈ **77 %** — comfortable. If model training is
approved later, the train/val extraction (~105 GB) still fits C: at ~88 %,
but the budget will be re-derived then.

### 9.1-legacy (superseded rev 2 — SSD-phased plan, kept for the record)

Premise (2026-07-16, now on hold): a **1 TB NVMe SSD (Patriot P300)** is
added, arriving before Step 4 of the re-evaluation plan. Steps 1–3
(environment, smoke, single-scene pipeline check) run on the existing C:
drive; the full streaming pass (Step 4) and everything after runs on the
SSD.

**Phase 1 — Steps 1–3 on C:** (700 GB used / 252 GB free at start):

| Item (ext4-resident) | Size est. |
|---|---|
| EMRDM env (torch, flash-attn, natten; incl. caches) | ~17 GB |
| EMRDM repo + SEN12MS-CR weights | ~3 GB |
| One test scene (pipeline check) | ~3 GB |
| Headroom/logs | ~2 GB |
| **vhdx total** | **~25 GB** |

Peak C: usage ≈ 725 GB = **76 %** — comfortable.

**Phase 2 — Step 4+ on the SSD** (WSL storage relocated, §9.2):

| Item | Size est. |
|---|---|
| Relocated distro (env + weights) | ~25 GB |
| Test rasters (7,899 patches) | ~31 GB |
| Val rasters (canonical 10 scenes, 7,176 patches) | ~27 GB |
| Train subset rasters (~19.5k patches) | ~78 GB |
| Prediction chunk peak + logs/manifests | ~9 GB |
| **SSD total** | **~170 GB of 1 TB = 17 %** |

C: returns to ~700 GB used once the old vhdx is unregistered. No budget
line is anywhere near 90 % in either phase.

### 9.2 SSD attachment: options compared — **ON HOLD (2026-07-17)**, option (b) was approved with three conditions (re-measure sparse after import; set default user in /etc/wsl.conf; unregister old distro only after full verification) and executes only if reactivated

`/mnt/d/`-style NTFS drvfs mounts are ruled out (9p, violates the
ext4-only rule). Two compliant designs:

**(a) `wsl --mount \\.\PHYSICALDRIVE<n> --bare` + mkfs.ext4 — raw disk passthrough**
- Procedure: attach the physical disk to WSL (`--bare`), `mkfs.ext4` it
  inside the distro, mount at `~/data`. Requires admin; the disk becomes
  invisible to Windows while attached.
- Performance: best possible (raw NVMe under native ext4, no vhdx layer).
- Risks: **the mount does not survive reboots or `wsl --shutdown`** — it
  must be re-attached each time (automatable via Scheduled Task, but a
  missed remount mid-Step-4 aborts the streaming pass exactly when it
  hurts); whole disk is consumed (no NTFS partition possible alongside);
  `wsl --mount` failures are opaque on some USB/NVMe bridges.
- Rollback: detach, re-initialize the disk NTFS in Windows Disk Management.
- Existing 2.3→25 GB vhdx: stays on C: as the distro root (only data moves).

**(b) `wsl --export` / `--import` — relocate the whole distro to the SSD** ← **recommended**
- Procedure: `wsl --shutdown` → `wsl --export Ubuntu-24.04 D:\wsl\ubuntu2404.tar`
  → `wsl --import Ubuntu-24.04-nvme D:\wsl\Ubuntu-24.04 D:\wsl\ubuntu2404.tar`
  → set default user in `/etc/wsl.conf` (imports default to root) →
  re-enable sparse on the new vhdx → verify boot + `nvidia-smi` → after
  verification, `wsl --unregister Ubuntu-24.04` frees C:.
- Performance: ext4-in-vhdx on NTFS — the same storage path the distro
  uses today on C:, i.e. known-good for dataloaders; a few percent vhdx
  overhead vs (a).
- Risks: minor and front-loaded (import defaults to root; Store-app
  shortcut points at the old distro) — all detectable before any data
  lands. **Survives reboots and `wsl --shutdown` with zero intervention**,
  which matters for a multi-day 292 GB streaming pass on a machine that
  takes Windows Updates.
- Rollback: export/import back to C: (or anywhere); the .tar is itself a
  backup artifact.
- Existing vhdx: consumed by the export, then unregistered — nothing
  stranded on C:.

Recommendation: **(b)**. The raw-IO edge of (a) is not the bottleneck
(the dataloader reads ~4 MB patches; the TUM server and the GPU are the
bottlenecks), while (a)'s remount-on-every-boot failure mode directly
threatens the one-shot streaming pass. Execute (b) when the SSD arrives,
before Step 4.
