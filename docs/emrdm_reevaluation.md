# EMRDM Released-Weights Re-evaluation — 2-Arm Design

**Status: DESIGNED, blocked on data (see §7). Phase 1 extension** — this
deliverable precedes and does not depend on any model we train. Tolerances
in §3 are declared *before* any result exists and may not be revised after.

## 1. Question

EMRDM (CVPR 2025) reports SEN12MS-CR test metrics from its released
weights: **PSNR 32.14 / SSIM 0.924 / SAM 5.267° / MAE 0.018**. DB-CR
reports SAM 4.740° under conditions shown non-identical
(`design_decisions.md` §2.3). Two questions:

1. Do EMRDM's reported numbers reproduce from its own weights, code, and
   preprocessing? (Arm A — control)
2. How do those same weights score under Nadir's unified protocol, and
   which individual protocol factor moves the numbers how much? (Arm B)

## 2. Data

Canonical UnCRtainTS test split only (10 scenes; DB-CR reports 7,899
patches). Downloaded via `python scripts/download_data.py --split test`
(streams all 12 season archives, ~292 GB transfer, stores only test scenes,
~31 GB). The actual extracted patch count will be recorded here and checked
against 7,899; a mismatch is itself a finding about split ambiguity.

## 3. Arm A — control: EMRDM verbatim

Everything theirs: released SEN12MS-CR weights, their inference code, their
`default` preprocessing (S2 [0,10000]→[0,1]→[-1,1]; S1 VV & VH [-25,0]),
their hardcoded split, their `metrics.py`, NFE and sampler per their
released test config.

**Declared tolerances vs the paper table (fixed in advance):**

| Metric | Reported | Tolerance | Rationale |
|---|---|---|---|
| PSNR | 32.14 | ± 0.10 dB | table rounding (2 dp) + GPU nondeterminism |
| SSIM | 0.924 | ± 0.005 | table rounding (3 dp) |
| SAM | 5.267° | ± 0.05° | table rounding (3 dp) |
| MAE | 0.018 | ± 0.001 | table rounding (3 dp) |

Additionally, EMRDM's released artifacts include their own test log
(`test/sentinel/testtube/version_0/metrics.csv`, retrieved 2026-07-16) with
the measured values at full precision. **These are the primary Arm A
targets** (the paper table is these rounded to 2–3 decimals — consistency
verified):

| Metric | Their log value | Paper table |
|---|---|---|
| MAE | 0.018326831981539726 | 0.018 |
| PSNR | 32.13542556762695 | 32.14 |
| RMSE | 0.028028929606080055 | (not in table) |
| SAM | 5.266563415527344 | 5.267 |
| SSIM | 0.9244527220726013 | 0.924 |

Context columns in the same log: `raw_PSNR 18.14 / raw_SAM 13.37 /
raw_SSIM 0.659` — the cloudy-input-vs-target baseline, i.e. what "no model
at all" scores. Useful sanity anchor for our harness.

**Gate: if any Arm A metric lands outside tolerance, Arm B does not run.**
An out-of-tolerance Arm A means our harness (env, data extraction, weight
loading, or run configuration) is wrong — that gets debugged and documented;
EMRDM is not blamed and no Nadir-protocol numbers are produced.

## 4. Arm B — one factor at a time

Each sub-arm changes exactly one thing relative to Arm A; B4 combines them.
Predictions are produced ONCE per input-convention variant; metric-side
arms (B2, B3) re-consume stored predictions, they do not re-run inference.

| Arm | Change vs Arm A | Isolates |
|---|---|---|
| **B1** | S1 **VH clip -25 → -32.5 dB** at input (weights untouched; DB-CR's convention) | input-convention sensitivity of a trained model |
| **B2** | EMRDM predictions + **Nadir SAM implementation** (float64, same formula family) | metric-implementation delta (expected ≈ 0; nonzero would indicate implementation divergence) |
| **B3** | + **mask 3-split reporting** (full/cloud/clear) using s2cloudless-based masks | what full-image averages hide |
| **B4** | full Nadir protocol (the exact harness our Phase-2 model will face) | end-to-end protocol delta |

Dependency note for B3: requires the real cloud-mask implementation
(s2cloudless + shadow detection) — already a Phase-1 TODO
(`cloud_mask.py`); it lands before B3 runs, with thresholds declared in
`protocol.md` §3 beforehand.

## 5. Pre-registered hypothesis (H1)

> Changing only the SAR VH clipping from -25 to -32.5 dB (B1 vs Arm A)
> moves test-set SAM by **more than 0.527°** — the entire gap between
> DB-CR (4.740°) and EMRDM (5.267°).

- If **confirmed**: the two papers' SAM values are separated by less than
  one preprocessing constant's worth of sensitivity; cross-paper SAM
  ranking in this field is meaningless without protocol declarations. This
  is the project's first standalone finding.
- If **rejected**: the protocol is more robust to this constant than
  suspected — reported as-is, with the measured Δ. Either outcome is a
  result; the hypothesis is falsifiable and stays as written.

Secondary readouts (no pre-registered thresholds, exploratory): B1's effect
on PSNR/MAE/SSIM; B3's cloud-vs-clear SAM split (does the full-image SAM
flatter the model via clear pixels?).

## 6. Execution architecture (dependency isolation, binding per protocol.md §7)

```
[emrdm-venv]  EMRDM repo, its pinned deps (flash_attn/natten/lightning...)
    └─ runs inference → writes predictions to disk
         predictions/<patch_id>.npz  (uint16 DN, 13×256×256, + metadata json)
[main venv]   Nadir harness
    └─ reads predictions + ground truth → Arm A check (their metrics rerun
       for sanity) and Arm B metric arms
```

- No imports across environments; the file manifest is the only interface.
- EMRDM's own metric numbers for Arm A come from *their* code run in
  *their* env; Nadir's metric code never touches Arm A's pass/fail.
- Storage for predictions: 7,899 × ~1.7 MB ≈ 13 GB (uint16, before
  compression) — counted in the disk budget (§7).

Risk (Windows): flash_attn/natten have no reliable Windows builds. Plan:
run the EMRDM env under WSL2 + CUDA; if that fails, fall back to disabling
the attention kernels if the config allows, and if *that* changes outputs,
Arm A is invalid on this machine and needs a Linux box — recorded before
execution rather than discovered silently.

## 7. Execution environment: local WSL2 (decided 2026-07-16)

All earlier cloud-volume premises are discarded. Execution is local:
RTX 4080 16 GB, Windows 11 Home + WSL2 Ubuntu-24.04, single C: drive with
252 GB free at start. Full facts, vhdx cap (1 TB, measured), ext4-only data
rule, reclaim procedure, and the disk budget table live in
`protocol.md` §9. Running Arm A under WSL2 (Linux userland) also removes
the flash_attn/natten Windows-build risk from the reproduction verdict.

Prediction lifecycle (binding): predictions are a deletable intermediate.
Per scene: EMRDM venv infers → writes uint16 DN predictions → the metric
harness consumes them → per-patch metric records (JSON/CSV) are appended →
raster predictions for that scene are deleted before the next scene runs.
Peak prediction footprint stays ≤ ~4 GB. Only metric tables and a small
fixed visualization set (16 patches, chosen by seeded sample at the start)
survive. After each arm completes: `fstrim` + vhdx compact
(`protocol.md` §9.2) so the 60 GB-class intermediates never become resident
vhdx growth.

### 7.1 Environment build log (Step 1 — executed 2026-07-16)

**Outcome: SUCCESS with zero source compilation.** The feared
flash_attn/natten build risk did not materialize — every pinned package had
a prebuilt Linux wheel. Environment: WSL2 Ubuntu-24.04, venv at
`~/emrdm/venv` (Python 3.11.15 via uv), EMRDM repo at `~/emrdm/EMRDM`.
Footprint: venv 7.2 GB + repo 33 MB; vhdx grew 2.3 → 9.6 GB.

| # | Attempt | Result |
|---|---|---|
| 1 | `nvidia-smi` inside WSL2 | ✓ RTX 4080 visible, driver 591.86 (CUDA 13.1 capable — cu121 wheels bundle their own runtime, backward-compatible) |
| 2 | Inline `wsl -- bash -lc "..."` setup one-liner | **FAILED** — PowerShell string interpolation mangled `$HOME`/semicolons. Lesson: all WSL provisioning goes through `.sh` script files, never inline quoting |
| 3 | `torch==2.2.1+cu121` + `torchvision==0.17.1+cu121` (pytorch cu121 index) | ✓ `TORCH_OK 2.2.1+cu121 True NVIDIA GeForce RTX 4080`; NumPy-2 ABI warning observed → pin `numpy==1.26.4` next |
| 4 | `flash_attn==2.5.9.post1` prebuilt wheel `cu122torch2.2cxx11abiFALSE-cp311` from the official GitHub release (cu122 kernels on cu121 torch runtime: both CUDA 12.x, compatible; torch 2.2 manylinux wheels are pre-cxx11 ABI) | ✓ `FLASH_ATTN_OK 2.5.9.post1` |
| 5 | `natten==0.17.1+torch220cu121` from `shi-labs.com/natten/wheels` | **FAILED** — the wheel host's TLS certificate expired 2025-12-03 (verified in the error: cert not valid after UNIX 1764806399). Did NOT bypass TLS |
| 6 | Same wheel from the official SHI-Labs/NATTEN GitHub release v0.17.1 | ✓ `NATTEN_OK 0.17.1` (451 MB wheel) |
| 7 | `pytorch-lightning==2.3.3` import | **FAILED** — `pkg_resources` missing: setuptools ≥ 81 removed it (2026 reality); lightning 2.3.3 still imports it | 
| 8 | `setuptools<81` | ✓ `LIGHTNING_OK 2.3.3` (deprecation warning acknowledged) |
| 9 | `import sgm` iterative dependency discovery (EMRDM's own pins from its requirements.txt) | ✓ after installing `dctorch==0.1.2`, `pandas==2.2.3`, `opencv-python-headless==4.10.0.84`, `matplotlib==3.9.2` → `SGM_OK` |
| 10 | `ResidualDiffusionEngine` init (Step 2b) | needed `lpips==0.1.4` (their metrics module) → engine loads |

Not installed (deliberately): the full requirements.txt (README warns
against it), realesrgan/sdata editable extras, anything training-only.
Missing pieces, if any, surface at Step 2 (weight load + smoke inference)
and get appended to this table.

### 7.2 Kernel numerical validation (Step 2a — executed 2026-07-16, RTX 4080)

Motivation: the wheels cross version tags (flash_attn built for cu122 on a
cu121 torch; natten tagged torch220 on torch 2.2.1). Import success does not
rule out ABI/kernel corruption, which can silently bias numbers inside the
SAM ±0.05° gate. Method: flash-attn's own test criterion — kernel error vs
a float32 naive reference must be ≤ 2× the naive-in-dtype error.

**flash_attn 2.5.9.post1 — PASS, 12/12 configs.** Max-abs-error vs fp32
reference (ratio = kernel_err / naive_err; criterion ratio ≤ 2):

| dtype | (B,S,H,D) | causal | kernel_err | naive_err | ratio |
|---|---|---|---|---|---|
| fp16 | 2,1024,8,64 | no / yes | 1.32e-4 / 8.96e-4 | 5.11e-4 / 1.27e-3 | 0.26 / 0.71 |
| fp16 | 1,4096,8,64 | no / yes | 6.45e-5 / 1.00e-3 | 1.94e-4 / 1.23e-3 | 0.33 / 0.82 |
| fp16 | 2,512,4,128 | no / yes | 1.80e-4 / 7.91e-4 | 7.55e-4 / 2.34e-3 | 0.24 / 0.34 |
| bf16 | 2,1024,8,64 | no / yes | 9.90e-4 / 7.57e-3 | 3.85e-3 / 1.29e-2 | 0.26 / 0.59 |
| bf16 | 1,4096,8,64 | no / yes | 5.33e-4 / 8.28e-3 | 2.49e-3 / 8.71e-3 | 0.21 / 0.95 |
| bf16 | 2,512,4,128 | no / yes | 1.90e-3 / 6.53e-3 | 7.15e-3 / 1.64e-2 | 0.27 / 0.40 |

The kernel is consistently *more* accurate than naive PyTorch (ratio < 1
everywhere): no ABI anomaly.

**natten 0.17.1 — PASS after root-causing an fp32 deviation.**
fp16/bf16 vs fp32 reference: kernel_err 1.20e-3 / 6.13e-3, ratios 1.23 /
1.00 — PASS. The initial fp32 "tight" check failed (2.77e-3 vs a 1e-4
limit); investigation against a float64 reference:

| natten fp32 config | max err vs fp64 ref |
|---|---|
| defaults (GEMM-NA on, TF32 on) | 2.774e-3 |
| TF32 off | 8.245e-4 |
| GEMM-NA disabled (naive CUDA path) | **2.906e-7** |
| naive PyTorch fp32 (reference sanity) | 2.174e-7 |

Errors were spatially uniform (interior 1.97e-3 / border 2.77e-3), ruling
out window-semantics mismatch. Conclusion: the deviation is natten's
documented default tensor-core GEMM-NA path (TF32 + reduced-precision
accumulation), not corruption — with that path disabled the kernel agrees
with the reference at the fp32 noise floor, which simultaneously validates
our reference's border semantics. **Arm A decision:** run EMRDM with natten
defaults, because upstream defaults are what produced the paper's numbers;
`natten.disable_tf32()` / `disable_gemm_na()` remain available if a
strict-fp32 arm is ever needed.

**Verdict: Step 2a gate PASSED — wheel combination approved for Arm A.**

### 7.3 Weights + dummy smoke (Step 2b — executed 2026-07-16)

**Provenance.** Downloaded from the official EMRDM release Drive folder
(README → `drive.google.com/drive/folders/1T3OwRNP5r5qVLQZujnl2WDBVXHC1Am65`,
file `train/sentinel/checkpoints/last.ckpt`, Drive id
`1bPU1HzxRQmMXsrWFg2Z8W1bBGdhrlWeD`, Drive mtime 2025-03-28). No official
checksum is published; **our received hash is the record**:

```
sha256(last.ckpt) = edf7b5d1ef35aea27f0e33c7ff048dc6d053c0382615f9513dfe3ebb9091ecda
size = 626,349,094 bytes
```

Also retrieved (same folder, hashes in `~/emrdm/artifacts/sentinel/SHA256SUMS`):
their exact test config (`test/sentinel/configs/2025-03-11T11-11-46-project.yaml`),
test metrics log (§3), hparams, and the training config. Extra dep
discovered at engine init: `lpips==0.1.4` (their metrics module imports it)
— appended to the §7.1 environment.

**Smoke result (synthetic batch, zero data, quality deliberately not
assessed — 8.6σ lesson from the EO-VAE probe):**

```
Restored from artifacts/sentinel/last.ckpt with 0 missing and 0 unexpected keys
ENGINE_LOADED params=39.1M sampler=ResidualEulerEDMSampler num_steps=5
OUTPUT shape=(1, 13, 256, 256) dtype=torch.float32 range=(-2.203,3.210) finite=True
SMOKE_OK
```

**Inference recipe extracted from code (for Arm A):**
- Entrypoints: `python main.py --base configs/example_training/sentinel.yaml
  --enable_tf32 -t false` (test metrics) and `... --no-test true --predict
  true` (writes per-patch .tif predictions + metrics.csv; asserts batch=1).
- Engine `ResidualDiffusionEngine`: `input_key: target`, **`mean_key: S2`**
  — the cloudy image is the mean-reverting anchor. `use_ema: true`
  (predict runs under EMA weights; ckpt carries 121 EMA tensors).
  Notably `use_flash_attn2: false` in the sentinel config — flash_attn is
  not exercised by this model; the §7.2 validation stands as environment
  insurance.
- Conditioning: `IndentityEmbedder` on batch key `S1S2` (2ch SAR + 13ch
  cloudy, all in [-1,1]) → `CloudRemovalWrapper` concats onto the noisy
  target → denoiser `ImageTransformerDenoiserModelInterface` (k-diffusion
  image transformer; 28→13 ch; neighborhood attention k=7 at the two
  high-res levels — natten — and global attention below).
- Sampler: `ResidualEulerEDMSampler`, **num_steps=5**, EDM
  preconditioning (`ResidualEDMScaling`, sigma_input=sigma_mu=1.0).
- Preprocessing entrypoint: `SEN12MSCRInterface` (rescale=True → all
  tensors mapped [0,1]→[-1,1]; `rescale_method='default'`; test-time
  `cloud_masks: None`).

## 8. Results

*(empty — to be filled by execution, tolerances above may not be edited)*
