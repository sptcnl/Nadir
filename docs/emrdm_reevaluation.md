# EMRDM Released-Weights Re-evaluation — 2-Arm Design

**Status: Arm A COMPLETE on the 9-scene set (2026-07-20); both harness gates
pass; Arm B (H1) open. Phase 1 extension** — this deliverable precedes and
does not depend on any model we train. Tolerances in §3 are declared *before*
any result exists and may not be revised after. Key outcomes: summer-73
unrecoverable (§2.1/§2.2.1 — mirror corruption), so Arm A runs on 7,116
patches (9 scenes), full-7,899 paper reproduction deferred; harness validated
via Gate 0 (determinism, §2.3.2) + Gate 1 (internal consistency, §2.3.3).

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

### 2.1 Data-acquisition finding: `ROIs1868_summer_s2.tar.gz` is corrupt on the TUM mirror (2026-07-19)

11 of the 12 season archives streamed/downloaded cleanly. The 12th,
`ROIs1868_summer_s2.tar.gz` (the summer **clear** optical, 40,141,465,440 B),
has a **stable ~2 MB unreadable region at byte offset 16,609,967,160
(~15.84 GB, 41 % into the archive)** on the only distribution source
(dataserv.ub.tum.de — HTTP, FTP, and rsync all read the same backend; no
HuggingFace/Zenodo/Kaggle mirror of the raw season archives exists).

Evidence (all measured, scripts in scratch):
- Every transport — HTTP streaming, HTTP `download` with Range resume, and
  rsync (`--append`, `--inplace` delta) — halts at exactly 16,609,967,160 B.
- Range probes: bounded windows at OFF−32 MB and OFF+2 MB … OFF+32 MB return
  `206` with full payload; the band **[OFF, OFF+2 MB)** returns `303`→timeout.
- Fine map (1 MB windows): dead = exactly [OFF, OFF+2 MB); readable from
  OFF+2 MB onward.
- Hammer test (256 KB windows × 8 retries = 64 attempts across the 2 MB):
  **0/64 recovered** → stable server-side corruption, not transient.

Because gzip is a single continuous stream, this 2 MB gap makes everything
after 15.84 GB undecodable by a normal `gunzip`. Decoding the readable
prefix `[0, OFF)` recovers the tar members stored in the first 41 %:
- summer test scene **119: clear 782/782 patches recovered** ✓
- summer test scene **73: 0 patches** (its members are stored past the gap).

**Impact:** the summer-73 clear targets (783 patches, ~10 % of the 7,899-patch
test set) are the only missing data; the other 9 test scenes are complete.

**This is itself a §5.1-class reproducibility finding:** the distribution
infrastructure of a widely-cited public benchmark is itself a reproducibility
variable — a bit-rotted archive on the sole mirror silently blocks exact
reproduction of the published 7,899-patch number. Recorded, not smoothed over.

**Recovery in progress (chosen path):** download the servable tail
`[OFF+2 MB, END)` in bounded windows, zero-fill the 2 MB gap, and run
`gzrecover` (gzip recovery toolkit — finds deflate block boundaries after a
corrupt region) to recover scene-73 members past the gap. **Recovery is
gated, not trusted** (§2.2): recovered summer-73 patches are validated
(count vs 783, 13 bands, uint16, value range); patches straddling the gap
that cannot be recovered are identified, dropped, and counted. The Arm A
gate is then built on `7899 − N` surviving patches with `N` stated exactly —
never rounded back to 7,899. Fallback if recovery/validation fails: §2.3
(9-scene internal-consistency validation).

### 2.2 Scene-73 recovery pipeline and validation gate

Steps (scripts in `scripts/reeval/`):
1. **Tail download** — fetch `[OFF+2 MB, 40,141,465,440)` (~23.5 GB) in
   bounded 256 MB range windows (bounded windows past the gap serve
   reliably; open-ended resume ranges are what wedge the gateway).
   Resumable, per-window retry.
2. **Assemble** — `prefix[0,OFF)` + `2 MB zeros` + `tail` → a full-length
   archive with a zeroed 2 MB gap.
3. **`gzrecover`** the assembled archive → a decompressed byte stream that
   resyncs to deflate block boundaries after the zeroed gap.
4. **Extract** `ROIs1868_summer_s2/s2_73/*` from the recovered stream
   (`tar --ignore-zeros`).
5. **Validation gate (nothing trusted without it):**
   - recovered scene-73 clear count vs the S1 reference (783);
   - every recovered patch: 13 bands, uint16, 256×256, plausible reflectance
     range (reuse `verify_extraction.py` semantics);
   - **cross-check against the cloudy pair**: each recovered clear patch must
     have its `s2_cloudy_73` and `s1_73` partners (already on disk) — orphans
     are dropped;
   - patches that gzrecover cannot reconstruct (those straddling the gap) are
     listed by patch id and **dropped**, and the count `N` is recorded here
     and in `protocol.md`.
6. **Gate assembly** — Arm A runs on `7899 − N` patches; the reproduced
   aggregate is compared to a re-derivation of EMRDM's expected value on the
   **same** surviving-patch set (their inference over the identical set, not
   the published 7,899-average), so the tolerance comparison stays valid on a
   reduced set. `N` and the exact patch count are stated in every reported
   number.

### 2.2.1 Recovery OUTCOME (2026-07-20): FAILED — scene 73 unrecoverable

The tail `[OFF+2 MB, END)` downloaded cleanly (23,529,401,128 B, exact).
`gzrecover` was run to completion on the assembled `prefix + 2 MB zeros +
tail`. Result: the recovered tar tops out at **24 GB — exactly the
pre-gap content** (decompressed prefix; contains summer scenes up to the
gap incl. s2_119 and s2_27), and **s2_73 is absent** from the full output.
gzrecover decoded the pre-gap stream but **could not resync DEFLATE after
the 2 MB gap** — deflate is stateful (32 KB sliding window + Huffman
tables), and the lost 2 MB destroys the state needed to decode any
subsequent block; a block-boundary search cannot reconstruct it. This is a
fundamental property, not a tuning problem: scene-73's clear targets are
**unrecoverable from this mirror**. (Two premature kills on misread progress
signals cost time but did not change the outcome — the completed run
confirms 24 GB/pre-gap-only.)

Confirmed data state (2026-07-20): **9 of 10 test scenes complete = 7,116
triplets** (spring 31/44/106/123/140, summer 119, fall 139, winter 63/108);
summer **73 = 783 patches missing** (clear=0; s1 & cloudy present).
**7,116 + 783 = 7,899** — exactly DB-CR's stated test count, which
independently confirms the split is the canonical UnCRtainTS one.
Recovery intermediates (tail/assembled/recovered tar, ~88 GB) reclaimed.

→ Proceeding with the §2.3 fallback.

### 2.3 Fallback: 9-scene internal-consistency validation (ACTIVE)

If recovery or its validation gate fails, Arm A cannot reproduce the
published 7,899-patch number at all. The fallback re-scopes the deliverable
honestly: run **both** our harness and EMRDM's own code over the 9 complete
scenes (7,116 patches) and show they agree to the pre-registered tolerances.
The claim becomes **"our harness == EMRDM's code"** (internal consistency),
NOT "we reproduced the paper's 5.267 over the full set" — the latter is
explicitly deferred and recorded as blocked by §2.1 corruption. This still
validates the harness at scale, which is all Arm B needs: H1 (B1) is a
*delta* measurement (VH −25 vs −32.5), valid on any fixed patch set.

### 2.3.1 Arm A result on the 9-scene set (EMRDM code, 2026-07-20)

EMRDM's released weights through **their own test loop** (`main.py --base
<sentinel-derived> --enable_tf32 -t false`, single GPU, NFE=5, EMA), on the
7,116-patch set (their loader auto-excluded the incomplete summer-73 triplet
— dataloader built exactly 7,116, matching our count):

| Metric | 9-scene (7,116) | Paper (7,899) | Δ (9-scene − paper) |
|---|---|---|---|
| PSNR | 31.4795 | 32.1354 | **−0.656 dB** |
| SAM | 5.6378° | 5.2666° | **+0.371°** |
| SSIM | 0.91916 | 0.92445 | **−0.0053** |
| MAE | 0.019385 | 0.018327 | **+0.00106** |

**Paper-value comparison (reference only — NOT a reproduction claim).** The
9-scene numbers are on a different, smaller patch set and are **not directly
comparable** to the published 7,899-patch averages. Not asserted as
reproduction; recorded side-by-side only.

**Seasonal-composition bias — observation + hypothesis (fixed here BEFORE
H1, per the confound concern).** All four metrics move the *same* direction
(worse than paper: PSNR↓ SAM↑ SSIM↓ MAE↑). Four-for-four coherence is
unlikely to be pure sampling noise; it points to a composition shift from
dropping summer scene 73:

| Season | Full 10-scene | 9-scene (realized) |
|---|---|---|
| spring | 3,983 (50.4%) | 3,983 (**56.0%**) |
| summer | 1,565 (19.8%) | 782 (**11.0%**) |
| fall | 784 (9.9%) | 784 (11.0%) |
| winter | 1,567 (19.8%) | 1,567 (22.0%) |

Dropping summer-73 **halves the summer share** (19.8%→11.0%) and raises
spring/winter/fall. *Hypothesis (not asserted):* summer scenes — vegetated,
clearer atmosphere, higher illumination — tend to be the easier cloud-removal
targets, so halving summer and over-weighting spring/winter yields a
**harder evaluation set**, consistent with all four metrics degrading. This
is an *observation with a plausible mechanism*, not a proven cause; subset
variance alone is not excluded.

**Consequence flagged for H1 (Arm B / B1).** H1 measures the VH-clipping
(−25 vs −32.5) SAM delta on this same 7,116 set. If seasonal composition
influences that delta (e.g. SAR backscatter dynamic range differs by
season/land-cover), the "VH effect" H1 isolates could carry a seasonal
confound. Recording the composition here, before running H1, so its result
can be read with this confound in view (e.g. by also reporting the delta
per season). **This bias affects absolute values only; it does NOT affect
the §2.3.2 internal-consistency gate**, which compares two metric
implementations on the *identical* prediction set — agreement is
composition-independent.

### 2.3.2 Gate 0 — reproduction determinism (2026-07-20)

Before any harness comparison: is EMRDM inference deterministic? Two facts,
separated:

1. **Deterministic given a fixed seed — PROVEN.** `emrdm_infer_scene.py` run
   **twice with identical config** (seed 3407, TF32 on, winter-63/60 patches)
   produced **byte-identical per-patch metrics** (SAM 7.943093 == 7.943093).
   No hidden nondeterminism (TF32 tensor-core ops, cuDNN, etc.).
2. **Stochastic across seeds — the sampler injects noise.** The released
   sampler is `ResidualEulerEDMSampler` with **`s_churn=5.0`, `s_noise=1.023`
   → stochastic**, not a deterministic ODE. So the *seed* (and the RNG
   consumption pattern) changes the result.

Consequence: the prediction-saving pass (my replica, seed 0, per-scene RNG
reset) and the authoritative Arm A `main.py` run (seed **3407** — its default,
confirmed in the log "Seed set to 3407"; continuous RNG over all 7,116
patches) are **two different valid random draws**, and their 7,116-aggregates
differ accordingly:

| Metric | replica (seed 0) | main.py (seed 3407) | Δ |
|---|---|---|---|
| SAM | 5.6516° | 5.6378° | 0.0138° |
| PSNR | 31.4672 | 31.4795 | 0.0123 |
| SSIM | 0.91905 | 0.91916 | 0.00012 |
| MAE | 0.019397 | 0.019385 | 0.000012 |

**Verdict:** NOT bit-identical to main.py — but the cause is *traced and
benign*: stochastic sampler (`s_churn=5.0`) sampled under a different seed
(0 vs 3407) and RNG stream (per-scene reset vs continuous). This is **not a
determinism bug** (proven identical given a fixed seed); it is a
**reproducibility finding in its own right — EMRDM's reported SEN12MS-CR
metrics are seed-sensitive at ≈0.014° SAM (7,116 aggregate); the published
5.267 is one seed's realization**, which no paper table discloses.

**Why Gate 1 remains valid.** Gate 1 does not compare two *inferences*; it
scores **one** saved prediction set with two *metric implementations*
(EMRDM `img_metrics` vs Nadir harness). Agreement of two metric
implementations on the *identical* image is seed-independent, so Gate 1
runs on the seed-0 predictions unaffected. (Gate 0's stated worry — "if the
two inferences produce different predictions the comparison is meaningless"
— applies to inference-vs-inference, which Gate 1 does not do.)

### 2.3.2a SAM value reconciliation (three numbers, one config)

Three SAM values appeared and must be reconciled BEFORE Gate 1. The apparent
5.638 → 7.943 "2.3° gap" is **entirely a patch-set difference, not a config
difference** — traced exhaustively:

| Value | Run | Patch set | Seed |
|---|---|---|---|
| **5.6378°** | Arm A `main.py` | all 9 scenes, **7,116** | 3407 |
| **5.6516°** | replica (gate 0 pool) | all 9 scenes, **7,116** | 0 |
| **7.943°** | determinism spot-check | **winter-63 only, 60 patches** | 3407 |

Config diff of the 5.638 vs 7.943 runs — checked line by line, everything
identical except the patch set:
- **Preprocessing:** both `SEN12MSCRInterface` default (S2 [0,10000]→[0,1]→
  [-1,1]; S1 [-25,0]) — identical.
- **Sampler:** byte-identical (`num_steps 5, s_churn 5.0, s_tmin 0, s_tmax
  1e8, s_noise 1.023, sigma_min 0.001, sigma_max 100`), EMA on — identical.
- **Seed:** both 3407 — identical.
- **Patch set:** 7,116 (all 9 scenes) vs 60 (winter-63 partial) — **the sole
  difference.**

Why a 60-patch winter subset reads 7.943 while the full set reads 5.64:
**per-scene SAM ranges 2.48°–10.60°** (replica seed 0), and winter-63 is a
hard ~8° scene:

| scene | n | SAM° | | scene | n | SAM° |
|---|---|---|---|---|---|---|
| spring 106 | 784 | 2.48 | | spring 123 | 781 | 5.41 |
| spring 31 | 784 | 3.43 | | winter 108 | 783 | 6.73 |
| fall 139 | 784 | 3.75 | | **winter 63** | 784 | **8.05** |
| spring 140 | 850 | 5.06 | | spring 44 | 784 | 10.60 |
| summer 119 | 782 | 5.41 | | | | |

The **patch-weighted mean of these nine = 5.6516°**, exactly the replica's
7,116 pooled aggregate. winter-63's full 784 patches = 8.05° (seed 0); the
60-patch seed-3407 spot-check = 7.943° — the same hard scene, sub-sampled.
**There is no unexplained discrepancy: 7.943 was never a full-set number.**

**Confirmed "correct Arm A" = the full 7,116-patch aggregate** (5.6378°
main.py / 5.6516° replica; the 0.014° gap is the seed sensitivity of §2.3.2).
**Gate 1 proceeds on the full 7,116 seed-0 predictions** (all 9 scenes) —
never on the winter-63 spot-check.

### 2.3.3 Gate 1 — harness vs EMRDM code, internal consistency (2026-07-20): PASS

EMRDM `img_metrics` vs the Nadir harness on the **identical** seed-0
predictions (all 9 scenes, 7,116 patches pooled per-patch):

| Metric | EMRDM code | Nadir harness | \|Δ\| | limit | verdict |
|---|---|---|---|---|---|
| SAM | 5.651574° | 5.651177° | **0.000397°** | 0.05 | PASS (gated) |
| PSNR | 31.467209 | 31.467423 | **0.000214** | 0.10 | PASS (gated) |
| MAE | 0.019397 | 0.019396 | **0.000001** | 0.001 | PASS (gated) |
| SSIM | 0.919048 | 0.904503 | 0.014545 | — | **EXCLUDED — not comparable** |

The three gated (identical-formula) metrics agree with **>100× margin**
(SAM Δ 0.0004° vs 0.05° tolerance). **Verdict: harness == EMRDM code at full
pipeline scale.**

**SSIM is EXCLUDED from the comparison — it is not a comparison target, not a
"pass".** The two SSIM implementations are different windows *by design*
(theirs `pytorch_ssim` gaussian 11×11; ours skimage uniform 7×7), so their
outputs are not expected to match and comparing them is meaningless. This
exclusion is **pre-registered**, proven by git to precede this run (which is
the *only* thing that makes the exclusion legitimate rather than a failing
metric explained away):

| Item | Commit | Timestamp (KST) |
|---|---|---|
| 3b gate script — `GATED={SAM,PSNR,MAE,RMSE}`, SSIM sanity-only, comment *"recorded, not gated (different implementations by design)"* | `e15e85a` | 2026-07-16 22:52:05 |
| 3b doc gate §3.1 — SSIM row naming *skimage uniform-7 vs pytorch_ssim gaussian-11* | `c38c289` | 2026-07-16 22:41:34 |
| **Gate 1 run** (`internal_consistency.json` written) | — | **2026-07-20 13:11:04** |

Pre-registration precedes the run by **~3.6 days**, with the technical
rationale (not just a timestamp) recorded in both commits. The
`internal_consistency.py` script had wrongly *gated* SSIM at ±0.005 — that
±0.005 is the §3 *paper-reproduction* tolerance (same implementation, does
not apply cross-implementation); the script failed to follow the 3b
pre-registration. Correcting the script to the pre-registration is
justified; **no gated tolerance was changed**, and SSIM is recorded (Δ
0.0145, consistent with the pre-quantified ~0.06 per-patch difference) as a
datum, not scored.

**What Gate 1 adds over 3b (one line).** 3b compared raw (no-model)
cloudy-vs-clear metrics; Gate 1 compares metrics on **model predictions** —
so Gate 1 additionally validates the *model-inference + prediction
save(uint16 npz)/load* path that 3b never exercised. Both now pass: the
harness is verified end-to-end, from raw preprocessing through model
inference to metric computation.

**Combined verdict — both gates pass, separately:**
- **Gate 0 (determinism):** PASS — deterministic given a fixed seed (byte-
  identical on repeat); the replica-vs-main.py 0.014° is traced to the
  stochastic sampler + seed, a recorded finding, not a bug.
- **Gate 1 (harness == EMRDM code):** PASS — gated metrics agree to
  <0.0004° SAM over 7,116 patches.

→ **Arm B (H1) gate is OPEN.**

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

### 3.1 Step-3 gates (pre-registered BEFORE any measurement; committed first)

**3a — granularity of their released metrics log (inspected 2026-07-16):**
`test-metrics.csv` is **aggregate-only**: 2 lines (header + one row), 26
columns, no patch or scene identifiers — a Lightning epoch-level log.
Per-patch comparison against their run is therefore impossible; Step 3
validates the pipeline via the gates below. Side-finding for §5.1: the file
has `final_MAE_cloudfree` / `final_MAE_cloudy` columns — their code
supports mask-split metrics (`img_metrics(masks=...)`) — but they are NaN
because the released test config sets `cloud_masks: None`. The mask-split
capability existed and was switched off in the published evaluation.

**3b — no-model harness gate (runs BEFORE any inference):**
For every patch of winter scene 63, compute cloudy-vs-clear ("raw")
metrics twice: (i) EMRDM's own pipeline (their `process_MS`/`process_SAR`
preprocessing + their `img_metrics`, in the emrdm venv), (ii) Nadir's
harness (our preprocessing + our metric suite, in the main venv), with
per-patch results exchanged as files. Declared per-patch tolerances:

| Metric | Tolerance | Basis |
|---|---|---|
| SAM | ≤ 0.01° | identical formula; fp32-vs-fp64 acos noise |
| PSNR | ≤ 0.01 dB | identical formula (20·log10(1/RMSE)) |
| MAE | ≤ 1e-5 | identical formula |
| RMSE | ≤ 1e-5 | identical formula (ad-hoc in our compare script) |
| SSIM | **not gated** — measured & recorded; sanity bound ≤ 0.05 | different implementations by design: theirs `pytorch_ssim` (gaussian 11×11), ours skimage (uniform 7×7). The delta itself is a B2-type datum |

**Why this SSIM exclusion earned its keep (2026-07-20).** At Gate 1
(§2.3.3), the comparator script did *not* honor this pre-registration — it
applied the §3 same-implementation SSIM tolerance (±0.005) to the
cross-implementation comparison and returned **FAIL (Δ 0.0145)**. The
metrics were correct; the two SSIM rulers simply differ (gaussian-11 vs
uniform-7). Had this pre-registration not existed, that FAIL would have read
as a harness defect (or, in the published-table setting, as a performance
gap). This is the §5.1 undeclared-convention thesis reproduced *inside our
own pipeline*: without knowing the implementations differ, one misreads two
rulers as a result. The pre-registration here — committed 3.6 days before
the run (`e15e85a`/`c38c289`; provenance table in §2.3.3) — is exactly what
prevented the misread.

Gate passes only if SAM/PSNR/MAE/RMSE are within tolerance for **every**
patch. Failure ⇒ the harness (data loading, preprocessing, or metric
implementation) is wrong; no inference runs until it is fixed.
Scene-63 raw values are NOT compared against the full-test-set
`raw_SAM 13.37` (different sample); that comparison only becomes valid at
Step 5 over all 7,899 patches.

**3c — TF32 sensitivity (measured, not assumed):**
EMRDM's README test command passes `--enable_tf32`, so their published
numbers presumably ran TF32-on — but the GPU model is undeclared and TF32
behavior is architecture-dependent. On winter scene 63, run their full
inference twice: (i) TF32 on (their flag + natten defaults), (ii) TF32 off
(`torch.backends` flags off + `natten.disable_tf32()`), same seed, and
record per-metric deltas. Decision rule (pre-registered): if |ΔSAM| <
0.05° (the Arm A tolerance), TF32 is immaterial — keep defaults and close;
if ≥ 0.05°, that is a finding of the §5.1 kind (an undeclared numerics
setting that moves reported metrics ⇒ hardware-dependent reproducibility),
to be added to §5.1 and declared as a [choice] in protocol.md.

**3b RESULT (executed 2026-07-17): GATE PASS.** Winter scene 63, 784/784
triplets (integrity gate passed first: modalities paired, 13-band uint16 S2,
2-band float32 S1, ranges valid). Per-patch worst deltas, their pipeline
(emrdm venv) vs Nadir harness (main venv), file handoff:

| Metric | worst per-patch delta | limit | verdict |
|---|---|---|---|
| SAM | 5.72e-6 ° | 0.01° | pass (fp noise) |
| PSNR | 3.81e-6 dB | 0.01 dB | pass |
| MAE | 5.96e-8 | 1e-5 | pass |
| RMSE | 5.96e-8 | 1e-5 | pass |
| SSIM | **0.0596** | 0.05 sanity (not gated) | recorded — exceeds sanity bound |

Data loading, preprocessing, and SAM/PSNR/MAE/RMSE implementations agree to
floating-point noise. The SSIM cross-implementation delta (gaussian 11×11
`pytorch_ssim` vs uniform 7×7 skimage) reaches **0.06 per patch — larger
than the published EMRDM-vs-DiffCR SSIM gap (0.924−0.902 = 0.022)**.

**SSIM finding completed (2026-07-17) — verdict: convention-sensitivity
warning, NOT an invalidation of the 0.022 margin.** Source audit:

- **EMRDM**: `sgm/modules/learning/metrics.py:36` calls
  `pytorch_ssim.ssim(target, pred)`;
  `sgm/modules/learning/pytorch_ssim/__init__.py` is the classic
  Po-Hsun-Su implementation — gaussian 11×11, σ=1.5, C1=0.01², C2=0.03²
  (data_range=1 implicit), depthwise over all 13 channels, zero-padded
  borders included.
- **UnCRtainTS** (the metric code DB-CR states it uses):
  `util/pytorch_ssim/__init__.py` + `model/src/learning/metrics.py` —
  byte-for-byte the same convention (EMRDM's metrics module is this code's
  lineage). ⇒ DB-CR and EMRDM SSIMs share one convention.
- **DiffCR's own repo** carries two *different* conventions —
  `evaluation/eval.py`: skimage `compare_ssim(multichannel=True,
  gaussian_weights=True, use_sample_covariance=False, sigma=1.5)` on
  PIL-loaded (uint8) images; `evaluation/psnr_ssim.py`: BasicSR-style with
  C1/C2 hardcoded at 255-scale plus an 11×11×11 3D-conv variant. Neither
  matters for the SEN12MS-CR table, because:
- **EMRDM produced the DiffCR row itself**: the appendix states *"we
  implement the algorithms ourselves … if pre-trained weights are
  available, we directly use them; otherwise, we retrain the models from
  scratch"* and, specifically, *"for DiffCR, which lacks official
  implementation details for the SEN12MS-CR dataset, we reproduce it on
  this dataset"* (DiffCR's paper never evaluated SEN12MS-CR). A
  same-authors, same-harness reproduction makes 0.924 vs 0.902 a
  **same-ruler comparison; the 0.022 margin stands as a valid
  within-table performance difference.** (Caveat kept honest: the paper
  nowhere *explicitly* says "all rows use one metric implementation"; the
  inference rests on the reproduction being inside their codebase.)

What remains true and matters: any SSIM comparison **across** conventions
(pytorch_ssim-lineage vs skimage-lineage vs 255-scale variants) can move by
up to the 0.06 we measured — triple the margins these tables report — and
none of the papers declare their convention in the text. Nadir's own SSIM
convention is now declared in `protocol.md` §6, and the B4 arm will report
SSIM under both conventions (dual reporting) so our numbers can be placed
next to either lineage.

Environment addendum (this step): their loader imports `s2cloudless` at
module top even with `cloud_masks: None`; lightgbm needs system
`libgomp.so.1` (sudo required, unavailable non-interactively). 3b ran with
an import shim on the *harness* side (their preprocessing untouched, any
actual detector use raises). libgomp1 must be installed before Arm A runs
through their `main.py`. Also: `albumentations==1.4.10` resolves `albucore`
to a stringzilla-sdist-dependent version — pinned `albucore==0.0.13`.

**3c RESULT (executed 2026-07-17): TF32 is immaterial — defaults kept.**
Full 784-patch inference on winter scene 63, twice, same seed (identical
sampler noise), RTX 4080: (i) TF32 on (replicating `main.py --enable_tf32`
+ natten defaults), (ii) TF32 off (matmul/cuDNN/natten GEMM-NA all off).

| Metric | agg on | agg off | \|agg Δ\| | worst patch Δ | mean patch Δ |
|---|---|---|---|---|---|
| SAM (°) | 8.048504 | 8.046769 | **0.0017** | 0.0296 | 0.0063 |
| PSNR (dB) | 28.998953 | 29.004332 | 0.0054 | 0.0489 | 0.0106 |
| SSIM | 0.887575 | 0.887488 | 0.0001 | 0.0005 | 0.0001 |
| MAE | 0.025721 | 0.025726 | 6e-6 | 1.4e-4 | 3.2e-5 |
| RMSE | 0.036270 | 0.036252 | 1.9e-5 | 1.8e-4 | 4.1e-5 |

Pre-registered rule: |ΔSAM| < 0.05° ⇒ immaterial. Measured 0.0017°
aggregate (worst single patch 0.0296°, still under). **Verdict: TF32 does
not threaten the Arm A gate on this hardware — Arm A runs with their
`--enable_tf32` defaults, matter closed.** (Scene-63 aggregate SAM ≈ 8.05
is a single winter scene and is NOT comparable to the full-test-set 5.267;
no conclusion drawn from that difference.)

**3d RESULT (executed 2026-07-17): pipeline pass-through OK, handoff format
approved.** The tf32-on run wrote all 784 predictions as uint16 DN .npz;
the Nadir harness (main venv) consumed them against ground-truth rasters
and reproduced EMRDM's in-run per-patch metrics with worst deltas
SAM 8.3e-5° / PSNR 4.2e-5 dB / MAE 1e-6 — i.e. the uint16 quantization in
the file handoff costs three orders of magnitude less than the Arm A
tolerance. Environment note: the user installed system `libgomp1`;
`s2cloudless`/lightgbm now import natively and the 3b import shim is
dormant (engages only on ImportError).

**Step 3 verdict: all gates passed (3a aggregate-only, 3b harness fp-level
agreement, 3c TF32 immaterial, 3d handoff validated). Cleared for Step 4
(full streaming pass) once the SSD arrives, then Step 5 (Arm A proper).**

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

### 5.1 H1 measurement design — PRE-REGISTERED 2026-07-20 (before any B1 run)

Committed before measurement; the decision rule below may not be revised
after seeing results (same discipline as §3 / the 3b gate).

**Conditions (single-factor, B1 principle).** Two input preprocessings that
differ **only** in the S1 VH lower clip bound; everything else identical
(VV clip −25 dB, the same [0,1] rescale, same weights, same sampler, same
9-scene 7,116-patch set, same seed policy = per-scene reset):
- `vh25` — VH clip [−25, 0] (Arm A / EMRDM `default` convention);
- `vh325` — VH clip [−32.5, 0] (DB-CR convention).

**Seeds (stochastic-sampler control, ≥3).** Each condition is run at seeds
**{3407, 0, 42}**. Because the sampler is stochastic (`s_churn=5.0`, Gate 0),
a single-seed delta is confounded by ≈0.014° seed noise; the two conditions
share the **same seed** within each pair so sampler noise cancels in the
paired difference, and 3 seeds quantify the residual spread.

**Statistic.** For seed *s*: `VHeff(s) = SAM_agg(vh325, s) − SAM_agg(vh25, s)`
(paired). Report `mean_s VHeff`, `sd_s VHeff` (the seed-noise floor), and
`VHeff` **per season** (spring/summer/fall/winter) — the seasonal confound
fixed in §2.3.2a means a season-varying VH effect is itself a result.

**Pre-registered decision rule (three outcomes, fixed now):**
1. `|mean VHeff| > 0.527°` **and** `|mean VHeff| > sd_s VHeff` →
   **H1 SUPPORTED**: one undeclared preprocessing constant moves SAM by more
   than the entire DB-CR↔EMRDM gap ⇒ cross-paper SAM ranking is meaningless
   without protocol disclosure.
2. `sd_s VHeff < |mean VHeff| ≤ 0.527°` → **H1 REJECTED**: VH clipping does
   move SAM, but by less than the paper gap ⇒ the protocol is more robust to
   this constant than suspected. Reported as-is with the measured Δ.
3. `|mean VHeff| ≤ sd_s VHeff` → **INCONCLUSIVE**: the VH effect is not
   distinguishable from seed noise at 3 seeds; report the bound, do not
   conclude.

**Cost/coverage note.** 2 conditions × 3 seeds = 6 full 7,116-patch passes
(~24 min each). No subsampling: the per-season readout needs all 9 scenes.
Any deviation (fewer seeds, subset) will be logged, not silently taken.

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
