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

Additionally, EMRDM's released artifacts include their own test logs; if
those logs contain the measured values, Arm A must match the *logs* to
their printed precision (stricter than the table check).

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

### 7.1 Environment build log (Step 1 — to be filled during execution)

*(records every install attempt including failures, per the EO-VAE probe
precedent)*

## 8. Results

*(empty — to be filled by execution, tolerances above may not be edited)*
