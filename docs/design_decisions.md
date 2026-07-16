# Phase 2 Design Decisions: Representation Space for SAR-Conditioned Diffusion

**Status: APPROVED (candidate A + B′ probe), rev 3. No model code exists for Phase 2 yet.**
Date: 2026-07-16. Revision history: rev 1 contained three factual errors (an
inverted reading of EO-VAE, a missing frozen-tokenizer candidate, a wrong
attribution of DB-CR's NFE=1 to pixel space rather than its bridge
formulation) — corrected in rev 2. Rev 3 adds: the executed EO-VAE weight
probe (§6.1), the DB-CR/EMRDM comparability audit (§2.3), and the EMRDM
released-weights re-evaluation as the project's first contribution item
(§5.1), scheduled as a Phase 1 extension.

Scope: choice of the generative backbone's representation space (pixel vs.
latent) for 13-band Sentinel-2 cloud removal conditioned on Sentinel-1 SAR.

---

## 1. The two questions this document answers

**Q1 — How do 13 bands reach the generative model?**
Stable Diffusion's VAE is trained on 3-channel natural images. Any design
that funnels only RGB through the generative path destroys the object our
first-class metric (SAM, 13-band spectral angle) is defined on. Whether a
*domain* VAE changes this calculus is now an evidence-based question (§2,
EO-VAE), not a rhetorical one.

**Q2 — Does this task need a strong natural-image generative prior at all?**
Cloud removal is *conditional restoration*: SAR + the cloudy image carry most
of the information. The strongest published systems (DB-CR, EMRDM) do not
even start their generative process from Gaussian noise — they anchor it at
the cloudy image (§2.1). That is the clearest statement in the literature
that this task is restoration-with-uncertainty, not free-form generation.

## 2. Evidence from prior work (web-verified 2026-07-16)

| Work | Space | Formulation | S2 bands | SAR fusion | Params | Sampling | Code public? |
|---|---|---|---|---|---|---|---|
| DB-CR (arXiv 2504.03607 / MERL TR2025-138, 2025) | pixel | **diffusion bridge**: forward process bridges cloud-free↔cloudy distributions; sampling starts *from the cloudy image* | 13 | cross-attention (Q=optical, K/V=SAR) | 18.06M | **NFE=1** (3–5 ablated) | **No** (checked merlresearch org + GitHub search, 2026-07-16) |
| EMRDM (CVPR 2025, arXiv 2503.23717) | pixel | **mean-reverting**: forward diffusion keeps the cloudy image as the distribution *mean*; EDM-style preconditioning, ODE samplers | S2 (SEN12MS-CR config) + optional aux modality via concat | condition concat | n/s | NFE≈5 | **Yes** (github.com/Ly403/EMRDM, incl. SEN12MS-CR weights) |
| DiffCR (TGRS 2024) | pixel | conditional diffusion, Gaussian start | S2 (Sen2_MTC) | none | 22.91M / 45.86 GMACs | 1–5 steps | Yes |
| DDPM-CR (RemSens 2023) | pixel | DDPM, Gaussian start | 13 + SAR | input concat | n/s | many-step | partial |
| DMDiff (RemSens 17(6), 2025) | pixel | conditional DDPM | S2 + SAR | dual-branch guidance | n/s | DDPM-style | n/s |
| SAR-DeCR (RemSens 17(13), 2025) | latent (texture stage) | LDM-style | RGB-focused stage | SAR fused *before* the latent stage | n/s | LDM-style | n/s |
| **EO-VAE** (arXiv 2602.12177, 2026) | domain latent (multi-sensor tokenizer) | tokenizer (Flux.2 AE + channel hypernetworks) | configs incl. **S2 L1C 13-band** and S1 RTC | n/a | 106.5M | n/a | **Yes — weights on HF (`nilsleh/eo-vae`), Apache-2.0** |

SEN12MS-CR headline numbers, re-verified against the original tables
(2026-07-16): DB-CR Table I reports **PSNR 33.47, SSIM 0.922, SAM 4.740°,
MAE 0.016**; EMRDM reports **PSNR 32.14, SSIM 0.924, SAM 5.267°, MAE 0.018**,
beating DiffCR (31.77 / 0.902 / 5.821 / 0.019) in its own table. Placing
4.740 against 5.267 as a ranking is **not supported**: the comparability
audit in §2.3 found a confirmed preprocessing mismatch and two unverifiable
equivalences between the two papers' setups — they are *not comparable as
reported*, not merely "concurrent works without cross-comparison". EMRDM
remains **the reproducible reference** (code + SEN12MS-CR weights public);
DB-CR's number cannot be audited at all without code.

### 2.1 What DB-CR and EMRDM actually share (rev-1 error #3 corrected)

Rev 1 credited DB-CR's NFE=1 to pixel space. Wrong: **the enabler is the
bridge/mean-reverting formulation.** Both methods replace the
Gaussian-endpoint diffusion with a process anchored at the cloudy image —
DB-CR as an explicit two-endpoint bridge, EMRDM as a mean-reverting SDE whose
terminal state is the cloudy image plus noise. The trajectory the sampler
must traverse is then short and informative (cloudy→clear is a small
perturbation compared to noise→image), which is *why* 1–5 NFE suffices.
Pixel space is where they happen to do it; the few-step property travels
with the formulation, and would equally apply in a latent space.

### 2.2 What EO-VAE actually shows (rev-1 error #1 corrected)

Rev 1 claimed domain VAEs need "millions of tiles" and dismissed candidate B
on that basis. The paper says the opposite:

- Trained on **only a subset of TerraMesh ("first 25 shards")** — explicitly
  *less* data than TerraMind's tokenizers — yet reconstructs S2 L2A at
  **PSNR 42.80 dB vs TerraMind's 22.95 dB**, SAM 0.0842 vs 0.3568 (units as
  reported), and **3.5× lower NDVI MAE** (0.0410 vs 0.1403). Inter-band
  ratios — the thing SAM guards — are demonstrably preservable by a domain
  VAE.
- In their latent-vs-pixel super-resolution comparison (Sen2NAIP, RGBN),
  latent diffusion with EO-VAE matches pixel-space quality (PSNR 21.60 vs
  21.76) at **~18× lower inference time**.

Honest caveats when transferring this to our task: the super-res comparison
is RGBN, not 13-band; it is not mask-constrained inpainting; there is no SAR
fusion; and EO-VAE's headline numbers are on TerraMesh distributions, while
SEN12MS-CR is L1C TOA with clouds — the roundtrip quality on *our* data is
exactly what the B′ probe (§6) measures instead of assumes.

### 2.3 Comparability audit: DB-CR 4.740° vs EMRDM 5.267° (added rev 3)

Conditions extracted from the DB-CR paper (experimental setup section) and
from EMRDM's released code (`sgm/data/sentinel/sentinel.py`,
`sgm/modules/learning/metrics.py`, `configs/example_training/sentinel.yaml`),
since EMRDM's paper defers preprocessing to an appendix:

| Condition | DB-CR (paper) | EMRDM (released code) | Match? |
|---|---|---|---|
| S2 clipping | "[0, 10,000], then scaled to [0, 1]" | `clip(img, 0, 10000)` → rescale to [0,1] (→[-1,1] in training wrapper) | ✓ |
| S2 bands | 13 ("13 spectral bands") | 13 (loader reads full stack, `[13,256,256]`) | ✓ |
| S1 normalization | VV clipped [-25, 0], **VH clipped [-32.5, 0]**, shifted/scaled to [0,1] | config leaves `rescale_method='default'` → **both VV and VH clipped [-25, 0]** (the `resnet` variant with -32.5 exists in the loader but is not selected) | **✗ CONFIRMED MISMATCH** |
| Test split | "following the dataset splits provided in [UnCRtainTS]": train/val/test = 114,056 / 7,176 / 7,899 patches | hardcoded ROI-scene split lists in the loader (UnCRtainTS-style, e.g. `splits['test'] = ['ROIs1158_spring_s1/s1_106', ...]`); patch counts not stated | ~ (likely same lists; **not verified**) |
| SAM computation | "calculated using the code provided by UnCRtainTS" (formula not shown) | verified in `metrics.py`: channel-axis dot product, `acos`, degrees, per-pixel mean (UnCRtainTS-identical structure); PSNR uses data_range 1 ⇒ metrics in [0,1] space | ~ (same formula family; DB-CR side **not auditable**) |

Reading: the confirmed mismatch is on the *input* side (SAR VH dynamic
range), which is a legitimate per-method design choice rather than a
benchmark violation — but it means the two systems were trained and
evaluated under different input conditions. Combined with the two
unverifiable equivalences (exact test-patch set; metric implementation on
the DB-CR side), the accurate statement is: **the reported 4.740° and
5.267° were produced under conditions that cannot be shown to be identical,
so they are not comparable as reported.** The only way to place them on one
scale is to re-evaluate under a single protocol — which is exactly the
EMRDM re-evaluation deliverable in §5.1 (and, for DB-CR, impossible until
code or weights are released).

## 3. Candidates

### A — Pixel-space conditional diffusion (256×256, 13 bands directly)

- No VAE anywhere: the model runs a bridge/mean-reverting process between
  cloudy and clear 13-band images, conditioned on SAR. SAM is measured
  end-to-end on the actual model output; the inpainting constraint (Step 3)
  applies in the space where the cloud mask is defined, at full resolution.
- **Params:** 20–60M (DiffCR/DB-CR class). **A100 40GB estimate:** at
  ~46 GMACs/sample, bf16, batch 16, ~35% MFU → order of 20 steps/s →
  **300k steps ≈ 4–10 h; < 1 day** including EMA/val. (Engineering
  estimate, not a benchmark.)
- **Convergence on 20k patches:** plausible — conditional restoration with
  an image-anchored process converges far easier than Gaussian-start
  generation; D4 augmentation ×8 the effective set; geographic val split
  monitors overfitting.
- **SAM measurable:** yes, directly, no floor beneath it.
- **Prior work:** DB-CR, EMRDM, DiffCR, DDPM-CR, DMDiff.

### B — Train a domain VAE from scratch (13-band → latent) + latent diffusion

- 13ch KL-VAE trained on our subset, then latent diffusion. EO-VAE proves
  the *concept* (domain VAEs preserve spectra); the open question is the
  data floor — "25 TerraMesh shards" is still far more than 20k patches,
  and every decoder artifact is unrecoverable downstream.
- **Params:** VAE ~30–105M + latent UNet ~30–100M. **A100 estimate:** VAE
  100–200k steps (~0.5–1.5 days) + ceiling study + latent diffusion →
  **~2–4 days**, plus a second model to maintain.
- **SAM measurable:** after decode, ceiling-limited; requires the §6
  protocol as a standing reference.
- Dominated by B′ unless the B′ probe fails for fixable reasons
  (fine-tuning the decoder on SEN12MS-CR would be a B/B′ hybrid).

### B′ — Frozen EO-VAE tokenizer + latent diffusion  *(added in rev 2)*

- Use the released EO-VAE weights as a frozen encoder/decoder; train only
  the latent bridge/MR diffusion. **Weight availability is no longer a
  paper claim — it was executed and verified on 2026-07-16 (§6.1):**
  checkpoint `eo-vae.ckpt` + `model_config.yaml` from HF repo
  `https://huggingface.co/nilsleh/eo-vae` (Apache-2.0), loaded via
  `EOFluxVAE.from_pretrained(repo_id="nilsleh/eo-vae", ...)`, 95.5M params,
  and a 13-band S2 L1C tensor roundtrips to a finite (1, 13, 256, 256)
  output through a (32, 32, 32) latent (f=8 spatial, 32 channels).
- **Cost structure is nothing like B:** no VAE training at all. The
  reconstruction-ceiling measurement is *inference only* — encode→decode
  our held-out test ROIs and run the Phase-1 metric suite. **~half a day of
  engineering + <1 h compute on the local RTX 4080.** The latent diffusion
  itself is hours on A100 (32×-ish smaller spatial grid).
- **Risks to measure, not assume:** L1C/TOA + cloud-domain shift from
  TerraMesh; mask granularity (a 256² cloud mask becomes ~32² in latent —
  boundary control for Step 3 is coarser than pixel space); SAR conditioning
  enters in latent space where DB-CR-style pixel-aligned cross-attention is
  less direct.
- **SAM measurable:** yes end-to-end, with a *measured, frozen* ceiling —
  strictly better bookkeeping than B.
- **Prior work:** EO-VAE itself demonstrates latent diffusion on EO tasks;
  no cloud-removal instance yet — adopting B′ would be the novel-ish path,
  which cuts both ways (§5.1).

### C — Frozen SD VAE, RGB through diffusion, other 10 bands elsewhere

- Easiest to stand up, and that is the only thing in its favor.
- **Why it conflicts with the project thesis (unchanged from rev 1):**
  1. Only RGB passes through the generative model; the other 10 bands come
     from a bolted-on non-generative pathway, so end-to-end 13-band SAM no
     longer measures the diffusion model. The central claim
     ("SAR-conditioned diffusion preserves spectra under clouds") becomes
     unfalsifiable.
  2. NIR/SWIR — the bands NDVI/NDWI depend on — are exactly what gets
     excluded from the generative path.
  3. The SD VAE adds an uncontrolled reconstruction floor on data it was
     never trained on; EO-VAE's TerraMind comparison shows how badly
     mismatched tokenizers reconstruct EO data (22.95 dB).
  4. A LAION prior is the hallucination source Q2 warns about, with the
     least ability to control it.
- **Prior work:** SAR-DeCR is adjacent and keeps SAR fusion *outside* its
  latent stage.

## 4. Comparison table

| Criterion | A: pixel-space | B: own VAE + LDM | B′: frozen EO-VAE + LDM | C: SD VAE (RGB) |
|---|---|---|---|---|
| Trainable params | 20–60M | 60–205M | 30–100M (VAE frozen) | ~50–90M |
| Est. training time (A100 40GB) | **< 1 day** | 2–4 days + ceiling study | ceiling probe ½ day (local GPU) + LDM hours | ~1 day |
| Converges on 20k patches? | plausible | diffusion yes, **VAE doubtful** | yes (only LDM trains) | yes |
| 13-band SAM end-to-end? | **yes, no floor** | yes, unmeasured ceiling | yes, **measured frozen ceiling** | **no** |
| Mask/inpainting control (Step 3) | **native, full-res** | coarse (latent-res mask) | coarse (latent-res mask) | weakest |
| SAR fusion precedent | DB-CR cross-attn, pixel-aligned | none | none in latent | SAR-DeCR: kept out of latent |
| Few-step support | bridge/MR: NFE 1–5 proven on this task | formulation transfers | formulation transfers | LDM solvers |
| Prior work (this task) | **DB-CR, EMRDM, DiffCR, DDPM-CR, DMDiff** | none at our scale | none (would be new) | SAR-DeCR (partial) |
| Extra moving parts | none | VAE training + upkeep | weight dependency, domain-shift risk | dual pathway |

## 5. Recommendation

**Candidate A: pixel-space bridge/mean-reverting diffusion on all 13 bands**,
with the **B′ ceiling probe run immediately after approval** as a cheap
hedge (it is half a day, inference-only, on hardware we already have).

Rationale (rebuilt after the EO-VAE correction — the "domain VAEs can't
preserve spectra" argument is dead and is not used below):

1. **Metric integrity.** A is the only design where SAM measures the trained
   model end-to-end with no reconstruction floor and no ceiling bookkeeping.
   B′ is acceptable on this axis only after its ceiling is measured on
   SEN12MS-CR L1C specifically — EO-VAE's numbers are on a different
   distribution and were not measured under clouds.
2. **Task-specific precedent.** Both reproducible SOTA lines on our exact
   dataset (EMRDM; DB-CR unreproducible but consistent) are pixel-space
   image-anchored formulations. EO-VAE's latent-vs-pixel win is on
   super-resolution without masks or SAR; transferring it to
   mask-constrained, SAR-fused cloud removal is an open research bet, not
   an engineering default.
3. **Step-3 requirements.** The inpainting constraint, feathered masks, and
   thin/thick soft weighting all operate on the pixel-space cloud mask. In
   any latent design the mask must be downsampled ~8×, and clear-region
   preservation can only be enforced after decode — exactly where seam
   artifacts and spectral drift enter.
4. **The efficiency argument for latents is real but not decisive here.**
   EO-VAE's 18× speedup attacks per-NFE cost; the bridge formulation attacks
   NFE count (1–5 on this task). At 256², a 20–60M pixel-space model at
   NFE≤5 is well inside both our A100 budget and deployment envelope.

**Fallback path:** if A under-fits thick-cloud regions at converged loss, or
per-NFE cost blocks Phase-4 interactivity targets, B′ is the fallback —
gated on its measured ceiling (§6). B (training our own VAE) is now
third-line, considered only if the B′ probe fails for reasons a
SEN12MS-CR-finetuned decoder would fix. C remains rejected outright.

### 5.1 This project's architecture is not novel — and that is by design

Stated plainly so no one (including us) oversells later: **pixel-space
SAR-fused diffusion for cloud removal on SEN12MS-CR already exists.** DB-CR
and EMRDM published it in 2025 with strong results; DDPM-CR and DMDiff
preceded them with weaker formulations. Choosing candidate A means our
backbone is an implementation of known ideas, and the Phase 2 write-up must
cite DB-CR/EMRDM as the architectural basis, not claim invention.

The project's contribution is deliberately located elsewhere:

1. **EMRDM released-weights re-evaluation** *(scheduled as a Phase 1
   extension — it precedes and does not depend on any model we train).*
   EMRDM publishes SEN12MS-CR weights (github.com/Ly403/EMRDM). We run
   those weights through this project's unified protocol — mask-split
   (full/cloud/clear) metrics, fixed preprocessing, our own SAM
   implementation — and measure whether the paper's reported numbers
   reproduce. This is an independent deliverable regardless of how our own
   model performs, and §2.3 shows why it is necessary: the field's two
   best claims cannot currently be placed on one scale. Its output also
   fixes the coordinate system in which our own model's numbers must be
   read — before we train anything.
2. **Evaluation protocol:** mask-split (full/cloud/clear) reporting of
   SAM-first metrics. Published tables (incl. EMRDM's) report full-image
   averages, which §2's own logic says can hide clear-pixel copying.
3. **Controlled conditioning ablation:** the (a) no-SAR / (b) concat /
   (c) zero-conv branch / (d) cross-attention ladder under identical seeds,
   data, and steps — including the thick-cloud-subset analysis of *when*
   SAR actually contributes. No published work isolates this.
4. **Failure analysis:** negative and null results (e.g. "SAR adds nothing
   outside thick cloud") reported as findings, with W&B evidence trails.
5. **Reproducibility:** open pipeline with geographic splits, fixed seeds,
   and released configs — against a field where the strongest claimed
   result (DB-CR) has no public code.
6. **Deployment curves:** NFE-vs-metric trade-off curves (Step 5), asking
   "where does SAM collapse" rather than "how low can NFE go".

## 6. Reconstruction ceiling probes (updated for B′)

**B′ probe (run first — inference only, ~half a day, local GPU):**
1. Load frozen EO-VAE (HF `nilsleh/eo-vae`, S2 L1C 13-band config).
2. Encode→decode held-out SEN12MS-CR test-ROI clear images; also a cloudy
   subset (the encoder must survive cloud radiometry it never saw).
3. Report PSNR/SAM/SSIM with the Phase-1 metric suite.
4. **Gate:** roundtrip SAM must sit well below the Phase-1 DSen2-CR
   baseline's cloud-region SAM on the same ROIs. Pass → B′ is a live
   fallback; fail → B′ dead, and B only revives with decoder fine-tuning
   evidence.
5. Corollary: SD VAE RGB roundtrip on the same set documents C's floor for
   the record.

**B probe (only if B ever revives):** as rev 1 — train 13ch KL-VAE on
training ROIs, same measurement, same gate.

### 6.1 B′ weight-availability probe — EXECUTED 2026-07-16 (verdict: B′ stays alive)

Environment: isolated venv (EO-VAE pins `torchvision==0.16.2`; installed
`--no-deps` with torch 2.13 CPU + einops/lightning/safetensors/omegaconf/
huggingface_hub/focal-frequency-loss instead — works). Probe script kept at
`scripts/probe_eovae.py`.

```text
Loading weights from ...huggingface\hub\models--nilsleh--eo-vae\snapshots\7f675ab7d242a34a63a03e6fd2d19f28bb73cdd2\eo-vae.ckpt
Checkpoint loaded: 0 missing (expected), 0 unexpected (ignored)
loaded EOFluxVAE, params=95.5M
input: (1, 13, 256, 256) z-scored range=(-1.334,8.558)
latent: (1, 32, 32, 32)
recon:  (1, 13, 256, 256) finite=True range=(-1.409,4.375)
dummy-data roundtrip (NOT a quality claim): PSNR=23.73 dB, MAE=0.0478, SAM=14.039 deg
PROBE OK
```

Facts established:
- Exact weight location: HF repo **`nilsleh/eo-vae`** (snapshot `7f675ab`),
  files `eo-vae.ckpt` + `model_config.yaml`; loader
  `EOFluxVAE.from_pretrained(repo_id="nilsleh/eo-vae", ...)`. 95.5M params.
- The S2 L1C 13-band wavelength configuration is accepted; latent is
  (32, 32, 32) for a 256² input — f=8 spatial compression, 32 channels
  (so a 256² cloud mask maps to 32² in latent, confirming §3's mask
  granularity concern).
- **Input convention (found the hard way):** per-band z-score of raw DN
  with TerraMesh statistics (`eo_vae/datasets/terramesh.py`,
  `MultimodalNormalize`) — *not* [0,1] reflectance. A first run with [0,1]
  inputs silently produced garbage-grade output (PSNR 17.1 dB); any future
  B′ work must use the verified convention in `scripts/probe_eovae.py`.
- The roundtrip numbers above are on a **synthetic dummy patch** whose
  spectra sit up to 8.6σ outside TerraMesh statistics; they are a smoke
  log, not a ceiling measurement. **The §6 gate still requires real
  SEN12MS-CR test ROIs** (pending real-data download).

Verdict: the rev-2 discard condition ("weights don't exist / don't load /
reject 13-band input") did **not** trigger. B′ remains the live fallback;
its quality gate is still open pending real data.

## 7. Consequences for Steps 2–5 (rewritten after §2.1)

- **Default formulation: image-anchored bridge / mean-reverting diffusion**
  (EMRDM-style reformulated MRDM with EDM preconditioning and deterministic
  ODE sampler as the reference design; DB-CR's bridge as the alternative if
  MR training proves unstable). Gaussian-start conditional diffusion is the
  ablation control, not the default. The network predicts the clean-image
  endpoint under preconditioning (x0-like target) — the earlier framing of
  "x0- vs v-prediction choice" is subsumed by the preconditioning design and
  will be fixed at Step 2 implementation, documented against EMRDM's
  elucidated design space.
- **Step 2 (SAR conditioning):** variants (none/concat/zero-conv
  branch/cross-attention) attach to the pixel-space denoiser; DB-CR's
  cross-modal attention is the variant to beat; EMRDM's plain concat is the
  strong-simple control. Control strength stays a scalar config knob.
- **Step 3 (mask-aware):** full-resolution pixel-space masks; feathering and
  thin/thick soft weighting operate on the same mask the metrics use;
  RePaint-style resampling remains available since there is no latent
  boundary mismatch.
- **Step 5 (few-step):** the bridge/MR formulation *natively* targets
  NFE 1–8; primary work is the NFE-vs-SAM curve with the ODE sampler, then
  solver tweaks; distillation/reflow only if the curve demands it.

## References

- Meraner et al., DSen2-CR, ISPRS 2020 — Phase 1 baseline.
- *Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite
  Image Cloud Removal* (DB-CR), arXiv 2504.03607 / MERL TR2025-138, 2025.
  https://arxiv.org/abs/2504.03607 — no public code as of 2026-07-16
  (merlresearch GitHub org and GitHub search checked).
- Liu et al., *Effective Cloud Removal for Remote Sensing Images by an
  Improved Mean-Reverting Denoising Model with Elucidated Design Space*
  (EMRDM), CVPR 2025. https://arxiv.org/abs/2503.23717 — code + SEN12MS-CR
  weights: https://github.com/Ly403/EMRDM
- Zou et al., *DiffCR*, IEEE TGRS 2024. https://arxiv.org/abs/2308.04417
- Jing et al., *DDPM-CR*, Remote Sensing 15(9), 2023.
  https://www.mdpi.com/2072-4292/15/9/2217
- *DMDiff*, Remote Sensing 17(6), 2025. https://www.mdpi.com/2072-4292/17/6/965
- *SAR-DeCR: Latent Diffusion for SAR-Fused Thick Cloud Removal*, Remote
  Sensing 17(13):2241, 2025. https://doi.org/10.3390/rs17132241
- Lehmann et al., *EO-VAE: Towards A Multi-sensor Tokenizer for Earth
  Observation Data*, arXiv 2602.12177, 2026.
  https://arxiv.org/abs/2602.12177 — code: https://github.com/nilsleh/eo-vae,
  weights: https://huggingface.co/nilsleh/eo-vae (`eo-vae.ckpt`,
  `model_config.yaml`; Apache-2.0) — **download + 13-band roundtrip verified
  2026-07-16, see §6.1.**
- *Fusing Sentinel-1 and Sentinel-2 data with diffusion models for cloud
  removal*, Remote Sensing of Environment, 2025.
  https://www.sciencedirect.com/science/article/abs/pii/S0034425725004535
