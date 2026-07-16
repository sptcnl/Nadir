# Phase 2 Design Decisions: Representation Space for SAR-Conditioned Diffusion

**Status: PROPOSED — awaiting approval. No model code exists for Phase 2 yet.**
Date: 2026-07-16. Scope: choice of the generative backbone's representation
space (pixel vs. latent) for 13-band Sentinel-2 cloud removal conditioned on
Sentinel-1 SAR.

---

## 1. The two questions this document answers

**Q1 — How do 13 bands reach the generative model?**
Stable Diffusion's VAE is trained on 3-channel natural images. Any design
that funnels only RGB through the generative path destroys the object our
first-class metric (SAM, 13-band spectral angle) is defined on.

**Q2 — Does this task need a strong natural-image generative prior at all?**
Cloud removal is *conditional restoration*: SAR + the cloudy image carry most
of the information. A natural-image prior contributes texture priors — which
in this domain is a hallucination source to be controlled, not a feature.

## 2. Evidence from prior work (web-verified 2026-07-16)

| Work | Space | S2 bands | SAR fusion | Params | Sampling | Dataset |
|---|---|---|---|---|---|---|
| DB-CR, diffusion bridge (arXiv 2504.03607, 2025) | **pixel** | **13** | cross-attention (Q=optical, K/V=SAR) | 18.06M | **NFE=1** (3–5 ablated) | SEN12MS-CR, 256², 122k triplets |
| DiffCR (TGRS 2024) | **pixel** | S2 (Sen2_MTC) | none (optical only) | 22.91M / 45.86 GMACs | **1 step** ("absolute convergence in 3–5") | Sen2_MTC, 256² |
| DDPM-CR (RemSens 2023) | pixel | 13 + SAR | input concat of DDPM features | n/s | DDPM (many-step) | SEN12MS-CR-like |
| DMDiff (RemSens 2025) | pixel | S2 + SAR | dual-branch conditional guidance | n/s | DDPM-style | SEN12MS-CR |
| SAR-DeCR (RemSens 17(13):2241, 2025) | **latent** (encoder→latent→decoder stage) | RGB-focused texture stage | separate SAR-Fusion module before latent stage | n/s | LDM-style | SEN12MS-CR |
| EO-VAE (arXiv 2602.12177, 2026) / TerraMind tokenizers | domain latent (tokenizer for EO sensors) | multi-sensor incl. S2/S1 | n/a (tokenizer) | n/s | n/a | TerraMesh (millions of tiles) |

Reading of the evidence:

1. **The methods that win on SEN12MS-CR with all 13 bands are pixel-space,
   small (≈20M), task-specific conditional models** — not adapted
   text-to-image priors. DB-CR reports the field's best spectral fidelity
   (SAM 4.74°) in pixel space with 1-step inference.
2. Few-step sampling is *already proven* in this exact task (DiffCR 1–3
   steps, DB-CR NFE=1), de-risking our Phase 2 Step 5.
3. Domain tokenizers for multispectral latents exist (EO-VAE, TerraMind)
   but are trained on **millions of tiles**; nothing suggests a good 13-band
   VAE emerges from a 20k-patch subset.
4. The latent-space cloud-removal work we found (SAR-DeCR) confines the
   latent stage to texture synthesis and handles SAR fusion *outside* it —
   consistent with the latent path being spectrally lossy.

Answer to Q2, from the same evidence: **no strong natural-image prior is
needed**. Every competitive result trains the conditional model from scratch
at DSen2-CR scale (our Phase 1 baseline is 18.9M — same class). The prior
that matters is the *conditioning signal*, not ImageNet/LAION texture
statistics.

## 3. Candidates

### A — Pixel-space conditional diffusion (256×256, 13 bands directly)

- No VAE anywhere: the model denoises the 13-band target conditioned on
  SAR (2ch) + cloudy S2 (13ch). SAM is measured end-to-end on the actual
  model output. Inpainting constraints (Step 3) apply directly in pixel
  space where the cloud mask lives.
- Cost center is per-step compute at 256². Mitigation: few-step sampling
  (Step 5), which prior work shows is achievable in this task.
- **Params:** 20–60M UNet (DiffCR-class). **A100 40GB estimate:** at
  ~46 GMACs/sample (DiffCR-scale), bf16, batch 16, ~35% MFU → ~4.8 TFLOP/step
  → order of 20 steps/s → **300k steps ≈ 4–10 h; < 1 day** including EMA/val.
  (Engineering estimate, not a benchmark.)
- **Convergence on 20k patches:** plausible. Conditional restoration
  converges far easier than unconditional generation; DiffCR's Sen2_MTC is
  not larger per-scene, and D4 augmentation ×8 our effective set. Risk is
  overfitting, monitored via the geographic val split.
- **SAM measurable:** yes, directly on model output.
- **Prior work:** DB-CR, DiffCR, DDPM-CR, DMDiff — the dominant choice.

### B — Train a domain VAE (13-band → latent) + latent diffusion

- 13ch KL-VAE (f=8: 256²×13 → 32²×c) trained on our data, then diffusion in
  that latent. Reconstruction error is controllable during VAE training
  (add SAM to the VAE loss).
- **The VAE's encode→decode reconstruction quality is a hard ceiling on the
  whole system.** This must be measured before any diffusion work
  (protocol in §5) — if the roundtrip SAM is not far below the baseline's
  cloud-region SAM, B is dead on arrival.
- **Params:** VAE ~30–80M + latent UNet ~30–100M. **A100 estimate:** VAE
  100–200k steps (~0.5–1.5 days) + latent diffusion (cheap, hours) + the
  ceiling-measurement day → **~2–4 days**, roughly 3× candidate A, plus one
  extra model to maintain and ablate.
- **Convergence on 20k patches:** the *diffusion* converges (latents are
  small); the *VAE* is the risk. Good tokenizers in this domain (EO-VAE,
  TerraMind) train on millions of tiles; 20k patches likely yields a
  blurry/spectrally biased decoder, and every artifact it makes is
  unrecoverable downstream.
- **SAM measurable:** yes, after decoding — but the metric then conflates
  diffusion error with decoder error; diagnosing regressions requires the
  ceiling measurement as a standing reference.
- **Prior work:** exists for tokenizers (EO-VAE/TerraMind) but at data
  scales we don't have; no SEN12MS-CR cloud-removal SOTA takes this path.

### C — Frozen SD VAE, RGB through diffusion, other 10 bands elsewhere

- Easiest to stand up (pretrained SD VAE + established LDM tooling), and
  that is the only thing in its favor.
- **Why it conflicts with the project thesis (explicit, as required):**
  1. The generative model only ever sees RGB. The remaining 10 bands must be
     produced by a second, non-generative pathway (regression head or
     copy-through), so end-to-end 13-band SAM no longer measures the
     diffusion model at all — it measures whichever auxiliary head we bolt
     on. The project's central claim ("SAR-conditioned diffusion preserves
     spectra under clouds") becomes unfalsifiable in this design.
  2. NIR/SWIR are precisely the bands that carry vegetation/moisture signal
     (NDVI, NDWI) — the downstream products we argue cloud removal must not
     corrupt. Excluding them from the generative path optimizes the bands
     that matter least scientifically.
  3. The SD VAE itself is lossy on reflectance data it was never trained on
     (SD-class VAEs reconstruct natural images at roughly mid-20s dB PSNR;
     reflectance stretch/distribution differs), so even the RGB channels
     inherit an uncontrolled reconstruction floor.
  4. A LAION-trained prior is exactly the hallucination source Q2 warns
     about, injected into the pipeline with the least ability to control it.
- **Params/time:** UNet fine-tune ~0.5–1 day; cheapest. **Convergence:**
  fine. **SAM:** not meaningfully measurable end-to-end (see above).
- **Prior work:** SAR-DeCR is adjacent (latent texture stage), and notably
  keeps SAR fusion *out* of the latent stage.

## 4. Comparison table

| Criterion | A: pixel-space | B: domain VAE + LDM | C: SD VAE (RGB) |
|---|---|---|---|
| Trainable params | 20–60M | 60–180M (VAE+UNet) | ~50–90M (UNet; VAE frozen) |
| Est. training time (A100 40GB) | **< 1 day** | 2–4 days (+ceiling study) | ~1 day |
| Converges on 20k patches? | plausible (conditional task) | diffusion yes, **VAE doubtful** | yes |
| 13-band SAM measurable end-to-end? | **yes, directly** | yes, ceiling-limited by decoder | **no** (10 bands bypass the model) |
| Hallucination control | direct pixel-space inpainting constraint | indirect (latent blending) | weakest (LAION prior, latent-only masking) |
| Prior-work support (this task) | **DB-CR, DiffCR, DDPM-CR, DMDiff** | none at our data scale | SAR-DeCR (partial) |
| Extra moving parts | none | VAE training + ceiling upkeep | dual pathway for 10 bands |

## 5. Recommendation

**Candidate A: pixel-space conditional diffusion on all 13 bands.**

Rationale, in order of weight:
1. It is the only design in which SAM measures the model we actually train,
   end-to-end, with no reconstruction floor beneath it.
2. The strongest published results on our exact dataset and task are
   pixel-space conditional models of ~20M params with 1–5 step inference —
   the compute objection to pixel space is already answered in the
   literature, and it aligns with our own Step 5 plan.
3. It answers Q2 honestly: no imported prior, hallucination controlled at
   the source via the Step-3 inpainting constraint in the space where the
   cloud mask is defined.
4. Lowest cost and fewest moving parts on a 16GB local GPU / A100 subset
   budget; leaves ablation cycles (Step 4's 4 conditioning variants) inside
   the compute envelope.

**Fallback path:** if A plateaus on capacity (visibly under-fitting thick
cloud regions at converged loss), B is the fallback — *gated* on the ceiling
measurement below. C is rejected outright; adopting it would require
re-scoping the project's thesis, not an engineering decision.

## 6. VAE reconstruction ceiling protocol (gate for B — run BEFORE any B work)

One-day experiment, no diffusion involved:

1. Train a 13ch KL-VAE (f=8, ~30M params, L1 + KL + SAM-augmented recon
   loss) on the real-data training ROIs only.
2. Encode→decode the **held-out test ROIs** (clear images; no clouds
   involved — this isolates pure reconstruction).
3. Report PSNR / SAM / SSIM with the Phase-1 metric suite, full-image
   region (mask irrelevant here).
4. **Gate:** the roundtrip SAM must be well below the *cloud-region* SAM of
   the Phase-1 DSen2-CR baseline on the same test ROIs (measured in Phase 2
   Step 4). If reconstruction alone eats a comparable error budget, B cannot
   beat the baseline and is rejected without further investment.
5. Cheap corollary probe (also informs C's floor): run the frozen SD VAE
   roundtrip on the RGB bands of the same test set and log PSNR/SAM(RGB).

## 7. Consequences for Steps 2–5 (design constraints locked by this choice)

- **Step 2 (SAR conditioning):** conditioning variants (none / concat /
  ControlNet-style zero-conv branch / cross-attention) attach to a
  pixel-space UNet; DB-CR's result makes (d) cross-attention the variant to
  beat, (b) concat the cheap default. Control strength stays a scalar
  config knob for all variants.
- **Step 3 (mask-aware):** inpainting constraint (keep clear pixels from
  the input, generate only under the mask) implemented directly in pixel
  space; feathering/soft-mask experiments operate on the same mask the
  metrics use. RePaint-style resampling is available at inference because
  there is no latent mismatch at the boundary.
- **Step 5 (few-step):** target regime 1–8 NFE; literature indicates the
  task supports it. Solver-only improvements (DPM-Solver++) first, then
  distillation/rectification only if needed.
- **Prediction target:** x0- or v-prediction (not epsilon) to keep the
  mask-blending and spectral losses expressible on the predicted clean
  image at every step. (Final choice documented with Step 2 implementation.)

## References

- Meraner et al., DSen2-CR, ISPRS 2020 — Phase 1 baseline.
- Zou et al., *DiffCR: A Fast Conditional Diffusion Framework for Cloud
  Removal From Optical Satellite Images*, IEEE TGRS 2024.
  https://arxiv.org/abs/2308.04417 / https://github.com/XavierJiezou/DiffCR
- *Multimodal Diffusion Bridge with Attention-Based SAR Fusion for
  Satellite Image Cloud Removal* (DB-CR), 2025. https://arxiv.org/abs/2504.03607
- Jing et al., *Denoising Diffusion Probabilistic Feature-Based Network for
  Cloud Removal in Sentinel-2 Imagery* (DDPM-CR), Remote Sensing 15(9), 2023.
  https://www.mdpi.com/2072-4292/15/9/2217
- *DMDiff: A Dual-Branch Multimodal Conditional Guided Diffusion Model for
  Cloud Removal Through SAR-Optical Data Fusion*, Remote Sensing 17(6), 2025.
  https://www.mdpi.com/2072-4292/17/6/965
- *SAR-DeCR: Latent Diffusion for SAR-Fused Thick Cloud Removal*, Remote
  Sensing 17(13):2241, 2025. https://doi.org/10.3390/rs17132241
- *EO-VAE: Towards A Multi-sensor Tokenizer for Earth Observation Data*,
  2026. https://arxiv.org/abs/2602.12177
- *Fusing Sentinel-1 and Sentinel-2 data with diffusion models for cloud
  removal*, Remote Sensing of Environment, 2025.
  https://www.sciencedirect.com/science/article/abs/pii/S0034425725004535
