# Nadir

**SAR-conditioned cloud removal for Sentinel-2 imagery — with a reproducibility-first evaluation protocol.**

Clouds block optical satellite imagery; Sentinel-1 SAR penetrates cloud and
is the only observation of the ground beneath it. Nadir studies how to
condition a cloud-removal model on SAR, and — equally — how to *evaluate*
such models honestly. Spectral fidelity (SAM, the 13-band spectral angle) is
treated as a first-class metric, not an afterthought behind PSNR, because a
model can score well on pixel fidelity while destroying the band ratios that
downstream products (NDVI, NDWI) depend on.

> **Status.** Phase-1 infrastructure (reproducible data pipeline, metrics,
> DSen2-CR baseline, dummy end-to-end training) is complete. The current
> headline result is a **standalone re-evaluation of a published SOTA model
> (EMRDM, CVPR 2025)** that surfaced concrete reproducibility findings. The
> project's own SAR-conditioned diffusion model is *designed* (pixel-space,
> see `docs/design_decisions.md`) but **not yet trained**.

---

## Motivation

Two questions drive the work:

1. **How should 13-band multispectral data enter a generative model?** A
   Stable-Diffusion VAE is 3-channel; routing only RGB through it destroys
   the object SAM is defined on. `docs/design_decisions.md` surveys the
   options (web-verified against DiffCR, DB-CR, EMRDM, EO-VAE) and selects a
   **pixel-space** design.
2. **Can published SAR-optical cloud-removal numbers be compared at all?**
   Before training anything, we tried to place the field's strongest
   SEN12MS-CR SAM claims side by side — DB-CR (4.740°) and EMRDM (5.267°) —
   and found we could not, because the preprocessing conventions behind
   those numbers are undeclared. That investigation became the deliverable
   below.

## Method — a two-arm re-evaluation

Full design in `docs/emrdm_reevaluation.md`. Everything is **pre-registered**:
decision rules and tolerances are committed *before* the measurement they
judge (verifiable in `git log`).

- **Arm A (control):** run EMRDM's *released weights* through *their own*
  code and preprocessing; confirm the harness reproduces their inference.
- **Arm B (experiment):** change exactly one factor at a time and measure
  its effect on SAM. B1 (executed) isolates the S1 VH clipping constant.

The harness was validated at full pipeline scale by two separated gates:
**Gate 0** (determinism — inference is bit-identical given a fixed seed) and
**Gate 1** (our metric implementation agrees with EMRDM's `img_metrics` on
identical predictions to <0.0004° SAM over 7,116 patches).

## Results — undeclared-convention audit

EMRDM released weights, 9-scene / 7,116-patch SEN12MS-CR test subset, 3 seeds.
Each row: *hypothesis → measurement → verdict*.

| Convention | Result |
|---|---|
| **S1 VH clip** (−25 vs −32.5 dB) | Changing only this constant moves SAM by **+0.831°** (actual set) / **+0.777°** (reweighted to full-set season mix), seed sd 0.012° → **SUPPORTED**: the shift exceeds the entire 0.527° DB-CR↔EMRDM gap, so cross-paper SAM ranking is not meaningful without protocol disclosure. |
| **Stochastic sampler + seed** (`s_churn=5.0`) | SAM varies **≈0.014°** across seeds (Gate 0 and H1 independently agree) → **methodological finding**: a single reported SAM without seed/variance is under-specified. |
| **SSIM window** (pytorch_ssim gaussian-11 vs skimage uniform-7) | Cross-implementation Δ up to **0.06** per-patch (aggregate 0.0145), larger than the ~0.02 margins papers rank on → **not comparable**; declare the implementation or dual-report. |
| **TF32** (on/off, undeclared) | ΔSAM **0.0017°** → **immaterial** (rejected as a confound) — but only knowable because it was measured. |

The VH-clip result (H1) is the first standalone finding. It is **scoped**:
measured on EMRDM's weights, a spring-heavy 9-scene subset, and 3 seeds; no
generalization to the full 7,899-patch set or other models is claimed. The
effect direction is opposite to DB-CR's (so it does not reproduce their
number) — H1 pre-registered a *sensitivity magnitude*, and that is what is
shown.

## Reproduction

```bash
# Phase-1 infra: dummy data + baseline training smoke test (no real data)
uv venv && uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
python scripts/make_dummy_data.py
python -m nadir.train +experiment=dsen2cr_dummy
```

The EMRDM re-evaluation runs in an **isolated environment** (WSL2 + a
separate venv; EMRDM pins `torch 2.2.1`, `flash_attn`, `natten`). Its
scripts live in `scripts/reeval/`; each stage (download → integrity gate →
Gate 0 → Gate 1 → B1/H1) is a documented, resumable step in
`docs/emrdm_reevaluation.md`. SEN12MS-CR itself is fetched from the official
TUM mirror via `scripts/download_data.py` (see Limitations — the summer
archive is partially corrupt upstream).

## Limitations

- **9-scene subset, seasonally biased.** One of the 10 canonical test scenes
  (summer-73, 783 patches) is **unrecoverable** — a stable 2 MB corruption
  in `ROIs1868_summer_s2.tar.gz` on the only public mirror (verified;
  `docs/emrdm_reevaluation.md` §2.1). The remaining 7,116 patches are
  spring-heavy; all numbers are reported over the exact 7,116, never rounded
  to 7,899, and the seasonal bias is quantified and corrected-for where it
  matters.
- **EMRDM-only.** The audit uses one model's released weights. DB-CR has no
  public code, so its number cannot be audited at all.
- **SAM-first, and B1-only so far.** PSNR/MAE/SSIM shifts under B1 are
  exploratory; B2–B4 are future work.

## Future work

- **B2** — swap in Nadir's own SAM/metric implementation across the arm.
- **B3** — mask-split (full / cloud / clear) reporting with an s2cloudless-
  based cloud mask.
- **B4** — the full unified protocol end-to-end.
- **The Nadir model** — pixel-space SAR-conditioned diffusion with a
  swappable conditioning interface (none / concat / zero-conv branch /
  cross-attention) and an inpainting-constrained, mask-aware loss.

## Development workflow

This project is developed with **AI assistance under human verification**.
The pattern throughout: the AI drafts code, runs experiments, and writes
analysis; the human sets direction, challenges results, and demands evidence
before conclusions. Two conventions enforce honesty:

- **Pre-registration.** Decision rules and tolerances are committed before
  the measurement they judge; the git history is the audit trail. This
  caught a live mistake — the Gate-1 comparator initially mis-scored SSIM by
  applying a same-implementation tolerance across two different SSIM windows;
  the pre-registered exclusion (committed 3.6 days earlier) flagged it as a
  ruler mismatch, not a result.
- **Correcting the record.** Findings are revised when wrong. The design
  document's first revision *inverted* the EO-VAE citation (claimed domain
  VAEs need millions of tiles; the paper shows the opposite); this was
  caught, corrected, and the correction preserved in history rather than
  silently overwritten. See `docs/design_decisions.md` revision notes.

## Repository layout

```
src/nadir/        data pipeline, metrics, DSen2-CR baseline, training
scripts/          data download, dummy-data generation
scripts/reeval/   EMRDM re-evaluation harness (isolated env)
docs/             design_decisions.md · protocol.md · emrdm_reevaluation.md
configs/          Hydra configs (data / model / train / experiment)
tests/            pipeline, metrics, split-leakage, model, loss tests
```

## License

MIT.
