# v17 Structure-Carrier-Conditioned Diffusion Report

## 1. Objective and status

The v15/v16 measurement outputs exclude generated pixels to maximize measurement safety. That choice avoids structural hallucination but limits further cleanup inside lamella bodies and hazy background regions. v17 evaluates a different architecture: move the audited v16 structural carrier into the generator as an explicit condition, apply differentiable structure/detail losses during generation, and retain the independent per-lamella audit after sampling.

Because the v17 output contains generated residual pixels, it is labelled `MEASUREMENT_CANDIDATE`. It does not replace v16; v16 remains the immutable fallback and numerical reference.

## 2. Conditional generative architecture

`app/structure_conditioned_diffusion.py` implements a compact bounded-residual DDIM. The network does not freely synthesize the whole image. It predicts a limited residual around the carrier:

\[
\hat{x}_0=C+A\odot\delta_{\max}\tanh R_\theta(x_t,G,t).
\]

`C` is the v16 structural carrier, `A` is a spatial permission field, and `G` contains ten condition channels:

1. structural-carrier intensity;
2. low-frequency generative appearance proposal;
3. carrier transverse gradient;
4. carrier longitudinal gradient;
5. curved lamella centerline field;
6. finite-width boundary field;
7. top/bottom endpoint protection field;
8. interlayer field;
9. raw/carrier dual-evidence confidence;
10. blind-denoising uncertainty.

Residual freedom approaches zero at boundaries and endpoints and remains lower inside measurable lamellae than in flat gaps or the central solid region. The output head is zero initialized, so an untrained or failed model starts from the carrier rather than a random image.

## 3. Single-image diffusion adaptation target

Only one input image is currently available, which is insufficient for training a general generator from scratch. v17 therefore performs patch-based single-image adaptation against a boundary-protected pseudo-clean target.

The target combines:

- non-local means at strength 0.58 outside lamella structure;
- zero-phase axial smoothing at strength 0.24 inside lamellae;
- a strength-0.10 capped appearance low-frequency difference in low-gradient areas;
- a hard total-change cap derived from carrier MAD noise;
- no direct proposal-pixel writeback.

The production run used 320 iterations, 96×96 patches, batch size 2, 12 base feature channels, 48 diffusion noise levels, and six deterministic DDIM steps. Full-resolution inference used 192-pixel tiles with 40-pixel overlap.

## 4. Composite structure and detail loss

The training objective is:

\[
\begin{aligned}
L={}&0.20L_{diff}+3.00L_{recon}+0.80L_{carrier}\\
&+3.00L_{edge-x}+2.20L_{edge-y}\\
&+2.00L_{width}+1.50L_{endpoint}\\
&+1.20L_{detail}+0.80L_{gap}+2.00L_{boundary}.
\end{aligned}
\]

- `diff`: diffusion noise-prediction consistency;
- `recon`: Charbonnier pseudo-clean reconstruction;
- `carrier`: endpoint-, boundary-, and confidence-weighted carrier consistency;
- `edge-x`: transverse thickness-edge preservation;
- `edge-y`: longitudinal endpoint-edge preservation;
- `width`: transverse projection gradients preserving width distribution;
- `endpoint`: longitudinal projection gradients preserving lengths and endpoints;
- `detail`: 3/9-pixel scale difference preserving midscale axial detail;
- `gap`: interlayer low-frequency separation consistency;
- `boundary`: finite-width boundary-field gradient-magnitude consistency.

The losses act on every predicted clean image during diffusion training, not only after generation.

## 5. Sampling and post-processing constraints

After sampling, a zero-phase low-structure residual cleanup is applied and every change is projected back into a carrier-centered amplitude interval. No scaling, registration, coordinate warp, or analytic lamella redraw is used.

Residual strengths `0, 0.12, 0.20, 0.30, 0.42, 0.55, 0.70, 0.85, 1.00` are evaluated. Every candidate is remeasured over 100 lamellae and 98 interlayers for endpoints, lengths, FWHM widths, gap geometry, dual-evidence counts, clarity, multiscale correlations, per-lamella axial detail, and SSIM. Strength zero is the mandatory v16 fallback.

## 6. Current-image result

The production run selected residual strength 0.85:

| Metric | v17 versus v16 |
|---|---:|
| Dimensions/type | 2200×1600 / uint16 |
| Lamella axial-noise reduction | 1.05% |
| Central high-frequency reduction | 2.67% |
| Flat-region high-frequency reduction | 1.51% |
| Edge-clarity change | +0.24% |
| Lamella-width P95 error | 0.3409% |
| Endpoint-shift P95 | 0.00043 px |
| Length-shift P95 | 0.05993 px |
| Interlayer-width P95 error | 0.1911% |
| Interlayer-length P95 error | 0.07970 px |
| Low/mid-frequency correlation | 1.000000 / 0.999990 |
| Gradient correlation | 0.999982 |
| Median/P10 axial-detail correlation | 0.999991 / 0.999985 |
| SSIM | 0.999968 |

All continuous geometry, detail, edge, and SSIM limits pass. Pixels outside the configured target ROI are identical to v16.

Strengths 0.30–0.70 and 1.00 reduced the discrete lamella dual-evidence count and were rejected. Strength 0.85 recovered the baseline count while retaining small continuous errors. This non-monotonic threshold behavior reinforces that a single-image audit is not evidence of cross-sample safety.

## 7. Run

```bash
docker compose run --rm ct-v17-structure-diffusion
```

or:

```bash
python app/structure_conditioned_diffusion.py \
  --source input/source_16bit.tif \
  --carrier results_generative_shape_v16_measurement_quality/FINAL_MEASUREMENT_v16_quality_enhanced_2200x1600_16bit.tif \
  --proposal input/generative_candidate_visual_only.png \
  --uncertainty results_sota/01_sota_blind/MEASUREMENT_sota_uncertainty_float32.tif \
  --outdir results_generative_shape_v17_structure_conditioned_diffusion
```

Primary outputs are the 16-bit candidate, comparison, boundary/condition audits, per-lamella and per-interlayer CSV files, the complete JSON manifest, and the trained `.pt` checkpoint.

## 8. Use boundary

v17 demonstrates the complete engineering loop of structural-carrier conditioning, in-training generative constraints, sampling-time amplitude protection, and independent audit. It has only been adapted to one image. Before length or thickness use, validate bias with repeated same-device scans, calibrated phantoms, pixel-size calibration, and scanner PSF/MTF measurements. Until that validation is complete, v16 remains the more conservative measurement input.
