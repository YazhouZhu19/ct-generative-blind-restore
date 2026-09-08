# v7 Geometry-Safe Residual Denoising Report

## Outcome

Version 7 adds region-separated residual denoising after v6 boundary cleanup while retaining the full generative-prior blind model, original enhancement/post-processing chain, and v6 cleanup. The recommended full 2200 × 1600, 16-bit grayscale output is:

```text
results_sota/04_residual_denoise/QUALITY_v7_residual_denoised_16bit.tif
```

On the production image, flat-region high-frequency noise in the left/right measurement regions falls by **38.38%** versus raw, compared with 28.65% for v5 and 35.37% for v6. Noise inside the central highlight falls by **19.91%** versus raw. Final digital edge acutance remains **5.39%** above raw and local contrast remains **2.02%** above raw.

## Hybrid denoising method

### Lamella regions

Lamella interiors use a zero-phase axial filter with `σ=(2.50,0.18)`. Only residuals statistically consistent with noise are shrunk, while a transverse-gradient gate protects thickness edges:

\[
r=I-G_{(2.50,0.18)}I,
\quad w_n=\exp[-(|r|/(2.6\sigma_n))^4]
\]

\[
I'=I-\alpha M_{inside}M_{edge}M_{endpoint}w_n r
\]

This mainly suppresses variation along lamella length without averaging across left/right thickness boundaries.

### Central highlight

The central highlight lies outside the lamella endpoint envelopes, so it receives a separate fast non-local means pass with `patch=5`, `distance=5`, and `h=0.85σ`. An 18-pixel soft mask prevents smoothing across the central-block boundary.

### Geometry protection

- The raw image supplies coordinates for 100 lamella endpoints but no raw pixels are written back.
- Upper/lower endpoint cores are immutable.
- Each v7 pixel change is capped at `0.65×MAD`.
- Candidates must pass layer-count, FWHM, endpoint, length, acutance, contrast, central-mean, and SSIM guardrails.
- The independent length audit uses the fixed v6 geometry guide so before/after paths cannot be silently redetected.

## Production measurements

| Metric | v7 result |
|---|---:|
| Additional lamella axial-noise reduction | 6.97% |
| Additional central-noise reduction | 8.41% |
| Cumulative v5→v7 lamella axial-noise reduction | 16.38% |
| Flat-region high-frequency reduction versus raw | 38.38% |
| Central-region noise reduction versus raw | 19.91% |
| Edge-acutance gain versus raw | 5.39% |
| Local-contrast gain versus raw | 2.02% |
| Transverse edge-acutance retention versus v6 | 99.78% |
| SSIM versus v6 | 0.998548 |
| Maximum median-FWHM change | 1.23% |

The fixed-guide independent audit detects 100 lamellae: 98 pass automatically and two require review because their raw endpoint references are intrinsically uncertain. For reliable lamellae, endpoint-shift P95 is 0.248 px, length-change P95 is 0.402 px, and maximum length change is 0.499 px. All geometry guardrails pass.

## Reproduction

Complete five-stage container pipeline:

```bash
docker compose run --rm ct-sota
```

Run v7 independently:

```bash
python app/residual_denoise.py \
  --source input/source_16bit.tif \
  --input results_sota/03_boundary_cleanup/QUALITY_boundary_clean_16bit.tif \
  --outdir results_sota/04_residual_denoise
```

Before measurement use, inspect `residual_denoise_metrics.json`, `QUALITY_v7_boundary_overlay.png`, and `05_length_audit/length_qa.json`.

## Limitations

This remains single-image internal validation. Absolute physical length requires pixel-size calibration, a traceable standard, PSF/MTF characterization, and repeated scans. No generated or reference pixels replace measurement pixels, but absolute denoising error cannot be established without paired clean truth.
