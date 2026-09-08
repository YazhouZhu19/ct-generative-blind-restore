# v11 Denoised-Guide Dual-Evidence Generative Constraint Report

## Purpose

Version 11 measures geometry on a geometry-preserving blind-denoised guide instead of directly tracking the noisy 16-bit source. The raw image remains the sole generative content and field-of-view target and supplies matched-coordinate validation only.

## Separated Evidence Roles

- **Raw image:** sole scene, framing, and generative edit target.
- **Blind-denoised guide:** layer detection, center paths, subpixel endpoints, and 25-section transverse FWHM.
- **Raw matched evidence:** independent endpoint/width agreement and confidence; it is never averaged back into guide coordinates.
- **Appearance reference:** boundary cleanliness, finite-width plate appearance, dark gaps, and a complete central highlight only.

## Constraint Improvements

1. **Guide-first dual evidence.** Confidence combines guide repeatability, endpoint SNR, valid width samples, and raw/guide endpoint, length, and width agreement. The guide measurement remains authoritative.
2. **Joint lamella-bundle regularization.** Same-side tracks share a robust low-frequency deformation mode, while only 30% of a strongly smoothed individual residual is retained. Median row-step standard deviation fell from `0.2769 px` to `0.00767 px` without forcing the whole stack to be perfectly straight.
3. **Row-wise interlayer geometry.** Every gap is evaluated over the adjacent layers' common longitudinal interval as center separation minus both half-widths. Median, P10, P90, sample count, common length, and confidence are exported. The condition map encodes full blue gap corridors rather than gap centerlines alone.
4. **Generator-period removal and finite-width slabs.** The hard stage suppresses the generator's invented stripe frequency with a transverse low-pass scale larger than one measured pitch, then reconstructs 100 analytic finite-width dual-logistic slabs rather than pointed Gaussian lines.
5. **Closed-loop per-layer FWHM.** Three deterministic calibration rounds use the same FWHM operator as the audit to compensate for low-frequency background and subpixel rasterization without moving centerlines or endpoints.

## Experimental Result

The guide produced 49 left and 51 right lamellae, 98 interlayers, and 100/100 passing constraints. Median constraint confidence is `0.9849`; raw/guide endpoint disagreement P95 is `0.184 px`. The final constraint-matched detector resolves all 100 layers.

| Metric | v10 | v11 |
|---|---:|---:|
| Endpoint shift P95 | 0.083 px | 0.0856 px |
| Length delta P95 | 0.111 px | 0.1006 px |
| Median FWHM error | 4.54% | **0.0182%** |
| FWHM error P95 | 12.33% | **0.2010%** |
| Output median FWHM | 4.497 px | **4.251 px** |

The target guide median FWHM is `4.251 px`. An ordinary unmatched peak detector remains a diagnostic only because broad plate profiles and very close pairs can create or merge peaks; acceptance uses one-to-one constraint-coordinate matching.

## Commands

```bash
python app/generative_shape_constraint.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --outdir results_generative_shape_v11/00_guide_constraints

python app/generative_postprocess.py \
  --generated results_generative_shape_v11/01_generated/GENERATIVE_denoised_guide_shape_conditioned_raw.png \
  --outdir results_generative_shape_v11/02_postprocessed \
  --match-source input/source_16bit.tif \
  --resize-before-postprocess

python app/generative_shape_project.py \
  --profile v11 \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated results_generative_shape_v11/02_postprocessed/GENERATIVE_visual_only_postprocessed.png \
  --outdir results_generative_shape_v11/03_hard_shape_projection
```

The result is still generative and is not metrology-certified. Its geometry is analytically constrained, but its low-frequency appearance and texture include synthetic content. Calibrated non-generative outputs remain required for thickness, length, and defect acceptance.
