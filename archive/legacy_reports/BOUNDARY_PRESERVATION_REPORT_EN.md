# Boundary-Fidelity and Structure-Aware Defogging v4 Technical Report

## Conclusion

Version 4 follows the required order: the original enhancement chain runs in full, geometry is corrected afterward, and a weak structure-aware cleanup runs last. Single-image blind denoising, data consistency, directional continuity, edge enhancement, structure enhancement, change clipping, and ROI feathering retain their original algorithms and parameters. Geometry moves only already enhanced pixels, and cleanup is computed only from the enhanced result. The raw image supplies target endpoint coordinates; no raw pixels are written back.

The recommended output is `QUALITY_boundary_preserved_16bit.tif`. Flat-region high-frequency variation is reduced by **26.28%** relative to raw. Relative to the geometry stage, residual low-structure roughness falls by **4.54%**, while **99.09%** of digital edge acutance and **99.72%** of local contrast are retained. Cleanup-stage SSIM is **0.999951**.

## Processing order

```text
Original 16-bit TIFF
  → original self-supervised blind denoising (unchanged)
  → original directional continuity and data consistency (unchanged)
  → original quality search, edge/structure enhancement, clipping and feathering
  → save QUALITY_original_enhancement_16bit.tif
  → local sub-pixel geometry displacement using enhanced pixels only
  → save QUALITY_geometry_only_16bit.tif
  → weak structure-aware fine-grain and residual-fog cleanup
  → length, width, endpoint, edge, contrast and noise validation
  → QUALITY_boundary_preserved_16bit.tif
```

## Why raw-pixel anchoring was removed

The previous experiment wrote raw pixels into approximately 1.8% of the endpoint support. Geometry improved, but local raw noise returned and high-frequency-noise reduction fell from approximately 23% to 16.7%. Versions 3 and 4 therefore remove raw-pixel writeback entirely.

## Sub-pixel geometry correction

### Coordinates from raw data

Three neighboring profiles are sampled along each curved lamella center path. Upper and lower endpoints are measured independently on the raw and enhanced images using smoothed gradient extrema and three-point quadratic fitting. A raw reference is marked `review` when its internal uncertainty exceeds 3 px or edge SNR is insufficient.

### Displacing enhanced pixels rather than blending raw data

For each reliable endpoint:

\[
\Delta y=y_{enhanced}-y_{raw}
\]

A narrow displacement field is then applied to the enhanced image:

\[
I_{final}(x,y)=I_{enhanced}(x,y+d_y(x,y))
\]

The field follows the curved center path, with 1 px longitudinal and transverse radii and a normal maximum displacement of 0.35 px per endpoint. If detected error exceeds 3 px, indicating association with a competing peak, one local reassociation step of at most 1 px is allowed. Cubic interpolation samples the already denoised and enhanced image. The correction function receives the enhanced image and geometry coordinates, but not the raw pixel array, structurally preventing raw-noise leakage.

### Conservative handling of competing peaks

Some endpoints contain adjacent gradient peaks. Applying the full 2–4 px detected offset can make localization jump between them. Version 4 caps normal corrections at 0.35 px and association outliers above 3 px at 1 px, correcting peak association without stretching the whole lamella or affecting its neighbors.

## Structure-aware residual-fog cleanup

A low-structure gate is computed from the `σ=0.70` smoothed gradient of the geometry-stage image. All measured raw endpoint coordinates are surrounded by protected support:

\[
g=\exp[-(|\nabla G_{0.70}(I)|/T_{45})^2](1-P_{endpoint})
\]

Only gated pixels receive a blend of fine-scale (`σ=0.90`) and mid-scale (`σ=2.20`) smoothing:

\[
I'=I+g[0.12(G_{0.90}(I)-I)+0.015(G_{2.20}(I)-G_{0.90}(I))]
\]

The change is clipped to 0.65 times the estimated noise MAD and feathered at the target ROI boundary. Candidate strengths are searched automatically. A candidate is rejected if edge retention falls below 99%, contrast retention below 99.5%, SSIM against the geometry stage below 0.9999, or any geometry guardrail fails.

## Quality and geometry guardrails

The original quality search remains unchanged. The final result must additionally satisfy:

- layer-count change no greater than 1;
- maximum layer-center displacement no greater than 1 px;
- median FWHM change no greater than 4% versus raw;
- flat-region high-frequency variation no greater than 3% above the blind baseline;
- ROI SSIM of at least 0.985;
- reliable-layer endpoint-shift 95th percentile at most 0.35 px and maximum at most 0.75 px;
- reliable-layer length-delta 95th percentile at most 0.50 px and maximum at most 1.00 px.
- cleanup edge-acutance retention of at least 99%;
- cleanup local-contrast retention of at least 99.5%;
- cleanup SSIM against the geometry stage of at least 0.9999.

## Validation results

| Metric | Original enhancement | Final v4 result |
|---|---:|---:|
| Parameters | `edge=0.32`, `structure=0.06` | unchanged |
| High-frequency-noise reduction vs raw | about 23.03% | 26.28% |
| Digital edge-acutance gain vs raw | about 5.85% | 4.89% |
| Local-contrast gain vs raw | about 3.01% | 2.72% |
| Median absolute endpoint shift | 0.125 px | 0.034 px |
| Reliable raw boundaries | 99/100 | 99/100 |
| 95th-percentile absolute endpoint shift | 0.430 px | 0.260 px |
| Maximum reliable endpoint shift | 4.175 px | 0.695 px |
| Median absolute length delta | 0.245 px | 0.076 px |
| 95th-percentile absolute length delta | 0.877 px | 0.433 px |
| Maximum reliable length delta | 4.281 px | 0.704 px |
| Maximum median-FWHM change vs raw | — | 0.188% |
| Low-structure roughness reduction after geometry | — | 4.54% |
| Edge-acutance / local-contrast retention | — | 99.09% / 99.72% |

Geometry correction changes only about **0.20%** of all image pixels. The median change among modified pixels is approximately 17/65535 intensity levels, and SSIM versus the original enhancement is 0.999996. The subsequent weak cleanup has an SSIM of 0.999951 against the geometry stage.

## Outputs

- `QUALITY_original_enhancement_16bit.tif`: unchanged original enhancement/post-processing output for pixel-level audit.
- `QUALITY_geometry_only_16bit.tif`: audit image after geometry correction and before residual-fog cleanup.
- `QUALITY_boundary_preserved_16bit.tif`: recommended output containing geometry correction and guarded weak cleanup.
- `QUALITY_balanced_16bit.tif`: compatibility alias of the recommended output.
- `QUALITY_comparison.png`: raw, blind-denoised, original enhanced, and final v4 results.
- `QUALITY_fog_cleanup_closeup.png`: fixed-window before/after close-up and amplified change audit.
- `QUALITY_boundary_overlay.png`: raw endpoints in green, enhanced endpoints in red, uncertain raw references in orange.
- `boundary_geometry.csv` and `quality_metrics.json`: per-layer data and complete validation records.

## Limitations

This validates consistency in pixel coordinates, not absolute millimetre or micrometre metrology. Absolute accuracy still requires pixel-size calibration, system PSF/MTF characterization, a traceable reference, and repeated scans. Generative results remain visual-only and never enter the measurement chain.
