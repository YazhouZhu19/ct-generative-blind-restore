# v16 Measurement-Image Quality Refinement Report

## Objective

The v15 measurement image is pixel-identical to its blind-denoised structural carrier, which maximizes structural safety, but visible grain remains in the central solid region. v16 reduces that noise without generated pixels, spatial transforms, or measurable lamella/interlayer drift.

## Method

`app/measurement_quality_optimize.py` treats v15 as an immutable baseline and evaluates zero-phase, capped candidates:

- axial residual shrinkage inside lamellae, protected by transverse gradients and endpoint masks;
- feathered non-local means inside the central solid region;
- per-pixel changes capped by a blind-noise MAD estimate;
- no generated-pixel readback or writeback;
- no scaling, registration, or deformation.

Every candidate is re-audited over 100 lamellae and 98 interlayers. Acceptance requires endpoint, length, FWHM, interlayer width/length, raw-nonregression, edge-clarity, multiscale-structure, and SSIM checks to pass simultaneously.

## Parameter selection

An initially stronger candidate reduced lamella axial noise by 5.87%, but the independent per-lamella audit found a 4.80% P95 width change, so it was rejected. Even weaker nonzero lamella strengths degraded the dual-evidence status of a few marginal thin layers.

The selected parameters are:

```text
lamella_strength = 0.00
central_strength = 0.92
```

Lamella and interlayer pixels therefore remain unchanged from v15; only the central solid region receives feathered non-local means.

## Current-image result

| Metric | v16 result |
|---|---:|
| Dimensions/type | 2200×1600 / uint16 |
| Generated-pixel weight | 0 |
| Central high-frequency noise reduction | 9.77% |
| Additional lamella processing strength | 0 |
| Lamella-width median/P95 error | 0% / 0% |
| Lamella endpoint P95 error | 0 px |
| Lamella length P95 error | 0.0599 px |
| Interlayer-width P95 error | 0% |
| Interlayer-length P95 error | 0.0799 px |
| Low/mid-frequency correlation | 1.0000 / 0.9999 |
| Gradient correlation | 0.99985 |
| Median/P10 axial-detail correlation | 1.0000 / 1.0000 |
| SSIM against v15 | 0.99963 |

All strict guardrails pass.

## Run

```bash
python app/measurement_quality_optimize.py \
  --source input/source_16bit.tif \
  --input results_generative_shape_v15_dual_output/FINAL_MEASUREMENT_structure_preserved_2200x1600_16bit.tif \
  --outdir results_generative_shape_v16_measurement_quality
```

or:

```bash
docker compose run --rm ct-v16-measurement-quality
```

The output is `MEASUREMENT_v16_quality_enhanced_16bit.tif`. Preserve v15 as the traceable baseline for cross-checking.

## Limitation

v16 deliberately does not further smooth lamella bodies, so visible improvement there is limited. This is the safe result of the measurement constraints, not a failed optimization. Demonstrating substantially stronger lamella denoising without subpixel bias requires repeated scans, paired high-dose ground truth, or a calibrated reference object from the same imaging system.
