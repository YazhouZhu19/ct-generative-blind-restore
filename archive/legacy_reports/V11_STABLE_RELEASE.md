# v11 Stable Release

This file freezes v11 as the selected guide-first generative method. It does not use registration. Later v12/v13 experiments and the rejected v14 registration experiment remain in the repository for research comparison, but they do not replace this profile.

## Preserved pipeline

1. Blind-denoise the raw image without registration or any coordinate transform.
2. Measure 49 left and 51 right lamellae, including center paths, subpixel endpoints, finite FWHM, and 98 row-wise interlayers, on that guide.
3. Cross-check the measurements against the raw image without averaging noisy raw measurements back into the guide coordinates.
4. Generate a visual candidate from the raw target, v11 geometry condition, and unregistered appearance reference.
5. Immediately resize the native generator output to the source width and height with deterministic Lanczos resampling.
6. Run the original deterministic visual post-processing entirely on the source-sized grid; for the supplied image, generated input, source, guide, projection input, and final output are all `2200×1600`.
7. Remove generator-invented periodicity and rebuild every lamella with the v11 finite-width dual-logistic profile.
8. Perform three per-lamella FWHM calibration rounds and write dimension and geometry audits.

The frozen v11 profile uses a `0.38 px` transverse transition, a `0.55 px` endpoint transition, and no direct blind-guide texture fusion (`guide_detail_weight=0`).

Registration is explicitly disabled: no translation, affine, piecewise, or
deformable transform is estimated or applied. The release manifest records
`registration.enabled=false` and `transform_applied=false`.

The provider candidate may retain its native dimensions for provenance. In the supplied experiment it is `1470×1070`; the runner first creates `GENERATIVE_denoised_guide_shape_conditioned_2200x1600.png`, then performs every enhancement operation at `2200×1600`. The stage order is recorded in `generation_preprocess_size_manifest.json`, and the projector asserts that its input and output arrays exactly match the source shape.

## One-command reproduction after generation

The external generative candidate cannot be deterministically recreated by the local container. Once that candidate exists, all geometry measurement, post-processing, projection, and auditing are reproducible with:

```bash
python app/run_v11_pipeline.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated input/GENERATIVE_denoised_guide_shape_conditioned_raw.png \
  --outdir results_generative_shape_v11
```

Container equivalent:

```bash
docker run --rm --entrypoint python \
  -v "$PWD:/workspace" -w /workspace \
  ct-generative-blind-restore:cpu \
  app/run_v11_pipeline.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated input/GENERATIVE_denoised_guide_shape_conditioned_raw.png \
  --outdir results_generative_shape_v11
```

Parameters and expected metrics are frozen in [`config/v11_profile.json`](config/v11_profile.json). The complete technical explanation is in the [English](GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT_EN.md) and [Chinese](GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT.md) reports.

Machine-readable proof of the container test and exact archived/reproduced pixel comparison is stored in [`config/v11_reproduction_validation.json`](config/v11_reproduction_validation.json).

Per-run dimension evidence is also written to `02_postprocessed/size_lock_manifest.json`, `03_hard_shape_projection/hard_shape_projection_metrics.json`, and the root `v11_release_manifest.json`.

## Supplied-image reference result

| Metric | Preserved v11 result |
|---|---:|
| Lamellae | 49 left + 51 right |
| Interlayers | 98 |
| Constraint-matched lamellae | 100 / 100 |
| Endpoint shift P95 | 0.0856 px |
| Length change P95 | 0.1006 px |
| FWHM relative error median | 0.0182% |
| FWHM relative error P95 | 0.2010% |
| Target/output median FWHM | 4.251057 / 4.251983 px |

## Repository-safe contents

The Git repository contains code, container files, tests, configuration, and numeric reports. Raw TIFFs, references, model weights, and generated results remain ignored to avoid committing data. A separate local full-experiment archive may include those data files.

## Scope boundary

The final v11 image contains generated pixels. Its geometry is analytically constrained, but it remains visual-only and is not a metrology-certified measurement image or clean ground truth.
