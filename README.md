# Generative-Prior Blind CT Denoising with Geometry-Preserving Enhancement

This repository contains a containerized workflow for single-image blind denoising, measurement-aware enhancement, and lamella-length analysis on 16-bit industrial CT/X-ray images. The current v8 path combines an adaptive re-visible Poisson-Gaussian blind model, a safely clipped generative low-frequency prior, geometry losses during training, transverse coordinate anchors during data projection, geometry-locked strong axial denoising, the complete original enhancement/post-processing chain, boundary cleanup, hybrid residual denoising, and an independent fixed-guide per-lamella audit.

For the latest work, see the [English v8 report](GEOMETRY_LOCKED_DIRECTIONAL_V8_REPORT_EN.md) or [Chinese v8 report](GEOMETRY_LOCKED_DIRECTIONAL_V8_REPORT.md). The v7 residual stage remains documented in its [English](RESIDUAL_DENOISE_V7_REPORT_EN.md) and [Chinese](RESIDUAL_DENOISE_V7_REPORT.md) reports. Boundary cleanup is documented in the v6 reports. The complete legacy-to-current code guide is in [`METHOD_AND_CODE_GUIDE.md`](METHOD_AND_CODE_GUIDE.md).

The actual Docker Desktop build and end-to-end smoke run are recorded in [`CONTAINER_VALIDATION.md`](CONTAINER_VALIDATION.md).

The preserved generative baseline is **v11 stable without registration**. No translation, affine, piecewise, or deformable warp is applied. Its frozen entry point, parameters, expected metrics, package boundary, and reproduction commands are documented in the [English stable-release guide](V11_STABLE_RELEASE.md) and [Chinese stable-release guide](V11_STABLE_RELEASE_ZH.md). The technical method is documented in the [English v11 report](GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT_EN.md) and [Chinese v11 report](GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT.md).

For lamella and interlayer measurement, the latest recommended architecture is **v15 dual output**. It emits a clean generative companion for visual inspection and a separate full-resolution measurement-assist image whose pixel contribution from the generator is exactly zero. The latter retains every local structure from the same-coordinate blind-denoised guide and never analytically redraws a lamella. See the [English v15 report](STRUCTURE_CARRIER_DUAL_OUTPUT_V15_REPORT_EN.md) and [Chinese v15 report](STRUCTURE_CARRIER_DUAL_OUTPUT_V15_REPORT.md).

The organized code, documentation, runtime-data boundaries, and release archives are indexed in [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

Versions v12 and v13 remain available as research profiles; they do not alter the frozen v11 profile or archived v11 result. v15 reuses a tuned v13 projection only for its explicitly named `VISUAL_ONLY` companion. See the [v12 English](GENERATIVE_RAW_DUAL_EVIDENCE_V12_REPORT_EN.md), [v12 Chinese](GENERATIVE_RAW_DUAL_EVIDENCE_V12_REPORT.md), [v13 English](GENERATIVE_BLIND_GUIDE_DETAIL_V13_REPORT_EN.md), and [v13 Chinese](GENERATIVE_BLIND_GUIDE_DETAIL_V13_REPORT.md) reports. Generative pixels are never used in the v15 measurement output.

For the preserved v11 result, guide-first measurement finds 49 left and 51 right lamellae plus 98 interlayers. All 100 constraints pass; endpoint-shift P95 is `0.0856 px`, length-change P95 is `0.1006 px`, and FWHM-error median/P95 are `0.0182%`/`0.2010%` after resizing the generator output before post-processing.

## Recommended v15 Dual-Output Workflow for Measurement

The central design change is that visual quality and measurement fidelity are no longer forced into the same pixels:

```text
source-sized generation -> deterministic post-processing -> VISUAL_ONLY companion
blind-denoised guide -------------------------------------> MEASUREMENT structure carrier
```

Run the full pipeline with:

```bash
docker compose run --rm ct-v15-dual
```

or directly:

```bash
python app/run_v15_pipeline.py \
  --source input/source_16bit.tif \
  --guide input/measurement_guide_16bit.tif \
  --generated input/generative_candidate_visual_only.png \
  --outdir results_generative_shape_v15_dual_output
```

The two final products are:

- `FINAL_VISUAL_ONLY_enhanced_2200x1600_16bit.tif`: stronger denoising and display clarity; generated pixels remain and the file must not be measured.
- `FINAL_MEASUREMENT_structure_preserved_2200x1600_16bit.tif`: full 2200×1600 blind-guide structural carrier; generator weight is zero and no warp, resize, intensity remapping, or analytic lamella replacement occurs after the guide is formed.

On the supplied image, the measurement branch retained 100 lamellae and 98 interlayers. Lamella-width median/P95 error against the guide was `0% / 0%`, endpoint P95 error was `0 px`, interlayer-width P95 error was `0%`, and interlayer-length P95 error was `0.080 px`. Low-, mid-frequency, gradient, and per-lamella axial-detail correlations were effectively `1.0`. The visual companion retained a `17.6%` edge-clarity gain over the raw image, while its generated pixels remain explicitly excluded from metrology.

## Preserved v11 Guide-First Generative Workflow (Selected)

The active order is deliberately registration-free:

```text
generation -> source-size resize -> visual post-processing
           -> hard geometry projection
```

For the container entry point, place the source, blind-denoised measurement
guide, and generated candidate under the following names, then run:

```bash
docker compose run --rm ct-v11-generative
```

```text
input/source_16bit.tif
input/measurement_guide_16bit.tif
input/generative_candidate_visual_only.png
```

The resulting `v11_release_manifest.json` explicitly records
`registration.enabled=false` and `transform_applied=false`.

```bash
python app/generative_shape_constraint.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --outdir results_generative_shape_v11/00_guide_constraints

# Supply the raw source, generated condition map, and appearance reference to
# a generative editor; then run the existing visual post-processing function.

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

For the saved experiment, guide-first measurement found 49 left and 51 right lamellae and 98 interlayers. All 100 constraints passed, raw/guide endpoint disagreement P95 was `0.184 px`, final endpoint-shift P95 was `0.0856 px`, median FWHM error was `0.0182%`, and FWHM-error P95 was `0.2010%`.

For a single command that recreates the constraints, archives the supplied soft candidate, runs post-processing, and performs the frozen projection, use `app/run_v11_pipeline.py` as documented in [`V11_STABLE_RELEASE.md`](V11_STABLE_RELEASE.md).

The runner resizes the native generated candidate to the source dimensions **before** deterministic post-processing. Consequently, the source-sized generated candidate, all post-processing operations, the projection input, final PNG, and final 16-bit TIFF use the same `2200×1600` coordinate grid. The provider-native `1470×1070` file remains archived for provenance only. The order and dimensions are recorded in `generation_preprocess_size_manifest.json`, `size_lock_manifest.json`, and the projection audit.

## Rejected v14 Registration Research (Inactive)

> **Status:** This path is retained only as an audit/research record. Visual
> evaluation found the registered result too soft, so it is not part of the
> selected workflow and no default container service invokes it.

The optional v14 path addresses generator-induced disagreement at the extreme
left and right component boundaries. It preserves v11 and inserts registration
after the native generator output is resized to the source canvas, but before
visual post-processing and hard geometry projection:

```text
generation -> source-size resize -> constrained registration
           -> visual post-processing -> hard geometry projection
```

The registration stage uses gradient phase correlation inside the component
ROI for a coarse translation prior. It then detects the left outer, left inner,
right inner, and right outer guide envelopes plus the common top/bottom
envelope and fits a monotone piecewise-affine map. Scale and translation
guardrails reject unsafe solutions. Only the generated visual candidate is
warped; the raw/guide constraints remain in original source coordinates and
are reapplied by the existing hard projector.

```bash
python app/run_v14_pipeline.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated results_generative_shape_v11/01_generated/GENERATIVE_denoised_guide_shape_conditioned_raw.png \
  --outdir results_generative_shape_v14_registration
```

To reproduce the rejected experiment explicitly, opt into its Compose profile:

```bash
docker compose --profile registration-experimental run --rm ct-v14-registration
```

```text
input/source_16bit.tif
input/measurement_guide_16bit.tif
input/generative_candidate_visual_only.png
```

The current validation image reduced six-landmark envelope P95 error from
`62.41 px` before registration to `0.65 px` immediately afterward. The final
hard-projected image retained horizontal-envelope P95 error below `1 px`, while
all 100 lamella constraints passed.

Seven registration variants are now evaluated at exactly the same pre-
postprocessing position with `app/registration_benchmark.py`: phase-only
translation, global affine, PCHIP envelope, piecewise linear, piecewise cubic,
piecewise quintic, and piecewise edge-preserving registration. Phase-only and
global affine registration are rejected because their envelope P95 errors are
`27.67 px` and `5.72 px`. The selected edge-preserving piecewise method has a
registration P95 error of `0.84 px` and the strongest weak-side outer-edge
gradient among passing candidates. Its final horizontal-envelope P95 is
`0.874 px`; all 100 lamella constraints remain matched.

The comparison also shows that the apparent softness in the final image is not
introduced primarily by registration: the selected registered image has
higher component and outer-edge gradients than the unregistered candidate.
Most later smoothing comes from the existing hard-projection background and
periodicity-suppression stage, which deliberately replaces generator texture.

See [`REGISTRATION_CONSTRAINED_V14_REPORT.md`](REGISTRATION_CONSTRAINED_V14_REPORT.md)
for the method, guardrails, outputs, and interpretation.

## Optional v13 Research Comparison

Generate the soft candidate with four strictly separated inputs: raw target, source-coordinate blind-denoised detail guide, v11 constraint map, and unregistered appearance reference. Then run:

```bash
python app/generative_postprocess.py \
  --generated results_generative_shape_v13/01_generated/GENERATIVE_v13_blind_guide_detail_conditioned_raw.png \
  --outdir results_generative_shape_v13/02_postprocessed

python app/generative_shape_project.py \
  --profile v13 \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated results_generative_shape_v13/02_postprocessed/GENERATIVE_visual_only_postprocessed.png \
  --outdir results_generative_shape_v13/03_detail_consistent_projection \
  --detail-guide-weight 0.55
```

The v13 projector uses guide structure for low/mid-frequency foreground detail and bounded per-lamella axial modulation. It is an optional comparison and does not alter the default v11 profile.

The preceding v4 boundary-accuracy update remains documented in the [English boundary-preservation report](BOUNDARY_PRESERVATION_REPORT_EN.md) and [Chinese boundary-preservation report](BOUNDARY_PRESERVATION_REPORT.md).

## Bilingual Technical Innovation Reports

- [English Markdown report](reports/technical_innovation/TECHNICAL_INNOVATION_REPORT_EN.md)
- [English self-contained HTML report](reports/technical_innovation/TECHNICAL_INNOVATION_REPORT_EN.html)
- [Chinese Markdown report](reports/technical_innovation/TECHNICAL_INNOVATION_REPORT.md)
- [Chinese self-contained HTML report](reports/technical_innovation/TECHNICAL_INNOVATION_REPORT.html)

The report directory also contains reproducible Chinese and English `artifact*.json` files, SQLite snapshots, source SQL, validation receipts, and three numeric-only experiment manifests.

> **Data safety:** this repository contains code, container configuration, technical documentation, and numeric validation evidence only. Raw TIFF files, reference images, generated/enhanced images, model weights, and runtime result directories are excluded by `.gitignore`. Provide image data locally or through read-only mounts.

## Measurement-Safety Architecture

The project preserves two deliberately separated output domains:

1. A generative candidate is converted to a clipped, low-frequency training prior only in raw low-gradient regions. It supplies no layer coordinates and none of its pixels are copied into a measurement output.
2. `MEASUREMENT_*` contains outputs derived from the original 16-bit image through 16-phase blind prediction, Poisson-Gaussian data projection, raw-coordinate geometry anchoring, a raw-x/y-gradient and endpoint-locked axial filter, bounded residual selection, and explicit width/endpoint/length guardrails. Residual and uncertainty maps are emitted for audit.

No generated pixel is written into a `MEASUREMENT_*` image. The reference JPEG is not a registered pixel-level target.

The blind formulation is a clean-room CT adaptation inspired by [Blind2Sound (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Blind2Sound_Self-Supervised_Image_Denoising_without_Residual_Noise_ICCV_2025_paper.html), selected as a recent peer-reviewed, reproducible best fit for single-channel blind denoising. It is not the authors' official implementation and is not claimed to be universally strongest on every modality.

## Recommended v8 Workflow

Place `source_16bit.tif` and `generative_candidate_visual_only.png` in `input/`, then run the complete five-stage chain:

```bash
docker compose run --rm ct-sota
```

Equivalent native command:

```bash
python app/run_sota_pipeline.py \
  --source input/source_16bit.tif \
  --generative-prior input/generative_candidate_visual_only.png \
  --outdir results_sota --iterations 600 --device cpu
```

An optional unregistered appearance reference can be recorded with `--reference-style input/reference_style.jpg`. It is never used for registration, geometry, or output pixels.

The recommended result is `results_sota/04_residual_denoise/QUALITY_v7_residual_denoised_16bit.tif` (the residual module retains its compatible v7 filename). In a v8 run, the file receives the newly geometry-locked axial result. Check `directional_projection_selection` in `01_sota_blind/sota_run_manifest.json`, then the quality, cleanup, residual, and fixed-guide length audit manifests before measurement use.

## Legacy v4 Workflow

Run the single-image blind denoiser first, followed by constrained directional edge and local-contrast enhancement:

```bash
docker compose run --rm ct-restore-cpu
docker compose run --rm ct-quality
```

`ct-quality` first runs the original edge/structure post-processing and selects its parameters with the original appearance guardrails. Only afterward, a geometry stage applies a conservative local sub-pixel displacement to the already enhanced pixels. A final weak, edge-gated cleanup suppresses residual fine grain and mid-scale fog in low-structure regions. The raw image supplies endpoint coordinates but no raw pixels are copied back. Final guardrails cover transverse FWHM, per-lamella endpoints, length drift, edge/contrast retention, and SSIM against the geometry-only stage.

The recommended result is:

```text
results_quality/QUALITY_boundary_preserved_16bit.tif
```

It retains the original 2200 × 1600 dimensions and 16-bit grayscale representation and contains no generative pixels.

## v8 Output Reference

- `results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif` — blind posterior plus selected geometry-locked axial denoising, before the original enhancement chain.
- `results_sota/01_sota_blind/AUDIT_blind_prediction_16bit.tif` — unprojected 16-phase blind prediction for audit only.
- `results_sota/01_sota_blind/AUDIT_safe_generative_prior_16bit.tif` — clipped low-frequency prior for audit only.
- `results_sota/01_sota_blind/MEASUREMENT_sota_uncertainty_float32.tif` — learned predictive uncertainty.
- `results_sota/02_geometry_quality/QUALITY_boundary_preserved_16bit.tif` — complete v5 enhancement result and input to v6 cleanup.
- `results_sota/02_geometry_quality/QUALITY_boundary_overlay.png` — raw/enhanced endpoint overlay.
- `results_sota/03_boundary_cleanup/QUALITY_boundary_clean_16bit.tif` — recommended full-resolution 16-bit v6 result.
- `results_sota/03_boundary_cleanup/QUALITY_boundary_cleanup_comparison.png` — v5/v6/delta comparison.
- `results_sota/03_boundary_cleanup/AUDIT_boundary_cleanup_masks.png` — raw-coordinate boundary/exterior support audit.
- `results_sota/03_boundary_cleanup/boundary_cleanup_metrics.json` — parameter search and cleanup/geometry guardrails.
- `results_sota/04_residual_denoise/QUALITY_v7_residual_denoised_16bit.tif` — recommended full-resolution 16-bit v8 pipeline result; filename retained for residual-stage compatibility.
- `results_sota/04_residual_denoise/QUALITY_v7_comparison.png` — pre/post residual-denoising and absolute-delta comparison.
- `results_sota/04_residual_denoise/AUDIT_v7_residual_denoise_masks.png` — lamella, central-highlight, and endpoint-protection audit.
- `results_sota/04_residual_denoise/residual_denoise_metrics.json` — v7 search and geometry/appearance guardrails.
- `results_sota/05_length_audit/layer_lengths.csv` — independent fixed-guide per-lamella audit.

## Legacy Output Reference

### Blind-Denoising Outputs

- `results/MEASUREMENT_blind_denoised_16bit.tif` — the only denoised candidate eligible for subsequent calibration and measurement validation.
- `results/MEASUREMENT_residual_float32.tif` — processed result minus the raw observation; inspect it for removed structure.
- `results/MEASUREMENT_uncertainty_float32.tif` — standard deviation across masked predictions; high-value regions require review.
- `results/run_manifest.json` — method, parameters, lamella count, FWHM, displacement, SSIM, and guardrail results.
- `results/GENERATIVE_visual_only_postprocessed*.{png,tif}` — visual-only generative results; never use them for measurement.
- `results/comparison.png` — raw, measurement-safe denoised, and generative visual candidates side by side.

### Quality-Optimization Outputs

- `results_quality/QUALITY_boundary_preserved_16bit.tif` — preferred full-resolution result after boundary correction and guarded residual-fog cleanup.
- `results_quality/QUALITY_original_enhancement_16bit.tif` — unchanged output of the original enhancement and post-processing chain, saved for audit.
- `results_quality/QUALITY_geometry_only_16bit.tif` — enhanced result after geometry correction but before residual-fog cleanup.
- `results_quality/QUALITY_balanced_16bit.tif` — backward-compatible alias containing the same selected pixels.
- `results_quality/QUALITY_balanced_preview.png` — 8-bit preview of the preferred result.
- `results_quality/QUALITY_display_only.png` — local-contrast display mapping; quantitative intensities are not preserved.
- `results_quality/QUALITY_comparison.png` — enlarged comparison of the raw image, blind-denoised baseline, unchanged original enhancement, and final v4 result.
- `results_quality/QUALITY_fog_cleanup_closeup.png` — fixed-window geometry-stage/final close-up plus an auto-scaled absolute-change audit.
- `results_quality/QUALITY_boundary_overlay.png` — green raw and red enhanced endpoint detections; orange marks uncertain raw references requiring review.
- `results_quality/boundary_geometry.csv` — per-lamella raw/enhanced endpoints, length deltas, uncertainty, and reference-quality flag.
- `results_quality/quality_metrics.json` — parameter search, quality gains, and structural guardrails.

## Docker CPU: macOS or Hosts Without NVIDIA GPUs

```bash
docker compose run --rm ct-restore-cpu
```

Equivalent manual commands:

```bash
docker build -f Dockerfile.cpu -t ct-generative-blind-restore:cpu .
docker run --rm \
  -v "$PWD/input:/data/input:ro" \
  -v "$PWD/results:/data/results" \
  ct-generative-blind-restore:cpu \
  --input /data/input/source_16bit.tif \
  --outdir /data/results --device cpu
```

## Docker CUDA: NVIDIA Hosts

NVIDIA Container Toolkit is required:

```bash
docker compose --profile cuda run --rm ct-restore-cuda
```

The default run performs 600 single-image training iterations. For a quick smoke test, append `--iterations 120 --passes 4`. Production review should use at least 600 iterations and inspect `guardrail_pass`, the residual image, and the uncertainty map.

## Optional Length Optimization and Per-Lamella Measurement

The length module is retained as an independent audit. Run it after `ct-quality`; its default `--sharpen-amount 0` mode does not change any pixels:

```bash
docker compose run --rm ct-length
```

The module tracks the curved center path of each lamella row by row and fits subpixel endpoints on both the raw and final enhanced images. Optional bounded zero-phase sharpening is available only when explicitly requested. Its outputs are written to `results_length/`:

- `MEASUREMENT_length_optimized_16bit.tif` — visual candidate for length inspection.
- `layer_lengths.csv` — recommended pixel length, endpoints, internal uncertainty, and quality flag for each lamella.
- `layer_length_overlay_roi.png` — green indicates pass; red indicates manual review.
- `length_qa.json` — aggregate accuracy and data-quality checks.

If a reference standard provides a calibrated pixel size—for example, `0.012 mm` per pixel—run:

```bash
docker compose run --rm ct-length \
  --source /data/input/source_16bit.tif \
  --denoised /data/results_quality/QUALITY_boundary_preserved_16bit.tif \
  --outdir /data/results_length \
  --sharpen-amount 0 --pixel-size 0.012 --unit mm
```

Do not infer physical pixel size from the displayed TIFF dimensions. The current source file contains no XResolution, YResolution, or ResolutionUnit metadata.

## Why FoundIR-v2 Is Not Used as the Measurement Image

FoundIR-v2 relies on large-model components such as SDXL and LLaVA, and its official inference workflow targets one or two CUDA GPUs. It is a general image-restoration model rather than a system calibrated for this industrial CT modality, 16-bit intensity domain, or scanner PSF.

It may produce exceptionally clear-looking lamellae, but visual clarity is not measurement truth. Adding, deleting, duplicating, or moving even one layer invalidates thickness conclusions. The project therefore accepts an arbitrary generative result through `--generated` for visual-only post-processing, but never mixes it into a `MEASUREMENT_*` output.

## Validation Environment and Limits

The v8 chain and fourteen unit tests were validated with a local PyTorch CPU environment using the dependency versions pinned for the container. The supplied Compose service reproduces the same five stages on a Docker-capable CPU host. A ten-iteration end-to-end Docker smoke run also completed with all blind, directional, width, endpoint, cleanup, residual-denoising, and independent-length guardrails passing.

The saved experiment is a single-image internal validation, not a cross-device benchmark or metrology certification. Absolute millimetre or micrometre accuracy still requires calibrated pixel size, a reference standard, and system PSF/MTF characterization.
