# 16-Bit CT Self-Supervised Blind Denoising and Fidelity-Preserving Enhancement

This repository contains a containerized workflow for single-image self-supervised blind denoising, measurement-aware edge enhancement, and optional lamella-length analysis on 16-bit industrial CT/X-ray images.

For the full method, equations, parameters, quality assurance, and code structure, see [`METHOD_AND_CODE_GUIDE.md`](METHOD_AND_CODE_GUIDE.md).

## Bilingual Technical Innovation Reports

- [English Markdown report](reports/technical_innovation/TECHNICAL_INNOVATION_REPORT_EN.md)
- [English self-contained HTML report](reports/technical_innovation/TECHNICAL_INNOVATION_REPORT_EN.html)
- [Chinese Markdown report](reports/technical_innovation/TECHNICAL_INNOVATION_REPORT.md)
- [Chinese self-contained HTML report](reports/technical_innovation/TECHNICAL_INNOVATION_REPORT.html)

The report directory also contains reproducible Chinese and English `artifact*.json` files, SQLite snapshots, source SQL, validation receipts, and three numeric-only experiment manifests.

> **Data safety:** this repository contains code, container configuration, technical documentation, and numeric validation evidence only. Raw TIFF files, reference images, generated/enhanced images, model weights, and runtime result directories are excluded by `.gitignore`. Provide image data locally or through read-only mounts.

## Measurement-Safety Architecture

The project preserves two deliberately separated output domains:

1. `GENERATIVE_*` contains generative visual candidates and deterministic post-processing outputs. These files are for visual exploration only and must not be used for thickness measurement, defect acceptance, or ground truth.
2. `MEASUREMENT_*` contains outputs derived from the original 16-bit image through adjacent-pixel replacement, Noise2Self-style masked prediction, multi-mask uncertainty estimation, and edge/data-consistency gating. Residual and uncertainty maps are emitted for audit.

No generated pixel is allowed into the `MEASUREMENT_*` pipeline. The reference JPEG is not used as a pixel-level training target.

This implementation is an engineering adaptation for the supplied 16-bit grayscale image. It does not claim to reproduce the complete APR-RD, Blind2Sound, or FoundIR-v2 methods.

## Recommended Quality-Optimization Workflow

Run the single-image blind denoiser first, followed by constrained directional edge and local-contrast enhancement:

```bash
docker compose run --rm ct-restore-cpu
docker compose run --rm ct-quality
```

`ct-quality` searches edge-gain and structure-gain parameters automatically. Candidates must satisfy guardrails for lamella-count stability, center displacement, FWHM change, flat-region high-frequency roughness, and SSIM before being ranked for clarity.

The recommended result is:

```text
results_quality/QUALITY_balanced_16bit.tif
```

It retains the original 2200 × 1600 dimensions and 16-bit grayscale representation and contains no generative pixels.

## Output Reference

### Blind-Denoising Outputs

- `results/MEASUREMENT_blind_denoised_16bit.tif` — the only denoised candidate eligible for subsequent calibration and measurement validation.
- `results/MEASUREMENT_residual_float32.tif` — processed result minus the raw observation; inspect it for removed structure.
- `results/MEASUREMENT_uncertainty_float32.tif` — standard deviation across masked predictions; high-value regions require review.
- `results/run_manifest.json` — method, parameters, lamella count, FWHM, displacement, SSIM, and guardrail results.
- `results/GENERATIVE_visual_only_postprocessed*.{png,tif}` — visual-only generative results; never use them for measurement.
- `results/comparison.png` — raw, measurement-safe denoised, and generative visual candidates side by side.

### Quality-Optimization Outputs

- `results_quality/QUALITY_balanced_16bit.tif` — preferred full-resolution result for edge clarity, overall quality, and denoising.
- `results_quality/QUALITY_balanced_preview.png` — 8-bit preview of the preferred result.
- `results_quality/QUALITY_display_only.png` — local-contrast display mapping; quantitative intensities are not preserved.
- `results_quality/QUALITY_comparison.png` — enlarged comparison of the raw image, blind-denoised baseline, optimized result, and display-only version.
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

The length module is retained but is not part of the default quality workflow. Run it only when length analysis is required, after blind denoising:

```bash
docker compose run --rm ct-length
```

The module protects the upper and lower endpoints, performs bounded zero-phase symmetric sharpening, tracks the curved center path of each lamella row by row, and fits subpixel endpoints along that path. Its outputs are written to `results_length/`:

- `MEASUREMENT_length_optimized_16bit.tif` — visual candidate for length inspection.
- `layer_lengths.csv` — recommended pixel length, endpoints, internal uncertainty, and quality flag for each lamella.
- `layer_length_overlay_roi.png` — green indicates pass; red indicates manual review.
- `length_qa.json` — aggregate accuracy and data-quality checks.

If a reference standard provides a calibrated pixel size—for example, `0.012 mm` per pixel—run:

```bash
docker compose run --rm --entrypoint python ct-restore-cpu \
  app/length_optimize.py \
  --source /data/input/source_16bit.tif \
  --denoised /data/results/MEASUREMENT_blind_denoised_16bit.tif \
  --outdir /data/results_length \
  --pixel-size 0.012 --unit mm
```

Do not infer physical pixel size from the displayed TIFF dimensions. The current source file contains no XResolution, YResolution, or ResolutionUnit metadata.

## Why FoundIR-v2 Is Not Used as the Measurement Image

FoundIR-v2 relies on large-model components such as SDXL and LLaVA, and its official inference workflow targets one or two CUDA GPUs. It is a general image-restoration model rather than a system calibrated for this industrial CT modality, 16-bit intensity domain, or scanner PSF.

It may produce exceptionally clear-looking lamellae, but visual clarity is not measurement truth. Adding, deleting, duplicating, or moving even one layer invalidates thickness conclusions. The project therefore accepts an arbitrary generative result through `--generated` for visual-only post-processing, but never mixes it into a `MEASUREMENT_*` output.

## Validation Environment and Limits

The original delivery host was an 8 GB Apple Silicon machine without Docker, Podman, or OrbStack. Container definitions were generated, while the core scripts were validated with a local PyTorch CPU environment using the same dependency set. The workflow can be rerun directly on a Docker-capable machine.

The saved experiment is a single-image internal validation, not a cross-device benchmark or metrology certification. Absolute millimetre or micrometre accuracy still requires calibrated pixel size, a reference standard, and system PSF/MTF characterization.
