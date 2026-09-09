# CT Generative Blind Restore

A focused, reproducible enhancement pipeline for industrial CT images with two
lamella stacks and a central block. This release freezes the workflow that
produced the accepted result: the external generative candidate is resized to
the native source canvas **immediately after generation**, then a blind-denoised
measurement guide supplies ridge, endpoint, width, and spacing constraints for
a bounded continuous 2-D warp.

The current code replaces the historical v1-v21 experiment chain. It has one
production entry point, one geometry profile, auditable intermediate files,
CPU/CUDA containers, and tests.

## Processing contract

```text
raw 16-bit CT ───────────────┐
                            ├─ blind-denoised guide ─ measure native geometry ─┐
external generated image ─ source-size lock ───────────────────────────────────┤
                                                                                └─ bounded 2-D structure guidance
                                                                                   ├─ final PNG
                                                                                   ├─ final 16-bit TIFF
                                                                                   └─ metrics + audit images
```

Key invariants:

- Output width and height equal the raw source exactly; there is no crop or padding.
- Size correction happens before structure guidance or any later processing.
- The central block and pixels outside the two lamella masks are bit-identical to
  the source-sized generated candidate.
- The accepted candidate supplies all displayed intensity values. Raw/guide
  intensities are never copied into the final image.
- The selected warp is searched over conservative strengths and must pass SSIM,
  writable-mask, ridge-median, and ridge-P95 guardrails.
- If every nonzero warp would worsen an already aligned candidate, the selector
  records and returns an exact strength-zero identity result instead of forcing
  a harmful displacement.
- The bundled reference run selected strength `0.16`, corresponding to about
  `0.8 px` maximum applied displacement in either axis.

> The output is a measurement-assist image, not calibrated ground truth. Keep the
> raw CT, blind guide, geometry profile, and manifests with each result.

## Required inputs

Place files in `input/`:

```text
input/
├── source_16bit.tif                 # native CT, single-channel integer image
├── generative_candidate.png         # external generator result; any size
└── measurement_guide_16bit.tif      # optional, native-size blind-denoised guide
```

The repository deliberately does not embed a remote image-generation API,
credentials, or proprietary weights. Any generator can be used if it follows
[the generator contract](docs/GENERATOR_INTERFACE.md). When `--guide` is omitted,
the included self-supervised blind model trains on the current CT image and
creates the guide automatically.

## Quick start with Docker

Fast reproducible path using a precomputed guide:

```bash
docker compose build enhance
docker compose run --rm enhance
```

Full path that trains the blind guide for this image (slower on CPU):

```bash
docker compose --profile train-guide run --rm enhance-auto-guide
```

Results are written to `output/`. To use CUDA, build `Dockerfile.cuda` and pass
`--blind-device cuda` to `app/run_pipeline.py`.

## Run with Python

Python 3.11 is recommended.

```bash
python -m pip install -r requirements-cpu.txt

python app/run_pipeline.py \
  --source input/source_16bit.tif \
  --generated input/generative_candidate.png \
  --guide input/measurement_guide_16bit.tif \
  --outdir output
```

Omit `--guide` to train the blind-denoised guide:

```bash
python app/run_pipeline.py \
  --source input/source_16bit.tif \
  --generated input/generative_candidate.png \
  --outdir output \
  --blind-iterations 600 \
  --blind-device auto
```

Use `--overwrite` only when intentionally replacing outputs previously managed
by this pipeline. Unrelated files in the output directory are preserved.

## Adapting to similar images

The default coordinates target the same acquisition layout as the accepted
sample and scale automatically from the `2200 × 1600` reference canvas. For a
different framing, copy and edit
[`config/reference_geometry.json`](config/reference_geometry.json), then pass:

```bash
--geometry-config config/my_geometry.json
```

The left/right body ROIs should contain the straight, measurable lamella bodies;
the top/bottom ranges should cover endpoints; and `central_lock_roi` must cover
the complete center block. See [method details](docs/METHOD.md).

## Outputs

```text
output/
├── 01_source_sized_generation/
├── 02_blind_guide/
├── 03_structure_guidance/
│   ├── AUDIT_measured_structure_overlay.png
│   ├── AUDIT_native_warp_write_mask.png
│   ├── STRUCTURE_WARP_GUIDED_comparison.png
│   └── native_structure_warp_metrics.json
├── FINAL_enhanced_<width>x<height>.png
├── FINAL_enhanced_<width>x<height>_16bit.tif
└── RUN_MANIFEST.json
```

`RUN_MANIFEST.json` records inputs, hashes, dimensions, stage order, selected
candidate, geometry metrics, and invariants.

## Tests

```bash
python -m unittest discover -s tests -v
```

Container validation and golden-sample reproduction are documented in
[docs/VALIDATION.md](docs/VALIDATION.md). A Chinese guide is available in
[README_ZH.md](README_ZH.md).

## Method provenance

The blind stage is a clean-room CT adaptation of the Blind2Sound idea with
Poisson-Gaussian noise estimation, self-supervised re-visible masking, low-frequency
generative guidance, and geometry guardrails. “SOTA” here describes the selected
research family and adaptation; it is not a universal benchmark claim. The final
structure step is project-specific and deterministic.
