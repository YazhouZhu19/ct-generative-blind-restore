# Native-Structure Guidance from the Accepted Size-Corrected Generation

## Scope

This experiment intentionally branches from one frozen input: the accepted
`2200x1600` result immediately after generative output size correction. No
later enhancement, post-processing, registration result, hard projection, or
v12-v21 measurement candidate is used as an image input.

The goal is narrow: improve agreement between the left/right generated edge
structure and the original-image evidence while preserving the accepted
denoising appearance and exact output dimensions.

## Inputs and authority

- Raw 16-bit source: independent evidence and measurement validation.
- Registered blind-denoised guide: authoritative source for ridge paths,
  finite-width edges, and top/bottom endpoints.
- Accepted source-sized generation: the only source of output pixel intensity.

The guide and raw image affect coordinates only. Their intensity pixels are
never copied into the result.

## Method

1. Measure 49 left and 51 right guide ridges on the original `2200x1600`
   coordinate grid. Track each curved centerline and its subpixel endpoints.
2. Detect and track the corresponding ridges in the accepted generation.
3. Use order-preserving dynamic programming to match every measured ridge.
   Extra generated detections may be skipped, but a guide constraint may not be
   silently dropped when the generated count is sufficient.
4. Interpolate matched ridge paths and endpoints into one continuous local
   two-axis inverse-sampling field. Identity anchors outside each stack keep the
   surrounding canvas fixed.
5. Cap the full displacement at 5 px, then search soft application strengths
   `0.12`, `0.16`, `0.20`, and `0.24`. Candidate selection is performed after
   uint16 quantization and checks source size, outside-mask identity, global
   SSIM, and measured ridge-position non-regression.
6. Bilinearly sample only from the accepted generation. No analytic lamella is
   rendered and no guide/raw detail residual is injected.

## Selected single-image result

The selected strength is `0.16`. Consequently, the actual applied displacement
is tightly bounded:

- x displacement: median `0.79999 px`, P95 `0.80005 px`, maximum `0.80005 px`;
- y displacement: median `0.48987 px`, P95 `0.79999 px`, maximum `0.80005 px`;
- output dimensions: exactly `2200x1600`;
- maximum change outside the writable lamella masks: `0`;
- maximum change in the protected central block: `0`.

Against the same-coordinate blind guide, the ridge-center absolute-offset
median changes from `2.6267 px` to `2.6137 px`, and P95 changes from
`4.9283 px` to `4.9222 px`. The gradient-magnitude correlation rises from
`0.02449` to `0.03710`. The measured lamella-width relative-error median falls
from `32.35%` to `25.29%`; P95 falls from `59.75%` to `58.81%`.

Endpoint median error changes from `4.4915 px` to `4.4765 px`; endpoint P95 is
`7.1496 px` versus `7.0676 px` for the accepted generation. This small tail
difference is explicitly reported rather than hidden. The image is therefore a
visual/measurement-assist candidate, not calibrated metrology ground truth.

## Reproduction

```bash
python app/native_structure_warp.py \
  --source input/source_16bit.tif \
  --guide results_sota_v8/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated results_generative_shape_v11/01_generated/GENERATIVE_denoised_guide_shape_conditioned_2200x1600.png \
  --outdir results_native_structure_warp
```

The implementation refuses mismatched dimensions instead of resizing the
generation. Its machine-readable audit is `native_structure_warp_metrics.json`.
Use the 16-bit TIFF for archival inspection and the PNG/comparison/overlay for
visual review.

## Limitations

- This is a single-image internal experiment without a calibrated clean target.
- The generator and guide render some plate phases differently, so signed
  band-pass correlation is an appearance diagnostic, not an absolute geometry
  criterion.
- Formal length/thickness reporting should continue to use raw/guide-derived
  numeric constraints, calibrated pixel size, and PSF/MTF validation.
