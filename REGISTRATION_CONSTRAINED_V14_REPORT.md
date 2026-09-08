# Constrained Registration for Geometry-Guided Generative CT Enhancement (v14)

> **Status: rejected and inactive.** Visual evaluation found the registered
> result too soft. This document and the v14 code are retained only as an
> audit/research record; the selected v11 release does not call registration.

## Purpose

The v11 hard projection accurately restores measured lamella centerlines,
endpoints, FWHM widths, and interlayer spacing, but it does not register the
generator's full component envelope. Generator-induced scale differences can
therefore remain at the extreme left and right side walls. v14 adds an explicit
registration stage while keeping the v11 measurement constraints authoritative.

## Stage order

1. Measure geometry on the blind-denoised guide in unchanged raw coordinates.
2. Resize the native generated candidate to the exact source canvas.
3. Register only the generated candidate to the blind guide.
4. Run the existing visual denoising and enhancement post-processing.
5. Hard-project every measured lamella and audit width, endpoints, and length.

Registration before post-processing avoids resampling sharpened edges. Hard
projection after registration removes any metrology dependence on the warped
synthetic stripes.

## Registration method

`app/generative_registration.py` implements a coarse-to-fine, ROI-restricted
registration:

- A low-frequency gradient representation suppresses noise and stripe texture.
- Phase correlation estimates a coarse translation inside the component ROI.
- Robust 80th-percentile profiles locate six structural landmarks: left outer,
  left inner, right inner, right outer, top, and bottom envelopes.
- A monotone piecewise-affine inverse map aligns these landmarks. Cubic
  interpolation is used only inside a feathered component ROI.
- Pixels outside the ROI remain unchanged.

This is deliberately lower freedom than dense optical flow or unconstrained
non-rigid registration. It can correct group-level generation drift without
allowing arbitrary local warps that could disguise a dimensional defect.

## Safety guardrails

- Source, guide, moving candidate, and output dimensions must be identical.
- Horizontal segment scales and vertical scale must stay within `0.88..1.12`.
- The absolute coarse translation is limited to `120 px` per axis.
- Post-registration envelope P95 error must be no greater than `2 px` and must
  improve over the unregistered candidate.
- The registered image remains visual-only. Numerical measurements continue to
  come from the raw/guide constraint tables.
- The existing hard-projection width, endpoint, and length guardrails remain
  mandatory after registration.

## Audit outputs

- `02_registered/GENERATIVE_registered_to_guide.png`
- `02_registered/GENERATIVE_registered_to_guide_16bit.tif`
- `02_registered/REGISTRATION_comparison.png`
- `02_registered/REGISTRATION_envelope_overlay.png`
- `02_registered/registration_metrics.json`
- `04_hard_shape_projection/hard_shape_projection_metrics.json`
- `FINAL_horizontal_envelope_audit.json`

Green and red lines in the envelope overlay represent the guide and registered
candidate respectively. Their overlap is an audit visualization, while the
JSON contains the subpixel landmark values and before/after errors.

## Current-image result

On the supplied image, the unregistered generated candidate had a six-landmark
envelope P95 error of `62.41 px`; the largest disagreement was predominantly
vertical, while the right outer boundary was displaced by about `39.27 px`.
Constrained registration reduced the six-landmark P95 error to `0.65 px` and
the maximum to `0.80 px`. After the unchanged visual post-processing and hard
projection, all 100 lamella constraints still passed, endpoint-shift P95 was
`0.065 px`, FWHM relative-error P95 was `0.197%`, and edge clarity improved by
`18.0%` over the raw image. The separate final horizontal-envelope audit
reported P95/max errors of `0.892 px`/`0.911 px`, verifying that the later
stages do not reintroduce the left/right scale error.

## Multi-algorithm comparison

All methods were inserted after source-size resize and before the unchanged
visual post-processing stage. The same fixed guide, ROI, landmark detector,
and metrics were used.

| Registration method | Envelope P95 (px) | Outer-edge gradient P10 | Decision |
|---|---:|---:|---|
| Phase-correlation translation | 27.67 | 0.251 | reject |
| Global affine | 5.72 | 0.366 | reject |
| PCHIP envelope | 0.72 | 0.325 | pass |
| Piecewise linear | 0.54 | 0.336 | pass |
| Piecewise cubic | 0.65 | 0.351 | pass |
| Piecewise quintic | 0.64 | 0.352 | pass |
| Piecewise edge-preserving | 0.84 | 0.362 | selected |

The selected method adds a bounded `0.35` high-frequency recovery term after
cubic spatial resampling. Its registered component-gradient P95 increased from
`0.218` to `0.235`, and the outer-edge gradient P10 increased from `0.255` to
`0.362`; registration therefore did not blur this candidate. After the
unchanged hard projection, the final horizontal-envelope P95 remained
`0.874 px`, endpoint P95 was `0.064 px`, FWHM relative-error P95 was `0.213%`,
and all 100 lamella constraints passed.

The final projected results from cubic and edge-preserving registration are
visually and numerically close because hard projection replaces most
generator-frequency content with analytic measured ribbons. Consequently,
additional final-image sharpness work should target the projection background
and periodicity-suppression design rather than increasing registration freedom.

## Reproduction

```bash
python app/run_v14_pipeline.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated path/to/native_generated_candidate.png \
  --outdir results_generative_shape_v14_registration
```

The final PNG and 16-bit TIFF keep the original source dimensions. They are
quality-enhanced visualization products, not substitutes for measurements made
on the original or measurement-safe blind-denoised data.
