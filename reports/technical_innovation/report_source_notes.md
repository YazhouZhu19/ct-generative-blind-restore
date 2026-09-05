# Report source notes

## Reporting job

- Question: explain the algorithm as a technical report and foreground defensible innovation.
- Audience: technical readers evaluating image restoration and measurement integrity.
- Decision supported: whether the method is sufficiently novel, auditable and validated to justify broader experiments.
- Scope: the saved 2026-09-05 single-image experiment only.
- Comparison baselines: original 16-bit image, blind-denoised baseline and the quality-optimized candidate.
- Success criteria: exact metrics reconcile to saved JSON; innovation claims distinguish inherited methods from engineering adaptation; metrology limitations remain visible.

## Validation result

- Overall assessment: Share with caveats.
- Recomputed blind-stage continuity reductions: left 30.6027%, right 27.7185%.
- Recomputed blind-stage dropout reductions: left 53.3333%, right 76.0000%.
- Recomputed blind-stage FWHM drift: left 0.04894%, right 0.35829%.
- Recomputed quality-stage flat high-frequency reduction: 23.0259%.
- Recomputed quality-stage mean edge-acutance gain: 5.8659%.
- Recomputed quality-stage mean contrast gain: 2.9999%.
- Recomputed length pass fraction: 99/100 = 99%.
- All three saved guardrail flags are true.

## Required caveats

- One image is not an external benchmark or generalization study.
- No paired clean target, repeated scan, physical pixel size, standard sample or PSF/MTF is available.
- Image-domain gradient gains do not establish higher scanner resolution.
- The local method is APR-RD-inspired; it is not an exact reproduction and no SOTA superiority claim is made.

## Chart contract

- Section: blind-denoising evidence.
- Analytical question: did the algorithm reduce discontinuity while preserving geometric width?
- Takeaway: continuity and dropout metrics fell on both sides; separate guardrails show negligible FWHM drift.
- Family: comparison.
- Chart type: grouped vertical bar.
- Data: eight reviewed rows, four metrics × before/after.
- Encodings: x=metric, y=value ratio, color=stage; side and improvement retained for audit tooltips.
- Palette: hard two-root cap; blue for before and orange for after, with stage labels/legend so color is not the only distinction.
- Scale: zero-based ratio scale; lower is better for every plotted metric.
- Delivery: native chart inside the portable HTML artifact.
- QA: verify in the packaged report at desktop and narrow width; semantic table fallback must remain readable.

## Omitted visuals

- No raw CT image is embedded in the portable report to avoid duplicating potentially sensitive image data and increasing artifact size. The quantitative chart plus exact audit tables provide the visual evidence for this report.
- No time-series chart is appropriate because the evidence is a single image rather than a temporal dataset.
