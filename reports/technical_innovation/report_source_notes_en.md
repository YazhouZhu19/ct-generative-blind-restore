# English Report Source Notes

## Reporting Job

- Question: explain the algorithm in English as a technical report and foreground its defensible innovations.
- Audience: technical readers evaluating image restoration and measurement integrity.
- Decision supported: whether the method is sufficiently novel, auditable, and validated to justify broader experiments.
- Scope: the saved 2026-09-05 single-image experiment only.
- Comparison baselines: the original 16-bit image, the blind-denoised baseline, and the quality-optimized candidate.
- Success criteria: English metrics reconcile exactly to the saved JSON; innovation claims distinguish inherited methods from repository-specific engineering; metrology limitations remain visible.

## Validation Result

- Overall assessment: shareable with caveats.
- Recomputed blind-stage continuity reductions: left 30.6027%, right 27.7185%.
- Recomputed blind-stage dropout reductions: left 53.3333%, right 76.0000%.
- Recomputed blind-stage FWHM drift: left 0.04894%, right 0.35829%.
- Recomputed quality-stage flat high-frequency reduction: 23.0259%.
- Recomputed quality-stage mean edge-acutance gain: 5.8659%.
- Recomputed quality-stage mean contrast gain: 2.9999%.
- Recomputed length pass fraction: 99/100 = 99%.
- All three saved guardrail flags are true.

## Required Caveats

- One image is not an external benchmark or generalization study.
- No paired clean target, repeat scan, physical pixel size, reference standard, or PSF/MTF is available.
- Image-domain gradient gains do not establish improved physical scanner resolution.
- The local implementation is APR-RD-inspired; it is not an exact reproduction, and no SOTA superiority claim is made.

## Chart Contract

- Section: blind-denoising evidence.
- Analytical question: did the algorithm reduce discontinuity while preserving geometric width?
- Takeaway: continuity and dropout fell on both sides, while separate guardrails show negligible FWHM drift.
- Family and type: comparison; grouped vertical bar.
- Data sufficiency: eight reviewed rows, four metrics × before/after.
- Encodings: x = metric, y = ratio, color = stage; side and improvement are retained in audit tooltips.
- Palette policy: hard two-root cap with stage labels/legend so color is not the only distinction.
- Scale: zero-based ratio scale; lower is better for all plotted metrics.
- Delivery: native chart in the portable HTML artifact.
- QA: artifact validation and structural packaging passed; browser-level viewport and source-dialog testing were unavailable because no compatible Chromium executable was installed. The semantic chart table remains embedded and readable.

## Omitted Visuals

- The raw CT image is not embedded, avoiding duplication of potentially sensitive image data and keeping the report portable.
- A time-series chart would be misleading because the evidence contains one image rather than temporal observations.
