# v12 Edge-Clarity and Raw Dual-Evidence Constraint Report

## Objective

Version 12 extends the v11 guide-first generative pipeline in two ways: sharper lamella side/end boundaries and complete three-way comparison of lamella and interlayer length/width against both the blind-denoised guide and the original raw input.

## Improvements

1. **Sharper analytic slabs.** The finite-width dual-logistic transverse transition is reduced from `0.38 px` to `0.24 px`, and the endpoint transition from `0.55 px` to `0.40 px`. Three FWHM calibration rounds preserve target width after sharpening.
2. **Per-lamella raw comparison.** Each of 100 CSV rows contains guide, raw, and output endpoints, length, FWHM, edge clarity, errors, raw non-regression flags, and a final dual-evidence decision.
3. **Per-interlayer raw comparison.** All 98 gaps are evaluated for row-wise width and common longitudinal length against guide and raw evidence.
4. **Noise-aware raw guardrail.** Raw data is evidence, not a coordinate average. The output-to-raw error may not exceed the existing guide-to-raw disagreement plus raw measurement tolerance. This prevents regression toward noisy raw estimates while still enforcing raw consistency.
5. **Five-part acceptance.** Guide layer geometry, raw layer non-regression, guide interlayer geometry, raw interlayer non-regression, and edge clarity must all pass.

## Result

| Metric | Result |
|---|---:|
| Constraint-matched lamellae | 49 left + 51 right = 100 |
| Interlayers | 98 |
| Lamella dual-evidence pass | 100 / 100 |
| Interlayer dual-evidence pass | 98 / 98 |
| Guide endpoint error P95 | 0.0374 px |
| Guide length error P95 | 0.0443 px |
| Raw endpoint difference P95 | 0.1451 px |
| Raw length difference P95 | 0.1705 px |
| Guide lamella FWHM error P95 | 0.1699% |
| Guide interlayer-width error P95 | 0.1260% |
| Raw interlayer-width difference P95 | 7.31% |
| Raw interlayer-length difference P95 | 0.2377 px |
| Raw / v11 / v12 edge clarity | 0.3061 / 0.3599 / **0.3883** |

The v12 edge score improves by about 7.9% over v11 and 26.9% over raw. Target median lamella width is `4.251057 px`; output median is `4.251003 px`.

Raw lamella FWHM difference P95 remains 17.77% because FWHM is unstable in the noisy source. It is therefore retained as an evidence interval rather than directly replacing the denoised-guide target. All 100 layers nevertheless pass the raw non-regression rule.

## Deliverables

- `GENERATIVE_shape_hard_projected_16bit.tif`: 2200×1600 16-bit visual output.
- `lamella_dual_evidence_comparison.csv`: 100 three-way layer comparisons.
- `interlayer_dual_evidence_comparison.csv`: 98 three-way gap comparisons.
- `GENERATIVE_raw_input_comparison_overlay.png`: raw/output endpoint overlay.
- `hard_shape_projection_metrics.json`: aggregate metrics and all guardrails.

The result remains generative and is not metrology-certified. Its analytically constrained geometry is auditable, but synthetic low-frequency appearance and texture are still unsuitable as final calibrated measurement evidence.
