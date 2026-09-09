# v19 Measurement-Invariant Zoned Restoration Technical Report

> Report status: this document describes the current `app/structure_anchored_multiregion_denoise.py` implementation, full candidate search, and uint16 round-trip audit. Section 2 reports a full-resolution 2200×1600 run in the same CPU container. A single-image result is not presented as a general optimum or a state-of-the-art claim.

## 1. Technical summary

V19 preserves the v17 generative restoration and v18 clean-boundary enhancement, then performs zoned restoration under measurement-invariance constraints. It does not invoke a new generative model and does not redraw lamellae. Instead, it treats v18 as the pixel baseline and independently estimates and shrinks high-frequency residuals in four regions: complete lamella/interlayer stacks, the central solid, endpoint-exterior fog, and low-structure background.

The full search selects `stack=0.02`, `central=0.50`, `fog=0.60`, and `flat=0.50`. Relative to v18, high-frequency residuals decrease by approximately 9.51% in the central solid, 14.26% in endpoint-exterior fog, and 11.99% in low-structure background. Reductions within lamellae and interlayers are deliberately limited to 0.0561% and 0.1097% to avoid trading measurable geometry for visual smoothness. The result reports a 0.315009% lamella-width P95 error, 0.209611% interlayer-width P95 error, 0.004734 px endpoint P95 deviation, 99/96 lamella/interlayer dual-evidence passes, and 0.999892 SSIM against v18. Nine of 100 marginal lamellae are restored exactly, and the second release audit after writing and reloading the TIFF passes every gate.

The central design choice is not stronger global filtering, but separation of regions that can accept more denoising from pixels whose measurement response must remain stable. Endpoint and contour-sensitive hard-anchor pixels are restored from the exact same v18 coordinates after every candidate. The lamella stack is filtered only along its long axis, with no transverse mixing. Each candidate is remeasured after uint16 quantization, and any regressing lamella plus adjacent interlayers is restored to v18. Because the released image still contains v17-generated pixels, it remains labelled `MEASUREMENT_CANDIDATE` and is not calibrated metrological truth.

## 2. Container-validated result: material cleanup in low-structure regions under strict width gates

### 2.1 Selected profile

| Region | Selected strength | Direction and purpose |
|---|---:|---|
| Complete lamella/interlayer stack | 0.02 | Suppress random axial variation without mixing across lamella width |
| Central solid | 0.50 | Isotropic local-Wiener residual shrinkage for grain-like interior noise |
| Endpoint-exterior fog | 0.60 | Reduce fog-like high-frequency residual and positive halo outside hard anchors |
| Low-structure background | 0.50 | Stronger cleanup in low-gradient, low-protection-weight background |

### 2.2 Quantitative evidence

Noise reductions below use the unchanged v18 image as their baseline. Structure errors come from the same-coordinate audit among the raw image, v16 carrier, and output. SSIM is measured against v18. Percentages are relative quantities.

| Metric | Full-resolution release candidate |
|---|---:|
| Central-solid high-frequency residual reduction | 9.51% |
| Endpoint-exterior fog residual reduction | 14.26% |
| Low-structure background residual reduction | 11.99% |
| Lamella axial-residual reduction | 0.0561% |
| Interlayer high-frequency residual reduction | 0.1097% |
| Lamella-width relative-error P95 | 0.315009% |
| Interlayer-width relative-error P95 | 0.209611% |
| Endpoint-deviation P95 | 0.004734 px |
| Lamella/interlayer dual-evidence pass counts | 99 / 96 |
| Row-width drift P95 / maximum | 0.003053 / 0.051013 px |
| Selective rollback | 9 / 100 lamellae |
| SSIM against v18 | 0.999892 |
| Post-write uint16 release audit | All gates passed |

The result shows that the visually important central, fog, and background regions can receive substantially more denoising than the lamella bodies. The smaller stack reductions are an intentional safety tradeoff, not a failure of the estimator: any stronger candidate must still pass FWHM, endpoint, topology, detail-correlation, and row-wise width gates.

These values come from the completed search recorded in `structure_anchored_multiregion_v19_metrics.json` and its `post_write_uint16_release_audit`. High-frequency residual reduction is a fixed-operator proxy, not an absolute denoising rate against noise-free truth.

## 3. Scope, inputs, and metric definitions

### 3.1 Each of the three inputs has a distinct role

- **Raw 16-bit image:** supplies the original same-coordinate observation used to measure lamella and interlayer constraints.
- **V16 non-generative structural carrier:** provides robust guidance for centerlines, pitch, endpoints, and finite-width measurement; it remains the authoritative geometry carrier.
- **V18 enhanced image:** serves as the v19 pixel baseline and already contains the v17 structure-conditioned generated residual plus v18 finite-width boundary and endpoint cleanup.

V19 performs incremental post-processing only on v18 pixels. It neither replaces v17/v18 nor draws the v16 carrier into the result.

### 3.2 Metric definitions

- **High-frequency residual RMS:** root mean square, within a specified mask, of the difference between the image and a zero-phase Gaussian low-pass estimate. V19 reduction is measured against v18.
- **Lamella-width P95 error:** the 95th percentile of per-lamella output FWHM relative error against the structural constraint.
- **Interlayer-width P95 error:** the 95th percentile of finite-width gap relative error between adjacent lamellae.
- **Endpoint-deviation P95:** the 95th percentile absolute difference between constrained and output endpoints, in pixels.
- **Dual-evidence pass count:** the number of lamellae or interlayers supported by both raw-image evidence and carrier evidence.
- **SSIM:** structural similarity between v19 and v18 within the target ROI. It is a global non-regression signal, not a substitute for per-structure geometry audits.

## 4. End-to-end pipeline: preserve generation and enhancement, then denoise by zone and audit the release

```text
raw 16-bit image ───────────────┐
                                ├─ same-coordinate lamella, interlayer,
v16 non-generative carrier ─────┘  centerline, endpoint, and FWHM measurement
                                │
                                ├─ structure maps, endpoint envelope,
                                │  and measurement-operator support
                                │
v17 structure-conditioned generated result
        │
        `─ v18 finite-width boundary cleanup and per-structure rollback
                                │
                                v
                          v18 pixel baseline
                                │
          ┌─────────────────────┼────────────────────┐
          │                     │                    │
  axial stack filtering   central-solid filter   fog and flat-background filters
          └─────────────────────┼────────────────────┘
                                │
                   restore endpoint/contour hard anchors
                                │
                   quantize to uint16 and remeasure
                                │
          restore any width-, endpoint-, or evidence-regressing
                 local structure to exact v18 pixels
                                │
                 up to three rollback rounds + global gates
                                │
                 write TIFF, read it back, and release-audit again
                                v
                     MEASUREMENT_CANDIDATE v19
```

The pipeline performs no resize, registration, affine or non-rigid warp, coordinate resampling, or analytic lamella redraw. Output dimensions must exactly equal the raw input dimensions and the written type is `uint16`. The implementation verifies dimensions, type, and pixel round-trip identity after writing.

## 5. Method and model specification

### 5.1 Four masks decouple denoising budget from structure risk

1. **Complete lamella/interlayer stack:** the intersection of the structure-interior mask and left/right stack ROIs, excluding endpoints and sensitive contours. Filtering is strictly axial; transverse filter sigma is zero, so samples are not borrowed from adjacent lamellae or interlayers.
2. **Central solid:** the configured central ROI excluding hard anchors. This region has no dense periodic width boundaries and can use a stronger isotropic estimate.
3. **Endpoint-exterior fog:** the exterior of the measured structure envelope, sufficiently far from its boundary threshold and close to the left/right stacks. It targets the fog-like region outside the lamella endpoints.
4. **Low-structure background:** pixels below the 30th percentile of gradient magnitude after smoothing, excluding the side stacks, central solid, hard anchors, and high-protection-weight structure.

Independent strengths remove the coupling risk in which stronger boundary cleanup also changes width.

### 5.2 Adaptive local-Wiener residual shrinkage

For each region, a symmetric Gaussian kernel provides a low-frequency estimate:

\[
L=G_{\sigma_f}*I,\qquad H=I-L.
\]

High-frequency local energy is then estimated over a wider neighborhood:

\[
E=G_{\sigma_l}*(H^2),\qquad
q=\operatorname{clip}\left(\frac{E-\sigma_r^2}{E+\varepsilon},0,1\right).
\]

Here, \(\sigma_r=k\hat\sigma_n\), where \(\hat\sigma_n\) is the target-region MAD noise estimate. The denoising increment is

\[
\Delta I=-(1-q)H.
\]

The four increments are weighted by their independent strengths, summed, and clipped to a per-pixel cap of `max(0.85·noise estimate, 2/65535)`. All filters are symmetric and zero phase, avoiding the systematic edge displacement associated with causal filtering.

The current estimator settings are:

| Region | Filter σ (longitudinal, transverse) | Local-energy σ | Noise factor |
|---|---:|---:|---:|
| Lamella/interlayer stack | (1.60, 0.00) | (3.20, 0.00) | 0.60 |
| Central solid | (0.75, 0.75) | (2.20, 2.20) | 0.70 |
| Endpoint-exterior fog | (0.85, 0.85) | (2.50, 2.50) | 0.78 |
| Low-structure background | (0.70, 0.70) | (2.00, 2.00) | 0.75 |

### 5.3 Hard anchors and measurement invariance use different mechanisms

V19 distinguishes two protections:

- **Endpoint/contour hard anchors:** the union of a dilated endpoint structure field and endpoint-envelope boundary. After every candidate, these coordinates are restored to the exact v18 pixels. The release audit requires zero changed uint16 values in this mask.
- **Width-measurement invariance:** the implementation rasterizes the support actually read by the deployed FWHM and endpoint detectors, making measurement-sensitive samples explicit for audit. Hard-freezing every scattered FWHM sampling band created visible seams, so width invariance is not claimed from freezing the full support. It is instead enforced through axial-only stack filtering, row-wise FWHM remeasurement, per-layer limits, and selective rollback.

In addition to the 25 deployed width sample rows, v19 samples FWHM along the entire lamella at a three-row stride. A lamella is unsafe when its row-wise drift P95 exceeds 0.025 px or its maximum exceeds 0.060 px. The global release gate is stricter on P95: no more than 0.020 px, with the same 0.060 px maximum.

### 5.4 Candidate search and selective rollback

The search first increases central, fog, and flat-region strengths independently of the stack, then fixes the best safe-region tuple and tests axial stack strengths from 0.02 through 0.12. This avoids an unnecessary four-dimensional Cartesian grid.

Every candidate enters the following loop:

1. Apply zoned residual increments and restore hard anchors.
2. Quantize to the exact uint16 pixels that would be released.
3. Remeasure every lamella and interlayer and run the denser row-wise FWHM audit.
4. Restore any lamella and its neighborhood to v18 when width, endpoint, length, or dual evidence regresses.
5. Repeat for at most three rounds, then recompute noise, correlations, SSIM, and topology.
6. Admit the candidate to scoring only if every gate passes.

The score prioritizes the weakest reduction among the three lower-risk regions, then adds a weighted multi-region denoising benefit. This prevents the optimizer from cleaning only the easiest background while leaving the user-visible fog region unchanged.

### 5.5 Release gates operate inside the uint16 quantization loop

The principal gates require:

- endpoint P95 within the permitted baseline increment, with no lamella- or interlayer-length regression;
- lamella-width error P95 no greater than 0.35%;
- interlayer-width error P95 no greater than the baseline plus 0.02 percentage points and no greater than 0.40% in absolute terms;
- lamella and interlayer dual-evidence pass counts no lower than v18;
- edge clarity no lower than v18;
- low-frequency, mid-frequency, and gradient correlations of at least 0.99999, 0.99990, and 0.99985;
- median/P10 lamella axial-detail correlations of at least 0.99997/0.99994;
- target-ROI SSIM against v18 of at least 0.99985;
- selective rollback no greater than 15% of all lamellae;
- unchanged ordinary peak count, pitch within 0.02 px, and unchanged constraint-matched counts;
- bit-exact uint16 hard-anchor pixels.

After the TIFF is written, it is reloaded and the structure, local-width, topology, SSIM, and hard-anchor audits are repeated. A release metrics manifest is created only when the round-tripped file passes.

## 6. Project-level innovations and engineering contributions

The following are engineering improvements over this project's v17/v18 pipeline, not unverified SOTA claims.

1. **Generation quality is decoupled from measurement safety.** V17 owns structure-conditioned generative restoration, v18 owns finite-width edge cleanup, and v19 owns the safe denoising budget and release audit. Each stage can be compared or rolled back independently.
2. **Protection is measurement-operator aware.** The design targets the support and response of the actual FWHM, endpoint, and interlayer measurements rather than a generic edge map. Hard-anchor and response-invariance requirements use distinct mechanisms.
3. **Anisotropy and zoned processing are designed together.** Periodic stacks exchange information only axially, while central, fog, and background regions receive independent isotropic strengths, combining visual cleanliness with transverse thickness stability.
4. **Rollback is structure-specific.** A failed candidate does not force rejection of the whole image or acceptance of an average metric. The affected lamella and adjacent interlayers are restored to exact v18 pixels.
5. **Release is quantization aware.** Candidate selection and final acceptance both operate on uint16 pixels, followed by a TIFF write/read audit, preventing a float-only pass from becoming an invalid released file.

## 7. Robustness checks and negative-result handling

- **No spatial geometry operation:** no resize, registration, perspective correction, or non-rigid transform can introduce interpolation-based length or thickness changes.
- **No analytic redraw:** ideal ribbons do not replace observed lamellae; local defects, waviness, and axial detail continue to come from v17/v18 pixels.
- **Bit-level endpoint invariance:** hard anchors are compared pixel-by-pixel as uint16 values, not only through an average endpoint position.
- **Per-layer rather than global-only acceptance:** a global P95 pass is insufficient; any layer exceeding local row-width limits triggers rollback.
- **Strong candidates can be rejected:** larger central, fog, flat, or stack strengths are eligible only if all geometry and detail gates pass.

The current result is based on one target image and does not establish equal performance across scanners, exposures, materials, or noise distributions. The metrics quantify agreement with v18 and v16-derived constraints; they are not absolute error against an unavailable noise-free ground truth.

## 8. Reproduction and outputs

```bash
python app/structure_anchored_multiregion_denoise.py \
  --source input/source_16bit.tif \
  --carrier results_generative_shape_v16_measurement_quality/FINAL_MEASUREMENT_v16_quality_enhanced_2200x1600_16bit.tif \
  --input results_generative_shape_v18_clean_edges_detail_preserved/MEASUREMENT_CANDIDATE_v18_clean_edges_detail_preserved_16bit.tif \
  --outdir results_generative_shape_v19_structure_anchored_multiregion
```

Primary artifacts are:

- `MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion_16bit.tif`: same-size 16-bit candidate image;
- `MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion.png`: preview;
- `MEASUREMENT_CANDIDATE_v19_comparison.png`: v18, v19, and absolute-delta comparison;
- `AUDIT_v19_multiregion_masks.png`: hard anchors, operator support, and the four region masks;
- `lamella_v19_comparison.csv` and `interlayer_v19_comparison.csv`: per-structure comparisons;
- `local_row_width_v19_comparison.csv`: denser row-wise FWHM invariance audit;
- `structure_detail_v19.csv`: structure-detail correlations;
- `structure_anchored_multiregion_v19_metrics.json`: candidate grid, rollback records, gates, and post-write audit.

## 9. Limitations and correct-use boundary

1. V19 does not train or execute a new generative model. Generated pixels originate in upstream v17 and are retained through v18 and v19.
2. `MEASUREMENT_CANDIDATE` means that the image passes this project's internal same-coordinate constraint audit; it does not mean metrology certification.
3. The v16 carrier and exported per-structure CSVs should remain the measurement-review references. The enhanced image supports detection, display, and assisted measurement but should not replace raw data before calibration.
4. Very low SNR, merged lamellae, broken structures, or layouts outside the current ROI assumptions require renewed mask, threshold, and search validation.
5. High-frequency Gaussian residuals are noise proxies and may contain some real detail. They must therefore be interpreted jointly with detail-correlation and geometry gates.

## 10. Recommended next steps

- Rerun the same container grid on more fixed input hashes to confirm cross-image stability of the selected parameters.
- Use repeated scans from the same device to assess length and thickness repeatability rather than relying only on single-image similarity.
- Introduce a phantom with known thickness and spacing to calibrate pixel size, PSF/MTF, and endpoint bias separately.
- Stratify rollback rates by exposure, material, and lamella position to determine whether parameters should vary by scanner or noise regime.
- Release the raw image, v16 carrier, v18 input, v19 image, parameter manifest, and per-layer audit CSVs together as a traceable measurement chain.

## 11. Further questions

- Will the fixed four-zone noise factors remain stable across a multi-image set, or should they be selected from each image's local noise distribution?
- How strongly does the endpoint-exterior positive-halo metric correlate with expert ratings of a “clean boundary”?
- For marginal lamellae with frequent rollback, should a conservative structure-conditioned residual branch be trained instead of further increasing post-processing strength?
- After phantom validation, which internal pixel-space gates can be converted into a physical-unit uncertainty budget?
