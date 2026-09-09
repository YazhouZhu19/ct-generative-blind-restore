# v20 Measurement-Safe Low-Strength TV Post-Processing Technical Report

> Report status: this document is based on `app/measurement_safe_postprocess.py` and the final full-resolution `measurement_safe_postprocess_v20_metrics.json` run. The evidence covers one 2200×1600 industrial CT/X-ray image. It is a single-image internal validation, not evidence of a general optimum, state-of-the-art performance, or metrology certification.

## 1. Executive result

V20 is a post-processing-only stage after v19. It retains the v17-generated pixels, v18 finite-width boundary cleanup, and v19 zoned denoising; v20 itself does not train, load, or execute a generative model. Its new operation is a bounded, low-strength Chambolle total-variation (TV) residual in three non-measurement writable regions: the central ROI core, endpoint-exterior fog, and low-structure background.

The selected profile among seven fixed candidates is `tv_balanced_strong_post`. All three regions use `tv12` (TV weight 0.0012), with residual blends of 0.40, 0.55, and 0.50 for central, fog, and flat regions; lamella/interlayer stack strength is fixed to zero. Relative to v19, fixed writable-support high-frequency RMS falls by 1.6893%, 0.6311%, and 5.1796%. On the same frozen spatial supports, the three-direction Haar-detail mean absolute value falls by 2.4734%, 0.6880%, and 7.1663%. These are complementary no-reference high-frequency proxies, not statistically independent proof and not error against unavailable noise-free truth.

The complete lamella/interlayer bundles, deployed measurement-operator support, endpoint envelope, configured central ROI border/ring, and strong edges are copied bit-for-bit from same-coordinate v19 `uint16` pixels. Direct output-geometry comparison over 100 lamella and 98 interlayer rows has a maximum difference of 0 px; median, P95, and maximum drift are all 0 px over 41,327 every-row width samples. All changes are contained inside the allowed writable support. SSIM against v19 is 0.99999435, and every release gate passes again after TIFF write/read.

The output remains a `MEASUREMENT_CANDIDATE`. The non-generated v16 carrier and exported per-structure evidence remain the authoritative review baseline until multi-image, phantom, pixel-size, and PSF/MTF calibration is completed.

## 2. Selected profile and candidate search

“Low-strength TV” means a small regularization weight, partial residual blending, and an explicit amplitude cap. It does not imply that the acquisition dose is known or low.

| Region | Estimator | TV weight | Blend | Write policy |
|---|---|---:|---:|---|
| Central ROI core | `tv12` | 0.0012 | 0.40 | Remove the gate-weighted residual mean, then write only inside the writable support |
| Endpoint-exterior fog | `tv12` | 0.0012 | 0.55 | Write only through the soft gate outside the endpoint envelope |
| Low-structure background | `tv12` | 0.0012 | 0.50 | Write only through the low-gradient, low-structure soft gate |
| Complete lamella/interlayer bundles | — | — | 0.00 | Bit-exact `uint16` lock; no v20 denoising |

Every TV estimate uses `eps=2×10⁻⁵` and at most 80 iterations. The normalized residual cap is 0.0038118. The fixed search grid contains seven candidates:

1. `identity`;
2. `tv_conservative`;
3. `tv_recommended`;
4. `tv_stronger_fog`;
5. `tv_stronger_all`;
6. `tv_balanced_strong_post`;
7. `tv_aggressive_audit_only`.

`tv_recommended`, `tv_stronger_fog`, `tv_stronger_all`, and `tv_balanced_strong_post` pass the complete safety and benefit gates. Identity and the conservative profile miss minimum benefit; the aggressive audit-only profile is rejected by safety gates. Scoring is performed only among candidates that pass every constraint. If no nonzero candidate meets both safety and benefit requirements, the program publishes no new v20 TIFF and leaves v19 as the safe fallback.

## 3. Spatial gates and hard locks

### 3.1 Explicit exterior zero padding and an 8 px distance ramp

Each binary writable-region mask array is padded by one explicit exterior zero contour before it is converted to a compactly supported gate with an 8 px interior distance-transform smoothstep ramp. The gate is designed so that:

- weight is exactly zero outside the writable support;
- weight is exactly zero on the first inside contour of the binary support;
- weight rises smoothly only farther inside;
- central, fog, and flat gates are mutually disjoint and disjoint from the hard lock.

This replaces the discontinuous write boundary that can result from multiplying a binary mask by a Gaussian blur. In the final audit, the zero-weight outer contour changes by 0 DN. The inner soft-gate seam has absolute-delta P95/maximum values of 1/7 DN.

### 3.2 Hard-locked content

The following pixels are restored bit-for-bit from v19 after quantization:

- complete lamella, interlayer, finite-width boundary, and endpoint bundles;
- actual deployed FWHM and endpoint measurement-operator support;
- endpoint envelopes and low-confidence complete structure cells;
- the configured central ROI border/ring;
- the image-wide strong-edge protection set.

“Configured central ROI border/ring” is the fixed protection and audit region used by this implementation. It is not described as an automatically detected, segmented, or certified physical object boundary. The central audit uses a fixed-row/fixed-column subpixel gradient tracker and demonstrates numerical stability only for this ROI configuration on the supplied image.

## 4. End-to-end flow

```text
raw 16-bit image + v16 non-generated structural carrier
        |
        `-- rebuild lamella, interlayer, endpoint, FWHM,
            and deployed measurement-operator supports

v19 candidate (already retaining v17 generation,
v18 boundary cleanup, and v19 zoned denoising)
        |
        |-- build complete-structure, operator, endpoint, and strong-edge locks
        |-- build configured central ROI border/ring lock
        `-- build three mutually exclusive writable supports
                |
                |-- calculate low-strength tv04, tv08, and tv12 estimates
                |-- form amplitude-capped regional residuals
                |-- apply 8 px distance-transform smoothstep gates
                `-- quantize to uint16 and restore every hard lock
                        |
                        |-- remeasure structures, every-row width, and topology
                        |-- audit configured-ROI fixed-line tracking and local SSIM/gradients
                        |-- audit fixed-support HF proxies, allowed writes, and seams
                        `-- score only after all release gates pass
                                |
                                |-- write the source-size 16-bit TIFF
                                `-- reload and repeat the full release audit
                                        v
                              MEASUREMENT_CANDIDATE v20
```

The stage performs no resize, registration, spatial warp, coordinate resampling, analytic lamella redraw, CLAHE, sharpening, black-level clipping, or global contrast remapping.

## 5. Low-strength TV residual model

For the v19 target image \(I_{19}\), each TV estimator approximately solves

\[
U_w=\arg\min_U\left[\frac{1}{2}\lVert U-I_{19}\rVert_2^2+w\,\operatorname{TV}(U)\right],
\]

where \(w\in\{0.0004,0.0008,0.0012\}\). V20 does not replace the image with \(U_w\). It uses only a capped residual,

\[
R_w=\operatorname{clip}(U_w-I_{19},-c,c),
\]

combined through three disjoint soft gates:

\[
I_{20}=I_{19}+\lambda_c M_cR_{w_c}
                 +\lambda_f M_fR_{w_f}
                 +\lambda_b M_bR_{w_b}.
\]

The central residual is made gate-weighted zero mean to limit DC drift. The composition is immediately quantized to `uint16`, after which every hard-lock coordinate is overwritten with the exact v19 digital value. All acceptance decisions operate on the actual release pixels.

## 6. Final numerical result

### 6.1 Complementary fixed-support high-frequency proxies

| Region | Fixed samples | Fixed-HF RMS reduction | Three-direction Haar-detail mean-absolute reduction | Wider v19 fixed-operator reduction |
|---|---:|---:|---:|---:|
| Central ROI core | 73,164 | 1.6893% | 2.4734% | 1.2454% |
| Endpoint-exterior fog | 6,032 | 0.6311% | 0.6880% | 0.0138% |
| Low-structure background | 86,867 | 5.1796% | 7.1663% | 0.5186% |
| Lamellae | Locked | — | — | 0.0000% |
| Interlayers | Locked | — | — | 0.0000% |

The two primary proxies use the same spatial supports frozen before candidate selection but aggregate high-frequency residuals differently. They are complementary metrics, not independent sensors or external truth. The wider v19 operators include many untouched pixels in their denominators, so the smaller fog and background gains provide a more conservative view of whole-region visibility.

### 6.2 Structure and measurement consistency

| Metric | v20 release result |
|---|---:|
| Output size / type | 2200×1600 / `uint16` |
| Lamella-width relative-error P95 | 0.315009% |
| Interlayer-width relative-error P95 | 0.209611% |
| Endpoint-deviation P95 | 0.004734 px |
| Lamella-length-deviation P95 | 0.060711 px |
| Interlayer-length-deviation P95 | 0.082604 px |
| Lamella / interlayer dual-evidence passes | 99 / 96 |
| Direct lamella / interlayer rows compared | 100 / 98 |
| Maximum direct output-geometry difference | 0 px |
| Every-row width samples | 41,327 |
| Width drift median / P95 / maximum | 0 / 0 / 0 px |
| Complete bundle, operator, central-ROI ring, and strong-edge changes | 0 for each set |
| Changes outside writable support | 0 |
| Changes outside target ROI | 0 |
| Edge clarity | 0.298482, unchanged from v19 |
| Low-/mid-frequency/gradient correlation | 0.99999937 / 0.99992168 / 0.99986889 |
| Lamella axial-detail correlation median / P10 | 0.99998638 / 0.99997343 |
| SSIM against v19 | 0.99999435 |
| TIFF write/read identity | Exact |

The configured central ROI fixed-line tracker reports edge-position P95/maximum drift of 0.0000003/0.001604 px, transition-width P95/maximum drift of 0.000012/0.000246 px, width P95/maximum drift of 0/0.000962 px, and height P95/maximum drift of 0.000019/0.001604 px. P05 retention is 1.0 for both edge-peak strength and confidence. Central-core-to-exterior contrast retention is 1.00086 and robust CNR retention is 1.00212. These are fixed-line results for the configured ROI, not physical boundary calibration.

### 6.3 Writable-region local-detail guards

| Region | Masked global SSIM | SSIM-map mean / P01 | Gradient correlation | Gradient RMS retention | Gradient relative RMSE |
|---|---:|---:|---:|---:|---:|
| Central | 0.99996559 | 0.99992883 / 0.99981827 | 0.99989177 | 0.98613 | 0.01684 |
| Fog | 0.99999962 | 0.99999952 / 0.99998665 | 0.99999419 | 0.99849 | 0.00358 |
| Flat | 0.99999666 | 0.99999326 / 0.99984622 | 0.99753573 | 0.91373 | 0.10864 |

These local metrics directly limit structure displacement and TV-staircasing risk inside writable regions. They are evaluated together with global correlations, hard locks, and pixel-delta limits.

### 6.4 Write support, seams, and amplitude

- Changed pixels: 88,375, approximately 2.51% of the full image.
- Writable-support pixels: 170,554.
- Changed pixels outside writable support: 0.
- Overlapping writable-gate pixels: 0.
- Writable/hard-lock overlap pixels: 0.
- Zero-weight outer-contour pixels: 19,221; maximum gate weight 0 and maximum delta 0 DN.
- Inner-seam pixels: 11,680; absolute-delta P95/maximum 1/7 DN.
- Changed-pixel absolute-delta P50/P95/P99/maximum: 9/44/66/118 DN.
- New zero-clipped and saturation-clipped pixels: 0 for both.

The P99 and maximum deltas pass their 72 DN and 128 DN release limits.

## 7. Release audit

Both the quantized candidate and reloaded TIFF are checked for:

- endpoint, length, aggregate FWHM, dual-evidence counts, and topology;
- every-row transverse width and all 100+98 direct geometry records;
- every `uint16` lock set, outside-target canvas, and allowed-write support;
- zero-weight contour, inner seam, and gate disjointness;
- configured central ROI fixed-line position, transition width, dimensions, peak strength, contrast, and CNR;
- local SSIM, gradient correlation, gradient energy, and near-zero-gradient fraction in every writable region;
- two complementary fixed-support high-frequency proxies, wider v19 operators, global correlations, and SSIM;
- new clipping, delta P99/maximum, and TIFF pixel round-trip identity.

The selected candidate passes every gate. The codebase has 59 passing tests, including 12 v20-specific tests covering the zero-weight first smoothstep contour, disjoint gates, allowed-write containment, direct structure drift, Haar-detail proxy behavior, and TIFF round-trip fidelity.

## 8. Engineering advances

These are project-level engineering advances, not state-of-the-art claims established on public multi-dataset benchmarks:

1. **Bit-exact measurement-bundle freezing.** Complete lamella/interlayer bundles and deployed measurement-operator support are locked to v19, preserving every-row widths and direct geometry records exactly.
2. **Configured ROI border/core separation.** The configured ROI border/ring is locked while low-strength TV acts only in the deeper writable core, with a fixed-line tracker for numerical auditing.
3. **Compactly supported smooth gating.** Explicit one-pixel exterior zero padding plus the 8 px distance-transform smoothstep ramp is exactly zero outside support and on its first contour, making seam behavior auditable.
4. **Bounded residual rather than image replacement.** TV estimates contribute only small capped residuals, retaining the v19 structural and intensity baseline.
5. **Local fidelity plus allowed-write auditing.** Candidates must pass local SSIM/gradient gates and prove that every changed pixel belongs to writable support, in addition to global checks.
6. **Quantization-in-the-loop.** Geometry, structure, high-frequency proxies, and locks are evaluated on released `uint16` pixels and repeated after file reload.

## 9. Limitations and correct-use boundary

1. V20 does not execute a generative model; generated pixels originate only in upstream v17 and are retained through v18/v19.
2. The lamella/interlayer bundles receive no new v20 denoising, preventing accumulation of new thickness or length errors.
3. Only one image is available and there is no paired noise-free truth. High-frequency proxy reductions are not direct noise-error reductions.
4. TV can still create staircasing or erase faint defects. Local SSIM/gradient gates reduce this risk but cannot replace multi-image and phantom validation.
5. The configured central ROI border/ring is not an automatically identified physical object boundary. Fixed-line tracking cannot be extrapolated to physical dimensional accuracy.
6. Upstream generated pixels remain in the output. The image is suitable for display, detection, and measurement assistance, while physical measurement still requires the v16 carrier, per-structure CSVs, pixel-size calibration, and imaging-system validation.

## 10. Run and outputs

```bash
python app/measurement_safe_postprocess.py \
  --source input/source_16bit.tif \
  --carrier results_generative_shape_v16_measurement_quality/FINAL_MEASUREMENT_v16_quality_enhanced_2200x1600_16bit.tif \
  --input results_generative_shape_v19_structure_anchored_multiregion/MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion_16bit.tif \
  --outdir results_generative_shape_v20_measurement_safe_postprocess
```

Primary artifacts:

- `MEASUREMENT_CANDIDATE_v20_measurement_safe_postprocessed_16bit.tif`: source-size 16-bit candidate;
- `MEASUREMENT_CANDIDATE_v20_measurement_safe_postprocessed.png`: display preview;
- `MEASUREMENT_CANDIDATE_v20_comparison.png`: v19, v20, and absolute difference;
- `AUDIT_v20_measurement_safe_masks.png`: hard locks, writable supports, and gate audit;
- `lamella_v20_comparison.csv` and `interlayer_v20_comparison.csv`: per-structure comparison;
- `local_row_width_v20_comparison.csv`: 41,327 every-row width samples;
- `structure_detail_v20.csv`: multiscale structure-detail results;
- `measurement_safe_postprocess_v20_metrics.json`: seven candidate profiles, fixed supports, all guards, and post-write audit.

## 11. Next steps

- Rerun the same frozen grid over repeated scans, exposure settings, materials, and object sizes, and report distributions of benefit and guard outcomes.
- Use a phantom with known lamella, interlayer, and central-structure dimensions for physical-unit calibration.
- Add multi-image tests for TV staircasing, faint-defect retention, and spectral bias.
- Preserve the raw image, v16, v19, v20, metrics JSON, and per-structure CSVs as one traceable processing chain.
- If future work denoises inside the lamella/interlayer bundles, isolate it as a new experiment and repeat all thickness and length validation; v20's zero-drift conclusion cannot be reused.
