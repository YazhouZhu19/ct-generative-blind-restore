# v21 Native-Coordinate Structure-Detail Precision Report

## Executive summary

V21 refines the lamella regions of the accepted v20 image. It preserves the upstream generative blind-denoising and post-processing chain while applying a small additional reduction of high-frequency residuals. Length, width, independent left/right edges, and center position are constrained on the final `uint16` pixels.

The validated profile is `precision_balanced`. It performs no resizing, registration, coordinate warp, resampling, or analytic lamella redraw. All changes are bounded photometric operations on the original `2200×1600` grid. The formal TIFF is first written to a temporary file and reloaded; only a 57/57 release-gate pass permits an atomic replacement of the canonical output. A failed audit cannot overwrite the previous formal file.

For this single-image internal audit:

- 41,327 row-wise FWHM samples have absolute drift P95/maximum of `0.003318 / 0.013469 px`;
- all 49,713 requested bilateral-boundary samples are valid, with edge-position drift P95/maximum of `0.003269 / 0.019967 px`;
- center drift P95/maximum is `0.002184 / 0.009979 px`;
- lamella/interlayer width relative-error P95 is `0.315009% / 0.208542%`;
- directional high-frequency proxies in lamellae/interlayers decrease by `0.075599% / 0.004555%` relative to v20;
- 25,305 pixels change, all inside the fixed writable mask; the central highlight, hard anchors, and pixels outside the target ROI remain bit-exact;
- 57/57 post-write release gates and 73/73 in-container tests pass.

These are v20-relative results under the implemented operators and provisional thresholds, not calibrated physical-accuracy claims. The upstream chain still contains v17-generated pixels, so the output remains a `MEASUREMENT_CANDIDATE`; the non-generated v16 carrier remains the measurement fallback.

## 1. Problem and objectives

V20 bit-locks the complete lamella/interlayer stacks to avoid length and thickness drift. That is safe, but it also prevents further cleanup inside the lamellae. V21 reopens only low-risk pixels supported by two same-coordinate evidence sources and retains a change only when the deployed measurement response remains within its release contract.

The objectives are to:

1. retain the v17 generative result and all v18–v20 processing;
2. reduce remaining lamella/interlayer noise on the native canvas;
3. constrain independent edges, center, FWHM, endpoints, length, per-lamella width, and per-gap width;
4. audit the quantized and reloaded file rather than only a float tensor;
5. fail closed whenever a structural contract cannot be proven.

## 2. Inputs, output, and pixel lineage

Inputs are the raw 16-bit source, the non-generated v16 blind-denoised carrier, and the v20 candidate. V20 is the pixel baseline and already contains the v17 generated residual plus the v18–v20 processing stages.

```text
raw source ───────────────┐
v16 non-generated carrier ├─> dual-evidence constraint extraction
v20 enhanced candidate ───┘
             ↓
native-grid photometric proposal
             ↓
uint16 row/layer/gap projection
             ↓
temporary TIFF -> reload -> 57 release gates -> atomic publish
```

The formal output is `MEASUREMENT_CANDIDATE_v21_structure_detail_precision_16bit.tif`.

This work also corrects the upstream v17 audit input: generated-to-candidate detail checks now use the actual `generated_full` image instead of accidentally substituting the carrier, and v17 archives `GENERATIVE_v17_raw_16bit.tif`. This strengthens future from-v17 reruns. The present v21 run consumes the already fixed v20 TIFF, so this maintenance correction does not retroactively alter the current v21 pixels.

## 3. Method

### 3.1 Native-coordinate photometric model

Along the measured curved centerlines, the algorithm estimates one shared interlayer background per side and one axial amplitude trace per lamella. It changes existing intensities only. It neither moves coordinates nor constructs synthetic rectangular or pointed lamellae.

“Native-coordinate” means that no coordinate transform is executed. It does not claim mathematically zero movement of a measured subpixel edge; the quantized measurement loop bounds that response.

### 3.2 Dual-evidence axial shrinkage

A zero-phase low/high-frequency decomposition is applied to each axial signal. A second-difference MAD estimates local noise. Raw/v16 phase agreement, amplitude agreement, and current-signal energy form a soft evidence weight. High frequency is attenuated only when these heuristic signals provide weak support.

### 3.3 Half-height-fixed transverse proposal

The monotone normalized curve fixes `0`, `0.5`, and `1`. This preserves the half-height crossing only for an ideal normalized profile; it is not the deployed FWHM guarantee. The real operator also averages five rows (`y±2`), smooths transversely, and re-estimates local peak and background.

### 3.4 Material gate and residual cap

The material gate is zero below normalized response `0.08`, fully open above `0.30`, and smoothly interpolated between them. It prevents the lamella model from filling deep gaps. The blind noise estimate caps the proposal; the validated maximum normalized change is `0.00350947`.

### 3.5 Joint every-eligible-row projection

After `uint16` quantization, every eligible lamella row is evaluated with:

- five-row deployed FWHM, tolerance `0.018 px`;
- independently signed left and right edges, tolerance `0.020 px`;
- bilateral center, tolerance `0.010 px`;
- weaker edge strength and gradient-sign purity;
- edge-search-window boundary hits;
- fail-closed handling of changed invalid or baseline-ambiguous rows.

A violation restores only the actual measurement footprint. An additional read-only scan follows the last allowed repair. If violations remain at the round limit, `converged=false` and the candidate cannot be published.

### 3.6 Per-lamella and per-interlayer non-regression

The algorithm then compares 100 lamellae and 98 interlayers for median width, endpoint, length, and edge clarity. Because a gap depends on both adjacent lamellae, a gap regression identifies both neighbors. The 25-row measurement footprint is restored first; a complete local cell is only the fallback for a repeated failure.

For this output, `L12`, `L13`, and `L14` required 10,402 measurement-footprint pixels to be restored. No complete cell was rolled back, and the final strict and legacy unsafe-ID sets are empty.

### 3.7 Fixed write contract and atomic publication

The writable mask is frozen before candidate construction. Release requires bit-exact pixels outside that mask, outside the target ROI, in the central-highlight region, and on all hard anchors. New 0/65535 clipping is forbidden.

The formal file is written to a unique temporary TIFF in the destination directory. Dtype, dimensions, pixel-exact round trip, and all release gates are checked before an atomic rename updates the canonical file.

## 4. Engineering contributions

1. **Measurement-response geometry constraint:** independent edges and center complement FWHM and catch common bilateral translations that preserve width.
2. **Quantize-before-closure:** structural claims are evaluated on the storage grid actually delivered to downstream tools.
3. **Dual-evidence detail retention:** raw and non-generated blind-denoised evidence jointly guide axial cleanup.
4. **Minimal-footprint repair:** only pixels read by the failing measurement operator are restored when possible.
5. **Explicit terminal convergence:** an extra no-repair scan prevents publishing an unverified last-round state.
6. **Release-level write contract:** protected regions are bit-locked and the canonical file is updated atomically only after audit.

These contributions are auditable engineering safeguards. They do not establish cross-scanner, cross-exposure, or absolute metrology accuracy.

## 5. Validated configuration

| Parameter | Value |
|---|---:|
| Profile | `precision_balanced` |
| Axial strength | `0.30` |
| Transverse curve strength | `0.07` |
| Residual cap factor | `0.60` |
| Safety row step | `1` |
| FWHM / signed-edge / center tolerance | `0.018 / 0.020 / 0.010 px` |
| Material gate | `0.08 → 0.30` |

`--candidate-row-step` is retained only as compatibility metadata. It cannot affect repair, selection, or release; every safety decision uses row step 1.

## 6. Results

### 6.1 Boundary and width response

| Metric | Result |
|---|---:|
| Row-wise FWHM samples | 41,327 |
| FWHM absolute drift P95 / max | `0.00331760 / 0.01346904 px` |
| Bilateral requested / valid samples | `49,713 / 49,713` |
| Edge-position absolute drift P95 / max | `0.00326859 / 0.01996698 px` |
| Center absolute drift P95 / max | `0.00218440 / 0.00997894 px` |
| Bilateral-width drift P95 / max | `0.00530130 / 0.03625082 px` |
| Minimum weaker-edge retention on changed support | `0.98147070` |
| Changed invalid/ambiguous rows; changed window-boundary rows | `0; 0` |

### 6.2 Aggregate structure

| Metric | Result |
|---|---:|
| Lamellae / interlayers | `100 / 98` |
| Endpoint deviation P95 | `0.00473384 px` |
| Length deviation P95 | `0.06071100 px` |
| Lamella / interlayer width relative-error P95 | `0.315009% / 0.208542%` |
| Interlayer length absolute-error P95 | `0.08260450 px` |
| Edge clarity | `0.29848182` (equal to v20) |
| Strict/legacy unsafe structure sets | empty |

### 6.3 Denoising, change budget, and fidelity

| Metric | Result |
|---|---:|
| Lamella directional HF RMS | `0.00340937 → 0.00340679` |
| Lamella HF-proxy reduction | `0.0755986%` |
| Interlayer directional HF RMS | `0.00320469 → 0.00320454` |
| Interlayer HF-proxy reduction | `0.00455535%` |
| SSIM against v20 | `0.9999991876` |
| Changed pixels | `25,305` (`0.718892%` of canvas) |
| Changed-pixel absolute DN P50/P95/P99/max | `13 / 54 / 107.96 / 230` |
| Changes outside allowed mask / ROI / central highlight | `0 / 0 / 0` |
| New zero / saturation pixels | `0 / 0` |
| Post-write release checks | `57 / 57` passed |

The 57 entries are emitted Boolean gates; some are composite or cover related risks, so they are not 57 statistically independent pieces of evidence.

The high-frequency RMS values are fixed directional no-reference proxies. Their small reduction reflects the deliberately narrow write contract and must not be interpreted as error against a noise-free ground truth.

Using the same row-wise FWHM operator, the hardened result improves on the pre-hardening balanced trial: P95 drift decreases from `0.00384724` to `0.00331760 px` (`13.77%`), maximum drift decreases from `0.01772927` to `0.01346904 px` (`24.03%`), and changed pixels fall from 35,844 to 25,305 (`29.40%`). The main gain is therefore tighter structural response and a narrower write footprint, not a stronger filter. This remains an engineering comparison rather than error against physical ground truth.

## 7. Visual audit

Full-frame and native-pixel crops show no new lamella break, double edge, halo, central seam, endpoint contamination, or extra periodic grid. Changes are confined to the two lamella stacks, and the central block is fully frozen. The two outermost walls remain the most sensitive locations and should be reviewed first if future profiles increase strength.

The pre-existing central-block grain and joint roughness remain visible by design; they are not introduced by v21.

## 8. Container reproduction

Prepare the three data files excluded by `.gitignore`:

```text
input/source_16bit.tif
results_generative_shape_v16_measurement_quality/FINAL_MEASUREMENT_v16_quality_enhanced_2200x1600_16bit.tif
results_generative_shape_v20_measurement_safe_postprocess/MEASUREMENT_CANDIDATE_v20_measurement_safe_postprocessed_16bit.tif
```

Then run:

```bash
docker compose build ct-v21-structure-detail-precision
docker compose run --rm ct-v21-structure-detail-precision
```

Validated image digest:

```text
sha256:a3b375804794c995a56eec8f2ef3b86298af5818609d39aea06ee38989eaafe8
```

To test the app embedded in the image, mount only the tests:

```bash
docker run --rm --entrypoint python \
  -v "$PWD/tests:/workspace/tests:ro" \
  -w /workspace \
  ct-generative-blind-restore:cpu \
  -B -m unittest discover -s tests -v
```

All `73/73` tests pass, and `docker compose -f compose.yaml config --quiet` succeeds.

## 9. Provenance hashes

| Artifact | SHA-256 |
|---|---|
| Raw source | `c851b8b55af28e81ab5feeca57e03038cb9f1dc31e3a00392a97305f77a7eac0` |
| v16 carrier | `46f590a5d6a39e77996b0f51c7a5f2a2b54c89696be212fa562479f8d2b4e157` |
| v20 input | `4a958a48f4fc22168b11c7b9bf7beeb2c55d2ae97b90b2d5700570da8b54b1e2` |
| v21 output | `21d8e2b85cc861d60000ce664f6a96bc4c09e5b4a9af8cd3c6d6e36e6c024f17` |

## 10. Outputs

- `MEASUREMENT_CANDIDATE_v21_structure_detail_precision_16bit.tif`
- `MEASUREMENT_CANDIDATE_v21_structure_detail_precision.png`
- `MEASUREMENT_CANDIDATE_v21_comparison.png`
- `MEASUREMENT_CANDIDATE_v21_boundary_overlay.png`
- `lamella_v21_comparison.csv`
- `interlayer_v21_comparison.csv`
- `local_row_width_v21_comparison.csv`
- `bilateral_boundary_v21_comparison.csv`
- `axial_detail_v21_comparison.csv`
- `legacy_structure_detail_v21.csv`
- `local_width_constraints_v21.csv`
- `structure_detail_precision_v21_metrics.json`

## 11. Limitations and conclusion

- This is a one-image study without paired noise-free ground truth.
- V21 inherits upstream generated pixels.
- Thresholds are validated only on this image and these operators.
- Pixel response is not physical thickness; pixel-size, PSF/MTF, exposure, scanner, and phantom calibration remain necessary.
- The visual change is deliberately subtle because measurable-structure non-regression takes priority over aggressive sharpening.

V21 extends shape protection from a single width statistic to quantized independent-edge, center, FWHM, per-lamella, and per-interlayer closure, with terminal convergence, a fixed write contract, and atomic publication. On this image, `precision_balanced` adds a small directional denoising gain while passing every implemented release gate. It is suitable as a `MEASUREMENT_CANDIDATE` for downstream detection experiments, not as a substitute for calibrated measurement validation.
