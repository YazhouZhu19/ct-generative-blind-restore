# v15 Structure-Carrier Dual-Output Technical Report

## 1. Problem and conclusion

The v11/v13 paths can project generated lamella centerlines, endpoints, widths, and interlayer geometry onto blind-denoised constraints. However, analytic stripe replacement still changes real local undulations, defects, axial intensity variation, and interlayer texture. Similar geometry statistics are not equivalent to pixel-level structural fidelity.

v15 stops asking one image to perform the conflicting jobs of generative beautification and quantitative measurement. It emits two semantically separate products:

```text
source-sized generation -> original post-processing -> v13 detail projection -> VISUAL_ONLY
raw image -> blind denoising and geometry locking --------------------------> MEASUREMENT
```

The visual output retains generative enhancement. The measurement output has exactly zero generated-pixel contribution. Both remain 2200×1600, but only the measurement branch is a candidate input for downstream detection; physical measurements still require pixel calibration and scanner PSF/MTF validation.

## 2. Method contributions

### 2.1 Observational data consistency instead of parameter-only constraints

Earlier profiles constrain centers, dimensions, and endpoints while permitting the generator to modify local structure between them. v15 defines the same-coordinate blind-denoised image as an immutable structural carrier: every measurement-output pixel comes from that carrier, and generated grayscale or texture is excluded.

### 2.2 No hidden structural transform

The default measurement profile enforces:

- no scaling, resampling, registration, or deformation;
- no analytic lamella replacement;
- no generated low frequency, texture, or background in the measurement output;
- no display-range remapping of the carrier, because saturation can move an FWHM crossing;
- optional edge-adaptive residual shrinkage and transverse unsharp parameters remain disabled unless separately validated.

### 2.3 Dual output instead of ROI blending

An intermediate experiment used the blind guide inside the component ROI and generated pixels outside. Although the ROI structure correlation approached one, the boundary produced a visible seam and made pixel provenance ambiguous to downstream software. The final implementation uses a full-frame structural-carrier measurement image and emits the generative result separately.

### 2.4 Multiscale structural audit

In addition to endpoint, length, FWHM, and spacing audits for 100 lamellae and 98 interlayers, v15 measures:

- low-frequency correlation after Gaussian `sigma=4`;
- mid-frequency correlation using `G_0.8-G_4`;
- smoothed gradient-magnitude correlation;
- per-lamella axial contrast correlation against adjacent gaps.

Detailed evidence is written to `structure_detail_consistency.csv`, `lamella_dual_evidence_comparison.csv`, `interlayer_dual_evidence_comparison.csv`, and `hard_shape_projection_metrics.json`.

## 3. Current-image result

| Metric | v15 measurement | v15 visual |
|---|---:|---:|
| Dimensions | 2200×1600 | 2200×1600 |
| Generated-pixel weight | 0 | Nonzero; visual only |
| Lamellae | 49 left + 51 right = 100 | 100 |
| Interlayers | 98 | 98 |
| Lamella-width median / P95 error | 0% / 0% | 0.0169% / 0.2157% |
| Lamella endpoint P95 error | 0 px | 0.1603 px |
| Lamella length P95 error | 0.0599 px | 0.1954 px |
| Interlayer-width P95 error | 0% | 0.1404% |
| Interlayer-length P95 error | 0.0799 px | 0.1655 px |
| Low-frequency structural correlation | 1.0000 | 0.9414 |
| Mid-frequency structural correlation | 1.0000 | 0.6474 |
| Gradient correlation | 1.0000 | 0.4842 |
| Median / P10 axial-detail correlation | 1.0000 / 1.0000 | 0.8913 / 0.8221 |
| Edge clarity relative to raw | -2.78% | +17.58% |

The small edge-clarity reduction in the measurement branch is a consequence of blind-noise suppression. That output exactly matches the measurement guide and passes every guide-geometry, raw-nonregression, interlayer, and clarity guardrail. The visual companion is sharper but is excluded from quantitative measurement.

## 4. Code and reproduction

```bash
python app/run_v15_pipeline.py \
  --source input/source_16bit.tif \
  --guide input/measurement_guide_16bit.tif \
  --generated input/generative_candidate_visual_only.png \
  --outdir results_generative_shape_v15_dual_output
```

Docker entry point:

```bash
docker compose run --rm ct-v15-dual
```

The core is `structure_carrier_projection()` in `app/generative_shape_project.py`; `app/run_v15_pipeline.py` provides one-command orchestration. The complete workflow ran successfully in the `ct-generative-blind-restore:cpu` container. After adding the v16, v17, and v18 guardrail tests, the current 41-test suite passes.

## 5. Usage boundary

- `FINAL_VISUAL_ONLY_*.tif` is for inspection, presentation, and model research; do not measure it.
- `FINAL_MEASUREMENT_structure_preserved_*.tif` is a candidate image input for detection/measurement, but numerical values should be cross-checked against the exported lamella/interlayer CSV files.
- Converting pixels to millimetres or micrometres requires calibrated pixel size plus scanner PSF/MTF and reference-object validation.
- This is a single-image internal validation, not metrology certification or evidence of cross-device generalization.
