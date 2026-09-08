# v13 Direct Blind-Guide and Structural-Detail Consistency Report

## 1. Objective

Version 11 measures 100 lamellae and 98 interlayers from a registered blind-denoised image and uses an analytic hard projection to lock centerlines, endpoints, lengths, and FWHM. Its limitation is that the denoised image participates only through derived geometry: the hard renderer can make longitudinal lamella appearance overly uniform.

Version 13 deliberately returns to the v11 finite-width renderer instead of adopting the sharper v12 transition constants. It adds a second, same-coordinate detail path: the same blind-denoised image is used both to measure the constraints and to guide the generator and final projector directly. The goal is to retain authentic longitudinal attenuation, low-frequency structure, central-bridge detail, and surrounding-object continuity without permitting any length or width change.

## 2. Strict Input Roles

```text
raw 16-bit image ----------------------> sole scene, FOV, and content target
       |
       +-- blind denoising --> guide ---+--> v11 geometry map and CSV
                                       +--> direct generation detail guide
                                       +--> hard-projection axial detail guide
unregistered appearance reference ----> boundary cleanliness and tone only
```

The raw image defines the full scene. The registered blind-denoised image supplies geometry measurements and authentic structure. The condition map encodes exactly 49 left lamellae, 51 right lamellae, and 98 interlayers. The appearance reference cannot supply coordinates, counts, or dimensions.

## 3. Method

### 3.1 Direct Detail Guidance During Generation

The generative editor receives the raw target, registered blind-denoised guide, v11 condition map, and appearance reference as four explicitly separated roles. The prompt requires same-coordinate low/mid-frequency attenuation, per-lamella longitudinal brightness variation, and central-bridge structure to follow the guide. Geometry constraints outrank visual aesthetics.

### 3.2 Robust Intensity Matching and Foreground Fusion

The guide's `1%–99.5%` intensity interval is mapped to the generated candidate without registration or warping. Within a feathered foreground region spanning both stacks and the central bridge,

\[
B=(1-\lambda)G_{\sigma}(I_{gen})+\lambda G_{\sigma}(I_{guide}),
\qquad \lambda=0.55.
\]

Transverse low-pass filtering suppresses generator-invented periodicity inside the stacks; lightly smoothed guide structure is retained in the central bridge and other non-lamella foreground.

### 3.3 Per-Lamella Axial Structure Curves

For lamella `i`, contrast is sampled at every row against its adjacent gaps:

\[
c_i(y)=I(y,p_i(y))-\frac{I(y,p_i(y)-d_i)+I(y,p_i(y)+d_i)}{2}.
\]

After robust normalization and axial smoothing, the modulation is clipped to `[0.58, 1.42]`. It changes analytic-ribbon amplitude only; it cannot change the measured centerline, width, or endpoints. Detail modulation fades to one within `4–18 px` of each endpoint so that a real brightness transition cannot bias the half-height endpoint operator.

### 3.4 Unchanged v11 Geometry Lock

Each lamella retains the v11 dual-logistic finite-width profile: `0.38 px` transverse transition and `0.55 px` endpoint transition. Three closed-loop rounds use the audit FWHM operator to correct rasterization. Centerlines, endpoints, lengths, widths, and interlayers come exclusively from the blind-denoised measurement constraints.

### 3.5 New Structural-Detail Audit

Version 13 adds foreground low-frequency, mid-frequency, gradient-magnitude, and 100 per-lamella axial-detail correlations. Per-lamella values are written to `structure_detail_consistency.csv`; aggregates and guardrails are written to `hard_shape_projection_metrics.json`.

## 4. Result on the Supplied Image

| Metric | v13 result |
|---|---:|
| Constraint-matched lamellae | 49 left + 51 right = 100 |
| Constraint-matched interlayers | 98 |
| Lamella/interlayer dual-evidence pass | 100/100; 98/98 |
| Guide-coordinate endpoint error P95 | 0.0425 px |
| Guide-coordinate length error P95 | 0.0555 px |
| Lamella FWHM relative error median / P95 | 0.0124% / 0.1521% |
| Interlayer-width relative error P95 | 0.0830% |
| Target / output median lamella width | 4.251057 / 4.251422 px |
| Raw / guide / output edge clarity | 0.3061 / 0.2976 / 0.3631 |
| Soft / final low-frequency correlation | 0.9495 / 0.9141 |
| Soft / final mid-frequency correlation | 0.7060 / 0.6597 |
| Soft / final median axial-detail correlation | 0.3212 / **0.8884** |
| Final axial-detail correlation P10 | 0.8120 |

Removing generator-invented periodicity and restoring exact FWHM reduces global low-frequency correlation by 0.0354, still above 0.90 and within the allowed 0.05 drop. The geometry-locked per-lamella axial correlation improves by 0.5672. All geometry, clarity, and detail guardrails pass.

## 5. Reproduction

```bash
python app/generative_postprocess.py \
  --generated results_generative_shape_v13/01_generated/GENERATIVE_v13_blind_guide_detail_conditioned_raw.png \
  --outdir results_generative_shape_v13/02_postprocessed

python app/generative_shape_project.py \
  --profile v13 \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated results_generative_shape_v13/02_postprocessed/GENERATIVE_visual_only_postprocessed.png \
  --outdir results_generative_shape_v13/03_detail_consistent_projection \
  --detail-guide-weight 0.55
```

Weights `0.35`, `0.45`, and `0.55` were evaluated. The final value of `0.55` had the highest structural correlation while every geometry guardrail remained satisfied.

## 6. Scope Boundary

The final image still contains generated content. It is suitable for visual enhancement, algorithm research, and assisted inspection, but not as metrology-certified data or noise-free truth. Thickness, length, and defect acceptance require the non-generative 16-bit measurement track, calibrated pixel size, and system PSF/MTF validation.
