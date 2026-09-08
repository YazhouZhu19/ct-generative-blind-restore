# Project Structure

## v15 dual-output measurement workflow

```text
app/
  run_v15_pipeline.py              one-command dual-output runner
  generative_shape_project.py      v11/v13/v15 profile-aware projection and audit
compose.yaml                       ct-v15-dual container entry point
STRUCTURE_CARRIER_DUAL_OUTPUT_V15_REPORT*.md
                                    bilingual v15 method and validation reports
```

v15 emits an explicitly synthetic `FINAL_VISUAL_ONLY_*` companion and a separate `FINAL_MEASUREMENT_structure_preserved_*` image. The latter contains zero generated pixels, no analytic stripe replacement, and no post-guide resampling, warping, or intensity remapping.

## v16 measurement-quality refinement

- `app/measurement_quality_optimize.py`: searches guide-only, capped, zero-phase residual candidates under strict v15 geometry/detail guardrails.
- `MEASUREMENT_QUALITY_V16_REPORT*.md`: bilingual method, rejection evidence, selected parameters, and limitations.
- `ct-v16-measurement-quality`: container entry point.

The selected current-image profile freezes lamella/interlayer pixels and denoises only the central solid region. v15 remains the immutable audit baseline.

## v17 structure-conditioned diffusion experiment

- `app/structure_conditioned_diffusion.py`: compact bounded-residual DDIM conditioned by the v16 carrier, curved centerlines, finite-width boundaries, endpoints, interlayers, confidence, uncertainty, and a low-frequency appearance proposal.
- `STRUCTURE_CONDITIONED_DIFFUSION_V17_REPORT*.md`: bilingual architecture, loss, sampling, current-image results, and validation boundary.
- `ct-v17-structure-diffusion`: CPU container entry point.

Unlike v15/v16 measurement outputs, the v17 result contains generated residual pixels. It is therefore labelled `MEASUREMENT_CANDIDATE` and must retain v16 as the fallback and audit reference.

## v18 constrained-detail fusion experiment

- `app/constrained_detail_fusion.py`: applies zero-phase boundary/endpoint enhancement and eroded-interlayer residual shrinkage inside v11-style finite-width constraint fields, followed by selective per-layer rollback.
- `tests/test_constrained_detail_fusion.py`: verifies identity behavior, canvas preservation, row-wise width normalization, and exact unsafe-structure restoration.
- `CONSTRAINED_DETAIL_FUSION_V18_REPORT*.md`: bilingual method, supplied-image metrics, and validation boundary.
- `ct-v18-constrained-detail-fusion`: CPU container entry point.

V18 writes no analytic ribbon pixels and performs no resize, registration, or warp. It retains generated v17 pixels and is therefore still a `MEASUREMENT_CANDIDATE`, with v16 and the exported numeric constraints remaining authoritative.

## v19 measurement-invariant zoned restoration experiment

- `app/structure_anchored_multiregion_denoise.py`: retains v17/v18, applies independently gated residual cleanup to the complete lamella/interlayer stacks, central solid, endpoint-exterior fog, and low-structure background, then performs selective per-layer rollback.
- `tests/test_structure_anchored_multiregion_denoise.py`: covers identity behavior, endpoint hard anchors, transverse non-mixing, uint16 quantization, measurement-operator support, and row-wise width invariance.
- `MEASUREMENT_INVARIANT_ZONED_RESTORATION_V19_REPORT*.md`: bilingual design, audit protocol, current-image evidence, and validation limits.
- `ct-v19-measurement-invariant-zoned`: default CPU Compose entry point with read-only v16/v18 inputs and an isolated v19 result volume.

V19 performs no resize, registration, warp, or analytic lamella redraw. Endpoint-envelope pixels are restored exactly, candidates are audited after uint16 quantization, and the written TIFF is reloaded for a second release audit. Because retained v17 generative pixels remain present, the result is labelled `MEASUREMENT_CANDIDATE`; v16 and the exported numeric constraints remain authoritative until calibrated multi-image validation.

## v20 measurement-safe TV post-processing experiment

- `app/measurement_safe_postprocess.py`: searches seven low-strength Chambolle-TV residual profiles only inside eroded, softly gated non-measurement zones while hard-copying every protected v19 uint16 pixel. Each mask receives one explicit exterior zero-padding contour before an 8 px distance-transform smoothstep ramp, which is exactly zero outside each writable support and on the first inside contour.
- `tests/test_measurement_safe_postprocess.py`: contains 12 v20-specific tests covering identity, disjoint gates, zero-mean weighted residuals, the smoothstep contour, allowed-write containment, configured central ROI locking, direct structure drift, fixed-support high-frequency metrics, and TIFF round-trip fidelity. The complete repository suite contains 59 passing tests.
- `MEASUREMENT_SAFE_TV_POSTPROCESS_V20_REPORT*.md`: bilingual method, filter selection evidence, supplied-image audit, and validation boundary.
- `ct-v20-measurement-safe-postprocess`: default CPU Compose entry point with read-only source/v16/v19 mounts and an isolated v20 result volume.

V20 runs no generator and performs no resize, registration, resampling, warp, analytic redraw, contrast remapping, or sharpening. The complete lamella/interlayer stack, measurement-operator support, configured central ROI border/ring, endpoint-envelope support, and strong edges are locked to v19 bit-for-bit. The selected `tv_balanced_strong_post` profile uses `tv12` with blends 0.40/0.55/0.50 for central/fog/flat regions. On the supplied image, fixed writable-support HF-RMS reductions are 1.6893%/0.6311%/5.1796%, complementary fixed-support Haar-detail reductions are 2.4734%/0.6880%/7.1663%, and SSIM against v19 is 0.99999435. Candidate and post-write release audits cover every image row, direct equality of 100 lamella and 98 interlayer geometry rows, the configured-ROI fixed-line tracker, local SSIM/gradient fidelity, allowed-write containment, soft-gate seams, clipping, and global SSIM. All 88,375 changed pixels are confined to writable support; their absolute-delta P99/maximum is 66/118 DN, while the inner seam is 1/7 DN at P95/maximum. These are single-image complementary no-reference high-frequency proxies, not independent evidence against clean truth. No claim is made that the configured ROI ring is a detected physical object boundary. Retained v17 pixels mean the result remains a `MEASUREMENT_CANDIDATE` and requires calibrated multi-image validation.

## Preserved v11 release

```text
app/
  run_v11_pipeline.py              one-command deterministic v11 runner
  generative_shape_constraint.py   guide-first lamella/interlayer measurement
  generative_postprocess.py        original post-processing plus explicit source-size lock
  generative_shape_project.py      v11/v13/v15 profile-aware projection
config/
  v11_profile.json                 frozen v11 parameters and expected metrics
  v11_reproduction_validation.json exact container and pixel-match evidence
tests/                              geometry, denoising, and projection tests
Dockerfile.cpu / Dockerfile.cuda   container images
compose.yaml                       standard container services
requirements-*.txt                 pinned dependency sets
V11_STABLE_RELEASE*.md             stable-release instructions
GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT*.md
                                    complete v11 technical reports
```

`generative_shape_project.py` defaults to `--profile v11`. The later direct-detail experiment is activated only by explicitly passing `--profile v13`; a non-zero detail-guide weight is rejected under the frozen v11 profile.

## Supporting measurement chain

- `pipeline.py`: original blind denoising and visual post-processing implementation.
- `sota_geometry_blind.py`: stronger blind model and geometry-locked directional stage.
- `quality_optimize.py`: guarded contrast and edge enhancement.
- `boundary_cleanup.py`: geometry-aware boundary cleanup.
- `residual_denoise.py`: bounded residual denoising.
- `length_optimize.py`: fixed-guide endpoint and length audit.
- `run_sota_pipeline.py`: complete non-generative measurement-track orchestration.

## Documentation

- `README.md`: English project entry point; identifies v11 as preserved.
- `METHOD_AND_CODE_GUIDE.md`: complete Chinese method and code guide.
- `V11_STABLE_RELEASE.md` / `V11_STABLE_RELEASE_ZH.md`: frozen release use.
- `CONTAINER_VALIDATION.md`: Docker validation record.
- Versioned reports remain available for audit and historical comparison.
- `reports/technical_innovation/`: bilingual long-form technical report artifacts.

## Runtime results

Runtime image data is excluded from Git. In the organized local experiment bundle it is stored as:

```text
experiment/input/source_16bit.tif
experiment/guide/MEASUREMENT_sota_geometry_blind_16bit.tif
experiment/reference/appearance_reference.jpg
experiment/results_v11/
  00_guide_constraints/
  01_generated/                       native candidate + source-sized pre-postprocess copy
  02_postprocessed/
  03_hard_shape_projection/
  generation_manifest.json
experiment/results_v11_size_locked/
  02_postprocessed/
    GENERATIVE_visual_only_postprocessed_size_locked.png
    GENERATIVE_visual_only_postprocessed_size_locked_16bit.tif
    size_lock_manifest.json
  03_hard_shape_projection/
    GENERATIVE_shape_hard_projected.png
    GENERATIVE_shape_hard_projected_16bit.tif
    hard_shape_projection_metrics.json
  v11_release_manifest.json
experiment/results_generative_shape_v15_dual_output/
  FINAL_VISUAL_ONLY_enhanced_2200x1600_16bit.tif
  FINAL_MEASUREMENT_structure_preserved_2200x1600_16bit.tif
  03_visual_detail_projection/
  04_measurement_structure_carrier/
  v15_release_manifest.json
experiment/results_generative_shape_v19_structure_anchored_multiregion/
  MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion_16bit.tif
  MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion.png
  MEASUREMENT_CANDIDATE_v19_comparison.png
  AUDIT_v19_multiregion_masks.png
  lamella_v19_comparison.csv
  interlayer_v19_comparison.csv
  local_row_width_v19_comparison.csv
  structure_detail_v19.csv
  structure_anchored_multiregion_v19_metrics.json
experiment/results_generative_shape_v20_measurement_safe_postprocess/
  MEASUREMENT_CANDIDATE_v20_measurement_safe_postprocessed_16bit.tif
  MEASUREMENT_CANDIDATE_v20_measurement_safe_postprocessed.png
  MEASUREMENT_CANDIDATE_v20_comparison.png
  AUDIT_v20_measurement_safe_masks.png
  lamella_v20_comparison.csv
  interlayer_v20_comparison.csv
  local_row_width_v20_comparison.csv
  structure_detail_v20.csv
  measurement_safe_postprocess_v20_metrics.json
```

`results_v11_size_locked` is the recommended organized result. The native generated candidate is retained for provenance, while its source-sized copy is created before post-processing; all enhancement operations, the projection input, and final PNG/TIFF therefore use exactly the same width and height as the source image.

## Deliverable split

- `ct-generative-blind-restore-v11-source.zip`: safe source package for GitHub; no image data.
- `ct-generative-blind-restore-v11-full.zip`: local full experiment archive with user image data; do not upload publicly without review.

Both expanded folders contain `FILE_MANIFEST_SHA256.txt`. The parent deliverables directory contains checksums for the ZIP archives.
