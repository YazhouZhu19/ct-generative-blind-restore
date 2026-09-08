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
```

`results_v11_size_locked` is the recommended organized result. The native generated candidate is retained for provenance, while its source-sized copy is created before post-processing; all enhancement operations, the projection input, and final PNG/TIFF therefore use exactly the same width and height as the source image.

## Deliverable split

- `ct-generative-blind-restore-v11-source.zip`: safe source package for GitHub; no image data.
- `ct-generative-blind-restore-v11-full.zip`: local full experiment archive with user image data; do not upload publicly without review.

Both expanded folders contain `FILE_MANIFEST_SHA256.txt`. The parent deliverables directory contains checksums for the ZIP archives.
