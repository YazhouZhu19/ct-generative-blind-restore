# v8 Geometry-Locked Strong Directional Denoising Report

## Outcome

Version 8 revisits `directional_continuity` from the public initial repository. Its `sigma=(3,0)` axial filter is the strongest deterministic denoising step in that workflow, but the original gate protects transverse thickness edges only. It does not explicitly protect upper and lower endpoints, so it cannot by itself demonstrate preserved length.

The generative-prior blind model, original enhancement, v6 boundary cleanup, and v7 residual denoising remain active. v8 inserts a **geometry-locked directional projection** after Poisson-Gaussian data projection and before any sharpening or contrast enhancement. A strong axial candidate can enter the downstream chain only if it passes fixed-raw-coordinate thickness, endpoint, length, acutance, contrast, and SSIM guardrails together.

## Where shape consistency is enforced

Final-only correction is insufficient: once aggressive denoising merges an adjacent boundary or erases an endpoint, post-processing can move surviving pixels but cannot reliably recreate lost measurement evidence. v8 therefore constrains geometry at six locations:

1. Before processing: freeze raw centers, curved paths, transverse FWHM, and subpixel endpoints.
2. During training: x/y edge, width-profile, endpoint-profile, and multiscale shape losses.
3. During data projection: raw transverse center anchors and a bounded posterior change.
4. Before strong denoising enters the output: raw transverse and longitudinal gradient gates plus fixed endpoint exclusion zones.
5. During candidate selection: reject any strength that violates count, FWHM, endpoint, length, acutance, contrast, or SSIM limits.
6. After every post-processing stage: run an independent length audit on fixed pre-processing guides and paths.

Step 4 is the principal v8 change. It retains the strong axial cleanup of the initial method while adding the missing longitudinal geometry lock.

## Method

For raw image `I0` and accepted blind posterior projection `I`, compute a zero-phase axial filter:

\[
A=G_{(3,0)}(I)
\]

The raw image supplies transverse and longitudinal safety gates, while fixed raw endpoints form an exclusion mask:

\[
M=\exp[-(g_x/T_x)^2]\,\exp[-(g_y/T_y)^2]\,(1-P_{endpoint})
\]

Each candidate is:

\[
I_\beta=I+\beta M(A-I)
\]

Changes remain inside the selected Poisson-Gaussian projection bound and endpoint cores retain the accepted projection pixels. Neither raw nor generative pixels are written back. The search evaluates `beta={0,0.25,0.40,0.55,0.70}` and maximizes roughness reduction subject to every geometry and appearance constraint.

## Production 600-iteration result

The selector chose `beta=0.55`. Although `beta=0.70` increased blind-stage roughness reduction to 32.18%, its SSIM against the projection input fell to 0.99418, below the 0.995 threshold, so it was automatically rejected.

| Metric | v8 production result |
|---|---:|
| Blind projection roughness reduction vs raw | 26.72% |
| Geometry-locked directional result vs raw | 31.94% |
| Additional directional reduction vs projection | 7.12% |
| Directional SSIM vs projection input | 0.996309 |
| Directional endpoint-shift P95 | 0.034 px |
| Directional length-change P95 | 0.057 px |
| Final lamella-region high-frequency reduction vs raw | 39.47% |
| Final central-highlight noise reduction vs raw | 23.96% |
| Final edge-acutance gain vs raw | 5.21% |
| Final local-contrast gain vs raw | 1.94% |
| Final maximum median-FWHM change | 1.28% |

Relative to the v7 final image, v8 reduces lamella-region high-frequency noise by another 1.77% and central-highlight noise by another 5.05%, while retaining 99.83% of v7 acutance and 99.92% of v7 local contrast. The fixed-guide final audit detects 100 lamellae: 98 pass automatically and two require review because their raw endpoint evidence is intrinsically uncertain. Reliable-lamella endpoint-shift P95 is 0.241 px and length-change P95 is 0.403 px.

## Code and reproduction

- `app/sota_geometry_blind.py`: `geometry_locked_directional_continuity` and strength selection.
- `app/run_sota_pipeline.py`: five-stage orchestration and v8 metrics.
- `tests/test_sota_geometry_blind.py`: endpoint-core immutability test.

Run the complete workflow:

```bash
docker compose run --rm ct-sota \
  --source /data/input/source_16bit.tif \
  --generative-prior /data/input/generative_candidate_visual_only.png \
  --outdir /data/results_sota --iterations 600 --device cpu
```

The production output retains the compatible v7 residual-stage filename `04_residual_denoise/QUALITY_v7_residual_denoised_16bit.tif`, but in the v8 workflow it receives the new geometry-locked directional result. Before measurement, inspect `directional_projection_selection` in `01_sota_blind/sota_run_manifest.json` and the final `05_length_audit/length_qa.json`.

## Limitations

These numbers are single-image internal consistency results, not clean-ground-truth error or metrology certification. Physical length still requires calibrated pixel size, a traceable standard, PSF/MTF characterization, and repeated scans.
