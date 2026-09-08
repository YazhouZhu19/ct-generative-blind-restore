# Docker Container Validation

## Environment

- Docker Desktop server: `29.7.2`
- Platform: Linux/arm64
- Allocated resources: 8 CPUs, approximately 3.8 GiB RAM
- Image: `ct-generative-blind-restore:cpu`
- Final image digest/ID: `sha256:c8a2910b13f950f7c2955c47ef944ea035a3abeadc5ce4e75a0c73188915e929`
- Final image size: 271,981,539 bytes

The image was built successfully from `Dockerfile.cpu` with the pinned Python packages. A whitelist-style `.dockerignore` prevents raw images, runtime results, model weights, Git metadata, reports, and host Python caches from entering the Docker build context. The final rebuild transferred only the whitelisted context and reused the verified dependency layers.

## End-to-end smoke command

```bash
docker compose run --rm ct-sota \
  --source /data/input/source_16bit.tif \
  --generative-prior /data/input/generative_candidate_visual_only.png \
  --outdir /data/results_sota/container_smoke_v8 \
  --iterations 10 --features 8 --batch 1 --device cpu
```

All five container stages completed with exit code 0:

| Check | Result |
|---|---:|
| Blind projection roughness reduction vs raw | 10.94% |
| Geometry-locked directional result vs raw | 19.80% |
| Additional directional reduction vs blind projection | 9.94% |
| Selected directional strength | 0.55 |
| Directional SSIM vs projection input | 0.995287 |
| Pre-cleanup high-frequency reduction vs raw | 21.96% |
| Final high-frequency reduction vs raw | 29.79% |
| Final central-highlight noise reduction vs raw | 16.59% |
| Final edge-acutance gain vs raw | 16.50% |
| Boundary-exterior noise reduction vs enhanced input | 5.32% |
| Boundary-exterior positive-halo reduction vs enhanced input | 11.68% |
| Lamella-interior axial-noise reduction vs enhanced input | 9.90% |
| Additional v7 residual lamella reduction | 7.30% |
| Additional v7 residual central reduction | 9.90% |
| Maximum median-FWHM change vs raw | 3.25% |
| Independently detected lamellae | 100 |
| Automatic passes | 98 |
| Independent endpoint-shift P95 | 0.237 px |
| Independent length-delta P95 | 0.385 px |
| Blind, directional, width, endpoint, boundary-cleanup, residual-denoising and length guardrails | pass |

The smoke run intentionally uses only ten iterations to validate image loading, training, strict blind inference, geometry-locked axial candidate selection, the complete original enhancement/fog-cleanup chain, v6 boundary cleanup, v7 hybrid residual denoising, TIFF writing, v8 summary metrics, and the fixed-guide independent audit inside the container. The production result uses 600 iterations in the same pinned dependency environment. At that historical revision, the repository suite contained 41 passing unit tests.

The Compose run used `--rm`; no stopped smoke-test container remained afterward. Runtime outputs are under `results_sota/container_smoke_v8/` and are excluded from Git.

## v13 Direct Blind-Guide Validation

The v13 post-processing and hard-projection commands were executed with the same `ct-generative-blind-restore:cpu` image and completed with exit code 0. The final file is a 2200×1600, 16-bit grayscale TIFF. The 41-test suite at that stage passed inside the container, including tests that verify blind-guide axial-detail transfer, reject mismatched guide dimensions, enforce the explicit source-size lock, prove zero generated-pixel contribution in the v15 measurement core, and verify v18's canvas-safe selective rollback.

| v13 check | Result |
|---|---:|
| Constraint-matched lamellae | 100 / 100 |
| Constraint-matched interlayers | 98 / 98 |
| Endpoint error P95 | 0.0425 px |
| Length error P95 | 0.0555 px |
| Lamella FWHM error P95 | 0.1521% |
| Interlayer-width error P95 | 0.0830% |
| Final median axial-detail correlation | 0.8884 |
| Final low/mid-frequency correlation | 0.9141 / 0.6597 |
| Geometry, edge-clarity, and detail guardrails | pass |

## Frozen v11 Reproduction Validation

The preserved `app/run_v11_pipeline.py` entry point was executed in the same CPU container using the archived v11 soft candidate. It regenerated the constraints, original deterministic post-processing, v11 hard projection, and release manifest with exit code 0.

- The regenerated post-processed PNG is byte-for-byte identical to the archived v11 PNG.
- The regenerated final 2200×1600 PNG is byte-for-byte identical to the archived v11 final PNG.
- The regenerated and archived 16-bit TIFF pixel arrays have maximum absolute difference `0` and `0` differing pixels.
- The native post-processed candidate is 1470×1070. The explicit Lanczos size-lock output, hard-projection input, final PNG, and final TIFF are all 2200×1600, matching the source and guide exactly.
- The run recovers 49 left and 51 right lamellae, 98 interlayers, endpoint-shift P95 `0.088836 px`, length-change P95 `0.105491 px`, and FWHM-error P95 `0.332364%`.
- All 37 repository unit tests pass in the container.

The v11 profile rejects a non-zero direct guide-detail weight. Later v13 behavior is available only through an explicit `--profile v13`, so the preserved v11 command cannot silently change behavior.

## v15 Structure-Carrier Dual-Output Validation

`app/run_v15_pipeline.py` was executed end to end in the same CPU container. It completed guide-first constraints, source-size preprocessing, the unchanged visual post-processing stage, the tuned v13 visual projection, the v15 measurement projection, and release-manifest writing with exit code 0. Both final TIFF files are 2200×1600 and 16-bit grayscale.

| v15 check | Measurement branch | Visual branch |
|---|---:|---:|
| Generated-pixel weight | 0 | nonzero |
| Constraint-matched lamellae | 100 / 100 | 100 / 100 |
| Interlayers | 98 | 98 |
| Lamella-width error P95 | 0% | 0.2157% |
| Endpoint error P95 | 0 px | 0.1603 px |
| Length error P95 | 0.0599 px | 0.1954 px |
| Interlayer-width error P95 | 0% | 0.1404% |
| Interlayer-length error P95 | 0.0799 px | 0.1655 px |
| Low/mid-frequency correlation | 1.0000 / 1.0000 | 0.9414 / 0.6474 |
| Median axial-detail correlation | 1.0000 | 0.8913 |
| Geometry, raw-nonregression, interlayer, clarity, and structure guardrails | pass | pass |

The container also ran all 41 unit tests successfully. Two dedicated v15 tests verify that changing the generated image cannot change measurement-core pixels and that a mismatched guide size is rejected; the v16 test rejects excessive lamella-width drift. Five v17 tests cover condition-field protection, the carrier-safe zero initialization, differentiable geometry-loss response, residual capping, and full tile coverage. Four v18 tests cover identity behavior, target-ROI isolation, row-wise width normalization, and exact unsafe-neighborhood restoration.

## v16 Measurement-Quality Validation

The v16 selector was run against the v15 2200×1600 uint16 measurement carrier in the CPU container. Stronger lamella denoising was rejected after the independent per-lamella audit detected unacceptable P95 width drift. The selected candidate uses lamella strength `0.00` and central-region NLM strength `0.92`.

- Central high-frequency noise reduction: `9.77%`
- Lamella/interlayer width P95 error: `0% / 0%`
- Endpoint P95 error: `0 px`
- Low/mid-frequency correlation: `1.0000 / 0.9999`
- Median axial-detail correlation: `1.0000`
- SSIM against v15: `0.99963`
- All strict v16 geometry, detail, edge-retention, and SSIM guardrails: pass

## v17 Structure-Conditioned Diffusion Validation

The production v17 run used the same CPU image, 320 single-image adaptation iterations, 48 diffusion noise levels, six deterministic DDIM sampling steps, 192-pixel tiles, and 40-pixel overlap. It completed with exit code 0 and wrote a 2200×1600 uint16 TIFF.

| v17 check | Result |
|---|---:|
| Selected generated-residual strength | 0.85 |
| Lamella axial-noise reduction vs v16 | 1.05% |
| Central high-frequency reduction vs v16 | 2.67% |
| Flat-region high-frequency reduction vs v16 | 1.51% |
| Lamella-width P95 error | 0.3409% |
| Endpoint-shift P95 | 0.00043 px |
| Interlayer-width P95 error | 0.1911% |
| Interlayer-length P95 error | 0.07970 px |
| SSIM against v16 | 0.999968 |
| Median/P10 axial-detail correlation | 0.999991 / 0.999985 |
| Strict geometry/detail/edge/SSIM guardrails | pass |

Only the configured target ROI changed; the rest of the 2200×1600 canvas is pixel-identical to v16. Generated residual pixels are present, so the result remains a measurement candidate pending multi-image and calibrated-phantom validation.

## v18 Constrained-Detail Fusion Validation

The v18 full-resolution run completed in the same `ct-generative-blind-restore:cpu` image. It consumed the accepted v17 TIFF and v16 carrier, wrote a 2200×1600 uint16 TIFF, and applied no resampling, registration, warp, or analytic-ribbon pixel replacement.

- Rebuilt CPU image ID: `sha256:95e0e008639fd9db093529ab104af12c4ce7e1574a7d769b645fab53a7097b8f`.
- Final v18 TIFF SHA-256: `d9eee06e32596423e982fc57a9f4538767073e40d5f8ba0ffabefbc2643a4439`.

| v18 check | Result |
|---|---:|
| Lamellae retaining enhancement / rolled back | 74 / 26 |
| Interlayer high-frequency reduction vs v17 | 0.472% |
| Edge-clarity gain vs v17 | 0.0617% |
| Lamella-width P95 error | 0.3407% |
| Endpoint-shift P95 | 0.00459 px |
| Interlayer-width P95 error | 0.2211% |
| Interlayer-length P95 error | 0.08270 px |
| Median/P10 axial-detail correlation | 0.999986 / 0.999974 |
| SSIM against v17 | 0.999998 |
| Strict geometry/detail/edge/SSIM guardrails | pass |

Every stronger candidate failed the lamella-width P95 gate and was rejected. The four v18 tests and all 37 pre-existing tests pass in the container.

## v19 Measurement-Invariant Zoned Restoration Validation

The CPU image was rebuilt after adding `app/structure_anchored_multiregion_denoise.py`, then the full 2200×1600 candidate grid was run directly from the image without bind-mounting application code. The run completed with exit code 0, selected `(stack, central, fog, flat) = (0.02, 0.50, 0.60, 0.50)`, wrote a uint16 TIFF, reloaded it, and passed the second release audit.

- Rebuilt CPU image ID: `sha256:43b880db6e99c2218910a8daf876de1c3c1e4fa25d5d024b62613c37ea0a4c30`.
- Final v19 TIFF SHA-256: `3db3b3b6221a929772ef477b356359123b8f9e0cf7c3cb26969cdd5fc7e1b80a`.
- Container-read TIFF metadata: `(1600, 2200)`, `uint16`, intensity range `0..65535`.
- All six v19-specific tests and the complete 47-test repository suite pass against the rebuilt image.

| v19 check | Result |
|---|---:|
| Central-solid high-frequency residual reduction vs v18 | 9.51% |
| Endpoint-exterior fog residual reduction vs v18 | 14.26% |
| Low-structure background residual reduction vs v18 | 11.99% |
| Lamella / interlayer residual reduction vs v18 | 0.0561% / 0.1097% |
| Selectively restored marginal lamellae | 9 / 100 |
| Lamella-width P95 error | 0.3150% |
| Interlayer-width P95 error | 0.2096% |
| Endpoint-shift P95 | 0.004734 px |
| Row-wise width-drift P95 / maximum | 0.003053 / 0.051013 px |
| Lamella / interlayer dual-evidence passes | 99 / 96 |
| SSIM against v18 | 0.999892 |
| Changed uint16 hard-anchor pixels | 0 |
| Post-write geometry, topology, detail, SSIM, and anchor audit | pass |

The residual reductions are fixed-operator high-frequency proxies rather than error against an unavailable noise-free ground truth. Generated v17 pixels remain in the v19 image, so the release is labelled `MEASUREMENT_CANDIDATE`; v16 and the exported numeric constraints remain the authoritative fallback until multi-image and calibrated-phantom validation is available.

## v20 Measurement-Safe TV Post-Processing Validation

The seven-member 2200×1600 v20 candidate grid was executed directly from the rebuilt CPU image without an application-code bind mount and completed with exit code 0. V20 consumed the v19 candidate and v16 carrier, ran no new generator, and selected `tv_balanced_strong_post`: all three writable regions use `tv12`, with `(central, fog, flat)` strengths `(0.40, 0.55, 0.50)`; the complete lamella/interlayer stack strength is zero. The more aggressive audit-only candidate was rejected by incremental global-correlation, local-fidelity, clipping, and uint16-delta limits. The output TIFF was reloaded and passed the second uint16 release audit.

- Rebuilt CPU image ID: `sha256:c8a2910b13f950f7c2955c47ef944ea035a3abeadc5ce4e75a0c73188915e929`.
- Final v20 TIFF SHA-256: `4a958a48f4fc22168b11c7b9bf7beeb2c55d2ae97b90b2d5700570da8b54b1e2`.
- TIFF metadata from the release audit: `(1600, 2200)`, `uint16`, pixel round trip exact.
- All 12 v20-specific tests and the complete 59-test repository suite pass with the image's embedded application code and the repository tests mounted read-only; `docker compose config --quiet` also passes.

| v20 check | Result |
|---|---:|
| Generator executed in v20 | no |
| Fixed writable-zone high-frequency reduction, central / fog / flat | 1.6893% / 0.6311% / 5.1796% |
| Fixed writable-zone Haar-detail mean-absolute reduction, central / fog / flat | 2.4734% / 0.6880% / 7.1663% |
| Full v19-operator high-frequency reduction, central / fog / flat | 1.2454% / 0.0138% / 0.5186% |
| Complete lamella / interlayer residual change | 0% / 0% |
| Direct per-record output-geometry drift, 100 lamellae / 98 interlayers | 0 px / 0 px maximum |
| Endpoint-shift P95 | 0.004734 px |
| Length-delta P95 | 0.060711 px |
| Lamella / interlayer width P95 error | 0.3150% / 0.2096% |
| Every-row width-drift median / P95 / maximum | 0 / 0 / 0 px |
| Configured central-ROI edge-position drift P95 / maximum | 0.0000003 / 0.001604 px |
| Configured central-ROI tracker width drift P95 / maximum | 0 / 0.000962 px |
| Configured central-ROI tracker height drift P95 / maximum | 0.000019 / 0.001604 px |
| Configured central-ROI edge-transition-width drift P95 / maximum | 0.000012 / 0.000246 px |
| Configured central-ROI edge peak-strength / confidence P05 retention | 1.0000 / 1.0000 |
| Changed hard/stack/central-boundary/strong-edge/operator-lock pixels | 0 / 0 / 0 / 0 / 0 |
| Changed pixels outside target ROI | 0 |
| Changed pixels outside declared writable support | 0 |
| Zero-weight contour delta maximum; next soft-seam P95 / maximum | 0; 1 / 7 DN |
| Changed pixels in full image | 88,375 (2.5107%) |
| Absolute uint16 delta P50 / P95 / P99 / maximum | 9 / 44 / 66 / 118 DN |
| New zero / saturation clipping pixels | 0 / 0 |
| Local SSIM-map mean, central / fog / flat | 0.9999288 / 0.9999995 / 0.9999933 |
| Local gradient correlation, central / fog / flat | 0.9998918 / 0.9999942 / 0.9975357 |
| Low / mid / gradient correlation | 0.9999994 / 0.9999217 / 0.9998689 |
| Full-target SSIM against v19 (corruption sentinel only) | 0.9999943 |
| Post-write geometry, topology, central-integrity, detail, noise, clipping, and lock audit | pass |

The first two residual rows use immutable masks restricted to pixels v20 is permitted to modify and provide complementary fixed-Gaussian and Haar-detail checks; the third applies v19's broader fixed regional operators. They are no-reference residual proxies, not error against a noise-free target and not evidence of SOTA performance. “Central ROI” refers to the configured rectangle and its fixed-line edge tracker, not a validated physical-object boundary. V20 itself introduces no generated pixels, but the input retains v17 generated residuals, so the release remains a `MEASUREMENT_CANDIDATE` and v16 remains the authoritative fallback.
