# Docker Container Validation

## Environment

- Docker Desktop server: `29.7.2`
- Platform: Linux/arm64
- Allocated resources: 8 CPUs, approximately 3.8 GiB RAM
- Image: `ct-generative-blind-restore:cpu`
- Final image digest/ID: `sha256:51bd596a3b3266317b9496b2817476970e21ae3a004a551271a37670542be5ec`
- Final image size: 271,904,966 bytes

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

The smoke run intentionally uses only ten iterations to validate image loading, training, strict blind inference, geometry-locked axial candidate selection, the complete original enhancement/fog-cleanup chain, v6 boundary cleanup, v7 hybrid residual denoising, TIFF writing, v8 summary metrics, and the fixed-guide independent audit inside the container. The production result uses 600 iterations in the same pinned dependency environment. The current repository suite contains 32 passing unit tests.

The Compose run used `--rm`; no stopped smoke-test container remained afterward. Runtime outputs are under `results_sota/container_smoke_v8/` and are excluded from Git.

## v13 Direct Blind-Guide Validation

The v13 post-processing and hard-projection commands were executed with the same `ct-generative-blind-restore:cpu` image and completed with exit code 0. The final file is a 2200×1600, 16-bit grayscale TIFF. The current 32-test suite passes inside the container, including tests that verify blind-guide axial-detail transfer, reject mismatched guide dimensions, enforce the explicit source-size lock, and prove zero generated-pixel contribution in the v15 measurement core.

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
- All 32 repository unit tests pass in the container.

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

The container also ran all 32 unit tests successfully. Two dedicated v15 tests verify that changing the generated image cannot change measurement-core pixels and that a mismatched guide size is rejected; the v16 test rejects excessive lamella-width drift.

## v16 Measurement-Quality Validation

The v16 selector was run against the v15 2200×1600 uint16 measurement carrier in the CPU container. Stronger lamella denoising was rejected after the independent per-lamella audit detected unacceptable P95 width drift. The selected candidate uses lamella strength `0.00` and central-region NLM strength `0.92`.

- Central high-frequency noise reduction: `9.77%`
- Lamella/interlayer width P95 error: `0% / 0%`
- Endpoint P95 error: `0 px`
- Low/mid-frequency correlation: `1.0000 / 0.9999`
- Median axial-detail correlation: `1.0000`
- SSIM against v15: `0.99963`
- All strict v16 geometry, detail, edge-retention, and SSIM guardrails: pass
