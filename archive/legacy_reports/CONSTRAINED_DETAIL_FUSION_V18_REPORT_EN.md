# v18 Clean-Boundary and Detail-Preserving Fusion Report

## 1. Objective

V11 obtains visually regular boundaries through finite-width analytic ribbons and a three-round FWHM loop, but whole-ribbon reconstruction can overwrite real local waviness, defects, and axial detail. V17 preserves same-coordinate structure much more closely, yet stronger edge enhancement can push a few marginal lamellae beyond width or dual-evidence limits.

V18 keeps the v17 generated-denoising result and imports only v11's finite-width localization and per-structure closed-loop acceptance principle. No analytic ribbon pixels are written into the output. Generated pixels remain present, so the result is still labelled `MEASUREMENT_CANDIDATE`.

## 2. Pipeline

```text
raw 16-bit image + v16 structural carrier
        |-- measure 100 lamellae, 98 interlayers, paths, endpoints, and FWHM
        `-- export robust row-wise width trajectories for constraint audit

v17 structure-conditioned generated result
        |-- zero-phase transverse detail in finite-width boundary fields
        |-- zero-phase longitudinal detail in endpoint fields
        `-- capped residual shrinkage in eroded interlayer cores
                |-- remeasure every lamella and interlayer
                |-- restore each regressing structure exactly to v17
                `-- release only if every global guardrail passes
```

There is no resize, registration, spatial warp, coordinate resampling, or whole-lamella analytic redraw. Pixels outside the target ROI are identical to v17.

## 3. Selected profile

- boundary gain: `0.06`;
- endpoint gain: `0.025`;
- interlayer residual shrinkage: `0.08`;
- maximum per-pixel normalized change: `0.0041896`;
- row-wise width-variation retention for the exported audit constraint: `0.35`.

The interlayer operator starts only after approximately 1.6 pixels of distance from a measured lamella boundary. This keeps denoising away from the samples that determine FWHM.

After initial processing, 74 lamellae retain the enhancement and 26 marginal lamellae are restored to exact v17 pixels together with their adjacent measurement neighborhoods.

## 4. Supplied-image results

| Metric | v18 versus v17 or carrier constraint |
|---|---:|
| Output size/type | 2200×1600 / uint16 |
| Interlayer high-frequency noise reduction | 0.472% |
| Edge-clarity gain | 0.0617% |
| Lamella-width P95 error | 0.3407% |
| Endpoint P95 error | 0.00459 px |
| Length P95 error | 0.06065 px |
| Interlayer-width P95 error | 0.2211% |
| Interlayer-length P95 error | 0.08270 px |
| Lamella/interlayer dual-evidence pass counts | 99 / 96 |
| Low-/mid-frequency correlation | 1.000000 / 0.999987 |
| Gradient correlation | 0.999978 |
| Median/P10 axial-detail correlation | 0.999986 / 0.999974 |
| SSIM against v17 | 0.999998 |

Stronger candidates produced larger visual edge gains but exceeded the `0.35%` lamella-width P95 gate and were rejected. The selected improvement is intentionally limited by the measurement requirement.

## 5. Run

```bash
docker compose run --rm ct-v18-constrained-detail-fusion
```

The main output is `MEASUREMENT_CANDIDATE_v18_clean_edges_detail_preserved_16bit.tif`. The output directory also contains a preview, comparison and intervention audit images, lamella/interlayer/detail CSVs, row-wise width constraints, and a machine-readable metrics manifest.

## 6. Validation boundary

V18 provides auditable incremental visual cleanup with strong consistency relative to the carrier; it is not metrology certification. Physical measurements should continue to use the exported constraints or non-generated v16 carrier until repeated same-device scans, a calibrated phantom, pixel-size calibration, and PSF/MTF validation are complete.
