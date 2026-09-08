# v6 Geometry-Locked Boundary Cleanup Report

## Outcome

Version 6 retains the v5 generative-prior blind denoiser and the complete original enhancement/post-processing chain, then adds a geometry-locked boundary-cleanup stage. The final 16-bit output is:

```text
results_sota/03_boundary_cleanup/QUALITY_boundary_clean_16bit.tif
```

The supplied reference image defines appearance intent only: a quiet dark exterior, clean sheet boundaries, and an intact central highlight. It is not used for registration, training supervision, geometry fitting, or pixel writeback. All boundary masks come from subpixel endpoint coordinates measured on the raw 16-bit image, while output pixels are computed only from the v5 enhanced result.

## New method

1. **Raw-coordinate endpoint envelopes** fit upper and lower curves for the left and right lamella groups and produce soft boundary, exterior, and interior masks.
2. **Axial fine denoising** uses an anisotropic `σ=(2.00, 0.18)` kernel, smoothing mainly along lamella length without averaging across transverse thickness edges.
3. **Exterior cleanup** applies `σ=1.15` fine smoothing only outside the endpoint envelope and suppresses positive `G₂.₂-G₇` veil residuals.
4. **Endpoint-core protection** prevents changes around every raw upper/lower endpoint; all remaining pixel changes are capped at `0.8×MAD`.
5. **Guarded selection** searches eight parameter sets and jointly checks FWHM, endpoints, length, acutance, contrast, boundary gradient, and SSIM.

## Measurements on the supplied image

Relative to the complete v5 enhanced output:

| Metric | v6 result |
|---|---:|
| Exterior high-frequency noise reduction | 4.26% |
| Exterior positive-halo reduction | 10.99% |
| Interior axial-noise reduction | 10.12% |
| Transverse edge-acutance retention | 100.01% |
| Local-contrast retention | 100.00% |
| Boundary-gradient retention | 95.70% |
| SSIM versus v5 | 0.998538 |
| Maximum median-FWHM change versus raw | 1.19% |

The independent audit detects 100 lamellae: 98 pass automatically and two require review because their raw endpoint references are intrinsically uncertain. For reliable lamellae, endpoint-shift P95 is 0.248 px, length-change P95 is 0.402 px, and maximum length change is 0.499 px. All configured guardrails pass.

## Reproduction

```bash
docker compose run --rm ct-sota
```

The cleanup stage can also run independently:

```bash
python app/boundary_cleanup.py \
  --source input/source_16bit.tif \
  --input results_sota/02_geometry_quality/QUALITY_boundary_preserved_16bit.tif \
  --reference-style input/reference_style.jpg \
  --outdir results_sota/03_boundary_cleanup
```

`--reference-style` records only the reference dimensions and role; omitting it does not change the algorithm. Before measurement use, inspect `boundary_cleanup_metrics.json`, the endpoint overlay, and `04_length_audit/length_qa.json`.

## Limitations

This remains a single-image internal validation without paired clean truth, repeated scans, or cross-device testing. Pixel-coordinate consistency is not millimetre/micrometre certification; physical measurement still requires pixel calibration, a traceable standard, system PSF/MTF characterization, and repeatability studies.
