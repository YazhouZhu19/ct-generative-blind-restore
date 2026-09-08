# Generative-Prior, Geometry-Constrained Blind Denoising v5

## Outcome

Version 5 implements and validates the requested order: a generative prior, geometry-constrained blind denoising, the original enhancement/post-processing chain, and an independent length audit. The recommended output is:

```text
results_sota/02_geometry_quality/QUALITY_boundary_preserved_16bit.tif
```

On the supplied 2200 × 1600 16-bit image, final flat-region high-frequency roughness falls by **28.66%** versus raw, digital edge acutance rises by **5.60%**, and local contrast rises by **2.04%**. The maximum left/right median-FWHM change is **1.22%**. The independent audit detects 100 lamellae: 98 pass automatically and two are flagged because their raw endpoints are intrinsically uncertain.

## Model selection scope

The blind formulation is based primarily on [Blind2Sound (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Blind2Sound_Self-Supervised_Image_Denoising_without_Residual_Noise_ICCV_2025_paper.html). Its adaptive re-visible formulation and Poisson-Gaussian modelling are a particularly strong recent fit for unpaired, single-channel blind denoising.

“State of the art” is used narrowly here: among recent peer-reviewed methods with publicly inspectable implementations, it is one of the best matches for a single monochrome image with unknown Poisson-Gaussian noise. This repository contains a clean-room industrial-CT adaptation, not the authors' official code, and makes no universal best-on-every-dataset claim.

## End-to-end order

```text
Raw 16-bit image + generated candidate
  → clipped low-frequency prior only
  → 4×4 sub-lattice blind training + adaptive re-visible likelihood
  → x/y edge, transverse-width, endpoint and multiscale shape losses
  → strict 16-phase blind inference and variance prediction
  → Poisson-Gaussian data projection + raw-coordinate ridge anchors
  → strongest projection that passes every geometry guardrail
  → original edge/DoG enhancement, MAD clipping and feathering
  → endpoint warp only if the already enhanced result needs it
  → original structure-aware fine-noise and haze cleanup
  → independent 100-lamella audit
```

## Safe generative prior

The generated candidate is resized and percentile-matched. Both raw and generated images are low-pass filtered with `σ=6`, and prior changes are retained only in raw low-gradient regions:

\[
p=x+g_{low}\,\operatorname{clip}(G_6(y)-G_6(x),-c_p,c_p)
\]

with `c_p=max(1.25σ_n,4/65535)`. This prior appears only in a low-frequency loss. It supplies no ridge count, width, center, endpoint, or output pixels.

## Adaptive re-visible blind training

Each iteration samples 96 × 96 patches and hides one phase of a 4 × 4 sub-lattice. Hidden samples are interpolated from four neighbours. The U-Net predicts signal mean and variance. Given blind predictions `(m_b,v_b)`, stop-gradient visible predictions `(m_v,v_v)`, and a visible weight `β` increasing from 3 to 11:

\[
m=\frac{m_b+\beta\operatorname{sg}(m_v)}{1+\beta},\quad
v=\frac{v_b+\beta^2\operatorname{sg}(v_v)}{(1+\beta)^2}
\]

Learned Poisson-Gaussian variance is:

\[
v_n=\sigma_g^2+s_p\max(m,10^{-4})
\]

and the principal likelihood is:

\[
\mathcal L_{nll}=\frac{(x-m)^2}{v+v_n}+\log(v+v_n)
\]

A masked Charbonnier term is added at blind pixels.

## Geometry inside training

Geometry is not merely repaired after inference. The blind training prediction receives:

\[
\mathcal L_g=0.34L_{edge-x}+0.26L_{edge-y}+0.20L_{width}+0.12L_{endpoint}+0.08L_{shape}
\]

The transverse profile term constrains measurable width, the longitudinal profile term constrains endpoints, and the multiscale term constrains overall shape. The total objective is:

\[
\mathcal L=\mathcal L_{nll}+0.35L_{blind}+7.5L_g+0.04L_{prior}
\]

## Strict blind inference and coordinate anchors

Inference enumerates all 16 sub-lattice phases. Every interior output uses only a prediction made while that center pixel was hidden. Predicted mean and variance are then combined with the observation under the learned Poisson-Gaussian model.

A raw-gradient gate protects strong edges. Raw ridge centers detected in the two measurement profiles are also anchored during data projection. The anchors constrain coordinates; generated pixels and post-hoc raw patches are never written into the output.

Projection residual strengths `0, 0.65, 0.80, 1.00, 1.20, 1.50` are evaluated. Each candidate is re-measured across 100 lamellae. The strongest roughness reduction passing layer-count, center, FWHM, highlight, SSIM, endpoint and length limits is selected. This image selects `1.50`.

## Original enhancement and cleanup remain active

The existing transverse edge gate, DoG structure enhancement, MAD change cap, 18-pixel feathering, optional enhanced-pixel endpoint warp, and `σ=0.90/2.20` structure-aware cleanup still run after blind denoising.

Because the new blind baseline is smoother, the unchanged enhancement operator searches a wider strength range up to 2.0, while retaining its original MAD-derived pixel-change cap. The selected parameters are `edge=2.0`, `structure=0.06`, `fine=0.12`, and `haze=0.015`. The enhanced result already passes geometry, so a redundant second endpoint resampling is skipped.

## Measured comparison

| Metric | v4 | v5 |
|---|---:|---:|
| High-frequency roughness reduction | 26.28% | **28.66%** |
| Digital edge-acutance gain | 4.89% | **5.60%** |
| Local-contrast gain | **2.72%** | 2.04% |
| Maximum median-FWHM change | **0.188%** | 1.218% |
| Endpoint-shift P95 | 0.260 px | **0.152 px** |
| Maximum reliable endpoint shift | 0.695 px | **0.486 px** |
| Integrated length-delta P95 | 0.433 px | **0.259 px** |
| Independent length-delta P95 | 0.454 px | **0.287 px** |

Version 5 improves denoising, edge acutance, and tail endpoint/length consistency. Its FWHM change is higher than v4 but remains well inside the 4% guardrail. Local contrast is slightly lower, an explicit trade-off for stronger denoising.

## Reproduction

```bash
docker compose run --rm ct-sota
```

or:

```bash
python app/run_sota_pipeline.py \
  --source input/source_16bit.tif \
  --generative-prior input/generative_candidate_visual_only.png \
  --outdir results_sota --iterations 600 --device cpu
```

Tests:

```bash
python -m unittest discover -s tests -v
```

## Limitations

This is single-image internal validation without paired clean truth, repeated scans, or cross-device data. Pixel-coordinate consistency is not millimetre/micrometre certification. Absolute measurement requires calibrated pixel size, a traceable standard, PSF/MTF characterization, and repeatability studies. The two low-confidence raw endpoint references require manual review.

Primary comparison sources: [Blind2Sound official repository](https://github.com/Jiazheng-Liu/Blind2Sound), [TBSN official repository](https://github.com/nagejacob/TBSN), and [BIR-D at NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/25869dbf7682272357bc2cbbf860e1c8-Abstract-Conference.html).
