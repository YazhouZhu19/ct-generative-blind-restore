# Single-Image Self-Supervised Blind Denoising and Fidelity-Preserving Enhancement for Measurable Lamellae

## Technical Summary

This algorithm targets dense, sheet-like lamellae in a single 16-bit industrial CT/X-ray image. It addresses a central conflict: denoising and sharpening can improve visual clarity while changing the number, position, or thickness of the structures that must be measured. Instead of feeding a generic generative restoration directly into the measurement path, the system establishes two physically separated pipelines. The generative model supplies only a visual target, while the measurement pipeline starts from the original 16-bit data and applies adjacent-pixel replacement self-supervision, masked-only ensemble prediction, noise-scale-constrained data consistency, directional continuity enhancement, and measurement guardrails.

In the current 1600 × 2200 single-image experiment, the quality-optimized result reduced flat-region high-frequency roughness by 23.03% relative to the raw image, increased digital edge acutance by 5.87%, and increased local contrast by 3.00%. ROI SSIM against the blind-denoised baseline was 0.99857, the maximum lamella-center displacement was 1 px, and the largest median FWHM change was 2.62%. The optional length module detected 100 lamellae, of which 99 passed its internal pixel-domain quality checks.

These results establish an auditable clarity–geometry trade-off for this image under the implemented guardrails. They do not establish millimetre-scale absolute accuracy, recovery of unobserved physical detail, or equivalent performance across other scanners, exposure conditions, and specimens.

## Innovation Positioning

The defensible contribution is **system-level integration under industrial measurement constraints**. It is not a claim that adjacent replacement, blind-spot learning, unsharp masking, or subpixel fitting were invented here.

### Innovation 1: Hard Separation of Generative Visualization and Measurement Data

Generic generative models can produce complete, continuous, regular-looking lamellae, but they may also add, remove, duplicate, or straighten structures. This system confines generated content to the `GENERATIVE_visual_only_*` output domain. No generated pixel is permitted to enter a `MEASUREMENT_*` output.

The innovation is architectural: the desired appearance and the measurement evidence are separated by naming rules, file flow, and container entry points. This reduces the risk that a visually plausible hallucination is accidentally used for thickness measurement or defect acceptance.

### Innovation 2: Adjacent-Pixel Replacement Adapted to 16-Bit Single-Image Supervision

During training, approximately 12% of center pixels are randomly selected and replaced by one of eight neighboring offsets. The Charbonnier loss is evaluated only at masked positions:

`x_masked(i) = x(i + delta_i)`

`L = mean_M sqrt((f_theta(x_masked)_i - x_i)^2 + epsilon^2)`

This removes the need for paired clean targets and prevents direct copying of center-pixel noise at supervised positions. Unlike directly applying natural-image pretrained weights, the model adapts to the current image's lamellar period, 16-bit intensity range, and noise statistics.

### Innovation 3: Masked-Only Prediction Ensembles with an Uncertainty Output

Inference performs eight random masking passes. A pixel prediction contributes to the ensemble mean only when that pixel was hidden in the corresponding pass. The standard deviation across masked-context predictions is also retained:

`mu_i = mean_k(prediction_i^k | i is masked)`

`u_i = std_k(prediction_i^k | i is masked)`

This avoids reintroducing center-pixel leakage at inference and produces a pixelwise uncertainty map. The value `u_i` is not a statistically calibrated confidence interval, but it exposes locations where contextual predictions disagree and therefore supports targeted review.

### Innovation 4: Noise-Scale-Limited Data Consistency

The network prediction never directly replaces the observation. The noise scale is first estimated from the median absolute deviation of a high-pass residual:

`sigma_n = median(abs(h - median(h))) / 0.6745`

The total per-pixel modification is then capped at `±2.75 sigma_n`, while a gradient gate reduces corrections near strong edges:

`z = x + alpha * edge_gate * clip(mu - x, -c, c)`

This predict-then-limit design turns the neural network into a candidate generator constrained by the measured data, rather than the final authority. It lowers the risk of over-smoothing or reconstructing thickness boundaries.

### Innovation 5: Direction-Aware Continuity Enhancement Without Lateral Layer Mixing

The dominant lamellae are approximately vertical. Continuity enhancement therefore uses an anisotropic Gaussian kernel with `sigma=(3,0)`, aggregating information only along the lamellar extension direction. A horizontal Sobel gradient protects the left and right thickness boundaries.

In the current experiment, continuity variation decreased by 30.60% on the left and 27.72% on the right, while dropout decreased by 53.33% and 76.00%, respectively. The corresponding FWHM changes were only 0.049% and 0.358%, with a maximum layer-center displacement of 0 px.

### Innovation 6: Enhancement as Geometry-Guarded Parameter Optimization

The quality stage does not manually select the visually sharpest image. It searches an edge-gain and structure-gain grid. A candidate is eligible for the clarity score only if all of the following guardrails pass:

- Change in lamella count is no greater than 1 on either side.
- Center displacement is no greater than 1 px.
- Relative median FWHM change is no greater than 5%.
- Flat-region high-frequency roughness increases by no more than 3% versus the blind-denoised baseline.
- ROI SSIM is at least 0.985.

The selected candidate for this image uses `edge_amount=0.32` and `structure_gain=0.06`.

### Innovation 7: Separation of Recommended Measurements and Enhanced-Image Diagnostics

The optional length module uses the enhanced image to improve endpoint visibility only. Recommended lengths are still derived from curved center paths and subpixel endpoint fits anchored to the original 16-bit image. Original-data weighting is reintroduced in endpoint regions so that the continuity operation cannot silently move the upper or lower endpoints.

This avoids the closed-loop bias of using an enhanced image to validate itself. Of the 100 detected lamellae, 99 passed. One lamella with an internal uncertainty of 3.411 px was explicitly marked for manual review instead of being hidden by an aggregate score.

### Innovation 8: Evidence-Producing Output Rather Than a Single Attractive Image

Each run produces:

- A full-resolution uint16 candidate image.
- A float32 residual image.
- A float32 uncertainty map.
- Lamella counts, locations, FWHM, continuity, dropout, and SSIM metrics.
- All candidate parameters and guardrail outcomes.
- Reproducible weights, random seeds, and Docker configuration.

The core output is therefore an evidence bundle that answers what changed, by how much, and whether the change crossed a measurement boundary—not merely an enhanced image.

## Complete Processing Workflow

### Stage A: Input and Region Definition

The raw uint16 TIFF is normalized by 65535 into `[0,1]`; the reference JPEG is never used to recalibrate intensity. Training and enhancement are restricted to the specimen ROI, observations outside the ROI remain unchanged, and the boundary is blended with an 18 px feather.

### Stage B: Single-Image Self-Supervised Blind Denoising

The lightweight `AprN2SLite` network consists of five 3 × 3 convolution layers with 12 intermediate channels. It is trained for 600 iterations using 96 × 96 patches, a batch size of 4, AdamW, and cosine learning-rate annealing.

This is an industrial single-image adaptation that combines the adjacent-replacement idea from APR-RD with Noise2Self-style masked supervision. It is not a complete reproduction of the official APR-RD method.

### Stage C: Fidelity-Preserving Fusion and Continuity Enhancement

After eight masked inference passes, the blind-spot estimate is written back through the MAD noise cap and a two-dimensional gradient gate. Continuity enhancement then operates only along the lamella-length direction. The total modification is capped again by the estimated noise scale.

### Stage D: Deterministic Quality Optimization

Using the blind-denoised result as the baseline, the system extracts horizontal detail and a difference-of-Gaussian structure component. Twenty parameter candidates are screened by geometry and noise guardrails, then ranked by a combined edge-acutance, local-contrast, and noise-penalty score.

### Stage E: Optional Length Measurement

The length module estimates lamellar spacing, detects layer centers, tracks curved center paths from the body toward the upper and lower endpoints, and performs three-point quadratic subpixel fitting on endpoint gradients. Recommended values remain anchored to the raw image; redetection in the enhanced image is a stability diagnostic only.

## Experimental Results and Interpretation

### Blind Denoising Preserved Lamellar Geometry

| Metric | Left ROI | Right ROI |
|---|---:|---:|
| Detected lamellae | 25 → 25 | 30 → 30 |
| Maximum center displacement | 0 px | 0 px |
| Relative FWHM change | 0.049% | 0.358% |
| Continuity-CV reduction | 30.60% | 27.72% |
| Dropout reduction | 53.33% | 76.00% |

The joint measurement-ROI SSIM was 0.97338. Because this stage intentionally changes noise texture, its SSIM is lower than the SSIM between the later, weaker quality enhancement and its baseline. The value must be interpreted together with lamella count, displacement, FWHM, and residual images.

### Quality Enhancement Improved Digital Clarity Within the Guardrails

| Metric | Change versus raw image |
|---|---:|
| Flat-region high-frequency roughness | -23.03% |
| Digital edge acutance | +5.87% |
| Local contrast | +3.00% |

Relative to the blind-denoised baseline, the quality result achieved ROI SSIM 0.99857, maximum center displacement of 1 px, maximum FWHM change of 2.62%, and a 2.08% increase in flat-region high-frequency roughness—still within the 3% guardrail.

The increase in digital edge acutance is an image-gradient result. It does not imply that the physical spatial resolution of the scanner improved.

### The Length Extension Is Internally Consistent but Lacks an Absolute Scale

The length module detected 100 lamellae, with 99 passing internal checks. Median length uncertainty was 0.369 px and the 95th percentile was 0.991 px. The median absolute difference between raw-anchored and enhanced-image length estimates was 0.016 px, with a 95th percentile of 0.231 px.

The TIFF contains no XResolution, YResolution, or ResolutionUnit metadata, and no system PSF/MTF or reference-standard calibration is available. These results therefore describe pixel-domain repeatability only; they cannot be converted into certified millimetre or micrometre accuracy.

## Relationship to Existing Methods

### Compared with Conventional Filtering

Gaussian or non-local-means denoising typically uses a global denoising strength. This method learns contextual statistics from the current image, caps the learned correction at the observed noise scale, and lets lamellar geometry—not visual sharpness alone—determine whether a candidate is acceptable.

### Compared with Generic Blind-Spot Networks

Generic blind-spot methods primarily optimize PSNR or SSIM. This system adds 16-bit data preservation, adjacent replacement for correlated noise, masked-only ensembles, pixelwise uncertainty, direction-aware continuity, and FWHM/center-displacement guardrails.

### Compared with Generative Restoration Models

General diffusion or restoration foundation models optimize perceptual quality. Here, their output cannot enter the measurement pipeline. The system intentionally gives up some visual perfection in exchange for traceability to the raw observation, residual, and measured structure change.

## Evidence Strength and Limitations

### Conclusions Supported by the Current Evidence

- The algorithm passes its internal structural guardrails on this 16-bit image and the current ROIs and thresholds.
- It reduces flat-region high-frequency roughness while increasing digital edge acutance and local contrast.
- The current automated analysis detects no change in the principal lamella counts, while center and FWHM changes remain bounded.
- The optional length module identifies and isolates at least one unstable lamella.

### Conclusions Not Supported by the Current Evidence

- Equivalent performance on other images, devices, specimens, or exposure settings has not been established.
- The method has not been shown to recover physical high frequencies that the acquisition system failed to capture.
- Internal SSIM and FWHM stability do not establish unbiased physical thickness.
- No superiority claim can be made over full APR-RD, Blind2Sound, FoundIR-v2, or other published methods.
- Absolute measurement accuracy in millimetres or micrometres cannot be reported.

### Validation Level

The recommended sharing status is: **shareable with explicit single-image internal-validation and missing-physical-calibration caveats**.

## Recommended Next Validation Steps

1. Acquire at least 20 repeat scans of the same location across exposure, fixture, and temporal variation.
2. Establish `mm/pixel` and thickness/length ground truth using a reference standard or microscopy.
3. Measure the system PSF/MTF to separate digital sharpening from physical resolving power.
4. Construct a blinded set with known layer counts, thicknesses, dropouts, fractures, and fusions.
5. Compare BM3D, Noise2Void/AP-BSN, APR-RD, Blind2Sound, and a domain-adapted diffusion model under the same geometry guardrails.
6. Ablate the noise-cap multiplier, directional smoothing strength, and guardrail thresholds.
7. Calibrate the uncertainty map against reviewed measurement error to define an interpretable rejection threshold.

## Further Research Questions

- Should adjacent-replacement offsets adapt to the measured noise correlation length?
- Can a local structure tensor replace the fixed vertical direction for curved or rotated lamellae?
- Is there a transferable calibration between multi-mask prediction variance and physical measurement error?
- When projection-domain or repeat-exposure data become available, should image-domain self-supervision be replaced by a physics-based forward constraint?
- Can the guardrail framework be extended to topological events such as missing, fused, or locally fractured lamellae?

## Code and Evidence

- Core blind denoising: `app/pipeline.py`
- Quality optimization: `app/quality_optimize.py`
- Length extension: `app/length_optimize.py`
- Blind-denoising metrics: `results/run_manifest.json`
- Quality metrics: `results_quality/quality_metrics.json`
- Length validation: `results_length/length_qa.json`

## Related Work

- APR-RD, AAAI 2025: <https://ojs.aaai.org/index.php/AAAI/article/view/32447>
- AP-BSN, CVPR 2022: <https://github.com/wooseoklee4/AP-BSN>
- Blind2Sound, ICCV 2025: <https://github.com/Jiazheng-Liu/Blind2Sound>
- FoundIR-v2, CVPR 2026: <https://github.com/cschenxiang/FoundIR-v2>
- Diffusion X-ray image denoising, MIDL 2024: <https://proceedings.mlr.press/v250/sanderson24a.html>
