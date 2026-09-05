#!/usr/bin/env python3
"""Build the English report artifact from the validated Chinese artifact.

Numeric datasets and block order are inherited from artifact.json so that both
language editions stay evidence-equivalent. Only reader-facing text and the
categorical labels used by the English chart and tables are translated here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifact.json"
OUTPUT = ROOT / "artifact_en.json"


def by_id(items: list[dict], item_id: str) -> dict:
    return next(item for item in items if item["id"] == item_id)


def update_sources(sources: list[dict]) -> None:
    labels = {
        "headline_metrics_sql": "Headline metrics SQLite query",
        "blind_before_after_sql": "Blind-denoising before/after SQLite query",
        "innovation_matrix_sql": "Innovation matrix SQLite query",
        "validation_summary_sql": "Validation summary SQLite query",
        "blind_manifest": "Blind-denoising run manifest",
        "quality_metrics": "Quality-optimization metrics",
        "length_qa": "Length-extension quality checks",
        "code_bundle": "Algorithm implementation",
        "apr_rd": "APR-RD, AAAI 2025",
        "foundir_v2": "FoundIR-v2 official code",
    }
    descriptions = {
        "headline_metrics_sql": "Reads report headline metrics materialized after cross-checking the three saved result JSON files.",
        "blind_before_after_sql": "Reads left/right lamellar continuity and dropout before and after blind denoising.",
        "innovation_matrix_sql": "Reads the innovation-positioning matrix audited against the code and saved results.",
        "validation_summary_sql": "Reads key guardrail results from blind denoising, quality optimization, and length extension.",
    }
    sql_paths = {
        "headline_metrics_sql": "queries_en/headline_metrics.sql",
        "blind_before_after_sql": "queries_en/blind_before_after.sql",
        "innovation_matrix_sql": "queries_en/innovation_matrix.sql",
        "validation_summary_sql": "queries_en/validation_summary.sql",
    }
    sql_text = {
        "headline_metrics_sql": "SELECT noise_reduction, edge_gain, contrast_gain, max_fwhm_drift, quality_ssim, length_pass_fraction, image_count, image_width, image_height, bit_depth FROM headline_metrics LIMIT 1;",
        "blind_before_after_sql": "SELECT metric, stage, value, side, improvement, fwhm_drift FROM blind_before_after ORDER BY CASE metric WHEN 'Left continuity CV' THEN 1 WHEN 'Right continuity CV' THEN 2 WHEN 'Left dropout rate' THEN 3 WHEN 'Right dropout rate' THEN 4 ELSE 5 END, CASE stage WHEN 'Before' THEN 1 ELSE 2 END;",
        "innovation_matrix_sql": "SELECT \"order\", innovation, prior_basis, local_contribution, evidence, boundary FROM innovation_matrix ORDER BY \"order\" ASC;",
        "validation_summary_sql": "SELECT \"order\", metric, observed, threshold, status, interpretation FROM validation_summary ORDER BY \"order\" ASC;",
    }

    for source in sources:
        source["label"] = labels[source["id"]]
        if source["id"] in sql_paths:
            source["path"] = sql_paths[source["id"]]
            source["query"]["sql"] = sql_text[source["id"]]
            source["query"]["description"] = descriptions[source["id"]]


def main() -> None:
    artifact = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    manifest = artifact["manifest"]
    manifest["title"] = "Single-Image Self-Supervised Blind Denoising and Fidelity-Preserving Enhancement for Measurable Lamellae"
    manifest["description"] = "A technical report focused on engineering innovation, validation evidence, and metrology boundaries."

    card_text = {
        "noise_reduction_card": ("Flat-region high-frequency roughness reduction in the quality result versus the raw image.", "High-frequency roughness reduction"),
        "edge_gain_card": ("Mean digital edge-acutance improvement across the left and right measurement ROIs versus the raw image.", "Digital edge acutance"),
        "contrast_gain_card": ("Mean local-contrast improvement across the left and right measurement ROIs versus the raw image.", "Local contrast"),
        "geometry_card": ("Worst relative median FWHM change in quality optimization versus the blind-denoised baseline.", "Maximum FWHM drift"),
        "length_pass_card": ("Fraction of detected lamellae passing the optional length module's internal pixel-domain checks.", "Internal length pass rate"),
    }
    for card in manifest["cards"]:
        card["description"], card["metrics"][0]["label"] = card_text[card["id"]]

    chart = by_id(manifest["charts"], "blind_before_after_chart")
    chart.update(
        title="Lamellar continuity and dropout before and after blind denoising",
        subtitle="Left and right ROIs; lower is better for all four metrics",
        question="Does self-supervised blind denoising reduce both continuity variation and dropout in the left and right lamellar regions?",
        rationale="Each of four discrete metrics has a directly comparable before/after value, so grouped bars expose the reduction without implying a time trend.",
    )
    chart["encodings"] = {
        "x": {"field": "metric", "type": "nominal", "label": "Metric"},
        "y": {"field": "value", "type": "quantitative", "label": "Ratio"},
        "color": {"field": "stage", "type": "nominal", "label": "Stage"},
        "tooltip": [
            {"field": "side", "type": "nominal", "label": "ROI"},
            {"field": "improvement", "type": "quantitative", "label": "Reduction", "format": "percent"},
            {"field": "fwhm_drift", "type": "quantitative", "label": "FWHM drift", "format": "percent"},
        ],
    }

    innovation_table = by_id(manifest["tables"], "innovation_table")
    innovation_table["title"] = "Innovation, prior basis, and repository-specific constraints"
    innovation_table["subtitle"] = "Positioned as auditable system integration for industrial measurement"
    innovation_table["columns"] = [
        {"field": "order", "label": "No.", "type": "number"},
        {"field": "innovation", "label": "Innovation", "type": "text"},
        {"field": "prior_basis", "label": "Prior basis", "type": "text"},
        {"field": "local_contribution", "label": "Repository contribution", "type": "text"},
        {"field": "evidence", "label": "Current evidence", "type": "text"},
        {"field": "boundary", "label": "Claim boundary", "type": "text"},
    ]

    validation_table = by_id(manifest["tables"], "validation_table")
    validation_table["title"] = "Structural and quality guardrail results"
    validation_table["subtitle"] = "Final selected candidates recorded in saved JSON; thresholds match the executed code"
    validation_table["columns"] = [
        {"field": "order", "label": "No.", "type": "number"},
        {"field": "metric", "label": "Guardrail / metric", "type": "text"},
        {"field": "observed", "label": "Observed", "type": "text"},
        {"field": "threshold", "label": "Threshold", "type": "text"},
        {"field": "status", "label": "Status", "type": "text"},
        {"field": "interpretation", "label": "Interpretation", "type": "text"},
    ]

    blocks = {block["id"]: block for block in manifest["blocks"]}
    blocks["title"]["body"] = "# Single-Image Self-Supervised Blind Denoising and Fidelity-Preserving Enhancement for Measurable Lamellae"
    blocks["technical_summary"]["body"] = """## Technical Summary

This algorithm addresses the central conflict in a single 16-bit industrial CT image: denoising and sharpening can alter the geometry of the lamellae being measured. Its main contribution is not the direct use of a generic generative model for measurement. Instead, generative visualization is hard-separated from the measurement data, while single-image self-supervision, masked-only ensembles, noise-scale limits, directional continuity enhancement, and geometry guardrails form an auditable measurement-safe pipeline.

For this image, flat-region high-frequency roughness decreased by **23.03%**, digital edge acutance increased by **5.87%**, and local contrast increased by **3.00%** versus the raw image. Maximum FWHM drift was **2.62%**, ROI SSIM against the blind-denoised baseline was **0.99857**, and 99 of 100 lamellae passed the optional length module's internal checks.

**Evidence level: shareable with caveats.** These are single-image internal-validation results, not external ground-truth accuracy, improved physical scanner resolution, or cross-device generalization."""
    blocks["finding_heading"]["body"] = """## Continuity Improved Without Material Thickness Drift

Blind denoising reduced within-layer variation and dropout on both sides. All four plotted values are ratios for which lower is better. Their decreases occur while independent FWHM guardrails show width drift of only 0.049% on the left and 0.358% on the right. For this image, the evidence supports the interpretation that the method mainly suppressed discontinuous noise instead of collapsing sheet-like lamellae into sharp lines."""
    blocks["quality_finding"]["body"] = """## Automated Search Confined Clarity Gains to Measurement Guardrails

The quality stage does not choose the visually sharpest candidate. It first rejects any candidate that fails lamella-count, center-displacement, FWHM, flat-region noise, or SSIM limits, then optimizes edge and contrast among the survivors. This image selected `edge_amount=0.32` and `structure_gain=0.06`. Maximum center displacement was 1 px, maximum FWHM change was 2.62%, and flat-region roughness increased by 2.08% versus the blind-denoised baseline, all within their guardrails. Digital acutance gain does not imply better physical scanner resolution."""
    blocks["innovation_heading"]["body"] = """## The Defensible Innovation Is Measurement-Constrained System Integration

Adjacent replacement, blind-spot prediction, unsharp masking, and subpixel fitting all have established precedents. The defensible contribution in this repository is their integration into a closed loop that explicitly handles 16-bit single-image data, correlated noise, directional lamellae, and measurement risk—and produces rejection evidence on every run. The table separates inherited ideas, repository-specific contributions, current evidence, and claim boundaries so that engineering innovation is not misrepresented as unverified academic priority."""
    blocks["scope"]["body"] = """## Evidence Scope and Metric Definitions

The input is one 1600 × 2200 uint16 TIFF. Blind-denoising audit metrics are computed separately in the left and right lamellar ROIs: peak count, nearest-neighbor center displacement, median full width at half maximum (FWHM), continuity coefficient of variation (CV) along each lamella, and dropout rate. The quality stage evaluates high-frequency roughness, edge acutance, and local contrast against the raw image, while using the blind-denoised image as the geometry-guardrail baseline. The length module's statistical unit is one visible bright lamella, with 100 rows in total.

Continuity CV is the standard deviation of the row-wise intensity residual around a smooth trend divided by median intensity. Dropout is the fraction of rows below the local trend threshold. “Acutance” is an image-domain gradient metric. “Length pass rate” is the fraction satisfying the pixel-domain stability thresholds implemented in code."""
    blocks["method"]["body"] = """## Model and Processing Pipeline

### 1. Adjacent-replacement self-supervision
Training hides approximately 12% of center pixels, replaces each with one of eight neighboring offsets, and computes Charbonnier loss only at masked positions. A five-layer 3×3 convolutional network with 12 intermediate channels is trained for 600 iterations on the current image.

### 2. Masked-only ensemble inference
Inference performs eight passes with approximately 55% random masking. A prediction contributes to the pixel mean and standard deviation only when that pixel is hidden, avoiding center-value leakage and producing an uncertainty map.

### 3. Data consistency and directional prior
A high-pass MAD estimate sets the noise scale, and the total per-pixel modification is limited to ±2.75σ. A two-dimensional gradient gate protects strong edges. Continuity enhancement then acts only along the lamella-length direction, while a horizontal Sobel gradient protects thickness boundaries.

### 4. Guardrail-constrained quality optimization
Twenty edge/structure-gain candidates must pass lamella count, center displacement, FWHM, flat-region noise, and SSIM guardrails before clarity scoring.

### 5. Raw-data-anchored length extension
The optional module tracks curved center paths and fits subpixel endpoints. Recommended lengths come from the raw 16-bit path; redetection in the enhanced image is a diagnostic only."""
    blocks["validation_heading"]["body"] = """## Saved Results Pass Configured Guardrails but Do Not Constitute Metrology Certification

Key percentages were independently recomputed from the three saved JSON result files, and the stored guardrail flags for all three stages are true. The table focuses on structural quantities most vulnerable to enhancement artifacts. Automated guardrails can expose major topology or width damage; without external ground truth, they cannot exclude shared bias or blind spots in the detector itself."""
    blocks["limitations"]["body"] = """## Single-Image Evidence Limits Generalization of the Innovation Claim

- There is no paired clean target, repeat-scan study, or cross-device test; generalization has not been established.
- The TIFF lacks physical pixel dimensions, and no reference standard or system PSF/MTF is available; millimetre or micrometre accuracy cannot be reported.
- Multi-mask standard deviation has not been calibrated to physical measurement error and should currently be treated only as a review cue.
- The implementation borrows APR-RD's adjacent-replacement idea but is not a full reproduction, so it cannot support superiority claims over APR-RD, Blind2Sound, or FoundIR-v2.
- Parameters and ROIs assume the current lamellar orientation; substantial rotation, curvature, or a different specimen requires a new direction field and new guardrails."""
    blocks["next_steps"]["body"] = """## The Next Validation Round Should Replace Internal Guardrails with External Accuracy Evidence

1. Acquire at least 20 repeat scans of the same location across exposure, fixture, and temporal variation.
2. Establish thickness and length ground truth with a reference standard or microscopy and add `mm/pixel` calibration.
3. Measure PSF/MTF to distinguish digital sharpening from physical resolving power.
4. Build a blinded set containing missing layers, fusions, fractures, and multiple thicknesses.
5. Compare BM3D, AP-BSN/Noise2Void, APR-RD, Blind2Sound, and domain-adapted diffusion under one guardrail framework.
6. Ablate the noise-cap multiplier, directional strength, and guardrail thresholds, then calibrate the uncertainty rejection threshold."""
    blocks["questions"]["body"] = """## Open Research Questions

- Can adjacent-replacement distance adapt to the measured noise correlation length?
- Can a local structure tensor replace the fixed vertical direction for curved or rotated lamellae?
- Is there a transferable calibration between the uncertainty map and actual thickness or endpoint error?
- When projection-domain or repeat-exposure data become available, can a physics-based forward constraint further reduce bias?
- Can the guardrails cover topological events such as missing, fused, or fractured lamellae, rather than only peak count and FWHM?"""

    blind_rows = artifact["snapshot"]["datasets"]["blind_before_after"]
    metric_map = {
        "左侧连续性 CV": "Left continuity CV",
        "右侧连续性 CV": "Right continuity CV",
        "左侧断裂率": "Left dropout rate",
        "右侧断裂率": "Right dropout rate",
    }
    for row in blind_rows:
        row["metric"] = metric_map[row["metric"]]
        row["stage"] = "Before" if row["stage"] == "处理前" else "After"
        row["side"] = "Left" if row["side"] == "左侧" else "Right"

    artifact["snapshot"]["datasets"]["innovation_matrix"] = [
        {"order": 1, "innovation": "Hard separation of generative and measurement paths", "prior_basis": "Generative restoration and conventional measurement workflows", "local_contribution": "Naming, file flow, and container entry points prevent generated pixels from entering measurement outputs", "evidence": "Generated results are emitted only as VISUAL_ONLY", "boundary": "A safety-architecture contribution, not a new generative model"},
        {"order": 2, "innovation": "16-bit single-image adjacent-replacement supervision", "prior_basis": "APR-RD and Noise2Self/blind-spot networks", "local_contribution": "Adapts to the current intensity domain, correlated noise, and within-lamella statistics", "evidence": "600 single-image iterations; left/right layer counts unchanged", "boundary": "Not a complete APR-RD reproduction"},
        {"order": 3, "innovation": "Masked-only prediction ensemble", "prior_basis": "Masked self-supervision and ensemble estimation", "local_contribution": "Uses predictions only when each pixel is hidden and exports their pixelwise standard deviation", "evidence": "Eight masking passes; uncertainty TIFF saved", "boundary": "Standard deviation is not yet a calibrated confidence interval"},
        {"order": 4, "innovation": "Noise-scale cap with edge gating", "prior_basis": "Robust noise estimation and data consistency", "local_contribution": "Constrains network corrections by ±2.75σ and a gradient gate", "evidence": "Residual cap and structural guardrails are recorded", "boundary": "Cannot guarantee that every subtle defect is unaffected"},
        {"order": 5, "innovation": "Directional continuity enhancement", "prior_basis": "Anisotropic filtering", "local_contribution": "Smooths along the lamella direction while horizontal Sobel gradients protect thickness boundaries", "evidence": "Continuity improves about 28%–31%; FWHM drift remains below 0.36%", "boundary": "Fixed direction requires adaptation for rotated or curved images"},
        {"order": 6, "innovation": "Guardrail-constrained automatic parameter search", "prior_basis": "Image-enhancement parameter search", "local_contribution": "Candidates must pass topology, displacement, width, noise, and SSIM limits before clarity comparison", "evidence": "The final quality candidate passes every guardrail", "boundary": "Thresholds are not yet calibrated against external truth"},
        {"order": 7, "innovation": "Raw-data-anchored measurement", "prior_basis": "Center-path tracking and subpixel endpoints", "local_contribution": "Recommended values come from the raw image; enhanced-image redetection is diagnostic only", "evidence": "99/100 layers pass; the unstable layer is explicitly rejected", "boundary": "Only pixel-domain internal consistency is established"},
        {"order": 8, "innovation": "Evidence-producing outputs", "prior_basis": "MLOps and quality control", "local_contribution": "Exports 16-bit results, residuals, uncertainty, guardrails, parameters, and weights together", "evidence": "Three JSON manifests and reproducible containers", "boundary": "Auditability is not metrology certification"},
    ]

    artifact["snapshot"]["datasets"]["validation_summary"] = [
        {"order": 1, "metric": "Quality ROI SSIM", "observed": "0.99857", "threshold": ">= 0.985", "status": "Pass", "interpretation": "Quality optimization changes little relative to the blind-denoised baseline"},
        {"order": 2, "metric": "Maximum lamella-center displacement", "observed": "1 px", "threshold": "<= 1 px", "status": "Pass", "interpretation": "At the guardrail boundary; residuals should still be reviewed"},
        {"order": 3, "metric": "Maximum median FWHM change", "observed": "2.62%", "threshold": "<= 5%", "status": "Pass", "interpretation": "No material width reconstruction was detected"},
        {"order": 4, "metric": "Flat-region roughness versus blind baseline", "observed": "+2.08%", "threshold": "<= +3%", "status": "Pass", "interpretation": "Sharpening-related noise reinjection remains within the cap"},
        {"order": 5, "metric": "Maximum peak-center displacement after blind denoising", "observed": "0 px", "threshold": "<= 1 px", "status": "Pass", "interpretation": "Principal peak positions are preserved"},
        {"order": 6, "metric": "Length-module pass rate", "observed": "99%", "threshold": ">= 90%", "status": "Pass", "interpretation": "One lamella is explicitly marked for review"},
        {"order": 7, "metric": "95th percentile length uncertainty", "observed": "0.991 px", "threshold": "Per layer <= 3 px", "status": "Pass", "interpretation": "Pixel-domain repeatability is high, but physical calibration is absent"},
    ]

    update_sources(manifest["sources"])
    update_sources(artifact["sources"])
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
