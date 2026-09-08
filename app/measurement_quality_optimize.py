#!/usr/bin/env python3
"""v16 measurement-quality refinement with v15 structural guardrails.

The full-resolution blind-denoised guide is the only pixel source. Candidate
changes are zero-phase, capped residual reductions. A candidate is released
only if lamella, interlayer, raw-nonregression, multiscale-detail, and SSIM
checks remain within strict limits relative to the unmodified v15 baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
from tifffile import imwrite

import boundary_cleanup as boundary
import generative_shape_constraint as shape
import generative_shape_project as project
import pipeline as base
import quality_optimize as quality
import residual_denoise as residual


def metric_summary(audit: dict) -> dict:
    boundary_geometry = audit["constraint_coordinate_boundary_geometry"]
    layer = audit["layer_geometry_comparison"]
    gap = audit["interlayer_geometry_comparison"]
    detail = audit["structure_detail_consistency"]["hard_projected"]
    return {
        "endpoint_shift_abs_p95_px": boundary_geometry["endpoint_shift_abs_p95_px"],
        "length_delta_abs_p95_px": boundary_geometry["length_delta_abs_p95_px"],
        "lamella_width_relative_error_p95": layer["guide_width_relative_error_p95"],
        "lamella_dual_evidence_pass_count": layer["dual_evidence_pass_count"],
        "interlayer_width_relative_error_p95": gap["guide_width_relative_error_p95"],
        "interlayer_length_abs_error_p95_px": gap["guide_length_abs_error_p95_px"],
        "interlayer_dual_evidence_pass_count": gap["dual_evidence_pass_count"],
        "edge_clarity": audit["edge_clarity"]["output_median"],
        "low_frequency_correlation": detail["low_frequency_correlation"],
        "mid_frequency_correlation": detail["mid_frequency_correlation"],
        "gradient_magnitude_correlation": detail["gradient_magnitude_correlation"],
        "lamella_axial_detail_correlation_median": (
            detail["lamella_axial_detail_correlation_median"]
        ),
        "lamella_axial_detail_correlation_p10": detail["lamella_axial_detail_correlation_p10"],
    }


def strict_guardrail(summary: dict, baseline: dict, target_ssim: float) -> tuple[bool, dict]:
    checks = {
        "endpoint_p95_le_0_10_px": summary["endpoint_shift_abs_p95_px"] <= 0.10,
        "length_p95_le_0_15_px": summary["length_delta_abs_p95_px"] <= 0.15,
        "lamella_width_p95_le_1_5_percent": (
            summary["lamella_width_relative_error_p95"] <= 0.015
        ),
        "lamella_dual_evidence_nonregression": (
            summary["lamella_dual_evidence_pass_count"]
            >= baseline["lamella_dual_evidence_pass_count"]
        ),
        "interlayer_width_p95_le_2_percent": (
            summary["interlayer_width_relative_error_p95"] <= 0.02
        ),
        "interlayer_length_p95_le_0_15_px": (
            summary["interlayer_length_abs_error_p95_px"] <= 0.15
        ),
        "interlayer_dual_evidence_nonregression": (
            summary["interlayer_dual_evidence_pass_count"]
            >= baseline["interlayer_dual_evidence_pass_count"]
        ),
        "edge_clarity_retention_ge_99_5_percent": (
            summary["edge_clarity"] >= 0.995 * baseline["edge_clarity"]
        ),
        "low_frequency_correlation_ge_0_9999": (
            summary["low_frequency_correlation"] >= 0.9999
        ),
        "mid_frequency_correlation_ge_0_999": (
            summary["mid_frequency_correlation"] >= 0.999
        ),
        "gradient_correlation_ge_0_9985": (
            summary["gradient_magnitude_correlation"] >= 0.9985
        ),
        "axial_detail_median_ge_0_999": (
            summary["lamella_axial_detail_correlation_median"] >= 0.999
        ),
        "axial_detail_p10_ge_0_9975": (
            summary["lamella_axial_detail_correlation_p10"] >= 0.9975
        ),
        "ssim_ge_0_9995": target_ssim >= 0.9995,
    }
    return bool(all(checks.values())), checks


def save_comparison(path: Path, baseline: np.ndarray, enhanced: np.ndarray) -> None:
    roi = base.Roi(600, 1240, 300, 1900).clamp(baseline.shape)
    before = baseline[roi.slices()]
    after = enhanced[roi.slices()]
    delta = np.abs(after - before)
    lo, hi = (float(value) for value in np.percentile(before, (0.5, 99.7)))
    panels = []
    for values, title in (
        (before, "v15 structure carrier"),
        (after, "v16 quality-refined measurement"),
        (delta, "absolute change (auto-scaled)"),
    ):
        if title.startswith("absolute"):
            scale = max(float(np.percentile(values, 99.7)), 1.0 / 65535.0)
            mapped = np.clip(values / scale, 0.0, 1.0)
        else:
            mapped = np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        panel = Image.fromarray(np.rint(mapped * 255).astype(np.uint8), mode="L")
        panel = panel.resize((760, 304), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (760, 330), 0)
        canvas.paste(panel, (0, 26))
        ImageDraw.Draw(canvas).text((8, 7), title, fill=255, font=ImageFont.load_default())
        panels.append(canvas)
    output = Image.new("L", (2280, 330), 0)
    for index, panel in enumerate(panels):
        output.paste(panel, (760 * index, 0))
    output.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-roi", type=base.parse_roi, default=base.Roi(600, 1240, 300, 1900))
    parser.add_argument("--central-roi", type=base.parse_roi, default=base.Roi(700, 1140, 985, 1205))
    parser.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    parser.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    parser.add_argument("--left-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    parser.add_argument("--right-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    parser.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    parser.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    current, current_info = base.load_gray(args.input)
    if source.shape != current.shape:
        raise ValueError(f"shape mismatch: source={source.shape}, input={current.shape}")

    target = args.target_roi.clamp(source.shape)
    central_roi = args.central_roi.clamp(source.shape)
    left = args.left_roi.clamp(source.shape)
    right = args.right_roi.clamp(source.shape)
    references = quality.build_boundary_references(
        source,
        current,
        args.left_body_roi.clamp(source.shape),
        args.right_body_roi.clamp(source.shape),
        args.top_range,
        args.bottom_range,
    )
    masks, mask_info = boundary.build_endpoint_envelope_masks(
        source.shape, references, target
    )
    components, component_method = residual.prepare_components(
        current, references, target, central_roi, masks
    )
    sigma_n = component_method["estimated_noise_sigma_normalized"]
    layers = shape.measure_layers(
        source, left, right, args.top_range, args.bottom_range, guide=current
    )
    baseline_clean = boundary.boundary_cleanliness_metrics(current, target, masks)
    baseline_center = residual.central_noise_rms(current, central_roi)

    parameter_grid = (
        (0.00, 0.00),
        # The lamella carrier remains immutable. Stronger lamella residual
        # shrinkage was evaluated and rejected because several marginal thin
        # layers crossed the per-layer dual-evidence threshold.
        (0.00, 0.40),
        (0.00, 0.55),
        (0.00, 0.70),
        (0.00, 0.82),
        (0.00, 0.92),
    )
    candidates = []
    selected = None
    baseline_summary = None
    for lamella_strength, central_strength in parameter_grid:
        candidate, operation = residual.apply_candidate(
            current,
            target,
            components,
            sigma_n,
            lamella_strength,
            central_strength,
        )
        audit = project.audit_projection(
            source,
            current,
            current,
            candidate,
            layers,
            left,
            right,
            args.top_range,
            args.bottom_range,
        )
        summary = metric_summary(audit)
        if baseline_summary is None:
            baseline_summary = summary
        clean = boundary.boundary_cleanliness_metrics(candidate, target, masks)
        lamella_reduction = 1.0 - clean["interior_axial_noise_rms"] / max(
            baseline_clean["interior_axial_noise_rms"], 1e-8
        )
        central_reduction = 1.0 - residual.central_noise_rms(
            candidate, central_roi
        ) / max(baseline_center, 1e-8)
        target_ssim = float(structural_similarity(
            current[target.slices()], candidate[target.slices()], data_range=1.0
        ))
        passed, checks = strict_guardrail(summary, baseline_summary, target_ssim)
        score = 1.0 * lamella_reduction + 0.80 * central_reduction
        item = {
            **operation,
            "lamella_axial_noise_reduction": lamella_reduction,
            "central_noise_reduction": central_reduction,
            "ssim_vs_v15": target_ssim,
            "structure_geometry": summary,
            "guardrail_checks": checks,
            "guardrail_pass": passed,
            "score": score,
        }
        candidates.append(item)
        if passed and (selected is None or score > selected["summary"]["score"]):
            selected = {"summary": item, "image": candidate, "audit": audit}

    if selected is None:
        raise RuntimeError("No v16 measurement-quality candidate passed all guardrails")

    final = selected["image"]
    audit = selected["audit"]
    condition_rows = audit.pop("condition_boundary_rows")
    audit.pop("raw_boundary_rows")
    layer_rows = audit.pop("layer_comparison_rows")
    gap_rows = audit.pop("gap_comparison_rows")
    detail_rows = audit.pop("structure_detail_rows")

    imwrite(
        args.outdir / "MEASUREMENT_v16_quality_enhanced_16bit.tif",
        base.to_uint16(final),
        photometric="minisblack",
        description=(
            "MEASUREMENT_ASSIST v16: guide-only capped zero-phase residual denoising; "
            "no generated pixels or spatial warp."
        ),
    )
    base.save_preview(args.outdir / "MEASUREMENT_v16_quality_enhanced.png", final)
    quality.save_boundary_overlay(
        args.outdir / "MEASUREMENT_v16_boundary_overlay.png",
        final,
        condition_rows,
        target,
    )
    residual.save_mask_audit(
        args.outdir / "AUDIT_v16_protection_masks.png", current, target, components
    )
    save_comparison(args.outdir / "MEASUREMENT_v16_comparison.png", current, final)
    shape.write_csv(args.outdir / "lamella_v16_comparison.csv", layer_rows)
    shape.write_csv(args.outdir / "interlayer_v16_comparison.csv", gap_rows)
    shape.write_csv(args.outdir / "structure_detail_v16.csv", detail_rows)

    payload = {
        "completed": True,
        "release": "v16-measurement-quality-refinement",
        "source": source_info,
        "input": current_info,
        "pixel_source": "v15 blind-denoised structure carrier only",
        "generated_pixel_weight": 0.0,
        "spatial_transform": None,
        "method": component_method,
        "protection_masks": mask_info,
        "selection_rule": (
            "maximum lamella plus central noise reduction among candidates passing "
            "strict v15 lamella/interlayer/detail/SSIM guardrails"
        ),
        "baseline": baseline_summary,
        "selected": selected["summary"],
        "audit": audit,
        "candidate_grid": candidates,
        "measurement_warning": (
            "Measurement-assist output; physical units require calibrated pixel size and "
            "scanner PSF/MTF validation."
        ),
    }
    (args.outdir / "measurement_quality_v16_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completed": True,
        "output": str(args.outdir / "MEASUREMENT_v16_quality_enhanced_16bit.tif"),
        "selected": selected["summary"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
