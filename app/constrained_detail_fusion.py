#!/usr/bin/env python3
"""v18 boundary-constrained, detail-preserving fusion.

This stage combines v11's finite-width constraint localization and closed-loop
measurement principle with v17's bounded generative denoising.  It never copies
an analytic ribbon into the image.  Instead it applies symmetric, zero-phase
cleanup to existing v17 pixels inside boundary/endpoint/interlayer constraint
fields, then restores every layer that regresses to the exact v17 baseline.
Slowly varying per-row width trajectories are exported for downstream audit.
There is no registration, resampling, coordinate warp, or whole-layer redraw.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, median_filter
from skimage.metrics import structural_similarity
from tifffile import imwrite

import generative_shape_constraint as shape
import generative_shape_project as project
import measurement_quality_optimize as v16
import pipeline as base
import structure_conditioned_diffusion as v17


def measure_local_width_profiles(
    carrier: np.ndarray,
    layers: list[dict],
    variation_strength: float = 0.70,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Measure robust, slowly varying row-wise thickness on the carrier.

    Missing or outlying row measurements are interpolated, Hampel-like clipped
    around the existing 25-sample robust range, median filtered, and then
    zero-phase smoothed.  The profile median is normalized back to the audited
    per-lamella FWHM, so local variation cannot move the authoritative median.
    """
    variation_strength = float(np.clip(variation_strength, 0.0, 1.0))
    profiles: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    height = carrier.shape[0]
    for layer in layers:
        target = float(
            layer.get("width_median_px")
            or max(1.2, 0.15 * float(layer["pitch_px"]))
        )
        top = max(0, int(math.ceil(float(layer["top_y_px"]))))
        bottom = min(height - 1, int(math.floor(float(layer["bottom_y_px"]))))
        ys = np.arange(top, bottom + 1, dtype=np.int32)
        measured = np.asarray(
            [
                shape.transverse_width_at_row(
                    carrier, layer["path_x"], int(y), float(layer["pitch_px"])
                )
                for y in ys
            ],
            dtype=object,
        )
        valid_mask = np.asarray([value is not None for value in measured], dtype=bool)
        values = np.asarray(
            [float(value) if value is not None else np.nan for value in measured],
            dtype=np.float64,
        )
        valid_count = int(np.sum(valid_mask))
        if valid_count >= 2:
            valid_y = ys[valid_mask].astype(np.float64)
            valid_values = values[valid_mask]
            values = np.interp(ys.astype(np.float64), valid_y, valid_values)
            p10 = float(layer.get("width_p10_px") or np.percentile(valid_values, 10.0))
            p90 = float(layer.get("width_p90_px") or np.percentile(valid_values, 90.0))
            spread = max(p90 - p10, 0.08 * target)
            values = np.clip(
                values,
                max(0.65 * target, p10 - 0.20 * spread),
                min(1.35 * target, p90 + 0.20 * spread),
            )
            window = min(9, len(values) if len(values) % 2 else len(values) - 1)
            values = median_filter(values, size=max(1, window))
            values = gaussian_filter1d(values, sigma=min(6.0, max(1.0, len(values) / 30.0)))
        else:
            values = np.full(len(ys), target, dtype=np.float64)
        values = target + variation_strength * (values - float(np.median(values)))
        values *= target / max(float(np.median(values)), 1e-8)
        values = np.clip(values, 0.70 * target, 1.30 * target)
        profile = np.full(height, target, dtype=np.float32)
        profile[ys] = values.astype(np.float32)
        profiles[layer["layer_id"]] = profile
        for y, width in zip(ys, values):
            rows.append(
                {
                    "side": layer["side"],
                    "layer_id": layer["layer_id"],
                    "y_px": int(y),
                    "target_local_width_px": float(width),
                    "audited_median_width_px": target,
                }
            )
    return profiles, rows


def hybrid_guardrail(
    summary: dict,
    baseline: dict,
    target_ssim: float,
) -> tuple[bool, dict]:
    """Tighter geometry than v17 while retaining measurable carrier detail."""
    checks = {
        "endpoint_p95_le_0_10_px": summary["endpoint_shift_abs_p95_px"] <= 0.10,
        "length_p95_le_0_15_px": summary["length_delta_abs_p95_px"] <= 0.15,
        "lamella_width_p95_le_0_35_percent": (
            summary["lamella_width_relative_error_p95"] <= 0.0035
        ),
        "lamella_dual_evidence_nonregression": (
            summary["lamella_dual_evidence_pass_count"]
            >= baseline["lamella_dual_evidence_pass_count"]
        ),
        "interlayer_width_p95_le_0_40_percent": (
            summary["interlayer_width_relative_error_p95"] <= 0.0040
        ),
        "interlayer_length_p95_le_0_15_px": (
            summary["interlayer_length_abs_error_p95_px"] <= 0.15
        ),
        "interlayer_dual_evidence_nonregression": (
            summary["interlayer_dual_evidence_pass_count"]
            >= baseline["interlayer_dual_evidence_pass_count"]
        ),
        "edge_clarity_nonregression": summary["edge_clarity"] >= baseline["edge_clarity"],
        "low_frequency_correlation_ge_0_999": summary["low_frequency_correlation"] >= 0.999,
        "mid_frequency_correlation_ge_0_990": summary["mid_frequency_correlation"] >= 0.990,
        "gradient_correlation_ge_0_985": summary["gradient_magnitude_correlation"] >= 0.985,
        "axial_detail_median_ge_0_995": (
            summary["lamella_axial_detail_correlation_median"] >= 0.995
        ),
        "axial_detail_p10_ge_0_990": (
            summary["lamella_axial_detail_correlation_p10"] >= 0.990
        ),
        "ssim_ge_0_997": target_ssim >= 0.997,
    }
    return bool(all(checks.values())), checks


def symmetric_constraint_cleanup(
    generated: np.ndarray,
    target: base.Roi,
    maps: dict[str, np.ndarray],
    *,
    boundary_gain: float,
    endpoint_gain: float,
    interlayer_denoise: float,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Enhance existing edges without importing an analytic intensity profile.

    The finite-width constraints are localization masks only.  Zero-phase
    symmetric residuals are applied around both sides of every slab and both
    endpoints, so no coordinate can be translated.  Interlayer cleanup is a
    capped residual shrinkage that does not flatten its low-frequency signal.
    """
    crop = generated[target.slices()]
    transverse_low = gaussian_filter(crop, sigma=(0.12, 0.72))
    longitudinal_low = gaussian_filter(crop, sigma=(0.72, 0.12))
    isotropic_low = gaussian_filter(crop, sigma=0.58)
    transverse_detail = crop - transverse_low
    longitudinal_detail = crop - longitudinal_low
    noise_residual = crop - isotropic_low

    boundary_weight = np.clip(maps["boundary"] * (0.35 + 0.65 * maps["confidence"]), 0.0, 1.0)
    endpoint_weight = np.clip(maps["endpoint"] * (0.35 + 0.65 * maps["confidence"]), 0.0, 1.0)
    gap_binary = maps["interlayer"] > 0.55
    gap_distance = distance_transform_edt(gap_binary)
    gap_core = np.clip((gap_distance - 1.6) / 0.8, 0.0, 1.0)
    gap_weight = np.clip(
        gap_core * maps["interlayer"] * (1.0 - maps["protection"]), 0.0, 1.0
    )
    sigma_n = base.noise_sigma_mad(crop)
    cap = max(0.70 * sigma_n, 3.0 / 65535.0)
    delta = (
        float(boundary_gain) * boundary_weight * transverse_detail
        + float(endpoint_gain) * endpoint_weight * longitudinal_detail
        - float(interlayer_denoise) * gap_weight * noise_residual
    )
    delta = np.clip(delta, -cap, cap)
    cleaned_crop = np.clip(crop + delta, 0.0, 1.0).astype(np.float32)
    output = generated.copy()
    output[target.slices()] = cleaned_crop
    influence = np.clip(
        boundary_weight + endpoint_weight + gap_weight, 0.0, 1.0
    ).astype(np.float32)
    return output, {
        "method": "zero-phase symmetric constraint-field edge cleanup",
        "boundary_intensity_source": "v17 pixels only",
        "analytic_constraint_role": "localization mask only",
        "spatial_transform": None,
        "resampling": None,
        "whole_lamella_redraw": False,
        "boundary_gain": float(boundary_gain),
        "endpoint_gain": float(endpoint_gain),
        "interlayer_residual_shrinkage": float(interlayer_denoise),
        "maximum_absolute_change_normalized": float(cap),
        "changed_support_fraction": float(np.mean(influence > 1e-4)),
    }, influence


def unsafe_constraint_ids(audit: dict, baseline_audit: dict) -> set[str]:
    """Return structures that must be restored to the exact v17 baseline."""
    unsafe: set[str] = set()
    baseline_layers = {
        row["layer_id"]: row for row in baseline_audit["layer_comparison_rows"]
    }
    for row in audit["layer_comparison_rows"]:
        baseline = baseline_layers[row["layer_id"]]
        width_limit = max(
            0.0035,
            float(baseline["output_vs_guide_width_relative"]) + 0.00015,
        )
        length_limit = max(
            0.10,
            float(baseline["output_vs_guide_length_abs_px"]) + 0.005,
        )
        if (
            float(row["output_vs_guide_width_relative"]) > width_limit
            or float(row["output_vs_guide_length_abs_px"]) > length_limit
            or (bool(baseline["dual_evidence_pass"]) and not bool(row["dual_evidence_pass"]))
        ):
            unsafe.add(row["layer_id"])
    baseline_gaps = {
        row["gap_id"]: row for row in baseline_audit["gap_comparison_rows"]
    }
    for row in audit["gap_comparison_rows"]:
        baseline = baseline_gaps[row["gap_id"]]
        width_limit = max(
            0.0040,
            float(baseline["output_vs_guide_width_relative"]) + 0.00020,
        )
        length_limit = max(
            0.12,
            float(baseline["output_vs_guide_length_abs_px"]) + 0.005,
        )
        if (
            float(row["output_vs_guide_width_relative"]) > width_limit
            or float(row["output_vs_guide_length_abs_px"]) > length_limit
            or (bool(baseline["dual_evidence_pass"]) and not bool(row["dual_evidence_pass"]))
        ):
            unsafe.update((row["left_layer_id"], row["right_layer_id"]))
    return unsafe


def rollback_unsafe_structures(
    candidate: np.ndarray,
    baseline: np.ndarray,
    layers: list[dict],
    unsafe_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Restore unsafe lamella/gap neighborhoods without touching accepted ones."""
    mask = np.zeros_like(candidate, dtype=np.float32)
    for row in layers:
        if row["layer_id"] not in unsafe_ids:
            continue
        pitch = max(float(row["pitch_px"]), 4.0)
        path = np.asarray(row["path_x"], dtype=np.float32)
        y0 = max(0, int(math.floor(float(row["top_y_px"]) - 9.0)))
        y1 = min(candidate.shape[0] - 1, int(math.ceil(float(row["bottom_y_px"]) + 9.0)))
        radius = max(6, int(math.ceil(0.62 * pitch)))
        for y in range(y0, y1 + 1):
            center = int(round(float(path[y])))
            x0 = max(0, center - radius)
            x1 = min(candidate.shape[1], center + radius + 1)
            mask[y, x0:x1] = 1.0
    # Full restoration over every measurement profile with a one-pixel outer
    # feather to avoid introducing a visible seam.
    feather = np.clip(gaussian_filter(mask, sigma=0.65), 0.0, 1.0)
    feather[mask > 0.5] = 1.0
    output = candidate * (1.0 - feather) + baseline * feather
    return output.astype(np.float32), feather.astype(np.float32)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_projection_audit(
    path: Path,
    generated: np.ndarray,
    projected: np.ndarray,
    alpha: np.ndarray,
    target: base.Roi,
) -> None:
    limits = tuple(float(value) for value in np.percentile(generated[target.slices()], (0.5, 99.7)))
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    items = (
        (generated, "v17 input"),
        (projected, "v18 hybrid"),
        (alpha, "boundary/endpoint projection weight"),
        (np.abs(projected - generated), "absolute change"),
    )
    for values, title in items:
        crop = values[target.slices()]
        if title in {"v17 input", "v18 hybrid"}:
            lo, hi = limits
        else:
            lo, hi = 0.0, max(float(np.percentile(crop, 99.7)), 1e-8)
        rendered = np.rint(np.clip((crop - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
        panel = Image.fromarray(rendered, mode="L").convert("RGB")
        canvas = Image.new("RGB", (panel.width, panel.height + 24), "black")
        canvas.paste(panel, (0, 24))
        ImageDraw.Draw(canvas).text((8, 7), title, fill="white", font=font)
        panels.append(canvas)
    output = Image.new("RGB", (2 * panels[0].width, 2 * panels[0].height), "black")
    for index, panel in enumerate(panels):
        output.paste(panel, ((index % 2) * panel.width, (index // 2) * panel.height))
    output.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-roi", type=base.parse_roi, default=base.Roi(600, 1240, 300, 1900))
    parser.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    parser.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    parser.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    parser.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    carrier, carrier_info = base.load_gray(args.carrier)
    generated, generated_info = base.load_gray(args.generated)
    if source.shape != carrier.shape or source.shape != generated.shape:
        raise ValueError(
            f"all inputs must share source dimensions: source={source.shape}, "
            f"carrier={carrier.shape}, generated={generated.shape}"
        )
    target = args.target_roi.clamp(source.shape)
    left = args.left_roi.clamp(source.shape)
    right = args.right_roi.clamp(source.shape)
    layers = shape.measure_layers(
        source, left, right, args.top_range, args.bottom_range, guide=carrier
    )
    baseline_audit = project.audit_projection(
        source, carrier, generated, generated, layers, left, right, args.top_range, args.bottom_range
    )
    baseline_summary = v16.metric_summary(baseline_audit)
    maps = v17.build_structure_maps(source.shape, target, layers)
    gap_distance = distance_transform_edt(maps["interlayer"] > 0.55)
    gap_audit_mask = np.clip((gap_distance - 1.6) / 0.8, 0.0, 1.0).astype(np.float32)
    baseline_gap_noise = v17.masked_noise_rms(generated, gap_audit_mask, target)

    candidates: list[dict] = []
    selected: dict | None = None
    grid = (
        (0.06, 0.025, 0.08),
        (0.10, 0.040, 0.12),
        (0.16, 0.060, 0.18),
        (0.24, 0.085, 0.24),
    )
    _, width_rows = measure_local_width_profiles(
        carrier, layers, variation_strength=0.35
    )
    selected_width_rows: list[dict] = []
    for boundary_gain, endpoint_gain, interlayer_denoise in grid:
        candidate, method, alpha = symmetric_constraint_cleanup(
            generated,
            target,
            maps,
            boundary_gain=boundary_gain,
            endpoint_gain=endpoint_gain,
            interlayer_denoise=interlayer_denoise,
        )
        audit = project.audit_projection(
            source, carrier, generated, candidate, layers, left, right, args.top_range, args.bottom_range
        )
        pre_rollback_summary = v16.metric_summary(audit)
        rollback_rounds = []
        rolled_back_ids: set[str] = set()
        for rollback_round in range(3):
            unsafe = unsafe_constraint_ids(audit, baseline_audit)
            unsafe -= rolled_back_ids
            if not unsafe:
                break
            candidate, rollback_mask = rollback_unsafe_structures(
                candidate, generated, layers, unsafe
            )
            alpha *= 1.0 - rollback_mask[target.slices()]
            rolled_back_ids.update(unsafe)
            rollback_rounds.append(
                {
                    "round": rollback_round + 1,
                    "restored_layer_count": len(unsafe),
                    "restored_layer_ids": sorted(unsafe),
                }
            )
            audit = project.audit_projection(
                source,
                carrier,
                generated,
                candidate,
                layers,
                left,
                right,
                args.top_range,
                args.bottom_range,
            )
        summary = v16.metric_summary(audit)
        target_ssim = float(
            structural_similarity(generated[target.slices()], candidate[target.slices()], data_range=1.0)
        )
        passed, checks = hybrid_guardrail(summary, baseline_summary, target_ssim)
        edge_gain = summary["edge_clarity"] / max(baseline_summary["edge_clarity"], 1e-8) - 1.0
        gap_noise = v17.masked_noise_rms(candidate, gap_audit_mask, target)
        gap_noise_reduction = 1.0 - gap_noise / max(baseline_gap_noise, 1e-8)
        score = (
            2.0 * edge_gain
            + 0.50 * gap_noise_reduction
            - 12.0 * summary["lamella_width_relative_error_p95"]
            - 6.0 * summary["interlayer_width_relative_error_p95"]
            + 0.20 * summary["lamella_axial_detail_correlation_median"]
        )
        item = {
            "boundary_gain": boundary_gain,
            "endpoint_gain": endpoint_gain,
            "interlayer_denoise": interlayer_denoise,
            "local_width_constraint_variation_strength": 0.35,
            "ssim_vs_v17": target_ssim,
            "edge_clarity_gain_vs_v17": edge_gain,
            "interlayer_high_frequency_noise_reduction_vs_v17": gap_noise_reduction,
            "structure_geometry": summary,
            "guardrail_checks": checks,
            "guardrail_pass": passed,
            "score": score,
            "method": method,
            "pre_rollback_structure_geometry": pre_rollback_summary,
            "selective_rollback": rollback_rounds,
            "total_rolled_back_layer_count": len(rolled_back_ids),
        }
        candidates.append(item)
        print(json.dumps({"candidate": item}, ensure_ascii=False), flush=True)
        if passed and (selected is None or score > selected["summary"]["score"]):
            selected = {
                "summary": item,
                "image": candidate,
                "alpha": alpha,
                "audit": audit,
            }
            selected_width_rows = width_rows
    if selected is None:
        (args.outdir / "FAILED_candidate_grid_v18.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError("No v18 hybrid candidate passed geometry and detail guardrails")

    final = selected["image"]
    alpha = selected["alpha"]
    audit = selected["audit"]
    condition_rows = audit.pop("condition_boundary_rows")
    audit.pop("raw_boundary_rows")
    layer_rows = audit.pop("layer_comparison_rows")
    gap_rows = audit.pop("gap_comparison_rows")
    detail_rows = audit.pop("structure_detail_rows")

    output_tif = args.outdir / "MEASUREMENT_CANDIDATE_v18_clean_edges_detail_preserved_16bit.tif"
    imwrite(
        output_tif,
        base.to_uint16(final),
        photometric="minisblack",
        description=(
            "MEASUREMENT_CANDIDATE v18: v17 generative denoising plus zero-phase "
            "finite-width constraint-field cleanup and selective per-layer rollback."
        ),
    )
    base.save_preview(args.outdir / "MEASUREMENT_CANDIDATE_v18_clean_edges_detail_preserved.png", final)
    project.save_comparison(
        args.outdir / "MEASUREMENT_CANDIDATE_v18_comparison.png",
        source,
        carrier,
        generated,
        final,
    )
    save_projection_audit(
        args.outdir / "AUDIT_v18_boundary_only_projection.png",
        generated,
        final,
        alpha,
        target,
    )
    write_rows(args.outdir / "lamella_v18_comparison.csv", layer_rows)
    write_rows(args.outdir / "interlayer_v18_comparison.csv", gap_rows)
    write_rows(args.outdir / "structure_detail_v18.csv", detail_rows)
    write_rows(args.outdir / "local_width_constraints_v18.csv", selected_width_rows)

    changed = np.abs(final - generated) > (0.5 / 65535.0)
    payload = {
        "completed": True,
        "release": "v18-clean-boundary-detail-preserving-fusion",
        "status": "MEASUREMENT_CANDIDATE; calibrated multi-image validation required",
        "source": source_info,
        "carrier": carrier_info,
        "generated_input": generated_info,
        "dimensions": {"width": source.shape[1], "height": source.shape[0]},
        "design": {
            "v17_generated_denoising_retained": True,
            "v11_finite_width_boundary_principle_retained": True,
            "whole_lamella_analytic_redraw": False,
            "analytic_renderer_pixel_writeback": False,
            "row_varying_width_constraint_audit_from_carrier": True,
            "interior_core_detail_source": "same-coordinate v17 result",
            "boundary_and_endpoint_source": "zero-phase filtering of existing v17 pixels inside carrier-measured constraint fields",
            "spatial_transform": None,
            "resampling": None,
        },
        "baseline_v17": baseline_summary,
        "selected": selected["summary"],
        "candidate_grid": candidates,
        "audit": audit,
        "pixel_change": {
            "changed_pixels": int(np.sum(changed)),
            "changed_fraction_full_image": float(np.mean(changed)),
            "changed_outside_target_roi": int(
                np.sum(changed) - np.sum(changed[target.slices()])
            ),
        },
        "measurement_warning": (
            "The output contains generated pixels and constraint-gated filtered pixels. "
            "CSV constraints and the non-generated carrier remain the authoritative "
            "measurement references until phantom calibration is complete."
        ),
    }
    (args.outdir / "constrained_detail_fusion_v18_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"completed": True, "output": str(output_tif), "selected": selected["summary"]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
