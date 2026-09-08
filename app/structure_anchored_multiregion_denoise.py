#!/usr/bin/env python3
"""v19 structure-anchored multi-region denoising after v18.

V18 already contains the accepted structure-conditioned generative residual and
finite-width edge cleanup.  This stage deliberately leaves those operations in
place and optimizes denoising separately. Endpoint-envelope pixels are restored
exactly after every candidate. The complete lamella/interlayer bundles use a
strictly axial estimate (no transverse mixing); the central solid, endpoint
exterior, and low-structure background use independent zero-phase estimates.
Every candidate is remeasured and unsafe lamella/gap neighborhoods are restored
to the exact v18 baseline before release. No registration, resize, warp, or
analytic redraw is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
)
from skimage.metrics import structural_similarity
from tifffile import imread, imwrite

import boundary_cleanup as boundary
import constrained_detail_fusion as v18
import generative_shape_constraint as shape
import generative_shape_project as project
import measurement_quality_optimize as v16
import pipeline as base
import quality_optimize as quality
import structure_conditioned_diffusion as v17


def adaptive_wiener_delta(
    crop: np.ndarray,
    noise_sigma: float,
    filter_sigma: float | tuple[float, float],
    local_sigma: float | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a zero-phase local-Wiener residual reduction and shrinkage map."""
    low = gaussian_filter(crop, sigma=filter_sigma)
    high = crop - low
    local_energy = gaussian_filter(np.square(high), sigma=local_sigma)
    noise_variance = float(noise_sigma) ** 2
    residual_keep = np.clip(
        (local_energy - noise_variance) / (local_energy + 1e-12), 0.0, 1.0
    )
    delta = -(1.0 - residual_keep) * high
    return delta.astype(np.float32), (1.0 - residual_keep).astype(np.float32)


def measurement_operator_anchor(
    image_shape: tuple[int, int],
    target: base.Roi,
    layers: list[dict],
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """Rasterize every pixel that can affect the deployed geometry operators.

    Width measurement averages ``y +/- 2``, smooths a complete local profile,
    finds a local peak, and uses the profile ends as its background estimate.
    Marking only a fitted half-height contour is therefore insufficient.  This
    audit map includes the complete profile support at all 25 deployed sample
    rows.  The endpoint detector evaluates three tracked profiles over fixed
    search ranges, so those samples and a four-row Gaussian support margin are
    marked as well.  Low-confidence structures conservatively mark their full
    local Voronoi cell.  This map is diagnostic; only the separate endpoint /
    envelope mask is restored bit exactly, while width invariance is enforced
    by remeasurement and selective rollback.
    """
    height = target.y1 - target.y0
    width = target.x1 - target.x0
    anchor = np.zeros((height, width), dtype=bool)
    width_support = np.zeros_like(anchor)
    endpoint_support = np.zeros_like(anchor)
    review_support = np.zeros_like(anchor)

    def paint(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> None:
        y0 = max(target.y0, y0) - target.y0
        y1 = min(target.y1, y1) - target.y0
        x0 = max(target.x0, x0) - target.x0
        x1 = min(target.x1, x1) - target.x0
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] = True

    for row in layers:
        path = np.asarray(row["path_x"], dtype=np.float32)
        pitch = max(float(row["pitch_px"]), 4.0)
        top = float(row["top_y_px"])
        bottom = float(row["bottom_y_px"])
        margin = max(5.0, 0.10 * (bottom - top))
        if bottom - top > 2.0 * margin:
            for position in np.linspace(top + margin, bottom - margin, 25):
                y = int(np.clip(round(position), 2, image_shape[0] - 3))
                center = float(path[y])
                radius = max(4, int(math.floor(0.45 * pitch)))
                x0 = int(math.floor(center)) - radius - 3
                x1 = int(math.ceil(center)) + radius + 4
                paint(width_support, y - 2, y + 3, x0, x1)

        # ``sample_tracked_profile`` uses path offsets -1/0/+1, with a
        # three-pixel transverse average.  The four-row longitudinal margin
        # covers the effective sigma=1.05 endpoint smoothing support.
        for search in (top_range, bottom_range):
            for y in range(max(0, search[0] - 4), min(image_shape[0], search[1] + 4)):
                center = int(round(float(path[y])))
                paint(endpoint_support, y, y + 1, center - 3, center + 4)

        if row.get("reference_quality") != "pass":
            radius = max(5, int(math.ceil(0.52 * pitch)))
            for y in range(
                max(target.y0, int(math.floor(top)) - 8),
                min(target.y1, int(math.ceil(bottom)) + 9),
            ):
                center = int(round(float(path[y])))
                paint(review_support, y, y + 1, center - radius, center + radius + 1)

    anchor |= width_support | endpoint_support | review_support
    return anchor, {
        "method": "exact deployed FWHM/endpoint sampling-support rasterization",
        "width_sample_rows_per_lamella": 25,
        "width_longitudinal_support_radius_px": 2,
        "width_transverse_filter_support_margin_px": 3,
        "endpoint_longitudinal_filter_support_margin_px": 4,
        "low_confidence_full_cell_freeze": True,
        "width_support_fraction_target": float(np.mean(width_support)),
        "endpoint_support_fraction_target": float(np.mean(endpoint_support)),
        "review_support_fraction_target": float(np.mean(review_support)),
        "union_fraction_target": float(np.mean(anchor)),
    }


def build_multiregion_components(
    image: np.ndarray,
    target: base.Roi,
    central_roi: base.Roi,
    left_roi: base.Roi,
    right_roi: base.Roi,
    maps: dict[str, np.ndarray],
    envelope_masks: dict[str, np.ndarray],
    operator_anchor: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build the four empirically safe v19 regions and Wiener residuals.

    The complete left/right bundles are filtered only along their long axis;
    no x-direction sample is borrowed across a lamella or gap.  The central
    solid, endpoint-exterior fog ring, and low-gradient outer background are
    disjoint and receive their own isotropic estimates.  Endpoint contours are
    bit-exact anchors.  The full FWHM/endpoint operator support is retained as
    an audit field: freezing its scattered sample bands created visible seams,
    so invariance there is enforced by row-level measurement and rollback.
    """
    crop = image[target.slices()]
    sigma_n = base.noise_sigma_mad(crop)
    yy, xx = np.indices(crop.shape)
    side_x = (
        (xx >= left_roi.x0 - target.x0 - 8)
        & (xx < left_roi.x1 - target.x0 + 8)
    ) | (
        (xx >= right_roi.x0 - target.x0 - 8)
        & (xx < right_roi.x1 - target.x0 + 8)
    )
    tip_anchor = binary_dilation(maps["endpoint"] > 0.08, iterations=2) | (
        envelope_masks["boundary"] > 0.22
    )
    stack_gate = (
        (envelope_masks["interior"] > 0.62) & side_x & (~tip_anchor)
    )

    central_roi = central_roi.clamp(image.shape)
    central_gate = np.zeros_like(crop, dtype=bool)
    cy0, cy1 = central_roi.y0 - target.y0, central_roi.y1 - target.y0
    cx0, cx1 = central_roi.x0 - target.x0, central_roi.x1 - target.x0
    central_gate[cy0:cy1, cx0:cx1] = True
    central_gate &= ~tip_anchor

    fog_gate = (
        (envelope_masks["exterior"] > 0.18)
        & (envelope_masks["boundary"] < 0.15)
        & side_x
        & (~tip_anchor)
    )

    smoothed = gaussian_filter(crop, sigma=0.80)
    gradient = np.hypot(*np.gradient(smoothed))
    gradient_threshold = float(np.percentile(gradient, 30.0))
    flat_gate = (
        (gradient < gradient_threshold)
        & (~side_x)
        & (~central_gate)
        & (~tip_anchor)
        & (maps["protection"] < 0.04)
    )

    stack_delta, stack_shrink = adaptive_wiener_delta(
        crop, 0.60 * sigma_n, filter_sigma=(1.60, 0.0), local_sigma=(3.20, 0.0)
    )
    central_delta, central_shrink = adaptive_wiener_delta(
        crop, 0.70 * sigma_n, filter_sigma=(0.75, 0.75), local_sigma=(2.20, 2.20)
    )
    fog_delta, fog_shrink = adaptive_wiener_delta(
        crop, 0.78 * sigma_n, filter_sigma=(0.85, 0.85), local_sigma=(2.50, 2.50)
    )
    flat_delta, flat_shrink = adaptive_wiener_delta(
        crop, 0.75 * sigma_n, filter_sigma=(0.70, 0.70), local_sigma=(2.00, 2.00)
    )
    component_cap = max(0.85 * sigma_n, 2.0 / 65535.0)
    stack_delta = np.clip(stack_delta, -component_cap, component_cap)
    central_delta = np.clip(central_delta, -component_cap, component_cap)
    fog_delta = np.clip(fog_delta, -component_cap, component_cap)
    flat_delta = np.clip(flat_delta, -component_cap, component_cap)

    gap_distance = distance_transform_edt(maps["interlayer"] > 0.55)
    gap_metric = (gap_distance >= 2.5) & stack_gate
    lamella_metric = (maps["lamella"] > 0.70) & stack_gate

    components = {
        "crop": crop.astype(np.float32),
        "hard_anchor": tip_anchor.astype(bool),
        "measurement_operator_support": operator_anchor.astype(bool),
        "side_x": side_x.astype(bool),
        "stack_gate": stack_gate.astype(np.float32),
        "lamella_metric": lamella_metric.astype(np.float32),
        "gap_metric": gap_metric.astype(np.float32),
        "central_gate": central_gate.astype(np.float32),
        "fog_gate": fog_gate.astype(np.float32),
        "flat_gate": flat_gate.astype(np.float32),
        "stack_delta": stack_delta.astype(np.float32),
        "central_delta": central_delta,
        "fog_delta": fog_delta.astype(np.float32),
        "flat_delta": flat_delta,
    }
    method = {
        "method": "endpoint-anchored four-zone adaptive Wiener residual denoising",
        "estimated_noise_sigma_normalized": float(sigma_n),
        "measurement_anchor": (
            "exact v18 pixels throughout endpoint/tip contour support"
        ),
        "hard_anchor_fraction_target": float(np.mean(tip_anchor)),
        "measurement_operator_support_fraction_target": float(
            np.mean(operator_anchor)
        ),
        "stack_filter_sigma_px": [1.60, 0.0],
        "stack_local_variance_sigma_px": [3.20, 0.0],
        "stack_noise_factor": 0.60,
        "central_filter_sigma_px": [0.75, 0.75],
        "central_noise_factor": 0.70,
        "fog_filter_sigma_px": [0.85, 0.85],
        "fog_noise_factor": 0.78,
        "flat_filter_sigma_px": [0.70, 0.70],
        "flat_noise_factor": 0.75,
        "component_change_cap_normalized": float(component_cap),
        "gradient_gate_threshold": gradient_threshold,
        "spatial_transform": None,
        "resampling": None,
        "analytic_redraw": False,
        "generated_pixel_source": "retained v17 pixels already accepted by v18",
        "shrinkage_fraction_median": {
            "stack": float(np.median(stack_shrink[stack_gate]))
            if np.any(stack_gate)
            else 0.0,
            "central": float(np.median(central_shrink[central_gate]))
            if np.any(central_gate)
            else 0.0,
            "fog": float(np.median(fog_shrink[fog_gate]))
            if np.any(fog_gate)
            else 0.0,
            "flat": float(np.median(flat_shrink[flat_gate > 0.05]))
            if np.any(flat_gate > 0.05)
            else 0.0,
        },
    }
    return components, method


def apply_multiregion_candidate(
    image: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
    noise_sigma: float,
    strengths: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Apply four independent zones and restore endpoint anchors exactly."""
    stack, central, fog, flat = (float(value) for value in strengths)
    weighted = (
        stack * components["stack_gate"] * components["stack_delta"]
        + central * components["central_gate"] * components["central_delta"]
        + fog * components["fog_gate"] * components["fog_delta"]
        + flat * components["flat_gate"] * components["flat_delta"]
    )
    cap = max(0.85 * float(noise_sigma), 2.0 / 65535.0)
    delta = np.clip(weighted, -cap, cap).astype(np.float32)
    crop = np.clip(components["crop"] + delta, 0.0, 1.0).astype(np.float32)
    anchor = components["hard_anchor"]
    crop[anchor] = components["crop"][anchor]
    output = base.feather_insert(image, crop, target, ramp=18)
    influence = np.clip(
        stack * components["stack_gate"]
        + central * components["central_gate"]
        + fog * components["fog_gate"]
        + flat * components["flat_gate"],
        0.0,
        1.0,
    ).astype(np.float32)
    influence[anchor] = 0.0
    return output, {
        "lamella_interlayer_stack_strength": stack,
        "central_strength": central,
        "endpoint_exterior_fog_strength": fog,
        "flat_background_strength": flat,
        "change_cap_normalized": cap,
        "hard_anchor_restoration": True,
        "phase_response": "zero-phase symmetric filters",
        "spatial_transform": None,
        "resampling": None,
    }, influence


def masked_residual_rms(
    image: np.ndarray,
    mask: np.ndarray,
    target: base.Roi,
    sigma: float | tuple[float, float],
) -> float:
    crop = image[target.slices()]
    selected = mask > 0.20
    if not np.any(selected):
        return 0.0
    residual_values = crop - gaussian_filter(crop, sigma=sigma)
    return float(np.sqrt(np.mean(np.square(residual_values[selected]))))


def noise_metrics(
    image: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
) -> dict:
    crop = image[target.slices()]
    medium = gaussian_filter(crop, sigma=2.20)
    broad = gaussian_filter(crop, sigma=7.0)
    positive_halo = np.maximum(medium - broad, 0.0)
    return {
        "lamella_axial_rms": masked_residual_rms(
            image, components["lamella_metric"], target, 1.0
        ),
        "interlayer_high_frequency_rms": masked_residual_rms(
            image, components["gap_metric"], target, 1.0
        ),
        "central_high_frequency_rms": masked_residual_rms(
            image, components["central_gate"], target, 1.15
        ),
        "endpoint_exterior_high_frequency_rms": masked_residual_rms(
            image, components["fog_gate"], target, 1.0
        ),
        "flat_background_high_frequency_rms": masked_residual_rms(
            image, components["flat_gate"], target, 1.15
        ),
        "endpoint_exterior_positive_halo_mean": float(
            np.sum(positive_halo * components["fog_gate"])
            / max(float(np.sum(components["fog_gate"])), 1e-12)
        ),
    }


def reduction_metrics(baseline: dict, candidate: dict) -> dict:
    result = {}
    for name, before in baseline.items():
        after = candidate[name]
        result[name.replace("_rms", "_reduction").replace("_mean", "_reduction")] = (
            1.0 - float(after) / max(float(before), 1e-12)
        )
    return result


def quantized_float(image: np.ndarray) -> np.ndarray:
    """Evaluate exactly the pixels that will be present in the uint16 TIFF."""
    return (base.to_uint16(image).astype(np.float32) / 65535.0).astype(np.float32)


def local_width_drift(
    baseline: np.ndarray,
    candidate: np.ndarray,
    layers: list[dict],
    row_step: int = 3,
) -> tuple[dict, list[dict], set[str]]:
    """Compare row-level FWHM outside the 25 deployed aggregate samples."""
    rows: list[dict] = []
    per_layer: list[dict] = []
    unsafe: set[str] = set()
    all_errors: list[float] = []
    for layer in layers:
        y0 = max(2, int(math.ceil(float(layer["top_y_px"]))))
        y1 = min(
            baseline.shape[0] - 3, int(math.floor(float(layer["bottom_y_px"])))
        )
        errors: list[float] = []
        for y in range(y0, y1 + 1, max(1, int(row_step))):
            before = shape.transverse_width_at_row(
                baseline, layer["path_x"], y, float(layer["pitch_px"])
            )
            after = shape.transverse_width_at_row(
                candidate, layer["path_x"], y, float(layer["pitch_px"])
            )
            if before is None or after is None:
                continue
            error = abs(float(after) - float(before))
            errors.append(error)
            all_errors.append(error)
            rows.append(
                {
                    "side": layer["side"],
                    "layer_id": layer["layer_id"],
                    "y_px": y,
                    "v18_width_px": float(before),
                    "v19_width_px": float(after),
                    "absolute_drift_px": error,
                }
            )
        p95 = float(np.percentile(errors, 95.0)) if errors else float("inf")
        maximum = float(np.max(errors)) if errors else float("inf")
        layer_item = {
            "side": layer["side"],
            "layer_id": layer["layer_id"],
            "sample_count": len(errors),
            "absolute_drift_p95_px": p95,
            "absolute_drift_max_px": maximum,
        }
        per_layer.append(layer_item)
        if not errors or p95 > 0.025 or maximum > 0.060:
            unsafe.add(layer["layer_id"])
    summary = {
        "row_step": int(row_step),
        "sample_count": len(all_errors),
        "absolute_drift_median_px": float(np.median(all_errors))
        if all_errors
        else None,
        "absolute_drift_p95_px": float(np.percentile(all_errors, 95.0))
        if all_errors
        else None,
        "absolute_drift_max_px": float(np.max(all_errors)) if all_errors else None,
        "per_layer_p95_limit_px": 0.025,
        "per_layer_max_limit_px": 0.060,
        "unsafe_layer_count": len(unsafe),
        "per_layer": per_layer,
    }
    return summary, rows, unsafe


def unsafe_v19_ids(
    audit: dict,
    baseline_audit: dict,
    local_width_unsafe: set[str],
) -> set[str]:
    """Combine v18 limits with direct v18 endpoint and local-width invariance."""
    unsafe = v18.unsafe_constraint_ids(audit, baseline_audit) | set(local_width_unsafe)
    baseline_rows = {
        row["layer_id"]: row for row in baseline_audit["layer_comparison_rows"]
    }
    for row in audit["layer_comparison_rows"]:
        before = baseline_rows[row["layer_id"]]
        endpoint_delta = max(
            abs(float(row["output_top_y_px"]) - float(before["output_top_y_px"])),
            abs(
                float(row["output_bottom_y_px"])
                - float(before["output_bottom_y_px"])
            ),
        )
        width_delta = abs(
            float(row["output_width_px"]) - float(before["output_width_px"])
        )
        if endpoint_delta > 0.005 or width_delta > 0.020:
            unsafe.add(row["layer_id"])
    return unsafe


def topology_invariant(audit: dict, baseline_audit: dict) -> tuple[bool, dict]:
    checks: dict[str, bool] = {}
    before = baseline_audit["ordinary_unmatched_peak_detector_diagnostic"]
    after = audit["ordinary_unmatched_peak_detector_diagnostic"]
    for side in ("left", "right"):
        checks[f"{side}_peak_count_equal"] = int(after[side]["count"]) == int(
            before[side]["count"]
        )
        checks[f"{side}_pitch_delta_le_0_02_px"] = abs(
            float(after[side]["median_pitch_px"])
            - float(before[side]["median_pitch_px"])
        ) <= 0.02
    checks["constraint_matched_counts_equal"] = (
        audit["constraint_matched_layer_detection"]["left"]
        == baseline_audit["constraint_matched_layer_detection"]["left"]
        and audit["constraint_matched_layer_detection"]["right"]
        == baseline_audit["constraint_matched_layer_detection"]["right"]
    )
    return bool(all(checks.values())), checks


def v19_guardrail(
    summary: dict,
    baseline: dict,
    target_ssim: float,
    local_width_summary: dict,
    topology_pass: bool,
    changed_hard_anchor_pixels: int,
) -> tuple[bool, dict]:
    """Retain v18 geometry while allowing visible low-structure denoising."""
    checks = {
        "endpoint_p95_strict_nonregression": (
            summary["endpoint_shift_abs_p95_px"]
            <= max(0.010, baseline["endpoint_shift_abs_p95_px"] + 0.005)
        ),
        "length_p95_strict_nonregression": (
            summary["length_delta_abs_p95_px"]
            <= max(0.080, baseline["length_delta_abs_p95_px"] + 0.005)
        ),
        "lamella_width_p95_le_0_35_percent": (
            summary["lamella_width_relative_error_p95"] <= 0.0035
        ),
        "lamella_dual_evidence_nonregression": (
            summary["lamella_dual_evidence_pass_count"]
            >= baseline["lamella_dual_evidence_pass_count"]
        ),
        "interlayer_width_p95_strict_nonregression": (
            summary["interlayer_width_relative_error_p95"]
            <= min(
                0.0040,
                baseline["interlayer_width_relative_error_p95"] + 0.00020,
            )
        ),
        "interlayer_length_p95_strict_nonregression": (
            summary["interlayer_length_abs_error_p95_px"]
            <= max(
                0.100,
                baseline["interlayer_length_abs_error_p95_px"] + 0.005,
            )
        ),
        "interlayer_dual_evidence_nonregression": (
            summary["interlayer_dual_evidence_pass_count"]
            >= baseline["interlayer_dual_evidence_pass_count"]
        ),
        "edge_clarity_nonregression": (
            summary["edge_clarity"] >= baseline["edge_clarity"]
        ),
        "low_frequency_correlation_ge_0_99999": (
            summary["low_frequency_correlation"] >= 0.99999
        ),
        "mid_frequency_correlation_ge_0_99990": (
            summary["mid_frequency_correlation"] >= 0.99990
        ),
        "gradient_correlation_ge_0_99985": (
            summary["gradient_magnitude_correlation"] >= 0.99985
        ),
        "axial_detail_median_ge_0_99997": (
            summary["lamella_axial_detail_correlation_median"] >= 0.99997
        ),
        "axial_detail_p10_ge_0_99994": (
            summary["lamella_axial_detail_correlation_p10"] >= 0.99994
        ),
        # Whole-ROI SSIM necessarily falls when noise is removed from the
        # deliberately unprotected central/fog/background zones.  Structural
        # pixels are governed by the considerably stricter geometry, edge,
        # multiscale and row-FWHM checks above, so this remains a global
        # corruption sentinel instead of suppressing useful safe-zone work.
        "ssim_ge_0_99985": target_ssim >= 0.99985,
        "row_width_global_p95_le_0_02_px": (
            local_width_summary["absolute_drift_p95_px"] is not None
            and local_width_summary["absolute_drift_p95_px"] <= 0.020
        ),
        "row_width_global_max_le_0_06_px": (
            local_width_summary["absolute_drift_max_px"] is not None
            and local_width_summary["absolute_drift_max_px"] <= 0.060
        ),
        "topology_invariant": bool(topology_pass),
        "hard_anchor_pixels_bit_exact": int(changed_hard_anchor_pixels) == 0,
    }
    return bool(all(checks.values())), checks


def save_mask_audit(
    path: Path,
    image: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
) -> None:
    crop = image[target.slices()]
    lo, hi = (float(value) for value in np.percentile(crop, (0.5, 99.7)))
    gray = np.rint(
        np.clip((crop - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255
    ).astype(np.uint8)
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    fields = (
        (components["hard_anchor"].astype(np.float32), "hard endpoint anchors"),
        (components["measurement_operator_support"].astype(np.float32), "measurement operator support"),
        (components["stack_gate"], "lamella/interlayer axial stack"),
        (components["central_gate"], "central solid"),
        (components["fog_gate"], "endpoint exterior / fog"),
        (components["flat_gate"], "low-structure background"),
    )
    for field, title in fields:
        rgb = np.repeat(gray[..., None], 3, axis=2)
        overlay = np.rint(220 * np.clip(field, 0.0, 1.0)).astype(np.uint8)
        rgb[..., 1] = np.maximum(rgb[..., 1], overlay)
        panel = Image.fromarray(rgb, mode="RGB")
        canvas = Image.new("RGB", (panel.width, panel.height + 24), "black")
        canvas.paste(panel, (0, 24))
        ImageDraw.Draw(canvas).text((8, 7), title, fill="white", font=font)
        panels.append(canvas)
    output = Image.new("RGB", (3 * panels[0].width, 2 * panels[0].height), "black")
    for index, panel in enumerate(panels):
        output.paste(panel, ((index % 3) * panel.width, (index // 3) * panel.height))
    output.save(path)


def save_comparison(
    path: Path,
    baseline: np.ndarray,
    enhanced: np.ndarray,
    target: base.Roi,
) -> None:
    before = baseline[target.slices()]
    after = enhanced[target.slices()]
    lo, hi = (float(value) for value in np.percentile(before, (0.5, 99.7)))
    delta = np.abs(after - before)
    items = (
        (before, "v18 structure-constrained input"),
        (after, "v19 anchored multi-region denoise"),
        (delta, "absolute delta (auto-scaled)"),
    )
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    for values, title in items:
        if title.startswith("absolute"):
            scale = max(float(np.percentile(values, 99.7)), 1.0 / 65535.0)
            mapped = np.clip(values / scale, 0.0, 1.0)
        else:
            mapped = np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        panel = Image.fromarray(np.rint(mapped * 255).astype(np.uint8), mode="L")
        canvas = Image.new("L", (panel.width, panel.height + 24), 0)
        canvas.paste(panel, (0, 24))
        ImageDraw.Draw(canvas).text((8, 7), title, fill=255, font=font)
        panels.append(canvas)
    output = Image.new("L", (3 * panels[0].width, panels[0].height), 0)
    for index, panel in enumerate(panels):
        output.paste(panel, (index * panel.width, 0))
    output.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--target-roi", type=base.parse_roi, default=base.Roi(600, 1240, 300, 1900)
    )
    parser.add_argument(
        "--central-roi", type=base.parse_roi, default=base.Roi(700, 1140, 985, 1205)
    )
    parser.add_argument(
        "--left-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970)
    )
    parser.add_argument(
        "--right-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830)
    )
    parser.add_argument(
        "--left-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970)
    )
    parser.add_argument(
        "--right-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830)
    )
    parser.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    parser.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    carrier, carrier_info = base.load_gray(args.carrier)
    current, input_info = base.load_gray(args.input)
    if source.shape != carrier.shape or source.shape != current.shape:
        raise ValueError(
            "source, carrier, and input must have identical dimensions: "
            f"source={source.shape}, carrier={carrier.shape}, input={current.shape}"
        )

    target = args.target_roi.clamp(source.shape)
    central = args.central_roi.clamp(source.shape)
    left = args.left_roi.clamp(source.shape)
    right = args.right_roi.clamp(source.shape)
    layers = shape.measure_layers(
        source, left, right, args.top_range, args.bottom_range, guide=carrier
    )
    maps = v17.build_structure_maps(source.shape, target, layers)
    references = quality.build_boundary_references(
        source,
        carrier,
        args.left_body_roi.clamp(source.shape),
        args.right_body_roi.clamp(source.shape),
        args.top_range,
        args.bottom_range,
    )
    envelope_masks, envelope_info = boundary.build_endpoint_envelope_masks(
        source.shape, references, target
    )
    operator_anchor, operator_anchor_info = measurement_operator_anchor(
        source.shape, target, layers, args.top_range, args.bottom_range
    )
    components, method = build_multiregion_components(
        current,
        target,
        central,
        left,
        right,
        maps,
        envelope_masks,
        operator_anchor,
    )
    sigma_n = float(method["estimated_noise_sigma_normalized"])

    baseline_audit = project.audit_projection(
        source,
        carrier,
        current,
        current,
        layers,
        left,
        right,
        args.top_range,
        args.bottom_range,
    )
    baseline_summary = v16.metric_summary(baseline_audit)
    baseline_noise = noise_metrics(current, target, components)

    # Stage A searches non-stack strength. Stage B keeps the empirically best
    # safe-region tuple and independently searches the purely axial stack
    # strength. This avoids a wasteful four-dimensional Cartesian product.
    parameter_grid = (
        (0.00, 0.00, 0.00, 0.00),
        (0.00, 0.20, 0.25, 0.20),
        (0.00, 0.30, 0.35, 0.30),
        (0.00, 0.40, 0.45, 0.40),
        (0.00, 0.50, 0.60, 0.50),
        (0.00, 0.65, 0.80, 0.65),
        (0.02, 0.50, 0.60, 0.50),
        (0.04, 0.50, 0.60, 0.50),
        (0.06, 0.50, 0.60, 0.50),
        (0.08, 0.50, 0.60, 0.50),
        (0.10, 0.50, 0.60, 0.50),
        (0.12, 0.50, 0.60, 0.50),
    )
    candidates: list[dict] = []
    selected: dict | None = None
    for strengths in parameter_grid:
        candidate, operation, influence = apply_multiregion_candidate(
            current, target, components, sigma_n, strengths
        )
        # Geometry selection is performed on the exact uint16 candidate, not
        # on an optimistic in-memory float array.
        candidate = quantized_float(candidate)
        audit = project.audit_projection(
            source,
            carrier,
            current,
            candidate,
            layers,
            left,
            right,
            args.top_range,
            args.bottom_range,
        )
        pre_rollback_summary = v16.metric_summary(audit)
        local_width_summary, local_width_rows, local_width_unsafe = local_width_drift(
            current, candidate, layers
        )
        rollback_rounds: list[dict] = []
        rolled_back_ids: set[str] = set()
        for rollback_round in range(3):
            unsafe = unsafe_v19_ids(
                audit, baseline_audit, local_width_unsafe
            ) - rolled_back_ids
            if not unsafe:
                break
            candidate, rollback_mask = v18.rollback_unsafe_structures(
                candidate, current, layers, unsafe
            )
            candidate = quantized_float(candidate)
            influence *= 1.0 - rollback_mask[target.slices()]
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
                current,
                candidate,
                layers,
                left,
                right,
                args.top_range,
                args.bottom_range,
            )
            (
                local_width_summary,
                local_width_rows,
                local_width_unsafe,
            ) = local_width_drift(current, candidate, layers)
        summary = v16.metric_summary(audit)
        target_ssim = float(
            structural_similarity(
                current[target.slices()], candidate[target.slices()], data_range=1.0
            )
        )
        current_noise = noise_metrics(candidate, target, components)
        reductions = reduction_metrics(baseline_noise, current_noise)
        anchor_changed_count = int(
            np.sum(
                base.to_uint16(candidate[target.slices()])[components["hard_anchor"]]
                != base.to_uint16(current[target.slices()])[components["hard_anchor"]]
            )
        )
        topology_pass, topology_checks = topology_invariant(audit, baseline_audit)
        passed, checks = v19_guardrail(
            summary,
            baseline_summary,
            target_ssim,
            local_width_summary,
            topology_pass,
            anchor_changed_count,
        )
        rms_reduction_keys = (
            "lamella_axial_reduction",
            "interlayer_high_frequency_reduction",
            "central_high_frequency_reduction",
            "endpoint_exterior_high_frequency_reduction",
            "flat_background_high_frequency_reduction",
        )
        region_nonregression = all(
            reductions[name] >= -1e-4 for name in rms_reduction_keys
        )
        visible_safe_region_gain = max(
            reductions["central_high_frequency_reduction"],
            reductions["endpoint_exterior_high_frequency_reduction"],
            reductions["flat_background_high_frequency_reduction"],
        ) >= 0.03
        checks["all_fixed_region_noise_metrics_nonregression"] = region_nonregression
        checks["at_least_one_safe_region_noise_reduction_ge_3_percent"] = (
            visible_safe_region_gain
        )
        rollback_fraction = len(rolled_back_ids) / max(len(layers), 1)
        bounded_rollback = rollback_fraction <= 0.15
        checks["selective_rollback_fraction_le_15_percent"] = bounded_rollback
        passed = bool(
            passed
            and region_nonregression
            and visible_safe_region_gain
            and bounded_rollback
        )
        weighted_reduction = (
            0.23 * reductions["lamella_axial_reduction"]
            + 0.24 * reductions["interlayer_high_frequency_reduction"]
            + 0.20 * reductions["central_high_frequency_reduction"]
            + 0.18 * reductions["endpoint_exterior_high_frequency_reduction"]
            + 0.10 * reductions["flat_background_high_frequency_reduction"]
            + 0.05 * reductions["endpoint_exterior_positive_halo_reduction"]
        )
        edge_retention = summary["edge_clarity"] / max(
            baseline_summary["edge_clarity"], 1e-12
        )
        minimum_safe_region_reduction = min(
            reductions["central_high_frequency_reduction"],
            reductions["endpoint_exterior_high_frequency_reduction"],
            reductions["flat_background_high_frequency_reduction"],
        )
        score = minimum_safe_region_reduction + 0.25 * weighted_reduction
        item = {
            **operation,
            "ssim_vs_v18": target_ssim,
            "noise_before": baseline_noise,
            "noise_after": current_noise,
            "noise_reduction_vs_v18": reductions,
            "weighted_noise_reduction": weighted_reduction,
            "minimum_safe_region_noise_reduction": minimum_safe_region_reduction,
            "edge_clarity_retention_vs_v18": edge_retention,
            "structure_geometry": summary,
            "pre_rollback_structure_geometry": pre_rollback_summary,
            "selective_rollback": rollback_rounds,
            "total_rolled_back_layer_count": len(rolled_back_ids),
            "selective_rollback_fraction": rollback_fraction,
            "local_row_width_invariance": {
                key: value
                for key, value in local_width_summary.items()
                if key != "per_layer"
            },
            "topology_checks": topology_checks,
            "changed_hard_anchor_pixels_uint16": anchor_changed_count,
            "guardrail_checks": checks,
            "guardrail_pass": passed,
            "score": score,
        }
        candidates.append(item)
        log_item = dict(item)
        print(json.dumps({"candidate": log_item}, ensure_ascii=False), flush=True)
        if passed and (selected is None or score > selected["summary"]["score"]):
            selected = {
                "summary": item,
                "image": candidate,
                "audit": audit,
                "influence": influence,
                "local_width_rows": local_width_rows,
            }

    if selected is None:
        (args.outdir / "FAILED_candidate_grid_v19.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("No v19 candidate passed the v18 geometry/detail guardrails")

    final = selected["image"]
    audit = selected["audit"]
    condition_rows = audit.pop("condition_boundary_rows")
    audit.pop("raw_boundary_rows")
    layer_rows = audit.pop("layer_comparison_rows")
    gap_rows = audit.pop("gap_comparison_rows")
    detail_rows = audit.pop("structure_detail_rows")

    output_tif = (
        args.outdir
        / "MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion_16bit.tif"
    )
    imwrite(
        output_tif,
        base.to_uint16(final),
        photometric="minisblack",
        description=(
            "MEASUREMENT_CANDIDATE v19: retained v17/v18 generative enhancement, "
            "exact endpoint anchors, axial-stack/adaptive regional denoising, and "
            "selective per-layer rollback."
        ),
    )
    reloaded_u16 = np.asarray(imread(output_tif))
    if reloaded_u16.shape != source.shape or reloaded_u16.dtype != np.uint16:
        raise RuntimeError(
            f"release TIFF mismatch: shape={reloaded_u16.shape}, dtype={reloaded_u16.dtype}"
        )
    reloaded = reloaded_u16.astype(np.float32) / 65535.0
    if not np.array_equal(reloaded_u16, base.to_uint16(final)):
        raise RuntimeError("release TIFF pixels changed during write/read round trip")
    release_audit = project.audit_projection(
        source,
        carrier,
        current,
        reloaded,
        layers,
        left,
        right,
        args.top_range,
        args.bottom_range,
    )
    release_summary = v16.metric_summary(release_audit)
    release_local_summary, _, _ = local_width_drift(current, reloaded, layers)
    release_topology_pass, release_topology_checks = topology_invariant(
        release_audit, baseline_audit
    )
    release_anchor_changed = int(
        np.sum(
            reloaded_u16[target.slices()][components["hard_anchor"]]
            != base.to_uint16(current[target.slices()])[components["hard_anchor"]]
        )
    )
    release_ssim = float(
        structural_similarity(
            current[target.slices()], reloaded[target.slices()], data_range=1.0
        )
    )
    release_pass, release_checks = v19_guardrail(
        release_summary,
        baseline_summary,
        release_ssim,
        release_local_summary,
        release_topology_pass,
        release_anchor_changed,
    )
    if not release_pass:
        raise RuntimeError(
            "Written uint16 v19 TIFF failed the post-write release audit: "
            + json.dumps(release_checks, ensure_ascii=False)
        )
    base.save_preview(
        args.outdir / "MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion.png",
        final,
    )
    save_comparison(
        args.outdir / "MEASUREMENT_CANDIDATE_v19_comparison.png",
        current,
        final,
        target,
    )
    save_mask_audit(
        args.outdir / "AUDIT_v19_multiregion_masks.png", current, target, components
    )
    shape.write_csv(args.outdir / "lamella_v19_comparison.csv", layer_rows)
    shape.write_csv(args.outdir / "interlayer_v19_comparison.csv", gap_rows)
    shape.write_csv(args.outdir / "structure_detail_v19.csv", detail_rows)
    shape.write_csv(
        args.outdir / "local_row_width_v19_comparison.csv",
        selected["local_width_rows"],
    )

    changed = np.abs(final - current) > (0.5 / 65535.0)
    anchor_changed = np.abs(
        final[target.slices()][components["hard_anchor"]]
        - current[target.slices()][components["hard_anchor"]]
    ) > (0.5 / 65535.0)
    payload = {
        "completed": True,
        "release": "v19-structure-anchored-multiregion-denoise",
        "status": "MEASUREMENT_CANDIDATE; calibrated multi-image validation required",
        "source": source_info,
        "carrier": carrier_info,
        "input_v18": input_info,
        "dimensions": {"width": source.shape[1], "height": source.shape[0]},
        "design": {
            "v17_generative_enhancement_retained": True,
            "v18_finite_width_edge_cleanup_retained": True,
            "endpoint_anchor_pixel_source": "exact same-coordinate v18 pixels",
            "width_invariance_mechanism": (
                "pure axial stack filtering plus per-row FWHM audit and selective rollback"
            ),
            "independent_regions": [
                "complete lamella/interlayer stacks, axial-only",
                "central solid",
                "endpoint exterior/fog",
                "low-structure background",
            ],
            "analytic_renderer_pixel_writeback": False,
            "spatial_transform": None,
            "resampling": None,
        },
        "method": method,
        "endpoint_envelope": envelope_info,
        "measurement_operator_support": operator_anchor_info,
        "baseline_v18": baseline_summary,
        "selected": selected["summary"],
        "candidate_grid": candidates,
        "audit": audit,
        "post_write_uint16_release_audit": {
            "passed": release_pass,
            "checks": release_checks,
            "topology_checks": release_topology_checks,
            "ssim_vs_v18": release_ssim,
            "structure_geometry": release_summary,
            "local_row_width_invariance": release_local_summary,
            "changed_hard_anchor_pixels": release_anchor_changed,
        },
        "pixel_change": {
            "changed_pixels": int(np.sum(changed)),
            "changed_fraction_full_image": float(np.mean(changed)),
            "changed_outside_target_roi": int(
                np.sum(changed) - np.sum(changed[target.slices()])
            ),
            "changed_hard_anchor_pixels_after_uint16_tolerance": int(
                np.sum(anchor_changed)
            ),
        },
        "measurement_warning": (
            "Generated v17 pixels remain in this image. The exact hard anchors, "
            "per-structure CSV audit, and non-generated v16 carrier remain the "
            "authoritative references until calibrated phantom validation."
        ),
    }
    (args.outdir / "structure_anchored_multiregion_v19_metrics.json").write_text(
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
