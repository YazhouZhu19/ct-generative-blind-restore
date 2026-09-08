#!/usr/bin/env python3
"""v20 measurement-safe low-strength TV post-processing after v19.

The v17 generative estimate, v18 finite-width cleanup, and v19 zoned denoising
are retained exactly.  V20 does not run another generator.  It applies a
low-strength Chambolle total-variation residual only inside three non-measurement zones:
the configured central-ROI interior, endpoint-exterior fog, and low-structure background.

The complete lamella/interlayer bundles, their endpoint/envelope support, the
deployed measurement-operator support, the configured central-ROI border, and other
strong edges are copied bit-for-bit from v19 in uint16 space.  Every candidate
is audited after quantization and the released TIFF is read back and audited a
second time.  There is no registration, resize, warp, resampling, analytic
redraw, CLAHE, sharpening, or black-level clipping in this measurement file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
)
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_tv_chambolle
from tifffile import imread, imwrite

import boundary_cleanup as boundary
import generative_shape_constraint as shape
import generative_shape_project as project
import measurement_quality_optimize as v16
import pipeline as base
import quality_optimize as quality
import structure_anchored_multiregion_denoise as v19
import structure_conditioned_diffusion as v17


def smoothstep(values: np.ndarray) -> np.ndarray:
    """C1-continuous clamp used for seam-free spatial gates."""
    clipped = np.clip(values, 0.0, 1.0).astype(np.float32)
    return (clipped * clipped * (3.0 - 2.0 * clipped)).astype(np.float32)


def distance_soft_region(binary: np.ndarray, ramp_px: float = 6.0) -> np.ndarray:
    """Return a compact, seam-safe gate with a zero-valued outer contour.

    A blurred binary mask remains non-zero outside its intended support; simply
    multiplying that blur by the binary mask then introduces a step at the
    support boundary.  This distance gate is exactly zero outside and on the
    first inside contour, and reaches one through a C1-continuous smoothstep.
    """
    support = np.asarray(binary, dtype=bool)
    if not np.any(support):
        return np.zeros(support.shape, dtype=np.float32)
    # Explicit zero padding makes crop-border support taper exactly like an
    # internal boundary; scipy's EDT otherwise has no exterior array samples.
    padded = np.pad(support, 1, mode="constant", constant_values=False)
    distance = distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)
    scale = max(float(ramp_px) - 1.0, 1.0)
    gate = smoothstep((distance - 1.0) / scale)
    gate *= support.astype(np.float32)
    return gate.astype(np.float32)


def weighted_zero_mean(delta: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Remove a region's DC shift without changing values outside its gate."""
    denominator = float(np.sum(gate))
    if denominator <= 1e-8:
        return np.zeros_like(delta, dtype=np.float32)
    mean = float(np.sum(delta * gate) / denominator)
    return (delta - mean).astype(np.float32)


def central_masks(
    target: base.Roi,
    central_roi: base.Roi,
    shape_: tuple[int, int],
    *,
    lock_width: int = 16,
    full_weight_distance: int = 26,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return central rectangle, a hard boundary ring, and eroded soft core."""
    height, width = shape_
    rectangle = np.zeros((height, width), dtype=bool)
    y0 = max(0, central_roi.y0 - target.y0)
    y1 = min(height, central_roi.y1 - target.y0)
    x0 = max(0, central_roi.x0 - target.x0)
    x1 = min(width, central_roi.x1 - target.x0)
    if y1 <= y0 or x1 <= x0:
        return rectangle, rectangle.copy(), rectangle.astype(np.float32)
    rectangle[y0:y1, x0:x1] = True
    inside_distance = distance_transform_edt(rectangle)
    outside_ring = binary_dilation(rectangle, iterations=max(1, lock_width // 2))
    boundary_lock = (rectangle & (inside_distance <= lock_width)) | (
        outside_ring & (~rectangle)
    )
    scale = max(1, full_weight_distance - lock_width)
    core = smoothstep((inside_distance - float(lock_width)) / float(scale))
    core *= rectangle.astype(np.float32)
    return rectangle, boundary_lock, core


def build_v20_components(
    current: np.ndarray,
    target: base.Roi,
    central_roi: base.Roi,
    maps: dict[str, np.ndarray],
    envelope_masks: dict[str, np.ndarray],
    v19_components: dict[str, np.ndarray],
    operator_support: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build disjoint writable zones and low-strength TV residual estimates."""
    crop = current[target.slices()].astype(np.float32)
    side_x = v19_components["side_x"]

    # Freeze every same-coordinate pixel belonging to the full measurable
    # lamella/interlayer bundles.  V20 therefore cannot accumulate another
    # width or endpoint error on top of v19.
    stack_lock = (
        ((envelope_masks["interior"] > 0.04) & side_x)
        | (maps["lamella"] > 0.015)
        | (maps["interlayer"] > 0.04)
        | (maps["boundary"] > 0.015)
        | (maps["endpoint"] > 0.015)
        | operator_support
        | v19_components["hard_anchor"]
    )
    stack_lock = binary_dilation(stack_lock, iterations=1)

    central_rectangle, central_boundary_lock, central_core = central_masks(
        target, central_roi, crop.shape
    )

    smoothed = gaussian_filter(crop, sigma=0.80)
    gy, gx = np.gradient(smoothed)
    gradient = np.hypot(gx, gy)
    gradient_threshold = float(np.percentile(gradient, 98.5))
    strong_edge_lock = binary_dilation(
        gradient >= max(gradient_threshold, 1.0 / 65535.0), iterations=4
    )

    hard_lock = (
        stack_lock
        | central_boundary_lock
        | strong_edge_lock
        | v19_components["hard_anchor"]
        | operator_support
    )
    # Start from the empirically validated v19 regions, erode the central
    # rectangle, then soften every binary support.  V19's flat/fog regions are
    # fixed before candidate search so all candidates use identical samples.
    central_binary = (v19_components["central_gate"] > 0.20) & (central_core > 0.0)
    fog_binary = (v19_components["fog_gate"] > 0.20) & (~central_rectangle)
    flat_binary = (v19_components["flat_gate"] > 0.20) & (~central_rectangle)
    central_binary &= ~hard_lock
    fog_binary &= ~hard_lock
    flat_binary &= ~hard_lock
    fog_binary &= ~central_binary
    flat_binary &= ~(central_binary | fog_binary)
    gate_ramp_px = 8.0
    central_gate = distance_soft_region(central_binary, gate_ramp_px) * central_core
    fog_gate = distance_soft_region(fog_binary, gate_ramp_px)
    flat_gate = distance_soft_region(flat_binary, gate_ramp_px)
    writable_support = (central_gate > 0.0) | (fog_gate > 0.0) | (flat_gate > 0.0)
    binary_support = central_binary | fog_binary | flat_binary
    # This contains the original first contour (and any too-narrow support)
    # while being definitionally disjoint from every final non-zero gate.
    gate_zero_contour = binary_support & (~writable_support)
    gate_seam_inner = (
        ((central_gate > 0.0) & (~binary_erosion(central_gate > 0.0, iterations=1)))
        | ((fog_gate > 0.0) & (~binary_erosion(fog_gate > 0.0, iterations=1)))
        | ((flat_gate > 0.0) & (~binary_erosion(flat_gate > 0.0, iterations=1)))
    )

    sigma_n = float(base.noise_sigma_mad(crop))
    cap = max(0.65 * sigma_n, 2.0 / 65535.0)
    tv_estimates = {
        "tv04": denoise_tv_chambolle(
            crop,
            weight=0.0004,
            eps=2e-5,
            max_num_iter=80,
            channel_axis=None,
        ).astype(np.float32),
        "tv08": denoise_tv_chambolle(
            crop,
            weight=0.0008,
            eps=2e-5,
            max_num_iter=80,
            channel_axis=None,
        ).astype(np.float32),
        "tv12": denoise_tv_chambolle(
            crop,
            weight=0.0012,
            eps=2e-5,
            max_num_iter=80,
            channel_axis=None,
        ).astype(np.float32),
    }
    deltas: dict[str, np.ndarray] = {}
    for estimator, estimate in tv_estimates.items():
        for region, gate in (
            ("central", central_gate),
            ("fog", fog_gate),
            ("flat", flat_gate),
        ):
            # Only the configured central-ROI interior has a one-DN mean-preservation release
            # limit.  TV itself is range preserving; leaving fog/background
            # residuals uncentered also avoids creating new clipped pixels.
            residual = estimate - crop
            centered = (
                weighted_zero_mean(residual, gate)
                if region == "central"
                else residual.astype(np.float32)
            )
            deltas[f"{region}_{estimator}_delta"] = np.clip(
                centered, -cap, cap
            ).astype(np.float32)

    components = {
        "crop": crop,
        "stack_lock": stack_lock.astype(bool),
        "central_boundary_lock": central_boundary_lock.astype(bool),
        "strong_edge_lock": strong_edge_lock.astype(bool),
        "operator_support": operator_support.astype(bool),
        "hard_lock": hard_lock.astype(bool),
        "central_gate": central_gate.astype(np.float32),
        "fog_gate": fog_gate.astype(np.float32),
        "flat_gate": flat_gate.astype(np.float32),
        "writable_support": writable_support.astype(bool),
        "gate_zero_contour": gate_zero_contour.astype(bool),
        "gate_seam_inner": gate_seam_inner.astype(bool),
        **deltas,
    }
    method = {
        "method": "measurement-locked low-strength Chambolle-TV residual post-processing",
        "generator_executed_in_v20": False,
        "retained_generated_pixel_source": "v17 output retained through v18 and v19",
        "postprocess_only": True,
        "tv_weights": {"tv04": 0.0004, "tv08": 0.0008, "tv12": 0.0012},
        "tv_epsilon": 2e-5,
        "tv_max_iterations": 80,
        "soft_gate": "compact distance-transform smoothstep",
        "soft_gate_ramp_px": gate_ramp_px,
        "soft_gate_first_inside_contour_weight": 0.0,
        "soft_gate_explicit_zero_padding": True,
        "fixed_noise_metric_gate_threshold": 0.05,
        "local_fidelity_gate_threshold": 0.05,
        "local_fidelity_gate_erosion_px": 3,
        "local_ssim_window_px": 11,
        "local_ssim_gaussian_sigma_px": 1.5,
        "local_gradient_gaussian_sigma_px": 0.80,
        "estimated_target_noise_sigma_normalized": sigma_n,
        "delta_cap_normalized": cap,
        "writable_fraction_target": {
            "central": float(np.mean(central_gate > 0.05)),
            "endpoint_exterior_fog": float(np.mean(fog_gate > 0.05)),
            "flat_background": float(np.mean(flat_gate > 0.05)),
        },
        "hard_lock_fraction_target": float(np.mean(hard_lock)),
        "stack_lock_fraction_target": float(np.mean(stack_lock)),
        "central_boundary_lock_fraction_target": float(np.mean(central_boundary_lock)),
        "strong_edge_lock_fraction_target": float(np.mean(strong_edge_lock)),
        "gradient_lock_percentile": 98.5,
        "spatial_transform": None,
        "registration": None,
        "resize": None,
        "resampling": None,
        "analytic_redraw": False,
        "contrast_remapping": False,
        "sharpening": False,
    }
    return components, method


def apply_postprocess_candidate(
    current: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
    spec: dict,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Apply disjoint region residuals and hard-copy locked uint16 pixels."""
    central = float(spec["central_strength"])
    fog = float(spec["fog_strength"])
    flat = float(spec["flat_strength"])
    central_estimator = str(spec["central_estimator"])
    fog_estimator = str(spec["fog_estimator"])
    flat_estimator = str(spec["flat_estimator"])
    influence = np.clip(
        central * components["central_gate"]
        + fog * components["fog_gate"]
        + flat * components["flat_gate"],
        0.0,
        1.0,
    ).astype(np.float32)
    delta = (
        central
        * components["central_gate"]
        * components[f"central_{central_estimator}_delta"]
        + fog
        * components["fog_gate"]
        * components[f"fog_{fog_estimator}_delta"]
        + flat
        * components["flat_gate"]
        * components[f"flat_{flat_estimator}_delta"]
    )
    candidate_u16 = base.to_uint16(current)
    target_u16 = base.to_uint16(
        np.clip(components["crop"] + delta, 0.0, 1.0).astype(np.float32)
    )
    baseline_target_u16 = base.to_uint16(current[target.slices()])
    hard_lock = components["hard_lock"]
    target_u16[hard_lock] = baseline_target_u16[hard_lock]
    candidate_u16[target.slices()] = target_u16
    influence[hard_lock] = 0.0
    candidate = (candidate_u16.astype(np.float32) / 65535.0).astype(np.float32)
    return candidate, {
        "candidate_name": str(spec.get("name", "unnamed")),
        "central_strength": central,
        "central_estimator": central_estimator,
        "endpoint_exterior_fog_strength": fog,
        "endpoint_exterior_fog_estimator": fog_estimator,
        "flat_background_strength": flat,
        "flat_background_estimator": flat_estimator,
        "stack_strength": 0.0,
        "hard_lock_uint16_writeback": True,
        "outside_target_writeback": False,
        "spatial_transform": None,
        "registration": None,
        "resize": None,
        "resampling": None,
    }, influence


def haar_mad(image: np.ndarray, mask: np.ndarray) -> float:
    """Four-grid 2x2 diagonal Haar MAD on a fixed spatial mask."""
    estimates: list[float] = []
    selected = mask > 0.05
    for oy in (0, 1):
        for ox in (0, 1):
            h = (image.shape[0] - oy) // 2 * 2
            w = (image.shape[1] - ox) // 2 * 2
            if h < 2 or w < 2:
                continue
            values = image[oy : oy + h, ox : ox + w]
            local_mask = selected[oy : oy + h, ox : ox + w]
            a = values[0::2, 0::2]
            b = values[0::2, 1::2]
            c = values[1::2, 0::2]
            d = values[1::2, 1::2]
            valid = (
                local_mask[0::2, 0::2]
                & local_mask[0::2, 1::2]
                & local_mask[1::2, 0::2]
                & local_mask[1::2, 1::2]
            )
            if np.sum(valid) < 32:
                continue
            hh = (a - b - c + d)[valid] * 0.5
            center = float(np.median(hh))
            estimates.append(
                float(np.median(np.abs(hh - center)) / 0.6744897501960817)
            )
    return float(np.median(estimates)) if estimates else 0.0


def haar_mean_absolute(image: np.ndarray, mask: np.ndarray) -> float:
    """Four-grid mean absolute diagonal Haar energy on a fixed mask.

    Unlike the robust MAD estimator, this remains informative when more than
    half of a sparse CT background has exactly zero diagonal coefficients.
    """
    estimates: list[float] = []
    selected = mask > 0.05
    for oy in (0, 1):
        for ox in (0, 1):
            height = (image.shape[0] - oy) // 2 * 2
            width = (image.shape[1] - ox) // 2 * 2
            if height < 2 or width < 2:
                continue
            values = image[oy : oy + height, ox : ox + width]
            local_mask = selected[oy : oy + height, ox : ox + width]
            a = values[0::2, 0::2]
            b = values[0::2, 1::2]
            c = values[1::2, 0::2]
            d = values[1::2, 1::2]
            valid = (
                local_mask[0::2, 0::2]
                & local_mask[0::2, 1::2]
                & local_mask[1::2, 0::2]
                & local_mask[1::2, 1::2]
            )
            if np.sum(valid) < 32:
                continue
            coefficients = np.abs((a - b - c + d)[valid] * 0.5)
            estimates.append(float(np.mean(coefficients)))
    return float(np.median(estimates)) if estimates else 0.0


def haar_detail_mean_absolute(image: np.ndarray, mask: np.ndarray) -> float:
    """Four-grid mean absolute energy across horizontal/vertical/diagonal detail."""
    estimates: list[float] = []
    selected = mask > 0.05
    for oy in (0, 1):
        for ox in (0, 1):
            height = (image.shape[0] - oy) // 2 * 2
            width = (image.shape[1] - ox) // 2 * 2
            if height < 2 or width < 2:
                continue
            values = image[oy : oy + height, ox : ox + width]
            local_mask = selected[oy : oy + height, ox : ox + width]
            a = values[0::2, 0::2]
            b = values[0::2, 1::2]
            c = values[1::2, 0::2]
            d = values[1::2, 1::2]
            valid = (
                local_mask[0::2, 0::2]
                & local_mask[0::2, 1::2]
                & local_mask[1::2, 0::2]
                & local_mask[1::2, 1::2]
            )
            if np.sum(valid) < 32:
                continue
            horizontal = np.abs((a - b + c - d)[valid] * 0.5)
            vertical = np.abs((a + b - c - d)[valid] * 0.5)
            diagonal = np.abs((a - b - c + d)[valid] * 0.5)
            estimates.append(
                float(np.mean((horizontal + vertical + diagonal) / 3.0))
            )
    return float(np.median(estimates)) if estimates else 0.0


def fixed_noise_metrics(
    image: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Complementary Haar and fixed-Gaussian high-frequency proxies."""
    crop = image[target.slices()]
    residual = crop - gaussian_filter(crop, sigma=1.05)
    output: dict[str, dict[str, float]] = {}
    for name in ("central", "fog", "flat"):
        gate = components[f"{name}_gate"]
        selected = gate > 0.05
        output[name] = {
            "haar_diagonal_mad": haar_mad(crop, gate),
            "haar_diagonal_mean_absolute": haar_mean_absolute(crop, gate),
            "haar_detail_mean_absolute": haar_detail_mean_absolute(crop, gate),
            "fixed_high_frequency_rms": float(
                np.sqrt(np.mean(np.square(residual[selected])))
            )
            if np.any(selected)
            else 0.0,
            "sample_pixel_count": int(np.sum(selected)),
        }
    return output


def fixed_noise_reduction(before: dict, after: dict) -> dict:
    metric_names = {
        "haar_diagonal_mad": "haar_diagonal_mad_reduction",
        "haar_diagonal_mean_absolute": "haar_diagonal_mean_absolute_reduction",
        "haar_detail_mean_absolute": "haar_detail_mean_absolute_reduction",
        "fixed_high_frequency_rms": "fixed_high_frequency_reduction",
    }
    result: dict[str, dict[str, float]] = {}
    for region in ("central", "fog", "flat"):
        result[region] = {}
        for metric, key in metric_names.items():
            baseline = float(before[region][metric])
            value = float(after[region][metric])
            result[region][key] = (
                1.0 - value / baseline if baseline > 1e-10 else 0.0
            )
    return result


def masked_global_ssim(
    baseline: np.ndarray,
    candidate: np.ndarray,
    selected: np.ndarray,
) -> float:
    """SSIM formula evaluated only on a fixed set of writable pixels."""
    a = np.asarray(baseline[selected], dtype=np.float64)
    b = np.asarray(candidate[selected], dtype=np.float64)
    if a.size < 32:
        return 1.0 if np.array_equal(a, b) else 0.0
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.mean(np.square(a - mean_a)))
    var_b = float(np.mean(np.square(b - mean_b)))
    covariance = float(np.mean((a - mean_a) * (b - mean_b)))
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    if denominator <= 1e-18:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(
        ((2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2))
        / denominator
    )


def safe_flat_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with deterministic handling of constant vectors."""
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    if left.size < 32 or right.size != left.size:
        return 1.0 if np.array_equal(left, right) else 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def fixed_region_fidelity(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Audit each writable region without dilution by bit-exact lock pixels."""
    before = (
        base.to_uint16(baseline[target.slices()]).astype(np.float32) / 65535.0
    ).astype(np.float32)
    after = (
        base.to_uint16(candidate[target.slices()]).astype(np.float32) / 65535.0
    ).astype(np.float32)
    _, ssim_map = structural_similarity(
        before,
        after,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        win_size=11,
        use_sample_covariance=False,
        full=True,
    )
    before_gradient_image = gaussian_filter(before, sigma=0.80)
    after_gradient_image = gaussian_filter(after, sigma=0.80)
    before_gy, before_gx = np.gradient(before_gradient_image)
    after_gy, after_gx = np.gradient(after_gradient_image)
    before_gradient = np.hypot(before_gx, before_gy)
    after_gradient = np.hypot(after_gx, after_gy)
    delta_dn = np.abs(
        base.to_uint16(after).astype(np.int32)
        - base.to_uint16(before).astype(np.int32)
    )
    output: dict[str, dict[str, float]] = {}
    for name in ("central", "fog", "flat"):
        raw_selected = components[f"{name}_gate"] > 0.05
        selected = binary_erosion(raw_selected, iterations=3)
        if np.sum(selected) < 256:
            selected = raw_selected
        if not np.any(selected):
            output[name] = {
                "sample_pixel_count": 0,
                "ssim_map_mean": 1.0,
                "ssim_map_p01": 1.0,
                "masked_global_ssim": 1.0,
                "gradient_magnitude_correlation": 1.0,
                "gradient_rms_retention": 1.0,
                "gradient_relative_rmse": 0.0,
                "near_zero_gradient_fraction_before": 0.0,
                "near_zero_gradient_fraction_after": 0.0,
                "near_zero_gradient_fraction_increase": 0.0,
                "absolute_delta_p95_dn": 0.0,
                "absolute_delta_max_dn": 0.0,
            }
            continue
        gradient_before_values = before_gradient[selected]
        gradient_after_values = after_gradient[selected]
        gradient_before_rms = float(
            np.sqrt(np.mean(np.square(gradient_before_values)))
        )
        gradient_after_rms = float(
            np.sqrt(np.mean(np.square(gradient_after_values)))
        )
        threshold = max(
            float(np.percentile(gradient_before_values, 20.0)),
            1.0 / 65535.0,
        )
        before_flat_fraction = float(np.mean(gradient_before_values <= threshold))
        after_flat_fraction = float(np.mean(gradient_after_values <= threshold))
        local_delta = delta_dn[selected]
        output[name] = {
            "sample_pixel_count": int(np.sum(selected)),
            "ssim_map_mean": float(np.mean(ssim_map[selected])),
            "ssim_map_p01": float(np.percentile(ssim_map[selected], 1.0)),
            "masked_global_ssim": masked_global_ssim(before, after, selected),
            "gradient_magnitude_correlation": safe_flat_correlation(
                gradient_before_values, gradient_after_values
            ),
            "gradient_rms_retention": gradient_after_rms
            / max(gradient_before_rms, 1e-12),
            "gradient_relative_rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(gradient_after_values - gradient_before_values)
                    )
                )
                / max(gradient_before_rms, 1e-12)
            ),
            "near_zero_gradient_threshold": threshold,
            "near_zero_gradient_fraction_before": before_flat_fraction,
            "near_zero_gradient_fraction_after": after_flat_fraction,
            "near_zero_gradient_fraction_increase": (
                after_flat_fraction - before_flat_fraction
            ),
            "absolute_delta_p95_dn": float(np.percentile(local_delta, 95.0)),
            "absolute_delta_max_dn": float(np.max(local_delta)),
        }
    return output


def exact_structure_row_drift(audit: dict, baseline_audit: dict) -> dict:
    """Directly compare all released lamella/interlayer output geometry rows."""
    families = {
        "lamella": (
            "layer_comparison_rows",
            "layer_id",
            ("output_top_y_px", "output_bottom_y_px", "output_length_px", "output_width_px"),
        ),
        "interlayer": (
            "gap_comparison_rows",
            "gap_id",
            ("output_length_px", "output_width_px"),
        ),
    }
    output: dict[str, object] = {}
    all_maxima: list[float] = []
    for family, (rows_key, id_key, fields) in families.items():
        before_rows = {str(row[id_key]): row for row in baseline_audit[rows_key]}
        after_rows = {str(row[id_key]): row for row in audit[rows_key]}
        ids_equal = set(before_rows) == set(after_rows)
        field_maxima: dict[str, float] = {}
        for field in fields:
            errors: list[float] = []
            for identifier in sorted(set(before_rows) & set(after_rows)):
                before_value = before_rows[identifier].get(field)
                after_value = after_rows[identifier].get(field)
                if before_value is None or after_value is None:
                    if before_value != after_value:
                        errors.append(float("inf"))
                    continue
                errors.append(abs(float(after_value) - float(before_value)))
            maximum = float(np.max(errors)) if errors else 0.0
            field_maxima[f"{field}_max_abs_drift_px"] = maximum
            all_maxima.append(maximum)
        boolean_mismatch_count = sum(
            before_rows[identifier].get("dual_evidence_pass")
            != after_rows[identifier].get("dual_evidence_pass")
            for identifier in set(before_rows) & set(after_rows)
        )
        output[family] = {
            "baseline_row_count": len(before_rows),
            "candidate_row_count": len(after_rows),
            "identifiers_equal": ids_equal,
            "dual_evidence_mismatch_count": int(boolean_mismatch_count),
            **field_maxima,
        }
    output["maximum_geometry_absolute_drift_px"] = (
        float(np.max(all_maxima)) if all_maxima else 0.0
    )
    output["bit_exact_geometry_rows"] = bool(
        all(
            bool(output[family]["identifiers_equal"])
            and output[family]["baseline_row_count"]
            == output[family]["candidate_row_count"]
            and output[family]["dual_evidence_mismatch_count"] == 0
            for family in ("lamella", "interlayer")
        )
        and output["maximum_geometry_absolute_drift_px"] == 0.0
    )
    return output


def gate_boundary_audit(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Check seam contours and prove every changed pixel is writable."""
    before = baseline[target.slices()]
    after = candidate[target.slices()]
    before_u16 = base.to_uint16(before)
    after_u16 = base.to_uint16(after)
    changed = before_u16 != after_u16
    writable = components["writable_support"]
    zero_contour = components["gate_zero_contour"]
    seam = components["gate_seam_inner"]
    delta_dn = np.abs(after_u16.astype(np.int32) - before_u16.astype(np.int32))
    seam_values = delta_dn[seam]
    zero_contour_values = delta_dn[zero_contour]
    gates = [components[f"{name}_gate"] > 0.0 for name in ("central", "fog", "flat")]
    combined_gate = np.maximum.reduce(
        [components[f"{name}_gate"] for name in ("central", "fog", "flat")]
    )
    overlap = (gates[0].astype(np.uint8) + gates[1].astype(np.uint8) + gates[2].astype(np.uint8)) > 1
    return {
        "changed_outside_writable_support_uint16": int(np.sum(changed & (~writable))),
        "changed_inside_writable_support_uint16": int(np.sum(changed & writable)),
        "writable_support_pixel_count": int(np.sum(writable)),
        "gate_overlap_pixel_count": int(np.sum(overlap)),
        "writable_hard_lock_overlap_pixel_count": int(
            np.sum(writable & components["hard_lock"])
        ),
        "zero_weight_outer_contour_pixel_count": int(np.sum(zero_contour)),
        "zero_weight_outer_contour_gate_max": (
            float(np.max(combined_gate[zero_contour])) if np.any(zero_contour) else 0.0
        ),
        "zero_weight_outer_contour_absolute_delta_max_dn": (
            int(np.max(zero_contour_values)) if zero_contour_values.size else 0
        ),
        "seam_inner_pixel_count": int(np.sum(seam)),
        "seam_inner_gate_max": (
            float(np.max(combined_gate[seam])) if np.any(seam) else 0.0
        ),
        "seam_inner_absolute_delta_p95_dn": (
            float(np.percentile(seam_values, 95.0)) if seam_values.size else 0.0
        ),
        "seam_inner_absolute_delta_max_dn": (
            int(np.max(seam_values)) if seam_values.size else 0
        ),
    }


def subpixel_peak(values: np.ndarray, start: int, stop: int) -> float:
    start = max(1, int(start))
    stop = min(len(values) - 1, int(stop))
    if stop <= start:
        return float("nan")
    local = np.abs(values[start:stop])
    index = start + int(np.argmax(local))
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    y0, y1, y2 = (float(abs(values[index + offset])) for offset in (-1, 0, 1))
    denominator = y0 - 2.0 * y1 + y2
    offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (y0 - y2) / denominator
    return float(index + np.clip(offset, -0.5, 0.5))


def peak_characteristics(
    values: np.ndarray, start: int, stop: int
) -> tuple[float, float, float, float]:
    """Return subpixel position, peak strength, FWHM, and peak confidence."""
    start = max(1, int(start))
    stop = min(len(values) - 1, int(stop))
    if stop <= start:
        return float("nan"), float("nan"), float("nan"), float("nan")
    absolute = np.abs(np.asarray(values[start:stop], dtype=np.float64))
    local_index = int(np.argmax(absolute))
    index = start + local_index
    position = subpixel_peak(values, start, stop)
    peak = float(absolute[local_index])
    confidence = peak / max(float(np.median(absolute)), 1e-12)
    half = 0.5 * peak
    left_index = local_index
    while left_index > 0 and absolute[left_index] >= half:
        left_index -= 1
    right_index = local_index
    while right_index < len(absolute) - 1 and absolute[right_index] >= half:
        right_index += 1

    def crossing(lo: int, hi: int) -> float:
        y0 = float(absolute[lo])
        y1 = float(absolute[hi])
        if abs(y1 - y0) <= 1e-12:
            return float(lo)
        return float(lo + np.clip((half - y0) / (y1 - y0), 0.0, 1.0))

    left_crossing = (
        crossing(left_index, left_index + 1)
        if left_index < local_index
        else float(local_index) - 0.5
    )
    right_crossing = (
        crossing(right_index - 1, right_index)
        if right_index > local_index
        else float(local_index) + 0.5
    )
    width = max(0.0, right_crossing - left_crossing)
    return position, peak, width, confidence


def central_edge_tracks(image: np.ndarray, roi: base.Roi) -> dict[str, np.ndarray]:
    """Track configured central-ROI edge evidence on fixed rows/columns."""
    smooth = gaussian_filter(image, sigma=0.80)
    gy, gx = np.gradient(smooth)
    height = roi.y1 - roi.y0
    width = roi.x1 - roi.x0
    rows = range(roi.y0 + int(0.20 * height), roi.y1 - int(0.20 * height), 3)
    columns = range(roi.x0 + int(0.20 * width), roi.x1 - int(0.20 * width), 3)
    records: dict[str, list[float]] = {}
    for side in ("left", "right", "top", "bottom"):
        for metric in ("position", "strength", "transition_width", "confidence"):
            records[f"{side}_{metric}"] = []
    for y in rows:
        for side, values in (
            ("left", peak_characteristics(gx[y], roi.x0 - 16, roi.x0 + 17)),
            ("right", peak_characteristics(gx[y], roi.x1 - 17, roi.x1 + 16)),
        ):
            for metric, value in zip(
                ("position", "strength", "transition_width", "confidence"), values
            ):
                records[f"{side}_{metric}"].append(value)
    for x in columns:
        for side, values in (
            ("top", peak_characteristics(gy[:, x], roi.y0 - 16, roi.y0 + 17)),
            ("bottom", peak_characteristics(gy[:, x], roi.y1 - 17, roi.y1 + 16)),
        ):
            for metric, value in zip(
                ("position", "strength", "transition_width", "confidence"), values
            ):
                records[f"{side}_{metric}"].append(value)
    return {
        key: np.asarray(values, dtype=np.float64) for key, values in records.items()
    }


def central_contrast_metrics(image: np.ndarray, roi: base.Roi) -> dict[str, float]:
    """Robust configured-central-ROI core/exterior contrast and CNR."""
    height, width = image.shape
    yy, xx = np.indices(image.shape)
    rectangle = (yy >= roi.y0) & (yy < roi.y1) & (xx >= roi.x0) & (xx < roi.x1)
    inside = distance_transform_edt(rectangle)
    expanded = (
        (yy >= max(0, roi.y0 - 24))
        & (yy < min(height, roi.y1 + 24))
        & (xx >= max(0, roi.x0 - 24))
        & (xx < min(width, roi.x1 + 24))
    )
    near = (
        (yy >= max(0, roi.y0 - 8))
        & (yy < min(height, roi.y1 + 8))
        & (xx >= max(0, roi.x0 - 8))
        & (xx < min(width, roi.x1 + 8))
    )
    core = rectangle & (inside > 26.0)
    exterior = expanded & (~near)

    def robust(values: np.ndarray) -> tuple[float, float]:
        median = float(np.median(values))
        sigma = float(
            np.median(np.abs(values - median)) / 0.6744897501960817
        )
        return median, sigma

    core_median, core_sigma = robust(image[core])
    exterior_median, exterior_sigma = robust(image[exterior])
    contrast = abs(core_median - exterior_median)
    cnr = contrast / max(np.hypot(core_sigma, exterior_sigma), 1e-12)
    return {
        "core_median": core_median,
        "exterior_median": exterior_median,
        "core_robust_sigma": core_sigma,
        "exterior_robust_sigma": exterior_sigma,
        "absolute_contrast": contrast,
        "robust_cnr": cnr,
        "core_sample_pixel_count": int(np.sum(core)),
        "exterior_sample_pixel_count": int(np.sum(exterior)),
    }


def central_integrity_metrics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    roi: base.Roi,
    boundary_lock: np.ndarray,
    target: base.Roi,
) -> dict:
    before = central_edge_tracks(baseline, roi)
    after = central_edge_tracks(candidate, roi)
    edge_errors = np.concatenate(
        [
            np.abs(after[f"{name}_position"] - before[f"{name}_position"])
            for name in ("left", "right", "top", "bottom")
        ]
    )
    width_error = np.abs(
        (after["right_position"] - after["left_position"])
        - (before["right_position"] - before["left_position"])
    )
    height_error = np.abs(
        (after["bottom_position"] - after["top_position"])
        - (before["bottom_position"] - before["top_position"])
    )
    strength_retention = np.concatenate(
        [
            after[f"{name}_strength"]
            / np.maximum(before[f"{name}_strength"], 1e-12)
            for name in ("left", "right", "top", "bottom")
        ]
    )
    confidence_retention = np.concatenate(
        [
            after[f"{name}_confidence"]
            / np.maximum(before[f"{name}_confidence"], 1e-12)
            for name in ("left", "right", "top", "bottom")
        ]
    )
    transition_width_error = np.concatenate(
        [
            np.abs(
                after[f"{name}_transition_width"]
                - before[f"{name}_transition_width"]
            )
            for name in ("left", "right", "top", "bottom")
        ]
    )
    finite_evidence = (
        np.all(np.isfinite(edge_errors))
        and np.all(np.isfinite(strength_retention))
        and np.all(np.isfinite(confidence_retention))
        and np.all(np.isfinite(transition_width_error))
    )
    local_before = baseline[roi.slices()]
    local_after = candidate[roi.slices()]
    local_height, local_width = local_before.shape
    yy, xx = np.indices(local_before.shape)
    distance = np.minimum.reduce(
        (yy + 1, xx + 1, local_height - yy, local_width - xx)
    ).astype(np.float32)
    core = distance > 26.0
    # The boundary ring is in target-local coordinates and is copied exactly.
    changed_boundary = int(
        np.sum(
            base.to_uint16(candidate[target.slices()])[boundary_lock]
            != base.to_uint16(baseline[target.slices()])[boundary_lock]
        )
    )
    core_mean_drift_dn = float(
        abs(np.mean(local_after[core]) - np.mean(local_before[core])) * 65535.0
    )
    core_std_before = float(np.std(local_before[core]))
    core_std_after = float(np.std(local_after[core]))
    contrast_before = central_contrast_metrics(baseline, roi)
    contrast_after = central_contrast_metrics(candidate, roi)
    return {
        "configured_roi_edge_evidence_finite": bool(finite_evidence),
        "edge_position_absolute_drift_p95_px": float(np.percentile(edge_errors, 95.0)),
        "edge_position_absolute_drift_max_px": float(np.max(edge_errors)),
        "edge_peak_strength_retention_p05": float(
            np.percentile(strength_retention, 5.0)
        ),
        "edge_peak_confidence_retention_p05": float(
            np.percentile(confidence_retention, 5.0)
        ),
        "edge_transition_width_absolute_drift_p95_px": float(
            np.percentile(transition_width_error, 95.0)
        ),
        "edge_transition_width_absolute_drift_max_px": float(
            np.max(transition_width_error)
        ),
        "width_absolute_drift_p95_px": float(np.percentile(width_error, 95.0)),
        "width_absolute_drift_max_px": float(np.max(width_error)),
        "height_absolute_drift_p95_px": float(np.percentile(height_error, 95.0)),
        "height_absolute_drift_max_px": float(np.max(height_error)),
        "changed_boundary_lock_pixels_uint16": changed_boundary,
        "core_mean_absolute_drift_dn": core_mean_drift_dn,
        "core_standard_deviation_before": core_std_before,
        "core_standard_deviation_after": core_std_after,
        "core_standard_deviation_reduction": 1.0
        - core_std_after / max(core_std_before, 1e-12),
        "core_exterior_absolute_contrast_before": contrast_before[
            "absolute_contrast"
        ],
        "core_exterior_absolute_contrast_after": contrast_after[
            "absolute_contrast"
        ],
        "core_exterior_absolute_contrast_retention": float(
            contrast_after["absolute_contrast"]
            / max(contrast_before["absolute_contrast"], 1e-12)
        ),
        "core_exterior_robust_cnr_before": contrast_before["robust_cnr"],
        "core_exterior_robust_cnr_after": contrast_after["robust_cnr"],
        "core_exterior_robust_cnr_retention": float(
            contrast_after["robust_cnr"]
            / max(contrast_before["robust_cnr"], 1e-12)
        ),
        "core_contrast_sample_pixel_count": contrast_before[
            "core_sample_pixel_count"
        ],
        "exterior_contrast_sample_pixel_count": contrast_before[
            "exterior_sample_pixel_count"
        ],
    }


def strict_structure_invariance(
    audit: dict,
    baseline_audit: dict,
    local_width: dict,
) -> tuple[bool, dict, dict]:
    summary = v16.metric_summary(audit)
    baseline = v16.metric_summary(baseline_audit)
    topology_pass, topology_checks = v19.topology_invariant(audit, baseline_audit)
    direct_drift = exact_structure_row_drift(audit, baseline_audit)
    checks = {
        "endpoint_p95_le_0_010_px": summary["endpoint_shift_abs_p95_px"] <= 0.010,
        "length_p95_le_0_080_px": summary["length_delta_abs_p95_px"] <= 0.080,
        "lamella_width_p95_le_0_35_percent": summary["lamella_width_relative_error_p95"] <= 0.0035,
        "interlayer_width_p95_nonregression": summary["interlayer_width_relative_error_p95"] <= baseline["interlayer_width_relative_error_p95"] + 1e-6,
        "interlayer_length_p95_le_0_10_px": summary["interlayer_length_abs_error_p95_px"] <= 0.10,
        "lamella_dual_evidence_equal": summary["lamella_dual_evidence_pass_count"] == baseline["lamella_dual_evidence_pass_count"],
        "interlayer_dual_evidence_equal": summary["interlayer_dual_evidence_pass_count"] == baseline["interlayer_dual_evidence_pass_count"],
        "edge_clarity_nonregression": summary["edge_clarity"] >= baseline["edge_clarity"] - 1e-12,
        "low_frequency_correlation_ge_0_99999": summary["low_frequency_correlation"] >= 0.99999,
        "low_frequency_correlation_incremental_drop_le_5e_8": summary["low_frequency_correlation"] >= baseline["low_frequency_correlation"] - 5e-8,
        "mid_frequency_correlation_ge_0_99990": summary["mid_frequency_correlation"] >= 0.99990,
        "mid_frequency_correlation_incremental_drop_le_2e_5": summary["mid_frequency_correlation"] >= baseline["mid_frequency_correlation"] - 2e-5,
        "gradient_correlation_ge_0_99985": summary["gradient_magnitude_correlation"] >= 0.99985,
        "gradient_correlation_incremental_drop_le_3e_5": summary["gradient_magnitude_correlation"] >= baseline["gradient_magnitude_correlation"] - 3e-5,
        "axial_detail_median_ge_0_99997": summary["lamella_axial_detail_correlation_median"] >= 0.99997,
        "axial_detail_p10_ge_0_99994": summary["lamella_axial_detail_correlation_p10"] >= 0.99994,
        "row_width_median_zero": local_width["absolute_drift_median_px"] is not None and local_width["absolute_drift_median_px"] <= 1e-9,
        "row_width_p95_zero": local_width["absolute_drift_p95_px"] is not None and local_width["absolute_drift_p95_px"] <= 1e-9,
        "row_width_max_zero": local_width["absolute_drift_max_px"] is not None and local_width["absolute_drift_max_px"] <= 1e-9,
        "lamella_interlayer_output_geometry_rows_exact": direct_drift[
            "bit_exact_geometry_rows"
        ],
        "topology_invariant": topology_pass,
    }
    checks.update({f"topology_{name}": value for name, value in topology_checks.items()})
    return bool(all(checks.values())), checks, direct_drift


def evaluate_candidate(
    candidate: np.ndarray,
    *,
    source: np.ndarray,
    carrier: np.ndarray,
    current: np.ndarray,
    target: base.Roi,
    central_roi: base.Roi,
    left: base.Roi,
    right: base.Roi,
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
    layers: list[dict],
    components: dict[str, np.ndarray],
    v19_noise_components: dict[str, np.ndarray],
    baseline_audit: dict,
    baseline_noise: dict,
    baseline_v19_noise: dict,
) -> dict:
    audit = project.audit_projection(
        source,
        carrier,
        current,
        candidate,
        layers,
        left,
        right,
        top_range,
        bottom_range,
    )
    local_width, local_rows, _ = v19.local_width_drift(
        current, candidate, layers, row_step=1
    )
    structure_pass, structure_checks, direct_structure_drift = strict_structure_invariance(
        audit, baseline_audit, local_width
    )
    baseline_u16 = base.to_uint16(current[target.slices()])
    candidate_u16 = base.to_uint16(candidate[target.slices()])
    lock_counts = {
        name: int(np.sum(candidate_u16[components[name]] != baseline_u16[components[name]]))
        for name in (
            "hard_lock",
            "stack_lock",
            "central_boundary_lock",
            "strong_edge_lock",
            "operator_support",
        )
    }
    outside = np.ones(current.shape, dtype=bool)
    outside[target.slices()] = False
    changed_outside = int(
        np.sum(base.to_uint16(candidate)[outside] != base.to_uint16(current)[outside])
    )
    target_ssim = float(
        structural_similarity(
            current[target.slices()], candidate[target.slices()], data_range=1.0
        )
    )
    noise_after = fixed_noise_metrics(candidate, target, components)
    reduction = fixed_noise_reduction(baseline_noise, noise_after)
    local_fidelity = fixed_region_fidelity(current, candidate, target, components)
    gate_audit = gate_boundary_audit(current, candidate, target, components)
    v19_noise_after = v19.noise_metrics(candidate, target, v19_noise_components)
    v19_reduction = v19.reduction_metrics(baseline_v19_noise, v19_noise_after)
    central_integrity = central_integrity_metrics(
        current,
        candidate,
        central_roi,
        components["central_boundary_lock"],
        target,
    )
    before_u16 = base.to_uint16(current)
    after_u16 = base.to_uint16(candidate)
    new_low_clipping = int(np.sum((after_u16 == 0) & (before_u16 != 0)))
    new_high_clipping = int(np.sum((after_u16 == 65535) & (before_u16 != 65535)))
    delta_dn = np.abs(after_u16.astype(np.int32) - before_u16.astype(np.int32))
    changed_delta = delta_dn[delta_dn > 0]
    delta_summary = {
        "changed_pixel_count": int(changed_delta.size),
        "absolute_delta_p50_dn": float(np.percentile(changed_delta, 50.0)) if changed_delta.size else 0.0,
        "absolute_delta_p95_dn": float(np.percentile(changed_delta, 95.0)) if changed_delta.size else 0.0,
        "absolute_delta_p99_dn": float(np.percentile(changed_delta, 99.0)) if changed_delta.size else 0.0,
        "absolute_delta_max_dn": int(np.max(changed_delta)) if changed_delta.size else 0,
        "new_zero_clipping_pixels": new_low_clipping,
        "new_saturation_clipping_pixels": new_high_clipping,
    }
    hard_checks = {
        "all_locked_pixels_bit_exact": all(value == 0 for value in lock_counts.values()),
        "outside_target_bit_exact": changed_outside == 0,
        "changes_only_inside_writable_support": gate_audit[
            "changed_outside_writable_support_uint16"
        ] == 0,
        "writable_support_disjoint_from_hard_lock": gate_audit[
            "writable_hard_lock_overlap_pixel_count"
        ] == 0,
        "writable_region_gates_disjoint": gate_audit["gate_overlap_pixel_count"] == 0,
        "zero_weight_gate_contour_bit_exact": gate_audit[
            "zero_weight_outer_contour_absolute_delta_max_dn"
        ] == 0,
        "soft_gate_inner_seam_p95_le_2_dn": gate_audit[
            "seam_inner_absolute_delta_p95_dn"
        ] <= 2.0,
        "soft_gate_inner_seam_max_le_8_dn": gate_audit[
            "seam_inner_absolute_delta_max_dn"
        ] <= 8,
        "configured_central_roi_edge_evidence_finite": central_integrity[
            "configured_roi_edge_evidence_finite"
        ],
        "central_edge_position_p95_le_0_010_px": central_integrity["edge_position_absolute_drift_p95_px"] <= 0.010,
        "central_edge_position_max_le_0_030_px": central_integrity["edge_position_absolute_drift_max_px"] <= 0.030,
        "central_edge_peak_strength_retention_p05_ge_0_995": central_integrity[
            "edge_peak_strength_retention_p05"
        ] >= 0.995,
        "central_edge_peak_confidence_retention_p05_ge_0_99": central_integrity[
            "edge_peak_confidence_retention_p05"
        ] >= 0.99,
        "central_edge_transition_width_p95_le_0_030_px": central_integrity[
            "edge_transition_width_absolute_drift_p95_px"
        ] <= 0.030,
        "central_edge_transition_width_max_le_0_080_px": central_integrity[
            "edge_transition_width_absolute_drift_max_px"
        ] <= 0.080,
        "central_width_p95_le_0_020_px": central_integrity["width_absolute_drift_p95_px"] <= 0.020,
        "central_width_max_le_0_050_px": central_integrity["width_absolute_drift_max_px"] <= 0.050,
        "central_height_p95_le_0_020_px": central_integrity["height_absolute_drift_p95_px"] <= 0.020,
        "central_height_max_le_0_050_px": central_integrity["height_absolute_drift_max_px"] <= 0.050,
        "central_boundary_bit_exact": central_integrity["changed_boundary_lock_pixels_uint16"] == 0,
        "central_core_mean_drift_le_1_dn": central_integrity["core_mean_absolute_drift_dn"] <= 1.0,
        "central_core_exterior_contrast_retention_ge_0_999": central_integrity[
            "core_exterior_absolute_contrast_retention"
        ] >= 0.999,
        "central_core_exterior_cnr_retention_ge_0_995": central_integrity[
            "core_exterior_robust_cnr_retention"
        ] >= 0.995,
        "ssim_vs_v19_ge_0_99999": target_ssim >= 0.99999,
        "no_new_clipping": new_low_clipping == 0 and new_high_clipping == 0,
        "absolute_delta_p99_le_72_dn": delta_summary["absolute_delta_p99_dn"] <= 72,
        "absolute_delta_max_le_128_dn": delta_summary["absolute_delta_max_dn"] <= 128,
    }
    local_fidelity_checks: dict[str, bool] = {}
    for region in ("central", "fog", "flat"):
        metrics = local_fidelity[region]
        local_fidelity_checks.update(
            {
                f"{region}_writable_masked_global_ssim_ge_0_9995": metrics[
                    "masked_global_ssim"
                ] >= 0.9995,
                f"{region}_near_zero_gradient_fraction_increase_le_0_02": metrics[
                    "near_zero_gradient_fraction_increase"
                ] <= 0.02,
            }
        )
    central_fidelity = local_fidelity["central"]
    local_fidelity_checks.update(
        {
            "central_writable_ssim_map_mean_ge_0_99990": central_fidelity[
                "ssim_map_mean"
            ] >= 0.99990,
            "central_writable_ssim_map_p01_ge_0_99950": central_fidelity[
                "ssim_map_p01"
            ] >= 0.99950,
            "central_writable_gradient_correlation_ge_0_99980": central_fidelity[
                "gradient_magnitude_correlation"
            ] >= 0.99980,
            "central_writable_gradient_rms_retention_ge_0_97": central_fidelity[
                "gradient_rms_retention"
            ] >= 0.97,
            "central_writable_gradient_rms_retention_le_1_005": central_fidelity[
                "gradient_rms_retention"
            ] <= 1.005,
            "central_writable_gradient_relative_rmse_le_0_025": central_fidelity[
                "gradient_relative_rmse"
            ] <= 0.025,
        }
    )
    for region in ("fog", "flat"):
        metrics = local_fidelity[region]
        local_fidelity_checks.update(
            {
                f"{region}_writable_ssim_map_mean_ge_0_999": metrics[
                    "ssim_map_mean"
                ] >= 0.999,
                f"{region}_writable_ssim_map_p01_ge_0_995": metrics[
                    "ssim_map_p01"
                ] >= 0.995,
                f"{region}_writable_gradient_correlation_ge_0_95": metrics[
                    "gradient_magnitude_correlation"
                ] >= 0.95,
                f"{region}_writable_gradient_rms_retention_le_1_01": metrics[
                    "gradient_rms_retention"
                ] <= 1.01,
                f"{region}_writable_gradient_relative_rmse_le_0_20": metrics[
                    "gradient_relative_rmse"
                ] <= 0.20,
            }
        )
    noise_checks = {
        f"{region}_{metric}_nonregression": value >= -0.002
        for region, metrics in reduction.items()
        for metric, value in metrics.items()
    }
    measured_regions = [
        region
        for region in ("central", "fog", "flat")
        if baseline_noise[region]["sample_pixel_count"] >= 32
        and baseline_noise[region]["fixed_high_frequency_rms"] > 1e-10
    ]
    minimum_hf = min(
        reduction[region]["fixed_high_frequency_reduction"]
        for region in measured_regions
    ) if measured_regions else 0.0
    mean_haar = float(
        np.mean(
            [
                reduction[region]["haar_detail_mean_absolute_reduction"]
                for region in ("central", "fog", "flat")
            ]
        )
    )
    v19_proxy_checks = {
        "lamella_proxy_bit_exact": abs(v19_reduction["lamella_axial_reduction"]) <= 1e-12,
        "interlayer_proxy_bit_exact": abs(v19_reduction["interlayer_high_frequency_reduction"]) <= 1e-12,
        "central_proxy_reduction_ge_0_5_percent": v19_reduction["central_high_frequency_reduction"] >= 0.005,
        "fog_full_region_proxy_nonregression": v19_reduction["endpoint_exterior_high_frequency_reduction"] >= -1e-4,
        "flat_full_region_proxy_reduction_ge_0_4_percent": v19_reduction[
            "flat_background_high_frequency_reduction"
        ] >= 0.004,
        "central_writable_zone_fixed_hf_reduction_ge_0_5_percent": reduction["central"]["fixed_high_frequency_reduction"] >= 0.005,
        "fog_writable_zone_fixed_hf_reduction_ge_0_2_percent": reduction["fog"][
            "fixed_high_frequency_reduction"
        ] >= 0.002,
        "flat_writable_zone_fixed_hf_reduction_ge_2_percent": reduction["flat"]["fixed_high_frequency_reduction"] >= 0.020,
        "central_haar_detail_mean_absolute_reduction_ge_0_2_percent": reduction[
            "central"
        ]["haar_detail_mean_absolute_reduction"] >= 0.002,
        "fog_haar_detail_mean_absolute_reduction_ge_0_2_percent": reduction["fog"][
            "haar_detail_mean_absolute_reduction"
        ] >= 0.002,
        "flat_haar_detail_mean_absolute_reduction_ge_0_2_percent": reduction["flat"][
            "haar_detail_mean_absolute_reduction"
        ] >= 0.002,
    }
    all_checks = {
        **structure_checks,
        **hard_checks,
        **local_fidelity_checks,
        **noise_checks,
    }
    all_checks.update(v19_proxy_checks)
    passed = bool(
        structure_pass
        and all(hard_checks.values())
        and all(local_fidelity_checks.values())
        and all(noise_checks.values())
        and all(v19_proxy_checks.values())
    )
    safe_proxy_minimum = min(
        v19_reduction["central_high_frequency_reduction"],
        v19_reduction["endpoint_exterior_high_frequency_reduction"],
        v19_reduction["flat_background_high_frequency_reduction"],
    )
    return {
        "audit": audit,
        "local_width_rows": local_rows,
        "structure_geometry": v16.metric_summary(audit),
        "local_row_width_invariance": {
            key: value for key, value in local_width.items() if key != "per_layer"
        },
        "direct_structure_geometry_drift_vs_v19": direct_structure_drift,
        "lock_changed_pixels_uint16": lock_counts,
        "changed_outside_target_uint16": changed_outside,
        "central_integrity": central_integrity,
        "ssim_vs_v19": target_ssim,
        "writable_region_fidelity": local_fidelity,
        "gate_boundary_and_write_audit": gate_audit,
        "noise_after": noise_after,
        "noise_reduction_vs_v19": reduction,
        "v19_fixed_operator_noise_after": v19_noise_after,
        "v19_fixed_operator_noise_reduction": v19_reduction,
        "pixel_delta": delta_summary,
        "guardrail_checks": all_checks,
        "guardrail_pass": passed,
        "score": float(safe_proxy_minimum + 0.20 * minimum_hf + 0.05 * mean_haar),
    }


def save_mask_audit(
    path: Path,
    current: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
) -> None:
    crop = current[target.slices()]
    lo, hi = (float(value) for value in np.percentile(crop, (0.5, 99.7)))
    gray = np.rint(np.clip((crop - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
    fields = (
        (components["hard_lock"].astype(np.float32), "all uint16 hard locks"),
        (components["stack_lock"].astype(np.float32), "complete lamella / interlayer lock"),
        (components["central_boundary_lock"].astype(np.float32), "configured central ROI border lock"),
        (components["central_gate"], "central writable core"),
        (components["fog_gate"], "endpoint-exterior fog gate"),
        (components["flat_gate"], "low-structure background gate"),
    )
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    for field, title in fields:
        rgb = np.repeat(gray[..., None], 3, axis=2)
        rgb[..., 1] = np.maximum(
            rgb[..., 1], np.rint(220.0 * np.clip(field, 0.0, 1.0)).astype(np.uint8)
        )
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
    before: np.ndarray,
    after: np.ndarray,
    target: base.Roi,
) -> None:
    crop_before = before[target.slices()]
    crop_after = after[target.slices()]
    lo, hi = (float(value) for value in np.percentile(crop_before, (0.5, 99.7)))
    delta = np.abs(crop_after - crop_before)
    items = (
        (crop_before, "v19 measurement-anchored input"),
        (crop_after, "v20 measurement-safe postprocess"),
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
    carrier, carrier_info = base.load_gray(args.carrier)
    current, input_info = base.load_gray(args.input)
    if source.shape != carrier.shape or source.shape != current.shape:
        raise ValueError(
            "source, carrier, and v19 input must have identical dimensions: "
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
    operator_support, operator_info = v19.measurement_operator_anchor(
        source.shape, target, layers, args.top_range, args.bottom_range
    )
    v19_components, _ = v19.build_multiregion_components(
        current,
        target,
        central,
        left,
        right,
        maps,
        envelope_masks,
        operator_support,
    )
    components, method = build_v20_components(
        current,
        target,
        central,
        maps,
        envelope_masks,
        v19_components,
        operator_support,
    )

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
    baseline_noise = fixed_noise_metrics(current, target, components)
    baseline_v19_noise = v19.noise_metrics(current, target, v19_components)
    print(json.dumps({"v20_method": method}, ensure_ascii=False), flush=True)

    parameter_grid = (
        {
            "name": "identity",
            "central_estimator": "tv04",
            "central_strength": 0.00,
            "fog_estimator": "tv04",
            "fog_strength": 0.00,
            "flat_estimator": "tv04",
            "flat_strength": 0.00,
        },
        {
            "name": "tv_conservative",
            "central_estimator": "tv04",
            "central_strength": 0.35,
            "fog_estimator": "tv04",
            "fog_strength": 0.45,
            "flat_estimator": "tv04",
            "flat_strength": 0.45,
        },
        {
            "name": "tv_recommended",
            "central_estimator": "tv08",
            "central_strength": 0.35,
            "fog_estimator": "tv04",
            "fog_strength": 0.55,
            "flat_estimator": "tv08",
            "flat_strength": 0.55,
        },
        {
            "name": "tv_stronger_fog",
            "central_estimator": "tv12",
            "central_strength": 0.35,
            "fog_estimator": "tv04",
            "fog_strength": 0.75,
            "flat_estimator": "tv08",
            "flat_strength": 0.55,
        },
        {
            "name": "tv_stronger_all",
            "central_estimator": "tv12",
            "central_strength": 0.40,
            "fog_estimator": "tv08",
            "fog_strength": 0.45,
            "flat_estimator": "tv12",
            "flat_strength": 0.45,
        },
        {
            "name": "tv_balanced_strong_post",
            "central_estimator": "tv12",
            "central_strength": 0.40,
            "fog_estimator": "tv12",
            "fog_strength": 0.55,
            "flat_estimator": "tv12",
            "flat_strength": 0.50,
        },
        {
            "name": "tv_aggressive_audit_only",
            "central_estimator": "tv12",
            "central_strength": 0.55,
            "fog_estimator": "tv12",
            "fog_strength": 0.60,
            "flat_estimator": "tv12",
            "flat_strength": 0.65,
        },
    )
    candidates: list[dict] = []
    selected: dict | None = None
    for spec in parameter_grid:
        candidate, operation, influence = apply_postprocess_candidate(
            current, target, components, spec
        )
        evaluation = evaluate_candidate(
            candidate,
            source=source,
            carrier=carrier,
            current=current,
            target=target,
            central_roi=central,
            left=left,
            right=right,
            top_range=args.top_range,
            bottom_range=args.bottom_range,
            layers=layers,
            components=components,
            v19_noise_components=v19_components,
            baseline_audit=baseline_audit,
            baseline_noise=baseline_noise,
            baseline_v19_noise=baseline_v19_noise,
        )
        item = {**operation, **{key: value for key, value in evaluation.items() if key not in {"audit", "local_width_rows"}}}
        candidates.append(item)
        print(json.dumps({"candidate": item}, ensure_ascii=False), flush=True)
        if evaluation["guardrail_pass"] and (
            selected is None or evaluation["score"] > selected["summary"]["score"]
        ):
            selected = {
                "summary": item,
                "image": candidate,
                "audit": evaluation["audit"],
                "local_width_rows": evaluation["local_width_rows"],
                "influence": influence,
            }

    if selected is None:
        (args.outdir / "FAILED_candidate_grid_v20.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("No v20 candidate passed the v19 measurement guardrails")

    final = selected["image"]
    output_tif = args.outdir / "MEASUREMENT_CANDIDATE_v20_measurement_safe_postprocessed_16bit.tif"
    imwrite(
        output_tif,
        base.to_uint16(final),
        photometric="minisblack",
        description=(
            "MEASUREMENT_CANDIDATE v20: v19 pixels with measurement-locked "
            "low-strength Chambolle-TV residual post-processing in non-measurement zones."
        ),
    )
    reloaded_u16 = np.asarray(imread(output_tif))
    expected_u16 = base.to_uint16(final)
    if reloaded_u16.shape != source.shape or reloaded_u16.dtype != np.uint16:
        raise RuntimeError(
            f"release TIFF mismatch: shape={reloaded_u16.shape}, dtype={reloaded_u16.dtype}"
        )
    if not np.array_equal(reloaded_u16, expected_u16):
        raise RuntimeError("release TIFF pixels changed during write/read round trip")
    reloaded = (reloaded_u16.astype(np.float32) / 65535.0).astype(np.float32)
    release_evaluation = evaluate_candidate(
        reloaded,
        source=source,
        carrier=carrier,
        current=current,
        target=target,
        central_roi=central,
        left=left,
        right=right,
        top_range=args.top_range,
        bottom_range=args.bottom_range,
        layers=layers,
        components=components,
        v19_noise_components=v19_components,
        baseline_audit=baseline_audit,
        baseline_noise=baseline_noise,
        baseline_v19_noise=baseline_v19_noise,
    )
    if not release_evaluation["guardrail_pass"]:
        raise RuntimeError(
            "Written uint16 v20 TIFF failed the post-write release audit: "
            + json.dumps(release_evaluation["guardrail_checks"], ensure_ascii=False)
        )

    audit = selected["audit"]
    audit_payload = dict(audit)
    condition_rows = audit_payload.pop("condition_boundary_rows")
    audit_payload.pop("raw_boundary_rows")
    layer_rows = audit_payload.pop("layer_comparison_rows")
    gap_rows = audit_payload.pop("gap_comparison_rows")
    detail_rows = audit_payload.pop("structure_detail_rows")
    del condition_rows

    base.save_preview(
        args.outdir / "MEASUREMENT_CANDIDATE_v20_measurement_safe_postprocessed.png",
        final,
    )
    save_comparison(
        args.outdir / "MEASUREMENT_CANDIDATE_v20_comparison.png",
        current,
        final,
        target,
    )
    save_mask_audit(
        args.outdir / "AUDIT_v20_measurement_safe_masks.png",
        current,
        target,
        components,
    )
    shape.write_csv(args.outdir / "lamella_v20_comparison.csv", layer_rows)
    shape.write_csv(args.outdir / "interlayer_v20_comparison.csv", gap_rows)
    shape.write_csv(args.outdir / "structure_detail_v20.csv", detail_rows)
    shape.write_csv(
        args.outdir / "local_row_width_v20_comparison.csv",
        selected["local_width_rows"],
    )

    changed = base.to_uint16(final) != base.to_uint16(current)
    payload = {
        "completed": True,
        "release": "v20-measurement-safe-tv-postprocess",
        "status": "MEASUREMENT_CANDIDATE; calibrated multi-image validation required",
        "source": source_info,
        "carrier": carrier_info,
        "input_v19": input_info,
        "dimensions": {"width": source.shape[1], "height": source.shape[0]},
        "design": {
            "v17_generative_enhancement_retained": True,
            "v18_finite_width_edge_cleanup_retained": True,
            "v19_multiregion_denoising_retained": True,
            "generator_executed_in_v20": False,
            "v20_operation": "post-processing only",
            "complete_lamella_interlayer_stack_uint16_locked": True,
            "configured_central_roi_border_ring_uint16_locked": True,
            "spatial_transform": None,
            "registration": None,
            "resize": None,
            "resampling": None,
            "analytic_renderer_pixel_writeback": False,
        },
        "method": method,
        "endpoint_envelope": envelope_info,
        "measurement_operator_support": operator_info,
        "baseline_v19": {
            "structure_geometry": baseline_summary,
            "fixed_noise_metrics": baseline_noise,
            "v19_fixed_operator_noise_metrics": baseline_v19_noise,
        },
        "selected": selected["summary"],
        "candidate_grid": candidates,
        "audit": audit_payload,
        "post_write_uint16_release_audit": {
            "passed": release_evaluation["guardrail_pass"],
            **{key: value for key, value in release_evaluation.items() if key not in {"audit", "local_width_rows"}},
            "shape": list(reloaded_u16.shape),
            "dtype": str(reloaded_u16.dtype),
            "pixel_round_trip_exact": True,
        },
        "pixel_change": {
            "changed_pixels": int(np.sum(changed)),
            "changed_fraction_full_image": float(np.mean(changed)),
            "changed_outside_target_roi": int(np.sum(changed) - np.sum(changed[target.slices()])),
        },
        "measurement_warning": (
            "This TIFF retains generated v17 pixels. V20 itself is post-processing, "
            "not a new generator. The v16 carrier and exported numerical audits "
            "remain authoritative until calibrated phantom and multi-image validation."
        ),
        "metric_interpretation": (
            "Haar-detail and fixed-Gaussian high-frequency reductions are complementary "
            "no-reference residual proxies relative to v19, not error against a "
            "noise-free ground truth."
        ),
    }
    metrics_path = args.outdir / "measurement_safe_postprocess_v20_metrics.json"
    metrics_path.write_text(
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
