#!/usr/bin/env python3
"""Project generative appearance onto measured lamella and interlayer geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tifffile import imwrite

import generative_shape_constraint as shape
import length_optimize as length
import pipeline as base
import quality_optimize as quality


def resized_gray(path: Path, output_shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (output_shape[1], output_shape[0]):
        image = image.resize((output_shape[1], output_shape[0]), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def expanded_support(layers: list[dict], image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for row in layers:
        expanded = dict(row)
        expanded["top_y_px"] = max(0.0, row["top_y_px"] - 12.0)
        expanded["bottom_y_px"] = min(float(h - 1), row["bottom_y_px"] + 12.0)
        expanded["width_median_px"] = 1.10 * row["pitch_px"]
        draw.polygon(shape.layer_polygon(expanded, image_shape), fill=255)
    return gaussian_filter(np.asarray(canvas, dtype=np.float32) / 255.0, sigma=2.0)


def robust_layer_amplitudes(
    generated: np.ndarray,
    background: np.ndarray,
    layers: list[dict],
) -> dict[str, float]:
    measured: dict[str, float] = {}
    by_side: dict[str, list[float]] = {"left": [], "right": []}
    for row in layers:
        margin = max(12.0, 0.12 * row["length_px"])
        ys = np.linspace(row["top_y_px"] + margin, row["bottom_y_px"] - margin, 25)
        values = []
        for yf in ys:
            y = int(np.clip(round(yf), 0, generated.shape[0] - 1))
            x = int(np.clip(round(float(row["path_x"][y])), 0, generated.shape[1] - 1))
            values.append(float(generated[y, x] - background[y, x]))
        amplitude = max(float(np.percentile(values, 65.0)), 0.0)
        measured[row["layer_id"]] = amplitude
        if amplitude > 0:
            by_side[row["side"]].append(amplitude)
    side_defaults = {
        side: max(float(np.median(values)) if values else 0.16, 0.08)
        for side, values in by_side.items()
    }
    return {
        row["layer_id"]: float(np.clip(
            measured[row["layer_id"]],
            0.95 * side_defaults[row["side"]],
            1.35 * side_defaults[row["side"]],
        ))
        for row in layers
    }


def robust_intensity_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map source contrast into target display range without spatial warping."""
    source_lo, source_hi = np.percentile(source, (1.0, 99.5))
    target_lo, target_hi = np.percentile(target, (1.0, 99.5))
    source_span = max(float(source_hi - source_lo), 1e-6)
    matched = (source - float(source_lo)) / source_span
    matched = matched * float(target_hi - target_lo) + float(target_lo)
    return np.clip(matched, 0.0, 1.0).astype(np.float32)


def foreground_structure_region(
    layers: list[dict], image_shape: tuple[int, int]
) -> np.ndarray:
    """Feathered ROI spanning both lamella groups and the central bridge."""
    h, w = image_shape
    min_x = min(float(np.min(row["path_x"])) for row in layers)
    max_x = max(float(np.max(row["path_x"])) for row in layers)
    min_y = min(float(row["top_y_px"]) for row in layers)
    max_y = max(float(row["bottom_y_px"]) for row in layers)
    pitch = float(np.median([row["pitch_px"] for row in layers]))
    x0 = max(0, int(math.floor(min_x - 1.5 * pitch)))
    x1 = min(w, int(math.ceil(max_x + 1.5 * pitch)))
    y0 = max(0, int(math.floor(min_y - 2.0 * pitch)))
    y1 = min(h, int(math.ceil(max_y + 2.0 * pitch)))
    mask = np.zeros(image_shape, dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return gaussian_filter(mask, sigma=8.0)


def foreground_structure_core(
    layers: list[dict], image_shape: tuple[int, int]
) -> np.ndarray:
    """Binary component ROI in which generated pixels are forbidden for v15."""
    h, w = image_shape
    min_x = min(float(np.min(row["path_x"])) for row in layers)
    max_x = max(float(np.max(row["path_x"])) for row in layers)
    min_y = min(float(row["top_y_px"]) for row in layers)
    max_y = max(float(row["bottom_y_px"]) for row in layers)
    pitch = float(np.median([row["pitch_px"] for row in layers]))
    x0 = max(0, int(math.floor(min_x - 1.5 * pitch)))
    x1 = min(w, int(math.ceil(max_x + 1.5 * pitch)))
    y0 = max(0, int(math.floor(min_y - 2.0 * pitch)))
    y1 = min(h, int(math.ceil(max_y + 2.0 * pitch)))
    core = np.zeros(image_shape, dtype=bool)
    core[y0:y1, x0:x1] = True
    return core


def structure_carrier_projection(
    generated: np.ndarray,
    detail_guide: np.ndarray,
    layers: list[dict],
    smoothing_sigma: float = 0.0,
    residual_floor: float = 1.0,
    transverse_unsharp_amount: float = 0.0,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Keep a guide-derived structural carrier inside the measurement ROI.

    Every measurement-output pixel is derived from the same-coordinate blind guide;
    the generator is retained only as a separately emitted visual companion. A weak,
    optional edge-adaptive residual shrinkage can remove residual speckle while
    retaining the guide residual at layer and endpoint gradients. No resampling,
    warping, analytic stripe replacement, or generated-pixel blending occurs.
    """
    if detail_guide.shape != generated.shape:
        raise ValueError(
            f"Detail-guide shape {detail_guide.shape} does not match generated shape "
            f"{generated.shape}"
        )
    smoothing_sigma = float(np.clip(smoothing_sigma, 0.0, 2.0))
    residual_floor = float(np.clip(residual_floor, 0.0, 1.0))
    transverse_unsharp_amount = float(np.clip(transverse_unsharp_amount, 0.0, 1.0))
    core = foreground_structure_core(layers, generated.shape)

    guide_smooth = gaussian_filter(detail_guide, sigma=smoothing_sigma)
    gradient_source = gaussian_filter(detail_guide, sigma=0.65)
    gradient = np.hypot(*np.gradient(gradient_source))
    gradient_values = gradient[core]
    gradient_lo, gradient_hi = np.percentile(gradient_values, (55.0, 92.0))
    edge_protection = np.clip(
        (gradient - float(gradient_lo)) /
        max(float(gradient_hi - gradient_lo), 1e-8),
        0.0,
        1.0,
    )
    edge_protection = gaussian_filter(edge_protection, sigma=0.45)
    residual_keep = residual_floor + (1.0 - residual_floor) * edge_protection
    carrier = guide_smooth + residual_keep * (detail_guide - guide_smooth)
    # Recover transverse layer-edge slope only where the guide itself contains
    # a strong edge. Flat-region residuals are not sharpened, so speckle and fog
    # do not receive the same gain as lamella boundaries.
    transverse_low = gaussian_filter(carrier, sigma=(0.0, 0.85))
    carrier = carrier + (
        transverse_unsharp_amount * edge_protection * (carrier - transverse_low)
    )
    # Measurement pixels retain the guide's native intensity scale. Even a
    # clipped global display-range mapping can move an FWHM crossing at a bright
    # endpoint, so display matching is deliberately excluded from the core.
    # A quantitative image must not contain a hidden seam between synthetic and
    # observed content. The complete v15 measurement output therefore uses the
    # guide-derived carrier. The generator remains an audited companion input
    # and is emitted separately by the v15 dual-output pipeline.
    projected = carrier.copy()
    method = {
        "method": "v15: guide-derived structure carrier with hard data consistency",
        "measurement_core_pixel_source": "same-coordinate blind-denoised guide only",
        "generated_pixel_role": "not used in the measurement output",
        "generated_pixel_weight_inside_measurement_core": 0.0,
        "generated_pixel_weight_entire_measurement_output": 0.0,
        "spatial_resampling_inside_measurement_core": False,
        "spatial_warp_inside_measurement_core": False,
        "analytic_lamella_replacement_inside_measurement_core": False,
        "intensity_remapping_inside_measurement_core": False,
        "edge_adaptive_residual_smoothing_sigma_px": smoothing_sigma,
        "minimum_guide_residual_keep": residual_floor,
        "maximum_guide_residual_keep": 1.0,
        "guide_edge_gated_transverse_unsharp_amount": transverse_unsharp_amount,
        "guide_edge_gated_transverse_unsharp_sigma_px": 0.85,
        "synthetic_observed_blending_seam": False,
        "source_pixel_writeback": False,
        "metrology_role": (
            "measurement-assist image; exported raw/guide constraints remain authoritative"
        ),
    }
    return np.clip(projected, 0.0, 1.0).astype(np.float32), method, core


def layer_axial_contrast_profile(image: np.ndarray, row: dict) -> np.ndarray:
    """Sample one layer against its two adjacent gaps in the same coordinates."""
    h = image.shape[0]
    profile = np.zeros(h, dtype=np.float32)
    half_width = max(1, int(round(0.20 * float(row["width_median_px"]))))
    gap_offset = max(3, int(round(0.42 * float(row["pitch_px"]))))
    for y in range(h):
        center = int(np.clip(round(float(row["path_x"][y])), 0, image.shape[1] - 1))
        center_slice = image[y, max(0, center - half_width): min(image.shape[1], center + half_width + 1)]
        left = int(np.clip(center - gap_offset, 0, image.shape[1] - 1))
        right = int(np.clip(center + gap_offset, 0, image.shape[1] - 1))
        profile[y] = float(np.mean(center_slice) - 0.5 * (image[y, left] + image[y, right]))
    return gaussian_filter1d(profile, sigma=3.0).astype(np.float32)


def guide_axial_modulations(
    detail_guide: np.ndarray,
    layers: list[dict],
) -> dict[str, np.ndarray]:
    """Convert guide contrast into bounded, geometry-neutral axial modulation."""
    clean = gaussian_filter(detail_guide, sigma=0.65)
    modulations: dict[str, np.ndarray] = {}
    for row in layers:
        contrast = layer_axial_contrast_profile(clean, row)
        margin = max(10, int(round(0.10 * float(row["length_px"]))))
        y0 = int(np.clip(math.ceil(row["top_y_px"]) + margin, 0, clean.shape[0] - 1))
        y1 = int(np.clip(math.floor(row["bottom_y_px"]) - margin + 1, y0 + 1, clean.shape[0]))
        positive = contrast[y0:y1]
        positive = positive[positive > 1e-5]
        normalizer = float(np.median(positive)) if len(positive) else 1.0
        modulation = gaussian_filter1d(contrast / max(normalizer, 1e-5), sigma=5.0)
        modulation = np.clip(modulation, 0.58, 1.42)
        # Detail is allowed only after the endpoint measurement zone. This
        # prevents a real axial brightness change from moving the half-height
        # endpoint used by the geometry audit.
        ys = np.arange(clean.shape[0], dtype=np.float32)
        endpoint_distance = np.minimum(
            np.abs(ys - float(row["top_y_px"])),
            np.abs(ys - float(row["bottom_y_px"])),
        )
        endpoint_blend = np.clip((endpoint_distance - 4.0) / 14.0, 0.0, 1.0)
        modulation = 1.0 + (modulation - 1.0) * endpoint_blend
        modulations[row["layer_id"]] = modulation.astype(np.float32)
    return modulations


def hard_shape_projection(
    generated: np.ndarray,
    layers: list[dict],
    detail_guide: np.ndarray | None = None,
    detail_weight: float = 0.55,
) -> tuple[np.ndarray, dict]:
    """Fuse guide detail with generative appearance and measured analytic ribbons."""
    if detail_guide is not None and detail_guide.shape != generated.shape:
        raise ValueError(
            f"Detail-guide shape {detail_guide.shape} does not match generated shape {generated.shape}"
        )
    detail_weight = float(np.clip(detail_weight, 0.0, 1.0))
    median_pitch = float(np.median([row["pitch_px"] for row in layers]))
    # Remove the generator's own group-level endpoint and layer periodicity;
    # otherwise those edges compete with the measured analytic ribbons.
    # A generator can invent a different stripe period.  Suppress at least one
    # complete measured pitch transversely before adding analytic ribbons;
    # otherwise residual invented stripes create duplicate detections and bias
    # FWHM even though the intended ribbon coordinates are correct.
    background_sigma_x = max(8.0, 1.15 * median_pitch)
    generated_background = gaussian_filter(generated, sigma=(10.0, background_sigma_x))
    support = expanded_support(layers, generated.shape)
    structure_region = foreground_structure_region(layers, generated.shape)
    if detail_guide is None:
        matched_guide = generated
        guide_background = generated_background
        structural_base = generated
        axial_modulations = {
            row["layer_id"]: np.ones(generated.shape[0], dtype=np.float32)
            for row in layers
        }
    else:
        matched_guide = robust_intensity_match(detail_guide, generated)
        guide_background = gaussian_filter(matched_guide, sigma=(6.0, background_sigma_x))
        guide_structure = gaussian_filter(matched_guide, sigma=0.85)
        structural_base = (
            generated * (1.0 - detail_weight) + guide_structure * detail_weight
        )
        structural_base = (
            generated * (1.0 - structure_region) + structural_base * structure_region
        )
        axial_modulations = guide_axial_modulations(matched_guide, layers)
    background = (
        generated_background * (1.0 - detail_weight)
        + guide_background * detail_weight
    )
    amplitudes = robust_layer_amplitudes(generated, generated_background, layers)
    # v11 transition constants are retained: v13 adds structural detail but
    # deliberately does not substitute v12's sharper geometry renderer.
    transition = 0.55
    edge_transition = 0.38
    width_scales = {row["layer_id"]: 1.0 for row in layers}

    def render_once() -> np.ndarray:
        signal = np.zeros_like(generated, dtype=np.float32)
        for row in layers:
            width = row["width_median_px"] or max(1.0, 0.28 * row["pitch_px"])
            effective_width = float(width) * width_scales[row["layer_id"]]
            radius = max(4, int(math.ceil(0.5 * effective_width + 4.5 * edge_transition)))
            amplitude = amplitudes[row["layer_id"]]
            y0 = max(0, int(math.floor(row["top_y_px"] - 6.0)))
            y1 = min(generated.shape[0] - 1, int(math.ceil(row["bottom_y_px"] + 6.0)))
            for y in range(y0, y1 + 1):
                top_gate = 1.0 / (1.0 + math.exp(-(y - row["top_y_px"]) / transition))
                bottom_gate = 1.0 / (1.0 + math.exp((y - row["bottom_y_px"]) / transition))
                axial_gate = top_gate * bottom_gate
                center = float(row["path_x"][y])
                xi0 = max(0, int(math.floor(center)) - radius)
                xi1 = min(generated.shape[1], int(math.ceil(center)) + radius + 1)
                xs = np.arange(xi0, xi1, dtype=np.float32)
                left_edge = center - 0.5 * effective_width
                right_edge = center + 0.5 * effective_width
                left_gate = 1.0 / (
                    1.0 + np.exp(-np.clip((xs - left_edge) / edge_transition, -30.0, 30.0))
                )
                right_gate = 1.0 / (
                    1.0 + np.exp(np.clip((xs - right_edge) / edge_transition, -30.0, 30.0))
                )
                guide_detail = float(axial_modulations[row["layer_id"]][y])
                axial_amplitude = amplitude * (
                    (1.0 - detail_weight) + detail_weight * guide_detail
                )
                profile = axial_amplitude * axial_gate * left_gate * right_gate
                signal[y, xi0:xi1] = np.maximum(
                    signal[y, xi0:xi1], profile.astype(np.float32)
                )
        projected = np.clip(background + signal, 0.0, 1.0)
        return structural_base * (1.0 - support) + projected * support

    # FWHM is affected slightly by the retained low-frequency background and
    # subpixel rasterization. Calibrate every ribbon against the exact same
    # measurement operator used by the audit, without moving its centerline or
    # endpoints.
    calibration_rounds = 3
    result = render_once()
    for _ in range(calibration_rounds):
        for row in layers:
            target = row["width_median_px"]
            measured = shape.measure_layer_width(
                result,
                row["path_x"],
                row["top_y_px"],
                row["bottom_y_px"],
                row["pitch_px"],
            )["median"]
            if target is None or measured is None:
                continue
            correction = float(np.clip(target / max(measured, 1e-8), 0.78, 1.22))
            width_scales[row["layer_id"]] = float(np.clip(
                width_scales[row["layer_id"]] * correction**0.85, 0.55, 1.35
            ))
        result = render_once()
    detail_enabled = detail_guide is not None and detail_weight > 0.0
    return np.clip(result, 0.0, 1.0).astype(np.float32), {
        "method": (
            "v13: v11 measured-ribbon hard projection plus same-coordinate blind-guide detail"
            if detail_enabled else
            "v11: generative low-frequency appearance plus measured-ribbon hard projection"
        ),
        "median_pitch_px": median_pitch,
        "generative_periodicity_suppression_sigma_x_px": background_sigma_x,
        "transverse_profile": "finite-width dual-logistic slab with closed-loop measured FWHM",
        "transverse_edge_transition_px": edge_transition,
        "endpoint_profile": "logistic half-height at measured subpixel endpoints",
        "endpoint_transition_px": transition,
        "per_lamella_fwhm_calibration_rounds": calibration_rounds,
        "effective_width_scale_min": float(min(width_scales.values())),
        "effective_width_scale_max": float(max(width_scales.values())),
        "support_feather_sigma_px": 2.0,
        "blind_guide_detail_weight": detail_weight,
        "blind_guide_roles": ([
                "same-coordinate foreground low/mid-frequency structure",
                "central-bridge detail",
                "per-lamella axial contrast modulation",
            ] if detail_enabled else ["geometry measurement only"]),
        "geometry_source": "measured centerlines, endpoints, widths and interlayer gaps only",
        "generated_pixel_role": "appearance proposal and residual background",
        "source_pixel_writeback": False,
    }


def transverse_edge_clarity_at_row(
    image: np.ndarray,
    path_x: np.ndarray,
    y: int,
    pitch: float,
) -> float | None:
    """Measure the weaker normalized side-edge slope of one lamella."""
    y = int(np.clip(y, 2, image.shape[0] - 3))
    center = float(path_x[y])
    radius = max(5, int(math.floor(0.47 * pitch)))
    x0 = max(0, int(math.floor(center)) - radius)
    x1 = min(image.shape[1], int(math.ceil(center)) + radius + 1)
    if x1 - x0 < 9:
        return None
    profile = gaussian_filter1d(
        image[y - 2 : y + 3, x0:x1].mean(axis=0), sigma=0.65
    )
    expected = int(np.clip(round(center) - x0, 2, len(profile) - 3))
    peak = int(expected - 2 + np.argmax(profile[expected - 2 : expected + 3]))
    baseline = float(np.median(np.concatenate((profile[:2], profile[-2:]))))
    contrast = float(profile[peak] - baseline)
    if contrast <= 1.0 / 65535.0 or peak < 2 or peak > len(profile) - 3:
        return None
    gradient = np.gradient(profile)
    left_slope = float(np.max(gradient[: peak + 1]))
    right_slope = float(np.max(-gradient[peak:]))
    if min(left_slope, right_slope) <= 0.0:
        return None
    return float(min(left_slope, right_slope) / contrast)


def measure_layer_edge_clarity(image: np.ndarray, row: dict, samples: int = 25) -> dict:
    margin = max(8.0, 0.12 * row["length_px"])
    if row["bottom_y_px"] - row["top_y_px"] <= 2.0 * margin:
        return {"median": None, "p10": None, "samples": 0}
    ys = np.linspace(row["top_y_px"] + margin, row["bottom_y_px"] - margin, samples)
    values = [
        transverse_edge_clarity_at_row(
            image, row["path_x"], int(round(y)), row["pitch_px"]
        )
        for y in ys
    ]
    valid = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(valid):
        return {"median": None, "p10": None, "samples": 0}
    return {
        "median": float(np.median(valid)),
        "p10": float(np.percentile(valid, 10.0)),
        "samples": int(len(valid)),
    }


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if len(a) < 3 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def normalized_for_structure(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    lo, hi = np.percentile(values, (1.0, 99.0))
    return np.clip((image - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def structure_detail_consistency(
    guide: np.ndarray,
    generated: np.ndarray,
    projected: np.ndarray,
    layers: list[dict],
) -> tuple[dict, list[dict]]:
    """Audit same-coordinate structure without treating synthetic texture as metrology."""
    region = foreground_structure_region(layers, guide.shape) >= 0.50
    guide_n = normalized_for_structure(guide, region)
    generated_n = normalized_for_structure(generated, region)
    projected_n = normalized_for_structure(projected, region)

    guide_low = gaussian_filter(guide_n, sigma=4.0)
    guide_mid = gaussian_filter(guide_n, sigma=0.8) - guide_low
    guide_gradient = np.hypot(*np.gradient(gaussian_filter(guide_n, sigma=1.0)))

    def frequency_metrics(candidate: np.ndarray) -> dict:
        candidate_low = gaussian_filter(candidate, sigma=4.0)
        candidate_mid = gaussian_filter(candidate, sigma=0.8) - candidate_low
        candidate_gradient = np.hypot(*np.gradient(gaussian_filter(candidate, sigma=1.0)))
        return {
            "low_frequency_correlation": safe_correlation(
                guide_low[region], candidate_low[region]
            ),
            "mid_frequency_correlation": safe_correlation(
                guide_mid[region], candidate_mid[region]
            ),
            "gradient_magnitude_correlation": safe_correlation(
                guide_gradient[region], candidate_gradient[region]
            ),
        }

    layer_rows = []
    for row in layers:
        margin = max(12, int(round(0.10 * float(row["length_px"]))))
        y0 = int(np.clip(math.ceil(row["top_y_px"]) + margin, 0, guide.shape[0] - 1))
        y1 = int(np.clip(
            math.floor(row["bottom_y_px"]) - margin + 1, y0 + 1, guide.shape[0]
        ))
        guide_profile = layer_axial_contrast_profile(guide_n, row)[y0:y1]
        generated_profile = layer_axial_contrast_profile(generated_n, row)[y0:y1]
        projected_profile = layer_axial_contrast_profile(projected_n, row)[y0:y1]
        generated_corr = safe_correlation(guide_profile, generated_profile)
        projected_corr = safe_correlation(guide_profile, projected_profile)
        layer_rows.append({
            "side": row["side"],
            "layer_id": row["layer_id"],
            "sample_top_y_px": y0,
            "sample_bottom_y_px": y1 - 1,
            "soft_generated_axial_detail_correlation": generated_corr,
            "projected_axial_detail_correlation": projected_corr,
            "correlation_gain": (
                projected_corr - generated_corr
                if generated_corr is not None and projected_corr is not None else None
            ),
        })

    generated_values = np.asarray([
        row["soft_generated_axial_detail_correlation"] for row in layer_rows
        if row["soft_generated_axial_detail_correlation"] is not None
    ], dtype=np.float64)
    projected_values = np.asarray([
        row["projected_axial_detail_correlation"] for row in layer_rows
        if row["projected_axial_detail_correlation"] is not None
    ], dtype=np.float64)
    generated_metrics = frequency_metrics(generated_n)
    projected_metrics = frequency_metrics(projected_n)
    generated_metrics.update({
        "lamella_axial_detail_correlation_median": float(np.median(generated_values)),
        "lamella_axial_detail_correlation_p10": float(np.percentile(generated_values, 10.0)),
    })
    projected_metrics.update({
        "lamella_axial_detail_correlation_median": float(np.median(projected_values)),
        "lamella_axial_detail_correlation_p10": float(np.percentile(projected_values, 10.0)),
    })
    return {
        "reference": "blind-denoised guide in unchanged source coordinates",
        "soft_generated": generated_metrics,
        "hard_projected": projected_metrics,
        "axial_detail_correlation_median_gain": float(
            np.median(projected_values) - np.median(generated_values)
        ),
        "guardrail_pass": bool(
            np.median(projected_values) >= np.median(generated_values)
            and projected_metrics["low_frequency_correlation"]
            >= max(0.80, generated_metrics["low_frequency_correlation"] - 0.05)
            and projected_metrics["mid_frequency_correlation"] >= 0.50
        ),
        "guardrails": {
            "axial_detail_correlation_nonregression": True,
            "low_frequency_correlation_min": 0.80,
            "low_frequency_allowed_drop_from_soft_generation": 0.05,
            "mid_frequency_correlation_min": 0.50,
        },
    }, layer_rows


def audit_projection(
    source: np.ndarray,
    guide: np.ndarray,
    generated: np.ndarray,
    projected: np.ndarray,
    layers: list[dict],
    left_roi: base.Roi,
    right_roi: base.Roi,
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
) -> dict:
    condition_references = []
    raw_validation_references = []
    for row in layers:
        guide_endpoint = length.EndpointMeasurement(
            top=row["guide_top_y_px"],
            bottom=row["guide_bottom_y_px"],
            length=row["guide_length_px"],
            top_spread=0.0,
            bottom_spread=0.0,
            uncertainty=row["endpoint_uncertainty_px"],
            top_snr=row["top_snr"],
            bottom_snr=row["bottom_snr"],
        )
        raw_endpoint = length.EndpointMeasurement(
            top=row["raw_validation_top_y_px"],
            bottom=row["raw_validation_bottom_y_px"],
            length=row["raw_validation_length_px"],
            top_spread=0.0,
            bottom_spread=0.0,
            uncertainty=row["endpoint_uncertainty_px"],
            top_snr=row["top_snr"],
            bottom_snr=row["bottom_snr"],
        )
        common = {
            "side": row["side"],
            "layer_id": row["layer_id"],
            "center": row["center_x_px"],
            "pitch": row["pitch_px"],
            "path_x": row["path_x"],
        }
        condition_references.append({**common, "raw": guide_endpoint})
        raw_validation_references.append({**common, "raw": raw_endpoint})
    raw_track_boundary, raw_boundary_rows = quality.boundary_geometry(
        projected, raw_validation_references, top_range, bottom_range
    )
    condition_boundary, condition_boundary_rows = quality.boundary_geometry(
        projected, condition_references, top_range, bottom_range
    )
    condition_by_id = {row["layer_id"]: row for row in condition_boundary_rows}
    raw_boundary_by_id = {row["layer_id"]: row for row in raw_boundary_rows}
    layer_comparison_rows = []
    output_layers = []
    for row in layers:
        guide_width = row["width_median_px"]
        output_width_stats = shape.measure_layer_width(
            projected,
            row["path_x"],
            row["top_y_px"],
            row["bottom_y_px"],
            row["pitch_px"],
        )
        output_width = output_width_stats["median"]
        guide_boundary = condition_by_id[row["layer_id"]]
        raw_boundary = raw_boundary_by_id[row["layer_id"]]
        output_top = float(guide_boundary["enhanced_top_y_px"])
        output_bottom = float(guide_boundary["enhanced_bottom_y_px"])
        output_length = float(guide_boundary["enhanced_length_px"])
        raw_width = row.get("raw_validation_width_median_px")
        guide_edge = measure_layer_edge_clarity(guide, row)
        raw_edge = measure_layer_edge_clarity(source, row)
        output_edge = measure_layer_edge_clarity(projected, row)
        raw_width_spread = (
            float(row["raw_validation_width_p90_px"] - row["raw_validation_width_p10_px"])
            if row.get("raw_validation_width_p90_px") is not None
            and row.get("raw_validation_width_p10_px") is not None else 0.0
        )
        raw_width_tolerance = max(0.15, 0.5 * raw_width_spread)
        raw_length_tolerance = max(
            0.25, 2.0 * float(row.get("raw_validation_endpoint_uncertainty_px", 0.0))
        )
        guide_width_relative_error = (
            abs(float(output_width - guide_width)) / max(float(guide_width), 1e-8)
            if output_width is not None and guide_width is not None else None
        )
        raw_width_relative_error = (
            abs(float(output_width - raw_width)) / max(float(raw_width), 1e-8)
            if output_width is not None and raw_width is not None else None
        )
        raw_width_nonregression = bool(
            output_width is not None and raw_width is not None and guide_width is not None
            and abs(float(output_width - raw_width))
            <= abs(float(guide_width - raw_width)) + raw_width_tolerance
        )
        raw_length_nonregression = bool(
            abs(output_length - float(row["raw_validation_length_px"]))
            <= abs(float(row["guide_length_px"] - row["raw_validation_length_px"]))
            + raw_length_tolerance
        )
        layer_comparison_rows.append({
            "side": row["side"],
            "layer_id": row["layer_id"],
            "constraint_confidence": row["constraint_confidence"],
            "guide_top_y_px": row["guide_top_y_px"],
            "raw_top_y_px": row["raw_validation_top_y_px"],
            "output_top_y_px": output_top,
            "guide_bottom_y_px": row["guide_bottom_y_px"],
            "raw_bottom_y_px": row["raw_validation_bottom_y_px"],
            "output_bottom_y_px": output_bottom,
            "guide_length_px": row["guide_length_px"],
            "raw_length_px": row["raw_validation_length_px"],
            "output_length_px": output_length,
            "output_vs_guide_length_abs_px": abs(output_length - row["guide_length_px"]),
            "output_vs_raw_length_abs_px": abs(output_length - row["raw_validation_length_px"]),
            "guide_width_px": guide_width,
            "raw_width_px": raw_width,
            "output_width_px": output_width,
            "output_vs_guide_width_relative": guide_width_relative_error,
            "output_vs_raw_width_relative": raw_width_relative_error,
            "guide_edge_clarity": guide_edge["median"],
            "raw_edge_clarity": raw_edge["median"],
            "output_edge_clarity": output_edge["median"],
            "raw_width_nonregression": raw_width_nonregression,
            "raw_length_nonregression": raw_length_nonregression,
            "dual_evidence_pass": bool(
                guide_width_relative_error is not None
                and guide_width_relative_error <= 0.015
                and abs(output_length - row["guide_length_px"]) <= 0.35
                and raw_width_nonregression
                and raw_length_nonregression
            ),
        })
        output_row = dict(row)
        output_row.update({
            "top_y_px": output_top,
            "bottom_y_px": output_bottom,
            "length_px": output_length,
            "width_median_px": output_width,
        })
        output_layers.append(output_row)

    guide_gaps = shape.measure_gaps(layers)
    output_gaps = shape.measure_gaps(output_layers)
    gap_comparison_rows = []
    for guide_gap, output_gap in zip(guide_gaps, output_gaps):
        raw_width = guide_gap.get("raw_validation_gap_width_px")
        raw_length = guide_gap.get("raw_validation_length_px")
        output_width = output_gap["gap_width_px"]
        output_length = output_gap["length_px"]
        guide_width = guide_gap["gap_width_px"]
        guide_length = guide_gap["length_px"]
        raw_spread = (
            float(guide_gap["raw_validation_gap_width_p90_px"]
                  - guide_gap["raw_validation_gap_width_p10_px"])
            if guide_gap.get("raw_validation_gap_width_p90_px") is not None
            and guide_gap.get("raw_validation_gap_width_p10_px") is not None else 0.0
        )
        raw_width_tolerance = max(0.20, 0.5 * raw_spread)
        raw_length_tolerance = 0.50
        guide_width_relative_error = abs(output_width - guide_width) / max(guide_width, 1e-8)
        raw_width_relative_error = (
            abs(output_width - raw_width) / max(raw_width, 1e-8)
            if raw_width is not None else None
        )
        raw_width_nonregression = bool(
            raw_width is not None
            and abs(output_width - raw_width)
            <= abs(guide_width - raw_width) + raw_width_tolerance
        )
        raw_length_nonregression = bool(
            raw_length is not None
            and abs(output_length - raw_length)
            <= abs(guide_length - raw_length) + raw_length_tolerance
        )
        gap_comparison_rows.append({
            "side": guide_gap["side"],
            "gap_id": guide_gap["gap_id"],
            "left_layer_id": guide_gap["left_layer_id"],
            "right_layer_id": guide_gap["right_layer_id"],
            "constraint_confidence": guide_gap["constraint_confidence"],
            "guide_width_px": guide_width,
            "raw_width_px": raw_width,
            "output_width_px": output_width,
            "output_vs_guide_width_relative": guide_width_relative_error,
            "output_vs_raw_width_relative": raw_width_relative_error,
            "guide_length_px": guide_length,
            "raw_length_px": raw_length,
            "output_length_px": output_length,
            "output_vs_guide_length_abs_px": abs(output_length - guide_length),
            "output_vs_raw_length_abs_px": (
                abs(output_length - raw_length) if raw_length is not None else None
            ),
            "raw_width_nonregression": raw_width_nonregression,
            "raw_length_nonregression": raw_length_nonregression,
            "dual_evidence_pass": bool(
                guide_width_relative_error <= 0.02
                and abs(output_length - guide_length) <= 0.50
                and raw_width_nonregression
                and raw_length_nonregression
            ),
        })

    layer_guide_width_errors = np.asarray([
        row["output_vs_guide_width_relative"] for row in layer_comparison_rows
        if row["output_vs_guide_width_relative"] is not None
    ], dtype=np.float64)
    layer_raw_width_errors = np.asarray([
        row["output_vs_raw_width_relative"] for row in layer_comparison_rows
        if row["output_vs_raw_width_relative"] is not None
    ], dtype=np.float64)
    gap_guide_width_errors = np.asarray([
        row["output_vs_guide_width_relative"] for row in gap_comparison_rows
    ], dtype=np.float64)
    gap_raw_width_errors = np.asarray([
        row["output_vs_raw_width_relative"] for row in gap_comparison_rows
        if row["output_vs_raw_width_relative"] is not None
    ], dtype=np.float64)
    guide_edge_values = np.asarray([
        row["guide_edge_clarity"] for row in layer_comparison_rows
        if row["guide_edge_clarity"] is not None
    ], dtype=np.float64)
    raw_edge_values = np.asarray([
        row["raw_edge_clarity"] for row in layer_comparison_rows
        if row["raw_edge_clarity"] is not None
    ], dtype=np.float64)
    output_edge_values = np.asarray([
        row["output_edge_clarity"] for row in layer_comparison_rows
        if row["output_edge_clarity"] is not None
    ], dtype=np.float64)
    detected = {}
    for side, roi in (("left", left_roi), ("right", right_roi)):
        centers, pitch, _ = length.detect_centers(projected, roi)
        detected[side] = {"count": int(len(centers)), "median_pitch_px": float(pitch)}
    matched_contrast = {"left": 0, "right": 0}
    matched_contrast_values = []
    for row in layers:
        margin = max(10.0, 0.15 * row["length_px"])
        ys = np.linspace(row["top_y_px"] + margin, row["bottom_y_px"] - margin, 21)
        contrasts = []
        for yf in ys:
            y = int(np.clip(round(yf), 0, projected.shape[0] - 1))
            center = float(row["path_x"][y])
            xc = int(np.clip(round(center), 0, projected.shape[1] - 1))
            dx = max(3, int(round(0.42 * row["pitch_px"])))
            xl = int(np.clip(xc - dx, 0, projected.shape[1] - 1))
            xr = int(np.clip(xc + dx, 0, projected.shape[1] - 1))
            contrasts.append(float(projected[y, xc] - 0.5 * (projected[y, xl] + projected[y, xr])))
        contrast = float(np.median(contrasts))
        matched_contrast_values.append(contrast)
        if contrast >= 0.015:
            matched_contrast[row["side"]] += 1
    detail_consistency, detail_rows = structure_detail_consistency(
        guide, generated, projected, layers
    )
    return {
        "analytic_layer_count": len(layers),
        "ordinary_unmatched_peak_detector_diagnostic": detected,
        "constraint_matched_layer_detection": {
            "left": matched_contrast["left"],
            "right": matched_contrast["right"],
            "total": matched_contrast["left"] + matched_contrast["right"],
            "minimum_median_center_to_gap_contrast": float(min(matched_contrast_values)),
            "contrast_threshold": 0.015,
        },
        "constraint_coordinate_boundary_geometry": condition_boundary,
        "raw_input_boundary_geometry_comparison": raw_track_boundary,
        "layer_geometry_comparison": {
            "count": len(layer_comparison_rows),
            "guide_width_relative_error_median": float(np.median(layer_guide_width_errors)),
            "guide_width_relative_error_p95": float(np.percentile(layer_guide_width_errors, 95.0)),
            "raw_width_relative_error_median": float(np.median(layer_raw_width_errors)),
            "raw_width_relative_error_p95": float(np.percentile(layer_raw_width_errors, 95.0)),
            "raw_width_nonregression_count": sum(
                row["raw_width_nonregression"] for row in layer_comparison_rows
            ),
            "raw_length_nonregression_count": sum(
                row["raw_length_nonregression"] for row in layer_comparison_rows
            ),
            "dual_evidence_pass_count": sum(
                row["dual_evidence_pass"] for row in layer_comparison_rows
            ),
        },
        "interlayer_geometry_comparison": {
            "count": len(gap_comparison_rows),
            "guide_width_relative_error_median": float(np.median(gap_guide_width_errors)),
            "guide_width_relative_error_p95": float(np.percentile(gap_guide_width_errors, 95.0)),
            "raw_width_relative_error_median": float(np.median(gap_raw_width_errors)),
            "raw_width_relative_error_p95": float(np.percentile(gap_raw_width_errors, 95.0)),
            "guide_length_abs_error_p95_px": float(np.percentile([
                row["output_vs_guide_length_abs_px"] for row in gap_comparison_rows
            ], 95.0)),
            "raw_length_abs_error_p95_px": float(np.percentile([
                row["output_vs_raw_length_abs_px"] for row in gap_comparison_rows
                if row["output_vs_raw_length_abs_px"] is not None
            ], 95.0)),
            "raw_width_nonregression_count": sum(
                row["raw_width_nonregression"] for row in gap_comparison_rows
            ),
            "raw_length_nonregression_count": sum(
                row["raw_length_nonregression"] for row in gap_comparison_rows
            ),
            "dual_evidence_pass_count": sum(
                row["dual_evidence_pass"] for row in gap_comparison_rows
            ),
        },
        "edge_clarity": {
            "metric": "weaker-side normalized transverse edge slope; higher is sharper",
            "raw_median": float(np.median(raw_edge_values)),
            "guide_median": float(np.median(guide_edge_values)),
            "output_median": float(np.median(output_edge_values)),
            "output_vs_raw_gain": float(
                np.median(output_edge_values) / max(np.median(raw_edge_values), 1e-8) - 1.0
            ),
            "output_vs_guide_gain": float(
                np.median(output_edge_values) / max(np.median(guide_edge_values), 1e-8) - 1.0
            ),
        },
        "structure_detail_consistency": detail_consistency,
        "dual_evidence_guardrails": {
            "guide_layer_geometry_pass": bool(
                condition_boundary["endpoint_shift_abs_p95_px"] <= 0.20
                and condition_boundary["length_delta_abs_p95_px"] <= 0.25
                and np.percentile(layer_guide_width_errors, 95.0) <= 0.015
            ),
            "raw_layer_nonregression_pass": bool(
                np.mean([row["dual_evidence_pass"] for row in layer_comparison_rows]) >= 0.95
            ),
            "guide_interlayer_geometry_pass": bool(
                np.percentile(gap_guide_width_errors, 95.0) <= 0.02
                and np.percentile([
                    row["output_vs_guide_length_abs_px"] for row in gap_comparison_rows
                ], 95.0) <= 0.50
            ),
            "raw_interlayer_nonregression_pass": bool(
                np.mean([row["dual_evidence_pass"] for row in gap_comparison_rows]) >= 0.95
            ),
            "edge_clarity_pass": bool(
                np.median(output_edge_values) >= np.median(guide_edge_values)
            ),
        },
        "guide_width_median_px": float(np.median([
            row["guide_width_px"] for row in layer_comparison_rows
            if row["guide_width_px"] is not None
        ])),
        "raw_width_median_px": float(np.median([
            row["raw_width_px"] for row in layer_comparison_rows
            if row["raw_width_px"] is not None
        ])),
        "output_width_median_px": float(np.median([
            row["output_width_px"] for row in layer_comparison_rows
            if row["output_width_px"] is not None
        ])),
        "condition_boundary_rows": condition_boundary_rows,
        "raw_boundary_rows": raw_boundary_rows,
        "layer_comparison_rows": layer_comparison_rows,
        "gap_comparison_rows": gap_comparison_rows,
        "structure_detail_rows": detail_rows,
    }


def save_comparison(
    path: Path,
    source: np.ndarray,
    guide: np.ndarray,
    generated: np.ndarray,
    projected: np.ndarray,
) -> None:
    panels = [
        base.panel(source, (550, 400), "raw 16-bit source"),
        base.panel(guide, (550, 400), "blind-denoised detail guide"),
        base.panel(generated, (550, 400), "soft guided generation"),
        base.panel(projected, (550, 400), "geometry-locked detail projection"),
    ]
    canvas = Image.new("L", (2200, 428), 0)
    for index, panel in enumerate(panels):
        canvas.paste(panel, (550 * index, 0))
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument(
        "--guide", type=Path,
        help=("Registered blind-denoised image used for both geometry measurement and "
              "same-coordinate structural-detail guidance. Defaults to source."),
    )
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument(
        "--profile", choices=("v11", "v13", "v15"), default="v11",
        help=("Stable v11 projection by default; v13 enables guide-detail fusion; "
              "v15 forbids generated pixels inside the measurement core."),
    )
    ap.add_argument(
        "--detail-guide-weight", type=float,
        help="Override guide-detail contribution. Defaults to 0 for v11 and 0.55 for v13.",
    )
    ap.add_argument("--carrier-smoothing-sigma", type=float, default=0.0)
    ap.add_argument("--carrier-residual-floor", type=float, default=1.0)
    ap.add_argument("--carrier-transverse-unsharp", type=float, default=0.0)
    ap.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    ap.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    ap.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    ap.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    guide, guide_info = base.load_gray(args.guide) if args.guide else (source, source_info)
    if guide.shape != source.shape:
        raise ValueError(f"Guide shape {guide.shape} does not match source shape {source.shape}")
    with Image.open(args.generated) as generated_file:
        generated_input_width, generated_input_height = generated_file.size
    generated = resized_gray(args.generated, source.shape)
    left_roi = args.left_roi.clamp(source.shape)
    right_roi = args.right_roi.clamp(source.shape)
    layers = shape.measure_layers(
        source, left_roi, right_roi, args.top_range, args.bottom_range, guide=guide
    )
    detail_weight = (
        float(args.detail_guide_weight)
        if args.detail_guide_weight is not None else (0.55 if args.profile == "v13" else 0.0)
    )
    if args.profile == "v11" and detail_weight != 0.0:
        raise ValueError("The frozen v11 profile requires --detail-guide-weight 0")
    measurement_core = None
    if args.profile == "v15":
        projected, method, measurement_core = structure_carrier_projection(
            generated,
            guide,
            layers,
            smoothing_sigma=args.carrier_smoothing_sigma,
            residual_floor=args.carrier_residual_floor,
            transverse_unsharp_amount=args.carrier_transverse_unsharp,
        )
    else:
        projected, method = hard_shape_projection(
            generated,
            layers,
            detail_guide=guide if args.profile == "v13" else None,
            detail_weight=detail_weight,
        )
    audit = audit_projection(
        source, guide, generated, projected, layers,
        left_roi, right_roi, args.top_range, args.bottom_range
    )
    audit["structure_detail_consistency"]["applicable_to_profile"] = (
        args.profile in {"v13", "v15"}
    )
    if args.profile == "v11":
        # v11 intentionally uses the guide for geometry only. A direct-detail
        # correlation threshold introduced in v13 is therefore informative,
        # not an acceptance criterion for the frozen v11 release.
        audit["structure_detail_consistency"]["guardrail_pass"] = None
    if generated.shape != source.shape or projected.shape != source.shape:
        raise RuntimeError(
            f"Dimension lock failed: source={source.shape}, generated={generated.shape}, "
            f"projected={projected.shape}"
        )

    Image.fromarray(np.rint(projected * 255).astype(np.uint8), mode="L").save(
        args.outdir / "GENERATIVE_shape_hard_projected.png"
    )
    imwrite(
        args.outdir / "GENERATIVE_shape_hard_projected_16bit.tif",
        base.to_uint16(projected),
        photometric="minisblack",
        description=(
            "VISUAL_ONLY: generative appearance hard-projected onto measured lamella geometry."
            if args.profile == "v11" else
            "VISUAL_ONLY: guide-conditioned generative appearance with measured-geometry "
            "and same-coordinate detail projection."
            if args.profile == "v13" else
            "MEASUREMENT_ASSIST: the component ROI is derived only from the same-coordinate "
            "blind-denoised guide; validate calibration before quantitative use."
        ),
    )
    if args.profile == "v15":
        Image.fromarray(np.rint(projected * 255).astype(np.uint8), mode="L").save(
            args.outdir / "MEASUREMENT_structure_carrier_enhanced.png"
        )
        imwrite(
            args.outdir / "MEASUREMENT_structure_carrier_enhanced_16bit.tif",
            base.to_uint16(projected),
            photometric="minisblack",
            description=(
                "MEASUREMENT_ASSIST: zero generated-pixel contribution inside the component "
                "measurement ROI; same-coordinate blind-guide structural carrier."
            ),
        )
        Image.fromarray(measurement_core.astype(np.uint8) * 255, mode="L").save(
            args.outdir / "AUDIT_measurement_structure_core_mask.png"
        )
    save_comparison(
        args.outdir / "GENERATIVE_shape_projection_comparison.png",
        source,
        guide,
        generated,
        projected,
    )
    condition_boundary_rows = audit.pop("condition_boundary_rows")
    raw_boundary_rows = audit.pop("raw_boundary_rows")
    layer_comparison_rows = audit.pop("layer_comparison_rows")
    gap_comparison_rows = audit.pop("gap_comparison_rows")
    structure_detail_rows = audit.pop("structure_detail_rows")
    quality.save_boundary_overlay(
        args.outdir / "GENERATIVE_shape_hard_projected_boundary_overlay.png",
        projected,
        condition_boundary_rows,
        base.Roi(600, 1240, 300, 1900).clamp(source.shape),
    )
    quality.save_boundary_overlay(
        args.outdir / "GENERATIVE_raw_input_comparison_overlay.png",
        projected,
        raw_boundary_rows,
        base.Roi(600, 1240, 300, 1900).clamp(source.shape),
    )
    shape.write_csv(args.outdir / "lamella_dual_evidence_comparison.csv", layer_comparison_rows)
    shape.write_csv(args.outdir / "interlayer_dual_evidence_comparison.csv", gap_comparison_rows)
    shape.write_csv(args.outdir / "structure_detail_consistency.csv", structure_detail_rows)
    payload = {
        "completed": True,
        "source": source_info,
        "measurement_guide": guide_info,
        "generated_input": str(args.generated),
        "profile": args.profile,
        "dimension_audit": {
            "source_height": int(source.shape[0]),
            "source_width": int(source.shape[1]),
            "guide_height": int(guide.shape[0]),
            "guide_width": int(guide.shape[1]),
            "generated_input_height": int(generated_input_height),
            "generated_input_width": int(generated_input_width),
            "projection_input_height": int(generated.shape[0]),
            "projection_input_width": int(generated.shape[1]),
            "output_height": int(projected.shape[0]),
            "output_width": int(projected.shape[1]),
            "strict_input_output_match": bool(projected.shape == source.shape),
        },
        "method": method,
        "audit": audit,
        "measurement_warning": (
            "MEASUREMENT_ASSIST: no generated pixels occur inside the component ROI, but "
            "calibration and the exported raw/guide constraints remain authoritative."
            if args.profile == "v15" else
            "Still generative and not metrology-certified; geometry is analytically "
            "constrained but texture is synthetic."
        ),
    }
    (args.outdir / "hard_shape_projection_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed": True, "audit": audit}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
