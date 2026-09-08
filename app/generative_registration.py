#!/usr/bin/env python3
"""Register a source-sized generative candidate to a blind-denoised guide.

The generated image is visual-only.  Geometry is estimated exclusively from
the same-coordinate blind guide and its exported lamella constraints.  A
coarse phase-correlation translation is refined by a monotone piecewise-affine
envelope mapping inside the inspected component ROI.  The background remains
unchanged and the output canvas always retains the source dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, sobel
from skimage.registration import phase_cross_correlation
from tifffile import imread, imwrite


def load_gray(path: Path) -> np.ndarray:
    """Load common 8/16-bit image formats into float32 [0, 1]."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        array = np.asarray(imread(path))
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim == 3:
        rgb = array[..., :3].astype(np.float32)
        array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    original_dtype = array.dtype
    array = array.astype(np.float32)
    if np.issubdtype(original_dtype, np.unsignedinteger):
        scale = float(np.iinfo(original_dtype).max)
    elif float(np.nanmax(array)) > 255.0:
        scale = 65535.0
    elif float(np.nanmax(array)) > 1.0:
        scale = 255.0
    else:
        scale = 1.0
    return np.clip(array / max(scale, 1.0), 0.0, 1.0).astype(np.float32)


def robust_normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = (float(value) for value in np.percentile(image, (0.5, 99.7)))
    return np.clip((image - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)


def load_constraints(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "center_x_px", "pitch_px", "top_y_px", "bottom_y_px", "length_px"
    }
    parsed = []
    for row in rows:
        parsed.append({
            key: float(value) if key in numeric else value
            for key, value in row.items()
        })
    if not parsed or {row["side"] for row in parsed} != {"left", "right"}:
        raise ValueError("Constraint CSV must contain both left and right lamella groups")
    return parsed


def subpixel_extremum(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, center, right = (float(values[index - 1]), float(values[index]),
                           float(values[index + 1]))
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    offset = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return float(index + offset)


def locate_profile_feature(
    profile: np.ndarray,
    lower: float,
    upper: float,
    mode: str,
) -> float:
    lo = max(1, int(math.floor(lower)))
    hi = min(len(profile) - 1, int(math.ceil(upper)))
    if hi - lo < 3:
        raise ValueError(f"Invalid feature search interval: {lower:.2f}..{upper:.2f}")
    gradient = np.gradient(profile)
    if mode == "positive_gradient":
        score = gradient
    elif mode == "negative_gradient":
        score = -gradient
    elif mode == "peak":
        score = profile
    else:
        raise ValueError(f"Unknown profile feature mode: {mode}")
    local_index = int(np.argmax(score[lo:hi])) + lo
    return subpixel_extremum(score, local_index)


def edge_representation(image: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    smooth = gaussian_filter(robust_normalize(image), sigma=sigma)
    return np.hypot(sobel(smooth, axis=0), sobel(smooth, axis=1)).astype(np.float32)


def coarse_phase_translation(
    fixed: np.ndarray,
    moving: np.ndarray,
    constraints: list[dict],
) -> dict:
    centers = np.asarray([row["center_x_px"] for row in constraints], dtype=np.float64)
    pitches = np.asarray([row["pitch_px"] for row in constraints], dtype=np.float64)
    tops = np.asarray([row["top_y_px"] for row in constraints], dtype=np.float64)
    bottoms = np.asarray([row["bottom_y_px"] for row in constraints], dtype=np.float64)
    pitch = float(np.median(pitches))
    x0 = max(0, int(math.floor(np.min(centers) - 6.0 * pitch)))
    x1 = min(fixed.shape[1], int(math.ceil(np.max(centers) + 7.0 * pitch)))
    y0 = max(0, int(math.floor(np.min(tops) - 11.0 * pitch)))
    y1 = min(fixed.shape[0], int(math.ceil(np.max(bottoms) + 7.0 * pitch)))
    fixed_edges = edge_representation(fixed[y0:y1, x0:x1])
    moving_edges = edge_representation(moving[y0:y1, x0:x1])
    wy = np.hanning(fixed_edges.shape[0]).astype(np.float32)
    wx = np.hanning(fixed_edges.shape[1]).astype(np.float32)
    window = wy[:, None] * wx[None, :]
    shift, error, phase = phase_cross_correlation(
        fixed_edges * window,
        moving_edges * window,
        upsample_factor=10,
        normalization=None,
        disambiguate=True,
    )
    shift_y = float(np.clip(shift[0], -120.0, 120.0))
    shift_x = float(np.clip(shift[1], -120.0, 120.0))
    return {
        "shift_to_apply_y_px": shift_y,
        "shift_to_apply_x_px": shift_x,
        "normalized_rms_error": float(error),
        "phase_difference_rad": float(phase),
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }


def constraint_groups(constraints: list[dict]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for side in ("left", "right"):
        rows = [row for row in constraints if row["side"] == side]
        centers = np.asarray([row["center_x_px"] for row in rows], dtype=np.float64)
        groups[side] = {
            "centers": centers,
            "pitch": float(np.median([row["pitch_px"] for row in rows])),
        }
    return groups


def horizontal_profile(image: np.ndarray, y0: int, y1: int) -> np.ndarray:
    y0 = int(np.clip(y0, 0, image.shape[0] - 2))
    y1 = int(np.clip(y1, y0 + 2, image.shape[0]))
    return gaussian_filter1d(np.percentile(image[y0:y1], 80.0, axis=0), sigma=4.0)


def detect_horizontal_anchors(
    fixed: np.ndarray,
    moving: np.ndarray,
    constraints: list[dict],
    coarse: dict,
) -> tuple[dict, dict]:
    groups = constraint_groups(constraints)
    tops = np.asarray([row["top_y_px"] for row in constraints], dtype=np.float64)
    bottoms = np.asarray([row["bottom_y_px"] for row in constraints], dtype=np.float64)
    lengths = bottoms - tops
    fixed_y0 = int(math.ceil(np.max(tops) + 0.10 * np.median(lengths)))
    fixed_y1 = int(math.floor(np.min(bottoms) - 0.10 * np.median(lengths)))
    shift_y = float(coarse["shift_to_apply_y_px"])
    moving_y0 = int(round(fixed_y0 - shift_y))
    moving_y1 = int(round(fixed_y1 - shift_y))
    fixed_profile = horizontal_profile(fixed, fixed_y0, fixed_y1)
    moving_profile = horizontal_profile(moving, moving_y0, moving_y1)

    left = groups["left"]
    right = groups["right"]
    fixed_searches = {
        "left_outer": (
            float(np.min(left["centers"]) - 6.0 * left["pitch"]),
            float(np.min(left["centers"]) + 2.0 * left["pitch"]),
            "positive_gradient",
        ),
        "left_inner": (
            float(np.max(left["centers"]) + 0.5 * left["pitch"]),
            float(np.max(left["centers"]) + 4.0 * left["pitch"]),
            "peak",
        ),
        "right_inner": (
            float(np.min(right["centers"]) - 4.0 * right["pitch"]),
            float(np.min(right["centers"]) - 0.2 * right["pitch"]),
            "peak",
        ),
        "right_outer": (
            float(np.max(right["centers"]) - 2.0 * right["pitch"]),
            float(np.max(right["centers"]) + 6.0 * right["pitch"]),
            "negative_gradient",
        ),
    }
    fixed_anchors = {
        name: locate_profile_feature(fixed_profile, lower, upper, mode)
        for name, (lower, upper, mode) in fixed_searches.items()
    }
    shift_x = float(coarse["shift_to_apply_x_px"])
    moving_anchors = {}
    for name, (_, _, mode) in fixed_searches.items():
        side = "left" if name.startswith("left") else "right"
        pitch = groups[side]["pitch"]
        radius = max(30.0, (8.0 if name.endswith("outer") else 4.0) * pitch)
        expected = fixed_anchors[name] - shift_x
        moving_anchors[name] = locate_profile_feature(
            moving_profile, expected - radius, expected + radius, mode
        )
    profile_info = {
        "fixed_y_interval": [fixed_y0, fixed_y1],
        "moving_y_interval": [moving_y0, moving_y1],
    }
    return {"fixed": fixed_anchors, "moving": moving_anchors}, profile_info


def vertical_profile(
    image: np.ndarray,
    horizontal_anchors: dict,
) -> np.ndarray:
    left = image[:, int(math.floor(horizontal_anchors["left_outer"])):
                       int(math.ceil(horizontal_anchors["left_inner"]))]
    right = image[:, int(math.floor(horizontal_anchors["right_inner"])):
                        int(math.ceil(horizontal_anchors["right_outer"]))]
    if left.shape[1] < 10 or right.shape[1] < 10:
        raise ValueError("Detected horizontal envelope is too narrow")
    values = np.concatenate((left, right), axis=1)
    return gaussian_filter1d(np.percentile(values, 80.0, axis=1), sigma=4.0)


def detect_vertical_anchors(
    fixed: np.ndarray,
    moving: np.ndarray,
    constraints: list[dict],
    horizontal: dict,
    coarse: dict,
) -> dict:
    tops = np.asarray([row["top_y_px"] for row in constraints], dtype=np.float64)
    bottoms = np.asarray([row["bottom_y_px"] for row in constraints], dtype=np.float64)
    pitch = float(np.median([row["pitch_px"] for row in constraints]))
    fixed_profile = vertical_profile(fixed, horizontal["fixed"])
    moving_profile = vertical_profile(moving, horizontal["moving"])
    fixed_top = locate_profile_feature(
        fixed_profile, np.min(tops) - 9.0 * pitch, np.max(tops) + 7.0 * pitch,
        "positive_gradient",
    )
    fixed_bottom = locate_profile_feature(
        fixed_profile, np.min(bottoms) - 7.0 * pitch, np.max(bottoms) + 9.0 * pitch,
        "negative_gradient",
    )
    shift_y = float(coarse["shift_to_apply_y_px"])
    radius = max(30.0, 5.0 * pitch)
    moving_top = locate_profile_feature(
        moving_profile,
        fixed_top - shift_y - radius,
        fixed_top - shift_y + radius,
        "positive_gradient",
    )
    moving_bottom = locate_profile_feature(
        moving_profile,
        fixed_bottom - shift_y - radius,
        fixed_bottom - shift_y + radius,
        "negative_gradient",
    )
    return {
        "fixed": {"top": fixed_top, "bottom": fixed_bottom},
        "moving": {"top": moving_top, "bottom": moving_bottom},
    }


def validate_anchor_order(horizontal: dict, vertical: dict) -> None:
    order = ("left_outer", "left_inner", "right_inner", "right_outer")
    for role in ("fixed", "moving"):
        values = [horizontal[role][name] for name in order]
        if not np.all(np.diff(values) > 20.0):
            raise ValueError(f"Non-monotone {role} horizontal anchors: {values}")
        if vertical[role]["bottom"] - vertical[role]["top"] < 50.0:
            raise ValueError(f"Invalid {role} vertical envelope: {vertical[role]}")


def estimate_registration(
    fixed: np.ndarray,
    moving: np.ndarray,
    constraints: list[dict],
) -> dict:
    if fixed.shape != moving.shape:
        raise ValueError(f"Registration requires equal dimensions: {fixed.shape} vs {moving.shape}")
    fixed_n = robust_normalize(fixed)
    moving_n = robust_normalize(moving)
    coarse = coarse_phase_translation(fixed_n, moving_n, constraints)
    horizontal, profile_info = detect_horizontal_anchors(
        fixed_n, moving_n, constraints, coarse
    )
    vertical = detect_vertical_anchors(
        fixed_n, moving_n, constraints, horizontal, coarse
    )
    validate_anchor_order(horizontal, vertical)
    names = ("left_outer", "left_inner", "right_inner", "right_outer")
    fixed_x = np.asarray([horizontal["fixed"][name] for name in names])
    moving_x = np.asarray([horizontal["moving"][name] for name in names])
    horizontal_scales = np.diff(fixed_x) / np.diff(moving_x)
    vertical_scale = (
        (vertical["fixed"]["bottom"] - vertical["fixed"]["top"])
        / (vertical["moving"]["bottom"] - vertical["moving"]["top"])
    )
    guardrail = bool(
        np.all((horizontal_scales >= 0.88) & (horizontal_scales <= 1.12))
        and 0.88 <= vertical_scale <= 1.12
        and abs(coarse["shift_to_apply_y_px"]) <= 120.0
        and abs(coarse["shift_to_apply_x_px"]) <= 120.0
    )
    if not guardrail:
        raise RuntimeError(
            "Registration transform exceeds geometry-safety limits: "
            f"horizontal_scales={horizontal_scales.tolist()}, vertical_scale={vertical_scale}"
        )
    return {
        "coarse_phase_correlation": coarse,
        "horizontal": horizontal,
        "vertical": vertical,
        "profile_intervals": profile_info,
        "horizontal_forward_scales": horizontal_scales.tolist(),
        "vertical_forward_scale": float(vertical_scale),
        "guardrail_pass": guardrail,
        "guardrails": {
            "horizontal_scale_min": 0.88,
            "horizontal_scale_max": 1.12,
            "vertical_scale_min": 0.88,
            "vertical_scale_max": 1.12,
            "coarse_translation_abs_max_px": 120.0,
        },
    }


REGISTRATION_METHODS = (
    "phase-translation",
    "global-affine",
    "pchip-envelope",
    "piecewise-linear",
    "piecewise-affine",
    "piecewise-quintic",
    "piecewise-edge-preserving",
)


def registration_coordinates(
    model: dict,
    shape: tuple[int, int],
    method: str = "piecewise-affine",
) -> tuple[np.ndarray, np.ndarray, dict]:
    if method not in REGISTRATION_METHODS:
        raise ValueError(f"Unknown registration method: {method}")
    height, width = shape
    names = ("left_outer", "left_inner", "right_inner", "right_outer")
    fixed_x = np.asarray([model["horizontal"]["fixed"][name] for name in names])
    moving_x = np.asarray([model["horizontal"]["moving"][name] for name in names])
    pitch_like = max(10.0, float(np.median(np.diff(fixed_x[[0, 1, 2, 3]]))) / 20.0)
    margin_x = max(48.0, 4.0 * pitch_like)
    left_guard = max(0.0, min(fixed_x[0], moving_x[0]) - margin_x)
    right_guard = min(float(width - 1), max(fixed_x[-1], moving_x[-1]) + margin_x)
    fixed_x_map = np.concatenate(([left_guard], fixed_x, [right_guard]))
    moving_x_map = np.concatenate(([left_guard], moving_x, [right_guard]))
    output_x = np.arange(width, dtype=np.float32)

    fixed_y = model["vertical"]["fixed"]
    moving_y = model["vertical"]["moving"]
    margin_y = 80.0
    top_guard = max(0.0, min(fixed_y["top"], moving_y["top"]) - margin_y)
    bottom_guard = min(
        float(height - 1), max(fixed_y["bottom"], moving_y["bottom"]) + margin_y
    )
    fixed_y_map = np.asarray([top_guard, fixed_y["top"], fixed_y["bottom"], bottom_guard])
    moving_y_map = np.asarray([top_guard, moving_y["top"], moving_y["bottom"], bottom_guard])
    output_y = np.arange(height, dtype=np.float32)
    interpolation_order = 3
    edge_recovery = 0.0
    if method == "phase-translation":
        coarse = model["coarse_phase_correlation"]
        input_x = output_x - float(coarse["shift_to_apply_x_px"])
        input_y = output_y - float(coarse["shift_to_apply_y_px"])
    elif method == "global-affine":
        slope_x, intercept_x = np.polyfit(fixed_x, moving_x, deg=1)
        slope_y = (
            (moving_y["bottom"] - moving_y["top"])
            / (fixed_y["bottom"] - fixed_y["top"])
        )
        intercept_y = moving_y["top"] - slope_y * fixed_y["top"]
        input_x = slope_x * output_x + intercept_x
        input_y = slope_y * output_y + intercept_y
    elif method == "pchip-envelope":
        input_x = PchipInterpolator(
            fixed_x_map, moving_x_map, extrapolate=True
        )(output_x)
        input_y = PchipInterpolator(
            fixed_y_map, moving_y_map, extrapolate=True
        )(output_y)
    else:
        input_x = np.interp(output_x, fixed_x_map, moving_x_map)
        input_y = np.interp(output_y, fixed_y_map, moving_y_map)
        if method == "piecewise-linear":
            interpolation_order = 1
        elif method == "piecewise-quintic":
            interpolation_order = 5
        elif method == "piecewise-edge-preserving":
            edge_recovery = 0.35
    input_x = np.clip(input_x, 0.0, float(width - 1)).astype(np.float32)
    input_y = np.clip(input_y, 0.0, float(height - 1)).astype(np.float32)
    return input_y, input_x, {
        "registration_method": method,
        "left_guard_x": float(left_guard),
        "right_guard_x": float(right_guard),
        "top_guard_y": float(top_guard),
        "bottom_guard_y": float(bottom_guard),
        "target_x_anchors": fixed_x_map.tolist(),
        "sample_x_anchors": moving_x_map.tolist(),
        "target_y_anchors": fixed_y_map.tolist(),
        "sample_y_anchors": moving_y_map.tolist(),
        "interpolation_order": interpolation_order,
        "edge_recovery_amount": edge_recovery,
    }


def apply_registration(
    moving: np.ndarray,
    model: dict,
    method: str = "piecewise-affine",
) -> tuple[np.ndarray, dict]:
    input_y, input_x, mapping = registration_coordinates(model, moving.shape, method=method)
    grid_y = np.broadcast_to(input_y[:, None], moving.shape)
    grid_x = np.broadcast_to(input_x[None, :], moving.shape)
    interpolation_order = int(mapping["interpolation_order"])
    warped = map_coordinates(
        moving,
        [grid_y, grid_x],
        order=interpolation_order,
        mode="nearest",
        prefilter=interpolation_order > 1,
    ).astype(np.float32)
    edge_recovery = float(mapping["edge_recovery_amount"])
    if edge_recovery > 0.0:
        warped = warped + edge_recovery * (
            warped - gaussian_filter(warped, sigma=0.70)
        )
    mask = np.zeros(moving.shape, dtype=np.float32)
    x0 = max(0, int(math.floor(mapping["left_guard_x"])))
    x1 = min(moving.shape[1], int(math.ceil(mapping["right_guard_x"] + 1.0)))
    y0 = max(0, int(math.floor(mapping["top_guard_y"])))
    y1 = min(moving.shape[0], int(math.ceil(mapping["bottom_guard_y"] + 1.0)))
    mask[y0:y1, x0:x1] = 1.0
    mask = gaussian_filter(mask, sigma=5.0)
    registered = moving * (1.0 - mask) + warped * mask
    displacement_y = np.abs(input_y - np.arange(moving.shape[0], dtype=np.float32))
    displacement_x = np.abs(input_x - np.arange(moving.shape[1], dtype=np.float32))
    mapping.update({
        "interpolation": {
            1: "linear spline",
            3: "cubic spline",
            5: "quintic spline",
        }.get(interpolation_order, f"spline order {interpolation_order}"),
        "roi_feather_sigma_px": 5.0,
        "max_abs_displacement_x_px": float(np.max(displacement_x[x0:x1])),
        "max_abs_displacement_y_px": float(np.max(displacement_y[y0:y1])),
        "background_outside_roi_unchanged": True,
    })
    return np.clip(registered, 0.0, 1.0).astype(np.float32), mapping


def edge_sharpness_metrics(image: np.ndarray, model: dict) -> dict:
    """Measure component and outer-wall gradients after robust intensity normalization."""
    normalized = robust_normalize(image)
    gradient_x = sobel(normalized, axis=1) / 8.0
    gradient_y = sobel(normalized, axis=0) / 8.0
    gradient = np.hypot(gradient_x, gradient_y)
    fixed_x = model["horizontal"]["fixed"]
    fixed_y = model["vertical"]["fixed"]
    span_y = fixed_y["bottom"] - fixed_y["top"]
    y0 = max(0, int(math.floor(fixed_y["top"] + 0.12 * span_y)))
    y1 = min(image.shape[0], int(math.ceil(fixed_y["bottom"] - 0.12 * span_y)))
    x0 = max(0, int(math.floor(fixed_x["left_outer"] - 8.0)))
    x1 = min(image.shape[1], int(math.ceil(fixed_x["right_outer"] + 9.0)))
    roi_gradient = gradient[y0:y1, x0:x1]
    outer_samples = []
    for x in (fixed_x["left_outer"], fixed_x["right_outer"]):
        center = int(round(x))
        strip = np.abs(
            gradient_x[y0:y1, max(0, center - 8):min(image.shape[1], center + 9)]
        )
        outer_samples.extend(np.max(strip, axis=1).tolist())
    values = np.asarray(outer_samples, dtype=np.float64)
    return {
        "component_gradient_p95": float(np.percentile(roi_gradient, 95.0)),
        "component_gradient_p99": float(np.percentile(roi_gradient, 99.0)),
        "outer_edge_gradient_median": float(np.median(values)),
        "outer_edge_gradient_p10": float(np.percentile(values, 10.0)),
    }


def observed_envelope(
    image: np.ndarray,
    constraints: list[dict],
    target_model: dict,
) -> dict:
    horizontal = observed_horizontal_envelope(image, constraints, target_model)
    normalized = robust_normalize(image)
    y_profile = vertical_profile(normalized, horizontal)
    pitch = float(np.median([row["pitch_px"] for row in constraints]))
    vertical = {}
    for name, mode in (("top", "positive_gradient"), ("bottom", "negative_gradient")):
        target = target_model["vertical"]["fixed"][name]
        vertical[name] = locate_profile_feature(
            y_profile, target - 4.0 * pitch, target + 4.0 * pitch, mode
        )
    return {"horizontal": horizontal, "vertical": vertical}


def observed_horizontal_envelope(
    image: np.ndarray,
    constraints: list[dict],
    target_model: dict,
) -> dict:
    """Detect the four lateral envelope landmarks near their fixed targets."""
    normalized = robust_normalize(image)
    fixed_interval = target_model["profile_intervals"]["fixed_y_interval"]
    profile = horizontal_profile(normalized, fixed_interval[0], fixed_interval[1])
    groups = constraint_groups(constraints)
    result = {}
    modes = {
        "left_outer": "positive_gradient",
        "left_inner": "peak",
        "right_inner": "peak",
        "right_outer": "negative_gradient",
    }
    for name, target in target_model["horizontal"]["fixed"].items():
        side = "left" if name.startswith("left") else "right"
        radius = max(24.0, 2.5 * groups[side]["pitch"])
        result[name] = locate_profile_feature(
            profile, target - radius, target + radius, modes[name]
        )
    return result


def final_horizontal_envelope_audit(
    image: np.ndarray,
    constraints: list[dict],
    target_model: dict,
) -> dict:
    observed = observed_horizontal_envelope(image, constraints, target_model)
    target = target_model["horizontal"]["fixed"]
    rows = {}
    values = []
    for name, target_value in target.items():
        delta = float(observed[name] - target_value)
        rows[name] = {
            "target_px": float(target_value),
            "observed_px": float(observed[name]),
            "signed_error_px": delta,
            "absolute_error_px": abs(delta),
        }
        values.append(abs(delta))
    errors = np.asarray(values, dtype=np.float64)
    p95 = float(np.percentile(errors, 95.0))
    return {
        "landmarks": rows,
        "summary": {
            "horizontal_envelope_error_p95_px": p95,
            "horizontal_envelope_error_max_px": float(np.max(errors)),
            "guardrail_pass": bool(p95 <= 2.0),
            "acceptance_p95_px_max": 2.0,
        },
    }


def registration_audit(
    moving: np.ndarray,
    registered: np.ndarray,
    constraints: list[dict],
    model: dict,
) -> dict:
    target = {
        "horizontal": model["horizontal"]["fixed"],
        "vertical": model["vertical"]["fixed"],
    }
    before = {
        "horizontal": model["horizontal"]["moving"],
        "vertical": model["vertical"]["moving"],
    }
    after = observed_envelope(registered, constraints, model)

    def errors(observed: dict) -> tuple[dict, np.ndarray]:
        rows = {}
        values = []
        for axis in ("horizontal", "vertical"):
            rows[axis] = {}
            for name, target_value in target[axis].items():
                delta = float(observed[axis][name] - target_value)
                rows[axis][name] = {
                    "target_px": float(target_value),
                    "observed_px": float(observed[axis][name]),
                    "signed_error_px": delta,
                    "absolute_error_px": abs(delta),
                }
                values.append(abs(delta))
        return rows, np.asarray(values, dtype=np.float64)

    before_rows, before_values = errors(before)
    after_rows, after_values = errors(after)
    before_p95 = float(np.percentile(before_values, 95.0))
    after_p95 = float(np.percentile(after_values, 95.0))
    return {
        "before": before_rows,
        "after": after_rows,
        "summary": {
            "envelope_error_before_p95_px": before_p95,
            "envelope_error_after_p95_px": after_p95,
            "envelope_error_before_max_px": float(np.max(before_values)),
            "envelope_error_after_max_px": float(np.max(after_values)),
            "p95_improvement_percent": float(
                100.0 * (before_p95 - after_p95) / max(before_p95, 1e-8)
            ),
            "guardrail_pass": bool(after_p95 <= 2.0 and after_p95 < before_p95),
            "acceptance": {
                "envelope_error_after_p95_px_max": 2.0,
                "must_improve_over_unregistered": True,
            },
        },
    }


def display_gray(image: np.ndarray) -> Image.Image:
    normalized = robust_normalize(image)
    return Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L")


def save_visual_audit(
    outdir: Path,
    fixed: np.ndarray,
    moving: np.ndarray,
    registered: np.ndarray,
    model: dict,
    audit: dict,
) -> None:
    panel_width = 550
    panel_height = 400
    labels = ("blind guide", "before registration", "after registration")
    images = (fixed, moving, registered)
    comparison = Image.new("RGB", (3 * panel_width, panel_height + 28), "black")
    draw = ImageDraw.Draw(comparison)
    for index, (label, image) in enumerate(zip(labels, images)):
        panel = display_gray(image).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        comparison.paste(panel.convert("RGB"), (index * panel_width, 28))
        draw.text((index * panel_width + 8, 7), label, fill=(240, 240, 240))
    comparison.save(outdir / "REGISTRATION_comparison.png")

    overlay = display_gray(registered).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    fixed_h = model["horizontal"]["fixed"]
    fixed_v = model["vertical"]["fixed"]
    after = audit["after"]
    colors = {"target": (0, 255, 80), "registered": (255, 60, 40)}
    for name in ("left_outer", "left_inner", "right_inner", "right_outer"):
        tx = fixed_h[name]
        rx = after["horizontal"][name]["observed_px"]
        draw.line((tx, fixed_v["top"], tx, fixed_v["bottom"]), fill=colors["target"], width=2)
        draw.line((rx, fixed_v["top"], rx, fixed_v["bottom"]), fill=colors["registered"], width=1)
    for name in ("top", "bottom"):
        ty = fixed_v[name]
        ry = after["vertical"][name]["observed_px"]
        draw.line((fixed_h["left_outer"], ty, fixed_h["right_outer"], ty),
                  fill=colors["target"], width=2)
        draw.line((fixed_h["left_outer"], ry, fixed_h["right_outer"], ry),
                  fill=colors["registered"], width=1)
    draw.rectangle((8, 8, 610, 38), fill=(0, 0, 0))
    draw.text((16, 15), "green=guide envelope  red=registered envelope", fill=(255, 255, 255))
    overlay.save(outdir / "REGISTRATION_envelope_overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Original image, used for dimension validation only.")
    parser.add_argument("--guide", type=Path, required=True,
                        help="Blind-denoised fixed image in original coordinates.")
    parser.add_argument("--generated", type=Path, required=True,
                        help="Source-sized visual-only generated moving image.")
    parser.add_argument("--constraints", type=Path, required=True,
                        help="Guide-measured lamella_geometry.csv.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--method", choices=REGISTRATION_METHODS, default="piecewise-affine",
        help="Registration transform and resampling variant.",
    )
    parser.add_argument(
        "--allow-review", action="store_true",
        help="Write a diagnostic candidate even when its envelope guardrail fails.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source = load_gray(args.source)
    guide = load_gray(args.guide)
    generated = load_gray(args.generated)
    if guide.shape != source.shape or generated.shape != source.shape:
        raise ValueError(
            f"All registration inputs must share source dimensions: source={source.shape}, "
            f"guide={guide.shape}, generated={generated.shape}"
        )
    constraints = load_constraints(args.constraints)
    model = estimate_registration(guide, generated, constraints)
    registered, mapping = apply_registration(generated, model, method=args.method)
    audit = registration_audit(generated, registered, constraints, model)
    sharpness = {
        "moving": edge_sharpness_metrics(generated, model),
        "registered": edge_sharpness_metrics(registered, model),
    }
    if not audit["summary"]["guardrail_pass"] and not args.allow_review:
        raise RuntimeError(f"Registration envelope audit failed: {audit['summary']}")

    output_png = args.outdir / "GENERATIVE_registered_to_guide.png"
    Image.fromarray(np.rint(registered * 255.0).astype(np.uint8), mode="L").save(output_png)
    imwrite(
        args.outdir / "GENERATIVE_registered_to_guide_16bit.tif",
        np.rint(registered * 65535.0).astype(np.uint16),
        photometric="minisblack",
        description=("VISUAL_ONLY: constrained coarse-to-fine registration to the "
                     "same-coordinate blind-denoised guide."),
    )
    payload = {
        "completed": True,
        "method": args.method,
        "fixed_geometry_source": str(args.guide),
        "raw_dimension_reference": str(args.source),
        "moving_visual_source": str(args.generated),
        "constraint_source": str(args.constraints),
        "output": str(output_png),
        "dimensions": {"width": source.shape[1], "height": source.shape[0]},
        "model": model,
        "mapping": mapping,
        "audit": audit,
        "sharpness": sharpness,
        "metrology_policy": {
            "generated_pixels_are_measurement_truth": False,
            "registration_transform_applies_to": "generated visual candidate only",
            "numeric_constraints_remain_in": "original source coordinates",
            "hard_projection_required_after_registration": True,
        },
    }
    (args.outdir / "registration_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_visual_audit(args.outdir, guide, generated, registered, model, audit)
    print(output_png, flush=True)


if __name__ == "__main__":
    main()
