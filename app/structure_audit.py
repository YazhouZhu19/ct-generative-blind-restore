"""Structure masks, measurement audits, and image comparisons.

These helpers support the accepted native-coordinate warp. They never modify an
image and deliberately reject inputs that are not already on the source canvas.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.metrics import structural_similarity

import generative_shape_constraint as constraint
import length_optimize as length
import pipeline as base


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 8 or float(np.std(a)) < 1e-10 or float(np.std(b)) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def subpixel_maximum(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return float(index)
    left, center, right = (float(value) for value in values[index - 1 : index + 2])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index + np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))


def require_native_coordinates(
    source: np.ndarray,
    guide: np.ndarray,
    generated: np.ndarray,
) -> None:
    if source.shape != guide.shape or source.shape != generated.shape:
        raise ValueError(
            "Structure guidance requires an already source-sized generation: "
            f"source={source.shape}, guide={guide.shape}, generated={generated.shape}"
        )


def build_constraint_fields(
    image_shape: tuple[int, int],
    layers: list[dict],
    feather_sigma: float = 2.0,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """Rasterize measured paths, finite widths, endpoints, and writable cells."""
    height, width = image_shape
    side_cells = {
        "left": np.zeros(image_shape, dtype=np.float32),
        "right": np.zeros(image_shape, dtype=np.float32),
    }
    centerline = np.zeros(image_shape, dtype=np.float32)
    body = np.zeros(image_shape, dtype=np.float32)
    boundary = np.zeros(image_shape, dtype=np.float32)
    endpoint = np.zeros(image_shape, dtype=np.float32)
    endpoint_cell = np.zeros(image_shape, dtype=np.float32)
    confidence = np.zeros(image_shape, dtype=np.float32)

    for row in layers:
        side = str(row["side"])
        path = np.asarray(row["path_x"], dtype=np.float32)
        pitch = max(float(row["pitch_px"]), 4.0)
        measured_width = row.get("width_median_px")
        lamella_width = float(
            measured_width if measured_width is not None else max(0.18 * pitch, 1.2)
        )
        half_width = max(0.5 * lamella_width, 0.6)
        top = float(row["top_y_px"])
        bottom = float(row["bottom_y_px"])
        row_confidence = float(row.get("constraint_confidence", 1.0))
        y0 = max(0, int(math.floor(top - 9.0)))
        y1 = min(height - 1, int(math.ceil(bottom + 9.0)))
        for y in range(y0, y1 + 1):
            center = float(path[y])
            x0 = max(0, int(math.floor(center - 0.58 * pitch - 4.0)))
            x1 = min(width, int(math.ceil(center + 0.58 * pitch + 4.0)) + 1)
            if x1 <= x0:
                continue
            dx = np.arange(x0, x1, dtype=np.float32) - center
            horizontal_cell = smoothstep(
                np.clip((0.58 * pitch + 2.0 - np.abs(dx)) / 2.0, 0.0, 1.0)
            )
            vertical_cell = min(
                np.clip((y - (top - 9.0)) / 5.0, 0.0, 1.0),
                np.clip(((bottom + 9.0) - y) / 5.0, 0.0, 1.0),
            )
            current_cell = horizontal_cell * float(smoothstep(np.asarray(vertical_cell)))
            side_cells[side][y, x0:x1] = np.maximum(
                side_cells[side][y, x0:x1], current_cell
            )

            axial_gate = min(
                np.clip((y - top + 2.0) / 4.0, 0.0, 1.0),
                np.clip((bottom - y + 2.0) / 4.0, 0.0, 1.0),
            )
            axial_gate = float(smoothstep(np.asarray(axial_gate)))
            current_center = np.exp(-0.5 * np.square(dx / max(0.65, 0.23 * lamella_width)))
            current_body = np.exp(
                -np.power(np.abs(dx) / max(half_width + 0.55, 1.05), 4.0)
            )
            current_boundary = np.maximum(
                np.exp(-0.5 * np.square((dx - half_width) / 0.72)),
                np.exp(-0.5 * np.square((dx + half_width) / 0.72)),
            )
            centerline[y, x0:x1] = np.maximum(
                centerline[y, x0:x1], axial_gate * current_center
            )
            body[y, x0:x1] = np.maximum(body[y, x0:x1], axial_gate * current_body)
            boundary[y, x0:x1] = np.maximum(
                boundary[y, x0:x1], axial_gate * current_boundary
            )
            confidence[y, x0:x1] = np.maximum(
                confidence[y, x0:x1], row_confidence * horizontal_cell * axial_gate
            )

            for end_y in (top, bottom):
                end_gate = math.exp(-0.5 * ((y - end_y) / 1.35) ** 2)
                cell_end_gate = math.exp(-0.5 * ((y - end_y) / 5.5) ** 2)
                end_profile = np.exp(
                    -np.power(np.abs(dx) / max(half_width + 1.1, 1.4), 4.0)
                )
                endpoint[y, x0:x1] = np.maximum(
                    endpoint[y, x0:x1], end_gate * end_profile
                )
                endpoint_cell[y, x0:x1] = np.maximum(
                    endpoint_cell[y, x0:x1], cell_end_gate * horizontal_cell
                )

    for side in side_cells:
        side_cells[side] = np.clip(
            gaussian_filter(side_cells[side], sigma=feather_sigma), 0.0, 1.0
        ).astype(np.float32)
    centerline = np.clip(gaussian_filter(centerline, sigma=0.45), 0.0, 1.0)
    body = np.clip(gaussian_filter(body, sigma=0.38), 0.0, 1.0)
    boundary = np.clip(gaussian_filter(boundary, sigma=0.45), 0.0, 1.0)
    endpoint = np.clip(gaussian_filter(endpoint, sigma=(0.55, 0.40)), 0.0, 1.0)
    endpoint_cell = np.clip(
        gaussian_filter(endpoint_cell, sigma=(0.55, 0.55)), 0.0, 1.0
    )
    analytic_support = np.maximum.reduce((centerline, boundary, endpoint)).astype(np.float32)
    stack = np.maximum(side_cells["left"], side_cells["right"])
    confidence = np.where(stack > 1e-5, np.maximum(confidence, 0.58), 0.0)
    return {
        "side_cells": side_cells,
        "stack": stack.astype(np.float32),
        "centerline": centerline.astype(np.float32),
        "body": body.astype(np.float32),
        "boundary": boundary.astype(np.float32),
        "endpoint": endpoint.astype(np.float32),
        "endpoint_cell": endpoint_cell.astype(np.float32),
        "analytic_support": analytic_support,
        "confidence": confidence.astype(np.float32),
    }


def normalized_on_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    lower, upper = (float(value) for value in np.percentile(values, (1.0, 99.0)))
    return np.clip((image - lower) / max(upper - lower, 1e-8), 0.0, 1.0)


def ridge_offsets(
    image: np.ndarray,
    layers: list[dict],
    row_step: int = 11,
) -> np.ndarray:
    offsets: list[float] = []
    for row in layers:
        path = np.asarray(row["path_x"], dtype=np.float32)
        pitch = float(row["pitch_px"])
        margin = max(8, int(round(0.08 * float(row["length_px"]))))
        y0 = max(0, int(math.ceil(float(row["top_y_px"]))) + margin)
        y1 = min(image.shape[0], int(math.floor(float(row["bottom_y_px"]))) - margin + 1)
        for y in range(y0, y1, max(1, int(row_step))):
            center = float(path[y])
            radius = max(3, int(math.floor(0.38 * pitch)))
            x0 = max(1, int(math.floor(center)) - radius)
            x1 = min(image.shape[1] - 1, int(math.ceil(center)) + radius + 1)
            if x1 - x0 < 5:
                continue
            profile = gaussian_filter1d(image[y, x0:x1], sigma=0.65)
            highpass = profile - gaussian_filter1d(profile, sigma=max(2.0, 0.30 * pitch))
            peak_index = int(np.argmax(highpass))
            peak = x0 + subpixel_maximum(highpass, peak_index)
            offsets.append(abs(float(peak) - center))
    return np.asarray(offsets, dtype=np.float64)


def fast_alignment_metrics(
    guide: np.ndarray,
    candidate: np.ndarray,
    generated: np.ndarray,
    layers: list[dict],
    fields: dict,
) -> dict:
    region = np.asarray(fields["stack"]) >= 0.45
    guide_n = normalized_on_mask(guide, region)
    candidate_n = normalized_on_mask(candidate, region)
    guide_mid = gaussian_filter(guide_n, sigma=0.75) - gaussian_filter(guide_n, sigma=4.0)
    candidate_mid = gaussian_filter(candidate_n, sigma=0.75) - gaussian_filter(
        candidate_n, sigma=4.0
    )
    guide_gradient = np.hypot(*np.gradient(gaussian_filter(guide_n, sigma=0.85)))
    candidate_gradient = np.hypot(*np.gradient(gaussian_filter(candidate_n, sigma=0.85)))
    offsets = ridge_offsets(candidate, layers)
    global_ssim = float(structural_similarity(generated, candidate, data_range=1.0))
    outside = np.asarray(fields["stack"]) <= 1e-5
    return {
        "guide_mid_frequency_correlation": safe_correlation(
            guide_mid[region], candidate_mid[region]
        ),
        "guide_gradient_magnitude_correlation": safe_correlation(
            guide_gradient[region], candidate_gradient[region]
        ),
        "ridge_center_absolute_offset_median_px": float(np.median(offsets)),
        "ridge_center_absolute_offset_p95_px": float(np.percentile(offsets, 95.0)),
        "ridge_sample_count": int(offsets.size),
        "global_ssim_to_accepted_generation": global_ssim,
        "outside_writable_max_abs_change": float(
            np.max(np.abs(candidate[outside] - generated[outside]))
        ),
    }


def endpoint_and_width_audit(
    image: np.ndarray,
    layers: list[dict],
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
) -> dict:
    endpoint_errors: list[float] = []
    width_relative_errors: list[float] = []
    rows: list[dict] = []
    for row in layers:
        hint = length.EndpointMeasurement(
            top=float(row["top_y_px"]),
            bottom=float(row["bottom_y_px"]),
            length=float(row["length_px"]),
            top_spread=0.0,
            bottom_spread=0.0,
            uncertainty=float(row["endpoint_uncertainty_px"]),
            top_snr=float(row["top_snr"]),
            bottom_snr=float(row["bottom_snr"]),
        )
        measured = length.measure_endpoints(
            image,
            float(row["center_x_px"]),
            float(row["pitch_px"]),
            top_range,
            bottom_range,
            hint=hint,
            association_radius=6,
            path_x=np.asarray(row["path_x"]),
        )
        top_error = abs(float(measured.top) - float(row["top_y_px"]))
        bottom_error = abs(float(measured.bottom) - float(row["bottom_y_px"]))
        endpoint_errors.extend((top_error, bottom_error))
        width_stats = constraint.measure_layer_width(
            image,
            np.asarray(row["path_x"]),
            float(row["top_y_px"]),
            float(row["bottom_y_px"]),
            float(row["pitch_px"]),
        )
        target_width = row.get("width_median_px")
        measured_width = width_stats["median"]
        width_error = None
        if target_width is not None and measured_width is not None:
            width_error = abs(float(measured_width) - float(target_width)) / max(
                float(target_width), 1e-8
            )
            width_relative_errors.append(width_error)
        rows.append({
            "side": row["side"],
            "layer_id": row["layer_id"],
            "target_top_y_px": float(row["top_y_px"]),
            "measured_top_y_px": float(measured.top),
            "top_absolute_error_px": top_error,
            "target_bottom_y_px": float(row["bottom_y_px"]),
            "measured_bottom_y_px": float(measured.bottom),
            "bottom_absolute_error_px": bottom_error,
            "target_width_px": target_width,
            "measured_width_px": measured_width,
            "width_relative_error": width_error,
        })
    endpoint_values = np.asarray(endpoint_errors, dtype=np.float64)
    width_values = np.asarray(width_relative_errors, dtype=np.float64)
    return {
        "summary": {
            "endpoint_absolute_error_median_px": float(np.median(endpoint_values)),
            "endpoint_absolute_error_p95_px": float(np.percentile(endpoint_values, 95.0)),
            "width_relative_error_median": (
                float(np.median(width_values)) if width_values.size else None
            ),
            "width_relative_error_p95": (
                float(np.percentile(width_values, 95.0)) if width_values.size else None
            ),
            "layer_count": len(layers),
        },
        "rows": rows,
    }


def save_overlay(path: Path, generated: np.ndarray, layers: list[dict]) -> None:
    background = Image.fromarray(
        np.rint(np.clip(generated, 0.0, 1.0) * 255).astype(np.uint8), mode="L"
    ).convert("RGB")
    draw = ImageDraw.Draw(background)
    for row in layers:
        path_x = np.asarray(row["path_x"])
        width = float(row.get("width_median_px") or max(1.2, 0.18 * float(row["pitch_px"])))
        y0 = max(0, int(math.ceil(float(row["top_y_px"]))))
        y1 = min(generated.shape[0] - 1, int(math.floor(float(row["bottom_y_px"]))))
        rows = list(range(y0, y1 + 1, 4))
        if rows and rows[-1] != y1:
            rows.append(y1)
        center_points = [(float(path_x[y]), float(y)) for y in rows]
        left_points = [(float(path_x[y] - 0.5 * width), float(y)) for y in rows]
        right_points = [(float(path_x[y] + 0.5 * width), float(y)) for y in rows]
        if len(rows) >= 2:
            draw.line(center_points, fill=(0, 255, 255), width=1)
            draw.line(left_points, fill=(40, 255, 70), width=1)
            draw.line(right_points, fill=(40, 255, 70), width=1)
        for y in (y0, y1):
            x = float(path_x[y])
            draw.line(
                (x - 0.5 * width - 1.0, y, x + 0.5 * width + 1.0, y),
                fill=(255, 60, 45),
                width=2,
            )
    background.save(path)


def labeled_panel(image: np.ndarray, size: tuple[int, int], label: str) -> Image.Image:
    return base.panel(image, size, label).convert("L")


def save_comparison(
    path: Path,
    generated: np.ndarray,
    guide: np.ndarray,
    refined: np.ndarray,
) -> None:
    size = (720, 524)
    panels = [
        labeled_panel(generated, size, "accepted source-sized generation"),
        labeled_panel(guide, size, "blind-denoised structural guide"),
        labeled_panel(refined, size, "native structure-guided result"),
    ]
    canvas = Image.new("L", (size[0] * 3, size[1]), 0)
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * size[0], 0))
    canvas.save(path)
