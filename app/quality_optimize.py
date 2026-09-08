#!/usr/bin/env python3
"""Boundary-preserving quality refinement for the blind denoiser.

The original edge/structure enhancement is selected unchanged, then a separate
sub-pixel geometry stage moves enhanced pixels only. Raw data supplies endpoint
coordinates but no raw pixels are written back. A final weak, structure-aware
cleanup suppresses residual grain/fog away from strong edges. Final width,
endpoint, length, SSIM, edge-retention and noise guardrails must all pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, sobel
from scipy.signal import find_peaks, peak_widths
from skimage import exposure
from skimage.metrics import structural_similarity
from tifffile import imwrite

import pipeline as base
import length_optimize as length


def analyze_region(img: np.ndarray, roi: base.Roi) -> dict:
    crop = img[roi.slices()]
    # Keep the metrology scale identical to pipeline.profile_metrics.  A
    # different sigma can cause a weak ridge to be associated with its
    # neighbour and report a discontinuous, artificial FWHM jump.
    profile = gaussian_filter1d(crop.mean(axis=0), sigma=1.2)
    prominence = max(float(np.std(profile)) * 0.20, 2.0 / 65535.0)
    peaks, props = find_peaks(profile, distance=12, prominence=prominence)
    peaks = peaks[(peaks > 10) & (peaks < profile.size - 10)]
    widths = peak_widths(profile, peaks, rel_height=0.5)[0] if len(peaks) else np.asarray([])
    grad_profile = np.abs(np.gradient(profile))
    acutance = []
    for p in peaks:
        lo, hi = max(0, p - 5), min(profile.size, p + 6)
        acutance.append(float(np.max(grad_profile[lo:hi])))

    smoothed = gaussian_filter(crop, sigma=0.8)
    grad = np.hypot(sobel(smoothed, axis=0), sobel(smoothed, axis=1))
    flat = grad < np.percentile(grad, 42.0)
    highpass = crop - gaussian_filter(crop, sigma=1.1)
    roughness = float(np.sqrt(np.mean(np.square(highpass[flat]))))
    detrended = profile - gaussian_filter1d(profile, sigma=10.0)
    return {
        "count": int(len(peaks)),
        "positions": [int(p + roi.x0) for p in peaks],
        "median_fwhm_px": float(np.median(widths)) if len(widths) else None,
        "median_edge_acutance": float(np.median(acutance)) if acutance else None,
        "profile_local_contrast_rms": float(np.sqrt(np.mean(np.square(detrended)))),
        "flat_region_roughness": roughness,
    }


def nearest_shift(before: dict, after: dict) -> float:
    a = np.asarray(before["positions"], dtype=np.float64)
    b = np.asarray(after["positions"], dtype=np.float64)
    if not len(a) or not len(b):
        return math.inf
    shifts = []
    for p in a:
        q = float(b[np.argmin(np.abs(b - p))])
        if abs(q - p) <= 5:
            shifts.append(abs(q - p))
    return float(max(shifts)) if shifts else math.inf


def build_boundary_references(
    source: np.ndarray,
    guide: np.ndarray,
    left: base.Roi,
    right: base.Roi,
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
) -> list[dict]:
    """Measure raw endpoints once and retain the matched ridge paths."""
    refs: list[dict] = []
    for side, roi in (("left", left), ("right", right)):
        centers, pitch, _ = length.detect_centers(guide, roi)
        for index, center in enumerate(centers, start=1):
            path_x = length.track_ridge(guide, center, pitch, top_range[0], bottom_range[1])
            raw = length.measure_endpoints(
                source, center, pitch, top_range, bottom_range, path_x=path_x
            )
            refs.append({
                "side": side,
                "layer_id": f"{side[0].upper()}{index:02d}",
                "center": float(center),
                "pitch": float(pitch),
                "path_x": path_x,
                "raw": raw,
            })
    return refs


def boundary_geometry(
    candidate: np.ndarray,
    references: list[dict],
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
) -> tuple[dict, list[dict]]:
    """Return robust and worst-case geometry drift against raw evidence."""
    rows: list[dict] = []
    endpoint_abs: list[float] = []
    length_abs: list[float] = []
    endpoint_abs_all: list[float] = []
    length_abs_all: list[float] = []
    for ref in references:
        raw = ref["raw"]
        measured = length.measure_endpoints(
            candidate,
            ref["center"],
            ref["pitch"],
            top_range,
            bottom_range,
            hint=raw,
            association_radius=3,
            path_x=ref["path_x"],
        )
        top_shift = float(measured.top - raw.top)
        bottom_shift = float(measured.bottom - raw.bottom)
        length_delta = float(measured.length - raw.length)
        axis = np.arange(candidate.shape[0], dtype=np.float64)
        top_x = float(np.interp(raw.top, axis, ref["path_x"]))
        bottom_x = float(np.interp(raw.bottom, axis, ref["path_x"]))
        reliable = bool(raw.uncertainty <= 3.0 and min(raw.top_snr, raw.bottom_snr) >= 4.0)
        endpoint_abs_all.extend((abs(top_shift), abs(bottom_shift)))
        length_abs_all.append(abs(length_delta))
        if reliable:
            endpoint_abs.extend((abs(top_shift), abs(bottom_shift)))
            length_abs.append(abs(length_delta))
        rows.append({
            "side": ref["side"],
            "layer_id": ref["layer_id"],
            "center_x_px": round(ref["center"], 4),
            "top_x_px": round(top_x, 4),
            "bottom_x_px": round(bottom_x, 4),
            "source_top_y_px": round(raw.top, 4),
            "source_bottom_y_px": round(raw.bottom, 4),
            "source_length_px": round(raw.length, 4),
            "enhanced_top_y_px": round(measured.top, 4),
            "enhanced_bottom_y_px": round(measured.bottom, 4),
            "enhanced_length_px": round(measured.length, 4),
            "top_shift_px": round(top_shift, 4),
            "bottom_shift_px": round(bottom_shift, 4),
            "length_delta_px": round(length_delta, 4),
            "source_length_uncertainty_px": round(raw.uncertainty, 4),
            "reference_quality": "pass" if reliable else "review",
        })
    if not endpoint_abs or not length_abs:
        raise RuntimeError("No reliable raw endpoint references were detected")
    endpoint_values = np.asarray(endpoint_abs, dtype=np.float64)
    length_values = np.asarray(length_abs, dtype=np.float64)
    endpoint_values_all = np.asarray(endpoint_abs_all, dtype=np.float64)
    length_values_all = np.asarray(length_abs_all, dtype=np.float64)
    summary = {
        "layer_count": len(rows),
        "reliable_layer_count": int(sum(row["reference_quality"] == "pass" for row in rows)),
        "review_required_count": int(sum(row["reference_quality"] == "review" for row in rows)),
        "endpoint_shift_abs_median_px": float(np.median(endpoint_values)),
        "endpoint_shift_abs_p95_px": float(np.percentile(endpoint_values, 95.0)),
        "endpoint_shift_abs_max_px": float(np.max(endpoint_values)),
        "length_delta_abs_median_px": float(np.median(length_values)),
        "length_delta_abs_p95_px": float(np.percentile(length_values, 95.0)),
        "length_delta_abs_max_px": float(np.max(length_values)),
        "all_layers_endpoint_shift_abs_max_px_diagnostic": float(np.max(endpoint_values_all)),
        "all_layers_length_delta_abs_max_px_diagnostic": float(np.max(length_values_all)),
    }
    return summary, rows


def boundary_guardrail(summary: dict) -> bool:
    return bool(
        summary["endpoint_shift_abs_p95_px"] <= 0.35
        and summary["endpoint_shift_abs_max_px"] <= 0.75
        and summary["length_delta_abs_p95_px"] <= 0.50
        and summary["length_delta_abs_max_px"] <= 1.00
    )


def endpoint_geometry_warp(
    candidate: np.ndarray,
    references: list[dict],
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
    iterations: int = 1,
    radius_y: int = 1,
    radius_x: int = 1,
    max_shift: float = 0.35,
    large_shift_trigger: float = 3.0,
    large_shift_cap: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Move enhanced endpoints to raw coordinates without copying raw pixels.

    A smooth local displacement field is fitted from reliable sub-pixel
    endpoint offsets.  The already enhanced image is resampled through this
    field, so its denoising, contrast, and sharpening remain intact.
    """
    out = candidate.astype(np.float32, copy=True)
    h, w = out.shape
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    history: list[dict] = []
    for iteration in range(max(1, int(iterations))):
        summary, rows = boundary_geometry(out, references, top_range, bottom_range)
        history.append({"iteration": iteration, **summary})
        numerator = np.zeros_like(out, dtype=np.float32)
        support = np.zeros_like(out, dtype=np.float32)
        applied = 0
        for ref, row in zip(references, rows):
            if row["reference_quality"] != "pass":
                continue
            for endpoint in ("top", "bottom"):
                delta = float(row[f"enhanced_{endpoint}_y_px"] - row[f"source_{endpoint}_y_px"])
                if abs(delta) < 0.005:
                    continue
                cap = large_shift_cap if abs(delta) >= large_shift_trigger else max_shift
                delta = float(np.clip(delta, -cap, cap))
                target_y = float(row[f"source_{endpoint}_y_px"])
                cy = int(round(target_y))
                y0, y1 = max(0, cy - 3 * radius_y), min(h, cy + 3 * radius_y + 1)
                if y1 <= y0:
                    continue
                yy = np.arange(y0, y1, dtype=np.int32)
                center_x = ref["path_x"][yy].astype(np.float32)
                for local_index, image_y in enumerate(yy):
                    cx = int(round(float(center_x[local_index])))
                    x0, x1 = max(0, cx - 3 * radius_x), min(w, cx + 3 * radius_x + 1)
                    if x1 <= x0:
                        continue
                    xx = np.arange(x0, x1, dtype=np.float32)
                    weight_y = math.exp(-0.5 * ((float(image_y) - target_y) / radius_y) ** 2)
                    weight_x = np.exp(-0.5 * np.square((xx - center_x[local_index]) / radius_x))
                    weight = (weight_y * weight_x).astype(np.float32)
                    numerator[image_y, x0:x1] += weight * delta
                    support[image_y, x0:x1] += weight
                applied += 1
        displacement_y = numerator / np.maximum(1.0, support)
        max_displacement = float(np.max(np.abs(displacement_y)))
        if applied == 0 or max_displacement < 0.005:
            break
        # If an enhanced edge was detected delta pixels below the raw edge,
        # output at the raw coordinate samples the enhanced image delta pixels
        # below. Cubic interpolation preserves the enhanced edge profile.
        out = map_coordinates(
            out,
            [grid_y + displacement_y, grid_x],
            order=3,
            mode="nearest",
            prefilter=True,
        )
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        history[-1]["applied_endpoint_count"] = applied
        history[-1]["max_displacement_px"] = max_displacement
    final_summary, _ = boundary_geometry(out, references, top_range, bottom_range)
    return out, {
        "method": "sub-pixel local displacement of enhanced pixels only",
        "raw_pixel_writeback": False,
        "iterations_requested": iterations,
        "radius_y": radius_y,
        "radius_x": radius_x,
        "max_shift_px": max_shift,
        "large_shift_trigger_px": large_shift_trigger,
        "large_shift_cap_px": large_shift_cap,
        "history": history,
        "final_boundary_geometry": final_summary,
    }


def endpoint_coordinate_protection(
    shape: tuple[int, int],
    references: list[dict],
    radius_y: int = 5,
    radius_x: int = 3,
) -> np.ndarray:
    """Protect endpoint coordinates without reading or copying raw pixels."""
    h, w = shape
    mask = np.zeros(shape, dtype=np.float32)
    axis = np.arange(h, dtype=np.float64)
    for ref in references:
        for y in (ref["raw"].top, ref["raw"].bottom):
            yi = int(round(y))
            for yy in range(max(0, yi - radius_y), min(h, yi + radius_y + 1)):
                xi = int(round(float(np.interp(yy, axis, ref["path_x"]))))
                mask[yy, max(0, xi - radius_x) : min(w, xi + radius_x + 1)] = 1.0
    core = mask > 0
    mask = gaussian_filter(mask, sigma=(1.0, 0.7))
    # The halo is soft, but the measured endpoint support itself is immutable.
    mask[core] = 1.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def structure_aware_fog_cleanup(
    image: np.ndarray,
    references: list[dict],
    target: base.Roi,
    fine_strength: float,
    haze_strength: float,
    gradient_percentile: float = 45.0,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Weakly suppress grain and mist only in low-structure regions.

    The operation is applied after the original enhancement and geometry warp.
    Strong gradients and coordinate-only endpoint supports are excluded.  Both
    filtered terms are computed from the enhanced image, never the raw image.
    """
    ys, xs = target.slices()
    crop = image[ys, xs]
    smooth_for_gate = gaussian_filter(crop, sigma=0.70)
    gy, gx = np.gradient(smooth_for_gate)
    gradient = np.hypot(gx, gy)
    positive = crop[crop > 0]
    intensity_floor = float(np.percentile(positive, 5.0)) if positive.size else 0.0
    gradient_sample = gradient[crop > intensity_floor]
    threshold = float(np.percentile(gradient_sample, gradient_percentile)) + 1e-8
    low_structure_gate = np.exp(-np.square(gradient / threshold)).astype(np.float32)
    endpoint_protection = endpoint_coordinate_protection(image.shape, references)[ys, xs]
    cleanup_gate = low_structure_gate * (1.0 - endpoint_protection)

    fine = gaussian_filter(crop, sigma=0.90)
    medium = gaussian_filter(crop, sigma=2.20)
    delta = cleanup_gate * (
        float(fine_strength) * (fine - crop)
        + float(haze_strength) * (medium - fine)
    )
    sigma_n = base.noise_sigma_mad(crop)
    cap = max(0.65 * sigma_n, 1.0 / 65535.0)
    cleaned = np.clip(crop + np.clip(delta, -cap, cap), 0.0, 1.0).astype(np.float32)
    # Feather the cleanup delta at the target boundary to avoid a rectangular
    # seam in the complete 16-bit image.
    out = base.feather_insert(image, cleaned, target, ramp=18)
    info = {
        "method": "edge-gated enhanced-domain fine noise and mid-scale fog suppression",
        "raw_pixel_writeback": False,
        "fine_sigma": 0.90,
        "medium_sigma": 2.20,
        "fine_strength": float(fine_strength),
        "haze_strength": float(haze_strength),
        "gradient_percentile": float(gradient_percentile),
        "gradient_threshold": threshold,
        "cleanup_gate_mean": float(np.mean(cleanup_gate)),
        "estimated_noise_sigma_normalized": sigma_n,
        "change_cap_normalized": cap,
    }
    return out.astype(np.float32), info, gradient


def low_structure_roughness(
    image: np.ndarray,
    target: base.Roi,
    reference_gradient: np.ndarray,
) -> float:
    crop = image[target.slices()]
    flat = reference_gradient < np.percentile(reference_gradient, 45.0)
    residual = crop - gaussian_filter(crop, sigma=1.10)
    return float(np.sqrt(np.mean(np.square(residual[flat]))))


def quality_refine_crop(
    source: np.ndarray,
    edge_amount: float,
    structure_gain: float,
) -> np.ndarray:
    # Horizontal-only blur isolates detail normal to the nearly vertical layers.
    blur_x = gaussian_filter(source, sigma=(0.30, 1.00))
    detail_x = source - blur_x
    smooth = gaussian_filter(source, sigma=0.65)
    gx = np.abs(sobel(smooth, axis=1))
    threshold = float(np.percentile(gx, 72.0)) + 1e-8
    softness = max(0.22 * threshold, 1e-8)
    edge_gate = 1.0 / (1.0 + np.exp(np.clip(-(gx - threshold) / softness, -30.0, 30.0)))
    edge_gate = gaussian_filter(edge_gate, sigma=0.45)

    # Difference of Gaussians improves plate/background separation without a
    # full histogram remap of the 16-bit measurement values.
    structure = gaussian_filter(source, sigma=0.85) - gaussian_filter(source, sigma=7.0)
    candidate = source + edge_amount * edge_gate * detail_x + structure_gain * structure

    sigma_n = base.noise_sigma_mad(source)
    cap = max(1.15 * sigma_n, 2.0 / 65535.0)
    candidate = source + np.clip(candidate - source, -cap, cap)
    return np.clip(candidate, 0.0, 1.0).astype(np.float32)


def evaluate(
    candidate: np.ndarray,
    baseline: np.ndarray,
    left: base.Roi,
    right: base.Roi,
    baseline_metrics: dict,
    source_metrics: dict,
) -> dict:
    metrics = {"left": analyze_region(candidate, left), "right": analyze_region(candidate, right)}
    edge_gain = float(np.median([
        metrics[s]["median_edge_acutance"] / baseline_metrics[s]["median_edge_acutance"]
        for s in ("left", "right")
    ]))
    contrast_gain = float(np.median([
        metrics[s]["profile_local_contrast_rms"] / baseline_metrics[s]["profile_local_contrast_rms"]
        for s in ("left", "right")
    ]))
    roughness_ratio = float(np.median([
        metrics[s]["flat_region_roughness"] / baseline_metrics[s]["flat_region_roughness"]
        for s in ("left", "right")
    ]))
    count_ok = all(abs(metrics[s]["count"] - baseline_metrics[s]["count"]) <= 1 for s in ("left", "right"))
    shift = max(nearest_shift(baseline_metrics[s], metrics[s]) for s in ("left", "right"))
    width_change = max(
        abs(metrics[s]["median_fwhm_px"] / baseline_metrics[s]["median_fwhm_px"] - 1.0)
        for s in ("left", "right")
    )
    source_width_change = max(
        abs(metrics[s]["median_fwhm_px"] / source_metrics[s]["median_fwhm_px"] - 1.0)
        for s in ("left", "right")
    )
    union = base.Roi(min(left.y0, right.y0), max(left.y1, right.y1), left.x0, right.x1)
    ssim = float(structural_similarity(baseline[union.slices()], candidate[union.slices()], data_range=1.0))
    guardrail = bool(
        count_ok
        and shift <= 1.0
        and width_change <= 0.05
        and roughness_ratio <= 1.03
        and ssim >= 0.985
    )
    score = 2.4 * (edge_gain - 1.0) + 1.0 * (contrast_gain - 1.0) - 2.0 * max(0.0, roughness_ratio - 1.0)
    return {
        **metrics,
        "edge_gain": edge_gain,
        "contrast_gain": contrast_gain,
        "roughness_ratio": roughness_ratio,
        "max_center_shift_px": shift,
        "max_fwhm_relative_change": float(width_change),
        "max_fwhm_relative_change_vs_source": float(source_width_change),
        "ssim_vs_blind_baseline": ssim,
        "guardrail_pass": guardrail,
        "score": float(score),
    }


def display_version(img: np.ndarray, roi: base.Roi) -> np.ndarray:
    out = img.copy()
    crop = img[roi.slices()]
    lo, hi = (float(v) for v in np.percentile(crop, (0.5, 99.7)))
    normalized = np.clip((crop - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    local = exposure.equalize_adapthist(normalized, kernel_size=72, clip_limit=0.006)
    out[roi.slices()] = 0.76 * normalized + 0.24 * local
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def save_boundary_overlay(
    path: Path,
    image: np.ndarray,
    rows: list[dict],
    roi: base.Roi,
) -> None:
    """Draw raw (green) and enhanced (red) endpoint detections together."""
    view = image[roi.slices()]
    lo, hi = (float(v) for v in np.percentile(view, (0.5, 99.7)))
    u8 = np.rint(np.clip((view - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
    canvas = Image.fromarray(u8, mode="L").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((4, 4, 318, 25), fill=(0, 0, 0))
    draw.text((8, 8), "green=raw  red=enhanced  orange=raw review", fill=(255, 255, 255), font=font)
    for row in rows:
        raw_color = (255, 165, 0) if row["reference_quality"] == "review" else (40, 255, 90)
        for endpoint, x_key in (("top", "top_x_px"), ("bottom", "bottom_x_px")):
            x = float(row[x_key]) - roi.x0
            raw_y = float(row[f"source_{endpoint}_y_px"]) - roi.y0
            enhanced_y = float(row[f"enhanced_{endpoint}_y_px"]) - roi.y0
            if 0 <= x < canvas.width and 0 <= raw_y < canvas.height:
                draw.line((x - 6, raw_y, x + 6, raw_y), fill=raw_color, width=3)
            if 0 <= x < canvas.width and 0 <= enhanced_y < canvas.height:
                draw.line((x - 4, enhanced_y, x + 4, enhanced_y), fill=(255, 40, 40), width=1)
    canvas.save(path)


def panel(img: np.ndarray, title: str, size: tuple[int, int] = (760, 550)) -> Image.Image:
    nz = img[img > 0]
    values = nz if nz.size else img.reshape(-1)
    lo, hi = (float(v) for v in np.percentile(values, (0.5, 99.7)))
    u8 = np.rint(np.clip((img - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size[0], size[1] + 28), 0)
    canvas.paste(im, ((size[0] - im.width) // 2, 28 + (size[1] - im.height) // 2))
    ImageDraw.Draw(canvas).text((8, 8), title, fill=255, font=ImageFont.load_default())
    return canvas


def save_comparison(
    path: Path,
    observed: np.ndarray,
    baseline: np.ndarray,
    original_enhancement: np.ndarray,
    final: np.ndarray,
    roi: base.Roi,
) -> None:
    # Use the same target crop in every panel so the subtle edge/noise changes
    # remain visible at normal screen zoom. The separately saved preview is the
    # complete 2200 x 1600 image.
    ys, xs = roi.slices()
    items = [
        panel(observed[ys, xs], "Original 16-bit", size=(900, 390)),
        panel(baseline[ys, xs], "Previous blind-denoised model", size=(900, 390)),
        panel(original_enhancement[ys, xs], "Original enhancement + post-processing", size=(900, 390)),
        panel(final[ys, xs], "Geometry + structure-aware fog cleanup", size=(900, 390)),
    ]
    canvas = Image.new("L", (1800, 836), 0)
    for i, item in enumerate(items):
        canvas.paste(item, ((i % 2) * 900, (i // 2) * 418))
    canvas.save(path)


def save_cleanup_closeup(
    path: Path,
    before: np.ndarray,
    after: np.ndarray,
    roi: base.Roi,
) -> None:
    """Save a fixed-window before/after crop plus an amplified change audit."""
    a = before[roi.slices()]
    b = after[roi.slices()]
    lo, hi = (float(v) for v in np.percentile(a, (0.5, 99.7)))

    def make_panel(values: np.ndarray, title: str, absolute_delta: bool = False) -> Image.Image:
        if absolute_delta:
            scale = max(float(np.percentile(values, 99.7)), 1.0 / 65535.0)
            u8 = np.rint(np.clip(values / scale, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            u8 = np.rint(np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
        im = Image.fromarray(u8, mode="L")
        im = im.resize((560, 350), Image.Resampling.LANCZOS)
        panel_image = Image.new("L", (560, 378), 0)
        panel_image.paste(im, (0, 28))
        ImageDraw.Draw(panel_image).text((8, 8), title, fill=255, font=ImageFont.load_default())
        return panel_image

    items = [
        make_panel(a, "Geometry stage (before cleanup)"),
        make_panel(b, "Selected weak fog cleanup"),
        make_panel(np.abs(b - a), "Absolute cleanup delta (99.7% auto-scale)", absolute_delta=True),
    ]
    canvas = Image.new("L", (1680, 378), 0)
    for i, item in enumerate(items):
        canvas.paste(item, (i * 560, 0))
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True, help="previous blind-denoised uint16 TIFF")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--target-roi", type=base.parse_roi, default=base.Roi(620, 1210, 320, 1880))
    ap.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(700, 1110, 370, 970))
    ap.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(700, 1110, 1220, 1830))
    ap.add_argument("--left-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    ap.add_argument("--right-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    ap.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    ap.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    ap.add_argument("--geometry-iterations", type=int, default=1)
    ap.add_argument("--geometry-radius-y", type=int, default=1)
    ap.add_argument("--geometry-radius-x", type=int, default=1)
    ap.add_argument("--geometry-max-shift", type=float, default=0.35)
    ap.add_argument("--geometry-large-shift-trigger", type=float, default=3.0)
    ap.add_argument("--geometry-large-shift-cap", type=float, default=1.0)
    ap.add_argument("--fog-gradient-percentile", type=float, default=45.0)
    ap.add_argument("--fog-edge-retention-min", type=float, default=0.99)
    ap.add_argument("--fog-contrast-retention-min", type=float, default=0.995)
    ap.add_argument("--fog-ssim-min", type=float, default=0.9999)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    observed, source_info = base.load_gray(args.source)
    baseline, baseline_info = base.load_gray(args.baseline)
    if observed.shape != baseline.shape:
        raise ValueError(f"shape mismatch: source={observed.shape}, baseline={baseline.shape}")
    target = args.target_roi.clamp(observed.shape)
    left, right = args.left_roi.clamp(observed.shape), args.right_roi.clamp(observed.shape)
    left_body = args.left_body_roi.clamp(observed.shape)
    right_body = args.right_body_roi.clamp(observed.shape)
    observed_metrics = {"left": analyze_region(observed, left), "right": analyze_region(observed, right)}
    baseline_metrics = {"left": analyze_region(baseline, left), "right": analyze_region(baseline, right)}
    boundary_refs = build_boundary_references(
        observed, baseline, left_body, right_body, args.top_range, args.bottom_range
    )

    grid = []
    best_base = None
    original_enhancement = None
    # The generative blind baseline is smoother than the original AprN2S
    # baseline.  Keep the same guarded enhancement operator, but search a
    # wider strength range; quality_refine_crop still caps every change from
    # the denoised baseline by the same MAD-derived limit.
    for edge_amount in (0.08, 0.14, 0.20, 0.26, 0.32, 0.50, 0.80, 1.20, 1.50, 2.00):
        for structure_gain in (0.03, 0.06, 0.09, 0.12):
            crop = quality_refine_crop(baseline[target.slices()], edge_amount, structure_gain)
            candidate = base.feather_insert(baseline, crop, target, ramp=18)
            metrics = evaluate(candidate, baseline, left, right, baseline_metrics, observed_metrics)
            item = {"edge_amount": edge_amount, "structure_gain": structure_gain, **metrics}
            grid.append(item)
            if metrics["guardrail_pass"] and (best_base is None or metrics["score"] > best_base["score"]):
                best_base, original_enhancement = item, candidate
    if best_base is None or original_enhancement is None:
        raise RuntimeError("No original quality candidate passed the configured appearance guardrails")

    base_boundary_summary, _ = boundary_geometry(
        original_enhancement, boundary_refs, args.top_range, args.bottom_range
    )
    base_width_guardrail = bool(best_base["max_fwhm_relative_change_vs_source"] <= 0.04)
    if base_width_guardrail and boundary_guardrail(base_boundary_summary):
        geometry_image = original_enhancement
        warp_info = {
            "method": "endpoint warp skipped because original enhancement already passed geometry guardrails",
            "applied": False,
            "raw_pixel_writeback": False,
            "final_boundary_geometry": base_boundary_summary,
        }
    else:
        geometry_image, warp_info = endpoint_geometry_warp(
            original_enhancement,
            boundary_refs,
            args.top_range,
            args.bottom_range,
            iterations=args.geometry_iterations,
            radius_y=args.geometry_radius_y,
            radius_x=args.geometry_radius_x,
            max_shift=args.geometry_max_shift,
            large_shift_trigger=args.geometry_large_shift_trigger,
            large_shift_cap=args.geometry_large_shift_cap,
        )
    geometry_metrics = evaluate(
        geometry_image, baseline, left, right, baseline_metrics, observed_metrics
    )
    geometry_boundary_summary, _ = boundary_geometry(
        geometry_image, boundary_refs, args.top_range, args.bottom_range
    )
    geometry_width_guardrail = bool(geometry_metrics["max_fwhm_relative_change_vs_source"] <= 0.04)
    geometry_boundary_guardrail = boundary_guardrail(geometry_boundary_summary)
    if not (
        geometry_metrics["guardrail_pass"]
        and geometry_width_guardrail
        and geometry_boundary_guardrail
    ):
        raise RuntimeError("Geometry-corrected result failed final appearance, width, or endpoint guardrails")

    # The original enhancement and the geometry correction above remain
    # untouched. Search only a very weak post-geometry cleanup range, and
    # reject any candidate that measurably erodes edge/contrast/geometry.
    fog_grid = []
    selected_cleanup = None
    best_image = None
    final_metrics = None
    final_boundary_summary = None
    best_boundary_rows = None
    cleanup_pairs = ((0.0, 0.0), (0.08, 0.010), (0.10, 0.012), (0.12, 0.015), (0.14, 0.018))
    for fine_strength, haze_strength in cleanup_pairs:
        candidate, cleanup_info, reference_gradient = structure_aware_fog_cleanup(
            geometry_image,
            boundary_refs,
            target,
            fine_strength=fine_strength,
            haze_strength=haze_strength,
            gradient_percentile=args.fog_gradient_percentile,
        )
        candidate_metrics = evaluate(
            candidate, baseline, left, right, baseline_metrics, observed_metrics
        )
        candidate_boundary, candidate_rows = boundary_geometry(
            candidate, boundary_refs, args.top_range, args.bottom_range
        )
        roughness_before = low_structure_roughness(geometry_image, target, reference_gradient)
        roughness_after = low_structure_roughness(candidate, target, reference_gradient)
        fog_reduction = 1.0 - roughness_after / max(roughness_before, 1e-8)
        edge_retention = candidate_metrics["edge_gain"] / max(geometry_metrics["edge_gain"], 1e-8)
        contrast_retention = candidate_metrics["contrast_gain"] / max(geometry_metrics["contrast_gain"], 1e-8)
        cleanup_ssim = float(structural_similarity(
            geometry_image[target.slices()], candidate[target.slices()], data_range=1.0
        ))
        width_pass = bool(candidate_metrics["max_fwhm_relative_change_vs_source"] <= 0.04)
        endpoint_pass = boundary_guardrail(candidate_boundary)
        cleanup_pass = bool(
            candidate_metrics["guardrail_pass"]
            and width_pass
            and endpoint_pass
            and edge_retention >= args.fog_edge_retention_min
            and contrast_retention >= args.fog_contrast_retention_min
            and cleanup_ssim >= args.fog_ssim_min
        )
        item = {
            **cleanup_info,
            "low_structure_roughness_before": roughness_before,
            "low_structure_roughness_after": roughness_after,
            "low_structure_roughness_reduction": fog_reduction,
            "edge_acutance_retention_vs_geometry_stage": edge_retention,
            "local_contrast_retention_vs_geometry_stage": contrast_retention,
            "ssim_vs_geometry_stage": cleanup_ssim,
            "max_fwhm_relative_change_vs_source": candidate_metrics["max_fwhm_relative_change_vs_source"],
            "boundary_geometry": candidate_boundary,
            "appearance_guardrail_pass": candidate_metrics["guardrail_pass"],
            "width_guardrail_pass": width_pass,
            "boundary_guardrail_pass": endpoint_pass,
            "cleanup_guardrail_pass": cleanup_pass,
        }
        fog_grid.append(item)
        if cleanup_pass and (
            selected_cleanup is None
            or fog_reduction > selected_cleanup["low_structure_roughness_reduction"]
        ):
            selected_cleanup = item
            best_image = candidate
            final_metrics = candidate_metrics
            final_boundary_summary = candidate_boundary
            best_boundary_rows = candidate_rows
    if (
        selected_cleanup is None
        or best_image is None
        or final_metrics is None
        or final_boundary_summary is None
        or best_boundary_rows is None
    ):
        raise RuntimeError("No fog-cleanup candidate passed edge, contrast, SSIM, width, and endpoint guardrails")

    final_width_guardrail = bool(final_metrics["max_fwhm_relative_change_vs_source"] <= 0.04)
    final_boundary_guardrail = boundary_guardrail(final_boundary_summary)

    best = {
        "edge_amount": best_base["edge_amount"],
        "structure_gain": best_base["structure_gain"],
        **final_metrics,
        "boundary_geometry": final_boundary_summary,
        "boundary_guardrail_pass": final_boundary_guardrail,
        "width_guardrail_pass": final_width_guardrail,
    }

    display = display_version(best_image, target)
    imwrite(
        args.outdir / "QUALITY_original_enhancement_16bit.tif",
        base.to_uint16(original_enhancement),
        photometric="minisblack",
        description="Unmodified original quality enhancement and post-processing output; no geometry correction.",
    )
    imwrite(
        args.outdir / "QUALITY_geometry_only_16bit.tif",
        base.to_uint16(geometry_image),
        photometric="minisblack",
        description="Original enhanced pixels after local sub-pixel geometry correction; before residual fog cleanup; no raw-pixel writeback.",
    )
    imwrite(
        args.outdir / "QUALITY_boundary_preserved_16bit.tif",
        base.to_uint16(best_image),
        photometric="minisblack",
        description="Original enhanced pixels after local sub-pixel geometry correction and weak edge-gated fog cleanup; no raw-pixel writeback; no generative pixels.",
    )
    imwrite(
        args.outdir / "QUALITY_balanced_16bit.tif",
        base.to_uint16(best_image),
        photometric="minisblack",
        description="Blind-denoised quality refinement; no generative pixels; not a physical-resolution claim.",
    )
    base.save_preview(args.outdir / "QUALITY_balanced_preview.png", best_image)
    Image.fromarray(np.rint(display * 255).astype(np.uint8), mode="L").save(args.outdir / "QUALITY_display_only.png")
    save_comparison(
        args.outdir / "QUALITY_comparison.png",
        observed,
        baseline,
        original_enhancement,
        best_image,
        target,
    )
    cleanup_detail = base.Roi(990, 1240, 650, 1050).clamp(best_image.shape)
    save_cleanup_closeup(
        args.outdir / "QUALITY_fog_cleanup_closeup.png",
        geometry_image,
        best_image,
        cleanup_detail,
    )
    save_boundary_overlay(args.outdir / "QUALITY_boundary_overlay.png", best_image, best_boundary_rows, target)
    with (args.outdir / "boundary_geometry.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_boundary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_boundary_rows)

    roughness_ratio_vs_source = float(np.median([
        best[s]["flat_region_roughness"] / observed_metrics[s]["flat_region_roughness"]
        for s in ("left", "right")
    ]))
    edge_gain_vs_source = float(np.median([
        best[s]["median_edge_acutance"] / observed_metrics[s]["median_edge_acutance"]
        for s in ("left", "right")
    ]))
    contrast_gain_vs_source = float(np.median([
        best[s]["profile_local_contrast_rms"] / observed_metrics[s]["profile_local_contrast_rms"]
        for s in ("left", "right")
    ]))
    correction_delta = geometry_image[target.slices()] - original_enhancement[target.slices()]
    correction_abs = np.abs(correction_delta)
    changed = correction_abs > (0.5 / 65535.0)
    correction_ssim = float(structural_similarity(
        original_enhancement[target.slices()], geometry_image[target.slices()], data_range=1.0
    ))
    cleanup_delta = best_image - geometry_image
    cleanup_abs = np.abs(cleanup_delta)
    cleanup_changed = cleanup_abs > (0.5 / 65535.0)

    payload = {
        "source": source_info,
        "baseline": baseline_info,
        "objective": "original denoising/enhancement + raw-referenced boundary preservation + weak structure-aware residual fog cleanup",
        "selected": best,
        "original_enhancement_selected_before_geometry": {
            **best_base,
            "boundary_geometry": base_boundary_summary,
        },
        "quality_vs_original": {
            "flat_high_frequency_ratio": roughness_ratio_vs_source,
            "flat_high_frequency_reduction": 1.0 - roughness_ratio_vs_source,
            "edge_acutance_gain": edge_gain_vs_source,
            "local_contrast_gain": contrast_gain_vs_source,
        },
        "original_metrics": observed_metrics,
        "baseline_metrics": baseline_metrics,
        "guardrails": {
            "layer_count_delta_max": 1,
            "center_shift_max_px": 1.0,
            "fwhm_relative_change_max": 0.05,
            "fwhm_relative_change_vs_source_max": 0.04,
            "flat_roughness_ratio_max": 1.03,
            "ssim_vs_blind_baseline_min": 0.985,
            "endpoint_shift_abs_p95_max_px": 0.35,
            "endpoint_shift_abs_max_px": 0.75,
            "length_delta_abs_p95_max_px": 0.50,
            "length_delta_abs_max_px": 1.00,
            "fog_edge_acutance_retention_min": args.fog_edge_retention_min,
            "fog_local_contrast_retention_min": args.fog_contrast_retention_min,
            "fog_ssim_vs_geometry_stage_min": args.fog_ssim_min,
        },
        "geometry_correction": {
            **warp_info,
            "top_range": list(args.top_range),
            "bottom_range": list(args.bottom_range),
            "ssim_vs_original_enhancement": correction_ssim,
            "absolute_change_p99_normalized": float(np.percentile(np.abs(correction_delta), 99.0)),
            "changed_pixel_fraction_in_target_roi": float(np.mean(changed)),
            "absolute_change_median_of_changed_normalized": float(np.median(correction_abs[changed])) if np.any(changed) else 0.0,
            "absolute_change_max_normalized": float(np.max(correction_abs)),
            "flat_roughness_ratio_vs_original_enhancement": float(
                geometry_metrics["roughness_ratio"] / max(best_base["roughness_ratio"], 1e-8)
            ),
        },
        "fog_cleanup": {
            "selected": selected_cleanup,
            "candidate_grid": fog_grid,
            "selection_rule": "maximum low-structure roughness reduction among candidates passing all appearance, width, endpoint, edge-retention, contrast-retention, and SSIM guardrails",
            "change_statistics": {
                "changed_pixel_fraction_full_image": float(np.mean(cleanup_changed)),
                "absolute_change_p95_of_changed_normalized": float(np.percentile(cleanup_abs[cleanup_changed], 95.0)) if np.any(cleanup_changed) else 0.0,
                "absolute_change_p99_of_changed_normalized": float(np.percentile(cleanup_abs[cleanup_changed], 99.0)) if np.any(cleanup_changed) else 0.0,
                "absolute_change_max_normalized": float(np.max(cleanup_abs)),
            },
        },
        "grid": grid,
        "notes": [
            "QUALITY_balanced_16bit.tif contains no generative pixels.",
            "QUALITY_original_enhancement_16bit.tif is the unchanged original enhancement/post-processing result.",
            "QUALITY_geometry_only_16bit.tif is the enhanced result after geometry correction and before residual fog cleanup.",
            "QUALITY_boundary_preserved_16bit.tif adds weak structure-aware fog cleanup after geometry correction and writes back no raw pixels.",
            "Raw data supplies endpoint coordinates only; low-confidence raw endpoints are marked for review.",
            "QUALITY_display_only.png uses a display mapping and must not be treated as preserved intensity data.",
            "Improved digital acutance does not increase scanner physical resolution.",
        ],
    }
    (args.outdir / "quality_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "completed": True,
        "edge_amount": best["edge_amount"],
        "structure_gain": best["structure_gain"],
        "edge_gain": best["edge_gain"],
        "contrast_gain": best["contrast_gain"],
        "roughness_ratio": best["roughness_ratio"],
        "flat_high_frequency_reduction_vs_original": 1.0 - roughness_ratio_vs_source,
        "guardrail_pass": best["guardrail_pass"],
        "boundary_guardrail_pass": best["boundary_guardrail_pass"],
        "width_guardrail_pass": best["width_guardrail_pass"],
        "geometry_ssim_vs_original_enhancement": correction_ssim,
        "fog_fine_strength": selected_cleanup["fine_strength"],
        "fog_haze_strength": selected_cleanup["haze_strength"],
        "fog_low_structure_roughness_reduction": selected_cleanup["low_structure_roughness_reduction"],
        "fog_edge_acutance_retention": selected_cleanup["edge_acutance_retention_vs_geometry_stage"],
        "fog_local_contrast_retention": selected_cleanup["local_contrast_retention_vs_geometry_stage"],
        "fog_ssim_vs_geometry_stage": selected_cleanup["ssim_vs_geometry_stage"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
