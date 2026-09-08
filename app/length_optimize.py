#!/usr/bin/env python3
"""Independent sub-pixel lamella-length audit and optional refinement.

The default geometry-only mode does not change input pixels. If optional
zero-phase sharpening is requested, it operates on enhanced pixels only; raw
pixels are never written back. Results are reported in pixels unless a
calibrated pixel size is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, sobel
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity
from tifffile import imwrite

import pipeline as base


@dataclass
class EndpointMeasurement:
    top: float
    bottom: float
    length: float
    top_spread: float
    bottom_spread: float
    uncertainty: float
    top_snr: float
    bottom_snr: float


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - med)))


def estimate_pitch(profile: np.ndarray, minimum: int = 14, maximum: int = 34) -> float:
    signal = profile - gaussian_filter1d(profile, sigma=10.0)
    signal = signal - signal.mean()
    corr = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
    maximum = min(maximum, len(corr) - 1)
    if maximum <= minimum:
        return 20.0
    lag = int(np.argmax(corr[minimum : maximum + 1]) + minimum)
    return float(lag)


def detect_centers(img: np.ndarray, roi: base.Roi) -> tuple[np.ndarray, float, np.ndarray]:
    crop = img[roi.slices()]
    profile = gaussian_filter1d(crop.mean(axis=0), sigma=0.9)
    highpass = profile - gaussian_filter1d(profile, sigma=9.0)
    prominence = max(float(np.std(highpass)) * 0.10, 2.0 / 65535.0)
    peaks, _ = find_peaks(highpass, distance=7, prominence=prominence)
    peaks = peaks[(peaks > 7) & (peaks < profile.size - 7)]
    refined = np.asarray([quadratic_peak_position(highpass, int(p)) for p in peaks], dtype=np.float64)
    diffs = np.diff(refined)
    plausible = diffs[(diffs >= 6.0) & (diffs <= 18.0)]
    pitch = float(np.median(plausible)) if len(plausible) else 12.0
    return refined + roi.x0, pitch, profile


def quadratic_peak_position(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, center, right = (float(v) for v in values[index - 1 : index + 2])
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return float(index)
    offset = 0.5 * (left - right) / denom
    return float(index + np.clip(offset, -0.75, 0.75))


def one_profile_endpoint(
    profile: np.ndarray,
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
    noise_range: tuple[int, int],
) -> tuple[float, float, float, float]:
    smooth = gaussian_filter1d(profile.astype(np.float64), sigma=1.05)
    gradient = np.gradient(smooth)
    ta, tb = top_range
    ba, bb = bottom_range
    top_local = int(np.argmax(gradient[ta:tb])) + ta
    bottom_local = int(np.argmin(gradient[ba:bb])) + ba
    top = quadratic_peak_position(gradient, top_local)
    bottom = quadratic_peak_position(-gradient, bottom_local)
    na, nb = noise_range
    noise = robust_sigma(gradient[na:nb])
    if noise < 1e-8:
        noise = robust_sigma(gradient)
    top_snr = float(abs(gradient[top_local]) / max(noise, 1e-8))
    bottom_snr = float(abs(gradient[bottom_local]) / max(noise, 1e-8))
    return top, bottom, top_snr, bottom_snr


def track_ridge(
    guide: np.ndarray,
    center_x: float,
    pitch: float,
    y0: int,
    y1: int,
) -> np.ndarray:
    """Track one bright ridge while preventing jumps to an adjacent ridge."""
    smooth = gaussian_filter(guide, sigma=(0.9, 0.8))
    h, w = guide.shape
    y0, y1 = max(0, y0), min(h, y1)
    path = np.full(h, float(center_x), dtype=np.float32)
    radius = max(3, int(round(0.38 * pitch)))
    global_lo = max(1, int(math.floor(center_x)) - radius)
    global_hi = min(w - 2, int(math.ceil(center_x)) + radius)
    mid = min(y1 - 1, max(y0, (y0 + y1) // 2))
    seed_lo, seed_hi = max(global_lo, int(round(center_x)) - 2), min(global_hi, int(round(center_x)) + 2)
    seed_candidates = np.arange(seed_lo, seed_hi + 1)
    seed = int(seed_candidates[np.argmax(smooth[mid, seed_candidates])])
    path[mid] = seed
    for direction in (-1, 1):
        previous = seed
        rows = range(mid - 1, y0 - 1, -1) if direction < 0 else range(mid + 1, y1)
        for y in rows:
            lo, hi = max(global_lo, previous - 2), min(global_hi, previous + 2)
            candidates = np.arange(lo, hi + 1)
            scores = smooth[y, candidates] - 0.004 * np.square(candidates - previous)
            previous = int(candidates[np.argmax(scores)])
            path[y] = previous
    return path


def sample_tracked_profile(img: np.ndarray, path_x: np.ndarray, offset: int = 0) -> np.ndarray:
    h, w = img.shape
    profile = np.empty(h, dtype=np.float64)
    for y in range(h):
        center = int(round(float(path_x[y]))) + offset
        lo, hi = max(0, center - 1), min(w, center + 2)
        profile[y] = float(np.mean(img[y, lo:hi]))
    return profile


def measure_endpoints(
    img: np.ndarray,
    center_x: float,
    pitch: float,
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
    hint: EndpointMeasurement | None = None,
    association_radius: int = 3,
    path_x: np.ndarray | None = None,
) -> EndpointMeasurement:
    half = max(3, min(7, int(round(pitch * 0.26))))
    center = int(round(center_x))
    tops: list[float] = []
    bottoms: list[float] = []
    top_snrs: list[float] = []
    bottom_snrs: list[float] = []
    noise_range = (max(0, top_range[0] - 80), max(1, top_range[0] - 10))
    if hint is not None:
        top_range = (
            max(top_range[0], int(math.floor(hint.top)) - association_radius),
            min(top_range[1], int(math.ceil(hint.top)) + association_radius + 1),
        )
        bottom_range = (
            max(bottom_range[0], int(math.floor(hint.bottom)) - association_radius),
            min(bottom_range[1], int(math.ceil(hint.bottom)) + association_radius + 1),
        )
    offsets = range(-half, half + 1, 2) if path_x is None else (-1, 0, 1)
    # Parallel tracked profiles provide an internal repeatability estimate.
    for dx in offsets:
        if path_x is None:
            x0 = max(0, center + dx - 1)
            x1 = min(img.shape[1], center + dx + 2)
            profile = np.mean(img[:, x0:x1], axis=1)
        else:
            profile = sample_tracked_profile(img, path_x, offset=dx)
        top, bottom, ts, bs = one_profile_endpoint(profile, top_range, bottom_range, noise_range)
        tops.append(top)
        bottoms.append(bottom)
        top_snrs.append(ts)
        bottom_snrs.append(bs)
    top = float(np.median(tops))
    bottom = float(np.median(bottoms))
    top_spread = robust_sigma(np.asarray(tops))
    bottom_spread = robust_sigma(np.asarray(bottoms))
    interpolation_floor = 0.20
    uncertainty = float(math.sqrt(top_spread**2 + bottom_spread**2 + interpolation_floor**2))
    return EndpointMeasurement(
        top=top,
        bottom=bottom,
        length=bottom - top,
        top_spread=top_spread,
        bottom_spread=bottom_spread,
        uncertainty=uncertainty,
        top_snr=float(np.median(top_snrs)),
        bottom_snr=float(np.median(bottom_snrs)),
    )


def length_aware_refine(
    source: np.ndarray,
    denoised: np.ndarray,
    roi: base.Roi,
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
    sharpen_amount: float,
    sharpen_sigma: float,
) -> tuple[np.ndarray, dict]:
    ys, xs = roi.slices()
    raw = source[ys, xs]
    current = denoised[ys, xs]

    # The raw image supplies geometry gates only.  Its pixel values are never
    # copied into the already denoised/enhanced candidate.
    raw_smooth = gaussian_filter(raw, sigma=0.70)
    gx = np.abs(sobel(raw_smooth, axis=1))
    gy = np.abs(sobel(raw_smooth, axis=0))
    tx = float(np.percentile(gx, 86.0)) + 1e-8
    ty = float(np.percentile(gy, 70.0)) + 1e-8
    smooth = gaussian_filter(current, sigma=sharpen_sigma)
    detail = current - smooth
    edge_strength = np.hypot(gx / tx, gy / ty)
    refine_gate = 1.0 - np.exp(-(edge_strength**2))
    candidate = current + float(sharpen_amount) * refine_gate * detail

    sigma_n = base.noise_sigma_mad(raw)
    cap = max(2.75 * sigma_n, 2.0 / 65535.0)
    candidate = current + np.clip(candidate - current, -cap, cap)
    out = base.feather_insert(denoised, np.clip(candidate, 0.0, 1.0).astype(np.float32), roi, ramp=18)
    return out, {
        "sharpen_amount": sharpen_amount,
        "sharpen_sigma": sharpen_sigma,
        "raw_pixel_writeback": False,
        "raw_geometry_guide": "Sobel gates only",
        "endpoint_ranges": [list(top_range), list(bottom_range)],
        "estimated_noise_sigma_normalized": sigma_n,
        "change_cap_normalized": cap,
        "operation": "symmetric Gaussian unsharp of enhanced pixels only; zero phase",
    }


def same_coordinate_gradient_gain(
    source: np.ndarray,
    refined: np.ndarray,
    center_x: float,
    pitch: float,
    top_y: float,
    bottom_y: float,
    path_x: np.ndarray | None = None,
) -> tuple[float, float]:
    if path_x is None:
        half = max(2, min(5, int(round(pitch * 0.22))))
        center = int(round(center_x))
        x0, x1 = max(0, center - half), min(source.shape[1], center + half + 1)
        raw_profile = source[:, x0:x1].mean(axis=1)
        out_profile = refined[:, x0:x1].mean(axis=1)
    else:
        raw_profile = sample_tracked_profile(source, path_x)
        out_profile = sample_tracked_profile(refined, path_x)
    raw_grad = np.abs(np.gradient(gaussian_filter1d(raw_profile.astype(np.float64), sigma=1.05)))
    out_grad = np.abs(np.gradient(gaussian_filter1d(out_profile.astype(np.float64), sigma=1.05)))
    axis = np.arange(len(raw_grad), dtype=np.float64)
    gains = []
    for position in (top_y, bottom_y):
        before = float(np.interp(position, axis, raw_grad))
        after = float(np.interp(position, axis, out_grad))
        gains.append(after / max(before, 1e-8))
    return float(gains[0]), float(gains[1])


def make_overlay(
    path: Path,
    image: np.ndarray,
    rows: list[dict],
    crop_roi: base.Roi | None = None,
) -> None:
    if crop_roi is None:
        view = image
        offset_x = offset_y = 0
    else:
        view = image[crop_roi.slices()]
        offset_x, offset_y = crop_roi.x0, crop_roi.y0
    nz = view[view > 0]
    sample = nz if nz.size else view.reshape(-1)
    lo, hi = (float(v) for v in np.percentile(sample, (0.5, 99.6)))
    u8 = np.rint(np.clip((view - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
    canvas = Image.fromarray(u8, mode="L").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, row in enumerate(rows):
        x_top = float(row["recommended_top_x_px"]) - offset_x
        x_bottom = float(row["recommended_bottom_x_px"]) - offset_x
        top = float(row["recommended_top_y_px"]) - offset_y
        bottom = float(row["recommended_bottom_y_px"]) - offset_y
        if max(x_top, x_bottom) < 0 or min(x_top, x_bottom) >= canvas.width or bottom < 0 or top >= canvas.height:
            continue
        color = (70, 255, 110) if row["quality"] == "pass" else (255, 80, 70)
        draw.line((x_top, top, x_bottom, bottom), fill=color, width=1)
        draw.line((x_top - 4, top, x_top + 4, top), fill=color, width=2)
        draw.line((x_bottom - 4, bottom, x_bottom + 4, bottom), fill=color, width=2)
        if idx % 5 == 0:
            draw.text((x_top + 3, max(0, top - 11)), str(row["layer_id"]), fill=color, font=font)
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="original uint16 TIFF")
    ap.add_argument("--denoised", type=Path, required=True, help="measurement-chain uint16 TIFF")
    ap.add_argument(
        "--geometry-guide",
        type=Path,
        help="fixed pre-stage image used only to detect centers and track paths",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--refine-roi", type=base.parse_roi, default=base.Roi(600, 1230, 320, 1880))
    ap.add_argument("--left-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    ap.add_argument("--right-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    ap.add_argument("--top-range", type=str, default="600,790")
    ap.add_argument("--bottom-range", type=str, default="1010,1240")
    ap.add_argument("--sharpen-amount", type=float, default=0.0, help="0 performs geometry-only audit without changing pixels")
    ap.add_argument("--sharpen-sigma", type=float, default=0.90)
    ap.add_argument("--pixel-size", type=float, help="calibrated physical units per pixel")
    ap.add_argument("--unit", default="mm", help="unit used with --pixel-size")
    ap.add_argument("--uncertainty-max", type=float, default=3.0, help="per-layer review threshold in pixels")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    denoised, denoised_info = base.load_gray(args.denoised)
    if source.shape != denoised.shape:
        raise ValueError(f"shape mismatch: source={source.shape}, denoised={denoised.shape}")
    if args.geometry_guide:
        geometry_guide, geometry_guide_info = base.load_gray(args.geometry_guide)
        if geometry_guide.shape != source.shape:
            raise ValueError(
                f"shape mismatch: source={source.shape}, geometry_guide={geometry_guide.shape}"
            )
    else:
        geometry_guide, geometry_guide_info = denoised, denoised_info
    refine_roi = args.refine_roi.clamp(source.shape)
    left_roi = args.left_body_roi.clamp(source.shape)
    right_roi = args.right_body_roi.clamp(source.shape)
    top_range = tuple(int(v) for v in args.top_range.split(","))
    bottom_range = tuple(int(v) for v in args.bottom_range.split(","))
    if len(top_range) != 2 or len(bottom_range) != 2:
        raise ValueError("ranges must be start,stop")

    refined, refine_info = length_aware_refine(
        source, denoised, refine_roi, top_range, bottom_range, args.sharpen_amount, args.sharpen_sigma
    )
    rows: list[dict] = []
    for side, roi in (("left", left_roi), ("right", right_roi)):
        centers, pitch, _ = detect_centers(geometry_guide, roi)
        for index, center in enumerate(centers, start=1):
            path_x = track_ridge(geometry_guide, center, pitch, top_range[0], bottom_range[1])
            raw_m = measure_endpoints(source, center, pitch, top_range, bottom_range, path_x=path_x)
            enhanced_m = measure_endpoints(
                refined, center, pitch, top_range, bottom_range, hint=raw_m, path_x=path_x
            )
            top_shift = enhanced_m.top - raw_m.top
            bottom_shift = enhanced_m.bottom - raw_m.bottom
            length_delta = enhanced_m.length - raw_m.length
            top_gain, bottom_gain = same_coordinate_gradient_gain(
                source, refined, center, pitch, raw_m.top, raw_m.bottom, path_x=path_x
            )
            top_x = float(np.interp(raw_m.top, np.arange(len(path_x)), path_x))
            bottom_x = float(np.interp(raw_m.bottom, np.arange(len(path_x)), path_x))
            quality = "pass"
            reasons = []
            if min(raw_m.top_snr, raw_m.bottom_snr) < 4.0:
                quality, reasons = "review", reasons + ["low_edge_snr"]
            if raw_m.uncertainty > args.uncertainty_max:
                quality, reasons = "review", reasons + ["high_length_uncertainty"]
            enhancement_reasons = []
            if abs(top_shift) > 0.75:
                enhancement_reasons.append("top_redetection_shift")
            if abs(bottom_shift) > 0.75:
                enhancement_reasons.append("bottom_redetection_shift")
            if abs(length_delta) > 1.00:
                enhancement_reasons.append("redetected_length_delta")
            if min(top_gain, bottom_gain) < 0.98:
                enhancement_reasons.append("local_gradient_not_improved")
            row = {
                "side": side,
                "layer_id": f"{side[0].upper()}{index:02d}",
                "center_x_px": round(float(center), 4),
                "recommended_top_x_px": round(top_x, 4),
                "recommended_bottom_x_px": round(bottom_x, 4),
                "estimated_pitch_px": round(float(pitch), 4),
                "source_top_y_px": round(raw_m.top, 4),
                "source_bottom_y_px": round(raw_m.bottom, 4),
                "source_length_px": round(raw_m.length, 4),
                "recommended_top_y_px": round(raw_m.top, 4),
                "recommended_bottom_y_px": round(raw_m.bottom, 4),
                "recommended_length_px": round(raw_m.length, 4),
                "recommended_length_uncertainty_px": round(raw_m.uncertainty, 4),
                "enhanced_top_y_px": round(enhanced_m.top, 4),
                "enhanced_bottom_y_px": round(enhanced_m.bottom, 4),
                "enhanced_length_px": round(enhanced_m.length, 4),
                "top_shift_px": round(top_shift, 4),
                "bottom_shift_px": round(bottom_shift, 4),
                "length_delta_px": round(length_delta, 4),
                "length_uncertainty_px": round(enhanced_m.uncertainty, 4),
                "top_gradient_gain_same_coordinate": round(top_gain, 4),
                "bottom_gradient_gain_same_coordinate": round(bottom_gain, 4),
                "source_top_edge_snr": round(raw_m.top_snr, 3),
                "source_bottom_edge_snr": round(raw_m.bottom_snr, 3),
                "top_edge_snr": round(enhanced_m.top_snr, 3),
                "bottom_edge_snr": round(enhanced_m.bottom_snr, 3),
                "quality": quality,
                "review_reason": ";".join(reasons),
                "enhancement_diagnostic": "pass" if not enhancement_reasons else "review",
                "enhancement_review_reason": ";".join(enhancement_reasons),
            }
            if args.pixel_size is not None:
                row[f"recommended_length_{args.unit}"] = round(raw_m.length * args.pixel_size, 6)
                row[f"length_uncertainty_{args.unit}"] = round(raw_m.uncertainty * args.pixel_size, 6)
            rows.append(row)

    csv_path = args.outdir / "layer_lengths.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    imwrite(
        args.outdir / "MEASUREMENT_length_optimized_16bit.tif",
        base.to_uint16(refined),
        photometric="minisblack",
        description="Length-aware measurement candidate. Requires pixel-size and PSF calibration for absolute metrology.",
    )
    base.save_preview(args.outdir / "MEASUREMENT_length_optimized_preview.png", refined)
    make_overlay(args.outdir / "layer_length_overlay_full.png", refined, rows)
    make_overlay(args.outdir / "layer_length_overlay_roi.png", refined, rows, refine_roi)

    y0, y1, x0, x1 = refine_roi.y0, refine_roi.y1, refine_roi.x0, refine_roi.x1
    before_crop = source[y0:y1, x0:x1]
    after_crop = refined[y0:y1, x0:x1]
    endpoint_shifts = np.asarray(
        [[float(r["top_shift_px"]), float(r["bottom_shift_px"])] for r in rows], dtype=np.float64
    )
    length_deltas = np.asarray([float(r["length_delta_px"]) for r in rows], dtype=np.float64)
    uncertainties = np.asarray([float(r["recommended_length_uncertainty_px"]) for r in rows], dtype=np.float64)
    gradient_gains = np.asarray(
        [[float(r["top_gradient_gain_same_coordinate"]), float(r["bottom_gradient_gain_same_coordinate"])] for r in rows],
        dtype=np.float64,
    )
    source_snr = np.asarray(
        [[float(r["source_top_edge_snr"]), float(r["source_bottom_edge_snr"])] for r in rows], dtype=np.float64
    )
    enhanced_snr = np.asarray(
        [[float(r["top_edge_snr"]), float(r["bottom_edge_snr"])] for r in rows], dtype=np.float64
    )
    passed = sum(r["quality"] == "pass" for r in rows)
    reliable_mask = np.asarray([r["quality"] == "pass" for r in rows], dtype=bool)
    reliable_endpoint_shifts = endpoint_shifts[reliable_mask]
    reliable_length_deltas = length_deltas[reliable_mask]
    if not len(reliable_length_deltas):
        raise RuntimeError("No reliable raw endpoint references were detected")
    geometry_only_audit = bool(abs(args.sharpen_amount) < 1e-12)
    appearance_guardrail = bool(
        geometry_only_audit
        or (
            structural_similarity(before_crop, after_crop, data_range=1.0) >= 0.990
            and np.median(gradient_gains) >= 1.0
        )
    )
    qa = {
        "source": source_info,
        "denoised": denoised_info,
        "geometry_guide": {
            **geometry_guide_info,
            "role": "fixed center/path coordinates only",
            "pixel_writeback": False,
        },
        "refinement": refine_info,
        "rois": {
            "refine": asdict(refine_roi),
            "left_body": asdict(left_roi),
            "right_body": asdict(right_roi),
            "top_range": list(top_range),
            "bottom_range": list(bottom_range),
        },
        "pixel_calibration": {
            "pixel_size": args.pixel_size,
            "unit": args.unit if args.pixel_size is not None else None,
            "absolute_length_available": args.pixel_size is not None,
        },
        "layer_count": len(rows),
        "pass_count": passed,
        "review_count": len(rows) - passed,
        "pass_fraction": passed / max(1, len(rows)),
        "audit_mode": "geometry_only_no_pixel_change" if geometry_only_audit else "optional_visual_refinement",
        "refine_roi_ssim_vs_source": float(structural_similarity(before_crop, after_crop, data_range=1.0)),
        "endpoint_shift_abs_median_px": float(np.median(np.abs(reliable_endpoint_shifts))),
        "endpoint_shift_abs_p95_px": float(np.percentile(np.abs(reliable_endpoint_shifts), 95.0)),
        "endpoint_shift_abs_max_px": float(np.max(np.abs(reliable_endpoint_shifts))),
        "length_delta_abs_median_px": float(np.median(np.abs(reliable_length_deltas))),
        "length_delta_abs_p95_px": float(np.percentile(np.abs(reliable_length_deltas), 95.0)),
        "length_delta_abs_max_px": float(np.max(np.abs(reliable_length_deltas))),
        "all_layers_endpoint_shift_abs_max_px_diagnostic": float(np.max(np.abs(endpoint_shifts))),
        "all_layers_length_delta_abs_max_px_diagnostic": float(np.max(np.abs(length_deltas))),
        "length_uncertainty_median_px": float(np.median(uncertainties)),
        "length_uncertainty_p95_px": float(np.percentile(uncertainties, 95.0)),
        "edge_snr_median_source": float(np.median(source_snr)),
        "edge_snr_median_enhanced": float(np.median(enhanced_snr)),
        "edge_snr_median_gain": float(np.median(enhanced_snr) / max(np.median(source_snr), 1e-8)),
        "edge_gradient_gain_same_coordinate_median": float(np.median(gradient_gains)),
        "edge_gradient_gain_same_coordinate_p05": float(np.percentile(gradient_gains, 5.0)),
        "guardrails": {
            "per_layer_top_shift_abs_max_px": 0.75,
            "per_layer_bottom_shift_abs_max_px": 0.75,
            "per_layer_length_delta_abs_max_px": 1.0,
            "endpoint_shift_abs_p95_max_px": 0.35,
            "length_delta_abs_p95_max_px": 0.50,
            "edge_snr_min": 4.0,
            "length_uncertainty_max_px": args.uncertainty_max,
            "required_pass_fraction": 0.90,
            "refine_roi_ssim_min": 0.990,
            "edge_gradient_gain_median_min": 1.0,
            "appearance_checks_required_only_when_sharpening": True,
        },
        "guardrail_pass": bool(
            passed / max(1, len(rows)) >= 0.90
            and appearance_guardrail
            and np.percentile(np.abs(reliable_endpoint_shifts), 95.0) <= 0.35
            and np.max(np.abs(reliable_endpoint_shifts)) <= 0.75
            and np.percentile(np.abs(reliable_length_deltas), 95.0) <= 0.50
            and np.max(np.abs(reliable_length_deltas)) <= 1.00
        ),
        "data_quality": {
            "native_bit_depth": 16,
            "x_resolution_tag": None,
            "y_resolution_tag": None,
            "resolution_unit_tag": None,
            "absolute_accuracy_status": "blocked_without_pixel_size_and_system_PSF_calibration",
        },
    }
    (args.outdir / "length_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "layers": len(rows), "pass": passed, "guardrail_pass": qa["guardrail_pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
