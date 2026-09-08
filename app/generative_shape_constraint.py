#!/usr/bin/env python3
"""Measure lamella/gap geometry and render a generation-conditioning blueprint.

Geometry is measured primarily on a blind-denoised guide. The original image
is sampled at the same coordinates as independent evidence and controls the
reported confidence, but its noisy pixels do not define the condition. The
CSV/JSON files remain the authoritative numeric record; an image generator is
not expected to reproduce subpixel dimensions exactly without a hard spatial
conditioning mechanism such as ControlNet or an explicit projection stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from tifffile import imwrite

import length_optimize as length
import pipeline as base


def crossing_position(values: np.ndarray, start: int, step: int, level: float) -> float | None:
    """Return a linearly interpolated half-height crossing."""
    index = int(start)
    while 0 <= index + step < len(values):
        nxt = index + step
        a, b = float(values[index]), float(values[nxt])
        if (a - level) * (b - level) <= 0.0 and a != b:
            fraction = (level - a) / (b - a)
            return float(index + fraction * step)
        index = nxt
    return None


def transverse_width_at_row(
    image: np.ndarray,
    path_x: np.ndarray,
    y: int,
    pitch: float,
) -> float | None:
    """Estimate local transverse FWHM without including neighbouring layers."""
    y = int(np.clip(y, 2, image.shape[0] - 3))
    center = float(path_x[y])
    radius = max(4, int(math.floor(0.45 * pitch)))
    x0 = max(0, int(math.floor(center)) - radius)
    x1 = min(image.shape[1], int(math.ceil(center)) + radius + 1)
    if x1 - x0 < 7:
        return None
    profile = gaussian_filter1d(image[y - 2 : y + 3, x0:x1].mean(axis=0), sigma=0.65)
    expected = int(np.clip(round(center) - x0, 1, len(profile) - 2))
    lo, hi = max(0, expected - 2), min(len(profile), expected + 3)
    peak = int(lo + np.argmax(profile[lo:hi]))
    edge_samples = np.concatenate((profile[:2], profile[-2:]))
    baseline = float(np.median(edge_samples))
    maximum = float(profile[peak])
    if maximum - baseline <= 1.0 / 65535.0:
        return None
    half = baseline + 0.5 * (maximum - baseline)
    left = crossing_position(profile, peak, -1, half)
    right = crossing_position(profile, peak, 1, half)
    if left is None or right is None or right <= left:
        return None
    width = float(right - left)
    if not (0.4 <= width <= 0.90 * pitch):
        return None
    return width


def measure_layer_width(
    image: np.ndarray,
    path_x: np.ndarray,
    top: float,
    bottom: float,
    pitch: float,
    samples: int = 25,
) -> dict:
    margin = max(5.0, 0.10 * (bottom - top))
    if bottom - top <= 2.0 * margin:
        return {"median": None, "p10": None, "p90": None, "samples": 0}
    rows = np.linspace(top + margin, bottom - margin, samples)
    widths = [transverse_width_at_row(image, path_x, int(round(y)), pitch) for y in rows]
    valid = np.asarray([value for value in widths if value is not None], dtype=np.float64)
    if not len(valid):
        return {"median": None, "p10": None, "p90": None, "samples": 0}
    return {
        "median": float(np.median(valid)),
        "p10": float(np.percentile(valid, 10.0)),
        "p90": float(np.percentile(valid, 90.0)),
        "samples": int(len(valid)),
    }


def smooth_centerline(path_x: np.ndarray, pitch: float) -> np.ndarray:
    """Return a low-noise, physically plausible centerline from a clean guide.

    The smoothing scale follows the measured pitch instead of using the old
    fixed 45-pixel value. This removes row noise while retaining slow bowing of
    an individual lamella.
    """
    sigma_y = float(np.clip(1.35 * pitch, 10.0, 22.0))
    return gaussian_filter1d(path_x.astype(np.float64), sigma=sigma_y).astype(np.float32)


def regularize_centerline_bundles(
    rows: list[dict],
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
) -> None:
    """Jointly regularize same-side lamellae while retaining slow real bowing.

    Scanner noise makes independent ridge tracks wiggle in unrelated ways.
    Physical lamellae in one stack share a common low-frequency deformation,
    so we estimate that common mode robustly and retain only a strongly
    smoothed fraction of each lamella's individual deviation.
    """
    y0 = max(0, int(top_range[0]))
    y1 = min(len(rows[0]["measurement_path_x"]), int(bottom_range[1])) if rows else y0
    for side in ("left", "right"):
        current = sorted(
            (row for row in rows if row["side"] == side),
            key=lambda row: row["center_x_px"],
        )
        if not current or y1 <= y0:
            continue
        measured = np.stack([row["measurement_path_x"] for row in current]).astype(np.float64)
        anchors = np.median(measured[:, y0:y1], axis=1)
        displacement = measured - anchors[:, None]
        common_mode = np.median(displacement, axis=0)
        common_mode = gaussian_filter1d(common_mode, sigma=28.0)
        for index, row in enumerate(current):
            individual = displacement[index] - common_mode
            individual = gaussian_filter1d(individual, sigma=58.0)
            fitted = anchors[index] + common_mode + 0.30 * individual
            fitted += np.median(measured[index, y0:y1] - fitted[y0:y1])
            pitch = float(row["pitch_px"])
            fallback = smooth_centerline(measured[index], pitch)
            # The joint solution is authoritative in the measurement span;
            # smoothly blend to the independent path outside it.
            blend = np.zeros(measured.shape[1], dtype=np.float64)
            blend[max(0, y0 - 30) : min(len(blend), y1 + 30)] = 1.0
            blend = gaussian_filter1d(blend, sigma=10.0)
            final = blend * fitted + (1.0 - blend) * fallback
            row["path_x"] = final.astype(np.float32)
            row["centerline_raw_step_std_px"] = float(
                np.std(np.diff(measured[index, y0:y1]))
            )
            row["centerline_regularized_step_std_px"] = float(
                np.std(np.diff(final[y0:y1]))
            )


def constraint_confidence(
    guide_endpoint: length.EndpointMeasurement,
    raw_endpoint: length.EndpointMeasurement,
    guide_width: dict,
    raw_width: dict,
) -> dict:
    """Combine guide repeatability and raw/guide agreement into confidence."""
    endpoint_delta = max(
        abs(float(guide_endpoint.top - raw_endpoint.top)),
        abs(float(guide_endpoint.bottom - raw_endpoint.bottom)),
    )
    length_delta = abs(float(guide_endpoint.length - raw_endpoint.length))
    gw, rw = guide_width["median"], raw_width["median"]
    width_relative_delta = (
        abs(float(gw - rw)) / max(float(gw), 1e-8)
        if gw is not None and rw is not None
        else None
    )
    repeatability = math.exp(-0.5 * (float(guide_endpoint.uncertainty) / 1.50) ** 2)
    snr = min(float(guide_endpoint.top_snr), float(guide_endpoint.bottom_snr))
    snr_score = float(np.clip((snr - 2.0) / 8.0, 0.0, 1.0))
    sample_score = float(np.clip(guide_width["samples"] / 18.0, 0.0, 1.0))
    endpoint_agreement = math.exp(-0.5 * (endpoint_delta / 2.0) ** 2)
    length_agreement = math.exp(-0.5 * (length_delta / 2.5) ** 2)
    width_agreement = (
        math.exp(-0.5 * (width_relative_delta / 0.20) ** 2)
        if width_relative_delta is not None
        else 0.55
    )
    # The guide has priority; raw agreement is a validation term rather than
    # an averaging term that could pull geometry back toward raw noise.
    guide_quality = (repeatability * max(snr_score, 0.15) * max(sample_score, 0.25)) ** (1.0 / 3.0)
    evidence_agreement = (endpoint_agreement * length_agreement * width_agreement) ** (1.0 / 3.0)
    confidence = float(np.clip(0.68 * guide_quality + 0.32 * evidence_agreement, 0.0, 1.0))
    return {
        "constraint_confidence": confidence,
        "endpoint_agreement_max_abs_px": float(endpoint_delta),
        "length_agreement_abs_px": float(length_delta),
        "width_agreement_relative": width_relative_delta,
        "guide_quality_score": float(guide_quality),
        "raw_guide_agreement_score": float(evidence_agreement),
    }


def measure_layers(
    source: np.ndarray,
    left_roi: base.Roi,
    right_roi: base.Roi,
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
    guide: np.ndarray | None = None,
) -> list[dict]:
    guide = source if guide is None else guide
    if guide.shape != source.shape:
        raise ValueError(f"Guide shape {guide.shape} does not match source shape {source.shape}")
    rows = []
    for side, roi in (("left", left_roi), ("right", right_roi)):
        centers, pitch, _ = length.detect_centers(guide, roi)
        for index, center in enumerate(centers, start=1):
            measured_path = length.track_ridge(
                guide, center, pitch, top_range[0], bottom_range[1]
            ).astype(np.float32)
            guide_endpoint = length.measure_endpoints(
                guide, center, pitch, top_range, bottom_range, path_x=measured_path
            )
            raw_endpoint = length.measure_endpoints(
                source,
                center,
                pitch,
                top_range,
                bottom_range,
                hint=guide_endpoint,
                association_radius=4,
                path_x=measured_path,
            )
            condition_path = smooth_centerline(measured_path, pitch)
            guide_width = measure_layer_width(
                guide, measured_path, guide_endpoint.top, guide_endpoint.bottom, pitch
            )
            raw_width = measure_layer_width(
                source, measured_path, guide_endpoint.top, guide_endpoint.bottom, pitch
            )
            confidence = constraint_confidence(
                guide_endpoint, raw_endpoint, guide_width, raw_width
            )
            reliable = bool(
                guide_width["median"] is not None
                and guide_endpoint.uncertainty <= 3.0
                and min(guide_endpoint.top_snr, guide_endpoint.bottom_snr) >= 4.0
                and confidence["constraint_confidence"] >= 0.58
            )
            rows.append({
                "side": side,
                "layer_id": f"{side[0].upper()}{index:02d}",
                "center_x_px": float(center),
                "pitch_px": float(pitch),
                "top_y_px": float(guide_endpoint.top),
                "bottom_y_px": float(guide_endpoint.bottom),
                "length_px": float(guide_endpoint.length),
                "width_median_px": guide_width["median"],
                "width_p10_px": guide_width["p10"],
                "width_p90_px": guide_width["p90"],
                "width_sample_count": guide_width["samples"],
                "endpoint_uncertainty_px": float(guide_endpoint.uncertainty),
                "top_snr": float(guide_endpoint.top_snr),
                "bottom_snr": float(guide_endpoint.bottom_snr),
                "measurement_source": "blind_denoised_guide",
                "guide_top_y_px": float(guide_endpoint.top),
                "guide_bottom_y_px": float(guide_endpoint.bottom),
                "guide_length_px": float(guide_endpoint.length),
                "guide_width_median_px": guide_width["median"],
                "raw_validation_top_y_px": float(raw_endpoint.top),
                "raw_validation_bottom_y_px": float(raw_endpoint.bottom),
                "raw_validation_length_px": float(raw_endpoint.length),
                "raw_validation_width_median_px": raw_width["median"],
                "raw_validation_width_p10_px": raw_width["p10"],
                "raw_validation_width_p90_px": raw_width["p90"],
                "raw_validation_width_sample_count": raw_width["samples"],
                "raw_validation_endpoint_uncertainty_px": float(raw_endpoint.uncertainty),
                "raw_validation_top_snr": float(raw_endpoint.top_snr),
                "raw_validation_bottom_snr": float(raw_endpoint.bottom_snr),
                **confidence,
                "reference_quality": "pass" if reliable else "review",
                "path_x": condition_path,
                "measurement_path_x": measured_path,
            })
    regularize_centerline_bundles(rows, top_range, bottom_range)
    # Re-measure every scalar on the final joint centerline so the tabular
    # constraints, rendered blueprint and hard projection share coordinates.
    for row in rows:
        path = row["path_x"]
        center = row["center_x_px"]
        pitch = row["pitch_px"]
        guide_endpoint = length.measure_endpoints(
            guide, center, pitch, top_range, bottom_range, path_x=path
        )
        raw_endpoint = length.measure_endpoints(
            source,
            center,
            pitch,
            top_range,
            bottom_range,
            hint=guide_endpoint,
            association_radius=4,
            path_x=path,
        )
        guide_width = measure_layer_width(
            guide, path, guide_endpoint.top, guide_endpoint.bottom, pitch
        )
        raw_width = measure_layer_width(
            source, path, guide_endpoint.top, guide_endpoint.bottom, pitch
        )
        confidence = constraint_confidence(
            guide_endpoint, raw_endpoint, guide_width, raw_width
        )
        reliable = bool(
            guide_width["median"] is not None
            and guide_endpoint.uncertainty <= 3.0
            and min(guide_endpoint.top_snr, guide_endpoint.bottom_snr) >= 4.0
            and confidence["constraint_confidence"] >= 0.58
        )
        row.update({
            "top_y_px": float(guide_endpoint.top),
            "bottom_y_px": float(guide_endpoint.bottom),
            "length_px": float(guide_endpoint.length),
            "width_median_px": guide_width["median"],
            "width_p10_px": guide_width["p10"],
            "width_p90_px": guide_width["p90"],
            "width_sample_count": guide_width["samples"],
            "endpoint_uncertainty_px": float(guide_endpoint.uncertainty),
            "top_snr": float(guide_endpoint.top_snr),
            "bottom_snr": float(guide_endpoint.bottom_snr),
            "guide_top_y_px": float(guide_endpoint.top),
            "guide_bottom_y_px": float(guide_endpoint.bottom),
            "guide_length_px": float(guide_endpoint.length),
            "guide_width_median_px": guide_width["median"],
            "raw_validation_top_y_px": float(raw_endpoint.top),
            "raw_validation_bottom_y_px": float(raw_endpoint.bottom),
            "raw_validation_length_px": float(raw_endpoint.length),
            "raw_validation_width_median_px": raw_width["median"],
            "raw_validation_width_p10_px": raw_width["p10"],
            "raw_validation_width_p90_px": raw_width["p90"],
            "raw_validation_width_sample_count": raw_width["samples"],
            "raw_validation_endpoint_uncertainty_px": float(raw_endpoint.uncertainty),
            "raw_validation_top_snr": float(raw_endpoint.top_snr),
            "raw_validation_bottom_snr": float(raw_endpoint.bottom_snr),
            **confidence,
            "reference_quality": "pass" if reliable else "review",
        })
    return rows


def measure_gaps(layers: list[dict]) -> list[dict]:
    gaps = []
    for side in ("left", "right"):
        current = sorted((row for row in layers if row["side"] == side), key=lambda row: row["center_x_px"])
        for index, (a, b) in enumerate(zip(current, current[1:]), start=1):
            wa = a["width_median_px"] or 0.0
            wb = b["width_median_px"] or 0.0
            gap_top = float(max(a["top_y_px"], b["top_y_px"]))
            gap_bottom = float(min(a["bottom_y_px"], b["bottom_y_px"]))
            y0 = max(0, int(math.ceil(gap_top)))
            y1 = min(len(a["path_x"]) - 1, int(math.floor(gap_bottom)))
            if y1 < y0:
                ys = np.asarray([int(np.clip(round(0.5 * (gap_top + gap_bottom)), 0, len(a["path_x"]) - 1))])
            else:
                ys = np.arange(
                    y0,
                    y1 + 1,
                    max(1, int(round((y1 - y0 + 1) / 80.0))),
                )
            spacings = np.asarray(
                [float(b["path_x"][y] - a["path_x"][y]) for y in ys], dtype=np.float64
            )
            gap_widths = np.maximum(0.0, spacings - 0.5 * (wa + wb))
            raw_wa = a.get("raw_validation_width_median_px")
            raw_wb = b.get("raw_validation_width_median_px")
            has_raw_width = raw_wa is not None and raw_wb is not None
            raw_gap_widths = (
                np.maximum(0.0, spacings - 0.5 * (float(raw_wa) + float(raw_wb)))
                if has_raw_width else None
            )
            raw_top_a = a.get("raw_validation_top_y_px")
            raw_top_b = b.get("raw_validation_top_y_px")
            raw_bottom_a = a.get("raw_validation_bottom_y_px")
            raw_bottom_b = b.get("raw_validation_bottom_y_px")
            has_raw_length = all(
                value is not None
                for value in (raw_top_a, raw_top_b, raw_bottom_a, raw_bottom_b)
            )
            raw_gap_top = float(max(raw_top_a, raw_top_b)) if has_raw_length else None
            raw_gap_bottom = float(min(raw_bottom_a, raw_bottom_b)) if has_raw_length else None
            raw_gap_length = (
                float(max(0.0, raw_gap_bottom - raw_gap_top))
                if has_raw_length else None
            )
            confidence = float(min(
                a.get("constraint_confidence", 1.0), b.get("constraint_confidence", 1.0)
            ))
            gaps.append({
                "side": side,
                "gap_id": f"{side[0].upper()}G{index:02d}",
                "left_layer_id": a["layer_id"],
                "right_layer_id": b["layer_id"],
                "center_spacing_px": float(np.median(spacings)),
                "center_spacing_p10_px": float(np.percentile(spacings, 10.0)),
                "center_spacing_p90_px": float(np.percentile(spacings, 90.0)),
                "gap_width_px": float(np.median(gap_widths)),
                "gap_width_p10_px": float(np.percentile(gap_widths, 10.0)),
                "gap_width_p90_px": float(np.percentile(gap_widths, 90.0)),
                "gap_width_sample_count": int(len(gap_widths)),
                "raw_validation_gap_width_px": (
                    float(np.median(raw_gap_widths)) if raw_gap_widths is not None else None
                ),
                "raw_validation_gap_width_p10_px": (
                    float(np.percentile(raw_gap_widths, 10.0))
                    if raw_gap_widths is not None else None
                ),
                "raw_validation_gap_width_p90_px": (
                    float(np.percentile(raw_gap_widths, 90.0))
                    if raw_gap_widths is not None else None
                ),
                "top_y_px": gap_top,
                "bottom_y_px": gap_bottom,
                "length_px": float(max(0.0, gap_bottom - gap_top)),
                "raw_validation_top_y_px": raw_gap_top,
                "raw_validation_bottom_y_px": raw_gap_bottom,
                "raw_validation_length_px": raw_gap_length,
                "raw_guide_gap_width_abs_delta_px": (
                    float(abs(np.median(gap_widths) - np.median(raw_gap_widths)))
                    if raw_gap_widths is not None else None
                ),
                "raw_guide_gap_length_abs_delta_px": (
                    float(abs(max(0.0, gap_bottom - gap_top) - raw_gap_length))
                    if raw_gap_length is not None else None
                ),
                "constraint_confidence": confidence,
                "reference_quality": "pass" if confidence >= 0.58 else "review",
                "left_path_x": a["path_x"],
                "right_path_x": b["path_x"],
            })
    return gaps


def layer_polygon(row: dict, image_shape: tuple[int, int]) -> list[tuple[float, float]]:
    y0 = max(0, int(math.ceil(row["top_y_px"])))
    y1 = min(image_shape[0] - 1, int(math.floor(row["bottom_y_px"])))
    width = row["width_median_px"] or max(1.0, 0.28 * row["pitch_px"])
    ys = np.arange(y0, y1 + 1, 2, dtype=np.int32)
    if not len(ys) or ys[-1] != y1:
        ys = np.append(ys, y1)
    path = row["path_x"]
    left = [(float(path[y] - 0.5 * width), float(y)) for y in ys]
    right = [(float(path[y] + 0.5 * width), float(y)) for y in ys[::-1]]
    return left + right


def render_conditions(
    source: np.ndarray,
    guide: np.ndarray,
    layers: list[dict],
    gaps: list[dict],
    outdir: Path,
) -> None:
    h, w = source.shape
    mask_image = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    condition = Image.new("RGB", (w, h), (0, 0, 0))
    condition_draw = ImageDraw.Draw(condition)
    # Render the full interlayer corridors first. This explicitly conditions
    # both material and void dimensions rather than encoding only gap centers.
    for gap in gaps:
        y0, y1 = int(round(gap["top_y_px"])), int(round(gap["bottom_y_px"]))
        ys = np.arange(max(0, y0), min(h, y1 + 1), 3, dtype=np.int32)
        if len(ys):
            left = [(float(gap["left_path_x"][y]), float(y)) for y in ys]
            right = [(float(gap["right_path_x"][y]), float(y)) for y in ys[::-1]]
            blue = int(round(70 + 150 * gap.get("constraint_confidence", 1.0)))
            condition_draw.polygon(left + right, fill=(0, 0, blue))
    for row in layers:
        polygon = layer_polygon(row, source.shape)
        mask_draw.polygon(polygon, fill=255)
        confidence = float(row.get("constraint_confidence", 1.0))
        body = int(round(150 + 95 * confidence))
        condition_draw.polygon(polygon, fill=(body, body, body))
        y0, y1 = int(round(row["top_y_px"])), int(round(row["bottom_y_px"]))
        centerline = [(float(row["path_x"][y]), float(y)) for y in range(max(0, y0), min(h, y1 + 1), 3)]
        if len(centerline) >= 2:
            condition_draw.line(centerline, fill=(0, 255, 0), width=1)
        for y in (y0, y1):
            if 0 <= y < h:
                x = float(row["path_x"][y])
                half = 0.5 * (row["width_median_px"] or max(1.0, 0.28 * row["pitch_px"]))
                condition_draw.line((x - half - 1, y, x + half + 1, y), fill=(255, 40, 40), width=2)
    for gap in gaps:
        y0, y1 = int(round(gap["top_y_px"])), int(round(gap["bottom_y_px"]))
        points = []
        for y in range(max(0, y0), min(h, y1 + 1), 5):
            x = 0.5 * (float(gap["left_path_x"][y]) + float(gap["right_path_x"][y]))
            points.append((x, float(y)))
        if len(points) >= 2:
            condition_draw.line(points, fill=(40, 120, 255), width=1)

    mask_image.save(outdir / "SHAPE_exact_lamella_mask.png")
    imwrite(
        outdir / "SHAPE_exact_lamella_mask_16bit.tif",
        np.asarray(mask_image, dtype=np.uint16) * 257,
        photometric="minisblack",
        description="STRUCTURAL_CONDITION: lamella envelopes measured on blind-denoised guide; not an enhanced image.",
    )
    condition.save(outdir / "SHAPE_generation_condition.png")

    lo, hi = (float(v) for v in np.percentile(guide, (0.4, 99.7)))
    gray = np.rint(np.clip((guide - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
    background = Image.fromarray(gray, mode="L").convert("RGB")
    overlay = Image.blend(background, condition, 0.42)
    overlay.save(outdir / "SHAPE_measurement_overlay.png")

    raw_lo, raw_hi = (float(v) for v in np.percentile(source, (0.4, 99.7)))
    raw_gray = np.rint(
        np.clip((source - raw_lo) / max(raw_hi - raw_lo, 1e-8), 0.0, 1.0) * 255
    ).astype(np.uint8)
    raw_background = Image.fromarray(raw_gray, mode="L").convert("RGB")
    Image.blend(raw_background, condition, 0.42).save(
        outdir / "SHAPE_raw_validation_overlay.png"
    )


def serializable(row: dict) -> dict:
    return {key: value for key, value in row.items() if not isinstance(value, np.ndarray)}


def write_csv(path: Path, rows: list[dict]) -> None:
    clean = [serializable(row) for row in rows]
    if not clean:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean[0]))
        writer.writeheader()
        writer.writerows(clean)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument(
        "--guide", type=Path,
        help="Blind-denoised measurement guide. Defaults to source for backward compatibility.",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    ap.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    ap.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    ap.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    guide, guide_info = base.load_gray(args.guide) if args.guide else (source, source_info)
    layers = measure_layers(
        source,
        args.left_roi.clamp(source.shape),
        args.right_roi.clamp(source.shape),
        args.top_range,
        args.bottom_range,
        guide=guide,
    )
    gaps = measure_gaps(layers)
    render_conditions(source, guide, layers, gaps, args.outdir)
    write_csv(args.outdir / "lamella_geometry.csv", layers)
    write_csv(args.outdir / "interlayer_geometry.csv", gaps)

    widths = np.asarray([row["width_median_px"] for row in layers if row["width_median_px"] is not None])
    lengths = np.asarray([row["length_px"] for row in layers])
    gap_widths = np.asarray([row["gap_width_px"] for row in gaps])
    gap_lengths = np.asarray([row["length_px"] for row in gaps])
    payload = {
        "completed": True,
        "source": source_info,
        "measurement_guide": guide_info,
        "measurement_design": {
            "primary_geometry": "blind-denoised guide",
            "raw_role": "matched-coordinate validation and confidence only",
            "generation_target": "original raw image",
            "fusion_rule": "guide geometry retained; raw evidence is not averaged into coordinates",
        },
        "coordinate_system": "original image pixels",
        "condition_encoding": {
            "white": "guide-measured lamella body; brightness encodes confidence",
            "green": "tracked lamella centerline",
            "red": "subpixel endpoint cap rounded to the rendered pixel grid",
            "blue": "measured interlayer corridor plus centerline over the common longitudinal interval",
            "centerline_fit": "joint same-side bundle common-mode fit plus 30% strongly smoothed individual residual; pitch-adaptive fallback outside measurement span",
        },
        "summary": {
            "layer_count": len(layers),
            "reliable_layer_count": sum(row["reference_quality"] == "pass" for row in layers),
            "review_required_count": sum(row["reference_quality"] == "review" for row in layers),
            "constraint_confidence_median": float(np.median([
                row["constraint_confidence"] for row in layers
            ])),
            "raw_guide_endpoint_agreement_p95_px": float(np.percentile([
                row["endpoint_agreement_max_abs_px"] for row in layers
            ], 95.0)),
            "raw_guide_width_agreement_relative_p95": float(np.percentile([
                row["width_agreement_relative"] for row in layers
                if row["width_agreement_relative"] is not None
            ], 95.0)),
            "interlayer_count": len(gaps),
            "lamella_width_median_px": float(np.median(widths)),
            "lamella_width_p05_px": float(np.percentile(widths, 5.0)),
            "lamella_width_p95_px": float(np.percentile(widths, 95.0)),
            "lamella_length_median_px": float(np.median(lengths)),
            "interlayer_width_median_px": float(np.median(gap_widths)),
            "interlayer_length_median_px": float(np.median(gap_lengths)),
            "raw_guide_interlayer_width_abs_delta_p95_px": float(np.percentile([
                row["raw_guide_gap_width_abs_delta_px"] for row in gaps
                if row["raw_guide_gap_width_abs_delta_px"] is not None
            ], 95.0)),
            "raw_guide_interlayer_length_abs_delta_p95_px": float(np.percentile([
                row["raw_guide_gap_length_abs_delta_px"] for row in gaps
                if row["raw_guide_gap_length_abs_delta_px"] is not None
            ], 95.0)),
        },
        "layers": [serializable(row) for row in layers],
        "interlayers": [serializable(row) for row in gaps],
    }
    (args.outdir / "shape_constraint_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed": True, **payload["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
