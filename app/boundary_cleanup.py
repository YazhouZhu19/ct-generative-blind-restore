#!/usr/bin/env python3
"""Geometry-gated boundary cleanup after the complete v5 enhancement chain.

The unregistered reference image contributes visual intent only: a quiet dark
exterior, sheet-like measurable ridges, and an intact central highlight.  All
geometry masks are built from raw 16-bit endpoint measurements.  The output is
computed exclusively from the enhanced measurement image; no reference or raw
pixel is copied into it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.metrics import structural_similarity
from tifffile import imwrite

import pipeline as base
import quality_optimize as quality


def build_endpoint_envelope_masks(
    shape: tuple[int, int],
    references: list[dict],
    target: base.Roi,
    boundary_radius: float = 8.0,
    exterior_radius: float = 24.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build soft boundary/exterior masks from raw endpoint coordinates."""
    h, w = shape
    th, tw = target.y1 - target.y0, target.x1 - target.x0
    yy = np.arange(target.y0, target.y1, dtype=np.float32)[:, None]
    boundary = np.zeros((th, tw), dtype=np.float32)
    exterior = np.zeros((th, tw), dtype=np.float32)
    interior = np.zeros((th, tw), dtype=np.float32)
    side_details: dict[str, dict] = {}
    for side in ("left", "right"):
        side_refs = [ref for ref in references if ref["side"] == side]
        if not side_refs:
            continue
        axis = np.arange(h, dtype=np.float64)
        top_points = []
        bottom_points = []
        pitches = []
        for ref in side_refs:
            raw = ref["raw"]
            top_x = float(np.interp(raw.top, axis, ref["path_x"]))
            bottom_x = float(np.interp(raw.bottom, axis, ref["path_x"]))
            top_points.append((top_x, float(raw.top)))
            bottom_points.append((bottom_x, float(raw.bottom)))
            pitches.append(float(ref["pitch"]))
        top_points.sort()
        bottom_points.sort()
        pitch = float(np.median(pitches))
        x_min = max(target.x0, int(np.floor(min(p[0] for p in top_points + bottom_points) - 0.55 * pitch)))
        x_max = min(target.x1, int(np.ceil(max(p[0] for p in top_points + bottom_points) + 0.55 * pitch)) + 1)
        if x_max <= x_min:
            continue
        gx = np.arange(x_min, x_max, dtype=np.float64)
        top_curve = np.interp(gx, [p[0] for p in top_points], [p[1] for p in top_points])
        bottom_curve = np.interp(gx, [p[0] for p in bottom_points], [p[1] for p in bottom_points])
        top_curve = gaussian_filter1d(top_curve, sigma=1.2).astype(np.float32)[None, :]
        bottom_curve = gaussian_filter1d(bottom_curve, sigma=1.2).astype(np.float32)[None, :]
        distance_top = yy - top_curve
        distance_bottom = bottom_curve - yy
        inside_distance = np.minimum(distance_top, distance_bottom)
        signed_outside = np.maximum(-distance_top, -distance_bottom)
        current_boundary = np.exp(-0.5 * np.square(inside_distance / max(boundary_radius, 1e-3)))
        current_exterior = np.where(
            signed_outside > 0.0,
            np.exp(-0.5 * np.square(signed_outside / max(exterior_radius, 1e-3))),
            0.0,
        )
        current_interior = np.where(
            inside_distance >= 0.0,
            1.0 - np.exp(-0.5 * np.square(inside_distance / 10.0)),
            0.0,
        )
        local = slice(x_min - target.x0, x_max - target.x0)
        boundary[:, local] = np.maximum(boundary[:, local], current_boundary.astype(np.float32))
        exterior[:, local] = np.maximum(exterior[:, local], current_exterior.astype(np.float32))
        interior[:, local] = np.maximum(interior[:, local], current_interior.astype(np.float32))
        side_details[side] = {
            "reference_count": len(side_refs),
            "x_range_px": [x_min, x_max],
            "median_pitch_px": pitch,
            "top_y_range_px": [float(np.min(top_curve)), float(np.max(top_curve))],
            "bottom_y_range_px": [float(np.min(bottom_curve)), float(np.max(bottom_curve))],
        }
    masks = {
        "boundary": np.clip(boundary, 0.0, 1.0),
        "exterior": np.clip(exterior, 0.0, 1.0),
        "interior": np.clip(interior, 0.0, 1.0),
    }
    return masks, {
        "method": "raw-endpoint upper/lower envelope interpolation",
        "boundary_radius_px": float(boundary_radius),
        "exterior_radius_px": float(exterior_radius),
        "sides": side_details,
        "reference_pixel_writeback": False,
        "raw_pixel_writeback": False,
    }


def clean_candidate(
    image: np.ndarray,
    references: list[dict],
    target: base.Roi,
    masks: dict[str, np.ndarray],
    fine_strength: float,
    halo_strength: float,
    boundary_sharpen: float,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    ys, xs = target.slices()
    crop = image[ys, xs]
    gate_source = gaussian_filter(crop, sigma=0.70)
    gy, gx = np.gradient(gate_source)
    gradient = np.hypot(gx, gy)
    positive = crop[crop > 0]
    intensity_floor = float(np.percentile(positive, 5.0)) if positive.size else 0.0
    sample = gradient[crop > intensity_floor]
    low_threshold = float(np.percentile(sample, 52.0)) + 1e-8
    edge_threshold = float(np.percentile(sample, 72.0)) + 1e-8
    low_gate = np.exp(-np.square(gradient / low_threshold)).astype(np.float32)
    vertical_edge_gate = 1.0 - np.exp(-np.square(np.abs(gy) / edge_threshold)).astype(np.float32)
    endpoint_protection = quality.endpoint_coordinate_protection(
        image.shape, references, radius_y=2, radius_x=2
    )[ys, xs]

    # The sheets are predominantly vertical.  Denoising along their long axis
    # removes granular variation without averaging across the two transverse
    # edges that define measurable thickness.  A separate isotropic estimate is
    # used only outside the raw endpoint envelope.
    longitudinal_fine = gaussian_filter(crop, sigma=(2.00, 0.18))
    exterior_fine = gaussian_filter(crop, sigma=1.15)
    medium = gaussian_filter(crop, sigma=2.20)
    broad = gaussian_filter(crop, sigma=7.0)
    positive_halo = np.maximum(medium - broad, 0.0)
    vertical_blur = gaussian_filter(crop, sigma=(1.15, 0.25))
    vertical_detail = crop - vertical_blur

    interior_fine_gate = low_gate * masks["interior"] * (1.0 - endpoint_protection)
    exterior_fine_gate = low_gate * masks["exterior"] * (1.0 - endpoint_protection)
    halo_gate = masks["exterior"] * (0.30 + 0.70 * low_gate) * (1.0 - endpoint_protection)
    sharpen_gate = masks["boundary"] * vertical_edge_gate * (1.0 - endpoint_protection)
    delta = (
        float(fine_strength) * interior_fine_gate * (longitudinal_fine - crop)
        + float(fine_strength) * exterior_fine_gate * (exterior_fine - crop)
        - float(halo_strength) * halo_gate * positive_halo
        + float(boundary_sharpen) * sharpen_gate * vertical_detail
    )
    sigma_n = base.noise_sigma_mad(crop)
    cap = max(0.80 * sigma_n, 1.0 / 65535.0)
    cleaned_crop = np.clip(crop + np.clip(delta, -cap, cap), 0.0, 1.0).astype(np.float32)
    cleaned = base.feather_insert(image, cleaned_crop, target, ramp=18)
    gates = {
        "interior_fine": interior_fine_gate,
        "exterior_fine": exterior_fine_gate,
        "halo": halo_gate,
        "sharpen": sharpen_gate,
        "positive_halo": positive_halo,
    }
    return cleaned, {
        "method": "geometry-gated axial fine denoise, positive exterior-halo suppression, and zero-phase longitudinal boundary sharpening",
        "fine_strength": float(fine_strength),
        "halo_strength": float(halo_strength),
        "boundary_sharpen": float(boundary_sharpen),
        "interior_fine_sigma_px": [2.00, 0.18],
        "exterior_fine_sigma_px": 1.15,
        "halo_sigmas_px": [2.20, 7.0],
        "boundary_sharpen_sigma_px": [1.15, 0.25],
        "estimated_noise_sigma_normalized": sigma_n,
        "change_cap_normalized": cap,
        "reference_pixel_writeback": False,
        "raw_pixel_writeback": False,
    }, gates


def boundary_cleanliness_metrics(
    image: np.ndarray,
    target: base.Roi,
    masks: dict[str, np.ndarray],
) -> dict:
    crop = image[target.slices()]
    # Exclude the true endpoint band from the noise estimate.  Otherwise the
    # expected bright sheet tips dominate the metric and a cleaner exterior can
    # incorrectly look noisier after edge isolation.
    exterior = (masks["exterior"] > 0.18) & (masks["boundary"] < 0.20)
    if not np.any(exterior):
        raise RuntimeError("No exterior boundary support was constructed")
    highpass = crop - gaussian_filter(crop, sigma=1.10)
    axial_highpass = crop - gaussian_filter(crop, sigma=(2.00, 0.18))
    medium = gaussian_filter(crop, sigma=2.20)
    broad = gaussian_filter(crop, sigma=7.0)
    positive_halo = np.maximum(medium - broad, 0.0)
    boundary = masks["boundary"] > 0.20
    interior = masks["interior"] > 0.55
    return {
        "exterior_high_frequency_rms": float(np.sqrt(np.mean(np.square(highpass[exterior])))),
        "exterior_positive_halo_mean": float(np.mean(positive_halo[exterior])),
        "interior_axial_noise_rms": float(np.sqrt(np.mean(np.square(axial_highpass[interior])))),
        "boundary_vertical_gradient_median": float(np.median(np.abs(np.gradient(crop, axis=0))[boundary])),
    }


def save_gate_preview(path: Path, image: np.ndarray, target: base.Roi, masks: dict[str, np.ndarray]) -> None:
    crop = image[target.slices()]
    lo, hi = (float(v) for v in np.percentile(crop, (0.5, 99.7)))
    gray = np.rint(np.clip((crop - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[..., 0] = np.maximum(rgb[..., 0], np.rint(180 * masks["boundary"]).astype(np.uint8))
    rgb[..., 2] = np.maximum(rgb[..., 2], np.rint(180 * masks["exterior"]).astype(np.uint8))
    Image.fromarray(rgb, mode="RGB").save(path)


def save_comparison(path: Path, before: np.ndarray, after: np.ndarray, target: base.Roi) -> None:
    ys, xs = target.slices()
    a, b = before[ys, xs], after[ys, xs]
    lo, hi = (float(v) for v in np.percentile(a, (0.5, 99.7)))
    panels = []
    for values, title in ((a, "v5 input"), (b, "v6 boundary-clean"), (np.abs(b - a), "absolute delta")):
        if title == "absolute delta":
            scale = max(float(np.percentile(values, 99.7)), 1.0 / 65535.0)
            u8 = np.rint(np.clip(values / scale, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            u8 = np.rint(np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
        panel = Image.fromarray(u8, mode="L").resize((760, 288), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (760, 314), 0)
        canvas.paste(panel, (0, 26))
        ImageDraw.Draw(canvas).text((8, 7), title, fill=255, font=ImageFont.load_default())
        panels.append(canvas)
    out = Image.new("L", (2280, 314), 0)
    for index, panel in enumerate(panels):
        out.paste(panel, (760 * index, 0))
    out.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--reference-style", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--target-roi", type=base.parse_roi, default=base.Roi(600, 1240, 300, 1900))
    ap.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(700, 1110, 370, 970))
    ap.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(700, 1110, 1220, 1830))
    ap.add_argument("--left-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    ap.add_argument("--right-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    ap.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    ap.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    enhanced, enhanced_info = base.load_gray(args.input)
    if source.shape != enhanced.shape:
        raise ValueError(f"shape mismatch: source={source.shape}, enhanced={enhanced.shape}")
    target = args.target_roi.clamp(source.shape)
    left, right = args.left_roi.clamp(source.shape), args.right_roi.clamp(source.shape)
    references = quality.build_boundary_references(
        source,
        enhanced,
        args.left_body_roi.clamp(source.shape),
        args.right_body_roi.clamp(source.shape),
        args.top_range,
        args.bottom_range,
    )
    masks, mask_info = build_endpoint_envelope_masks(source.shape, references, target)
    before_clean = boundary_cleanliness_metrics(enhanced, target, masks)
    source_metrics = {"left": quality.analyze_region(source, left), "right": quality.analyze_region(source, right)}
    baseline_metrics = {"left": quality.analyze_region(enhanced, left), "right": quality.analyze_region(enhanced, right)}

    grid_parameters = (
        (0.00, 0.00, 0.00),
        (0.20, 0.40, 0.00),
        (0.35, 0.80, 0.00),
        (0.50, 1.20, 0.00),
        (0.65, 1.80, 0.00),
        (0.80, 2.50, 0.00),
        (0.90, 3.20, 0.00),
        (1.00, 4.00, 0.00),
    )
    grid = []
    selected = None
    for fine_strength, halo_strength, boundary_sharpen in grid_parameters:
        candidate, operation, gates = clean_candidate(
            enhanced,
            references,
            target,
            masks,
            fine_strength,
            halo_strength,
            boundary_sharpen,
        )
        appearance = quality.evaluate(candidate, enhanced, left, right, baseline_metrics, source_metrics)
        boundary, rows = quality.boundary_geometry(candidate, references, args.top_range, args.bottom_range)
        clean = boundary_cleanliness_metrics(candidate, target, masks)
        exterior_noise_reduction = 1.0 - clean["exterior_high_frequency_rms"] / max(
            before_clean["exterior_high_frequency_rms"], 1e-8
        )
        exterior_halo_reduction = 1.0 - clean["exterior_positive_halo_mean"] / max(
            before_clean["exterior_positive_halo_mean"], 1e-8
        )
        interior_noise_reduction = 1.0 - clean["interior_axial_noise_rms"] / max(
            before_clean["interior_axial_noise_rms"], 1e-8
        )
        ssim = float(structural_similarity(enhanced[target.slices()], candidate[target.slices()], data_range=1.0))
        edge_retention = appearance["edge_gain"]
        contrast_retention = appearance["contrast_gain"]
        width_pass = bool(appearance["max_fwhm_relative_change_vs_source"] <= 0.04)
        endpoint_pass = quality.boundary_guardrail(boundary)
        guardrail = bool(
            appearance["guardrail_pass"]
            and width_pass
            and endpoint_pass
            and edge_retention >= 0.985
            and contrast_retention >= 0.990
            and clean["boundary_vertical_gradient_median"] >= 0.94
            * before_clean["boundary_vertical_gradient_median"]
            and ssim >= 0.9985
        )
        score = (
            1.7 * exterior_noise_reduction
            + 1.1 * exterior_halo_reduction
            + 0.8 * interior_noise_reduction
            + 0.25 * max(
            0.0, clean["boundary_vertical_gradient_median"] / max(
                before_clean["boundary_vertical_gradient_median"], 1e-8
            ) - 1.0
            )
        )
        item = {
            **operation,
            "boundary_cleanliness": clean,
            "exterior_noise_reduction": exterior_noise_reduction,
            "exterior_halo_reduction": exterior_halo_reduction,
            "interior_axial_noise_reduction": interior_noise_reduction,
            "boundary_gradient_gain": clean["boundary_vertical_gradient_median"] / max(
                before_clean["boundary_vertical_gradient_median"], 1e-8
            ),
            "edge_acutance_retention": edge_retention,
            "local_contrast_retention": contrast_retention,
            "ssim_vs_v5_input": ssim,
            "max_fwhm_relative_change_vs_source": appearance["max_fwhm_relative_change_vs_source"],
            "boundary_geometry": boundary,
            "guardrail_pass": guardrail,
            "score": score,
        }
        grid.append(item)
        if guardrail and (selected is None or score > selected["summary"]["score"]):
            selected = {
                "summary": item,
                "image": candidate,
                "rows": rows,
                "gates": gates,
                "appearance": appearance,
            }
    if selected is None:
        raise RuntimeError("No boundary-cleanup candidate passed appearance, width, endpoint and length guardrails")

    final = selected["image"]
    imwrite(
        args.outdir / "QUALITY_boundary_clean_16bit.tif",
        base.to_uint16(final),
        photometric="minisblack",
        description="Geometry-gated v6 boundary cleanup; reference style only; no reference/raw pixel writeback.",
    )
    base.save_preview(args.outdir / "QUALITY_boundary_clean_preview.png", final)
    quality.save_boundary_overlay(args.outdir / "QUALITY_boundary_clean_overlay.png", final, selected["rows"], target)
    save_gate_preview(args.outdir / "AUDIT_boundary_cleanup_masks.png", enhanced, target, masks)
    save_comparison(args.outdir / "QUALITY_boundary_cleanup_comparison.png", enhanced, final, target)

    style_info = {
        "path": str(args.reference_style) if args.reference_style else None,
        "role": "unregistered visual intent only: quiet exterior, sheet-like measurable ridges, intact central highlight",
        "reference_pixel_writeback": False,
        "reference_used_for_geometry": False,
    }
    if args.reference_style:
        with Image.open(args.reference_style) as reference:
            style_info["dimensions"] = list(reference.size)
    payload = {
        "completed": True,
        "source": source_info,
        "input": enhanced_info,
        "reference_style": style_info,
        "geometry_masks": mask_info,
        "before_boundary_cleanliness": before_clean,
        "selected": selected["summary"],
        "appearance": selected["appearance"],
        "candidate_grid": grid,
        "guardrails": {
            "edge_acutance_retention_min": 0.985,
            "local_contrast_retention_min": 0.990,
            "ssim_vs_v5_input_min": 0.9985,
            "boundary_vertical_gradient_retention_min": 0.94,
            "median_fwhm_relative_change_vs_source_max": 0.04,
            "endpoint_shift_abs_p95_max_px": 0.35,
            "endpoint_shift_abs_max_px": 0.75,
            "length_delta_abs_p95_max_px": 0.50,
            "length_delta_abs_max_px": 1.00,
        },
    }
    (args.outdir / "boundary_cleanup_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completed": True,
        "output": str(args.outdir / "QUALITY_boundary_clean_16bit.tif"),
        "selected": selected["summary"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
