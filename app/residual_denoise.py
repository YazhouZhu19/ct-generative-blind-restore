#!/usr/bin/env python3
"""v7 residual denoising after the complete v6 measurement chain.

Lamella interiors use zero-phase axial residual shrinkage so transverse
thickness edges are not averaged.  The central highlight, which is excluded
from the lamella masks, receives a separately feathered non-local means pass.
Raw pixels provide endpoint coordinates only and are never written back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_nl_means
from tifffile import imwrite

import boundary_cleanup as boundary
import pipeline as base
import quality_optimize as quality


def soft_roi_mask(target: base.Roi, roi: base.Roi, ramp: int = 18) -> np.ndarray:
    """Return a feathered ROI mask in target-local coordinates."""
    mask = np.zeros((target.y1 - target.y0, target.x1 - target.x0), dtype=np.float32)
    y0, y1 = max(target.y0, roi.y0), min(target.y1, roi.y1)
    x0, x1 = max(target.x0, roi.x0), min(target.x1, roi.x1)
    if y1 <= y0 or x1 <= x0:
        return mask
    h, w = y1 - y0, x1 - x0
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])
    gate = np.clip(np.minimum(yy[:, None], xx[None, :]) / max(1, ramp), 0.0, 1.0)
    mask[y0 - target.y0 : y1 - target.y0, x0 - target.x0 : x1 - target.x0] = gate
    return mask


def prepare_components(
    image: np.ndarray,
    references: list[dict],
    target: base.Roi,
    central_roi: base.Roi,
    masks: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    ys, xs = target.slices()
    crop = image[ys, xs]
    sigma_n = base.noise_sigma_mad(crop)
    smooth = gaussian_filter(crop, sigma=0.70)
    gx = np.gradient(smooth, axis=1)
    positive = crop[crop > 0]
    floor = float(np.percentile(positive, 5.0)) if positive.size else 0.0
    sample = np.abs(gx)[crop > floor]
    transverse_threshold = float(np.percentile(sample, 62.0)) + 1e-8
    transverse_safe = np.exp(-np.square(np.abs(gx) / transverse_threshold)).astype(np.float32)
    endpoint_protection = quality.endpoint_coordinate_protection(
        image.shape, references, radius_y=5, radius_x=3
    )[ys, xs]

    axial_smooth = gaussian_filter(crop, sigma=(2.50, 0.18))
    axial_residual = crop - axial_smooth
    axial_noise_likelihood = np.exp(
        -np.power(np.abs(axial_residual) / max(2.60 * sigma_n, 1e-8), 4.0)
    ).astype(np.float32)
    lamella_gate = (
        masks["interior"]
        * transverse_safe
        * axial_noise_likelihood
        * (1.0 - endpoint_protection)
    )

    central_roi = central_roi.clamp(image.shape)
    central = image[central_roi.slices()]
    central_sigma = base.noise_sigma_mad(central)
    central_nlm = denoise_nl_means(
        central,
        h=0.85 * central_sigma,
        sigma=central_sigma,
        fast_mode=True,
        patch_size=5,
        patch_distance=5,
        channel_axis=None,
        preserve_range=True,
    ).astype(np.float32)
    central_delta = np.zeros_like(crop)
    cy0, cy1 = central_roi.y0 - target.y0, central_roi.y1 - target.y0
    cx0, cx1 = central_roi.x0 - target.x0, central_roi.x1 - target.x0
    central_delta[cy0:cy1, cx0:cx1] = central_nlm - central
    central_gate = soft_roi_mask(target, central_roi, ramp=18) * (1.0 - endpoint_protection)

    return {
        "crop": crop,
        "lamella_delta": -lamella_gate * axial_residual,
        "central_delta": central_gate * central_delta,
        "lamella_gate": lamella_gate,
        "central_gate": central_gate,
        "endpoint_protection": endpoint_protection,
    }, {
        "method": "lamella axial residual shrinkage plus central-highlight non-local means",
        "lamella_axial_sigma_px": [2.50, 0.18],
        "axial_noise_likelihood_scale_sigma": 2.60,
        "central_nlm_h_sigma": 0.85,
        "central_nlm_patch_size": 5,
        "central_nlm_patch_distance": 5,
        "estimated_noise_sigma_normalized": sigma_n,
        "central_noise_sigma_normalized": central_sigma,
        "transverse_edge_threshold": transverse_threshold,
        "raw_pixel_writeback": False,
        "reference_pixel_writeback": False,
    }


def apply_candidate(
    image: np.ndarray,
    target: base.Roi,
    components: dict[str, np.ndarray],
    sigma_n: float,
    lamella_strength: float,
    central_strength: float,
) -> tuple[np.ndarray, dict]:
    delta = (
        float(lamella_strength) * components["lamella_delta"]
        + float(central_strength) * components["central_delta"]
    )
    cap = max(0.65 * sigma_n, 1.0 / 65535.0)
    crop = np.clip(components["crop"] + np.clip(delta, -cap, cap), 0.0, 1.0).astype(np.float32)
    result = base.feather_insert(image, crop, target, ramp=18)
    return result, {
        "lamella_strength": float(lamella_strength),
        "central_strength": float(central_strength),
        "change_cap_normalized": cap,
        "raw_pixel_writeback": False,
        "reference_pixel_writeback": False,
        "phase_response": "zero-phase symmetric filters",
    }


def central_noise_rms(image: np.ndarray, roi: base.Roi) -> float:
    crop = image[roi.slices()]
    residual = crop - gaussian_filter(crop, sigma=1.15)
    return float(np.sqrt(np.mean(np.square(residual))))


def save_mask_audit(path: Path, image: np.ndarray, target: base.Roi, components: dict[str, np.ndarray]) -> None:
    crop = image[target.slices()]
    lo, hi = (float(v) for v in np.percentile(crop, (0.5, 99.7)))
    gray = np.rint(np.clip((crop - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[..., 1] = np.maximum(rgb[..., 1], np.rint(180 * components["lamella_gate"]).astype(np.uint8))
    rgb[..., 2] = np.maximum(rgb[..., 2], np.rint(180 * components["central_gate"]).astype(np.uint8))
    rgb[..., 0] = np.maximum(rgb[..., 0], np.rint(200 * components["endpoint_protection"]).astype(np.uint8))
    Image.fromarray(rgb, mode="RGB").save(path)


def save_comparison(path: Path, before: np.ndarray, after: np.ndarray, target: base.Roi) -> None:
    a, b = before[target.slices()], after[target.slices()]
    lo, hi = (float(v) for v in np.percentile(a, (0.5, 99.7)))
    panels = []
    for values, title in (
        (a, "pre-residual input"),
        (b, "guarded residual-denoised"),
        (np.abs(b - a), "absolute delta"),
    ):
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
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--target-roi", type=base.parse_roi, default=base.Roi(600, 1240, 300, 1900))
    ap.add_argument("--central-roi", type=base.parse_roi, default=base.Roi(700, 1140, 985, 1205))
    ap.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(700, 1110, 370, 970))
    ap.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(700, 1110, 1220, 1830))
    ap.add_argument("--left-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    ap.add_argument("--right-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    ap.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    ap.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    current, current_info = base.load_gray(args.input)
    if source.shape != current.shape:
        raise ValueError(f"shape mismatch: source={source.shape}, input={current.shape}")
    target = args.target_roi.clamp(source.shape)
    central_roi = args.central_roi.clamp(source.shape)
    left, right = args.left_roi.clamp(source.shape), args.right_roi.clamp(source.shape)
    references = quality.build_boundary_references(
        source,
        current,
        args.left_body_roi.clamp(source.shape),
        args.right_body_roi.clamp(source.shape),
        args.top_range,
        args.bottom_range,
    )
    masks, mask_info = boundary.build_endpoint_envelope_masks(source.shape, references, target)
    components, method = prepare_components(current, references, target, central_roi, masks)
    sigma_n = method["estimated_noise_sigma_normalized"]
    baseline_clean = boundary.boundary_cleanliness_metrics(current, target, masks)
    baseline_center = central_noise_rms(current, central_roi)
    source_metrics = {"left": quality.analyze_region(source, left), "right": quality.analyze_region(source, right)}
    baseline_metrics = {"left": quality.analyze_region(current, left), "right": quality.analyze_region(current, right)}

    grid = []
    selected = None
    for lamella_strength, central_strength in (
        (0.00, 0.00),
        (0.20, 0.30),
        (0.30, 0.50),
        (0.40, 0.60),
        (0.50, 0.70),
        (0.60, 0.70),
        (0.65, 0.80),
    ):
        candidate, operation = apply_candidate(
            current, target, components, sigma_n, lamella_strength, central_strength
        )
        appearance = quality.evaluate(candidate, current, left, right, baseline_metrics, source_metrics)
        geometry, rows = quality.boundary_geometry(candidate, references, args.top_range, args.bottom_range)
        clean = boundary.boundary_cleanliness_metrics(candidate, target, masks)
        lamella_reduction = 1.0 - clean["interior_axial_noise_rms"] / max(
            baseline_clean["interior_axial_noise_rms"], 1e-8
        )
        center_reduction = 1.0 - central_noise_rms(candidate, central_roi) / max(baseline_center, 1e-8)
        target_ssim = float(structural_similarity(
            current[target.slices()], candidate[target.slices()], data_range=1.0
        ))
        center_mean_delta = float(np.mean(candidate[central_roi.slices()]) - np.mean(current[central_roi.slices()]))
        guardrail = bool(
            appearance["guardrail_pass"]
            and appearance["max_fwhm_relative_change_vs_source"] <= 0.04
            and quality.boundary_guardrail(geometry)
            and appearance["edge_gain"] >= 0.9975
            and appearance["contrast_gain"] >= 0.9970
            and target_ssim >= 0.9985
            and abs(center_mean_delta) <= 2.0e-4
        )
        score = 1.5 * lamella_reduction + 0.9 * center_reduction
        item = {
            **operation,
            "lamella_axial_noise_reduction": lamella_reduction,
            "central_noise_reduction": center_reduction,
            "central_mean_delta_normalized": center_mean_delta,
            "edge_acutance_retention": appearance["edge_gain"],
            "local_contrast_retention": appearance["contrast_gain"],
            "max_fwhm_relative_change_vs_source": appearance["max_fwhm_relative_change_vs_source"],
            "ssim_vs_v6_input": target_ssim,
            "boundary_geometry": geometry,
            "guardrail_pass": guardrail,
            "score": score,
        }
        grid.append(item)
        if guardrail and (selected is None or score > selected["summary"]["score"]):
            selected = {"summary": item, "image": candidate, "rows": rows, "appearance": appearance}
    if selected is None:
        raise RuntimeError("No residual-denoise candidate passed appearance and geometry guardrails")

    final = selected["image"]
    final_flat_roughness = float(np.median([
        selected["appearance"][side]["flat_region_roughness"] for side in ("left", "right")
    ]))
    source_flat_roughness = float(np.median([
        source_metrics[side]["flat_region_roughness"] for side in ("left", "right")
    ]))
    final_edge = float(np.median([
        selected["appearance"][side]["median_edge_acutance"] for side in ("left", "right")
    ]))
    source_edge = float(np.median([
        source_metrics[side]["median_edge_acutance"] for side in ("left", "right")
    ]))
    final_contrast = float(np.median([
        selected["appearance"][side]["profile_local_contrast_rms"] for side in ("left", "right")
    ]))
    source_contrast = float(np.median([
        source_metrics[side]["profile_local_contrast_rms"] for side in ("left", "right")
    ]))
    validation_vs_source = {
        "flat_high_frequency_reduction": 1.0 - final_flat_roughness / max(source_flat_roughness, 1e-8),
        "central_high_frequency_reduction": 1.0 - central_noise_rms(final, central_roi) / max(
            central_noise_rms(source, central_roi), 1e-8
        ),
        "edge_acutance_gain": final_edge / max(source_edge, 1e-8) - 1.0,
        "local_contrast_gain": final_contrast / max(source_contrast, 1e-8) - 1.0,
        "max_median_fwhm_relative_change": selected["summary"]["max_fwhm_relative_change_vs_source"],
    }
    imwrite(
        args.outdir / "QUALITY_v7_residual_denoised_16bit.tif",
        base.to_uint16(final),
        photometric="minisblack",
        description="v7 guarded residual denoising; zero-phase; no reference/raw pixel writeback.",
    )
    base.save_preview(args.outdir / "QUALITY_v7_residual_denoised_preview.png", final)
    quality.save_boundary_overlay(args.outdir / "QUALITY_v7_boundary_overlay.png", final, selected["rows"], target)
    save_mask_audit(args.outdir / "AUDIT_v7_residual_denoise_masks.png", current, target, components)
    save_comparison(args.outdir / "QUALITY_v7_comparison.png", current, final, target)
    payload = {
        "completed": True,
        "source": source_info,
        "input": current_info,
        "method": method,
        "geometry_masks": mask_info,
        "selected": selected["summary"],
        "appearance": selected["appearance"],
        "validation_vs_source": validation_vs_source,
        "candidate_grid": grid,
        "guardrails": {
            "edge_acutance_retention_min": 0.9975,
            "local_contrast_retention_min": 0.9970,
            "ssim_vs_v6_input_min": 0.9985,
            "central_mean_delta_abs_max_normalized": 2.0e-4,
            "median_fwhm_relative_change_vs_source_max": 0.04,
            "endpoint_shift_abs_p95_max_px": 0.35,
            "endpoint_shift_abs_max_px": 0.75,
            "length_delta_abs_p95_max_px": 0.50,
            "length_delta_abs_max_px": 1.00,
        },
    }
    (args.outdir / "residual_denoise_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completed": True,
        "output": str(args.outdir / "QUALITY_v7_residual_denoised_16bit.tif"),
        "selected": selected["summary"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
