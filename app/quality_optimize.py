#!/usr/bin/env python3
"""Quality-first refinement for the previous self-supervised blind denoiser.

The objective is visual edge clarity, local contrast and noise suppression. It
does not perform length measurement and never mixes generative pixels into the
16-bit output. Parameters are selected by a small deterministic grid search
with layer-count, center-shift, FWHM, SSIM and noise-roughness guardrails.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, sobel
from scipy.signal import find_peaks, peak_widths
from skimage import exposure
from skimage.metrics import structural_similarity
from tifffile import imwrite

import pipeline as base


def analyze_region(img: np.ndarray, roi: base.Roi) -> dict:
    crop = img[roi.slices()]
    profile = gaussian_filter1d(crop.mean(axis=0), sigma=1.0)
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
    union = base.Roi(min(left.y0, right.y0), max(left.y1, right.y1), left.x0, right.x1)
    ssim = float(structural_similarity(baseline[union.slices()], candidate[union.slices()], data_range=1.0))
    guardrail = bool(count_ok and shift <= 1.0 and width_change <= 0.05 and roughness_ratio <= 1.03 and ssim >= 0.985)
    score = 2.4 * (edge_gain - 1.0) + 1.0 * (contrast_gain - 1.0) - 2.0 * max(0.0, roughness_ratio - 1.0)
    return {
        **metrics,
        "edge_gain": edge_gain,
        "contrast_gain": contrast_gain,
        "roughness_ratio": roughness_ratio,
        "max_center_shift_px": shift,
        "max_fwhm_relative_change": float(width_change),
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
    final: np.ndarray,
    display: np.ndarray,
    roi: base.Roi,
) -> None:
    # Use the same target crop in every panel so the subtle edge/noise changes
    # remain visible at normal screen zoom. The separately saved preview is the
    # complete 2200 x 1600 image.
    ys, xs = roi.slices()
    items = [
        panel(observed[ys, xs], "Original 16-bit", size=(900, 390)),
        panel(baseline[ys, xs], "Previous blind-denoised model", size=(900, 390)),
        panel(final[ys, xs], "Balanced quality refinement 16-bit", size=(900, 390)),
        panel(display[ys, xs], "Display-only local contrast", size=(900, 390)),
    ]
    canvas = Image.new("L", (1800, 836), 0)
    for i, item in enumerate(items):
        canvas.paste(item, ((i % 2) * 900, (i // 2) * 418))
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True, help="previous blind-denoised uint16 TIFF")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--target-roi", type=base.parse_roi, default=base.Roi(620, 1210, 320, 1880))
    ap.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(700, 1110, 370, 970))
    ap.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(700, 1110, 1220, 1830))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    observed, source_info = base.load_gray(args.source)
    baseline, baseline_info = base.load_gray(args.baseline)
    if observed.shape != baseline.shape:
        raise ValueError(f"shape mismatch: source={observed.shape}, baseline={baseline.shape}")
    target = args.target_roi.clamp(observed.shape)
    left, right = args.left_roi.clamp(observed.shape), args.right_roi.clamp(observed.shape)
    observed_metrics = {"left": analyze_region(observed, left), "right": analyze_region(observed, right)}
    baseline_metrics = {"left": analyze_region(baseline, left), "right": analyze_region(baseline, right)}

    grid = []
    best = None
    best_image = None
    for edge_amount in (0.08, 0.14, 0.20, 0.26, 0.32):
        for structure_gain in (0.03, 0.06, 0.09, 0.12):
            crop = quality_refine_crop(baseline[target.slices()], edge_amount, structure_gain)
            candidate = base.feather_insert(baseline, crop, target, ramp=18)
            metrics = evaluate(candidate, baseline, left, right, baseline_metrics)
            item = {"edge_amount": edge_amount, "structure_gain": structure_gain, **metrics}
            grid.append(item)
            if metrics["guardrail_pass"] and (best is None or metrics["score"] > best["score"]):
                best, best_image = item, candidate
    if best is None or best_image is None:
        raise RuntimeError("No quality candidate passed the configured guardrails")

    display = display_version(best_image, target)
    imwrite(
        args.outdir / "QUALITY_balanced_16bit.tif",
        base.to_uint16(best_image),
        photometric="minisblack",
        description="Blind-denoised quality refinement; no generative pixels; not a physical-resolution claim.",
    )
    base.save_preview(args.outdir / "QUALITY_balanced_preview.png", best_image)
    Image.fromarray(np.rint(display * 255).astype(np.uint8), mode="L").save(args.outdir / "QUALITY_display_only.png")
    save_comparison(args.outdir / "QUALITY_comparison.png", observed, baseline, best_image, display, target)

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

    payload = {
        "source": source_info,
        "baseline": baseline_info,
        "objective": "edge clarity + local contrast + denoising; no length measurement",
        "selected": best,
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
            "flat_roughness_ratio_max": 1.03,
            "ssim_vs_blind_baseline_min": 0.985,
        },
        "grid": grid,
        "notes": [
            "QUALITY_balanced_16bit.tif contains no generative pixels.",
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
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
