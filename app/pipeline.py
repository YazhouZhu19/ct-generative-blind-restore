#!/usr/bin/env python3
"""Dual-track restoration pipeline for 16-bit industrial CT/radiography.

The two tracks are intentionally kept separate:

* ``measurement``: a single-image self-supervised blind denoiser inspired by
  APR-RD's adjacent-pixel replacement idea.  It uses masked prediction,
  uncertainty estimation and an edge/data-consistency gate.  This is the only
  track that may be evaluated for metrology.
* ``visual``: post-processes an externally generated candidate.  It is always
  labelled VISUAL_ONLY because a generative prior can invent or remove layers.

This is an engineering adaptation for one grayscale 16-bit image, not a claim
to reproduce the full APR-RD or FoundIR-v2 training recipes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import find_peaks, peak_widths
from skimage import exposure
from skimage.metrics import structural_similarity
from tifffile import imread, imwrite


@dataclass(frozen=True)
class Roi:
    y0: int
    y1: int
    x0: int
    x1: int

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def clamp(self, shape: tuple[int, int]) -> "Roi":
        h, w = shape
        y0, y1 = sorted((max(0, self.y0), min(h, self.y1)))
        x0, x1 = sorted((max(0, self.x0), min(w, self.x1)))
        if y1 - y0 < 32 or x1 - x0 < 32:
            raise ValueError(f"ROI too small after clamping: {(y0, y1, x0, x1)}")
        return Roi(y0, y1, x0, x1)


def parse_roi(text: str) -> Roi:
    values = [int(v.strip()) for v in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must be y0,y1,x0,x1")
    return Roi(*values)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_gray(path: Path) -> tuple[np.ndarray, dict]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        a = np.asarray(imread(path))
    else:
        a = np.asarray(Image.open(path).convert("L"))
    if a.ndim != 2:
        raise ValueError(f"Expected a single-channel image, got shape={a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"Expected an integer image, got dtype={a.dtype}")
    info = {
        "path": str(path),
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": int(a.min()),
        "max": int(a.max()),
    }
    scale = float(np.iinfo(a.dtype).max)
    return a.astype(np.float32) / scale, {**info, "integer_scale": scale}


def to_uint16(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, 0.0, 1.0) * 65535.0).astype(np.uint16)


def save_preview(path: Path, x: np.ndarray, limits: tuple[float, float] | None = None) -> tuple[float, float]:
    if limits is None:
        nz = x[x > 0]
        sample = nz if nz.size else x.reshape(-1)
        limits = tuple(float(v) for v in np.percentile(sample, (0.5, 99.7)))
    lo, hi = limits
    y = np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    Image.fromarray(np.rint(y * 255).astype(np.uint8), mode="L").save(path)
    return limits


class AprN2SLite(nn.Module):
    """Small absolute-prediction CNN used only on pixels hidden from its input."""

    def __init__(self, channels: int = 12, depth: int = 5):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, channels, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)])
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


NEIGHBOURS = ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1))


def adjacent_replace(x: torch.Tensor, probability: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide selected centers by replacing each with a randomly selected neighbour."""
    b, _, h, w = x.shape
    mask = torch.rand((b, 1, h, w), generator=generator, device=x.device) < probability
    mask[..., :2, :] = False
    mask[..., -2:, :] = False
    mask[..., :, :2] = False
    mask[..., :, -2:] = False
    choices = torch.randint(len(NEIGHBOURS), (b, 1, h, w), generator=generator, device=x.device)
    replacement = torch.zeros_like(x)
    for k, (dy, dx) in enumerate(NEIGHBOURS):
        shifted = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        replacement = torch.where(choices == k, shifted, replacement)
    return torch.where(mask, replacement, x), mask


def random_patches(x: torch.Tensor, batch: int, patch: int, generator: torch.Generator) -> torch.Tensor:
    _, _, h, w = x.shape
    if h < patch or w < patch:
        raise ValueError(f"Training ROI {h}x{w} is smaller than patch={patch}")
    out = []
    for _ in range(batch):
        y = int(torch.randint(h - patch + 1, (1,), generator=generator, device=x.device).item())
        z = int(torch.randint(w - patch + 1, (1,), generator=generator, device=x.device).item())
        p = x[..., y : y + patch, z : z + patch]
        if bool(torch.randint(2, (1,), generator=generator, device=x.device).item()):
            p = torch.flip(p, (-1,))
        out.append(p)
    return torch.cat(out, dim=0)


def charbonnier_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    values = torch.sqrt((pred - target).square() + eps * eps)
    return values[mask].mean()


def train_measurement_model(
    x: np.ndarray,
    roi: Roi,
    iterations: int,
    patch: int,
    batch: int,
    seed: int,
    device: torch.device,
) -> tuple[AprN2SLite, list[dict]]:
    seed_all(seed)
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    generator = torch.Generator(device=device.type).manual_seed(seed)
    ys, xs = roi.slices()
    source = torch.from_numpy(np.ascontiguousarray(x[ys, xs]))[None, None].to(device)
    net = AprN2SLite().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, iterations), eta_min=3e-5)
    history: list[dict] = []
    started = time.time()
    net.train()
    for step in range(1, iterations + 1):
        clean_target = random_patches(source, batch, patch, generator)
        hidden, mask = adjacent_replace(clean_target, probability=0.12, generator=generator)
        pred = net(hidden)
        loss = charbonnier_masked(pred, clean_target, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        schedule.step()
        if step == 1 or step % 50 == 0 or step == iterations:
            item = {"iteration": step, "masked_loss": float(loss.item()), "elapsed_s": round(time.time() - started, 2)}
            history.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    return net, history


@torch.no_grad()
def masked_ensemble(
    net: AprN2SLite,
    crop: np.ndarray,
    passes: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict every pixel only when it is hidden; return mean/std/count."""
    net.eval()
    source = torch.from_numpy(np.ascontiguousarray(crop))[None, None].to(device)
    generator = torch.Generator(device=device.type).manual_seed(seed + 991)
    total = torch.zeros_like(source)
    square = torch.zeros_like(source)
    count = torch.zeros_like(source)
    for _ in range(passes):
        hidden, mask = adjacent_replace(source, probability=0.55, generator=generator)
        pred = net(hidden).clamp(0.0, 1.0)
        m = mask.to(source.dtype)
        total += pred * m
        square += pred.square() * m
        count += m
    mean = torch.where(count > 0, total / count.clamp_min(1), source)
    variance = torch.where(count > 1, square / count.clamp_min(1) - mean.square(), torch.zeros_like(mean))
    return (
        mean[0, 0].cpu().numpy().astype(np.float32),
        variance.clamp_min(0).sqrt()[0, 0].cpu().numpy().astype(np.float32),
        count[0, 0].cpu().numpy().astype(np.float32),
    )


def noise_sigma_mad(x: np.ndarray) -> float:
    high = x - gaussian_filter(x, sigma=1.2)
    return float(np.median(np.abs(high - np.median(high))) / 0.6745)


def data_consistent_blend(observed: np.ndarray, predicted: np.ndarray, strength: float) -> tuple[np.ndarray, dict]:
    smooth = gaussian_filter(observed, sigma=0.7)
    gradient = np.hypot(sobel(smooth, axis=0), sobel(smooth, axis=1))
    threshold = float(np.percentile(gradient, 78.0)) + 1e-8
    edge_gate = np.exp(-((gradient / threshold) ** 2))
    sigma = noise_sigma_mad(observed)
    cap = max(2.75 * sigma, 2.0 / 65535.0)
    delta = np.clip(predicted - observed, -cap, cap)
    out = observed + float(strength) * edge_gate * delta
    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "estimated_noise_sigma_normalized": sigma,
        "change_cap_normalized": cap,
        "edge_gate_threshold": threshold,
    }


def directional_continuity(
    observed: np.ndarray,
    current: np.ndarray,
    sigma_y: float,
    strength: float,
    change_cap: float,
) -> np.ndarray:
    """Suppress discontinuous speckle along plates without mixing adjacent plates."""
    vertical = gaussian_filter(current, sigma=(sigma_y, 0.0))
    gx = np.abs(sobel(gaussian_filter(observed, sigma=0.7), axis=1))
    threshold = float(np.percentile(gx, 76.0)) + 1e-8
    edge_gate = np.exp(-((gx / threshold) ** 2))
    candidate = current + float(strength) * edge_gate * (vertical - current)
    total_delta = np.clip(candidate - observed, -change_cap, change_cap)
    return np.clip(observed + total_delta, 0.0, 1.0).astype(np.float32)


def feather_insert(full: np.ndarray, crop: np.ndarray, roi: Roi, ramp: int = 18) -> np.ndarray:
    ys, xs = roi.slices()
    h, w = crop.shape
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])
    gate = np.minimum(yy[:, None], xx[None, :]).astype(np.float32)
    gate = np.clip(gate / max(1, ramp), 0.0, 1.0)
    out = full.copy()
    out[ys, xs] = full[ys, xs] + gate * (crop - full[ys, xs])
    return out.astype(np.float32)


def profile_metrics(img: np.ndarray, roi: Roi) -> dict:
    crop = img[roi.slices()]
    profile = gaussian_filter(crop.mean(axis=0), sigma=1.2)
    prominence = max(float(profile.std()) * 0.20, 2e-5)
    peaks, _ = find_peaks(profile, distance=12, prominence=prominence)
    peaks = peaks[(peaks > 10) & (peaks < profile.size - 10)]
    widths = peak_widths(profile, peaks, rel_height=0.5)[0] if len(peaks) else np.array([], dtype=np.float32)
    continuity = []
    dropout = []
    for p in peaks:
        lo, hi = max(0, p - 4), min(crop.shape[1], p + 5)
        line = crop[:, lo:hi].max(axis=1)
        trend = gaussian_filter(line, sigma=10.0)
        residual = line - trend
        med = max(float(np.median(line)), 1e-8)
        continuity.append(float(np.std(residual) / med))
        dropout.append(float(np.mean(line < 0.70 * np.median(trend))))
    return {
        "detected_peaks": int(len(peaks)),
        "peak_positions": [int(p + roi.x0) for p in peaks],
        "median_fwhm_px": float(np.median(widths)) if len(widths) else None,
        "median_continuity_cv": float(np.median(continuity)) if continuity else None,
        "median_dropout_fraction": float(np.median(dropout)) if dropout else None,
    }


def center_shift(before: dict, after: dict) -> dict:
    a = np.asarray(before["peak_positions"], dtype=np.float32)
    b = np.asarray(after["peak_positions"], dtype=np.float32)
    shifts = []
    for p in a:
        if b.size:
            q = float(b[np.argmin(np.abs(b - p))])
            if abs(q - p) <= 5:
                shifts.append(q - p)
    return {
        "matched": len(shifts),
        "median_abs_px": float(np.median(np.abs(shifts))) if shifts else None,
        "max_abs_px": float(np.max(np.abs(shifts))) if shifts else None,
    }


def boundary_highlight_metric(img: np.ndarray, y0: int, y1: int, center_x: int, half_width: int = 14) -> dict:
    band = img[y0:y1, max(0, center_x - half_width) : min(img.shape[1], center_x + half_width + 1)]
    line = band.max(axis=1)
    trend = gaussian_filter(line, sigma=12.0)
    baseline = max(float(np.median(trend)), 1e-8)
    return {
        "continuity_cv": float(np.std(line - trend) / baseline),
        "dropout_fraction": float(np.mean(line < 0.70 * np.median(trend))),
        "median_intensity": float(np.median(line)),
    }


def measurement_qa(before: np.ndarray, after: np.ndarray, left: Roi, right: Roi, uncertainty: np.ndarray) -> dict:
    b_left, a_left = profile_metrics(before, left), profile_metrics(after, left)
    b_right, a_right = profile_metrics(before, right), profile_metrics(after, right)
    union = Roi(min(left.y0, right.y0), max(left.y1, right.y1), left.x0, right.x1)
    ys, xs = union.slices()
    residual = after[ys, xs] - before[ys, xs]
    ssim = float(structural_similarity(before[ys, xs], after[ys, xs], data_range=1.0))
    changes = np.abs(residual)
    payload = {
        "left": {"before": b_left, "after": a_left, "shift": center_shift(b_left, a_left)},
        "right": {"before": b_right, "after": a_right, "shift": center_shift(b_right, a_right)},
        "measurement_roi_ssim": ssim,
        "residual_mean": float(residual.mean()),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "residual_abs_p99": float(np.percentile(changes, 99.0)),
        "uncertainty_median": float(np.median(uncertainty)),
        "uncertainty_p99": float(np.percentile(uncertainty, 99.0)),
    }
    y0, y1 = max(left.y0, right.y0), min(left.y1, right.y1)
    payload["central_highlight_boundaries"] = {}
    for name, center_x in (("left_edge", left.x1), ("right_edge", right.x0)):
        bb = boundary_highlight_metric(before, y0, y1, center_x)
        aa = boundary_highlight_metric(after, y0, y1, center_x)
        payload["central_highlight_boundaries"][name] = {"before": bb, "after": aa}
    count_ok = all(abs(payload[side]["after"]["detected_peaks"] - payload[side]["before"]["detected_peaks"]) <= 1 for side in ("left", "right"))
    shift_ok = all((payload[side]["shift"]["max_abs_px"] or 0.0) <= 1.0 for side in ("left", "right"))
    width_change = {}
    for side in ("left", "right"):
        wb = payload[side]["before"]["median_fwhm_px"]
        wa = payload[side]["after"]["median_fwhm_px"]
        width_change[side] = abs(float(wa) / float(wb) - 1.0) if wb and wa else math.inf
    payload["fwhm_relative_change"] = width_change
    width_ok = all(v <= 0.05 for v in width_change.values())
    highlight_ok = all(
        item["after"]["dropout_fraction"] <= item["before"]["dropout_fraction"] + 0.005
        for item in payload["central_highlight_boundaries"].values()
    )
    payload["guardrail_pass"] = bool(count_ok and shift_ok and width_ok and highlight_ok and ssim >= 0.970)
    payload["guardrails"] = {
        "layer_count_delta_max": 1,
        "peak_shift_max_px": 1.0,
        "fwhm_relative_change_max": 0.05,
        "central_highlight_dropout_allowed_increase": 0.005,
        "ssim_min": 0.970,
    }
    return payload


def postprocess_visual(candidate_path: Path, outdir: Path) -> np.ndarray:
    candidate, _ = load_gray(candidate_path)
    lo, hi = (float(v) for v in np.percentile(candidate, (0.4, 99.6)))
    base = np.clip((candidate - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    local = exposure.equalize_adapthist(base, kernel_size=64, clip_limit=0.008)
    mixed = 0.72 * base + 0.28 * local
    smooth = gaussian_filter(mixed, sigma=0.65)
    detail = mixed - gaussian_filter(mixed, sigma=1.35)
    grad = np.hypot(sobel(smooth, axis=0), sobel(smooth, axis=1))
    threshold = float(np.percentile(grad, 82.0)) + 1e-8
    gate = 1.0 - np.exp(-((grad / threshold) ** 2))
    final = np.clip(smooth + 0.16 * gate * detail, 0.0, 1.0).astype(np.float32)
    Image.fromarray(np.rint(final * 255).astype(np.uint8), mode="L").save(outdir / "GENERATIVE_visual_only_postprocessed.png")
    imwrite(
        outdir / "GENERATIVE_visual_only_postprocessed_16bit.tif",
        to_uint16(final),
        photometric="minisblack",
        description="VISUAL_ONLY: generative content; prohibited for thickness measurement or defect acceptance.",
    )
    return final


def panel(x: np.ndarray, size: tuple[int, int], title: str) -> Image.Image:
    nz = x[x > 0]
    sample = nz if nz.size else x.reshape(-1)
    lo, hi = (float(v) for v in np.percentile(sample, (0.5, 99.5)))
    u8 = np.rint(np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size[0], size[1] + 28), 0)
    canvas.paste(im, ((size[0] - im.width) // 2, 28 + (size[1] - im.height) // 2))
    ImageDraw.Draw(canvas).text((8, 8), title, fill=255, font=ImageFont.load_default())
    return canvas


def comparison(path: Path, observed: np.ndarray, measured: np.ndarray, generated: np.ndarray | None) -> None:
    items = [panel(observed, (700, 510), "Observed 16-bit"), panel(measured, (700, 510), "Measurement-preserving blind-denoise candidate")]
    if generated is not None:
        items.append(panel(generated, (700, 510), "GENERATIVE / VISUAL ONLY"))
    canvas = Image.new("L", (700, 538 * len(items)), 0)
    for i, item in enumerate(items):
        canvas.paste(item, (0, i * 538))
    canvas.save(path)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="16-bit grayscale source TIFF")
    ap.add_argument("--generated", type=Path, help="external generative candidate; visual track only")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--train-roi", type=parse_roi, default=Roi(620, 1210, 320, 1880))
    ap.add_argument("--left-roi", type=parse_roi, default=Roi(700, 1110, 370, 970))
    ap.add_argument("--right-roi", type=parse_roi, default=Roi(700, 1110, 1220, 1830))
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.20)
    ap.add_argument("--directional-strength", type=float, default=0.70)
    ap.add_argument("--directional-sigma", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    observed, source_info = load_gray(args.input)
    train_roi = args.train_roi.clamp(observed.shape)
    left_roi = args.left_roi.clamp(observed.shape)
    right_roi = args.right_roi.clamp(observed.shape)
    device = choose_device(args.device)
    started = time.time()

    net, history = train_measurement_model(observed, train_roi, args.iterations, args.patch, args.batch, args.seed, device)
    ys, xs = train_roi.slices()
    predicted, uncertainty_crop, count = masked_ensemble(net, observed[ys, xs], args.passes, args.seed, device)
    conservative, blend_info = data_consistent_blend(observed[ys, xs], predicted, args.strength)
    conservative = directional_continuity(
        observed[ys, xs],
        conservative,
        sigma_y=args.directional_sigma,
        strength=args.directional_strength,
        change_cap=float(blend_info["change_cap_normalized"]),
    )
    measured = feather_insert(observed, conservative, train_roi)
    uncertainty = np.zeros_like(observed, dtype=np.float32)
    uncertainty[ys, xs] = uncertainty_crop
    residual = measured - observed

    imwrite(args.outdir / "MEASUREMENT_blind_denoised_16bit.tif", to_uint16(measured), photometric="minisblack")
    imwrite(args.outdir / "MEASUREMENT_residual_float32.tif", residual.astype(np.float32), photometric="minisblack")
    imwrite(args.outdir / "MEASUREMENT_uncertainty_float32.tif", uncertainty.astype(np.float32), photometric="minisblack")
    save_preview(args.outdir / "MEASUREMENT_blind_denoised_preview.png", measured)
    lim = float(np.percentile(np.abs(residual[ys, xs]), 99.5)) + 1e-8
    save_preview(args.outdir / "MEASUREMENT_residual_preview.png", residual, (-lim, lim))
    save_preview(args.outdir / "MEASUREMENT_uncertainty_preview.png", uncertainty, (0.0, float(np.percentile(uncertainty_crop, 99.5)) + 1e-8))
    torch.save({"state_dict": net.state_dict(), "architecture": "AprN2SLite", "channels": 12, "depth": 5}, args.outdir / "measurement_model.pt")

    generated = postprocess_visual(args.generated, args.outdir) if args.generated else None
    qa = measurement_qa(observed, measured, left_roi, right_roi, uncertainty_crop)
    manifest = {
        "status": "measurement output passed guardrails" if qa["guardrail_pass"] else "measurement output requires review",
        "source": source_info,
        "device": str(device),
        "training_seconds": round(time.time() - started, 2),
        "parameters": {
            "train_roi": train_roi.__dict__,
            "left_roi": left_roi.__dict__,
            "right_roi": right_roi.__dict__,
            "iterations": args.iterations,
            "patch": args.patch,
            "batch": args.batch,
            "passes": args.passes,
            "strength": args.strength,
            "directional_strength": args.directional_strength,
            "directional_sigma": args.directional_sigma,
            "seed": args.seed,
        },
        "method": {
            "measurement": "APR-inspired adjacent replacement + Noise2Self masked prediction + uncertainty ensemble + edge/data consistency",
            "visual": "external generative candidate + deterministic local-contrast post-processing" if args.generated else None,
            "claim_boundary": "Engineering adaptation; not an exact reproduction of APR-RD, Blind2Sound, or FoundIR-v2.",
        },
        "blend": blend_info,
        "mask_count": {"min": float(count.min()), "mean": float(count.mean()), "max": float(count.max())},
        "training_history": history,
        "qa": qa,
        "prohibitions": [
            "Do not use GENERATIVE_* files for thickness measurement, defect acceptance, or ground truth.",
            "Absolute thickness still requires pixel-size and system-PSF calibration.",
        ],
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison(args.outdir / "comparison.png", observed, measured, generated)
    print(json.dumps({"completed": True, "outdir": str(args.outdir), "guardrail_pass": qa["guardrail_pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
