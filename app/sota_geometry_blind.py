#!/usr/bin/env python3
"""Generative-prior, Blind2Sound-style denoising with geometry constraints.

This is a clean-room CT adaptation of the adaptive re-visible formulation in
Blind2Sound (ICCV 2025), not a copy of the authors' dataset-oriented code.  A
generative image supplies only a clipped low-frequency prior in weak-gradient
regions.  Raw 16-bit evidence supplies all edge, width, endpoint and data-
consistency constraints.  No generated pixel is copied into the measurement
result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, sobel
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity
from tifffile import imwrite

import length_optimize as length
import pipeline as base
import quality_optimize as quality


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveVisibleUNet(nn.Module):
    """Compact single-channel U-Net predicting signal mean and variance."""

    def __init__(self, features: int = 24):
        super().__init__()
        self.enc1 = ConvBlock(1, features)
        self.enc2 = ConvBlock(features, features * 2)
        self.enc3 = ConvBlock(features * 2, features * 4)
        self.bottleneck = ConvBlock(features * 4, features * 4)
        self.dec3 = ConvBlock(features * 8, features * 2)
        self.dec2 = ConvBlock(features * 4, features)
        self.dec1 = ConvBlock(features * 2, features)
        self.head = nn.Conv2d(features, 2, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias[0] = 0.0
            self.head.bias[1] = -5.0

    @staticmethod
    def up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        z = self.bottleneck(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat((self.up(z, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up(d2, e1), e1), dim=1))
        out = self.head(d1)
        mean = torch.clamp(x + 0.12 * torch.tanh(out[:, :1]), 0.0, 1.0)
        variance = F.softplus(out[:, 1:2]) + 1e-6
        return mean, variance


def inverse_softplus(value: float) -> float:
    return float(math.log(math.expm1(max(value, 1e-8))))


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def load_generative_prior(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((shape[1], shape[0]), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def safe_low_frequency_prior(
    source: np.ndarray,
    generated: np.ndarray,
    target: base.Roi,
) -> tuple[np.ndarray, dict]:
    """Distill appearance without accepting generated geometry or pixels."""
    ys, xs = target.slices()
    raw = source[ys, xs]
    gen = generated[ys, xs]
    raw_lo, raw_hi = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
    gen_lo, gen_hi = (float(v) for v in np.percentile(gen, (1.0, 99.0)))
    matched = (gen - gen_lo) / max(gen_hi - gen_lo, 1e-8)
    matched = np.clip(matched, 0.0, 1.0) * (raw_hi - raw_lo) + raw_lo
    raw_low = gaussian_filter(raw, sigma=6.0)
    generated_low = gaussian_filter(matched, sigma=6.0)
    gradient = np.hypot(sobel(gaussian_filter(raw, 0.8), axis=0), sobel(gaussian_filter(raw, 0.8), axis=1))
    threshold = float(np.percentile(gradient, 45.0)) + 1e-8
    low_structure = np.exp(-np.square(gradient / threshold)).astype(np.float32)
    sigma_n = base.noise_sigma_mad(raw)
    cap = max(1.25 * sigma_n, 4.0 / 65535.0)
    delta = low_structure * np.clip(generated_low - raw_low, -cap, cap)
    safe_crop = np.clip(raw + delta, 0.0, 1.0).astype(np.float32)
    safe = base.feather_insert(source, safe_crop, target, ramp=24)
    return safe, {
        "method": "clipped low-frequency generative-prior distillation in raw low-gradient regions",
        "generated_pixel_writeback": False,
        "prior_sigma_px": 6.0,
        "gradient_percentile": 45.0,
        "gradient_threshold": threshold,
        "change_cap_normalized": cap,
        "ssim_vs_source_target_roi": float(structural_similarity(raw, safe_crop, data_range=1.0)),
    }


def random_paired_patches(
    source: torch.Tensor,
    prior: torch.Tensor,
    batch: int,
    patch: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, h, w = source.shape
    if patch % 8:
        raise ValueError("patch size must be divisible by 8")
    if min(h, w) < patch:
        raise ValueError(f"training crop {h}x{w} is smaller than patch={patch}")
    raw_patches = []
    prior_patches = []
    for _ in range(batch):
        y = int(torch.randint(h - patch + 1, (1,), generator=generator, device=source.device).item())
        x = int(torch.randint(w - patch + 1, (1,), generator=generator, device=source.device).item())
        raw_patch = source[..., y : y + patch, x : x + patch]
        prior_patch = prior[..., y : y + patch, x : x + patch]
        if bool(torch.randint(2, (1,), generator=generator, device=source.device).item()):
            raw_patch = torch.flip(raw_patch, (-1,))
            prior_patch = torch.flip(prior_patch, (-1,))
        raw_patches.append(raw_patch)
        prior_patches.append(prior_patch)
    return torch.cat(raw_patches), torch.cat(prior_patches)


def sublattice_blind_mask(
    image: torch.Tensor,
    generator: torch.Generator,
    width: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask one random sub-lattice phase and interpolate only from neighbours."""
    mask = torch.zeros_like(image, dtype=torch.bool)
    for i in range(image.shape[0]):
        phase = int(torch.randint(width * width, (1,), generator=generator, device=image.device).item())
        oy, ox = divmod(phase, width)
        mask[i : i + 1, :, oy::width, ox::width] = True
    mask[..., 0, :] = False
    mask[..., -1, :] = False
    mask[..., :, 0] = False
    mask[..., :, -1] = False
    interpolated = 0.25 * (
        torch.roll(image, 1, -2)
        + torch.roll(image, -1, -2)
        + torch.roll(image, 1, -1)
        + torch.roll(image, -1, -1)
    )
    return torch.where(mask, interpolated, image), mask


def fixed_sublattice_blind_mask(
    image: torch.Tensor,
    phase: int,
    width: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(image, dtype=torch.bool)
    oy, ox = divmod(int(phase), width)
    mask[..., oy::width, ox::width] = True
    mask[..., 0, :] = False
    mask[..., -1, :] = False
    mask[..., :, 0] = False
    mask[..., :, -1] = False
    interpolated = 0.25 * (
        torch.roll(image, 1, -2)
        + torch.roll(image, -1, -2)
        + torch.roll(image, 1, -1)
        + torch.roll(image, -1, -1)
    )
    return torch.where(mask, interpolated, image), mask


def smooth_tensor(x: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2)


def gradient_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def masked_charbonnier(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = torch.sqrt((pred - target).square() + 1e-6)
    return values[mask].mean()


def geometry_consistency_loss(pred: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Preserve transverse widths and longitudinal endpoints during training."""
    src_smooth = smooth_tensor(source, 3)
    pred_smooth = smooth_tensor(pred, 3)
    src_gx, src_gy = gradient_xy(src_smooth)
    pred_gx, pred_gy = gradient_xy(pred_smooth)
    tx = torch.quantile(src_gx.detach().abs().flatten(1), 0.72, dim=1).view(-1, 1, 1, 1)
    ty = torch.quantile(src_gy.detach().abs().flatten(1), 0.78, dim=1).view(-1, 1, 1, 1)
    wx = torch.sigmoid((src_gx.detach().abs() - tx) / (0.20 * tx + 1e-6))
    wy = torch.sigmoid((src_gy.detach().abs() - ty) / (0.20 * ty + 1e-6))
    edge_x = (wx * (pred_gx - src_gx).abs()).sum() / (wx.sum() + 1e-6)
    edge_y = (wy * (pred_gy - src_gy).abs()).sum() / (wy.sum() + 1e-6)
    source_width = source.mean(dim=-2)
    pred_width = pred.mean(dim=-2)
    width_loss = (gradient_xy(source_width.unsqueeze(-2))[0] - gradient_xy(pred_width.unsqueeze(-2))[0]).abs().mean()
    source_length = source.mean(dim=-1)
    pred_length = pred.mean(dim=-1)
    endpoint_loss = (gradient_xy(source_length.unsqueeze(-1))[1] - gradient_xy(pred_length.unsqueeze(-1))[1]).abs().mean()
    shape_loss = (smooth_tensor(pred, 5) - smooth_tensor(source, 5)).abs().mean()
    total = 0.34 * edge_x + 0.26 * edge_y + 0.20 * width_loss + 0.12 * endpoint_loss + 0.08 * shape_loss
    return total, {
        "edge_x": float(edge_x.detach()),
        "edge_y": float(edge_y.detach()),
        "width_profile": float(width_loss.detach()),
        "endpoint_profile": float(endpoint_loss.detach()),
        "multiscale_shape": float(shape_loss.detach()),
    }


def generative_prior_loss(pred: torch.Tensor, prior: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    src_smooth = smooth_tensor(source, 3)
    gx, gy = gradient_xy(src_smooth)
    gx = F.pad(gx.abs(), (0, 1, 0, 0))
    gy = F.pad(gy.abs(), (0, 0, 0, 1))
    gradient = torch.sqrt(gx.square() + gy.square() + 1e-12)
    threshold = torch.quantile(gradient.detach().flatten(1), 0.45, dim=1).view(-1, 1, 1, 1)
    gate = torch.exp(-torch.square(gradient / (threshold + 1e-6))).detach()
    return (gate * (smooth_tensor(pred, 7) - smooth_tensor(prior, 7)).abs()).sum() / (gate.sum() + 1e-6)


def train_model(
    source_image: np.ndarray,
    safe_prior: np.ndarray,
    target: base.Roi,
    iterations: int,
    patch: int,
    batch: int,
    features: int,
    geometry_weight: float,
    prior_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[AdaptiveVisibleUNet, dict, list[dict]]:
    base.seed_all(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    generator = torch.Generator(device=device.type).manual_seed(seed)
    ys, xs = target.slices()
    source = torch.from_numpy(np.ascontiguousarray(source_image[ys, xs]))[None, None].to(device)
    prior = torch.from_numpy(np.ascontiguousarray(safe_prior[ys, xs]))[None, None].to(device)
    sigma_n = base.noise_sigma_mad(source_image[ys, xs])
    model = AdaptiveVisibleUNet(features=features).to(device)
    log_gaussian = nn.Parameter(torch.tensor(inverse_softplus(sigma_n), device=device))
    log_poisson = nn.Parameter(torch.tensor(inverse_softplus(max(sigma_n * 0.10, 1e-5)), device=device))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [log_gaussian, log_poisson],
        lr=4e-4,
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(iterations, 1), eta_min=2e-5)
    history = []
    started = time.time()
    model.train()
    for step in range(1, iterations + 1):
        noisy, prior_patch = random_paired_patches(source, prior, batch, patch, generator)
        hidden, mask = sublattice_blind_mask(noisy, generator)
        masked_mean, masked_var = model(hidden)
        visible_mean, visible_var = model(noisy)
        progress = step / max(iterations, 1)
        beta = 3.0 if progress <= 0.40 else 3.0 + (progress - 0.40) / 0.60 * 8.0
        beta = min(beta, 11.0)
        medium_mean = (masked_mean + beta * visible_mean.detach()) / (1.0 + beta)
        medium_var = (masked_var + beta * beta * visible_var.detach()) / ((1.0 + beta) ** 2)
        gaussian_sigma = F.softplus(log_gaussian) + 1e-6
        poisson_scale = F.softplus(log_poisson) + 1e-7
        noise_var = gaussian_sigma.square() + poisson_scale * medium_mean.detach().clamp_min(1e-4)
        total_var = medium_var + noise_var + 1e-6
        likelihood_map = (noisy - medium_mean).square() / total_var + torch.log(total_var)
        likelihood = likelihood_map[mask].mean()
        blind_loss = masked_charbonnier(masked_mean, noisy, mask)
        # Geometry and prior losses are evaluated on the assembled blind
        # prediction, so only pixels hidden from the network are modified.
        blind_geometry_prediction = torch.where(mask, masked_mean, noisy)
        geometry_loss, geometry_parts = geometry_consistency_loss(blind_geometry_prediction, noisy)
        prior_loss = generative_prior_loss(blind_geometry_prediction, prior_patch, noisy)
        loss = likelihood + 0.35 * blind_loss + geometry_weight * geometry_loss + prior_weight * prior_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()) + [log_gaussian, log_poisson], 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0 or step == iterations:
            item = {
                "iteration": step,
                "loss": float(loss.detach()),
                "adaptive_revisible_nll": float(likelihood.detach()),
                "blind_masked_loss": float(blind_loss.detach()),
                "geometry_loss": float(geometry_loss.detach()),
                "generative_prior_loss": float(prior_loss.detach()),
                "visible_beta": float(beta),
                "gaussian_sigma": float(gaussian_sigma.detach()),
                "poisson_scale": float(poisson_scale.detach()),
                "geometry_parts": geometry_parts,
                "elapsed_s": round(time.time() - started, 2),
            }
            history.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    noise = {
        "gaussian_sigma_normalized": float((F.softplus(log_gaussian) + 1e-6).detach()),
        "poisson_scale_normalized": float((F.softplus(log_poisson) + 1e-7).detach()),
    }
    return model, noise, history


@torch.no_grad()
def tiled_predict(
    model: AdaptiveVisibleUNet,
    crop: np.ndarray,
    tile: int,
    overlap: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    h, w = crop.shape
    tile = min(tile, h, w)
    stride = max(16, tile - overlap)
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    axis = np.hanning(tile).astype(np.float32)
    weight = np.maximum(np.outer(axis, axis), 0.05).astype(np.float32)
    mean_sum = np.zeros_like(crop, dtype=np.float32)
    var_sum = np.zeros_like(crop, dtype=np.float32)
    weights = np.zeros_like(crop, dtype=np.float32)
    phase_count = 16
    phase_batch = 4
    for y in ys:
        for x in xs:
            patch = torch.from_numpy(np.ascontiguousarray(crop[y : y + tile, x : x + tile]))[None, None].to(device)
            blind_mean = torch.zeros_like(patch)
            blind_var = torch.zeros_like(patch)
            blind_count = torch.zeros_like(patch)
            for start in range(0, phase_count, phase_batch):
                hidden_batch = []
                mask_batch = []
                for phase in range(start, min(start + phase_batch, phase_count)):
                    hidden, phase_mask = fixed_sublattice_blind_mask(patch, phase)
                    hidden_batch.append(hidden)
                    mask_batch.append(phase_mask)
                hidden_tensor = torch.cat(hidden_batch, dim=0)
                phase_masks = torch.cat(mask_batch, dim=0)
                phase_means, phase_vars = model(hidden_tensor)
                for index in range(phase_means.shape[0]):
                    current_mask = phase_masks[index : index + 1]
                    blind_mean += torch.where(current_mask, phase_means[index : index + 1], torch.zeros_like(patch))
                    blind_var += torch.where(current_mask, phase_vars[index : index + 1], torch.zeros_like(patch))
                    blind_count += current_mask.to(patch.dtype)
            direct_mean, direct_var = model(patch)
            blind_mean = torch.where(blind_count > 0, blind_mean / blind_count.clamp_min(1.0), direct_mean)
            blind_var = torch.where(blind_count > 0, blind_var / blind_count.clamp_min(1.0), direct_var)
            mean_np = blind_mean[0, 0].cpu().numpy()
            var_np = blind_var[0, 0].cpu().numpy()
            mean_sum[y : y + tile, x : x + tile] += weight * mean_np
            var_sum[y : y + tile, x : x + tile] += weight * var_np
            weights[y : y + tile, x : x + tile] += weight
    return mean_sum / np.maximum(weights, 1e-8), var_sum / np.maximum(weights, 1e-8)


def data_geometry_projection(
    observed: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_var: np.ndarray,
    noise: dict,
    strength: float,
    transverse_anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    gaussian_var = noise["gaussian_sigma_normalized"] ** 2
    noise_var = gaussian_var + noise["poisson_scale_normalized"] * np.maximum(predicted_mean, 1e-4)
    posterior_mean = (observed * predicted_var + predicted_mean * noise_var) / np.maximum(predicted_var + noise_var, 1e-8)
    # Retain part of the visible prediction so conservative posterior shrinkage
    # cannot collapse the new denoiser back to the observation.
    posterior = 0.65 * posterior_mean + 0.35 * predicted_mean
    gradient = np.hypot(sobel(gaussian_filter(observed, 0.7), axis=0), sobel(gaussian_filter(observed, 0.7), axis=1))
    threshold = float(np.percentile(gradient, 70.0)) + 1e-8
    gate = np.exp(-np.square(gradient / threshold)).astype(np.float32)
    if transverse_anchor is not None:
        if transverse_anchor.shape != observed.shape:
            raise ValueError("transverse geometry anchor must match the projection crop")
        gate *= 1.0 - np.clip(transverse_anchor, 0.0, 1.0)
    sigma_n = base.noise_sigma_mad(observed)
    cap = max(2.4 * sigma_n, 4.0 / 65535.0)
    delta = gate * np.clip(posterior - observed, -cap, cap)
    out = np.clip(observed + strength * delta, 0.0, 1.0).astype(np.float32)
    return out, {
        "method": "Poisson-Gaussian posterior mean + raw strong-edge projection",
        "raw_pixel_writeback": False,
        "strength": strength,
        "gradient_percentile": 70.0,
        "gradient_threshold": threshold,
        "change_cap_normalized": cap,
        "transverse_geometry_anchor": transverse_anchor is not None,
    }


def geometry_locked_directional_continuity(
    source: np.ndarray,
    current: np.ndarray,
    target: base.Roi,
    references: list[dict],
    strength: float,
    sigma_y: float,
    change_cap: float,
) -> tuple[np.ndarray, dict]:
    """Apply the original axial smoother without crossing measured geometry.

    The public v1 pipeline protected transverse gradients only. This version
    also protects raw longitudinal gradients and fixed subpixel endpoint
    supports before smoothing is allowed into a projection candidate.
    """
    ys, xs = target.slices()
    observed = source[ys, xs]
    crop = current[ys, xs]
    raw_smooth = gaussian_filter(observed, sigma=0.70)
    gx = np.abs(sobel(raw_smooth, axis=1))
    gy = np.abs(sobel(raw_smooth, axis=0))
    tx = float(np.percentile(gx, 76.0)) + 1e-8
    ty = float(np.percentile(gy, 70.0)) + 1e-8
    transverse_safe = np.exp(-np.square(gx / tx)).astype(np.float32)
    longitudinal_safe = np.exp(-np.square(gy / ty)).astype(np.float32)
    endpoint_protection = quality.endpoint_coordinate_protection(
        source.shape, references, radius_y=7, radius_x=4
    )[ys, xs]
    geometry_gate = transverse_safe * longitudinal_safe * (1.0 - endpoint_protection)
    axial = gaussian_filter(crop, sigma=(float(sigma_y), 0.0))
    candidate = crop + float(strength) * geometry_gate * (axial - crop)
    # Never move farther outside the selected raw-data projection bound, while
    # also avoiding a forced change to an already accepted baseline pixel.
    lower = np.minimum(observed - change_cap, crop)
    upper = np.maximum(observed + change_cap, crop)
    candidate = np.clip(candidate, lower, upper)
    candidate = np.where(endpoint_protection >= 0.999, crop, candidate)
    result = base.feather_insert(
        current, np.clip(candidate, 0.0, 1.0).astype(np.float32), target, ramp=18
    )
    return result, {
        "method": "v1 axial continuity with raw transverse, longitudinal, and endpoint geometry locks",
        "strength": float(strength),
        "sigma_y_px": float(sigma_y),
        "transverse_gradient_percentile": 76.0,
        "longitudinal_gradient_percentile": 70.0,
        "transverse_gradient_threshold": tx,
        "longitudinal_gradient_threshold": ty,
        "endpoint_protection_radius_px": [7, 4],
        "change_cap_normalized": float(change_cap),
        "raw_pixel_writeback": False,
        "generated_pixel_writeback": False,
    }


def transverse_geometry_anchor(
    source: np.ndarray,
    target: base.Roi,
    measurement_rois: tuple[base.Roi, ...],
    radius: int = 0,
) -> tuple[np.ndarray, dict]:
    """Anchor raw layer center coordinates while leaving other pixels denoisable.

    A one-pixel center line is enough to prevent weak ridges from being
    re-associated with a neighbouring maximum.  The constraint is applied in
    the data projection, not by copying generated or post-hoc raw patches.
    """
    anchor = np.zeros((target.y1 - target.y0, target.x1 - target.x0), dtype=np.float32)
    positions: dict[str, list[int]] = {}
    for index, roi in enumerate(measurement_rois):
        clipped = roi.clamp(source.shape)
        crop = source[clipped.slices()]
        profile = gaussian_filter1d(crop.mean(axis=0), sigma=1.2)
        prominence = max(float(profile.std()) * 0.20, 2e-5)
        peaks, _ = find_peaks(profile, distance=12, prominence=prominence)
        peaks = peaks[(peaks > 10) & (peaks < profile.size - 10)]
        global_positions = [int(p + clipped.x0) for p in peaks]
        positions[f"roi_{index}"] = global_positions
        for global_x in global_positions:
            local_x = global_x - target.x0
            x0 = max(0, local_x - int(radius))
            x1 = min(anchor.shape[1], local_x + int(radius) + 1)
            if x1 > x0:
                anchor[:, x0:x1] = 1.0
    return anchor, {
        "method": "raw transverse ridge-center coordinate anchoring during data projection",
        "radius_px": int(radius),
        "anchored_layer_count": int(sum(len(values) for values in positions.values())),
        "positions_x_px": positions,
        "generated_pixel_writeback": False,
        "post_hoc_raw_patch_writeback": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--generative-prior", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--target-roi", type=base.parse_roi, default=base.Roi(620, 1210, 320, 1880))
    ap.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(700, 1110, 370, 970))
    ap.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(700, 1110, 1220, 1830))
    ap.add_argument("--left-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    ap.add_argument("--right-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    ap.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    ap.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--features", type=int, default=24)
    ap.add_argument("--geometry-weight", type=float, default=7.5)
    ap.add_argument("--prior-weight", type=float, default=0.04)
    ap.add_argument("--projection-strength", type=float, default=1.50)
    ap.add_argument("--transverse-anchor-radius", type=int, default=0)
    ap.add_argument(
        "--directional-strength",
        type=float,
        default=0.70,
        help="maximum strength searched by geometry-locked v1 axial continuity; 0 disables it",
    )
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=48)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    target = args.target_roi.clamp(source.shape)
    left = args.left_roi.clamp(source.shape)
    right = args.right_roi.clamp(source.shape)
    generated = load_generative_prior(args.generative_prior, source.shape)
    safe_prior, prior_info = safe_low_frequency_prior(source, generated, target)
    device = choose_device(args.device)
    started = time.time()
    model, noise, history = train_model(
        source,
        safe_prior,
        target,
        args.iterations,
        args.patch,
        args.batch,
        args.features,
        args.geometry_weight,
        args.prior_weight,
        args.seed,
        device,
    )
    ys, xs = target.slices()
    mean_crop, var_crop = tiled_predict(model, source[ys, xs], args.tile, args.overlap, device)
    uncertainty_crop = np.sqrt(np.maximum(var_crop, 0.0)).astype(np.float32)
    blind_prediction = base.feather_insert(source, mean_crop, target, ramp=18)
    transverse_anchor, anchor_info = transverse_geometry_anchor(
        source, target, (left, right), radius=args.transverse_anchor_radius
    )
    raw_regions = [quality.analyze_region(source, roi) for roi in (left, right)]
    raw_roughness = float(np.median([region["flat_region_roughness"] for region in raw_regions]))
    projection_grid = []
    selected = None
    strengths = sorted(set((0.0, 0.65, 0.80, 1.00, 1.20, 1.50, float(args.projection_strength))))
    for projection_strength in strengths:
        projected_crop, current_projection = data_geometry_projection(
            source[ys, xs], mean_crop, var_crop, noise, projection_strength, transverse_anchor
        )
        preliminary = base.feather_insert(source, projected_crop, target, ramp=18)
        current_references = quality.build_boundary_references(
            source,
            preliminary,
            args.left_body_roi.clamp(source.shape),
            args.right_body_roi.clamp(source.shape),
            args.top_range,
            args.bottom_range,
        )
        preliminary_boundary, preliminary_rows = quality.boundary_geometry(
            preliminary, current_references, args.top_range, args.bottom_range
        )
        preliminary_qa = base.measurement_qa(source, preliminary, left, right, uncertainty_crop)
        if quality.boundary_guardrail(preliminary_boundary) and preliminary_qa["guardrail_pass"]:
            current_measured = preliminary
            current_geometry = {
                "method": "endpoint warp skipped because the training/data-projection geometry already passed",
                "applied": False,
                "raw_pixel_writeback": False,
            }
            current_boundary, current_rows = preliminary_boundary, preliminary_rows
            current_qa = preliminary_qa
        else:
            current_measured, current_geometry = quality.endpoint_geometry_warp(
                preliminary,
                current_references,
                args.top_range,
                args.bottom_range,
                iterations=1,
                radius_y=1,
                radius_x=1,
                max_shift=0.35,
                large_shift_trigger=3.0,
                large_shift_cap=1.0,
            )
            current_boundary, current_rows = quality.boundary_geometry(
                current_measured, current_references, args.top_range, args.bottom_range
            )
            current_qa = base.measurement_qa(source, current_measured, left, right, uncertainty_crop)
        current_boundary_pass = quality.boundary_guardrail(current_boundary)
        regions = [quality.analyze_region(current_measured, roi) for roi in (left, right)]
        roughness = float(np.median([region["flat_region_roughness"] for region in regions]))
        noise_reduction = 1.0 - roughness / max(raw_roughness, 1e-8)
        item = {
            "projection_strength": projection_strength,
            "noise_roughness_reduction_vs_raw": noise_reduction,
            "measurement_guardrail_pass": current_qa["guardrail_pass"],
            "boundary_guardrail_pass": current_boundary_pass,
            "fwhm_relative_change": current_qa["fwhm_relative_change"],
            "measurement_roi_ssim": current_qa["measurement_roi_ssim"],
            "boundary_geometry": current_boundary,
        }
        projection_grid.append(item)
        if current_qa["guardrail_pass"] and current_boundary_pass and (
            selected is None or noise_reduction > selected["summary"]["noise_roughness_reduction_vs_raw"]
        ):
            selected = {
                "summary": item,
                "measured": current_measured,
                "projection": current_projection,
                "geometry": current_geometry,
                "boundary": current_boundary,
                "rows": current_rows,
                "references": current_references,
                "qa": current_qa,
            }
    if selected is None:
        raise RuntimeError("No SOTA blind candidate passed width, shape, endpoint, and length guardrails")
    projection_measured = selected["measured"]
    projection_info = selected["projection"]
    geometry_info = selected["geometry"]
    fixed_references = selected["references"]

    # Reintroduce the public repository's strong axial continuity at the point
    # where it is most useful: immediately after data projection and before any
    # enhancement. Every strength is tested against the same fixed raw geometry.
    directional_grid = []
    directional_selected = None
    maximum_directional = max(0.0, float(args.directional_strength))
    directional_strengths = {0.0, maximum_directional}
    directional_strengths.update(
        value for value in (0.25, 0.40, 0.55, 0.70) if value <= maximum_directional
    )
    projection_metrics = {
        "left": quality.analyze_region(projection_measured, left),
        "right": quality.analyze_region(projection_measured, right),
    }
    source_metrics = {
        "left": quality.analyze_region(source, left),
        "right": quality.analyze_region(source, right),
    }
    for directional_strength in sorted(directional_strengths):
        candidate, directional_operation = geometry_locked_directional_continuity(
            source,
            projection_measured,
            target,
            fixed_references,
            strength=directional_strength,
            sigma_y=3.0,
            change_cap=projection_info["change_cap_normalized"],
        )
        candidate_appearance = quality.evaluate(
            candidate, projection_measured, left, right, projection_metrics, source_metrics
        )
        candidate_boundary, candidate_rows = quality.boundary_geometry(
            candidate, fixed_references, args.top_range, args.bottom_range
        )
        candidate_qa = base.measurement_qa(source, candidate, left, right, uncertainty_crop)
        candidate_ssim = float(structural_similarity(
            projection_measured[target.slices()], candidate[target.slices()], data_range=1.0
        ))
        candidate_regions = [quality.analyze_region(candidate, roi) for roi in (left, right)]
        candidate_roughness = float(np.median([
            region["flat_region_roughness"] for region in candidate_regions
        ]))
        candidate_noise_reduction = 1.0 - candidate_roughness / max(raw_roughness, 1e-8)
        candidate_pass = bool(
            candidate_qa["guardrail_pass"]
            and quality.boundary_guardrail(candidate_boundary)
            and candidate_appearance["guardrail_pass"]
            and candidate_appearance["edge_gain"] >= 0.995
            and candidate_appearance["contrast_gain"] >= 0.995
            and candidate_appearance["max_fwhm_relative_change_vs_source"] <= 0.04
            and candidate_ssim >= 0.995
        )
        item = {
            **directional_operation,
            "noise_roughness_reduction_vs_raw": candidate_noise_reduction,
            "additional_roughness_reduction_vs_projection": 1.0
            - candidate_roughness / max(float(np.median([
                projection_metrics[side]["flat_region_roughness"] for side in ("left", "right")
            ])), 1e-8),
            "edge_acutance_retention": candidate_appearance["edge_gain"],
            "local_contrast_retention": candidate_appearance["contrast_gain"],
            "max_fwhm_relative_change_vs_source": candidate_appearance["max_fwhm_relative_change_vs_source"],
            "ssim_vs_projection_input": candidate_ssim,
            "boundary_geometry": candidate_boundary,
            "guardrail_pass": candidate_pass,
        }
        directional_grid.append(item)
        if candidate_pass and (
            directional_selected is None
            or candidate_noise_reduction
            > directional_selected["summary"]["noise_roughness_reduction_vs_raw"]
        ):
            directional_selected = {
                "summary": item,
                "measured": candidate,
                "boundary": candidate_boundary,
                "rows": candidate_rows,
                "qa": candidate_qa,
            }
    if directional_selected is None:
        raise RuntimeError("No geometry-locked directional candidate passed the guardrails")

    measured = directional_selected["measured"]
    boundary_summary = directional_selected["boundary"]
    boundary_rows = directional_selected["rows"]
    qa = directional_selected["qa"]
    boundary_pass = quality.boundary_guardrail(boundary_summary)

    uncertainty = np.zeros_like(source, dtype=np.float32)
    uncertainty[ys, xs] = uncertainty_crop
    residual = measured - source
    imwrite(
        args.outdir / "MEASUREMENT_sota_geometry_blind_16bit.tif",
        base.to_uint16(measured),
        photometric="minisblack",
        description="Geometry-constrained Blind2Sound-style CT adaptation; generative prior is low-frequency loss only; no generated-pixel writeback.",
    )
    imwrite(args.outdir / "MEASUREMENT_sota_residual_float32.tif", residual, photometric="minisblack")
    imwrite(args.outdir / "MEASUREMENT_sota_uncertainty_float32.tif", uncertainty, photometric="minisblack")
    imwrite(
        args.outdir / "AUDIT_safe_generative_prior_16bit.tif",
        base.to_uint16(safe_prior),
        photometric="minisblack",
        description="AUDIT ONLY: clipped low-frequency generative prior; not a measurement output.",
    )
    imwrite(
        args.outdir / "AUDIT_blind_prediction_16bit.tif",
        base.to_uint16(blind_prediction),
        photometric="minisblack",
        description="AUDIT ONLY: 16-phase blind prediction before Poisson-Gaussian data/geometry projection.",
    )
    base.save_preview(args.outdir / "MEASUREMENT_sota_geometry_blind_preview.png", measured)
    quality.save_boundary_overlay(
        args.outdir / "MEASUREMENT_sota_boundary_overlay.png", measured, boundary_rows, target
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "AdaptiveVisibleUNet",
            "features": args.features,
            "noise": noise,
        },
        args.outdir / "sota_geometry_blind_model.pt",
    )
    parameters = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            parameters[key] = str(value)
        elif isinstance(value, base.Roi):
            parameters[key] = value.__dict__
        else:
            parameters[key] = value
    manifest = {
        "completed": True,
        "source": source_info,
        "generative_prior": str(args.generative_prior),
        "device": str(device),
        "elapsed_seconds": round(time.time() - started, 2),
        "method": {
            "blind_backbone": "clean-room CT adaptation of Blind2Sound adaptive re-visible Poisson-Gaussian denoising",
            "paper": "Blind2Sound, ICCV 2025",
            "generative_role": "clipped low-frequency loss in raw low-gradient regions only",
            "training_geometry": "strong x/y edge, transverse width-profile, longitudinal endpoint-profile, and multiscale shape losses",
            "post_inference_geometry": "local sub-pixel displacement of denoised pixels to reliable raw endpoint coordinates",
            "directional_geometry": "v1 axial continuity constrained before application by raw x/y gradients and fixed endpoints",
            "generated_pixel_writeback": False,
            "raw_pixel_writeback": False,
        },
        "parameters": parameters,
        "noise_model": noise,
        "prior": prior_info,
        "projection": projection_info,
        "transverse_geometry_anchor": anchor_info,
        "projection_selection": {
            "selected": selected["summary"],
            "candidate_grid": projection_grid,
            "rule": "maximum flat-region roughness reduction among candidates passing width, shape, endpoint, and length guardrails",
        },
        "directional_projection_selection": {
            "selected": directional_selected["summary"],
            "candidate_grid": directional_grid,
            "rule": "maximum roughness reduction with fixed raw-coordinate FWHM, endpoint, length, acutance, contrast, and SSIM guardrails",
        },
        "geometry_correction": geometry_info,
        "boundary_geometry": boundary_summary,
        "measurement_qa": qa,
        "training_history": history,
    }
    (args.outdir / "sota_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completed": True,
        "output": str(args.outdir / "MEASUREMENT_sota_geometry_blind_16bit.tif"),
        "device": str(device),
        "measurement_guardrail_pass": qa["guardrail_pass"],
        "boundary_guardrail_pass": boundary_pass,
        "boundary_geometry": boundary_summary,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
