#!/usr/bin/env python3
"""v17 structure-carrier-conditioned residual diffusion for CT enhancement.

The model generates a bounded residual around a source-coordinate structural
carrier.  Lamella centerlines, finite-width boundaries, endpoints,
interlayers, raw/carrier confidence, and carrier detail are explicit
conditioning channels.  Differentiable geometry/detail losses are applied to
every predicted clean image.  A generated result is released only after the
independent v15/v16 per-lamella audit passes; otherwise the carrier is kept.

This is a single-image adaptation experiment, not a metrology-certified model.
The output is therefore labelled MEASUREMENT_CANDIDATE rather than ground truth.
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
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, sobel
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_nl_means
from tifffile import imread, imwrite

import boundary_cleanup as boundary
import generative_shape_constraint as shape
import generative_shape_project as project
import measurement_quality_optimize as v16
import pipeline as base
import quality_optimize as quality


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(1, min(4, out_channels // 4))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class StructureConditionedResidualUNet(nn.Module):
    """Compact conditional diffusion decoder predicting a bounded residual."""

    def __init__(self, condition_channels: int, features: int = 12):
        super().__init__()
        channels = 1 + condition_channels + 1  # noisy image + conditions + time map
        self.enc1 = ConvBlock(channels, features)
        self.enc2 = ConvBlock(features, features * 2)
        self.enc3 = ConvBlock(features * 2, features * 4)
        self.middle = ConvBlock(features * 4, features * 4)
        self.dec3 = ConvBlock(features * 8, features * 2)
        self.dec2 = ConvBlock(features * 4, features)
        self.dec1 = ConvBlock(features * 2, features)
        self.head = nn.Conv2d(features, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def resize(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            values, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self,
        noisy: torch.Tensor,
        conditions: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        time_map = time_fraction[:, None, None, None].expand(
            -1, 1, noisy.shape[-2], noisy.shape[-1]
        )
        values = torch.cat((noisy, conditions, time_map), dim=1)
        enc1 = self.enc1(values)
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        enc3 = self.enc3(F.avg_pool2d(enc2, 2))
        middle = self.middle(F.avg_pool2d(enc3, 2))
        dec3 = self.dec3(torch.cat((self.resize(middle, enc3), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.resize(dec3, enc2), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.resize(dec2, enc1), enc1), dim=1))
        return torch.tanh(self.head(dec1))


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


def normalize_map(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    scale = float(np.percentile(np.abs(values), percentile)) + 1e-8
    return np.clip(np.abs(values) / scale, 0.0, 1.0).astype(np.float32)


def match_proposal(proposal_path: Path, carrier: np.ndarray) -> np.ndarray:
    with Image.open(proposal_path) as image:
        image = image.convert("L")
        image = image.resize((carrier.shape[1], carrier.shape[0]), Image.Resampling.LANCZOS)
        proposal = np.asarray(image, dtype=np.float32) / 255.0
    carrier_lo, carrier_hi = (float(value) for value in np.percentile(carrier, (1.0, 99.0)))
    proposal_lo, proposal_hi = (float(value) for value in np.percentile(proposal, (1.0, 99.0)))
    matched = (proposal - proposal_lo) / max(proposal_hi - proposal_lo, 1e-8)
    return np.clip(matched * (carrier_hi - carrier_lo) + carrier_lo, 0.0, 1.0).astype(
        np.float32
    )


def build_structure_maps(
    image_shape: tuple[int, int],
    target: base.Roi,
    layers: list[dict],
) -> dict[str, np.ndarray]:
    """Rasterize curved finite-width lamella and interlayer constraints."""
    height, width = target.y1 - target.y0, target.x1 - target.x0
    lamella = np.zeros((height, width), dtype=np.float32)
    centerline = np.zeros_like(lamella)
    boundary_map = np.zeros_like(lamella)
    endpoint = np.zeros_like(lamella)
    confidence = np.zeros_like(lamella)
    interlayer = np.zeros_like(lamella)

    for row in layers:
        path = np.asarray(row["path_x"], dtype=np.float32)
        pitch = max(float(row["pitch_px"]), 4.0)
        measured_width = row.get("width_median_px")
        if measured_width is None:
            measured_width = max(0.15 * pitch, 1.2)
        half_width = max(0.5 * float(measured_width), 0.6)
        top = max(target.y0, int(math.floor(float(row["top_y_px"]))))
        bottom = min(target.y1 - 1, int(math.ceil(float(row["bottom_y_px"]))))
        row_confidence = float(row.get("constraint_confidence", 1.0))
        for global_y in range(top, bottom + 1):
            local_y = global_y - target.y0
            center = float(path[global_y]) - target.x0
            x0 = max(0, int(math.floor(center - 0.55 * pitch)))
            x1 = min(width, int(math.ceil(center + 0.55 * pitch)) + 1)
            if x1 <= x0:
                continue
            dx = np.arange(x0, x1, dtype=np.float32) - center
            current_lamella = np.exp(-np.power(np.abs(dx) / (half_width + 0.8), 4.0))
            current_center = np.exp(-0.5 * np.square(dx / 1.1))
            current_boundary = np.exp(
                -0.5 * np.square((np.abs(dx) - half_width) / 0.75)
            )
            lamella[local_y, x0:x1] = np.maximum(
                lamella[local_y, x0:x1], current_lamella
            )
            centerline[local_y, x0:x1] = np.maximum(
                centerline[local_y, x0:x1], current_center
            )
            boundary_map[local_y, x0:x1] = np.maximum(
                boundary_map[local_y, x0:x1], current_boundary
            )
            confidence[local_y, x0:x1] = np.maximum(
                confidence[local_y, x0:x1], row_confidence * current_lamella
            )
        endpoint_radius_y = 6
        endpoint_radius_x = max(3, int(round(0.35 * pitch)))
        for position in (float(row["top_y_px"]), float(row["bottom_y_px"])):
            global_y0 = max(target.y0, int(round(position)) - endpoint_radius_y)
            global_y1 = min(target.y1, int(round(position)) + endpoint_radius_y + 1)
            for global_y in range(global_y0, global_y1):
                center = int(round(float(path[global_y]))) - target.x0
                x0 = max(0, center - endpoint_radius_x)
                x1 = min(width, center + endpoint_radius_x + 1)
                if x1 > x0:
                    dy = (global_y - position) / max(endpoint_radius_y, 1)
                    endpoint[global_y - target.y0, x0:x1] = np.maximum(
                        endpoint[global_y - target.y0, x0:x1],
                        float(math.exp(-0.5 * dy * dy)),
                    )

    for side in ("left", "right"):
        rows = sorted(
            (row for row in layers if row["side"] == side),
            key=lambda item: float(item["center_x_px"]),
        )
        for first, second in zip(rows, rows[1:]):
            top = max(
                target.y0,
                int(math.ceil(max(float(first["top_y_px"]), float(second["top_y_px"])))),
            )
            bottom = min(
                target.y1 - 1,
                int(math.floor(min(float(first["bottom_y_px"]), float(second["bottom_y_px"])))),
            )
            first_path = np.asarray(first["path_x"], dtype=np.float32)
            second_path = np.asarray(second["path_x"], dtype=np.float32)
            first_width = first.get("width_median_px")
            second_width = second.get("width_median_px")
            first_half = 0.5 * float(
                first_width if first_width is not None else max(0.15 * float(first["pitch_px"]), 1.2)
            )
            second_half = 0.5 * float(
                second_width if second_width is not None else max(0.15 * float(second["pitch_px"]), 1.2)
            )
            for global_y in range(top, bottom + 1):
                x0 = int(math.ceil(float(first_path[global_y]) + first_half)) - target.x0
                x1 = int(math.floor(float(second_path[global_y]) - second_half)) - target.x0
                x0, x1 = max(0, x0), min(width, x1)
                if x1 > x0:
                    interlayer[global_y - target.y0, x0:x1] = 1.0

    boundary_map = np.clip(gaussian_filter(boundary_map, sigma=0.55), 0.0, 1.0)
    endpoint = np.clip(gaussian_filter(endpoint, sigma=(0.75, 0.50)), 0.0, 1.0)
    interlayer = np.clip(gaussian_filter(interlayer, sigma=0.65), 0.0, 1.0)
    confidence = np.clip(confidence, 0.0, 1.0)
    protection = np.maximum(boundary_map, endpoint)
    allowed = np.clip(1.0 - 0.985 * protection, 0.015, 1.0)
    # Keep residual freedom lower inside measurable lamellae than in flat gaps.
    allowed *= 1.0 - 0.55 * lamella * (1.0 - boundary_map)
    return {
        "lamella": lamella.astype(np.float32),
        "centerline": centerline.astype(np.float32),
        "boundary": boundary_map.astype(np.float32),
        "endpoint": endpoint.astype(np.float32),
        "interlayer": interlayer.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "protection": protection.astype(np.float32),
        "allowed": allowed.astype(np.float32),
    }


def load_uncertainty(
    path: Path | None,
    shape_: tuple[int, int],
    target: base.Roi,
) -> np.ndarray:
    if path is None:
        return np.zeros((target.y1 - target.y0, target.x1 - target.x0), dtype=np.float32)
    uncertainty = np.asarray(imread(path), dtype=np.float32)
    if uncertainty.ndim != 2:
        raise ValueError(f"expected single-channel uncertainty, got {uncertainty.shape}")
    if uncertainty.shape != shape_:
        raise ValueError(f"uncertainty shape {uncertainty.shape} != source shape {shape_}")
    values = uncertainty[target.slices()]
    return normalize_map(values, 99.0)


def build_conditions(
    carrier: np.ndarray,
    proposal: np.ndarray,
    target: base.Roi,
    maps: dict[str, np.ndarray],
    uncertainty: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    carrier_crop = carrier[target.slices()]
    proposal_crop = proposal[target.slices()]
    smoothed = gaussian_filter(carrier_crop, sigma=0.70)
    gradient_x = normalize_map(sobel(smoothed, axis=1), 99.0)
    gradient_y = normalize_map(sobel(smoothed, axis=0), 99.0)
    conditions = np.stack(
        (
            carrier_crop,
            gaussian_filter(proposal_crop, sigma=2.0),
            gradient_x,
            gradient_y,
            maps["centerline"],
            maps["boundary"],
            maps["endpoint"],
            maps["interlayer"],
            maps["confidence"],
            uncertainty,
        ),
        axis=0,
    ).astype(np.float32)
    names = [
        "structure_carrier",
        "low_frequency_appearance_proposal",
        "carrier_gradient_x",
        "carrier_gradient_y",
        "lamella_centerline_field",
        "finite_width_boundary_field",
        "endpoint_protection_field",
        "interlayer_field",
        "dual_evidence_confidence",
        "blind_uncertainty",
    ]
    return conditions, names


def build_pseudo_clean_target(
    carrier_crop: np.ndarray,
    proposal_crop: np.ndarray,
    maps: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict]:
    sigma_n = base.noise_sigma_mad(carrier_crop)
    nlm = denoise_nl_means(
        carrier_crop,
        h=0.78 * sigma_n,
        sigma=sigma_n,
        fast_mode=True,
        patch_size=5,
        patch_distance=5,
        channel_axis=None,
        preserve_range=True,
    ).astype(np.float32)
    axial = gaussian_filter(carrier_crop, sigma=(1.60, 0.18))
    carrier_low = gaussian_filter(carrier_crop, sigma=6.0)
    proposal_low = gaussian_filter(proposal_crop, sigma=6.0)
    smooth_gradient = np.hypot(
        sobel(gaussian_filter(carrier_crop, 0.75), axis=0),
        sobel(gaussian_filter(carrier_crop, 0.75), axis=1),
    )
    threshold = float(np.percentile(smooth_gradient, 45.0)) + 1e-8
    low_structure = np.exp(-np.square(smooth_gradient / threshold)).astype(np.float32)
    lamella_interior = maps["lamella"] * (1.0 - maps["boundary"])
    denoise_delta = (
        (0.58 * (1.0 - lamella_interior) * (nlm - carrier_crop))
        + (0.24 * lamella_interior * (axial - carrier_crop))
    )
    appearance_cap = max(0.30 * sigma_n, 2.0 / 65535.0)
    appearance_delta = 0.10 * low_structure * np.clip(
        proposal_low - carrier_low, -appearance_cap, appearance_cap
    )
    total_cap = max(1.10 * sigma_n, 5.0 / 65535.0)
    delta = maps["allowed"] * np.clip(
        denoise_delta + appearance_delta, -total_cap, total_cap
    )
    target = np.clip(carrier_crop + delta, 0.0, 1.0).astype(np.float32)
    return target, {
        "method": "geometry-gated NLM/axial pseudo-clean target plus capped low-frequency appearance",
        "noise_sigma_normalized": sigma_n,
        "nlm_h_sigma": 0.78,
        "nlm_patch_size": 5,
        "nlm_patch_distance": 5,
        "lamella_axial_sigma_px": [1.60, 0.18],
        "lamella_pseudo_target_strength": 0.24,
        "non_lamella_pseudo_target_strength": 0.58,
        "appearance_strength": 0.10,
        "appearance_change_cap_normalized": appearance_cap,
        "total_change_cap_normalized": total_cap,
        "generated_proposal_pixel_writeback": False,
    }


def finite_gradients(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = F.pad(values[..., :, 1:] - values[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(values[..., 1:, :] - values[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def smooth_tensor(values: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.avg_pool2d(values, kernel, stride=1, padding=kernel // 2)


def structure_detail_losses(
    prediction: torch.Tensor,
    pseudo_target: torch.Tensor,
    carrier: torch.Tensor,
    map_tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable carrier, boundary, width, endpoint, gap and detail losses."""
    lamella = map_tensor[:, 0:1]
    boundary_map = map_tensor[:, 2:3]
    endpoint = map_tensor[:, 3:4]
    interlayer = map_tensor[:, 4:5]
    confidence = map_tensor[:, 5:6]
    protection = torch.maximum(boundary_map, endpoint)

    charbonnier = torch.sqrt((prediction - pseudo_target).square() + 1e-6).mean()
    carrier_weight = 0.20 + 2.8 * protection + 0.45 * confidence
    carrier_consistency = (
        carrier_weight * torch.sqrt((prediction - carrier).square() + 1e-6)
    ).mean()

    pred_dx, pred_dy = finite_gradients(smooth_tensor(prediction, 3))
    carrier_dx, carrier_dy = finite_gradients(smooth_tensor(carrier, 3))
    edge_x = ((0.15 + boundary_map) * (pred_dx - carrier_dx).abs()).mean()
    edge_y = ((0.10 + endpoint) * (pred_dy - carrier_dy).abs()).mean()

    pred_width_profile = prediction.mean(dim=-2)
    carrier_width_profile = carrier.mean(dim=-2)
    width_gradient = (
        finite_gradients(pred_width_profile.unsqueeze(-2))[0]
        - finite_gradients(carrier_width_profile.unsqueeze(-2))[0]
    ).abs().mean()
    pred_length_profile = prediction.mean(dim=-1)
    carrier_length_profile = carrier.mean(dim=-1)
    endpoint_gradient = (
        finite_gradients(pred_length_profile.unsqueeze(-1))[1]
        - finite_gradients(carrier_length_profile.unsqueeze(-1))[1]
    ).abs().mean()

    pred_mid = smooth_tensor(prediction, 3) - smooth_tensor(prediction, 9)
    carrier_mid = smooth_tensor(carrier, 3) - smooth_tensor(carrier, 9)
    detail = ((0.10 + lamella) * (pred_mid - carrier_mid).abs()).mean()
    gap = (
        interlayer * (smooth_tensor(prediction, 5) - smooth_tensor(carrier, 5)).abs()
    ).sum() / (interlayer.sum() + 1e-6)

    pred_magnitude = torch.sqrt(pred_dx.square() + pred_dy.square() + 1e-8)
    carrier_magnitude = torch.sqrt(carrier_dx.square() + carrier_dy.square() + 1e-8)
    boundary_field = (
        boundary_map * (pred_magnitude - carrier_magnitude).abs()
    ).sum() / (boundary_map.sum() + 1e-6)
    return {
        "reconstruction": charbonnier,
        "carrier_consistency": carrier_consistency,
        "edge_x": edge_x,
        "edge_y": edge_y,
        "width_profile": width_gradient,
        "endpoint_profile": endpoint_gradient,
        "midscale_detail": detail,
        "interlayer": gap,
        "boundary_field": boundary_field,
    }


def diffusion_schedule(steps: int, device: torch.device) -> torch.Tensor:
    beta = torch.linspace(1e-4, 0.018, steps, device=device)
    return torch.cumprod(1.0 - beta, dim=0)


def random_patches(
    arrays: list[np.ndarray],
    batch: int,
    patch: int,
    generator: np.random.Generator,
) -> list[np.ndarray]:
    height, width = arrays[0].shape[-2:]
    if patch % 8:
        raise ValueError("patch size must be divisible by 8")
    if min(height, width) < patch:
        raise ValueError(f"condition crop {height}x{width} is smaller than patch={patch}")
    batches: list[list[np.ndarray]] = [[] for _ in arrays]
    for _ in range(batch):
        y = int(generator.integers(0, height - patch + 1))
        # Draw equally from left lamellae, central body, and right lamellae.
        zone = int(generator.integers(0, 3))
        zone_edges = np.linspace(0, width, 4, dtype=int)
        low = max(0, int(zone_edges[zone]) - patch // 4)
        high = min(width - patch, int(zone_edges[zone + 1]) - 3 * patch // 4)
        x = int(generator.integers(low, max(low + 1, high + 1)))
        for index, values in enumerate(arrays):
            batches[index].append(values[..., y : y + patch, x : x + patch])
    return [np.stack(values, axis=0) for values in batches]


def train_model(
    carrier_crop: np.ndarray,
    pseudo_target: np.ndarray,
    conditions: np.ndarray,
    maps: dict[str, np.ndarray],
    iterations: int,
    patch: int,
    batch: int,
    features: int,
    diffusion_steps: int,
    max_delta: float,
    seed: int,
    device: torch.device,
) -> tuple[StructureConditionedResidualUNet, list[dict], torch.Tensor]:
    torch.manual_seed(seed)
    np_generator = np.random.default_rng(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    map_stack = np.stack(
        (
            maps["lamella"],
            maps["centerline"],
            maps["boundary"],
            maps["endpoint"],
            maps["interlayer"],
            maps["confidence"],
        ),
        axis=0,
    ).astype(np.float32)
    model = StructureConditionedResidualUNet(conditions.shape[0], features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(iterations, 1), eta_min=2e-5
    )
    alpha_bar = diffusion_schedule(diffusion_steps, device)
    history: list[dict] = []
    model.train()
    for step in range(1, iterations + 1):
        carrier_batch, target_batch, condition_batch, map_batch, allowed_batch = random_patches(
            [
                carrier_crop[None],
                pseudo_target[None],
                conditions,
                map_stack,
                maps["allowed"][None],
            ],
            batch,
            patch,
            np_generator,
        )
        carrier_tensor = torch.from_numpy(carrier_batch).to(device)
        target_tensor = torch.from_numpy(target_batch).to(device)
        condition_tensor = torch.from_numpy(condition_batch).to(device)
        map_tensor = torch.from_numpy(map_batch).to(device)
        allowed_tensor = torch.from_numpy(allowed_batch).to(device)
        time_index = torch.randint(4, diffusion_steps, (batch,), device=device)
        current_alpha = alpha_bar[time_index][:, None, None, None]
        noise = torch.randn_like(target_tensor)
        noisy = current_alpha.sqrt() * target_tensor + (1.0 - current_alpha).sqrt() * noise
        residual_prediction = model(
            noisy, condition_tensor, time_index.float() / max(diffusion_steps - 1, 1)
        )
        prediction = torch.clamp(
            carrier_tensor + max_delta * allowed_tensor * residual_prediction, 0.0, 1.0
        )
        epsilon_prediction = (
            noisy - current_alpha.sqrt() * prediction
        ) / (1.0 - current_alpha).sqrt().clamp_min(1e-4)
        diffusion_loss = F.mse_loss(epsilon_prediction, noise)
        losses = structure_detail_losses(
            prediction, target_tensor, carrier_tensor, map_tensor
        )
        total = (
            0.20 * diffusion_loss
            + 3.00 * losses["reconstruction"]
            + 0.80 * losses["carrier_consistency"]
            + 3.00 * losses["edge_x"]
            + 2.20 * losses["edge_y"]
            + 2.00 * losses["width_profile"]
            + 1.50 * losses["endpoint_profile"]
            + 1.20 * losses["midscale_detail"]
            + 0.80 * losses["interlayer"]
            + 2.00 * losses["boundary_field"]
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 20 == 0 or step == iterations:
            item = {
                "iteration": step,
                "total": float(total.detach()),
                "diffusion": float(diffusion_loss.detach()),
            }
            item.update({name: float(value.detach()) for name, value in losses.items()})
            history.append(item)
            print(json.dumps({"training": item}), flush=True)
    return model, history, alpha_bar


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def cosine_window(height: int, width: int, floor: float = 0.08) -> np.ndarray:
    wy = np.hanning(height) if height > 2 else np.ones(height)
    wx = np.hanning(width) if width > 2 else np.ones(width)
    return np.maximum(np.outer(wy, wx), floor).astype(np.float32)


@torch.no_grad()
def ddim_tile(
    model: StructureConditionedResidualUNet,
    carrier: torch.Tensor,
    conditions: torch.Tensor,
    allowed: torch.Tensor,
    alpha_bar: torch.Tensor,
    sample_steps: int,
    max_delta: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=carrier.device).manual_seed(seed)
    noisy = torch.randn(carrier.shape, generator=generator, device=carrier.device)
    indices = torch.linspace(
        len(alpha_bar) - 1, 0, max(sample_steps, 2), device=carrier.device
    ).round().long().unique_consecutive()
    prediction = carrier
    for position, time_index in enumerate(indices):
        batch_time = torch.full(
            (carrier.shape[0],),
            float(time_index) / max(len(alpha_bar) - 1, 1),
            device=carrier.device,
        )
        residual_prediction = model(noisy, conditions, batch_time)
        prediction = torch.clamp(
            carrier + max_delta * allowed * residual_prediction, 0.0, 1.0
        )
        if position == len(indices) - 1:
            break
        current_alpha = alpha_bar[time_index]
        next_alpha = alpha_bar[indices[position + 1]]
        epsilon = (noisy - current_alpha.sqrt() * prediction) / (
            1.0 - current_alpha
        ).sqrt().clamp_min(1e-4)
        noisy = next_alpha.sqrt() * prediction + (1.0 - next_alpha).sqrt() * epsilon
    return prediction


def generate_tiled(
    model: StructureConditionedResidualUNet,
    carrier_crop: np.ndarray,
    conditions: np.ndarray,
    allowed: np.ndarray,
    alpha_bar: torch.Tensor,
    tile: int,
    overlap: int,
    sample_steps: int,
    max_delta: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    height, width = carrier_crop.shape
    result = np.zeros_like(carrier_crop, dtype=np.float32)
    weights = np.zeros_like(carrier_crop, dtype=np.float32)
    window = cosine_window(tile, tile)
    for y in tile_starts(height, tile, overlap):
        for x in tile_starts(width, tile, overlap):
            carrier_tensor = torch.from_numpy(
                carrier_crop[None, None, y : y + tile, x : x + tile]
            ).to(device)
            condition_tensor = torch.from_numpy(
                conditions[None, :, y : y + tile, x : x + tile]
            ).to(device)
            allowed_tensor = torch.from_numpy(
                allowed[None, None, y : y + tile, x : x + tile]
            ).to(device)
            prediction = ddim_tile(
                model,
                carrier_tensor,
                condition_tensor,
                allowed_tensor,
                alpha_bar,
                sample_steps,
                max_delta,
                seed + 1009 * y + 9176 * x,
            )[0, 0].cpu().numpy()
            result[y : y + tile, x : x + tile] += window * prediction
            weights[y : y + tile, x : x + tile] += window
    return (result / np.maximum(weights, 1e-8)).astype(np.float32)


def postprocess_generated(
    generated_crop: np.ndarray,
    carrier_crop: np.ndarray,
    maps: dict[str, np.ndarray],
    max_delta: float,
) -> tuple[np.ndarray, dict]:
    """Zero-phase residual cleanup without any spatial transform."""
    fine = gaussian_filter(generated_crop, sigma=0.55)
    low_structure = 1.0 - np.maximum(maps["boundary"], maps["endpoint"])
    cleanup = 0.22 * low_structure * (fine - generated_crop)
    delta = np.clip(
        generated_crop + cleanup - carrier_crop, -max_delta, max_delta
    )
    output = np.clip(carrier_crop + maps["allowed"] * delta, 0.0, 1.0).astype(np.float32)
    return output, {
        "method": "zero-phase low-structure residual cleanup and carrier-centered cap",
        "cleanup_strength": 0.22,
        "maximum_change_normalized": max_delta,
        "spatial_transform": None,
        "raw_pixel_writeback": False,
    }


def masked_noise_rms(image: np.ndarray, mask: np.ndarray, target: base.Roi) -> float:
    crop = image[target.slices()]
    residual = crop - gaussian_filter(crop, sigma=1.15)
    selected = mask > 0.35
    if not np.any(selected):
        return 0.0
    return float(np.sqrt(np.mean(np.square(residual[selected]))))


def save_condition_audit(
    path: Path,
    carrier_crop: np.ndarray,
    maps: dict[str, np.ndarray],
) -> None:
    lo, hi = (float(value) for value in np.percentile(carrier_crop, (0.5, 99.7)))
    gray = np.rint(
        np.clip((carrier_crop - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255
    ).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[..., 0] = np.maximum(rgb[..., 0], np.rint(220 * maps["endpoint"]).astype(np.uint8))
    rgb[..., 1] = np.maximum(rgb[..., 1], np.rint(200 * maps["boundary"]).astype(np.uint8))
    rgb[..., 2] = np.maximum(rgb[..., 2], np.rint(180 * maps["interlayer"]).astype(np.uint8))
    Image.fromarray(rgb, mode="RGB").save(path)


def save_comparison(
    path: Path,
    carrier: np.ndarray,
    generated: np.ndarray,
    selected: np.ndarray,
    target: base.Roi,
) -> None:
    images = [carrier[target.slices()], generated[target.slices()], selected[target.slices()]]
    titles = ["v16 carrier", "v17 generated raw", "v17 guarded candidate"]
    lo, hi = (float(value) for value in np.percentile(images[0], (0.5, 99.7)))
    panels = []
    for values, title in zip(images, titles):
        mapped = np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        panel = Image.fromarray(np.rint(mapped * 255).astype(np.uint8), mode="L")
        panel = panel.resize((760, 304), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (760, 330), 0)
        canvas.paste(panel, (0, 26))
        ImageDraw.Draw(canvas).text((8, 7), title, fill=255, font=ImageFont.load_default())
        panels.append(canvas)
    output = Image.new("L", (2280, 330), 0)
    for index, panel in enumerate(panels):
        output.paste(panel, (760 * index, 0))
    output.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-roi", type=base.parse_roi, default=base.Roi(600, 1240, 300, 1900))
    parser.add_argument("--central-roi", type=base.parse_roi, default=base.Roi(700, 1140, 985, 1205))
    parser.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    parser.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    parser.add_argument("--left-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    parser.add_argument("--right-body-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    parser.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    parser.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    parser.add_argument("--iterations", type=int, default=320)
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--diffusion-steps", type=int, default=48)
    parser.add_argument("--sample-steps", type=int, default=6)
    parser.add_argument("--tile", type=int, default=192)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    carrier, carrier_info = base.load_gray(args.carrier)
    if source.shape != carrier.shape:
        raise ValueError(f"source shape {source.shape} != carrier shape {carrier.shape}")
    target = args.target_roi.clamp(source.shape)
    central = args.central_roi.clamp(source.shape)
    left = args.left_roi.clamp(source.shape)
    right = args.right_roi.clamp(source.shape)
    proposal = match_proposal(args.proposal, carrier)
    uncertainty = load_uncertainty(args.uncertainty, source.shape, target)
    layers = shape.measure_layers(
        source, left, right, args.top_range, args.bottom_range, guide=carrier
    )
    maps = build_structure_maps(source.shape, target, layers)
    conditions, condition_names = build_conditions(
        carrier, proposal, target, maps, uncertainty
    )
    carrier_crop = carrier[target.slices()]
    proposal_crop = proposal[target.slices()]
    pseudo_target, pseudo_info = build_pseudo_clean_target(
        carrier_crop, proposal_crop, maps
    )
    max_delta = max(
        1.20 * float(pseudo_info["noise_sigma_normalized"]), 6.0 / 65535.0
    )
    device = choose_device(args.device)
    started = time.time()
    model, history, alpha_bar = train_model(
        carrier_crop,
        pseudo_target,
        conditions,
        maps,
        args.iterations,
        args.patch,
        args.batch,
        args.features,
        args.diffusion_steps,
        max_delta,
        args.seed,
        device,
    )
    generated_crop = generate_tiled(
        model,
        carrier_crop,
        conditions,
        maps["allowed"],
        alpha_bar,
        args.tile,
        args.overlap,
        args.sample_steps,
        max_delta,
        args.seed,
        device,
    )
    postprocessed_crop, postprocess_info = postprocess_generated(
        generated_crop, carrier_crop, maps, max_delta
    )
    generated_full = base.feather_insert(carrier, generated_crop, target, ramp=18)

    references = quality.build_boundary_references(
        source,
        carrier,
        args.left_body_roi.clamp(source.shape),
        args.right_body_roi.clamp(source.shape),
        args.top_range,
        args.bottom_range,
    )
    endpoint_masks, _ = boundary.build_endpoint_envelope_masks(
        source.shape, references, target
    )
    baseline_audit = project.audit_projection(
        source, carrier, carrier, carrier, layers, left, right, args.top_range, args.bottom_range
    )
    baseline_summary = v16.metric_summary(baseline_audit)
    baseline_lamella_noise = boundary.boundary_cleanliness_metrics(
        carrier, target, endpoint_masks
    )["interior_axial_noise_rms"]
    baseline_central_noise = masked_noise_rms(
        carrier,
        np.pad(
            np.ones((central.y1 - central.y0, central.x1 - central.x0), dtype=np.float32),
            (
                (central.y0 - target.y0, target.y1 - central.y1),
                (central.x0 - target.x0, target.x1 - central.x1),
            ),
        ),
        target,
    )
    baseline_flat_noise = masked_noise_rms(
        carrier, (1.0 - maps["protection"]) * (1.0 - 0.5 * maps["lamella"]), target
    )

    candidates = []
    selected = None
    for strength in (0.0, 0.12, 0.20, 0.30, 0.42, 0.55, 0.70, 0.85, 1.0):
        crop = np.clip(
            carrier_crop + float(strength) * (postprocessed_crop - carrier_crop),
            0.0,
            1.0,
        ).astype(np.float32)
        candidate = base.feather_insert(carrier, crop, target, ramp=18)
        audit = project.audit_projection(
            source,
            carrier,
            carrier,
            candidate,
            layers,
            left,
            right,
            args.top_range,
            args.bottom_range,
        )
        summary = v16.metric_summary(audit)
        target_ssim = float(structural_similarity(
            carrier[target.slices()], candidate[target.slices()], data_range=1.0
        ))
        guardrail_pass, checks = v16.strict_guardrail(
            summary, baseline_summary, target_ssim
        )
        lamella_noise = boundary.boundary_cleanliness_metrics(
            candidate, target, endpoint_masks
        )["interior_axial_noise_rms"]
        central_noise = masked_noise_rms(
            candidate,
            np.pad(
                np.ones((central.y1 - central.y0, central.x1 - central.x0), dtype=np.float32),
                (
                    (central.y0 - target.y0, target.y1 - central.y1),
                    (central.x0 - target.x0, target.x1 - central.x1),
                ),
            ),
            target,
        )
        flat_noise = masked_noise_rms(
            candidate,
            (1.0 - maps["protection"]) * (1.0 - 0.5 * maps["lamella"]),
            target,
        )
        lamella_reduction = 1.0 - lamella_noise / max(baseline_lamella_noise, 1e-8)
        central_reduction = 1.0 - central_noise / max(baseline_central_noise, 1e-8)
        flat_reduction = 1.0 - flat_noise / max(baseline_flat_noise, 1e-8)
        score = 0.55 * lamella_reduction + 0.25 * central_reduction + 0.20 * flat_reduction
        item = {
            "generated_residual_strength": float(strength),
            "lamella_axial_noise_reduction": lamella_reduction,
            "central_high_frequency_noise_reduction": central_reduction,
            "flat_region_high_frequency_noise_reduction": flat_reduction,
            "ssim_vs_carrier": target_ssim,
            "structure_geometry": summary,
            "guardrail_checks": checks,
            "guardrail_pass": guardrail_pass,
            "score": score,
        }
        candidates.append(item)
        if guardrail_pass and (selected is None or score > selected["summary"]["score"]):
            selected = {"summary": item, "image": candidate, "audit": audit}
    if selected is None:
        raise RuntimeError("No v17 candidate passed the independent v15/v16 guardrails")

    final = selected["image"]
    audit = selected["audit"]
    condition_rows = audit.pop("condition_boundary_rows")
    audit.pop("raw_boundary_rows")
    layer_rows = audit.pop("layer_comparison_rows")
    gap_rows = audit.pop("gap_comparison_rows")
    detail_rows = audit.pop("structure_detail_rows")

    imwrite(
        args.outdir / "MEASUREMENT_CANDIDATE_v17_structure_conditioned_16bit.tif",
        base.to_uint16(final),
        photometric="minisblack",
        description=(
            "MEASUREMENT_CANDIDATE v17: structure-carrier-conditioned residual diffusion; "
            "generated pixels present; independent geometry guardrails passed."
        ),
    )
    base.save_preview(
        args.outdir / "MEASUREMENT_CANDIDATE_v17_structure_conditioned.png", final
    )
    base.save_preview(args.outdir / "GENERATIVE_v17_raw.png", generated_full)
    base.save_preview(args.outdir / "PSEUDO_TARGET_v17_audit_only.png", base.feather_insert(
        carrier, pseudo_target, target, ramp=18
    ))
    save_condition_audit(
        args.outdir / "AUDIT_v17_structure_conditions.png", carrier_crop, maps
    )
    save_comparison(
        args.outdir / "MEASUREMENT_CANDIDATE_v17_comparison.png",
        carrier,
        generated_full,
        final,
        target,
    )
    quality.save_boundary_overlay(
        args.outdir / "MEASUREMENT_CANDIDATE_v17_boundary_overlay.png",
        final,
        condition_rows,
        target,
    )
    shape.write_csv(args.outdir / "lamella_v17_comparison.csv", layer_rows)
    shape.write_csv(args.outdir / "interlayer_v17_comparison.csv", gap_rows)
    shape.write_csv(args.outdir / "structure_detail_v17.csv", detail_rows)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "StructureConditionedResidualUNet",
            "condition_channels": condition_names,
            "features": args.features,
            "diffusion_steps": args.diffusion_steps,
            "max_delta_normalized": max_delta,
        },
        args.outdir / "structure_conditioned_diffusion_v17.pt",
    )
    payload = {
        "completed": True,
        "release": "v17-structure-carrier-conditioned-diffusion",
        "status": "MEASUREMENT_CANDIDATE; requires multi-image and calibrated-phantom validation",
        "source": source_info,
        "carrier": carrier_info,
        "proposal": str(args.proposal),
        "uncertainty": str(args.uncertainty) if args.uncertainty else None,
        "dimensions": {"width": source.shape[1], "height": source.shape[0]},
        "model": {
            "architecture": "conditional bounded-residual DDIM adaptation",
            "condition_channels": condition_names,
            "generated_pixels_present": bool(
                selected["summary"]["generated_residual_strength"] > 0.0
            ),
            "selected_generated_residual_strength": selected["summary"][
                "generated_residual_strength"
            ],
            "external_proposal_pixel_writeback": False,
            "training_iterations": args.iterations,
            "patch": args.patch,
            "batch": args.batch,
            "features": args.features,
            "diffusion_steps": args.diffusion_steps,
            "sampling_steps": args.sample_steps,
            "max_delta_normalized": max_delta,
            "losses": [
                "diffusion epsilon",
                "pseudo-clean reconstruction",
                "carrier confidence",
                "transverse edge",
                "longitudinal endpoint",
                "width projection",
                "length projection",
                "midscale lamella detail",
                "interlayer consistency",
                "finite-width boundary field",
            ],
        },
        "pseudo_target": pseudo_info,
        "postprocess": postprocess_info,
        "selection_rule": (
            "maximum weighted lamella/central/flat noise reduction among generated "
            "residual strengths passing the independent v15/v16 audit"
        ),
        "selected": selected["summary"],
        "candidate_grid": candidates,
        "training_history": history,
        "audit": audit,
        "elapsed_seconds": round(time.time() - started, 2),
        "fallback": (
            "strength 0 is the unmodified carrier and remains eligible; use the v16 "
            "carrier whenever no nonzero generated candidate has external validation"
        ),
    }
    (args.outdir / "structure_conditioned_diffusion_v17_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completed": True,
        "output": str(
            args.outdir / "MEASUREMENT_CANDIDATE_v17_structure_conditioned_16bit.tif"
        ),
        "selected": selected["summary"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
