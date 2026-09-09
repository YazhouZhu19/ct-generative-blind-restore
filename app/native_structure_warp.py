#!/usr/bin/env python3
"""Native-canvas soft coordinate guidance for a source-sized generation.

The measured blind-guide ridges/endpoints and the generated ridges/endpoints
are monotonically matched on each lamella stack.  A bounded, continuous local
two-axis displacement then moves the generated structures toward those
coordinates.  Pixels are sampled exclusively from the accepted generation:
no guide/raw intensity is written into the result, no canvas resize occurs,
and no analytic structure is drawn.  The central block and all pixels outside
the two stack masks are locked.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity
from tifffile import imwrite

import generative_shape_constraint as constraint
import length_optimize as length
import pipeline as base
import structure_audit as audit


def monotonic_match(
    fixed: np.ndarray,
    moving: np.ndarray,
    skip_fixed_cost: float = 0.90,
    skip_moving_cost: float = 0.62,
    position_scale: float = 0.030,
) -> list[tuple[int, int]]:
    """Return order-preserving correspondences, retaining the smaller set.

    In this application the guide is the authoritative structural inventory.
    When the generated detector returns the same or a larger ridge count, every
    guide ridge must therefore receive a match; only generated detections may
    be skipped.  This prevents a locally difficult match from silently dropping
    a measured constraint.
    """
    fixed = np.asarray(fixed, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if fixed.size < 3 or moving.size < 3:
        raise ValueError("At least three fixed and moving ridges are required")
    if fixed.size > moving.size:
        reverse = monotonic_match(
            moving,
            fixed,
            skip_fixed_cost=skip_moving_cost,
            skip_moving_cost=skip_fixed_cost,
            position_scale=position_scale,
        )
        return [(second, first) for first, second in reverse]
    fixed_span = max(float(fixed[-1] - fixed[0]), 1e-8)
    moving_span = max(float(moving[-1] - moving[0]), 1e-8)
    fixed_n = (fixed - fixed[0]) / fixed_span
    moving_n = (moving - moving[0]) / moving_span
    n, m = fixed.size, moving.size
    score = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    action = np.zeros((n + 1, m + 1), dtype=np.uint8)
    score[0, 0] = 0.0
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + skip_moving_cost
        action[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            position_error = abs(float(fixed_n[i - 1] - moving_n[j - 1]))
            match_cost = min((position_error / position_scale) ** 2, 3.0)
            match = score[i - 1, j - 1] + match_cost
            skip_moving = score[i, j - 1] + skip_moving_cost
            if match <= skip_moving:
                score[i, j] = match
                action[i, j] = 1
            else:
                score[i, j] = skip_moving
                action[i, j] = 3
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        current = int(action[i, j])
        if current == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif current == 3:
            j -= 1
        else:
            raise RuntimeError("Invalid dynamic-programming backtrack state")
    pairs.reverse()
    if len(pairs) != min(n, m):
        raise RuntimeError(
            f"Incomplete monotonic ridge matching: {len(pairs)} != {min(n, m)}"
        )
    return pairs


def generated_ridges(
    generated: np.ndarray,
    roi: base.Roi,
    y_range: tuple[int, int],
) -> tuple[np.ndarray, list[np.ndarray], float]:
    centers, pitch, _ = length.detect_centers(generated, roi)
    paths = [
        length.track_ridge(generated, float(center), float(pitch), y_range[0], y_range[1])
        for center in centers
    ]
    return centers.astype(np.float64), paths, float(pitch)


def matched_paths(
    generated: np.ndarray,
    layers: list[dict],
    left_roi: base.Roi,
    right_roi: base.Roi,
    top_range: tuple[int, int],
    bottom_range: tuple[int, int],
) -> tuple[dict[str, list[dict]], dict]:
    result: dict[str, list[dict]] = {}
    diagnostics: dict[str, dict] = {}
    for side, roi in (("left", left_roi), ("right", right_roi)):
        fixed_rows = sorted(
            (row for row in layers if row["side"] == side),
            key=lambda row: float(row["center_x_px"]),
        )
        fixed_centers = np.asarray(
            [float(row["center_x_px"]) for row in fixed_rows], dtype=np.float64
        )
        moving_centers, moving_paths, moving_pitch = generated_ridges(
            generated, roi, (top_range[0], bottom_range[1])
        )
        pairs = monotonic_match(fixed_centers, moving_centers)
        current: list[dict] = []
        endpoint_displacements: list[float] = []
        for fixed_index, moving_index in pairs:
            fixed_row = fixed_rows[fixed_index]
            hint = length.EndpointMeasurement(
                top=float(fixed_row["top_y_px"]),
                bottom=float(fixed_row["bottom_y_px"]),
                length=float(fixed_row["length_px"]),
                top_spread=0.0,
                bottom_spread=0.0,
                uncertainty=float(fixed_row["endpoint_uncertainty_px"]),
                top_snr=float(fixed_row["top_snr"]),
                bottom_snr=float(fixed_row["bottom_snr"]),
            )
            moving_endpoint = length.measure_endpoints(
                generated,
                float(moving_centers[moving_index]),
                moving_pitch,
                top_range,
                bottom_range,
                hint=hint,
                association_radius=14,
                path_x=np.asarray(moving_paths[moving_index], dtype=np.float32),
            )
            endpoint_displacements.extend((
                float(moving_endpoint.top - hint.top),
                float(moving_endpoint.bottom - hint.bottom),
            ))
            current.append({
                "layer_id": fixed_rows[fixed_index]["layer_id"],
                "fixed_index": fixed_index,
                "moving_index": moving_index,
                "fixed_center_x_px": float(fixed_centers[fixed_index]),
                "moving_center_x_px": float(moving_centers[moving_index]),
                "fixed_path_x": np.asarray(fixed_rows[fixed_index]["path_x"], dtype=np.float32),
                "moving_path_x": np.asarray(moving_paths[moving_index], dtype=np.float32),
                "fixed_top_y_px": float(hint.top),
                "fixed_bottom_y_px": float(hint.bottom),
                "moving_top_y_px": float(moving_endpoint.top),
                "moving_bottom_y_px": float(moving_endpoint.bottom),
            })
        result[side] = current
        center_displacements = np.asarray(
            [item["moving_center_x_px"] - item["fixed_center_x_px"] for item in current]
        )
        diagnostics[side] = {
            "fixed_ridge_count": int(fixed_centers.size),
            "generated_ridge_count": int(moving_centers.size),
            "matched_ridge_count": len(current),
            "generated_detected_pitch_px": moving_pitch,
            "moving_minus_fixed_center_median_px": float(np.median(center_displacements)),
            "moving_minus_fixed_center_p95_abs_px": float(
                np.percentile(np.abs(center_displacements), 95.0)
            ),
            "moving_minus_fixed_endpoint_median_px": float(
                np.median(np.asarray(endpoint_displacements))
            ),
            "moving_minus_fixed_endpoint_p95_abs_px": float(
                np.percentile(np.abs(np.asarray(endpoint_displacements)), 95.0)
            ),
            "skipped_fixed_indices": sorted(
                set(range(fixed_centers.size)) - {item["fixed_index"] for item in current}
            ),
            "skipped_generated_indices": sorted(
                set(range(moving_centers.size)) - {item["moving_index"] for item in current}
            ),
        }
    return result, diagnostics


def row_source_coordinates(
    fixed_positions: np.ndarray,
    moving_positions: np.ndarray,
    x_coordinates: np.ndarray,
    maximum_displacement: float,
) -> np.ndarray:
    """Map output/fixed x coordinates to source/moving coordinates."""
    fixed_positions = np.asarray(fixed_positions, dtype=np.float64)
    moving_positions = np.asarray(moving_positions, dtype=np.float64)
    if fixed_positions.size != moving_positions.size or fixed_positions.size < 2:
        raise ValueError("Fixed and moving control points must have equal length >= 2")
    fixed_positions = np.maximum.accumulate(fixed_positions)
    moving_positions = np.maximum.accumulate(moving_positions)
    keep = np.concatenate(([True], np.diff(fixed_positions) > 0.25))
    fixed_positions = fixed_positions[keep]
    moving_positions = moving_positions[keep]
    if fixed_positions.size < 2:
        return x_coordinates.copy()
    mapped = np.interp(
        x_coordinates,
        fixed_positions,
        moving_positions,
        left=float(moving_positions[0] + x_coordinates[0] - fixed_positions[0]),
        right=float(moving_positions[-1] + x_coordinates[-1] - fixed_positions[-1]),
    )
    displacement = np.clip(
        mapped - x_coordinates, -float(maximum_displacement), float(maximum_displacement)
    )
    return (x_coordinates + displacement).astype(np.float32)


def vertical_source_coordinate(
    target_y: float,
    fixed_top: float,
    fixed_bottom: float,
    moving_top: float,
    moving_bottom: float,
    maximum_displacement: float,
    exterior_fade: float = 12.0,
) -> float:
    """Map one target y to its generated y with a continuous endpoint fade."""
    if fixed_bottom <= fixed_top + 1.0 or moving_bottom <= moving_top + 1.0:
        return float(target_y)
    if fixed_top <= target_y <= fixed_bottom:
        fraction = (target_y - fixed_top) / (fixed_bottom - fixed_top)
        mapped = moving_top + fraction * (moving_bottom - moving_top)
    elif target_y < fixed_top:
        distance = fixed_top - target_y
        gate = float(np.clip(1.0 - distance / max(exterior_fade, 1e-8), 0.0, 1.0))
        gate = gate * gate * (3.0 - 2.0 * gate)
        mapped = target_y + gate * (moving_top - fixed_top)
    else:
        distance = target_y - fixed_bottom
        gate = float(np.clip(1.0 - distance / max(exterior_fade, 1e-8), 0.0, 1.0))
        gate = gate * gate * (3.0 - 2.0 * gate)
        mapped = target_y + gate * (moving_bottom - fixed_bottom)
    return float(
        target_y
        + np.clip(
            mapped - target_y,
            -float(maximum_displacement),
            float(maximum_displacement),
        )
    )


def apply_native_warp(
    generated: np.ndarray,
    fields: dict,
    matches: dict[str, list[dict]],
    strength: float,
    maximum_displacement: float = 5.0,
    maximum_vertical_displacement: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Move generated ridges/endpoints with one continuous native-canvas warp."""
    output = generated.copy()
    height, width = generated.shape
    axis = np.arange(width, dtype=np.float32)
    x_displacement_samples: list[float] = []
    y_displacement_samples: list[float] = []
    for side in ("left", "right"):
        cell = np.asarray(fields["side_cells"][side], dtype=np.float32)
        current = matches[side]
        for y in range(height):
            active = np.flatnonzero(cell[y] > 1e-5)
            if active.size == 0:
                continue
            x0 = max(0, int(active[0]) - 2)
            x1 = min(width, int(active[-1]) + 3)
            fixed = np.asarray([item["fixed_path_x"][y] for item in current], dtype=np.float64)
            moving = np.asarray([item["moving_path_x"][y] for item in current], dtype=np.float64)
            moving_y = np.asarray([
                vertical_source_coordinate(
                    float(y),
                    float(item["fixed_top_y_px"]),
                    float(item["fixed_bottom_y_px"]),
                    float(item["moving_top_y_px"]),
                    float(item["moving_bottom_y_px"]),
                    maximum_displacement=float(maximum_vertical_displacement),
                )
                for item in current
            ], dtype=np.float64)
            # Identity anchors outside the stack keep the surrounding canvas fixed.
            fixed = np.concatenate(([x0 - 10.0], fixed, [x1 + 10.0]))
            moving = np.concatenate(([x0 - 10.0], moving, [x1 + 10.0]))
            moving_y = np.concatenate(([float(y)], moving_y, [float(y)]))
            local_axis = axis[x0:x1]
            full_source = row_source_coordinates(
                fixed, moving, local_axis, maximum_displacement=maximum_displacement
            )
            source_coordinates = local_axis + float(strength) * (full_source - local_axis)
            full_source_y = np.interp(local_axis, fixed, moving_y).astype(np.float32)
            source_y = float(y) + float(strength) * (full_source_y - float(y))
            warped = map_coordinates(
                generated,
                np.stack((source_y, source_coordinates), axis=0),
                order=1,
                mode="nearest",
                prefilter=False,
            ).astype(np.float32)
            blend = cell[y, x0:x1]
            output[y, x0:x1] = (
                (1.0 - blend) * generated[y, x0:x1] + blend * warped
            )
            x_displacement_samples.extend(
                np.abs(source_coordinates[blend >= 0.50] - local_axis[blend >= 0.50]).tolist()
            )
            y_displacement_samples.extend(
                np.abs(source_y[blend >= 0.50] - float(y)).tolist()
            )
    writable = np.asarray(fields["stack"]) > 1e-5
    output[~writable] = generated[~writable]
    x_displacements = np.asarray(x_displacement_samples, dtype=np.float64)
    y_displacements = np.asarray(y_displacement_samples, dtype=np.float64)
    return np.clip(output, 0.0, 1.0).astype(np.float32), {
        "strength": float(strength),
        "maximum_full_displacement_px": float(maximum_displacement),
        "applied_x_displacement_median_px": float(np.median(x_displacements)),
        "applied_x_displacement_p95_px": float(np.percentile(x_displacements, 95.0)),
        "applied_x_displacement_max_px": float(np.max(x_displacements)),
        "maximum_full_vertical_displacement_px": float(maximum_vertical_displacement),
        "applied_y_displacement_median_px": float(np.median(y_displacements)),
        "applied_y_displacement_p95_px": float(np.percentile(y_displacements, 95.0)),
        "applied_y_displacement_max_px": float(np.max(y_displacements)),
        "guide_or_raw_pixel_writeback": False,
        "vertical_coordinate_changed": True,
        "endpoint_mapping_enabled": True,
        "canvas_resized": False,
    }


def choose_warp(
    strengths: list[float],
    guide: np.ndarray,
    generated: np.ndarray,
    layers: list[dict],
    fields: dict,
    matches: dict[str, list[dict]],
    minimum_ssim: float,
    maximum_displacement: float,
    maximum_vertical_displacement: float,
) -> tuple[np.ndarray, dict, list[dict]]:
    baseline = audit.fast_alignment_metrics(guide, generated, generated, layers, fields)
    baseline_mid = float(baseline["guide_mid_frequency_correlation"] or -1.0)
    baseline_median = float(baseline["ridge_center_absolute_offset_median_px"])
    baseline_p95 = float(baseline["ridge_center_absolute_offset_p95_px"])
    records: list[dict] = []
    candidates: list[tuple[np.ndarray, dict]] = []
    for strength in strengths:
        candidate, warp = apply_native_warp(
            generated,
            fields,
            matches,
            strength=strength,
            maximum_displacement=maximum_displacement,
            maximum_vertical_displacement=maximum_vertical_displacement,
        )
        candidate = base.to_uint16(candidate).astype(np.float32) / 65535.0
        metrics = audit.fast_alignment_metrics(guide, candidate, generated, layers, fields)
        mid = float(metrics["guide_mid_frequency_correlation"] or -1.0)
        ridge_median = float(metrics["ridge_center_absolute_offset_median_px"])
        ridge_p95 = float(metrics["ridge_center_absolute_offset_p95_px"])
        guardrail = bool(
            metrics["global_ssim_to_accepted_generation"] >= minimum_ssim
            and metrics["outside_writable_max_abs_change"] <= 0.5 / 65535.0
            # The accepted generator renders the plate phase differently from
            # the guide, so signed band-pass correlation can become slightly
            # more negative even when measured ridge positions improve.  Keep
            # it as a bounded appearance guard rather than the primary target.
            and mid >= baseline_mid - 0.05
            and ridge_median <= baseline_median
            and ridge_p95 <= baseline_p95
        )
        score = float(
            0.45 * mid
            - 0.35 * ridge_median / 4.5
            - 0.20 * ridge_p95 / 5.5
        )
        record = {
            "warp": warp,
            "alignment": metrics,
            "guardrail_pass": guardrail,
            "selection_score": score,
        }
        records.append(record)
        candidates.append((candidate, record))
    eligible = [candidate for candidate in candidates if candidate[1]["guardrail_pass"]]
    if not eligible:
        compact = [
            {
                "strength": row["warp"]["strength"],
                "ssim": row["alignment"]["global_ssim_to_accepted_generation"],
                "mid": row["alignment"]["guide_mid_frequency_correlation"],
                "ridge_median": row["alignment"]["ridge_center_absolute_offset_median_px"],
                "ridge_p95": row["alignment"]["ridge_center_absolute_offset_p95_px"],
            }
            for row in records
        ]
        raise RuntimeError(
            "No native-coordinate warp candidate passed all guardrails: "
            + json.dumps(compact)
        )
    selected, selected_record = max(eligible, key=lambda item: item[1]["selection_score"])
    return selected, {"baseline": baseline, "selected": selected_record}, records


def parse_strengths(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("strengths must be comma-separated values in (0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Align generated left/right ridges to measured guide paths by a bounded, "
            "native-canvas two-axis soft warp."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--left-roi", type=base.parse_roi, default=base.Roi(720, 1060, 370, 970))
    parser.add_argument("--right-roi", type=base.parse_roi, default=base.Roi(720, 1060, 1220, 1830))
    parser.add_argument("--top-range", type=base.parse_range, default=(600, 790))
    parser.add_argument("--bottom-range", type=base.parse_range, default=(1010, 1240))
    parser.add_argument(
        "--central-roi",
        type=base.parse_roi,
        default=base.Roi(700, 1140, 985, 1205),
        help="y0,y1,x0,x1 region that must remain bit-identical to the generation",
    )
    parser.add_argument("--strengths", type=parse_strengths, default=parse_strengths("0.12,0.16,0.20,0.24"))
    parser.add_argument("--maximum-displacement", type=float, default=5.0)
    parser.add_argument("--maximum-vertical-displacement", type=float, default=5.0)
    parser.add_argument("--minimum-global-ssim", type=float, default=0.965)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    source, source_info = base.load_gray(args.source)
    guide, guide_info = base.load_gray(args.guide)
    generated, generated_info = base.load_gray(args.generated)
    audit.require_native_coordinates(source, guide, generated)
    left_roi = args.left_roi.clamp(source.shape)
    right_roi = args.right_roi.clamp(source.shape)
    layers = constraint.measure_layers(
        source,
        left_roi,
        right_roi,
        args.top_range,
        args.bottom_range,
        guide=guide,
    )
    fields = audit.build_constraint_fields(source.shape, layers)
    matches, match_diagnostics = matched_paths(
        generated,
        layers,
        left_roi,
        right_roi,
        args.top_range,
        args.bottom_range,
    )
    refined, selection, candidates = choose_warp(
        args.strengths,
        guide,
        generated,
        layers,
        fields,
        matches,
        minimum_ssim=float(args.minimum_global_ssim),
        maximum_displacement=float(args.maximum_displacement),
        maximum_vertical_displacement=float(args.maximum_vertical_displacement),
    )
    central = args.central_roi.clamp(source.shape)
    # The central component is a hard exclusion zone even if a future geometry
    # profile accidentally lets a feathered stack mask overlap it.
    refined[central.slices()] = generated[central.slices()]
    central_change = float(np.max(np.abs(refined[central.slices()] - generated[central.slices()])))
    if central_change > 0.5 / 65535.0:
        raise RuntimeError(f"Central-block lock failed: {central_change}")

    height, width = source.shape
    output_png = args.outdir / f"STRUCTURE_WARP_GUIDED_generated_{width}x{height}.png"
    output_tif = args.outdir / f"STRUCTURE_WARP_GUIDED_generated_{width}x{height}_16bit.tif"
    Image.fromarray(np.rint(refined * 255.0).astype(np.uint8), mode="L").save(output_png)
    imwrite(
        output_tif,
        base.to_uint16(refined),
        photometric="minisblack",
        description=(
            "MEASUREMENT_ASSIST: source-sized accepted generation with bounded 2-D "
            "structure-coordinate guidance; no guide/raw pixel writeback."
        ),
    )
    Image.fromarray(
        np.rint(np.clip(np.asarray(fields["stack"]), 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    ).save(args.outdir / "AUDIT_native_warp_write_mask.png")
    audit.save_overlay(args.outdir / "AUDIT_measured_structure_overlay.png", generated, layers)
    audit.save_comparison(
        args.outdir / "STRUCTURE_WARP_GUIDED_comparison.png", generated, guide, refined
    )

    baseline_geometry = audit.endpoint_and_width_audit(
        generated, layers, args.top_range, args.bottom_range
    )
    refined_geometry = audit.endpoint_and_width_audit(
        refined, layers, args.top_range, args.bottom_range
    )
    manifest = {
        "completed": True,
        "release": "native-coordinate-soft-structure-warp",
        "baseline_contract": {
            "input": str(args.generated),
            "description": (
                f"accepted {width}x{height} image immediately after source-size lock"
            ),
            "later_optimization_used": False,
            "canvas_resize_used": False,
            "hard_geometry_redraw_used": False,
        },
        "inputs": {
            "source": source_info,
            "blind_denoised_guide": guide_info,
            "accepted_generation": generated_info,
        },
        "dimensions": {"width": source.shape[1], "height": source.shape[0]},
        "method": {
            "operation": "monotonic ridge/endpoint matching plus bounded local 2-D warp",
            "guide_or_raw_pixel_writeback": False,
            "central_block_locked": True,
            "central_lock_roi": central.__dict__,
            "background_locked": True,
            "vertical_coordinates_locked": False,
            "maximum_full_displacement_px": float(args.maximum_displacement),
            "maximum_full_vertical_displacement_px": float(
                args.maximum_vertical_displacement
            ),
        },
        "ridge_matching": match_diagnostics,
        "selection": selection,
        "candidate_search": candidates,
        "geometry_audit": {
            "accepted_generation": baseline_geometry["summary"],
            "structure_warp_guided": refined_geometry["summary"],
        },
        "central_block_max_abs_change": central_change,
        "outputs": {
            "png": str(output_png),
            "tif_16bit": str(output_tif),
            "comparison": str(args.outdir / "STRUCTURE_WARP_GUIDED_comparison.png"),
            "constraint_overlay": str(args.outdir / "AUDIT_measured_structure_overlay.png"),
            "write_mask": str(args.outdir / "AUDIT_native_warp_write_mask.png"),
        },
        "warning": (
            "This is a visual/measurement-assist image, not calibrated ground truth. "
            "Report quantitative dimensions from the raw/guide constraint table."
        ),
    }
    (args.outdir / "native_structure_warp_metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "completed": True,
                "output": str(output_png),
                "selected_strength": selection["selected"]["warp"]["strength"],
                "matching": match_diagnostics,
                "baseline_alignment": selection["baseline"],
                "refined_alignment": selection["selected"]["alignment"],
                "baseline_geometry": baseline_geometry["summary"],
                "refined_geometry": refined_geometry["summary"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
