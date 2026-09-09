#!/usr/bin/env python3
"""Run the frozen source-sized generative CT enhancement workflow.

The generator itself is intentionally an external boundary.  This runner
accepts its candidate, locks it to the source canvas immediately, obtains or
builds a blind-denoised measurement guide, and finally applies the bounded
native-coordinate structure warp that produced the accepted reference result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tifffile import imwrite

import pipeline as base
from size_lock import lock_generated_to_source


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
DEFAULT_CONFIG = REPO_DIR / "config" / "reference_geometry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Only geometry-config schema_version=1 is supported")
    reference = config.get("reference_size", {})
    if int(reference.get("width", 0)) <= 0 or int(reference.get("height", 0)) <= 0:
        raise ValueError("geometry config requires a positive reference_size")
    return config


def scaled_roi(values: list[int], shape: tuple[int, int], config: dict[str, Any]) -> list[int]:
    if len(values) != 4:
        raise ValueError(f"ROI must contain y0,y1,x0,x1, got {values}")
    height, width = shape
    reference = config["reference_size"]
    if config.get("scale_coordinates_to_input", True):
        sy = height / float(reference["height"])
        sx = width / float(reference["width"])
    else:
        sy = sx = 1.0
    y0, y1, x0, x1 = values
    return [
        int(round(y0 * sy)),
        int(round(y1 * sy)),
        int(round(x0 * sx)),
        int(round(x1 * sx)),
    ]


def coordinate_scales(shape: tuple[int, int], config: dict[str, Any]) -> tuple[float, float]:
    """Return native/reference scale factors as (sy, sx)."""
    if not config.get("scale_coordinates_to_input", True):
        return 1.0, 1.0
    height, width = shape
    reference = config["reference_size"]
    return height / float(reference["height"]), width / float(reference["width"])


def scaled_range(values: list[int], height: int, config: dict[str, Any]) -> list[int]:
    if len(values) != 2:
        raise ValueError(f"Range must contain start,stop, got {values}")
    reference_height = float(config["reference_size"]["height"])
    scale = height / reference_height if config.get("scale_coordinates_to_input", True) else 1.0
    return [int(round(values[0] * scale)), int(round(values[1] * scale))]


def csv(values: list[int] | list[float]) -> str:
    return ",".join(str(value) for value in values)


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def prepare_outdir(outdir: Path, overwrite: bool) -> None:
    managed = (
        "01_source_sized_generation",
        "02_blind_guide",
        "03_structure_guidance",
    )
    final_patterns = (
        "FINAL_enhanced_*.png",
        "FINAL_enhanced_*_16bit.tif",
        "RUN_MANIFEST.json",
    )
    existing = [path for path in outdir.iterdir()] if outdir.exists() else []
    meaningful = [path for path in existing if path.name != ".gitkeep"]
    if meaningful and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {outdir}. Use --overwrite to replace "
            "only outputs managed by this pipeline."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in managed:
            path = outdir / name
            if path.exists():
                shutil.rmtree(path)
        for pattern in final_patterns:
            for path in outdir.glob(pattern):
                if path.is_file():
                    path.unlink()


def copy_guide(guide_path: Path, source_shape: tuple[int, int], outdir: Path) -> Path:
    guide, _ = base.load_gray(guide_path)
    if guide.shape != source_shape:
        raise ValueError(
            f"Guide dimensions {guide.shape} do not match source {source_shape}; "
            "measurement guides are never resized."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "MEASUREMENT_blind_guide_16bit.tif"
    imwrite(
        output,
        base.to_uint16(guide),
        photometric="minisblack",
        description="MEASUREMENT_ASSIST: supplied blind-denoised guide; native canvas retained.",
    )
    return output


def build_blind_guide(
    source: Path,
    generated: Path,
    source_shape: tuple[int, int],
    outdir: Path,
    config: dict[str, Any],
    iterations: int,
    device: str,
    seed: int,
) -> Path:
    blind = config["blind_denoise"]
    command = [
        sys.executable,
        str(APP_DIR / "sota_geometry_blind.py"),
        "--source", str(source),
        "--generative-prior", str(generated),
        "--outdir", str(outdir),
        "--target-roi", csv(scaled_roi(blind["target_roi"], source_shape, config)),
        "--left-roi", csv(scaled_roi(blind["left_roi"], source_shape, config)),
        "--right-roi", csv(scaled_roi(blind["right_roi"], source_shape, config)),
        "--left-body-roi", csv(scaled_roi(blind["left_body_roi"], source_shape, config)),
        "--right-body-roi", csv(scaled_roi(blind["right_body_roi"], source_shape, config)),
        "--top-range", csv(scaled_range(blind["top_range"], source_shape[0], config)),
        "--bottom-range", csv(scaled_range(blind["bottom_range"], source_shape[0], config)),
        "--iterations", str(iterations),
        "--device", device,
        "--seed", str(seed),
    ]
    run_checked(command)
    output = outdir / "MEASUREMENT_sota_geometry_blind_16bit.tif"
    if not output.is_file():
        raise RuntimeError(f"Blind-denoise stage did not create {output}")
    return output


def run_structure_guidance(
    source: Path,
    guide: Path,
    generated: Path,
    source_shape: tuple[int, int],
    outdir: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, Path]:
    geometry = config["structure_guidance"]
    width, height = source_shape[1], source_shape[0]
    sy, sx = coordinate_scales(source_shape, config)
    command = [
        sys.executable,
        str(APP_DIR / "native_structure_warp.py"),
        "--source", str(source),
        "--guide", str(guide),
        "--generated", str(generated),
        "--outdir", str(outdir),
        "--left-roi", csv(scaled_roi(geometry["left_roi"], source_shape, config)),
        "--right-roi", csv(scaled_roi(geometry["right_roi"], source_shape, config)),
        "--top-range", csv(scaled_range(geometry["top_range"], height, config)),
        "--bottom-range", csv(scaled_range(geometry["bottom_range"], height, config)),
        "--central-roi", csv(scaled_roi(geometry["central_lock_roi"], source_shape, config)),
        "--strengths", csv(geometry["strengths"]),
        "--maximum-displacement", str(geometry["maximum_horizontal_displacement_px"] * sx),
        "--maximum-vertical-displacement", str(geometry["maximum_vertical_displacement_px"] * sy),
        "--minimum-global-ssim", str(geometry["minimum_global_ssim"]),
    ]
    run_checked(command)
    output_png = outdir / f"STRUCTURE_WARP_GUIDED_generated_{width}x{height}.png"
    output_tif = outdir / f"STRUCTURE_WARP_GUIDED_generated_{width}x{height}_16bit.tif"
    metrics = outdir / "native_structure_warp_metrics.json"
    for path in (output_png, output_tif, metrics):
        if not path.is_file():
            raise RuntimeError(f"Structure-guidance stage did not create {path}")
    return output_png, output_tif, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Complete reusable CT enhancement flow: immediate source-size lock, "
            "blind guide, and bounded native-coordinate structure guidance."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument(
        "--guide",
        type=Path,
        help=(
            "Optional precomputed blind-denoised guide. If omitted, the self-supervised "
            "blind model is trained for this image."
        ),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--blind-iterations", type=int, default=600)
    parser.add_argument("--blind-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--max-generated-aspect-ratio-error",
        type=float,
        default=0.01,
        help="Reject size locking when relative aspect-ratio mismatch exceeds this value",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for label, path in (("source", args.source), ("generated", args.generated)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if args.guide is not None and not args.guide.is_file():
        raise FileNotFoundError(f"guide file not found: {args.guide}")
    if args.blind_iterations <= 0:
        raise ValueError("--blind-iterations must be positive")

    config = load_config(args.geometry_config)
    source, source_info = base.load_gray(args.source)
    source_shape = source.shape
    prepare_outdir(args.outdir, args.overwrite)

    size_dir = args.outdir / "01_source_sized_generation"
    generated_png, generated_tif, size_manifest = lock_generated_to_source(
        args.source,
        args.generated,
        size_dir,
        max_aspect_ratio_relative_error=args.max_generated_aspect_ratio_error,
    )

    guide_dir = args.outdir / "02_blind_guide"
    if args.guide is not None:
        guide_path = copy_guide(args.guide, source_shape, guide_dir)
        guide_origin = "supplied_precomputed_blind_guide"
    else:
        guide_dir.mkdir(parents=True, exist_ok=True)
        guide_path = build_blind_guide(
            args.source,
            generated_png,
            source_shape,
            guide_dir,
            config,
            args.blind_iterations,
            args.blind_device,
            args.seed,
        )
        guide_origin = "self_supervised_blind_denoise_trained_for_this_image"

    structure_dir = args.outdir / "03_structure_guidance"
    structure_dir.mkdir(parents=True, exist_ok=True)
    guided_png, guided_tif, metrics_path = run_structure_guidance(
        args.source,
        guide_path,
        generated_png,
        source_shape,
        structure_dir,
        config,
    )

    height, width = source_shape
    final_png = args.outdir / f"FINAL_enhanced_{width}x{height}.png"
    final_tif = args.outdir / f"FINAL_enhanced_{width}x{height}_16bit.tif"
    shutil.copy2(guided_png, final_png)
    shutil.copy2(guided_tif, final_tif)
    structure_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    manifest = {
        "completed": True,
        "release": "source-sized-generative-native-structure-guidance-v1",
        "source": source_info,
        "dimensions": {"width": width, "height": height},
        "stage_order": [
            "external generative candidate",
            "immediate source-size lock",
            "blind-denoised measurement guide (supplied or trained)",
            "measurement of ridge paths, endpoints, layer width, and inter-layer spacing",
            "bounded continuous two-axis structure guidance on the native canvas",
        ],
        "generator_boundary": {
            "external": True,
            "candidate": str(args.generated),
            "reason": "No generator weights or remote API credentials are embedded in this repository.",
        },
        "geometry_config": str(args.geometry_config),
        "guide_origin": guide_origin,
        "size_lock": size_manifest,
        "structure_selection": structure_metrics["selection"],
        "geometry_audit": structure_metrics["geometry_audit"],
        "invariants": {
            "output_matches_source_dimensions": True,
            "no_crop_or_padding": True,
            "central_block_locked": True,
            "pixels_outside_lamella_masks_locked": True,
            "guide_or_raw_intensity_writeback": False,
            "hard_structure_redraw": False,
        },
        "files": {
            "source_sized_generation_png": str(generated_png),
            "source_sized_generation_tif_16bit": str(generated_tif),
            "blind_guide_tif_16bit": str(guide_path),
            "final_png": str(final_png),
            "final_tif_16bit": str(final_tif),
            "structure_metrics": str(metrics_path),
        },
        "sha256": {
            "source": sha256(args.source),
            "generated_input": sha256(args.generated),
            "supplied_guide_input": sha256(args.guide) if args.guide is not None else None,
            "canonical_blind_guide": sha256(guide_path),
            "final_png": sha256(final_png),
            "final_tif_16bit": sha256(final_tif),
        },
        "measurement_warning": (
            "The enhanced image is measurement-assist output, not calibrated ground truth. "
            "Archive the source image, guide, geometry config, and manifests with every run."
        ),
    }
    manifest_path = args.outdir / "RUN_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "completed": True,
                "final_png": str(final_png),
                "final_tif_16bit": str(final_tif),
                "manifest": str(manifest_path),
                "dimensions": manifest["dimensions"],
                "selected_strength": structure_metrics["selection"]["selected"]["warp"]["strength"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
