#!/usr/bin/env python3
"""Run the deterministic visual-only post-processing stage independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import imwrite

import pipeline


def resize_generated_to_source_dimensions(
    generated_path: Path,
    source_path: Path,
    outdir: Path,
) -> tuple[Path, dict]:
    """Resize the generator output before any enhancement or projection."""
    source, source_info = pipeline.load_gray(source_path)
    target_height, target_width = source.shape
    outdir.mkdir(parents=True, exist_ok=True)
    with Image.open(generated_path) as image:
        gray = image.convert("L")
        input_width, input_height = gray.size
        if gray.size != (target_width, target_height):
            gray = gray.resize((target_width, target_height), Image.Resampling.LANCZOS)
        output_path = outdir / (
            f"GENERATIVE_denoised_guide_shape_conditioned_{target_width}x{target_height}.png"
        )
        gray.save(output_path)
        resized = np.asarray(gray, dtype=np.float32) / 255.0
    imwrite(
        outdir / (
            f"GENERATIVE_denoised_guide_shape_conditioned_{target_width}x{target_height}_16bit.tif"
        ),
        pipeline.to_uint16(resized),
        photometric="minisblack",
        description=("VISUAL_ONLY: generator output resized to source dimensions before "
                     "post-processing and geometry projection."),
    )
    audit = {
        "completed": True,
        "stage_order": "generation -> source-size resize -> post-processing -> geometry projection",
        "source": source_info,
        "generated_input": str(generated_path),
        "generated_input_width": input_width,
        "generated_input_height": input_height,
        "target_width": target_width,
        "target_height": target_height,
        "output_width": int(resized.shape[1]),
        "output_height": int(resized.shape[0]),
        "resampling": (
            "PIL Lanczos" if (input_width, input_height) != (target_width, target_height)
            else "none"
        ),
        "strict_dimension_match": bool(resized.shape == source.shape),
        "postprocessing_applied_before_resize": False,
        "warning": "VISUAL_ONLY: size equality does not make generated pixels metrology-safe.",
    }
    (outdir / "generation_preprocess_size_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not audit["strict_dimension_match"]:
        raise RuntimeError("Generated-image resize failed to match the source dimensions")
    return output_path, audit


def lock_to_source_dimensions(
    processed_path: Path,
    source_path: Path,
    outdir: Path,
) -> tuple[Path, dict]:
    """Save a deterministic source-sized copy without modifying geometry afterward."""
    source, source_info = pipeline.load_gray(source_path)
    target_height, target_width = source.shape
    with Image.open(processed_path) as image:
        gray = image.convert("L")
        input_width, input_height = gray.size
        if gray.size != (target_width, target_height):
            gray = gray.resize((target_width, target_height), Image.Resampling.LANCZOS)
        output_path = outdir / "GENERATIVE_visual_only_postprocessed_size_locked.png"
        gray.save(output_path)
        locked = np.asarray(gray, dtype=np.float32) / 255.0
    imwrite(
        outdir / "GENERATIVE_visual_only_postprocessed_size_locked_16bit.tif",
        pipeline.to_uint16(locked),
        photometric="minisblack",
        description=("VISUAL_ONLY: deterministic size lock to source dimensions; "
                     "generative content is not measurement truth."),
    )
    audit = {
        "completed": True,
        "source": source_info,
        "processed_input": str(processed_path),
        "processed_input_width": input_width,
        "processed_input_height": input_height,
        "target_width": target_width,
        "target_height": target_height,
        "output_width": int(locked.shape[1]),
        "output_height": int(locked.shape[0]),
        "resampling": (
            "PIL Lanczos" if (input_width, input_height) != (target_width, target_height)
            else "none"
        ),
        "aspect_ratio_input": float(input_width / input_height),
        "aspect_ratio_target": float(target_width / target_height),
        "strict_dimension_match": bool(locked.shape == source.shape),
        "warning": "VISUAL_ONLY: size equality does not make generated pixels metrology-safe.",
    }
    (outdir / "size_lock_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not audit["strict_dimension_match"]:
        raise RuntimeError("Size lock failed to match the source dimensions")
    return output_path, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument(
        "--match-source", type=Path,
        help="Also save a deterministic post-processed image matching this source size.",
    )
    ap.add_argument(
        "--resize-before-postprocess", action="store_true",
        help=("Resize the generated candidate to --match-source dimensions before "
              "post-processing. This is the recommended geometry-coordinate workflow."),
    )
    ap.add_argument(
        "--resize-only", action="store_true",
        help=("Stop after the source-dimension resize. Intended for a registration "
              "stage that must run before visual post-processing."),
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    generated_input = args.generated
    if args.resize_before_postprocess:
        if not args.match_source:
            ap.error("--resize-before-postprocess requires --match-source")
        generated_input, _ = resize_generated_to_source_dimensions(
            args.generated, args.match_source, args.outdir
        )
    if args.resize_only:
        if not args.resize_before_postprocess:
            ap.error("--resize-only requires --resize-before-postprocess")
        print(generated_input, flush=True)
        return
    pipeline.postprocess_visual(generated_input, args.outdir)
    output = args.outdir / "GENERATIVE_visual_only_postprocessed.png"
    if args.match_source:
        output, _ = lock_to_source_dimensions(output, args.match_source, args.outdir)
    print(output, flush=True)


if __name__ == "__main__":
    main()
