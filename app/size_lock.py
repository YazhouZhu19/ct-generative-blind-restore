#!/usr/bin/env python3
"""Lock an external generator candidate to the native source-image canvas.

This is deliberately the first operation after generation.  No enhancement,
warping, geometry projection, padding, or cropping is performed beforehand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import imwrite

import pipeline as base


def lock_generated_to_source(
    source_path: Path,
    generated_path: Path,
    outdir: Path,
    max_aspect_ratio_relative_error: float = 0.01,
) -> tuple[Path, Path, dict]:
    """Save an 8-bit PNG and 16-bit TIFF on the exact source canvas."""
    if max_aspect_ratio_relative_error < 0.0:
        raise ValueError("max_aspect_ratio_relative_error must be non-negative")
    source, source_info = base.load_gray(source_path)
    target_height, target_width = source.shape
    outdir.mkdir(parents=True, exist_ok=True)

    with Image.open(generated_path) as image:
        gray = image.convert("L")
        input_width, input_height = gray.size
        input_aspect_ratio = input_width / float(input_height)
        target_aspect_ratio = target_width / float(target_height)
        aspect_ratio_relative_error = abs(
            input_aspect_ratio / target_aspect_ratio - 1.0
        )
        if aspect_ratio_relative_error > max_aspect_ratio_relative_error:
            raise ValueError(
                "Generated/source aspect-ratio mismatch is too large for safe size "
                f"locking: relative_error={aspect_ratio_relative_error:.4%}, "
                f"limit={max_aspect_ratio_relative_error:.4%}"
            )
        resized = gray.size != (target_width, target_height)
        if resized:
            gray = gray.resize(
                (target_width, target_height),
                resample=Image.Resampling.LANCZOS,
            )
        pixels = np.asarray(gray, dtype=np.uint8)

    if pixels.shape != source.shape:
        raise RuntimeError(
            f"Size lock failed: generated={pixels.shape}, source={source.shape}"
        )

    stem = f"GENERATION_source_sized_{target_width}x{target_height}"
    output_png = outdir / f"{stem}.png"
    output_tif = outdir / f"{stem}_16bit.tif"
    Image.fromarray(pixels, mode="L").save(output_png)
    imwrite(
        output_tif,
        pixels.astype(np.uint16) * np.uint16(257),
        photometric="minisblack",
        description=(
            "VISUAL_ONLY: external generator output immediately locked to the "
            "source dimensions; no geometry correction or enhancement applied."
        ),
    )

    manifest = {
        "completed": True,
        "operation_order": "external generation -> immediate source-size lock",
        "source": source_info,
        "generated_input": str(generated_path),
        "generated_input_dimensions": {
            "width": input_width,
            "height": input_height,
        },
        "output_dimensions": {
            "width": target_width,
            "height": target_height,
        },
        "resized": resized,
        "resampling": "PIL Lanczos" if resized else "none",
        "input_aspect_ratio": input_aspect_ratio,
        "target_aspect_ratio": target_aspect_ratio,
        "aspect_ratio_relative_error": aspect_ratio_relative_error,
        "maximum_allowed_aspect_ratio_relative_error": max_aspect_ratio_relative_error,
        "cropping_used": False,
        "padding_used": False,
        "enhancement_before_size_lock": False,
        "geometry_projection_before_size_lock": False,
        "outputs": {"png": str(output_png), "tif_16bit": str(output_tif)},
        "warning": (
            "Canvas equality does not make generated intensities measurement truth."
        ),
    }
    (outdir / "size_lock_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_png, output_tif, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Immediately resize an external generated candidate to source dimensions."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-aspect-ratio-error", type=float, default=0.01)
    args = parser.parse_args()
    output_png, output_tif, manifest = lock_generated_to_source(
        args.source,
        args.generated,
        args.outdir,
        max_aspect_ratio_relative_error=args.max_aspect_ratio_error,
    )
    print(
        json.dumps(
            {
                "completed": True,
                "png": str(output_png),
                "tif_16bit": str(output_tif),
                "resized": manifest["resized"],
                "dimensions": manifest["output_dimensions"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
