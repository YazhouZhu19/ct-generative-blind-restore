#!/usr/bin/env python3
"""Reproduce the frozen v11 guide-first workflow without registration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def run_script(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(APP_DIR / script), *arguments], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=("Freeze and reproduce v11: blind-denoised guide measurement, "
                     "existing visual post-processing, and measured-ribbon projection.")
    )
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--guide", type=Path, required=True)
    ap.add_argument(
        "--generated", type=Path, required=True,
        help="Previously generated v11 soft candidate (visual-only).",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    constraint_dir = args.outdir / "00_guide_constraints"
    generated_dir = args.outdir / "01_generated"
    postprocess_dir = args.outdir / "02_postprocessed"
    projection_dir = args.outdir / "03_hard_shape_projection"
    generated_dir.mkdir(parents=True, exist_ok=True)

    run_script(
        "generative_shape_constraint.py",
        "--source", str(args.source),
        "--guide", str(args.guide),
        "--outdir", str(constraint_dir),
    )
    archived_candidate = generated_dir / "GENERATIVE_denoised_guide_shape_conditioned_raw.png"
    if args.generated.resolve() != archived_candidate.resolve():
        shutil.copy2(args.generated, archived_candidate)
    run_script(
        "generative_postprocess.py",
        "--generated", str(archived_candidate),
        "--outdir", str(postprocess_dir),
        "--match-source", str(args.source),
        "--resize-before-postprocess",
    )
    generation_resize_metrics = json.loads(
        (postprocess_dir / "generation_preprocess_size_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_sized_name = (
        "GENERATIVE_denoised_guide_shape_conditioned_"
        f"{generation_resize_metrics['target_width']}x"
        f"{generation_resize_metrics['target_height']}.png"
    )
    source_sized_candidate = postprocess_dir / source_sized_name
    archived_source_sized_candidate = generated_dir / source_sized_name
    shutil.copy2(source_sized_candidate, archived_source_sized_candidate)
    size_locked = postprocess_dir / "GENERATIVE_visual_only_postprocessed_size_locked.png"
    run_script(
        "generative_shape_project.py",
        "--profile", "v11",
        "--source", str(args.source),
        "--guide", str(args.guide),
        "--generated", str(size_locked),
        "--outdir", str(projection_dir),
    )

    projection_metrics = json.loads(
        (projection_dir / "hard_shape_projection_metrics.json").read_text(encoding="utf-8")
    )
    size_lock_metrics = json.loads(
        (postprocess_dir / "size_lock_manifest.json").read_text(encoding="utf-8")
    )
    target_width = int(generation_resize_metrics["target_width"])
    target_height = int(generation_resize_metrics["target_height"])
    final_png = (
        args.outdir
        / f"FINAL_no_registration_enhanced_{target_width}x{target_height}.png"
    )
    final_tif = (
        args.outdir
        / f"FINAL_no_registration_enhanced_{target_width}x{target_height}_16bit.tif"
    )
    shutil.copy2(
        projection_dir / "GENERATIVE_shape_hard_projected.png", final_png
    )
    shutil.copy2(
        projection_dir / "GENERATIVE_shape_hard_projected_16bit.tif", final_tif
    )
    release = {
        "completed": True,
        "release": "v11-stable-no-registration",
        "profile": "v11",
        "stage_order": (
            "generation -> source-size resize -> visual post-processing -> "
            "hard geometry projection"
        ),
        "registration": {
            "enabled": False,
            "method": None,
            "transform_applied": False,
            "reason": (
                "Selected release disables translation, affine, deformable, and "
                "piecewise registration. All geometry remains in source coordinates."
            ),
        },
        "source": str(args.source),
        "blind_denoised_measurement_guide": str(args.guide),
        "soft_generated_candidate": str(args.generated),
        "detail_guide_weight": 0.0,
        "stages": {
            "constraints": str(constraint_dir),
            "archived_generation": str(archived_candidate),
            "source_sized_generation_before_postprocessing": str(
                archived_source_sized_candidate
            ),
            "postprocessing": str(postprocess_dir),
            "size_locked_postprocessing": str(size_locked),
            "hard_projection": str(projection_dir),
            "final_png": str(final_png),
            "final_16bit_tiff": str(final_tif),
        },
        "dimension_audit": {
            "generation_preprocess_resize": generation_resize_metrics,
            "size_lock": size_lock_metrics,
            "projection": projection_metrics["dimension_audit"],
        },
        "projection_audit": projection_metrics["audit"],
        "warning": "VISUAL_ONLY: generated texture is not metrology-certified.",
    }
    (args.outdir / "v11_release_manifest.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "completed": True,
                "release": "v11-stable-no-registration",
                "registration_enabled": False,
                "final_png": str(final_png),
                "final_16bit_tiff": str(final_tif),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
