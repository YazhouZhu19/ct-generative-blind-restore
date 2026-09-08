#!/usr/bin/env python3
"""Run the v14 registered generative enhancement workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import generative_registration as registration


APP_DIR = Path(__file__).resolve().parent


def run_script(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(APP_DIR / script), *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("v14: resize generation to source dimensions, register its component "
                     "envelope, post-process, then hard-project measured lamella geometry.")
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--registration-method",
        choices=registration.REGISTRATION_METHODS,
        default="piecewise-edge-preserving",
    )
    args = parser.parse_args()

    constraint_dir = args.outdir / "00_guide_constraints"
    generated_dir = args.outdir / "01_generated"
    registration_dir = args.outdir / "02_registered"
    postprocess_dir = args.outdir / "03_postprocessed"
    projection_dir = args.outdir / "04_hard_shape_projection"
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
        "--outdir", str(generated_dir),
        "--match-source", str(args.source),
        "--resize-before-postprocess",
        "--resize-only",
    )
    size_manifest = json.loads(
        (generated_dir / "generation_preprocess_size_manifest.json").read_text(encoding="utf-8")
    )
    source_sized = generated_dir / (
        "GENERATIVE_denoised_guide_shape_conditioned_"
        f"{size_manifest['target_width']}x{size_manifest['target_height']}.png"
    )
    run_script(
        "generative_registration.py",
        "--source", str(args.source),
        "--guide", str(args.guide),
        "--generated", str(source_sized),
        "--constraints", str(constraint_dir / "lamella_geometry.csv"),
        "--outdir", str(registration_dir),
        "--method", args.registration_method,
    )
    registered = registration_dir / "GENERATIVE_registered_to_guide.png"
    run_script(
        "generative_postprocess.py",
        "--generated", str(registered),
        "--outdir", str(postprocess_dir),
        "--match-source", str(args.source),
    )
    postprocessed = postprocess_dir / "GENERATIVE_visual_only_postprocessed_size_locked.png"
    run_script(
        "generative_shape_project.py",
        "--profile", "v11",
        "--source", str(args.source),
        "--guide", str(args.guide),
        "--generated", str(postprocessed),
        "--outdir", str(projection_dir),
    )

    final_png = args.outdir / f"FINAL_registered_enhanced_{source_sized.stem.rsplit('_', 1)[-1]}.png"
    final_tif = args.outdir / f"FINAL_registered_enhanced_{source_sized.stem.rsplit('_', 1)[-1]}_16bit.tif"
    shutil.copy2(projection_dir / "GENERATIVE_shape_hard_projected.png", final_png)
    shutil.copy2(projection_dir / "GENERATIVE_shape_hard_projected_16bit.tif", final_tif)

    registration_metrics = json.loads(
        (registration_dir / "registration_metrics.json").read_text(encoding="utf-8")
    )
    projection_metrics = json.loads(
        (projection_dir / "hard_shape_projection_metrics.json").read_text(encoding="utf-8")
    )
    final_horizontal_audit = registration.final_horizontal_envelope_audit(
        registration.load_gray(final_png),
        registration.load_constraints(constraint_dir / "lamella_geometry.csv"),
        registration_metrics["model"],
    )
    (args.outdir / "FINAL_horizontal_envelope_audit.json").write_text(
        json.dumps(final_horizontal_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not final_horizontal_audit["summary"]["guardrail_pass"]:
        raise RuntimeError(
            f"Final horizontal-envelope audit failed: {final_horizontal_audit['summary']}"
        )
    release = {
        "completed": True,
        "release": "v14-registration-experiment",
        "stage_order": (
            "generation -> source-size resize -> constrained registration -> "
            "post-processing -> hard geometry projection"
        ),
        "source": str(args.source),
        "guide": str(args.guide),
        "generated": str(args.generated),
        "registration_method": args.registration_method,
        "stages": {
            "constraints": str(constraint_dir),
            "source_sized_generation": str(source_sized),
            "registration": str(registration_dir),
            "postprocessing": str(postprocess_dir),
            "hard_projection": str(projection_dir),
        },
        "final_png": str(final_png),
        "final_tif": str(final_tif),
        "registration_audit": registration_metrics["audit"],
        "final_horizontal_envelope_audit": final_horizontal_audit,
        "projection_audit": projection_metrics["audit"],
        "warning": "VISUAL_ONLY: use raw/guide constraints, not generated pixels, for metrology.",
    }
    (args.outdir / "v14_release_manifest.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed": True, "final": str(final_png)}), flush=True)


if __name__ == "__main__":
    main()
