#!/usr/bin/env python3
"""Run the v15 dual-output visual and measurement-safe CT workflow."""

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


def guardrails_pass(metrics: dict) -> bool:
    audit = metrics["audit"]
    geometry = audit["dual_evidence_guardrails"]
    structure = audit["structure_detail_consistency"]
    return bool(all(geometry.values()) and structure["guardrail_pass"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "v15: source-size generation and post-processing, followed by separate "
            "visual-only and guide-derived measurement outputs."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    constraint_dir = args.outdir / "00_guide_constraints"
    generated_dir = args.outdir / "01_source_sized_generation"
    postprocess_dir = args.outdir / "02_postprocessed"
    visual_dir = args.outdir / "03_visual_detail_projection"
    measurement_dir = args.outdir / "04_measurement_structure_carrier"
    generated_dir.mkdir(parents=True, exist_ok=True)

    run_script(
        "generative_shape_constraint.py",
        "--source", str(args.source),
        "--guide", str(args.guide),
        "--outdir", str(constraint_dir),
    )
    archived_candidate = generated_dir / "GENERATIVE_candidate.png"
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
        (generated_dir / "generation_preprocess_size_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_sized = generated_dir / (
        "GENERATIVE_denoised_guide_shape_conditioned_"
        f"{size_manifest['target_width']}x{size_manifest['target_height']}.png"
    )
    run_script(
        "generative_postprocess.py",
        "--generated", str(source_sized),
        "--outdir", str(postprocess_dir),
        "--match-source", str(args.source),
    )
    postprocessed = postprocess_dir / "GENERATIVE_visual_only_postprocessed_size_locked.png"

    common = (
        "--source", str(args.source),
        "--guide", str(args.guide),
        "--generated", str(postprocessed),
    )
    run_script(
        "generative_shape_project.py",
        "--profile", "v13",
        *common,
        "--outdir", str(visual_dir),
        "--detail-guide-weight", "0.90",
    )
    run_script(
        "generative_shape_project.py",
        "--profile", "v15",
        *common,
        "--outdir", str(measurement_dir),
    )

    dimensions = f"{size_manifest['target_width']}x{size_manifest['target_height']}"
    visual_png = args.outdir / f"FINAL_VISUAL_ONLY_enhanced_{dimensions}.png"
    visual_tif = args.outdir / f"FINAL_VISUAL_ONLY_enhanced_{dimensions}_16bit.tif"
    measurement_png = args.outdir / f"FINAL_MEASUREMENT_structure_preserved_{dimensions}.png"
    measurement_tif = (
        args.outdir / f"FINAL_MEASUREMENT_structure_preserved_{dimensions}_16bit.tif"
    )
    shutil.copy2(visual_dir / "GENERATIVE_shape_hard_projected.png", visual_png)
    shutil.copy2(visual_dir / "GENERATIVE_shape_hard_projected_16bit.tif", visual_tif)
    shutil.copy2(
        measurement_dir / "MEASUREMENT_structure_carrier_enhanced.png", measurement_png
    )
    shutil.copy2(
        measurement_dir / "MEASUREMENT_structure_carrier_enhanced_16bit.tif",
        measurement_tif,
    )

    visual_metrics = json.loads(
        (visual_dir / "hard_shape_projection_metrics.json").read_text(encoding="utf-8")
    )
    measurement_metrics = json.loads(
        (measurement_dir / "hard_shape_projection_metrics.json").read_text(encoding="utf-8")
    )
    if not guardrails_pass(visual_metrics):
        raise RuntimeError("v15 visual companion failed a geometry or structure guardrail")
    if not guardrails_pass(measurement_metrics):
        raise RuntimeError("v15 measurement output failed a geometry or structure guardrail")

    release = {
        "completed": True,
        "release": "v15-dual-output-structure-carrier",
        "stage_order": (
            "generation -> source-size resize -> post-processing -> separate visual and "
            "measurement projections"
        ),
        "source": str(args.source),
        "blind_denoised_guide": str(args.guide),
        "generated_candidate": str(args.generated),
        "dimensions": {
            "width": int(size_manifest["target_width"]),
            "height": int(size_manifest["target_height"]),
        },
        "outputs": {
            "visual_only_png": str(visual_png),
            "visual_only_tif": str(visual_tif),
            "measurement_structure_png": str(measurement_png),
            "measurement_structure_tif": str(measurement_tif),
        },
        "visual_profile": {
            "profile": "v13",
            "blind_guide_detail_weight": 0.90,
            "audit": visual_metrics["audit"],
            "warning": "VISUAL_ONLY: generated pixels remain; do not use for metrology.",
        },
        "measurement_profile": {
            "profile": "v15",
            "generated_pixel_weight": 0.0,
            "audit": measurement_metrics["audit"],
            "warning": (
                "MEASUREMENT_ASSIST: use calibrated raw/guide constraints as the "
                "authoritative numerical result."
            ),
        },
    }
    (args.outdir / "v15_release_manifest.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed": True, "outputs": release["outputs"]}), flush=True)


if __name__ == "__main__":
    main()
