#!/usr/bin/env python3
"""Run blind denoising, enhancement, boundary cleanup, residual denoising, and length QA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print(json.dumps({"running": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--generative-prior", type=Path, required=True)
    ap.add_argument(
        "--reference-style",
        type=Path,
        help="optional unregistered appearance reference; never used for geometry or pixel writeback",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--features", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument(
        "--directional-strength",
        type=float,
        default=0.70,
        help="maximum geometry-locked axial-continuity strength searched in the blind stage",
    )
    args = ap.parse_args()

    app = Path(__file__).resolve().parent
    blind_dir = args.outdir / "01_sota_blind"
    quality_dir = args.outdir / "02_geometry_quality"
    cleanup_dir = args.outdir / "03_boundary_cleanup"
    residual_dir = args.outdir / "04_residual_denoise"
    length_dir = args.outdir / "05_length_audit"
    for directory in (blind_dir, quality_dir, cleanup_dir, residual_dir, length_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        str(app / "sota_geometry_blind.py"),
        "--source", str(args.source),
        "--generative-prior", str(args.generative_prior),
        "--outdir", str(blind_dir),
        "--iterations", str(args.iterations),
        "--device", args.device,
        "--features", str(args.features),
        "--batch", str(args.batch),
        "--directional-strength", str(args.directional_strength),
    ])
    run([
        sys.executable,
        str(app / "quality_optimize.py"),
        "--source", str(args.source),
        "--baseline", str(blind_dir / "MEASUREMENT_sota_geometry_blind_16bit.tif"),
        "--outdir", str(quality_dir),
    ])
    run([
        sys.executable,
        str(app / "boundary_cleanup.py"),
        "--source", str(args.source),
        "--input", str(quality_dir / "QUALITY_boundary_preserved_16bit.tif"),
        "--outdir", str(cleanup_dir),
    ] + (["--reference-style", str(args.reference_style)] if args.reference_style else []))
    run([
        sys.executable,
        str(app / "residual_denoise.py"),
        "--source", str(args.source),
        "--input", str(cleanup_dir / "QUALITY_boundary_clean_16bit.tif"),
        "--outdir", str(residual_dir),
    ])
    run([
        sys.executable,
        str(app / "length_optimize.py"),
        "--source", str(args.source),
        "--denoised", str(residual_dir / "QUALITY_v7_residual_denoised_16bit.tif"),
        "--geometry-guide", str(cleanup_dir / "QUALITY_boundary_clean_16bit.tif"),
        "--outdir", str(length_dir),
        "--sharpen-amount", "0",
    ])
    blind_metrics = json.loads((blind_dir / "sota_run_manifest.json").read_text(encoding="utf-8"))
    quality_metrics = json.loads((quality_dir / "quality_metrics.json").read_text(encoding="utf-8"))
    cleanup_metrics = json.loads((cleanup_dir / "boundary_cleanup_metrics.json").read_text(encoding="utf-8"))
    residual_metrics = json.loads((residual_dir / "residual_denoise_metrics.json").read_text(encoding="utf-8"))
    length_metrics = json.loads((length_dir / "length_qa.json").read_text(encoding="utf-8"))
    selected_blind = blind_metrics["projection_selection"]["selected"]
    selected_directional = blind_metrics["directional_projection_selection"]["selected"]
    selected_quality = quality_metrics["selected"]
    summary = {
        "completed": bool(
            selected_blind["measurement_guardrail_pass"]
            and selected_blind["boundary_guardrail_pass"]
            and selected_directional["guardrail_pass"]
            and selected_quality["guardrail_pass"]
            and selected_quality["width_guardrail_pass"]
            and selected_quality["boundary_guardrail_pass"]
            and cleanup_metrics["selected"]["guardrail_pass"]
            and residual_metrics["selected"]["guardrail_pass"]
            and length_metrics["guardrail_pass"]
        ),
        "blind_output": str(blind_dir / "MEASUREMENT_sota_geometry_blind_16bit.tif"),
        "final_output": str(residual_dir / "QUALITY_v7_residual_denoised_16bit.tif"),
        "quality_metrics": str(quality_dir / "quality_metrics.json"),
        "boundary_cleanup_metrics": str(cleanup_dir / "boundary_cleanup_metrics.json"),
        "residual_denoise_metrics": str(residual_dir / "residual_denoise_metrics.json"),
        "length_metrics": str(length_dir / "length_qa.json"),
        "validation": {
            "blind_projection_noise_roughness_reduction_vs_raw": selected_blind["noise_roughness_reduction_vs_raw"],
            "blind_noise_roughness_reduction_vs_raw": selected_directional["noise_roughness_reduction_vs_raw"],
            "directional_strength": selected_directional["strength"],
            "directional_additional_roughness_reduction_vs_projection": selected_directional["additional_roughness_reduction_vs_projection"],
            "directional_ssim_vs_projection_input": selected_directional["ssim_vs_projection_input"],
            "directional_edge_acutance_retention": selected_directional["edge_acutance_retention"],
            "directional_local_contrast_retention": selected_directional["local_contrast_retention"],
            "directional_endpoint_shift_abs_p95_px": selected_directional["boundary_geometry"]["endpoint_shift_abs_p95_px"],
            "directional_length_delta_abs_p95_px": selected_directional["boundary_geometry"]["length_delta_abs_p95_px"],
            "pre_cleanup_high_frequency_reduction_vs_raw": quality_metrics["quality_vs_original"]["flat_high_frequency_reduction"],
            "final_flat_high_frequency_reduction_vs_raw": residual_metrics["validation_vs_source"]["flat_high_frequency_reduction"],
            "final_central_high_frequency_reduction_vs_raw": residual_metrics["validation_vs_source"]["central_high_frequency_reduction"],
            "final_edge_acutance_gain_vs_raw": residual_metrics["validation_vs_source"]["edge_acutance_gain"],
            "final_local_contrast_gain_vs_raw": residual_metrics["validation_vs_source"]["local_contrast_gain"],
            "final_max_median_fwhm_relative_change_vs_raw": residual_metrics["validation_vs_source"]["max_median_fwhm_relative_change"],
            "boundary_exterior_noise_reduction_vs_enhanced": cleanup_metrics["selected"]["exterior_noise_reduction"],
            "boundary_exterior_halo_reduction_vs_enhanced": cleanup_metrics["selected"]["exterior_halo_reduction"],
            "boundary_interior_axial_noise_reduction_vs_enhanced": cleanup_metrics["selected"]["interior_axial_noise_reduction"],
            "residual_lamella_axial_noise_reduction_vs_v6": residual_metrics["selected"]["lamella_axial_noise_reduction"],
            "residual_central_noise_reduction_vs_v6": residual_metrics["selected"]["central_noise_reduction"],
            "cumulative_lamella_axial_noise_reduction_vs_v5": 1.0
            - (1.0 - cleanup_metrics["selected"]["interior_axial_noise_reduction"])
            * (1.0 - residual_metrics["selected"]["lamella_axial_noise_reduction"]),
            "independent_layer_count": length_metrics["layer_count"],
            "independent_pass_count": length_metrics["pass_count"],
            "independent_endpoint_shift_abs_p95_px": length_metrics["endpoint_shift_abs_p95_px"],
            "independent_length_delta_abs_p95_px": length_metrics["length_delta_abs_p95_px"],
        },
    }
    (args.outdir / "SOTA_PIPELINE_COMPLETE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
