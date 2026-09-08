#!/usr/bin/env python3
"""Compare registration algorithms at the fixed pre-postprocessing stage."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import generative_registration as registration


METHODS = (
    "phase-translation",
    "global-affine",
    "pchip-envelope",
    "piecewise-linear",
    "piecewise-affine",
    "piecewise-quintic",
    "piecewise-edge-preserving",
)


def save_candidate(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.rint(image * 255.0).astype(np.uint8), mode="L").save(path)


def comparison_figure(
    path: Path,
    fixed: np.ndarray,
    candidates: list[tuple[str, np.ndarray, dict]],
    model: dict,
) -> None:
    all_panels = [("blind guide", fixed, None), *candidates]
    columns = 2
    panel_width, panel_height, label_height = 820, 360, 30
    rows = int(np.ceil(len(all_panels) / columns))
    canvas = Image.new("RGB", (columns * panel_width, rows * (panel_height + label_height)), "black")
    draw = ImageDraw.Draw(canvas)
    fixed_x = model["horizontal"]["fixed"]
    fixed_y = model["vertical"]["fixed"]
    crop_box = (
        max(0, int(fixed_x["left_outer"] - 80)),
        max(0, int(fixed_y["top"] - 80)),
        min(fixed.shape[1], int(fixed_x["right_outer"] + 80)),
        min(fixed.shape[0], int(fixed_y["bottom"] + 80)),
    )
    for index, panel in enumerate(all_panels):
        name, image, metrics = panel
        normalized = registration.robust_normalize(image)
        display = Image.fromarray(
            np.rint(normalized * 255.0).astype(np.uint8), mode="L"
        ).crop(crop_box)
        display = display.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        x = (index % columns) * panel_width
        y = (index // columns) * (panel_height + label_height)
        canvas.paste(display.convert("RGB"), (x, y + label_height))
        if metrics is None:
            label = name
        else:
            summary = metrics["audit"]["summary"]
            sharpness = metrics["sharpness"]["registered"]
            label = (
                f"{name} | envelope P95={summary['envelope_error_after_p95_px']:.2f}px "
                f"| outer-edge P10={sharpness['outer_edge_gradient_p10']:.3f}"
            )
        draw.text((x + 8, y + 8), label, fill=(245, 245, 245))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    source = registration.load_gray(args.source)
    fixed = registration.load_gray(args.guide)
    moving = registration.load_gray(args.generated)
    if source.shape != fixed.shape or source.shape != moving.shape:
        raise ValueError(
            f"All benchmark inputs must share dimensions: {source.shape}, "
            f"{fixed.shape}, {moving.shape}"
        )
    constraints = registration.load_constraints(args.constraints)
    model = registration.estimate_registration(fixed, moving, constraints)
    moving_sharpness = registration.edge_sharpness_metrics(moving, model)
    candidate_panels = []
    results = []
    for method in METHODS:
        candidate, mapping = registration.apply_registration(moving, model, method=method)
        audit = registration.registration_audit(moving, candidate, constraints, model)
        registered_sharpness = registration.edge_sharpness_metrics(candidate, model)
        metrics = {
            "method": method,
            "mapping": mapping,
            "audit": audit,
            "sharpness": {
                "moving": moving_sharpness,
                "registered": registered_sharpness,
                "outer_edge_p10_retention": float(
                    registered_sharpness["outer_edge_gradient_p10"]
                    / max(moving_sharpness["outer_edge_gradient_p10"], 1e-8)
                ),
            },
        }
        geometry_pass = bool(audit["summary"]["guardrail_pass"])
        metrics["accepted_for_full_pipeline"] = geometry_pass
        # Once geometry passes, prefer the method with the strongest weak-side
        # outer edge. The small error term breaks near-ties toward accuracy.
        metrics["selection_score"] = (
            float(registered_sharpness["outer_edge_gradient_p10"])
            - 0.002 * float(audit["summary"]["envelope_error_after_p95_px"])
            if geometry_pass else None
        )
        method_dir = args.outdir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        output = method_dir / "GENERATIVE_registered_candidate.png"
        save_candidate(output, candidate)
        (method_dir / "candidate_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(metrics)
        candidate_panels.append((method, candidate, metrics))

    accepted = [row for row in results if row["accepted_for_full_pipeline"]]
    if not accepted:
        raise RuntimeError("No registration algorithm passed the envelope guardrail")
    selected = max(accepted, key=lambda row: row["selection_score"])
    selected_method = selected["method"]
    selected_source = args.outdir / selected_method / "GENERATIVE_registered_candidate.png"
    selected_output = args.outdir / "SELECTED_registered_candidate.png"
    shutil.copy2(selected_source, selected_output)
    payload = {
        "completed": True,
        "stage_position": "after source-size resize and before visual post-processing",
        "selection_policy": (
            "reject envelope P95 above 2 px; among passing methods maximize weak-side "
            "outer-edge gradient with a small registration-error penalty"
        ),
        "selected_method": selected_method,
        "selected_output": str(selected_output),
        "source_dimensions": {"width": source.shape[1], "height": source.shape[0]},
        "candidates": results,
    }
    (args.outdir / "registration_algorithm_benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comparison_figure(
        args.outdir / "REGISTRATION_algorithm_comparison.png",
        fixed,
        candidate_panels,
        model,
    )
    print(json.dumps({"completed": True, "selected_method": selected_method}), flush=True)


if __name__ == "__main__":
    main()
