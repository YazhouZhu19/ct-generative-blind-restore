#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import native_structure_warp as warp  # noqa: E402
import structure_audit as audit  # noqa: E402


class NativeStructureWarpTests(unittest.TestCase):
    @staticmethod
    def layers(height: int) -> list[dict]:
        rows = []
        for index, center in enumerate((28.0, 40.0, 52.0, 64.0), start=1):
            rows.append({
                "side": "left",
                "layer_id": f"L{index:02d}",
                "center_x_px": center,
                "pitch_px": 12.0,
                "top_y_px": 12.0,
                "bottom_y_px": 68.0,
                "length_px": 56.0,
                "width_median_px": 3.0,
                "constraint_confidence": 0.95,
                "path_x": np.full(height, center, dtype=np.float32),
            })
        return rows

    def test_monotonic_match_skips_extra_generated_ridge(self) -> None:
        fixed = np.asarray((10.0, 20.0, 30.0, 40.0))
        moving = np.asarray((9.0, 19.0, 24.0, 29.0, 39.0))
        pairs = warp.monotonic_match(fixed, moving)
        self.assertEqual([first for first, _ in pairs], [0, 1, 2, 3])
        self.assertEqual([second for _, second in pairs], [0, 1, 3, 4])

    def test_row_mapping_is_bounded_and_hits_control_points(self) -> None:
        axis = np.arange(100, dtype=np.float32)
        mapped = warp.row_source_coordinates(
            np.asarray((20.0, 50.0, 80.0)),
            np.asarray((23.0, 48.0, 86.0)),
            axis,
            maximum_displacement=4.0,
        )
        self.assertAlmostEqual(float(mapped[20]), 23.0, places=5)
        self.assertAlmostEqual(float(mapped[50]), 48.0, places=5)
        self.assertAlmostEqual(float(mapped[80]), 84.0, places=5)
        self.assertLessEqual(float(np.max(np.abs(mapped - axis))), 4.0)

    def test_native_warp_moves_generated_ridge_and_locks_outside(self) -> None:
        shape = (80, 112)
        layers = self.layers(shape[0])
        # Add an empty right-side field so the production routine can exercise
        # both named stacks without changing the test target.
        fields = audit.build_constraint_fields(shape, layers)
        generated = np.full(shape, 0.08, dtype=np.float32)
        matches = {"left": [], "right": []}
        for row in layers:
            moving = np.asarray(row["path_x"]) + 3.0
            center = int(round(float(moving[0])))
            generated[12:69, center - 1 : center + 2] = 0.80
            matches["left"].append({
                "fixed_path_x": np.asarray(row["path_x"]),
                "moving_path_x": moving,
                "fixed_top_y_px": 12.0,
                "fixed_bottom_y_px": 68.0,
                "moving_top_y_px": 12.0,
                "moving_bottom_y_px": 68.0,
            })
        before = generated.copy()
        output, metrics = warp.apply_native_warp(
            generated, fields, matches, strength=1.0, maximum_displacement=5.0
        )
        self.assertGreater(float(output[40, 28]), float(before[40, 28]))
        self.assertLess(float(output[40, 31]), float(before[40, 31]))
        outside = np.asarray(fields["stack"]) <= 1e-5
        self.assertTrue(np.array_equal(output[outside], before[outside]))
        self.assertFalse(metrics["canvas_resized"])
        self.assertFalse(metrics["guide_or_raw_pixel_writeback"])
        self.assertTrue(metrics["vertical_coordinate_changed"])

    @staticmethod
    def alignment_metrics(mid: float, median: float, p95: float) -> dict:
        return {
            "guide_mid_frequency_correlation": mid,
            "ridge_center_absolute_offset_median_px": median,
            "ridge_center_absolute_offset_p95_px": p95,
            "global_ssim_to_accepted_generation": 1.0,
            "outside_writable_max_abs_change": 0.0,
        }

    def test_choose_warp_uses_identity_when_every_warp_worsens_alignment(self) -> None:
        generated = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(6, 8)
        baseline = self.alignment_metrics(mid=0.60, median=1.0, p95=2.0)
        worse_first = self.alignment_metrics(mid=0.58, median=1.2, p95=2.3)
        worse_second = self.alignment_metrics(mid=0.55, median=1.4, p95=2.6)

        def fake_apply(*args, strength: float, **kwargs):
            return np.full_like(generated, strength), {"strength": strength}

        with (
            mock.patch.object(warp, "apply_native_warp", side_effect=fake_apply) as apply,
            mock.patch.object(
                warp.audit,
                "fast_alignment_metrics",
                side_effect=(baseline, worse_first, worse_second),
            ),
        ):
            selected, selection, records = warp.choose_warp(
                [0.20, 0.40],
                generated,
                generated,
                [],
                {},
                {},
                minimum_ssim=0.965,
                maximum_displacement=5.0,
                maximum_vertical_displacement=5.0,
            )

        self.assertTrue(np.array_equal(selected, generated))
        self.assertEqual(selection["selected"]["warp"]["strength"], 0.0)
        self.assertTrue(selection["selected"]["warp"]["identity_no_op"])
        self.assertEqual([row["warp"]["strength"] for row in records], [0.0, 0.20, 0.40])
        self.assertTrue(records[0]["guardrail_pass"])
        self.assertFalse(records[1]["guardrail_pass"])
        self.assertFalse(records[2]["guardrail_pass"])
        self.assertEqual(apply.call_count, 2)

    def test_choose_warp_keeps_better_nonzero_candidate(self) -> None:
        generated = np.full((6, 8), 0.25, dtype=np.float32)
        baseline = self.alignment_metrics(mid=0.40, median=2.0, p95=4.0)
        improved = self.alignment_metrics(mid=0.48, median=0.8, p95=1.5)

        def fake_apply(*args, strength: float, **kwargs):
            return np.full_like(generated, 0.75), {"strength": strength}

        with (
            mock.patch.object(warp, "apply_native_warp", side_effect=fake_apply),
            mock.patch.object(
                warp.audit,
                "fast_alignment_metrics",
                side_effect=(baseline, improved),
            ),
        ):
            selected, selection, records = warp.choose_warp(
                [0.20],
                generated,
                generated,
                [],
                {},
                {},
                minimum_ssim=0.965,
                maximum_displacement=5.0,
                maximum_vertical_displacement=5.0,
            )

        expected = warp.base.to_uint16(np.full_like(generated, 0.75)).astype(
            np.float32
        ) / 65535.0
        self.assertTrue(np.array_equal(selected, expected))
        self.assertEqual(selection["selected"]["warp"]["strength"], 0.20)
        self.assertGreater(
            records[1]["selection_score"], records[0]["selection_score"]
        )

    def test_choose_warp_preserves_guardrail_passing_historical_selection(self) -> None:
        generated = np.full((6, 8), 0.25, dtype=np.float32)
        baseline = self.alignment_metrics(mid=0.60, median=2.0, p95=4.0)
        # Ridge alignment improves and the established mid-frequency tolerance
        # passes, although the weighted score is slightly below the identity.
        historical = self.alignment_metrics(mid=0.56, median=1.95, p95=3.95)

        def fake_apply(*args, strength: float, **kwargs):
            return np.full_like(generated, 0.75), {"strength": strength}

        with (
            mock.patch.object(warp, "apply_native_warp", side_effect=fake_apply),
            mock.patch.object(
                warp.audit,
                "fast_alignment_metrics",
                side_effect=(baseline, historical),
            ),
        ):
            selected, selection, records = warp.choose_warp(
                [0.16],
                generated,
                generated,
                [],
                {},
                {},
                minimum_ssim=0.965,
                maximum_displacement=5.0,
                maximum_vertical_displacement=5.0,
            )

        self.assertFalse(selection["identity_fallback_used"])
        self.assertEqual(selection["selected"]["warp"]["strength"], 0.16)
        self.assertLess(records[1]["selection_score"], records[0]["selection_score"])
        self.assertFalse(np.array_equal(selected, generated))


if __name__ == "__main__":
    unittest.main()
