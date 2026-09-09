#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import structure_audit as audit  # noqa: E402


class StructureAuditTests(unittest.TestCase):
    @staticmethod
    def layers(height: int) -> list[dict]:
        result = []
        for index, center in enumerate((30.0, 42.0, 54.0), start=1):
            result.append({
                "side": "left",
                "layer_id": f"L{index:02d}",
                "center_x_px": center,
                "pitch_px": 12.0,
                "top_y_px": 14.0,
                "bottom_y_px": 78.0,
                "length_px": 64.0,
                "width_median_px": 3.0,
                "constraint_confidence": 0.95,
                "endpoint_uncertainty_px": 0.2,
                "top_snr": 20.0,
                "bottom_snr": 20.0,
                "path_x": np.full(height, center, dtype=np.float32),
            })
        return result

    def test_native_coordinate_contract_rejects_implicit_resize(self) -> None:
        source = np.zeros((32, 48), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "already source-sized"):
            audit.require_native_coordinates(
                source, source.copy(), np.zeros((31, 48), dtype=np.float32)
            )

    def test_constraint_fields_lock_center_and_background(self) -> None:
        shape = (96, 128)
        fields = audit.build_constraint_fields(shape, self.layers(shape[0]))
        stack = np.asarray(fields["stack"])
        self.assertGreater(float(stack[45, 42]), 0.8)
        self.assertEqual(float(stack[45, 100]), 0.0)
        self.assertLess(float(stack[0, 42]), 1e-3)

    def test_alignment_metrics_detect_ridge_improvement(self) -> None:
        shape = (96, 128)
        layers = self.layers(shape[0])
        guide = np.full(shape, 0.08, dtype=np.float32)
        shifted = guide.copy()
        aligned = guide.copy()
        for center in (30, 42, 54):
            guide[14:79, center - 1 : center + 2] = 0.72
            shifted[14:79, center + 1 : center + 4] = 0.72
            aligned[14:79, center - 1 : center + 2] = 0.72
        fields = audit.build_constraint_fields(shape, layers)
        before = audit.fast_alignment_metrics(guide, shifted, shifted, layers, fields)
        after = audit.fast_alignment_metrics(guide, aligned, shifted, layers, fields)
        self.assertLess(
            float(after["ridge_center_absolute_offset_median_px"]),
            float(before["ridge_center_absolute_offset_median_px"]),
        )


if __name__ == "__main__":
    unittest.main()
