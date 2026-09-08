import sys
import unittest
from pathlib import Path

import numpy as np


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import constrained_detail_fusion as fusion  # noqa: E402
import pipeline as base  # noqa: E402


class ConstrainedDetailFusionTests(unittest.TestCase):
    @staticmethod
    def synthetic_layer() -> dict:
        return {
            "side": "left",
            "layer_id": "L01",
            "center_x_px": 24.0,
            "pitch_px": 12.0,
            "top_y_px": 8.0,
            "bottom_y_px": 40.0,
            "length_px": 32.0,
            "width_median_px": 4.0,
            "width_p10_px": 3.8,
            "width_p90_px": 4.2,
            "path_x": np.full(48, 24.0, dtype=np.float32),
        }

    def test_zero_strength_cleanup_is_identity_and_keeps_canvas(self) -> None:
        image = np.random.default_rng(7).random((48, 48), dtype=np.float32)
        target = base.Roi(8, 40, 8, 40)
        zeros = np.zeros((32, 32), dtype=np.float32)
        maps = {
            "boundary": zeros,
            "endpoint": zeros,
            "interlayer": zeros,
            "protection": zeros,
            "lamella": zeros,
            "confidence": zeros,
        }
        output, method, _ = fusion.symmetric_constraint_cleanup(
            image,
            target,
            maps,
            boundary_gain=0.0,
            endpoint_gain=0.0,
            interlayer_denoise=0.0,
        )
        self.assertTrue(np.array_equal(output, image))
        self.assertIsNone(method["spatial_transform"])
        self.assertIsNone(method["resampling"])

    def test_cleanup_changes_only_target_roi(self) -> None:
        image = np.random.default_rng(11).random((48, 48), dtype=np.float32)
        target = base.Roi(8, 40, 8, 40)
        ones = np.ones((32, 32), dtype=np.float32)
        zeros = np.zeros_like(ones)
        maps = {
            "boundary": ones,
            "endpoint": zeros,
            "interlayer": zeros,
            "protection": ones,
            "lamella": ones,
            "confidence": ones,
        }
        output, _, _ = fusion.symmetric_constraint_cleanup(
            image,
            target,
            maps,
            boundary_gain=0.25,
            endpoint_gain=0.0,
            interlayer_denoise=0.0,
        )
        outside = np.ones_like(image, dtype=bool)
        outside[target.slices()] = False
        self.assertTrue(np.array_equal(output[outside], image[outside]))
        self.assertGreater(float(np.max(np.abs(output[target.slices()] - image[target.slices()]))), 0.0)

    def test_local_width_profile_keeps_authoritative_median(self) -> None:
        image = np.zeros((48, 48), dtype=np.float32)
        image[8:41, 22:27] = 1.0
        layer = self.synthetic_layer()
        profiles, rows = fusion.measure_local_width_profiles(image, [layer], 0.35)
        values = profiles["L01"][8:41]
        self.assertAlmostEqual(float(np.median(values)), 4.0, places=5)
        self.assertEqual(len(rows), 33)

    def test_rollback_restores_unsafe_measurement_neighborhood(self) -> None:
        baseline = np.zeros((48, 48), dtype=np.float32)
        candidate = np.ones_like(baseline)
        layer = self.synthetic_layer()
        output, mask = fusion.rollback_unsafe_structures(
            candidate, baseline, [layer], {"L01"}
        )
        self.assertEqual(float(output[24, 24]), 0.0)
        self.assertEqual(float(mask[24, 24]), 1.0)
        self.assertEqual(float(output[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
