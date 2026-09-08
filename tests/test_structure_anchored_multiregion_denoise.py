import sys
import unittest
from pathlib import Path

import numpy as np


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import pipeline as base  # noqa: E402
import structure_anchored_multiregion_denoise as v19  # noqa: E402


class StructureAnchoredMultiregionDenoiseTests(unittest.TestCase):
    @staticmethod
    def synthetic_layer(
        *,
        height: int = 48,
        center_x: float = 24.0,
        reference_quality: str = "pass",
    ) -> dict:
        return {
            "side": "left",
            "layer_id": "L01",
            "center_x_px": center_x,
            "pitch_px": 12.0,
            "top_y_px": 8.0,
            "bottom_y_px": 40.0,
            "length_px": 32.0,
            "width_median_px": 4.0,
            "width_p10_px": 3.8,
            "width_p90_px": 4.2,
            "path_x": np.full(height, center_x, dtype=np.float32),
            "reference_quality": reference_quality,
        }

    @staticmethod
    def candidate_components(image: np.ndarray, target: base.Roi) -> dict:
        crop = image[target.slices()].copy()
        shape = crop.shape
        ones = np.ones(shape, dtype=np.float32)
        zeros = np.zeros(shape, dtype=np.float32)
        return {
            "crop": crop,
            "hard_anchor": np.zeros(shape, dtype=bool),
            "stack_gate": ones,
            "central_gate": zeros,
            "fog_gate": zeros,
            "flat_gate": zeros,
            "stack_delta": np.full(shape, 0.02, dtype=np.float32),
            "central_delta": zeros,
            "fog_delta": zeros,
            "flat_delta": zeros,
        }

    def test_zero_strength_is_identity_and_keeps_canvas(self) -> None:
        image = np.random.default_rng(17).random((48, 52), dtype=np.float32)
        target = base.Roi(8, 40, 10, 42)
        components = self.candidate_components(image, target)

        output, method, influence = v19.apply_multiregion_candidate(
            image,
            target,
            components,
            noise_sigma=0.01,
            strengths=(0.0, 0.0, 0.0, 0.0),
        )

        self.assertEqual(output.shape, image.shape)
        self.assertTrue(np.array_equal(output, image))
        self.assertFalse(np.any(influence))
        self.assertIsNone(method["spatial_transform"])
        self.assertIsNone(method["resampling"])

    def test_hard_endpoint_anchor_is_bit_exact_after_uint16_quantization(self) -> None:
        image = np.full((48, 52), 20000.0 / 65535.0, dtype=np.float32)
        target = base.Roi(8, 40, 10, 42)
        components = self.candidate_components(image, target)
        components["hard_anchor"][12:20, 13:19] = True

        output, _, influence = v19.apply_multiregion_candidate(
            image,
            target,
            components,
            noise_sigma=0.03,
            strengths=(1.0, 0.0, 0.0, 0.0),
        )

        full_anchor = np.zeros(image.shape, dtype=bool)
        full_anchor[target.slices()] = components["hard_anchor"]
        before_u16 = base.to_uint16(image)
        after_u16 = base.to_uint16(output)
        self.assertTrue(np.array_equal(after_u16[full_anchor], before_u16[full_anchor]))
        self.assertTrue(np.all(influence[components["hard_anchor"]] == 0.0))
        self.assertTrue(np.any(after_u16[~full_anchor] != before_u16[~full_anchor]))

    def test_pure_axial_wiener_does_not_mix_in_x_direction(self) -> None:
        crop = np.zeros((31, 11), dtype=np.float32)
        crop[15, 5] = 1.0

        delta, shrinkage = v19.adaptive_wiener_delta(
            crop,
            noise_sigma=0.08,
            filter_sigma=(1.6, 0.0),
            local_sigma=(3.2, 0.0),
        )

        self.assertGreater(float(np.max(np.abs(delta[:, 5]))), 0.0)
        self.assertTrue(np.array_equal(delta[:, :5], np.zeros_like(delta[:, :5])))
        self.assertTrue(np.array_equal(delta[:, 6:], np.zeros_like(delta[:, 6:])))
        self.assertTrue(np.all(shrinkage[:, :5] == 1.0))
        self.assertTrue(np.all(shrinkage[:, 6:] == 1.0))

    def test_quantized_float_matches_uint16_storage_grid(self) -> None:
        image = np.asarray(
            [
                -0.2,
                0.0,
                0.49 / 65535.0,
                0.51 / 65535.0,
                1.49 / 65535.0,
                1.51 / 65535.0,
                0.5,
                1.0,
                1.2,
            ],
            dtype=np.float32,
        )
        expected_u16 = np.asarray(
            [0, 0, 0, 1, 1, 2, 32768, 65535, 65535], dtype=np.uint16
        )

        quantized = v19.quantized_float(image)

        self.assertEqual(quantized.dtype, np.float32)
        self.assertEqual(quantized.shape, image.shape)
        self.assertTrue(np.array_equal(base.to_uint16(quantized), expected_u16))
        self.assertTrue(np.array_equal(v19.quantized_float(quantized), quantized))

    def test_measurement_operator_anchor_has_target_shape_and_expected_support(self) -> None:
        image_shape = (60, 80)
        target = base.Roi(5, 55, 10, 70)
        layer = self.synthetic_layer(height=image_shape[0], center_x=30.0)
        layer["top_y_px"] = 15.0
        layer["bottom_y_px"] = 45.0

        anchor, method = v19.measurement_operator_anchor(
            image_shape,
            target,
            [layer],
            top_range=(10, 20),
            bottom_range=(40, 50),
        )

        self.assertEqual(anchor.shape, (50, 60))
        self.assertEqual(anchor.dtype, np.bool_)
        # The endpoint profile support includes the tracked centerline.
        self.assertTrue(anchor[10 - target.y0, 30 - target.x0])
        # A deployed middle width sample includes y +/- 2 and transverse support.
        self.assertTrue(anchor[30 - target.y0, 22 - target.x0])
        self.assertTrue(anchor[30 - target.y0, 38 - target.x0])
        self.assertFalse(anchor[30 - target.y0, 10 - target.x0])
        self.assertGreater(method["width_support_fraction_target"], 0.0)
        self.assertGreater(method["endpoint_support_fraction_target"], 0.0)
        self.assertGreater(method["union_fraction_target"], 0.0)
        self.assertLess(method["union_fraction_target"], 1.0)

    def test_local_width_drift_is_zero_for_identical_images(self) -> None:
        yy, xx = np.indices((48, 48), dtype=np.float32)
        del yy
        image = np.exp(-0.5 * np.square((xx - 24.0) / 2.0)).astype(np.float32)
        layer = self.synthetic_layer()

        summary, rows, unsafe = v19.local_width_drift(
            image, image.copy(), [layer], row_step=3
        )

        self.assertGreater(summary["sample_count"], 0)
        self.assertEqual(summary["unsafe_layer_count"], 0)
        self.assertEqual(unsafe, set())
        self.assertEqual(summary["absolute_drift_median_px"], 0.0)
        self.assertEqual(summary["absolute_drift_p95_px"], 0.0)
        self.assertEqual(summary["absolute_drift_max_px"], 0.0)
        self.assertTrue(rows)
        self.assertTrue(all(row["absolute_drift_px"] == 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
