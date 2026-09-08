import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import imread, imwrite


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import measurement_safe_postprocess as v20  # noqa: E402
import pipeline as base  # noqa: E402


class MeasurementSafePostprocessTests(unittest.TestCase):
    @staticmethod
    def components(image: np.ndarray, target: base.Roi) -> dict:
        crop = image[target.slices()].copy()
        shape = crop.shape
        zero = np.zeros(shape, dtype=np.float32)
        components: dict[str, np.ndarray] = {
            "crop": crop,
            "central_gate": zero.copy(),
            "fog_gate": zero.copy(),
            "flat_gate": zero.copy(),
            "hard_lock": np.zeros(shape, dtype=bool),
            "writable_support": np.zeros(shape, dtype=bool),
            "gate_zero_contour": np.zeros(shape, dtype=bool),
            "gate_seam_inner": np.zeros(shape, dtype=bool),
        }
        for region in ("central", "fog", "flat"):
            for estimator in ("tv04", "tv08", "tv12"):
                components[f"{region}_{estimator}_delta"] = zero.copy()
        return components

    @staticmethod
    def spec(**overrides: object) -> dict:
        values = {
            "name": "test",
            "central_estimator": "tv08",
            "central_strength": 0.0,
            "fog_estimator": "tv04",
            "fog_strength": 0.0,
            "flat_estimator": "tv08",
            "flat_strength": 0.0,
        }
        values.update(overrides)
        return values

    def test_zero_strength_is_exact_uint16_identity_and_keeps_shape(self) -> None:
        image = np.random.default_rng(20).random((38, 46), dtype=np.float32)
        target = base.Roi(5, 33, 7, 39)
        components = self.components(image, target)

        output, operation, influence = v20.apply_postprocess_candidate(
            image, target, components, self.spec()
        )

        self.assertEqual(output.shape, image.shape)
        self.assertTrue(np.array_equal(base.to_uint16(output), base.to_uint16(image)))
        self.assertFalse(np.any(influence))
        self.assertEqual(operation["stack_strength"], 0.0)
        self.assertIsNone(operation["resize"])
        self.assertIsNone(operation["resampling"])

    def test_hard_lock_and_pixels_outside_target_are_bit_exact(self) -> None:
        image = np.full((38, 46), 22000.0 / 65535.0, dtype=np.float32)
        target = base.Roi(5, 33, 7, 39)
        components = self.components(image, target)
        components["central_gate"][:] = 1.0
        components["central_tv08_delta"][:] = 80.0 / 65535.0
        components["hard_lock"][8:15, 9:18] = True

        output, _, influence = v20.apply_postprocess_candidate(
            image,
            target,
            components,
            self.spec(central_strength=1.0),
        )

        before = base.to_uint16(image)
        after = base.to_uint16(output)
        full_lock = np.zeros(image.shape, dtype=bool)
        full_lock[target.slices()] = components["hard_lock"]
        outside = np.ones(image.shape, dtype=bool)
        outside[target.slices()] = False
        self.assertTrue(np.array_equal(after[full_lock], before[full_lock]))
        self.assertTrue(np.array_equal(after[outside], before[outside]))
        self.assertTrue(np.any(after[~(full_lock | outside)] != before[~(full_lock | outside)]))
        self.assertTrue(np.all(influence[components["hard_lock"]] == 0.0))

    def test_region_strengths_do_not_cross_write_between_disjoint_gates(self) -> None:
        image = np.full((30, 34), 0.4, dtype=np.float32)
        target = base.Roi(3, 27, 4, 30)
        components = self.components(image, target)
        components["central_gate"][2:8, 2:8] = 1.0
        components["fog_gate"][9:15, 9:15] = 1.0
        components["flat_gate"][16:22, 16:22] = 1.0
        components["central_tv08_delta"][:] = 50.0 / 65535.0
        components["fog_tv04_delta"][:] = 70.0 / 65535.0
        components["flat_tv08_delta"][:] = 90.0 / 65535.0

        output, _, _ = v20.apply_postprocess_candidate(
            image,
            target,
            components,
            self.spec(central_strength=1.0),
        )
        changed = base.to_uint16(output[target.slices()]) != base.to_uint16(
            image[target.slices()]
        )

        self.assertTrue(np.any(changed[components["central_gate"] > 0]))
        self.assertFalse(np.any(changed[components["fog_gate"] > 0]))
        self.assertFalse(np.any(changed[components["flat_gate"] > 0]))

    def test_central_masks_have_locked_ring_and_eroded_core(self) -> None:
        target = base.Roi(10, 70, 20, 100)
        central = base.Roi(20, 60, 35, 85)

        rectangle, boundary, core = v20.central_masks(
            target, central, (60, 80), lock_width=6, full_weight_distance=12
        )

        self.assertTrue(rectangle[10, 15])
        self.assertTrue(boundary[10, 15])
        self.assertEqual(core[10, 15], 0.0)
        self.assertGreater(core[30, 40], 0.99)
        self.assertFalse(boundary[30, 40])
        self.assertTrue(np.all(core[boundary & rectangle] == 0.0))

    def test_distance_gate_is_compact_zero_on_first_contour_and_monotonic(self) -> None:
        support = np.zeros((15, 15), dtype=bool)
        support[3:12, 3:12] = True

        gate = v20.distance_soft_region(support, ramp_px=5.0)
        first_contour = support.copy()
        first_contour[4:11, 4:11] = False

        self.assertTrue(np.all(gate[~support] == 0.0))
        self.assertTrue(np.all(gate[first_contour] == 0.0))
        self.assertEqual(float(gate[7, 7]), 1.0)
        self.assertTrue(np.all(np.diff(gate[3:8, 7]) >= 0.0))

        border_support = np.ones((12, 14), dtype=bool)
        border_gate = v20.distance_soft_region(border_support, ramp_px=5.0)
        self.assertTrue(np.all(border_gate[0] == 0.0))
        self.assertTrue(np.all(border_gate[-1] == 0.0))
        self.assertTrue(np.all(border_gate[:, 0] == 0.0))
        self.assertTrue(np.all(border_gate[:, -1] == 0.0))

    def test_gate_audit_detects_write_outside_allowed_support(self) -> None:
        image = np.full((20, 24), 12000.0 / 65535.0, dtype=np.float32)
        target = base.Roi(2, 18, 3, 21)
        components = self.components(image, target)
        components["central_gate"][4:12, 5:13] = 1.0
        components["fog_gate"][5, 6] = 1.0
        components["writable_support"] = components["central_gate"] > 0.0
        components["writable_support"] |= components["fog_gate"] > 0.0
        components["hard_lock"][5, 6] = True
        candidate = image.copy()
        local = base.to_uint16(candidate[target.slices()])
        local[1, 1] += np.uint16(1)
        candidate[target.slices()] = local.astype(np.float32) / 65535.0

        audit = v20.gate_boundary_audit(image, candidate, target, components)

        self.assertEqual(audit["changed_outside_writable_support_uint16"], 1)
        self.assertEqual(audit["gate_overlap_pixel_count"], 1)
        self.assertEqual(audit["writable_hard_lock_overlap_pixel_count"], 1)

    def test_identity_has_per_region_local_fidelity_of_one(self) -> None:
        image = np.random.default_rng(22).random((54, 60), dtype=np.float32)
        target = base.Roi(4, 50, 5, 55)
        components = self.components(image, target)
        for index, region in enumerate(("central", "fog", "flat")):
            components[f"{region}_gate"][8:38, 2 + index * 15 : 14 + index * 15] = 1.0

        metrics = v20.fixed_region_fidelity(image, image.copy(), target, components)

        for values in metrics.values():
            self.assertAlmostEqual(values["ssim_map_mean"], 1.0, places=7)
            self.assertAlmostEqual(values["ssim_map_p01"], 1.0, places=7)
            self.assertAlmostEqual(values["gradient_magnitude_correlation"], 1.0, places=7)
            self.assertAlmostEqual(values["gradient_rms_retention"], 1.0, places=7)
            self.assertAlmostEqual(values["gradient_relative_rmse"], 0.0, places=7)

    def test_haar_mean_absolute_is_positive_and_falls_after_smoothing(self) -> None:
        rng = np.random.default_rng(23)
        yy, xx = np.indices((80, 84))
        checker = ((yy + xx) % 2).astype(np.float32) * 0.006
        noisy = np.clip(0.4 + checker + rng.normal(0.0, 0.004, checker.shape), 0.0, 1.0)
        smoothed = gaussian_filter(noisy, sigma=0.7)
        mask = np.ones(noisy.shape, dtype=np.float32)

        before = v20.haar_mean_absolute(noisy, mask)
        after = v20.haar_mean_absolute(smoothed, mask)
        before_all = v20.haar_detail_mean_absolute(noisy, mask)
        after_all = v20.haar_detail_mean_absolute(smoothed, mask)

        self.assertGreater(before, 0.0)
        self.assertLess(after, before)
        self.assertGreater(before_all, 0.0)
        self.assertLess(after_all, before_all)

    def test_exact_structure_rows_reject_any_numeric_or_boolean_change(self) -> None:
        baseline = {
            "layer_comparison_rows": [
                {
                    "layer_id": "L1",
                    "output_top_y_px": 1.0,
                    "output_bottom_y_px": 8.0,
                    "output_length_px": 7.0,
                    "output_width_px": 2.0,
                    "dual_evidence_pass": True,
                }
            ],
            "gap_comparison_rows": [
                {
                    "gap_id": "G1",
                    "output_length_px": 6.0,
                    "output_width_px": 3.0,
                    "dual_evidence_pass": True,
                }
            ],
        }
        same = {
            key: [dict(row) for row in rows] for key, rows in baseline.items()
        }
        changed = {
            key: [dict(row) for row in rows] for key, rows in baseline.items()
        }
        changed["layer_comparison_rows"][0]["output_width_px"] += 1e-12

        self.assertTrue(v20.exact_structure_row_drift(same, baseline)["bit_exact_geometry_rows"])
        self.assertFalse(v20.exact_structure_row_drift(changed, baseline)["bit_exact_geometry_rows"])

    def test_weighted_zero_mean_removes_dc_component(self) -> None:
        delta = np.arange(30, dtype=np.float32).reshape(5, 6) / 1000.0
        gate = np.zeros_like(delta)
        gate[1:4, 1:5] = 1.0

        centered = v20.weighted_zero_mean(delta, gate)

        self.assertAlmostEqual(float(np.sum(centered * gate)), 0.0, places=7)

    def test_zero_baseline_noise_metric_has_zero_reduction(self) -> None:
        before = {
            region: {
                "haar_diagonal_mad": 0.0,
                "haar_diagonal_mean_absolute": 0.0,
                "haar_detail_mean_absolute": 0.0,
                "fixed_high_frequency_rms": 0.0,
                "sample_pixel_count": 100,
            }
            for region in ("central", "fog", "flat")
        }
        after = {
            region: {
                "haar_diagonal_mad": 0.0,
                "haar_diagonal_mean_absolute": 0.0,
                "haar_detail_mean_absolute": 0.0,
                "fixed_high_frequency_rms": 0.0,
                "sample_pixel_count": 100,
            }
            for region in ("central", "fog", "flat")
        }

        reduction = v20.fixed_noise_reduction(before, after)

        self.assertTrue(
            all(
                value == 0.0
                for metrics in reduction.values()
                for value in metrics.values()
            )
        )

    def test_uint16_tiff_round_trip_preserves_shape_dtype_and_pixels(self) -> None:
        values = np.random.default_rng(21).integers(
            0, 65536, size=(31, 43), dtype=np.uint16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.tif"
            imwrite(path, values, photometric="minisblack")
            reloaded = np.asarray(imread(path))

        self.assertEqual(reloaded.shape, values.shape)
        self.assertEqual(reloaded.dtype, np.uint16)
        self.assertTrue(np.array_equal(reloaded, values))


if __name__ == "__main__":
    unittest.main()
