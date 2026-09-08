import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import imwrite


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import generative_shape_constraint as shape  # noqa: E402
import generative_postprocess as postprocess  # noqa: E402
import generative_registration as registration  # noqa: E402
import generative_shape_project as project  # noqa: E402
import measurement_quality_optimize as measurement_quality  # noqa: E402


class GenerativeShapeConstraintTests(unittest.TestCase):
    def test_v16_strict_guardrail_accepts_identity_and_rejects_width_drift(self) -> None:
        baseline = {
            "endpoint_shift_abs_p95_px": 0.0,
            "length_delta_abs_p95_px": 0.05,
            "lamella_width_relative_error_p95": 0.0,
            "lamella_dual_evidence_pass_count": 99,
            "interlayer_width_relative_error_p95": 0.0,
            "interlayer_length_abs_error_p95_px": 0.08,
            "interlayer_dual_evidence_pass_count": 96,
            "edge_clarity": 0.30,
            "low_frequency_correlation": 1.0,
            "mid_frequency_correlation": 1.0,
            "gradient_magnitude_correlation": 1.0,
            "lamella_axial_detail_correlation_median": 1.0,
            "lamella_axial_detail_correlation_p10": 1.0,
        }
        passed, _ = measurement_quality.strict_guardrail(baseline, baseline, 1.0)
        self.assertTrue(passed)
        drifted = dict(baseline)
        drifted["lamella_width_relative_error_p95"] = 0.016
        passed, checks = measurement_quality.strict_guardrail(drifted, baseline, 1.0)
        self.assertFalse(passed)
        self.assertFalse(checks["lamella_width_p95_le_1_5_percent"])

    def test_transverse_width_recovers_synthetic_plate(self) -> None:
        image = np.full((80, 90), 0.1, dtype=np.float32)
        image[:, 43:48] = 0.9
        path = np.full(image.shape[0], 45.0, dtype=np.float32)
        width = shape.transverse_width_at_row(image, path, 40, pitch=14.0)
        self.assertIsNotNone(width)
        self.assertAlmostEqual(float(width), 5.0, delta=1.0)

    def test_gap_width_uses_adjacent_measured_envelopes(self) -> None:
        path_a = np.full(40, 10.0, dtype=np.float32)
        path_b = np.full(40, 24.0, dtype=np.float32)
        layers = [
            {"side": "left", "layer_id": "L01", "center_x_px": 10.0, "top_y_px": 5.0,
             "bottom_y_px": 35.0, "width_median_px": 4.0, "path_x": path_a},
            {"side": "left", "layer_id": "L02", "center_x_px": 24.0, "top_y_px": 7.0,
             "bottom_y_px": 32.0, "width_median_px": 6.0, "path_x": path_b},
        ]
        gap = shape.measure_gaps(layers)[0]
        self.assertAlmostEqual(gap["gap_width_px"], 9.0)
        self.assertAlmostEqual(gap["length_px"], 25.0)

    def test_gap_width_follows_curved_paths_over_common_length(self) -> None:
        path_a = np.full(60, 10.0, dtype=np.float32)
        path_b = np.linspace(23.0, 27.0, 60, dtype=np.float32)
        layers = [
            {"side": "left", "layer_id": "L01", "center_x_px": 10.0,
             "top_y_px": 10.0, "bottom_y_px": 50.0, "width_median_px": 4.0,
             "constraint_confidence": 0.9, "path_x": path_a},
            {"side": "left", "layer_id": "L02", "center_x_px": 25.0,
             "top_y_px": 15.0, "bottom_y_px": 45.0, "width_median_px": 6.0,
             "constraint_confidence": 0.8, "path_x": path_b},
        ]
        gap = shape.measure_gaps(layers)[0]
        expected_spacing = np.median(path_b[15:46] - path_a[15:46])
        self.assertAlmostEqual(gap["gap_width_px"], float(expected_spacing - 5.0), delta=0.1)
        self.assertGreater(gap["gap_width_p90_px"], gap["gap_width_p10_px"])
        self.assertAlmostEqual(gap["constraint_confidence"], 0.8)

    def test_gap_exports_matched_raw_width_and_length(self) -> None:
        path_a = np.full(50, 10.0, dtype=np.float32)
        path_b = np.full(50, 24.0, dtype=np.float32)
        layers = [
            {"side": "left", "layer_id": "L01", "center_x_px": 10.0,
             "top_y_px": 5.0, "bottom_y_px": 45.0, "width_median_px": 4.0,
             "raw_validation_top_y_px": 6.0, "raw_validation_bottom_y_px": 44.0,
             "raw_validation_width_median_px": 5.0, "path_x": path_a},
            {"side": "left", "layer_id": "L02", "center_x_px": 24.0,
             "top_y_px": 7.0, "bottom_y_px": 42.0, "width_median_px": 6.0,
             "raw_validation_top_y_px": 8.0, "raw_validation_bottom_y_px": 41.0,
             "raw_validation_width_median_px": 7.0, "path_x": path_b},
        ]
        gap = shape.measure_gaps(layers)[0]
        self.assertAlmostEqual(gap["gap_width_px"], 9.0)
        self.assertAlmostEqual(gap["raw_validation_gap_width_px"], 8.0)
        self.assertAlmostEqual(gap["raw_validation_length_px"], 33.0)

    def test_edge_clarity_metric_prefers_clean_boundary(self) -> None:
        sharp = np.full((80, 90), 0.10, dtype=np.float32)
        sharp[:, 42:48] = 0.90
        blurred = project.gaussian_filter(sharp, sigma=(0.0, 1.4))
        path = np.full(sharp.shape[0], 45.0, dtype=np.float32)
        sharp_score = project.transverse_edge_clarity_at_row(sharp, path, 40, 14.0)
        blurred_score = project.transverse_edge_clarity_at_row(blurred, path, 40, 14.0)
        self.assertIsNotNone(sharp_score)
        self.assertIsNotNone(blurred_score)
        self.assertGreater(float(sharp_score), float(blurred_score))

    def test_pitch_adaptive_centerline_removes_row_jitter(self) -> None:
        y = np.arange(240, dtype=np.float32)
        clean = 45.0 + 0.7 * np.sin(y / 75.0)
        noisy = clean + 0.45 * np.sin(y * 1.7)
        smoothed = shape.smooth_centerline(noisy, pitch=12.0)
        self.assertLess(float(np.std(np.diff(smoothed))), float(np.std(np.diff(noisy))) * 0.2)
        self.assertLess(float(np.max(np.abs(smoothed - clean))), 0.25)

    def test_hard_projection_uses_measured_width_and_endpoints(self) -> None:
        generated = np.full((80, 90), 0.12, dtype=np.float32)
        generated[10:70, 20:70] += 0.18
        path = np.full(generated.shape[0], 45.0, dtype=np.float32)
        layer = {
            "side": "left", "layer_id": "L01", "center_x_px": 45.0,
            "pitch_px": 14.0, "top_y_px": 20.0, "bottom_y_px": 60.0,
            "length_px": 40.0, "width_median_px": 5.0, "path_x": path,
        }
        output, info = project.hard_shape_projection(generated, [layer])
        width = shape.transverse_width_at_row(output, path, 40, pitch=14.0)
        self.assertIsNotNone(width)
        self.assertAlmostEqual(float(width), 5.0, delta=1.0)
        self.assertFalse(info["source_pixel_writeback"])

    def test_blind_guide_adds_axial_detail_without_changing_width(self) -> None:
        generated = np.full((120, 100), 0.10, dtype=np.float32)
        generated[20:101, 48:53] = 0.55
        guide = np.full_like(generated, 0.10)
        axial = 0.48 + 0.16 * np.sin(np.arange(120, dtype=np.float32) / 8.0)
        guide[20:101, 48:53] = axial[20:101, None]
        path = np.full(generated.shape[0], 50.0, dtype=np.float32)
        layer = {
            "side": "left", "layer_id": "L01", "center_x_px": 50.0,
            "pitch_px": 16.0, "top_y_px": 20.0, "bottom_y_px": 100.0,
            "length_px": 80.0, "width_median_px": 5.0, "path_x": path,
        }
        output, info = project.hard_shape_projection(
            generated, [layer], detail_guide=guide, detail_weight=0.65
        )
        guide_profile = project.layer_axial_contrast_profile(guide, layer)[35:86]
        generated_profile = project.layer_axial_contrast_profile(generated, layer)[35:86]
        output_profile = project.layer_axial_contrast_profile(output, layer)[35:86]
        generated_corr = project.safe_correlation(guide_profile, generated_profile)
        output_corr = project.safe_correlation(guide_profile, output_profile)
        width = shape.transverse_width_at_row(output, path, 60, pitch=16.0)
        self.assertIsNotNone(output_corr)
        self.assertGreater(float(output_corr), float(generated_corr or 0.0) + 0.50)
        self.assertAlmostEqual(float(width), 5.0, delta=1.0)
        self.assertEqual(info["geometry_source"],
                         "measured centerlines, endpoints, widths and interlayer gaps only")

    def test_detail_guide_shape_must_match_generated(self) -> None:
        generated = np.zeros((20, 20), dtype=np.float32)
        guide = np.zeros((21, 20), dtype=np.float32)
        path = np.full(20, 10.0, dtype=np.float32)
        layer = {
            "side": "left", "layer_id": "L01", "center_x_px": 10.0,
            "pitch_px": 8.0, "top_y_px": 3.0, "bottom_y_px": 16.0,
            "length_px": 13.0, "width_median_px": 3.0, "path_x": path,
        }
        with self.assertRaises(ValueError):
            project.hard_shape_projection(generated, [layer], detail_guide=guide)

    def test_v15_measurement_core_is_independent_of_generated_pixels(self) -> None:
        yy, xx = np.indices((120, 100))
        generated_a = np.where((yy + xx) % 2 == 0, 0.10, 0.85).astype(np.float32)
        guide = np.full_like(generated_a, 0.08)
        guide[20:101, 48:53] = 0.62
        guide[57:63, 48:53] = 0.38
        path = np.full(generated_a.shape[0], 50.0, dtype=np.float32)
        layer = {
            "side": "left", "layer_id": "L01", "center_x_px": 50.0,
            "pitch_px": 16.0, "top_y_px": 20.0, "bottom_y_px": 100.0,
            "length_px": 80.0, "width_median_px": 5.0, "path_x": path,
        }
        expected_core = project.foreground_structure_core([layer], generated_a.shape)
        generated_b = generated_a.copy()
        generated_b[expected_core] = 0.95 - generated_b[expected_core]
        output_a, info_a, core = project.structure_carrier_projection(
            generated_a, guide, [layer]
        )
        output_b, _, _ = project.structure_carrier_projection(
            generated_b, guide, [layer]
        )
        self.assertTrue(np.allclose(output_a[core], output_b[core], atol=1e-7, rtol=0.0))
        self.assertEqual(info_a["generated_pixel_weight_inside_measurement_core"], 0.0)
        self.assertFalse(info_a["spatial_resampling_inside_measurement_core"])
        self.assertFalse(info_a["analytic_lamella_replacement_inside_measurement_core"])

    def test_v15_measurement_core_rejects_mismatched_guide_shape(self) -> None:
        generated = np.zeros((20, 20), dtype=np.float32)
        guide = np.zeros((21, 20), dtype=np.float32)
        path = np.full(20, 10.0, dtype=np.float32)
        layer = {
            "side": "left", "layer_id": "L01", "center_x_px": 10.0,
            "pitch_px": 8.0, "top_y_px": 3.0, "bottom_y_px": 16.0,
            "length_px": 13.0, "width_median_px": 3.0, "path_x": path,
        }
        with self.assertRaises(ValueError):
            project.structure_carrier_projection(generated, guide, [layer])

    def test_postprocess_size_lock_matches_source_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.tif"
            processed_path = root / "processed.png"
            imwrite(source_path, np.zeros((80, 110), dtype=np.uint16))
            Image.fromarray(np.zeros((40, 55), dtype=np.uint8), mode="L").save(processed_path)
            output_path, audit = postprocess.lock_to_source_dimensions(
                processed_path, source_path, root
            )
            with Image.open(output_path) as output:
                self.assertEqual(output.size, (110, 80))
            self.assertTrue(audit["strict_dimension_match"])
            self.assertEqual(audit["resampling"], "PIL Lanczos")

    def test_generated_image_is_resized_before_postprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.tif"
            generated_path = root / "generated.png"
            imwrite(source_path, np.zeros((80, 110), dtype=np.uint16))
            Image.fromarray(np.zeros((40, 55), dtype=np.uint8), mode="L").save(generated_path)
            output_path, audit = postprocess.resize_generated_to_source_dimensions(
                generated_path, source_path, root
            )
            with Image.open(output_path) as output:
                self.assertEqual(output.size, (110, 80))
            self.assertEqual(
                audit["stage_order"],
                "generation -> source-size resize -> post-processing -> geometry projection",
            )
            self.assertFalse(audit["postprocessing_applied_before_resize"])

    def test_registration_coordinate_map_hits_every_envelope_anchor(self) -> None:
        model = {
            "horizontal": {
                "fixed": {
                    "left_outer": 25.0, "left_inner": 70.0,
                    "right_inner": 100.0, "right_outer": 145.0,
                },
                "moving": {
                    "left_outer": 20.0, "left_inner": 73.0,
                    "right_inner": 104.0, "right_outer": 153.0,
                },
            },
            "vertical": {
                "fixed": {"top": 30.0, "bottom": 90.0},
                "moving": {"top": 22.0, "bottom": 84.0},
            },
        }
        input_y, input_x, _ = registration.registration_coordinates(model, (120, 180))
        for fixed, moving in zip((25, 70, 100, 145), (20, 73, 104, 153)):
            self.assertAlmostEqual(float(input_x[fixed]), float(moving), places=5)
        self.assertAlmostEqual(float(input_y[30]), 22.0, places=5)
        self.assertAlmostEqual(float(input_y[90]), 84.0, places=5)

    def test_registration_preserves_canvas_and_background(self) -> None:
        image = np.linspace(0.0, 1.0, 120 * 180, dtype=np.float32).reshape(120, 180)
        model = {
            "horizontal": {
                "fixed": {
                    "left_outer": 25.0, "left_inner": 70.0,
                    "right_inner": 100.0, "right_outer": 145.0,
                },
                "moving": {
                    "left_outer": 20.0, "left_inner": 73.0,
                    "right_inner": 104.0, "right_outer": 153.0,
                },
            },
            "vertical": {
                "fixed": {"top": 30.0, "bottom": 90.0},
                "moving": {"top": 22.0, "bottom": 84.0},
            },
        }
        output, mapping = registration.apply_registration(image, model)
        self.assertEqual(output.shape, image.shape)
        self.assertAlmostEqual(float(output[0, 0]), float(image[0, 0]), places=7)
        self.assertAlmostEqual(float(output[-1, -1]), float(image[-1, -1]), places=7)
        self.assertTrue(mapping["background_outside_roi_unchanged"])

    def test_all_registration_variants_preserve_dimensions_and_finite_pixels(self) -> None:
        image = np.linspace(0.0, 1.0, 120 * 180, dtype=np.float32).reshape(120, 180)
        model = {
            "coarse_phase_correlation": {
                "shift_to_apply_x_px": -3.0,
                "shift_to_apply_y_px": 6.0,
            },
            "horizontal": {
                "fixed": {
                    "left_outer": 25.0, "left_inner": 70.0,
                    "right_inner": 100.0, "right_outer": 145.0,
                },
                "moving": {
                    "left_outer": 20.0, "left_inner": 73.0,
                    "right_inner": 104.0, "right_outer": 153.0,
                },
            },
            "vertical": {
                "fixed": {"top": 30.0, "bottom": 90.0},
                "moving": {"top": 22.0, "bottom": 84.0},
            },
        }
        for method in registration.REGISTRATION_METHODS:
            with self.subTest(method=method):
                output, mapping = registration.apply_registration(
                    image, model, method=method
                )
                self.assertEqual(output.shape, image.shape)
                self.assertTrue(np.all(np.isfinite(output)))
                self.assertEqual(mapping["registration_method"], method)

    def test_final_horizontal_envelope_audit_rejects_large_residual(self) -> None:
        image = np.zeros((120, 180), dtype=np.float32)
        image[30:91, 25:71] = 0.4
        image[30:91, 100:146] = 0.4
        image[30:91, 68:73] = 0.9
        image[30:91, 98:103] = 0.9
        constraints = []
        for side, centers in (("left", range(34, 67, 8)), ("right", range(108, 141, 8))):
            for center in centers:
                constraints.append({
                    "side": side, "center_x_px": float(center), "pitch_px": 8.0,
                    "top_y_px": 30.0, "bottom_y_px": 90.0, "length_px": 60.0,
                })
        model = {
            "horizontal": {"fixed": {
                "left_outer": 24.5, "left_inner": 70.0,
                "right_inner": 100.0, "right_outer": 145.5,
            }},
            "profile_intervals": {"fixed_y_interval": [40, 80]},
        }
        audit = registration.final_horizontal_envelope_audit(image, constraints, model)
        self.assertTrue(audit["summary"]["guardrail_pass"])
        shifted = np.roll(image, 8, axis=1)
        shifted_audit = registration.final_horizontal_envelope_audit(
            shifted, constraints, model
        )
        self.assertFalse(shifted_audit["summary"]["guardrail_pass"])


if __name__ == "__main__":
    unittest.main()
