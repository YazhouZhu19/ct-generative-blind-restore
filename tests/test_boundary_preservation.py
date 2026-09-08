import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import length_optimize as length  # noqa: E402
import pipeline as base  # noqa: E402
import quality_optimize as quality  # noqa: E402


class BoundaryPreservationTests(unittest.TestCase):
    def test_original_directional_continuity_keeps_change_cap(self) -> None:
        rng = np.random.default_rng(17)
        observed = rng.normal(0.35, 0.03, (64, 80)).astype(np.float32)
        current = np.clip(observed * 0.8 + 0.1, 0.0, 1.0).astype(np.float32)
        output = base.directional_continuity(
            observed,
            current,
            sigma_y=3.0,
            strength=0.7,
            change_cap=0.08,
        )
        self.assertLessEqual(float(np.max(np.abs(output - observed))), 0.080001)

    def test_geometry_warp_reduces_endpoint_error_without_raw_pixels(self) -> None:
        h, w = 120, 60
        path = np.full(h, 30.0, dtype=np.float32)
        source = np.zeros((h, w), dtype=np.float32)
        source[30:90, 27:34] = 1.0
        source = gaussian_filter(source, sigma=(1.0, 0.7))
        enhanced = np.zeros((h, w), dtype=np.float32)
        enhanced[32:90, 27:34] = 1.0
        enhanced = gaussian_filter(enhanced, sigma=(1.0, 0.7))
        raw = length.measure_endpoints(source, 30.0, 12.0, (20, 45), (75, 100), path_x=path)
        refs = [{
            "side": "left",
            "layer_id": "L01",
            "center": 30.0,
            "pitch": 12.0,
            "path_x": path,
            "raw": raw,
        }]
        before, _ = quality.boundary_geometry(enhanced, refs, (20, 45), (75, 100))
        corrected, info = quality.endpoint_geometry_warp(
            enhanced, refs, (20, 45), (75, 100), iterations=1, radius_y=1, radius_x=1, max_shift=0.35
        )
        after, _ = quality.boundary_geometry(corrected, refs, (20, 45), (75, 100))
        self.assertLess(after["endpoint_shift_abs_max_px"], before["endpoint_shift_abs_max_px"])
        self.assertFalse(info["raw_pixel_writeback"])

    def test_boundary_guardrail_checks_tail_and_worst_case(self) -> None:
        passing = {
            "endpoint_shift_abs_p95_px": 0.2,
            "endpoint_shift_abs_max_px": 0.7,
            "length_delta_abs_p95_px": 0.4,
            "length_delta_abs_max_px": 0.9,
        }
        self.assertTrue(quality.boundary_guardrail(passing))
        failing = dict(passing, endpoint_shift_abs_max_px=0.8)
        self.assertFalse(quality.boundary_guardrail(failing))

    def test_fog_cleanup_reduces_flat_noise_and_protects_endpoint_core(self) -> None:
        rng = np.random.default_rng(23)
        h, w = 160, 120
        image = rng.normal(0.25, 0.018, (h, w)).astype(np.float32)
        image[35:126, 57:64] += 0.40
        image = np.clip(image, 0.0, 1.0)
        path = np.full(h, 60.0, dtype=np.float32)
        raw = length.EndpointMeasurement(
            top=35.0,
            bottom=125.0,
            length=90.0,
            top_spread=0.2,
            bottom_spread=0.2,
            uncertainty=0.3,
            top_snr=10.0,
            bottom_snr=10.0,
        )
        refs = [{
            "side": "left",
            "layer_id": "L01",
            "center": 60.0,
            "pitch": 12.0,
            "path_x": path,
            "raw": raw,
        }]
        target = base.Roi(10, 150, 10, 110)
        cleaned, info, gradient = quality.structure_aware_fog_cleanup(
            image, refs, target, fine_strength=0.12, haze_strength=0.015
        )
        before = quality.low_structure_roughness(image, target, gradient)
        after = quality.low_structure_roughness(cleaned, target, gradient)
        self.assertLess(after, before)
        self.assertEqual(float(cleaned[35, 60]), float(image[35, 60]))
        self.assertEqual(float(cleaned[125, 60]), float(image[125, 60]))
        self.assertFalse(info["raw_pixel_writeback"])


if __name__ == "__main__":
    unittest.main()
