import sys
import unittest
from pathlib import Path

import numpy as np
import torch


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import pipeline as base  # noqa: E402
import sota_geometry_blind as sota  # noqa: E402
import length_optimize as length  # noqa: E402


class SotaGeometryBlindTests(unittest.TestCase):
    def test_model_predicts_finite_mean_and_variance(self) -> None:
        model = sota.AdaptiveVisibleUNet(features=4)
        image = torch.rand(2, 1, 32, 40)
        mean, variance = model(image)
        self.assertEqual(mean.shape, image.shape)
        self.assertEqual(variance.shape, image.shape)
        self.assertTrue(bool(torch.isfinite(mean).all()))
        self.assertTrue(bool((variance > 0).all()))

    def test_all_blind_phases_cover_every_interior_pixel_once(self) -> None:
        image = torch.rand(1, 1, 20, 24)
        coverage = torch.zeros_like(image, dtype=torch.int32)
        for phase in range(16):
            hidden, mask = sota.fixed_sublattice_blind_mask(image, phase)
            coverage += mask.to(torch.int32)
            self.assertTrue(bool(torch.equal(hidden[~mask], image[~mask])))
        self.assertTrue(bool((coverage[..., 1:-1, 1:-1] == 1).all()))
        self.assertTrue(bool((coverage[..., 0, :] == 0).all()))
        self.assertTrue(bool((coverage[..., -1, :] == 0).all()))

    def test_safe_prior_is_clipped_and_keeps_raw_edges_dominant(self) -> None:
        source = np.full((96, 128), 0.25, dtype=np.float32)
        source[:, 48:80] = 0.75
        generated = np.linspace(0.0, 1.0, source.size, dtype=np.float32).reshape(source.shape)
        target = base.Roi(8, 88, 8, 120)
        prior, info = sota.safe_low_frequency_prior(source, generated, target)
        cap = info["change_cap_normalized"]
        self.assertLessEqual(float(np.max(np.abs(prior - source))), cap + 1e-6)
        self.assertFalse(info["generated_pixel_writeback"])
        raw_edge = float(np.mean(np.abs(np.diff(source, axis=1))[:, 47:80]))
        prior_edge = float(np.mean(np.abs(np.diff(prior, axis=1))[:, 47:80]))
        self.assertGreater(prior_edge, 0.95 * raw_edge)

    def test_geometry_loss_penalizes_shifted_geometry(self) -> None:
        source = torch.zeros(1, 1, 64, 64)
        source[..., 12:52, 27:36] = 1.0
        same, _ = sota.geometry_consistency_loss(source, source)
        shifted = torch.roll(source, shifts=(2, 3), dims=(-2, -1))
        changed, _ = sota.geometry_consistency_loss(shifted, source)
        self.assertLess(float(same), 1e-8)
        self.assertGreater(float(changed), float(same) + 1e-3)

    def test_transverse_anchor_marks_detected_raw_centers(self) -> None:
        image = np.zeros((80, 120), dtype=np.float32)
        image[:, 30] = 0.8
        image[:, 55] = 1.0
        image[:, 82] = 0.7
        target = base.Roi(5, 75, 10, 110)
        roi = base.Roi(10, 70, 15, 105)
        anchor, info = sota.transverse_geometry_anchor(image, target, (roi,), radius=0)
        positions = info["positions_x_px"]["roi_0"]
        self.assertEqual(info["anchored_layer_count"], len(positions))
        self.assertGreaterEqual(len(positions), 3)
        for position in positions:
            self.assertTrue(bool((anchor[:, position - target.x0] == 1.0).all()))
        self.assertLess(float(anchor.mean()), 0.10)

    def test_geometry_locked_directional_filter_protects_endpoints(self) -> None:
        rng = np.random.default_rng(121)
        source = np.clip(rng.normal(0.18, 0.02, (128, 96)), 0.0, 1.0).astype(np.float32)
        source[32:97, 46:51] += 0.55
        source = np.clip(source, 0.0, 1.0)
        current = source.copy()
        target = base.Roi(8, 120, 8, 88)
        path = np.full(source.shape[0], 48.0, dtype=np.float32)
        endpoint = length.EndpointMeasurement(
            top=32.0,
            bottom=96.0,
            length=64.0,
            top_spread=0.2,
            bottom_spread=0.2,
            uncertainty=0.35,
            top_snr=20.0,
            bottom_snr=20.0,
        )
        references = [{
            "side": "left",
            "layer_id": "L01",
            "center": 48.0,
            "pitch": 12.0,
            "path_x": path,
            "raw": endpoint,
        }]
        filtered, info = sota.geometry_locked_directional_continuity(
            source,
            current,
            target,
            references,
            strength=0.70,
            sigma_y=3.0,
            change_cap=0.04,
        )
        self.assertEqual(float(filtered[32, 48]), float(current[32, 48]))
        self.assertEqual(float(filtered[96, 48]), float(current[96, 48]))
        self.assertFalse(info["raw_pixel_writeback"])
        self.assertFalse(info["generated_pixel_writeback"])


if __name__ == "__main__":
    unittest.main()
