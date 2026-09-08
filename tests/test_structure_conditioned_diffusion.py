import sys
import unittest
from pathlib import Path

import numpy as np
import torch


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import pipeline as base  # noqa: E402
import structure_conditioned_diffusion as diffusion  # noqa: E402


class StructureConditionedDiffusionTests(unittest.TestCase):
    def synthetic_layer(self) -> dict:
        path = np.full(64, 32.0, dtype=np.float32)
        return {
            "side": "left",
            "layer_id": "L01",
            "center_x_px": 32.0,
            "pitch_px": 12.0,
            "top_y_px": 10.0,
            "bottom_y_px": 54.0,
            "width_median_px": 4.0,
            "constraint_confidence": 0.9,
            "path_x": path,
        }

    def test_condition_maps_protect_boundaries_and_endpoints(self) -> None:
        target = base.Roi(0, 64, 0, 64)
        maps = diffusion.build_structure_maps((64, 64), target, [self.synthetic_layer()])
        self.assertGreater(float(maps["centerline"][30, 32]), 0.9)
        self.assertGreater(float(maps["endpoint"][10, 32]), 0.8)
        boundary_x = 34
        self.assertGreater(float(maps["boundary"][30, boundary_x]), 0.7)
        self.assertLess(float(maps["allowed"][10, 32]), 0.1)
        self.assertLess(float(maps["allowed"][30, boundary_x]), 0.2)

    def test_zero_initialized_generator_starts_at_carrier(self) -> None:
        model = diffusion.StructureConditionedResidualUNet(10, features=8)
        noisy = torch.randn(2, 1, 32, 32)
        conditions = torch.randn(2, 10, 32, 32)
        time = torch.tensor([0.2, 0.8])
        residual = model(noisy, conditions, time)
        self.assertEqual(tuple(residual.shape), (2, 1, 32, 32))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_structure_losses_penalize_shifted_width_and_endpoints(self) -> None:
        carrier = torch.zeros(1, 1, 32, 32)
        carrier[:, :, 7:25, 14:18] = 1.0
        maps = torch.zeros(1, 6, 32, 32)
        maps[:, 0:1, 7:25, 14:18] = 1.0
        maps[:, 2:3, 7:25, 13:19] = 1.0
        maps[:, 3:4, 5:9, 12:20] = 1.0
        maps[:, 3:4, 23:27, 12:20] = 1.0
        maps[:, 5:6] = maps[:, 0:1]
        aligned = diffusion.structure_detail_losses(carrier, carrier, carrier, maps)
        shifted = torch.roll(carrier, shifts=(2, 2), dims=(-2, -1))
        drifted = diffusion.structure_detail_losses(shifted, carrier, carrier, maps)
        aligned_geometry = sum(aligned[name] for name in (
            "edge_x", "edge_y", "width_profile", "endpoint_profile", "boundary_field"
        ))
        drifted_geometry = sum(drifted[name] for name in (
            "edge_x", "edge_y", "width_profile", "endpoint_profile", "boundary_field"
        ))
        self.assertGreater(float(drifted_geometry), float(aligned_geometry))

    def test_postprocess_caps_residual_and_protects_boundary(self) -> None:
        carrier = np.full((24, 24), 0.4, dtype=np.float32)
        generated = np.full((24, 24), 0.8, dtype=np.float32)
        allowed = np.ones_like(carrier)
        allowed[:, 11:13] = 0.02
        maps = {
            "allowed": allowed,
            "boundary": (allowed < 0.5).astype(np.float32),
            "endpoint": np.zeros_like(carrier),
        }
        output, _ = diffusion.postprocess_generated(generated, carrier, maps, 0.03)
        delta = output - carrier
        self.assertLessEqual(float(np.max(np.abs(delta))), 0.030001)
        self.assertLess(float(np.max(np.abs(delta[:, 11:13]))), 0.001)
        self.assertGreater(float(np.mean(delta[:, :8])), 0.02)

    def test_tile_starts_cover_final_pixel(self) -> None:
        starts = diffusion.tile_starts(641, 192, 40)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + 192, 641)
        self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))


if __name__ == "__main__":
    unittest.main()
