import sys
import unittest
from pathlib import Path

import numpy as np


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import boundary_cleanup as boundary  # noqa: E402
import length_optimize as length  # noqa: E402
import pipeline as base  # noqa: E402
import residual_denoise as residual  # noqa: E402


def references(height: int) -> list[dict]:
    out = []
    for index, center in enumerate((28.0, 40.0, 88.0, 100.0)):
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
        out.append({
            "side": "left" if center < 64 else "right",
            "layer_id": f"T{index:02d}",
            "center": center,
            "pitch": 12.0,
            "path_x": np.full(height, center, dtype=np.float32),
            "raw": endpoint,
        })
    return out


class ResidualDenoiseTests(unittest.TestCase):
    def test_soft_roi_mask_has_feathered_interior(self) -> None:
        target = base.Roi(8, 120, 8, 120)
        roi = base.Roi(32, 96, 48, 80)
        mask = residual.soft_roi_mask(target, roi, ramp=8)
        self.assertEqual(mask.shape, (112, 112))
        self.assertEqual(float(mask[24, 40]), 0.0)
        self.assertEqual(float(mask[40, 56]), 1.0)

    def test_hybrid_residual_cleanup_respects_cap_and_endpoints(self) -> None:
        rng = np.random.default_rng(101)
        shape = (128, 128)
        image = np.clip(rng.normal(0.20, 0.018, shape), 0.0, 1.0).astype(np.float32)
        for center in (28, 40, 88, 100):
            image[32:97, center - 1:center + 2] += 0.42
        image = np.clip(image, 0.0, 1.0)
        target = base.Roi(8, 120, 8, 120)
        central = base.Roi(32, 96, 48, 80)
        refs = references(shape[0])
        masks, _ = boundary.build_endpoint_envelope_masks(shape, refs, target)
        components, method = residual.prepare_components(image, refs, target, central, masks)
        cleaned, operation = residual.apply_candidate(
            image,
            target,
            components,
            method["estimated_noise_sigma_normalized"],
            lamella_strength=0.60,
            central_strength=0.70,
        )
        self.assertEqual(float(cleaned[32, 28]), float(image[32, 28]))
        self.assertEqual(float(cleaned[96, 28]), float(image[96, 28]))
        self.assertLessEqual(
            float(np.max(np.abs(cleaned - image))),
            operation["change_cap_normalized"] + 1e-7,
        )
        self.assertFalse(method["raw_pixel_writeback"])
        self.assertFalse(operation["reference_pixel_writeback"])


if __name__ == "__main__":
    unittest.main()
