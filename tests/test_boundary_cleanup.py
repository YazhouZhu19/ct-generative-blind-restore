import sys
import unittest
from pathlib import Path

import numpy as np


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import boundary_cleanup as cleanup  # noqa: E402
import length_optimize as length  # noqa: E402
import pipeline as base  # noqa: E402


def synthetic_references(height: int) -> list[dict]:
    references = []
    for index, center in enumerate((28.0, 40.0, 64.0, 76.0)):
        side = "left" if center < 50 else "right"
        endpoint = length.EndpointMeasurement(
            top=30.0,
            bottom=90.0,
            length=60.0,
            top_spread=0.2,
            bottom_spread=0.2,
            uncertainty=0.35,
            top_snr=20.0,
            bottom_snr=20.0,
        )
        references.append({
            "side": side,
            "layer_id": f"{side[0].upper()}{index:02d}",
            "center": center,
            "pitch": 12.0,
            "path_x": np.full(height, center, dtype=np.float32),
            "raw": endpoint,
        })
    return references


class BoundaryCleanupTests(unittest.TestCase):
    def test_endpoint_envelopes_separate_exterior_and_interior(self) -> None:
        shape = (120, 104)
        target = base.Roi(8, 112, 8, 96)
        masks, info = cleanup.build_endpoint_envelope_masks(
            shape, synthetic_references(shape[0]), target
        )
        self.assertEqual(masks["boundary"].shape, (104, 88))
        self.assertGreater(float(masks["exterior"][12, 20]), 0.0)
        self.assertGreater(float(masks["interior"][52, 20]), 0.8)
        self.assertFalse(info["raw_pixel_writeback"])
        self.assertFalse(info["reference_pixel_writeback"])

    def test_cleanup_reduces_axial_noise_and_protects_endpoint_core(self) -> None:
        rng = np.random.default_rng(91)
        shape = (120, 104)
        image = np.clip(rng.normal(0.16, 0.018, shape), 0.0, 1.0).astype(np.float32)
        for center in (28, 40, 64, 76):
            image[30:91, center - 1:center + 2] += 0.42
        image = np.clip(image, 0.0, 1.0)
        target = base.Roi(8, 112, 8, 96)
        references = synthetic_references(shape[0])
        masks, _ = cleanup.build_endpoint_envelope_masks(shape, references, target)
        before = cleanup.boundary_cleanliness_metrics(image, target, masks)
        cleaned, operation, _ = cleanup.clean_candidate(
            image,
            references,
            target,
            masks,
            fine_strength=0.65,
            halo_strength=1.8,
            boundary_sharpen=0.0,
        )
        after = cleanup.boundary_cleanliness_metrics(cleaned, target, masks)
        self.assertLess(after["interior_axial_noise_rms"], before["interior_axial_noise_rms"])
        self.assertEqual(float(cleaned[30, 28]), float(image[30, 28]))
        self.assertEqual(float(cleaned[90, 28]), float(image[90, 28]))
        self.assertLessEqual(
            float(np.max(np.abs(cleaned - image))),
            operation["change_cap_normalized"] + 1e-7,
        )
        self.assertFalse(operation["raw_pixel_writeback"])
        self.assertFalse(operation["reference_pixel_writeback"])


if __name__ == "__main__":
    unittest.main()
