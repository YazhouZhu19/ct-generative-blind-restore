#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import imread, imwrite


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from size_lock import lock_generated_to_source  # noqa: E402


class SizeLockTests(unittest.TestCase):
    def test_resizes_before_any_other_operation_and_preserves_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.arange(60 * 84, dtype=np.uint16).reshape(60, 84)
            generated = np.arange(20 * 28, dtype=np.uint8).reshape(20, 28)
            source_path = root / "source.tif"
            generated_path = root / "generated.png"
            outdir = root / "output"
            imwrite(source_path, source, photometric="minisblack")
            Image.fromarray(generated, mode="L").save(generated_path)

            png_path, tif_path, manifest = lock_generated_to_source(
                source_path, generated_path, outdir
            )

            self.assertEqual(np.asarray(Image.open(png_path)).shape, source.shape)
            self.assertEqual(np.asarray(imread(tif_path)).shape, source.shape)
            self.assertTrue(manifest["resized"])
            self.assertEqual(manifest["resampling"], "PIL Lanczos")
            self.assertFalse(manifest["cropping_used"])
            self.assertFalse(manifest["enhancement_before_size_lock"])
            disk_manifest = json.loads(
                (outdir / "size_lock_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(disk_manifest["output_dimensions"], {"width": 84, "height": 60})

    def test_same_size_is_pixel_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.zeros((48, 72), dtype=np.uint16)
            generated = np.arange(48 * 72, dtype=np.uint8).reshape(48, 72)
            source_path = root / "source.tif"
            generated_path = root / "generated.png"
            imwrite(source_path, source, photometric="minisblack")
            Image.fromarray(generated, mode="L").save(generated_path)

            png_path, _, manifest = lock_generated_to_source(
                source_path, generated_path, root / "output"
            )

            self.assertTrue(np.array_equal(np.asarray(Image.open(png_path)), generated))
            self.assertFalse(manifest["resized"])
            self.assertEqual(manifest["resampling"], "none")

    def test_rejects_unsafe_aspect_ratio_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.tif"
            generated_path = root / "generated.png"
            imwrite(source_path, np.zeros((60, 90), dtype=np.uint16))
            Image.fromarray(np.zeros((40, 40), dtype=np.uint8), mode="L").save(
                generated_path
            )
            with self.assertRaisesRegex(ValueError, "aspect-ratio mismatch"):
                lock_generated_to_source(source_path, generated_path, root / "output")


if __name__ == "__main__":
    unittest.main()
