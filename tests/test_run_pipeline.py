#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import run_pipeline as runner  # noqa: E402


class PipelineRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "schema_version": 1,
            "reference_size": {"width": 200, "height": 100},
            "scale_coordinates_to_input": True,
        }

    def test_scales_roi_in_both_axes(self) -> None:
        actual = runner.scaled_roi([10, 80, 20, 180], (200, 100), self.config)
        self.assertEqual(actual, [20, 160, 10, 90])

    def test_scales_vertical_range(self) -> None:
        self.assertEqual(runner.scaled_range([20, 70], 250, self.config), [50, 175])

    def test_coordinate_scales_are_axis_specific(self) -> None:
        self.assertEqual(runner.coordinate_scales((200, 100), self.config), (2.0, 0.5))

    def test_rejects_nonempty_output_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            (outdir / "user-file.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.prepare_outdir(outdir, overwrite=False)

    def test_overwrite_removes_only_pipeline_managed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            (outdir / "01_source_sized_generation").mkdir()
            (outdir / "01_source_sized_generation" / "old.txt").write_text("old")
            (outdir / "FINAL_enhanced_10x10.png").write_text("old")
            (outdir / "user-file.txt").write_text("keep")
            runner.prepare_outdir(outdir, overwrite=True)
            self.assertFalse((outdir / "01_source_sized_generation").exists())
            self.assertFalse((outdir / "FINAL_enhanced_10x10.png").exists())
            self.assertTrue((outdir / "user-file.txt").exists())

    def test_loads_bundled_schema(self) -> None:
        config = runner.load_config(runner.DEFAULT_CONFIG)
        self.assertEqual(config["schema_version"], 1)
        self.assertIn("structure_guidance", config)


if __name__ == "__main__":
    unittest.main()
