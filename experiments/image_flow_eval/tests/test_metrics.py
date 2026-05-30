from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.image_flow_eval.image_flow_eval.metrics import (
    compute_blank_check,
    compute_psnr,
    compute_visual_diff,
)


class ImageMetricTest(unittest.TestCase):
    def test_blank_image_fails_blank_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "blank.png"
            Image.new("RGB", (16, 16), "white").save(image_path)

            result = compute_blank_check(
                image_path,
                thresholds={
                    "min_pixel_stddev": 3.0,
                    "min_non_background_ratio": 0.001,
                    "min_file_bytes": 1.0,
                },
            )

            self.assertEqual(result.status, "failed")

    def test_identical_images_have_zero_visual_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            reference = np.zeros((8, 8, 3), dtype=np.uint8)
            candidate = reference.copy()

            result = compute_visual_diff(
                prefix="reference",
                reference_rgb=reference,
                candidate_rgb=candidate,
                selected_mask=None,
                output_dir=output_dir,
                case_id="case",
                variant_id="baseline",
                thresholds={"max_diff_pixel_ratio": 0.0, "max_mean_abs_error": 0.0},
                visual_delta_threshold=8,
                extra_details={},
            )

            self.assertEqual(result.status, "passed")
            self.assertEqual(result.value, 0.0)
            self.assertTrue(
                (output_dir / "case" / "baseline" / "reference_diff.png").exists()
            )

    def test_masked_inside_change_preserves_outside_psnr(self) -> None:
        reference = np.zeros((4, 4, 3), dtype=np.uint8)
        candidate = reference.copy()
        candidate[0, 0] = [255, 255, 255]
        selected_outside = np.ones((4, 4), dtype=np.bool_)
        selected_outside[0, 0] = False

        result = compute_psnr(
            prefix="outside_mask_preservation",
            reference_rgb=reference,
            candidate_rgb=candidate,
            selected_mask=selected_outside,
            thresholds={"min_psnr": 99.0},
            extra_details={},
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.value, 100.0)


if __name__ == "__main__":
    unittest.main()
