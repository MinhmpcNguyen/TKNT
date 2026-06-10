from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from experiments.image_flow_eval.image_flow_eval.metrics import (
    DetectionBox,
    compute_blank_check,
    compute_detector_layout,
    compute_detector_spatial_relations,
    compute_layout_depth_structure,
    compute_layout_dino_object_similarity,
    compute_layout_dino_patch_similarity,
    compute_layout_edge_structure,
    compute_layout_gate_score,
    compute_provider_observability_metrics,
    compute_psnr,
    compute_structural_layout_score,
    compute_visual_diff,
)
from experiments.image_flow_eval.image_flow_eval.io import load_manifest
from experiments.image_flow_eval.image_flow_eval.types import (
    MetricResult,
    ProviderResult,
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

    def test_manifest_accepts_local_layout_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                """
                {
                  "defaults": {
                    "metrics": [
                      "layout_lpips",
                      "layout_dino_similarity",
                      "layout_dino_object_similarity",
                      "layout_dino_patch_similarity",
                      "layout_edge_structure",
                      "layout_depth_structure",
                      "structural_layout_score",
                      "layout_gate_score",
                      "detector_layout",
                      "detector_spatial_relations",
                      "preset_fidelity"
                    ],
                    "preset_selection": {
                      "style": "japandi",
                      "lighting": "daylight_6",
                      "scenery": "garden"
                    }
                  },
                  "cases": [{"id": "case", "variants": [{"id": "baseline"}]}]
                }
                """,
                encoding="utf-8",
            )

            manifest = load_manifest(path)

        self.assertEqual(
            manifest.metrics,
            (
                "layout_lpips",
                "layout_dino_similarity",
                "layout_dino_object_similarity",
                "layout_dino_patch_similarity",
                "layout_edge_structure",
                "layout_depth_structure",
                "structural_layout_score",
                "layout_gate_score",
                "detector_layout",
                "detector_spatial_relations",
                "preset_fidelity",
            ),
        )
        self.assertIsNotNone(manifest.cases[0].preset_selection)
        preset_selection = manifest.cases[0].preset_selection
        assert preset_selection is not None
        self.assertEqual(
            preset_selection.to_mapping(),
            {
                "style": "japandi",
                "lighting": "daylight_6",
                "scenery": "garden",
            },
        )

    def test_dino_object_similarity_uses_visible_object_bboxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.png"
            candidate_path = root / "candidate.png"
            Image.new("RGB", (32, 32), "white").save(reference_path)
            Image.new("RGB", (32, 32), "white").save(candidate_path)
            payload = {
                "visibleObjects": [
                    {
                        "id": "sofa",
                        "type": "sofa",
                        "screenBboxPx": {
                            "minX": 4,
                            "minY": 4,
                            "maxX": 20,
                            "maxY": 20,
                        },
                    },
                    {
                        "id": "chair",
                        "type": "chair",
                        "screenBboxPx": {
                            "minX": 20,
                            "minY": 4,
                            "width": 8,
                            "height": 10,
                        },
                    },
                ]
            }

            def fake_embeddings(_images: object) -> np.ndarray:
                return np.asarray(
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_dino_embeddings_for_images",
                side_effect=fake_embeddings,
            ):
                result = compute_layout_dino_object_similarity(
                    reference_image=reference_path,
                    candidate_image=candidate_path,
                    source_payload=payload,
                    thresholds={"min_layout_dino_object_similarity": 0.4},
                )

        self.assertEqual(result.status, "passed")
        self.assertAlmostEqual(result.value or 0.0, 0.5)
        assert result.details is not None
        self.assertEqual(result.details["object_count"], 2)

    def test_dino_patch_similarity_compares_same_position_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.png"
            candidate_path = root / "candidate.png"
            Image.new("RGB", (32, 32), "white").save(reference_path)
            Image.new("RGB", (32, 32), "white").save(candidate_path)
            embeddings = np.asarray(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [1.0, 0.0]],
                ],
                dtype=np.float32,
            )

            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_dino_patch_embeddings_for_images",
                return_value=embeddings,
            ):
                result = compute_layout_dino_patch_similarity(
                    reference_image=reference_path,
                    candidate_image=candidate_path,
                    thresholds={"min_layout_dino_patch_similarity": 0.5},
                )

        self.assertEqual(result.status, "passed")
        self.assertAlmostEqual(result.value or 0.0, 0.5)

    def test_edge_structure_rewards_matching_layout_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.png"
            candidate_path = root / "candidate.png"
            shifted_path = root / "shifted.png"
            reference = Image.new("RGB", (48, 48), "black")
            candidate = Image.new("RGB", (48, 48), "black")
            shifted = Image.new("RGB", (48, 48), "black")
            ImageDraw.Draw(reference).rectangle((12, 12, 32, 32), fill="white")
            ImageDraw.Draw(candidate).rectangle((12, 12, 32, 32), fill="white")
            ImageDraw.Draw(shifted).rectangle((24, 12, 44, 32), fill="white")
            reference.save(reference_path)
            candidate.save(candidate_path)
            shifted.save(shifted_path)

            same = compute_layout_edge_structure(
                reference_image=reference_path,
                candidate_image=candidate_path,
                thresholds={"min_layout_edge_structure_score": 0.9},
            )
            moved = compute_layout_edge_structure(
                reference_image=reference_path,
                candidate_image=shifted_path,
                thresholds={"min_layout_edge_structure_score": 0.75},
            )

        self.assertEqual(same.status, "passed")
        self.assertGreaterEqual(same.value or 0.0, 0.99)
        self.assertEqual(moved.status, "failed")

    def test_depth_structure_uses_relative_depth_and_skips_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.png"
            candidate_path = root / "candidate.png"
            Image.new("RGB", (16, 16), "white").save(reference_path)
            Image.new("RGB", (16, 16), "white").save(candidate_path)
            depth = np.zeros((16, 16), dtype=np.float32)
            depth[4:12, 4:12] = 1.0
            flat = np.zeros((16, 16), dtype=np.float32)

            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_depth_maps_for_images",
                return_value=np.stack([depth, depth]),
            ):
                same = compute_layout_depth_structure(
                    reference_image=reference_path,
                    candidate_image=candidate_path,
                    thresholds={"min_layout_depth_structure_score": 0.9},
                )
            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_depth_maps_for_images",
                return_value=np.stack([depth, flat]),
            ):
                different = compute_layout_depth_structure(
                    reference_image=reference_path,
                    candidate_image=candidate_path,
                    thresholds={"min_layout_depth_structure_score": 0.8},
                )
            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_depth_maps_for_images",
                side_effect=OSError("not cached"),
            ):
                skipped = compute_layout_depth_structure(
                    reference_image=reference_path,
                    candidate_image=candidate_path,
                    thresholds={},
                )

        self.assertEqual(same.status, "passed")
        self.assertEqual(different.status, "failed")
        self.assertEqual(skipped.status, "skipped")

    def test_layout_gate_caps_failed_hard_checks(self) -> None:
        result = compute_layout_gate_score(
            metrics=[
                MetricResult(
                    name="layout_fidelity",
                    status="failed",
                    value=0.30,
                    threshold=0.82,
                    direction="min",
                    details={
                        "raw_score": 0.98,
                        "score_cap": 0.30,
                        "hard_check_failures": [{"check": "object_position"}],
                    },
                ),
                MetricResult(name="layout_edge_structure", status="passed", value=1.0),
                MetricResult(name="layout_depth_structure", status="passed", value=1.0),
                MetricResult(
                    name="layout_dino_patch_similarity",
                    status="passed",
                    value=1.0,
                ),
                MetricResult(name="preset_fidelity", status="passed", value=1.0),
                MetricResult(name="detector_layout", status="passed", value=1.0),
                MetricResult(
                    name="detector_spatial_relations",
                    status="passed",
                    value=1.0,
                ),
            ],
            thresholds={"min_layout_gate_score": 0.65},
        )

        self.assertEqual(result.status, "failed")
        self.assertLessEqual(result.value or 1.0, 0.30)

    def test_structural_layout_score_ignores_failed_llm_layout_gate(self) -> None:
        result = compute_structural_layout_score(
            metrics=[
                MetricResult(name="layout_fidelity", status="failed", value=0.08),
                MetricResult(name="layout_gate_score", status="failed", value=0.2),
                MetricResult(
                    name="layout_edge_structure",
                    status="passed",
                    value=0.77,
                ),
                MetricResult(
                    name="layout_depth_structure",
                    status="passed",
                    value=0.76,
                ),
                MetricResult(
                    name="layout_dino_patch_similarity",
                    status="passed",
                    value=0.75,
                ),
                MetricResult(
                    name="layout_dino_object_similarity",
                    status="warning",
                    value=0.61,
                ),
            ],
            thresholds={"min_structural_layout_score": 0.58},
        )

        self.assertEqual(result.status, "passed")
        self.assertGreater(result.value or 0.0, 0.7)

    def test_provider_observability_metrics_read_runtime_and_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "tknt_snapshot_result.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "provider_runtime_s": 12.5,
                        "visual_preflight": {
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 10,
                                "total_tokens": 110,
                            }
                        },
                        "render_result": {
                            "usage": {
                                "input_tokens": 20,
                                "output_tokens": 5,
                                "total_tokens": 25,
                            },
                            "elapsed_ms": 8000,
                        },
                        "edit_result": {"image": {}},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "render.png"
            Image.new("RGB", (4, 4), "white").save(output)

            runtime, usage = compute_provider_observability_metrics(
                ProviderResult(
                    case_id="case",
                    variant_id="variant",
                    output_image=output,
                    metadata_path=metadata_path,
                )
            )

        self.assertEqual(runtime.name, "provider_runtime")
        self.assertEqual(runtime.value, 12.5)
        self.assertEqual(usage.name, "provider_token_usage")
        self.assertEqual(usage.status, "passed")
        self.assertEqual(usage.value, 135.0)
        assert usage.details is not None
        self.assertEqual(usage.details["input_tokens"], 120)
        self.assertEqual(usage.details["output_tokens"], 15)
        self.assertEqual(usage.details["missing_usage_sources"], ["edit_result"])

    def test_provider_observability_metrics_sum_two_pass_results_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "tknt_snapshot_result.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "pass_results": {
                            "pass1": {
                                "usage": {
                                    "input_tokens": 10,
                                    "output_tokens": 1,
                                    "total_tokens": 11,
                                },
                                "elapsed_ms": 3000,
                            },
                            "pass2": {
                                "usage": {
                                    "input_tokens": 20,
                                    "output_tokens": 2,
                                    "total_tokens": 22,
                                },
                                "elapsed_ms": 4000,
                            },
                        },
                        "render_result": {
                            "usage": {
                                "input_tokens": 999,
                                "output_tokens": 999,
                                "total_tokens": 1998,
                            },
                            "elapsed_ms": 999000,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "render.png"
            Image.new("RGB", (4, 4), "white").save(output)

            runtime, usage = compute_provider_observability_metrics(
                ProviderResult(
                    case_id="case",
                    variant_id="two_pass_geometry_repair",
                    output_image=output,
                    metadata_path=metadata_path,
                )
            )

        self.assertEqual(runtime.value, 7.0)
        self.assertEqual(usage.value, 33.0)
        assert usage.details is not None
        self.assertEqual(
            [item["source"] for item in usage.details["source_usages"]],
            ["pass_results.pass1", "pass_results.pass2"],
        )

    def test_detector_layout_and_spatial_relations_use_bbox_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate.png"
            Image.new("RGB", (100, 100), "white").save(candidate_path)
            payload = {
                "visibleObjects": [
                    {
                        "id": "sofa",
                        "type": "sofa",
                        "screenBboxPx": {
                            "minX": 10,
                            "minY": 40,
                            "maxX": 40,
                            "maxY": 70,
                        },
                    },
                    {
                        "id": "chair",
                        "type": "chair",
                        "screenBboxPx": {
                            "minX": 60,
                            "minY": 40,
                            "maxX": 80,
                            "maxY": 65,
                        },
                    },
                ]
            }
            detections = [
                DetectionBox(
                    label="couch",
                    confidence=0.9,
                    bbox_norm={"x": 0.1, "y": 0.4, "width": 0.3, "height": 0.3},
                ),
                DetectionBox(
                    label="chair",
                    confidence=0.9,
                    bbox_norm={"x": 0.6, "y": 0.4, "width": 0.2, "height": 0.25},
                ),
            ]
            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics._detect_objects",
                return_value=detections,
            ):
                layout = compute_detector_layout(
                    candidate_image=candidate_path,
                    source_payload=payload,
                    thresholds={"min_detector_layout_match_ratio": 1.0},
                )
                relations = compute_detector_spatial_relations(
                    candidate_image=candidate_path,
                    source_payload=payload,
                    thresholds={"min_detector_spatial_relation_match_ratio": 1.0},
                )

        self.assertEqual(layout.status, "passed")
        self.assertEqual(layout.value, 1.0)
        self.assertEqual(relations.status, "passed")
        self.assertEqual(relations.value, 1.0)


if __name__ == "__main__":
    unittest.main()
