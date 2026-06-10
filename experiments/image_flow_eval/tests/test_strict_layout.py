from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from experiments.image_flow_eval.image_flow_eval.io import load_manifest
from experiments.image_flow_eval.image_flow_eval.metrics import (
    _fidelity_eval_prompt,
    compute_edit_scope_fidelity,
    compute_layout_fidelity,
    compute_render_layout_fidelity,
)
from experiments.image_flow_eval.image_flow_eval.providers import run_case_variant
from experiments.image_flow_eval.image_flow_eval.types import (
    BBoxNorm,
    CaseConfig,
    EditOperationConfig,
    ProviderResult,
    VariantConfig,
)
from pipeline.snapshot_image_renderer import (
    SnapshotImageRenderConfig,
    _sanitize_generate_user_prompt,
    render_snapshot_image,
)


class StrictLayoutFlowTest(unittest.TestCase):
    def test_generate_prompt_sanitizer_drops_design_brief(self) -> None:
        prompt = (
            "Could you please design a living room. Include a sofa, a TV console, "
            "and a bookshelf. Use warm realistic lighting and natural material texture."
        )

        sanitized = _sanitize_generate_user_prompt(prompt)

        self.assertNotIn("Include a sofa", sanitized)
        self.assertNotIn("design a living room", sanitized)
        self.assertIn("realistic lighting", sanitized)

    def test_generate_prompt_sanitizer_falls_back_when_only_layout_directives(
        self,
    ) -> None:
        prompt = "Could you please design the room. Include a sofa and move the TV."

        sanitized = _sanitize_generate_user_prompt(prompt)

        self.assertIn("Materialize the exact 3D source view", sanitized)
        self.assertNotIn("move the TV", sanitized)

    def test_edit_materialization_uses_geometry_reference_maps(self) -> None:
        data_url = _data_url(Path(tempfile.gettempdir()) / "tknt_geometry_ref.png")

        with patch(
            "pipeline.snapshot_image_renderer._generate_image_with_openai",
            return_value={"data_url": data_url, "mime_type": "image/png"},
        ) as generator:
            result = render_snapshot_image(
                {"canvas": {"widthPx": 8, "heightPx": 8}, "visibleObjects": []},
                snapshot_image_data_url=data_url,
                depth_map_image_data_url=data_url,
                edge_map_image_data_url=data_url,
                segmentation_map_image_data_url=data_url,
                annotated_reference_image_data_url=data_url,
                edit_source_image_data_url=data_url,
                user_prompt="Photorealize without layout drift.",
                layout_materialization_mode=True,
                render_mode="edit",
                config=SnapshotImageRenderConfig(
                    include_geometry_references=True,
                    include_annotated_reference=True,
                    openai_api_key="test-key",
                ),
            )

        _, kwargs = generator.call_args
        self.assertIsNotNone(kwargs["depth_map_image_data_url"])
        self.assertIsNotNone(kwargs["edge_map_image_data_url"])
        self.assertIsNotNone(kwargs["segmentation_map_image_data_url"])
        self.assertIn("deterministic depth map", kwargs["user_prompt"])
        self.assertTrue(result["request"]["geometry_references_used"]["depth_map"])

    def test_manifest_accepts_remove_operation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_payload = root / "snapshot_payload.json"
            snapshot_payload.write_text("{}", encoding="utf-8")
            snapshot_image = root / "snapshot.png"
            Image.new("RGB", (8, 8), "white").save(snapshot_image)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "case_remove",
                                "snapshot_payload": str(snapshot_payload),
                                "snapshot_image": str(snapshot_image),
                                "edit_operations": [
                                    {
                                        "object_id": "region_1",
                                        "operation_type": "remove",
                                        "bbox_norm": {
                                            "x": 0.1,
                                            "y": 0.1,
                                            "width": 0.2,
                                            "height": 0.2,
                                        },
                                    }
                                ],
                                "variants": [{"id": "operation_dsl"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_manifest(manifest_path)

        self.assertEqual(manifest.cases[0].edit_operations[0].operation_type, "remove")

    def test_provider_aligns_static_snapshot_source_to_output_aspect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case(root, image_size=(12, 10), canvas_size=(12, 10))
            variant = VariantConfig(id="contract_objects", mode="contract_objects")

            with _mock_renderer_data_url():
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root / "runs",
                )

            self.assertIsNotNone(result.source_image)
            self.assertIsNotNone(result.source_snapshot_payload)
            assert result.source_image is not None
            assert result.source_snapshot_payload is not None
            with Image.open(result.source_image) as image:
                self.assertEqual(image.size, (1536, 1024))
            payload = json.loads(result.source_snapshot_payload.read_text())
            self.assertEqual(payload["canvas"], {"widthPx": 1536, "heightPx": 1024})
            self.assertAlmostEqual(payload["camera"]["aspect"], 1.5)

    def test_layout_fidelity_fails_aspect_mismatch_before_vision_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            Image.new("RGB", (12, 10), "white").save(source)
            Image.new("RGB", (15, 10), "white").save(output)
            case = CaseConfig(id="case", snapshot_image=source)
            variant = VariantConfig(id="baseline")

            result = compute_layout_fidelity(
                case=case,
                variant=variant,
                output_image=output,
                output_dir=root,
                thresholds={"max_layout_aspect_delta": 0.01},
                provider_result=None,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.details["reason"], "source_candidate_aspect_mismatch")

    def test_edit_scope_fidelity_fails_outside_bbox_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = root / "render.png"
            output = root / "edit.png"
            Image.new("RGB", (10, 10), "black").save(render)
            edited = Image.new("RGB", (10, 10), "black")
            edited.putpixel((9, 9), (255, 255, 255))
            edited.save(output)
            case = CaseConfig(
                id="case",
                edit_operations=(
                    EditOperationConfig(
                        object_id="region_1",
                        operation_type="recolor",
                        bbox_norm=BBoxNorm(x=0, y=0, width=0.2, height=0.2),
                    ),
                ),
            )
            provider_result = ProviderResult(
                case_id="case",
                variant_id="operation_dsl",
                output_image=output,
                metadata_path=None,
                render_image=render,
            )

            result = compute_edit_scope_fidelity(
                case=case,
                variant=VariantConfig(id="operation_dsl"),
                output_image=output,
                thresholds={"max_edit_outside_change_ratio": 0.0},
                provider_result=provider_result,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, "failed")

    def test_layout_fidelity_hard_fails_when_silhouette_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            Image.new("RGB", (10, 10), "white").save(source)
            Image.new("RGB", (10, 10), "blue").save(output)
            case = CaseConfig(id="case", snapshot_image=source)

            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_evaluate_fidelity_with_vision",
                return_value={
                    "status": "passed",
                    "score": 0.99,
                    "hard_checks": {
                        "camera_framing": {"status": "passed"},
                        "room_shell": {"status": "passed"},
                        "openings": {"status": "passed"},
                        "object_inventory": {"status": "passed"},
                        "object_position": {"status": "passed"},
                        "object_scale": {"status": "passed"},
                        "object_silhouette": {
                            "status": "failed",
                            "reason": "sofa silhouette changed",
                            "violations": ["sofa"],
                        },
                        "unauthorized_additions": {"status": "passed"},
                    },
                },
            ):
                result = compute_layout_fidelity(
                    case=case,
                    variant=VariantConfig(id="full"),
                    output_image=output,
                    output_dir=root,
                    thresholds={},
                    provider_result=None,
                )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.value, 0.35)
        self.assertEqual(result.details["raw_score"], 0.99)
        self.assertEqual(result.details["effective_score"], 0.35)
        self.assertEqual(result.details["score_cap"], 0.35)
        self.assertEqual(
            result.details["hard_check_failures"][0]["check"],
            "object_silhouette",
        )

    def test_layout_fidelity_fails_closed_when_hard_checks_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            Image.new("RGB", (10, 10), "white").save(source)
            Image.new("RGB", (10, 10), "blue").save(output)

            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_evaluate_fidelity_with_vision",
                return_value={"status": "passed", "score": 0.99},
            ):
                result = compute_layout_fidelity(
                    case=CaseConfig(id="case", snapshot_image=source),
                    variant=VariantConfig(id="full"),
                    output_image=output,
                    output_dir=root,
                    thresholds={},
                    provider_result=None,
                )

        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.details["hard_check_failures"]), 8)
        self.assertEqual(result.value, 0.20)
        self.assertEqual(result.details["score_cap"], 0.20)

    def test_layout_fidelity_uses_hard_checks_instead_of_freeform_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            Image.new("RGB", (10, 10), "white").save(source)
            Image.new("RGB", (10, 10), "blue").save(output)

            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_evaluate_fidelity_with_vision",
                return_value={
                    "status": "failed",
                    "score": 0.96,
                    "hard_checks": {
                        "camera_framing": {"status": "passed"},
                        "room_shell": {"status": "passed"},
                        "openings": {"status": "passed"},
                        "object_inventory": {"status": "passed"},
                        "object_position": {"status": "passed"},
                        "object_scale": {"status": "passed"},
                        "object_silhouette": {"status": "passed"},
                        "unauthorized_additions": {"status": "passed"},
                    },
                },
            ):
                result = compute_layout_fidelity(
                    case=CaseConfig(id="case", snapshot_image=source),
                    variant=VariantConfig(id="full"),
                    output_image=output,
                    output_dir=root,
                    thresholds={},
                    provider_result=None,
                )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.value, 0.96)
        self.assertEqual(result.details["raw_score"], 0.96)
        self.assertEqual(result.details["effective_score"], 0.96)
        self.assertIsNone(result.details["score_cap"])
        self.assertEqual(result.details["raw_status"], "failed")

    def test_layout_prompt_treats_existing_window_scenery_as_allowed(self) -> None:
        prompt = _fidelity_eval_prompt(
            case=CaseConfig(id="case"),
            variant=VariantConfig(id="full"),
            source_payload={
                "room": {
                    "openings": {
                        "windows": [{"id": "window", "screenBboxPx": None}],
                    }
                }
            },
            evaluation_kind="create",
            required_hard_checks=("openings",),
        )

        self.assertIn("null screenBboxPx", prompt)
        self.assertIn("exterior scenery", prompt)

    def test_layout_prompt_includes_semantic_room_shell_locks(self) -> None:
        prompt = _fidelity_eval_prompt(
            case=CaseConfig(id="case"),
            variant=VariantConfig(id="single_edit_edge_silhouette_guided"),
            source_payload={},
            evaluation_kind="create",
            required_hard_checks=("room_shell", "unauthorized_additions"),
            semantic_layout_locks=(
                "Room-shell lock: preserve wall_1 as a real wall plane near the left edge.",
                "Shadow-vs-geometry lock: wall shadows are not floor lamps.",
            ),
        )

        self.assertIn("Semantic room-shell/crop locks", prompt)
        self.assertIn("preserve wall_1", prompt)
        self.assertIn("wall shadows are not floor lamps", prompt)
        self.assertIn("fail room_shell", prompt)

    def test_layout_fidelity_reads_semantic_locks_from_provider_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            metadata_path = root / "tknt_snapshot_result.json"
            Image.new("RGB", (10, 10), "white").save(source)
            Image.new("RGB", (10, 10), "white").save(output)
            metadata_path.write_text(
                json.dumps(
                    {
                        "edge_silhouette_guide": {
                            "semantic_room_shell_locks": [
                                "Room-shell lock: preserve wall_1 near left crop."
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            provider_result = ProviderResult(
                case_id="case",
                variant_id="single_edit_edge_silhouette_guided",
                output_image=output,
                metadata_path=metadata_path,
                source_image=source,
            )

            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_evaluate_fidelity_with_vision",
                return_value={
                    "status": "passed",
                    "score": 0.95,
                    "hard_checks": {
                        "camera_framing": {"status": "passed"},
                        "room_shell": {"status": "passed"},
                        "openings": {"status": "passed"},
                        "object_inventory": {"status": "passed"},
                        "object_position": {"status": "passed"},
                        "object_scale": {"status": "passed"},
                        "object_silhouette": {"status": "passed"},
                        "unauthorized_additions": {"status": "passed"},
                    },
                },
            ) as evaluator:
                result = compute_layout_fidelity(
                    case=CaseConfig(id="case", snapshot_image=source),
                    variant=VariantConfig(id="single_edit_edge_silhouette_guided"),
                    output_image=output,
                    output_dir=root,
                    thresholds={},
                    provider_result=provider_result,
                )

        self.assertEqual(result.status, "passed")
        _, kwargs = evaluator.call_args
        self.assertEqual(
            kwargs["semantic_layout_locks"],
            ("Room-shell lock: preserve wall_1 near left crop.",),
        )

    def test_remove_scope_passes_when_change_is_localized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = root / "render.png"
            output = root / "edit.png"
            Image.new("RGB", (20, 20), "black").save(render)
            edited = Image.new("RGB", (20, 20), "black")
            for x in range(8, 12):
                for y in range(8, 12):
                    edited.putpixel((x, y), (255, 255, 255))
            edited.save(output)
            case = CaseConfig(
                id="case",
                edit_operations=(
                    EditOperationConfig(
                        object_id="floor_lamp",
                        operation_type="remove",
                        bbox_norm=BBoxNorm(x=0.4, y=0.4, width=0.2, height=0.2),
                    ),
                ),
            )

            result = compute_edit_scope_fidelity(
                case=case,
                variant=VariantConfig(id="operation_dsl_preflight"),
                output_image=output,
                thresholds={"max_edit_outside_change_ratio": 0.0},
                provider_result=ProviderResult(
                    case_id="case",
                    variant_id="operation_dsl_preflight",
                    output_image=output,
                    metadata_path=None,
                    render_image=render,
                ),
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, "passed")

    def test_render_layout_metric_reuses_approved_create_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            approved_render = root / "create" / "full" / "render.png"
            Image.new("RGB", (10, 10), "white").save(source)
            approved_render.parent.mkdir(parents=True)
            Image.new("RGB", (10, 10), "blue").save(approved_render)
            (approved_render.parent / "metrics.json").write_text(
                json.dumps(
                    [
                        {
                            "name": "layout_fidelity",
                            "status": "passed",
                            "value": 0.93,
                            "threshold": 0.82,
                            "direction": "min",
                            "details": {},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "experiments.image_flow_eval.image_flow_eval.metrics."
                "_evaluate_fidelity_with_vision"
            ) as evaluator:
                result = compute_render_layout_fidelity(
                    case=CaseConfig(id="edit", snapshot_image=source),
                    variant=VariantConfig(id="operation_dsl_preflight"),
                    output_dir=root,
                    thresholds={},
                    provider_result=ProviderResult(
                        case_id="edit",
                        variant_id="operation_dsl_preflight",
                        output_image=approved_render,
                        metadata_path=None,
                        render_image=approved_render,
                        source_image=source,
                        approved_render_source=approved_render,
                    ),
                )

            self.assertEqual(result.status, "passed")
            self.assertEqual(
                result.details["reason"], "reused_approved_render_layout_gate"
            )
            evaluator.assert_not_called()


def _case(
    root: Path,
    *,
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> CaseConfig:
    snapshot_payload = root / "snapshot_payload.json"
    snapshot_payload.write_text(
        json.dumps(
            {
                "canvas": {
                    "widthPx": canvas_size[0],
                    "heightPx": canvas_size[1],
                },
                "camera": {"aspect": canvas_size[0] / canvas_size[1]},
                "visibleObjects": [
                    {
                        "id": "sofa",
                        "type": "sofa",
                        "screenCenterPx": {"x": 6, "y": 5},
                        "screenBboxPx": {
                            "minX": 3,
                            "minY": 3,
                            "maxX": 9,
                            "maxY": 7,
                            "width": 6,
                            "height": 4,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot_image = root / "snapshot.png"
    Image.new("RGB", image_size, "white").save(snapshot_image)
    return CaseConfig(
        id="case",
        render_prompt="Could you please design a room. Include a sofa.",
        snapshot_payload=snapshot_payload,
        snapshot_image=snapshot_image,
    )


def _mock_renderer_data_url():
    from unittest.mock import patch

    data_url = _data_url(Path(tempfile.gettempdir()) / "tknt_strict_layout_test.png")
    return patch(
        "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
        return_value={"image": {"data_url": data_url}},
    )


def _data_url(path: Path) -> str:
    if not path.exists():
        Image.new("RGB", (2, 2), "blue").save(path)

    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(
        "ascii"
    )


if __name__ == "__main__":
    unittest.main()
