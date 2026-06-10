from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from experiments.image_flow_eval.image_flow_eval import cli
from experiments.image_flow_eval.image_flow_eval.io import encode_image_to_data_url
from experiments.image_flow_eval.image_flow_eval.preflight import VisualPreflightResult
from experiments.image_flow_eval.image_flow_eval.providers import run_case_variant
from experiments.image_flow_eval.image_flow_eval.types import (
    BBoxNorm,
    CaseConfig,
    EditSourceVariantConfig,
    EditOperationConfig,
    RenderPresetSelection,
    VariantConfig,
)


class ImageFlowResumeTest(unittest.TestCase):
    def test_resume_and_skip_existing_parse_to_same_option(self) -> None:
        for flag in ("--resume", "--skip-existing"):
            with self.subTest(flag=flag):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "image_flow_eval",
                        "run",
                        "--manifest",
                        "manifest.json",
                        flag,
                    ],
                ):
                    args = cli._parse_args()

                self.assertTrue(args.skip_existing)

    def test_evaluate_only_reuses_generated_render_when_output_image_is_omitted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case(root)
            variant = VariantConfig(id="contract_objects", mode="contract_objects")
            render_path = root / case.id / variant.id / "render.png"
            _write_png(render_path)

            result = run_case_variant(
                provider="evaluate_only",
                case=case,
                variant=variant,
                output_root=root,
            )

            self.assertEqual(result.provider_status, "reused_existing")
            self.assertEqual(result.output_image, render_path)
            self.assertEqual(result.render_image, render_path)

    def test_create_variant_with_existing_render_skips_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case(root)
            variant = VariantConfig(id="contract_objects", mode="contract_objects")
            render_path = root / case.id / variant.id / "render.png"
            _write_png(render_path)

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                    skip_existing=True,
                )

            self.assertEqual(result.provider_status, "reused_existing")
            self.assertEqual(result.output_image, render_path)
            renderer.assert_not_called()

    def test_edit_variant_with_existing_edit_skips_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case(root, edit_prompt="Recolor only the sofa.")
            variant = VariantConfig(id="operation_dsl", mode="operation_dsl")
            edit_path = root / case.id / variant.id / "edit.png"
            _write_png(edit_path)

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                    skip_existing=True,
                )

            self.assertEqual(result.provider_status, "reused_existing")
            self.assertEqual(result.output_image, edit_path)
            renderer.assert_not_called()

    def test_edit_variant_with_existing_render_runs_only_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case(root, edit_prompt="Recolor only the sofa.")
            variant = VariantConfig(id="operation_dsl", mode="operation_dsl")
            render_path = root / case.id / variant.id / "render.png"
            _write_png(render_path)

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                    skip_existing=True,
                )

            self.assertEqual(result.provider_status, "reused_render_partial")
            self.assertEqual(result.render_image, render_path)
            self.assertEqual(
                result.output_image, root / case.id / variant.id / "edit.png"
            )
            self.assertEqual(renderer.call_count, 1)
            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertTrue(
                str(kwargs["edit_source_image_data_url"]).startswith("data:image/")
            )

    def test_empty_render_output_does_not_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case(root)
            variant = VariantConfig(id="contract_objects", mode="contract_objects")
            render_path = root / case.id / variant.id / "render.png"
            render_path.parent.mkdir(parents=True)
            render_path.write_bytes(b"")

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                    skip_existing=True,
                )

            self.assertEqual(result.provider_status, "generated")
            self.assertGreater(render_path.stat().st_size, 0)
            self.assertEqual(renderer.call_count, 1)

    def test_edit_variant_reuses_approved_render_and_skips_generate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_render = root / "create_case" / "full" / "render.png"
            _write_png(approved_render)
            _write_passed_layout_metrics(approved_render.parent / "metrics.json")
            case = replace(
                _case(root, edit_prompt="Recolor only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="full",
                ),
            )
            variant = VariantConfig(
                id="operation_dsl_preflight",
                mode="operation_dsl_preflight",
            )

            with (
                patch(
                    "experiments.image_flow_eval.image_flow_eval.providers."
                    "run_visual_preflight"
                ) as preflight,
                _mock_renderer() as renderer,
            ):
                preflight.return_value.source_valid = True
                preflight.return_value.required_counts = {}
                preflight.return_value.priority_locks = ()
                preflight.return_value.edit_scope_locks = ()
                preflight.return_value.to_json.return_value = {}
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            self.assertEqual(result.provider_status, "reused_approved_render")
            self.assertEqual(result.render_image, approved_render)
            self.assertEqual(renderer.call_count, 1)
            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            _, preflight_kwargs = preflight.call_args
            self.assertEqual(preflight_kwargs["source_image"], approved_render)
            self.assertEqual(preflight_kwargs["profile"], "edit_scope")
            self.assertEqual(
                tuple(preflight_kwargs["edit_operations"]),
                case.edit_operations,
            )

    def test_edit_variant_does_not_reapply_render_presets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_render = root / "create_case" / "full" / "render.png"
            _write_png(approved_render)
            _write_passed_layout_metrics(approved_render.parent / "metrics.json")
            case = replace(
                _case(root, edit_prompt="Recolor only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="full",
                ),
                preset_selection=RenderPresetSelection(
                    style="japandi",
                    lighting="daylight_6",
                    scenery="garden",
                ),
            )
            variant = VariantConfig(
                id="operation_dsl_preflight",
                mode="operation_dsl_preflight",
            )

            with (
                patch(
                    "experiments.image_flow_eval.image_flow_eval.providers."
                    "run_visual_preflight"
                ) as preflight,
                _mock_renderer() as renderer,
            ):
                preflight.return_value.source_valid = True
                preflight.return_value.required_counts = {}
                preflight.return_value.priority_locks = ()
                preflight.return_value.edit_scope_locks = ()
                preflight.return_value.to_json.return_value = {}
                run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertEqual(kwargs["preset_selection"], {})

    def test_edit_prompt_image_only_uses_prompt_clean_render_and_bbox_without_dsl(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_render = root / "create_case" / "full" / "render.png"
            _write_png(approved_render)
            _write_passed_layout_metrics(approved_render.parent / "metrics.json")
            case = replace(
                _case(root, edit_prompt="Recolor only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="full",
                ),
            )
            variant = VariantConfig(
                id="edit_prompt_image_only",
                mode="edit_prompt_image_only",
            )

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            self.assertEqual(result.provider_status, "reused_approved_render")
            self.assertEqual(renderer.call_count, 1)
            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertEqual(kwargs["user_prompt"], "Recolor only the sofa.")
            self.assertEqual(len(kwargs["edit_operations"]), 1)
            operation = kwargs["edit_operations"][0]
            self.assertEqual(operation.object_id, "sofa")
            self.assertIsNotNone(operation.bbox_norm)
            self.assertIsNone(kwargs["layout_reference_image_data_url"])
            self.assertFalse(kwargs["config"].include_edit_operation_dsl)

    def test_edit_prompt_image_only_keeps_replace_reference_and_bbox_without_dsl(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_render = root / "create_case" / "full" / "render.png"
            replacement_image = root / "replacement.png"
            _write_png(approved_render)
            _write_png(replacement_image)
            _write_passed_layout_metrics(approved_render.parent / "metrics.json")
            case = replace(
                _case(root, edit_prompt="Replace only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="full",
                ),
                edit_operations=(
                    EditOperationConfig(
                        object_id="sofa",
                        object_name="sofa",
                        operation_type="replace",
                        prompt="Replace only the sofa.",
                        bbox_norm=BBoxNorm(x=0.1, y=0.1, width=0.4, height=0.3),
                        replacement_image=replacement_image,
                    ),
                ),
            )
            variant = VariantConfig(
                id="edit_prompt_image_only",
                mode="edit_prompt_image_only",
            )

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            self.assertEqual(result.provider_status, "reused_approved_render")
            self.assertEqual(renderer.call_count, 1)
            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertEqual(kwargs["user_prompt"], "Replace only the sofa.")
            self.assertEqual(len(kwargs["edit_operations"]), 1)
            operation = kwargs["edit_operations"][0]
            self.assertIsNotNone(operation.bbox_norm)
            self.assertIsNotNone(operation.replacement_image_data_url)
            self.assertIsNone(kwargs["layout_reference_image_data_url"])
            self.assertIsNone(kwargs["annotated_reference_image_data_url"])
            self.assertFalse(kwargs["config"].include_edit_operation_dsl)

    def test_edit_object_scope_preflight_uses_bbox_guide_and_object_guard_mask(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_render = root / "create_case" / "full" / "render.png"
            bbox_guide = root / "bbox_guide.png"
            image_size = (1536, 1024)
            approved_render.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", image_size, "red").save(approved_render)
            Image.new("RGB", image_size, "red").save(bbox_guide)
            _write_passed_layout_metrics(approved_render.parent / "metrics.json")
            case = replace(
                _case(root, edit_prompt="Recolor only the wardrobe."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="full",
                ),
                annotated_reference_image=bbox_guide,
                edit_operations=(
                    EditOperationConfig(
                        object_id="manual_box_1",
                        object_name="wardrobe",
                        operation_type="replace",
                        prompt="Replace only the wardrobe.",
                        bbox_norm=BBoxNorm(x=0.5, y=0.2, width=0.2, height=0.4),
                        target_color="muted forest green",
                    ),
                ),
            )
            case.snapshot_payload.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 1536, "heightPx": 1024},
                        "visibleObjects": [
                            {
                                "id": "wardrobe_001",
                                "type": "wardrobe",
                                "screenBboxPx": {
                                    "minX": 798,
                                    "minY": 190,
                                    "maxX": 1259,
                                    "maxY": 717,
                                },
                            },
                            {
                                "id": "table_lamp_001",
                                "type": "table_lamp",
                                "screenBboxPx": {
                                    "minX": 1044,
                                    "minY": 512,
                                    "maxX": 1167,
                                    "maxY": 800,
                                },
                            },
                            {
                                "id": "armchair",
                                "type": "armchair",
                                "screenBboxPx": {
                                    "minX": 0,
                                    "minY": 0,
                                    "maxX": 1536,
                                    "maxY": 1024,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            Image.new("RGB", image_size, "red").save(case.snapshot_image)
            variant = VariantConfig(
                id="edit_object_scope_preflight",
                mode="edit_object_scope_preflight",
            )

            with (
                patch(
                    "experiments.image_flow_eval.image_flow_eval.providers."
                    "run_visual_preflight"
                ) as preflight,
                _mock_renderer() as renderer,
            ):
                preflight.return_value.source_valid = True
                preflight.return_value.required_counts = {}
                preflight.return_value.priority_locks = ()
                preflight.return_value.edit_scope_locks = (
                    "Keep the desk lamp and wardrobe handles unchanged.",
                )
                preflight.return_value.to_json.return_value = {}
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            self.assertEqual(result.provider_status, "reused_approved_render")
            _, preflight_kwargs = preflight.call_args
            self.assertEqual(preflight_kwargs["source_image"], bbox_guide)
            self.assertEqual(preflight_kwargs["profile"], "edit_object_scope")
            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertIsNotNone(kwargs["annotated_reference_image_data_url"])
            self.assertIsNotNone(kwargs["layout_guard_mask_image_data_url"])
            self.assertTrue(kwargs["config"].include_edit_operation_dsl)
            self.assertFalse(kwargs["config"].disable_edit_mask)

            mask_path = (
                root
                / case.id
                / variant.id
                / "object_scope_edit_mask.png"
            )
            self.assertTrue(mask_path.is_file())
            with Image.open(mask_path) as mask:
                self.assertEqual(mask.getpixel((850, 300))[3], 0)
                self.assertEqual(mask.getpixel((1050, 550))[3], 255)
                self.assertEqual(mask.getpixel((10, 10))[3], 255)
            metadata = json.loads(
                (
                    root
                    / case.id
                    / variant.id
                    / "tknt_snapshot_result.json"
                ).read_text(encoding="utf-8")
            )
            mask_metadata = metadata["object_scope_edit_mask"]
            self.assertGreater(mask_metadata["editable_pixel_ratio"], 0)
            rejected_ids = {
                item["object_id"]
                for item in mask_metadata["rejected_protection_objects"]
            }
            self.assertIn("armchair", rejected_ids)

    def test_edit_object_scope_preflight_keeps_recolor_unmasked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_render = root / "create_case" / "full" / "render.png"
            bbox_guide = root / "bbox_guide.png"
            image_size = (1536, 1024)
            approved_render.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", image_size, "red").save(approved_render)
            Image.new("RGB", image_size, "red").save(bbox_guide)
            _write_passed_layout_metrics(approved_render.parent / "metrics.json")
            case = replace(
                _case(root, edit_prompt="Recolor only the wardrobe."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="full",
                ),
                annotated_reference_image=bbox_guide,
                edit_operations=(
                    EditOperationConfig(
                        object_id="manual_box_1",
                        object_name="wardrobe",
                        operation_type="recolor",
                        prompt="Recolor only the wardrobe.",
                        bbox_norm=BBoxNorm(x=0.5, y=0.2, width=0.2, height=0.4),
                        target_color="muted forest green",
                    ),
                ),
            )
            case.snapshot_payload.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 1536, "heightPx": 1024},
                        "visibleObjects": [
                            {
                                "id": "wardrobe_001",
                                "type": "wardrobe",
                                "screenBboxPx": {
                                    "minX": 798,
                                    "minY": 190,
                                    "maxX": 1259,
                                    "maxY": 717,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            Image.new("RGB", image_size, "red").save(case.snapshot_image)
            variant = VariantConfig(
                id="edit_object_scope_preflight",
                mode="edit_object_scope_preflight",
            )

            with (
                patch(
                    "experiments.image_flow_eval.image_flow_eval.providers."
                    "run_visual_preflight"
                ) as preflight,
                _mock_renderer() as renderer,
            ):
                preflight.return_value.source_valid = True
                preflight.return_value.required_counts = {}
                preflight.return_value.priority_locks = ()
                preflight.return_value.edit_scope_locks = ()
                preflight.return_value.to_json.return_value = {}
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            self.assertEqual(result.provider_status, "reused_approved_render")
            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertIsNotNone(kwargs["annotated_reference_image_data_url"])
            self.assertIsNone(kwargs["layout_guard_mask_image_data_url"])
            self.assertTrue(kwargs["config"].disable_edit_mask)
            mask_path = (
                root
                / case.id
                / variant.id
                / "object_scope_edit_mask.png"
            )
            self.assertFalse(mask_path.exists())
            metadata = json.loads(
                (
                    root
                    / case.id
                    / variant.id
                    / "tknt_snapshot_result.json"
                ).read_text(encoding="utf-8")
            )
            mask_metadata = metadata["object_scope_edit_mask"]
            self.assertFalse(mask_metadata["enabled"])
            self.assertEqual(
                mask_metadata["reason"],
                "disabled_for_stable_camera_operation",
            )

    def test_edit_variant_uses_best_passed_create_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured_render = (
                root / "create_case" / "contract_objects_preflight" / "render.png"
            )
            best_render = (
                root
                / "create_case"
                / "contract_objects_preflight_locked_geometry"
                / "render.png"
            )
            weaker_render = root / "create_case" / "contract_objects" / "render.png"
            _write_png(configured_render)
            _write_passed_layout_metrics(
                configured_render.parent / "metrics.json",
                value=0.91,
            )
            _write_png(best_render)
            _write_passed_layout_metrics(
                best_render.parent / "metrics.json",
                value=0.97,
            )
            _write_png(weaker_render)
            _write_passed_layout_metrics(
                weaker_render.parent / "metrics.json",
                value=0.88,
            )
            case = replace(
                _case(root, edit_prompt="Recolor only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="contract_objects_preflight",
                ),
            )
            variant = VariantConfig(id="operation_dsl", mode="operation_dsl")

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            self.assertEqual(result.provider_status, "reused_approved_render")
            self.assertEqual(result.render_image, best_render)
            self.assertEqual(result.approved_render_source, best_render)
            self.assertEqual(renderer.call_count, 1)

    def test_edit_variant_exact_source_does_not_scan_best_passed_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_render = root / "create_case" / "simple_prompt" / "render.png"
            best_render = (
                root
                / "create_case"
                / "contract_objects_preflight_locked_geometry"
                / "render.png"
            )
            _write_png(selected_render)
            _write_png(best_render)
            _write_passed_layout_metrics(
                best_render.parent / "metrics.json",
                value=0.97,
            )
            case = replace(
                _case(root, edit_prompt="Recolor only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="simple_prompt",
                    selection_mode="exact",
                ),
            )
            variant = VariantConfig(id="operation_dsl", mode="operation_dsl")

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            self.assertEqual(result.provider_status, "reused_approved_render")
            self.assertEqual(result.render_image, selected_render)
            self.assertEqual(result.approved_render_source, selected_render)
            self.assertEqual(renderer.call_count, 1)

    def test_best_passed_create_render_ignores_removed_variant_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            removed_render = (
                root / "create_case" / "llm_visual_preflight_no_metadata" / "render.png"
            )
            approved_render = (
                root
                / "create_case"
                / "contract_objects_preflight_locked_geometry"
                / "render.png"
            )
            _write_png(removed_render)
            _write_passed_layout_metrics(
                removed_render.parent / "metrics.json",
                value=0.99,
            )
            _write_png(approved_render)
            _write_passed_layout_metrics(
                approved_render.parent / "metrics.json",
                value=0.91,
            )
            case = replace(
                _case(root, edit_prompt="Recolor only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="contract_objects_preflight_locked_geometry",
                ),
            )

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=VariantConfig(id="operation_dsl", mode="operation_dsl"),
                    output_root=root,
                )

            self.assertEqual(result.provider_status, "reused_approved_render")
            self.assertEqual(result.render_image, approved_render)
            self.assertEqual(result.approved_render_source, approved_render)
            self.assertEqual(renderer.call_count, 1)

    def test_locked_geometry_variant_passes_atlas_to_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case(root)
            snapshot_payload = root / "snapshot_payload.json"
            snapshot_payload.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 64, "heightPx": 64},
                        "visibleObjects": [
                            {
                                "id": "sofa",
                                "type": "sofa",
                                "screenBboxPx": {
                                    "minX": 8,
                                    "minY": 10,
                                    "maxX": 42,
                                    "maxY": 32,
                                },
                            },
                            {
                                "id": "floor_lamp",
                                "type": "floor_lamp",
                                "screenBboxPx": {
                                    "minX": 46,
                                    "minY": 8,
                                    "width": 8,
                                    "height": 34,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            Image.new("RGB", (64, 64), "white").save(root / "snapshot.png")
            preflight = VisualPreflightResult(
                source_valid=True,
                risk_profile=(),
                required_counts={},
                priority_locks=(),
                reason="",
                model="test-model",
                cache_key="cache",
                cache_status="miss",
                usage={},
            )
            variant = VariantConfig(
                id="contract_objects_preflight_locked_geometry",
                mode="contract_objects_preflight_locked_geometry",
            )

            with (
                patch(
                    "experiments.image_flow_eval.image_flow_eval.providers."
                    "run_visual_preflight",
                    return_value=preflight,
                ),
                _mock_renderer() as renderer,
            ):
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root,
                )

            _, kwargs = renderer.call_args
            atlases = kwargs["locked_geometry_atlases"]
            self.assertEqual(len(atlases), 1)
            self.assertEqual(len(atlases[0].objects), 2)
            self.assertTrue(atlases[0].image_data_url.startswith("data:image/"))
            self.assertTrue(
                (root / case.id / variant.id / "locked_geometry_atlas.png").is_file()
            )
            self.assertEqual(
                result.output_image, root / case.id / variant.id / "render.png"
            )

    def test_edit_variant_rejects_unapproved_shared_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_render = root / "create_case" / "full" / "render.png"
            _write_png(approved_render)
            case = replace(
                _case(root, edit_prompt="Recolor only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="full",
                ),
            )

            with _mock_renderer() as renderer:
                with self.assertRaisesRegex(ValueError, "no passed layout score"):
                    run_case_variant(
                        provider="tknt_snapshot",
                        case=case,
                        variant=VariantConfig(
                            id="operation_dsl",
                            mode="operation_dsl",
                        ),
                        output_root=root,
                    )

            renderer.assert_not_called()

    def test_resume_keeps_original_render_for_existing_shared_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_render = root / "create_case" / "full" / "render.png"
            _write_png(approved_render)
            original_render = root / "case_1" / "operation_dsl" / "render.png"
            existing_edit = root / "case_1" / "operation_dsl" / "edit.png"
            _write_png(original_render)
            _write_png(existing_edit)
            case = replace(
                _case(root, edit_prompt="Recolor only the sofa."),
                edit_source_variant=EditSourceVariantConfig(
                    case_id="create_case",
                    variant="full",
                ),
            )

            with _mock_renderer() as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=VariantConfig(id="operation_dsl", mode="operation_dsl"),
                    output_root=root,
                    skip_existing=True,
                )

            self.assertEqual(result.provider_status, "reused_existing")
            self.assertEqual(result.render_image, original_render)
            renderer.assert_not_called()


def _case(root: Path, *, edit_prompt: str | None = None) -> CaseConfig:
    snapshot_payload = root / "snapshot_payload.json"
    snapshot_payload.write_text("{}", encoding="utf-8")
    snapshot_image = root / "snapshot.png"
    _write_png(snapshot_image)
    edit_operations: tuple[EditOperationConfig, ...] = ()
    if edit_prompt is not None:
        edit_operations = (
            EditOperationConfig(
                object_id="sofa",
                object_name="sofa",
                operation_type="recolor",
                prompt="Recolor the sofa only.",
                bbox_norm=BBoxNorm(x=0.1, y=0.1, width=0.4, height=0.3),
                target_color="camel leather",
            ),
        )
    return CaseConfig(
        id="case_1",
        render_prompt="Render a room.",
        edit_prompt=edit_prompt,
        snapshot_payload=snapshot_payload,
        snapshot_image=snapshot_image,
        edit_operations=edit_operations,
    )


def _mock_renderer():
    return patch(
        "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
        return_value={
            "image": {"data_url": encode_image_to_data_url(_tiny_png_path())}
        },
    )


def _tiny_png_path() -> Path:
    path = Path(tempfile.gettempdir()) / "tknt_image_flow_resume_test.png"
    if not path.exists():
        _write_png(path)
    return path


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), "red").save(path)


def _write_passed_layout_metrics(path: Path, *, value: float | None = None) -> None:
    metric = '{"name": "layout_fidelity", "status": "passed"'
    if value is not None:
        metric = f'{metric}, "value": {value}'
    path.write_text(f"[{metric}}}]", encoding="utf-8")


def _write_failed_layout_metrics(path: Path) -> None:
    path.write_text(
        '[{"name": "layout_fidelity", "status": "failed"}]',
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
