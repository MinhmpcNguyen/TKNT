from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from experiments.image_flow_eval.image_flow_eval.ablations import (
    _SIMPLE_PHOTOREALIZATION_PROMPT,
    apply_ablation,
)
from experiments.image_flow_eval.image_flow_eval.io import load_manifest
from experiments.image_flow_eval.image_flow_eval.preflight import (
    VisualPreflightResult,
    _cache_key,
    _metadata_summary,
    _normalize_report,
    _preflight_prompt,
    compile_visual_preflight_prompt,
    run_visual_preflight,
)
from experiments.image_flow_eval.image_flow_eval.providers import run_case_variant
from experiments.image_flow_eval.image_flow_eval.types import (
    BBoxNorm,
    CaseConfig,
    EditOperationConfig,
    RenderPresetSelection,
    VariantConfig,
)


class VisualPreflightTest(unittest.TestCase):
    def test_normalize_report_keeps_grounded_required_counts_only(self) -> None:
        result = _normalize_report(
            {
                "source_valid": True,
                "risk_profile": ["repeated_objects", "unsupported"],
                "required_counts": {"armchair": 2, "coffee_table": 99},
                "priority_locks": ["Keep both armchairs visible."],
                "reason": "Repeated seating may merge.",
            },
            source_payload={
                "visibleObjects": [
                    {"id": "armchair", "type": "armchair"},
                    {"id": "armchair_2", "type": "armchair_2"},
                    {"id": "coffee_table", "type": "coffee_table"},
                ]
            },
            model="gpt-5.4-mini",
            cache_key="cache",
            cache_status="miss",
            usage={},
        )

        self.assertEqual(result.required_counts, {"armchair": 2})
        self.assertEqual(result.risk_profile, ("repeated_objects",))
        self.assertIn(
            "Preserve exactly 2 visible armchair instance(s).",
            compile_visual_preflight_prompt(result),
        )

    def test_compile_prompt_adds_zero_window_metadata_lock(self) -> None:
        result = VisualPreflightResult(
            source_valid=True,
            risk_profile=("opening_visibility",),
            required_counts={},
            priority_locks=("Do not change the room framing.",),
            reason="No clear openings are visible.",
            model="gpt-5.4-mini",
            cache_key="cache",
            cache_status="miss",
            usage={},
        )

        prompt = compile_visual_preflight_prompt(
            result,
            source_payload={
                "room": {
                    "openings": {
                        "doors": [],
                        "windows": [],
                    }
                }
            },
            profile="room_shell",
        )

        self.assertIn("preserve exactly 0 window opening(s)", prompt)
        self.assertIn("do not create exterior garden/city/mountain views", prompt)

    def test_compile_prompt_adds_room_shell_locks(self) -> None:
        result = VisualPreflightResult(
            source_valid=True,
            risk_profile=("cropped_room_shell",),
            required_counts={},
            priority_locks=(),
            room_shell_locks=(
                "Preserve the narrow foreground wall plane clipped along the "
                "far left edge.",
            ),
            reason="A cropped wall strip is visible at the image border.",
            model="gpt-5.4-mini",
            cache_key="cache",
            cache_status="miss",
            usage={},
        )

        prompt = compile_visual_preflight_prompt(
            result,
            source_payload={
                "room": {
                    "polygonMm": [
                        {"x": 0, "y": 0},
                        {"x": 4, "y": 0},
                        {"x": 4, "y": 4},
                        {"x": 0, "y": 4},
                    ],
                    "openings": {
                        "doors": [],
                        "windows": [],
                    },
                }
            },
            profile="room_shell",
        )

        self.assertIn("closed room shell footprint with 4 wall segment(s)", prompt)
        self.assertIn("narrow foreground wall plane clipped along the far left", prompt)

    def test_preflight_prompt_requests_room_shell_inspection(self) -> None:
        prompt = _preflight_prompt(
            metadata_summary={},
            include_metadata=True,
            profile="room_shell",
        )

        self.assertIn("First inspect the room shell before furniture", prompt)
        self.assertIn("room_shell_locks", prompt)
        self.assertIn("cropped_room_shell", prompt)

    def test_classic_preflight_prompt_does_not_request_room_shell_locks(self) -> None:
        prompt = _preflight_prompt(
            metadata_summary={},
            include_metadata=True,
            profile="classic",
        )

        self.assertNotIn("room_shell_locks", prompt)
        self.assertNotIn("cropped_room_shell", prompt)
        self.assertIn("priority_locks", prompt)

    def test_object_anchor_prompt_adds_shadow_and_bbox_locks(self) -> None:
        result = VisualPreflightResult(
            source_valid=True,
            risk_profile=("cropped_room_shell",),
            required_counts={"armchair": 2},
            priority_locks=("Keep low media furniture on the right wall.",),
            room_shell_locks=("Preserve the left cropped wall strip.",),
            object_anchor_locks=("Keep the small speaker beside the media shelf.",),
            reason="Room shell and object anchors are fragile.",
            model="gpt-5.4-mini",
            cache_key="cache",
            cache_status="miss",
            usage={},
            preflight_profile="object_anchors",
        )

        prompt = compile_visual_preflight_prompt(
            result,
            source_payload={
                "canvas": {"widthPx": 100, "heightPx": 100},
                "room": {
                    "polygonMm": [
                        {"x": 0, "y": 0},
                        {"x": 4, "y": 0},
                        {"x": 4, "y": 4},
                        {"x": 0, "y": 4},
                    ],
                    "openings": {"doors": [], "windows": []},
                },
                "visibleObjects": [
                    {
                        "id": "floor_lamp",
                        "type": "floor_lamp",
                        "screenBboxPx": {
                            "minX": 0,
                            "minY": 80,
                            "maxX": 45,
                            "maxY": 100,
                        },
                    }
                ],
            },
        )

        self.assertIn("Geometry-vs-shadow lock", prompt)
        self.assertIn("Object anchor lock: preserve floor_lamp", prompt)
        self.assertIn("left edge and bottom edge", prompt)
        self.assertIn("Keep the small speaker", prompt)

    def test_edit_scope_preflight_prompt_requests_local_edit_scope(self) -> None:
        prompt = _preflight_prompt(
            metadata_summary={
                "edit_operations": [
                    {
                        "object_id": "manual_box_1",
                        "object_name": "wardrobe",
                        "operation_type": "recolor",
                        "bbox_norm": {
                            "x": 0.2,
                            "y": 0.3,
                            "width": 0.4,
                            "height": 0.5,
                        },
                    }
                ]
            },
            include_metadata=True,
            profile="edit_scope",
        )

        self.assertIn("clean approved photoreal render", prompt)
        self.assertIn("identify the selected target", prompt)
        self.assertIn("Everything outside the selected bbox/target must remain", prompt)
        self.assertIn("edit_scope_locks", prompt)

    def test_edit_object_scope_preflight_prompt_handles_partial_bbox(self) -> None:
        prompt = _preflight_prompt(
            metadata_summary={
                "edit_operations": [
                    {
                        "object_id": "manual_box_1",
                        "object_name": "wardrobe",
                        "operation_type": "recolor",
                        "target_color": "muted forest green",
                        "bbox_norm": {
                            "x": 0.2,
                            "y": 0.3,
                            "width": 0.4,
                            "height": 0.5,
                        },
                    }
                ]
            },
            include_metadata=True,
            profile="edit_object_scope",
        )

        self.assertIn("bbox is not the object itself", prompt)
        self.assertIn("infer the exact target object", prompt)
        self.assertIn("bbox cuts through part of the object", prompt)
        self.assertIn("foreground/neighboring objects", prompt)
        self.assertIn("decorative motifs", prompt)

    def test_edit_scope_compile_prompt_adds_operation_scope_locks(self) -> None:
        result = VisualPreflightResult(
            source_valid=True,
            risk_profile=("bbox_boundary_risk",),
            required_counts={},
            priority_locks=("Do not alter the desk or walls.",),
            edit_scope_locks=("Keep the wardrobe silhouette and handles fixed.",),
            reason="Wardrobe recolor is local.",
            model="gpt-5.4-mini",
            cache_key="cache",
            cache_status="miss",
            usage={},
            preflight_profile="edit_scope",
        )

        prompt = compile_visual_preflight_prompt(
            result,
            profile="edit_scope",
            edit_operations=(
                EditOperationConfig(
                    object_id="manual_box_1",
                    object_name="wardrobe",
                    operation_type="recolor",
                    target_color="muted forest green",
                    bbox_norm=BBoxNorm(x=0.2, y=0.3, width=0.4, height=0.5),
                ),
            ),
        )

        self.assertIn("Edit-specific visual preflight locks", prompt)
        self.assertIn("muted forest green", prompt)
        self.assertIn("bbox_norm x=0.2000", prompt)
        self.assertIn("preserve camera, room geometry", prompt)
        self.assertIn("Keep the wardrobe silhouette", prompt)
        self.assertNotIn("Metadata opening lock", prompt)

    def test_llm_visual_preflight_variant_appends_compiled_locks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_payload = root / "snapshot_payload.json"
            snapshot_payload.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 8, "heightPx": 8},
                        "visibleObjects": [{"id": "sofa", "type": "sofa"}],
                    }
                ),
                encoding="utf-8",
            )
            snapshot_image = root / "snapshot.png"
            Image.new("RGB", (8, 8), "white").save(snapshot_image)
            case = CaseConfig(
                id="case",
                render_prompt="Photorealize the exact layout.",
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
            )
            variant = VariantConfig(
                id="llm_visual_preflight",
                mode="llm_visual_preflight",
            )
            preflight = VisualPreflightResult(
                source_valid=True,
                risk_profile=("scale_sensitive_object",),
                required_counts={"sofa": 1},
                priority_locks=("Keep the sofa footprint unchanged.",),
                reason="The sofa is the main scale anchor.",
                model="gpt-5.4-mini",
                cache_key="cache",
                cache_status="miss",
                usage={},
            )

            with (
                patch(
                    "experiments.image_flow_eval.image_flow_eval.providers."
                    "run_visual_preflight",
                    return_value=preflight,
                ) as preflight_runner,
                patch(
                    "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                    return_value={"image": {"data_url": _data_url(root)}},
                ) as renderer,
            ):
                run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root / "runs",
                )

            _, kwargs = renderer.call_args
            prompt = str(kwargs["user_prompt"])
            _, preflight_kwargs = preflight_runner.call_args
            self.assertEqual(preflight_kwargs["profile"], "classic")
            self.assertIn("Scene-specific visual preflight locks", prompt)
            self.assertIn("Preserve exactly 1 visible sofa instance(s).", prompt)
            self.assertNotIn("Room shell metadata lock", prompt)

    def test_object_anchor_variant_passes_room_shell_guide_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_payload = root / "snapshot_payload.json"
            snapshot_payload.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 32, "heightPx": 32},
                        "room": {
                            "polygonMm": [
                                {"x": 0, "y": 0},
                                {"x": 4, "y": 0},
                                {"x": 4, "y": 4},
                                {"x": 0, "y": 4},
                            ],
                            "openings": {"doors": [], "windows": []},
                        },
                        "visibleObjects": [
                            {
                                "id": "floor_lamp",
                                "type": "floor_lamp",
                                "screenBboxPx": {
                                    "minX": 0,
                                    "minY": 22,
                                    "maxX": 16,
                                    "maxY": 32,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            snapshot_image = root / "snapshot.png"
            Image.new("RGB", (32, 32), "white").save(snapshot_image)
            case = CaseConfig(
                id="case",
                render_prompt="Photorealize the exact layout.",
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
            )
            variant = VariantConfig(
                id="llm_visual_preflight_object_anchors",
                mode="llm_visual_preflight_object_anchors",
            )
            preflight = VisualPreflightResult(
                source_valid=True,
                risk_profile=("cropped_room_shell",),
                required_counts={},
                priority_locks=(),
                room_shell_locks=("Preserve the left wall strip.",),
                object_anchor_locks=("Keep the cropped floor lamp at the edge.",),
                reason="",
                model="gpt-5.4-mini",
                cache_key="cache",
                cache_status="miss",
                usage={},
                preflight_profile="object_anchors",
            )

            with (
                patch(
                    "experiments.image_flow_eval.image_flow_eval.providers."
                    "run_visual_preflight",
                    return_value=preflight,
                ) as preflight_runner,
                patch(
                    "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                    return_value={"image": {"data_url": _data_url(root)}},
                ) as renderer,
            ):
                run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root / "runs",
                )

            _, preflight_kwargs = preflight_runner.call_args
            self.assertEqual(preflight_kwargs["profile"], "object_anchors")
            _, kwargs = renderer.call_args
            self.assertTrue(
                str(kwargs["annotated_reference_image_data_url"]).startswith(
                    "data:image/"
                )
            )
            prompt = str(kwargs["user_prompt"])
            self.assertIn("Geometry-vs-shadow lock", prompt)
            self.assertIn("Object anchor lock: preserve floor_lamp", prompt)
            metadata = json.loads(
                (
                    root
                    / "runs"
                    / "case"
                    / "llm_visual_preflight_object_anchors"
                    / "tknt_snapshot_result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["visual_preflight_profile"], "object_anchors")
            self.assertTrue(Path(metadata["room_shell_guide"]["image"]).is_file())

    def test_single_edit_edge_silhouette_variant_uses_source_edit_and_structural_guide(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_payload = root / "snapshot_payload.json"
            snapshot_payload.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 64, "heightPx": 64},
                        "camera": {
                            "positionM": {"x": 1.8586, "y": 1.35, "z": -1.6511},
                            "targetM": {"x": 0, "y": 1.026, "z": 0.4575},
                            "fovDeg": 84,
                            "aspect": 1.0,
                        },
                        "room": {
                            "boundsMm": {
                                "minX": 0,
                                "minY": 0,
                                "maxX": 4000,
                                "maxY": 4500,
                            },
                            "polygonMm": [
                                {"x": 0, "y": 0},
                                {"x": 4000, "y": 0},
                                {"x": 4000, "y": 4500},
                                {"x": 0, "y": 4500},
                            ],
                            "openings": {"doors": [], "windows": []},
                        },
                        "visibleObjects": [
                            {
                                "id": "armchair",
                                "type": "armchair",
                                "screenBboxPx": {
                                    "minX": 1,
                                    "minY": 0,
                                    "maxX": 63,
                                    "maxY": 64,
                                },
                            },
                            {
                                "id": "sofa",
                                "type": "sofa",
                                "screenBboxPx": {
                                    "minX": 16,
                                    "minY": 30,
                                    "maxX": 44,
                                    "maxY": 44,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            snapshot_image = root / "snapshot.png"
            Image.new("RGB", (64, 64), "white").save(snapshot_image)
            case = CaseConfig(
                id="case",
                render_prompt="Photorealize the exact layout.",
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
            )
            variant = VariantConfig(
                id="single_edit_edge_silhouette_guided",
                mode="single_edit_edge_silhouette_guided",
            )

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                return_value={"image": {"data_url": _data_url(root)}},
            ) as renderer:
                result = run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root / "runs",
                )

            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertTrue(kwargs["layout_materialization_mode"])
            self.assertTrue(
                kwargs["edit_source_image_data_url"].startswith("data:image/")
            )
            self.assertNotIn("layout_guard_mask_image_data_url", kwargs)
            self.assertTrue(
                kwargs["annotated_reference_image_data_url"].startswith("data:image/")
            )
            self.assertFalse(kwargs["config"].include_preset_reference_images)
            prompt = str(kwargs["user_prompt"])
            self.assertIn("Semantic layout locks", prompt)
            self.assertIn("Shadow-vs-geometry lock", prompt)
            self.assertIn("Rejected-anchor lock", prompt)
            self.assertIn("armchair", prompt)
            self.assertEqual(result.render_image, result.output_image)

            variant_dir = root / "runs" / "case" / "single_edit_edge_silhouette_guided"
            metadata = json.loads(
                (variant_dir / "tknt_snapshot_result.json").read_text(encoding="utf-8")
            )
            guide = metadata["edge_silhouette_guide"]
            self.assertEqual(guide["guide_type"], "room_shell_semantic")
            self.assertEqual(guide["debug_guide_type"], "edge_silhouette")
            self.assertTrue(Path(guide["guide_image"]).is_file())
            self.assertTrue(Path(guide["debug_image"]).is_file())
            self.assertIsNone(guide["mask_image"])
            self.assertFalse(guide["mask_used_for_openai"])
            self.assertEqual(guide["mask_strategy"], "disabled_structural_guide_only")
            self.assertEqual(guide["semantic_projection_status"], "ok")
            self.assertGreater(guide["semantic_surface_count"], 0)
            self.assertTrue(guide["semantic_room_shell_locks"])
            self.assertIn("near_crop_edges", guide["semantic_surfaces"][0])
            self.assertIn("bbox_norm", guide["semantic_surfaces"][0])
            rejected_ids = {item["object_id"] for item in guide["rejected_objects"]}
            self.assertIn("armchair", rejected_ids)
            self.assertIn("semantic_layout_lock_prompt", metadata)

    def test_single_edit_geometry_locked_passes_geometry_refs_and_atlas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_payload = _geometry_snapshot_payload(root)
            snapshot_image = root / "snapshot.png"
            Image.new("RGB", (64, 64), "white").save(snapshot_image)
            _write_geometry_maps(root)
            case = CaseConfig(
                id="case",
                render_prompt="Photorealize the exact layout.",
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
            )
            variant = VariantConfig(
                id="single_edit_geometry_locked",
                mode="single_edit_geometry_locked",
            )

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                return_value={"image": {"data_url": _data_url(root)}},
            ) as renderer:
                run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root / "runs",
                )

            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertTrue(kwargs["layout_materialization_mode"])
            self.assertTrue(
                kwargs["depth_map_image_data_url"].startswith("data:image/")
            )
            self.assertTrue(kwargs["edge_map_image_data_url"].startswith("data:image/"))
            self.assertTrue(
                kwargs["segmentation_map_image_data_url"].startswith("data:image/")
            )
            self.assertEqual(len(kwargs["locked_geometry_atlases"]), 1)
            self.assertTrue(kwargs["config"].include_geometry_references)
            self.assertTrue(kwargs["config"].include_locked_geometry_references)
            self.assertFalse(kwargs["config"].include_preset_reference_images)

            metadata = json.loads(
                (
                    root
                    / "runs"
                    / "case"
                    / "single_edit_geometry_locked"
                    / "tknt_snapshot_result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["geometry_reference_status"], "full_geometry")
            self.assertTrue(
                Path(metadata["geometry_references"]["depth_map"]["path"]).is_file()
            )
            self.assertTrue(Path(metadata["locked_geometry"]["atlas_image"]).is_file())

    def test_two_pass_geometry_repair_uses_pass1_source_and_repair_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_payload = _geometry_snapshot_payload(root)
            snapshot_image = root / "snapshot.png"
            Image.new("RGB", (64, 64), "white").save(snapshot_image)
            _write_geometry_maps(root)
            case = CaseConfig(
                id="case",
                render_prompt="Photorealize the exact layout.",
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
            )
            variant = VariantConfig(
                id="two_pass_geometry_repair",
                mode="two_pass_geometry_repair",
            )

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                return_value={"image": {"data_url": _data_url(root)}},
            ) as renderer:
                run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root / "runs",
                )

            self.assertEqual(renderer.call_count, 2)
            first_kwargs = renderer.call_args_list[0].kwargs
            second_kwargs = renderer.call_args_list[1].kwargs
            self.assertEqual(first_kwargs["render_mode"], "edit")
            self.assertNotIn("layout_guard_mask_image_data_url", first_kwargs)
            self.assertEqual(second_kwargs["render_mode"], "edit")
            self.assertEqual(
                second_kwargs["scene_reference_mode"],
                "scene_reference_camera_transfer",
            )
            self.assertTrue(
                second_kwargs["layout_guard_mask_image_data_url"].startswith(
                    "data:image/"
                )
            )
            self.assertIn(
                "Second pass layout repair", str(second_kwargs["user_prompt"])
            )

            variant_dir = root / "runs" / "case" / "two_pass_geometry_repair"
            metadata = json.loads(
                (variant_dir / "tknt_snapshot_result.json").read_text(encoding="utf-8")
            )
            self.assertTrue((variant_dir / "render_pass1.png").is_file())
            mask_path = Path(metadata["repair_mask"]["image"])
            self.assertTrue(mask_path.is_file())
            with Image.open(mask_path) as mask:
                alpha = mask.convert("RGBA").getchannel("A")
                get_flattened_data = getattr(alpha, "get_flattened_data", None)
                alpha_values = list(
                    get_flattened_data()
                    if callable(get_flattened_data)
                    else alpha.getdata()
                )
            self.assertIn(0, alpha_values)
            self.assertIn(255, alpha_values)
            self.assertEqual(
                metadata["repair_mask"]["mask_semantics"],
                "transparent_pixels_are_editable_opaque_pixels_are_preserved",
            )
            self.assertIn("pass1", metadata["pass_results"])
            self.assertIn("pass2", metadata["pass_results"])

    def test_ablation_mode_selects_expected_preflight_strategy(self) -> None:
        case = CaseConfig(id="case")

        preflight = apply_ablation(
            case,
            VariantConfig(id="llm_visual_preflight", mode="llm_visual_preflight"),
        )

        self.assertEqual(preflight.visual_preflight_mode, "snapshot_and_metadata")
        self.assertEqual(preflight.visual_preflight_profile, "classic")

        object_anchors = apply_ablation(
            case,
            VariantConfig(
                id="llm_visual_preflight_object_anchors",
                mode="llm_visual_preflight_object_anchors",
            ),
        )

        self.assertEqual(object_anchors.visual_preflight_mode, "snapshot_and_metadata")
        self.assertEqual(object_anchors.visual_preflight_profile, "object_anchors")
        self.assertTrue(
            object_anchors.renderer_config_overrides["include_annotated_reference"]
        )

        single_edit = apply_ablation(
            case,
            VariantConfig(
                id="single_edit_edge_silhouette_guided",
                mode="single_edit_edge_silhouette_guided",
            ),
        )

        self.assertTrue(single_edit.single_call_layout_guard)
        self.assertEqual(single_edit.visual_preflight_mode, "disabled")
        self.assertTrue(
            single_edit.renderer_config_overrides["include_annotated_reference"]
        )
        self.assertFalse(
            single_edit.renderer_config_overrides["include_preset_reference_images"]
        )

        geometry_locked = apply_ablation(
            case,
            VariantConfig(
                id="single_edit_geometry_locked",
                mode="single_edit_geometry_locked",
            ),
        )

        self.assertTrue(geometry_locked.single_call_layout_guard)
        self.assertTrue(geometry_locked.geometry_locked_materialization)
        self.assertTrue(
            geometry_locked.renderer_config_overrides["include_geometry_references"]
        )
        self.assertTrue(
            geometry_locked.renderer_config_overrides[
                "include_locked_geometry_references"
            ]
        )

        two_pass = apply_ablation(
            case,
            VariantConfig(
                id="two_pass_geometry_repair",
                mode="two_pass_geometry_repair",
            ),
        )

        self.assertTrue(two_pass.single_call_layout_guard)
        self.assertTrue(two_pass.geometry_locked_materialization)
        self.assertTrue(two_pass.two_pass_geometry_repair)

    def test_simple_prompt_uses_fixed_photorealization_prompt(self) -> None:
        case = CaseConfig(
            id="case",
            render_prompt="Design and arrange a new room with a sofa.",
        )

        ablated = apply_ablation(
            case,
            VariantConfig(id="simple_prompt", mode="simple_prompt"),
        )

        self.assertEqual(ablated.render_prompt, _SIMPLE_PHOTOREALIZATION_PROMPT)
        self.assertNotIn("Design", str(ablated.render_prompt))

    def test_instance_overlay_variants_request_annotated_reference_only(self) -> None:
        case = CaseConfig(
            id="case",
            render_prompt="Photorealize the exact layout.",
        )

        simple = apply_ablation(
            case,
            VariantConfig(
                id="simple_prompt_with_instance_overlay",
                mode="simple_prompt_with_instance_overlay",
            ),
        )
        llm = apply_ablation(
            case,
            VariantConfig(
                id="llm_visual_preflight_with_instance_overlay",
                mode="llm_visual_preflight_with_instance_overlay",
            ),
        )

        self.assertEqual(simple.render_prompt, _SIMPLE_PHOTOREALIZATION_PROMPT)
        self.assertTrue(simple.instance_overlay_reference)
        self.assertTrue(simple.renderer_config_overrides["include_annotated_reference"])
        self.assertFalse(simple.renderer_config_overrides["include_geometry_references"])
        self.assertTrue(llm.instance_overlay_reference)
        self.assertEqual(llm.visual_preflight_mode, "snapshot_and_metadata")
        self.assertTrue(llm.renderer_config_overrides["include_annotated_reference"])
        self.assertFalse(llm.renderer_config_overrides["include_geometry_references"])

    def test_simple_overlay_variant_passes_overlay_as_annotated_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_payload = _geometry_snapshot_payload(root)
            snapshot_image = root / "snapshot.png"
            Image.new("RGB", (64, 64), "white").save(snapshot_image)
            _write_geometry_maps(root)
            case = CaseConfig(
                id="case",
                render_prompt="Photorealize the exact layout.",
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
            )
            variant = VariantConfig(
                id="simple_prompt_with_instance_overlay",
                mode="simple_prompt_with_instance_overlay",
            )

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                return_value={"image": {"data_url": _data_url(root)}},
            ) as renderer:
                run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root / "runs",
                )

            _, kwargs = renderer.call_args
            self.assertEqual(kwargs["render_mode"], "generate")
            self.assertTrue(
                str(kwargs["annotated_reference_image_data_url"]).startswith(
                    "data:image/"
                )
            )
            self.assertNotIn("depth_map_image_data_url", kwargs)
            self.assertNotIn("edge_map_image_data_url", kwargs)
            self.assertNotIn("segmentation_map_image_data_url", kwargs)
            self.assertTrue(kwargs["config"].include_annotated_reference)
            self.assertFalse(kwargs["config"].include_geometry_references)

            metadata = json.loads(
                (
                    root
                    / "runs"
                    / "case"
                    / "simple_prompt_with_instance_overlay"
                    / "tknt_snapshot_result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["geometry_reference_status"], "full_geometry")
            overlay = metadata["geometry_references"]["instance_outline_overlay"]
            self.assertTrue(Path(overlay["path"]).is_file())

    def test_provider_passes_case_preset_selection_to_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_payload = root / "snapshot_payload.json"
            snapshot_payload.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 8, "heightPx": 8},
                        "visibleObjects": [],
                    }
                ),
                encoding="utf-8",
            )
            snapshot_image = root / "snapshot.png"
            Image.new("RGB", (8, 8), "white").save(snapshot_image)
            case = CaseConfig(
                id="case",
                render_prompt="Photorealize the exact layout.",
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
                preset_selection=RenderPresetSelection(
                    style="japandi",
                    lighting="daylight_6",
                    scenery="garden",
                ),
            )
            variant = VariantConfig(id="simple_prompt", mode="simple_prompt")

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                return_value={"image": {"data_url": _data_url(root)}},
            ) as renderer:
                run_case_variant(
                    provider="tknt_snapshot",
                    case=case,
                    variant=variant,
                    output_root=root / "runs",
                )

            _, kwargs = renderer.call_args
            self.assertEqual(
                kwargs["preset_selection"],
                {
                    "style": "japandi",
                    "lighting": "daylight_6",
                    "scenery": "garden",
                },
            )

    def test_combined_variants_keep_contract_and_select_preflight_strategy(
        self,
    ) -> None:
        case = CaseConfig(id="case")

        metadata = apply_ablation(
            case,
            VariantConfig(
                id="contract_objects_preflight",
                mode="contract_objects_preflight",
            ),
        )
        locked_geometry = apply_ablation(
            case,
            VariantConfig(
                id="contract_objects_preflight_locked_geometry",
                mode="contract_objects_preflight_locked_geometry",
            ),
        )
        visual_only = apply_ablation(
            case,
            VariantConfig(
                id="contract_objects_preflight_visual_only",
                mode="contract_objects_preflight_visual_only",
            ),
        )

        self.assertTrue(metadata.renderer_config_overrides["text_contract_mode"])
        self.assertEqual(metadata.visual_preflight_mode, "snapshot_and_metadata")
        self.assertEqual(metadata.visual_preflight_profile, "room_shell")
        self.assertTrue(locked_geometry.renderer_config_overrides["text_contract_mode"])
        self.assertEqual(
            locked_geometry.visual_preflight_mode,
            "snapshot_and_metadata",
        )
        self.assertEqual(locked_geometry.visual_preflight_profile, "room_shell")
        self.assertTrue(
            locked_geometry.renderer_config_overrides[
                "include_locked_geometry_references"
            ]
        )
        self.assertTrue(visual_only.renderer_config_overrides["text_contract_mode"])
        self.assertEqual(visual_only.visual_preflight_mode, "snapshot_only")
        self.assertEqual(visual_only.visual_preflight_profile, "room_shell")

    def test_shared_cache_is_reused_across_variant_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "snapshot.png"
            Image.new("RGB", (8, 8), "white").save(image)
            payload = {"visibleObjects": [{"id": "sofa", "type": "sofa"}]}
            model = "gpt-5.4-mini"
            cache_key = _cache_key(
                source_image=image,
                metadata_summary=_metadata_summary(payload),
                include_metadata=True,
                model=model,
                profile="room_shell",
            )
            cache_dir = root / "cache"
            cache_dir.mkdir()
            (cache_dir / f"{cache_key}.json").write_text(
                json.dumps(
                    {
                        "source_valid": True,
                        "risk_profile": [],
                        "required_counts": {},
                        "priority_locks": ["Keep the sofa footprint unchanged."],
                        "reason": "",
                        "model": model,
                        "cache_key": cache_key,
                        "cache_status": "miss",
                        "usage": {},
                    }
                ),
                encoding="utf-8",
            )
            report_path = root / "variant" / "visual_preflight_report.json"

            with patch(
                "experiments.image_flow_eval.image_flow_eval.preflight._model",
                return_value=model,
            ):
                result = run_visual_preflight(
                    source_image=image,
                    source_payload=payload,
                    include_metadata=True,
                    report_path=report_path,
                    cache_dir=cache_dir,
                    profile="room_shell",
                )

            self.assertEqual(result.cache_status, "hit")
            self.assertTrue(report_path.exists())

    def test_edit_preflight_ablations_isolate_dsl_and_mask(self) -> None:
        case = CaseConfig(
            id="case",
            edit_operations=(
                EditOperationConfig(
                    object_id="sofa",
                    bbox_norm=BBoxNorm(x=0.1, y=0.1, width=0.3, height=0.2),
                ),
            ),
        )

        full = apply_ablation(
            case,
            VariantConfig(
                id="operation_dsl_preflight",
                mode="operation_dsl_preflight",
            ),
        )
        no_dsl = apply_ablation(
            case,
            VariantConfig(
                id="operation_dsl_preflight_no_dsl",
                mode="operation_dsl_preflight_no_dsl",
            ),
        )
        no_mask = apply_ablation(
            case,
            VariantConfig(
                id="operation_dsl_preflight_no_edit_mask",
                mode="operation_dsl_preflight_no_edit_mask",
            ),
        )
        image_only = apply_ablation(
            case,
            VariantConfig(
                id="edit_prompt_image_only",
                mode="edit_prompt_image_only",
            ),
        )
        object_scope = apply_ablation(
            case,
            VariantConfig(
                id="edit_object_scope_preflight",
                mode="edit_object_scope_preflight",
            ),
        )

        self.assertTrue(full.renderer_config_overrides["include_edit_operation_dsl"])
        self.assertFalse(no_dsl.renderer_config_overrides["include_edit_operation_dsl"])
        self.assertEqual(full.visual_preflight_mode, "snapshot_and_metadata")
        self.assertEqual(full.visual_preflight_profile, "edit_scope")
        self.assertEqual(no_dsl.visual_preflight_profile, "edit_scope")
        self.assertEqual(no_mask.visual_preflight_profile, "edit_scope")
        self.assertIsNone(no_mask.edit_operations[0].bbox_norm)
        self.assertEqual(image_only.edit_operations, case.edit_operations)
        self.assertIsNotNone(image_only.edit_operations[0].bbox_norm)
        self.assertTrue(
            image_only.renderer_config_overrides["include_annotated_reference"]
        )
        self.assertEqual(object_scope.visual_preflight_mode, "snapshot_and_metadata")
        self.assertEqual(object_scope.visual_preflight_profile, "edit_object_scope")
        self.assertTrue(
            object_scope.renderer_config_overrides["include_annotated_reference"]
        )
        self.assertTrue(
            object_scope.renderer_config_overrides["include_edit_operation_dsl"]
        )
        self.assertTrue(object_scope.renderer_config_overrides["disable_edit_mask"])

        replace_reference = apply_ablation(
            CaseConfig(
                id="case",
                edit_operations=(
                    EditOperationConfig(
                        object_id="sofa",
                        bbox_norm=BBoxNorm(x=0.1, y=0.1, width=0.3, height=0.2),
                        replacement_image=Path("/tmp/replacement.png"),
                    ),
                ),
            ),
            VariantConfig(
                id="edit_prompt_image_only",
                mode="edit_prompt_image_only",
            ),
        )
        self.assertEqual(len(replace_reference.edit_operations), 1)
        self.assertIsNotNone(replace_reference.edit_operations[0].bbox_norm)
        self.assertEqual(
            replace_reference.edit_operations[0].replacement_image,
            Path("/tmp/replacement.png"),
        )
        self.assertFalse(image_only.renderer_config_overrides["text_contract_mode"])
        self.assertFalse(
            image_only.renderer_config_overrides["include_edit_operation_dsl"]
        )

    def test_manifest_accepts_preflight_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "case",
                                "snapshot_image": "snapshot.png",
                                "variants": [
                                    {"id": "llm_visual_preflight"},
                                    {
                                        "id": (
                                            "llm_visual_preflight_with_instance_overlay"
                                        )
                                    },
                                    {"id": "llm_visual_preflight_object_anchors"},
                                    {"id": "llm_visual_preflight_no_metadata"},
                                    {"id": "simple_prompt_with_instance_overlay"},
                                    {"id": "single_edit_edge_silhouette_guided"},
                                    {"id": "single_edit_geometry_locked"},
                                    {"id": "two_pass_geometry_repair"},
                                    {"id": "edit_object_scope_preflight"},
                                    {"id": "contract_objects_preflight"},
                                    {
                                        "id": (
                                            "contract_objects_preflight_locked_geometry"
                                        )
                                    },
                                    {"id": "contract_objects_preflight_visual_only"},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            case = load_manifest(manifest_path).cases[0]

        self.assertEqual(
            [variant.mode for variant in case.variants],
            [
                "llm_visual_preflight",
                "llm_visual_preflight_with_instance_overlay",
                "llm_visual_preflight_object_anchors",
                "llm_visual_preflight_no_metadata",
                "simple_prompt_with_instance_overlay",
                "single_edit_edge_silhouette_guided",
                "single_edit_geometry_locked",
                "two_pass_geometry_repair",
                "edit_object_scope_preflight",
                "contract_objects_preflight",
                "contract_objects_preflight_locked_geometry",
                "contract_objects_preflight_visual_only",
            ],
        )


def _data_url(root: Path) -> str:
    path = root / "output.png"
    Image.new("RGB", (2, 2), "blue").save(path)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(
        "ascii"
    )


def _geometry_snapshot_payload(root: Path) -> Path:
    snapshot_payload = root / "snapshot_payload.json"
    snapshot_payload.write_text(
        json.dumps(
            {
                "canvas": {"widthPx": 64, "heightPx": 64},
                "camera": {
                    "positionM": {"x": 1.8, "y": 1.35, "z": -1.6},
                    "targetM": {"x": 0, "y": 1.0, "z": 0.45},
                    "fovDeg": 84,
                    "aspect": 1.0,
                },
                "room": {
                    "boundsMm": {
                        "minX": 0,
                        "minY": 0,
                        "maxX": 4000,
                        "maxY": 4500,
                    },
                    "polygonMm": [
                        {"x": 0, "y": 0},
                        {"x": 4000, "y": 0},
                        {"x": 4000, "y": 4500},
                        {"x": 0, "y": 4500},
                    ],
                    "openings": {
                        "doors": [],
                        "windows": [
                            {
                                "screenBboxPx": {
                                    "minX": 8,
                                    "minY": 8,
                                    "maxX": 20,
                                    "maxY": 24,
                                }
                            }
                        ],
                    },
                },
                "visibleObjects": [
                    {
                        "id": "sofa",
                        "type": "sofa",
                        "screenBboxPx": {
                            "minX": 18,
                            "minY": 32,
                            "maxX": 46,
                            "maxY": 46,
                        },
                    },
                    {
                        "id": "speaker",
                        "type": "speaker",
                        "screenBboxPx": {
                            "minX": 48,
                            "minY": 24,
                            "maxX": 54,
                            "maxY": 42,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return snapshot_payload


def _write_geometry_maps(root: Path) -> None:
    Image.new("RGB", (64, 64), "black").save(root / "depth_map.png")
    Image.new("RGB", (64, 64), "white").save(root / "edge_map.png")
    Image.new("RGB", (64, 64), "red").save(root / "segmentation_map.png")
    Image.new("RGB", (64, 64), "green").save(root / "instance_outline_guide.png")
    Image.new("RGB", (64, 64), "blue").save(root / "instance_outline_overlay.png")


if __name__ == "__main__":
    unittest.main()
