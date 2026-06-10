from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .types import (
    CaseConfig,
    EditOperationConfig,
    VariantConfig,
    VisualPreflightMode,
    VisualPreflightProfile,
)

_SIMPLE_PHOTOREALIZATION_PROMPT = (
    "make this more photorealistic, don't change anything."
)
_SINGLE_EDIT_EDGE_GUIDED_PROMPT = (
    "Photorealize this exact 3D source view. Preserve the camera, crop, room "
    "shell, openings, object count, object positions, object scale, silhouettes, "
    "colors, and depth order. Do not add windows, doors, decor, furniture, or "
    "foreground objects."
)
_SINGLE_EDIT_GEOMETRY_LOCKED_PROMPT = (
    "Photorealize this exact 3D source view using the geometry reference images "
    "as hard structural evidence. Preserve camera, crop, wall planes, openings, "
    "object count, footprint, silhouette, pixel scale, depth order, occlusion, "
    "and TV/media/object internal details. Use only text presets for lighting, "
    "style, and scenery; do not add windows, doors, furniture, decor, or "
    "foreground objects."
)
_TWO_PASS_GEOMETRY_REPAIR_PROMPT = (
    "Photorealize this exact 3D source view, then repair any layout drift against "
    "the geometry references. Preserve camera, crop, room shell, object footprint, "
    "object silhouette, object pixel scale, depth order, occlusion, openings, and "
    "TV/media/object clusters. Do not introduce new architecture, openings, "
    "furniture, decor, or foreground objects."
)


@dataclass(frozen=True)
class AblatedInputs:
    render_prompt: str | None
    edit_prompt: str | None
    layout_reference_image: Path | None
    annotated_reference_image: Path | None
    scene_reference_image: Path | None
    edit_operations: tuple[EditOperationConfig, ...]
    renderer_config_overrides: dict[str, object]
    visual_preflight_mode: VisualPreflightMode
    visual_preflight_profile: VisualPreflightProfile
    instance_overlay_reference: bool = False
    single_call_layout_guard: bool = False
    geometry_locked_materialization: bool = False
    two_pass_geometry_repair: bool = False


def apply_ablation(case: CaseConfig, variant: VariantConfig) -> AblatedInputs:
    edit_prompt = variant.prompt or case.edit_prompt
    render_prompt = variant.prompt or case.render_prompt
    layout_reference_image = case.layout_reference_image
    annotated_reference_image = case.annotated_reference_image
    scene_reference_image = case.scene_reference_image
    edit_operations = case.edit_operations
    renderer_config_overrides: dict[str, object] = {}
    visual_preflight_mode: VisualPreflightMode = "disabled"
    visual_preflight_profile: VisualPreflightProfile = "classic"

    if variant.mode in {
        "simple_prompt",
        "simple_prompt_with_instance_overlay",
        "llm_visual_preflight",
        "llm_visual_preflight_with_instance_overlay",
        "llm_visual_preflight_object_anchors",
        "llm_visual_preflight_no_metadata",
    }:
        renderer_config_overrides.update(
            {
                "simple_prompt_mode": True,
                "text_contract_mode": False,
                "text_contract_object_limit": 0,
                "include_layout_2d_reference": False,
                "include_annotated_reference": False,
                "include_geometry_references": False,
                "include_locked_geometry_references": False,
                "strict_layout_mode": False,
            }
        )

    if variant.mode == "edit_prompt_image_only":
        layout_reference_image = None
        scene_reference_image = None
        renderer_config_overrides.update(
            {
                "simple_prompt_mode": True,
                "text_contract_mode": False,
                "text_contract_object_limit": 0,
                "include_layout_2d_reference": False,
                "include_annotated_reference": True,
                "include_geometry_references": False,
                "include_locked_geometry_references": False,
                "include_preset_reference_images": False,
                "include_edit_operation_dsl": False,
                "strict_layout_mode": False,
            }
        )

    if variant.mode == "edit_object_scope_preflight":
        layout_reference_image = None
        scene_reference_image = None
        visual_preflight_mode = "snapshot_and_metadata"
        visual_preflight_profile = "edit_object_scope"
        renderer_config_overrides.update(
            {
                "simple_prompt_mode": True,
                "text_contract_mode": True,
                "text_contract_object_limit": 10,
                "include_layout_2d_reference": False,
                "include_annotated_reference": True,
                "include_geometry_references": False,
                "include_locked_geometry_references": False,
                "include_preset_reference_images": False,
                "include_edit_operation_dsl": True,
                "disable_edit_mask": True,
                "strict_layout_mode": False,
            }
        )

    if variant.mode == "simple_prompt":
        render_prompt = _SIMPLE_PHOTOREALIZATION_PROMPT

    if variant.mode == "simple_prompt_with_instance_overlay":
        render_prompt = _SIMPLE_PHOTOREALIZATION_PROMPT
        renderer_config_overrides["include_annotated_reference"] = True

    if variant.mode == "llm_visual_preflight":
        visual_preflight_mode = "snapshot_and_metadata"

    if variant.mode == "llm_visual_preflight_with_instance_overlay":
        visual_preflight_mode = "snapshot_and_metadata"
        renderer_config_overrides["include_annotated_reference"] = True

    if variant.mode == "llm_visual_preflight_object_anchors":
        visual_preflight_mode = "snapshot_and_metadata"
        visual_preflight_profile = "object_anchors"
        renderer_config_overrides["include_annotated_reference"] = True

    if variant.mode == "llm_visual_preflight_no_metadata":
        visual_preflight_mode = "snapshot_only"

    if variant.mode in {
        "single_edit_edge_silhouette_guided",
        "single_edit_geometry_locked",
        "two_pass_geometry_repair",
    }:
        if variant.mode == "single_edit_geometry_locked":
            render_prompt = _SINGLE_EDIT_GEOMETRY_LOCKED_PROMPT
        elif variant.mode == "two_pass_geometry_repair":
            render_prompt = _TWO_PASS_GEOMETRY_REPAIR_PROMPT
        else:
            render_prompt = _SINGLE_EDIT_EDGE_GUIDED_PROMPT
        renderer_config_overrides.update(
            {
                "simple_prompt_mode": False,
                "text_contract_mode": False,
                "text_contract_object_limit": 0,
                "include_layout_2d_reference": False,
                "include_annotated_reference": True,
                "include_geometry_references": variant.mode
                in {
                    "single_edit_geometry_locked",
                    "two_pass_geometry_repair",
                },
                "include_locked_geometry_references": variant.mode
                in {
                    "single_edit_geometry_locked",
                    "two_pass_geometry_repair",
                },
                "include_preset_reference_images": False,
                "strict_layout_mode": False,
            }
        )

    if variant.mode == "contract_lite":
        renderer_config_overrides.update(
            {
                "simple_prompt_mode": True,
                "text_contract_mode": True,
                "text_contract_object_limit": 0,
                "include_layout_2d_reference": False,
                "include_annotated_reference": False,
                "include_geometry_references": False,
                "include_locked_geometry_references": False,
            }
        )

    if variant.mode in {
        "contract_objects",
        "contract_objects_preflight",
        "contract_objects_preflight_locked_geometry",
        "contract_objects_preflight_visual_only",
        "operation_dsl",
        "operation_dsl_preflight",
        "operation_dsl_preflight_no_dsl",
        "operation_dsl_preflight_no_edit_mask",
    }:
        renderer_config_overrides.update(
            {
                "simple_prompt_mode": True,
                "text_contract_mode": True,
                "text_contract_object_limit": 10,
                "include_layout_2d_reference": False,
                "include_annotated_reference": False,
                "include_geometry_references": False,
                "include_locked_geometry_references": False,
            }
        )

    if variant.mode in {
        "contract_objects_preflight",
        "contract_objects_preflight_locked_geometry",
    }:
        visual_preflight_mode = "snapshot_and_metadata"
        visual_preflight_profile = "room_shell"

    if variant.mode in {
        "operation_dsl_preflight",
        "operation_dsl_preflight_no_dsl",
        "operation_dsl_preflight_no_edit_mask",
    }:
        visual_preflight_mode = "snapshot_and_metadata"
        visual_preflight_profile = "edit_scope"

    if variant.mode == "edit_object_scope_preflight":
        visual_preflight_mode = "snapshot_and_metadata"
        visual_preflight_profile = "edit_object_scope"

    if variant.mode == "contract_objects_preflight_locked_geometry":
        renderer_config_overrides["include_locked_geometry_references"] = True

    if variant.mode == "contract_objects_preflight_visual_only":
        visual_preflight_mode = "snapshot_only"
        visual_preflight_profile = "room_shell"

    if variant.mode in {"operation_dsl", "operation_dsl_preflight"}:
        renderer_config_overrides["include_edit_operation_dsl"] = True

    if variant.mode == "edit_object_scope_preflight":
        renderer_config_overrides["include_edit_operation_dsl"] = True

    if variant.mode == "operation_dsl_preflight_no_dsl":
        renderer_config_overrides["include_edit_operation_dsl"] = False

    if variant.mode == "no_layout_reference":
        layout_reference_image = None
        annotated_reference_image = None

    if variant.mode in {"no_edit_mask", "operation_dsl_preflight_no_edit_mask"}:
        edit_operations = tuple(
            replace(operation, bbox_norm=None) for operation in edit_operations
        )

    if variant.mode == "no_preservation_prompt":
        requested = edit_prompt or render_prompt or "Apply the requested image change."
        edit_prompt = f"Apply this image edit: {requested}"
        render_prompt = requested

    if variant.mode == "no_reference_images":
        scene_reference_image = None
        edit_operations = tuple(
            replace(operation, replacement_image=None) for operation in edit_operations
        )

    if variant.mode == "no_preset_reference_images":
        renderer_config_overrides["include_preset_reference_images"] = False

    return AblatedInputs(
        render_prompt=render_prompt,
        edit_prompt=edit_prompt,
        layout_reference_image=layout_reference_image,
        annotated_reference_image=annotated_reference_image,
        scene_reference_image=scene_reference_image,
        edit_operations=edit_operations,
        renderer_config_overrides=renderer_config_overrides,
        visual_preflight_mode=visual_preflight_mode,
        visual_preflight_profile=visual_preflight_profile,
        instance_overlay_reference=variant.mode
        in {
            "simple_prompt_with_instance_overlay",
            "llm_visual_preflight_with_instance_overlay",
        },
        single_call_layout_guard=variant.mode
        in {
            "single_edit_edge_silhouette_guided",
            "single_edit_geometry_locked",
            "two_pass_geometry_repair",
        },
        geometry_locked_materialization=variant.mode
        in {"single_edit_geometry_locked", "two_pass_geometry_repair"},
        two_pass_geometry_repair=(variant.mode == "two_pass_geometry_repair"),
    )
