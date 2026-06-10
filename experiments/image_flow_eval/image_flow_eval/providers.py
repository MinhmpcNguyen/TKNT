from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError

from .ablations import apply_ablation
from .io import encode_image_to_data_url, write_data_url_image
from .preflight import compile_visual_preflight_prompt, run_visual_preflight
from .types import (
    BBoxNorm,
    CaseConfig,
    JSONValue,
    ProviderName,
    ProviderResult,
    ProviderStatus,
    VariantConfig,
)

if TYPE_CHECKING:
    from pipeline.snapshot_image_renderer import SnapshotEditOperation

ConfigT = TypeVar("ConfigT")
AtlasT = TypeVar("AtlasT")
AtlasObjectT = TypeVar("AtlasObjectT")
Vec3 = tuple[float, float, float]
Point2 = tuple[float, float]


@dataclass(frozen=True)
class LockedGeometryAtlasBuild(Generic[AtlasT]):
    atlases: list[AtlasT]
    metadata: dict[str, JSONValue]


@dataclass(frozen=True)
class RoomShellGuideBuild:
    image_path: Path | None
    metadata: dict[str, JSONValue]


@dataclass(frozen=True)
class EdgeSilhouetteGuideBuild:
    guide_image_path: Path | None
    debug_image_path: Path | None
    mask_image_path: Path | None
    metadata: dict[str, JSONValue]


@dataclass(frozen=True)
class GeometryReferenceBuild:
    depth_map_path: Path | None
    edge_map_path: Path | None
    segmentation_map_path: Path | None
    instance_outline_path: Path | None
    instance_outline_overlay_path: Path | None
    metadata: dict[str, JSONValue]


@dataclass(frozen=True)
class LayoutRepairMaskBuild:
    image_path: Path | None
    metadata: dict[str, JSONValue]


@dataclass(frozen=True)
class ObjectScopeEditMaskBuild:
    image_path: Path | None
    metadata: dict[str, JSONValue]


@dataclass(frozen=True)
class ProjectedRoomShellSurface:
    surface_id: str
    surface_type: str
    polygon_px: list[Point2]
    average_depth_m: float
    touches_crop_edges: tuple[str, ...]
    near_crop_edges: tuple[str, ...]
    bbox_norm: dict[str, float]


_ALIGNED_SOURCE_IMAGE_NAME = "source_3d.png"
_ALIGNED_SOURCE_PAYLOAD_NAME = "source_snapshot_payload.json"
_LOCKED_GEOMETRY_ATLAS_NAME = "locked_geometry_atlas.png"
_ROOM_SHELL_GUIDE_NAME = "room_shell_guide.png"
_ROOM_SHELL_SEMANTIC_GUIDE_NAME = "room_shell_semantic_guide.png"
_EDGE_SILHOUETTE_GUIDE_NAME = "edge_silhouette_guide.png"
_LAYOUT_GUARD_MASK_NAME = "layout_guard_mask.png"
_DEPTH_MAP_NAME = "depth_map.png"
_EDGE_MAP_NAME = "edge_map.png"
_SEGMENTATION_MAP_NAME = "segmentation_map.png"
_INSTANCE_OUTLINE_GUIDE_NAME = "instance_outline_guide.png"
_INSTANCE_OUTLINE_OVERLAY_NAME = "instance_outline_overlay.png"
_RENDER_PASS1_NAME = "render_pass1.png"
_LAYOUT_REPAIR_MASK_NAME = "layout_repair_mask.png"
_OBJECT_SCOPE_EDIT_MASK_NAME = "object_scope_edit_mask.png"
_LOCKED_GEOMETRY_MAX_OBJECTS = 16
_LOCKED_GEOMETRY_CELL_SIZE_PX = 192
_LOCKED_GEOMETRY_CELL_LABEL_HEIGHT_PX = 18
_LOCKED_GEOMETRY_CROP_PADDING_RATIO = 0.08
_LOCKED_GEOMETRY_MIN_BBOX_PX = 4
_ROOM_SHELL_GUIDE_MAX_OBJECTS = 12
_EDGE_SILHOUETTE_GUIDE_MAX_OBJECTS = 16
_ROOM_SHELL_DEFAULT_CEILING_HEIGHT_MM = 2700.0
_ROOM_SHELL_PROJECTION_NEAR_M = 0.05
_ROOM_SHELL_PROJECTED_POINT_LIMIT = 100000.0
_ROOM_SHELL_TOUCH_CROP_MARGIN_RATIO = 0.025
_ROOM_SHELL_NEAR_CROP_MARGIN_RATIO = 0.12
_LAYOUT_GUARD_BORDER_RATIO = 0.018
_LAYOUT_GUARD_OPENING_PADDING_RATIO = 0.01
_LAYOUT_GUARD_DEFAULT_MAX_OBJECT_AREA = 0.22
_LAYOUT_GUARD_MAX_OBJECT_AREA_BY_TYPE: Mapping[str, float] = {
    "armchair": 0.2,
    "bed": 0.34,
    "bookshelf": 0.28,
    "cabinet": 0.3,
    "coffee_table": 0.16,
    "floor_lamp": 0.08,
    "media_shelf": 0.18,
    "nightstand": 0.12,
    "rug": 0.42,
    "side_table": 0.14,
    "sofa": 0.34,
    "speaker": 0.08,
    "table_lamp": 0.08,
    "tv_console": 0.22,
    "vase": 0.08,
    "wardrobe": 0.34,
}
_LAYOUT_REPAIR_OBJECT_TYPES = {
    "armchair",
    "bed",
    "bookshelf",
    "cabinet",
    "chair",
    "coffee_table",
    "desk",
    "floor_lamp",
    "media_shelf",
    "nightstand",
    "rug",
    "side_table",
    "sofa",
    "speaker",
    "table_lamp",
    "tv",
    "tv_console",
    "wardrobe",
}
_TWO_PASS_REPAIR_PROMPT = (
    "Second pass layout repair: edit only the transparent mask regions so they "
    "match Image 1 and the geometry references. Restore any lost crop-edge wall "
    "surfaces, openings, foreground/cropped anchors, rug, sofa/bed, TV/media "
    "cluster, speaker, and large furniture footprints. Preserve opaque masked "
    "areas from the first pass exactly."
)
_LAYOUT_GUARD_SLENDER_OBJECT_TYPES = {
    "bookshelf",
    "curtain",
    "floor_lamp",
    "table_lamp",
    "wall_art",
}
_OBJECT_SCOPE_MASK_PADDING_RATIO = 0.012
_OBJECT_SCOPE_MASK_NEAR_PADDING_RATIO = 0.02
_OBJECT_SCOPE_MAX_LOCKED_TARGET_FRACTION = 0.55
_OBJECT_SCOPE_MIN_EDITABLE_PIXEL_RATIO = 0.002
_OBJECT_SCOPE_MASK_OPERATION_TYPES = frozenset({"add", "replace"})
_APPROVED_RENDER_TIE_BREAK_VARIANTS: tuple[str, ...] = (
    "contract_objects_preflight_locked_geometry",
    "llm_visual_preflight_object_anchors",
    "two_pass_geometry_repair",
    "single_edit_geometry_locked",
    "single_edit_edge_silhouette_guided",
    "contract_objects_preflight",
    "contract_objects",
    "llm_visual_preflight",
    "simple_prompt",
)


def run_case_variant(
    *,
    provider: ProviderName,
    case: CaseConfig,
    variant: VariantConfig,
    output_root: Path,
    skip_existing: bool = False,
) -> ProviderResult:
    if provider == "evaluate_only":
        return _evaluate_only(case=case, variant=variant, output_root=output_root)
    return _run_tknt_snapshot(
        case=case,
        variant=variant,
        output_root=output_root,
        skip_existing=skip_existing,
    )


def resolve_provider(
    default_provider: ProviderName, variant: VariantConfig
) -> ProviderName:
    if variant.provider is not None:
        return variant.provider
    if variant.output_image is not None:
        return "evaluate_only"
    return default_provider


def _evaluate_only(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_root: Path,
) -> ProviderResult:
    variant_dir = output_root / case.id / variant.id
    render_path = variant_dir / "render.png"
    edit_path = variant_dir / "edit.png"
    output_image = variant.output_image
    if output_image is None:
        output_image = edit_path if _is_reusable_file(edit_path) else render_path
    if not _is_reusable_file(output_image):
        raise FileNotFoundError(f"Output image not found: {output_image}")
    source_image_path = variant_dir / _ALIGNED_SOURCE_IMAGE_NAME
    source_payload_path = variant_dir / _ALIGNED_SOURCE_PAYLOAD_NAME
    metadata_path = variant_dir / "tknt_snapshot_result.json"
    return ProviderResult(
        case_id=case.id,
        variant_id=variant.id,
        output_image=output_image,
        metadata_path=metadata_path if _is_reusable_file(metadata_path) else None,
        render_image=render_path if _is_reusable_file(render_path) else None,
        source_image=source_image_path
        if _is_reusable_file(source_image_path)
        else case.source_image or case.snapshot_image,
        source_snapshot_payload=source_payload_path
        if _is_reusable_file(source_payload_path)
        else case.snapshot_payload,
        provider_status="reused_existing",
    )


def _run_tknt_snapshot(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_root: Path,
    skip_existing: bool = False,
) -> ProviderResult:
    if case.snapshot_payload is None:
        raise ValueError(f"Case {case.id} requires snapshot_payload for tknt_snapshot.")
    if case.snapshot_image is None:
        raise ValueError(f"Case {case.id} requires snapshot_image for tknt_snapshot.")
    if not case.snapshot_payload.exists():
        raise FileNotFoundError(f"Snapshot payload not found: {case.snapshot_payload}")
    if not case.snapshot_image.exists():
        raise FileNotFoundError(f"Snapshot image not found: {case.snapshot_image}")

    ablated = apply_ablation(case, variant)
    variant_dir = output_root / case.id / variant.id
    render_path = variant_dir / "render.png"
    edit_path = variant_dir / "edit.png"
    metadata_path = variant_dir / "tknt_snapshot_result.json"
    source_image_path = variant_dir / _ALIGNED_SOURCE_IMAGE_NAME
    source_payload_path = variant_dir / _ALIGNED_SOURCE_PAYLOAD_NAME
    single_call_layout_guard = ablated.single_call_layout_guard
    instance_overlay_reference = ablated.instance_overlay_reference
    geometry_locked_materialization = ablated.geometry_locked_materialization
    two_pass_geometry_repair = ablated.two_pass_geometry_repair
    should_edit = bool(ablated.edit_prompt or ablated.edit_operations)
    approved_render_source = (
        _configured_approved_render_path(output_root=output_root, case=case)
        if should_edit
        else None
    )

    if skip_existing:
        if should_edit and _is_reusable_file(edit_path):
            reusable_render = (
                render_path
                if _is_reusable_file(render_path)
                else approved_render_source
                if approved_render_source is not None
                and _is_reusable_file(approved_render_source)
                else render_path
            )
            return ProviderResult(
                case_id=case.id,
                variant_id=variant.id,
                output_image=edit_path,
                metadata_path=metadata_path
                if _is_reusable_file(metadata_path)
                else None,
                render_image=reusable_render
                if _is_reusable_file(reusable_render)
                else None,
                source_image=source_image_path
                if _is_reusable_file(source_image_path)
                else case.snapshot_image,
                source_snapshot_payload=source_payload_path
                if _is_reusable_file(source_payload_path)
                else case.snapshot_payload,
                approved_render_source=approved_render_source,
                provider_status="reused_existing",
            )
        if (single_call_layout_guard or not should_edit) and _is_reusable_file(
            render_path
        ):
            return ProviderResult(
                case_id=case.id,
                variant_id=variant.id,
                output_image=render_path,
                metadata_path=metadata_path
                if _is_reusable_file(metadata_path)
                else None,
                render_image=render_path,
                source_image=source_image_path
                if _is_reusable_file(source_image_path)
                else case.snapshot_image,
                source_snapshot_payload=source_payload_path
                if _is_reusable_file(source_payload_path)
                else case.snapshot_payload,
                approved_render_source=None,
                provider_status="reused_existing",
            )

    variant_dir.mkdir(parents=True, exist_ok=True)
    if approved_render_source is not None:
        _require_approved_render(
            case=case,
            approved_render_source=approved_render_source,
        )

    from pipeline.snapshot_image_renderer import (
        SnapshotEditOperation,
        SnapshotImageRenderConfig,
        SnapshotLockedGeometryAtlas,
        SnapshotLockedGeometryAtlasObject,
        render_snapshot_image_from_path,
    )

    render_config = _snapshot_render_config_for_ablation(
        SnapshotImageRenderConfig.from_env(),
        ablated.renderer_config_overrides,
    )
    source_alignment = _prepare_aligned_snapshot_source(
        snapshot_payload_path=case.snapshot_payload,
        snapshot_image_path=case.snapshot_image,
        output_payload_path=source_payload_path,
        output_image_path=source_image_path,
        image_size=str(getattr(render_config, "openai_image_size", "")),
    )
    source_payload = _read_json_mapping(source_payload_path)
    geometry_reference_build = _prepare_geometry_references(
        case=case,
        source_alignment=source_alignment,
        output_dir=variant_dir,
        enabled=instance_overlay_reference
        or single_call_layout_guard
        or (
            geometry_locked_materialization
            and bool(getattr(render_config, "include_geometry_references", False))
        ),
    )
    effective_render_prompt = ablated.render_prompt
    effective_edit_prompt = ablated.edit_prompt
    visual_preflight_result = None
    visual_preflight_prompt = ""
    if ablated.visual_preflight_mode != "disabled":
        preflight_source_image = source_image_path
        if (
            ablated.visual_preflight_profile == "edit_object_scope"
            and ablated.annotated_reference_image is not None
        ):
            preflight_source_image = ablated.annotated_reference_image
        elif (
            ablated.visual_preflight_profile
            in {"edit_scope", "edit_object_scope"}
            and approved_render_source is not None
        ):
            preflight_source_image = approved_render_source
        visual_preflight_result = run_visual_preflight(
            source_image=preflight_source_image,
            source_payload=source_payload,
            include_metadata=ablated.visual_preflight_mode == "snapshot_and_metadata",
            report_path=variant_dir / "visual_preflight_report.json",
            cache_dir=output_root / ".visual_preflight_cache",
            profile=ablated.visual_preflight_profile,
            edit_operations=ablated.edit_operations,
        )
        if not visual_preflight_result.source_valid:
            rejected_source = (
                "edit source render"
                if ablated.visual_preflight_profile
                in {"edit_scope", "edit_object_scope"}
                else "source snapshot"
            )
            raise ValueError(
                f"Visual preflight rejected the {rejected_source}: "
                f"{visual_preflight_result.reason or 'source_valid=false'}"
            )
        visual_preflight_prompt = compile_visual_preflight_prompt(
            visual_preflight_result,
            source_payload=source_payload
            if ablated.visual_preflight_mode == "snapshot_and_metadata"
            else None,
            profile=ablated.visual_preflight_profile,
            edit_operations=ablated.edit_operations,
        )
        effective_render_prompt = _append_prompt(
            effective_render_prompt,
            visual_preflight_prompt,
        )
        effective_edit_prompt = _append_prompt(
            effective_edit_prompt,
            visual_preflight_prompt,
        )

    snapshot_image_data_url = encode_image_to_data_url(source_image_path)
    locked_geometry_build = (
        _build_locked_geometry_atlas(
            source_image_path=source_image_path,
            source_payload=source_payload,
            output_path=variant_dir / _LOCKED_GEOMETRY_ATLAS_NAME,
            atlas_factory=SnapshotLockedGeometryAtlas,
            atlas_object_factory=SnapshotLockedGeometryAtlasObject,
        )
        if getattr(render_config, "include_locked_geometry_references", False)
        else LockedGeometryAtlasBuild(
            atlases=[],
            metadata={
                "enabled": False,
                "atlas_count": 0,
                "object_count": 0,
                "max_objects": _LOCKED_GEOMETRY_MAX_OBJECTS,
            },
        )
    )
    locked_geometry_atlases = locked_geometry_build.atlases
    depth_map_data_url = _optional_data_url(geometry_reference_build.depth_map_path)
    edge_map_data_url = _optional_data_url(geometry_reference_build.edge_map_path)
    segmentation_map_data_url = _optional_data_url(
        geometry_reference_build.segmentation_map_path
    )
    room_shell_guide_build = (
        _build_room_shell_guide_image(
            source_image_path=source_image_path,
            source_payload=source_payload,
            output_path=variant_dir / _ROOM_SHELL_GUIDE_NAME,
        )
        if ablated.visual_preflight_profile == "object_anchors"
        else RoomShellGuideBuild(
            image_path=None,
            metadata={"enabled": False, "image": None},
        )
    )
    edge_silhouette_guide_build = (
        _build_edge_silhouette_guide(
            source_image_path=source_image_path,
            source_payload=source_payload,
            semantic_output_path=variant_dir / _ROOM_SHELL_SEMANTIC_GUIDE_NAME,
            debug_output_path=variant_dir / _EDGE_SILHOUETTE_GUIDE_NAME,
        )
        if single_call_layout_guard
        else EdgeSilhouetteGuideBuild(
            guide_image_path=None,
            debug_image_path=None,
            mask_image_path=None,
            metadata={
                "enabled": False,
                "guide_image": None,
                "debug_image": None,
                "mask_image": None,
            },
        )
    )
    semantic_layout_lock_prompt = (
        _semantic_layout_lock_prompt(edge_silhouette_guide_build.metadata)
        if single_call_layout_guard
        else ""
    )
    if semantic_layout_lock_prompt:
        effective_render_prompt = _append_prompt(
            effective_render_prompt,
            semantic_layout_lock_prompt,
        )
        effective_edit_prompt = _append_prompt(
            effective_edit_prompt,
            semantic_layout_lock_prompt,
        )
    preset_selection = (
        case.preset_selection.to_mapping() if case.preset_selection is not None else {}
    )
    layout_reference_data_url = _optional_data_url(ablated.layout_reference_image)
    generated_annotated_reference_path = (
        geometry_reference_build.instance_outline_overlay_path
        or geometry_reference_build.instance_outline_path
        or edge_silhouette_guide_build.guide_image_path
        or room_shell_guide_build.image_path
    )
    annotated_reference_data_url = (
        encode_image_to_data_url(generated_annotated_reference_path)
        if generated_annotated_reference_path is not None
        else _optional_data_url(ablated.annotated_reference_image)
    )
    scene_reference_data_url = _optional_data_url(ablated.scene_reference_image)
    edit_operations = [
        SnapshotEditOperation(
            object_id=operation.object_id,
            object_name=operation.object_name,
            operation_type=operation.operation_type,
            prompt=operation.prompt,
            bbox_norm=_bbox_mapping(operation.bbox_norm),
            replacement_image_data_url=_optional_data_url(operation.replacement_image),
            target_color=operation.target_color,
        )
        for operation in ablated.edit_operations
    ]
    if approved_render_source is not None:
        render_reused = True
        provider_status: ProviderStatus = "reused_approved_render"
        render_image = approved_render_source
    else:
        render_reused = skip_existing and should_edit and _is_reusable_file(render_path)
        provider_status = "reused_render_partial" if render_reused else "generated"
        render_image = render_path

    if render_reused:
        render_result: Mapping[str, object] | None = None
        render_data_url = encode_image_to_data_url(render_image)
    pass_results: dict[str, JSONValue] | None = None
    render_pass1_image: Path | None = None
    repair_mask_build = LayoutRepairMaskBuild(
        image_path=None,
        metadata={"enabled": False, "image": None},
    )
    object_scope_edit_mask_build = ObjectScopeEditMaskBuild(
        image_path=None,
        metadata={"enabled": False, "image": None},
    )
    geometry_reference_kwargs = (
        {
            "depth_map_image_data_url": depth_map_data_url,
            "edge_map_image_data_url": edge_map_data_url,
            "segmentation_map_image_data_url": segmentation_map_data_url,
            "locked_geometry_atlases": locked_geometry_atlases,
        }
        if geometry_locked_materialization
        else {}
    )
    if render_reused:
        pass
    elif two_pass_geometry_repair:
        render_pass1_path = variant_dir / _RENDER_PASS1_NAME
        render_pass1_result = render_snapshot_image_from_path(
            source_payload_path,
            snapshot_image_data_url=snapshot_image_data_url,
            user_prompt=effective_render_prompt,
            annotated_reference_image_data_url=annotated_reference_data_url,
            edit_source_image_data_url=snapshot_image_data_url,
            layout_materialization_mode=True,
            render_mode="edit",
            preset_selection=preset_selection,
            config=render_config,
            trace_id=f"{case.id}:{variant.id}:geometry_materialize_pass1",
            **geometry_reference_kwargs,
        )
        render_pass1_data_url = _extract_result_data_url(render_pass1_result)
        render_pass1_image = write_data_url_image(
            render_pass1_data_url,
            render_pass1_path,
        )
        repair_mask_build = _build_layout_repair_mask(
            source_payload=source_payload,
            edge_silhouette_metadata=edge_silhouette_guide_build.metadata,
            output_path=variant_dir / _LAYOUT_REPAIR_MASK_NAME,
            source_image_path=source_image_path,
        )
        repair_prompt = _append_prompt(
            effective_render_prompt,
            _TWO_PASS_REPAIR_PROMPT,
        )
        render_result = render_snapshot_image_from_path(
            source_payload_path,
            snapshot_image_data_url=snapshot_image_data_url,
            user_prompt=repair_prompt,
            annotated_reference_image_data_url=annotated_reference_data_url,
            edit_source_image_data_url=render_pass1_data_url,
            layout_guard_mask_image_data_url=_optional_data_url(
                repair_mask_build.image_path
            ),
            layout_materialization_mode=True,
            scene_reference_mode="scene_reference_camera_transfer",
            render_mode="edit",
            preset_selection=preset_selection,
            config=render_config,
            trace_id=f"{case.id}:{variant.id}:geometry_repair_pass2",
            **geometry_reference_kwargs,
        )
        render_data_url = _extract_result_data_url(render_result)
        render_image = write_data_url_image(render_data_url, render_path)
        pass_results = {
            "pass1": _summarize_render_result(render_pass1_result),
            "pass2": _summarize_render_result(render_result),
        }
    elif single_call_layout_guard:
        render_result = render_snapshot_image_from_path(
            source_payload_path,
            snapshot_image_data_url=snapshot_image_data_url,
            user_prompt=effective_render_prompt,
            annotated_reference_image_data_url=annotated_reference_data_url,
            edit_source_image_data_url=snapshot_image_data_url,
            layout_materialization_mode=True,
            render_mode="edit",
            preset_selection=preset_selection,
            config=render_config,
            trace_id=f"{case.id}:{variant.id}:single_edit_materialize",
            **geometry_reference_kwargs,
        )
        render_data_url = _extract_result_data_url(render_result)
        render_image = write_data_url_image(render_data_url, render_path)
    else:
        render_result = render_snapshot_image_from_path(
            source_payload_path,
            snapshot_image_data_url=snapshot_image_data_url,
            user_prompt=effective_render_prompt,
            layout_reference_image_data_url=layout_reference_data_url,
            annotated_reference_image_data_url=annotated_reference_data_url,
            scene_reference_image_data_url=scene_reference_data_url,
            locked_geometry_atlases=locked_geometry_atlases,
            scene_reference_mode="target_layout_with_scene_reference"
            if scene_reference_data_url
            else "none",
            render_mode="generate",
            preset_selection=preset_selection,
            config=render_config,
            trace_id=f"{case.id}:{variant.id}:generate",
        )
        render_data_url = _extract_result_data_url(render_result)
        render_image = write_data_url_image(render_data_url, render_path)

    output_image = render_image
    edit_result: Mapping[str, object] | None = None
    if should_edit:
        if (
            ablated.visual_preflight_profile == "edit_object_scope"
            and _should_use_object_scope_edit_mask(edit_operations)
        ):
            object_scope_edit_mask_build = _build_object_scope_edit_mask(
                source_payload=source_payload,
                edit_operations=edit_operations,
                output_path=variant_dir / _OBJECT_SCOPE_EDIT_MASK_NAME,
                source_image_path=render_image,
            )
        elif ablated.visual_preflight_profile == "edit_object_scope":
            object_scope_edit_mask_build = ObjectScopeEditMaskBuild(
                image_path=None,
                metadata={
                    "enabled": False,
                    "image": None,
                    "reason": "disabled_for_stable_camera_operation",
                    "operation_types": _json_safe(
                        sorted(
                            {
                                operation.operation_type or "prompt"
                                for operation in edit_operations
                            }
                        )
                    ),
                    "mask_enabled_operation_types": sorted(
                        _OBJECT_SCOPE_MASK_OPERATION_TYPES
                    ),
                    "mask_semantics": (
                        "transparent_pixels_are_editable_opaque_pixels_are_preserved"
                    ),
                },
            )
        edit_render_config = (
            replace(render_config, disable_edit_mask=False)
            if object_scope_edit_mask_build.image_path is not None
            else render_config
        )
        edit_result = render_snapshot_image_from_path(
            source_payload_path,
            snapshot_image_data_url=snapshot_image_data_url,
            user_prompt=effective_edit_prompt,
            layout_reference_image_data_url=layout_reference_data_url,
            annotated_reference_image_data_url=annotated_reference_data_url,
            layout_guard_mask_image_data_url=_optional_data_url(
                object_scope_edit_mask_build.image_path
            ),
            edit_source_image_data_url=render_data_url,
            locked_geometry_atlases=locked_geometry_atlases,
            render_mode="edit",
            preset_selection={},
            edit_operations=edit_operations,
            config=edit_render_config,
            trace_id=f"{case.id}:{variant.id}:edit",
        )
        output_image = write_data_url_image(
            _extract_result_data_url(edit_result),
            edit_path,
        )

    metadata_path.write_text(
        json.dumps(
            {
                "case_id": case.id,
                "variant_id": variant.id,
                "mode": variant.mode,
                "provider_status": provider_status,
                "skip_existing": skip_existing,
                "renderer_config_overrides": _json_safe(
                    ablated.renderer_config_overrides
                ),
                "source_image": str(source_image_path),
                "source_snapshot_payload": str(source_payload_path),
                "source_alignment": _json_safe(source_alignment),
                "visual_preflight_mode": ablated.visual_preflight_mode,
                "visual_preflight_profile": ablated.visual_preflight_profile,
                "preset_selection": _json_safe(preset_selection),
                "visual_preflight_prompt": visual_preflight_prompt or None,
                "visual_preflight": visual_preflight_result.to_json()
                if visual_preflight_result is not None
                else None,
                "locked_geometry": _json_safe(locked_geometry_build.metadata),
                "geometry_reference_status": _json_safe(
                    geometry_reference_build.metadata.get("status")
                ),
                "geometry_references": _json_safe(geometry_reference_build.metadata),
                "room_shell_guide": _json_safe(room_shell_guide_build.metadata),
                "edge_silhouette_guide": _json_safe(
                    edge_silhouette_guide_build.metadata
                ),
                "repair_mask": _json_safe(repair_mask_build.metadata),
                "object_scope_edit_mask": _json_safe(
                    object_scope_edit_mask_build.metadata
                ),
                "semantic_layout_lock_prompt": semantic_layout_lock_prompt or None,
                "render_image": str(render_image),
                "render_pass1_image": str(render_pass1_image)
                if render_pass1_image is not None
                else None,
                "approved_render_source": str(approved_render_source)
                if approved_render_source is not None
                else None,
                "output_image": str(output_image),
                "pass_results": _json_safe(pass_results),
                "render_result": _summarize_render_result(render_result),
                "edit_result": _summarize_render_result(edit_result)
                if edit_result is not None
                else None,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return ProviderResult(
        case_id=case.id,
        variant_id=variant.id,
        output_image=output_image,
        metadata_path=metadata_path,
        render_image=render_image,
        source_image=source_image_path,
        source_snapshot_payload=source_payload_path,
        approved_render_source=approved_render_source,
        provider_status=provider_status,
    )


def _append_prompt(prompt: str | None, extra: str) -> str | None:
    if not extra:
        return prompt
    if not prompt:
        return extra
    return f"{prompt}\n\n{extra}"


def _prepare_aligned_snapshot_source(
    *,
    snapshot_payload_path: Path,
    snapshot_image_path: Path,
    output_payload_path: Path,
    output_image_path: Path,
    image_size: str,
) -> dict[str, JSONValue]:
    target_size = _image_size_dimensions(image_size)
    payload = _read_json_mapping(snapshot_payload_path)
    try:
        source_image = Image.open(snapshot_image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Unable to decode snapshot image: {snapshot_image_path}"
        ) from exc

    source_width, source_height = source_image.size
    if target_size is None:
        target_width, target_height = source_width, source_height
    else:
        target_width, target_height = target_size

    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Snapshot image has invalid size: {snapshot_image_path}")

    scale = min(target_width / source_width, target_height / source_height)
    drawn_width = max(1, round(source_width * scale))
    drawn_height = max(1, round(source_height * scale))
    offset_x = round((target_width - drawn_width) / 2)
    offset_y = round((target_height - drawn_height) / 2)
    resized_source = source_image.resize(
        (drawn_width, drawn_height),
        Image.Resampling.LANCZOS,
    )
    background = _corner_background_rgb(source_image)
    aligned_image = Image.new("RGB", (target_width, target_height), background)
    aligned_image.paste(resized_source, (offset_x, offset_y))
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_image.save(output_image_path)

    aligned_payload = _transform_snapshot_payload_screen_space(
        payload,
        target_width=target_width,
        target_height=target_height,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    output_payload_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload_path.write_text(
        json.dumps(aligned_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "method": "letterbox_to_openai_output_size",
        "source_size": [source_width, source_height],
        "target_size": [target_width, target_height],
        "drawn_size": [drawn_width, drawn_height],
        "offset_px": {"x": offset_x, "y": offset_y},
        "scale": round(scale, 6),
        "source_aspect": round(source_width / source_height, 6),
        "target_aspect": round(target_width / target_height, 6),
        "image_size": image_size,
    }


def _prepare_geometry_references(
    *,
    case: CaseConfig,
    source_alignment: Mapping[str, JSONValue],
    output_dir: Path,
    enabled: bool,
) -> GeometryReferenceBuild:
    if not enabled:
        return GeometryReferenceBuild(
            depth_map_path=None,
            edge_map_path=None,
            segmentation_map_path=None,
            instance_outline_path=None,
            instance_outline_overlay_path=None,
            metadata={
                "enabled": False,
                "status": "disabled",
                "depth_map": _geometry_reference_metadata(None, None),
                "edge_map": _geometry_reference_metadata(None, None),
                "segmentation_map": _geometry_reference_metadata(None, None),
                "instance_outline_guide": _geometry_reference_metadata(None, None),
                "instance_outline_overlay": _geometry_reference_metadata(None, None),
            },
        )
    if case.snapshot_image is None or case.snapshot_payload is None:
        raise ValueError(f"Case {case.id} requires snapshot geometry inputs.")

    prepared: dict[str, Path | None] = {}
    entries: dict[str, JSONValue] = {}
    for key, output_name in (
        ("depth_map", _DEPTH_MAP_NAME),
        ("edge_map", _EDGE_MAP_NAME),
        ("segmentation_map", _SEGMENTATION_MAP_NAME),
        ("instance_outline_guide", _INSTANCE_OUTLINE_GUIDE_NAME),
        ("instance_outline_overlay", _INSTANCE_OUTLINE_OVERLAY_NAME),
    ):
        source_path = _find_geometry_reference_path(
            snapshot_image_path=case.snapshot_image,
            snapshot_payload_path=case.snapshot_payload,
            key=key,
        )
        output_path = output_dir / output_name if source_path is not None else None
        if source_path is not None and output_path is not None:
            _align_geometry_reference_image(
                source_path=source_path,
                output_path=output_path,
                source_alignment=source_alignment,
                nearest_neighbor=key in {"segmentation_map", "instance_outline_guide"},
            )
            prepared[key] = output_path
        else:
            prepared[key] = None
        entries[key] = _geometry_reference_metadata(source_path, prepared[key])

    used_count = sum(1 for path in prepared.values() if path is not None)
    status = (
        "full_geometry"
        if used_count == 5
        else "partial_geometry"
        if used_count > 0
        else "missing_geometry"
    )
    return GeometryReferenceBuild(
        depth_map_path=prepared["depth_map"],
        edge_map_path=prepared["edge_map"],
        segmentation_map_path=prepared["segmentation_map"],
        instance_outline_path=prepared["instance_outline_guide"],
        instance_outline_overlay_path=prepared["instance_outline_overlay"],
        metadata={
            "enabled": True,
            "status": status,
            "used_count": used_count,
            "depth_map": entries["depth_map"],
            "edge_map": entries["edge_map"],
            "segmentation_map": entries["segmentation_map"],
            "instance_outline_guide": entries["instance_outline_guide"],
            "instance_outline_overlay": entries["instance_outline_overlay"],
        },
    )


def _geometry_reference_metadata(
    source_path: Path | None,
    output_path: Path | None,
) -> dict[str, JSONValue]:
    return {
        "source": str(source_path) if source_path is not None else None,
        "path": str(output_path) if output_path is not None else None,
        "used": output_path is not None,
        "missing": source_path is None,
    }


def _find_geometry_reference_path(
    *,
    snapshot_image_path: Path,
    snapshot_payload_path: Path,
    key: str,
) -> Path | None:
    camera_payload_stem = snapshot_payload_path.name.removesuffix(
        ".snapshot_payload.json"
    )
    candidates = [
        snapshot_image_path.with_name(f"{snapshot_image_path.stem}.{key}.png"),
        snapshot_image_path.with_name(f"{snapshot_image_path.stem}_{key}.png"),
        snapshot_image_path.with_name(f"{key}.png"),
        snapshot_payload_path.with_name(f"{camera_payload_stem}.{key}.png"),
        snapshot_payload_path.with_name(f"{camera_payload_stem}_{key}.png"),
        snapshot_payload_path.with_name(f"{key}.png"),
        snapshot_image_path.parent.parent / f"{key}.png",
    ]
    for candidate in candidates:
        if _is_reusable_file(candidate):
            return candidate
    return None


def _align_geometry_reference_image(
    *,
    source_path: Path,
    output_path: Path,
    source_alignment: Mapping[str, JSONValue],
    nearest_neighbor: bool,
) -> None:
    try:
        with Image.open(source_path) as image:
            source_image = image.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Unable to decode geometry reference image: {source_path}"
        ) from exc

    target_size = _alignment_size(source_alignment, "target_size") or source_image.size
    alignment_source_size = _alignment_size(source_alignment, "source_size")
    drawn_size = _alignment_size(source_alignment, "drawn_size")
    offset = _alignment_offset(source_alignment)
    use_alignment = (
        alignment_source_size == source_image.size
        and drawn_size is not None
        and offset is not None
    )
    if use_alignment:
        drawn_width, drawn_height = drawn_size
        offset_x, offset_y = offset
    else:
        target_width, target_height = target_size
        source_width, source_height = source_image.size
        scale = min(target_width / source_width, target_height / source_height)
        drawn_width = max(1, round(source_width * scale))
        drawn_height = max(1, round(source_height * scale))
        offset_x = round((target_width - drawn_width) / 2)
        offset_y = round((target_height - drawn_height) / 2)

    resampling = (
        Image.Resampling.NEAREST if nearest_neighbor else Image.Resampling.LANCZOS
    )
    resized = source_image.resize((drawn_width, drawn_height), resampling)
    aligned = Image.new("RGB", target_size, "black")
    aligned.paste(resized, (offset_x, offset_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.save(output_path)


def _alignment_size(
    source_alignment: Mapping[str, JSONValue],
    key: str,
) -> tuple[int, int] | None:
    value = source_alignment.get(key)
    if not isinstance(value, list) or len(value) != 2:
        return None
    width = _integer_value(value[0], 0)
    height = _integer_value(value[1], 0)
    if width <= 0 or height <= 0:
        return None
    return width, height


def _alignment_offset(
    source_alignment: Mapping[str, JSONValue],
) -> tuple[int, int] | None:
    offset = _mapping(source_alignment.get("offset_px"))
    x = _integer_value(offset.get("x"), -1)
    y = _integer_value(offset.get("y"), -1)
    if x < 0 or y < 0:
        return None
    return x, y


def _integer_value(value: object, fallback: int) -> int:
    number = _optional_number(value)
    if number is None:
        return fallback
    return int(number)


@dataclass(frozen=True)
class LockedGeometryCropCandidate:
    object_id: str
    object_name: str | None
    screen_bbox_px: dict[str, float]
    crop_box_px: tuple[int, int, int, int]


def _build_locked_geometry_atlas(
    *,
    source_image_path: Path,
    source_payload: Mapping[str, object],
    output_path: Path,
    atlas_factory: Callable[..., AtlasT],
    atlas_object_factory: Callable[..., AtlasObjectT],
) -> LockedGeometryAtlasBuild[AtlasT]:
    try:
        with Image.open(source_image_path) as image:
            source_image = image.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Unable to decode locked geometry source image: {source_image_path}"
        ) from exc

    candidates = _visible_object_crop_candidates(
        source_payload,
        image_size=source_image.size,
    )[:_LOCKED_GEOMETRY_MAX_OBJECTS]
    if not candidates:
        return LockedGeometryAtlasBuild(
            atlases=[],
            metadata={
                "enabled": True,
                "atlas_count": 0,
                "object_count": 0,
                "atlas_image": None,
                "max_objects": _LOCKED_GEOMETRY_MAX_OBJECTS,
                "objects": [],
            },
        )

    columns = min(4, len(candidates))
    rows = math.ceil(len(candidates) / columns)
    atlas_width = columns * _LOCKED_GEOMETRY_CELL_SIZE_PX
    atlas_height = rows * _LOCKED_GEOMETRY_CELL_SIZE_PX
    atlas = Image.new("RGB", (atlas_width, atlas_height), "white")
    draw = ImageDraw.Draw(atlas)
    atlas_objects: list[AtlasObjectT] = []
    metadata_objects: list[dict[str, JSONValue]] = []

    for index, candidate in enumerate(candidates, start=1):
        cell_index = index - 1
        column = cell_index % columns
        row = cell_index // columns
        cell_left = column * _LOCKED_GEOMETRY_CELL_SIZE_PX
        cell_top = row * _LOCKED_GEOMETRY_CELL_SIZE_PX
        cell_bbox_norm = _normalized_cell_bbox(
            cell_left=cell_left,
            cell_top=cell_top,
            atlas_width=atlas_width,
            atlas_height=atlas_height,
        )
        cell_label = f"{index}. {candidate.object_id}"
        draw.rectangle(
            [
                cell_left,
                cell_top,
                cell_left + _LOCKED_GEOMETRY_CELL_SIZE_PX - 1,
                cell_top + _LOCKED_GEOMETRY_CELL_SIZE_PX - 1,
            ],
            outline=(180, 180, 180),
            width=1,
        )
        draw.text(
            (cell_left + 4, cell_top + 2),
            cell_label[:40],
            fill=(20, 20, 20),
        )

        crop = source_image.crop(candidate.crop_box_px)
        crop.thumbnail(
            (
                _LOCKED_GEOMETRY_CELL_SIZE_PX - 8,
                _LOCKED_GEOMETRY_CELL_SIZE_PX
                - _LOCKED_GEOMETRY_CELL_LABEL_HEIGHT_PX
                - 8,
            ),
            Image.Resampling.LANCZOS,
        )
        paste_x = cell_left + (_LOCKED_GEOMETRY_CELL_SIZE_PX - crop.width) // 2
        paste_y = (
            cell_top
            + _LOCKED_GEOMETRY_CELL_LABEL_HEIGHT_PX
            + (
                _LOCKED_GEOMETRY_CELL_SIZE_PX
                - _LOCKED_GEOMETRY_CELL_LABEL_HEIGHT_PX
                - crop.height
            )
            // 2
        )
        atlas.paste(crop, (paste_x, paste_y))

        atlas_object = atlas_object_factory(
            object_id=candidate.object_id,
            object_name=candidate.object_name,
            screen_bbox_px=candidate.screen_bbox_px,
            atlas_index=1,
            cell_index=index,
            cell_label=cell_label,
            cell_bbox_norm=cell_bbox_norm,
        )
        atlas_objects.append(atlas_object)
        metadata_objects.append(
            {
                "object_id": candidate.object_id,
                "object_name": candidate.object_name,
                "screen_bbox_px": candidate.screen_bbox_px,
                "crop_box_px": list(candidate.crop_box_px),
                "atlas_index": 1,
                "cell_index": index,
                "cell_label": cell_label,
                "cell_bbox_norm": cell_bbox_norm,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output_path)
    atlas_model = atlas_factory(
        atlas_index=1,
        image_data_url=encode_image_to_data_url(output_path),
        source=str(output_path),
        objects=atlas_objects,
    )
    return LockedGeometryAtlasBuild(
        atlases=[atlas_model],
        metadata={
            "enabled": True,
            "atlas_count": 1,
            "object_count": len(atlas_objects),
            "atlas_image": str(output_path),
            "max_objects": _LOCKED_GEOMETRY_MAX_OBJECTS,
            "objects": metadata_objects,
        },
    )


def _build_room_shell_guide_image(
    *,
    source_image_path: Path,
    source_payload: Mapping[str, object],
    output_path: Path,
) -> RoomShellGuideBuild:
    try:
        with Image.open(source_image_path) as image:
            source_image = image.convert("RGBA")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Unable to decode room shell guide source image: {source_image_path}"
        ) from exc

    image_width, image_height = source_image.size
    overlay = Image.new("RGBA", source_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke_width = max(3, round(min(image_width, image_height) / 260))
    crop_band = max(10, round(min(image_width, image_height) * 0.024))

    draw.rectangle(
        [0, 0, image_width - 1, image_height - 1],
        outline=(255, 132, 0, 230),
        width=stroke_width,
    )
    for box in (
        [0, 0, crop_band, image_height],
        [image_width - crop_band, 0, image_width, image_height],
    ):
        draw.rectangle(
            box,
            fill=(255, 132, 0, 28),
            outline=(255, 132, 0, 170),
            width=1,
        )

    opening_count = 0
    for label, bbox in _room_opening_guide_boxes(source_payload):
        opening_count += 1
        _draw_guide_bbox(
            draw,
            bbox,
            label=label,
            color=(0, 112, 255, 230),
            width=stroke_width,
        )

    object_candidates = sorted(
        _visible_object_crop_candidates(source_payload, image_size=source_image.size),
        key=lambda candidate: _bbox_area(candidate.screen_bbox_px),
        reverse=True,
    )[:_ROOM_SHELL_GUIDE_MAX_OBJECTS]
    for candidate in object_candidates:
        _draw_guide_bbox(
            draw,
            candidate.screen_bbox_px,
            label=candidate.object_id,
            color=(0, 190, 105, 220),
            width=max(2, stroke_width - 1),
        )

    room = _mapping(source_payload.get("room"))
    polygon = room.get("polygonMm")
    wall_count = len(polygon) if isinstance(polygon, list) and polygon else None
    guide_lines = [
        "ROOM SHELL GUIDE",
        "Orange: preserve crop edges and visible wall shell.",
        "Green: preserve object bbox anchors, scale, crop, support.",
        "Blue: preserve opening bbox; do not invent openings.",
        "Shadows/gradients are lighting, not geometry.",
    ]
    if wall_count is not None:
        guide_lines.append(f"Metadata room footprint: {wall_count} wall segments.")
    _draw_guide_panel(draw, guide_lines)

    guide_image = Image.alpha_composite(source_image, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    guide_image.save(output_path)
    return RoomShellGuideBuild(
        image_path=output_path,
        metadata={
            "enabled": True,
            "image": str(output_path),
            "wall_segment_count": wall_count,
            "opening_bbox_count": opening_count,
            "object_bbox_count": len(object_candidates),
        },
    )


def _build_edge_silhouette_guide(
    *,
    source_image_path: Path,
    source_payload: Mapping[str, object],
    semantic_output_path: Path,
    debug_output_path: Path,
) -> EdgeSilhouetteGuideBuild:
    try:
        with Image.open(source_image_path) as image:
            source_image = image.convert("RGBA")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Unable to decode edge silhouette source image: {source_image_path}"
        ) from exc

    image_width, image_height = source_image.size
    clean_objects, rejected_objects = _layout_guard_object_candidates(
        source_payload,
        image_size=source_image.size,
    )
    opening_boxes = _room_opening_guide_boxes(source_payload)
    debug_guide = _edge_silhouette_guide_image(
        source_image=source_image,
        clean_objects=clean_objects,
        rejected_objects=rejected_objects,
        opening_boxes=opening_boxes,
    )
    semantic_guide, semantic_metadata = _room_shell_semantic_guide_image(
        source_image=source_image,
        source_payload=source_payload,
        clean_objects=clean_objects,
        rejected_objects=rejected_objects,
        opening_boxes=opening_boxes,
    )

    semantic_output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_guide.convert("RGB").save(semantic_output_path)
    debug_guide.convert("RGB").save(debug_output_path)
    return EdgeSilhouetteGuideBuild(
        guide_image_path=semantic_output_path,
        debug_image_path=debug_output_path,
        mask_image_path=None,
        metadata={
            "enabled": True,
            "guide_image": str(semantic_output_path),
            "guide_type": "room_shell_semantic",
            "debug_image": str(debug_output_path),
            "debug_guide_type": "edge_silhouette",
            "mask_image": None,
            "mask_used_for_openai": False,
            "mask_strategy": "disabled_structural_guide_only",
            "image_size": [image_width, image_height],
            "clean_object_count": len(clean_objects),
            "rejected_object_count": len(rejected_objects),
            "opening_bbox_count": len(opening_boxes),
            "clean_objects": clean_objects,
            "rejected_objects": rejected_objects,
            **semantic_metadata,
        },
    )


def _room_shell_semantic_guide_image(
    *,
    source_image: Image.Image,
    source_payload: Mapping[str, object],
    clean_objects: list[dict[str, JSONValue]],
    rejected_objects: list[dict[str, JSONValue]],
    opening_boxes: list[tuple[str, dict[str, float]]],
) -> tuple[Image.Image, dict[str, JSONValue]]:
    guide = source_image.convert("RGBA")
    image_size = guide.size
    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    image_width, image_height = image_size
    stroke_width = max(3, round(min(image_width, image_height) / 300))

    surfaces, projection_metadata = _project_room_shell_surfaces(
        source_payload,
        image_size=image_size,
    )
    surface_colors: Mapping[str, tuple[int, int, int, int]] = {
        "floor": (245, 158, 11, 70),
        "ceiling": (168, 85, 247, 58),
        "wall": (6, 182, 212, 62),
    }
    surface_outline_colors: Mapping[str, tuple[int, int, int, int]] = {
        "floor": (217, 119, 6, 235),
        "ceiling": (126, 34, 206, 230),
        "wall": (8, 145, 178, 235),
    }
    wall_palette = (
        (6, 182, 212, 62),
        (59, 130, 246, 58),
        (16, 185, 129, 58),
        (236, 72, 153, 54),
        (34, 197, 94, 54),
        (99, 102, 241, 54),
    )
    wall_outline_palette = (
        (8, 145, 178, 235),
        (37, 99, 235, 230),
        (5, 150, 105, 230),
        (219, 39, 119, 225),
        (22, 163, 74, 225),
        (79, 70, 229, 225),
    )
    for index, surface in enumerate(
        sorted(surfaces, key=lambda item: item.average_depth_m, reverse=True)
    ):
        if len(surface.polygon_px) < 3:
            continue
        if surface.surface_type == "wall":
            fill = wall_palette[index % len(wall_palette)]
            outline = wall_outline_palette[index % len(wall_outline_palette)]
        else:
            fill = surface_colors.get(surface.surface_type, surface_colors["wall"])
            outline = surface_outline_colors.get(
                surface.surface_type,
                surface_outline_colors["wall"],
            )
        polygon = [(round(x), round(y)) for x, y in surface.polygon_px]
        draw.polygon(polygon, fill=fill)
        draw.line([*polygon, polygon[0]], fill=outline, width=stroke_width)

    crop_band = max(8, round(min(image_width, image_height) * 0.018))
    draw.rectangle(
        [0, 0, image_width - 1, image_height - 1],
        outline=(255, 132, 0, 235),
        width=max(2, stroke_width - 1),
    )
    for box in (
        [0, 0, crop_band, image_height],
        [image_width - crop_band, 0, image_width, image_height],
        [0, 0, image_width, crop_band],
        [0, image_height - crop_band, image_width, image_height],
    ):
        draw.rectangle(box, fill=(255, 132, 0, 18))

    for _, bbox in opening_boxes:
        _draw_semantic_bbox(
            draw,
            bbox,
            fill=(0, 112, 255, 46),
            outline=(0, 112, 255, 245),
            width=stroke_width + 1,
        )
    for item in clean_objects[:_EDGE_SILHOUETTE_GUIDE_MAX_OBJECTS]:
        bbox = _json_float_mapping(item.get("screen_bbox_px"))
        if bbox:
            _draw_semantic_bbox(
                draw,
                bbox,
                fill=None,
                outline=(0, 190, 105, 235),
                width=max(2, stroke_width - 1),
            )
    for item in rejected_objects:
        bbox = _json_float_mapping(item.get("screen_bbox_px"))
        if bbox and bool(item.get("draw_rejected_bbox")):
            _draw_semantic_bbox(
                draw,
                bbox,
                fill=None,
                outline=(220, 38, 38, 220),
                width=max(2, stroke_width - 1),
            )

    guide = Image.alpha_composite(guide, overlay)
    metadata_surfaces: list[JSONValue] = []
    for surface in surfaces:
        metadata_surfaces.append(
            {
                "id": surface.surface_id,
                "type": surface.surface_type,
                "point_count": len(surface.polygon_px),
                "average_depth_m": round(surface.average_depth_m, 4),
                "touches_crop_edges": list(surface.touches_crop_edges),
                "near_crop_edges": list(surface.near_crop_edges),
                "bbox_norm": surface.bbox_norm,
            }
        )
    room_shell_locks = _semantic_room_shell_lock_lines(
        surfaces=surfaces,
        opening_boxes=opening_boxes,
        rejected_objects=rejected_objects,
    )
    return guide, {
        "semantic_projection_status": projection_metadata["status"],
        "semantic_projection_reason": projection_metadata.get("reason"),
        "semantic_surface_count": len(surfaces),
        "semantic_surfaces": metadata_surfaces,
        "semantic_room_shell_locks": room_shell_locks,
        "semantic_color_key": {
            "wall_surfaces": "cyan/blue/green/pink translucent filled planes",
            "floor": "amber translucent filled plane",
            "ceiling": "purple translucent filled plane",
            "openings": "solid blue boxes",
            "clean_object_anchors": "green outlines",
            "rejected_noisy_anchors": "red outlines when drawable",
            "crop_guard": "orange border and crop bands",
        },
    }


def _semantic_room_shell_lock_lines(
    *,
    surfaces: list[ProjectedRoomShellSurface],
    opening_boxes: list[tuple[str, dict[str, float]]],
    rejected_objects: list[dict[str, JSONValue]],
) -> list[str]:
    lines: list[str] = []
    wall_surfaces = [surface for surface in surfaces if surface.surface_type == "wall"]
    near_crop_walls = [
        surface
        for surface in wall_surfaces
        if surface.touches_crop_edges or surface.near_crop_edges
    ]
    for surface in near_crop_walls[:6]:
        edge_label = _edge_label(surface.touches_crop_edges or surface.near_crop_edges)
        proximity = (
            "touching the image crop"
            if surface.touches_crop_edges
            else "near the image crop"
        )
        bbox = _bbox_norm_label(surface.bbox_norm)
        lines.append(
            "Room-shell lock: preserve "
            f"{surface.surface_id} as a real wall plane {proximity} on the "
            f"{edge_label}; keep its wall seam, crop silhouette, perspective, "
            f"and occlusion footprint unchanged ({bbox}). Do not flatten it away, "
            "turn it into empty space, complete it as a nicer rectangular room, "
            "or convert it into a window, door, curtain, lamp, shelf, or decor."
        )
    if wall_surfaces:
        lines.append(
            "Shadow-vs-geometry lock: dark vertical lines, gradients, and cast "
            "shadows that lie inside colored wall planes are wall seams/shadows, "
            "not furniture. Never convert a wall shadow or corner edge into a "
            "floor lamp, pole, shelf, curtain, window, or door."
        )
    if opening_boxes:
        lines.append(
            "Opening lock: preserve exactly the existing blue opening box(es) "
            "from the semantic guide; keep their count, wall association, apparent "
            "size, and silhouette. Do not add new windows, doors, exterior views, "
            "curtains, or framed openings outside those boxes."
        )
    else:
        lines.append(
            "Opening lock: the source view has no trusted visible opening boxes "
            "in the semantic guide. Do not invent windows, doors, exterior views, "
            "curtains, or new wall cutouts."
        )
    for item in rejected_objects[:6]:
        lock_line = _rejected_object_lock_line(item)
        if lock_line:
            lines.append(lock_line)
    return lines


def _semantic_layout_lock_prompt(metadata: Mapping[str, JSONValue]) -> str:
    raw_lines = metadata.get("semantic_room_shell_locks")
    if not isinstance(raw_lines, list):
        return ""
    lines = [
        str(item).strip()
        for item in raw_lines
        if isinstance(item, str) and item.strip()
    ]
    if not lines:
        return ""
    bullet_lines = "\n".join(f"- {line}" for line in lines[:12])
    return (
        "Semantic layout locks from the structural guide:\n"
        f"{bullet_lines}\n"
        "These locks are more important than photorealistic room cleanup. Preserve "
        "awkward crop walls, partial foreground occluders, wall shadows, and exact "
        "opening/object footprints even when they look unusual."
    )


def _rejected_object_lock_line(item: Mapping[str, JSONValue]) -> str:
    object_id = _text(item.get("object_id")) or "unknown_object"
    object_type = _text(item.get("object_type")) or "object"
    reason = _text(item.get("reason")) or "noisy_bbox"
    bbox = _mapping(item.get("bbox_norm"))
    edge_touch_count = _optional_number(item.get("edge_touch_count")) or 0.0
    area_ratio = _optional_number(item.get("area_ratio")) or 0.0
    bbox_label = _bbox_norm_label(bbox)
    if edge_touch_count >= 1:
        return (
            "Rejected-anchor lock: "
            f"{object_id} ({object_type}) has an unreliable edge/crop bbox "
            f"({reason}, area={area_ratio:.3f}, {bbox_label}), but the visible "
            "source pixels still represent a real cropped/foreground object or "
            "occluder. Preserve its visible silhouette and occlusion from Image 1; "
            "do not complete, remove, replace, expand, or move it based on the "
            "noisy bbox."
        )
    return (
        "Rejected-anchor lock: "
        f"{object_id} ({object_type}) has an unreliable bbox ({reason}, "
        f"area={area_ratio:.3f}, {bbox_label}). Treat the bbox as a warning only; "
        "preserve the source-visible object pixels and do not use the noisy bbox "
        "to add, move, resize, or replace objects."
    )


def _edge_label(edges: tuple[str, ...]) -> str:
    if not edges:
        return "image frame"
    if len(edges) == 1:
        return f"{edges[0]} edge"
    return ", ".join(f"{edge} edge" for edge in edges[:-1]) + f", and {edges[-1]} edge"


def _bbox_norm_label(bbox: Mapping[str, object]) -> str:
    x = _optional_number(bbox.get("x"))
    y = _optional_number(bbox.get("y"))
    width = _optional_number(bbox.get("width"))
    height = _optional_number(bbox.get("height"))
    if x is None or y is None or width is None or height is None:
        return "bbox unavailable"
    return f"bbox_norm x={x:.3f}, y={y:.3f}, width={width:.3f}, height={height:.3f}"


def _project_room_shell_surfaces(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> tuple[list[ProjectedRoomShellSurface], dict[str, JSONValue]]:
    room = _mapping(source_payload.get("room"))
    polygon = _room_polygon_points(room.get("polygonMm"))
    if len(polygon) < 3:
        return [], {"status": "skipped", "reason": "room_polygon_missing"}
    camera = _projection_camera(source_payload)
    if camera is None:
        return [], {"status": "skipped", "reason": "camera_missing_or_invalid"}

    center_x, center_y = _room_center_mm(room, polygon)
    ceiling_height_mm = _room_ceiling_height_mm(source_payload)
    surfaces: list[ProjectedRoomShellSurface] = []
    floor_points = [
        _room_point_to_world(point, height_mm=0.0, center_x=center_x, center_y=center_y)
        for point in polygon
    ]
    ceiling_points = [
        _room_point_to_world(
            point,
            height_mm=ceiling_height_mm,
            center_x=center_x,
            center_y=center_y,
        )
        for point in polygon
    ]
    floor_surface = _project_surface(
        "floor",
        "floor",
        floor_points,
        camera=camera,
        image_size=image_size,
    )
    ceiling_surface = _project_surface(
        "ceiling",
        "ceiling",
        ceiling_points,
        camera=camera,
        image_size=image_size,
    )
    if floor_surface is not None:
        surfaces.append(floor_surface)
    if ceiling_surface is not None:
        surfaces.append(ceiling_surface)

    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        wall_points = [
            _room_point_to_world(
                start,
                height_mm=0.0,
                center_x=center_x,
                center_y=center_y,
            ),
            _room_point_to_world(
                end,
                height_mm=0.0,
                center_x=center_x,
                center_y=center_y,
            ),
            _room_point_to_world(
                end,
                height_mm=ceiling_height_mm,
                center_x=center_x,
                center_y=center_y,
            ),
            _room_point_to_world(
                start,
                height_mm=ceiling_height_mm,
                center_x=center_x,
                center_y=center_y,
            ),
        ]
        surface = _project_surface(
            f"wall_{index + 1}",
            "wall",
            wall_points,
            camera=camera,
            image_size=image_size,
        )
        if surface is not None:
            surfaces.append(surface)
    return surfaces, {
        "status": "ok" if surfaces else "skipped",
        "reason": None if surfaces else "no_projected_surfaces",
        "ceiling_height_mm": round(ceiling_height_mm, 2),
        "room_center_mm": {"x": round(center_x, 2), "y": round(center_y, 2)},
    }


def _project_surface(
    surface_id: str,
    surface_type: str,
    points: list[Vec3],
    *,
    camera: Mapping[str, object],
    image_size: tuple[int, int],
) -> ProjectedRoomShellSurface | None:
    projected: list[Point2] = []
    depths: list[float] = []
    for point in points:
        projected_point = _project_world_point(
            point, camera=camera, image_size=image_size
        )
        if projected_point is None:
            return None
        screen_point, depth = projected_point
        projected.append(screen_point)
        depths.append(depth)
    if len(projected) < 3:
        return None
    touches = _projected_polygon_crop_edges(
        projected,
        image_size=image_size,
        margin_ratio=_ROOM_SHELL_TOUCH_CROP_MARGIN_RATIO,
    )
    near_crop_edges = _projected_polygon_crop_edges(
        projected,
        image_size=image_size,
        margin_ratio=_ROOM_SHELL_NEAR_CROP_MARGIN_RATIO,
    )
    return ProjectedRoomShellSurface(
        surface_id=surface_id,
        surface_type=surface_type,
        polygon_px=projected,
        average_depth_m=sum(depths) / max(1, len(depths)),
        touches_crop_edges=touches,
        near_crop_edges=near_crop_edges,
        bbox_norm=_projected_polygon_bbox_norm(projected, image_size=image_size),
    )


def _project_world_point(
    point: Vec3,
    *,
    camera: Mapping[str, object],
    image_size: tuple[int, int],
) -> tuple[Point2, float] | None:
    position = cast(Vec3, camera["position"])
    right = cast(Vec3, camera["right"])
    up = cast(Vec3, camera["up"])
    forward = cast(Vec3, camera["forward"])
    tan_half_fov = float(camera["tan_half_fov"])
    aspect = float(camera["aspect"])
    image_width, image_height = image_size
    relative = _vec_sub(point, position)
    depth = _vec_dot(relative, forward)
    if depth <= _ROOM_SHELL_PROJECTION_NEAR_M:
        return None
    camera_x = _vec_dot(relative, right)
    camera_y = _vec_dot(relative, up)
    ndc_x = camera_x / max(0.0001, depth * tan_half_fov * aspect)
    ndc_y = camera_y / max(0.0001, depth * tan_half_fov)
    screen_x = (ndc_x + 1.0) * 0.5 * image_width
    screen_y = (1.0 - ndc_y) * 0.5 * image_height
    screen_x = max(
        -_ROOM_SHELL_PROJECTED_POINT_LIMIT,
        min(_ROOM_SHELL_PROJECTED_POINT_LIMIT, screen_x),
    )
    screen_y = max(
        -_ROOM_SHELL_PROJECTED_POINT_LIMIT,
        min(_ROOM_SHELL_PROJECTED_POINT_LIMIT, screen_y),
    )
    return (screen_x, screen_y), depth


def _projection_camera(
    source_payload: Mapping[str, object],
) -> dict[str, object] | None:
    camera = _mapping(source_payload.get("camera"))
    position = _vec3_from_mapping(camera.get("positionM"))
    target = _vec3_from_mapping(camera.get("targetM"))
    fov_deg = _optional_number(camera.get("fovDeg"))
    aspect = _optional_number(camera.get("aspect"))
    if position is None or target is None or fov_deg is None or aspect is None:
        return None
    forward = _vec_normalize(_vec_sub(target, position))
    if forward is None:
        return None
    world_up: Vec3 = (0.0, 1.0, 0.0)
    right = _vec_normalize(_vec_cross(forward, world_up))
    if right is None:
        return None
    up = _vec_normalize(_vec_cross(right, forward))
    if up is None:
        return None
    return {
        "position": position,
        "forward": forward,
        "right": right,
        "up": up,
        "tan_half_fov": math.tan(math.radians(fov_deg) / 2.0),
        "aspect": max(0.0001, aspect),
    }


def _room_polygon_points(value: object) -> list[Point2]:
    points: list[Point2] = []
    for item in _sequence(value):
        payload = _mapping(item)
        x = _optional_number(payload.get("x"))
        y = _optional_number(payload.get("y"))
        if x is None or y is None:
            continue
        points.append((x, y))
    return points


def _room_center_mm(room: Mapping[str, object], polygon: list[Point2]) -> Point2:
    bounds = _mapping(room.get("boundsMm"))
    min_x = _optional_number(bounds.get("minX"))
    max_x = _optional_number(bounds.get("maxX"))
    min_y = _optional_number(bounds.get("minY"))
    max_y = _optional_number(bounds.get("maxY"))
    if (
        min_x is not None
        and max_x is not None
        and min_y is not None
        and max_y is not None
    ):
        return (min_x + max_x) * 0.5, (min_y + max_y) * 0.5
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5


def _room_ceiling_height_mm(source_payload: Mapping[str, object]) -> float:
    room = _mapping(source_payload.get("room"))
    for key in ("ceilingHeightMm", "ceiling_height_mm", "heightMm", "height_mm"):
        value = _optional_number(room.get(key))
        if value is not None and value > 0:
            return value
    return _ROOM_SHELL_DEFAULT_CEILING_HEIGHT_MM


def _room_point_to_world(
    point: Point2,
    *,
    height_mm: float,
    center_x: float,
    center_y: float,
) -> Vec3:
    return (
        (point[0] - center_x) / 1000.0,
        height_mm / 1000.0,
        (center_y - point[1]) / 1000.0,
    )


def _projected_polygon_crop_edges(
    polygon: list[Point2],
    *,
    image_size: tuple[int, int],
    margin_ratio: float,
) -> tuple[str, ...]:
    image_width, image_height = image_size
    margin = max(8.0, min(image_width, image_height) * margin_ratio)
    edges: list[str] = []
    if any(x <= margin for x, _ in polygon):
        edges.append("left")
    if any(x >= image_width - margin for x, _ in polygon):
        edges.append("right")
    if any(y <= margin for _, y in polygon):
        edges.append("top")
    if any(y >= image_height - margin for _, y in polygon):
        edges.append("bottom")
    return tuple(edges)


def _projected_polygon_bbox_norm(
    polygon: list[Point2],
    *,
    image_size: tuple[int, int],
) -> dict[str, float]:
    image_width, image_height = image_size
    min_x = max(0.0, min((x for x, _ in polygon), default=0.0))
    min_y = max(0.0, min((y for _, y in polygon), default=0.0))
    max_x = min(float(image_width), max((x for x, _ in polygon), default=0.0))
    max_y = min(float(image_height), max((y for _, y in polygon), default=0.0))
    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)
    return {
        "x": round(min_x / max(1, image_width), 6),
        "y": round(min_y / max(1, image_height), 6),
        "width": round(width / max(1, image_width), 6),
        "height": round(height / max(1, image_height), 6),
    }


def _draw_semantic_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Mapping[str, float],
    *,
    fill: tuple[int, int, int, int] | None,
    outline: tuple[int, int, int, int],
    width: int,
) -> None:
    box = [
        round(bbox["minX"]),
        round(bbox["minY"]),
        round(bbox["maxX"]),
        round(bbox["maxY"]),
    ]
    if fill is not None:
        draw.rectangle(box, fill=fill)
    draw.rectangle(box, outline=outline, width=width)


def _vec3_from_mapping(value: object) -> Vec3 | None:
    payload = _mapping(value)
    x = _optional_number(payload.get("x"))
    y = _optional_number(payload.get("y"))
    z = _optional_number(payload.get("z"))
    if x is None or y is None or z is None:
        return None
    return x, y, z


def _vec_sub(left: Vec3, right: Vec3) -> Vec3:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _vec_dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vec_cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _vec_normalize(vector: Vec3) -> Vec3 | None:
    length = math.sqrt(_vec_dot(vector, vector))
    if length <= 0.000001:
        return None
    return vector[0] / length, vector[1] / length, vector[2] / length


def _edge_silhouette_guide_image(
    *,
    source_image: Image.Image,
    clean_objects: list[dict[str, JSONValue]],
    rejected_objects: list[dict[str, JSONValue]],
    opening_boxes: list[tuple[str, dict[str, float]]],
) -> Image.Image:
    guide = source_image.convert("RGBA")
    overlay = Image.new("RGBA", guide.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    image_width, image_height = guide.size
    stroke_width = max(2, round(min(image_width, image_height) / 360))

    edge_mask = ImageOps.autocontrast(
        guide.convert("L").filter(ImageFilter.FIND_EDGES)
    ).point(lambda value: 150 if value >= 38 else 0)
    edge_overlay = Image.new("RGBA", guide.size, (255, 88, 0, 0))
    edge_overlay.putalpha(edge_mask)
    overlay = Image.alpha_composite(overlay, edge_overlay)
    draw = ImageDraw.Draw(overlay)

    border = max(8, round(min(image_width, image_height) * _LAYOUT_GUARD_BORDER_RATIO))
    draw.rectangle(
        [0, 0, image_width - 1, image_height - 1],
        outline=(255, 132, 0, 240),
        width=stroke_width + 1,
    )
    for box in (
        [0, 0, border, image_height],
        [image_width - border, 0, image_width, image_height],
        [0, 0, image_width, border],
        [0, image_height - border, image_width, image_height],
    ):
        draw.rectangle(box, fill=(255, 132, 0, 32))

    for label, bbox in opening_boxes:
        _draw_guide_bbox(
            draw,
            bbox,
            label=label,
            color=(0, 112, 255, 235),
            width=stroke_width + 1,
        )
    for item in clean_objects[:_EDGE_SILHOUETTE_GUIDE_MAX_OBJECTS]:
        bbox = _json_float_mapping(item.get("screen_bbox_px"))
        if bbox:
            _draw_guide_bbox(
                draw,
                bbox,
                label=str(item.get("object_id") or "object"),
                color=(0, 190, 105, 230),
                width=stroke_width,
            )
    for item in rejected_objects:
        if not bool(item.get("draw_rejected_bbox")):
            continue
        bbox = _json_float_mapping(item.get("screen_bbox_px"))
        if bbox:
            _draw_guide_bbox(
                draw,
                bbox,
                label=f"reject {item.get('object_id') or 'object'}",
                color=(220, 38, 38, 225),
                width=stroke_width,
            )

    panel_lines = [
        "EDGE/SILHOUETTE GUIDE",
        "Orange: source edges/crop guard. Green: clean object anchors.",
        "Blue: trusted opening zones. Red: rejected noisy anchors.",
        "Use as structure only; never render labels, boxes, or edge marks.",
    ]
    if rejected_objects:
        rejected_ids = ", ".join(
            str(item.get("object_id") or "object") for item in rejected_objects[:5]
        )
        panel_lines.append(f"Rejected anchors: {rejected_ids}")
    _draw_guide_panel(draw, panel_lines)
    return Image.alpha_composite(guide, overlay)


def _layout_guard_mask_image(
    *,
    image_size: tuple[int, int],
    opening_boxes: list[tuple[str, dict[str, float]]],
    rejected_objects: list[dict[str, JSONValue]],
) -> tuple[Image.Image, dict[str, JSONValue]]:
    image_width, image_height = image_size
    mask = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    border = max(8, round(min(image_width, image_height) * _LAYOUT_GUARD_BORDER_RATIO))
    locked_regions = 0
    for box in (
        [0, 0, border, image_height],
        [image_width - border, 0, image_width, image_height],
        [0, 0, image_width, border],
        [0, image_height - border, image_width, image_height],
    ):
        draw.rectangle(box, fill=(255, 255, 255, 255))
        locked_regions += 1

    opening_padding = max(
        2, round(min(image_width, image_height) * _LAYOUT_GUARD_OPENING_PADDING_RATIO)
    )
    for _, bbox in opening_boxes:
        _draw_mask_bbox(
            draw,
            bbox,
            padding_px=opening_padding,
            image_size=image_size,
        )
        locked_regions += 1

    rejected_mask_lock_count = 0
    for item in rejected_objects:
        if not bool(item.get("mask_locked")):
            continue
        bbox = _json_float_mapping(item.get("screen_bbox_px"))
        if not bbox:
            continue
        _draw_mask_bbox(draw, bbox, padding_px=0, image_size=image_size)
        locked_regions += 1
        rejected_mask_lock_count += 1

    alpha = mask.getchannel("A")
    get_flattened_data = getattr(alpha, "get_flattened_data", None)
    alpha_data = (
        get_flattened_data() if callable(get_flattened_data) else alpha.getdata()
    )
    alpha_values = list(alpha_data)
    locked_pixels = sum(1 for value in alpha_values if value > 0)
    total_pixels = max(1, len(alpha_values))
    locked_ratio = locked_pixels / total_pixels
    return mask, {
        "mask_locked_region_count": locked_regions,
        "mask_locked_rejected_object_count": rejected_mask_lock_count,
        "mask_locked_pixel_ratio": round(locked_ratio, 6),
        "mask_editable_pixel_ratio": round(1.0 - locked_ratio, 6),
        "mask_semantics": "transparent_pixels_are_editable_opaque_pixels_are_preserved",
    }


def _build_layout_repair_mask(
    *,
    source_payload: Mapping[str, object],
    edge_silhouette_metadata: Mapping[str, object],
    output_path: Path,
    source_image_path: Path,
) -> LayoutRepairMaskBuild:
    try:
        with Image.open(source_image_path) as image:
            image_size = image.size
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Unable to decode repair mask source: {source_image_path}"
        ) from exc

    mask, metadata = _layout_repair_mask_image(
        image_size=image_size,
        source_payload=source_payload,
        edge_silhouette_metadata=edge_silhouette_metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)
    return LayoutRepairMaskBuild(
        image_path=output_path,
        metadata={
            "enabled": True,
            "image": str(output_path),
            **metadata,
        },
    )


def _build_object_scope_edit_mask(
    *,
    source_payload: Mapping[str, object],
    edit_operations: Sequence[SnapshotEditOperation],
    output_path: Path,
    source_image_path: Path,
) -> ObjectScopeEditMaskBuild:
    try:
        with Image.open(source_image_path) as image:
            image_size = image.size
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Unable to decode object-scope edit mask source: {source_image_path}"
        ) from exc

    mask, metadata = _object_scope_edit_mask_image(
        image_size=image_size,
        source_payload=source_payload,
        edit_operations=edit_operations,
    )
    if not bool(metadata.get("enabled")):
        return ObjectScopeEditMaskBuild(image_path=None, metadata=metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)
    return ObjectScopeEditMaskBuild(
        image_path=output_path,
        metadata={
            **metadata,
            "image": str(output_path),
        },
    )


def _should_use_object_scope_edit_mask(
    edit_operations: Sequence[SnapshotEditOperation],
) -> bool:
    return any(
        (operation.operation_type or "prompt") in _OBJECT_SCOPE_MASK_OPERATION_TYPES
        for operation in edit_operations
    )


def _object_scope_edit_mask_image(
    *,
    image_size: tuple[int, int],
    source_payload: Mapping[str, object],
    edit_operations: Sequence[SnapshotEditOperation],
) -> tuple[Image.Image, dict[str, JSONValue]]:
    image_width, image_height = image_size
    mask = Image.new("RGBA", image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    padding_px = max(
        1, round(min(image_width, image_height) * _OBJECT_SCOPE_MASK_PADDING_RATIO)
    )
    near_padding_px = max(
        padding_px,
        round(min(image_width, image_height) * _OBJECT_SCOPE_MASK_NEAR_PADDING_RATIO),
    )
    object_candidates = _object_scope_visible_object_candidates(
        source_payload,
        image_size=image_size,
    )
    operation_records: list[dict[str, JSONValue]] = []
    protected_objects: list[dict[str, JSONValue]] = []
    rejected_protection_objects: list[dict[str, JSONValue]] = []
    editable_region_count = 0

    for index, operation in enumerate(edit_operations, start=1):
        if operation.bbox_norm is None:
            continue
        operation_bbox = _bbox_norm_to_screen_bbox(
            operation.bbox_norm,
            image_size=image_size,
        )
        if operation_bbox is None:
            continue

        _draw_editable_repair_bbox(
            draw,
            operation_bbox,
            padding_px=padding_px,
            image_size=image_size,
        )
        editable_region_count += 1
        target_tokens = _object_scope_target_tokens(operation)
        operation_type = operation.operation_type or "prompt"
        expanded_bbox = _expand_screen_bbox(
            operation_bbox,
            padding_px=near_padding_px,
            image_size=image_size,
        )
        protected_for_operation: list[dict[str, JSONValue]] = []

        for candidate in object_candidates:
            candidate_bbox = _json_float_mapping(candidate.get("screen_bbox_px"))
            if not candidate_bbox:
                continue
            is_target = (
                operation_type != "add"
                and bool(target_tokens)
                and _object_scope_candidate_matches_target(candidate, target_tokens)
            )
            if is_target:
                continue
            if not _object_scope_bbox_should_be_preserved(
                candidate_bbox,
                expanded_bbox,
            ):
                continue

            rejection_reason = _object_scope_protection_rejection_reason(
                candidate=candidate,
                candidate_bbox=candidate_bbox,
                operation_bbox=operation_bbox,
                image_size=image_size,
                padding_px=padding_px,
            )
            if rejection_reason is not None:
                rejected = {
                    "operation_id": operation.object_id,
                    "object_id": _json_safe(candidate.get("object_id")),
                    "object_type": _json_safe(candidate.get("object_type")),
                    "reason": rejection_reason,
                    "screen_bbox_px": _json_safe(candidate_bbox),
                    "bbox_norm": _json_safe(candidate.get("bbox_norm")),
                }
                rejected_protection_objects.append(rejected)
                continue

            _draw_mask_bbox(
                draw,
                candidate_bbox,
                padding_px=padding_px,
                image_size=image_size,
            )
            protected = {
                "operation_id": operation.object_id,
                "object_id": _json_safe(candidate.get("object_id")),
                "object_type": _json_safe(candidate.get("object_type")),
                "reason": "non_target_object_overlaps_edit_region",
                "screen_bbox_px": _json_safe(candidate_bbox),
                "bbox_norm": _json_safe(candidate.get("bbox_norm")),
            }
            protected_for_operation.append(protected)
            protected_objects.append(protected)

        operation_records.append(
            {
                "operation_index": index,
                "operation_id": operation.object_id,
                "operation_type": operation_type,
                "object_name": operation.object_name,
                "target_tokens": target_tokens,
                "screen_bbox_px": _json_safe(operation_bbox),
                "protected_object_count": len(protected_for_operation),
                "protected_objects": protected_for_operation[:20],
                "rejected_protection_object_count": sum(
                    1
                    for item in rejected_protection_objects
                    if item.get("operation_id") == operation.object_id
                ),
            }
        )

    if editable_region_count == 0:
        return mask, {
            "enabled": False,
            "image": None,
            "reason": "no_bbox_edit_operations",
            "mask_semantics": "transparent_pixels_are_editable_opaque_pixels_are_preserved",
            "editable_region_count": 0,
            "protected_object_count": 0,
            "operations": operation_records,
        }

    alpha = mask.getchannel("A")
    get_flattened_data = getattr(alpha, "get_flattened_data", None)
    alpha_data = (
        get_flattened_data() if callable(get_flattened_data) else alpha.getdata()
    )
    alpha_values = list(alpha_data)
    editable_pixels = sum(1 for value in alpha_values if value == 0)
    total_pixels = max(1, len(alpha_values))
    editable_ratio = editable_pixels / total_pixels
    if (
        editable_ratio < _OBJECT_SCOPE_MIN_EDITABLE_PIXEL_RATIO
        and protected_objects
    ):
        fallback_mask, fallback_metadata = _object_scope_bbox_only_mask_image(
            image_size=image_size,
            edit_operations=edit_operations,
            padding_px=padding_px,
        )
        return fallback_mask, {
            **fallback_metadata,
            "fallback_reason": "protected_objects_closed_edit_region",
            "discarded_protected_object_count": len(protected_objects),
            "discarded_protected_objects": protected_objects[:80],
            "rejected_protection_object_count": len(rejected_protection_objects),
            "rejected_protection_objects": rejected_protection_objects[:80],
            "operations": operation_records,
        }
    return mask, {
        "enabled": True,
        "image": None,
        "mask_semantics": "transparent_pixels_are_editable_opaque_pixels_are_preserved",
        "editable_region_count": editable_region_count,
        "editable_pixel_ratio": round(editable_ratio, 6),
        "preserved_pixel_ratio": round(1.0 - editable_ratio, 6),
        "protected_object_count": len(protected_objects),
        "protected_objects": protected_objects[:80],
        "rejected_protection_object_count": len(rejected_protection_objects),
        "rejected_protection_objects": rejected_protection_objects[:80],
        "operations": operation_records,
    }


def _object_scope_visible_object_candidates(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> list[dict[str, JSONValue]]:
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return []
    candidates: list[dict[str, JSONValue]] = []
    for item in visible_objects:
        payload = _mapping(item)
        object_id = _text(payload.get("id"))
        bbox = _screen_bbox(payload.get("screenBboxPx"))
        if object_id is None or bbox is None:
            continue
        normalized = _normalized_bbox_metrics(
            bbox,
            image_width=image_size[0],
            image_height=image_size[1],
        )
        object_type = _object_type_label(payload)
        quality_rejection_reason = _layout_guard_bbox_rejection_reason(
            object_type=object_type,
            normalized=normalized,
        )
        object_name = (
            _text(payload.get("label"))
            or _text(payload.get("name"))
            or _text(payload.get("type"))
            or object_id
        )
        candidates.append(
            {
                "object_id": object_id,
                "object_type": object_type,
                "object_name": object_name,
                "screen_bbox_px": bbox,
                "bbox_norm": {
                    "x": round(normalized["x"], 6),
                    "y": round(normalized["y"], 6),
                    "width": round(normalized["width"], 6),
                    "height": round(normalized["height"], 6),
                },
                "area_ratio": round(normalized["area"], 6),
                "edge_touch_count": int(normalized["edge_touch_count"]),
                "aspect_ratio": round(normalized["aspect_ratio"], 6),
                "quality_rejection_reason": quality_rejection_reason,
            }
        )
    return candidates


def _object_scope_protection_rejection_reason(
    *,
    candidate: Mapping[str, object],
    candidate_bbox: Mapping[str, float],
    operation_bbox: Mapping[str, float],
    image_size: tuple[int, int],
    padding_px: int,
) -> str | None:
    quality_rejection_reason = _text(candidate.get("quality_rejection_reason"))
    if quality_rejection_reason is not None:
        return quality_rejection_reason

    protected_bbox = _expand_screen_bbox(
        candidate_bbox,
        padding_px=padding_px,
        image_size=image_size,
    )
    operation_area = max(1.0, _bbox_area(operation_bbox))
    locked_fraction = _bbox_intersection_area(protected_bbox, operation_bbox) / (
        operation_area
    )
    if locked_fraction > _OBJECT_SCOPE_MAX_LOCKED_TARGET_FRACTION:
        return "would_close_edit_region"
    return None


def _object_scope_bbox_only_mask_image(
    *,
    image_size: tuple[int, int],
    edit_operations: Sequence[SnapshotEditOperation],
    padding_px: int,
) -> tuple[Image.Image, dict[str, JSONValue]]:
    mask = Image.new("RGBA", image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    editable_region_count = 0
    for operation in edit_operations:
        if operation.bbox_norm is None:
            continue
        operation_bbox = _bbox_norm_to_screen_bbox(
            operation.bbox_norm,
            image_size=image_size,
        )
        if operation_bbox is None:
            continue
        _draw_editable_repair_bbox(
            draw,
            operation_bbox,
            padding_px=padding_px,
            image_size=image_size,
        )
        editable_region_count += 1

    editable_ratio = _mask_editable_pixel_ratio(mask)
    return mask, {
        "enabled": editable_region_count > 0,
        "image": None,
        "mask_semantics": "transparent_pixels_are_editable_opaque_pixels_are_preserved",
        "editable_region_count": editable_region_count,
        "editable_pixel_ratio": round(editable_ratio, 6),
        "preserved_pixel_ratio": round(1.0 - editable_ratio, 6),
        "protected_object_count": 0,
        "protected_objects": [],
    }


def _mask_editable_pixel_ratio(mask: Image.Image) -> float:
    alpha = mask.getchannel("A")
    get_flattened_data = getattr(alpha, "get_flattened_data", None)
    alpha_data = (
        get_flattened_data() if callable(get_flattened_data) else alpha.getdata()
    )
    alpha_values = list(alpha_data)
    editable_pixels = sum(1 for value in alpha_values if value == 0)
    return editable_pixels / max(1, len(alpha_values))


def _object_scope_target_tokens(operation: SnapshotEditOperation) -> list[str]:
    raw_tokens = (
        operation.object_id,
        operation.object_name,
    )
    tokens: list[str] = []
    for raw_token in raw_tokens:
        normalized = _object_scope_normalized_token(raw_token)
        if normalized and normalized not in tokens and normalized != "manual_box":
            tokens.append(normalized)
    return tokens


def _object_scope_candidate_matches_target(
    candidate: Mapping[str, object],
    target_tokens: Sequence[str],
) -> bool:
    candidate_tokens = [
        _object_scope_normalized_token(candidate.get("object_id")),
        _object_scope_normalized_token(candidate.get("object_type")),
        _object_scope_normalized_token(candidate.get("object_name")),
    ]
    for target_token in target_tokens:
        for candidate_token in candidate_tokens:
            if not candidate_token:
                continue
            if target_token == candidate_token:
                return True
            if len(target_token) >= 4 and target_token in candidate_token:
                return True
            if len(candidate_token) >= 4 and candidate_token in target_token:
                return True
    return False


def _object_scope_normalized_token(value: object) -> str:
    text = _text(value)
    if text is None:
        return ""
    normalized_chars = [
        character.lower() if character.isalnum() else "_"
        for character in text.strip()
    ]
    normalized = "_".join(
        part for part in "".join(normalized_chars).split("_") if part
    )
    parts = normalized.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def _object_scope_bbox_should_be_preserved(
    candidate_bbox: Mapping[str, float],
    edit_bbox: Mapping[str, float],
) -> bool:
    if _bbox_intersection_area(candidate_bbox, edit_bbox) > 0:
        return True
    center_x = candidate_bbox["minX"] + candidate_bbox["width"] / 2
    center_y = candidate_bbox["minY"] + candidate_bbox["height"] / 2
    return (
        edit_bbox["minX"] <= center_x <= edit_bbox["maxX"]
        and edit_bbox["minY"] <= center_y <= edit_bbox["maxY"]
    )


def _bbox_intersection_area(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    left = max(first["minX"], second["minX"])
    top = max(first["minY"], second["minY"])
    right = min(first["maxX"], second["maxX"])
    bottom = min(first["maxY"], second["maxY"])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _expand_screen_bbox(
    bbox: Mapping[str, float],
    *,
    padding_px: int,
    image_size: tuple[int, int],
) -> dict[str, float]:
    image_width, image_height = image_size
    min_x = max(0.0, bbox["minX"] - padding_px)
    min_y = max(0.0, bbox["minY"] - padding_px)
    max_x = min(float(image_width), bbox["maxX"] + padding_px)
    max_y = min(float(image_height), bbox["maxY"] + padding_px)
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "width": max(0.0, max_x - min_x),
        "height": max(0.0, max_y - min_y),
    }


def _layout_repair_mask_image(
    *,
    image_size: tuple[int, int],
    source_payload: Mapping[str, object],
    edge_silhouette_metadata: Mapping[str, object],
) -> tuple[Image.Image, dict[str, JSONValue]]:
    image_width, image_height = image_size
    mask = Image.new("RGBA", image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    padding_px = max(2, round(min(image_width, image_height) * 0.025))
    repair_regions: list[dict[str, JSONValue]] = []

    def add_region(
        *,
        region_type: str,
        label: str,
        bbox: Mapping[str, float],
        padding: int = padding_px,
    ) -> None:
        if _bbox_area(bbox) <= 0:
            return
        _draw_editable_repair_bbox(
            draw,
            bbox,
            padding_px=padding,
            image_size=image_size,
        )
        repair_regions.append(
            {
                "type": region_type,
                "label": label,
                "screen_bbox_px": {
                    "minX": round(bbox["minX"], 2),
                    "minY": round(bbox["minY"], 2),
                    "maxX": round(bbox["maxX"], 2),
                    "maxY": round(bbox["maxY"], 2),
                    "width": round(bbox["width"], 2),
                    "height": round(bbox["height"], 2),
                },
            }
        )

    crop_band = max(8, round(min(image_width, image_height) * 0.03))
    for label, bbox in (
        (
            "left_crop_edge",
            {"minX": 0, "minY": 0, "maxX": crop_band, "maxY": image_height},
        ),
        (
            "right_crop_edge",
            {
                "minX": image_width - crop_band,
                "minY": 0,
                "maxX": image_width,
                "maxY": image_height,
            },
        ),
    ):
        screen_bbox = _screen_bbox(bbox)
        if screen_bbox is not None:
            add_region(
                region_type="near_crop_wall_surface",
                label=label,
                bbox=screen_bbox,
                padding=0,
            )

    for surface in _sequence(edge_silhouette_metadata.get("semantic_surfaces")):
        payload = _mapping(surface)
        near_edges = _sequence(payload.get("near_crop_edges")) or _sequence(
            payload.get("touches_crop_edges")
        )
        if not near_edges:
            continue
        bbox = _bbox_norm_to_screen_bbox(
            _mapping(payload.get("bbox_norm")),
            image_size=image_size,
        )
        if bbox is not None:
            add_region(
                region_type="near_crop_wall_surface",
                label=_text(payload.get("surface_id")) or "room_surface",
                bbox=bbox,
            )

    for label, bbox in _room_opening_guide_boxes(source_payload):
        add_region(region_type="opening", label=label, bbox=bbox)

    for item in _sequence(edge_silhouette_metadata.get("rejected_objects")):
        payload = _mapping(item)
        if not bool(payload.get("draw_rejected_bbox")):
            continue
        bbox = _json_float_mapping(payload.get("screen_bbox_px"))
        if bbox:
            add_region(
                region_type="rejected_anchor",
                label=_text(payload.get("object_id")) or "rejected_anchor",
                bbox=bbox,
            )

    for item in _sequence(edge_silhouette_metadata.get("clean_objects")):
        payload = _mapping(item)
        object_type = _text(payload.get("object_type")) or "object"
        area_ratio = _number(payload.get("area_ratio")) or 0.0
        if object_type not in _LAYOUT_REPAIR_OBJECT_TYPES and area_ratio < 0.035:
            continue
        bbox = _json_float_mapping(payload.get("screen_bbox_px"))
        if bbox:
            add_region(
                region_type="object_footprint",
                label=_text(payload.get("object_id")) or object_type,
                bbox=bbox,
            )

    alpha = mask.getchannel("A")
    get_flattened_data = getattr(alpha, "get_flattened_data", None)
    alpha_data = (
        get_flattened_data() if callable(get_flattened_data) else alpha.getdata()
    )
    alpha_values = list(alpha_data)
    editable_pixels = sum(1 for value in alpha_values if value == 0)
    total_pixels = max(1, len(alpha_values))
    editable_ratio = editable_pixels / total_pixels
    return mask, {
        "mask_semantics": "transparent_pixels_are_editable_opaque_pixels_are_preserved",
        "editable_pixel_ratio": round(editable_ratio, 6),
        "preserved_pixel_ratio": round(1.0 - editable_ratio, 6),
        "repair_region_count": len(repair_regions),
        "repair_regions": repair_regions[:80],
    }


def _draw_editable_repair_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Mapping[str, float],
    *,
    padding_px: int,
    image_size: tuple[int, int],
) -> None:
    image_width, image_height = image_size
    left = max(0, round(bbox["minX"]) - padding_px)
    top = max(0, round(bbox["minY"]) - padding_px)
    right = min(image_width, round(bbox["maxX"]) + padding_px)
    bottom = min(image_height, round(bbox["maxY"]) + padding_px)
    if right <= left or bottom <= top:
        return
    draw.rectangle([left, top, right, bottom], fill=(0, 0, 0, 0))


def _bbox_norm_to_screen_bbox(
    bbox_norm: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> dict[str, float] | None:
    x = _optional_number(bbox_norm.get("x"))
    y = _optional_number(bbox_norm.get("y"))
    width = _optional_number(bbox_norm.get("width"))
    height = _optional_number(bbox_norm.get("height"))
    if x is None or y is None or width is None or height is None:
        return None
    image_width, image_height = image_size
    return _screen_bbox(
        {
            "minX": x * image_width,
            "minY": y * image_height,
            "width": width * image_width,
            "height": height * image_height,
        }
    )


def _layout_guard_object_candidates(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> tuple[list[dict[str, JSONValue]], list[dict[str, JSONValue]]]:
    image_width, image_height = image_size
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return [], []
    clean_objects: list[dict[str, JSONValue]] = []
    rejected_objects: list[dict[str, JSONValue]] = []
    for item in visible_objects:
        payload = _mapping(item)
        object_id = _text(payload.get("id"))
        object_type = _object_type_label(payload)
        bbox = _screen_bbox(payload.get("screenBboxPx"))
        if object_id is None or bbox is None:
            continue
        normalized = _normalized_bbox_metrics(
            bbox,
            image_width=image_width,
            image_height=image_height,
        )
        reason = _layout_guard_bbox_rejection_reason(
            object_type=object_type,
            normalized=normalized,
        )
        serialized = {
            "object_id": object_id,
            "object_type": object_type,
            "screen_bbox_px": bbox,
            "bbox_norm": {
                "x": round(normalized["x"], 6),
                "y": round(normalized["y"], 6),
                "width": round(normalized["width"], 6),
                "height": round(normalized["height"], 6),
            },
            "area_ratio": round(normalized["area"], 6),
            "edge_touch_count": int(normalized["edge_touch_count"]),
            "aspect_ratio": round(normalized["aspect_ratio"], 6),
        }
        if reason is None:
            clean_objects.append(serialized)
            continue
        mask_locked = normalized["area"] <= 0.12
        rejected_objects.append(
            {
                **serialized,
                "reason": reason,
                "mask_locked": mask_locked,
                "draw_rejected_bbox": normalized["area"] <= 0.35,
            }
        )
    clean_objects.sort(key=lambda item: float(item["area_ratio"]), reverse=True)
    return clean_objects[:_EDGE_SILHOUETTE_GUIDE_MAX_OBJECTS], rejected_objects


def _layout_guard_bbox_rejection_reason(
    *,
    object_type: str,
    normalized: Mapping[str, float],
) -> str | None:
    area = normalized["area"]
    width = normalized["width"]
    height = normalized["height"]
    aspect_ratio = normalized["aspect_ratio"]
    edge_touch_count = int(normalized["edge_touch_count"])
    max_area = _LAYOUT_GUARD_MAX_OBJECT_AREA_BY_TYPE.get(
        object_type,
        _LAYOUT_GUARD_DEFAULT_MAX_OBJECT_AREA,
    )
    if area > max_area:
        return "bbox_area_too_large_for_object_type"
    if width > 0.66 and height > 0.66 and object_type != "rug":
        return "bbox_covers_too_much_of_frame"
    if edge_touch_count >= 3 and object_type != "rug":
        return "bbox_touches_too_many_image_edges"
    if (
        aspect_ratio > 10.0 or aspect_ratio < 0.1
    ) and object_type not in _LAYOUT_GUARD_SLENDER_OBJECT_TYPES:
        return "bbox_aspect_ratio_implausible"
    return None


def _normalized_bbox_metrics(
    bbox: Mapping[str, float],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    width = max(0.0, bbox["width"] / max(1, image_width))
    height = max(0.0, bbox["height"] / max(1, image_height))
    x = max(0.0, min(1.0, bbox["minX"] / max(1, image_width)))
    y = max(0.0, min(1.0, bbox["minY"] / max(1, image_height)))
    edge_touch_count = float(
        sum(
            (
                x <= 0.01,
                y <= 0.01,
                x + width >= 0.99,
                y + height >= 0.99,
            )
        )
    )
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": width / max(height, 0.0001),
        "edge_touch_count": edge_touch_count,
    }


def _draw_mask_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Mapping[str, float],
    *,
    padding_px: int,
    image_size: tuple[int, int],
) -> None:
    image_width, image_height = image_size
    left = max(0, round(bbox["minX"]) - padding_px)
    top = max(0, round(bbox["minY"]) - padding_px)
    right = min(image_width, round(bbox["maxX"]) + padding_px)
    bottom = min(image_height, round(bbox["maxY"]) + padding_px)
    if right <= left or bottom <= top:
        return
    draw.rectangle([left, top, right, bottom], fill=(255, 255, 255, 255))


def _json_float_mapping(value: object) -> dict[str, float]:
    payload = _mapping(value)
    parsed: dict[str, float] = {}
    for key in ("minX", "minY", "maxX", "maxY", "width", "height"):
        number = _optional_number(payload.get(key))
        if number is not None:
            parsed[key] = number
    return parsed


def _object_type_label(payload: Mapping[str, object]) -> str:
    raw = (_text(payload.get("type")) or _text(payload.get("id")) or "object").lower()
    parts = raw.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts) or "object"


def _room_opening_guide_boxes(
    source_payload: Mapping[str, object],
) -> list[tuple[str, dict[str, float]]]:
    room = _mapping(source_payload.get("room"))
    openings = _mapping(room.get("openings"))
    boxes: list[tuple[str, dict[str, float]]] = []
    for opening_type in ("windows", "doors"):
        for index, item in enumerate(_sequence(openings.get(opening_type)), start=1):
            bbox = _screen_bbox(_mapping(item).get("screenBboxPx"))
            if bbox is not None:
                boxes.append((f"{opening_type[:-1]} {index}", bbox))
    return boxes


def _draw_guide_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Mapping[str, float],
    *,
    label: str,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    min_x = round(bbox["minX"])
    min_y = round(bbox["minY"])
    max_x = round(bbox["maxX"])
    max_y = round(bbox["maxY"])
    draw.rectangle([min_x, min_y, max_x, max_y], outline=color, width=width)
    label_text = label[:30]
    text_x = min_x + 4
    text_y = max(0, min_y - 18)
    draw.rectangle(
        [text_x - 3, text_y, text_x + 8 + len(label_text) * 7, text_y + 16],
        fill=(255, 255, 255, 220),
    )
    draw.text((text_x, text_y + 2), label_text, fill=color)


def _draw_guide_panel(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
) -> None:
    panel_x = 12
    panel_y = 12
    line_height = 18
    panel_width = max(360, max(len(line) for line in lines) * 7 + 24)
    panel_height = len(lines) * line_height + 18
    draw.rectangle(
        [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
        fill=(255, 255, 255, 230),
        outline=(255, 132, 0, 230),
        width=2,
    )
    for index, line in enumerate(lines):
        fill = (30, 30, 30, 255) if index else (180, 72, 0, 255)
        draw.text(
            (panel_x + 10, panel_y + 9 + index * line_height),
            line,
            fill=fill,
        )


def _bbox_area(bbox: Mapping[str, float]) -> float:
    return max(0.0, bbox["width"]) * max(0.0, bbox["height"])


def _visible_object_crop_candidates(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> list[LockedGeometryCropCandidate]:
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return []
    candidates: list[LockedGeometryCropCandidate] = []
    for item in visible_objects:
        payload = _mapping(item)
        object_id = _text(payload.get("id"))
        bbox = _screen_bbox(payload.get("screenBboxPx"))
        if object_id is None or bbox is None:
            continue
        crop_box = _clamped_crop_box(bbox, image_size=image_size)
        if crop_box is None:
            continue
        object_name = (
            _text(payload.get("label"))
            or _text(payload.get("name"))
            or _text(payload.get("type"))
            or object_id
        )
        candidates.append(
            LockedGeometryCropCandidate(
                object_id=object_id,
                object_name=object_name,
                screen_bbox_px=bbox,
                crop_box_px=crop_box,
            )
        )
    return candidates


def _screen_bbox(value: object) -> dict[str, float] | None:
    payload = _mapping(value)
    min_x = _optional_number(payload.get("minX"))
    min_y = _optional_number(payload.get("minY"))
    if min_x is None:
        min_x = _optional_number(payload.get("x"))
    if min_y is None:
        min_y = _optional_number(payload.get("y"))
    if min_x is None or min_y is None:
        return None

    width = _optional_number(payload.get("width"))
    height = _optional_number(payload.get("height"))
    max_x = _optional_number(payload.get("maxX"))
    max_y = _optional_number(payload.get("maxY"))
    if max_x is None and width is not None:
        max_x = min_x + width
    if max_y is None and height is not None:
        max_y = min_y + height
    if max_x is None or max_y is None:
        return None

    width = max_x - min_x
    height = max_y - min_y
    if width < _LOCKED_GEOMETRY_MIN_BBOX_PX or height < _LOCKED_GEOMETRY_MIN_BBOX_PX:
        return None
    return {
        "minX": round(min_x, 2),
        "minY": round(min_y, 2),
        "maxX": round(max_x, 2),
        "maxY": round(max_y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
    }


def _clamped_crop_box(
    bbox: Mapping[str, float],
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return None
    bbox_width = max(0.0, bbox["maxX"] - bbox["minX"])
    bbox_height = max(0.0, bbox["maxY"] - bbox["minY"])
    padding = max(
        2.0, max(bbox_width, bbox_height) * _LOCKED_GEOMETRY_CROP_PADDING_RATIO
    )
    left = max(0, math.floor(bbox["minX"] - padding))
    top = max(0, math.floor(bbox["minY"] - padding))
    right = min(image_width, math.ceil(bbox["maxX"] + padding))
    bottom = min(image_height, math.ceil(bbox["maxY"] + padding))
    if (
        right - left < _LOCKED_GEOMETRY_MIN_BBOX_PX
        or bottom - top < _LOCKED_GEOMETRY_MIN_BBOX_PX
    ):
        return None
    return left, top, right, bottom


def _normalized_cell_bbox(
    *,
    cell_left: int,
    cell_top: int,
    atlas_width: int,
    atlas_height: int,
) -> dict[str, float]:
    return {
        "x": round(cell_left / atlas_width, 6),
        "y": round(cell_top / atlas_height, 6),
        "width": round(_LOCKED_GEOMETRY_CELL_SIZE_PX / atlas_width, 6),
        "height": round(_LOCKED_GEOMETRY_CELL_SIZE_PX / atlas_height, 6),
    }


def _optional_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _image_size_dimensions(image_size: str) -> tuple[int, int] | None:
    normalized = image_size.strip().lower()
    if normalized == "auto" or "x" not in normalized:
        return None
    width_text, height_text = normalized.split("x", maxsplit=1)
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _corner_background_rgb(image: Image.Image) -> tuple[int, int, int]:
    corners = [
        image.getpixel((0, 0)),
        image.getpixel((max(0, image.width - 1), 0)),
        image.getpixel((0, max(0, image.height - 1))),
        image.getpixel((max(0, image.width - 1), max(0, image.height - 1))),
    ]
    channels = []
    for index in range(3):
        channel_values = sorted(int(pixel[index]) for pixel in corners)
        channels.append(channel_values[len(channel_values) // 2])
    return channels[0], channels[1], channels[2]


def _transform_snapshot_payload_screen_space(
    payload: Mapping[str, object],
    *,
    target_width: int,
    target_height: int,
    scale: float,
    offset_x: int,
    offset_y: int,
) -> dict[str, JSONValue]:
    transformed = _json_clone(payload)
    canvas = _mapping(transformed.get("canvas"))
    transformed["canvas"] = {
        **canvas,
        "widthPx": target_width,
        "heightPx": target_height,
    }
    camera = _mapping(transformed.get("camera"))
    if camera:
        transformed["camera"] = {
            **camera,
            "aspect": round(target_width / max(1, target_height), 6),
        }
    for key in ("visibleObjects", "allObjects"):
        objects = transformed.get(key)
        if isinstance(objects, list):
            transformed[key] = [
                _transform_screen_fields(item, scale, offset_x, offset_y)
                if isinstance(item, dict)
                else item
                for item in objects
            ]
    room = transformed.get("room")
    if isinstance(room, dict):
        openings = room.get("openings")
        if isinstance(openings, dict):
            for opening_key in ("doors", "windows"):
                items = openings.get(opening_key)
                if isinstance(items, list):
                    openings[opening_key] = [
                        _transform_screen_fields(item, scale, offset_x, offset_y)
                        if isinstance(item, dict)
                        else item
                        for item in items
                    ]
    alignment = _mapping(transformed.get("sourceAlignment"))
    transformed["sourceAlignment"] = {
        **alignment,
        "screenSpaceAligned": True,
        "screenScale": round(scale, 6),
        "screenOffsetPx": {"x": offset_x, "y": offset_y},
        "canvasWidthPx": target_width,
        "canvasHeightPx": target_height,
    }
    return transformed


def _transform_screen_fields(
    item: dict[str, JSONValue],
    scale: float,
    offset_x: int,
    offset_y: int,
) -> dict[str, JSONValue]:
    transformed = dict(item)
    bbox = transformed.get("screenBboxPx")
    if isinstance(bbox, dict):
        transformed["screenBboxPx"] = _transform_screen_bbox(
            bbox,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
        )
    center = transformed.get("screenCenterPx")
    if isinstance(center, dict):
        transformed["screenCenterPx"] = _transform_screen_point(
            center,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
        )
    return transformed


def _transform_screen_bbox(
    bbox: Mapping[str, JSONValue],
    *,
    scale: float,
    offset_x: int,
    offset_y: int,
) -> dict[str, JSONValue]:
    source_min_x = _optional_number(bbox.get("minX"))
    source_min_y = _optional_number(bbox.get("minY"))
    if source_min_x is None:
        source_min_x = _optional_number(bbox.get("x")) or 0.0
    if source_min_y is None:
        source_min_y = _optional_number(bbox.get("y")) or 0.0

    source_width = _optional_number(bbox.get("width"))
    source_height = _optional_number(bbox.get("height"))
    source_max_x = _optional_number(bbox.get("maxX"))
    source_max_y = _optional_number(bbox.get("maxY"))
    if source_max_x is None:
        source_max_x = source_min_x + (source_width or 0.0)
    if source_max_y is None:
        source_max_y = source_min_y + (source_height or 0.0)

    min_x = source_min_x * scale + offset_x
    min_y = source_min_y * scale + offset_y
    max_x = source_max_x * scale + offset_x
    max_y = source_max_y * scale + offset_y
    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)
    return {
        "minX": round(min_x, 2),
        "minY": round(min_y, 2),
        "maxX": round(max_x, 2),
        "maxY": round(max_y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
    }


def _transform_screen_point(
    point: Mapping[str, JSONValue],
    *,
    scale: float,
    offset_x: int,
    offset_y: int,
) -> dict[str, JSONValue]:
    return {
        "x": round(_number(point.get("x")) * scale + offset_x, 2),
        "y": round(_number(point.get("y")) * scale + offset_y, 2),
    }


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _json_clone(value: Mapping[str, object]) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], deepcopy(_json_safe(value)))


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _snapshot_render_config_for_ablation(
    config: ConfigT,
    overrides: Mapping[str, object],
) -> ConfigT:
    if not overrides:
        return config
    return cast(
        ConfigT,
        replace(
            config,
            simple_prompt_mode=_bool_override(
                overrides,
                "simple_prompt_mode",
                getattr(config, "simple_prompt_mode"),
            ),
            text_contract_mode=_bool_override(
                overrides,
                "text_contract_mode",
                getattr(config, "text_contract_mode"),
            ),
            text_contract_object_limit=_int_override(
                overrides,
                "text_contract_object_limit",
                getattr(config, "text_contract_object_limit"),
            ),
            include_layout_2d_reference=_bool_override(
                overrides,
                "include_layout_2d_reference",
                getattr(config, "include_layout_2d_reference"),
            ),
            include_annotated_reference=_bool_override(
                overrides,
                "include_annotated_reference",
                getattr(config, "include_annotated_reference"),
            ),
            include_geometry_references=_bool_override(
                overrides,
                "include_geometry_references",
                getattr(config, "include_geometry_references"),
            ),
            include_locked_geometry_references=_bool_override(
                overrides,
                "include_locked_geometry_references",
                getattr(config, "include_locked_geometry_references"),
            ),
            include_preset_reference_images=_bool_override(
                overrides,
                "include_preset_reference_images",
                getattr(config, "include_preset_reference_images"),
            ),
            include_edit_operation_dsl=_bool_override(
                overrides,
                "include_edit_operation_dsl",
                getattr(config, "include_edit_operation_dsl"),
            ),
            disable_edit_mask=_bool_override(
                overrides,
                "disable_edit_mask",
                getattr(config, "disable_edit_mask", False),
            ),
            strict_layout_mode=_bool_override(
                overrides,
                "strict_layout_mode",
                getattr(config, "strict_layout_mode"),
            ),
        ),
    )


def _bool_override(
    overrides: Mapping[str, object],
    key: str,
    fallback: object,
) -> bool:
    value = overrides.get(key, fallback)
    return bool(value)


def _int_override(
    overrides: Mapping[str, object],
    key: str,
    fallback: object,
) -> int:
    value = overrides.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _optional_data_url(path: Path | None) -> str | None:
    if path is None:
        return None
    return encode_image_to_data_url(path)


def _configured_approved_render_path(
    *, output_root: Path, case: CaseConfig
) -> Path | None:
    source = case.edit_source_variant
    if source is None:
        return None
    configured_path = output_root / source.case_id / source.variant / "render.png"
    if source.selection_mode == "exact":
        return configured_path
    best_path = _best_approved_render_path(
        output_root=output_root,
        case_id=source.case_id,
        configured_path=configured_path,
    )
    return best_path or configured_path


def _best_approved_render_path(
    *,
    output_root: Path,
    case_id: str,
    configured_path: Path,
) -> Path | None:
    case_dir = output_root / case_id
    if not case_dir.is_dir():
        return None
    best_path: Path | None = None
    best_score = float("-inf")
    best_rank = len(_APPROVED_RENDER_TIE_BREAK_VARIANTS) + 1
    for variant_dir in sorted(case_dir.iterdir(), key=lambda path: path.name):
        if not variant_dir.is_dir():
            continue
        if (
            variant_dir != configured_path.parent
            and variant_dir.name not in _APPROVED_RENDER_TIE_BREAK_VARIANTS
        ):
            continue
        candidate = variant_dir / "render.png"
        if not _is_reusable_file(candidate):
            continue
        metric = _approved_layout_score_metric(candidate.parent / "metrics.json")
        if metric is None:
            continue
        score = _layout_score(metric)
        rank = _approved_render_tie_rank(candidate, configured_path)
        if score > best_score or (
            score == best_score and (best_path is None or rank < best_rank)
        ):
            best_path = candidate
            best_score = score
            best_rank = rank
    return best_path


def _approved_render_tie_rank(candidate: Path, configured_path: Path) -> int:
    if candidate == configured_path:
        return -1
    variant = candidate.parent.name
    try:
        return _APPROVED_RENDER_TIE_BREAK_VARIANTS.index(variant)
    except ValueError:
        return len(_APPROVED_RENDER_TIE_BREAK_VARIANTS)


def _require_approved_render(
    *,
    case: CaseConfig,
    approved_render_source: Path,
) -> None:
    if not _is_reusable_file(approved_render_source):
        raise FileNotFoundError(
            f"Approved render source not found for case {case.id}: "
            f"{approved_render_source}"
        )
    source = case.edit_source_variant
    if source is None:
        raise ValueError(f"Case {case.id} has no edit_source_variant.")
    if source.selection_mode == "exact":
        return
    metrics_path = approved_render_source.parent / "metrics.json"
    if not _metrics_include_passed_approved_layout_score(metrics_path):
        raise ValueError(
            f"Approved render source has no passed layout score metric: {metrics_path}"
        )


def _approved_render_is_ready(path: Path) -> bool:
    return (
        _is_reusable_file(path)
        and _approved_layout_score_metric(path.parent / "metrics.json") is not None
    )


def _metrics_include_passed_approved_layout_score(path: Path) -> bool:
    return _approved_layout_score_metric(path) is not None


def _approved_layout_score_metric(path: Path) -> Mapping[str, object] | None:
    if not _is_reusable_file(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    passed_metrics = [
        item
        for item in payload
        if (
            isinstance(item, dict)
            and item.get("status") == "passed"
            and item.get("name")
            in {"structural_layout_score", "layout_gate_score", "layout_fidelity"}
        )
    ]
    for metric_name in (
        "structural_layout_score",
        "layout_gate_score",
        "layout_fidelity",
    ):
        for item in passed_metrics:
            if item.get("name") == metric_name:
                return item
    return None


def _layout_score(metric: Mapping[str, object]) -> float:
    value = metric.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("-inf")
    return float("-inf")


def _is_reusable_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _bbox_mapping(bbox: BBoxNorm | None) -> Mapping[str, float] | None:
    if bbox is None:
        return None
    return {
        "x": bbox.x,
        "y": bbox.y,
        "width": bbox.width,
        "height": bbox.height,
    }


def _extract_result_data_url(result: Mapping[str, object]) -> str:
    image_payload = result.get("image")
    if not isinstance(image_payload, dict):
        raise ValueError("Render result is missing image payload.")
    data_url = image_payload.get("data_url")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise ValueError("Render result image payload has no data_url.")
    return data_url


def _summarize_render_result(result: Mapping[str, object] | None) -> JSONValue:
    if result is None:
        return None
    image_payload = _mapping(result.get("image"))
    usage = result.get("usage") or image_payload.get("usage")
    elapsed_ms = result.get("elapsed_ms") or image_payload.get("elapsed_ms")
    return {
        "request": _json_safe(result.get("request")),
        "models": _json_safe(result.get("models")),
        "metadata": _json_safe(result.get("metadata")),
        "usage": _json_safe(usage) if usage is not None else None,
        "elapsed_ms": _json_safe(elapsed_ms) if elapsed_ms is not None else None,
        "image": {
            "mime_type": _json_safe(image_payload.get("mime_type")),
            "size": _json_safe(image_payload.get("size")),
            "quality": _json_safe(image_payload.get("quality")),
            "aspect_ratio": _json_safe(image_payload.get("aspect_ratio")),
            "retry_count": _json_safe(image_payload.get("retry_count")),
            "usage": _json_safe(image_payload.get("usage"))
            if image_payload.get("usage") is not None
            else None,
            "elapsed_ms": _json_safe(image_payload.get("elapsed_ms"))
            if image_payload.get("elapsed_ms") is not None
            else None,
        },
    }


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return {}


def _sequence(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _json_safe(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
