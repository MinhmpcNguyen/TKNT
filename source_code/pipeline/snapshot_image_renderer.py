from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import httpx
from PIL import Image, ImageChops, ImageFilter, UnidentifiedImageError

from config import config_file, load_config
from pipeline.image_flow_logging import (
    log_image_flow_event,
    redact_for_image_log,
    summarize_image_data_url,
)
from pipeline.snapshot_prompt_compiler import (
    SnapshotPromptCompilerConfig,
    compile_snapshot_prompt,
)

logger = logging.getLogger(__name__)

SNAPSHOT_RENDER_MAX_TOKENS_ENV = "TKNT_SNAPSHOT_RENDER_MAX_TOKENS"
SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE_ENV = (
    "TKNT_SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE"
)
SNAPSHOT_RENDER_INCLUDE_ANNOTATED_REFERENCE_ENV = (
    "TKNT_SNAPSHOT_RENDER_INCLUDE_ANNOTATED_REFERENCE"
)
SNAPSHOT_RENDER_INCLUDE_GEOMETRY_REFERENCES_ENV = (
    "TKNT_SNAPSHOT_RENDER_INCLUDE_GEOMETRY_REFERENCES"
)
SNAPSHOT_RENDER_INCLUDE_LOCKED_GEOMETRY_REFERENCES_ENV = (
    "TKNT_SNAPSHOT_RENDER_INCLUDE_LOCKED_GEOMETRY_REFERENCES"
)
SNAPSHOT_RENDER_INCLUDE_PRESET_REFERENCE_IMAGES_ENV = (
    "TKNT_SNAPSHOT_RENDER_INCLUDE_PRESET_REFERENCE_IMAGES"
)
SNAPSHOT_RENDER_SIMPLE_PROMPT_ENV = "TKNT_SNAPSHOT_RENDER_SIMPLE_PROMPT"
SNAPSHOT_RENDER_TEXT_CONTRACT_ENV = "TKNT_SNAPSHOT_RENDER_TEXT_CONTRACT"
SNAPSHOT_RENDER_TEXT_CONTRACT_OBJECT_LIMIT_ENV = (
    "TKNT_SNAPSHOT_RENDER_TEXT_CONTRACT_OBJECT_LIMIT"
)
SNAPSHOT_RENDER_STRICT_LAYOUT_ENV = "TKNT_SNAPSHOT_RENDER_STRICT_LAYOUT"
OPENAI_IMAGE_API_KEY_ENV = "OPENAI_IMAGE_API_KEY"
OPENAI_IMAGE_MODEL_ENV = "TKNT_OPENAI_IMAGE_MODEL"
OPENAI_IMAGE_SIZE_ENV = "TKNT_OPENAI_IMAGE_SIZE"
OPENAI_IMAGE_QUALITY_ENV = "TKNT_OPENAI_IMAGE_QUALITY"
OPENAI_IMAGE_STREAM_PARTIAL_IMAGES_ENV = "TKNT_OPENAI_IMAGE_STREAM_PARTIAL_IMAGES"

_DEFAULT_MAX_TOKENS = 256
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-2"
_DEFAULT_OPENAI_IMAGE_SIZE = "1536x1024"
_DEFAULT_OPENAI_IMAGE_QUALITY = "medium"
_DEFAULT_OPENAI_IMAGE_STREAM_PARTIAL_IMAGES = 2
_DEFAULT_TEXT_CONTRACT_OBJECT_LIMIT = 10
_MAX_TEXT_CONTRACT_OBJECT_LIMIT = 24
_DEFAULT_STRICT_LAYOUT = True
_SUPPORTED_OPENAI_IMAGE_SIZES = {
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "auto",
}
_SUPPORTED_OPENAI_IMAGE_QUALITIES = {"low", "medium", "high", "auto"}
_BYTES_PER_MB = 1_000_000
_DEFAULT_USER_PROMPT = (
    "Convert the exact visible 3D blockout into a photorealistic interior by "
    "improving material realism, lighting, and finishes. Keep the same camera, "
    "architecture, room shell, openings, object count, object colors, object "
    "shapes, main furniture, and visible layout."
)
_DEFAULT_SIMPLE_USER_PROMPT = "make this more photorealistic, don't change anything."
_STRICT_LAYOUT_USER_PROMPT = (
    "Materialize the exact 3D source view as a photorealistic interior. "
    "Keep the same camera, crop, room shell, openings, object inventory, "
    "object positions, object scale, object silhouettes, object colors, and "
    "depth order. Only improve material realism, finish detail, lighting "
    "quality, and exterior scenery visible through existing windows."
)
_TV_CONSOLE_RENDER_IDENTITY_LABEL = "TV console with integrated flat-screen TV"
_TV_CONSOLE_RENDER_IDENTITY_HINT = (
    "Preserve the integrated black flat-screen TV/screen panel together with the "
    "console cabinet; do not simplify this object into a plain cabinet or box "
    "and do not add a separate duplicate TV."
)
_SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_SUPPORTED_ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("1:1", 1.0),
    ("1:4", 1 / 4),
    ("1:8", 1 / 8),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:1", 4.0),
    ("4:3", 4 / 3),
    ("4:5", 4 / 5),
    ("5:4", 5 / 4),
    ("8:1", 8.0),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("21:9", 21 / 9),
)
_GENERATE_SYSTEM_PROMPT = (
    "You are converting a coarse 3D interior blockout into a final photorealistic render.\n\n"
    "Use the input 3D layout image as the absolute source of truth for the camera view, framing, room layout, room shell, openings, furniture placement, object scale, object shape, object color, surface color, and occlusion.\n"
    "This is visual refinement only. Do not rationalize, simplify, redesign, complete, or improve the layout beyond making the exact visible 3D scene look photorealistic.\n"
    "The labels, dark structural edge overlays, and guide marks are only there to identify objects and boundaries. Never render labels, guide lines, callouts, colored boxes, black outline strokes, UI marks, numbers, or any other annotation.\n"
    "Object reference images may be provided for appearance memory. Use them only for the paired object's style, material, and identity, never for camera or layout.\n"
    "Do not add, remove, move, resize, rotate, straighten, center, or replace architecture, openings, built-ins, or main furniture unless the prompt explicitly assigns an object reference to it.\n"
    "Preserve the exact visible wall panels and corner seams from the 3D room shell. Do not omit a wall, erase a corner, open a closed side of the room, or move any wall edge.\n"
    "The existing furniture and room surfaces may gain realistic texture and finish detail, but their visible color palette, silhouette, footprint, pose, size, and position stay locked unless the user explicitly changed them in the layout before rendering.\n"
    "Do not add decorative accessories, rugs, plants, wall art, pillows, bedding, curtains, or surface decor unless those objects already exist in the 3D layout or are explicitly requested.\n"
    "Do not invent extra walls, doors, windows, columns, stairs, fireplaces, cabinets, wardrobes, beds, desks, tables, sofas, chairs, or other main furniture.\n"
    "Render a photorealistic version of the exact view shown in the 3D layout image. Treat this as visual refinement, not a redesign.\n"
    "No people, no text, no watermark, no room-type change."
)
_CAMERA_RERENDER_SYSTEM_PROMPT = (
    "You are re-rendering a previously approved interior design from a new 3D camera view.\n\n"
    "Use the new input 3D layout image as the absolute source of truth for camera, room shell, openings, object placement, object scale, object silhouettes, depth order, and occlusion.\n"
    "Use the previous photorealistic render and object reference images only as appearance memory for style continuity, materials, colors, and the identity of objects the user already edited.\n"
    "Do not infer the new camera angle from the old photo. The new 3D layout view controls all geometry.\n"
    "Do not add, remove, move, resize, rotate, straighten, center, or replace architecture, openings, built-ins, or main furniture. Do not rationalize or redesign the layout.\n"
    "The labels, dark structural edge overlays, and guide marks are only instructions. Never render labels, guide lines, callouts, colored boxes, black outline strokes, UI marks, numbers, or any annotation.\n"
    "Output a clean photorealistic render of the exact new 3D view. No people, no text, no watermark, no room-type change."
)
_EDIT_SYSTEM_PROMPT = (
    "You are editing an existing photorealistic interior image.\n\n"
    "The first input image is the approved photorealistic source image to edit and is the source of truth for every unedited pixel region, object, material, light relationship, camera, and layout.\n"
    "If an edit-region guide image is provided, it contains colored bounding boxes and region ID labels (such as R1, R2, R3) drawn over a copy of the photo. Use the boxes and labels only to locate which regions to edit and what operation applies to each. The boxes, ID text, outlines, colored fills, tinted overlays, and all other guide graphics are instructions only — they must never appear in the output image.\n"
    "If replacement reference images are provided, use them only for the appearance of the explicitly paired object or region.\n"
    "Change only the explicitly requested selected object, drawn region, or global preset attribute. Preserve room identity, materials, style continuity, architecture, openings, camera, layout, object count, object shapes, positions, scale, and every unedited object unless the prompt explicitly asks for that exact target to change.\n"
    "Output a clean photorealistic interior photograph. No colored rectangles, no region IDs, no label text, no guide graphics, no annotations of any kind, no people, no watermarks, no room-type change."
)
_LAYOUT_MATERIALIZATION_EDIT_SYSTEM_PROMPT = (
    "You are materializing an exact 3D interior source image into a photorealistic interior using one source-guided edit.\n\n"
    "The first input image is the exact 3D source view and is the source of truth for camera, crop, room shell, openings, object inventory, object positions, object scale, silhouettes, colors, depth order, and occlusion.\n"
    "If an edit mask is provided, transparent pixels may be photorealized, while opaque pixels preserve the source and must not be redrawn or moved.\n"
    "If a structural guide image is provided, use its semantic room-shell surfaces, crop guard, opening marks, object anchors, edge map, and rejected-anchor notes only to understand layout. Never render guide colors, lines, labels, boxes, masks, text, or annotations.\n"
    "This is material and lighting refinement only. Do not invent, add, remove, resize, move, complete, or replace walls, openings, built-ins, furniture, decor, or foreground objects.\n"
    "Output a clean photorealistic interior photograph. No colored rectangles, no labels, no guide graphics, no people, no text, no watermark, no room-type change."
)
_CAMERA_TRANSFER_EDIT_SYSTEM_PROMPT = (
    "You are editing an existing photorealistic interior image while transferring it to a new 3D camera view.\n\n"
    "The first input image is the approved photorealistic source image. Preserve its object identity, materials, colors, lighting style, and realistic finish unless explicit edit operations or changed presets say otherwise.\n"
    "The second input image is the sharp target 3D layout/camera reference. It is the source of truth for the new camera angle, framing, perspective, room shell, openings, object placement, object scale, depth order, and occlusion.\n"
    "Do not preserve the old camera from the first image. Change the output viewpoint to match the second image.\n"
    "If a structural guide image is provided, use it only to understand room-shell surfaces, crop edges, openings, object anchors, and noisy-anchor warnings. If an edit-region guide image is provided, use its colored boxes and region labels only to locate requested local edits. Guide colors, boxes, labels, outlines, tinted overlays, and all other guide graphics must never appear in the output image.\n"
    "If preset or replacement reference images are provided, use them only for the explicitly mapped preset or edit operation. Do not copy their room, camera, layout, architecture, object count, object placement, or furniture arrangement.\n"
    "Output a clean photorealistic interior photograph from the target 3D camera. No colored rectangles, no region IDs, no label text, no guide graphics, no annotations of any kind, no people, no watermarks, no room-type change."
)

RenderMode = Literal["generate", "edit", "camera_rerender"]
SceneReferenceMode = Literal[
    "none",
    "target_layout_with_scene_reference",
    "scene_reference_camera_transfer",
]
OpenAIImageQuality = Literal["low", "medium", "high"]
OpenAIImageSize = Literal["1024x1024", "1024x1536", "1536x1024"]
OpenAIImageStreamEventHandler = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class SnapshotImageRenderConfig:
    max_output_tokens: int = _DEFAULT_MAX_TOKENS
    include_layout_2d_reference: bool = False
    include_annotated_reference: bool = False
    include_geometry_references: bool = False
    include_locked_geometry_references: bool = False
    include_preset_reference_images: bool = True
    simple_prompt_mode: bool = False
    text_contract_mode: bool = False
    text_contract_object_limit: int = _DEFAULT_TEXT_CONTRACT_OBJECT_LIMIT
    include_edit_operation_dsl: bool = True
    disable_edit_mask: bool = False
    strict_layout_mode: bool = _DEFAULT_STRICT_LAYOUT
    openai_api_key: str | None = None
    openai_image_model: str = _DEFAULT_OPENAI_IMAGE_MODEL
    openai_image_size: str = _DEFAULT_OPENAI_IMAGE_SIZE
    openai_image_quality: str = _DEFAULT_OPENAI_IMAGE_QUALITY
    openai_image_stream_partial_images: int = (
        _DEFAULT_OPENAI_IMAGE_STREAM_PARTIAL_IMAGES
    )

    @classmethod
    def from_env(cls) -> SnapshotImageRenderConfig:
        max_tokens = _read_int_env(
            SNAPSHOT_RENDER_MAX_TOKENS_ENV,
            default=_DEFAULT_MAX_TOKENS,
        )

        openai_api_key = _secret_string(os.getenv(OPENAI_IMAGE_API_KEY_ENV))
        if not openai_api_key:
            try:
                from config.openai_config import OpenAIConfig  # noqa: PLC0415

                openai_api_key = OpenAIConfig.OPENAI_API_KEY
            except Exception:
                pass
        openai_image_model = _string(
            os.getenv(OPENAI_IMAGE_MODEL_ENV),
            fallback=_DEFAULT_OPENAI_IMAGE_MODEL,
        )
        openai_image_size = _normalize_openai_image_size(
            os.getenv(OPENAI_IMAGE_SIZE_ENV)
        )
        openai_image_quality = _normalize_openai_image_quality(
            os.getenv(OPENAI_IMAGE_QUALITY_ENV)
        )
        openai_image_stream_partial_images = _normalize_openai_stream_partial_images(
            os.getenv(OPENAI_IMAGE_STREAM_PARTIAL_IMAGES_ENV)
        )

        return cls(
            max_output_tokens=max(32, max_tokens),
            include_layout_2d_reference=_read_bool_env(
                SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE_ENV,
                default=False,
            ),
            include_annotated_reference=_read_bool_env(
                SNAPSHOT_RENDER_INCLUDE_ANNOTATED_REFERENCE_ENV,
                default=False,
            ),
            include_geometry_references=_read_bool_env(
                SNAPSHOT_RENDER_INCLUDE_GEOMETRY_REFERENCES_ENV,
                default=False,
            ),
            include_locked_geometry_references=_read_bool_env(
                SNAPSHOT_RENDER_INCLUDE_LOCKED_GEOMETRY_REFERENCES_ENV,
                default=False,
            ),
            include_preset_reference_images=_read_bool_env(
                SNAPSHOT_RENDER_INCLUDE_PRESET_REFERENCE_IMAGES_ENV,
                default=True,
            ),
            simple_prompt_mode=_read_bool_env(
                SNAPSHOT_RENDER_SIMPLE_PROMPT_ENV,
                default=False,
            ),
            text_contract_mode=_read_bool_env(
                SNAPSHOT_RENDER_TEXT_CONTRACT_ENV,
                default=False,
            ),
            text_contract_object_limit=_normalize_text_contract_object_limit(
                _read_int_env(
                    SNAPSHOT_RENDER_TEXT_CONTRACT_OBJECT_LIMIT_ENV,
                    default=_DEFAULT_TEXT_CONTRACT_OBJECT_LIMIT,
                )
            ),
            include_edit_operation_dsl=True,
            strict_layout_mode=_read_bool_env(
                SNAPSHOT_RENDER_STRICT_LAYOUT_ENV,
                default=_DEFAULT_STRICT_LAYOUT,
            ),
            openai_api_key=openai_api_key,
            openai_image_model=openai_image_model,
            openai_image_size=openai_image_size,
            openai_image_quality=openai_image_quality,
            openai_image_stream_partial_images=openai_image_stream_partial_images,
        )


@dataclass(frozen=True)
class SnapshotEditOperation:
    object_id: str
    object_name: str | None = None
    operation_type: Literal["prompt", "add", "replace", "recolor", "remove"] | None = (
        None
    )
    prompt: str | None = None
    bbox_norm: Mapping[str, float] | None = None
    replacement_image_data_url: str | None = None
    target_color: str | None = None


@dataclass(frozen=True)
class SnapshotLockedGeometryReference:
    object_id: str
    object_name: str | None = None
    image_data_url: str | None = None
    source: str | None = None
    screen_bbox_px: Mapping[str, float] | None = None


@dataclass(frozen=True)
class SnapshotLockedGeometryAtlasObject:
    object_id: str
    object_name: str | None = None
    screen_bbox_px: Mapping[str, float] | None = None
    atlas_index: int = 0
    cell_index: int = 0
    cell_label: str | None = None
    cell_bbox_norm: Mapping[str, float] | None = None


@dataclass(frozen=True)
class SnapshotLockedGeometryAtlas:
    atlas_index: int
    image_data_url: str | None = None
    source: str | None = None
    objects: list[SnapshotLockedGeometryAtlasObject] | None = None


@dataclass(frozen=True)
class SnapshotPresetReferenceImage:
    group: str
    key: str
    label: str
    image_data_url: str


def _system_prompt_for_render_mode(render_mode: RenderMode) -> str:
    if render_mode == "edit":
        return _EDIT_SYSTEM_PROMPT
    if render_mode == "camera_rerender":
        return _CAMERA_RERENDER_SYSTEM_PROMPT
    return _GENERATE_SYSTEM_PROMPT


def _layout_lock_mode_for_render_mode(render_mode: RenderMode) -> str:
    if render_mode == "edit":
        return "source_image_edit_lock"
    if render_mode == "camera_rerender":
        return "camera_rerender_layout_lock"
    return "strict_generate_layout_lock"


def render_snapshot_image(
    snapshot_payload: Mapping[str, object],
    *,
    snapshot_image_data_url: str,
    depth_map_image_data_url: str | None = None,
    edge_map_image_data_url: str | None = None,
    segmentation_map_image_data_url: str | None = None,
    user_prompt: str | None = None,
    layout_reference_image_data_url: str | None = None,
    annotated_reference_image_data_url: str | None = None,
    layout_guard_mask_image_data_url: str | None = None,
    layout_materialization_mode: bool = False,
    scene_reference_image_data_url: str | None = None,
    scene_reference_mode: SceneReferenceMode = "none",
    render_mode: RenderMode = "generate",
    preset_selection: Mapping[str, object] | None = None,
    edit_operations: list[SnapshotEditOperation] | None = None,
    locked_geometry_references: list[SnapshotLockedGeometryReference] | None = None,
    locked_geometry_atlases: list[SnapshotLockedGeometryAtlas] | None = None,
    edit_source_image_data_url: str | None = None,
    image_quality: OpenAIImageQuality | str | None = None,
    image_size: OpenAIImageSize | str | None = None,
    streamed: bool = False,
    stream_partial_images: int | None = None,
    stream_event_handler: OpenAIImageStreamEventHandler | None = None,
    config: SnapshotImageRenderConfig | None = None,
    trace_id: str | None = None,
) -> dict[str, object]:
    image_flow_trace_id = trace_id or uuid4().hex
    resolved_config = config or SnapshotImageRenderConfig.from_env()
    effective_image_size = _normalize_openai_image_size(
        image_size or resolved_config.openai_image_size
    )
    effective_image_quality = _normalize_openai_image_quality(
        image_quality or resolved_config.openai_image_quality
    )
    effective_stream_partial_images = (
        _normalize_openai_stream_partial_images(stream_partial_images)
        if stream_partial_images is not None
        else resolved_config.openai_image_stream_partial_images
    )
    if not streamed:
        effective_stream_partial_images = 0
    geometry_references_requested = _geometry_references_used_payload(
        depth_map_image_data_url=depth_map_image_data_url,
        edge_map_image_data_url=edge_map_image_data_url,
        segmentation_map_image_data_url=segmentation_map_image_data_url,
    )
    geometry_reference_requested = any(geometry_references_requested.values())
    geometry_reference_allowed = render_mode in {"generate", "camera_rerender"} or (
        render_mode == "edit"
        and (
            layout_materialization_mode or layout_guard_mask_image_data_url is not None
        )
    )
    geometry_reference_used = (
        geometry_reference_allowed
        and geometry_reference_requested
        and resolved_config.include_geometry_references
    )
    if (
        render_mode == "camera_rerender"
        and resolved_config.include_geometry_references
        and not all(
            (
                depth_map_image_data_url,
                edge_map_image_data_url,
                segmentation_map_image_data_url,
            )
        )
    ):
        raise ValueError(
            "Camera re-render requires depth_map_image_data_url, "
            "edge_map_image_data_url, and segmentation_map_image_data_url when "
            f"{SNAPSHOT_RENDER_INCLUDE_GEOMETRY_REFERENCES_ENV}=true."
        )
    structural_image_data_url = _normalize_image_data_url(snapshot_image_data_url)
    normalized_depth_map = (
        _normalize_image_data_url(depth_map_image_data_url)
        if geometry_reference_used and depth_map_image_data_url
        else None
    )
    normalized_edge_map = (
        _normalize_image_data_url(edge_map_image_data_url)
        if geometry_reference_used and edge_map_image_data_url
        else None
    )
    normalized_segmentation_map = (
        _normalize_image_data_url(segmentation_map_image_data_url)
        if geometry_reference_used and segmentation_map_image_data_url
        else None
    )
    if render_mode in {"generate", "camera_rerender"} and any(
        (
            normalized_depth_map,
            normalized_edge_map,
            normalized_segmentation_map,
        )
    ):
        if resolved_config.openai_image_model != _DEFAULT_OPENAI_IMAGE_MODEL:
            raise ValueError(
                f"Geometry reference rendering is locked to {_DEFAULT_OPENAI_IMAGE_MODEL}. "
                f"Configured OpenAI image model: {resolved_config.openai_image_model}."
            )
    normalized_scene_reference = (
        _normalize_image_data_url(scene_reference_image_data_url)
        if scene_reference_image_data_url
        else None
    )
    resolved_scene_reference_mode = _resolve_scene_reference_mode(
        scene_reference_mode=scene_reference_mode,
        render_mode=render_mode,
        scene_reference_image_data_url=normalized_scene_reference,
    )
    reference_only_camera_transfer_used = (
        resolved_scene_reference_mode == "scene_reference_camera_transfer"
    )
    scene_reference_used = (
        resolved_scene_reference_mode == "target_layout_with_scene_reference"
    )
    edit_camera_transfer_reference_used = (
        render_mode == "edit"
        and resolved_scene_reference_mode == "scene_reference_camera_transfer"
        and edit_source_image_data_url is not None
    )
    source_image_data_url = (
        _normalize_image_data_url(edit_source_image_data_url)
        if render_mode == "edit" and edit_source_image_data_url
        else normalized_scene_reference
        if reference_only_camera_transfer_used
        and normalized_scene_reference is not None
        else structural_image_data_url
    )
    visibility_diagnostics = _build_visibility_diagnostics(snapshot_payload)
    effective_snapshot_payload = _snapshot_payload_with_effective_visible_objects(
        snapshot_payload,
        visibility_diagnostics=visibility_diagnostics,
    )
    visible_objects = _extract_visible_objects(effective_snapshot_payload)
    normalized_operations = _normalize_edit_operations(
        edit_operations or [],
        visible_objects=visible_objects,
    )
    locked_geometry_reference_requested_count = len(locked_geometry_references or [])
    locked_geometry_atlas_requested_count = len(locked_geometry_atlases or [])
    normalized_locked_geometry_references = (
        _normalize_locked_geometry_references(locked_geometry_references or [])
        if resolved_config.include_locked_geometry_references
        else []
    )
    normalized_locked_geometry_atlases = (
        _normalize_locked_geometry_atlases(locked_geometry_atlases or [])
        if resolved_config.include_locked_geometry_references
        else []
    )
    preset_prompt = _build_preset_prompt(preset_selection)
    preset_reference_images = (
        _build_preset_reference_images(preset_selection)
        if resolved_config.include_preset_reference_images
        and render_mode in {"generate", "camera_rerender", "edit"}
        else []
    )
    structural_reference_used = (
        normalized_depth_map is not None
        or normalized_edge_map is not None
        or normalized_segmentation_map is not None
        or bool(normalized_locked_geometry_references)
        or bool(normalized_locked_geometry_atlases)
    )
    if (
        render_mode in {"generate", "camera_rerender"}
        and structural_reference_used
        and resolved_config.openai_image_model != _DEFAULT_OPENAI_IMAGE_MODEL
    ):
        raise ValueError(
            f"Geometry reference rendering is locked to {_DEFAULT_OPENAI_IMAGE_MODEL}. "
            f"Configured OpenAI image model: {resolved_config.openai_image_model}."
        )
    normalized_annotated_reference = (
        _normalize_image_data_url(annotated_reference_image_data_url)
        if annotated_reference_image_data_url
        else None
    )
    normalized_layout_guard_mask = (
        _normalize_image_data_url(layout_guard_mask_image_data_url)
        if layout_guard_mask_image_data_url
        else None
    )
    layout_guard_mask_used = (
        render_mode == "edit" and normalized_layout_guard_mask is not None
    )
    layout_materialization_mode_used = render_mode == "edit" and (
        layout_materialization_mode or layout_guard_mask_used
    )
    annotated_reference_requested = normalized_annotated_reference is not None
    edit_region_guide_used = (
        render_mode == "edit"
        and edit_camera_transfer_reference_used
        and annotated_reference_requested
        and any(operation.bbox_norm is not None for operation in normalized_operations)
    )
    annotated_reference_used = annotated_reference_requested and (
        resolved_config.include_annotated_reference or edit_region_guide_used
    )
    normalized_layout_reference = (
        _normalize_image_data_url(layout_reference_image_data_url)
        if layout_reference_image_data_url
        else None
    )
    layout_reference_enabled = resolved_config.include_layout_2d_reference
    layout_reference_used = (
        render_mode == "generate"
        and not reference_only_camera_transfer_used
        and not annotated_reference_used
        and resolved_config.include_layout_2d_reference
        and normalized_layout_reference is not None
    )
    text_contract_used = resolved_config.text_contract_mode
    strict_layout_used = resolved_config.strict_layout_mode
    configured_text_contract_object_limit = resolved_config.text_contract_object_limit
    text_contract_object_limit = (
        min(
            _MAX_TEXT_CONTRACT_OBJECT_LIMIT,
            max(configured_text_contract_object_limit, len(visible_objects)),
        )
        if text_contract_used
        and strict_layout_used
        and configured_text_contract_object_limit > 0
        else configured_text_contract_object_limit
    )
    simple_prompt_used = (
        render_mode in {"generate", "camera_rerender"}
        and resolved_config.simple_prompt_mode
    )
    effective_operations = [] if simple_prompt_used else normalized_operations
    layout_lock_prompt = (
        _build_generate_layout_lock_prompt(
            scene_reference_mode=resolved_scene_reference_mode,
            snapshot_payload=effective_snapshot_payload,
        )
        if render_mode in {"generate", "camera_rerender"} and not simple_prompt_used
        else ""
    )
    locked_items = _build_locked_items(effective_snapshot_payload)
    layout_lock_mode = _layout_lock_mode_for_render_mode(render_mode)
    normalized_user_prompt = (
        _build_simple_generate_user_prompt(
            user_prompt=user_prompt,
            preset_prompt=preset_prompt,
            preset_reference_images=preset_reference_images,
            snapshot_payload=effective_snapshot_payload,
            include_text_contract=text_contract_used,
            text_contract_object_limit=text_contract_object_limit,
            strict_layout_mode=strict_layout_used,
        )
        if simple_prompt_used
        else _build_final_user_prompt(
            user_prompt=user_prompt,
            render_mode=render_mode,
            preset_prompt=preset_prompt,
            edit_operations=effective_operations,
            visible_objects=visible_objects,
            layout_lock_prompt=layout_lock_prompt,
            layout_reference_used=layout_reference_used,
            annotated_reference_used=annotated_reference_used,
            depth_map_used=normalized_depth_map is not None,
            edge_map_used=normalized_edge_map is not None,
            segmentation_map_used=normalized_segmentation_map is not None,
            scene_reference_mode=resolved_scene_reference_mode,
            locked_geometry_references=normalized_locked_geometry_references,
            locked_geometry_atlases=normalized_locked_geometry_atlases,
            preset_reference_count=len(preset_reference_images),
            camera_transfer_reference_used=edit_camera_transfer_reference_used,
            text_contract_used=text_contract_used,
            include_edit_operation_dsl=resolved_config.include_edit_operation_dsl,
            strict_layout_mode=strict_layout_used,
            layout_guard_mask_used=layout_guard_mask_used,
            layout_materialization_mode_used=layout_materialization_mode_used,
        )
    )
    simple_prompt_image_role_hint_used = simple_prompt_used
    aspect_ratio = _resolve_canvas_aspect_ratio(effective_snapshot_payload)
    reference_images = [
        operation.replacement_image_data_url
        for operation in effective_operations
        if operation.replacement_image_data_url is not None
    ]
    effective_model_name = resolved_config.openai_image_model

    log_image_flow_event(
        "snapshot_image.render_input",
        {
            "trace_id": image_flow_trace_id,
            "render_mode": render_mode,
            "image_provider": "openai",
            "model_name": effective_model_name,
            "image_size": effective_image_size,
            "image_quality": effective_image_quality,
            "streamed": streamed,
            "stream_partial_images": effective_stream_partial_images,
            "max_output_tokens": resolved_config.max_output_tokens,
            "aspect_ratio": aspect_ratio,
            "simple_prompt_used": simple_prompt_used,
            "simple_prompt_config": resolved_config.simple_prompt_mode,
            "text_contract_used": text_contract_used,
            "text_contract_config": resolved_config.text_contract_mode,
            "text_contract_object_limit": text_contract_object_limit,
            "include_edit_operation_dsl": resolved_config.include_edit_operation_dsl,
            "disable_edit_mask": resolved_config.disable_edit_mask,
            "strict_layout_used": strict_layout_used,
            "strict_layout_config": resolved_config.strict_layout_mode,
            "simple_prompt_image_role_hint_used": (simple_prompt_image_role_hint_used),
            "scene_reference_mode": resolved_scene_reference_mode,
            "layout_reference_enabled": layout_reference_enabled,
            "layout_reference_used": layout_reference_used,
            "annotated_reference_requested": annotated_reference_requested,
            "annotated_reference_used": annotated_reference_used,
            "edit_region_guide_used": edit_region_guide_used,
            "layout_guard_mask_used": layout_guard_mask_used,
            "layout_materialization_mode_used": layout_materialization_mode_used,
            "camera_transfer_reference_used": edit_camera_transfer_reference_used,
            "include_annotated_reference_config": (
                resolved_config.include_annotated_reference
            ),
            "geometry_references_requested": geometry_references_requested,
            "geometry_references_used": _geometry_references_used_payload(
                depth_map_image_data_url=normalized_depth_map,
                edge_map_image_data_url=normalized_edge_map,
                segmentation_map_image_data_url=normalized_segmentation_map,
            ),
            "include_geometry_references_config": (
                resolved_config.include_geometry_references
            ),
            "scene_reference_used": scene_reference_used,
            "reference_only_camera_transfer_used": (
                reference_only_camera_transfer_used
            ),
            "layout_lock_mode": layout_lock_mode,
            "locked_item_count": len(locked_items),
            "locked_items": locked_items,
            "visibility_diagnostics": visibility_diagnostics,
            "locked_geometry_references": [
                _serialize_locked_geometry_reference(reference)
                for reference in normalized_locked_geometry_references
            ],
            "locked_geometry_reference_requested_count": (
                locked_geometry_reference_requested_count
            ),
            "locked_geometry_reference_used_count": len(
                normalized_locked_geometry_references
            ),
            "locked_geometry_atlases": [
                _serialize_locked_geometry_atlas(atlas)
                for atlas in normalized_locked_geometry_atlases
            ],
            "locked_geometry_atlas_requested_count": (
                locked_geometry_atlas_requested_count
            ),
            "locked_geometry_atlas_used_count": len(normalized_locked_geometry_atlases),
            "include_locked_geometry_references_config": (
                resolved_config.include_locked_geometry_references
            ),
            **_geometry_lock_metadata(
                locked_items=locked_items,
                atlases=normalized_locked_geometry_atlases,
                references=normalized_locked_geometry_references,
            ),
            "raw_user_prompt": _string(user_prompt) or None,
            "normalized_user_prompt": normalized_user_prompt,
            "preset_prompt": preset_prompt,
            "preset_selection": dict(preset_selection or {}),
            "include_preset_reference_images_config": (
                resolved_config.include_preset_reference_images
            ),
            "preset_reference_images": [
                _serialize_preset_reference_image(reference)
                for reference in preset_reference_images
            ],
            "visible_object_count": len(visible_objects),
            "visible_object_ids": list(visible_objects.keys()),
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in effective_operations
            ],
            "snapshot_payload": dict(snapshot_payload),
            "images": {
                "source": summarize_image_data_url(source_image_data_url),
                "structural": summarize_image_data_url(structural_image_data_url),
                "depth_map": summarize_image_data_url(normalized_depth_map),
                "edge_map": summarize_image_data_url(normalized_edge_map),
                "segmentation_map": summarize_image_data_url(
                    normalized_segmentation_map
                ),
                "annotated_reference": summarize_image_data_url(
                    normalized_annotated_reference if annotated_reference_used else None
                ),
                "layout_guard_mask": summarize_image_data_url(
                    normalized_layout_guard_mask if layout_guard_mask_used else None
                ),
                "layout_reference": summarize_image_data_url(
                    normalized_layout_reference if layout_reference_used else None
                ),
                "scene_reference": summarize_image_data_url(
                    normalized_scene_reference
                    if resolved_scene_reference_mode != "none"
                    else None
                ),
                "replacement_references": [
                    summarize_image_data_url(data_url) for data_url in reference_images
                ],
                "preset_references": [
                    summarize_image_data_url(reference.image_data_url)
                    for reference in preset_reference_images
                ],
                "locked_geometry_references": [
                    summarize_image_data_url(reference.image_data_url)
                    for reference in normalized_locked_geometry_references
                ],
                "locked_geometry_atlases": [
                    summarize_image_data_url(atlas.image_data_url)
                    for atlas in normalized_locked_geometry_atlases
                ],
            },
        },
    )

    active_system_prompt = (
        ""
        if simple_prompt_used and not text_contract_used and not strict_layout_used
        else _LAYOUT_MATERIALIZATION_EDIT_SYSTEM_PROMPT
        if layout_materialization_mode_used
        else _CAMERA_TRANSFER_EDIT_SYSTEM_PROMPT
        if edit_camera_transfer_reference_used
        else _system_prompt_for_render_mode(render_mode)
    )
    try:
        image_result = _generate_image_with_openai(
            trace_id=image_flow_trace_id,
            model_name=resolved_config.openai_image_model,
            aspect_ratio=aspect_ratio,
            system_prompt=active_system_prompt,
            user_prompt=normalized_user_prompt,
            source_image_data_url=source_image_data_url,
            depth_map_image_data_url=normalized_depth_map,
            edge_map_image_data_url=normalized_edge_map,
            segmentation_map_image_data_url=normalized_segmentation_map,
            layout_guard_mask_image_data_url=(
                normalized_layout_guard_mask if layout_guard_mask_used else None
            ),
            annotated_reference_image_data_url=(
                normalized_annotated_reference if annotated_reference_used else None
            ),
            camera_reference_image_data_url=(
                structural_image_data_url
                if edit_camera_transfer_reference_used
                else None
            ),
            scene_reference_image_data_url=(
                normalized_scene_reference if scene_reference_used else None
            ),
            layout_reference_image_data_url=(
                normalized_layout_reference if layout_reference_used else None
            ),
            preset_reference_images=preset_reference_images,
            reference_image_data_urls=reference_images,
            locked_geometry_references=normalized_locked_geometry_references,
            locked_geometry_atlases=normalized_locked_geometry_atlases,
            edit_operations=effective_operations,
            render_mode=render_mode,
            disable_edit_mask=(
                resolved_config.disable_edit_mask
                or (edit_camera_transfer_reference_used and not layout_guard_mask_used)
            ),
            disable_edit_scope_validation=edit_camera_transfer_reference_used,
            locked_items=locked_items,
            openai_api_key=resolved_config.openai_api_key,
            openai_image_size=effective_image_size,
            openai_image_quality=effective_image_quality,
            openai_base_url=_DEFAULT_OPENAI_BASE_URL,
            streamed=streamed,
            stream_partial_images=effective_stream_partial_images,
            stream_event_handler=stream_event_handler,
        )
    except Exception as exc:
        log_image_flow_event(
            "snapshot_image.render_error",
            {
                "trace_id": image_flow_trace_id,
                "render_mode": render_mode,
                "model_name": effective_model_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise

    validation_payload = image_result.get("validation")
    if isinstance(validation_payload, Mapping):
        image_result["validation"] = {
            **dict(validation_payload),
            "visibility_diagnostics": visibility_diagnostics,
        }

    log_image_flow_event(
        "snapshot_image.render_output",
        {
            "trace_id": image_flow_trace_id,
            "render_mode": render_mode,
            "model_name": effective_model_name,
            "image": redact_for_image_log(image_result),
        },
    )

    return {
        "request": {
            "render_mode": render_mode,
            "system_prompt": active_system_prompt,
            "user_prompt": normalized_user_prompt,
            "raw_user_prompt": _string(user_prompt) or None,
            "preset_prompt": preset_prompt,
            "preset_selection": dict(preset_selection or {}),
            "image_quality_requested": image_quality,
            "image_size_requested": image_size,
            "stream_partial_images": effective_stream_partial_images,
            "streamed": streamed,
            "simple_prompt_used": simple_prompt_used,
            "simple_prompt_config": resolved_config.simple_prompt_mode,
            "text_contract_used": text_contract_used,
            "text_contract_config": resolved_config.text_contract_mode,
            "text_contract_object_limit": text_contract_object_limit,
            "include_edit_operation_dsl": resolved_config.include_edit_operation_dsl,
            "disable_edit_mask": resolved_config.disable_edit_mask,
            "strict_layout_used": strict_layout_used,
            "strict_layout_config": resolved_config.strict_layout_mode,
            "simple_prompt_image_role_hint_used": (simple_prompt_image_role_hint_used),
            "aspect_ratio": aspect_ratio,
            "source_image_mime_type": _extract_data_url_mime_type(
                source_image_data_url
            ),
            "structural_image_mime_type": _extract_data_url_mime_type(
                structural_image_data_url
            ),
            "geometry_references_used": _geometry_references_used_payload(
                depth_map_image_data_url=normalized_depth_map,
                edge_map_image_data_url=normalized_edge_map,
                segmentation_map_image_data_url=normalized_segmentation_map,
            ),
            "geometry_references_requested": geometry_references_requested,
            "include_geometry_references_config": (
                resolved_config.include_geometry_references
            ),
            "geometry_reference_image_mime_types": (
                _geometry_reference_mime_types_payload(
                    depth_map_image_data_url=normalized_depth_map,
                    edge_map_image_data_url=normalized_edge_map,
                    segmentation_map_image_data_url=normalized_segmentation_map,
                )
            ),
            "layout_reference_enabled": layout_reference_enabled,
            "layout_reference_used": layout_reference_used,
            "layout_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_layout_reference)
                if layout_reference_used and normalized_layout_reference is not None
                else None
            ),
            "annotated_reference_requested": annotated_reference_requested,
            "annotated_reference_used": annotated_reference_used,
            "edit_region_guide_used": edit_region_guide_used,
            "layout_guard_mask_used": layout_guard_mask_used,
            "layout_materialization_mode_used": layout_materialization_mode_used,
            "camera_transfer_reference_used": edit_camera_transfer_reference_used,
            "include_annotated_reference_config": (
                resolved_config.include_annotated_reference
            ),
            "annotated_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_annotated_reference)
                if annotated_reference_used
                and normalized_annotated_reference is not None
                else None
            ),
            "scene_reference_mode": resolved_scene_reference_mode,
            "scene_reference_used": scene_reference_used,
            "reference_only_camera_transfer_used": (
                reference_only_camera_transfer_used
            ),
            "scene_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_scene_reference)
                if resolved_scene_reference_mode != "none"
                and normalized_scene_reference is not None
                else None
            ),
            "reference_image_count": len(reference_images),
            "preset_reference_image_count": len(preset_reference_images),
            "include_preset_reference_images_config": (
                resolved_config.include_preset_reference_images
            ),
            "preset_reference_images": [
                _serialize_preset_reference_image(reference)
                for reference in preset_reference_images
            ],
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in effective_operations
            ],
            "layout_lock_mode": layout_lock_mode,
            "visibility_diagnostics": visibility_diagnostics,
            "object_memory_used": [
                {
                    "object_id": operation.object_id,
                    "object_name": operation.object_name,
                }
                for operation in effective_operations
                if operation.replacement_image_data_url is not None
                and operation.bbox_norm is None
            ],
            "locked_items": locked_items,
            "locked_geometry_reference_requested_count": (
                locked_geometry_reference_requested_count
            ),
            "locked_geometry_reference_used_count": len(
                normalized_locked_geometry_references
            ),
            "locked_geometry_references": [
                _serialize_locked_geometry_reference(reference)
                for reference in normalized_locked_geometry_references
            ],
            "locked_geometry_atlas_requested_count": (
                locked_geometry_atlas_requested_count
            ),
            "locked_geometry_atlas_used_count": len(normalized_locked_geometry_atlases),
            "include_locked_geometry_references_config": (
                resolved_config.include_locked_geometry_references
            ),
            "locked_geometry_atlases": [
                _serialize_locked_geometry_atlas(atlas)
                for atlas in normalized_locked_geometry_atlases
            ],
            **_geometry_lock_metadata(
                locked_items=locked_items,
                atlases=normalized_locked_geometry_atlases,
                references=normalized_locked_geometry_references,
            ),
        },
        "image": image_result,
        "models": {
            "image_model_name": effective_model_name,
            "max_tokens": resolved_config.max_output_tokens,
        },
        "metadata": {
            "camera": snapshot_payload.get("camera"),
            "visible_objects": list(visible_objects.values()),
            "visible_object_ids": list(visible_objects.keys()),
            "render_mode": render_mode,
            "preset_selection": dict(preset_selection or {}),
            "image_quality_requested": image_quality,
            "image_size_requested": image_size,
            "stream_partial_images": effective_stream_partial_images,
            "streamed": streamed,
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in effective_operations
            ],
            "layout_lock_mode": layout_lock_mode,
            "simple_prompt_used": simple_prompt_used,
            "simple_prompt_config": resolved_config.simple_prompt_mode,
            "text_contract_used": text_contract_used,
            "text_contract_config": resolved_config.text_contract_mode,
            "text_contract_object_limit": text_contract_object_limit,
            "strict_layout_used": strict_layout_used,
            "strict_layout_config": resolved_config.strict_layout_mode,
            "simple_prompt_image_role_hint_used": (simple_prompt_image_role_hint_used),
            "annotated_reference_requested": annotated_reference_requested,
            "annotated_reference_used": annotated_reference_used,
            "edit_region_guide_used": edit_region_guide_used,
            "layout_guard_mask_used": layout_guard_mask_used,
            "layout_materialization_mode_used": layout_materialization_mode_used,
            "camera_transfer_reference_used": edit_camera_transfer_reference_used,
            "include_annotated_reference_config": (
                resolved_config.include_annotated_reference
            ),
            "geometry_references_requested": geometry_references_requested,
            "geometry_references_used": _geometry_references_used_payload(
                depth_map_image_data_url=normalized_depth_map,
                edge_map_image_data_url=normalized_edge_map,
                segmentation_map_image_data_url=normalized_segmentation_map,
            ),
            "include_geometry_references_config": (
                resolved_config.include_geometry_references
            ),
            "preset_reference_image_count": len(preset_reference_images),
            "include_preset_reference_images_config": (
                resolved_config.include_preset_reference_images
            ),
            "visibility_diagnostics": visibility_diagnostics,
            "object_memory_used": [
                {
                    "object_id": operation.object_id,
                    "object_name": operation.object_name,
                }
                for operation in effective_operations
                if operation.replacement_image_data_url is not None
                and operation.bbox_norm is None
            ],
            "locked_items": locked_items,
            "locked_geometry_reference_requested_count": (
                locked_geometry_reference_requested_count
            ),
            "locked_geometry_reference_used_count": len(
                normalized_locked_geometry_references
            ),
            "locked_geometry_references": [
                _serialize_locked_geometry_reference(reference)
                for reference in normalized_locked_geometry_references
            ],
            "locked_geometry_atlas_requested_count": (
                locked_geometry_atlas_requested_count
            ),
            "locked_geometry_atlas_used_count": len(normalized_locked_geometry_atlases),
            "include_locked_geometry_references_config": (
                resolved_config.include_locked_geometry_references
            ),
            "locked_geometry_atlases": [
                _serialize_locked_geometry_atlas(atlas)
                for atlas in normalized_locked_geometry_atlases
            ],
            **_geometry_lock_metadata(
                locked_items=locked_items,
                atlases=normalized_locked_geometry_atlases,
                references=normalized_locked_geometry_references,
            ),
            "validation": image_result.get("validation"),
        },
    }


def render_snapshot_image_from_path(
    snapshot_path: str | Path,
    *,
    user_prompt: str | None = None,
    snapshot_image_data_url: str | None = None,
    depth_map_image_data_url: str | None = None,
    edge_map_image_data_url: str | None = None,
    segmentation_map_image_data_url: str | None = None,
    layout_reference_image_data_url: str | None = None,
    annotated_reference_image_data_url: str | None = None,
    layout_guard_mask_image_data_url: str | None = None,
    layout_materialization_mode: bool = False,
    scene_reference_image_data_url: str | None = None,
    scene_reference_mode: SceneReferenceMode = "none",
    render_mode: RenderMode = "generate",
    preset_selection: Mapping[str, object] | None = None,
    edit_operations: list[SnapshotEditOperation] | None = None,
    locked_geometry_references: list[SnapshotLockedGeometryReference] | None = None,
    locked_geometry_atlases: list[SnapshotLockedGeometryAtlas] | None = None,
    edit_source_image_data_url: str | None = None,
    image_quality: OpenAIImageQuality | str | None = None,
    image_size: OpenAIImageSize | str | None = None,
    streamed: bool = False,
    stream_partial_images: int | None = None,
    stream_event_handler: OpenAIImageStreamEventHandler | None = None,
    config: SnapshotImageRenderConfig | None = None,
    trace_id: str | None = None,
) -> dict[str, object]:
    path = Path(snapshot_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Snapshot file must contain a JSON object.")

    image_data_url = snapshot_image_data_url
    if image_data_url is None:
        sibling_image = _find_snapshot_image_path(path)
        if sibling_image is None:
            raise FileNotFoundError(
                "No sibling snapshot image found. Provide snapshot_image_data_url "
                "or save a PNG/JPG beside the snapshot JSON."
            )
        image_data_url = _encode_file_as_data_url(sibling_image)

    return render_snapshot_image(
        payload,
        snapshot_image_data_url=image_data_url,
        depth_map_image_data_url=depth_map_image_data_url,
        edge_map_image_data_url=edge_map_image_data_url,
        segmentation_map_image_data_url=segmentation_map_image_data_url,
        user_prompt=user_prompt,
        layout_reference_image_data_url=layout_reference_image_data_url,
        annotated_reference_image_data_url=annotated_reference_image_data_url,
        layout_guard_mask_image_data_url=layout_guard_mask_image_data_url,
        layout_materialization_mode=layout_materialization_mode,
        scene_reference_image_data_url=scene_reference_image_data_url,
        scene_reference_mode=scene_reference_mode,
        render_mode=render_mode,
        preset_selection=preset_selection,
        edit_operations=edit_operations,
        locked_geometry_references=locked_geometry_references,
        locked_geometry_atlases=locked_geometry_atlases,
        edit_source_image_data_url=edit_source_image_data_url,
        image_quality=image_quality,
        image_size=image_size,
        streamed=streamed,
        stream_partial_images=stream_partial_images,
        stream_event_handler=stream_event_handler,
        config=config,
        trace_id=trace_id,
    )


def _data_url_to_bytes(data_url: str) -> bytes:
    _, _, image_base64 = data_url.partition(",")
    return base64.b64decode(image_base64)


def _normalize_openai_image_size(value: str | None) -> str:
    normalized = _string(value, fallback=_DEFAULT_OPENAI_IMAGE_SIZE).lower()
    if normalized in _SUPPORTED_OPENAI_IMAGE_SIZES:
        return normalized
    return _DEFAULT_OPENAI_IMAGE_SIZE


def _normalize_openai_image_quality(value: str | None) -> str:
    normalized = _string(value, fallback=_DEFAULT_OPENAI_IMAGE_QUALITY).lower()
    if normalized in _SUPPORTED_OPENAI_IMAGE_QUALITIES:
        return normalized
    return _DEFAULT_OPENAI_IMAGE_QUALITY


def _normalize_openai_stream_partial_images(value: int | str | None) -> int:
    try:
        parsed = (
            int(value)
            if value is not None
            else _DEFAULT_OPENAI_IMAGE_STREAM_PARTIAL_IMAGES
        )
    except (TypeError, ValueError):
        parsed = _DEFAULT_OPENAI_IMAGE_STREAM_PARTIAL_IMAGES
    return max(0, min(3, parsed))


def _normalize_text_contract_object_limit(value: int | str | None) -> int:
    try:
        parsed = (
            int(value) if value is not None else _DEFAULT_TEXT_CONTRACT_OBJECT_LIMIT
        )
    except (TypeError, ValueError):
        parsed = _DEFAULT_TEXT_CONTRACT_OBJECT_LIMIT
    return max(0, min(_MAX_TEXT_CONTRACT_OBJECT_LIMIT, parsed))


def _geometry_references_used_payload(
    *,
    depth_map_image_data_url: str | None,
    edge_map_image_data_url: str | None,
    segmentation_map_image_data_url: str | None,
) -> dict[str, bool]:
    return {
        "depth_map": depth_map_image_data_url is not None,
        "edge_map": edge_map_image_data_url is not None,
        "segmentation_map": segmentation_map_image_data_url is not None,
    }


def _geometry_reference_mime_types_payload(
    *,
    depth_map_image_data_url: str | None,
    edge_map_image_data_url: str | None,
    segmentation_map_image_data_url: str | None,
) -> dict[str, str | None]:
    return {
        "depth_map": (
            _extract_data_url_mime_type(depth_map_image_data_url)
            if depth_map_image_data_url is not None
            else None
        ),
        "edge_map": (
            _extract_data_url_mime_type(edge_map_image_data_url)
            if edge_map_image_data_url is not None
            else None
        ),
        "segmentation_map": (
            _extract_data_url_mime_type(segmentation_map_image_data_url)
            if segmentation_map_image_data_url is not None
            else None
        ),
    }


def _data_url_to_openai_image_file(
    data_url: str,
    *,
    filename_stem: str,
    force_png: bool = False,
) -> tuple[str, bytes, str]:
    if force_png:
        return (
            f"{filename_stem}.png",
            _data_url_to_png_bytes(data_url),
            "image/png",
        )
    mime_type = _extract_data_url_mime_type(data_url)
    extension = _image_extension_for_mime_type(mime_type)
    return (
        f"{filename_stem}.{extension}",
        _data_url_to_bytes(data_url),
        mime_type,
    )


def _data_url_to_png_bytes(data_url: str) -> bytes:
    image = Image.open(BytesIO(_data_url_to_bytes(data_url))).convert("RGBA")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _build_openai_edit_mask_file(
    source_image_data_url: str,
    *,
    edit_operations: list[SnapshotEditOperation],
) -> tuple[str, bytes, str] | None:
    region_operations = [
        operation for operation in edit_operations if operation.bbox_norm is not None
    ]
    if not region_operations:
        return None

    source_image = Image.open(BytesIO(_data_url_to_bytes(source_image_data_url)))
    width, height = source_image.size
    if width <= 0 or height <= 0:
        return None

    mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    has_edit_region = False
    for operation in region_operations:
        bbox = operation.bbox_norm
        if bbox is None:
            continue
        left = int(max(0.0, min(1.0, float(bbox.get("x", 0.0)))) * width)
        top = int(max(0.0, min(1.0, float(bbox.get("y", 0.0)))) * height)
        right = int(
            max(
                0.0, min(1.0, float(bbox.get("x", 0.0)) + float(bbox.get("width", 0.0)))
            )
            * width
        )
        bottom = int(
            max(
                0.0,
                min(1.0, float(bbox.get("y", 0.0)) + float(bbox.get("height", 0.0))),
            )
            * height
        )
        if right <= left or bottom <= top:
            continue
        mask.paste((0, 0, 0, 0), box=(left, top, right, bottom))
        has_edit_region = True

    if not has_edit_region:
        return None

    output = BytesIO()
    mask.save(output, format="PNG")
    return ("mask.png", output.getvalue(), "image/png")


def _build_openai_layout_guard_mask_file(
    source_image_data_url: str,
    *,
    layout_guard_mask_image_data_url: str,
) -> tuple[str, bytes, str] | None:
    source_image = Image.open(BytesIO(_data_url_to_bytes(source_image_data_url)))
    mask_image = Image.open(
        BytesIO(_data_url_to_bytes(layout_guard_mask_image_data_url))
    ).convert("RGBA")
    if source_image.size != mask_image.size:
        raise ValueError(
            "Layout guard mask size must match source image size: "
            f"source={source_image.size}, mask={mask_image.size}"
        )
    output = BytesIO()
    mask_image.save(output, format="PNG")
    return ("layout_guard_mask.png", output.getvalue(), "image/png")


def _image_extension_for_mime_type(mime_type: str) -> str:
    normalized = mime_type.lower()
    if normalized == "image/jpeg":
        return "jpg"
    if normalized == "image/webp":
        return "webp"
    if normalized == "image/gif":
        return "gif"
    return "png"


def _generate_image_with_openai(
    *,
    trace_id: str,
    model_name: str,
    aspect_ratio: str,
    system_prompt: str,
    user_prompt: str,
    source_image_data_url: str,
    depth_map_image_data_url: str | None,
    edge_map_image_data_url: str | None,
    segmentation_map_image_data_url: str | None,
    layout_guard_mask_image_data_url: str | None,
    annotated_reference_image_data_url: str | None,
    camera_reference_image_data_url: str | None,
    scene_reference_image_data_url: str | None,
    layout_reference_image_data_url: str | None,
    preset_reference_images: list[SnapshotPresetReferenceImage],
    reference_image_data_urls: list[str],
    locked_geometry_references: list[SnapshotLockedGeometryReference],
    locked_geometry_atlases: list[SnapshotLockedGeometryAtlas],
    edit_operations: list[SnapshotEditOperation],
    render_mode: RenderMode,
    disable_edit_mask: bool,
    disable_edit_scope_validation: bool,
    locked_items: list[dict[str, object]],
    openai_api_key: str | None,
    openai_image_size: str,
    openai_image_quality: str,
    openai_base_url: str,
    streamed: bool = False,
    stream_partial_images: int = 0,
    stream_event_handler: OpenAIImageStreamEventHandler | None = None,
) -> dict[str, object]:
    if not openai_api_key:
        raise ValueError(
            "Missing OpenAI image API key. Set OPENAI_IMAGE_API_KEY or configure "
            "services.openai in app-config.yaml before using OpenAI image rendering."
        )

    size = _normalize_openai_image_size(openai_image_size)
    quality = _normalize_openai_image_quality(openai_image_quality)
    combined_prompt = "\n\n".join(
        part for part in (_string(system_prompt), _string(user_prompt)) if part
    )
    geometry_references_used = (
        any(
            value is not None
            for value in (
                depth_map_image_data_url,
                edge_map_image_data_url,
                segmentation_map_image_data_url,
            )
        )
        or bool(locked_geometry_references)
        or bool(locked_geometry_atlases)
    )
    mask_file = (
        None
        if disable_edit_mask
        else _build_openai_layout_guard_mask_file(
            source_image_data_url,
            layout_guard_mask_image_data_url=layout_guard_mask_image_data_url,
        )
        if layout_guard_mask_image_data_url is not None
        else _build_openai_edit_mask_file(
            source_image_data_url,
            edit_operations=edit_operations,
        )
    )

    image_files: list[tuple[str, tuple[str, bytes, str]]] = []
    image_slots: list[dict[str, object]] = []

    def append_image_slot(
        *,
        role: str,
        data_url: str,
        filename_stem: str,
        force_png: bool = False,
    ) -> None:
        image_slots.append(
            {
                "image_number": len(image_slots) + 1,
                "role": role,
                "summary": summarize_image_data_url(data_url),
            }
        )
        image_files.append(
            (
                "image[]",
                _data_url_to_openai_image_file(
                    data_url,
                    filename_stem=filename_stem,
                    force_png=force_png,
                ),
            )
        )

    append_image_slot(
        role="current_render_source" if render_mode == "edit" else "clean_3d_source",
        data_url=source_image_data_url,
        filename_stem="source",
        force_png=mask_file is not None,
    )
    if camera_reference_image_data_url is not None:
        append_image_slot(
            role="target_3d_camera_reference",
            data_url=camera_reference_image_data_url,
            filename_stem="camera_reference",
        )
    if depth_map_image_data_url is not None:
        append_image_slot(
            role="depth_map_geometry_reference",
            data_url=depth_map_image_data_url,
            filename_stem="depth_map",
        )
    if edge_map_image_data_url is not None:
        append_image_slot(
            role="edge_map_geometry_reference",
            data_url=edge_map_image_data_url,
            filename_stem="edge_map",
        )
    if segmentation_map_image_data_url is not None:
        append_image_slot(
            role="segmentation_map_geometry_reference",
            data_url=segmentation_map_image_data_url,
            filename_stem="segmentation_map",
        )
    if scene_reference_image_data_url is not None:
        append_image_slot(
            role="scene_reference_appearance_only",
            data_url=scene_reference_image_data_url,
            filename_stem="scene_reference",
        )
    if layout_reference_image_data_url is not None:
        append_image_slot(
            role="top_down_layout_reference",
            data_url=layout_reference_image_data_url,
            filename_stem="layout_reference",
        )
    if annotated_reference_image_data_url is not None:
        append_image_slot(
            role="structural_identity_guide",
            data_url=annotated_reference_image_data_url,
            filename_stem="edit_guide",
        )
    for atlas in locked_geometry_atlases:
        if atlas.image_data_url is None:
            continue
        append_image_slot(
            role=f"object_geometry_atlas_{atlas.atlas_index}",
            data_url=atlas.image_data_url,
            filename_stem=f"geometry_atlas_{atlas.atlas_index}",
        )
    for i, reference in enumerate(locked_geometry_references):
        if reference.image_data_url is None:
            continue
        append_image_slot(
            role=f"object_geometry_lock_reference_{i + 1}",
            data_url=reference.image_data_url,
            filename_stem=f"geometry_lock_{i + 1}",
        )
    for i, reference in enumerate(preset_reference_images):
        append_image_slot(
            role=f"preset_{reference.group}_reference",
            data_url=reference.image_data_url,
            filename_stem=f"preset_{reference.group}_{i + 1}",
        )
    for i, ref_url in enumerate(reference_image_data_urls):
        append_image_slot(
            role=f"object_appearance_reference_{i + 1}",
            data_url=ref_url,
            filename_stem=f"reference_{i + 1}",
        )
    if mask_file is not None:
        image_files.append(("mask", mask_file))

    endpoint = f"{openai_base_url.rstrip('/')}/images/edits"
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    form_data = {
        "model": model_name,
        "prompt": combined_prompt,
        "n": "1",
        "size": size,
        "quality": quality,
    }

    image_result = _post_openai_images_edit(
        trace_id=trace_id,
        model_name=model_name,
        endpoint=endpoint,
        headers=headers,
        form_data=form_data,
        image_files=image_files,
        image_slots=image_slots,
        mask_used=mask_file is not None,
        aspect_ratio=aspect_ratio,
        size=size,
        quality=quality,
        attempt_name="geometry_locked" if geometry_references_used else "standard",
        combined_prompt=combined_prompt,
        streamed=streamed,
        stream_partial_images=stream_partial_images,
        stream_event_handler=stream_event_handler,
    )
    validation = _validate_generated_geometry_against_reference(
        generated_image_data_url=str(image_result["data_url"]),
        edge_map_image_data_url=edge_map_image_data_url,
        expected_aspect_ratio=aspect_ratio,
        locked_items=locked_items,
        locked_geometry_references=locked_geometry_references,
        locked_geometry_atlases=locked_geometry_atlases,
    )
    edit_scope_validation = (
        None
        if disable_edit_scope_validation
        else _validate_edit_change_scope(
            source_image_data_url=source_image_data_url,
            generated_image_data_url=str(image_result["data_url"]),
            edit_operations=edit_operations,
            render_mode=render_mode,
        )
    )
    if edit_scope_validation is not None:
        validation = {**validation, "edit_scope": edit_scope_validation}
    image_result["validation"] = validation
    image_result["retry_count"] = 0
    image_result["request_strategy"] = (
        "openai_edit_geometry_locked" if geometry_references_used else "openai_edit"
    )
    image_result["generation_config_applied"] = True
    image_result["layout_guard_mask_used"] = (
        layout_guard_mask_image_data_url is not None and mask_file is not None
    )

    if geometry_references_used and _geometry_validation_should_retry(validation):
        retry_prompt = _build_geometry_retry_prompt(combined_prompt, validation)
        log_image_flow_event(
            "openai.images_edit.geometry_retry",
            {
                "trace_id": trace_id,
                "model_name": model_name,
                "validation": validation,
                "retry_reason": validation.get("reason"),
            },
        )
        retry_result = _post_openai_images_edit(
            trace_id=trace_id,
            model_name=model_name,
            endpoint=endpoint,
            headers=headers,
            form_data={**form_data, "prompt": retry_prompt},
            image_files=image_files,
            image_slots=image_slots,
            mask_used=mask_file is not None,
            aspect_ratio=aspect_ratio,
            size=size,
            quality=quality,
            attempt_name="geometry_retry",
            combined_prompt=retry_prompt,
            streamed=streamed,
            stream_partial_images=stream_partial_images,
            stream_event_handler=stream_event_handler,
        )
        retry_validation = _validate_generated_geometry_against_reference(
            generated_image_data_url=str(retry_result["data_url"]),
            edge_map_image_data_url=edge_map_image_data_url,
            expected_aspect_ratio=aspect_ratio,
            locked_items=locked_items,
            locked_geometry_references=locked_geometry_references,
            locked_geometry_atlases=locked_geometry_atlases,
        )
        if retry_validation.get("status") == "failed":
            retry_validation = {
                **retry_validation,
                "status": "failed_best_effort",
            }
        retry_result["validation"] = retry_validation
        retry_result["retry_count"] = 1
        retry_result["request_strategy"] = "openai_edit_geometry_retry"
        retry_result["generation_config_applied"] = True
        return retry_result

    return image_result


def _post_openai_images_edit(
    *,
    trace_id: str,
    model_name: str,
    endpoint: str,
    headers: Mapping[str, str],
    form_data: Mapping[str, str],
    image_files: list[tuple[str, tuple[str, bytes, str]]],
    image_slots: list[dict[str, object]],
    mask_used: bool,
    aspect_ratio: str,
    size: str,
    quality: str,
    attempt_name: str,
    combined_prompt: str,
    streamed: bool = False,
    stream_partial_images: int = 0,
    stream_event_handler: OpenAIImageStreamEventHandler | None = None,
) -> dict[str, object]:
    if streamed:
        return _post_openai_images_edit_stream(
            trace_id=trace_id,
            model_name=model_name,
            endpoint=endpoint,
            headers=headers,
            form_data=form_data,
            image_files=image_files,
            image_slots=image_slots,
            mask_used=mask_used,
            aspect_ratio=aspect_ratio,
            size=size,
            quality=quality,
            attempt_name=attempt_name,
            combined_prompt=combined_prompt,
            stream_partial_images=stream_partial_images,
            stream_event_handler=stream_event_handler,
        )

    log_image_flow_event(
        "openai.images_edit.attempt_input",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "attempt": attempt_name,
            "endpoint": endpoint,
            "size": size,
            "quality": quality,
            "aspect_ratio": aspect_ratio,
            "image_count": sum(
                1 for field_name, _ in image_files if field_name == "image[]"
            ),
            "image_slots": image_slots,
            "mask_used": mask_used,
            "combined_prompt": combined_prompt,
        },
    )

    started_at = perf_counter()
    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            endpoint,
            headers=headers,
            data=form_data,
            files=image_files,
        )
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        log_image_flow_event(
            "openai.images_edit.http_error",
            {
                "trace_id": trace_id,
                "model_name": model_name,
                "attempt": attempt_name,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "response_body": body[:2000],
            },
        )
        detail = f"{exc}. OpenAI response: {body}" if body else str(exc)
        raise ValueError(detail) from exc

    parsed = response.json()
    data_items = parsed.get("data") or []
    if not data_items:
        raise ValueError(
            f"OpenAI image response contained no image data. Response: {str(parsed)[:400]}"
        )

    b64_data = data_items[0].get("b64_json") or ""
    if not b64_data:
        raise ValueError(
            f"OpenAI image response missing b64_json. Response: {str(parsed)[:400]}"
        )

    image_bytes = base64.b64decode(b64_data)
    mime_type = _guess_image_mime_type(image_bytes)

    log_image_flow_event(
        "openai.images_edit.image_output",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "attempt": attempt_name,
            "elapsed_ms": elapsed_ms,
            "mime_type": mime_type,
            "byte_length": len(image_bytes),
            "size": size,
            "quality": quality,
        },
    )

    result: dict[str, object] = {
        "mime_type": mime_type,
        "image_base64": b64_data,
        "data_url": f"data:{mime_type};base64,{b64_data}",
        "size": size,
        "quality": quality,
        "aspect_ratio": aspect_ratio,
        "elapsed_ms": elapsed_ms,
        "raw_response": {k: v for k, v in parsed.items() if k != "data"},
        "partial_image_count": 0,
    }
    if isinstance(parsed.get("usage"), Mapping):
        result["usage"] = dict(parsed["usage"])
    return result


def _post_openai_images_edit_stream(
    *,
    trace_id: str,
    model_name: str,
    endpoint: str,
    headers: Mapping[str, str],
    form_data: Mapping[str, str],
    image_files: list[tuple[str, tuple[str, bytes, str]]],
    image_slots: list[dict[str, object]],
    mask_used: bool,
    aspect_ratio: str,
    size: str,
    quality: str,
    attempt_name: str,
    combined_prompt: str,
    stream_partial_images: int,
    stream_event_handler: OpenAIImageStreamEventHandler | None,
) -> dict[str, object]:
    normalized_partial_images = _normalize_openai_stream_partial_images(
        stream_partial_images
    )
    stream_form_data = {
        **dict(form_data),
        "stream": "true",
        "partial_images": str(normalized_partial_images),
    }
    log_image_flow_event(
        "openai.images_edit.stream_attempt_input",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "attempt": attempt_name,
            "endpoint": endpoint,
            "size": size,
            "quality": quality,
            "stream_partial_images": normalized_partial_images,
            "aspect_ratio": aspect_ratio,
            "image_count": sum(
                1 for field_name, _ in image_files if field_name == "image[]"
            ),
            "image_slots": image_slots,
            "mask_used": mask_used,
            "combined_prompt": combined_prompt,
        },
    )
    _emit_openai_stream_event(
        stream_event_handler,
        "status",
        {
            "message": "OpenAI image stream started.",
            "attempt": attempt_name,
        },
    )

    started_at = perf_counter()
    final_event: Mapping[str, object] | None = None
    final_b64_data = ""
    final_usage: Mapping[str, object] | None = None
    partial_image_count = 0
    with httpx.Client(timeout=600.0) as client:
        with client.stream(
            "POST",
            endpoint,
            headers=headers,
            data=stream_form_data,
            files=image_files,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = response.text.strip()
                log_image_flow_event(
                    "openai.images_edit.stream_http_error",
                    {
                        "trace_id": trace_id,
                        "model_name": model_name,
                        "attempt": attempt_name,
                        "status_code": response.status_code,
                        "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                        "response_body": body[:2000],
                    },
                )
                detail = f"{exc}. OpenAI response: {body}" if body else str(exc)
                raise ValueError(detail) from exc

            for event in _iter_openai_sse_payloads(response):
                event_type = _string(event.get("type"))
                if isinstance(event.get("usage"), Mapping):
                    final_usage = event["usage"]

                if event_type == "partial_image" or event_type.endswith(
                    ".partial_image"
                ):
                    partial_b64 = _openai_stream_event_b64(event)
                    if not partial_b64:
                        continue
                    partial_image_count += 1
                    partial_bytes = base64.b64decode(partial_b64)
                    partial_mime_type = _guess_image_mime_type(partial_bytes)
                    _emit_openai_stream_event(
                        stream_event_handler,
                        "partial_image",
                        {
                            "partial_image_index": _openai_stream_partial_index(
                                event,
                                fallback=partial_image_count,
                            ),
                            "image_data_url": (
                                f"data:{partial_mime_type};base64,{partial_b64}"
                            ),
                            "size": size,
                            "quality": quality,
                            "attempt": attempt_name,
                        },
                    )
                    continue

                if event_type == "completed" or event_type.endswith(".completed"):
                    final_event = event
                    event_b64 = _openai_stream_event_b64(event)
                    if event_b64:
                        final_b64_data = event_b64
                    continue

                if event_type in {"error", "failed"} or event_type.endswith(".failed"):
                    message = _string(
                        event.get("message"), fallback="OpenAI image stream failed."
                    )
                    raise ValueError(message)

    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    if not final_b64_data:
        raise ValueError("OpenAI image stream completed without final b64_json data.")

    image_bytes = base64.b64decode(final_b64_data)
    mime_type = _guess_image_mime_type(image_bytes)

    log_image_flow_event(
        "openai.images_edit.stream_image_output",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "attempt": attempt_name,
            "elapsed_ms": elapsed_ms,
            "mime_type": mime_type,
            "byte_length": len(image_bytes),
            "size": size,
            "quality": quality,
            "partial_image_count": partial_image_count,
        },
    )

    result: dict[str, object] = {
        "mime_type": mime_type,
        "image_base64": final_b64_data,
        "data_url": f"data:{mime_type};base64,{final_b64_data}",
        "size": size,
        "quality": quality,
        "aspect_ratio": aspect_ratio,
        "elapsed_ms": elapsed_ms,
        "raw_response": _redact_openai_stream_event(final_event),
        "partial_image_count": partial_image_count,
    }
    if final_usage is not None:
        result["usage"] = dict(final_usage)
    return result


def _emit_openai_stream_event(
    stream_event_handler: OpenAIImageStreamEventHandler | None,
    event_type: str,
    payload: Mapping[str, object],
) -> None:
    if stream_event_handler is None:
        return
    stream_event_handler({"type": event_type, **dict(payload)})


def _iter_openai_sse_payloads(response: httpx.Response) -> Iterator[dict[str, object]]:
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            parsed = _parse_openai_sse_event(raw_event)
            if parsed is not None:
                yield parsed
    parsed = _parse_openai_sse_event(buffer)
    if parsed is not None:
        yield parsed


def _parse_openai_sse_event(raw_event: str) -> dict[str, object] | None:
    data_lines = []
    for raw_line in raw_event.splitlines():
        line = raw_line.strip()
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    raw_data = "\n".join(data_lines).strip()
    if not raw_data or raw_data == "[DONE]":
        return None
    try:
        parsed = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise ValueError("Unable to parse OpenAI image stream event.") from exc
    if not isinstance(parsed, dict):
        return None
    return parsed


def _openai_stream_event_b64(event: Mapping[str, object]) -> str:
    direct = event.get("b64_json")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    data = event.get("data")
    if isinstance(data, Mapping):
        nested = data.get("b64_json")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, Mapping):
                continue
            nested = item.get("b64_json")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _openai_stream_partial_index(
    event: Mapping[str, object],
    *,
    fallback: int,
) -> int:
    raw_index = event.get("partial_image_index")
    if raw_index is None:
        data = event.get("data")
        if isinstance(data, Mapping):
            raw_index = data.get("partial_image_index")
    try:
        parsed = int(raw_index) if raw_index is not None else fallback
    except (TypeError, ValueError):
        parsed = fallback
    return max(1, parsed)


def _redact_openai_stream_event(
    event: Mapping[str, object] | None,
) -> dict[str, object]:
    if event is None:
        return {}
    return {
        key: value for key, value in event.items() if key not in {"b64_json", "data"}
    }


def _validate_generated_geometry_against_reference(
    *,
    generated_image_data_url: str,
    edge_map_image_data_url: str | None,
    expected_aspect_ratio: str,
    locked_items: list[dict[str, object]] | None = None,
    locked_geometry_references: list[SnapshotLockedGeometryReference] | None = None,
    locked_geometry_atlases: list[SnapshotLockedGeometryAtlas] | None = None,
) -> dict[str, object]:
    if edge_map_image_data_url is None:
        return {
            "status": "skipped",
            "reason": "edge_map_reference_missing",
            "locked_item_count": len(locked_items or []),
        }

    try:
        generated_image = Image.open(
            BytesIO(_data_url_to_bytes(generated_image_data_url))
        ).convert("RGB")
        edge_map = Image.open(
            BytesIO(_data_url_to_bytes(edge_map_image_data_url))
        ).convert("L")
    except (binascii.Error, OSError, UnidentifiedImageError) as exc:
        return {
            "status": "skipped",
            "reason": "geometry_validation_image_decode_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "locked_item_count": len(locked_items or []),
        }

    generated_width, generated_height = generated_image.size
    if generated_width <= 2 or generated_height <= 2:
        return {
            "status": "skipped",
            "reason": "generated_image_too_small_for_geometry_validation",
            "generated_size": [generated_width, generated_height],
            "locked_item_count": len(locked_items or []),
        }

    if edge_map.size != generated_image.size:
        edge_map = edge_map.resize(generated_image.size, Image.Resampling.BILINEAR)

    reference_mask = _reference_edge_mask(edge_map)
    reference_count = sum(1 for value in reference_mask if value)
    total_pixels = max(1, len(reference_mask))
    reference_coverage = reference_count / total_pixels
    if reference_coverage < 0.001:
        return {
            "status": "skipped",
            "reason": "edge_map_reference_too_sparse",
            "reference_edge_coverage": round(reference_coverage, 6),
            "locked_item_count": len(locked_items or []),
        }

    generated_edge_image = generated_image.convert("L").filter(ImageFilter.FIND_EDGES)
    generated_mask_image = _threshold_edge_image(generated_edge_image)
    dilated_generated = generated_mask_image.filter(ImageFilter.MaxFilter(7))
    dilated_generated_data = list(dilated_generated.getdata())
    matched_reference_count = sum(
        1
        for reference_value, generated_value in zip(
            reference_mask, dilated_generated_data, strict=False
        )
        if reference_value and generated_value > 0
    )
    edge_alignment_score = matched_reference_count / max(1, reference_count)
    generated_aspect = generated_width / generated_height
    target_aspect = _aspect_ratio_value(expected_aspect_ratio)
    aspect_delta = (
        abs(generated_aspect - target_aspect) / target_aspect
        if target_aspect > 0
        else 0.0
    )
    edge_alignment_threshold = 0.095
    status = "passed" if edge_alignment_score >= edge_alignment_threshold else "failed"
    reason = (
        "camera_or_edge_alignment_score_below_threshold"
        if status == "failed"
        else "geometry_reference_alignment_acceptable"
    )
    locked_item_visibility = _validate_locked_item_visibility(
        generated_image,
        locked_items or [],
    )
    locked_item_shape = _validate_locked_item_shape(
        generated_image=generated_image,
        locked_items=locked_items or [],
        locked_geometry_references=locked_geometry_references or [],
        locked_geometry_atlases=locked_geometry_atlases or [],
    )
    if locked_item_shape.get("status") == "failed":
        status = "failed"
        reason = "locked_item_shape_drift"

    return {
        "status": status,
        "reason": reason,
        "edge_alignment_score": round(edge_alignment_score, 4),
        "edge_alignment_threshold": edge_alignment_threshold,
        "reference_edge_coverage": round(reference_coverage, 6),
        "generated_size": [generated_width, generated_height],
        "expected_aspect_ratio": expected_aspect_ratio,
        "generated_aspect_ratio": round(generated_aspect, 4),
        "aspect_delta": round(aspect_delta, 4),
        "locked_item_count": len(locked_items or []),
        "locked_item_visibility": locked_item_visibility,
        "locked_item_shape": locked_item_shape,
    }


def _geometry_validation_should_retry(validation: Mapping[str, object]) -> bool:
    return validation.get("status") == "failed"


def _build_geometry_retry_prompt(
    combined_prompt: str,
    validation: Mapping[str, object],
) -> str:
    return (
        combined_prompt + "\n\nGeometry retry instruction:\n"
        "The previous output drifted away from the supplied geometry references. "
        "Regenerate the same scene with stronger adherence to the clean 3D source, "
        "depth map, edge map, segmentation map, and structural guide. Keep wall "
        "corners, window/opening positions, object silhouettes, screen bboxes, "
        "depth order, and camera framing aligned with those references. Do not "
        "zoom out, dolly backward, widen the field of view, shrink the furniture, "
        "or reveal more room than the clean 3D source image. Treat style, "
        "lighting, and scenery presets as finish-only changes.\n"
        "Validation reason: "
        + _string(validation.get("reason"), fallback="geometry_alignment_failed")
    )


def _validate_locked_item_visibility(
    generated_image: Image.Image,
    locked_items: list[dict[str, object]],
) -> dict[str, object]:
    checked: list[dict[str, object]] = []
    width, height = generated_image.size
    if width <= 0 or height <= 0:
        return {"status": "skipped", "reason": "invalid_generated_size"}

    for item in locked_items[:60]:
        bbox = item.get("screen_bbox_px")
        if not isinstance(bbox, Mapping):
            continue
        source_width = max(1.0, _float(item.get("canvas_width_px"), width))
        source_height = max(1.0, _float(item.get("canvas_height_px"), height))
        left = int(max(0.0, _float(bbox.get("minX"), 0.0)) / source_width * width)
        top = int(max(0.0, _float(bbox.get("minY"), 0.0)) / source_height * height)
        right = int(max(left + 1, _float(bbox.get("maxX"), 0.0) / source_width * width))
        bottom = int(
            max(top + 1, _float(bbox.get("maxY"), 0.0) / source_height * height)
        )
        left = max(0, min(width - 1, left))
        top = max(0, min(height - 1, top))
        right = max(left + 1, min(width, right))
        bottom = max(top + 1, min(height, bottom))
        crop = generated_image.crop((left, top, right, bottom)).convert("L")
        edge_density = _edge_density(crop)
        checked.append(
            {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "type": item.get("type"),
                "edge_density": round(edge_density, 5),
                "status": "weak" if edge_density < 0.002 else "present",
            }
        )

    weak_items = [item for item in checked if item.get("status") == "weak"]
    return {
        "status": "warning" if weak_items else "passed",
        "checked_count": len(checked),
        "weak_count": len(weak_items),
        "weak_items": weak_items[:12],
    }


def _validate_locked_item_shape(
    *,
    generated_image: Image.Image,
    locked_items: list[dict[str, object]],
    locked_geometry_references: list[SnapshotLockedGeometryReference],
    locked_geometry_atlases: list[SnapshotLockedGeometryAtlas],
) -> dict[str, object]:
    reference_images = _locked_geometry_reference_images_by_object(
        locked_geometry_references=locked_geometry_references,
        locked_geometry_atlases=locked_geometry_atlases,
    )
    if not reference_images:
        return {"status": "skipped", "reason": "no_locked_geometry_references"}
    locked_by_id = {
        _string(item.get("id")): item
        for item in locked_items
        if _string(item.get("id"))
    }
    checked: list[dict[str, object]] = []
    for object_id, reference_image in list(reference_images.items())[:40]:
        locked_item = locked_by_id.get(object_id)
        if locked_item is None:
            continue
        bbox = locked_item.get("screen_bbox_px")
        if not isinstance(bbox, Mapping):
            continue
        generated_crop = _crop_generated_image_by_source_bbox(
            generated_image,
            bbox=bbox,
            source_width=_float(
                locked_item.get("canvas_width_px"),
                generated_image.width,
            ),
            source_height=_float(
                locked_item.get("canvas_height_px"),
                generated_image.height,
            ),
        )
        if generated_crop is None:
            continue
        shape_score = _edge_shape_alignment_score(
            reference_image,
            generated_crop.convert("L"),
        )
        object_type = _string(locked_item.get("type"))
        threshold = 0.075
        checked.append(
            {
                "id": object_id,
                "type": object_type,
                "shape_alignment_score": round(shape_score, 4),
                "shape_alignment_threshold": threshold,
                "status": "failed" if shape_score < threshold else "passed",
            }
        )

    failed_items = [item for item in checked if item.get("status") == "failed"]
    if not checked:
        return {
            "status": "skipped",
            "reason": "no_locked_geometry_reference_with_bbox",
            "checked_count": 0,
        }
    return {
        "status": "failed" if failed_items else "passed",
        "checked_count": len(checked),
        "failed_count": len(failed_items),
        "failed_items": failed_items[:12],
    }


def _locked_geometry_reference_images_by_object(
    *,
    locked_geometry_references: list[SnapshotLockedGeometryReference],
    locked_geometry_atlases: list[SnapshotLockedGeometryAtlas],
) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    for atlas in locked_geometry_atlases:
        if atlas.image_data_url is None:
            continue
        try:
            atlas_image = Image.open(
                BytesIO(_data_url_to_bytes(atlas.image_data_url))
            ).convert("L")
        except (binascii.Error, OSError, UnidentifiedImageError):
            continue
        for item in atlas.objects or []:
            object_id = _string(item.object_id)
            if not object_id or object_id in images:
                continue
            crop = _crop_image_by_norm_bbox(
                atlas_image,
                _mapping(item.cell_bbox_norm),
            )
            if crop is not None:
                images[object_id] = crop

    for reference in locked_geometry_references:
        object_id = _string(reference.object_id)
        if not object_id or object_id in images or reference.image_data_url is None:
            continue
        try:
            images[object_id] = Image.open(
                BytesIO(_data_url_to_bytes(reference.image_data_url))
            ).convert("L")
        except (binascii.Error, OSError, UnidentifiedImageError):
            continue
    return images


def _crop_image_by_norm_bbox(
    image: Image.Image,
    bbox_norm: Mapping[str, object],
) -> Image.Image | None:
    if not bbox_norm:
        return None
    width, height = image.size
    if width <= 0 or height <= 0:
        return None
    x = max(0.0, min(1.0, _float(bbox_norm.get("x"), 0.0)))
    y = max(0.0, min(1.0, _float(bbox_norm.get("y"), 0.0)))
    box_width = max(0.0, min(1.0 - x, _float(bbox_norm.get("width"), 0.0)))
    box_height = max(0.0, min(1.0 - y, _float(bbox_norm.get("height"), 0.0)))
    if box_width <= 0 or box_height <= 0:
        return None
    left = int(x * width)
    top = int(y * height)
    right = int(max(left + 1, (x + box_width) * width))
    bottom = int(max(top + 1, (y + box_height) * height))
    return image.crop((left, top, min(width, right), min(height, bottom)))


def _crop_generated_image_by_source_bbox(
    image: Image.Image,
    *,
    bbox: Mapping[str, object],
    source_width: float,
    source_height: float,
) -> Image.Image | None:
    width, height = image.size
    if width <= 0 or height <= 0 or source_width <= 0 or source_height <= 0:
        return None
    left = int(max(0.0, _float(bbox.get("minX"), 0.0)) / source_width * width)
    top = int(max(0.0, _float(bbox.get("minY"), 0.0)) / source_height * height)
    right = int(max(left + 1, _float(bbox.get("maxX"), 0.0) / source_width * width))
    bottom = int(max(top + 1, _float(bbox.get("maxY"), 0.0) / source_height * height))
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return image.crop((left, top, right, bottom))


def _edge_shape_alignment_score(
    reference_image: Image.Image,
    generated_crop: Image.Image,
) -> float:
    if reference_image.width <= 2 or reference_image.height <= 2:
        return 0.0
    if generated_crop.width <= 2 or generated_crop.height <= 2:
        return 0.0
    reference = reference_image.resize(
        generated_crop.size,
        Image.Resampling.BILINEAR,
    )
    reference_mask = _threshold_edge_image(reference.filter(ImageFilter.FIND_EDGES))
    generated_mask = _threshold_edge_image(
        generated_crop.filter(ImageFilter.FIND_EDGES)
    ).filter(ImageFilter.MaxFilter(5))
    reference_values = list(reference_mask.getdata())
    generated_values = list(generated_mask.getdata())
    reference_count = sum(1 for value in reference_values if value > 0)
    if reference_count <= 0:
        return 0.0
    matched_count = sum(
        1
        for reference_value, generated_value in zip(
            reference_values,
            generated_values,
            strict=False,
        )
        if reference_value > 0 and generated_value > 0
    )
    return matched_count / max(1, reference_count)


def _edge_density(image: Image.Image) -> float:
    if image.width <= 2 or image.height <= 2:
        return 0.0
    edge_image = image.filter(ImageFilter.FIND_EDGES)
    thresholded = _threshold_edge_image(edge_image)
    values = list(thresholded.getdata())
    return sum(1 for value in values if value > 0) / max(1, len(values))


def _validate_edit_change_scope(
    *,
    source_image_data_url: str,
    generated_image_data_url: str,
    edit_operations: list[SnapshotEditOperation],
    render_mode: RenderMode,
) -> dict[str, object] | None:
    if render_mode != "edit":
        return None
    region_operations = [
        operation for operation in edit_operations if operation.bbox_norm is not None
    ]
    if not region_operations:
        return None
    try:
        source = Image.open(BytesIO(_data_url_to_bytes(source_image_data_url))).convert(
            "RGB"
        )
        generated = Image.open(
            BytesIO(_data_url_to_bytes(generated_image_data_url))
        ).convert("RGB")
    except (binascii.Error, OSError, UnidentifiedImageError) as exc:
        return {
            "status": "skipped",
            "reason": "edit_scope_image_decode_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    if generated.size != source.size:
        generated = generated.resize(source.size, Image.Resampling.BILINEAR)
    width, height = source.size
    editable_mask = Image.new("L", (width, height), 0)
    for operation in region_operations:
        bbox = operation.bbox_norm
        if bbox is None:
            continue
        left = int(max(0.0, min(1.0, _float(bbox.get("x"), 0.0))) * width)
        top = int(max(0.0, min(1.0, _float(bbox.get("y"), 0.0))) * height)
        right = int(
            max(
                0.0,
                min(
                    1.0,
                    _float(bbox.get("x"), 0.0) + _float(bbox.get("width"), 0.0),
                ),
            )
            * width
        )
        bottom = int(
            max(
                0.0,
                min(
                    1.0,
                    _float(bbox.get("y"), 0.0) + _float(bbox.get("height"), 0.0),
                ),
            )
            * height
        )
        editable_mask.paste(
            255, box=(left, top, max(left + 1, right), max(top + 1, bottom))
        )

    diff = ImageChops.difference(source, generated).convert("L")
    diff_mask = diff.point(lambda value: 255 if value >= 22 else 0)
    editable_values = list(editable_mask.getdata())
    diff_values = list(diff_mask.getdata())
    changed_outside = sum(
        1
        for editable, changed in zip(editable_values, diff_values, strict=False)
        if editable == 0 and changed > 0
    )
    outside_total = sum(1 for editable in editable_values if editable == 0)
    outside_change_ratio = changed_outside / max(1, outside_total)
    return {
        "status": "warning" if outside_change_ratio > 0.08 else "passed",
        "outside_change_ratio": round(outside_change_ratio, 5),
        "outside_change_threshold": 0.08,
        "region_count": len(region_operations),
    }


def _reference_edge_mask(edge_map: Image.Image) -> list[bool]:
    normalized = edge_map.convert("L")
    histogram = normalized.histogram()
    dark_count = sum(histogram[:48])
    bright_count = sum(histogram[208:])
    invert = dark_count < bright_count
    data = list(normalized.getdata())
    if invert:
        return [value < 96 for value in data]
    return [value > 96 for value in data]


def _threshold_edge_image(edge_image: Image.Image) -> Image.Image:
    histogram = edge_image.histogram()
    total = sum(histogram)
    target = max(1, int(total * 0.9))
    running = 0
    threshold = 32
    for value, count in enumerate(histogram):
        running += count
        if running >= target:
            threshold = max(24, value)
            break
    return edge_image.point(lambda value: 255 if value >= threshold else 0)


def _aspect_ratio_value(value: str) -> float:
    left, separator, right = value.partition(":")
    if not separator:
        return 0.0
    try:
        width = float(left)
        height = float(right)
    except ValueError:
        return 0.0
    if width <= 0 or height <= 0:
        return 0.0
    return width / height


def _collect_inline_image_summaries(value: object) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if isinstance(value, Mapping):
        for inline_key in ("inlineData", "inline_data"):
            inline_payload = value.get(inline_key)
            if not isinstance(inline_payload, Mapping):
                continue
            data = inline_payload.get("data")
            mime_type = inline_payload.get("mimeType") or inline_payload.get(
                "mime_type"
            )
            if isinstance(data, str):
                resolved_mime_type = (
                    mime_type
                    if isinstance(mime_type, str)
                    else "application/octet-stream"
                )
                summaries.append(
                    {
                        "mime_type": resolved_mime_type,
                        "data": summarize_image_data_url(
                            f"data:{resolved_mime_type};base64,{data}"
                        ),
                    }
                )
        for item in value.values():
            summaries.extend(_collect_inline_image_summaries(item))
    elif isinstance(value, list):
        for item in value:
            summaries.extend(_collect_inline_image_summaries(item))
    return summaries


def _build_preset_prompt(preset_selection: Mapping[str, object] | None) -> str:
    if preset_selection is None:
        return ""

    preset_config = load_config(config_file).presets
    requested_presets = (
        ("Style", preset_config.styles, _string(preset_selection.get("style"))),
        ("Lighting", preset_config.lights, _string(preset_selection.get("lighting"))),
        (
            "Scenery",
            preset_config.sceneries,
            _string(preset_selection.get("scenery")),
        ),
    )
    prompt_parts: list[str] = []
    for label, options, key in requested_presets:
        if not key:
            continue
        option = options.get(key)
        if option is None:
            raise ValueError(f"Unknown {label.lower()} preset: {key}")
        suffix = _string(option.prompt_suffix)
        if not suffix:
            continue
        prompt_parts.append(f"{label} preset ({option.label}): {suffix}")
    return "\n".join(prompt_parts)


def _build_preset_reference_images(
    preset_selection: Mapping[str, object] | None,
) -> list[SnapshotPresetReferenceImage]:
    if preset_selection is None:
        return []

    preset_config = load_config(config_file).presets
    requested_presets = (
        ("style", preset_config.styles, _string(preset_selection.get("style"))),
        ("lighting", preset_config.lights, _string(preset_selection.get("lighting"))),
        (
            "scenery",
            preset_config.sceneries,
            _string(preset_selection.get("scenery")),
        ),
    )
    references: list[SnapshotPresetReferenceImage] = []
    for group, options, key in requested_presets:
        if not key:
            continue
        option = options.get(key)
        if option is None:
            raise ValueError(f"Unknown {group} preset: {key}")
        image_data_url = _resolve_preset_reference_image_data_url(
            option.reference_image
        )
        if image_data_url is None:
            continue
        references.append(
            SnapshotPresetReferenceImage(
                group=group,
                key=key,
                label=option.label,
                image_data_url=image_data_url,
            )
        )
    return references


def _resolve_preset_reference_image_data_url(reference_image: str | None) -> str | None:
    raw_reference = _string(reference_image)
    if not raw_reference:
        return None
    if raw_reference.startswith("data:image/"):
        return _normalize_image_data_url(raw_reference)
    if raw_reference.startswith(("http://", "https://", "blob:")):
        logger.warning(
            "Skipping non-local preset reference image for OpenAI input: %s",
            raw_reference,
        )
        return None

    config_root = Path(config_file).resolve().parent
    image_path = (config_root / raw_reference.lstrip("/")).resolve()
    if not image_path.is_file():
        logger.warning("Preset reference image not found: %s", image_path)
        return None
    return _encode_file_as_data_url(image_path)


def _build_simple_generate_user_prompt(
    *,
    user_prompt: str | None,
    preset_prompt: str,
    preset_reference_images: list[SnapshotPresetReferenceImage],
    snapshot_payload: Mapping[str, object],
    include_text_contract: bool,
    text_contract_object_limit: int,
    strict_layout_mode: bool,
) -> str:
    prompt_parts = [
        _build_simple_prompt_image_role_hint(preset_reference_images),
    ]
    if include_text_contract:
        prompt_parts.append(
            _build_text_scene_contract_prompt(
                snapshot_payload,
                object_limit=text_contract_object_limit,
            )
        )
    photorealization_goal = (
        _sanitize_generate_user_prompt(user_prompt)
        if strict_layout_mode
        else _string(user_prompt)
    )
    prompt_parts.append(photorealization_goal or _DEFAULT_SIMPLE_USER_PROMPT)
    if preset_prompt:
        prompt_parts.append(_build_preset_isolation_prompt(preset_prompt))
    return "\n\n".join(prompt_parts)


def _sanitize_generate_user_prompt(user_prompt: str | None) -> str:
    raw_prompt = _string(user_prompt)
    if not raw_prompt:
        return _STRICT_LAYOUT_USER_PROMPT

    safe_sentences: list[str] = []
    saw_layout_directive = False
    for sentence in _prompt_sentences(raw_prompt):
        normalized = sentence.lower()
        if _is_image_role_sentence(normalized):
            saw_layout_directive = True
            continue
        if _is_layout_design_sentence(normalized):
            saw_layout_directive = True
            if _is_finish_sentence(normalized):
                safe_sentences.append(sentence)
            continue
        if _is_finish_sentence(normalized):
            safe_sentences.append(sentence)

    if safe_sentences:
        return " ".join(safe_sentences)
    if saw_layout_directive:
        return _STRICT_LAYOUT_USER_PROMPT
    return raw_prompt


def _prompt_sentences(prompt: str) -> list[str]:
    candidates = prompt.replace("\n", " ").split(".")
    return [candidate.strip(" \t:-") for candidate in candidates if candidate.strip()]


def _is_image_role_sentence(normalized_sentence: str) -> bool:
    return normalized_sentence.startswith("image 1") or "target 3d layout/source" in (
        normalized_sentence
    )


def _is_layout_design_sentence(normalized_sentence: str) -> bool:
    layout_keywords = (
        "design",
        "arrange",
        "include",
        "add ",
        "add a",
        "place ",
        "put ",
        "create ",
        "with a ",
        "with an ",
        "furniture",
        "layout",
        "room containing",
    )
    return any(keyword in normalized_sentence for keyword in layout_keywords)


def _is_finish_sentence(normalized_sentence: str) -> bool:
    finish_keywords = (
        "photoreal",
        "realistic",
        "material",
        "finish",
        "texture",
        "lighting",
        "illumination",
        "style",
        "mood",
        "scenery",
        "exterior",
        "window view",
        "through existing",
    )
    return any(keyword in normalized_sentence for keyword in finish_keywords)


def _build_preset_isolation_prompt(preset_prompt: str) -> str:
    return (
        "Preset isolation contract:\n"
        "Apply the selected presets only as bounded finish controls: material "
        "texture, lighting mood, illumination, and exterior scenery visible "
        "through existing openings. Do not transfer room layout, furniture, "
        "camera, crop, architecture, object count, object shape, object color, "
        "object size, object placement, or depth order from preset references.\n"
        + preset_prompt
    )


def _build_text_scene_contract_prompt(
    snapshot_payload: Mapping[str, object],
    *,
    object_limit: int,
) -> str:
    normalized_limit = _normalize_text_contract_object_limit(object_limit)
    canvas = _mapping(snapshot_payload.get("canvas"))
    canvas_width_px = max(1.0, _float(canvas.get("widthPx"), 1.0))
    canvas_height_px = max(1.0, _float(canvas.get("heightPx"), 1.0))
    visible_objects = _extract_visible_objects(snapshot_payload)
    room = _mapping(snapshot_payload.get("room"))
    openings = _mapping(room.get("openings"))
    doors = [
        opening
        for opening in _sequence(openings.get("doors"))
        if isinstance(opening, Mapping)
    ]
    windows = [
        opening
        for opening in _sequence(openings.get("windows"))
        if isinstance(opening, Mapping)
    ]

    prompt_parts = [
        "Text-only scene contract from the current 3D snapshot. This metadata is "
        "instructional and must not appear as visible text in the image.",
        (
            "Priority order: Image 1 visual evidence first, then this contract, "
            "then style/preset goals. If any metadata conflicts with Image 1, "
            "preserve Image 1."
        ),
        (
            "Camera/crop lock: keep the same viewpoint, apparent field of view, "
            "visible crop, object pixel scale, depth order, and occlusion. Do not "
            "zoom out, widen the lens, reveal extra floor/walls, or shrink the "
            "furniture in frame."
        ),
        (
            f"Room shell lock: preserve the same room boundaries, wall panels, "
            f"corner seams, {len(doors)} door(s), and {len(windows)} window(s)."
        ),
    ]

    palette_line = _build_text_contract_palette_line(snapshot_payload)
    if palette_line:
        prompt_parts.append(palette_line)

    object_count = len(visible_objects)
    if object_count:
        prompt_parts.append(
            f"Visible object lock: preserve {object_count} visible object(s), "
            "their count, identity, approximate screen position, scale, color, "
            "silhouette, and relative depth."
        )

    object_lines = _build_text_contract_object_lines(
        snapshot_payload,
        object_limit=normalized_limit,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
    )
    if object_lines:
        prompt_parts.append(
            "Salient visible objects to preserve:\n" + "\n".join(object_lines)
        )

    opening_lines = _build_text_contract_opening_lines(
        doors=doors,
        windows=windows,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
    )
    if opening_lines:
        prompt_parts.append("Opening placement reminders:\n" + "\n".join(opening_lines))

    prompt_parts.append(
        "Allowed change: only improve photorealistic finish, material realism, "
        "lighting quality, and selected preset mood. Do not redesign, add "
        "furniture, remove furniture, recolor locked objects, or add decor unless "
        "already visible in Image 1."
    )
    return "\n".join(prompt_parts)


def _build_text_contract_palette_line(
    snapshot_payload: Mapping[str, object],
) -> str:
    room = _mapping(snapshot_payload.get("room"))
    surface_payload = _mapping(room.get("surfaceColors"))
    surface_lines = []
    for label, key in (
        ("walls", "wallColorHex"),
        ("floor", "floorColorHex"),
        ("ceiling", "ceilingColorHex"),
    ):
        color = _string(surface_payload.get(key))
        if _is_hex_color(color):
            surface_lines.append(f"{label}={color}")
    object_lines = []
    for object_id, payload in _top_visible_object_items(snapshot_payload, limit=8):
        color = _string(payload.get("colorHex"))
        if _is_hex_color(color):
            object_lines.append(f"{object_id}={color}")
    if not surface_lines and not object_lines:
        return ""
    parts = ["Palette lock:"]
    if surface_lines:
        parts.append("surfaces " + ", ".join(surface_lines))
    if object_lines:
        parts.append("objects " + ", ".join(object_lines))
    return "; ".join(parts) + "."


def _build_text_contract_object_lines(
    snapshot_payload: Mapping[str, object],
    *,
    object_limit: int,
    canvas_width_px: float,
    canvas_height_px: float,
) -> list[str]:
    if object_limit <= 0:
        return []
    lines: list[str] = []
    for object_id, payload in _top_visible_object_items(
        snapshot_payload,
        limit=object_limit,
    ):
        bbox = _normalized_screen_bbox(payload.get("screenBboxPx"))
        dimensions = _mapping(payload.get("dimensionsMm"))
        color = _string(payload.get("colorHex")) or "source-color"
        material = _string(payload.get("material")) or "source-material"
        position = _screen_position_phrase(
            bbox,
            canvas_width_px=canvas_width_px,
            canvas_height_px=canvas_height_px,
        )
        bbox_text = _bbox_norm_text(
            bbox,
            canvas_width_px=canvas_width_px,
            canvas_height_px=canvas_height_px,
        )
        size_text = _dimensions_text(dimensions)
        distance = _float(payload.get("distanceToCameraM"), math.nan)
        distance_text = (
            f", distance_m={round(distance, 2)}" if math.isfinite(distance) else ""
        )
        lines.append(
            "- "
            f"{object_id}: label={_visible_object_name(payload)}, "
            f"position={position}, bbox_norm={bbox_text}, color={color}, "
            f"material={material}{size_text}{distance_text}"
        )
    return lines


def _build_text_contract_opening_lines(
    *,
    doors: list[Mapping[str, object]],
    windows: list[Mapping[str, object]],
    canvas_width_px: float,
    canvas_height_px: float,
) -> list[str]:
    lines: list[str] = []
    for opening_kind, raw_openings in (("door", doors), ("window", windows)):
        for index, opening in enumerate(raw_openings[:6], start=1):
            opening_id = _string(
                opening.get("id"),
                fallback=f"{opening_kind}_{index}",
            )
            bbox = _normalized_screen_bbox(opening.get("screenBboxPx"))
            position = _screen_position_phrase(
                bbox,
                canvas_width_px=canvas_width_px,
                canvas_height_px=canvas_height_px,
            )
            bbox_text = _bbox_norm_text(
                bbox,
                canvas_width_px=canvas_width_px,
                canvas_height_px=canvas_height_px,
            )
            lines.append(
                f"- {opening_kind}:{opening_id}: position={position}, "
                f"bbox_norm={bbox_text}"
            )
    return lines


def _top_visible_object_items(
    snapshot_payload: Mapping[str, object],
    *,
    limit: int,
) -> list[tuple[str, Mapping[str, object]]]:
    if limit <= 0:
        return []
    items = list(_extract_visible_objects(snapshot_payload).items())

    def sort_key(item: tuple[str, Mapping[str, object]]) -> tuple[float, float, str]:
        object_id, payload = item
        bbox = _normalized_screen_bbox(payload.get("screenBboxPx"))
        area = 0.0
        if bbox is not None:
            width = min(
                _float(bbox.get("width"), 0.0),
                _float(bbox.get("maxX"), 0.0),
            )
            height = min(
                _float(bbox.get("height"), 0.0),
                _float(bbox.get("maxY"), 0.0),
            )
            area = max(0.0, width) * max(0.0, height)
        role_penalty = _visible_object_contract_penalty(payload)
        distance = _float(payload.get("distanceToCameraM"), math.inf)
        return (role_penalty, -area, distance, object_id)

    return sorted(items, key=sort_key)[:limit]


def _visible_object_contract_penalty(payload: Mapping[str, object]) -> float:
    object_type = _string(payload.get("type")).lower()
    object_name = _visible_object_name(payload).lower()
    minor_keywords = (
        "lamp",
        "light",
        "art",
        "decor",
        "cushion",
        "pillow",
        "plant",
        "rug",
        "laptop",
    )
    if any(
        keyword in object_type or keyword in object_name for keyword in minor_keywords
    ):
        return 1.0
    return 0.0


def _screen_position_phrase(
    bbox: Mapping[str, float] | None,
    *,
    canvas_width_px: float,
    canvas_height_px: float,
) -> str:
    if bbox is None:
        return "unknown"
    center_x = (_float(bbox.get("minX"), 0.0) + _float(bbox.get("maxX"), 0.0)) / 2
    center_y = (_float(bbox.get("minY"), 0.0) + _float(bbox.get("maxY"), 0.0)) / 2
    x_ratio = center_x / max(1.0, canvas_width_px)
    y_ratio = center_y / max(1.0, canvas_height_px)
    horizontal = "left" if x_ratio < 0.33 else "right" if x_ratio > 0.67 else "center"
    vertical = "upper" if y_ratio < 0.33 else "lower" if y_ratio > 0.67 else "middle"
    return f"{vertical}-{horizontal}"


def _bbox_norm_text(
    bbox: Mapping[str, float] | None,
    *,
    canvas_width_px: float,
    canvas_height_px: float,
) -> str:
    if bbox is None:
        return "unknown"
    x = _float(bbox.get("minX"), 0.0) / max(1.0, canvas_width_px)
    y = _float(bbox.get("minY"), 0.0) / max(1.0, canvas_height_px)
    width = _float(bbox.get("width"), 0.0) / max(1.0, canvas_width_px)
    height = _float(bbox.get("height"), 0.0) / max(1.0, canvas_height_px)
    return f"x={round(x, 3)},y={round(y, 3)},w={round(width, 3)},h={round(height, 3)}"


def _dimensions_text(dimensions: Mapping[str, object]) -> str:
    if not dimensions:
        return ""
    width = _float(dimensions.get("width"), 0.0)
    depth = _float(dimensions.get("depth"), 0.0)
    height = _float(dimensions.get("height"), 0.0)
    if width <= 0 or depth <= 0 or height <= 0:
        return ""
    return f", size_mm={round(width)}x{round(depth)}x{round(height)}"


def _build_edit_operation_contract_prompt(
    edit_operations: list[SnapshotEditOperation],
) -> str:
    if not edit_operations:
        return ""
    lines = [
        "Edit operation DSL:",
        (
            "Execute only the operations below. Regions/objects not listed here "
            "are immutable unless camera transfer is explicitly active."
        ),
    ]
    for index, operation in enumerate(edit_operations, start=1):
        target_kind = "region" if operation.bbox_norm is not None else "object"
        operation_type = _operation_type_label(operation.operation_type)
        pieces = [
            f"E{index}",
            f"target={target_kind}:{operation.object_id}",
            f"operation={operation_type}",
        ]
        if operation.object_name:
            pieces.append(f"name={operation.object_name}")
        if operation.bbox_norm is not None:
            pieces.append(
                "bbox_norm="
                + json.dumps(
                    dict(operation.bbox_norm),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            )
        if operation.target_color is not None:
            pieces.append(f"target_color={operation.target_color}")
        if operation.replacement_image_data_url is not None:
            pieces.append("reference_image=appearance_only")
        if operation.prompt is not None:
            pieces.append(f"instruction={operation.prompt}")
        lines.append("- " + "; ".join(pieces))
    lines.append(
        "For add/replace/recolor/remove, preserve placement, scale, perspective, "
        "occlusion, and all surrounding pixels outside the target. For recolor, "
        "change only the target object's finish/color while preserving handles, "
        "panel seams, drawer lines, decorative motifs, visible grain/texture "
        "structure, and any foreground or neighboring object crossing the box. "
        "For add/replace/remove, any foreground or neighboring object that crosses, "
        "touches, or partially occludes the target box is protected unless it is "
        "the named target itself."
    )
    return "\n".join(lines)


def _build_simple_prompt_image_role_hint(
    preset_reference_images: list[SnapshotPresetReferenceImage],
) -> str:
    prompt_lines = ["Image 1 is the only target 3D layout/source to transform."]
    if not preset_reference_images:
        return "\n".join(prompt_lines)

    last_image_number = len(preset_reference_images) + 1
    if len(preset_reference_images) == 1:
        prompt_lines.append(
            "Image 2 is a preset reference only. Use it only for finish, "
            "material, lighting, or scenery cues."
        )
    else:
        prompt_lines.append(
            f"Images 2-{last_image_number} are preset references only. Use them "
            "only for finish, material texture, lighting mood, or exterior "
            "scenery cues through existing openings."
        )
    for image_number, reference in enumerate(preset_reference_images, start=2):
        prompt_lines.append(
            f"Image {image_number}: {reference.group} preset reference "
            f"({reference.label}) only."
        )
    prompt_lines.append(
        "Do not copy room, camera, furniture, openings, layout, object count, "
        "object size, object color, or object placement from preset reference "
        "images."
    )
    return "\n".join(prompt_lines)


def _build_final_user_prompt(
    *,
    user_prompt: str | None,
    render_mode: RenderMode,
    preset_prompt: str,
    edit_operations: list[SnapshotEditOperation],
    visible_objects: Mapping[str, Mapping[str, object]],
    layout_lock_prompt: str = "",
    layout_reference_used: bool = False,
    annotated_reference_used: bool = False,
    depth_map_used: bool = False,
    edge_map_used: bool = False,
    segmentation_map_used: bool = False,
    scene_reference_mode: SceneReferenceMode = "none",
    locked_geometry_references: list[SnapshotLockedGeometryReference] | None = None,
    locked_geometry_atlases: list[SnapshotLockedGeometryAtlas] | None = None,
    preset_reference_count: int = 0,
    camera_transfer_reference_used: bool = False,
    text_contract_used: bool = False,
    include_edit_operation_dsl: bool = True,
    strict_layout_mode: bool = False,
    layout_guard_mask_used: bool = False,
    layout_materialization_mode_used: bool = False,
) -> str:
    raw_user_prompt = (
        _sanitize_generate_user_prompt(user_prompt)
        if strict_layout_mode and render_mode in {"generate", "camera_rerender"}
        else _string(user_prompt)
    )
    if render_mode in {"generate", "camera_rerender"}:
        geometry_reference_list = locked_geometry_references or []
        geometry_atlas_list = locked_geometry_atlases or []
        prompt_parts: list[str] = []
        image_role_prompt = _build_generate_image_role_prompt(
            render_mode=render_mode,
            scene_reference_mode=scene_reference_mode,
            layout_reference_used=layout_reference_used,
            annotated_reference_used=annotated_reference_used,
            depth_map_used=depth_map_used,
            edge_map_used=edge_map_used,
            segmentation_map_used=segmentation_map_used,
            locked_geometry_atlas_count=len(geometry_atlas_list),
            locked_geometry_reference_count=len(geometry_reference_list),
            preset_reference_count=preset_reference_count,
            appearance_reference_count=sum(
                1
                for operation in edit_operations
                if operation.replacement_image_data_url is not None
            ),
        )
        if image_role_prompt:
            prompt_parts.append(image_role_prompt)
        if layout_lock_prompt:
            prompt_parts.append(layout_lock_prompt)
        if render_mode == "camera_rerender":
            prompt_parts.append(
                "Appearance and continuity goal, always secondary to the new 3D target geometry: "
                + (raw_user_prompt or _DEFAULT_USER_PROMPT)
            )
        else:
            prompt_parts.append(
                "Material, lighting, and scenery goal, secondary to the exact input view and locked palette: "
                + (raw_user_prompt or _DEFAULT_USER_PROMPT)
            )
        locked_geometry_prompt = _build_locked_geometry_reference_prompt(
            geometry_reference_list,
            locked_geometry_atlases=geometry_atlas_list,
        )
        if locked_geometry_prompt:
            prompt_parts.append(locked_geometry_prompt)
        appearance_memory_prompt = _build_object_appearance_memory_prompt(
            edit_operations
        )
        if appearance_memory_prompt:
            prompt_parts.append(appearance_memory_prompt)
        if preset_prompt:
            prompt_parts.append(
                "Apply these selected presets as finish realism, material texture, "
                "lighting/illumination, and exterior scenery visible through existing "
                "openings only. They must not change architecture, openings, "
                "built-ins, object colors, object shape, object count, object size, "
                "camera, depth order, occlusion, or object placement:\n" + preset_prompt
            )
        return "\n\n".join(prompt_parts)

    if not raw_user_prompt and not preset_prompt and not edit_operations:
        raise ValueError(
            "Provide at least one edit operation, preset change, or prompt change."
        )

    prompt_parts: list[str] = [
        (
            "Transform Image 1, which is the exact 3D source view and the primary "
            "layout source."
        )
        if layout_materialization_mode_used
        else "Transform Image 1, which is the current rendered photo and the primary source image."
    ]
    geometry_reference_list = locked_geometry_references or []
    geometry_atlas_list = locked_geometry_atlases or []
    if layout_guard_mask_used:
        prompt_parts.append(
            "The edit mask is a layout guard: photorealize transparent editable "
            "areas, and preserve opaque masked areas exactly. Use the mask to "
            "avoid changing crop borders, trusted openings, and do-not-hallucinate "
            "regions. If opaque protected pixels appear inside a drawn edit box, "
            "they mark foreground or neighboring non-target objects that must stay "
            "unchanged."
        )
    next_image_number = 2
    if camera_transfer_reference_used:
        prompt_parts.append(
            f"Image {next_image_number} is the sharp target 3D layout/camera reference. "
            "Use it only to match the new camera angle, framing, perspective, "
            "depth order, occlusion, and visible room/object placement. Preserve "
            "Image 1's materials, object identity, colors, and realism unless "
            "changed by explicit edit operations or changed presets."
        )
        next_image_number += 1
    if depth_map_used:
        prompt_parts.append(
            f"Image {next_image_number} is a deterministic depth map from the same "
            "3D camera. Use it only to preserve depth order, distance relationships, "
            "foreground/background separation, and object scale. Do not render it as "
            "texture, lighting, or material."
        )
        next_image_number += 1
    if edge_map_used:
        prompt_parts.append(
            f"Image {next_image_number} is a deterministic edge map from the same "
            "3D source. Align wall seams, openings, crop-edge walls, furniture "
            "silhouettes, TV/media outlines, and major object boundaries to it. Do "
            "not render black edge lines in the final image."
        )
        next_image_number += 1
    if segmentation_map_used:
        prompt_parts.append(
            f"Image {next_image_number} is a deterministic segmentation map from "
            "the same 3D scene. Use flat color regions only to preserve object and "
            "room-surface footprints, silhouettes, object count, and occlusion. Do "
            "not copy colors from this map."
        )
        next_image_number += 1
    if layout_reference_used:
        prompt_parts.append(
            f"Image {next_image_number} is an optional same-camera structural guide for identifying existing objects only. Do not transfer camera, layout, or geometry from it, and do not reproduce any guide graphics."
        )
        next_image_number += 1
    elif not camera_transfer_reference_used:
        prompt_parts.append(
            "Preserve the current camera, room layout, architecture, openings, object count, and all unselected objects."
        )
    if layout_materialization_mode_used:
        prompt_parts.append(
            "Materialization rule: improve realism, materials, lighting, and surface "
            "texture only. Do not change the source camera, crop, wall seams, "
            "openings, object inventory, object placement, object scale, object "
            "silhouettes, perspective, depth order, or occlusion."
        )
    else:
        prompt_parts.append(
            "Immutable edit rule: pixels and objects outside the selected object or drawn edit boxes must remain unchanged in shape, position, scale, perspective, material identity, lighting relationship, and visibility unless a camera-transfer reference is explicitly provided."
        )
    if text_contract_used and include_edit_operation_dsl:
        edit_contract_prompt = _build_edit_operation_contract_prompt(edit_operations)
        if edit_contract_prompt:
            prompt_parts.append(edit_contract_prompt)
    if annotated_reference_used:
        if layout_materialization_mode_used or camera_transfer_reference_used:
            prompt_parts.append(
                f"Image {next_image_number} is a structural geometry guide. It may "
                "show semantic room-shell planes, sharpened wall/object edges, crop "
                "guards, opening boxes, clean object anchors, and rejected noisy "
                "anchors. Use it only to preserve the same camera, crop, wall planes, "
                "shell corners, openings, object positions, scale, silhouettes, and "
                "depth/crop relationships. The final output must not contain guide "
                "colors, guide edges, colored boxes, labels, text, masks, overlays, "
                "or annotations."
            )
        else:
            prompt_parts.append(
                f"Image {next_image_number} is the edit-region guide. "
                "It shows colored bounding boxes drawn over the photo. Each box marks a region to edit; "
                "the small label inside (e.g. R1, R2, R3) is the region ID that matches the instructions below. "
                "Use the boxes and IDs only as your instruction map — they are never part of the scene. "
                "The final output must be a clean photorealistic photo with NONE of the following: "
                "no colored box outlines, no bounding rectangles, no region IDs, no label text, "
                "no tinted overlays, no guide graphics, no annotations of any kind."
            )
        next_image_number += 1
    if geometry_atlas_list:
        end_image_number = next_image_number + len(geometry_atlas_list) - 1
        prompt_parts.append(
            f"Images {next_image_number}-{end_image_number} are object geometry "
            "atlas contact sheets from the clean 3D source. Each atlas cell maps to "
            "an object id in the prompt. Use them only for exact silhouettes, "
            "proportions, shelf/TV/media details, and object footprints. Do not "
            "render atlas labels, borders, or material."
        )
        next_image_number = end_image_number + 1
    if geometry_reference_list:
        end_image_number = next_image_number + len(geometry_reference_list) - 1
        prompt_parts.append(
            f"Images {next_image_number}-{end_image_number} are object-level "
            "geometry lock crops from the clean 3D source. Use them only to "
            "preserve the listed object silhouettes and proportions."
        )
        next_image_number = end_image_number + 1
    if preset_reference_count > 0:
        end_image_number = next_image_number + preset_reference_count - 1
        if preset_reference_count == 1:
            prompt_parts.append(
                f"Image {next_image_number} is a changed preset reference only. "
                "Use it only for the explicitly changed finish, material texture, "
                "lighting mood, or exterior scenery. Do not copy its room, camera, "
                "furniture, openings, layout, object count, object shape, object "
                "size, object color, or object placement."
            )
        else:
            prompt_parts.append(
                f"Images {next_image_number}-{end_image_number} are changed preset "
                "references only. Use them only for the explicitly changed finish, "
                "material texture, lighting mood, or exterior scenery. Do not copy "
                "their rooms, camera, furniture, openings, layout, object count, "
                "object shape, object size, object color, or object placement."
            )
        next_image_number = end_image_number + 1
    appearance_reference_count = sum(
        1
        for operation in edit_operations
        if operation.replacement_image_data_url is not None
    )
    if appearance_reference_count > 0:
        end_image_number = next_image_number + appearance_reference_count - 1
        if appearance_reference_count == 1:
            prompt_parts.append(
                f"Image {next_image_number} is an uploaded object/appearance reference for an explicit edit operation only."
            )
        else:
            prompt_parts.append(
                f"Images {next_image_number}-{end_image_number} are uploaded object/appearance references for explicit edit operations only."
            )
    if visible_objects:
        visible_summary = [
            f"{object_id} ({_visible_object_name(payload)})"
            for object_id, payload in visible_objects.items()
        ]
        visible_context = "target reference" if layout_reference_used else "the image"
        prompt_parts.append(
            f"Objects visible in {visible_context}: " + ", ".join(visible_summary)
        )
    target_lock_prompt = _build_edit_target_geometry_prompt(
        edit_operations=edit_operations,
        visible_objects=visible_objects,
    )
    if target_lock_prompt:
        prompt_parts.append(target_lock_prompt)
    locked_geometry_prompt = _build_locked_geometry_reference_prompt(
        geometry_reference_list,
        locked_geometry_atlases=geometry_atlas_list,
    )
    if locked_geometry_prompt:
        prompt_parts.append(locked_geometry_prompt)
    if raw_user_prompt:
        prompt_parts.append("Requested edit: " + raw_user_prompt)
    if preset_prompt:
        prompt_parts.append("Preset update:\n" + preset_prompt)

    replacement_index = 1
    for operation in edit_operations:
        if operation.operation_type == "remove":
            if operation.bbox_norm is not None:
                prompt_parts.append(
                    f"Drawn region id: {operation.object_id}.\n"
                    "Remove the object or visual content inside the matching boxed region. "
                    "Fill the area naturally with the surrounding wall, floor, bedding, or "
                    "background surfaces as appropriate. Do not remove or alter objects "
                    "outside the boxed region."
                )
            else:
                prompt_parts.append(
                    f"Selected object id: {operation.object_id}.\n"
                    f"Remove only the selected {operation.object_name or 'object'} and fill the area naturally. "
                    "Do not alter any unselected object."
                )
        if operation.prompt is not None:
            target_text = (
                f"drawn region {operation.object_id}"
                if operation.bbox_norm is not None
                else f"selected object {operation.object_id}"
            )
            prompt_parts.append(
                f"Instruction for {target_text}: {operation.prompt}"
                " Apply this instruction only to that target."
            )
        if operation.replacement_image_data_url is not None:
            if operation.bbox_norm is not None and operation.operation_type == "add":
                prompt_parts.append(
                    f"Drawn region id: {operation.object_id}. Reference image {replacement_index} belongs to this region.\n"
                    f"Add the referenced {operation.object_name or 'object'} inside the matching boxed region from the edit-region guide. "
                    "Fit it to that region's perspective, scale, and occlusion. "
                    "Do not place the new object outside that region. Do not cover, erase, redraw, or merge any existing object that crosses the box or is protected by the mask. Do not move architecture or existing main furniture outside the boxed region."
                )
            elif operation.bbox_norm is not None:
                prompt_parts.append(
                    f"Drawn region id: {operation.object_id}. Reference image {replacement_index} belongs to this region.\n"
                    f"Use reference image {replacement_index} only for the replacement appearance inside the matching boxed region from the edit-region guide. "
                    "Replace the object/content in that boxed region while preserving the region's placement, scale, perspective, and occlusion. Remove the old object/content in the box instead of duplicating it. "
                    "If another object crosses, overlaps, stands in front of, or touches the boxed region, preserve that other object exactly; it is not part of the replacement target. Do not change architecture or main furniture outside the boxed region."
                )
                replacement_index += 1
                continue
            else:
                prompt_parts.append(
                    "Selected object id: "
                    f"{operation.object_id}. Reference image {replacement_index} belongs to this operation.\n"
                    "Selected object geometry locks from the source image and target layout override reference-image scale, crop, camera, and composition. "
                    "Use the uploaded image as an appearance-only replacement reference, not a scale, camera, crop, layout, or composition reference. "
                    f"Use reference image {replacement_index} only for the selected {operation.object_name}'s appearance. "
                    "Keep the target-view placement, size, angle, depth order, and occlusion from the input images. "
                    "Ignore the reference image background, room, crop, and camera. "
                    "Do not cover, remove, redraw, or relocate any unselected object."
                )
            replacement_index += 1
        if operation.target_color is not None:
            if operation.bbox_norm is not None:
                prompt_parts.append(
                    f"Drawn region id: {operation.object_id}.\n"
                    f"Change only the color/material of the object or surface inside the matching boxed region to {operation.target_color}. "
                    "Keep its silhouette, footprint, perspective, panel gaps, drawer lines, handles, decorative motifs, visible grain/texture structure, and shadow/lighting continuity. "
                    "If another object crosses, overlaps, stands in front of, or touches the boxed region, preserve that other object exactly; do not repaint, remove, or merge it into the target. "
                    "Everything outside the intended target object must remain unchanged."
                )
            else:
                prompt_parts.append(
                    f"Selected object id: {operation.object_id}.\n"
                    f"Change only the color of the selected {operation.object_name} to {operation.target_color}. "
                    "Keep its silhouette, panel gaps, drawer lines, handles, decorative motifs, visible grain/texture structure, placement, and all other objects unchanged."
                )

    return "\n\n".join(prompt_parts)


def _build_edit_target_geometry_prompt(
    *,
    edit_operations: list[SnapshotEditOperation],
    visible_objects: Mapping[str, Mapping[str, object]],
) -> str:
    prompt_parts: list[str] = []
    for operation in edit_operations:
        operation_label = _operation_type_label(operation.operation_type)
        if operation.bbox_norm is not None:
            object_name = operation.object_name or operation.object_id
            prompt_parts.append(
                f"{operation.object_id} ({operation_label}, {object_name})"
            )
            continue
        visible_payload = visible_objects.get(operation.object_id)
        object_name = operation.object_name or (
            _visible_object_name(visible_payload)
            if visible_payload is not None
            else "object"
        )
        screen_bbox = (
            _mapping(visible_payload.get("screenBboxPx"))
            if visible_payload is not None
            else {}
        )
        if screen_bbox:
            bbox_json = json.dumps(
                {"screenBboxPx": dict(screen_bbox)},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            prompt_parts.append(
                f"{operation.object_id} ({object_name}) target geometry {bbox_json}"
            )
        else:
            prompt_parts.append(f"{operation.object_id} ({object_name})")

    if not prompt_parts:
        return ""
    return (
        "Selected object ids or drawn edit regions. Only these targets may change:\n"
        + "\n".join(prompt_parts)
    )


def _operation_type_label(
    operation_type: Literal["prompt", "add", "replace", "recolor", "remove"] | None,
) -> str:
    if operation_type == "add":
        return "add object"
    if operation_type == "replace":
        return "replace object"
    if operation_type == "recolor":
        return "change color"
    if operation_type == "remove":
        return "remove object"
    return "prompt edit"


def _build_object_appearance_memory_prompt(
    edit_operations: list[SnapshotEditOperation],
) -> str:
    paired_operations = [
        operation
        for operation in edit_operations
        if operation.replacement_image_data_url is not None
    ]
    if not paired_operations:
        return ""

    prompt_parts = [
        "Object appearance memory:",
        (
            "After the scene/layout images, each reference image is paired with "
            "one object id below. Use these references only for that object's "
            "style, material, silhouette cues, and recognizable appearance. The "
            "labeled 3D layout image still controls camera, placement, size, "
            "visibility, and occlusion."
        ),
    ]
    for index, operation in enumerate(paired_operations, start=1):
        object_name = operation.object_name or "object"
        prompt_parts.append(
            f"Reference image {index}: object id {operation.object_id} ({object_name})."
        )
    return "\n".join(prompt_parts)


def _build_locked_geometry_reference_prompt(
    references: list[SnapshotLockedGeometryReference],
    *,
    locked_geometry_atlases: list[SnapshotLockedGeometryAtlas],
) -> str:
    usable_references = [
        reference for reference in references if reference.image_data_url is not None
    ]
    usable_atlases = [
        atlas
        for atlas in locked_geometry_atlases
        if atlas.image_data_url is not None and atlas.objects
    ]
    if not usable_references and not usable_atlases:
        return ""

    prompt_parts = ["Object geometry lock references:"]
    if usable_atlases:
        prompt_parts.append(
            "Each object geometry atlas is a contact sheet of exact object crops "
            "from the clean 3D source. Use atlas cells only for exact shape, "
            "silhouette, opening/shelf relationships, and proportions from the 3D "
            "layout. Do not use atlas labels, atlas borders, material, lighting, "
            "scenery, camera, or style in the final render."
        )
        for atlas in usable_atlases:
            atlas_index = atlas.atlas_index
            prompt_parts.append(f"Geometry atlas {atlas_index}:")
            for item in atlas.objects or []:
                object_name = item.object_name or "object"
                bbox_text = ""
                if isinstance(item.screen_bbox_px, Mapping):
                    bbox_text = " screen_bbox_px=" + json.dumps(
                        dict(item.screen_bbox_px),
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                cell_label = item.cell_label or f"cell {item.cell_index}"
                prompt_parts.append(
                    f"- {cell_label}: object id {item.object_id} ({object_name}). "
                    "Preserve this object's exact 3D layout silhouette, "
                    f"height/width/depth proportions, and visible internal details.{bbox_text}"
                )

    if usable_references:
        prompt_parts.append(
            "Legacy per-object geometry crops are also provided below. They are "
            "secondary to atlas and 3D layout geometry."
        )
    for index, reference in enumerate(usable_references, start=1):
        object_name = reference.object_name or "object"
        bbox = reference.screen_bbox_px
        bbox_text = ""
        if isinstance(bbox, Mapping):
            bbox_text = " screen_bbox_px=" + json.dumps(
                dict(bbox), ensure_ascii=True, separators=(",", ":")
            )
        prompt_parts.append(
            f"Geometry lock reference {index}: object id {reference.object_id} "
            f"({object_name}). Preserve this object's exact 3D layout silhouette, "
            f"height/width/depth proportions, and internal shelf/cabinet openings.{bbox_text}"
        )
    return "\n".join(prompt_parts)


def _build_generate_image_role_prompt(
    *,
    render_mode: RenderMode,
    scene_reference_mode: SceneReferenceMode,
    layout_reference_used: bool,
    annotated_reference_used: bool,
    depth_map_used: bool,
    edge_map_used: bool,
    segmentation_map_used: bool,
    locked_geometry_atlas_count: int = 0,
    locked_geometry_reference_count: int = 0,
    preset_reference_count: int = 0,
    appearance_reference_count: int = 0,
) -> str:
    if render_mode == "camera_rerender":
        prompt_parts: list[str] = [
            "Image 1 is the clean 3D target render from the newly selected camera. It is the primary source of truth for camera, framing, room shell, object positions, object sizes, visible colors, occlusion, and depth order."
        ]
    elif scene_reference_mode == "scene_reference_camera_transfer":
        prompt_parts: list[str] = [
            "Image 1 is a previously rendered source reference. Transfer it to "
            "the newly selected target camera only where it does not conflict "
            "with the geometry references and labeled 3D layout guide."
        ]
    else:
        prompt_parts = [
            "Image 1 is the clean 3D source render and the primary source of "
            "truth for camera, framing, room shell, object positions, object "
            "sizes, visible colors, occlusion, and depth order. Preserve the "
            "same apparent pixel scale: do not dolly out, zoom out, widen the "
            "lens, reveal extra floor/walls, or make the furniture smaller in "
            "frame."
        ]
    next_image_number = 2
    if depth_map_used:
        prompt_parts.append(
            f"Image {next_image_number} is a deterministic depth map from the "
            "same Three.js camera. Use it only as a geometry/depth-order "
            "reference, never as style or material."
        )
        next_image_number += 1
    if edge_map_used:
        prompt_parts.append(
            f"Image {next_image_number} is a deterministic edge map from the "
            "same clean 3D source. Align wall seams, openings, furniture "
            "silhouettes, and major object boundaries to it at the same screen "
            "scale and crop."
        )
        next_image_number += 1
    if segmentation_map_used:
        prompt_parts.append(
            f"Image {next_image_number} is a deterministic segmentation map "
            "from the same Three.js scene. Each flat color marks a locked "
            "object or room surface region; use it to preserve object count, "
            "placement, and silhouettes."
        )
        next_image_number += 1
    if scene_reference_mode == "target_layout_with_scene_reference":
        prompt_parts.append(
            f"Image {next_image_number} is a previously rendered same-room reference. Use it only for realistic finish continuity and object identity when it agrees with the 3D layout."
        )
        next_image_number += 1

    if layout_reference_used:
        prompt_parts.append(
            f"Image {next_image_number} is the top-down layout reference. Use it only to clarify the room footprint and openings."
        )
        next_image_number += 1
    if annotated_reference_used:
        prompt_parts.append(
            f"Image {next_image_number} is a same-camera labeled structural guide. Do not render the guide lines, labels, outlines, or UI marks in the final image."
        )
        next_image_number += 1
    if locked_geometry_atlas_count > 0:
        end_image_number = next_image_number + locked_geometry_atlas_count - 1
        prompt_parts.append(
            f"Images {next_image_number}-{end_image_number} are object geometry atlas contact sheets from the clean 3D source. Each atlas cell maps to an object id in the prompt. Use them only to preserve listed object silhouettes, proportions, and visible structural details."
        )
        next_image_number = end_image_number + 1
    if locked_geometry_reference_count > 0:
        end_image_number = next_image_number + locked_geometry_reference_count - 1
        prompt_parts.append(
            f"Images {next_image_number}-{end_image_number} are legacy exact object-level geometry lock crops from the clean 3D source. Use them only to preserve listed object silhouettes, proportions, and structural relationships."
        )
        next_image_number = end_image_number + 1
    if preset_reference_count > 0:
        end_image_number = next_image_number + preset_reference_count - 1
        if preset_reference_count == 1:
            prompt_parts.append(
                f"Image {next_image_number} is a selected preset reference image. "
                "Use it only for finish realism, material texture, lighting mood, "
                "or exterior scenery through existing openings. Do not use it to "
                "change camera, layout, architecture, object count, object shape, "
                "object color, object size, or object placement."
            )
        else:
            prompt_parts.append(
                f"Images {next_image_number}-{end_image_number} are selected preset "
                "reference images. Use them only for finish realism, material "
                "texture, lighting mood, or exterior scenery through existing "
                "openings. Do not use them to change camera, layout, architecture, "
                "object count, object shape, object color, object size, or object "
                "placement."
            )
        next_image_number = end_image_number + 1
    if appearance_reference_count > 0:
        end_image_number = next_image_number + appearance_reference_count - 1
        prompt_parts.append(
            f"Images {next_image_number}-{end_image_number} are object appearance memory references. They are secondary to the 3D layout geometry and object geometry lock crops."
        )

    return "\n".join(prompt_parts)


def _resolve_scene_reference_mode(
    *,
    scene_reference_mode: SceneReferenceMode,
    render_mode: RenderMode,
    scene_reference_image_data_url: str | None,
) -> SceneReferenceMode:
    if render_mode == "edit":
        if scene_reference_mode == "scene_reference_camera_transfer":
            return "scene_reference_camera_transfer"
        return "none"
    if render_mode not in {"generate", "camera_rerender"}:
        return "none"
    if scene_reference_image_data_url is None:
        return "none"
    if scene_reference_mode == "target_layout_with_scene_reference":
        return "target_layout_with_scene_reference"
    if render_mode == "camera_rerender":
        return "target_layout_with_scene_reference"
    if scene_reference_mode == "scene_reference_camera_transfer":
        return "scene_reference_camera_transfer"
    return "none"


def _build_generate_layout_lock_prompt(
    *,
    scene_reference_mode: SceneReferenceMode,
    snapshot_payload: Mapping[str, object],
) -> str:
    prompt_parts = [
        "Layout lock instructions:",
        "Use the input 3D layout image as the exact target view.",
        "Match the camera angle, framing, perspective, room boundaries, visible wall panels, crisp corner seams, openings, furniture placement, object count, object color, object shape, object size, depth order, and occlusion shown in that image.",
        "Camera crop lock: keep the same apparent field of view and object pixel scale as the 3D source. Do not zoom out, dolly backward, use a wider lens, shrink the furniture, or reveal additional floor/walls beyond the source framing.",
        "The labels, dark structural edge overlays, and guide marks are only for understanding the layout. They must not appear as labels, black outlines, or guide graphics in the final render.",
        "Existing objects may gain photorealistic texture and finish detail, but not a different color, identity, silhouette, footprint, pose, size, or position.",
        "Do not add extra decor or accessories unless they already exist in the 3D layout or the user explicitly requested them.",
        "Do not add extra walls, remove walls, open closed sides, move wall edges, move doors or windows, or invent columns, stairs, fireplaces, cabinets, wardrobes, beds, desks, tables, sofas, chairs, built-ins, or other furniture.",
    ]
    compiled_layout_prompt = _build_compiled_layout_lock_prompt(snapshot_payload)
    if compiled_layout_prompt:
        prompt_parts.append(compiled_layout_prompt)
    locked_item_prompt = _build_locked_item_prompt(snapshot_payload)
    if locked_item_prompt:
        prompt_parts.append(locked_item_prompt)
    if scene_reference_mode == "target_layout_with_scene_reference":
        prompt_parts.append(
            "Use the previous render only for continuity of realistic finish and object identity. Do not copy its camera, layout, room shape, or color palette if they conflict with the 3D layout."
        )
    elif scene_reference_mode == "scene_reference_camera_transfer":
        camera_payload = _mapping(snapshot_payload.get("camera"))
        if camera_payload:
            prompt_parts.append(
                "Target camera metadata: "
                + json.dumps(
                    dict(camera_payload),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    default=str,
                )
            )

    prompt_parts.append(
        "Do not add, remove, move, resize, rotate, recolor, or replace any visible object unless it has an explicit object reference or color edit. Improve photorealistic appearance without changing the locked structure."
    )
    return "\n".join(prompt_parts)


def _build_compiled_layout_lock_prompt(
    snapshot_payload: Mapping[str, object],
) -> str:
    try:
        compilation = compile_snapshot_prompt(
            snapshot_payload,
            config=SnapshotPromptCompilerConfig(include_minor_decor=True),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Snapshot prompt compilation skipped: %s", exc)
        return ""

    strict_facts = [
        item
        for item in _sequence(compilation.get("strict_layout_facts"))
        if isinstance(item, str) and item.strip()
    ]
    layout_constraints = [
        item
        for item in _sequence(compilation.get("layout_constraints"))
        if isinstance(item, str)
        and item.strip()
        and not item.startswith("Minor decor may be omitted")
    ]
    prompt_parts: list[str] = []
    palette_prompt = _build_exact_palette_lock_prompt(snapshot_payload)
    if palette_prompt:
        prompt_parts.append(palette_prompt)
    if strict_facts:
        prompt_parts.append(
            "Exact layout facts to preserve:\n"
            + "\n".join(f"- {fact}" for fact in strict_facts[:18])
        )
    if layout_constraints:
        prompt_parts.append(
            "Strict visual constraints:\n"
            + "\n".join(f"- {constraint}" for constraint in layout_constraints[:14])
        )
    return "\n".join(prompt_parts)


def _build_exact_palette_lock_prompt(
    snapshot_payload: Mapping[str, object],
) -> str:
    room_payload = _mapping(snapshot_payload.get("room"))
    surface_payload = _mapping(room_payload.get("surfaceColors"))
    surface_lines = []
    for label, key in (
        ("walls", "wallColorHex"),
        ("floor", "floorColorHex"),
        ("ceiling", "ceilingColorHex"),
    ):
        color = _string(surface_payload.get(key))
        if _is_hex_color(color):
            surface_lines.append(f"{label} {color}")

    object_lines = []
    for object_id, payload in _extract_visible_objects(snapshot_payload).items():
        color = _string(payload.get("colorHex"))
        if not _is_hex_color(color):
            continue
        object_lines.append(f"{object_id} ({_visible_object_name(payload)}) {color}")

    if not surface_lines and not object_lines:
        return ""

    prompt_parts = [
        "Exact color palette from the current 3D layout. Preserve these colors in the photorealistic render unless the user changed them in the layout or requested an explicit color edit."
    ]
    if surface_lines:
        prompt_parts.append("Room surfaces: " + "; ".join(surface_lines) + ".")
    if object_lines:
        prompt_parts.append("Visible objects: " + "; ".join(object_lines[:18]) + ".")
    return "\n".join(prompt_parts)


def _build_locked_item_prompt(snapshot_payload: Mapping[str, object]) -> str:
    locked_items = _build_locked_items(snapshot_payload)
    if not locked_items:
        return ""

    lines = [
        "Locked visible object and opening map. Every listed item must remain present, recognizable, and aligned with its screen-space position, scale, silhouette, depth order, and color from the 3D source:"
    ]
    for item in locked_items[:40]:
        item_id = _string(item.get("id"), fallback="unknown")
        kind = _string(item.get("kind"), fallback="object")
        item_type = _string(item.get("type"), fallback="unknown")
        item_label = _string(item.get("label"))
        bbox = item.get("screen_bbox_px")
        bbox_text = ""
        if isinstance(bbox, Mapping):
            bbox_text = " screen_bbox_px=" + json.dumps(
                dict(bbox), ensure_ascii=True, separators=(",", ":")
            )
        dimensions = item.get("dimensions_mm")
        dimensions_text = ""
        if isinstance(dimensions, Mapping):
            dimensions_text = " dimensions_mm=" + json.dumps(
                dict(dimensions),
                ensure_ascii=True,
                separators=(",", ":"),
            )
        color = _string(item.get("color_hex"))
        color_text = f" color={color}" if color else ""
        label_text = f", {item_label}" if item_label and item_label != item_type else ""
        identity_hint = _string(item.get("render_identity_hint"))
        identity_text = f" render_identity={identity_hint}" if identity_hint else ""
        lines.append(
            f"- {kind} {item_id} ({item_type}{label_text})"
            f"{bbox_text}{dimensions_text}{color_text}{identity_text}"
        )
    if len(locked_items) > 40:
        lines.append(f"- plus {len(locked_items) - 40} additional locked item(s).")
    lines.append(
        "Do not omit or blur away supporting shelves, bookcases, TV/media consoles, wardrobes, cabinets, beds, doors, or windows when they are listed here."
    )
    return "\n".join(lines)


def _build_locked_items(
    snapshot_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    locked_items: list[dict[str, object]] = []
    canvas = _mapping(snapshot_payload.get("canvas"))
    canvas_width_px = max(1.0, _float(canvas.get("widthPx"), 1.0))
    canvas_height_px = max(1.0, _float(canvas.get("heightPx"), 1.0))
    for object_id, payload in _extract_visible_objects(snapshot_payload).items():
        dimensions = _mapping(payload.get("dimensionsMm"))
        item: dict[str, object] = {
            "kind": "object",
            "id": object_id,
            "type": _string(payload.get("type"), fallback="object"),
            "label": _visible_object_name(payload),
            "render_identity_hint": _visible_object_render_identity_hint(payload),
            "color_hex": _string(payload.get("colorHex")) or None,
            "screen_bbox_px": _normalized_screen_bbox(payload.get("screenBboxPx")),
            "canvas_width_px": round(canvas_width_px, 2),
            "canvas_height_px": round(canvas_height_px, 2),
            "dimensions_mm": {
                "width": round(_float(dimensions.get("width"), 0.0), 2),
                "depth": round(_float(dimensions.get("depth"), 0.0), 2),
                "height": round(_float(dimensions.get("height"), 0.0), 2),
            },
            "distance_to_camera_m": round(
                _float(payload.get("distanceToCameraM"), 0.0),
                3,
            ),
        }
        locked_items.append(item)

    room = _mapping(snapshot_payload.get("room"))
    openings = _mapping(room.get("openings"))
    for opening_kind in ("doors", "windows"):
        raw_openings = _sequence(openings.get(opening_kind))
        for index, opening in enumerate(raw_openings, start=1):
            if not isinstance(opening, Mapping):
                continue
            opening_id = _string(
                opening.get("id"),
                fallback=f"{opening_kind[:-1]}_{index}",
            )
            locked_items.append(
                {
                    "kind": "opening",
                    "id": f"{opening_kind[:-1]}:{opening_id}",
                    "type": opening_kind[:-1],
                    "screen_bbox_px": _normalized_screen_bbox(
                        opening.get("screenBboxPx")
                    ),
                    "screen_center_px": _normalized_point(
                        opening.get("screenCenterPx")
                    ),
                    "canvas_width_px": round(canvas_width_px, 2),
                    "canvas_height_px": round(canvas_height_px, 2),
                    "segment_mm": opening.get("segmentMm"),
                }
            )
    return locked_items


def _normalized_screen_bbox(value: object) -> dict[str, float] | None:
    bbox = _mapping(value)
    if not bbox:
        return None
    min_x = _float(bbox.get("minX"), math.nan)
    min_y = _float(bbox.get("minY"), math.nan)
    max_x = _float(bbox.get("maxX"), math.nan)
    max_y = _float(bbox.get("maxY"), math.nan)
    width = _float(bbox.get("width"), max_x - min_x)
    height = _float(bbox.get("height"), max_y - min_y)
    if not all(math.isfinite(item) for item in (min_x, min_y, max_x, max_y)):
        return None
    if width <= 0 or height <= 0:
        return None
    return {
        "minX": round(min_x, 2),
        "minY": round(min_y, 2),
        "maxX": round(max_x, 2),
        "maxY": round(max_y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
    }


def _normalized_point(value: object) -> dict[str, float] | None:
    point = _mapping(value)
    if not point:
        return None
    x = _float(point.get("x"), math.nan)
    y = _float(point.get("y"), math.nan)
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return {"x": round(x, 2), "y": round(y, 2)}


def _normalize_edit_operations(
    operations: list[SnapshotEditOperation],
    *,
    visible_objects: Mapping[str, Mapping[str, object]],
) -> list[SnapshotEditOperation]:
    normalized: list[SnapshotEditOperation] = []
    for operation in operations:
        object_id = _string(operation.object_id)
        if not object_id:
            raise ValueError("Edit operation object_id is required.")
        bbox_norm = _normalize_bbox_norm(operation.bbox_norm)
        is_region_operation = bbox_norm is not None
        visible_payload = visible_objects.get(object_id)
        if visible_payload is None and not is_region_operation:
            raise ValueError(
                f"Object {object_id} is not visible in the source image and cannot be edited."
            )

        replacement_image_data_url = (
            _normalize_image_data_url(operation.replacement_image_data_url)
            if operation.replacement_image_data_url
            else None
        )
        target_color = _string(operation.target_color) or None
        prompt = _string(operation.prompt) or None
        operation_type = operation.operation_type
        if operation_type not in {
            "prompt",
            "add",
            "replace",
            "recolor",
            "remove",
            None,
        }:
            operation_type = None
        if operation_type is None:
            operation_type = (
                "replace"
                if replacement_image_data_url
                else "recolor"
                if target_color
                else "prompt"
            )
        if (
            replacement_image_data_url is None
            and target_color is None
            and prompt is None
            and not is_region_operation
        ):
            continue

        object_name = _string(operation.object_name) or (
            _visible_object_name(visible_payload)
            if visible_payload is not None
            else object_id
        )
        normalized.append(
            SnapshotEditOperation(
                object_id=object_id,
                object_name=object_name,
                operation_type=operation_type,
                prompt=prompt,
                bbox_norm=bbox_norm,
                replacement_image_data_url=replacement_image_data_url,
                target_color=target_color,
            )
        )
    return normalized


def _normalize_bbox_norm(
    bbox_norm: Mapping[str, float] | None,
) -> dict[str, float] | None:
    if not bbox_norm:
        return None
    try:
        x = float(bbox_norm.get("x", 0.0))
        y = float(bbox_norm.get("y", 0.0))
        width = float(bbox_norm.get("width", 0.0))
        height = float(bbox_norm.get("height", 0.0))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    width = max(0.0, min(1.0 - x, width))
    height = max(0.0, min(1.0 - y, height))
    if width <= 0.002 or height <= 0.002:
        return None
    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "width": round(width, 4),
        "height": round(height, 4),
    }


def _build_visibility_diagnostics(
    snapshot_payload: Mapping[str, object],
) -> dict[str, object]:
    raw_visible = _sequence(snapshot_payload.get("visibleObjects"))
    raw_all = _sequence(snapshot_payload.get("allObjects"))
    visible_objects = [
        item
        for item in raw_visible
        if isinstance(item, Mapping) and _string(item.get("id"))
    ]
    all_objects = [
        item
        for item in raw_all
        if isinstance(item, Mapping) and _string(item.get("id"))
    ]
    fallback_objects = _fallback_visible_objects_from_all_objects(
        snapshot_payload,
        visible_objects=visible_objects,
    )
    opening_ids_missing_bbox = _opening_ids_missing_screen_bbox(snapshot_payload)
    reasons: list[str] = []
    if not visible_objects and all_objects:
        reasons.append("visible_objects_empty_with_all_objects_present")
    if fallback_objects:
        reasons.append("all_objects_fallback_locked")
    if opening_ids_missing_bbox:
        reasons.append("opening_screen_bbox_missing")
    declared_visible_count = _float(
        snapshot_payload.get("visibleObjectCount"), math.nan
    )
    if math.isfinite(declared_visible_count) and int(declared_visible_count) != len(
        visible_objects
    ):
        reasons.append("visible_object_count_mismatch")

    return {
        "status": "degraded" if reasons else "ok",
        "declared_visible_object_count": (
            int(declared_visible_count)
            if math.isfinite(declared_visible_count)
            else None
        ),
        "visible_object_count": len(visible_objects),
        "all_object_count": len(all_objects),
        "fallback_visible_object_count": len(fallback_objects),
        "fallback_visible_object_ids": [
            _string(item.get("id")) for item in fallback_objects
        ],
        "opening_bbox_missing_count": len(opening_ids_missing_bbox),
        "opening_ids_missing_bbox": opening_ids_missing_bbox,
        "reasons": reasons,
    }


def _snapshot_payload_with_effective_visible_objects(
    snapshot_payload: Mapping[str, object],
    *,
    visibility_diagnostics: Mapping[str, object],
) -> Mapping[str, object]:
    visible_objects = [
        item
        for item in _sequence(snapshot_payload.get("visibleObjects"))
        if isinstance(item, Mapping) and _string(item.get("id"))
    ]
    fallback_objects = _fallback_visible_objects_from_all_objects(
        snapshot_payload,
        visible_objects=visible_objects,
    )
    if not fallback_objects:
        return snapshot_payload

    effective_visible = [dict(item) for item in visible_objects]
    seen_ids = {_string(item.get("id")) for item in effective_visible}
    for item in fallback_objects:
        object_id = _string(item.get("id"))
        if object_id in seen_ids:
            continue
        effective_visible.append(dict(item))
        seen_ids.add(object_id)

    effective_payload = dict(snapshot_payload)
    effective_payload["visibleObjects"] = effective_visible
    effective_payload["visibleObjectCount"] = len(effective_visible)
    effective_payload["visibilityDiagnostics"] = dict(visibility_diagnostics)
    return effective_payload


def _fallback_visible_objects_from_all_objects(
    snapshot_payload: Mapping[str, object],
    *,
    visible_objects: list[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    visible_ids = {_string(item.get("id")) for item in visible_objects}
    all_objects = [
        item
        for item in _sequence(snapshot_payload.get("allObjects"))
        if isinstance(item, Mapping) and _string(item.get("id"))
    ]
    if visible_objects:
        return [
            item
            for item in all_objects
            if _string(item.get("id")) not in visible_ids
            and _normalized_screen_bbox(item.get("screenBboxPx")) is not None
        ]
    return all_objects[:40]


def _opening_ids_missing_screen_bbox(
    snapshot_payload: Mapping[str, object],
) -> list[str]:
    missing: list[str] = []
    room = _mapping(snapshot_payload.get("room"))
    openings = _mapping(room.get("openings"))
    for opening_kind in ("doors", "windows"):
        for index, opening in enumerate(_sequence(openings.get(opening_kind)), start=1):
            if not isinstance(opening, Mapping):
                continue
            if _normalized_screen_bbox(opening.get("screenBboxPx")) is not None:
                continue
            opening_id = _string(
                opening.get("id"),
                fallback=f"{opening_kind[:-1]}_{index}",
            )
            missing.append(f"{opening_kind[:-1]}:{opening_id}")
    return missing


def _normalize_locked_geometry_references(
    references: list[SnapshotLockedGeometryReference],
) -> list[SnapshotLockedGeometryReference]:
    normalized: list[SnapshotLockedGeometryReference] = []
    for reference in references:
        object_id = _string(reference.object_id)
        if not object_id or reference.image_data_url is None:
            continue
        image_data_url = _normalize_image_data_url(reference.image_data_url)
        normalized.append(
            SnapshotLockedGeometryReference(
                object_id=object_id,
                object_name=_string(reference.object_name) or None,
                image_data_url=image_data_url,
                source=_string(reference.source) or None,
                screen_bbox_px=_normalized_screen_bbox(reference.screen_bbox_px),
            )
        )
    return normalized


def _normalize_locked_geometry_atlases(
    atlases: list[SnapshotLockedGeometryAtlas],
) -> list[SnapshotLockedGeometryAtlas]:
    normalized: list[SnapshotLockedGeometryAtlas] = []
    for atlas_index, atlas in enumerate(atlases, start=1):
        if atlas.image_data_url is None:
            continue
        image_data_url = _normalize_image_data_url(atlas.image_data_url)
        normalized_objects: list[SnapshotLockedGeometryAtlasObject] = []
        for item in atlas.objects or []:
            object_id = _string(item.object_id)
            if not object_id:
                continue
            normalized_objects.append(
                SnapshotLockedGeometryAtlasObject(
                    object_id=object_id,
                    object_name=_string(item.object_name) or None,
                    screen_bbox_px=_normalized_screen_bbox(item.screen_bbox_px),
                    atlas_index=int(
                        item.atlas_index or atlas.atlas_index or atlas_index
                    ),
                    cell_index=int(item.cell_index or len(normalized_objects) + 1),
                    cell_label=_string(item.cell_label) or None,
                    cell_bbox_norm=_normalize_bbox_norm(item.cell_bbox_norm),
                )
            )
        if not normalized_objects:
            continue
        normalized.append(
            SnapshotLockedGeometryAtlas(
                atlas_index=int(atlas.atlas_index or atlas_index),
                image_data_url=image_data_url,
                source=_string(atlas.source) or None,
                objects=normalized_objects,
            )
        )
    return normalized


def _serialize_locked_geometry_reference(
    reference: SnapshotLockedGeometryReference,
) -> dict[str, object]:
    return {
        "object_id": reference.object_id,
        "object_name": reference.object_name,
        "source": reference.source,
        "screen_bbox_px": (
            dict(reference.screen_bbox_px)
            if isinstance(reference.screen_bbox_px, Mapping)
            else None
        ),
        "image_provided": reference.image_data_url is not None,
    }


def _serialize_preset_reference_image(
    reference: SnapshotPresetReferenceImage,
) -> dict[str, object]:
    return {
        "group": reference.group,
        "key": reference.key,
        "label": reference.label,
        "image_provided": True,
    }


def _serialize_locked_geometry_atlas(
    atlas: SnapshotLockedGeometryAtlas,
) -> dict[str, object]:
    return {
        "atlas_index": atlas.atlas_index,
        "source": atlas.source,
        "image_provided": atlas.image_data_url is not None,
        "object_count": len(atlas.objects or []),
        "objects": [
            {
                "object_id": item.object_id,
                "object_name": item.object_name,
                "screen_bbox_px": (
                    dict(item.screen_bbox_px)
                    if isinstance(item.screen_bbox_px, Mapping)
                    else None
                ),
                "atlas_index": item.atlas_index,
                "cell_index": item.cell_index,
                "cell_label": item.cell_label,
                "cell_bbox_norm": (
                    dict(item.cell_bbox_norm)
                    if isinstance(item.cell_bbox_norm, Mapping)
                    else None
                ),
            }
            for item in atlas.objects or []
        ],
    }


def _geometry_lock_metadata(
    *,
    locked_items: list[dict[str, object]],
    atlases: list[SnapshotLockedGeometryAtlas],
    references: list[SnapshotLockedGeometryReference],
) -> dict[str, object]:
    atlas_object_ids = {
        item.object_id
        for atlas in atlases
        for item in (atlas.objects or [])
        if item.object_id
    }
    legacy_object_ids = {reference.object_id for reference in references}
    locked_object_ids = {
        _string(item.get("id"))
        for item in locked_items
        if item.get("kind") == "object" and _string(item.get("id"))
    }
    omitted_ids = sorted(locked_object_ids - atlas_object_ids)
    if atlases:
        policy = "all_visible_objects_atlas"
    elif references:
        policy = "legacy_per_object_references"
    else:
        policy = "metadata_only"
    return {
        "locked_geometry_atlas_count": len(atlases),
        "locked_geometry_object_count": len(atlas_object_ids),
        "geometry_atlas_omitted_object_ids": omitted_ids,
        "legacy_locked_geometry_object_count": len(legacy_object_ids),
        "geometry_lock_policy": policy,
    }


def _extract_visible_objects(
    snapshot_payload: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_visible = snapshot_payload.get("visibleObjects")
    if not isinstance(raw_visible, list):
        return {}

    visible: dict[str, Mapping[str, object]] = {}
    for item in raw_visible:
        if not isinstance(item, Mapping):
            continue
        object_id = _string(item.get("id"))
        if object_id:
            visible[object_id] = item
    return visible


def _visible_object_name(payload: Mapping[str, object]) -> str:
    render_identity_label = _string(
        payload.get("renderIdentityLabel") or payload.get("render_identity_label")
    )
    if render_identity_label:
        return render_identity_label
    if _string(payload.get("type")).lower() == "tv_console":
        return _TV_CONSOLE_RENDER_IDENTITY_LABEL
    for key in (
        "label",
        "assetId",
        "asset_id",
        "rawAssetId",
        "raw_asset_id",
        "type",
        "canonicalType",
        "canonical_type",
        "raw_type",
    ):
        value = _string(payload.get(key))
        if value:
            return value.replace("_", " ")
    object_id = _string(payload.get("id"))
    return object_id or "object"


def _visible_object_render_identity_hint(payload: Mapping[str, object]) -> str | None:
    hint = _string(
        payload.get("renderIdentityHint") or payload.get("render_identity_hint")
    )
    if hint:
        return hint
    object_type = _string(payload.get("type")).lower()
    if object_type == "tv_console":
        return _TV_CONSOLE_RENDER_IDENTITY_HINT
    return None


def _serialize_edit_operation(operation: SnapshotEditOperation) -> dict[str, object]:
    return {
        "object_id": operation.object_id,
        "object_name": operation.object_name,
        "operation_type": operation.operation_type,
        "prompt": operation.prompt,
        "bbox_provided": operation.bbox_norm is not None,
        "replacement_image_provided": operation.replacement_image_data_url is not None,
        "target_color": operation.target_color,
    }


def _resolve_canvas_aspect_ratio(snapshot_payload: Mapping[str, object]) -> str:
    canvas_payload = _mapping(snapshot_payload.get("canvas"))
    width = _positive_float(canvas_payload.get("widthPx"), fallback=1.0)
    height = _positive_float(canvas_payload.get("heightPx"), fallback=1.0)
    ratio = width / max(height, 1.0)
    best_label = "1:1"
    best_distance = math.inf
    for label, candidate_ratio in _SUPPORTED_ASPECT_RATIOS:
        distance = abs(candidate_ratio - ratio)
        if distance < best_distance:
            best_label = label
            best_distance = distance
    return best_label


def _find_snapshot_image_path(snapshot_path: Path) -> Path | None:
    for suffix in _SUPPORTED_IMAGE_EXTENSIONS:
        candidate = snapshot_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _encode_file_as_data_url(path: Path) -> str:
    image_bytes = path.read_bytes()
    mime_type = _guess_image_mime_type(image_bytes)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def _normalize_image_data_url(data_url: str | None) -> str:
    if data_url is None:
        raise ValueError("Image data URL is required.")
    normalized = data_url.strip()
    if not normalized.startswith("data:image/") or "," not in normalized:
        raise ValueError("Image data URL must be a valid image data URL.")
    _, _, image_base64 = normalized.partition(",")
    try:
        base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image data URL contains invalid base64 image data.") from exc
    return normalized


def _extract_data_url_mime_type(data_url: str) -> str:
    prefix, _, _ = data_url.partition(",")
    mime_section = prefix.removeprefix("data:")
    mime_type, _, _ = mime_section.partition(";")
    return mime_type or "application/octet-stream"


def _guess_image_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return "application/octet-stream"


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _is_hex_color(value: str | None) -> bool:
    if not value or len(value) != 7 or not value.startswith("#"):
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value[1:])


def _positive_float(value: object, *, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if numeric <= 0:
        return fallback
    return numeric


def _float(value: object, fallback: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric):
        return fallback
    return numeric


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean in %s=%r. Using default %s.", name, raw, default)
    return default


def _read_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer in %s=%r. Using default %s.", name, raw, default
        )
        return default


def _string(value: object, *, fallback: str = "") -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return fallback


def _secret_string(value: object) -> str | None:
    normalized = _string(value)
    if not normalized:
        return None
    if normalized.startswith("${") and normalized.endswith("}"):
        return None
    if normalized.lower().startswith("your-"):
        return None
    return normalized
