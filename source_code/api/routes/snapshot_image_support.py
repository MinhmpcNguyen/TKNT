from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, NoReturn

from fastapi import HTTPException
from PIL import Image
from psycopg import Error as PsycopgError
from pydantic import BaseModel, Field, ValidationError

from db.models import UserAccount
from pipeline.image_flow_edit_masks import (
    object_scope_edit_mask_image,
    should_use_object_scope_edit_mask,
)
from pipeline.image_flow_logging import log_image_flow_event
from pipeline.image_flow_preflight import (
    BBoxNorm as PreflightBBoxNorm,
    EditOperationConfig as PreflightEditOperationConfig,
    compile_visual_preflight_prompt,
    run_visual_preflight,
)
from pipeline.semantic_layout_locks import semantic_layout_guide_for_snapshot_image
from pipeline.snapshot_image_renderer import (
    SnapshotEditOperation,
    SnapshotImageRenderConfig,
    SnapshotLockedGeometryAtlas,
    SnapshotLockedGeometryAtlasObject,
    SnapshotLockedGeometryReference,
    render_snapshot_image,
    render_snapshot_image_from_path,
)
from services.user_content_service import UserContentService

SnapshotImageQuality = Literal["low", "medium", "high"]
SnapshotImageSize = Literal["1024x1024", "1024x1536", "1536x1024"]
SnapshotImageRenderFlow = Literal[
    "default",
    "single_edit_edge_silhouette_guided",
    "edit_object_scope_preflight",
]
_SINGLE_EDIT_EDGE_GUIDED_PROMPT = (
    "Photorealize this exact 3D source view. Preserve the camera, crop, room "
    "shell, openings, object count, object positions, object scale, silhouettes, "
    "colors, and depth order. Do not add windows, doors, decor, furniture, or "
    "foreground objects."
)
_IMAGE_FLOW_PREFLIGHT_DIR = Path("generated/image_flow_preflight")
_IMAGE_FLOW_INPUT_DEBUG_DIR = Path("generated/image_flow_inputs")


class SnapshotImagePresetSelection(BaseModel):
    style: str | None = None
    lighting: str | None = None
    scenery: str | None = None


class SnapshotImageEditOperationRequest(BaseModel):
    object_id: str
    object_name: str | None = None
    operation_type: Literal["prompt", "add", "replace", "recolor", "remove"] | None = (
        None
    )
    prompt: str | None = None
    bbox_norm: dict[str, float] | None = None
    replacement_image_data_url: str | None = None
    target_color: str | None = None


class SnapshotImageLockedGeometryReferenceRequest(BaseModel):
    object_id: str
    object_name: str | None = None
    image_data_url: str | None = None
    source: str | None = None
    screen_bbox_px: dict[str, float] | None = None


class SnapshotImageLockedGeometryAtlasObjectRequest(BaseModel):
    object_id: str
    object_name: str | None = None
    screen_bbox_px: dict[str, float] | None = None
    atlas_index: int = 0
    cell_index: int = 0
    cell_label: str | None = None
    cell_bbox_norm: dict[str, float] | None = None


class SnapshotImageLockedGeometryAtlasRequest(BaseModel):
    atlas_index: int = 0
    image_data_url: str | None = None
    source: str | None = None
    objects: list[SnapshotImageLockedGeometryAtlasObjectRequest] = Field(
        default_factory=list
    )


class SnapshotImageRenderRequest(BaseModel):
    snapshot_payload: dict[str, Any] | None = None
    snapshot_path: str | None = None
    snapshot_image_data_url: str | None = None
    depth_map_image_data_url: str | None = None
    edge_map_image_data_url: str | None = None
    segmentation_map_image_data_url: str | None = None
    layout_reference_image_data_url: str | None = None
    annotated_reference_image_data_url: str | None = None
    scene_reference_image_data_url: str | None = None
    scene_reference_mode: Literal[
        "none",
        "target_layout_with_scene_reference",
        "scene_reference_camera_transfer",
    ] = "none"
    user_prompt: str | None = None
    render_mode: Literal["generate", "edit", "camera_rerender"] = "generate"
    render_flow: SnapshotImageRenderFlow = "default"
    preset_selection: SnapshotImagePresetSelection = Field(
        default_factory=lambda: SnapshotImagePresetSelection()
    )
    root_layout_id: str | None = None
    case_id: str | None = None
    option_id: str | None = None
    edit_operations: list[SnapshotImageEditOperationRequest] = Field(
        default_factory=list
    )
    locked_geometry_references: list[SnapshotImageLockedGeometryReferenceRequest] = (
        Field(default_factory=list)
    )
    locked_geometry_atlases: list[SnapshotImageLockedGeometryAtlasRequest] = Field(
        default_factory=list
    )
    edit_source_image_data_url: str | None = None
    image_quality: SnapshotImageQuality | None = None
    image_size: SnapshotImageSize | None = None


class SnapshotImageRenderResponse(BaseModel):
    render: dict[str, Any]
    saved_render: dict[str, Any] | None = None


class SnapshotImagePresetOptionResponse(BaseModel):
    label: str
    prompt_suffix: str | None = None
    reference_image: str | None = None


class SnapshotImagePresetsResponse(BaseModel):
    styles: dict[str, SnapshotImagePresetOptionResponse]
    lights: dict[str, SnapshotImagePresetOptionResponse]
    sceneries: dict[str, SnapshotImagePresetOptionResponse]


def _append_snapshot_flow_prompt(base: str | None, addition: str | None) -> str | None:
    base_text = (base or "").strip()
    addition_text = (addition or "").strip()
    if not base_text:
        return addition_text or None
    if not addition_text:
        return base_text
    return f"{base_text}\n\n{addition_text}"


def _render_config_for_flow(
    flow: SnapshotImageRenderFlow,
) -> SnapshotImageRenderConfig:
    base_config = SnapshotImageRenderConfig.from_env()
    if flow == "single_edit_edge_silhouette_guided":
        return replace(
            base_config,
            simple_prompt_mode=False,
            text_contract_mode=False,
            text_contract_object_limit=0,
            include_layout_2d_reference=False,
            include_annotated_reference=True,
            include_geometry_references=False,
            include_locked_geometry_references=False,
            include_preset_reference_images=False,
            strict_layout_mode=False,
        )
    if flow == "edit_object_scope_preflight":
        return replace(
            base_config,
            simple_prompt_mode=True,
            text_contract_mode=True,
            text_contract_object_limit=10,
            include_layout_2d_reference=False,
            include_annotated_reference=True,
            include_geometry_references=False,
            include_locked_geometry_references=False,
            include_preset_reference_images=False,
            include_edit_operation_dsl=True,
            disable_edit_mask=True,
            strict_layout_mode=False,
        )
    return base_config


def _snapshot_payload_for_flow(req: SnapshotImageRenderRequest) -> Mapping[str, object]:
    if req.snapshot_payload is not None:
        return req.snapshot_payload
    if not req.snapshot_path:
        raise ValueError("Provide either snapshot_payload or snapshot_path.")
    path = Path(req.snapshot_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Snapshot file must contain a JSON object.")
    return payload


def _decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    header, separator, encoded = data_url.partition(",")
    if separator != "," or not header.startswith("data:image/"):
        raise ValueError("Expected an image data URL.")
    mime_type = header.removeprefix("data:").split(";", maxsplit=1)[0]
    try:
        return base64.b64decode(encoded, validate=True), mime_type
    except binascii.Error as exc:
        raise ValueError("Image data URL is not valid base64.") from exc


def _image_data_url_size(data_url: str) -> tuple[int, int]:
    image_bytes, _ = _decode_image_data_url(data_url)
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.size
    except OSError as exc:
        raise ValueError("Unable to decode image data URL.") from exc


def _png_data_url_from_image(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _preflight_image_path(
    *,
    image_data_url: str,
    trace_id: str,
    label: str,
) -> Path:
    image_bytes, mime_type = _decode_image_data_url(image_data_url)
    extension = "jpg" if mime_type in {"image/jpeg", "image/jpg"} else "png"
    safe_label = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in label
    )
    path = (
        _IMAGE_FLOW_PREFLIGHT_DIR / "sources" / f"{trace_id}_{safe_label}.{extension}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    return path


def _debug_input_image_paths(
    *,
    trace_id: str,
    render_flow: SnapshotImageRenderFlow,
    images: Mapping[str, str | None],
    atlases: list[SnapshotLockedGeometryAtlas],
) -> dict[str, str]:
    saved_paths: dict[str, str] = {}
    failures: dict[str, str] = {}
    for label, image_data_url in images.items():
        if image_data_url is None:
            continue
        try:
            saved_paths[label] = str(
                _debug_input_image_path(
                    image_data_url=image_data_url,
                    trace_id=trace_id,
                    label=label,
                )
            )
        except ValueError as exc:
            failures[label] = str(exc)
    for index, atlas in enumerate(atlases, start=1):
        if atlas.image_data_url is None:
            continue
        label = f"locked_geometry_atlas_{index}"
        try:
            saved_paths[label] = str(
                _debug_input_image_path(
                    image_data_url=atlas.image_data_url,
                    trace_id=trace_id,
                    label=label,
                )
            )
        except ValueError as exc:
            failures[label] = str(exc)
    if saved_paths or failures:
        log_image_flow_event(
            "snapshot_image.debug_input_files",
            {
                "trace_id": trace_id,
                "render_flow": render_flow,
                "saved_paths": saved_paths,
                "failures": failures,
            },
        )
    return saved_paths


def _debug_input_image_path(
    *,
    image_data_url: str,
    trace_id: str,
    label: str,
) -> Path:
    image_bytes, mime_type = _decode_image_data_url(image_data_url)
    extension = "jpg" if mime_type in {"image/jpeg", "image/jpg"} else "png"
    safe_label = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in label
    )
    path = _IMAGE_FLOW_INPUT_DEBUG_DIR / trace_id / f"{safe_label}.{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    return path


def _preflight_bbox_norm(
    bbox_norm: Mapping[str, float] | None,
) -> PreflightBBoxNorm | None:
    if bbox_norm is None:
        return None
    try:
        return PreflightBBoxNorm(
            x=float(bbox_norm["x"]),
            y=float(bbox_norm["y"]),
            width=float(bbox_norm["width"]),
            height=float(bbox_norm["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "edit_operations[].bbox_norm must include x, y, width, height."
        ) from exc


def _preflight_edit_operations(
    edit_operations: list[SnapshotEditOperation],
) -> list[PreflightEditOperationConfig]:
    return [
        PreflightEditOperationConfig(
            object_id=operation.object_id,
            object_name=operation.object_name,
            operation_type=operation.operation_type,
            prompt=operation.prompt,
            bbox_norm=_preflight_bbox_norm(operation.bbox_norm),
            replacement_image=(
                Path(f"reference_{index}.png")
                if operation.replacement_image_data_url
                else None
            ),
            target_color=operation.target_color,
        )
        for index, operation in enumerate(edit_operations, start=1)
    ]


def _edit_object_scope_preflight_prompt(
    *,
    req: SnapshotImageRenderRequest,
    source_payload: Mapping[str, object],
    edit_operations: list[SnapshotEditOperation],
    trace_id: str,
) -> tuple[str, dict[str, Any]]:
    source_image_data_url = (
        req.annotated_reference_image_data_url
        or req.edit_source_image_data_url
        or req.snapshot_image_data_url
    )
    if not source_image_data_url:
        raise ValueError("edit_object_scope_preflight requires an edit source image.")
    source_image = _preflight_image_path(
        image_data_url=source_image_data_url,
        trace_id=trace_id,
        label="edit_object_scope",
    )
    preflight_operations = _preflight_edit_operations(edit_operations)
    result = run_visual_preflight(
        source_image=source_image,
        source_payload=source_payload,
        include_metadata=True,
        report_path=_IMAGE_FLOW_PREFLIGHT_DIR / "reports" / f"{trace_id}.json",
        cache_dir=_IMAGE_FLOW_PREFLIGHT_DIR / "cache",
        profile="edit_object_scope",
        edit_operations=preflight_operations,
    )
    if not result.source_valid:
        raise ValueError(
            "Visual preflight rejected the edit source render: "
            f"{result.reason or 'source_valid=false'}"
        )
    prompt = compile_visual_preflight_prompt(
        result,
        edit_operations=preflight_operations,
    )
    return prompt, result.to_json()


def _object_scope_mask_for_request(
    *,
    source_image_data_url: str,
    source_payload: Mapping[str, object],
    edit_operations: list[SnapshotEditOperation],
) -> tuple[str | None, dict[str, Any]]:
    if not should_use_object_scope_edit_mask(edit_operations):
        operation_types = sorted(
            {operation.operation_type or "prompt" for operation in edit_operations}
        )
        return None, {
            "enabled": False,
            "image": None,
            "reason": "disabled_for_stable_camera_operation",
            "operation_types": operation_types,
            "mask_enabled_operation_types": ["add", "replace"],
            "mask_semantics": (
                "transparent_pixels_are_editable_opaque_pixels_are_preserved"
            ),
        }

    image_size = _image_data_url_size(source_image_data_url)
    mask, metadata = object_scope_edit_mask_image(
        image_size=image_size,
        source_payload=source_payload,
        edit_operations=edit_operations,
    )
    if not bool(metadata.get("enabled")):
        return None, dict(metadata)
    return _png_data_url_from_image(mask), {
        **metadata,
        "image": "inline_data_url",
    }


def _apply_render_flow_metadata(
    render: dict[str, Any],
    *,
    render_flow: SnapshotImageRenderFlow,
    visual_preflight: dict[str, Any] | None,
    object_scope_edit_mask: dict[str, Any] | None,
    semantic_layout_guide: dict[str, Any] | None,
) -> dict[str, Any]:
    request_payload = render.get("request")
    if isinstance(request_payload, dict):
        request_payload["render_flow"] = render_flow
    metadata_payload = render.get("metadata")
    if isinstance(metadata_payload, dict):
        metadata_payload["render_flow"] = render_flow
        if visual_preflight is not None:
            metadata_payload["visual_preflight"] = visual_preflight
        if object_scope_edit_mask is not None:
            metadata_payload["object_scope_edit_mask"] = object_scope_edit_mask
        if semantic_layout_guide is not None:
            metadata_payload["semantic_layout_guide"] = semantic_layout_guide
    return render


def _render_snapshot_image_for_request(
    req: SnapshotImageRenderRequest,
    *,
    trace_id: str,
    streamed: bool,
    stream_event_handler: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, Any]:
    if req.snapshot_payload is None and not req.snapshot_path:
        log_image_flow_event(
            "snapshot_image.api_error",
            {
                "trace_id": trace_id,
                "status_code": 400,
                "error_message": "Provide either snapshot_payload or snapshot_path.",
            },
        )
        raise HTTPException(
            status_code=400,
            detail="Provide either snapshot_payload or snapshot_path.",
        )

    edit_operations = [
        SnapshotEditOperation(
            object_id=item.object_id,
            object_name=item.object_name,
            operation_type=item.operation_type,
            prompt=item.prompt,
            bbox_norm=item.bbox_norm,
            replacement_image_data_url=item.replacement_image_data_url,
            target_color=item.target_color,
        )
        for item in req.edit_operations
    ]
    locked_geometry_references = [
        SnapshotLockedGeometryReference(
            object_id=item.object_id,
            object_name=item.object_name,
            image_data_url=item.image_data_url,
            source=item.source,
            screen_bbox_px=item.screen_bbox_px,
        )
        for item in req.locked_geometry_references
    ]
    locked_geometry_atlases = [
        SnapshotLockedGeometryAtlas(
            atlas_index=atlas.atlas_index,
            image_data_url=atlas.image_data_url,
            source=atlas.source,
            objects=[
                SnapshotLockedGeometryAtlasObject(
                    object_id=item.object_id,
                    object_name=item.object_name,
                    screen_bbox_px=item.screen_bbox_px,
                    atlas_index=item.atlas_index,
                    cell_index=item.cell_index,
                    cell_label=item.cell_label,
                    cell_bbox_norm=item.cell_bbox_norm,
                )
                for item in atlas.objects
            ],
        )
        for atlas in req.locked_geometry_atlases
    ]
    preset_selection = req.preset_selection.model_dump()
    render_config = _render_config_for_flow(req.render_flow)
    effective_render_mode = req.render_mode
    effective_user_prompt = req.user_prompt
    effective_edit_source_image_data_url = req.edit_source_image_data_url
    effective_annotated_reference_image_data_url = (
        req.annotated_reference_image_data_url
    )
    layout_materialization_mode = False
    layout_guard_mask_image_data_url: str | None = None
    visual_preflight: dict[str, Any] | None = None
    object_scope_edit_mask: dict[str, Any] | None = None
    semantic_layout_guide: dict[str, Any] | None = None
    semantic_layout_guide_image_data_url: str | None = None

    if req.render_flow == "single_edit_edge_silhouette_guided":
        effective_render_mode = "edit"
        effective_edit_source_image_data_url = (
            req.edit_source_image_data_url or req.snapshot_image_data_url
        )
        layout_materialization_mode = True
        source_payload = _snapshot_payload_for_flow(req)
        effective_user_prompt = _append_snapshot_flow_prompt(
            effective_user_prompt,
            _SINGLE_EDIT_EDGE_GUIDED_PROMPT,
        )
        if not effective_edit_source_image_data_url:
            raise ValueError(
                "single_edit_edge_silhouette_guided requires snapshot_image_data_url."
            )
        if not req.annotated_reference_image_data_url:
            raise ValueError(
                "single_edit_edge_silhouette_guided requires "
                "instance outline overlay as annotated_reference_image_data_url."
            )
        guide = semantic_layout_guide_for_snapshot_image(
            snapshot_payload=source_payload,
            source_image_data_url=effective_edit_source_image_data_url,
        )
        semantic_layout_guide_image_data_url = guide.image_data_url
        semantic_layout_guide = guide.metadata
        semantic_layout_guide_used_as_fallback = False
        effective_user_prompt = _append_snapshot_flow_prompt(
            effective_user_prompt,
            guide.prompt,
        )
        log_image_flow_event(
            "snapshot_image.semantic_layout_guide",
            {
                "trace_id": trace_id,
                "render_flow": req.render_flow,
                "metadata": semantic_layout_guide,
                "preserved_ui_annotated_reference": bool(
                    req.annotated_reference_image_data_url
                    and semantic_layout_guide_image_data_url
                ),
                "semantic_layout_guide_used_as_fallback": (
                    semantic_layout_guide_used_as_fallback
                ),
            },
        )
    elif req.render_flow == "edit_object_scope_preflight":
        effective_render_mode = "edit"
        if not effective_edit_source_image_data_url:
            raise ValueError(
                "edit_object_scope_preflight requires edit_source_image_data_url."
            )
        source_payload = _snapshot_payload_for_flow(req)
        preflight_prompt, visual_preflight = _edit_object_scope_preflight_prompt(
            req=req,
            source_payload=source_payload,
            edit_operations=edit_operations,
            trace_id=trace_id,
        )
        effective_user_prompt = _append_snapshot_flow_prompt(
            effective_user_prompt,
            preflight_prompt,
        )
        layout_guard_mask_image_data_url, object_scope_edit_mask = (
            _object_scope_mask_for_request(
                source_image_data_url=effective_edit_source_image_data_url,
                source_payload=source_payload,
                edit_operations=edit_operations,
            )
        )
        if layout_guard_mask_image_data_url is not None:
            render_config = replace(render_config, disable_edit_mask=False)

    _debug_input_image_paths(
        trace_id=trace_id,
        render_flow=req.render_flow,
        images={
            "snapshot_image": req.snapshot_image_data_url,
            "edit_source_image": effective_edit_source_image_data_url,
            "ui_annotated_reference": req.annotated_reference_image_data_url,
            "semantic_layout_guide": semantic_layout_guide_image_data_url,
            "annotated_reference": effective_annotated_reference_image_data_url,
            "depth_map": req.depth_map_image_data_url,
            "edge_map": req.edge_map_image_data_url,
            "segmentation_map": req.segmentation_map_image_data_url,
            "layout_reference": req.layout_reference_image_data_url,
            "layout_guard_mask": layout_guard_mask_image_data_url,
            "scene_reference": req.scene_reference_image_data_url,
        },
        atlases=locked_geometry_atlases,
    )

    if req.snapshot_payload is not None:
        if not req.snapshot_image_data_url:
            log_image_flow_event(
                "snapshot_image.api_error",
                {
                    "trace_id": trace_id,
                    "status_code": 400,
                    "error_message": (
                        "Provide snapshot_image_data_url when rendering from "
                        "snapshot_payload."
                    ),
                },
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "Provide snapshot_image_data_url when rendering from "
                    "snapshot_payload."
                ),
            )
        render = render_snapshot_image(
            req.snapshot_payload,
            snapshot_image_data_url=req.snapshot_image_data_url,
            depth_map_image_data_url=req.depth_map_image_data_url,
            edge_map_image_data_url=req.edge_map_image_data_url,
            segmentation_map_image_data_url=req.segmentation_map_image_data_url,
            layout_reference_image_data_url=req.layout_reference_image_data_url,
            annotated_reference_image_data_url=(
                effective_annotated_reference_image_data_url
            ),
            layout_guard_mask_image_data_url=layout_guard_mask_image_data_url,
            layout_materialization_mode=layout_materialization_mode,
            scene_reference_image_data_url=req.scene_reference_image_data_url,
            scene_reference_mode=req.scene_reference_mode,
            user_prompt=effective_user_prompt,
            render_mode=effective_render_mode,
            preset_selection=preset_selection,
            edit_operations=edit_operations,
            locked_geometry_references=locked_geometry_references,
            locked_geometry_atlases=locked_geometry_atlases,
            edit_source_image_data_url=effective_edit_source_image_data_url,
            image_quality=req.image_quality,
            image_size=req.image_size,
            streamed=streamed,
            stream_event_handler=stream_event_handler,
            config=render_config,
            trace_id=trace_id,
        )
        return _apply_render_flow_metadata(
            render,
            render_flow=req.render_flow,
            visual_preflight=visual_preflight,
            object_scope_edit_mask=object_scope_edit_mask,
            semantic_layout_guide=semantic_layout_guide,
        )
    render = render_snapshot_image_from_path(
        req.snapshot_path or "",
        snapshot_image_data_url=req.snapshot_image_data_url,
        depth_map_image_data_url=req.depth_map_image_data_url,
        edge_map_image_data_url=req.edge_map_image_data_url,
        segmentation_map_image_data_url=req.segmentation_map_image_data_url,
        layout_reference_image_data_url=req.layout_reference_image_data_url,
        annotated_reference_image_data_url=(
            effective_annotated_reference_image_data_url
        ),
        layout_guard_mask_image_data_url=layout_guard_mask_image_data_url,
        layout_materialization_mode=layout_materialization_mode,
        scene_reference_image_data_url=req.scene_reference_image_data_url,
        scene_reference_mode=req.scene_reference_mode,
        user_prompt=effective_user_prompt,
        render_mode=effective_render_mode,
        preset_selection=preset_selection,
        edit_operations=edit_operations,
        locked_geometry_references=locked_geometry_references,
        locked_geometry_atlases=locked_geometry_atlases,
        edit_source_image_data_url=effective_edit_source_image_data_url,
        image_quality=req.image_quality,
        image_size=req.image_size,
        streamed=streamed,
        stream_event_handler=stream_event_handler,
        config=render_config,
        trace_id=trace_id,
    )
    return _apply_render_flow_metadata(
        render,
        render_flow=req.render_flow,
        visual_preflight=visual_preflight,
        object_scope_edit_mask=object_scope_edit_mask,
        semantic_layout_guide=semantic_layout_guide,
    )


def _resolve_render_layout_context_or_raise(
    *,
    req: SnapshotImageRenderRequest,
    current_user: UserAccount | None,
    user_content_service: UserContentService,
    trace_id: str,
) -> Any | None:
    if current_user is None:
        return None
    render_layout_context = user_content_service.resolve_render_layout_context(
        user=current_user,
        root_layout_id=req.root_layout_id,
        case_id=req.case_id,
        option_id=req.option_id,
    )
    if render_layout_context is not None:
        return render_layout_context
    log_image_flow_event(
        "snapshot_image.layout_context_error",
        {
            "trace_id": trace_id,
            "status_code": 404,
            "root_layout_id": req.root_layout_id,
            "case_id": req.case_id,
            "option_id": req.option_id,
            "error_message": "Render layout option not found.",
        },
    )
    raise HTTPException(status_code=404, detail="Render layout option not found.")


def _save_snapshot_render_for_request(
    *,
    render: dict[str, Any],
    current_user: UserAccount | None,
    render_layout_context: Any | None,
    user_content_service: UserContentService,
    trace_id: str,
) -> dict[str, Any] | None:
    saved_render: dict[str, Any] | None = None
    if current_user is not None:
        if render_layout_context is None:
            raise HTTPException(
                status_code=500,
                detail="Render layout context was not resolved.",
            )
        try:
            metadata = render.get("metadata") if isinstance(render, dict) else None
            metadata_payload = metadata if isinstance(metadata, dict) else {}
            saved = user_content_service.save_snapshot_render_from_data_url(
                user=current_user,
                image_data_url=str(render["image"]["data_url"]),
                model_name=str(render["models"]["image_model_name"]),
                prompt=str(render["request"]["user_prompt"]),
                negative_prompt=None,
                meta={
                    "user_prompt": str(render["request"]["user_prompt"]),
                    "raw_user_prompt": render["request"].get("raw_user_prompt"),
                    "render_mode": str(
                        render["request"].get("render_mode", "generate")
                    ),
                    "render_flow": render["request"].get("render_flow", "default"),
                    "layout_lock_mode": render["request"].get("layout_lock_mode"),
                    "text_contract_used": bool(
                        render["request"].get("text_contract_used", False)
                    ),
                    "text_contract_object_limit": render["request"].get(
                        "text_contract_object_limit"
                    ),
                    "object_memory_used": render["request"].get("object_memory_used"),
                    "locked_items": render["request"].get("locked_items"),
                    "visibility_diagnostics": render["request"].get(
                        "visibility_diagnostics"
                    ),
                    "locked_geometry_references": render["request"].get(
                        "locked_geometry_references"
                    ),
                    "locked_geometry_atlases": render["request"].get(
                        "locked_geometry_atlases"
                    ),
                    "locked_geometry_atlas_count": render["request"].get(
                        "locked_geometry_atlas_count"
                    ),
                    "locked_geometry_object_count": render["request"].get(
                        "locked_geometry_object_count"
                    ),
                    "geometry_atlas_omitted_object_ids": render["request"].get(
                        "geometry_atlas_omitted_object_ids"
                    ),
                    "geometry_lock_policy": render["request"].get(
                        "geometry_lock_policy"
                    ),
                    "preset_selection": render["request"].get("preset_selection"),
                    "edit_operations": render["request"].get("edit_operations"),
                    "aspect_ratio": str(render["image"]["aspect_ratio"]),
                    "source_image_mime_type": str(
                        render["request"]["source_image_mime_type"]
                    ),
                    "camera": metadata_payload.get("camera"),
                    "visible_objects": metadata_payload.get("visible_objects"),
                    "visible_object_ids": metadata_payload.get("visible_object_ids"),
                    "layout_reference_enabled": bool(
                        render["request"]["layout_reference_enabled"]
                    ),
                    "geometry_references_used": render["request"].get(
                        "geometry_references_used"
                    ),
                    "geometry_reference_image_mime_types": render["request"].get(
                        "geometry_reference_image_mime_types"
                    ),
                    "validation": metadata_payload.get("validation"),
                    "visual_preflight": metadata_payload.get("visual_preflight"),
                    "semantic_layout_guide": metadata_payload.get(
                        "semantic_layout_guide"
                    ),
                    "object_scope_edit_mask": metadata_payload.get(
                        "object_scope_edit_mask"
                    ),
                    "retry_count": render["image"].get("retry_count"),
                    "usage": render["image"].get("usage"),
                    "partial_image_count": render["image"].get("partial_image_count"),
                    "image_quality_requested": render["request"].get(
                        "image_quality_requested"
                    ),
                    "image_size_requested": render["request"].get(
                        "image_size_requested"
                    ),
                    "stream_partial_images": render["request"].get(
                        "stream_partial_images"
                    ),
                    "streamed": render["request"].get("streamed"),
                    "layout_reference_used": bool(
                        render["request"]["layout_reference_used"]
                    ),
                    "annotated_reference_used": bool(
                        render["request"]["annotated_reference_used"]
                    ),
                    "edit_region_guide_used": bool(
                        render["request"].get("edit_region_guide_used", False)
                    ),
                    "camera_transfer_reference_used": bool(
                        render["request"].get("camera_transfer_reference_used", False)
                    ),
                    "scene_reference_mode": str(
                        render["request"].get("scene_reference_mode", "none")
                    ),
                    "scene_reference_used": bool(
                        render["request"].get("scene_reference_used", False)
                    ),
                    "reference_only_camera_transfer_used": bool(
                        render["request"].get(
                            "reference_only_camera_transfer_used",
                            False,
                        )
                    ),
                    "root_layout_id": render_layout_context.root_layout_id,
                    "case_id": render_layout_context.case_id,
                    "option_id": render_layout_context.option_id,
                    "option_index": render_layout_context.option_index,
                },
                root_layout_id=render_layout_context.root_layout_id,
                case_id=render_layout_context.case_id,
                option_id=render_layout_context.option_id,
                option_index=render_layout_context.option_index,
            )
            saved_render = user_content_service.serialize_generated_render(
                saved,
                file_url=f"/account/renders/{saved.id}/file",
            )
            log_image_flow_event(
                "snapshot_image.saved_render.output",
                {
                    "trace_id": trace_id,
                    "render_id": str(saved.id),
                    "model_name": saved.model_name,
                    "mime_type": saved.mime_type,
                    "root_layout_id": str(saved.root_layout_id)
                    if saved.root_layout_id
                    else None,
                    "case_id": str(saved.case_id) if saved.case_id else None,
                    "option_id": saved.option_id,
                    "meta": dict(saved.meta or {}),
                },
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            binascii.Error,
            OSError,
            PsycopgError,
            ValidationError,
        ) as exc:
            log_image_flow_event(
                "snapshot_image.saved_render.error",
                {
                    "trace_id": trace_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            saved_render = None

    return saved_render


def _raise_snapshot_render_http_exception(trace_id: str, exc: Exception) -> NoReturn:
    if isinstance(exc, FileNotFoundError):
        status_code = 404
    elif isinstance(exc, ValueError):
        status_code = 400
    elif isinstance(exc, RuntimeError):
        status_code = 502
    else:
        status_code = 500
    log_image_flow_event(
        "snapshot_image.api_error",
        {
            "trace_id": trace_id,
            "status_code": status_code,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )
    raise HTTPException(status_code=status_code, detail=str(exc)) from exc
