from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Literal, NoReturn
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from api.deps import get_optional_current_user
from config import config_file, load_config
from db.models import UserAccount
from pipeline.image_flow_edit_masks import (
    object_scope_edit_mask_image,
    should_use_object_scope_edit_mask,
)
from pipeline.image_flow_preflight import (
    BBoxNorm as PreflightBBoxNorm,
    EditOperationConfig as PreflightEditOperationConfig,
    compile_visual_preflight_prompt,
    run_visual_preflight,
)
from pipeline.account_layout_autosave import build_saved_layout_option_drafts
from pipeline.image_flow_logging import log_image_flow_event
from pipeline.orchestrator import (
    _make_case_id,
    _now_utc_iso,
    _write_json,
    case_paths,
    run_case,
)
from pipeline.snapshot_image_renderer import (
    SnapshotEditOperation,
    SnapshotImageRenderConfig,
    SnapshotLockedGeometryAtlas,
    SnapshotLockedGeometryAtlasObject,
    SnapshotLockedGeometryReference,
    render_snapshot_image,
    render_snapshot_image_from_path,
)
from pipeline.snapshot_prompt_compiler import (
    compile_snapshot_prompt,
    compile_snapshot_prompt_from_path,
)
from services.user_content_service import (
    LayoutOptionPersistenceRequest,
    UserContentService,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)

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


class PipelineRunRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    input_payload: dict[str, Any]
    case_name: str | None = None
    description: str | None = None
    special_notes: str | None = None


class PipelineRunResponse(BaseModel):
    case_id: str
    case_dir: str
    status_path: str
    status: str


class PipelineStatusResponse(BaseModel):
    case_id: str
    stage: str
    updated_at_utc: str
    error: str | None = None
    message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)


class PipelineResultResponse(BaseModel):
    case_id: str
    result: dict[str, Any]


class ArtifactResponse(BaseModel):
    case_id: str
    name: str
    payload: dict[str, Any]


class ClusterListResponse(BaseModel):
    case_id: str
    clusters: list[str]


class SnapshotPromptCompileRequest(BaseModel):
    snapshot_payload: dict[str, Any] | None = None
    snapshot_path: str | None = None


class SnapshotPromptCompileResponse(BaseModel):
    compilation: dict[str, Any]


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


def get_user_content_service() -> UserContentService:
    return UserContentService()


def _enrich_rotation_ccw(
    *,
    stylist_payload: dict[str, Any],
    absolute_layout_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    UI needs normalized orientation metadata per object.
    Stylist output intentionally omits it; we stitch it from absolute_layout,
    which contains the final semantic rotation and front-facing direction.
    """
    if not isinstance(stylist_payload, dict):
        return stylist_payload
    if not absolute_layout_payload or not isinstance(absolute_layout_payload, dict):
        return stylist_payload

    abs_objects = absolute_layout_payload.get("objects") or []
    if not isinstance(abs_objects, list) or not abs_objects:
        return stylist_payload

    by_id: dict[str, dict[str, Any]] = {}
    by_bbox: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for ao in abs_objects:
        if not isinstance(ao, dict):
            continue
        oid = ao.get("object_id")
        rot = ao.get("rotation_ccw", ao.get("rot"))
        bbox = ao.get("bbox") or {}
        if not isinstance(oid, str) or not oid:
            continue
        try:
            rotation_ccw = int(rot) % 360
        except (TypeError, ValueError):
            continue
        orientation_payload = {
            "rotation_ccw": rotation_ccw,
            "front_world": deepcopy(ao.get("front_world"))
            if isinstance(ao.get("front_world"), dict)
            else None,
            "front_side_world": (
                str(ao.get("front_side_world")).strip().lower()
                if isinstance(ao.get("front_side_world"), str)
                and str(ao.get("front_side_world")).strip()
                else None
            ),
            "axis_world": deepcopy(ao.get("axis_world"))
            if isinstance(ao.get("axis_world"), dict)
            else None,
        }
        by_id[oid] = orientation_payload
        # Best-effort fallback matching: bbox equality.
        # This helps if stylist instance ids don't exactly match absolute layout ids.
        try:
            key = (
                int(bbox.get("min_x", 0)),
                int(bbox.get("min_y", 0)),
                int(bbox.get("max_x", 0)),
                int(bbox.get("max_y", 0)),
            )
            by_bbox[key] = orientation_payload
        except Exception:
            pass

    out = dict(stylist_payload)
    objects = out.get("objects") or []
    if not isinstance(objects, list) or not objects:
        return out

    enriched: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        inst = obj.get("instance_id") or obj.get("id")
        bbox = obj.get("bbox") or {}

        orientation_payload: dict[str, Any] | None = None
        if isinstance(inst, str) and inst in by_id:
            orientation_payload = by_id[inst]
        else:
            # bbox match (best-effort)
            try:
                key = (
                    int(bbox.get("min_x", 0)),
                    int(bbox.get("min_y", 0)),
                    int(bbox.get("max_x", 0)),
                    int(bbox.get("max_y", 0)),
                )
                if key in by_bbox:
                    orientation_payload = by_bbox[key]
            except Exception:
                orientation_payload = None

        next_obj = dict(obj)
        next_obj["rotation_ccw"] = (
            int(
                (orientation_payload or {}).get(
                    "rotation_ccw",
                    next_obj.get("rotation_ccw", next_obj.get("rot", 0)),
                )
                or 0
            )
            % 360
        )
        next_obj["front_world"] = deepcopy(
            (orientation_payload or {}).get("front_world")
        )
        next_obj["front_side_world"] = (orientation_payload or {}).get(
            "front_side_world"
        )
        next_obj["axis_world"] = deepcopy((orientation_payload or {}).get("axis_world"))
        enriched.append(next_obj)

    out["objects"] = enriched
    return out


def _run_in_thread(**kwargs: Any) -> None:
    account_user_id = kwargs.pop("account_user_id", None)
    account_case_name = kwargs.pop("account_case_name", None)
    try:
        run_case(**kwargs)
        if isinstance(account_user_id, str) and account_user_id.strip():
            try:
                _autosave_pipeline_case(
                    account_user_id=account_user_id,
                    case_id=str(kwargs["case_id"]),
                    cases_root=str(kwargs["cases_root"]),
                    case_name=account_case_name
                    if isinstance(account_case_name, str)
                    else None,
                )
            except Exception as exc:
                logger.exception(
                    "Pipeline account autosave failed for case %s: %s",
                    kwargs.get("case_id"),
                    exc,
                )
                _append_status_action(
                    case_id=str(kwargs["case_id"]),
                    cases_root=str(kwargs["cases_root"]),
                    stage="account_autosave_warning",
                    message=f"Generated layouts were not saved to account: {exc}",
                    error=str(exc),
                )
    except Exception as exc:  # noqa: BLE001
        paths = case_paths(kwargs["case_id"], kwargs["cases_root"])
        existing_payload: dict[str, Any] = {}
        if paths.status.exists():
            try:
                existing_payload = json.loads(paths.status.read_text())
            except Exception:
                existing_payload = {}
        actions = existing_payload.get("actions")
        action_history = (
            [item for item in actions if isinstance(item, dict)]
            if isinstance(actions, list)
            else []
        )
        updated_at_utc = _now_utc_iso()
        action_history.append(
            {
                "stage": "error",
                "message": str(exc),
                "updated_at_utc": updated_at_utc,
                "progress_current": existing_payload.get("progress_current"),
                "progress_total": existing_payload.get("progress_total"),
                "error": str(exc),
            }
        )
        _write_json(
            paths.status,
            {
                "case_id": paths.case_id,
                "stage": "error",
                "updated_at_utc": updated_at_utc,
                "error": str(exc),
                "actions": action_history,
            },
        )


def _autosave_pipeline_case(
    *,
    account_user_id: str,
    case_id: str,
    cases_root: str,
    case_name: str | None,
) -> None:
    service = UserContentService()
    user = service.get_user_by_id(account_user_id)
    if user is None:
        raise ValueError(f"Cannot autosave layouts for unknown user {account_user_id}.")

    paths = case_paths(case_id, cases_root)
    resolved_case_name = _resolve_account_case_name(case_name, case_id)
    drafts = build_saved_layout_option_drafts(
        case_id=case_id,
        case_name=resolved_case_name,
        room_payload=_load_json_payload(paths.room_interpreter),
        stylist_payload=_load_json_payload(paths.stylist),
        absolute_layout_payload=_load_json_payload(paths.absolute_layout),
        layout_variants_payload=_load_json_payload(paths.layout_variants),
    )
    if not drafts:
        _append_status_action(
            case_id=case_id,
            cases_root=cases_root,
            stage="account_autosave_warning",
            message="No generated layout options were available for account autosave.",
            error=None,
        )
        return

    saved = service.upsert_layout_case_options(
        user=user,
        case_id=case_id,
        case_name=resolved_case_name,
        options=[
            LayoutOptionPersistenceRequest(
                option_id=draft.option_id,
                option_index=draft.option_index,
                name=draft.name,
                floorplan_json=draft.floorplan_json,
                design_json=draft.design_json,
                styled_result_json=draft.styled_result_json,
                meta=draft.meta,
            )
            for draft in drafts
        ],
        meta={
            "source": "pipeline_autosave",
            "source_pipeline_case_id": case_id,
            "option_count": len(drafts),
        },
    )
    _append_status_action(
        case_id=case_id,
        cases_root=cases_root,
        stage="account_autosave",
        message=(
            f"Saved {len(saved)} generated layout option"
            f"{'' if len(saved) == 1 else 's'} to account case."
        ),
        error=None,
    )


def _resolve_account_case_name(case_name: str | None, case_id: str) -> str:
    if case_name is not None and case_name.strip():
        return case_name.strip()
    return f"Layout case {case_id}"


def _load_json_payload(path: Any) -> object:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _append_status_action(
    *,
    case_id: str,
    cases_root: str,
    stage: str,
    message: str,
    error: str | None,
) -> None:
    paths = case_paths(case_id, cases_root)
    payload: dict[str, Any] = {}
    if paths.status.exists():
        try:
            payload = json.loads(paths.status.read_text())
        except Exception:
            payload = {}
    actions = payload.get("actions")
    action_history = (
        [item for item in actions if isinstance(item, dict)]
        if isinstance(actions, list)
        else []
    )
    updated_at_utc = _now_utc_iso()
    action_history.append(
        {
            "stage": stage,
            "message": message,
            "updated_at_utc": updated_at_utc,
            "progress_current": payload.get("progress_current"),
            "progress_total": payload.get("progress_total"),
            "error": error,
        }
    )
    payload.update(
        {
            "case_id": case_id,
            "stage": payload.get("stage", "done"),
            "updated_at_utc": updated_at_utc,
            "actions": action_history,
        }
    )
    _write_json(paths.status, payload)


@router.post("/run", response_model=PipelineRunResponse)
def run_pipeline(
    req: PipelineRunRequest,
    background: BackgroundTasks,
    current_user: UserAccount | None = Depends(get_optional_current_user),
) -> PipelineRunResponse:
    pipeline_user_id = str(current_user.id) if current_user is not None else req.user_id
    case_id = _make_case_id(pipeline_user_id)
    paths = case_paths(case_id)
    paths.root.mkdir(parents=True, exist_ok=True)

    _write_json(
        paths.status,
        {
            "case_id": case_id,
            "stage": "queued",
            "updated_at_utc": _now_utc_iso(),
        },
    )

    background.add_task(
        _run_in_thread,
        input_payload=req.input_payload,
        user_id=pipeline_user_id,
        description=req.description,
        special_notes=req.special_notes,
        cases_root=str(paths.root.parent),
        case_id=case_id,
        account_user_id=str(current_user.id) if current_user is not None else None,
        account_case_name=req.case_name or req.description,
    )

    return PipelineRunResponse(
        case_id=case_id,
        case_dir=str(paths.root),
        status_path=str(paths.status),
        status="queued",
    )


@router.post("/compile-snapshot-prompt", response_model=SnapshotPromptCompileResponse)
def compile_snapshot_prompt_route(
    req: SnapshotPromptCompileRequest,
) -> SnapshotPromptCompileResponse:
    if req.snapshot_payload is None and not req.snapshot_path:
        raise HTTPException(
            status_code=400,
            detail="Provide either snapshot_payload or snapshot_path.",
        )

    try:
        if req.snapshot_payload is not None:
            compilation = compile_snapshot_prompt(req.snapshot_payload)
        else:
            compilation = compile_snapshot_prompt_from_path(req.snapshot_path or "")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SnapshotPromptCompileResponse(compilation=compilation)


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
        _IMAGE_FLOW_PREFLIGHT_DIR
        / "sources"
        / f"{trace_id}_{safe_label}.{extension}"
    )
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

    if req.render_flow == "single_edit_edge_silhouette_guided":
        effective_render_mode = "edit"
        effective_edit_source_image_data_url = (
            req.edit_source_image_data_url or req.snapshot_image_data_url
        )
        layout_materialization_mode = True
        effective_user_prompt = _append_snapshot_flow_prompt(
            effective_user_prompt,
            _SINGLE_EDIT_EDGE_GUIDED_PROMPT,
        )
        if not effective_edit_source_image_data_url:
            raise ValueError(
                "single_edit_edge_silhouette_guided requires snapshot_image_data_url."
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
                    "storage_path": saved.storage_path,
                    "root_layout_id": str(saved.root_layout_id)
                    if saved.root_layout_id
                    else None,
                    "case_id": str(saved.case_id) if saved.case_id else None,
                    "option_id": saved.option_id,
                    "meta": dict(saved.meta or {}),
                },
            )
        except Exception as exc:
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


@router.post("/render-snapshot-image", response_model=SnapshotImageRenderResponse)
def render_snapshot_image_route(
    req: SnapshotImageRenderRequest,
    current_user: UserAccount | None = Depends(get_optional_current_user),
    user_content_service: UserContentService = Depends(get_user_content_service),
) -> SnapshotImageRenderResponse:
    trace_id = uuid4().hex
    log_image_flow_event(
        "snapshot_image.api_input",
        {
            "trace_id": trace_id,
            "route": "/pipeline/render-snapshot-image",
            "authenticated": current_user is not None,
            "request": req.model_dump(),
        },
    )
    render_layout_context = _resolve_render_layout_context_or_raise(
        req=req,
        current_user=current_user,
        user_content_service=user_content_service,
        trace_id=trace_id,
    )
    try:
        render = _render_snapshot_image_for_request(
            req,
            trace_id=trace_id,
            streamed=False,
        )
    except HTTPException:
        raise
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _raise_snapshot_render_http_exception(trace_id, exc)

    saved_render = _save_snapshot_render_for_request(
        render=render,
        current_user=current_user,
        render_layout_context=render_layout_context,
        user_content_service=user_content_service,
        trace_id=trace_id,
    )
    return SnapshotImageRenderResponse(render=render, saved_render=saved_render)


def _encode_sse_event(event_name: str, payload: Mapping[str, Any]) -> str:
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
    )


@router.post("/render-snapshot-image/stream")
def render_snapshot_image_stream_route(
    req: SnapshotImageRenderRequest,
    current_user: UserAccount | None = Depends(get_optional_current_user),
    user_content_service: UserContentService = Depends(get_user_content_service),
) -> StreamingResponse:
    trace_id = uuid4().hex
    log_image_flow_event(
        "snapshot_image.api_input",
        {
            "trace_id": trace_id,
            "route": "/pipeline/render-snapshot-image/stream",
            "authenticated": current_user is not None,
            "request": req.model_dump(),
        },
    )
    event_queue: Queue[dict[str, Any] | None] = Queue()

    def enqueue(event_name: str, payload: Mapping[str, Any]) -> None:
        event_queue.put({"event": event_name, "data": dict(payload)})

    def forward_renderer_event(event: Mapping[str, object]) -> None:
        event_name = str(event.get("type") or "status")
        payload = {key: value for key, value in event.items() if key != "type"}
        enqueue(event_name, payload)

    def worker() -> None:
        try:
            enqueue("status", {"message": "Preparing image render.", "attempt": None})
            render_layout_context = _resolve_render_layout_context_or_raise(
                req=req,
                current_user=current_user,
                user_content_service=user_content_service,
                trace_id=trace_id,
            )
            render = _render_snapshot_image_for_request(
                req,
                trace_id=trace_id,
                streamed=True,
                stream_event_handler=forward_renderer_event,
            )
            saved_render = _save_snapshot_render_for_request(
                render=render,
                current_user=current_user,
                render_layout_context=render_layout_context,
                user_content_service=user_content_service,
                trace_id=trace_id,
            )
            enqueue(
                "completed",
                SnapshotImageRenderResponse(
                    render=render,
                    saved_render=saved_render,
                ).model_dump(),
            )
        except HTTPException as exc:
            enqueue("error", {"message": str(exc.detail)})
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            log_image_flow_event(
                "snapshot_image.api_error",
                {
                    "trace_id": trace_id,
                    "route": "/pipeline/render-snapshot-image/stream",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            enqueue("error", {"message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Snapshot image stream failed: %s", exc)
            log_image_flow_event(
                "snapshot_image.api_error",
                {
                    "trace_id": trace_id,
                    "route": "/pipeline/render-snapshot-image/stream",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            enqueue("error", {"message": str(exc)})
        finally:
            event_queue.put(None)

    def event_stream():
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield _encode_sse_event(
                str(item["event"]),
                item["data"] if isinstance(item["data"], dict) else {},
            )

    Thread(target=worker, daemon=True).start()
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/render-presets", response_model=SnapshotImagePresetsResponse)
def get_render_presets() -> SnapshotImagePresetsResponse:
    presets = load_config(config_file).presets
    return SnapshotImagePresetsResponse(
        styles={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.styles.items()
        },
        lights={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.lights.items()
        },
        sceneries={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.sceneries.items()
        },
    )


@router.get("/{case_id}/status", response_model=PipelineStatusResponse)
def get_status(case_id: str) -> PipelineStatusResponse:
    paths = case_paths(case_id)
    if not paths.status.exists():
        raise HTTPException(status_code=404, detail="case not found")
    payload = json.loads(paths.status.read_text())
    return PipelineStatusResponse(
        case_id=payload.get("case_id", case_id),
        stage=payload.get("stage", "unknown"),
        updated_at_utc=payload.get("updated_at_utc", ""),
        error=payload.get("error"),
        message=payload.get("message"),
        progress_current=payload.get("progress_current"),
        progress_total=payload.get("progress_total"),
        actions=payload.get("actions")
        if isinstance(payload.get("actions"), list)
        else [],
    )


@router.get("/{case_id}/result", response_model=PipelineResultResponse)
def get_result(case_id: str) -> PipelineResultResponse:
    paths = case_paths(case_id)
    if not paths.stylist.exists() and not paths.layout_variants.exists():
        raise HTTPException(status_code=404, detail="result not found")
    enriched: dict[str, Any] = {}
    if paths.stylist.exists():
        stylist_payload = json.loads(paths.stylist.read_text())
        absolute_payload: dict[str, Any] | None = None
        try:
            if paths.absolute_layout.exists():
                absolute_payload = json.loads(paths.absolute_layout.read_text())
        except Exception:
            absolute_payload = None

        enriched = _enrich_rotation_ccw(
            stylist_payload=stylist_payload,
            absolute_layout_payload=absolute_payload,
        )
    if paths.layout_variants.exists():
        variants_payload = json.loads(paths.layout_variants.read_text())
        if isinstance(variants_payload, dict):
            variants = variants_payload.get("variants")
            selected_variant_id = variants_payload.get("selected_variant_id")
            if isinstance(variants, list):
                enriched_variants: list[dict[str, Any]] = []
                for variant in variants:
                    if not isinstance(variant, dict):
                        continue
                    styled_variant = variant.get("styled_result")
                    absolute_variant = variant.get("absolute_layout")
                    next_variant = dict(variant)
                    if isinstance(styled_variant, dict):
                        next_variant["styled_result"] = _enrich_rotation_ccw(
                            stylist_payload=styled_variant,
                            absolute_layout_payload=absolute_variant
                            if isinstance(absolute_variant, dict)
                            else None,
                        )
                    enriched_variants.append(next_variant)
                if enriched_variants and not enriched:
                    first_variant = enriched_variants[0]
                    styled_result = first_variant.get("styled_result")
                    if isinstance(styled_result, dict):
                        enriched = dict(styled_result)
                enriched["variants"] = enriched_variants
            if isinstance(selected_variant_id, str) and selected_variant_id:
                enriched["selected_variant_id"] = selected_variant_id
    return PipelineResultResponse(case_id=case_id, result=enriched)


@router.get("/{case_id}/artifact/{name}", response_model=ArtifactResponse)
def get_artifact(case_id: str, name: str) -> ArtifactResponse:
    paths = case_paths(case_id)
    mapping = {
        "module_io_manifest": paths.module_io_manifest,
        "room_interpreter": paths.room_interpreter,
        "stylist_style_policy": paths.module_output("stylist_style_policy"),
        "cluster_forge": paths.cluster_forge,
        "tier_count": paths.tier_count,
        "tier_count_director": paths.module_output("tier_count_director"),
        "cluster_merged": paths.cluster_merged,
        "cluster_output_merger": paths.module_output("cluster_output_merger"),
        "cluster_relation_plan": paths.cluster_relation_plan,
        "seed_concept_relation_plan": paths.module_output("seed_concept_relation_plan"),
        "seed_concept_generator": paths.module_output("seed_concept_generator"),
        "cluster_placer": paths.cluster_placer,
        "phase2_controller": paths.module_output("phase2_controller"),
        "macro_cluster_solver": paths.module_output("macro_cluster_solver"),
        "macro_cluster_solver_dropped_inventory": paths.module_output(
            "macro_cluster_solver_dropped_inventory"
        ),
        "absolute_layout": paths.absolute_layout,
        "controlled_accessory_refill": paths.module_output(
            "controlled_accessory_refill"
        ),
        "stylist": paths.stylist,
        "cluster_outlines": paths.cluster_outlines_all,
        "cluster_outline_bundle": paths.module_output("cluster_outline_bundle"),
        "layout_variants": paths.layout_variants,
    }
    path = mapping.get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    payload = json.loads(path.read_text())
    return ArtifactResponse(case_id=case_id, name=name, payload=payload)


@router.get("/{case_id}/clusters", response_model=ClusterListResponse)
def list_clusters(case_id: str) -> ClusterListResponse:
    paths = case_paths(case_id)
    if not paths.clusters_dir.exists():
        raise HTTPException(status_code=404, detail="clusters not found")
    clusters = sorted(
        {
            p.name.replace("cluster_composer_", "").replace(".json", "")
            for p in paths.clusters_dir.glob("cluster_composer_*.json")
        }
    )
    return ClusterListResponse(case_id=case_id, clusters=clusters)


@router.get("/{case_id}/clusters/{cluster_id}", response_model=ArtifactResponse)
def get_cluster_composer(case_id: str, cluster_id: str) -> ArtifactResponse:
    paths = case_paths(case_id)
    path = paths.cluster_composer(cluster_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="cluster composer output not found")
    payload = json.loads(path.read_text())
    return ArtifactResponse(
        case_id=case_id, name=f"cluster_composer_{cluster_id}", payload=payload
    )


@router.get("/{case_id}/clusters/{cluster_id}/outline", response_model=ArtifactResponse)
def get_cluster_outline(case_id: str, cluster_id: str) -> ArtifactResponse:
    paths = case_paths(case_id)
    path = paths.cluster_outline(cluster_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="cluster outline not found")
    payload = json.loads(path.read_text())
    return ArtifactResponse(
        case_id=case_id, name=f"cluster_outline_{cluster_id}", payload=payload
    )
