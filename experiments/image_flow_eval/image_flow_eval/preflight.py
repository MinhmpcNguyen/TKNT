from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import httpx

from .io import encode_image_to_data_url
from .types import BBoxNorm, EditOperationConfig, JSONValue, VisualPreflightProfile

_MODEL_ENV = "TKNT_IMAGE_FLOW_PREFLIGHT_MODEL"
_BASE_URL_ENV = "TKNT_IMAGE_FLOW_PREFLIGHT_OPENAI_BASE_URL"
_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_OPENAI_IMAGE_API_KEY_ENV = "OPENAI_IMAGE_API_KEY"
_DEFAULT_MODEL = "gpt-5.4-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_PREFLIGHT_PROMPT_VERSIONS: Mapping[VisualPreflightProfile, str] = {
    "classic": "classic-v1",
    "room_shell": "room-shell-v1",
    "object_anchors": "object-anchors-v1",
    "edit_scope": "edit-scope-v1",
    "edit_object_scope": "edit-object-scope-v1",
}
_MAX_VISIBLE_OBJECTS = 24
_MAX_PRIORITY_LOCKS = 4
_MAX_ROOM_SHELL_LOCKS = 4
_MAX_OBJECT_ANCHOR_LOCKS = 8
_MAX_EDIT_SCOPE_LOCKS = 6
_MAX_LOCK_CHARS = 260
_OBJECT_ANCHOR_TYPES = {
    "armchair",
    "bed",
    "bookshelf",
    "cabinet",
    "coffee_table",
    "desk",
    "floor_lamp",
    "media_shelf",
    "nightstand",
    "side_table",
    "sofa",
    "speaker",
    "table_lamp",
    "tv",
    "tv_console",
    "wardrobe",
}
_RISK_LABELS = {
    "composite_object",
    "cropped_room_shell",
    "bbox_boundary_risk",
    "edit_target_ambiguous",
    "foreground_occlusion",
    "low_contrast_object",
    "opening_visibility",
    "replacement_reference_risk",
    "repeated_objects",
    "scale_sensitive_object",
    "tight_crop",
}


class VisualPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisualPreflightResult:
    source_valid: bool
    risk_profile: tuple[str, ...]
    required_counts: Mapping[str, int]
    priority_locks: tuple[str, ...]
    reason: str
    model: str
    cache_key: str
    cache_status: Literal["hit", "miss"]
    usage: Mapping[str, JSONValue]
    room_shell_locks: tuple[str, ...] = ()
    object_anchor_locks: tuple[str, ...] = ()
    edit_scope_locks: tuple[str, ...] = ()
    preflight_profile: VisualPreflightProfile = "classic"

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "source_valid": self.source_valid,
            "risk_profile": list(self.risk_profile),
            "required_counts": dict(self.required_counts),
            "room_shell_locks": list(self.room_shell_locks),
            "object_anchor_locks": list(self.object_anchor_locks),
            "edit_scope_locks": list(self.edit_scope_locks),
            "priority_locks": list(self.priority_locks),
            "reason": self.reason,
            "model": self.model,
            "cache_key": self.cache_key,
            "cache_status": self.cache_status,
            "preflight_profile": self.preflight_profile,
            "usage": dict(self.usage),
        }


def run_visual_preflight(
    *,
    source_image: Path,
    source_payload: Mapping[str, object],
    include_metadata: bool,
    report_path: Path,
    cache_dir: Path | None = None,
    profile: VisualPreflightProfile = "classic",
    edit_operations: Sequence[EditOperationConfig] = (),
) -> VisualPreflightResult:
    model = _model()
    profile = _coerce_preflight_profile(profile, fallback="classic")
    metadata_summary = dict(
        _preflight_metadata_summary(
            _metadata_summary(source_payload),
            profile=profile,
        )
    )
    if profile in {"edit_scope", "edit_object_scope"}:
        metadata_summary["edit_operations"] = _edit_operations_summary(
            edit_operations
        )
    cache_key = _cache_key(
        source_image=source_image,
        metadata_summary=metadata_summary if include_metadata else {},
        include_metadata=include_metadata,
        model=model,
        profile=profile,
    )
    shared_cache_path = (
        cache_dir / f"{cache_key}.json" if cache_dir is not None else None
    )
    cached = _load_cached_report(
        report_path,
        cache_key=cache_key,
        preflight_profile=profile,
    )
    if cached is None and shared_cache_path is not None:
        cached = _load_cached_report(
            shared_cache_path,
            cache_key=cache_key,
            preflight_profile=profile,
        )
    if cached is not None:
        _write_report(report_path, cached)
        return cached

    api_key = _api_key()
    if not api_key:
        raise VisualPreflightError(
            "Visual preflight requires OPENAI_API_KEY or OPENAI_IMAGE_API_KEY."
        )

    prompt = _preflight_prompt(
        metadata_summary=metadata_summary,
        include_metadata=include_metadata,
        profile=profile,
    )
    body: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a visual preflight controller for interior "
                    "photorealization. Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_to_data_url(source_image)},
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    endpoint = os.getenv(_BASE_URL_ENV, _DEFAULT_BASE_URL).rstrip("/")
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                endpoint + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
        response.raise_for_status()
        response_payload = response.json()
        report = _extract_chat_report(response_payload)
        usage = _json_mapping(_mapping(response_payload).get("usage"))
    except (httpx.HTTPError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise VisualPreflightError(
            f"Visual preflight request failed: {type(exc).__name__}: {exc}"
        ) from exc

    result = _normalize_report(
        report,
        source_payload=source_payload,
        model=model,
        cache_key=cache_key,
        cache_status="miss",
        usage=usage,
        preflight_profile=profile,
    )
    _write_report(report_path, result)
    if shared_cache_path is not None:
        _write_report(shared_cache_path, result)
    return result


def compile_visual_preflight_prompt(
    result: VisualPreflightResult,
    *,
    source_payload: Mapping[str, object] | None = None,
    profile: VisualPreflightProfile | None = None,
    edit_operations: Sequence[EditOperationConfig] = (),
) -> str:
    active_profile = profile or result.preflight_profile
    lines: list[str] = []
    if active_profile in {"edit_scope", "edit_object_scope"}:
        lines.extend(_edit_operation_scope_lock_lines(edit_operations))
        lines.extend(result.edit_scope_locks)
        lines.extend(result.priority_locks)
        if not lines:
            lines.append(
                "Edit-scope lock: apply only the requested edit to the selected "
                "target; preserve camera, room geometry, lighting, objects, "
                "materials, and background everywhere else."
            )
        return (
            "Edit-specific visual preflight locks for this exact approved render:\n- "
            + "\n- ".join(lines)
        )
    if source_payload is not None and active_profile in {
        "room_shell",
        "object_anchors",
    }:
        lines.extend(_room_shell_metadata_lock_lines(source_payload))
    if active_profile in {"room_shell", "object_anchors"}:
        lines.extend(result.room_shell_locks)
    if source_payload is not None and active_profile == "object_anchors":
        lines.extend(_room_geometry_shadow_lock_lines(source_payload))
        lines.extend(_object_anchor_lock_lines(source_payload))
    if active_profile == "object_anchors":
        lines.extend(result.object_anchor_locks)
    for object_type, count in sorted(result.required_counts.items()):
        lines.append(f"Preserve exactly {count} visible {object_type} instance(s).")
    if source_payload is not None and active_profile in {
        "room_shell",
        "object_anchors",
    }:
        lines.extend(_opening_lock_lines(source_payload))
    lines.extend(result.priority_locks)
    if not lines:
        return ""
    return (
        "Scene-specific visual preflight locks for this exact source:\n- "
        + "\n- ".join(lines)
    )


def _write_report(path: Path, result: VisualPreflightResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_cached_report(
    report_path: Path,
    *,
    cache_key: str,
    preflight_profile: VisualPreflightProfile,
) -> VisualPreflightResult | None:
    if not report_path.is_file():
        return None
    try:
        payload = _mapping(json.loads(report_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if payload.get("cache_key") != cache_key:
        return None
    return _normalize_report(
        payload,
        source_payload={},
        model=str(payload.get("model") or _model()),
        cache_key=cache_key,
        cache_status="hit",
        usage=_json_mapping(payload.get("usage")),
        preflight_profile=preflight_profile,
        trust_cached_counts=True,
    )


def _normalize_report(
    report: Mapping[str, object],
    *,
    source_payload: Mapping[str, object],
    model: str,
    cache_key: str,
    cache_status: Literal["hit", "miss"],
    usage: Mapping[str, JSONValue],
    preflight_profile: VisualPreflightProfile = "classic",
    trust_cached_counts: bool = False,
) -> VisualPreflightResult:
    normalized_profile = _coerce_preflight_profile(
        report.get("preflight_profile"),
        fallback=preflight_profile,
    )
    raw_source_valid = report.get("source_valid")
    source_valid = raw_source_valid if isinstance(raw_source_valid, bool) else False
    risk_profile = tuple(
        sorted(
            {
                label
                for item in _sequence(report.get("risk_profile"))
                if (label := _clean_label(item)) in _RISK_LABELS
            }
        )
    )
    available_counts = _visible_object_type_counts(source_payload)
    requested_counts = _mapping(report.get("required_counts"))
    required_counts: dict[str, int] = {}
    for raw_type, raw_count in requested_counts.items():
        object_type = _clean_label(raw_type)
        count = _positive_int(raw_count)
        if not object_type or count is None:
            continue
        if trust_cached_counts or available_counts.get(object_type) == count:
            required_counts[object_type] = count
    priority_locks = tuple(
        lock
        for item in _sequence(report.get("priority_locks"))[:_MAX_PRIORITY_LOCKS]
        if (lock := _clean_lock(item))
    )
    room_shell_locks = tuple(
        lock
        for item in _sequence(report.get("room_shell_locks"))[:_MAX_ROOM_SHELL_LOCKS]
        if (lock := _clean_lock(item))
    )
    object_anchor_locks = tuple(
        lock
        for item in _sequence(report.get("object_anchor_locks"))[
            :_MAX_OBJECT_ANCHOR_LOCKS
        ]
        if (lock := _clean_lock(item))
    )
    edit_scope_locks = tuple(
        lock
        for item in _sequence(report.get("edit_scope_locks"))[:_MAX_EDIT_SCOPE_LOCKS]
        if (lock := _clean_lock(item))
    )
    return VisualPreflightResult(
        source_valid=source_valid,
        risk_profile=risk_profile,
        required_counts=required_counts,
        priority_locks=priority_locks,
        reason=_clean_lock(report.get("reason")) or "",
        model=model,
        cache_key=cache_key,
        cache_status=cache_status,
        usage=usage,
        room_shell_locks=room_shell_locks,
        object_anchor_locks=object_anchor_locks,
        edit_scope_locks=edit_scope_locks,
        preflight_profile=normalized_profile,
    )


def _preflight_prompt(
    *,
    metadata_summary: Mapping[str, JSONValue],
    include_metadata: bool,
    profile: VisualPreflightProfile = "classic",
) -> str:
    metadata_text = (
        json.dumps(metadata_summary, ensure_ascii=False, separators=(",", ":"))
        if include_metadata
        else "omitted for this ablation"
    )
    if profile == "classic":
        return _classic_preflight_prompt(metadata_text)
    if profile == "edit_scope":
        return _edit_scope_preflight_prompt(metadata_text)
    if profile == "edit_object_scope":
        return _edit_object_scope_preflight_prompt(metadata_text)
    if profile == "object_anchors":
        return _object_anchor_preflight_prompt(metadata_text)
    return _room_shell_preflight_prompt(metadata_text)


def _classic_preflight_prompt(metadata_text: str) -> str:
    return (
        "Inspect Image 1, which is the exact 3D source that an image model will "
        "photorealize. Identify only the few scene-specific risks that could "
        "make the image model add, remove, merge, resize, or move visible "
        "objects, or change openings and framing. Do not redesign the room. Keep "
        "the output compact. source_valid must be false only when the source "
        "image itself is unusable or visibly inconsistent. required_counts "
        "should include only repeated or ambiguity-prone visible object types. "
        "priority_locks should contain at most four short object/opening "
        "preservation instructions. Treat metadata openings as authoritative "
        "when metadata is provided.\n\n"
        f"Metadata summary: {metadata_text}\n\n"
        "Return JSON with keys: source_valid (boolean), risk_profile (array "
        "using only composite_object, foreground_occlusion, "
        "low_contrast_object, opening_visibility, repeated_objects, "
        "scale_sensitive_object, tight_crop), required_counts (object mapping), "
        "priority_locks (array), reason (string)."
    )


def _room_shell_preflight_prompt(metadata_text: str) -> str:
    return (
        "Inspect Image 1, which is the exact 3D source that an image model will "
        "photorealize. First inspect the room shell before furniture: visible wall "
        "planes, floor-wall seams, ceiling-wall seams, corners, diagonal wall "
        "edges, and narrow cropped foreground wall strips along the image borders. "
        "A cropped wall strip is a real wall surface, not empty space, not a "
        "window, and not a camera artifact. Then identify only the few "
        "scene-specific risks that could make "
        "the image model add, remove, merge, resize, or move visible objects, or "
        "change openings and framing. Do not redesign the room. Keep the output "
        "compact. source_valid must be false only when the source image itself is "
        "unusable or visibly inconsistent. required_counts should include only "
        "repeated or ambiguity-prone visible object types. room_shell_locks should "
        "contain at most four short instructions for preserving wall planes, "
        "cropped wall strips, diagonal edges, corners, seams, and occlusions. Each "
        "room_shell_lock must state the location and what must not change. "
        "priority_locks should contain at most four short object/opening "
        "preservation instructions. Treat metadata "
        "openings as authoritative when metadata is provided. If metadata says "
        "there are zero windows or zero doors, explicitly protect the blank walls, "
        "cropped wall edges, curtains, and wall panels from being converted into "
        "new openings or exterior views. If a visible edge is ambiguous, describe "
        "it as an edge/crop/curtain unless the source clearly shows a window or "
        "door opening.\n\n"
        f"Metadata summary: {metadata_text}\n\n"
        "Return JSON with keys: source_valid (boolean), risk_profile (array using "
        "only composite_object, foreground_occlusion, low_contrast_object, "
        "opening_visibility, repeated_objects, scale_sensitive_object, tight_crop, "
        "cropped_room_shell), required_counts (object mapping), room_shell_locks "
        "(array), priority_locks (array), reason (string)."
    )


def _object_anchor_preflight_prompt(metadata_text: str) -> str:
    return (
        "Inspect Image 1, which is the exact 3D source that an image model will "
        "photorealize. First inspect the room shell before furniture: visible "
        "wall planes, floor-wall seams, ceiling-wall seams, corners, diagonal "
        "wall edges, and narrow cropped foreground wall strips along the image "
        "borders. Use room polygon/opening metadata as geometry truth when it is "
        "provided. Distinguish hard geometry from lighting: a soft shadow, "
        "gradient, or triangular brightness patch on a wall is not a slanted "
        "wall, chamfer, window, or opening unless the source and metadata both "
        "support that geometry. Then inspect object anchors: cropped objects at "
        "image edges, support relationships, repeated seating, low media units, "
        "speakers, lamps, shelves, and scale-sensitive furniture. Identify only "
        "the few scene-specific risks that could make the image model add, "
        "remove, merge, resize, or move visible objects, or change openings and "
        "framing. Do not redesign the room. Keep the output compact. "
        "source_valid must be false only when the source image itself is unusable "
        "or visibly inconsistent. required_counts should include only repeated "
        "or ambiguity-prone visible object types. room_shell_locks should contain "
        "at most four short instructions for preserving wall planes, cropped wall "
        "strips, corners, seams, and occlusions. object_anchor_locks should "
        "contain at most eight short instructions for preserving important object "
        "bbox location, silhouette, scale, crop, and support relationships. "
        "priority_locks should contain at most four extra object/opening "
        "preservation instructions. Treat metadata openings as authoritative "
        "when metadata is provided. If metadata says there are zero windows or "
        "zero doors, explicitly protect blank walls, cropped wall edges, curtains, "
        "and wall panels from being converted into new openings or exterior "
        "views.\n\n"
        f"Metadata summary: {metadata_text}\n\n"
        "Return JSON with keys: source_valid (boolean), risk_profile (array using "
        "only composite_object, foreground_occlusion, low_contrast_object, "
        "opening_visibility, repeated_objects, scale_sensitive_object, tight_crop, "
        "cropped_room_shell), required_counts (object mapping), room_shell_locks "
        "(array), object_anchor_locks (array), priority_locks (array), reason "
        "(string)."
    )


def _edit_scope_preflight_prompt(metadata_text: str) -> str:
    return (
        "Inspect Image 1, which is the clean approved photoreal render that will "
        "be edited. Use edit_operations in the metadata as the user's edit "
        "request. Your role is not to redesign or photorealize the whole room; "
        "your role is to identify the selected target object/region/attribute "
        "and produce short scope locks for a downstream image edit model. Treat "
        "object_id, object_name, operation_type, target_color, replacement_image, "
        "and bbox_norm as authoritative when provided. For recolor, change only "
        "the selected object's color/material. For replace, use the replacement "
        "reference only for the selected object and preserve the selected bbox, "
        "scale, perspective, and footprint. For remove, remove only the selected "
        "target and inpaint inside the target area. For add, add only inside the "
        "specified target area. Everything outside the selected bbox/target must "
        "remain unchanged: camera, room shell, openings, lighting, style, all "
        "other furniture, background, shadows, and object positions. If bbox_norm "
        "is missing or too broad, flag bbox_boundary_risk or edit_target_ambiguous "
        "instead of broadening the edit. Keep the output compact.\n\n"
        f"Metadata summary: {metadata_text}\n\n"
        "Return JSON with keys: source_valid (boolean), risk_profile (array using "
        "only edit_target_ambiguous, bbox_boundary_risk, "
        "replacement_reference_risk, foreground_occlusion, low_contrast_object, "
        "scale_sensitive_object, tight_crop), required_counts (object mapping; "
        "usually empty for edit), edit_scope_locks (array), priority_locks "
        "(array), reason (string)."
    )


def _edit_object_scope_preflight_prompt(metadata_text: str) -> str:
    return (
        "Inspect Image 1, which is the approved photoreal render with an "
        "edit-region guide overlay when available. The colored bbox is not the "
        "object itself; it is a locator for the user's intended edit. Use the "
        "bbox, object_name, operation_type, target_color, replacement_image, and "
        "visible object metadata together to infer the exact target object or "
        "surface. Your role is to write compact scope locks for a downstream "
        "image edit model so it edits the intended object only, even when the "
        "bbox cuts through part of the object or overlaps/occludes another "
        "object.\n\n"
        "For recolor, change only the target object's finish/color/material. "
        "Preserve the target object's silhouette, footprint, depth order, "
        "handles, seams, drawer lines, panel gaps, decorative motifs/ornaments, "
        "wood grain or fabric texture details that should remain visible, and "
        "all foreground/neighboring objects that cross or touch the bbox. For "
        "replace/remove/add, identify the exact selected object/placement area "
        "and protect every unrelated object inside or near the bbox. If the bbox "
        "is ambiguous, partial, or overlaps another object, flag "
        "bbox_boundary_risk or edit_target_ambiguous and add an explicit lock "
        "naming what must not change. Do not redesign the room. Keep output "
        "compact.\n\n"
        f"Metadata summary: {metadata_text}\n\n"
        "Return JSON with keys: source_valid (boolean), risk_profile (array "
        "using only edit_target_ambiguous, bbox_boundary_risk, "
        "replacement_reference_risk, foreground_occlusion, low_contrast_object, "
        "scale_sensitive_object, tight_crop), required_counts (object mapping; "
        "usually empty for edit), edit_scope_locks (array), priority_locks "
        "(array), reason (string)."
    )


def _metadata_summary(source_payload: Mapping[str, object]) -> dict[str, JSONValue]:
    visible_objects: list[JSONValue] = []
    for item in _sequence(source_payload.get("visibleObjects"))[:_MAX_VISIBLE_OBJECTS]:
        payload = _mapping(item)
        visible_objects.append(
            {
                "id": _string(payload.get("id")),
                "type": _canonical_object_type(payload),
                "screen_bbox_px": _json_safe(payload.get("screenBboxPx")),
                "dimensions_mm": _json_safe(payload.get("dimensionsMm")),
                "distance_to_camera_m": _json_safe(payload.get("distanceToCameraM")),
            }
        )
    room = _mapping(source_payload.get("room"))
    return {
        "visible_object_count": len(visible_objects),
        "visible_objects": visible_objects,
        "room_shell": {
            "bounds_mm": _json_safe(room.get("boundsMm")),
            "polygon_mm": _json_safe(room.get("polygonMm")),
            "surface_colors": _json_safe(room.get("surfaceColors")),
        },
        "openings": _json_safe(room.get("openings")),
        "canvas": _json_safe(source_payload.get("canvas")),
        "camera": _json_safe(source_payload.get("camera")),
    }


def _edit_operations_summary(
    edit_operations: Sequence[EditOperationConfig],
) -> list[JSONValue]:
    items: list[JSONValue] = []
    for operation in edit_operations:
        items.append(
            {
                "object_id": operation.object_id,
                "object_name": operation.object_name,
                "operation_type": operation.operation_type,
                "prompt": operation.prompt,
                "bbox_norm": _bbox_norm_json(operation.bbox_norm),
                "target_color": operation.target_color,
                "replacement_image": operation.replacement_image.name
                if operation.replacement_image is not None
                else None,
            }
        )
    return items


def _bbox_norm_json(bbox: BBoxNorm | None) -> JSONValue:
    if bbox is None:
        return None
    return {
        "x": round(bbox.x, 4),
        "y": round(bbox.y, 4),
        "width": round(bbox.width, 4),
        "height": round(bbox.height, 4),
    }


def _edit_operation_scope_lock_lines(
    edit_operations: Sequence[EditOperationConfig],
) -> list[str]:
    lines: list[str] = []
    for index, operation in enumerate(edit_operations, start=1):
        target = operation.object_name or operation.object_id or f"manual target {index}"
        action = _edit_operation_action_phrase(operation)
        bbox = _bbox_norm_phrase(operation.bbox_norm)
        lines.append(
            f"Edit-scope lock {index}: {action} only for {target}{bbox}; preserve "
            "camera, room geometry, openings, lighting, style, background, and all "
            "other objects outside the selected target."
        )
    return lines


def _edit_operation_action_phrase(operation: EditOperationConfig) -> str:
    operation_type = operation.operation_type or "prompt"
    if operation_type == "recolor" and operation.target_color:
        return (
            "change only the selected object's color/material to "
            f"{operation.target_color}"
        )
    if operation_type == "replace":
        if operation.replacement_image is not None:
            return (
                "replace only the selected object using the replacement reference "
                "while preserving bbox, scale, perspective, and footprint"
            )
        return (
            "replace only the selected object while preserving bbox, scale, "
            "perspective, and footprint"
        )
    if operation_type == "remove":
        return "remove only the selected object and inpaint naturally inside its area"
    if operation_type == "add":
        return "add only the requested object inside the selected area"
    return "apply only the requested local edit"


def _bbox_norm_phrase(bbox: BBoxNorm | None) -> str:
    if bbox is None:
        return " with no bbox; keep the edit as local as the named target allows"
    return (
        " inside bbox_norm "
        f"x={bbox.x:.4f}, y={bbox.y:.4f}, width={bbox.width:.4f}, "
        f"height={bbox.height:.4f}"
    )


def _preflight_metadata_summary(
    metadata_summary: Mapping[str, JSONValue],
    *,
    profile: VisualPreflightProfile,
) -> Mapping[str, JSONValue]:
    if profile != "classic":
        return metadata_summary
    return {
        key: metadata_summary[key]
        for key in (
            "visible_object_count",
            "visible_objects",
            "openings",
            "canvas",
            "camera",
        )
        if key in metadata_summary
    }


def _room_shell_metadata_lock_lines(source_payload: Mapping[str, object]) -> list[str]:
    room = _mapping(source_payload.get("room"))
    polygon = room.get("polygonMm")
    wall_count = len(polygon) if isinstance(polygon, list) and polygon else None
    if wall_count is None:
        return [
            "Room shell metadata lock: preserve all visible wall planes, wall "
            "edges, corners, and cropped foreground wall strips from the source."
        ]
    return [
        (
            f"Room shell metadata lock: preserve the closed room shell footprint "
            f"with {wall_count} wall segment(s), including all visible wall planes, "
            "corners, seams, and cropped foreground wall strips."
        )
    ]


def _room_geometry_shadow_lock_lines(
    source_payload: Mapping[str, object],
) -> list[str]:
    room = _mapping(source_payload.get("room"))
    polygon = room.get("polygonMm")
    wall_count = len(polygon) if isinstance(polygon, list) and polygon else None
    count_text = f"{wall_count} wall segment(s)" if wall_count is not None else "the"
    return [
        (
            "Geometry-vs-shadow lock: use the room polygon/opening metadata as "
            f"geometry truth for {count_text}; treat soft shadows, gradients, and "
            "brightness patches as lighting only, not new walls, chamfers, "
            "windows, or openings."
        )
    ]


def _object_anchor_lock_lines(source_payload: Mapping[str, object]) -> list[str]:
    canvas_width, canvas_height = _canvas_dimensions(source_payload)
    if canvas_width <= 0 or canvas_height <= 0:
        return []
    candidates: list[tuple[tuple[int, int, float], str]] = []
    for item in _sequence(source_payload.get("visibleObjects")):
        payload = _mapping(item)
        object_type = _canonical_object_type(payload)
        object_id = _string(payload.get("id")) or object_type
        bbox = _normalized_screen_bbox(
            payload.get("screenBboxPx"),
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        if not object_type or bbox is None:
            continue
        x, y, width, height = bbox
        area = width * height
        near_edge = _bbox_edge_phrase(bbox)
        is_anchor_type = object_type in _OBJECT_ANCHOR_TYPES
        if area < 0.004 and not near_edge and not is_anchor_type:
            continue
        position = _bbox_position_phrase(bbox)
        bbox_text = f"x={x:.3f}, y={y:.3f}, w={width:.3f}, h={height:.3f}"
        edge_text = (
            f" It touches/clips the {near_edge}; preserve that crop and do not "
            "complete it or move it inward."
            if near_edge
            else ""
        )
        line = (
            f"Object anchor lock: preserve {object_id} ({object_type}) at the "
            f"{position} source bbox ({bbox_text}); keep silhouette, scale, "
            f"support relation, and relative position.{edge_text}"
        )
        priority = (
            0 if near_edge else 1,
            0 if is_anchor_type else 1,
            -area,
        )
        candidates.append((priority, line))
    return [
        line
        for _, line in sorted(candidates, key=lambda candidate: candidate[0])[
            :_MAX_OBJECT_ANCHOR_LOCKS
        ]
    ]


def _opening_lock_lines(source_payload: Mapping[str, object]) -> list[str]:
    room = _mapping(source_payload.get("room"))
    openings = _mapping(room.get("openings"))
    raw_windows = openings.get("windows")
    raw_doors = openings.get("doors")
    window_count = len(raw_windows) if isinstance(raw_windows, list) else 0
    door_count = len(raw_doors) if isinstance(raw_doors, list) else 0
    lines = [
        (
            f"Metadata opening lock: preserve exactly {window_count} window "
            "opening(s); do not add, remove, resize, or move windows."
        ),
        (
            f"Metadata opening lock: preserve exactly {door_count} door "
            "opening(s); do not add, remove, resize, or move doors."
        ),
    ]
    if window_count == 0:
        lines.append(
            "No-window lock: keep walls, crop edges, curtains, and wall panels as "
            "non-window surfaces; do not create exterior garden/city/mountain views."
        )
    if door_count == 0:
        lines.append(
            "No-door lock: do not invent door panels, door frames, corridors, or "
            "open passageways."
        )
    return lines


def _canvas_dimensions(source_payload: Mapping[str, object]) -> tuple[float, float]:
    canvas = _mapping(source_payload.get("canvas"))
    width = _positive_float(canvas.get("widthPx")) or _positive_float(
        canvas.get("width")
    )
    height = _positive_float(canvas.get("heightPx")) or _positive_float(
        canvas.get("height")
    )
    return width or 0.0, height or 0.0


def _normalized_screen_bbox(
    value: object,
    *,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float, float, float] | None:
    payload = _mapping(value)
    min_x = _float_value(payload.get("minX"))
    min_y = _float_value(payload.get("minY"))
    if min_x is None:
        min_x = _float_value(payload.get("x"))
    if min_y is None:
        min_y = _float_value(payload.get("y"))
    if min_x is None or min_y is None:
        return None

    max_x = _float_value(payload.get("maxX"))
    max_y = _float_value(payload.get("maxY"))
    width = _float_value(payload.get("width"))
    height = _float_value(payload.get("height"))
    if max_x is None and width is not None:
        max_x = min_x + width
    if max_y is None and height is not None:
        max_y = min_y + height
    if max_x is None or max_y is None:
        return None

    clamped_min_x = min(max(min_x, 0.0), canvas_width)
    clamped_min_y = min(max(min_y, 0.0), canvas_height)
    clamped_max_x = min(max(max_x, 0.0), canvas_width)
    clamped_max_y = min(max(max_y, 0.0), canvas_height)
    normalized_width = max(0.0, (clamped_max_x - clamped_min_x) / canvas_width)
    normalized_height = max(0.0, (clamped_max_y - clamped_min_y) / canvas_height)
    if normalized_width <= 0.0 or normalized_height <= 0.0:
        return None
    return (
        clamped_min_x / canvas_width,
        clamped_min_y / canvas_height,
        normalized_width,
        normalized_height,
    )


def _bbox_edge_phrase(bbox: tuple[float, float, float, float]) -> str:
    x, y, width, height = bbox
    edges: list[str] = []
    if x <= 0.03:
        edges.append("left edge")
    if y <= 0.03:
        edges.append("top edge")
    if x + width >= 0.97:
        edges.append("right edge")
    if y + height >= 0.97:
        edges.append("bottom edge")
    return " and ".join(edges)


def _bbox_position_phrase(bbox: tuple[float, float, float, float]) -> str:
    x, y, width, height = bbox
    center_x = x + width / 2
    center_y = y + height / 2
    horizontal = "left" if center_x < 0.36 else "right" if center_x > 0.64 else "center"
    vertical = "upper" if center_y < 0.36 else "lower" if center_y > 0.64 else "middle"
    return f"{vertical}-{horizontal}"


def _visible_object_type_counts(source_payload: Mapping[str, object]) -> Counter[str]:
    return Counter(
        object_type
        for item in _sequence(source_payload.get("visibleObjects"))
        if (object_type := _canonical_object_type(_mapping(item)))
    )


def _canonical_object_type(payload: Mapping[str, object]) -> str:
    raw_type = _string(payload.get("type")) or _string(payload.get("id"))
    return re.sub(r"_\d+$", "", _clean_label(raw_type))


def _cache_key(
    *,
    source_image: Path,
    metadata_summary: Mapping[str, JSONValue],
    include_metadata: bool,
    model: str,
    profile: VisualPreflightProfile,
) -> str:
    digest = hashlib.sha256()
    digest.update(source_image.read_bytes())
    digest.update(b"\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(include_metadata).encode("ascii"))
    digest.update(b"\0")
    digest.update(_PREFLIGHT_PROMPT_VERSIONS[profile].encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(metadata_summary, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def _extract_chat_report(payload: object) -> Mapping[str, object]:
    root = _mapping(payload)
    choices = root.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Visual preflight response has no choices.")
    message = _mapping(_mapping(choices[0]).get("message"))
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Visual preflight response has no string content.")
    return _mapping(json.loads(content))


def _api_key() -> str | None:
    for name in (_OPENAI_API_KEY_ENV, _OPENAI_IMAGE_API_KEY_ENV):
        value = os.getenv(name, "").strip()
        if value:
            return value
    try:
        from config.openai_config import OpenAIConfig  # noqa: PLC0415

        return OpenAIConfig.OPENAI_API_KEY
    except Exception:
        return None


def _model() -> str:
    return os.getenv(_MODEL_ENV, "").strip() or _DEFAULT_MODEL


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: object) -> float | None:
    parsed = _float_value(value)
    if parsed is None or parsed <= 0.0:
        return None
    return parsed


def _float_value(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_preflight_profile(
    value: object,
    *,
    fallback: VisualPreflightProfile,
) -> VisualPreflightProfile:
    if value in {
        "classic",
        "room_shell",
        "object_anchors",
        "edit_scope",
        "edit_object_scope",
    }:
        return cast(VisualPreflightProfile, value)
    return fallback


def _clean_label(value: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", _string(value).lower()).strip("_")


def _clean_lock(value: object) -> str | None:
    cleaned = " ".join(_string(value).split())
    if not cleaned:
        return None
    return cleaned[:_MAX_LOCK_CHARS]


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sequence(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, dict) else {}


def _json_mapping(value: object) -> Mapping[str, JSONValue]:
    safe = _json_safe(value)
    return cast(Mapping[str, JSONValue], safe) if isinstance(safe, dict) else {}


def _json_safe(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
