from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import httpx

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
EditOperationType = Literal["prompt", "add", "replace", "recolor", "remove"]
VisualPreflightProfile = Literal[
    "classic",
    "room_shell",
    "object_anchors",
    "edit_scope",
    "edit_object_scope",
]

_MODEL_ENV = "TKNT_IMAGE_FLOW_PREFLIGHT_MODEL"
_BASE_URL_ENV = "TKNT_IMAGE_FLOW_PREFLIGHT_OPENAI_BASE_URL"
_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_OPENAI_IMAGE_API_KEY_ENV = "OPENAI_IMAGE_API_KEY"
_OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
_DEFAULT_MODEL = "gpt-5.4-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_PROMPT_VERSION = "edit-object-scope-v1"
_MAX_VISIBLE_OBJECTS = 24
_MAX_EDIT_SCOPE_LOCKS = 6
_MAX_PRIORITY_LOCKS = 4
_MAX_LOCK_CHARS = 260
_RISK_LABELS = {
    "bbox_boundary_risk",
    "edit_target_ambiguous",
    "foreground_occlusion",
    "low_contrast_object",
    "replacement_reference_risk",
    "scale_sensitive_object",
    "tight_crop",
}


class VisualPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class BBoxNorm:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class EditOperationConfig:
    object_id: str
    object_name: str | None = None
    operation_type: EditOperationType | None = None
    prompt: str | None = None
    bbox_norm: BBoxNorm | None = None
    replacement_image: Path | None = None
    target_color: str | None = None


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
    edit_scope_locks: tuple[str, ...] = ()
    preflight_profile: VisualPreflightProfile = "edit_object_scope"

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "source_valid": self.source_valid,
            "risk_profile": list(self.risk_profile),
            "required_counts": dict(self.required_counts),
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
    profile: VisualPreflightProfile = "edit_object_scope",
    edit_operations: Sequence[EditOperationConfig] = (),
) -> VisualPreflightResult:
    if profile != "edit_object_scope":
        raise VisualPreflightError(
            f"Unsupported production visual preflight profile: {profile}"
        )

    model = _model()
    metadata_summary = _metadata_summary(source_payload)
    metadata_summary["edit_operations"] = _edit_operations_summary(edit_operations)
    cache_key = _cache_key(
        source_image=source_image,
        metadata_summary=metadata_summary if include_metadata else {},
        include_metadata=include_metadata,
        model=model,
    )
    shared_cache_path = (
        cache_dir / f"{cache_key}.json" if cache_dir is not None else None
    )
    cached = _load_cached_report(report_path, cache_key=cache_key)
    if cached is None and shared_cache_path is not None:
        cached = _load_cached_report(shared_cache_path, cache_key=cache_key)
    if cached is not None:
        _write_report(report_path, cached)
        return cached

    api_key = _api_key()
    if not api_key:
        raise VisualPreflightError(
            "Visual preflight requires OPENAI_API_KEY or OPENAI_IMAGE_API_KEY."
        )

    metadata_text = (
        json.dumps(metadata_summary, ensure_ascii=False, separators=(",", ":"))
        if include_metadata
        else "omitted"
    )
    body: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a visual preflight controller for interior image "
                    "editing. Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _edit_object_scope_preflight_prompt(metadata_text),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_image_to_data_url(source_image)},
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    endpoint = _base_url()
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
    except (httpx.HTTPError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise VisualPreflightError(
            f"Visual preflight request failed: {type(exc).__name__}: {exc}"
        ) from exc

    result = _normalize_report(
        report,
        model=model,
        cache_key=cache_key,
        cache_status="miss",
        usage=usage,
    )
    _write_report(report_path, result)
    if shared_cache_path is not None:
        _write_report(shared_cache_path, result)
    return result


def compile_visual_preflight_prompt(
    result: VisualPreflightResult,
    *,
    edit_operations: Sequence[EditOperationConfig] = (),
) -> str:
    lines = _edit_operation_scope_lock_lines(edit_operations)
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
    return [
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
        for operation in edit_operations
    ]


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


def _normalize_report(
    report: Mapping[str, object],
    *,
    model: str,
    cache_key: str,
    cache_status: Literal["hit", "miss"],
    usage: Mapping[str, JSONValue],
) -> VisualPreflightResult:
    risk_profile = tuple(
        sorted(
            {
                label
                for item in _sequence(report.get("risk_profile"))
                if (label := _clean_label(item)) in _RISK_LABELS
            }
        )
    )
    return VisualPreflightResult(
        source_valid=report.get("source_valid") is True,
        risk_profile=risk_profile,
        required_counts={},
        priority_locks=tuple(
            lock
            for item in _sequence(report.get("priority_locks"))[:_MAX_PRIORITY_LOCKS]
            if (lock := _clean_lock(item))
        ),
        reason=_clean_lock(report.get("reason")) or "",
        model=model,
        cache_key=cache_key,
        cache_status=cache_status,
        usage=usage,
        edit_scope_locks=tuple(
            lock
            for item in _sequence(report.get("edit_scope_locks"))[
                :_MAX_EDIT_SCOPE_LOCKS
            ]
            if (lock := _clean_lock(item))
        ),
        preflight_profile="edit_object_scope",
    )


def _load_cached_report(
    report_path: Path,
    *,
    cache_key: str,
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
        model=str(payload.get("model") or _model()),
        cache_key=cache_key,
        cache_status="hit",
        usage=_json_mapping(payload.get("usage")),
    )


def _write_report(path: Path, result: VisualPreflightResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _cache_key(
    *,
    source_image: Path,
    metadata_summary: Mapping[str, JSONValue],
    include_metadata: bool,
    model: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(source_image.read_bytes())
    digest.update(b"\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(include_metadata).encode("ascii"))
    digest.update(b"\0")
    digest.update(_PROMPT_VERSION.encode("ascii"))
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
        from config.openai_config import OpenAIConfig
    except ImportError:
        return None
    value = getattr(OpenAIConfig, "OPENAI_API_KEY", "")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _model() -> str:
    return os.getenv(_MODEL_ENV, "").strip() or _DEFAULT_MODEL


def _base_url() -> str:
    raw_value = (
        os.getenv(_BASE_URL_ENV)
        or os.getenv(_OPENAI_BASE_URL_ENV)
        or _DEFAULT_BASE_URL
    )
    clean = raw_value.strip() if isinstance(raw_value, str) else ""
    if not clean or clean in {"v1", "/v1"} or (
        clean.startswith("${") and clean.endswith("}")
    ):
        return _DEFAULT_BASE_URL
    if "://" not in clean:
        clean = f"https://{clean.lstrip('/')}"
    return clean.rstrip("/")


def _encode_image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _canonical_object_type(payload: Mapping[str, object]) -> str:
    raw_type = _string(payload.get("type")) or _string(payload.get("id"))
    return re.sub(r"_\d+$", "", _clean_label(raw_type))


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
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
