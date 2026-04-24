from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from config import root_config

logger = logging.getLogger(__name__)

GEMINI_IMAGE_API_KEY_ENV = "GEMINI_IMAGE_API_KEY"
GEMINI_IMAGE_BASE_URL_ENV = "GEMINI_IMAGE_BASE_URL"
SNAPSHOT_RENDER_IMAGE_MODEL_ENV = "TKNT_SNAPSHOT_RENDER_IMAGE_MODEL"
SNAPSHOT_RENDER_IMAGE_SIZE_ENV = "TKNT_SNAPSHOT_RENDER_IMAGE_SIZE"
SNAPSHOT_RENDER_MAX_TOKENS_ENV = "TKNT_SNAPSHOT_RENDER_MAX_TOKENS"
SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE_ENV = (
    "TKNT_SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE"
)

_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
_DEFAULT_IMAGE_SIZE = "1K"
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_USER_PROMPT = (
    "Render a photorealistic premium interior with refined materials, realistic "
    "lighting, and polished finishes."
)
_SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_SUPPORTED_ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("1:1", 1.0),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:3", 4 / 3),
    ("4:5", 4 / 5),
    ("5:4", 5 / 4),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("21:9", 21 / 9),
)
_GENERATE_SYSTEM_PROMPT = (
    "You are converting a coarse 3D interior blockout into a final photorealistic render.\n\n"
    "The first input image is the only source of truth for geometry, architecture, layout, and object placement.\n"
    "If a second image is provided, it is only a supplemental top-down 2D layout reference for clarifying object identity and footprint when the first image is ambiguous.\n"
    "The second image must never override the first image's geometry, camera angle, framing, visibility, or object arrangement.\n"
    "All style, lighting, scenery, and reference-image cues are secondary and may only modify appearance.\n\n"
    "Preserve exactly:\n"
    "- room type and function\n"
    "- camera angle, framing, and perspective\n"
    "- room boundary, wall positions, floor extent, and ceiling shape\n"
    "- all visible doors, windows, and openings: preserve count, location, size, proportions, and relative arrangement\n"
    "- all major furniture and built-in elements: preserve count, placement, orientation, and scale\n"
    "- circulation paths and empty-space distribution\n\n"
    "Do not add, remove, move, enlarge, shrink, merge, split, or replace any wall, opening, door, window, partition, built-in, or major furniture item.\n"
    "Do not invent new architecture.\n"
    "Do not convert solid wall into glass.\n"
    "Do not convert windows into decoration or decoration into windows.\n\n"
    "Apply style, materials, lighting, and scenery only as surface-level refinements:\n"
    "- materials\n"
    "- textures\n"
    "- colors\n"
    "- lighting mood\n"
    "- realism and detail\n\n"
    "Exterior scenery may appear only within openings that are already clearly visible in the source image and must match their original position and size.\n\n"
    "Render a photorealistic premium interior while preserving the original blockout structure.\n"
    "No people, no text, no watermark, no room-type change."
)
_EDIT_SYSTEM_PROMPT = (
    "You are editing an existing photorealistic interior image.\n\n"
    "The first input image is the source image to edit. The app supplies the selected object id, object name, camera metadata, and visibility metadata; do not guess which object is selected from the pixels alone.\n"
    "If additional reference images are provided, use them only for the explicitly paired replacement instructions.\n\n"
    "Preserve the room layout, camera angle, perspective, framing, architecture, and every object that is not explicitly selected for editing.\n"
    "Do not add, remove, move, resize, merge, split, or reinterpret any unselected object.\n"
    "Apply only the requested object replacement, object color change, and/or whole-image preset changes.\n"
    "No people, no text, no watermark, no room-type change."
)

RenderMode = Literal["generate", "edit"]


@dataclass(frozen=True)
class SnapshotImageRenderConfig:
    image_model_name: str = _DEFAULT_IMAGE_MODEL
    image_size: str = _DEFAULT_IMAGE_SIZE
    max_output_tokens: int = _DEFAULT_MAX_TOKENS
    gemini_base_url: str = _DEFAULT_GEMINI_BASE_URL
    gemini_api_key: str | None = None
    include_layout_2d_reference: bool = False

    @classmethod
    def from_env(cls) -> SnapshotImageRenderConfig:
        image_config = root_config.services.gemini_image
        image_size = _string(
            os.getenv(SNAPSHOT_RENDER_IMAGE_SIZE_ENV),
            fallback=image_config.image_size or _DEFAULT_IMAGE_SIZE,
        ).upper()
        max_tokens = _read_int_env(
            SNAPSHOT_RENDER_MAX_TOKENS_ENV,
            default=image_config.max_output_tokens,
        )
        env_api_key = _secret_string(os.getenv(GEMINI_IMAGE_API_KEY_ENV))
        config_api_key = _secret_string(image_config.api_key)
        return cls(
            image_model_name=_normalize_gemini_model_name(
                _string(
                    os.getenv(SNAPSHOT_RENDER_IMAGE_MODEL_ENV),
                    fallback=image_config.model or _DEFAULT_IMAGE_MODEL,
                )
            ),
            image_size=image_size or _DEFAULT_IMAGE_SIZE,
            max_output_tokens=max(32, max_tokens),
            gemini_base_url=_string(
                os.getenv(GEMINI_IMAGE_BASE_URL_ENV),
                fallback=image_config.base_url or _DEFAULT_GEMINI_BASE_URL,
            ),
            gemini_api_key=env_api_key or config_api_key,
            include_layout_2d_reference=_read_bool_env(
                SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE_ENV,
                default=bool(image_config.include_layout_2d_reference),
            ),
        )


@dataclass(frozen=True)
class SnapshotEditOperation:
    object_id: str
    object_name: str | None = None
    replacement_image_data_url: str | None = None
    target_color: str | None = None


def render_snapshot_image(
    snapshot_payload: Mapping[str, object],
    *,
    snapshot_image_data_url: str,
    user_prompt: str | None = None,
    layout_reference_image_data_url: str | None = None,
    render_mode: RenderMode = "generate",
    preset_selection: Mapping[str, object] | None = None,
    edit_operations: list[SnapshotEditOperation] | None = None,
    edit_source_image_data_url: str | None = None,
    config: SnapshotImageRenderConfig | None = None,
) -> dict[str, object]:
    resolved_config = config or SnapshotImageRenderConfig.from_env()
    structural_image_data_url = _normalize_image_data_url(snapshot_image_data_url)
    source_image_data_url = (
        _normalize_image_data_url(edit_source_image_data_url)
        if render_mode == "edit" and edit_source_image_data_url
        else structural_image_data_url
    )
    visible_objects = _extract_visible_objects(snapshot_payload)
    normalized_operations = _normalize_edit_operations(
        edit_operations or [],
        visible_objects=visible_objects,
    )
    preset_prompt = _build_preset_prompt(preset_selection)
    normalized_user_prompt = _build_final_user_prompt(
        user_prompt=user_prompt,
        render_mode=render_mode,
        preset_prompt=preset_prompt,
        edit_operations=normalized_operations,
        camera_payload=_mapping(snapshot_payload.get("camera")),
        visible_objects=visible_objects,
    )
    aspect_ratio = _resolve_canvas_aspect_ratio(snapshot_payload)
    normalized_layout_reference = (
        _normalize_image_data_url(layout_reference_image_data_url)
        if layout_reference_image_data_url
        else None
    )
    layout_reference_used = (
        render_mode == "generate"
        and resolved_config.include_layout_2d_reference
        and normalized_layout_reference is not None
    )
    reference_images = [
        operation.replacement_image_data_url
        for operation in normalized_operations
        if operation.replacement_image_data_url is not None
    ]

    image_result = _generate_image_with_gemini(
        model_name=resolved_config.image_model_name,
        image_size=resolved_config.image_size,
        max_output_tokens=resolved_config.max_output_tokens,
        gemini_base_url=resolved_config.gemini_base_url,
        gemini_api_key=resolved_config.gemini_api_key,
        aspect_ratio=aspect_ratio,
        system_prompt=(
            _EDIT_SYSTEM_PROMPT if render_mode == "edit" else _GENERATE_SYSTEM_PROMPT
        ),
        user_prompt=normalized_user_prompt,
        source_image_data_url=source_image_data_url,
        layout_reference_image_data_url=(
            normalized_layout_reference if layout_reference_used else None
        ),
        reference_image_data_urls=reference_images,
    )

    return {
        "request": {
            "render_mode": render_mode,
            "system_prompt": (
                _EDIT_SYSTEM_PROMPT
                if render_mode == "edit"
                else _GENERATE_SYSTEM_PROMPT
            ),
            "user_prompt": normalized_user_prompt,
            "raw_user_prompt": _string(user_prompt) or None,
            "preset_prompt": preset_prompt,
            "preset_selection": dict(preset_selection or {}),
            "aspect_ratio": aspect_ratio,
            "source_image_mime_type": _extract_data_url_mime_type(
                source_image_data_url
            ),
            "structural_image_mime_type": _extract_data_url_mime_type(
                structural_image_data_url
            ),
            "layout_reference_enabled": resolved_config.include_layout_2d_reference,
            "layout_reference_used": layout_reference_used,
            "layout_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_layout_reference)
                if layout_reference_used and normalized_layout_reference is not None
                else None
            ),
            "reference_image_count": len(reference_images),
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in normalized_operations
            ],
        },
        "image": image_result,
        "models": {
            "image_model_name": resolved_config.image_model_name,
            "max_tokens": resolved_config.max_output_tokens,
        },
        "metadata": {
            "camera": snapshot_payload.get("camera"),
            "visible_objects": list(visible_objects.values()),
            "visible_object_ids": list(visible_objects.keys()),
            "render_mode": render_mode,
            "preset_selection": dict(preset_selection or {}),
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in normalized_operations
            ],
        },
    }


def render_snapshot_image_from_path(
    snapshot_path: str | Path,
    *,
    user_prompt: str | None = None,
    snapshot_image_data_url: str | None = None,
    layout_reference_image_data_url: str | None = None,
    render_mode: RenderMode = "generate",
    preset_selection: Mapping[str, object] | None = None,
    edit_operations: list[SnapshotEditOperation] | None = None,
    edit_source_image_data_url: str | None = None,
    config: SnapshotImageRenderConfig | None = None,
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
        user_prompt=user_prompt,
        layout_reference_image_data_url=layout_reference_image_data_url,
        render_mode=render_mode,
        preset_selection=preset_selection,
        edit_operations=edit_operations,
        edit_source_image_data_url=edit_source_image_data_url,
        config=config,
    )


def _generate_image_with_gemini(
    *,
    model_name: str,
    image_size: str,
    max_output_tokens: int,
    gemini_base_url: str,
    gemini_api_key: str | None,
    aspect_ratio: str,
    system_prompt: str,
    user_prompt: str,
    source_image_data_url: str,
    layout_reference_image_data_url: str | None,
    reference_image_data_urls: list[str],
) -> dict[str, object]:
    if not gemini_api_key:
        raise ValueError(
            "Missing Gemini image API key. Set services.gemini_image.api_key in "
            "app-config.yaml or set GEMINI_IMAGE_API_KEY before using AI image rendering."
        )

    endpoint = f"{gemini_base_url.rstrip('/')}/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_api_key,
    }
    content: list[dict[str, object]] = [
        {"text": user_prompt},
        _data_url_to_inline_part(source_image_data_url),
    ]
    if layout_reference_image_data_url is not None:
        content.append(_data_url_to_inline_part(layout_reference_image_data_url))
    content.extend(
        _data_url_to_inline_part(data_url) for data_url in reference_image_data_urls
    )

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": content,
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            },
            "maxOutputTokens": max_output_tokens,
        },
    }

    with httpx.Client(timeout=600.0) as client:
        response = client.post(endpoint, json=payload, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text.strip()
            detail = f"{exc}. Gemini response body: {body}" if body else str(exc)
            raise RuntimeError(detail) from exc

    parsed = response.json()
    if not isinstance(parsed, dict):
        raise ValueError("Gemini image response must be a JSON object.")
    image_bytes = _extract_image_bytes(parsed)
    if image_bytes is None:
        raise ValueError("Gemini image response did not contain image data.")
    mime_type = _guess_image_mime_type(image_bytes)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "mime_type": mime_type,
        "image_base64": image_base64,
        "data_url": f"data:{mime_type};base64,{image_base64}",
        "size": image_size,
        "aspect_ratio": aspect_ratio,
        "raw_response": parsed,
    }


def _build_preset_prompt(preset_selection: Mapping[str, object] | None) -> str:
    if preset_selection is None:
        return ""

    preset_config = root_config.presets
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


def _build_final_user_prompt(
    *,
    user_prompt: str | None,
    render_mode: RenderMode,
    preset_prompt: str,
    edit_operations: list[SnapshotEditOperation],
    camera_payload: Mapping[str, object],
    visible_objects: Mapping[str, Mapping[str, object]],
) -> str:
    raw_user_prompt = _string(user_prompt)
    if render_mode == "generate":
        prompt_parts = [raw_user_prompt or _DEFAULT_USER_PROMPT]
        if preset_prompt:
            prompt_parts.append("Apply these selected presets:\n" + preset_prompt)
        return "\n\n".join(prompt_parts)

    if not raw_user_prompt and not preset_prompt and not edit_operations:
        raise ValueError("Provide at least one edit operation or preset change.")

    prompt_parts = [
        "Edit the provided source image while preserving the room layout, camera, and every unselected object.",
        "Use the app-provided object ids and visibility metadata below to identify edit targets.",
    ]
    if camera_payload:
        prompt_parts.append("Camera metadata: " + _compact_json(camera_payload))
    if visible_objects:
        visible_summary = [
            f"{object_id} ({_visible_object_name(payload)})"
            for object_id, payload in visible_objects.items()
        ]
        prompt_parts.append(
            "Objects visible in this source image: " + ", ".join(visible_summary)
        )
    if raw_user_prompt:
        prompt_parts.append("Additional user direction: " + raw_user_prompt)
    if preset_prompt:
        prompt_parts.append("Whole-image preset update:\n" + preset_prompt)

    replacement_index = 1
    for operation in edit_operations:
        if operation.replacement_image_data_url is not None:
            prompt_parts.append(
                "Selected object id: "
                f"{operation.object_id}. Reference image {replacement_index} belongs to this operation.\n"
                f"Replace only the selected {operation.object_name} with the uploaded reference object. "
                "Keep the same position, scale, and function as much as possible. "
                "Preserve the room layout, camera, and all other objects."
            )
            replacement_index += 1
        if operation.target_color is not None:
            prompt_parts.append(
                f"Selected object id: {operation.object_id}.\n"
                f"Change only the color of the selected {operation.object_name} to {operation.target_color}. "
                "Keep its shape, material appearance, and position unchanged as much as possible. "
                "Preserve the room layout, camera, and all other objects."
            )

    return "\n\n".join(prompt_parts)


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
        visible_payload = visible_objects.get(object_id)
        if visible_payload is None:
            raise ValueError(
                f"Object {object_id} is not visible in the source image and cannot be edited."
            )

        replacement_image_data_url = (
            _normalize_image_data_url(operation.replacement_image_data_url)
            if operation.replacement_image_data_url
            else None
        )
        target_color = _string(operation.target_color) or None
        if replacement_image_data_url is None and target_color is None:
            continue

        object_name = _string(operation.object_name) or _visible_object_name(
            visible_payload
        )
        normalized.append(
            SnapshotEditOperation(
                object_id=object_id,
                object_name=object_name,
                replacement_image_data_url=replacement_image_data_url,
                target_color=target_color,
            )
        )
    return normalized


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
    for key in ("label", "type", "canonicalType", "canonical_type", "raw_type"):
        value = _string(payload.get(key))
        if value:
            return value.replace("_", " ")
    object_id = _string(payload.get("id"))
    return object_id or "object"


def _serialize_edit_operation(operation: SnapshotEditOperation) -> dict[str, object]:
    return {
        "object_id": operation.object_id,
        "object_name": operation.object_name,
        "replacement_image_provided": operation.replacement_image_data_url is not None,
        "target_color": operation.target_color,
    }


def _data_url_to_inline_part(data_url: str) -> dict[str, object]:
    mime_type = _extract_data_url_mime_type(data_url)
    _, _, image_base64 = data_url.partition(",")
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": image_base64,
        }
    }


def _normalize_user_prompt(user_prompt: str | None) -> str:
    normalized = _string(user_prompt)
    if normalized:
        return normalized
    return _DEFAULT_USER_PROMPT


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
    return normalized


def _extract_data_url_mime_type(data_url: str) -> str:
    prefix, _, _ = data_url.partition(",")
    mime_section = prefix.removeprefix("data:")
    mime_type, _, _ = mime_section.partition(";")
    return mime_type or "application/octet-stream"


def _extract_image_bytes(payload: Mapping[str, object]) -> bytes | None:
    for inline_key in ("inlineData", "inline_data"):
        inline_payload = payload.get(inline_key)
        if isinstance(inline_payload, Mapping):
            mime_type = inline_payload.get("mimeType") or inline_payload.get(
                "mime_type"
            )
            data = inline_payload.get("data")
            if isinstance(mime_type, str) and mime_type.startswith("image/"):
                if isinstance(data, str):
                    decoded = _decode_base64_image(data)
                    if decoded is not None:
                        return decoded

    direct_value = payload.get("b64_json") or payload.get("image") or payload.get("url")
    if isinstance(direct_value, str):
        return _decode_base64_image(direct_value)

    for value in payload.values():
        if isinstance(value, Mapping):
            nested = _extract_image_bytes(value)
            if nested is not None:
                return nested
        elif isinstance(value, list):
            nested = _extract_image_bytes_from_list(value)
            if nested is not None:
                return nested
    return None


def _extract_image_bytes_from_list(values: list[object]) -> bytes | None:
    for item in values:
        if isinstance(item, str):
            decoded = _decode_base64_image(item)
            if decoded is not None:
                return decoded
        elif isinstance(item, Mapping):
            nested = _extract_image_bytes(item)
            if nested is not None:
                return nested
    return None


def _decode_base64_image(value: str) -> bytes | None:
    normalized = value.strip()
    if normalized.startswith("data:image/") and "," in normalized:
        normalized = normalized.split(",", maxsplit=1)[1]
    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError):
        return None


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


def _positive_float(value: object, *, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if numeric <= 0:
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


def _normalize_gemini_model_name(value: str) -> str:
    normalized = _string(value, fallback=_DEFAULT_IMAGE_MODEL)
    if "/" in normalized:
        return normalized.rsplit("/", maxsplit=1)[-1]
    return normalized


def _compact_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
