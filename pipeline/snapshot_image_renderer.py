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

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
OPENROUTER_SITE_URL_ENV = "OPENROUTER_SITE_URL"
OPENROUTER_SITE_NAME_ENV = "OPENROUTER_SITE_NAME"
SNAPSHOT_RENDER_IMAGE_MODEL_ENV = "TKNT_SNAPSHOT_RENDER_IMAGE_MODEL"
SNAPSHOT_RENDER_IMAGE_SIZE_ENV = "TKNT_SNAPSHOT_RENDER_IMAGE_SIZE"
SNAPSHOT_RENDER_MAX_TOKENS_ENV = "TKNT_SNAPSHOT_RENDER_MAX_TOKENS"
SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE_ENV = (
    "TKNT_SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE"
)

_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_IMAGE_MODEL = "google/gemini-3.1-flash-image-preview"
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
_GEMINI_SYSTEM_PROMPT = (
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


@dataclass(frozen=True)
class SnapshotImageRenderConfig:
    image_model_name: str = _DEFAULT_IMAGE_MODEL
    image_size: str = _DEFAULT_IMAGE_SIZE
    max_tokens: int = _DEFAULT_MAX_TOKENS
    openrouter_base_url: str = _DEFAULT_OPENROUTER_BASE_URL
    openrouter_api_key: str | None = None
    openrouter_site_url: str | None = None
    openrouter_site_name: str | None = None
    include_layout_2d_reference: bool = False

    @classmethod
    def from_env(cls) -> SnapshotImageRenderConfig:
        image_size = _string(
            os.getenv(SNAPSHOT_RENDER_IMAGE_SIZE_ENV),
            fallback=_DEFAULT_IMAGE_SIZE,
        ).upper()
        max_tokens = _read_int_env(
            SNAPSHOT_RENDER_MAX_TOKENS_ENV,
            default=_DEFAULT_MAX_TOKENS,
        )
        return cls(
            image_model_name=_string(
                os.getenv(SNAPSHOT_RENDER_IMAGE_MODEL_ENV),
                fallback=_DEFAULT_IMAGE_MODEL,
            ),
            image_size=image_size or _DEFAULT_IMAGE_SIZE,
            max_tokens=max(32, max_tokens),
            openrouter_base_url=_string(
                os.getenv(OPENROUTER_BASE_URL_ENV),
                fallback=_DEFAULT_OPENROUTER_BASE_URL,
            ),
            openrouter_api_key=_string(os.getenv(OPENROUTER_API_KEY_ENV)) or None,
            openrouter_site_url=_string(os.getenv(OPENROUTER_SITE_URL_ENV)) or None,
            openrouter_site_name=_string(os.getenv(OPENROUTER_SITE_NAME_ENV)) or None,
            include_layout_2d_reference=_read_bool_env(
                SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE_ENV,
                default=False,
            ),
        )


def render_snapshot_image(
    snapshot_payload: Mapping[str, object],
    *,
    snapshot_image_data_url: str,
    user_prompt: str | None = None,
    layout_reference_image_data_url: str | None = None,
    config: SnapshotImageRenderConfig | None = None,
) -> dict[str, object]:
    resolved_config = config or SnapshotImageRenderConfig.from_env()
    source_image_data_url = _normalize_image_data_url(snapshot_image_data_url)
    normalized_user_prompt = _normalize_user_prompt(user_prompt)
    aspect_ratio = _resolve_canvas_aspect_ratio(snapshot_payload)
    normalized_layout_reference = (
        _normalize_image_data_url(layout_reference_image_data_url)
        if layout_reference_image_data_url
        else None
    )
    layout_reference_used = (
        resolved_config.include_layout_2d_reference
        and normalized_layout_reference is not None
    )

    image_result = _generate_image_with_openrouter(
        model_name=resolved_config.image_model_name,
        image_size=resolved_config.image_size,
        max_tokens=resolved_config.max_tokens,
        openrouter_base_url=resolved_config.openrouter_base_url,
        openrouter_api_key=resolved_config.openrouter_api_key,
        openrouter_site_url=resolved_config.openrouter_site_url,
        openrouter_site_name=resolved_config.openrouter_site_name,
        aspect_ratio=aspect_ratio,
        system_prompt=_GEMINI_SYSTEM_PROMPT,
        user_prompt=normalized_user_prompt,
        source_image_data_url=source_image_data_url,
        layout_reference_image_data_url=(
            normalized_layout_reference if layout_reference_used else None
        ),
    )

    return {
        "request": {
            "system_prompt": _GEMINI_SYSTEM_PROMPT,
            "user_prompt": normalized_user_prompt,
            "aspect_ratio": aspect_ratio,
            "source_image_mime_type": _extract_data_url_mime_type(
                source_image_data_url
            ),
            "layout_reference_enabled": resolved_config.include_layout_2d_reference,
            "layout_reference_used": layout_reference_used,
            "layout_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_layout_reference)
                if layout_reference_used and normalized_layout_reference is not None
                else None
            ),
        },
        "image": image_result,
        "models": {
            "image_model_name": resolved_config.image_model_name,
            "max_tokens": resolved_config.max_tokens,
        },
    }


def render_snapshot_image_from_path(
    snapshot_path: str | Path,
    *,
    user_prompt: str | None = None,
    snapshot_image_data_url: str | None = None,
    layout_reference_image_data_url: str | None = None,
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
        config=config,
    )


def _generate_image_with_openrouter(
    *,
    model_name: str,
    image_size: str,
    max_tokens: int,
    openrouter_base_url: str,
    openrouter_api_key: str | None,
    openrouter_site_url: str | None,
    openrouter_site_name: str | None,
    aspect_ratio: str,
    system_prompt: str,
    user_prompt: str,
    source_image_data_url: str,
    layout_reference_image_data_url: str | None,
) -> dict[str, object]:
    if not openrouter_api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Set it in the backend environment before "
            "using AI image rendering."
        )

    endpoint = openrouter_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if openrouter_site_url:
        headers["HTTP-Referer"] = openrouter_site_url
    if openrouter_site_name:
        headers["X-OpenRouter-Title"] = openrouter_site_name

    content: list[dict[str, object]] = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": source_image_data_url}},
    ]
    if layout_reference_image_data_url is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": layout_reference_image_data_url},
            }
        )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content,
            },
        ],
        "modalities": ["image"],
        "image_config": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
        },
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=600.0) as client:
        response = client.post(endpoint, json=payload, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text.strip()
            detail = f"{exc}. OpenRouter response body: {body}" if body else str(exc)
            raise RuntimeError(detail) from exc

    parsed = response.json()
    if not isinstance(parsed, dict):
        raise ValueError("OpenRouter image response must be a JSON object.")
    image_bytes = _extract_image_bytes(parsed)
    if image_bytes is None:
        raise ValueError("OpenRouter image response did not contain image data.")
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
