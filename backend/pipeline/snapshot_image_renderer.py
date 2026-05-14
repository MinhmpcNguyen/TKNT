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
from time import perf_counter
from typing import Literal
from uuid import uuid4

import httpx

from config import config_file, load_config, root_config
from pipeline.image_flow_logging import (
    log_image_flow_event,
    redact_for_image_log,
    summarize_gemini_payload,
    summarize_gemini_response,
    summarize_image_data_url,
)

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
_GEMINI_INLINE_REQUEST_LIMIT_BYTES = 20_000_000
_BYTES_PER_MB = 1_000_000
_DEFAULT_USER_PROMPT = (
    "Restyle the provided locked 3D target layout into a photorealistic premium "
    "interior with refined materials, realistic lighting, and polished finishes. "
    "Keep the exact target camera and screen-space placement of every visible "
    "object."
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
    "Create a photorealistic interior render from the input 3D layout image.\n"
    "The input 3D image is the exact camera and layout source. Match its view, "
    "framing, walls, openings, furniture placement, object count, object size, "
    "depth order, and occlusion.\n"
    "The 3D image may contain labels, colored boxes, guide lines, or callouts. "
    "Use them only to identify objects and boundaries. Do not render any label, "
    "number, guide line, box, outline, UI mark, or watermark.\n"
    "Only improve materials, finishes, lighting, scenery through existing "
    "openings, and overall photorealism. No people and no room-type change."
)
_EDIT_SYSTEM_PROMPT = (
    "Edit an existing photorealistic interior image.\n"
    "The first input image is the main image to edit.\n"
    "If a labeled 3D target-view reference is provided, change the main image to "
    "match that reference's camera, framing, walls, openings, furniture placement, "
    "object count, object size, depth order, and occlusion.\n"
    "Use any labels, colored boxes, guide lines, or callouts only to identify "
    "objects and boundaries. Do not render any label, number, guide line, box, "
    "outline, UI mark, or watermark.\n"
    "Apply only the requested object replacement, object color change, prompt "
    "change, or preset change. Preserve everything else. No people and no "
    "room-type change."
)

RenderMode = Literal["generate", "edit"]
SceneReferenceMode = Literal[
    "none",
    "target_layout_with_scene_reference",
    "scene_reference_camera_transfer",
]


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


@dataclass(frozen=True)
class _GeminiRequestAttempt:
    name: str
    payload: dict[str, object]
    generation_config_applied: bool


def render_snapshot_image(
    snapshot_payload: Mapping[str, object],
    *,
    snapshot_image_data_url: str,
    user_prompt: str | None = None,
    layout_reference_image_data_url: str | None = None,
    annotated_reference_image_data_url: str | None = None,
    scene_reference_image_data_url: str | None = None,
    scene_reference_mode: SceneReferenceMode = "none",
    render_mode: RenderMode = "generate",
    preset_selection: Mapping[str, object] | None = None,
    edit_operations: list[SnapshotEditOperation] | None = None,
    edit_source_image_data_url: str | None = None,
    config: SnapshotImageRenderConfig | None = None,
    trace_id: str | None = None,
) -> dict[str, object]:
    image_flow_trace_id = trace_id or uuid4().hex
    resolved_config = config or SnapshotImageRenderConfig.from_env()
    structural_image_data_url = _normalize_image_data_url(snapshot_image_data_url)
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
    source_image_data_url = (
        _normalize_image_data_url(edit_source_image_data_url)
        if render_mode == "edit" and edit_source_image_data_url
        else normalized_scene_reference
        if reference_only_camera_transfer_used
        and normalized_scene_reference is not None
        else structural_image_data_url
    )
    visible_objects = _extract_visible_objects(snapshot_payload)
    normalized_operations = _normalize_edit_operations(
        edit_operations or [],
        visible_objects=visible_objects,
    )
    preset_prompt = _build_preset_prompt(preset_selection)
    normalized_annotated_reference = (
        _normalize_image_data_url(annotated_reference_image_data_url)
        if annotated_reference_image_data_url
        else None
    )
    annotated_reference_used = (
        render_mode == "generate" and normalized_annotated_reference is not None
    )
    normalized_layout_reference = (
        _normalize_image_data_url(layout_reference_image_data_url)
        if layout_reference_image_data_url
        else None
    )
    layout_reference_enabled = (
        resolved_config.include_layout_2d_reference or render_mode == "edit"
    )
    layout_reference_used = (
        normalized_layout_reference is not None
        and (
            render_mode == "edit"
            or (
                render_mode == "generate"
                and not reference_only_camera_transfer_used
                and not annotated_reference_used
                and layout_reference_enabled
            )
        )
    )
    layout_lock_prompt = _build_generate_layout_lock_prompt(
        scene_reference_mode=resolved_scene_reference_mode,
    )
    normalized_user_prompt = _build_final_user_prompt(
        user_prompt=user_prompt,
        render_mode=render_mode,
        preset_prompt=preset_prompt,
        edit_operations=normalized_operations,
        visible_objects=visible_objects,
        layout_lock_prompt=layout_lock_prompt,
        layout_reference_used=layout_reference_used,
    )
    aspect_ratio = _resolve_canvas_aspect_ratio(snapshot_payload)
    reference_images = [
        operation.replacement_image_data_url
        for operation in normalized_operations
        if operation.replacement_image_data_url is not None
    ]

    log_image_flow_event(
        "snapshot_image.render_input",
        {
            "trace_id": image_flow_trace_id,
            "render_mode": render_mode,
            "model_name": resolved_config.image_model_name,
            "image_size": resolved_config.image_size,
            "max_output_tokens": resolved_config.max_output_tokens,
            "aspect_ratio": aspect_ratio,
            "scene_reference_mode": resolved_scene_reference_mode,
            "layout_reference_enabled": layout_reference_enabled,
            "layout_reference_used": layout_reference_used,
            "annotated_reference_used": annotated_reference_used,
            "scene_reference_used": scene_reference_used,
            "reference_only_camera_transfer_used": (
                reference_only_camera_transfer_used
            ),
            "raw_user_prompt": _string(user_prompt) or None,
            "normalized_user_prompt": normalized_user_prompt,
            "preset_prompt": preset_prompt,
            "preset_selection": dict(preset_selection or {}),
            "visible_object_count": len(visible_objects),
            "visible_object_ids": list(visible_objects.keys()),
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in normalized_operations
            ],
            "snapshot_payload": dict(snapshot_payload),
            "images": {
                "source": summarize_image_data_url(source_image_data_url),
                "structural": summarize_image_data_url(structural_image_data_url),
                "annotated_reference": summarize_image_data_url(
                    normalized_annotated_reference
                    if annotated_reference_used
                    else None
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
            },
        },
    )

    try:
        image_result = _generate_image_with_gemini(
            trace_id=image_flow_trace_id,
            model_name=resolved_config.image_model_name,
            image_size=resolved_config.image_size,
            max_output_tokens=resolved_config.max_output_tokens,
            gemini_base_url=resolved_config.gemini_base_url,
            gemini_api_key=resolved_config.gemini_api_key,
            aspect_ratio=aspect_ratio,
            system_prompt=(
                _EDIT_SYSTEM_PROMPT
                if render_mode == "edit"
                else _GENERATE_SYSTEM_PROMPT
            ),
            user_prompt=normalized_user_prompt,
            source_image_data_url=source_image_data_url,
            annotated_reference_image_data_url=(
                normalized_annotated_reference if annotated_reference_used else None
            ),
            scene_reference_image_data_url=(
                normalized_scene_reference if scene_reference_used else None
            ),
            layout_reference_image_data_url=(
                normalized_layout_reference if layout_reference_used else None
            ),
            reference_image_data_urls=reference_images,
        )
    except Exception as exc:
        log_image_flow_event(
            "snapshot_image.render_error",
            {
                "trace_id": image_flow_trace_id,
                "render_mode": render_mode,
                "model_name": resolved_config.image_model_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise

    log_image_flow_event(
        "snapshot_image.render_output",
        {
            "trace_id": image_flow_trace_id,
            "render_mode": render_mode,
            "model_name": resolved_config.image_model_name,
            "image": redact_for_image_log(image_result),
        },
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
            "layout_reference_enabled": layout_reference_enabled,
            "layout_reference_used": layout_reference_used,
            "layout_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_layout_reference)
                if layout_reference_used and normalized_layout_reference is not None
                else None
            ),
            "annotated_reference_used": annotated_reference_used,
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
    annotated_reference_image_data_url: str | None = None,
    scene_reference_image_data_url: str | None = None,
    scene_reference_mode: SceneReferenceMode = "none",
    render_mode: RenderMode = "generate",
    preset_selection: Mapping[str, object] | None = None,
    edit_operations: list[SnapshotEditOperation] | None = None,
    edit_source_image_data_url: str | None = None,
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
        user_prompt=user_prompt,
        layout_reference_image_data_url=layout_reference_image_data_url,
        annotated_reference_image_data_url=annotated_reference_image_data_url,
        scene_reference_image_data_url=scene_reference_image_data_url,
        scene_reference_mode=scene_reference_mode,
        render_mode=render_mode,
        preset_selection=preset_selection,
        edit_operations=edit_operations,
        edit_source_image_data_url=edit_source_image_data_url,
        config=config,
        trace_id=trace_id,
    )


def _generate_image_with_gemini(
    *,
    trace_id: str,
    model_name: str,
    image_size: str,
    max_output_tokens: int,
    gemini_base_url: str,
    gemini_api_key: str | None,
    aspect_ratio: str,
    system_prompt: str,
    user_prompt: str,
    source_image_data_url: str,
    annotated_reference_image_data_url: str | None,
    scene_reference_image_data_url: str | None,
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
    attempts = _build_gemini_request_attempts(
        model_name=model_name,
        image_size=image_size,
        max_output_tokens=max_output_tokens,
        aspect_ratio=aspect_ratio,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_image_data_url=source_image_data_url,
        annotated_reference_image_data_url=annotated_reference_image_data_url,
        scene_reference_image_data_url=scene_reference_image_data_url,
        layout_reference_image_data_url=layout_reference_image_data_url,
        reference_image_data_urls=reference_image_data_urls,
    )
    last_response: Mapping[str, object] | None = None
    with httpx.Client(timeout=600.0) as client:
        for attempt in attempts:
            parsed = _post_gemini_generate_content(
                client=client,
                trace_id=trace_id,
                model_name=model_name,
                attempt_name=attempt.name,
                endpoint=endpoint,
                headers=headers,
                payload=attempt.payload,
            )
            last_response = parsed
            image_bytes = _extract_image_bytes(parsed)
            if image_bytes is None:
                log_image_flow_event(
                    "gemini.generate_content.no_image",
                    {
                        "trace_id": trace_id,
                        "model_name": model_name,
                        "attempt": attempt.name,
                        "finish_reason": _extract_gemini_finish_reason(parsed),
                        "finish_message": _extract_gemini_finish_message(parsed),
                        "response_text": _extract_gemini_response_text(parsed),
                        "response": summarize_gemini_response(parsed),
                    },
                )
                if _should_retry_gemini_with_minimal_payload(
                    model_name=model_name,
                    attempt=attempt,
                    response_payload=parsed,
                ):
                    logger.warning(
                        "Gemini image response for %s returned no image with "
                        "finishReason=%s. Retrying with compatibility payload.",
                        model_name,
                        _extract_gemini_finish_reason(parsed) or "unknown",
                    )
                    continue
                raise ValueError(_build_gemini_missing_image_error(parsed))
            mime_type = _guess_image_mime_type(image_bytes)
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            log_image_flow_event(
                "gemini.generate_content.image_output",
                {
                    "trace_id": trace_id,
                    "model_name": model_name,
                    "attempt": attempt.name,
                    "mime_type": mime_type,
                    "byte_length": len(image_bytes),
                    "generation_config_applied": attempt.generation_config_applied,
                    "response": summarize_gemini_response(parsed),
                },
            )
            return {
                "mime_type": mime_type,
                "image_base64": image_base64,
                "data_url": f"data:{mime_type};base64,{image_base64}",
                "size": image_size,
                "aspect_ratio": aspect_ratio,
                "request_strategy": attempt.name,
                "generation_config_applied": attempt.generation_config_applied,
                "raw_response": parsed,
            }

    raise ValueError(
        _build_gemini_missing_image_error(last_response or {"error": "No response"})
    )


def _build_gemini_request_attempts(
    *,
    model_name: str,
    image_size: str,
    max_output_tokens: int,
    aspect_ratio: str,
    system_prompt: str,
    user_prompt: str,
    source_image_data_url: str,
    annotated_reference_image_data_url: str | None,
    scene_reference_image_data_url: str | None,
    layout_reference_image_data_url: str | None,
    reference_image_data_urls: list[str],
) -> list[_GeminiRequestAttempt]:
    resolved_user_prompt = user_prompt
    include_system_instruction = not _model_inlines_system_prompt(model_name)
    if not include_system_instruction:
        resolved_user_prompt = system_prompt + "\n\n" + user_prompt

    content: list[dict[str, object]] = [
        {"text": resolved_user_prompt},
        _data_url_to_inline_part(source_image_data_url),
    ]
    if scene_reference_image_data_url is not None:
        content.append(_data_url_to_inline_part(scene_reference_image_data_url))
    if annotated_reference_image_data_url is not None:
        content.append(_data_url_to_inline_part(annotated_reference_image_data_url))
    if layout_reference_image_data_url is not None:
        content.append(_data_url_to_inline_part(layout_reference_image_data_url))
    content.extend(
        _data_url_to_inline_part(data_url) for data_url in reference_image_data_urls
    )

    base_payload: dict[str, object] = {
        "contents": [
            {
                "role": "user",
                "parts": content,
            }
        ],
    }
    if include_system_instruction:
        base_payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }

    configured_payload = {
        **base_payload,
        "generationConfig": _build_gemini_generation_config(
            model_name=model_name,
            image_size=image_size,
            max_output_tokens=max_output_tokens,
            aspect_ratio=aspect_ratio,
        ),
    }
    attempts = [
        _GeminiRequestAttempt(
            name="configured",
            payload=configured_payload,
            generation_config_applied=True,
        )
    ]
    if _model_uses_compatibility_image_generation_config(model_name):
        attempts.append(
            _GeminiRequestAttempt(
                name="compatibility_no_generation_config",
                payload=base_payload,
                generation_config_applied=False,
            )
        )
    return attempts


def _build_gemini_generation_config(
    *,
    model_name: str,
    image_size: str,
    max_output_tokens: int,
    aspect_ratio: str,
) -> dict[str, object]:
    image_config = {
        "aspectRatio": aspect_ratio,
        "imageSize": image_size,
    }
    if _model_uses_nano_banana_2_generation_config(model_name):
        return {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": image_config,
        }
    if _model_uses_compatibility_image_generation_config(model_name):
        return {
            "responseModalities": ["IMAGE"],
            "imageConfig": image_config,
        }
    return {
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": image_config,
        "maxOutputTokens": max_output_tokens,
    }


def _model_inlines_system_prompt(model_name: str) -> bool:
    normalized = _normalize_gemini_model_name(model_name)
    return normalized == "gemini-3.1-flash-image-preview"


def _model_uses_nano_banana_2_generation_config(model_name: str) -> bool:
    normalized = _normalize_gemini_model_name(model_name)
    return normalized == "gemini-3.1-flash-image-preview"


def _model_uses_compatibility_image_generation_config(model_name: str) -> bool:
    normalized = _normalize_gemini_model_name(model_name)
    if "-image" not in normalized:
        return False
    if _model_uses_nano_banana_2_generation_config(normalized):
        return False
    return normalized.startswith("gemini-3")


def _post_gemini_generate_content(
    *,
    client: httpx.Client,
    trace_id: str,
    model_name: str,
    attempt_name: str,
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
) -> dict[str, object]:
    payload_size_bytes = _payload_json_size_bytes(payload)
    inline_image_summaries = _collect_inline_image_summaries(payload)
    log_image_flow_event(
        "gemini.generate_content.attempt_input",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "attempt": attempt_name,
            "endpoint": endpoint,
            "payload_size_bytes": payload_size_bytes,
            "inline_image_count": len(inline_image_summaries),
            "inline_images": inline_image_summaries,
            "payload": summarize_gemini_payload(dict(payload)),
        },
    )
    if payload_size_bytes >= _GEMINI_INLINE_REQUEST_LIMIT_BYTES:
        log_image_flow_event(
            "gemini.generate_content.payload_too_large",
            {
                "trace_id": trace_id,
                "model_name": model_name,
                "attempt": attempt_name,
                "payload_size_bytes": payload_size_bytes,
                "limit_bytes": _GEMINI_INLINE_REQUEST_LIMIT_BYTES,
                "inline_image_count": len(inline_image_summaries),
                "inline_images": inline_image_summaries,
            },
        )
        payload_size_mb = payload_size_bytes / _BYTES_PER_MB
        limit_mb = _GEMINI_INLINE_REQUEST_LIMIT_BYTES / _BYTES_PER_MB
        raise ValueError(
            "Gemini inline image request is "
            f"{payload_size_mb:.2f} MB, exceeding the {limit_mb:.0f} MB inline "
            "request limit. Reduce canvas/reference image size or switch this "
            "flow to the Gemini Files API for large images."
        )
    started_at = perf_counter()
    response = client.post(endpoint, json=payload, headers=dict(headers))
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        log_image_flow_event(
            "gemini.generate_content.http_error",
            {
                "trace_id": trace_id,
                "model_name": model_name,
                "attempt": attempt_name,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "response_body": _safe_response_body(response),
            },
        )
        detail = f"{exc}. Gemini response body: {body}" if body else str(exc)
        raise RuntimeError(detail) from exc

    parsed = response.json()
    if not isinstance(parsed, dict):
        raise ValueError("Gemini image response must be a JSON object.")
    log_image_flow_event(
        "gemini.generate_content.http_output",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "attempt": attempt_name,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "response": summarize_gemini_response(parsed),
        },
    )
    return parsed


def _payload_json_size_bytes(payload: Mapping[str, object]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return len(encoded)


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


def _safe_response_body(response: httpx.Response) -> object:
    try:
        return summarize_gemini_response(response.json())
    except ValueError:
        text = response.text.strip()
        return redact_for_image_log(text)


def _should_retry_gemini_with_minimal_payload(
    *,
    model_name: str,
    attempt: _GeminiRequestAttempt,
    response_payload: Mapping[str, object],
) -> bool:
    if not _model_uses_compatibility_image_generation_config(model_name):
        return False
    if attempt.name != "configured":
        return False
    _ = response_payload
    return True


def _build_gemini_missing_image_error(payload: Mapping[str, object]) -> str:
    detail_parts: list[str] = []
    finish_reason = _extract_gemini_finish_reason(payload)
    if finish_reason:
        detail_parts.append(f"finishReason={finish_reason}")
    finish_message = _extract_gemini_finish_message(payload)
    if finish_message:
        detail_parts.append(f"finishMessage={_truncate_text(finish_message)}")
    response_text = _extract_gemini_response_text(payload)
    if response_text:
        detail_parts.append(f"text={_truncate_text(response_text)}")
    detail = f" Details: {'; '.join(detail_parts)}" if detail_parts else ""
    return "Gemini image response did not contain image data." + detail


def _extract_gemini_finish_reason(payload: Mapping[str, object]) -> str | None:
    candidate = _extract_first_gemini_candidate(payload)
    if candidate is None:
        return None
    value = candidate.get("finishReason")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_gemini_finish_message(payload: Mapping[str, object]) -> str | None:
    candidate = _extract_first_gemini_candidate(payload)
    if candidate is None:
        return None
    value = candidate.get("finishMessage")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_gemini_response_text(payload: Mapping[str, object]) -> str:
    candidate = _extract_first_gemini_candidate(payload)
    if candidate is None:
        return ""
    content = candidate.get("content")
    if not isinstance(content, Mapping):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""

    texts: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def _extract_first_gemini_candidate(
    payload: Mapping[str, object],
) -> Mapping[str, object] | None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if isinstance(first, Mapping):
        return first
    return None


def _truncate_text(value: str, *, limit: int = 240) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


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


def _build_final_user_prompt(
    *,
    user_prompt: str | None,
    render_mode: RenderMode,
    preset_prompt: str,
    edit_operations: list[SnapshotEditOperation],
    visible_objects: Mapping[str, Mapping[str, object]],
    layout_lock_prompt: str = "",
    layout_reference_used: bool = False,
) -> str:
    raw_user_prompt = _string(user_prompt)
    if render_mode == "generate":
        prompt_parts: list[str] = []
        if layout_lock_prompt:
            prompt_parts.append(layout_lock_prompt)
        prompt_parts.append(
            "Style goal, secondary to the exact input view: "
            + (raw_user_prompt or _DEFAULT_USER_PROMPT)
        )
        appearance_memory_prompt = _build_object_appearance_memory_prompt(
            edit_operations
        )
        if appearance_memory_prompt:
            prompt_parts.append(appearance_memory_prompt)
        if preset_prompt:
            prompt_parts.append("Selected presets:\n" + preset_prompt)
        return "\n\n".join(prompt_parts)

    if (
        not raw_user_prompt
        and not preset_prompt
        and not edit_operations
        and not layout_reference_used
    ):
        raise ValueError(
            "Provide at least one edit operation, preset change, prompt change, "
            "or target-view reference."
        )

    prompt_parts = [
        "Edit the first image.",
    ]
    if layout_reference_used:
        prompt_parts.append(
            "Use the second image as the labeled 3D target-view reference. Match "
            "that reference's camera angle, framing, room layout, openings, object "
            "placement, object size, and occlusion. Do not render any label, guide, "
            "box, outline, or UI mark from the reference."
        )
    else:
        prompt_parts.append(
            "Preserve the current camera, room layout, and all unselected objects."
        )
    if visible_objects:
        visible_summary = [
            f"{object_id} ({_visible_object_name(payload)})"
            for object_id, payload in visible_objects.items()
        ]
        prompt_parts.append(
            "Objects visible in this source image: " + ", ".join(visible_summary)
        )
    target_lock_prompt = _build_edit_target_geometry_prompt(
        edit_operations=edit_operations,
        visible_objects=visible_objects,
    )
    if target_lock_prompt:
        prompt_parts.append(target_lock_prompt)
    if raw_user_prompt:
        prompt_parts.append("Requested edit: " + raw_user_prompt)
    if preset_prompt:
        prompt_parts.append("Preset update:\n" + preset_prompt)

    replacement_index = 1
    for operation in edit_operations:
        if operation.replacement_image_data_url is not None:
            prompt_parts.append(
                "Selected object id: "
                f"{operation.object_id}. Replacement reference {replacement_index} "
                "belongs to this object only.\n"
                f"Use replacement reference {replacement_index} for appearance only. "
                "Ignore its background, crop, room, lighting direction, and camera. "
                f"Replace only the selected {operation.object_name} and fit it into "
                "the selected object's place in the target view. Do not change any "
                "unselected object."
            )
            replacement_index += 1
        if operation.target_color is not None:
            prompt_parts.append(
                f"Selected object id: {operation.object_id}.\n"
                f"Change only the color of the selected {operation.object_name} to {operation.target_color}. "
                "Keep its shape and material character. Do not change any "
                "unselected object."
            )

    return "\n\n".join(prompt_parts)


def _build_edit_target_geometry_prompt(
    *,
    edit_operations: list[SnapshotEditOperation],
    visible_objects: Mapping[str, Mapping[str, object]],
) -> str:
    prompt_parts: list[str] = []
    for operation in edit_operations:
        visible_payload = visible_objects.get(operation.object_id)
        if visible_payload is None:
            continue
        object_name = operation.object_name or _visible_object_name(visible_payload)
        prompt_parts.append(f"{operation.object_id} ({object_name})")

    if not prompt_parts:
        return ""
    return (
        "Selected object ids. Only these objects may change:\n"
        + "\n".join(prompt_parts)
    )


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
            "Each additional object reference is paired with one object id below. "
            "Use it only for that object's appearance. The labeled 3D layout image "
            "still controls the camera, placement, size, and visibility."
        ),
    ]
    for index, operation in enumerate(paired_operations, start=1):
        object_name = operation.object_name or "object"
        prompt_parts.append(
            f"Reference image {index}: object id {operation.object_id} ({object_name})."
        )
    return "\n".join(prompt_parts)


def _resolve_scene_reference_mode(
    *,
    scene_reference_mode: SceneReferenceMode,
    render_mode: RenderMode,
    scene_reference_image_data_url: str | None,
) -> SceneReferenceMode:
    if render_mode != "generate" or scene_reference_image_data_url is None:
        return "none"
    if scene_reference_mode == "scene_reference_camera_transfer":
        return "scene_reference_camera_transfer"
    if scene_reference_mode == "target_layout_with_scene_reference":
        return "target_layout_with_scene_reference"
    return "none"


def _build_generate_layout_lock_prompt(
    *,
    scene_reference_mode: SceneReferenceMode,
) -> str:
    prompt_parts = [
        "Use the input labeled 3D layout image as the exact target view.",
        (
            "Match the camera angle, framing, perspective, room boundaries, openings, "
            "furniture placement, object count, object size, depth order, and "
            "occlusion shown in that image."
        ),
        (
            "The labels and guide marks are only for understanding the layout. "
            "They must not appear in the final render."
        ),
    ]
    if scene_reference_mode == "target_layout_with_scene_reference":
        prompt_parts.append(
            "If a previous render reference is present, use it only for style and "
            "appearance continuity. The labeled 3D image still controls the final "
            "camera and layout."
        )
    elif scene_reference_mode == "scene_reference_camera_transfer":
        prompt_parts.append(
            "For camera transfer, keep the previous render's style and object "
            "identity, but make the final view match the labeled 3D target view."
        )
    prompt_parts.append(
        "Do not add, remove, move, resize, rotate, or replace any visible major "
        "object unless it has an explicit object reference. Improve only the "
        "photorealistic appearance."
    )
    return "\n".join(prompt_parts)


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
