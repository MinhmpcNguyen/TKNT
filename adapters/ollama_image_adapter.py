from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import httpx

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

DEFAULT_OLLAMA_IMAGE_MODEL = "x/flux2-klein:latest"
DEFAULT_OLLAMA_IMAGE_SIZE = "1024x1024"
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_NEGATIVE_PROMPT = (
    "room, environment, floor plane, table, stand, pedestal, shelf, multiple objects, "
    "clutter, busy scene, colored background, gradient background, dark background, "
    "patterned background, text, watermark, logo, border, cropped object, cut off, "
    "partial object, person, hand, reflection, tiny object, distant object, far away, "
    "zoomed out, wide shot, excessive white margin, empty frame"
)


@dataclass(frozen=True)
class OllamaImageAdapterConfig:
    endpoint: str
    model_name: str = DEFAULT_OLLAMA_IMAGE_MODEL
    output_size: str = DEFAULT_OLLAMA_IMAGE_SIZE
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass(frozen=True)
class OllamaGeneratedImage:
    image_path: Path
    request_path: Path
    response_path: Path
    prompt_path: Path
    negative_prompt_path: Path


def generate_white_background_image(
    *,
    prompt: str,
    output_dir: Path,
    config: OllamaImageAdapterConfig,
    negative_prompt: str | None = None,
) -> OllamaGeneratedImage:
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = output_dir / "ollama_prompt.txt"
    negative_prompt_path = output_dir / "ollama_negative_prompt.txt"
    request_path = output_dir / "ollama_request.json"
    response_path = output_dir / "ollama_response.json"

    final_prompt = _build_white_background_prompt(prompt)
    final_negative_prompt = _build_negative_prompt(negative_prompt)
    request_payload: dict[str, JSONValue] = {
        "model": config.model_name,
        "prompt": final_prompt,
        "negative_prompt": final_negative_prompt,
        "size": config.output_size,
        "response_format": "b64_json",
    }

    prompt_path.write_text(final_prompt, encoding="utf-8")
    negative_prompt_path.write_text(final_negative_prompt, encoding="utf-8")
    request_path.write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    endpoint = config.endpoint.rstrip("/") + "/v1/images/generations"
    with httpx.Client(timeout=config.timeout_s) as client:
        response = client.post(endpoint, json=request_payload)

    response_payload = _parse_response_payload(response)
    response_path.write_text(
        json.dumps(response_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if response.status_code >= 400:
        detail = json.dumps(response_payload, ensure_ascii=False)[:400]
        raise RuntimeError(
            f"Ollama image generation failed with HTTP {response.status_code}. "
            f"See {response_path}. Response excerpt: {detail}"
        )

    image_bytes = _extract_image_bytes(response_payload)
    if image_bytes is None:
        raise RuntimeError(
            "Ollama did not return image bytes. "
            f"See {response_path} for the raw response."
        )

    image_path = _write_image(output_dir / "generated_image", image_bytes)
    return OllamaGeneratedImage(
        image_path=image_path,
        request_path=request_path,
        response_path=response_path,
        prompt_path=prompt_path,
        negative_prompt_path=negative_prompt_path,
    )


def normalize_ollama_endpoint(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized or None


def _build_white_background_prompt(prompt: str) -> str:
    return (
        "Create exactly one isolated object that matches the description below.\n"
        "Treat it as a clean studio product image prepared for single-image 3D reconstruction.\n"
        "The object must be centered, fully visible, and occupy most of the frame.\n"
        "Use a close camera framing so the object fills roughly 75% to 85% of the canvas.\n"
        "Leave only a small white margin around the object on all sides.\n"
        "Do not make the subject tiny, distant, zoomed out, or surrounded by large empty space.\n"
        "Use a pure solid white background only.\n"
        "Do not add any environment, room context, floor plane, props, hands, people, text, or watermark.\n"
        "Keep the silhouette clean and the object readable for 3D reconstruction.\n\n"
        f"Object description:\n{prompt.strip()}"
    )


def _build_negative_prompt(negative_prompt: str | None) -> str:
    if negative_prompt is None or not negative_prompt.strip():
        return DEFAULT_NEGATIVE_PROMPT
    return f"{DEFAULT_NEGATIVE_PROMPT}, {negative_prompt.strip()}"


def _parse_response_payload(response: httpx.Response) -> dict[str, JSONValue]:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama returned a non-JSON response with content-type "
            f"{response.headers.get('content-type', '<unknown>')}."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Ollama response must be a JSON object.")
    return payload


def _extract_image_bytes(payload: JSONValue) -> bytes | None:
    if isinstance(payload, dict):
        decoded = _decode_known_image_fields(payload)
        if decoded is not None:
            return decoded
        for value in payload.values():
            nested = _extract_image_bytes(value)
            if nested is not None:
                return nested
        return None

    if isinstance(payload, list):
        for item in payload:
            nested = _extract_image_bytes(item)
            if nested is not None:
                return nested
        return None

    return None


def _decode_known_image_fields(payload: dict[str, JSONValue]) -> bytes | None:
    for key in ("b64_json", "image"):
        candidate = payload.get(key)
        if isinstance(candidate, str):
            decoded = _decode_base64_image(candidate)
            if decoded is not None:
                return decoded

    images = payload.get("images")
    if isinstance(images, list):
        for item in images:
            if isinstance(item, str):
                decoded = _decode_base64_image(item)
                if decoded is not None:
                    return decoded
            elif isinstance(item, dict):
                decoded = _decode_known_image_fields(item)
                if decoded is not None:
                    return decoded

    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                decoded = _decode_known_image_fields(item)
                if decoded is not None:
                    return decoded

    return None


def _decode_base64_image(value: str) -> bytes | None:
    normalized = value.strip()
    if normalized.startswith("data:image/") and "," in normalized:
        normalized = normalized.split(",", maxsplit=1)[1]
    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError):
        return None


def _write_image(output_base: Path, image_bytes: bytes) -> Path:
    suffix = _guess_image_suffix(image_bytes)
    image_path = output_base.with_suffix(suffix)
    image_path.write_bytes(image_bytes)
    return image_path


def _guess_image_suffix(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    return ".bin"
