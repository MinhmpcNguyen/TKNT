from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import uuid
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_MODEL = "x/flux2-klein:latest"
DEFAULT_SIZE = "1024x1024"
REQUEST_TIMEOUT_S = 600


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


def resolve_prompt_text(
    prompt: str | None,
    prompt_file: Path | None,
    default_prompt: str,
) -> str:
    if prompt is not None and prompt.strip():
        return prompt.strip()
    if prompt_file is not None:
        return prompt_file.read_text(encoding="utf-8").strip()
    return default_prompt.strip()


def load_json_file(path: Path) -> dict[str, JSONValue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")
    return payload


def pretty_json_text(payload: Mapping[str, JSONValue]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def encode_image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def post_json(
    endpoint: str,
    route: str,
    payload: Mapping[str, JSONValue],
) -> HttpResponse:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _build_url(endpoint, route),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send(request)


def post_multipart(
    endpoint: str,
    route: str,
    fields: Mapping[str, str],
    files: Mapping[str, Path],
) -> HttpResponse:
    boundary = f"----codex-{uuid.uuid4().hex}"
    body_parts: list[bytes] = []

    for key, value in fields.items():
        body_parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    for key, path in files.items():
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body_parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{key}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    request = urllib.request.Request(
        _build_url(endpoint, route),
        data=b"".join(body_parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return _send(request)


def parse_json_body(response: HttpResponse) -> JSONValue | None:
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return None


def extract_image_bytes(response: HttpResponse) -> bytes | None:
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("image/"):
        return response.body

    payload = parse_json_body(response)
    if payload is None:
        return None
    return _extract_image_bytes_from_json(payload)


def write_image(output_path: Path, image_bytes: bytes) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual_path = output_path.with_suffix(guess_image_suffix(image_bytes))
    actual_path.write_bytes(image_bytes)
    return actual_path


def guess_image_suffix(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    trimmed = image_bytes.lstrip()
    if trimmed.startswith((b"<svg", b"<?xml")):
        return ".svg"
    return ".bin"


def write_response_snapshot(base_path: Path, response: HttpResponse) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    payload = parse_json_body(response)
    if payload is not None:
        snapshot_path = base_path.parent / f"{base_path.name}.json"
        snapshot_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return snapshot_path

    snapshot_path = base_path.parent / f"{base_path.name}.txt"
    snapshot_path.write_text(response.text, encoding="utf-8")
    return snapshot_path


def _build_url(endpoint: str, route: str) -> str:
    return f"{endpoint.rstrip('/')}/{route.lstrip('/')}"


def _send(request: urllib.request.Request) -> HttpResponse:
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return HttpResponse(
                status_code=response.status,
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status_code=exc.code,
            headers=dict(exc.headers.items()),
            body=exc.read(),
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {request.full_url}. "
            "Make sure `ollama serve` is running."
        ) from exc


def _extract_image_bytes_from_json(payload: JSONValue) -> bytes | None:
    if isinstance(payload, dict):
        decoded = _decode_known_image_fields(payload)
        if decoded is not None:
            return decoded

        for value in payload.values():
            if isinstance(value, (dict, list)):
                nested = _extract_image_bytes_from_json(value)
                if nested is not None:
                    return nested
        return None

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                nested = _extract_image_bytes_from_json(item)
                if nested is not None:
                    return nested
        return None

    return None


def _decode_known_image_fields(payload: Mapping[str, JSONValue]) -> bytes | None:
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
        image_bytes = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError):
        return None

    if guess_image_suffix(image_bytes) == ".bin":
        return None
    return image_bytes
