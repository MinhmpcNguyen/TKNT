from __future__ import annotations

import argparse
import base64
import binascii
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_MODEL = "gemini-2.5-flash-image"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini image generation or image editing via generateContent.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Gemini image model name.",
    )
    parser.add_argument("--prompt", type=str, help="Prompt text.")
    parser.add_argument("--prompt-file", type=Path, help="Optional prompt file.")
    parser.add_argument("--image", type=Path, help="Optional input image for editing.")
    parser.add_argument("--output", type=Path, required=True, help="Output image path.")
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        help='Optional aspect ratio like "1:1", "16:9", "9:16".',
    )
    parser.add_argument(
        "--image-size",
        type=str,
        choices=("1K", "2K", "4K"),
        help="Optional Gemini image size preset.",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Request only image output instead of text plus image.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prompt_text = resolve_prompt_text(prompt=args.prompt, prompt_file=args.prompt_file)
    validate_optional_image(args.image)
    ensure_parent_dir(args.output)

    api_key, base_url = resolve_gemini_credentials()
    payload = build_payload(
        prompt_text=prompt_text,
        image_path=args.image,
        image_only=args.image_only,
        aspect_ratio=args.aspect_ratio,
        image_size=args.image_size,
    )

    response = post_json(
        url=build_model_url(base_url=base_url, model_name=args.model),
        api_key=api_key,
        payload=payload,
    )

    response_path = args.output.parent / f"{args.output.stem}.response.json"
    response_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    texts, image_paths = save_response_parts(response=response, output_path=args.output)

    if texts:
        text_path = args.output.parent / f"{args.output.stem}.txt"
        text_path.write_text("\n\n".join(texts).strip() + "\n", encoding="utf-8")
        print(f"Saved text: {text_path}")

    print(f"Saved raw response: {response_path}")
    for image_path in image_paths:
        print(f"Saved image: {image_path}")

    if not image_paths:
        print("No image part returned. Inspect the raw response JSON.")
        return 1
    return 0


def resolve_prompt_text(*, prompt: str | None, prompt_file: Path | None) -> str:
    if prompt is not None and prompt.strip():
        return prompt.strip()
    if prompt_file is not None:
        return prompt_file.read_text(encoding="utf-8").strip()
    raise ValueError("Provide either --prompt or --prompt-file.")


def validate_optional_image(path: Path | None) -> None:
    if path is not None and not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_gemini_credentials() -> tuple[str, str]:
    api_key = "AIzaSyBDyF0vlgztCNrYebc34hSjxT74WbRkGtE"
    base_url = os.getenv("GEMINI_BASE_URL", DEFAULT_BASE_URL)

    if api_key and base_url:
        return api_key, base_url.rstrip("/")

    try:
        from config.gemini_config import GeminiConfig
    except Exception:
        if api_key:
            return api_key, base_url.rstrip("/")
        raise ValueError("Unable to load GeminiConfig and GEMINI_API_KEY is not set.")

    resolved_api_key = GeminiConfig.API_KEY or api_key
    resolved_base_url = GeminiConfig.BASE_URL or base_url
    if not resolved_api_key:
        raise ValueError("Missing Gemini API key.")
    parsed = urlparse(resolved_base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Gemini base URL: {resolved_base_url}")
    return resolved_api_key, resolved_base_url.rstrip("/")


def build_model_url(*, base_url: str, model_name: str) -> str:
    return f"{base_url}/models/{model_name}:generateContent"


def build_payload(
    *,
    prompt_text: str,
    image_path: Path | None,
    image_only: bool,
    aspect_ratio: str | None,
    image_size: str | None,
) -> dict[str, object]:
    parts: list[dict[str, object]] = [{"text": prompt_text}]
    if image_path is not None:
        parts.append(
            {
                "inlineData": {
                    "mimeType": guess_mime_type(image_path),
                    "data": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
                }
            }
        )

    generation_config: dict[str, object] = {
        "responseModalities": ["IMAGE"] if image_only else ["TEXT", "IMAGE"]
    }
    image_config: dict[str, object] = {}
    if aspect_ratio is not None:
        image_config["aspectRatio"] = aspect_ratio
    if image_size is not None:
        image_config["imageSize"] = image_size
    if image_config:
        generation_config["imageConfig"] = image_config

    return {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
    }


def guess_mime_type(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0]
    return mime_type or "application/octet-stream"


def post_json(
    *, url: str, api_key: str, payload: dict[str, object]
) -> dict[str, object]:
    import httpx

    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            raise httpx.HTTPStatusError(
                f"{exc}. Gemini response body: {detail}" if detail else str(exc),
                request=exc.request,
                response=exc.response,
            ) from exc
        data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Gemini response must be a JSON object.")
    return data


def save_response_parts(
    *,
    response: dict[str, object],
    output_path: Path,
) -> tuple[list[str], list[Path]]:
    texts: list[str] = []
    image_paths: list[Path] = []

    for index, part in enumerate(iter_parts(response), start=1):
        text = extract_text_part(part)
        if text is not None:
            texts.append(text)
            continue

        inline_data = extract_inline_data(part)
        if inline_data is None:
            continue

        mime_type = inline_data.get("mime_type") or inline_data.get("mimeType")
        data = inline_data.get("data")
        if not isinstance(mime_type, str) or not isinstance(data, str):
            continue

        try:
            image_bytes = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            continue

        target_path = choose_image_output_path(
            output_path=output_path,
            image_index=index,
            mime_type=mime_type,
        )
        target_path.write_bytes(image_bytes)
        image_paths.append(target_path)

    return texts, image_paths


def iter_parts(response: dict[str, object]) -> list[dict[str, object]]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return []

    result: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict):
                result.append(part)
    return result


def extract_text_part(part: dict[str, object]) -> str | None:
    text = part.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def extract_inline_data(part: dict[str, object]) -> dict[str, object] | None:
    inline_data = part.get("inlineData")
    if isinstance(inline_data, dict):
        return inline_data
    inline_data = part.get("inline_data")
    if isinstance(inline_data, dict):
        return inline_data
    return None


def choose_image_output_path(
    *,
    output_path: Path,
    image_index: int,
    mime_type: str,
) -> Path:
    suffix = suffix_for_mime_type(mime_type)
    if image_index == 1:
        return output_path.with_suffix(suffix)
    return output_path.parent / f"{output_path.stem}_{image_index}{suffix}"


def suffix_for_mime_type(mime_type: str) -> str:
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    return mapping.get(mime_type.lower(), ".bin")


if __name__ == "__main__":
    raise SystemExit(main())
