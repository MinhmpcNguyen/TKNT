from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "google/gemini-2.5-flash-image"
DEFAULT_PROMPT = "Create a cinematic poster of a cat astronaut on Mars."
DEFAULT_OUTPUT_PATH = Path("outputs/openrouter_gemini_image.png")
DEFAULT_MAX_TOKENS = 256
SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}


class ImageUrlPayload(TypedDict):
    url: str


class TextContentPart(TypedDict):
    type: str
    text: str


class ImageContentPart(TypedDict):
    type: str
    image_url: ImageUrlPayload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test OpenRouter image generation or editing with Gemini 2.5 Flash Image."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Image generation prompt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        help="Optional source image to edit.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="1:1",
        help="Requested aspect ratio, for example 1:1 or 16:9.",
    )
    parser.add_argument(
        "--image-size",
        default="1K",
        help="Requested image size, for example 1K, 2K, or 4K.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum text tokens to allow in the response.",
    )
    parser.add_argument(
        "--site-url",
        default=os.getenv("OPENROUTER_SITE_URL"),
        help="Optional HTTP-Referer header.",
    )
    parser.add_argument(
        "--site-name",
        default=os.getenv("OPENROUTER_SITE_NAME"),
        help="Optional X-OpenRouter-Title header.",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY")
        return 1

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    extra_headers = build_headers(
        site_url=args.site_url,
        site_name=args.site_name,
    )

    try:
        used_model, completion = request_completion(
            client=client,
            requested_model=args.model,
            prompt=args.prompt,
            input_image=args.input_image,
            extra_headers=extra_headers,
            aspect_ratio=args.aspect_ratio,
            image_size=args.image_size,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        print(f"Request failed: {exc}")
        return 1

    try:
        content_text, data_url = extract_response_payload(completion)
    except ValueError as exc:
        print(str(exc))
        return 1

    save_data_url_image(data_url=data_url, output_path=args.output)

    if used_model != args.model:
        print(f"Requested model: {args.model}")
    print(f"Model: {used_model}")
    print(f"Prompt: {args.prompt}")
    if args.input_image is not None:
        print(f"Input image: {args.input_image}")
    if content_text:
        print("Text:")
        print(content_text)
    print(f"Saved image to: {args.output}")
    return 0


def build_headers(site_url: str | None, site_name: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-OpenRouter-Title"] = site_name
    return headers


def request_completion(
    *,
    client: OpenAI,
    requested_model: str,
    prompt: str,
    input_image: Path | None,
    extra_headers: dict[str, str],
    aspect_ratio: str,
    image_size: str,
    max_tokens: int,
) -> tuple[str, object]:
    last_error: Exception | None = None
    model_candidates = resolve_model_candidates(requested_model)

    for index, model_name in enumerate(model_candidates):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": build_message_content(
                            prompt=prompt,
                            input_image=input_image,
                        ),
                    }
                ],
                extra_headers=extra_headers,
                extra_body={
                    "modalities": ["image", "text"],
                    "image_config": {
                        "aspect_ratio": aspect_ratio,
                        "image_size": image_size,
                    },
                    "max_tokens": max_tokens,
                },
            )
            return model_name, completion
        except Exception as exc:
            last_error = exc
            is_last_candidate = index == len(model_candidates) - 1
            if is_last_candidate or not is_missing_endpoint_error(exc):
                break

    if last_error is None:
        raise ValueError("OpenRouter request failed before a model was attempted.")
    raise last_error


def resolve_model_candidates(requested_model: str) -> list[str]:
    candidates = [requested_model]
    preview_suffix = "-preview"
    if requested_model.endswith(preview_suffix):
        canonical_model = requested_model.removesuffix(preview_suffix)
        if canonical_model not in candidates:
            candidates.append(canonical_model)
    return candidates


def is_missing_endpoint_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    return "No endpoints found" in str(exc)


def build_message_content(
    prompt: str,
    input_image: Path | None,
) -> str | list[TextContentPart | ImageContentPart]:
    if input_image is None:
        return prompt

    data_url = encode_image_as_data_url(input_image)
    return [
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": data_url,
            },
        },
    ]


def encode_image_as_data_url(image_path: Path) -> str:
    if not image_path.is_file():
        raise ValueError(f"Input image does not exist: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Unsupported input image type. Use PNG, JPEG, WEBP, or GIF.")

    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_response_payload(completion: object) -> tuple[str, str]:
    choices = getattr(completion, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter returned no choices.")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError("OpenRouter returned no message.")

    content = getattr(message, "content", None)
    text = content if isinstance(content, str) else ""

    model_extra = getattr(message, "model_extra", None)
    if not isinstance(model_extra, dict):
        raise ValueError("OpenRouter response did not include image metadata.")

    images = model_extra.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("OpenRouter response did not include generated images.")

    first_image = images[0]
    if not isinstance(first_image, dict):
        raise ValueError("Generated image payload has an unexpected shape.")

    image_url = first_image.get("image_url")
    if not isinstance(image_url, dict):
        raise ValueError("Generated image payload is missing image_url.")

    data_url = image_url.get("url")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise ValueError("Generated image URL is missing or not a data URL.")

    return text, data_url


def save_data_url_image(data_url: str, output_path: Path) -> None:
    try:
        _, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("Could not parse returned image data URL.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(encoded)
    output_path.write_bytes(image_bytes)


if __name__ == "__main__":
    raise SystemExit(main())
