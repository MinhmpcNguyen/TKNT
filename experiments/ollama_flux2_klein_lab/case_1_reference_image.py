from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from common import DEFAULT_ENDPOINT, DEFAULT_MODEL, DEFAULT_SIZE
from common import encode_image_to_base64, extract_image_bytes, post_json
from common import post_multipart, resolve_prompt_text, write_image
from common import write_response_snapshot

LOGGER = logging.getLogger("flux2_klein_case_1")

DEFAULT_PROMPT = """
Use the reference image as the main composition guide.
Preserve the overall layout, major object relationships, and viewing angle.
Improve the scene into a polished interior render with realistic materials,
soft natural daylight, and coherent shadows.
""".strip()


@dataclass(frozen=True)
class AttemptSummary:
    name: str
    route: str
    status_code: int
    snapshot_path: str
    image_path: str | None
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether FLUX.2 Klein accepts reference image + prompt locally.",
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to a reference image.")
    parser.add_argument("--prompt", type=str, help="Prompt text.")
    parser.add_argument("--prompt-file", type=Path, help="Optional prompt file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=DEFAULT_SIZE,
        help="Requested output size for OpenAI-style image editing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "case_1_reference_result.png",
        help="Preferred output path. The suffix may be adjusted to match the actual image format.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    prompt_text = resolve_prompt_text(args.prompt, args.prompt_file, DEFAULT_PROMPT)

    if not args.image.exists():
        raise FileNotFoundError(f"Reference image not found: {args.image}")

    attempts = [
        probe_openai_edits(
            endpoint=args.endpoint,
            model=args.model,
            image_path=args.image,
            prompt_text=prompt_text,
            size=args.size,
            output_path=args.output,
        ),
        probe_api_generate(
            endpoint=args.endpoint,
            model=args.model,
            image_path=args.image,
            prompt_text=prompt_text,
            output_path=args.output,
        ),
    ]

    report_path = args.output.parent / f"{args.output.stem}.probe_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(attempt) for attempt in attempts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for attempt in attempts:
        if attempt.image_path is not None:
            LOGGER.info("Success via %s", attempt.name)
            LOGGER.info("Generated image: %s", attempt.image_path)
            LOGGER.info("Probe report: %s", report_path)
            return 0

    LOGGER.error("No probe route returned an image.")
    LOGGER.error("Probe report: %s", report_path)
    for attempt in attempts:
        LOGGER.error(
            "%s -> status=%s, raw=%s, note=%s",
            attempt.name,
            attempt.status_code,
            attempt.snapshot_path,
            attempt.note,
        )
    return 1


def probe_openai_edits(
    *,
    endpoint: str,
    model: str,
    image_path: Path,
    prompt_text: str,
    size: str,
    output_path: Path,
    image_field_name: str = "image",
    mask_path: Path | None = None,
) -> AttemptSummary:
    files = {image_field_name: image_path}
    if mask_path is not None:
        files["mask"] = mask_path

    response = post_multipart(
        endpoint=endpoint,
        route="/v1/images/edits",
        fields={
            "model": model,
            "prompt": prompt_text,
            "size": size,
            "response_format": "b64_json",
        },
        files=files,
    )
    snapshot_path = write_response_snapshot(
        output_path.parent
        / f"{output_path.stem}.v1_images_edits_{sanitize_field_name(image_field_name)}_response",
        response,
    )
    image_bytes = extract_image_bytes(response)
    if image_bytes is not None:
        actual_output = write_image(
            output_path.parent
            / f"{output_path.stem}.v1_images_edits_{sanitize_field_name(image_field_name)}.png",
            image_bytes,
        )
        return AttemptSummary(
            name=f"openai_images_edits[{image_field_name}]",
            route="/v1/images/edits",
            status_code=response.status_code,
            snapshot_path=str(snapshot_path),
            image_path=str(actual_output),
            note="Received image data.",
        )

    note = "No image bytes found in response."
    if response.status_code >= 400:
        note = f"HTTP {response.status_code}: {response.text[:240]}"
    return AttemptSummary(
        name=f"openai_images_edits[{image_field_name}]",
        route="/v1/images/edits",
        status_code=response.status_code,
        snapshot_path=str(snapshot_path),
        image_path=None,
        note=note,
    )


def probe_api_generate(
    *,
    endpoint: str,
    model: str,
    image_path: Path,
    prompt_text: str,
    output_path: Path,
) -> AttemptSummary:
    response = post_json(
        endpoint=endpoint,
        route="/api/generate",
        payload={
            "model": model,
            "prompt": prompt_text,
            "images": [encode_image_to_base64(image_path)],
            "stream": False,
        },
    )
    snapshot_path = write_response_snapshot(
        output_path.parent / f"{output_path.stem}.api_generate_response",
        response,
    )
    image_bytes = extract_image_bytes(response)
    if image_bytes is not None:
        actual_output = write_image(
            output_path.parent / f"{output_path.stem}.api_generate.png",
            image_bytes,
        )
        return AttemptSummary(
            name="api_generate_with_images",
            route="/api/generate",
            status_code=response.status_code,
            snapshot_path=str(snapshot_path),
            image_path=str(actual_output),
            note="Received image data.",
        )

    note = "No image bytes found in response."
    if response.status_code >= 400:
        note = f"HTTP {response.status_code}: {response.text[:240]}"
    return AttemptSummary(
        name="api_generate_with_images",
        route="/api/generate",
        status_code=response.status_code,
        snapshot_path=str(snapshot_path),
        image_path=None,
        note=note,
    )


def sanitize_field_name(field_name: str) -> str:
    return field_name.replace("[", "_").replace("]", "")


if __name__ == "__main__":
    sys.exit(main())
