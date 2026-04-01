from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common import DEFAULT_ENDPOINT, DEFAULT_MODEL, DEFAULT_SIZE
from common import JSONValue, extract_image_bytes, load_json_file
from common import post_json, pretty_json_text
from common import resolve_prompt_text, write_image, write_response_snapshot

LOGGER = logging.getLogger("flux2_klein_case_2")

DEFAULT_PROMPT = """
Create a clean top-down interior render from the provided layout JSON.
Preserve object count, relative scale, circulation, and adjacency.
Use warm neutral materials, realistic floor texture, soft daylight,
and a polished architectural-visualization look.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image from final layout JSON + prompt using FLUX.2 Klein.",
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        required=True,
        help="Path to the final layout JSON file.",
    )
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
        help="Requested output size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "case_2_layout_json_result.png",
        help="Preferred output path. The suffix may be adjusted to match the actual image format.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    prompt_text = resolve_prompt_text(args.prompt, args.prompt_file, DEFAULT_PROMPT)

    if not args.layout_json.exists():
        raise FileNotFoundError(f"Layout JSON not found: {args.layout_json}")

    layout_payload = load_json_file(args.layout_json)
    composed_prompt = build_layout_prompt(prompt_text, layout_payload)

    response = post_json(
        endpoint=args.endpoint,
        route="/v1/images/generations",
        payload={
            "model": args.model,
            "prompt": composed_prompt,
            "size": args.size,
            "response_format": "b64_json",
        },
    )
    snapshot_path = write_response_snapshot(
        args.output.parent / f"{args.output.stem}.response",
        response,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}. "
            f"See raw response at {snapshot_path}."
        )

    image_bytes = extract_image_bytes(response)
    if image_bytes is None:
        raise RuntimeError(
            "The response did not contain image data. "
            f"See raw response at {snapshot_path}."
        )

    actual_output = write_image(args.output, image_bytes)
    LOGGER.info("Generated image: %s", actual_output)
    LOGGER.info("Raw response: %s", snapshot_path)
    return 0


def build_layout_prompt(prompt_text: str, layout_payload: dict[str, JSONValue]) -> str:
    layout_text = pretty_json_text(layout_payload)
    return (
        f"{prompt_text}\n\n"
        "Use the following final layout JSON as ground truth.\n"
        "Keep the furniture count, approximate footprints, spacing, and layout logic consistent.\n"
        "Do not add large new elements that change the plan.\n\n"
        f"Layout JSON:\n{layout_text}"
    )


if __name__ == "__main__":
    sys.exit(main())
