from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from case_1_reference_image import AttemptSummary, probe_api_generate, probe_openai_edits
from common import DEFAULT_ENDPOINT, DEFAULT_MODEL, DEFAULT_SIZE, resolve_prompt_text

LOGGER = logging.getLogger("flux2_klein_case_1_exact")

DEFAULT_PROMPT_FILE = (
    Path(__file__).resolve().parent / "prompts" / "exact_preservation_prompt.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stricter exact-preservation reference-image test.",
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to a reference image.")
    parser.add_argument("--prompt", type=str, help="Optional extra prompt override.")
    parser.add_argument("--prompt-file", type=Path, help="Optional custom prompt file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--mode",
        choices=("api-generate", "edits-only", "edits-then-api-generate"),
        default="api-generate",
        help="Select which route strategy to run.",
    )
    parser.add_argument(
        "--edits-image-field",
        choices=("image", "image[]", "both"),
        default="both",
        help="Field name variant to use for /v1/images/edits.",
    )
    parser.add_argument("--mask", type=Path, help="Optional mask image for edits.")
    parser.add_argument(
        "--size",
        type=str,
        default=DEFAULT_SIZE,
        help="Requested output size for the experimental edits route.",
    )
    parser.add_argument(
        "--try-openai-edits",
        action="store_true",
        help="Also probe /v1/images/edits. Disabled by default because it often fails.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "outputs_exact"
            / "case_1_exact_preservation_result.png"
        ),
        help="Preferred output path. The suffix may be adjusted to match the actual image format.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Reference image not found: {args.image}")

    if args.mask is not None and not args.mask.exists():
        raise FileNotFoundError(f"Mask image not found: {args.mask}")

    prompt_text = resolve_prompt_text(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        default_prompt=DEFAULT_PROMPT_FILE.read_text(encoding="utf-8"),
    )
    prompt_path = write_prompt_snapshot(args.output, prompt_text)

    attempts = build_attempts(
        endpoint=args.endpoint,
        model=args.model,
        image_path=args.image,
        prompt_text=prompt_text,
        output_path=args.output,
        size=args.size,
        try_openai_edits=args.try_openai_edits,
        mode=args.mode,
        edits_image_field=args.edits_image_field,
        mask_path=args.mask,
    )

    report_path = args.output.parent / f"{args.output.stem}.report.json"
    report_payload = {
        "reference_image": str(args.image.resolve()),
        "used_prompt_path": str(prompt_path),
        "routes_tested": [attempt.name for attempt in attempts],
        "attempts": [asdict(attempt) for attempt in attempts],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for attempt in attempts:
        if attempt.image_path is not None:
            LOGGER.info("Success via %s", attempt.name)
            LOGGER.info("Generated image: %s", attempt.image_path)
            LOGGER.info("Used prompt: %s", prompt_path)
            LOGGER.info("Report: %s", report_path)
            return 0

    LOGGER.error("No route returned an image.")
    LOGGER.error("Used prompt: %s", prompt_path)
    LOGGER.error("Report: %s", report_path)
    for attempt in attempts:
        LOGGER.error(
            "%s -> status=%s, raw=%s, note=%s",
            attempt.name,
            attempt.status_code,
            attempt.snapshot_path,
            attempt.note,
        )
    return 1


def build_attempts(
    *,
    endpoint: str,
    model: str,
    image_path: Path,
    prompt_text: str,
    output_path: Path,
    size: str,
    try_openai_edits: bool,
    mode: str,
    edits_image_field: str,
    mask_path: Path | None,
) -> list[AttemptSummary]:
    attempts: list[AttemptSummary] = []
    resolved_mode = resolve_mode(mode=mode, try_openai_edits=try_openai_edits)

    if resolved_mode in ("edits-only", "edits-then-api-generate"):
        for image_field_name in resolve_edits_field_names(edits_image_field):
            attempts.append(
                probe_openai_edits(
                    endpoint=endpoint,
                    model=model,
                    image_path=image_path,
                    prompt_text=prompt_text,
                    size=size,
                    output_path=output_path,
                    image_field_name=image_field_name,
                    mask_path=mask_path,
                )
            )

    if resolved_mode in ("api-generate", "edits-then-api-generate"):
        attempts.append(
            probe_api_generate(
                endpoint=endpoint,
                model=model,
                image_path=image_path,
                prompt_text=prompt_text,
                output_path=output_path,
            )
        )
    return attempts


def write_prompt_snapshot(output_path: Path, prompt_text: str) -> Path:
    prompt_path = output_path.parent / f"{output_path.stem}.prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")
    return prompt_path


def resolve_mode(*, mode: str, try_openai_edits: bool) -> str:
    if try_openai_edits and mode == "api-generate":
        return "edits-then-api-generate"
    return mode


def resolve_edits_field_names(field_name: str) -> list[str]:
    if field_name == "both":
        return ["image", "image[]"]
    return [field_name]


if __name__ == "__main__":
    sys.exit(main())
