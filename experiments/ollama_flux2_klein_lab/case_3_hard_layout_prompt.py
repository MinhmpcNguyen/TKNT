from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_SIZE,
    extract_image_bytes,
    post_json,
    pretty_json_text,
    resolve_prompt_text,
    write_image,
    write_response_snapshot,
)

LOGGER = logging.getLogger("flux2_klein_case_3_hard_layout")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "hard_layout_prompt.txt"
DEFAULT_NEGATIVE_PROMPT_FILE = PROMPTS_DIR / "hard_layout_negative_prompt.txt"
DEFAULT_CONSTRAINTS_FILE = PROMPTS_DIR / "hard_layout_compiler_constraints.txt"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs_hard_layout"
    / "case_3_hard_layout_result.png"
)
HARD_PREFIX = (
    "Floor-plan fidelity is mandatory. Treat all layout instructions as hard "
    "constraints, not suggestions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a hard-constrained direct prompt to Ollama "
            "x/flux2-klein:latest without using the repo compiler."
        ),
    )
    parser.add_argument("--prompt", type=str, help="Optional prompt override.")
    parser.add_argument("--prompt-file", type=Path, help="Optional custom prompt file.")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        help="Optional negative prompt override.",
    )
    parser.add_argument(
        "--negative-prompt-file",
        type=Path,
        help="Optional custom negative prompt file.",
    )
    parser.add_argument(
        "--constraints-file",
        type=Path,
        default=DEFAULT_CONSTRAINTS_FILE,
        help="Optional compiler constraints file to append to the main prompt.",
    )
    parser.add_argument(
        "--skip-constraints",
        action="store_true",
        help="Do not append the compiler constraints block to the prompt.",
    )
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
        help="Requested output size, for example 1024x1024.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Preferred output path. The suffix may be adjusted to match the actual image format.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    prompt_text = resolve_prompt_text(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        default_prompt=DEFAULT_PROMPT_FILE.read_text(encoding="utf-8"),
    )
    negative_prompt = resolve_prompt_text(
        prompt=args.negative_prompt,
        prompt_file=args.negative_prompt_file,
        default_prompt=DEFAULT_NEGATIVE_PROMPT_FILE.read_text(encoding="utf-8"),
    )
    constraints_text = ""
    if not args.skip_constraints:
        constraints_path = args.constraints_file
        if constraints_path is None:
            raise ValueError("Constraints file is required unless --skip-constraints is set.")
        constraints_text = constraints_path.read_text(encoding="utf-8").strip()

    final_prompt = compose_final_prompt(
        prompt_text=prompt_text,
        constraints_text=constraints_text,
    )
    request_payload = {
        "model": args.model,
        "prompt": final_prompt,
        "negative_prompt": negative_prompt.strip(),
        "size": args.size,
        "response_format": "b64_json",
    }

    output_base = args.output.expanduser().resolve()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = output_base.parent / f"{output_base.stem}.prompt.txt"
    negative_path = output_base.parent / f"{output_base.stem}.negative_prompt.txt"
    payload_path = output_base.parent / f"{output_base.stem}.request.json"
    prompt_path.write_text(final_prompt, encoding="utf-8")
    negative_path.write_text(negative_prompt.strip(), encoding="utf-8")
    payload_path.write_text(
        pretty_json_text(request_payload),
        encoding="utf-8",
    )

    LOGGER.info("Sending request to %s with model %s", args.endpoint, args.model)
    response = post_json(
        endpoint=args.endpoint,
        route="/v1/images/generations",
        payload=request_payload,
    )

    snapshot_path = write_response_snapshot(output_base.with_suffix(""), response)
    if response.status_code >= 400:
        LOGGER.error("Ollama returned status %s", response.status_code)
        LOGGER.error("Prompt snapshot: %s", prompt_path)
        LOGGER.error("Negative prompt snapshot: %s", negative_path)
        LOGGER.error("Request payload: %s", payload_path)
        LOGGER.error("Raw response snapshot: %s", snapshot_path)
        return 1

    image_bytes = extract_image_bytes(response)
    if image_bytes is None:
        LOGGER.error("Response did not include image bytes.")
        LOGGER.error("Prompt snapshot: %s", prompt_path)
        LOGGER.error("Negative prompt snapshot: %s", negative_path)
        LOGGER.error("Request payload: %s", payload_path)
        LOGGER.error("Raw response snapshot: %s", snapshot_path)
        return 1

    image_path = write_image(output_base, image_bytes)
    LOGGER.info("Generated image: %s", image_path)
    LOGGER.info("Prompt snapshot: %s", prompt_path)
    LOGGER.info("Negative prompt snapshot: %s", negative_path)
    LOGGER.info("Request payload: %s", payload_path)
    LOGGER.info("Raw response snapshot: %s", snapshot_path)
    return 0


def compose_final_prompt(*, prompt_text: str, constraints_text: str) -> str:
    blocks = [HARD_PREFIX, prompt_text.strip()]
    if constraints_text.strip():
        blocks.extend(
            [
                "Compiler constraints:",
                constraints_text.strip(),
            ]
        )
    return "\n\n".join(block for block in blocks if block)


if __name__ == "__main__":
    sys.exit(main())
