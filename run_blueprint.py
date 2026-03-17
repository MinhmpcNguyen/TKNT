from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from blueprint import BlueprintInputs, BlueprintRunner
from config.logging_config import setup_logging

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


def _load_user_input(path: str) -> Mapping[str, JsonValue]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "user_input" in payload:
        user_input = payload["user_input"]
        if isinstance(user_input, Mapping):
            return user_input
    if isinstance(payload, Mapping) and "description" in payload:
        return payload
    raise ValueError(
        "Input JSON must be a user_input object or contain a user_input key."
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run multi-agent blueprint pipeline.")
    parser.add_argument("input", help="Path to user_input JSON.")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--rag-query", default=None)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    user_input = _load_user_input(args.input)
    runner = BlueprintRunner()
    result = runner.run(
        BlueprintInputs(
            user_input=user_input,
            tenant_id=args.tenant_id,
            rag_query=args.rag_query,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir,
            persist_outputs=True,
            persist_iterations=True,
        )
    )

    print(f"status={result.status} iterations={result.iterations}")
    print(f"outputs dir: {args.output_dir}")


if __name__ == "__main__":
    main()
