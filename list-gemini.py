from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_FORGE_MODELS = [
    "gemma-4-31b-it",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemma-3-27b-it",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
DEFAULT_PROMPT = "Return a compact JSON object with keys status and note."


def _read_app_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _lookup_nested(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key
    config = _read_app_config(Path(args.config))
    config_key = _lookup_nested(config, ["services", "gemini", "api_key"])
    if isinstance(config_key, str) and config_key.strip():
        return config_key.strip()
    raise SystemExit(
        "Missing Gemini API key. Pass --api-key, set GEMINI_API_KEY, or keep it in app-config.yaml."
    )


def _resolve_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    config = _read_app_config(Path(args.config))
    config_url = _lookup_nested(config, ["services", "gemini", "base_url"])
    if isinstance(config_url, str) and config_url.strip():
        return config_url.rstrip("/")
    return DEFAULT_BASE_URL


def _http_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> tuple[int, dict[str, Any], str]:
    raw_payload = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url=url,
        data=raw_payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            if not isinstance(parsed, dict):
                parsed = {"raw": parsed}
            return int(response.status), parsed, body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"raw": body}
        if not isinstance(parsed, dict):
            parsed = {"raw": parsed}
        return int(exc.code), parsed, body
    except URLError as exc:
        raise RuntimeError(f"Network error while calling Gemini: {exc}") from exc


def _list_models(
    *, base_url: str, api_key: str, timeout_s: float
) -> list[dict[str, Any]]:
    _, parsed, _ = _http_json(
        url=f"{base_url}/models",
        api_key=api_key,
        payload=None,
        timeout_s=timeout_s,
    )
    models = parsed.get("models")
    return models if isinstance(models, list) else []


def _normalize_model_names(raw_models: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_models:
        name = raw_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _extract_text(parsed: dict[str, Any]) -> str:
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        return ""
    parts_out: list[str] = []
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
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts_out.append(text.strip())
    return "\n".join(parts_out).strip()


def _build_generate_payload(
    prompt: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }


def _probe_model(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    timeout_s: float,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    status_code, parsed, body = _http_json(
        url=f"{base_url}/models/{model_name}:generateContent",
        api_key=api_key,
        payload=_build_generate_payload(prompt, temperature, max_tokens),
        timeout_s=timeout_s,
    )
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    text = _extract_text(parsed)
    text_preview = text[:240] if text else ""
    json_ok = False
    if text:
        try:
            json.loads(text)
            json_ok = True
        except json.JSONDecodeError:
            json_ok = False
    error_message = ""
    error = parsed.get("error")
    if isinstance(error, dict):
        error_message = str(error.get("message") or "")
    return {
        "model": model_name,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "json_ok": json_ok,
        "text_preview": text_preview,
        "finish_reason": (
            parsed.get("candidates", [{}])[0].get("finishReason")
            if isinstance(parsed.get("candidates"), list) and parsed.get("candidates")
            else None
        ),
        "error_message": error_message,
        "raw_excerpt": body[:500],
    }


def _print_model_list(models: list[dict[str, Any]]) -> None:
    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name") or "")
        methods = (
            model.get("supportedGenerationMethods")
            or model.get("supported_actions")
            or []
        )
        print(f"{name}\t{methods}")


def _print_probe_results(results: list[dict[str, Any]]) -> None:
    print(
        "model\tstatus\tlatency_ms\tjson_ok\tfinish_reason\terror_message\ttext_preview"
    )
    for row in results:
        print(
            f"{row['model']}\t{row['status_code']}\t{row['latency_ms']}\t"
            f"{row['json_ok']}\t{row['finish_reason']}\t"
            f"{row['error_message']}\t{row['text_preview']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List Gemini models and benchmark generateContent latency."
    )
    parser.add_argument(
        "--config",
        default="/Users/Yuki/TKNT/app-config.yaml",
        help="Path to app-config.yaml.",
    )
    parser.add_argument("--api-key", help="Gemini API key override.")
    parser.add_argument("--base-url", help="Gemini base URL override.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_FORGE_MODELS,
        help="Models to benchmark.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt sent to generateContent.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional path to a text file for the prompt.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="maxOutputTokens for the probe request.",
    )
    args = parser.parse_args()

    api_key = _resolve_api_key(args)
    base_url = _resolve_base_url(args)
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    if args.list:
        models = _list_models(
            base_url=base_url, api_key=api_key, timeout_s=args.timeout
        )
        _print_model_list(models)
        return 0

    model_names = _normalize_model_names(args.models)
    if not model_names:
        raise SystemExit("No models provided.")

    results = [
        _probe_model(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            timeout_s=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        for model_name in model_names
    ]
    _print_probe_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
