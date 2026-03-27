from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PROMPT = "Introduce yourself in one short sentence."


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY")
        return 1

    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}")
        print(error_body)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}")
        return 1

    elapsed = time.perf_counter() - started_at
    data = json.loads(raw)
    text = extract_text(data)

    print(f"Model: {model}")
    print(f"Elapsed: {elapsed:.2f}s")
    print("Response:")
    print(text)
    return 0


def extract_text(data: dict[str, object]) -> str:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return json.dumps(data, ensure_ascii=False, indent=2)

    first = candidates[0]
    if not isinstance(first, dict):
        return json.dumps(data, ensure_ascii=False, indent=2)

    content = first.get("content")
    if not isinstance(content, dict):
        return json.dumps(data, ensure_ascii=False, indent=2)

    parts = content.get("parts")
    if not isinstance(parts, list):
        return json.dumps(data, ensure_ascii=False, indent=2)

    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts).strip() or json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
