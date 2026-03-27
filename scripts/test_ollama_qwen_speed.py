from __future__ import annotations

import json
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() or "Say hello and count from 1 to 5."
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []

    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            event = json.loads(line)
            content = ((event.get("message") or {}).get("content")) or ""
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks.append(content)
                print(content, end="", flush=True)
            if event.get("done"):
                break

    end = time.perf_counter()
    print("\n")
    print(f"Model: {MODEL}")
    print(f"Prompt chars: {len(prompt)}")
    if first_token_at is not None:
        print(f"Time to first token: {first_token_at - start:.2f}s")
    else:
        print("Time to first token: no streamed token received")
    print(f"Total time: {end - start:.2f}s")
    print(f"Response chars: {len(''.join(chunks))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
