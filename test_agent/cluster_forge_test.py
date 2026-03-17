from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def _parse_raw_json(raw: str) -> dict[str, object] | None:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for idx in range(1, len(parts), 2):
            candidate = parts[idx].strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip("\n").strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


from agent.cluster_forge import ClusterForge


def main() -> int:
    room_type = "bedroom"
    agent = ClusterForge()

    try:
        raw = agent.generate_raw(room_type)
        print("RAW_RESPONSE")
        print(raw)
        print("\nPARSED_RESPONSE")
        parsed = agent.generate(room_type)
        parsed_payload = parsed.model_dump()
        print(parsed_payload)
    except Exception as exc:
        print("TEST_FAILED")
        print(str(exc))
        return 1

    raw_payload = _parse_raw_json(raw)

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "cluster_forge_output.json"
    result = {
        "room_type": room_type,
        "raw_text": raw,
        "raw": raw_payload,
        "parsed": parsed_payload,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved result to: {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
