from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.room_interpreter import RoomInterpreter


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


def main() -> int:
    sample_input = {
        "tenant_id": "demo_tenant",
        "user_input": {
            "height": 2800,
            # L-shape: outer 6000x4000, cut-out 2000x2000 at top-right
            "shape_points": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 2000},
                {"x": 4000, "y": 2000},
                {"x": 4000, "y": 4000},
                {"x": 0, "y": 4000},
            ],
            "room_type": "bedroom",
            "windows": 1,
            "window_direction": "SE",
            "style": "japandi",
        },
        "floorplan_geometry": {
            "room": {
                "room_id": "room_1",
                "name": "Main",
                "polygon_mm": [
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 2000},
                    {"x": 4000, "y": 2000},
                    {"x": 4000, "y": 4000},
                    {"x": 0, "y": 4000},
                ],
            },
            "walls": [],
            "openings": {
                "doors": [
                    {
                        "door_id": "door_1",
                        "start_mm": {"x": 0, "y": 2000},
                        "end_mm": {"x": 0, "y": 2600},
                        "hinge_side": "left",
                    }
                ],
                "windows": [
                    {
                        "window_id": "window_1",
                        "start_mm": {"x": 2000, "y": 0},
                        "end_mm": {"x": 4000, "y": 0},
                    }
                ],
            },
        },
        "constraints": {},
        "max_iterations": 12,
    }

    agent = RoomInterpreter()
    try:
        raw = agent.generate_raw(sample_input)
        print("RAW_RESPONSE")
        print(raw)
        print("\nPARSED_RESPONSE")
        parsed = agent.generate(sample_input)
        parsed_payload = parsed.model_dump()
        print(parsed_payload)
    except Exception as exc:
        print("TEST_FAILED")
        print(str(exc))
        return 1

    raw_payload = _parse_raw_json(raw)

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "room_interpreter_output.json"
    result = {
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
