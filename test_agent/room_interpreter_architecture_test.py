from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fake_clients_package = types.ModuleType("clients")
fake_clients_package.__path__ = []  # type: ignore[attr-defined]
sys.modules["clients"] = fake_clients_package

fake_base_client_module = types.ModuleType("clients.base_client")


class _FakeChatMessage(TypedDict, total=False):
    role: str
    content: str


fake_base_client_module.ChatMessage = _FakeChatMessage
sys.modules["clients.base_client"] = fake_base_client_module

fake_llm_module = types.ModuleType("clients.llm_client")


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    def chat_completion(self, *args: object, **kwargs: object) -> _FakeResponse:
        _ = args, kwargs
        return _FakeResponse(
            json.dumps(
                {
                    "notes": [
                        "Japandi bedroom with calm visual tone.",
                        "Keep circulation clear and avoid clutter.",
                        "Include a practical work corner if space allows.",
                    ]
                }
            )
        )


fake_llm_module.get_llm_client = lambda: _FakeClient()
sys.modules["clients.llm_client"] = fake_llm_module

_room_interpreter = importlib.import_module("agent.room_interpreter")
RoomInterpreter = _room_interpreter.RoomInterpreter


def _sample_input() -> dict[str, object]:
    return {
        "tenant_id": "demo_tenant",
        "user_input": {
            "description": "Thiết kế bedroom phong cách japandi",
            "special_description": "có góc làm việc nhỏ, tránh bừa bộn",
            "room_type": "bedroom",
            "window_direction": "SE",
            "style": "japandi",
            "shape_points": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 4000},
                {"x": 0, "y": 4000},
            ],
        },
        "floorplan_geometry": {
            "room": {
                "room_id": "room_1",
                "name": "Main",
                "polygon_mm": [
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 4000},
                    {"x": 0, "y": 4000},
                ],
            },
            "walls": [],
            "openings": {
                "doors": [
                    {
                        "door_id": "door_1",
                        "start_mm": {"x": 0, "y": 1600},
                        "end_mm": {"x": 0, "y": 2500},
                        "leaf_width_mm": 900,
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
        "constraints": {"feng_shui": "Keep the bed away from the door direct line."},
        "max_iterations": 12,
    }


class RoomInterpreterArchitectureTest(unittest.TestCase):
    def test_generate_uses_deterministic_geometry_and_converts_openings(self) -> None:
        result = RoomInterpreter().generate(
            _sample_input(),
            description="Need a calm bedroom.",
            special_notes="Prefer low clutter.",
        )
        payload = result.model_dump()

        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["room"]["room_id"], "room_1")
        self.assertEqual(len(payload["room"]["polygon_ccw"]), 4)

        doors = payload["openings"]["doors"]
        self.assertEqual(len(doors), 1)
        self.assertEqual(doors[0]["id"], "door_1")
        self.assertEqual(
            doors[0]["segment_mm"],
            [{"x": 0, "y": 1600}, {"x": 0, "y": 2500}],
        )
        self.assertEqual(doors[0]["swing_radius_mm"], 900)
        self.assertEqual(doors[0]["hinge_hint"], "LEFT")

        windows = payload["openings"]["windows"]
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["id"], "window_1")
        self.assertEqual(
            windows[0]["segment_mm"],
            [{"x": 2000, "y": 0}, {"x": 4000, "y": 0}],
        )
        self.assertEqual(windows[0]["clearance_mm"], 150)

        obstacle_ids = {
            row["id"]
            for row in payload["obstacles"]
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
        self.assertIn("door_1_swing", obstacle_ids)
        self.assertIn("door_1_entry_clearance", obstacle_ids)
        self.assertIn("window_1_clearance", obstacle_ids)

        self.assertIn("hard_obstacles", payload)
        self.assertIn("affordance_map", payload)
        self.assertIn("topology", payload)
        self.assertIn("quality_meta", payload)
        affordance_map = payload["affordance_map"]
        self.assertGreaterEqual(len(affordance_map["usable_walls"]), 1)
        self.assertGreaterEqual(len(affordance_map["entry_landing_zones"]), 1)
        self.assertGreaterEqual(len(affordance_map["circulation_corridors"]), 1)
        self.assertGreaterEqual(len(affordance_map["daylight_regions"]), 1)
        self.assertGreaterEqual(len(affordance_map["privacy_regions"]), 1)
        self.assertGreaterEqual(len(affordance_map["focal_surfaces"]), 1)
        self.assertEqual(
            payload["quality_meta"]["settings"]["window_clearance_mm"],
            150,
        )

        self.assertGreaterEqual(len(payload["notes"]), 2)
        self.assertTrue(any("japandi" in note.lower() for note in payload["notes"]))

    def test_generate_raw_returns_full_room_payload_json(self) -> None:
        raw = RoomInterpreter().generate_raw(_sample_input())
        parsed = json.loads(raw)

        self.assertEqual(parsed["status"], "OK")
        self.assertIn("room", parsed)
        self.assertIn("openings", parsed)
        self.assertIn("affordance_map", parsed)
        self.assertIn("topology", parsed)
        self.assertIn("quality_meta", parsed)

    def test_generate_respects_explicit_empty_openings(self) -> None:
        payload = {
            "user_input": {
                "description": "Small bedroom without windows in the payload.",
                "room_type": "bedroom",
                "windows": 0,
                "shape_points": [
                    {"x": 0, "y": 0},
                    {"x": 3000, "y": 0},
                    {"x": 3000, "y": 4000},
                    {"x": 0, "y": 4000},
                ],
            },
            "doors": [],
            "windows": [],
        }

        result = RoomInterpreter().generate(payload)
        data = result.model_dump()

        self.assertEqual(data["openings"]["doors"], [])
        self.assertEqual(data["openings"]["windows"], [])
        self.assertNotIn("windows", data["missing"])
        self.assertFalse(
            any("generated a deterministic default" in note for note in data["notes"])
        )


if __name__ == "__main__":
    unittest.main()
