from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.pop("agent.initial_intent_planner", None)

fake_clients_package = types.ModuleType("clients")
fake_clients_package.__path__ = []  # type: ignore[attr-defined]
sys.modules["clients"] = fake_clients_package

fake_base_client = types.ModuleType("clients.base_client")
fake_base_client.ChatMessage = dict
sys.modules["clients.base_client"] = fake_base_client

fake_llm_module = types.ModuleType("clients.llm_client")
_FAKE_CONTENT = "not-json"
_LAST_MODEL_NAME: str | None = None


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
        global _LAST_MODEL_NAME
        _ = args
        _LAST_MODEL_NAME = str(kwargs.get("model_name") or "")
        return _FakeResponse(_FAKE_CONTENT)


fake_llm_module.get_llm_client = lambda *args, **kwargs: _FakeClient()
sys.modules["clients.llm_client"] = fake_llm_module

fake_config_module = types.ModuleType("config.gemini_config")
fake_config_module.GeminiConfig = type(
    "GeminiConfig",
    (),
    {"AGENT_MODELS": {"planner": "planner-model"}},
)
sys.modules["config.gemini_config"] = fake_config_module

planner_module = importlib.import_module("agent.initial_intent_planner")
InitialIntentPlanner = planner_module.InitialIntentPlanner


def _room_model() -> dict[str, object]:
    return {
        "room": {
            "room_id": "room_1",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 3500},
                {"x": 0, "y": 3500},
            ],
        },
        "meta": {"room_type": "bedroom", "style": "japandi"},
        "openings": {"doors": [], "windows": []},
    }


class InitialIntentPlannerArchitectureTest(unittest.TestCase):
    def test_generate_uses_planner_model_name(self) -> None:
        global _FAKE_CONTENT, _LAST_MODEL_NAME
        _FAKE_CONTENT = json.dumps(
            {
                "status": "OK",
                "room_id": "room_1",
                "intents": [
                    {
                        "intent_id": "intent_1",
                        "label": "Restful Anchor",
                        "summary": "Rest-first bedroom.",
                        "focus_mode": "rest",
                        "primary_tag": "sleep",
                        "secondary_tag": "storage",
                        "circulation_priority": "high",
                        "center_open_preference": "high",
                        "support_cluster_behavior": "recede",
                        "distribution_mode": "edge_weighted",
                        "forge_guidance": ["Keep sleep dominant."],
                        "composer_guidance": ["Favor calm symmetry."],
                        "notes": ["Primary branch."],
                    }
                ],
                "notes": [],
                "missing": [],
            }
        )
        _LAST_MODEL_NAME = None

        InitialIntentPlanner().generate(
            room_model_json=_room_model(),
            description="Design a modern bedroom for 1-2 people.",
            special_notes="Keep circulation clear.",
        )

        self.assertEqual(_LAST_MODEL_NAME, "planner-model")

    def test_generate_falls_back_to_ten_generic_intents(self) -> None:
        global _FAKE_CONTENT
        _FAKE_CONTENT = "invalid-json"

        result = InitialIntentPlanner().generate(
            room_model_json=_room_model(),
            description="Design a modern bedroom for 1-2 people.",
            special_notes="Keep circulation clear.",
        )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["room_id"], "room_1")
        self.assertEqual(len(result["intents"]), 10)
        self.assertEqual(result["intents"][0]["primary_tag"], "sleep")
        self.assertEqual(result["intents"][0]["focus_mode"], "rest")
        self.assertIn("deterministic fallback", result["notes"][0].lower())

    def test_generate_deduplicates_near_identical_intent_families(self) -> None:
        global _FAKE_CONTENT
        _FAKE_CONTENT = json.dumps(
            {
                "status": "OK",
                "room_id": "room_1",
                "intents": [
                    {
                        "intent_id": "intent_a",
                        "label": "Viewing Perimeter A",
                        "summary": "Viewing-first perimeter layout.",
                        "focus_mode": "viewing",
                        "primary_tag": "living",
                        "secondary_tag": "storage",
                        "circulation_priority": "high",
                        "center_open_preference": "high",
                        "support_cluster_behavior": "recede",
                        "distribution_mode": "edge_weighted",
                        "forge_guidance": ["Keep the center open."],
                        "composer_guidance": ["Prefer perimeter seating."],
                        "notes": [],
                    },
                    {
                        "intent_id": "intent_b",
                        "label": "Viewing Perimeter B",
                        "summary": "Another viewing-first perimeter layout.",
                        "focus_mode": "viewing",
                        "primary_tag": "living",
                        "secondary_tag": "misc",
                        "circulation_priority": "medium",
                        "center_open_preference": "high",
                        "support_cluster_behavior": "balanced",
                        "distribution_mode": "edge_weighted",
                        "forge_guidance": ["Keep the center calm."],
                        "composer_guidance": ["Prefer perimeter seating."],
                        "notes": [],
                    },
                ],
                "notes": [],
                "missing": [],
            }
        )

        result = InitialIntentPlanner().generate(
            room_model_json={
                **_room_model(),
                "meta": {"room_type": "living room", "style": "modern"},
            },
            description="Design a modern living room for TV viewing.",
            special_notes="Keep the center open.",
        )

        families = {
            (
                row["focus_mode"],
                row["primary_tag"],
                row["distribution_mode"],
                row["center_open_preference"],
            )
            for row in result["intents"]
        }
        self.assertEqual(len(result["intents"]), 10)
        self.assertEqual(len(families), len(result["intents"]))


if __name__ == "__main__":
    unittest.main()
