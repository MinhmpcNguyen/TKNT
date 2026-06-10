from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.pop("agent.stylist", None)
sys.modules.pop("stylist.deterministic_layout", None)

fake_clients_package = types.ModuleType("clients")
fake_clients_package.__path__ = []  # type: ignore[attr-defined]
sys.modules["clients"] = fake_clients_package

fake_llm_module = types.ModuleType("clients.llm_client")
LAST_CHAT_KWARGS: dict[str, object] = {}
CHAT_COMPLETION_CALLS: list[dict[str, object]] = []
CHAT_COMPLETION_OUTCOMES: list[object] = []


def _default_style_plan_payload() -> dict[str, object]:
    return {
        "room_surfaces": {
            "wall_color_hex": "#EDE3D6",
            "floor_color_hex": "#AF7F52",
            "ceiling_color_hex": "#FAF6EF",
        },
        "opening_colors": {
            "doors": [{"id": "door_1", "color_hex": "#7B5F49"}],
            "windows": [{"id": "window_1", "color_hex": "#E6D8C8"}],
        },
        "object_styles": [
            {
                "instance_id": "desk_1",
                "color_hex": "#8D6B4D",
                "material": "oak_wood",
            },
            {
                "instance_id": "bed_1",
                "color_hex": "#C2B5A3",
                "material": "linen",
            },
            {
                "instance_id": "desk_lamp_001",
                "color_hex": "#C76B42",
                "material": "brushed_metal",
            },
            {
                "instance_id": "laptop_001",
                "color_hex": "#444444",
                "material": "anodized_aluminum",
            },
            {
                "instance_id": "rug_001",
                "color_hex": "#B9835A",
                "material": "woven_wool",
            },
            {
                "instance_id": "curtain_001",
                "color_hex": "#D8C4AE",
                "material": "linen",
            },
            {
                "instance_id": "ceiling_light_001",
                "color_hex": "#D5A45D",
                "material": "brass",
            },
        ],
        "notes": ["Palette locked from the requested Japandi direction."],
    }


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
        _ = args
        LAST_CHAT_KWARGS.clear()
        LAST_CHAT_KWARGS.update(kwargs)
        CHAT_COMPLETION_CALLS.append(dict(kwargs))
        if CHAT_COMPLETION_OUTCOMES:
            outcome = CHAT_COMPLETION_OUTCOMES.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, str):
                return _FakeResponse(outcome)
            return _FakeResponse(json.dumps(outcome))
        return _FakeResponse(json.dumps(_default_style_plan_payload()))


fake_llm_module.get_llm_client = lambda *args, **kwargs: _FakeClient()
sys.modules["clients.llm_client"] = fake_llm_module

fake_tools_module = types.ModuleType("stylist.tools")


def _fake_list_inventory_by_types(
    *,
    tenant_id: str | None,
    types: list[str],
    limit: int | None = 200,
) -> dict[str, object]:
    _ = tenant_id, types, limit
    return {
        "items": [
            {
                "id": "inv_desk_lamp",
                "type": "desk_lamp",
                "asset_type": "desk_lamp",
                "dimensions": {
                    "length_mm": 220,
                    "width_mm": 220,
                    "height_mm": 480,
                },
            },
            {
                "id": "inv_laptop",
                "type": "laptop",
                "asset_type": "laptop",
                "dimensions": {
                    "length_mm": 350,
                    "width_mm": 240,
                    "height_mm": 20,
                },
            },
            {
                "id": "inv_rug",
                "type": "rug",
                "asset_type": "rug",
                "dimensions": {
                    "length_mm": 1800,
                    "width_mm": 1200,
                    "height_mm": 10,
                },
            },
            {
                "id": "inv_pet_bed",
                "type": "pet_bed",
                "asset_type": "pet_bed",
                "dimensions": {
                    "length_mm": 700,
                    "width_mm": 550,
                    "height_mm": 180,
                },
            },
            {
                "id": "inv_curtain",
                "type": "curtain",
                "asset_type": "curtain",
                "dimensions": {
                    "length_mm": 1800,
                    "width_mm": 140,
                    "height_mm": 1800,
                },
            },
            {
                "id": "inv_ceiling_light",
                "type": "ceiling_light",
                "asset_type": "ceiling_light",
                "dimensions": {
                    "length_mm": 450,
                    "width_mm": 450,
                    "height_mm": 180,
                },
            },
        ]
    }


fake_tools_module.ListInventoryByTypes = _fake_list_inventory_by_types
fake_tools_module.TOOL_REGISTRY = {}
fake_tools_module.TOOL_SCHEMAS = []
sys.modules["stylist.tools"] = fake_tools_module

_stylist = importlib.import_module("agent.stylist")
_deterministic_layout = importlib.import_module("stylist.deterministic_layout")
Stylist = _stylist.Stylist


class _OpenAIStyleConfig:
    PROVIDER = "openai"
    STRICT_SINGLE_TEXT_MODEL = ""

    @classmethod
    def agent_model(cls, key: str) -> str | None:
        _ = key
        return "gpt-5.4-mini"

    @classmethod
    def primary_model_name(cls) -> str:
        return "gpt-5.4-mini"


def _sample_layout() -> dict[str, object]:
    return {
        "status": "OK",
        "room": {
            "room_id": "room_1",
            "room_type": "bedroom",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 4000, "y": 0},
                {"x": 4000, "y": 4000},
                {"x": 0, "y": 4000},
            ],
            "obstacles": [],
            "openings": {
                "doors": [
                    {
                        "id": "door_1",
                        "segment_mm": [
                            {"x": 4000, "y": 900},
                            {"x": 4000, "y": 1700},
                        ],
                    }
                ],
                "windows": [
                    {
                        "id": "window_1",
                        "segment_mm": [
                            {"x": 0, "y": 1000},
                            {"x": 0, "y": 2500},
                        ],
                    }
                ],
            },
        },
        "openings": {
            "doors": [
                {
                    "id": "door_1",
                    "segment_mm": [
                        {"x": 4000, "y": 900},
                        {"x": 4000, "y": 1700},
                    ],
                }
            ],
            "windows": [
                {
                    "id": "window_1",
                    "segment_mm": [
                        {"x": 0, "y": 1000},
                        {"x": 0, "y": 2500},
                    ],
                }
            ],
        },
        "objects": [
            {
                "instance_id": "desk_1",
                "object_type": "desk",
                "cluster_id": "work",
                "bbox": {
                    "min_x": 2500,
                    "min_y": 300,
                    "max_x": 3600,
                    "max_y": 900,
                },
                "polygon_ccw": [
                    {"x": 2500, "y": 300},
                    {"x": 3600, "y": 300},
                    {"x": 3600, "y": 900},
                    {"x": 2500, "y": 900},
                ],
            },
            {
                "instance_id": "bed_1",
                "object_type": "bed",
                "cluster_id": "sleep",
                "bbox": {
                    "min_x": 400,
                    "min_y": 2200,
                    "max_x": 2000,
                    "max_y": 3200,
                },
                "polygon_ccw": [
                    {"x": 400, "y": 2200},
                    {"x": 2000, "y": 2200},
                    {"x": 2000, "y": 3200},
                    {"x": 400, "y": 3200},
                ],
            },
        ],
    }


def _living_room_layout_with_tv_console() -> dict[str, object]:
    return {
        "status": "OK",
        "room": {
            "room_id": "living_room_1",
            "room_type": "living_room",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 4000},
                {"x": 0, "y": 4000},
            ],
            "obstacles": [],
            "openings": {
                "doors": [],
                "windows": [],
            },
            "surfaces": {},
        },
        "objects": [
            {
                "instance_id": "tv_console_1",
                "object_type": "tv_console",
                "polygon_ccw": [
                    {"x": 4700, "y": 1200},
                    {"x": 5600, "y": 1200},
                    {"x": 5600, "y": 1600},
                    {"x": 4700, "y": 1600},
                ],
                "bbox": {
                    "min_x": 4700,
                    "min_y": 1200,
                    "max_x": 5600,
                    "max_y": 1600,
                },
                "color_hex": "#8D6B4D",
                "material": "oak_wood",
                "place_on": None,
            }
        ],
        "openings": {
            "doors": [],
            "windows": [],
        },
    }


class StylistArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        LAST_CHAT_KWARGS.clear()
        CHAT_COMPLETION_CALLS.clear()
        CHAT_COMPLETION_OUTCOMES.clear()

    def test_generate_style_plan_forwards_explicit_model_name(self) -> None:
        Stylist().generate_style_plan(
            layout_json=_sample_layout(),
            user_context_json={"tenant_id": "demo_tenant", "style": "japandi"},
            tenant_id="demo_tenant",
            model_name="gpt-5.4-mini",
        )

        self.assertEqual(LAST_CHAT_KWARGS.get("model_name"), "gpt-5.4-mini")

    def test_variant_uses_configured_openai_model(self) -> None:
        with mock.patch.object(_stylist, "TextLLMConfig", _OpenAIStyleConfig):
            self.assertEqual(
                _stylist.stylist_model_name_for_variant(5),
                "gpt-5.4-mini",
            )

    def test_generate_style_plan_falls_back_on_503_unavailable(self) -> None:
        CHAT_COMPLETION_OUTCOMES.extend(
            [
                RuntimeError(
                    "Server error '503 Service Unavailable' for url "
                    "'https://api.openai.com/v1/chat/completions' "
                    'OpenAI response body: {"error":{"code":503,"message":"This model is currently experiencing high demand.","status":"UNAVAILABLE"}}'
                ),
            ]
        )

        with mock.patch.object(_stylist, "TextLLMConfig", _OpenAIStyleConfig):
            style_plan = Stylist().generate_style_plan(
                layout_json=_sample_layout(),
                user_context_json={"tenant_id": "demo_tenant", "style": "japandi"},
                tenant_id="demo_tenant",
                model_name="gpt-5.4-mini",
            )

        self.assertEqual(len(CHAT_COMPLETION_CALLS), 1)
        self.assertEqual(
            [call.get("model_name") for call in CHAT_COMPLETION_CALLS],
            [
                "gpt-5.4-mini",
            ],
        )
        self.assertEqual(style_plan, {})

    def test_generate_style_plan_retries_next_model_on_invalid_json(self) -> None:
        CHAT_COMPLETION_OUTCOMES.extend(
            [
                '{"room_surfaces":{"wall_color_hex":"#F0F0F0"',
                _default_style_plan_payload(),
            ]
        )

        style_plan = Stylist().generate_style_plan(
            layout_json=_sample_layout(),
            user_context_json={"tenant_id": "demo_tenant", "style": "japandi"},
            tenant_id="demo_tenant",
            model_name="gpt-5.4-mini",
        )

        self.assertEqual(len(CHAT_COMPLETION_CALLS), 2)
        self.assertEqual(
            [call.get("model_name") for call in CHAT_COMPLETION_CALLS],
            [
                "gpt-5.4-mini",
                "gpt-5.4-mini",
            ],
        )
        self.assertEqual(
            style_plan["room_surfaces"]["wall_color_hex"],
            "#EDE3D6",
        )

    def test_generate_style_plan_repairs_code_fenced_trailing_comma_json(self) -> None:
        CHAT_COMPLETION_OUTCOMES.append(
            """```json
{
  "room_surfaces": {
    "wall_color_hex": "#EDE3D6",
    "floor_color_hex": "#AF7F52",
    "ceiling_color_hex": "#FAF6EF",
  },
  "opening_colors": {
    "doors": [{"id": "door_1", "color_hex": "#7B5F49"}],
    "windows": [{"id": "window_1", "color_hex": "#E6D8C8"}],
  },
  "object_styles": [],
  "notes": ["Palette locked from the requested Japandi direction."],
}
```"""
        )

        style_plan = Stylist().generate_style_plan(
            layout_json=_sample_layout(),
            user_context_json={"tenant_id": "demo_tenant", "style": "japandi"},
            tenant_id="demo_tenant",
            model_name="gpt-5.4-mini",
        )

        self.assertEqual(len(CHAT_COMPLETION_CALLS), 1)
        self.assertEqual(
            style_plan["room_surfaces"]["wall_color_hex"],
            "#EDE3D6",
        )
        self.assertEqual(
            style_plan["opening_colors"]["windows"][0]["id"],
            "window_1",
        )

    def test_generate_hard_codes_accessory_layout_and_only_uses_llm_for_style(
        self,
    ) -> None:
        result = Stylist().generate(
            layout_json=_sample_layout(),
            user_context_json={
                "tenant_id": "demo_tenant",
                "style": "japandi",
                "room_notes": [
                    "Keep the palette warm and quiet.",
                    "The room should feel soft and uncluttered.",
                ],
            },
            tenant_id="demo_tenant",
        )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["room"]["surfaces"]["wall_color_hex"], "#EDE3D6")
        self.assertEqual(result["room"]["surfaces"]["floor_color_hex"], "#AF7F52")
        self.assertEqual(
            result["room"]["opening_colors"]["doors"][0]["color_hex"], "#7B5F49"
        )
        self.assertEqual(
            result["room"]["opening_colors"]["windows"][0]["color_hex"], "#E6D8C8"
        )

        objects_by_id = {
            row["instance_id"]: row
            for row in result["objects"]
            if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
        }

        lamp = objects_by_id["desk_lamp_001"]
        laptop = objects_by_id["laptop_001"]
        rug = objects_by_id["rug_001"]
        curtain = objects_by_id["curtain_001"]
        ceiling_light = objects_by_id["ceiling_light_001"]

        self.assertEqual(lamp["place_on"]["method"], "on_top")
        self.assertEqual(lamp["place_on"]["target_instance_id"], "desk_1")
        self.assertEqual(laptop["place_on"]["method"], "on_top")
        self.assertEqual(laptop["place_on"]["target_instance_id"], "desk_1")
        self.assertEqual(laptop["bbox"]["max_x"] + 50, lamp["bbox"]["min_x"])
        self.assertEqual(lamp["bbox"]["min_y"], 340)
        self.assertEqual(laptop["bbox"]["min_y"], 340)

        self.assertEqual(rug["place_on"]["method"], "floor")
        self.assertEqual(rug["place_on"]["target_instance_id"], "bed_1")
        self.assertEqual(
            (rug["bbox"]["min_x"] + rug["bbox"]["max_x"]) // 2,
            1200,
        )
        self.assertEqual(
            (rug["bbox"]["min_y"] + rug["bbox"]["max_y"]) // 2,
            2700,
        )
        self.assertNotIn("throw_blanket_001", objects_by_id)
        self.assertNotIn("cushion_001", objects_by_id)
        self.assertNotIn("pet_bed_001", objects_by_id)

        self.assertEqual(curtain["place_on"]["method"], "hang_on")
        self.assertEqual(curtain["place_on"]["target_instance_id"], "window_1")
        self.assertEqual(ceiling_light["place_on"]["method"], "hang_on")
        self.assertEqual(ceiling_light["place_on"]["target_instance_id"], "ceiling")

        self.assertEqual(lamp["color_hex"], "#C76B42")
        self.assertEqual(laptop["color_hex"], "#444444")
        self.assertEqual(rug["color_hex"], "#B9835A")
        self.assertIn(
            "Palette locked from the requested Japandi direction.",
            result["notes"],
        )

    def test_existing_floor_side_object_is_preserved_as_regular_layout_object(
        self,
    ) -> None:
        layout = _sample_layout()
        objects = layout["objects"]
        self.assertIsInstance(objects, list)
        objects.append(
            {
                "instance_id": "pet_bed_1",
                "object_type": "pet_bed",
                "cluster_id": "sleep",
                "bbox": {
                    "min_x": 2200,
                    "min_y": 2400,
                    "max_x": 2900,
                    "max_y": 2950,
                },
                "polygon_ccw": [
                    {"x": 2200, "y": 2400},
                    {"x": 2900, "y": 2400},
                    {"x": 2900, "y": 2950},
                    {"x": 2200, "y": 2950},
                ],
            }
        )

        result = Stylist().apply_style_plan(
            layout_json=layout,
            user_context_json={"tenant_id": "demo_tenant", "style": "japandi"},
            tenant_id="demo_tenant",
            style_plan={},
        )

        objects_by_id = {
            row["instance_id"]: row
            for row in result["objects"]
            if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
        }

        self.assertIn("pet_bed_1", objects_by_id)
        self.assertEqual(objects_by_id["pet_bed_1"]["object_type"], "pet_bed")
        self.assertEqual(objects_by_id["pet_bed_1"]["source"], "existing")
        self.assertNotIn("pet_bed_001", objects_by_id)

    def test_floor_side_inventory_is_not_selected_as_auto_decor(self) -> None:
        self.assertEqual(
            _deterministic_layout._select_anchor_candidates(
                anchor="bed",
                room_type="bedroom",
                candidates=["rug", "pet_bed"],
            ),
            ["rug"],
        )
        self.assertEqual(
            _deterministic_layout._select_anchor_candidates(
                anchor="__utility_zone__",
                room_type="bedroom",
                candidates=["dehumidifier", "heater"],
            ),
            [],
        )

    def test_ceiling_accessory_is_clamped_inside_room_bounds(self) -> None:
        room_rect = _deterministic_layout._Rect(0, 0, 4000, 3500)
        support = _deterministic_layout._Support(
            instance_id="dining_table",
            object_type="dining_table",
            cluster_id="kitchen_dining_core",
            rect=_deterministic_layout._Rect(1250, 0, 2790, 720),
        )

        rows = _deterministic_layout._place_ceiling_items(
            room_rect=room_rect,
            supports=[support],
            item_types=["track_light"],
            inventory_profiles={},
            existing_ids=set(),
            counters={},
        )

        self.assertEqual(len(rows), 1)
        bbox = rows[0]["bbox"]
        self.assertGreaterEqual(bbox["min_x"], 0)
        self.assertGreaterEqual(bbox["min_y"], 0)
        self.assertLessEqual(bbox["max_x"], 4000)
        self.assertLessEqual(bbox["max_y"], 3500)

    def test_generate_does_not_add_tv_on_top_of_tv_console(self) -> None:
        result = Stylist().generate(
            layout_json=_living_room_layout_with_tv_console(),
            user_context_json={
                "tenant_id": "demo_tenant",
                "style": "japandi",
            },
            tenant_id="demo_tenant",
        )

        object_types = {
            str(row.get("object_type") or "")
            for row in result["objects"]
            if isinstance(row, dict)
        }
        self.assertIn("tv_console", object_types)
        self.assertNotIn("tv", object_types)


if __name__ == "__main__":
    unittest.main()
