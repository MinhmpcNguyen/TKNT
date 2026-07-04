from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from typing import TypedDict
from unittest import mock

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
fake_llm_module.get_llm_client = lambda *args, **kwargs: None
sys.modules["clients.llm_client"] = fake_llm_module

_seed_relation_planner = importlib.import_module("agent.seed_relation_planner")
_semantic_layout_planner = importlib.import_module("agent.semantic_layout_planner")
_stylist_rules = importlib.import_module("stylist.rules")
SeedRelationPlannerOutput = importlib.import_module(
    "agent_schema.clusterF_schema"
).SeedRelationPlannerOutput
GLOBAL_LAYOUT_GRID_MM = importlib.import_module(
    "layout.grid_policy"
).GLOBAL_LAYOUT_GRID_MM


class SeedRelationPlannerArchitectureTest(unittest.TestCase):
    def test_build_input_json_uses_room_rule_for_room_type(self) -> None:
        agent = _seed_relation_planner.SeedRelationPlanner()

        payload = agent.build_input_json("bedroom")

        self.assertEqual(payload[0]["room_type"], "bedroom")
        self.assertIn("selection_guidance", payload[0])
        self.assertNotIn("no_stack", payload[0])
        guidance = payload[0]["selection_guidance"]
        self.assertIsInstance(guidance, dict)
        templates = guidance["cluster_templates"]
        self.assertIsInstance(templates, list)
        self.assertEqual(templates[0]["cluster_id"], "sleep_core")

    def test_room_rule_loader_finds_nested_room_entries(self) -> None:
        rule = _stylist_rules.get_room_selection_rule("guest_room")

        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule["room_type"], "guest_room")
        self.assertIn("selection_guidance", rule)

    def test_generate_raw_requests_json_mode_without_token_cap(self) -> None:
        captured: dict[str, object] = {}

        class _FakeClient:
            def chat_completion(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content='{"status":"OK"}')
                        )
                    ]
                )

        with mock.patch.object(
            _seed_relation_planner,
            "get_llm_client",
            return_value=_FakeClient(),
        ):
            agent = _seed_relation_planner.SeedRelationPlanner()
            raw = agent.generate_raw("living_room", description="hello")

        self.assertEqual(raw, '{"status":"OK"}')
        self.assertEqual(
            captured["kwargs"]["response_mime_type"],
            "application/json",
        )
        self.assertIsNone(captured["kwargs"]["max_tokens"])

    def test_normalize_coerces_string_notes_and_missing_to_lists(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "sleep_area",
                    "tag": "sleep",
                    "cluster_rules": {},
                }
            ],
            "notes": "",
            "missing": "need more context",
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        parsed = SeedRelationPlannerOutput.model_validate(normalized)

        self.assertEqual(parsed.notes, [])
        self.assertEqual(parsed.missing, ["need more context"])
        self.assertEqual(
            parsed.clusters[0].cluster_rules.grid_mm,
            GLOBAL_LAYOUT_GRID_MM,
        )

    def test_normalize_recovers_common_constraint_shape_errors(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "sleep_area",
                    "tag": "sleep",
                    "members": ["bed", "nightstand"],
                    "anchors": ["bed"],
                    "hard_constraints": [
                        {
                            "type": "contain_in",
                            "a": "nightstand",
                            "b": "bed",
                        },
                        {
                            "type": "dock_to_edge",
                            "a": "nightstand",
                            "b": "bed",
                            "b_edge": "foot",
                            "gap_min": 5,
                            "gap_max": 10,
                        },
                    ],
                    "soft_constraints": [
                        {
                            "type": "prefer_near",
                            "a": "nightstand",
                            "b": "bed",
                        }
                    ],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        parsed = SeedRelationPlannerOutput.model_validate(normalized)

        dock = parsed.clusters[0].hard_constraints[0]
        near = parsed.clusters[0].soft_constraints[0]

        self.assertEqual(dock.type, "dock_to_edge")
        self.assertEqual(dock.b_edge, "back")
        self.assertEqual(dock.span, "any")
        self.assertEqual(dock.gap_min, 0)
        self.assertEqual(dock.gap_max, GLOBAL_LAYOUT_GRID_MM)
        self.assertEqual(near.type, "prefer_near")
        self.assertEqual(near.weight, 5)
        self.assertTrue(
            any("contain_in between floor objects" in note for note in parsed.notes)
        )

    def test_normalize_adds_semantic_intent_for_serving_surface_seating(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "work_area",
                    "tag": "work",
                    "members": ["desk", "office_chair"],
                    "anchors": ["desk"],
                    "hard_constraints": [
                        {
                            "type": "dock_to_edge",
                            "a": "office_chair",
                            "b": "desk",
                            "b_edge": "front",
                            "span": "short_edge",
                            "gap_min": 0,
                            "gap_max": 50,
                        }
                    ],
                    "soft_constraints": [],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        cluster = normalized["clusters"][0]
        dock = cluster["hard_constraints"][0]
        semantic = cluster["cluster_rules"]["semantic_placements"][0]

        self.assertEqual(dock["span"], "center")
        self.assertEqual(semantic["kind"], "dock_to_edge")
        self.assertEqual(semantic["span"], "center")
        self.assertEqual(semantic["orientation"], "face_base")
        self.assertEqual(semantic["gap_min"], 0)
        self.assertEqual(semantic["gap_max"], 50)

    def test_normalize_adds_bedside_side_options_for_generic_head_relation(
        self,
    ) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "sleep_area",
                    "tag": "sleep",
                    "members": ["bed", "nightstand"],
                    "anchors": ["bed"],
                    "hard_constraints": [
                        {
                            "type": "anchor_side",
                            "a": "nightstand",
                            "b": "bed",
                            "side": "head",
                            "gap_min": 0,
                            "gap_max": 50,
                        }
                    ],
                    "soft_constraints": [],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        semantic = normalized["clusters"][0]["cluster_rules"]["semantic_placements"][0]

        self.assertEqual(semantic["kind"], "anchor_side")
        self.assertEqual(semantic["side_options"], ["head_left", "head_right"])
        self.assertEqual(semantic["orientation"], "same_direction")
        self.assertEqual(semantic["gap_min"], 0)
        self.assertEqual(semantic["gap_max"], 50)
        self.assertEqual(semantic["proximity"], "compact")

    def test_normalize_defaults_semantic_proximity_to_balanced(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "living_area",
                    "tag": "living",
                    "members": ["sectional_sofa", "coffee_table"],
                    "anchors": ["sectional_sofa"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {
                        "semantic_placements": [
                            {
                                "id": "coffee_table",
                                "relative_to": "sectional_sofa",
                                "kind": "anchor_side",
                                "side_options": ["head"],
                                "gap_min": 100,
                                "gap_max": 300,
                            }
                        ]
                    },
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        semantic = normalized["clusters"][0]["cluster_rules"]["semantic_placements"][0]

        self.assertEqual(semantic["proximity"], "balanced")

    def test_normalize_adds_loose_storage_semantics_without_storage_tag(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "support_misc",
                    "tag": "misc",
                    "members": ["wardrobe", "bookshelf", "dresser"],
                    "anchors": ["wardrobe"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        semantic_rows = normalized["clusters"][0]["cluster_rules"][
            "semantic_placements"
        ]
        semantic_by_id = {
            str(row.get("id") or ""): row
            for row in semantic_rows
            if isinstance(row, dict)
        }

        self.assertEqual(semantic_by_id["bookshelf"]["relative_to"], "wardrobe")
        self.assertEqual(semantic_by_id["bookshelf"]["proximity"], "loose")
        self.assertEqual(
            semantic_by_id["bookshelf"]["side_options"],
            ["left", "right", "head_left", "head_right"],
        )
        self.assertEqual(semantic_by_id["dresser"]["proximity"], "loose")

    def test_normalize_sanitizes_invalid_facing_rules(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "mixed_area",
                    "tag": "misc",
                    "members": [
                        "bench",
                        "floor_lamp",
                        "wardrobe",
                        "storage_cabinet",
                        "tv_console",
                    ],
                    "anchors": ["wardrobe"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {
                        "facing": {
                            "bench": {"front": "back"},
                            "floor_lamp": {"front": "any"},
                            "wardrobe": {"front": "any"},
                            "storage_cabinet": {"front": "any"},
                            "tv_console": {"front": "any"},
                        }
                    },
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        parsed = SeedRelationPlannerOutput.model_validate(normalized)
        facing = parsed.clusters[0].cluster_rules.facing or {}

        self.assertEqual(facing["bench"].front, "bottom")
        self.assertNotIn("floor_lamp", facing)
        self.assertEqual(facing["wardrobe"].front, "top")
        self.assertEqual(facing["storage_cabinet"].front, "top")
        self.assertEqual(facing["tv_console"].front, "top")

    def test_normalize_sanitizes_facing_even_when_soft_constraints_shape_is_bad(
        self,
    ) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "storage_area",
                    "tag": "storage",
                    "members": ["wardrobe", "storage_cabinet"],
                    "anchors": ["wardrobe"],
                    "hard_constraints": [],
                    "soft_constraints": "",
                    "cluster_rules": {
                        "facing": {
                            "wardrobe": {"front": "any"},
                            "storage_cabinet": {"front": "any"},
                        }
                    },
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        parsed = SeedRelationPlannerOutput.model_validate(normalized)
        facing = parsed.clusters[0].cluster_rules.facing or {}

        self.assertEqual(facing["wardrobe"].front, "top")
        self.assertEqual(facing["storage_cabinet"].front, "top")

    def test_normalize_repairs_anchor_not_present_in_members(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "living_storage",
                    "tag": "storage",
                    "members": [
                        "ottoman",
                        "coffee_table",
                        "console_table",
                        "tv_console",
                        "media_shelf",
                        "bookshelf",
                        "storage_cabinet",
                    ],
                    "anchors": ["sofa"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        parsed = SeedRelationPlannerOutput.model_validate(normalized)
        anchors = parsed.clusters[0].anchors

        self.assertNotIn("sofa", anchors)
        self.assertEqual(anchors, ["storage_cabinet"])
        self.assertTrue(
            any(
                "dropped anchors not present in members" in note
                for note in parsed.notes
            )
        )

    def test_normalize_rejects_anchor_outside_dominant_candidates(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "main_seating",
                    "tag": "living",
                    "members": ["coffee_table", "armchair", "sofa"],
                    "anchors": ["coffee_table"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {
                        "dominant_anchor_candidates": ["sofa", "sectional_sofa"]
                    },
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)

        self.assertEqual(normalized["clusters"][0]["anchors"], ["sofa"])
        self.assertTrue(
            any(
                "outside dominant anchor candidates" in note
                for note in normalized["notes"]
            )
        )

    def test_semantic_payload_prefers_candidate_anchor_over_corrupt_role(
        self,
    ) -> None:
        semantic_program = {
            "status": "OK",
            "room_type": "living_room",
            "active_clusters": [
                {
                    "cluster_id": "main_seating",
                    "priority": "core",
                    "activation_reason": "test",
                    "semantic_role": "core_social_zone",
                    "dominant_anchor_candidates": ["sofa", "sectional_sofa"],
                    "required_bundles": [
                        {
                            "bundle_id": "main_seating_bundle",
                            "objects": [
                                {
                                    "object_type": "coffee_table",
                                    "role": "dominant_anchor",
                                    "required": True,
                                },
                                {
                                    "object_type": "sofa",
                                    "role": "dominant_anchor",
                                    "required": True,
                                },
                            ],
                        }
                    ],
                    "zone_claims": {},
                    "relation_intents": [],
                    "degradation_ladder": [],
                }
            ],
            "notes": [],
            "missing": [],
        }

        payload = _semantic_layout_planner.semantic_program_to_seed_relation_payload(
            semantic_program
        )
        cluster = payload["clusters"][0]

        self.assertEqual(cluster["anchors"], ["sofa"])
        self.assertEqual(cluster["tag"], "living")
        self.assertEqual(
            cluster["cluster_rules"]["dominant_anchor_candidates"],
            ["sofa"],
        )

    def test_cluster_tag_treats_main_seating_with_coffee_table_as_living(
        self,
    ) -> None:
        tag = _semantic_layout_planner._cluster_tag(
            "main_seating",
            ["coffee_table", "armchair", "side_table", "sofa"],
            room_type="living_room",
        )

        self.assertEqual(tag, "living")

    def test_normalize_keeps_single_primary_anchor_for_living_cluster(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "living_zone",
                    "tag": "living",
                    "members": [
                        "sectional_sofa",
                        "sofa",
                        "coffee_table",
                        "armchair",
                    ],
                    "anchors": ["sofa", "sectional_sofa"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)

        self.assertEqual(
            normalized["clusters"][0]["anchors"],
            ["sectional_sofa"],
        )
        self.assertTrue(any("primary anchor" in note for note in normalized["notes"]))

    def test_normalize_adds_living_anchor_semantics_for_support_items(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "living_zone",
                    "tag": "living",
                    "members": [
                        "sectional_sofa",
                        "coffee_table",
                        "armchair",
                        "side_table",
                        "floor_lamp",
                    ],
                    "anchors": ["sectional_sofa"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        semantic_by_id = {
            item["id"]: item
            for item in normalized["clusters"][0]["cluster_rules"][
                "semantic_placements"
            ]
        }

        self.assertEqual(
            semantic_by_id["coffee_table"]["relative_to"],
            "sectional_sofa",
        )
        self.assertEqual(
            semantic_by_id["coffee_table"]["side_options"],
            ["head", "left", "right"],
        )
        self.assertEqual(semantic_by_id["armchair"]["relative_to"], "sectional_sofa")
        self.assertEqual(
            semantic_by_id["armchair"]["side_options"],
            ["left", "right", "head_left", "head_right"],
        )
        self.assertEqual(semantic_by_id["side_table"]["relative_to"], "armchair")
        self.assertEqual(semantic_by_id["floor_lamp"]["relative_to"], "armchair")

    def test_normalize_adds_generic_work_anchor_semantics_even_for_misc_cluster(
        self,
    ) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "flex_zone",
                    "tag": "misc",
                    "members": [
                        "desk",
                        "desk_chair",
                        "side_table",
                        "floor_lamp",
                    ],
                    "anchors": ["desk"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        semantic_by_id = {
            item["id"]: item
            for item in normalized["clusters"][0]["cluster_rules"][
                "semantic_placements"
            ]
        }

        self.assertEqual(semantic_by_id["desk_chair"]["kind"], "dock_to_edge")
        self.assertEqual(semantic_by_id["desk_chair"]["relative_to"], "desk")
        self.assertEqual(semantic_by_id["desk_chair"]["b_edge"], "front")
        self.assertEqual(semantic_by_id["desk_chair"]["span"], "center")
        self.assertEqual(semantic_by_id["side_table"]["relative_to"], "desk")
        self.assertEqual(
            semantic_by_id["side_table"]["side_options"],
            ["left", "right"],
        )
        self.assertEqual(semantic_by_id["floor_lamp"]["relative_to"], "desk")

    def test_normalize_adds_generic_media_support_semantics_even_for_misc_cluster(
        self,
    ) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "display_zone",
                    "tag": "misc",
                    "members": [
                        "tv_console",
                        "media_shelf",
                        "bookshelf",
                    ],
                    "anchors": ["tv_console"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _seed_relation_planner._normalize_seed_relation_payload(payload)
        semantic_by_id = {
            item["id"]: item
            for item in normalized["clusters"][0]["cluster_rules"][
                "semantic_placements"
            ]
        }

        self.assertEqual(semantic_by_id["media_shelf"]["relative_to"], "tv_console")
        self.assertEqual(
            semantic_by_id["media_shelf"]["side_options"],
            ["left", "right"],
        )
        self.assertEqual(
            semantic_by_id["media_shelf"]["orientation"],
            "same_direction",
        )


if __name__ == "__main__":
    unittest.main()
