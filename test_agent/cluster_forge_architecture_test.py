from __future__ import annotations

import importlib
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
fake_llm_module.get_llm_client = lambda *args, **kwargs: None
sys.modules["clients.llm_client"] = fake_llm_module

_cluster_forge = importlib.import_module("agent.cluster_forge")
ClusterForgeOutput = importlib.import_module(
    "agent_schema.clusterF_schema"
).ClusterForgeOutput
GLOBAL_LAYOUT_GRID_MM = importlib.import_module(
    "layout.grid_policy"
).GLOBAL_LAYOUT_GRID_MM


class ClusterForgeArchitectureTest(unittest.TestCase):
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

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
        parsed = ClusterForgeOutput.model_validate(normalized)

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

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
        parsed = ClusterForgeOutput.model_validate(normalized)

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

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
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

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
        semantic = normalized["clusters"][0]["cluster_rules"]["semantic_placements"][0]

        self.assertEqual(semantic["kind"], "anchor_side")
        self.assertEqual(semantic["side_options"], ["head_left", "head_right"])
        self.assertEqual(semantic["orientation"], "same_direction")
        self.assertEqual(semantic["gap_min"], 0)
        self.assertEqual(semantic["gap_max"], 50)

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

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
        parsed = ClusterForgeOutput.model_validate(normalized)
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

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
        parsed = ClusterForgeOutput.model_validate(normalized)
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

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
        parsed = ClusterForgeOutput.model_validate(normalized)
        anchors = parsed.clusters[0].anchors

        self.assertNotIn("sofa", anchors)
        self.assertEqual(anchors, ["storage_cabinet"])
        self.assertTrue(
            any(
                "dropped anchors not present in members" in note
                for note in parsed.notes
            )
        )


if __name__ == "__main__":
    unittest.main()
