from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_semantic_layout_planner = importlib.import_module("agent.semantic_layout_planner")
SemanticLayoutPlanner = _semantic_layout_planner.SemanticLayoutPlanner
semantic_program_to_seed_relation_payload = (
    _semantic_layout_planner.semantic_program_to_seed_relation_payload
)
_tier_count_director = importlib.import_module("agent.tier_count_director")
TierCountDirector = _tier_count_director.TierCountDirector
_kitchen_profile = importlib.import_module("layout.kitchen_profile")
kitchen_fallback_size_profile = _kitchen_profile.kitchen_fallback_size_profile


def _kitchen_room() -> dict[str, Any]:
    return {
        "room_type": "kitchen",
        "room": {
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 4000, "y": 0},
                {"x": 4000, "y": 3500},
                {"x": 0, "y": 3500},
            ],
            "openings": {"doors": [], "windows": []},
        },
        "obstacles": [],
    }


def _profile_map(object_types: set[str]) -> dict[str, Any]:
    profiles: dict[str, Any] = {
        "__generic__": {
            "rep_dims_m": {
                "S": {"L": 0.4, "W": 0.4, "A": 0.16},
                "M": {"L": 0.7, "W": 0.6, "A": 0.42},
                "L": {"L": 1.0, "W": 0.8, "A": 0.80},
            }
        }
    }
    for object_type in object_types:
        profile = kitchen_fallback_size_profile(object_type)
        if profile is not None:
            profiles[object_type] = profile
    return {"size_profiles_by_category": profiles}


class KitchenProfileArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_get_tool_registry = _tier_count_director._get_tool_registry

    def tearDown(self) -> None:
        _tier_count_director._get_tool_registry = self._original_get_tool_registry

    def test_semantic_planner_uses_gated_kitchen_workflow_core(self) -> None:
        program = SemanticLayoutPlanner().generate(
            room_model_json=_kitchen_room(),
            room_type="kitchen",
            brief_text=(
                "Design a compact kitchen with a fridge, sink, stove, and prep counter."
            ),
            use_llm=False,
        )
        payload = program.model_dump()

        clusters_by_id = {
            row["cluster_id"]: row
            for row in payload["active_clusters"]
            if isinstance(row, dict)
        }
        self.assertIn("kitchen_workflow_core", clusters_by_id)
        self.assertNotIn("kitchen_island_optional", clusters_by_id)

        core_objects = {
            row["object_type"]: row
            for bundle in clusters_by_id["kitchen_workflow_core"]["required_bundles"]
            for row in bundle["objects"]
        }
        self.assertEqual(
            set(core_objects),
            {"dishwasher", "fridge", "kitchen_base_cabinet", "sink", "stove"},
        )
        self.assertEqual(
            core_objects["kitchen_base_cabinet"]["role"], "dominant_anchor"
        )
        self.assertEqual(core_objects["fridge"]["role"], "workflow_anchor")
        self.assertEqual(core_objects["sink"]["role"], "workflow_anchor")
        self.assertEqual(core_objects["stove"]["role"], "workflow_anchor")

        cluster_payload = semantic_program_to_seed_relation_payload(payload)
        cluster_tags = {
            row["cluster_id"]: row["tag"] for row in cluster_payload["clusters"]
        }
        self.assertEqual(cluster_tags["kitchen_workflow_core"], "kitchen")

    def test_kitchen_dining_core_is_floating_and_solver_droppable(self) -> None:
        program = SemanticLayoutPlanner().generate(
            room_model_json=_kitchen_room(),
            room_type="kitchen",
            brief_text="Design a compact eat-in kitchen with a dining table and chair.",
            use_llm=False,
        )
        payload = program.model_dump()

        clusters_by_id = {
            row["cluster_id"]: row
            for row in payload["active_clusters"]
            if isinstance(row, dict)
        }
        dining = clusters_by_id["kitchen_dining_core"]
        self.assertEqual(dining["layout_role"], "support")
        self.assertEqual(dining["semantic_role"], "kitchen_dining_zone")
        self.assertTrue(dining["zone_claims"]["floating_allowed"])
        self.assertEqual(dining["zone_claims"]["wall_affinity"], "medium")
        self.assertEqual(dining["zone_claims"]["daylight_affinity"], "high")
        self.assertIn(
            "claim_daylight",
            {row["type"] for row in dining["relation_intents"]},
        )

        cluster_payload = semantic_program_to_seed_relation_payload(payload)
        dining_payload = next(
            row
            for row in cluster_payload["clusters"]
            if row["cluster_id"] == "kitchen_dining_core"
        )
        anchor_policy = dining_payload["cluster_rules"]["anchor_first_policy"]
        self.assertIn("dining_table", anchor_policy["protected_ids"])
        self.assertIn("dining_chair", anchor_policy["droppable_ids"])
        self.assertIn(
            "drop_dining_chair",
            dining_payload["cluster_rules"]["degradation_ladder"],
        )

    def test_kitchen_dining_generic_drop_action_keeps_chair_droppable(self) -> None:
        program = (
            SemanticLayoutPlanner()
            .generate(
                room_model_json=_kitchen_room(),
                room_type="kitchen",
                brief_text="Design a compact eat-in kitchen with a dining table and chair.",
                use_llm=False,
            )
            .model_dump()
        )
        dining = next(
            row
            for row in program["active_clusters"]
            if row["cluster_id"] == "kitchen_dining_core"
        )
        dining["degradation_ladder"] = [
            "drop_chair",
            "reduce_chair_count",
            "drop_table_last",
        ]

        cluster_payload = semantic_program_to_seed_relation_payload(program)
        dining_payload = next(
            row
            for row in cluster_payload["clusters"]
            if row["cluster_id"] == "kitchen_dining_core"
        )
        anchor_policy = dining_payload["cluster_rules"]["anchor_first_policy"]

        self.assertIn("dining_chair", anchor_policy["droppable_ids"])
        self.assertNotIn("dining_table", anchor_policy["droppable_ids"])

    def test_kitchen_dining_profile_normalization_restores_floating_claims(
        self,
    ) -> None:
        program = (
            SemanticLayoutPlanner()
            .generate(
                room_model_json=_kitchen_room(),
                room_type="kitchen",
                brief_text="Design a compact eat-in kitchen with a dining table and chair.",
                use_llm=False,
            )
            .model_dump()
        )
        dining = next(
            row
            for row in program["active_clusters"]
            if row["cluster_id"] == "kitchen_dining_core"
        )
        dining["zone_claims"] = {
            "preferred_regions": ["floating_center_zone"],
            "avoid_regions": ["center_access_lane"],
            "wall_affinity": "high",
            "daylight_affinity": "medium",
            "privacy_affinity": "low",
            "floating_allowed": False,
        }

        normalized = _semantic_layout_planner._normalize_profile_semantic_program(
            program,
            affordance_summary=_semantic_layout_planner.summarize_room_affordance(
                _kitchen_room()
            ),
        )
        normalized_dining = next(
            row
            for row in normalized["active_clusters"]
            if row["cluster_id"] == "kitchen_dining_core"
        )

        self.assertEqual(
            normalized_dining["semantic_role"],
            "kitchen_dining_zone",
        )
        self.assertEqual(normalized_dining["layout_role"], "support")
        self.assertTrue(normalized_dining["zone_claims"]["floating_allowed"])
        self.assertEqual(normalized_dining["zone_claims"]["wall_affinity"], "medium")
        self.assertEqual(normalized_dining["zone_claims"]["daylight_affinity"], "high")

    def test_tier_count_preserves_kitchen_core_without_forcing_island(self) -> None:
        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": kwargs["decisions"],
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {
                    "kitchen_workflow_core": 4.0,
                    "kitchen_storage_support": 1.2,
                },
            }

        _tier_count_director._get_tool_registry = lambda: {
            "estimate_budget": fake_estimate_budget,
        }

        semantic_program = (
            SemanticLayoutPlanner()
            .generate(
                room_model_json=_kitchen_room(),
                room_type="kitchen",
                brief_text="Design a compact kitchen with a fridge, sink, stove, and counter.",
                use_llm=False,
            )
            .model_dump()
        )
        clusters_json = semantic_program_to_seed_relation_payload(semantic_program)
        object_types = {
            member
            for cluster in clusters_json["clusters"]
            for member in cluster.get("members", [])
            if isinstance(member, str)
        }

        result = TierCountDirector().generate(
            description="Design a compact kitchen with a fridge, sink, stove, and counter.",
            special_notes="Keep circulation clear.",
            room_model_json=_kitchen_room(),
            user_intent_json={"user_input": {"room_type": "kitchen"}},
            clusters_json=clusters_json,
            size_profiles_json=_profile_map(object_types),
        )

        self.assertEqual(result["status"], "OK")
        decisions = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict)
        }
        for required_type in ("fridge", "kitchen_base_cabinet", "sink", "stove"):
            self.assertEqual(decisions[required_type]["quantity"], 1)
            self.assertEqual(decisions[required_type]["min_keep"], 1)
        self.assertNotIn("kitchen_island", decisions)


if __name__ == "__main__":
    unittest.main()
