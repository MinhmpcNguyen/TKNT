from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ablation_modes import (  # noqa: E402
    ablation_metadata,
    ablation_system_id,
    apply_tier_output_for_mode,
    cluster_output_for_mode,
    resolve_ablation_mode,
    room_output_for_mode,
    skip_accessory_refill,
    style_policy_for_mode,
    target_final_count_for_mode,
    use_concept_repair,
    use_gallery_selector,
    use_material_macro_repair,
    use_request_contract,
    use_semantic_layout_planner,
    use_stylist_llm,
)


class AblationModesTest(unittest.TestCase):
    def test_resolve_accepts_aliases_and_hyphenated_modes(self) -> None:
        self.assertEqual(resolve_ablation_mode("baseline"), "full")
        self.assertEqual(resolve_ablation_mode("no-stylist-llm"), "no_stylist_llm")

        previous = os.environ.get("TKNT_ABLATION_MODE")
        os.environ["TKNT_ABLATION_MODE"] = "single-concept"
        try:
            self.assertEqual(resolve_ablation_mode(), "single_concept")
        finally:
            if previous is None:
                os.environ.pop("TKNT_ABLATION_MODE", None)
            else:
                os.environ["TKNT_ABLATION_MODE"] = previous

    def test_metadata_uses_stable_system_id(self) -> None:
        self.assertEqual(
            ablation_system_id("no_stylist_llm"),
            "TKNT-AB-NoStylistLLM",
        )
        self.assertEqual(
            ablation_metadata("no_request_contract")["ablation_mode"],
            "no_request_contract",
        )

    def test_behavior_flags_are_component_scoped(self) -> None:
        self.assertFalse(use_semantic_layout_planner("no_semantic_layout_planner"))
        self.assertFalse(use_request_contract("no_request_contract"))
        self.assertFalse(use_concept_repair("no_concept_repair"))
        self.assertFalse(use_material_macro_repair("no_material_macro_repair"))
        self.assertFalse(use_stylist_llm("no_stylist_llm"))
        self.assertFalse(use_gallery_selector("no_gallery_selector"))
        self.assertTrue(use_gallery_selector("full"))

    def test_no_tier_capacity_keeps_refill_enabled(self) -> None:
        tier_output = {
            "decisions": [
                {"object_type": "chair", "quantity": 0},
                {"object_type": "desk", "quantity": 1, "size_tier": "M"},
            ],
        }

        ablated = apply_tier_output_for_mode(
            mode="no_tier_capacity",
            tier_output=tier_output,
        )

        self.assertEqual(ablated["decisions"][0]["quantity"], 1)
        self.assertEqual(ablated["decisions"][0]["size_tier"], "S")
        self.assertTrue(ablated["capacity_control_disabled"])
        self.assertFalse(skip_accessory_refill("no_tier_capacity"))
        self.assertTrue(skip_accessory_refill("no_capacity_control"))

    def test_single_concept_reduces_target_count(self) -> None:
        self.assertEqual(target_final_count_for_mode("single_concept", 5), 1)
        self.assertEqual(target_final_count_for_mode("full", 5), 5)

    def test_no_style_policy_returns_neutral_policy(self) -> None:
        policy = style_policy_for_mode(
            mode="no_style_policy",
            room_type="bedroom",
            current_policy={"style_name": "industrial"},
        )

        self.assertEqual(policy["style_name"], "balanced")
        self.assertFalse(policy["controller"]["llm_seed_used"])

    def test_no_request_contract_strips_contract_fields_recursively(self) -> None:
        cluster_output = {
            "request_contract": {"objects": []},
            "semantic_layout_program": {
                "request_contract": {"objects": []},
                "active_clusters": [
                    {
                        "required_bundles": [
                            {
                                "objects": [
                                    {
                                        "object_type": "desk",
                                        "request_contract_intent": "must_keep",
                                    }
                                ]
                            }
                        ]
                    }
                ],
            },
        }

        stripped = cluster_output_for_mode(
            mode="no_request_contract",
            cluster_output=cluster_output,
        )

        self.assertNotIn("request_contract", stripped)
        semantic = stripped["semantic_layout_program"]
        self.assertNotIn("request_contract", semantic)
        obj = semantic["active_clusters"][0]["required_bundles"][0]["objects"][0]
        self.assertNotIn("request_contract_intent", obj)
        self.assertTrue(stripped["request_contract_disabled"])

    def test_minimal_room_context_strips_derived_guidance(self) -> None:
        room_output = {
            "room": {"polygon_ccw": []},
            "openings": {"doors": [], "windows": []},
            "hard_obstacles": [{"id": "door_clearance"}],
            "affordance_map": {"usable_walls": [{"id": "wall_1"}]},
            "topology": {"wall_graph": {"walls": []}, "room_subzones": [{"id": "a"}]},
            "quality_meta": {"notes": ["derived note"]},
            "notes": ["derived guidance"],
        }

        stripped = room_output_for_mode(
            mode="minimal_room_context",
            room_output=room_output,
        )

        self.assertEqual(stripped["hard_obstacles"], [{"id": "door_clearance"}])
        self.assertEqual(stripped["affordance_map"]["usable_walls"], [])
        self.assertEqual(stripped["topology"], {"wall_graph": {"walls": []}})
        self.assertEqual(stripped["quality_meta"]["notes"], [])
        self.assertEqual(stripped["notes"], [])


if __name__ == "__main__":
    unittest.main()
