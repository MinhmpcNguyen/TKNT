from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_registry = importlib.import_module("layout.room_profiles.registry")


class RoomProfileRegistryTest(unittest.TestCase):
    def test_kitchen_profile_supplies_executable_policies(self) -> None:
        profile = _registry.resolve_room_profile("kitchen")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.profile_id, "kitchen")
        self.assertIn("fridge", _registry.all_profile_object_aliases())
        self.assertIn(
            "range_hood", _registry.all_profile_non_functional_contract_types()
        )
        self.assertIn("range_hood", _registry.all_profile_non_functional_layout_specs())
        self.assertIsNotNone(_registry.semantic_room_rule_for("kitchen"))
        self.assertTrue(_registry.is_profile_workflow_object("refrigerator"))
        self.assertTrue(_registry.is_profile_wall_backed_object("prep counter"))

    def test_kitchen_profile_defaults_to_eat_in_dining(self) -> None:
        room_rule = _registry.semantic_room_rule_for("kitchen")

        self.assertIsNotNone(room_rule)
        clusters = {
            str(cluster.get("cluster_id")): cluster
            for cluster in room_rule.get("clusters", [])
            if isinstance(cluster, dict)
        }
        dining = clusters["kitchen_dining_core"]
        object_program = dining["object_program"]
        tier_hints = dining["tier_count_hints"]
        object_hints = {
            str(row["object_type"]): row for row in tier_hints["object_hints"]
        }

        self.assertEqual(dining["priority"], "core")
        self.assertTrue(dining["activation"]["always_consider"])
        self.assertIn("dining_table", object_program["required"])
        self.assertIn("dining_chair", object_program["required_if_kept"])
        self.assertEqual(object_hints["dining_table"]["min_keep"], 1)
        self.assertEqual(object_hints["dining_chair"]["min_keep"], 1)
        self.assertEqual(
            _registry.canonical_profile_object_type("kitchen table"), "dining_table"
        )

    def test_kitchen_profile_places_dining_chairs_around_table(self) -> None:
        placements = _registry.semantic_placements_for_members(
            room_type="kitchen",
            cluster_id="kitchen_dining_core",
            members=["dining_table", "dining_chair"],
            anchors=["dining_table"],
        )

        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0]["id"], "dining_chair")
        self.assertEqual(placements[0]["relative_to"], "dining_table")
        self.assertEqual(placements[0]["support_role"], "secondary_seat")

    def test_bedroom_and_living_profiles_are_legacy_wrappers(self) -> None:
        bedroom = _registry.resolve_room_profile("bedroom")
        living = _registry.resolve_room_profile("living_room")

        self.assertIsNotNone(bedroom)
        self.assertIsNotNone(living)
        self.assertEqual(bedroom.profile_id, "bedroom_legacy")
        self.assertEqual(living.profile_id, "living_room_legacy")
        self.assertIsNone(_registry.semantic_room_rule_for("bedroom"))
        self.assertIsNone(_registry.semantic_room_rule_for("living_room"))
        self.assertFalse(_registry.is_profile_workflow_object("bed"))
        self.assertFalse(_registry.is_profile_workflow_object("sofa"))


if __name__ == "__main__":
    unittest.main()
