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
