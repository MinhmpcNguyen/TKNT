from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
        trace = _registry.profile_shadow_trace_for_active_clusters(
            room_type="kitchen",
            active_clusters=[
                {
                    "cluster_id": "kitchen_workflow_core",
                    "required_bundles": [
                        {
                            "objects": [
                                {"object_type": "fridge"},
                                {"object_type": "sink"},
                            ]
                        }
                    ],
                }
            ],
        )
        self.assertEqual(trace["promotion_readiness"]["status"], "active")

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
        self.assertIsNotNone(_registry.semantic_room_rule_for("bedroom"))
        self.assertIsNotNone(_registry.semantic_room_rule_for("living_room"))
        self.assertFalse(_registry.is_profile_workflow_object("bed"))
        self.assertFalse(_registry.is_profile_workflow_object("sofa"))

    def test_profile_rule_selection_defaults_to_unified_with_legacy_diff(
        self,
    ) -> None:
        profile_rule = _registry.semantic_room_rule_for("living_room")
        legacy_rule = dict(profile_rule)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TKNT_ROOM_PROFILE_RULE_MODE", None)
            selection = _registry.select_profile_room_rule(
                room_type="living_room",
                profile_rule=profile_rule,
                legacy_rule=legacy_rule,
            )

        self.assertEqual(selection.mode, "unified")
        self.assertEqual(selection.selected_source, "profile")
        self.assertTrue(selection.profile_first_active)
        self.assertEqual(selection.dry_run_diff["status"], "equivalent")

    def test_profile_rule_selection_canary_uses_room_gate(self) -> None:
        profile_rule = _registry.semantic_room_rule_for("bedroom")
        legacy_rule = dict(profile_rule)

        with patch.dict(
            os.environ,
            {
                "TKNT_ROOM_PROFILE_RULE_MODE": "canary",
                "TKNT_ROOM_PROFILE_FIRST_ROOMS": "living_room",
            },
        ):
            bedroom = _registry.select_profile_room_rule(
                room_type="bedroom",
                profile_rule=profile_rule,
                legacy_rule=legacy_rule,
            )

        with patch.dict(
            os.environ,
            {
                "TKNT_ROOM_PROFILE_RULE_MODE": "canary",
                "TKNT_ROOM_PROFILE_FIRST_ROOMS": "bedroom",
            },
        ):
            bedroom_enabled = _registry.select_profile_room_rule(
                room_type="bedroom",
                profile_rule=profile_rule,
                legacy_rule=legacy_rule,
            )

        self.assertEqual(bedroom.selected_source, "legacy")
        self.assertFalse(bedroom.profile_first_active)
        self.assertEqual(bedroom_enabled.selected_source, "profile")
        self.assertTrue(bedroom_enabled.profile_first_active)

    def test_unified_mode_falls_back_when_legacy_profile_differs(self) -> None:
        profile_rule = _registry.semantic_room_rule_for("bedroom")
        legacy_rule = dict(profile_rule)
        profile_rule = dict(profile_rule)
        profile_rule["clusters"] = []

        with patch.dict(
            os.environ,
            {"TKNT_ROOM_PROFILE_RULE_MODE": "unified"},
        ):
            selection = _registry.select_profile_room_rule(
                room_type="bedroom",
                profile_rule=profile_rule,
                legacy_rule=legacy_rule,
            )

        self.assertEqual(selection.selected_source, "legacy")
        self.assertFalse(selection.profile_first_active)
        self.assertTrue(selection.fallback_used)
        self.assertEqual(
            selection.fallback_reason,
            "profile_dry_run_diff_not_equivalent",
        )

    def test_bedroom_shadow_profile_interprets_legacy_clusters_without_activation(
        self,
    ) -> None:
        trace = _registry.profile_shadow_trace_for_active_clusters(
            room_type="bedroom",
            active_clusters=[
                {
                    "cluster_id": "sleep_core",
                    "required_bundles": [
                        {
                            "objects": [
                                {"object_type": "bed"},
                                {"object_type": "nightstand"},
                            ]
                        }
                    ],
                },
                {
                    "cluster_id": "work_study",
                    "required_bundles": [
                        {
                            "objects": [
                                {"object_type": "vanity"},
                                {"object_type": "vanity chair"},
                            ]
                        }
                    ],
                },
            ],
        )

        self.assertIsNotNone(trace)
        self.assertEqual(trace["profile_id"], "bedroom_legacy")
        self.assertEqual(trace["mode"], "shadow")
        self.assertFalse(trace["layout_traits_enabled"])
        self.assertEqual(trace["recognized_count"], 4)
        self.assertIn("anchor", trace["trait_counts"])
        self.assertIn("support", trace["trait_counts"])

    def test_living_room_shadow_profile_interprets_core_furniture(self) -> None:
        trace = _registry.profile_shadow_trace_for_active_clusters(
            room_type="living_room",
            active_clusters=[
                {
                    "cluster_id": "living_primary",
                    "required_bundles": [
                        {
                            "objects": [
                                {"object_type": "couch"},
                                {"object_type": "coffee table"},
                                {"object_type": "media console"},
                                {"object_type": "media_shelf"},
                            ]
                        }
                    ],
                }
            ],
        )

        self.assertIsNotNone(trace)
        self.assertEqual(trace["profile_id"], "living_room_legacy")
        self.assertEqual(trace["mode"], "shadow")
        self.assertEqual(trace["coverage_ratio"], 1.0)
        self.assertEqual(trace["trait_counts"]["anchor"], 2)
        self.assertEqual(trace["trait_counts"]["surface"], 3)
        self.assertEqual(trace["trait_counts"]["storage"], 1)
        self.assertEqual(
            trace["promotion_readiness"]["status"], "ready_for_profile_first"
        )
        self.assertTrue(trace["promotion_readiness"]["eligible_for_profile_first"])
        self.assertEqual(
            _registry.canonical_profile_object_type("media shelving"),
            "media_shelf",
        )


if __name__ == "__main__":
    unittest.main()
