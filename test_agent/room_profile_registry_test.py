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

    def test_profile_layout_semantics_are_registry_driven_for_kitchen(self) -> None:
        affordance_summary = {
            "entry_landing_zones": ["entry_zone"],
            "floating_zone_candidates": ["floating_center_zone"],
            "focal_surfaces": ["wall_north"],
            "primary_circulation_corridors": ["center_lane"],
            "wall_anchor_candidates": ["wall_west"],
        }

        semantic_role = _registry.profile_semantic_role_for_objects(
            cluster_id="kitchen_dining_core",
            object_types=["dining_table", "dining_chair"],
            priority="core",
            room_type="kitchen",
        )
        zone_claims = _registry.profile_zone_claims_for_objects(
            cluster_id="kitchen_dining_core",
            object_types=["dining_table", "dining_chair"],
            room_type="kitchen",
            affordance_summary=affordance_summary,
        )
        workflow_intents = _registry.profile_relation_intents_for_objects(
            cluster_id="kitchen_workflow_core",
            object_types=["fridge", "sink", "stove"],
            room_type="kitchen",
        )
        dining_intents = _registry.profile_relation_intents_for_objects(
            cluster_id="kitchen_dining_core",
            object_types=["dining_table", "dining_chair"],
            room_type="kitchen",
        )

        self.assertEqual(
            _registry.profile_cluster_tag_for_objects(
                ["dining_table"],
                room_type="kitchen",
            ),
            "kitchen",
        )
        self.assertEqual(semantic_role, "kitchen_dining_zone")
        self.assertIsNotNone(zone_claims)
        assert zone_claims is not None
        self.assertTrue(zone_claims["floating_allowed"])
        self.assertEqual(zone_claims["wall_affinity"], "medium")
        self.assertIn(
            {"type": "claim_wall", "target": "usable_wall", "strength": "hard"},
            workflow_intents,
        )
        self.assertIn(
            {
                "type": "near",
                "target_cluster": "kitchen_workflow_core",
                "strength": "soft",
            },
            dining_intents,
        )

    def test_profile_layout_semantics_have_room_level_gate(self) -> None:
        with patch.dict(
            os.environ,
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": "unified"},
        ):
            self.assertIsNone(
                _registry.profile_cluster_tag_for_objects(
                    ["sofa", "coffee_table"],
                    room_type="living_room",
                )
            )
            self.assertIsNone(
                _registry.profile_semantic_role_for_objects(
                    cluster_id="main_seating",
                    object_types=["sofa", "coffee_table"],
                    priority="core",
                    room_type="living_room",
                )
            )

    def test_bedroom_and_living_profiles_are_legacy_wrappers(self) -> None:
        bedroom = _registry.resolve_room_profile("bedroom")
        living = _registry.resolve_room_profile("living_room")

        self.assertIsNotNone(bedroom)
        self.assertIsNotNone(living)
        self.assertEqual(bedroom.profile_id, "bedroom_legacy")
        self.assertEqual(living.profile_id, "living_room_legacy")
        self.assertIsNotNone(_registry.semantic_room_rule_for("bedroom"))
        self.assertIsNotNone(_registry.semantic_room_rule_for("living_room"))
        with patch.dict(
            os.environ,
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": "shadow"},
        ):
            self.assertFalse(_registry.is_profile_workflow_object("bed"))
            self.assertFalse(_registry.is_profile_trait_object("sofa"))

    def test_layout_traits_canary_enables_living_room_only(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TKNT_ROOM_PROFILE_TRAITS_MODE": "canary",
                "TKNT_ROOM_PROFILE_TRAIT_ROOMS": "living_room",
            },
        ):
            self.assertTrue(_registry.is_profile_trait_object("sofa"))
            self.assertTrue(_registry.is_profile_floating_object("sofa"))
            self.assertFalse(_registry.is_profile_trait_object("bed"))
            self.assertEqual(
                _registry.profile_id_for_objects(["sofa"]), "living_room_legacy"
            )
            self.assertEqual(
                _registry.profile_room_type_for_objects(["sofa"]), "living_room"
            )
            self.assertIsNotNone(_registry.fallback_profile_size("media shelf"))

    def test_layout_traits_unified_enables_ready_legacy_profiles(self) -> None:
        with patch.dict(
            os.environ,
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": "unified"},
        ):
            self.assertTrue(_registry.is_profile_trait_object("sofa"))
            self.assertTrue(_registry.is_profile_trait_object("bed"))
            self.assertTrue(_registry.is_profile_wall_backed_object("wardrobe"))
            self.assertTrue(_registry.is_profile_storage_object("media_shelf"))
            self.assertEqual(
                _registry.profile_id_for_objects(
                    ["bookshelf"],
                    room_type="living_room",
                ),
                "living_room_legacy",
            )
            self.assertEqual(
                _registry.profile_room_type_for_objects(
                    ["bookshelf"],
                    room_type="bedroom",
                ),
                "bedroom",
            )

    def test_layout_traits_default_rollout_is_unified_with_shadow_rollback(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": ""},
        ):
            self.assertTrue(_registry.is_profile_trait_object("sofa"))
            trace = _registry.profile_shadow_trace_for_active_clusters(
                room_type="living_room",
                active_clusters=[
                    {
                        "cluster_id": "main_seating",
                        "required_bundles": [
                            {
                                "objects": [
                                    {"object_type": "sofa"},
                                    {"object_type": "coffee_table"},
                                ]
                            }
                        ],
                    }
                ],
            )

        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace["layout_traits_mode"], "unified")
        self.assertEqual(trace["mode"], "active")

        with patch.dict(
            os.environ,
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": "shadow"},
        ):
            self.assertFalse(_registry.is_profile_trait_object("sofa"))

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
        with patch.dict(
            os.environ,
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": "shadow"},
        ):
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
        with patch.dict(
            os.environ,
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": "shadow"},
        ):
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

    def test_living_room_trace_reports_active_when_traits_canary_is_enabled(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "TKNT_ROOM_PROFILE_TRAITS_MODE": "canary",
                "TKNT_ROOM_PROFILE_TRAIT_ROOMS": "living_room",
            },
        ):
            trace = _registry.profile_shadow_trace_for_active_clusters(
                room_type="living_room",
                active_clusters=[
                    {
                        "cluster_id": "media",
                        "required_bundles": [
                            {
                                "objects": [
                                    {"object_type": "tv_console"},
                                    {"object_type": "media_shelf"},
                                ]
                            }
                        ],
                    }
                ],
            )

        self.assertEqual(trace["mode"], "active")
        self.assertTrue(trace["layout_traits_enabled"])
        self.assertFalse(trace["layout_traits_intrinsic_enabled"])
        self.assertEqual(trace["promotion_readiness"]["status"], "active")


if __name__ == "__main__":
    unittest.main()
