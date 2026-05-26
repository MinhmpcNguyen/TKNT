from __future__ import annotations

import unittest
from unittest.mock import patch

from agent import semantic_layout_planner


class SemanticLayoutPlannerArchitectureTest(unittest.TestCase):
    def test_request_contract_llm_failure_raises_without_heuristic_fallback(
        self,
    ) -> None:
        class FailingPlanner(semantic_layout_planner.SemanticLayoutPlanner):
            def _call_adaptive_stage(
                self,
                *args: object,
                **kwargs: object,
            ) -> dict[str, object]:
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            FailingPlanner()._attach_llm_request_contract_if_enabled(
                deterministic_program={
                    "active_clusters": [{"cluster_id": "work", "members": ["desk"]}]
                },
                room_type="bedroom",
                brief_text="Include a desk.",
                room_model_json={},
                inventory_catalog={},
                temperature=0.2,
                top_p=0.9,
                max_tokens=256,
                use_llm=True,
            )

    def test_adaptive_request_contract_has_response_schema(self) -> None:
        schema = semantic_layout_planner._adaptive_stage_response_schema(
            "adaptive_request_contract"
        )

        self.assertIsInstance(schema, dict)
        assert isinstance(schema, dict)
        self.assertEqual(schema["type"], "OBJECT")
        properties = schema["properties"]
        self.assertIsInstance(properties, dict)
        self.assertIn("objects", properties)
        self.assertIn("groups", properties)

    def test_adaptive_cluster_semantics_has_response_schema(self) -> None:
        schema = semantic_layout_planner._adaptive_stage_response_schema(
            "adaptive_cluster_semantics"
        )

        self.assertIsInstance(schema, dict)
        assert isinstance(schema, dict)
        self.assertEqual(schema["type"], "OBJECT")
        properties = schema["properties"]
        self.assertIsInstance(properties, dict)
        cluster_semantics = properties["cluster_semantics"]
        self.assertIsInstance(cluster_semantics, dict)
        self.assertEqual(cluster_semantics["type"], "ARRAY")

    def test_cluster_semantics_accepts_cluster_id_mapping_shape(self) -> None:
        deterministic_program = {
            "active_clusters": [
                {
                    "cluster_id": "sleep_cluster",
                    "priority": "core",
                    "required_bundles": [
                        {
                            "bundle_id": "sleep_main",
                            "objects": [{"object_type": "bed"}],
                        }
                    ],
                }
            ],
            "selection_constraints": {},
        }
        affordance_summary = {
            "usable_wall_regions": ["wall_north"],
            "entry_landing_zones": ["entry_zone"],
        }

        result = semantic_layout_planner._sanitize_cluster_semantics(
            {
                "sleep_cluster": {
                    "layout_role": "primary",
                    "semantic_role": "main sleep anchor",
                    "dominant_anchor_candidates": ["bed"],
                    "zone_claims": {
                        "preferred_regions": ["wall_north"],
                        "avoid_regions": ["entry_zone"],
                        "wall_affinity": "high",
                        "daylight_affinity": "neutral",
                        "privacy_affinity": "high",
                        "floating_allowed": False,
                    },
                    "relation_intents": [],
                    "degradation_ladder": ["drop support decor first"],
                }
            },
            deterministic_program=deterministic_program,
            affordance_summary=affordance_summary,
        )

        active_clusters = result["active_clusters"]
        self.assertEqual(len(active_clusters), 1)
        cluster = active_clusters[0]
        self.assertEqual(cluster["cluster_id"], "sleep_cluster")
        self.assertEqual(cluster["layout_role"], "primary")
        self.assertEqual(cluster["dominant_anchor_candidates"], ["bed"])
        self.assertEqual(cluster["zone_claims"]["preferred_regions"], ["wall_north"])

    def test_role_sanitizer_repairs_all_support_bedroom_clusters(self) -> None:
        program = semantic_layout_planner._normalize_profile_semantic_program(
            {
                "room_type": "bedroom",
                "active_clusters": [
                    {
                        "cluster_id": "sleep_core",
                        "layout_role": "support",
                        "priority": "core",
                        "semantic_role": "primary_anchor_zone",
                        "required_bundles": [
                            {
                                "bundle_id": "sleep_core_bundle",
                                "objects": [{"object_type": "bed"}],
                            }
                        ],
                    },
                    {
                        "cluster_id": "media_optional",
                        "layout_role": "support",
                        "priority": "support",
                        "semantic_role": "secondary_support_zone",
                        "required_bundles": [
                            {
                                "bundle_id": "media_bundle",
                                "objects": [{"object_type": "tv_console"}],
                            }
                        ],
                    },
                    {
                        "cluster_id": "storage_core",
                        "layout_role": "support",
                        "priority": "core",
                        "semantic_role": "storage_zone",
                        "required_bundles": [
                            {
                                "bundle_id": "storage_bundle",
                                "objects": [{"object_type": "wardrobe"}],
                            }
                        ],
                    },
                ],
            },
            affordance_summary={},
        )

        roles = {
            cluster["cluster_id"]: cluster["layout_role"]
            for cluster in program["active_clusters"]
        }
        self.assertEqual(roles["sleep_core"], "primary")
        self.assertEqual(roles["media_optional"], "secondary")
        self.assertEqual(roles["storage_core"], "support")
        trace = program["role_normalization"]
        self.assertTrue(trace["applied"])
        self.assertEqual(trace["primary_cluster_id"], "sleep_core")
        self.assertEqual(trace["secondary_cluster_id"], "media_optional")

    def test_bedroom_profile_trace_is_shadow_only_for_legacy_layout(self) -> None:
        with patch.dict(
            "os.environ",
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": "shadow"},
        ):
            program = semantic_layout_planner._build_deterministic_program(
                room_type="bedroom",
                room_rules={"room_type": "bedroom", "global_program": {}},
                room_model_json={},
                affordance_summary={
                    "privacy_regions": ["private_back_zone"],
                    "wall_anchor_candidates": ["quiet_wall_zone"],
                    "entry_landing_zones": ["entry_zone"],
                    "primary_circulation_corridors": ["center_lane"],
                    "center_openness_regions": ["room_center"],
                },
                candidates=[
                    {
                        "cluster_id": "sleep_core",
                        "priority": "core",
                        "active_by_rule": True,
                        "useful": True,
                        "objects": [
                            {
                                "object_type": "bed",
                                "role": "dominant_anchor",
                                "required": True,
                            },
                            {
                                "object_type": "nightstand",
                                "role": "secondary_support",
                                "required": False,
                            },
                        ],
                        "missing": [],
                        "semantic": {"dominant_anchor_candidates": ["bed"]},
                        "degradation_hints": {},
                        "tier_count_hints": {},
                        "viability": {"inventory_support": 1.0},
                    }
                ],
                brief_text="calm bedroom with a bed and nightstand",
                inventory_catalog={},
            )

        trace = program["profile_shadow_trace"]
        self.assertEqual(trace["profile_id"], "bedroom_legacy")
        self.assertEqual(trace["mode"], "shadow")
        self.assertEqual(trace["coverage_ratio"], 1.0)
        self.assertEqual(
            program["active_clusters"][0]["zone_claims"]["privacy_affinity"],
            "high",
        )

    def test_room_rule_plan_can_force_legacy_mode(self) -> None:
        with patch.dict(
            "os.environ",
            {"TKNT_ROOM_PROFILE_RULE_MODE": "legacy"},
        ):
            plan = semantic_layout_planner._room_rule_plan_for(
                room_type="living_room",
                semantic_program_rules=None,
            )

        self.assertEqual(plan.trace["selected_source"], "legacy")
        self.assertFalse(plan.trace["profile_first_active"])
        self.assertEqual(plan.trace["dry_run_diff"]["status"], "equivalent")

    def test_generate_attaches_profile_rule_trace(self) -> None:
        program = semantic_layout_planner.SemanticLayoutPlanner().generate(
            room_model_json={},
            room_type="living_room",
            brief_text="simple living room with sofa and tv console",
            inventory_catalog=None,
            use_llm=False,
        )

        self.assertIsNotNone(program.profile_rule_trace)
        assert program.profile_rule_trace is not None
        self.assertEqual(program.profile_rule_trace["selected_source"], "profile")
        self.assertTrue(program.profile_rule_trace["profile_first_active"])

    def test_generate_reports_active_living_trace_when_traits_canary_is_enabled(
        self,
    ) -> None:
        with patch.dict(
            "os.environ",
            {
                "TKNT_ROOM_PROFILE_TRAITS_MODE": "canary",
                "TKNT_ROOM_PROFILE_TRAIT_ROOMS": "living_room",
            },
        ):
            program = semantic_layout_planner.SemanticLayoutPlanner().generate(
                room_model_json={},
                room_type="living_room",
                brief_text="simple living room with sofa and tv console",
                inventory_catalog=None,
                use_llm=False,
            )

        self.assertIsNotNone(program.profile_shadow_trace)
        self.assertIsNotNone(program.profile_layout_trace)
        assert program.profile_shadow_trace is not None
        assert program.profile_layout_trace is not None
        self.assertEqual(program.profile_layout_trace, program.profile_shadow_trace)
        self.assertEqual(program.profile_shadow_trace["mode"], "active")
        self.assertTrue(program.profile_shadow_trace["layout_traits_enabled"])
        cluster_payload = (
            semantic_layout_planner.semantic_program_to_cluster_forge_payload(
                program.model_dump()
            )
        )
        cluster_tags = {
            row["cluster_id"]: row["tag"] for row in cluster_payload["clusters"]
        }
        self.assertEqual(cluster_tags["main_seating"], "living")
        self.assertEqual(cluster_tags["media"], "living")
        self.assertNotIn("kitchen", cluster_tags.values())

    def test_bedroom_canary_traits_keep_sleep_semantics(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TKNT_ROOM_PROFILE_TRAITS_MODE": "canary",
                "TKNT_ROOM_PROFILE_TRAIT_ROOMS": "bedroom",
            },
        ):
            program = semantic_layout_planner._build_deterministic_program(
                room_type="bedroom",
                room_rules={"room_type": "bedroom", "global_program": {}},
                room_model_json={},
                affordance_summary={
                    "privacy_regions": ["private_back_zone"],
                    "wall_anchor_candidates": ["quiet_wall_zone"],
                    "entry_landing_zones": ["entry_zone"],
                    "primary_circulation_corridors": ["center_lane"],
                    "center_openness_regions": ["room_center"],
                },
                candidates=[
                    {
                        "cluster_id": "sleep_core",
                        "priority": "core",
                        "active_by_rule": True,
                        "useful": True,
                        "objects": [
                            {
                                "object_type": "bed",
                                "role": "dominant_anchor",
                                "required": True,
                            },
                            {
                                "object_type": "nightstand",
                                "role": "secondary_support",
                                "required": False,
                            },
                        ],
                        "missing": [],
                        "semantic": {"dominant_anchor_candidates": ["bed"]},
                        "degradation_hints": {},
                        "tier_count_hints": {},
                        "viability": {"inventory_support": 1.0},
                    }
                ],
                brief_text="calm bedroom with a bed and nightstand",
                inventory_catalog={},
            )

        trace = program["profile_shadow_trace"]
        self.assertEqual(trace["mode"], "active")
        cluster = program["active_clusters"][0]
        self.assertEqual(cluster["semantic_role"], "primary_anchor_zone")
        self.assertEqual(cluster["zone_claims"]["privacy_affinity"], "high")
        self.assertFalse(
            any(
                intent.get("target_cluster") == "kitchen_workflow_core"
                for intent in cluster["relation_intents"]
            )
        )
        cluster_payload = (
            semantic_layout_planner.semantic_program_to_cluster_forge_payload(program)
        )
        self.assertEqual(cluster_payload["clusters"][0]["tag"], "sleep")
        self.assertEqual(
            semantic_layout_planner._cluster_tag(
                "work_study",
                ["desk", "chair"],
                room_type="bedroom",
            ),
            "work",
        )

    def test_unified_traits_keep_room_semantics_partitioned(self) -> None:
        with patch.dict(
            "os.environ",
            {"TKNT_ROOM_PROFILE_TRAITS_MODE": "unified"},
        ):
            living = semantic_layout_planner.SemanticLayoutPlanner().generate(
                room_model_json={},
                room_type="living_room",
                brief_text="simple living room with sofa and tv console",
                inventory_catalog=None,
                use_llm=False,
            )
            kitchen = semantic_layout_planner.SemanticLayoutPlanner().generate(
                room_model_json={
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
                },
                room_type="kitchen",
                brief_text="compact kitchen with dining table",
                inventory_catalog=None,
                use_llm=False,
            )

        living_payload = (
            semantic_layout_planner.semantic_program_to_cluster_forge_payload(
                living.model_dump()
            )
        )
        kitchen_payload = (
            semantic_layout_planner.semantic_program_to_cluster_forge_payload(
                kitchen.model_dump()
            )
        )
        living_tags = {
            row["cluster_id"]: row["tag"] for row in living_payload["clusters"]
        }
        kitchen_tags = {
            row["cluster_id"]: row["tag"] for row in kitchen_payload["clusters"]
        }
        kitchen_roles = {
            row["cluster_id"]: row["semantic_role"]
            for row in kitchen.model_dump()["active_clusters"]
        }

        self.assertIsNotNone(living.profile_shadow_trace)
        self.assertIsNotNone(kitchen.profile_shadow_trace)
        assert living.profile_shadow_trace is not None
        assert kitchen.profile_shadow_trace is not None
        self.assertEqual(living.profile_shadow_trace["mode"], "active")
        self.assertNotIn("kitchen", living_tags.values())
        self.assertEqual(living_tags["main_seating"], "living")
        self.assertEqual(living_tags["media"], "living")
        self.assertEqual(kitchen.profile_shadow_trace["mode"], "active")
        self.assertTrue(set(kitchen_tags.values()) <= {"kitchen"})
        self.assertEqual(
            kitchen_roles["kitchen_dining_core"],
            "kitchen_dining_zone",
        )


if __name__ == "__main__":
    raise SystemExit(unittest.main())
