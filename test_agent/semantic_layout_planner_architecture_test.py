from __future__ import annotations

import unittest

from agent import semantic_layout_planner


class SemanticLayoutPlannerArchitectureTest(unittest.TestCase):
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

    def test_bedroom_profile_trace_is_shadow_only_for_legacy_layout(self) -> None:
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


if __name__ == "__main__":
    raise SystemExit(unittest.main())
