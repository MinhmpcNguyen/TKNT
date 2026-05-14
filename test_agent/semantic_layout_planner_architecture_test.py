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


if __name__ == "__main__":
    raise SystemExit(unittest.main())
