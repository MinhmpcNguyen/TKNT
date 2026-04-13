from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

seed_variant_policy = importlib.import_module("pipeline.seed_variant_policy")


def _planner_clusters_json() -> dict[str, object]:
    return {
        "clusters": {
            "seating_zone": {
                "cluster_id": "seating_zone",
                "local_placements": [
                    {"id": "sectional_sofa"},
                    {"id": "coffee_table"},
                ],
                "seed_state": {"region_tags": ["near_center"]},
            },
            "media_zone": {
                "cluster_id": "media_zone",
                "local_placements": [{"id": "tv_console"}],
                "seed_state": {"region_tags": ["far_from_entry"]},
            },
            "utility_storage": {
                "cluster_id": "utility_storage",
                "local_placements": [{"id": "bookshelf"}],
                "seed_state": {"region_tags": ["entry_side", "edge_or_perimeter"]},
            },
        }
    }


def _base_relation_plan() -> dict[str, object]:
    return {
        "status": "OK",
        "layout_intent_profile": {
            "primary_cluster_id": "seating_zone",
            "secondary_cluster_id": "media_zone",
            "focus_mode": "viewing",
            "center_open_preference": "medium",
            "support_cluster_behavior": "balanced",
            "distribution_mode": "balanced",
        },
        "cluster_relations": [],
        "cluster_directional_relations": [],
        "cluster_affinities": [],
        "placement_guidelines": [],
        "circulation_plan": {"main_paths": [], "keep_open_regions": []},
        "notes": [],
    }


class SeedVariantPolicyArchitectureTest(unittest.TestCase):
    def test_build_seed_variant_briefs_returns_distinct_distribution_modes(
        self,
    ) -> None:
        briefs = seed_variant_policy.build_seed_variant_briefs(5)

        self.assertEqual(
            [brief.source for brief in briefs],
            [
                "seed_stable",
                "seed_focal_axis",
                "seed_open_center",
                "seed_edge_weighted",
                "seed_zoned",
            ],
        )
        self.assertEqual(
            [brief.distribution_mode for brief in briefs],
            [
                "balanced",
                "focal_grouped",
                "balanced",
                "edge_weighted",
                "zoned",
            ],
        )
        self.assertEqual(
            [brief.constraint_tokens for brief in briefs],
            [
                (
                    "seed_region_preservation",
                    "primary_center_anchor",
                    "balanced_support",
                ),
                (
                    "focal_axis_lock",
                    "primary_faces_secondary",
                    "secondary_wall_backed",
                ),
                (
                    "center_lane_lock",
                    "support_far_from_primary",
                    "support_edge_bias",
                ),
                (
                    "perimeter_bias_lock",
                    "all_clusters_avoid_center",
                    "wall_backed_primary_pair",
                ),
                (
                    "primary_zone_claim",
                    "support_zone_split",
                    "support_separate_from_secondary",
                ),
            ],
        )

    def test_open_center_policy_adds_center_lane_and_recedes_support_clusters(
        self,
    ) -> None:
        brief = seed_variant_policy.build_seed_variant_briefs(5)[2]

        plan = seed_variant_policy.apply_seed_variant_policy(
            relation_plan=_base_relation_plan(),
            planner_clusters_json=_planner_clusters_json(),
            brief=brief,
        )

        profile = plan["layout_intent_profile"]
        self.assertEqual(profile["center_open_preference"], "high")
        self.assertEqual(profile["support_cluster_behavior"], "recede")
        keep_open_regions = plan["circulation_plan"]["keep_open_regions"]
        self.assertTrue(
            any(
                row.get("type") == "center_lane" and row.get("near") == "room_center"
                for row in keep_open_regions
            )
        )
        support_affinity = next(
            row
            for row in plan["cluster_affinities"]
            if row.get("cluster_id") == "utility_storage"
        )
        self.assertIn("recess_or_edge", support_affinity["prefer"])
        self.assertIn("center", support_affinity["avoid"])

    def test_zoned_policy_separates_support_cluster_from_primary_pair(self) -> None:
        brief = seed_variant_policy.build_seed_variant_briefs(5)[4]

        plan = seed_variant_policy.apply_seed_variant_policy(
            relation_plan=_base_relation_plan(),
            planner_clusters_json=_planner_clusters_json(),
            brief=brief,
        )

        profile = plan["layout_intent_profile"]
        self.assertEqual(profile["distribution_mode"], "zoned")
        support_relations = [
            row
            for row in plan["cluster_relations"]
            if row.get("a") == "utility_storage" or row.get("b") == "utility_storage"
        ]
        self.assertTrue(
            any(row.get("relation") == "far_if_possible" for row in support_relations)
        )
        support_affinity = next(
            row
            for row in plan["cluster_affinities"]
            if row.get("cluster_id") == "utility_storage"
        )
        self.assertIn("entry_side", support_affinity["prefer"])

    def test_variant_policies_emit_distinct_structural_fingerprints(self) -> None:
        fingerprints: list[str] = []
        primary_affinity_seen = False
        focal_orientation_seen = False

        for brief in seed_variant_policy.build_seed_variant_briefs(5):
            plan = seed_variant_policy.apply_seed_variant_policy(
                relation_plan=_base_relation_plan(),
                planner_clusters_json=_planner_clusters_json(),
                brief=brief,
            )
            metadata = plan.get("seed_variant_policy")
            self.assertIsInstance(metadata, dict)
            structural_fingerprint = metadata.get("structural_fingerprint")
            self.assertIsInstance(structural_fingerprint, str)
            self.assertTrue(structural_fingerprint)
            fingerprints.append(structural_fingerprint)

            if brief.source == "seed_stable":
                primary_affinity = next(
                    row
                    for row in plan["cluster_affinities"]
                    if row.get("cluster_id") == "seating_zone"
                )
                primary_affinity_seen = "center" in primary_affinity["prefer"]
            if brief.source == "seed_focal_axis":
                focal_orientation = next(
                    row
                    for row in plan["cluster_orientations"]
                    if row.get("cluster_id") == "seating_zone"
                )
                focal_orientation_seen = (
                    "face_cluster" in focal_orientation["intents"]
                    and focal_orientation.get("target_cluster_id") == "media_zone"
                )

        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        self.assertTrue(primary_affinity_seen)
        self.assertTrue(focal_orientation_seen)


if __name__ == "__main__":
    unittest.main()
