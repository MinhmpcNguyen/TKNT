from __future__ import annotations

import unittest

from pipeline.layout_eligibility import (
    annotate_layout_coverage,
    summarize_layout_coverage,
)


class LayoutEligibilityTest(unittest.TestCase):
    def test_summarize_layout_coverage_marks_missing_primary_cluster(self) -> None:
        merged_output = {
            "clusters": [
                {
                    "cluster_id": "seating_hub",
                    "decisions": [{"object_type": "sofa", "quantity": 1}],
                },
                {
                    "cluster_id": "media_storage",
                    "decisions": [{"object_type": "tv_console", "quantity": 1}],
                },
            ]
        }
        relation_plan = {"layout_intent_profile": {"primary_cluster_id": "seating_hub"}}
        absolute_layout = {
            "status": "OK",
            "objects": [
                {"cluster_id": "media_storage"},
            ],
            "missing": [],
        }

        summary = summarize_layout_coverage(
            absolute_layout=absolute_layout,
            merged_output=merged_output,
            relation_plan=relation_plan,
            cluster_outlines=None,
            hard_valid=True,
        )

        self.assertEqual(summary["missing_cluster_ids"], ["seating_hub"])
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["gallery_eligible"])
        self.assertFalse(summary["primary_cluster_present"])
        self.assertAlmostEqual(summary["coverage_ratio"], 0.5)

    def test_annotate_layout_coverage_appends_missing_cluster_issue(self) -> None:
        merged_output = {
            "clusters": [
                {
                    "cluster_id": "seating_hub",
                    "decisions": [{"object_type": "sofa", "quantity": 1}],
                },
                {
                    "cluster_id": "media_storage",
                    "decisions": [{"object_type": "tv_console", "quantity": 1}],
                },
            ]
        }
        relation_plan = {"layout_intent_profile": {"primary_cluster_id": "seating_hub"}}
        absolute_layout = {
            "status": "OK",
            "objects": [
                {"cluster_id": "media_storage"},
            ],
            "missing": [],
        }

        payload = annotate_layout_coverage(
            absolute_layout=absolute_layout,
            merged_output=merged_output,
            relation_plan=relation_plan,
            cluster_outlines=None,
            hard_valid=True,
        )

        self.assertFalse(payload["gallery_eligible"])
        self.assertEqual(payload["missing_cluster_ids"], ["seating_hub"])
        self.assertTrue(
            any(
                isinstance(item, dict)
                and item.get("code") == "MISSING_CLUSTER"
                and item.get("cluster_id") == "seating_hub"
                for item in payload["missing"]
            )
        )

    def test_unacceptable_layout_is_not_gallery_eligible(self) -> None:
        merged_output = {
            "clusters": [
                {
                    "cluster_id": "seating_hub",
                    "decisions": [{"object_type": "sofa", "quantity": 1}],
                }
            ]
        }
        absolute_layout = {
            "status": "OK",
            "objects": [{"cluster_id": "seating_hub"}],
            "missing": [],
        }

        summary = summarize_layout_coverage(
            absolute_layout=absolute_layout,
            merged_output=merged_output,
            relation_plan=None,
            cluster_outlines=None,
            hard_valid=True,
            acceptable_valid=False,
        )
        payload = annotate_layout_coverage(
            absolute_layout=absolute_layout,
            merged_output=merged_output,
            relation_plan=None,
            cluster_outlines=None,
            hard_valid=True,
            acceptable_valid=False,
        )

        self.assertTrue(summary["complete"])
        self.assertFalse(summary["acceptable_valid"])
        self.assertFalse(summary["gallery_eligible"])
        self.assertFalse(payload["acceptable_valid"])
        self.assertFalse(payload["gallery_eligible"])


if __name__ == "__main__":
    unittest.main()
