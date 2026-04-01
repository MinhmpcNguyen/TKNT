from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

tools = importlib.import_module("cluster_placer.tools")


class ClusterPlacerArchitectureTest(unittest.TestCase):
    def test_select_diverse_candidates_keeps_multiple_macro_regions(self) -> None:
        candidates = []
        for idx in range(6):
            candidates.append(
                {
                    "cluster_id": "work",
                    "variant_id": "base",
                    "variant_priority": 1.0,
                    "anchor_priority": 1.0 - 0.01 * idx,
                    "rough_score": 1000 - idx,
                    "critical_orientation_penalty_mm": 0,
                    "focal_orientation_penalty_mm": 0,
                    "orientation_penalty_mm": 0,
                    "variant_family": "base",
                    "anchor_kind": f"wall_right_{idx}",
                    "rot": 0,
                    "x": 4800 + 50 * idx,
                    "y": 300 + 50 * idx,
                }
            )

        candidates.append(
            {
                "cluster_id": "work",
                "variant_id": "base",
                "variant_priority": 0.8,
                "anchor_priority": 0.6,
                "rough_score": 700,
                "critical_orientation_penalty_mm": 0,
                "focal_orientation_penalty_mm": 0,
                "orientation_penalty_mm": 0,
                "variant_family": "base",
                "anchor_kind": "wall_left_alt",
                "rot": 0,
                "x": 300,
                "y": 1200,
            }
        )

        selected = tools._select_diverse_candidates_for_search(
            candidates,
            room_bbox=(0.0, 0.0, 6000.0, 3000.0),
            limit=4,
        )

        region_buckets = {
            tools._candidate_region_bucket(item, room_bbox=(0.0, 0.0, 6000.0, 3000.0))
            for item in selected
        }
        self.assertIn(("right", "top"), region_buckets)
        self.assertIn(("left", "middle"), region_buckets)


if __name__ == "__main__":
    unittest.main()
