from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

tools = importlib.import_module("cluster_placer.tools")


class ClusterPlacerArchitectureTest(unittest.TestCase):
    def test_normalize_relation_plan_drops_cross_cluster_non_anchor_object_pressure(
        self,
    ) -> None:
        cinfo_by_id = {
            "seating": {
                "cluster_id": "seating",
                "anchors": ["sectional_sofa"],
                "decisions": [
                    {"object_type": "sectional_sofa", "priority": "anchor"},
                    {"object_type": "armchair", "priority": "primary"},
                ],
            },
            "media": {
                "cluster_id": "media",
                "anchors": ["tv_console"],
                "decisions": [
                    {"object_type": "tv_console", "priority": "anchor"},
                ],
            },
        }
        relation_plan = {
            "object_orientations": [
                {
                    "cluster_id": "seating",
                    "object_id": "armchair",
                    "intents": ["face_object"],
                    "target_object_id": "tv_console",
                    "target_object_cluster_id": "media",
                },
                {
                    "cluster_id": "seating",
                    "object_id": "sectional_sofa",
                    "intents": ["face_object"],
                    "target_object_id": "tv_console",
                    "target_object_cluster_id": "media",
                },
            ]
        }

        normalized = tools._normalize_relation_plan(relation_plan, cinfo_by_id)

        self.assertEqual(
            normalized["object_orientations"],
            [
                {
                    "cluster_id": "seating",
                    "object_id": "sectional_sofa",
                    "intents": ["face_object"],
                    "target_object_id": "tv_console",
                    "target_object_cluster_id": "media",
                    "target_cluster_id": "media",
                    "priority": "medium",
                    "reason": "",
                }
            ],
        )

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

    def test_rescue_semantic_anchor_specs_expand_wall_and_center_search(self) -> None:
        specs = tools._rescue_semantic_anchor_specs_for_cluster(
            room_bbox=(0.0, 0.0, 6000.0, 4000.0),
            opening_ctx={
                "doors": [{"midpoint": (0.0, 2000.0)}],
                "windows": [{"midpoint": (3000.0, 4000.0)}],
            },
            width=1800.0,
            height=900.0,
            grid_mm=50,
            hints={
                "prefer_wall": True,
                "avoid_center": True,
                "prefer_center": False,
            },
        )

        kinds = {item[2] for item in specs}
        self.assertIn("rescue_left", kinds)
        self.assertIn("rescue_right", kinds)
        self.assertIn("rescue_center_col_top", kinds)
        self.assertIn("rescue_window_bottom", kinds)

    def test_rotation_anchor_viability_prefers_wall_backed_long_wall_alignment(
        self,
    ) -> None:
        cinfo = {
            "orientation_meta": {
                "cluster_front_local": {"dx": 0, "dy": -1},
                "cluster_axis_local": {"dx": 1, "dy": 0},
            }
        }
        hints = {
            "prefer_wall": True,
            "prefer_long_wall": True,
            "prefer_short_wall": False,
            "need_open_front": True,
        }

        wall_backed_score = tools._rotation_anchor_viability(
            cinfo=cinfo,
            rot=180,
            anchor_kind="refine_wall_top",
            hints=hints,
        )
        sideways_score = tools._rotation_anchor_viability(
            cinfo=cinfo,
            rot=90,
            anchor_kind="refine_wall_top",
            hints=hints,
        )
        open_front_wall_backed_score = tools._rotation_anchor_viability(
            cinfo=cinfo,
            rot=180,
            anchor_kind="open_front_top",
            hints=hints,
        )

        self.assertGreater(wall_backed_score, sideways_score)
        self.assertGreater(open_front_wall_backed_score, sideways_score)

    def test_back_to_wall_penalty_uses_back_clearance_not_nearest_centroid_wall(self) -> None:
        room_poly = Polygon([(0.0, 0.0), (6000.0, 0.0), (6000.0, 4000.0), (0.0, 4000.0)])
        wall_backed_penalty, _dot, wall_backed_debug = tools._penalty_back_to_wall(
            point=(300.0, 2000.0),
            front=(1.0, 0.0),
            room_poly=room_poly,
            blocker_geoms=[],
            weight=1.0,
        )
        sideways_penalty, _dot, sideways_debug = tools._penalty_back_to_wall(
            point=(300.0, 2000.0),
            front=(0.0, 1.0),
            room_poly=room_poly,
            blocker_geoms=[],
            weight=1.0,
        )

        self.assertLess(wall_backed_penalty, sideways_penalty)
        self.assertLess(
            wall_backed_debug["back_clear_mm"],
            wall_backed_debug["front_clear_mm"],
        )
        self.assertEqual(
            sideways_debug["front_advantage_mm"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
