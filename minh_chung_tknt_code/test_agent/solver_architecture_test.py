from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.solver import solver as _solver  # noqa: E402


class SolverArchitectureTest(unittest.TestCase):
    def test_macro_cluster_solver_public_surface_is_object_level_only(self) -> None:
        solver = _solver.MacroClusterSolver(max_rounds=2, num_workers=4)

        self.assertTrue(hasattr(solver, "generate_object_layout"))
        self.assertFalse(hasattr(solver, "generate"))
        self.assertFalse(hasattr(solver, "generate_bundle"))
        self.assertEqual(solver.max_rounds, 2)
        self.assertEqual(solver.num_workers, 4)

    def test_softened_primary_corridor_protected_region_is_cluster_scoped(self) -> None:
        protected_regions = [
            {
                "region_id": "door_1_to_room_center_corridor",
                "bbox": (0, 0, 1000, 1000),
                "max_overlap_ratio": 0.0,
                "priority": "high",
                "enforcement": "hard_soft",
                "violation_severity": "blocking",
                "zone_type": "primary_circulation_corridor",
                "softened_for_clusters": ["sleep_core"],
            }
        ]

        issues = _solver._object_level_protected_region_issues(
            placed_objects=[
                {
                    "cluster_id": "sleep_core",
                    "object_id": "bed",
                    "rect": (0, 0, 500, 500),
                },
                {
                    "cluster_id": "work_study",
                    "object_id": "desk",
                    "rect": (0, 0, 500, 500),
                },
            ],
            protected_regions=protected_regions,
        )

        blocking_by_cluster = {
            issue["cluster_id"]: issue["blocking"] for issue in issues
        }
        self.assertFalse(blocking_by_cluster["sleep_core"])
        self.assertTrue(blocking_by_cluster["work_study"])

    def test_object_pre_repair_rank_key_penalizes_dropped_inventory(self) -> None:
        clean = {
            "pre_repair_score": 100,
            "anchor_score": 60,
            "support_score": 40,
            "dropped_inventory_by_cluster": {},
        }
        dropped = {
            **clean,
            "dropped_inventory_by_cluster": {"support": [{"object_id": "lamp"}]},
        }

        self.assertGreater(
            _solver._object_pre_repair_rank_key(clean),
            _solver._object_pre_repair_rank_key(dropped),
        )


if __name__ == "__main__":
    unittest.main()
