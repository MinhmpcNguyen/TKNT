from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layout.orientation_contract import (  # noqa: E402
    effective_front_rotation,
    effective_front_side,
    rotate_side_ccw_90s,
    rotation_from_front_world,
)


class OrientationContractTest(unittest.TestCase):
    def test_effective_front_side_rotates_base_front_consistently(self) -> None:
        self.assertEqual(effective_front_side(base_front="bottom", rotation=0), "bottom")
        self.assertEqual(
            effective_front_side(base_front="bottom", rotation=180),
            "top",
        )
        self.assertEqual(
            effective_front_side(base_front="left", rotation=90),
            "bottom",
        )

    def test_effective_front_rotation_matches_world_facing_side(self) -> None:
        self.assertEqual(effective_front_rotation(rotation=0, front_side="top"), 0)
        self.assertEqual(effective_front_rotation(rotation=0, front_side="bottom"), 180)
        self.assertEqual(effective_front_rotation(rotation=90, front_side="bottom"), 90)

    def test_rotation_from_front_world_snaps_to_semantic_rotation(self) -> None:
        self.assertEqual(rotation_from_front_world({"dx": 0, "dy": -1}), 0)
        self.assertEqual(rotation_from_front_world({"dx": 1, "dy": 0}), 90)
        self.assertEqual(rotation_from_front_world({"dx": 0, "dy": 1}), 180)
        self.assertEqual(rotation_from_front_world({"dx": -1, "dy": 0}), 270)

    def test_rotate_side_ccw_90s_matches_vector_rotation(self) -> None:
        self.assertEqual(rotate_side_ccw_90s("bottom", 90), "right")
        self.assertEqual(rotate_side_ccw_90s("right", 90), "top")


if __name__ == "__main__":
    unittest.main()
