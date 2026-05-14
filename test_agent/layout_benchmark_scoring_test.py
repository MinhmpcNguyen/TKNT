from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.layout_benchmark.schema import LayoutItem, Point2D
from evaluation.layout_benchmark.normalize import normalize_tknt_payload
from evaluation.layout_benchmark.scoring import (
    _has_non_blocking_placement,
    _is_blocking,
    canonical_object_type,
)


class LayoutBenchmarkScoringTest(unittest.TestCase):
    def test_desk_lamp_is_not_counted_as_desk(self) -> None:
        self.assertEqual(canonical_object_type("desk_lamp"), "table_lamp")
        self.assertEqual(canonical_object_type("desk_lamp_001"), "table_lamp")

    def test_desk_variants_still_match_desk(self) -> None:
        self.assertEqual(canonical_object_type("compact_desk"), "desk")
        self.assertEqual(canonical_object_type("study-table"), "desk")

    def test_numbered_generic_lamps_are_table_lamps(self) -> None:
        self.assertEqual(canonical_object_type("lamp5"), "table_lamp")
        self.assertEqual(canonical_object_type("lamp6"), "table_lamp")
        self.assertEqual(canonical_object_type("floor lamp5"), "floor_lamp")

    def test_table_lamps_are_non_blocking_but_floor_lamps_are_blocking(self) -> None:
        self.assertFalse(_is_blocking("table_lamp"))
        self.assertTrue(_is_blocking("floor_lamp"))

    def test_on_top_accessories_are_non_blocking_placements(self) -> None:
        item = LayoutItem(
            object_id="vase_001",
            object_type="vase",
            polygon=(
                Point2D(0, 0),
                Point2D(1, 0),
                Point2D(1, 1),
                Point2D(0, 1),
            ),
            rotation_degrees=0.0,
            raw={
                "place_on": {
                    "target_instance_id": "coffee_table",
                    "method": "on_top",
                }
            },
        )
        self.assertTrue(_has_non_blocking_placement(item))

    def test_benchmark_empty_openings_override_tknt_default_openings(self) -> None:
        benchmark_case = {
            "id": "empty_openings_case",
            "room": {
                "polygon": [
                    {"x": 0, "y": 0},
                    {"x": 3000, "y": 0},
                    {"x": 3000, "y": 3000},
                    {"x": 0, "y": 3000},
                ],
                "openings": {"doors": [], "windows": []},
            },
        }
        payload = {
            "final_output": {
                "room": {
                    "polygon": benchmark_case["room"]["polygon"],
                    "openings": {
                        "doors": [
                            {
                                "id": "door_1",
                                "segment_mm": [
                                    {"x": 0, "y": 1000},
                                    {"x": 0, "y": 1800},
                                ],
                            }
                        ],
                        "windows": [],
                    },
                },
                "objects": [],
            }
        }

        normalized = normalize_tknt_payload(
            payload=payload,
            benchmark_case=benchmark_case,
            system_id="tknt",
        )

        self.assertEqual(normalized.openings, ())


if __name__ == "__main__":
    unittest.main()
