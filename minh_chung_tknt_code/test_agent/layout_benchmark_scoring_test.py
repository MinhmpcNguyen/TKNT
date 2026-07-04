from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.layout_benchmark.normalize import normalize_tknt_payload  # noqa: E402
from evaluation.layout_benchmark.schema import (  # noqa: E402
    LayoutItem,
    NormalizedLayout,
    Point2D,
    normalized_layout_to_json,
)
from evaluation.layout_benchmark.scoring import (  # noqa: E402
    _has_non_blocking_placement,
    _is_blocking,
    canonical_object_type,
    score_layout,
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

    def test_tknt_normalization_preserves_orientation_metadata(self) -> None:
        normalized = normalize_tknt_payload(
            payload={
                "final_output": {
                    "room": {
                        "polygon_ccw": [
                            {"x": 0, "y": 0},
                            {"x": 3000, "y": 0},
                            {"x": 3000, "y": 3000},
                            {"x": 0, "y": 3000},
                        ]
                    },
                    "objects": [
                        {
                            "instance_id": "bed",
                            "object_type": "bed",
                            "polygon_ccw": [
                                {"x": 0, "y": 0},
                                {"x": 1200, "y": 0},
                                {"x": 1200, "y": 2000},
                                {"x": 0, "y": 2000},
                            ],
                            "rotation_ccw": 270,
                            "front_world": {"dx": -1, "dy": 0},
                            "front_side_world": "left",
                            "axis_world": {"dx": 0, "dy": 1},
                        }
                    ],
                }
            },
            benchmark_case={"case_id": "orientation_case"},
            system_id="tknt",
        )

        payload = normalized_layout_to_json(normalized)
        bed = payload["objects"][0]
        self.assertEqual(bed["rotation_degrees"], 270.0)
        self.assertEqual(bed["front_world"], {"dx": -1, "dy": 0})
        self.assertEqual(bed["front_side_world"], "left")
        self.assertEqual(bed["axis_world"], {"dx": 0, "dy": 1})

    def test_simple_simple_case_scores_solid_count_instead_of_exact_type_ranges(
        self,
    ) -> None:
        layout = _layout_with_objects(
            case_id="simple_simple_bedroom_01_3m0x4m0",
            objects=[
                ("bed", "bed", 100, 100, 1300, 2100),
                ("wardrobe", "wardrobe", 1500, 100, 2300, 700),
            ],
        )
        score = score_layout(
            benchmark_case={
                "case_id": "simple_simple_bedroom_01_3m0x4m0",
                "room_type": "bedroom",
                "expected": {
                    "valid_blocking_count_min": 2,
                    "valid_blocking_count_max": 3,
                    "count_ranges": {
                        "bed": {"min": 1, "max": 1},
                        "desk": {"min": 1, "max": 1},
                        "chair": {"min": 1, "max": 1},
                    },
                },
            },
            layout=layout,
        )

        self.assertEqual(score["component_scores"]["request_following"], 100.0)
        self.assertEqual(
            [row["type"] for row in score["request_checks"]],
            ["solid_count_min", "room_primary_anchor"],
        )

    def test_valid_blocking_count_ignores_legacy_maximum(self) -> None:
        layout = _layout_with_objects(
            case_id="simple_simple_bedroom_01_3m0x4m0",
            objects=[
                ("bed", "bed", 100, 100, 1300, 2100),
                ("wardrobe", "wardrobe", 1500, 100, 2300, 700),
                ("desk", "desk", 100, 2300, 900, 3000),
                ("chair", "chair", 1000, 2300, 1400, 2800),
            ],
        )
        score = score_layout(
            benchmark_case={
                "case_id": "simple_simple_bedroom_01_3m0x4m0",
                "room_type": "bedroom",
                "expected": {
                    "valid_blocking_count_min": 2,
                    "valid_blocking_count_max": 3,
                },
            },
            layout=layout,
        )

        self.assertEqual(score["component_scores"]["request_following"], 100.0)
        self.assertEqual(score["component_scores"]["valid_object_count"], 100.0)
        self.assertEqual(score["request_checks"][0]["type"], "solid_count_min")

    def test_simple_simple_case_still_requires_primary_room_anchor(self) -> None:
        layout = _layout_with_objects(
            case_id="simple_simple_bedroom_01_3m0x4m0",
            objects=[
                ("wardrobe", "wardrobe", 100, 100, 900, 700),
                ("desk", "desk", 1200, 100, 2200, 700),
            ],
        )
        score = score_layout(
            benchmark_case={
                "case_id": "simple_simple_bedroom_01_3m0x4m0",
                "room_type": "bedroom",
                "expected": {
                    "valid_blocking_count_min": 2,
                    "valid_blocking_count_max": 3,
                    "count_ranges": {
                        "bed": {"min": 1, "max": 1},
                    },
                },
            },
            layout=layout,
        )

        self.assertEqual(score["component_scores"]["request_following"], 50.0)
        anchor_check = score["request_checks"][1]
        self.assertEqual(anchor_check["type"], "room_primary_anchor")
        self.assertFalse(anchor_check["passed"])

    def test_non_simple_simple_case_keeps_exact_type_count_ranges(self) -> None:
        layout = _layout_with_objects(
            case_id="simple_bedroom_01_rest_study",
            objects=[
                ("bed", "bed", 100, 100, 1300, 2100),
                ("wardrobe", "wardrobe", 1500, 100, 2300, 700),
            ],
        )
        score = score_layout(
            benchmark_case={
                "case_id": "simple_bedroom_01_rest_study",
                "room_type": "bedroom",
                "expected": {
                    "count_ranges": {
                        "bed": {"min": 1, "max": 1},
                        "desk": {"min": 1, "max": 1},
                    },
                },
            },
            layout=layout,
        )

        self.assertEqual(score["component_scores"]["request_following"], 50.0)
        self.assertEqual(
            [row["type"] for row in score["request_checks"]],
            ["count_range", "count_range"],
        )


def _layout_with_objects(
    *,
    case_id: str,
    objects: list[tuple[str, str, int, int, int, int]],
) -> NormalizedLayout:
    return NormalizedLayout(
        case_id=case_id,
        system_id="tknt",
        room_polygon=(
            Point2D(0, 0),
            Point2D(4000, 0),
            Point2D(4000, 4000),
            Point2D(0, 4000),
        ),
        openings=(),
        objects=tuple(
            LayoutItem(
                object_id=object_id,
                object_type=object_type,
                polygon=_box(min_x, min_y, max_x, max_y),
            )
            for object_id, object_type, min_x, min_y, max_x, max_y in objects
        ),
    )


def _box(
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
) -> tuple[Point2D, ...]:
    return (
        Point2D(min_x, min_y),
        Point2D(max_x, min_y),
        Point2D(max_x, max_y),
        Point2D(min_x, max_y),
    )


if __name__ == "__main__":
    unittest.main()
