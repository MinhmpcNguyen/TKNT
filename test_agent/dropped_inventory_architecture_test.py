from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

dropped_inventory = importlib.import_module("pipeline.dropped_inventory")


def _room_output() -> dict[str, object]:
    return {
        "room": {
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 5000, "y": 0},
                {"x": 5000, "y": 4000},
                {"x": 0, "y": 4000},
            ]
        }
    }


def _room_output_with_opening_guards() -> dict[str, object]:
    return {
        "room": {
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 5000, "y": 0},
                {"x": 5000, "y": 4000},
                {"x": 0, "y": 4000},
            ]
        },
        "openings": {
            "doors": [
                {
                    "id": "door_1",
                    "segment_mm": [
                        {"x": 0, "y": 1200},
                        {"x": 0, "y": 2600},
                    ],
                    "swing_radius_mm": 1000,
                }
            ],
            "windows": [
                {
                    "id": "window_1",
                    "segment_mm": [
                        {"x": 1600, "y": 0},
                        {"x": 2600, "y": 0},
                    ],
                    "clearance_mm": 1000,
                }
            ],
        },
    }


def _cluster_output() -> dict[str, object]:
    return {
        "clusters": [
            {
                "cluster_id": "seating_zone",
                "cluster_rules": {
                    "allowed_rotations": {"ottoman": [0, 90, 180, 270]},
                    "facing": {"ottoman": {"front": "bottom"}},
                },
            },
            {
                "cluster_id": "media_zone",
                "cluster_rules": {
                    "allowed_rotations": {"media_shelf": [0, 90, 180, 270]},
                },
            },
        ]
    }


def _absolute_layout() -> dict[str, object]:
    return {
        "status": "OK",
        "objects": [
            {
                "object_id": "sofa",
                "cluster_id": "seating_zone",
                "rotation_ccw": 0,
                "bbox": {"min_x": 1200, "min_y": 1400, "max_x": 3000, "max_y": 2200},
                "polygon_ccw": [
                    {"x": 1200, "y": 1400},
                    {"x": 3000, "y": 1400},
                    {"x": 3000, "y": 2200},
                    {"x": 1200, "y": 2200},
                ],
                "source_rect": {"x": 0, "y": 0, "w": 1800, "h": 800},
            }
        ],
    }


def _absolute_layout_with_perimeter_blockers() -> dict[str, object]:
    return {
        "status": "OK",
        "objects": [
            {
                "object_id": "sofa",
                "cluster_id": "seating_zone",
                "rotation_ccw": 0,
                "bbox": {"min_x": 1800, "min_y": 1600, "max_x": 3200, "max_y": 2400},
                "polygon_ccw": [
                    {"x": 1800, "y": 1600},
                    {"x": 3200, "y": 1600},
                    {"x": 3200, "y": 2400},
                    {"x": 1800, "y": 2400},
                ],
                "source_rect": {"x": 0, "y": 0, "w": 1400, "h": 800},
            },
            {
                "object_id": "north_blocker",
                "cluster_id": "blockers",
                "rotation_ccw": 0,
                "bbox": {"min_x": 0, "min_y": 0, "max_x": 5000, "max_y": 800},
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 5000, "y": 0},
                    {"x": 5000, "y": 800},
                    {"x": 0, "y": 800},
                ],
            },
            {
                "object_id": "west_blocker",
                "cluster_id": "blockers",
                "rotation_ccw": 0,
                "bbox": {"min_x": 0, "min_y": 800, "max_x": 800, "max_y": 4000},
                "polygon_ccw": [
                    {"x": 0, "y": 800},
                    {"x": 800, "y": 800},
                    {"x": 800, "y": 4000},
                    {"x": 0, "y": 4000},
                ],
            },
            {
                "object_id": "east_blocker",
                "cluster_id": "blockers",
                "rotation_ccw": 0,
                "bbox": {"min_x": 4200, "min_y": 800, "max_x": 5000, "max_y": 4000},
                "polygon_ccw": [
                    {"x": 4200, "y": 800},
                    {"x": 5000, "y": 800},
                    {"x": 5000, "y": 4000},
                    {"x": 4200, "y": 4000},
                ],
            },
            {
                "object_id": "south_blocker",
                "cluster_id": "blockers",
                "rotation_ccw": 0,
                "bbox": {"min_x": 800, "min_y": 3200, "max_x": 4200, "max_y": 4000},
                "polygon_ccw": [
                    {"x": 800, "y": 3200},
                    {"x": 4200, "y": 3200},
                    {"x": 4200, "y": 4000},
                    {"x": 800, "y": 4000},
                ],
            },
        ],
    }


class DroppedInventoryArchitectureTest(unittest.TestCase):
    def test_collect_seed_omitted_inventory_tracks_zero_quantity_rows(self) -> None:
        tier_output = {
            "decisions": [
                {
                    "object_type": "sofa",
                    "cluster_id": "seating_zone",
                    "quantity": 0,
                    "size_tier": "S",
                    "priority": "anchor",
                    "rep_dims_m": {"L": 1.79, "W": 0.81},
                    "rationale": "Primary seed kept one representative anchor.",
                },
                {
                    "object_type": "media_shelf",
                    "cluster_id": "media_zone",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "secondary",
                    "rep_dims_m": {"L": 0.9, "W": 0.3},
                },
            ]
        }

        omitted = dropped_inventory.collect_seed_omitted_inventory(
            tier_output=tier_output
        )

        self.assertEqual(sorted(omitted.keys()), ["seating_zone"])
        self.assertEqual(omitted["seating_zone"][0]["object_type"], "sofa")
        self.assertEqual(omitted["seating_zone"][0]["drop_reason"], "seed_omission")

    def test_collect_dropped_inventory_groups_removed_items_by_cluster(self) -> None:
        previous_tier_output = {
            "decisions": [
                {
                    "object_type": "ottoman",
                    "cluster_id": "seating_zone",
                    "quantity": 2,
                    "size_tier": "S",
                    "priority": "secondary",
                    "rep_dims_m": {"L": 0.6, "W": 0.5},
                },
                {
                    "object_type": "media_shelf",
                    "cluster_id": "media_zone",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "secondary",
                    "rep_dims_m": {"L": 0.9, "W": 0.3},
                },
            ]
        }
        next_tier_output = {
            "decisions": [
                {
                    "object_type": "ottoman",
                    "cluster_id": "seating_zone",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "secondary",
                    "rep_dims_m": {"L": 0.6, "W": 0.5},
                },
                {
                    "object_type": "media_shelf",
                    "cluster_id": "media_zone",
                    "quantity": 0,
                    "size_tier": "S",
                    "priority": "secondary",
                    "rep_dims_m": {"L": 0.9, "W": 0.3},
                },
            ]
        }

        removed = dropped_inventory.collect_dropped_inventory(
            previous_tier_output=previous_tier_output,
            next_tier_output=next_tier_output,
            attempt=2,
        )

        self.assertEqual(sorted(removed.keys()), ["media_zone", "seating_zone"])
        self.assertEqual(len(removed["seating_zone"]), 1)
        self.assertEqual(len(removed["media_zone"]), 1)
        self.assertEqual(removed["media_zone"][0]["drop_attempt"], 2)

    def test_reintroduce_dropped_inventory_uses_round_robin_across_clusters(
        self,
    ) -> None:
        absolute_layout, summary = dropped_inventory.reintroduce_dropped_inventory(
            room_output=_room_output(),
            absolute_layout=_absolute_layout(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={
                "seating_zone": [
                    {
                        "cluster_id": "seating_zone",
                        "object_type": "ottoman",
                        "size_tier": "S",
                        "priority": "secondary",
                        "rep_dims_m": {"L": 0.6, "W": 0.5},
                    }
                ],
                "media_zone": [
                    {
                        "cluster_id": "media_zone",
                        "object_type": "media_shelf",
                        "size_tier": "S",
                        "priority": "secondary",
                        "rep_dims_m": {"L": 0.9, "W": 0.3},
                    }
                ],
            },
            grid_mm=50,
        )

        self.assertEqual(summary["inserted_count"], 2)
        self.assertEqual(summary["remaining_count"], 0)
        self.assertEqual(len(absolute_layout["objects"]), 3)
        self.assertIn("seating_zone", summary["reintroduced_objects_by_cluster"])
        self.assertIn("media_zone", summary["reintroduced_objects_by_cluster"])

    def test_reintroduced_object_prefers_wall_backed_perimeter_placement(self) -> None:
        absolute_layout, summary = dropped_inventory.reintroduce_dropped_inventory(
            room_output=_room_output(),
            absolute_layout=_absolute_layout(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={
                "seating_zone": [
                    {
                        "cluster_id": "seating_zone",
                        "object_type": "ottoman",
                        "size_tier": "S",
                        "priority": "secondary",
                        "rep_dims_m": {"L": 0.6, "W": 0.5},
                    }
                ]
            },
            grid_mm=50,
        )

        self.assertEqual(summary["inserted_count"], 1)
        inserted = next(
            row for row in absolute_layout["objects"] if row.get("object_id") != "sofa"
        )
        room_polygon = dropped_inventory._room_polygon(_room_output())
        polygon = dropped_inventory._polygon_from_rows(inserted.get("polygon_ccw"))

        self.assertIsNotNone(room_polygon)
        self.assertIsNotNone(polygon)
        self.assertTrue(
            inserted["bbox"]["min_x"] <= 150
            or inserted["bbox"]["min_y"] <= 150
            or (5000 - inserted["bbox"]["max_x"]) <= 150
            or (4000 - inserted["bbox"]["max_y"]) <= 150
        )
        self.assertLessEqual(
            dropped_inventory._back_edge_wall_distance(
                room_polygon=room_polygon,
                polygon=polygon,
                rotation=dropped_inventory._effective_front_rotation(
                    rotation=int(inserted["rotation_ccw"]),
                    front_side="bottom",
                ),
            ),
            120.0,
        )

    def test_reintroduced_object_is_skipped_when_only_center_gap_remains(self) -> None:
        absolute_layout, summary = dropped_inventory.reintroduce_dropped_inventory(
            room_output=_room_output(),
            absolute_layout=_absolute_layout_with_perimeter_blockers(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={
                "seating_zone": [
                    {
                        "cluster_id": "seating_zone",
                        "object_type": "ottoman",
                        "size_tier": "S",
                        "priority": "secondary",
                        "rep_dims_m": {"L": 0.6, "W": 0.5},
                    }
                ]
            },
            grid_mm=50,
        )

        self.assertEqual(summary["inserted_count"], 0)
        self.assertEqual(summary["remaining_count"], 1)
        self.assertEqual(len(absolute_layout["objects"]), 5)

    def test_reintroduce_dropped_inventory_keeps_doors_and_windows_clear(self) -> None:
        absolute_layout, summary = dropped_inventory.reintroduce_dropped_inventory(
            room_output=_room_output_with_opening_guards(),
            absolute_layout=_absolute_layout(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={
                "seating_zone": [
                    {
                        "cluster_id": "seating_zone",
                        "object_type": "ottoman",
                        "size_tier": "S",
                        "priority": "secondary",
                        "rep_dims_m": {"L": 0.6, "W": 0.5},
                    }
                ]
            },
            grid_mm=50,
        )

        self.assertEqual(summary["inserted_count"], 1)
        inserted = next(
            row for row in absolute_layout["objects"] if row.get("object_id") != "sofa"
        )
        room_polygon = dropped_inventory._room_polygon(
            _room_output_with_opening_guards()
        )
        polygon = dropped_inventory._polygon_from_rows(inserted.get("polygon_ccw"))
        blockers = dropped_inventory._blocked_polygons(
            _room_output_with_opening_guards(),
            [
                row
                for row in absolute_layout["objects"]
                if row.get("object_id") != inserted["object_id"]
            ],
        )

        self.assertIsNotNone(room_polygon)
        self.assertIsNotNone(polygon)
        self.assertTrue(
            inserted["bbox"]["min_x"] <= 150
            or inserted["bbox"]["min_y"] <= 150
            or (5000 - inserted["bbox"]["max_x"]) <= 150
            or (4000 - inserted["bbox"]["max_y"]) <= 150
        )
        self.assertLessEqual(
            dropped_inventory._back_edge_wall_distance(
                room_polygon=room_polygon,
                polygon=polygon,
                rotation=dropped_inventory._effective_front_rotation(
                    rotation=int(inserted["rotation_ccw"]),
                    front_side="bottom",
                ),
            ),
            120.0,
        )
        self.assertFalse(dropped_inventory._overlaps_blockers(polygon, blockers))


if __name__ == "__main__":
    unittest.main()
