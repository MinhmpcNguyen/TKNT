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


def _decor_item(object_type: str) -> dict[str, object]:
    return {
        "cluster_id": "seating_zone",
        "object_type": object_type,
        "category": "decor",
        "rep_dims_m": {"L": 0.35, "W": 0.35},
    }


class DroppedInventoryArchitectureTest(unittest.TestCase):
    def test_dropped_inventory_payload_summarizes_solver_drops(self) -> None:
        payload = dropped_inventory.dropped_inventory_payload(
            {
                "media_zone": [_decor_item("plant")],
                "seating_zone": [_decor_item("vase"), _decor_item("wall_art")],
            }
        )

        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["total_dropped_items"], 3)
        self.assertEqual(
            [row["cluster_id"] for row in payload["clusters"]],
            ["media_zone", "seating_zone"],
        )

    def test_controlled_accessory_refill_adds_limited_decor_accessories(self) -> None:
        absolute_layout, summary = dropped_inventory.controlled_accessory_refill(
            room_output=_room_output(),
            absolute_layout=_absolute_layout(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={
                "seating_zone": [_decor_item("plant")],
                "media_zone": [
                    {
                        **_decor_item("vase"),
                        "cluster_id": "media_zone",
                    }
                ],
            },
            grid_mm=50,
        )

        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["module"], "controlled_accessory_refill")
        self.assertEqual(summary["refill_count"], 2)
        self.assertEqual(summary["remaining_count"], 0)
        self.assertEqual(len(absolute_layout["objects"]), 3)
        self.assertIn("seating_zone", summary["refill_objects_by_cluster"])
        self.assertIn("media_zone", summary["refill_objects_by_cluster"])
        self.assertIn("accessory_refill_objects_by_cluster", absolute_layout)

    def test_controlled_accessory_refill_respects_zero_refill_policy(self) -> None:
        absolute_layout, summary = dropped_inventory.controlled_accessory_refill(
            room_output=_room_output(),
            absolute_layout=_absolute_layout(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={"seating_zone": [_decor_item("plant")]},
            refill_policy={
                "max_refills_total": 0,
                "disabled_reason": "functional_request_contract_unmet",
                "missing_request_object_types": ["desk"],
            },
            grid_mm=50,
        )

        self.assertEqual(len(absolute_layout["objects"]), 1)
        self.assertEqual(summary["refill_count"], 0)
        self.assertEqual(summary["remaining_count"], 1)
        self.assertIn("functional_request_contract_unmet", " ".join(summary["notes"]))
        self.assertIn("desk", " ".join(summary["notes"]))

    def test_controlled_accessory_refill_keeps_doors_and_windows_clear(self) -> None:
        absolute_layout, summary = dropped_inventory.controlled_accessory_refill(
            room_output=_room_output_with_opening_guards(),
            absolute_layout=_absolute_layout(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={"seating_zone": [_decor_item("plant")]},
            grid_mm=50,
        )

        self.assertEqual(summary["refill_count"], 1)
        inserted = next(
            row for row in absolute_layout["objects"] if row.get("object_id") != "sofa"
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

        self.assertIsNotNone(polygon)
        self.assertIn(
            inserted.get("front_side_world"), {"top", "bottom", "left", "right"}
        )
        self.assertIsInstance(inserted.get("front_world"), dict)
        self.assertFalse(dropped_inventory._overlaps_blockers(polygon, blockers))

    def test_controlled_accessory_refill_rejects_functional_support(self) -> None:
        absolute_layout, summary = dropped_inventory.controlled_accessory_refill(
            room_output=_room_output(),
            absolute_layout=_absolute_layout(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={
                "seating_zone": [
                    {
                        "object_type": "ottoman",
                        "category": "furniture",
                        "rep_dims_m": {"L": 0.6, "W": 0.5},
                    }
                ]
            },
        )

        self.assertEqual(len(absolute_layout["objects"]), 1)
        self.assertEqual(summary["refill_count"], 0)
        self.assertEqual(
            summary["rejected_refill_objects_by_cluster"]["seating_zone"][0][
                "refill_rejected_reason"
            ],
            "functional_support_never_refilled",
        )

    def test_controlled_accessory_refill_rejects_cushion(self) -> None:
        absolute_layout, summary = dropped_inventory.controlled_accessory_refill(
            room_output=_room_output(),
            absolute_layout=_absolute_layout(),
            cluster_output=_cluster_output(),
            dropped_inventory_by_cluster={
                "seating_zone": [
                    {
                        "object_type": "cushion",
                        "category": "decor",
                        "rep_dims_m": {"L": 0.42, "W": 0.32},
                    }
                ]
            },
        )

        self.assertEqual(len(absolute_layout["objects"]), 1)
        self.assertEqual(summary["refill_count"], 0)
        self.assertEqual(
            summary["rejected_refill_objects_by_cluster"]["seating_zone"][0][
                "refill_rejected_reason"
            ],
            "accessory_refill_type_disabled",
        )


if __name__ == "__main__":
    unittest.main()
