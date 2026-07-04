from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.manual_placements import (  # noqa: E402
    apply_manual_placements_to_tier_output,
    augment_seed_relation_plan_with_manual_placements,
    build_manual_placements_guidance_text,
    merge_manual_placements_into_styled_output,
    prepare_input_payload_for_manual_placements,
)
from agent_schema.clusterF_schema import SeedRelationPlannerOutput  # noqa: E402


class ManualPlacementsTest(unittest.TestCase):
    def test_prepare_input_payload_adds_no_go_zone(self) -> None:
        payload = {
            "constraints": {},
            "manual_placements": [
                {
                    "placement_id": "desk_manual_1",
                    "asset_id": "desk_asset",
                    "category": "desk",
                    "position_mm": {"x": 1200, "y": 900},
                    "rotation_deg": 0,
                    "footprint_mm": {"w": 1000, "d": 600},
                    "height_mm": 780,
                    "color_hex": "#AABBCC",
                    "constraint_mode": "pinned",
                }
            ],
        }

        prepared, manual_placements = prepare_input_payload_for_manual_placements(
            payload
        )

        self.assertEqual(len(manual_placements), 1)
        no_go_zones = prepared["constraints"]["no_go_zones"]
        self.assertEqual(len(no_go_zones), 1)
        self.assertEqual(no_go_zones[0]["type"], "no_go")
        self.assertEqual(no_go_zones[0]["id"], "manual_desk_manual_1")
        self.assertTrue(no_go_zones[0]["hard"])

    def test_apply_manual_placements_reduces_matching_tier_decisions(self) -> None:
        tier_output = {
            "status": "OK",
            "decisions": [
                {
                    "object_type": "desk",
                    "category": "desk",
                    "cluster_id": "work",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "anchor",
                    "rationale": "Anchor desk kept.",
                },
                {
                    "object_type": "bed",
                    "category": "bed",
                    "cluster_id": "sleep",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "anchor",
                    "rationale": "Anchor bed kept.",
                },
            ],
            "global_notes": [],
        }
        manual_placements = [
            {
                "placement_id": "desk_manual_1",
                "asset_id": "desk_asset",
                "category": "desk",
                "position_mm": {"x": 1200, "y": 900},
                "rotation_deg": 0,
                "footprint_mm": {"w": 1000, "d": 600},
                "height_mm": 780,
                "color_hex": "#AABBCC",
                "anchor": "center",
                "source": "manual",
                "constraint_mode": "pinned",
            }
        ]

        adjusted = apply_manual_placements_to_tier_output(
            tier_output, manual_placements
        )

        self.assertEqual(adjusted["decisions"][0]["quantity"], 0)
        self.assertEqual(adjusted["decisions"][1]["quantity"], 1)
        self.assertTrue(
            any("desk x1" in note for note in adjusted["global_notes"]),
        )

    def test_apply_manual_placements_prefers_matching_cluster_when_cluster_match_exists(
        self,
    ) -> None:
        tier_output = {
            "status": "OK",
            "decisions": [
                {
                    "object_type": "chair",
                    "category": "chair",
                    "cluster_id": "living_area",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "secondary",
                    "rationale": "Reading seat.",
                },
                {
                    "object_type": "office_chair",
                    "category": "office_chair",
                    "cluster_id": "work_area",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "secondary",
                    "rationale": "Task chair.",
                },
            ],
            "global_notes": [],
        }
        cluster_output = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "living_area",
                    "tag": "living",
                    "members": ["armchair", "chair", "side_table"],
                    "anchors": ["armchair"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {"grid_mm": 50},
                },
                {
                    "cluster_id": "work_area",
                    "tag": "work",
                    "members": ["desk", "office_chair"],
                    "anchors": ["desk"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {"grid_mm": 50},
                },
            ],
            "notes": [],
            "missing": [],
        }
        manual_placements = [
            {
                "placement_id": "office_chair_manual_1",
                "asset_id": "office_chair_asset",
                "category": "office_chair",
                "position_mm": {"x": 1500, "y": 900},
                "rotation_deg": 0,
                "footprint_mm": {"w": 500, "d": 500},
                "height_mm": 900,
                "color_hex": "#AABBCC",
                "anchor": "center",
                "source": "manual",
                "constraint_mode": "pinned",
            }
        ]

        adjusted = apply_manual_placements_to_tier_output(
            tier_output,
            manual_placements,
            clusters_json=cluster_output,
        )

        self.assertEqual(adjusted["decisions"][0]["quantity"], 1)
        self.assertEqual(adjusted["decisions"][1]["quantity"], 0)

    def test_guidance_and_cluster_annotation_capture_manual_existing_objects(
        self,
    ) -> None:
        manual_placements = [
            {
                "placement_id": "desk_manual_1",
                "asset_id": "desk_asset",
                "category": "desk",
                "position_mm": {"x": 1200, "y": 900},
                "rotation_deg": 90,
                "footprint_mm": {"w": 1000, "d": 600},
                "height_mm": 780,
                "color_hex": "#AABBCC",
                "anchor": "center",
                "source": "manual",
                "constraint_mode": "locked",
            }
        ]
        cluster_output = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "work_area",
                    "tag": "work",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "cluster_rules": {"grid_mm": 50},
                    "notes": [],
                }
            ],
            "notes": [],
            "missing": [],
        }

        guidance = build_manual_placements_guidance_text(manual_placements)
        augmented = augment_seed_relation_plan_with_manual_placements(
            cluster_output,
            manual_placements,
        )
        parsed = SeedRelationPlannerOutput.model_validate(augmented)

        self.assertIn("EXISTING MANUAL OBJECTS FROM EDITOR", guidance)
        self.assertIn("desk", guidance)
        manual_existing_objects = (
            parsed.clusters[0].cluster_rules.model_extra or {}
        ).get("manual_existing_objects")
        self.assertIsInstance(manual_existing_objects, list)
        self.assertEqual(manual_existing_objects[0]["placement_id"], "desk_manual_1")
        self.assertTrue(
            any(
                "already present" in note
                for note in augmented["clusters"][0].get("notes", [])
            ),
        )

    def test_merge_manual_placements_into_styled_output_preserves_manual_input(
        self,
    ) -> None:
        styled_output = {
            "status": "OK",
            "room": {"room_id": "room_1", "room_type": "office", "polygon_ccw": []},
            "objects": [],
            "notes": [],
            "missing": [],
        }
        manual_placements = [
            {
                "placement_id": "desk_manual_1",
                "asset_id": "desk_asset",
                "category": "desk",
                "position_mm": {"x": 1200, "y": 900},
                "rotation_deg": 90,
                "footprint_mm": {"w": 1000, "d": 600},
                "height_mm": 780,
                "color_hex": "#AABBCC",
                "anchor": "center",
                "source": "manual",
                "constraint_mode": "locked",
            }
        ]

        merged = merge_manual_placements_into_styled_output(
            styled_output,
            manual_placements,
        )

        self.assertEqual(len(merged["objects"]), 1)
        obj = merged["objects"][0]
        self.assertEqual(obj["instance_id"], "desk_manual_1")
        self.assertEqual(obj["object_type"], "desk")
        self.assertEqual(obj["rotation_ccw"], 90)
        self.assertEqual(obj["manual_input"]["constraint_mode"], "locked")


if __name__ == "__main__":
    unittest.main()
