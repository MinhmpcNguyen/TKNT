from __future__ import annotations

import sys
import unittest
import importlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

merge_cluster_outputs = importlib.import_module(
    "cluster_composer.merge"
).merge_cluster_outputs


def _decision(
    object_type: str,
    *,
    cluster_id: str = "sleep_core",
    quantity: int = 1,
    role: str = "support",
    preserve_level: str = "medium",
    min_keep: int = 0,
    protected: bool = False,
    droppable: bool = False,
) -> dict[str, Any]:
    return {
        "object_type": object_type,
        "category": object_type,
        "cluster_id": cluster_id,
        "quantity": quantity,
        "role": role,
        "priority": "anchor" if role == "dominant_anchor" else "primary",
        "preserve_level": preserve_level,
        "min_keep": min_keep,
        "protected": protected,
        "droppable": droppable,
        "size_tier": "M",
        "rep_dims_m": {"L": 0.5, "W": 0.45, "H": 0.55, "source_id": object_type},
    }


class ClusterMergeArchitectureTest(unittest.TestCase):
    def test_merge_materializes_quantity_instances_in_object_program(self) -> None:
        cluster_forge = {
            "clusters": [
                {
                    "cluster_id": "sleep_core",
                    "tag": "sleep",
                    "members": ["bed", "nightstand"],
                    "anchors": ["bed"],
                    "cluster_rules": {
                        "allowed_rotations": {
                            "bed": [0, 180],
                            "nightstand": [0, 90, 180, 270],
                        },
                        "facing": {
                            "nightstand": {"front": "south"},
                        },
                        "access_requirements": [
                            {"id": "nightstand", "type": "front_access"},
                        ],
                        "semantic_placements": [
                            {
                                "id": "nightstand",
                                "relative_to": "bed",
                                "kind": "anchor_side",
                                "side_options": ["head_left", "head_right"],
                                "support_role": "side_support",
                                "band_intent": "beside_base",
                            },
                        ],
                        "anchor_first_policy": {
                            "dominant_anchor_id": "bed",
                            "placement_order": ["bed", "nightstand"],
                            "droppable_ids": ["nightstand"],
                        },
                    },
                },
            ],
        }
        tier_count = {
            "decisions": [
                _decision(
                    "bed",
                    role="dominant_anchor",
                    preserve_level="highest",
                ),
                _decision("nightstand", quantity=2),
            ],
        }

        merged = merge_cluster_outputs(cluster_forge, tier_count)
        cluster = merged["clusters"][0]
        object_program = cluster["object_program"]

        self.assertEqual(cluster["members"], ["bed", "nightstand", "nightstand_2"])
        self.assertEqual(object_program["placement_order"], cluster["members"])
        self.assertEqual(object_program["anchors"], ["bed"])
        self.assertEqual(
            set(object_program["object_specs_by_id"]),
            {"bed", "nightstand", "nightstand_2"},
        )
        self.assertEqual(
            object_program["object_specs_by_id"]["nightstand_2"]["category"],
            "nightstand",
        )
        self.assertEqual(
            [edge["object_id"] for edge in object_program["support_edges"]],
            ["nightstand", "nightstand_2"],
        )
        self.assertEqual(
            [edge["side_options"] for edge in object_program["support_edges"]],
            [["left"], ["right"]],
        )
        self.assertEqual(
            [row["id"] for row in object_program["access_requirements"]],
            ["nightstand", "nightstand_2"],
        )
        self.assertEqual(
            object_program["droppable_ids"],
            ["nightstand", "nightstand_2"],
        )

    def test_merge_uses_min_keep_not_preserve_level_for_required_copies(self) -> None:
        cluster_forge = {
            "clusters": [
                {
                    "cluster_id": "activity_core",
                    "members": ["main_surface", "support_surface"],
                    "anchors": ["main_surface"],
                    "cluster_rules": {
                        "anchor_first_policy": {
                            "dominant_anchor_id": "main_surface",
                            "placement_order": ["main_surface", "support_surface"],
                            "protected_ids": ["main_surface"],
                        },
                    },
                }
            ],
        }
        tier_count = {
            "decisions": [
                _decision(
                    "main_surface",
                    cluster_id="activity_core",
                    role="dominant_anchor",
                    preserve_level="highest",
                    min_keep=1,
                    protected=True,
                ),
                _decision(
                    "support_surface",
                    cluster_id="activity_core",
                    quantity=2,
                    preserve_level="highest",
                    min_keep=1,
                    protected=True,
                ),
            ],
        }

        merged = merge_cluster_outputs(cluster_forge, tier_count)
        object_program = merged["clusters"][0]["object_program"]

        self.assertEqual(
            object_program["required_object_ids"],
            ["main_surface", "support_surface"],
        )
        self.assertEqual(object_program["optional_object_ids"], ["support_surface_2"])
        self.assertIn("support_surface_2", object_program["droppable_ids"])
        self.assertFalse(
            object_program["object_specs_by_id"]["support_surface_2"]["protected"]
        )

    def test_merge_preserves_drop_actions_after_instance_expansion(self) -> None:
        cluster_forge = {
            "clusters": [
                {
                    "cluster_id": "kitchen_dining_core",
                    "members": ["dining_table", "dining_chair"],
                    "anchors": ["dining_table"],
                    "cluster_rules": {
                        "degradation_ladder": [
                            "drop_chair",
                            "reduce_chair_count",
                            "drop_table_last",
                        ],
                        "anchor_first_policy": {
                            "dominant_anchor_id": "dining_table",
                            "protected_ids": ["dining_table"],
                            "droppable_ids": ["dining_chair"],
                        },
                    },
                }
            ],
        }
        tier_count = {
            "decisions": [
                _decision(
                    "dining_table",
                    cluster_id="kitchen_dining_core",
                    role="dominant_anchor",
                    min_keep=1,
                    protected=True,
                ),
                _decision(
                    "dining_chair",
                    cluster_id="kitchen_dining_core",
                    quantity=2,
                    min_keep=1,
                    droppable=True,
                ),
            ],
        }

        merged = merge_cluster_outputs(cluster_forge, tier_count)
        object_program = merged["clusters"][0]["object_program"]

        self.assertEqual(
            object_program["degradation_ladder"],
            [
                "drop_dining_chair",
                "drop_dining_chair_2",
                "reduce_chair_count",
                "drop_dining_table",
            ],
        )
        self.assertEqual(
            object_program["droppable_ids"],
            ["dining_chair", "dining_chair_2"],
        )


if __name__ == "__main__":
    unittest.main()
