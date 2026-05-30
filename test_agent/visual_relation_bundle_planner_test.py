from __future__ import annotations

import unittest
from collections.abc import Mapping

from agent.seed_concept_generator import solver_plan_from_concept
from agent.visual_relation_planner import VisualRelationBundlePlanner


_IMAGE_DATA_URL = "data:image/png;base64,iVBORw0KGgo="


class _FakeVisionJsonClient:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def create_json_response(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        max_tokens: int,
    ) -> Mapping[str, object]:
        self.calls.append(
            {
                "model_name": model_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_data_url": image_data_url,
                "max_tokens": max_tokens,
            }
        )
        return self.payload


class VisualRelationBundlePlannerTest(unittest.TestCase):
    def test_valid_llm_guidance_becomes_solver_compatible_macro_bundle(self) -> None:
        client = _FakeVisionJsonClient(
            {
                "family_order": ["zoned", "open_center", "focal_axis"],
                "concept_blueprints": [
                    {
                        "concept_family": "zoned",
                        "topology_policy": {
                            "reserve_center_degree": "medium",
                            "wall_loading_bias": "zone_specific",
                            "daylight_bias": "work_near_window",
                        },
                        "cluster_zone_overrides": [
                            {
                                "cluster_id": "main_seating",
                                "zone_assignment": "floating_center_zone",
                                "preferred_wall_side": "center",
                                "center_usage": "primary",
                            },
                            {
                                "cluster_id": "media",
                                "zone_assignment": "right_wall_zone",
                                "preferred_wall_side": "right_wall",
                                "center_usage": "none",
                            },
                        ],
                        "notes": ["Keep social seating readable from the entry."],
                    }
                ],
                "notes": ["Use the screenshot to keep circulation open."],
            }
        )

        result = VisualRelationBundlePlanner(
            client=client,
            model_name="gpt-5.4-mini",
            max_tokens=900,
        ).generate_bundle(
            room_model_json=_room_model(),
            merged_clusters_json=_clusters(),
            floorplan_image_data_url=_IMAGE_DATA_URL,
            target_count=3,
            description="living room",
        )

        self.assertEqual(result.model_name, "gpt-5.4-mini")
        self.assertEqual(client.calls[0]["image_data_url"], _IMAGE_DATA_URL)
        self.assertEqual(
            result.bundle["controller"], "visual_llm_macro_concept_generator"
        )
        concepts = result.bundle["concepts"]
        self.assertTrue(concepts)
        self.assertEqual(concepts[0]["concept_family"], "zoned")
        solver_plan = solver_plan_from_concept(
            concept=concepts[0],
            room_model_json=_room_model(),
            room_type="living_room",
        )
        self.assertTrue(solver_plan["anchor_layout_hints_by_cluster"])
        self.assertIn("media", solver_plan["anchor_layout_hints_by_cluster"])

    def test_rejects_raw_geometry_fields(self) -> None:
        client = _FakeVisionJsonClient(
            {
                "family_order": ["zoned"],
                "concept_blueprints": [
                    {
                        "concept_family": "zoned",
                        "cluster_zone_overrides": [
                            {
                                "cluster_id": "media",
                                "zone_assignment": "right_wall_zone",
                                "x_mm": 1200,
                            }
                        ],
                    }
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "raw geometry"):
            VisualRelationBundlePlanner(client=client).generate_bundle(
                room_model_json=_room_model(),
                merged_clusters_json=_clusters(),
                floorplan_image_data_url=_IMAGE_DATA_URL,
            )

    def test_rejects_invented_cluster_or_region_ids(self) -> None:
        client = _FakeVisionJsonClient(
            {
                "family_order": ["zoned"],
                "concept_blueprints": [
                    {
                        "concept_family": "zoned",
                        "cluster_zone_overrides": [
                            {
                                "cluster_id": "imaginary_cluster",
                                "zone_assignment": "imaginary_region",
                            }
                        ],
                    }
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "unknown cluster"):
            VisualRelationBundlePlanner(client=client).generate_bundle(
                room_model_json=_room_model(),
                merged_clusters_json=_clusters(),
                floorplan_image_data_url=_IMAGE_DATA_URL,
            )


def _room_model() -> dict[str, object]:
    return {
        "room": {
            "room_id": "living_test",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 5000, "y": 0},
                {"x": 5000, "y": 3600},
                {"x": 0, "y": 3600},
            ],
        },
        "meta": {"room_type": "living_room"},
        "openings": {
            "doors": [{"id": "entry_door"}],
            "windows": [{"id": "main_window"}],
        },
        "affordance_map": {
            "focal_surfaces": ["wall_north"],
            "usable_wall_segments": ["wall_north", "wall_south", "wall_east"],
            "entry_landing_zones": ["entry_landing"],
            "primary_circulation_corridors": ["entry_to_center"],
            "center_openness_regions": ["room_center"],
            "daylight_regions": ["main_window_zone"],
            "wall_anchor_candidates": ["wall_south", "wall_east"],
        },
    }


def _clusters() -> dict[str, object]:
    return {
        "clusters": [
            {
                "cluster_id": "main_seating",
                "members": ["sectional_sofa", "coffee_table"],
                "anchors": ["sectional_sofa"],
                "object_program": {
                    "required_object_ids": ["sectional_sofa", "coffee_table"],
                    "optional_object_ids": [],
                    "droppable_ids": [],
                },
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "medium",
                        "daylight_affinity": "medium",
                        "floating_allowed": True,
                    }
                },
            },
            {
                "cluster_id": "media",
                "members": ["tv_console"],
                "anchors": ["tv_console"],
                "object_program": {
                    "required_object_ids": ["tv_console"],
                    "optional_object_ids": [],
                    "droppable_ids": [],
                },
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "high",
                        "daylight_affinity": "low",
                        "floating_allowed": False,
                    }
                },
            },
            {
                "cluster_id": "storage",
                "members": ["bookcase"],
                "anchors": ["bookcase"],
                "object_program": {
                    "required_object_ids": ["bookcase"],
                    "optional_object_ids": [],
                    "droppable_ids": [],
                },
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "high",
                        "floating_allowed": False,
                    }
                },
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
