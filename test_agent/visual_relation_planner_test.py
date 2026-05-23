from __future__ import annotations

import unittest

from agent.visual_relation_planner import (
    VisualRelationPlanner,
    validate_floorplan_image_data_url,
    validate_visual_relation_output,
    visual_relation_plan_from_output,
)


class VisualRelationPlannerTest(unittest.TestCase):
    def test_validation_rejects_invented_cluster_and_coordinates(self) -> None:
        payload = _valid_visual_payload()
        payload["cluster_zone_plan"][0]["x_mm"] = 1200

        with self.assertRaisesRegex(ValueError, "schema validation failed"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
            )

        payload = _valid_visual_payload()
        payload["object_region_plan"][0]["x_mm"] = 1200

        with self.assertRaisesRegex(ValueError, "schema validation failed"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
            )

        payload = _valid_visual_payload()
        payload["cluster_zone_plan"][0]["cluster_id"] = "invented"

        with self.assertRaisesRegex(ValueError, "invented cluster ids"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
            )

    def test_validation_rejects_unknown_region_duplicate_object_and_bad_status(
        self,
    ) -> None:
        payload = _valid_visual_payload()
        payload["cluster_zone_plan"][0]["zone_assignment"] = "secret_zone"

        with self.assertRaisesRegex(ValueError, "unknown visual regions"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
            )

        payload = _valid_visual_payload()
        payload["object_region_plan"][0]["region_id"] = "secret_zone"

        with self.assertRaisesRegex(ValueError, "unknown visual regions"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
            )

        payload = _valid_visual_payload()
        payload["object_orientations"].append(dict(payload["object_orientations"][0]))

        with self.assertRaisesRegex(ValueError, "schema validation failed"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
            )

        payload = _valid_visual_payload()
        payload["status"] = "MAYBE"

        with self.assertRaisesRegex(ValueError, "schema validation failed"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
            )

    def test_validation_rejects_unknown_or_missing_semantic_anchor(self) -> None:
        payload = _valid_visual_payload()
        payload["object_region_plan"][1][
            "semantic_anchor_region_id"
        ] = "invented_anchor"

        with self.assertRaisesRegex(ValueError, "unknown semantic anchor"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
                available_semantic_anchor_region_ids={
                    "left_wall_anchor",
                    "right_wall_focal",
                },
            )

        payload = _valid_visual_payload()
        payload["object_region_plan"][1].pop("semantic_anchor_region_id")

        with self.assertRaisesRegex(ValueError, "omitted semantic anchor"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
                available_semantic_anchor_region_ids={
                    "left_wall_anchor",
                    "right_wall_focal",
                },
            )

        payload = _valid_visual_payload()
        payload["object_region_plan"][1][
            "semantic_anchor_region_id"
        ] = "center_openness_core"

        with self.assertRaisesRegex(ValueError, "disallowed role"):
            validate_visual_relation_output(
                payload,
                available_cluster_ids={"main_seating", "media"},
                object_ids_by_cluster={
                    "main_seating": {"sofa"},
                    "media": {"tv_console"},
                },
                available_region_ids={"left_wall_zone", "right_wall_zone"},
                available_semantic_anchor_region_ids={
                    "left_wall_anchor",
                    "right_wall_focal",
                    "center_openness_core",
                },
                available_semantic_anchor_regions={
                    "center_openness_core": {
                        "allowed_roles": ["decor"],
                        "lockable": False,
                    }
                },
            )

    def test_validation_accepts_object_region_from_semantic_anchor_catalog(
        self,
    ) -> None:
        payload = _valid_visual_payload()
        payload["object_region_plan"][0]["region_id"] = "privacy_back_zone"
        payload["object_region_plan"][0][
            "semantic_anchor_region_id"
        ] = "privacy_back_zone"

        output = validate_visual_relation_output(
            payload,
            available_cluster_ids={"main_seating", "media"},
            object_ids_by_cluster={
                "main_seating": {"sofa"},
                "media": {"tv_console"},
            },
            available_region_ids={"left_wall_zone", "right_wall_zone"},
            available_semantic_anchor_region_ids={
                "left_wall_anchor",
                "right_wall_focal",
                "privacy_back_zone",
            },
            available_semantic_anchor_regions={
                "privacy_back_zone": {
                    "allowed_roles": ["anchor", "storage", "support"],
                    "lockable": True,
                },
                "right_wall_focal": {
                    "allowed_roles": ["visual_focus"],
                    "lockable": True,
                },
            },
        )

        self.assertEqual(
            output.object_region_plan[0].semantic_anchor_region_id,
            "privacy_back_zone",
        )

    def test_validation_accepts_cluster_zone_from_semantic_anchor_catalog(
        self,
    ) -> None:
        payload = _valid_visual_payload()
        payload["cluster_zone_plan"][0]["zone_assignment"] = "privacy_back_zone"
        payload["object_region_plan"][0]["region_id"] = "privacy_back_zone"
        payload["object_region_plan"][0][
            "semantic_anchor_region_id"
        ] = "privacy_back_zone"

        output = validate_visual_relation_output(
            payload,
            available_cluster_ids={"main_seating", "media"},
            object_ids_by_cluster={
                "main_seating": {"sofa"},
                "media": {"tv_console"},
            },
            available_region_ids={"left_wall_zone", "right_wall_zone"},
            available_semantic_anchor_region_ids={
                "left_wall_anchor",
                "right_wall_focal",
                "privacy_back_zone",
            },
            available_semantic_anchor_regions={
                "privacy_back_zone": {
                    "allowed_roles": ["anchor", "storage", "support"],
                    "lockable": True,
                },
                "right_wall_focal": {
                    "allowed_roles": ["visual_focus"],
                    "lockable": True,
                },
            },
        )

        self.assertEqual(
            output.cluster_zone_plan[0].zone_assignment,
            "privacy_back_zone",
        )

    def test_valid_output_converts_to_solver_relation_plan(self) -> None:
        output = validate_visual_relation_output(
            _valid_visual_payload(),
            available_cluster_ids={"main_seating", "media"},
            object_ids_by_cluster={
                "main_seating": {"sofa"},
                "media": {"tv_console"},
            },
            available_region_ids={"left_wall_zone", "right_wall_zone"},
            available_semantic_anchor_region_ids={
                "left_wall_anchor",
                "right_wall_focal",
            },
        )

        relation_plan = visual_relation_plan_from_output(
            output=output,
            room_model_json=_room_model(),
            macro_region_map={"regions": []},
            semantic_anchor_catalog=(
                {
                    "anchor_region_id": "left_wall_anchor",
                    "source_region_id": "left_wall_anchor",
                    "kind": "wall_anchor",
                    "wall_side": "left_wall",
                    "corner_hint": "top_left",
                    "allowed_roles": ["anchor"],
                    "lockable": True,
                    "score": 0.7,
                },
                {
                    "anchor_region_id": "right_wall_focal",
                    "source_region_id": "right_wall_focal",
                    "kind": "focal_wall",
                    "wall_side": "right_wall",
                    "corner_hint": "top_right",
                    "allowed_roles": ["visual_focus"],
                    "lockable": True,
                    "score": 0.9,
                },
            ),
        )

        self.assertEqual(relation_plan["planner_kind"], "visual_relation_planner")
        self.assertEqual(
            relation_plan["layout_intent_profile"]["primary_cluster_id"],
            "main_seating",
        )
        self.assertEqual(
            relation_plan["anchor_region_preferences_by_cluster"]["main_seating"][
                "zone_assignment"
            ],
            "left_wall_zone",
        )
        self.assertEqual(relation_plan["object_orientations"][0]["object_id"], "sofa")
        self.assertEqual(
            relation_plan["object_region_preferences_by_cluster"]["main_seating"][
                "sofa"
            ]["wall_side"],
            "left_wall",
        )
        self.assertEqual(
            relation_plan["functional_pair_hints"][0]["object_id"],
            "sofa",
        )
        self.assertEqual(
            relation_plan["semantic_anchor_preferences_by_cluster"]["media"][
                "tv_console"
            ]["semantic_anchor_region_id"],
            "right_wall_focal",
        )
        self.assertNotIn("x_mm", str(relation_plan))

    def test_mock_model_response_generates_relation_plan(self) -> None:
        class _FakeVisionClient:
            def create_json_response(self, **_kwargs):
                return _valid_visual_payload()

        planner = VisualRelationPlanner(
            client=_FakeVisionClient(),
            model_name="gpt-5.4-mini",
            max_tokens=512,
        )
        result = planner.generate(
            room_model_json=_room_model(),
            merged_clusters_json=_merged_clusters(),
            floorplan_image_data_url="data:image/png;base64,AA==",
            description="Compact living room.",
        )

        self.assertEqual(result.model_name, "gpt-5.4-mini")
        self.assertEqual(result.output.primary_cluster_id, "main_seating")
        self.assertEqual(
            result.relation_plan["planner_kind"], "visual_relation_planner"
        )
        self.assertIn("macro_concept", result.relation_plan)

    def test_model_adapter_normalizes_case_like_response(self) -> None:
        class _FakeVisionClient:
            def create_json_response(self, **_kwargs):
                return {
                    "cluster_zone_plan": [
                        {
                            "cluster_id": "main_seating",
                            "zone_assignment": "left_wall_zone",
                            "confidence": 0.92,
                        },
                        {
                            "cluster_id": "media",
                            "zone_assignment": "right_wall_zone",
                            "confidence": 0.88,
                        },
                    ],
                    "notes": "Use the visual floorplan to keep the center readable.",
                    "macro_region_intent": [
                        {
                            "region_id": "left_wall_zone",
                            "intent": "primary focus",
                        }
                    ],
                    "object_zone_preferences": [
                        {
                            "object_id": "sofa",
                            "preferred_zone": "left_wall_zone",
                        }
                    ],
                    "missing": ["mirror"],
                }

        planner = VisualRelationPlanner(
            client=_FakeVisionClient(),
            model_name="gpt-5.4-mini",
            max_tokens=512,
        )
        result = planner.generate(
            room_model_json=_room_model(),
            merged_clusters_json=_merged_clusters(),
            floorplan_image_data_url="data:image/png;base64,AA==",
            description="Compact living room.",
        )

        self.assertEqual(result.output.status, "OK")
        self.assertEqual(result.output.primary_cluster_id, "main_seating")
        self.assertTrue(
            result.output.cluster_zone_plan[0].reason.startswith(
                "Visual floorplan intent"
            )
        )
        self.assertIn("Filled missing `status` as OK.", result.normalization_notes)
        self.assertIn("Converted string `notes` to a list.", result.normalization_notes)
        self.assertIn(
            "Cleared `missing` because status is not NEED_INFO.",
            result.normalization_notes,
        )
        self.assertEqual(result.output.missing, [])
        self.assertNotIn("confidence", str(result.normalized_output))
        self.assertEqual(
            result.relation_plan["planner_kind"], "visual_relation_planner"
        )

    def test_image_data_url_validation(self) -> None:
        self.assertEqual(
            validate_floorplan_image_data_url("data:image/png;base64,AA=="),
            "data:image/png;base64,AA==",
        )
        with self.assertRaisesRegex(ValueError, "PNG, JPEG, or WEBP"):
            validate_floorplan_image_data_url("data:text/plain;base64,AA==")


def _valid_visual_payload() -> dict[str, object]:
    return {
        "status": "OK",
        "concept_id": "visual_concept_01",
        "concept_family": "zoned",
        "primary_cluster_id": "main_seating",
        "secondary_cluster_id": "media",
        "cluster_zone_plan": [
            {
                "cluster_id": "main_seating",
                "zone_assignment": "left_wall_zone",
                "preferred_wall_side": "left_wall",
                "center_usage": "primary",
                "anchor_strength": "strong",
                "priority": "high",
                "reason": "Seating reads as the main social anchor.",
            },
            {
                "cluster_id": "media",
                "zone_assignment": "right_wall_zone",
                "preferred_wall_side": "right_wall",
                "center_usage": "none",
                "anchor_strength": "strong",
                "priority": "high",
                "reason": "Media should face the seating cluster.",
            },
        ],
        "object_orientations": [
            {
                "cluster_id": "main_seating",
                "object_id": "sofa",
                "intents": ["face_object", "preserve_front_access"],
                "target_object_id": "tv_console",
                "target_object_cluster_id": "media",
                "priority": "high",
                "reason": "Sofa should face the TV console.",
            }
        ],
        "object_region_plan": [
            {
                "cluster_id": "main_seating",
                "object_id": "sofa",
                "region_id": "left_wall_zone",
                "wall_side": "left_wall",
                "corner_hint": "none",
                "semantic_anchor_region_id": "left_wall_anchor",
                "anchor_lock_strength": "hard",
                "placement_role": "anchor",
                "facing_intent": "face_target",
                "target_object_id": "tv_console",
                "target_object_cluster_id": "media",
                "priority": "high",
                "reason": "Sofa belongs on the left side and faces the media wall.",
            },
            {
                "cluster_id": "media",
                "object_id": "tv_console",
                "region_id": "right_wall_zone",
                "wall_side": "right_wall",
                "corner_hint": "none",
                "semantic_anchor_region_id": "right_wall_focal",
                "anchor_lock_strength": "hard",
                "placement_role": "visual_focus",
                "facing_intent": "face_target",
                "target_object_id": "sofa",
                "target_object_cluster_id": "main_seating",
                "priority": "high",
                "reason": "TV console belongs on the right side facing seating.",
            },
        ],
        "object_viewing_pairs": [
            {
                "viewer_cluster_id": "main_seating",
                "viewer_object_id": "sofa",
                "target_cluster_id": "media",
                "target_object_id": "tv_console",
                "pair_type": "viewer_faces_target",
                "priority": "high",
                "reason": "Primary viewing relation.",
            }
        ],
        "cluster_relations": [
            {
                "a": "main_seating",
                "b": "media",
                "relation": "far_if_possible",
                "priority": "medium",
                "reason": "Keep enough viewing distance.",
            }
        ],
        "notes": ["Visual plan accepted."],
        "missing": [],
    }


def _room_model() -> dict[str, object]:
    return {
        "room": {
            "room_id": "room_1",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 4000, "y": 0},
                {"x": 4000, "y": 3000},
                {"x": 0, "y": 3000},
            ],
        },
        "openings": {"doors": [], "windows": []},
        "affordance_map": {
            "wall_anchor_candidates": [
                {
                    "id": "left_wall_anchor",
                    "wall_id": "wall_left",
                    "anchor_polygon_ccw": [
                        {"x": 0, "y": 0},
                        {"x": 700, "y": 0},
                        {"x": 700, "y": 3000},
                        {"x": 0, "y": 3000},
                    ],
                    "score": 0.7,
                }
            ],
            "focal_surfaces": [
                {
                    "id": "right_wall_focal",
                    "wall_id": "wall_right",
                    "segment_mm": [
                        {"x": 4000, "y": 0},
                        {"x": 4000, "y": 3000},
                    ],
                    "score": 0.9,
                }
            ],
        },
        "meta": {"room_type": "living_room"},
    }


def _merged_clusters() -> dict[str, object]:
    return {
        "clusters": [
            {
                "cluster_id": "main_seating",
                "tag": "seating",
                "members": ["sofa"],
                "anchors": ["sofa"],
                "decisions": [
                    {
                        "object_type": "sofa",
                        "quantity": 1,
                        "size_tier": "M",
                        "priority": "anchor",
                    }
                ],
            },
            {
                "cluster_id": "media",
                "tag": "media",
                "members": ["tv_console"],
                "anchors": ["tv_console"],
                "decisions": [
                    {
                        "object_type": "tv_console",
                        "quantity": 1,
                        "size_tier": "M",
                        "priority": "anchor",
                    }
                ],
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
