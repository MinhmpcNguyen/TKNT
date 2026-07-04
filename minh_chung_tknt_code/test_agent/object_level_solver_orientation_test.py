from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.solver.solver import (  # noqa: E402
    _anchor_pair_compatibility_report,
    _anchor_pair_solution_score,
    _anchor_forbidden_overlap_exceeds_limit,
    _build_object_solver_world,
    _collect_cluster_forbidden_regions,
    _droppable_object_ids,
    _front_matches_any_required_direction,
    _generate_anchor_pose_candidates,
    _materialize_object_row,
    _near_wall_inward_front_options,
    _normalized_support_side_options,
    _object_level_face_pair_issues,
    _object_level_interaction_quality,
    _object_level_orientation_issues,
    _object_level_required_face_pairs,
    _object_rect_is_usable,
    _place_support_objects_for_solution,
    _repair_object_level_solution_geometry,
    _rect_inside_polygon,
    _rect_inside_room_footprint,
    _rect_overlap_ratio,
    _required_support_standalone_slot_candidates,
    _search_anchor_solutions,
    _support_slot_relaxation_penalty,
    _support_slot_candidates,
    _verify_object_level_solution,
    _wall_flush_origin_candidates,
    solve_object_level_layout,
)
from agent.seed_concept_generator import solver_plan_from_concept  # noqa: E402


def _room_model() -> dict[str, object]:
    return {
        "room": {
            "room_id": "room_1",
            "room_type": "living_room",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 4000},
                {"x": 0, "y": 4000},
            ],
            "wall_thickness_mm": 0,
        },
        "regions": [
            {
                "id": "top_storage",
                "bbox": {"min_x": 0, "min_y": 3400, "max_x": 6000, "max_y": 4000},
            },
            {
                "id": "center_lounge",
                "bbox": {"min_x": 1500, "min_y": 1200, "max_x": 4500, "max_y": 3000},
            },
        ],
    }


def _object_program_cluster(
    *,
    cluster_id: str,
    object_id: str,
    category: str,
    dims: dict[str, int],
    support_edges: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    members = [object_id]
    if support_edges:
        members.extend(str(edge["object_id"]) for edge in support_edges)
    specs = {
        object_id: {
            "object_id": object_id,
            "category": category,
            "role": "anchor",
            "priority": "anchor",
            "rep_dims_mm": dims,
            "allowed_rotations": [0, 90, 180, 270],
            "front": "top",
        }
    }
    for edge in support_edges or []:
        child_id = str(edge["object_id"])
        specs[child_id] = {
            "object_id": child_id,
            "category": child_id,
            "role": "secondary",
            "priority": "support",
            "rep_dims_mm": {"L": 700, "W": 700, "H": 800},
            "allowed_rotations": [0, 90, 180, 270],
            "front": "top",
        }
    return {
        "cluster_id": cluster_id,
        "members": members,
        "anchors": [object_id],
        "object_program": {
            "cluster_id": cluster_id,
            "members": members,
            "anchors": [object_id],
            "dominant_anchor_id": object_id,
            "dominant_anchor_candidates": [object_id],
            "placement_order": members,
            "support_edges": support_edges or [],
            "protected_ids": [object_id],
            "droppable_ids": [],
            "degradation_ladder": [],
            "object_specs_by_id": specs,
        },
        "cluster_rules": {},
    }


class ObjectLevelSolverOrientationTest(unittest.TestCase):
    def _support_slot_world(
        self,
        *,
        side_options: list[str],
        droppable_support: bool,
        protected_support: bool,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        cluster = _object_program_cluster(
            cluster_id="main_seating",
            object_id="sofa",
            category="sofa",
            dims={"L": 1000, "W": 600, "H": 800},
            support_edges=[
                {
                    "object_id": "floor_lamp",
                    "relative_to": "sofa",
                    "side_options": side_options,
                    "gap_min": 100,
                    "gap_max": 100,
                    "orientation": "same_direction",
                }
            ],
        )
        object_program = cluster["object_program"]
        object_program["droppable_ids"] = ["floor_lamp"] if droppable_support else []
        if protected_support:
            object_program["protected_ids"] = ["sofa", "floor_lamp"]

        room_model = _room_model()
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=None,
            cluster_constraints=None,
            grid_mm=50,
        )
        anchor_solution = {
            "anchor_solution": {
                "main_seating": {
                    "cluster_id": "main_seating",
                    "object_id": "sofa",
                    "x": 2000,
                    "y": 1500,
                    "rot": 0,
                    "rect": (2000, 1500, 3000, 2100),
                    "candidate_score": 0.0,
                }
            },
            "anchor_score": 0.0,
            "dropped_inventory_by_cluster": {},
        }
        return room_model, world, anchor_solution

    def _solve_case_first_concept(self, case_root: Path) -> dict[str, object]:
        room_model = json.loads((case_root / "01_room_interpreter.json").read_text())
        merged_clusters = json.loads((case_root / "04_cluster_merged.json").read_text())
        cluster_constraints = json.loads(
            (case_root / "02_cluster_forge.json").read_text()
        )
        concept_bundle = json.loads(
            (case_root / "05c_seed_relation_plans.json").read_text()
        )
        concept = concept_bundle["concepts"][0]
        room = (
            room_model.get("room") if isinstance(room_model.get("room"), dict) else {}
        )
        relation_plan = solver_plan_from_concept(
            concept=concept,
            room_model_json=room_model,
            room_type=str(room.get("room_type") or room_model.get("room_type") or ""),
        )
        return solve_object_level_layout(
            room_model=room_model,
            merged_clusters=merged_clusters,
            relation_plan=relation_plan,
            cluster_constraints=cluster_constraints,
            grid_mm=50,
            max_rounds=3,
        )

    def test_rect_inside_polygon_accepts_interior_points_with_slanted_wall(
        self,
    ) -> None:
        polygon = [(0.0, 0.0), (5100.0, 0.0), (5050.0, 4000.0), (0.0, 4000.0)]

        self.assertTrue(_rect_inside_polygon((0, 0, 2060, 960), polygon))
        self.assertTrue(_rect_inside_polygon((1500, 1050, 2000, 1950), polygon))
        self.assertFalse(_rect_inside_polygon((5060, 1000, 5100, 2000), polygon))

    def test_rectangular_room_footprint_uses_bbox_fast_path(self) -> None:
        world = _build_object_solver_world(
            room_model=_room_model(),
            merged_clusters={"clusters": []},
            relation_plan=None,
            cluster_constraints=None,
            grid_mm=50,
        )

        self.assertTrue(world["room_polygon_is_axis_aligned_rectangle"])
        self.assertTrue(_rect_inside_room_footprint((100, 100, 1200, 1000), world))
        self.assertFalse(_rect_inside_room_footprint((-50, 100, 1200, 1000), world))

    def test_wall_thickness_insets_floor_footprint_from_room_boundary(self) -> None:
        room_model = _room_model()
        room = room_model["room"]
        self.assertIsInstance(room, dict)
        if not isinstance(room, dict):
            raise AssertionError("room fixture must be a dict")
        room["wall_thickness_mm"] = 100
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": []},
            relation_plan=None,
            cluster_constraints=None,
            grid_mm=50,
        )

        self.assertEqual(world["placement_room_bbox"], (50, 50, 5950, 3950))
        self.assertFalse(_rect_inside_room_footprint((0, 100, 500, 600), world))
        self.assertTrue(_rect_inside_room_footprint((50, 100, 550, 600), world))
        self.assertTrue(
            _rect_inside_room_footprint(
                (0, 100, 500, 600),
                world,
                row={"collision_layer": "wall"},
            )
        )

    def test_wall_backed_anchor_candidates_align_to_inner_face(self) -> None:
        room_model = _room_model()
        room = room_model["room"]
        self.assertIsInstance(room, dict)
        if not isinstance(room, dict):
            raise AssertionError("room fixture must be a dict")
        room["wall_thickness_mm"] = 100
        cluster = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1000, "W": 500, "H": 2200},
        )
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "storage_core": {
                    "zone_assignment": "left_wall_zone",
                    "preferred_region_ids": ["left_wall_zone"],
                    "placement_behavior": {"wall_backing": "required"},
                }
            }
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["storage_core"],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        self.assertTrue(all(candidate["rect"][0] >= 50 for candidate in candidates))
        self.assertIn(50, {candidate["rect"][0] for candidate in candidates})

    def test_storage_wall_options_mark_front_inward_parallel_candidates(self) -> None:
        room_model = _room_model()
        cluster = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1200, "W": 420, "H": 2200},
        )
        cluster["cluster_rules"]["access_requirements"] = [
            {"id": "wardrobe", "type": "front_clearance", "required": True}
        ]
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "storage_core": {
                    "zone_assignment": "left_wall_zone",
                    "preferred_region_ids": ["left_wall_zone"],
                    "role_kind": "storage",
                    "scenario_role": "secondary",
                    "placement_behavior": {"wall_backing": "required"},
                }
            }
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["storage_core"],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        wall_options = world["_anchor_wall_options_by_cluster"]["storage_core"]
        left_wall = wall_options["left"]
        self.assertGreater(left_wall["valid_candidate_count"], 0)
        self.assertGreater(left_wall["front_inward_candidate_count"], 0)
        self.assertGreater(left_wall["wall_parallel_candidate_count"], 0)
        best = left_wall["best_candidate"]
        self.assertTrue(best["front_inward_match"])
        self.assertTrue(best["long_axis_wall_parallel"])
        self.assertGreater(best["storage_wall_fit_score"], 0.0)
        rehome = left_wall["best_rehome_candidate"]
        self.assertTrue(rehome["front_inward_match"])
        self.assertTrue(rehome["long_axis_wall_parallel"])

    def test_interaction_quality_prefers_axial_view_target_over_bare_diagonal(
        self,
    ) -> None:
        relation_plan = {
            "object_viewing_pairs": [
                {
                    "viewer_cluster_id": "sleep_core",
                    "viewer_object_id": "bed",
                    "viewer_object_type": "bed",
                    "target_cluster_id": "media_optional",
                    "target_object_id": "tv_console",
                    "target_object_type": "tv_console",
                    "relation": "view_target",
                    "required": True,
                }
            ]
        }
        axial = _object_level_interaction_quality(
            placed_objects=[
                {
                    "cluster_id": "sleep_core",
                    "object_id": "bed",
                    "rect": (0, 0, 2000, 1600),
                    "front_world": {"dx": 1, "dy": 0},
                },
                {
                    "cluster_id": "media_optional",
                    "object_id": "tv_console",
                    "rect": (3200, 500, 4200, 900),
                    "front_world": {"dx": -1, "dy": 0},
                },
            ],
            relation_plan=relation_plan,
        )
        diagonal = _object_level_interaction_quality(
            placed_objects=[
                {
                    "cluster_id": "sleep_core",
                    "object_id": "bed",
                    "rect": (0, 0, 2000, 1600),
                    "front_world": {"dx": 1, "dy": 0},
                },
                {
                    "cluster_id": "media_optional",
                    "object_id": "tv_console",
                    "rect": (3200, 2100, 4200, 2500),
                    "front_world": {"dx": -1, "dy": 0},
                },
            ],
            relation_plan=relation_plan,
        )

        self.assertGreater(axial["score"], diagonal["score"])
        self.assertGreater(
            axial["view_targets"][0]["axis_alignment"],
            diagonal["view_targets"][0]["axis_alignment"],
        )

    def test_interaction_quality_penalizes_chair_not_facing_desk(self) -> None:
        good = _object_level_interaction_quality(
            placed_objects=[
                {
                    "cluster_id": "work_study",
                    "object_id": "desk",
                    "category": "desk",
                    "rect": (1000, 1000, 2200, 1600),
                    "front_world": {"dx": 0, "dy": 1},
                },
                {
                    "cluster_id": "work_study",
                    "object_id": "chair",
                    "category": "chair",
                    "rect": (1300, 1800, 1800, 2300),
                    "front_world": {"dx": 0, "dy": -1},
                },
            ],
            relation_plan={},
        )
        bad = _object_level_interaction_quality(
            placed_objects=[
                {
                    "cluster_id": "work_study",
                    "object_id": "desk",
                    "category": "desk",
                    "rect": (1000, 1000, 2200, 1600),
                    "front_world": {"dx": 0, "dy": 1},
                },
                {
                    "cluster_id": "work_study",
                    "object_id": "chair",
                    "category": "chair",
                    "rect": (1300, 1800, 1800, 2300),
                    "front_world": {"dx": 1, "dy": 0},
                },
            ],
            relation_plan={},
        )

        self.assertGreater(good["score"], bad["score"])
        self.assertGreater(
            good["workflow_pairs"][0]["chair_to_surface_dot"],
            bad["workflow_pairs"][0]["chair_to_surface_dot"],
        )

    def test_non_rectangular_room_footprint_keeps_polygon_check(self) -> None:
        l_room_model = {
            "room": {
                "room_id": "room_l",
                "room_type": "living_room",
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 4000},
                    {"x": 3000, "y": 4000},
                    {"x": 3000, "y": 2000},
                    {"x": 0, "y": 2000},
                ],
            }
        }
        world = _build_object_solver_world(
            room_model=l_room_model,
            merged_clusters={"clusters": []},
            relation_plan=None,
            cluster_constraints=None,
            grid_mm=50,
        )

        self.assertFalse(world["room_polygon_is_axis_aligned_rectangle"])
        self.assertTrue(_rect_inside_room_footprint((3500, 2500, 4500, 3500), world))
        self.assertFalse(_rect_inside_room_footprint((500, 2500, 1500, 3500), world))

    def test_anchor_search_preserves_optional_trial_skip_branch(self) -> None:
        world = {
            "clusters_by_id": {
                "core": {
                    "cluster_id": "core",
                    "object_program": {
                        "dominant_anchor_id": "bed",
                        "anchors": ["bed"],
                        "dominant_anchor_candidates": ["bed"],
                        "object_specs_by_id": {
                            "bed": {
                                "object_id": "bed",
                                "rep_dims_mm": {"L": 1000, "W": 1000, "H": 500},
                            }
                        },
                    },
                },
                "trial": {
                    "cluster_id": "trial",
                    "object_program": {
                        "dominant_anchor_id": "desk",
                        "anchors": ["desk"],
                        "dominant_anchor_candidates": ["desk"],
                        "object_specs_by_id": {
                            "desk": {
                                "object_id": "desk",
                                "solver_trial": True,
                                "trial_optional": True,
                                "rep_dims_mm": {"L": 800, "W": 500, "H": 750},
                            }
                        },
                    },
                },
            },
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        solutions = _search_anchor_solutions(
            anchor_order=["core", "trial"],
            anchor_candidates_by_cluster={
                "core": [
                    {
                        "cluster_id": "core",
                        "object_id": "bed",
                        "x": 0,
                        "y": 0,
                        "rot": 0,
                        "rect": (0, 0, 1000, 1000),
                        "candidate_score": 0.0,
                    }
                ],
                "trial": [
                    {
                        "cluster_id": "trial",
                        "object_id": "desk",
                        "x": 2000,
                        "y": 0,
                        "rot": 0,
                        "rect": (2000, 0, 2800, 500),
                        "candidate_score": 5000.0,
                    }
                ],
            },
            world=world,
            room_model={},
            relation_plan=None,
            max_solutions=1,
        )

        self.assertEqual(len(solutions), 1)
        self.assertIn("trial", solutions[0]["dropped_inventory_by_cluster"])

    def test_top_wall_storage_faces_inward(self) -> None:
        merged_clusters = {
            "clusters": [
                _object_program_cluster(
                    cluster_id="storage",
                    object_id="bookshelf",
                    category="bookshelf",
                    dims={"L": 1000, "W": 300, "H": 1800},
                )
            ]
        }
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "storage": {
                    "zone_assignment": "top_storage",
                    "preferred_region_ids": ["top_storage"],
                }
            },
            "object_orientations": [
                {
                    "cluster_id": "storage",
                    "object_id": "bookshelf",
                    "intents": ["back_to_wall", "front_to_open_space"],
                    "priority": "high",
                }
            ],
        }

        result = solve_object_level_layout(
            room_model=_room_model(),
            merged_clusters=merged_clusters,
            relation_plan=relation_plan,
        )

        self.assertEqual(result["status"], "OK")
        bookshelf = result["absolute_layout"]["objects"][0]
        self.assertEqual(bookshelf["front_world"], {"dx": 0, "dy": -1})
        self.assertEqual(bookshelf["rotation_ccw"], 180)
        self.assertTrue(result["hard_valid"])

    def test_near_wall_front_access_anchor_candidates_face_inward(self) -> None:
        cluster = _object_program_cluster(
            cluster_id="storage",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1750, "W": 510, "H": 2020},
        )
        cluster["cluster_rules"]["access_requirements"] = [
            {"id": "wardrobe", "type": "front_clearance", "required": True}
        ]
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "storage": {
                    "zone_assignment": "top_storage",
                    "preferred_region_ids": ["top_storage"],
                    "placement_behavior": {
                        "wall_backing": "preferred",
                        "front_space": "optional",
                    },
                }
            },
            "object_orientations": [
                {
                    "cluster_id": "storage",
                    "object_id": "wardrobe",
                    "intents": ["back_to_wall"],
                    "priority": "medium",
                }
            ],
        }
        world = _build_object_solver_world(
            room_model=_room_model(),
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["storage"],
            room_model=_room_model(),
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        near_wall_candidates = [
            candidate
            for candidate in candidates
            if world["room_bbox"][3] - candidate["rect"][3] <= 150
        ]
        self.assertTrue(near_wall_candidates)
        for candidate in near_wall_candidates:
            expected_fronts = _near_wall_inward_front_options(
                rect=candidate["rect"],
                room_bbox=world["room_bbox"],
            )
            front_world = candidate["front_world"]
            self.assertTrue(expected_fronts)
            self.assertTrue(
                _front_matches_any_required_direction(
                    front_world=(front_world["dx"], front_world["dy"]),
                    desired_fronts=expected_fronts,
                )
            )

    def test_entry_support_anchor_flanks_door_clearance_without_blocking(self) -> None:
        room_model = {
            "room": {
                "room_id": "entry_support_room",
                "room_type": "living_room",
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 4100, "y": 0},
                    {"x": 4100, "y": 5050},
                    {"x": 0, "y": 5050},
                ],
            },
            "openings": {
                "doors": [
                    {
                        "id": "door_1",
                        "segment_mm": [{"x": 0, "y": 2761}, {"x": 0, "y": 1761}],
                    }
                ]
            },
            "affordance_map": {
                "entry_landing_zones": [
                    {
                        "id": "door_1_entry_clearance",
                        "polygon_ccw": [
                            {"x": 0, "y": 2761},
                            {"x": 0, "y": 1761},
                            {"x": 900, "y": 1761},
                            {"x": 900, "y": 2761},
                        ],
                    }
                ],
                "circulation_corridors": [
                    {
                        "id": "corridor_1",
                        "from": "entry",
                        "to": "room_center",
                        "polyline_mm": [
                            {"x": 22, "y": 2261},
                            {"x": 2043, "y": 2525},
                        ],
                        "width_mm": 900,
                    }
                ],
            },
        }
        cluster = _object_program_cluster(
            cluster_id="pet_or_entry_support",
            object_id="shoe_rack",
            category="shoe_rack",
            dims={"L": 810, "W": 250, "H": 780},
        )
        relation_plan = {
            "macro_region_map": {
                "regions": [
                    {
                        "region_id": "entry_adjacent_active_zone",
                        "source_ids": ["door_1_entry_clearance"],
                    }
                ]
            },
            "anchor_layout_hints_by_cluster": {
                "pet_or_entry_support": {
                    "zone_assignment": "entry_adjacent_active_zone",
                    "preferred_region_ids": ["entry_adjacent_active_zone"],
                    "forbidden_region_ids": [
                        "door_1_entry_clearance",
                        "entry_to_center_corridor",
                    ],
                    "entry_relation": "buffer_entry_if_possible",
                    "placement_behavior": {
                        "wall_backing": "required",
                        "front_space": "optional",
                    },
                }
            },
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["pet_or_entry_support"],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        best = candidates[0]
        self.assertEqual(best["anchor_kind"], "entry_side")
        center_x = (best["rect"][0] + best["rect"][2]) / 2
        center_y = (best["rect"][1] + best["rect"][3]) / 2
        distance_sq = (center_x - 22) ** 2 + (center_y - 2261) ** 2
        self.assertLessEqual(distance_sq, 1500**2)
        self.assertEqual(
            _rect_overlap_ratio(
                best["rect"], world["region_index"]["door_1_entry_clearance"]
            ),
            0.0,
        )
        self.assertEqual(
            _rect_overlap_ratio(
                best["rect"], world["region_index"]["entry_to_center_corridor"]
            ),
            0.0,
        )

    def test_near_wall_front_access_anchor_blocks_wall_facing_pose(self) -> None:
        cluster = _object_program_cluster(
            cluster_id="storage",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1750, "W": 510, "H": 2020},
        )
        cluster["cluster_rules"]["access_requirements"] = [
            {"id": "wardrobe", "type": "front_clearance", "required": True}
        ]
        relation_plan = {
            "object_orientations": [
                {
                    "cluster_id": "storage",
                    "object_id": "wardrobe",
                    "intents": ["back_to_wall"],
                    "priority": "medium",
                }
            ]
        }
        wardrobe = _materialize_object_row(cluster, "wardrobe", 2800, 3450, 0)
        world = {
            "room_bbox": (0, 0, 6000, 4000),
            "clusters_by_id": {"storage": cluster},
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        verify = _verify_object_level_solution(
            solution={"placed_objects": [wardrobe]},
            world=world,
            room_model={},
            relation_plan=relation_plan,
        )

        self.assertFalse(verify["hard_valid"])
        self.assertFalse(verify["gallery_eligible"])
        self.assertEqual(
            verify["blocking_orientation_issues"][0]["reason"],
            "near_wall_inward",
        )
        self.assertIn("front_inward_contract_failed", verify["quality_gate_reasons"])

    def test_wall_contact_corner_accepts_any_inward_front(self) -> None:
        cluster = _object_program_cluster(
            cluster_id="storage",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1750, "W": 510, "H": 2020},
        )
        cluster["cluster_rules"]["access_requirements"] = [
            {"id": "wardrobe", "type": "front_clearance", "required": True}
        ]
        relation_plan = {
            "object_orientations": [
                {
                    "cluster_id": "storage",
                    "object_id": "wardrobe",
                    "intents": ["back_to_wall"],
                    "priority": "medium",
                }
            ]
        }
        wardrobe = _materialize_object_row(cluster, "wardrobe", 4250, 3490, 180)
        world = {
            "room_bbox": (0, 0, 6000, 4000),
            "clusters_by_id": {"storage": cluster},
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        verify = _verify_object_level_solution(
            solution={"placed_objects": [wardrobe]},
            world=world,
            room_model={},
            relation_plan=relation_plan,
        )

        self.assertTrue(verify["hard_valid"])
        self.assertEqual(verify["blocking_orientation_issues"], [])

    def test_dominant_corner_wall_blocks_short_side_inward_front(self) -> None:
        cluster = _object_program_cluster(
            cluster_id="storage",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1750, "W": 510, "H": 2020},
        )
        cluster["object_program"]["object_specs_by_id"]["wardrobe"]["front"] = "left"
        cluster["cluster_rules"]["access_requirements"] = [
            {"id": "wardrobe", "type": "front_clearance", "required": True}
        ]
        relation_plan = {
            "object_orientations": [
                {
                    "cluster_id": "storage",
                    "object_id": "wardrobe",
                    "intents": ["back_to_wall"],
                    "priority": "medium",
                }
            ]
        }
        wardrobe = _materialize_object_row(cluster, "wardrobe", 4250, 0, 0)
        world = {
            "room_bbox": (0, 0, 6000, 4000),
            "clusters_by_id": {"storage": cluster},
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        verify = _verify_object_level_solution(
            solution={"placed_objects": [wardrobe]},
            world=world,
            room_model={},
            relation_plan=relation_plan,
        )

        self.assertFalse(verify["hard_valid"])
        self.assertEqual(
            verify["blocking_orientation_issues"][0]["reason"],
            "wall_contact_inward",
        )
        self.assertEqual(
            verify["blocking_orientation_issues"][0]["desired_front_world"],
            {"dx": 0.0, "dy": 1.0},
        )
        self.assertIn("front_inward_contract_failed", verify["quality_gate_reasons"])

    def test_dominant_corner_wall_accepts_long_side_inward_front(self) -> None:
        cluster = _object_program_cluster(
            cluster_id="storage",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1750, "W": 510, "H": 2020},
        )
        cluster["cluster_rules"]["access_requirements"] = [
            {"id": "wardrobe", "type": "front_clearance", "required": True}
        ]
        relation_plan = {
            "object_orientations": [
                {
                    "cluster_id": "storage",
                    "object_id": "wardrobe",
                    "intents": ["back_to_wall"],
                    "priority": "medium",
                }
            ]
        }
        wardrobe = _materialize_object_row(cluster, "wardrobe", 4250, 0, 0)
        world = {
            "room_bbox": (0, 0, 6000, 4000),
            "clusters_by_id": {"storage": cluster},
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        verify = _verify_object_level_solution(
            solution={"placed_objects": [wardrobe]},
            world=world,
            room_model={},
            relation_plan=relation_plan,
        )

        self.assertTrue(verify["hard_valid"])
        self.assertEqual(verify["blocking_orientation_issues"], [])

    def test_corner_wall_contact_tie_accepts_either_inward_front(self) -> None:
        cluster = _object_program_cluster(
            cluster_id="storage",
            object_id="storage_cabinet",
            category="cabinet",
            dims={"L": 600, "W": 600, "H": 1400},
        )
        cluster["object_program"]["object_specs_by_id"]["storage_cabinet"][
            "front"
        ] = "left"
        cluster["cluster_rules"]["access_requirements"] = [
            {"id": "storage_cabinet", "type": "front_clearance", "required": True}
        ]
        relation_plan = {
            "object_orientations": [
                {
                    "cluster_id": "storage",
                    "object_id": "storage_cabinet",
                    "intents": ["back_to_wall"],
                    "priority": "medium",
                }
            ]
        }
        cabinet = _materialize_object_row(cluster, "storage_cabinet", 5400, 0, 0)
        world = {
            "room_bbox": (0, 0, 6000, 4000),
            "clusters_by_id": {"storage": cluster},
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        verify = _verify_object_level_solution(
            solution={"placed_objects": [cabinet]},
            world=world,
            room_model={},
            relation_plan=relation_plan,
        )

        self.assertTrue(verify["hard_valid"])
        self.assertEqual(verify["blocking_orientation_issues"], [])

    def test_anchor_candidates_reject_short_side_corner_inward_front(self) -> None:
        room_model = {
            "room": {
                "room_id": "room_1",
                "room_type": "bedroom",
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 4000},
                    {"x": 0, "y": 4000},
                ],
                "wall_thickness_mm": 0,
            },
            "regions": [
                {
                    "id": "top_right_wall",
                    "bbox": {
                        "min_x": 4250,
                        "min_y": 0,
                        "max_x": 6000,
                        "max_y": 600,
                    },
                }
            ],
        }
        cluster = _object_program_cluster(
            cluster_id="storage",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1750, "W": 510, "H": 2020},
        )
        cluster["object_program"]["object_specs_by_id"]["wardrobe"]["front"] = "left"
        cluster["cluster_rules"]["access_requirements"] = [
            {"id": "wardrobe", "type": "front_clearance", "required": True}
        ]
        relation_plan = {
            "anchor_region_preferences_by_cluster": {
                "storage": {
                    "zone_assignment": "top_right_wall",
                    "committed_region_ids": ["top_right_wall"],
                    "preferred_region_ids": ["top_right_wall"],
                    "placement_behavior": {
                        "wall_backing": "required",
                        "front_space": "required",
                    },
                }
            },
            "object_orientations": [
                {
                    "cluster_id": "storage",
                    "object_id": "wardrobe",
                    "intents": ["back_to_wall"],
                    "priority": "medium",
                }
            ],
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=cluster,
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        self.assertFalse(
            any(
                int(candidate["rot"]) == 0
                and tuple(candidate["rect"]) == (4250, 0, 6000, 510)
                for candidate in candidates
            )
        )

    def test_lounge_support_seat_faces_base(self) -> None:
        merged_clusters = {
            "clusters": [
                _object_program_cluster(
                    cluster_id="main_seating",
                    object_id="sofa",
                    category="sofa",
                    dims={"L": 1800, "W": 850, "H": 800},
                    support_edges=[
                        {
                            "object_id": "armchair",
                            "relative_to": "sofa",
                            "side_options": ["left"],
                            "gap_min": 150,
                            "gap_max": 250,
                            "orientation": "face_base",
                        }
                    ],
                )
            ]
        }
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "main_seating": {
                    "zone_assignment": "center_lounge",
                    "preferred_region_ids": ["center_lounge"],
                }
            },
            "object_orientations": [],
        }

        result = solve_object_level_layout(
            room_model=_room_model(),
            merged_clusters=merged_clusters,
            relation_plan=relation_plan,
        )

        self.assertEqual(result["status"], "OK")
        objects = {
            str(row["object_id"]): row for row in result["absolute_layout"]["objects"]
        }
        sofa = objects["sofa"]
        armchair = objects["armchair"]
        dx = sofa["center"]["x"] - armchair["center"]["x"]
        dy = sofa["center"]["y"] - armchair["center"]["y"]
        front = armchair["front_world"]
        dot = front["dx"] * dx + front["dy"] * dy
        self.assertGreater(dot, 0)
        self.assertTrue(result["hard_valid"])

    def test_support_placement_prefers_front_access_safe_slots(self) -> None:
        room_model, world, anchor_solution = self._support_slot_world(
            side_options=["head", "left"],
            droppable_support=False,
            protected_support=False,
        )

        results = _place_support_objects_for_solution(
            solution=anchor_solution,
            world=world,
            room_model=room_model,
            relation_plan=None,
            grid_mm=50,
            max_solutions=4,
        )

        self.assertTrue(results)
        verify = _verify_object_level_solution(
            solution={"placed_objects": results[0]["placed_objects"]},
            world=world,
            room_model=room_model,
            relation_plan=None,
        )
        self.assertTrue(verify["front_access_ok"])

    def test_droppable_support_drops_dirty_slots_before_repair(self) -> None:
        room_model, world, anchor_solution = self._support_slot_world(
            side_options=["head"],
            droppable_support=True,
            protected_support=False,
        )

        results = _place_support_objects_for_solution(
            solution=anchor_solution,
            world=world,
            room_model=room_model,
            relation_plan=None,
            grid_mm=50,
            max_solutions=4,
        )

        self.assertTrue(results)
        placed_ids = {
            str(row.get("object_id") or "")
            for row in results[0]["placed_objects"]
            if isinstance(row, dict)
        }
        self.assertNotIn("floor_lamp", placed_ids)
        self.assertEqual(
            results[0]["dropped_inventory_by_cluster"],
            {
                "main_seating": [
                    {"object_id": "floor_lamp", "reason": "support_slot_failed"}
                ]
            },
        )

    def test_optional_bedside_nightstand_drop_keeps_bed_layout_complete(self) -> None:
        room_model = {
            "room": {
                "room_id": "room_1",
                "room_type": "bedroom",
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 2200, "y": 0},
                    {"x": 2200, "y": 2000},
                    {"x": 0, "y": 2000},
                ],
                "wall_thickness_mm": 0,
            },
            "regions": [],
        }
        cluster = _object_program_cluster(
            cluster_id="sleep_core",
            object_id="bed",
            category="bed",
            dims={"L": 1960, "W": 1690, "H": 480},
            support_edges=[
                {
                    "object_id": "nightstand",
                    "relative_to": "bed",
                    "side_options": ["head_left", "head_right"],
                    "gap_min": 0,
                    "gap_max": 100,
                    "support_role": "side_support",
                    "band_intent": "beside_base",
                    "orientation": "same_direction",
                }
            ],
        )
        object_program = cluster["object_program"]
        object_program["protected_ids"] = ["bed"]
        object_program["droppable_ids"] = ["nightstand"]
        object_program["required_object_ids"] = ["bed"]
        object_program["optional_object_ids"] = ["nightstand"]
        object_program["object_specs_by_id"]["nightstand"]["rep_dims_mm"] = {
            "L": 430,
            "W": 350,
            "H": 530,
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=None,
            cluster_constraints=None,
            grid_mm=50,
        )
        anchor_solution = {
            "anchor_solution": {
                "sleep_core": {
                    "cluster_id": "sleep_core",
                    "object_id": "bed",
                    "x": 120,
                    "y": 155,
                    "rot": 0,
                    "rect": (120, 155, 2080, 1845),
                    "candidate_score": 0.0,
                }
            },
            "anchor_score": 0.0,
            "dropped_inventory_by_cluster": {},
        }

        results = _place_support_objects_for_solution(
            solution=anchor_solution,
            world=world,
            room_model=room_model,
            relation_plan=None,
            grid_mm=50,
            max_solutions=4,
        )

        self.assertTrue(results)
        placed_ids = {
            str(row.get("object_id") or "")
            for row in results[0]["placed_objects"]
            if isinstance(row, dict)
        }
        self.assertEqual(placed_ids, {"bed"})
        self.assertEqual(
            results[0]["dropped_inventory_by_cluster"],
            {
                "sleep_core": [
                    {"object_id": "nightstand", "reason": "support_slot_failed"}
                ]
            },
        )
        verify = _verify_object_level_solution(
            solution={
                "placed_objects": results[0]["placed_objects"],
                "dropped_inventory_by_cluster": results[0][
                    "dropped_inventory_by_cluster"
                ],
            },
            world=world,
            room_model=room_model,
            relation_plan=None,
        )

        self.assertTrue(verify["hard_valid"])
        self.assertTrue(verify["complete"])

    def test_protected_support_keeps_repairable_dirty_slots(self) -> None:
        room_model, world, anchor_solution = self._support_slot_world(
            side_options=["head"],
            droppable_support=True,
            protected_support=True,
        )

        results = _place_support_objects_for_solution(
            solution=anchor_solution,
            world=world,
            room_model=room_model,
            relation_plan=None,
            grid_mm=50,
            max_solutions=4,
        )

        self.assertTrue(results)
        placed_ids = {
            str(row.get("object_id") or "")
            for row in results[0]["placed_objects"]
            if isinstance(row, dict)
        }
        self.assertIn("floor_lamp", placed_ids)
        self.assertEqual(results[0]["dropped_inventory_by_cluster"], {})

    def test_living_room_case_keeps_hard_valid_gallery_gate(self) -> None:
        case_root = REPO_ROOT / "cases" / "user_f1a4e69d3672aab8_20260521T142836Z"
        if not case_root.exists():
            self.skipTest("local case artifact is unavailable")

        result = self._solve_case_first_concept(case_root)

        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["geometry_valid"])
        self.assertTrue(result["hard_valid"])
        self.assertTrue(result["gallery_eligible"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["coverage_ratio"], 1.0)

    def test_bedroom_case_returns_hard_valid_gallery_candidate(self) -> None:
        case_root = REPO_ROOT / "cases" / "user_f1a4e69d3672aab8_20260521T140710Z"
        if not case_root.exists():
            self.skipTest("local case artifact is unavailable")

        result = self._solve_case_first_concept(case_root)

        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["geometry_valid"])
        self.assertTrue(result["hard_valid"])
        self.assertTrue(result["gallery_eligible"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["coverage_ratio"], 1.0)

    def test_regression_mixed_media_case_keeps_later_concepts_viable(self) -> None:
        case_root = REPO_ROOT / "cases" / "user_f1a4e69d3672aab8_20260522T150736Z"
        if not case_root.exists():
            self.skipTest("local case artifact is unavailable")

        room_model = json.loads((case_root / "01_room_interpreter.json").read_text())
        merged_clusters = json.loads((case_root / "04_cluster_merged.json").read_text())
        cluster_constraints = json.loads(
            (case_root / "02_cluster_forge.json").read_text()
        )
        concept_bundle = json.loads(
            (case_root / "05c_seed_relation_plans.json").read_text()
        )
        room = (
            room_model.get("room") if isinstance(room_model.get("room"), dict) else {}
        )
        room_type = str(room.get("room_type") or room_model.get("room_type") or "")

        status_by_concept: dict[str, str] = {}
        top_gate_reasons_by_concept: dict[str, list[str]] = {}
        for concept in concept_bundle["concepts"]:
            concept_id = str(concept.get("concept_id") or "")
            relation_plan = solver_plan_from_concept(
                concept=concept,
                room_model_json=room_model,
                room_type=room_type,
            )
            result = solve_object_level_layout(
                room_model=room_model,
                merged_clusters=merged_clusters,
                relation_plan=relation_plan,
                cluster_constraints=cluster_constraints,
                grid_mm=50,
                max_rounds=3,
            )
            report = result.get("solver_degradation_report")
            top_reasons = (
                report.get("top_quality_gate_reasons")
                if isinstance(report, dict)
                else []
            )
            status_by_concept[concept_id] = str(result.get("status") or "")
            top_gate_reasons_by_concept[concept_id] = [
                str(row.get("reason") or "")
                for row in top_reasons
                if isinstance(row, dict)
            ]

        for concept_id in ("concept_03", "concept_04", "concept_05"):
            with self.subTest(concept_id=concept_id):
                self.assertEqual(status_by_concept.get(concept_id), "OK")
                self.assertNotIn(
                    "required_face_contract_failed",
                    top_gate_reasons_by_concept.get(concept_id, []),
                )

    def test_protected_center_overlap_is_prefiltered_before_ranking(self) -> None:
        merged_clusters = {
            "clusters": [
                _object_program_cluster(
                    cluster_id="main_seating",
                    object_id="sofa",
                    category="sofa",
                    dims={"L": 1800, "W": 850, "H": 800},
                    support_edges=[
                        {
                            "object_id": "coffee_table",
                            "relative_to": "sofa",
                            "side_options": ["head"],
                            "gap_min": 100,
                            "gap_max": 100,
                            "support_role": "frontal_support",
                            "band_intent": "front_band",
                        }
                    ],
                )
            ]
        }
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "main_seating": {
                    "zone_assignment": "center_lounge",
                    "preferred_region_ids": ["center_lounge"],
                }
            },
            "macro_concept": {
                "macro_constraints": {
                    "protected_topology": [
                        {
                            "region": "center_openness_core",
                            "zone_type": "center_openness_core",
                            "priority": "high",
                            "enforcement": "hard_soft",
                            "max_overlap_ratio": 0.0,
                            "applies_to": ["core_clusters", "interaction_hulls"],
                            "violation_severity": "blocking",
                        }
                    ]
                }
            },
        }

        result = solve_object_level_layout(
            room_model=_room_model(),
            merged_clusters=merged_clusters,
            relation_plan=relation_plan,
        )

        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["geometry_valid"])
        self.assertTrue(result["complete"])
        self.assertTrue(result["hard_valid"])
        self.assertTrue(result["gallery_eligible"])
        objects = {
            str(row["object_id"]): row for row in result["absolute_layout"]["objects"]
        }
        self.assertIn("coffee_table", objects)
        verify = result["verify_summary"]
        self.assertNotIn(
            "blocking_protected_region_overlap", verify["quality_gate_reasons"]
        )

    def test_fully_blocking_protected_region_returns_unsat_without_layout(self) -> None:
        room_model = _room_model()
        regions = room_model["regions"]
        self.assertIsInstance(regions, list)
        regions.append(
            {
                "id": "whole_room",
                "bbox": {"min_x": 0, "min_y": 0, "max_x": 6000, "max_y": 4000},
            }
        )
        merged_clusters = {
            "clusters": [
                _object_program_cluster(
                    cluster_id="main_seating",
                    object_id="sofa",
                    category="sofa",
                    dims={"L": 1800, "W": 850, "H": 800},
                )
            ]
        }
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "main_seating": {
                    "zone_assignment": "whole_room",
                    "preferred_region_ids": ["whole_room"],
                }
            },
            "macro_concept": {
                "macro_constraints": {
                    "protected_topology": [
                        {
                            "region": "whole_room",
                            "zone_type": "primary_circulation_corridor",
                            "priority": "high",
                            "enforcement": "hard_soft",
                            "max_overlap_ratio": 0.0,
                            "applies_to": ["core_clusters"],
                            "violation_severity": "blocking",
                        }
                    ]
                }
            },
        }

        result = solve_object_level_layout(
            room_model=room_model,
            merged_clusters=merged_clusters,
            relation_plan=relation_plan,
        )

        self.assertEqual(result["status"], "UNSAT")
        self.assertNotIn("absolute_layout", result)
        self.assertEqual(
            result["solver_debug"]["anchor_candidate_counts"]["main_seating"], 0
        )

    def test_no_compatible_anchor_set_reports_pairwise_overlap_diagnostics(
        self,
    ) -> None:
        merged_clusters = {
            "clusters": [
                _object_program_cluster(
                    cluster_id="sleep_core",
                    object_id="bed",
                    category="bed",
                    dims={"L": 1800, "W": 900, "H": 600},
                ),
                _object_program_cluster(
                    cluster_id="storage_core",
                    object_id="wardrobe",
                    category="wardrobe",
                    dims={"L": 1600, "W": 600, "H": 2200},
                ),
            ]
        }

        def fake_anchor_candidates(
            *,
            cluster_program: dict[str, object],
            room_model: dict[str, object],
            relation_plan: dict[str, object] | None,
            world: dict[str, object],
            grid_mm: int,
        ) -> list[dict[str, object]]:
            _ = room_model, relation_plan, world, grid_mm
            cluster_id = str(cluster_program.get("cluster_id") or "")
            object_id = "bed" if cluster_id == "sleep_core" else "wardrobe"
            rect = (
                (1000, 1000, 2600, 2200)
                if cluster_id == "sleep_core"
                else (1200, 1100, 2500, 2300)
            )
            return [
                {
                    "cluster_id": cluster_id,
                    "object_id": object_id,
                    "category": object_id,
                    "x": rect[0],
                    "y": rect[1],
                    "rot": 0,
                    "rect": rect,
                    "candidate_score": 0.0,
                }
            ]

        with patch.dict(
            solve_object_level_layout.__globals__,
            {"_generate_anchor_pose_candidates": fake_anchor_candidates},
        ):
            result = solve_object_level_layout(
                room_model=_room_model(),
                merged_clusters=merged_clusters,
                relation_plan={},
                cluster_constraints=None,
            )

        self.assertEqual(result["status"], "UNSAT")
        self.assertEqual(result["failure_kind"], "no_compatible_anchor_set")
        report = result["solver_degradation_report"]
        self.assertEqual(report["failure_kind"], "no_compatible_anchor_set")
        incompatible = result["solver_debug"]["pairwise_incompatible_anchor_pairs"]
        self.assertEqual(
            incompatible,
            [
                {
                    "cluster_a": "sleep_core",
                    "cluster_b": "storage_core",
                    "candidate_pairs": 1,
                    "non_overlap_pairs": 0,
                    "overlap_pairs": 1,
                }
            ],
        )

    def test_anchor_pair_compatibility_reports_non_overlapping_pairs(self) -> None:
        report = _anchor_pair_compatibility_report(
            anchor_order=["sleep_core", "storage_core"],
            anchor_candidates_by_cluster={
                "sleep_core": [{"rect": (0, 0, 1000, 1000)}],
                "storage_core": [{"rect": (1500, 0, 2500, 1000)}],
            },
        )

        self.assertEqual(report[0]["candidate_pairs"], 1)
        self.assertEqual(report[0]["non_overlap_pairs"], 1)
        self.assertEqual(report[0]["overlap_pairs"], 0)

    def test_object_rect_is_usable_rejects_blocking_protected_region(self) -> None:
        relation_plan = {
            "macro_concept": {
                "macro_constraints": {
                    "protected_topology": [
                        {
                            "region": "center_lounge",
                            "zone_type": "center_openness_core",
                            "priority": "high",
                            "enforcement": "hard_soft",
                            "max_overlap_ratio": 0.0,
                            "applies_to": ["core_clusters"],
                            "violation_severity": "blocking",
                        }
                    ]
                }
            }
        }
        world = _build_object_solver_world(
            room_model=_room_model(),
            merged_clusters={"clusters": []},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=100,
        )

        self.assertFalse(
            _object_rect_is_usable(
                (2000, 1500, 3000, 2300),
                (),
                world,
                row={
                    "cluster_id": "main_seating",
                    "object_id": "sofa",
                    "anchor_kind": "center",
                },
            )
        )

    def test_wall_contact_requires_front_inward(self) -> None:
        issues = _object_level_orientation_issues(
            placed_objects=[
                {
                    "cluster_id": "main_seating",
                    "object_id": "sofa",
                    "priority": "anchor",
                    "relative_to": None,
                    "rect": (1000, 0, 2800, 850),
                    "front_world": {"dx": 0, "dy": -1},
                }
            ],
            world={"room_bbox": (0, 0, 6000, 4000), "clusters_by_id": {}},
            relation_plan={},
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["reason"], "wall_contact_inward")

    def test_face_each_other_without_alternative_is_required(self) -> None:
        relation_plan = {
            "cluster_directional_relations": [
                {
                    "a": "main_seating",
                    "b": "media",
                    "relation": "face_each_other",
                    "priority": "high",
                    "reason": "test",
                }
            ]
        }
        issues = _object_level_face_pair_issues(
            placed_objects=[
                {
                    "cluster_id": "main_seating",
                    "object_id": "sofa",
                    "priority": "anchor",
                    "rect": (1000, 1000, 2800, 1800),
                    "front_world": {"dx": -1, "dy": 0},
                },
                {
                    "cluster_id": "media",
                    "object_id": "tv_console",
                    "priority": "anchor",
                    "rect": (3600, 1000, 4600, 1400),
                    "front_world": {"dx": -1, "dy": 0},
                },
            ],
            relation_plan=relation_plan,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["reason"], "required_face_each_other_failed")

    def test_face_pair_missing_dropped_cluster_is_ignored(self) -> None:
        relation_plan = {
            "cluster_directional_relations": [
                {
                    "a": "main_seating",
                    "b": "lounge_reading",
                    "relation": "face_each_other",
                    "priority": "high",
                    "reason": "optional trial cluster may be dropped",
                }
            ]
        }

        issues = _object_level_face_pair_issues(
            placed_objects=[
                {
                    "cluster_id": "main_seating",
                    "object_id": "sofa",
                    "priority": "anchor",
                    "rect": (1000, 1000, 2800, 1800),
                    "front_world": {"dx": 1, "dy": 0},
                }
            ],
            relation_plan=relation_plan,
        )

        self.assertEqual(issues, [])

    def test_face_each_other_rejects_weak_diagonal_alignment(self) -> None:
        relation_plan = {
            "cluster_directional_relations": [
                {
                    "a": "main_seating",
                    "b": "media",
                    "relation": "face_each_other",
                    "priority": "high",
                    "reason": "test",
                }
            ]
        }
        issues = _object_level_face_pair_issues(
            placed_objects=[
                {
                    "cluster_id": "main_seating",
                    "object_id": "sofa",
                    "priority": "anchor",
                    "rect": (1950, 0, 4010, 960),
                    "front_world": {"dx": 0, "dy": 1},
                },
                {
                    "cluster_id": "media",
                    "object_id": "tv_console",
                    "priority": "anchor",
                    "rect": (5250, 1350, 5630, 2610),
                    "front_world": {"dx": -1, "dy": 0},
                },
            ],
            relation_plan=relation_plan,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["reason"], "required_face_each_other_failed")

    def test_object_viewing_pair_verifies_exact_requested_objects(self) -> None:
        relation_plan = {
            "object_viewing_pairs": [
                {
                    "viewer_cluster_id": "sleep_core",
                    "viewer_object_id": "bed",
                    "viewer_object_type": "bed",
                    "target_cluster_id": "media_optional",
                    "target_object_id": "tv_console",
                    "target_object_type": "tv_console",
                    "relation": "view_target",
                    "required": True,
                }
            ]
        }

        issues = _object_level_face_pair_issues(
            placed_objects=[
                {
                    "cluster_id": "sleep_core",
                    "object_id": "bed",
                    "priority": "anchor",
                    "rect": (0, 1050, 1690, 2960),
                    "front_world": {"dx": 1, "dy": 0},
                },
                {
                    "cluster_id": "media_optional",
                    "object_id": "tv_console",
                    "priority": "anchor",
                    "rect": (2500, 3500, 3760, 3880),
                    "front_world": {"dx": 0, "dy": -1},
                },
                {
                    "cluster_id": "storage_core",
                    "object_id": "wardrobe",
                    "priority": "anchor",
                    "rect": (3900, 1150, 4410, 2900),
                    "front_world": {"dx": -1, "dy": 0},
                },
            ],
            relation_plan=relation_plan,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["viewer_object_id"], "bed")
        self.assertEqual(issues[0]["target_object_id"], "tv_console")
        self.assertEqual(issues[0]["reason"], "view_target_contract_failed")

    def test_view_target_allows_bed_media_diagonal_alignment(self) -> None:
        placed_objects = [
            {
                "cluster_id": "sleep_core",
                "object_id": "bed",
                "priority": "anchor",
                "rect": (0, 0, 100, 100),
                "front_world": {"dx": 0, "dy": 1},
            },
            {
                "cluster_id": "media_optional",
                "object_id": "tv_console",
                "priority": "anchor",
                "rect": (615, 789, 715, 889),
                "front_world": {"dx": 0, "dy": -1},
            },
        ]
        view_target_plan = {
            "object_viewing_pairs": [
                {
                    "viewer_cluster_id": "sleep_core",
                    "viewer_object_id": "bed",
                    "viewer_object_type": "bed",
                    "target_cluster_id": "media_optional",
                    "target_object_id": "tv_console",
                    "target_object_type": "tv_console",
                    "relation": "view_target",
                    "required": True,
                }
            ]
        }
        mutual_face_plan = {
            "object_viewing_pairs": [
                {
                    "viewer_cluster_id": "sleep_core",
                    "viewer_object_id": "bed",
                    "viewer_object_type": "bed",
                    "target_cluster_id": "media_optional",
                    "target_object_id": "tv_console",
                    "target_object_type": "tv_console",
                    "relation": "face_each_other",
                    "required": True,
                }
            ]
        }

        self.assertEqual(
            _object_level_face_pair_issues(
                placed_objects=placed_objects,
                relation_plan=view_target_plan,
            ),
            [],
        )
        issues = _object_level_face_pair_issues(
            placed_objects=placed_objects,
            relation_plan=mutual_face_plan,
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["reason"], "required_face_each_other_failed")

    def test_object_viewing_pair_score_prefers_tv_over_storage_decoy(self) -> None:
        relation_plan = {
            "object_viewing_pairs": [
                {
                    "viewer_cluster_id": "sleep_core",
                    "viewer_object_id": "bed",
                    "viewer_object_type": "bed",
                    "target_cluster_id": "media_optional",
                    "target_object_id": "tv_console",
                    "target_object_type": "tv_console",
                    "relation": "view_target",
                    "required": True,
                }
            ]
        }
        weak_solution = {
            "sleep_core": {
                "cluster_id": "sleep_core",
                "object_id": "bed",
                "rect": (0, 1050, 1690, 2960),
                "front_world": {"dx": 1, "dy": 0},
                "anchor_kind": "left_wall",
            },
            "media_optional": {
                "cluster_id": "media_optional",
                "object_id": "tv_console",
                "rect": (2500, 3500, 3760, 3880),
                "front_world": {"dx": 0, "dy": -1},
                "anchor_kind": "top_wall",
            },
            "storage_core": {
                "cluster_id": "storage_core",
                "object_id": "wardrobe",
                "rect": (3900, 1150, 4410, 2900),
                "front_world": {"dx": -1, "dy": 0},
                "anchor_kind": "right_wall",
            },
        }
        strong_solution = {
            "sleep_core": weak_solution["sleep_core"],
            "media_optional": {
                "cluster_id": "media_optional",
                "object_id": "tv_console",
                "rect": (3150, 1700, 4410, 2080),
                "front_world": {"dx": -1, "dy": 0},
                "anchor_kind": "right_wall",
            },
            "storage_core": {
                "cluster_id": "storage_core",
                "object_id": "wardrobe",
                "rect": (2500, 3500, 3010, 4000),
                "front_world": {"dx": 0, "dy": -1},
                "anchor_kind": "top_wall",
            },
        }

        self.assertGreater(
            _anchor_pair_solution_score(strong_solution, relation_plan),
            _anchor_pair_solution_score(weak_solution, relation_plan),
        )

    def test_face_each_other_with_directional_alternative_is_not_required(self) -> None:
        relation_plan = {
            "cluster_directional_relations": [
                {
                    "a": "main_seating",
                    "b": "media",
                    "relation": "face_each_other",
                    "priority": "high",
                    "reason": "test",
                },
                {
                    "a": "media",
                    "b": "main_seating",
                    "relation": "perpendicular_alignment",
                    "priority": "medium",
                    "reason": "test alternative",
                },
            ]
        }

        self.assertEqual(_object_level_required_face_pairs(relation_plan), set())

    def test_mixed_media_pair_with_inferred_face_cluster_is_not_required(self) -> None:
        relation_plan = {
            "layout_intent_profile": {
                "focus_mode": "mixed",
                "primary_cluster_id": "sleep_core",
                "secondary_cluster_id": "media_optional",
            },
            "cluster_orientations": [
                {
                    "cluster_id": "sleep_core",
                    "target_cluster_id": "media_optional",
                    "intents": ["back_to_wall", "face_cluster"],
                    "priority": "high",
                },
                {
                    "cluster_id": "media_optional",
                    "target_cluster_id": "sleep_core",
                    "intents": ["back_to_wall", "face_cluster"],
                    "priority": "high",
                },
            ],
        }

        issues = _object_level_face_pair_issues(
            placed_objects=[
                {
                    "cluster_id": "sleep_core",
                    "object_id": "bed",
                    "priority": "anchor",
                    "rect": (0, 1050, 1690, 2960),
                    "front_world": {"dx": 1, "dy": 0},
                },
                {
                    "cluster_id": "media_optional",
                    "object_id": "tv_console",
                    "priority": "anchor",
                    "rect": (1900, 3500, 3160, 3880),
                    "front_world": {"dx": 0, "dy": -1},
                },
            ],
            relation_plan=relation_plan,
        )

        self.assertEqual(issues, [])
        self.assertEqual(_object_level_required_face_pairs(relation_plan), set())

    def test_mixed_media_pair_with_buffered_support_is_not_face_required(self) -> None:
        relation_plan = {
            "layout_intent_profile": {
                "focus_mode": "mixed",
                "primary_cluster_id": "main_seating",
                "secondary_cluster_id": "media",
            },
            "concept_readiness_requirements": {
                "required_pair_contracts": ["buffered_support"],
            },
            "cluster_orientations": [
                {
                    "cluster_id": "main_seating",
                    "target_cluster_id": "media",
                    "intents": ["face_cluster"],
                    "priority": "high",
                },
                {
                    "cluster_id": "media",
                    "target_cluster_id": "main_seating",
                    "intents": ["face_cluster"],
                    "priority": "high",
                },
            ],
        }

        issues = _object_level_face_pair_issues(
            placed_objects=[
                {
                    "cluster_id": "main_seating",
                    "object_id": "sofa",
                    "priority": "anchor",
                    "rect": (5000, 950, 5960, 3010),
                    "front_world": {"dx": -1, "dy": 0},
                },
                {
                    "cluster_id": "media",
                    "object_id": "tv_console",
                    "priority": "anchor",
                    "rect": (2350, 3450, 3610, 3830),
                    "front_world": {"dx": 0, "dy": -1},
                },
            ],
            relation_plan=relation_plan,
        )

        self.assertEqual(issues, [])

    def test_cluster_face_target_fallback_does_not_crash(self) -> None:
        merged_clusters = {
            "clusters": [
                _object_program_cluster(
                    cluster_id="main_seating",
                    object_id="sofa",
                    category="sofa",
                    dims={"L": 1000, "W": 500, "H": 800},
                ),
                _object_program_cluster(
                    cluster_id="media",
                    object_id="tv_console",
                    category="tv_console",
                    dims={"L": 800, "W": 300, "H": 600},
                ),
            ]
        }
        relation_plan = {
            "layout_intent_profile": {
                "primary_cluster_id": "main_seating",
                "secondary_cluster_id": "media",
            },
            "anchor_layout_hints_by_cluster": {
                "main_seating": {
                    "zone_assignment": "center_lounge",
                    "preferred_region_ids": ["center_lounge"],
                },
                "media": {
                    "zone_assignment": "top_storage",
                    "preferred_region_ids": ["top_storage"],
                },
            },
            "cluster_orientations": [
                {
                    "cluster_id": "main_seating",
                    "intents": ["face_cluster"],
                    "target_cluster_id": "media",
                    "priority": "high",
                },
                {
                    "cluster_id": "media",
                    "intents": ["back_to_wall", "face_cluster"],
                    "target_cluster_id": "main_seating",
                    "priority": "high",
                },
            ],
            "cluster_relations": [
                {
                    "a": "main_seating",
                    "b": "media",
                    "relation": "near",
                    "priority": "high",
                }
            ],
        }

        result = solve_object_level_layout(
            room_model=_room_model(),
            merged_clusters=merged_clusters,
            relation_plan=relation_plan,
        )

        self.assertIn(result["status"], {"OK", "PARTIAL"})
        self.assertEqual(result["solver_kind"], "object_level_anchor_first")

    def test_drop_secondary_support_degrades_unprotected_supports(self) -> None:
        object_program = {
            "members": ["sofa", "coffee_table", "side_table", "floor_lamp"],
            "protected_ids": ["sofa", "coffee_table"],
            "droppable_ids": [],
            "optional_object_ids": [],
            "degradation_ladder": ["shrink_optional_support", "drop_secondary_support"],
            "support_edges": [
                {"object_id": "coffee_table", "relative_to": "sofa"},
                {"object_id": "side_table", "relative_to": "sofa"},
                {"object_id": "floor_lamp", "relative_to": "sofa"},
            ],
            "object_specs_by_id": {
                "sofa": {"role": "dominant_anchor", "priority": "anchor"},
                "coffee_table": {"role": "support", "priority": "secondary"},
                "side_table": {"role": "support", "priority": "secondary"},
                "floor_lamp": {"role": "support", "priority": "secondary"},
            },
        }

        droppable = _droppable_object_ids(object_program)

        self.assertEqual(droppable, {"side_table", "floor_lamp"})

    def test_surface_without_required_seat_fails_gallery_eligibility(self) -> None:
        cluster_program = _object_program_cluster(
            cluster_id="work_study",
            object_id="desk",
            category="desk",
            dims={"L": 1000, "W": 500, "H": 760},
            support_edges=[
                {
                    "object_id": "chair",
                    "relative_to": "desk",
                    "support_role": "secondary_seat",
                    "band_intent": "flank_band",
                    "orientation": "face_base",
                }
            ],
        )
        desk = _materialize_object_row(cluster_program, "desk", 1000, 1000, 0)

        verify = _verify_object_level_solution(
            solution={
                "placed_objects": [desk],
                "dropped_inventory_by_cluster": {
                    "work_study": [
                        {"object_id": "chair", "reason": "support_slot_failed"}
                    ]
                },
            },
            world={
                "room_bbox": (0, 0, 6000, 4000),
                "room_polygon": (),
                "clusters_by_id": {"work_study": cluster_program},
                "protected_regions": [],
                "cluster_forbidden_regions": [],
            },
            room_model={},
            relation_plan={},
        )

        self.assertFalse(verify["hard_valid"])
        self.assertFalse(verify["gallery_eligible"])
        self.assertIn("missing_surface_seat_pair", verify["quality_gate_reasons"])

    def test_frontal_support_ignores_side_band_options(self) -> None:
        edge = {
            "object_id": "coffee_table",
            "side_options": ["head", "left", "right"],
            "support_role": "frontal_support",
            "band_intent": "front_band",
        }

        self.assertEqual(_normalized_support_side_options(edge), ["head"])

    def test_side_support_preserves_explicit_head_side_options(self) -> None:
        edge = {
            "object_id": "nightstand",
            "side_options": ["head_left", "head_right"],
            "support_role": "side_support",
            "band_intent": "beside_base",
        }

        self.assertEqual(
            _normalized_support_side_options(edge),
            ["head_left", "head_right"],
        )

    def test_bedside_nightstand_head_options_place_beside_bed_head(self) -> None:
        cluster_program = _object_program_cluster(
            cluster_id="sleep_core",
            object_id="bed",
            category="bed",
            dims={"L": 1960, "W": 1690, "H": 480},
            support_edges=[
                {
                    "object_id": "nightstand",
                    "relative_to": "bed",
                    "side_options": ["head_left", "head_right"],
                    "gap_min": 0,
                    "gap_max": 100,
                    "support_role": "side_support",
                    "band_intent": "beside_base",
                    "orientation": "same_direction",
                }
            ],
        )
        cluster_program["object_program"]["object_specs_by_id"]["nightstand"][
            "rep_dims_mm"
        ] = {"L": 430, "W": 350, "H": 530}
        base_row = {
            "cluster_id": "sleep_core",
            "object_id": "bed",
            "category": "bed",
            "rect": (2650, 1850, 4610, 3540),
            "rot": 180,
            "front_token": "top",
            "front_world": {"dx": 0, "dy": -1},
        }

        slots = _support_slot_candidates(
            cluster_program=cluster_program,
            object_id="nightstand",
            base_row=base_row,
            edge=cluster_program["object_program"]["support_edges"][0],
            grid_mm=50,
        )

        self.assertTrue(slots)
        bed_rect = base_row["rect"]
        slot_rects = [slot["rect"] for slot in slots]
        self.assertTrue(
            all(rect[2] <= bed_rect[0] or rect[0] >= bed_rect[2] for rect in slot_rects)
        )
        self.assertLessEqual(min(rect[1] for rect in slot_rects), bed_rect[1] + 450)
        self.assertGreaterEqual(min(rect[1] for rect in slot_rects), bed_rect[1])

    def test_wall_support_preserves_forward_escape_options(self) -> None:
        edge = {
            "object_id": "bookshelf",
            "side_options": ["left", "right", "head_left", "head_right"],
            "support_role": "wall_support",
            "band_intent": "wall_band",
        }

        self.assertEqual(
            _normalized_support_side_options(edge),
            ["left", "right", "head_left", "head_right"],
        )

    def test_blocking_protected_region_overlap_fails_hard_valid(self) -> None:
        verify = _verify_object_level_solution(
            solution={
                "placed_objects": [
                    {
                        "cluster_id": "storage",
                        "object_id": "bookshelf",
                        "priority": "anchor",
                        "rect": (0, 0, 800, 800),
                        "front_world": {"dx": 0, "dy": 1},
                    }
                ]
            },
            world={
                "room_bbox": (0, 0, 6000, 4000),
                "clusters_by_id": {"storage": {}},
                "protected_regions": [
                    {
                        "region_id": "door_1_entry_clearance",
                        "bbox": (0, 0, 1000, 1000),
                        "max_overlap_ratio": 0.0,
                        "priority": "high",
                        "enforcement": "hard",
                        "violation_severity": "blocking",
                        "zone_type": "entry_landing",
                    }
                ],
            },
            room_model={},
            relation_plan={},
        )

        self.assertFalse(verify["hard_valid"])
        self.assertFalse(verify["gallery_eligible"])
        self.assertIn(
            "blocking_protected_region_overlap", verify["quality_gate_reasons"]
        )

    def test_explicit_support_front_access_blocks_occluded_bookshelf(self) -> None:
        cluster_program = {
            "cluster_id": "storage_core",
            "object_program": {
                "object_specs_by_id": {
                    "wardrobe": {
                        "object_id": "wardrobe",
                        "category": "wardrobe",
                        "role": "dominant_anchor",
                        "priority": "anchor",
                        "rep_dims_mm": {"L": 1750, "W": 510, "H": 2020},
                        "allowed_rotations": [180],
                        "front": "top",
                    },
                    "bookshelf": {
                        "object_id": "bookshelf",
                        "category": "bookshelf",
                        "role": "support",
                        "priority": "support",
                        "rep_dims_mm": {"L": 630, "W": 250, "H": 2220},
                        "allowed_rotations": [270],
                        "front": "top",
                    },
                }
            },
            "cluster_rules": {
                "access_requirements": [
                    {"id": "wardrobe", "type": "front_clearance", "required": True},
                    {"id": "bookshelf", "type": "front_clearance", "required": True},
                ]
            },
        }
        wardrobe = _materialize_object_row(cluster_program, "wardrobe", 2400, 3450, 180)
        bookshelf = _materialize_object_row(
            cluster_program,
            "bookshelf",
            4250,
            3200,
            270,
            relative_to="wardrobe",
        )
        world = {
            "room_bbox": (0, 0, 4500, 4000),
            "clusters_by_id": {"storage_core": cluster_program},
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        self.assertTrue(bookshelf["requires_front_access"])
        penalty = _support_slot_relaxation_penalty(
            candidate=bookshelf,
            placed_objects=[wardrobe],
            protected_regions=[],
            world=world,
            relation_plan={},
        )
        self.assertGreaterEqual(penalty, 1400.0)

        verify = _verify_object_level_solution(
            solution={"placed_objects": [wardrobe, bookshelf]},
            world=world,
            room_model={},
            relation_plan={},
        )

        self.assertFalse(verify["front_access_ok"])
        self.assertIn("front_access_zone_blocked", verify["quality_gate_reasons"])
        self.assertEqual(
            verify["blocking_functional_geometry_issues"][0]["blocked_object_id"],
            "bookshelf",
        )

    def test_softened_front_access_requirement_is_not_blocking(self) -> None:
        cluster_program = {
            "cluster_id": "storage_core",
            "object_program": {
                "object_specs_by_id": {
                    "wardrobe": {
                        "object_id": "wardrobe",
                        "category": "wardrobe",
                        "role": "dominant_anchor",
                        "priority": "anchor",
                        "rep_dims_mm": {"L": 1750, "W": 510, "H": 2020},
                        "allowed_rotations": [180],
                        "front": "top",
                    },
                    "bookshelf": {
                        "object_id": "bookshelf",
                        "category": "bookshelf",
                        "role": "support",
                        "priority": "support",
                        "rep_dims_mm": {"L": 630, "W": 250, "H": 2220},
                        "allowed_rotations": [270],
                        "front": "top",
                    },
                }
            },
            "cluster_rules": {
                "access_requirements": [
                    {"id": "wardrobe", "type": "front_clearance", "required": True},
                    {"id": "bookshelf", "type": "front_clearance", "required": False},
                ]
            },
        }
        wardrobe = _materialize_object_row(cluster_program, "wardrobe", 2400, 3450, 180)
        bookshelf = _materialize_object_row(
            cluster_program,
            "bookshelf",
            4250,
            3200,
            270,
            relative_to="wardrobe",
        )
        world = {
            "room_bbox": (0, 0, 4500, 4000),
            "clusters_by_id": {"storage_core": cluster_program},
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        self.assertFalse(bookshelf["requires_front_access"])
        verify = _verify_object_level_solution(
            solution={"placed_objects": [wardrobe, bookshelf]},
            world=world,
            room_model={},
            relation_plan={},
        )

        self.assertTrue(verify["front_access_ok"])
        self.assertNotIn("front_access_zone_blocked", verify["quality_gate_reasons"])
        self.assertEqual(verify["blocking_functional_geometry_issues"], [])

    def test_geometry_repair_keeps_support_out_of_anchor_front_access(self) -> None:
        cluster_program = {
            "cluster_id": "kitchen_workflow_core",
            "object_program": {
                "object_specs_by_id": {
                    "kitchen_base_cabinet": {
                        "object_id": "kitchen_base_cabinet",
                        "category": "kitchen_base_cabinet",
                        "role": "dominant_anchor",
                        "priority": "anchor",
                        "rep_dims_mm": {"L": 620, "W": 560, "H": 900},
                        "allowed_rotations": [270],
                        "front": "top",
                    },
                    "fridge": {
                        "object_id": "fridge",
                        "category": "fridge",
                        "role": "support",
                        "priority": "secondary",
                        "rep_dims_mm": {"L": 640, "W": 600, "H": 1800},
                        "allowed_rotations": [270],
                        "front": "top",
                    },
                }
            },
            "cluster_rules": {
                "access_requirements": [
                    {
                        "id": "kitchen_base_cabinet",
                        "type": "front_clearance",
                        "required": True,
                    },
                    {"id": "fridge", "type": "front_clearance", "required": True},
                ]
            },
        }
        base_cabinet = _materialize_object_row(
            cluster_program,
            "kitchen_base_cabinet",
            2400,
            1950,
            270,
        )
        blocking_fridge = _materialize_object_row(
            cluster_program,
            "fridge",
            1800,
            1950,
            270,
            relative_to="kitchen_base_cabinet",
        )
        world = {
            "room_bbox": (0, 0, 4500, 4000),
            "room_polygon": (),
            "clusters_by_id": {"kitchen_workflow_core": cluster_program},
            "protected_regions": [],
            "cluster_forbidden_regions": [],
        }

        before = _verify_object_level_solution(
            solution={"placed_objects": [base_cabinet, blocking_fridge]},
            world=world,
            room_model={},
            relation_plan={},
        )
        self.assertFalse(before["front_access_ok"])

        repair = _repair_object_level_solution_geometry(
            solution={"placed_objects": [base_cabinet, blocking_fridge]},
            world=world,
            grid_mm=50,
        )
        assert repair is not None
        repaired_rows = repair["placed_objects"]
        after = _verify_object_level_solution(
            solution={"placed_objects": repaired_rows},
            world=world,
            room_model={},
            relation_plan={},
        )

        self.assertTrue(after["front_access_ok"])
        self.assertEqual(repair["summary"]["status"], "repaired")

    def test_cluster_forbidden_regions_reuse_protected_topology_policy(self) -> None:
        corridor_bbox = (0, 0, 1200, 1800)
        center_bbox = (2200, 1300, 3800, 2700)
        relation_plan = {
            "anchor_region_preferences_by_cluster": {
                "sleep_core": {
                    "forbidden_region_ids": [
                        "entry_to_center_corridor",
                        "center_openness_core",
                    ],
                }
            },
            "macro_concept": {
                "macro_constraints": {
                    "protected_topology": [
                        {
                            "region": "door_1_to_room_center_corridor",
                            "zone_type": "primary_circulation_corridor",
                            "priority": "high",
                            "enforcement": "hard_soft",
                            "max_overlap_ratio": 0.05,
                            "violation_severity": "blocking",
                        },
                        {
                            "region": "center_openness_core",
                            "zone_type": "center_openness_core",
                            "priority": "high",
                            "enforcement": "hard_soft",
                            "max_overlap_ratio": 0.12,
                            "violation_severity": "blocking",
                        },
                    ]
                }
            },
        }
        rows = _collect_cluster_forbidden_regions(
            clusters_by_id={
                "sleep_core": {
                    "object_program": {
                        "zone_claims": {"avoid_regions": ["entry_to_center_corridor"]}
                    },
                    "cluster_rules": {},
                }
            },
            room_model=_room_model(),
            relation_plan=relation_plan,
            region_index={
                "entry_to_center_corridor": corridor_bbox,
                "door_1_to_room_center_corridor": corridor_bbox,
                "center_openness_core": center_bbox,
            },
        )

        rows_by_region = {str(row["region_id"]): row for row in rows}
        self.assertEqual(
            rows_by_region["entry_to_center_corridor"]["zone_type"],
            "primary_circulation_corridor",
        )
        self.assertEqual(
            rows_by_region["entry_to_center_corridor"]["enforcement"], "hard_soft"
        )
        self.assertEqual(
            rows_by_region["entry_to_center_corridor"]["max_overlap_ratio"], 0.05
        )
        self.assertEqual(
            rows_by_region["center_openness_core"]["max_overlap_ratio"], 0.12
        )

    def test_daylight_forbidden_policy_depends_on_core_height(self) -> None:
        low_cluster = _object_program_cluster(
            cluster_id="sleep_core",
            object_id="bed",
            category="bed",
            dims={"L": 1800, "W": 900, "H": 1150},
        )
        high_cluster = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1600, "W": 600, "H": 2200},
        )
        low_cluster["object_program"]["zone_claims"] = {
            "avoid_regions": ["window_1_daylight"]
        }
        high_cluster["object_program"]["zone_claims"] = {
            "avoid_regions": ["window_1_daylight"]
        }

        rows = _collect_cluster_forbidden_regions(
            clusters_by_id={
                "sleep_core": low_cluster,
                "storage_core": high_cluster,
            },
            room_model=_room_model(),
            relation_plan={},
            region_index={"window_1_daylight": (1000, 0, 3600, 900)},
        )

        rows_by_cluster = {str(row["cluster_id"]): row for row in rows}
        self.assertEqual(rows_by_cluster["sleep_core"]["enforcement"], "soft")
        self.assertEqual(
            rows_by_cluster["sleep_core"]["zone_type"], "daylight_quality_soft"
        )
        self.assertEqual(rows_by_cluster["storage_core"]["enforcement"], "hard")

    def test_visual_storage_support_gets_object_specific_window_blocking(self) -> None:
        cluster = _object_program_cluster(
            cluster_id="media",
            object_id="tv_console",
            category="tv_console",
            dims={"L": 1640, "W": 400, "H": 350},
            support_edges=[
                {
                    "object_id": "media_shelf",
                    "relative_to": "tv_console",
                    "side_options": ["left", "right"],
                    "gap_min": 0,
                    "gap_max": 200,
                    "support_role": "wall_support",
                    "band_intent": "wall_band",
                    "orientation": "same_direction",
                }
            ],
        )
        object_program = cluster["object_program"]
        object_program["zone_claims"] = {"avoid_regions": ["window_1_daylight"]}
        object_program["object_specs_by_id"]["media_shelf"]["rep_dims_mm"] = {
            "L": 870,
            "W": 300,
            "H": 1360,
        }

        rows = _collect_cluster_forbidden_regions(
            clusters_by_id={"media": cluster},
            room_model=_room_model(),
            relation_plan={},
            region_index={
                "window_1_clearance": (1100, 3850, 3100, 4000),
                "window_1_daylight": (900, 2600, 3300, 4000),
            },
        )

        shelf_rows = [
            row for row in rows if str(row.get("object_id") or "") == "media_shelf"
        ]
        self.assertEqual(
            {str(row["region_id"]) for row in shelf_rows},
            {"window_1_clearance", "window_1_daylight"},
        )
        self.assertTrue(all(str(row["enforcement"]) == "hard" for row in shelf_rows))
        console_rows = [
            row for row in rows if str(row.get("object_id") or "") == "tv_console"
        ]
        self.assertEqual(
            {str(row["region_id"]) for row in console_rows},
            {"window_1_clearance"},
        )
        cluster_daylight = next(
            row
            for row in rows
            if str(row["region_id"]) == "window_1_daylight"
            and not str(row.get("object_id") or "")
        )
        self.assertEqual(cluster_daylight["enforcement"], "soft")

    def test_low_console_may_use_daylight_region_when_window_clearance_is_clear(
        self,
    ) -> None:
        cluster = _object_program_cluster(
            cluster_id="media",
            object_id="tv_console",
            category="tv_console",
            dims={"L": 1640, "W": 400, "H": 350},
            support_edges=[
                {
                    "object_id": "media_shelf",
                    "relative_to": "tv_console",
                    "side_options": ["left"],
                    "gap_min": 0,
                    "gap_max": 0,
                    "support_role": "wall_support",
                    "band_intent": "wall_band",
                    "orientation": "same_direction",
                }
            ],
        )
        cluster["object_program"]["object_specs_by_id"]["media_shelf"][
            "rep_dims_mm"
        ] = {"L": 870, "W": 300, "H": 1360}
        world = _build_object_solver_world(
            room_model={
                "room": _room_model()["room"],
                "regions": [
                    {
                        "id": "window_1_daylight",
                        "bbox": {
                            "min_x": 1100,
                            "min_y": 3200,
                            "max_x": 3100,
                            "max_y": 4000,
                        },
                    }
                ],
            },
            merged_clusters={"clusters": [cluster]},
            relation_plan={},
            cluster_constraints=None,
            grid_mm=50,
        )
        tv_console = _materialize_object_row(cluster, "tv_console", 1400, 3600, 180)

        verify = _verify_object_level_solution(
            solution={"placed_objects": [tv_console]},
            world=world,
            room_model={},
            relation_plan={},
        )

        self.assertTrue(verify["hard_valid"])
        self.assertEqual(verify["blocking_forbidden_region_issues"], [])

    def test_droppable_visual_storage_support_drops_window_blocking_slot(
        self,
    ) -> None:
        cluster = _object_program_cluster(
            cluster_id="media",
            object_id="tv_console",
            category="tv_console",
            dims={"L": 1000, "W": 300, "H": 350},
            support_edges=[
                {
                    "object_id": "media_shelf",
                    "relative_to": "tv_console",
                    "side_options": ["left"],
                    "gap_min": 0,
                    "gap_max": 0,
                    "support_role": "wall_support",
                    "band_intent": "wall_band",
                    "orientation": "same_direction",
                }
            ],
        )
        object_program = cluster["object_program"]
        object_program["droppable_ids"] = ["media_shelf"]
        object_program["object_specs_by_id"]["media_shelf"]["rep_dims_mm"] = {
            "L": 870,
            "W": 300,
            "H": 1360,
        }
        room_model = {
            "room": _room_model()["room"],
            "regions": [
                {
                    "id": "window_1_clearance",
                    "bbox": {
                        "min_x": 1100,
                        "min_y": 3400,
                        "max_x": 2100,
                        "max_y": 4000,
                    },
                }
            ],
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan={},
            cluster_constraints=None,
            grid_mm=50,
        )
        anchor_solution = {
            "anchor_solution": {
                "media": {
                    "cluster_id": "media",
                    "object_id": "tv_console",
                    "x": 2000,
                    "y": 3600,
                    "rot": 180,
                    "rect": (2000, 3600, 3000, 3900),
                    "candidate_score": 0.0,
                }
            },
            "anchor_score": 0.0,
            "dropped_inventory_by_cluster": {},
        }

        results = _place_support_objects_for_solution(
            solution=anchor_solution,
            world=world,
            room_model=room_model,
            relation_plan={},
            grid_mm=50,
            max_solutions=4,
        )

        self.assertTrue(results)
        placed_ids = {
            str(row.get("object_id") or "")
            for row in results[0]["placed_objects"]
            if isinstance(row, dict)
        }
        self.assertNotIn("media_shelf", placed_ids)
        self.assertEqual(
            results[0]["dropped_inventory_by_cluster"],
            {"media": [{"object_id": "media_shelf", "reason": "support_slot_failed"}]},
        )

    def test_solver_objects_auto_avoid_window_clearance(self) -> None:
        low_cluster = _object_program_cluster(
            cluster_id="sleep_core",
            object_id="bed",
            category="bed",
            dims={"L": 1800, "W": 900, "H": 650},
        )
        tall_cluster = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1600, "W": 600, "H": 2200},
        )
        curtain_cluster = _object_program_cluster(
            cluster_id="window_treatment",
            object_id="curtain",
            category="curtain",
            dims={"L": 2000, "W": 80, "H": 2400},
        )

        rows = _collect_cluster_forbidden_regions(
            clusters_by_id={
                "sleep_core": low_cluster,
                "storage_core": tall_cluster,
                "window_treatment": curtain_cluster,
            },
            room_model=_room_model(),
            relation_plan={},
            region_index={"window_1_clearance": (1000, 3850, 3000, 4000)},
        )

        rows_by_key = {
            (str(row["cluster_id"]), str(row.get("object_id") or "")): row
            for row in rows
        }
        self.assertEqual(
            rows_by_key[("sleep_core", "bed")]["region_id"], "window_1_clearance"
        )
        self.assertNotIn(("window_treatment", "curtain"), rows_by_key)
        self.assertEqual(
            rows_by_key[("storage_core", "wardrobe")]["region_id"],
            "window_1_clearance",
        )
        self.assertEqual(
            rows_by_key[("storage_core", "wardrobe")]["enforcement"], "hard"
        )

    def test_tall_visual_objects_auto_avoid_clearance_and_daylight_regions(
        self,
    ) -> None:
        tall_cluster = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1600, "W": 600, "H": 2200},
        )

        rows = _collect_cluster_forbidden_regions(
            clusters_by_id={"storage_core": tall_cluster},
            room_model=_room_model(),
            relation_plan={},
            region_index={
                "window_1_clearance": (1000, 3850, 3000, 4000),
                "window_1_daylight": (800, 2600, 3200, 4000),
            },
        )

        object_rows = [
            row for row in rows if str(row.get("object_id") or "") == "wardrobe"
        ]
        self.assertEqual(
            {str(row["region_id"]) for row in object_rows},
            {"window_1_clearance", "window_1_daylight"},
        )
        self.assertTrue(all(str(row["enforcement"]) == "hard" for row in object_rows))

    def test_secondary_daylight_relaxation_keeps_window_clearance_hard(
        self,
    ) -> None:
        cluster = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1600, "W": 600, "H": 2200},
        )
        cluster["object_program"]["object_specs_by_id"]["wardrobe"][
            "daylight_quality_relaxed"
        ] = True

        rows = _collect_cluster_forbidden_regions(
            clusters_by_id={"storage_core": cluster},
            room_model=_room_model(),
            relation_plan={},
            region_index={
                "window_1_clearance": (1000, 3850, 3000, 4000),
                "window_1_daylight": (800, 2600, 3200, 4000),
            },
        )

        rows_by_region = {str(row["region_id"]): row for row in rows}
        self.assertEqual(rows_by_region["window_1_clearance"]["enforcement"], "hard")
        self.assertEqual(rows_by_region["window_1_clearance"]["zone_type"], "window_clearance")
        self.assertEqual(rows_by_region["window_1_daylight"]["enforcement"], "soft")
        self.assertEqual(
            rows_by_region["window_1_daylight"]["zone_type"],
            "daylight_quality_soft",
        )

    def test_relaxed_secondary_daylight_overlap_is_soft_verify_issue(self) -> None:
        cluster = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1600, "W": 600, "H": 2200},
        )
        cluster["object_program"]["object_specs_by_id"]["wardrobe"][
            "daylight_quality_relaxed"
        ] = True
        room_model = {
            "room": _room_model()["room"],
            "regions": [
                {
                    "id": "window_1_daylight",
                    "bbox": {
                        "min_x": 800,
                        "min_y": 2600,
                        "max_x": 3200,
                        "max_y": 4000,
                    },
                }
            ],
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan={},
            cluster_constraints=None,
            grid_mm=50,
        )
        wardrobe = _materialize_object_row(cluster, "wardrobe", 1200, 3000, 0)

        verify = _verify_object_level_solution(
            solution={"placed_objects": [wardrobe]},
            world=world,
            room_model=room_model,
            relation_plan={},
        )

        self.assertTrue(verify["hard_valid"])
        self.assertEqual(verify["blocking_forbidden_region_issues"], [])
        self.assertEqual(
            verify["forbidden_region_issues"][0]["zone_type"],
            "daylight_quality_soft",
        )

    def test_tall_anchor_candidates_do_not_overlap_window_clearance(self) -> None:
        room_model = {
            "room": {
                "room_id": "room_1",
                "room_type": "bedroom",
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 4500, "y": 0},
                    {"x": 4500, "y": 4000},
                    {"x": 0, "y": 4000},
                ],
            },
            "regions": [
                {
                    "id": "window_wall_zone",
                    "bbox": {
                        "min_x": 0,
                        "min_y": 3400,
                        "max_x": 4500,
                        "max_y": 4000,
                    },
                },
                {
                    "id": "window_1_clearance",
                    "bbox": {
                        "min_x": 333,
                        "min_y": 3850,
                        "max_x": 2333,
                        "max_y": 4000,
                    },
                },
            ],
        }
        cluster = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 1750, "W": 510, "H": 2020},
        )
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "storage_core": {
                    "zone_assignment": "window_wall_zone",
                    "preferred_region_ids": ["window_wall_zone"],
                }
            }
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["storage_core"],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        window_bbox = (333, 3850, 2333, 4000)
        for candidate in candidates:
            rect = candidate["rect"]
            overlap_w = max(
                0, min(rect[2], window_bbox[2]) - max(rect[0], window_bbox[0])
            )
            overlap_h = max(
                0, min(rect[3], window_bbox[3]) - max(rect[1], window_bbox[1])
            )
            self.assertEqual(overlap_w * overlap_h, 0)

    def test_tall_optional_support_does_not_hard_ban_anchor_from_daylight(
        self,
    ) -> None:
        room_model = {
            "room": {
                "room_id": "room_1",
                "room_type": "living_room",
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 3000, "y": 0},
                    {"x": 3000, "y": 3600},
                    {"x": 0, "y": 3600},
                ],
            },
            "regions": [
                {
                    "id": "window_side_zone",
                    "bbox": {
                        "min_x": 0,
                        "min_y": 2000,
                        "max_x": 3000,
                        "max_y": 3600,
                    },
                },
                {
                    "id": "window_1_clearance",
                    "bbox": {
                        "min_x": 900,
                        "min_y": 3150,
                        "max_x": 2100,
                        "max_y": 3600,
                    },
                },
                {
                    "id": "window_1_daylight",
                    "bbox": {
                        "min_x": 750,
                        "min_y": 2000,
                        "max_x": 2250,
                        "max_y": 3600,
                    },
                },
            ],
        }
        cluster = _object_program_cluster(
            cluster_id="main_seating",
            object_id="sofa",
            category="sofa",
            dims={"L": 2060, "W": 960, "H": 860},
            support_edges=[
                {
                    "object_id": "floor_lamp",
                    "relative_to": "sofa",
                    "side_options": ["left", "right"],
                    "gap_min": 50,
                    "gap_max": 200,
                    "orientation": "same_direction",
                }
            ],
        )
        object_program = cluster["object_program"]
        object_program["droppable_ids"] = ["floor_lamp"]
        object_program["object_specs_by_id"]["floor_lamp"]["rep_dims_mm"] = {
            "L": 350,
            "W": 320,
            "H": 1510,
        }
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "main_seating": {
                    "zone_assignment": "window_side_zone",
                    "committed_region_ids": ["window_side_zone"],
                    "preferred_region_ids": ["window_side_zone"],
                    "anchor_strength": "strong",
                    "placement_behavior": {"front_space": "required"},
                }
            }
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["main_seating"],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        diagnostics = world["_anchor_candidate_preflight_by_cluster"]["main_seating"]
        self.assertEqual(
            diagnostics["auto_window_forbidden_region_ids"],
            ["window_1_clearance"],
        )
        clearance_bbox = (900, 3150, 2100, 3600)
        daylight_bbox = (750, 2000, 2250, 3600)
        self.assertTrue(
            any(
                _rect_overlap_ratio(candidate["rect"], daylight_bbox) > 0
                for candidate in candidates
            )
        )
        self.assertTrue(
            all(
                _rect_overlap_ratio(candidate["rect"], clearance_bbox) == 0
                for candidate in candidates
            )
        )

    def test_solver_can_drop_unplaceable_optional_trial_anchor_cluster(self) -> None:
        core_cluster = _object_program_cluster(
            cluster_id="sleep_core",
            object_id="bed",
            category="bed",
            dims={"L": 1800, "W": 900, "H": 650},
        )
        trial_cluster = _object_program_cluster(
            cluster_id="work_trial",
            object_id="desk",
            category="desk",
            dims={"L": 9000, "W": 9000, "H": 760},
        )
        trial_spec = trial_cluster["object_program"]["object_specs_by_id"]["desk"]
        trial_spec["trial_optional"] = True
        trial_spec["solver_trial"] = True
        trial_spec["budget_trial"] = True
        trial_cluster["object_program"]["protected_ids"] = []

        result = solve_object_level_layout(
            room_model=_room_model(),
            merged_clusters={"clusters": [core_cluster, trial_cluster]},
            relation_plan={},
            grid_mm=50,
            max_rounds=1,
        )

        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["complete"])
        self.assertNotIn("work_trial", result["verify_summary"]["missing_cluster_ids"])
        self.assertIn("work_trial", result["dropped_inventory_by_cluster"])
        self.assertEqual(
            result["verify_summary"]["dropped_trial_cluster_ids"], ["work_trial"]
        )

    def test_anchor_forbidden_overlap_uses_region_policy(self) -> None:
        rect = (0, 0, 1000, 1000)
        corridor = {
            "cluster_id": "sleep_core",
            "region_id": "entry_to_center_corridor",
            "bbox": (900, 0, 1900, 1000),
            "max_overlap_ratio": 0.12,
        }

        self.assertFalse(
            _anchor_forbidden_overlap_exceeds_limit(
                rect=rect,
                forbidden_regions=[corridor],
            )
        )
        self.assertTrue(
            _anchor_forbidden_overlap_exceeds_limit(
                rect=rect,
                forbidden_regions=[{**corridor, "max_overlap_ratio": 0.05}],
            )
        )
        self.assertFalse(
            _anchor_forbidden_overlap_exceeds_limit(
                rect=rect,
                forbidden_regions=[
                    {
                        **corridor,
                        "max_overlap_ratio": 0.05,
                        "enforcement": "hard_soft",
                    }
                ],
            )
        )

    def test_wall_flush_origin_candidates_keep_odd_dimensions_inside_room(self) -> None:
        origins = _wall_flush_origin_candidates(
            region_bbox=(5600, 0, 5800, 3800),
            room_bbox=(0, 0, 6000, 4000),
            zone_assignment="quiet_private_deep_zone",
            placement_behavior={"wall_backing": "required"},
            w_mm=2319,
            h_mm=2306,
            grid_mm=50,
        )

        self.assertTrue(origins)
        self.assertTrue(
            all(
                x >= 0 and y >= 0 and x + 2319 <= 6000 and y + 2306 <= 4000
                for x, y in origins
            )
        )

    def test_anchor_generation_expands_when_required_region_cannot_seed(self) -> None:
        room_model = _room_model()
        room_model["regions"].append(
            {
                "id": "tiny_zone",
                "bbox": {"min_x": 1000, "min_y": 1000, "max_x": 1100, "max_y": 1100},
            }
        )
        cluster = _object_program_cluster(
            cluster_id="sleep_core",
            object_id="bed",
            category="bed",
            dims={"L": 1800, "W": 900, "H": 900},
        )
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "sleep_core": {
                    "zone_assignment": "tiny_zone",
                    "required_region_ids": ["tiny_zone"],
                    "preferred_region_ids": ["tiny_zone"],
                    "placement_behavior": {"wall_backing": "required"},
                }
            },
            "anchor_region_preferences_by_cluster": {
                "sleep_core": {
                    "zone_assignment": "tiny_zone",
                    "required_region_ids": ["tiny_zone"],
                    "preferred_region_ids": ["tiny_zone"],
                    "placement_behavior": {"wall_backing": "required"},
                }
            },
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["sleep_core"],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        self.assertTrue(
            any(
                row.get("search_stage") == "expanded_wall_fallback"
                for row in candidates
            )
        )

    def test_committed_anchor_region_is_generated_before_broad_preferred_regions(
        self,
    ) -> None:
        room_model = _room_model()
        room_model["regions"].extend(
            [
                {
                    "id": "left_wall_zone",
                    "bbox": {
                        "min_x": 0,
                        "min_y": 0,
                        "max_x": 700,
                        "max_y": 4000,
                    },
                },
                {
                    "id": "right_wall_zone",
                    "bbox": {
                        "min_x": 5300,
                        "min_y": 0,
                        "max_x": 6000,
                        "max_y": 4000,
                    },
                },
            ]
        )
        cluster = _object_program_cluster(
            cluster_id="sleep_core",
            object_id="bed",
            category="bed",
            dims={"L": 1800, "W": 900, "H": 900},
        )
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "sleep_core": {
                    "zone_assignment": "left_wall_zone",
                    "committed_region_ids": ["left_wall_zone"],
                    "committed_wall_side": "left_wall",
                    "preferred_region_ids": ["right_wall_zone", "left_wall_zone"],
                    "preferred_wall_side": "right_wall",
                    "anchor_strength": "strong",
                    "placement_behavior": {"wall_backing": "preferred"},
                }
            }
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["sleep_core"],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        self.assertEqual(
            {candidate["search_stage"] for candidate in candidates},
            {"committed_regions"},
        )
        self.assertTrue(
            all(candidate["committed_candidate_count"] > 0 for candidate in candidates)
        )
        self.assertTrue(
            all(
                candidate["committed_region_ids"] == ["left_wall_zone"]
                for candidate in candidates
            )
        )

    def test_committed_anchor_miss_is_marked_when_solver_falls_back(self) -> None:
        room_model = _room_model()
        room_model["regions"].append(
            {
                "id": "tiny_zone",
                "bbox": {"min_x": 1000, "min_y": 1000, "max_x": 1050, "max_y": 1050},
            }
        )
        cluster = _object_program_cluster(
            cluster_id="sleep_core",
            object_id="bed",
            category="bed",
            dims={"L": 1800, "W": 900, "H": 900},
        )
        relation_plan = {
            "anchor_layout_hints_by_cluster": {
                "sleep_core": {
                    "zone_assignment": "tiny_zone",
                    "committed_region_ids": ["tiny_zone"],
                    "preferred_region_ids": ["top_storage"],
                    "anchor_strength": "strong",
                    "placement_behavior": {"wall_backing": "preferred"},
                }
            }
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        candidates = _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"]["sleep_core"],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=50,
        )

        self.assertTrue(candidates)
        self.assertTrue(
            any(candidate["committed_region_missed"] for candidate in candidates)
        )
        self.assertTrue(
            any(
                candidate["search_stage"]
                in {"fallback_preferred_regions", "expanded_wall_fallback"}
                for candidate in candidates
            )
        )

    def test_view_corridor_blocker_fails_hard_valid(self) -> None:
        verify = _verify_object_level_solution(
            solution={
                "placed_objects": [
                    {
                        "cluster_id": "main_seating",
                        "object_id": "sofa",
                        "priority": "anchor",
                        "rect": (1000, 1000, 2600, 1800),
                        "center": {"x": 1800, "y": 1400},
                        "front_world": {"dx": 1, "dy": 0},
                    },
                    {
                        "cluster_id": "media",
                        "object_id": "tv_console",
                        "priority": "anchor",
                        "rect": (4200, 1000, 5000, 1600),
                        "center": {"x": 4600, "y": 1300},
                        "front_world": {"dx": -1, "dy": 0},
                    },
                    {
                        "cluster_id": "main_seating",
                        "object_id": "side_table",
                        "priority": "support",
                        "relative_to": "sofa",
                        "rect": (2900, 1100, 3400, 1600),
                        "center": {"x": 3150, "y": 1350},
                        "front_world": {"dx": 0, "dy": 1},
                    },
                ]
            },
            world={
                "room_bbox": (0, 0, 6000, 4000),
                "clusters_by_id": {"main_seating": {}, "media": {}},
                "protected_regions": [],
            },
            room_model={},
            relation_plan={
                "layout_intent_profile": {
                    "primary_cluster_id": "main_seating",
                    "secondary_cluster_id": "media",
                    "focus_mode": "viewing",
                }
            },
        )

        self.assertFalse(verify["hard_valid"])
        self.assertFalse(verify["gallery_eligible"])
        self.assertFalse(verify["view_corridor_ok"])
        self.assertIn("view_corridor_blocked", verify["quality_gate_reasons"])

    def test_support_slot_candidates_sample_multiple_positions_within_gap_band(
        self,
    ) -> None:
        cluster_program = _object_program_cluster(
            cluster_id="main_seating",
            object_id="sofa",
            category="sofa",
            dims={"L": 1800, "W": 850, "H": 800},
            support_edges=[
                {
                    "object_id": "armchair",
                    "relative_to": "sofa",
                    "side_options": ["left", "right"],
                    "gap_min": 100,
                    "gap_max": 350,
                    "orientation": "face_base",
                }
            ],
        )
        base_row = {
            "cluster_id": "main_seating",
            "object_id": "sofa",
            "rect": (2000, 500, 3800, 1350),
            "rot": 0,
            "front_token": "top",
            "front_world": {"dx": 0, "dy": 1},
        }

        slots = _support_slot_candidates(
            cluster_program=cluster_program,
            object_id="armchair",
            base_row=base_row,
            edge=cluster_program["object_program"]["support_edges"][0],
            grid_mm=50,
        )

        unique_slots = {
            (slot["x"], slot["y"], slot["rot"])
            for slot in slots
            if isinstance(slot, dict)
        }
        self.assertGreater(len(unique_slots), 4)

    def test_required_support_standalone_slots_prefer_inward_wall_orientation(
        self,
    ) -> None:
        cluster_program = _object_program_cluster(
            cluster_id="storage_core",
            object_id="wardrobe",
            category="wardrobe",
            dims={"L": 900, "W": 600, "H": 2200},
            support_edges=[
                {
                    "object_id": "bookshelf",
                    "relative_to": "wardrobe",
                    "side_options": ["left", "right"],
                    "support_role": "wall_support",
                    "band_intent": "wall_band",
                }
            ],
        )
        object_program = cluster_program["object_program"]
        object_program["required_object_ids"] = ["wardrobe", "bookshelf"]
        object_program["required_support_ids"] = ["bookshelf"]
        object_program["relaxable_required_ids"] = ["bookshelf"]
        object_program["protected_ids"] = ["wardrobe"]
        object_program["object_specs_by_id"]["bookshelf"].update(
            {
                "role": "support",
                "priority": "primary",
                "rep_dims_mm": {"L": 630, "W": 250, "H": 2220},
                "required_support": True,
                "relaxable_required": True,
            }
        )
        room_model = _room_model()
        relation_plan = {
            "anchor_region_preferences_by_cluster": {
                "storage_core": {
                    "zone_assignment": "top_wall_zone",
                    "preferred_region_ids": ["top_wall_zone", "edge_loading_zone"],
                }
            }
        }
        world = _build_object_solver_world(
            room_model=room_model,
            merged_clusters={"clusters": [cluster_program]},
            relation_plan=relation_plan,
            cluster_constraints=None,
            grid_mm=50,
        )

        slots = _required_support_standalone_slot_candidates(
            cluster_program=cluster_program,
            object_id="bookshelf",
            relation_plan=relation_plan,
            world=world,
            room_model=room_model,
            grid_mm=50,
        )

        self.assertTrue(slots)
        self.assertGreater(float(slots[0]["orientation_score"]), 1000.0)
        self.assertEqual(slots[0]["rect"][1], 0)

    def test_required_support_is_not_droppable_from_degradation_ladder(self) -> None:
        object_program = {
            "members": ["wardrobe", "bookshelf"],
            "required_object_ids": ["wardrobe", "bookshelf"],
            "required_support_ids": ["bookshelf"],
            "protected_ids": ["wardrobe"],
            "droppable_ids": ["bookshelf"],
            "degradation_ladder": ["drop_bookshelf"],
            "support_edges": [{"object_id": "bookshelf"}],
        }

        self.assertNotIn("bookshelf", _droppable_object_ids(object_program))


if __name__ == "__main__":
    unittest.main()
