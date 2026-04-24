from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fake_ortools = types.ModuleType("ortools")
fake_sat = types.ModuleType("ortools.sat")
fake_python = types.ModuleType("ortools.sat.python")
fake_cp_model = types.ModuleType("ortools.sat.python.cp_model")
fake_cp_model.IntVar = object
fake_cp_model.CpModel = object
fake_cp_model.CpSolver = object
fake_cp_model.OPTIMAL = 4
fake_cp_model.FEASIBLE = 2
sys.modules.setdefault("ortools", fake_ortools)
sys.modules.setdefault("ortools.sat", fake_sat)
sys.modules.setdefault("ortools.sat.python", fake_python)
sys.modules.setdefault("ortools.sat.python.cp_model", fake_cp_model)

from agent.solver.solver import (  # noqa: E402
    _droppable_object_ids,
    _normalized_support_side_options,
    _object_level_face_pair_issues,
    _object_level_orientation_issues,
    _object_level_required_face_pairs,
    _support_slot_candidates,
    _verify_object_level_solution,
    solve_object_level_layout,
)


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

    def test_protected_center_overlap_is_ranked_fallback_not_unsat(self) -> None:
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

        self.assertEqual(result["status"], "PARTIAL")
        self.assertTrue(result["geometry_valid"])
        self.assertTrue(result["complete"])
        self.assertFalse(result["hard_valid"])
        objects = {
            str(row["object_id"]): row for row in result["absolute_layout"]["objects"]
        }
        self.assertIn("coffee_table", objects)
        verify = result["verify_summary"]
        self.assertIn(
            "blocking_protected_region_overlap", verify["quality_gate_reasons"]
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

    def test_mixed_media_pair_with_buffered_support_is_required(self) -> None:
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

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["reason"], "required_face_each_other_failed")

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

    def test_frontal_support_ignores_side_band_options(self) -> None:
        edge = {
            "object_id": "coffee_table",
            "side_options": ["head", "left", "right"],
            "support_role": "frontal_support",
            "band_intent": "front_band",
        }

        self.assertEqual(_normalized_support_side_options(edge), ["head"])

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


if __name__ == "__main__":
    unittest.main()
