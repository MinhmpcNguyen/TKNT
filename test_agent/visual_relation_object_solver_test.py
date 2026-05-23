from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    from agent.solver import solver as object_solver
    from agent.solver.solver import MacroClusterSolver
    from agent.visual_relation_object_solver import (
        VisualRelationObjectSolver,
        _relax_visual_paired_front_access_issues,
        _semantic_anchor_lock_violation,
    )
except RuntimeError as exc:  # pragma: no cover - dependency guard
    MacroClusterSolver = None
    VisualRelationObjectSolver = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_CASE_ROOT = Path("/Users/Yuki/TKNT/cases/user_f1a4e69d3672aab8_20260522T080307Z")
_DESK_CHAIR_CASE_ROOT = Path(
    "/Users/Yuki/TKNT/cases/user_f1a4e69d3672aab8_20260522T090422Z"
)
_EXACT_ANCHOR_CASE_ROOT = Path(
    "/Users/Yuki/TKNT/cases/user_f1a4e69d3672aab8_20260522T093805Z"
)


@unittest.skipIf(_IMPORT_ERROR is not None, "OR-Tools is not available")
@unittest.skipUnless(_CASE_ROOT.exists(), "visual solver fixture case is unavailable")
class VisualRelationObjectSolverTest(unittest.TestCase):
    def test_visual_solver_v2_solves_known_visual_case(self) -> None:
        room, merged, cluster_constraints, relation_plan = _load_case_inputs()

        result = VisualRelationObjectSolver().generate_object_layout(
            room_model_json=room,
            merged_clusters_json=merged,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_constraints,
            grid_mm=50,
        )

        self.assertEqual(result["solver_kind"], "visual_relation_object_solver_v2")
        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["geometry_valid"])
        self.assertTrue(result["hard_valid"])
        self.assertTrue(result["gallery_eligible"])
        self.assertGreaterEqual(len(result["solutions"]), 1)

        diagnostics = result["visual_solver_diagnostics"]
        self.assertEqual(diagnostics["strict_solver_status"], "UNSAT")
        self.assertGreater(
            diagnostics["support_fallback_counts"]["same_cluster_independent_wall"],
            0,
        )
        self.assertIn("lounge_reading", diagnostics["dropped_optional_inventory"])

        object_keys = {
            (row["cluster_id"], row["object_id"])
            for row in result["absolute_layout"]["objects"]
        }
        self.assertIn(("storage_core", "bookshelf"), object_keys)
        self.assertIn(("work_study", "chair"), object_keys)
        self.assertIn(("media_optional", "tv_console"), object_keys)

    def test_strict_solver_path_remains_unsat_for_known_case(self) -> None:
        room, merged, cluster_constraints, relation_plan = _load_case_inputs()

        result = MacroClusterSolver().generate_object_layout(
            room_model_json=room,
            merged_clusters_json=merged,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_constraints,
            grid_mm=50,
        )

        self.assertEqual(result["solver_kind"], "object_level_anchor_first")
        self.assertEqual(result["status"], "UNSAT")
        self.assertEqual(result.get("solutions") or [], [])

    @unittest.skipUnless(
        _DESK_CHAIR_CASE_ROOT.exists(),
        "visual desk-chair fixture case is unavailable",
    )
    def test_visual_solver_v2_accepts_valid_desk_chair_docking(self) -> None:
        room, merged, cluster_constraints, relation_plan = _load_case_inputs(
            _DESK_CHAIR_CASE_ROOT
        )

        result = VisualRelationObjectSolver().generate_object_layout(
            room_model_json=room,
            merged_clusters_json=merged,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_constraints,
            grid_mm=50,
        )

        self.assertEqual(result["solver_kind"], "visual_relation_object_solver_v2")
        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["geometry_valid"])
        self.assertTrue(result["hard_valid"])
        self.assertTrue(result["gallery_eligible"])
        self.assertGreaterEqual(
            result["visual_solver_diagnostics"]["candidate_count"], 1
        )
        self.assertGreater(
            result["visual_solver_diagnostics"]["gallery_eligible_solution_count"],
            0,
        )
        self.assertNotIn(
            "front_access_zone_blocked",
            result["verify_summary"]["quality_gate_reasons"],
        )

    @unittest.skipUnless(
        _DESK_CHAIR_CASE_ROOT.exists(),
        "visual desk-chair fixture case is unavailable",
    )
    def test_visual_solver_v2_rejects_wrong_direction_desk_chair_docking(
        self,
    ) -> None:
        room, merged, cluster_constraints, relation_plan = _load_case_inputs(
            _DESK_CHAIR_CASE_ROOT
        )
        world = object_solver._build_object_solver_world(
            room_model=room,
            merged_clusters=merged,
            relation_plan=relation_plan,
            cluster_constraints=cluster_constraints,
            grid_mm=50,
        )
        result = VisualRelationObjectSolver().generate_object_layout(
            room_model_json=room,
            merged_clusters_json=merged,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_constraints,
            grid_mm=50,
        )
        placed_objects = [
            dict(row) for row in result["absolute_layout"]["object_placements"]
        ]
        desk = next(
            row
            for row in placed_objects
            if row["cluster_id"] == "work_study" and row["object_id"] == "desk"
        )
        for row in placed_objects:
            if row["cluster_id"] == "work_study" and row["object_id"] == "chair":
                row["front_world"] = _opposite_front_vector(row, desk)
                break

        solution = {"placed_objects": placed_objects}
        verify = object_solver._verify_object_level_solution(
            solution=solution,
            world=world,
            room_model=room,
            relation_plan=relation_plan,
        )
        relaxed = _relax_visual_paired_front_access_issues(
            verify=verify,
            solution=solution,
            world=world,
            relation_plan=relation_plan,
        )

        self.assertFalse(relaxed["gallery_eligible"])
        self.assertIn("front_inward_contract_failed", relaxed["quality_gate_reasons"])
        self.assertTrue(relaxed["blocking_orientation_issues"])

    @unittest.skipUnless(
        _EXACT_ANCHOR_CASE_ROOT.exists(),
        "visual exact-anchor fixture case is unavailable",
    )
    def test_visual_solver_v2_rejects_center_tv_when_exact_wall_anchor_locked(
        self,
    ) -> None:
        room, merged, cluster_constraints, relation_plan = _load_case_inputs(
            _EXACT_ANCHOR_CASE_ROOT
        )
        relation_plan = _with_tv_exact_anchor_lock(relation_plan)
        world = object_solver._build_object_solver_world(
            room_model=room,
            merged_clusters=merged,
            relation_plan=relation_plan,
            cluster_constraints=cluster_constraints,
            grid_mm=50,
        )

        violation = _semantic_anchor_lock_violation(
            row={
                "cluster_id": "media_optional",
                "object_id": "tv_console",
                "rect": (2350, 1300, 3610, 1680),
            },
            world=world,
            room_model=room,
            relation_plan=relation_plan,
        )

        self.assertIsNotNone(violation)
        self.assertEqual(violation["reason"], "outside_semantic_anchor_region")

    @unittest.skipUnless(
        _EXACT_ANCHOR_CASE_ROOT.exists(),
        "visual exact-anchor fixture case is unavailable",
    )
    def test_visual_solver_v2_exact_anchor_does_not_publish_center_tv(
        self,
    ) -> None:
        room, merged, cluster_constraints, relation_plan = _load_case_inputs(
            _EXACT_ANCHOR_CASE_ROOT
        )
        relation_plan = _with_tv_exact_anchor_lock(relation_plan)

        result = VisualRelationObjectSolver().generate_object_layout(
            room_model_json=room,
            merged_clusters_json=merged,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_constraints,
            grid_mm=50,
        )

        diagnostics = result["visual_solver_diagnostics"]
        if result["status"] != "OK":
            self.assertIn(
                "media_optional/tv_console",
                diagnostics["unplaceable_locked_objects"],
            )
            return

        tv = next(
            row
            for row in result["absolute_layout"]["objects"]
            if row["cluster_id"] == "media_optional"
            and row["object_id"] == "tv_console"
        )
        self.assertGreaterEqual(tv["rect"][2], 5850)
        self.assertTrue(result["gallery_eligible"])


def _load_case_inputs(
    case_root: Path = _CASE_ROOT,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    return (
        _load_json(case_root, "01_room_interpreter.json"),
        _load_json(case_root, "04_cluster_merged.json"),
        _load_json(case_root, "02_cluster_forge.json"),
        _load_json(case_root, "module_outputs/visual_relation_plan.json"),
    )


def _load_json(case_root: Path, relative_path: str) -> dict[str, object]:
    with (case_root / relative_path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"{relative_path} must contain a JSON object")
    return payload


def _with_tv_exact_anchor_lock(
    relation_plan: dict[str, object],
) -> dict[str, object]:
    next_plan = json.loads(json.dumps(relation_plan))
    next_plan["semantic_anchor_preferences_by_cluster"] = {
        "media_optional": {
            "tv_console": {
                "cluster_id": "media_optional",
                "object_id": "tv_console",
                "semantic_anchor_region_id": "wall_2_usable_200_2800_focal",
                "source_region_id": "wall_2_usable_200_2800_focal",
                "kind": "focal_wall",
                "wall_side": "right_wall",
                "corner_hint": "top_right",
                "placement_role": "visual_focus",
                "anchor_lock_strength": "hard",
                "lockable": True,
                "priority": "high",
            }
        }
    }
    anchor_hints = next_plan.setdefault("anchor_layout_hints_by_cluster", {})
    if isinstance(anchor_hints, dict):
        media_hint = anchor_hints.setdefault("media_optional", {})
        if isinstance(media_hint, dict):
            media_hint["zone_assignment"] = "wall_2_usable_200_2800_focal"
            media_hint["preferred_region_ids"] = ["wall_2_usable_200_2800_focal"]
            media_hint["required_region_ids"] = ["wall_2_usable_200_2800_focal"]
            media_hint["preferred_wall_side"] = "right_wall"
            media_hint["anchor_strength"] = "hard"
    return next_plan


def _opposite_front_vector(
    row: dict[str, object],
    target: dict[str, object],
) -> dict[str, int]:
    row_center = row.get("center") if isinstance(row.get("center"), dict) else {}
    target_center = (
        target.get("center") if isinstance(target.get("center"), dict) else {}
    )
    dx = float(target_center.get("x") or 0) - float(row_center.get("x") or 0)
    dy = float(target_center.get("y") or 0) - float(row_center.get("y") or 0)
    if abs(dx) >= abs(dy):
        return {"dx": -1 if dx > 0 else 1, "dy": 0}
    return {"dx": 0, "dy": -1 if dy > 0 else 1}


if __name__ == "__main__":
    unittest.main()
