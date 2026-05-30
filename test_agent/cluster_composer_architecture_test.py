from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fake_client_module = types.ModuleType("clients.openai_client")


class _DummyOpenAIClient:
    pass


fake_client_module.OpenAIClient = _DummyOpenAIClient
sys.modules["clients.openai_client"] = fake_client_module

_cluster_composer = importlib.import_module("agent.cluster_composer")
ClusterComposer = _cluster_composer.ClusterComposer
_cluster_composer_tools = importlib.import_module("cluster_composer.tools")
_constraint_score = importlib.import_module("cluster_composer.constraint_score")
_find_cluster = _cluster_composer._find_cluster
_greedy_improve_valid_layout = _cluster_composer._greedy_improve_valid_layout
_run_verifier_once = _cluster_composer._run_verifier_once
_seed_local_layout = _cluster_composer._seed_local_layout
_build_search_seed_states = _cluster_composer._build_search_seed_states
_generate_compaction_moves = _cluster_composer._generate_compaction_moves
_infer_cluster_type = _cluster_composer._infer_cluster_type
_tool_rank = _cluster_composer._tool_rank
_build_preferred_patches = _cluster_composer_tools._build_preferred_patches
_extract_grid_mm = _cluster_composer._extract_grid_mm
local_cluster_verifier = _cluster_composer_tools.local_cluster_verifier
_tool_rank_key_from_scoring = _cluster_composer_tools._tool_rank_key_from_scoring
score_cluster_constraints = _constraint_score.score_cluster_constraints
GLOBAL_LAYOUT_GRID_MM = importlib.import_module(
    "layout.grid_policy"
).GLOBAL_LAYOUT_GRID_MM

CASE_ROOT = REPO_ROOT / "cases" / "demo_tenant_20260326T074655Z"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_work_cluster_with_semantic_intent() -> dict[str, Any]:
    return {
        "cluster_id": "work_area",
        "members": ["desk", "office_chair"],
        "anchors": ["desk"],
        "hard_constraints": [
            {"type": "no_overlap", "a": "desk", "b": "office_chair"},
            {
                "type": "dock_to_edge",
                "a": "office_chair",
                "b": "desk",
                "b_edge": "front",
                "span": "center",
                "gap_min": 0,
                "gap_max": 50,
            },
            {"type": "requires_access", "id": "desk", "mode": "front_clearance"},
        ],
        "soft_constraints": [
            {
                "type": "prefer_facing",
                "a": "office_chair",
                "b": "desk",
                "mode": "face_each_other",
                "weight": 8,
            }
        ],
        "cluster_rules": {
            "grid_mm": 50,
            "allowed_rotations": {
                "desk": [0],
                "office_chair": [0, 90, 180, 270],
            },
            "facing": {
                "desk": {"front": "top", "notes": "Desk is used from the top edge."},
                "office_chair": {
                    "front": "top",
                    "notes": "Chair front should face the desk.",
                },
            },
            "access_requirements": [
                {
                    "id": "desk",
                    "type": "front_clearance",
                    "required": True,
                }
            ],
            "semantic_placements": [
                {
                    "id": "office_chair",
                    "relative_to": "desk",
                    "kind": "dock_to_edge",
                    "b_edge": "front",
                    "span": "center",
                    "gap_min": 0,
                    "gap_max": 50,
                    "selection": "best_fit",
                    "orientation": "face_base",
                }
            ],
        },
        "decisions": [
            {
                "object_type": "desk",
                "rep_dims_m": {"L": 1.2, "W": 0.6},
                "priority": "anchor",
                "clearance_mm": 0,
            },
            {
                "object_type": "office_chair",
                "rep_dims_m": {"L": 0.55, "W": 0.55},
                "priority": "primary",
                "clearance_mm": 0,
            },
        ],
    }


def _build_dense_living_cluster() -> dict[str, Any]:
    return {
        "cluster_id": "seating_zone",
        "members": [
            "sectional_sofa",
            "sofa",
            "coffee_table",
            "ottoman",
            "recliner",
        ],
        "anchors": ["sectional_sofa", "sofa"],
        "hard_constraints": [
            {"type": "no_overlap", "a": "sectional_sofa", "b": "sofa"},
            {"type": "no_overlap", "a": "sectional_sofa", "b": "coffee_table"},
            {"type": "no_overlap", "a": "sectional_sofa", "b": "ottoman"},
            {"type": "no_overlap", "a": "sectional_sofa", "b": "recliner"},
            {"type": "no_overlap", "a": "sofa", "b": "coffee_table"},
            {"type": "no_overlap", "a": "sofa", "b": "ottoman"},
            {"type": "no_overlap", "a": "sofa", "b": "recliner"},
            {"type": "no_overlap", "a": "coffee_table", "b": "ottoman"},
            {"type": "no_overlap", "a": "coffee_table", "b": "recliner"},
            {"type": "no_overlap", "a": "ottoman", "b": "recliner"},
        ],
        "cluster_rules": {
            "grid_mm": 50,
            "allowed_rotations": {
                "sectional_sofa": [0, 90],
                "sofa": [0, 90],
                "coffee_table": [0, 90],
                "ottoman": [0, 90],
                "recliner": [0, 90],
            },
            "access_requirements": [],
        },
        "decisions": [
            {
                "object_type": "sectional_sofa",
                "rep_dims_m": {"L": 2.54, "W": 0.92},
                "priority": "anchor",
                "clearance_mm": 0,
            },
            {
                "object_type": "sofa",
                "rep_dims_m": {"L": 1.79, "W": 0.81},
                "priority": "anchor",
                "clearance_mm": 0,
            },
            {
                "object_type": "coffee_table",
                "rep_dims_m": {"L": 0.9, "W": 0.5},
                "priority": "primary",
                "clearance_mm": 0,
            },
            {
                "object_type": "ottoman",
                "rep_dims_m": {"L": 0.54, "W": 0.5},
                "priority": "secondary",
                "clearance_mm": 0,
            },
            {
                "object_type": "recliner",
                "rep_dims_m": {"L": 0.88, "W": 0.76},
                "priority": "primary",
                "clearance_mm": 0,
            },
        ],
    }


def _build_compact_living_anchor_cluster() -> dict[str, Any]:
    return {
        "cluster_id": "living_zone",
        "members": ["sectional_sofa", "coffee_table"],
        "anchors": ["sectional_sofa"],
        "hard_constraints": [
            {"type": "no_overlap", "a": "sectional_sofa", "b": "coffee_table"}
        ],
        "soft_constraints": [],
        "cluster_rules": {
            "grid_mm": 50,
            "allowed_rotations": {
                "sectional_sofa": [0],
                "coffee_table": [0],
            },
            "facing": {
                "sectional_sofa": {
                    "front": "top",
                    "notes": "Sectional faces upward in local cluster space.",
                }
            },
            "semantic_placements": [
                {
                    "id": "coffee_table",
                    "relative_to": "sectional_sofa",
                    "kind": "anchor_side",
                    "side_options": ["head", "left", "right"],
                    "gap_min": 100,
                    "gap_max": 100,
                    "selection": "best_fit",
                }
            ],
        },
        "decisions": [
            {
                "object_type": "sectional_sofa",
                "rep_dims_m": {"L": 2.0, "W": 0.8},
                "priority": "anchor",
                "clearance_mm": 0,
            },
            {
                "object_type": "coffee_table",
                "rep_dims_m": {"L": 0.6, "W": 0.4},
                "priority": "primary",
                "clearance_mm": 0,
            },
        ],
    }


class ClusterComposerArchitectureTest(unittest.TestCase):
    def test_infer_cluster_type_prefers_structure_over_generic_tag(self) -> None:
        cluster = {
            "cluster_id": "main_seating",
            "tag": "work",
            "members": [
                "sofa",
                "coffee_table",
                "armchair",
                "side_table",
                "floor_lamp",
            ],
            "anchors": ["sofa"],
            "decisions": [
                {"object_type": "sofa"},
                {"object_type": "coffee_table"},
                {"object_type": "armchair"},
            ],
        }

        self.assertEqual(_infer_cluster_type(cluster), "living")

    def test_extract_grid_mm_uses_shared_layout_policy(self) -> None:
        self.assertEqual(
            _extract_grid_mm({"cluster_rules": {"grid_mm": 400}}),
            GLOBAL_LAYOUT_GRID_MM,
        )
        self.assertEqual(_extract_grid_mm({"cluster_rules": {}}), GLOBAL_LAYOUT_GRID_MM)

    def test_generate_ignores_misleading_tag_for_living_cluster(self) -> None:
        cluster = _build_compact_living_anchor_cluster()
        cluster["tag"] = "work"

        result = ClusterComposer().generate(
            merged_clusters={"clusters": [cluster]},
            cluster_id="living_zone",
            description="",
            special_notes="",
        )

        self.assertEqual(result["status"], "OK")
        self.assertGreaterEqual(len(result.get("variant_bundle") or []), 1)

    def test_seed_layout_keeps_head_left_anchor_semantics(self) -> None:
        merged = _load_json(CASE_ROOT / "04_cluster_merged.json")
        cluster = _find_cluster(merged, "sleep_area")

        placements = _seed_local_layout(cluster)
        by_id = {row["id"]: row for row in placements}

        self.assertIn("bed", by_id)
        self.assertIn("nightstand", by_id)
        self.assertGreaterEqual(int(by_id["nightstand"]["y"]), 1650)
        self.assertGreater(int(by_id["nightstand"]["x"]), 0)
        self.assertLessEqual(int(by_id["nightstand"]["x"]), 500)

    def test_valid_anchor_layout_emits_anchor_refinement_hint(self) -> None:
        merged = _load_json(CASE_ROOT / "04_cluster_merged.json")
        cluster = _find_cluster(merged, "sleep_area")
        placements = [
            {"id": "bed", "x": 0, "y": 0, "rot": 0},
            {"id": "nightstand", "x": 0, "y": 1700, "rot": 0},
            {"id": "dresser", "x": 2000, "y": 0, "rot": 270},
        ]

        verifier = _run_verifier_once(
            cluster=cluster,
            placements=placements,
            access_clearance_ratio=0.25,
        )

        self.assertEqual(verifier.get("result"), "VALID")
        self.assertTrue(
            any(
                isinstance(move, dict)
                and move.get("reason") == "ANCHOR_REFINEMENT"
                and move.get("move_object") == "nightstand"
                for move in verifier.get("suggested_moves") or []
            )
        )

    def test_greedy_valid_refinement_improves_verified_layout(self) -> None:
        merged = _load_json(CASE_ROOT / "04_cluster_merged.json")
        cluster = _find_cluster(merged, "sleep_area")
        placements = [
            {"id": "bed", "x": 0, "y": 0, "rot": 0},
            {"id": "nightstand", "x": 0, "y": 1700, "rot": 0},
            {"id": "dresser", "x": 3000, "y": 0, "rot": 270},
        ]

        baseline_eval = _run_verifier_once(
            cluster=cluster,
            placements=placements,
            access_clearance_ratio=0.25,
        )

        self.assertEqual(baseline_eval.get("result"), "VALID")

        refined, refined_eval, rounds = _greedy_improve_valid_layout(
            cluster=cluster,
            access_clearance_ratio=0.25,
            placements=placements,
            verifier_eval=baseline_eval,
            max_rounds=8,
        )

        self.assertGreater(rounds, 0)
        self.assertEqual(refined_eval.get("result"), "VALID")
        self.assertLess(_tool_rank(refined_eval), _tool_rank(baseline_eval))
        self.assertLess(
            next(
                row["x"]
                for row in refined
                if isinstance(row, dict) and row.get("id") == "dresser"
            ),
            3000,
        )

    def test_verifier_reports_soft_constraints_in_summary(self) -> None:
        merged = _load_json(CASE_ROOT / "04_cluster_merged.json")
        cluster = _find_cluster(merged, "sleep_area")
        placements = [
            {"id": "bed", "x": 0, "y": 0, "rot": 0},
            {"id": "nightstand", "x": 0, "y": 1700, "rot": 0},
            {"id": "dresser", "x": 3000, "y": 0, "rot": 270},
        ]

        verifier = _run_verifier_once(
            cluster=cluster,
            placements=placements,
            access_clearance_ratio=0.25,
        )

        self.assertEqual(verifier.get("result"), "VALID")
        summary = ((verifier.get("debug") or {}).get("constraint_scores") or {}).get(
            "soft_summary"
        ) or {}
        self.assertGreater(int(summary.get("num_constraints") or 0), 0)

    def test_preferred_patches_use_soft_constraint_penalty(self) -> None:
        hard_constraints = [
            {"type": "no_overlap", "a": "anchor", "b": "target"},
            {"type": "no_overlap", "a": "anchor", "b": "guard"},
            {"type": "no_overlap", "a": "target", "b": "guard"},
        ]
        soft_constraints = [
            {"type": "prefer_near", "a": "target", "b": "anchor", "weight": 10}
        ]
        objects = {
            "anchor": {"id": "anchor", "w": 100, "h": 100, "allowed_rotations": [0]},
            "target": {"id": "target", "w": 100, "h": 100, "allowed_rotations": [0]},
            "guard": {"id": "guard", "w": 100, "h": 100, "allowed_rotations": [0]},
        }
        placements = [
            {"id": "anchor", "x": 0, "y": 0, "rot": 0},
            {"id": "target", "x": 200, "y": 0, "rot": 0},
            {"id": "guard", "x": 400, "y": 0, "rot": 0},
        ]

        current_scoring = score_cluster_constraints(
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            objects=objects,
            local_placements=placements,
            grid_mm=50,
            cluster_rules=None,
            use_clearance=True,
        )
        current_rank_key = _tool_rank_key_from_scoring(
            grid_violations=[],
            rot_violations=[],
            scoring=current_scoring,
            shape_quality=current_scoring.get("layout_quality") or {},
        )

        preferred = _build_preferred_patches(
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            objects=objects,
            placements=placements,
            grid_mm=50,
            cluster_rules=None,
            use_clearance=True,
            suggested_moves=[
                {
                    "reason": "MOVE_TARGET_CLOSER",
                    "move_object": "target",
                    "dx": -50,
                    "dy": 0,
                    "new_x": 150,
                    "new_y": 0,
                },
                {
                    "reason": "MOVE_TARGET_FARTHER",
                    "move_object": "target",
                    "dx": 50,
                    "dy": 0,
                    "new_x": 250,
                    "new_y": 0,
                },
            ],
            current_rank_key=current_rank_key,
            top_k=5,
        )

        self.assertEqual(len(preferred), 1)
        self.assertEqual(
            ((preferred[0].get("selected_move") or {}).get("reason")),
            "MOVE_TARGET_CLOSER",
        )

    def test_rank_key_prioritizes_semantic_soft_score_before_compactness(self) -> None:
        hard_constraints = [
            {"type": "no_overlap", "a": "anchor", "b": "target"},
            {"type": "no_overlap", "a": "anchor", "b": "guard"},
            {"type": "no_overlap", "a": "target", "b": "guard"},
        ]
        soft_constraints = [
            {"type": "prefer_near", "a": "target", "b": "anchor", "weight": 10}
        ]
        objects = {
            "anchor": {"id": "anchor", "w": 100, "h": 100, "allowed_rotations": [0]},
            "target": {"id": "target", "w": 100, "h": 100, "allowed_rotations": [0]},
            "guard": {"id": "guard", "w": 100, "h": 100, "allowed_rotations": [0]},
        }

        semantically_better_but_larger = [
            {"id": "anchor", "x": 0, "y": 0, "rot": 0},
            {"id": "target", "x": 150, "y": 0, "rot": 0},
            {"id": "guard", "x": 1000, "y": 0, "rot": 0},
        ]
        more_compact_but_semantically_worse = [
            {"id": "anchor", "x": 0, "y": 0, "rot": 0},
            {"id": "target", "x": 400, "y": 0, "rot": 0},
            {"id": "guard", "x": 550, "y": 0, "rot": 0},
        ]

        better_scoring = score_cluster_constraints(
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            objects=objects,
            local_placements=semantically_better_but_larger,
            grid_mm=50,
            cluster_rules=None,
            use_clearance=True,
        )
        compact_scoring = score_cluster_constraints(
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            objects=objects,
            local_placements=more_compact_but_semantically_worse,
            grid_mm=50,
            cluster_rules=None,
            use_clearance=True,
        )

        better_rank = _tool_rank_key_from_scoring(
            grid_violations=[],
            rot_violations=[],
            scoring=better_scoring,
            shape_quality=better_scoring.get("layout_quality") or {},
        )
        compact_rank = _tool_rank_key_from_scoring(
            grid_violations=[],
            rot_violations=[],
            scoring=compact_scoring,
            shape_quality=compact_scoring.get("layout_quality") or {},
        )

        self.assertLess(
            better_rank,
            compact_rank,
            "Forge semantics should outrank compactness when both layouts are valid.",
        )

    def test_verifier_suggests_soft_prefer_near_move(self) -> None:
        verifier = local_cluster_verifier(
            hard_constraints=[
                {"type": "no_overlap", "a": "anchor", "b": "target"},
            ],
            soft_constraints=[
                {"type": "prefer_near", "a": "target", "b": "anchor", "weight": 10}
            ],
            objects={
                "anchor": {
                    "id": "anchor",
                    "w": 100,
                    "h": 100,
                    "allowed_rotations": [0],
                },
                "target": {
                    "id": "target",
                    "w": 100,
                    "h": 100,
                    "allowed_rotations": [0],
                },
            },
            local_placements=[
                {"id": "anchor", "x": 0, "y": 0, "rot": 0},
                {"id": "target", "x": 250, "y": 0, "rot": 0},
            ],
            grid_mm=50,
            use_clearance=True,
            cluster_rules=None,
        )

        self.assertEqual(verifier.get("result"), "VALID")
        self.assertTrue(
            any(
                isinstance(move, dict)
                and move.get("reason") == "SOFT_PREFER_NEAR"
                and move.get("move_object") == "target"
                for move in verifier.get("suggested_moves") or []
            )
        )

    def test_seed_layout_centers_chair_on_desk_front_and_faces_base(self) -> None:
        cluster = _build_work_cluster_with_semantic_intent()

        placements = _seed_local_layout(cluster)
        by_id = {row["id"]: row for row in placements}

        self.assertEqual(by_id["desk"]["rot"], 0)
        self.assertEqual(by_id["office_chair"]["rot"], 180)
        self.assertEqual(by_id["office_chair"]["y"], 600)
        self.assertEqual(by_id["office_chair"]["x"], 300)

    def test_seed_layout_prefers_smallest_anchor_side_footprint(self) -> None:
        cluster = _build_compact_living_anchor_cluster()

        placements = _seed_local_layout(cluster)
        by_id = {row["id"]: row for row in placements}

        self.assertEqual(by_id["sectional_sofa"]["x"], 0)
        self.assertEqual(by_id["sectional_sofa"]["y"], 0)
        self.assertEqual(by_id["coffee_table"]["x"], by_id["sectional_sofa"]["x"])
        self.assertGreater(by_id["coffee_table"]["y"], by_id["sectional_sofa"]["y"])

    def test_build_search_seed_states_includes_rotated_layouts(self) -> None:
        cluster = _build_work_cluster_with_semantic_intent()

        seeds = _build_search_seed_states(cluster)
        signatures = {
            tuple((row["id"], row["x"], row["y"], row["rot"]) for row in seed)
            for seed in seeds
        }

        self.assertGreaterEqual(len(signatures), 4)

    def test_compact_wrap_seed_reduces_span_for_dense_cluster(self) -> None:
        cluster = _build_dense_living_cluster()

        row_seed = _seed_local_layout(cluster, strategy="row")
        compact_seed = _seed_local_layout(cluster, strategy="compact_wrap")

        row_bbox = _cluster_composer._build_cluster_output_from_placements(
            cluster=cluster,
            placements=row_seed,
        )["cluster_footprint"]["local_bbox"]
        compact_bbox = _cluster_composer._build_cluster_output_from_placements(
            cluster=cluster,
            placements=compact_seed,
        )["cluster_footprint"]["local_bbox"]

        row_span = int(row_bbox["max_x"]) - int(row_bbox["min_x"])
        compact_span = int(compact_bbox["max_x"]) - int(compact_bbox["min_x"])
        self.assertLess(compact_span, row_span)

    def test_cluster_output_records_effective_front_side(self) -> None:
        cluster = _build_work_cluster_with_semantic_intent()

        output = _cluster_composer._build_cluster_output_from_placements(
            cluster=cluster,
            placements=[
                {"id": "desk", "x": 0, "y": 0, "rot": 180},
                {"id": "office_chair", "x": 300, "y": 600, "rot": 180},
            ],
        )

        orientation_meta = output["orientation_meta"]
        important = orientation_meta["important_objects"]

        self.assertEqual(orientation_meta["cluster_effective_front_side"], "bottom")
        self.assertEqual(important["desk"]["effective_front_side"], "bottom")
        self.assertEqual(important["office_chair"]["effective_front_side"], "bottom")

    def test_build_search_seed_states_include_non_linear_topology(self) -> None:
        cluster = _build_dense_living_cluster()

        seeds = _build_search_seed_states(cluster)
        row_seed = _seed_local_layout(cluster, strategy="row")
        row_signature = tuple(
            (row["id"], row["x"], row["y"], row["rot"]) for row in row_seed
        )

        self.assertTrue(
            any(
                tuple((row["id"], row["x"], row["y"], row["rot"]) for row in seed)
                != row_signature
                for seed in seeds
            )
        )

    def test_generate_compaction_moves_pulls_objects_toward_origin(self) -> None:
        cluster = _build_work_cluster_with_semantic_intent()

        moves = _generate_compaction_moves(
            cluster=cluster,
            placements=[
                {"id": "desk", "x": 0, "y": 0, "rot": 0},
                {"id": "office_chair", "x": 900, "y": 600, "rot": 180},
            ],
            grid_mm=50,
        )

        self.assertTrue(
            any(
                isinstance(move, dict)
                and move.get("move_object") == "office_chair"
                and move.get("reason") in {"COMPACT_LEFT_EDGE", "COMPACT_TOP_EDGE"}
                for move in moves
            )
        )

    def test_verifier_enforces_center_span_for_docked_dependents(self) -> None:
        cluster = _build_work_cluster_with_semantic_intent()

        valid = _run_verifier_once(
            cluster=cluster,
            placements=[
                {"id": "desk", "x": 0, "y": 0, "rot": 0},
                {"id": "office_chair", "x": 300, "y": 600, "rot": 180},
            ],
            access_clearance_ratio=0.25,
        )
        invalid = _run_verifier_once(
            cluster=cluster,
            placements=[
                {"id": "desk", "x": 0, "y": 0, "rot": 0},
                {"id": "office_chair", "x": 0, "y": 600, "rot": 180},
            ],
            access_clearance_ratio=0.25,
        )

        self.assertEqual(valid.get("result"), "VALID")
        self.assertEqual(invalid.get("result"), "INVALID")
        hard_evaluations = (
            (invalid.get("debug") or {}).get("constraint_scores") or {}
        ).get("hard_evaluations") or []
        self.assertTrue(
            any(
                isinstance(score, dict)
                and score.get("type") == "dock_to_edge"
                and float((score.get("components") or {}).get("span_penalty") or 0) > 0
                for score in hard_evaluations
            )
        )


if __name__ == "__main__":
    unittest.main()
