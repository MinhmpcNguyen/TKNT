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
_cluster_composer_tools = importlib.import_module("cluster_composer.tools")
_constraint_score = importlib.import_module("cluster_composer.constraint_score")
_find_cluster = _cluster_composer._find_cluster
_coerce_verifier_args = _cluster_composer._coerce_verifier_args
_greedy_improve_valid_layout = _cluster_composer._greedy_improve_valid_layout
_run_verifier_once = _cluster_composer._run_verifier_once
_seed_local_layout = _cluster_composer._seed_local_layout
_tool_rank = _cluster_composer._tool_rank
_build_preferred_patches = _cluster_composer_tools._build_preferred_patches
local_cluster_verifier = _cluster_composer_tools.local_cluster_verifier
_tool_rank_key_from_scoring = _cluster_composer_tools._tool_rank_key_from_scoring
score_cluster_constraints = _constraint_score.score_cluster_constraints

CASE_ROOT = REPO_ROOT / "cases" / "demo_tenant_20260326T074655Z"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class ClusterComposerArchitectureTest(unittest.TestCase):
    def test_coerce_verifier_args_populates_soft_constraints(self) -> None:
        merged = _load_json(CASE_ROOT / "04_cluster_merged.json")
        cluster = _find_cluster(merged, "sleep_area")

        args = _coerce_verifier_args(
            args={
                "local_placements": [
                    {"id": "bed", "x": 0, "y": 0, "rot": 0},
                    {"id": "nightstand", "x": 0, "y": 1700, "rot": 0},
                    {"id": "dresser", "x": 2000, "y": 0, "rot": 270},
                ]
            },
            cluster=cluster,
            last_draft=None,
            access_clearance_ratio=0.25,
        )

        self.assertEqual(
            args.get("soft_constraints"),
            cluster.get("soft_constraints"),
        )

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


if __name__ == "__main__":
    unittest.main()
