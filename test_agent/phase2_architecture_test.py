from __future__ import annotations

import importlib
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_cluster_placer = importlib.import_module("agent.cluster_placer")
MacroClusterPlacer = _cluster_placer.MacroClusterPlacer
build_phase2_payload = _cluster_placer.build_phase2_payload
make_no_improvement_repair = _cluster_placer.make_no_improvement_repair
prepare_phase2_llm_payload = _cluster_placer.prepare_phase2_llm_payload
normalize_and_validate_repair_output = (
    _cluster_placer.normalize_and_validate_repair_output
)

_phase2_controller = importlib.import_module("agent.phase2_controller")
Phase2Controller = _phase2_controller.Phase2Controller
_choose_fallback_proposal = _phase2_controller._choose_fallback_proposal
_select_shortlist_candidate = _phase2_controller._select_shortlist_candidate
_should_stop_object_refine_search = _phase2_controller._should_stop_object_refine_search
_should_promote_candidate = _phase2_controller._should_promote_candidate
_should_replace_with_deterministic_candidate = (
    _phase2_controller._should_replace_with_deterministic_candidate
)

Phase2JudgeOutput = importlib.import_module(
    "agent_schema.phase2_judge_schema"
).Phase2JudgeOutput
run_phase2_judge = importlib.import_module("agent.phase2_judge").run_phase2_judge

_tools_v2 = importlib.import_module("cluster_placer.tools_v2")
CompileAcceptedPhase2Proposal = _tools_v2.CompileAcceptedPhase2Proposal
EnumeratePhase2RepairMoves = _tools_v2.EnumeratePhase2RepairMoves
EvaluatePhase2Proposal = _tools_v2.EvaluatePhase2Proposal


CASE_ROOT = REPO_ROOT / "cases" / "demo_tenant_20260326T040117Z"
CASE_ROOT_WINDOW = REPO_ROOT / "cases" / "demo_tenant_20260326T083558Z"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_payload() -> dict[str, Any]:
    room = _load_json(CASE_ROOT / "01_room_interpreter.json")
    merged = _load_json(CASE_ROOT / "04_cluster_merged.json")
    outlines = _load_json(CASE_ROOT / "05_cluster_outlines.json")
    relation_plan = _load_json(CASE_ROOT / "05b_cluster_relation_plan.json")
    solver = _load_json(CASE_ROOT / "06a_cluster_solver.json")
    return build_phase2_payload(
        room_01=room,
        clusters_04=merged,
        outlines_05=outlines,
        relation_05b=relation_plan,
        placer_06=solver,
        cluster_constraints=None,
    )


def _window_payload() -> dict[str, Any]:
    room = _load_json(CASE_ROOT_WINDOW / "01_room_interpreter.json")
    merged = _load_json(CASE_ROOT_WINDOW / "04_cluster_merged.json")
    outlines = _load_json(CASE_ROOT_WINDOW / "05_cluster_outlines.json")
    relation_plan = _load_json(CASE_ROOT_WINDOW / "05b_cluster_relation_plan.json")
    solver = _load_json(CASE_ROOT_WINDOW / "06a_cluster_solver.json")
    return build_phase2_payload(
        room_01=room,
        clusters_04=merged,
        outlines_05=outlines,
        relation_05b=relation_plan,
        placer_06=solver,
        cluster_constraints=None,
    )


def _merge_repair(
    payload: dict[str, Any], partial: dict[str, Any], notes: list[str] | None = None
) -> dict[str, Any]:
    seed = make_no_improvement_repair(payload)
    transform_map = {
        row["cluster_id"]: deepcopy(row)
        for row in seed.get("cluster_transforms") or []
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
    }
    variant_map = {
        row["cluster_id"]: deepcopy(row)
        for row in seed.get("selected_variants") or []
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
    }

    for row in partial.get("cluster_transforms") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            transform_map[row["cluster_id"]] = deepcopy(row)
    for row in partial.get("selected_variants") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            variant_map[row["cluster_id"]] = deepcopy(row)

    return {
        "status": "REPAIRED",
        "cluster_transforms": [
            deepcopy(transform_map[cid]) for cid in sorted(transform_map)
        ],
        "selected_variants": [
            deepcopy(variant_map[cid]) for cid in sorted(variant_map)
        ],
        "object_repairs": deepcopy(partial.get("object_repairs") or []),
        "notes": notes or [],
    }


def _hard_valid_negative_delta_proposal(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for move in (
        EnumeratePhase2RepairMoves(payload=payload, limit=40).get("moves") or []
    ):
        proposal = _merge_repair(
            payload,
            move.get("proposal") or {},
            notes=[str(move.get("reason") or "")],
        )
        evaluation = EvaluatePhase2Proposal(payload=payload, repair=proposal)
        if (
            bool(evaluation.get("hard_valid"))
            and int(
                ((evaluation.get("baseline_comparison") or {}).get("delta_score")) or 0
            )
            < 0
        ):
            return proposal, evaluation
    raise AssertionError("Expected at least one hard-valid negative-delta proposal")


class _StaticPlacer:
    def __init__(self, proposal: dict[str, Any]) -> None:
        self._proposal = proposal

    def generate_from_payload(
        self, *, payload: dict[str, Any], max_attempts: int = 3
    ) -> dict[str, Any]:
        _ = payload
        _ = max_attempts
        return self._proposal


class _RuleJudge:
    def generate(
        self,
        input_payload: dict[str, Any],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Phase2JudgeOutput:
        _ = temperature
        _ = max_tokens
        hard_valid = bool(
            (input_payload.get("hard_check_summary") or {}).get("hard_valid")
        )
        if not hard_valid:
            data = {
                "reasonableness_score": 10,
                "verdict": "REJECT",
                "top_issues": ["Candidate is hard-invalid."],
                "repair_advice": ["Repair hard validity first."],
                "priority_clusters": [],
            }
        else:
            data = {
                "reasonableness_score": 85,
                "verdict": "ACCEPT",
                "top_issues": [],
                "repair_advice": [],
                "priority_clusters": [],
            }
        return Phase2JudgeOutput.model_validate(data)


class _RevisingJudge:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def generate(
        self,
        input_payload: dict[str, Any],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Phase2JudgeOutput:
        _ = temperature
        _ = max_tokens
        self.payloads.append(input_payload)
        data = {
            "reasonableness_score": 65,
            "verdict": "REVISE",
            "top_issues": ["Still needs a better macro direction."],
            "repair_advice": ["Try a different cluster or move family."],
            "priority_clusters": ["sleep_area"],
        }
        return Phase2JudgeOutput.model_validate(data)


class Phase2ArchitectureTest(unittest.TestCase):
    def test_placer_returns_proposal_only(self) -> None:
        payload = _sample_payload()
        noop = make_no_improvement_repair(payload)

        def _llm_call(*, system_prompt: str, user_payload_json: str) -> dict[str, Any]:
            _ = system_prompt
            _ = user_payload_json
            return noop

        placer = MacroClusterPlacer(llm_call=_llm_call)
        proposal = placer.generate_from_payload(payload=payload)

        self.assertEqual(proposal["status"], "NO_IMPROVEMENT")
        self.assertIn("cluster_transforms", proposal)
        self.assertIn("selected_variants", proposal)
        self.assertIn("object_repairs", proposal)
        self.assertNotIn("room", proposal)
        self.assertNotIn("objects", proposal)
        self.assertNotIn("missing", proposal)

    def test_repair_output_must_include_full_layout(self) -> None:
        payload = _sample_payload()
        invalid = {
            "status": "REPAIRED",
            "cluster_transforms": [payload["seed_layout"]["cluster_transforms"][0]],
            "selected_variants": [payload["seed_layout"]["selected_variants"][0]],
            "object_repairs": [],
            "notes": ["Partial output should fail."],
        }

        with self.assertRaises(ValueError):
            normalize_and_validate_repair_output(payload, invalid)

    def test_tool_evaluate_and_compile_keep_downstream_shape(self) -> None:
        payload = _sample_payload()
        noop = make_no_improvement_repair(payload)

        evaluation = EvaluatePhase2Proposal(payload=payload, repair=noop)
        compiled = CompileAcceptedPhase2Proposal(payload=payload, repair=noop)

        self.assertIn("hard_valid", evaluation)
        self.assertIn("errors", evaluation)
        self.assertIn("violations_by_cluster", evaluation)
        self.assertIn("materialized_layout", evaluation)
        self.assertIn("metrics", evaluation)
        self.assertIn("diagnosis", evaluation)
        self.assertIn("baseline_comparison", evaluation)
        self.assertEqual(compiled["status"], "OK")
        self.assertIn("room", compiled)
        self.assertIn("objects", compiled)
        self.assertIn("missing", compiled)

    def test_controller_recovers_when_placer_candidate_is_hard_invalid(self) -> None:
        payload = _sample_payload()
        invalid = make_no_improvement_repair(payload)
        invalid["status"] = "REPAIRED"
        invalid["cluster_transforms"] = [
            {
                **row,
                "x": int(row["x"]) + 5000
                if row["cluster_id"] == "sleep_area"
                else row["x"],
            }
            for row in invalid["cluster_transforms"]
        ]
        invalid["notes"] = ["Force a hard-invalid translation for controller fallback."]

        controller = Phase2Controller(
            placer=_StaticPlacer(invalid),  # type: ignore[arg-type]
            judge=_RuleJudge(),  # type: ignore[arg-type]
            max_iterations=1,
        )
        result = controller.generate(
            room_interpreter_json=_load_json(CASE_ROOT / "01_room_interpreter.json"),
            cluster_merged_json=_load_json(CASE_ROOT / "04_cluster_merged.json"),
            cluster_outlines_json=_load_json(CASE_ROOT / "05_cluster_outlines.json"),
            relation_plan_json=_load_json(CASE_ROOT / "05b_cluster_relation_plan.json"),
            solver_output_json=_load_json(CASE_ROOT / "06a_cluster_solver.json"),
            cluster_constraints_json=None,
        )

        self.assertEqual(result["proposal"]["status"], "REPAIRED")
        self.assertEqual(result["absolute_layout"]["status"], "OK")
        self.assertTrue(result["history"])
        self.assertTrue(result["history"][0]["tool_evaluation"]["hard_valid"])
        self.assertTrue(
            any(
                ("Deterministic shortlist" in note) or ("Controller forced" in note)
                for note in result["history"][0]["proposal"]["notes"]
            )
        )

    def test_controller_synthesizes_fallback_when_placer_returns_no_improvement(
        self,
    ) -> None:
        payload = _sample_payload()
        noop = make_no_improvement_repair(payload)

        controller = Phase2Controller(
            placer=_StaticPlacer(noop),  # type: ignore[arg-type]
            judge=_RuleJudge(),  # type: ignore[arg-type]
            max_iterations=1,
        )
        result = controller.generate(
            room_interpreter_json=_load_json(CASE_ROOT / "01_room_interpreter.json"),
            cluster_merged_json=_load_json(CASE_ROOT / "04_cluster_merged.json"),
            cluster_outlines_json=_load_json(CASE_ROOT / "05_cluster_outlines.json"),
            relation_plan_json=_load_json(CASE_ROOT / "05b_cluster_relation_plan.json"),
            solver_output_json=_load_json(CASE_ROOT / "06a_cluster_solver.json"),
            cluster_constraints_json=None,
        )

        self.assertTrue(result["history"])
        self.assertEqual(result["history"][0]["proposal"]["status"], "REPAIRED")
        self.assertTrue(
            any(
                ("Controller fallback" in note)
                or ("Deterministic shortlist" in note)
                or ("Controller forced" in note)
                for note in result["history"][0]["proposal"]["notes"]
            )
        )

    def test_judge_receives_recent_attempt_context(self) -> None:
        payload = _sample_payload()
        noop = make_no_improvement_repair(payload)
        judge = _RevisingJudge()

        controller = Phase2Controller(
            placer=_StaticPlacer(noop),  # type: ignore[arg-type]
            judge=judge,  # type: ignore[arg-type]
            max_iterations=2,
        )
        controller.generate(
            room_interpreter_json=_load_json(CASE_ROOT / "01_room_interpreter.json"),
            cluster_merged_json=_load_json(CASE_ROOT / "04_cluster_merged.json"),
            cluster_outlines_json=_load_json(CASE_ROOT / "05_cluster_outlines.json"),
            relation_plan_json=_load_json(CASE_ROOT / "05b_cluster_relation_plan.json"),
            solver_output_json=_load_json(CASE_ROOT / "06a_cluster_solver.json"),
            cluster_constraints_json=None,
        )

        self.assertGreaterEqual(len(judge.payloads), 2)
        second_context = judge.payloads[1].get("controller_context") or {}
        self.assertTrue(second_context.get("recent_attempts"))
        first_payload = judge.payloads[0]
        self.assertIn("comparison_summary", first_payload)
        self.assertIn("generic_signals", first_payload)
        self.assertTrue((first_payload.get("generic_signals") or {}).get("room_notes"))

    def test_negative_delta_candidate_is_not_promoted(self) -> None:
        payload = _sample_payload()
        _, worse_evaluation = _hard_valid_negative_delta_proposal(payload)
        self.assertTrue(worse_evaluation["hard_valid"])
        self.assertLess(
            int(
                (worse_evaluation.get("baseline_comparison") or {}).get("delta_score")
                or 0
            ),
            0,
        )
        self.assertFalse(
            _should_promote_candidate(
                worse_evaluation,
                {
                    "verdict": "REVISE",
                    "reasonableness_score": 95,
                    "top_issues": [],
                },
            )
        )
        self.assertFalse(
            _should_promote_candidate(
                worse_evaluation,
                {
                    "verdict": "ACCEPT",
                    "reasonableness_score": 95,
                    "top_issues": [],
                },
            )
        )

    def test_negative_delta_shortlist_candidate_never_replaces(self) -> None:
        payload = _sample_payload()
        noop = make_no_improvement_repair(payload)
        baseline_evaluation = EvaluatePhase2Proposal(payload=payload, repair=noop)
        _, worse_evaluation = _hard_valid_negative_delta_proposal(payload)

        self.assertTrue(baseline_evaluation["hard_valid"])
        self.assertTrue(worse_evaluation["hard_valid"])
        self.assertLess(
            int(
                (worse_evaluation.get("baseline_comparison") or {}).get("delta_score")
                or 0
            ),
            0,
        )
        self.assertFalse(
            _should_replace_with_deterministic_candidate(
                worse_evaluation,
                baseline_evaluation,
            )
        )

    def test_repair_move_enumeration_respects_search_phase(self) -> None:
        payload = _sample_payload()

        payload["phase_control"]["repair_phase"] = "macro_layout"
        macro_moves = EnumeratePhase2RepairMoves(payload=payload, limit=24)["moves"]
        self.assertTrue(macro_moves)
        self.assertTrue(
            all(str(move.get("kind") or "") != "object_pose" for move in macro_moves)
        )

        payload["phase_control"]["repair_phase"] = "object_refine"
        object_moves = EnumeratePhase2RepairMoves(payload=payload, limit=24)["moves"]
        self.assertTrue(object_moves)
        self.assertTrue(
            all(str(move.get("kind") or "") == "object_pose" for move in object_moves)
        )
        self.assertTrue(
            all(
                bool(((move.get("proposal") or {}).get("object_repairs") or []))
                for move in object_moves
            )
        )

    def test_object_refine_fallback_returns_object_repair(self) -> None:
        payload = _sample_payload()
        payload["phase_control"]["repair_phase"] = "object_refine"
        baseline = EvaluatePhase2Proposal(
            payload=payload,
            repair=make_no_improvement_repair(payload),
        )

        proposal = _choose_fallback_proposal(
            payload=payload,
            baseline_evaluation=baseline,
            attempted_signatures=set(),
            history=[],
            search_phase="object_refine",
        )

        self.assertEqual(proposal["status"], "REPAIRED")
        self.assertTrue(proposal["object_repairs"])

    def test_object_refine_moves_stay_in_safe_neighborhood(self) -> None:
        payload = _sample_payload()
        payload["phase_control"]["repair_phase"] = "object_refine"
        object_moves = EnumeratePhase2RepairMoves(payload=payload, limit=40)["moves"]

        self.assertTrue(object_moves)
        allowed_ops = {
            str(
                (((move.get("proposal") or {}).get("object_repairs") or [{}])[0]).get(
                    "op"
                )
                or ""
            )
            for move in object_moves
        }
        self.assertTrue(allowed_ops.issubset({"rotate_object", "nudge_object"}))

    def test_object_refine_rejects_macro_edits_and_nonenumerated_moves(self) -> None:
        payload = _sample_payload()
        payload["phase_control"]["repair_phase"] = "object_refine"
        seed = make_no_improvement_repair(payload)

        invalid_macro = {
            "status": "REPAIRED",
            "cluster_transforms": [
                {
                    **deepcopy(seed["cluster_transforms"][0]),
                    "x": int(seed["cluster_transforms"][0]["x"]) + 50,
                },
                *deepcopy(seed["cluster_transforms"][1:]),
            ],
            "selected_variants": deepcopy(seed["selected_variants"]),
            "object_repairs": [
                {
                    "cluster_id": payload["objects_world"][0]["cluster_id"],
                    "object_id": payload["objects_world"][0]["object_id"],
                    "op": "nudge_object",
                    "params": {"dx": 50, "dy": 0},
                }
            ],
            "notes": ["Invalid macro drift in object refine."],
        }
        with self.assertRaises(ValueError):
            normalize_and_validate_repair_output(payload, invalid_macro)

        invalid_swap = {
            "status": "REPAIRED",
            "cluster_transforms": deepcopy(seed["cluster_transforms"]),
            "selected_variants": deepcopy(seed["selected_variants"]),
            "object_repairs": [
                {
                    "cluster_id": payload["objects_world"][0]["cluster_id"],
                    "object_id": payload["objects_world"][0]["object_id"],
                    "op": "swap_objects",
                    "params": {
                        "other_object_id": payload["objects_world"][1]["object_id"]
                    },
                }
            ],
            "notes": ["Unsupported risky swap."],
        }
        with self.assertRaises(ValueError):
            normalize_and_validate_repair_output(payload, invalid_swap)

    def test_window_case_scores_front_to_window_and_affinity_preferences(self) -> None:
        payload = _window_payload()
        evaluation = EvaluatePhase2Proposal(
            payload=payload,
            repair=make_no_improvement_repair(payload),
        )

        top_object_intents = (
            evaluation.get("metrics", {})
            .get("goal_alignment_summary", {})
            .get("top_object_intents", [])
        )
        affinity_rows = evaluation.get("metrics", {}).get(
            "cluster_affinity_to_preferred_zone", []
        )
        path_rows = (
            evaluation.get("metrics", {})
            .get("main_path_clearance", {})
            .get("paths", [])
        )

        self.assertTrue(
            any(
                row.get("cluster_id") == "work_area"
                and row.get("intent") == "front_to_window"
                for row in top_object_intents
            )
        )
        self.assertTrue(
            any(
                row.get("cluster_id") == "sleep_area"
                and row.get("preferred_zone") in {"far_from_entry", "long_wall"}
                for row in affinity_rows
            )
        )
        self.assertTrue(
            any(
                row.get("cluster_id") == "storage_area"
                and row.get("preferred_zone") == "recess_or_edge"
                for row in affinity_rows
            )
        )
        self.assertTrue(
            any(
                row.get("target_cluster_id") == "sleep_area"
                and isinstance(row.get("min_clearance_mm"), int)
                and "clearance_shortage_mm" in row
                for row in path_rows
            )
        )

    def test_prepare_payload_exposes_main_path_hints(self) -> None:
        payload = _window_payload()
        compact = prepare_phase2_llm_payload(payload)
        scoring_hints = (compact.get("tool_context") or {}).get("scoring_hints") or {}
        self.assertTrue(scoring_hints.get("main_path_clearance"))
        self.assertIn("opening_band_blocking", scoring_hints)

    def test_shortlist_selection_prefers_macro_diversity_when_plateauing(self) -> None:
        repeated_history = [
            {
                "search_phase": "macro_layout",
                "direction_summary": {
                    "primary_clusters": ["sleep_area"],
                    "move_families": ["cluster_translation"],
                },
            },
            {
                "search_phase": "macro_layout",
                "direction_summary": {
                    "primary_clusters": ["sleep_area"],
                    "move_families": ["cluster_translation"],
                },
            },
        ]
        candidates = [
            {
                "evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": 24, "candidate_score": 10},
                    "errors": [],
                },
                "direction_summary": {
                    "primary_clusters": ["sleep_area"],
                    "move_families": ["cluster_translation"],
                },
            },
            {
                "evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": 12, "candidate_score": 8},
                    "errors": [],
                },
                "direction_summary": {
                    "primary_clusters": ["work_area"],
                    "move_families": ["cluster_translation"],
                },
            },
        ]

        selected = _select_shortlist_candidate(
            candidates, repeated_history, "macro_layout"
        )
        self.assertEqual(selected, candidates[1])

    def test_object_refine_stops_after_repeated_stall(self) -> None:
        history = [
            {
                "search_phase": "object_refine",
                "tool_evaluation": {
                    "hard_valid": False,
                    "baseline_comparison": {"delta_score": -100},
                },
                "judge_evaluation": {"verdict": "REJECT"},
            },
            {
                "search_phase": "object_refine",
                "tool_evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": -10},
                },
                "judge_evaluation": {"verdict": "REJECT"},
            },
            {
                "search_phase": "object_refine",
                "tool_evaluation": {
                    "hard_valid": False,
                    "baseline_comparison": {"delta_score": -200},
                },
                "judge_evaluation": {"verdict": "REJECT"},
            },
        ]
        shortlist = [
            {
                "evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": 0},
                }
            }
        ]
        self.assertTrue(
            _should_stop_object_refine_search(
                history=history,
                deterministic_shortlist=shortlist,
                search_phase="object_refine",
            )
        )

    def test_judge_calibration_breaks_flat_revise_scores(self) -> None:
        payload = {
            "hard_check_summary": {"hard_valid": True},
            "comparison_summary": {"delta_score": 1500},
            "metrics": {
                "cluster_affinity_to_preferred_zone": [],
                "opening_band_blocking": [],
                "main_path_clearance": {"paths": []},
                "central_congestion": [],
                "cluster_edge_vs_center_fit": [],
                "cluster_internal_constraint_fidelity": [],
            },
            "diagnosis": {"prioritized_clusters": [{"cluster_id": "work_area"}]},
        }

        result = run_phase2_judge(
            payload=payload,
            llm_call=lambda system_prompt, user_payload_json: {
                "reasonableness_score": 70,
                "verdict": "REVISE",
                "top_issues": [],
                "repair_advice": ["keep refining"],
                "priority_clusters": [],
            },
        )

        self.assertEqual(result["verdict"], "ACCEPT")
        self.assertGreaterEqual(result["reasonableness_score"], 85)
        self.assertEqual(result["repair_advice"], [])

    def test_judge_calibration_demotes_accept_when_main_path_is_severely_blocked(
        self,
    ) -> None:
        payload = {
            "hard_check_summary": {"hard_valid": True},
            "comparison_summary": {"delta_score": 900},
            "metrics": {
                "cluster_affinity_to_preferred_zone": [],
                "opening_band_blocking": [],
                "main_path_clearance": {
                    "paths": [
                        {
                            "path_id": "door_1->sleep_area",
                            "target_cluster_id": "sleep_area",
                            "min_clearance_mm": 320,
                            "clearance_shortage_mm": 260,
                            "blocked_samples": 3,
                            "sample_count": 5,
                            "blocked_ratio": 0.22,
                        }
                    ]
                },
                "central_congestion": [],
                "cluster_edge_vs_center_fit": [],
                "cluster_internal_constraint_fidelity": [],
            },
            "diagnosis": {"prioritized_clusters": [{"cluster_id": "sleep_area"}]},
        }

        result = run_phase2_judge(
            payload=payload,
            llm_call=lambda system_prompt, user_payload_json: {
                "reasonableness_score": 91,
                "verdict": "ACCEPT",
                "top_issues": [],
                "repair_advice": [],
                "priority_clusters": [],
            },
        )

        self.assertEqual(result["verdict"], "REVISE")
        self.assertLessEqual(result["reasonableness_score"], 84)


if __name__ == "__main__":
    unittest.main()
