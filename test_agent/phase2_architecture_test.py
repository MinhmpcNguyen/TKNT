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
RetryablePlacerModelError = _cluster_placer.RetryablePlacerModelError
normalize_and_validate_repair_output = (
    _cluster_placer.normalize_and_validate_repair_output
)

_phase2_controller = importlib.import_module("agent.phase2_controller")
Phase2Controller = _phase2_controller.Phase2Controller
_choose_fallback_proposal = _phase2_controller._choose_fallback_proposal
_postsolve_orientation_snap = _phase2_controller._postsolve_orientation_snap
build_phase2_preview_candidate = _phase2_controller.build_phase2_preview_candidate
_select_shortlist_candidate = _phase2_controller._select_shortlist_candidate
_has_macro_orientation_pressure = _phase2_controller._has_macro_orientation_pressure
_should_enter_object_refine_phase = _phase2_controller._should_enter_object_refine_phase
_should_stop_macro_search = _phase2_controller._should_stop_macro_search
_should_stop_object_refine_search = _phase2_controller._should_stop_object_refine_search
_should_promote_candidate = _phase2_controller._should_promote_candidate
_should_replace_with_deterministic_candidate = (
    _phase2_controller._should_replace_with_deterministic_candidate
)

Phase2JudgeOutput = importlib.import_module(
    "agent_schema.phase2_judge_schema"
).Phase2JudgeOutput
run_phase2_judge = importlib.import_module("agent.phase2_judge").run_phase2_judge
Phase2Judge = importlib.import_module("agent.phase2_judge").Phase2Judge

_tools_v2 = importlib.import_module("cluster_placer.tools_v2")
CompileAcceptedPhase2Proposal = _tools_v2.CompileAcceptedPhase2Proposal
EnumeratePhase2RepairMoves = _tools_v2.EnumeratePhase2RepairMoves
EvaluatePhase2Proposal = _tools_v2.EvaluatePhase2Proposal


CASE_ROOT = REPO_ROOT / "cases" / "demo_tenant_20260326T040117Z"
CASE_ROOT_WINDOW = REPO_ROOT / "cases" / "demo_tenant_20260326T083558Z"
CASE_ROOT_VIEWING = REPO_ROOT / "cases" / "demo_tenant_20260401T105018Z"
CASE_ROOT_OBJECT_ROTATION = REPO_ROOT / "cases" / "demo_tenant_20260401T135135Z"
CASE_ROOT_VARIANT_PREVIEW = REPO_ROOT / "cases" / "demo_tenant_20260401T144630Z"


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


def _viewing_payload() -> dict[str, Any]:
    room = _load_json(CASE_ROOT_VIEWING / "01_room_interpreter.json")
    merged = _load_json(CASE_ROOT_VIEWING / "04_cluster_merged.json")
    outlines = _load_json(CASE_ROOT_VIEWING / "05_cluster_outlines.json")
    relation_plan = _load_json(CASE_ROOT_VIEWING / "05b_cluster_relation_plan.json")
    solver = _load_json(CASE_ROOT_VIEWING / "06a_cluster_solver.json")
    return build_phase2_payload(
        room_01=room,
        clusters_04=merged,
        outlines_05=outlines,
        relation_05b=relation_plan,
        placer_06=solver,
        cluster_constraints=None,
    )


def _object_rotation_payload() -> dict[str, Any]:
    room = _load_json(CASE_ROOT_OBJECT_ROTATION / "01_room_interpreter.json")
    merged = _load_json(CASE_ROOT_OBJECT_ROTATION / "04_cluster_merged.json")
    outlines = _load_json(CASE_ROOT_OBJECT_ROTATION / "05_cluster_outlines.json")
    relation_plan = _load_json(
        CASE_ROOT_OBJECT_ROTATION / "05b_cluster_relation_plan.json"
    )
    solver = _load_json(CASE_ROOT_OBJECT_ROTATION / "06a_cluster_solver.json")
    return build_phase2_payload(
        room_01=room,
        clusters_04=merged,
        outlines_05=outlines,
        relation_05b=relation_plan,
        placer_06=solver,
        cluster_constraints=None,
    )


def _variant_preview_payload() -> dict[str, Any]:
    room = _load_json(CASE_ROOT_VARIANT_PREVIEW / "01_room_interpreter.json")
    merged = _load_json(CASE_ROOT_VARIANT_PREVIEW / "04_cluster_merged.json")
    outlines = _load_json(CASE_ROOT_VARIANT_PREVIEW / "05_cluster_outlines.json")
    relation_plan = _load_json(
        CASE_ROOT_VARIANT_PREVIEW / "05b_cluster_relation_plan.json"
    )
    solver = _load_json(CASE_ROOT_VARIANT_PREVIEW / "06a_cluster_solver.json")
    return build_phase2_payload(
        room_01=room,
        clusters_04=merged,
        outlines_05=outlines,
        relation_05b=relation_plan,
        placer_06=solver,
        cluster_constraints=None,
    )


def _payload_with_semantic_spacing_pressure() -> tuple[dict[str, Any], str, str]:
    payload = _variant_preview_payload()
    cards = payload.get("cluster_cards") or []

    def _apply_spacing_pressure(card_like: dict[str, Any]) -> str | None:
        cluster_rules = card_like.setdefault("cluster_rules", {})
        if not isinstance(cluster_rules, dict):
            cluster_rules = {}
            card_like["cluster_rules"] = cluster_rules
        existing_semantic = cluster_rules.get("semantic_placements") or []
        if isinstance(existing_semantic, list) and existing_semantic:
            first = existing_semantic[0]
            if isinstance(first, dict):
                cluster_rules["semantic_placements"] = [
                    {
                        **deepcopy(first),
                        "gap_min": 600,
                        "gap_max": 1000,
                        "proximity": "loose",
                    }
                ]
                return str(first.get("id") or "")
        local_ids = [
            str(row.get("id") or "")
            for row in (card_like.get("local_placements") or [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        ]
        unique_ids = list(dict.fromkeys(local_ids))
        if len(unique_ids) < 2:
            return None
        cluster_rules["semantic_placements"] = [
            {
                "id": unique_ids[1],
                "relative_to": unique_ids[0],
                "kind": "anchor_side",
                "side_options": ["left", "right", "head", "foot"],
                "gap_min": 600,
                "gap_max": 1000,
                "proximity": "loose",
                "selection": "best_fit",
                "orientation": "same_direction",
            }
        ]
        return unique_ids[1]

    for card in cards:
        if not isinstance(card, dict):
            continue
        object_id = _apply_spacing_pressure(card)
        if object_id:
            for variant in card.get("available_variants") or card.get("variants") or []:
                if isinstance(variant, dict) and isinstance(variant.get("card"), dict):
                    _apply_spacing_pressure(variant["card"])
            return payload, str(card.get("cluster_id") or ""), object_id
    raise AssertionError("Expected a cluster card containing at least two objects")


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
    def test_default_placer_uses_deterministic_search(self) -> None:
        payload = _sample_payload()
        placer = MacroClusterPlacer()

        proposal = placer.generate_from_payload(payload=payload, max_attempts=1)

        self.assertIn(proposal["status"], {"REPAIRED", "NO_IMPROVEMENT"})
        self.assertIn("cluster_transforms", proposal)
        self.assertIn("selected_variants", proposal)

    def test_default_judge_generates_without_llm(self) -> None:
        judge = Phase2Judge()
        result = judge.generate(
            {
                "hard_check_summary": {"hard_valid": True},
                "comparison_summary": {"delta_score": 300},
                "metrics": {
                    "cluster_affinity_to_preferred_zone": [],
                    "opening_band_blocking": [],
                    "main_path_clearance": {"paths": []},
                    "central_congestion": [],
                    "cluster_edge_vs_center_fit": [],
                    "cluster_internal_constraint_fidelity": [],
                },
                "diagnosis": {"prioritized_clusters": [{"cluster_id": "sleep_area"}]},
            }
        )

        self.assertIn(result.verdict, {"REVISE", "ACCEPT"})
        self.assertIn(result.next_step_mode, {"macro_layout", "object_refine", "stop"})

    def test_viewing_macro_moves_include_viewing_partner_repositioning(self) -> None:
        payload = _viewing_payload()
        moves = EnumeratePhase2RepairMoves(payload=payload, limit=24).get("moves") or []
        reasons = [str(move.get("reason") or "") for move in moves]

        self.assertTrue(
            any("viewing partner" in reason.lower() for reason in reasons),
            msg=f"Expected a viewing-partner macro move, got reasons={reasons}",
        )

    def test_viewing_macro_moves_include_pair_realignments(self) -> None:
        payload = _viewing_payload()
        moves = EnumeratePhase2RepairMoves(payload=payload, limit=64).get("moves") or []
        pair_moves = [
            move
            for move in moves
            if "realign viewing pair" in str(move.get("reason") or "").lower()
            and len(((move.get("proposal") or {}).get("cluster_transforms") or [])) >= 2
        ]

        self.assertTrue(pair_moves)

    def test_viewing_baseline_surfaces_directional_pair_pressure(self) -> None:
        payload = _viewing_payload()
        baseline = EvaluatePhase2Proposal(
            payload=payload,
            repair=make_no_improvement_repair(payload),
        )

        directional_rows = [
            row
            for row in (baseline.get("metrics", {}).get("orientation_debug") or [])
            if isinstance(row, dict)
            and str(row.get("kind") or "") == "cluster_directional_relation"
            and str(row.get("relation") or "") == "face_each_other"
        ]

        self.assertTrue(directional_rows)

    def test_viewing_baseline_keeps_macro_search_in_phase2(self) -> None:
        payload = _viewing_payload()
        baseline = EvaluatePhase2Proposal(
            payload=payload,
            repair=make_no_improvement_repair(payload),
        )

        self.assertTrue(_has_macro_orientation_pressure(baseline))
        self.assertFalse(
            _should_enter_object_refine_phase(
                iteration=6,
                history=[],
                macro_shortlist=[],
                baseline_evaluation=baseline,
            )
        )

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

    def test_partial_repair_output_is_merged_with_seed_layout(self) -> None:
        payload = _sample_payload()
        partial = {
            "status": "REPAIRED",
            "cluster_transforms": [
                {
                    **deepcopy(payload["seed_layout"]["cluster_transforms"][0]),
                    "x": int(payload["seed_layout"]["cluster_transforms"][0]["x"]) + 50,
                }
            ],
            "selected_variants": [],
            "object_repairs": [],
            "notes": ["Partial macro patch should inherit the rest from seed."],
        }

        normalized = normalize_and_validate_repair_output(payload, partial)
        transform_map = {
            row["cluster_id"]: row for row in normalized.get("cluster_transforms") or []
        }
        variant_map = {
            row["cluster_id"]: row for row in normalized.get("selected_variants") or []
        }

        self.assertEqual(
            set(transform_map.keys()),
            {
                row["cluster_id"]
                for row in payload["seed_layout"]["cluster_transforms"]
                if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
            },
        )
        self.assertEqual(
            set(variant_map.keys()),
            {
                row["cluster_id"]
                for row in payload["seed_layout"]["selected_variants"]
                if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
            },
        )
        changed_cluster_id = payload["seed_layout"]["cluster_transforms"][0][
            "cluster_id"
        ]
        self.assertEqual(
            transform_map[changed_cluster_id]["x"],
            int(payload["seed_layout"]["cluster_transforms"][0]["x"]) + 50,
        )

    def test_placer_rotates_models_on_quota_without_consuming_attempt(self) -> None:
        payload = _sample_payload()
        noop = make_no_improvement_repair(payload)
        llm_calls: list[str] = []

        def _llm_call(*, system_prompt: str, user_payload_json: str) -> dict[str, Any]:
            _ = system_prompt
            _ = user_payload_json
            llm_calls.append("call")
            if len(llm_calls) == 1:
                raise RetryablePlacerModelError(
                    failed_model="gemma-3-27b-it",
                    next_model="gemini-2.5-flash",
                    retry_delay_s=13.4,
                    detail="quota exceeded",
                )
            return noop

        placer = MacroClusterPlacer(llm_call=_llm_call)
        proposal = placer.generate_from_payload(payload=payload, max_attempts=1)

        self.assertEqual(proposal["status"], "NO_IMPROVEMENT")
        self.assertEqual(len(llm_calls), 2)

    def test_partial_repair_output_still_rejects_unknown_cluster_ids(self) -> None:
        payload = _sample_payload()
        invalid = {
            "status": "REPAIRED",
            "cluster_transforms": [
                {"cluster_id": "ghost_cluster", "x": 0, "y": 0, "rot": 0}
            ],
            "selected_variants": [],
            "object_repairs": [],
            "notes": ["Unknown cluster id should fail."],
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

    def test_compile_preserves_object_level_local_rotations(self) -> None:
        payload = _object_rotation_payload()
        noop = make_no_improvement_repair(payload)

        compiled = CompileAcceptedPhase2Proposal(payload=payload, repair=noop)
        by_object_id = {
            str(row["object_id"]): int(row["rotation_ccw"])
            for row in compiled["objects"]
            if isinstance(row, dict)
        }

        self.assertEqual(by_object_id.get("armchair"), 90)
        self.assertEqual(by_object_id.get("coffee_table"), 180)
        self.assertEqual(by_object_id.get("sofa"), 0)

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
                    "next_step_mode": "macro_layout",
                    "reasonableness_score": 95,
                    "top_issues": [],
                },
                search_phase="macro_layout",
            )
        )
        self.assertFalse(
            _should_promote_candidate(
                worse_evaluation,
                {
                    "verdict": "ACCEPT",
                    "next_step_mode": "stop",
                    "reasonableness_score": 95,
                    "top_issues": [],
                },
                search_phase="macro_layout",
            )
        )

    def test_small_revise_delta_is_not_promoted(self) -> None:
        evaluation = {
            "hard_valid": True,
            "baseline_comparison": {"delta_score": 60},
        }
        self.assertFalse(
            _should_promote_candidate(
                evaluation,
                {
                    "verdict": "REVISE",
                    "next_step_mode": "macro_layout",
                    "reasonableness_score": 78,
                    "top_issues": ["Orientation still needs a larger cluster move."],
                },
                search_phase="object_refine",
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
            str(repair.get("op") or "")
            for move in object_moves
            for repair in ((move.get("proposal") or {}).get("object_repairs") or [])
        }
        self.assertTrue(
            allowed_ops.issubset(
                {
                    "rotate_object",
                    "nudge_object",
                    "mirror_object",
                    "swap_objects",
                    "set_front_override",
                }
            )
        )
        self.assertTrue(
            all(
                len(
                    {
                        str(repair.get("cluster_id") or "")
                        for repair in (
                            (move.get("proposal") or {}).get("object_repairs") or []
                        )
                        if str(repair.get("cluster_id") or "")
                    }
                )
                <= 2
                for move in object_moves
            )
        )

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

        other_cluster_row = next(
            row
            for row in payload["objects_world"]
            if row["cluster_id"] != payload["objects_world"][0]["cluster_id"]
        )
        invalid_swap = {
            "status": "REPAIRED",
            "cluster_transforms": deepcopy(seed["cluster_transforms"]),
            "selected_variants": deepcopy(seed["selected_variants"]),
            "object_repairs": [
                {
                    "cluster_id": payload["objects_world"][0]["cluster_id"],
                    "object_id": payload["objects_world"][0]["object_id"],
                    "op": "swap_objects",
                    "params": {"other_object_id": other_cluster_row["object_id"]},
                }
            ],
            "notes": ["Unsupported risky swap."],
        }
        with self.assertRaises(ValueError):
            normalize_and_validate_repair_output(payload, invalid_swap)

    def test_object_refine_accepts_object_only_patch(self) -> None:
        payload = _sample_payload()
        payload["phase_control"]["repair_phase"] = "object_refine"
        payload = prepare_phase2_llm_payload(payload)
        enumerated = (payload.get("tool_context") or {}).get("enumerated_moves") or []
        proposal = next(
            (
                deepcopy(move.get("proposal"))
                for move in enumerated
                if isinstance(move, dict)
                and isinstance(move.get("proposal"), dict)
                and ((move.get("proposal") or {}).get("object_repairs") or [])
            ),
            None,
        )

        self.assertIsNotNone(proposal)
        if proposal is None:
            return

        proposal["status"] = "REPAIRED"
        proposal["cluster_transforms"] = []
        proposal["selected_variants"] = []
        proposal["notes"] = [
            "Object-only repair patch should inherit seed macro layout."
        ]

        normalized = normalize_and_validate_repair_output(payload, proposal)

        self.assertTrue(normalized["object_repairs"])
        self.assertEqual(
            normalized["cluster_transforms"],
            make_no_improvement_repair(payload)["cluster_transforms"],
        )
        self.assertEqual(
            normalized["selected_variants"],
            make_no_improvement_repair(payload)["selected_variants"],
        )

    def test_controller_merges_partial_patch_before_judge_and_follow_up(self) -> None:
        seed_payload = _sample_payload()
        changed_cluster = deepcopy(seed_payload["seed_layout"]["cluster_transforms"][0])
        changed_cluster["x"] = int(changed_cluster["x"])
        changed_cluster_id = str(changed_cluster["cluster_id"])
        expected_cluster_ids = {
            str(row["cluster_id"])
            for row in seed_payload["seed_layout"]["cluster_transforms"]
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
        }

        partial = {
            "status": "REPAIRED",
            "cluster_transforms": [changed_cluster],
            "selected_variants": [],
            "object_repairs": [],
            "notes": ["Partial macro patch should be merged before evaluation."],
        }

        def _llm_call(*, system_prompt: str, user_payload_json: str) -> dict[str, Any]:
            _ = system_prompt
            _ = user_payload_json
            return deepcopy(partial)

        judge = _RevisingJudge()
        controller = Phase2Controller(
            placer=MacroClusterPlacer(llm_call=_llm_call),
            judge=judge,  # type: ignore[arg-type]
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

        proposal_map = {
            row["cluster_id"]: row for row in result["proposal"]["cluster_transforms"]
        }

        self.assertEqual(set(proposal_map.keys()), expected_cluster_ids)
        self.assertEqual(proposal_map[changed_cluster_id]["x"], changed_cluster["x"])
        self.assertTrue(judge.payloads)

    def test_postsolve_orientation_snap_preserves_hard_validity(self) -> None:
        payload = _sample_payload()
        repair = make_no_improvement_repair(payload)

        snapped_payload, snapped_repair, snapped_eval = _postsolve_orientation_snap(
            payload=payload,
            repair=repair,
            rounds=1,
            move_limit=16,
        )

        self.assertTrue(snapped_eval["hard_valid"])
        self.assertIn("cluster_transforms", snapped_repair)
        self.assertIn("seed_layout", snapped_payload)

    def test_build_phase2_payload_injects_generic_cluster_variants(self) -> None:
        payload = _viewing_payload()

        cards = payload.get("cluster_cards") or []
        variant_counts = [
            len(card.get("available_variants") or [])
            for card in cards
            if isinstance(card, dict)
        ]
        self.assertTrue(any(count >= 3 for count in variant_counts))

        first_variant = next(
            variant
            for card in cards
            if isinstance(card, dict)
            for variant in (card.get("available_variants") or [])
            if isinstance(variant, dict)
        )
        self.assertIsInstance(first_variant.get("card"), dict)

    def test_postsolve_orientation_snap_prefers_object_move_when_it_improves(
        self,
    ) -> None:
        payload = {
            "seed_layout": {
                "cluster_transforms": [
                    {"cluster_id": "tv_media", "x": 0, "y": 0, "rot": 0}
                ],
                "selected_variants": [
                    {"cluster_id": "tv_media", "variant_id": "tv_media::base"}
                ],
            },
            "phase_control": {"repair_phase": "macro_layout"},
            "goals": {"relation_plan_used": {}},
        }
        repair = {
            "status": "REPAIRED",
            "cluster_transforms": [
                {"cluster_id": "tv_media", "x": 0, "y": 0, "rot": 0}
            ],
            "selected_variants": [
                {"cluster_id": "tv_media", "variant_id": "tv_media::base"}
            ],
            "object_repairs": [],
            "notes": [],
        }

        original_enum = _phase2_controller.EnumeratePhase2RepairMoves
        original_eval = _phase2_controller.EvaluatePhase2Proposal
        original_promote = _phase2_controller.PromotePhase2RepairToSeedPayload

        def _fake_enum(*, payload: dict[str, Any], limit: int = 48) -> dict[str, Any]:
            _ = limit
            phase = (
                (payload.get("phase_control") or {}).get("repair_phase") or ""
            ).strip()
            if phase == "object_refine":
                return {
                    "moves": [
                        {
                            "kind": "object_pose",
                            "cluster_id": "tv_media",
                            "object_id": "tv_console",
                            "proposal": {
                                "cluster_transforms": [],
                                "selected_variants": [],
                                "object_repairs": [
                                    {
                                        "cluster_id": "tv_media",
                                        "object_id": "tv_console",
                                        "op": "rotate_object",
                                        "params": {"rot": 90},
                                    }
                                ],
                            },
                            "reason": "Rotate tv console toward seating",
                        }
                    ]
                }
            return {
                "moves": [
                    {
                        "kind": "cluster_pose",
                        "cluster_id": "tv_media",
                        "proposal": {
                            "cluster_transforms": [
                                {
                                    "cluster_id": "tv_media",
                                    "x": 0,
                                    "y": 0,
                                    "rot": 90,
                                }
                            ],
                            "selected_variants": [],
                            "object_repairs": [],
                        },
                        "reason": "Rotate the whole TV cluster",
                    }
                ]
            }

        def _fake_eval(
            *, payload: dict[str, Any], repair: dict[str, Any]
        ) -> dict[str, Any]:
            _ = payload
            has_object_repairs = bool(repair.get("object_repairs"))
            delta = 18 if has_object_repairs else 60
            return {
                "hard_valid": True,
                "baseline_comparison": {
                    "delta_score": delta,
                    "candidate_score": delta,
                },
            }

        def _fake_promote(
            *, payload: dict[str, Any], repair: dict[str, Any]
        ) -> dict[str, Any]:
            promoted = deepcopy(payload)
            promoted["selected_kind"] = (
                "object" if repair.get("object_repairs") else "cluster"
            )
            return promoted

        try:
            _phase2_controller.EnumeratePhase2RepairMoves = _fake_enum
            _phase2_controller.EvaluatePhase2Proposal = _fake_eval
            _phase2_controller.PromotePhase2RepairToSeedPayload = _fake_promote

            snapped_payload, _, snapped_eval = _postsolve_orientation_snap(
                payload=payload,
                repair=repair,
                rounds=1,
                move_limit=8,
            )
        finally:
            _phase2_controller.EnumeratePhase2RepairMoves = original_enum
            _phase2_controller.EvaluatePhase2Proposal = original_eval
            _phase2_controller.PromotePhase2RepairToSeedPayload = original_promote

        self.assertTrue(snapped_eval["hard_valid"])
        self.assertEqual(snapped_payload.get("selected_kind"), "object")

    def test_face_object_orientation_is_scored_for_targeted_media_objects(self) -> None:
        payload = _variant_preview_payload()

        evaluation = EvaluatePhase2Proposal(
            payload=payload,
            repair=make_no_improvement_repair(payload),
        )

        rows = [
            row
            for row in (evaluation.get("metrics", {}).get("orientation_debug") or [])
            if isinstance(row, dict)
            and row.get("cluster_id") == "media_zone"
            and row.get("object_id") == "tv_console"
            and row.get("intent") == "face_object"
        ]

        self.assertTrue(rows)
        self.assertEqual(rows[0].get("target_object_id"), "sectional_sofa")

    def test_object_refine_enumeration_includes_relation_driven_tv_rotation(
        self,
    ) -> None:
        payload = _variant_preview_payload()
        payload.setdefault("phase_control", {})["repair_phase"] = "object_refine"

        moves = EnumeratePhase2RepairMoves(payload=payload, limit=64).get("moves") or []
        tv_rotation_moves = [
            move
            for move in moves
            if move.get("cluster_id") == "media_zone"
            and move.get("object_id") == "tv_console"
            and (
                ((move.get("proposal") or {}).get("object_repairs") or [{}])[0].get(
                    "op"
                )
                == "rotate_object"
            )
        ]

        self.assertTrue(tv_rotation_moves)

    def test_phase2_scores_semantic_proximity_from_cluster_rules(self) -> None:
        payload, cluster_id, object_id = _payload_with_semantic_spacing_pressure()

        evaluation = EvaluatePhase2Proposal(
            payload=payload,
            repair=make_no_improvement_repair(payload),
        )

        rows = [
            row
            for row in (
                evaluation.get("metrics", {}).get(
                    "cluster_internal_constraint_fidelity"
                )
                or []
            )
            if isinstance(row, dict)
            and row.get("cluster_id") == cluster_id
            and row.get("constraint_type") == "semantic_proximity"
            and (row.get("subjects") or {}).get("a") == object_id
        ]

        self.assertTrue(rows)
        self.assertTrue(any(row.get("proximity") == "loose" for row in rows))
        self.assertTrue(
            any(
                row.get("mapped_side") in {"left", "right", "top", "bottom"}
                for row in rows
            )
        )
        self.assertTrue(any(int(row.get("penalty_mm") or 0) > 0 for row in rows))

    def test_object_refine_enumeration_includes_spacing_nudge_moves(self) -> None:
        payload, cluster_id, object_id = _payload_with_semantic_spacing_pressure()
        payload.setdefault("phase_control", {})["repair_phase"] = "object_refine"

        moves = (
            EnumeratePhase2RepairMoves(payload=payload, limit=200).get("moves") or []
        )
        spacing_moves = [
            move
            for move in moves
            if move.get("cluster_id") == cluster_id
            and move.get("object_id") == object_id
            and "spacing" in str(move.get("reason") or "").lower()
            and (
                ((move.get("proposal") or {}).get("object_repairs") or [{}])[0].get(
                    "op"
                )
                == "nudge_object"
            )
        ]

        self.assertTrue(spacing_moves)

    def test_object_refine_enumeration_includes_two_cluster_repairs(self) -> None:
        payload = _variant_preview_payload()
        payload.setdefault("phase_control", {})["repair_phase"] = "object_refine"

        moves = EnumeratePhase2RepairMoves(payload=payload, limit=96).get("moves") or []
        paired_moves = [
            move
            for move in moves
            if len(((move.get("proposal") or {}).get("object_repairs") or [])) == 2
            and len(
                {
                    str(repair.get("cluster_id") or "")
                    for repair in (
                        (move.get("proposal") or {}).get("object_repairs") or []
                    )
                    if str(repair.get("cluster_id") or "")
                }
            )
            == 2
        ]

        self.assertTrue(paired_moves)

    def test_build_phase2_preview_candidate_compiles_fast_absolute_layout(self) -> None:
        payload = _variant_preview_payload()

        preview = build_phase2_preview_candidate(
            payload=payload,
            rounds=1,
            move_limit=16,
        )

        self.assertEqual(preview["absolute_layout"].get("status"), "OK")
        self.assertTrue(preview["evaluation"].get("hard_valid"))

    def test_compile_accepted_phase2_proposal_exports_directional_rotation_from_front_world(
        self,
    ) -> None:
        payload = _variant_preview_payload()

        compiled = CompileAcceptedPhase2Proposal(
            payload=payload,
            repair=make_no_improvement_repair(payload),
        )

        rotation_by_object = {
            str(row.get("object_id") or ""): int(row.get("rotation_ccw") or 0)
            for row in compiled.get("objects") or []
            if isinstance(row, dict)
        }

        self.assertEqual(rotation_by_object.get("tv_console"), 180)
        self.assertEqual(rotation_by_object.get("sectional_sofa"), 0)

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

    def test_object_refine_can_start_when_planner_object_pressure_dominates(
        self,
    ) -> None:
        baseline = {
            "metrics": {
                "orientation_debug": [
                    {
                        "kind": "cluster_directional_relation",
                        "cluster_id": "seating_zone",
                        "penalty_mm": 900,
                        "relation": "face_each_other",
                    },
                    {
                        "kind": "object_orientation",
                        "cluster_id": "seating_zone",
                        "object_id": "sectional_sofa",
                        "intent": "face_object",
                        "penalty_mm": 2200,
                    },
                    {
                        "kind": "object_orientation",
                        "cluster_id": "media_zone",
                        "object_id": "tv_console",
                        "intent": "face_object",
                        "penalty_mm": 1800,
                    },
                ]
            },
            "diagnosis": {
                "prioritized_objects": [
                    {"cluster_id": "seating_zone", "object_id": "sectional_sofa"},
                    {"cluster_id": "media_zone", "object_id": "tv_console"},
                ]
            },
        }

        self.assertTrue(
            _should_enter_object_refine_phase(
                iteration=2,
                history=[
                    {
                        "search_phase": "macro_layout",
                        "tool_evaluation": {
                            "hard_valid": True,
                            "baseline_comparison": {"delta_score": 0},
                        },
                        "judge_evaluation": {"verdict": "REVISE"},
                    }
                ],
                macro_shortlist=[],
                baseline_evaluation=baseline,
            )
        )

    def test_macro_search_stops_after_repeated_small_revise(self) -> None:
        history = [
            {
                "search_phase": "macro_layout",
                "tool_evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": 40},
                },
                "judge_evaluation": {
                    "verdict": "REVISE",
                    "priority_clusters": ["seating_area", "tv_zone"],
                    "top_issues": ["TV and sofa orientation still feels wrong."],
                    "repair_advice": ["Try a different macro direction."],
                    "next_step_mode": "macro_layout",
                },
                "direction_summary": {
                    "primary_clusters": ["seating_area"],
                    "move_families": ["cluster_translation"],
                },
            },
            {
                "search_phase": "macro_layout",
                "tool_evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": 55},
                },
                "judge_evaluation": {
                    "verdict": "REVISE",
                    "priority_clusters": ["seating_area", "tv_zone"],
                    "top_issues": ["TV and sofa orientation still feels wrong."],
                    "repair_advice": ["Try a different macro direction."],
                    "next_step_mode": "macro_layout",
                },
                "direction_summary": {
                    "primary_clusters": ["seating_area"],
                    "move_families": ["cluster_translation"],
                },
            },
            {
                "search_phase": "macro_layout",
                "tool_evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": 70},
                },
                "judge_evaluation": {
                    "verdict": "REVISE",
                    "priority_clusters": ["seating_area", "tv_zone"],
                    "top_issues": ["TV and sofa orientation still feels wrong."],
                    "repair_advice": ["Try a different macro direction."],
                    "next_step_mode": "macro_layout",
                },
                "direction_summary": {
                    "primary_clusters": ["tv_zone"],
                    "move_families": ["cluster_rotation"],
                },
            },
            {
                "search_phase": "macro_layout",
                "tool_evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": 65},
                },
                "judge_evaluation": {
                    "verdict": "REVISE",
                    "priority_clusters": ["seating_area", "tv_zone"],
                    "top_issues": ["TV and sofa orientation still feels wrong."],
                    "repair_advice": ["Try a different macro direction."],
                    "next_step_mode": "macro_layout",
                },
                "direction_summary": {
                    "primary_clusters": ["seating_area"],
                    "move_families": ["cluster_translation"],
                },
            },
        ]
        shortlist = [
            {
                "evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"delta_score": 90},
                }
            }
        ]
        self.assertTrue(
            _should_stop_macro_search(
                history=history,
                deterministic_shortlist=shortlist,
                search_phase="macro_layout",
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
        self.assertEqual(result["next_step_mode"], "stop")

    def test_judge_calibration_prefers_macro_layout_for_orientation_issues(
        self,
    ) -> None:
        payload = {
            "hard_check_summary": {"hard_valid": True},
            "comparison_summary": {"delta_score": 90},
            "metrics": {
                "cluster_affinity_to_preferred_zone": [],
                "opening_band_blocking": [],
                "main_path_clearance": {"paths": []},
                "central_congestion": [],
                "cluster_edge_vs_center_fit": [],
                "cluster_internal_constraint_fidelity": [],
            },
            "diagnosis": {
                "prioritized_clusters": [
                    {"cluster_id": "seating_area"},
                    {"cluster_id": "tv_zone"},
                ]
            },
        }

        result = run_phase2_judge(
            payload=payload,
            llm_call=lambda system_prompt, user_payload_json: {
                "reasonableness_score": 68,
                "verdict": "REVISE",
                "top_issues": ["TV and sofa orientation still feels wrong."],
                "repair_advice": ["Try a different macro direction."],
                "priority_clusters": ["seating_area", "tv_zone"],
            },
        )

        self.assertEqual(result["next_step_mode"], "macro_layout")

    def test_judge_calibration_prefers_object_refine_for_local_clearance_issue(
        self,
    ) -> None:
        payload = {
            "hard_check_summary": {"hard_valid": True},
            "comparison_summary": {"delta_score": 220},
            "metrics": {
                "cluster_affinity_to_preferred_zone": [],
                "opening_band_blocking": [],
                "main_path_clearance": {"paths": []},
                "central_congestion": [],
                "cluster_edge_vs_center_fit": [],
                "cluster_internal_constraint_fidelity": [],
            },
            "diagnosis": {
                "prioritized_clusters": [{"cluster_id": "work_area"}],
                "prioritized_objects": [
                    {"cluster_id": "work_area", "object_id": "desk_chair"}
                ],
            },
        }

        result = run_phase2_judge(
            payload=payload,
            llm_call=lambda system_prompt, user_payload_json: {
                "reasonableness_score": 76,
                "verdict": "REVISE",
                "top_issues": ["Chair still needs front clearance tuning."],
                "repair_advice": ["Use a smaller local object adjustment."],
                "priority_clusters": ["work_area"],
            },
        )

        self.assertEqual(result["next_step_mode"], "object_refine")

    def test_judge_calibration_prefers_object_refine_for_planner_object_pressure(
        self,
    ) -> None:
        payload = {
            "hard_check_summary": {"hard_valid": True},
            "comparison_summary": {"delta_score": 240},
            "metrics": {
                "cluster_affinity_to_preferred_zone": [],
                "opening_band_blocking": [],
                "main_path_clearance": {"paths": []},
                "central_congestion": [],
                "cluster_edge_vs_center_fit": [],
                "cluster_internal_constraint_fidelity": [],
                "orientation_debug": [
                    {
                        "kind": "cluster_directional_relation",
                        "cluster_id": "seating_zone",
                        "penalty_mm": 800,
                        "relation": "face_each_other",
                    },
                    {
                        "kind": "object_orientation",
                        "cluster_id": "seating_zone",
                        "object_id": "sectional_sofa",
                        "intent": "face_object",
                        "penalty_mm": 2100,
                    },
                    {
                        "kind": "object_orientation",
                        "cluster_id": "media_zone",
                        "object_id": "tv_console",
                        "intent": "face_object",
                        "penalty_mm": 1800,
                    },
                ],
            },
            "diagnosis": {
                "prioritized_clusters": [
                    {"cluster_id": "seating_zone"},
                    {"cluster_id": "media_zone"},
                ],
                "prioritized_objects": [
                    {"cluster_id": "seating_zone", "object_id": "sectional_sofa"},
                    {"cluster_id": "media_zone", "object_id": "tv_console"},
                ],
            },
        }

        result = run_phase2_judge(
            payload=payload,
            llm_call=lambda system_prompt, user_payload_json: {
                "reasonableness_score": 74,
                "verdict": "REVISE",
                "top_issues": ["TV and sofa orientation still feels wrong."],
                "repair_advice": ["Keep refining object orientation."],
                "priority_clusters": ["seating_zone", "media_zone"],
            },
        )

        self.assertEqual(result["next_step_mode"], "object_refine")

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
