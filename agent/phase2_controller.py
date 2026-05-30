from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agent.cluster_placer import (
    MacroClusterPlacer,
    build_phase2_payload,
    compile_phase2_to_final_output,
    make_no_improvement_repair,
)
from agent.micro_refiner import MicroRefiner
from cluster_placer.tools_v2 import (
    EnumeratePhase2RepairMoves,
    EvaluatePhase2Proposal,
    PromotePhase2RepairToSeedPayload,
)

logger = logging.getLogger(__name__)


def _normalize_solver_output_for_phase2(
    solver_output: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = deepcopy(solver_output or {})
    placer_seed = normalized.get("placer_seed") or {}
    seed_layout = placer_seed.get("seed_layout") or {}

    if isinstance(seed_layout, dict):
        if not normalized.get("cluster_transforms") and seed_layout.get(
            "cluster_transforms"
        ):
            normalized["cluster_transforms"] = deepcopy(
                seed_layout.get("cluster_transforms") or []
            )
        if not normalized.get("selected_variants") and seed_layout.get(
            "selected_variants"
        ):
            normalized["selected_variants"] = deepcopy(
                seed_layout.get("selected_variants") or []
            )

    return normalized


def _phase2_controller_result(
    *,
    proposal: dict[str, Any],
    evaluation: dict[str, Any],
    absolute_layout: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    hard_fix: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    hard_valid = bool(evaluation.get("hard_valid"))
    refined_layout_solution = {
        "status": "OK"
        if str(absolute_layout.get("status") or "").upper() == "OK"
        else "PARTIAL_OK",
        "hard_valid": hard_valid,
        "quality_before": {},
        "quality_after": {},
        "applied_moves": [],
        "notes": list(notes or []),
    }
    return {
        "proposal": proposal,
        "tool_evaluation": evaluation,
        "judge_evaluation": {
            "verdict": "ACCEPT" if hard_valid else "REVISE",
            "reasonableness_score": int(
                max(
                    0,
                    (
                        (evaluation.get("baseline_comparison") or {}).get(
                            "candidate_score"
                        )
                        or 0
                    ),
                )
                // 100
            ),
            "next_step_mode": "stop" if hard_valid else "macro_layout",
            "top_issues": [],
            "repair_advice": list(notes or []),
            "priority_clusters": [],
        },
        "absolute_layout": absolute_layout,
        "refined_layout_solution": refined_layout_solution,
        "history": history or [],
        "hard_fix": hard_fix or {"result": "SKIPPED", "attempts": []},
    }


def _candidate_score(evaluation: dict[str, Any]) -> int:
    return int(
        ((evaluation.get("baseline_comparison") or {}).get("candidate_score")) or 0
    )


def _delta_score(evaluation: dict[str, Any]) -> int:
    return int(((evaluation.get("baseline_comparison") or {}).get("delta_score")) or 0)


def _search_candidate_rank(
    evaluation: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    delta = _delta_score(evaluation)
    return (
        1 if bool(evaluation.get("hard_valid")) else 0,
        1 if delta > 0 else 0,
        delta,
        _candidate_score(evaluation),
        -len(evaluation.get("errors") or []),
    )


def _normalize_search_phase(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "object_refine":
        return "object_refine"
    return "macro_layout"


def _set_search_phase(payload: dict[str, Any], search_phase: str) -> None:
    phase_control = deepcopy(payload.get("phase_control") or {})
    phase_control["repair_phase"] = _normalize_search_phase(search_phase)
    payload["phase_control"] = phase_control


def _seed_transform_map(seed_layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in seed_layout.get("cluster_transforms") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _seed_variant_map(seed_layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in seed_layout.get("selected_variants") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _canonical_repair(
    payload: dict[str, Any],
    repair: dict[str, Any] | None = None,
    *,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)
    vmap = _seed_variant_map(seed_layout)

    for row in (repair or {}).get("cluster_transforms") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            tmap[row["cluster_id"]] = deepcopy(row)

    for row in (repair or {}).get("selected_variants") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            vmap[row["cluster_id"]] = deepcopy(row)

    merged_notes: list[str] = []
    for source in (((repair or {}).get("notes") or []), notes or []):
        for item in source:
            text = str(item).strip()
            if text and text not in merged_notes:
                merged_notes.append(text)

    cluster_ids = sorted(tmap.keys())
    return {
        "status": "REPAIRED",
        "cluster_transforms": [deepcopy(tmap[cid]) for cid in cluster_ids],
        "selected_variants": [
            deepcopy(vmap[cid]) for cid in cluster_ids if cid in vmap
        ],
        "object_repairs": deepcopy((repair or {}).get("object_repairs") or []),
        "notes": merged_notes,
    }


def _proposal_signature(proposal: dict[str, Any]) -> str:
    return json.dumps(
        {
            "cluster_transforms": proposal.get("cluster_transforms") or [],
            "selected_variants": proposal.get("selected_variants") or [],
            "object_repairs": proposal.get("object_repairs") or [],
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _is_orientation_snap_move(payload: dict[str, Any], move: dict[str, Any]) -> bool:
    if not isinstance(move, dict):
        return False
    kind = str(move.get("kind") or "")
    proposal = move.get("proposal") or {}
    if kind == "cluster_variant":
        return bool((proposal.get("selected_variants") or []))
    if kind != "cluster_pose":
        return False

    transforms = proposal.get("cluster_transforms") or []
    if len(transforms) != 1 or not isinstance(transforms[0], dict):
        return False
    transform = transforms[0]
    cluster_id = str(transform.get("cluster_id") or "")
    if not cluster_id:
        return False
    seed_map = _seed_transform_map(payload.get("seed_layout") or {})
    seed_row = seed_map.get(cluster_id) or {}
    return int(transform.get("rot") or 0) != int(seed_row.get("rot") or 0)


def _is_object_orientation_snap_move(move: dict[str, Any]) -> bool:
    if not isinstance(move, dict):
        return False
    if str(move.get("kind") or "") != "object_pose":
        return False
    repairs = (move.get("proposal") or {}).get("object_repairs") or []
    if len(repairs) != 1 or not isinstance(repairs[0], dict):
        return False
    return str(repairs[0].get("op") or "") in {
        "rotate_object",
        "mirror_object",
        "set_front_override",
    }


def _relation_plan_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    return relation_plan if isinstance(relation_plan, dict) else {}


def _normalized_cluster_pair(a: Any, b: Any) -> tuple[str, str] | None:
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right or left == right:
        return None
    return (left, right)


def _anchor_layout_hints_by_cluster(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    relation_plan = _relation_plan_from_payload(payload)
    candidates = []
    for container in (
        relation_plan,
        relation_plan.get("macro_concept")
        if isinstance(relation_plan.get("macro_concept"), dict)
        else None,
        (payload.get("goals") or {}),
    ):
        if isinstance(container, dict):
            candidates.append(container.get("anchor_layout_hints_by_cluster"))
    out: dict[str, dict[str, Any]] = {}
    for mapping in candidates:
        if not isinstance(mapping, dict):
            continue
        for cluster_id, row in mapping.items():
            cid = str(cluster_id or "").strip()
            if not cid or not isinstance(row, dict):
                continue
            out[cid] = deepcopy(row)
    return out


def _anchor_pairs_from_payload(payload: dict[str, Any]) -> list[tuple[str, str, str]]:
    relation_plan = _relation_plan_from_payload(payload)
    pair_rows: list[tuple[str, str, str]] = []
    raw_sources = [
        relation_plan.get("anchor_pairs"),
        (relation_plan.get("macro_concept") or {}).get("anchor_pairs")
        if isinstance(relation_plan.get("macro_concept"), dict)
        else None,
        (
            (payload.get("goals") or {}).get("anchor_pairs")
            if isinstance((payload.get("goals") or {}), dict)
            else None
        ),
    ]
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_sources:
        if not isinstance(raw, list):
            continue
        for row in raw:
            if not isinstance(row, dict):
                continue
            pair = _normalized_cluster_pair(
                row.get("left_cluster_id") or row.get("a") or row.get("left"),
                row.get("right_cluster_id") or row.get("b") or row.get("right"),
            )
            if pair is None:
                continue
            mode = (
                str(
                    row.get("mode")
                    or row.get("relation")
                    or row.get("focus_mode")
                    or ""
                )
                .strip()
                .lower()
            )
            if mode in {"viewing", "face_each_other", "turn_toward"}:
                relation = "face_each_other"
            elif mode in {"access", "access_faces_other", "mixed"}:
                relation = "access_faces_other"
            else:
                relation = "face_each_other"
            key = (pair[0], pair[1], relation)
            if key in seen:
                continue
            seen.add(key)
            pair_rows.append(key)
    return pair_rows


def _viewing_cluster_pairs(payload: dict[str, Any]) -> list[tuple[str, str]]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    layout_intent = relation_plan.get("layout_intent_profile") or {}
    primary_cluster_id = layout_intent.get("primary_cluster_id")
    secondary_cluster_id = layout_intent.get("secondary_cluster_id")
    focus_mode = str(layout_intent.get("focus_mode") or "").strip().lower()
    pairs: list[tuple[str, str]] = []

    if (
        focus_mode in {"viewing", "mixed"}
        and isinstance(primary_cluster_id, str)
        and primary_cluster_id
        and isinstance(secondary_cluster_id, str)
        and secondary_cluster_id
        and secondary_cluster_id != primary_cluster_id
    ):
        pairs.append((primary_cluster_id, secondary_cluster_id))

    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation not in {"face_each_other", "access_faces_other"}:
            continue
        a = row.get("a")
        b = row.get("b")
        if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
            continue
        pair = (a, b)
        if pair not in pairs and (b, a) not in pairs:
            pairs.append(pair)

    return pairs


def _needs_primary_pair_axis_repair(
    payload: dict[str, Any],
    evaluation: dict[str, Any],
) -> bool:
    pair_clusters: set[str] = set()
    for a, b in _viewing_cluster_pairs(payload):
        pair_clusters.add(a)
        pair_clusters.add(b)
    if not pair_clusters:
        return False
    metrics = evaluation.get("metrics") or {}
    gate_reasons = {
        str(reason).strip()
        for reason in (metrics.get("score_summary") or {}).get(
            "quality_gate_reasons", []
        )
        if str(reason).strip()
    }
    if {
        "critical_orientation_penalty_too_high",
        "focal_pair_penalty_too_high",
    }.intersection(gate_reasons):
        return True
    for row in metrics.get("orientation_debug") or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("penalty_mm") or 0) < 180:
            continue
        kind = str(row.get("kind") or "").strip()
        cluster_id = str(row.get("cluster_id") or "").strip()
        if kind in {
            "cluster_directional_relation",
            "cluster_orientation",
            "object_orientation",
        }:
            if not cluster_id or cluster_id in pair_clusters:
                return True
    diagnosis = evaluation.get("diagnosis") or {}
    prioritized_objects = {
        str(row.get("object_id") or row.get("id") or "").strip()
        for row in (diagnosis.get("prioritized_objects") or [])
        if isinstance(row, dict)
    }
    if prioritized_objects.intersection(
        {"sofa", "armchair", "coffee_table", "tv_console"}
    ):
        return True
    return False


def _anchor_contract_cluster_ids(payload: dict[str, Any]) -> set[str]:
    relation_plan = _relation_plan_from_payload(payload)
    protected_cluster_ids: set[str] = set()
    for left_cluster_id, right_cluster_id in _viewing_cluster_pairs(payload):
        protected_cluster_ids.add(left_cluster_id)
        protected_cluster_ids.add(right_cluster_id)
    for left_cluster_id, right_cluster_id, _relation in _anchor_pairs_from_payload(
        payload
    ):
        protected_cluster_ids.add(left_cluster_id)
        protected_cluster_ids.add(right_cluster_id)
    for cluster_id, hint in _anchor_layout_hints_by_cluster(payload).items():
        if not isinstance(hint, dict):
            continue
        if bool(hint.get("protect_anchor_orientation", True)) or bool(
            hint.get("protect_pair_axis", False)
        ):
            protected_cluster_ids.add(cluster_id)
    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        intents = {
            str(intent).strip().lower()
            for intent in (row.get("intents") or [])
            if str(intent).strip()
        }
        if "face_cluster" not in intents:
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        target_cluster_id = str(row.get("target_cluster_id") or "").strip()
        if cluster_id:
            protected_cluster_ids.add(cluster_id)
        if target_cluster_id:
            protected_cluster_ids.add(target_cluster_id)
    return protected_cluster_ids


def _anchor_ids_by_cluster(payload: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    hints_by_cluster = _anchor_layout_hints_by_cluster(payload)
    for card in payload.get("cluster_cards") or []:
        if not isinstance(card, dict):
            continue
        cluster_id = str(card.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        anchors = {
            str(anchor).strip()
            for anchor in (card.get("anchors") or [])
            if isinstance(anchor, str) and str(anchor).strip()
        }
        anchor_override = str(card.get("anchor_override") or "").strip()
        if anchor_override:
            anchors.add(anchor_override)
        rules = card.get("cluster_rules") or {}
        if isinstance(rules, dict):
            anchor_first_policy = rules.get("anchor_first_policy") or {}
            if isinstance(anchor_first_policy, dict):
                dominant_anchor_id = str(
                    anchor_first_policy.get("dominant_anchor_id") or ""
                ).strip()
                if dominant_anchor_id:
                    anchors.add(dominant_anchor_id)
                for candidate in (
                    anchor_first_policy.get("dominant_anchor_candidates") or []
                ):
                    if isinstance(candidate, str) and candidate.strip():
                        anchors.add(candidate.strip())
        hint = hints_by_cluster.get(cluster_id) or {}
        if isinstance(hint, dict):
            for key in (
                "dominant_anchor_id",
                "anchor_object_id",
                "primary_anchor_object_id",
            ):
                value = str(hint.get(key) or "").strip()
                if value:
                    anchors.add(value)
        for row in card.get("decisions") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("priority") or "").strip().lower() != "anchor":
                continue
            object_id = str(row.get("object_type") or row.get("category") or "").strip()
            if object_id:
                anchors.add(object_id)
        out[cluster_id] = anchors
    return out


def _proposal_breaks_anchor_contract(
    payload: dict[str, Any],
    proposal: dict[str, Any],
) -> bool:
    protected_cluster_ids = _anchor_contract_cluster_ids(payload)
    if not protected_cluster_ids:
        return False

    seed_map = _seed_transform_map(payload.get("seed_layout") or {})
    for row in proposal.get("cluster_transforms") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        if cluster_id not in protected_cluster_ids:
            continue
        seed_rot = int((seed_map.get(cluster_id) or {}).get("rot") or 0) % 360
        proposal_rot = int(row.get("rot") or 0) % 360
        if proposal_rot != seed_rot:
            return True

    anchor_ids_by_cluster = _anchor_ids_by_cluster(payload)
    for repair in proposal.get("object_repairs") or []:
        if not isinstance(repair, dict):
            continue
        cluster_id = str(repair.get("cluster_id") or "").strip()
        if cluster_id not in protected_cluster_ids:
            continue
        op = str(repair.get("op") or "").strip()
        object_id = str(repair.get("object_id") or "").strip()
        anchor_ids = anchor_ids_by_cluster.get(cluster_id, set())
        if op == "set_anchor":
            return True
        if op in {"rotate_object", "mirror_object", "set_front_override"}:
            if object_id and object_id in anchor_ids:
                return True
        if op == "swap_objects":
            other_object_id = str(
                (repair.get("params") or {}).get("other_object_id") or ""
            ).strip()
            if object_id in anchor_ids or other_object_id in anchor_ids:
                return True
    return False


def _postsolve_orientation_snap(
    *,
    payload: dict[str, Any],
    repair: dict[str, Any],
    rounds: int = 2,
    move_limit: int = 48,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    promoted_payload = PromotePhase2RepairToSeedPayload(payload=payload, repair=repair)
    baseline_repair = _canonical_repair(promoted_payload)
    baseline_evaluation = EvaluatePhase2Proposal(
        payload=promoted_payload,
        repair=baseline_repair,
    )

    if not bool(baseline_evaluation.get("hard_valid")):
        return payload, repair, baseline_evaluation

    current_payload = promoted_payload
    current_repair = baseline_repair
    current_evaluation = baseline_evaluation

    for _ in range(max(1, int(rounds))):
        macro_moves = (
            EnumeratePhase2RepairMoves(
                payload=current_payload,
                limit=move_limit,
            ).get("moves")
            or []
        )
        object_payload = deepcopy(current_payload)
        _set_search_phase(object_payload, "object_refine")
        object_moves = (
            EnumeratePhase2RepairMoves(
                payload=object_payload,
                limit=move_limit,
            ).get("moves")
            or []
        )

        best_macro_candidate: tuple[dict[str, Any], dict[str, Any]] | None = None
        best_macro_delta = 0
        best_object_candidate: tuple[dict[str, Any], dict[str, Any]] | None = None
        best_object_delta = 0
        seen_signatures: set[str] = set()
        orientation_moves_by_cluster: dict[str, list[dict[str, Any]]] = {}

        for move in object_moves:
            if not _is_object_orientation_snap_move(move):
                continue
            proposal = _canonical_repair(
                current_payload,
                move.get("proposal") or {},
                notes=[
                    "Post-solve object orientation snap: "
                    f"{move.get('reason') or 'object orientation move'}"
                ],
            )
            signature = _proposal_signature(proposal)
            if signature in seen_signatures:
                continue
            if _proposal_breaks_anchor_contract(current_payload, proposal):
                continue
            seen_signatures.add(signature)
            evaluation = EvaluatePhase2Proposal(
                payload=current_payload,
                repair=proposal,
            )
            if not bool(evaluation.get("hard_valid")):
                continue
            delta_score = _delta_score(evaluation)
            if delta_score > best_object_delta:
                best_object_delta = delta_score
                best_object_candidate = (proposal, evaluation)

        for move in macro_moves:
            if not _is_orientation_snap_move(current_payload, move):
                continue
            cluster_id = str(move.get("cluster_id") or "")
            if cluster_id:
                orientation_moves_by_cluster.setdefault(cluster_id, []).append(move)
            proposal = _canonical_repair(
                current_payload,
                move.get("proposal") or {},
                notes=[
                    f"Post-solve orientation snap: {move.get('reason') or 'orientation move'}"
                ],
            )
            signature = _proposal_signature(proposal)
            if signature in seen_signatures:
                continue
            if _proposal_breaks_anchor_contract(current_payload, proposal):
                continue
            seen_signatures.add(signature)
            evaluation = EvaluatePhase2Proposal(
                payload=current_payload, repair=proposal
            )
            if not bool(evaluation.get("hard_valid")):
                continue
            delta_score = _delta_score(evaluation)
            if delta_score > best_macro_delta:
                best_macro_delta = delta_score
                best_macro_candidate = (proposal, evaluation)

        for left_cluster_id, right_cluster_id in _viewing_cluster_pairs(
            current_payload
        ):
            left_moves = orientation_moves_by_cluster.get(left_cluster_id) or []
            right_moves = orientation_moves_by_cluster.get(right_cluster_id) or []
            for left_move in left_moves[:4]:
                for right_move in right_moves[:4]:
                    pair_partial = {
                        "cluster_transforms": deepcopy(
                            (
                                (left_move.get("proposal") or {}).get(
                                    "cluster_transforms"
                                )
                                or []
                            )
                        )
                        + deepcopy(
                            (
                                (right_move.get("proposal") or {}).get(
                                    "cluster_transforms"
                                )
                                or []
                            )
                        ),
                        "selected_variants": deepcopy(
                            (
                                (left_move.get("proposal") or {}).get(
                                    "selected_variants"
                                )
                                or []
                            )
                        )
                        + deepcopy(
                            (
                                (right_move.get("proposal") or {}).get(
                                    "selected_variants"
                                )
                                or []
                            )
                        ),
                        "object_repairs": [],
                    }
                    proposal = _canonical_repair(
                        current_payload,
                        pair_partial,
                        notes=[
                            "Post-solve orientation snap: paired focal reorientation for "
                            f"{left_cluster_id} and {right_cluster_id}"
                        ],
                    )
                    signature = _proposal_signature(proposal)
                    if signature in seen_signatures:
                        continue
                    if _proposal_breaks_anchor_contract(current_payload, proposal):
                        continue
                    seen_signatures.add(signature)
                    evaluation = EvaluatePhase2Proposal(
                        payload=current_payload,
                        repair=proposal,
                    )
                    if not bool(evaluation.get("hard_valid")):
                        continue
                    delta_score = _delta_score(evaluation)
                    if delta_score > best_macro_delta:
                        best_macro_delta = delta_score
                        best_macro_candidate = (proposal, evaluation)

        chosen_candidate: tuple[dict[str, Any], dict[str, Any]] | None = None
        chosen_delta = 0
        if best_object_candidate is not None and best_object_delta > 0:
            chosen_candidate = best_object_candidate
            chosen_delta = best_object_delta
        elif best_macro_candidate is not None and best_macro_delta > 0:
            chosen_candidate = best_macro_candidate
            chosen_delta = best_macro_delta

        if chosen_candidate is None or chosen_delta <= 0:
            break

        candidate_repair, candidate_evaluation = chosen_candidate
        current_payload = PromotePhase2RepairToSeedPayload(
            payload=current_payload,
            repair=candidate_repair,
        )
        current_repair = _canonical_repair(current_payload)
        current_evaluation = EvaluatePhase2Proposal(
            payload=current_payload,
            repair=current_repair,
        )
        if not bool(current_evaluation.get("hard_valid")):
            break

    return current_payload, current_repair, current_evaluation


def build_phase2_preview_candidate(
    *,
    payload: dict[str, Any],
    rounds: int = 1,
    move_limit: int = 24,
    note: str = "Fast preview from solver seed.",
) -> dict[str, Any]:
    preview_payload = deepcopy(payload)
    seed_verify = (preview_payload.get("repair_debug") or {}).get("seed_verify") or {}
    if isinstance(seed_verify, dict) and _needs_primary_pair_axis_repair(
        preview_payload, seed_verify
    ):
        _set_search_phase(preview_payload, "object_refine")
    baseline_repair = make_no_improvement_repair(preview_payload, note=note)
    snapped_payload, snapped_repair, snapped_eval = _postsolve_orientation_snap(
        payload=preview_payload,
        repair=baseline_repair,
        rounds=rounds,
        move_limit=move_limit,
    )
    compilable_payload = deepcopy(snapped_payload)
    compilable_payload["phase2_placer"] = {"phase2_repair": snapped_repair}
    absolute_layout = compile_phase2_to_final_output(compilable_payload)
    return {
        "payload": snapped_payload,
        "proposal": snapped_repair,
        "evaluation": snapped_eval,
        "absolute_layout": absolute_layout,
    }


@dataclass(frozen=True)
class Phase2Controller:
    placer: MacroClusterPlacer = field(default_factory=MacroClusterPlacer)
    max_iterations: int = 20

    def generate(
        self,
        *,
        room_interpreter_json: dict[str, Any],
        cluster_merged_json: dict[str, Any],
        cluster_outlines_json: dict[str, Any],
        relation_plan_json: dict[str, Any],
        solver_output_json: dict[str, Any],
        cluster_constraints_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_solver_output = _normalize_solver_output_for_phase2(
            solver_output_json
        )
        current_payload = build_phase2_payload(
            room_01=room_interpreter_json,
            clusters_04=cluster_merged_json,
            outlines_05=cluster_outlines_json,
            relation_05b=relation_plan_json,
            placer_06=normalized_solver_output,
            cluster_constraints=cluster_constraints_json,
        )

        result = MicroRefiner().refine(current_payload).as_controller_result()
        absolute_layout = result.get("absolute_layout") or {}
        if str(absolute_layout.get("status") or "").upper() == "OK":
            return result

        # Fallback path: if the solver produced a hard-valid seed that is still
        # macro-blocked or locally invalid after materialization, run the
        # deterministic phase-2 placer and explicitly try an object-refine pass
        # when the seed diagnostics say the primary viewing pair is wrong.
        fallback_notes = [
            "Phase-2 fallback activated: deterministic repair from solver seed.",
        ]
        try:
            seed_verify = (
                (normalized_solver_output.get("placer_seed") or {}).get("seed_verify")
                if isinstance(normalized_solver_output, dict)
                else None
            )
            candidate_payloads: list[tuple[str, dict[str, Any], list[str]]] = [
                ("macro_layout", deepcopy(current_payload), list(fallback_notes))
            ]
            if isinstance(seed_verify, dict) and _needs_primary_pair_axis_repair(
                current_payload,
                seed_verify,
            ):
                object_payload = deepcopy(current_payload)
                _set_search_phase(object_payload, "object_refine")
                candidate_payloads.insert(
                    0,
                    (
                        "object_refine",
                        object_payload,
                        [
                            *fallback_notes,
                            "Primary viewing-pair/object-orientation pressure detected; object-refine fallback promoted before generic macro repair.",
                        ],
                    ),
                )

            best_candidate: (
                tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str], str]
                | None
            ) = None
            best_rank: tuple[int, int, int, int, int] | None = None

            for mode, candidate_payload, candidate_notes in candidate_payloads:
                fallback_raw = self.placer.generate_from_payload(
                    payload=candidate_payload,
                    max_attempts=3,
                )
                fallback_proposal = _canonical_repair(
                    candidate_payload,
                    fallback_raw if isinstance(fallback_raw, dict) else {},
                    notes=candidate_notes,
                )
                fallback_eval = EvaluatePhase2Proposal(
                    payload=candidate_payload,
                    repair=fallback_proposal,
                )
                if not bool(fallback_eval.get("hard_valid")):
                    continue
                snapped_payload, snapped_repair, snapped_eval = (
                    _postsolve_orientation_snap(
                        payload=candidate_payload,
                        repair=fallback_proposal,
                        rounds=2,
                        move_limit=64 if mode == "object_refine" else 48,
                    )
                )
                final_payload = deepcopy(snapped_payload)
                final_payload["phase2_placer"] = {"phase2_repair": snapped_repair}
                final_absolute_layout = compile_phase2_to_final_output(final_payload)
                rank = _search_candidate_rank(snapped_eval)
                if str(final_absolute_layout.get("status") or "").upper() == "OK":
                    rank = (rank[0] + 1, *rank[1:])
                if best_candidate is None or rank > (best_rank or (0, 0, 0, 0, 0)):
                    best_candidate = (
                        snapped_repair,
                        snapped_eval,
                        final_absolute_layout,
                        candidate_notes,
                        mode,
                    )
                    best_rank = rank

            if (
                best_candidate is not None
                and str(best_candidate[2].get("status") or "").upper() == "OK"
            ):
                (
                    snapped_repair,
                    snapped_eval,
                    final_absolute_layout,
                    candidate_notes,
                    mode,
                ) = best_candidate
                return _phase2_controller_result(
                    proposal=snapped_repair,
                    evaluation=snapped_eval,
                    absolute_layout=final_absolute_layout,
                    history=[
                        {
                            "stage": "micro_refiner",
                            "status": "invalid_seed",
                            "absolute_layout_status": str(
                                absolute_layout.get("status") or "UNKNOWN"
                            ),
                        },
                        {
                            "stage": "deterministic_phase2_fallback",
                            "mode": mode,
                            "status": "accepted",
                            "proposal_status": str(
                                snapped_repair.get("status") or "REPAIRED"
                            ),
                        },
                    ],
                    hard_fix={
                        "result": "APPLIED",
                        "attempts": [
                            {
                                "mode": mode,
                                "hard_valid": True,
                                "notes": candidate_notes,
                            }
                        ],
                    },
                    notes=candidate_notes,
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Phase-2 deterministic fallback failed: %s", exc)

        return result
