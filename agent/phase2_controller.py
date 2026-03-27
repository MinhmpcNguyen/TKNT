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
)
from agent.phase2_judge import Phase2Judge
from cluster_placer.tools_v2 import (
    EnumeratePhase2RepairMoves,
    EvaluatePhase2Proposal,
    PromotePhase2RepairToSeedPayload,
    ResolvePhase2HardViolations,
)

logger = logging.getLogger(__name__)


def _judge_rank(verdict: str) -> int:
    if verdict == "ACCEPT":
        return 2
    if verdict == "REVISE":
        return 1
    return 0


def _evaluation_rank(
    evaluation: dict[str, Any], judge_output: dict[str, Any]
) -> tuple[int, int, int, int, int]:
    hard_valid = 1 if bool(evaluation.get("hard_valid")) else 0
    verdict_rank = _judge_rank(str(judge_output.get("verdict") or "REJECT"))
    tool_score = int(
        ((evaluation.get("baseline_comparison") or {}).get("candidate_score")) or 0
    )
    judge_score = int(judge_output.get("reasonableness_score") or 0)
    issue_penalty = -len(judge_output.get("top_issues") or [])
    return (hard_valid, verdict_rank, tool_score, judge_score, issue_penalty)


def _baseline_judge_stub(evaluation: dict[str, Any]) -> dict[str, Any]:
    if bool(evaluation.get("hard_valid")):
        return {
            "reasonableness_score": 50,
            "verdict": "REVISE",
            "top_issues": [],
            "repair_advice": [],
            "priority_clusters": [],
        }
    return {
        "reasonableness_score": 0,
        "verdict": "REJECT",
        "top_issues": ["Baseline seed is hard-invalid."],
        "repair_advice": ["Repair hard validity before soft reasoning."],
        "priority_clusters": [],
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


def _primary_cluster_from_direction(direction_summary: dict[str, Any]) -> str:
    clusters = direction_summary.get("primary_clusters") or []
    if clusters and isinstance(clusters[0], str):
        return clusters[0]
    return ""


def _proposal_has_object_repairs(proposal: dict[str, Any]) -> bool:
    return bool((proposal.get("object_repairs") or []))


def _normalize_search_phase(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "object_refine":
        return "object_refine"
    return "macro_layout"


def _current_search_phase(payload: dict[str, Any]) -> str:
    phase_control = payload.get("phase_control") or {}
    return _normalize_search_phase(phase_control.get("repair_phase"))


def _set_search_phase(payload: dict[str, Any], search_phase: str) -> None:
    phase_control = deepcopy(payload.get("phase_control") or {})
    phase_control["repair_phase"] = _normalize_search_phase(search_phase)
    payload["phase_control"] = phase_control


def _move_matches_search_phase(move: dict[str, Any], search_phase: str) -> bool:
    kind = str(move.get("kind") or "")
    normalized_phase = _normalize_search_phase(search_phase)
    if normalized_phase == "object_refine":
        return kind == "object_pose"
    return kind in {"cluster_variant", "cluster_pose"}


def _should_replace_with_deterministic_candidate(
    shortlist_evaluation: dict[str, Any],
    placer_evaluation: dict[str, Any],
) -> bool:
    if not bool(shortlist_evaluation.get("hard_valid")):
        return False
    if _delta_score(shortlist_evaluation) < 0:
        return False
    return _search_candidate_rank(shortlist_evaluation) > _search_candidate_rank(
        placer_evaluation
    )


def _invalid_judge_stub(evaluation: dict[str, Any]) -> dict[str, Any]:
    errors = evaluation.get("errors") or []
    top_issues: list[str] = []
    for error in errors[:3]:
        if not isinstance(error, dict):
            continue
        code = str(error.get("code") or "UNKNOWN")
        if code == "OBJECT_OVERLAP":
            top_issues.append("Candidate is hard-invalid because objects overlap.")
        elif code == "OBJECT_OUT_OF_BOUNDS":
            top_issues.append(
                "Candidate is hard-invalid because at least one object leaves the room."
            )
        elif code == "OBJECT_HITS_OBSTACLE":
            top_issues.append(
                "Candidate is hard-invalid because it intersects a hard obstacle."
            )
        else:
            top_issues.append(f"Candidate is hard-invalid ({code}).")
    if not top_issues:
        top_issues = ["Candidate is hard-invalid."]
    return {
        "reasonableness_score": 10,
        "verdict": "REJECT",
        "top_issues": top_issues,
        "repair_advice": ["Repair hard validity before spending more search."],
        "priority_clusters": [],
    }


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


def _summarize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation.get("metrics") or {}
    diagnosis = evaluation.get("diagnosis") or {}
    layout = evaluation.get("materialized_layout") or {}

    cluster_summary = []
    zone_by_cluster = {
        str(row.get("cluster_id") or ""): str(row.get("zone") or "")
        for row in metrics.get("zone_usage_summary") or []
        if isinstance(row, dict)
    }
    entry_by_cluster = {
        str(row.get("cluster_id") or ""): row.get("distance_to_nearest_door_mm")
        for row in metrics.get("cluster_entry_proximity") or []
        if isinstance(row, dict)
    }
    window_by_cluster = {
        str(row.get("cluster_id") or ""): row.get("distance_to_nearest_window_mm")
        for row in metrics.get("cluster_window_alignment") or []
        if isinstance(row, dict)
    }
    priority_cluster_scores = {
        str(row.get("cluster_id") or ""): row.get("score")
        for row in diagnosis.get("prioritized_clusters") or []
        if isinstance(row, dict)
    }

    for cluster in (layout.get("clusters") or [])[:8]:
        cid = str(cluster.get("cluster_id") or "")
        cluster_summary.append(
            {
                "cluster_id": cid,
                "variant_id": cluster.get("variant_id"),
                "zone": zone_by_cluster.get(cid),
                "distance_to_nearest_door_mm": entry_by_cluster.get(cid),
                "distance_to_nearest_window_mm": window_by_cluster.get(cid),
                "soft_score": priority_cluster_scores.get(cid),
            }
        )

    object_summary = []
    for row in (metrics.get("object_front_clearance") or [])[:8]:
        if not isinstance(row, dict):
            continue
        object_summary.append(
            {
                "cluster_id": row.get("cluster_id"),
                "object_id": row.get("object_id"),
                "current_front_clear_mm": row.get("current_front_clear_mm"),
                "best_open_clear_mm": row.get("best_open_clear_mm"),
                "shortage_mm": row.get("shortage_mm"),
            }
        )

    return {
        "hard_valid": bool(evaluation.get("hard_valid")),
        "score_summary": deepcopy(metrics.get("score_summary") or {}),
        "goal_alignment_summary": deepcopy(metrics.get("goal_alignment_summary") or {}),
        "cluster_summary": cluster_summary,
        "object_summary": object_summary,
        "key_findings": deepcopy(diagnosis.get("key_findings") or []),
    }


def _summarize_proposal_direction(
    payload: dict[str, Any], proposal: dict[str, Any]
) -> dict[str, Any]:
    seed_layout = payload.get("seed_layout") or {}
    seed_tmap = _seed_transform_map(seed_layout)
    seed_vmap = _seed_variant_map(seed_layout)
    changed_clusters: list[str] = []
    move_families: list[str] = []

    for row in proposal.get("cluster_transforms") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str):
            continue
        seed_row = seed_tmap.get(cid, {})
        if int(row.get("rot") or 0) != int(seed_row.get("rot") or 0):
            move_families.append("cluster_rotation")
            changed_clusters.append(cid)
        if int(row.get("x") or 0) != int(seed_row.get("x") or 0) or int(
            row.get("y") or 0
        ) != int(seed_row.get("y") or 0):
            move_families.append("cluster_translation")
            changed_clusters.append(cid)

    for row in proposal.get("selected_variants") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str):
            continue
        seed_row = seed_vmap.get(cid, {})
        if str(row.get("variant_id") or "") != str(seed_row.get("variant_id") or ""):
            move_families.append("variant_switch")
            changed_clusters.append(cid)

    for row in proposal.get("object_repairs") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if isinstance(cluster_id, str):
            changed_clusters.append(cluster_id)
        move_families.append(str(row.get("op") or "object_repair"))

    if not move_families:
        move_families.append("seed_retain")

    dedup_clusters = list(dict.fromkeys(changed_clusters))[:4]
    dedup_families = list(dict.fromkeys(move_families))
    summary = ", ".join(dedup_families)
    if dedup_clusters:
        summary = f"{summary} on {', '.join(dedup_clusters)}"

    return {
        "primary_clusters": dedup_clusters,
        "move_families": dedup_families,
        "summary": summary,
    }


def _build_history_context(history: list[dict[str, Any]]) -> dict[str, Any]:
    recent_attempts: list[dict[str, Any]] = []
    cluster_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}

    for entry in history[-8:]:
        direction = entry.get("direction_summary") or {}
        tool_eval = entry.get("tool_evaluation") or {}
        judge_eval = entry.get("judge_evaluation") or {}
        attempt = {
            "iteration": entry.get("iteration"),
            "search_phase": _normalize_search_phase(entry.get("search_phase")),
            "summary": direction.get("summary"),
            "primary_clusters": deepcopy(direction.get("primary_clusters") or []),
            "move_families": deepcopy(direction.get("move_families") or []),
            "hard_valid": bool(tool_eval.get("hard_valid")),
            "judge_verdict": judge_eval.get("verdict"),
            "judge_score": judge_eval.get("reasonableness_score"),
            "delta_score": (
                (tool_eval.get("baseline_comparison") or {}).get("delta_score")
            ),
        }
        recent_attempts.append(attempt)
        for cluster_id in attempt["primary_clusters"]:
            cluster_counts[str(cluster_id)] = cluster_counts.get(str(cluster_id), 0) + 1
        for family in attempt["move_families"]:
            family_counts[str(family)] = family_counts.get(str(family), 0) + 1

    stuck_clusters = [
        cluster_id
        for cluster_id, count in sorted(
            cluster_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 3
    ]
    avoid_move_families = [
        family
        for family, count in sorted(
            family_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 3
    ]

    stuck_patterns: list[str] = []
    if stuck_clusters:
        stuck_patterns.append(
            "Repeated focus on the same clusters without acceptance: "
            + ", ".join(stuck_clusters[:4])
        )
    if avoid_move_families:
        stuck_patterns.append(
            "Repeated move families without acceptance: "
            + ", ".join(avoid_move_families[:4])
        )

    return {
        "recent_attempts": recent_attempts,
        "stuck_clusters": stuck_clusters,
        "avoid_move_families": avoid_move_families,
        "stuck_patterns": stuck_patterns,
    }


def _select_shortlist_candidate(
    candidates: list[dict[str, Any]],
    history: list[dict[str, Any]],
    search_phase: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    best = candidates[0]
    if _normalize_search_phase(search_phase) != "macro_layout" or len(candidates) == 1:
        return best

    best_delta = _delta_score(best.get("evaluation") or {})
    recent_macro_attempts = [
        entry
        for entry in history[-3:]
        if _normalize_search_phase(entry.get("search_phase")) == "macro_layout"
    ]
    recent_clusters = {
        _primary_cluster_from_direction(entry.get("direction_summary") or {})
        for entry in recent_macro_attempts
    }
    recent_clusters.discard("")
    recent_families = {
        str(family)
        for entry in recent_macro_attempts
        for family in (
            (entry.get("direction_summary") or {}).get("move_families") or []
        )
        if isinstance(family, str)
    }

    if best_delta >= 160 or not recent_macro_attempts:
        return best

    for candidate in candidates[1:]:
        evaluation = candidate.get("evaluation") or {}
        if not bool(evaluation.get("hard_valid")):
            continue
        if _delta_score(evaluation) < 0:
            continue
        direction = candidate.get("direction_summary") or {}
        primary_cluster = _primary_cluster_from_direction(direction)
        move_families = {
            str(family)
            for family in (direction.get("move_families") or [])
            if isinstance(family, str)
        }
        if primary_cluster and primary_cluster not in recent_clusters:
            return candidate
        if move_families and move_families.isdisjoint(recent_families):
            return candidate

    return best


def _should_stop_object_refine_search(
    *,
    history: list[dict[str, Any]],
    deterministic_shortlist: list[dict[str, Any]],
    search_phase: str,
) -> bool:
    if _normalize_search_phase(search_phase) != "object_refine":
        return False

    recent_object_attempts = [
        entry
        for entry in history[-3:]
        if _normalize_search_phase(entry.get("search_phase")) == "object_refine"
    ]
    if len(recent_object_attempts) < 3:
        return False

    if any(
        bool((item.get("evaluation") or {}).get("hard_valid"))
        and _delta_score(item.get("evaluation") or {}) > 0
        for item in deterministic_shortlist
    ):
        return False

    for entry in recent_object_attempts:
        tool_eval = entry.get("tool_evaluation") or {}
        judge_eval = entry.get("judge_evaluation") or {}
        if bool(tool_eval.get("hard_valid")) and _delta_score(tool_eval) > 0:
            return False
        if str(judge_eval.get("verdict") or "").upper() == "ACCEPT":
            return False

    return True


def _should_enter_object_refine_phase(
    *,
    iteration: int,
    history: list[dict[str, Any]],
    macro_shortlist: list[dict[str, Any]],
) -> bool:
    positive_macro = any(
        bool(item.get("evaluation", {}).get("hard_valid"))
        and _delta_score(item.get("evaluation") or {}) > 0
        for item in macro_shortlist
    )
    if positive_macro:
        return False

    recent_macro_attempts = [
        entry
        for entry in history[-3:]
        if _normalize_search_phase(entry.get("search_phase")) == "macro_layout"
    ]
    stalled_macro_attempts = [
        entry
        for entry in recent_macro_attempts
        if int(
            (
                (entry.get("tool_evaluation") or {})
                .get("baseline_comparison", {})
                .get("delta_score")
            )
            or 0
        )
        <= 0
    ]

    return iteration >= 4 or len(stalled_macro_attempts) >= 2


def _build_judge_payload(
    *,
    payload: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    candidate_evaluation: dict[str, Any],
    iteration: int,
    max_iterations: int,
    history: list[dict[str, Any]],
    direction_summary: dict[str, Any],
) -> dict[str, Any]:
    room_model = (payload.get("room_context") or {}).get("room_model_used") or {}
    room_notes = deepcopy(room_model.get("notes") or [])
    candidate_metrics = candidate_evaluation.get("metrics") or {}
    return {
        "room_context": deepcopy(payload.get("room_context") or {}),
        "design_goals": deepcopy(payload.get("goals") or {}),
        "baseline_summary": _summarize_evaluation(baseline_evaluation),
        "candidate_summary": _summarize_evaluation(candidate_evaluation),
        "comparison_summary": deepcopy(
            candidate_evaluation.get("baseline_comparison") or {}
        ),
        "hard_check_summary": {
            "hard_valid": bool(candidate_evaluation.get("hard_valid")),
            "errors": deepcopy(candidate_evaluation.get("errors") or []),
            "violations_by_cluster": deepcopy(
                candidate_evaluation.get("violations_by_cluster") or {}
            ),
        },
        "metrics": deepcopy(candidate_metrics),
        "diagnosis": deepcopy(candidate_evaluation.get("diagnosis") or {}),
        "generic_signals": {
            "room_notes": room_notes,
            "goal_alignment_summary": deepcopy(
                candidate_metrics.get("goal_alignment_summary") or {}
            ),
            "path_obstruction_summary": deepcopy(
                candidate_metrics.get("path_obstruction_summary") or {}
            ),
            "zone_usage_summary": deepcopy(
                candidate_metrics.get("zone_usage_summary") or []
            ),
        },
        "controller_context": {
            "iteration": iteration,
            "max_iterations": max_iterations,
            "current_direction": deepcopy(direction_summary),
            **_build_history_context(history),
        },
    }


def _build_controller_feedback(
    *,
    iteration: int,
    evaluation: dict[str, Any],
    judge_output: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnosis = evaluation.get("diagnosis") or {}
    errors = evaluation.get("errors") or []
    history_context = _build_history_context(history)
    priority_clusters = []
    if isinstance(judge_output, dict) and judge_output.get("priority_clusters"):
        priority_clusters = deepcopy(judge_output.get("priority_clusters") or [])
    else:
        priority_clusters = [
            row.get("cluster_id")
            for row in diagnosis.get("prioritized_clusters") or []
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
        ][:4]

    candidate_status = "hard_invalid"
    repair_direction = "Fix hard validity before making new soft edits."
    judge_top_issues: list[str] = []
    if bool(evaluation.get("hard_valid")):
        candidate_status = "soft_revision_needed"
        repair_direction = "Keep the useful direction, but switch tactic if the same move keeps repeating."
    if isinstance(judge_output, dict):
        verdict = str(judge_output.get("verdict") or "").upper()
        if verdict == "ACCEPT":
            candidate_status = "accepted"
            repair_direction = "This candidate is acceptable."
        elif verdict == "REJECT":
            candidate_status = "rejected"
            repair_direction = "Try a different cluster or move family."
        judge_top_issues = deepcopy(judge_output.get("top_issues") or [])
        if judge_output.get("repair_advice"):
            repair_direction = str(
                (judge_output.get("repair_advice") or [""])[0] or repair_direction
            )

    return {
        "iteration": iteration,
        "candidate_status": candidate_status,
        "hard_invalid_reasons": deepcopy(errors),
        "judge_top_issues": judge_top_issues,
        "priority_clusters": priority_clusters,
        "priority_objects": deepcopy((diagnosis.get("prioritized_objects") or [])[:6]),
        "repair_direction": repair_direction,
        "stuck_clusters": deepcopy(history_context.get("stuck_clusters") or []),
        "avoid_move_families": deepcopy(
            history_context.get("avoid_move_families") or []
        ),
        "recent_attempt_summaries": deepcopy(
            history_context.get("recent_attempts") or []
        ),
    }


def _fallback_moves(
    payload: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    history: list[dict[str, Any]],
    search_phase: str,
) -> list[dict[str, Any]]:
    history_context = _build_history_context(history)
    stuck_clusters = set(history_context.get("stuck_clusters") or [])
    avoid_move_families = set(history_context.get("avoid_move_families") or [])

    moves = deepcopy(
        (baseline_evaluation.get("diagnosis") or {}).get("enumerated_moves") or []
    )
    if not moves:
        moves = deepcopy(
            (EnumeratePhase2RepairMoves(payload=payload, limit=32).get("moves") or [])
        )
    moves = [
        move
        for move in moves
        if isinstance(move, dict) and _move_matches_search_phase(move, search_phase)
    ]

    def _move_family(move: dict[str, Any]) -> str:
        kind = str(move.get("kind") or "")
        if kind == "cluster_variant":
            return "variant_switch"
        if kind == "cluster_pose":
            proposal = move.get("proposal") or {}
            transform = ((proposal.get("cluster_transforms") or [None])[0]) or {}
            seed_row = _seed_transform_map(payload.get("seed_layout") or {}).get(
                str(transform.get("cluster_id") or ""),
                {},
            )
            if int(transform.get("rot") or 0) != int(seed_row.get("rot") or 0):
                return "cluster_rotation"
            return "cluster_translation"
        if kind == "object_pose":
            repair_row = ((move.get("proposal") or {}).get("object_repairs") or [None])[
                0
            ] or {}
            return str(repair_row.get("op") or "object_repair")
        return kind or "unknown"

    moves.sort(
        key=lambda move: (
            1 if str(move.get("cluster_id") or "") in stuck_clusters else 0,
            1 if _move_family(move) in avoid_move_families else 0,
        )
    )
    cluster_order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for move in moves:
        cluster_id = str(move.get("cluster_id") or "")
        grouped.setdefault(cluster_id, []).append(move)
        if cluster_id not in cluster_order:
            cluster_order.append(cluster_id)

    interleaved: list[dict[str, Any]] = []
    appended = True
    while appended:
        appended = False
        for cluster_id in cluster_order:
            bucket = grouped.get(cluster_id) or []
            if not bucket:
                continue
            interleaved.append(bucket.pop(0))
            appended = True
    return interleaved


def _forced_macro_moves(
    payload: dict[str, Any], history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    history_context = _build_history_context(history)
    stuck_clusters = set(history_context.get("stuck_clusters") or [])
    diagnosis = payload.get("repair_debug") or {}
    targets = (
        (diagnosis.get("repair_targets") or {}).get("prioritized_clusters")
        or (diagnosis.get("seed_verify") or {})
        .get("repair_guidance", {})
        .get("prioritized_clusters")
        or []
    )
    cluster_ids = [
        row.get("cluster_id")
        for row in targets
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
    ]
    if not cluster_ids:
        cluster_ids = sorted(
            _seed_transform_map(payload.get("seed_layout") or {}).keys()
        )
    cluster_ids = sorted(
        cluster_ids, key=lambda cluster_id: (cluster_id in stuck_clusters, cluster_id)
    )

    grid_mm = int(((payload.get("room_context") or {}).get("grid_mm") or 50))
    tmap = _seed_transform_map(payload.get("seed_layout") or {})
    forced: list[dict[str, Any]] = []
    for cluster_id in cluster_ids[:4]:
        seed_row = tmap.get(cluster_id)
        if not isinstance(seed_row, dict):
            continue
        for rot in (0, 90, 180, 270):
            if rot == int(seed_row.get("rot") or 0):
                continue
            forced.append(
                {
                    "cluster_transforms": [{**deepcopy(seed_row), "rot": rot}],
                    "selected_variants": [],
                    "object_repairs": [],
                    "notes": [f"Controller forced macro rotation for {cluster_id}."],
                }
            )
        for dx, dy in (
            (grid_mm, 0),
            (-grid_mm, 0),
            (0, grid_mm),
            (0, -grid_mm),
        ):
            forced.append(
                {
                    "cluster_transforms": [
                        {
                            **deepcopy(seed_row),
                            "x": int(seed_row.get("x") or 0) + dx,
                            "y": int(seed_row.get("y") or 0) + dy,
                        }
                    ],
                    "selected_variants": [],
                    "object_repairs": [],
                    "notes": [f"Controller forced macro translation for {cluster_id}."],
                }
            )
    return forced


def _forced_object_moves(
    payload: dict[str, Any], history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    history_context = _build_history_context(history)
    stuck_clusters = set(history_context.get("stuck_clusters") or [])

    prioritized_objects = []
    diagnosis = payload.get("repair_debug") or {}
    targets = (
        (diagnosis.get("repair_targets") or {}).get("prioritized_objects")
        or (diagnosis.get("seed_verify") or {})
        .get("repair_guidance", {})
        .get("prioritized_objects")
        or []
    )
    for row in targets:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        oid = row.get("object_id")
        if isinstance(cid, str) and isinstance(oid, str):
            prioritized_objects.append((cid, oid))

    if not prioritized_objects:
        for row in payload.get("objects_world") or []:
            if not isinstance(row, dict):
                continue
            cid = row.get("cluster_id")
            oid = row.get("object_id")
            if isinstance(cid, str) and isinstance(oid, str):
                prioritized_objects.append((cid, oid))

    prioritized_objects = list(
        dict.fromkeys(
            sorted(
                prioritized_objects,
                key=lambda item: (item[0] in stuck_clusters, item[0], item[1]),
            )
        )
    )

    forced: list[dict[str, Any]] = []
    for cluster_id, object_id in prioritized_objects[:6]:
        for rot in (90, 180, 270):
            forced.append(
                {
                    "cluster_transforms": [],
                    "selected_variants": [],
                    "object_repairs": [
                        {
                            "cluster_id": cluster_id,
                            "object_id": object_id,
                            "op": "rotate_object",
                            "params": {"rot": rot},
                        }
                    ],
                    "notes": [
                        f"Controller forced object rotation for {cluster_id}.{object_id}."
                    ],
                }
            )
        for dx, dy in ((50, 0), (-50, 0), (0, 50), (0, -50)):
            forced.append(
                {
                    "cluster_transforms": [],
                    "selected_variants": [],
                    "object_repairs": [
                        {
                            "cluster_id": cluster_id,
                            "object_id": object_id,
                            "op": "nudge_object",
                            "params": {"dx": dx, "dy": dy},
                        }
                    ],
                    "notes": [
                        f"Controller forced object nudge for {cluster_id}.{object_id}."
                    ],
                }
            )
    return forced


def _choose_fallback_proposal(
    *,
    payload: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    attempted_signatures: set[str],
    history: list[dict[str, Any]],
    search_phase: str,
) -> dict[str, Any]:
    deterministic_shortlist = _shortlist_deterministic_candidates(
        payload=payload,
        baseline_evaluation=baseline_evaluation,
        attempted_signatures=attempted_signatures,
        history=history,
        search_phase=search_phase,
        max_candidates=16,
        shortlist_size=1,
    )
    if deterministic_shortlist:
        return deepcopy(deterministic_shortlist[0]["proposal"])

    for move in _fallback_moves(payload, baseline_evaluation, history, search_phase):
        reason = str(move.get("reason") or "Controller fallback move.")
        proposal = _canonical_repair(
            payload,
            move.get("proposal") or {},
            notes=[f"Controller fallback: {reason}"],
        )
        if _proposal_signature(proposal) not in attempted_signatures:
            return proposal

    forced_moves = (
        _forced_object_moves(payload, history)
        if _normalize_search_phase(search_phase) == "object_refine"
        else _forced_macro_moves(payload, history)
    )
    for partial in forced_moves:
        proposal = _canonical_repair(payload, partial)
        if _proposal_signature(proposal) not in attempted_signatures:
            return proposal

    return _canonical_repair(
        payload,
        notes=[
            "Controller retained the current baseline after exhausting deterministic move search."
        ],
    )


def _shortlist_deterministic_candidates(
    *,
    payload: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    attempted_signatures: set[str],
    history: list[dict[str, Any]],
    search_phase: str,
    max_candidates: int = 12,
    shortlist_size: int = 3,
) -> list[dict[str, Any]]:
    shortlisted: list[dict[str, Any]] = []
    local_signatures: set[str] = set()
    deterministic_moves = _fallback_moves(
        payload, baseline_evaluation, history, search_phase
    )
    fallback_budget = max(4, int(max(1, int(max_candidates)) * 0.6))

    def _append_candidate(proposal: dict[str, Any], source: str) -> None:
        signature = _proposal_signature(proposal)
        if signature in attempted_signatures or signature in local_signatures:
            return
        evaluation = EvaluatePhase2Proposal(payload=payload, repair=proposal)
        direction_summary = _summarize_proposal_direction(payload, proposal)
        shortlisted.append(
            {
                "proposal": proposal,
                "evaluation": evaluation,
                "direction_summary": direction_summary,
                "source": source,
                "signature": signature,
            }
        )
        local_signatures.add(signature)

    for move in deterministic_moves:
        reason = str(move.get("reason") or "Controller shortlist move.")
        proposal = _canonical_repair(
            payload,
            move.get("proposal") or {},
            notes=[f"Deterministic shortlist: {reason}"],
        )
        _append_candidate(proposal, reason)
        if len(shortlisted) >= fallback_budget:
            break

    if len(shortlisted) < max(1, int(max_candidates)):
        forced_moves = (
            _forced_object_moves(payload, history)
            if _normalize_search_phase(search_phase) == "object_refine"
            else _forced_macro_moves(payload, history)
        )
        for partial in forced_moves:
            proposal = _canonical_repair(payload, partial)
            _append_candidate(proposal, "Controller forced move.")
            if len(shortlisted) >= max(1, int(max_candidates)):
                break

    if len(shortlisted) < max(1, int(max_candidates)):
        for move in deterministic_moves[fallback_budget:]:
            reason = str(move.get("reason") or "Controller shortlist move.")
            proposal = _canonical_repair(
                payload,
                move.get("proposal") or {},
                notes=[f"Deterministic shortlist: {reason}"],
            )
            _append_candidate(proposal, reason)
            if len(shortlisted) >= max(1, int(max_candidates)):
                break

    shortlisted.sort(
        key=lambda item: _search_candidate_rank(item["evaluation"]),
        reverse=True,
    )
    return shortlisted[: max(1, int(shortlist_size))]


def _should_use_deterministic_candidate(
    evaluation: dict[str, Any], history: list[dict[str, Any]]
) -> bool:
    if not bool(evaluation.get("hard_valid")):
        return False
    delta = _delta_score(evaluation)
    if delta > 0:
        return True
    history_context = _build_history_context(history)
    return bool(
        delta >= 0
        and (
            history_context.get("stuck_clusters")
            or history_context.get("avoid_move_families")
        )
    )


def _should_promote_candidate(
    evaluation: dict[str, Any], judge_output: dict[str, Any]
) -> bool:
    if not bool(evaluation.get("hard_valid")):
        return False
    verdict = str(judge_output.get("verdict") or "").upper()
    if verdict == "REJECT":
        return False
    delta = _delta_score(evaluation)
    if delta > 0:
        return True
    if delta < 0:
        return False
    judge_score = int(judge_output.get("reasonableness_score") or 0)
    return verdict == "ACCEPT" and judge_score >= 85


@dataclass(frozen=True)
class Phase2Controller:
    placer: MacroClusterPlacer = field(default_factory=MacroClusterPlacer)
    judge: Phase2Judge = field(default_factory=Phase2Judge)
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
        current_payload = build_phase2_payload(
            room_01=room_interpreter_json,
            clusters_04=cluster_merged_json,
            outlines_05=cluster_outlines_json,
            relation_05b=relation_plan_json,
            placer_06=solver_output_json,
            cluster_constraints=cluster_constraints_json,
        )
        current_repair = _canonical_repair(current_payload)
        current_baseline = EvaluatePhase2Proposal(
            payload=current_payload,
            repair=current_repair,
        )
        logger.info(
            "Phase2Controller start: baseline_hard_valid=%s baseline_score=%s",
            current_baseline.get("hard_valid"),
            (
                (current_baseline.get("baseline_comparison") or {}).get(
                    "candidate_score"
                )
            ),
        )

        best_payload = deepcopy(current_payload)
        best_proposal = deepcopy(current_repair)
        best_tool_eval = current_baseline
        best_judge_eval = _baseline_judge_stub(current_baseline)

        best_hard_valid_payload = deepcopy(current_payload)
        best_hard_valid_proposal = deepcopy(current_repair)
        best_hard_valid_eval = current_baseline
        best_hard_valid_judge = _baseline_judge_stub(current_baseline)

        history: list[dict[str, Any]] = []
        controller_feedback: dict[str, Any] | None = None
        attempted_signatures = {_proposal_signature(current_repair)}

        for iteration in range(1, max(1, int(self.max_iterations)) + 1):
            if controller_feedback is not None:
                current_payload["controller_feedback"] = deepcopy(controller_feedback)
            else:
                current_payload.pop("controller_feedback", None)

            search_phase = _current_search_phase(current_payload)
            _set_search_phase(current_payload, search_phase)

            deterministic_shortlist = _shortlist_deterministic_candidates(
                payload=current_payload,
                baseline_evaluation=current_baseline,
                attempted_signatures=attempted_signatures,
                history=history,
                search_phase=search_phase,
            )

            if _normalize_search_phase(search_phase) == "macro_layout" and (
                _should_enter_object_refine_phase(
                    iteration=iteration,
                    history=history,
                    macro_shortlist=deterministic_shortlist,
                )
            ):
                search_phase = "object_refine"
                _set_search_phase(current_payload, search_phase)
                deterministic_shortlist = _shortlist_deterministic_candidates(
                    payload=current_payload,
                    baseline_evaluation=current_baseline,
                    attempted_signatures=attempted_signatures,
                    history=history,
                    search_phase=search_phase,
                )

            shortlist_candidate = _select_shortlist_candidate(
                deterministic_shortlist, history, search_phase
            )

            if _should_stop_object_refine_search(
                history=history,
                deterministic_shortlist=deterministic_shortlist,
                search_phase=search_phase,
            ):
                logger.info(
                    "Phase2Controller iteration %s: stopping object-refine search because recent attempts stalled and no positive deterministic object move remains",
                    iteration,
                )
                break

            if shortlist_candidate and _should_use_deterministic_candidate(
                shortlist_candidate["evaluation"], history
            ):
                logger.info(
                    "Phase2Controller iteration %s/%s: using deterministic shortlist candidate phase=%s source=%s delta_score=%s",
                    iteration,
                    self.max_iterations,
                    search_phase,
                    shortlist_candidate.get("source"),
                    _delta_score(shortlist_candidate["evaluation"]),
                )
                proposal = deepcopy(shortlist_candidate["proposal"])
                tool_eval = deepcopy(shortlist_candidate["evaluation"])
            else:
                logger.info(
                    "Phase2Controller iteration %s/%s: calling placer phase=%s",
                    iteration,
                    self.max_iterations,
                    search_phase,
                )
                raw_proposal = self.placer.generate_from_payload(
                    payload=current_payload
                )
                logger.info(
                    "Phase2Controller iteration %s: placer_status=%s notes=%s",
                    iteration,
                    raw_proposal.get("status"),
                    raw_proposal.get("notes"),
                )

                if str(raw_proposal.get("status") or "").upper() == "REPAIRED":
                    proposal = _canonical_repair(current_payload, raw_proposal)
                else:
                    logger.info(
                        "Phase2Controller iteration %s: synthesizing fallback because placer returned %s",
                        iteration,
                        raw_proposal.get("status"),
                    )
                    proposal = _choose_fallback_proposal(
                        payload=current_payload,
                        baseline_evaluation=current_baseline,
                        attempted_signatures=attempted_signatures,
                        history=history,
                        search_phase=search_phase,
                    )

                if _proposal_signature(proposal) in attempted_signatures:
                    logger.info(
                        "Phase2Controller iteration %s: replacing repeated proposal with deterministic fallback",
                        iteration,
                    )
                    proposal = _choose_fallback_proposal(
                        payload=current_payload,
                        baseline_evaluation=current_baseline,
                        attempted_signatures=attempted_signatures,
                        history=history,
                        search_phase=search_phase,
                    )

                if _normalize_search_phase(
                    search_phase
                ) == "object_refine" and not _proposal_has_object_repairs(proposal):
                    logger.info(
                        "Phase2Controller iteration %s: object-refine phase requires object_repairs; replacing proposal with object fallback",
                        iteration,
                    )
                    proposal = _choose_fallback_proposal(
                        payload=current_payload,
                        baseline_evaluation=current_baseline,
                        attempted_signatures=attempted_signatures,
                        history=history,
                        search_phase=search_phase,
                    )

                tool_eval = EvaluatePhase2Proposal(
                    payload=current_payload, repair=proposal
                )

                if shortlist_candidate and _should_replace_with_deterministic_candidate(
                    shortlist_candidate["evaluation"], tool_eval
                ):
                    logger.info(
                        "Phase2Controller iteration %s: replacing placer candidate with deterministic shortlist phase=%s source=%s",
                        iteration,
                        search_phase,
                        shortlist_candidate.get("source"),
                    )
                    proposal = deepcopy(shortlist_candidate["proposal"])
                    tool_eval = deepcopy(shortlist_candidate["evaluation"])

            attempted_signatures.add(_proposal_signature(proposal))
            logger.info(
                "Phase2Controller iteration %s: tool hard_valid=%s delta_score=%s",
                iteration,
                tool_eval.get("hard_valid"),
                _delta_score(tool_eval),
            )

            direction_summary = _summarize_proposal_direction(current_payload, proposal)
            if bool(tool_eval.get("hard_valid")):
                judge_payload = _build_judge_payload(
                    payload=current_payload,
                    baseline_evaluation=current_baseline,
                    candidate_evaluation=tool_eval,
                    iteration=iteration,
                    max_iterations=self.max_iterations,
                    history=history,
                    direction_summary=direction_summary,
                )
                judge_eval = self.judge.generate(judge_payload).model_dump()
                logger.info(
                    "Phase2Controller iteration %s: judge verdict=%s score=%s priority_clusters=%s",
                    iteration,
                    judge_eval.get("verdict"),
                    judge_eval.get("reasonableness_score"),
                    judge_eval.get("priority_clusters"),
                )
            else:
                judge_eval = _invalid_judge_stub(tool_eval)
                logger.info(
                    "Phase2Controller iteration %s: skipping judge because tool marked candidate hard-invalid",
                    iteration,
                )

            candidate_payload = PromotePhase2RepairToSeedPayload(
                payload=current_payload,
                repair=proposal,
            )

            history.append(
                {
                    "iteration": iteration,
                    "search_phase": search_phase,
                    "proposal": deepcopy(proposal),
                    "tool_evaluation": deepcopy(tool_eval),
                    "judge_evaluation": deepcopy(judge_eval),
                    "direction_summary": direction_summary,
                }
            )

            if _evaluation_rank(tool_eval, judge_eval) > _evaluation_rank(
                best_tool_eval, best_judge_eval
            ):
                logger.info(
                    "Phase2Controller iteration %s: candidate became new best-by-judge",
                    iteration,
                )
                best_payload = deepcopy(candidate_payload)
                best_proposal = deepcopy(proposal)
                best_tool_eval = deepcopy(tool_eval)
                best_judge_eval = deepcopy(judge_eval)

            if bool(tool_eval.get("hard_valid")) and _evaluation_rank(
                tool_eval, judge_eval
            ) > _evaluation_rank(best_hard_valid_eval, best_hard_valid_judge):
                best_hard_valid_payload = deepcopy(candidate_payload)
                best_hard_valid_proposal = deepcopy(proposal)
                best_hard_valid_eval = deepcopy(tool_eval)
                best_hard_valid_judge = deepcopy(judge_eval)

            if (
                bool(tool_eval.get("hard_valid"))
                and str(judge_eval.get("verdict") or "").upper() == "ACCEPT"
            ):
                logger.info(
                    "Phase2Controller iteration %s: accepted by judge", iteration
                )
                best_payload = deepcopy(candidate_payload)
                best_proposal = deepcopy(proposal)
                best_tool_eval = deepcopy(tool_eval)
                best_judge_eval = deepcopy(judge_eval)
                break

            controller_feedback = _build_controller_feedback(
                iteration=iteration,
                evaluation=tool_eval,
                judge_output=judge_eval,
                history=history,
            )

            if _should_promote_candidate(tool_eval, judge_eval):
                logger.info(
                    "Phase2Controller iteration %s: promoting candidate to next baseline",
                    iteration,
                )
                current_payload = deepcopy(candidate_payload)
                current_baseline = EvaluatePhase2Proposal(
                    payload=current_payload,
                    repair=_canonical_repair(current_payload),
                )
            else:
                logger.info(
                    "Phase2Controller iteration %s: keeping current baseline for next search round",
                    iteration,
                )

        final_payload = deepcopy(best_payload)
        final_repair = _canonical_repair(
            final_payload,
            notes=deepcopy(best_proposal.get("notes") or []),
        )
        hard_fix_result = ResolvePhase2HardViolations(
            payload=final_payload,
            repair=final_repair,
            max_rounds=120,
            max_radius_steps=12,
        )
        final_repair = deepcopy(hard_fix_result.get("repair") or final_repair)
        final_tool_eval = deepcopy(hard_fix_result.get("evaluation") or best_tool_eval)

        if not bool(final_tool_eval.get("hard_valid")):
            logger.warning(
                "Phase2Controller hard-fix did not reach hard-valid output; falling back to best hard-valid candidate"
            )
            final_payload = deepcopy(best_hard_valid_payload)
            final_repair = _canonical_repair(
                final_payload,
                notes=deepcopy(best_hard_valid_proposal.get("notes") or []),
            )
            final_tool_eval = deepcopy(best_hard_valid_eval)
            best_judge_eval = deepcopy(best_hard_valid_judge)

        final_payload["phase2_placer"] = {"phase2_repair": deepcopy(final_repair)}
        absolute_layout = compile_phase2_to_final_output(final_payload)
        logger.info(
            "Phase2Controller done: final_status=%s best_reasonableness=%s hard_fix_attempts=%s",
            absolute_layout.get("status"),
            best_judge_eval.get("reasonableness_score"),
            len(hard_fix_result.get("attempts") or []),
        )

        return {
            "proposal": final_repair,
            "tool_evaluation": final_tool_eval,
            "judge_evaluation": best_judge_eval,
            "absolute_layout": absolute_layout,
            "history": history,
            "hard_fix": hard_fix_result,
        }
