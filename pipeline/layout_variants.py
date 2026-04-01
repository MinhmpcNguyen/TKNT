from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent.cluster_placer import (
    build_phase2_payload,
    compile_phase2_to_final_output,
    make_no_improvement_repair,
)
from agent.stylist import Stylist
from cluster_placer.tools_v2 import (
    EnumeratePhase2RepairMoves,
    EvaluatePhase2Proposal,
    PromotePhase2RepairToSeedPayload,
)
from pipeline.manual_placements import (
    merge_manual_placements_into_absolute_layout,
    merge_manual_placements_into_styled_output,
)


@dataclass(frozen=True)
class _VariantCandidate:
    absolute_layout: dict[str, Any]
    styled_result: dict[str, Any]
    source: str
    reason: str
    layout_score: int
    hard_valid: bool
    notes: list[str]
    promoted_payload: dict[str, Any]


def build_layout_variants_artifact(
    *,
    room_output: dict[str, Any],
    merged_output: dict[str, Any],
    cluster_outlines: dict[str, Any],
    relation_plan: dict[str, Any],
    solver_output: dict[str, Any],
    phase2_result: dict[str, Any],
    absolute_layout: dict[str, Any],
    stylist_output: dict[str, Any],
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    style_plan: dict[str, Any],
    manual_placements: list[dict[str, Any]] | None = None,
    max_variants: int = 5,
) -> dict[str, Any]:
    base_payload = build_phase2_payload(
        room_output,
        merged_output,
        cluster_outlines,
        relation_plan,
        solver_output,
        None,
    )
    selected_repair = phase2_result.get("proposal")
    if not isinstance(selected_repair, dict):
        selected_repair = make_no_improvement_repair(base_payload)
    selected_eval = phase2_result.get("tool_evaluation")
    if not isinstance(selected_eval, dict):
        selected_eval = EvaluatePhase2Proposal(
            payload=base_payload, repair=selected_repair
        )

    selected_promoted = PromotePhase2RepairToSeedPayload(
        payload=base_payload,
        repair=selected_repair,
    )
    variants: list[_VariantCandidate] = []
    seen_signatures: set[str] = set()

    variants.append(
        _VariantCandidate(
            absolute_layout=deepcopy(absolute_layout),
            styled_result=deepcopy(stylist_output),
            source="phase2_selected",
            reason="Selected by the phase-2 controller as the best repaired layout.",
            layout_score=_candidate_score(selected_eval),
            hard_valid=bool(selected_eval.get("hard_valid")),
            notes=_collect_notes(
                selected_repair.get("notes"), absolute_layout.get("notes")
            ),
            promoted_payload=selected_promoted,
        )
    )
    seen_signatures.add(_absolute_layout_signature(absolute_layout))

    seed_repair = make_no_improvement_repair(
        base_payload,
        note="Baseline solver seed layout before phase-2 refinement.",
    )
    seed_eval = EvaluatePhase2Proposal(payload=base_payload, repair=seed_repair)
    seed_candidate = _build_variant_from_repair(
        payload=base_payload,
        repair=seed_repair,
        source="solver_seed",
        reason="Baseline hard-valid solver seed.",
        evaluation=seed_eval,
        stylist=stylist,
        stylist_user_context_json=stylist_user_context_json,
        stylist_tenant_id=stylist_tenant_id,
        style_plan=style_plan,
        manual_placements=manual_placements,
    )
    if seed_candidate is not None:
        signature = _absolute_layout_signature(seed_candidate.absolute_layout)
        if signature not in seen_signatures:
            variants.append(seed_candidate)
            seen_signatures.add(signature)

    queue: list[tuple[dict[str, Any], int]] = [(selected_promoted, 0)]
    if seed_candidate is not None:
        queue.append((seed_candidate.promoted_payload, 0))

    while queue and len(variants) < max_variants:
        current_payload, depth = queue.pop(0)
        moves = (
            EnumeratePhase2RepairMoves(payload=current_payload, limit=32).get("moves")
            or []
        )
        for move in moves:
            if len(variants) >= max_variants:
                break
            if not isinstance(move, dict):
                continue
            repair = move.get("proposal")
            if not isinstance(repair, dict):
                continue
            evaluation = EvaluatePhase2Proposal(payload=current_payload, repair=repair)
            if not bool(evaluation.get("hard_valid")):
                continue
            candidate = _build_variant_from_repair(
                payload=current_payload,
                repair=repair,
                source=str(move.get("kind") or "phase2_move"),
                reason=str(
                    move.get("reason") or "Enumerated deterministic phase-2 move."
                ),
                evaluation=evaluation,
                stylist=stylist,
                stylist_user_context_json=stylist_user_context_json,
                stylist_tenant_id=stylist_tenant_id,
                style_plan=style_plan,
                manual_placements=manual_placements,
            )
            if candidate is None:
                continue
            signature = _absolute_layout_signature(candidate.absolute_layout)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            variants.append(candidate)
            if depth < 1:
                queue.append((candidate.promoted_payload, depth + 1))

    payload_variants: list[dict[str, Any]] = []
    for index, variant in enumerate(variants[:max_variants], start=1):
        payload_variants.append(
            {
                "variant_id": f"variant_{index}",
                "label": f"Option {index}",
                "source": variant.source,
                "reason": variant.reason,
                "layout_score": variant.layout_score,
                "hard_valid": variant.hard_valid,
                "notes": variant.notes,
                "absolute_layout": variant.absolute_layout,
                "styled_result": variant.styled_result,
            }
        )

    return {
        "status": "OK",
        "selected_variant_id": payload_variants[0]["variant_id"]
        if payload_variants
        else None,
        "variants": payload_variants,
    }


def _build_variant_from_repair(
    *,
    payload: dict[str, Any],
    repair: dict[str, Any],
    source: str,
    reason: str,
    evaluation: dict[str, Any],
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    style_plan: dict[str, Any],
    manual_placements: list[dict[str, Any]] | None,
) -> _VariantCandidate | None:
    promoted_payload = PromotePhase2RepairToSeedPayload(payload=payload, repair=repair)
    compilable_payload = deepcopy(promoted_payload)
    compilable_payload["phase2_placer"] = {
        "phase2_repair": make_no_improvement_repair(
            promoted_payload,
            note=reason,
        )
    }
    absolute_layout = compile_phase2_to_final_output(compilable_payload)
    if str(absolute_layout.get("status") or "").upper() != "OK":
        return None
    if manual_placements:
        absolute_layout = merge_manual_placements_into_absolute_layout(
            absolute_layout,
            manual_placements,
        )
    styled_result = stylist.apply_style_plan(
        layout_json=absolute_layout,
        user_context_json=stylist_user_context_json,
        tenant_id=stylist_tenant_id,
        style_plan=style_plan,
    )
    if manual_placements:
        styled_result = merge_manual_placements_into_styled_output(
            styled_result,
            manual_placements,
        )
    return _VariantCandidate(
        absolute_layout=absolute_layout,
        styled_result=styled_result,
        source=source,
        reason=reason,
        layout_score=_candidate_score(evaluation),
        hard_valid=bool(evaluation.get("hard_valid")),
        notes=_collect_notes(repair.get("notes"), absolute_layout.get("notes")),
        promoted_payload=promoted_payload,
    )


def _candidate_score(evaluation: dict[str, Any]) -> int:
    comparison = evaluation.get("baseline_comparison")
    if isinstance(comparison, dict):
        return int(comparison.get("candidate_score") or 0)
    return 0


def _collect_notes(*sources: Any) -> list[str]:
    notes: list[str] = []
    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            text = str(item).strip()
            if text and text not in notes:
                notes.append(text)
    return notes[:8]


def _absolute_layout_signature(layout: dict[str, Any]) -> str:
    objects = layout.get("objects")
    if not isinstance(objects, list):
        return json.dumps(layout, ensure_ascii=True, sort_keys=True)
    rows: list[str] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
        rows.append(
            "|".join(
                [
                    str(item.get("cluster_id") or ""),
                    str(item.get("object_id") or ""),
                    str(int(item.get("rotation_ccw") or 0)),
                    str(int(bbox.get("min_x") or 0)),
                    str(int(bbox.get("min_y") or 0)),
                    str(int(bbox.get("max_x") or 0)),
                    str(int(bbox.get("max_y") or 0)),
                ]
            )
        )
    rows.sort()
    return "\n".join(rows)
