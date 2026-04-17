from __future__ import annotations

import colorsys
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agent.cluster_placer import (
    build_phase2_payload,
    compile_phase2_to_final_output,
    make_no_improvement_repair,
)
from agent.stylist import Stylist, stylist_model_name_for_variant
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
    styled_result: dict[str, Any] | None
    source: str
    reason: str
    layout_score: int
    hard_valid: bool
    acceptable_valid: bool
    notes: list[str]
    promoted_payload: dict[str, Any]
    family: str
    complete: bool = True
    gallery_eligible: bool = True
    coverage_ratio: float = 1.0
    missing_cluster_ids: list[str] = field(default_factory=list)


def build_layout_variants_artifact(
    *,
    room_output: dict[str, Any],
    merged_output: dict[str, Any],
    cluster_outlines: dict[str, Any],
    relation_plan: dict[str, Any],
    solver_output: dict[str, Any],
    phase2_result: dict[str, Any],
    absolute_layout: dict[str, Any],
    stylist_output: dict[str, Any] | None,
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    style_plan: dict[str, Any] | None,
    manual_placements: list[dict[str, Any]] | None = None,
    external_candidates: list[dict[str, Any]] | None = None,
    max_variants: int = 5,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
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
    candidate_pool: list[_VariantCandidate] = []
    seen_signatures: set[str] = set()
    selected_coverage_fields = _extract_candidate_coverage_fields(
        absolute_layout=absolute_layout,
        hard_valid=bool(selected_eval.get("hard_valid")),
        acceptable_valid=bool(selected_eval.get("acceptable_valid")),
    )

    selected_candidate = _VariantCandidate(
        absolute_layout=deepcopy(absolute_layout),
        styled_result=deepcopy(stylist_output)
        if isinstance(stylist_output, dict)
        else None,
        source="phase2_selected",
        reason="Selected by the phase-2 controller as the best repaired layout.",
        layout_score=_candidate_score(selected_eval),
        hard_valid=bool(selected_eval.get("hard_valid")),
        notes=_collect_notes(
            selected_repair.get("notes"), absolute_layout.get("notes")
        ),
        promoted_payload=selected_promoted,
        family=_variant_family("phase2_selected", ""),
        **selected_coverage_fields,
    )
    candidate_pool.append(selected_candidate)
    seen_signatures.add(_absolute_layout_signature(absolute_layout))

    for external in external_candidates or []:
        candidate = _candidate_from_external(external)
        if candidate is None:
            continue
        signature = _absolute_layout_signature(candidate.absolute_layout)
        if signature in seen_signatures:
            continue
        candidate_pool.append(candidate)
        seen_signatures.add(signature)

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
            candidate_pool.append(seed_candidate)
            seen_signatures.add(signature)

    macro_seed_candidates = _build_macro_diverse_seed_candidates(
        payload=base_payload,
        seed_payloads=[
            selected_promoted,
            *([seed_candidate.promoted_payload] if seed_candidate is not None else []),
        ],
        stylist=stylist,
        stylist_user_context_json=stylist_user_context_json,
        stylist_tenant_id=stylist_tenant_id,
        style_plan=style_plan or {},
        manual_placements=manual_placements,
        max_candidates=max(max_variants * 3, 8),
    )
    for candidate in macro_seed_candidates:
        signature = _absolute_layout_signature(candidate.absolute_layout)
        if signature in seen_signatures:
            continue
        candidate_pool.append(candidate)
        seen_signatures.add(signature)

    external_count = len(external_candidates or [])
    run_internal_search = len(candidate_pool) < max_variants or external_count == 0
    queue: list[tuple[dict[str, Any], int]] = []
    if run_internal_search:
        queue.append((selected_promoted, 0))
        if seed_candidate is not None:
            queue.append((seed_candidate.promoted_payload, 0))
        for candidate in macro_seed_candidates[:max_variants]:
            if (
                isinstance(candidate.promoted_payload, dict)
                and candidate.promoted_payload
            ):
                queue.append((candidate.promoted_payload, 0))

    max_search_depth = 2 if external_count else 3
    move_limit = 48 if external_count else 96
    max_pool_size = max(max_variants * (8 if external_count else 12), 16)
    while queue and len(candidate_pool) < max_pool_size:
        current_payload, depth = queue.pop(0)
        moves = (
            EnumeratePhase2RepairMoves(payload=current_payload, limit=move_limit).get(
                "moves"
            )
            or []
        )
        for move in moves:
            if len(candidate_pool) >= max_pool_size:
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
            candidate_pool.append(candidate)
            if depth < max_search_depth:
                queue.append((candidate.promoted_payload, depth + 1))

    variants = _select_diverse_candidates(
        candidate_pool,
        max_variants=max_variants,
    )
    payload_variants: list[dict[str, Any]] = []
    anchor_style_plan = deepcopy(style_plan) if isinstance(style_plan, dict) else None
    anchor_styled_result = (
        deepcopy(stylist_output) if isinstance(stylist_output, dict) else None
    )
    if variants and (
        not isinstance(anchor_style_plan, dict)
        or not isinstance(anchor_styled_result, dict)
    ):
        anchor_layout = deepcopy(variants[0].absolute_layout)
        anchor_style_plan = _generate_style_plan_for_option(
            stylist=stylist,
            layout_json=anchor_layout,
            user_context_json=stylist_user_context_json,
            tenant_id=stylist_tenant_id,
            option_index=1,
            fallback_style_plan=None,
            fallback_styled_result=None,
        )
        anchor_styled_result = stylist.apply_style_plan(
            layout_json=anchor_layout,
            user_context_json=stylist_user_context_json,
            tenant_id=stylist_tenant_id,
            style_plan=anchor_style_plan,
        )
        if manual_placements:
            anchor_styled_result = merge_manual_placements_into_styled_output(
                anchor_styled_result,
                manual_placements,
            )
    for index, variant in enumerate(variants[:max_variants], start=1):
        if (
            index == 1
            and isinstance(anchor_styled_result, dict)
            and isinstance(anchor_style_plan, dict)
        ):
            styled_result = deepcopy(anchor_styled_result)
        else:
            styled_result = _materialize_variant_style(
                candidate=variant,
                option_index=index,
                stylist=stylist,
                stylist_user_context_json=stylist_user_context_json,
                stylist_tenant_id=stylist_tenant_id,
                base_style_plan=anchor_style_plan or {},
                base_styled_result=anchor_styled_result or {},
                manual_placements=manual_placements,
            )
        payload_variants.append(
            {
                "variant_id": f"variant_{index}",
                "label": f"Option {index}",
                "source": variant.source,
                "reason": variant.reason,
                "layout_score": variant.layout_score,
                "hard_valid": variant.hard_valid,
                "acceptable_valid": variant.acceptable_valid,
                "complete": variant.complete,
                "gallery_eligible": variant.gallery_eligible,
                "coverage_ratio": variant.coverage_ratio,
                "missing_cluster_ids": deepcopy(variant.missing_cluster_ids),
                "notes": variant.notes,
                "absolute_layout": variant.absolute_layout,
                "styled_result": styled_result,
            }
        )
        if on_progress is not None:
            on_progress(
                _build_layout_variants_payload(
                    payload_variants=payload_variants,
                    status="BUILDING",
                )
            )

    return _build_layout_variants_payload(
        payload_variants=payload_variants,
        status="OK",
    )


def select_top_variant_candidates(
    *,
    candidates: list[dict[str, Any]],
    max_variants: int,
) -> list[dict[str, Any]]:
    candidate_pool: list[_VariantCandidate] = []
    signature_to_candidate: dict[str, dict[str, Any]] = {}
    signature_to_rank: dict[str, tuple[int, int, float, int]] = {}

    for external in candidates:
        candidate = _candidate_from_external(external)
        if candidate is None:
            continue
        signature = _absolute_layout_signature(candidate.absolute_layout)
        rank = _candidate_rank_tuple(candidate)
        if signature in signature_to_candidate:
            existing_rank = signature_to_rank.get(signature)
            if existing_rank is not None and existing_rank >= rank:
                continue
        candidate_pool.append(candidate)
        signature_to_candidate[signature] = deepcopy(external)
        signature_to_rank[signature] = rank

    candidate_pool.sort(
        key=_candidate_rank_tuple,
        reverse=True,
    )

    selected = _select_diverse_candidates(
        candidate_pool,
        max_variants=max(0, int(max_variants)),
    )
    out: list[dict[str, Any]] = []
    for candidate in selected:
        signature = _absolute_layout_signature(candidate.absolute_layout)
        raw = signature_to_candidate.get(signature)
        if raw is None:
            continue
        out.append(deepcopy(raw))
    return out


def select_distinct_final_gallery_candidates(
    *,
    candidates: list[dict[str, Any]],
    max_variants: int,
) -> list[dict[str, Any]]:
    ranked_entries: list[dict[str, Any]] = []
    for original_index, raw_candidate in enumerate(candidates):
        candidate = _candidate_from_external(raw_candidate)
        if candidate is None:
            continue
        state_signature = _absolute_layout_signature(
            candidate.absolute_layout,
            include_reintroduced=False,
        )
        macro_signature = (
            _macro_region_signature(
                candidate.absolute_layout,
                include_reintroduced=False,
            )
            or state_signature
        )
        rank = _candidate_rank_tuple(candidate)
        ranked_entries.append(
            {
                "original_index": original_index,
                "candidate": candidate,
                "raw": deepcopy(raw_candidate),
                "rank": rank,
                "state_signature": state_signature,
                "macro_layout_signature": macro_signature,
            }
        )

    ranked_entries.sort(
        key=lambda row: row.get("rank") or (0, 0, 0.0, 0),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_layouts: list[dict[str, Any]] = []
    selected_entry_indices: set[int] = set()
    seen_macro_signatures: set[str] = set()
    for entry in ranked_entries:
        candidate = entry.get("candidate")
        if not isinstance(candidate, _VariantCandidate):
            continue
        macro_signature = str(entry.get("macro_layout_signature") or "").strip()
        if macro_signature and macro_signature in seen_macro_signatures:
            continue
        if any(
            _is_near_duplicate_layout(
                candidate.absolute_layout,
                layout,
                include_reintroduced=False,
            )
            for layout in selected_layouts
        ):
            continue
        raw = deepcopy(entry.get("raw") or {})
        raw["state_signature"] = str(entry.get("state_signature") or "")
        raw["macro_layout_signature"] = macro_signature
        raw["gallery_selection_mode"] = "distinct"
        selected.append(raw)
        selected_layouts.append(candidate.absolute_layout)
        selected_entry_indices.add(int(entry.get("original_index") or 0))
        if macro_signature:
            seen_macro_signatures.add(macro_signature)
        if len(selected) >= max(0, int(max_variants)):
            return selected

    for entry in ranked_entries:
        original_index = int(entry.get("original_index") or 0)
        if original_index in selected_entry_indices:
            continue
        raw = deepcopy(entry.get("raw") or {})
        raw["state_signature"] = str(entry.get("state_signature") or "")
        raw["macro_layout_signature"] = str(
            entry.get("macro_layout_signature") or ""
        ).strip()
        raw["gallery_selection_mode"] = "backfill"
        selected.append(raw)
        selected_entry_indices.add(original_index)
        if len(selected) >= max(0, int(max_variants)):
            break
    return selected


def build_final_layout_variants_payload(
    *,
    candidates: list[dict[str, Any]],
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    manual_placements: list[dict[str, Any]] | None = None,
    max_variants: int = 5,
) -> dict[str, Any]:
    selected_candidates = [
        deepcopy(candidate) for candidate in candidates[:max_variants]
    ]
    if not selected_candidates:
        return _build_layout_variants_payload(payload_variants=[], status="OK")

    anchor_layout = selected_candidates[0].get("absolute_layout")
    if not isinstance(anchor_layout, dict):
        return _build_layout_variants_payload(payload_variants=[], status="OK")

    base_style_plan = _generate_style_plan_for_option(
        stylist=stylist,
        layout_json=anchor_layout,
        user_context_json=stylist_user_context_json,
        tenant_id=stylist_tenant_id,
        option_index=1,
        fallback_style_plan=None,
        fallback_styled_result=None,
    )
    anchor_styled_result = stylist.apply_style_plan(
        layout_json=anchor_layout,
        user_context_json=stylist_user_context_json,
        tenant_id=stylist_tenant_id,
        style_plan=base_style_plan,
    )
    if manual_placements:
        anchor_styled_result = merge_manual_placements_into_styled_output(
            anchor_styled_result,
            manual_placements,
        )

    payload_variants: list[dict[str, Any]] = []
    anchor_styled_payload = deepcopy(anchor_styled_result)
    for index, raw_candidate in enumerate(selected_candidates[:max_variants], start=1):
        candidate = _candidate_from_external(raw_candidate)
        if candidate is None:
            continue
        if index == 1:
            styled_result = deepcopy(anchor_styled_payload)
        else:
            styled_result = _materialize_variant_style(
                candidate=candidate,
                option_index=index,
                stylist=stylist,
                stylist_user_context_json=stylist_user_context_json,
                stylist_tenant_id=stylist_tenant_id,
                base_style_plan=base_style_plan,
                base_styled_result=anchor_styled_payload,
                manual_placements=manual_placements,
            )
        payload_variants.append(
            {
                "variant_id": f"variant_{index}",
                "label": f"Option {index}",
                "source": candidate.source,
                "state_signature": _absolute_layout_signature(
                    candidate.absolute_layout
                ),
                "macro_layout_signature": _macro_region_signature(
                    candidate.absolute_layout
                ),
                "reason": candidate.reason,
                "layout_score": candidate.layout_score,
                "hard_valid": candidate.hard_valid,
                "complete": candidate.complete,
                "gallery_eligible": candidate.gallery_eligible,
                "coverage_ratio": candidate.coverage_ratio,
                "missing_cluster_ids": deepcopy(candidate.missing_cluster_ids),
                "notes": deepcopy(candidate.notes),
                "absolute_layout": deepcopy(candidate.absolute_layout),
                "styled_result": styled_result,
            }
        )

    return _build_layout_variants_payload(
        payload_variants=payload_variants,
        status="OK",
    )


def _build_layout_variants_payload(
    *,
    payload_variants: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "selected_variant_id": payload_variants[0]["variant_id"]
        if payload_variants
        else None,
        "variants": deepcopy(payload_variants),
    }


def _extract_candidate_coverage_fields(
    *,
    absolute_layout: dict[str, Any],
    hard_valid: bool,
    acceptable_valid: bool | None = None,
) -> dict[str, Any]:
    coverage = (
        absolute_layout.get("coverage")
        if isinstance(absolute_layout.get("coverage"), dict)
        else {}
    )
    complete = bool(
        absolute_layout.get(
            "complete",
            coverage.get("complete", hard_valid),
        )
    )
    acceptable = bool(
        absolute_layout.get(
            "acceptable_valid",
            coverage.get(
                "acceptable_valid",
                hard_valid if acceptable_valid is None else acceptable_valid,
            ),
        )
    )
    gallery_eligible = bool(
        absolute_layout.get(
            "gallery_eligible",
            coverage.get("gallery_eligible", complete and hard_valid and acceptable),
        )
    ) and acceptable
    coverage_ratio = float(
        absolute_layout.get(
            "coverage_ratio",
            coverage.get("coverage_ratio", 1.0 if complete else 0.0),
        )
        or 0.0
    )
    missing_cluster_ids = absolute_layout.get("missing_cluster_ids")
    if not isinstance(missing_cluster_ids, list):
        missing_cluster_ids = coverage.get("missing_cluster_ids")
    return {
        "complete": complete,
        "acceptable_valid": acceptable,
        "gallery_eligible": gallery_eligible,
        "coverage_ratio": coverage_ratio,
        "missing_cluster_ids": [
            str(item).strip()
            for item in (missing_cluster_ids or [])
            if str(item).strip()
        ],
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
    coverage_fields = _extract_candidate_coverage_fields(
        absolute_layout=absolute_layout,
        hard_valid=bool(evaluation.get("hard_valid")),
        acceptable_valid=bool(evaluation.get("acceptable_valid")),
    )
    return _VariantCandidate(
        absolute_layout=absolute_layout,
        styled_result=None,
        source=source,
        reason=reason,
        layout_score=_candidate_score(evaluation),
        hard_valid=bool(evaluation.get("hard_valid")),
        acceptable_valid=bool(coverage_fields["acceptable_valid"]),
        notes=_collect_notes(repair.get("notes"), absolute_layout.get("notes")),
        promoted_payload=promoted_payload,
        family=_variant_family(source, reason),
        complete=bool(coverage_fields["complete"]),
        gallery_eligible=bool(coverage_fields["gallery_eligible"]),
        coverage_ratio=float(coverage_fields["coverage_ratio"]),
        missing_cluster_ids=list(coverage_fields["missing_cluster_ids"]),
    )


def _candidate_from_external(external: dict[str, Any]) -> _VariantCandidate | None:
    if not isinstance(external, dict):
        return None
    absolute_layout = external.get("absolute_layout")
    if not isinstance(absolute_layout, dict):
        return None
    source = str(external.get("source") or "intent_branch").strip() or "intent_branch"
    reason = str(external.get("reason") or "Intent-derived branch candidate.").strip()
    if not reason:
        reason = "Intent-derived branch candidate."
    notes = _collect_notes(external.get("notes"))
    coverage_fields = _extract_candidate_coverage_fields(
        absolute_layout=absolute_layout,
        hard_valid=bool(external.get("hard_valid", True)),
        acceptable_valid=(
            bool(external.get("acceptable_valid"))
            if "acceptable_valid" in external
            else None
        ),
    )
    return _VariantCandidate(
        absolute_layout=deepcopy(absolute_layout),
        styled_result=deepcopy(external.get("styled_result"))
        if isinstance(external.get("styled_result"), dict)
        else None,
        source=source,
        reason=reason,
        layout_score=int(external.get("layout_score") or 0),
        hard_valid=bool(external.get("hard_valid", True)),
        acceptable_valid=bool(coverage_fields["acceptable_valid"]),
        notes=notes,
        promoted_payload=deepcopy(external.get("promoted_payload"))
        if isinstance(external.get("promoted_payload"), dict)
        else {},
        family=str(external.get("family") or source),
        complete=bool(coverage_fields["complete"]),
        gallery_eligible=bool(coverage_fields["gallery_eligible"]),
        coverage_ratio=float(coverage_fields["coverage_ratio"]),
        missing_cluster_ids=list(coverage_fields["missing_cluster_ids"]),
    )


def _candidate_score(evaluation: dict[str, Any]) -> int:
    comparison = evaluation.get("baseline_comparison")
    if isinstance(comparison, dict):
        return int(comparison.get("candidate_score") or 0)
    return 0


def _candidate_rank_tuple(candidate: _VariantCandidate) -> tuple[int, int, float, int]:
    return (
        1 if candidate.gallery_eligible else 0,
        1 if candidate.complete else 0,
        float(candidate.coverage_ratio),
        int(candidate.layout_score),
    )


def _build_macro_diverse_seed_candidates(
    *,
    payload: dict[str, Any],
    seed_payloads: list[dict[str, Any]],
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    style_plan: dict[str, Any],
    manual_placements: list[dict[str, Any]] | None,
    max_candidates: int,
) -> list[_VariantCandidate]:
    candidates: list[_VariantCandidate] = []
    seen_buckets: set[tuple[str, str, str]] = set()
    seen_layout_signatures: set[str] = set()

    for seed_payload in seed_payloads:
        if not isinstance(seed_payload, dict) or not seed_payload:
            continue
        moves = (
            EnumeratePhase2RepairMoves(payload=seed_payload, limit=96).get("moves")
            or []
        )
        for move in moves:
            if len(candidates) >= max_candidates:
                return candidates
            if not isinstance(move, dict):
                continue
            kind = str(move.get("kind") or "").strip()
            if kind not in {"cluster_pose", "cluster_variant"}:
                continue
            repair = move.get("proposal")
            if not isinstance(repair, dict):
                continue
            evaluation = EvaluatePhase2Proposal(payload=seed_payload, repair=repair)
            if not bool(evaluation.get("hard_valid")):
                continue
            candidate = _build_variant_from_repair(
                payload=seed_payload,
                repair=repair,
                source=kind,
                reason=str(
                    move.get("reason") or "Enumerated deterministic macro layout move."
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
            layout_signature = _absolute_layout_signature(candidate.absolute_layout)
            if layout_signature in seen_layout_signatures:
                continue
            macro_signature = _macro_region_signature(candidate.absolute_layout)
            cluster_id = str(move.get("cluster_id") or "").strip()
            reason_bucket = _macro_reason_bucket(str(move.get("reason") or ""))
            bucket = (cluster_id, kind, macro_signature or reason_bucket)
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            seen_layout_signatures.add(layout_signature)
            candidates.append(candidate)
    return candidates


def _macro_reason_bucket(reason: str) -> str:
    lowered = reason.strip().lower()
    if "entry" in lowered:
        return "entry"
    if "window" in lowered:
        return "window"
    if "edge" in lowered or "wall" in lowered:
        return "edge"
    if "view" in lowered:
        return "view"
    if "variant" in lowered:
        return "variant"
    if "rotation" in lowered:
        return "rotation"
    return lowered or "macro"


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


def _absolute_layout_signature(
    layout: dict[str, Any],
    *,
    include_reintroduced: bool = True,
) -> str:
    objects = layout.get("objects")
    if not isinstance(objects, list):
        return json.dumps(layout, ensure_ascii=True, sort_keys=True)
    rows: list[str] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        if not include_reintroduced and _is_reintroduced_layout_object(item):
            continue
        bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
        identity_key = _layout_identity_key(item)
        rows.append(
            "|".join(
                [
                    identity_key,
                    str(int(item.get("rotation_ccw") or 0)),
                    str(int(bbox.get("min_x") or 0)),
                    str(int(bbox.get("min_y") or 0)),
                    str(int(bbox.get("max_x") or 0)),
                    str(int(bbox.get("max_y") or 0)),
                ]
            )
        )
    rows.sort()
    if rows:
        return "\n".join(rows)
    return json.dumps(layout, ensure_ascii=True, sort_keys=True)


def _select_diverse_candidates(
    candidates: list[_VariantCandidate],
    *,
    max_variants: int,
) -> list[_VariantCandidate]:
    if not candidates:
        return []
    if len(candidates) <= max_variants:
        return candidates

    anchor = candidates[0]
    remaining = sorted(candidates[1:], key=lambda row: row.layout_score, reverse=True)
    selected: list[_VariantCandidate] = [anchor]
    family_counts: dict[str, int] = {anchor.family: 1}
    macro_signatures = {_macro_region_signature(anchor.absolute_layout)}
    best_score = max(row.layout_score for row in candidates)

    while remaining and len(selected) < max_variants:
        best_index = 0
        best_value: float | None = None
        for index, candidate in enumerate(remaining):
            macro_signature = _macro_region_signature(candidate.absolute_layout)
            min_distance = min(
                _layout_distance_score(candidate.absolute_layout, row.absolute_layout)
                for row in selected
            )
            near_duplicate = any(
                _is_near_duplicate_layout(
                    candidate.absolute_layout, row.absolute_layout
                )
                for row in selected
            )
            quality_delta = candidate.layout_score - best_score
            eligibility_bonus = 5200 if candidate.gallery_eligible else -5200
            completeness_bonus = 2200 if candidate.complete else -2600
            coverage_bonus = int(round(candidate.coverage_ratio * 1600))
            family_bonus = 1400 if family_counts.get(candidate.family, 0) == 0 else 0
            family_penalty = 900 * family_counts.get(candidate.family, 0)
            macro_bonus = 4200 if macro_signature not in macro_signatures else -3200
            duplicate_penalty = 2400 if near_duplicate else 0
            composite = (
                (quality_delta * 4)
                + (min_distance * 12)
                + eligibility_bonus
                + completeness_bonus
                + coverage_bonus
                + family_bonus
                + macro_bonus
                - family_penalty
                - duplicate_penalty
            )
            if best_value is None or composite > best_value:
                best_index = index
                best_value = composite
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        family_counts[chosen.family] = family_counts.get(chosen.family, 0) + 1
        macro_signatures.add(_macro_region_signature(chosen.absolute_layout))

    return selected[:max_variants]


def _layout_distance_score(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    include_reintroduced: bool = True,
) -> int:
    left_boxes = _cluster_boxes(left, include_reintroduced=include_reintroduced)
    right_boxes = _cluster_boxes(right, include_reintroduced=include_reintroduced)
    keys = set(left_boxes) | set(right_boxes)
    if not keys:
        return 0
    score = 0.0
    for key in keys:
        left_box = left_boxes.get(key)
        right_box = right_boxes.get(key)
        if left_box is None or right_box is None:
            score += 5000
            continue
        left_cx = (left_box["min_x"] + left_box["max_x"]) / 2.0
        left_cy = (left_box["min_y"] + left_box["max_y"]) / 2.0
        right_cx = (right_box["min_x"] + right_box["max_x"]) / 2.0
        right_cy = (right_box["min_y"] + right_box["max_y"]) / 2.0
        score += abs(left_cx - right_cx) + abs(left_cy - right_cy)
        score += 0.35 * (
            abs(
                (left_box["max_x"] - left_box["min_x"])
                - (right_box["max_x"] - right_box["min_x"])
            )
            + abs(
                (left_box["max_y"] - left_box["min_y"])
                - (right_box["max_y"] - right_box["min_y"])
            )
        )
    return int(score)


def _is_near_duplicate_layout(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    include_reintroduced: bool = True,
) -> bool:
    if _macro_region_signature(
        left,
        include_reintroduced=include_reintroduced,
    ) == _macro_region_signature(
        right,
        include_reintroduced=include_reintroduced,
    ):
        return True
    left_boxes = _cluster_boxes(left, include_reintroduced=include_reintroduced)
    right_boxes = _cluster_boxes(right, include_reintroduced=include_reintroduced)
    shared_cluster_count = len(set(left_boxes) & set(right_boxes))
    threshold = max(350, shared_cluster_count * 140)
    return (
        _layout_distance_score(
            left,
            right,
            include_reintroduced=include_reintroduced,
        )
        < threshold
    )


def _cluster_boxes(
    layout: dict[str, Any],
    *,
    include_reintroduced: bool = True,
) -> dict[str, dict[str, int]]:
    objects = layout.get("objects")
    if not isinstance(objects, list):
        return {}
    boxes: dict[str, dict[str, int]] = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        if not include_reintroduced and _is_reintroduced_layout_object(item):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, dict):
            continue
        min_x = int(bbox.get("min_x") or 0)
        min_y = int(bbox.get("min_y") or 0)
        max_x = int(bbox.get("max_x") or 0)
        max_y = int(bbox.get("max_y") or 0)
        key = _layout_identity_key(item)
        if not key:
            continue
        current = boxes.get(key)
        if current is None:
            boxes[key] = {
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
            }
            continue
        current["min_x"] = min(current["min_x"], min_x)
        current["min_y"] = min(current["min_y"], min_y)
        current["max_x"] = max(current["max_x"], max_x)
        current["max_y"] = max(current["max_y"], max_y)
    return boxes


def _cluster_macro_boxes(
    layout: dict[str, Any],
    *,
    include_reintroduced: bool = True,
) -> dict[str, dict[str, int]]:
    objects = layout.get("objects")
    if not isinstance(objects, list):
        return {}
    boxes: dict[str, dict[str, int]] = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        if not include_reintroduced and _is_reintroduced_layout_object(item):
            continue
        cluster_id = str(item.get("cluster_id") or "").strip()
        bbox = item.get("bbox")
        if not cluster_id or not isinstance(bbox, dict):
            continue
        min_x = int(bbox.get("min_x") or 0)
        min_y = int(bbox.get("min_y") or 0)
        max_x = int(bbox.get("max_x") or 0)
        max_y = int(bbox.get("max_y") or 0)
        current = boxes.get(cluster_id)
        if current is None:
            boxes[cluster_id] = {
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
            }
            continue
        current["min_x"] = min(current["min_x"], min_x)
        current["min_y"] = min(current["min_y"], min_y)
        current["max_x"] = max(current["max_x"], max_x)
        current["max_y"] = max(current["max_y"], max_y)
    return boxes


def _macro_region_signature(
    layout: dict[str, Any],
    *,
    include_reintroduced: bool = True,
) -> str:
    boxes = _cluster_macro_boxes(layout, include_reintroduced=include_reintroduced)
    if not boxes:
        return ""
    room = layout.get("room")
    polygon = room.get("polygon_ccw") if isinstance(room, dict) else None
    xs: list[int] = []
    ys: list[int] = []
    if isinstance(polygon, list):
        for point in polygon:
            if not isinstance(point, dict):
                continue
            xs.append(int(point.get("x") or 0))
            ys.append(int(point.get("y") or 0))
    if not xs or not ys:
        for box in boxes.values():
            xs.extend([box["min_x"], box["max_x"]])
            ys.extend([box["min_y"], box["max_y"]])
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)

    def _bucket(value: float, origin: int, span: int) -> int:
        normalized = (value - origin) / span
        if normalized < (1.0 / 3.0):
            return 0
        if normalized < (2.0 / 3.0):
            return 1
        return 2

    parts: list[str] = []
    for cluster_id in sorted(boxes):
        box = boxes[cluster_id]
        center_x = (box["min_x"] + box["max_x"]) / 2.0
        center_y = (box["min_y"] + box["max_y"]) / 2.0
        parts.append(
            f"{cluster_id}:{_bucket(center_x, min_x, width)}{_bucket(center_y, min_y, height)}"
        )
    return "|".join(parts)


def _is_reintroduced_layout_object(item: dict[str, Any]) -> bool:
    object_id = str(item.get("object_id") or item.get("instance_id") or "").strip()
    return "__reintroduced_" in object_id


def _layout_identity_key(item: dict[str, Any]) -> str:
    object_id = str(item.get("object_id") or item.get("instance_id") or "").strip()
    if object_id:
        return object_id
    cluster_id = str(item.get("cluster_id") or "").strip()
    if cluster_id:
        return cluster_id
    return ""


def _variant_family(source: str, reason: str) -> str:
    lowered = f"{source} {reason}".lower()
    if "phase2_selected" in lowered:
        return "phase2_selected"
    if "solver_seed" in lowered:
        return "solver_seed"
    if "variant" in lowered or "flip" in lowered or "mirror" in lowered:
        return "cluster_variant"
    if "object" in lowered:
        return "object_refine"
    if "cluster_pose" in lowered or "move" in lowered or "away" in lowered:
        return "cluster_pose"
    return source.strip().lower() or "phase2_move"


def _materialize_variant_style(
    *,
    candidate: _VariantCandidate,
    option_index: int,
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    base_style_plan: dict[str, Any],
    base_styled_result: dict[str, Any],
    manual_placements: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if option_index == 1 and isinstance(candidate.styled_result, dict):
        return deepcopy(candidate.styled_result)

    style_plan = _generate_style_plan_for_option(
        stylist=stylist,
        layout_json=candidate.absolute_layout,
        user_context_json=stylist_user_context_json,
        tenant_id=stylist_tenant_id,
        option_index=option_index,
        fallback_style_plan=base_style_plan,
        fallback_styled_result=base_styled_result,
    )
    styled_result = stylist.apply_style_plan(
        layout_json=candidate.absolute_layout,
        user_context_json=stylist_user_context_json,
        tenant_id=stylist_tenant_id,
        style_plan=style_plan,
    )
    if manual_placements:
        styled_result = merge_manual_placements_into_styled_output(
            styled_result,
            manual_placements,
        )
    return styled_result


def _generate_style_plan_for_option(
    *,
    stylist: Stylist,
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any],
    tenant_id: str | None,
    option_index: int,
    fallback_style_plan: dict[str, Any] | None,
    fallback_styled_result: dict[str, Any] | None,
) -> dict[str, Any]:
    generator = getattr(stylist, "generate_style_plan", None)
    if callable(generator):
        style_plan = generator(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
            model_name=stylist_model_name_for_variant(option_index),
        )
        if isinstance(style_plan, dict):
            return style_plan

    if option_index <= 1:
        return (
            deepcopy(fallback_style_plan)
            if isinstance(fallback_style_plan, dict)
            else {}
        )

    return _build_variant_style_plan(
        base_style_plan=fallback_style_plan or {},
        base_styled_result=fallback_styled_result or {},
        variant_index=option_index - 1,
    )


def _build_variant_style_plan(
    *,
    base_style_plan: dict[str, Any],
    base_styled_result: dict[str, Any],
    variant_index: int,
) -> dict[str, Any]:
    template = _style_plan_from_styled_result(base_styled_result)
    if isinstance(base_style_plan, dict):
        for key in ("room_surfaces", "opening_colors", "object_styles", "notes"):
            value = base_style_plan.get(key)
            if value is not None:
                template[key] = deepcopy(value)
    if variant_index <= 0:
        return template

    profile = _style_variant_profile(variant_index)
    room_surfaces = template.get("room_surfaces")
    if isinstance(room_surfaces, dict):
        for key in ("wall_color_hex", "floor_color_hex", "ceiling_color_hex"):
            color_hex = room_surfaces.get(key)
            if isinstance(color_hex, str):
                room_surfaces[key] = _transform_hex_color(
                    color_hex,
                    hue_shift_deg=profile["surface_hue_shift_deg"],
                    saturation_scale=profile["surface_saturation_scale"],
                    lightness_delta=profile["surface_lightness_delta"],
                )

    opening_colors = template.get("opening_colors")
    if isinstance(opening_colors, dict):
        for key in ("doors", "windows"):
            rows = opening_colors.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                color_hex = row.get("color_hex")
                if isinstance(color_hex, str):
                    row["color_hex"] = _transform_hex_color(
                        color_hex,
                        hue_shift_deg=profile["surface_hue_shift_deg"] / 2.0,
                        saturation_scale=profile["surface_saturation_scale"],
                        lightness_delta=profile["surface_lightness_delta"] / 2.0,
                    )

    object_styles = template.get("object_styles")
    if isinstance(object_styles, list):
        for row in object_styles:
            if not isinstance(row, dict):
                continue
            color_hex = row.get("color_hex")
            if not isinstance(color_hex, str):
                continue
            instance_id = str(row.get("instance_id") or "")
            object_hue_shift = profile["object_hue_shift_deg"]
            if instance_id:
                object_hue_shift += ((sum(ord(ch) for ch in instance_id) % 3) - 1) * 3.0
            row["color_hex"] = _transform_hex_color(
                color_hex,
                hue_shift_deg=object_hue_shift,
                saturation_scale=profile["object_saturation_scale"],
                lightness_delta=profile["object_lightness_delta"],
            )

    notes = template.get("notes")
    if not isinstance(notes, list):
        notes = []
        template["notes"] = notes
    style_note = f"Variant palette: {profile['label']}."
    if style_note not in notes:
        notes.append(style_note)
    template["notes"] = notes[:6]
    return template


def _style_plan_from_styled_result(styled_result: dict[str, Any]) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "room_surfaces": {},
        "opening_colors": {"doors": [], "windows": []},
        "object_styles": [],
        "notes": [],
    }
    room = styled_result.get("room")
    if isinstance(room, dict):
        surfaces = room.get("surfaces")
        if isinstance(surfaces, dict):
            plan["room_surfaces"] = {
                key: value for key, value in surfaces.items() if isinstance(value, str)
            }
        opening_colors = room.get("opening_colors")
        if isinstance(opening_colors, dict):
            for key in ("doors", "windows"):
                rows = opening_colors.get(key)
                if isinstance(rows, list):
                    plan["opening_colors"][key] = [
                        {
                            "id": row.get("id"),
                            "color_hex": row.get("color_hex"),
                        }
                        for row in rows
                        if isinstance(row, dict) and isinstance(row.get("id"), str)
                    ]
    objects = styled_result.get("objects")
    if isinstance(objects, list):
        plan["object_styles"] = [
            {
                "instance_id": row.get("instance_id"),
                "color_hex": row.get("color_hex"),
                "material": row.get("material"),
            }
            for row in objects
            if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
        ]
    notes = styled_result.get("notes")
    if isinstance(notes, list):
        plan["notes"] = [str(note).strip() for note in notes if str(note).strip()]
    return plan


def _style_variant_profile(variant_index: int) -> dict[str, float | str]:
    profiles: list[dict[str, float | str]] = [
        {
            "label": "warmer contrast",
            "surface_hue_shift_deg": 6.0,
            "surface_saturation_scale": 1.02,
            "surface_lightness_delta": 0.02,
            "object_hue_shift_deg": 10.0,
            "object_saturation_scale": 1.08,
            "object_lightness_delta": 0.01,
        },
        {
            "label": "cooler balance",
            "surface_hue_shift_deg": -7.0,
            "surface_saturation_scale": 1.04,
            "surface_lightness_delta": 0.01,
            "object_hue_shift_deg": -12.0,
            "object_saturation_scale": 1.1,
            "object_lightness_delta": -0.02,
        },
        {
            "label": "earthier depth",
            "surface_hue_shift_deg": 10.0,
            "surface_saturation_scale": 0.97,
            "surface_lightness_delta": -0.03,
            "object_hue_shift_deg": 14.0,
            "object_saturation_scale": 1.0,
            "object_lightness_delta": -0.04,
        },
        {
            "label": "lighter calm",
            "surface_hue_shift_deg": -3.0,
            "surface_saturation_scale": 0.94,
            "surface_lightness_delta": 0.05,
            "object_hue_shift_deg": -5.0,
            "object_saturation_scale": 0.96,
            "object_lightness_delta": 0.03,
        },
    ]
    return profiles[(variant_index - 1) % len(profiles)]


def _transform_hex_color(
    color_hex: str,
    *,
    hue_shift_deg: float,
    saturation_scale: float,
    lightness_delta: float,
) -> str:
    text = color_hex.strip()
    if len(text) != 7 or not text.startswith("#"):
        return color_hex
    try:
        red = int(text[1:3], 16) / 255.0
        green = int(text[3:5], 16) / 255.0
        blue = int(text[5:7], 16) / 255.0
    except ValueError:
        return color_hex
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    hue = (hue + (hue_shift_deg / 360.0)) % 1.0
    saturation = min(1.0, max(0.0, saturation * saturation_scale))
    lightness = min(0.93, max(0.08, lightness + lightness_delta))
    out_red, out_green, out_blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(out_red * 255)),
        int(round(out_green * 255)),
        int(round(out_blue * 255)),
    )
