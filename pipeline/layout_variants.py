from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


_HARD_REQUEST_INTENTS = {"must_keep", "must_try"}
_SOLVER_OBJECT_PARTICIPATIONS = {"hard_solver_object", "soft_solver_object"}
_POST_SOLVER_PARTICIPATIONS = {"post_solver_accessory", "ignored_unavailable"}


@dataclass(frozen=True)
class _VariantCandidate:
    absolute_layout: dict[str, Any]
    styled_result: dict[str, Any] | None
    source: str
    reason: str
    layout_score: int
    hard_valid: bool
    complete: bool
    gallery_eligible: bool
    coverage_ratio: float
    request_coverage_score: int
    missing_cluster_ids: list[str] = field(default_factory=list)
    missing_hard_requested_object_types: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    concept_signature: str = ""
    anchor_signature: str = ""
    macro_layout_signature: str = ""
    visual_family_signature: str = ""
    state_signature: str = ""


_BackfillRank = tuple[int, int, int, int, int, int, int, int, float, int, int]


def select_distinct_final_gallery_candidates(
    *,
    candidates: list[dict[str, Any]],
    max_variants: int,
) -> list[dict[str, Any]]:
    target_count = max(0, int(max_variants))
    if target_count <= 0:
        return []

    pool: list[tuple[_VariantCandidate, dict[str, Any]]] = []
    for raw in candidates:
        candidate = _coerce_candidate(raw)
        if candidate is None:
            continue
        if not candidate.hard_valid or not candidate.gallery_eligible:
            continue
        pool.append((candidate, deepcopy(raw)))

    pool.sort(key=lambda item: _candidate_rank_tuple(item[0]), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_candidates: list[_VariantCandidate] = []
    selected_indices: set[int] = set()
    seen_state: set[str] = set()
    seen_visual_family: set[str] = set()
    seen_concept: set[str] = set()
    seen_anchor: set[str] = set()

    for pool_index, (candidate, raw) in enumerate(pool):
        if len(selected) >= target_count:
            break
        if candidate.state_signature and candidate.state_signature in seen_state:
            continue
        visual_family_repeat = bool(
            candidate.visual_family_signature
            and candidate.visual_family_signature in seen_visual_family
        )
        if selected_candidates and visual_family_repeat:
            continue
        if any(
            _is_near_duplicate_layout(candidate.absolute_layout, row.absolute_layout)
            for row in selected_candidates
        ):
            continue

        concept_repeat = bool(
            candidate.concept_signature and candidate.concept_signature in seen_concept
        )
        anchor_repeat = bool(
            candidate.anchor_signature and candidate.anchor_signature in seen_anchor
        )
        if selected_candidates and concept_repeat and anchor_repeat:
            continue

        _accept_gallery_candidate(
            candidate=candidate,
            raw=raw,
            pool_index=pool_index,
            mode="distinct",
            selected=selected,
            selected_candidates=selected_candidates,
            selected_indices=selected_indices,
            seen_state=seen_state,
            seen_visual_family=seen_visual_family,
            seen_concept=seen_concept,
            seen_anchor=seen_anchor,
        )

    while len(selected) < target_count:
        backfill_row = _best_backfill_candidate(
            pool=pool,
            selected_candidates=selected_candidates,
            selected_indices=selected_indices,
            seen_state=seen_state,
            seen_visual_family=seen_visual_family,
            seen_concept=seen_concept,
            seen_anchor=seen_anchor,
        )
        if backfill_row is None:
            break
        pool_index, candidate, raw = backfill_row
        _accept_gallery_candidate(
            candidate=candidate,
            raw=raw,
            pool_index=pool_index,
            mode="fallback_near_distinct",
            selected=selected,
            selected_candidates=selected_candidates,
            selected_indices=selected_indices,
            seen_state=seen_state,
            seen_visual_family=seen_visual_family,
            seen_concept=seen_concept,
            seen_anchor=seen_anchor,
        )

    return selected[:target_count]


def material_macro_layout_signature(layout: dict[str, Any]) -> str:
    return _macro_layout_signature(layout)


def _accept_gallery_candidate(
    *,
    candidate: _VariantCandidate,
    raw: dict[str, Any],
    pool_index: int,
    mode: str,
    selected: list[dict[str, Any]],
    selected_candidates: list[_VariantCandidate],
    selected_indices: set[int],
    seen_state: set[str],
    seen_visual_family: set[str],
    seen_concept: set[str],
    seen_anchor: set[str],
) -> None:
    raw["state_signature"] = candidate.state_signature
    raw["macro_layout_signature"] = candidate.macro_layout_signature
    raw["actual_macro_layout_signature"] = candidate.macro_layout_signature
    raw["visual_family_signature"] = candidate.visual_family_signature
    raw["concept_signature"] = candidate.concept_signature
    raw["anchor_signature"] = candidate.anchor_signature
    raw["request_coverage_score"] = candidate.request_coverage_score
    raw["gallery_selection_mode"] = mode
    selected.append(raw)
    selected_candidates.append(candidate)
    selected_indices.add(pool_index)
    if candidate.state_signature:
        seen_state.add(candidate.state_signature)
    if candidate.visual_family_signature:
        seen_visual_family.add(candidate.visual_family_signature)
    if candidate.concept_signature:
        seen_concept.add(candidate.concept_signature)
    if candidate.anchor_signature:
        seen_anchor.add(candidate.anchor_signature)


def _best_backfill_candidate(
    *,
    pool: list[tuple[_VariantCandidate, dict[str, Any]]],
    selected_candidates: list[_VariantCandidate],
    selected_indices: set[int],
    seen_state: set[str],
    seen_visual_family: set[str],
    seen_concept: set[str],
    seen_anchor: set[str],
) -> tuple[int, _VariantCandidate, dict[str, Any]] | None:
    best_rank: _BackfillRank | None = None
    best_row: tuple[int, _VariantCandidate, dict[str, Any]] | None = None
    for pool_index, (candidate, raw) in enumerate(pool):
        if pool_index in selected_indices:
            continue
        if candidate.state_signature and candidate.state_signature in seen_state:
            continue
        if not _candidate_adds_backfill_diversity(
            candidate,
            selected_candidates=selected_candidates,
            seen_visual_family=seen_visual_family,
        ):
            continue
        rank = _candidate_backfill_rank_tuple(
            candidate,
            selected_candidates=selected_candidates,
            seen_visual_family=seen_visual_family,
            seen_concept=seen_concept,
            seen_anchor=seen_anchor,
            pool_index=pool_index,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_row = (pool_index, candidate, deepcopy(raw))
    return best_row


def _candidate_adds_backfill_diversity(
    candidate: _VariantCandidate,
    *,
    selected_candidates: list[_VariantCandidate],
    seen_visual_family: set[str],
) -> bool:
    if not selected_candidates:
        return True
    if (
        candidate.visual_family_signature
        and candidate.visual_family_signature not in seen_visual_family
    ):
        return True
    selected_macro_signatures = {
        row.macro_layout_signature
        for row in selected_candidates
        if row.macro_layout_signature
    }
    return bool(
        candidate.macro_layout_signature
        and candidate.macro_layout_signature not in selected_macro_signatures
    )


def _candidate_backfill_rank_tuple(
    candidate: _VariantCandidate,
    *,
    selected_candidates: list[_VariantCandidate],
    seen_visual_family: set[str],
    seen_concept: set[str],
    seen_anchor: set[str],
    pool_index: int,
) -> _BackfillRank:
    concept_is_new = int(
        bool(
            candidate.concept_signature
            and candidate.concept_signature not in seen_concept
        )
    )
    anchor_is_new = int(
        bool(
            candidate.anchor_signature and candidate.anchor_signature not in seen_anchor
        )
    )
    visual_family_is_new = int(
        bool(
            candidate.visual_family_signature
            and candidate.visual_family_signature not in seen_visual_family
        )
    )
    selected_macro_signatures = {
        row.macro_layout_signature
        for row in selected_candidates
        if row.macro_layout_signature
    }
    macro_layout_is_new = int(
        bool(
            candidate.macro_layout_signature
            and candidate.macro_layout_signature not in selected_macro_signatures
        )
    )
    if selected_candidates:
        distance = min(
            _layout_distance_score(candidate.absolute_layout, row.absolute_layout)
            for row in selected_candidates
        )
    else:
        distance = 0
    (
        gallery_eligible,
        complete,
        request_coverage_score,
        coverage_ratio,
        layout_score,
    ) = _candidate_rank_tuple(candidate)
    return (
        concept_is_new,
        anchor_is_new,
        macro_layout_is_new,
        visual_family_is_new,
        distance,
        gallery_eligible,
        complete,
        request_coverage_score,
        coverage_ratio,
        layout_score,
        -pool_index,
    )


def build_final_layout_variants_payload(
    *,
    candidates: list[dict[str, Any]],
    max_variants: int = 5,
) -> dict[str, Any]:
    selected = select_distinct_final_gallery_candidates(
        candidates=candidates,
        max_variants=max_variants,
    )
    payload_variants: list[dict[str, Any]] = []
    for index, raw in enumerate(selected, start=1):
        candidate = _coerce_candidate(raw)
        if candidate is None:
            continue
        payload_variants.append(
            {
                "variant_id": f"variant_{index}",
                "label": f"Option {index}",
                "source": candidate.source,
                "reason": candidate.reason,
                "layout_score": candidate.layout_score,
                "hard_valid": candidate.hard_valid,
                "complete": candidate.complete,
                "gallery_eligible": candidate.gallery_eligible,
                "coverage_ratio": candidate.coverage_ratio,
                "request_coverage_score": candidate.request_coverage_score,
                "missing_cluster_ids": deepcopy(candidate.missing_cluster_ids),
                "missing_hard_requested_object_types": deepcopy(
                    candidate.missing_hard_requested_object_types
                ),
                "notes": deepcopy(candidate.notes),
                "state_signature": candidate.state_signature,
                "macro_layout_signature": candidate.macro_layout_signature,
                "visual_family_signature": candidate.visual_family_signature,
                "concept_signature": candidate.concept_signature,
                "anchor_signature": candidate.anchor_signature,
                "gallery_selection_mode": str(raw.get("gallery_selection_mode") or ""),
                "absolute_layout": deepcopy(candidate.absolute_layout),
                "styled_result": deepcopy(candidate.styled_result)
                if isinstance(candidate.styled_result, dict)
                else None,
            }
        )
    return {
        "status": "OK",
        "selected_variant_id": payload_variants[0]["variant_id"]
        if payload_variants
        else None,
        "variants": payload_variants,
        "selection_summary": build_final_gallery_selection_summary(
            candidates=candidates,
            selected=selected,
            requested_count=max_variants,
        ),
    }


def build_final_gallery_selection_summary(
    *,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    requested_count: int,
) -> dict[str, Any]:
    candidate_rows = [
        candidate
        for raw in candidates
        if (candidate := _coerce_candidate(raw)) is not None
    ]
    selected_rows = [
        candidate
        for raw in selected
        if (candidate := _coerce_candidate(raw)) is not None
    ]
    candidate_macro_counts = _macro_signature_counts(candidate_rows)
    selected_macro_counts = _macro_signature_counts(selected_rows)
    candidate_visual_family_counts = _visual_family_signature_counts(candidate_rows)
    selected_visual_family_counts = _visual_family_signature_counts(selected_rows)
    candidate_macro_signatures = set(candidate_macro_counts)
    selected_macro_signatures = set(selected_macro_counts)
    candidate_visual_family_signatures = set(candidate_visual_family_counts)
    selected_visual_family_signatures = set(selected_visual_family_counts)
    planned_material_counts = _raw_signature_counts(
        candidates,
        "planned_material_macro_signature",
    )
    selected_planned_material_counts = _raw_signature_counts(
        selected,
        "planned_material_macro_signature",
    )
    actual_material_counts = _raw_actual_macro_signature_counts(candidates)
    selected_actual_material_counts = _raw_actual_macro_signature_counts(selected)
    hard_valid_candidate_count = sum(1 for row in candidate_rows if row.hard_valid)
    gallery_eligible_candidate_count = sum(
        1 for row in candidate_rows if row.gallery_eligible
    )
    eligible_candidate_count = sum(
        1 for row in candidate_rows if row.hard_valid and row.gallery_eligible
    )
    collapsed_macro_layouts = [
        {
            "macro_layout_signature": signature,
            "candidate_count": candidate_count,
            "returned_count": selected_macro_counts.get(signature, 0),
            "collapsed_count": candidate_count
            - selected_macro_counts.get(signature, 0),
        }
        for signature, candidate_count in sorted(candidate_macro_counts.items())
        if candidate_count > selected_macro_counts.get(signature, 0)
    ]
    collapsed_visual_families = [
        {
            "visual_family_signature": signature,
            "candidate_count": candidate_count,
            "returned_count": selected_visual_family_counts.get(signature, 0),
            "collapsed_count": candidate_count
            - selected_visual_family_counts.get(signature, 0),
        }
        for signature, candidate_count in sorted(candidate_visual_family_counts.items())
        if candidate_count > selected_visual_family_counts.get(signature, 0)
    ]
    fallback_near_distinct_count = sum(
        1
        for raw in selected
        if str(raw.get("gallery_selection_mode") or "") == "fallback_near_distinct"
    )
    macro_concept_collapsed_to_fallback_count = sum(
        1 for raw in candidates if bool(raw.get("macro_concept_collapsed_to_fallback"))
    )
    notes: list[str] = []
    requested = max(0, int(requested_count))
    if fallback_near_distinct_count:
        notes.append(
            "Final gallery backfilled near-distinct options after exhausting "
            "strongly distinct visual families."
        )
    material_macro_target = min(requested, 3) if requested >= 3 else requested
    diversity_degraded = (
        bool(selected_rows)
        and material_macro_target > 0
        and len(selected_macro_signatures) < material_macro_target
    )
    if diversity_degraded:
        notes.append(
            "Final gallery diversity is degraded because selected options collapse "
            "to too few material macro layouts."
        )
    if len(planned_material_counts) > len(actual_material_counts):
        notes.append(
            "Seed planned more material macro signatures than the solver realized; "
            "inspect planned_material_macro_signature versus actual_macro_layout_signature."
        )
    if macro_concept_collapsed_to_fallback_count:
        notes.append(
            "Some concepts missed their committed anchor target and collapsed to a "
            "previous fallback macro layout."
        )
    if not selected_rows and candidate_rows and eligible_candidate_count == 0:
        notes.append(
            "Final gallery returned no options because every candidate failed "
            "hard-valid/gallery eligibility gates."
        )
    elif len(selected_rows) < requested:
        if len(candidate_rows) <= len(selected_rows):
            notes.append(
                "Final gallery returned fewer options because fewer usable "
                "candidates were available than requested."
            )
        else:
            notes.append(
                "Final gallery returned fewer options because remaining candidates "
                "were too visually similar to selected options."
            )
    elif collapsed_visual_families:
        notes.append(
            "Final gallery preserved the requested count by selecting distinct "
            "visual families first, then backfilling when needed."
        )
    return {
        "requested_count": requested,
        "candidate_count": len(candidate_rows),
        "returned_count": len(selected_rows),
        "hard_valid_candidate_count": hard_valid_candidate_count,
        "gallery_eligible_candidate_count": gallery_eligible_candidate_count,
        "eligible_candidate_count": eligible_candidate_count,
        "top_quality_gate_reasons": _top_quality_gate_reasons(candidates),
        "macro_distinct_candidate_count": len(candidate_macro_signatures),
        "macro_distinct_returned_count": len(selected_macro_signatures),
        "planned_material_macro_distinct_candidate_count": len(planned_material_counts),
        "planned_material_macro_distinct_returned_count": len(
            selected_planned_material_counts
        ),
        "actual_macro_distinct_candidate_count": len(actual_material_counts),
        "actual_macro_distinct_returned_count": len(selected_actual_material_counts),
        "material_macro_target_count": material_macro_target,
        "diversity_degraded": diversity_degraded,
        "visual_family_distinct_candidate_count": len(
            candidate_visual_family_signatures
        ),
        "visual_family_distinct_returned_count": len(selected_visual_family_signatures),
        "fallback_near_distinct_count": fallback_near_distinct_count,
        "macro_concept_collapsed_to_fallback_count": (
            macro_concept_collapsed_to_fallback_count
        ),
        "planned_material_macro_signature_counts": planned_material_counts,
        "actual_macro_layout_signature_counts": actual_material_counts,
        "collapsed_macro_layouts": collapsed_macro_layouts,
        "collapsed_visual_families": collapsed_visual_families,
        "notes": notes,
    }


def _top_quality_gate_reasons(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw in candidates:
        for reason in _candidate_quality_gate_reasons(raw):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _candidate_quality_gate_reasons(raw: dict[str, Any]) -> list[str]:
    containers: list[dict[str, Any]] = [raw]
    for key in ("verify_summary", "solver_output"):
        value = raw.get(key)
        if isinstance(value, dict):
            containers.append(value)
    absolute_layout = raw.get("absolute_layout")
    if isinstance(absolute_layout, dict):
        containers.append(absolute_layout)
        verify = absolute_layout.get("verify_summary")
        if isinstance(verify, dict):
            containers.append(verify)

    reasons: list[str] = []
    for container in containers:
        value = container.get("quality_gate_reasons")
        if not isinstance(value, list):
            continue
        for item in value:
            reason = str(item or "").strip()
            if reason:
                reasons.append(reason)
    return reasons


def _macro_signature_counts(
    rows: list[_VariantCandidate],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not row.macro_layout_signature:
            continue
        counts[row.macro_layout_signature] = (
            counts.get(row.macro_layout_signature, 0) + 1
        )
    return counts


def _raw_signature_counts(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        signature = str(row.get(key) or "").strip()
        if not signature:
            continue
        counts[signature] = counts.get(signature, 0) + 1
    return counts


def _raw_actual_macro_signature_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = _raw_signature_counts(rows, "actual_macro_layout_signature")
    if counts:
        return counts
    candidate_rows = [
        candidate for row in rows if (candidate := _coerce_candidate(row)) is not None
    ]
    return _macro_signature_counts(candidate_rows)


def _visual_family_signature_counts(
    rows: list[_VariantCandidate],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not row.visual_family_signature:
            continue
        counts[row.visual_family_signature] = (
            counts.get(row.visual_family_signature, 0) + 1
        )
    return counts


def build_layout_variants_artifact(
    *,
    candidates: list[dict[str, Any]],
    max_variants: int = 5,
    **_: Any,
) -> dict[str, Any]:
    return build_final_layout_variants_payload(
        candidates=candidates,
        max_variants=max_variants,
    )


def select_top_variant_candidates(
    *,
    candidates: list[dict[str, Any]],
    max_variants: int,
) -> list[dict[str, Any]]:
    return select_distinct_final_gallery_candidates(
        candidates=candidates,
        max_variants=max_variants,
    )


def _coerce_candidate(raw: dict[str, Any]) -> _VariantCandidate | None:
    if not isinstance(raw, dict):
        return None
    absolute_layout = raw.get("absolute_layout")
    if not isinstance(absolute_layout, dict):
        return None
    styled_result = (
        raw.get("styled_result") if isinstance(raw.get("styled_result"), dict) else None
    )
    source = (
        str(raw.get("source") or "macro_cluster_solver").strip()
        or "macro_cluster_solver"
    )
    reason = (
        str(raw.get("reason") or "Anchor-first layout candidate.").strip()
        or "Anchor-first layout candidate."
    )
    complete = bool(raw.get("complete", absolute_layout.get("complete", False)))
    gallery_eligible = bool(
        raw.get("gallery_eligible", absolute_layout.get("gallery_eligible", False))
    )
    coverage_ratio = float(
        raw.get("coverage_ratio") or absolute_layout.get("coverage_ratio") or 0.0
    )
    missing_cluster_ids = raw.get("missing_cluster_ids")
    if not isinstance(missing_cluster_ids, list):
        missing_cluster_ids = (
            absolute_layout.get("missing_cluster_ids")
            if isinstance(absolute_layout.get("missing_cluster_ids"), list)
            else []
        )
    notes = [
        str(item).strip() for item in (raw.get("notes") or []) if str(item).strip()
    ]
    missing_hard_requested_object_types = _missing_hard_requested_solver_object_types(
        raw, absolute_layout
    )
    if missing_hard_requested_object_types:
        gallery_eligible = False
        complete = False
        notes.append(
            "Missing hard requested solver objects: "
            + ", ".join(missing_hard_requested_object_types)
        )
    relation_plan = (
        raw.get("relation_plan") if isinstance(raw.get("relation_plan"), dict) else {}
    )
    concept = raw.get("concept") if isinstance(raw.get("concept"), dict) else {}
    concept_signature = _concept_signature(
        raw, concept=concept, relation_plan=relation_plan
    )
    anchor_signature = _anchor_signature(concept=concept, relation_plan=relation_plan)
    macro_signature = _macro_layout_signature(absolute_layout)
    visual_family_signature = _visual_family_signature(absolute_layout)
    request_coverage_score = _request_coverage_score(raw, absolute_layout)
    return _VariantCandidate(
        absolute_layout=deepcopy(absolute_layout),
        styled_result=deepcopy(styled_result)
        if isinstance(styled_result, dict)
        else None,
        source=source,
        reason=reason,
        layout_score=int(raw.get("layout_score") or 0),
        hard_valid=bool(raw.get("hard_valid", absolute_layout.get("hard_valid", True))),
        complete=complete,
        gallery_eligible=gallery_eligible,
        coverage_ratio=coverage_ratio,
        request_coverage_score=request_coverage_score,
        missing_cluster_ids=[
            str(item).strip() for item in missing_cluster_ids if str(item).strip()
        ],
        missing_hard_requested_object_types=missing_hard_requested_object_types,
        notes=notes,
        concept_signature=concept_signature,
        anchor_signature=anchor_signature,
        macro_layout_signature=macro_signature,
        visual_family_signature=visual_family_signature,
        state_signature=_absolute_layout_signature(absolute_layout),
    )


def _candidate_rank_tuple(
    candidate: _VariantCandidate,
) -> tuple[int, int, int, float, int]:
    return (
        1 if candidate.gallery_eligible else 0,
        1 if candidate.complete else 0,
        int(candidate.request_coverage_score),
        float(candidate.coverage_ratio),
        int(candidate.layout_score),
    )


def _request_coverage_score(
    raw: dict[str, Any],
    absolute_layout: dict[str, Any],
) -> int:
    explicit = raw.get("request_coverage_score")
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    expected = _expected_request_object_types(raw)
    if not expected:
        return 0
    present = {
        canonical
        for item in _layout_objects(absolute_layout)
        if (canonical := _canonical_layout_object_type(item))
    }
    covered = expected & present
    return int(round(1000 * len(covered) / max(1, len(expected)))) + min(
        50, len(covered)
    )


def _missing_hard_requested_solver_object_types(
    raw: dict[str, Any],
    absolute_layout: dict[str, Any],
) -> list[str]:
    expected = _expected_hard_requested_solver_object_types(raw)
    if not expected:
        return []
    present = {
        canonical
        for item in _layout_objects(absolute_layout)
        if (canonical := _canonical_layout_object_type(item))
    }
    return sorted(expected - present)


def _expected_hard_requested_solver_object_types(raw: dict[str, Any]) -> set[str]:
    expected: set[str] = set()
    for root_key in ("tier_output", "merged_output", "cluster_output"):
        root = raw.get(root_key)
        if isinstance(root, dict):
            _collect_hard_requested_solver_object_types(root, expected)
    return expected


def _collect_hard_requested_solver_object_types(
    payload: dict[str, Any],
    expected: set[str],
) -> None:
    for decision in _iter_decision_rows(payload):
        if _row_is_hard_requested_solver_object(decision):
            object_type = _canonical_decision_object_type(decision)
            if object_type:
                expected.add(object_type)
    for request_contract in _request_contracts(payload):
        for item in request_contract.get("objects") or []:
            if not isinstance(item, dict):
                continue
            if not _row_is_hard_requested_solver_object(item):
                continue
            object_type = _canonical_expected_object_type(item)
            if object_type:
                expected.add(object_type)


def _row_is_hard_requested_solver_object(row: dict[str, Any]) -> bool:
    intent = (
        str(row.get("request_contract_intent") or row.get("intent") or "")
        .strip()
        .lower()
    )
    if intent not in _HARD_REQUEST_INTENTS:
        return False
    if (
        _positive_int(row.get("target_count")) <= 0
        and _positive_int(row.get("min_keep")) <= 0
        and _positive_int(row.get("quantity")) <= 0
    ):
        return False
    solver_participation = str(row.get("solver_participation") or "").strip().lower()
    if solver_participation in _POST_SOLVER_PARTICIPATIONS:
        return False
    if solver_participation:
        return solver_participation in _SOLVER_OBJECT_PARTICIPATIONS
    return bool(row.get("blocks_floor_area")) or _positive_int(row.get("min_keep")) > 0


def _expected_request_object_types(raw: dict[str, Any]) -> set[str]:
    expected: set[str] = set()
    for root_key in ("tier_output", "merged_output", "cluster_output"):
        root = raw.get(root_key)
        if isinstance(root, dict):
            _collect_expected_request_object_types(root, expected)
    return expected


def _collect_expected_request_object_types(
    payload: dict[str, Any],
    expected: set[str],
) -> None:
    for decision in _iter_decision_rows(payload):
        if _decision_counts_for_request_coverage(decision):
            object_type = _canonical_decision_object_type(decision)
            if object_type:
                expected.add(object_type)
    for request_contract in _request_contracts(payload):
        for item in request_contract.get("objects") or []:
            if not isinstance(item, dict):
                continue
            if not _request_contract_object_counts_for_coverage(item):
                continue
            object_type = _canonical_expected_object_type(item)
            if object_type:
                expected.add(object_type)


def _iter_decision_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decisions = payload.get("decisions")
    if isinstance(decisions, list):
        rows.extend(item for item in decisions if isinstance(item, dict))
    clusters = payload.get("clusters")
    if isinstance(clusters, list):
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            cluster_decisions = cluster.get("decisions")
            if isinstance(cluster_decisions, list):
                rows.extend(
                    item for item in cluster_decisions if isinstance(item, dict)
                )
            hints = cluster.get("tier_count_hints")
            if isinstance(hints, dict):
                object_hints = hints.get("object_hints")
                if isinstance(object_hints, list):
                    rows.extend(item for item in object_hints if isinstance(item, dict))
    return rows


def _request_contracts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    request_contract = payload.get("request_contract")
    if isinstance(request_contract, dict):
        contracts.append(request_contract)
    semantic_program = payload.get("semantic_layout_program")
    if isinstance(semantic_program, dict):
        nested = semantic_program.get("request_contract")
        if isinstance(nested, dict) and nested not in contracts:
            contracts.append(nested)
    return contracts


def _decision_counts_for_request_coverage(row: dict[str, Any]) -> bool:
    if (
        _positive_int(row.get("quantity")) <= 0
        and _positive_int(row.get("min_keep")) <= 0
    ):
        return False
    preserve_level = str(row.get("preserve_level") or "").strip().lower()
    role = str(row.get("role") or row.get("priority") or "").strip().lower()
    request_intent = (
        str(row.get("request_contract_intent") or row.get("intent") or "")
        .strip()
        .lower()
    )
    return (
        preserve_level in {"highest", "high"}
        or role in {"dominant_anchor", "support", "secondary_support"}
        or request_intent in {"must_keep", "target_if_viable"}
    )


def _request_contract_object_counts_for_coverage(row: dict[str, Any]) -> bool:
    if (
        _positive_int(row.get("target_count")) <= 0
        and _positive_int(row.get("min_keep")) <= 0
    ):
        return False
    intent = str(row.get("intent") or "").strip().lower()
    return intent in {"must_keep", "target_if_viable"}


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _canonical_decision_object_type(row: dict[str, Any]) -> str:
    raw = row.get("object_type") or row.get("category") or row.get("type") or ""
    return _normalize_object_type(raw)


def _canonical_expected_object_type(row: dict[str, Any]) -> str:
    raw = (
        row.get("fulfilled_by")
        or row.get("chosen_object_type")
        or row.get("object_type")
        or row.get("category")
        or row.get("type")
        or ""
    )
    return _normalize_object_type(raw)


def _canonical_layout_object_type(item: dict[str, Any]) -> str:
    raw = (
        item.get("object_type")
        or item.get("category")
        or item.get("type")
        or item.get("object_id")
        or item.get("instance_id")
        or ""
    )
    return _normalize_object_type(raw)


def _normalize_object_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    if text.endswith("_copy"):
        text = text[: -len("_copy")]
    parts = text.split("_")
    if parts and parts[-1].isdigit():
        text = "_".join(parts[:-1])
    return text.strip("_")


def _absolute_layout_signature(layout: dict[str, Any]) -> str:
    objects = _layout_objects(layout)
    rows: list[str] = []
    for item in objects:
        bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
        key = _layout_identity_key(item)
        if not key:
            continue
        rows.append(
            "|".join(
                [
                    key,
                    str(int(item.get("rotation_ccw") or 0)),
                    str(int(bbox.get("min_x") or 0)),
                    str(int(bbox.get("min_y") or 0)),
                    str(int(bbox.get("max_x") or 0)),
                    str(int(bbox.get("max_y") or 0)),
                ]
            )
        )
    rows.sort()
    return (
        "\n".join(rows)
        if rows
        else json.dumps(layout, ensure_ascii=True, sort_keys=True)
    )


def _macro_layout_signature(layout: dict[str, Any]) -> str:
    boxes = _cluster_boxes(layout)
    if not boxes:
        return ""
    room = layout.get("room") if isinstance(layout.get("room"), dict) else {}
    polygon = (
        room.get("polygon_ccw") if isinstance(room.get("polygon_ccw"), list) else []
    )
    xs: list[int] = []
    ys: list[int] = []
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
        normalized = (value - origin) / max(1, span)
        if normalized < (1.0 / 3.0):
            return 0
        if normalized < (2.0 / 3.0):
            return 1
        return 2

    rows: list[str] = []
    for cluster_id in sorted(boxes):
        box = boxes[cluster_id]
        center_x = (box["min_x"] + box["max_x"]) / 2.0
        center_y = (box["min_y"] + box["max_y"]) / 2.0
        rows.append(
            f"{cluster_id}:{_bucket(center_x, min_x, width)}"
            f"{_bucket(center_y, min_y, height)}:"
            f"{_macro_box_wall_zone(box, (min_x, min_y, max_x, max_y))}"
        )
    return "|".join(rows)


def _macro_box_wall_zone(
    box: dict[str, int],
    bounds: tuple[int, int, int, int],
) -> str:
    min_x, min_y, max_x, max_y = bounds
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    threshold_x = max(350, int(round(width * 0.08)))
    threshold_y = max(350, int(round(height * 0.08)))
    sides: list[str] = []
    if box["min_x"] - min_x <= threshold_x:
        sides.append("left")
    if max_x - box["max_x"] <= threshold_x:
        sides.append("right")
    if box["min_y"] - min_y <= threshold_y:
        sides.append("top")
    if max_y - box["max_y"] <= threshold_y:
        sides.append("bottom")
    if len(sides) >= 2:
        vertical = next((side for side in sides if side in {"top", "bottom"}), "")
        horizontal = next((side for side in sides if side in {"left", "right"}), "")
        if vertical and horizontal:
            return f"{vertical}_{horizontal}"
    if sides:
        return sides[0]
    return "center"


def _visual_family_signature(layout: dict[str, Any]) -> str:
    bounds = _room_bounds(layout)
    object_rows: list[str] = []
    anchor_rows: list[str] = []
    for item in _layout_objects(layout):
        object_id = str(item.get("object_id") or item.get("instance_id") or "")
        if "__reintroduced_" in object_id:
            continue
        bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else None
        visual_key = _visual_object_key(item)
        if not visual_key or bbox is None:
            continue
        object_rows.append(visual_key)
        if not _is_visual_anchor_object(item):
            continue
        box = {
            "min_x": int(bbox.get("min_x") or 0),
            "min_y": int(bbox.get("min_y") or 0),
            "max_x": int(bbox.get("max_x") or 0),
            "max_y": int(bbox.get("max_y") or 0),
        }
        rotation = _rotation_bucket(item.get("rotation_ccw"))
        anchor_rows.append(
            f"{visual_key}:{rotation}:{_visual_anchor_zone(box, bounds)}"
        )
    if not object_rows and not anchor_rows:
        return ""
    object_rows.sort()
    anchor_rows.sort()
    return f"objects={','.join(object_rows)}|anchors={','.join(anchor_rows)}"


def _room_bounds(layout: dict[str, Any]) -> tuple[int, int, int, int] | None:
    room = layout.get("room") if isinstance(layout.get("room"), dict) else {}
    polygon = (
        room.get("polygon_ccw") if isinstance(room.get("polygon_ccw"), list) else []
    )
    xs: list[int] = []
    ys: list[int] = []
    for point in polygon:
        if not isinstance(point, dict):
            continue
        xs.append(int(point.get("x") or 0))
        ys.append(int(point.get("y") or 0))
    if not xs or not ys:
        for item in _layout_objects(layout):
            bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else None
            if bbox is None:
                continue
            xs.extend([int(bbox.get("min_x") or 0), int(bbox.get("max_x") or 0)])
            ys.extend([int(bbox.get("min_y") or 0), int(bbox.get("max_y") or 0)])
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _visual_object_key(item: dict[str, Any]) -> str:
    key = _layout_identity_key(item).strip().lower()
    if key:
        return key.replace(" ", "_")
    object_type = str(item.get("object_type") or item.get("category") or "").strip()
    return object_type.lower().replace(" ", "_")


def _is_visual_anchor_object(item: dict[str, Any]) -> bool:
    role = str(item.get("role") or "").strip().lower()
    if "anchor" in role or "dominant" in role or "primary" in role:
        return True
    text = " ".join(
        str(item.get(key) or "").strip().lower()
        for key in (
            "cluster_id",
            "object_id",
            "instance_id",
            "object_type",
            "category",
            "type",
        )
    )
    support_tokens = (
        "nightstand",
        "side_table",
        "lamp",
        "rug",
        "curtain",
        "art",
        "bookshelf",
        "bookcase",
        "plant",
        "ottoman",
        "stool",
        "pouf",
        "shelf",
    )
    if any(token in text for token in support_tokens):
        return False
    anchor_tokens = (
        "bed",
        "bunk",
        "crib",
        "sofa",
        "sectional",
        "armchair",
        "wardrobe",
        "closet",
        "dresser",
        "cabinet",
        "storage",
        "desk",
        "dining",
        "table",
        "island",
        "counter",
        "vanity",
        "tv_console",
        "console",
    )
    return any(token in text for token in anchor_tokens)


def _rotation_bucket(value: Any) -> int:
    rotation = int(value or 0) % 360
    return int(round(rotation / 90.0) * 90) % 360


def _visual_anchor_zone(
    box: dict[str, int],
    bounds: tuple[int, int, int, int] | None,
) -> str:
    if bounds is None:
        return "unknown"
    min_x, min_y, max_x, max_y = bounds
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    center_x = (box["min_x"] + box["max_x"]) / 2.0
    center_y = (box["min_y"] + box["max_y"]) / 2.0
    distances = {
        "left": abs(box["min_x"] - min_x),
        "right": abs(max_x - box["max_x"]),
        "top": abs(box["min_y"] - min_y),
        "bottom": abs(max_y - box["max_y"]),
    }
    side, distance = min(distances.items(), key=lambda row: row[1])
    edge_threshold = max(250, int(min(width, height) * 0.12))
    close_horizontal = [
        side_name
        for side_name in ("left", "right")
        if distances[side_name] <= edge_threshold
    ]
    close_vertical = [
        side_name
        for side_name in ("top", "bottom")
        if distances[side_name] <= edge_threshold
    ]
    if close_horizontal and close_vertical:
        return f"corner:{close_vertical[0]}_{close_horizontal[0]}"
    if distance <= edge_threshold:
        if side in {"left", "right"}:
            band = _coarse_band(center_y, origin=min_y, span=height, divisions=2)
        else:
            band = _coarse_band(center_x, origin=min_x, span=width, divisions=2)
        return f"{side}:{band}"
    x_band = _coarse_band(center_x, origin=min_x, span=width, divisions=2)
    y_band = _coarse_band(center_y, origin=min_y, span=height, divisions=2)
    return f"interior:{x_band}{y_band}"


def _coarse_band(
    value: float,
    *,
    origin: int,
    span: int,
    divisions: int,
) -> int:
    normalized = (value - origin) / max(1, span)
    normalized = min(0.999, max(0.0, normalized))
    return int(normalized * max(1, divisions))


def _cluster_boxes(layout: dict[str, Any]) -> dict[str, dict[str, int]]:
    boxes: dict[str, dict[str, int]] = {}
    for item in _layout_objects(layout):
        object_id = str(item.get("object_id") or item.get("instance_id") or "")
        if "__reintroduced_" in object_id:
            continue
        cluster_id = str(item.get("cluster_id") or "").strip()
        bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else None
        if not cluster_id or bbox is None:
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


def _layout_distance_score(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_boxes = _object_boxes(left)
    right_boxes = _object_boxes(right)
    keys = set(left_boxes) | set(right_boxes)
    if not keys:
        return 0
    score = 0.0
    for key in keys:
        left_box = left_boxes.get(key)
        right_box = right_boxes.get(key)
        if left_box is None or right_box is None:
            score += 5000.0
            continue
        left_cx = (left_box["min_x"] + left_box["max_x"]) / 2.0
        left_cy = (left_box["min_y"] + left_box["max_y"]) / 2.0
        right_cx = (right_box["min_x"] + right_box["max_x"]) / 2.0
        right_cy = (right_box["min_y"] + right_box["max_y"]) / 2.0
        score += abs(left_cx - right_cx) + abs(left_cy - right_cy)
    return int(score)


def _is_near_duplicate_layout(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    distance_threshold_mm: int | None = None,
) -> bool:
    shared_cluster_count = len(set(_object_boxes(left)) & set(_object_boxes(right)))
    threshold = (
        max(350, shared_cluster_count * 140)
        if distance_threshold_mm is None
        else max(0, int(distance_threshold_mm))
    )
    return _layout_distance_score(left, right) < threshold


def _object_boxes(layout: dict[str, Any]) -> dict[str, dict[str, int]]:
    boxes: dict[str, dict[str, int]] = {}
    for item in _layout_objects(layout):
        key = _layout_identity_key(item)
        bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else None
        if not key or bbox is None:
            continue
        boxes[key] = {
            "min_x": int(bbox.get("min_x") or 0),
            "min_y": int(bbox.get("min_y") or 0),
            "max_x": int(bbox.get("max_x") or 0),
            "max_y": int(bbox.get("max_y") or 0),
        }
    return boxes


def _concept_signature(
    raw: dict[str, Any], *, concept: dict[str, Any], relation_plan: dict[str, Any]
) -> str:
    layout_intent = (
        relation_plan.get("layout_intent_profile")
        if isinstance(relation_plan.get("layout_intent_profile"), dict)
        else {}
    )
    parts = [
        str(raw.get("source") or "").strip(),
        str(
            concept.get("concept_family") or relation_plan.get("concept_family") or ""
        ).strip(),
        str(layout_intent.get("focus_mode") or "").strip(),
        str(layout_intent.get("primary_cluster_id") or "").strip(),
        str(layout_intent.get("secondary_cluster_id") or "").strip(),
    ]
    return "::".join(part for part in parts if part)


def _anchor_signature(*, concept: dict[str, Any], relation_plan: dict[str, Any]) -> str:
    hints = (
        relation_plan.get("anchor_layout_hints_by_cluster")
        if isinstance(relation_plan.get("anchor_layout_hints_by_cluster"), dict)
        else {}
    )
    if not hints and isinstance(concept.get("anchor_layout_hints_by_cluster"), dict):
        hints = concept["anchor_layout_hints_by_cluster"]
    rows: list[str] = []
    for cluster_id in sorted(hints):
        row = hints.get(cluster_id)
        if not isinstance(row, dict):
            continue
        dominant = str(
            row.get("dominant_anchor_object_id") or row.get("dominant_anchor_id") or ""
        ).strip()
        zone_assignment = str(row.get("zone_assignment") or "").strip()
        preferred = row.get("preferred_local_families")
        if isinstance(preferred, list):
            preferred_text = ",".join(
                str(item).strip() for item in preferred if str(item).strip()
            )
        else:
            preferred_text = ""
        rows.append(f"{cluster_id}:{dominant}:{zone_assignment}:{preferred_text}")
    return "|".join(rows)


def _layout_objects(layout: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("objects", "object_placements", "placements"):
        value = layout.get(key)
        if isinstance(value, list):
            return [deepcopy(row) for row in value if isinstance(row, dict)]
    return []


def _layout_identity_key(item: dict[str, Any]) -> str:
    object_id = str(item.get("object_id") or item.get("instance_id") or "").strip()
    if object_id:
        return object_id
    cluster_id = str(item.get("cluster_id") or "").strip()
    if cluster_id:
        object_type = str(item.get("object_type") or "").strip()
        return f"{cluster_id}:{object_type}" if object_type else cluster_id
    return ""
