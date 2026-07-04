"""
Object-level bedroom/room layout solver.

What this solver does
---------------------
- Builds anchor-first object layouts directly from merged cluster programs.
- Places support objects around committed anchors.
- Verifies geometry, protected regions, orientation contracts, and interaction
  quality before publishing the absolute layout.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM
from layout.room_profiles.registry import (
    is_profile_storage_object,
)
from layout.semantic_roles import is_seat_like, is_work_surface_like
from layout.variant_family import FALLBACK_GENERIC_VARIANT_FAMILY
from layout.variant_family import (
    SEMANTIC_VARIANT_FAMILIES as CORE_SEMANTIC_VARIANT_FAMILIES,
)

_CLUSTER_WALL_PREFER_TAGS = frozenset(
    {"wall", "long_wall", "short_wall", "recess_or_edge"}
)
_CLUSTER_CENTER_PREFER_TAGS = frozenset({"center", "near_center"})
_CLUSTER_CENTER_AVOID_TAGS = frozenset({"center"})
_CLUSTER_WINDOW_AVOID_TAGS = frozenset({"window_blocking", "window_clearance"})
_CLUSTER_ENTRY_AVOID_TAGS = frozenset({"entry_blocking", "door_swing", "main_path"})
_CLUSTER_WINDOW_PREFER_TAGS = frozenset({"window_side"})
_CLUSTER_ENTRY_PREFER_TAGS = frozenset({"entry_side", "near_entry"})
_WALL_ANCHOR_TOKENS = ("wall", "top", "bottom", "left", "right", "edge", "recess")
_CENTER_ANCHOR_TOKENS = ("center", "quadrant")
_WINDOW_ANCHOR_TOKENS = ("window",)
_ENTRY_ANCHOR_TOKENS = ("entry", "door")
_WINDOW_TREATMENT_OBJECT_TOKENS = frozenset(
    {
        "blind",
        "blinds",
        "curtain",
        "curtain_rod",
        "curtains",
        "drape",
        "drapes",
        "drapery",
        "drapery_rod",
        "shade",
        "shades",
        "sheer",
        "valance",
        "window_covering",
        "window_treatment",
    }
)
_VISUAL_STORAGE_OBJECT_TOKENS = frozenset(
    {
        "bookcase",
        "bookshelf",
        "cabinet",
        "dresser",
        "media_shelf",
        "shelf",
        "storage",
        "wardrobe",
    }
)

DEFAULT_WALL_THICKNESS_MM = 100
DEFAULT_MAX_DEGRADATION_ROUNDS = 3
OBJECT_LEVEL_MAX_ANCHOR_CANDIDATES_PER_CLUSTER = 36
OBJECT_LEVEL_MAX_SUPPORT_SOLUTIONS_PER_ANCHOR = 16
OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS = 24
OBJECT_LEVEL_MAX_SUPPORT_SLOT_CANDIDATES = 36
OBJECT_LEVEL_GEOMETRY_REPAIR_MAX_CANDIDATES = 120
OBJECT_LEVEL_GEOMETRY_REPAIR_LOCAL_RADIUS_MM = 1800
OBJECT_LEVEL_REPAIR_BATCH_SIZE = OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS * 2
OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT = 0.8
OBJECT_LEVEL_VIEW_TARGET_BED_VIEWER_MIN_DOT = 0.65
OBJECT_LEVEL_VIEW_TARGET_SEATED_VIEWER_MIN_DOT = 0.75
OBJECT_LEVEL_VIEW_TARGET_TARGET_MIN_DOT = 0.75
OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MIN_DISTANCE_MM = 1800.0
OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MAX_DISTANCE_MM = 4500.0
OBJECT_LEVEL_ENTRY_SUPPORT_PREFERRED_DISTANCE_MM = 1500.0
OBJECT_LEVEL_SECONDARY_STORAGE_WALL_FIT_BONUS = 420.0
OBJECT_LEVEL_SECONDARY_STORAGE_PRIMARY_PROXIMITY_PENALTY = 520.0
OBJECT_LEVEL_SUPPORT_ALIGNMENT_FACTORS = (-0.35, 0.0, 0.35)
OBJECT_LEVEL_WALL_CONTACT_TOLERANCE_MM = 1
OBJECT_LEVEL_NEAR_WALL_FRONT_ACCESS_TOLERANCE_MM = 150
OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT = 0.45
OBJECT_LEVEL_FRONT_ACCESS_DEPTH_MM = 750
OBJECT_LEVEL_VIEW_CORRIDOR_WIDTH_MM = 900
OBJECT_LEVEL_BLOCKING_PROTECTED_PRIORITIES = frozenset({"critical", "high"})
OBJECT_LEVEL_BLOCKING_PROTECTED_ENFORCEMENT = frozenset({"hard", "hard_soft"})
OBJECT_LEVEL_BLOCKING_PROTECTED_SEVERITIES = frozenset({"blocking", "critical"})
OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_SOFT_MAX_MM = 1400
OBJECT_LEVEL_VISUAL_STORAGE_DAYLIGHT_SOFT_MAX_MM = 1200
OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_MAX_OVERLAP_RATIO = 0.35
OBJECT_LEVEL_SECONDARY_DAYLIGHT_MAX_OVERLAP_RATIO = 0.35
OBJECT_LEVEL_HARD_SOFT_ANCHOR_REJECT_RATIO = 0.16

QUALITY_WEIGHTS = {
    "functionality": 0.32,
    "naturalness": 0.24,
    "semantic": 0.24,
    "spatial": 0.20,
}

PUBLISHABLE_BLOCKING_QUALITY_REASONS = frozenset(
    {
        "critical_orientation_penalty_too_high",
        "focal_pair_penalty_too_high",
        "critical_item_penalty_too_high",
    }
)

SEMANTIC_VARIANT_FAMILIES = CORE_SEMANTIC_VARIANT_FAMILIES | frozenset(
    {FALLBACK_GENERIC_VARIANT_FAMILY}
)

STRICT_ROLE_KINDS = frozenset({"focal", "kitchen", "media", "work", "workflow"})
VIEW_TARGET_RELATIONS = frozenset({"view_target", "viewer_target", "bed_views_media"})
MUTUAL_FACE_RELATIONS = frozenset({"", "face_each_other", "mutual_face"})
VIEW_TARGET_BED_VIEWER_TYPES = frozenset({"bed", "lounge_chair"})


@dataclass(frozen=True)
class MacroClusterSolver:
    max_rounds: int = DEFAULT_MAX_DEGRADATION_ROUNDS
    num_workers: int = 8

    def generate_object_layout(
        self,
        *,
        room_model_json: dict[str, Any],
        merged_clusters_json: dict[str, Any],
        relation_plan_json: dict[str, Any] | None,
        cluster_constraints_json: dict[str, Any] | None = None,
        grid_mm: int = GLOBAL_LAYOUT_GRID_MM,
    ) -> dict[str, Any]:
        return solve_object_level_layout(
            room_model=room_model_json,
            merged_clusters=merged_clusters_json,
            relation_plan=relation_plan_json,
            cluster_constraints=cluster_constraints_json,
            grid_mm=grid_mm,
            max_rounds=self.max_rounds,
        )


def _concept_from_relation_plan(
    relation_plan: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(relation_plan, Mapping):
        return {}
    concept = relation_plan.get("macro_concept")
    return concept if isinstance(concept, Mapping) else {}


def _concept_id_from_relation_plan(relation_plan: Mapping[str, Any] | None) -> str:
    concept = _concept_from_relation_plan(relation_plan)
    for key in ("concept_id", "id", "concept_family"):
        value = str(concept.get(key) or "").strip()
        if value:
            return value
    if isinstance(relation_plan, Mapping):
        value = str(relation_plan.get("concept_id") or "").strip()
        if value:
            return value
    return "concept_01"


def _normalized_policy_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _placement_behavior_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _placement_behavior_value(
    placement_behavior: Mapping[str, Any] | None,
    key: str,
) -> str:
    if not isinstance(placement_behavior, Mapping):
        return ""
    return _normalized_policy_token(placement_behavior.get(key))


def _placement_behavior_from_rows(
    *rows: Mapping[str, Any] | object,
) -> Mapping[str, Any]:
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        placement_behavior = _placement_behavior_mapping(row.get("placement_behavior"))
        if placement_behavior:
            return placement_behavior
    return {}


def _placement_behavior_for_cluster(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> Mapping[str, Any]:
    if not isinstance(relation_plan, Mapping):
        return {}
    for section_name in (
        "anchor_layout_hints_by_cluster",
        "anchor_region_preferences_by_cluster",
    ):
        section = relation_plan.get(section_name)
        if not isinstance(section, Mapping):
            continue
        row = section.get(cluster_id)
        placement_behavior = _placement_behavior_from_rows(row)
        if placement_behavior:
            return placement_behavior
    concept = _concept_from_relation_plan(relation_plan)
    for row in concept.get("cluster_zone_plan") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        return _placement_behavior_from_rows(row)
    return {}


def _placement_behavior_orientation_intents(
    placement_behavior: Mapping[str, Any] | None,
) -> set[str]:
    intents: set[str] = set()
    wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
    front_space = _placement_behavior_value(placement_behavior, "front_space")
    if wall_backing in {"preferred", "required"}:
        intents.add("back_to_wall")
    if wall_backing == "required":
        intents.add("axis_parallel_wall")
    if front_space in {"preferred", "required"}:
        intents.update({"access_to_open_space", "front_to_open_space"})
    return intents


def _normalized_text_set(values: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(values, list):
        return out
    for item in values:
        text = str(item or "").strip().lower()
        if text:
            out.add(text)
    return out


def _string_sequence(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return []
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _dedupe_string_sequence(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _extract_layout_primary_cluster_id(
    relation_plan: dict[str, Any] | None,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    layout_intent = relation_plan.get("layout_intent_profile")
    if not isinstance(layout_intent, dict):
        return None
    cluster_id = str(layout_intent.get("primary_cluster_id") or "").strip()
    return cluster_id or None


def _extract_layout_secondary_cluster_id(
    relation_plan: dict[str, Any] | None,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    layout_intent = relation_plan.get("layout_intent_profile")
    if not isinstance(layout_intent, dict):
        return None
    cluster_id = str(layout_intent.get("secondary_cluster_id") or "").strip()
    return cluster_id or None


def _normalized_cluster_pair(a: str, b: str) -> tuple[str, str] | None:
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right or left == right:
        return None
    return tuple(sorted((left, right)))


def _cluster_face_target_id(
    relation_plan: dict[str, Any] | None,
    cluster_id: str,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    target: str | None = None
    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        intents = _normalized_text_set(row.get("intents"))
        if "face_cluster" not in intents:
            continue
        target = str(row.get("target_cluster_id") or "").strip() or None
        if target is not None:
            return target
    return None


def _cluster_entry_relation_for_plan(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> str:
    if not isinstance(relation_plan, Mapping) or not cluster_id:
        return ""
    for section_name in (
        "anchor_layout_hints_by_cluster",
        "anchor_region_preferences_by_cluster",
    ):
        section = relation_plan.get(section_name)
        if not isinstance(section, Mapping):
            continue
        row = section.get(cluster_id)
        if not isinstance(row, Mapping):
            continue
        relation = str(row.get("entry_relation") or "").strip()
        if relation:
            return relation
    macro_concept = relation_plan.get("macro_concept")
    if not isinstance(macro_concept, Mapping):
        return ""
    zone_plan = macro_concept.get("cluster_zone_plan")
    if not isinstance(zone_plan, Sequence) or isinstance(zone_plan, str):
        return ""
    for row in zone_plan:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        return str(row.get("entry_relation") or "").strip()
    return ""


def _anchor_kind_contains(anchor_kind: str, tokens: Sequence[str]) -> bool:
    lowered = anchor_kind.strip().lower()
    return any(token in lowered for token in tokens)


def _anchor_kind_is_entry(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _ENTRY_ANCHOR_TOKENS)


# ---------------------------------------------------------------------------
# Object-level anchor-first solver
# ---------------------------------------------------------------------------


def solve_object_level_layout(
    *,
    room_model: dict[str, Any],
    merged_clusters: dict[str, Any],
    relation_plan: dict[str, Any] | None,
    cluster_constraints: dict[str, Any] | None = None,
    grid_mm: int = GLOBAL_LAYOUT_GRID_MM,
    max_rounds: int = 3,
) -> dict[str, Any]:
    room_bbox = _object_solver_room_bbox(room_model)
    if room_bbox[2] <= room_bbox[0] or room_bbox[3] <= room_bbox[1]:
        return {
            "status": "UNSAT",
            "solver_kind": "object_level_anchor_first",
            "notes": [
                "Room bounding box could not be resolved for object-level solving."
            ],
        }

    world = _build_object_solver_world(
        room_model=room_model,
        merged_clusters=merged_clusters,
        relation_plan=relation_plan,
        cluster_constraints=cluster_constraints,
        grid_mm=grid_mm,
    )
    anchor_order = world["anchor_cluster_order"]
    anchor_candidates_by_cluster = {
        cluster_id: _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"][cluster_id],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=grid_mm,
        )
        for cluster_id in anchor_order
    }
    anchor_preflight_by_cluster = (
        world.get("_anchor_candidate_preflight_by_cluster")
        if isinstance(world.get("_anchor_candidate_preflight_by_cluster"), Mapping)
        else {}
    )
    anchor_reject_reason_counts = (
        world.get("_anchor_reject_reason_counts")
        if isinstance(world.get("_anchor_reject_reason_counts"), Mapping)
        else {}
    )
    anchor_wall_options_by_cluster = (
        world.get("_anchor_wall_options_by_cluster")
        if isinstance(world.get("_anchor_wall_options_by_cluster"), Mapping)
        else {}
    )
    anchor_candidate_diagnostics_by_cluster = {
        cluster_id: _anchor_candidate_generation_diagnostics(
            rows,
            preflight=anchor_preflight_by_cluster.get(cluster_id)
            if isinstance(anchor_preflight_by_cluster, Mapping)
            else {},
            reject_reason_counts=anchor_reject_reason_counts.get(cluster_id)
            if isinstance(anchor_reject_reason_counts, Mapping)
            else {},
            wall_options=anchor_wall_options_by_cluster.get(cluster_id)
            if isinstance(anchor_wall_options_by_cluster, Mapping)
            else {},
        )
        for cluster_id, rows in anchor_candidates_by_cluster.items()
    }
    if any(
        not rows
        for cluster_id, rows in anchor_candidates_by_cluster.items()
        if not _cluster_is_solver_trial_optional(world["clusters_by_id"][cluster_id])
    ):
        offending = [
            cluster_id
            for cluster_id, rows in anchor_candidates_by_cluster.items()
            if not rows
            and not _cluster_is_solver_trial_optional(
                world["clusters_by_id"][cluster_id]
            )
        ]
        return {
            "status": "UNSAT",
            "solver_kind": "object_level_anchor_first",
            "offending_clusters": offending,
            "solver_degradation_report": _macro_cluster_solver_degradation_report(
                raw_solution_count=0,
                eligible_solutions=[],
                partial_solutions=[],
                anchor_candidate_counts={
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                },
                anchor_order=anchor_order,
                status="UNSAT",
                anchor_wall_options_by_cluster=anchor_wall_options_by_cluster,
            ),
            "notes": [
                "No anchor pose candidates could be generated for one or more core clusters."
            ],
            "solver_debug": {
                "anchor_candidate_counts": {
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                },
                "anchor_candidate_diagnostics_by_cluster": (
                    anchor_candidate_diagnostics_by_cluster
                ),
                "anchor_wall_options_by_cluster": anchor_wall_options_by_cluster,
                "auto_window_forbidden_regions_by_cluster": (
                    world.get("auto_window_forbidden_regions_by_cluster") or {}
                ),
                "auto_window_forbidden_regions_by_object": (
                    world.get("auto_window_forbidden_regions_by_object") or {}
                ),
                "geometry": _object_solver_geometry_diagnostics(world),
                "wall_inner_face_reject_reason_counts": (
                    _wall_inner_face_reject_reason_counts(
                        anchor_candidate_diagnostics_by_cluster
                    )
                ),
            },
        }

    anchor_solutions = list(
        _search_anchor_solutions(
            anchor_order=anchor_order,
            anchor_candidates_by_cluster=anchor_candidates_by_cluster,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            max_solutions=max(8, int(max_rounds) * 4),
        )
    )
    if not anchor_solutions:
        anchor_pair_compatibility = _anchor_pair_compatibility_report(
            anchor_order=anchor_order,
            anchor_candidates_by_cluster=anchor_candidates_by_cluster,
        )
        pairwise_incompatible_anchor_pairs = [
            row
            for row in anchor_pair_compatibility
            if int(row.get("candidate_pairs") or 0) > 0
            and int(row.get("non_overlap_pairs") or 0) <= 0
        ]
        fallback_retry = _retry_object_level_layout_with_committed_fallback(
            room_model=room_model,
            merged_clusters=merged_clusters,
            relation_plan=relation_plan,
            cluster_constraints=cluster_constraints,
            grid_mm=grid_mm,
            max_rounds=max_rounds,
            anchor_candidate_diagnostics_by_cluster=(
                anchor_candidate_diagnostics_by_cluster
            ),
            strict_failure_reason="no_compatible_anchor_set",
        )
        if fallback_retry is not None:
            return fallback_retry
        return {
            "status": "UNSAT",
            "solver_kind": "object_level_anchor_first",
            "failure_kind": "no_compatible_anchor_set",
            "offending_clusters": list(anchor_order),
            "solver_degradation_report": _macro_cluster_solver_degradation_report(
                raw_solution_count=0,
                eligible_solutions=[],
                partial_solutions=[],
                anchor_candidate_counts={
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                },
                anchor_order=anchor_order,
                status="UNSAT",
                failure_kind="no_compatible_anchor_set",
                anchor_pair_compatibility=anchor_pair_compatibility,
                pairwise_incompatible_anchor_pairs=(pairwise_incompatible_anchor_pairs),
                anchor_wall_options_by_cluster=anchor_wall_options_by_cluster,
            ),
            "notes": [
                "Anchor candidates were generated, but no compatible non-overlapping anchor set could be assembled."
            ],
            "solver_debug": {
                "failure_kind": "no_compatible_anchor_set",
                "anchor_candidate_counts": {
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                },
                "anchor_pair_compatibility": anchor_pair_compatibility,
                "pairwise_incompatible_anchor_pairs": (
                    pairwise_incompatible_anchor_pairs
                ),
                "anchor_candidate_diagnostics_by_cluster": (
                    anchor_candidate_diagnostics_by_cluster
                ),
                "anchor_wall_options_by_cluster": anchor_wall_options_by_cluster,
                "auto_window_forbidden_regions_by_cluster": (
                    world.get("auto_window_forbidden_regions_by_cluster") or {}
                ),
                "auto_window_forbidden_regions_by_object": (
                    world.get("auto_window_forbidden_regions_by_object") or {}
                ),
                "geometry": _object_solver_geometry_diagnostics(world),
                "wall_inner_face_reject_reason_counts": (
                    _wall_inner_face_reject_reason_counts(
                        anchor_candidate_diagnostics_by_cluster
                    )
                ),
                "anchor_order": list(anchor_order),
            },
        }

    raw_solution_pool: list[dict[str, Any]] = []
    for solution in anchor_solutions:
        support_results = _place_support_objects_for_solution(
            solution=solution,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            grid_mm=grid_mm,
            max_solutions=OBJECT_LEVEL_MAX_SUPPORT_SOLUTIONS_PER_ANCHOR,
        )
        if not support_results:
            continue
        for support_result in support_results:
            candidate_solution = {**solution, **support_result}
            candidate_solution["pre_repair_score"] = _object_pre_repair_score(
                candidate_solution
            )
            candidate_solution["pre_repair_signature"] = (
                _object_level_solution_signature(candidate_solution)
            )
            raw_solution_pool.append(candidate_solution)

    raw_solution_pool.sort(key=_object_pre_repair_rank_key, reverse=True)
    solution_pool: list[dict[str, Any]] = []
    partial_solution_pool: list[dict[str, Any]] = []
    repair_batch_size = max(
        OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS,
        OBJECT_LEVEL_REPAIR_BATCH_SIZE,
    )
    for repaired_index, candidate_solution in enumerate(raw_solution_pool, start=1):
        repair_summary = _repair_object_level_solution_geometry(
            solution=candidate_solution,
            world=world,
            grid_mm=grid_mm,
        )
        if repair_summary is None:
            continue
        candidate_solution["placed_objects"] = repair_summary["placed_objects"]
        candidate_solution["geometry_repair"] = repair_summary["summary"]
        verify = _verify_object_level_solution(
            solution=candidate_solution,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
        )
        _apply_object_level_geometry_repair_penalty(
            verify, candidate_solution["geometry_repair"]
        )
        if not bool(verify.get("geometry_valid")):
            continue
        candidate_solution["verify"] = verify
        candidate_solution["score"] = _object_solution_score(
            candidate_solution, relation_plan
        )
        candidate_solution["signature"] = _object_level_solution_signature(
            candidate_solution
        )
        if bool(verify.get("hard_valid")) and bool(verify.get("gallery_eligible")):
            solution_pool.append(candidate_solution)
        else:
            partial_solution_pool.append(candidate_solution)
        if (
            repaired_index % repair_batch_size == 0
            and _object_repair_pool_is_sufficient(solution_pool)
        ):
            break

    ranked_pool = _rank_object_level_solution_pool(solution_pool)[
        :OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS
    ]
    partial_ranked_pool = _rank_object_level_solution_pool(partial_solution_pool)[
        :OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS
    ]
    best_solution = ranked_pool[0] if ranked_pool else None
    if best_solution is None:
        fallback_retry = _retry_object_level_layout_with_committed_fallback(
            room_model=room_model,
            merged_clusters=merged_clusters,
            relation_plan=relation_plan,
            cluster_constraints=cluster_constraints,
            grid_mm=grid_mm,
            max_rounds=max_rounds,
            anchor_candidate_diagnostics_by_cluster=(
                anchor_candidate_diagnostics_by_cluster
            ),
            strict_failure_reason="no_publishable_committed_solution",
        )
        if fallback_retry is not None:
            return fallback_retry
        partial_solutions = [
            _build_object_level_solution_payload(
                solution=item,
                world=world,
                room_model=room_model,
                relation_plan=relation_plan,
                solution_index=index,
            )
            for index, item in enumerate(partial_ranked_pool[:3], start=1)
        ]
        best_partial = partial_solutions[0] if partial_solutions else None
        best_partial_verify = (
            best_partial.get("verify_summary")
            if isinstance(best_partial, Mapping)
            and isinstance(best_partial.get("verify_summary"), Mapping)
            else {}
        )
        return {
            "status": "UNSAT",
            "solver_kind": "object_level_anchor_first",
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan),
            "hard_valid": False,
            "geometry_valid": bool(best_partial_verify.get("geometry_valid")),
            "acceptable_valid": False,
            "complete": bool(best_partial_verify.get("complete")),
            "gallery_eligible": False,
            "coverage_ratio": float(best_partial_verify.get("coverage_ratio") or 0.0),
            "offending_clusters": list(
                best_partial_verify.get("offending_clusters") or anchor_order
            ),
            "best_partial": best_partial,
            "partial_solutions": partial_solutions,
            "solver_degradation_report": _macro_cluster_solver_degradation_report(
                raw_solution_count=len(raw_solution_pool),
                eligible_solutions=ranked_pool,
                partial_solutions=partial_ranked_pool,
                anchor_candidate_counts={
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                },
                anchor_order=anchor_order,
                status="UNSAT",
                anchor_wall_options_by_cluster=anchor_wall_options_by_cluster,
            ),
            "notes": [
                "Object-level solver produced no hard-valid/gallery-eligible layouts.",
                "Geometry-valid partial layouts are stored only under diagnostics and must not enter stylist/gallery.",
            ],
            "solver_debug": {
                "anchor_candidate_counts": {
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                },
                "anchor_candidate_diagnostics_by_cluster": (
                    anchor_candidate_diagnostics_by_cluster
                ),
                "anchor_wall_options_by_cluster": anchor_wall_options_by_cluster,
                "auto_window_forbidden_regions_by_cluster": (
                    world.get("auto_window_forbidden_regions_by_cluster") or {}
                ),
                "auto_window_forbidden_regions_by_object": (
                    world.get("auto_window_forbidden_regions_by_object") or {}
                ),
                "geometry": _object_solver_geometry_diagnostics(world),
                "wall_inner_face_reject_reason_counts": (
                    _wall_inner_face_reject_reason_counts(
                        anchor_candidate_diagnostics_by_cluster
                    )
                ),
                "anchor_order": anchor_order,
                "raw_solution_count": len(raw_solution_pool),
                "candidate_solution_count": len(solution_pool),
                "partial_candidate_solution_count": len(partial_solution_pool),
                "ranked_solution_count": 0,
            },
        }

    absolute_layout = _build_absolute_layout_from_object_solution(
        solution=best_solution,
        world=world,
        room_model=room_model,
        relation_plan=relation_plan,
    )
    ranked_solutions = [
        _build_object_level_solution_payload(
            solution=item,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            solution_index=index,
        )
        for index, item in enumerate(ranked_pool, start=1)
    ]
    return {
        "status": "OK",
        "solver_kind": "object_level_anchor_first",
        "selected_concept_id": _concept_id_from_relation_plan(relation_plan),
        "absolute_layout": absolute_layout,
        "solutions": ranked_solutions,
        "hard_valid": bool(best_solution["verify"].get("hard_valid")),
        "geometry_valid": bool(best_solution["verify"].get("geometry_valid")),
        "acceptable_valid": bool(best_solution["verify"].get("gallery_eligible")),
        "complete": bool(best_solution["verify"].get("complete")),
        "gallery_eligible": bool(best_solution["verify"].get("gallery_eligible")),
        "coverage_ratio": float(best_solution["verify"].get("coverage_ratio") or 0.0),
        "offending_clusters": list(
            best_solution["verify"].get("offending_clusters") or []
        ),
        "dropped_inventory_by_cluster": deepcopy(
            best_solution.get("dropped_inventory_by_cluster") or {}
        ),
        "solver_degradation_report": _macro_cluster_solver_degradation_report(
            raw_solution_count=len(raw_solution_pool),
            eligible_solutions=ranked_pool,
            partial_solutions=partial_ranked_pool,
            anchor_candidate_counts={
                k: len(v) for k, v in anchor_candidates_by_cluster.items()
            },
            anchor_order=anchor_order,
            status="OK",
            anchor_wall_options_by_cluster=anchor_wall_options_by_cluster,
        ),
        "cluster_transforms": [],
        "selected_variants": [],
        "verify_summary": deepcopy(best_solution["verify"]),
        "solver_debug": {
            "anchor_candidate_counts": {
                k: len(v) for k, v in anchor_candidates_by_cluster.items()
            },
            "anchor_candidate_diagnostics_by_cluster": (
                anchor_candidate_diagnostics_by_cluster
            ),
            "anchor_wall_options_by_cluster": anchor_wall_options_by_cluster,
            "auto_window_forbidden_regions_by_cluster": (
                world.get("auto_window_forbidden_regions_by_cluster") or {}
            ),
            "auto_window_forbidden_regions_by_object": (
                world.get("auto_window_forbidden_regions_by_object") or {}
            ),
            "geometry": _object_solver_geometry_diagnostics(world),
            "wall_inner_face_reject_reason_counts": (
                _wall_inner_face_reject_reason_counts(
                    anchor_candidate_diagnostics_by_cluster
                )
            ),
            "anchor_order": anchor_order,
            "object_count": len(best_solution.get("placed_objects") or []),
            "candidate_solution_count": len(solution_pool),
            "partial_candidate_solution_count": len(partial_solution_pool),
            "ranked_solution_count": len(ranked_solutions),
        },
        "notes": [
            "Solved directly at object level with anchor-first placement.",
            "Local support placement now enumerates multiple non-overlapping arrangements before ranking them.",
        ],
    }


def _retry_object_level_layout_with_committed_fallback(
    *,
    room_model: dict[str, Any],
    merged_clusters: dict[str, Any],
    relation_plan: dict[str, Any] | None,
    cluster_constraints: dict[str, Any] | None,
    grid_mm: int,
    max_rounds: int,
    anchor_candidate_diagnostics_by_cluster: Mapping[str, Any],
    strict_failure_reason: str,
) -> dict[str, Any] | None:
    if _relation_plan_allows_committed_fallback(relation_plan):
        return None
    if not _committed_anchor_fallback_retry_is_needed(
        anchor_candidate_diagnostics_by_cluster
    ):
        return None
    retry_plan = deepcopy(relation_plan) if isinstance(relation_plan, dict) else {}
    retry_plan["_allow_committed_fallback_after_strict_fail"] = True
    retry_result = solve_object_level_layout(
        room_model=room_model,
        merged_clusters=merged_clusters,
        relation_plan=retry_plan,
        cluster_constraints=cluster_constraints,
        grid_mm=grid_mm,
        max_rounds=max_rounds,
    )
    if str(retry_result.get("status") or "").upper() != "OK":
        return None
    retry_result["committed_anchor_fallback_retry_applied"] = True
    retry_result["committed_anchor_strict_failure_reason"] = strict_failure_reason
    retry_result["notes"] = [
        *[str(note) for note in retry_result.get("notes", []) if str(note).strip()],
        (
            "Strict committed material macro solve failed; broad preferred regions "
            "were enabled for this retry and are marked in candidate diagnostics."
        ),
    ]
    solver_debug = retry_result.setdefault("solver_debug", {})
    if isinstance(solver_debug, dict):
        solver_debug["committed_anchor_fallback_retry_applied"] = True
        solver_debug["committed_anchor_strict_failure_reason"] = strict_failure_reason
        solver_debug["strict_anchor_candidate_diagnostics_by_cluster"] = deepcopy(
            dict(anchor_candidate_diagnostics_by_cluster)
        )
    return retry_result


def _committed_anchor_fallback_retry_is_needed(
    anchor_candidate_diagnostics_by_cluster: Mapping[str, Any],
) -> bool:
    for row in anchor_candidate_diagnostics_by_cluster.values():
        if not isinstance(row, Mapping):
            continue
        committed_regions = row.get("committed_region_ids")
        if not committed_regions:
            continue
        try:
            committed_count = int(row.get("committed_candidate_count") or 0)
            fallback_count = int(row.get("fallback_candidate_count") or 0)
        except (TypeError, ValueError):
            continue
        if committed_count > 0 and fallback_count == 0:
            return True
    return False


def _build_object_solver_world(
    *,
    room_model: Mapping[str, Any],
    merged_clusters: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    cluster_constraints: Mapping[str, Any] | None,
    grid_mm: int,
) -> dict[str, Any]:
    clusters = merged_clusters.get("clusters")
    cluster_rows = (
        [row for row in clusters if isinstance(row, Mapping)]
        if isinstance(clusters, list)
        else []
    )
    clusters_by_id: dict[str, dict[str, Any]] = {}
    for row in cluster_rows:
        cluster_id = str(row.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        cluster_rules = (
            row.get("cluster_rules")
            if isinstance(row.get("cluster_rules"), Mapping)
            else {}
        )
        object_program = (
            row.get("object_program")
            if isinstance(row.get("object_program"), Mapping)
            else {}
        )
        if not object_program:
            object_program = {
                "cluster_id": cluster_id,
                "members": [
                    m for m in (row.get("members") or []) if isinstance(m, str)
                ],
                "anchors": [
                    m for m in (row.get("anchors") or []) if isinstance(m, str)
                ],
                "dominant_anchor_id": next(iter(row.get("anchors") or []), None),
                "dominant_anchor_candidates": list(
                    cluster_rules.get("dominant_anchor_candidates") or []
                ),
                "placement_order": [
                    m for m in (row.get("members") or []) if isinstance(m, str)
                ],
                "support_edges": [
                    dict(item)
                    for item in (cluster_rules.get("semantic_placements") or [])
                    if isinstance(item, Mapping)
                ],
                "protected_ids": list(
                    (cluster_rules.get("anchor_first_policy") or {}).get(
                        "protected_ids"
                    )
                    or []
                ),
                "droppable_ids": list(
                    (cluster_rules.get("anchor_first_policy") or {}).get(
                        "droppable_ids"
                    )
                    or []
                ),
                "degradation_ladder": list(
                    cluster_rules.get("degradation_ladder") or []
                ),
                "zone_claims": dict(cluster_rules.get("zone_claims") or {}),
                "required_object_ids": [
                    m for m in (row.get("anchors") or []) if isinstance(m, str)
                ],
                "optional_object_ids": [
                    m
                    for m in (row.get("members") or [])
                    if isinstance(m, str) and m not in set(row.get("anchors") or [])
                ],
                "object_specs_by_id": {},
            }
        object_specs = (
            object_program.get("object_specs_by_id")
            if isinstance(object_program.get("object_specs_by_id"), Mapping)
            else {}
        )
        if not object_specs:
            for decision in row.get("decisions") or []:
                if not isinstance(decision, Mapping):
                    continue
                object_id = str(
                    decision.get("object_type") or decision.get("category") or ""
                ).strip()
                rep_dims = (
                    decision.get("rep_dims_m")
                    if isinstance(decision.get("rep_dims_m"), Mapping)
                    else {}
                )
                object_specs[object_id] = {
                    "object_id": object_id,
                    "cluster_id": cluster_id,
                    "category": str(decision.get("category") or object_id),
                    "role": str(decision.get("role") or ""),
                    "priority": str(decision.get("priority") or ""),
                    "preserve_level": str(decision.get("preserve_level") or ""),
                    "rep_dims_mm": {
                        "L": int(round(float(rep_dims.get("L") or 0.0) * 1000.0)),
                        "W": int(round(float(rep_dims.get("W") or 0.0) * 1000.0)),
                        "H": int(round(float(rep_dims.get("H") or 0.0) * 1000.0)),
                    },
                    "allowed_rotations": list(
                        (
                            (cluster_rules.get("allowed_rotations") or {}).get(
                                object_id
                            )
                            or [0, 90, 180, 270]
                        )
                    ),
                    "front": (
                        (cluster_rules.get("facing") or {}).get(object_id) or {}
                    ).get("front"),
                }
            object_program = dict(object_program)
            object_program["object_specs_by_id"] = object_specs
        clusters_by_id[cluster_id] = {
            "cluster_id": cluster_id,
            "tag": str(row.get("tag") or ""),
            "anchors": [
                item
                for item in object_program.get("anchors") or []
                if isinstance(item, str)
            ],
            "object_program": deepcopy(object_program),
            "cluster_rules": dict(cluster_rules),
            "members": [
                item for item in row.get("members") or [] if isinstance(item, str)
            ],
        }

    anchor_order = _object_level_anchor_cluster_order(clusters_by_id, relation_plan)
    region_index = _index_room_regions(room_model)
    protected_regions = _collect_protected_regions(
        room_model, relation_plan, region_index
    )
    cluster_forbidden_regions = _collect_cluster_forbidden_regions(
        clusters_by_id, room_model, relation_plan, region_index
    )
    room_bbox = _object_solver_room_bbox(room_model)
    room_polygon = _object_solver_room_polygon(room_model)
    wall_inner_faces = _object_solver_wall_inner_faces(
        room_model,
        room_bbox,
        room_polygon,
    )
    wall_insets_by_side = _object_solver_wall_insets_by_side(
        wall_inner_faces,
        room_bbox,
    )
    placement_room_bbox = _object_solver_placement_room_bbox(
        room_bbox,
        wall_insets_by_side,
    )
    room_polygon_edges = tuple(
        (room_polygon[index], room_polygon[(index + 1) % len(room_polygon)])
        for index in range(len(room_polygon))
    )
    return {
        "clusters_by_id": clusters_by_id,
        "anchor_cluster_order": anchor_order,
        "grid_mm": max(25, int(grid_mm)),
        "region_index": region_index,
        "room_bbox": room_bbox,
        "raw_room_bbox": room_bbox,
        "placement_room_bbox": placement_room_bbox,
        "wall_insets_by_side": wall_insets_by_side,
        "wall_inner_faces": wall_inner_faces,
        "room_polygon": room_polygon,
        "room_polygon_edges": room_polygon_edges,
        "room_polygon_is_axis_aligned_rectangle": _room_polygon_is_axis_aligned_rectangle(
            room_polygon,
            room_bbox,
        ),
        "protected_regions": protected_regions,
        "cluster_forbidden_regions": cluster_forbidden_regions,
        "auto_window_forbidden_regions_by_cluster": (
            _auto_window_forbidden_regions_by_cluster(cluster_forbidden_regions)
        ),
        "auto_window_forbidden_regions_by_object": (
            _auto_window_forbidden_regions_by_object(cluster_forbidden_regions)
        ),
        "_anchor_candidate_preflight_by_cluster": {},
        "_anchor_reject_reason_counts": {},
        "_object_blocking_region_cache": {},
        "_object_forbidden_region_cache": {},
        "_rect_inside_room_footprint_cache": {},
    }


def _auto_window_forbidden_regions_by_cluster(
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in forbidden_regions:
        if (
            str(row.get("source") or "").startswith("auto_")
            and not str(row.get("object_id") or "").strip()
        ):
            cluster_id = str(row.get("cluster_id") or "").strip()
            region_id = str(row.get("region_id") or "").strip()
            if cluster_id and region_id:
                out.setdefault(cluster_id, []).append(region_id)
    return {key: sorted(set(value)) for key, value in out.items()}


def _auto_window_forbidden_regions_by_object(
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for row in forbidden_regions:
        if not str(row.get("source") or "").startswith("auto_"):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        object_id = str(row.get("object_id") or "").strip()
        region_id = str(row.get("region_id") or "").strip()
        if cluster_id and object_id and region_id:
            out.setdefault(cluster_id, {}).setdefault(object_id, []).append(region_id)
    return {
        cluster_id: {
            object_id: sorted(set(region_ids))
            for object_id, region_ids in object_rows.items()
        }
        for cluster_id, object_rows in out.items()
    }


def _object_solver_geometry_diagnostics(world: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_room_bbox": list(
            world.get("raw_room_bbox") or world.get("room_bbox") or []
        ),
        "placement_room_bbox": list(world.get("placement_room_bbox") or []),
        "wall_insets_by_side": dict(world.get("wall_insets_by_side") or {}),
    }


def _wall_inner_face_reject_reason_counts(
    anchor_candidate_diagnostics_by_cluster: Mapping[str, Any],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for cluster_id, diagnostics in anchor_candidate_diagnostics_by_cluster.items():
        if not isinstance(diagnostics, Mapping):
            continue
        counts = diagnostics.get("anchor_reject_reason_counts")
        if not isinstance(counts, Mapping):
            continue
        total = 0
        for reason, count in counts.items():
            if "wall_inner_face" not in str(reason):
                continue
            try:
                total += int(count)
            except (TypeError, ValueError):
                continue
        if total:
            out[str(cluster_id)] = total
    return out


def _object_level_anchor_cluster_order(
    clusters_by_id: Mapping[str, Mapping[str, Any]],
    relation_plan: Mapping[str, Any] | None,
) -> list[str]:
    cluster_ids = list(clusters_by_id.keys())
    primary = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    secondary = _extract_layout_secondary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    ordered: list[str] = []
    for candidate in (primary, secondary):
        if candidate and candidate in clusters_by_id and candidate not in ordered:
            ordered.append(candidate)
    for cluster_id, row in sorted(clusters_by_id.items()):
        if cluster_id in ordered:
            continue
        object_program = (
            row.get("object_program")
            if isinstance(row.get("object_program"), Mapping)
            else {}
        )
        if object_program.get("dominant_anchor_id"):
            ordered.append(cluster_id)
    for cluster_id in cluster_ids:
        if cluster_id not in ordered:
            ordered.append(cluster_id)
    return ordered


def _index_room_regions(
    room_model: Mapping[str, Any],
) -> dict[str, tuple[int, int, int, int]]:
    room_bbox = _object_solver_room_bbox(room_model)
    out: dict[str, tuple[int, int, int, int]] = {}

    def add_region(region_id: str, bbox: tuple[int, int, int, int] | None) -> None:
        if region_id and bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            out.setdefault(region_id, bbox)

    def bbox_from_mapping(row: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
        bbox = row.get("bbox") or row.get("bbox_mm")
        if isinstance(bbox, Mapping):
            return _bbox_tuple(
                bbox.get("min_x"),
                bbox.get("min_y"),
                bbox.get("max_x"),
                bbox.get("max_y"),
            )
        for key in (
            "near_polygon_ccw",
            "anchor_polygon_ccw",
            "polygon",
            "polygon_ccw",
            "points",
            "mid_polygon_ccw",
        ):
            points = row.get(key)
            if isinstance(points, Sequence) and not isinstance(points, str):
                parsed = []
                for item in points:
                    if not isinstance(item, Mapping):
                        continue
                    try:
                        parsed.append((float(item.get("x")), float(item.get("y"))))
                    except (TypeError, ValueError):
                        continue
                if len(parsed) >= 2:
                    xs = [p[0] for p in parsed]
                    ys = [p[1] for p in parsed]
                    return (
                        int(round(min(xs))),
                        int(round(min(ys))),
                        int(round(max(xs))),
                        int(round(max(ys))),
                    )
        polyline = row.get("polyline_mm")
        if isinstance(polyline, Sequence) and not isinstance(polyline, str):
            parsed = []
            for item in polyline:
                if not isinstance(item, Mapping):
                    continue
                try:
                    parsed.append((float(item.get("x")), float(item.get("y"))))
                except (TypeError, ValueError):
                    continue
            if len(parsed) >= 2:
                try:
                    width_mm = max(0, int(row.get("width_mm") or 0))
                except (TypeError, ValueError):
                    width_mm = 0
                pad = max(1, width_mm // 2)
                xs = [p[0] for p in parsed]
                ys = [p[1] for p in parsed]
                return (
                    int(round(min(xs))) - pad,
                    int(round(min(ys))) - pad,
                    int(round(max(xs))) + pad,
                    int(round(max(ys))) + pad,
                )
        return None

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            region_id = str(value.get("region_id") or value.get("id") or "").strip()
            if region_id:
                add_region(region_id, bbox_from_mapping(value))
            for child in value.values():
                walk(child)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                walk(item)

    walk(room_model)
    _add_primary_corridor_aliases(room_model, out, bbox_from_mapping)

    # Fallback synthetic regions
    min_x, min_y, max_x, max_y = room_bbox
    width = max_x - min_x
    height = max_y - min_y
    center_box = (
        min_x + int(round(width * 0.25)),
        min_y + int(round(height * 0.25)),
        max_x - int(round(width * 0.25)),
        max_y - int(round(height * 0.25)),
    )
    out.setdefault("center_openness_core", center_box)
    out.setdefault("floating_center_zone", center_box)
    out.setdefault(
        "wall_1_usable_200_5800_focal",
        (min_x, min_y, max_x, min_y + max(600, int(round(height * 0.18)))),
    )
    out.setdefault("wall_1_usable_200_5800_anchor", out["wall_1_usable_200_5800_focal"])
    out.setdefault(
        "wall_2_usable_200_2800_anchor",
        (max_x - max(600, int(round(width * 0.18))), min_y, max_x, max_y),
    )
    out.setdefault(
        "wall_3_usable_200_2300_anchor",
        (min_x, max_y - max(600, int(round(height * 0.18))), max_x, max_y),
    )
    out.setdefault(
        "wall_6_usable_2500_3800_focal",
        (min_x, min_y, min_x + max(600, int(round(width * 0.18))), max_y),
    )
    out.setdefault(
        "wall_6_usable_2500_3800_anchor", out["wall_6_usable_2500_3800_focal"]
    )
    out.setdefault(
        "door_1_entry_clearance",
        (
            min_x,
            max_y - max(850, int(round(height * 0.22))),
            min_x + max(1000, int(round(width * 0.22))),
            max_y,
        ),
    )
    out.setdefault(
        "door_1_to_room_center_corridor",
        (
            min_x + int(round(width * 0.15)),
            min_y + int(round(height * 0.35)),
            min_x + int(round(width * 0.55)),
            max_y - int(round(height * 0.15)),
        ),
    )
    out.setdefault(
        "window_1_daylight",
        (min_x, min_y, max_x, min_y + max(850, int(round(height * 0.24)))),
    )
    out.setdefault(
        "privacy_back_zone",
        (
            min_x + int(round(width * 0.45)),
            min_y + int(round(height * 0.45)),
            max_x,
            max_y,
        ),
    )
    return out


def _add_primary_corridor_aliases(
    room_model: Mapping[str, Any],
    region_index: dict[str, tuple[int, int, int, int]],
    bbox_from_mapping: Callable[[Mapping[str, Any]], tuple[int, int, int, int] | None],
) -> None:
    affordance = (
        room_model.get("affordance_map")
        if isinstance(room_model.get("affordance_map"), Mapping)
        else {}
    )
    corridors = affordance.get("circulation_corridors") if affordance else []
    if not isinstance(corridors, Sequence) or isinstance(corridors, str):
        return
    primary_bbox: tuple[int, int, int, int] | None = None
    for row in corridors:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("from") or "") != "entry" or str(row.get("to") or "") not in {
            "room_center",
            "center",
        }:
            continue
        primary_bbox = bbox_from_mapping(row)
        if primary_bbox is not None:
            break
    if primary_bbox is None:
        return
    region_index.setdefault("entry_to_center_corridor", primary_bbox)
    openings = (
        room_model.get("openings")
        if isinstance(room_model.get("openings"), Mapping)
        else {}
    )
    doors = openings.get("doors") if isinstance(openings, Mapping) else []
    if not isinstance(doors, Sequence) or isinstance(doors, str):
        return
    for index, door in enumerate(doors, start=1):
        door_id = (
            str(door.get("id") or "").strip()
            if isinstance(door, Mapping)
            else f"door_{index}"
        )
        if not door_id:
            door_id = f"door_{index}"
        region_index.setdefault(f"{door_id}_to_room_center_corridor", primary_bbox)


def _collect_protected_regions(
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    protected_rows: list[dict[str, Any]] = []
    concept = _concept_from_relation_plan(relation_plan)
    macro_constraints = (
        concept.get("macro_constraints") if isinstance(concept, Mapping) else {}
    )
    protected_topology = (
        macro_constraints.get("protected_topology")
        if isinstance(macro_constraints, Mapping)
        else []
    )
    if not isinstance(protected_topology, Sequence) or isinstance(
        protected_topology, str
    ):
        protected_topology = []
    for row in protected_topology:
        if not isinstance(row, Mapping):
            continue
        region_id = str(row.get("region") or "").strip()
        bbox = _region_bbox_from_ref(region_id, region_index, relation_plan, room_model)
        if bbox is None:
            continue
        applies_to = row.get("applies_to")
        if not isinstance(applies_to, Sequence) or isinstance(applies_to, str):
            applies_to = []
        protected_rows.append(
            {
                "region_id": region_id,
                "bbox": bbox,
                "max_overlap_ratio": float(row.get("max_overlap_ratio") or 0.0),
                "priority": str(row.get("priority") or "medium"),
                "enforcement": str(row.get("enforcement") or "soft"),
                "violation_severity": str(row.get("violation_severity") or "advisory"),
                "zone_type": str(row.get("zone_type") or ""),
                "applies_to": [str(item) for item in applies_to if str(item).strip()],
                "softened_for_clusters": _string_sequence(
                    row.get("softened_for_clusters") or row.get("relaxed_for_clusters")
                ),
            }
        )
    return protected_rows


def _protected_region_policy_indexes(
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> tuple[
    dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]],
    dict[tuple[int, int, int, int], dict[str, Any]],
]:
    policy_by_ref: dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]] = {}
    policy_by_bbox: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in _collect_protected_regions(room_model, relation_plan, region_index):
        region_id = str(row.get("region_id") or "").strip()
        bbox = row.get("bbox")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        policy = _cluster_forbidden_region_policy_from_protected_row(row)
        if region_id:
            policy_by_ref[(region_id, bbox)] = policy
        policy_by_bbox.setdefault(bbox, policy)
    return policy_by_ref, policy_by_bbox


def _cluster_forbidden_region_policy_from_protected_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "max_overlap_ratio": float(row.get("max_overlap_ratio") or 0.0),
        "priority": str(row.get("priority") or "medium"),
        "enforcement": str(row.get("enforcement") or "soft"),
        "violation_severity": str(row.get("violation_severity") or "advisory"),
        "zone_type": str(row.get("zone_type") or "cluster_forbidden_region"),
        "softened_for_clusters": _string_sequence(row.get("softened_for_clusters")),
    }


def _default_cluster_forbidden_region_policy() -> dict[str, Any]:
    return {
        "max_overlap_ratio": 0.0,
        "priority": "high",
        "enforcement": "hard",
        "violation_severity": "blocking",
        "zone_type": "cluster_forbidden_region",
    }


def _default_cluster_forbidden_region_policy_for_cluster(
    *,
    region_id: str,
    cluster_program: Mapping[str, Any],
) -> dict[str, Any]:
    if region_id.strip().lower().replace("-", "_") == "center_access_lane":
        return {
            "max_overlap_ratio": 0.35,
            "priority": "medium",
            "enforcement": "hard_soft",
            "violation_severity": "advisory",
            "zone_type": "conceptual_center_access_lane",
        }
    if _region_id_is_window_clearance(region_id):
        return _default_cluster_forbidden_region_policy()
    if (
        _region_id_is_daylight_quality_region(region_id)
        and _cluster_core_max_height_mm(cluster_program)
        <= OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_SOFT_MAX_MM
    ):
        return {
            "max_overlap_ratio": OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_MAX_OVERLAP_RATIO,
            "priority": "medium",
            "enforcement": "soft",
            "violation_severity": "advisory",
            "zone_type": "daylight_quality_soft",
        }
    return _default_cluster_forbidden_region_policy()


def _region_id_is_window_clearance(region_id: str) -> bool:
    token = region_id.strip().lower()
    return "window" in token and "clearance" in token


def _region_id_is_daylight_quality_region(region_id: str) -> bool:
    token = region_id.strip().lower()
    return "daylight" in token and not _region_id_is_window_clearance(region_id)


def _cluster_max_height_mm(cluster_program: Mapping[str, Any]) -> int:
    specs = _object_specs_by_id(cluster_program)
    heights: list[int] = []
    for spec in specs.values():
        if not isinstance(spec, Mapping):
            continue
        heights.append(_object_spec_height_mm(spec))
    return max(heights, default=0)


def _cluster_core_max_height_mm(cluster_program: Mapping[str, Any]) -> int:
    specs = _object_specs_by_id(cluster_program)
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    protected_ids = set(_string_sequence(object_program.get("protected_ids")))
    required_ids = set(_string_sequence(object_program.get("required_object_ids")))
    anchor_ids = set(_string_sequence(object_program.get("anchors")))
    core_ids = set(protected_ids)
    core_ids.update(required_ids)
    core_ids.update(anchor_ids)
    dominant_anchor_id = str(object_program.get("dominant_anchor_id") or "").strip()
    if dominant_anchor_id:
        core_ids.add(dominant_anchor_id)
    droppable_ids = set(_string_sequence(object_program.get("droppable_ids")))
    heights: list[int] = []
    for object_id, spec in specs.items():
        if not isinstance(spec, Mapping):
            continue
        key = str(object_id or spec.get("object_id") or "").strip()
        if key not in core_ids:
            continue
        if (
            key in droppable_ids
            and key not in protected_ids
            and key not in required_ids
            and key not in anchor_ids
            and key != dominant_anchor_id
        ):
            continue
        heights.append(_object_spec_height_mm(spec))
    if heights:
        return max(heights)
    return _cluster_max_height_mm(cluster_program)


def _object_specs_by_id(cluster_program: Mapping[str, Any]) -> Mapping[str, Any]:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    return specs


def _object_spec_height_mm(spec: Mapping[str, Any]) -> int:
    dims = (
        spec.get("rep_dims_mm") if isinstance(spec.get("rep_dims_mm"), Mapping) else {}
    )
    for key in ("H", "height_mm", "height"):
        raw_value = dims.get(key)
        if raw_value is None:
            raw_value = spec.get(key)
        if raw_value is None:
            continue
        try:
            return int(round(float(raw_value)))
        except (TypeError, ValueError):
            continue
    return 0


def _object_spec_tokens(object_id: str, spec: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(value or "").strip()
        for value in (
            object_id,
            spec.get("object_id"),
            spec.get("base_object_id"),
            spec.get("object_type"),
            spec.get("category"),
        )
        if str(value or "").strip()
    )


def _object_spec_is_window_treatment(object_id: str, spec: Mapping[str, Any]) -> bool:
    return any(
        _is_window_treatment_token(token)
        for token in _object_spec_tokens(object_id, spec)
    )


def _object_spec_is_visual_storage_like(
    object_id: str, spec: Mapping[str, Any]
) -> bool:
    for token in _object_spec_tokens(object_id, spec):
        normalized = token.strip().lower().replace("-", "_").replace(" ", "_")
        if is_profile_storage_object(normalized):
            return True
        if any(part in normalized for part in _VISUAL_STORAGE_OBJECT_TOKENS):
            return True
    return False


def _object_spec_blocks_window_by_height(
    object_id: str, spec: Mapping[str, Any]
) -> bool:
    if _object_spec_is_window_treatment(object_id, spec):
        return False
    if _object_spec_height_mm(spec) <= OBJECT_LEVEL_VISUAL_STORAGE_DAYLIGHT_SOFT_MAX_MM:
        return False
    return _object_spec_is_visual_storage_like(object_id, spec)


def _object_spec_blocks_window_clearance(
    object_id: str, spec: Mapping[str, Any]
) -> bool:
    return not _object_spec_is_window_treatment(object_id, spec)


def _window_clearance_region_ids_from_index(
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> list[str]:
    region_ids: list[str] = []
    for raw_region_id in region_index:
        region_id = str(raw_region_id or "").strip()
        if not region_id:
            continue
        if _region_id_is_window_clearance(region_id):
            region_ids.append(region_id)
    return sorted(set(region_ids))


def _daylight_quality_region_ids_from_index(
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> list[str]:
    region_ids: list[str] = []
    for raw_region_id in region_index:
        region_id = str(raw_region_id or "").strip()
        if not region_id:
            continue
        if _region_id_is_daylight_quality_region(region_id):
            region_ids.append(region_id)
    return sorted(set(region_ids))


def _auto_window_blocking_region_ids_for_object(
    *,
    cluster_program: Mapping[str, Any],
    object_id: str,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> list[str]:
    if _cluster_is_window_treatment_only(cluster_program):
        return []
    spec = _object_spec(cluster_program, object_id)
    if not isinstance(spec, Mapping):
        return []
    region_ids: list[str] = []
    if _object_spec_blocks_window_clearance(object_id, spec):
        region_ids.extend(_window_clearance_region_ids_from_index(region_index))
    if _object_spec_blocks_window_by_height(object_id, spec):
        region_ids.extend(_daylight_quality_region_ids_from_index(region_index))
    return sorted(set(region_ids))


def _auto_window_blocking_region_ids_by_object_for_cluster(
    *,
    cluster_program: Mapping[str, Any],
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> dict[str, list[str]]:
    if _cluster_is_window_treatment_only(cluster_program):
        return {}
    out: dict[str, list[str]] = {}
    for raw_object_id, spec in _object_specs_by_id(cluster_program).items():
        if not isinstance(spec, Mapping):
            continue
        object_id = str(raw_object_id or spec.get("object_id") or "").strip()
        if not object_id:
            continue
        region_ids = _auto_window_blocking_region_ids_for_object(
            cluster_program=cluster_program,
            object_id=object_id,
            region_index=region_index,
        )
        if region_ids:
            out[object_id] = region_ids
    return out


def _cluster_is_window_treatment_only(cluster_program: Mapping[str, Any]) -> bool:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    object_tokens: list[list[str]] = []
    for object_id, spec in specs.items():
        if not isinstance(spec, Mapping):
            continue
        object_tokens.append(
            [
                str(object_id or ""),
                str(spec.get("object_id") or ""),
                str(spec.get("object_type") or ""),
                str(spec.get("category") or ""),
            ]
        )
    if not object_tokens:
        members = object_program.get("members")
        if isinstance(members, Sequence) and not isinstance(members, str):
            object_tokens = [[str(member or "")] for member in members]
    return bool(object_tokens) and all(
        any(_is_window_treatment_token(token) for token in tokens)
        for tokens in object_tokens
    )


def _is_window_treatment_token(value: str) -> bool:
    token = value.strip().lower()
    if not token:
        return False
    normalized = token.replace("-", "_").replace(" ", "_")
    return any(part in normalized for part in _WINDOW_TREATMENT_OBJECT_TOKENS)


def _cluster_forbidden_region_policy(
    *,
    region_id: str,
    bbox: tuple[int, int, int, int],
    cluster_id: str,
    cluster_program: Mapping[str, Any],
    protected_policy_by_ref: Mapping[
        tuple[str, tuple[int, int, int, int]], Mapping[str, Any]
    ],
    protected_policy_by_bbox: Mapping[tuple[int, int, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    protected_policy = protected_policy_by_ref.get(
        (region_id, bbox)
    ) or protected_policy_by_bbox.get(bbox)
    if protected_policy is None:
        return _default_cluster_forbidden_region_policy_for_cluster(
            region_id=region_id,
            cluster_program=cluster_program,
        )
    return _protected_region_policy_for_cluster(protected_policy, cluster_id=cluster_id)


def _protected_region_policy_for_cluster(
    protected_policy: Mapping[str, Any],
    *,
    cluster_id: str,
) -> dict[str, Any]:
    policy = {
        "max_overlap_ratio": float(protected_policy.get("max_overlap_ratio") or 0.0),
        "priority": str(protected_policy.get("priority") or "medium"),
        "enforcement": str(protected_policy.get("enforcement") or "soft"),
        "violation_severity": str(
            protected_policy.get("violation_severity") or "advisory"
        ),
        "zone_type": str(
            protected_policy.get("zone_type") or "cluster_forbidden_region"
        ),
    }
    softened_for_clusters = set(
        _string_sequence(protected_policy.get("softened_for_clusters"))
    )
    if cluster_id and cluster_id in softened_for_clusters:
        policy["priority"] = "medium"
        policy["enforcement"] = "soft"
        policy["violation_severity"] = "advisory"
    return policy


def _collect_cluster_forbidden_regions(
    clusters_by_id: Mapping[str, Mapping[str, Any]],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hint_rows = _cluster_forbidden_hint_rows(relation_plan)
    protected_policy_by_ref, protected_policy_by_bbox = (
        _protected_region_policy_indexes(room_model, relation_plan, region_index)
    )
    seen: set[tuple[str, str, str, tuple[int, int, int, int]]] = set()
    for cluster_id, cluster_program in clusters_by_id.items():
        object_program = (
            cluster_program.get("object_program")
            if isinstance(cluster_program.get("object_program"), Mapping)
            else {}
        )
        cluster_rules = (
            cluster_program.get("cluster_rules")
            if isinstance(cluster_program.get("cluster_rules"), Mapping)
            else {}
        )
        object_zone_claims = (
            object_program.get("zone_claims")
            if isinstance(object_program.get("zone_claims"), Mapping)
            else {}
        )
        rule_zone_claims = (
            cluster_rules.get("zone_claims")
            if isinstance(cluster_rules.get("zone_claims"), Mapping)
            else {}
        )
        hint = hint_rows.get(cluster_id, {})
        region_ids = _dedupe_string_sequence(
            [
                *_string_sequence(object_zone_claims.get("avoid_regions")),
                *_string_sequence(rule_zone_claims.get("avoid_regions")),
                *_string_sequence(
                    hint.get("forbidden_region_ids")
                    if isinstance(hint, Mapping)
                    else []
                ),
            ]
        )
        for region_id in region_ids:
            for bbox in _region_bboxes_from_ref(
                region_id, region_index, relation_plan, room_model
            ):
                key = (cluster_id, "", region_id, bbox)
                if key in seen:
                    continue
                seen.add(key)
                policy = _cluster_forbidden_region_policy(
                    region_id=region_id,
                    bbox=bbox,
                    cluster_id=cluster_id,
                    cluster_program=cluster_program,
                    protected_policy_by_ref=protected_policy_by_ref,
                    protected_policy_by_bbox=protected_policy_by_bbox,
                )
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "region_id": region_id,
                        "bbox": bbox,
                        **policy,
                    }
                )
        object_region_ids = _auto_window_blocking_region_ids_by_object_for_cluster(
            cluster_program=cluster_program,
            region_index=region_index,
        )
        for object_id, object_forbidden_region_ids in object_region_ids.items():
            for region_id in object_forbidden_region_ids:
                for bbox in _region_bboxes_from_ref(
                    region_id, region_index, relation_plan, room_model
                ):
                    key = (cluster_id, object_id, region_id, bbox)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "cluster_id": cluster_id,
                            "object_id": object_id,
                            "region_id": region_id,
                            "bbox": bbox,
                            **_auto_window_forbidden_region_policy(
                                region_id=region_id,
                                cluster_program=cluster_program,
                                object_id=object_id,
                            ),
                            "source": _auto_window_forbidden_region_source(region_id),
                        }
                    )
    return rows


def _auto_window_forbidden_region_policy(
    *,
    region_id: str,
    cluster_program: Mapping[str, Any],
    object_id: str,
) -> dict[str, Any]:
    if _object_daylight_quality_relaxed(
        region_id=region_id,
        cluster_program=cluster_program,
        object_id=object_id,
    ):
        return {
            "max_overlap_ratio": OBJECT_LEVEL_SECONDARY_DAYLIGHT_MAX_OVERLAP_RATIO,
            "priority": "medium",
            "enforcement": "soft",
            "violation_severity": "advisory",
            "zone_type": "daylight_quality_soft",
            "relaxation_reason": "secondary_daylight_quality_relaxation",
        }
    return _default_auto_window_forbidden_region_policy(region_id)


def _object_daylight_quality_relaxed(
    *,
    region_id: str,
    cluster_program: Mapping[str, Any],
    object_id: str,
) -> bool:
    if not _region_id_is_daylight_quality_region(region_id):
        return False
    specs = _object_specs_by_id(cluster_program)
    spec = specs.get(object_id)
    if not isinstance(spec, Mapping):
        return False
    return bool(spec.get("daylight_quality_relaxed"))


def _default_auto_window_forbidden_region_policy(region_id: str) -> dict[str, Any]:
    if _region_id_is_daylight_quality_region(region_id):
        return {
            "max_overlap_ratio": 0.0,
            "priority": "high",
            "enforcement": "hard",
            "violation_severity": "blocking",
            "zone_type": "daylight_quality_blocking",
        }
    return {
        **_default_cluster_forbidden_region_policy(),
        "zone_type": "window_clearance",
    }


def _auto_window_forbidden_region_source(region_id: str) -> str:
    if _region_id_is_daylight_quality_region(region_id):
        return "auto_daylight_quality"
    if _region_id_is_window_clearance(region_id):
        return "auto_window_clearance"
    return "auto_window"


def _cluster_forbidden_hint_rows(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(relation_plan, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for section_name in (
        "anchor_layout_hints_by_cluster",
        "anchor_region_preferences_by_cluster",
    ):
        section = relation_plan.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for cluster_id, row in section.items():
            if not isinstance(row, Mapping):
                continue
            key = str(cluster_id or "").strip()
            if not key:
                continue
            target = out.setdefault(key, {})
            target["forbidden_region_ids"] = _dedupe_string_sequence(
                [
                    *_string_sequence(target.get("forbidden_region_ids")),
                    *_string_sequence(row.get("forbidden_region_ids")),
                ]
            )
    concept = _concept_from_relation_plan(relation_plan)
    macro_constraints = (
        concept.get("macro_constraints") if isinstance(concept, Mapping) else {}
    )
    anchor_constraints = (
        macro_constraints.get("anchor_region_constraints")
        if isinstance(macro_constraints, Mapping)
        else []
    )
    if isinstance(anchor_constraints, Sequence) and not isinstance(
        anchor_constraints, str
    ):
        for row in anchor_constraints:
            if not isinstance(row, Mapping):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            target = out.setdefault(cluster_id, {})
            target["forbidden_region_ids"] = _dedupe_string_sequence(
                [
                    *_string_sequence(target.get("forbidden_region_ids")),
                    *_string_sequence(row.get("forbidden_region_ids")),
                ]
            )
    return out


def _object_solver_room_bbox(
    room_model: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    room = room_model.get("room") if isinstance(room_model.get("room"), Mapping) else {}
    polygon = room.get("polygon_ccw") if isinstance(room, Mapping) else None
    points = []
    if isinstance(polygon, Sequence) and not isinstance(polygon, str):
        for row in polygon:
            if not isinstance(row, Mapping):
                continue
            try:
                points.append((float(row.get("x")), float(row.get("y"))))
            except (TypeError, ValueError):
                continue
    if len(points) >= 2:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs))),
            int(round(max(ys))),
        )
    bbox = room.get("bbox") if isinstance(room, Mapping) else None
    if isinstance(bbox, Mapping):
        parsed = _bbox_tuple(
            bbox.get("min_x"), bbox.get("min_y"), bbox.get("max_x"), bbox.get("max_y")
        )
        if parsed is not None:
            return parsed
    return (0, 0, 6000, 4000)


def _object_solver_room_polygon(
    room_model: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    room = room_model.get("room") if isinstance(room_model.get("room"), Mapping) else {}
    polygon = room.get("polygon_ccw") if isinstance(room, Mapping) else None
    points: list[tuple[float, float]] = []
    if isinstance(polygon, Sequence) and not isinstance(polygon, str):
        for row in polygon:
            if not isinstance(row, Mapping):
                continue
            try:
                points.append((float(row.get("x")), float(row.get("y"))))
            except (TypeError, ValueError):
                continue
    if len(points) < 3:
        return ()
    if points[0] == points[-1]:
        points = points[:-1]
    return tuple(points)


def _room_polygon_is_axis_aligned_rectangle(
    polygon: Sequence[tuple[float, float]],
    room_bbox: tuple[int, int, int, int],
) -> bool:
    if (
        len(polygon) != 4
        or room_bbox[2] <= room_bbox[0]
        or room_bbox[3] <= room_bbox[1]
    ):
        return False
    normalized_points: list[tuple[int, int]] = []
    for x_value, y_value in polygon:
        try:
            normalized_points.append(
                (int(round(float(x_value))), int(round(float(y_value))))
            )
        except (TypeError, ValueError):
            return False
    if len(set(normalized_points)) != 4:
        return False
    min_x, min_y, max_x, max_y = room_bbox
    return set(normalized_points) == {
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    }


def _object_solver_wall_inner_faces(
    room_model: Mapping[str, Any],
    room_bbox: tuple[int, int, int, int],
    room_polygon: Sequence[tuple[float, float]],
) -> tuple[dict[str, Any], ...]:
    room = room_model.get("room") if isinstance(room_model.get("room"), Mapping) else {}
    default_thickness = _coerce_nonnegative_int(
        room.get("wall_thickness_mm") if isinstance(room, Mapping) else None
    )
    if default_thickness is None:
        default_thickness = DEFAULT_WALL_THICKNESS_MM
    thickness_by_wall_id = (
        room.get("wall_thickness_by_wall_id")
        if isinstance(room, Mapping)
        and isinstance(room.get("wall_thickness_by_wall_id"), Mapping)
        else {}
    )
    topology = (
        room_model.get("topology")
        if isinstance(room_model.get("topology"), Mapping)
        else {}
    )
    wall_graph = (
        topology.get("wall_graph")
        if isinstance(topology, Mapping)
        and isinstance(topology.get("wall_graph"), Mapping)
        else {}
    )
    nodes = (
        wall_graph.get("nodes")
        if isinstance(wall_graph, Mapping)
        and isinstance(wall_graph.get("nodes"), Sequence)
        and not isinstance(wall_graph.get("nodes"), str)
        else []
    )
    faces: list[dict[str, Any]] = []
    for index, row in enumerate(nodes, start=1):
        if not isinstance(row, Mapping):
            continue
        start = _point_tuple_from_mapping(row.get("start_mm") or row.get("p1_mm"))
        end = _point_tuple_from_mapping(row.get("end_mm") or row.get("p2_mm"))
        if start is None or end is None:
            continue
        wall_id = str(row.get("id") or row.get("wall_id") or f"wall_{index}")
        thickness = _coerce_nonnegative_int(row.get("thickness_mm"))
        if thickness is None:
            thickness = _coerce_nonnegative_int(thickness_by_wall_id.get(wall_id))
        if thickness is None:
            thickness = default_thickness
        normal = _normal_tuple_from_mapping(row.get("inward_normal"))
        faces.extend(
            _wall_inner_face_from_segment(
                wall_id=wall_id,
                start=start,
                end=end,
                inward_normal=normal,
                thickness_mm=thickness,
            )
        )
    if faces:
        return tuple(faces)

    if len(room_polygon) >= 3:
        for index, start in enumerate(room_polygon):
            end = room_polygon[(index + 1) % len(room_polygon)]
            faces.extend(
                _wall_inner_face_from_segment(
                    wall_id=f"wall_{index + 1}",
                    start=(float(start[0]), float(start[1])),
                    end=(float(end[0]), float(end[1])),
                    inward_normal=None,
                    thickness_mm=default_thickness,
                )
            )
    elif room_bbox[2] > room_bbox[0] and room_bbox[3] > room_bbox[1]:
        min_x, min_y, max_x, max_y = room_bbox
        bbox_edges = (
            ((min_x, min_y), (max_x, min_y)),
            ((max_x, min_y), (max_x, max_y)),
            ((max_x, max_y), (min_x, max_y)),
            ((min_x, max_y), (min_x, min_y)),
        )
        for index, (start, end) in enumerate(bbox_edges, start=1):
            faces.extend(
                _wall_inner_face_from_segment(
                    wall_id=f"wall_{index}",
                    start=(float(start[0]), float(start[1])),
                    end=(float(end[0]), float(end[1])),
                    inward_normal=None,
                    thickness_mm=default_thickness,
                )
            )
    return tuple(faces)


def _wall_inner_face_from_segment(
    *,
    wall_id: str,
    start: tuple[float, float],
    end: tuple[float, float],
    inward_normal: tuple[float, float] | None,
    thickness_mm: int,
) -> list[dict[str, Any]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []
    tangent = (dx / length, dy / length)
    normal = inward_normal if inward_normal is not None else (-tangent[1], tangent[0])
    normal_len = math.hypot(normal[0], normal[1])
    if normal_len < 1e-6:
        normal = (-tangent[1], tangent[0])
        normal_len = 1.0
    normalized_normal = (normal[0] / normal_len, normal[1] / normal_len)
    return [
        {
            "wall_id": wall_id,
            "start": start,
            "end": end,
            "length_mm": length,
            "tangent": tangent,
            "inward_normal": normalized_normal,
            "thickness_mm": max(0, int(thickness_mm)),
            "inset_mm": max(0.0, float(thickness_mm) / 2.0),
        }
    ]


def _object_solver_wall_insets_by_side(
    wall_inner_faces: Sequence[Mapping[str, Any]],
    room_bbox: tuple[int, int, int, int],
) -> dict[str, int]:
    min_x, min_y, max_x, max_y = room_bbox
    insets = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    tolerance = 5.0
    for face in wall_inner_faces:
        start = face.get("start")
        end = face.get("end")
        normal = face.get("inward_normal")
        if (
            not isinstance(start, Sequence)
            or isinstance(start, str)
            or len(start) != 2
            or not isinstance(end, Sequence)
            or isinstance(end, str)
            or len(end) != 2
            or not isinstance(normal, Sequence)
            or isinstance(normal, str)
            or len(normal) != 2
        ):
            continue
        try:
            sx, sy = float(start[0]), float(start[1])
            ex, ey = float(end[0]), float(end[1])
            nx, ny = float(normal[0]), float(normal[1])
            inset = int(round(float(face.get("inset_mm") or 0.0)))
        except (TypeError, ValueError):
            continue
        if inset <= 0:
            continue
        if abs(sy - min_y) <= tolerance and abs(ey - min_y) <= tolerance and ny > 0.5:
            insets["top"] = max(insets["top"], inset)
        elif (
            abs(sx - max_x) <= tolerance and abs(ex - max_x) <= tolerance and nx < -0.5
        ):
            insets["right"] = max(insets["right"], inset)
        elif (
            abs(sy - max_y) <= tolerance and abs(ey - max_y) <= tolerance and ny < -0.5
        ):
            insets["bottom"] = max(insets["bottom"], inset)
        elif abs(sx - min_x) <= tolerance and abs(ex - min_x) <= tolerance and nx > 0.5:
            insets["left"] = max(insets["left"], inset)
    return insets


def _object_solver_placement_room_bbox(
    room_bbox: tuple[int, int, int, int],
    wall_insets_by_side: Mapping[str, int],
) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = room_bbox
    left = max(0, int(wall_insets_by_side.get("left") or 0))
    top = max(0, int(wall_insets_by_side.get("top") or 0))
    right = max(0, int(wall_insets_by_side.get("right") or 0))
    bottom = max(0, int(wall_insets_by_side.get("bottom") or 0))
    return (min_x + left, min_y + top, max_x - right, max_y - bottom)


def _point_tuple_from_mapping(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return (float(value.get("x")), float(value.get("y")))
    except (TypeError, ValueError):
        return None


def _normal_tuple_from_mapping(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return (float(value.get("x")), float(value.get("y")))
    except (TypeError, ValueError):
        return None


def _coerce_nonnegative_int(value: Any) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0, int(round(float(value))))


def _room_id(room_model: Mapping[str, Any]) -> str:
    room = room_model.get("room") if isinstance(room_model.get("room"), Mapping) else {}
    value = room.get("room_id") if isinstance(room, Mapping) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "room_1"


def _room_type(room_model: Mapping[str, Any]) -> str:
    meta = room_model.get("meta") if isinstance(room_model.get("meta"), Mapping) else {}
    value = meta.get("room_type") if isinstance(meta, Mapping) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = room_model.get("room_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "room"


def _bbox_tuple(
    min_x: Any, min_y: Any, max_x: Any, max_y: Any
) -> tuple[int, int, int, int] | None:
    try:
        x1 = int(round(float(min_x)))
        y1 = int(round(float(min_y)))
        x2 = int(round(float(max_x)))
        y2 = int(round(float(max_y)))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _region_bbox_from_ref(
    region_ref: str,
    region_index: Mapping[str, tuple[int, int, int, int]],
    relation_plan: Mapping[str, Any] | None,
    room_model: Mapping[str, Any],
) -> tuple[int, int, int, int] | None:
    if not region_ref:
        return None
    if region_ref in region_index:
        return region_index[region_ref]
    macro_region_map = (
        relation_plan.get("macro_region_map")
        if isinstance(relation_plan, Mapping)
        else {}
    )
    if isinstance(macro_region_map, Mapping):
        for row in macro_region_map.get("regions") or []:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("region_id") or "") != region_ref:
                continue
            bboxes = [
                _region_index_bbox(str(source_id), region_index)
                or _region_bbox_from_ref(
                    str(source_id),
                    region_index,
                    _without_macro_region_map(relation_plan),
                    room_model,
                )
                for source_id in (row.get("source_ids") or [])
            ]
            if bboxes:
                return _union_bboxes([bbox for bbox in bboxes if bbox is not None])
    # Fallback heuristics from name
    room_bbox = _object_solver_room_bbox(room_model)
    min_x, min_y, max_x, max_y = room_bbox
    width = max_x - min_x
    height = max_y - min_y
    token = str(region_ref).lower()
    if "bottom" in token and "wall" in token:
        return (min_x, max_y - max(700, int(round(height * 0.18))), max_x, max_y)
    if "left" in token and "wall" in token:
        return (min_x, min_y, min_x + max(700, int(round(width * 0.18))), max_y)
    if "right" in token and "wall" in token:
        return (max_x - max(700, int(round(width * 0.18))), min_y, max_x, max_y)
    if "focal" in token or ("wall" in token and "top" in token):
        return (min_x, min_y, max_x, min_y + max(700, int(round(height * 0.18))))
    if "daylight" in token or "window" in token:
        return (min_x, min_y, max_x, min_y + max(900, int(round(height * 0.24))))
    if "entry" in token or "door" in token:
        return (
            min_x,
            max_y - max(850, int(round(height * 0.22))),
            min_x + max(1000, int(round(width * 0.22))),
            max_y,
        )
    if "center" in token or "floating" in token:
        return (
            min_x + int(round(width * 0.25)),
            min_y + int(round(height * 0.25)),
            max_x - int(round(width * 0.25)),
            max_y - int(round(height * 0.25)),
        )
    if "privacy" in token or "deep" in token:
        return (
            min_x + int(round(width * 0.45)),
            min_y + int(round(height * 0.45)),
            max_x,
            max_y,
        )
    if "edge" in token or "storage" in token:
        return (min_x, min_y, max_x, max_y)
    return None


def _region_index_bbox(
    region_ref: str,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if region_ref in region_index:
        return region_index[region_ref]
    aliases = []
    if "_focal" in region_ref:
        aliases.append(region_ref.replace("_focal", "_anchor"))
    if "_anchor" in region_ref:
        aliases.append(region_ref.replace("_anchor", "_focal"))
    for alias in aliases:
        if alias in region_index:
            return region_index[alias]
    return None


def _without_macro_region_map(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fallback_plan = dict(relation_plan) if isinstance(relation_plan, Mapping) else {}
    fallback_plan.pop("macro_region_map", None)
    return fallback_plan


def _dedupe_region_bboxes(
    bboxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bbox in bboxes:
        if bbox in seen:
            continue
        seen.add(bbox)
        out.append(bbox)
    return out


def _region_bboxes_from_ref(
    region_ref: str,
    region_index: Mapping[str, tuple[int, int, int, int]],
    relation_plan: Mapping[str, Any] | None,
    room_model: Mapping[str, Any],
) -> list[tuple[int, int, int, int]]:
    if not region_ref:
        return []
    direct_bbox = _region_index_bbox(region_ref, region_index)
    if direct_bbox is not None:
        return [direct_bbox]

    macro_region_map = (
        relation_plan.get("macro_region_map")
        if isinstance(relation_plan, Mapping)
        else {}
    )
    if isinstance(macro_region_map, Mapping):
        for row in macro_region_map.get("regions") or []:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("region_id") or "") != region_ref:
                continue
            source_bboxes = [
                bbox
                for source_id in (row.get("source_ids") or [])
                if (
                    bbox := (
                        _region_index_bbox(str(source_id), region_index)
                        or _region_bbox_from_ref(
                            str(source_id),
                            region_index,
                            _without_macro_region_map(relation_plan),
                            room_model,
                        )
                    )
                )
                is not None
            ]
            if source_bboxes:
                return _dedupe_region_bboxes(source_bboxes)

    fallback_bbox = _region_bbox_from_ref(
        region_ref, region_index, _without_macro_region_map(relation_plan), room_model
    )
    return [fallback_bbox] if fallback_bbox is not None else []


def _union_bboxes(
    bboxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    rows = [bbox for bbox in bboxes if bbox is not None]
    if not rows:
        return None
    return (
        min(b[0] for b in rows),
        min(b[1] for b in rows),
        max(b[2] for b in rows),
        max(b[3] for b in rows),
    )


def _cluster_program_dominant_anchor(cluster_program: Mapping[str, Any]) -> str | None:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    for value in (
        object_program.get("dominant_anchor_id"),
        next(iter(object_program.get("anchors") or []), None),
        next(iter(object_program.get("dominant_anchor_candidates") or []), None),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _cluster_is_solver_trial_optional(cluster_program: Mapping[str, Any]) -> bool:
    anchor_id = _cluster_program_dominant_anchor(cluster_program)
    if anchor_id is None:
        return False
    spec = _object_spec(cluster_program, anchor_id)
    if not isinstance(spec, Mapping):
        return False
    return any(
        bool(spec.get(key))
        for key in ("trial_optional", "solver_trial", "budget_trial")
    )


def _object_spec(
    cluster_program: Mapping[str, Any], object_id: str
) -> Mapping[str, Any] | None:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    row = specs.get(object_id)
    return row if isinstance(row, Mapping) else None


def _rotate_dims_for_rot(length_mm: int, width_mm: int, rot: int) -> tuple[int, int]:
    return (length_mm, width_mm) if int(rot) % 180 == 0 else (width_mm, length_mm)


def _generate_anchor_pose_candidates(
    *,
    cluster_program: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    world: Mapping[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    cluster_id = str(cluster_program.get("cluster_id") or "")
    anchor_id = _cluster_program_dominant_anchor(cluster_program)
    if not anchor_id:
        return []
    spec = _object_spec(cluster_program, anchor_id)
    if spec is None:
        return []
    dims = (
        spec.get("rep_dims_mm") if isinstance(spec.get("rep_dims_mm"), Mapping) else {}
    )
    length_mm = max(200, int(dims.get("L") or 0))
    width_mm = max(200, int(dims.get("W") or 0))
    allowed_rotations = (
        spec.get("allowed_rotations")
        if isinstance(spec.get("allowed_rotations"), Sequence)
        else [0, 90, 180, 270]
    )
    relation_plan = relation_plan or {}
    anchor_hints = (
        relation_plan.get("anchor_layout_hints_by_cluster")
        if isinstance(relation_plan.get("anchor_layout_hints_by_cluster"), Mapping)
        else {}
    )
    anchor_region_preferences = (
        relation_plan.get("anchor_region_preferences_by_cluster")
        if isinstance(
            relation_plan.get("anchor_region_preferences_by_cluster"), Mapping
        )
        else {}
    )
    hint = (
        anchor_hints.get(cluster_id)
        if isinstance(anchor_hints, Mapping)
        and isinstance(anchor_hints.get(cluster_id), Mapping)
        else {}
    )
    region_preference = (
        anchor_region_preferences.get(cluster_id)
        if isinstance(anchor_region_preferences, Mapping)
        and isinstance(anchor_region_preferences.get(cluster_id), Mapping)
        else {}
    )
    if not hint and region_preference:
        hint = region_preference
    placement_behavior = _placement_behavior_from_rows(hint, region_preference)
    entry_relation = _cluster_entry_relation_for_plan(relation_plan, cluster_id)
    requires_near_wall_inward = _anchor_requires_near_wall_inward_contract(
        cluster_id=cluster_id,
        object_id=anchor_id,
        cluster_program=cluster_program,
        relation_plan=relation_plan,
    )
    zone_assignment = str(hint.get("zone_assignment") or "")
    required_region_ids = _string_sequence(hint.get("required_region_ids"))
    auto_window_forbidden_region_ids = _auto_window_blocking_region_ids_for_object(
        cluster_program=cluster_program,
        object_id=anchor_id,
        region_index=world["region_index"],
    )
    forbidden_region_ids = _dedupe_string_sequence(
        [
            *_string_sequence(hint.get("forbidden_region_ids")),
            *auto_window_forbidden_region_ids,
        ]
    )
    committed_region_ids = _dedupe_string_sequence(
        [
            *_string_sequence(hint.get("committed_region_ids")),
            *_string_sequence(region_preference.get("committed_region_ids")),
        ]
    )
    if not committed_region_ids and zone_assignment:
        committed_region_ids = [zone_assignment]
    committed_region_ids = [
        region_id
        for region_id in committed_region_ids
        if region_id and region_id != "keep_open_center"
    ]
    preferred_region_ids = _string_sequence(hint.get("preferred_region_ids"))
    preferred_region_ids = _dedupe_string_sequence(
        [
            *required_region_ids,
            zone_assignment,
            *preferred_region_ids,
            *[
                item
                for item in _string_sequence(
                    region_preference.get("preferred_region_ids")
                )
                if item != zone_assignment
            ],
        ]
    )
    preferred_wall_side = str(hint.get("preferred_wall_side") or "").strip()
    committed_wall_side = str(
        hint.get("committed_wall_side")
        or region_preference.get("committed_wall_side")
        or preferred_wall_side
    ).strip()
    anchor_strength = str(hint.get("anchor_strength") or "medium").strip().lower()
    committed_fallback_relaxed = _relation_plan_allows_committed_fallback(relation_plan)
    committed_region_bboxes = [
        bbox
        for region_id in committed_region_ids
        if region_id
        for bbox in _region_bboxes_from_ref(
            str(region_id), world["region_index"], relation_plan, room_model
        )
    ]
    committed_region_bboxes = _dedupe_region_bboxes(committed_region_bboxes)
    region_bboxes = [
        bbox
        for region_id in preferred_region_ids
        if region_id
        for bbox in _region_bboxes_from_ref(
            str(region_id), world["region_index"], relation_plan, room_model
        )
    ]
    region_bboxes = _dedupe_region_bboxes(region_bboxes)
    placement_room_bbox = _object_solver_fit_room_bbox(world)
    if not region_bboxes:
        region_bboxes = [placement_room_bbox]
    forbidden_region_bboxes = [
        bbox
        for region_id in forbidden_region_ids
        for bbox in _region_bboxes_from_ref(
            str(region_id), world["region_index"], relation_plan, room_model
        )
    ]
    forbidden_region_bboxes = _dedupe_region_bboxes(forbidden_region_bboxes)
    forbidden_regions = _anchor_forbidden_regions_for_cluster(
        cluster_id=cluster_id,
        object_id=anchor_id,
        world=world,
        resolved_region_bboxes=forbidden_region_bboxes,
    )
    entry_reference_bboxes = _entry_reference_bboxes_for_anchor(
        refs=[
            zone_assignment,
            *required_region_ids,
            *preferred_region_ids,
            *[
                region_id
                for region_id in forbidden_region_ids
                if _entry_reference_region_id_allowed(region_id)
            ],
        ],
        region_index=world["region_index"],
        relation_plan=relation_plan,
        room_model=room_model,
    )
    primary_cluster = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    storage_like_anchor = _anchor_is_secondary_storage_like(
        cluster_id=cluster_id,
        cluster_program=cluster_program,
        object_id=anchor_id,
        spec=spec,
        hint=hint,
        region_preference=region_preference,
    )
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    reject_reason_counts: dict[str, int] = {}
    reject_reason_counts_by_wall: dict[str, dict[str, int]] = {}

    if isinstance(world, dict):
        preflight_by_cluster = world.setdefault(
            "_anchor_candidate_preflight_by_cluster", {}
        )
        if isinstance(preflight_by_cluster, dict):
            preflight_by_cluster[cluster_id] = {
                "committed_region_ids": list(committed_region_ids),
                "committed_wall_side": committed_wall_side,
                "auto_window_forbidden_region_ids": list(
                    auto_window_forbidden_region_ids
                ),
                "forbidden_region_ids": list(forbidden_region_ids),
                "preferred_region_ids": list(preferred_region_ids),
            }

    def record_reject(reason: str, *, wall_side: str = "") -> None:
        if not reason:
            return
        reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1
        clean_wall = wall_side.strip().lower() or "unknown"
        wall_counts = reject_reason_counts_by_wall.setdefault(clean_wall, {})
        wall_counts[reason] = wall_counts.get(reason, 0) + 1

    def record_forbidden_reject(
        rect: tuple[int, int, int, int],
        forbidden_rows: Sequence[Mapping[str, Any]],
        *,
        wall_side: str = "",
    ) -> None:
        labels = _anchor_forbidden_reject_reason_labels(
            rect=rect,
            forbidden_regions=forbidden_rows,
        )
        if not labels:
            record_reject("forbidden_overlap", wall_side=wall_side)
            return
        for label in labels:
            record_reject(f"forbidden_overlap:{label}", wall_side=wall_side)

    def add_candidates_for_regions(
        candidate_region_bboxes: Sequence[tuple[int, int, int, int]],
        *,
        search_stage: str,
        score_adjustment: float,
        stage_preferred_wall_side: str,
        committed_region_missed: bool = False,
    ) -> None:
        for region_bbox in candidate_region_bboxes[:12]:
            min_x, min_y, max_x, max_y = region_bbox
            region_width = max_x - min_x
            region_height = max_y - min_y
            for rot in allowed_rotations:
                w_mm, h_mm = _rotate_dims_for_rot(length_mm, width_mm, int(rot))
                anchor_kind = _anchor_kind_for_candidate(
                    zone_assignment=zone_assignment,
                    room_bbox=world["room_bbox"],
                    region_bbox=region_bbox,
                )
                region_wall_side = _wall_side_for_anchor_kind(anchor_kind)
                if (
                    w_mm >= region_width
                    and h_mm >= region_height
                    and region_bbox not in {world["room_bbox"], placement_room_bbox}
                ):
                    record_reject(
                        "too_large_for_region",
                        wall_side=region_wall_side,
                    )
                    continue
                origin_positions = _wall_flush_origin_candidates(
                    region_bbox=region_bbox,
                    room_bbox=placement_room_bbox,
                    zone_assignment=zone_assignment,
                    placement_behavior=placement_behavior,
                    w_mm=w_mm,
                    h_mm=h_mm,
                    grid_mm=grid_mm,
                )
                if (
                    entry_relation == "buffer_entry_if_possible"
                    and _anchor_kind_is_entry(anchor_kind)
                ):
                    origin_positions.extend(
                        _entry_wall_flank_origin_candidates(
                            entry_bbox=region_bbox,
                            room_bbox=placement_room_bbox,
                            w_mm=w_mm,
                            h_mm=h_mm,
                            grid_mm=grid_mm,
                        )
                    )
                centers = _candidate_centers_for_region(
                    region_bbox=region_bbox,
                    room_bbox=placement_room_bbox,
                    zone_assignment=zone_assignment,
                    role_kind=str(
                        hint.get("role_kind") or cluster_program.get("tag") or ""
                    ),
                    primary=isinstance(primary_cluster, str)
                    and cluster_id == primary_cluster,
                )
                origin_positions.extend(
                    _snap_origin_inside_room(
                        x=int(round(cx - w_mm / 2.0)),
                        y=int(round(cy - h_mm / 2.0)),
                        w_mm=w_mm,
                        h_mm=h_mm,
                        room_bbox=placement_room_bbox,
                        grid_mm=grid_mm,
                    )
                    for cx, cy in centers
                )
                for x, y in origin_positions:
                    rect = (x, y, x + w_mm, y + h_mm)
                    wall_side = (
                        _wall_side_for_anchor_candidate(
                            anchor_kind=anchor_kind,
                            rect=rect,
                            room_bbox=placement_room_bbox,
                        )
                        or region_wall_side
                    )
                    probe_row = {
                        "cluster_id": cluster_id,
                        "object_id": anchor_id,
                        "blocks_floor_area": spec.get("blocks_floor_area", True),
                        "collision_layer": spec.get("collision_layer"),
                        "place_on": spec.get("place_on"),
                        "placement_participation": spec.get("placement_participation"),
                        "solver_participation": spec.get("solver_participation"),
                    }
                    if not _rect_inside_room_footprint(rect, world, row=probe_row):
                        record_reject(
                            _rect_room_footprint_reject_reason(
                                rect=rect,
                                world=world,
                                row=probe_row,
                            ),
                            wall_side=wall_side,
                        )
                        continue
                    if _anchor_forbidden_overlap_exceeds_limit(
                        rect=rect,
                        forbidden_regions=forbidden_regions,
                    ):
                        record_forbidden_reject(
                            rect,
                            forbidden_regions,
                            wall_side=wall_side,
                        )
                        continue
                    key = (x, y, int(rot) % 360)
                    if key in seen:
                        continue
                    seen.add(key)
                    front_token = (
                        spec.get("front")
                        if isinstance(spec.get("front"), str)
                        else "top"
                    )
                    front_world = _front_vector_from_rotation(front_token, int(rot))
                    desired_front = _desired_anchor_front_vector(
                        cluster_id=cluster_id,
                        object_id=anchor_id,
                        rect=rect,
                        room_bbox=placement_room_bbox,
                        relation_plan=relation_plan,
                        anchor_kind=anchor_kind,
                        placement_behavior=placement_behavior,
                    )
                    near_wall_fronts = _near_wall_inward_front_options(
                        rect=rect,
                        room_bbox=placement_room_bbox,
                    )
                    front_inward_match = bool(
                        near_wall_fronts
                        and _front_matches_any_required_direction(
                            front_world=front_world,
                            desired_fronts=near_wall_fronts,
                        )
                    )
                    long_axis_wall_parallel = _rect_long_axis_parallel_to_wall(
                        rect=rect,
                        wall_side=wall_side,
                    )
                    if (
                        requires_near_wall_inward
                        and near_wall_fronts
                        and not front_inward_match
                    ):
                        record_reject(
                            "near_wall_front_mismatch",
                            wall_side=wall_side,
                        )
                        continue
                    orientation_score = _front_alignment_score(
                        front_world=front_world,
                        desired_front=desired_front,
                    )
                    candidate = {
                        "cluster_id": cluster_id,
                        "object_id": anchor_id,
                        "x": x,
                        "y": y,
                        "rot": int(rot) % 360,
                        "w": w_mm,
                        "h": h_mm,
                        "rect": rect,
                        "anchor_kind": anchor_kind,
                        "front_world": {
                            "dx": front_world[0],
                            "dy": front_world[1],
                        },
                        "blocks_floor_area": spec.get("blocks_floor_area", True),
                        "collision_layer": spec.get("collision_layer"),
                        "place_on": deepcopy(spec.get("place_on"))
                        if isinstance(spec.get("place_on"), Mapping)
                        else None,
                        "placement_participation": spec.get("placement_participation"),
                        "solver_participation": spec.get("solver_participation"),
                        "desired_front_world": None
                        if desired_front is None
                        else {"dx": desired_front[0], "dy": desired_front[1]},
                        "orientation_score": orientation_score,
                        "wall_side": wall_side,
                        "front_inward_match": front_inward_match,
                        "long_axis_wall_parallel": long_axis_wall_parallel,
                        "search_stage": search_stage,
                        "committed_region_ids": list(committed_region_ids),
                        "committed_wall_side": committed_wall_side,
                        "fallback_preferred_region_ids": list(preferred_region_ids),
                        "committed_region_missed": committed_region_missed,
                        "committed_region_relaxed_after_strict_fail": (
                            committed_fallback_relaxed
                        ),
                    }
                    if not _object_rect_is_usable(rect, (), world, row=candidate):
                        record_reject(
                            _object_rect_unusable_reject_reason(
                                rect=rect,
                                world=world,
                                row=candidate,
                            ),
                            wall_side=wall_side,
                        )
                        continue
                    storage_wall_fit_score = _secondary_storage_wall_fit_score(
                        storage_like_anchor=storage_like_anchor,
                        wall_side=wall_side,
                        front_inward_match=front_inward_match,
                        long_axis_wall_parallel=long_axis_wall_parallel,
                    )
                    candidate["storage_wall_fit_score"] = storage_wall_fit_score
                    candidate["candidate_score"] = (
                        _anchor_candidate_score(
                            rect=rect,
                            room_bbox=world["room_bbox"],
                            region_bbox=region_bbox,
                            zone_assignment=zone_assignment,
                            anchor_kind=anchor_kind,
                            preferred_wall_side=stage_preferred_wall_side,
                            anchor_strength=anchor_strength,
                            placement_behavior=placement_behavior,
                            forbidden_regions=forbidden_regions,
                            cluster_id=cluster_id,
                            relation_plan=relation_plan,
                            entry_reference_bboxes=entry_reference_bboxes,
                        )
                        + orientation_score
                        + score_adjustment
                        + storage_wall_fit_score
                    )
                    candidates.append(candidate)

    committed_candidate_count = 0
    if committed_region_bboxes:
        add_candidates_for_regions(
            committed_region_bboxes,
            search_stage="committed_regions",
            score_adjustment=650.0
            if _anchor_should_prefer_committed_only(
                anchor_strength=anchor_strength,
                hint=hint,
                region_preference=region_preference,
                cluster_program=cluster_program,
                anchor_id=anchor_id,
            )
            else 250.0,
            stage_preferred_wall_side=committed_wall_side or preferred_wall_side,
        )
        committed_candidate_count = len(candidates)
    committed_only = _anchor_should_prefer_committed_only(
        anchor_strength=anchor_strength,
        hint=hint,
        region_preference=region_preference,
        cluster_program=cluster_program,
        anchor_id=anchor_id,
    ) and not _relation_plan_allows_committed_fallback(relation_plan)
    if not candidates or not committed_only:
        add_candidates_for_regions(
            region_bboxes,
            search_stage="fallback_preferred_regions"
            if committed_region_bboxes
            else "planned_regions",
            score_adjustment=-350.0 if committed_region_bboxes else 0.0,
            stage_preferred_wall_side=preferred_wall_side,
            committed_region_missed=bool(committed_region_bboxes)
            and committed_candidate_count == 0,
        )
    if not candidates:
        add_candidates_for_regions(
            _anchor_fallback_region_bboxes(
                world=world,
                existing_region_bboxes=region_bboxes,
            ),
            search_stage="expanded_wall_fallback",
            score_adjustment=-900.0,
            stage_preferred_wall_side=preferred_wall_side,
            committed_region_missed=bool(committed_region_bboxes),
        )
    fallback_candidate_count = sum(
        1
        for item in candidates
        if str(item.get("search_stage") or "")
        in {"fallback_preferred_regions", "expanded_wall_fallback"}
    )
    for candidate in candidates:
        candidate["committed_candidate_count"] = committed_candidate_count
        candidate["fallback_candidate_count"] = fallback_candidate_count
        if committed_region_bboxes and committed_candidate_count == 0:
            candidate["committed_region_missed"] = True
    candidates.sort(
        key=lambda item: (
            -float(item.get("candidate_score") or 0.0),
            item["y"],
            item["x"],
            item["rot"],
        )
    )
    if isinstance(world, dict):
        reject_counts_by_cluster = world.setdefault("_anchor_reject_reason_counts", {})
        if isinstance(reject_counts_by_cluster, dict):
            reject_counts_by_cluster[cluster_id] = dict(
                sorted(reject_reason_counts.items())
            )
        wall_options_by_cluster = world.setdefault(
            "_anchor_wall_options_by_cluster", {}
        )
        if isinstance(wall_options_by_cluster, dict):
            wall_options_by_cluster[cluster_id] = _anchor_wall_option_diagnostics(
                candidates=candidates,
                reject_reason_counts_by_wall=reject_reason_counts_by_wall,
            )
    return candidates[:OBJECT_LEVEL_MAX_ANCHOR_CANDIDATES_PER_CLUSTER]


def _anchor_should_prefer_committed_only(
    *,
    anchor_strength: str,
    hint: Mapping[str, Any],
    region_preference: Mapping[str, Any],
    cluster_program: Mapping[str, Any],
    anchor_id: str,
) -> bool:
    strength = anchor_strength.strip().lower()
    if strength in {"hard", "strong"}:
        return True
    scenario_role = str(
        hint.get("scenario_role") or region_preference.get("scenario_role") or ""
    ).strip()
    if scenario_role in {"primary", "secondary"}:
        return True
    spec = _object_spec(cluster_program, anchor_id)
    if not isinstance(spec, Mapping):
        return False
    return str(spec.get("preserve_level") or "").strip() in {
        "highest",
        "critical",
    } or bool(spec.get("protected"))


def _relation_plan_allows_committed_fallback(
    relation_plan: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(relation_plan, Mapping):
        return False
    return bool(
        relation_plan.get("_allow_committed_fallback_after_strict_fail")
        or relation_plan.get("allow_committed_fallback_after_strict_fail")
    )


def _anchor_candidate_generation_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    preflight: Mapping[str, Any] | None = None,
    reject_reason_counts: Mapping[str, Any] | None = None,
    wall_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage_counts: dict[str, int] = {}
    preflight = preflight if isinstance(preflight, Mapping) else {}
    committed_region_ids: list[str] = _string_sequence(
        preflight.get("committed_region_ids")
    )
    committed_wall_side = str(preflight.get("committed_wall_side") or "").strip()
    committed_region_missed = False
    committed_region_relaxed_after_strict_fail = False
    for row in rows:
        stage = str(row.get("search_stage") or "").strip()
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if not committed_region_ids:
            committed_region_ids = _string_sequence(row.get("committed_region_ids"))
        if not committed_wall_side:
            committed_wall_side = str(row.get("committed_wall_side") or "").strip()
        committed_region_missed = committed_region_missed or bool(
            row.get("committed_region_missed")
        )
        committed_region_relaxed_after_strict_fail = (
            committed_region_relaxed_after_strict_fail
            or bool(row.get("committed_region_relaxed_after_strict_fail"))
        )
    committed_count = stage_counts.get("committed_regions", 0)
    fallback_count = stage_counts.get(
        "fallback_preferred_regions", 0
    ) + stage_counts.get(
        "expanded_wall_fallback",
        0,
    )
    return {
        "candidate_count": len(rows),
        "committed_candidate_count": committed_count,
        "fallback_candidate_count": fallback_count,
        "committed_region_missed": committed_region_missed
        or (bool(committed_region_ids) and committed_count == 0 and fallback_count > 0),
        "committed_region_relaxed_after_strict_fail": (
            committed_region_relaxed_after_strict_fail
        ),
        "committed_region_ids": committed_region_ids,
        "committed_wall_side": committed_wall_side,
        "auto_window_forbidden_region_ids": _string_sequence(
            preflight.get("auto_window_forbidden_region_ids")
        ),
        "forbidden_region_ids": _string_sequence(preflight.get("forbidden_region_ids")),
        "preferred_region_ids": _string_sequence(preflight.get("preferred_region_ids")),
        "anchor_reject_reason_counts": dict(reject_reason_counts or {}),
        "wall_options": dict(wall_options or {}),
        "search_stage_counts": stage_counts,
    }


def _anchor_wall_option_diagnostics(
    *,
    candidates: Sequence[Mapping[str, Any]],
    reject_reason_counts_by_wall: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    by_wall: dict[str, dict[str, Any]] = {}
    for wall_side in ("top", "right", "bottom", "left", "interior", "unknown"):
        by_wall[wall_side] = {
            "valid_candidate_count": 0,
            "front_inward_candidate_count": 0,
            "wall_parallel_candidate_count": 0,
            "best_score": None,
            "best_candidate": None,
            "best_rehome_score": None,
            "best_rehome_candidate": None,
            "reject_reason_counts": dict(
                reject_reason_counts_by_wall.get(wall_side) or {}
            ),
        }
    for candidate in candidates:
        wall_side = str(candidate.get("wall_side") or "unknown").strip().lower()
        if wall_side not in by_wall:
            by_wall[wall_side] = {
                "valid_candidate_count": 0,
                "front_inward_candidate_count": 0,
                "wall_parallel_candidate_count": 0,
                "best_score": None,
                "best_candidate": None,
                "best_rehome_score": None,
                "best_rehome_candidate": None,
                "reject_reason_counts": dict(
                    reject_reason_counts_by_wall.get(wall_side) or {}
                ),
            }
        row = by_wall[wall_side]
        row["valid_candidate_count"] = int(row["valid_candidate_count"]) + 1
        if bool(candidate.get("front_inward_match")):
            row["front_inward_candidate_count"] = (
                int(row["front_inward_candidate_count"]) + 1
            )
        if bool(candidate.get("long_axis_wall_parallel")):
            row["wall_parallel_candidate_count"] = (
                int(row["wall_parallel_candidate_count"]) + 1
            )
        score = float(candidate.get("candidate_score") or 0.0)
        best_score = row.get("best_score")
        if best_score is None or score > float(best_score):
            row["best_score"] = score
            row["best_candidate"] = _anchor_wall_option_candidate_summary(candidate)
        if bool(candidate.get("front_inward_match")) and bool(
            candidate.get("long_axis_wall_parallel")
        ):
            best_rehome_score = row.get("best_rehome_score")
            if best_rehome_score is None or score > float(best_rehome_score):
                row["best_rehome_score"] = score
                row["best_rehome_candidate"] = _anchor_wall_option_candidate_summary(
                    candidate
                )
    return {wall: row for wall, row in by_wall.items() if _wall_option_has_signal(row)}


def _anchor_wall_option_candidate_summary(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cluster_id": str(candidate.get("cluster_id") or ""),
        "object_id": str(candidate.get("object_id") or ""),
        "rect": list(candidate.get("rect") or []),
        "rot": int(candidate.get("rot") or 0),
        "anchor_kind": str(candidate.get("anchor_kind") or ""),
        "wall_side": str(candidate.get("wall_side") or ""),
        "candidate_score": float(candidate.get("candidate_score") or 0.0),
        "orientation_score": float(candidate.get("orientation_score") or 0.0),
        "storage_wall_fit_score": float(candidate.get("storage_wall_fit_score") or 0.0),
        "front_inward_match": bool(candidate.get("front_inward_match")),
        "long_axis_wall_parallel": bool(candidate.get("long_axis_wall_parallel")),
        "front_world": deepcopy(candidate.get("front_world") or {}),
        "desired_front_world": deepcopy(candidate.get("desired_front_world") or {}),
        "search_stage": str(candidate.get("search_stage") or ""),
    }


def _wall_option_has_signal(row: Mapping[str, Any]) -> bool:
    return int(row.get("valid_candidate_count") or 0) > 0 or bool(
        row.get("reject_reason_counts")
    )


def _anchor_is_secondary_storage_like(
    *,
    cluster_id: str,
    cluster_program: Mapping[str, Any],
    object_id: str,
    spec: Mapping[str, Any],
    hint: Mapping[str, Any],
    region_preference: Mapping[str, Any],
) -> bool:
    role_kind = (
        str(
            hint.get("role_kind")
            or region_preference.get("role_kind")
            or cluster_program.get("tag")
            or ""
        )
        .strip()
        .lower()
    )
    scenario_role = (
        str(hint.get("scenario_role") or region_preference.get("scenario_role") or "")
        .strip()
        .lower()
    )
    if scenario_role == "primary" and role_kind not in {"storage", "wardrobe"}:
        return False
    if role_kind == "storage" or "storage" in cluster_id.lower():
        return True
    return _object_spec_is_visual_storage_like(object_id, spec)


def _wall_side_for_anchor_kind(anchor_kind: str) -> str:
    lowered = anchor_kind.strip().lower()
    if "left" in lowered:
        return "left"
    if "right" in lowered:
        return "right"
    if "top" in lowered:
        return "top"
    if "bottom" in lowered:
        return "bottom"
    return ""


def _wall_side_for_anchor_candidate(
    *,
    anchor_kind: str,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
) -> str:
    fronts = _dominant_wall_inward_front_options(
        rect=rect,
        room_bbox=room_bbox,
        tolerance_mm=OBJECT_LEVEL_NEAR_WALL_FRONT_ACCESS_TOLERANCE_MM,
    )
    if len(fronts) == 1:
        return _wall_side_from_inward_front(fronts[0])
    return _wall_side_for_anchor_kind(anchor_kind)


def _wall_side_from_inward_front(front: tuple[float, float]) -> str:
    dx, dy = front
    if dx > 0.5:
        return "left"
    if dx < -0.5:
        return "right"
    if dy > 0.5:
        return "top"
    if dy < -0.5:
        return "bottom"
    return ""


def _rect_long_axis_parallel_to_wall(
    *,
    rect: tuple[int, int, int, int],
    wall_side: str,
) -> bool:
    width = max(0, rect[2] - rect[0])
    height = max(0, rect[3] - rect[1])
    if wall_side in {"left", "right"}:
        return height >= width
    if wall_side in {"top", "bottom"}:
        return width >= height
    return False


def _secondary_storage_wall_fit_score(
    *,
    storage_like_anchor: bool,
    wall_side: str,
    front_inward_match: bool,
    long_axis_wall_parallel: bool,
) -> float:
    if not storage_like_anchor or wall_side not in {"top", "right", "bottom", "left"}:
        return 0.0
    if front_inward_match and long_axis_wall_parallel:
        return OBJECT_LEVEL_SECONDARY_STORAGE_WALL_FIT_BONUS
    if not front_inward_match:
        return -OBJECT_LEVEL_SECONDARY_STORAGE_WALL_FIT_BONUS
    if not long_axis_wall_parallel:
        return -OBJECT_LEVEL_SECONDARY_STORAGE_WALL_FIT_BONUS * 0.5
    return 0.0


def _entry_reference_bboxes_for_anchor(
    *,
    refs: Sequence[str],
    region_index: Mapping[str, tuple[int, int, int, int]],
    relation_plan: Mapping[str, Any] | None,
    room_model: Mapping[str, Any],
) -> list[tuple[int, int, int, int]]:
    bboxes: list[tuple[int, int, int, int]] = []
    for ref in refs:
        if not _entry_reference_region_id_allowed(ref):
            continue
        bboxes.extend(
            _region_bboxes_from_ref(ref, region_index, relation_plan, room_model)
        )
    return _dedupe_region_bboxes(bboxes)


def _entry_reference_region_id_allowed(region_id: str) -> bool:
    token = str(region_id or "").strip().lower()
    if not token or not ("entry" in token or "door" in token):
        return False
    return not any(part in token for part in ("corridor", "path", "center"))


def _anchor_forbidden_regions_for_cluster(
    *,
    cluster_id: str,
    object_id: str,
    world: Mapping[str, Any],
    resolved_region_bboxes: Sequence[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    if not resolved_region_bboxes:
        return []
    resolved_bbox_set = set(resolved_region_bboxes)
    raw_rows = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[int, int, int, int]]] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            continue
        if str(raw_row.get("cluster_id") or "") != cluster_id:
            continue
        forbidden_object_id = str(raw_row.get("object_id") or "").strip()
        if forbidden_object_id and forbidden_object_id != object_id:
            continue
        bbox = _rect_tuple(raw_row.get("bbox"))
        if bbox is None or bbox not in resolved_bbox_set:
            continue
        region_id = str(raw_row.get("region_id") or "")
        key = (forbidden_object_id, region_id, bbox)
        if key in seen:
            continue
        seen.add(key)
        row = dict(raw_row)
        row["bbox"] = bbox
        rows.append(row)
    if rows:
        return rows
    return [
        {
            "cluster_id": cluster_id,
            "region_id": "",
            "bbox": bbox,
            **_default_cluster_forbidden_region_policy(),
        }
        for bbox in resolved_region_bboxes
    ]


def _anchor_fallback_region_bboxes(
    *,
    world: Mapping[str, Any],
    existing_region_bboxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    region_index = (
        world.get("region_index")
        if isinstance(world.get("region_index"), Mapping)
        else {}
    )
    existing = set(existing_region_bboxes)
    scored: list[tuple[int, str, tuple[int, int, int, int]]] = []
    for raw_region_id, bbox in region_index.items():
        region_id = str(raw_region_id or "")
        rect = _rect_tuple(bbox)
        if rect is None or rect in existing:
            continue
        if not _region_id_is_anchor_fallback_region(region_id):
            continue
        scored.append((_anchor_fallback_region_priority(region_id), region_id, rect))
    scored.sort(key=lambda item: (item[0], item[1]))
    fallback_bboxes = _dedupe_region_bboxes([item[2] for item in scored])
    if fallback_bboxes:
        return fallback_bboxes
    room_bbox = _rect_tuple(world.get("placement_room_bbox")) or _rect_tuple(
        world.get("room_bbox")
    )
    return [] if room_bbox is None else [room_bbox]


def _region_id_is_anchor_fallback_region(region_id: str) -> bool:
    token = region_id.strip().lower()
    if not token:
        return False
    if any(part in token for part in ("door", "entry", "corridor", "clearance")):
        return False
    if any(part in token for part in ("center", "floating")):
        return False
    return any(
        part in token
        for part in (
            "_usable_",
            "anchor",
            "edge",
            "perimeter",
            "privacy",
            "private",
            "wall",
            "window_side",
        )
    )


def _anchor_fallback_region_priority(region_id: str) -> int:
    token = region_id.strip().lower()
    if "_usable_" in token and "anchor" in token:
        return 0
    if token in {
        "top_wall_zone",
        "right_wall_zone",
        "bottom_wall_zone",
        "left_wall_zone",
    }:
        return 1
    if "edge" in token or "perimeter" in token:
        return 2
    if "privacy" in token or "private" in token:
        return 3
    if "window_side" in token:
        return 4
    return 5


def _wall_flush_origin_candidates(
    *,
    region_bbox: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    zone_assignment: str,
    placement_behavior: Mapping[str, Any] | None,
    w_mm: int,
    h_mm: int,
    grid_mm: int,
) -> list[tuple[int, int]]:
    token = str(zone_assignment).lower()
    wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
    wall_backed_tokens = (
        "daylight",
        "edge",
        "focal",
        "private",
        "privacy",
        "quiet",
        "storage",
        "wall",
        "window",
    )
    if wall_backing not in {"preferred", "required"} and not any(
        part in token for part in wall_backed_tokens
    ):
        return []

    wall = _nearest_wall_name_for_region(region_bbox=region_bbox, room_bbox=room_bbox)
    min_x, min_y, max_x, max_y = region_bbox
    room_min_x, room_min_y, room_max_x, room_max_y = room_bbox
    room_cx = int(round((room_min_x + room_max_x) / 2.0))
    room_cy = int(round((room_min_y + room_max_y) / 2.0))
    origins: list[tuple[int, int]] = []

    def append_origin(x: int, y: int) -> None:
        origins.append(
            _snap_origin_inside_room(
                x=x,
                y=y,
                w_mm=w_mm,
                h_mm=h_mm,
                room_bbox=room_bbox,
                grid_mm=grid_mm,
            )
        )

    if wall in {"top_wall", "bottom_wall"}:
        y = room_min_y if wall == "top_wall" else room_max_y - h_mm
        span_min = max(room_min_x, min_x)
        span_max = min(room_max_x - w_mm, max_x - w_mm)
        if span_max < span_min:
            span_min = room_min_x
            span_max = room_max_x - w_mm
        for x in (span_min, (span_min + span_max) // 2, span_max, room_cx - w_mm // 2):
            append_origin(int(x), int(y))
    else:
        x = room_min_x if wall == "left_wall" else room_max_x - w_mm
        span_min = max(room_min_y, min_y)
        span_max = min(room_max_y - h_mm, max_y - h_mm)
        if span_max < span_min:
            span_min = room_min_y
            span_max = room_max_y - h_mm
        for y in (span_min, (span_min + span_max) // 2, span_max, room_cy - h_mm // 2):
            append_origin(int(x), int(y))

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for origin in origins:
        if origin in seen:
            continue
        seen.add(origin)
        out.append(origin)
    return out


def _entry_wall_flank_origin_candidates(
    *,
    entry_bbox: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    w_mm: int,
    h_mm: int,
    grid_mm: int,
) -> list[tuple[int, int]]:
    wall = _nearest_wall_name_for_region(region_bbox=entry_bbox, room_bbox=room_bbox)
    min_x, min_y, max_x, max_y = entry_bbox
    room_min_x, room_min_y, room_max_x, room_max_y = room_bbox
    margin = max(1, int(grid_mm))
    origins: list[tuple[int, int]] = []

    def append_origin(x: int, y: int) -> None:
        origins.append(
            _snap_origin_inside_room(
                x=x,
                y=y,
                w_mm=w_mm,
                h_mm=h_mm,
                room_bbox=room_bbox,
                grid_mm=grid_mm,
            )
        )

    if wall in {"left_wall", "right_wall"}:
        x = room_min_x if wall == "left_wall" else room_max_x - w_mm
        append_origin(x, min_y - h_mm - margin)
        append_origin(x, max_y + margin)
        append_origin(x, min_y - h_mm - margin * 3)
        append_origin(x, max_y + margin * 3)
    else:
        y = room_min_y if wall == "top_wall" else room_max_y - h_mm
        append_origin(min_x - w_mm - margin, y)
        append_origin(max_x + margin, y)
        append_origin(min_x - w_mm - margin * 3, y)
        append_origin(max_x + margin * 3, y)

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for origin in origins:
        if origin in seen:
            continue
        seen.add(origin)
        out.append(origin)
    return out


def _candidate_centers_for_region(
    *,
    region_bbox: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    zone_assignment: str,
    role_kind: str,
    primary: bool,
) -> list[tuple[int, int]]:
    min_x, min_y, max_x, max_y = region_bbox
    cx = int(round((min_x + max_x) / 2.0))
    cy = int(round((min_y + max_y) / 2.0))
    room_cx = int(round((room_bbox[0] + room_bbox[2]) / 2.0))
    room_cy = int(round((room_bbox[1] + room_bbox[3]) / 2.0))
    centers = [(cx, cy)]
    token = str(zone_assignment).lower()
    if "focal" in token or "wall" in token or "edge" in token or "storage" in token:
        centers.extend(
            [
                (cx, cy),
                (room_cx, cy),
                (cx, room_cy),
                (int(round(min_x + (max_x - min_x) * 0.25)), cy),
                (int(round(min_x + (max_x - min_x) * 0.75)), cy),
            ]
        )
    elif "daylight" in token or "window" in token:
        centers.extend(
            [
                (cx, cy),
                (room_cx, cy),
                (cx, min(cy + 250, room_bbox[3] - 250)),
                (int(round(min_x + (max_x - min_x) * 0.25)), cy),
                (int(round(min_x + (max_x - min_x) * 0.75)), cy),
            ]
        )
    elif "center" in token or "floating" in token:
        centers.extend(
            [
                (room_cx, room_cy),
                (cx, cy),
                (room_cx, cy),
                (int(round(min_x + (max_x - min_x) * 0.33)), room_cy),
                (int(round(min_x + (max_x - min_x) * 0.67)), room_cy),
            ]
        )
    if primary:
        centers.insert(0, (cx, cy))
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in centers:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:8]


def _anchor_kind_for_candidate(
    *,
    zone_assignment: str,
    room_bbox: tuple[int, int, int, int],
    region_bbox: tuple[int, int, int, int],
) -> str:
    token = str(zone_assignment).lower()
    if "center" in token or "floating" in token:
        return "center"
    if "daylight" in token or "window" in token:
        return "window_side"
    if "entry" in token or "door" in token:
        return "entry_side"
    # infer closest wall from region
    return _nearest_wall_name_for_region(region_bbox=region_bbox, room_bbox=room_bbox)


def _nearest_wall_name_for_region(
    *,
    region_bbox: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
) -> str:
    region_width = max(0, region_bbox[2] - region_bbox[0])
    region_height = max(0, region_bbox[3] - region_bbox[1])
    distances = {
        "top_wall": abs(region_bbox[1] - room_bbox[1]),
        "bottom_wall": abs(room_bbox[3] - region_bbox[3]),
        "left_wall": abs(region_bbox[0] - room_bbox[0]),
        "right_wall": abs(room_bbox[2] - region_bbox[2]),
    }
    horizontal_wall = (
        "top_wall"
        if distances["top_wall"] <= distances["bottom_wall"]
        else "bottom_wall"
    )
    vertical_wall = (
        "left_wall"
        if distances["left_wall"] <= distances["right_wall"]
        else "right_wall"
    )
    if region_width <= max(700, int(round(region_height * 0.45))) and distances[
        vertical_wall
    ] <= max(900, distances[horizontal_wall] + 900):
        return vertical_wall
    if region_height <= max(700, int(round(region_width * 0.45))) and distances[
        horizontal_wall
    ] <= max(900, distances[vertical_wall] + 900):
        return horizontal_wall
    return min(distances.items(), key=lambda item: item[1])[0]


def _anchor_candidate_score(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    region_bbox: tuple[int, int, int, int],
    zone_assignment: str,
    anchor_kind: str,
    preferred_wall_side: str,
    anchor_strength: str,
    placement_behavior: Mapping[str, Any] | None,
    forbidden_regions: Sequence[Mapping[str, Any]],
    cluster_id: str,
    relation_plan: Mapping[str, Any] | None,
    entry_reference_bboxes: Sequence[tuple[int, int, int, int]] = (),
) -> float:
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    rcx = (region_bbox[0] + region_bbox[2]) / 2.0
    rcy = (region_bbox[1] + region_bbox[3]) / 2.0
    room_cx = (room_bbox[0] + room_bbox[2]) / 2.0
    room_cy = (room_bbox[1] + room_bbox[3]) / 2.0
    dist_region = math.hypot(cx - rcx, cy - rcy)
    dist_center = math.hypot(cx - room_cx, cy - room_cy)
    token = str(zone_assignment).lower()
    score = 10000.0 - dist_region
    if "center" in token or "floating" in token:
        score += 1200.0 - dist_center
    elif any(part in token for part in ("focal", "wall", "edge", "storage")):
        score += dist_center * 0.25
    wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
    front_space = _placement_behavior_value(placement_behavior, "front_space")
    pose_flexibility = _placement_behavior_value(
        placement_behavior,
        "pose_flexibility",
    )
    if wall_backing in {"preferred", "required"}:
        if anchor_kind in {
            "bottom_wall",
            "left_wall",
            "right_wall",
            "top_wall",
            "window_side",
        }:
            score += 700.0 if wall_backing == "required" else 350.0
        else:
            score -= 700.0 if wall_backing == "required" else 250.0
    if front_space == "required":
        score += 180.0
    elif front_space == "preferred":
        score += 80.0
    if pose_flexibility == "low":
        score += 120.0
    if cluster_id == _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    ):
        score += 300.0
    preferred = preferred_wall_side.strip().lower()
    if preferred:
        strength = anchor_strength.strip().lower()
        match_bonus = {"hard": 3600.0, "strong": 2600.0}.get(strength, 1200.0)
        mismatch_penalty = {"hard": 5200.0, "strong": 3200.0}.get(strength, 1200.0)
        if _anchor_kind_matches_preferred_side(anchor_kind, preferred):
            score += match_bonus
        else:
            score -= mismatch_penalty
    if forbidden_regions:
        excess_overlap = _forbidden_region_excess_overlap(
            rect=rect,
            forbidden_regions=forbidden_regions,
        )
        if excess_overlap > 0.0:
            penalty = {"hard": 4200.0, "strong": 2600.0}.get(
                anchor_strength.strip().lower(),
                1200.0,
            )
            score -= min(1.5, excess_overlap) * penalty
    if _cluster_entry_relation_for_plan(relation_plan, cluster_id) == (
        "buffer_entry_if_possible"
    ):
        if _anchor_kind_is_entry(anchor_kind):
            score += 2400.0
        entry_distance = _rect_center_distance_to_nearest_bbox(
            rect,
            entry_reference_bboxes,
        )
        if entry_distance is None and ("entry" in token or "door" in token):
            entry_distance = dist_region
        if entry_distance is not None:
            preferred_distance = OBJECT_LEVEL_ENTRY_SUPPORT_PREFERRED_DISTANCE_MM
            if entry_distance <= preferred_distance:
                score += 2800.0 + (preferred_distance - entry_distance) * 1.2
            else:
                score -= min(4500.0, (entry_distance - preferred_distance) * 1.8)
    return score


def _rect_center_distance_to_nearest_bbox(
    rect: tuple[int, int, int, int],
    bboxes: Sequence[tuple[int, int, int, int]],
) -> float | None:
    if not bboxes:
        return None
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    return min(
        math.hypot(cx - (bbox[0] + bbox[2]) / 2.0, cy - (bbox[1] + bbox[3]) / 2.0)
        for bbox in bboxes
    )


def _forbidden_region_excess_overlap(
    *,
    rect: tuple[int, int, int, int],
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> float:
    excess_overlap = 0.0
    for forbidden in forbidden_regions:
        bbox = _rect_tuple(forbidden.get("bbox"))
        if bbox is None:
            continue
        max_overlap = float(forbidden.get("max_overlap_ratio") or 0.0)
        excess_overlap += max(0.0, _rect_overlap_ratio(rect, bbox) - max_overlap)
    return excess_overlap


def _anchor_forbidden_overlap_exceeds_limit(
    *,
    rect: tuple[int, int, int, int],
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> bool:
    if not forbidden_regions:
        return False
    return any(
        _forbidden_region_rejects_anchor_candidate(rect=rect, forbidden=forbidden)
        for forbidden in forbidden_regions
    )


def _anchor_forbidden_reject_reason_labels(
    *,
    rect: tuple[int, int, int, int],
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> list[str]:
    labels: list[str] = []
    for forbidden in forbidden_regions:
        if not _forbidden_region_rejects_anchor_candidate(
            rect=rect,
            forbidden=forbidden,
        ):
            continue
        labels.append(_forbidden_region_reason_label(forbidden))
    return sorted(set(labels))


def _forbidden_region_reason_label(forbidden: Mapping[str, Any]) -> str:
    region_id = str(forbidden.get("region_id") or "").strip()
    zone_type = str(forbidden.get("zone_type") or "").strip()
    if _region_id_is_window_clearance(region_id) or zone_type == "window_clearance":
        return "window_clearance"
    if (
        _region_id_is_daylight_quality_region(region_id)
        or zone_type == "daylight_quality_blocking"
        or zone_type == "daylight_quality_soft"
    ):
        return "daylight_quality"
    if "corridor" in region_id or "corridor" in zone_type:
        return "primary_circulation_corridor"
    if "entry" in region_id or "entry" in zone_type:
        return "entry_clearance"
    if region_id:
        return region_id
    if zone_type:
        return zone_type
    return "cluster_forbidden_region"


def _forbidden_region_rejects_anchor_candidate(
    *,
    rect: tuple[int, int, int, int],
    forbidden: Mapping[str, Any],
) -> bool:
    bbox = _rect_tuple(forbidden.get("bbox"))
    if bbox is None:
        return False
    max_overlap = float(forbidden.get("max_overlap_ratio") or 0.0)
    ratio = _rect_overlap_ratio(rect, bbox)
    if ratio <= max_overlap + 1e-9:
        return False
    enforcement = str(forbidden.get("enforcement") or "hard").strip().lower()
    if enforcement == "hard":
        return True
    if enforcement == "hard_soft":
        return ratio > _hard_soft_anchor_reject_ratio(forbidden) + 1e-9
    return False


def _hard_soft_anchor_reject_ratio(forbidden: Mapping[str, Any]) -> float:
    max_overlap = float(forbidden.get("max_overlap_ratio") or 0.0)
    return max(OBJECT_LEVEL_HARD_SOFT_ANCHOR_REJECT_RATIO, max_overlap)


def _anchor_kind_matches_preferred_side(anchor_kind: str, preferred: str) -> bool:
    actual = anchor_kind.strip().lower()
    target = preferred.strip().lower()
    if not target:
        return True
    if target == "center":
        return actual == "center"
    if target == "window_side":
        return actual == "window_side"
    return actual == target


def _desired_anchor_front_vector(
    *,
    cluster_id: str,
    object_id: str,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    relation_plan: Mapping[str, Any] | None,
    anchor_kind: str,
    placement_behavior: Mapping[str, Any] | None = None,
) -> tuple[float, float] | None:
    intents = _object_level_orientation_intents(
        cluster_id=cluster_id,
        object_id=object_id,
        relation_plan=relation_plan,
        include_cluster=True,
    )
    contact_front = _wall_contact_inward_front(rect=rect, room_bbox=room_bbox)
    if contact_front is not None:
        return contact_front
    if anchor_kind in {
        "bottom_wall",
        "entry_side",
        "left_wall",
        "right_wall",
        "top_wall",
        "window_side",
    }:
        wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
        front_space = _placement_behavior_value(placement_behavior, "front_space")
        if (
            {"back_to_wall", "front_to_open_space"} & intents
            or wall_backing in {"preferred", "required"}
            or front_space in {"preferred", "required"}
        ):
            return _nearest_wall_inward_front(rect=rect, room_bbox=room_bbox)
    return None


def _object_level_orientation_intents(
    *,
    cluster_id: str,
    object_id: str,
    relation_plan: Mapping[str, Any] | None,
    include_cluster: bool,
) -> set[str]:
    if not isinstance(relation_plan, Mapping):
        return set()
    intents: set[str] = set()
    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        if str(row.get("object_id") or "").strip() != object_id:
            continue
        intents.update(_normalized_text_set(row.get("intents")))
    if include_cluster:
        for row in relation_plan.get("cluster_orientations") or []:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("cluster_id") or "").strip() == cluster_id:
                intents.update(_normalized_text_set(row.get("intents")))
        intents.update(
            _placement_behavior_orientation_intents(
                _placement_behavior_for_cluster(relation_plan, cluster_id)
            )
        )
    return intents


def _orientation_intents_require_near_wall_inward(intents: set[str]) -> bool:
    return bool(
        intents
        & {
            "access_to_open_space",
            "back_to_wall",
            "front_to_open_space",
        }
    )


def _anchor_requires_near_wall_inward_contract(
    *,
    cluster_id: str,
    object_id: str,
    cluster_program: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> bool:
    if not _object_requires_front_access(cluster_program, object_id):
        return False
    intents = _object_level_orientation_intents(
        cluster_id=cluster_id,
        object_id=object_id,
        relation_plan=relation_plan,
        include_cluster=True,
    )
    return _orientation_intents_require_near_wall_inward(intents)


def _row_requires_near_wall_inward_contract(
    *,
    row: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> bool:
    if not _is_cluster_anchor_row(row):
        return False
    if not _row_requires_front_access_zone(row):
        return False
    cluster_id = str(row.get("cluster_id") or "")
    object_id = str(row.get("object_id") or "")
    if not cluster_id or not object_id:
        return False
    intents = _object_level_orientation_intents(
        cluster_id=cluster_id,
        object_id=object_id,
        relation_plan=relation_plan,
        include_cluster=True,
    )
    return _orientation_intents_require_near_wall_inward(intents)


def _object_level_target_cluster_id(
    *,
    cluster_id: str,
    object_id: str,
    relation_plan: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(relation_plan, Mapping):
        return None
    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        if str(row.get("object_id") or "").strip() != object_id:
            continue
        target = str(row.get("target_cluster_id") or "").strip()
        if target:
            return target
    return _cluster_face_target_id(dict(relation_plan), cluster_id)


def _nearest_wall_inward_front(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
) -> tuple[float, float]:
    distances = [
        (abs(rect[0] - room_bbox[0]), (1.0, 0.0)),
        (abs(room_bbox[2] - rect[2]), (-1.0, 0.0)),
        (abs(rect[1] - room_bbox[1]), (0.0, 1.0)),
        (abs(room_bbox[3] - rect[3]), (0.0, -1.0)),
    ]
    return min(distances, key=lambda item: item[0])[1]


def _wall_contact_inward_front(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    tolerance_mm: int = OBJECT_LEVEL_WALL_CONTACT_TOLERANCE_MM,
) -> tuple[float, float] | None:
    fronts = _dominant_wall_inward_front_options(
        rect=rect,
        room_bbox=room_bbox,
        tolerance_mm=tolerance_mm,
    )
    if not fronts:
        return None
    return fronts[0]


def _wall_contact_inward_front_options(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    tolerance_mm: int = OBJECT_LEVEL_WALL_CONTACT_TOLERANCE_MM,
) -> list[tuple[float, float]]:
    return _dominant_wall_inward_front_options(
        rect=rect,
        room_bbox=room_bbox,
        tolerance_mm=tolerance_mm,
    )


def _near_wall_inward_front_options(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    tolerance_mm: int = OBJECT_LEVEL_NEAR_WALL_FRONT_ACCESS_TOLERANCE_MM,
) -> list[tuple[float, float]]:
    return _dominant_wall_inward_front_options(
        rect=rect,
        room_bbox=room_bbox,
        tolerance_mm=tolerance_mm,
    )


def _dominant_wall_inward_front_options(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    tolerance_mm: int,
) -> list[tuple[float, float]]:
    width = max(0, rect[2] - rect[0])
    height = max(0, rect[3] - rect[1])
    contacts = [
        (abs(rect[0] - room_bbox[0]), height, (1.0, 0.0)),
        (abs(room_bbox[2] - rect[2]), height, (-1.0, 0.0)),
        (abs(rect[1] - room_bbox[1]), width, (0.0, 1.0)),
        (abs(room_bbox[3] - rect[3]), width, (0.0, -1.0)),
    ]
    candidates = [
        (span, front)
        for distance, span, front in contacts
        if distance <= max(0, int(tolerance_mm))
    ]
    if not candidates:
        return []
    max_span = max(span for span, _front in candidates)
    return [front for span, front in candidates if span == max_span]


def _front_alignment_score(
    *,
    front_world: tuple[float, float],
    desired_front: tuple[float, float] | None,
) -> float:
    if desired_front is None:
        return 0.0
    dot = _dot_normalized(front_world, desired_front)
    return dot * 1400.0


def _front_matches_required_direction(
    *,
    front_world: tuple[float, float] | None,
    desired_front: tuple[float, float] | None,
    min_dot: float = OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT,
) -> bool:
    if front_world is None or desired_front is None:
        return False
    return _dot_normalized(front_world, desired_front) >= float(min_dot)


def _front_matches_any_required_direction(
    *,
    front_world: tuple[float, float],
    desired_fronts: Sequence[tuple[float, float]],
    min_dot: float = OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT,
) -> bool:
    return any(
        _front_matches_required_direction(
            front_world=front_world,
            desired_front=desired_front,
            min_dot=min_dot,
        )
        for desired_front in desired_fronts
    )


def _best_aligned_required_front(
    *,
    front_world: tuple[float, float],
    desired_fronts: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    if not desired_fronts:
        return None
    return max(
        desired_fronts,
        key=lambda desired_front: _dot_normalized(front_world, desired_front),
    )


def _dot_normalized(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    left_norm = math.hypot(left[0], left[1])
    right_norm = math.hypot(right[0], right[1])
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return (left[0] / left_norm) * (right[0] / right_norm) + (left[1] / left_norm) * (
        right[1] / right_norm
    )


def _search_anchor_solutions(
    *,
    anchor_order: Sequence[str],
    anchor_candidates_by_cluster: Mapping[str, Sequence[dict[str, Any]]],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    max_solutions: int,
) -> Sequence[dict[str, Any]]:
    solutions: list[dict[str, Any]] = []
    visited_leaf_count = 0
    max_leaf_count = max(12000, int(max_solutions) * 512)
    protected_regions = (
        world.get("protected_regions")
        if isinstance(world.get("protected_regions"), Sequence)
        else []
    )
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )

    def rec(
        index: int,
        chosen: list[dict[str, Any]],
        dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]],
    ) -> None:
        nonlocal visited_leaf_count
        if visited_leaf_count >= max_leaf_count:
            return
        if index >= len(anchor_order):
            visited_leaf_count += 1
            placed = {row["cluster_id"]: row for row in chosen}
            score = _anchor_pair_solution_score(placed, relation_plan) + sum(
                float(row.get("candidate_score") or 0.0) for row in chosen
            )
            score -= _object_level_protected_region_penalty(
                _object_level_protected_region_issues(
                    placed_objects=chosen,
                    protected_regions=protected_regions,
                )
            )
            score -= _object_level_protected_region_penalty(
                _object_level_cluster_forbidden_region_issues(
                    placed_objects=chosen,
                    forbidden_regions=forbidden_regions,
                )
            )
            solutions.append(
                {
                    "anchor_solution": placed,
                    "anchor_score": score,
                    "dropped_inventory_by_cluster": deepcopy(
                        {
                            cluster_id: rows
                            for cluster_id, rows in dropped_inventory_by_cluster.items()
                            if rows
                        }
                    ),
                }
            )
            _trim_anchor_solution_pool(solutions, max_solutions=max_solutions)
            return
        cluster_id = anchor_order[index]
        cluster_program = world["clusters_by_id"][cluster_id]
        if _cluster_is_solver_trial_optional(cluster_program):
            drop_records = _optional_trial_cluster_drop_records(cluster_program)
            dropped_inventory_by_cluster.setdefault(cluster_id, []).extend(drop_records)
            rec(index + 1, chosen, dropped_inventory_by_cluster)
            if drop_records:
                del dropped_inventory_by_cluster[cluster_id][-len(drop_records) :]
        for candidate in anchor_candidates_by_cluster.get(cluster_id, []):
            rect = candidate["rect"]
            if any(_rects_overlap(rect, row["rect"]) for row in chosen):
                continue
            chosen.append(candidate)
            rec(index + 1, chosen, dropped_inventory_by_cluster)
            chosen.pop()

    rec(0, [], {})
    return solutions


def _anchor_pair_compatibility_report(
    *,
    anchor_order: Sequence[str],
    anchor_candidates_by_cluster: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left_cluster_id in enumerate(anchor_order):
        left_candidates = anchor_candidates_by_cluster.get(left_cluster_id, [])
        for right_cluster_id in anchor_order[left_index + 1 :]:
            right_candidates = anchor_candidates_by_cluster.get(right_cluster_id, [])
            candidate_pairs = len(left_candidates) * len(right_candidates)
            non_overlap_pairs = 0
            for left_candidate in left_candidates:
                left_rect = left_candidate.get("rect")
                if not isinstance(left_rect, tuple) or len(left_rect) != 4:
                    continue
                for right_candidate in right_candidates:
                    right_rect = right_candidate.get("rect")
                    if not isinstance(right_rect, tuple) or len(right_rect) != 4:
                        continue
                    if not _rects_overlap(left_rect, right_rect):
                        non_overlap_pairs += 1
            rows.append(
                {
                    "cluster_a": str(left_cluster_id),
                    "cluster_b": str(right_cluster_id),
                    "candidate_pairs": candidate_pairs,
                    "non_overlap_pairs": non_overlap_pairs,
                    "overlap_pairs": max(0, candidate_pairs - non_overlap_pairs),
                }
            )
    return rows


def _trim_anchor_solution_pool(
    solutions: list[dict[str, Any]],
    *,
    max_solutions: int,
) -> None:
    if len(solutions) <= max_solutions:
        solutions.sort(
            key=lambda item: float(item.get("anchor_score") or 0.0),
            reverse=True,
        )
        return

    ranked = sorted(
        solutions,
        key=lambda item: float(item.get("anchor_score") or 0.0),
        reverse=True,
    )
    kept = ranked[:max_solutions]
    trial_candidates = [
        item for item in ranked if _anchor_solution_has_dropped_trial(item)
    ]
    if trial_candidates and not any(
        _anchor_solution_has_dropped_trial(item) for item in kept
    ):
        kept[-1] = trial_candidates[0]
    solutions[:] = kept


def _anchor_solution_has_dropped_trial(solution: Mapping[str, Any]) -> bool:
    dropped = solution.get("dropped_inventory_by_cluster")
    if not isinstance(dropped, Mapping):
        return False
    return any(
        isinstance(rows, Sequence) and not isinstance(rows, str) and bool(rows)
        for rows in dropped.values()
    )


def _optional_trial_cluster_drop_records(
    cluster_program: Mapping[str, Any],
) -> list[dict[str, str]]:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    members = [
        item
        for item in object_program.get("members")
        or cluster_program.get("members")
        or []
        if isinstance(item, str) and item.strip()
    ]
    if not members:
        anchor_id = _cluster_program_dominant_anchor(cluster_program)
        if anchor_id is not None:
            members = [anchor_id]
    return [
        {
            "object_id": object_id,
            "reason": "optional_trial_anchor_cluster_not_placed",
        }
        for object_id in members
    ]


def _anchor_pair_solution_score(
    anchor_solution: Mapping[str, dict[str, Any]],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    object_viewing_pairs = _object_level_required_viewing_pairs(relation_plan)
    if object_viewing_pairs:
        return _anchor_object_viewing_pair_solution_score(
            anchor_solution=anchor_solution,
            object_viewing_pairs=object_viewing_pairs,
        )
    primary = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    secondary = _extract_layout_secondary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    if (
        not primary
        or not secondary
        or primary not in anchor_solution
        or secondary not in anchor_solution
    ):
        return 0.0
    a = anchor_solution[primary]["rect"]
    b = anchor_solution[secondary]["rect"]
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    distance = math.hypot(ac[0] - bc[0], ac[1] - bc[1])
    desired = 2400.0
    pair_score = 1800.0 - abs(distance - desired)
    if a[1] < b[1]:
        pair_score += 120.0
    pair_score += _scenario_anchor_pair_score(
        primary_anchor=anchor_solution[primary],
        secondary_anchor=anchor_solution[secondary],
        relation_plan=relation_plan,
    )
    pair_score += _anchor_pair_orientation_score(
        left_cluster_id=primary,
        right_cluster_id=secondary,
        left_anchor=anchor_solution[primary],
        right_anchor=anchor_solution[secondary],
        relation_plan=relation_plan,
    )
    return pair_score


def _anchor_object_viewing_pair_solution_score(
    *,
    anchor_solution: Mapping[str, dict[str, Any]],
    object_viewing_pairs: Sequence[Mapping[str, Any]],
) -> float:
    score = 0.0
    for pair in object_viewing_pairs:
        viewer_cluster_id = str(pair.get("viewer_cluster_id") or "")
        target_cluster_id = str(pair.get("target_cluster_id") or "")
        viewer = anchor_solution.get(viewer_cluster_id)
        target = anchor_solution.get(target_cluster_id)
        if viewer is None or target is None:
            continue
        if not _anchor_matches_viewing_pair_object(
            viewer, str(pair.get("viewer_object_id") or "")
        ) or not _anchor_matches_viewing_pair_object(
            target, str(pair.get("target_object_id") or "")
        ):
            continue
        score += _viewing_pair_anchor_score(viewer, target, pair)
        score -= _non_participant_viewing_target_penalty(
            anchor_solution=anchor_solution,
            viewer=viewer,
            target=target,
        )
    return score


def _anchor_matches_viewing_pair_object(
    anchor: Mapping[str, Any],
    object_id: str,
) -> bool:
    if not object_id:
        return True
    return str(anchor.get("object_id") or "") == object_id


def _viewing_pair_relation(pair: Mapping[str, Any]) -> str:
    relation = str(pair.get("relation") or "face_each_other").strip().lower()
    if relation == "mutual_face":
        return "face_each_other"
    return relation


def _view_target_viewer_min_dot(
    pair: Mapping[str, Any],
    viewer: Mapping[str, Any],
) -> float:
    tokens = _object_type_tokens(
        pair.get("viewer_object_type"),
        pair.get("viewer_object_id"),
        viewer.get("object_id"),
        viewer.get("category"),
        viewer.get("anchor_kind"),
    )
    if tokens & VIEW_TARGET_BED_VIEWER_TYPES:
        return OBJECT_LEVEL_VIEW_TARGET_BED_VIEWER_MIN_DOT
    return OBJECT_LEVEL_VIEW_TARGET_SEATED_VIEWER_MIN_DOT


def _object_type_tokens(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        normalized = str(value or "").strip().lower().replace("-", "_")
        if not normalized:
            continue
        tokens.add(normalized)
        tokens.update(part for part in normalized.split("_") if part)
    return tokens


def _viewing_pair_anchor_score(
    viewer: Mapping[str, Any],
    target: Mapping[str, Any],
    pair: Mapping[str, Any],
) -> float:
    viewer_rect = viewer.get("rect")
    target_rect = target.get("rect")
    if not (
        isinstance(viewer_rect, tuple)
        and len(viewer_rect) == 4
        and isinstance(target_rect, tuple)
        and len(target_rect) == 4
    ):
        return 0.0
    viewer_to_target = _vector_between_rect_centers(viewer_rect, target_rect)
    target_to_viewer = _vector_between_rect_centers(target_rect, viewer_rect)
    viewer_dot = _dot_for_anchor_front(viewer, viewer_to_target)
    target_dot = _dot_for_anchor_front(target, target_to_viewer)
    if viewer_dot is None or target_dot is None or viewer_to_target is None:
        return 0.0
    pair_dot = min(viewer_dot, target_dot)
    viewer_center = _rect_center(viewer_rect)
    target_center = _rect_center(target_rect)
    distance = math.hypot(
        viewer_center[0] - target_center[0],
        viewer_center[1] - target_center[1],
    )
    axis_alignment = max(abs(viewer_to_target[0]), abs(viewer_to_target[1]))
    relation = _viewing_pair_relation(pair)
    if relation in VIEW_TARGET_RELATIONS:
        viewer_min_dot = _view_target_viewer_min_dot(pair, viewer)
        target_min_dot = OBJECT_LEVEL_VIEW_TARGET_TARGET_MIN_DOT
        score = 1800.0 - abs(distance - 3000.0) * 0.35
        score += viewer_dot * 6200.0
        score += target_dot * 5600.0
        if viewer_dot >= viewer_min_dot and target_dot >= target_min_dot:
            score += 5200.0
        else:
            score -= max(viewer_min_dot - viewer_dot, 0.0) * 10000.0
            score -= max(target_min_dot - target_dot, 0.0) * 10000.0
        if distance < OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MIN_DISTANCE_MM:
            score -= (
                OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MIN_DISTANCE_MM - distance
            ) * 1.2
        elif distance > OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MAX_DISTANCE_MM:
            score -= (
                distance - OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MAX_DISTANCE_MM
            ) * 0.8
        else:
            score += 900.0
        score += axis_alignment * 700.0
        return score
    score = 1800.0 - abs(distance - 2400.0)
    score += pair_dot * 7200.0
    if pair_dot >= OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT:
        score += 4200.0
    else:
        score -= (OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT - pair_dot) * 9000.0
    score += axis_alignment * 900.0
    if _anchor_kinds_are_opposite(
        str(viewer.get("anchor_kind") or ""),
        str(target.get("anchor_kind") or ""),
    ):
        score += 1800.0
    return score


def _non_participant_viewing_target_penalty(
    *,
    anchor_solution: Mapping[str, dict[str, Any]],
    viewer: Mapping[str, Any],
    target: Mapping[str, Any],
) -> float:
    target_quality = _mutual_viewing_quality(viewer, target)
    if target_quality is None:
        return 0.0
    penalty = 0.0
    participant_clusters = {
        str(viewer.get("cluster_id") or ""),
        str(target.get("cluster_id") or ""),
    }
    for cluster_id, candidate in anchor_solution.items():
        if cluster_id in participant_clusters:
            continue
        intruder_quality = _mutual_viewing_quality(viewer, candidate)
        if intruder_quality is None:
            continue
        acceptable_floor = max(0.62, target_quality + 0.08)
        if intruder_quality > acceptable_floor:
            penalty += (intruder_quality - acceptable_floor) * 5200.0
        if intruder_quality > target_quality and _anchor_kinds_are_opposite(
            str(viewer.get("anchor_kind") or ""),
            str(candidate.get("anchor_kind") or ""),
        ):
            penalty += (intruder_quality - target_quality) * 1800.0
    return penalty


def _mutual_viewing_quality(
    viewer: Mapping[str, Any],
    target: Mapping[str, Any],
) -> float | None:
    viewer_rect = viewer.get("rect")
    target_rect = target.get("rect")
    if not (
        isinstance(viewer_rect, tuple)
        and len(viewer_rect) == 4
        and isinstance(target_rect, tuple)
        and len(target_rect) == 4
    ):
        return None
    viewer_to_target = _vector_between_rect_centers(viewer_rect, target_rect)
    target_to_viewer = _vector_between_rect_centers(target_rect, viewer_rect)
    viewer_dot = _dot_for_anchor_front(viewer, viewer_to_target)
    target_dot = _dot_for_anchor_front(target, target_to_viewer)
    if viewer_dot is None or target_dot is None:
        return None
    return min(viewer_dot, target_dot)


def _dot_for_anchor_front(
    anchor: Mapping[str, Any],
    desired_front: tuple[float, float] | None,
) -> float | None:
    front = _front_tuple(anchor.get("front_world"))
    if front is None or desired_front is None:
        return None
    return _dot_normalized(front, desired_front)


def _scenario_anchor_pair_score(
    *,
    primary_anchor: Mapping[str, Any],
    secondary_anchor: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    contract = _primary_anchor_pair_contract(relation_plan)
    pair_type = str(contract.get("pair_type") or "").strip().lower()
    if not pair_type:
        return 0.0
    primary_kind = str(primary_anchor.get("anchor_kind") or "").strip().lower()
    secondary_kind = str(secondary_anchor.get("anchor_kind") or "").strip().lower()
    if pair_type in {"opposite_walls", "daylight_opposite"}:
        if _anchor_kinds_are_opposite(primary_kind, secondary_kind):
            return 2200.0
        if primary_kind == secondary_kind:
            return -2600.0
        return -650.0
    if pair_type == "floating_to_wall":
        score = 0.0
        score += 2400.0 if primary_kind == "center" else -2800.0
        score += 1200.0 if secondary_kind.endswith("_wall") else -900.0
        return score
    if pair_type in {"adjacent_walls", "offset_axis"}:
        if _anchor_kinds_are_adjacent(primary_kind, secondary_kind):
            return 1500.0
        if primary_kind == secondary_kind:
            return -1600.0
    return 0.0


def _primary_anchor_pair_contract(
    relation_plan: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(relation_plan, Mapping):
        return {}
    contracts = relation_plan.get("anchor_pair_contracts")
    if isinstance(contracts, Sequence) and not isinstance(contracts, str):
        for row in contracts:
            if isinstance(row, Mapping):
                return row
    return {}


def _anchor_kinds_are_opposite(left: str, right: str) -> bool:
    pairs = {
        frozenset(("top_wall", "bottom_wall")),
        frozenset(("left_wall", "right_wall")),
        frozenset(("window_side", "bottom_wall")),
    }
    return frozenset((left, right)) in pairs


def _anchor_kinds_are_adjacent(left: str, right: str) -> bool:
    if not left.endswith("_wall") or not right.endswith("_wall"):
        return False
    return not _anchor_kinds_are_opposite(left, right) and left != right


def _anchor_pair_orientation_score(
    *,
    left_cluster_id: str,
    right_cluster_id: str,
    left_anchor: Mapping[str, Any],
    right_anchor: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    if not isinstance(relation_plan, Mapping):
        return 0.0
    score = 0.0
    pairs = (
        (left_cluster_id, left_anchor, right_cluster_id, right_anchor),
        (right_cluster_id, right_anchor, left_cluster_id, left_anchor),
    )
    for cluster_id, anchor, target_cluster_id, target_anchor in pairs:
        object_id = str(anchor.get("object_id") or "")
        intents = _object_level_orientation_intents(
            cluster_id=cluster_id,
            object_id=object_id,
            relation_plan=relation_plan,
            include_cluster=True,
        )
        if "face_cluster" not in intents:
            continue
        configured_target = _object_level_target_cluster_id(
            cluster_id=cluster_id,
            object_id=object_id,
            relation_plan=relation_plan,
        )
        if configured_target and configured_target != target_cluster_id:
            continue
        front = _front_tuple(anchor.get("front_world"))
        desired = _vector_between_rect_centers(
            tuple(anchor["rect"]),
            tuple(target_anchor["rect"]),
        )
        if front is None or desired is None:
            continue
        score += _dot_normalized(front, desired) * 1600.0
    return score


def _place_support_objects_for_solution(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    grid_mm: int,
    max_solutions: int,
) -> list[dict[str, Any]]:
    placed_objects: list[dict[str, Any]] = []
    dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]] = {
        str(cluster_id): [dict(row) for row in rows if isinstance(row, Mapping)]
        for cluster_id, rows in (
            solution.get("dropped_inventory_by_cluster") or {}
        ).items()
        if isinstance(rows, Sequence) and not isinstance(rows, str)
    }
    anchor_solution = (
        solution.get("anchor_solution")
        if isinstance(solution.get("anchor_solution"), Mapping)
        else {}
    )
    occupied = [row["rect"] for row in anchor_solution.values()]
    # materialize anchors first
    for cluster_id, anchor_row in anchor_solution.items():
        cluster_program = world["clusters_by_id"][cluster_id]
        anchor_id = anchor_row["object_id"]
        placed_objects.append(
            _materialize_object_row(
                cluster_program,
                anchor_id,
                anchor_row["x"],
                anchor_row["y"],
                anchor_row["rot"],
            )
        )
    placed_by_id = {
        (row["cluster_id"], row["object_id"]): row for row in placed_objects
    }
    protected_regions = (
        world.get("protected_regions")
        if isinstance(world.get("protected_regions"), Sequence)
        else []
    )
    seated_surface_supports = _surface_seat_supports_by_cluster(world)
    results: list[dict[str, Any]] = []
    tasks: list[tuple[str, str, Mapping[str, Any], str]] = []

    for cluster_id in world["anchor_cluster_order"]:
        cluster_program = world["clusters_by_id"][cluster_id]
        if cluster_id not in anchor_solution and _cluster_is_solver_trial_optional(
            cluster_program
        ):
            continue
        object_program = cluster_program["object_program"]
        anchor_id = _cluster_program_dominant_anchor(cluster_program)
        placement_order = [
            object_id
            for object_id in object_program.get("placement_order") or []
            if isinstance(object_id, str) and object_id != anchor_id
        ]
        support_edges = {
            str(row.get("object_id")): row
            for row in object_program.get("support_edges") or []
            if isinstance(row, Mapping) and isinstance(row.get("object_id"), str)
        }
        for object_id in placement_order:
            spec = _object_spec(cluster_program, object_id)
            if spec is None:
                continue
            edge = support_edges.get(object_id, {})
            base_id = str(edge.get("relative_to") or anchor_id or "")
            tasks.append((cluster_id, object_id, edge, base_id))

    def search(index: int, support_score: float) -> None:
        if len(results) >= max(1, int(max_solutions)):
            return
        if index >= len(tasks):
            results.append(
                {
                    "placed_objects": deepcopy(placed_objects),
                    "dropped_inventory_by_cluster": deepcopy(
                        {
                            cluster_id: rows
                            for cluster_id, rows in dropped_inventory_by_cluster.items()
                            if rows
                        }
                    ),
                    "support_score": round(float(support_score), 3),
                    "protected_region_issue_count": _object_level_protected_region_issue_count(
                        placed_objects=placed_objects,
                        protected_regions=protected_regions,
                    ),
                }
            )
            return
        cluster_id, object_id, edge, base_id = tasks[index]
        cluster_program = world["clusters_by_id"][cluster_id]
        object_program = cluster_program["object_program"]
        anchor_id = _cluster_program_dominant_anchor(cluster_program)
        required_support_ids = _required_support_object_ids(object_program)
        droppable = _droppable_object_ids(object_program)
        protected = set(object_program.get("protected_ids") or [])

        base_row = placed_by_id.get((cluster_id, base_id))
        if base_row is None and anchor_id is not None:
            base_row = placed_by_id.get((cluster_id, anchor_id))
        if base_row is None:
            if object_id in droppable and object_id not in protected:
                dropped_inventory_by_cluster.setdefault(cluster_id, []).append(
                    {"object_id": object_id, "reason": "base_missing"}
                )
                search(index + 1, support_score - 120.0)
                dropped_inventory_by_cluster[cluster_id].pop()
                return
            return

        slots = _support_slot_candidates(
            cluster_program=cluster_program,
            object_id=object_id,
            base_row=base_row,
            edge=edge,
            grid_mm=grid_mm,
        )
        if object_id in required_support_ids:
            slots.extend(
                _required_support_standalone_slot_candidates(
                    cluster_program=cluster_program,
                    object_id=object_id,
                    relation_plan=relation_plan,
                    world=world,
                    room_model=room_model,
                    grid_mm=grid_mm,
                )
            )
        clean_slots: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        repair_slots: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for slot in slots:
            rect = slot["rect"]
            if any(_rects_overlap(rect, occ) for occ in occupied):
                continue
            materialized = _materialize_object_row(
                cluster_program,
                object_id,
                slot["x"],
                slot["y"],
                slot["rot"],
                relative_to=slot.get("relative_to", base_id),
            )
            if not _rect_inside_room_footprint(rect, world, row=materialized):
                continue
            slot_is_clean = _object_rect_is_usable(
                rect,
                occupied,
                world,
                row=materialized,
                front_access_rows=tuple(placed_by_id.values()),
            )
            if (
                not slot_is_clean
                and object_id in droppable
                and object_id not in protected
            ):
                continue
            relax_penalty = _support_slot_relaxation_penalty(
                candidate=materialized,
                placed_objects=placed_objects,
                protected_regions=protected_regions,
                world=world,
                relation_plan=relation_plan,
            )
            if slot_is_clean:
                clean_slots.append((relax_penalty, slot, materialized))
            else:
                repair_slots.append((relax_penalty, slot, materialized))
        viable_slots = clean_slots or repair_slots
        viable_slots.sort(
            key=lambda item: (
                item[0],
                -float(item[1].get("orientation_score") or 0.0),
                item[1]["y"],
                item[1]["x"],
                item[1]["rot"],
            )
        )

        for relax_penalty, slot, materialized in viable_slots:
            rect = slot["rect"]
            placed_objects.append(materialized)
            placed_by_id[(cluster_id, object_id)] = materialized
            occupied.append(rect)
            search(
                index + 1,
                support_score
                + float(slot.get("orientation_score") or 0.0)
                - relax_penalty,
            )
            occupied.pop()
            placed_by_id.pop((cluster_id, object_id), None)
            placed_objects.pop()
            if len(results) >= max(1, int(max_solutions)):
                break

        if object_id in droppable and object_id not in protected:
            if _dropping_last_surface_seat_support(
                cluster_id=cluster_id,
                object_id=object_id,
                base_id=base_id,
                placed_by_id=placed_by_id,
                remaining_tasks=tasks[index + 1 :],
                seated_surface_supports=seated_surface_supports,
            ):
                return
            dropped_inventory_by_cluster.setdefault(cluster_id, []).append(
                {"object_id": object_id, "reason": "support_slot_failed"}
            )
            search(index + 1, support_score - 90.0)
            dropped_inventory_by_cluster[cluster_id].pop()
            return
        return

    search(0, 0.0)
    return results


def _support_slot_relaxation_penalty(
    *,
    candidate: Mapping[str, Any],
    placed_objects: Sequence[Mapping[str, Any]],
    protected_regions: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    protected_issues = _object_level_protected_region_issues(
        placed_objects=[candidate],
        protected_regions=protected_regions,
    )
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    forbidden_issues = _object_level_cluster_forbidden_region_issues(
        placed_objects=[candidate],
        forbidden_regions=forbidden_regions,
    )
    functional_issues = _object_level_functional_geometry_issues(
        placed_objects=[*placed_objects, candidate],
        world=world,
        relation_plan=relation_plan,
    )
    blocking_functional = [
        issue
        for issue in functional_issues
        if (
            (
                str(issue.get("cluster_id") or "")
                == str(candidate.get("cluster_id") or "")
                and str(issue.get("object_id") or "")
                == str(candidate.get("object_id") or "")
            )
            or (
                str(issue.get("blocked_cluster_id") or "")
                == str(candidate.get("cluster_id") or "")
                and str(issue.get("blocked_object_id") or "")
                == str(candidate.get("object_id") or "")
            )
        )
        and str(issue.get("violation_severity") or "").strip().lower() == "blocking"
    ]
    return (
        _object_level_protected_region_penalty(protected_issues)
        + _object_level_protected_region_penalty(forbidden_issues)
        + 1400.0 * len(blocking_functional)
        + 180.0 * max(0, len(functional_issues) - len(blocking_functional))
    )


def _repair_object_level_solution_geometry(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    grid_mm: int,
) -> dict[str, Any] | None:
    rows = [
        deepcopy(row)
        for row in (solution.get("placed_objects") or [])
        if isinstance(row, Mapping)
    ]
    if not rows:
        return {
            "placed_objects": [],
            "summary": {
                "status": "clean",
                "moved_objects": [],
                "total_shift_mm": 0.0,
                "penalty": 0,
            },
        }

    original_rows_by_key = {
        _object_repair_key(row): row
        for row in rows
        if _object_repair_key(row) is not None
    }
    repaired_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    repaired_by_index: dict[int, dict[str, Any]] = {}
    occupied_rects: list[tuple[int, int, int, int]] = []
    moved_objects: list[dict[str, Any]] = []

    for index, row in sorted(
        enumerate(rows), key=lambda item: _object_repair_order_key(item[1])
    ):
        rect = _rect_tuple(row.get("rect"))
        if rect is None:
            return None
        candidate_rect = _best_object_geometry_repair_rect(
            row=row,
            original_rect=rect,
            original_rows_by_key=original_rows_by_key,
            repaired_by_key=repaired_by_key,
            occupied_rects=occupied_rects,
            front_access_rows=tuple(repaired_by_key.values()),
            world=world,
            grid_mm=grid_mm,
        )
        if candidate_rect is None:
            return None
        repaired_row = _object_row_with_rect(row, candidate_rect)
        repaired_by_index[index] = repaired_row
        key = _object_repair_key(repaired_row)
        if key is not None:
            repaired_by_key[key] = repaired_row
        occupied_rects.append(candidate_rect)
        if candidate_rect != rect:
            old_center = _rect_center(rect)
            new_center = _rect_center(candidate_rect)
            shift_mm = math.hypot(
                new_center[0] - old_center[0], new_center[1] - old_center[1]
            )
            moved_objects.append(
                {
                    "cluster_id": str(row.get("cluster_id") or ""),
                    "object_id": str(row.get("object_id") or ""),
                    "from_rect": list(rect),
                    "to_rect": list(candidate_rect),
                    "shift_mm": round(shift_mm, 3),
                    "reason": "polygon_oob_or_overlap_guardrail",
                }
            )

    repaired_rows = [repaired_by_index[index] for index in range(len(rows))]
    total_shift = sum(float(item["shift_mm"]) for item in moved_objects)
    penalty = int(round(total_shift * 1.8 + len(moved_objects) * 180.0))
    return {
        "placed_objects": repaired_rows,
        "summary": {
            "status": "repaired" if moved_objects else "clean",
            "moved_objects": moved_objects,
            "total_shift_mm": round(total_shift, 3),
            "penalty": penalty,
        },
    }


def _best_object_geometry_repair_rect(
    *,
    row: Mapping[str, Any],
    original_rect: tuple[int, int, int, int],
    original_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    repaired_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    occupied_rects: Sequence[tuple[int, int, int, int]],
    front_access_rows: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    grid_mm: int,
) -> tuple[int, int, int, int] | None:
    if _object_rect_is_usable(
        original_rect,
        occupied_rects,
        world,
        row=row,
        front_access_rows=front_access_rows,
    ):
        return original_rect

    room_bbox = _object_solver_fit_room_bbox(world, row=row)
    width = original_rect[2] - original_rect[0]
    height = original_rect[3] - original_rect[1]
    if width <= 0 or height <= 0:
        return None

    grid = max(25, int(grid_mm))
    original_center = _rect_center(original_rect)
    target_center = _object_repair_target_center(
        row=row,
        original_center=original_center,
        original_rows_by_key=original_rows_by_key,
        repaired_by_key=repaired_by_key,
    )
    x_min = int(room_bbox[0])
    y_min = int(room_bbox[1])
    x_max = int(room_bbox[2]) - width
    y_max = int(room_bbox[3]) - height
    if x_max < x_min or y_max < y_min:
        return None

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int]] = set()

    def add_origin(x: int, y: int) -> None:
        sx = _snap_to_grid(x, grid)
        sy = _snap_to_grid(y, grid)
        sx = max(x_min, min(x_max, sx))
        sy = max(y_min, min(y_max, sy))
        key = (sx, sy)
        if key in seen:
            return
        seen.add(key)
        rect = (sx, sy, sx + width, sy + height)
        if not _object_rect_is_usable(
            rect,
            occupied_rects,
            world,
            row=row,
            front_access_rows=front_access_rows,
        ):
            return
        candidates.append(
            (
                _object_geometry_repair_score(
                    row=row,
                    rect=rect,
                    original_rect=original_rect,
                    target_center=target_center,
                    original_rows_by_key=original_rows_by_key,
                    repaired_by_key=repaired_by_key,
                ),
                rect,
            )
        )

    original_origin = (original_rect[0], original_rect[1])
    target_origin = (
        int(round(target_center[0] - width / 2.0)),
        int(round(target_center[1] - height / 2.0)),
    )
    add_origin(original_origin[0], original_origin[1])
    add_origin(target_origin[0], target_origin[1])
    local_radius = max(grid, OBJECT_LEVEL_GEOMETRY_REPAIR_LOCAL_RADIUS_MM)
    for center_x, center_y in (original_origin, target_origin):
        for radius in range(grid, local_radius + 1, grid):
            for dx in range(-radius, radius + 1, grid):
                add_origin(center_x + dx, center_y - radius)
                add_origin(center_x + dx, center_y + radius)
            for dy in range(-radius + grid, radius, grid):
                add_origin(center_x - radius, center_y + dy)
                add_origin(center_x + radius, center_y + dy)
            if len(candidates) >= OBJECT_LEVEL_GEOMETRY_REPAIR_MAX_CANDIDATES:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    for x in range(x_min, x_max + 1, grid):
        for y in range(y_min, y_max + 1, grid):
            add_origin(x, y)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _object_rect_is_usable(
    rect: tuple[int, int, int, int],
    occupied_rects: Sequence[tuple[int, int, int, int]],
    world: Mapping[str, Any],
    *,
    row: Mapping[str, Any] | None = None,
    front_access_rows: Sequence[Mapping[str, Any]] = (),
) -> bool:
    if not _rect_inside_room_footprint(rect, world, row=row):
        return False
    if any(_rects_overlap(rect, occupied) for occupied in occupied_rects):
        return False
    if row is None:
        return True
    if _rect_blocks_front_access_rows(
        rect=rect,
        row=row,
        front_access_rows=front_access_rows,
        room_bbox=world["room_bbox"],
    ):
        return False
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    protected_regions = (
        world.get("protected_regions")
        if isinstance(world.get("protected_regions"), Sequence)
        else []
    )
    if not forbidden_regions and not protected_regions:
        return True
    blocking_cache = (
        world.get("_object_blocking_region_cache")
        if isinstance(world.get("_object_blocking_region_cache"), dict)
        else None
    )
    cache_key = (
        str(row.get("cluster_id") or ""),
        str(row.get("object_id") or ""),
        str(row.get("anchor_kind") or ""),
        rect,
    )
    if blocking_cache is not None and cache_key in blocking_cache:
        return not bool(blocking_cache[cache_key])
    candidate = dict(row)
    candidate["rect"] = rect
    protected_blocked = any(
        _protected_region_issue_is_blocking(issue)
        for issue in _object_level_protected_region_issues(
            placed_objects=[candidate],
            protected_regions=protected_regions,
        )
    )
    forbidden_blocked = any(
        _forbidden_region_issue_blocks_geometry(issue)
        for issue in _object_level_cluster_forbidden_region_issues(
            placed_objects=[candidate],
            forbidden_regions=forbidden_regions,
        )
    )
    blocked = protected_blocked or forbidden_blocked
    if blocking_cache is not None:
        blocking_cache[cache_key] = blocked
    return not blocked


def _object_rect_unusable_reject_reason(
    *,
    rect: tuple[int, int, int, int],
    world: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    candidate = dict(row)
    candidate["rect"] = rect
    protected_regions = (
        world.get("protected_regions")
        if isinstance(world.get("protected_regions"), Sequence)
        else []
    )
    protected_issues = _object_level_protected_region_issues(
        placed_objects=[candidate],
        protected_regions=protected_regions,
    )
    if any(_protected_region_issue_is_blocking(issue) for issue in protected_issues):
        return "object_rect_not_usable:protected_region"
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    forbidden_issues = _object_level_cluster_forbidden_region_issues(
        placed_objects=[candidate],
        forbidden_regions=forbidden_regions,
    )
    blocking_forbidden = [
        issue
        for issue in forbidden_issues
        if _forbidden_region_issue_blocks_geometry(issue)
    ]
    if blocking_forbidden:
        region_id = str(blocking_forbidden[0].get("region_id") or "")
        zone_type = str(blocking_forbidden[0].get("zone_type") or "")
        return f"object_rect_not_usable:{_forbidden_region_reason_label({'region_id': region_id, 'zone_type': zone_type})}"
    footprint_reason = _rect_room_footprint_reject_reason(
        rect=rect,
        world=world,
        row=row,
    )
    if footprint_reason == "wall_inner_face":
        return "object_rect_not_usable:wall_inner_face"
    if footprint_reason != "outside_room":
        return f"object_rect_not_usable:{footprint_reason}"
    return "object_rect_not_usable"


def _rect_blocks_front_access_rows(
    *,
    rect: tuple[int, int, int, int],
    row: Mapping[str, Any],
    front_access_rows: Sequence[Mapping[str, Any]],
    room_bbox: tuple[int, int, int, int],
) -> bool:
    candidate = dict(row)
    candidate["rect"] = rect
    if _object_allowed_in_front_access_zone(candidate):
        return False
    candidate_key = _object_repair_key(candidate)
    for front_access_row in front_access_rows:
        front_access_key = _object_repair_key(front_access_row)
        if candidate_key is not None and front_access_key == candidate_key:
            continue
        access_zone = _front_access_zone_for_row(front_access_row, room_bbox)
        if access_zone is None:
            continue
        if _rect_overlap_ratio(rect, access_zone) > 0.02:
            return True
    return False


def _forbidden_region_issue_blocks_geometry(issue: Mapping[str, Any]) -> bool:
    return _protected_region_issue_is_blocking(issue)


def _object_repair_target_center(
    *,
    row: Mapping[str, Any],
    original_center: tuple[float, float],
    original_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    repaired_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, float]:
    cluster_id = str(row.get("cluster_id") or "")
    relative_to = str(row.get("relative_to") or "")
    if not cluster_id or not relative_to:
        return original_center
    base_key = (cluster_id, relative_to)
    original_base = original_rows_by_key.get(base_key)
    repaired_base = repaired_by_key.get(base_key)
    if original_base is None or repaired_base is None:
        return original_center
    original_base_rect = _rect_tuple(original_base.get("rect"))
    repaired_base_rect = _rect_tuple(repaired_base.get("rect"))
    if original_base_rect is None or repaired_base_rect is None:
        return original_center
    original_base_center = _rect_center(original_base_rect)
    repaired_base_center = _rect_center(repaired_base_rect)
    return (
        repaired_base_center[0] + original_center[0] - original_base_center[0],
        repaired_base_center[1] + original_center[1] - original_base_center[1],
    )


def _object_geometry_repair_score(
    *,
    row: Mapping[str, Any],
    rect: tuple[int, int, int, int],
    original_rect: tuple[int, int, int, int],
    target_center: tuple[float, float],
    original_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    repaired_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    center = _rect_center(rect)
    original_center = _rect_center(original_rect)
    score = math.hypot(center[0] - original_center[0], center[1] - original_center[1])
    score += 0.55 * math.hypot(
        center[0] - target_center[0], center[1] - target_center[1]
    )
    score += _object_relative_side_penalty(
        row=row,
        candidate_center=center,
        original_center=original_center,
        original_rows_by_key=original_rows_by_key,
        repaired_by_key=repaired_by_key,
    )
    return score


def _object_relative_side_penalty(
    *,
    row: Mapping[str, Any],
    candidate_center: tuple[float, float],
    original_center: tuple[float, float],
    original_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    repaired_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    cluster_id = str(row.get("cluster_id") or "")
    relative_to = str(row.get("relative_to") or "")
    if not cluster_id or not relative_to:
        return 0.0
    base_key = (cluster_id, relative_to)
    original_base_rect = _rect_tuple(
        (original_rows_by_key.get(base_key) or {}).get("rect")
    )
    repaired_base_rect = _rect_tuple((repaired_by_key.get(base_key) or {}).get("rect"))
    if original_base_rect is None or repaired_base_rect is None:
        return 0.0
    original_base_center = _rect_center(original_base_rect)
    repaired_base_center = _rect_center(repaired_base_rect)
    original_vec = (
        original_center[0] - original_base_center[0],
        original_center[1] - original_base_center[1],
    )
    candidate_vec = (
        candidate_center[0] - repaired_base_center[0],
        candidate_center[1] - repaired_base_center[1],
    )
    penalty = 0.0
    if abs(original_vec[0]) >= abs(original_vec[1]) and abs(original_vec[0]) > 1e-6:
        if original_vec[0] * candidate_vec[0] <= 0:
            penalty += 2400.0
    elif abs(original_vec[1]) > 1e-6 and original_vec[1] * candidate_vec[1] <= 0:
        penalty += 2400.0
    dot = _dot_normalized(original_vec, candidate_vec)
    if dot < 0.25:
        penalty += (0.25 - dot) * 1800.0
    return penalty


def _object_repair_order_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    priority = str(row.get("priority") or "").strip().lower()
    role = str(row.get("role") or "").strip().lower()
    has_relative = bool(str(row.get("relative_to") or "").strip())
    if priority in {"anchor", "dominant_anchor"} or role == "dominant_anchor":
        rank = 0
    elif has_relative:
        rank = 2
    else:
        rank = 1
    return rank, str(row.get("cluster_id") or ""), str(row.get("object_id") or "")


def _object_repair_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    cluster_id = str(row.get("cluster_id") or "").strip()
    object_id = str(row.get("object_id") or "").strip()
    if not cluster_id or not object_id:
        return None
    return cluster_id, object_id


def _object_row_with_rect(
    row: Mapping[str, Any], rect: tuple[int, int, int, int]
) -> dict[str, Any]:
    next_row = dict(row)
    next_row["x"] = rect[0]
    next_row["y"] = rect[1]
    next_row["w"] = rect[2] - rect[0]
    next_row["h"] = rect[3] - rect[1]
    next_row["rect"] = rect
    next_row["bbox"] = {
        "min_x": rect[0],
        "min_y": rect[1],
        "max_x": rect[2],
        "max_y": rect[3],
    }
    next_row["center"] = {
        "x": int(round((rect[0] + rect[2]) / 2.0)),
        "y": int(round((rect[1] + rect[3]) / 2.0)),
    }
    return next_row


def _apply_object_level_geometry_repair_penalty(
    verify: dict[str, Any], repair_summary: Mapping[str, Any] | None
) -> None:
    if not isinstance(repair_summary, Mapping):
        return
    moved_objects = repair_summary.get("moved_objects")
    if not isinstance(moved_objects, Sequence) or isinstance(moved_objects, str):
        return
    moved_count = len(moved_objects)
    if moved_count <= 0:
        verify["geometry_repair"] = deepcopy(dict(repair_summary))
        verify["geometry_repair_penalty"] = 0
        return
    penalty = int(repair_summary.get("penalty") or 0)
    verify["geometry_repair"] = deepcopy(dict(repair_summary))
    verify["geometry_repair_penalty"] = penalty
    verify["soft_issue_count"] = int(verify.get("soft_issue_count") or 0) + moved_count
    verify["layout_score"] = int(verify.get("layout_score") or 0) - penalty


def _droppable_object_ids(object_program: Mapping[str, Any]) -> set[str]:
    members = {str(item) for item in (object_program.get("members") or []) if str(item)}
    protected = {str(item) for item in (object_program.get("protected_ids") or [])}
    required = {str(item) for item in (object_program.get("required_object_ids") or [])}
    required_support = _required_support_object_ids(object_program)
    non_droppable = protected | required | required_support
    optional = {str(item) for item in (object_program.get("optional_object_ids") or [])}
    support_edge_ids = {
        str(row.get("object_id"))
        for row in (object_program.get("support_edges") or [])
        if isinstance(row, Mapping) and str(row.get("object_id") or "")
    }
    object_specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    droppable = {
        str(item)
        for item in (object_program.get("droppable_ids") or [])
        if str(item) in members and str(item) not in non_droppable
    }
    droppable.update(
        item for item in optional if item in members and item not in non_droppable
    )
    for action in object_program.get("degradation_ladder") or []:
        token = str(action or "").strip()
        if token == "drop_secondary_support":
            for object_id in members:
                spec = (
                    object_specs.get(object_id)
                    if isinstance(object_specs, Mapping)
                    else None
                )
                role = (
                    str(spec.get("role") or "").strip().lower()
                    if isinstance(spec, Mapping)
                    else ""
                )
                priority = (
                    str(spec.get("priority") or "").strip().lower()
                    if isinstance(spec, Mapping)
                    else ""
                )
                if (
                    object_id not in non_droppable
                    and object_id in support_edge_ids
                    and (
                        priority in {"secondary", "optional", "support"}
                        or role
                        in {
                            "support",
                            "secondary",
                            "secondary_support",
                            "decor_light",
                        }
                    )
                ):
                    droppable.add(object_id)
            continue
        if token.startswith("drop_"):
            object_id = token.removeprefix("drop_")
            if object_id and object_id in members and object_id not in non_droppable:
                droppable.add(object_id)
    return droppable


def _required_support_object_ids(object_program: Mapping[str, Any]) -> set[str]:
    members = {str(item) for item in (object_program.get("members") or []) if str(item)}
    out = {
        str(item)
        for item in (
            object_program.get("required_support_ids")
            or object_program.get("relaxable_required_ids")
            or []
        )
        if str(item) in members
    }
    specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    for object_id, spec in specs.items():
        if not isinstance(spec, Mapping):
            continue
        if str(object_id) in members and (
            bool(spec.get("required_support")) or bool(spec.get("relaxable_required"))
        ):
            out.add(str(object_id))
    return out


def _missing_required_object_rows(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
) -> list[dict[str, str]]:
    placed = {
        (str(row.get("cluster_id") or ""), str(row.get("object_id") or ""))
        for row in placed_objects
        if isinstance(row, Mapping)
    }
    clusters_by_id = (
        world.get("clusters_by_id")
        if isinstance(world.get("clusters_by_id"), Mapping)
        else {}
    )
    missing: list[dict[str, str]] = []
    for cluster_id, cluster_program in clusters_by_id.items():
        if not isinstance(cluster_program, Mapping):
            continue
        object_program = (
            cluster_program.get("object_program")
            if isinstance(cluster_program.get("object_program"), Mapping)
            else {}
        )
        required_ids = {
            str(item)
            for item in (object_program.get("required_object_ids") or [])
            if str(item)
        }
        required_ids.update(_required_support_object_ids(object_program))
        for object_id in sorted(required_ids):
            key = (str(cluster_id), object_id)
            if key not in placed:
                missing.append(
                    {
                        "cluster_id": str(cluster_id),
                        "object_id": object_id,
                        "reason": "required_object_missing",
                    }
                )
    return missing


def _surface_seat_supports_by_cluster(
    world: Mapping[str, Any],
) -> dict[str, dict[str, set[str]]]:
    clusters_by_id = (
        world.get("clusters_by_id")
        if isinstance(world.get("clusters_by_id"), Mapping)
        else {}
    )
    out: dict[str, dict[str, set[str]]] = {}
    for cluster_id, cluster_program in clusters_by_id.items():
        if not isinstance(cluster_program, Mapping):
            continue
        object_program = (
            cluster_program.get("object_program")
            if isinstance(cluster_program.get("object_program"), Mapping)
            else {}
        )
        by_base = _surface_seat_supports_by_base(object_program)
        if by_base:
            out[str(cluster_id)] = by_base
    return out


def _surface_seat_supports_by_base(
    object_program: Mapping[str, Any],
) -> dict[str, set[str]]:
    support_edges = (
        object_program.get("support_edges")
        if isinstance(object_program.get("support_edges"), Sequence)
        and not isinstance(object_program.get("support_edges"), str)
        else []
    )
    out: dict[str, set[str]] = {}
    for edge in support_edges:
        if not isinstance(edge, Mapping):
            continue
        object_id = str(edge.get("object_id") or "").strip()
        relative_to = str(edge.get("relative_to") or "").strip()
        if not object_id or not relative_to:
            continue
        support_role = str(edge.get("support_role") or "").strip().lower()
        band_intent = str(edge.get("band_intent") or "").strip().lower()
        if not _object_program_id_is_work_surface(object_program, relative_to):
            continue
        if not _object_program_id_is_seat(object_program, object_id):
            continue
        if support_role == "secondary_seat" or band_intent == "flank_band":
            out.setdefault(relative_to, set()).add(object_id)
    return out


def _object_program_id_is_work_surface(
    object_program: Mapping[str, Any], object_id: str
) -> bool:
    spec = _object_program_spec(object_program, object_id)
    return is_work_surface_like(_object_program_kind(object_id, spec))


def _object_program_id_is_seat(
    object_program: Mapping[str, Any], object_id: str
) -> bool:
    spec = _object_program_spec(object_program, object_id)
    return is_seat_like(_object_program_kind(object_id, spec))


def _object_program_spec(
    object_program: Mapping[str, Any], object_id: str
) -> Mapping[str, Any] | None:
    specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    spec = specs.get(object_id)
    return spec if isinstance(spec, Mapping) else None


def _object_program_kind(
    object_id: str,
    spec: Mapping[str, Any] | None,
) -> str:
    if isinstance(spec, Mapping):
        for key in ("category", "base_object_id", "object_id"):
            value = spec.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return object_id


def _dropping_last_surface_seat_support(
    *,
    cluster_id: str,
    object_id: str,
    base_id: str,
    placed_by_id: Mapping[tuple[str, str], Mapping[str, Any]],
    remaining_tasks: Sequence[tuple[str, str, Mapping[str, Any], str]],
    seated_surface_supports: Mapping[str, Mapping[str, set[str]]],
) -> bool:
    required_seats = seated_surface_supports.get(cluster_id, {}).get(base_id, set())
    if object_id not in required_seats:
        return False
    if any(
        (cluster_id, seat_id) in placed_by_id
        for seat_id in required_seats
        if seat_id != object_id
    ):
        return False
    return not any(
        next_cluster_id == cluster_id
        and next_base_id == base_id
        and next_object_id in required_seats
        and next_object_id != object_id
        for next_cluster_id, next_object_id, _edge, next_base_id in remaining_tasks
    )


def _normalized_support_side_options(edge: Mapping[str, Any]) -> list[str]:
    raw_options = [
        str(item).strip()
        for item in (edge.get("side_options") or ["front"])
        if str(item).strip()
    ]
    support_role = str(edge.get("support_role") or "").strip().lower()
    band_intent = str(edge.get("band_intent") or "").strip().lower()
    object_id = str(edge.get("object_id") or "").strip().lower()
    tokens = {support_role, band_intent, object_id}
    front_band_tokens = {
        "front_band",
        "frontal_support",
        "front_support",
        "coffee_table",
    }
    side_band_tokens = {
        "beside_base",
        "side_support",
        "side_table",
    }
    wall_band_tokens = {
        "wall_band",
        "wall_support",
    }
    flank_band_tokens = {"flank_band", "secondary_seat", "armchair", "chair"}
    if tokens & front_band_tokens:
        filtered = [
            option
            for option in raw_options
            if option
            in {
                "front",
                "head",
                "front_center",
                "head_center",
                "front_left",
                "front_right",
                "head_left",
                "head_right",
            }
        ]
        return filtered or ["front"]
    if tokens & side_band_tokens:
        rich_side_options = {
            "head_left",
            "head_right",
            "front_left",
            "front_right",
        }
        filtered = [
            option
            for option in raw_options
            if option in {"left", "right", *rich_side_options}
        ]
        if any(option in rich_side_options for option in filtered):
            return filtered
        return filtered or ["left", "right"]
    if tokens & wall_band_tokens:
        filtered = [
            option
            for option in raw_options
            if option
            in {
                "left",
                "right",
                "head",
                "front",
                "head_center",
                "front_center",
                "head_left",
                "head_right",
                "front_left",
                "front_right",
            }
        ]
        return filtered or ["left", "right"]
    if tokens & flank_band_tokens:
        filtered = [option for option in raw_options if option in {"left", "right"}]
        return filtered or ["left", "right"]
    return raw_options or ["front"]


def _required_support_standalone_slot_candidates(
    *,
    cluster_program: Mapping[str, Any],
    object_id: str,
    relation_plan: Mapping[str, Any] | None,
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    spec = _object_spec(cluster_program, object_id)
    if spec is None:
        return []
    dims = (
        spec.get("rep_dims_mm") if isinstance(spec.get("rep_dims_mm"), Mapping) else {}
    )
    length_mm = max(160, int(dims.get("L") or 0))
    width_mm = max(160, int(dims.get("W") or 0))
    allowed_rotations = (
        spec.get("allowed_rotations")
        if isinstance(spec.get("allowed_rotations"), Sequence)
        else [0, 90, 180, 270]
    )
    cluster_id = str(cluster_program.get("cluster_id") or "")
    relation_plan = relation_plan or {}
    anchor_region_preferences = (
        relation_plan.get("anchor_region_preferences_by_cluster")
        if isinstance(
            relation_plan.get("anchor_region_preferences_by_cluster"), Mapping
        )
        else {}
    )
    hint = (
        anchor_region_preferences.get(cluster_id)
        if isinstance(anchor_region_preferences, Mapping)
        and isinstance(anchor_region_preferences.get(cluster_id), Mapping)
        else {}
    )
    zone_assignment = str(hint.get("zone_assignment") or "edge_loading_zone")
    preferred_region_ids = _dedupe_string_sequence(
        [
            zone_assignment,
            *_string_sequence(hint.get("preferred_region_ids")),
            "edge_loading_zone",
            "bottom_wall_zone",
            "top_wall_zone",
            "left_wall_zone",
            "right_wall_zone",
        ]
    )
    region_bboxes = [
        bbox
        for region_id in preferred_region_ids
        for bbox in _region_bboxes_from_ref(
            str(region_id),
            world["region_index"],
            relation_plan,
            room_model,
        )
    ]
    region_bboxes = _dedupe_region_bboxes(region_bboxes)
    if not region_bboxes:
        region_bboxes = _anchor_fallback_region_bboxes(
            world=world,
            existing_region_bboxes=(),
        )
    placement_behavior = {
        "wall_backing": "preferred",
        "front_space": "preferred",
        "pose_flexibility": "high",
    }
    front_token = spec.get("front") if isinstance(spec.get("front"), str) else "top"
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    placement_room_bbox = _object_solver_fit_room_bbox(world)
    for region_bbox in region_bboxes[:16]:
        for rot in allowed_rotations:
            final_rot = int(rot) % 360
            w_mm, h_mm = _rotate_dims_for_rot(length_mm, width_mm, final_rot)
            origins = _wall_flush_origin_candidates(
                region_bbox=region_bbox,
                room_bbox=placement_room_bbox,
                zone_assignment=zone_assignment,
                placement_behavior=placement_behavior,
                w_mm=w_mm,
                h_mm=h_mm,
                grid_mm=grid_mm,
            )
            for x, y in origins:
                rect = (x, y, x + w_mm, y + h_mm)
                probe_row = {
                    "cluster_id": cluster_id,
                    "object_id": object_id,
                    "blocks_floor_area": spec.get("blocks_floor_area", True),
                    "collision_layer": spec.get("collision_layer"),
                    "place_on": spec.get("place_on"),
                    "placement_participation": spec.get("placement_participation"),
                    "solver_participation": spec.get("solver_participation"),
                }
                if not _rect_inside_room_footprint(rect, world, row=probe_row):
                    continue
                key = (x, y, final_rot)
                if key in seen:
                    continue
                seen.add(key)
                front_world = _front_vector_from_rotation(front_token, final_rot)
                desired_fronts = _wall_contact_inward_front_options(
                    rect=rect,
                    room_bbox=placement_room_bbox,
                ) or _near_wall_inward_front_options(
                    rect=rect,
                    room_bbox=placement_room_bbox,
                )
                desired_front = _best_aligned_required_front(
                    front_world=front_world,
                    desired_fronts=desired_fronts,
                )
                out.append(
                    {
                        "x": x,
                        "y": y,
                        "rot": final_rot,
                        "w": w_mm,
                        "h": h_mm,
                        "rect": rect,
                        "orientation_score": _front_alignment_score(
                            front_world=front_world,
                            desired_front=desired_front,
                        ),
                        "blocks_floor_area": spec.get("blocks_floor_area", True),
                        "collision_layer": spec.get("collision_layer"),
                        "place_on": deepcopy(spec.get("place_on"))
                        if isinstance(spec.get("place_on"), Mapping)
                        else None,
                        "placement_participation": spec.get("placement_participation"),
                        "solver_participation": spec.get("solver_participation"),
                        "relative_to": None,
                        "required_support_relaxed": True,
                    }
                )
    out.sort(
        key=lambda item: (
            -float(item.get("orientation_score") or 0.0),
            item["y"],
            item["x"],
            item["rot"],
        )
    )
    return out[:OBJECT_LEVEL_MAX_SUPPORT_SLOT_CANDIDATES]


def _support_slot_candidates(
    *,
    cluster_program: Mapping[str, Any],
    object_id: str,
    base_row: Mapping[str, Any],
    edge: Mapping[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    spec = _object_spec(cluster_program, object_id)
    if spec is None:
        return []
    dims = (
        spec.get("rep_dims_mm") if isinstance(spec.get("rep_dims_mm"), Mapping) else {}
    )
    length_mm = max(160, int(dims.get("L") or 0))
    width_mm = max(160, int(dims.get("W") or 0))
    allowed_rotations = (
        spec.get("allowed_rotations")
        if isinstance(spec.get("allowed_rotations"), Sequence)
        else [0, 90, 180, 270]
    )
    side_options = _normalized_support_side_options(edge)
    gap_min = int(edge.get("gap_min_mm") or edge.get("gap_min") or 60)
    gap_max = int(
        edge.get("gap_max_mm") or edge.get("gap_max") or max(gap_min + 50, 180)
    )
    base_rect = base_row["rect"]
    base_center = (
        (base_rect[0] + base_rect[2]) / 2.0,
        (base_rect[1] + base_rect[3]) / 2.0,
    )
    base_w = base_rect[2] - base_rect[0]
    base_h = base_rect[3] - base_rect[1]
    base_front = _front_vector_from_rotation(
        base_row.get("front_token"), int(base_row.get("rot") or 0)
    )
    side_vec = (-base_front[1], base_front[0])
    orientation = str(edge.get("orientation") or "").strip().lower()
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for side_option in side_options:
        slot_side_option = _bedside_nightstand_slot_option(
            side_option=side_option,
            object_id=object_id,
            base_row=base_row,
        )
        for gap in _support_gap_samples(gap_min=gap_min, gap_max=gap_max):
            for rot in allowed_rotations:
                final_rot = int(rot) % 360
                if orientation == "same_direction":
                    final_rot = int(base_row.get("rot") or 0) % 360
                w_mm, h_mm = _rotate_dims_for_rot(length_mm, width_mm, final_rot)
                base_offset = _slot_offset(
                    side_option=slot_side_option,
                    gap=gap,
                    base_w=base_w,
                    base_h=base_h,
                    obj_w=w_mm,
                    obj_h=h_mm,
                    front=base_front,
                    side=side_vec,
                )
                for slide in _support_slot_slide_offsets(
                    side_option=slot_side_option,
                    base_w=base_w,
                    base_h=base_h,
                    front=base_front,
                    side=side_vec,
                ):
                    x = _snap_to_grid(
                        int(
                            round(
                                base_center[0] + base_offset[0] + slide[0] - w_mm / 2.0
                            )
                        ),
                        grid_mm,
                    )
                    y = _snap_to_grid(
                        int(
                            round(
                                base_center[1] + base_offset[1] + slide[1] - h_mm / 2.0
                            )
                        ),
                        grid_mm,
                    )
                    key = (x, y, final_rot)
                    if key in seen:
                        continue
                    seen.add(key)
                    rect = (x, y, x + w_mm, y + h_mm)
                    desired_front = _desired_support_front_vector(
                        orientation=orientation,
                        rect=rect,
                        base_row=base_row,
                    )
                    front_world = _front_vector_from_rotation(
                        spec.get("front")
                        if isinstance(spec.get("front"), str)
                        else "top",
                        final_rot,
                    )
                    out.append(
                        {
                            "x": x,
                            "y": y,
                            "rot": final_rot,
                            "w": w_mm,
                            "h": h_mm,
                            "rect": rect,
                            "orientation_score": _front_alignment_score(
                                front_world=front_world,
                                desired_front=desired_front,
                            ),
                        }
                    )
    out.sort(
        key=lambda item: (
            -float(item.get("orientation_score") or 0.0),
            item["y"],
            item["x"],
            item["rot"],
        )
    )
    return out[:OBJECT_LEVEL_MAX_SUPPORT_SLOT_CANDIDATES]


def _bedside_nightstand_slot_option(
    *,
    side_option: str,
    object_id: str,
    base_row: Mapping[str, Any],
) -> str:
    token = str(side_option).strip().lower()
    if token not in {"head_left", "head_right"}:
        return side_option

    object_key = object_id.strip().lower()
    base_key = str(base_row.get("category") or base_row.get("object_id") or "").lower()
    if "nightstand" not in object_key or "bed" not in base_key:
        return side_option

    return f"bedside_{token}"


def _support_gap_samples(*, gap_min: int, gap_max: int) -> list[int]:
    lower = max(0, int(gap_min))
    upper = max(lower, int(gap_max))
    if lower == upper:
        return [lower]
    midpoint = int(round((lower + upper) / 2.0))
    return sorted({lower, midpoint, upper})


def _support_slot_slide_offsets(
    *,
    side_option: str,
    base_w: int,
    base_h: int,
    front: tuple[int, int],
    side: tuple[int, int],
) -> list[tuple[float, float]]:
    token = str(side_option).lower()
    if token in {"bedside_head_left", "bedside_head_right"}:
        span = float(base_h if abs(front[1]) == 1 else base_w)
        return [
            (front[0] * span * factor, front[1] * span * factor)
            for factor in (0.35, 0.2, 0.0)
        ]
    if token in {"left", "right"}:
        span = float(base_h if abs(front[1]) == 1 else base_w)
        return [
            (front[0] * span * factor, front[1] * span * factor)
            for factor in OBJECT_LEVEL_SUPPORT_ALIGNMENT_FACTORS
        ]
    if token in {
        "head",
        "front",
        "head_center",
        "head_left",
        "front_left",
        "head_right",
        "front_right",
    }:
        span = float(base_w if abs(side[0]) == 1 else base_h)
        return [
            (side[0] * span * factor, side[1] * span * factor)
            for factor in OBJECT_LEVEL_SUPPORT_ALIGNMENT_FACTORS
        ]
    return [(0.0, 0.0)]


def _desired_support_front_vector(
    *,
    orientation: str,
    rect: tuple[int, int, int, int],
    base_row: Mapping[str, Any],
) -> tuple[float, float] | None:
    token = orientation.strip().lower()
    if token == "same_direction":
        return _front_tuple(base_row.get("front_world"))
    if token in {"face_base", "face_cluster"}:
        base_rect = base_row.get("rect")
        if not isinstance(base_rect, tuple) or len(base_rect) != 4:
            return None
        return _vector_between_rect_centers(rect, base_rect)
    return None


def _slot_offset(
    *,
    side_option: str,
    gap: int,
    base_w: int,
    base_h: int,
    obj_w: int,
    obj_h: int,
    front: tuple[int, int],
    side: tuple[int, int],
) -> tuple[float, float]:
    front_clear = (
        (base_h / 2.0) + (obj_h / 2.0) + gap
        if abs(front[1]) == 1
        else (base_w / 2.0) + (obj_w / 2.0) + gap
    )
    side_clear = (
        (base_w / 2.0) + (obj_w / 2.0) + gap
        if abs(side[0]) == 1
        else (base_h / 2.0) + (obj_h / 2.0) + gap
    )
    token = str(side_option).lower()
    fx, fy = front
    sx, sy = side
    if token in {"head", "front", "head_center"}:
        return (fx * front_clear, fy * front_clear)
    if token in {"left", "bedside_head_left"}:
        return (sx * -side_clear, sy * -side_clear)
    if token in {"right", "bedside_head_right"}:
        return (sx * side_clear, sy * side_clear)
    if token in {"head_left", "front_left"}:
        return (
            fx * front_clear - sx * side_clear * 0.85,
            fy * front_clear - sy * side_clear * 0.85,
        )
    if token in {"head_right", "front_right"}:
        return (
            fx * front_clear + sx * side_clear * 0.85,
            fy * front_clear + sy * side_clear * 0.85,
        )
    return (fx * front_clear, fy * front_clear)


def _front_vector_from_rotation(front_token: Any, rot: int) -> tuple[int, int]:
    token = str(front_token or "top").strip().lower()
    base = {"top": (0, 1), "bottom": (0, -1), "left": (-1, 0), "right": (1, 0)}.get(
        token, (0, 1)
    )
    turns = (int(rot) % 360) // 90
    x, y = base
    for _ in range(turns):
        x, y = y, -x
    return (int(x), int(y))


def _materialize_object_row(
    cluster_program: Mapping[str, Any],
    object_id: str,
    x: int,
    y: int,
    rot: int,
    *,
    relative_to: str | None = None,
) -> dict[str, Any]:
    spec = _object_spec(cluster_program, object_id) or {}
    dims = (
        spec.get("rep_dims_mm") if isinstance(spec.get("rep_dims_mm"), Mapping) else {}
    )
    length_mm = max(160, int(dims.get("L") or 0))
    width_mm = max(160, int(dims.get("W") or 0))
    w_mm, h_mm = _rotate_dims_for_rot(length_mm, width_mm, rot)
    rect = (int(x), int(y), int(x) + w_mm, int(y) + h_mm)
    front_token = spec.get("front") if isinstance(spec.get("front"), str) else "top"
    front_world = _front_vector_from_rotation(front_token, rot)
    front_side_world = {
        (0, 1): "top",
        (0, -1): "bottom",
        (-1, 0): "left",
        (1, 0): "right",
    }.get(front_world)
    return {
        "cluster_id": str(cluster_program.get("cluster_id") or ""),
        "object_id": object_id,
        "category": str(spec.get("category") or object_id),
        "x": int(x),
        "y": int(y),
        "rot": int(rot) % 360,
        "rotation_ccw": int(rot) % 360,
        "w": w_mm,
        "h": h_mm,
        "rect": rect,
        "bbox": {
            "min_x": rect[0],
            "min_y": rect[1],
            "max_x": rect[2],
            "max_y": rect[3],
        },
        "center": {
            "x": int(round((rect[0] + rect[2]) / 2.0)),
            "y": int(round((rect[1] + rect[3]) / 2.0)),
        },
        "front_token": front_token,
        "front_world": {"dx": front_world[0], "dy": front_world[1]},
        "front_side_world": front_side_world,
        "relative_to": relative_to,
        "role": str(spec.get("role") or ""),
        "priority": str(spec.get("priority") or ""),
        "blocks_floor_area": spec.get("blocks_floor_area") is not False,
        "collision_layer": spec.get("collision_layer"),
        "place_on": deepcopy(spec.get("place_on"))
        if isinstance(spec.get("place_on"), Mapping)
        else None,
        "placement_participation": spec.get("placement_participation"),
        "solver_participation": spec.get("solver_participation"),
        "requires_front_access": _object_requires_front_access(
            cluster_program, object_id
        ),
    }


def _object_requires_front_access(
    cluster_program: Mapping[str, Any],
    object_id: str,
) -> bool:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    cluster_rules = (
        cluster_program.get("cluster_rules")
        if isinstance(cluster_program.get("cluster_rules"), Mapping)
        else {}
    )
    rows: list[Any] = []
    for source in (object_program, cluster_rules):
        raw_rows = (
            source.get("access_requirements") if isinstance(source, Mapping) else []
        )
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, str):
            rows.extend(raw_rows)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(
            row.get("id") or row.get("object_id") or row.get("target_id") or ""
        ).strip()
        if candidate_id != object_id:
            continue
        requirement_type = str(row.get("type") or row.get("mode") or "").strip().lower()
        if requirement_type not in {
            "",
            "access",
            "requires_access",
            "front_access",
            "front_clearance",
        }:
            continue
        if row.get("required") is False:
            continue
        return True
    return False


def _verify_object_level_solution(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    placed_objects = [
        row for row in solution.get("placed_objects") or [] if isinstance(row, Mapping)
    ]
    geometry_valid = True
    hard_valid = True
    offending_clusters: list[str] = []
    protected_regions = (
        world.get("protected_regions")
        if isinstance(world.get("protected_regions"), Sequence)
        else []
    )
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    protected_region_issues = _object_level_protected_region_issues(
        placed_objects=placed_objects,
        protected_regions=protected_regions,
    )
    forbidden_region_issues = _object_level_cluster_forbidden_region_issues(
        placed_objects=placed_objects,
        forbidden_regions=forbidden_regions,
    )
    wall_inset_violations = [
        issue
        for row in placed_objects
        for issue in _object_level_wall_inset_violations(row=row, world=world)
    ]
    blocking_protected_region_issues = [
        issue
        for issue in protected_region_issues
        if _protected_region_issue_is_blocking(issue)
    ]
    blocking_forbidden_region_issues = [
        issue
        for issue in forbidden_region_issues
        if _protected_region_issue_is_blocking(issue)
    ]
    protected_region_penalty = _object_level_protected_region_penalty(
        protected_region_issues
    )
    forbidden_region_penalty = _object_level_protected_region_penalty(
        forbidden_region_issues
    )
    if blocking_protected_region_issues:
        hard_valid = False
        offending_clusters.extend(
            str(issue.get("cluster_id") or "")
            for issue in blocking_protected_region_issues
        )
    if blocking_forbidden_region_issues:
        hard_valid = False
        offending_clusters.extend(
            str(issue.get("cluster_id") or "")
            for issue in blocking_forbidden_region_issues
        )
    for row in placed_objects:
        rect = row.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            continue
        if not _rect_inside_room_footprint(rect, world, row=row):
            geometry_valid = False
            hard_valid = False
            offending_clusters.append(str(row.get("cluster_id") or ""))
    for index, left in enumerate(placed_objects):
        for right in placed_objects[index + 1 :]:
            if _rects_overlap(left["rect"], right["rect"]):
                geometry_valid = False
                hard_valid = False
                offending_clusters.extend(
                    [
                        str(left.get("cluster_id") or ""),
                        str(right.get("cluster_id") or ""),
                    ]
                )
    orientation_issues = _object_level_orientation_issues(
        placed_objects=placed_objects,
        world=world,
        relation_plan=relation_plan,
    )
    blocking_orientation_issues = _object_level_blocking_orientation_issues(
        orientation_issues=orientation_issues,
        placed_objects=placed_objects,
    )
    if blocking_orientation_issues:
        hard_valid = False
        offending_clusters.extend(
            str(issue.get("cluster_id") or "") for issue in blocking_orientation_issues
        )
    face_pair_issues = _object_level_face_pair_issues(
        placed_objects=placed_objects,
        relation_plan=relation_plan,
    )
    if face_pair_issues:
        hard_valid = False
        for item in face_pair_issues:
            offending_clusters.extend(
                [
                    str(item.get("a") or ""),
                    str(item.get("b") or ""),
                ]
            )
    functional_geometry_issues = _object_level_functional_geometry_issues(
        placed_objects=placed_objects,
        world=world,
        relation_plan=relation_plan,
    )
    functional_geometry_issues.extend(
        _object_level_surface_seat_pair_issues(
            placed_objects=placed_objects,
            world=world,
        )
    )
    blocking_functional_geometry_issues = [
        issue
        for issue in functional_geometry_issues
        if str(issue.get("violation_severity") or "").strip().lower() == "blocking"
    ]
    if blocking_functional_geometry_issues:
        hard_valid = False
        offending_clusters.extend(
            str(issue.get("cluster_id") or "")
            for issue in blocking_functional_geometry_issues
        )
    expected_clusters = list(world["clusters_by_id"].keys())
    present_clusters = sorted(
        {
            str(row.get("cluster_id") or "")
            for row in placed_objects
            if str(row.get("cluster_id") or "").strip()
        }
    )
    missing_required_objects = _missing_required_object_rows(
        placed_objects=placed_objects,
        world=world,
    )
    if missing_required_objects:
        hard_valid = False
        offending_clusters.extend(
            str(item.get("cluster_id") or "") for item in missing_required_objects
        )
    missing_clusters = [
        cluster_id
        for cluster_id in expected_clusters
        if cluster_id not in present_clusters
        and not _solution_dropped_solver_trial_cluster(
            solution=solution,
            world=world,
            cluster_id=cluster_id,
        )
    ]
    dropped_trial_clusters = [
        cluster_id
        for cluster_id in expected_clusters
        if _solution_dropped_solver_trial_cluster(
            solution=solution,
            world=world,
            cluster_id=cluster_id,
        )
    ]
    coverage_denominator = max(
        1,
        len(
            [
                cluster_id
                for cluster_id in expected_clusters
                if cluster_id not in set(dropped_trial_clusters)
            ]
        ),
    )
    complete = geometry_valid and not missing_clusters and not missing_required_objects
    pair_ok = _object_level_primary_pair_ok(placed_objects, relation_plan) and not (
        face_pair_issues
    )
    interaction_quality = _object_level_interaction_quality(
        placed_objects=placed_objects,
        relation_plan=relation_plan,
    )
    interaction_quality_score = int(interaction_quality.get("score") or 0)
    critical_issue_count = sum(
        1
        for issue in protected_region_issues
        if str(issue.get("priority") or "").strip().lower() == "critical"
    )
    blocking_issue_count = (
        len(blocking_protected_region_issues)
        + len(blocking_forbidden_region_issues)
        + len(wall_inset_violations)
        + len(blocking_orientation_issues)
        + len(face_pair_issues)
        + len(blocking_functional_geometry_issues)
    )
    view_corridor_ok = not any(
        str(issue.get("reason") or "") == "view_corridor_blocked"
        for issue in blocking_functional_geometry_issues
    )
    front_access_ok = not any(
        str(issue.get("reason") or "") == "front_access_zone_blocked"
        for issue in blocking_functional_geometry_issues
    )
    gallery_eligible = (
        complete
        and hard_valid
        and critical_issue_count == 0
        and blocking_issue_count == 0
        and view_corridor_ok
        and front_access_ok
    )
    soft_issue_count = (
        len(orientation_issues)
        - len(blocking_orientation_issues)
        + len(protected_region_issues)
        - len(blocking_protected_region_issues)
        + len(forbidden_region_issues)
        - len(blocking_forbidden_region_issues)
        + len(functional_geometry_issues)
        - len(blocking_functional_geometry_issues)
    )
    soft_constraint_score = max(0, len(placed_objects) * 2 - soft_issue_count)
    quality_gate_reasons = _object_level_quality_gate_reasons(
        gallery_eligible=gallery_eligible,
        complete=complete,
        face_pair_issues=face_pair_issues,
        blocking_protected_region_issues=blocking_protected_region_issues,
        blocking_forbidden_region_issues=blocking_forbidden_region_issues,
        wall_inset_violations=wall_inset_violations,
        blocking_orientation_issues=blocking_orientation_issues,
        blocking_functional_geometry_issues=blocking_functional_geometry_issues,
    )
    return {
        "geometry_valid": geometry_valid,
        "hard_valid": hard_valid,
        "complete": complete,
        "gallery_eligible": gallery_eligible,
        "coverage_ratio": float(len(present_clusters) / coverage_denominator),
        "missing_cluster_ids": missing_clusters,
        "missing_required_object_ids": missing_required_objects,
        "dropped_trial_cluster_ids": dropped_trial_clusters,
        "present_cluster_ids": present_clusters,
        "offending_clusters": sorted(
            {cluster_id for cluster_id in offending_clusters if cluster_id}
        ),
        "primary_pair_ok": pair_ok,
        "orientation_issues": orientation_issues,
        "blocking_orientation_issues": blocking_orientation_issues,
        "protected_region_issues": protected_region_issues,
        "blocking_protected_region_issues": blocking_protected_region_issues,
        "forbidden_region_issues": forbidden_region_issues,
        "blocking_forbidden_region_issues": blocking_forbidden_region_issues,
        "wall_inset_violations": wall_inset_violations,
        "functional_geometry_issues": functional_geometry_issues,
        "blocking_functional_geometry_issues": blocking_functional_geometry_issues,
        "face_pair_issues": face_pair_issues,
        "critical_issue_count": critical_issue_count,
        "blocking_issue_count": blocking_issue_count,
        "view_corridor_ok": view_corridor_ok,
        "front_access_ok": front_access_ok,
        "protected_region_penalty": round(protected_region_penalty, 3),
        "forbidden_region_penalty": round(forbidden_region_penalty, 3),
        "soft_issue_count": soft_issue_count,
        "soft_constraint_score": soft_constraint_score,
        "interaction_quality_score": interaction_quality_score,
        "interaction_quality": interaction_quality,
        "quality_gate_reasons": quality_gate_reasons,
        "raw_room_bbox": list(
            world.get("raw_room_bbox") or world.get("room_bbox") or []
        ),
        "placement_room_bbox": list(world.get("placement_room_bbox") or []),
        "wall_insets_by_side": dict(world.get("wall_insets_by_side") or {}),
        "layout_score": int(
            1000
            + len(placed_objects) * 40
            + (200 if pair_ok else -120)
            + (600 if geometry_valid else -2000)
            + (200 if hard_valid else -500)
            + soft_constraint_score * 30
            - protected_region_penalty
            - forbidden_region_penalty
            - len(orientation_issues) * 30
            - len(functional_geometry_issues) * 90
            - blocking_issue_count * 1200
            + interaction_quality_score
        ),
    }


def _solution_dropped_solver_trial_cluster(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    cluster_id: str,
) -> bool:
    clusters_by_id = (
        world.get("clusters_by_id")
        if isinstance(world.get("clusters_by_id"), Mapping)
        else {}
    )
    cluster_program = clusters_by_id.get(cluster_id)
    if not isinstance(cluster_program, Mapping):
        return False
    if not _cluster_is_solver_trial_optional(cluster_program):
        return False
    dropped = solution.get("dropped_inventory_by_cluster")
    if not isinstance(dropped, Mapping):
        return False
    rows = dropped.get(cluster_id)
    return isinstance(rows, Sequence) and not isinstance(rows, str) and bool(rows)


def _object_level_protected_region_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    protected_regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in placed_objects:
        rect = row.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            continue
        cluster_id = str(row.get("cluster_id") or "")
        for protected in protected_regions:
            bbox = protected.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            applies_to = _string_sequence(protected.get("applies_to"))
            if (
                "support_clusters" in applies_to
                and "core_clusters" not in applies_to
                and str(row.get("anchor_kind") or "").strip().lower() == "center"
            ):
                continue
            policy = _protected_region_policy_for_cluster(
                protected,
                cluster_id=cluster_id,
            )
            ratio = _rect_overlap_ratio(rect, bbox)
            max_overlap = float(policy.get("max_overlap_ratio") or 0.0)
            if ratio <= max_overlap + 1e-9:
                continue
            overlap_area = _rect_overlap_area(rect, bbox)
            excess_overlap_ratio = max(0.0, ratio - max_overlap)
            priority = str(policy.get("priority") or "medium").strip().lower()
            enforcement = str(policy.get("enforcement") or "soft").strip().lower()
            violation_severity = (
                str(policy.get("violation_severity") or "advisory").strip().lower()
            )
            issues.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": str(row.get("object_id") or ""),
                    "region_id": str(protected.get("region_id") or ""),
                    "zone_type": str(policy.get("zone_type") or ""),
                    "priority": priority,
                    "enforcement": enforcement,
                    "violation_severity": violation_severity,
                    "overlap_ratio": round(ratio, 3),
                    "max_overlap_ratio": round(max_overlap, 3),
                    "excess_overlap_ratio": round(excess_overlap_ratio, 3),
                    "overlap_area_mm2": int(round(overlap_area)),
                    "blocking": _protected_region_policy_is_blocking(
                        priority=priority,
                        enforcement=enforcement,
                        violation_severity=violation_severity,
                    ),
                }
            )
    return issues


def _object_level_cluster_forbidden_region_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in placed_objects:
        rect = row.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            continue
        cluster_id = str(row.get("cluster_id") or "")
        object_id = str(row.get("object_id") or "")
        for forbidden in forbidden_regions:
            if str(forbidden.get("cluster_id") or "") != cluster_id:
                continue
            forbidden_object_id = str(forbidden.get("object_id") or "").strip()
            if forbidden_object_id and forbidden_object_id != object_id:
                continue
            bbox = forbidden.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            ratio = _rect_overlap_ratio(rect, bbox)
            max_overlap = float(forbidden.get("max_overlap_ratio") or 0.0)
            if ratio <= max_overlap + 1e-9:
                continue
            overlap_area = _rect_overlap_area(rect, bbox)
            excess_overlap_ratio = max(0.0, ratio - max_overlap)
            priority = str(forbidden.get("priority") or "high").strip().lower()
            enforcement = str(forbidden.get("enforcement") or "hard").strip().lower()
            violation_severity = (
                str(forbidden.get("violation_severity") or "blocking").strip().lower()
            )
            issues.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": object_id,
                    "region_id": str(forbidden.get("region_id") or ""),
                    "zone_type": str(
                        forbidden.get("zone_type") or "cluster_forbidden_region"
                    ),
                    "priority": priority,
                    "enforcement": enforcement,
                    "violation_severity": violation_severity,
                    "overlap_ratio": round(ratio, 3),
                    "max_overlap_ratio": round(max_overlap, 3),
                    "excess_overlap_ratio": round(excess_overlap_ratio, 3),
                    "overlap_area_mm2": int(round(overlap_area)),
                    "blocking": _protected_region_policy_is_blocking(
                        priority=priority,
                        enforcement=enforcement,
                        violation_severity=violation_severity,
                    ),
                }
            )
    return issues


def _object_level_wall_inset_violations(
    *,
    row: Mapping[str, Any],
    world: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _row_uses_wall_inset(row):
        return []
    rect = _rect_tuple(row.get("rect"))
    raw_bbox = _rect_tuple(world.get("room_bbox"))
    placement_bbox = _rect_tuple(world.get("placement_room_bbox"))
    if rect is None or raw_bbox is None or placement_bbox is None:
        return []
    if not _rect_inside_room(rect, raw_bbox):
        return []
    violated_sides: list[str] = []
    if rect[0] < placement_bbox[0]:
        violated_sides.append("left")
    if rect[1] < placement_bbox[1]:
        violated_sides.append("top")
    if rect[2] > placement_bbox[2]:
        violated_sides.append("right")
    if rect[3] > placement_bbox[3]:
        violated_sides.append("bottom")
    if not violated_sides and _rect_respects_wall_inner_faces(rect, world):
        return []
    return [
        {
            "cluster_id": str(row.get("cluster_id") or ""),
            "object_id": str(row.get("object_id") or ""),
            "reason": "wall_inner_face_overlap",
            "violation_severity": "blocking",
            "rect": list(rect),
            "raw_room_bbox": list(raw_bbox),
            "placement_room_bbox": list(placement_bbox),
            "violated_sides": violated_sides,
        }
    ]


def _object_level_protected_region_issue_count(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    protected_regions: Sequence[Mapping[str, Any]],
) -> int:
    return len(
        _object_level_protected_region_issues(
            placed_objects=placed_objects,
            protected_regions=protected_regions,
        )
    )


def _protected_region_policy_is_blocking(
    *,
    priority: str,
    enforcement: str,
    violation_severity: str,
) -> bool:
    return (
        priority in OBJECT_LEVEL_BLOCKING_PROTECTED_PRIORITIES
        and enforcement in OBJECT_LEVEL_BLOCKING_PROTECTED_ENFORCEMENT
        and violation_severity in OBJECT_LEVEL_BLOCKING_PROTECTED_SEVERITIES
    )


def _protected_region_issue_is_blocking(issue: Mapping[str, Any]) -> bool:
    if bool(issue.get("blocking")):
        return True
    return _protected_region_policy_is_blocking(
        priority=str(issue.get("priority") or "").strip().lower(),
        enforcement=str(issue.get("enforcement") or "").strip().lower(),
        violation_severity=str(issue.get("violation_severity") or "").strip().lower(),
    )


def _protected_region_priority_weight(priority: Any) -> float:
    token = str(priority or "medium").strip().lower()
    return {
        "critical": 8.0,
        "high": 5.0,
        "medium": 2.5,
        "low": 1.0,
    }.get(token, 2.0)


def _object_level_protected_region_penalty(
    protected_region_issues: Sequence[Mapping[str, Any]],
) -> float:
    penalty = 0.0
    for issue in protected_region_issues:
        excess = float(issue.get("excess_overlap_ratio") or 0.0)
        affected_area = float(issue.get("overlap_area_mm2") or 0.0)
        priority_weight = _protected_region_priority_weight(issue.get("priority"))
        penalty += priority_weight * excess * (affected_area / 800.0)
        if _protected_region_issue_is_blocking(issue):
            penalty += 900.0
    return penalty


def _object_level_blocking_orientation_issues(
    *,
    orientation_issues: Sequence[Mapping[str, Any]],
    placed_objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key = {
        (str(row.get("cluster_id") or ""), str(row.get("object_id") or "")): row
        for row in placed_objects
        if isinstance(row, Mapping)
    }
    blocking: list[dict[str, Any]] = []
    for issue in orientation_issues:
        if str(issue.get("reason") or "").strip().lower() not in {
            "near_wall_inward",
            "wall_contact_inward",
        }:
            continue
        row = rows_by_key.get(
            (str(issue.get("cluster_id") or ""), str(issue.get("object_id") or ""))
        )
        if row is not None and _row_requires_front_access_zone(row):
            blocking.append(dict(issue))
    return blocking


def _object_level_quality_gate_reasons(
    *,
    gallery_eligible: bool,
    complete: bool,
    face_pair_issues: Sequence[Mapping[str, Any]],
    blocking_protected_region_issues: Sequence[Mapping[str, Any]],
    blocking_forbidden_region_issues: Sequence[Mapping[str, Any]],
    wall_inset_violations: Sequence[Mapping[str, Any]] = (),
    blocking_orientation_issues: Sequence[Mapping[str, Any]],
    blocking_functional_geometry_issues: Sequence[Mapping[str, Any]],
) -> list[str]:
    if gallery_eligible:
        return []
    reasons: list[str] = []
    if face_pair_issues:
        issue_reasons = {
            str(issue.get("reason") or "").strip()
            for issue in face_pair_issues
            if isinstance(issue, Mapping)
        }
        if "view_target_contract_failed" in issue_reasons:
            reasons.append("view_target_contract_failed")
        if issue_reasons - {"view_target_contract_failed"}:
            reasons.append("required_face_contract_failed")
    if blocking_protected_region_issues:
        reasons.append("blocking_protected_region_overlap")
    if blocking_forbidden_region_issues:
        if any(
            _forbidden_issue_is_daylight_quality(issue)
            for issue in blocking_forbidden_region_issues
        ):
            reasons.append("blocking_daylight_quality_overlap")
    if any(
        not _forbidden_issue_is_daylight_quality(issue)
        for issue in blocking_forbidden_region_issues
    ):
        reasons.append("blocking_forbidden_region_overlap")
    if wall_inset_violations:
        reasons.append("wall_inner_face_overlap")
    if blocking_orientation_issues:
        reasons.append("front_inward_contract_failed")
    for issue in blocking_functional_geometry_issues:
        reason = str(issue.get("reason") or "").strip()
        if reason:
            reasons.append(reason)
    if not complete:
        reasons.append("layout_incomplete")
    if not reasons:
        reasons.append("soft_constraints_relaxed_for_ranking")
    return sorted(set(reasons))


def _forbidden_issue_is_daylight_quality(issue: Mapping[str, Any]) -> bool:
    region_id = str(issue.get("region_id") or "").strip()
    zone_type = str(issue.get("zone_type") or "").strip()
    return _region_id_is_daylight_quality_region(region_id) or zone_type in {
        "daylight_quality_blocking",
        "daylight_quality_soft",
    }


def _object_level_functional_geometry_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    room_bbox = _object_solver_fit_room_bbox(world)
    rows = [row for row in placed_objects if isinstance(row, Mapping)]
    issues: list[dict[str, Any]] = []

    for base in rows:
        access_zone = _front_access_zone_for_row(base, room_bbox)
        if access_zone is None:
            continue
        for blocker in rows:
            if blocker is base:
                continue
            blocker_rect = blocker.get("rect")
            if not isinstance(blocker_rect, tuple) or len(blocker_rect) != 4:
                continue
            ratio = _rect_overlap_ratio(blocker_rect, access_zone)
            if ratio <= 0.02:
                continue
            if _object_allowed_in_front_access_zone(blocker):
                continue
            issues.append(
                {
                    "cluster_id": str(blocker.get("cluster_id") or ""),
                    "object_id": str(blocker.get("object_id") or ""),
                    "blocked_cluster_id": str(base.get("cluster_id") or ""),
                    "blocked_object_id": str(base.get("object_id") or ""),
                    "reason": "front_access_zone_blocked",
                    "violation_severity": "blocking",
                    "overlap_ratio": round(ratio, 3),
                    "overlap_area_mm2": int(
                        round(_rect_overlap_area(blocker_rect, access_zone))
                    ),
                }
            )

    corridor = _primary_view_corridor_rect(
        placed_objects=rows,
        room_bbox=room_bbox,
        relation_plan=relation_plan,
    )
    if corridor is not None:
        corridor_rect, allowed_clusters = corridor
        for row in rows:
            cluster_id = str(row.get("cluster_id") or "")
            if cluster_id in allowed_clusters and _is_cluster_anchor_row(row):
                continue
            if _object_allowed_in_view_corridor(row):
                continue
            rect = row.get("rect")
            if not isinstance(rect, tuple) or len(rect) != 4:
                continue
            ratio = _rect_overlap_ratio(rect, corridor_rect)
            if ratio <= 0.04:
                continue
            issues.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": str(row.get("object_id") or ""),
                    "reason": "view_corridor_blocked",
                    "violation_severity": "blocking",
                    "overlap_ratio": round(ratio, 3),
                    "overlap_area_mm2": int(
                        round(_rect_overlap_area(rect, corridor_rect))
                    ),
                }
            )
    return issues


def _object_level_surface_seat_pair_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seated_surface_supports = _surface_seat_supports_by_cluster(world)
    if not seated_surface_supports:
        return []

    rows = [row for row in placed_objects if isinstance(row, Mapping)]
    placed_by_cluster_and_id = {
        (str(row.get("cluster_id") or ""), str(row.get("object_id") or ""))
        for row in rows
    }
    issues: list[dict[str, Any]] = []
    for row in rows:
        cluster_id = str(row.get("cluster_id") or "")
        object_id = str(row.get("object_id") or "")
        required_seats = seated_surface_supports.get(cluster_id, {}).get(
            object_id,
            set(),
        )
        if not required_seats:
            continue
        if any(
            (cluster_id, seat_id) in placed_by_cluster_and_id
            for seat_id in required_seats
        ):
            continue
        issues.append(
            {
                "cluster_id": cluster_id,
                "object_id": object_id,
                "reason": "missing_surface_seat_pair",
                "violation_severity": "blocking",
                "missing_support_ids": sorted(required_seats),
            }
        )
    return issues


def _front_access_zone_for_row(
    row: Mapping[str, Any],
    room_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if not _row_requires_front_access_zone(row):
        return None
    rect = row.get("rect")
    if not isinstance(rect, tuple) or len(rect) != 4:
        return None
    front = _dominant_cardinal_front(row.get("front_world"))
    if front is None:
        return None
    depth = OBJECT_LEVEL_FRONT_ACCESS_DEPTH_MM
    if front == (1, 0):
        zone = (rect[2], rect[1], rect[2] + depth, rect[3])
    elif front == (-1, 0):
        zone = (rect[0] - depth, rect[1], rect[0], rect[3])
    elif front == (0, 1):
        zone = (rect[0], rect[3], rect[2], rect[3] + depth)
    else:
        zone = (rect[0], rect[1] - depth, rect[2], rect[1])
    return _clip_rect_to_room(zone, room_bbox)


def _row_requires_front_access_zone(row: Mapping[str, Any]) -> bool:
    if bool(row.get("requires_front_access")):
        return True
    if not _is_cluster_anchor_row(row):
        return False
    tokens = _object_semantic_tokens(row)
    return bool(
        tokens
        & {
            "sofa",
            "loveseat",
            "settee",
            "sectional",
            "armchair",
            "chair",
            "recliner",
            "tv",
            "media",
            "console",
            "bookshelf",
            "bookcase",
            "shelving",
            "shelf",
            "cabinet",
        }
    )


def _object_allowed_in_front_access_zone(row: Mapping[str, Any]) -> bool:
    tokens = _object_semantic_tokens(row)
    return bool(tokens & {"coffee", "coffee_table", "ottoman", "rug"})


def _object_allowed_in_view_corridor(row: Mapping[str, Any]) -> bool:
    tokens = _object_semantic_tokens(row)
    return bool(tokens & {"coffee", "coffee_table", "ottoman", "rug"})


def _object_semantic_tokens(row: Mapping[str, Any]) -> set[str]:
    raw_values = [
        row.get("object_id"),
        row.get("category"),
        row.get("object_type"),
        row.get("role"),
    ]
    tokens: set[str] = set()
    for value in raw_values:
        normalized = str(value or "").strip().lower().replace("-", "_")
        if not normalized:
            continue
        tokens.add(normalized)
        tokens.update(part for part in normalized.split("_") if part)
    return tokens


def _dominant_cardinal_front(value: Any) -> tuple[int, int] | None:
    front = _front_tuple(value)
    if front is None:
        return None
    fx, fy = front
    if abs(fx) >= abs(fy) and abs(fx) > 1e-6:
        return (1 if fx > 0 else -1, 0)
    if abs(fy) > 1e-6:
        return (0, 1 if fy > 0 else -1)
    return None


def _primary_view_corridor_rect(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    room_bbox: tuple[int, int, int, int],
    relation_plan: Mapping[str, Any] | None,
) -> tuple[tuple[int, int, int, int], set[str]] | None:
    primary = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    secondary = _extract_layout_secondary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    if not primary or not secondary:
        return None
    by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in placed_objects:
        by_cluster.setdefault(str(row.get("cluster_id") or ""), []).append(row)
    left = _dominant_row_for_cluster(by_cluster.get(primary, []))
    right = _dominant_row_for_cluster(by_cluster.get(secondary, []))
    if left is None or right is None:
        return None
    left_rect = left.get("rect")
    right_rect = right.get("rect")
    if not (
        isinstance(left_rect, tuple)
        and len(left_rect) == 4
        and isinstance(right_rect, tuple)
        and len(right_rect) == 4
    ):
        return None
    lx = (left_rect[0] + left_rect[2]) / 2.0
    ly = (left_rect[1] + left_rect[3]) / 2.0
    rx = (right_rect[0] + right_rect[2]) / 2.0
    ry = (right_rect[1] + right_rect[3]) / 2.0
    half_width = OBJECT_LEVEL_VIEW_CORRIDOR_WIDTH_MM // 2
    if abs(rx - lx) >= abs(ry - ly):
        cy = int(round((ly + ry) / 2.0))
        rect = (
            int(round(min(lx, rx))),
            cy - half_width,
            int(round(max(lx, rx))),
            cy + half_width,
        )
    else:
        cx = int(round((lx + rx) / 2.0))
        rect = (
            cx - half_width,
            int(round(min(ly, ry))),
            cx + half_width,
            int(round(max(ly, ry))),
        )
    clipped = _clip_rect_to_room(rect, room_bbox)
    if clipped is None:
        return None
    return clipped, {primary, secondary}


def _object_level_orientation_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in placed_objects:
        by_cluster.setdefault(str(row.get("cluster_id") or ""), []).append(row)
    support_edges = _support_edges_by_cluster(world)
    room_bbox = _object_solver_fit_room_bbox(world)
    issues: list[dict[str, Any]] = []
    for row in placed_objects:
        cluster_id = str(row.get("cluster_id") or "")
        object_id = str(row.get("object_id") or "")
        rect = row.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            continue
        required_checks: list[tuple[str, tuple[float, float]]] = []
        edge = support_edges.get((cluster_id, object_id))
        orientation = str(edge.get("orientation") or "").strip().lower() if edge else ""
        if orientation in {"same_direction", "face_base", "face_cluster"}:
            base_id = str(edge.get("relative_to") or "")
            base_row = next(
                (
                    candidate
                    for candidate in by_cluster.get(cluster_id, [])
                    if str(candidate.get("object_id") or "") == base_id
                ),
                None,
            )
            if base_row is not None:
                desired_support_front = _desired_support_front_vector(
                    orientation=orientation,
                    rect=rect,
                    base_row=base_row,
                )
                if desired_support_front is not None:
                    required_checks.append((orientation, desired_support_front))
        intents = _object_level_orientation_intents(
            cluster_id=cluster_id,
            object_id=object_id,
            relation_plan=relation_plan,
            include_cluster=_is_cluster_anchor_row(row),
        )
        wall_contact_fronts = _wall_contact_inward_front_options(
            rect=rect,
            room_bbox=room_bbox,
        )
        front = _front_tuple(row.get("front_world"))
        if wall_contact_fronts and front is not None:
            wall_contact_front = _best_aligned_required_front(
                front_world=front,
                desired_fronts=wall_contact_fronts,
            )
            if (
                wall_contact_front is not None
                and not _front_matches_any_required_direction(
                    front_world=front,
                    desired_fronts=wall_contact_fronts,
                )
            ):
                required_checks.append(("wall_contact_inward", wall_contact_front))
        elif _row_requires_near_wall_inward_contract(
            row=row,
            relation_plan=relation_plan,
        ):
            near_wall_fronts = _near_wall_inward_front_options(
                rect=rect,
                room_bbox=room_bbox,
            )
            if near_wall_fronts and front is not None:
                near_wall_front = _best_aligned_required_front(
                    front_world=front,
                    desired_fronts=near_wall_fronts,
                )
                if (
                    near_wall_front is not None
                    and not _front_matches_any_required_direction(
                        front_world=front,
                        desired_fronts=near_wall_fronts,
                    )
                ):
                    required_checks.append(("near_wall_inward", near_wall_front))
        if "face_cluster" in intents:
            target_cluster_id = _object_level_target_cluster_id(
                cluster_id=cluster_id,
                object_id=object_id,
                relation_plan=relation_plan,
            )
            target_row = _dominant_row_for_cluster(
                by_cluster.get(target_cluster_id or "", [])
            )
            if target_row is not None:
                target_rect = target_row.get("rect")
                if isinstance(target_rect, tuple) and len(target_rect) == 4:
                    desired_cluster_front = _vector_between_rect_centers(
                        rect, target_rect
                    )
                    if desired_cluster_front is not None:
                        required_checks.append(("face_cluster", desired_cluster_front))
        if not required_checks:
            continue
        front = _front_tuple(row.get("front_world"))
        if front is None:
            continue
        for reason, desired_front in required_checks:
            dot = _dot_normalized(front, desired_front)
            if dot < OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT:
                issues.append(
                    {
                        "cluster_id": cluster_id,
                        "object_id": object_id,
                        "reason": reason,
                        "dot": round(dot, 3),
                        "front_world": {"dx": front[0], "dy": front[1]},
                        "desired_front_world": {
                            "dx": round(desired_front[0], 3),
                            "dy": round(desired_front[1], 3),
                        },
                    }
                )
    return issues


def _object_level_required_face_pairs(
    relation_plan: Mapping[str, Any] | None,
) -> set[tuple[str, str]]:
    if not isinstance(relation_plan, Mapping):
        return set()

    required = {
        pair
        for row in _object_level_required_viewing_pairs(relation_plan)
        if _viewing_pair_relation(row) in MUTUAL_FACE_RELATIONS
        if (
            pair := _normalized_cluster_pair(
                str(row.get("viewer_cluster_id") or ""),
                str(row.get("target_cluster_id") or ""),
            )
        )
        is not None
    }
    relations_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, Mapping):
            continue
        pair = _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        if pair is None:
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation:
            relations_by_pair.setdefault(pair, set()).add(relation)

    required.update(
        pair
        for pair, relations in relations_by_pair.items()
        if "face_each_other" in relations and len(relations) == 1
    )

    layout_intent = relation_plan.get("layout_intent_profile")
    if isinstance(layout_intent, Mapping):
        focus_mode = str(layout_intent.get("focus_mode") or "").strip().lower()
        primary = str(layout_intent.get("primary_cluster_id") or "").strip()
        secondary = str(layout_intent.get("secondary_cluster_id") or "").strip()
        pair = _normalized_cluster_pair(primary, secondary)
        if focus_mode == "viewing" and pair is not None:
            alternatives = relations_by_pair.get(pair, set())
            if alternatives and alternatives <= {"face_each_other"}:
                required.add(pair)
    requirements = relation_plan.get("concept_readiness_requirements")
    if isinstance(requirements, Mapping):
        required_contracts = _normalized_text_set(
            requirements.get("required_pair_contracts")
        )
        if required_contracts & {
            "face_each_other",
            "buffered_support",
            "legible_primary_pair",
            "focal_face_axis",
        }:
            primary = _extract_layout_primary_cluster_id(
                dict(relation_plan) if isinstance(relation_plan, dict) else None
            )
            secondary = _extract_layout_secondary_cluster_id(
                dict(relation_plan) if isinstance(relation_plan, dict) else None
            )
            pair = _normalized_cluster_pair(primary or "", secondary or "")
            if pair is not None:
                alternatives = relations_by_pair.get(pair, set())
                if alternatives and alternatives <= {"face_each_other"}:
                    required.add(pair)
    return required


def _object_level_required_viewing_pairs(
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(relation_plan, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for row in relation_plan.get("object_viewing_pairs") or []:
        if not isinstance(row, Mapping):
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation not in MUTUAL_FACE_RELATIONS | VIEW_TARGET_RELATIONS:
            continue
        if row.get("required") is False:
            continue
        viewer_cluster_id = str(row.get("viewer_cluster_id") or "").strip()
        target_cluster_id = str(row.get("target_cluster_id") or "").strip()
        viewer_object_id = str(row.get("viewer_object_id") or "").strip()
        target_object_id = str(row.get("target_object_id") or "").strip()
        if (
            not viewer_cluster_id
            or not target_cluster_id
            or not viewer_object_id
            or not target_object_id
            or viewer_cluster_id == target_cluster_id
        ):
            continue
        out.append(
            {
                "viewer_cluster_id": viewer_cluster_id,
                "target_cluster_id": target_cluster_id,
                "viewer_object_id": viewer_object_id,
                "target_object_id": target_object_id,
                "viewer_object_type": str(row.get("viewer_object_type") or "").strip(),
                "target_object_type": str(row.get("target_object_type") or "").strip(),
                "relation": "face_each_other"
                if relation in {"", "mutual_face"}
                else relation,
            }
        )
    return out


def _object_level_face_pair_issues(
    placed_objects: Sequence[Mapping[str, Any]],
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    object_viewing_pairs = _object_level_required_viewing_pairs(relation_plan)
    if object_viewing_pairs:
        return _object_level_viewing_pair_issues(
            placed_objects=placed_objects,
            object_viewing_pairs=object_viewing_pairs,
        )
    required_pairs = _object_level_required_face_pairs(relation_plan)
    if not required_pairs:
        return []
    rows_by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in placed_objects:
        rows_by_cluster.setdefault(str(row.get("cluster_id") or ""), []).append(row)

    issues: list[dict[str, Any]] = []
    for left_cluster_id, right_cluster_id in sorted(required_pairs):
        left = _dominant_row_for_cluster(rows_by_cluster.get(left_cluster_id, []))
        right = _dominant_row_for_cluster(rows_by_cluster.get(right_cluster_id, []))
        if left is None or right is None:
            continue
        left_rect = left.get("rect")
        right_rect = right.get("rect")
        if not (
            isinstance(left_rect, tuple)
            and len(left_rect) == 4
            and isinstance(right_rect, tuple)
            and len(right_rect) == 4
        ):
            continue
        left_desired = _vector_between_rect_centers(left_rect, right_rect)
        right_desired = _vector_between_rect_centers(right_rect, left_rect)
        left_front = _front_tuple(left.get("front_world"))
        right_front = _front_tuple(right.get("front_world"))
        left_ok = _front_matches_required_direction(
            front_world=left_front,
            desired_front=left_desired,
            min_dot=OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT,
        )
        right_ok = _front_matches_required_direction(
            front_world=right_front,
            desired_front=right_desired,
            min_dot=OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT,
        )
        if left_ok and right_ok:
            continue
        issues.append(
            {
                "a": left_cluster_id,
                "b": right_cluster_id,
                "reason": "required_face_each_other_failed",
                "left_dot": None
                if left_front is None or left_desired is None
                else round(_dot_normalized(left_front, left_desired), 3),
                "right_dot": None
                if right_front is None or right_desired is None
                else round(_dot_normalized(right_front, right_desired), 3),
            }
        )
    return issues


def _object_level_viewing_pair_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    object_viewing_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key = {
        (
            str(row.get("cluster_id") or ""),
            str(row.get("object_id") or ""),
        ): row
        for row in placed_objects
    }
    placed_cluster_ids = {cluster_id for cluster_id, _object_id in rows_by_key}
    issues: list[dict[str, Any]] = []
    for pair in object_viewing_pairs:
        relation = _viewing_pair_relation(pair)
        viewer_key = (
            str(pair.get("viewer_cluster_id") or ""),
            str(pair.get("viewer_object_id") or ""),
        )
        target_key = (
            str(pair.get("target_cluster_id") or ""),
            str(pair.get("target_object_id") or ""),
        )
        viewer = rows_by_key.get(viewer_key)
        target = rows_by_key.get(target_key)
        if viewer is None or target is None:
            if (
                viewer_key[0] not in placed_cluster_ids
                or target_key[0] not in placed_cluster_ids
            ):
                continue
            issues.append(
                {
                    "a": viewer_key[0],
                    "b": target_key[0],
                    "viewer_object_id": viewer_key[1],
                    "target_object_id": target_key[1],
                    "reason": "required_face_pair_missing_anchor",
                }
            )
            continue
        left_rect = viewer.get("rect")
        right_rect = target.get("rect")
        if not (
            isinstance(left_rect, tuple)
            and len(left_rect) == 4
            and isinstance(right_rect, tuple)
            and len(right_rect) == 4
        ):
            continue
        left_desired = _vector_between_rect_centers(left_rect, right_rect)
        right_desired = _vector_between_rect_centers(right_rect, left_rect)
        left_front = _front_tuple(viewer.get("front_world"))
        right_front = _front_tuple(target.get("front_world"))
        viewer_min_dot = (
            _view_target_viewer_min_dot(pair, viewer)
            if relation in VIEW_TARGET_RELATIONS
            else OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT
        )
        target_min_dot = (
            OBJECT_LEVEL_VIEW_TARGET_TARGET_MIN_DOT
            if relation in VIEW_TARGET_RELATIONS
            else OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT
        )
        left_ok = _front_matches_required_direction(
            front_world=left_front,
            desired_front=left_desired,
            min_dot=viewer_min_dot,
        )
        right_ok = _front_matches_required_direction(
            front_world=right_front,
            desired_front=right_desired,
            min_dot=target_min_dot,
        )
        if left_ok and right_ok:
            continue
        left_dot = None
        if left_front is not None and left_desired is not None:
            left_dot = round(_dot_normalized(left_front, left_desired), 3)
        right_dot = None
        if right_front is not None and right_desired is not None:
            right_dot = round(_dot_normalized(right_front, right_desired), 3)
        if relation in VIEW_TARGET_RELATIONS:
            issues.append(
                {
                    "a": viewer_key[0],
                    "b": target_key[0],
                    "viewer_cluster_id": viewer_key[0],
                    "target_cluster_id": target_key[0],
                    "viewer_object_id": viewer_key[1],
                    "target_object_id": target_key[1],
                    "relation": relation,
                    "reason": "view_target_contract_failed",
                    "viewer_dot": left_dot,
                    "target_dot": right_dot,
                    "viewer_min_dot": viewer_min_dot,
                    "target_min_dot": target_min_dot,
                    "left_dot": left_dot,
                    "right_dot": right_dot,
                }
            )
            continue
        issues.append(
            {
                "a": viewer_key[0],
                "b": target_key[0],
                "viewer_object_id": viewer_key[1],
                "target_object_id": target_key[1],
                "reason": "required_face_each_other_failed",
                "left_dot": left_dot,
                "right_dot": right_dot,
            }
        )
    return issues


def _object_level_interaction_quality(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    viewing_rows = _object_level_view_target_quality_rows(
        placed_objects=placed_objects,
        object_viewing_pairs=_object_level_required_viewing_pairs(relation_plan),
    )
    workflow_rows = _object_level_workflow_quality_rows(placed_objects)
    score = sum(int(row.get("score") or 0) for row in viewing_rows)
    score += sum(int(row.get("score") or 0) for row in workflow_rows)
    return {
        "score": int(score),
        "view_targets": viewing_rows,
        "workflow_pairs": workflow_rows,
    }


def _object_level_view_target_quality_rows(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    object_viewing_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key = {
        (
            str(row.get("cluster_id") or ""),
            str(row.get("object_id") or ""),
        ): row
        for row in placed_objects
    }
    quality_rows: list[dict[str, Any]] = []
    for pair in object_viewing_pairs:
        relation = _viewing_pair_relation(pair)
        viewer = rows_by_key.get(
            (
                str(pair.get("viewer_cluster_id") or ""),
                str(pair.get("viewer_object_id") or ""),
            )
        )
        target = rows_by_key.get(
            (
                str(pair.get("target_cluster_id") or ""),
                str(pair.get("target_object_id") or ""),
            )
        )
        if viewer is None or target is None:
            continue
        viewer_rect = viewer.get("rect")
        target_rect = target.get("rect")
        if not (
            isinstance(viewer_rect, tuple)
            and len(viewer_rect) == 4
            and isinstance(target_rect, tuple)
            and len(target_rect) == 4
        ):
            continue
        viewer_desired = _vector_between_rect_centers(viewer_rect, target_rect)
        target_desired = _vector_between_rect_centers(target_rect, viewer_rect)
        viewer_front = _front_tuple(viewer.get("front_world"))
        target_front = _front_tuple(target.get("front_world"))
        if viewer_desired is None or target_desired is None:
            continue
        viewer_dot = (
            _dot_normalized(viewer_front, viewer_desired)
            if viewer_front is not None
            else 0.0
        )
        target_dot = (
            _dot_normalized(target_front, target_desired)
            if target_front is not None
            else 0.0
        )
        viewer_center = _rect_center(viewer_rect)
        target_center = _rect_center(target_rect)
        distance = math.hypot(
            viewer_center[0] - target_center[0],
            viewer_center[1] - target_center[1],
        )
        axis_alignment = max(abs(viewer_desired[0]), abs(viewer_desired[1]))
        viewer_min_dot = (
            _view_target_viewer_min_dot(pair, viewer)
            if relation in VIEW_TARGET_RELATIONS
            else OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT
        )
        target_min_dot = (
            OBJECT_LEVEL_VIEW_TARGET_TARGET_MIN_DOT
            if relation in VIEW_TARGET_RELATIONS
            else OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT
        )
        score = _interaction_dot_score(viewer_dot, viewer_min_dot, weight=260)
        score += _interaction_dot_score(target_dot, target_min_dot, weight=260)
        score += int(round(axis_alignment * 160.0))
        if (
            OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MIN_DISTANCE_MM
            <= distance
            <= OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MAX_DISTANCE_MM
        ):
            score += 120
        else:
            score -= min(
                220,
                int(
                    abs(
                        distance
                        - (
                            OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MIN_DISTANCE_MM
                            if distance
                            < OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MIN_DISTANCE_MM
                            else OBJECT_LEVEL_VIEW_TARGET_PREFERRED_MAX_DISTANCE_MM
                        )
                    )
                    / 10.0
                ),
            )
        quality_rows.append(
            {
                "viewer_cluster_id": str(pair.get("viewer_cluster_id") or ""),
                "target_cluster_id": str(pair.get("target_cluster_id") or ""),
                "viewer_object_id": str(pair.get("viewer_object_id") or ""),
                "target_object_id": str(pair.get("target_object_id") or ""),
                "relation": relation,
                "viewer_dot": round(viewer_dot, 3),
                "target_dot": round(target_dot, 3),
                "axis_alignment": round(axis_alignment, 3),
                "distance_mm": round(distance, 1),
                "score": int(score),
            }
        )
    return quality_rows


def _interaction_dot_score(dot: float, min_dot: float, *, weight: int) -> int:
    if dot >= min_dot:
        headroom = max(1.0 - min_dot, 0.01)
        return int(round(weight * (0.65 + 0.35 * (dot - min_dot) / headroom)))
    return -int(round(weight * min((min_dot - dot) / max(min_dot, 0.01), 1.0)))


def _object_level_workflow_quality_rows(
    placed_objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in placed_objects:
        rows_by_cluster.setdefault(str(row.get("cluster_id") or ""), []).append(row)

    quality_rows: list[dict[str, Any]] = []
    for cluster_id, rows in rows_by_cluster.items():
        desks = [row for row in rows if _object_row_is_desk(row)]
        chairs = [row for row in rows if _object_row_is_work_chair(row)]
        if not desks or not chairs:
            continue
        desk = desks[0]
        for chair in chairs:
            chair_rect = chair.get("rect")
            desk_rect = desk.get("rect")
            if not (
                isinstance(chair_rect, tuple)
                and len(chair_rect) == 4
                and isinstance(desk_rect, tuple)
                and len(desk_rect) == 4
            ):
                continue
            desired = _vector_between_rect_centers(chair_rect, desk_rect)
            chair_front = _front_tuple(chair.get("front_world"))
            if desired is None or chair_front is None:
                continue
            dot = _dot_normalized(chair_front, desired)
            chair_center = _rect_center(chair_rect)
            desk_center = _rect_center(desk_rect)
            distance = math.hypot(
                chair_center[0] - desk_center[0],
                chair_center[1] - desk_center[1],
            )
            score = _interaction_dot_score(dot, 0.65, weight=220)
            score += 100 if 450.0 <= distance <= 1300.0 else -80
            quality_rows.append(
                {
                    "cluster_id": cluster_id,
                    "surface_object_id": str(desk.get("object_id") or ""),
                    "seat_object_id": str(chair.get("object_id") or ""),
                    "chair_to_surface_dot": round(dot, 3),
                    "distance_mm": round(distance, 1),
                    "score": int(score),
                }
            )
    return quality_rows


def _object_row_type_tokens(row: Mapping[str, Any]) -> set[str]:
    return _object_type_tokens(
        row.get("object_id"),
        row.get("category"),
        row.get("asset_type"),
        row.get("object_type"),
    )


def _object_row_is_desk(row: Mapping[str, Any]) -> bool:
    tokens = _object_row_type_tokens(row)
    return "desk" in tokens and "lamp" not in tokens


def _object_row_is_work_chair(row: Mapping[str, Any]) -> bool:
    tokens = _object_row_type_tokens(row)
    return bool(tokens & {"chair", "office_chair", "desk_chair"})


def _support_edges_by_cluster(
    world: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    clusters_by_id = (
        world.get("clusters_by_id")
        if isinstance(world.get("clusters_by_id"), Mapping)
        else {}
    )
    for cluster_id, cluster_program in clusters_by_id.items():
        if not isinstance(cluster_program, Mapping):
            continue
        object_program = (
            cluster_program.get("object_program")
            if isinstance(cluster_program.get("object_program"), Mapping)
            else {}
        )
        for edge in object_program.get("support_edges") or []:
            if not isinstance(edge, Mapping):
                continue
            object_id = str(edge.get("object_id") or "").strip()
            if object_id:
                out[(str(cluster_id), object_id)] = edge
    return out


def _is_cluster_anchor_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("priority") or "") in {
        "anchor",
        "dominant_anchor",
    } or not row.get("relative_to")


def _dominant_row_for_cluster(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if str(row.get("priority") or "") in {"anchor", "dominant_anchor"}
        ),
        rows[0] if rows else None,
    )


def _front_tuple(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        try:
            return (float(value.get("dx")), float(value.get("dy")))
        except (TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _vector_between_rect_centers(
    source: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    sx = (source[0] + source[2]) / 2.0
    sy = (source[1] + source[3]) / 2.0
    tx = (target[0] + target[2]) / 2.0
    ty = (target[1] + target[3]) / 2.0
    dx, dy = tx - sx, ty - sy
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return (dx / norm, dy / norm)


def _object_level_primary_pair_ok(
    placed_objects: Sequence[Mapping[str, Any]], relation_plan: Mapping[str, Any] | None
) -> bool:
    if _object_level_required_viewing_pairs(relation_plan):
        return not _object_level_face_pair_issues(
            placed_objects=placed_objects,
            relation_plan=relation_plan,
        )
    primary = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    secondary = _extract_layout_secondary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    if not primary or not secondary:
        return True
    left = next(
        (
            row
            for row in placed_objects
            if row.get("cluster_id") == primary
            and str(row.get("priority") or "") in {"anchor", "dominant_anchor"}
        ),
        None,
    )
    if left is None:
        left = next(
            (row for row in placed_objects if row.get("cluster_id") == primary), None
        )
    right = next(
        (
            row
            for row in placed_objects
            if row.get("cluster_id") == secondary
            and str(row.get("priority") or "") in {"anchor", "dominant_anchor"}
        ),
        None,
    )
    if right is None:
        right = next(
            (row for row in placed_objects if row.get("cluster_id") == secondary), None
        )
    if left is None or right is None:
        return False
    lc = left.get("center") if isinstance(left.get("center"), Mapping) else {}
    rc = right.get("center") if isinstance(right.get("center"), Mapping) else {}
    try:
        lx, ly = float(lc.get("x")), float(lc.get("y"))
        rx, ry = float(rc.get("x")), float(rc.get("y"))
    except (TypeError, ValueError):
        return False
    dx, dy = rx - lx, ry - ly
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return False
    direction = (dx / norm, dy / norm)
    left_front = (
        left.get("front_world") if isinstance(left.get("front_world"), Mapping) else {}
    )
    right_front = (
        right.get("front_world")
        if isinstance(right.get("front_world"), Mapping)
        else {}
    )
    try:
        left_vec = (float(left_front.get("dx")), float(left_front.get("dy")))
        right_vec = (float(right_front.get("dx")), float(right_front.get("dy")))
    except (TypeError, ValueError):
        return False
    pair = _normalized_cluster_pair(primary, secondary)
    min_dot = (
        OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT
        if pair is not None and pair in _object_level_required_face_pairs(relation_plan)
        else OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT
    )
    return _front_matches_required_direction(
        front_world=left_vec,
        desired_front=direction,
        min_dot=min_dot,
    ) and _front_matches_required_direction(
        front_world=right_vec,
        desired_front=(-direction[0], -direction[1]),
        min_dot=min_dot,
    )


def _object_dropped_inventory_count(solution: Mapping[str, Any]) -> int:
    dropped = solution.get("dropped_inventory_by_cluster")
    if not isinstance(dropped, Mapping):
        return 0
    return sum(
        len(rows)
        for rows in dropped.values()
        if isinstance(rows, Sequence) and not isinstance(rows, str)
    )


def _object_pre_repair_score(solution: Mapping[str, Any]) -> float:
    return (
        float(solution.get("anchor_score") or 0.0)
        + float(solution.get("support_score") or 0.0)
        - 220.0 * float(_object_dropped_inventory_count(solution))
        - 120.0 * float(solution.get("protected_region_issue_count") or 0.0)
    )


def _object_pre_repair_rank_key(solution: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(solution.get("pre_repair_score") or 0.0),
        float(solution.get("anchor_score") or 0.0),
        float(solution.get("support_score") or 0.0),
        -float(_object_dropped_inventory_count(solution)),
        -float(solution.get("protected_region_issue_count") or 0.0),
    )


def _object_repair_pool_is_sufficient(
    solution_pool: Sequence[Mapping[str, Any]],
) -> bool:
    if len(solution_pool) < OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS:
        return False
    ranked = _rank_object_level_solution_pool(solution_pool)[
        :OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS
    ]
    if len(ranked) < OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS:
        return False
    best = ranked[0] if ranked else {}
    verify = best.get("verify") if isinstance(best.get("verify"), Mapping) else {}
    return bool(verify.get("gallery_eligible"))


def _macro_cluster_solver_degradation_report(
    *,
    raw_solution_count: int,
    eligible_solutions: Sequence[Mapping[str, Any]],
    partial_solutions: Sequence[Mapping[str, Any]],
    anchor_candidate_counts: Mapping[str, int],
    anchor_order: Sequence[str],
    status: str,
    failure_kind: str = "",
    anchor_pair_compatibility: Sequence[Mapping[str, Any]] | None = None,
    pairwise_incompatible_anchor_pairs: Sequence[Mapping[str, Any]] | None = None,
    anchor_wall_options_by_cluster: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    all_solutions = [*eligible_solutions, *partial_solutions]
    dropped_summary = _object_level_dropped_inventory_summary(all_solutions)
    hard_valid_count = sum(
        1
        for solution in all_solutions
        if isinstance(solution.get("verify"), Mapping)
        and bool(solution["verify"].get("hard_valid"))
    )
    gallery_eligible_count = sum(
        1
        for solution in all_solutions
        if isinstance(solution.get("verify"), Mapping)
        and bool(solution["verify"].get("gallery_eligible"))
    )
    zero_anchor_candidate_clusters = sorted(
        cluster_id
        for cluster_id, count in anchor_candidate_counts.items()
        if int(count) <= 0
    )
    report = {
        "status": status,
        "attempts": [
            {
                "attempt_id": "full_program",
                "raw_solution_count": int(raw_solution_count),
                "geometry_valid_candidate_count": len(all_solutions),
                "hard_valid_candidate_count": hard_valid_count,
                "gallery_eligible_candidate_count": gallery_eligible_count,
                "status": "OK" if gallery_eligible_count else "UNSAT",
            },
            {
                "attempt_id": "drop_solver_trial_optional_clusters",
                "applied": any(
                    str(item.get("reason") or "")
                    == "optional_trial_anchor_cluster_not_placed"
                    for item in dropped_summary
                ),
                "dropped": [
                    item
                    for item in dropped_summary
                    if str(item.get("reason") or "")
                    == "optional_trial_anchor_cluster_not_placed"
                ],
            },
            {
                "attempt_id": "drop_non_requested_optional_supports",
                "applied": any(
                    str(item.get("reason") or "")
                    in {"base_missing", "support_slot_failed"}
                    for item in dropped_summary
                ),
                "dropped": [
                    item
                    for item in dropped_summary
                    if str(item.get("reason") or "")
                    in {"base_missing", "support_slot_failed"}
                ],
            },
            {
                "attempt_id": "drop_target_if_viable_soft_objects",
                "applied": False,
                "reason": "Object-level solver can drop objects only when the merged object program marks them droppable.",
            },
            {
                "attempt_id": "shrink_unspecified_hard_objects",
                "applied": False,
                "reason": "Skipped because merged object specs contain selected dimensions, not alternate tier dimensions.",
            },
            {
                "attempt_id": "relax_optional_face_pair_contracts",
                "applied": True,
                "reason": "Face-pair checks ignore missing clusters; completeness remains responsible for required hard clusters.",
            },
        ],
        "raw_solution_count": int(raw_solution_count),
        "geometry_valid_candidate_count": len(all_solutions),
        "hard_valid_candidate_count": hard_valid_count,
        "gallery_eligible_candidate_count": gallery_eligible_count,
        "eligible_candidate_count": len(eligible_solutions),
        "partial_candidate_count": len(partial_solutions),
        "top_quality_gate_reasons": _object_level_top_quality_gate_reasons(
            all_solutions
        ),
        "blocking_objects": _object_level_blocking_object_summary(all_solutions),
        "dropped_inventory": dropped_summary,
        "anchor_candidate_counts": dict(anchor_candidate_counts),
        "anchor_wall_options_by_cluster": deepcopy(
            dict(anchor_wall_options_by_cluster or {})
        ),
        "zero_anchor_candidate_clusters": zero_anchor_candidate_clusters,
        "anchor_order": list(anchor_order),
    }
    if failure_kind:
        report["failure_kind"] = failure_kind
    if anchor_pair_compatibility is not None:
        report["anchor_pair_compatibility"] = [
            dict(row) for row in anchor_pair_compatibility if isinstance(row, Mapping)
        ]
    if pairwise_incompatible_anchor_pairs is not None:
        report["pairwise_incompatible_anchor_pairs"] = [
            dict(row)
            for row in pairwise_incompatible_anchor_pairs
            if isinstance(row, Mapping)
        ]
    return report


def _object_level_top_quality_gate_reasons(
    solutions: Sequence[Mapping[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for solution in solutions:
        verify = (
            solution.get("verify")
            if isinstance(solution.get("verify"), Mapping)
            else {}
        )
        for reason in verify.get("quality_gate_reasons") or []:
            token = str(reason or "").strip()
            if token:
                counts[token] = counts.get(token, 0) + 1
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def _object_level_blocking_object_summary(
    solutions: Sequence[Mapping[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    issue_keys = (
        "blocking_protected_region_issues",
        "blocking_forbidden_region_issues",
        "blocking_orientation_issues",
        "blocking_functional_geometry_issues",
    )
    for solution in solutions:
        verify = (
            solution.get("verify")
            if isinstance(solution.get("verify"), Mapping)
            else {}
        )
        for issue_key in issue_keys:
            for issue in verify.get(issue_key) or []:
                if not isinstance(issue, Mapping):
                    continue
                reason = str(
                    issue.get("reason") or issue.get("zone_type") or issue_key
                ).strip()
                cluster_id = str(
                    issue.get("cluster_id") or issue.get("blocked_cluster_id") or ""
                ).strip()
                object_id = str(
                    issue.get("object_id") or issue.get("blocked_object_id") or ""
                ).strip()
                region_id = str(issue.get("region_id") or "").strip()
                key = (cluster_id, object_id, region_id, reason)
                counts[key] = counts.get(key, 0) + 1
    return [
        {
            "cluster_id": cluster_id,
            "object_id": object_id,
            "region_id": region_id,
            "reason": reason,
            "count": count,
        }
        for (cluster_id, object_id, region_id, reason), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def _object_level_dropped_inventory_summary(
    solutions: Sequence[Mapping[str, Any]],
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = {}
    for solution in solutions:
        dropped = solution.get("dropped_inventory_by_cluster")
        if not isinstance(dropped, Mapping):
            continue
        for cluster_id, rows in dropped.items():
            if not isinstance(rows, Sequence) or isinstance(rows, str):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                key = (
                    str(cluster_id),
                    str(row.get("object_id") or ""),
                    str(row.get("reason") or ""),
                )
                counts[key] = counts.get(key, 0) + 1
    return [
        {
            "cluster_id": cluster_id,
            "object_id": object_id,
            "reason": reason,
            "count": count,
        }
        for (cluster_id, object_id, reason), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def _object_solution_score(
    solution: Mapping[str, Any], relation_plan: Mapping[str, Any] | None
) -> float:
    verify = (
        solution.get("verify") if isinstance(solution.get("verify"), Mapping) else {}
    )
    score = float(solution.get("anchor_score") or 0.0)
    score += float(verify.get("layout_score") or 0.0)
    if verify.get("gallery_eligible"):
        score += 5000.0
    elif verify.get("hard_valid"):
        score += 1000.0
    score -= 250.0 * len(
        [
            item
            for rows in (solution.get("dropped_inventory_by_cluster") or {}).values()
            for item in rows
        ]
    )
    score += float(solution.get("support_score") or 0.0)
    score -= 80.0 * float(solution.get("protected_region_issue_count") or 0.0)
    score -= float(verify.get("protected_region_penalty") or 0.0)
    score -= float(verify.get("forbidden_region_penalty") or 0.0)
    score -= 1200.0 * float(verify.get("blocking_issue_count") or 0.0)
    score -= _secondary_storage_primary_proximity_penalty(solution)
    return score


def _secondary_storage_primary_proximity_penalty(solution: Mapping[str, Any]) -> float:
    placed_objects = [
        row for row in solution.get("placed_objects") or [] if isinstance(row, Mapping)
    ]
    storage_rows = [
        row
        for row in placed_objects
        if _object_row_is_visual_storage_like(row)
        and str(row.get("priority") or "").strip() in {"anchor", "dominant_anchor"}
    ]
    if not storage_rows:
        return 0.0
    primary_rows = [row for row in placed_objects if _object_row_is_primary_anchor(row)]
    penalty = 0.0
    for storage in storage_rows:
        storage_rect = _rect_tuple(storage.get("rect"))
        if storage_rect is None:
            continue
        for primary in primary_rows:
            primary_rect = _rect_tuple(primary.get("rect"))
            if primary_rect is None:
                continue
            if _rects_overlap(storage_rect, _expand_rect(primary_rect, 180)):
                penalty += OBJECT_LEVEL_SECONDARY_STORAGE_PRIMARY_PROXIMITY_PENALTY
    return penalty


def _object_row_is_visual_storage_like(row: Mapping[str, Any]) -> bool:
    tokens = (
        str(row.get("object_id") or ""),
        str(row.get("category") or ""),
        str(row.get("base_object_id") or ""),
    )
    for token in tokens:
        normalized = token.strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            continue
        if is_profile_storage_object(normalized):
            return True
        if any(part in normalized for part in _VISUAL_STORAGE_OBJECT_TOKENS):
            return True
    return False


def _object_row_is_primary_anchor(row: Mapping[str, Any]) -> bool:
    # object_id = str(row.get("object_id") or "").strip().lower()
    # if object_id in {"bed", "desk"}:
    #     return True
    cluster_id = str(row.get("cluster_id") or "").strip().lower()
    priority = str(row.get("priority") or "").strip()
    return priority in {"anchor", "dominant_anchor"} and (
        "sleep" in cluster_id or "work" in cluster_id or "study" in cluster_id
    )


def _expand_rect(
    rect: tuple[int, int, int, int],
    padding_mm: int,
) -> tuple[int, int, int, int]:
    padding = max(0, int(padding_mm))
    return (
        rect[0] - padding,
        rect[1] - padding,
        rect[2] + padding,
        rect[3] + padding,
    )


def _rank_object_level_solution_pool(
    solutions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(item) for item in solutions if isinstance(item, Mapping)),
        key=lambda item: (
            bool((item.get("verify") or {}).get("hard_valid")),
            int((item.get("verify") or {}).get("critical_issue_count") or 0) == 0,
            int((item.get("verify") or {}).get("blocking_issue_count") or 0) == 0,
            bool((item.get("verify") or {}).get("view_corridor_ok")),
            bool((item.get("verify") or {}).get("front_access_ok")),
            bool((item.get("verify") or {}).get("complete")),
            bool((item.get("verify") or {}).get("primary_pair_ok")),
            int((item.get("verify") or {}).get("interaction_quality_score") or 0),
            int((item.get("verify") or {}).get("soft_constraint_score") or 0),
            -int((item.get("verify") or {}).get("soft_issue_count") or 0),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        signature = str(item.get("signature") or "")
        if signature and signature in seen:
            continue
        if signature:
            seen.add(signature)
        out.append(item)
    return out


def _object_level_solution_signature(solution: Mapping[str, Any]) -> str:
    rows = [
        row
        for row in (solution.get("placed_objects") or [])
        if isinstance(row, Mapping)
    ]
    tokens = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("cluster_id") or ""),
            str(item.get("object_id") or ""),
        ),
    ):
        bbox = row.get("bbox") if isinstance(row.get("bbox"), Mapping) else {}
        tokens.append(
            "|".join(
                [
                    str(row.get("cluster_id") or ""),
                    str(row.get("object_id") or ""),
                    str(int(row.get("rotation_ccw") or 0)),
                    str(int(bbox.get("min_x") or 0)),
                    str(int(bbox.get("min_y") or 0)),
                    str(int(bbox.get("max_x") or 0)),
                    str(int(bbox.get("max_y") or 0)),
                ]
            )
        )
    return "\n".join(tokens)


def _build_object_level_solution_payload(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    solution_index: int,
) -> dict[str, Any]:
    verify = (
        solution.get("verify") if isinstance(solution.get("verify"), Mapping) else {}
    )
    absolute_layout = _build_absolute_layout_from_object_solution(
        solution=solution,
        world=world,
        room_model=room_model,
        relation_plan=relation_plan,
    )
    return {
        "solution_id": f"object_sol_{solution_index:02d}",
        "absolute_layout": absolute_layout,
        "layout_score": int(verify.get("layout_score") or 0),
        "hard_valid": bool(verify.get("hard_valid")),
        "geometry_valid": bool(verify.get("geometry_valid")),
        "complete": bool(verify.get("complete")),
        "gallery_eligible": bool(verify.get("gallery_eligible")),
        "coverage_ratio": float(verify.get("coverage_ratio") or 0.0),
        "missing_cluster_ids": list(verify.get("missing_cluster_ids") or []),
        "missing_required_object_ids": deepcopy(
            verify.get("missing_required_object_ids") or []
        ),
        "verify_summary": deepcopy(dict(verify)),
        "dropped_inventory_by_cluster": deepcopy(
            solution.get("dropped_inventory_by_cluster") or {}
        ),
        "state_signature": str(solution.get("signature") or ""),
        "notes": [
            "Object-level solver kept this geometry-valid arrangement for ranked constraint search.",
            f"soft_issue_count={int(verify.get('soft_issue_count') or 0)}",
            f"blocking_issue_count={int(verify.get('blocking_issue_count') or 0)}",
        ],
    }


def _build_absolute_layout_from_object_solution(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    verify = (
        solution.get("verify") if isinstance(solution.get("verify"), Mapping) else {}
    )
    placed_objects = [
        deepcopy(row)
        for row in (solution.get("placed_objects") or [])
        if isinstance(row, Mapping)
    ]
    clusters: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in placed_objects:
        grouped.setdefault(str(row.get("cluster_id") or ""), []).append(row)
    for cluster_id, rows in grouped.items():
        anchor_row = next(
            (
                row
                for row in rows
                if str(row.get("priority") or "") in {"anchor", "dominant_anchor"}
            ),
            rows[0] if rows else {},
        )
        bbox = _union_bboxes(
            [tuple(row["rect"]) for row in rows if isinstance(row.get("rect"), tuple)]
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "object_ids": [str(row.get("object_id") or "") for row in rows],
                "dominant_anchor_id": str(anchor_row.get("object_id") or ""),
                "bbox": {
                    "min_x": bbox[0],
                    "min_y": bbox[1],
                    "max_x": bbox[2],
                    "max_y": bbox[3],
                }
                if bbox is not None
                else {},
            }
        )
    coverage = {
        "expected_cluster_ids": list(world["clusters_by_id"].keys()),
        "present_cluster_ids": list(verify.get("present_cluster_ids") or []),
        "missing_cluster_ids": list(verify.get("missing_cluster_ids") or []),
    }
    notes = [
        "Absolute layout was emitted directly by the object-level solver.",
        "Cluster local composition and phase-2 repair were bypassed in this flow.",
    ]
    geometry_repair = (
        solution.get("geometry_repair")
        if isinstance(solution.get("geometry_repair"), Mapping)
        else {}
    )
    if geometry_repair and geometry_repair.get("status") == "repaired":
        notes.append(
            "Backend geometry guardrail repaired polygon bounds and overlap before styling."
        )
    return {
        "status": "OK" if verify.get("hard_valid") else "PARTIAL",
        "layout_kind": "object_level_anchor_first",
        "room_id": _room_id(room_model),
        "room_type": _room_type(room_model),
        "room": deepcopy(
            room_model.get("room")
            if isinstance(room_model.get("room"), Mapping)
            else {}
        ),
        "openings": deepcopy(
            room_model.get("openings")
            if isinstance(room_model.get("openings"), Mapping)
            else {}
        ),
        "affordance_map": deepcopy(
            room_model.get("affordance_map")
            if isinstance(room_model.get("affordance_map"), Mapping)
            else {}
        ),
        "topology": deepcopy(
            room_model.get("topology")
            if isinstance(room_model.get("topology"), Mapping)
            else {}
        ),
        "objects": deepcopy(placed_objects),
        "placements": deepcopy(placed_objects),
        "object_placements": deepcopy(placed_objects),
        "clusters": clusters,
        "cluster_instances": deepcopy(clusters),
        "coverage": coverage,
        "complete": bool(verify.get("complete")),
        "gallery_eligible": bool(verify.get("gallery_eligible")),
        "coverage_ratio": float(verify.get("coverage_ratio") or 0.0),
        "missing_cluster_ids": list(verify.get("missing_cluster_ids") or []),
        "missing_required_object_ids": deepcopy(
            verify.get("missing_required_object_ids") or []
        ),
        "notes": notes,
        "solver_debug": {
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan),
            "dropped_inventory_by_cluster": deepcopy(
                solution.get("dropped_inventory_by_cluster") or {}
            ),
            "verify_summary": deepcopy(verify),
            "geometry_repair": deepcopy(solution.get("geometry_repair") or {}),
            "geometry": _object_solver_geometry_diagnostics(world),
        },
    }


def _rect_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 4:
        return None
    try:
        rect = tuple(int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return None
    return (rect[0], rect[1], rect[2], rect[3])


def _rect_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)


def _rect_inside_room_footprint(
    rect: tuple[int, int, int, int],
    world: Mapping[str, Any],
    *,
    row: Mapping[str, Any] | None = None,
) -> bool:
    fit_bbox = _object_solver_fit_room_bbox(world, row=row)
    if not _rect_inside_room(rect, fit_bbox):
        return False
    uses_wall_inset = _row_uses_wall_inset(row)
    if bool(world.get("room_polygon_is_axis_aligned_rectangle")):
        return True
    footprint_cache = (
        world.get("_rect_inside_room_footprint_cache")
        if isinstance(world.get("_rect_inside_room_footprint_cache"), dict)
        else None
    )
    cache_key = (rect, uses_wall_inset)
    if footprint_cache is not None and cache_key in footprint_cache:
        return bool(footprint_cache[cache_key])
    polygon = world.get("room_polygon")
    if (
        not isinstance(polygon, Sequence)
        or isinstance(polygon, str)
        or len(polygon) < 3
    ):
        return True
    points: list[tuple[float, float]] = []
    for item in polygon:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            return True
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            return True
    polygon_edges = (
        world.get("room_polygon_edges")
        if isinstance(world.get("room_polygon_edges"), Sequence)
        and not isinstance(world.get("room_polygon_edges"), str)
        else ()
    )
    result = _rect_inside_polygon_with_edges(rect, points, polygon_edges)
    if result and uses_wall_inset:
        result = _rect_respects_wall_inner_faces(rect, world)
    if footprint_cache is not None:
        footprint_cache[cache_key] = result
    return result


def _object_solver_fit_room_bbox(
    world: Mapping[str, Any],
    *,
    row: Mapping[str, Any] | None = None,
) -> tuple[int, int, int, int]:
    if _row_uses_wall_inset(row):
        placement_bbox = _rect_tuple(world.get("placement_room_bbox"))
        if placement_bbox is not None:
            return placement_bbox
    raw_bbox = _rect_tuple(world.get("room_bbox"))
    return raw_bbox if raw_bbox is not None else (0, 0, 0, 0)


def _row_uses_wall_inset(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return True
    if row.get("blocks_floor_area") is False:
        return False
    collision_layer = str(row.get("collision_layer") or "").strip().lower()
    if collision_layer in {"annotation", "ceiling", "opening", "visual", "wall"}:
        return False
    place_on = row.get("place_on")
    if isinstance(place_on, Mapping):
        method = str(place_on.get("method") or "").strip().lower()
        if method in {
            "attach_to",
            "attach_to_wall",
            "hang_on",
            "mount_on",
            "on_ceiling",
            "on_door",
            "on_wall",
            "on_window",
        }:
            return False
    placement_participation = (
        str(row.get("placement_participation") or "").strip().lower()
    )
    if placement_participation in {
        "ceiling_mounted",
        "decorative_non_blocking",
        "non_floor",
        "opening_attached",
        "visual_only",
        "wall_mounted",
    }:
        return False
    solver_participation = str(row.get("solver_participation") or "").strip().lower()
    if solver_participation in {"decorative_only", "non_solver_object", "visual_only"}:
        return False
    return True


def _rect_respects_wall_inner_faces(
    rect: tuple[int, int, int, int],
    world: Mapping[str, Any],
) -> bool:
    faces = world.get("wall_inner_faces")
    if not isinstance(faces, Sequence) or isinstance(faces, str):
        return True
    for point in _rect_probe_points(rect):
        for face in faces:
            if not isinstance(face, Mapping):
                continue
            if _point_violates_wall_inner_face(point, face):
                return False
    return True


def _point_violates_wall_inner_face(
    point: tuple[float, float],
    face: Mapping[str, Any],
) -> bool:
    start = face.get("start")
    tangent = face.get("tangent")
    normal = face.get("inward_normal")
    if (
        not isinstance(start, Sequence)
        or isinstance(start, str)
        or len(start) != 2
        or not isinstance(tangent, Sequence)
        or isinstance(tangent, str)
        or len(tangent) != 2
        or not isinstance(normal, Sequence)
        or isinstance(normal, str)
        or len(normal) != 2
    ):
        return False
    try:
        px, py = float(point[0]), float(point[1])
        sx, sy = float(start[0]), float(start[1])
        tx, ty = float(tangent[0]), float(tangent[1])
        nx, ny = float(normal[0]), float(normal[1])
        length = float(face.get("length_mm") or 0.0)
        inset = float(face.get("inset_mm") or 0.0)
    except (TypeError, ValueError):
        return False
    if inset <= 0.0 or length <= 0.0:
        return False
    vx = px - sx
    vy = py - sy
    projected = vx * tx + vy * ty
    if projected < -1e-6 or projected > length + 1e-6:
        return False
    signed_distance = vx * nx + vy * ny
    return signed_distance + 1e-6 < inset


def _rect_room_footprint_reject_reason(
    *,
    rect: tuple[int, int, int, int],
    world: Mapping[str, Any],
    row: Mapping[str, Any] | None = None,
) -> str:
    raw_bbox = _rect_tuple(world.get("room_bbox"))
    if raw_bbox is not None and not _rect_inside_room(rect, raw_bbox):
        return "outside_room"
    if _row_uses_wall_inset(row):
        placement_bbox = _rect_tuple(world.get("placement_room_bbox"))
        if placement_bbox is not None and not _rect_inside_room(rect, placement_bbox):
            return "wall_inner_face"
    if not bool(
        world.get("room_polygon_is_axis_aligned_rectangle")
    ) and not _rect_inside_room_footprint(rect, world, row=row):
        if _row_uses_wall_inset(row) and not _rect_respects_wall_inner_faces(
            rect,
            world,
        ):
            return "wall_inner_face"
        return "outside_room_polygon"
    return "outside_room"


def _rect_inside_polygon(
    rect: tuple[int, int, int, int], polygon: Sequence[tuple[float, float]]
) -> bool:
    if len(polygon) < 3:
        return True
    polygon_edges = [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]
    return _rect_inside_polygon_with_edges(rect, polygon, polygon_edges)


def _rect_inside_polygon_with_edges(
    rect: tuple[int, int, int, int],
    polygon: Sequence[tuple[float, float]],
    polygon_edges: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> bool:
    if len(polygon) < 3:
        return True
    probe_points = _rect_probe_points(rect)
    if any(not _point_in_or_on_polygon(point, polygon) for point in probe_points):
        return False
    rect_edges = _rect_edges(rect)
    edges = (
        polygon_edges
        if polygon_edges
        else [
            (polygon[index], polygon[(index + 1) % len(polygon)])
            for index in range(len(polygon))
        ]
    )
    return not any(
        _segments_strictly_cross(rect_edge[0], rect_edge[1], poly_edge[0], poly_edge[1])
        for rect_edge in rect_edges
        for poly_edge in edges
    )


def _rect_probe_points(
    rect: tuple[int, int, int, int],
) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = rect
    cx, cy = _rect_center(rect)
    return (
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
        (cx, cy),
        (cx, y1),
        (x2, cy),
        (cx, y2),
        (x1, cy),
    )


def _rect_edges(
    rect: tuple[int, int, int, int],
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
]:
    x1, y1, x2, y2 = rect
    return (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )


def _point_in_or_on_polygon(
    point: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> bool:
    if any(
        _point_on_segment(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ):
        return True
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        if (current[1] > y) != (previous[1] > y):
            denom = previous[1] - current[1]
            if abs(denom) > 1e-12:
                x_intersect = (previous[0] - current[0]) * (
                    y - current[1]
                ) / denom + current[0]
                if x < x_intersect:
                    inside = not inside
        j = i
    return inside


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    eps: float = 1e-6,
) -> bool:
    cross = (point[1] - start[1]) * (end[0] - start[0]) - (point[0] - start[0]) * (
        end[1] - start[1]
    )
    if abs(cross) > eps:
        return False
    dot_product = (point[0] - start[0]) * (end[0] - start[0]) + (
        point[1] - start[1]
    ) * (end[1] - start[1])
    if dot_product < -eps:
        return False
    squared_len = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
    return dot_product <= squared_len + eps


def _segments_strictly_cross(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    *,
    eps: float = 1e-6,
) -> bool:
    if (
        _point_on_segment(a, c, d, eps=eps)
        or _point_on_segment(b, c, d, eps=eps)
        or _point_on_segment(c, a, b, eps=eps)
        or _point_on_segment(d, a, b, eps=eps)
    ):
        return False
    ab_c = _orient(a, b, c)
    ab_d = _orient(a, b, d)
    cd_a = _orient(c, d, a)
    cd_b = _orient(c, d, b)
    return (ab_c * ab_d < -eps) and (cd_a * cd_b < -eps)


def _orient(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _snap_to_grid(value: int, grid_mm: int) -> int:
    grid = max(1, int(grid_mm))
    return int(round(float(value) / float(grid))) * grid


def _snap_origin_inside_room(
    *,
    x: int,
    y: int,
    w_mm: int,
    h_mm: int,
    room_bbox: tuple[int, int, int, int],
    grid_mm: int,
) -> tuple[int, int]:
    return (
        _snap_value_inside_span(
            value=x,
            lower=room_bbox[0],
            upper=room_bbox[2] - w_mm,
            grid_mm=grid_mm,
        ),
        _snap_value_inside_span(
            value=y,
            lower=room_bbox[1],
            upper=room_bbox[3] - h_mm,
            grid_mm=grid_mm,
        ),
    )


def _snap_value_inside_span(
    *,
    value: int,
    lower: int,
    upper: int,
    grid_mm: int,
) -> int:
    grid = max(1, int(grid_mm))
    if upper < lower:
        return _snap_to_grid(value, grid)
    snapped = _snap_to_grid(value, grid)
    if lower <= snapped <= upper:
        return snapped
    lower_grid = int(math.ceil(float(lower) / float(grid))) * grid
    upper_grid = int(math.floor(float(upper) / float(grid))) * grid
    candidates = [
        candidate
        for candidate in (
            int(math.floor(float(value) / float(grid))) * grid,
            int(math.ceil(float(value) / float(grid))) * grid,
            lower_grid,
            upper_grid,
        )
        if lower <= candidate <= upper
    ]
    if candidates:
        return min(candidates, key=lambda candidate: abs(candidate - value))
    return max(lower, min(upper, value))


def _rect_inside_room(
    rect: tuple[int, int, int, int], room_bbox: tuple[int, int, int, int]
) -> bool:
    return (
        rect[0] >= room_bbox[0]
        and rect[1] >= room_bbox[1]
        and rect[2] <= room_bbox[2]
        and rect[3] <= room_bbox[3]
    )


def _rects_overlap(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    return (
        left[0] < right[2]
        and left[2] > right[0]
        and left[1] < right[3]
        and left[3] > right[1]
    )


def _rect_overlap_ratio(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = float((ix2 - ix1) * (iy2 - iy1))
    area = float(max(1, (left[2] - left[0]) * (left[3] - left[1])))
    return intersection / area


def _rect_overlap_area(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def _clip_rect_to_room(
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    clipped = (
        max(rect[0], room_bbox[0]),
        max(rect[1], room_bbox[1]),
        min(rect[2], room_bbox[2]),
        min(rect[3], room_bbox[3]),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped
