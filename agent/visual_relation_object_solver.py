from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM

from agent.solver import solver as object_solver

_SOLVER_KIND = "visual_relation_object_solver_v2"
_STRICT_SOLVER_KIND = "object_level_anchor_first"
_MAX_ANCHOR_SOLUTIONS_ENV = "TKNT_VISUAL_RELATION_SOLVER_MAX_ANCHOR_SOLUTIONS"
_MAX_ANCHOR_CANDIDATES_ENV = "TKNT_VISUAL_RELATION_SOLVER_MAX_ANCHOR_CANDIDATES"
_MAX_SUPPORT_CANDIDATES_ENV = "TKNT_VISUAL_RELATION_SOLVER_MAX_SUPPORT_CANDIDATES"
_MAX_OUTPUT_SOLUTIONS_ENV = "TKNT_VISUAL_RELATION_SOLVER_MAX_OUTPUT_SOLUTIONS"
_DEFAULT_MAX_ANCHOR_SOLUTIONS = 72
_DEFAULT_MAX_ANCHOR_CANDIDATES = 60
_DEFAULT_MAX_SUPPORT_CANDIDATES = 12
_DEFAULT_MAX_OUTPUT_SOLUTIONS = 24


@dataclass(frozen=True)
class VisualRelationObjectSolver:
    max_anchor_solutions: int | None = None
    max_anchor_candidates_per_cluster: int | None = None
    max_support_candidates_per_object: int | None = None
    max_output_solutions: int | None = None

    def generate_object_layout(
        self,
        *,
        room_model_json: dict[str, Any],
        merged_clusters_json: dict[str, Any],
        relation_plan_json: dict[str, Any] | None,
        cluster_constraints_json: dict[str, Any] | None = None,
        grid_mm: int = GLOBAL_LAYOUT_GRID_MM,
    ) -> dict[str, Any]:
        strict_output = object_solver.MacroClusterSolver().generate_object_layout(
            room_model_json=room_model_json,
            merged_clusters_json=merged_clusters_json,
            relation_plan_json=relation_plan_json,
            cluster_constraints_json=cluster_constraints_json,
            grid_mm=grid_mm,
        )
        strict_lock_violations = _strict_output_semantic_anchor_lock_violations(
            output=strict_output,
            room_model=room_model_json,
            relation_plan=relation_plan_json,
            grid_mm=grid_mm,
        )
        if _strict_output_is_publishable(strict_output) and not strict_lock_violations:
            out = deepcopy(strict_output)
            out["solver_kind"] = _SOLVER_KIND
            out["visual_solver_diagnostics"] = {
                "strict_solver_kind": _STRICT_SOLVER_KIND,
                "strict_solver_status": str(strict_output.get("status") or ""),
                "used_visual_fallback": False,
                "semantic_anchor_locks": _semantic_anchor_lock_summary(
                    relation_plan_json
                ),
                "anchor_lock_violations": [],
                "hard_lock_rejected_count": 0,
                "unplaceable_locked_objects": [],
                "support_fallback_counts": {},
                "dropped_optional_inventory": deepcopy(
                    strict_output.get("dropped_inventory_by_cluster") or {}
                ),
                "rejected_protected_items": [],
            }
            solver_debug = (
                out.get("solver_debug")
                if isinstance(out.get("solver_debug"), dict)
                else {}
            )
            solver_debug["visual_solver_diagnostics"] = deepcopy(
                out["visual_solver_diagnostics"]
            )
            out["solver_debug"] = solver_debug
            notes = list(out.get("notes") or [])
            notes.append("Visual solver V2 reused the strict object-level solution.")
            out["notes"] = _unique_strings(notes)
            return out

        return _solve_visual_friendly_layout(
            room_model=room_model_json,
            merged_clusters=merged_clusters_json,
            relation_plan=relation_plan_json,
            cluster_constraints=cluster_constraints_json,
            grid_mm=grid_mm,
            strict_output=strict_output,
            strict_lock_violations=strict_lock_violations,
            max_anchor_solutions=self.max_anchor_solutions
            or _int_env(_MAX_ANCHOR_SOLUTIONS_ENV, _DEFAULT_MAX_ANCHOR_SOLUTIONS),
            max_anchor_candidates_per_cluster=self.max_anchor_candidates_per_cluster
            or _int_env(
                _MAX_ANCHOR_CANDIDATES_ENV,
                _DEFAULT_MAX_ANCHOR_CANDIDATES,
            ),
            max_support_candidates_per_object=self.max_support_candidates_per_object
            or _int_env(
                _MAX_SUPPORT_CANDIDATES_ENV,
                _DEFAULT_MAX_SUPPORT_CANDIDATES,
            ),
            max_output_solutions=self.max_output_solutions
            or _int_env(_MAX_OUTPUT_SOLUTIONS_ENV, _DEFAULT_MAX_OUTPUT_SOLUTIONS),
        )


def _solve_visual_friendly_layout(
    *,
    room_model: Mapping[str, Any],
    merged_clusters: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    cluster_constraints: Mapping[str, Any] | None,
    grid_mm: int,
    strict_output: Mapping[str, Any],
    strict_lock_violations: Sequence[Mapping[str, Any]],
    max_anchor_solutions: int,
    max_anchor_candidates_per_cluster: int,
    max_support_candidates_per_object: int,
    max_output_solutions: int,
) -> dict[str, Any]:
    world = object_solver._build_object_solver_world(
        room_model=room_model,
        merged_clusters=merged_clusters,
        relation_plan=relation_plan,
        cluster_constraints=cluster_constraints,
        grid_mm=grid_mm,
    )
    anchor_order = list(world["anchor_cluster_order"])
    anchor_candidates_by_cluster = {
        cluster_id: _rank_anchor_candidates_for_visual_hints(
            object_solver._generate_anchor_pose_candidates(
                cluster_program=world["clusters_by_id"][cluster_id],
                room_model=room_model,
                relation_plan=relation_plan,
                world=world,
                grid_mm=grid_mm,
            ),
            cluster_id=cluster_id,
            relation_plan=relation_plan,
            world=world,
            room_model=room_model,
        )
        for cluster_id in anchor_order
    }
    diagnostics: dict[str, Any] = {
        "strict_solver_kind": _STRICT_SOLVER_KIND,
        "strict_solver_status": str(strict_output.get("status") or ""),
        "strict_solver_notes": list(strict_output.get("notes") or []),
        "strict_anchor_lock_violations": [dict(row) for row in strict_lock_violations],
        "used_visual_fallback": True,
        "semantic_anchor_locks": _semantic_anchor_lock_summary(relation_plan),
        "anchor_lock_violations": [],
        "hard_lock_rejected_count": 0,
        "unplaceable_locked_objects": _locked_anchor_objects_with_no_candidates(
            anchor_order=anchor_order,
            anchor_candidates_by_cluster=anchor_candidates_by_cluster,
            world=world,
            relation_plan=relation_plan,
        ),
        "anchor_order": anchor_order,
        "anchor_candidate_counts": {
            key: len(value) for key, value in anchor_candidates_by_cluster.items()
        },
        "support_fallback_counts": {},
        "dropped_optional_inventory": {},
        "rejected_protected_items": [],
    }
    anchor_solutions = _search_visual_anchor_solutions(
        anchor_order=anchor_order,
        anchor_candidates_by_cluster=anchor_candidates_by_cluster,
        world=world,
        max_solutions=max_anchor_solutions,
        max_candidates_per_cluster=max_anchor_candidates_per_cluster,
    )
    diagnostics["anchor_solution_count"] = len(anchor_solutions)
    if not anchor_solutions:
        return _unsat_output(
            relation_plan=relation_plan,
            grid_mm=grid_mm,
            diagnostics=diagnostics,
            notes=[
                "Visual solver V2 could not assemble a usable non-overlapping anchor set."
            ],
        )

    raw_pool: list[dict[str, Any]] = []
    for anchor_solution in anchor_solutions:
        support_results = _place_visual_support_objects_for_solution(
            solution=anchor_solution,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            grid_mm=grid_mm,
            max_solutions=max_support_candidates_per_object,
            max_support_candidates_per_object=max_support_candidates_per_object,
        )
        for support_result in support_results:
            candidate_solution = {**anchor_solution, **support_result}
            candidate_solution["pre_repair_score"] = (
                object_solver._object_pre_repair_score(candidate_solution)
            )
            candidate_solution["pre_repair_signature"] = (
                object_solver._object_level_solution_signature(candidate_solution)
            )
            raw_pool.append(candidate_solution)

    raw_pool.sort(key=object_solver._object_pre_repair_rank_key, reverse=True)
    diagnostics["raw_solution_count"] = len(raw_pool)
    solution_pool: list[dict[str, Any]] = []
    for candidate_solution in raw_pool:
        repair_summary = object_solver._repair_object_level_solution_geometry(
            solution=candidate_solution,
            world=world,
            grid_mm=grid_mm,
        )
        if repair_summary is None:
            continue
        candidate_solution["placed_objects"] = repair_summary["placed_objects"]
        candidate_solution["geometry_repair"] = repair_summary["summary"]
        lock_violations = _semantic_anchor_lock_violations(
            placed_objects=repair_summary["placed_objects"],
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
        )
        if lock_violations:
            diagnostics["hard_lock_rejected_count"] = int(
                diagnostics.get("hard_lock_rejected_count") or 0
            ) + 1
            _extend_unique_violations(
                diagnostics["anchor_lock_violations"],
                lock_violations,
            )
            _extend_unique_strings(
                diagnostics["unplaceable_locked_objects"],
                _locked_object_labels(lock_violations),
            )
            continue
        verify = object_solver._verify_object_level_solution(
            solution=candidate_solution,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
        )
        object_solver._apply_object_level_geometry_repair_penalty(
            verify,
            candidate_solution["geometry_repair"],
        )
        verify = _relax_visual_paired_front_access_issues(
            verify=verify,
            solution=candidate_solution,
            world=world,
            relation_plan=relation_plan,
        )
        if not bool(verify.get("geometry_valid")):
            continue
        candidate_solution["verify"] = verify
        candidate_solution["score"] = _visual_solution_score(
            candidate_solution,
            relation_plan,
        )
        candidate_solution["signature"] = (
            object_solver._object_level_solution_signature(candidate_solution)
        )
        solution_pool.append(candidate_solution)
        if len(solution_pool) >= max(1, int(max_output_solutions)) * 3:
            break

    ranked_pool = object_solver._rank_object_level_solution_pool(solution_pool)[
        : max(1, int(max_output_solutions))
    ]
    diagnostics["geometry_valid_solution_count"] = len(solution_pool)
    diagnostics["hard_valid_solution_count"] = sum(
        1
        for item in solution_pool
        if isinstance(item.get("verify"), Mapping)
        and bool(item["verify"].get("hard_valid"))
    )
    diagnostics["gallery_eligible_solution_count"] = sum(
        1
        for item in solution_pool
        if isinstance(item.get("verify"), Mapping)
        and bool(item["verify"].get("gallery_eligible"))
    )
    diagnostics["top_failure_reasons"] = _top_failure_reasons(solution_pool)
    diagnostics["ranked_solution_count"] = len(ranked_pool)
    diagnostics["candidate_count"] = len(ranked_pool)
    best_solution = ranked_pool[0] if ranked_pool else None
    if best_solution is None:
        return _unsat_output(
            relation_plan=relation_plan,
            grid_mm=grid_mm,
            diagnostics=diagnostics,
            notes=[
                "Visual solver V2 anchor placement succeeded, but no geometry-valid support arrangement survived verification."
            ],
        )

    best_trace = (
        best_solution.get("visual_solver_trace")
        if isinstance(best_solution.get("visual_solver_trace"), Mapping)
        else {}
    )
    diagnostics["support_fallback_counts"] = dict(
        best_trace.get("support_fallback_counts") or {}
    )
    diagnostics["dropped_optional_inventory"] = deepcopy(
        best_solution.get("dropped_inventory_by_cluster") or {}
    )
    diagnostics["rejected_protected_items"] = _dedupe_rejections(
        best_trace.get("rejected_protected_items") or []
    )
    _extend_unique_violations(
        diagnostics["anchor_lock_violations"],
        _mapping_rows(best_trace.get("anchor_lock_violations")),
    )
    diagnostics["hard_lock_rejected_count"] = int(
        diagnostics.get("hard_lock_rejected_count") or 0
    ) + int(best_trace.get("hard_lock_rejected_count") or 0)
    _extend_unique_strings(
        diagnostics["unplaceable_locked_objects"],
        _string_sequence(best_trace.get("unplaceable_locked_objects")),
    )
    absolute_layout = object_solver._build_absolute_layout_from_object_solution(
        solution=best_solution,
        world=world,
        room_model=room_model,
        relation_plan=relation_plan,
    )
    ranked_solutions = [
        _build_visual_solution_payload(
            solution=item,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            solution_index=index,
        )
        for index, item in enumerate(ranked_pool, start=1)
    ]
    for payload in ranked_solutions:
        payload["visual_solver_diagnostics"] = deepcopy(diagnostics)
    verify = (
        best_solution.get("verify")
        if isinstance(best_solution.get("verify"), Mapping)
        else {}
    )
    solver_debug = {
        "anchor_candidate_counts": diagnostics["anchor_candidate_counts"],
        "anchor_order": anchor_order,
        "object_count": len(best_solution.get("placed_objects") or []),
        "candidate_solution_count": len(solution_pool),
        "ranked_solution_count": len(ranked_solutions),
        "visual_solver_diagnostics": deepcopy(diagnostics),
    }
    return {
        "status": "OK" if verify.get("hard_valid") else "PARTIAL",
        "solver_kind": _SOLVER_KIND,
        "selected_concept_id": object_solver._concept_id_from_relation_plan(
            relation_plan
        ),
        "absolute_layout": absolute_layout,
        "solutions": ranked_solutions,
        "hard_valid": bool(verify.get("hard_valid")),
        "geometry_valid": bool(verify.get("geometry_valid")),
        "acceptable_valid": bool(verify.get("gallery_eligible")),
        "complete": bool(verify.get("complete")),
        "gallery_eligible": bool(verify.get("gallery_eligible")),
        "coverage_ratio": float(verify.get("coverage_ratio") or 0.0),
        "offending_clusters": list(verify.get("offending_clusters") or []),
        "dropped_inventory_by_cluster": deepcopy(
            best_solution.get("dropped_inventory_by_cluster") or {}
        ),
        "cluster_transforms": [],
        "selected_variants": [],
        "verify_summary": deepcopy(dict(verify)),
        "visual_solver_diagnostics": diagnostics,
        "solver_debug": solver_debug,
        "notes": [
            "Solved with experimental visual-friendly object solver V2.",
            "Exact semantic anchor regions were enforced for hard-locked visual objects.",
            "Protected/requested objects were preserved while optional trial clusters could be dropped.",
        ],
    }


def _search_visual_anchor_solutions(
    *,
    anchor_order: Sequence[str],
    anchor_candidates_by_cluster: Mapping[str, Sequence[Mapping[str, Any]]],
    world: Mapping[str, Any],
    max_solutions: int,
    max_candidates_per_cluster: int,
) -> list[dict[str, Any]]:
    solutions: list[dict[str, Any]] = []

    def rec(
        index: int,
        chosen: list[dict[str, Any]],
        placed_rows: list[dict[str, Any]],
        dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]],
    ) -> None:
        if len(solutions) >= max(1, int(max_solutions)):
            return
        if index >= len(anchor_order):
            score = sum(float(row.get("candidate_score") or 0.0) for row in chosen)
            solutions.append(
                {
                    "anchor_solution": {str(row["cluster_id"]): row for row in chosen},
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
            return

        cluster_id = anchor_order[index]
        cluster_program = world["clusters_by_id"][cluster_id]
        if object_solver._cluster_is_solver_trial_optional(cluster_program):
            drop_records = object_solver._optional_trial_cluster_drop_records(
                cluster_program
            )
            dropped_inventory_by_cluster.setdefault(cluster_id, []).extend(drop_records)
            rec(index + 1, chosen, placed_rows, dropped_inventory_by_cluster)
            if drop_records:
                del dropped_inventory_by_cluster[cluster_id][-len(drop_records) :]

        for raw_candidate in list(anchor_candidates_by_cluster.get(cluster_id) or [])[
            : max(1, int(max_candidates_per_cluster))
        ]:
            candidate = dict(raw_candidate)
            materialized = object_solver._materialize_object_row(
                cluster_program,
                str(candidate.get("object_id") or ""),
                int(candidate.get("x") or 0),
                int(candidate.get("y") or 0),
                int(candidate.get("rot") or 0),
            )
            rect = materialized["rect"]
            if any(
                object_solver._rects_overlap(rect, row["rect"]) for row in placed_rows
            ):
                continue
            if not object_solver._object_rect_is_usable(
                rect,
                [row["rect"] for row in placed_rows],
                world,
                row=materialized,
                front_access_rows=tuple(placed_rows),
            ):
                continue
            chosen.append(candidate)
            placed_rows.append(materialized)
            rec(index + 1, chosen, placed_rows, dropped_inventory_by_cluster)
            placed_rows.pop()
            chosen.pop()

    rec(0, [], [], {})
    return solutions


def _place_visual_support_objects_for_solution(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    grid_mm: int,
    max_solutions: int,
    max_support_candidates_per_object: int,
) -> list[dict[str, Any]]:
    placed_objects: list[dict[str, Any]] = []
    placed_by_id: dict[tuple[str, str], dict[str, Any]] = {}
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
    for cluster_id, anchor_row in anchor_solution.items():
        cluster_program = world["clusters_by_id"][cluster_id]
        object_id = str(anchor_row.get("object_id") or "")
        row = object_solver._materialize_object_row(
            cluster_program,
            object_id,
            int(anchor_row.get("x") or 0),
            int(anchor_row.get("y") or 0),
            int(anchor_row.get("rot") or 0),
        )
        placed_objects.append(row)
        placed_by_id[(str(cluster_id), object_id)] = row

    tasks = _support_tasks(world, anchor_solution)
    seated_surface_supports = object_solver._surface_seat_supports_by_cluster(world)
    results: list[dict[str, Any]] = []
    fallback_counts: Counter[str] = Counter()
    rejected_protected_items: list[dict[str, str]] = []
    anchor_lock_violations: list[dict[str, Any]] = []
    hard_lock_rejected_count = 0
    unplaceable_locked_objects: list[str] = []

    def search(index: int, support_score: float) -> None:
        nonlocal hard_lock_rejected_count
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
                    "visual_solver_trace": {
                        "support_fallback_counts": dict(fallback_counts),
                        "rejected_protected_items": _dedupe_rejections(
                            rejected_protected_items
                        ),
                        "anchor_lock_violations": deepcopy(anchor_lock_violations),
                        "hard_lock_rejected_count": hard_lock_rejected_count,
                        "unplaceable_locked_objects": list(
                            _unique_strings(unplaceable_locked_objects)
                        ),
                    },
                }
            )
            return

        cluster_id, object_id, edge, base_id = tasks[index]
        cluster_program = world["clusters_by_id"][cluster_id]
        object_program = cluster_program["object_program"]
        anchor_id = object_solver._cluster_program_dominant_anchor(cluster_program)
        protected = {str(item) for item in object_program.get("protected_ids") or []}
        droppable = object_solver._droppable_object_ids(object_program)
        base_row = placed_by_id.get((cluster_id, base_id))
        if base_row is None and anchor_id is not None:
            base_row = placed_by_id.get((cluster_id, anchor_id))

        candidates = _strict_support_candidates(
            cluster_program=cluster_program,
            object_id=object_id,
            base_id=base_id,
            base_row=base_row,
            edge=edge,
            placed_objects=placed_objects,
            placed_by_id=placed_by_id,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            grid_mm=grid_mm,
            limit=max_support_candidates_per_object,
        )
        fallback_kind = "strict_edge"
        if not candidates and (object_id in protected or object_id not in droppable):
            candidates = _independent_support_candidates(
                cluster_program=cluster_program,
                object_id=object_id,
                base_row=base_row,
                placed_objects=placed_objects,
                placed_by_id=placed_by_id,
                world=world,
                room_model=room_model,
                relation_plan=relation_plan,
                grid_mm=grid_mm,
                limit=max_support_candidates_per_object,
            )
            fallback_kind = "same_cluster_independent_wall"

        if not candidates and object_id in protected:
            rejected_protected_items.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": object_id,
                    "reason": "no_visual_support_candidate",
                }
            )
        if (
            not candidates
            and _semantic_anchor_preference_is_hard(
                _semantic_anchor_preference(
                    relation_plan=relation_plan,
                    cluster_id=cluster_id,
                    object_id=object_id,
                )
            )
        ):
            violation = {
                "cluster_id": cluster_id,
                "object_id": object_id,
                "semantic_anchor_region_id": str(
                    _semantic_anchor_preference(
                        relation_plan=relation_plan,
                        cluster_id=cluster_id,
                        object_id=object_id,
                    ).get("semantic_anchor_region_id")
                    or ""
                ),
                "reason": "no_candidate_inside_semantic_anchor",
            }
            _extend_unique_violations(anchor_lock_violations, [violation])
            hard_lock_rejected_count += 1
            unplaceable_locked_objects.append(f"{cluster_id}/{object_id}")

        for candidate_score, row in candidates:
            placed_objects.append(row)
            placed_by_id[(cluster_id, object_id)] = row
            fallback_counts[fallback_kind] += 1
            search(index + 1, support_score + candidate_score)
            fallback_counts[fallback_kind] -= 1
            if fallback_counts[fallback_kind] <= 0:
                del fallback_counts[fallback_kind]
            placed_by_id.pop((cluster_id, object_id), None)
            placed_objects.pop()

        if object_id in droppable and object_id not in protected:
            if object_solver._dropping_last_surface_seat_support(
                cluster_id=cluster_id,
                object_id=object_id,
                base_id=base_id,
                placed_by_id=placed_by_id,
                remaining_tasks=tasks[index + 1 :],
                seated_surface_supports=seated_surface_supports,
            ):
                return
            dropped_inventory_by_cluster.setdefault(cluster_id, []).append(
                {
                    "object_id": object_id,
                    "reason": "visual_v2_optional_support_drop",
                }
            )
            fallback_counts["optional_support_drop"] += 1
            search(index + 1, support_score - 90.0)
            fallback_counts["optional_support_drop"] -= 1
            if fallback_counts["optional_support_drop"] <= 0:
                del fallback_counts["optional_support_drop"]
            dropped_inventory_by_cluster[cluster_id].pop()

    search(0, 0.0)
    return results


def _support_tasks(
    world: Mapping[str, Any],
    anchor_solution: Mapping[str, Any],
) -> list[tuple[str, str, Mapping[str, Any], str]]:
    tasks: list[tuple[str, str, Mapping[str, Any], str]] = []
    for cluster_id in world["anchor_cluster_order"]:
        cluster_program = world["clusters_by_id"][cluster_id]
        if (
            cluster_id not in anchor_solution
            and object_solver._cluster_is_solver_trial_optional(cluster_program)
        ):
            continue
        object_program = cluster_program["object_program"]
        anchor_id = object_solver._cluster_program_dominant_anchor(cluster_program)
        support_edges = {
            str(row.get("object_id")): row
            for row in object_program.get("support_edges") or []
            if isinstance(row, Mapping) and isinstance(row.get("object_id"), str)
        }
        for object_id in object_program.get("placement_order") or []:
            if not isinstance(object_id, str) or object_id == anchor_id:
                continue
            edge = support_edges.get(object_id, {})
            base_id = str(edge.get("relative_to") or anchor_id or "")
            tasks.append((cluster_id, object_id, edge, base_id))
    return tasks


def _strict_support_candidates(
    *,
    cluster_program: Mapping[str, Any],
    object_id: str,
    base_id: str,
    base_row: Mapping[str, Any] | None,
    edge: Mapping[str, Any],
    placed_objects: Sequence[Mapping[str, Any]],
    placed_by_id: Mapping[tuple[str, str], Mapping[str, Any]],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    grid_mm: int,
    limit: int,
) -> list[tuple[float, dict[str, Any]]]:
    if base_row is None:
        return []
    rows: list[tuple[float, dict[str, Any]]] = []
    for slot in object_solver._support_slot_candidates(
        cluster_program=cluster_program,
        object_id=object_id,
        base_row=base_row,
        edge=edge,
        grid_mm=grid_mm,
    ):
        row = object_solver._materialize_object_row(
            cluster_program,
            object_id,
            int(slot.get("x") or 0),
            int(slot.get("y") or 0),
            int(slot.get("rot") or 0),
            relative_to=base_id,
        )
        if not _visual_object_row_is_usable(
            row=row,
            placed_objects=placed_objects,
            placed_by_id=placed_by_id,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
        ):
            continue
        rows.append((float(slot.get("orientation_score") or 0.0), row))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows[: max(1, int(limit))]


def _independent_support_candidates(
    *,
    cluster_program: Mapping[str, Any],
    object_id: str,
    base_row: Mapping[str, Any] | None,
    placed_objects: Sequence[Mapping[str, Any]],
    placed_by_id: Mapping[tuple[str, str], Mapping[str, Any]],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    grid_mm: int,
    limit: int,
) -> list[tuple[float, dict[str, Any]]]:
    spec = _object_spec(cluster_program, object_id)
    allowed_rotations = [
        int(item) % 360
        for item in (spec.get("allowed_rotations") or [0, 90, 180, 270])
        if isinstance(item, int | float)
    ] or [0, 90, 180, 270]
    room_bbox = world["room_bbox"]
    step = max(int(grid_mm), 150)
    base_center = _row_center(base_row)
    rows: list[tuple[float, dict[str, Any]]] = []

    for rotation in allowed_rotations:
        probe = object_solver._materialize_object_row(
            cluster_program,
            object_id,
            0,
            0,
            rotation,
            relative_to=str(base_row.get("object_id") or "") if base_row else None,
        )
        width = int(probe.get("w") or 0)
        height = int(probe.get("h") or 0)
        if width <= 0 or height <= 0:
            continue
        for y in range(int(room_bbox[1]), int(room_bbox[3]) - height + 1, step):
            for x in range(int(room_bbox[0]), int(room_bbox[2]) - width + 1, step):
                row = object_solver._materialize_object_row(
                    cluster_program,
                    object_id,
                    x,
                    y,
                    rotation,
                    relative_to=(
                        str(base_row.get("object_id") or "") if base_row else None
                    ),
                )
                if not _visual_object_row_is_usable(
                    row=row,
                    placed_objects=placed_objects,
                    placed_by_id=placed_by_id,
                    world=world,
                    room_model=room_model,
                    relation_plan=relation_plan,
                ):
                    continue
                rows.append(
                    (
                        _independent_support_score(
                            row=row,
                            room_bbox=room_bbox,
                            base_center=base_center,
                        ),
                        row,
                    )
                )

    rows.sort(key=lambda item: item[0], reverse=True)
    return rows[: max(1, int(limit))]


def _visual_object_row_is_usable(
    *,
    row: Mapping[str, Any],
    placed_objects: Sequence[Mapping[str, Any]],
    placed_by_id: Mapping[tuple[str, str], Mapping[str, Any]],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> bool:
    rect = row.get("rect")
    if not isinstance(rect, tuple) or len(rect) != 4:
        return False
    if any(object_solver._rects_overlap(rect, item["rect"]) for item in placed_objects):
        return False
    if not object_solver._object_rect_is_usable(
        rect,
        [item["rect"] for item in placed_objects],
        world,
        row=row,
        front_access_rows=tuple(placed_by_id.values()),
    ):
        return False
    if _semantic_anchor_lock_violation(
        row=row,
        world=world,
        room_model=room_model,
        relation_plan=relation_plan,
    ):
        return False
    orientation_issues = object_solver._object_level_orientation_issues(
        placed_objects=[*placed_objects, row],
        world=world,
        relation_plan=relation_plan,
    )
    blocking = object_solver._object_level_blocking_orientation_issues(
        orientation_issues=orientation_issues,
        placed_objects=[*placed_objects, row],
    )
    return not any(
        str(issue.get("cluster_id") or "") == str(row.get("cluster_id") or "")
        and str(issue.get("object_id") or "") == str(row.get("object_id") or "")
        for issue in blocking
    )


def _relax_visual_paired_front_access_issues(
    *,
    verify: Mapping[str, Any],
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    placed_objects = _mapping_rows(solution.get("placed_objects"))
    verify = _enforce_visual_paired_support_orientation(
        verify=dict(verify),
        solution=solution,
        world=world,
        relation_plan=relation_plan,
        placed_objects=placed_objects,
    )
    blocking_issues = _mapping_rows(verify.get("blocking_functional_geometry_issues"))
    if not blocking_issues:
        return dict(verify)

    relaxed: list[dict[str, Any]] = []
    kept_blocking: list[dict[str, Any]] = []
    for issue in blocking_issues:
        if _visual_front_access_issue_is_allowed(
            issue=issue,
            placed_objects=placed_objects,
            world=world,
            relation_plan=relation_plan,
        ):
            relaxed.append(issue)
        else:
            kept_blocking.append(issue)

    if not relaxed:
        return dict(verify)

    relaxed_keys = {_functional_issue_key(item) for item in relaxed}
    functional_issues = [
        item
        for item in _mapping_rows(verify.get("functional_geometry_issues"))
        if _functional_issue_key(item) not in relaxed_keys
    ]
    out = dict(verify)
    out["functional_geometry_issues"] = functional_issues
    out["blocking_functional_geometry_issues"] = kept_blocking
    out["visual_relaxed_functional_geometry_issues"] = relaxed
    out["front_access_ok"] = not any(
        str(item.get("reason") or "") == "front_access_zone_blocked"
        for item in kept_blocking
    )
    out["view_corridor_ok"] = not any(
        str(item.get("reason") or "") == "view_corridor_blocked"
        for item in kept_blocking
    )
    out["blocking_issue_count"] = (
        len(_mapping_rows(out.get("blocking_protected_region_issues")))
        + len(_mapping_rows(out.get("blocking_forbidden_region_issues")))
        + len(_mapping_rows(out.get("blocking_orientation_issues")))
        + len(_mapping_rows(out.get("face_pair_issues")))
        + len(kept_blocking)
    )
    out["hard_valid"] = (
        bool(out.get("geometry_valid")) and int(out["blocking_issue_count"]) == 0
    )
    out["gallery_eligible"] = (
        bool(out.get("complete"))
        and bool(out.get("hard_valid"))
        and int(out.get("critical_issue_count") or 0) == 0
        and int(out.get("blocking_issue_count") or 0) == 0
        and bool(out.get("view_corridor_ok"))
        and bool(out.get("front_access_ok"))
    )
    out["offending_clusters"] = _offending_clusters_from_verify(out)
    out["quality_gate_reasons"] = object_solver._object_level_quality_gate_reasons(
        gallery_eligible=bool(out.get("gallery_eligible")),
        complete=bool(out.get("complete")),
        face_pair_issues=_mapping_rows(out.get("face_pair_issues")),
        blocking_protected_region_issues=_mapping_rows(
            out.get("blocking_protected_region_issues")
        ),
        blocking_forbidden_region_issues=_mapping_rows(
            out.get("blocking_forbidden_region_issues")
        ),
        blocking_orientation_issues=_mapping_rows(
            out.get("blocking_orientation_issues")
        ),
        blocking_functional_geometry_issues=kept_blocking,
    )
    _recompute_visual_layout_score(out, placed_object_count=len(placed_objects))
    return out


def _enforce_visual_paired_support_orientation(
    *,
    verify: dict[str, Any],
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    placed_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    del solution
    blocking_orientation_issues = _mapping_rows(
        verify.get("blocking_orientation_issues")
    )
    blocking_keys = {
        _functional_issue_key(item) for item in blocking_orientation_issues
    }
    added: list[dict[str, Any]] = []
    for issue in _mapping_rows(verify.get("orientation_issues")):
        if not _visual_orientation_issue_is_paired_support_blocking(
            issue=issue,
            world=world,
            relation_plan=relation_plan,
        ):
            continue
        key = _functional_issue_key(issue)
        if key in blocking_keys:
            continue
        blocking_keys.add(key)
        blocking_orientation_issues.append(issue)
        added.append(issue)

    if not added:
        return verify

    verify["blocking_orientation_issues"] = blocking_orientation_issues
    verify["visual_enforced_orientation_issues"] = added
    verify["blocking_issue_count"] = (
        len(_mapping_rows(verify.get("blocking_protected_region_issues")))
        + len(_mapping_rows(verify.get("blocking_forbidden_region_issues")))
        + len(blocking_orientation_issues)
        + len(_mapping_rows(verify.get("face_pair_issues")))
        + len(_mapping_rows(verify.get("blocking_functional_geometry_issues")))
    )
    verify["hard_valid"] = False
    verify["gallery_eligible"] = False
    verify["quality_gate_reasons"] = object_solver._object_level_quality_gate_reasons(
        gallery_eligible=False,
        complete=bool(verify.get("complete")),
        face_pair_issues=_mapping_rows(verify.get("face_pair_issues")),
        blocking_protected_region_issues=_mapping_rows(
            verify.get("blocking_protected_region_issues")
        ),
        blocking_forbidden_region_issues=_mapping_rows(
            verify.get("blocking_forbidden_region_issues")
        ),
        blocking_orientation_issues=blocking_orientation_issues,
        blocking_functional_geometry_issues=_mapping_rows(
            verify.get("blocking_functional_geometry_issues")
        ),
    )
    _recompute_visual_layout_score(verify, placed_object_count=len(placed_objects))
    return verify


def _visual_orientation_issue_is_paired_support_blocking(
    *,
    issue: Mapping[str, Any],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> bool:
    cluster_id = str(issue.get("cluster_id") or "").strip()
    object_id = str(issue.get("object_id") or "").strip()
    if not cluster_id or not object_id:
        return False
    edge = _support_edge_for(world=world, cluster_id=cluster_id, object_id=object_id)
    relative_to = str(edge.get("relative_to") or "").strip()
    if not relative_to:
        return False
    relation_hint_allows = _relation_hint_allows_target_front_access(
        relation_plan=relation_plan,
        cluster_id=cluster_id,
        object_id=object_id,
        target_object_id=relative_to,
    )
    support_role = str(edge.get("support_role") or "").strip().lower()
    band_intent = str(edge.get("band_intent") or "").strip().lower()
    if not (
        relation_hint_allows
        or support_role in {"secondary_seat", "paired_seat"}
        or band_intent == "flank_band"
    ):
        return False
    cluster_program = _cluster_program(world, cluster_id)
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    return object_solver._object_program_id_is_work_surface(
        object_program,
        relative_to,
    ) and object_solver._object_program_id_is_seat(object_program, object_id)


def _visual_front_access_issue_is_allowed(
    *,
    issue: Mapping[str, Any],
    placed_objects: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> bool:
    if str(issue.get("reason") or "") != "front_access_zone_blocked":
        return False
    cluster_id = str(issue.get("cluster_id") or "").strip()
    blocker_id = str(issue.get("object_id") or "").strip()
    blocked_cluster_id = str(issue.get("blocked_cluster_id") or "").strip()
    blocked_id = str(issue.get("blocked_object_id") or "").strip()
    if not cluster_id or cluster_id != blocked_cluster_id or not blocker_id:
        return False
    if not blocked_id or blocker_id == blocked_id:
        return False

    rows = {
        (str(row.get("cluster_id") or ""), str(row.get("object_id") or "")): row
        for row in placed_objects
    }
    blocker = rows.get((cluster_id, blocker_id))
    blocked = rows.get((cluster_id, blocked_id))
    if blocker is None or blocked is None:
        return False

    edge = _support_edge_for(
        world=world,
        cluster_id=cluster_id,
        object_id=blocked_id,
    )
    edge_targets_blocker = str(edge.get("relative_to") or "") == blocker_id
    relation_hint_allows = _relation_hint_allows_target_front_access(
        relation_plan=relation_plan,
        cluster_id=cluster_id,
        object_id=blocked_id,
        target_object_id=blocker_id,
    )
    if not edge_targets_blocker and not relation_hint_allows:
        return False

    cluster_program = _cluster_program(world, cluster_id)
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    support_role = str(edge.get("support_role") or "").strip().lower()
    band_intent = str(edge.get("band_intent") or "").strip().lower()
    if not (
        relation_hint_allows
        or support_role in {"secondary_seat", "paired_seat"}
        or band_intent == "flank_band"
    ):
        return False
    if not object_solver._object_program_id_is_work_surface(
        object_program,
        blocker_id,
    ):
        return False
    if not object_solver._object_program_id_is_seat(object_program, blocked_id):
        return False
    return _row_faces_target(blocked, blocker)


def _support_edge_for(
    *,
    world: Mapping[str, Any],
    cluster_id: str,
    object_id: str,
) -> Mapping[str, Any]:
    cluster_program = _cluster_program(world, cluster_id)
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    for edge in object_program.get("support_edges") or []:
        if not isinstance(edge, Mapping):
            continue
        if str(edge.get("object_id") or "") == object_id:
            return edge
    return {}


def _cluster_program(world: Mapping[str, Any], cluster_id: str) -> Mapping[str, Any]:
    clusters = (
        world.get("clusters_by_id")
        if isinstance(world.get("clusters_by_id"), Mapping)
        else {}
    )
    cluster_program = clusters.get(cluster_id)
    return cluster_program if isinstance(cluster_program, Mapping) else {}


def _relation_hint_allows_target_front_access(
    *,
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
    object_id: str,
    target_object_id: str,
) -> bool:
    if not isinstance(relation_plan, Mapping):
        return False
    for hint in relation_plan.get("functional_pair_hints") or []:
        if not isinstance(hint, Mapping):
            continue
        if str(hint.get("cluster_id") or "") != cluster_id:
            continue
        if str(hint.get("object_id") or "") != object_id:
            continue
        if str(hint.get("target_object_id") or "") != target_object_id:
            continue
        if bool(hint.get("allow_target_in_front_access")):
            return True
    preferences = relation_plan.get("object_region_preferences_by_cluster")
    if not isinstance(preferences, Mapping):
        return False
    cluster_preferences = preferences.get(cluster_id)
    if not isinstance(cluster_preferences, Mapping):
        return False
    row = cluster_preferences.get(object_id)
    if not isinstance(row, Mapping):
        return False
    return str(row.get("target_object_id") or "") == target_object_id and str(
        row.get("placement_role") or ""
    ) in {"paired_seat", "functional_support"}


def _row_faces_target(
    row: Mapping[str, Any],
    target: Mapping[str, Any],
) -> bool:
    rect = row.get("rect")
    target_rect = target.get("rect")
    if not (
        isinstance(rect, tuple)
        and len(rect) == 4
        and isinstance(target_rect, tuple)
        and len(target_rect) == 4
    ):
        return False
    desired = _vector_between_rect_centers(rect, target_rect)
    front = object_solver._front_tuple(row.get("front_world"))
    if desired is None or front is None:
        return False
    return _dot_normalized(front, desired) >= 0.55


def _recompute_visual_layout_score(
    verify: dict[str, Any],
    *,
    placed_object_count: int,
) -> None:
    orientation_issues = _mapping_rows(verify.get("orientation_issues"))
    blocking_orientation_issues = _mapping_rows(
        verify.get("blocking_orientation_issues")
    )
    protected_region_issues = _mapping_rows(verify.get("protected_region_issues"))
    blocking_protected_region_issues = _mapping_rows(
        verify.get("blocking_protected_region_issues")
    )
    forbidden_region_issues = _mapping_rows(verify.get("forbidden_region_issues"))
    blocking_forbidden_region_issues = _mapping_rows(
        verify.get("blocking_forbidden_region_issues")
    )
    functional_geometry_issues = _mapping_rows(verify.get("functional_geometry_issues"))
    blocking_functional_geometry_issues = _mapping_rows(
        verify.get("blocking_functional_geometry_issues")
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
        + (1 if int(verify.get("geometry_repair_penalty") or 0) > 0 else 0)
    )
    soft_constraint_score = max(0, placed_object_count * 2 - soft_issue_count)
    blocking_issue_count = int(verify.get("blocking_issue_count") or 0)
    pair_ok = bool(verify.get("primary_pair_ok"))
    geometry_valid = bool(verify.get("geometry_valid"))
    hard_valid = bool(verify.get("hard_valid"))
    protected_penalty = float(verify.get("protected_region_penalty") or 0.0)
    forbidden_penalty = float(verify.get("forbidden_region_penalty") or 0.0)
    repair_penalty = int(verify.get("geometry_repair_penalty") or 0)
    verify["soft_issue_count"] = soft_issue_count
    verify["soft_constraint_score"] = soft_constraint_score
    verify["layout_score"] = int(
        1000
        + placed_object_count * 40
        + (200 if pair_ok else -120)
        + (600 if geometry_valid else -2000)
        + (200 if hard_valid else -500)
        + soft_constraint_score * 30
        - protected_penalty
        - forbidden_penalty
        - len(orientation_issues) * 30
        - len(functional_geometry_issues) * 90
        - blocking_issue_count * 1200
        - repair_penalty
    )


def _offending_clusters_from_verify(verify: Mapping[str, Any]) -> list[str]:
    out: set[str] = set()
    for key in (
        "blocking_protected_region_issues",
        "blocking_forbidden_region_issues",
        "blocking_orientation_issues",
        "blocking_functional_geometry_issues",
    ):
        for issue in _mapping_rows(verify.get(key)):
            cluster_id = str(issue.get("cluster_id") or "")
            if cluster_id:
                out.add(cluster_id)
    for issue in _mapping_rows(verify.get("face_pair_issues")):
        for key in ("a", "b"):
            cluster_id = str(issue.get(key) or "")
            if cluster_id:
                out.add(cluster_id)
    return sorted(out)


def _functional_issue_key(issue: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(issue.get("cluster_id") or ""),
        str(issue.get("object_id") or ""),
        str(issue.get("blocked_cluster_id") or ""),
        str(issue.get("blocked_object_id") or ""),
        str(issue.get("reason") or ""),
    )


def _mapping_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _strict_output_semantic_anchor_lock_violations(
    *,
    output: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    grid_mm: int,
) -> list[dict[str, Any]]:
    if not _semantic_anchor_lock_summary(relation_plan):
        return []
    layout = output.get("absolute_layout")
    if not isinstance(layout, Mapping):
        return []
    rows = _mapping_rows(layout.get("object_placements")) or _mapping_rows(
        layout.get("objects")
    )
    if not rows:
        return []
    return _semantic_anchor_lock_violations(
        placed_objects=rows,
        world=_minimal_lock_world(room_model=room_model, grid_mm=grid_mm),
        room_model=room_model,
        relation_plan=relation_plan,
    )


def _minimal_lock_world(
    *,
    room_model: Mapping[str, Any],
    grid_mm: int,
) -> dict[str, Any]:
    return {
        "grid_mm": max(25, int(grid_mm)),
        "region_index": object_solver._index_room_regions(room_model),
        "room_bbox": object_solver._object_solver_room_bbox(room_model),
    }


def _semantic_anchor_lock_summary(
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cluster_id, rows in _semantic_anchor_preferences(relation_plan).items():
        for object_id, preference in rows.items():
            if not _semantic_anchor_preference_is_hard(preference):
                continue
            out.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": object_id,
                    "semantic_anchor_region_id": str(
                        preference.get("semantic_anchor_region_id") or ""
                    ),
                    "anchor_lock_strength": str(
                        preference.get("anchor_lock_strength") or ""
                    ),
                    "wall_side": str(preference.get("wall_side") or ""),
                    "placement_role": str(preference.get("placement_role") or ""),
                }
            )
    return out


def _semantic_anchor_lock_violations(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in placed_objects:
        violation = _semantic_anchor_lock_violation(
            row=row,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
        )
        if violation:
            violations.append(violation)
    return _dedupe_lock_violations(violations)


def _semantic_anchor_lock_violation(
    *,
    row: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    cluster_id = str(row.get("cluster_id") or "")
    object_id = str(row.get("object_id") or "")
    preference = _semantic_anchor_preference(
        relation_plan=relation_plan,
        cluster_id=cluster_id,
        object_id=object_id,
    )
    if not _semantic_anchor_preference_is_hard(preference):
        return None
    rect = _rect_tuple(row.get("rect"))
    if rect is None:
        return {
            "cluster_id": cluster_id,
            "object_id": object_id,
            "semantic_anchor_region_id": str(
                preference.get("semantic_anchor_region_id") or ""
            ),
            "reason": "missing_rect_for_semantic_anchor_lock",
        }
    bboxes = _semantic_anchor_region_bboxes(
        preference=preference,
        world=world,
        room_model=room_model,
        relation_plan=relation_plan,
    )
    if not bboxes:
        return {
            "cluster_id": cluster_id,
            "object_id": object_id,
            "semantic_anchor_region_id": str(
                preference.get("semantic_anchor_region_id") or ""
            ),
            "reason": "unresolved_semantic_anchor_region",
        }
    if any(
        _rect_satisfies_semantic_anchor_lock(
            rect=rect,
            bbox=bbox,
            preference=preference,
            world=world,
        )
        for bbox in bboxes
    ):
        return None
    return {
        "cluster_id": cluster_id,
        "object_id": object_id,
        "semantic_anchor_region_id": str(
            preference.get("semantic_anchor_region_id") or ""
        ),
        "reason": "outside_semantic_anchor_region",
    }


def _semantic_anchor_region_bboxes(
    *,
    preference: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> list[tuple[int, int, int, int]]:
    region_ids = _unique_strings(
        [
            str(preference.get("semantic_anchor_region_id") or ""),
            str(preference.get("source_region_id") or ""),
        ]
    )
    bboxes: list[tuple[int, int, int, int]] = []
    for region_id in region_ids:
        bboxes.extend(
            object_solver._region_bboxes_from_ref(
                region_id,
                world["region_index"],
                relation_plan,
                room_model,
            )
        )
    return object_solver._dedupe_region_bboxes(bboxes)


def _rect_satisfies_semantic_anchor_lock(
    *,
    rect: tuple[int, int, int, int],
    bbox: tuple[int, int, int, int],
    preference: Mapping[str, Any],
    world: Mapping[str, Any],
) -> bool:
    wall_side = str(preference.get("wall_side") or "")
    if wall_side in {"left_wall", "right_wall", "top_wall", "bottom_wall"}:
        return _rect_is_flush_to_anchor_wall(
            rect=rect,
            bbox=bbox,
            wall_side=wall_side,
            tolerance=max(150, int(world.get("grid_mm") or 50) * 2),
        )
    overlap_ratio = object_solver._rect_overlap_ratio(rect, bbox)
    return overlap_ratio >= 0.45 or _rect_center_inside(rect, bbox)


def _rect_is_flush_to_anchor_wall(
    *,
    rect: tuple[int, int, int, int],
    bbox: tuple[int, int, int, int],
    wall_side: str,
    tolerance: int,
) -> bool:
    if wall_side == "left_wall":
        return abs(rect[0] - bbox[0]) <= tolerance and _range_overlap_ratio(
            rect[1],
            rect[3],
            bbox[1],
            bbox[3],
        ) >= 0.45
    if wall_side == "right_wall":
        return abs(rect[2] - bbox[2]) <= tolerance and _range_overlap_ratio(
            rect[1],
            rect[3],
            bbox[1],
            bbox[3],
        ) >= 0.45
    if wall_side == "top_wall":
        return abs(rect[1] - bbox[1]) <= tolerance and _range_overlap_ratio(
            rect[0],
            rect[2],
            bbox[0],
            bbox[2],
        ) >= 0.45
    if wall_side == "bottom_wall":
        return abs(rect[3] - bbox[3]) <= tolerance and _range_overlap_ratio(
            rect[0],
            rect[2],
            bbox[0],
            bbox[2],
        ) >= 0.45
    return False


def _range_overlap_ratio(
    left_min: int,
    left_max: int,
    right_min: int,
    right_max: int,
) -> float:
    overlap = max(0, min(left_max, right_max) - max(left_min, right_min))
    denominator = max(1, min(left_max - left_min, right_max - right_min))
    return float(overlap) / float(denominator)


def _rect_center_inside(
    rect: tuple[int, int, int, int],
    bbox: tuple[int, int, int, int],
) -> bool:
    center_x = (rect[0] + rect[2]) / 2.0
    center_y = (rect[1] + rect[3]) / 2.0
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _semantic_anchor_preferences(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    if not isinstance(relation_plan, Mapping):
        return {}
    raw_preferences = relation_plan.get("semantic_anchor_preferences_by_cluster")
    if not isinstance(raw_preferences, Mapping):
        return {}
    out: dict[str, dict[str, Mapping[str, Any]]] = {}
    for cluster_id, rows in raw_preferences.items():
        if not isinstance(rows, Mapping):
            continue
        out[str(cluster_id)] = {
            str(object_id): row
            for object_id, row in rows.items()
            if isinstance(row, Mapping)
        }
    return out


def _semantic_anchor_preference(
    *,
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
    object_id: str,
) -> Mapping[str, Any]:
    return _semantic_anchor_preferences(relation_plan).get(cluster_id, {}).get(
        object_id, {}
    )


def _semantic_anchor_preference_is_hard(
    preference: Mapping[str, Any] | None,
) -> bool:
    return (
        isinstance(preference, Mapping)
        and bool(str(preference.get("semantic_anchor_region_id") or "").strip())
        and str(preference.get("anchor_lock_strength") or "") == "hard"
        and bool(preference.get("lockable", True))
    )


def _locked_anchor_objects_with_no_candidates(
    *,
    anchor_order: Sequence[str],
    anchor_candidates_by_cluster: Mapping[str, Sequence[Mapping[str, Any]]],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> list[str]:
    out: list[str] = []
    for cluster_id in anchor_order:
        cluster_program = world["clusters_by_id"][cluster_id]
        anchor_id = object_solver._cluster_program_dominant_anchor(cluster_program)
        if not anchor_id:
            continue
        if anchor_candidates_by_cluster.get(cluster_id):
            continue
        if _semantic_anchor_preference_is_hard(
            _semantic_anchor_preference(
                relation_plan=relation_plan,
                cluster_id=cluster_id,
                object_id=anchor_id,
            )
        ):
            out.append(f"{cluster_id}/{anchor_id}")
    return _unique_strings(out)


def _locked_object_labels(violations: Sequence[Mapping[str, Any]]) -> list[str]:
    return _unique_strings(
        [
            f"{row.get('cluster_id')}/{row.get('object_id')}"
            for row in violations
            if row.get("cluster_id") and row.get("object_id")
        ]
    )


def _extend_unique_violations(
    target: list[Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    seen = {_lock_violation_key(row) for row in target if isinstance(row, Mapping)}
    for row in rows:
        key = _lock_violation_key(row)
        if key in seen:
            continue
        seen.add(key)
        target.append(dict(row))


def _dedupe_lock_violations(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    _extend_unique_violations(out, rows)
    return out


def _lock_violation_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("cluster_id") or ""),
        str(row.get("object_id") or ""),
        str(row.get("semantic_anchor_region_id") or ""),
        str(row.get("reason") or ""),
    )


def _extend_unique_strings(target: list[Any], rows: Sequence[Any]) -> None:
    for row in _unique_strings(rows):
        if row not in target:
            target.append(row)


def _build_visual_solution_payload(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    solution_index: int,
) -> dict[str, Any]:
    payload = object_solver._build_object_level_solution_payload(
        solution=solution,
        world=world,
        room_model=room_model,
        relation_plan=relation_plan,
        solution_index=solution_index,
    )
    payload["solver_kind"] = _SOLVER_KIND
    payload["visual_solver_trace"] = deepcopy(solution.get("visual_solver_trace") or {})
    return payload


def _rank_anchor_candidates_for_visual_hints(
    candidates: Sequence[Mapping[str, Any]],
    *,
    cluster_id: str,
    relation_plan: Mapping[str, Any] | None,
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    preferences = _object_region_preferences(relation_plan).get(cluster_id, {})
    semantic_preferences = _semantic_anchor_preferences(relation_plan).get(
        cluster_id, {}
    )
    if not preferences and not semantic_preferences:
        return [dict(candidate) for candidate in candidates]

    out: list[dict[str, Any]] = []
    for candidate in candidates:
        next_candidate = dict(candidate)
        object_id = str(next_candidate.get("object_id") or "")
        preference = preferences.get(object_id)
        semantic_preference = semantic_preferences.get(object_id)
        if _semantic_anchor_preference_is_hard(semantic_preference):
            violation = _semantic_anchor_lock_violation(
                row=next_candidate,
                world=world,
                room_model=room_model,
                relation_plan=relation_plan,
            )
            if violation:
                continue
            next_candidate["semantic_anchor_region_id"] = str(
                semantic_preference.get("semantic_anchor_region_id") or ""
            )
        if isinstance(preference, Mapping):
            hint_score = _visual_anchor_hint_score(
                candidate=next_candidate,
                preference=preference,
                world=world,
            )
            next_candidate["visual_hint_score"] = round(hint_score, 3)
            next_candidate["candidate_score"] = (
                float(next_candidate.get("candidate_score") or 0.0) + hint_score
            )
        out.append(next_candidate)
    out.sort(
        key=lambda item: (
            float(item.get("candidate_score") or 0.0),
            float(item.get("visual_hint_score") or 0.0),
        ),
        reverse=True,
    )
    return out


def _object_region_preferences(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    if not isinstance(relation_plan, Mapping):
        return {}
    raw_preferences = relation_plan.get("object_region_preferences_by_cluster")
    if isinstance(raw_preferences, Mapping):
        out: dict[str, dict[str, Mapping[str, Any]]] = {}
        for cluster_id, rows in raw_preferences.items():
            if not isinstance(rows, Mapping):
                continue
            out[str(cluster_id)] = {
                str(object_id): row
                for object_id, row in rows.items()
                if isinstance(row, Mapping)
            }
        return out

    out: dict[str, dict[str, Mapping[str, Any]]] = {}
    rows = relation_plan.get("object_region_plan")
    if isinstance(rows, Sequence) and not isinstance(rows, str):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            cluster_id = str(row.get("cluster_id") or "")
            object_id = str(row.get("object_id") or "")
            if cluster_id and object_id:
                out.setdefault(cluster_id, {})[object_id] = row
    return out


def _visual_anchor_hint_score(
    *,
    candidate: Mapping[str, Any],
    preference: Mapping[str, Any],
    world: Mapping[str, Any],
) -> float:
    rect = _rect_tuple(candidate.get("rect"))
    if rect is None:
        return 0.0
    room_bbox = world["room_bbox"]
    score = 180.0
    wall_side = str(preference.get("wall_side") or "")
    candidate_wall_side = _candidate_wall_side(rect, room_bbox)
    if wall_side and wall_side == candidate_wall_side:
        score += 380.0
    if wall_side == "window_side" and _region_or_wall_suggests_window(preference):
        if candidate_wall_side in {"top_wall", "bottom_wall"}:
            score += 260.0
    corner_hint = str(preference.get("corner_hint") or "none")
    if corner_hint != "none" and corner_hint == _candidate_corner_hint(rect, room_bbox):
        score += 260.0
    region_id = str(preference.get("region_id") or "")
    if "daylight" in region_id and candidate_wall_side in {"top_wall", "bottom_wall"}:
        score += 160.0
    if "private" in region_id and _candidate_corner_hint(rect, room_bbox).startswith(
        "top_"
    ):
        score += 120.0
    if "storage" in region_id and candidate_wall_side in {
        "left_wall",
        "right_wall",
        "top_wall",
        "bottom_wall",
    }:
        score += 120.0
    return score


def _candidate_wall_side(
    rect: tuple[int, int, int, int],
    room_bbox: Sequence[int],
) -> str:
    distances = {
        "left_wall": abs(rect[0] - int(room_bbox[0])),
        "right_wall": abs(int(room_bbox[2]) - rect[2]),
        "top_wall": abs(rect[1] - int(room_bbox[1])),
        "bottom_wall": abs(int(room_bbox[3]) - rect[3]),
    }
    return min(distances, key=distances.get)


def _candidate_corner_hint(
    rect: tuple[int, int, int, int],
    room_bbox: Sequence[int],
) -> str:
    center_x = (rect[0] + rect[2]) / 2.0
    center_y = (rect[1] + rect[3]) / 2.0
    mid_x = (float(room_bbox[0]) + float(room_bbox[2])) / 2.0
    mid_y = (float(room_bbox[1]) + float(room_bbox[3])) / 2.0
    vertical = "top" if center_y <= mid_y else "bottom"
    horizontal = "left" if center_x <= mid_x else "right"
    return f"{vertical}_{horizontal}"


def _region_or_wall_suggests_window(preference: Mapping[str, Any]) -> bool:
    return "window" in " ".join(
        str(preference.get(key) or "") for key in ("region_id", "wall_side", "reason")
    )


def _visual_solution_score(
    solution: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    verify = (
        solution.get("verify") if isinstance(solution.get("verify"), Mapping) else {}
    )
    score = object_solver._object_solution_score(solution, relation_plan)
    if verify.get("geometry_valid"):
        score += 2500.0
    if verify.get("hard_valid"):
        score += 3000.0
    if verify.get("gallery_eligible"):
        score += 5000.0
    score += 1200.0 * float(verify.get("coverage_ratio") or 0.0)
    score -= 350.0 * _dropped_inventory_count(solution)
    return score


def _independent_support_score(
    *,
    row: Mapping[str, Any],
    room_bbox: Sequence[int],
    base_center: tuple[float, float] | None,
) -> float:
    rect = row["rect"]
    center_x = (rect[0] + rect[2]) / 2.0
    center_y = (rect[1] + rect[3]) / 2.0
    distance_penalty = (
        math.hypot(center_x - base_center[0], center_y - base_center[1]) * 0.2
        if base_center is not None
        else 0.0
    )
    wall_distance = min(
        center_x - float(room_bbox[0]),
        float(room_bbox[2]) - center_x,
        center_y - float(room_bbox[1]),
        float(room_bbox[3]) - center_y,
    )
    return -distance_penalty - wall_distance * 0.3


def _strict_output_is_publishable(output: Mapping[str, Any]) -> bool:
    return (
        str(output.get("status") or "").upper() == "OK"
        and bool(output.get("geometry_valid"))
        and bool(output.get("hard_valid"))
        and bool(output.get("gallery_eligible"))
        and bool(output.get("solutions"))
    )


def _unsat_output(
    *,
    relation_plan: Mapping[str, Any] | None,
    grid_mm: int,
    diagnostics: Mapping[str, Any],
    notes: Sequence[str],
) -> dict[str, Any]:
    next_diagnostics = dict(diagnostics)
    if not next_diagnostics.get("unplaceable_locked_objects"):
        lock_rows = _mapping_rows(next_diagnostics.get("anchor_lock_violations"))
        lock_rows.extend(
            _mapping_rows(next_diagnostics.get("strict_anchor_lock_violations"))
        )
        if lock_rows:
            next_diagnostics["unplaceable_locked_objects"] = _locked_object_labels(
                lock_rows
            )
    return {
        "status": "UNSAT",
        "solver_kind": _SOLVER_KIND,
        "selected_concept_id": object_solver._concept_id_from_relation_plan(
            relation_plan
        ),
        "solutions": [],
        "best_solution_id": None,
        "grid_mm": grid_mm,
        "cluster_transforms": [],
        "selected_variants": [],
        "global_quality": {},
        "diagnostics": {},
        "degradation_applied": [],
        "missing": [],
        "conflicts": list(notes),
        "notes": list(notes),
        "visual_solver_diagnostics": next_diagnostics,
        "solver_debug": {"visual_solver_diagnostics": next_diagnostics},
    }


def _object_spec(
    cluster_program: Mapping[str, Any], object_id: str
) -> Mapping[str, Any]:
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
    spec = specs.get(object_id)
    return spec if isinstance(spec, Mapping) else {}


def _row_center(row: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if row is None:
        return None
    center = row.get("center") if isinstance(row.get("center"), Mapping) else {}
    try:
        return (float(center.get("x")), float(center.get("y")))
    except (TypeError, ValueError):
        return None


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


def _vector_between_rect_centers(
    source: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    source_x = (source[0] + source[2]) / 2.0
    source_y = (source[1] + source[3]) / 2.0
    target_x = (target[0] + target[2]) / 2.0
    target_y = (target[1] + target[3]) / 2.0
    dx = target_x - source_x
    dy = target_y - source_y
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return (dx / norm, dy / norm)


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


def _dropped_inventory_count(solution: Mapping[str, Any]) -> int:
    dropped = solution.get("dropped_inventory_by_cluster")
    if not isinstance(dropped, Mapping):
        return 0
    return sum(
        len(rows)
        for rows in dropped.values()
        if isinstance(rows, Sequence) and not isinstance(rows, str)
    )


def _top_failure_reasons(
    solutions: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for solution in solutions:
        verify = (
            solution.get("verify")
            if isinstance(solution.get("verify"), Mapping)
            else {}
        )
        for reason in verify.get("quality_gate_reasons") or []:
            clean = str(reason or "").strip()
            if clean:
                counts[clean] += 1
    return dict(counts.most_common())


def _dedupe_rejections(value: Sequence[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row = {
            "cluster_id": str(item.get("cluster_id") or ""),
            "object_id": str(item.get("object_id") or ""),
            "reason": str(item.get("reason") or ""),
        }
        key = (row["cluster_id"], row["object_id"], row["reason"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _unique_strings(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        clean = value.strip()
        if clean and clean not in out:
            out.append(clean)
    return out


def _string_sequence(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return _unique_strings(value)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default
