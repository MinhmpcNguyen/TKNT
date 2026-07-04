from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.seed_relation_planner import SeedRelationPlanner
from agent.room_interpreter import RoomInterpreter
from agent.seed_concept_generator import SeedConceptGenerator, solver_plan_from_concept
from agent.solver.solver import MacroClusterSolver
from agent.stylist import Stylist, stylist_model_name_for_variant
from agent.tier_count_director import TierCountDirector
from agent.request_contract import (
    missing_functional_contract_types,
    request_contract_from_payload,
)
from object_program.builder import build_solver_object_program
from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM
from layout.orientation_metadata import enrich_styled_output_orientations
from pipeline.ablation_modes import (
    AblationMode,
    ablation_metadata,
    apply_tier_output_for_mode,
    cluster_output_for_mode,
    resolve_ablation_mode,
    room_output_for_mode,
    skip_accessory_refill,
    style_policy_for_mode,
    target_final_count_for_mode,
    use_concept_repair,
    use_gallery_selector,
    use_material_macro_repair,
    use_semantic_layout_planner,
    use_stylist_llm,
    uses_neutral_style_policy,
)
from pipeline.dropped_inventory import (
    controlled_accessory_refill,
    dropped_inventory_payload,
)
from pipeline.layout_variants import (
    build_final_gallery_selection_summary,
    core_anchor_layout_signature,
    material_macro_layout_signature,
    select_distinct_final_gallery_candidates,
)
from pipeline.manual_placements import (
    append_manual_placements_guidance,
    apply_manual_placements_to_tier_output,
    augment_seed_relation_plan_with_manual_placements,
    merge_manual_placements_into_absolute_layout,
    merge_manual_placements_into_styled_output,
    prepare_input_payload_for_manual_placements,
)
from pipeline.orchestrator_artifacts import (
    CasePaths,
    _mark_downstream_module_outputs_stale,
    _now_utc_iso,
    _update_status,
    _write_json,
    _write_module_io_manifest,
    _write_module_json,
)
from pipeline.request_non_functional_refill import (
    ensure_requested_non_functional_contract_objects,
)
from stylist.style_policy import decor_refill_policy, extract_style_policy

logger = logging.getLogger(__name__)
_CONCEPT_REPAIR_MAX_ATTEMPTS = 5
_CONCEPT_REPAIR_MAX_DEPTH = 2
_FRONT_ACCESS_RELAXATION_REASON = "front_access_pair_conflict_after_anchor_pair_spread"
_CONCEPT_REPAIR_BLOCKED_REGION_TOKENS = (
    "door",
    "entry",
    "corridor",
    "center_openness",
    "keep_open",
    "floating_center",
)
_CONCEPT_REPAIR_FUNCTIONAL_REGION_PRIORITY = (
    "edge_loading_zone",
    "bottom_wall_zone",
    "window_side_zone",
    "left_wall_zone",
    "right_wall_zone",
    "top_wall_zone",
    "quiet_private_deep_zone",
)
_PRIMARY_CORRIDOR_RELAXATION_ZONE_TYPES = {"primary_circulation_corridor"}
_HARD_REQUEST_CONTRACT_INTENTS = {"must_keep", "must_try"}
_MATERIAL_MACRO_REPAIR_REGION_PRIORITY = (
    "left_wall_zone",
    "right_wall_zone",
    "top_wall_zone",
    "bottom_wall_zone",
    "window_side_zone",
    "quiet_private_deep_zone",
    "primary_focal_wall_zone",
)
_MATERIAL_MACRO_COMPLEMENT_BY_REGION = {
    "left_wall_zone": "right_wall_zone",
    "right_wall_zone": "left_wall_zone",
    "top_wall_zone": "bottom_wall_zone",
    "bottom_wall_zone": "top_wall_zone",
    "window_side_zone": "bottom_wall_zone",
    "quiet_private_deep_zone": "right_wall_zone",
    "primary_focal_wall_zone": "bottom_wall_zone",
}
_CORE_ANCHOR_DIVERSITY_TARGET_CLUSTERS = (
    "storage_core",
    "media_optional",
    "work_study",
)


@dataclass(frozen=True)
class _ConceptRepairAttempt:
    repair_id: str
    concept: dict[str, Any]
    relation_plan: dict[str, Any]
    merged_output: dict[str, Any]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _ObjectLevelSolveResult:
    bundles: list[dict[str, Any]]
    failure_diagnostics: list[dict[str, Any]]


def _build_runtime_solver() -> MacroClusterSolver:
    return MacroClusterSolver()


def _build_primary_guidance_text(
    *,
    room_output: dict[str, Any],
    input_payload: dict[str, Any],
    description: str | None,
    special_notes: str | None,
) -> str:
    notes = room_output.get("notes")
    if isinstance(notes, list):
        cleaned_notes = [
            note.strip() for note in notes if isinstance(note, str) and note.strip()
        ]
        if cleaned_notes:
            return "\n".join(f"- {note}" for note in cleaned_notes)

    legacy_parts: list[str] = []
    for value in (description, special_notes):
        if isinstance(value, str) and value.strip():
            legacy_parts.append(value.strip())

    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        for key in (
            "description",
            "special_description",
            "special_notes",
            "notes",
            "feng_shui",
            "style",
        ):
            value = user_input.get(key)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if text and text not in legacy_parts:
                legacy_parts.append(text)

    return "\n".join(legacy_parts)


def _validate_semantic_cluster_unique_members(
    cluster_output: dict[str, Any],
) -> dict[str, Any]:
    clusters = cluster_output.get("clusters")
    if not isinstance(clusters, list):
        return cluster_output

    output = deepcopy(cluster_output)
    mutable_clusters = [
        cluster for cluster in output.get("clusters", []) if isinstance(cluster, dict)
    ]
    owner_by_member: dict[str, tuple[int, str, int]] = {}
    removals_by_cluster_index: dict[int, set[str]] = {}
    notes: list[str] = []

    for index, cluster in enumerate(mutable_clusters):
        cluster_id = str(cluster.get("cluster_id") or f"cluster_{index}").strip()
        cluster_score = _semantic_cluster_member_owner_score(cluster)
        for member in _semantic_cluster_member_ids(cluster):
            owner = owner_by_member.get(member)
            if owner is None:
                owner_by_member[member] = (index, cluster_id, cluster_score)
                continue
            owner_index, owner_cluster_id, owner_score = owner
            if cluster_score > owner_score:
                removals_by_cluster_index.setdefault(owner_index, set()).add(member)
                owner_by_member[member] = (index, cluster_id, cluster_score)
                notes.append(
                    f"Removed duplicate member {member} from {owner_cluster_id}; kept in {cluster_id}."
                )
                continue
            removals_by_cluster_index.setdefault(index, set()).add(member)
            notes.append(
                f"Removed duplicate member {member} from {cluster_id}; already owned by {owner_cluster_id}."
            )

    if not removals_by_cluster_index:
        return cluster_output

    surviving_clusters: list[dict[str, Any]] = []
    surviving_cluster_ids: set[str] = set()
    dropped_cluster_ids: set[str] = set()
    for index, cluster in enumerate(mutable_clusters):
        removed_members = removals_by_cluster_index.get(index, set())
        if removed_members:
            cluster = _remove_semantic_cluster_members(cluster, removed_members)
        members = _semantic_cluster_member_ids(cluster)
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not members:
            if cluster_id:
                dropped_cluster_ids.add(cluster_id)
                notes.append(f"Dropped empty semantic support cluster {cluster_id}.")
            continue
        surviving_clusters.append(cluster)
        if cluster_id:
            surviving_cluster_ids.add(cluster_id)

    output["clusters"] = surviving_clusters
    _filter_semantic_layout_program_after_member_dedupe(
        output,
        removals_by_cluster_index=removals_by_cluster_index,
        dropped_cluster_ids=dropped_cluster_ids,
        surviving_cluster_ids=surviving_cluster_ids,
        original_clusters=mutable_clusters,
    )
    output["notes"] = _collect_unique_notes(output.get("notes"), notes)
    return output


def _semantic_cluster_member_owner_score(cluster: dict[str, Any]) -> int:
    cluster_id = str(cluster.get("cluster_id") or "").strip().lower()
    tag = str(cluster.get("tag") or "").strip().lower()
    priority = str(cluster.get("priority") or "").strip().lower()
    layout_role = str(cluster.get("layout_role") or "").strip().lower()
    semantic_role = str(cluster.get("semantic_role") or "").strip().lower()
    rules = cluster.get("cluster_rules")
    if isinstance(rules, dict):
        priority = priority or str(rules.get("priority") or "").strip().lower()
        layout_role = layout_role or str(rules.get("layout_role") or "").strip().lower()
    text = " ".join([cluster_id, tag, priority, layout_role, semantic_role])
    score = 0
    if priority == "core":
        score += 80
    if layout_role == "primary":
        score += 70
    if any(token in text for token in ("main", "core", "primary", "anchor")):
        score += 30
    if any(token in text for token in ("support", "lighting", "decor", "optional")):
        score -= 20
    anchors = cluster.get("anchors")
    if isinstance(anchors, list) and anchors:
        score += 10
    return score


def _semantic_cluster_member_ids(cluster: dict[str, Any]) -> list[str]:
    members = cluster.get("members")
    if not isinstance(members, list):
        return []
    out: list[str] = []
    for member in members:
        if not isinstance(member, str):
            continue
        text = member.strip()
        if text and text not in out:
            out.append(text)
    return out


def _remove_semantic_cluster_members(
    cluster: dict[str, Any],
    removed_members: set[str],
) -> dict[str, Any]:
    payload = deepcopy(cluster)
    members = [
        member
        for member in _semantic_cluster_member_ids(payload)
        if member not in removed_members
    ]
    kept = set(members)
    payload["members"] = members
    anchors = payload.get("anchors")
    if isinstance(anchors, list):
        payload["anchors"] = [
            anchor for anchor in anchors if isinstance(anchor, str) and anchor in kept
        ]
    rules = payload.get("cluster_rules")
    if isinstance(rules, dict):
        payload["cluster_rules"] = _filter_cluster_rules_local(rules, kept)
    footprint = payload.get("cluster_footprint")
    if isinstance(footprint, dict) and isinstance(footprint.get("rects"), list):
        footprint = deepcopy(footprint)
        footprint["rects"] = [
            rect
            for rect in footprint["rects"]
            if isinstance(rect, dict)
            and str(rect.get("id") or "").strip() not in removed_members
        ]
        payload["cluster_footprint"] = footprint
    object_program = payload.get("object_program")
    if isinstance(object_program, dict):
        object_program = deepcopy(object_program)
        specs = object_program.get("object_specs_by_id")
        if isinstance(specs, dict):
            object_program["object_specs_by_id"] = {
                key: value for key, value in specs.items() if key in kept
            }
        payload["object_program"] = object_program
    return payload


def _filter_semantic_layout_program_after_member_dedupe(
    output: dict[str, Any],
    *,
    removals_by_cluster_index: dict[int, set[str]],
    dropped_cluster_ids: set[str],
    surviving_cluster_ids: set[str],
    original_clusters: list[dict[str, Any]],
) -> None:
    program = output.get("semantic_layout_program")
    if not isinstance(program, dict):
        return
    active_clusters = program.get("active_clusters")
    if not isinstance(active_clusters, list):
        return

    removals_by_cluster_id: dict[str, set[str]] = {}
    for index, removed_members in removals_by_cluster_index.items():
        if index >= len(original_clusters):
            continue
        cluster_id = str(original_clusters[index].get("cluster_id") or "").strip()
        if cluster_id:
            removals_by_cluster_id[cluster_id] = set(removed_members)

    filtered_active_clusters: list[dict[str, Any]] = []
    for active_cluster in active_clusters:
        if not isinstance(active_cluster, dict):
            continue
        cluster_id = str(active_cluster.get("cluster_id") or "").strip()
        if (
            not cluster_id
            or cluster_id in dropped_cluster_ids
            or cluster_id not in surviving_cluster_ids
        ):
            continue
        removed_members = removals_by_cluster_id.get(cluster_id, set())
        filtered_active_clusters.append(
            _filter_active_cluster_required_bundles(active_cluster, removed_members)
        )
    program["active_clusters"] = filtered_active_clusters


def _filter_active_cluster_required_bundles(
    active_cluster: dict[str, Any],
    removed_members: set[str],
) -> dict[str, Any]:
    if not removed_members:
        return active_cluster
    payload = deepcopy(active_cluster)
    bundles = payload.get("required_bundles")
    if not isinstance(bundles, list):
        return payload
    filtered_bundles: list[dict[str, Any]] = []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        next_bundle = deepcopy(bundle)
        objects = next_bundle.get("objects")
        if isinstance(objects, list):
            next_bundle["objects"] = [
                item
                for item in objects
                if isinstance(item, dict)
                and str(item.get("object_type") or "").strip() not in removed_members
            ]
        if next_bundle.get("objects"):
            filtered_bundles.append(next_bundle)
    payload["required_bundles"] = filtered_bundles
    return payload


def _log_seed_relation_output(
    cluster_output: dict[str, Any],
    *,
    context: str,
) -> None:
    clusters = cluster_output.get("clusters")
    if not isinstance(clusters, list):
        logger.info("Seed relation planner %s produced no clusters.", context)
        return

    member_to_clusters: dict[str, list[str]] = {}
    logger.info(
        "Seed relation planner %s produced %s cluster(s).", context, len(clusters)
    )
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip() or "unknown_cluster"
        tag = str(cluster.get("tag") or "").strip() or "unknown"
        members = [
            member.strip()
            for member in (cluster.get("members") or [])
            if isinstance(member, str) and member.strip()
        ]
        anchors = [
            anchor.strip()
            for anchor in (cluster.get("anchors") or [])
            if isinstance(anchor, str) and anchor.strip()
        ]
        logger.info(
            "Cluster forge %s cluster %s [%s]: members=%s anchors=%s",
            context,
            cluster_id,
            tag,
            members,
            anchors,
        )
        for member in members:
            member_to_clusters.setdefault(member, []).append(cluster_id)

    duplicates = {
        member: cluster_ids
        for member, cluster_ids in sorted(member_to_clusters.items())
        if len(cluster_ids) > 1
    }
    if duplicates:
        logger.warning(
            "Cluster forge %s duplicate members across clusters: %s",
            context,
            duplicates,
        )


def _filter_cluster_rules_local(
    rules: dict[str, Any],
    kept_ids: set[str],
) -> dict[str, Any]:
    out = deepcopy(rules)
    allowed_rotations = out.get("allowed_rotations")
    if isinstance(allowed_rotations, dict):
        out["allowed_rotations"] = {
            key: value for key, value in allowed_rotations.items() if key in kept_ids
        }

    facing = out.get("facing")
    if isinstance(facing, dict):
        out["facing"] = {key: value for key, value in facing.items() if key in kept_ids}

    access_requirements = out.get("access_requirements")
    if isinstance(access_requirements, list):
        out["access_requirements"] = [
            item
            for item in access_requirements
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item.get("id") in kept_ids
        ]

    semantic_placements = out.get("semantic_placements")
    if isinstance(semantic_placements, list):
        out["semantic_placements"] = [
            item
            for item in semantic_placements
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item.get("id") in kept_ids
            and (
                not isinstance(item.get("relative_to"), str)
                or item.get("relative_to") in kept_ids
            )
        ]
    return out


def run_case(
    *,
    input_payload: dict[str, Any],
    user_id: str,
    description: str | None = None,
    special_notes: str | None = None,
    cases_root: str | Path = "cases",
    case_id: str | None = None,
    ablation_mode: str | None = None,
) -> dict[str, Any]:
    resolved_ablation_mode = resolve_ablation_mode(ablation_mode)
    prepared_input_payload, manual_placements = (
        prepare_input_payload_for_manual_placements(input_payload)
    )
    case_id = case_id or _make_case_id(user_id)
    case_root = Path(cases_root) / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    paths = CasePaths(case_id=case_id, root=case_root)
    paths.clusters_dir.mkdir(parents=True, exist_ok=True)
    paths.module_outputs_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        paths.meta,
        {
            "case_id": case_id,
            "user_id": user_id,
            "created_at_utc": _now_utc_iso(),
            **ablation_metadata(resolved_ablation_mode),
        },
    )
    _write_module_io_manifest(paths)
    _update_status(paths, "room_interpreter")

    room_interpreter = RoomInterpreter()
    room_output = room_interpreter.generate(
        prepared_input_payload,
        description=description,
        special_notes=special_notes,
    ).model_dump()
    room_output = _strip_raw_text(room_output)
    room_output = room_output_for_mode(
        mode=resolved_ablation_mode,
        room_output=room_output,
    )
    _write_module_json(
        paths, "room_interpreter", room_output, legacy_path=paths.room_interpreter
    )

    guidance_text = _build_primary_guidance_text(
        room_output=room_output,
        input_payload=prepared_input_payload,
        description=description,
        special_notes=special_notes,
    )
    planning_guidance_text = append_manual_placements_guidance(
        guidance_text, manual_placements
    )
    room_type = _extract_room_type(prepared_input_payload, room_output)

    seed_relation_planner = SeedRelationPlanner()
    tier_count = TierCountDirector()
    concept_generator = SeedConceptGenerator()
    solver = _build_runtime_solver()
    stylist = Stylist()

    room_meta = room_output.get("meta") if isinstance(room_output, dict) else None
    stylist_tenant_id = _extract_tenant_id_from_payload(prepared_input_payload)
    stylist_user_context = {
        "user_id": user_id,
        "tenant_id": stylist_tenant_id,
        "style": room_meta.get("style") if isinstance(room_meta, dict) else "",
        "room_notes": room_output.get("notes", [])
        if isinstance(room_output, dict)
        else [],
        "guidance_text": guidance_text,
        "user_input": prepared_input_payload.get("user_input", {})
        if isinstance(prepared_input_payload, dict)
        else {},
    }

    _update_status(paths, "stylist_style_policy")
    compiled_style_policy = (
        {}
        if uses_neutral_style_policy(resolved_ablation_mode)
        else stylist.compile_style_policy(
            room_type=room_type,
            brief_text=planning_guidance_text,
            room_model_json=room_output,
        )
    )
    style_policy = style_policy_for_mode(
        mode=resolved_ablation_mode,
        room_type=room_type,
        current_policy=compiled_style_policy,
    )
    stylist_user_context["style_policy"] = style_policy
    _write_module_json(paths, "stylist_style_policy", style_policy)

    _update_status(paths, "seed_relation_planner")
    cluster_output = seed_relation_planner.generate(
        room_type,
        description=planning_guidance_text or None,
        special_notes=None,
        room_model_json=room_output,
        inventory_catalog=_extract_inventory_catalog(prepared_input_payload),
        style_policy_json=style_policy,
        use_semantic_planner=use_semantic_layout_planner(resolved_ablation_mode),
    ).model_dump()
    cluster_output = _strip_raw_text(cluster_output)
    cluster_output = augment_seed_relation_plan_with_manual_placements(
        cluster_output, manual_placements
    )
    cluster_output = _validate_semantic_cluster_unique_members(cluster_output)
    cluster_output = cluster_output_for_mode(
        mode=resolved_ablation_mode,
        cluster_output=cluster_output,
    )
    _log_seed_relation_output(cluster_output, context="primary")
    _write_module_json(
        paths,
        "seed_relation_planner",
        cluster_output,
        legacy_path=paths.seed_relation_planner_legacy,
    )
    request_contract = request_contract_from_payload(cluster_output)
    if request_contract:
        _write_module_json(paths, "request_contract", request_contract)

    _update_status(paths, "tier_count")
    tier_output = tier_count.generate(
        description=planning_guidance_text,
        special_notes="",
        room_model_json=room_output,
        user_intent_json=prepared_input_payload,
        clusters_json=cluster_output,
    )
    tier_output = _strip_raw_text(tier_output)
    tier_output = apply_manual_placements_to_tier_output(
        tier_output, manual_placements, clusters_json=cluster_output
    )
    tier_output = apply_tier_output_for_mode(
        mode=resolved_ablation_mode,
        tier_output=tier_output,
    )
    _write_module_json(
        paths, "tier_count_director", tier_output, legacy_path=paths.tier_count
    )

    _update_status(paths, "cluster_output_merger")
    merged_output = build_solver_object_program(cluster_output, tier_output)
    merged_output = _strip_raw_text(merged_output)
    _write_module_json(
        paths, "cluster_output_merger", merged_output, legacy_path=paths.cluster_merged
    )

    target_final_count = target_final_count_for_mode(resolved_ablation_mode, 5)
    _update_status(
        paths,
        "seed_concept_generator",
        message=_variant_progress_message(
            "Planning layout concepts", 0, target_final_count
        ),
        progress_current=0,
        progress_total=target_final_count,
    )
    seed_relation_plans = concept_generator.generate_bundle(
        room_model_json=room_output,
        clusters_json=merged_output,
        target_count=target_final_count,
    )
    seed_relation_plans = _strip_raw_text(seed_relation_plans)
    _write_module_json(
        paths,
        "seed_concept_generator",
        seed_relation_plans,
        legacy_path=paths.seed_relation_plans,
    )
    concepts = [
        item
        for item in (seed_relation_plans.get("concepts") or [])
        if isinstance(item, dict)
    ]
    if not concepts:
        _update_status(
            paths,
            "error",
            error="layout concept planner could not produce any viable intent plans",
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "layout concept planner could not produce any viable intent plans",
        }

    solved_bundles: list[dict[str, Any]] = []
    solved_bundles.extend(
        _solve_seed_concept_pool(
            concepts=concepts,
            room_output=room_output,
            room_type=room_type,
            cluster_output=cluster_output,
            tier_output=tier_output,
            merged_output=merged_output,
            solver=solver,
            manual_placements=manual_placements,
            ablation_mode=resolved_ablation_mode,
            target_final_count=target_final_count,
            paths=paths,
            progress_offset=len(solved_bundles),
            write_first_relation_plan=True,
            write_first_bundle_artifacts=True,
        )
    )

    styling_candidates = _select_final_styling_candidates(
        solved_bundles,
        max_variants=target_final_count,
        ablation_mode=resolved_ablation_mode,
    )
    selection_summary = build_final_gallery_selection_summary(
        candidates=solved_bundles,
        selected=styling_candidates,
        requested_count=target_final_count,
    )
    material_repair_concepts = _build_material_macro_diversification_repair_concepts(
        concepts=concepts,
        solved_bundles=solved_bundles,
        selection_summary=selection_summary,
        target_final_count=target_final_count,
    )
    if material_repair_concepts and use_material_macro_repair(resolved_ablation_mode):
        seed_relation_plans["material_macro_diversification_repair"] = {
            "attempted": True,
            "repair_concept_count": len(material_repair_concepts),
            "reason": "planned_material_macro_signatures_collapsed_in_solver",
        }
        _update_status(
            paths,
            "solver",
            message="Repairing material macro diversity with committed anchor targets.",
            progress_current=0,
            progress_total=len(material_repair_concepts),
        )
        repair_bundles = _solve_seed_concept_pool(
            concepts=material_repair_concepts,
            room_output=room_output,
            room_type=room_type,
            cluster_output=cluster_output,
            tier_output=tier_output,
            merged_output=merged_output,
            solver=solver,
            manual_placements=manual_placements,
            ablation_mode=resolved_ablation_mode,
            target_final_count=len(material_repair_concepts),
            paths=paths,
            progress_offset=len(solved_bundles),
            status_label="Solving material macro diversity repairs",
        )
        solved_bundles.extend(
            _mark_material_macro_repair_bundles(
                repair_bundles,
                repair_reason="planned_material_macro_signatures_collapsed_in_solver",
            )
        )
        _mark_macro_concept_fallback_collapses(solved_bundles)
        styling_candidates = _select_final_styling_candidates(
            solved_bundles,
            max_variants=target_final_count,
            ablation_mode=resolved_ablation_mode,
        )
        selection_summary = build_final_gallery_selection_summary(
            candidates=solved_bundles,
            selected=styling_candidates,
            requested_count=target_final_count,
        )
        seed_relation_plans["material_macro_diversification_repair"].update(
            {
                "solved_bundle_count": len(repair_bundles),
                "selected_after_repair_count": len(styling_candidates),
                "actual_macro_distinct_after_repair": selection_summary.get(
                    "actual_macro_distinct_candidate_count"
                ),
            }
        )
        _write_module_json(
            paths,
            "seed_concept_generator",
            seed_relation_plans,
            legacy_path=paths.seed_relation_plans,
        )
    elif material_repair_concepts:
        seed_relation_plans["material_macro_diversification_repair"] = {
            "attempted": False,
            "skipped_by_ablation": True,
            "repair_concept_count": len(material_repair_concepts),
            "reason": "no_material_macro_repair",
        }
        _write_module_json(
            paths,
            "seed_concept_generator",
            seed_relation_plans,
            legacy_path=paths.seed_relation_plans,
        )
    core_anchor_repair_concepts = _build_core_anchor_diversity_repair_concepts(
        concepts=concepts,
        solved_bundles=solved_bundles,
        selection_summary=selection_summary,
        target_final_count=target_final_count,
    )
    if (
        core_anchor_repair_concepts
        and use_material_macro_repair(resolved_ablation_mode)
        and use_concept_repair(resolved_ablation_mode)
    ):
        seed_relation_plans["core_anchor_diversity_repair"] = {
            "attempted": True,
            "repair_concept_count": len(core_anchor_repair_concepts),
            "reason": "core_anchor_signatures_collapsed_in_solver",
        }
        _update_status(
            paths,
            "solver",
            message="Repairing core anchor diversity with secondary anchor retargets.",
            progress_current=0,
            progress_total=len(core_anchor_repair_concepts),
        )
        repair_bundles = _solve_seed_concept_pool(
            concepts=core_anchor_repair_concepts,
            room_output=room_output,
            room_type=room_type,
            cluster_output=cluster_output,
            tier_output=tier_output,
            merged_output=merged_output,
            solver=solver,
            manual_placements=manual_placements,
            ablation_mode=resolved_ablation_mode,
            target_final_count=len(core_anchor_repair_concepts),
            paths=paths,
            progress_offset=len(solved_bundles),
            status_label="Solving core anchor diversity repairs",
        )
        solved_bundles.extend(
            _mark_material_macro_repair_bundles(
                repair_bundles,
                repair_reason="core_anchor_signatures_collapsed_in_solver",
            )
        )
        _mark_macro_concept_fallback_collapses(solved_bundles)
        styling_candidates = _select_final_styling_candidates(
            solved_bundles,
            max_variants=target_final_count,
            ablation_mode=resolved_ablation_mode,
        )
        selection_summary = build_final_gallery_selection_summary(
            candidates=solved_bundles,
            selected=styling_candidates,
            requested_count=target_final_count,
        )
        seed_relation_plans["core_anchor_diversity_repair"].update(
            {
                "solved_bundle_count": len(repair_bundles),
                "selected_after_repair_count": len(styling_candidates),
                "core_anchor_distinct_after_repair": selection_summary.get(
                    "core_anchor_distinct_candidate_count"
                ),
            }
        )
        _write_module_json(
            paths,
            "seed_concept_generator",
            seed_relation_plans,
            legacy_path=paths.seed_relation_plans,
        )
    elif core_anchor_repair_concepts:
        seed_relation_plans["core_anchor_diversity_repair"] = {
            "attempted": False,
            "skipped_by_ablation": True,
            "repair_concept_count": len(core_anchor_repair_concepts),
            "reason": "core_anchor_diversity_repair_disabled",
        }
        _write_module_json(
            paths,
            "seed_concept_generator",
            seed_relation_plans,
            legacy_path=paths.seed_relation_plans,
        )
    if not solved_bundles:
        _update_status(
            paths,
            "error",
            error="object-level solver could not produce any feasible anchor-first layouts",
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "object-level solver could not produce any feasible anchor-first layouts",
        }

    if not styling_candidates:
        layout_variants = _build_layout_variants_payload_from_final_candidates(
            [], status="UNSAT"
        )
        layout_variants["selection_summary"] = build_final_gallery_selection_summary(
            candidates=solved_bundles,
            selected=[],
            requested_count=target_final_count,
        )
        _write_module_json(
            paths,
            "layout_variants",
            layout_variants,
            legacy_path=paths.layout_variants,
        )
        error = "object-level solver produced no hard-valid/gallery-eligible layouts"
        _update_status(paths, "error", error=error)
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": error,
        }
    styling_target_count = max(1, len(styling_candidates))

    _update_status(
        paths,
        "stylist",
        message=_variant_progress_message(
            "Styling final layouts", 0, styling_target_count
        ),
        progress_current=0,
        progress_total=styling_target_count,
    )
    final_candidates: list[dict[str, Any]] = []
    for variant_index, solved_bundle in enumerate(styling_candidates, start=1):
        finalized_bundle = _style_judged_variant_bundle(
            bundle=solved_bundle,
            stylist=stylist,
            stylist_user_context_json=_style_variant_user_context(
                stylist_user_context, variant_index=variant_index
            ),
            stylist_tenant_id=stylist_tenant_id,
            manual_placements=manual_placements,
            variant_index=variant_index,
            ablation_mode=resolved_ablation_mode,
        )
        if finalized_bundle is None:
            continue
        final_candidates.append(finalized_bundle)
        _write_partial_layout_variants_artifact(
            paths=paths,
            final_candidates=final_candidates,
            target_final_count=target_final_count,
            ablation_mode=resolved_ablation_mode,
        )
        _update_status(
            paths,
            "stylist",
            message=_variant_progress_message(
                "Styling final layouts",
                len(final_candidates),
                styling_target_count,
            ),
            progress_current=len(final_candidates),
            progress_total=styling_target_count,
        )

    if not final_candidates:
        _update_status(
            paths, "error", error="stylist could not finalize any anchor-first layouts"
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "stylist could not finalize any anchor-first layouts",
        }

    _update_status(
        paths,
        "layout_variants",
        message=_variant_progress_message("Assembling final gallery", 0, 1),
        progress_current=0,
        progress_total=1,
    )
    final_gallery_candidates = _select_final_styling_candidates(
        final_candidates,
        max_variants=target_final_count,
        ablation_mode=resolved_ablation_mode,
    )
    layout_variants = _build_layout_variants_payload_from_final_candidates(
        final_gallery_candidates, status="OK"
    )
    layout_variants["selection_summary"] = build_final_gallery_selection_summary(
        candidates=final_candidates,
        selected=final_gallery_candidates,
        requested_count=target_final_count,
    )
    _write_module_json(
        paths, "layout_variants", layout_variants, legacy_path=paths.layout_variants
    )
    payload_variants = layout_variants.get("variants")
    if not isinstance(payload_variants, list) or not payload_variants:
        _update_status(
            paths, "error", error="final gallery assembly produced no layout options"
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "final gallery assembly produced no layout options",
        }

    canonical_variant = payload_variants[0]
    canonical_absolute_layout = canonical_variant.get("absolute_layout")
    canonical_styled_output = canonical_variant.get("styled_result")
    if not isinstance(canonical_absolute_layout, dict) or not isinstance(
        canonical_styled_output, dict
    ):
        _update_status(
            paths,
            "error",
            error="canonical final option missing styled or layout payload",
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "canonical final option missing styled or layout payload",
        }

    canonical_bundle = deepcopy(final_gallery_candidates[0])
    canonical_bundle["source"] = str(
        canonical_variant.get("source")
        or final_gallery_candidates[0].get("source")
        or "macro_cluster_solver"
    )
    canonical_bundle["reason"] = str(
        canonical_variant.get("reason")
        or final_gallery_candidates[0].get("reason")
        or "Primary canonical variant selected from the final anchor-first layout set."
    )
    _write_module_json(
        paths, "stylist", canonical_styled_output, legacy_path=paths.stylist
    )
    _write_canonical_branch_artifacts(
        paths=paths,
        bundle=canonical_bundle,
        absolute_layout=canonical_absolute_layout,
        styled_output=canonical_styled_output,
    )

    _update_status(
        paths,
        "done",
        message=f"{len(payload_variants)}/{target_final_count} final layout options ready.",
        progress_current=len(payload_variants),
        progress_total=target_final_count,
    )
    return {
        "case_id": case_id,
        "case_dir": str(case_root),
        "final_output": canonical_styled_output,
    }


def _solve_seed_concept_pool(
    *,
    concepts: list[dict[str, Any]],
    room_output: dict[str, Any],
    room_type: str,
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    merged_output: dict[str, Any],
    solver: MacroClusterSolver,
    manual_placements: list[dict[str, Any]] | None,
    ablation_mode: AblationMode,
    target_final_count: int,
    paths: CasePaths | None,
    progress_offset: int = 0,
    status_label: str = "Solving layout concepts",
    write_first_relation_plan: bool = False,
    write_first_bundle_artifacts: bool = False,
) -> list[dict[str, Any]]:
    entries: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for concept_index, concept in enumerate(concepts[:target_final_count], start=1):
        relation_plan = solver_plan_from_concept(
            concept=concept,
            room_model_json=room_output,
            room_type=room_type,
        )
        relation_plan = _strip_raw_text(relation_plan)
        entries.append((concept_index, concept, relation_plan))
        if write_first_relation_plan and concept_index == 1 and paths is not None:
            _write_module_json(
                paths,
                "seed_concept_relation_plan",
                relation_plan,
                legacy_path=paths.cluster_relation_plan,
            )
            _write_module_json(paths, "seed_concept_relation_plan", relation_plan)

    if not entries:
        return []

    cloneable_solver = _solver_is_cloneable(solver)
    parallel_tasks = min(_solver_parallel_limit(target_final_count), len(entries))
    if not cloneable_solver:
        parallel_tasks = 1
    workers_per_task = (
        _solver_workers_per_task(solver, parallel_tasks=parallel_tasks)
        if cloneable_solver
        else 1
    )

    solved_by_concept: list[tuple[int, _ObjectLevelSolveResult]] = []
    if parallel_tasks <= 1:
        for entry in entries:
            concept_index = entry[0]
            if paths is not None:
                _update_status(
                    paths,
                    "solver",
                    message=(
                        f"Concept {concept_index}/{target_final_count}: "
                        "solving object-level anchor-first layout."
                    ),
                    progress_current=progress_offset + len(solved_by_concept),
                    progress_total=target_final_count,
                )
            solved_by_concept.append(
                (
                    concept_index,
                    _solve_seed_concept_entry(
                        entry=entry,
                        room_output=room_output,
                        cluster_output=cluster_output,
                        tier_output=tier_output,
                        merged_output=merged_output,
                        solver=solver,
                        manual_placements=manual_placements,
                        ablation_mode=ablation_mode,
                    ),
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=parallel_tasks) as executor:
            future_by_index = {
                executor.submit(
                    _solve_seed_concept_entry,
                    entry=entry,
                    room_output=room_output,
                    cluster_output=cluster_output,
                    tier_output=tier_output,
                    merged_output=merged_output,
                    solver=_solver_with_workers(solver, num_workers=workers_per_task),
                    manual_placements=manual_placements,
                    ablation_mode=ablation_mode,
                ): entry[0]
                for entry in entries
            }
            for future in as_completed(future_by_index):
                concept_index = future_by_index[future]
                solved_by_concept.append((concept_index, future.result()))
                if paths is not None:
                    _update_status(
                        paths,
                        "solver",
                        message=_variant_progress_message(
                            status_label,
                            progress_offset + len(solved_by_concept),
                            target_final_count,
                        ),
                        progress_current=progress_offset + len(solved_by_concept),
                        progress_total=target_final_count,
                    )

    solved_bundles: list[dict[str, Any]] = []
    failure_diagnostics: list[dict[str, Any]] = []
    first_artifacts_written = False
    for concept_index, solve_result in sorted(
        solved_by_concept, key=lambda row: row[0]
    ):
        failure_diagnostics.extend(solve_result.failure_diagnostics)
        for solution_offset, solved_bundle in enumerate(
            solve_result.bundles,
            start=1,
        ):
            solved_bundle = _refill_judged_variant_accessories(
                bundle=solved_bundle,
                room_output=room_output,
                variant_index=progress_offset + len(solved_bundles) + 1,
                ablation_mode=ablation_mode,
            )
            solved_bundles.append(solved_bundle)
            if (
                write_first_bundle_artifacts
                and not first_artifacts_written
                and paths is not None
                and concept_index == 1
                and solution_offset == 1
            ):
                _write_bundle_stage_artifacts(paths, solved_bundle)
                first_artifacts_written = True

    _mark_macro_concept_fallback_collapses(solved_bundles)
    if paths is not None and failure_diagnostics:
        if not solved_bundles:
            _mark_downstream_module_outputs_stale(
                paths,
                blocked_by="macro_cluster_solver_failure",
            )
        _write_module_json(
            paths,
            "macro_cluster_solver_failure",
            _macro_cluster_solver_failure_payload(failure_diagnostics),
        )

    if paths is not None:
        _update_status(
            paths,
            "solver",
            message=_variant_progress_message(
                status_label,
                progress_offset + len(solved_bundles),
                target_final_count,
            ),
            progress_current=progress_offset + len(solved_bundles),
            progress_total=target_final_count,
        )
    return solved_bundles


def _solve_seed_concept_entry(
    *,
    entry: tuple[int, dict[str, Any], dict[str, Any]],
    room_output: dict[str, Any],
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    merged_output: dict[str, Any],
    solver: MacroClusterSolver,
    manual_placements: list[dict[str, Any]] | None,
    ablation_mode: AblationMode,
) -> _ObjectLevelSolveResult:
    concept_index, concept, relation_plan = entry
    return _solve_object_level_variant_bundle(
        concept=concept,
        relation_plan=relation_plan,
        room_output=room_output,
        cluster_output=cluster_output,
        tier_output=tier_output,
        merged_output=merged_output,
        solver=solver,
        manual_placements=manual_placements,
        variant_index=concept_index,
        ablation_mode=ablation_mode,
    )


def _solver_is_cloneable(solver: object) -> bool:
    return all(
        hasattr(solver, attribute)
        for attribute in (
            "max_rounds",
            "num_workers",
        )
    )


def _summary_int(summary: Mapping[str, Any], key: str) -> int:
    try:
        return int(summary.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _build_material_macro_diversification_repair_concepts(
    *,
    concepts: list[dict[str, Any]],
    solved_bundles: list[dict[str, Any]],
    selection_summary: Mapping[str, Any],
    target_final_count: int,
) -> list[dict[str, Any]]:
    target = min(max(1, int(target_final_count)), 3)
    if target <= 1 or not concepts or not solved_bundles:
        return []
    planned_distinct = _summary_int(
        selection_summary,
        "planned_material_macro_distinct_candidate_count",
    )
    actual_distinct = _summary_int(
        selection_summary,
        "actual_macro_distinct_candidate_count",
    )
    returned_count = _summary_int(selection_summary, "returned_count")
    if actual_distinct >= target and returned_count >= target:
        return []
    if planned_distinct <= actual_distinct and returned_count >= target:
        return []

    available_regions = _material_macro_available_regions(concepts)
    if len(available_regions) < 2:
        return []
    existing_signatures = {
        str(bundle.get("planned_material_macro_signature") or "").strip()
        for bundle in solved_bundles
        if str(bundle.get("planned_material_macro_signature") or "").strip()
    }
    repair_count = max(target_final_count, target + 1)
    repairs: list[dict[str, Any]] = []
    for index, target_region in enumerate(available_regions):
        source = concepts[index % len(concepts)]
        repaired = _retarget_material_macro_concept(
            source,
            primary_region=target_region,
            available_regions=available_regions,
            repair_index=len(repairs) + 1,
        )
        if repaired is None:
            continue
        signature = str(repaired.get("planned_material_macro_signature") or "").strip()
        if not signature or signature in existing_signatures:
            continue
        existing_signatures.add(signature)
        repairs.append(repaired)
        if len(repairs) >= repair_count:
            break
    return repairs


def _build_core_anchor_diversity_repair_concepts(
    *,
    concepts: list[dict[str, Any]],
    solved_bundles: list[dict[str, Any]],
    selection_summary: Mapping[str, Any],
    target_final_count: int,
) -> list[dict[str, Any]]:
    target = min(max(1, int(target_final_count)), 3)
    if target <= 1 or not concepts or not solved_bundles:
        return []
    returned_count = _summary_int(selection_summary, "returned_count")
    core_distinct = _summary_int(
        selection_summary,
        "core_anchor_distinct_candidate_count",
    )
    if returned_count >= target_final_count and core_distinct >= target:
        return []
    available_regions = _material_macro_available_regions(concepts)
    if len(available_regions) < 2:
        return []
    existing_signatures = {
        str(bundle.get("planned_material_macro_signature") or "").strip()
        for bundle in solved_bundles
        if str(bundle.get("planned_material_macro_signature") or "").strip()
    }
    repair_count = max(target_final_count, target + 1)
    repairs: list[dict[str, Any]] = []
    for target_cluster in _CORE_ANCHOR_DIVERSITY_TARGET_CLUSTERS:
        for source in concepts:
            for target_region in available_regions:
                repaired = _retarget_core_anchor_diversity_concept(
                    source,
                    target_cluster_id=target_cluster,
                    target_region=target_region,
                    repair_index=len(repairs) + 1,
                )
                if repaired is None:
                    continue
                signature = str(
                    repaired.get("planned_material_macro_signature") or ""
                ).strip()
                if not signature or signature in existing_signatures:
                    continue
                existing_signatures.add(signature)
                repairs.append(repaired)
                if len(repairs) >= repair_count:
                    return repairs
    return repairs


def _retarget_core_anchor_diversity_concept(
    concept: dict[str, Any],
    *,
    target_cluster_id: str,
    target_region: str,
    repair_index: int,
) -> dict[str, Any] | None:
    if not _material_macro_region_is_assignable(target_region):
        return None
    repaired = deepcopy(concept)
    rows = [
        dict(row)
        for row in repaired.get("cluster_zone_plan") or []
        if isinstance(row, Mapping)
    ]
    target_row = next(
        (
            row
            for row in rows
            if str(row.get("cluster_id") or "").strip() == target_cluster_id
        ),
        None,
    )
    if target_row is None or not _core_anchor_diversity_row_can_retarget(target_row):
        return None
    current_region = str(target_row.get("zone_assignment") or "").strip()
    if target_region == current_region:
        return None
    if target_region in set(_clean_str_list(target_row.get("forbidden_region_ids"))):
        return None
    if not _core_anchor_diversity_region_allowed(target_row, target_region):
        return None
    _retarget_material_macro_row(target_row, target_region)
    signature = _planned_material_macro_signature_from_rows(rows)
    if not signature:
        return None
    repaired["cluster_zone_plan"] = rows
    repaired["concept_id"] = (
        f"{concept.get('concept_id') or 'concept'}_core_anchor_repair_{repair_index:02d}"
    )
    repaired["material_macro_signature_target"] = signature
    repaired["planned_material_macro_signature"] = signature
    repaired["core_anchor_diversity_repair_applied"] = True
    repaired["core_anchor_diversity_repair_source_concept_id"] = str(
        concept.get("concept_id") or ""
    )
    repaired["core_anchor_diversity_target_cluster_id"] = target_cluster_id
    repaired["core_anchor_diversity_target_region"] = target_region
    repaired["anchor_region_preferences_by_cluster"] = (
        _material_macro_anchor_preferences(rows)
    )
    macro_constraints = deepcopy(repaired.get("macro_constraints") or {})
    if isinstance(macro_constraints, dict):
        macro_constraints["anchor_region_constraints"] = (
            _material_macro_anchor_constraints(rows)
        )
        repaired["macro_constraints"] = macro_constraints
    repaired["diversity_signature"] = (
        f"{repaired.get('concept_family') or 'core_anchor_repair'}|{signature}"
    )
    repaired["guidance_notes"] = _collect_unique_notes(
        repaired.get("guidance_notes"),
        ["Deterministic core-anchor diversity repair retargeted a secondary anchor."],
    )
    return repaired


def _core_anchor_diversity_row_can_retarget(row: Mapping[str, Any]) -> bool:
    cluster_id = str(row.get("cluster_id") or "").strip()
    if cluster_id == "sleep_core":
        return False
    return cluster_id in _CORE_ANCHOR_DIVERSITY_TARGET_CLUSTERS


def _core_anchor_diversity_region_allowed(
    row: Mapping[str, Any],
    target_region: str,
) -> bool:
    cluster_id = str(row.get("cluster_id") or "").strip()
    if cluster_id == "work_study":
        token = target_region.strip().lower()
        return "window" in token or "daylight" in token
    return True


def _material_macro_available_regions(
    concepts: list[dict[str, Any]],
) -> list[str]:
    region_ids: set[str] = set()
    for concept in concepts:
        macro_map = concept.get("macro_region_map_snapshot")
        if isinstance(macro_map, Mapping):
            for row in macro_map.get("regions") or []:
                if not isinstance(row, Mapping):
                    continue
                region_id = str(row.get("region_id") or "").strip()
                if region_id:
                    region_ids.add(region_id)
        for row in concept.get("cluster_zone_plan") or []:
            if not isinstance(row, Mapping):
                continue
            region_id = str(row.get("zone_assignment") or "").strip()
            if region_id:
                region_ids.add(region_id)
            region_ids.update(_clean_str_list(row.get("preferred_region_ids")))
    out = [
        region_id
        for region_id in _MATERIAL_MACRO_REPAIR_REGION_PRIORITY
        if region_id in region_ids and _material_macro_region_is_assignable(region_id)
    ]
    out.extend(
        sorted(
            region_id
            for region_id in region_ids
            if region_id not in out and _material_macro_region_is_assignable(region_id)
        )
    )
    return out


def _material_macro_region_is_assignable(region_id: str) -> bool:
    token = region_id.strip().lower()
    if not token:
        return False
    return not any(
        blocked in token
        for blocked in (
            "entry",
            "door",
            "corridor",
            "keep_open",
            "center_openness",
            "floating_center",
        )
    )


def _retarget_material_macro_concept(
    concept: dict[str, Any],
    *,
    primary_region: str,
    available_regions: Sequence[str],
    repair_index: int,
) -> dict[str, Any] | None:
    repaired = deepcopy(concept)
    rows = [
        dict(row)
        for row in repaired.get("cluster_zone_plan") or []
        if isinstance(row, Mapping)
    ]
    primary_row = _material_macro_primary_row(rows)
    if primary_row is None:
        return None
    secondary_row = _material_macro_secondary_row(rows, primary_row)
    _retarget_material_macro_row(primary_row, primary_region)
    if secondary_row is not None:
        complement = _material_macro_complement_region(
            primary_region=primary_region,
            available_regions=available_regions,
        )
        if complement is not None and complement != primary_region:
            _retarget_material_macro_row(secondary_row, complement)
    signature = _planned_material_macro_signature_from_rows(rows)
    repaired["cluster_zone_plan"] = rows
    repaired["concept_id"] = (
        f"{concept.get('concept_id') or 'concept'}_material_repair_{repair_index:02d}"
    )
    repaired["material_macro_signature_target"] = signature
    repaired["planned_material_macro_signature"] = signature
    repaired["material_macro_repair_applied"] = True
    repaired["material_macro_repair_source_concept_id"] = str(
        concept.get("concept_id") or ""
    )
    repaired["anchor_region_preferences_by_cluster"] = (
        _material_macro_anchor_preferences(rows)
    )
    macro_constraints = deepcopy(repaired.get("macro_constraints") or {})
    if isinstance(macro_constraints, dict):
        macro_constraints["anchor_region_constraints"] = (
            _material_macro_anchor_constraints(rows)
        )
        repaired["macro_constraints"] = macro_constraints
    repaired["diversity_signature"] = (
        f"{repaired.get('concept_family') or 'material_repair'}|{signature}"
    )
    repaired["guidance_notes"] = _collect_unique_notes(
        repaired.get("guidance_notes"),
        ["Deterministic material macro diversity repair retargeted core anchors."],
    )
    return repaired


def _material_macro_primary_row(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("scenario_role") or "") == "primary":
            return row
    for row in rows:
        if _material_row_is_core(row):
            return row
    return rows[0] if rows else None


def _material_macro_secondary_row(
    rows: list[dict[str, Any]],
    primary_row: Mapping[str, Any],
) -> dict[str, Any] | None:
    primary_cluster_id = str(primary_row.get("cluster_id") or "").strip()
    for row in rows:
        if (
            str(row.get("cluster_id") or "").strip() != primary_cluster_id
            and str(row.get("scenario_role") or "") == "secondary"
        ):
            return row
    for row in rows:
        if str(
            row.get("cluster_id") or ""
        ).strip() != primary_cluster_id and _material_row_is_core(row):
            return row
    return None


def _retarget_material_macro_row(row: dict[str, Any], region_id: str) -> None:
    wall_side = _material_macro_wall_side(region_id)
    row["zone_assignment"] = region_id
    row["committed_region_ids"] = [region_id]
    row["committed_wall_side"] = wall_side
    if wall_side:
        row["preferred_wall_side"] = wall_side
    preferred = [region_id, *_clean_str_list(row.get("preferred_region_ids"))]
    row["preferred_region_ids"] = _ordered_unique_strings(preferred)
    if str(row.get("scenario_role") or "") in {"primary", "secondary"}:
        row["required_region_ids"] = [region_id]


def _material_macro_wall_side(region_id: str) -> str:
    token = region_id.strip().lower()
    if "window" in token or "daylight" in token:
        return "window_side"
    for wall_side in ("left_wall", "right_wall", "top_wall", "bottom_wall"):
        if wall_side in token:
            return wall_side
    if "center" in token or "floating" in token:
        return "center"
    return ""


def _material_macro_complement_region(
    *,
    primary_region: str,
    available_regions: Sequence[str],
) -> str | None:
    preferred = _MATERIAL_MACRO_COMPLEMENT_BY_REGION.get(primary_region)
    if preferred in available_regions:
        return preferred
    primary_wall = _material_macro_wall_side(primary_region)
    for region_id in available_regions:
        if region_id == primary_region:
            continue
        wall_side = _material_macro_wall_side(region_id)
        if wall_side and wall_side != primary_wall:
            return region_id
    return next(
        (region_id for region_id in available_regions if region_id != primary_region),
        None,
    )


def _material_macro_anchor_preferences(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        cluster_id = str(row.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        out[cluster_id] = {
            "zone_assignment": str(row.get("zone_assignment") or ""),
            "committed_region_ids": _clean_str_list(row.get("committed_region_ids")),
            "committed_wall_side": str(row.get("committed_wall_side") or ""),
            "required_region_ids": _clean_str_list(row.get("required_region_ids")),
            "preferred_region_ids": _clean_str_list(row.get("preferred_region_ids")),
            "forbidden_region_ids": _clean_str_list(row.get("forbidden_region_ids")),
            "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
            "anchor_strength": str(row.get("anchor_strength") or "medium"),
            "scenario_role": str(row.get("scenario_role") or "support"),
            "placement_behavior": deepcopy(row.get("placement_behavior") or {}),
        }
    return out


def _material_macro_anchor_constraints(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "cluster_id": str(row.get("cluster_id") or ""),
            "committed_region_ids": _clean_str_list(row.get("committed_region_ids")),
            "committed_wall_side": str(row.get("committed_wall_side") or ""),
            "required_region_ids": _clean_str_list(row.get("required_region_ids")),
            "preferred_region_ids": _clean_str_list(row.get("preferred_region_ids")),
            "forbidden_region_ids": _clean_str_list(row.get("forbidden_region_ids")),
            "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
            "anchor_strength": str(row.get("anchor_strength") or "medium"),
            "scenario_role": str(row.get("scenario_role") or "support"),
        }
        for row in rows
        if str(row.get("cluster_id") or "").strip()
    ]


def _ordered_unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _mark_material_macro_repair_bundles(
    bundles: list[dict[str, Any]],
    *,
    repair_reason: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bundle in bundles:
        next_bundle = deepcopy(bundle)
        source = str(next_bundle.get("source") or "")
        if source:
            next_bundle["source"] = f"material_macro_repair:{source}"
        next_bundle["material_macro_repair_applied"] = True
        next_bundle["material_macro_repair_reason"] = repair_reason
        next_bundle["notes"] = _collect_unique_notes(
            next_bundle.get("notes"),
            ["Deterministic material macro diversity repair applied before styling."],
        )
        out.append(next_bundle)
    return out


def _planned_material_macro_signature(
    *,
    concept: Mapping[str, Any],
    relation_plan: Mapping[str, Any],
) -> str:
    payloads = [
        concept,
        relation_plan,
        relation_plan.get("macro_concept")
        if isinstance(relation_plan, Mapping)
        else {},
    ]
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in (
            "planned_material_macro_signature",
            "material_macro_signature_target",
        ):
            signature = str(payload.get(key) or "").strip()
            if signature:
                return signature
    return _planned_material_macro_signature_from_rows(
        concept.get("cluster_zone_plan")
        if isinstance(concept.get("cluster_zone_plan"), list)
        else []
    )


def _planned_material_macro_signature_from_rows(rows: Sequence[Any]) -> str:
    signature_rows: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        if not cluster_id or not _material_row_is_core(row):
            continue
        region_ids = _clean_str_list(row.get("committed_region_ids"))
        region = (
            region_ids[0]
            if region_ids
            else str(row.get("zone_assignment") or "").strip()
        )
        if not region:
            continue
        wall_side = str(
            row.get("committed_wall_side") or row.get("preferred_wall_side") or ""
        ).strip()
        signature_rows.append(
            f"{cluster_id}:{_material_macro_target_token(region, wall_side)}"
        )
    if not signature_rows:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            region = str(row.get("zone_assignment") or "").strip()
            if cluster_id and region:
                signature_rows.append(f"{cluster_id}:{region}")
    signature_rows.sort()
    return "|".join(signature_rows)


def _material_row_is_core(row: Mapping[str, Any]) -> bool:
    if str(row.get("priority") or "").strip() == "core":
        return True
    if str(row.get("scenario_role") or "").strip() in {"primary", "secondary"}:
        return True
    return str(row.get("anchor_strength") or "").strip().lower() in {
        "hard",
        "strong",
    }


def _material_macro_target_token(region: str, wall_side: str) -> str:
    if not wall_side or wall_side == "center" or wall_side in region:
        return region
    if any(token in region for token in ("private", "quiet", "edge", "focal")):
        return f"{region}@{wall_side}"
    return region


def _mark_macro_concept_fallback_collapses(bundles: list[dict[str, Any]]) -> None:
    seen_actual: dict[str, dict[str, str]] = {}
    for bundle in bundles:
        actual_signature = str(
            bundle.get("actual_macro_layout_signature") or ""
        ).strip()
        planned_signature = str(
            bundle.get("planned_material_macro_signature") or ""
        ).strip()
        if not actual_signature:
            absolute_layout = bundle.get("absolute_layout")
            if isinstance(absolute_layout, dict):
                actual_signature = material_macro_layout_signature(absolute_layout)
                bundle["actual_macro_layout_signature"] = actual_signature
        if not actual_signature:
            continue
        previous = seen_actual.get(actual_signature)
        if previous is not None and planned_signature != previous.get("planned"):
            used_committed_fallback = _bundle_used_committed_anchor_fallback(bundle)
            bundle["macro_concept_collapsed_to_fallback"] = used_committed_fallback
            bundle["macro_concept_fallback_duplicate_of"] = previous.get("source", "")
            if used_committed_fallback:
                bundle["notes"] = _collect_unique_notes(
                    bundle.get("notes"),
                    [
                        "Committed material macro target missed; solver used a fallback anchor macro already seen."
                    ],
                )
            continue
        seen_actual[actual_signature] = {
            "planned": planned_signature,
            "source": str(bundle.get("source") or ""),
        }


def _bundle_used_committed_anchor_fallback(bundle: Mapping[str, Any]) -> bool:
    diagnostics = bundle.get("anchor_candidate_diagnostics_by_cluster")
    if not isinstance(diagnostics, Mapping):
        solver_output = bundle.get("solver_output")
        if isinstance(solver_output, Mapping):
            solver_debug = solver_output.get("solver_debug")
            if isinstance(solver_debug, Mapping):
                diagnostics = solver_debug.get(
                    "anchor_candidate_diagnostics_by_cluster"
                )
    if not isinstance(diagnostics, Mapping):
        return False
    return any(
        isinstance(row, Mapping)
        and (
            bool(row.get("committed_region_missed"))
            or bool(row.get("committed_region_relaxed_after_strict_fail"))
        )
        for row in diagnostics.values()
    )


def _solve_object_level_variant_bundle(
    *,
    concept: dict[str, Any],
    relation_plan: dict[str, Any],
    room_output: dict[str, Any],
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    merged_output: dict[str, Any],
    solver: MacroClusterSolver,
    manual_placements: list[dict[str, Any]] | None,
    variant_index: int,
    ablation_mode: AblationMode,
) -> _ObjectLevelSolveResult:
    failure_diagnostics: list[dict[str, Any]] = []
    try:
        solver_output = _run_object_level_solver_output(
            solver=solver,
            room_output=room_output,
            merged_output=merged_output,
            relation_plan=relation_plan,
            cluster_output=cluster_output,
        )
        bundles = _object_level_solver_output_to_bundles(
            concept=concept,
            relation_plan=relation_plan,
            room_output=room_output,
            cluster_output=cluster_output,
            tier_output=tier_output,
            merged_output=merged_output,
            solver_output=solver_output,
            manual_placements=manual_placements,
            variant_index=variant_index,
            ablation_mode=ablation_mode,
            repair_attempt=None,
        )
        if bundles:
            return _ObjectLevelSolveResult(
                bundles=bundles,
                failure_diagnostics=failure_diagnostics,
            )
        failure_diagnostics.append(
            _object_level_failure_diagnostic(
                concept=concept,
                relation_plan=relation_plan,
                merged_output=merged_output,
                solver_output=solver_output,
                repair_attempt=None,
            )
        )
        if not use_concept_repair(ablation_mode):
            return _ObjectLevelSolveResult(
                bundles=[],
                failure_diagnostics=failure_diagnostics,
            )

        repair_queue = [
            (attempt, 1)
            for attempt in _build_concept_repair_attempts(
                concept=concept,
                relation_plan=relation_plan,
                merged_output=merged_output,
                solver_output=solver_output,
            )
        ]
        seen_repair_signatures = {
            _concept_repair_attempt_signature(attempt)
            for attempt, _depth in repair_queue
        }
        repair_attempt_count = 0
        while repair_queue and repair_attempt_count < _CONCEPT_REPAIR_MAX_ATTEMPTS:
            repair_attempt, repair_depth = repair_queue.pop(0)
            repair_attempt_count += 1
            repaired_solver_output = _run_object_level_solver_output(
                solver=solver,
                room_output=room_output,
                merged_output=repair_attempt.merged_output,
                relation_plan=repair_attempt.relation_plan,
                cluster_output=cluster_output,
            )
            repaired_bundles = _object_level_solver_output_to_bundles(
                concept=repair_attempt.concept,
                relation_plan=repair_attempt.relation_plan,
                room_output=room_output,
                cluster_output=cluster_output,
                tier_output=tier_output,
                merged_output=repair_attempt.merged_output,
                solver_output=repaired_solver_output,
                manual_placements=manual_placements,
                variant_index=variant_index,
                ablation_mode=ablation_mode,
                repair_attempt=repair_attempt,
            )
            if repaired_bundles:
                return _ObjectLevelSolveResult(
                    bundles=repaired_bundles,
                    failure_diagnostics=failure_diagnostics,
                )
            failure_diagnostics.append(
                _object_level_failure_diagnostic(
                    concept=repair_attempt.concept,
                    relation_plan=repair_attempt.relation_plan,
                    merged_output=repair_attempt.merged_output,
                    solver_output=repaired_solver_output,
                    repair_attempt=repair_attempt,
                )
            )
            if repair_depth >= _CONCEPT_REPAIR_MAX_DEPTH:
                continue
            for follow_up_attempt in _build_concept_repair_attempts(
                concept=repair_attempt.concept,
                relation_plan=repair_attempt.relation_plan,
                merged_output=repair_attempt.merged_output,
                solver_output=repaired_solver_output,
            ):
                signature = _concept_repair_attempt_signature(follow_up_attempt)
                if signature in seen_repair_signatures:
                    continue
                seen_repair_signatures.add(signature)
                repair_queue.append((follow_up_attempt, repair_depth + 1))
        return _ObjectLevelSolveResult(
            bundles=[],
            failure_diagnostics=failure_diagnostics,
        )
    except Exception as exc:
        logger.warning(
            "Skipping concept %s because object-level solving failed: %s",
            concept.get("concept_id") or variant_index,
            exc,
        )
        failure_diagnostics.append(
            {
                "concept_id": str(concept.get("concept_id") or variant_index),
                "repair_attempt": None,
                "status": "EXCEPTION",
                "error": str(exc),
            }
        )
        return _ObjectLevelSolveResult(
            bundles=[],
            failure_diagnostics=failure_diagnostics,
        )


def _concept_repair_attempt_signature(attempt: _ConceptRepairAttempt) -> str:
    return json.dumps(
        {
            "relation_plan": attempt.relation_plan,
            "merged_output": attempt.merged_output,
        },
        sort_keys=True,
        default=str,
    )


def _run_object_level_solver_output(
    *,
    solver: MacroClusterSolver,
    room_output: dict[str, Any],
    merged_output: dict[str, Any],
    relation_plan: dict[str, Any],
    cluster_output: dict[str, Any],
) -> dict[str, Any]:
    solver_output = solver.generate_object_layout(
        room_model_json=room_output,
        merged_clusters_json=merged_output,
        relation_plan_json=relation_plan,
        cluster_constraints_json=cluster_output,
        grid_mm=GLOBAL_LAYOUT_GRID_MM,
    )
    return _strip_raw_text(solver_output)


def _macro_cluster_solver_failure_payload(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    quality_reasons: dict[str, int] = {}
    offending_objects: dict[str, int] = {}
    zero_anchor_clusters: dict[str, int] = {}
    for row in diagnostics:
        for reason in row.get("quality_gate_reasons") or []:
            key = str(reason or "").strip()
            if key:
                quality_reasons[key] = quality_reasons.get(key, 0) + 1
        for cluster_id in row.get("zero_anchor_candidate_clusters") or []:
            key = str(cluster_id or "").strip()
            if key:
                zero_anchor_clusters[key] = zero_anchor_clusters.get(key, 0) + 1
        for issue in row.get("blocking_objects") or []:
            object_id = str(issue.get("object_id") or "").strip()
            if object_id:
                offending_objects[object_id] = offending_objects.get(object_id, 0) + 1
    return {
        "status": "UNSAT",
        "attempt_count": len(diagnostics),
        "top_quality_gate_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                quality_reasons.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "top_offending_objects": [
            {"object_id": object_id, "count": count}
            for object_id, count in sorted(
                offending_objects.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "zero_anchor_candidate_clusters": [
            {"cluster_id": cluster_id, "count": count}
            for cluster_id, count in sorted(
                zero_anchor_clusters.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "attempts": diagnostics,
    }


def _object_level_failure_diagnostic(
    *,
    concept: Mapping[str, Any],
    relation_plan: Mapping[str, Any],
    merged_output: Mapping[str, Any],
    solver_output: Mapping[str, Any],
    repair_attempt: _ConceptRepairAttempt | None,
) -> dict[str, Any]:
    best_partial = _as_mapping(solver_output.get("best_partial"))
    best_verify = _as_mapping(best_partial.get("verify_summary"))
    degradation = _as_mapping(solver_output.get("solver_degradation_report"))
    repair_diagnostics = _concept_repair_diagnostics(solver_output)
    return {
        "concept_id": str(concept.get("concept_id") or ""),
        "concept_family": str(concept.get("concept_family") or ""),
        "repair_attempt": repair_attempt.repair_id if repair_attempt else None,
        "repair_notes": list(repair_attempt.notes) if repair_attempt else [],
        "status": str(solver_output.get("status") or ""),
        "quality_gate_reasons": _clean_str_list(
            best_verify.get("quality_gate_reasons")
        ),
        "offending_clusters": _clean_str_list(best_verify.get("offending_clusters")),
        "zero_anchor_candidate_clusters": repair_diagnostics[
            "zero_anchor_candidate_clusters"
        ],
        "blocking_objects": _blocking_object_rows_from_verify(best_verify),
        "front_access_blocking_pairs": repair_diagnostics[
            "front_access_blocking_pairs"
        ],
        "daylight_quality_blocking_pairs": repair_diagnostics[
            "daylight_quality_blocking_pairs"
        ],
        "protected_ids_by_cluster": _solver_policy_map(
            relation_plan,
            "protected_ids_by_cluster",
        ),
        "droppable_ids_by_cluster": _solver_policy_map(
            relation_plan,
            "droppable_ids_by_cluster",
        ),
        "required_support_ids_by_cluster": _required_support_ids_by_cluster(
            merged_output,
            relation_plan,
        ),
        "best_partial_verify_summary": deepcopy(dict(best_verify)),
        "solver_debug": deepcopy(dict(_as_mapping(solver_output.get("solver_debug")))),
        "solver_degradation_report": deepcopy(dict(degradation)),
    }


def _blocking_object_rows_from_verify(
    verify: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "blocking_orientation_issues",
        "blocking_protected_region_issues",
        "blocking_forbidden_region_issues",
        "blocking_functional_geometry_issues",
    ):
        for issue in verify.get(key) or []:
            if isinstance(issue, Mapping):
                rows.append(
                    {
                        "issue_type": key,
                        "cluster_id": str(issue.get("cluster_id") or ""),
                        "object_id": str(
                            issue.get("object_id")
                            or issue.get("blocked_object_id")
                            or ""
                        ),
                        "reason": str(issue.get("reason") or ""),
                    }
                )
    return rows


def _solver_policy_map(
    relation_plan: Mapping[str, Any],
    key: str,
) -> dict[str, list[str]]:
    policy = _as_mapping(relation_plan.get("object_solver_policy"))
    value = policy.get(key)
    if not isinstance(value, Mapping):
        return {}
    return {
        str(cluster_id): _clean_str_list(ids)
        for cluster_id, ids in value.items()
        if str(cluster_id).strip()
    }


def _required_support_ids_by_cluster(
    merged_output: Mapping[str, Any],
    relation_plan: Mapping[str, Any],
) -> dict[str, list[str]]:
    out = _solver_policy_map(relation_plan, "required_support_ids_by_cluster")
    for cluster in merged_output.get("clusters") or []:
        if not isinstance(cluster, Mapping):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        object_program = _as_mapping(cluster.get("object_program"))
        support_ids = _clean_str_list(
            object_program.get("required_support_ids")
            or object_program.get("relaxable_required_ids")
        )
        if support_ids:
            out[cluster_id] = support_ids
    return out


def _object_level_solver_output_to_bundles(
    *,
    concept: dict[str, Any],
    relation_plan: dict[str, Any],
    room_output: dict[str, Any],
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    merged_output: dict[str, Any],
    solver_output: dict[str, Any],
    manual_placements: list[dict[str, Any]] | None,
    variant_index: int,
    ablation_mode: AblationMode,
    repair_attempt: _ConceptRepairAttempt | None,
) -> list[dict[str, Any]]:
    solver_candidates = [
        item
        for item in (solver_output.get("solutions") or [])
        if isinstance(item, dict)
    ]
    if not solver_candidates:
        solver_candidates = [solver_output]

    bundles: list[dict[str, Any]] = []
    for solution_index, solver_candidate in enumerate(solver_candidates, start=1):
        solver_candidate = deepcopy(solver_candidate)
        absolute_layout = _strip_raw_text(
            solver_candidate.get("absolute_layout")
            or solver_output.get("absolute_layout")
            or {}
        )
        if not isinstance(absolute_layout, dict) or str(
            absolute_layout.get("status") or ""
        ).upper() not in {"OK", "PARTIAL"}:
            continue
        absolute_layout = merge_manual_placements_into_absolute_layout(
            absolute_layout,
            manual_placements,
        )
        absolute_layout = _annotate_object_level_layout_coverage(
            absolute_layout=absolute_layout,
            merged_output=merged_output,
            solver_output=solver_candidate,
        )
        hard_valid = bool(
            solver_candidate.get("hard_valid", solver_output.get("hard_valid", False))
        )
        geometry_valid = bool(
            solver_candidate.get(
                "geometry_valid",
                solver_output.get(
                    "geometry_valid",
                    solver_candidate.get(
                        "hard_valid", solver_output.get("hard_valid", False)
                    ),
                ),
            )
        )
        complete = bool(
            solver_candidate.get("complete", solver_output.get("complete", False))
        )
        gallery_eligible = bool(
            solver_candidate.get(
                "gallery_eligible",
                solver_output.get("gallery_eligible", False),
            )
        )
        if repair_attempt is not None and not (hard_valid and gallery_eligible):
            continue

        base_note = (
            f"Anchor-first concept {concept.get('concept_id') or variant_index} "
            "solved directly at object level."
        )
        repair_notes: list[str] = []
        if repair_attempt is not None:
            repair_notes = [
                f"Deterministic concept repair applied: {repair_attempt.repair_id}.",
                *repair_attempt.notes,
            ]
        notes = _collect_unique_notes(
            [base_note],
            repair_notes,
            absolute_layout.get("notes") if isinstance(absolute_layout, dict) else [],
            solver_candidate.get("notes") if isinstance(solver_candidate, dict) else [],
            solver_output.get("notes") if isinstance(solver_output, dict) else [],
        )
        if repair_attempt is not None:
            solver_candidate["concept_repair_applied"] = True
            solver_candidate["concept_repair_attempt"] = repair_attempt.repair_id
            solver_candidate["concept_repair_notes"] = list(repair_attempt.notes)
            absolute_layout["concept_repair_applied"] = True
            absolute_layout["concept_repair_attempt"] = repair_attempt.repair_id
        planned_material_signature = _planned_material_macro_signature(
            concept=concept,
            relation_plan=relation_plan,
        )
        actual_macro_signature = material_macro_layout_signature(absolute_layout)
        core_anchor_signature = core_anchor_layout_signature(absolute_layout)
        bundle = {
            "concept": deepcopy(concept),
            "relation_plan": relation_plan,
            "room_output": room_output,
            "cluster_output": cluster_output,
            "tier_output": tier_output,
            "merged_output": merged_output,
            "solver_output": solver_candidate,
            "absolute_layout": absolute_layout,
            "layout_score": int(
                (solver_candidate.get("verify_summary") or {}).get("layout_score")
                or (solver_output.get("verify_summary") or {}).get("layout_score")
                or 0
            ),
            "hard_valid": hard_valid,
            "geometry_valid": geometry_valid,
            "complete": complete,
            "gallery_eligible": gallery_eligible,
            "coverage_ratio": float(
                solver_candidate.get("coverage_ratio")
                or solver_output.get("coverage_ratio")
                or 0.0
            ),
            "missing_cluster_ids": deepcopy(
                (solver_candidate.get("verify_summary") or {}).get(
                    "missing_cluster_ids"
                )
                or (solver_output.get("verify_summary") or {}).get(
                    "missing_cluster_ids"
                )
                or []
            ),
            "notes": notes,
            "source": f"concept:{concept.get('concept_id') or variant_index}",
            "reason": (
                "Solved from macro concept "
                f"{concept.get('concept_family') or concept.get('concept_id') or variant_index} "
                "using object-level anchor-first search."
            ),
            "dropped_inventory_by_cluster": deepcopy(
                solver_candidate.get("dropped_inventory_by_cluster")
                or solver_output.get("dropped_inventory_by_cluster")
                or {}
            ),
            "state_signature": str(
                solver_candidate.get("state_signature")
                or concept.get("diversity_signature")
                or concept.get("concept_id")
                or f"{variant_index}_{solution_index}"
            ),
            "macro_layout_signature": str(
                concept.get("diversity_signature")
                or concept.get("concept_family")
                or variant_index
            ),
            "planned_material_macro_signature": planned_material_signature,
            "actual_macro_layout_signature": actual_macro_signature,
            "core_anchor_signature": core_anchor_signature,
            "anchor_candidate_diagnostics_by_cluster": deepcopy(
                (
                    solver_candidate.get("solver_debug")
                    if isinstance(solver_candidate.get("solver_debug"), dict)
                    else solver_output.get("solver_debug")
                    if isinstance(solver_output.get("solver_debug"), dict)
                    else {}
                ).get("anchor_candidate_diagnostics_by_cluster")
                or {}
            ),
            "ablation_mode": ablation_mode,
        }
        if repair_attempt is not None:
            bundle["concept_repair_applied"] = True
            bundle["concept_repair_attempt"] = repair_attempt.repair_id
            bundle["concept_repair_notes"] = list(repair_attempt.notes)
        bundles.append(bundle)
    return bundles


def _build_concept_repair_attempts(
    *,
    concept: dict[str, Any],
    relation_plan: dict[str, Any],
    merged_output: dict[str, Any],
    solver_output: dict[str, Any],
) -> list[_ConceptRepairAttempt]:
    diagnostics = _concept_repair_diagnostics(solver_output)
    if str(solver_output.get("status") or "").upper() != "UNSAT":
        return []

    attempts: list[_ConceptRepairAttempt] = []
    zone_pileup_clusters = _relation_plan_zone_pileup_clusters(relation_plan)
    zero_anchor_candidate_clusters = set(diagnostics["zero_anchor_candidate_clusters"])
    needs_anchor_pair_spread = diagnostics.get(
        "failure_kind"
    ) == "no_compatible_anchor_set" and bool(
        diagnostics.get("pairwise_incompatible_anchor_pairs")
    )
    needs_zone_spread = bool(zone_pileup_clusters) or (
        diagnostics["raw_solution_count"] == 0 and not needs_anchor_pair_spread
    )
    needs_functional = bool(
        set(diagnostics["quality_gate_reasons"])
        & {
            "view_corridor_blocked",
            "front_access_zone_blocked",
            "front_inward_contract_failed",
        }
    )
    needs_secondary_front_access = "front_access_zone_blocked" in set(
        diagnostics["quality_gate_reasons"]
    ) and bool(diagnostics.get("front_access_blocking_pairs"))
    needs_secondary_storage_rehome = needs_secondary_front_access and bool(
        diagnostics.get("anchor_wall_options_by_cluster")
    )
    needs_secondary_daylight_quality = "blocking_daylight_quality_overlap" in set(
        diagnostics["quality_gate_reasons"]
    ) and bool(diagnostics.get("daylight_quality_blocking_pairs"))
    support_changed = _support_droppable_repair_has_changes(merged_output)
    required_support_changed = _required_support_relaxation_has_changes(merged_output)
    needs_required_support = required_support_changed and bool(
        set(diagnostics["quality_gate_reasons"])
        & {
            "blocking_forbidden_region_overlap",
            "blocking_protected_region_overlap",
            "front_inward_contract_failed",
            "layout_incomplete",
        }
    )

    if needs_anchor_pair_spread:
        attempt = _make_concept_repair_attempt(
            repair_id="anchor_pair_spread",
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=True,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=False,
            apply_functional_gate=False,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    if zero_anchor_candidate_clusters:
        attempt = _make_concept_repair_attempt(
            repair_id="primary_corridor_softening",
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=True,
            apply_zone_spread=False,
            apply_functional_gate=False,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    if needs_zone_spread:
        attempt = _make_concept_repair_attempt(
            repair_id="zone_spread",
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=True,
            apply_functional_gate=False,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    if needs_secondary_storage_rehome:
        attempt = _make_concept_repair_attempt(
            repair_id="secondary_storage_wall_rehome",
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=False,
            apply_functional_gate=False,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            apply_secondary_storage_wall_rehome=True,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    if needs_secondary_front_access:
        attempt = _make_concept_repair_attempt(
            repair_id="secondary_front_access_relaxation",
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=False,
            apply_functional_gate=False,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            apply_secondary_front_access_relaxation=True,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    if needs_secondary_daylight_quality:
        attempt = _make_concept_repair_attempt(
            repair_id="secondary_daylight_quality_relaxation",
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=False,
            apply_functional_gate=False,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            apply_secondary_daylight_quality_relaxation=True,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    if needs_functional:
        attempt = _make_concept_repair_attempt(
            repair_id="functional_gate",
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=False,
            apply_functional_gate=True,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    if needs_required_support:
        attempt = _make_concept_repair_attempt(
            repair_id="required_support_relaxation",
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=False,
            apply_functional_gate=False,
            apply_required_support_relaxation=True,
            apply_support_droppable=False,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    if support_changed:
        repair_id = (
            "combined_zone_support"
            if needs_zone_spread or needs_functional
            else "support_droppable"
        )
        attempt = _make_concept_repair_attempt(
            repair_id=repair_id,
            concept=concept,
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics=diagnostics,
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=bool(zero_anchor_candidate_clusters),
            apply_zone_spread=needs_zone_spread,
            apply_functional_gate=needs_functional,
            apply_required_support_relaxation=needs_required_support,
            apply_support_droppable=True,
            zone_pileup_clusters=zone_pileup_clusters,
        )
        if attempt is not None:
            attempts.append(attempt)

    unique_attempts: list[_ConceptRepairAttempt] = []
    seen: set[str] = set()
    for attempt in attempts:
        signature = json.dumps(
            {
                "relation_plan": attempt.relation_plan,
                "merged_output": attempt.merged_output,
            },
            sort_keys=True,
            default=str,
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique_attempts.append(attempt)
        if len(unique_attempts) >= _CONCEPT_REPAIR_MAX_ATTEMPTS:
            break
    return unique_attempts


def _concept_repair_diagnostics(solver_output: Mapping[str, Any]) -> dict[str, Any]:
    debug = _as_mapping(solver_output.get("solver_debug"))
    degradation = _as_mapping(solver_output.get("solver_degradation_report"))
    best_partial = _as_mapping(solver_output.get("best_partial"))
    best_verify = _as_mapping(best_partial.get("verify_summary"))

    raw_solution_count = _int_or_none(debug.get("raw_solution_count"))
    if raw_solution_count is None:
        raw_solution_count = _int_or_none(degradation.get("raw_solution_count"))

    quality_gate_reasons = set(_clean_str_list(best_verify.get("quality_gate_reasons")))
    for item in degradation.get("top_quality_gate_reasons") or []:
        if isinstance(item, Mapping):
            reason = str(item.get("reason") or "").strip()
            if reason:
                quality_gate_reasons.add(reason)

    offending_clusters = set(_clean_str_list(solver_output.get("offending_clusters")))
    offending_clusters.update(_clean_str_list(best_verify.get("offending_clusters")))
    for issue_key in (
        "blocking_protected_region_issues",
        "blocking_forbidden_region_issues",
        "blocking_orientation_issues",
        "blocking_functional_geometry_issues",
    ):
        for issue in best_verify.get(issue_key) or []:
            if isinstance(issue, Mapping):
                cluster_id = str(issue.get("cluster_id") or "").strip()
                if cluster_id:
                    offending_clusters.add(cluster_id)
    front_access_blocking_pairs = _front_access_blocking_pairs_from_verify(best_verify)
    daylight_quality_blocking_pairs = _daylight_quality_blocking_pairs_from_verify(
        best_verify
    )

    anchor_counts = _as_mapping(
        debug.get("anchor_candidate_counts")
        or degradation.get("anchor_candidate_counts")
    )
    failure_kind = str(
        solver_output.get("failure_kind")
        or debug.get("failure_kind")
        or degradation.get("failure_kind")
        or ""
    ).strip()
    anchor_pair_compatibility = [
        dict(item)
        for item in (
            debug.get("anchor_pair_compatibility")
            or degradation.get("anchor_pair_compatibility")
            or []
        )
        if isinstance(item, Mapping)
    ]
    pairwise_incompatible_anchor_pairs = [
        dict(item)
        for item in (
            debug.get("pairwise_incompatible_anchor_pairs")
            or degradation.get("pairwise_incompatible_anchor_pairs")
            or []
        )
        if isinstance(item, Mapping)
    ]
    anchor_wall_options_by_cluster = _as_mapping(
        debug.get("anchor_wall_options_by_cluster")
        or degradation.get("anchor_wall_options_by_cluster")
    )
    zero_anchor_candidate_clusters = sorted(
        cluster_id
        for cluster_id, count in anchor_counts.items()
        if (_int_or_none(count) or 0) <= 0
    )
    offending_clusters.update(zero_anchor_candidate_clusters)
    for row in pairwise_incompatible_anchor_pairs:
        for key in ("cluster_a", "cluster_b"):
            cluster_id = str(row.get(key) or "").strip()
            if cluster_id:
                offending_clusters.add(cluster_id)
    return {
        "failure_kind": failure_kind,
        "raw_solution_count": raw_solution_count,
        "quality_gate_reasons": sorted(quality_gate_reasons),
        "offending_clusters": sorted(offending_clusters),
        "anchor_candidate_counts": dict(anchor_counts),
        "zero_anchor_candidate_clusters": zero_anchor_candidate_clusters,
        "anchor_pair_compatibility": anchor_pair_compatibility,
        "pairwise_incompatible_anchor_pairs": pairwise_incompatible_anchor_pairs,
        "front_access_blocking_pairs": front_access_blocking_pairs,
        "daylight_quality_blocking_pairs": daylight_quality_blocking_pairs,
        "anchor_wall_options_by_cluster": deepcopy(
            dict(anchor_wall_options_by_cluster)
        ),
    }


def _front_access_blocking_pairs_from_verify(
    verify: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for issue in verify.get("blocking_functional_geometry_issues") or []:
        if not isinstance(issue, Mapping):
            continue
        if str(issue.get("reason") or "").strip() != "front_access_zone_blocked":
            continue
        blocker_cluster_id = str(issue.get("cluster_id") or "").strip()
        blocker_object_id = str(issue.get("object_id") or "").strip()
        blocked_cluster_id = str(issue.get("blocked_cluster_id") or "").strip()
        blocked_object_id = str(issue.get("blocked_object_id") or "").strip()
        if not blocker_cluster_id or not blocker_object_id:
            continue
        if not blocked_cluster_id or not blocked_object_id:
            continue
        pairs.append(
            {
                "blocker_cluster_id": blocker_cluster_id,
                "blocker_object_id": blocker_object_id,
                "blocked_cluster_id": blocked_cluster_id,
                "blocked_object_id": blocked_object_id,
                "overlap_ratio": issue.get("overlap_ratio"),
                "overlap_area_mm2": issue.get("overlap_area_mm2"),
            }
        )
    return pairs


def _daylight_quality_blocking_pairs_from_verify(
    verify: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for issue in verify.get("blocking_forbidden_region_issues") or []:
        if not isinstance(issue, Mapping):
            continue
        region_id = str(issue.get("region_id") or "").strip().lower()
        zone_type = str(issue.get("zone_type") or "").strip().lower()
        if "daylight" not in region_id and zone_type not in {
            "daylight_quality_blocking",
            "daylight_quality_soft",
        }:
            continue
        cluster_id = str(issue.get("cluster_id") or "").strip()
        object_id = str(issue.get("object_id") or "").strip()
        if not cluster_id or not object_id:
            continue
        pairs.append(
            {
                "cluster_id": cluster_id,
                "object_id": object_id,
                "region_id": str(issue.get("region_id") or ""),
                "zone_type": str(issue.get("zone_type") or ""),
                "overlap_ratio": issue.get("overlap_ratio"),
                "overlap_area_mm2": issue.get("overlap_area_mm2"),
            }
        )
    return pairs


def _make_concept_repair_attempt(
    *,
    repair_id: str,
    concept: dict[str, Any],
    relation_plan: dict[str, Any],
    merged_output: dict[str, Any],
    diagnostics: Mapping[str, Any],
    apply_anchor_pair_spread: bool,
    apply_primary_corridor_relaxation: bool,
    apply_zone_spread: bool,
    apply_functional_gate: bool,
    apply_required_support_relaxation: bool,
    apply_support_droppable: bool,
    zone_pileup_clusters: set[str],
    apply_secondary_storage_wall_rehome: bool = False,
    apply_secondary_front_access_relaxation: bool = False,
    apply_secondary_daylight_quality_relaxation: bool = False,
) -> _ConceptRepairAttempt | None:
    repaired_concept = deepcopy(concept)
    repaired_relation_plan = deepcopy(relation_plan)
    repaired_merged_output = deepcopy(merged_output)
    notes: list[str] = []
    changed = False

    if apply_anchor_pair_spread:
        anchor_pair_changed = _apply_anchor_pair_spread_repair(
            relation_plan=repaired_relation_plan,
            concept=repaired_concept,
            diagnostics=diagnostics,
        )
        changed = anchor_pair_changed or changed
        if anchor_pair_changed:
            notes.append(
                "Spread pairwise-incompatible hard anchors across alternate preferred regions."
            )

    if apply_primary_corridor_relaxation:
        corridor_changed = _apply_primary_corridor_relaxation_repair(
            relation_plan=repaired_relation_plan,
            concept=repaired_concept,
            merged_output=repaired_merged_output,
            diagnostics=diagnostics,
        )
        changed = corridor_changed or changed
        if corridor_changed:
            notes.append(
                "Softened primary room-center corridor constraints for zero-candidate anchors."
            )

    if apply_zone_spread:
        changed = (
            _apply_zone_spread_repair(
                relation_plan=repaired_relation_plan,
                concept=repaired_concept,
                diagnostics=diagnostics,
                zone_pileup_clusters=zone_pileup_clusters,
            )
            or changed
        )
        if changed:
            notes.append("Relaxed crowded zone/wall pins before retrying the solver.")

    if apply_functional_gate:
        functional_changed = _apply_functional_gate_repair(
            relation_plan=repaired_relation_plan,
            concept=repaired_concept,
            diagnostics=diagnostics,
        )
        changed = functional_changed or changed
        if functional_changed:
            notes.append(
                "Moved offending clusters toward safer edge/window/bottom regions "
                "while preserving protected topology."
            )

    if apply_required_support_relaxation:
        required_support_changed = _apply_required_support_relaxation(
            merged_output=repaired_merged_output,
            relation_plan=repaired_relation_plan,
            concept=repaired_concept,
        )
        changed = required_support_changed or changed
        if required_support_changed:
            notes.append(
                "Relaxed required support placement while keeping it required for final coverage."
            )

    if apply_support_droppable:
        support_changed = _apply_support_droppable_repair(
            merged_output=repaired_merged_output,
            relation_plan=repaired_relation_plan,
            concept=repaired_concept,
        )
        changed = support_changed or changed
        if support_changed:
            notes.append(
                "Moved optional support objects from protected to droppable sets."
            )

    if apply_secondary_storage_wall_rehome:
        storage_rehome_changed = _apply_secondary_storage_wall_rehome(
            relation_plan=repaired_relation_plan,
            concept=repaired_concept,
            merged_output=repaired_merged_output,
            diagnostics=diagnostics,
        )
        changed = storage_rehome_changed or changed
        if storage_rehome_changed:
            notes.append(
                "Moved secondary storage anchor toward a legal alternate wall before relaxing front access."
            )

    if apply_secondary_front_access_relaxation:
        secondary_front_changed = _apply_secondary_front_access_relaxation(
            merged_output=repaired_merged_output,
            relation_plan=repaired_relation_plan,
            concept=repaired_concept,
            diagnostics=diagnostics,
        )
        changed = secondary_front_changed or changed
        if secondary_front_changed:
            notes.append(
                "Relaxed secondary front-access clearance after hard anchor spacing failed."
            )

    if apply_secondary_daylight_quality_relaxation:
        secondary_daylight_changed = _apply_secondary_daylight_quality_relaxation(
            merged_output=repaired_merged_output,
            diagnostics=diagnostics,
        )
        changed = secondary_daylight_changed or changed
        if secondary_daylight_changed:
            notes.append(
                "Softened daylight-quality overlap for secondary storage/media after strict layout failed."
            )

    if "front_inward_contract_failed" in set(
        diagnostics.get("quality_gate_reasons") or []
    ):
        front_changed = _apply_storage_front_token_repair(
            merged_output=repaired_merged_output,
            diagnostics=diagnostics,
        )
        changed = front_changed or changed
        if front_changed:
            notes.append("Normalized missing storage-anchor front metadata.")

    if not changed:
        return None
    _append_concept_repair_note(repaired_relation_plan, repair_id, notes)
    return _ConceptRepairAttempt(
        repair_id=repair_id,
        concept=repaired_concept,
        relation_plan=repaired_relation_plan,
        merged_output=repaired_merged_output,
        notes=tuple(notes),
    )


def _apply_anchor_pair_spread_repair(
    *,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    anchor_prefs = _anchor_region_preferences_by_cluster(relation_plan)
    if not anchor_prefs:
        return False

    target_clusters: list[str] = []
    for pair in diagnostics.get("pairwise_incompatible_anchor_pairs") or []:
        if not isinstance(pair, Mapping):
            continue
        cluster_id = _select_anchor_pair_spread_target(pair, anchor_prefs)
        if cluster_id and cluster_id not in target_clusters:
            target_clusters.append(cluster_id)
    if not target_clusters:
        return False

    used_regions = _assigned_repair_regions(anchor_prefs)
    changed = False
    softened_corridor = False
    for cluster_id in target_clusters:
        pref = _as_mutable_mapping(anchor_prefs.get(cluster_id))
        if not pref:
            continue
        current_region = str(pref.get("zone_assignment") or "").strip()
        used_regions.discard(current_region)
        next_region = _select_repair_region(
            pref,
            used_regions=used_regions,
            functional=False,
        )
        updates: dict[str, Any] = {}
        if next_region and next_region != current_region:
            updates["zone_assignment"] = next_region
            updates["committed_region_ids"] = [next_region]
            updates["preferred_region_ids"] = _prepend_unique_region(
                pref.get("preferred_region_ids"),
                next_region,
            )
            updates["committed_wall_side"] = ""
            used_regions.add(next_region)
        else:
            used_regions.add(current_region)
        if str(pref.get("preferred_wall_side") or "").strip():
            updates["preferred_wall_side"] = ""
        if str(pref.get("anchor_strength") or "").strip().lower() == "strong":
            updates["anchor_strength"] = "medium"
        if str(pref.get("placement_bias") or "").strip() == "wall_backed":
            updates["placement_bias"] = "balanced"
        placement_behavior = _as_mutable_mapping(pref.get("placement_behavior"))
        if placement_behavior:
            next_behavior = dict(placement_behavior)
            if str(next_behavior.get("wall_backing") or "") == "required":
                next_behavior["wall_backing"] = "preferred"
            if str(next_behavior.get("pose_flexibility") or "") == "low":
                next_behavior["pose_flexibility"] = "medium"
            if next_behavior != placement_behavior:
                updates["placement_behavior"] = next_behavior
        if not updates:
            continue
        _sync_repair_cluster_updates(
            relation_plan=relation_plan,
            concept=concept,
            cluster_id=cluster_id,
            updates=updates,
        )
        changed = True
    corridor_targets = set(target_clusters)
    if corridor_targets:
        for target in _relation_plan_relaxation_targets(relation_plan, concept):
            softened_corridor = (
                _strip_primary_corridor_forbidden_regions(
                    target,
                    cluster_ids=corridor_targets,
                )
                or softened_corridor
            )
            softened_corridor = (
                _soften_primary_corridor_protected_topology(
                    target,
                    cluster_ids=corridor_targets,
                    relaxation_reason=(
                        "anchor_pair_spread_corridor_softening_after_strict_solver_failure"
                    ),
                )
                or softened_corridor
            )
    changed = softened_corridor or changed
    return changed


def _select_anchor_pair_spread_target(
    pair: Mapping[str, Any],
    anchor_prefs: Mapping[str, dict[str, Any]],
) -> str:
    candidates = [
        str(pair.get(key) or "").strip()
        for key in ("cluster_a", "cluster_b")
        if str(pair.get(key) or "").strip() in anchor_prefs
    ]
    if not candidates:
        return ""
    return min(
        candidates,
        key=lambda cluster_id: _anchor_pair_spread_target_rank(
            cluster_id,
            anchor_prefs.get(cluster_id) or {},
        ),
    )


def _anchor_pair_spread_target_rank(
    cluster_id: str,
    pref: Mapping[str, Any],
) -> tuple[int, int, int, str]:
    role_tokens = {
        str(pref.get(key) or "").strip().lower()
        for key in ("scenario_role", "layout_role", "priority", "semantic_role")
    }
    role_kind = str(pref.get("role_kind") or "").strip().lower()
    primary_rank = 1 if "primary" in role_tokens else 0
    sleep_rank = 1 if role_kind == "sleep" or "sleep" in cluster_id.lower() else 0
    strength_rank = {
        "low": 0,
        "weak": 0,
        "medium": 1,
        "": 1,
        "strong": 2,
    }.get(str(pref.get("anchor_strength") or "").strip().lower(), 1)
    return (primary_rank, sleep_rank, strength_rank, cluster_id)


def _apply_primary_corridor_relaxation_repair(
    *,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    merged_output: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    zero_anchor_clusters = set(
        _clean_str_list(diagnostics.get("zero_anchor_candidate_clusters"))
    )
    if not zero_anchor_clusters:
        return False

    changed = False
    for target in _relation_plan_relaxation_targets(relation_plan, concept):
        changed = (
            _strip_primary_corridor_forbidden_regions(
                target,
                cluster_ids=zero_anchor_clusters,
            )
            or changed
        )
        changed = (
            _soften_primary_corridor_protected_topology(
                target,
                cluster_ids=zero_anchor_clusters,
            )
            or changed
        )
    changed = (
        _strip_primary_corridor_avoid_regions_from_merged_output(
            merged_output,
            cluster_ids=zero_anchor_clusters,
        )
        or changed
    )
    return changed


def _relation_plan_relaxation_targets(
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = [relation_plan]
    macro_concept = relation_plan.get("macro_concept")
    if isinstance(macro_concept, dict):
        targets.append(macro_concept)
    if concept is not relation_plan:
        targets.append(concept)
    return targets


def _strip_primary_corridor_forbidden_regions(
    plan: dict[str, Any],
    *,
    cluster_ids: set[str],
) -> bool:
    changed = False
    for section_name in (
        "anchor_layout_hints_by_cluster",
        "anchor_region_preferences_by_cluster",
    ):
        section = plan.get(section_name)
        if not isinstance(section, dict):
            continue
        for cluster_id in cluster_ids:
            row = section.get(cluster_id)
            if isinstance(row, dict):
                changed = _filter_primary_corridor_region_list(row) or changed

    macro_constraints = _as_mutable_mapping(plan.get("macro_constraints"))
    rows = macro_constraints.get("anchor_region_constraints")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            if cluster_id in cluster_ids:
                changed = _filter_primary_corridor_region_list(row) or changed
    return changed


def _filter_primary_corridor_region_list(row: dict[str, Any]) -> bool:
    region_ids = _clean_str_list(row.get("forbidden_region_ids"))
    if not region_ids:
        return False
    kept = [
        region_id
        for region_id in region_ids
        if not _region_id_is_relaxable_primary_corridor(region_id)
    ]
    if kept == region_ids:
        return False
    row["forbidden_region_ids"] = kept
    return True


def _soften_primary_corridor_protected_topology(
    plan: dict[str, Any],
    *,
    cluster_ids: set[str],
    relaxation_reason: str = (
        "zero_anchor_candidate_corridor_softening_after_strict_solver_failure"
    ),
) -> bool:
    macro_constraints = _as_mutable_mapping(plan.get("macro_constraints"))
    rows = macro_constraints.get("protected_topology")
    if not isinstance(rows, list):
        return False
    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _protected_topology_row_is_relaxable_primary_corridor(row):
            continue
        current = set(_clean_str_list(row.get("softened_for_clusters")))
        updated = current | cluster_ids
        if updated == current:
            continue
        row["softened_for_clusters"] = sorted(updated)
        row["relaxation_reason"] = relaxation_reason
        changed = True
    return changed


def _strip_primary_corridor_avoid_regions_from_merged_output(
    merged_output: dict[str, Any],
    *,
    cluster_ids: set[str],
) -> bool:
    clusters = merged_output.get("clusters")
    if not isinstance(clusters, list):
        return False
    changed = False
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if cluster_id not in cluster_ids:
            continue
        for claims in _cluster_zone_claim_targets(cluster):
            changed = _filter_primary_corridor_avoid_regions(claims) or changed
    return changed


def _cluster_zone_claim_targets(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for container_name in ("cluster_rules", "object_program"):
        container = cluster.get(container_name)
        if not isinstance(container, dict):
            continue
        claims = container.get("zone_claims")
        if isinstance(claims, dict):
            targets.append(claims)
    claims = cluster.get("zone_claims")
    if isinstance(claims, dict):
        targets.append(claims)
    return targets


def _filter_primary_corridor_avoid_regions(claims: dict[str, Any]) -> bool:
    avoid_regions = _clean_str_list(claims.get("avoid_regions"))
    if not avoid_regions:
        return False
    kept = [
        region_id
        for region_id in avoid_regions
        if not _region_id_is_relaxable_primary_corridor(region_id)
    ]
    if kept == avoid_regions:
        return False
    claims["avoid_regions"] = kept
    return True


def _protected_topology_row_is_relaxable_primary_corridor(
    row: Mapping[str, Any],
) -> bool:
    region_id = str(row.get("region") or row.get("region_id") or "").strip()
    if _region_id_is_strict_opening_or_daylight(region_id):
        return False
    zone_type = str(row.get("zone_type") or "").strip().lower()
    if zone_type in _PRIMARY_CORRIDOR_RELAXATION_ZONE_TYPES:
        return True
    return _region_id_is_relaxable_primary_corridor(region_id)


def _region_id_is_strict_opening_or_daylight(region_id: str) -> bool:
    token = region_id.strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return False
    if "corridor" in token:
        return False
    return any(
        part in token
        for part in (
            "door_swing",
            "entry_clearance",
            "window_clearance",
            "window",
            "daylight",
        )
    )


def _region_id_is_relaxable_primary_corridor(region_id: str) -> bool:
    token = region_id.strip().lower().replace("-", "_").replace(" ", "_")
    if not token or _region_id_is_strict_opening_or_daylight(token):
        return False
    if token in {"entry_to_center_corridor", "primary_circulation_corridor"}:
        return True
    if token.endswith("_to_room_center_corridor") or token.endswith(
        "_to_center_corridor"
    ):
        return True
    return "corridor" in token and (
        "room_center" in token or "center" in token or "primary" in token
    )


def _apply_zone_spread_repair(
    *,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    diagnostics: Mapping[str, Any],
    zone_pileup_clusters: set[str],
) -> bool:
    anchor_prefs = _anchor_region_preferences_by_cluster(relation_plan)
    if not anchor_prefs:
        return False
    offenders = set(_clean_str_list(diagnostics.get("offending_clusters")))
    raw_solution_count = diagnostics.get("raw_solution_count")
    affected_clusters = set(zone_pileup_clusters)
    if raw_solution_count == 0:
        affected_clusters.update(anchor_prefs.keys())
    if offenders:
        affected_clusters.update(offenders)
    if not affected_clusters:
        affected_clusters.update(anchor_prefs.keys())

    used_regions = _assigned_repair_regions(anchor_prefs)
    changed = False
    for cluster_id in sorted(affected_clusters):
        pref = _as_mutable_mapping(anchor_prefs.get(cluster_id))
        if not pref:
            continue
        current_region = str(pref.get("zone_assignment") or "").strip()
        used_regions.discard(current_region)
        next_region = _select_repair_region(
            pref,
            used_regions=used_regions,
            functional=False,
        )
        updates: dict[str, Any] = {}
        if next_region and next_region != current_region:
            updates["zone_assignment"] = next_region
            updates["preferred_region_ids"] = _prepend_unique_region(
                pref.get("preferred_region_ids"),
                next_region,
            )
            used_regions.add(next_region)
        else:
            used_regions.add(current_region)
        if str(pref.get("preferred_wall_side") or "").strip():
            updates["preferred_wall_side"] = ""
        if str(pref.get("anchor_strength") or "").strip().lower() == "strong":
            updates["anchor_strength"] = "medium"
        if str(pref.get("placement_bias") or "").strip() in {
            "wall_backed",
            "float_allowed",
        }:
            updates["placement_bias"] = "balanced"
        if updates:
            _sync_repair_cluster_updates(
                relation_plan=relation_plan,
                concept=concept,
                cluster_id=cluster_id,
                updates=updates,
            )
            changed = True
    return changed


def _apply_functional_gate_repair(
    *,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    anchor_prefs = _anchor_region_preferences_by_cluster(relation_plan)
    offenders = set(_clean_str_list(diagnostics.get("offending_clusters")))
    if not anchor_prefs or not offenders:
        return False
    used_regions = _assigned_repair_regions(anchor_prefs)
    changed = False
    for cluster_id in sorted(offenders):
        pref = _as_mutable_mapping(anchor_prefs.get(cluster_id))
        if not pref:
            continue
        current_region = str(pref.get("zone_assignment") or "").strip()
        used_regions.discard(current_region)
        next_region = _select_repair_region(
            pref,
            used_regions=used_regions,
            functional=True,
        )
        updates: dict[str, Any] = {}
        if next_region and next_region != current_region:
            updates["zone_assignment"] = next_region
            updates["preferred_region_ids"] = _prepend_unique_region(
                pref.get("preferred_region_ids"),
                next_region,
            )
            used_regions.add(next_region)
        else:
            used_regions.add(current_region)
        if str(pref.get("preferred_wall_side") or "").strip():
            updates["preferred_wall_side"] = ""
        if str(pref.get("anchor_strength") or "").strip().lower() == "strong":
            updates["anchor_strength"] = "medium"
        if str(pref.get("center_usage") or "").strip().lower() == "primary":
            updates["center_usage"] = "partial"
        placement_behavior = _as_mutable_mapping(pref.get("placement_behavior"))
        if placement_behavior:
            next_behavior = dict(placement_behavior)
            if str(next_behavior.get("wall_backing") or "") == "required":
                next_behavior["wall_backing"] = "preferred"
            if str(next_behavior.get("pose_flexibility") or "") == "low":
                next_behavior["pose_flexibility"] = "medium"
            if next_behavior != placement_behavior:
                updates["placement_behavior"] = next_behavior
        if updates:
            _sync_repair_cluster_updates(
                relation_plan=relation_plan,
                concept=concept,
                cluster_id=cluster_id,
                updates=updates,
            )
            changed = True
    return changed


def _support_droppable_repair_has_changes(merged_output: Mapping[str, Any]) -> bool:
    for cluster in merged_output.get("clusters") or []:
        if not isinstance(cluster, Mapping):
            continue
        optional_ids = _optional_support_member_ids(cluster)
        if optional_ids:
            return True
    return False


def _required_support_relaxation_has_changes(merged_output: Mapping[str, Any]) -> bool:
    for cluster in merged_output.get("clusters") or []:
        if not isinstance(cluster, Mapping):
            continue
        if _required_support_member_ids(cluster):
            return True
    return False


def _apply_required_support_relaxation(
    *,
    merged_output: dict[str, Any],
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
) -> bool:
    changed = False
    for cluster in merged_output.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        required_support_ids = _required_support_member_ids(cluster)
        if not required_support_ids:
            continue
        object_program = _as_mutable_mapping(cluster.get("object_program"))
        if object_program:
            current_required_support = _clean_str_list(
                object_program.get("required_support_ids")
            )
            next_required_support = _ordered_unique(
                [*current_required_support, *sorted(required_support_ids)]
            )
            if next_required_support != current_required_support:
                object_program["required_support_ids"] = next_required_support
                changed = True
            current_relaxable = _clean_str_list(
                object_program.get("relaxable_required_ids")
            )
            next_relaxable = _ordered_unique(
                [*current_relaxable, *sorted(required_support_ids)]
            )
            if next_relaxable != current_relaxable:
                object_program["relaxable_required_ids"] = next_relaxable
                changed = True
            protected = _clean_str_list(object_program.get("protected_ids"))
            next_protected = [
                item for item in protected if item not in required_support_ids
            ]
            if next_protected != protected:
                object_program["protected_ids"] = next_protected
                changed = True
            droppable = _clean_str_list(object_program.get("droppable_ids"))
            next_droppable = [
                item for item in droppable if item not in required_support_ids
            ]
            if next_droppable != droppable:
                object_program["droppable_ids"] = next_droppable
                changed = True
            specs = _as_mutable_mapping(object_program.get("object_specs_by_id"))
            for object_id in required_support_ids:
                spec = _as_mutable_mapping(specs.get(object_id))
                if not spec:
                    continue
                for key, value in (
                    ("protected", False),
                    ("droppable", False),
                    ("required_support", True),
                    ("relaxable_required", True),
                ):
                    if spec.get(key) != value:
                        spec[key] = value
                        changed = True

        rules = _as_mutable_mapping(cluster.get("cluster_rules"))
        policy = _as_mutable_mapping(rules.get("anchor_first_policy"))
        if policy:
            protected = _clean_str_list(policy.get("protected_ids"))
            next_protected = [
                item for item in protected if item not in required_support_ids
            ]
            if next_protected != protected:
                policy["protected_ids"] = next_protected
                changed = True
            droppable = _clean_str_list(policy.get("droppable_ids"))
            next_droppable = [
                item for item in droppable if item not in required_support_ids
            ]
            if next_droppable != droppable:
                policy["droppable_ids"] = next_droppable
                changed = True

        _sync_object_solver_policy_support_sets(
            relation_plan=relation_plan,
            concept=concept,
            cluster_id=cluster_id,
            required_support_ids=sorted(required_support_ids),
        )
        _sync_object_solver_policy_drop_sets(
            relation_plan=relation_plan,
            concept=concept,
            cluster_id=cluster_id,
            protected_ids=_clean_str_list(
                object_program.get("protected_ids") if object_program else []
            ),
            droppable_ids=_clean_str_list(
                object_program.get("droppable_ids") if object_program else []
            ),
        )
        changed = True
    return changed


def _apply_support_droppable_repair(
    *,
    merged_output: dict[str, Any],
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
) -> bool:
    changed = False
    for cluster in merged_output.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        optional_ids = _optional_support_member_ids(cluster)
        if not optional_ids:
            continue
        rules = _as_mutable_mapping(cluster.get("cluster_rules"))
        policy = _as_mutable_mapping(rules.get("anchor_first_policy"))
        if not policy:
            continue
        protected = _clean_str_list(policy.get("protected_ids"))
        droppable = _clean_str_list(policy.get("droppable_ids"))
        next_protected = [item for item in protected if item not in optional_ids]
        next_droppable = _ordered_unique([*droppable, *sorted(optional_ids)])
        if next_protected != protected or next_droppable != droppable:
            policy["protected_ids"] = next_protected
            policy["droppable_ids"] = next_droppable
            changed = True
        if _omit_optional_support_from_cluster_program(cluster, optional_ids):
            changed = True
        _sync_object_solver_policy_drop_sets(
            relation_plan=relation_plan,
            concept=concept,
            cluster_id=cluster_id,
            protected_ids=next_protected,
            droppable_ids=next_droppable,
        )
    return changed


def _apply_secondary_front_access_relaxation(
    *,
    merged_output: dict[str, Any],
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    pairs = [
        pair
        for pair in diagnostics.get("front_access_blocking_pairs") or []
        if isinstance(pair, Mapping)
    ]
    if not pairs:
        return False
    clusters_by_id = {
        str(cluster.get("cluster_id") or "").strip(): cluster
        for cluster in merged_output.get("clusters") or []
        if isinstance(cluster, dict) and str(cluster.get("cluster_id") or "").strip()
    }
    changed = False
    for pair in pairs:
        cluster_id = str(pair.get("blocked_cluster_id") or "").strip()
        object_id = str(pair.get("blocked_object_id") or "").strip()
        if not cluster_id or not object_id:
            continue
        cluster = clusters_by_id.get(cluster_id)
        if not isinstance(cluster, dict):
            continue
        if _front_access_target_is_droppable_support(cluster, object_id):
            changed = (
                _drop_front_access_optional_support(
                    cluster=cluster,
                    object_id=object_id,
                    relation_plan=relation_plan,
                    concept=concept,
                )
                or changed
            )
            continue
        if not _secondary_front_access_target_can_relax(
            cluster=cluster,
            object_id=object_id,
            relation_plan=relation_plan,
        ):
            continue
        changed = (
            _soften_front_access_requirement(
                cluster=cluster,
                object_id=object_id,
            )
            or changed
        )
    return changed


def _apply_secondary_storage_wall_rehome(
    *,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    merged_output: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    pairs = [
        pair
        for pair in diagnostics.get("front_access_blocking_pairs") or []
        if isinstance(pair, Mapping)
    ]
    if not pairs:
        return False
    clusters_by_id = {
        str(cluster.get("cluster_id") or "").strip(): cluster
        for cluster in merged_output.get("clusters") or []
        if isinstance(cluster, dict) and str(cluster.get("cluster_id") or "").strip()
    }
    wall_options_by_cluster = _as_mapping(
        diagnostics.get("anchor_wall_options_by_cluster")
    )
    anchor_prefs = _anchor_region_preferences_by_cluster(relation_plan)
    changed = False
    for pair in pairs:
        cluster_id = str(pair.get("blocked_cluster_id") or "").strip()
        object_id = str(pair.get("blocked_object_id") or "").strip()
        if not cluster_id or not object_id:
            continue
        cluster = clusters_by_id.get(cluster_id)
        if not isinstance(cluster, dict):
            continue
        if _front_access_target_is_droppable_support(cluster, object_id):
            continue
        if not _secondary_front_access_target_can_relax(
            cluster=cluster,
            object_id=object_id,
            relation_plan=relation_plan,
        ):
            continue
        pref = _as_mutable_mapping(anchor_prefs.get(cluster_id))
        if not pref:
            continue
        wall_option = _select_secondary_storage_rehome_wall_option(
            options=_as_mapping(wall_options_by_cluster.get(cluster_id)),
            current_wall_side=_current_wall_side_for_repair(pref),
        )
        if not wall_option:
            continue
        wall_side = str(wall_option.get("wall_side") or "").strip().lower()
        target_region = _wall_side_region_id(wall_side)
        if not target_region or _region_blocked_for_repair(target_region):
            continue
        forbidden_regions = set(_clean_str_list(pref.get("forbidden_region_ids")))
        if target_region in forbidden_regions:
            continue
        current_region = str(pref.get("zone_assignment") or "").strip()
        if target_region == current_region:
            continue
        updates: dict[str, Any] = {
            "zone_assignment": target_region,
            "committed_region_ids": [target_region],
            "preferred_region_ids": _prepend_unique_region(
                pref.get("preferred_region_ids"),
                target_region,
            ),
            "committed_wall_side": f"{wall_side}_wall",
            "preferred_wall_side": f"{wall_side}_wall",
        }
        if str(pref.get("anchor_strength") or "").strip().lower() == "strong":
            updates["anchor_strength"] = "medium"
        if str(pref.get("placement_bias") or "").strip() == "wall_backed":
            updates["placement_bias"] = "balanced"
        placement_behavior = _as_mutable_mapping(pref.get("placement_behavior"))
        if placement_behavior:
            next_behavior = dict(placement_behavior)
            if str(next_behavior.get("wall_backing") or "") == "required":
                next_behavior["wall_backing"] = "preferred"
            if str(next_behavior.get("pose_flexibility") or "") == "low":
                next_behavior["pose_flexibility"] = "medium"
            if next_behavior != placement_behavior:
                updates["placement_behavior"] = next_behavior
        _sync_repair_cluster_updates(
            relation_plan=relation_plan,
            concept=concept,
            cluster_id=cluster_id,
            updates=updates,
        )
        changed = True
    return changed


def _apply_secondary_daylight_quality_relaxation(
    *,
    merged_output: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    pairs = [
        pair
        for pair in diagnostics.get("daylight_quality_blocking_pairs") or []
        if isinstance(pair, Mapping)
    ]
    if not pairs:
        return False
    clusters_by_id = {
        str(cluster.get("cluster_id") or "").strip(): cluster
        for cluster in merged_output.get("clusters") or []
        if isinstance(cluster, dict) and str(cluster.get("cluster_id") or "").strip()
    }
    changed = False
    for pair in pairs:
        cluster_id = str(pair.get("cluster_id") or "").strip()
        object_id = str(pair.get("object_id") or "").strip()
        if not cluster_id or not object_id:
            continue
        cluster = clusters_by_id.get(cluster_id)
        if not isinstance(cluster, dict):
            continue
        if not _secondary_daylight_quality_target_can_relax(
            cluster=cluster,
            object_id=object_id,
        ):
            continue
        object_program = _as_mutable_mapping(cluster.get("object_program"))
        specs = _as_mutable_mapping(object_program.get("object_specs_by_id"))
        spec = _as_mutable_mapping(specs.get(object_id))
        if not spec:
            continue
        if spec.get("daylight_quality_relaxed") is not True:
            spec["daylight_quality_relaxed"] = True
            spec["daylight_quality_relaxation_reason"] = (
                "secondary_daylight_quality_conflict_after_strict_solver_failure"
            )
            changed = True
    return changed


def _secondary_daylight_quality_target_can_relax(
    *,
    cluster: Mapping[str, Any],
    object_id: str,
) -> bool:
    if _primary_front_access_object(cluster=cluster, object_id=object_id):
        return False
    object_tokens = {
        object_id.strip().lower(),
        object_id.strip().lower().replace("-", "_"),
    }
    object_program = _as_mapping(cluster.get("object_program"))
    specs = _as_mapping(object_program.get("object_specs_by_id"))
    spec = _as_mapping(specs.get(object_id))
    for key in ("category", "object_type", "base_object_id", "source_id"):
        value = str(spec.get(key) or "").strip().lower().replace("-", "_")
        if value:
            object_tokens.add(value)
    cluster_id = str(cluster.get("cluster_id") or "").strip().lower()
    relaxable_tokens = {
        "bookcase",
        "bookshelf",
        "cabinet",
        "dresser",
        "media",
        "media_shelf",
        "shelf",
        "storage",
        "tv_console",
        "wardrobe",
    }
    if object_tokens & relaxable_tokens:
        return True
    return "storage" in cluster_id or "media" in cluster_id


def _select_secondary_storage_rehome_wall_option(
    *,
    options: Mapping[str, Any],
    current_wall_side: str,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for wall_side, row in options.items():
        clean_wall = str(wall_side or "").strip().lower()
        if clean_wall not in {"top", "right", "bottom", "left"}:
            continue
        if clean_wall == current_wall_side:
            continue
        option = _as_mapping(row)
        if int(option.get("valid_candidate_count") or 0) <= 0:
            continue
        if int(option.get("front_inward_candidate_count") or 0) <= 0:
            continue
        if int(option.get("wall_parallel_candidate_count") or 0) <= 0:
            continue
        best_candidate = _as_mapping(
            option.get("best_rehome_candidate") or option.get("best_candidate")
        )
        if not bool(best_candidate.get("front_inward_match")):
            continue
        if not bool(best_candidate.get("long_axis_wall_parallel")):
            continue
        candidates.append(
            {
                "wall_side": clean_wall,
                "score": float(
                    option.get("best_rehome_score") or option.get("best_score") or 0.0
                ),
            }
        )
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["wall_side"])))
    return candidates[0]


def _current_wall_side_for_repair(pref: Mapping[str, Any]) -> str:
    for key in ("committed_wall_side", "preferred_wall_side", "zone_assignment"):
        wall_side = _wall_side_from_region_token(str(pref.get(key) or ""))
        if wall_side:
            return wall_side
    return ""


def _wall_side_from_region_token(value: str) -> str:
    lowered = value.strip().lower()
    for wall_side in ("top", "right", "bottom", "left"):
        if lowered.startswith(f"{wall_side}_") or f"{wall_side}_wall" in lowered:
            return wall_side
    return ""


def _wall_side_region_id(wall_side: str) -> str:
    if wall_side in {"top", "right", "bottom", "left"}:
        return f"{wall_side}_wall_zone"
    return ""


def _front_access_target_is_droppable_support(
    cluster: Mapping[str, Any],
    object_id: str,
) -> bool:
    if object_id in _optional_support_member_ids(cluster):
        return True
    if object_id in set(_clean_str_list(cluster.get("anchors"))):
        return False
    object_program = _as_mapping(cluster.get("object_program"))
    if object_id in set(_clean_str_list(object_program.get("protected_ids"))):
        return False
    if object_id in set(_clean_str_list(object_program.get("droppable_ids"))):
        return True
    specs = _as_mapping(object_program.get("object_specs_by_id"))
    spec = _as_mapping(specs.get(object_id))
    role = str(spec.get("role") or spec.get("priority") or "").strip().lower()
    return bool(spec.get("droppable")) and role not in {"anchor", "dominant_anchor"}


def _drop_front_access_optional_support(
    *,
    cluster: dict[str, Any],
    object_id: str,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
) -> bool:
    cluster_id = str(cluster.get("cluster_id") or "").strip()
    if not cluster_id:
        return False
    changed = _omit_optional_support_from_cluster_program(cluster, {object_id})
    object_program = _as_mutable_mapping(cluster.get("object_program"))
    rules = _as_mutable_mapping(cluster.get("cluster_rules"))
    policy = _as_mutable_mapping(rules.get("anchor_first_policy"))
    protected_ids = _clean_str_list(
        policy.get("protected_ids")
        if policy
        else object_program.get("protected_ids")
        if object_program
        else []
    )
    droppable_ids = _clean_str_list(
        policy.get("droppable_ids")
        if policy
        else object_program.get("droppable_ids")
        if object_program
        else []
    )
    next_protected = [item for item in protected_ids if item != object_id]
    next_droppable = _ordered_unique([*droppable_ids, object_id])
    if policy:
        if policy.get("protected_ids") != next_protected:
            policy["protected_ids"] = next_protected
            changed = True
        if policy.get("droppable_ids") != next_droppable:
            policy["droppable_ids"] = next_droppable
            changed = True
    _sync_object_solver_policy_drop_sets(
        relation_plan=relation_plan,
        concept=concept,
        cluster_id=cluster_id,
        protected_ids=next_protected,
        droppable_ids=next_droppable,
    )
    return True if next_droppable != droppable_ids else changed


def _secondary_front_access_target_can_relax(
    *,
    cluster: Mapping[str, Any],
    object_id: str,
    relation_plan: Mapping[str, Any],
) -> bool:
    if _primary_front_access_object(cluster=cluster, object_id=object_id):
        return False
    if _storage_like_object_id(object_id):
        return True
    object_program = _as_mapping(cluster.get("object_program"))
    specs = _as_mapping(object_program.get("object_specs_by_id"))
    spec = _as_mapping(specs.get(object_id))
    category = str(spec.get("category") or spec.get("object_type") or "").strip()
    if _storage_like_object_id(category):
        return True
    cluster_id = str(cluster.get("cluster_id") or "").strip()
    pref = _anchor_region_preferences_by_cluster(relation_plan).get(cluster_id, {})
    role_kind = str(pref.get("role_kind") or cluster.get("tag") or "").strip().lower()
    return role_kind == "storage"


def _primary_front_access_object(
    *,
    cluster: Mapping[str, Any],
    object_id: str,
) -> bool:
    lowered = object_id.strip().lower()
    if lowered in {"bed", "desk"}:
        return True
    cluster_id = str(cluster.get("cluster_id") or "").strip().lower()
    anchor_id = _dominant_anchor_id_for_repair(cluster)
    if object_id != anchor_id:
        return False
    if "sleep" in cluster_id:
        return True
    if "work" in cluster_id or "study" in cluster_id:
        return True
    return False


def _soften_front_access_requirement(
    *,
    cluster: dict[str, Any],
    object_id: str,
) -> bool:
    changed = False
    for container in (
        _as_mutable_mapping(cluster.get("object_program")),
        _as_mutable_mapping(cluster.get("cluster_rules")),
    ):
        rows = container.get("access_requirements")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not _front_access_requirement_matches(row, object_id):
                continue
            if row.get("required") is not False:
                row["required"] = False
                changed = True
            if row.get("relaxation_reason") != _FRONT_ACCESS_RELAXATION_REASON:
                row["relaxation_reason"] = _FRONT_ACCESS_RELAXATION_REASON
                changed = True
    return changed


def _front_access_requirement_matches(row: Mapping[str, Any], object_id: str) -> bool:
    candidate_id = str(
        row.get("id") or row.get("object_id") or row.get("target_id") or ""
    ).strip()
    if candidate_id != object_id:
        return False
    requirement_type = str(row.get("type") or row.get("mode") or "").strip().lower()
    return requirement_type in {
        "",
        "access",
        "requires_access",
        "front_access",
        "front_clearance",
    }


def _apply_storage_front_token_repair(
    *,
    merged_output: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    affected_clusters = set(_clean_str_list(diagnostics.get("offending_clusters")))
    valid_front_tokens = {"top", "bottom", "left", "right"}
    changed = False
    for cluster in merged_output.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if affected_clusters and cluster_id not in affected_clusters:
            continue
        anchor_id = _dominant_anchor_id_for_repair(cluster)
        if not anchor_id or not _storage_like_object_id(anchor_id):
            continue
        object_program = _as_mutable_mapping(cluster.get("object_program"))
        specs = _as_mutable_mapping(object_program.get("object_specs_by_id"))
        spec = _as_mutable_mapping(specs.get(anchor_id))
        if spec:
            front = str(spec.get("front") or "").strip().lower()
            if front not in valid_front_tokens:
                spec["front"] = "top"
                changed = True
        rules = _as_mutable_mapping(cluster.get("cluster_rules"))
        facing = rules.setdefault("facing", {})
        if isinstance(facing, dict):
            object_facing = facing.setdefault(anchor_id, {})
            if isinstance(object_facing, dict):
                front = str(object_facing.get("front") or "").strip().lower()
                if front not in valid_front_tokens:
                    object_facing["front"] = "top"
                    changed = True
    return changed


def _dominant_anchor_id_for_repair(cluster: Mapping[str, Any]) -> str:
    rules = _as_mapping(cluster.get("cluster_rules"))
    policy = _as_mapping(rules.get("anchor_first_policy"))
    for source in (
        policy.get("dominant_anchor_id"),
        _as_mapping(cluster.get("object_program")).get("dominant_anchor_id"),
    ):
        value = str(source or "").strip()
        if value:
            return value
    for source in (cluster.get("anchors"), policy.get("dominant_anchor_candidates")):
        values = _clean_str_list(source)
        if values:
            return values[0]
    return ""


def _storage_like_object_id(object_id: str) -> bool:
    lowered = object_id.lower()
    return any(
        token in lowered
        for token in ("wardrobe", "cabinet", "bookshelf", "shelf", "dresser", "storage")
    )


def _required_support_member_ids(cluster: Mapping[str, Any]) -> set[str]:
    object_program = _as_mapping(cluster.get("object_program"))
    members = _clean_str_list(cluster.get("members"))
    out = set(
        _clean_str_list(
            object_program.get("required_support_ids")
            or object_program.get("relaxable_required_ids")
        )
    )
    specs = object_program.get("object_specs_by_id")
    if isinstance(specs, Mapping):
        for object_id, spec in specs.items():
            if not isinstance(spec, Mapping):
                continue
            object_id_text = str(object_id or "").strip()
            if not object_id_text:
                continue
            if bool(spec.get("required_support")) or bool(
                spec.get("relaxable_required")
            ):
                out.add(object_id_text)
    decisions = cluster.get("decisions")
    if isinstance(decisions, list):
        for row in decisions:
            if not isinstance(row, Mapping) or not _decision_requires_solver_keep(row):
                continue
            object_type = _clean_str(
                row.get("object_type")
                or row.get("base_type")
                or row.get("canonical_type")
                or row.get("category")
            )
            if object_type is None or not _relaxable_required_support_type(object_type):
                continue
            out.update(_member_ids_matching_type(members, object_type))
    if not members:
        return out
    member_set = set(members)
    return {
        item
        for item in out
        if item in member_set
        or any(
            member.startswith(f"{item}_") or item.startswith(f"{member}_")
            for member in member_set
        )
    }


def _relaxable_required_support_type(object_type: str) -> bool:
    token = object_type.strip().lower()
    return token in {
        "bookcase",
        "bookshelf",
        "display_shelf",
        "floor_lamp",
        "media_shelf",
        "side_table",
        "storage_shelf",
    }


def _optional_support_member_ids(cluster: Mapping[str, Any]) -> set[str]:
    rules = _as_mapping(cluster.get("cluster_rules"))
    tier_hints = _as_mapping(rules.get("tier_count_hints"))
    optional_types = {
        str(item.get("object_type") or "").strip()
        for item in tier_hints.get("object_hints") or []
        if isinstance(item, Mapping) and _int_or_none(item.get("min_keep")) == 0
    }
    optional_types.discard("")
    if not optional_types:
        return set()
    member_ids = _clean_str_list(cluster.get("members"))
    required_ids = set(_clean_str_list(cluster.get("anchors")))
    required_ids.update(_required_member_ids_from_tier_hints(cluster))
    required_ids.update(_required_member_ids_from_decisions(cluster))
    required_ids.update(_required_member_ids_from_object_program(cluster))
    droppable: set[str] = set()
    for member_id in member_ids:
        if member_id in required_ids:
            continue
        if any(
            member_id == optional_type or member_id.startswith(f"{optional_type}_")
            for optional_type in optional_types
        ):
            droppable.add(member_id)
    return droppable


def _required_member_ids_from_decisions(cluster: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    members = _clean_str_list(cluster.get("members"))
    decisions = cluster.get("decisions")
    if not isinstance(decisions, list):
        return out
    for row in decisions:
        if not isinstance(row, Mapping):
            continue
        if not _decision_requires_solver_keep(row):
            continue
        object_type = _clean_str(
            row.get("object_type")
            or row.get("base_type")
            or row.get("canonical_type")
            or row.get("category")
        )
        if object_type is None:
            continue
        out.update(_member_ids_matching_type(members, object_type))
    return out


def _required_member_ids_from_object_program(cluster: Mapping[str, Any]) -> set[str]:
    object_program = _as_mapping(cluster.get("object_program"))
    if not object_program:
        return set()
    out = set(_clean_str_list(object_program.get("required_object_ids")))
    specs = object_program.get("object_specs_by_id")
    if isinstance(specs, Mapping):
        for object_id, spec in specs.items():
            if not isinstance(spec, Mapping) or not _decision_requires_solver_keep(
                spec
            ):
                continue
            object_id_text = _clean_str(object_id)
            if object_id_text is not None:
                out.add(object_id_text)
    members = set(_clean_str_list(cluster.get("members")))
    if not members:
        return out
    return {
        item
        for item in out
        if item in members
        or any(
            member.startswith(f"{item}_") or item.startswith(f"{member}_")
            for member in members
        )
    }


def _decision_requires_solver_keep(row: Mapping[str, Any]) -> bool:
    intent = str(row.get("request_contract_intent") or row.get("intent") or "").strip()
    return (
        intent in _HARD_REQUEST_CONTRACT_INTENTS
        or (_int_or_none(row.get("min_keep")) or 0) > 0
    )


def _member_ids_matching_type(members: list[str], object_type: str) -> set[str]:
    return {
        member_id
        for member_id in members
        if member_id == object_type or member_id.startswith(f"{object_type}_")
    }


def _required_member_ids_from_tier_hints(cluster: Mapping[str, Any]) -> set[str]:
    rules = _as_mapping(cluster.get("cluster_rules"))
    tier_hints = _as_mapping(rules.get("tier_count_hints"))
    required_types = {
        str(item.get("object_type") or "").strip()
        for item in tier_hints.get("object_hints") or []
        if isinstance(item, Mapping) and (_int_or_none(item.get("min_keep")) or 0) > 0
    }
    out: set[str] = set()
    for member_id in _clean_str_list(cluster.get("members")):
        if any(
            member_id == required_type or member_id.startswith(f"{required_type}_")
            for required_type in required_types
        ):
            out.add(member_id)
    return out


def _omit_optional_support_from_cluster_program(
    cluster: dict[str, Any],
    optional_ids: set[str],
) -> bool:
    if not optional_ids:
        return False
    changed = False
    object_program = _as_mutable_mapping(cluster.get("object_program"))
    if object_program:
        for key in ("placement_order", "members", "optional_object_ids"):
            current = _clean_str_list(object_program.get(key))
            if not current:
                continue
            next_values = [item for item in current if item not in optional_ids]
            if next_values != current:
                object_program[key] = next_values
                changed = True
        support_edges = object_program.get("support_edges")
        if isinstance(support_edges, list):
            next_edges = [
                item
                for item in support_edges
                if not (
                    isinstance(item, Mapping)
                    and str(item.get("object_id") or "").strip() in optional_ids
                )
            ]
            if next_edges != support_edges:
                object_program["support_edges"] = next_edges
                changed = True
        object_program["droppable_ids"] = _ordered_unique(
            [
                *_clean_str_list(object_program.get("droppable_ids")),
                *sorted(optional_ids),
            ]
        )
        object_program["protected_ids"] = [
            item
            for item in _clean_str_list(object_program.get("protected_ids"))
            if item not in optional_ids
        ]

    rules = _as_mutable_mapping(cluster.get("cluster_rules"))
    semantic_placements = rules.get("semantic_placements")
    if isinstance(semantic_placements, list):
        next_placements = [
            item
            for item in semantic_placements
            if not (
                isinstance(item, Mapping)
                and str(item.get("id") or item.get("object_id") or "").strip()
                in optional_ids
            )
        ]
        if next_placements != semantic_placements:
            rules["semantic_placements"] = next_placements
            changed = True

    if not object_program:
        current_members = _clean_str_list(cluster.get("members"))
        next_members = [item for item in current_members if item not in optional_ids]
        if next_members != current_members:
            cluster["members"] = next_members
            changed = True
    return changed


def _relation_plan_zone_pileup_clusters(relation_plan: Mapping[str, Any]) -> set[str]:
    anchor_prefs = _anchor_region_preferences_by_cluster(relation_plan)
    zone_groups: dict[str, list[str]] = {}
    wall_groups: dict[str, list[str]] = {}
    for cluster_id, pref in anchor_prefs.items():
        if not isinstance(pref, Mapping):
            continue
        zone = str(pref.get("zone_assignment") or "").strip()
        if zone:
            zone_groups.setdefault(zone, []).append(cluster_id)
        wall = str(pref.get("preferred_wall_side") or "").strip()
        if wall:
            wall_groups.setdefault(wall, []).append(cluster_id)

    pileup: set[str] = set()
    for group in [*zone_groups.values(), *wall_groups.values()]:
        if len(group) > 1:
            pileup.update(group)
    return pileup


def _anchor_region_preferences_by_cluster(
    relation_plan: Mapping[str, Any],
) -> dict[str, Any]:
    value = relation_plan.get("anchor_region_preferences_by_cluster")
    if not isinstance(value, dict):
        return {}
    return {
        str(cluster_id): pref
        for cluster_id, pref in value.items()
        if isinstance(cluster_id, str) and isinstance(pref, dict)
    }


def _sync_repair_cluster_updates(
    *,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    cluster_id: str,
    updates: Mapping[str, Any],
) -> None:
    for section_name in (
        "anchor_region_preferences_by_cluster",
        "anchor_layout_hints_by_cluster",
    ):
        section = (
            relation_plan.setdefault(section_name, {})
            if section_name == "anchor_region_preferences_by_cluster"
            else relation_plan.get(section_name)
        )
        if not isinstance(section, dict):
            continue
        pref = section.setdefault(cluster_id, {})
        if isinstance(pref, dict):
            pref.update(deepcopy(dict(updates)))

    macro_concept = _as_mutable_mapping(relation_plan.get("macro_concept"))
    _sync_cluster_zone_plan_updates(macro_concept, cluster_id, updates)
    _sync_cluster_zone_plan_updates(concept, cluster_id, updates)

    macro_constraints = _as_mutable_mapping(relation_plan.get("macro_constraints"))
    constraints = macro_constraints.get("anchor_region_constraints")
    if isinstance(constraints, list):
        for item in constraints:
            if isinstance(item, dict) and item.get("cluster_id") == cluster_id:
                item.update(deepcopy(dict(updates)))
    next_region = updates.get("zone_assignment")
    if isinstance(next_region, str) and next_region.strip():
        _sync_orientation_target_region(relation_plan, cluster_id, next_region.strip())
        _sync_orientation_target_region(macro_concept, cluster_id, next_region.strip())
        _sync_orientation_target_region(concept, cluster_id, next_region.strip())


def _sync_cluster_zone_plan_updates(
    payload: dict[str, Any],
    cluster_id: str,
    updates: Mapping[str, Any],
) -> None:
    cluster_zone_plan = payload.get("cluster_zone_plan")
    if not isinstance(cluster_zone_plan, list):
        return
    for row in cluster_zone_plan:
        if isinstance(row, dict) and row.get("cluster_id") == cluster_id:
            row.update(deepcopy(dict(updates)))


def _sync_orientation_target_region(
    payload: dict[str, Any],
    cluster_id: str,
    target_region: str,
) -> None:
    for key in ("object_orientations", "critical_object_orientations"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("cluster_id") != cluster_id:
                continue
            if "target_region" in row:
                row["target_region"] = target_region


def _sync_object_solver_policy_drop_sets(
    *,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    cluster_id: str,
    protected_ids: list[str],
    droppable_ids: list[str],
) -> None:
    for payload in (relation_plan, concept):
        object_policy = payload.setdefault("object_solver_policy", {})
        if not isinstance(object_policy, dict):
            continue
        protected_map = object_policy.setdefault("protected_ids_by_cluster", {})
        droppable_map = object_policy.setdefault("droppable_ids_by_cluster", {})
        if isinstance(protected_map, dict):
            protected_map[cluster_id] = list(protected_ids)
        if isinstance(droppable_map, dict):
            droppable_map[cluster_id] = list(droppable_ids)


def _sync_object_solver_policy_support_sets(
    *,
    relation_plan: dict[str, Any],
    concept: dict[str, Any],
    cluster_id: str,
    required_support_ids: list[str],
) -> None:
    for payload in (relation_plan, concept):
        object_policy = payload.setdefault("object_solver_policy", {})
        if not isinstance(object_policy, dict):
            continue
        for key in (
            "required_support_ids_by_cluster",
            "relaxable_required_ids_by_cluster",
        ):
            value = object_policy.setdefault(key, {})
            if isinstance(value, dict):
                value[cluster_id] = list(required_support_ids)


def _append_concept_repair_note(
    relation_plan: dict[str, Any],
    repair_id: str,
    notes: list[str],
) -> None:
    existing = relation_plan.get("notes")
    note_list = existing if isinstance(existing, list) else []
    relation_plan["notes"] = _collect_unique_notes(
        note_list,
        [f"Deterministic concept repair attempt={repair_id}.", *notes],
    )


def _assigned_repair_regions(anchor_prefs: Mapping[str, Any]) -> set[str]:
    regions: set[str] = set()
    for pref in anchor_prefs.values():
        if isinstance(pref, Mapping):
            region = str(pref.get("zone_assignment") or "").strip()
            if region:
                regions.add(region)
    return regions


def _select_repair_region(
    pref: Mapping[str, Any],
    *,
    used_regions: set[str],
    functional: bool,
) -> str:
    current = str(pref.get("zone_assignment") or "").strip()
    forbidden = set(_clean_str_list(pref.get("forbidden_region_ids")))
    preferred_regions = _clean_str_list(pref.get("preferred_region_ids"))
    priority = list(_CONCEPT_REPAIR_FUNCTIONAL_REGION_PRIORITY)
    candidates = (
        [*priority, *preferred_regions]
        if functional
        else [*preferred_regions, *priority]
    )
    candidates = _ordered_unique(candidates)

    for allow_used in (False, True):
        for candidate in candidates:
            if not candidate or candidate == current:
                continue
            if candidate in forbidden or _region_blocked_for_repair(candidate):
                continue
            if not allow_used and candidate in used_regions:
                continue
            return candidate
    return ""


def _region_blocked_for_repair(region_id: str) -> bool:
    lowered = region_id.lower()
    if lowered.startswith("window_") and lowered.endswith("_daylight"):
        return True
    return any(token in lowered for token in _CONCEPT_REPAIR_BLOCKED_REGION_TOKENS)


def _prepend_unique_region(value: Any, region_id: str) -> list[str]:
    return _ordered_unique([region_id, *_clean_str_list(value)])


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _clean_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _clean_str(value: Any) -> str | None:
    clean = str(value or "").strip()
    return clean or None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_mutable_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _annotate_object_level_layout_coverage(
    *,
    absolute_layout: dict[str, Any],
    merged_output: dict[str, Any],
    solver_output: dict[str, Any],
) -> dict[str, Any]:
    expected_cluster_ids = [
        str(cluster.get("cluster_id") or "")
        for cluster in (merged_output.get("clusters") or [])
        if isinstance(cluster, dict) and str(cluster.get("cluster_id") or "").strip()
    ]
    coverage = (
        absolute_layout.get("coverage")
        if isinstance(absolute_layout.get("coverage"), dict)
        else {}
    )
    present_cluster_ids = (
        coverage.get("present_cluster_ids")
        if isinstance(coverage.get("present_cluster_ids"), list)
        else []
    )
    if not present_cluster_ids:
        present_cluster_ids = sorted(
            {
                str(row.get("cluster_id") or "")
                for row in (absolute_layout.get("clusters") or [])
                if isinstance(row, dict) and str(row.get("cluster_id") or "").strip()
            }
        )
    missing_cluster_ids = [
        cluster_id
        for cluster_id in expected_cluster_ids
        if cluster_id not in set(present_cluster_ids)
    ]
    hard_valid = bool(solver_output.get("hard_valid", False))
    geometry_valid = bool(solver_output.get("geometry_valid", hard_valid))
    complete = geometry_valid and not missing_cluster_ids
    gallery_eligible = complete and bool(solver_output.get("gallery_eligible", False))
    absolute_layout["coverage"] = {
        "expected_cluster_ids": expected_cluster_ids,
        "present_cluster_ids": present_cluster_ids,
        "missing_cluster_ids": missing_cluster_ids,
    }
    absolute_layout["geometry_valid"] = geometry_valid
    absolute_layout["complete"] = complete
    absolute_layout["gallery_eligible"] = gallery_eligible
    absolute_layout["coverage_ratio"] = float(
        len(present_cluster_ids) / max(1, len(expected_cluster_ids))
    )
    absolute_layout["missing_cluster_ids"] = missing_cluster_ids
    return absolute_layout


def _make_case_id(user_id: str) -> str:
    safe_user = re.sub(r"[^A-Za-z0-9_-]", "_", user_id or "user")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_user}_{ts}"


def _extract_room_type(
    input_payload: dict[str, Any], room_output: dict[str, Any]
) -> str:
    user_input = (
        input_payload.get("user_input") if isinstance(input_payload, dict) else None
    )
    if isinstance(user_input, dict):
        rt = user_input.get("room_type")
        if isinstance(rt, str) and rt:
            return rt
    meta = room_output.get("meta") if isinstance(room_output, dict) else None
    if isinstance(meta, dict):
        rt = meta.get("room_type")
        if isinstance(rt, str) and rt:
            return rt
    return "unknown"


def _extract_tenant_id_from_payload(input_payload: dict[str, Any]) -> str | None:
    tenant_id = input_payload.get("tenant_id")
    if isinstance(tenant_id, str) and tenant_id.strip():
        return tenant_id.strip()
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        tenant_id = user_input.get("tenant_id")
        if isinstance(tenant_id, str) and tenant_id.strip():
            return tenant_id.strip()
    return None


def _extract_inventory_catalog(
    input_payload: dict[str, Any],
) -> list[dict[str, Any]] | None:
    for key in ("inventory_catalog", "inventory", "inventory_items"):
        value = input_payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]

    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        for key in ("inventory_catalog", "inventory", "inventory_items"):
            value = user_input.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]

    constraints = input_payload.get("constraints")
    if isinstance(constraints, dict):
        value = constraints.get("inventory_catalog")
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return None


def _collect_unique_notes(*sources: Any) -> list[str]:
    notes: list[str] = []
    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            text = str(item).strip()
            if text and text not in notes:
                notes.append(text)
    return notes[:8]


def _variant_progress_message(label: str, current: int, total: int) -> str:
    safe_total = max(1, int(total))
    safe_current = min(max(0, int(current)), safe_total)
    return f"{label} {safe_current}/{safe_total}"


def _style_variant_user_context(
    user_context_json: dict[str, Any],
    *,
    variant_index: int,
) -> dict[str, Any]:
    palette_notes = [
        "Palette direction: keep the requested style, but lean warmer and slightly richer in contrast.",
        "Palette direction: keep the requested style, but lean cooler and calmer with balanced neutrals.",
        "Palette direction: keep the requested style, but lean earthier with grounded depth and darker anchor tones.",
        "Palette direction: keep the requested style, but lean lighter, softer, and more airy.",
        "Palette direction: keep the requested style, but use a bolder accent contrast while staying cohesive.",
    ]
    context = deepcopy(user_context_json)
    palette_note = palette_notes[(variant_index - 1) % len(palette_notes)]
    notes = context.get("notes")
    merged_notes = (
        [str(item).strip() for item in notes if str(item).strip()]
        if isinstance(notes, list)
        else []
    )
    if palette_note not in merged_notes:
        merged_notes.append(palette_note)
    context["notes"] = merged_notes[:8]
    guidance_text = str(context.get("guidance_text") or "").strip()
    if palette_note not in guidance_text:
        context["guidance_text"] = (
            f"{guidance_text}\n{palette_note}".strip()
            if guidance_text
            else palette_note
        )
    return context


def _solver_parallel_limit(target_final_count: int) -> int:
    return max(1, min(int(target_final_count), 2))


def _solver_workers_per_task(
    solver: MacroClusterSolver,
    *,
    parallel_tasks: int,
) -> int:
    base_workers = max(1, int(solver.num_workers))
    if parallel_tasks <= 1:
        return base_workers
    return max(1, min(4, base_workers // parallel_tasks))


def _solver_with_workers(
    solver: MacroClusterSolver, *, num_workers: int
) -> MacroClusterSolver:
    return MacroClusterSolver(
        max_rounds=int(solver.max_rounds),
        num_workers=max(1, int(num_workers)),
    )


def _write_bundle_stage_artifacts(paths: CasePaths, bundle: dict[str, Any]) -> None:
    cluster_output = bundle.get("cluster_output")
    if isinstance(cluster_output, dict):
        _write_module_json(
            paths,
            "seed_relation_planner",
            cluster_output,
            legacy_path=paths.seed_relation_planner_legacy,
        )

    tier_output = bundle.get("tier_output")
    if isinstance(tier_output, dict):
        _write_module_json(
            paths,
            "tier_count_director",
            tier_output,
            legacy_path=paths.tier_count,
        )

    merged_output = bundle.get("merged_output")
    if isinstance(merged_output, dict):
        _write_module_json(
            paths,
            "cluster_output_merger",
            merged_output,
            legacy_path=paths.cluster_merged,
        )

    relation_plan = bundle.get("relation_plan")
    if isinstance(relation_plan, dict):
        _write_module_json(
            paths,
            "seed_concept_relation_plan",
            relation_plan,
            legacy_path=paths.cluster_relation_plan,
        )
        _write_module_json(paths, "seed_concept_relation_plan", relation_plan)

    solver_output = bundle.get("solver_output")
    if isinstance(solver_output, dict):
        _write_module_json(
            paths,
            "macro_cluster_solver",
            solver_output,
            legacy_path=paths.cluster_solver,
        )

    dropped_inventory = bundle.get("dropped_inventory_by_cluster")
    if isinstance(dropped_inventory, dict):
        _write_module_json(
            paths,
            "macro_cluster_solver_dropped_inventory",
            dropped_inventory_payload(dropped_inventory),
            legacy_path=paths.solver_dropped_inventory,
        )


def _build_layout_variants_payload_from_final_candidates(
    candidates: list[dict[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    payload_variants: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        absolute_layout = candidate.get("absolute_layout")
        styled_result = candidate.get("styled_result")
        if not isinstance(absolute_layout, dict) or not isinstance(styled_result, dict):
            continue
        enriched_styled_result = enrich_styled_output_orientations(
            styled_payload=styled_result,
            absolute_layout_payload=absolute_layout,
        )
        payload_variants.append(
            {
                "variant_id": f"variant_{index}",
                "label": f"Option {index}",
                "source": str(candidate.get("source") or f"loop_variant:{index}"),
                "state_signature": str(
                    candidate.get("state_signature")
                    or candidate.get("layout_signature")
                    or ""
                ),
                "macro_layout_signature": str(
                    candidate.get("macro_layout_signature") or ""
                ),
                "planned_material_macro_signature": str(
                    candidate.get("planned_material_macro_signature") or ""
                ),
                "actual_macro_layout_signature": str(
                    candidate.get("actual_macro_layout_signature") or ""
                ),
                "core_anchor_signature": str(
                    candidate.get("core_anchor_signature") or ""
                ),
                "gallery_selection_mode": str(
                    candidate.get("gallery_selection_mode") or ""
                ),
                "reason": str(
                    candidate.get("reason")
                    or "Feasible loop variant finalized through planner and phase-2."
                ),
                "layout_score": int(candidate.get("layout_score") or 0),
                "hard_valid": bool(candidate.get("hard_valid", True)),
                "complete": bool(candidate.get("complete", False)),
                "gallery_eligible": bool(candidate.get("gallery_eligible", False)),
                "coverage_ratio": float(candidate.get("coverage_ratio") or 0.0),
                "request_coverage_score": int(
                    candidate.get("request_coverage_score") or 0
                ),
                "missing_cluster_ids": deepcopy(
                    candidate.get("missing_cluster_ids") or []
                ),
                "notes": deepcopy(candidate.get("notes") or []),
                "absolute_layout": deepcopy(absolute_layout),
                "styled_result": enriched_styled_result,
            }
        )

    return {
        "status": status,
        "selected_variant_id": payload_variants[0]["variant_id"]
        if payload_variants
        else None,
        "variants": payload_variants,
    }


def _write_partial_layout_variants_artifact(
    *,
    paths: CasePaths,
    final_candidates: list[dict[str, Any]],
    target_final_count: int,
    ablation_mode: AblationMode,
) -> None:
    if not final_candidates:
        return

    partial_candidates = final_candidates[: max(1, int(target_final_count))]
    selected = _select_final_styling_candidates(
        final_candidates,
        max_variants=target_final_count,
        ablation_mode=ablation_mode,
    )
    layout_variants = _build_layout_variants_payload_from_final_candidates(
        partial_candidates,
        status="IN_PROGRESS",
    )
    layout_variants["is_partial"] = True
    payload_variants = layout_variants.get("variants")
    layout_variants["ready_count"] = (
        len(payload_variants) if isinstance(payload_variants, list) else 0
    )
    layout_variants["target_count"] = target_final_count
    layout_variants["selection_summary"] = build_final_gallery_selection_summary(
        candidates=final_candidates,
        selected=selected,
        requested_count=target_final_count,
    )
    _write_module_json(
        paths,
        "layout_variants",
        layout_variants,
        legacy_path=paths.layout_variants,
    )


def _select_final_styling_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_variants: int,
    ablation_mode: AblationMode = "full",
) -> list[dict[str, Any]]:
    if not use_gallery_selector(ablation_mode):
        return candidates[: max(1, int(max_variants))]
    return select_distinct_final_gallery_candidates(
        candidates=candidates,
        max_variants=max_variants,
    )


def _refill_judged_variant_accessories(
    *,
    bundle: dict[str, Any],
    room_output: dict[str, Any],
    variant_index: int,
    ablation_mode: AblationMode,
) -> dict[str, Any]:
    if skip_accessory_refill(ablation_mode):
        return bundle
    absolute_layout = bundle.get("absolute_layout")
    cluster_output = bundle.get("cluster_output")
    dropped_inventory = bundle.get("dropped_inventory_by_cluster")
    if not isinstance(absolute_layout, dict):
        return bundle
    if not isinstance(cluster_output, dict):
        return bundle

    updated_layout, request_refill_summary = (
        ensure_requested_non_functional_contract_objects(
            absolute_layout=absolute_layout,
            room_output=room_output,
            cluster_output=cluster_output,
        )
    )
    working_bundle = bundle
    request_refill_count = int(request_refill_summary.get("added_count") or 0)
    if request_refill_count > 0:
        working_bundle = deepcopy(bundle)
        working_bundle["absolute_layout"] = updated_layout
        working_bundle["request_non_functional_refill_summary"] = request_refill_summary
        working_bundle["notes"] = _collect_unique_notes(
            working_bundle.get("notes"),
            [
                "Request Contract Refill added "
                f"{request_refill_count} non-blocking requested object(s) to final layout {variant_index}."
            ],
        )
        absolute_layout = updated_layout

    if not isinstance(dropped_inventory, dict) or not dropped_inventory:
        return working_bundle

    refill_policy = decor_refill_policy(extract_style_policy(cluster_output))
    missing_contract_types = missing_functional_contract_types(
        contract=request_contract_from_payload(cluster_output),
        objects=[
            row for row in absolute_layout.get("objects") or [] if isinstance(row, dict)
        ],
    )
    if missing_contract_types:
        refill_policy["max_refills_total"] = 0
        refill_policy["disabled_reason"] = "functional_request_contract_unmet"
        refill_policy["missing_request_object_types"] = missing_contract_types

    updated_layout, refill_summary = controlled_accessory_refill(
        room_output=room_output,
        absolute_layout=absolute_layout,
        cluster_output=cluster_output,
        dropped_inventory_by_cluster=dropped_inventory,
        refined_layout_solution=None,
        refill_policy=refill_policy,
        grid_mm=25,
    )
    updated_bundle = deepcopy(working_bundle)
    updated_bundle["absolute_layout"] = updated_layout
    updated_bundle["accessory_refill_summary"] = refill_summary
    refill_count = int(refill_summary.get("refill_count") or 0)
    if refill_count > 0:
        updated_bundle["notes"] = _collect_unique_notes(
            updated_bundle.get("notes"),
            [
                "Controlled Accessory Refill added "
                f"{refill_count} decor/accessory object(s) to final layout {variant_index}."
            ],
        )
    return updated_bundle


def _style_judged_variant_bundle(
    *,
    bundle: dict[str, Any],
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    manual_placements: list[dict[str, Any]] | None,
    variant_index: int,
    ablation_mode: AblationMode = "full",
) -> dict[str, Any] | None:
    try:
        absolute_layout = bundle.get("absolute_layout")
        if not isinstance(absolute_layout, dict):
            return None

        style_model_name = stylist_model_name_for_variant(variant_index)
        if use_stylist_llm(ablation_mode):
            style_plan = stylist.generate_style_plan(
                layout_json=absolute_layout,
                user_context_json=stylist_user_context_json,
                tenant_id=stylist_tenant_id,
                model_name=style_model_name,
            )
        else:
            style_plan = {}
            style_model_name = "deterministic_fallback"
        styled_result = stylist.apply_style_plan(
            layout_json=absolute_layout,
            user_context_json=stylist_user_context_json,
            tenant_id=stylist_tenant_id,
            style_plan=style_plan,
        )
        styled_result = _strip_raw_text(styled_result)
        styled_result = merge_manual_placements_into_styled_output(
            styled_result,
            manual_placements,
        )
        finalized_bundle = deepcopy(bundle)
        finalized_bundle["styled_result"] = styled_result
        finalized_bundle["source"] = str(
            bundle.get("source") or f"loop_variant:{variant_index}"
        )
        finalized_bundle["reason"] = str(
            bundle.get("reason")
            or f"Variant {variant_index} finalized through judge and stylist."
        )
        finalized_bundle["stylist_model_name"] = style_model_name
        return finalized_bundle
    except Exception as exc:
        logger.warning(
            "Skipping judged variant %s because styling failed: %s",
            int(variant_index),
            exc,
        )
        return None


def _write_canonical_branch_artifacts(
    *,
    paths: CasePaths,
    bundle: dict[str, Any],
    absolute_layout: dict[str, Any],
    styled_output: dict[str, Any],
) -> None:
    cluster_output = bundle.get("cluster_output")
    if isinstance(cluster_output, dict):
        _write_module_json(
            paths,
            "seed_relation_planner",
            cluster_output,
            legacy_path=paths.seed_relation_planner_legacy,
        )

    tier_output = bundle.get("tier_output")
    if isinstance(tier_output, dict):
        _write_module_json(
            paths,
            "tier_count_director",
            tier_output,
            legacy_path=paths.tier_count,
        )

    merged_output = bundle.get("merged_output")
    if isinstance(merged_output, dict):
        _write_module_json(
            paths,
            "cluster_output_merger",
            merged_output,
            legacy_path=paths.cluster_merged,
        )

    relation_plan = bundle.get("relation_plan")
    if isinstance(relation_plan, dict):
        _write_module_json(
            paths,
            "seed_concept_relation_plan",
            relation_plan,
            legacy_path=paths.cluster_relation_plan,
        )
        _write_module_json(paths, "seed_concept_relation_plan", relation_plan)

    solver_output = bundle.get("solver_output")
    if isinstance(solver_output, dict):
        _write_module_json(
            paths,
            "macro_cluster_solver",
            solver_output,
            legacy_path=paths.cluster_solver,
        )

    dropped_inventory = bundle.get("dropped_inventory_by_cluster")
    if isinstance(dropped_inventory, dict):
        _write_module_json(
            paths,
            "macro_cluster_solver_dropped_inventory",
            dropped_inventory_payload(dropped_inventory),
            legacy_path=paths.solver_dropped_inventory,
        )

    _write_module_json(
        paths,
        "absolute_layout",
        absolute_layout,
        legacy_path=paths.absolute_layout,
    )
    accessory_refill_summary = bundle.get("accessory_refill_summary")
    if isinstance(accessory_refill_summary, dict):
        _write_module_json(
            paths,
            "controlled_accessory_refill",
            accessory_refill_summary,
            legacy_path=paths.accessory_refill,
        )
    _write_module_json(paths, "stylist", styled_output, legacy_path=paths.stylist)


def _strip_raw_text(payload: Any) -> Any:
    if isinstance(payload, dict):
        payload.pop("raw_text", None)
        for key, value in list(payload.items()):
            payload[key] = _strip_raw_text(value)
        return payload
    if isinstance(payload, list):
        return [_strip_raw_text(item) for item in payload]
    return payload
