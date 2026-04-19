from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from agent.tool_call_parser import extract_tool_calls as parse_tool_calls
from prompt.tier_count import TIER_COUNT_DIRECTOR, TIER_COUNT_DIRECTOR_SYSTEM
from stylist.style_policy import extract_style_policy

logger = logging.getLogger(__name__)
_LAYOUT_SPEED_MODE_ENV = "TKNT_LAYOUT_SPEED_MODE"


def _layout_speed_mode() -> str:
    return str(os.getenv(_LAYOUT_SPEED_MODE_ENV) or "").strip().lower()


def _is_fast_layout_mode() -> bool:
    return _layout_speed_mode() in {"fast", "speed", "speed-first", "speed_first"}


def _get_tool_registry() -> dict[str, Any]:
    from tier_count.tools import TOOL_REGISTRY

    return TOOL_REGISTRY


def _get_tool_schemas() -> list[dict[str, Any]]:
    from tier_count.tools import TOOL_SCHEMAS

    return list(TOOL_SCHEMAS)


@dataclass(frozen=True)
class TierCountDirector:
    system_prompt: str = TIER_COUNT_DIRECTOR_SYSTEM
    prompt_template: str = TIER_COUNT_DIRECTOR

    def generate(
        self,
        *,
        description: str,
        special_notes: str,
        room_model_json: dict[str, Any],
        user_intent_json: dict[str, Any],
        clusters_json: dict[str, Any],
        size_profiles_json: dict[str, Any] | None = None,
        layout_failure_report_json: dict[str, Any] | None = None,
        max_steps: int = 20,
        access_clearance_ratio: float = 0.25,
    ) -> dict[str, Any]:
        clusters_json_clean = _strip_raw_text(
            _unwrap_payload(clusters_json, key="clusters")
        )

        tenant_id = ""
        if isinstance(user_intent_json, dict):
            tenant_id = str(user_intent_json.get("tenant_id") or "")
        if not tenant_id and isinstance(room_model_json, dict):
            tenant_id = str(room_model_json.get("tenant_id") or "")

        context = {
            "description": description,
            "room_model_json": room_model_json,
            "user_intent_json": user_intent_json,
            "special_notes": special_notes,
            "clusters_json": clusters_json_clean,
            "layout_failure_report_json": layout_failure_report_json,
            "tenant_id": tenant_id or None,
        }

        size_profiles_by_category: dict[str, Any] | None = None
        if isinstance(size_profiles_json, dict):
            sp = size_profiles_json.get("size_profiles_by_category")
            if isinstance(sp, dict):
                size_profiles_by_category = sp

        return _run_hardcoded_tier_count(
            context=context,
            max_steps=max_steps,
            size_profiles_by_category=size_profiles_by_category,
        )

    def tools(self) -> list[dict[str, Any]]:
        return []


PROFILE_CATEGORY_ALIASES: dict[str, str] = {
    "wardrobe_or_closet": "wardrobe",
    "book_shelf": "bookshelf",
    "desk_chair": "chair",
}

OPTIONAL_MEMBER_TOKENS = (
    "lamp",
    "basket",
    "bar_cart",
    "bean_bag",
    "pet_bed",
    "stool",
)

SECONDARY_MEMBER_TOKENS = (
    "nightstand",
    "side_table",
    "storage_cabinet",
    "bookshelf",
    "book_shelf",
    "dresser",
    "chair",
    "ottoman",
    "bench",
    "media_shelf",
)

LARGE_MEMBER_TOKENS = (
    "bed",
    "wardrobe",
    "closet",
    "sofa",
    "sectional",
    "tv_console",
    "fridge",
    "stove",
    "dishwasher",
    "sink",
    "pantry_cabinet",
    "kitchen_island",
)

MEDIUM_MEMBER_TOKENS = (
    "desk",
    "dresser",
    "storage_cabinet",
    "bookshelf",
    "book_shelf",
    "coffee_table",
    "console_table",
    "armchair",
    "tv_console",
    "kitchen_base_cabinet",
    "kitchen_tall_cabinet",
)

MINIMAL_NOTE_TOKENS = (
    "airy",
    "uncluttered",
    "minimal",
    "minimalist",
    "clean",
    "open",
    "compact",
    "simple",
)

GENEROUS_NOTE_TOKENS = (
    "cozy",
    "luxury",
    "luxurious",
    "ample",
    "generous",
    "full",
    "comfortable",
    "layered",
)


def _run_hardcoded_tier_count(
    *,
    context: dict[str, Any],
    max_steps: int,
    size_profiles_by_category: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = max_steps
    clusters_json = context.get("clusters_json")
    required_types = _extract_member_types(clusters_json)
    if not required_types:
        return {
            "status": "NEED_INFO",
            "assumptions": [
                "No cluster members were available for deterministic tier counting."
            ],
            "decisions": [],
            "global_notes": ["ClusterForge did not provide usable cluster members."],
        }

    members_by_cluster = _extract_members_by_cluster(clusters_json)
    anchors_by_cluster = _extract_anchors_by_cluster(clusters_json)
    droppable_clusters = _extract_droppable_clusters(clusters_json)

    furnishing_mode = _infer_furnishing_mode(context)
    available_area_m2 = _estimate_available_area_m2(context.get("room_model_json"))
    room_scale = _classify_room_scale(available_area_m2)
    room_affordances = _extract_room_affordances(context.get("room_model_json"))
    semantic_program = _extract_semantic_program(clusters_json)
    style_policy = extract_style_policy(clusters_json)
    room_type = _infer_room_type(context.get("room_model_json"), semantic_program)
    capacity_model = _compute_capacity_model(
        room_model_json=context.get("room_model_json"),
        available_area_m2=available_area_m2,
        room_scale=room_scale,
        furnishing_mode=furnishing_mode,
    )
    capacity_model = _apply_style_policy_to_capacity_model(
        capacity_model,
        style_policy=style_policy,
    )

    logger.info(
        "TierCount utility mode: available_area_m2=%.2f room_scale=%s furnishing_mode=%s",
        available_area_m2,
        room_scale,
        furnishing_mode,
    )

    size_profiles = _ensure_size_profiles(
        required_types=required_types,
        tenant_id=context.get("tenant_id"),
        existing=size_profiles_by_category,
    )

    bundles = _build_candidate_decision_set(
        clusters_json=clusters_json,
        semantic_program=semantic_program,
        anchors_by_cluster=anchors_by_cluster,
        droppable_clusters=droppable_clusters,
    )
    draft = _select_inventory_decision_program(
        bundles=bundles,
        room_type=room_type,
        room_scale=room_scale,
        furnishing_mode=furnishing_mode,
        capacity_model=capacity_model,
        size_profiles_by_category=size_profiles,
        semantic_program=semantic_program,
        style_policy=style_policy,
    )
    decisions = draft["decisions"]

    valid, detail = _validate_decisions(
        decisions,
        required_types,
        members_by_cluster=members_by_cluster,
        anchors_by_cluster=anchors_by_cluster,
        droppable_clusters=droppable_clusters,
    )
    if not valid:
        draft["status"] = "NEEDS_REVIEW"
        draft.setdefault("conflicts", []).append(detail)

    draft["assumptions"] = _build_deterministic_assumptions(
        context=context,
        furnishing_mode=furnishing_mode,
        available_area_m2=available_area_m2,
        room_affordances=room_affordances,
        style_policy=style_policy,
    )
    draft["global_notes"] = _build_deterministic_notes(
        context=context,
        furnishing_mode=furnishing_mode,
    )
    draft["budget_valid"] = draft["status"] == "OK"
    _attach_rep_dims(draft, size_profiles)
    return draft


def _extract_semantic_program(clusters_json: Any) -> dict[str, Any]:
    if not isinstance(clusters_json, dict):
        return {}
    semantic_program = clusters_json.get("semantic_layout_program")
    if isinstance(semantic_program, dict):
        return semantic_program
    if isinstance(clusters_json.get("active_clusters"), list):
        return clusters_json
    return {}


def _infer_room_type(room_model_json: Any, semantic_program: dict[str, Any]) -> str:
    if isinstance(room_model_json, dict):
        for key in ("room_type", "type"):
            value = room_model_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        room = room_model_json.get("room")
        if isinstance(room, dict):
            value = room.get("room_type") or room.get("type")
            if isinstance(value, str) and value.strip():
                return value.strip()
        meta = room_model_json.get("meta")
        if isinstance(meta, dict):
            value = meta.get("room_type") or meta.get("type")
            if isinstance(value, str) and value.strip():
                return value.strip()

    value = semantic_program.get("room_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "room"


def _compute_capacity_model(
    *,
    room_model_json: Any,
    available_area_m2: float,
    room_scale: str,
    furnishing_mode: str,
) -> dict[str, Any]:
    room_area_m2 = available_area_m2
    room_polygon: list[dict[str, int]] = []
    affordance_map: dict[str, Any] = {}

    if isinstance(room_model_json, dict):
        area_mm2 = room_model_json.get("area_mm2")
        if isinstance(area_mm2, (int, float)) and area_mm2 > 0:
            room_area_m2 = float(area_mm2) / 1_000_000.0

        room = room_model_json.get("room")
        if isinstance(room, dict):
            polygon = room.get("polygon_ccw")
            if isinstance(polygon, list):
                room_polygon = [p for p in polygon if isinstance(p, dict)]

        raw_affordance = room_model_json.get("affordance_map")
        if isinstance(raw_affordance, dict):
            affordance_map = raw_affordance

    perimeter_m = _polygon_perimeter_m(room_polygon)
    usable_wall_segments = _list_value(affordance_map.get("usable_wall_segments"))
    floating_zones = _list_value(affordance_map.get("floating_zone_candidates"))
    center_regions = _list_value(affordance_map.get("center_openness_regions"))
    corridors = _list_value(affordance_map.get("primary_circulation_corridors"))
    entry_zones = _list_value(affordance_map.get("entry_landing_zones"))

    wall_signal = (
        len(usable_wall_segments) if usable_wall_segments else perimeter_m / 2.5
    )
    floating_signal = len(floating_zones) if floating_zones else 1.0
    center_signal = len(center_regions) if center_regions else 1.0
    circulation_signal = len(corridors) if corridors else 1.0

    density_ratio = {"minimal": 0.30, "neutral": 0.38, "generous": 0.44}.get(
        furnishing_mode,
        0.38,
    )
    if room_scale == "small":
        density_ratio -= 0.04
    elif room_scale == "large":
        density_ratio += 0.04
    density_ratio = max(0.24, min(0.50, density_ratio))

    wall_capacity_m2 = max(room_area_m2 * 0.18, wall_signal * 0.75)
    floating_capacity_m2 = max(room_area_m2 * 0.16, floating_signal * 1.15)
    center_openness_budget_m2 = room_area_m2 * (0.26 + 0.03 * center_signal)
    circulation_budget_m2 = room_area_m2 * (0.18 + 0.02 * circulation_signal)
    clutter_budget_m2 = room_area_m2 * density_ratio

    return {
        "available_area_m2": room_area_m2,
        "target_density": "balanced",
        "density_ratio": density_ratio,
        "wall_capacity_m2": wall_capacity_m2,
        "floating_capacity_m2": floating_capacity_m2,
        "center_openness_budget_m2": center_openness_budget_m2,
        "circulation_budget_m2": circulation_budget_m2,
        "clutter_budget_m2": clutter_budget_m2,
        "entry_conflict_sensitivity": "high" if entry_zones else "moderate",
        "center_openness_weight": "high",
        "circulation_penalty_weight": "high",
        "semantic_core_preserve_weight": "very_high",
        "signals": {
            "usable_wall_segments": len(usable_wall_segments),
            "floating_zone_candidates": len(floating_zones),
            "center_openness_regions": len(center_regions),
            "primary_circulation_corridors": len(corridors),
            "entry_landing_zones": len(entry_zones),
        },
    }


def _apply_style_policy_to_capacity_model(
    capacity_model: dict[str, Any],
    *,
    style_policy: dict[str, Any],
) -> dict[str, Any]:
    if not style_policy:
        return capacity_model
    out = dict(capacity_model)
    layout_policy = _style_layout_policy(style_policy)
    weights = style_policy.get("policy_weights")
    weights = weights if isinstance(weights, dict) else {}
    density_multiplier = float(weights.get("density_multiplier") or 1.0)
    density_ratio = float(out.get("density_ratio") or 0.38) * density_multiplier
    if _style_level(layout_policy.get("center_openness_bias")) >= 3:
        density_ratio = min(density_ratio, 0.34)
        out["center_openness_weight"] = "very_high"
        out["circulation_penalty_weight"] = "very_high"
    if _style_level(layout_policy.get("clutter_tolerance")) <= 1:
        density_ratio = min(density_ratio, 0.32)
    if _style_level(layout_policy.get("clutter_tolerance")) >= 3:
        density_ratio = max(density_ratio, 0.40)
    density_ratio = max(0.22, min(0.52, density_ratio))
    room_area = max(0.1, float(out.get("available_area_m2") or 0.1))
    out["target_density"] = _style_target_density(style_policy)
    out["density_ratio"] = density_ratio
    out["clutter_budget_m2"] = room_area * density_ratio
    out["style_policy"] = {
        "style_name": style_policy.get("style_name"),
        "layout_policy": layout_policy,
    }
    return out


def _polygon_perimeter_m(points_mm: list[dict[str, int]]) -> float:
    if len(points_mm) < 2:
        return 0.0
    total_mm = 0.0
    for idx, point in enumerate(points_mm):
        nxt = points_mm[(idx + 1) % len(points_mm)]
        dx = float(nxt.get("x", 0)) - float(point.get("x", 0))
        dy = float(nxt.get("y", 0)) - float(point.get("y", 0))
        total_mm += (dx * dx + dy * dy) ** 0.5
    return total_mm / 1000.0


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _build_candidate_decision_set(
    *,
    clusters_json: Any,
    semantic_program: dict[str, Any],
    anchors_by_cluster: dict[str, set[str]],
    droppable_clusters: set[str],
) -> list[dict[str, Any]]:
    clusters = _cluster_list(clusters_json)
    semantic_clusters = _semantic_clusters_by_id(semantic_program)

    bundles: list[dict[str, Any]] = []
    for cluster in clusters[:8]:
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        semantic_cluster = semantic_clusters.get(cluster_id, {})
        cluster_bundles = _bundles_from_semantic_cluster(
            cluster=cluster,
            semantic_cluster=semantic_cluster,
            anchors=anchors_by_cluster.get(cluster_id, set()),
            droppable=cluster_id in droppable_clusters,
        )
        if not cluster_bundles:
            cluster_bundles = [
                _fallback_bundle_from_cluster(
                    cluster=cluster,
                    anchors=anchors_by_cluster.get(cluster_id, set()),
                    droppable=cluster_id in droppable_clusters,
                )
            ]
        bundles.extend(cluster_bundles[:4])

    return bundles


def _cluster_list(clusters_json: Any) -> list[dict[str, Any]]:
    if not isinstance(clusters_json, dict):
        return []
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return []
    return [cluster for cluster in clusters if isinstance(cluster, dict)]


def _semantic_clusters_by_id(
    semantic_program: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    active_clusters = semantic_program.get("active_clusters")
    if not isinstance(active_clusters, list):
        return out
    for row in active_clusters:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if isinstance(cluster_id, str) and cluster_id.strip():
            out[cluster_id] = row
    return out


def _bundles_from_semantic_cluster(
    *,
    cluster: dict[str, Any],
    semantic_cluster: dict[str, Any],
    anchors: set[str],
    droppable: bool,
) -> list[dict[str, Any]]:
    raw_bundles = semantic_cluster.get("required_bundles")
    if not isinstance(raw_bundles, list):
        return []

    members = _string_list_from_any(cluster.get("members"))
    member_lookup = _members_by_base_type(members)
    cluster_priority = str(semantic_cluster.get("priority") or "").strip().lower()
    bundles: list[dict[str, Any]] = []

    for raw_bundle in raw_bundles:
        if not isinstance(raw_bundle, dict):
            continue
        raw_objects = raw_bundle.get("objects")
        if not isinstance(raw_objects, list):
            continue

        objects: list[dict[str, Any]] = []
        for raw_object in raw_objects:
            if not isinstance(raw_object, dict):
                continue
            base_type = str(raw_object.get("object_type") or "").strip()
            if not base_type:
                continue
            matched_members = member_lookup.get(
                _profile_category_for_member(base_type),
                [],
            )
            for member in matched_members:
                role = str(raw_object.get("role") or "support")
                if droppable:
                    role = "decor_light"
                objects.append(
                    {
                        "object_type": member,
                        "base_type": _profile_category_for_member(member),
                        "role": role,
                        "required": bool(raw_object.get("required", False))
                        and not droppable,
                        "max_keep": _positive_int(
                            raw_object.get("max_keep"),
                            default=1,
                        ),
                    }
                )

        if not objects:
            continue
        bundle_id = str(
            raw_bundle.get("bundle_id") or f"{cluster.get('cluster_id')}_bundle"
        )
        bundles.append(
            {
                "cluster_id": str(cluster.get("cluster_id")),
                "bundle_id": bundle_id,
                "bundle_class": _bundle_class(cluster_priority, objects, droppable),
                "preserve_level": _preserve_level(
                    cluster_priority,
                    objects,
                    droppable,
                ),
                "objects": _dedupe_bundle_objects(objects, anchors),
                "cluster_priority": cluster_priority or "support",
                "droppable": droppable,
            }
        )

    return bundles


def _fallback_bundle_from_cluster(
    *,
    cluster: dict[str, Any],
    anchors: set[str],
    droppable: bool,
) -> dict[str, Any]:
    cluster_id = str(cluster.get("cluster_id") or "cluster")
    tag = str(cluster.get("tag") or "").strip().lower()
    objects: list[dict[str, Any]] = []
    for member in _string_list_from_any(cluster.get("members")):
        role = "decor_light" if droppable else _fallback_role(member, anchors)
        objects.append(
            {
                "object_type": member,
                "base_type": _profile_category_for_member(member),
                "role": role,
                "required": member in anchors and not droppable,
                "max_keep": _fallback_max_keep(member, tag),
            }
        )

    priority = "optional" if droppable else ("core" if anchors else "support")
    return {
        "cluster_id": cluster_id,
        "bundle_id": f"{cluster_id}_bundle",
        "bundle_class": _bundle_class(priority, objects, droppable),
        "preserve_level": _preserve_level(priority, objects, droppable),
        "objects": objects,
        "cluster_priority": priority,
        "droppable": droppable,
    }


def _members_by_base_type(members: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for member in members:
        out.setdefault(_profile_category_for_member(member), []).append(member)
    return out


def _dedupe_bundle_objects(
    objects: list[dict[str, Any]],
    anchors: set[str],
) -> list[dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for obj in objects:
        object_type = str(obj.get("object_type") or "")
        if not object_type:
            continue
        existing = by_type.get(object_type)
        if existing is None:
            by_type[object_type] = obj
            continue
        existing["required"] = bool(existing.get("required")) or bool(
            obj.get("required")
        )
        if object_type in anchors:
            existing["role"] = "dominant_anchor"
    return list(by_type.values())


def _fallback_role(member: str, anchors: set[str]) -> str:
    if member in anchors:
        return "dominant_anchor"
    member_key = _norm_key(member)
    if any(token in member_key for token in OPTIONAL_MEMBER_TOKENS):
        return "decor_light"
    if any(token in member_key for token in SECONDARY_MEMBER_TOKENS):
        return "support"
    if any(token in member_key for token in MEDIUM_MEMBER_TOKENS):
        return "workflow_anchor"
    return "support"


def _fallback_max_keep(member: str, cluster_tag: str) -> int:
    member_key = _norm_key(member)
    if member_key == "nightstand":
        return 2
    if "chair" in member_key and cluster_tag == "dining":
        return 4
    return 1


def _bundle_class(
    priority: str,
    objects: list[dict[str, Any]],
    droppable: bool,
) -> str:
    if droppable:
        return "decor_light"
    if priority == "core" or any(
        str(obj.get("role")) == "dominant_anchor" and bool(obj.get("required"))
        for obj in objects
    ):
        return "indispensable"
    if priority in {"support", "strong_support"} or any(
        bool(obj.get("required")) for obj in objects
    ):
        return "strong_support"
    return "optional"


def _preserve_level(
    priority: str,
    objects: list[dict[str, Any]],
    droppable: bool,
) -> str:
    if droppable:
        return "low"
    if priority == "core" or any(
        str(obj.get("role")) == "dominant_anchor" for obj in objects
    ):
        return "highest"
    if priority in {"support", "strong_support"}:
        return "high"
    return "medium"


def _select_inventory_decision_program(
    *,
    bundles: list[dict[str, Any]],
    room_type: str,
    room_scale: str,
    furnishing_mode: str,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    semantic_program: dict[str, Any],
    style_policy: dict[str, Any],
) -> dict[str, Any]:
    used_footprint_m2 = 0.0
    selected_count = 0
    dropped_types: list[str] = []
    conflicts: list[str] = []
    decisions: list[dict[str, Any]] = []
    cluster_decisions_by_id: dict[str, dict[str, Any]] = {}

    for bundle in sorted(bundles, key=_bundle_sort_key):
        cluster_id = str(bundle.get("cluster_id") or "cluster")
        bundle_decision = {
            "bundle_id": str(bundle.get("bundle_id") or f"{cluster_id}_bundle"),
            "preserve_level": str(bundle.get("preserve_level") or "medium"),
            "bundle_class": str(bundle.get("bundle_class") or "optional"),
            "objects": [],
        }
        cluster_decision = cluster_decisions_by_id.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "decision_status": "active",
                "selected_bundles": [],
            },
        )
        cluster_decision["selected_bundles"].append(bundle_decision)

        for obj in _bundle_objects(bundle):
            score = _score_object_utility(
                obj=obj,
                bundle=bundle,
                used_footprint_m2=used_footprint_m2,
                capacity_model=capacity_model,
                size_profiles_by_category=size_profiles_by_category,
                style_policy=style_policy,
            )
            quantity = _select_quantity(
                obj=obj,
                bundle=bundle,
                score=score,
                used_footprint_m2=used_footprint_m2,
                capacity_model=capacity_model,
                room_scale=room_scale,
                furnishing_mode=furnishing_mode,
            )
            size_tier = _select_size_tier(
                obj=obj,
                quantity=quantity,
                score=score,
                used_footprint_m2=used_footprint_m2,
                capacity_model=capacity_model,
                size_profiles_by_category=size_profiles_by_category,
                room_scale=room_scale,
                furnishing_mode=furnishing_mode,
            )
            footprint_m2 = _footprint_for_object(
                obj=obj,
                quantity=quantity,
                size_tier=size_tier or "S",
                size_profiles_by_category=size_profiles_by_category,
            )

            if quantity > 0:
                used_footprint_m2 += footprint_m2
                selected_count += quantity
            else:
                dropped_types.append(str(obj["object_type"]))

            role = str(obj.get("role") or "support")
            reason = _decision_reason(quantity=quantity, role=role, score=score)
            decision_object = {
                "object_type": str(obj["object_type"]),
                "quantity": quantity,
                "size_tier": size_tier if quantity > 0 else None,
                "role": role,
                "decision_reason": reason,
                "utility_score": round(float(score["total"]), 3),
                "utility_breakdown": {
                    key: round(float(value), 3) for key, value in score.items()
                },
            }
            bundle_decision["objects"].append(decision_object)

            decisions.append(
                {
                    "object_type": str(obj["object_type"]),
                    "category": str(obj["object_type"]),
                    "cluster_id": cluster_id,
                    "quantity": quantity,
                    "size_tier": size_tier or "S",
                    "priority": _legacy_priority(role, bundle),
                    "preserve_level": str(bundle.get("preserve_level") or "medium"),
                    "bundle_id": str(bundle.get("bundle_id") or ""),
                    "role": role,
                    "rationale": reason,
                    "utility_score": round(float(score["total"]), 3),
                }
            )

            if quantity > 0 and _is_core_object(obj, bundle) and score["total"] < 0:
                conflicts.append(
                    f"{obj['object_type']} is semantically required but has negative utility under capacity."
                )

    circulation_pressure = _circulation_pressure(
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )
    status = _decision_status(
        decisions=decisions,
        conflicts=conflicts,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )

    return {
        "status": status,
        "room_type": room_type,
        "cluster_decisions": list(cluster_decisions_by_id.values()),
        "global_density_policy": {
            "target_density": _style_target_density(style_policy),
            "clutter_budget_level": _style_clutter_budget_level(
                furnishing_mode,
                style_policy,
            ),
            "center_openness_preserved": circulation_pressure <= 0.72,
            "style_policy_applied": bool(style_policy),
        },
        "capacity_model": capacity_model,
        "decision_summary": {
            "selected_object_count": selected_count,
            "dropped_object_types": sorted(set(dropped_types)),
            "estimated_footprint_mm2": int(round(used_footprint_m2 * 1_000_000)),
            "estimated_circulation_pressure": round(circulation_pressure, 3),
        },
        "degradation_ready_order": _build_degradation_ready_order(decisions),
        "missing": list(semantic_program.get("missing") or []),
        "conflicts": conflicts,
        "confidence": _inventory_confidence(status, conflicts, decisions),
        "notes": [
            "Tier Count used deterministic utility, capacity, and bundle-preserving rules.",
            "No pose, macro placement, cluster composition, or LLM step was used.",
        ],
        "decisions": decisions,
    }


def _bundle_sort_key(bundle: dict[str, Any]) -> tuple[int, str]:
    rank = {
        "indispensable": 0,
        "strong_support": 1,
        "optional": 2,
        "decor_light": 3,
    }
    return (
        rank.get(str(bundle.get("bundle_class") or "optional"), 4),
        str(bundle.get("bundle_id") or ""),
    )


def _bundle_objects(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    objects = bundle.get("objects")
    if not isinstance(objects, list):
        return []
    return [obj for obj in objects if isinstance(obj, dict)]


def _score_object_utility(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    style_policy: dict[str, Any],
) -> dict[str, float]:
    role = str(obj.get("role") or "support")
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    required = bool(obj.get("required"))
    bundle_class = str(bundle.get("bundle_class") or "optional")

    functional = _functional_utility(role, base_type)
    semantic = _semantic_utility(
        role=role,
        required=required,
        bundle_class=bundle_class,
        preserve_level=str(bundle.get("preserve_level") or "medium"),
    )
    completeness = _completeness_utility(base_type, bundle)
    footprint_m2 = _footprint_for_object(
        obj=obj,
        quantity=1,
        size_tier="M",
        size_profiles_by_category=size_profiles_by_category,
    )
    roomfit = _roomfit_utility(
        base_type=base_type,
        footprint_m2=footprint_m2,
        capacity_model=capacity_model,
    )
    clutter = _clutter_penalty(
        footprint_m2=footprint_m2,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )
    circulation = _circulation_penalty(
        base_type=base_type,
        footprint_m2=footprint_m2,
        capacity_model=capacity_model,
    )
    redundancy = _redundancy_penalty(object_type, bundle)
    style_bias = _style_object_utility_bias(
        obj=obj,
        bundle=bundle,
        style_policy=style_policy,
    )
    space_surplus_bonus = _surplus_optional_support_bonus(
        obj=obj,
        bundle=bundle,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
        style_policy=style_policy,
    )
    total = (
        functional
        + semantic
        + completeness
        + roomfit
        + style_bias
        + space_surplus_bonus
        - clutter
        - circulation
        - redundancy
    )

    return {
        "function": functional,
        "semantic": semantic,
        "completeness": completeness,
        "roomfit": roomfit,
        "clutter_penalty": clutter,
        "circulation_penalty": circulation,
        "redundancy_penalty": redundancy,
        "style_policy_bias": style_bias,
        "space_surplus_bonus": space_surplus_bonus,
        "total": total,
    }


def _functional_utility(role: str, base_type: str) -> float:
    role_scores = {
        "dominant_anchor": 5.0,
        "workflow_anchor": 4.3,
        "support": 2.7,
        "secondary_support": 2.1,
        "decor_light": 0.9,
        "optional": 0.8,
    }
    score = role_scores.get(role, 2.0)
    if any(
        token in base_type
        for token in ("bed", "desk", "dining_table", "sofa", "tv_console")
    ):
        score += 0.6
    return score


def _semantic_utility(
    *,
    role: str,
    required: bool,
    bundle_class: str,
    preserve_level: str,
) -> float:
    score = 1.2 if required else 0.0
    if bundle_class == "indispensable":
        score += 2.0
    elif bundle_class == "strong_support":
        score += 1.0
    elif bundle_class == "decor_light":
        score -= 0.3
    if preserve_level == "highest":
        score += 1.0
    if role == "dominant_anchor":
        score += 0.8
    return score


def _completeness_utility(base_type: str, bundle: dict[str, Any]) -> float:
    bundle_types = {
        str(
            obj.get("base_type")
            or _profile_category_for_member(str(obj.get("object_type") or ""))
        )
        for obj in _bundle_objects(bundle)
    }
    if base_type == "nightstand" and "bed" in bundle_types:
        return 1.5
    if base_type in {"desk_chair", "chair"} and "desk" in bundle_types:
        return 1.3
    if base_type in {"coffee_table", "side_table"} and any(
        token in bundle_types for token in ("sofa", "sectional_sofa", "armchair")
    ):
        return 1.1
    if "chair" in base_type and "dining_table" in bundle_types:
        return 1.4
    return 0.45


def _roomfit_utility(
    *,
    base_type: str,
    footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    placement_mode = _placement_mode(base_type)
    capacity_key = (
        "wall_capacity_m2"
        if placement_mode == "wall_backed"
        else "floating_capacity_m2"
    )
    capacity = max(0.1, float(capacity_model.get(capacity_key) or 0.1))
    ratio = footprint_m2 / capacity
    if ratio <= 0.22:
        return 1.4
    if ratio <= 0.38:
        return 0.8
    if ratio <= 0.55:
        return 0.2
    return -0.8


def _clutter_penalty(
    *,
    footprint_m2: float,
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    after_ratio = (used_footprint_m2 + footprint_m2) / budget
    if after_ratio <= 0.72:
        return 0.15
    if after_ratio <= 1.0:
        return 0.8 + after_ratio * 0.5
    return 1.8 + (after_ratio - 1.0) * 4.0


def _circulation_penalty(
    *,
    base_type: str,
    footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    budget = max(0.1, float(capacity_model.get("circulation_budget_m2") or 0.1))
    pressure = footprint_m2 / budget
    multiplier = 1.4 if _placement_mode(base_type) == "floating" else 0.8
    if any(
        token in base_type for token in ("bed", "sofa", "sectional", "dining_table")
    ):
        multiplier += 0.35
    return pressure * multiplier


def _redundancy_penalty(object_type: str, bundle: dict[str, Any]) -> float:
    base_type = _profile_category_for_member(object_type)
    similar_count = sum(
        1
        for obj in _bundle_objects(bundle)
        if _profile_category_for_member(str(obj.get("object_type") or "")) == base_type
    )
    if similar_count <= 1:
        return 0.0
    if base_type in {"chair", "armchair", "nightstand"}:
        return 0.25
    return 0.65 * (similar_count - 1)


def _select_quantity(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    score: dict[str, float],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
) -> int:
    if _is_core_object(obj, bundle):
        return _core_quantity(obj, bundle, room_scale, furnishing_mode)

    total = float(score["total"])
    surplus_keep = _allow_surplus_optional_keep(
        obj=obj,
        bundle=bundle,
        score=score,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )
    if (
        str(bundle.get("bundle_class")) == "decor_light"
        and total < 3.4
        and not surplus_keep
    ):
        return 0
    if total < 2.2 and not surplus_keep:
        return 0

    footprint_m2 = _footprint_for_object(
        obj=obj,
        quantity=1,
        size_tier="M",
        size_profiles_by_category={},
    )
    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    if (
        used_footprint_m2 + footprint_m2 > clutter_budget * 1.04
        and total < 4.0
        and not surplus_keep
    ):
        return 0

    base_type = str(obj.get("base_type") or "")
    max_keep = _positive_int(obj.get("max_keep"), default=1)
    if base_type == "nightstand" and _bundle_contains_base_type(bundle, "bed"):
        if furnishing_mode != "minimal" and room_scale in {"medium", "large"}:
            return min(2, max_keep)
    return min(1, max_keep)


def _core_quantity(
    obj: dict[str, Any],
    bundle: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
) -> int:
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    max_keep = _positive_int(obj.get("max_keep"), default=1)
    if base_type == "nightstand" and _bundle_contains_base_type(bundle, "bed"):
        if furnishing_mode == "minimal" or room_scale == "small":
            return 1
        return min(2, max_keep)
    if "chair" in base_type and _bundle_contains_base_type(bundle, "dining_table"):
        if room_scale == "small":
            return min(2, max_keep)
        return min(4, max_keep)
    return min(1, max_keep)


def _select_size_tier(
    *,
    obj: dict[str, Any],
    quantity: int,
    score: dict[str, float],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
) -> str | None:
    if quantity <= 0:
        return None

    candidates = _available_tiers_for_object(obj, size_profiles_by_category)
    if not candidates:
        return "M"

    scored = [
        (
            _score_size_tier(
                obj=obj,
                tier=tier,
                quantity=quantity,
                utility_score=float(score["total"]),
                used_footprint_m2=used_footprint_m2,
                capacity_model=capacity_model,
                size_profiles_by_category=size_profiles_by_category,
                room_scale=room_scale,
                furnishing_mode=furnishing_mode,
            ),
            _tier_tiebreak(tier),
            tier,
        )
        for tier in candidates[:3]
    ]
    return max(scored, key=lambda row: (row[0], row[1]))[2]


def _score_size_tier(
    *,
    obj: dict[str, Any],
    tier: str,
    quantity: int,
    utility_score: float,
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
) -> float:
    role = str(obj.get("role") or "support")
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    footprint_m2 = _footprint_for_object(
        obj=obj,
        quantity=quantity,
        size_tier=tier,
        size_profiles_by_category=size_profiles_by_category,
    )
    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    after_ratio = (used_footprint_m2 + footprint_m2) / clutter_budget

    preferred = "M" if role in {"dominant_anchor", "workflow_anchor"} else "S"
    if (
        role == "dominant_anchor"
        and room_scale == "large"
        and furnishing_mode == "generous"
    ):
        preferred = "L"
    if base_type == "bed":
        preferred = "M"
        if (
            room_scale == "large"
            and furnishing_mode == "generous"
            and after_ratio < 0.72
        ):
            preferred = "L"

    dominance = 1.0 if tier == preferred else 0.35
    if role not in {"dominant_anchor", "workflow_anchor"} and tier == "L":
        dominance -= 0.9
    if tier == "L" and after_ratio > 0.82:
        dominance -= 1.2
    if tier == "M" and after_ratio > 0.95:
        dominance -= 0.7

    fit = 1.2 if after_ratio <= 0.72 else 0.3 if after_ratio <= 1.0 else -1.2
    naturalness = {"S": 0.18, "M": 0.12, "L": -0.18}.get(tier, 0.0)
    return utility_score * 0.12 + dominance + fit + naturalness


def _available_tiers_for_object(
    obj: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
) -> list[str]:
    profile = _profile_for_object(obj, size_profiles_by_category)
    rep_dims = profile.get("rep_dims_m") if isinstance(profile, dict) else None
    if not isinstance(rep_dims, dict):
        return ["S", "M", "L"]
    tiers = [tier for tier in ("S", "M", "L") if isinstance(rep_dims.get(tier), dict)]
    return tiers or ["S", "M", "L"]


def _tier_tiebreak(tier: str) -> int:
    return {"S": 2, "M": 1, "L": 0}.get(tier, 0)


def _profile_for_object(
    obj: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
) -> dict[str, Any]:
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    for key in (object_type, base_type, _profile_category_for_member(object_type)):
        profile = size_profiles_by_category.get(key)
        if isinstance(profile, dict):
            return profile
    generic = size_profiles_by_category.get("__generic__")
    return generic if isinstance(generic, dict) else {}


def _footprint_for_object(
    *,
    obj: dict[str, Any],
    quantity: int,
    size_tier: str,
    size_profiles_by_category: dict[str, Any],
) -> float:
    if quantity <= 0:
        return 0.0
    profile = _profile_for_object(obj, size_profiles_by_category)
    rep_dims = profile.get("rep_dims_m") if isinstance(profile, dict) else None
    rep = rep_dims.get(size_tier.upper()) if isinstance(rep_dims, dict) else None
    if isinstance(rep, dict):
        area = float(rep.get("A") or 0.0)
        if area <= 0:
            area = float(rep.get("L") or 0.0) * float(rep.get("W") or 0.0)
        if area > 0:
            return area * quantity

    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    fallback = {
        "bed": 3.6,
        "sofa": 2.1,
        "sectional_sofa": 3.4,
        "desk": 1.2,
        "dining_table": 1.8,
        "wardrobe": 1.3,
        "tv_console": 0.8,
        "coffee_table": 0.7,
        "nightstand": 0.25,
        "chair": 0.45,
        "armchair": 0.75,
        "floor_lamp": 0.12,
    }.get(base_type, 0.55)
    tier_mult = {"S": 0.75, "M": 1.0, "L": 1.35}.get(size_tier.upper(), 1.0)
    return fallback * tier_mult * quantity


def _placement_mode(base_type: str) -> str:
    if any(
        token in base_type
        for token in (
            "bed",
            "wardrobe",
            "closet",
            "bookshelf",
            "cabinet",
            "dresser",
            "tv_console",
            "console",
            "desk",
        )
    ):
        return "wall_backed"
    return "floating"


def _space_surplus_ratio(
    *,
    capacity_model: dict[str, Any],
    used_footprint_m2: float,
) -> float:
    room_area = max(0.1, float(capacity_model.get("available_area_m2") or 0.1))
    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    room_headroom = max(0.0, 1.0 - used_footprint_m2 / room_area)
    budget_headroom = max(0.0, 1.0 - used_footprint_m2 / clutter_budget)
    return max(0.0, min(1.0, 0.45 * room_headroom + 0.55 * budget_headroom))


def _surplus_optional_support_bonus(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    style_policy: dict[str, Any],
) -> float:
    role = str(obj.get("role") or "support")
    bundle_class = str(bundle.get("bundle_class") or "optional")
    if role in {"dominant_anchor", "workflow_anchor"}:
        return 0.0
    if bundle_class not in {
        "optional",
        "decor_light",
        "strong_support",
    } and role not in {"support", "secondary_support", "optional"}:
        return 0.0

    layout_policy = _style_layout_policy(style_policy)
    target_density = str(layout_policy.get("target_density") or "balanced")
    if target_density not in {"low_to_balanced", "balanced", "medium", "moderate_high"}:
        return 0.0

    surplus = _space_surplus_ratio(
        capacity_model=capacity_model,
        used_footprint_m2=used_footprint_m2,
    )
    if surplus < 0.38:
        return 0.0

    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    placement_mode = _placement_mode(base_type)
    role_bonus = 0.0
    if placement_mode == "wall_backed":
        role_bonus += 0.45
    if any(
        token in base_type
        for token in (
            "bookshelf",
            "cabinet",
            "shelf",
            "console",
            "shoe",
            "rack",
            "storage",
        )
    ):
        role_bonus += 0.35
    if role in {"support", "secondary_support"}:
        role_bonus += 0.18
    if bundle_class == "decor_light":
        role_bonus -= 0.10

    return max(0.0, min(1.1, role_bonus * min(1.0, surplus / 0.75)))


def _allow_surplus_optional_keep(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    score: dict[str, float],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> bool:
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    role = str(obj.get("role") or "support")
    bundle_class = str(bundle.get("bundle_class") or "optional")
    if role in {"dominant_anchor", "workflow_anchor"}:
        return False
    if bundle_class not in {
        "optional",
        "decor_light",
        "strong_support",
    } and role not in {"support", "secondary_support", "optional"}:
        return False
    if _placement_mode(base_type) != "wall_backed":
        return False
    surplus = _space_surplus_ratio(
        capacity_model=capacity_model, used_footprint_m2=used_footprint_m2
    )
    if surplus < 0.42:
        return False
    if float(score.get("circulation_penalty") or 0.0) > 0.28:
        return False
    if float(score.get("clutter_penalty") or 0.0) > 1.55:
        return False
    return float(score.get("total") or 0.0) >= 0.0


def _is_core_object(obj: dict[str, Any], bundle: dict[str, Any]) -> bool:
    if bool(bundle.get("droppable")):
        return False
    role = str(obj.get("role") or "")
    return bool(obj.get("required")) or role in {"dominant_anchor", "workflow_anchor"}


def _bundle_contains_base_type(bundle: dict[str, Any], base_type: str) -> bool:
    return any(
        str(
            obj.get("base_type")
            or _profile_category_for_member(str(obj.get("object_type") or ""))
        )
        == base_type
        for obj in _bundle_objects(bundle)
    )


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _string_list_from_any(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _legacy_priority(role: str, bundle: dict[str, Any]) -> str:
    _ = bundle
    if role == "dominant_anchor":
        return "anchor"
    if role == "workflow_anchor":
        return "primary"
    if role in {"decor_light", "optional"}:
        return "optional"
    return "secondary"


def _decision_reason(
    *,
    quantity: int,
    role: str,
    score: dict[str, float],
) -> str:
    if quantity <= 0:
        return "dropped by utility ranking under clutter and circulation budget"
    if role == "dominant_anchor":
        return "highest functional utility and semantic anchor value"
    if role == "workflow_anchor":
        return "required workflow utility under room-fit budget"
    if float(score.get("space_surplus_bonus") or 0.0) >= 0.35:
        return "kept because the room still has spatial surplus after core layout needs"
    if float(score["completeness"]) >= 1.0:
        return "high completeness utility under clutter budget"
    return "positive utility after room-fit and circulation penalties"


def _circulation_pressure(
    *,
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    room_area = max(0.1, float(capacity_model.get("available_area_m2") or 0.1))
    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    raw = 0.6 * (used_footprint_m2 / room_area) + 0.4 * (
        used_footprint_m2 / clutter_budget
    )
    return max(0.0, min(1.0, raw))


def _decision_status(
    *,
    decisions: list[dict[str, Any]],
    conflicts: list[str],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> str:
    if conflicts:
        return "NEEDS_REVIEW"
    core_kept = [
        decision
        for decision in decisions
        if str(decision.get("priority")) in {"anchor", "primary"}
        and int(decision.get("quantity") or 0) > 0
    ]
    if not core_kept:
        return "UNSAT"
    budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    if used_footprint_m2 > budget * 1.15:
        return "NEEDS_REVIEW"
    return "OK"


def _build_degradation_ready_order(decisions: list[dict[str, Any]]) -> list[str]:
    active = [
        decision
        for decision in decisions
        if isinstance(decision, dict) and int(decision.get("quantity") or 0) > 0
    ]
    drop_first = [
        str(row.get("role") or row.get("priority") or "support")
        for row in sorted(active, key=_degradation_sort_key)
        if str(row.get("priority")) in {"optional", "secondary"}
    ]
    shrink_first = [
        str(row.get("object_type"))
        for row in sorted(active, key=_degradation_sort_key)
        if str(row.get("size_tier")) in {"M", "L"}
    ]
    order: list[str] = []
    if drop_first:
        order.append(f"drop {drop_first[0]}")
    if shrink_first:
        order.append(f"shrink {shrink_first[0]} tier")
    order.extend(
        f"reduce {str(row.get('object_type'))} count"
        for row in sorted(active, key=_degradation_sort_key)
        if int(row.get("quantity") or 0) > 1
    )
    return _uniq(order)


def _degradation_sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    priority_rank = {"optional": 0, "secondary": 1, "primary": 2, "anchor": 3}
    priority = str(row.get("priority") or "secondary")
    score = float(row.get("utility_score") or 0.0)
    return (priority_rank.get(priority, 1), score, str(row.get("object_type") or ""))


def _clutter_budget_level(furnishing_mode: str) -> str:
    if furnishing_mode == "minimal":
        return "low"
    if furnishing_mode == "generous":
        return "moderate_high"
    return "moderate"


def _style_clutter_budget_level(
    furnishing_mode: str,
    style_policy: dict[str, Any],
) -> str:
    layout_policy = _style_layout_policy(style_policy)
    clutter_tolerance = _style_level(layout_policy.get("clutter_tolerance"))
    if clutter_tolerance <= 1:
        return "low"
    if clutter_tolerance >= 3:
        return "moderate_high"
    return _clutter_budget_level(furnishing_mode)


def _style_target_density(style_policy: dict[str, Any]) -> str:
    layout_policy = _style_layout_policy(style_policy)
    value = layout_policy.get("target_density")
    return str(value or "balanced")


def _style_object_utility_bias(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    style_policy: dict[str, Any],
) -> float:
    if not style_policy:
        return 0.0
    role = str(obj.get("role") or "support")
    bundle_class = str(bundle.get("bundle_class") or "optional")
    object_type = str(obj.get("object_type") or "").lower()
    layout_policy = _style_layout_policy(style_policy)
    weights = style_policy.get("policy_weights")
    weights = weights if isinstance(weights, dict) else {}
    bias = 0.0
    if role in {"decor", "decor_light"} or bundle_class == "decor_light":
        bias += float(weights.get("decor_utility_bias") or 0.0)
    if role in {"secondary_support", "support", "optional"}:
        bias += float(weights.get("optional_utility_bias") or 0.0)
    if _style_level(layout_policy.get("decor_tolerance")) <= 1 and any(
        token in object_type for token in ("art", "plant", "vase", "decor", "cushion")
    ):
        bias -= 0.35
    if (
        _style_level(layout_policy.get("clutter_tolerance")) <= 1
        and bundle_class == "optional"
    ):
        bias -= 0.35
    return bias


def _style_layout_policy(style_policy: dict[str, Any]) -> dict[str, Any]:
    layout_policy = style_policy.get("layout_policy")
    return dict(layout_policy) if isinstance(layout_policy, dict) else {}


def _style_level(value: Any) -> int:
    text = str(value or "").lower()
    if "very_high" in text or text == "high":
        return 4
    if "medium_high" in text:
        return 3
    if "low_to_medium" in text or "low_to_balanced" in text:
        return 1
    if "medium" in text or "balanced" in text:
        return 2
    if "low" in text:
        return 0
    return 2


def _inventory_confidence(
    status: str,
    conflicts: list[str],
    decisions: list[dict[str, Any]],
) -> float:
    if status == "UNSAT":
        return 0.35
    active_count = sum(int(row.get("quantity") or 0) for row in decisions)
    base = 0.86 if active_count else 0.45
    base -= min(0.22, len(conflicts) * 0.07)
    if status == "NEEDS_REVIEW":
        base -= 0.12
    return round(max(0.25, min(0.94, base)), 2)


def _uniq(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ensure_size_profiles(
    *,
    required_types: list[str],
    tenant_id: str | None,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    tool_registry = _get_tool_registry()

    if isinstance(existing, dict) and existing:
        profiles = dict(existing)
        _enrich_profiles_for_required_types(
            profiles=profiles,
            required_types=required_types,
        )
        return profiles

    categories = []
    for member in required_types:
        category = _profile_category_for_member(member)
        if category not in categories:
            categories.append(category)

    result = tool_registry["get_size_profiles"](
        categories=categories,
        tenant_id=tenant_id,
    )
    if _tool_output_has_error(result):
        raise RuntimeError(f"get_size_profiles failed: {_tool_error_text(result)}")

    profiles = result.get("size_profiles_by_category")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Deterministic tier count could not load size profiles.")
    _enrich_profiles_for_required_types(
        profiles=profiles,
        required_types=required_types,
    )
    return profiles


def _build_deterministic_assumptions(
    *,
    context: dict[str, Any],
    furnishing_mode: str,
    available_area_m2: float,
    room_affordances: dict[str, Any],
    style_policy: dict[str, Any],
) -> list[str]:
    assumptions = [
        "Deterministic tier-count policy applied from semantic bundles, room capacity, and utility scores."
    ]
    if furnishing_mode == "minimal":
        assumptions.append(
            "User notes suggest a clutter-light layout, so optional items start conservative."
        )
    elif furnishing_mode == "generous":
        assumptions.append(
            "User notes suggest a fuller furnishing plan, so anchors and primaries start slightly larger."
        )
    assumptions.append(f"Estimated available room area is {available_area_m2:.2f} m2.")
    fill_ratio = float(room_affordances.get("fill_ratio") or 1.0)
    if fill_ratio < 0.92:
        assumptions.append(
            "Room geometry is irregular enough that room-fit and circulation penalties are weighted more carefully."
        )
    if style_policy:
        assumptions.append(
            "Style policy adjusted density, clutter, and decor utility without overriding required functional anchors."
        )
    return assumptions


def _build_deterministic_notes(
    *,
    context: dict[str, Any],
    furnishing_mode: str,
) -> list[str]:
    notes = ["TierCountDirector used deterministic rules instead of an LLM."]
    style = _infer_style(context)
    if style:
        notes.append(f"Utility scoring considered the requested style: {style}.")
    notes.append(f"Initial furnishing mode: {furnishing_mode}.")
    return notes


def _infer_furnishing_mode(context: dict[str, Any]) -> str:
    text_parts: list[str] = []
    description = context.get("description")
    if isinstance(description, str) and description.strip():
        text_parts.append(description.lower())

    special_notes = context.get("special_notes")
    if isinstance(special_notes, str) and special_notes.strip():
        text_parts.append(special_notes.lower())

    user_intent = context.get("user_intent_json")
    if isinstance(user_intent, dict):
        user_input = user_intent.get("user_input")
        if isinstance(user_input, dict):
            for key in ("description", "special_description", "style"):
                value = user_input.get(key)
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.lower())

    text = " ".join(text_parts)
    minimal_score = sum(token in text for token in MINIMAL_NOTE_TOKENS)
    generous_score = sum(token in text for token in GENEROUS_NOTE_TOKENS)

    if minimal_score > generous_score:
        return "minimal"
    if generous_score > minimal_score:
        return "generous"
    return "neutral"


def _estimate_available_area_m2(room_model_json: Any) -> float:
    if not isinstance(room_model_json, dict):
        return 0.0

    room = room_model_json.get("room")
    room_polygon = room.get("polygon_ccw") if isinstance(room, dict) else None
    room_area = _polygon_area_m2(room_polygon if isinstance(room_polygon, list) else [])

    obstacles = room_model_json.get("obstacles")
    obstacle_area = 0.0
    if isinstance(obstacles, list):
        for obstacle in obstacles:
            if not isinstance(obstacle, dict):
                continue
            if obstacle.get("hard", True) is False:
                continue
            polygon = obstacle.get("polygon_ccw")
            if isinstance(polygon, list):
                obstacle_area += _polygon_area_m2(polygon)

    return max(0.0, room_area - obstacle_area)


def _extract_room_affordances(room_model_json: Any) -> dict[str, Any]:
    if not isinstance(room_model_json, dict):
        return {
            "bbox_width_mm": 0,
            "bbox_height_mm": 0,
            "bbox_area_m2": 0.0,
            "room_area_m2": 0.0,
            "fill_ratio": 1.0,
        }

    room = room_model_json.get("room")
    room_polygon = room.get("polygon_ccw") if isinstance(room, dict) else None
    polygon = room_polygon if isinstance(room_polygon, list) else []
    room_area_m2 = _polygon_area_m2(polygon)
    bbox = _polygon_bbox_mm(polygon)
    bbox_width_mm = max(0, int(bbox["max_x"]) - int(bbox["min_x"]))
    bbox_height_mm = max(0, int(bbox["max_y"]) - int(bbox["min_y"]))
    bbox_area_m2 = (bbox_width_mm * bbox_height_mm) / 1_000_000.0
    fill_ratio = room_area_m2 / bbox_area_m2 if bbox_area_m2 > 0 else 1.0
    return {
        "bbox_width_mm": bbox_width_mm,
        "bbox_height_mm": bbox_height_mm,
        "bbox_area_m2": bbox_area_m2,
        "room_area_m2": room_area_m2,
        "fill_ratio": max(0.0, min(1.0, fill_ratio)),
    }


def _polygon_area_m2(points_ccw_mm: list[dict[str, int]]) -> float:
    if len(points_ccw_mm) < 3:
        return 0.0
    total = 0
    for idx, point in enumerate(points_ccw_mm):
        if not isinstance(point, dict):
            continue
        nxt = points_ccw_mm[(idx + 1) % len(points_ccw_mm)]
        if not isinstance(nxt, dict):
            continue
        total += int(point.get("x", 0)) * int(nxt.get("y", 0)) - int(
            nxt.get("x", 0)
        ) * int(point.get("y", 0))
    return abs(total) / 2.0 / 1_000_000.0


def _polygon_bbox_mm(points_ccw_mm: list[dict[str, int]]) -> dict[str, int]:
    xs: list[int] = []
    ys: list[int] = []
    for point in points_ccw_mm:
        if not isinstance(point, dict):
            continue
        xs.append(int(point.get("x", 0)))
        ys.append(int(point.get("y", 0)))
    if not xs or not ys:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


def _classify_room_scale(area_m2: float) -> str:
    if area_m2 < 10.0:
        return "small"
    if area_m2 < 18.0:
        return "medium"
    return "large"


def _profile_category_for_member(member: str) -> str:
    normalized = _norm_key(member)
    canonical = re.sub(r"(?:_\d+)+$", "", normalized)
    return PROFILE_CATEGORY_ALIASES.get(canonical, canonical)


def _enrich_profiles_for_required_types(
    *,
    profiles: dict[str, Any],
    required_types: list[str],
) -> None:
    for member in required_types:
        if member in profiles:
            continue
        alias = _profile_category_for_member(member)
        if alias != member and isinstance(profiles.get(alias), dict):
            profiles[member] = profiles[alias]


def _select_next_budget_decisions(
    *,
    current_decisions: list[dict[str, Any]],
    budget_out: dict[str, Any],
) -> list[dict[str, Any]] | None:
    recommended_decisions = budget_out.get("recommended_decisions")
    if (
        isinstance(recommended_decisions, list)
        and recommended_decisions != current_decisions
    ):
        return recommended_decisions

    details = budget_out.get("decision_footprint_details")
    if not isinstance(details, list):
        return None

    detail_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in details:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        category = row.get("category")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        if not isinstance(category, str) or not category:
            continue
        detail_by_key[(cluster_id, category)] = row

    changed = False
    next_decisions: list[dict[str, Any]] = []
    for decision in current_decisions:
        if not isinstance(decision, dict):
            continue
        cluster_id = decision.get("cluster_id")
        category = decision.get("category")
        key = (
            str(cluster_id or ""),
            str(category or decision.get("object_type") or ""),
        )
        row = detail_by_key.get(key)
        if row is None:
            next_decisions.append(dict(decision))
            continue

        updated = dict(decision)
        recommended_quantity = row.get("recommended_quantity")
        recommended_size_tier = row.get("recommended_size_tier")

        if isinstance(
            recommended_quantity, int
        ) and recommended_quantity != updated.get("quantity"):
            updated["quantity"] = recommended_quantity
            changed = True

        if isinstance(recommended_size_tier, str):
            normalized_tier = recommended_size_tier.upper()
            if normalized_tier != updated.get("size_tier"):
                updated["size_tier"] = normalized_tier
                changed = True

        next_decisions.append(updated)

    return next_decisions if changed else None


def _norm_key(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


# ============================================================
# Prompt helpers
# ============================================================


def _json_block(obj: Any) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=True, indent=2)


def _build_prompt(
    *,
    template: str,
    description: str,
    special_notes: str,
    room_model_json: dict[str, Any],
    user_intent_json: dict[str, Any],
    clusters_json: dict[str, Any],
    size_profiles_json: dict[str, Any] | None,
    layout_failure_report_json: dict[str, Any] | None,
) -> str:
    mapping = {
        "DESCRIPTION": description or "",
        "SPECIAL_NOTES": special_notes or "",
        "ROOM_MODEL_JSON": _json_block(room_model_json),
        "USER_INTENT_JSON": _json_block(user_intent_json),
        "CLUSTERS_JSON": _json_block(clusters_json),
        "SIZE_PROFILES_JSON": _json_block(size_profiles_json),
        "LAYOUT_FAILURE_REPORT_JSON": _json_block(layout_failure_report_json),
    }
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def _unwrap_payload(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload
    parsed = payload.get("parsed") if isinstance(payload, dict) else None
    if isinstance(parsed, dict) and isinstance(parsed.get(key), list):
        return parsed
    raw = payload.get("raw") if isinstance(payload, dict) else None
    if isinstance(raw, dict) and isinstance(raw.get(key), list):
        return raw
    return payload


def _strip_raw_text(payload: Any) -> Any:
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("raw_text", None)
        for k, v in list(payload.items()):
            payload[k] = _strip_raw_text(v)
        return payload
    if isinstance(payload, list):
        return [_strip_raw_text(x) for x in payload]
    return payload


# ============================================================
# Core tool loop
# ============================================================


def _run_with_tools(
    *,
    messages: list[dict[str, Any]],
    context: dict[str, Any],
    max_steps: int,
    size_profiles_by_category: dict[str, Any] | None,
) -> dict[str, Any]:
    from clients.llm_client import get_llm_client

    client = get_llm_client()
    tool_registry = _get_tool_registry()

    clusters_json = context.get("clusters_json")
    required_types = _extract_member_types(clusters_json)
    members_by_cluster = _extract_members_by_cluster(clusters_json)
    anchors_by_cluster = _extract_anchors_by_cluster(clusters_json)
    droppable_clusters = _extract_droppable_clusters(clusters_json)

    last_draft: dict[str, Any] | None = None
    frozen_cluster_budget_limits_m2: dict[str, float] | None = None
    rescue_mode_used = False

    for step in range(max_steps):
        logger.info("TierCount step %s/%s", step + 1, max_steps)

        response = client.chat_completion(
            messages,
            model_key="primary",
            temperature=0.0,
            max_tokens=None,
            tools=_get_tool_schemas(),
        )

        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)
        content = getattr(message, "content", "") or ""

        draft = _try_parse_json_object(content)
        if isinstance(draft, dict):
            last_draft = draft

        # ----------------------------------------------------
        # TOOL CALLS
        # ----------------------------------------------------
        if tool_calls:
            logger.info(
                "Tool calls requested: %s",
                [(c.get("function", {}) or {}).get("name") for c in tool_calls],
            )

            prepared_calls: list[dict[str, Any]] = []
            preflight_error_message: str | None = None

            for idx, call in enumerate(tool_calls):
                call_id = call.get("id") or f"tool_{idx}"
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                args_text = fn.get("arguments") or "{}"
                args = _safe_json_loads(args_text)

                if not isinstance(name, str) or name not in tool_registry:
                    prepared_calls.append(
                        {
                            "call_id": call_id,
                            "name": str(name),
                            "args": args,
                            "kind": "unknown_tool",
                        }
                    )
                    continue

                if name == "get_size_profiles":
                    args = _coerce_get_size_profiles_args(
                        args=args,
                        context=context,
                        last_draft=last_draft,
                        required_types=required_types,
                    )

                    if (
                        not isinstance(args.get("categories"), list)
                        or not args["categories"]
                    ):
                        logger.warning(
                            "get_size_profiles has no categories after coercion. last_draft=%s required_types=%s",
                            isinstance(last_draft, dict),
                            required_types,
                        )
                        preflight_error_message = (
                            "Before calling get_size_profiles, provide categories or decisions. "
                            "You may include a JSON draft with decisions first, or call get_size_profiles "
                            f"with categories derived from cluster members: {required_types}"
                        )
                        break

                elif name == "estimate_budget":
                    args = _coerce_estimate_budget_args(
                        args=args,
                        context=context,
                        size_profiles_by_category=size_profiles_by_category,
                        last_draft=last_draft,
                        frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                    )

                    ok, detail = _validate_decisions(
                        args.get("decisions"),
                        required_types,
                        members_by_cluster=members_by_cluster,
                        anchors_by_cluster=anchors_by_cluster,
                        droppable_clusters=droppable_clusters,
                    )
                    if not ok:
                        logger.warning(
                            "estimate_budget preflight failed: detail=%s | args_keys=%s | has_last_draft=%s",
                            detail,
                            sorted(list(args.keys())),
                            isinstance(last_draft, dict),
                        )
                        preflight_error_message = (
                            "Before calling estimate_budget, you MUST provide a valid decisions list. "
                            "Requirement: exactly one decision per member within its own cluster, "
                            "quantity must be an integer >= 0, and each CORE cluster must keep at least one anchor "
                            "(quantity >= 1, using CLUSTERS_JSON.anchors when present). "
                            "DROPPABLE clusters (for example misc/decor/accent/accessory) may be reduced to all quantities = 0. "
                            f"Issue: {detail}. "
                            "You can either include decisions in the tool arguments, or output a JSON draft "
                            "with decisions first so the controller can reuse it."
                        )
                        break

                    if not isinstance(args.get("size_profiles_by_category"), dict):
                        cats = _extract_categories_from_decisions(
                            {"decisions": args.get("decisions")}
                        )
                        if not cats:
                            cats = required_types[:]

                        if cats:
                            logger.info(
                                "Auto-fetching size profiles before estimate_budget for categories=%s",
                                cats,
                            )
                            sp_out = tool_registry["get_size_profiles"](
                                categories=cats,
                                tenant_id=context.get("tenant_id"),
                            )
                            if _tool_output_has_error(sp_out):
                                raise RuntimeError(
                                    f"get_size_profiles failed during preflight: {_tool_error_text(sp_out)}"
                                )
                            sp = sp_out.get("size_profiles_by_category")
                            if isinstance(sp, dict):
                                size_profiles_by_category = sp
                                args["size_profiles_by_category"] = sp

                    if not isinstance(args.get("size_profiles_by_category"), dict):
                        logger.warning(
                            "estimate_budget still missing size_profiles_by_category after auto-fetch"
                        )
                        preflight_error_message = (
                            "Size profiles are still missing, so estimate_budget cannot run. "
                            "Call get_size_profiles first, or provide size_profiles_by_category."
                        )
                        break

                prepared_calls.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": args,
                        "kind": "normal",
                    }
                )

            if preflight_error_message is not None:
                messages.append({"role": "user", "content": preflight_error_message})
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            for item in prepared_calls:
                call_id = item["call_id"]
                name = item["name"]
                args = item["args"]
                kind = item["kind"]

                if kind == "unknown_tool":
                    tool_output = {
                        "error": "unknown_tool",
                        "tool": name,
                        "args": args,
                    }
                else:
                    tool_output = _safe_run_tool(name, args)

                logger.info(
                    "Tool %s output: %s",
                    name,
                    json.dumps(tool_output, ensure_ascii=True),
                )

                if _tool_output_has_error(tool_output):
                    raise RuntimeError(
                        f"{name} failed: {_tool_error_text(tool_output)}"
                    )

                if name == "get_size_profiles":
                    sp = tool_output.get("size_profiles_by_category")
                    if isinstance(sp, dict):
                        size_profiles_by_category = sp

                if (
                    name == "estimate_budget"
                    and frozen_cluster_budget_limits_m2 is None
                    and isinstance(tool_output.get("cluster_budget_limits_m2"), dict)
                    and not bool(tool_output.get("rescue_mode", False))
                ):
                    frozen_cluster_budget_limits_m2 = tool_output[
                        "cluster_budget_limits_m2"
                    ]

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": str(name),
                        "content": json.dumps(tool_output, ensure_ascii=True),
                    }
                )

                if name != "estimate_budget":
                    continue

                active_budget_out = tool_output
                active_decisions = args.get("decisions")
                if not isinstance(active_decisions, list):
                    active_decisions = []

                input_fit = bool(active_budget_out.get("input_decisions_fit", False))
                recommended_fit = bool(
                    active_budget_out.get("recommended_decisions_fit", False)
                )
                repair_exhausted = _tool_reports_repair_exhausted(active_budget_out)
                hard_unsat = _tool_reports_hard_unsat(active_budget_out)

                repair_hints = _preview_budget_repair_hints(active_budget_out)
                violations_after = active_budget_out.get("violations_after_fit") or []
                recommended_decisions = active_budget_out.get("recommended_decisions")

                if (
                    not input_fit
                    and not recommended_fit
                    and not rescue_mode_used
                    and repair_exhausted
                ):
                    rescue_args = dict(args)
                    if (
                        isinstance(recommended_decisions, list)
                        and recommended_decisions
                    ):
                        rescue_args["decisions"] = recommended_decisions
                    rescue_args["rescue_mode"] = True
                    rescue_args.pop("frozen_cluster_budget_limits_m2", None)

                    rescue_budget_out = _safe_run_tool("estimate_budget", rescue_args)
                    if _tool_output_has_error(rescue_budget_out):
                        raise RuntimeError(
                            f"estimate_budget rescue failed: {_tool_error_text(rescue_budget_out)}"
                        )

                    rescue_mode_used = True

                    logger.info(
                        "Rescue estimate_budget output: %s",
                        json.dumps(rescue_budget_out, ensure_ascii=True),
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"{call_id}_rescue",
                            "name": "estimate_budget",
                            "content": json.dumps(rescue_budget_out, ensure_ascii=True),
                        }
                    )

                    active_budget_out = rescue_budget_out
                    input_fit = bool(
                        active_budget_out.get("input_decisions_fit", False)
                    )
                    recommended_fit = bool(
                        active_budget_out.get("recommended_decisions_fit", False)
                    )
                    repair_exhausted = _tool_reports_repair_exhausted(active_budget_out)
                    hard_unsat = _tool_reports_hard_unsat(active_budget_out)

                    repair_hints = _preview_budget_repair_hints(active_budget_out)
                    violations_after = (
                        active_budget_out.get("violations_after_fit") or []
                    )
                    recommended_decisions = active_budget_out.get(
                        "recommended_decisions"
                    )

                if input_fit:
                    result = _build_ok_result(
                        decisions=active_decisions,
                        base_draft=last_draft,
                        size_profiles_by_category=size_profiles_by_category,
                        budget_mode="rescue"
                        if bool(active_budget_out.get("rescue_mode", False))
                        else None,
                    )
                    return result

                if recommended_fit and isinstance(recommended_decisions, list):
                    result = _build_ok_result(
                        decisions=recommended_decisions,
                        base_draft=last_draft,
                        size_profiles_by_category=size_profiles_by_category,
                        budget_mode="rescue"
                        if bool(active_budget_out.get("rescue_mode", False))
                        else None,
                    )
                    return result

                if hard_unsat:
                    result = _build_unsat_result(
                        base_draft=last_draft,
                        fallback_decisions=recommended_decisions
                        if isinstance(recommended_decisions, list)
                        else active_decisions,
                        budget_out=active_budget_out,
                        size_profiles_by_category=size_profiles_by_category,
                    )
                    return result

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Budget check still fails. "
                            "You must continue revising until input_decisions_fit=true. "
                            "Use the recommended_decisions below as the new base draft and make stronger minimal reductions if needed "
                            "(you may change multiple non-anchor items in one revision). "
                            f"Remaining violations: {json.dumps(_preview_budget_violations(violations_after), ensure_ascii=True)}. "
                            f"Recommended decisions: {json.dumps(recommended_decisions, ensure_ascii=True)}. "
                            f"Repair hints: {json.dumps(repair_hints, ensure_ascii=True)}"
                        ),
                    }
                )

            continue

        # ----------------------------------------------------
        # NO TOOL CALLS => EXPECT FINAL JSON
        # ----------------------------------------------------
        if not isinstance(content, str) or not content.strip():
            messages.append(
                {"role": "user", "content": "Return a JSON object only (no markdown)."}
            )
            continue

        logger.info("Assistant content length: %s", len(content))

        try:
            result = _parse_json(content)
            status = str(result.get("status") or "").upper()
        except ValueError:
            messages.append(
                {
                    "role": "user",
                    "content": "Your output must be valid JSON only (no markdown, no prose).",
                }
            )
            continue

        # IMPORTANT FIX:
        # Only return NEED_INFO if it truly has no valid decision set to continue with.
        if status == "NEED_INFO":
            ok_needinfo, _ = _validate_decisions(
                result.get("decisions"),
                required_types,
                members_by_cluster=members_by_cluster,
                anchors_by_cluster=anchors_by_cluster,
                droppable_clusters=droppable_clusters,
            )
            if not ok_needinfo:
                return result
            result["status"] = "OK"
            status = "OK"

        ok, detail = _validate_decisions(
            result.get("decisions"),
            required_types,
            members_by_cluster=members_by_cluster,
            anchors_by_cluster=anchors_by_cluster,
            droppable_clusters=droppable_clusters,
        )
        if not ok:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Decisions are required before final output. "
                        "You MUST output exactly one decision per member within its own cluster, "
                        "quantity must be an integer >= 0, and each CORE cluster must keep at least one anchor "
                        "(quantity >= 1, using CLUSTERS_JSON.anchors when present). "
                        "DROPPABLE clusters (for example misc/decor/accent/accessory) may be reduced to all quantities = 0. "
                        f"Issue: {detail}"
                    ),
                }
            )
            continue

        assistant_draft_added = False

        def ensure_assistant_draft_in_messages() -> None:
            nonlocal assistant_draft_added
            if assistant_draft_added:
                return
            assistant_content = (
                content
                if isinstance(content, str) and content.strip()
                else json.dumps(result, ensure_ascii=True)
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            )
            assistant_draft_added = True

        if not isinstance(size_profiles_by_category, dict):
            cats = _extract_categories_from_decisions(result)
            if not cats:
                cats = required_types[:]
            if cats:
                logger.info(
                    "Auto-fetching size profiles before finalization for categories=%s",
                    cats,
                )
                sp_out = tool_registry["get_size_profiles"](
                    categories=cats,
                    tenant_id=context.get("tenant_id"),
                )
                if _tool_output_has_error(sp_out):
                    raise RuntimeError(
                        f"get_size_profiles failed during finalization: {_tool_error_text(sp_out)}"
                    )
                sp = sp_out.get("size_profiles_by_category")
                if isinstance(sp, dict):
                    size_profiles_by_category = sp
                    logger.info(
                        "Auto tool get_size_profiles output: %s",
                        json.dumps(sp_out, ensure_ascii=True),
                    )
                    ensure_assistant_draft_in_messages()
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"auto_get_size_profiles_step_{step + 1}",
                            "name": "get_size_profiles",
                            "content": json.dumps(sp_out, ensure_ascii=True),
                        }
                    )

        if not isinstance(size_profiles_by_category, dict):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Unable to finalize because size profiles are missing. "
                        "Call get_size_profiles first or provide size_profiles_by_category."
                    ),
                }
            )
            continue

        if status == "UNSAT":
            budget_out_unsat = _run_budget_check_for_result(
                result=result,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                rescue_mode=False,
            )
            if _tool_output_has_error(budget_out_unsat):
                raise RuntimeError(
                    f"estimate_budget failed while verifying UNSAT: {_tool_error_text(budget_out_unsat)}"
                )
            logger.info("Auto-running estimate_budget to verify UNSAT draft.")
            logger.info(
                "UNSAT verification budget check output: %s",
                json.dumps(budget_out_unsat, ensure_ascii=True),
            )
            ensure_assistant_draft_in_messages()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"auto_estimate_budget_unsat_step_{step + 1}",
                    "name": "estimate_budget",
                    "content": json.dumps(budget_out_unsat, ensure_ascii=True),
                }
            )
            if _tool_reports_hard_unsat(budget_out_unsat):
                return _build_unsat_result(
                    base_draft=result,
                    fallback_decisions=result.get("decisions", []),
                    budget_out=budget_out_unsat,
                    size_profiles_by_category=size_profiles_by_category,
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Do not stop with status='UNSAT' yet. "
                        "Continue revising unless the budget tool explicitly confirms true infeasibility."
                    ),
                }
            )
            continue

        budget_out = _run_budget_check_for_result(
            result=result,
            context=context,
            size_profiles_by_category=size_profiles_by_category,
            frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
            rescue_mode=False,
        )
        if _tool_output_has_error(budget_out):
            raise RuntimeError(
                f"estimate_budget failed during finalization: {_tool_error_text(budget_out)}"
            )

        logger.info("Auto-running estimate_budget from JSON draft.")
        logger.info(
            "Finalization budget check output: %s",
            json.dumps(budget_out, ensure_ascii=True),
        )
        ensure_assistant_draft_in_messages()
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"auto_estimate_budget_step_{step + 1}",
                "name": "estimate_budget",
                "content": json.dumps(budget_out, ensure_ascii=True),
            }
        )

        if (
            frozen_cluster_budget_limits_m2 is None
            and isinstance(budget_out.get("cluster_budget_limits_m2"), dict)
            and not bool(budget_out.get("rescue_mode", False))
        ):
            frozen_cluster_budget_limits_m2 = budget_out["cluster_budget_limits_m2"]

        input_fit = bool(budget_out.get("input_decisions_fit", False))
        recommended_fit = bool(budget_out.get("recommended_decisions_fit", False))
        repair_exhausted = _tool_reports_repair_exhausted(budget_out)
        hard_unsat = _tool_reports_hard_unsat(budget_out)

        if input_fit:
            return _build_ok_result(
                decisions=result.get("decisions", []),
                base_draft=result,
                size_profiles_by_category=size_profiles_by_category,
                budget_mode=None,
            )

        recommended_decisions = budget_out.get("recommended_decisions")
        if recommended_fit and isinstance(recommended_decisions, list):
            return _build_ok_result(
                decisions=recommended_decisions,
                base_draft=result,
                size_profiles_by_category=size_profiles_by_category,
                budget_mode=None,
            )

        if repair_exhausted and not rescue_mode_used:
            rescue_budget_out = _run_budget_check_for_result(
                result=result,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=None,
                rescue_mode=True,
            )
            if _tool_output_has_error(rescue_budget_out):
                raise RuntimeError(
                    f"estimate_budget rescue failed during finalization: {_tool_error_text(rescue_budget_out)}"
                )

            rescue_mode_used = True
            logger.info(
                "Finalization rescue budget check output: %s",
                json.dumps(rescue_budget_out, ensure_ascii=True),
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"auto_estimate_budget_rescue_step_{step + 1}",
                    "name": "estimate_budget",
                    "content": json.dumps(rescue_budget_out, ensure_ascii=True),
                }
            )

            rescue_input_fit = bool(rescue_budget_out.get("input_decisions_fit", False))
            rescue_recommended_fit = bool(
                rescue_budget_out.get("recommended_decisions_fit", False)
            )
            rescue_hard_unsat = _tool_reports_hard_unsat(rescue_budget_out)
            rescue_recommended_decisions = rescue_budget_out.get(
                "recommended_decisions"
            )

            if rescue_input_fit:
                return _build_ok_result(
                    decisions=result.get("decisions", []),
                    base_draft=result,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if rescue_recommended_fit and isinstance(
                rescue_recommended_decisions, list
            ):
                return _build_ok_result(
                    decisions=rescue_recommended_decisions,
                    base_draft=result,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if rescue_hard_unsat:
                return _build_unsat_result(
                    base_draft=result,
                    fallback_decisions=rescue_recommended_decisions
                    if isinstance(rescue_recommended_decisions, list)
                    else result.get("decisions", []),
                    budget_out=rescue_budget_out,
                    size_profiles_by_category=size_profiles_by_category,
                )

            budget_out = rescue_budget_out
            recommended_decisions = budget_out.get("recommended_decisions")

        if hard_unsat:
            return _build_unsat_result(
                base_draft=result,
                fallback_decisions=recommended_decisions
                if isinstance(recommended_decisions, list)
                else result.get("decisions", []),
                budget_out=budget_out,
                size_profiles_by_category=size_profiles_by_category,
            )

        violations_after = budget_out.get("violations_after_fit") or []
        repair_hints = _preview_budget_repair_hints(budget_out)

        messages.append(
            {
                "role": "user",
                "content": (
                    "Your current decisions still fail budget. "
                    "Do not return UNSAT. Continue revising until input_decisions_fit=true. "
                    "Use the recommended_decisions below as the next base draft and make stronger minimal corrections if needed "
                    "(you may reduce several non-anchor items in one revision). "
                    f"Remaining violations: {json.dumps(_preview_budget_violations(violations_after), ensure_ascii=True)}. "
                    f"Recommended decisions: {json.dumps(recommended_decisions, ensure_ascii=True)}. "
                    f"Repair hints: {json.dumps(repair_hints, ensure_ascii=True)}"
                ),
            }
        )

    # --------------------------------------------------------
    # Timeout fallback
    # --------------------------------------------------------
    if isinstance(last_draft, dict) and isinstance(size_profiles_by_category, dict):
        try:
            budget_out = _run_budget_check_for_result(
                result=last_draft,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                rescue_mode=False,
            )
            if _tool_output_has_error(budget_out):
                raise RuntimeError(_tool_error_text(budget_out))

            logger.info(
                "Timeout fallback budget check output: %s",
                json.dumps(budget_out, ensure_ascii=True),
            )

            if bool(budget_out.get("input_decisions_fit", False)):
                return _build_ok_result(
                    decisions=last_draft.get("decisions", []),
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode=None,
                )

            recommended_decisions = budget_out.get("recommended_decisions")
            if bool(budget_out.get("recommended_decisions_fit", False)) and isinstance(
                recommended_decisions, list
            ):
                return _build_ok_result(
                    decisions=recommended_decisions,
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode=None,
                )

            rescue_budget_out = _run_budget_check_for_result(
                result=last_draft,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=None,
                rescue_mode=True,
            )
            if _tool_output_has_error(rescue_budget_out):
                raise RuntimeError(_tool_error_text(rescue_budget_out))

            logger.info(
                "Timeout fallback rescue budget check output: %s",
                json.dumps(rescue_budget_out, ensure_ascii=True),
            )

            if bool(rescue_budget_out.get("input_decisions_fit", False)):
                return _build_ok_result(
                    decisions=last_draft.get("decisions", []),
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            rescue_recommended_decisions = rescue_budget_out.get(
                "recommended_decisions"
            )
            if bool(
                rescue_budget_out.get("recommended_decisions_fit", False)
            ) and isinstance(rescue_recommended_decisions, list):
                return _build_ok_result(
                    decisions=rescue_recommended_decisions,
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if _tool_reports_hard_unsat(rescue_budget_out):
                return _build_unsat_result(
                    base_draft=last_draft,
                    fallback_decisions=rescue_recommended_decisions
                    if isinstance(rescue_recommended_decisions, list)
                    else last_draft.get("decisions", []),
                    budget_out=rescue_budget_out,
                    size_profiles_by_category=size_profiles_by_category,
                )
        except Exception as exc:
            logger.warning("Timeout fallback failed: %s", exc)

    raise TimeoutError(
        "TierCountDirector exceeded max_steps without reaching a budget-valid final JSON."
    )


# ============================================================
# Tool / args helpers
# ============================================================


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool_registry = _get_tool_registry()
    try:
        return tool_registry[name](**args)
    except TypeError as exc:
        return {
            "error": "invalid_tool_arguments",
            "tool": name,
            "message": str(exc),
            "args": args,
        }
    except Exception as exc:
        return {"error": "tool_failed", "tool": name, "message": str(exc), "args": args}


def _tool_output_has_error(tool_output: dict[str, Any]) -> bool:
    return isinstance(tool_output, dict) and isinstance(tool_output.get("error"), str)


def _tool_error_text(tool_output: dict[str, Any]) -> str:
    if not isinstance(tool_output, dict):
        return "unknown tool error"
    return json.dumps(tool_output, ensure_ascii=True)


def _coerce_get_size_profiles_args(
    *,
    args: dict[str, Any],
    context: dict[str, Any],
    last_draft: dict[str, Any] | None,
    required_types: list[str],
) -> dict[str, Any]:
    out = dict(args)

    # get_size_profiles expects furniture/object categories, not cluster tags (e.g. living/storage/misc).
    # Use CLUSTERS_JSON members as the primary source of truth when available.
    if required_types:
        cluster_tags: set[str] = set()
        clusters_json = context.get("clusters_json")
        if isinstance(clusters_json, dict):
            clusters = clusters_json.get("clusters")
            if isinstance(clusters, list):
                for cluster in clusters:
                    if not isinstance(cluster, dict):
                        continue
                    tag = cluster.get("tag")
                    if isinstance(tag, str) and tag.strip():
                        cluster_tags.add(tag.strip().lower())

        provided = out.get("categories")
        provided_cats: list[str] = []
        if isinstance(provided, list):
            for c in provided:
                if isinstance(c, str) and c.strip():
                    provided_cats.append(c.strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for c in provided_cats:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        required_set = {t.lower() for t in required_types if t}
        filtered: list[str] = []
        dropped: list[str] = []
        for c in deduped:
            ck = c.lower()
            if ck in cluster_tags and ck not in required_set:
                dropped.append(c)
                continue
            filtered.append(c)

        if dropped:
            logger.info(
                "Dropping cluster tags from get_size_profiles.categories: dropped=%s kept=%s",
                dropped,
                filtered,
            )

        if filtered and all(c.lower() in required_set for c in filtered):
            out["categories"] = filtered
        else:
            if filtered:
                logger.info(
                    "Ignoring get_size_profiles.categories not present in CLUSTERS_JSON.members: %s",
                    filtered,
                )
            out["categories"] = required_types[:]
    else:
        cats = out.get("categories")
        if not isinstance(cats, list) or not cats:
            cats = []
            if isinstance(last_draft, dict):
                cats = _extract_categories_from_decisions(last_draft)
            if not cats:
                cats = required_types[:]
            out["categories"] = cats

    if "tenant_id" not in out:
        out["tenant_id"] = context.get("tenant_id")

    return out


def _coerce_estimate_budget_args(
    *,
    args: dict[str, Any],
    context: dict[str, Any],
    size_profiles_by_category: dict[str, Any] | None,
    last_draft: dict[str, Any] | None,
    frozen_cluster_budget_limits_m2: dict[str, float] | None,
) -> dict[str, Any]:
    out = dict(args)

    if not isinstance(out.get("decisions"), list):
        if isinstance(last_draft, dict) and isinstance(
            last_draft.get("decisions"), list
        ):
            out["decisions"] = last_draft["decisions"]

    if "room_model" not in out:
        out["room_model"] = context.get("room_model_json", {})
    if isinstance(out.get("room_model"), str):
        parsed = _safe_json_loads(out["room_model"])
        if parsed:
            out["room_model"] = parsed

    if "size_profiles_by_category" not in out and isinstance(
        size_profiles_by_category, dict
    ):
        out["size_profiles_by_category"] = size_profiles_by_category
    if isinstance(out.get("size_profiles_by_category"), str):
        parsed = _safe_json_loads(out["size_profiles_by_category"])
        if parsed:
            out["size_profiles_by_category"] = parsed

    if "style" not in out:
        style = ""
        user_intent = context.get("user_intent_json") or {}
        if isinstance(user_intent, dict):
            ui = user_intent.get("user_input") or {}
            if isinstance(ui, dict):
                style = str(ui.get("style") or "")
        if not style:
            meta = (context.get("room_model_json") or {}).get("meta") or {}
            if isinstance(meta, dict):
                style = str(meta.get("style") or "")
        out["style"] = style

    if "user_notes" not in out:
        out["user_notes"] = str(context.get("special_notes") or "")

    if "clusters_json" not in out:
        out["clusters_json"] = context.get("clusters_json")
    if isinstance(out.get("clusters_json"), str):
        parsed = _safe_json_loads(out["clusters_json"])
        if parsed:
            out["clusters_json"] = parsed

    rescue_mode = bool(out.get("rescue_mode", False))
    if (
        "frozen_cluster_budget_limits_m2" not in out
        and isinstance(frozen_cluster_budget_limits_m2, dict)
        and not rescue_mode
    ):
        out["frozen_cluster_budget_limits_m2"] = frozen_cluster_budget_limits_m2

    return out


def _run_budget_check_for_result(
    *,
    result: dict[str, Any],
    context: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    frozen_cluster_budget_limits_m2: dict[str, float] | None,
    rescue_mode: bool,
) -> dict[str, Any]:
    args = {
        "room_model": context.get("room_model_json", {}),
        "decisions": result.get("decisions", []),
        "size_profiles_by_category": size_profiles_by_category,
        "style": _infer_style(context),
        "user_notes": str(context.get("special_notes") or ""),
        "clusters_json": context.get("clusters_json"),
        "rescue_mode": rescue_mode,
    }
    if frozen_cluster_budget_limits_m2 is not None and not rescue_mode:
        args["frozen_cluster_budget_limits_m2"] = frozen_cluster_budget_limits_m2
    return _safe_run_tool("estimate_budget", args)


def _infer_style(context: dict[str, Any]) -> str:
    user_intent = context.get("user_intent_json") or {}
    if isinstance(user_intent, dict):
        ui = user_intent.get("user_input") or {}
        if isinstance(ui, dict):
            style = str(ui.get("style") or "")
            if style:
                return style

    room_model = context.get("room_model_json") or {}
    if isinstance(room_model, dict):
        meta = room_model.get("meta") or {}
        if isinstance(meta, dict):
            style = str(meta.get("style") or "")
            if style:
                return style

    return ""


# ============================================================
# Parsing helpers
# ============================================================


def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _extract_tool_calls(message: object) -> list[dict[str, Any]]:
    return parse_tool_calls(message)


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(_extract_json_object(_coerce_json_text(text)))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("TierCountDirector returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("TierCountDirector response must be a JSON object")
    return payload


def _coerce_json_text(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    parts = text.split("```")
    for idx in range(1, len(parts), 2):
        candidate = parts[idx].strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip("\n").strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return text


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


# ============================================================
# Draft / result helpers
# ============================================================


def _merge_recommended_decisions_into_draft(
    draft: dict[str, Any] | None,
    recommended_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    base = dict(draft) if isinstance(draft, dict) else {}
    base["status"] = "OK"
    base["decisions"] = recommended_decisions
    if not isinstance(base.get("assumptions"), list):
        base["assumptions"] = []
    if not isinstance(base.get("global_notes"), list):
        base["global_notes"] = []
    return base


def _build_ok_result(
    *,
    decisions: list[dict[str, Any]],
    base_draft: dict[str, Any] | None,
    size_profiles_by_category: dict[str, Any] | None,
    budget_mode: str | None,
) -> dict[str, Any]:
    result = _merge_recommended_decisions_into_draft(base_draft, decisions)
    result["status"] = "OK"
    result["budget_valid"] = True
    if budget_mode:
        result["budget_mode"] = budget_mode
    if isinstance(size_profiles_by_category, dict):
        _attach_rep_dims(result, size_profiles_by_category)
    return result


def _build_unsat_result(
    *,
    base_draft: dict[str, Any] | None,
    fallback_decisions: list[dict[str, Any]],
    budget_out: dict[str, Any],
    size_profiles_by_category: dict[str, Any] | None,
) -> dict[str, Any]:
    result = dict(base_draft) if isinstance(base_draft, dict) else {}
    result["status"] = "UNSAT"
    result["decisions"] = (
        fallback_decisions if isinstance(fallback_decisions, list) else []
    )
    if not isinstance(result.get("assumptions"), list):
        result["assumptions"] = []
    if not isinstance(result.get("global_notes"), list):
        result["global_notes"] = []

    core_clusters = budget_out.get("core_minimum_infeasible_clusters")
    blocked_clusters = budget_out.get("repair_blocked_clusters")

    notes = list(result.get("global_notes") or [])
    if isinstance(core_clusters, list) and core_clusters:
        notes.append(
            f"Minimum surviving CORE clusters still exceed budget: {core_clusters}"
        )
    elif isinstance(blocked_clusters, list) and blocked_clusters:
        notes.append(f"Repair blocked clusters: {blocked_clusters}")
    else:
        notes.append("Budget remained infeasible after allowed minimal repairs.")

    result["global_notes"] = notes
    result["budget_valid"] = False
    result["budget_reason"] = "hard_unsat"

    if isinstance(size_profiles_by_category, dict):
        _attach_rep_dims(result, size_profiles_by_category)
    return result


def _tool_reports_repair_exhausted(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("repair_exhausted", False)):
        return True
    return _budget_repair_exhausted(budget_out)


def _tool_reports_hard_unsat(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("core_minimum_infeasible", False)):
        return True
    if bool(budget_out.get("repair_exhausted", False)) and bool(
        budget_out.get("global_repair_blocked", False)
    ):
        return True
    return False


# ============================================================
# Post-process: attach rep_dims_m for composer
# ============================================================


def _attach_rep_dims(
    result: dict[str, Any], size_profiles_by_category: dict[str, Any]
) -> None:
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        return

    for d in decisions:
        if not isinstance(d, dict):
            continue
        if "rep_dims_m" in d and isinstance(d.get("rep_dims_m"), dict):
            continue

        tier = d.get("size_tier")
        if not isinstance(tier, str) or not tier:
            continue

        category = d.get("category")
        object_type = d.get("object_type")

        profile = None
        if isinstance(category, str) and category in size_profiles_by_category:
            profile = size_profiles_by_category.get(category)
        elif isinstance(object_type, str) and object_type in size_profiles_by_category:
            profile = size_profiles_by_category.get(object_type)
        elif isinstance(size_profiles_by_category.get("__generic__"), dict):
            profile = size_profiles_by_category.get("__generic__")

        if not isinstance(profile, dict):
            continue

        rep = (profile.get("rep_dims_m") or {}).get(tier.upper())
        if isinstance(rep, dict):
            d["rep_dims_m"] = rep


def _extract_categories_from_decisions(result: dict[str, Any]) -> list[str]:
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        return []
    cats: list[str] = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        for key in ("category", "object_type"):
            v = d.get(key)
            if isinstance(v, str) and v and v not in cats:
                cats.append(v)
    return cats


def _extract_member_types(clusters_json: Any) -> list[str]:
    if not isinstance(clusters_json, dict):
        return []
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return []
    out: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        members = cluster.get("members")
        if not isinstance(members, list):
            continue
        for m in members:
            if isinstance(m, str) and m and m not in out:
                out.append(m)
    return out


def _extract_members_by_cluster(clusters_json: Any) -> dict[str, set[str]]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}

    out: dict[str, set[str]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        members = cluster.get("members")
        if not isinstance(members, list):
            continue
        out[cluster_id] = {m for m in members if isinstance(m, str) and m}
    return out


def _extract_anchors_by_cluster(clusters_json: Any) -> dict[str, set[str]]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}

    out: dict[str, set[str]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        members = cluster.get("members")
        member_set = (
            {member for member in members if isinstance(member, str) and member}
            if isinstance(members, list)
            else set()
        )
        anchors = cluster.get("anchors")
        if not isinstance(anchors, list):
            out.setdefault(cluster_id, set())
            continue
        out[cluster_id] = {
            anchor
            for anchor in anchors
            if isinstance(anchor, str) and anchor and anchor in member_set
        }
    return out


def _extract_droppable_clusters(clusters_json: Any) -> set[str]:
    if not isinstance(clusters_json, dict):
        return set()

    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return set()

    out: set[str] = set()
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue

        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue

        tag = str(cluster.get("tag") or "").strip().lower()
        rules = cluster.get("cluster_rules") or {}
        allow_empty = (
            isinstance(rules, dict) and rules.get("allow_empty_cluster") is True
        )

        if tag in {"misc", "decor", "accent", "accessory"} or allow_empty:
            out.add(cluster_id)

    return out


def _validate_decisions(
    decisions: Any,
    required_types: list[str],
    *,
    members_by_cluster: dict[str, set[str]] | None = None,
    anchors_by_cluster: dict[str, set[str]] | None = None,
    droppable_clusters: set[str] | None = None,
) -> tuple[bool, str]:
    if not isinstance(decisions, list) or not decisions:
        return False, "decisions list is missing or empty"

    types: list[str] = []
    bad_qty: list[str] = []
    bad_cluster_id: list[str] = []
    bad_cluster_assignment: list[str] = []
    unexpected_types: list[str] = []
    decisions_by_cluster: dict[str, dict[str, int]] = {}
    decision_counts_by_cluster: dict[str, dict[str, int]] = {}

    for d in decisions:
        if not isinstance(d, dict):
            continue
        t = d.get("object_type") or d.get("category")
        if not isinstance(t, str) or not t:
            continue

        types.append(t)

        if required_types and t not in required_types:
            unexpected_types.append(t)

        qty = d.get("quantity")
        if not isinstance(qty, int) or qty < 0:
            bad_qty.append(t)

        cluster_id = d.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            bad_cluster_id.append(t)
            continue

        if isinstance(members_by_cluster, dict) and members_by_cluster:
            members = members_by_cluster.get(cluster_id)
            if not isinstance(members, set):
                bad_cluster_assignment.append(f"{t}->{cluster_id}")
            elif t not in members:
                bad_cluster_assignment.append(f"{t}->{cluster_id}")

        decisions_by_cluster.setdefault(cluster_id, {})[t] = (
            qty if isinstance(qty, int) else 0
        )
        cluster_counts = decision_counts_by_cluster.setdefault(cluster_id, {})
        cluster_counts[t] = cluster_counts.get(t, 0) + 1

    if bad_qty:
        return False, f"quantity must be an integer >= 0 for: {sorted(set(bad_qty))}"

    if bad_cluster_id:
        return (
            False,
            f"cluster_id must be a non-empty string for: {sorted(set(bad_cluster_id))}",
        )

    if bad_cluster_assignment:
        return (
            False,
            "each decision must reference a valid cluster_id and a member in that cluster. "
            f"invalid assignments: {sorted(set(bad_cluster_assignment))}",
        )

    if unexpected_types:
        return False, f"unexpected decisions for types: {sorted(set(unexpected_types))}"

    if required_types:
        missing = [t for t in required_types if t not in types]
        if missing:
            return False, f"missing decisions for types: {missing}"

    duplicate_cluster_members = sorted(
        f"{cluster_id}.{object_type}"
        for cluster_id, counts in decision_counts_by_cluster.items()
        for object_type, count in counts.items()
        if count > 1
    )
    if duplicate_cluster_members:
        return (
            False,
            "multiple decisions found for cluster-scoped members: "
            f"{duplicate_cluster_members}",
        )

    droppable_clusters = droppable_clusters or set()

    if isinstance(members_by_cluster, dict) and members_by_cluster:
        for cluster_id in sorted(members_by_cluster.keys()):
            members = members_by_cluster.get(cluster_id) or set()
            anchors = (
                anchors_by_cluster.get(cluster_id, set())
                if isinstance(anchors_by_cluster, dict)
                else set()
            )
            cluster_decisions = decisions_by_cluster.get(cluster_id, {})
            missing_cluster_members = sorted(
                member for member in members if member not in cluster_decisions
            )
            if missing_cluster_members:
                return (
                    False,
                    "each cluster member must have exactly one decision in its own "
                    f"cluster. missing members for {cluster_id}: "
                    f"{missing_cluster_members}",
                )

            if cluster_id in droppable_clusters:
                continue

            if anchors:
                kept = any(cluster_decisions.get(a, 0) >= 1 for a in anchors)
                if not kept:
                    return (
                        False,
                        f"CORE cluster {cluster_id} must keep at least one anchor with quantity >= 1: {sorted(anchors)}",
                    )
            else:
                kept = any(
                    cluster_decisions.get(m, 0) >= 1
                    for m in members
                    if isinstance(m, str)
                )
                if not kept and members:
                    return (
                        False,
                        f"CORE cluster {cluster_id} must keep at least one member with quantity >= 1",
                    )

    return True, "ok"


def _preview_budget_repair_hints(
    budget_out: dict[str, Any],
    limit: int = 8,
) -> list[dict[str, Any]]:
    details = budget_out.get("decision_footprint_details")
    if not isinstance(details, list):
        return []

    out: list[dict[str, Any]] = []
    for row in details:
        if not isinstance(row, dict):
            continue

        orig_qty = row.get("quantity")
        rec_qty = row.get("recommended_quantity")
        orig_tier = row.get("size_tier")
        rec_tier = row.get("recommended_size_tier")

        changed_qty = (
            isinstance(orig_qty, int)
            and isinstance(rec_qty, int)
            and orig_qty != rec_qty
        )
        changed_tier = (
            isinstance(orig_tier, str)
            and isinstance(rec_tier, str)
            and orig_tier != rec_tier
        )

        if not changed_qty and not changed_tier:
            continue

        out.append(
            {
                "cluster_id": row.get("cluster_id"),
                "category": row.get("category"),
                "priority": row.get("priority"),
                "quantity": orig_qty,
                "recommended_quantity": rec_qty,
                "size_tier": orig_tier,
                "recommended_size_tier": rec_tier,
            }
        )

        if len(out) >= limit:
            break

    return out


def _budget_repair_exhausted(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("recommended_decisions_fit", False)):
        return False

    details = budget_out.get("decision_footprint_details")
    if not isinstance(details, list):
        return False

    for row in details:
        if not isinstance(row, dict):
            continue
        if row.get("quantity") != row.get("recommended_quantity"):
            return False
        if row.get("size_tier") != row.get("recommended_size_tier"):
            return False

    return True


def _preview_budget_violations(
    violations: list[dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in violations[:limit]:
        if not isinstance(v, dict):
            continue
        out.append(
            {
                "cluster_id": v.get("cluster_id"),
                "limit_m2": v.get("limit_m2"),
                "footprint_m2": v.get("footprint_m2"),
                "over_by_m2": v.get("over_by_m2"),
            }
        )
    return out
