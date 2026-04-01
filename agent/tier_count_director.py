from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent.tool_call_parser import extract_tool_calls as parse_tool_calls
from prompt.tier_count import TIER_COUNT_DIRECTOR, TIER_COUNT_DIRECTOR_SYSTEM

logger = logging.getLogger(__name__)


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

    logger.info(
        "TierCount deterministic mode: available_area_m2=%.2f room_scale=%s furnishing_mode=%s",
        available_area_m2,
        room_scale,
        furnishing_mode,
    )

    size_profiles = _ensure_size_profiles(
        required_types=required_types,
        tenant_id=context.get("tenant_id"),
        existing=size_profiles_by_category,
    )

    decisions = _build_seed_decisions(
        clusters_json=clusters_json,
        room_scale=room_scale,
        furnishing_mode=furnishing_mode,
        droppable_clusters=droppable_clusters,
    )

    draft = {
        "status": "OK",
        "assumptions": _build_deterministic_assumptions(
            context=context,
            furnishing_mode=furnishing_mode,
            available_area_m2=available_area_m2,
        ),
        "decisions": decisions,
        "global_notes": _build_deterministic_notes(
            context=context,
            furnishing_mode=furnishing_mode,
        ),
    }

    ok, detail = _validate_decisions(
        draft.get("decisions"),
        required_types,
        members_by_cluster=members_by_cluster,
        anchors_by_cluster=anchors_by_cluster,
        droppable_clusters=droppable_clusters,
    )
    if not ok:
        raise ValueError(f"Deterministic tier-count seed is invalid: {detail}")

    frozen_cluster_budget_limits_m2: dict[str, float] | None = None
    rescue_mode_used = False
    current_draft = draft
    latest_budget_out: dict[str, Any] | None = None

    for attempt in range(max(1, int(max_steps))):
        logger.info("TierCount deterministic budget pass %s/%s", attempt + 1, max_steps)

        budget_out = _run_budget_check_for_result(
            result=current_draft,
            context=context,
            size_profiles_by_category=size_profiles,
            frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
            rescue_mode=False,
        )
        if _tool_output_has_error(budget_out):
            raise RuntimeError(
                f"estimate_budget failed in deterministic tier count: {_tool_error_text(budget_out)}"
            )
        latest_budget_out = budget_out

        if (
            frozen_cluster_budget_limits_m2 is None
            and isinstance(budget_out.get("cluster_budget_limits_m2"), dict)
            and not bool(budget_out.get("rescue_mode", False))
        ):
            frozen_cluster_budget_limits_m2 = budget_out["cluster_budget_limits_m2"]

        if bool(budget_out.get("input_decisions_fit", False)):
            return _build_ok_result(
                decisions=current_draft.get("decisions", []),
                base_draft=current_draft,
                size_profiles_by_category=size_profiles,
                budget_mode=None,
            )

        recommended_decisions = budget_out.get("recommended_decisions")
        if bool(budget_out.get("recommended_decisions_fit", False)) and isinstance(
            recommended_decisions, list
        ):
            return _build_ok_result(
                decisions=recommended_decisions,
                base_draft=current_draft,
                size_profiles_by_category=size_profiles,
                budget_mode=None,
            )

        next_decisions = _select_next_budget_decisions(
            current_decisions=current_draft.get("decisions", []),
            budget_out=budget_out,
        )
        if next_decisions is not None:
            current_draft = _merge_recommended_decisions_into_draft(
                current_draft,
                next_decisions,
            )
            continue

        if not rescue_mode_used:
            rescue_budget_out = _run_budget_check_for_result(
                result=current_draft,
                context=context,
                size_profiles_by_category=size_profiles,
                frozen_cluster_budget_limits_m2=None,
                rescue_mode=True,
            )
            if _tool_output_has_error(rescue_budget_out):
                raise RuntimeError(
                    "estimate_budget rescue failed in deterministic tier count: "
                    f"{_tool_error_text(rescue_budget_out)}"
                )
            latest_budget_out = rescue_budget_out
            rescue_mode_used = True

            if bool(rescue_budget_out.get("input_decisions_fit", False)):
                return _build_ok_result(
                    decisions=current_draft.get("decisions", []),
                    base_draft=current_draft,
                    size_profiles_by_category=size_profiles,
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
                    base_draft=current_draft,
                    size_profiles_by_category=size_profiles,
                    budget_mode="rescue",
                )

            next_decisions = _select_next_budget_decisions(
                current_decisions=current_draft.get("decisions", []),
                budget_out=rescue_budget_out,
            )
            if next_decisions is not None:
                current_draft = _merge_recommended_decisions_into_draft(
                    current_draft,
                    next_decisions,
                )
                continue

            if _tool_reports_hard_unsat(rescue_budget_out):
                return _build_unsat_result(
                    base_draft=current_draft,
                    fallback_decisions=rescue_recommended_decisions
                    if isinstance(rescue_recommended_decisions, list)
                    else current_draft.get("decisions", []),
                    budget_out=rescue_budget_out,
                    size_profiles_by_category=size_profiles,
                )

        if _tool_reports_hard_unsat(budget_out):
            return _build_unsat_result(
                base_draft=current_draft,
                fallback_decisions=current_draft.get("decisions", []),
                budget_out=budget_out,
                size_profiles_by_category=size_profiles,
            )

        break

    return _build_unsat_result(
        base_draft=current_draft,
        fallback_decisions=current_draft.get("decisions", []),
        budget_out=latest_budget_out or {},
        size_profiles_by_category=size_profiles,
    )


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


def _build_seed_decisions(
    *,
    clusters_json: Any,
    room_scale: str,
    furnishing_mode: str,
    droppable_clusters: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(clusters_json, dict):
        return []

    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return []

    decisions: list[dict[str, Any]] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue

        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue

        tag = str(cluster.get("tag") or "").strip().lower()
        members = cluster.get("members")
        if not isinstance(members, list):
            continue

        anchors = {
            a
            for a in (cluster.get("anchors") or [])
            if isinstance(a, str) and a.strip()
        }
        access_required_ids = _extract_cluster_access_required_ids(cluster)
        droppable = cluster_id in droppable_clusters

        for member in members:
            if not isinstance(member, str) or not member.strip():
                continue

            priority = _determine_member_priority(
                member=member,
                cluster_tag=tag,
                anchors=anchors,
                access_required_ids=access_required_ids,
                droppable=droppable,
            )
            quantity = _determine_member_quantity(
                member=member,
                priority=priority,
                cluster_tag=tag,
                room_scale=room_scale,
                furnishing_mode=furnishing_mode,
                droppable=droppable,
                anchors=anchors,
            )
            size_tier = _determine_member_size_tier(
                member=member,
                priority=priority,
                room_scale=room_scale,
                furnishing_mode=furnishing_mode,
            )

            decisions.append(
                {
                    "object_type": member,
                    "category": member,
                    "cluster_id": cluster_id,
                    "quantity": quantity,
                    "size_tier": size_tier,
                    "priority": priority,
                    "rationale": _build_member_rationale(
                        member=member,
                        cluster_tag=tag,
                        priority=priority,
                        quantity=quantity,
                        size_tier=size_tier,
                        furnishing_mode=furnishing_mode,
                    ),
                }
            )

    return decisions


def _extract_cluster_access_required_ids(cluster: dict[str, Any]) -> set[str]:
    required_ids: set[str] = set()

    hard_constraints = cluster.get("hard_constraints")
    if isinstance(hard_constraints, list):
        for item in hard_constraints:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "requires_access":
                continue
            member_id = item.get("id")
            if isinstance(member_id, str) and member_id.strip():
                required_ids.add(member_id)

    rules = cluster.get("cluster_rules")
    if isinstance(rules, dict):
        access_requirements = rules.get("access_requirements")
        if isinstance(access_requirements, list):
            for item in access_requirements:
                if not isinstance(item, dict):
                    continue
                member_id = item.get("id")
                if isinstance(member_id, str) and member_id.strip():
                    required_ids.add(member_id)

    return required_ids


def _determine_member_priority(
    *,
    member: str,
    cluster_tag: str,
    anchors: set[str],
    access_required_ids: set[str],
    droppable: bool,
) -> str:
    if member in anchors:
        return "anchor"
    if member in access_required_ids:
        return "primary" if not droppable else "secondary"

    member_key = _norm_key(member)
    if any(token in member_key for token in OPTIONAL_MEMBER_TOKENS):
        return "optional" if droppable else "secondary"
    if any(token in member_key for token in SECONDARY_MEMBER_TOKENS):
        return "secondary"

    if droppable or cluster_tag in {"misc", "decor", "accent", "accessory"}:
        return (
            "secondary"
            if any(token in member_key for token in LARGE_MEMBER_TOKENS)
            else "optional"
        )

    if cluster_tag in {
        "sleep",
        "work",
        "living",
        "storage",
        "kitchen",
        "dining",
        "bathroom",
    }:
        return "primary"

    return "secondary"


def _determine_member_quantity(
    *,
    member: str,
    priority: str,
    cluster_tag: str,
    room_scale: str,
    furnishing_mode: str,
    droppable: bool,
    anchors: set[str],
) -> int:
    member_key = _norm_key(member)
    if priority == "anchor":
        return 1

    if member_key == "nightstand":
        if any(_norm_key(anchor) == "bed" for anchor in anchors):
            if furnishing_mode == "generous" and room_scale in {"medium", "large"}:
                return 2
            if furnishing_mode == "neutral" and room_scale == "large":
                return 2
        return 1

    if priority == "primary":
        return 1

    if priority == "secondary":
        if droppable and furnishing_mode == "minimal":
            return 0
        return 1

    if priority == "optional":
        if furnishing_mode == "generous" and room_scale != "small":
            return 1
        if (
            cluster_tag in {"living"}
            and room_scale == "large"
            and member_key == "floor_lamp"
        ):
            return 1
        return 0

    return 1


def _determine_member_size_tier(
    *,
    member: str,
    priority: str,
    room_scale: str,
    furnishing_mode: str,
) -> str:
    member_key = _norm_key(member)

    if member_key == "bed":
        if room_scale == "large" and furnishing_mode != "minimal":
            return "L"
        return "M"

    if any(token in member_key for token in LARGE_MEMBER_TOKENS):
        base_tier = "M"
    elif any(token in member_key for token in MEDIUM_MEMBER_TOKENS):
        base_tier = "M"
    elif priority in {"anchor", "primary"}:
        base_tier = "M"
    else:
        base_tier = "S"

    if furnishing_mode == "minimal":
        return _shift_tier(base_tier, -1 if base_tier != "S" else 0)
    if (
        furnishing_mode == "generous"
        and priority in {"anchor", "primary"}
        and room_scale in {"medium", "large"}
    ):
        return _shift_tier(base_tier, 1)
    if room_scale == "small" and priority != "anchor":
        return _shift_tier(base_tier, -1)
    return base_tier


def _build_member_rationale(
    *,
    member: str,
    cluster_tag: str,
    priority: str,
    quantity: int,
    size_tier: str,
    furnishing_mode: str,
) -> str:
    quantity_note = "kept" if quantity >= 1 else "dropped"
    cluster_note = cluster_tag or "cluster"
    return (
        f"Deterministic policy marked {member} as {priority} in {cluster_note}; "
        f"{quantity_note} at quantity {quantity} with size tier {size_tier} for a {furnishing_mode} furnishing plan."
    )


def _build_deterministic_assumptions(
    *,
    context: dict[str, Any],
    furnishing_mode: str,
    available_area_m2: float,
) -> list[str]:
    assumptions = [
        "Deterministic tier-count policy applied from cluster membership, room area, and budget heuristics."
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
    return assumptions


def _build_deterministic_notes(
    *,
    context: dict[str, Any],
    furnishing_mode: str,
) -> list[str]:
    notes = ["TierCountDirector used deterministic rules instead of an LLM."]
    style = _infer_style(context)
    if style:
        notes.append(f"Budget seeding considered the requested style: {style}.")
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


def _classify_room_scale(area_m2: float) -> str:
    if area_m2 < 10.0:
        return "small"
    if area_m2 < 18.0:
        return "medium"
    return "large"


def _shift_tier(tier: str, delta: int) -> str:
    ordered = ["S", "M", "L"]
    try:
        idx = ordered.index(tier.upper())
    except ValueError:
        idx = 1
    next_idx = max(0, min(len(ordered) - 1, idx + delta))
    return ordered[next_idx]


def _profile_category_for_member(member: str) -> str:
    normalized = _norm_key(member)
    return PROFILE_CATEGORY_ALIASES.get(normalized, member)


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
                            "Requirement: exactly one decision per member type across all clusters, "
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
                        "You MUST output exactly one decision per member type across all clusters, "
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

    duplicates = {t for t in types if types.count(t) > 1}
    if duplicates:
        return False, f"multiple decisions found for types: {sorted(duplicates)}"

    droppable_clusters = droppable_clusters or set()

    if isinstance(members_by_cluster, dict) and members_by_cluster:
        for cluster_id in sorted(members_by_cluster.keys()):
            if cluster_id in droppable_clusters:
                continue

            members = members_by_cluster.get(cluster_id) or set()
            anchors = (
                anchors_by_cluster.get(cluster_id, set())
                if isinstance(anchors_by_cluster, dict)
                else set()
            )
            cluster_decisions = decisions_by_cluster.get(cluster_id, {})

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
