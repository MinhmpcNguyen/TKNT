from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from layout.room_profiles.base import RoomProfile
from layout.room_profiles.registry import (
    canonical_profile_object_type,
    resolve_room_profile,
    semantic_room_rule_for,
)
from layout.semantic_roles import (
    is_lounge_anchor_like,
    is_seat_like,
    is_work_surface_like,
)

_HARD_INTENTS = frozenset({"must_keep", "must_try"})
_SOLVER_PARTICIPATIONS = frozenset({"hard_solver_object", "soft_solver_object"})
_POST_SOLVER_PARTICIPATIONS = frozenset(
    {"post_solver_accessory", "ignored_unavailable"}
)
_SURFACE_RELATION_PATTERN = re.compile(
    r"\b(?:chair|seat|stool|bench|armchair|bean\s+bag|bean_bag)\b"
    r"[^,.]{0,48}\b(?:in\s+front\s+of|facing|at|by|near)\b"
    r"[^,.]{0,32}\btable\b"
)
_BED_SIDE_SURFACE_PATTERN = re.compile(
    r"\b(?:table|side\s+table|bedside\s+table)\b"
    r"[^,.]{0,48}\b(?:next\s+to|beside|by|near)\b"
    r"[^,.]{0,24}\bbed\b"
)


def reconcile_request_contract_with_semantic_program(
    semantic_program: Mapping[str, Any],
    *,
    brief_text: str,
    inventory_object_types: Sequence[str] = (),
) -> dict[str, Any]:
    out = deepcopy(dict(semantic_program))
    contract = out.get("request_contract")
    if not isinstance(contract, Mapping):
        return out

    room_type = str(out.get("room_type") or "").strip()
    inventory_types = _canonical_set(inventory_object_types)
    contract_out = deepcopy(dict(contract))
    objects = _contract_objects(contract_out)
    objects = _append_relation_implied_objects(
        objects,
        room_type=room_type,
        brief_text=brief_text,
        inventory_types=inventory_types,
    )

    active_clusters = _active_clusters(out)
    room_rule = semantic_room_rule_for(room_type) or {}
    profile = resolve_room_profile(room_type)
    existing_members = _program_member_types(active_clusters)
    statuses: list[dict[str, Any]] = []
    missing: list[str] = _string_list(out.get("missing"))
    notes: list[str] = _string_list(out.get("notes"))

    for item in objects:
        object_type = _canonical_object_type(item.get("object_type"))
        if not object_type or not _requested(item):
            continue
        preferred_object_type = _preferred_object_type_from_evidence(
            object_type=object_type,
            item=item,
            inventory_types=inventory_types,
        )
        if preferred_object_type != object_type:
            item["request_reconciliation_status"] = "substituted"
            item["substituted_from"] = object_type
            item["chosen_object_type"] = preferred_object_type
            object_type = preferred_object_type

        item["available_in_inventory"] = (
            object_type in inventory_types
            if inventory_types
            else bool(item.get("inventory_available"))
        )
        item["available_in_program"] = object_type in existing_members

        fulfilled_by = _fulfilled_by_existing_object(
            object_type=object_type,
            item=item,
            existing_members=existing_members,
            room_type=room_type,
            profile=profile,
            brief_text=brief_text,
        )
        if fulfilled_by:
            item["request_reconciliation_status"] = "fulfilled"
            item["chosen_object_type"] = fulfilled_by
            item["fulfilled_by"] = fulfilled_by
            item["available_in_program"] = True
            statuses.append(
                _status_row(item, status="fulfilled", object_type=object_type)
            )
            continue

        if _post_solver_item(item):
            item["request_reconciliation_status"] = "fulfilled"
            item["chosen_object_type"] = object_type
            statuses.append(
                _status_row(item, status="fulfilled", object_type=object_type)
            )
            continue

        if inventory_types and object_type not in inventory_types:
            substitute = _substitute_object_type(
                object_type=object_type,
                inventory_types=inventory_types,
                room_type=room_type,
                profile=profile,
            )
            if substitute is None:
                item["request_reconciliation_status"] = "unfulfilled"
                item["unfulfilled_reason"] = (
                    "requested_object_not_available_in_inventory"
                )
                missing.append(object_type)
                statuses.append(
                    _status_row(
                        item,
                        status="unfulfilled",
                        object_type=object_type,
                        reason="requested_object_not_available_in_inventory",
                    )
                )
                continue
            item["request_reconciliation_status"] = "substituted"
            item["substituted_from"] = object_type
            item["chosen_object_type"] = substitute
            object_type = substitute

        cluster = _find_or_create_cluster_for_object(
            active_clusters=active_clusters,
            room_rule=room_rule,
            room_type=room_type,
            object_type=object_type,
            profile=profile,
        )
        _add_object_to_cluster(
            cluster, object_type=object_type, item=item, profile=profile
        )
        existing_members.add(object_type)
        item["available_in_program"] = True
        item.setdefault("chosen_object_type", object_type)
        if item.get("request_reconciliation_status") != "substituted":
            item["request_reconciliation_status"] = "reconciled"
        statuses.append(
            _status_row(
                item,
                status=str(item["request_reconciliation_status"]),
                object_type=object_type,
            )
        )

    contract_out["objects"] = objects
    contract_out["reconciliation"] = {
        "version": 1,
        "strategy": "hybrid_llm_rules",
        "statuses": statuses,
    }
    contract_notes = _string_list(contract_out.get("notes"))
    contract_notes.append(
        "Request contract was reconciled against room profile, inventory, and semantic clusters before tier count."
    )
    contract_out["notes"] = _dedupe(contract_notes)
    out["request_contract"] = contract_out
    out["active_clusters"] = active_clusters
    out["missing"] = _dedupe(missing)
    notes.append("Hybrid request reconciliation completed before tier count.")
    out["notes"] = _dedupe(notes)
    return out


def _append_relation_implied_objects(
    objects: list[dict[str, Any]],
    *,
    room_type: str,
    brief_text: str,
    inventory_types: set[str],
) -> list[dict[str, Any]]:
    text = _normalize_text(brief_text)
    out = [dict(item) for item in objects]
    existing = {
        canonical
        for item in out
        if (canonical := _canonical_object_type(item.get("object_type")))
    }
    if _SURFACE_RELATION_PATTERN.search(text):
        surface = _surface_for_seat_table_relation(
            room_type=room_type,
            inventory_types=inventory_types,
        )
        if surface and surface not in existing:
            out.append(
                _implied_contract_object(
                    object_type=surface,
                    evidence=_relation_evidence(brief_text, "table"),
                    reason="surface implied by an explicit seating-to-table relation",
                )
            )
            existing.add(surface)
    if _BED_SIDE_SURFACE_PATTERN.search(text):
        surface = _bedside_surface(room_type=room_type, inventory_types=inventory_types)
        if surface and surface not in existing:
            out.append(
                _implied_contract_object(
                    object_type=surface,
                    evidence=_relation_evidence(brief_text, "table"),
                    reason="bedside surface implied by an explicit table-near-bed relation",
                )
            )
    return out


def _surface_for_seat_table_relation(
    *,
    room_type: str,
    inventory_types: set[str],
) -> str | None:
    room_rule = semantic_room_rule_for(room_type) or {}
    profile = resolve_room_profile(room_type)
    candidates: list[str] = []
    for cluster in _room_rule_clusters(room_rule):
        program_types = _object_program_types(cluster.get("object_program"))
        if not any(_is_seat_object(item, profile=profile) for item in program_types):
            continue
        candidates.extend(
            item for item in program_types if _is_surface_object(item, profile=profile)
        )
    if profile is not None:
        candidates.extend(sorted(profile.surface_objects))
    ranked = sorted(
        _dedupe(
            [
                item
                for item in candidates
                if not inventory_types or item in inventory_types
            ]
        ),
        key=_surface_relation_rank,
    )
    return ranked[0] if ranked else None


def _bedside_surface(*, room_type: str, inventory_types: set[str]) -> str | None:
    profile = resolve_room_profile(room_type)
    candidates: list[str] = []
    if profile is not None:
        candidates.extend(sorted(profile.surface_objects))
    for candidate in ("nightstand", "side_table"):
        if candidate not in candidates:
            candidates.append(candidate)
    available = [
        item for item in candidates if not inventory_types or item in inventory_types
    ]
    return min(available, key=_bedside_surface_rank) if available else None


def _fulfilled_by_existing_object(
    *,
    object_type: str,
    item: Mapping[str, Any],
    existing_members: set[str],
    room_type: str,
    profile: RoomProfile | None,
    brief_text: str,
) -> str | None:
    if object_type in existing_members:
        return object_type
    evidence = _normalize_text(str(item.get("evidence") or brief_text))
    if (
        object_type == "side_table"
        and "bed" in evidence
        and "nightstand" in existing_members
    ):
        return "nightstand"
    if object_type == "nightstand" and "side_table" in existing_members:
        return "side_table"
    if _is_surface_object(object_type, profile=profile) and object_type not in {
        "coffee_table",
        "desk",
        "nightstand",
        "side_table",
    }:
        for candidate in existing_members:
            if _is_surface_object(candidate, profile=profile) and "table" in evidence:
                return candidate
    if _is_lighting_object(object_type, profile=profile):
        for candidate in existing_members:
            if _is_lighting_object(candidate, profile=profile):
                return candidate
    _ = room_type
    return None


def _substitute_object_type(
    *,
    object_type: str,
    inventory_types: set[str],
    room_type: str,
    profile: RoomProfile | None,
) -> str | None:
    if object_type == "bean_bag":
        for candidate in ("bean_bag", "armchair", "ottoman", "chair"):
            if candidate in inventory_types and _is_seat_object(
                candidate, profile=profile
            ):
                return candidate
    if _is_lighting_object(object_type, profile=profile):
        for candidate in ("table_lamp", "bedside_lamp", "floor_lamp", "desk_lamp"):
            if candidate in inventory_types:
                return candidate
    _ = room_type
    return None


def _preferred_object_type_from_evidence(
    *,
    object_type: str,
    item: Mapping[str, Any],
    inventory_types: set[str],
) -> str:
    evidence = _normalize_text(item.get("evidence"))
    if (
        object_type == "armchair"
        and "bean bag" in evidence
        and (not inventory_types or "bean_bag" in inventory_types)
    ):
        return "bean_bag"
    return object_type


def _find_or_create_cluster_for_object(
    *,
    active_clusters: list[dict[str, Any]],
    room_rule: Mapping[str, Any],
    room_type: str,
    object_type: str,
    profile: RoomProfile | None,
) -> dict[str, Any]:
    active_by_id = {
        str(cluster.get("cluster_id") or ""): cluster for cluster in active_clusters
    }
    best_rule = _best_cluster_rule_for_object(
        room_rule=room_rule,
        object_type=object_type,
        profile=profile,
        active_cluster_ids=set(active_by_id),
    )
    if best_rule is not None:
        cluster_id = str(best_rule.get("cluster_id") or "").strip()
        if cluster_id in active_by_id:
            return active_by_id[cluster_id]
        cluster = _cluster_from_rule(
            best_rule, object_type=object_type, room_type=room_type
        )
        active_clusters.append(cluster)
        return cluster
    if active_clusters:
        return max(
            active_clusters,
            key=lambda cluster: _cluster_role_score(
                _cluster_member_types(cluster), object_type=object_type, profile=profile
            ),
        )
    cluster = _fallback_cluster(object_type=object_type, room_type=room_type)
    active_clusters.append(cluster)
    return cluster


def _best_cluster_rule_for_object(
    *,
    room_rule: Mapping[str, Any],
    object_type: str,
    profile: RoomProfile | None,
    active_cluster_ids: set[str],
) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_score = -1
    for cluster in _room_rule_clusters(room_rule):
        program_types = _object_program_types(cluster.get("object_program"))
        score = _cluster_role_score(
            program_types, object_type=object_type, profile=profile
        )
        if object_type in program_types:
            score += 100
        cluster_id = str(cluster.get("cluster_id") or "")
        if cluster_id in active_cluster_ids:
            score += 12
        if score > best_score:
            best = cluster
            best_score = score
    return best if best_score > 0 else None


def _cluster_role_score(
    program_types: set[str],
    *,
    object_type: str,
    profile: RoomProfile | None,
) -> int:
    score = 0
    if _is_seat_object(object_type, profile=profile) and any(
        _is_seat_object(item, profile=profile) for item in program_types
    ):
        score += 40
    if is_lounge_anchor_like(object_type) and any(
        is_lounge_anchor_like(item) for item in program_types
    ):
        score += 25
    if _is_surface_object(object_type, profile=profile) and any(
        _is_surface_object(item, profile=profile) for item in program_types
    ):
        score += 35
    if _is_lighting_object(object_type, profile=profile) and any(
        _is_lighting_object(item, profile=profile) for item in program_types
    ):
        score += 30
    if _is_decor_object(object_type, profile=profile) and any(
        _is_decor_object(item, profile=profile) for item in program_types
    ):
        score += 20
    return score


def _cluster_from_rule(
    cluster_rule: Mapping[str, Any],
    *,
    object_type: str,
    room_type: str,
) -> dict[str, Any]:
    cluster_id = str(cluster_rule.get("cluster_id") or "requested_support").strip()
    priority = _priority(cluster_rule.get("priority"))
    return {
        "cluster_id": cluster_id,
        "layout_role": "support" if priority != "optional" else "optional",
        "priority": "support" if priority == "optional" else priority,
        "activation_reason": "activated by explicit request-contract reconciliation",
        "semantic_role": "explicit_request_zone",
        "dominant_anchor_candidates": _dominant_candidates(cluster_rule, object_type),
        "required_bundles": [
            {
                "bundle_id": f"{cluster_id}_request_contract",
                "objects": [],
            }
        ],
        "zone_claims": _default_zone_claims(
            object_type=object_type, room_type=room_type
        ),
        "relation_intents": [],
        "degradation_ladder": [],
        "tier_count_hints": _tier_count_hints([]),
        "viability_score": {
            "rule_support": 1.0,
            "affordance_support": 1.0,
            "brief_support": 1.0,
            "inventory_support": 1.0,
            "conflict_penalty": 0.0,
            "score": 1.0,
        },
    }


def _fallback_cluster(*, object_type: str, room_type: str) -> dict[str, Any]:
    return {
        "cluster_id": f"requested_{_role_name(object_type)}",
        "layout_role": "support",
        "priority": "support",
        "activation_reason": "created by explicit request-contract reconciliation",
        "semantic_role": "explicit_request_zone",
        "dominant_anchor_candidates": [object_type] if _anchorish(object_type) else [],
        "required_bundles": [
            {
                "bundle_id": f"requested_{_role_name(object_type)}_bundle",
                "objects": [],
            }
        ],
        "zone_claims": _default_zone_claims(
            object_type=object_type, room_type=room_type
        ),
        "relation_intents": [],
        "degradation_ladder": [],
        "tier_count_hints": _tier_count_hints([]),
        "viability_score": {
            "rule_support": 0.7,
            "affordance_support": 1.0,
            "brief_support": 1.0,
            "inventory_support": 1.0,
            "conflict_penalty": 0.0,
            "score": 0.85,
        },
    }


def _add_object_to_cluster(
    cluster: dict[str, Any],
    *,
    object_type: str,
    item: Mapping[str, Any],
    profile: RoomProfile | None,
) -> None:
    bundles = cluster.get("required_bundles")
    if not isinstance(bundles, list) or not bundles:
        bundles = [
            {
                "bundle_id": f"{cluster.get('cluster_id')}_request_contract",
                "objects": [],
            }
        ]
        cluster["required_bundles"] = bundles
    bundle = bundles[0] if isinstance(bundles[0], dict) else {}
    objects = bundle.get("objects")
    if not isinstance(objects, list):
        objects = []
        bundle["objects"] = objects
    if not any(
        _canonical_object_type(row.get("object_type")) == object_type
        for row in objects
        if isinstance(row, Mapping)
    ):
        objects.append(
            {
                "object_type": object_type,
                "role": _semantic_object_role(object_type, profile=profile),
                "required": _hard_item(item),
                "max_keep": _positive_int(item.get("max_keep")),
            }
        )
    if _anchorish(object_type) and object_type not in _string_list(
        cluster.get("dominant_anchor_candidates")
    ):
        cluster["dominant_anchor_candidates"] = [
            *_string_list(cluster.get("dominant_anchor_candidates")),
            object_type,
        ]
    existing_hints = cluster.get("tier_count_hints")
    cluster["tier_count_hints"] = _merge_tier_count_hint(
        existing_hints if isinstance(existing_hints, Mapping) else {},
        object_type=object_type,
        item=item,
    )


def _merge_tier_count_hint(
    value: Mapping[str, Any],
    *,
    object_type: str,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    out = _tier_count_hints(_hint_object_types(value))
    out.update(
        {
            key: deepcopy(value[key])
            for key in value
            if key in out and key != "object_hints"
        }
    )
    hints = [
        dict(row) for row in value.get("object_hints", []) if isinstance(row, Mapping)
    ]
    by_type = {
        str(row.get("object_type") or ""): row
        for row in hints
        if str(row.get("object_type") or "")
    }
    by_type[object_type] = {
        "object_type": object_type,
        "min_keep": max(1, _positive_int(item.get("min_keep"))),
        "max_keep": _positive_int(item.get("max_keep")),
        "keep_if_space_surplus": False,
        "space_surplus_threshold": 0.45,
        "drop_order_bias": "drop_last",
        "preserve_level": "highest" if _hard_item(item) else "high",
        "preferred_size_tier": None,
    }
    out["object_hints"] = list(by_type.values())
    out["bundle_class"] = "indispensable"
    out["preserve_level"] = "highest"
    out["drop_order_bias"] = "drop_last"
    return out


def _tier_count_hints(object_types: Sequence[str]) -> dict[str, Any]:
    return {
        "bundle_class": "indispensable",
        "preserve_level": "highest",
        "keep_if_space_surplus": False,
        "space_surplus_threshold": 0.45,
        "drop_order_bias": "drop_last",
        "object_hints": [
            {
                "object_type": object_type,
                "min_keep": 1,
                "max_keep": 1,
                "keep_if_space_surplus": False,
                "space_surplus_threshold": 0.45,
                "drop_order_bias": "drop_last",
                "preserve_level": "highest",
                "preferred_size_tier": None,
            }
            for object_type in object_types
        ],
    }


def _implied_contract_object(
    *,
    object_type: str,
    evidence: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "object_type": object_type,
        "intent": "must_keep",
        "min_keep": 1,
        "target_count": 1,
        "preferred_count": 1,
        "max_keep": 1,
        "evidence": evidence,
        "reason": reason,
        "confidence": 0.72,
        "placement_participation": "floor_blocking",
        "requested": True,
        "inventory_available": True,
        "available_in_inventory": True,
        "available_in_program": False,
        "solver_participation": "hard_solver_object",
        "blocks_floor_area": True,
        "can_overlap_floor_objects": False,
        "host_object_type": None,
        "mount_region_type": None,
        "llm_candidates": [object_type],
        "relation_edges": [{"relation": "inferred_pair", "target": "table"}],
    }


def _status_row(
    item: Mapping[str, Any],
    *,
    status: str,
    object_type: str,
    reason: str = "",
) -> dict[str, Any]:
    row = {
        "object_type": object_type,
        "status": status,
        "chosen_object_type": str(item.get("chosen_object_type") or object_type),
    }
    fulfilled_by = str(item.get("fulfilled_by") or "").strip()
    if fulfilled_by:
        row["fulfilled_by"] = fulfilled_by
    substituted_from = str(item.get("substituted_from") or "").strip()
    if substituted_from:
        row["substituted_from"] = substituted_from
    if reason:
        row["reason"] = reason
    return row


def _active_clusters(program: dict[str, Any]) -> list[dict[str, Any]]:
    clusters = program.get("active_clusters")
    if not isinstance(clusters, list):
        clusters = []
        program["active_clusters"] = clusters
    return [cluster for cluster in clusters if isinstance(cluster, dict)]


def _contract_objects(contract: dict[str, Any]) -> list[dict[str, Any]]:
    objects = contract.get("objects")
    if not isinstance(objects, list):
        return []
    return [dict(item) for item in objects if isinstance(item, Mapping)]


def _program_member_types(active_clusters: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for cluster in active_clusters:
        out.update(_cluster_member_types(cluster))
    return out


def _cluster_member_types(cluster: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for member in cluster.get("members") or []:
        if isinstance(member, str) and member.strip():
            out.add(member.strip())
    for bundle in cluster.get("required_bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        for item in bundle.get("objects") or []:
            if isinstance(item, Mapping):
                object_type = _canonical_object_type(item.get("object_type"))
                if object_type:
                    out.add(object_type)
    return out


def _object_program_types(value: object) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    out: set[str] = set()
    for key in (
        "required",
        "required_if_kept",
        "optional",
        "choose_exactly_one_from",
        "choose_exactly_one_from_if_kept",
        "choose_at_least_one_from",
    ):
        raw = value.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, str):
            for item in raw:
                if isinstance(item, Sequence) and not isinstance(item, str):
                    out.update(
                        _canonical
                        for row in item
                        if (_canonical := _canonical_object_type(row))
                    )
                else:
                    canonical = _canonical_object_type(item)
                    if canonical:
                        out.add(canonical)
    return out


def _room_rule_clusters(room_rule: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    clusters = room_rule.get("clusters")
    if not isinstance(clusters, list):
        return []
    return [cluster for cluster in clusters if isinstance(cluster, Mapping)]


def _dominant_candidates(
    cluster_rule: Mapping[str, Any],
    fallback_object_type: str,
) -> list[str]:
    semantic = cluster_rule.get("semantic")
    candidates = (
        _string_list(semantic.get("dominant_anchor_candidates"))
        if isinstance(semantic, Mapping)
        else []
    )
    if not candidates and _anchorish(fallback_object_type):
        candidates = [fallback_object_type]
    return candidates


def _surface_types(room_type: str) -> set[str]:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return {"desk", "table", "coffee_table", "side_table", "nightstand"}
    return set(profile.surface_objects) | {
        item
        for item in ("desk", "table", "coffee_table", "side_table", "nightstand")
        if profile.canonical_object_type(item) is not None
    }


def _is_surface_object(object_type: str, *, profile: RoomProfile | None) -> bool:
    return (
        bool(profile and object_type in profile.surface_objects)
        or is_work_surface_like(object_type)
        or object_type in {"nightstand", "side_table", "coffee_table", "table_lamp"}
    )


def _is_seat_object(object_type: str, *, profile: RoomProfile | None) -> bool:
    return bool(profile and object_type in profile.seating_objects) or is_seat_like(
        object_type
    )


def _is_lighting_object(object_type: str, *, profile: RoomProfile | None) -> bool:
    return (
        bool(profile and object_type in profile.lighting_objects)
        or "lamp" in object_type
        or object_type.endswith("_light")
    )


def _is_decor_object(object_type: str, *, profile: RoomProfile | None) -> bool:
    return bool(profile and object_type in profile.decor_objects) or object_type in {
        "rug",
        "plant",
        "wall_art",
    }


def _semantic_object_role(object_type: str, *, profile: RoomProfile | None) -> str:
    if _anchorish(object_type) or bool(
        profile and object_type in profile.anchor_objects
    ):
        return "dominant_anchor"
    if _is_surface_object(object_type, profile=profile) and object_type in {
        "desk",
        "dining_table",
    }:
        return "workflow_anchor"
    if _is_decor_object(object_type, profile=profile):
        return "decor"
    if _is_seat_object(object_type, profile=profile):
        return "secondary_support"
    return "support"


def _anchorish(object_type: str) -> bool:
    return is_lounge_anchor_like(object_type) or object_type in {
        "bed",
        "desk",
        "dining_table",
        "sofa",
        "sectional_sofa",
        "tv_console",
        "wardrobe",
    }


def _default_zone_claims(*, object_type: str, room_type: str) -> dict[str, Any]:
    _ = room_type
    return {
        "preferred_regions": ["floating_support_zone"]
        if _anchorish(object_type)
        else [],
        "avoid_regions": [],
        "wall_affinity": "medium",
        "daylight_affinity": "medium"
        if _is_lighting_object(object_type, profile=None)
        else "none",
        "privacy_affinity": "none",
        "floating_allowed": _anchorish(object_type) or is_seat_like(object_type),
    }


def _surface_relation_rank(object_type: str) -> tuple[int, str]:
    order = {
        "desk": 0,
        "dining_table": 1,
        "coffee_table": 2,
        "side_table": 3,
        "console_table": 4,
        "nightstand": 5,
    }
    return (order.get(object_type, 20), object_type)


def _bedside_surface_rank(object_type: str) -> tuple[int, str]:
    order = {"nightstand": 0, "side_table": 1, "desk": 2}
    return (order.get(object_type, 20), object_type)


def _hint_object_types(value: Mapping[str, Any]) -> list[str]:
    raw = value.get("object_hints")
    if not isinstance(raw, list):
        return []
    return [
        object_type
        for item in raw
        if isinstance(item, Mapping)
        and (object_type := _canonical_object_type(item.get("object_type")))
    ]


def _relation_evidence(brief_text: str, token: str) -> str:
    text = " ".join(str(brief_text or "").split())
    lowered = text.lower()
    index = lowered.find(token)
    if index < 0:
        return text[:120]
    left = max(0, lowered.rfind(",", 0, index) + 1)
    right = lowered.find(",", index)
    if right < 0:
        right = min(len(text), index + 90)
    return text[left:right].strip()


def _canonical_set(values: Sequence[str]) -> set[str]:
    return {
        canonical for value in values if (canonical := _canonical_object_type(value))
    }


def _canonical_object_type(value: object) -> str:
    profile_canonical = canonical_profile_object_type(value)
    if profile_canonical:
        return profile_canonical
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _role_name(object_type: str) -> str:
    if _is_seat_object(object_type, profile=None):
        return "seating"
    if _is_surface_object(object_type, profile=None):
        return "surface"
    if _is_lighting_object(object_type, profile=None):
        return "lighting"
    return "support"


def _requested(item: Mapping[str, Any]) -> bool:
    return bool(item.get("requested", True)) and str(item.get("intent") or "") != "max0"


def _hard_item(item: Mapping[str, Any]) -> bool:
    return str(item.get("intent") or "").strip().lower() in _HARD_INTENTS


def _post_solver_item(item: Mapping[str, Any]) -> bool:
    participation = str(item.get("solver_participation") or "").strip().lower()
    if participation:
        return participation in _POST_SOLVER_PARTICIPATIONS
    return str(item.get("placement_participation") or "").strip().lower() in {
        "textile_overlay",
        "host_attached",
        "opening_attached",
        "ceiling_attached",
        "wall_attached",
        "decor_only",
    } and not bool(item.get("blocks_floor_area"))


def _priority(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"core", "support", "optional"} else "support"


def _positive_int(value: object) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").split())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
