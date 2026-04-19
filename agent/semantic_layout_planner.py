from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_schema.semantic_layout_planner_schema import SemanticLayoutProgram
from clients.base_client import ChatMessage
from clients.llm_client import get_llm_client
from config.gemini_config import GeminiConfig
from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM
from layout.semantic_roles import (
    is_bed_like,
    is_bedside_support_like,
    is_bench_like,
    is_lounge_anchor_like,
    is_seat_like,
    is_work_surface_like,
)
from prompt.semantic_layout_planner import (
    SEMANTIC_LAYOUT_PLANNER_SYSTEM_PROMPT,
    SEMANTIC_LAYOUT_PLANNER_USER_PROMPT,
)
from stylist.semantic_program_rules import get_compiled_semantic_room_rule
from stylist.style_policy import apply_style_policy_to_semantic_program

logger = logging.getLogger(__name__)

_RESPONSE_MIME_TYPE = "application/json"
_LLM_TEMPERATURE = 0.2
_LLM_TOP_P = 0.9
_CLUSTER_CANDIDATE_CAP = 12
_RELATION_INTENT_CAP_PER_CLUSTER = 6
_ZONE_CLAIM_CAP_PER_CLUSTER = 4
_GLOBAL_RELATION_CAP = 20
_DEGRADATION_STEPS_CAP_PER_CLUSTER = 5

_MEDIA_TOKENS = ("tv", "media", "screen", "projector", "console")
_WORK_TOKENS = ("work", "study", "desk", "office", "laptop", "computer")
_READING_TOKENS = ("read", "reading", "lounge", "relax", "armchair")
_PET_TOKENS = ("pet", "dog", "cat")
_ENTRY_TOKENS = ("entry", "shoe", "foyer", "landing")
_LAUNDRY_TOKENS = ("laundry", "hamper", "daily storage", "basket")


@dataclass(frozen=True)
class SemanticLayoutPlanner:
    system_prompt: str = SEMANTIC_LAYOUT_PLANNER_SYSTEM_PROMPT
    prompt_template: str = SEMANTIC_LAYOUT_PLANNER_USER_PROMPT

    def generate(
        self,
        *,
        room_model_json: Mapping[str, Any],
        room_type: str,
        brief_text: str,
        inventory_catalog: Sequence[Mapping[str, Any]] | None = None,
        semantic_program_rules: Mapping[str, Any] | None = None,
        style_policy: Mapping[str, Any] | None = None,
        use_llm: bool = True,
        temperature: float = _LLM_TEMPERATURE,
        top_p: float = _LLM_TOP_P,
        max_tokens: int | None = None,
    ) -> SemanticLayoutProgram:
        room_rules = _room_rules_for(
            room_type=room_type,
            semantic_program_rules=semantic_program_rules,
        )
        inventory = _normalize_inventory_catalog(inventory_catalog, room_rules)
        affordance_summary = summarize_room_affordance(room_model_json)
        candidates = build_cluster_candidates(
            room_rules=room_rules,
            inventory_catalog=inventory,
            brief_text=brief_text,
            room_model_json=room_model_json,
        )
        deterministic_program = _build_deterministic_program(
            room_type=room_type,
            room_rules=room_rules,
            room_model_json=room_model_json,
            affordance_summary=affordance_summary,
            candidates=candidates,
            brief_text=brief_text,
            inventory_catalog=inventory,
        )
        deterministic_program = apply_style_policy_to_semantic_program(
            deterministic_program,
            style_policy,
        )

        if not use_llm or not candidates:
            return SemanticLayoutProgram.model_validate(deterministic_program)

        try:
            llm_payload = self._generate_llm_payload(
                room_type=room_type,
                brief_text=brief_text,
                room_rules=room_rules,
                affordance_summary=affordance_summary,
                candidates=candidates,
                deterministic_program=deterministic_program,
                style_policy=style_policy,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning(
                "SemanticLayoutPlanner LLM pass failed; using deterministic program: %s",
                exc,
            )
            return SemanticLayoutProgram.model_validate(deterministic_program)

        merged = _validate_and_normalize_llm_program(
            llm_payload=llm_payload,
            deterministic_program=deterministic_program,
            room_type=room_type,
            room_rules=room_rules,
            candidates=candidates,
            affordance_summary=affordance_summary,
        )
        merged = apply_style_policy_to_semantic_program(merged, style_policy)
        return SemanticLayoutProgram.model_validate(merged)

    def _generate_llm_payload(
        self,
        *,
        room_type: str,
        brief_text: str,
        room_rules: Mapping[str, Any],
        affordance_summary: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        deterministic_program: Mapping[str, Any],
        style_policy: Mapping[str, Any] | None,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        prompt_payload = {
            "room_type": room_type,
            "brief": {"text": brief_text},
            "room_affordance_summary": affordance_summary,
            "canonical_global_rules": room_rules.get("global_program") or {},
            "policy": room_rules.get("policy") or {},
            "style_policy": dict(style_policy or {}),
            "candidate_clusters": list(candidates)[:_CLUSTER_CANDIDATE_CAP],
            "deterministic_seed_program": {
                "active_clusters": deterministic_program.get("active_clusters") or [],
                "global_layout_intent": deterministic_program.get(
                    "global_layout_intent"
                )
                or {},
                "macro_relations": deterministic_program.get("macro_relations") or {},
            },
            "limits": {
                "relation_intent_cap_per_cluster": _RELATION_INTENT_CAP_PER_CLUSTER,
                "zone_claim_cap_per_cluster": _ZONE_CLAIM_CAP_PER_CLUSTER,
                "global_relation_cap": _GLOBAL_RELATION_CAP,
                "degradation_steps_cap_per_cluster": _DEGRADATION_STEPS_CAP_PER_CLUSTER,
                "llm_temperature": temperature,
                "llm_top_p": top_p,
            },
        }
        user_prompt = self.prompt_template.replace(
            "{PAYLOAD_JSON}",
            json.dumps(prompt_payload, ensure_ascii=True, indent=2),
        )
        messages: list[ChatMessage] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = get_llm_client(provider="gemini")
        model_name = GeminiConfig.AGENT_MODELS.get(
            "forge"
        ) or GeminiConfig.AGENT_MODELS.get("planner")
        last_raw = ""
        for attempt in range(2):
            response = client.chat_completion(
                messages,
                model_key="primary",
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_mime_type=_RESPONSE_MIME_TYPE,
            )
            last_raw = _extract_content(response)
            try:
                return _parse_json(last_raw)
            except ValueError:
                if attempt == 1:
                    raise
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Return valid strict JSON only, with no markdown or prose.",
                    }
                )
        raise ValueError("SemanticLayoutPlanner returned invalid JSON")


def build_cluster_candidates(
    *,
    room_rules: Mapping[str, Any],
    inventory_catalog: Mapping[str, Mapping[str, Any]],
    brief_text: str,
    room_model_json: Mapping[str, Any],
) -> list[dict[str, Any]]:
    clusters = room_rules.get("clusters")
    if not isinstance(clusters, list):
        return []

    candidates: list[dict[str, Any]] = []
    for cluster in clusters[:_CLUSTER_CANDIDATE_CAP]:
        if not isinstance(cluster, Mapping):
            continue
        candidate = _build_single_candidate(
            cluster_rule=cluster,
            inventory_catalog=inventory_catalog,
            brief_text=brief_text,
            room_model_json=room_model_json,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def summarize_room_affordance(room_model_json: Mapping[str, Any]) -> dict[str, Any]:
    affordance = room_model_json.get("affordance_map")
    affordance = affordance if isinstance(affordance, Mapping) else {}
    room = (
        room_model_json.get("room")
        if isinstance(room_model_json.get("room"), Mapping)
        else {}
    )
    openings = (
        room_model_json.get("openings")
        if isinstance(room_model_json.get("openings"), Mapping)
        else {}
    )
    doors = openings.get("doors") if isinstance(openings.get("doors"), list) else []
    windows = (
        openings.get("windows") if isinstance(openings.get("windows"), list) else []
    )
    polygon = room.get("polygon_ccw") or room_model_json.get("polygon_mm") or []
    wall_count = max(0, len(polygon) if isinstance(polygon, list) else 0)

    return {
        "usable_wall_segments": _region_labels(
            affordance.get("usable_wall_segments"),
            fallback=[f"wall_segment_{index}" for index in range(1, wall_count + 1)],
        ),
        "entry_landing_zones": _region_labels(
            affordance.get("entry_landing_zones"),
            fallback=_opening_labels(doors, "entry_landing_zone"),
        ),
        "primary_circulation_corridors": _region_labels(
            affordance.get("primary_circulation_corridors"),
            fallback=["entry_to_center_corridor"] if doors else ["center_access_lane"],
        ),
        "daylight_regions": _region_labels(
            affordance.get("daylight_regions"),
            fallback=_opening_labels(windows, "daylight_region"),
        ),
        "privacy_regions": _region_labels(
            affordance.get("privacy_regions"),
            fallback=["private_back_zone"] if doors else ["quiet_wall_zone"],
        ),
        "focal_surfaces": _region_labels(
            affordance.get("focal_surfaces"),
            fallback=["primary_focal_wall"] if wall_count else [],
        ),
        "center_openness_regions": _region_labels(
            affordance.get("center_openness_regions"),
            fallback=["room_center"],
        ),
        "wall_anchor_candidates": _region_labels(
            affordance.get("wall_anchor_candidates"),
            fallback=["quiet_wall_zone", "long_wall_zone"] if wall_count else [],
        ),
        "floating_zone_candidates": _region_labels(
            affordance.get("floating_zone_candidates"),
            fallback=["central_floating_zone"],
        ),
        "room_area_m2": round(_room_area_m2(room_model_json), 2),
    }


def semantic_program_to_cluster_forge_payload(
    semantic_program: Mapping[str, Any],
) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    for active_cluster in semantic_program.get("active_clusters") or []:
        if not isinstance(active_cluster, Mapping):
            continue
        cluster_id = str(active_cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        members = _objects_from_active_cluster(active_cluster)
        if not members:
            continue
        dominant_anchor_candidates = _dominant_anchor_candidates_from_active_cluster(
            active_cluster,
            members,
        )
        anchors = _anchors_from_active_cluster(
            active_cluster,
            members,
            dominant_anchor_candidates=dominant_anchor_candidates,
        )
        rules = {
            "grid_mm": GLOBAL_LAYOUT_GRID_MM,
            "allowed_rotations": {member: [0, 90, 180, 270] for member in members},
            "facing": {
                member: {"front": "top", "notes": "functional front"}
                for member in members
                if _needs_front_access(member)
            },
            "access_requirements": [
                {"id": member, "type": "front_clearance", "required": True}
                for member in members
                if _needs_front_access(member)
            ],
            "semantic_placements": [],
            "dominant_anchor_candidates": dominant_anchor_candidates,
            "allow_empty_cluster": active_cluster.get("priority") != "core",
            "zone_claims": active_cluster.get("zone_claims") or {},
            "degradation_ladder": active_cluster.get("degradation_ladder") or [],
        }
        clusters.append(
            {
                "cluster_id": cluster_id,
                "tag": _cluster_tag(cluster_id, members),
                "members": members,
                "anchors": anchors,
                "hard_constraints": _hard_constraints_for_members(members),
                "soft_constraints": _soft_constraints_for_members(members, anchors),
                "cluster_rules": rules,
                "notes": [str(active_cluster.get("activation_reason") or "").strip()],
            }
        )
    status = "OK" if semantic_program.get("status") != "UNSAT" else "UNSAT"
    return {
        "status": status,
        "clusters": clusters,
        "semantic_layout_program": dict(semantic_program),
        "notes": list(semantic_program.get("notes") or []),
        "missing": list(semantic_program.get("missing") or []),
    }


def _room_rules_for(
    *,
    room_type: str,
    semantic_program_rules: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if semantic_program_rules is not None:
        if semantic_program_rules.get("room_type") == room_type:
            return dict(semantic_program_rules)
        rooms = semantic_program_rules.get("rooms")
        if isinstance(rooms, list):
            for item in rooms:
                if isinstance(item, Mapping) and item.get("room_type") == room_type:
                    return dict(item)
    return get_compiled_semantic_room_rule(room_type) or {
        "room_type": room_type,
        "policy": {"selection_policy": "fallback"},
        "clusters": [],
        "global_program": {},
    }


def _normalize_inventory_catalog(
    inventory_catalog: Sequence[Mapping[str, Any]] | None,
    room_rules: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if inventory_catalog is not None:
        for item in inventory_catalog:
            object_type = _clean_object_type(
                item.get("object_type") or item.get("type") or item.get("category")
            )
            if object_type is None:
                continue
            out[object_type] = {
                "object_type": object_type,
                "available": bool(item.get("available", True)),
                "size_profiles": _string_list(item.get("size_profiles")),
                "functional_tags": _string_list(item.get("functional_tags")),
            }
    if out:
        return out

    for object_type in _all_rule_object_types(room_rules):
        out[object_type] = {
            "object_type": object_type,
            "available": True,
            "size_profiles": ["S", "M", "L"],
            "functional_tags": [],
        }
    return out


def _build_single_candidate(
    *,
    cluster_rule: Mapping[str, Any],
    inventory_catalog: Mapping[str, Mapping[str, Any]],
    brief_text: str,
    room_model_json: Mapping[str, Any],
) -> dict[str, Any] | None:
    cluster_id = str(cluster_rule.get("cluster_id") or "").strip()
    priority = _priority(cluster_rule.get("priority"))
    if not cluster_id:
        return None
    object_program = _merged_object_program_for_brief(cluster_rule, brief_text)
    objects, missing = _candidate_objects(
        object_program=object_program,
        cluster_rule=cluster_rule,
        inventory_catalog=inventory_catalog,
        priority=priority,
        brief_text=brief_text,
    )
    activation = (
        cluster_rule.get("activation")
        if isinstance(cluster_rule.get("activation"), Mapping)
        else {}
    )
    active_by_rule = bool(activation.get("always_consider")) or priority == "core"
    brief_support = _brief_support_score(cluster_id, objects, brief_text)
    room_area = _room_area_m2(room_model_json)
    useful = (
        active_by_rule
        or brief_support >= 0.45
        or (priority == "support" and room_area >= 12.0 and objects)
    )
    return {
        "cluster_id": cluster_id,
        "priority": priority,
        "activation": dict(activation),
        "active_by_rule": active_by_rule,
        "useful": useful,
        "object_program": object_program,
        "objects": objects,
        "missing": missing,
        "semantic": dict(cluster_rule.get("semantic") or {})
        if isinstance(cluster_rule.get("semantic"), Mapping)
        else {},
        "degradation_hints": dict(cluster_rule.get("degradation_hints") or {})
        if isinstance(cluster_rule.get("degradation_hints"), Mapping)
        else {},
        "viability": {
            "rule_support": 1.0,
            "brief_support": brief_support,
            "inventory_support": 1.0 if not missing else 0.45,
        },
    }


def _build_deterministic_program(
    *,
    room_type: str,
    room_rules: Mapping[str, Any],
    room_model_json: Mapping[str, Any],
    affordance_summary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    brief_text: str,
    inventory_catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _ = inventory_catalog
    active_clusters: list[dict[str, Any]] = []
    missing: list[str] = []
    conflicts: list[str] = []
    for candidate in candidates:
        candidate_missing = _string_list(candidate.get("missing"))
        if candidate.get("priority") == "core" and candidate_missing:
            missing.extend(candidate_missing)
            continue
        if not candidate.get("useful"):
            continue
        active_clusters.append(
            _active_cluster_from_candidate(
                candidate=candidate,
                affordance_summary=affordance_summary,
                brief_text=brief_text,
            )
        )

    global_program = room_rules.get("global_program")
    global_program = global_program if isinstance(global_program, Mapping) else {}
    active_clusters = _enforce_dominant_requirements(
        active_clusters,
        candidates=candidates,
        global_program=global_program,
        missing=missing,
    )
    active_clusters = _enforce_group_caps(active_clusters, global_program)
    active_clusters = active_clusters[:_CLUSTER_CANDIDATE_CAP]

    macro_relations = _build_macro_relations(
        active_clusters=active_clusters,
        affordance_summary=affordance_summary,
    )
    controlled_degradation = _build_controlled_degradation(
        active_clusters=active_clusters,
        global_program=global_program,
    )
    status = "OK"
    if missing:
        status = "UNSAT" if not active_clusters else "NEEDS_REVIEW"
    return {
        "status": status,
        "room_type": room_type,
        "active_clusters": active_clusters,
        "global_layout_intent": _global_layout_intent(
            room_rules=room_rules,
            active_clusters=active_clusters,
            brief_text=brief_text,
        ),
        "macro_relations": macro_relations,
        "selection_constraints": {
            "dominant_anchor_required": _string_list(
                global_program.get("dominant_anchor_required")
            ),
            "dominant_workflow_required": _string_list(
                global_program.get("dominant_workflow_required")
            ),
            "group_caps": list(global_program.get("group_caps") or []),
            "group_minimums": list(global_program.get("group_minimums") or []),
        },
        "controlled_degradation": controlled_degradation,
        "quality_targets": {
            "functionality_weight": 1.0,
            "naturalness_weight": 1.0,
            "semantic_coherence_weight": 1.0,
            "spatial_quality_weight": 1.0,
        },
        "missing": _uniq(missing),
        "conflicts": _uniq(conflicts),
        "confidence": _confidence(active_clusters, missing),
        "notes": [
            "SemanticLayoutPlanner used compiled_semantic_program.json as the canonical rule source.",
            "Planner output is cluster/bundle/zone level only; pose is left to downstream composer and solver.",
        ],
    }


def _validate_and_normalize_llm_program(
    *,
    llm_payload: Mapping[str, Any],
    deterministic_program: Mapping[str, Any],
    room_type: str,
    room_rules: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    _ = room_rules
    allowed_candidates = {str(row.get("cluster_id")): row for row in candidates}
    deterministic_by_id = {
        str(row.get("cluster_id")): row
        for row in deterministic_program.get("active_clusters") or []
        if isinstance(row, Mapping)
    }
    raw_clusters = llm_payload.get("active_clusters")
    active_clusters: list[dict[str, Any]] = []
    if isinstance(raw_clusters, list):
        for raw_cluster in raw_clusters:
            if not isinstance(raw_cluster, Mapping):
                continue
            cluster_id = str(raw_cluster.get("cluster_id") or "").strip()
            if cluster_id not in allowed_candidates:
                continue
            base = deterministic_by_id.get(cluster_id)
            if base is None:
                continue
            merged = dict(base)
            if isinstance(raw_cluster.get("semantic_role"), str):
                merged["semantic_role"] = str(raw_cluster["semantic_role"]).strip()
            if isinstance(raw_cluster.get("zone_claims"), Mapping):
                merged["zone_claims"] = _normalize_zone_claims(
                    raw_cluster.get("zone_claims"),
                    fallback=base.get("zone_claims"),
                    affordance_summary=affordance_summary,
                )
            if isinstance(raw_cluster.get("relation_intents"), list):
                merged["relation_intents"] = _normalize_relation_intents(
                    raw_cluster.get("relation_intents"),
                    allowed_cluster_ids=set(allowed_candidates),
                )
            if isinstance(raw_cluster.get("degradation_ladder"), list):
                merged["degradation_ladder"] = _string_list(
                    raw_cluster.get("degradation_ladder")
                )[:_DEGRADATION_STEPS_CAP_PER_CLUSTER]
            active_clusters.append(merged)
    if not active_clusters:
        return dict(deterministic_program)

    out = dict(deterministic_program)
    out["room_type"] = room_type
    out["active_clusters"] = active_clusters
    out["macro_relations"] = _build_macro_relations(
        active_clusters=active_clusters,
        affordance_summary=affordance_summary,
    )
    out["controlled_degradation"] = _build_controlled_degradation(
        active_clusters=active_clusters,
        global_program=out.get("selection_constraints") or {},
    )
    out["confidence"] = max(float(out.get("confidence") or 0.75), 0.82)
    return out


def _active_cluster_from_candidate(
    *,
    candidate: Mapping[str, Any],
    affordance_summary: Mapping[str, Any],
    brief_text: str,
) -> dict[str, Any]:
    cluster_id = str(candidate.get("cluster_id") or "")
    priority = _priority(candidate.get("priority"))
    objects = [
        dict(item)
        for item in candidate.get("objects") or []
        if isinstance(item, Mapping)
    ]
    zone_claims = _zone_claims_for(
        cluster_id=cluster_id,
        objects=objects,
        affordance_summary=affordance_summary,
    )
    relation_intents = _relation_intents_for(cluster_id, objects)
    viability = _semantic_viability(
        candidate=candidate,
        zone_claims=zone_claims,
        brief_text=brief_text,
    )
    return {
        "cluster_id": cluster_id,
        "priority": priority,
        "activation_reason": _activation_reason(candidate),
        "semantic_role": _semantic_role(cluster_id, objects, priority),
        "dominant_anchor_candidates": _string_list(
            (candidate.get("semantic") or {}).get("dominant_anchor_candidates")
            if isinstance(candidate.get("semantic"), Mapping)
            else []
        ),
        "required_bundles": [
            {
                "bundle_id": f"{cluster_id}_bundle",
                "objects": objects,
            }
        ],
        "zone_claims": zone_claims,
        "relation_intents": relation_intents[:_RELATION_INTENT_CAP_PER_CLUSTER],
        "degradation_ladder": _degradation_ladder(candidate),
        "viability_score": viability,
    }


def _candidate_objects(
    *,
    object_program: Mapping[str, Any],
    cluster_rule: Mapping[str, Any],
    inventory_catalog: Mapping[str, Mapping[str, Any]],
    priority: str,
    brief_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    objects: list[dict[str, Any]] = []
    missing: list[str] = []

    def add_object(object_type: str, *, role: str, required: bool) -> None:
        if object_type in {str(row.get("object_type")) for row in objects}:
            return
        if not _is_available(object_type, inventory_catalog):
            if required:
                missing.append(object_type)
            return
        objects.append(
            {
                "object_type": object_type,
                "role": role,
                "required": required,
                "max_keep": _max_keep_for_object(object_type, object_program),
            }
        )

    dominant_candidates = _string_list(
        (cluster_rule.get("semantic") or {}).get("dominant_anchor_candidates")
        if isinstance(cluster_rule.get("semantic"), Mapping)
        else []
    )
    for object_type in _string_list(object_program.get("required")):
        add_object(
            object_type,
            role="dominant_anchor" if object_type in dominant_candidates else "support",
            required=True,
        )
    for group in _list_of_string_lists(object_program.get("choose_exactly_one_from")):
        selected = _select_best_object(group, inventory_catalog, brief_text)
        if selected is None:
            missing.extend(group)
        else:
            add_object(selected, role="dominant_anchor", required=True)
    for group in _list_of_string_lists(object_program.get("choose_at_least_one_from")):
        selected = _select_best_object(group, inventory_catalog, brief_text)
        if selected is None:
            missing.extend(group)
        else:
            add_object(selected, role="dominant_anchor", required=True)
    kept_required_if = (
        priority == "core"
        or _brief_support_score(
            str(cluster_rule.get("cluster_id") or ""), [], brief_text
        )
        >= 0.45
    )
    if kept_required_if:
        for object_type in _string_list(object_program.get("required_if_kept")):
            add_object(object_type, role="workflow_anchor", required=True)
        for group in _list_of_string_lists(
            object_program.get("choose_exactly_one_from_if_kept")
        ):
            selected = _select_best_object(group, inventory_catalog, brief_text)
            if selected is not None:
                add_object(selected, role="support", required=True)

    optional = _string_list(object_program.get("optional"))
    optional_limit = _global_optional_limit(object_program)
    for object_type in optional[:optional_limit]:
        add_object(object_type, role=_support_role(object_type), required=False)

    return objects, _uniq(missing)


def _merged_object_program_for_brief(
    cluster_rule: Mapping[str, Any],
    brief_text: str,
) -> dict[str, Any]:
    base = dict(cluster_rule.get("object_program") or {})
    activation = cluster_rule.get("activation")
    conditions = activation.get("conditions") if isinstance(activation, Mapping) else []
    if not isinstance(conditions, list):
        return base
    for condition in conditions:
        if not isinstance(condition, Mapping):
            continue
        predicate = str(condition.get("predicate") or "")
        if not _predicate_supported(predicate, brief_text):
            continue
        effects = condition.get("effects")
        if isinstance(effects, Mapping):
            base = _merge_object_programs(base, effects)
    return base


def _merge_object_programs(
    base: Mapping[str, Any], effects: Mapping[str, Any]
) -> dict[str, Any]:
    out = dict(base)
    for key in (
        "required",
        "required_if_kept",
        "choose_exactly_one_from",
        "choose_exactly_one_from_if_kept",
        "choose_at_least_one_from",
        "optional",
    ):
        if key not in effects:
            continue
        if key.startswith("choose_"):
            existing = _list_of_string_lists(out.get(key))
            incoming = effects.get(key)
            if isinstance(incoming, list) and incoming:
                existing.extend(_list_of_string_lists(incoming))
            out[key] = existing
        else:
            out[key] = _uniq(
                _string_list(out.get(key)) + _string_list(effects.get(key))
            )
    limits = effects.get("optional_limits")
    if isinstance(limits, Mapping):
        out["optional_limits"] = dict(limits)
    return out


def _zone_claims_for(
    *,
    cluster_id: str,
    objects: Sequence[Mapping[str, Any]],
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    object_types = [str(row.get("object_type") or "") for row in objects]
    lowered = f"{cluster_id} {' '.join(object_types)}".lower()
    entry = _cap_regions(affordance_summary.get("entry_landing_zones"))
    daylight = _cap_regions(affordance_summary.get("daylight_regions"))
    privacy = _cap_regions(affordance_summary.get("privacy_regions"))
    walls = _cap_regions(affordance_summary.get("wall_anchor_candidates"))
    focal = _cap_regions(affordance_summary.get("focal_surfaces"))
    center = _cap_regions(affordance_summary.get("center_openness_regions"))
    corridors = _cap_regions(affordance_summary.get("primary_circulation_corridors"))

    if "sleep" in lowered or any(is_bed_like(obj) for obj in object_types):
        return {
            "preferred_regions": _cap_regions(privacy + walls),
            "avoid_regions": _cap_regions(entry + corridors),
            "wall_affinity": "high",
            "daylight_affinity": "medium",
            "privacy_affinity": "high",
            "floating_allowed": False,
        }
    if "work" in lowered or any(is_work_surface_like(obj) for obj in object_types):
        return {
            "preferred_regions": _cap_regions(daylight + walls),
            "avoid_regions": _cap_regions(entry + center),
            "wall_affinity": "high",
            "daylight_affinity": "high",
            "privacy_affinity": "medium",
            "floating_allowed": False,
        }
    if "media" in lowered or any(
        _contains_any(obj, _MEDIA_TOKENS) for obj in object_types
    ):
        return {
            "preferred_regions": _cap_regions(focal + walls),
            "avoid_regions": _cap_regions(daylight + entry),
            "wall_affinity": "high",
            "daylight_affinity": "low",
            "privacy_affinity": "none",
            "floating_allowed": False,
        }
    if "storage" in lowered or any(_is_storage_like(obj) for obj in object_types):
        return {
            "preferred_regions": _cap_regions(walls + privacy),
            "avoid_regions": _cap_regions(center + corridors + entry),
            "wall_affinity": "high",
            "daylight_affinity": "low",
            "privacy_affinity": "medium",
            "floating_allowed": False,
        }
    if "dining" in lowered:
        return {
            "preferred_regions": _cap_regions(
                affordance_summary.get("floating_zone_candidates")
            ),
            "avoid_regions": _cap_regions(entry + corridors),
            "wall_affinity": "medium",
            "daylight_affinity": "medium",
            "privacy_affinity": "none",
            "floating_allowed": True,
        }
    return {
        "preferred_regions": _cap_regions(focal + center + walls),
        "avoid_regions": _cap_regions(entry + corridors),
        "wall_affinity": "medium",
        "daylight_affinity": "medium",
        "privacy_affinity": "low",
        "floating_allowed": True,
    }


def _relation_intents_for(
    cluster_id: str, objects: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    object_types = [str(row.get("object_type") or "") for row in objects]
    intents: list[dict[str, Any]] = []
    if any(str(row.get("role")) == "dominant_anchor" for row in objects):
        intents.append({"type": "dominance", "target": "room", "strength": "hard"})
    if any(is_bed_like(obj) for obj in object_types):
        intents.append(
            {"type": "separate", "target_cluster": "work_study", "strength": "soft"}
        )
    if any(_contains_any(obj, _MEDIA_TOKENS) for obj in object_types):
        intents.append(
            {"type": "face", "target_cluster": "main_seating", "strength": "soft"}
        )
    if "work" in cluster_id:
        intents.append(
            {"type": "claim_daylight", "target": "daylight", "strength": "soft"}
        )
    return intents


def _build_macro_relations(
    *,
    active_clusters: Sequence[Mapping[str, Any]],
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    adjacency: list[dict[str, object]] = []
    separation: list[dict[str, object]] = []
    orientation: list[dict[str, object]] = []
    by_id = {str(row.get("cluster_id")): row for row in active_clusters}
    seating = _first_cluster_matching(active_clusters, _cluster_has_seating)
    media = _first_cluster_matching(active_clusters, _cluster_has_media)
    sleep = _first_cluster_matching(active_clusters, _cluster_has_sleep)
    work = _first_cluster_matching(active_clusters, _cluster_has_work)
    if seating and media:
        adjacency.append(
            {"a": seating, "b": media, "relation": "near", "priority": "high"}
        )
        orientation.append(
            {"a": seating, "b": media, "relation": "face", "priority": "high"}
        )
    if sleep and work and sleep in by_id and work in by_id:
        separation.append(
            {"a": sleep, "b": work, "relation": "separate", "priority": "medium"}
        )
    return {
        "adjacency_preferences": adjacency[:_GLOBAL_RELATION_CAP],
        "separation_preferences": separation[:_GLOBAL_RELATION_CAP],
        "orientation_preferences": orientation[:_GLOBAL_RELATION_CAP],
        "keep_open_regions": [
            {"region": region, "reason": "preserve open center and circulation"}
            for region in _cap_regions(
                affordance_summary.get("center_openness_regions")
            )
        ],
        "reserved_regions": [
            {"region": region, "reason": "entry and primary circulation stay clear"}
            for region in _cap_regions(
                _cap_regions(affordance_summary.get("entry_landing_zones"))
                + _cap_regions(affordance_summary.get("primary_circulation_corridors"))
            )
        ],
    }


def _build_controlled_degradation(
    *,
    active_clusters: Sequence[Mapping[str, Any]],
    global_program: Mapping[str, Any],
) -> dict[str, Any]:
    cluster_rank = {"optional": 0, "support": 1, "core": 2}
    ordered = sorted(
        active_clusters,
        key=lambda row: (
            cluster_rank.get(str(row.get("priority") or "support"), 1),
            str(row.get("cluster_id") or ""),
        ),
    )
    bundle_drop_order: list[str] = []
    never_drop_first = _string_list(global_program.get("dominant_anchor_required"))
    for cluster in ordered:
        for bundle in cluster.get("required_bundles") or []:
            if isinstance(bundle, Mapping):
                bundle_id = str(bundle.get("bundle_id") or "").strip()
                if bundle_id:
                    bundle_drop_order.append(bundle_id)
    return {
        "cluster_drop_order": [str(row.get("cluster_id")) for row in ordered],
        "bundle_drop_order": bundle_drop_order,
        "never_drop_first": never_drop_first,
    }


def _degradation_ladder(candidate: Mapping[str, Any]) -> list[str]:
    hints = candidate.get("degradation_hints")
    hints = hints if isinstance(hints, Mapping) else {}
    steps: list[str] = []
    if _string_list(hints.get("shrink_before_drop")):
        steps.append("shrink_optional_support")
    for object_type in _string_list(hints.get("drop_first")):
        steps.append(f"drop_{object_type}")
    for object_type in _string_list(hints.get("shrink_before_drop")):
        steps.append(f"reduce_{object_type}_count")
    if not steps:
        steps = ["shrink_optional_support", "drop_secondary_support"]
    return _uniq(steps)[:_DEGRADATION_STEPS_CAP_PER_CLUSTER]


def _semantic_viability(
    *,
    candidate: Mapping[str, Any],
    zone_claims: Mapping[str, Any],
    brief_text: str,
) -> dict[str, Any]:
    affordance_support = 1.0 if zone_claims.get("preferred_regions") else 0.55
    inventory_support = float(
        (candidate.get("viability") or {}).get("inventory_support", 1.0)
        if isinstance(candidate.get("viability"), Mapping)
        else 1.0
    )
    brief_support = _brief_support_score(
        str(candidate.get("cluster_id") or ""),
        candidate.get("objects") if isinstance(candidate.get("objects"), list) else [],
        brief_text,
    )
    conflict_penalty = 0.15 if not zone_claims.get("avoid_regions") else 0.0
    score = max(
        0.0,
        min(
            1.0,
            (0.3 * 1.0)
            + (0.25 * affordance_support)
            + (0.2 * max(brief_support, 0.35))
            + (0.25 * inventory_support)
            - conflict_penalty,
        ),
    )
    return {
        "rule_support": 1.0,
        "affordance_support": round(affordance_support, 3),
        "brief_support": round(max(brief_support, 0.35), 3),
        "inventory_support": round(inventory_support, 3),
        "conflict_penalty": round(conflict_penalty, 3),
        "score": round(score, 3),
    }


def _enforce_dominant_requirements(
    active_clusters: list[dict[str, Any]],
    *,
    candidates: Sequence[Mapping[str, Any]],
    global_program: Mapping[str, Any],
    missing: list[str],
) -> list[dict[str, Any]]:
    required = _string_list(global_program.get("dominant_anchor_required"))
    if not required:
        return active_clusters
    active_objects = {
        str(obj.get("object_type"))
        for cluster in active_clusters
        for bundle in cluster.get("required_bundles") or []
        if isinstance(bundle, Mapping)
        for obj in bundle.get("objects") or []
        if isinstance(obj, Mapping)
    }
    for object_type in required:
        if object_type in active_objects:
            continue
        candidate = _candidate_containing_object(candidates, object_type)
        if candidate is None:
            missing.append(object_type)
            continue
        active_clusters.append(
            _active_cluster_from_candidate(
                candidate=candidate,
                affordance_summary={
                    "privacy_regions": ["private_back_zone"],
                    "wall_anchor_candidates": ["quiet_wall_zone"],
                    "entry_landing_zones": ["entry_landing_zone"],
                    "primary_circulation_corridors": ["center_access_lane"],
                    "center_openness_regions": ["room_center"],
                },
                brief_text="",
            )
        )
    return active_clusters


def _enforce_group_caps(
    active_clusters: list[dict[str, Any]], global_program: Mapping[str, Any]
) -> list[dict[str, Any]]:
    caps = global_program.get("group_caps")
    if not isinstance(caps, list):
        return active_clusters
    for cap in caps:
        if not isinstance(cap, Mapping):
            continue
        capped = set(_string_list(cap.get("objects")))
        max_keep = _int_value(cap.get("max_keep"), default=len(capped))
        if not capped or max_keep < 0:
            continue
        kept = 0
        for cluster in active_clusters:
            for bundle in cluster.get("required_bundles") or []:
                if not isinstance(bundle, Mapping):
                    continue
                objects = bundle.get("objects")
                if not isinstance(objects, list):
                    continue
                filtered: list[dict[str, Any]] = []
                for obj in objects:
                    if not isinstance(obj, Mapping):
                        continue
                    object_type = str(obj.get("object_type") or "")
                    if object_type in capped:
                        kept += 1
                        if kept > max_keep:
                            continue
                    filtered.append(dict(obj))
                bundle["objects"] = filtered
    return active_clusters


def _global_layout_intent(
    *,
    room_rules: Mapping[str, Any],
    active_clusters: Sequence[Mapping[str, Any]],
    brief_text: str,
) -> dict[str, Any]:
    policy = (
        room_rules.get("policy")
        if isinstance(room_rules.get("policy"), Mapping)
        else {}
    )
    primary_focus = _primary_focus(active_clusters)
    space_character = str(policy.get("intent") or "").strip() or "balanced_functional"
    lowered = brief_text.lower()
    prefer_open_center = not any(
        token in lowered for token in ("dense", "maximal", "full")
    )
    return {
        "primary_focus": primary_focus,
        "space_character": _snake_token(space_character) or "balanced_functional",
        "prefer_open_center": prefer_open_center,
        "prefer_core_before_support": True,
        "prefer_clear_primary_circulation": True,
    }


def _objects_from_active_cluster(active_cluster: Mapping[str, Any]) -> list[str]:
    objects: list[str] = []
    for bundle in active_cluster.get("required_bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        for obj in bundle.get("objects") or []:
            if not isinstance(obj, Mapping):
                continue
            object_type = _clean_object_type(obj.get("object_type"))
            if object_type is not None and object_type not in objects:
                objects.append(object_type)
    return objects


def _anchors_from_active_cluster(
    active_cluster: Mapping[str, Any],
    members: Sequence[str],
    *,
    dominant_anchor_candidates: Sequence[str] | None = None,
) -> list[str]:
    for candidate in dominant_anchor_candidates or []:
        if candidate in members:
            return [candidate]
    for bundle in active_cluster.get("required_bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        for obj in bundle.get("objects") or []:
            if not isinstance(obj, Mapping):
                continue
            object_type = _clean_object_type(obj.get("object_type"))
            if object_type in members and obj.get("role") in {
                "dominant_anchor",
                "workflow_anchor",
            }:
                return [object_type]
    return [members[0]] if members else []


def _dominant_anchor_candidates_from_active_cluster(
    active_cluster: Mapping[str, Any],
    members: Sequence[str],
) -> list[str]:
    member_set = set(members)
    candidates = [
        candidate
        for candidate in _string_list(active_cluster.get("dominant_anchor_candidates"))
        if candidate in member_set
    ]
    if candidates:
        return candidates

    role_candidates: list[str] = []
    for bundle in active_cluster.get("required_bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        for obj in bundle.get("objects") or []:
            if not isinstance(obj, Mapping):
                continue
            object_type = _clean_object_type(obj.get("object_type"))
            if object_type in member_set and obj.get("role") == "dominant_anchor":
                role_candidates.append(object_type)
    return _uniq(role_candidates)


def _hard_constraints_for_members(members: Sequence[str]) -> list[dict[str, object]]:
    constraints: list[dict[str, object]] = []
    for idx, a in enumerate(members):
        for b in members[idx + 1 :]:
            constraints.append({"type": "no_overlap", "a": a, "b": b})
    for member in members:
        if _needs_front_access(member):
            constraints.append(
                {
                    "type": "requires_access",
                    "id": member,
                    "mode": "front_clearance",
                }
            )
    return constraints


def _soft_constraints_for_members(
    members: Sequence[str], anchors: Sequence[str]
) -> list[dict[str, object]]:
    if not anchors:
        return []
    anchor = anchors[0]
    out: list[dict[str, object]] = []
    for member in members:
        if member == anchor:
            continue
        out.append({"type": "prefer_near", "a": member, "b": anchor, "weight": 6})
    return out


def _normalize_zone_claims(
    value: Any,
    *,
    fallback: Any,
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    fallback = fallback if isinstance(fallback, Mapping) else {}
    value = value if isinstance(value, Mapping) else {}
    allowed_regions = set()
    for regions in affordance_summary.values():
        if isinstance(regions, list):
            allowed_regions.update(str(region) for region in regions)
    preferred = [
        region
        for region in _string_list(value.get("preferred_regions"))
        if region in allowed_regions
    ][:_ZONE_CLAIM_CAP_PER_CLUSTER]
    avoid = [
        region
        for region in _string_list(value.get("avoid_regions"))
        if region in allowed_regions
    ][:_ZONE_CLAIM_CAP_PER_CLUSTER]
    return {
        "preferred_regions": preferred
        or _string_list(fallback.get("preferred_regions"))[
            :_ZONE_CLAIM_CAP_PER_CLUSTER
        ],
        "avoid_regions": avoid
        or _string_list(fallback.get("avoid_regions"))[:_ZONE_CLAIM_CAP_PER_CLUSTER],
        "wall_affinity": _affinity(
            value.get("wall_affinity"), fallback.get("wall_affinity")
        ),
        "daylight_affinity": _affinity(
            value.get("daylight_affinity"), fallback.get("daylight_affinity")
        ),
        "privacy_affinity": _affinity(
            value.get("privacy_affinity"), fallback.get("privacy_affinity")
        ),
        "floating_allowed": bool(
            value.get("floating_allowed", fallback.get("floating_allowed", False))
        ),
    }


def _normalize_relation_intents(
    value: Any, *, allowed_cluster_ids: set[str]
) -> list[dict[str, Any]]:
    allowed_types = {
        "near",
        "separate",
        "face",
        "buffer",
        "claim_wall",
        "claim_daylight",
        "claim_privacy",
        "avoid_entry",
        "preserve_center",
        "dominance",
    }
    out: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, Mapping):
            continue
        intent_type = _snake_token(str(item.get("type") or ""))
        if intent_type not in allowed_types:
            continue
        target_cluster = str(item.get("target_cluster") or "").strip() or None
        if target_cluster is not None and target_cluster not in allowed_cluster_ids:
            continue
        strength = (
            "hard" if str(item.get("strength") or "").lower() == "hard" else "soft"
        )
        out.append(
            {
                "type": intent_type,
                "target": str(item.get("target") or "").strip() or None,
                "target_cluster": target_cluster,
                "strength": strength,
            }
        )
    return out[:_RELATION_INTENT_CAP_PER_CLUSTER]


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("SemanticLayoutPlanner returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("SemanticLayoutPlanner response must be a JSON object")
    return payload


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("OpenAI response missing message content")


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


def _all_rule_object_types(room_rules: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    clusters = room_rules.get("clusters")
    if not isinstance(clusters, list):
        return out
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        for program in (
            cluster.get("object_program"),
            *(condition.get("effects") for condition in _conditions(cluster)),
        ):
            if isinstance(program, Mapping):
                out.extend(_object_types_from_program(program))
    return _uniq(out)


def _object_types_from_program(program: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("required", "required_if_kept", "optional"):
        out.extend(_string_list(program.get(key)))
    for key in (
        "choose_exactly_one_from",
        "choose_exactly_one_from_if_kept",
        "choose_at_least_one_from",
    ):
        for group in _list_of_string_lists(program.get(key)):
            out.extend(group)
    return out


def _conditions(cluster: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    activation = cluster.get("activation")
    if not isinstance(activation, Mapping):
        return []
    conditions = activation.get("conditions")
    if not isinstance(conditions, list):
        return []
    return [condition for condition in conditions if isinstance(condition, Mapping)]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            text for item in value if (text := _clean_object_type(item)) is not None
        ]
    text = _clean_object_type(value)
    return [text] if text is not None else []


def _list_of_string_lists(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    if all(isinstance(item, str) for item in value):
        return [_string_list(value)]
    return [_string_list(item) for item in value if isinstance(item, list)]


def _clean_object_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _is_available(
    object_type: str, inventory_catalog: Mapping[str, Mapping[str, Any]]
) -> bool:
    item = inventory_catalog.get(object_type)
    if item is None:
        return False
    return bool(item.get("available", True))


def _select_best_object(
    options: Sequence[str],
    inventory_catalog: Mapping[str, Mapping[str, Any]],
    brief_text: str,
) -> str | None:
    available = [item for item in options if _is_available(item, inventory_catalog)]
    if not available:
        return None
    lowered = brief_text.lower()
    return max(
        available,
        key=lambda item: (
            1 if item.lower() in lowered else 0,
            _anchor_score(item),
            -available.index(item),
        ),
    )


def _anchor_score(object_type: str) -> int:
    if is_bed_like(object_type):
        return 100
    if is_lounge_anchor_like(object_type):
        return 90
    if is_work_surface_like(object_type):
        return 80
    if _is_storage_like(object_type):
        return 70
    if is_seat_like(object_type):
        return 50
    return 10


def _priority(value: Any) -> str:
    text = str(value or "support").strip().lower()
    if text in {"core", "support", "optional"}:
        return text
    return "support"


def _predicate_supported(predicate: str, brief_text: str) -> bool:
    text = brief_text.lower()
    key = predicate.lower()
    if "tv" in key or "media" in key:
        return any(token in text for token in _MEDIA_TOKENS)
    if "work" in key or "study" in key:
        return any(token in text for token in _WORK_TOKENS)
    if "pet" in key:
        return any(token in text for token in _PET_TOKENS)
    if "entry" in key:
        return any(token in text for token in _ENTRY_TOKENS)
    if "laundry" in key or "storage" in key:
        return any(token in text for token in _LAUNDRY_TOKENS)
    return False


def _brief_support_score(
    cluster_id: str,
    objects: Sequence[Any],
    brief_text: str,
) -> float:
    text = brief_text.lower()
    haystack = f"{cluster_id} " + " ".join(
        str(item.get("object_type") or item) if isinstance(item, Mapping) else str(item)
        for item in objects
    )
    tokens = set(_snake_token(haystack).split("_"))
    if not text:
        return 0.35
    score = 0.35
    if tokens & set(re.findall(r"[a-z0-9]+", text)):
        score += 0.25
    if any(token in text for token in _WORK_TOKENS) and "work" in tokens:
        score += 0.35
    if any(token in text for token in _READING_TOKENS) and (
        "lounge" in tokens or "reading" in tokens
    ):
        score += 0.3
    if any(token in text for token in _MEDIA_TOKENS) and (
        "media" in tokens or "tv" in tokens
    ):
        score += 0.35
    return min(score, 1.0)


def _max_keep_for_object(
    object_type: str, object_program: Mapping[str, Any]
) -> int | None:
    limits = object_program.get("optional_limits")
    if not isinstance(limits, Mapping):
        return None
    by_object = limits.get("by_object")
    if isinstance(by_object, Mapping) and object_type in by_object:
        return _int_value(by_object.get(object_type), default=1)
    return None


def _global_optional_limit(object_program: Mapping[str, Any]) -> int:
    optional = _string_list(object_program.get("optional"))
    limits = object_program.get("optional_limits")
    if isinstance(limits, Mapping):
        global_limit = limits.get("global")
        if global_limit is not None:
            return max(0, _int_value(global_limit, default=len(optional)))
    return len(optional)


def _support_role(object_type: str) -> str:
    if is_bedside_support_like(object_type) or is_bench_like(object_type):
        return "secondary_support"
    if "lamp" in object_type or "plant" in object_type or "decor" in object_type:
        return "decor"
    return "support"


def _semantic_role(
    cluster_id: str, objects: Sequence[Mapping[str, Any]], priority: str
) -> str:
    lowered = f"{cluster_id} {' '.join(str(row.get('object_type') or '') for row in objects)}".lower()
    if priority == "core" and ("sleep" in lowered or "bed" in lowered):
        return "primary_anchor_zone"
    if priority == "core":
        return "core_function_zone"
    if "work" in lowered:
        return "support_workflow_zone"
    if "storage" in lowered:
        return "support_storage_zone"
    return "secondary_support_zone"


def _activation_reason(candidate: Mapping[str, Any]) -> str:
    if candidate.get("active_by_rule"):
        return "required by compiled semantic rule and supported by inventory"
    if candidate.get("useful"):
        return (
            "activated because brief or room capacity makes this support cluster useful"
        )
    return "not activated"


def _primary_focus(active_clusters: Sequence[Mapping[str, Any]]) -> str:
    for cluster in active_clusters:
        if cluster.get("priority") != "core":
            continue
        cluster_id = str(cluster.get("cluster_id") or "").lower()
        if "sleep" in cluster_id:
            return "sleep"
        if "media" in cluster_id:
            return "media"
        if "seating" in cluster_id or "living" in cluster_id:
            return "living"
        if "work" in cluster_id:
            return "work"
    return "mixed"


def _cluster_tag(cluster_id: str, members: Sequence[str]) -> str:
    lowered = f"{cluster_id} {' '.join(members)}".lower()
    if "sleep" in lowered or any(is_bed_like(member) for member in members):
        return "sleep"
    if "dining" in lowered:
        return "dining"
    if (
        "seating" in lowered
        or "living" in lowered
        or any(_contains_any(member, _MEDIA_TOKENS) for member in members)
        or any(
            is_lounge_anchor_like(member) or is_seat_like(member) for member in members
        )
    ):
        return "living"
    if "work" in lowered or any(
        _is_work_anchor_surface_like(member) for member in members
    ):
        return "work"
    if "storage" in lowered or any(_is_storage_like(member) for member in members):
        return "storage"
    return "misc"


def _is_work_anchor_surface_like(object_type: str) -> bool:
    key = object_type.lower()
    if any(token in key for token in ("coffee_table", "side_table", "ottoman")):
        return False
    return is_work_surface_like(object_type)


def _is_storage_like(object_type: str) -> bool:
    key = object_type.lower()
    return any(
        token in key
        for token in ("wardrobe", "dresser", "cabinet", "bookshelf", "shelf", "storage")
    )


def _needs_front_access(object_type: str) -> bool:
    return (
        is_seat_like(object_type)
        or is_work_surface_like(object_type)
        or _is_storage_like(object_type)
        or _contains_any(object_type, _MEDIA_TOKENS)
        or is_bed_like(object_type)
    )


def _cluster_has_seating(cluster: Mapping[str, Any]) -> bool:
    return any(
        is_lounge_anchor_like(obj) or is_seat_like(obj)
        for obj in _objects_from_active_cluster(cluster)
    )


def _cluster_has_media(cluster: Mapping[str, Any]) -> bool:
    return any(
        _contains_any(obj, _MEDIA_TOKENS)
        for obj in _objects_from_active_cluster(cluster)
    )


def _cluster_has_sleep(cluster: Mapping[str, Any]) -> bool:
    return any(is_bed_like(obj) for obj in _objects_from_active_cluster(cluster))


def _cluster_has_work(cluster: Mapping[str, Any]) -> bool:
    return any(
        is_work_surface_like(obj) for obj in _objects_from_active_cluster(cluster)
    )


def _first_cluster_matching(
    active_clusters: Sequence[Mapping[str, Any]],
    predicate: Any,
) -> str | None:
    for cluster in active_clusters:
        if predicate(cluster):
            return str(cluster.get("cluster_id") or "")
    return None


def _candidate_containing_object(
    candidates: Sequence[Mapping[str, Any]], object_type: str
) -> Mapping[str, Any] | None:
    for candidate in candidates:
        for obj in candidate.get("objects") or []:
            if isinstance(obj, Mapping) and obj.get("object_type") == object_type:
                return candidate
    return None


def _room_area_m2(room_model_json: Mapping[str, Any]) -> float:
    room = (
        room_model_json.get("room")
        if isinstance(room_model_json.get("room"), Mapping)
        else {}
    )
    points = room.get("polygon_ccw") or room_model_json.get("polygon_mm") or []
    if not isinstance(points, list) or len(points) < 3:
        return 0.0
    clean: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, Mapping):
            clean.append((float(point.get("x") or 0.0), float(point.get("y") or 0.0)))
        elif isinstance(point, list) and len(point) >= 2:
            clean.append((float(point[0]), float(point[1])))
    if len(clean) < 3:
        return 0.0
    area2 = 0.0
    for idx, (x1, y1) in enumerate(clean):
        x2, y2 = clean[(idx + 1) % len(clean)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2_000_000.0


def _region_labels(value: Any, *, fallback: Sequence[str]) -> list[str]:
    labels = _cap_regions(value)
    return labels if labels else _uniq(list(fallback))[:_ZONE_CLAIM_CAP_PER_CLUSTER]


def _cap_regions(value: Any) -> list[str]:
    if isinstance(value, list):
        labels: list[str] = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, str) and item.strip():
                labels.append(_snake_token(item))
            elif isinstance(item, Mapping):
                label = (
                    item.get("id")
                    or item.get("label")
                    or item.get("name")
                    or f"region_{index}"
                )
                labels.append(_snake_token(str(label)))
        return _uniq(labels)[:_ZONE_CLAIM_CAP_PER_CLUSTER]
    return []


def _opening_labels(value: Sequence[Any], prefix: str) -> list[str]:
    labels: list[str] = []
    for index, item in enumerate(value, start=1):
        opening_id = None
        if isinstance(item, Mapping):
            opening_id = item.get("id") or item.get("door_id") or item.get("window_id")
        labels.append(_snake_token(f"{prefix}_{opening_id or index}"))
    return labels


def _affinity(value: Any, fallback: Any) -> str:
    text = str(value or fallback or "medium").strip().lower()
    if text in {"none", "low", "medium", "high"}:
        return text
    return "medium"


def _confidence(
    active_clusters: Sequence[Mapping[str, Any]], missing: Sequence[str]
) -> float:
    if missing:
        return 0.58 if active_clusters else 0.25
    if len(active_clusters) <= 1:
        return 0.72
    return 0.86


def _contains_any(value: str, tokens: Sequence[str]) -> bool:
    key = value.lower()
    return any(token in key for token in tokens)


def _snake_token(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _uniq(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _int_value(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(round(float(value.strip())))
        except ValueError:
            return default
    return default
