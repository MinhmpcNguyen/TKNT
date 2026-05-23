from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from agent.seed_concept_generator import _build_macro_region_map, _room_id, _room_type
from agent_schema.visual_relation_planner_schema import (
    VisualRelationPlannerOutput,
)
from config.openai_config import OpenAIConfig

_MODEL_ENV = "TKNT_VISUAL_RELATION_MODEL"
_MAX_TOKENS_ENV = "TKNT_VISUAL_RELATION_MAX_TOKENS"
_DEFAULT_MODEL = "gpt-5.4-mini"
_DEFAULT_MAX_TOKENS = 2200
_MAX_IMAGE_DATA_URL_CHARS = 4_500_000
_SUPPORTED_IMAGE_PREFIXES = (
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/jpg;base64,",
    "data:image/webp;base64,",
)
_NORMALIZABLE_ROOT_EXTRA_KEYS = frozenset(
    {
        "macro_region_intent",
        "object_zone_preferences",
    }
)
_NORMALIZABLE_CLUSTER_ZONE_EXTRA_KEYS = frozenset({"confidence"})
_ZONE_ASSIGNMENT_ALIASES = (
    "region_id",
    "macro_region_id",
    "preferred_region_id",
    "preferred_zone",
    "zone",
)
_SEMANTIC_WALL_BOUND_ROLES = frozenset({"anchor", "storage", "visual_focus"})
_SEMANTIC_REQUIRED_ROLES = frozenset({"storage", "visual_focus"})
_OUTPUT_CONTRACT: dict[str, object] = {
    "status": "OK",
    "concept_id": "visual_concept_01",
    "concept_family": "zoned",
    "primary_cluster_id": "one listed cluster_id",
    "secondary_cluster_id": None,
    "cluster_zone_plan": [
        {
            "cluster_id": "listed cluster_id",
            "zone_assignment": "listed macro region_id",
            "preferred_wall_side": "center|window_side|top_wall|right_wall|bottom_wall|left_wall",
            "center_usage": "none|partial|primary|open_reserved",
            "anchor_strength": "hard|strong|medium",
            "priority": "high|medium|low",
            "reason": "short visual reason",
        }
    ],
    "object_orientations": [],
    "object_region_plan": [
        {
            "cluster_id": "listed cluster_id",
            "object_id": "listed object_id",
            "region_id": "listed macro region_id",
            "wall_side": "center|window_side|top_wall|right_wall|bottom_wall|left_wall",
            "corner_hint": "none|top_left|top_right|bottom_left|bottom_right|near_window|far_from_entry",
            "semantic_anchor_region_id": "listed semantic anchor region_id or null for non-wall paired/support objects",
            "anchor_lock_strength": "hard|strong|soft",
            "placement_role": "anchor|support|functional_support|paired_seat|visual_focus|storage|decor",
            "facing_intent": "front_to_open_space|front_to_room_center|front_to_window|front_to_entry|back_to_wall|parallel_to_wall|face_target|face_away_from_target",
            "target_object_id": None,
            "target_object_cluster_id": None,
            "priority": "high|medium|low",
            "reason": "short visual reason",
        }
    ],
    "object_viewing_pairs": [],
    "cluster_relations": [],
    "notes": [],
    "missing": [],
}


class VisualJsonClientProtocol(Protocol):
    def create_json_response(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        max_tokens: int,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class VisualRelationPlannerResult:
    raw_output: dict[str, object]
    normalized_output: dict[str, object]
    normalization_notes: tuple[str, ...]
    output: VisualRelationPlannerOutput
    relation_plan: dict[str, object]
    macro_region_map: dict[str, object]
    semantic_anchor_catalog: tuple[dict[str, object], ...]
    model_name: str


@dataclass(frozen=True)
class VisualRelationPlanner:
    client: VisualJsonClientProtocol | None = None
    model_name: str | None = None
    max_tokens: int | None = None

    def generate(
        self,
        *,
        room_model_json: Mapping[str, object],
        merged_clusters_json: Mapping[str, object],
        floorplan_image_data_url: str,
        description: str | None = None,
    ) -> VisualRelationPlannerResult:
        image_data_url = validate_floorplan_image_data_url(floorplan_image_data_url)
        macro_region_map = _build_macro_region_map(room_model_json)
        region_ids = _available_macro_region_ids(macro_region_map)
        semantic_anchor_catalog = build_semantic_anchor_catalog(room_model_json)
        clusters = _cluster_contexts(merged_clusters_json)
        if not clusters:
            raise ValueError("visual relation planner requires at least one cluster")

        object_ids_by_cluster = {
            str(cluster["cluster_id"]): set(_string_sequence(cluster["object_ids"]))
            for cluster in clusters
        }
        cluster_ids = set(object_ids_by_cluster)
        prompt_context = {
            "room": _room_prompt_context(room_model_json),
            "macro_regions": [
                {
                    "region_id": region.region_id,
                    "region_type": region.region_type,
                    "label": region.label,
                    "tags": list(region.tags),
                }
                for region in region_ids
            ],
            "semantic_anchor_regions": [
                _semantic_anchor_prompt_payload(row)
                for row in semantic_anchor_catalog
            ],
            "clusters": clusters,
            "object_inventory": _object_inventory(clusters),
            "description": description or "",
            "output_rules": [
                "Return JSON only.",
                "Use every cluster_id exactly once in cluster_zone_plan.",
                "Map every listed object_id exactly once in object_region_plan when status is OK.",
                "Use only listed macro region_id values as zone_assignment.",
                "Use only listed macro region_id values as object_region_plan.region_id.",
                "Use only listed semantic_anchor_regions.anchor_region_id values as object_region_plan.semantic_anchor_region_id.",
                "Give every wall-bound anchor/storage/visual_focus object a semantic_anchor_region_id.",
                "Use anchor_lock_strength=hard for wall-bound storage, focal, and anchor objects.",
                "Use only listed object_id values. Do not invent furniture.",
                "Do not return coordinates, pixels, mm positions, bounding boxes, or rotations.",
                "Do not return confidence, macro_region_intent, object_zone_preferences, or other non-contract fields.",
                "For desk/chair, make chair a paired_seat or functional_support targeting desk when both exist.",
                "For wardrobe/bookshelf and bed/nightstand, target the support object to its main anchor when both exist.",
                "Return notes as a JSON array, never as a string.",
                "Return missing as an empty JSON array when status is OK.",
            ],
        }
        model_name = self.model_name or os.getenv(_MODEL_ENV, _DEFAULT_MODEL).strip()
        max_tokens = self.max_tokens or _max_tokens_from_env()
        raw_output = dict(
            (self.client or OpenAIVisionJsonClient()).create_json_response(
                model_name=model_name,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(prompt_context),
                image_data_url=image_data_url,
                max_tokens=max_tokens,
            )
        )
        normalized_output, normalization_notes = normalize_visual_relation_payload(
            raw_output,
            available_cluster_ids=cluster_ids,
        )
        output = validate_visual_relation_output(
            normalized_output,
            available_cluster_ids=cluster_ids,
            object_ids_by_cluster=object_ids_by_cluster,
            available_region_ids={region.region_id for region in region_ids},
            available_semantic_anchor_region_ids={
                str(row["anchor_region_id"]) for row in semantic_anchor_catalog
            },
            available_semantic_anchor_regions={
                str(row["anchor_region_id"]): row for row in semantic_anchor_catalog
            },
        )
        relation_plan = visual_relation_plan_from_output(
            output=output,
            room_model_json=room_model_json,
            macro_region_map=macro_region_map,
            semantic_anchor_catalog=semantic_anchor_catalog,
        )
        return VisualRelationPlannerResult(
            raw_output=raw_output,
            normalized_output=normalized_output,
            normalization_notes=normalization_notes,
            output=output,
            relation_plan=relation_plan,
            macro_region_map=macro_region_map,
            semantic_anchor_catalog=semantic_anchor_catalog,
            model_name=model_name,
        )


@dataclass(frozen=True)
class MacroRegionRef:
    region_id: str
    region_type: str
    label: str
    tags: tuple[str, ...]


class OpenAIVisionJsonClient:
    def create_json_response(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        max_tokens: int,
    ) -> Mapping[str, object]:
        client = self._build_client()
        params: dict[str, object] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }
        params[_max_tokens_param_name(model_name)] = max_tokens
        response = client.chat.completions.create(**params)
        content = _extract_chat_completion_text(response)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("visual relation planner returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("visual relation planner response must be a JSON object")
        return payload

    @staticmethod
    def _build_client() -> object:
        if not OpenAIConfig.OPENAI_API_KEY:
            raise ValueError("Missing OpenAI API key for visual relation planner")
        if OpenAIConfig.IS_OPENAI_AZURE:
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=OpenAIConfig.OPENAI_API_KEY,
                azure_endpoint=OpenAIConfig.BASE_URL,
                api_version=OpenAIConfig.API_VERSION,
            )
        from openai import OpenAI

        if OpenAIConfig.BASE_URL:
            return OpenAI(
                api_key=OpenAIConfig.OPENAI_API_KEY, base_url=OpenAIConfig.BASE_URL
            )
        return OpenAI(api_key=OpenAIConfig.OPENAI_API_KEY)


def validate_floorplan_image_data_url(value: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError("floorplan image data URL is empty")
    if len(clean) > _MAX_IMAGE_DATA_URL_CHARS:
        raise ValueError("floorplan image data URL exceeds visual planner size cap")
    lower = clean.lower()
    if not lower.startswith(_SUPPORTED_IMAGE_PREFIXES):
        raise ValueError("floorplan image must be a PNG, JPEG, or WEBP data URL")
    return clean


def normalize_visual_relation_payload(
    payload: Mapping[str, object],
    *,
    available_cluster_ids: set[str],
) -> tuple[dict[str, object], tuple[str, ...]]:
    normalized = dict(payload)
    notes: list[str] = []

    for key in _NORMALIZABLE_ROOT_EXTRA_KEYS:
        if key in normalized:
            normalized.pop(key)
            notes.append(f"Dropped unsupported root field `{key}`.")

    raw_notes = normalized.get("notes")
    if isinstance(raw_notes, str):
        normalized["notes"] = [raw_notes]
        notes.append("Converted string `notes` to a list.")

    cluster_zone_plan = normalized.get("cluster_zone_plan")
    if _is_non_string_sequence(cluster_zone_plan):
        normalized["cluster_zone_plan"] = _normalize_cluster_zone_plan(
            cluster_zone_plan,
            notes=notes,
        )
        if _clean_str(normalized.get("status")) is None:
            normalized["status"] = "OK"
            notes.append("Filled missing `status` as OK.")
        if _clean_str(normalized.get("primary_cluster_id")) is None:
            primary_cluster_id = _first_known_cluster_id(
                normalized["cluster_zone_plan"],
                available_cluster_ids=available_cluster_ids,
            )
            if primary_cluster_id is not None:
                normalized["primary_cluster_id"] = primary_cluster_id
                notes.append("Inferred `primary_cluster_id` from cluster_zone_plan.")

    if _clean_str(normalized.get("status")) != "NEED_INFO" and normalized.get(
        "missing"
    ):
        normalized["missing"] = []
        notes.append("Cleared `missing` because status is not NEED_INFO.")

    return normalized, tuple(_unique(notes))


def validate_visual_relation_output(
    payload: Mapping[str, object],
    *,
    available_cluster_ids: set[str],
    object_ids_by_cluster: Mapping[str, set[str]],
    available_region_ids: set[str],
    available_semantic_anchor_region_ids: set[str] | None = None,
    available_semantic_anchor_regions: Mapping[str, Mapping[str, object]]
    | None = None,
) -> VisualRelationPlannerOutput:
    try:
        output = VisualRelationPlannerOutput.model_validate(dict(payload))
    except ValidationError as exc:
        raise ValueError(
            f"visual relation planner schema validation failed: {exc}"
        ) from exc

    if output.status != "OK":
        raise ValueError(f"visual relation planner status is {output.status}")
    if output.primary_cluster_id not in available_cluster_ids:
        raise ValueError(
            f"visual relation planner invented primary cluster: {output.primary_cluster_id}"
        )
    if (
        output.secondary_cluster_id is not None
        and output.secondary_cluster_id not in available_cluster_ids
    ):
        raise ValueError(
            "visual relation planner invented secondary cluster: "
            f"{output.secondary_cluster_id}"
        )

    planned_cluster_ids = {row.cluster_id for row in output.cluster_zone_plan}
    unknown_clusters = planned_cluster_ids - available_cluster_ids
    if unknown_clusters:
        raise ValueError(
            "visual relation planner invented cluster ids: "
            + ", ".join(sorted(unknown_clusters))
        )
    missing_clusters = available_cluster_ids - planned_cluster_ids
    if missing_clusters:
        raise ValueError(
            "visual relation planner omitted cluster ids: "
            + ", ".join(sorted(missing_clusters))
        )

    available_visual_region_ids = set(available_region_ids)
    if available_semantic_anchor_region_ids is not None:
        available_visual_region_ids.update(available_semantic_anchor_region_ids)
    unknown_regions = {
        row.zone_assignment
        for row in output.cluster_zone_plan
        if row.zone_assignment not in available_visual_region_ids
    }
    unknown_regions.update(
        row.region_id
        for row in output.object_region_plan
        if row.region_id not in available_visual_region_ids
    )
    if unknown_regions:
        raise ValueError(
            "visual relation planner used unknown visual regions: "
            + ", ".join(sorted(unknown_regions))
        )

    for item in output.object_orientations:
        _validate_object_ref(
            cluster_id=item.cluster_id,
            object_id=item.object_id,
            object_ids_by_cluster=object_ids_by_cluster,
        )
        if item.target_object_id is not None:
            _validate_object_ref(
                cluster_id=item.target_object_cluster_id or item.cluster_id,
                object_id=item.target_object_id,
                object_ids_by_cluster=object_ids_by_cluster,
            )

    for item in output.object_region_plan:
        _validate_object_ref(
            cluster_id=item.cluster_id,
            object_id=item.object_id,
            object_ids_by_cluster=object_ids_by_cluster,
        )
        if (
            item.semantic_anchor_region_id is not None
            and available_semantic_anchor_region_ids is not None
            and item.semantic_anchor_region_id
            not in available_semantic_anchor_region_ids
        ):
            raise ValueError(
                "visual relation planner used unknown semantic anchor region: "
                f"{item.semantic_anchor_region_id}"
            )
        if (
            item.semantic_anchor_region_id is not None
            and available_semantic_anchor_regions is not None
        ):
            _validate_semantic_anchor_role(
                item=item.model_dump(),
                semantic_anchor=available_semantic_anchor_regions.get(
                    item.semantic_anchor_region_id, {}
                ),
            )
        if (
            available_semantic_anchor_region_ids is not None
            and _object_region_requires_semantic_anchor(item.model_dump())
            and item.semantic_anchor_region_id is None
        ):
            raise ValueError(
                "visual relation planner omitted semantic anchor region for "
                f"{item.cluster_id}/{item.object_id}"
            )
        if item.target_object_id is not None:
            _validate_object_ref(
                cluster_id=item.target_object_cluster_id or item.cluster_id,
                object_id=item.target_object_id,
                object_ids_by_cluster=object_ids_by_cluster,
            )

    for item in output.object_viewing_pairs:
        _validate_object_ref(
            cluster_id=item.viewer_cluster_id,
            object_id=item.viewer_object_id,
            object_ids_by_cluster=object_ids_by_cluster,
        )
        _validate_object_ref(
            cluster_id=item.target_cluster_id,
            object_id=item.target_object_id,
            object_ids_by_cluster=object_ids_by_cluster,
        )

    for item in output.cluster_relations:
        if item.a not in available_cluster_ids or item.b not in available_cluster_ids:
            raise ValueError(
                f"visual relation planner used unknown relation pair: {item.a}/{item.b}"
            )

    return output


def visual_relation_plan_from_output(
    *,
    output: VisualRelationPlannerOutput,
    room_model_json: Mapping[str, object],
    macro_region_map: Mapping[str, object],
    semantic_anchor_catalog: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    cluster_zone_plan = [_cluster_zone_payload(row) for row in output.cluster_zone_plan]
    anchor_preferences = {
        row["cluster_id"]: _anchor_preference(row) for row in cluster_zone_plan
    }
    anchor_hints = {
        row["cluster_id"]: {
            "zone_assignment": row["zone_assignment"],
            "preferred_region_ids": [row["zone_assignment"]],
            "preferred_wall_side": row["preferred_wall_side"],
            "center_usage": row["center_usage"],
            "anchor_strength": row["anchor_strength"],
            "placement_behavior": row["placement_behavior"],
        }
        for row in cluster_zone_plan
    }
    object_region_plan = [
        _object_region_payload(row) for row in output.object_region_plan
    ]
    semantic_anchor_preferences = _semantic_anchor_preferences_by_cluster(
        object_region_plan=object_region_plan,
        semantic_anchor_catalog=semantic_anchor_catalog,
    )
    _apply_semantic_anchor_hints(
        anchor_hints=anchor_hints,
        semantic_anchor_preferences=semantic_anchor_preferences,
    )
    pair_contracts = _anchor_pair_contracts(output)
    return {
        "status": "OK",
        "room_id": _room_id(room_model_json),
        "room_type": _room_type(room_model_json),
        "planner_kind": "visual_relation_planner",
        "macro_concept": {
            "concept_id": output.concept_id,
            "concept_family": output.concept_family,
            "cluster_zone_plan": cluster_zone_plan,
            "macro_region_map_snapshot": dict(macro_region_map),
            "semantic_anchor_regions": [
                _semantic_anchor_prompt_payload(row)
                for row in semantic_anchor_catalog
            ],
            "llm_guided": True,
            "guidance_notes": list(output.notes),
            "diversity_signature": _diversity_signature(output),
        },
        "cluster_affinities": [_cluster_affinity(row) for row in cluster_zone_plan],
        "cluster_orientations": _cluster_orientations(output, cluster_zone_plan),
        "object_orientations": [
            item.model_dump(exclude_none=True) for item in output.object_orientations
        ],
        "object_region_plan": object_region_plan,
        "object_region_preferences_by_cluster": (
            _object_region_preferences_by_cluster(object_region_plan)
        ),
        "semantic_anchor_preferences_by_cluster": semantic_anchor_preferences,
        "functional_pair_hints": _functional_pair_hints(output),
        "object_viewing_pairs": [
            item.model_dump(exclude_none=True) for item in output.object_viewing_pairs
        ],
        "anchor_region_preferences_by_cluster": anchor_preferences,
        "anchor_layout_hints_by_cluster": anchor_hints,
        "anchor_pair_contracts": pair_contracts,
        "object_solver_policy": {
            "enabled": True,
            "layout_flow": "object_level_anchor_first",
            "solve_level": "object",
            "anchor_first": True,
        },
        "macro_region_map": dict(macro_region_map),
        "semantic_anchor_region_catalog": [
            _semantic_anchor_prompt_payload(row)
            for row in semantic_anchor_catalog
        ],
        "allowed_variant_families_by_cluster": {
            row["cluster_id"]: _allowed_variant_families(row)
            for row in cluster_zone_plan
        },
        "variant_bias_weights": {"visual_relation_prior": 1.0},
        "concept_readiness_requirements": {
            "require_primary_pair_legibility": bool(pair_contracts),
        },
        "cluster_relations": [
            item.model_dump(exclude_none=True) for item in output.cluster_relations
        ],
        "cluster_directional_relations": _directional_relations(pair_contracts),
        "circulation_plan": {
            "main_paths": [],
            "keep_open_regions": [
                {
                    "type": "center_lane",
                    "near": "floating_center_zone",
                    "priority": "medium",
                    "reason": "Visual planner keeps circulation open while solver validates geometry.",
                }
            ],
        },
        "layout_intent_profile": {
            "focus_mode": "viewing" if pair_contracts else "mixed",
            "primary_cluster_id": output.primary_cluster_id,
            "secondary_cluster_id": output.secondary_cluster_id,
            "circulation_priority": "medium",
            "center_open_preference": "medium",
            "support_cluster_behavior": "balanced",
            "distribution_mode": "zoned",
        },
        "placement_guidelines": [
            "Use visual planner output as object-level wall, region, and facing intent.",
            "Solver must choose final coordinates and validate geometry.",
        ],
        "notes": [
            "Generated by experimental VisualRelationPlanner from 2D floorplan screenshot.",
            *output.notes,
        ],
        "missing": [],
    }


def _validate_object_ref(
    *,
    cluster_id: str,
    object_id: str,
    object_ids_by_cluster: Mapping[str, set[str]],
) -> None:
    if cluster_id not in object_ids_by_cluster:
        raise ValueError(f"visual relation planner used unknown cluster: {cluster_id}")
    if object_id not in object_ids_by_cluster[cluster_id]:
        raise ValueError(
            f"visual relation planner used unknown object: {cluster_id}/{object_id}"
        )


def _available_macro_region_ids(
    macro_region_map: Mapping[str, object],
) -> tuple[MacroRegionRef, ...]:
    rows = macro_region_map.get("regions")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return ()
    out: list[MacroRegionRef] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        region_id = _clean_str(row.get("region_id"))
        if region_id is None:
            continue
        out.append(
            MacroRegionRef(
                region_id=region_id,
                region_type=str(row.get("region_type") or ""),
                label=str(row.get("label") or region_id),
                tags=tuple(_string_sequence(row.get("tags"))),
            )
        )
    return tuple(out)


def _normalize_cluster_zone_plan(
    value: Sequence[object],
    *,
    notes: list[str],
) -> list[object]:
    rows: list[object] = []
    for row in value:
        if not isinstance(row, Mapping):
            rows.append(row)
            continue
        next_row = dict(row)
        _normalize_zone_assignment_alias(next_row, notes=notes)
        for key in _NORMALIZABLE_CLUSTER_ZONE_EXTRA_KEYS:
            if key in next_row:
                next_row.pop(key)
                notes.append(f"Dropped unsupported cluster_zone_plan field `{key}`.")
        if _clean_str(next_row.get("reason")) is None:
            cluster_id = _clean_str(next_row.get("cluster_id")) or "cluster"
            zone_assignment = _clean_str(next_row.get("zone_assignment")) or "zone"
            next_row["reason"] = _default_zone_reason(
                cluster_id=cluster_id,
                zone_assignment=zone_assignment,
            )
            notes.append("Filled missing cluster_zone_plan `reason`.")
        rows.append(next_row)
    return rows


def _normalize_zone_assignment_alias(
    row: dict[str, object],
    *,
    notes: list[str],
) -> None:
    if _clean_str(row.get("zone_assignment")) is not None:
        return
    for alias in _ZONE_ASSIGNMENT_ALIASES:
        value = _clean_str(row.get(alias))
        if value is None:
            continue
        row["zone_assignment"] = value
        notes.append(f"Mapped `{alias}` to `zone_assignment`.")
        return


def _default_zone_reason(*, cluster_id: str, zone_assignment: str) -> str:
    return (
        f"Visual floorplan intent placed `{cluster_id}` in `{zone_assignment}`; "
        "solver remains responsible for final geometry."
    )


def _first_known_cluster_id(
    value: object,
    *,
    available_cluster_ids: set[str],
) -> str | None:
    if not _is_non_string_sequence(value):
        return None
    for row in value:
        if not isinstance(row, Mapping):
            continue
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id in available_cluster_ids:
            return cluster_id
    return None


def _cluster_contexts(
    merged_clusters_json: Mapping[str, object],
) -> list[dict[str, object]]:
    clusters = merged_clusters_json.get("clusters")
    if not isinstance(clusters, Sequence) or isinstance(clusters, str):
        return []
    out: list[dict[str, object]] = []
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        cluster_id = _clean_str(cluster.get("cluster_id"))
        if cluster_id is None:
            continue
        object_ids = _object_ids_from_cluster(cluster)
        out.append(
            {
                "cluster_id": cluster_id,
                "tag": str(cluster.get("tag") or ""),
                "members": list(_string_sequence(cluster.get("members"))),
                "anchors": list(_string_sequence(cluster.get("anchors"))),
                "object_ids": object_ids,
                "decisions": _decision_contexts(cluster),
                "zone_claims": _zone_claims(cluster),
            }
        )
    return out


def _object_ids_from_cluster(cluster: Mapping[str, object]) -> list[str]:
    out: list[str] = []
    for item in _string_sequence(cluster.get("members")):
        if item not in out:
            out.append(item)
    object_program = cluster.get("object_program")
    if isinstance(object_program, Mapping):
        for item in _string_sequence(object_program.get("placement_order")):
            if item not in out:
                out.append(item)
        specs = object_program.get("object_specs")
        if isinstance(specs, Mapping):
            for item in specs:
                if isinstance(item, str) and item.strip() and item not in out:
                    out.append(item)
    for decision in _sequence_or_empty(cluster.get("decisions")):
        if not isinstance(decision, Mapping):
            continue
        object_id = _clean_str(decision.get("object_type") or decision.get("category"))
        quantity = int(decision.get("quantity") or 0)
        if object_id and quantity > 0 and object_id not in out:
            out.append(object_id)
    return out


def _object_inventory(
    clusters: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for cluster in clusters:
        cluster_id = _clean_str(cluster.get("cluster_id"))
        if cluster_id is None:
            continue
        for object_id in _string_sequence(cluster.get("object_ids")):
            out.append({"cluster_id": cluster_id, "object_id": object_id})
    return out


def build_semantic_anchor_catalog(
    room_model_json: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    affordance = room_model_json.get("affordance_map")
    if not isinstance(affordance, Mapping):
        affordance = {}
    room_bbox = _room_bbox(room_model_json)
    rows: list[dict[str, object]] = []

    def add_row(
        raw_row: Mapping[str, object],
        *,
        anchor_region_id: str,
        source_region_id: str,
        kind: str,
        allowed_roles: Sequence[str],
        lockable: bool = True,
        wall_side_override: str | None = None,
    ) -> None:
        if not anchor_region_id:
            return
        wall_side = wall_side_override or _wall_side_from_region(raw_row, room_bbox)
        corner_hint = _corner_hint_from_region(raw_row, room_bbox)
        rows.append(
            {
                "anchor_region_id": anchor_region_id,
                "source_region_id": source_region_id or anchor_region_id,
                "kind": kind,
                "wall_side": wall_side,
                "corner_hint": corner_hint,
                "label": _semantic_anchor_label(
                    kind=kind,
                    wall_side=wall_side,
                    corner_hint=corner_hint,
                ),
                "allowed_roles": list(_unique(allowed_roles)),
                "lockable": lockable,
                "score": _safe_float(raw_row.get("score"), default=0.0),
            }
        )

    for row in _mapping_sequence(affordance.get("focal_surfaces")):
        region_id = _clean_str(row.get("id"))
        if region_id is not None:
            add_row(
                row,
                anchor_region_id=region_id,
                source_region_id=region_id,
                kind="focal_wall",
                allowed_roles=("visual_focus", "anchor", "storage"),
            )

    for row in _mapping_sequence(affordance.get("wall_anchor_candidates")):
        region_id = _clean_str(row.get("id"))
        if region_id is not None:
            add_row(
                row,
                anchor_region_id=region_id,
                source_region_id=region_id,
                kind="wall_anchor",
                allowed_roles=("anchor", "storage", "visual_focus", "support", "decor"),
            )

    for row in _mapping_sequence(affordance.get("usable_walls")):
        region_id = _clean_str(row.get("id"))
        if region_id is not None:
            anchor_region_id = f"{region_id}_anchor"
            add_row(
                row,
                anchor_region_id=anchor_region_id,
                source_region_id=anchor_region_id,
                kind="usable_wall_anchor",
                allowed_roles=("anchor", "storage", "visual_focus", "support", "decor"),
            )

    for row in _mapping_sequence(affordance.get("daylight_regions")):
        region_id = _semantic_region_id(row)
        if region_id is not None:
            add_row(
                row,
                anchor_region_id=region_id,
                source_region_id=region_id,
                kind="daylight_region",
                allowed_roles=("anchor", "functional_support", "support", "decor"),
                wall_side_override="window_side",
            )

    for row in _mapping_sequence(affordance.get("privacy_regions")):
        region_id = _semantic_region_id(row)
        if region_id is not None:
            add_row(
                row,
                anchor_region_id=region_id,
                source_region_id=region_id,
                kind="privacy_region",
                allowed_roles=("anchor", "storage", "support"),
            )

    for row in _mapping_sequence(affordance.get("center_openness_regions")):
        region_id = _semantic_region_id(row)
        if region_id is not None:
            add_row(
                row,
                anchor_region_id=region_id,
                source_region_id=region_id,
                kind="center_region",
                allowed_roles=("anchor", "support", "decor"),
                lockable=False,
                wall_side_override="center",
            )

    return tuple(_dedupe_semantic_anchor_rows(rows))


def _decision_contexts(cluster: Mapping[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for decision in _sequence_or_empty(cluster.get("decisions")):
        if not isinstance(decision, Mapping):
            continue
        quantity = int(decision.get("quantity") or 0)
        if quantity <= 0:
            continue
        out.append(
            {
                "object_type": str(
                    decision.get("object_type") or decision.get("category") or ""
                ),
                "quantity": quantity,
                "size_tier": str(decision.get("size_tier") or ""),
                "priority": str(decision.get("priority") or ""),
                "role": str(decision.get("role") or ""),
            }
        )
    return out


def _zone_claims(cluster: Mapping[str, object]) -> dict[str, object]:
    cluster_rules = cluster.get("cluster_rules")
    if not isinstance(cluster_rules, Mapping):
        return {}
    zone_claims = cluster_rules.get("zone_claims")
    return dict(zone_claims) if isinstance(zone_claims, Mapping) else {}


def _semantic_anchor_prompt_payload(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "anchor_region_id": str(row.get("anchor_region_id") or ""),
        "source_region_id": str(row.get("source_region_id") or ""),
        "kind": str(row.get("kind") or ""),
        "wall_side": str(row.get("wall_side") or "center"),
        "corner_hint": str(row.get("corner_hint") or "none"),
        "label": str(row.get("label") or row.get("anchor_region_id") or ""),
        "allowed_roles": list(_string_sequence(row.get("allowed_roles"))),
        "lockable": bool(row.get("lockable")),
        "score": _safe_float(row.get("score"), default=0.0),
    }


def _object_region_requires_semantic_anchor(row: Mapping[str, object]) -> bool:
    placement_role = str(row.get("placement_role") or "")
    wall_side = str(row.get("wall_side") or "center")
    corner_hint = str(row.get("corner_hint") or "none")
    if placement_role in _SEMANTIC_REQUIRED_ROLES:
        return True
    return placement_role == "anchor" and (
        wall_side not in {"", "center"} or corner_hint != "none"
    )


def _validate_semantic_anchor_role(
    *,
    item: Mapping[str, object],
    semantic_anchor: Mapping[str, object],
) -> None:
    placement_role = str(item.get("placement_role") or "")
    allowed_roles = set(_string_sequence(semantic_anchor.get("allowed_roles")))
    if allowed_roles and placement_role not in allowed_roles:
        raise ValueError(
            "visual relation planner used semantic anchor with disallowed role: "
            f"{item.get('cluster_id')}/{item.get('object_id')} "
            f"{item.get('semantic_anchor_region_id')} for {placement_role}"
        )
    if _object_region_requires_semantic_anchor(item) and not bool(
        semantic_anchor.get("lockable", True)
    ):
        raise ValueError(
            "visual relation planner used non-lockable semantic anchor for "
            f"{item.get('cluster_id')}/{item.get('object_id')}"
        )


def _semantic_anchor_preferences_by_cluster(
    *,
    object_region_plan: Sequence[Mapping[str, object]],
    semantic_anchor_catalog: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    catalog_by_id = {
        str(row.get("anchor_region_id") or ""): row
        for row in semantic_anchor_catalog
        if str(row.get("anchor_region_id") or "").strip()
    }
    out: dict[str, dict[str, dict[str, object]]] = {}
    for row in object_region_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        object_id = _clean_str(row.get("object_id"))
        anchor_region_id = _clean_str(row.get("semantic_anchor_region_id"))
        if cluster_id is None or object_id is None or anchor_region_id is None:
            continue
        catalog_row = catalog_by_id.get(anchor_region_id, {})
        out.setdefault(cluster_id, {})[object_id] = {
            "cluster_id": cluster_id,
            "object_id": object_id,
            "semantic_anchor_region_id": anchor_region_id,
            "source_region_id": str(
                catalog_row.get("source_region_id") or anchor_region_id
            ),
            "kind": str(catalog_row.get("kind") or ""),
            "wall_side": str(catalog_row.get("wall_side") or row.get("wall_side") or ""),
            "corner_hint": str(
                catalog_row.get("corner_hint") or row.get("corner_hint") or "none"
            ),
            "placement_role": str(row.get("placement_role") or ""),
            "anchor_lock_strength": str(row.get("anchor_lock_strength") or "hard"),
            "lockable": bool(catalog_row.get("lockable", True)),
            "priority": str(row.get("priority") or "medium"),
        }
    return out


def _apply_semantic_anchor_hints(
    *,
    anchor_hints: dict[str, dict[str, object]],
    semantic_anchor_preferences: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> None:
    for cluster_id, preferences in semantic_anchor_preferences.items():
        if not isinstance(preferences, Mapping):
            continue
        selected = _cluster_semantic_anchor_hint(preferences)
        if selected is None:
            continue
        anchor_region_id = str(selected.get("semantic_anchor_region_id") or "")
        if not anchor_region_id:
            continue
        hint = anchor_hints.setdefault(str(cluster_id), {})
        existing_preferred = list(_string_sequence(hint.get("preferred_region_ids")))
        hint["zone_assignment"] = anchor_region_id
        hint["preferred_region_ids"] = _unique([anchor_region_id, *existing_preferred])
        if str(selected.get("anchor_lock_strength") or "") == "hard":
            hint["required_region_ids"] = [anchor_region_id]
            hint["anchor_strength"] = "hard"
        wall_side = str(selected.get("wall_side") or "")
        if wall_side:
            hint["preferred_wall_side"] = wall_side
        hint["placement_behavior"] = {
            "wall_backing": "required",
            "front_space": "preferred",
            "daylight_blocking": "neutral",
            "pose_flexibility": "low",
        }


def _cluster_semantic_anchor_hint(
    preferences: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object] | None:
    candidates = [
        row
        for row in preferences.values()
        if isinstance(row, Mapping)
        and str(row.get("anchor_lock_strength") or "") == "hard"
        and bool(row.get("lockable", True))
    ]
    if not candidates:
        return None
    role_rank = {"visual_focus": 0, "anchor": 1, "storage": 2, "support": 3}
    return sorted(
        candidates,
        key=lambda row: role_rank.get(str(row.get("placement_role") or ""), 9),
    )[0]


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _semantic_region_id(row: Mapping[str, object]) -> str | None:
    return _clean_str(row.get("id") or row.get("region_id"))


def _dedupe_semantic_anchor_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in sorted(
        rows,
        key=lambda item: (
            -_safe_float(item.get("score"), default=0.0),
            str(item.get("kind") or ""),
            str(item.get("anchor_region_id") or ""),
        ),
    ):
        anchor_region_id = str(row.get("anchor_region_id") or "")
        if not anchor_region_id or anchor_region_id in seen:
            continue
        seen.add(anchor_region_id)
        out.append(dict(row))
    return out


def _semantic_anchor_label(
    *,
    kind: str,
    wall_side: str,
    corner_hint: str,
) -> str:
    parts = [kind.replace("_", " ")]
    if wall_side and wall_side != "center":
        parts.append(wall_side.replace("_", " "))
    if corner_hint and corner_hint != "none":
        parts.append(corner_hint.replace("_", " "))
    return " / ".join(parts)


def _wall_side_from_region(
    row: Mapping[str, object],
    room_bbox: tuple[int, int, int, int],
) -> str:
    bbox = _bbox_from_region(row)
    if bbox is None:
        return "center"
    min_x, min_y, max_x, max_y = room_bbox
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    tolerances = {
        "left_wall": abs(bbox[0] - min_x) / width,
        "right_wall": abs(max_x - bbox[2]) / width,
        "top_wall": abs(bbox[1] - min_y) / height,
        "bottom_wall": abs(max_y - bbox[3]) / height,
    }
    wall_side, distance_ratio = min(tolerances.items(), key=lambda item: item[1])
    return wall_side if distance_ratio <= 0.08 else "center"


def _corner_hint_from_region(
    row: Mapping[str, object],
    room_bbox: tuple[int, int, int, int],
) -> str:
    bbox = _bbox_from_region(row)
    if bbox is None:
        return "none"
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    mid_x = (room_bbox[0] + room_bbox[2]) / 2.0
    mid_y = (room_bbox[1] + room_bbox[3]) / 2.0
    vertical = "top" if center_y <= mid_y else "bottom"
    horizontal = "left" if center_x <= mid_x else "right"
    return f"{vertical}_{horizontal}"


def _bbox_from_region(row: Mapping[str, object]) -> tuple[int, int, int, int] | None:
    bbox = row.get("bbox") or row.get("bbox_mm")
    if isinstance(bbox, Mapping):
        parsed = _bbox_tuple(
            bbox.get("min_x"),
            bbox.get("min_y"),
            bbox.get("max_x"),
            bbox.get("max_y"),
        )
        if parsed is not None:
            return parsed
    points = row.get("anchor_polygon_ccw") or row.get("segment_mm") or row.get(
        "polygon_ccw"
    )
    if not isinstance(points, Sequence) or isinstance(points, str):
        return None
    parsed_points = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        try:
            parsed_points.append((float(point.get("x")), float(point.get("y"))))
        except (TypeError, ValueError):
            continue
    if not parsed_points:
        return None
    xs = [point[0] for point in parsed_points]
    ys = [point[1] for point in parsed_points]
    return (
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    )


def _room_bbox(room_model_json: Mapping[str, object]) -> tuple[int, int, int, int]:
    room = room_model_json.get("room")
    if isinstance(room, Mapping):
        bbox = room.get("bbox_mm")
        if isinstance(bbox, Mapping):
            parsed = _bbox_tuple(
                bbox.get("min_x"),
                bbox.get("min_y"),
                bbox.get("max_x"),
                bbox.get("max_y"),
            )
            if parsed is not None:
                return parsed
        polygon = room.get("polygon_ccw")
        if isinstance(polygon, Sequence) and not isinstance(polygon, str):
            parsed_points = []
            for point in polygon:
                if not isinstance(point, Mapping):
                    continue
                try:
                    parsed_points.append((float(point.get("x")), float(point.get("y"))))
                except (TypeError, ValueError):
                    continue
            if parsed_points:
                xs = [point[0] for point in parsed_points]
                ys = [point[1] for point in parsed_points]
                return (
                    int(round(min(xs))),
                    int(round(min(ys))),
                    int(round(max(xs))),
                    int(round(max(ys))),
                )
    return (0, 0, 6000, 4000)


def _bbox_tuple(
    min_x: object,
    min_y: object,
    max_x: object,
    max_y: object,
) -> tuple[int, int, int, int] | None:
    try:
        out = (
            int(round(float(min_x))),
            int(round(float(min_y))),
            int(round(float(max_x))),
            int(round(float(max_y))),
        )
    except (TypeError, ValueError):
        return None
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def _safe_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _room_prompt_context(room_model_json: Mapping[str, object]) -> dict[str, object]:
    room = room_model_json.get("room")
    openings = room_model_json.get("openings")
    meta = room_model_json.get("meta")
    return {
        "room": dict(room) if isinstance(room, Mapping) else {},
        "openings": dict(openings) if isinstance(openings, Mapping) else {},
        "meta": dict(meta) if isinstance(meta, Mapping) else {},
    }


def _build_user_prompt(prompt_context: Mapping[str, object]) -> str:
    return (
        "Analyze the attached 2D floorplan screenshot together with this exact "
        "solver-ready JSON context. Return a single JSON object matching the exact "
        "contract below. Treat the screenshot as visual intent only; the JSON is "
        "authoritative for IDs, counts, and allowed regions.\n\n"
        "Required output contract:\n"
        f"{json.dumps(_OUTPUT_CONTRACT, ensure_ascii=True, separators=(',', ':'))}"
        "\n\nContext:\n"
        f"{json.dumps(prompt_context, ensure_ascii=True, separators=(',', ':'))}"
    )


def _cluster_zone_payload(row: object) -> dict[str, object]:
    item = row.model_dump() if hasattr(row, "model_dump") else dict(row)
    zone = str(item.get("zone_assignment") or "")
    preferred_wall_side = str(item.get("preferred_wall_side") or "center")
    center_usage = str(item.get("center_usage") or "none")
    return {
        **item,
        "wall_claim": _wall_claim(zone, preferred_wall_side),
        "scenario_role": "primary"
        if item.get("priority") == "high" or center_usage == "primary"
        else "support",
        "entry_relation": "avoid_direct_entry_conflict",
        "daylight_relation": "claim_daylight"
        if "daylight" in zone or preferred_wall_side == "window_side"
        else "avoid_window_blocking"
        if "window" in zone
        else "neutral",
        "placement_bias": _placement_bias(zone, preferred_wall_side, center_usage),
        "required_region_ids": [zone] if item.get("anchor_strength") == "hard" else [],
        "preferred_region_ids": [zone],
        "forbidden_region_ids": [],
        "placement_behavior": _placement_behavior(
            zone, preferred_wall_side, center_usage
        ),
    }


def _object_region_payload(row: object) -> dict[str, object]:
    item = (
        row.model_dump(exclude_none=True) if hasattr(row, "model_dump") else dict(row)
    )
    return {
        **item,
        "preferred_region_ids": [str(item.get("region_id") or "")],
        "wall_claim": _wall_claim(
            str(item.get("region_id") or ""),
            str(item.get("wall_side") or "center"),
        ),
        "placement_behavior": _placement_behavior(
            str(item.get("region_id") or ""),
            str(item.get("wall_side") or "center"),
            "none",
        ),
    }


def _object_region_preferences_by_cluster(
    object_region_plan: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    out: dict[str, dict[str, dict[str, object]]] = {}
    for row in object_region_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        object_id = _clean_str(row.get("object_id"))
        if cluster_id is None or object_id is None:
            continue
        out.setdefault(cluster_id, {})[object_id] = dict(row)
    return out


def _functional_pair_hints(
    output: VisualRelationPlannerOutput,
) -> list[dict[str, object]]:
    by_cluster: dict[str, set[str]] = {}
    explicit_targets: list[dict[str, object]] = []
    for row in output.object_region_plan:
        by_cluster.setdefault(row.cluster_id, set()).add(row.object_id)
        if row.target_object_id is not None:
            explicit_targets.append(
                {
                    "cluster_id": row.cluster_id,
                    "object_id": row.object_id,
                    "target_object_id": row.target_object_id,
                    "target_object_cluster_id": row.target_object_cluster_id
                    or row.cluster_id,
                    "placement_role": row.placement_role,
                    "facing_intent": row.facing_intent,
                    "allow_target_in_front_access": row.placement_role
                    in {"paired_seat", "functional_support"},
                }
            )

    hints = list(explicit_targets)
    for cluster_id, object_ids in by_cluster.items():
        if {"desk", "chair"} <= object_ids:
            hints.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": "chair",
                    "target_object_id": "desk",
                    "target_object_cluster_id": cluster_id,
                    "placement_role": "paired_seat",
                    "facing_intent": "face_target",
                    "allow_target_in_front_access": True,
                }
            )
        if {"wardrobe", "bookshelf"} <= object_ids:
            hints.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": "bookshelf",
                    "target_object_id": "wardrobe",
                    "target_object_cluster_id": cluster_id,
                    "placement_role": "storage",
                    "facing_intent": "front_to_open_space",
                    "allow_target_in_front_access": False,
                }
            )
        if {"bed", "nightstand"} <= object_ids:
            hints.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": "nightstand",
                    "target_object_id": "bed",
                    "target_object_cluster_id": cluster_id,
                    "placement_role": "support",
                    "facing_intent": "front_to_open_space",
                    "allow_target_in_front_access": False,
                }
            )
    return _dedupe_pair_hints(hints)


def _dedupe_pair_hints(
    hints: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in hints:
        key = (
            str(row.get("cluster_id") or ""),
            str(row.get("object_id") or ""),
            str(row.get("target_object_cluster_id") or ""),
            str(row.get("target_object_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _anchor_preference(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "zone_assignment": str(row.get("zone_assignment") or ""),
        "required_region_ids": list(_string_sequence(row.get("required_region_ids"))),
        "forbidden_region_ids": list(_string_sequence(row.get("forbidden_region_ids"))),
        "preferred_region_ids": list(_string_sequence(row.get("preferred_region_ids"))),
        "wall_claim": str(row.get("wall_claim") or "none"),
        "center_usage": str(row.get("center_usage") or "none"),
        "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
        "anchor_strength": str(row.get("anchor_strength") or "medium"),
        "scenario_role": str(row.get("scenario_role") or "support"),
        "entry_relation": str(row.get("entry_relation") or ""),
        "daylight_relation": str(row.get("daylight_relation") or ""),
        "placement_bias": str(row.get("placement_bias") or "balanced"),
        "placement_behavior": dict(row.get("placement_behavior") or {}),
    }


def _cluster_affinity(row: Mapping[str, object]) -> dict[str, object]:
    zone = str(row.get("zone_assignment") or "")
    center_usage = str(row.get("center_usage") or "")
    preferred_wall_side = str(row.get("preferred_wall_side") or "")
    prefer: list[str] = []
    avoid = ["entry_blocking", "main_path"]
    if "wall" in zone or "edge" in zone or preferred_wall_side.endswith("_wall"):
        prefer.extend(["wall", "recess_or_edge"])
    if "daylight" in zone or preferred_wall_side == "window_side":
        prefer.append("window_side")
    if "private" in zone:
        prefer.append("far_from_entry")
    if center_usage in {"partial", "primary"}:
        prefer.append("center")
    if center_usage in {"none", "open_reserved"}:
        avoid.append("center")
    return {
        "cluster_id": str(row.get("cluster_id") or ""),
        "prefer": _unique(prefer),
        "avoid": _unique(avoid),
        "priority": "high" if row.get("anchor_strength") == "hard" else "medium",
        "reason": str(row.get("reason") or "Visual floorplan relation intent."),
    }


def _cluster_orientations(
    output: VisualRelationPlannerOutput,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    secondary = output.secondary_cluster_id
    for row in cluster_zone_plan:
        cluster_id = str(row.get("cluster_id") or "")
        if not cluster_id:
            continue
        intents = ["access_to_open_space"]
        if str(row.get("preferred_wall_side") or "") != "center":
            intents.append("back_to_wall")
        payload: dict[str, object] = {
            "cluster_id": cluster_id,
            "intents": _unique(intents),
            "priority": "high" if cluster_id == output.primary_cluster_id else "medium",
            "reason": str(row.get("reason") or "Visual floorplan orientation intent."),
        }
        if secondary and cluster_id == output.primary_cluster_id:
            payload["intents"] = ["face_cluster", *_unique(intents)]
            payload["target_cluster_id"] = secondary
        rows.append(payload)
    return rows


def _anchor_pair_contracts(
    output: VisualRelationPlannerOutput,
) -> list[dict[str, object]]:
    if output.primary_cluster_id and output.secondary_cluster_id:
        return [
            {
                "pair_type": "face_each_other",
                "cluster_a": output.primary_cluster_id,
                "cluster_b": output.secondary_cluster_id,
                "strength": "high",
                "required": bool(output.object_viewing_pairs),
            }
        ]
    return []


def _directional_relations(
    pair_contracts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for contract in pair_contracts:
        a = _clean_str(contract.get("cluster_a"))
        b = _clean_str(contract.get("cluster_b"))
        if a and b and a != b:
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "relation": "face_each_other",
                    "priority": "high",
                    "reason": "Visual planner identified a focal/viewing relation.",
                }
            )
    return rows


def _allowed_variant_families(row: Mapping[str, object]) -> list[str]:
    zone = str(row.get("zone_assignment") or "")
    if "center" in zone or row.get("center_usage") in {"partial", "primary"}:
        return ["floating_center", "conversation_facing", "generic"]
    if "wall" in zone or "edge" in zone:
        return ["wall_backed", "media_facing", "generic"]
    return ["generic", "edge_aligned"]


def _wall_claim(zone: str, preferred_wall_side: str) -> str:
    if "wall" in zone or "edge" in zone or preferred_wall_side.endswith("_wall"):
        return "strong"
    if preferred_wall_side == "window_side" or "daylight" in zone:
        return "medium"
    return "none"


def _placement_bias(zone: str, preferred_wall_side: str, center_usage: str) -> str:
    if center_usage in {"partial", "primary"} or "center" in zone:
        return "centered"
    if preferred_wall_side == "window_side" or "daylight" in zone:
        return "daylight_edge"
    if "wall" in zone or "edge" in zone:
        return "wall_backed"
    return "balanced"


def _placement_behavior(
    zone: str, preferred_wall_side: str, center_usage: str
) -> dict[str, object]:
    wall_backing = "preferred"
    if "wall" in zone or "edge" in zone or preferred_wall_side.endswith("_wall"):
        wall_backing = "required"
    return {
        "wall_backing": wall_backing,
        "front_space": "required" if center_usage == "primary" else "preferred",
        "daylight_blocking": "prefer"
        if preferred_wall_side == "window_side" or "daylight" in zone
        else "neutral",
        "pose_flexibility": "low" if wall_backing == "required" else "medium",
    }


def _diversity_signature(output: VisualRelationPlannerOutput) -> str:
    rows = [
        f"{row.cluster_id}:{row.zone_assignment}:{row.preferred_wall_side}:{row.center_usage}"
        for row in output.cluster_zone_plan
    ]
    return f"visual|{output.concept_family}|" + "|".join(rows)


def _max_tokens_from_env() -> int:
    try:
        return max(256, int(os.getenv(_MAX_TOKENS_ENV, str(_DEFAULT_MAX_TOKENS))))
    except ValueError:
        return _DEFAULT_MAX_TOKENS


def _max_tokens_param_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized.startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _extract_chat_completion_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("OpenAI response missing choices")
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    raise ValueError("OpenAI response missing message content")


def _sequence_or_empty(value: object) -> Sequence[object]:
    if _is_non_string_sequence(value):
        return value
    return ()


def _is_non_string_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str)


def _string_sequence(value: object) -> tuple[str, ...]:
    out: list[str] = []
    for item in _sequence_or_empty(value):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return tuple(out)


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip()
    return clean or None


def _unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


_SYSTEM_PROMPT = (
    "You are an experimental visual relation planner for an interior layout solver. "
    "You translate a 2D floorplan screenshot into coarse zone and relation intent. "
    "You must never invent IDs or coordinates. Return exactly one JSON object "
    "matching the requested contract, with no extra fields."
)
