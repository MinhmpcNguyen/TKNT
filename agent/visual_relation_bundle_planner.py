from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agent.seed_concept_generator import (
    CONCEPT_FAMILIES,
    DEDUPE_THRESHOLD,
    MACRO_REGION_CAP,
    MAX_ASSIGNMENT_CANDIDATES_PER_CONCEPT,
    MAX_REGION_CANDIDATES_PER_CLUSTER,
    MIN_CONCEPT_DISTANCE,
    _available_macro_region_ids,
    _build_cluster_programs,
    _build_macro_region_map,
    _extract_style_policy,
    _guided_clusters_summary,
    _guided_room_summary,
    _instantiate_concept_families,
    _macro_regions_from_map,
    _match_clusters_to_macro_regions,
    _object_viewing_pairs_from_request_contract,
    _region_candidate_to_dict,
    _room_id,
    _room_type,
    _sanitize_macro_concept_guidance,
    _style_ranked_concept_families,
    _top_region_candidates_by_cluster,
    _unwrap_room_model,
)
from agent.visual_relation_planner import (
    OpenAIVisionJsonClient,
    VisualJsonClientProtocol,
    validate_floorplan_image_data_url,
)

_MODEL_ENV = "TKNT_VISUAL_RELATION_MODEL"
_MAX_TOKENS_ENV = "TKNT_VISUAL_RELATION_V2_MAX_TOKENS"
_DEFAULT_MODEL = "gpt-5.4-mini"
_DEFAULT_MAX_TOKENS = 3200
_RAW_COORDINATE_KEYS = frozenset(
    {
        "x",
        "y",
        "z",
        "x_mm",
        "y_mm",
        "z_mm",
        "left",
        "top",
        "right",
        "bottom",
        "bbox",
        "bbox_mm",
        "bounding_box",
        "rect",
        "rectangle",
        "pixel",
        "pixels",
        "rotation",
        "rotation_deg",
        "angle",
        "angle_deg",
        "coordinates",
        "coordinate",
        "position",
        "position_mm",
        "center",
        "center_mm",
    }
)
_OUTPUT_CONTRACT: dict[str, object] = {
    "family_order": [
        "focal_axis",
        "open_center",
        "edge_weighted",
        "zoned",
        "daylight_oriented",
    ],
    "concept_blueprints": [
        {
            "concept_family": "one available concept family",
            "topology_policy": {
                "reserve_center_degree": "low|medium|high|very_high",
                "wall_loading_bias": "focal_wall_first|edge_first|balanced|zone_specific",
                "daylight_bias": "primary_support_preferred|work_near_window|neutral",
                "entry_avoidance_strength": "medium|high|very_high",
                "secondary_zone_placement_bias": "daylight_if_not_focal|opposite_primary|edge_loaded",
            },
            "cluster_zone_overrides": [
                {
                    "cluster_id": "listed cluster_id",
                    "zone_assignment": "listed macro region_id",
                    "preferred_wall_side": "center|window_side|top_wall|right_wall|bottom_wall|left_wall",
                    "center_usage": "none|partial|primary|open_reserved",
                }
            ],
            "notes": ["short visual planning rationale"],
        }
    ],
    "notes": ["short summary"],
}


@dataclass(frozen=True)
class VisualRelationBundlePlannerResult:
    raw_output: dict[str, object]
    normalized_guidance: dict[str, object]
    bundle: dict[str, object]
    model_name: str


@dataclass(frozen=True)
class VisualRelationBundlePlanner:
    client: VisualJsonClientProtocol | None = None
    model_name: str | None = None
    max_tokens: int | None = None

    def generate_bundle(
        self,
        *,
        room_model_json: Mapping[str, object],
        merged_clusters_json: Mapping[str, object],
        floorplan_image_data_url: str,
        target_count: int = 5,
        description: str | None = None,
        special_notes: str | None = None,
    ) -> VisualRelationBundlePlannerResult:
        image_data_url = validate_floorplan_image_data_url(floorplan_image_data_url)
        room_model = _unwrap_room_model(room_model_json)
        room_type = _room_type(room_model)
        cluster_programs = _build_cluster_programs(merged_clusters_json)
        if not cluster_programs:
            raise ValueError("visual relation bundle planner requires clusters")
        style_policy = _extract_style_policy(merged_clusters_json)
        macro_region_map = _build_macro_region_map(room_model)
        region_candidates = _match_clusters_to_macro_regions(
            cluster_programs=cluster_programs,
            macro_regions=_macro_regions_from_map(macro_region_map),
        )
        object_viewing_pairs = _object_viewing_pairs_from_request_contract(
            clusters_json=merged_clusters_json,
            cluster_programs=cluster_programs,
        )
        if not region_candidates:
            raise ValueError("visual relation bundle planner found no region candidates")

        target_count = max(1, min(int(target_count), len(CONCEPT_FAMILIES)))
        cluster_ids = {cluster.cluster_id for cluster in cluster_programs}
        region_ids = _available_macro_region_ids(macro_region_map)
        prompt_context = {
            "room_type": room_type,
            "description": (description or "").strip(),
            "special_notes": (special_notes or "").strip(),
            "target_count": target_count,
            "style_policy": style_policy,
            "cluster_programs": [
                {
                    "cluster_id": cluster.cluster_id,
                    "semantic_role": cluster.semantic_role,
                    "layout_role": cluster.layout_role,
                    "role_kind": cluster.role_kind,
                    "priority": cluster.priority,
                    "zone_claims": dict(cluster.zone_claims),
                    "object_ids": list(cluster.object_ids),
                    "protected_object_ids": list(cluster.required_object_ids),
                    "droppable_object_ids": list(cluster.droppable_object_ids),
                }
                for cluster in cluster_programs
            ],
            "macro_regions": [
                {
                    "region_id": region.get("region_id"),
                    "region_type": region.get("region_type"),
                    "tags": region.get("tags"),
                    "source_ids": region.get("source_ids"),
                }
                for region in _sequence_or_empty(macro_region_map.get("regions"))
                if isinstance(region, Mapping)
            ],
            "protected_topology": macro_region_map.get("protected_topology") or {},
            "top_region_candidates_by_cluster": _top_region_candidates_by_cluster(
                region_candidates
            ),
            "available_concept_families": list(CONCEPT_FAMILIES),
            "deterministic_family_order": list(
                _style_ranked_concept_families(style_policy)
            ),
            "clusters_json_summary": _guided_clusters_summary(merged_clusters_json),
            "room_topology_summary": _guided_room_summary(room_model, macro_region_map),
            "output_rules": [
                "Return JSON only.",
                "Use only listed concept families.",
                "Use only listed cluster_id values.",
                "Use only listed macro region_id values for zone_assignment.",
                "Do not return coordinates, bbox, pixels, mm positions, exact object placements, or rotations.",
                "Create diverse concepts that preserve entry, circulation, windows, and center openness.",
            ],
        }
        model_name = self.model_name or os.getenv(_MODEL_ENV, _DEFAULT_MODEL).strip()
        max_tokens = self.max_tokens or _max_tokens_from_env()
        raw_output = dict(
            (self.client or OpenAIVisionJsonClient()).create_json_response(
                model_name=model_name,
                system_prompt=_system_prompt(),
                user_prompt=_build_user_prompt(prompt_context),
                image_data_url=image_data_url,
                max_tokens=max_tokens,
            )
        )
        _reject_raw_coordinates(raw_output)
        _validate_guidance_references(
            raw_output,
            cluster_ids=cluster_ids,
            region_ids=region_ids,
        )
        normalized_guidance = _sanitize_macro_concept_guidance(
            raw_output,
            target_count=target_count,
            available_families=CONCEPT_FAMILIES,
            cluster_ids=cluster_ids,
            region_ids=region_ids,
        )
        concepts = _instantiate_concept_families(
            room_model=room_model,
            room_type=room_type,
            cluster_programs=cluster_programs,
            macro_region_map=macro_region_map,
            region_candidates=region_candidates,
            style_policy=style_policy,
            target_count=target_count,
            object_viewing_pairs=object_viewing_pairs,
            guidance_bundle=normalized_guidance,
        )
        bundle = {
            "status": "OK" if concepts else "UNSAT",
            "room_id": _room_id(room_model),
            "room_type": room_type,
            "controller": "visual_llm_macro_concept_generator",
            "settings": {
                "macro_region_cap": MACRO_REGION_CAP,
                "concept_family_count": len(CONCEPT_FAMILIES),
                "max_region_candidates_per_cluster": MAX_REGION_CANDIDATES_PER_CLUSTER,
                "max_assignment_candidates_per_concept": MAX_ASSIGNMENT_CANDIDATES_PER_CONCEPT,
                "dedupe_threshold": DEDUPE_THRESHOLD,
                "min_concept_distance": MIN_CONCEPT_DISTANCE,
                "llm_guided": True,
                "vision_guided": True,
            },
            "style_policy": style_policy,
            "macro_region_map": macro_region_map,
            "region_assignment_candidates": [
                _region_candidate_to_dict(candidate)
                for candidate in region_candidates[
                    :MAX_ASSIGNMENT_CANDIDATES_PER_CONCEPT
                ]
            ],
            "concepts": concepts,
            "notes": [
                "Visual Relation Planner V2 generated macro concept guidance from the floorplan screenshot.",
                "Concepts do not place exact coordinates; solver owns pose search.",
                *_string_list(normalized_guidance.get("notes")),
            ],
            "llm_guidance": normalized_guidance,
        }
        return VisualRelationBundlePlannerResult(
            raw_output=raw_output,
            normalized_guidance=normalized_guidance,
            bundle=bundle,
            model_name=model_name,
        )


def _system_prompt() -> str:
    return (
        "You are a spatial macro-relation planner. Use the floorplan screenshot as "
        "visual evidence and the JSON context as the source of truth. Return only "
        "macro concept guidance. Never invent IDs and never output coordinates."
    )


def _build_user_prompt(prompt_context: Mapping[str, object]) -> str:
    return (
        "Analyze the 2D floorplan screenshot and JSON context. Decide diverse, "
        "reasonable macro relation concepts for the solver. Think silently, then "
        "return a single JSON object matching this contract:\n"
        f"{json.dumps(_OUTPUT_CONTRACT, ensure_ascii=True, separators=(',', ':'))}"
        "\n\nContext:\n"
        f"{json.dumps(prompt_context, ensure_ascii=True, separators=(',', ':'))}"
    )


def _reject_raw_coordinates(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            clean_key = str(key).strip().lower()
            if clean_key in _RAW_COORDINATE_KEYS or clean_key.endswith("_px"):
                raise ValueError(
                    f"visual relation bundle planner returned raw geometry at {path}.{key}"
                )
            _reject_raw_coordinates(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for index, item in enumerate(value):
            _reject_raw_coordinates(item, path=f"{path}[{index}]")


def _validate_guidance_references(
    payload: Mapping[str, object],
    *,
    cluster_ids: set[str],
    region_ids: set[str],
) -> None:
    for item in _sequence_or_empty(payload.get("concept_blueprints")):
        if not isinstance(item, Mapping):
            continue
        family = str(item.get("concept_family") or "").strip()
        if family and family not in CONCEPT_FAMILIES:
            raise ValueError(
                f"visual relation bundle planner used unknown concept family: {family}"
            )
        for override in _sequence_or_empty(item.get("cluster_zone_overrides")):
            if not isinstance(override, Mapping):
                continue
            cluster_id = str(override.get("cluster_id") or "").strip()
            region_id = str(override.get("zone_assignment") or "").strip()
            if cluster_id and cluster_id not in cluster_ids:
                raise ValueError(
                    "visual relation bundle planner used unknown cluster: "
                    f"{cluster_id}"
                )
            if region_id and region_id not in region_ids:
                raise ValueError(
                    "visual relation bundle planner used unknown macro region: "
                    f"{region_id}"
                )


def _sequence_or_empty(value: object) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return list(value)


def _string_list(value: object) -> list[str]:
    out: list[str] = []
    for item in _sequence_or_empty(value):
        if not isinstance(item, str):
            continue
        clean = item.strip()
        if clean:
            out.append(clean)
    return out


def _max_tokens_from_env() -> int:
    try:
        return max(512, int(os.getenv(_MAX_TOKENS_ENV, str(_DEFAULT_MAX_TOKENS))))
    except ValueError:
        return _DEFAULT_MAX_TOKENS
