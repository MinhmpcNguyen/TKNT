from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from layout.variant_family import SLEEP_VARIANT_FAMILIES

logger = logging.getLogger(__name__)

ConceptFamily = Literal[
    "focal_axis",
    "open_center",
    "edge_weighted",
    "zoned",
    "daylight_oriented",
]
Priority = Literal["core", "support", "optional"]
ZoneComplementRole = Literal["focal_claim", "topology_complement", "neutral"]
AnchorStrength = Literal["hard", "strong", "medium"]

CONCEPT_FAMILIES: tuple[ConceptFamily, ...] = (
    "focal_axis",
    "open_center",
    "edge_weighted",
    "zoned",
    "daylight_oriented",
)
MACRO_REGION_CAP = 16
MAX_REGION_CANDIDATES_PER_CLUSTER = 4
MAX_ASSIGNMENT_CANDIDATES_PER_CONCEPT = 32
DEDUPE_THRESHOLD = 0.75
MIN_CONCEPT_DISTANCE = 0.25
MAX_VARIANT_FAMILIES_PER_CLUSTER = 4
MIN_VARIANT_FAMILIES_PER_CLUSTER = 2


@dataclass(frozen=True)
class ClusterProgram:
    cluster_id: str
    semantic_role: str
    role_kind: str
    priority: Priority
    zone_claims: Mapping[str, object]
    relation_intents: tuple[Mapping[str, object], ...]
    seed_region_tags: tuple[str, ...]
    object_ids: tuple[str, ...]

    @property
    def dominant_anchor_object_id(self) -> str | None:
        return self.object_ids[0] if self.object_ids else None


@dataclass(frozen=True)
class MacroRegion:
    region_id: str
    region_type: str
    label: str
    source_ids: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class MacroScenario:
    scenario_id: str
    label: str
    primary_zone_id: str
    secondary_zone_id: str
    storage_zone_id: str
    support_zone_id: str
    primary_wall_side: str
    secondary_wall_side: str
    storage_wall_side: str
    pair_type: str
    center_policy: str
    primary_anchor_strength: AnchorStrength
    secondary_anchor_strength: AnchorStrength
    support_anchor_strength: AnchorStrength
    allow_primary_center_overlap: bool = False


@dataclass(frozen=True)
class RegionCandidate:
    cluster_id: str
    region_id: str
    region_type: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SeedConceptGenerator:
    """Deterministic topology-aware macro concept generator.

    This replaces the old LLM relation planner. It works at zone-claim and room
    topology level, then emits solver-compatible projections as an adapter layer.
    """

    def generate_bundle(
        self,
        *,
        room_model_json: Mapping[str, object],
        clusters_json: Mapping[str, object],
        target_count: int = 5,
    ) -> dict[str, object]:
        room_model = _unwrap_room_model(room_model_json)
        room_type = _room_type(room_model)
        cluster_programs = _build_cluster_programs(clusters_json)
        style_policy = _extract_style_policy(clusters_json)
        macro_region_map = _build_macro_region_map(room_model)
        region_candidates = _match_clusters_to_macro_regions(
            cluster_programs=cluster_programs,
            macro_regions=_macro_regions_from_map(macro_region_map),
        )
        concepts = _instantiate_concept_families(
            room_model=room_model,
            room_type=room_type,
            cluster_programs=cluster_programs,
            macro_region_map=macro_region_map,
            region_candidates=region_candidates,
            style_policy=style_policy,
            target_count=target_count,
        )

        status = "OK" if concepts and cluster_programs else "UNSAT"
        return {
            "status": status,
            "room_id": _room_id(room_model),
            "room_type": room_type,
            "controller": "deterministic_seed_concept_generator",
            "settings": {
                "macro_region_cap": MACRO_REGION_CAP,
                "concept_family_count": len(CONCEPT_FAMILIES),
                "max_region_candidates_per_cluster": MAX_REGION_CANDIDATES_PER_CLUSTER,
                "max_assignment_candidates_per_concept": MAX_ASSIGNMENT_CANDIDATES_PER_CONCEPT,
                "dedupe_threshold": DEDUPE_THRESHOLD,
                "min_concept_distance": MIN_CONCEPT_DISTANCE,
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
                "Relation planning is absorbed into semantic zone claims and deterministic macro concept generation.",
                "Concepts do not place exact coordinates; MacroClusterSolver owns pose search.",
            ],
        }

    def generate(
        self,
        *,
        room_model_json: Mapping[str, object],
        clusters_json: Mapping[str, object],
        description: str | None = None,
        special_notes: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        _ = description, model_name, temperature, max_tokens
        family = _family_from_text(special_notes)
        bundle = self.generate_bundle(
            room_model_json=room_model_json,
            clusters_json=clusters_json,
            target_count=len(CONCEPT_FAMILIES),
        )
        concepts = [
            concept
            for concept in bundle.get("concepts", [])
            if isinstance(concept, dict)
        ]
        concept = next(
            (item for item in concepts if item.get("concept_family") == family),
            concepts[0] if concepts else None,
        )
        room_model = _unwrap_room_model(room_model_json)
        if not isinstance(concept, dict):
            return _empty_solver_plan(room_model)
        return _concept_to_solver_plan(
            concept=concept,
            room_model=room_model,
            room_type=str(bundle.get("room_type") or _room_type(room_model_json)),
        )


def solver_plan_from_concept(
    *,
    concept: Mapping[str, object],
    room_model_json: Mapping[str, object],
    room_type: str | None = None,
) -> dict[str, object]:
    return _concept_to_solver_plan(
        concept=concept,
        room_model=_unwrap_room_model(room_model_json),
        room_type=room_type or _room_type(room_model_json),
    )


def _extract_style_policy(clusters_json: Mapping[str, object]) -> dict[str, object]:
    policy = clusters_json.get("style_policy")
    if isinstance(policy, Mapping):
        return dict(policy)
    semantic = clusters_json.get("semantic_layout_program")
    if isinstance(semantic, Mapping):
        policy = semantic.get("style_policy")
        if isinstance(policy, Mapping):
            return dict(policy)
    return {}


def _style_layout_policy(style_policy: Mapping[str, object]) -> Mapping[str, object]:
    layout_policy = style_policy.get("layout_policy")
    return layout_policy if isinstance(layout_policy, Mapping) else {}


def _style_ranked_concept_families(
    style_policy: Mapping[str, object],
) -> tuple[ConceptFamily, ...]:
    layout_policy = _style_layout_policy(style_policy)
    ranked: list[ConceptFamily] = list(CONCEPT_FAMILIES)
    if _bias_level(layout_policy.get("center_openness_bias")) >= 3:
        ranked = _promote_family(ranked, "open_center")
    if _bias_level(layout_policy.get("daylight_bias")) >= 3:
        ranked = _promote_family(ranked, "daylight_oriented")
    if str(layout_policy.get("wall_loading_bias") or "") in {
        "medium_high",
        "perimeter_heavy",
    }:
        ranked = _promote_family(ranked, "edge_weighted")
    if _bias_level(layout_policy.get("symmetry_bias")) >= 3:
        ranked = _promote_family(ranked, "focal_axis")
    return tuple(ranked)


def _promote_family(
    families: Sequence[ConceptFamily],
    family: ConceptFamily,
) -> list[ConceptFamily]:
    return [family, *[item for item in families if item != family]]


def _bias_level(value: object) -> int:
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


def _unwrap_room_model(room_model_json: Mapping[str, object]) -> Mapping[str, object]:
    room = room_model_json.get("room")
    if isinstance(room, Mapping):
        return room_model_json
    for key in ("parsed", "raw"):
        nested = room_model_json.get(key)
        if isinstance(nested, Mapping) and isinstance(nested.get("room"), Mapping):
            return nested
    return room_model_json


def _room_id(room_model: Mapping[str, object]) -> str:
    room = room_model.get("room")
    if isinstance(room, Mapping):
        value = room.get("room_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "room_1"


def _room_type(room_model: Mapping[str, object]) -> str:
    meta = room_model.get("meta")
    if isinstance(meta, Mapping):
        value = meta.get("room_type")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = room_model.get("room_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "room"


def _extract_clusters_map(
    clusters_json: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    clusters = clusters_json.get("clusters")
    if isinstance(clusters, Mapping):
        return {
            str(cluster_id): cluster
            for cluster_id, cluster in clusters.items()
            if isinstance(cluster_id, str) and isinstance(cluster, Mapping)
        }
    if isinstance(clusters, Sequence) and not isinstance(clusters, str):
        out: dict[str, Mapping[str, object]] = {}
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            cluster_id = _clean_str(cluster.get("cluster_id"))
            if cluster_id is not None:
                out[cluster_id] = cluster
        return out

    out = {}
    for cluster_id, cluster in clusters_json.items():
        if not isinstance(cluster_id, str) or not isinstance(cluster, Mapping):
            continue
        if {"cluster_id", "cluster_footprint", "local_placements"} & set(cluster):
            out[cluster_id] = cluster
    return out


def _build_cluster_programs(
    clusters_json: Mapping[str, object],
) -> list[ClusterProgram]:
    clusters = _extract_clusters_map(clusters_json)
    semantic_by_id = _semantic_clusters_by_id(clusters_json)
    programs: list[ClusterProgram] = []
    for cluster_id in sorted(clusters):
        cluster = clusters[cluster_id]
        semantic = semantic_by_id.get(cluster_id, {})
        zone_claims = _mapping_or_empty(
            semantic.get("zone_claims") or _cluster_rules(cluster).get("zone_claims")
        )
        object_ids = tuple(_cluster_object_ids(cluster))
        semantic_role = _clean_str(
            semantic.get("semantic_role")
        ) or _semantic_role_from_cluster(cluster_id, object_ids)
        role_kind = _role_kind(cluster_id, object_ids, semantic_role)
        priority = _priority(semantic.get("priority"), role_kind)
        relation_intents = tuple(
            item
            for item in _sequence_or_empty(semantic.get("relation_intents"))
            if isinstance(item, Mapping)
        )
        programs.append(
            ClusterProgram(
                cluster_id=cluster_id,
                semantic_role=semantic_role,
                role_kind=role_kind,
                priority=priority,
                zone_claims=zone_claims,
                relation_intents=relation_intents,
                seed_region_tags=tuple(_seed_region_tags(cluster)),
                object_ids=object_ids,
            )
        )
    return programs


def _semantic_clusters_by_id(
    clusters_json: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    program = clusters_json.get("semantic_layout_program")
    if not isinstance(program, Mapping):
        return {}
    active = program.get("active_clusters")
    if not isinstance(active, Sequence) or isinstance(active, str):
        return {}
    out: dict[str, Mapping[str, object]] = {}
    for cluster in active:
        if not isinstance(cluster, Mapping):
            continue
        cluster_id = _clean_str(cluster.get("cluster_id"))
        if cluster_id is not None:
            out[cluster_id] = cluster
    return out


def _build_macro_region_map(room_model: Mapping[str, object]) -> dict[str, object]:
    affordance = _mapping_or_empty(room_model.get("affordance_map"))
    topology = _mapping_or_empty(room_model.get("topology"))
    openings = _mapping_or_empty(room_model.get("openings"))
    room = _mapping_or_empty(room_model.get("room"))

    door_ids = _opening_ids(openings.get("doors"), "door")
    window_ids = _opening_ids(openings.get("windows"), "window")
    wall_ids = _wall_region_ids(room, affordance)
    center_ids = _region_refs(affordance.get("center_openness_regions")) or [
        "room_center"
    ]
    corridor_ids = _region_refs(affordance.get("primary_circulation_corridors"))
    if not corridor_ids and door_ids:
        corridor_ids = [f"{door_id}_to_room_center_corridor" for door_id in door_ids]
    entry_ids = _region_refs(affordance.get("entry_landing_zones")) or door_ids
    daylight_ids = _region_refs(affordance.get("daylight_regions")) or window_ids
    private_ids = _region_refs(affordance.get("privacy_regions")) or [
        "deep_zone_far_from_entry"
    ]
    side_wall_ids = (
        "top_wall_zone",
        "right_wall_zone",
        "bottom_wall_zone",
        "left_wall_zone",
    )

    regions = [
        MacroRegion(
            region_id="primary_focal_wall_zone",
            region_type="focal_wall_zone",
            label="primary focal wall zone",
            source_ids=tuple(
                _region_refs(affordance.get("focal_surfaces")) or wall_ids[:2]
            ),
            tags=("wall", "focal", "anchor"),
        ),
        MacroRegion(
            region_id="quiet_private_deep_zone",
            region_type="private_deep_zone",
            label="quiet private deep zone",
            source_ids=tuple(private_ids),
            tags=("privacy", "far_from_entry"),
        ),
        MacroRegion(
            region_id="daylight_biased_zone",
            region_type="daylight_zone",
            label="daylight-biased zone",
            source_ids=tuple(daylight_ids),
            tags=("daylight", "window_side"),
        ),
        MacroRegion(
            region_id="entry_adjacent_active_zone",
            region_type="entry_active_zone",
            label="entry-adjacent active zone",
            source_ids=tuple(entry_ids),
            tags=("entry_side", "active"),
        ),
        MacroRegion(
            region_id="keep_open_center",
            region_type="keep_open_center",
            label="keep-open center",
            source_ids=tuple(center_ids),
            tags=("center", "keep_open"),
        ),
        MacroRegion(
            region_id="floating_support_zone",
            region_type="floating_support_zone",
            label="floating support zone",
            source_ids=tuple(
                _region_refs(affordance.get("floating_zone_candidates"))
                or ["floating_center_zone"]
            ),
            tags=("floating", "support", "center"),
        ),
        MacroRegion(
            region_id="storage_service_edge_zone",
            region_type="storage_service_edge_zone",
            label="storage/service edge zone",
            source_ids=side_wall_ids or tuple(wall_ids),
            tags=("wall", "edge", "storage"),
        ),
        MacroRegion(
            region_id="edge_loading_zone",
            region_type="edge_loading_zone",
            label="edge loading zone",
            source_ids=tuple(
                _region_refs(affordance.get("wall_anchor_candidates"))
                or side_wall_ids
                or wall_ids
            ),
            tags=("wall", "edge", "perimeter"),
        ),
        MacroRegion(
            region_id="top_wall_zone",
            region_type="side_wall_zone",
            label="top wall zone",
            source_ids=("top_wall_zone",),
            tags=("wall", "edge", "top_wall"),
        ),
        MacroRegion(
            region_id="right_wall_zone",
            region_type="side_wall_zone",
            label="right wall zone",
            source_ids=("right_wall_zone",),
            tags=("wall", "edge", "right_wall"),
        ),
        MacroRegion(
            region_id="bottom_wall_zone",
            region_type="side_wall_zone",
            label="bottom wall zone",
            source_ids=("bottom_wall_zone",),
            tags=("wall", "edge", "bottom_wall"),
        ),
        MacroRegion(
            region_id="left_wall_zone",
            region_type="side_wall_zone",
            label="left wall zone",
            source_ids=("left_wall_zone",),
            tags=("wall", "edge", "left_wall"),
        ),
        MacroRegion(
            region_id="window_side_zone",
            region_type="window_side_zone",
            label="window side zone",
            source_ids=tuple(daylight_ids or ["window_side_zone"]),
            tags=("wall", "daylight", "window_side"),
        ),
        MacroRegion(
            region_id="floating_center_zone",
            region_type="floating_center_zone",
            label="floating center zone",
            source_ids=("floating_center_zone",),
            tags=("floating", "center"),
        ),
    ][:MACRO_REGION_CAP]
    return {
        "regions": [_macro_region_to_dict(region) for region in regions],
        "protected_topology": {
            "entry_landing_zones": entry_ids,
            "primary_circulation_corridors": corridor_ids,
            "center_openness_regions": center_ids,
            "operational_zones": _operational_protected_topology_zones(
                entry_refs=entry_ids,
                corridor_refs=corridor_ids,
                center_refs=center_ids,
            ),
        },
        "topology_summary": {
            "entry_nodes": _region_refs(topology.get("entry_nodes")) or door_ids,
            "window_nodes": _region_refs(topology.get("window_nodes")) or window_ids,
            "subzones": _region_refs(topology.get("subzones")),
        },
    }


def _macro_regions_from_map(
    macro_region_map: Mapping[str, object],
) -> list[MacroRegion]:
    out: list[MacroRegion] = []
    regions = macro_region_map.get("regions")
    if not isinstance(regions, Sequence) or isinstance(regions, str):
        return out
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        region_id = _clean_str(region.get("region_id"))
        region_type = _clean_str(region.get("region_type"))
        label = _clean_str(region.get("label")) or region_id
        if region_id is None or region_type is None or label is None:
            continue
        out.append(
            MacroRegion(
                region_id=region_id,
                region_type=region_type,
                label=label,
                source_ids=tuple(_string_list(region.get("source_ids"))),
                tags=tuple(_string_list(region.get("tags"))),
            )
        )
    return out


def _operational_protected_topology_zones(
    *,
    entry_refs: Sequence[str],
    corridor_refs: Sequence[str],
    center_refs: Sequence[str],
) -> list[dict[str, object]]:
    zones: list[dict[str, object]] = []
    for ref in entry_refs:
        zones.append(
            {
                "region": ref,
                "zone_type": "entry_landing",
                "priority": "high",
                "enforcement": "hard",
                "max_overlap_ratio": 0.0,
                "applies_to": [
                    "core_clusters",
                    "support_clusters",
                    "interaction_hulls",
                ],
                "violation_severity": "blocking",
            }
        )
    for ref in corridor_refs:
        zones.append(
            {
                "region": ref,
                "zone_type": "primary_circulation_corridor",
                "priority": "high",
                "enforcement": "hard_soft",
                "max_overlap_ratio": 0.05,
                "applies_to": [
                    "core_clusters",
                    "support_clusters",
                    "interaction_hulls",
                ],
                "violation_severity": "blocking",
            }
        )
    for ref in center_refs:
        zones.append(
            {
                "region": ref,
                "zone_type": "center_openness_core",
                "priority": "high",
                "enforcement": "hard_soft",
                "max_overlap_ratio": 0.10,
                "applies_to": ["core_clusters", "interaction_hulls"],
                "violation_severity": "blocking",
            }
        )
    return zones


def _match_clusters_to_macro_regions(
    *,
    cluster_programs: Sequence[ClusterProgram],
    macro_regions: Sequence[MacroRegion],
) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    for cluster in cluster_programs:
        scored = [
            _score_cluster_region(cluster=cluster, region=region)
            for region in macro_regions
        ]
        scored.sort(key=lambda item: (-item.score, item.region_id))
        candidates.extend(scored[:MAX_REGION_CANDIDATES_PER_CLUSTER])
    return candidates


def _score_cluster_region(
    *,
    cluster: ClusterProgram,
    region: MacroRegion,
) -> RegionCandidate:
    claims = cluster.zone_claims
    tags = set(region.tags)
    role = cluster.role_kind
    score = 0.0
    reasons: list[str] = []

    if cluster.priority == "core":
        score += 1.2
    elif cluster.priority == "support":
        score += 0.5

    wall_affinity = _affinity_level(claims.get("wall_affinity"))
    daylight_affinity = _affinity_level(claims.get("daylight_affinity"))
    privacy_affinity = _affinity_level(claims.get("privacy_affinity"))
    floating_allowed = bool(claims.get("floating_allowed"))

    if "wall" in tags:
        score += {"high": 3.0, "medium": 1.8, "low": 0.8, "none": -0.4}[wall_affinity]
        reasons.append(f"wall_affinity={wall_affinity}")
    if "daylight" in tags:
        score += {"high": 3.2, "medium": 1.8, "low": 0.8, "none": -0.5}[
            daylight_affinity
        ]
        reasons.append(f"daylight_affinity={daylight_affinity}")
    if "privacy" in tags or "far_from_entry" in tags:
        score += {"high": 2.8, "medium": 1.5, "low": 0.5, "none": 0.0}[privacy_affinity]
        reasons.append(f"privacy_affinity={privacy_affinity}")
    if "floating" in tags or "center" in tags:
        score += 1.1 if floating_allowed else -0.8
        reasons.append("floating_allowed" if floating_allowed else "floating_limited")

    if role in {"social_anchor", "lounge"} and region.region_id in {
        "primary_focal_wall_zone",
        "floating_support_zone",
    }:
        score += 1.6
        reasons.append("social_anchor_compatible")
    if role in {"media", "focal"} and "focal" in tags:
        score += 2.2
        reasons.append("focal_compatible")
    if role in {"storage", "service"} and ("storage" in tags or "edge" in tags):
        score += 2.0
        reasons.append("storage_edge_compatible")
    if role == "work" and "daylight" in tags:
        score += 2.0
        reasons.append("work_daylight_compatible")
    if role == "sleep" and "privacy" in tags:
        score += 2.2
        reasons.append("sleep_privacy_compatible")
    if "entry_side" in tags and _cluster_avoids_entry(cluster):
        score -= 2.5
        reasons.append("entry_conflict_penalty")
    if "keep_open" in tags:
        score -= 3.5
        reasons.append("keep_open_not_assignment")
    for seed_tag in cluster.seed_region_tags:
        if seed_tag in tags:
            score += 0.35
            reasons.append(f"seed_tag={seed_tag}")

    return RegionCandidate(
        cluster_id=cluster.cluster_id,
        region_id=region.region_id,
        region_type=region.region_type,
        score=round(score, 3),
        reasons=tuple(reasons[:5]),
    )


def _macro_scenario_for_family(family: ConceptFamily) -> MacroScenario:
    scenarios: dict[ConceptFamily, MacroScenario] = {
        "focal_axis": MacroScenario(
            scenario_id="focal_axis_top_bottom",
            label="primary seating top wall, focal media opposite",
            primary_zone_id="top_wall_zone",
            secondary_zone_id="bottom_wall_zone",
            storage_zone_id="right_wall_zone",
            support_zone_id="left_wall_zone",
            primary_wall_side="top_wall",
            secondary_wall_side="bottom_wall",
            storage_wall_side="right_wall",
            pair_type="opposite_walls",
            center_policy="balanced_open",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="medium",
        ),
        "open_center": MacroScenario(
            scenario_id="open_center_left_right",
            label="primary and media on opposing side walls with open center",
            primary_zone_id="left_wall_zone",
            secondary_zone_id="right_wall_zone",
            storage_zone_id="bottom_wall_zone",
            support_zone_id="top_wall_zone",
            primary_wall_side="left_wall",
            secondary_wall_side="right_wall",
            storage_wall_side="bottom_wall",
            pair_type="opposite_walls",
            center_policy="open_reserved",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="medium",
        ),
        "edge_weighted": MacroScenario(
            scenario_id="edge_weighted_bottom_top",
            label="perimeter loaded layout with primary bottom and media top",
            primary_zone_id="bottom_wall_zone",
            secondary_zone_id="top_wall_zone",
            storage_zone_id="left_wall_zone",
            support_zone_id="right_wall_zone",
            primary_wall_side="bottom_wall",
            secondary_wall_side="top_wall",
            storage_wall_side="left_wall",
            pair_type="opposite_walls",
            center_policy="open_reserved",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="strong",
        ),
        "zoned": MacroScenario(
            scenario_id="floating_primary_right_media",
            label="floating primary seating with wall-backed media zone",
            primary_zone_id="floating_center_zone",
            secondary_zone_id="right_wall_zone",
            storage_zone_id="left_wall_zone",
            support_zone_id="bottom_wall_zone",
            primary_wall_side="center",
            secondary_wall_side="right_wall",
            storage_wall_side="left_wall",
            pair_type="floating_to_wall",
            center_policy="floating_primary",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="medium",
            allow_primary_center_overlap=True,
        ),
        "daylight_oriented": MacroScenario(
            scenario_id="daylight_primary_bottom_media",
            label="primary seating claims daylight with media opposite",
            primary_zone_id="window_side_zone",
            secondary_zone_id="bottom_wall_zone",
            storage_zone_id="right_wall_zone",
            support_zone_id="left_wall_zone",
            primary_wall_side="window_side",
            secondary_wall_side="bottom_wall",
            storage_wall_side="right_wall",
            pair_type="daylight_opposite",
            center_policy="balanced_open",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="medium",
        ),
    }
    return scenarios[family]


def _macro_scenario_to_dict(scenario: MacroScenario) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "primary_zone_id": scenario.primary_zone_id,
        "secondary_zone_id": scenario.secondary_zone_id,
        "storage_zone_id": scenario.storage_zone_id,
        "support_zone_id": scenario.support_zone_id,
        "primary_wall_side": scenario.primary_wall_side,
        "secondary_wall_side": scenario.secondary_wall_side,
        "storage_wall_side": scenario.storage_wall_side,
        "pair_type": scenario.pair_type,
        "center_policy": scenario.center_policy,
        "primary_anchor_strength": scenario.primary_anchor_strength,
        "secondary_anchor_strength": scenario.secondary_anchor_strength,
        "support_anchor_strength": scenario.support_anchor_strength,
        "allow_primary_center_overlap": scenario.allow_primary_center_overlap,
    }


def _instantiate_concept_families(
    *,
    room_model: Mapping[str, object],
    room_type: str,
    cluster_programs: Sequence[ClusterProgram],
    macro_region_map: Mapping[str, object],
    region_candidates: Sequence[RegionCandidate],
    style_policy: Mapping[str, object],
    target_count: int,
) -> list[dict[str, object]]:
    family_order = _style_ranked_concept_families(style_policy)
    concept_count = max(1, min(int(target_count), len(family_order)))
    primary_cluster_id = _primary_cluster_id(cluster_programs)
    secondary_cluster_id = _secondary_cluster_id(cluster_programs, primary_cluster_id)
    concepts: list[dict[str, object]] = []
    for index, family in enumerate(family_order[:concept_count], start=1):
        scenario = _macro_scenario_for_family(family)
        topology_policy = _topology_policy_for_family(family, style_policy=style_policy)
        cluster_zone_plan = [
            _cluster_zone_assignment(
                family=family,
                scenario=scenario,
                cluster=cluster,
                primary_cluster_id=primary_cluster_id,
                secondary_cluster_id=secondary_cluster_id,
                region_candidates=region_candidates,
                topology_policy=topology_policy,
            )
            for cluster in cluster_programs
        ]
        cluster_zone_plan = _resolve_complementary_zone_assignments(
            family=family,
            scenario=scenario,
            cluster_programs=cluster_programs,
            cluster_zone_plan=cluster_zone_plan,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        macro_constraints = _macro_constraints_for_concept(
            family=family,
            scenario=scenario,
            room_model=room_model,
            macro_region_map=macro_region_map,
            cluster_zone_plan=cluster_zone_plan,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        primary_pair_contracts = _primary_pair_contracts_for_concept(
            family=family,
            scenario=scenario,
            cluster_zone_plan=cluster_zone_plan,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        concept = {
            "concept_id": f"concept_{index:02d}",
            "concept_family": family,
            "macro_scenario": _macro_scenario_to_dict(scenario),
            "spatial_character": _spatial_character(family, room_type),
            "macro_region_map_snapshot": dict(macro_region_map),
            "cluster_zone_plan": cluster_zone_plan,
            "allowed_variant_families_by_cluster": (
                _allowed_variant_families_by_cluster(
                    family=family,
                    cluster_zone_plan=cluster_zone_plan,
                )
            ),
            "critical_object_orientations": _critical_object_orientations_for_concept(
                family=family,
                cluster_programs=cluster_programs,
                cluster_zone_plan=cluster_zone_plan,
                primary_cluster_id=primary_cluster_id,
                secondary_cluster_id=secondary_cluster_id,
            ),
            "primary_pair_contracts": primary_pair_contracts,
            "anchor_region_preferences_by_cluster": (
                _anchor_region_preferences_by_cluster(
                    cluster_zone_plan=cluster_zone_plan,
                )
            ),
            "anchor_pair_contracts": _anchor_pair_contracts_for_concept(
                primary_pair_contracts=primary_pair_contracts,
                primary_cluster_id=primary_cluster_id,
                secondary_cluster_id=secondary_cluster_id,
            ),
            "object_solver_policy": _object_solver_policy_for_concept(
                family=family,
                cluster_programs=cluster_programs,
            ),
            "topology_policy": topology_policy,
            "macro_constraints": macro_constraints,
            "concept_readiness_requirements": _concept_readiness_requirements(
                family=family,
                primary_pair_contracts=primary_pair_contracts,
            ),
            "variant_bias_weights": _variant_bias_weights_for_concept(
                family,
                style_policy=style_policy,
            ),
            "concept_score_prior": _concept_score_prior(
                family,
                style_policy=style_policy,
            ),
            "diversity_signature": _diversity_signature(
                family=family,
                cluster_zone_plan=cluster_zone_plan,
                topology_policy=topology_policy,
            ),
        }
        concepts.append(concept)
    return concepts


def _cluster_zone_assignment(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster: ClusterProgram,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
    region_candidates: Sequence[RegionCandidate],
    topology_policy: Mapping[str, object],
) -> dict[str, object]:
    preferred_region_id = _preferred_region_for_family(
        family=family,
        scenario=scenario,
        cluster=cluster,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )
    candidates = [
        candidate
        for candidate in region_candidates
        if candidate.cluster_id == cluster.cluster_id
    ]
    candidate = next(
        (item for item in candidates if item.region_id == preferred_region_id),
        candidates[0] if candidates else None,
    )
    zone_assignment = preferred_region_id
    if candidate is not None and family not in {"open_center", "edge_weighted"}:
        zone_assignment = candidate.region_id
    scenario_role = _scenario_role_for_cluster(
        cluster=cluster,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )
    if scenario_role in {"primary", "secondary", "storage"}:
        zone_assignment = preferred_region_id
    wall_claim = _wall_claim_for(cluster, zone_assignment, family)
    center_usage = _center_usage_for(
        cluster=cluster,
        family=family,
        primary_cluster_id=primary_cluster_id,
        scenario=scenario,
        scenario_role=scenario_role,
    )
    anchor_strength = _anchor_strength_for_scenario_role(
        scenario=scenario,
        scenario_role=scenario_role,
    )
    preferred_wall_side = _preferred_wall_side_for_scenario_role(
        scenario=scenario,
        scenario_role=scenario_role,
    )
    return {
        "cluster_id": cluster.cluster_id,
        "semantic_role": cluster.semantic_role,
        "role_kind": cluster.role_kind,
        "priority": cluster.priority,
        "macro_scenario_id": scenario.scenario_id,
        "scenario_role": scenario_role,
        "scenario_locked": scenario_role in {"primary", "secondary", "storage"},
        "zone_assignment": zone_assignment,
        "required_region_ids": [zone_assignment]
        if scenario_role in {"primary", "secondary", "storage"}
        else [],
        "forbidden_region_ids": _forbidden_region_ids_for_scenario_role(
            scenario=scenario,
            scenario_role=scenario_role,
        ),
        "preferred_wall_side": preferred_wall_side,
        "anchor_strength": anchor_strength,
        "region_candidates": [
            _region_candidate_to_dict(item) for item in candidates[:3]
        ],
        "wall_claim": wall_claim,
        "center_usage": center_usage,
        "entry_relation": _entry_relation_for(cluster, topology_policy),
        "daylight_relation": _daylight_relation_for(cluster, zone_assignment, family),
        "privacy_relation": _privacy_relation_for(cluster, zone_assignment),
        "placement_bias": _placement_bias(zone_assignment, wall_claim, center_usage),
    }


def _resolve_complementary_zone_assignments(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster_programs: Sequence[ClusterProgram],
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    rows = [dict(row) for row in cluster_zone_plan]
    if primary_cluster_id is None or secondary_cluster_id is None:
        return rows

    clusters_by_id = {cluster.cluster_id: cluster for cluster in cluster_programs}
    primary = clusters_by_id.get(primary_cluster_id)
    secondary = clusters_by_id.get(secondary_cluster_id)
    if primary is None or secondary is None:
        return rows

    primary_row = _zone_row_by_cluster(rows, primary_cluster_id)
    secondary_row = _zone_row_by_cluster(rows, secondary_cluster_id)
    if primary_row is None or secondary_row is None:
        return rows
    if primary.priority != "core" or secondary.priority != "core":
        return rows
    if bool(primary_row.get("scenario_locked")) or bool(
        secondary_row.get("scenario_locked")
    ):
        _mark_zone_complement_role(
            rows,
            focal_cluster_id=_focal_claim_cluster_id(
                primary=primary,
                secondary=secondary,
                primary_cluster_id=primary_cluster_id,
            ),
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        return rows
    if not _zone_claims_same_logic_class(primary_row, secondary_row):
        _mark_zone_complement_role(
            rows,
            focal_cluster_id=_focal_claim_cluster_id(
                primary=primary,
                secondary=secondary,
                primary_cluster_id=primary_cluster_id,
            ),
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        return rows

    focal_cluster_id = _focal_claim_cluster_id(
        primary=primary,
        secondary=secondary,
        primary_cluster_id=primary_cluster_id,
    )
    focal = clusters_by_id[focal_cluster_id]
    focal_row = _zone_row_by_cluster(rows, focal_cluster_id)
    if focal_row is not None:
        _apply_zone_assignment_to_row(
            row=focal_row,
            cluster=focal,
            family=family,
            zone_assignment="primary_focal_wall_zone",
        )
    complement_cluster_id = (
        secondary_cluster_id
        if focal_cluster_id == primary_cluster_id
        else primary_cluster_id
    )
    complement = clusters_by_id[complement_cluster_id]
    complement_row = _zone_row_by_cluster(rows, complement_cluster_id)
    if complement_row is not None:
        zone_assignment = _complementary_zone_for_cluster(
            family=family,
            scenario=scenario,
            cluster=complement,
        )
        _apply_zone_assignment_to_row(
            row=complement_row,
            cluster=complement,
            family=family,
            zone_assignment=zone_assignment,
        )
    _mark_zone_complement_role(
        rows,
        focal_cluster_id=focal_cluster_id,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )
    return rows


def _zone_row_by_cluster(
    rows: Sequence[dict[str, object]],
    cluster_id: str,
) -> dict[str, object] | None:
    for row in rows:
        if row.get("cluster_id") == cluster_id:
            return row
    return None


def _zone_claims_same_logic_class(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> bool:
    return _zone_logic_class(first.get("zone_assignment")) == _zone_logic_class(
        second.get("zone_assignment")
    )


def _zone_logic_class(value: object) -> str:
    zone = str(value or "").strip()
    if "focal" in zone:
        return "focal_wall"
    if "daylight" in zone:
        return "daylight"
    if "edge" in zone or "storage" in zone:
        return "perimeter"
    if "private" in zone:
        return "private"
    if "floating" in zone or "center" in zone:
        return "center"
    if "entry" in zone:
        return "entry"
    return zone or "unknown"


def _focal_claim_cluster_id(
    *,
    primary: ClusterProgram,
    secondary: ClusterProgram,
    primary_cluster_id: str,
) -> str:
    for cluster in (primary, secondary):
        if cluster.role_kind in {"media", "focal"}:
            return cluster.cluster_id
    return primary_cluster_id


def _scenario_role_for_cluster(
    *,
    cluster: ClusterProgram,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> str:
    if cluster.cluster_id == primary_cluster_id:
        return "primary"
    if cluster.cluster_id == secondary_cluster_id:
        return "secondary"
    if cluster.role_kind == "storage":
        return "storage"
    return "support"


def _anchor_strength_for_scenario_role(
    *,
    scenario: MacroScenario,
    scenario_role: str,
) -> AnchorStrength:
    if scenario_role == "primary":
        return scenario.primary_anchor_strength
    if scenario_role == "secondary":
        return scenario.secondary_anchor_strength
    return scenario.support_anchor_strength


def _preferred_wall_side_for_scenario_role(
    *,
    scenario: MacroScenario,
    scenario_role: str,
) -> str:
    if scenario_role == "primary":
        return scenario.primary_wall_side
    if scenario_role == "secondary":
        return scenario.secondary_wall_side
    if scenario_role == "storage":
        return scenario.storage_wall_side
    return ""


def _forbidden_region_ids_for_scenario_role(
    *,
    scenario: MacroScenario,
    scenario_role: str,
) -> list[str]:
    if scenario.center_policy == "open_reserved" and scenario_role != "primary":
        return ["floating_center_zone", "keep_open_center"]
    if scenario.center_policy == "open_reserved" and scenario_role == "primary":
        return ["floating_center_zone"]
    if scenario.center_policy == "floating_primary" and scenario_role != "primary":
        return ["floating_center_zone", "keep_open_center"]
    return []


def _complementary_zone_for_cluster(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster: ClusterProgram,
) -> str:
    if cluster.role_kind == "work":
        return "daylight_biased_zone"
    if cluster.role_kind == "storage":
        return scenario.storage_zone_id
    if cluster.role_kind == "sleep":
        return "quiet_private_deep_zone"
    if family == "daylight_oriented" and cluster.role_kind in {
        "social_anchor",
        "lounge",
    }:
        return "daylight_biased_zone"
    if family == "zoned":
        return scenario.support_zone_id
    return scenario.support_zone_id


def _apply_zone_assignment_to_row(
    *,
    row: dict[str, object],
    cluster: ClusterProgram,
    family: ConceptFamily,
    zone_assignment: str,
) -> None:
    wall_claim = _wall_claim_for(cluster, zone_assignment, family)
    center_usage = str(row.get("center_usage") or "none")
    row["zone_assignment"] = zone_assignment
    row["wall_claim"] = wall_claim
    row["daylight_relation"] = _daylight_relation_for(
        cluster,
        zone_assignment,
        family,
    )
    row["privacy_relation"] = _privacy_relation_for(cluster, zone_assignment)
    row["placement_bias"] = _placement_bias(
        zone_assignment,
        wall_claim,
        center_usage,
    )


def _mark_zone_complement_role(
    rows: Sequence[dict[str, object]],
    *,
    focal_cluster_id: str,
    primary_cluster_id: str,
    secondary_cluster_id: str,
) -> None:
    for row in rows:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id == focal_cluster_id:
            role: ZoneComplementRole = "focal_claim"
        elif cluster_id in {primary_cluster_id, secondary_cluster_id}:
            role = "topology_complement"
        else:
            role = "neutral"
        row["zone_complement_role"] = role


def _preferred_region_for_family(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster: ClusterProgram,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> str:
    scenario_role = _scenario_role_for_cluster(
        cluster=cluster,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )
    if scenario_role == "primary":
        return scenario.primary_zone_id
    if scenario_role == "secondary":
        return scenario.secondary_zone_id
    if scenario_role == "storage":
        return scenario.storage_zone_id
    if scenario_role == "support" and cluster.role_kind not in {"work", "sleep"}:
        return scenario.support_zone_id
    if family == "focal_axis":
        if cluster.cluster_id in {primary_cluster_id, secondary_cluster_id}:
            return "primary_focal_wall_zone"
        return "storage_service_edge_zone"
    if family == "open_center":
        if cluster.cluster_id in {primary_cluster_id, secondary_cluster_id}:
            return "primary_focal_wall_zone"
        return "edge_loading_zone"
    if family == "edge_weighted":
        return (
            "storage_service_edge_zone"
            if cluster.role_kind == "storage"
            else "edge_loading_zone"
        )
    if family == "zoned":
        if cluster.cluster_id == primary_cluster_id:
            return "floating_support_zone"
        if cluster.cluster_id == secondary_cluster_id:
            return "primary_focal_wall_zone"
        if cluster.role_kind == "work":
            return "daylight_biased_zone"
        if cluster.role_kind == "sleep":
            return "quiet_private_deep_zone"
        return "storage_service_edge_zone"
    if family == "daylight_oriented":
        if cluster.role_kind in {"work", "lounge", "social_anchor"}:
            return "daylight_biased_zone"
        if cluster.cluster_id == secondary_cluster_id:
            return "primary_focal_wall_zone"
        return "edge_loading_zone"
    return "primary_focal_wall_zone"


def _topology_policy_for_family(
    family: ConceptFamily,
    *,
    style_policy: Mapping[str, object],
) -> dict[str, object]:
    policies: dict[ConceptFamily, dict[str, object]] = {
        "focal_axis": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "medium",
            "wall_loading_bias": "focal_wall_first",
            "daylight_bias": "secondary_cluster_preferred",
            "entry_avoidance_strength": "high",
            "secondary_zone_placement_bias": "focal_edge",
        },
        "open_center": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "high",
            "wall_loading_bias": "balanced_edge",
            "daylight_bias": "support_allowed",
            "entry_avoidance_strength": "high",
            "secondary_zone_placement_bias": "edge_recede",
        },
        "edge_weighted": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "very_high",
            "wall_loading_bias": "perimeter_heavy",
            "daylight_bias": "avoid_blocking",
            "entry_avoidance_strength": "high",
            "secondary_zone_placement_bias": "perimeter",
        },
        "zoned": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "medium",
            "wall_loading_bias": "zone_specific",
            "daylight_bias": "role_based",
            "entry_avoidance_strength": "medium",
            "secondary_zone_placement_bias": "separate_support_zone",
        },
        "daylight_oriented": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "medium",
            "wall_loading_bias": "balanced",
            "daylight_bias": "primary_support_preferred",
            "entry_avoidance_strength": "high",
            "secondary_zone_placement_bias": "daylight_if_not_focal",
        },
    }
    policy = dict(policies[family])
    layout_policy = _style_layout_policy(style_policy)
    if (
        _bias_level(layout_policy.get("center_openness_bias")) >= 3
        and family != "zoned"
    ):
        policy["reserve_center_degree"] = "very_high"
        policy["secondary_zone_placement_bias"] = "edge_recede"
    elif _bias_level(layout_policy.get("center_openness_bias")) <= 1:
        policy["reserve_center_degree"] = "medium"
    wall_loading_bias = str(layout_policy.get("wall_loading_bias") or "")
    if wall_loading_bias in {"medium_high", "perimeter_heavy"}:
        policy["wall_loading_bias"] = "perimeter_heavy"
        policy["secondary_zone_placement_bias"] = "perimeter"
    elif wall_loading_bias == "focal_balanced":
        policy["wall_loading_bias"] = "focal_wall_first"
    if _bias_level(layout_policy.get("daylight_bias")) >= 3:
        policy["daylight_bias"] = "primary_support_preferred"
    policy["style_name"] = str(style_policy.get("style_name") or "")
    policy["style_density_target"] = str(layout_policy.get("target_density") or "")
    policy["style_visual_balance"] = str(layout_policy.get("visual_balance_bias") or "")
    return policy


def _macro_constraints_for_concept(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    room_model: Mapping[str, object],
    macro_region_map: Mapping[str, object],
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> dict[str, object]:
    protected = _mapping_or_empty(macro_region_map.get("protected_topology"))
    entry_refs = _string_list(protected.get("entry_landing_zones"))
    corridor_refs = _string_list(protected.get("primary_circulation_corridors"))
    center_refs = _string_list(protected.get("center_openness_regions")) or [
        "room_center"
    ]
    window_refs = _opening_ids(
        _mapping_or_empty(room_model.get("openings")).get("windows"), "window"
    )

    keep_open_regions = [
        {"type": "entry_buffer", "near": ref, "priority": "high"} for ref in entry_refs
    ]
    if family in {"open_center", "edge_weighted"}:
        keep_open_regions.extend(
            {"type": "center_lane", "near": ref, "priority": "high"}
            for ref in center_refs
        )
    elif family == "daylight_oriented":
        keep_open_regions.extend(
            {"type": "window_buffer", "near": ref, "priority": "medium"}
            for ref in window_refs
        )

    reserved_regions = [
        {"region": ref, "reason": "entry landing must stay usable"}
        for ref in entry_refs
    ]
    reserved_regions.extend(
        {"region": ref, "reason": "primary circulation corridor must stay usable"}
        for ref in corridor_refs
    )
    if family in {"open_center", "edge_weighted"}:
        reserved_regions.extend(
            {"region": ref, "reason": "center openness is the concept driver"}
            for ref in center_refs
        )

    separation = []
    if family in {"open_center", "edge_weighted", "zoned"} and primary_cluster_id:
        for row in cluster_zone_plan:
            cluster_id = _clean_str(row.get("cluster_id"))
            if cluster_id and cluster_id not in {
                primary_cluster_id,
                secondary_cluster_id,
            }:
                separation.append(
                    {
                        "a": cluster_id,
                        "b": primary_cluster_id,
                        "preference": "separate_macro_fields",
                        "priority": "high" if family != "open_center" else "medium",
                    }
                )

    alignment = []
    if primary_cluster_id and secondary_cluster_id:
        alignment.append(
            {
                "a": primary_cluster_id,
                "b": secondary_cluster_id,
                "preference": "focal_face_axis"
                if family == "focal_axis"
                else scenario.pair_type,
                "priority": "high",
            }
        )
    anchor_region_constraints = [
        {
            "cluster_id": str(row.get("cluster_id")),
            "required_region_ids": list(
                _sequence_or_empty(row.get("required_region_ids"))
            ),
            "forbidden_region_ids": list(
                _sequence_or_empty(row.get("forbidden_region_ids"))
            ),
            "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
            "anchor_strength": str(row.get("anchor_strength") or "medium"),
            "scenario_role": str(row.get("scenario_role") or "support"),
        }
        for row in cluster_zone_plan
        if isinstance(row, Mapping) and _clean_str(row.get("cluster_id")) is not None
    ]
    return {
        "keep_open_regions": keep_open_regions,
        "reserved_regions": reserved_regions,
        "protected_topology": _concept_protected_topology(
            protected,
            family=family,
            scenario=scenario,
        ),
        "cluster_separation_preferences": separation,
        "cluster_alignment_preferences": alignment,
        "anchor_region_constraints": anchor_region_constraints,
    }


def _concept_protected_topology(
    protected: Mapping[str, object],
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
) -> list[dict[str, object]]:
    zones = [
        dict(zone)
        for zone in _sequence_or_empty(protected.get("operational_zones"))
        if isinstance(zone, Mapping)
    ]
    for zone in zones:
        if (
            family in {"open_center", "edge_weighted"}
            and zone.get("zone_type") == "center_openness_core"
        ):
            zone["priority"] = "critical"
            zone["max_overlap_ratio"] = 0.05
        elif (
            family == "daylight_oriented"
            and zone.get("zone_type") == "center_openness_core"
        ):
            zone["max_overlap_ratio"] = 0.12
        elif (
            scenario.allow_primary_center_overlap
            and zone.get("zone_type") == "center_openness_core"
        ):
            zone["priority"] = "medium"
            zone["enforcement"] = "soft"
            zone["max_overlap_ratio"] = 0.35
            zone["applies_to"] = ["support_clusters", "interaction_hulls"]
            zone["violation_severity"] = "advisory"
    return zones


def _allowed_variant_families_by_cluster(
    *,
    family: ConceptFamily,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        families = _allowed_variant_families_for_row(family=family, row=row)
        out[cluster_id] = families
    return out


def _allowed_variant_families_for_row(
    *,
    family: ConceptFamily,
    row: Mapping[str, object],
) -> list[str]:
    role_kind = str(row.get("role_kind") or "support")
    zone_assignment = str(row.get("zone_assignment") or "")
    wall_claim = str(row.get("wall_claim") or "")
    center_usage = str(row.get("center_usage") or "")
    families: list[str] = []

    if role_kind in {"social_anchor", "lounge"}:
        if center_usage in {"partial", "primary", "open_reserved"}:
            families.append("open_center")
        if _zone_logic_class(zone_assignment) in {"perimeter", "focal_wall"}:
            families.append("perimeter_facing")
        families.extend(["conversation_facing", "social_anchor"])
    elif role_kind in {"media", "focal"}:
        families.extend(["media_facing", "wall_backed_focal"])
        families.append("focal_axis" if family == "focal_axis" else "focal_media")
        if wall_claim == "strong":
            families.append("focal_wall")
    elif role_kind == "work":
        if "daylight" in zone_assignment:
            families.extend(["daylight_work", "window_oriented"])
        families.extend(["work_core", "workflow"])
    elif role_kind == "storage":
        if wall_claim in {"strong", "medium"}:
            families.append("storage_wall")
        families.extend(["edge_storage", "perimeter_storage", "edge_weighted"])
    elif role_kind == "sleep":
        families.extend(
            _sleep_variant_families_for_row(
                family=family,
                zone_assignment=zone_assignment,
                wall_claim=wall_claim,
            )
        )
    else:
        if "daylight" in zone_assignment:
            families.append("window_oriented")
        if _zone_logic_class(zone_assignment) == "perimeter":
            families.append("support_edge")
        families.extend(["base", "perimeter_facing"])

    if family == "open_center" and role_kind not in {
        "media",
        "focal",
        "storage",
        "sleep",
    }:
        families.insert(0, "open_center")
    elif family == "edge_weighted" and role_kind in {"storage", "support"}:
        families.insert(0, "edge_weighted")
    elif (
        family == "daylight_oriented"
        and role_kind != "sleep"
        and "daylight" in zone_assignment
    ):
        families.insert(0, "window_oriented")

    return _bounded_variant_family_list(_uniq(families))


def _sleep_variant_families_for_row(
    *,
    family: ConceptFamily,
    zone_assignment: str,
    wall_claim: str,
) -> list[str]:
    families = list(SLEEP_VARIANT_FAMILIES)
    if wall_claim == "strong":
        families = _promote_variant_family(
            families,
            target_family="headboard_wall_balanced",
        )
    elif wall_claim == "medium":
        families = _promote_variant_family(
            families,
            target_family="headboard_wall_single_side",
        )

    if "daylight" in zone_assignment or family == "daylight_oriented":
        families = _promote_variant_family(
            families,
            target_family="bed_plus_window_side_bench",
        )
    elif family == "edge_weighted":
        families = _promote_variant_family(
            families,
            target_family="bed_plus_storage_buffer",
        )
    elif family in {"focal_axis", "zoned"}:
        families = _promote_variant_family(
            families,
            target_family="headboard_wall_balanced",
        )

    return _uniq(families)


def _promote_variant_family(
    families: Sequence[str],
    *,
    target_family: str,
) -> list[str]:
    promoted = [family for family in families if family != target_family]
    promoted.insert(0, target_family)
    return promoted


def _bounded_variant_family_list(families: Sequence[str]) -> list[str]:
    out = list(families[:MAX_VARIANT_FAMILIES_PER_CLUSTER])
    for fallback in ("base", "support_edge", "perimeter_facing"):
        if len(out) >= MIN_VARIANT_FAMILIES_PER_CLUSTER:
            break
        if fallback not in out:
            out.append(fallback)
    return out[:MAX_VARIANT_FAMILIES_PER_CLUSTER]


def _anchor_region_preferences_by_cluster(
    *,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        region_candidates = [
            dict(candidate)
            for candidate in _sequence_or_empty(row.get("region_candidates"))
            if isinstance(candidate, Mapping)
        ]
        out[cluster_id] = {
            "zone_assignment": str(row.get("zone_assignment") or ""),
            "required_region_ids": list(_string_list(row.get("required_region_ids"))),
            "forbidden_region_ids": list(_string_list(row.get("forbidden_region_ids"))),
            "preferred_region_ids": [
                str(candidate.get("region_id") or "")
                for candidate in region_candidates
                if str(candidate.get("region_id") or "")
            ],
            "wall_claim": str(row.get("wall_claim") or "none"),
            "center_usage": str(row.get("center_usage") or "none"),
            "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
            "anchor_strength": str(row.get("anchor_strength") or "medium"),
            "scenario_role": str(row.get("scenario_role") or "support"),
            "entry_relation": str(row.get("entry_relation") or ""),
            "daylight_relation": str(row.get("daylight_relation") or ""),
            "placement_bias": str(row.get("placement_bias") or "balanced"),
        }
    return out


def _anchor_layout_hints_by_cluster(
    *,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        preferred_region_ids = [
            str(candidate.get("region_id") or "")
            for candidate in _sequence_or_empty(row.get("region_candidates"))
            if isinstance(candidate, Mapping) and str(candidate.get("region_id") or "")
        ]
        out[cluster_id] = {
            "zone_assignment": str(row.get("zone_assignment") or ""),
            "required_region_ids": list(_string_list(row.get("required_region_ids"))),
            "forbidden_region_ids": list(_string_list(row.get("forbidden_region_ids"))),
            "preferred_region_ids": preferred_region_ids,
            "role_kind": str(row.get("role_kind") or ""),
            "wall_claim": str(row.get("wall_claim") or "none"),
            "center_usage": str(row.get("center_usage") or "none"),
            "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
            "anchor_strength": str(row.get("anchor_strength") or "medium"),
            "scenario_role": str(row.get("scenario_role") or "support"),
            "placement_bias": str(row.get("placement_bias") or "balanced"),
        }
    return out


def _anchor_pair_contracts_for_concept(
    *,
    primary_pair_contracts: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in primary_pair_contracts:
        if not isinstance(row, Mapping):
            continue
        out.append(dict(row))
    if not out and primary_cluster_id and secondary_cluster_id:
        out.append(
            {
                "pair_type": "face_each_other",
                "cluster_a": primary_cluster_id,
                "cluster_b": secondary_cluster_id,
                "strength": "high",
                "required": True,
            }
        )
    return out


def _object_solver_policy_for_concept(
    *,
    family: ConceptFamily,
    cluster_programs: Sequence[ClusterProgram],
) -> dict[str, object]:
    protected_by_cluster: dict[str, list[str]] = {}
    droppable_by_cluster: dict[str, list[str]] = {}
    placement_order_by_cluster: dict[str, list[str]] = {}
    for cluster in cluster_programs:
        placement_order_by_cluster[cluster.cluster_id] = list(cluster.object_ids)
        if cluster.dominant_anchor_object_id:
            protected_by_cluster.setdefault(cluster.cluster_id, []).append(
                cluster.dominant_anchor_object_id
            )
        optional_ids = [
            object_id
            for object_id in cluster.object_ids
            if object_id not in protected_by_cluster.get(cluster.cluster_id, [])
        ]
        droppable_by_cluster[cluster.cluster_id] = optional_ids
    return {
        "placement_mode": "anchor_first_object_level",
        "anchor_stage_first": True,
        "family": family,
        "support_stage_enabled": True,
        "micro_refine_inside_solver": True,
        "placement_order_by_cluster": placement_order_by_cluster,
        "protected_ids_by_cluster": protected_by_cluster,
        "droppable_ids_by_cluster": droppable_by_cluster,
    }


def _critical_object_orientations_for_concept(
    *,
    family: ConceptFamily,
    cluster_programs: Sequence[ClusterProgram],
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    rows_by_cluster = {
        str(row.get("cluster_id")): row
        for row in cluster_zone_plan
        if _clean_str(row.get("cluster_id")) is not None
    }
    out: list[dict[str, object]] = []
    for cluster in cluster_programs:
        row = rows_by_cluster.get(cluster.cluster_id)
        if row is None:
            continue
        object_id = _critical_object_id(cluster)
        if object_id is None or not _object_orientation_is_critical(cluster, row):
            continue
        target_cluster_id = _orientation_target_cluster_id(
            cluster_id=cluster.cluster_id,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        intents = _critical_object_orientation_intents(
            family=family,
            cluster=cluster,
            row=row,
            target_cluster_id=target_cluster_id,
        )
        if not intents:
            continue
        item: dict[str, object] = {
            "cluster_id": cluster.cluster_id,
            "object_id": object_id,
            "intents": intents,
            "priority": "high"
            if cluster.cluster_id in {primary_cluster_id, secondary_cluster_id}
            else "medium",
            "reason": "Critical object orientation emitted by macro concept.",
        }
        if target_cluster_id is not None and "face_cluster" in intents:
            item["target_cluster_id"] = target_cluster_id
        else:
            item["target_region"] = str(row.get("zone_assignment") or "")
        out.append(item)
    return out


def _critical_object_id(cluster: ClusterProgram) -> str | None:
    return cluster.object_ids[0] if cluster.object_ids else None


def _object_orientation_is_critical(
    cluster: ClusterProgram,
    row: Mapping[str, object],
) -> bool:
    if cluster.priority == "core":
        return True
    if cluster.role_kind in {"work", "media", "focal"}:
        return True
    return cluster.role_kind == "storage" and row.get("wall_claim") == "strong"


def _orientation_target_cluster_id(
    *,
    cluster_id: str,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> str | None:
    if cluster_id == primary_cluster_id:
        return secondary_cluster_id
    if cluster_id == secondary_cluster_id:
        return primary_cluster_id
    return None


def _critical_object_orientation_intents(
    *,
    family: ConceptFamily,
    cluster: ClusterProgram,
    row: Mapping[str, object],
    target_cluster_id: str | None,
) -> list[str]:
    intents: list[str] = []
    if row.get("wall_claim") in {"strong", "medium"}:
        intents.append("back_to_wall")
    if target_cluster_id is not None:
        intents.extend(["face_cluster", "align_with_focal_axis"])
    if cluster.role_kind in {"social_anchor", "lounge", "work"}:
        intents.append("front_to_open_space")
    if cluster.role_kind in {"media", "focal"}:
        intents.extend(["front_to_open_space", "avoid_entry_axis"])
    if family in {"open_center", "edge_weighted"}:
        intents.append("front_to_open_space")
    if not intents and cluster.priority == "core":
        intents.append("front_to_open_space")
    return _uniq(intents)


def _primary_pair_contracts_for_concept(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    if primary_cluster_id is None or secondary_cluster_id is None:
        return []
    primary_row = next(
        (
            row
            for row in cluster_zone_plan
            if row.get("cluster_id") == primary_cluster_id
        ),
        {},
    )
    secondary_row = next(
        (
            row
            for row in cluster_zone_plan
            if row.get("cluster_id") == secondary_cluster_id
        ),
        {},
    )
    pair_type = _primary_pair_type_for_family(
        family=family,
        scenario=scenario,
        primary_row=primary_row,
        secondary_row=secondary_row,
    )
    contracts = [
        {
            "pair_type": pair_type,
            "cluster_a": primary_cluster_id,
            "cluster_b": secondary_cluster_id,
            "strength": "high",
            "required": True,
        }
    ]
    if family in {"focal_axis", "daylight_oriented"}:
        contracts.append(
            {
                "pair_type": "supports_use_axis",
                "cluster_a": primary_cluster_id,
                "cluster_b": secondary_cluster_id,
                "strength": "medium",
                "required": family == "focal_axis",
            }
        )
    return contracts


def _primary_pair_type_for_family(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    primary_row: Mapping[str, object],
    secondary_row: Mapping[str, object],
) -> str:
    if scenario.pair_type:
        return scenario.pair_type
    if family == "zoned":
        return "separate_but_legible"
    if family == "edge_weighted":
        return "buffered_support"
    if family == "open_center":
        return "ring_around_anchor"
    if (
        primary_row.get("zone_complement_role") == "focal_claim"
        or secondary_row.get("zone_complement_role") == "focal_claim"
    ):
        return "face_each_other"
    return "supports_use_axis"


def _concept_readiness_requirements(
    *,
    family: ConceptFamily,
    primary_pair_contracts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    required_pair_contracts = [
        str(contract.get("pair_type"))
        for contract in primary_pair_contracts
        if bool(contract.get("required")) and contract.get("pair_type")
    ]
    diagnostics = [
        "circulation_clear",
        "entry_preserved",
        "dominant_anchor_correct",
        "workflow_preserved",
    ]
    if family in {"open_center", "edge_weighted"}:
        diagnostics.append("center_openness_preserved")
    return {
        "required_publishable_diagnostics": diagnostics,
        "required_pair_contracts": required_pair_contracts,
        "required_center_preservation": "high"
        if family in {"open_center", "edge_weighted"}
        else "medium",
        "required_entry_preservation": True,
    }


def _variant_bias_weights_for_concept(
    family: ConceptFamily,
    *,
    style_policy: Mapping[str, object],
) -> dict[str, float]:
    weights: dict[ConceptFamily, dict[str, float]] = {
        "focal_axis": {
            "family_match_weight": 1.45,
            "center_preserve_weight": 1.10,
            "focal_axis_weight": 1.55,
            "entry_clear_weight": 1.45,
            "wall_loading_weight": 1.35,
            "daylight_alignment_weight": 1.05,
            "edge_loading_weight": 1.15,
            "symmetry_weight": 1.35,
        },
        "open_center": {
            "family_match_weight": 1.40,
            "center_preserve_weight": 1.55,
            "focal_axis_weight": 1.25,
            "entry_clear_weight": 1.50,
            "wall_loading_weight": 1.15,
            "daylight_alignment_weight": 1.10,
            "edge_loading_weight": 1.25,
            "symmetry_weight": 1.10,
        },
        "edge_weighted": {
            "family_match_weight": 1.35,
            "center_preserve_weight": 1.50,
            "focal_axis_weight": 1.10,
            "entry_clear_weight": 1.45,
            "wall_loading_weight": 1.50,
            "daylight_alignment_weight": 1.00,
            "edge_loading_weight": 1.55,
            "symmetry_weight": 1.05,
        },
        "zoned": {
            "family_match_weight": 1.35,
            "center_preserve_weight": 1.20,
            "focal_axis_weight": 1.20,
            "entry_clear_weight": 1.35,
            "wall_loading_weight": 1.20,
            "daylight_alignment_weight": 1.15,
            "edge_loading_weight": 1.25,
            "symmetry_weight": 1.05,
        },
        "daylight_oriented": {
            "family_match_weight": 1.40,
            "center_preserve_weight": 1.20,
            "focal_axis_weight": 1.30,
            "entry_clear_weight": 1.45,
            "wall_loading_weight": 1.15,
            "daylight_alignment_weight": 1.55,
            "edge_loading_weight": 1.10,
            "symmetry_weight": 1.05,
        },
    }
    out = dict(weights[family])
    layout_policy = _style_layout_policy(style_policy)
    if _bias_level(layout_policy.get("center_openness_bias")) >= 3:
        out["center_preserve_weight"] = min(out["center_preserve_weight"] + 0.15, 1.7)
    if _bias_level(layout_policy.get("daylight_bias")) >= 3:
        out["daylight_alignment_weight"] = min(
            out["daylight_alignment_weight"] + 0.15,
            1.7,
        )
    if str(layout_policy.get("wall_loading_bias") or "") in {
        "medium_high",
        "perimeter_heavy",
    }:
        out["wall_loading_weight"] = min(out["wall_loading_weight"] + 0.15, 1.7)
        out["edge_loading_weight"] = min(out["edge_loading_weight"] + 0.10, 1.7)
    return out


def _concept_to_solver_plan(
    *,
    concept: Mapping[str, object],
    room_model: Mapping[str, object],
    room_type: str,
) -> dict[str, object]:
    cluster_zone_plan = [
        row
        for row in _sequence_or_empty(concept.get("cluster_zone_plan"))
        if isinstance(row, Mapping)
    ]
    primary_cluster_id = _primary_from_concept(cluster_zone_plan)
    secondary_cluster_id = _secondary_from_concept(
        cluster_zone_plan, primary_cluster_id
    )
    family = str(concept.get("concept_family") or "focal_axis")
    policy = _mapping_or_empty(concept.get("topology_policy"))
    constraints = _mapping_or_empty(concept.get("macro_constraints"))
    pair_contracts = [
        row
        for row in _sequence_or_empty(concept.get("primary_pair_contracts"))
        if isinstance(row, Mapping)
    ]
    if not pair_contracts and primary_cluster_id and secondary_cluster_id:
        pair_contracts = [
            {
                "pair_type": "face_each_other",
                "cluster_a": primary_cluster_id,
                "cluster_b": secondary_cluster_id,
                "strength": "high",
                "required": True,
            }
        ]

    affinities = [
        _solver_affinity_for_assignment(row, policy) for row in cluster_zone_plan
    ]
    orientations = [
        item
        for row in cluster_zone_plan
        if (
            item := _solver_orientation_for_assignment(
                row, primary_cluster_id, secondary_cluster_id
            )
        )
        is not None
    ]
    relations = []
    directional_relations = []
    relations.extend(_solver_relations_from_pair_contracts(pair_contracts))
    directional_relations.extend(
        _solver_directional_relations_from_pair_contracts(pair_contracts)
    )
    for row in _sequence_or_empty(constraints.get("cluster_separation_preferences")):
        if not isinstance(row, Mapping):
            continue
        a = _clean_str(row.get("a"))
        b = _clean_str(row.get("b"))
        if a and b and a != b:
            relations.append(
                {
                    "a": a,
                    "b": b,
                    "relation": "far_if_possible",
                    "priority": _priority_text(row.get("priority")),
                    "reason": "Concept-specific topology keeps these macro fields distinct.",
                }
            )

    keep_open_regions = [
        {
            "type": str(row.get("type")),
            "near": str(row.get("near")),
            "priority": _priority_text(row.get("priority")),
            "reason": "Protected by deterministic macro concept policy.",
        }
        for row in _sequence_or_empty(constraints.get("keep_open_regions"))
        if isinstance(row, Mapping) and row.get("type") and row.get("near")
    ]
    return {
        "status": "OK",
        "room_id": _room_id(room_model),
        "room_type": room_type,
        "planner_kind": "seed_concept_generator",
        "macro_concept": {
            key: value
            for key, value in concept.items()
            if key != "solver_relation_plan"
        },
        "cluster_affinities": _dedupe_by_cluster(affinities),
        "cluster_orientations": _dedupe_by_cluster(orientations),
        "object_orientations": _concept_object_orientations_for_solver(concept),
        "anchor_region_preferences_by_cluster": dict(
            _mapping_or_empty(concept.get("anchor_region_preferences_by_cluster"))
        ),
        "anchor_layout_hints_by_cluster": _anchor_layout_hints_by_cluster(
            cluster_zone_plan=cluster_zone_plan
        ),
        "anchor_pair_contracts": list(
            _sequence_or_empty(concept.get("anchor_pair_contracts"))
        ),
        "object_solver_policy": dict(
            _mapping_or_empty(concept.get("object_solver_policy"))
        ),
        "macro_region_map": dict(
            _mapping_or_empty(concept.get("macro_region_map_snapshot"))
        ),
        "allowed_variant_families_by_cluster": dict(
            _mapping_or_empty(concept.get("allowed_variant_families_by_cluster"))
        ),
        "variant_bias_weights": dict(
            _mapping_or_empty(concept.get("variant_bias_weights"))
        ),
        "concept_readiness_requirements": dict(
            _mapping_or_empty(concept.get("concept_readiness_requirements"))
        ),
        "cluster_relations": _dedupe_relations(relations),
        "cluster_directional_relations": _dedupe_relations(directional_relations),
        "circulation_plan": {
            "main_paths": _main_paths(room_model, primary_cluster_id),
            "keep_open_regions": _dedupe_keep_open(keep_open_regions),
        },
        "layout_intent_profile": {
            "focus_mode": "viewing" if family == "focal_axis" else "mixed",
            "primary_cluster_id": primary_cluster_id,
            "secondary_cluster_id": secondary_cluster_id,
            "circulation_priority": "high"
            if policy.get("reserve_center_degree") in {"high", "very_high"}
            else "medium",
            "center_open_preference": "high"
            if policy.get("reserve_center_degree") in {"high", "very_high"}
            else "medium",
            "support_cluster_behavior": "recede"
            if family in {"open_center", "edge_weighted", "zoned"}
            else "balanced",
            "distribution_mode": "edge_weighted"
            if family == "edge_weighted"
            else "focal_grouped"
            if family == "focal_axis"
            else "zoned"
            if family == "zoned"
            else "balanced",
        },
        "placement_guidelines": [
            "Follow macro_concept.cluster_zone_plan before applying local repair.",
            "Preserve entry landing, primary corridor, and declared center openness before support clusters compete for space.",
        ],
        "notes": [
            f"Generated by deterministic SeedConceptGenerator family={family}.",
            "Legacy relation fields are a solver projection of the macro concept, not the planning source of truth.",
        ],
        "missing": [],
    }


def _solver_affinity_for_assignment(
    row: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict[str, object]:
    cluster_id = str(row.get("cluster_id") or "")
    zone = str(row.get("zone_assignment") or "")
    wall_claim = str(row.get("wall_claim") or "")
    center_usage = str(row.get("center_usage") or "")
    prefer: list[str] = []
    avoid = ["entry_blocking", "main_path"]
    if wall_claim in {"strong", "medium"} or "wall" in zone or "edge" in zone:
        prefer.extend(["wall", "recess_or_edge"])
    if zone == "daylight_biased_zone":
        prefer.append("window_side")
    if (
        zone == "quiet_private_deep_zone"
        or row.get("entry_relation") == "avoid_direct_entry_conflict"
    ):
        prefer.append("far_from_entry")
    if zone == "entry_adjacent_active_zone":
        prefer.append("entry_side")
    reserve_center = policy.get("reserve_center_degree") in {"high", "very_high"}
    if center_usage in {"none", "open_reserved"} or reserve_center:
        avoid.append("center")
    elif center_usage in {"partial", "primary"}:
        prefer.append("center")
    if str(row.get("daylight_relation") or "") == "avoid_window_blocking":
        avoid.append("window_blocking")
    return {
        "cluster_id": cluster_id,
        "prefer": _uniq(prefer),
        "avoid": _uniq(avoid),
        "priority": "high" if wall_claim == "strong" else "medium",
        "reason": "Projection from concept zone assignment.",
    }


def _solver_orientation_for_assignment(
    row: Mapping[str, object],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> dict[str, object] | None:
    cluster_id = _clean_str(row.get("cluster_id"))
    if cluster_id is None:
        return None
    intents: list[str] = []
    target_cluster_id: str | None = None
    if row.get("wall_claim") in {"strong", "medium"}:
        intents.extend(["back_to_wall", "axis_parallel_wall"])
    if cluster_id == primary_cluster_id and secondary_cluster_id:
        intents.extend(["face_cluster", "inward_to_room"])
        target_cluster_id = secondary_cluster_id
    elif cluster_id == secondary_cluster_id and primary_cluster_id:
        intents.extend(["face_cluster", "access_to_open_space"])
        target_cluster_id = primary_cluster_id
    elif row.get("placement_bias") != "wall_backed":
        intents.append("access_to_open_space")
    intents = _uniq(intents)
    if not intents:
        return None
    if "face_cluster" not in intents:
        target_cluster_id = None
    return {
        "cluster_id": cluster_id,
        "intents": intents,
        "target_cluster_id": target_cluster_id,
        "priority": "high"
        if cluster_id in {primary_cluster_id, secondary_cluster_id}
        else "medium",
        "reason": "Projection from concept wall/center/focal policy.",
    }


def _solver_relations_from_pair_contracts(
    pair_contracts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
    for contract in pair_contracts:
        cluster_a = _clean_str(contract.get("cluster_a"))
        cluster_b = _clean_str(contract.get("cluster_b"))
        if cluster_a is None or cluster_b is None or cluster_a == cluster_b:
            continue
        pair_type = str(contract.get("pair_type") or "")
        relation = "near"
        if pair_type in {"buffered_support", "separate_but_legible"}:
            relation = "far_if_possible"
        relations.append(
            {
                "a": cluster_a,
                "b": cluster_b,
                "relation": relation,
                "priority": _contract_priority(contract),
                "reason": f"Projection from primary_pair_contract={pair_type}.",
            }
        )
    return relations


def _solver_directional_relations_from_pair_contracts(
    pair_contracts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
    for contract in pair_contracts:
        cluster_a = _clean_str(contract.get("cluster_a"))
        cluster_b = _clean_str(contract.get("cluster_b"))
        if cluster_a is None or cluster_b is None or cluster_a == cluster_b:
            continue
        pair_type = str(contract.get("pair_type") or "")
        relation = _directional_relation_for_pair_type(pair_type)
        if relation is None:
            continue
        relations.append(
            {
                "a": cluster_a,
                "b": cluster_b,
                "relation": relation,
                "priority": _contract_priority(contract),
                "reason": f"Projection from primary_pair_contract={pair_type}.",
            }
        )
    return relations


def _directional_relation_for_pair_type(pair_type: str) -> str | None:
    if pair_type in {"face_each_other", "ring_around_anchor"}:
        return "face_each_other"
    if pair_type in {"supports_use_axis", "workflow_chain"}:
        return "access_faces_other"
    return None


def _contract_priority(contract: Mapping[str, object]) -> str:
    if bool(contract.get("required")):
        return "high"
    strength = str(contract.get("strength") or "medium").strip().lower()
    return strength if strength in {"high", "medium", "low"} else "medium"


def _concept_object_orientations_for_solver(
    concept: Mapping[str, object],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for item in _sequence_or_empty(concept.get("critical_object_orientations")):
        if not isinstance(item, Mapping):
            continue
        object_id = _clean_str(item.get("object_id"))
        cluster_id = _clean_str(item.get("cluster_id"))
        if object_id is None or cluster_id is None:
            continue
        out.append(dict(item))
    return out


def _empty_solver_plan(room_model: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "UNSAT",
        "room_id": _room_id(room_model),
        "cluster_affinities": [],
        "cluster_orientations": [],
        "object_orientations": [],
        "cluster_relations": [],
        "cluster_directional_relations": [],
        "circulation_plan": {"main_paths": [], "keep_open_regions": []},
        "layout_intent_profile": None,
        "placement_guidelines": [],
        "notes": ["No macro concept could be generated."],
        "missing": [],
    }


def _macro_region_to_dict(region: MacroRegion) -> dict[str, object]:
    return {
        "region_id": region.region_id,
        "region_type": region.region_type,
        "label": region.label,
        "source_ids": list(region.source_ids),
        "tags": list(region.tags),
    }


def _region_candidate_to_dict(candidate: RegionCandidate) -> dict[str, object]:
    return {
        "cluster_id": candidate.cluster_id,
        "region_id": candidate.region_id,
        "region_type": candidate.region_type,
        "score": candidate.score,
        "reasons": list(candidate.reasons),
    }


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence_or_empty(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return ()


def _string_list(value: object) -> list[str]:
    out: list[str] = []
    for item in _sequence_or_empty(value):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, Mapping):
            ref = _clean_str(
                item.get("id")
                or item.get("region")
                or item.get("region_id")
                or item.get("node_id")
                or item.get("label")
            )
            if ref is not None:
                out.append(ref)
    return _uniq(out)


def _region_refs(value: object) -> list[str]:
    return _string_list(value)


def _opening_ids(value: object, prefix: str) -> list[str]:
    out: list[str] = []
    for index, item in enumerate(_sequence_or_empty(value), start=1):
        if not isinstance(item, Mapping):
            continue
        out.append(_clean_str(item.get("id")) or f"{prefix}_{index}")
    return out


def _wall_region_ids(
    room: Mapping[str, object],
    affordance: Mapping[str, object],
) -> tuple[str, ...]:
    explicit = _region_refs(affordance.get("usable_wall_segments"))
    if explicit:
        return tuple(explicit)
    polygon = _sequence_or_empty(room.get("polygon_ccw"))
    count = max(1, min(6, len(polygon)))
    return tuple(f"wall_segment_{index}" for index in range(1, count + 1))


def _cluster_rules(cluster: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping_or_empty(cluster.get("cluster_rules"))


def _cluster_object_ids(cluster: Mapping[str, object]) -> list[str]:
    out: list[str] = []
    for key in ("members", "local_placements"):
        for item in _sequence_or_empty(cluster.get(key)):
            value = item.get("id") if isinstance(item, Mapping) else item
            text = _clean_str(value)
            if text is not None and text not in out:
                out.append(text)
    footprint = _mapping_or_empty(cluster.get("cluster_footprint"))
    for rect in _sequence_or_empty(footprint.get("rects")):
        if not isinstance(rect, Mapping):
            continue
        text = _clean_str(rect.get("id"))
        if text is not None and text not in out:
            out.append(text)
    return out


def _semantic_role_from_cluster(cluster_id: str, object_ids: Sequence[str]) -> str:
    role = _role_kind(cluster_id, object_ids, "")
    mapping = {
        "social_anchor": "social_anchor",
        "lounge": "social_anchor",
        "media": "focal_anchor",
        "focal": "focal_anchor",
        "storage": "secondary_storage",
        "work": "work_support",
        "sleep": "rest_anchor",
        "service": "service_support",
    }
    return mapping.get(role, "secondary_zone")


def _role_kind(cluster_id: str, object_ids: Sequence[str], semantic_role: str) -> str:
    tokens = " ".join([cluster_id, semantic_role, *object_ids]).lower()
    if any(token in tokens for token in ("sofa", "sectional", "lounge", "seating")):
        return "social_anchor"
    if any(token in tokens for token in ("tv", "media", "screen", "fireplace")):
        return "media"
    if any(
        token in tokens
        for token in ("storage", "cabinet", "console", "shelf", "wardrobe")
    ):
        return "storage"
    if any(token in tokens for token in ("desk", "work", "office", "study")):
        return "work"
    if any(token in tokens for token in ("bed", "sleep")):
        return "sleep"
    return "support"


def _priority(value: object, role_kind: str) -> Priority:
    text = str(value or "").strip().lower()
    if text in {"core", "support", "optional"}:
        return text  # type: ignore[return-value]
    if role_kind in {"social_anchor", "media", "sleep"}:
        return "core"
    if role_kind in {"storage", "work", "support"}:
        return "support"
    return "optional"


def _affinity_level(value: object) -> str:
    text = str(value or "none").strip().lower()
    if text in {"none", "low", "medium", "high"}:
        return text
    return "none"


def _cluster_avoids_entry(cluster: ClusterProgram) -> bool:
    claims = cluster.zone_claims
    avoid_regions = " ".join(_string_list(claims.get("avoid_regions"))).lower()
    relation_types = {
        str(intent.get("type") or "").strip().lower()
        for intent in cluster.relation_intents
    }
    return (
        "entry" in avoid_regions
        or "avoid_entry" in relation_types
        or cluster.role_kind in {"sleep", "media"}
    )


def _seed_region_tags(cluster: Mapping[str, object]) -> list[str]:
    seed_state = _mapping_or_empty(cluster.get("seed_state"))
    return _string_list(seed_state.get("region_tags"))


def _primary_cluster_id(clusters: Sequence[ClusterProgram]) -> str | None:
    ranked = sorted(
        clusters,
        key=lambda cluster: (
            0
            if cluster.role_kind == "social_anchor"
            else 1
            if cluster.priority == "core"
            else 2,
            cluster.cluster_id,
        ),
    )
    return ranked[0].cluster_id if ranked else None


def _secondary_cluster_id(
    clusters: Sequence[ClusterProgram],
    primary_cluster_id: str | None,
) -> str | None:
    ranked = sorted(
        (cluster for cluster in clusters if cluster.cluster_id != primary_cluster_id),
        key=lambda cluster: (
            0
            if cluster.role_kind == "media"
            else 1
            if cluster.priority == "core"
            else 2,
            cluster.cluster_id,
        ),
    )
    return ranked[0].cluster_id if ranked else None


def _wall_claim_for(
    cluster: ClusterProgram,
    zone_assignment: str,
    family: ConceptFamily,
) -> str:
    if family in {"focal_axis", "edge_weighted"} and (
        cluster.role_kind in {"media", "storage"} or "wall" in zone_assignment
    ):
        return "strong"
    if "wall" in zone_assignment or "edge" in zone_assignment:
        return "medium"
    if _affinity_level(cluster.zone_claims.get("wall_affinity")) == "high":
        return "strong"
    return "none"


def _center_usage_for(
    *,
    cluster: ClusterProgram,
    family: ConceptFamily,
    primary_cluster_id: str | None,
    scenario: MacroScenario,
    scenario_role: str,
) -> str:
    if scenario.center_policy == "floating_primary" and scenario_role == "primary":
        return "primary"
    if family in {"open_center", "edge_weighted"}:
        return "open_reserved"
    if family == "zoned" and cluster.cluster_id == primary_cluster_id:
        return "primary"
    if (
        family in {"focal_axis", "daylight_oriented"}
        and cluster.cluster_id == primary_cluster_id
    ):
        return "partial"
    return "none"


def _entry_relation_for(
    cluster: ClusterProgram,
    topology_policy: Mapping[str, object],
) -> str:
    if _cluster_avoids_entry(cluster):
        return "avoid_direct_entry_conflict"
    if topology_policy.get("entry_avoidance_strength") == "medium":
        return "buffer_entry_if_possible"
    return "neutral"


def _daylight_relation_for(
    cluster: ClusterProgram,
    zone_assignment: str,
    family: ConceptFamily,
) -> str:
    if zone_assignment == "daylight_biased_zone":
        return "claim_daylight"
    if family == "daylight_oriented" and cluster.role_kind in {"work", "social_anchor"}:
        return "daylight_preferred"
    if cluster.role_kind in {"media", "storage"}:
        return "avoid_window_blocking"
    return "neutral"


def _privacy_relation_for(cluster: ClusterProgram, zone_assignment: str) -> str:
    if zone_assignment == "quiet_private_deep_zone":
        return "claim_privacy"
    if _affinity_level(cluster.zone_claims.get("privacy_affinity")) in {
        "medium",
        "high",
    }:
        return "privacy_preferred"
    return "neutral"


def _placement_bias(zone_assignment: str, wall_claim: str, center_usage: str) -> str:
    if wall_claim == "strong":
        return "wall_backed"
    if zone_assignment == "daylight_biased_zone":
        return "daylight_edge"
    if center_usage in {"partial", "primary"}:
        return "float_allowed"
    if "edge" in zone_assignment or "storage" in zone_assignment:
        return "edge_loaded"
    return "balanced"


def _spatial_character(family: ConceptFamily, room_type: str) -> str:
    by_family = {
        "focal_axis": f"{room_type}_focal_axis_balanced",
        "open_center": f"{room_type}_open_center_clear_circulation",
        "edge_weighted": f"{room_type}_perimeter_loaded_open_field",
        "zoned": f"{room_type}_primary_secondary_zoned",
        "daylight_oriented": f"{room_type}_daylight_oriented_support",
    }
    return by_family[family]


def _concept_score_prior(
    family: ConceptFamily,
    *,
    style_policy: Mapping[str, object],
) -> dict[str, float]:
    priors = {
        "focal_axis": (0.95, 0.90, 0.96, 0.89),
        "open_center": (0.92, 0.93, 0.90, 0.94),
        "edge_weighted": (0.90, 0.91, 0.88, 0.93),
        "zoned": (0.93, 0.89, 0.94, 0.88),
        "daylight_oriented": (0.91, 0.90, 0.92, 0.91),
    }
    functionality, naturalness, semantic, spatial = priors[family]
    layout_policy = _style_layout_policy(style_policy)
    if (
        family == "open_center"
        and _bias_level(layout_policy.get("center_openness_bias")) >= 3
    ):
        naturalness += 0.04
        spatial += 0.04
    if (
        family == "daylight_oriented"
        and _bias_level(layout_policy.get("daylight_bias")) >= 3
    ):
        naturalness += 0.04
        semantic += 0.03
    if family == "edge_weighted" and str(layout_policy.get("wall_loading_bias")) in {
        "medium_high",
        "perimeter_heavy",
    }:
        spatial += 0.04
    if family == "focal_axis" and _bias_level(layout_policy.get("symmetry_bias")) >= 3:
        semantic += 0.04
        spatial += 0.03
    return {
        "functionality": min(functionality, 1.0),
        "naturalness": min(naturalness, 1.0),
        "semantic_coherence": min(semantic, 1.0),
        "spatial_quality": min(spatial, 1.0),
    }


def _diversity_signature(
    *,
    family: ConceptFamily,
    cluster_zone_plan: Sequence[Mapping[str, object]],
    topology_policy: Mapping[str, object],
) -> str:
    scenario_id = next(
        (
            str(row.get("macro_scenario_id") or "")
            for row in cluster_zone_plan
            if str(row.get("macro_scenario_id") or "")
        ),
        "",
    )
    assignments = ",".join(
        f"{row.get('cluster_id')}:{row.get('zone_assignment')}:"
        f"{row.get('preferred_wall_side')}:{row.get('center_usage')}:"
        f"{row.get('wall_claim')}"
        for row in cluster_zone_plan
    )
    return (
        f"{family}|scenario={scenario_id}|"
        f"center={topology_policy.get('reserve_center_degree')}|"
        f"wall={topology_policy.get('wall_loading_bias')}|{assignments}"
    )


def _primary_from_concept(
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> str | None:
    for row in cluster_zone_plan:
        if row.get("priority") == "core" and row.get("center_usage") in {
            "partial",
            "primary",
        }:
            return _clean_str(row.get("cluster_id"))
    for row in cluster_zone_plan:
        if row.get("priority") == "core":
            return _clean_str(row.get("cluster_id"))
    return (
        _clean_str(cluster_zone_plan[0].get("cluster_id"))
        if cluster_zone_plan
        else None
    )


def _secondary_from_concept(
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
) -> str | None:
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if (
            cluster_id
            and cluster_id != primary_cluster_id
            and row.get("wall_claim") == "strong"
        ):
            return cluster_id
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id and cluster_id != primary_cluster_id:
            return cluster_id
    return None


def _main_paths(
    room_model: Mapping[str, object],
    primary_cluster_id: str | None,
) -> list[dict[str, object]]:
    if primary_cluster_id is None:
        return []
    openings = _mapping_or_empty(room_model.get("openings"))
    return [
        {
            "from": door_id,
            "to_cluster": primary_cluster_id,
            "priority": "high",
            "reason": "Primary circulation path from entry to core cluster.",
        }
        for door_id in _opening_ids(openings.get("doors"), "door")[:2]
    ]


def _dedupe_by_cluster(
    rows: Sequence[Mapping[str, object] | None],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None or cluster_id in seen:
            continue
        out.append(dict(row))
        seen.add(cluster_id)
    return out


def _dedupe_relations(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        a = _clean_str(row.get("a"))
        b = _clean_str(row.get("b"))
        if a is None or b is None or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        out.append(dict(row))
        seen.add(key)
    return out


def _dedupe_keep_open(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        region_type = _clean_str(row.get("type"))
        near = _clean_str(row.get("near"))
        if region_type is None or near is None:
            continue
        key = (region_type, near)
        if key in seen:
            continue
        out.append(dict(row))
        seen.add(key)
    return out


def _priority_text(value: object) -> str:
    text = str(value or "medium").strip().lower()
    return text if text in {"high", "medium", "low"} else "medium"


def _family_from_text(text: str | None) -> ConceptFamily:
    lowered = str(text or "").lower()
    for family in CONCEPT_FAMILIES:
        if family in lowered or family.replace("_", "-") in lowered:
            return family
    if "open center" in lowered:
        return "open_center"
    if "edge" in lowered or "perimeter" in lowered:
        return "edge_weighted"
    if "daylight" in lowered or "window" in lowered:
        return "daylight_oriented"
    if "zoned" in lowered or "separate support" in lowered:
        return "zoned"
    return "focal_axis"


def _uniq(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out
