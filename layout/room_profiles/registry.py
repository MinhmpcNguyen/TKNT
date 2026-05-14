from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from layout.room_profiles.base import RoomProfile, normalize_profile_token
from layout.room_profiles.kitchen import KITCHEN_PROFILE
from layout.room_profiles.legacy import (
    BEDROOM_LEGACY_PROFILE,
    LIVING_ROOM_LEGACY_PROFILE,
)

ROOM_PROFILES: tuple[RoomProfile, ...] = (
    KITCHEN_PROFILE,
    BEDROOM_LEGACY_PROFILE,
    LIVING_ROOM_LEGACY_PROFILE,
)


def resolve_room_profile(room_type: object) -> RoomProfile | None:
    for profile in ROOM_PROFILES:
        if profile.matches_room_type(room_type):
            return profile
    return None


def semantic_room_rule_for(room_type: object) -> dict[str, Any] | None:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return None
    return profile.semantic_room_rule(str(room_type or ""))


def apply_profile_capacity_model(
    capacity_model: dict[str, Any],
    *,
    room_type: object,
) -> dict[str, Any]:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return capacity_model
    return profile.apply_capacity_policy(capacity_model, str(room_type or ""))


def semantic_placements_for_members(
    *,
    room_type: object,
    cluster_id: str,
    members: Sequence[str],
    anchors: Sequence[str],
) -> list[dict[str, Any]]:
    profile = resolve_room_profile(room_type)
    if profile is not None:
        rows = profile.semantic_placements_for_members(cluster_id, members, anchors)
        if rows:
            return rows
    for candidate in _profiles_for_members(members):
        rows = candidate.semantic_placements_for_members(cluster_id, members, anchors)
        if rows:
            return rows
    return []


def all_profile_object_aliases() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for profile in ROOM_PROFILES:
        for canonical, aliases in profile.object_aliases.items():
            out[normalize_profile_token(canonical)] = _request_alias_variants(
                canonical,
                aliases,
            )
    return out


def all_profile_scoring_aliases() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for profile in ROOM_PROFILES:
        aliases = profile.scoring_aliases or profile.object_aliases
        for canonical, values in aliases.items():
            out[normalize_profile_token(canonical)] = _scoring_alias_variants(
                canonical,
                values,
            )
    return out


def all_profile_non_functional_contract_types() -> frozenset[str]:
    values: set[str] = set()
    for profile in ROOM_PROFILES:
        values.update(profile.non_functional_contract_types)
    return frozenset(values)


def all_profile_non_functional_layout_specs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for profile in ROOM_PROFILES:
        for object_type, spec in profile.non_functional_layout_specs.items():
            if isinstance(spec, Mapping):
                out[normalize_profile_token(object_type)] = deepcopy(dict(spec))
    return out


def canonical_profile_object_type(object_type: object) -> str | None:
    for profile in ROOM_PROFILES:
        canonical = profile.canonical_object_type(object_type)
        if canonical is not None:
            return canonical
    return None


def profile_id_for_objects(object_types: Sequence[str]) -> str | None:
    for profile in _layout_enabled_profiles():
        if any(profile.has_trait_object(item) for item in object_types):
            return profile.profile_id
    return None


def is_profile_trait_object(object_type: object) -> bool:
    return any(
        profile.has_trait_object(object_type) for profile in _layout_enabled_profiles()
    )


def is_profile_workflow_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.workflow_objects for profile in _layout_enabled_profiles()
    )


def is_profile_wall_backed_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.wall_backed_objects
        for profile in _layout_enabled_profiles()
    )


def is_profile_floating_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.floating_objects for profile in _layout_enabled_profiles()
    )


def is_profile_mounted_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.mounted_objects for profile in _layout_enabled_profiles()
    )


def is_profile_storage_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.storage_objects for profile in _layout_enabled_profiles()
    )


def fallback_profile_size(object_type: object) -> dict[str, Any] | None:
    for profile in _layout_enabled_profiles():
        profile_size = profile.fallback_size_profile(object_type)
        if profile_size is not None:
            return profile_size
    return None


def profile_object_traits(
    *,
    room_type: object,
    object_type: object,
    include_shadow: bool = True,
) -> tuple[str, ...]:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return ()
    return profile.object_traits(object_type, include_shadow=include_shadow)


def profile_shadow_trace_for_active_clusters(
    *,
    room_type: object,
    active_clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return None

    total_count = 0
    recognized_count = 0
    traited_count = 0
    trait_counts: dict[str, int] = {}
    clusters: list[dict[str, Any]] = []

    for cluster in active_clusters:
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        object_types = _object_types_from_active_cluster(cluster)
        object_rows: list[dict[str, Any]] = []
        cluster_traits: set[str] = set()
        cluster_recognized = 0

        for object_type in object_types:
            total_count += 1
            canonical = profile.canonical_object_type(object_type)
            traits = profile.object_traits(object_type, include_shadow=True)
            if canonical is not None:
                recognized_count += 1
                cluster_recognized += 1
            if traits:
                traited_count += 1
            for trait in traits:
                trait_counts[trait] = trait_counts.get(trait, 0) + 1
                cluster_traits.add(trait)

            object_row: dict[str, Any] = {
                "object_type": object_type,
                "canonical_type": canonical,
                "traits": list(traits),
                "recognized": canonical is not None,
            }
            size_profile = profile.fallback_size_profile(object_type)
            if isinstance(size_profile, Mapping):
                rep_dims = size_profile.get("rep_dims_m")
                if isinstance(rep_dims, Mapping):
                    object_row["size_tiers"] = sorted(str(key) for key in rep_dims)
            object_rows.append(object_row)

        clusters.append(
            {
                "cluster_id": cluster_id,
                "object_count": len(object_types),
                "recognized_count": cluster_recognized,
                "dominant_traits": sorted(cluster_traits),
                "objects": object_rows,
            }
        )

    coverage_ratio = (
        round(recognized_count / total_count, 3) if total_count > 0 else 0.0
    )
    trait_coverage_ratio = (
        round(traited_count / total_count, 3) if total_count > 0 else 0.0
    )
    return {
        "profile_id": profile.profile_id,
        "mode": "active" if profile.layout_traits_enabled else "shadow",
        "layout_traits_enabled": profile.layout_traits_enabled,
        "semantic_rule_enabled": profile.semantic_room_rule_provider is not None,
        "object_count": total_count,
        "recognized_count": recognized_count,
        "coverage_ratio": coverage_ratio,
        "trait_coverage_ratio": trait_coverage_ratio,
        "trait_counts": dict(sorted(trait_counts.items())),
        "clusters": clusters,
    }


def _profiles_for_members(members: Sequence[str]) -> list[RoomProfile]:
    return [
        profile
        for profile in _layout_enabled_profiles()
        if any(profile.has_trait_object(member) for member in members)
    ]


def _layout_enabled_profiles() -> tuple[RoomProfile, ...]:
    return tuple(profile for profile in ROOM_PROFILES if profile.layout_traits_enabled)


def _object_types_from_active_cluster(cluster: Mapping[str, Any]) -> list[str]:
    object_types: list[str] = []
    bundles = cluster.get("required_bundles")
    if isinstance(bundles, Sequence) and not isinstance(bundles, str):
        for bundle in bundles:
            if not isinstance(bundle, Mapping):
                continue
            objects = bundle.get("objects")
            if not isinstance(objects, Sequence) or isinstance(objects, str):
                continue
            for obj in objects:
                if not isinstance(obj, Mapping):
                    continue
                object_type = str(obj.get("object_type") or "").strip()
                if object_type:
                    object_types.append(object_type)
    if object_types:
        return object_types

    members = cluster.get("members")
    if isinstance(members, Sequence) and not isinstance(members, str):
        return [str(member).strip() for member in members if str(member).strip()]
    return []


def _request_alias_variants(
    canonical: str,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for value in (canonical, canonical.replace("_", " "), *aliases):
        text = str(value or "").strip().lower()
        if not text:
            continue
        variants = (text, text.replace("_", " "))
        for variant in variants:
            if variant and variant not in values:
                values.append(variant)
    return tuple(values)


def _scoring_alias_variants(
    canonical: str,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for value in (canonical, *aliases):
        token = normalize_profile_token(value)
        if token and token not in values:
            values.append(token)
    return tuple(values)
