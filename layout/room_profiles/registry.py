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
    for profile in ROOM_PROFILES:
        if any(profile.has_trait_object(item) for item in object_types):
            return profile.profile_id
    return None


def is_profile_trait_object(object_type: object) -> bool:
    return any(profile.has_trait_object(object_type) for profile in ROOM_PROFILES)


def is_profile_workflow_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(canonical in profile.workflow_objects for profile in ROOM_PROFILES)


def is_profile_wall_backed_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(canonical in profile.wall_backed_objects for profile in ROOM_PROFILES)


def is_profile_floating_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(canonical in profile.floating_objects for profile in ROOM_PROFILES)


def is_profile_mounted_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(canonical in profile.mounted_objects for profile in ROOM_PROFILES)


def is_profile_storage_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(canonical in profile.storage_objects for profile in ROOM_PROFILES)


def fallback_profile_size(object_type: object) -> dict[str, Any] | None:
    for profile in ROOM_PROFILES:
        profile_size = profile.fallback_size_profile(object_type)
        if profile_size is not None:
            return profile_size
    return None


def _profiles_for_members(members: Sequence[str]) -> list[RoomProfile]:
    return [
        profile
        for profile in ROOM_PROFILES
        if any(profile.has_trait_object(member) for member in members)
    ]


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
