from __future__ import annotations

from collections.abc import Iterable

GENERIC_VARIANT_FAMILIES = frozenset(
    {
        "base",
        "mirror_x",
        "mirror_y",
        "mirror_xy",
        "front_flip",
        "front_flip_mirror_x",
        "front_flip_mirror_y",
        "composer",
        "variant",
    }
)

FALLBACK_GENERIC_VARIANT_FAMILY = "fallback_generic"

SEMANTIC_VARIANT_FAMILIES = frozenset(
    {
        "open_center",
        "perimeter_facing",
        "conversation_facing",
        "media_facing",
        "wall_backed_focal",
        "focal_media",
        "focal_axis",
        "focal_wall",
        "window_oriented",
        "daylight_work",
        "work_core",
        "workflow",
        "storage_wall",
        "edge_storage",
        "perimeter_storage",
        "support_edge",
    }
)

VARIANT_FAMILY_ALIASES = {
    "compact_storage_bank": "edge_storage",
    "focal_wall": "wall_backed_focal",
    "wall_media": "wall_backed_focal",
    "wall_media_linear": "wall_backed_focal",
    "wall_storage": "storage_wall",
    "wall_storage_linear": "storage_wall",
}

VARIANT_FAMILY_TOKENS = (
    SEMANTIC_VARIANT_FAMILIES
    | GENERIC_VARIANT_FAMILIES
    | frozenset({FALLBACK_GENERIC_VARIANT_FAMILY})
)

ROLE_VARIANT_FAMILY_ALLOWLISTS = {
    "social_anchor": frozenset(
        {
            "open_center",
            "perimeter_facing",
            "conversation_facing",
        }
    ),
    "lounge": frozenset(
        {
            "open_center",
            "perimeter_facing",
            "conversation_facing",
        }
    ),
    "media": frozenset(
        {
            "media_facing",
            "wall_backed_focal",
            "focal_media",
            "focal_axis",
        }
    ),
    "focal": frozenset(
        {
            "media_facing",
            "wall_backed_focal",
            "focal_media",
            "focal_axis",
        }
    ),
    "work": frozenset(
        {
            "daylight_work",
            "work_core",
            "window_oriented",
            "workflow",
        }
    ),
    "workflow": frozenset(
        {
            "daylight_work",
            "work_core",
            "window_oriented",
            "workflow",
        }
    ),
    "storage": frozenset(
        {
            "storage_wall",
            "edge_storage",
            "perimeter_storage",
        }
    ),
}


def normalize_policy_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_variant_family(value: object) -> str:
    token = normalize_policy_token(value)
    return VARIANT_FAMILY_ALIASES.get(token, token)


def canonical_semantic_variant_family(value: object) -> str | None:
    token = normalize_variant_family(value)
    if token in SEMANTIC_VARIANT_FAMILIES:
        return token
    return None


def normalized_variant_family_set(
    values: Iterable[object],
    *,
    allowed_tokens: frozenset[str] = VARIANT_FAMILY_TOKENS,
) -> set[str]:
    families = {normalize_variant_family(value) for value in values}
    return {family for family in families if family in allowed_tokens}
