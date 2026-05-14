from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from typing import Any

from layout.kitchen_profile import (
    KITCHEN_FLOATING_OBJECTS,
    KITCHEN_MOUNTED_OBJECTS,
    KITCHEN_OBJECT_ALIASES,
    KITCHEN_ROOM_TYPES,
    KITCHEN_STORAGE_OBJECTS,
    KITCHEN_WALL_BACKED_OBJECTS,
    KITCHEN_WORKFLOW_OBJECTS,
    is_kitchen_floating_object,
    is_kitchen_object_like,
    is_kitchen_storage_object,
    is_kitchen_wall_backed_object,
    is_kitchen_workflow_object,
    kitchen_fallback_size_profile,
    kitchen_semantic_room_rule,
)
from layout.room_profiles.base import RoomProfile

KITCHEN_NON_FUNCTIONAL_CONTRACT_TYPES = frozenset(
    {
        "cooktop",
        "kitchen_wall_cabinet",
        "pendant_light",
        "range_hood",
    }
)

KITCHEN_SCORING_ALIASES: dict[str, tuple[str, ...]] = {
    "bar_stool": ("bar_stool", "counter_stool", "island_stool"),
    "cooktop": ("cooktop", "hob", "induction_hob", "gas_hob"),
    "dining_chair": ("dining_chair", "breakfast_chair"),
    "dining_table": ("dining_table", "breakfast_table"),
    "dishwasher": ("dishwasher", "dish_washer"),
    "fridge": ("fridge", "refrigerator", "fridge_freezer", "freezer"),
    "kitchen_base_cabinet": (
        "kitchen_base_cabinet",
        "base_cabinet",
        "counter",
        "countertop",
        "prep_counter",
        "worktop",
    ),
    "kitchen_island": ("kitchen_island", "island", "prep_island"),
    "kitchen_tall_cabinet": ("kitchen_tall_cabinet", "tall_cabinet"),
    "kitchen_wall_cabinet": ("kitchen_wall_cabinet", "wall_cabinet", "upper_cabinet"),
    "pantry_cabinet": ("pantry_cabinet", "pantry"),
    "pendant_light": ("pendant_light", "island_pendant"),
    "range_hood": ("range_hood", "hood", "extractor_hood", "vent_hood"),
    "sink": ("sink", "kitchen_sink"),
    "stove": ("stove", "range", "oven_range", "cooker"),
}

KITCHEN_NON_FUNCTIONAL_LAYOUT_SPECS: dict[str, dict[str, Any]] = {
    "pendant_light": {
        "width": 320,
        "height": 320,
        "collision_layer": "ceiling",
        "target_object_types": ["kitchen_island", "dining_table"],
        "place_on": {"target_instance_id": "ceiling", "method": "hang_on"},
    },
    "range_hood": {
        "width": 700,
        "height": 360,
        "collision_layer": "wall_mounted",
        "target_object_types": ["stove", "cooktop"],
        "place_on": {"method": "hang_on"},
    },
    "kitchen_wall_cabinet": {
        "width": 1000,
        "height": 360,
        "collision_layer": "wall_mounted",
        "target_object_types": ["kitchen_base_cabinet", "sink", "stove"],
        "place_on": {"method": "wall"},
    },
    "cooktop": {
        "width": 650,
        "height": 550,
        "collision_layer": "countertop",
        "target_object_types": ["kitchen_base_cabinet"],
        "place_on": {"method": "on_top"},
    },
}


def apply_kitchen_capacity_model(
    capacity_model: dict[str, Any],
    room_type: str,
) -> dict[str, Any]:
    _ = room_type
    out = dict(capacity_model)
    room_area = max(0.1, float(out.get("available_area_m2") or 0.1))
    density_ratio = max(float(out.get("density_ratio") or 0.38), 0.40)
    if room_area >= 12.0:
        density_ratio = max(density_ratio, 0.43)
    out["target_density"] = "functional_kitchen"
    out["density_ratio"] = min(0.50, density_ratio)
    out["clutter_budget_m2"] = max(
        float(out.get("clutter_budget_m2") or 0.0),
        room_area * float(out["density_ratio"]),
    )
    out["wall_capacity_m2"] = float(out.get("wall_capacity_m2") or 0.0) * 1.30
    out["floating_capacity_m2"] = float(out.get("floating_capacity_m2") or 0.0) * 0.82
    out["center_openness_budget_m2"] = max(
        float(out.get("center_openness_budget_m2") or 0.0),
        room_area * 0.30,
    )
    out["circulation_budget_m2"] = max(
        float(out.get("circulation_budget_m2") or 0.0),
        room_area * 0.22,
    )
    out["center_openness_weight"] = "very_high"
    out["circulation_penalty_weight"] = "very_high"
    out["room_type_capacity_profile"] = "kitchen"
    return out


def kitchen_semantic_placements_for_members(
    cluster_id: str,
    members: Sequence[str],
    anchors: Sequence[str],
) -> list[dict[str, Any]]:
    if "kitchen" not in cluster_id.lower() and not any(
        is_kitchen_object_like(member) for member in members
    ):
        return []
    anchor = next((item for item in anchors if item in members), None)
    if anchor is None:
        anchor = next(
            (item for item in members if is_kitchen_workflow_object(item)), None
        )
    if anchor is None:
        return []

    rows: list[dict[str, Any]] = []
    previous_workflow = anchor
    for member in members:
        if member == anchor or not is_kitchen_workflow_object(member):
            continue
        relative_to = anchor
        if member in {"stove", "dishwasher"} and "sink" in members:
            relative_to = "sink"
        elif member == "sink":
            relative_to = anchor
        elif previous_workflow != anchor:
            relative_to = previous_workflow
        rows.append(
            {
                "id": member,
                "relative_to": relative_to,
                "kind": "anchor_side",
                "side_options": ["left", "right"],
                "gap_min": 0,
                "gap_max": 40,
                "support_role": "wall_support",
                "band_intent": "wall_band",
                "orientation": "same_direction",
                "proximity": "near",
                "selection": "best_fit",
            }
        )
        if member in {"fridge", "sink", "stove"}:
            previous_workflow = member
    return rows


def kitchen_layout_specs() -> dict[str, dict[str, Any]]:
    return deepcopy(KITCHEN_NON_FUNCTIONAL_LAYOUT_SPECS)


KITCHEN_PROFILE = RoomProfile(
    profile_id="kitchen",
    room_types=KITCHEN_ROOM_TYPES,
    object_aliases=KITCHEN_OBJECT_ALIASES,
    scoring_aliases=KITCHEN_SCORING_ALIASES,
    non_functional_contract_types=KITCHEN_NON_FUNCTIONAL_CONTRACT_TYPES,
    non_functional_layout_specs=KITCHEN_NON_FUNCTIONAL_LAYOUT_SPECS,
    workflow_objects=KITCHEN_WORKFLOW_OBJECTS,
    wall_backed_objects=KITCHEN_WALL_BACKED_OBJECTS,
    floating_objects=KITCHEN_FLOATING_OBJECTS,
    mounted_objects=KITCHEN_MOUNTED_OBJECTS,
    storage_objects=KITCHEN_STORAGE_OBJECTS,
    semantic_room_rule_provider=kitchen_semantic_room_rule,
    size_profile_provider=kitchen_fallback_size_profile,
    capacity_policy=apply_kitchen_capacity_model,
    semantic_placement_provider=kitchen_semantic_placements_for_members,
)

__all__ = (
    "KITCHEN_PROFILE",
    "is_kitchen_floating_object",
    "is_kitchen_object_like",
    "is_kitchen_storage_object",
    "is_kitchen_wall_backed_object",
    "is_kitchen_workflow_object",
    "kitchen_layout_specs",
)
