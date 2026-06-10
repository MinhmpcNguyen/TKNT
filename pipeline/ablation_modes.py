from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, cast

from stylist.style_policy import build_neutral_style_policy

AblationMode = Literal[
    "full",
    "minimal_room_context",
    "no_style_policy",
    "no_semantic_layout_planner",
    "no_request_contract",
    "no_tier_capacity",
    "no_capacity_control",
    "single_concept",
    "no_concept_repair",
    "no_material_macro_repair",
    "no_accessory_refill",
    "no_stylist_llm",
    "no_gallery_selector",
]

ABLATION_ENV = "TKNT_ABLATION_MODE"

_VALID_MODES: set[str] = {
    "full",
    "minimal_room_context",
    "no_style_policy",
    "no_semantic_layout_planner",
    "no_request_contract",
    "no_tier_capacity",
    "no_capacity_control",
    "single_concept",
    "no_concept_repair",
    "no_material_macro_repair",
    "no_accessory_refill",
    "no_stylist_llm",
    "no_gallery_selector",
}

_SYSTEM_ID_BY_MODE: dict[str, str] = {
    "full": "TKNT-Full",
    "minimal_room_context": "TKNT-AB-MinimalRoomContext",
    "no_style_policy": "TKNT-A1-NoStylePolicy",
    "no_semantic_layout_planner": "TKNT-AB-NoSemanticLayoutPlanner",
    "no_request_contract": "TKNT-AB-NoRequestContract",
    "no_tier_capacity": "TKNT-AB-NoTierCapacity",
    "no_capacity_control": "TKNT-A2-NoCapacityControl",
    "single_concept": "TKNT-A3-SingleConcept",
    "no_concept_repair": "TKNT-AB-NoConceptRepair",
    "no_material_macro_repair": "TKNT-AB-NoMaterialMacroRepair",
    "no_accessory_refill": "TKNT-AB-NoAccessoryRefill",
    "no_stylist_llm": "TKNT-AB-NoStylistLLM",
    "no_gallery_selector": "TKNT-AB-NoGallerySelector",
}

_TIER_CAPACITY_DISABLED_MODES: set[str] = {
    "no_tier_capacity",
    "no_capacity_control",
}

_EMPTY_AFFORDANCE_MAP: dict[str, Any] = {
    "usable_walls": [],
    "blocked_walls": [],
    "entry_landing_zones": [],
    "circulation_corridors": [],
    "daylight_regions": [],
    "privacy_regions": [],
    "focal_surfaces": [],
    "center_openness_regions": [],
    "wall_anchor_candidates": [],
    "floating_zone_candidates": [],
    "soft_usage_hints": {},
}


def resolve_ablation_mode(value: str | None = None) -> AblationMode:
    raw = (value if value is not None else os.getenv(ABLATION_ENV, "full")).strip()
    normalized = raw.lower().replace("-", "_")
    if normalized in {"", "none", "baseline"}:
        normalized = "full"
    if normalized not in _VALID_MODES:
        raise ValueError(
            f"Unknown ablation mode {raw!r}. Expected one of: "
            + ", ".join(sorted(_VALID_MODES))
        )
    # Safe because normalized is checked against the complete mode set above.
    return cast(AblationMode, normalized)


def ablation_system_id(mode: AblationMode) -> str:
    return _SYSTEM_ID_BY_MODE[mode]


def ablation_metadata(mode: AblationMode) -> dict[str, str]:
    return {
        "ablation_mode": mode,
        "system_id": ablation_system_id(mode),
        "ablation_file": "pipeline/ablation_modes.py",
    }


def uses_neutral_style_policy(mode: AblationMode) -> bool:
    return mode == "no_style_policy"


def use_semantic_layout_planner(mode: AblationMode) -> bool:
    return mode != "no_semantic_layout_planner"


def use_request_contract(mode: AblationMode) -> bool:
    return mode != "no_request_contract"


def use_concept_repair(mode: AblationMode) -> bool:
    return mode != "no_concept_repair"


def use_material_macro_repair(mode: AblationMode) -> bool:
    return mode != "no_material_macro_repair"


def use_stylist_llm(mode: AblationMode) -> bool:
    return mode != "no_stylist_llm"


def use_gallery_selector(mode: AblationMode) -> bool:
    return mode != "no_gallery_selector"


def room_output_for_mode(
    *,
    mode: AblationMode,
    room_output: Mapping[str, Any],
) -> dict[str, Any]:
    payload = deepcopy(dict(room_output))
    if mode != "minimal_room_context":
        return payload

    payload["affordance_map"] = deepcopy(_EMPTY_AFFORDANCE_MAP)
    topology = payload.get("topology")
    wall_graph = topology.get("wall_graph") if isinstance(topology, Mapping) else {}
    payload["topology"] = {"wall_graph": deepcopy(wall_graph or {})}
    payload["notes"] = []
    quality_meta = payload.get("quality_meta")
    if isinstance(quality_meta, Mapping):
        updated_quality_meta = deepcopy(dict(quality_meta))
        updated_quality_meta["notes"] = []
        payload["quality_meta"] = updated_quality_meta
    payload["ablation_minimal_room_context"] = True
    return payload


def style_policy_for_mode(
    *,
    mode: AblationMode,
    room_type: str,
    current_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if uses_neutral_style_policy(mode):
        return build_neutral_style_policy(
            room_type=room_type,
            reason="Ablation A1 disables style-derived layout policy.",
        )
    return deepcopy(dict(current_policy or {}))


def cluster_output_for_mode(
    *,
    mode: AblationMode,
    cluster_output: Mapping[str, Any],
) -> dict[str, Any]:
    payload = deepcopy(dict(cluster_output))
    if use_request_contract(mode):
        return payload
    stripped = _strip_request_contract_fields(payload)
    stripped["request_contract_disabled"] = True
    return stripped


def target_final_count_for_mode(mode: AblationMode, default_count: int) -> int:
    if mode == "single_concept":
        return 1
    return max(1, int(default_count))


def apply_tier_output_for_mode(
    *,
    mode: AblationMode,
    tier_output: Mapping[str, Any],
) -> dict[str, Any]:
    payload = deepcopy(dict(tier_output))
    if mode not in _TIER_CAPACITY_DISABLED_MODES:
        return payload

    changed = False
    decisions = payload.get("decisions")
    if isinstance(decisions, list):
        for row in decisions:
            if not isinstance(row, dict):
                continue
            changed = _force_keep_decision(row) or changed

    cluster_decisions = payload.get("cluster_decisions")
    if isinstance(cluster_decisions, list):
        for cluster in cluster_decisions:
            if not isinstance(cluster, dict):
                continue
            bundles = cluster.get("selected_bundles")
            if not isinstance(bundles, list):
                continue
            for bundle in bundles:
                if not isinstance(bundle, dict):
                    continue
                objects = bundle.get("objects")
                if not isinstance(objects, list):
                    continue
                for row in objects:
                    if isinstance(row, dict):
                        changed = _force_keep_decision(row) or changed

    notes = _string_list(payload.get("notes"))
    label = "A2 legacy capacity-control" if mode == "no_capacity_control" else "tier"
    note = (
        f"Ablation disabled {label} capacity pruning/backoff; zero-quantity tier "
        "decisions were forced active before layout solving."
    )
    if changed and note not in notes:
        notes.append(note)
    payload["notes"] = notes
    payload["capacity_control_disabled"] = True
    payload["budget_valid"] = False
    return payload


def skip_accessory_refill(mode: AblationMode) -> bool:
    return mode in {"no_accessory_refill", "no_capacity_control"}


def select_candidates_for_mode(
    *,
    mode: AblationMode,
    candidates: list[dict[str, Any]],
    max_variants: int,
) -> list[dict[str, Any]]:
    if use_gallery_selector(mode):
        return candidates
    return candidates[: max(1, int(max_variants))]


def _strip_request_contract_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = str(key)
            if clean_key == "request_contract" or clean_key.startswith(
                "request_contract_"
            ):
                continue
            out[clean_key] = _strip_request_contract_fields(item)
        return out
    if isinstance(value, list):
        return [_strip_request_contract_fields(item) for item in value]
    return deepcopy(value)


def _force_keep_decision(row: dict[str, Any]) -> bool:
    object_type = str(row.get("object_type") or row.get("category") or "").strip()
    if not object_type:
        return False
    quantity = _coerce_int(row.get("quantity"))
    if quantity is not None and quantity > 0:
        return False
    row["quantity"] = 1
    if not row.get("size_tier"):
        row["size_tier"] = "S"
    row["ablation_forced_active"] = True
    return True


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
