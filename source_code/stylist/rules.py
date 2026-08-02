from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]
JsonValue: TypeAlias = (
    dict[str, object] | list[object] | str | int | float | bool | None
)

_ROOM_SELECTION_RULE_PATH = Path(__file__).with_name("rule.json")
_COMPILED_SEMANTIC_RULE_PATH = Path(__file__).with_name(
    "compiled_semantic_program.json"
)


def get_soft_rule_templates() -> dict[str, dict[str, Any]]:
    return {
        "color_composition": {
            "context_fields_to_find": [
                "palette",
                "accent_ratio",
                "dominant_materials",
            ]
        },
        "style_specific_aesthetics": {
            "context_fields_to_find": [
                "materials",
                "forms",
                "textures",
            ]
        },
        "room_specific_feng_shui": {
            "context_fields_to_find": [
                "circulation",
                "entry_path",
                "primary_anchor",
            ]
        },
        "personal_feng_shui": {
            "context_fields_to_find": [
                "element_colors",
                "balance",
                "avoidances",
            ]
        },
        "lighting_color_interaction": {
            "context_fields_to_find": [
                "cct",
                "daylight_balance",
                "accent_lighting",
            ]
        },
    }


@lru_cache(maxsize=1)
def load_room_selection_rules(
    rule_path: Path = _ROOM_SELECTION_RULE_PATH,
) -> dict[str, JsonObject]:
    payload = json.loads(rule_path.read_text(encoding="utf-8"))
    rules: dict[str, JsonObject] = {}
    for item in _iter_room_selection_rule_objects(payload):
        room_type = item.get("room_type")
        if not isinstance(room_type, str):
            continue
        normalized_room_type = room_type.strip()
        if not normalized_room_type:
            continue
        rules[normalized_room_type] = _clone_json_object(item)
    return rules


def get_room_selection_rule(room_type: str) -> JsonObject | None:
    rule = load_room_selection_rules().get(room_type)
    if rule is None:
        return None
    return _clone_json_object(rule)


@lru_cache(maxsize=1)
def load_compiled_semantic_program(
    rule_path: Path = _COMPILED_SEMANTIC_RULE_PATH,
) -> JsonObject:
    payload = json.loads(rule_path.read_text(encoding="utf-8"))
    return _clone_json_object(payload if isinstance(payload, Mapping) else {})


def get_compiled_semantic_room_rule(room_type: str) -> JsonObject | None:
    normalized_room_type = room_type.strip()
    if not normalized_room_type:
        return None

    payload = load_compiled_semantic_program()
    for item in _iter_compiled_room_rules(payload.get("rooms")):
        if item.get("room_type") == normalized_room_type:
            return _clone_json_object(item)
    return None


def _iter_room_selection_rule_objects(value: object) -> Iterable[Mapping[str, object]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("room_type"), str):
            yield value
        return

    if isinstance(value, list):
        for child in value:
            yield from _iter_room_selection_rule_objects(child)


def _iter_compiled_room_rules(value: object) -> Iterable[Mapping[str, object]]:
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get("room_type"), str):
            yield item


def _clone_json_object(value: Mapping[str, object]) -> JsonObject:
    return {
        key: _clone_json_value(child)
        for key, child in value.items()
        if isinstance(key, str)
    }


def _clone_json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return _clone_json_object(value)
    if isinstance(value, list):
        return [_clone_json_value(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
