from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

ORIENTATION_KEYS = (
    "rotation_ccw",
    "front_world",
    "front_side_world",
    "axis_world",
)


def enrich_styled_output_orientations(
    *,
    styled_payload: Mapping[str, object],
    absolute_layout_payload: Mapping[str, object],
) -> dict[str, object]:
    by_id, by_bbox = _orientation_indexes(absolute_layout_payload)
    return _enrich_payload_objects(
        payload=styled_payload,
        by_id=by_id,
        by_bbox=by_bbox,
        normalized=False,
    )


def enrich_normalized_layout_orientations(
    *,
    normalized_payload: Mapping[str, object],
    absolute_layout_payload: Mapping[str, object],
) -> dict[str, object]:
    by_id, by_bbox = _orientation_indexes(absolute_layout_payload)
    return _enrich_payload_objects(
        payload=normalized_payload,
        by_id=by_id,
        by_bbox=by_bbox,
        normalized=True,
    )


def _enrich_payload_objects(
    *,
    payload: Mapping[str, object],
    by_id: Mapping[str, Mapping[str, object]],
    by_bbox: Mapping[tuple[float, float, float, float], Mapping[str, object]],
    normalized: bool,
) -> dict[str, object]:
    out = deepcopy(dict(payload))
    objects = out.get("objects")
    if not isinstance(objects, list):
        return out

    enriched_objects: list[object] = []
    for value in objects:
        item = _mapping(value)
        if not item:
            enriched_objects.append(value)
            continue
        enriched_objects.append(
            _enrich_object(
                item=item,
                orientation=_find_orientation(item=item, by_id=by_id, by_bbox=by_bbox),
                normalized=normalized,
            )
        )
    out["objects"] = enriched_objects
    return out


def _enrich_object(
    *,
    item: Mapping[str, object],
    orientation: Mapping[str, object] | None,
    normalized: bool,
) -> dict[str, object]:
    out = deepcopy(dict(item))
    if orientation is None:
        return out

    for key, value in orientation.items():
        out[key] = deepcopy(value)
    if not normalized:
        return out

    rotation_ccw = _number(orientation.get("rotation_ccw"))
    if rotation_ccw is not None:
        out["rotation_degrees"] = rotation_ccw
    raw = deepcopy(dict(_mapping(out.get("raw"))))
    for key, value in orientation.items():
        raw[key] = deepcopy(value)
    out["raw"] = raw
    return out


def _orientation_indexes(
    absolute_layout_payload: Mapping[str, object],
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[tuple[float, float, float, float], Mapping[str, object]],
]:
    by_id: dict[str, Mapping[str, object]] = {}
    by_bbox: dict[tuple[float, float, float, float], Mapping[str, object]] = {}
    for value in _sequence(absolute_layout_payload.get("objects")):
        item = _mapping(value)
        orientation = _orientation(item)
        if not orientation:
            continue
        object_id = _first_string(item, "object_id", "instance_id", "id")
        if object_id:
            by_id[object_id] = orientation
        bbox_key = _bbox_key(item.get("bbox"))
        if bbox_key is not None:
            by_bbox[bbox_key] = orientation
    return by_id, by_bbox


def _find_orientation(
    *,
    item: Mapping[str, object],
    by_id: Mapping[str, Mapping[str, object]],
    by_bbox: Mapping[tuple[float, float, float, float], Mapping[str, object]],
) -> Mapping[str, object] | None:
    object_id = _first_string(item, "instance_id", "id", "object_id")
    if object_id and object_id in by_id:
        return by_id[object_id]
    bbox_key = _bbox_key(item.get("bbox"))
    return by_bbox.get(bbox_key) if bbox_key is not None else None


def _orientation(item: Mapping[str, object]) -> dict[str, object]:
    return {
        key: deepcopy(item[key])
        for key in ORIENTATION_KEYS
        if key in item and item[key] is not None
    }


def _bbox_key(value: object) -> tuple[float, float, float, float] | None:
    item = _mapping(value)
    min_x = _number(item.get("min_x"))
    min_y = _number(item.get("min_y"))
    max_x = _number(item.get("max_x"))
    max_y = _number(item.get("max_y"))
    if min_x is None or min_y is None or max_x is None or max_y is None:
        return None
    return (min_x, min_y, max_x, max_y)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _first_string(item: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
