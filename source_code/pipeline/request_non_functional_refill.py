from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.request_contract import (
    canonical_object_type,
    missing_non_functional_contract_items,
    request_contract_from_payload,
)
from layout.room_profiles.registry import all_profile_non_functional_layout_specs


_REQUEST_NON_FUNCTIONAL_LAYOUT_SPECS: dict[str, dict[str, Any]] = {
    "rug": {
        "width_ratio": 0.56,
        "height_ratio": 0.44,
        "min_w": 900,
        "min_h": 700,
        "max_w": 2400,
        "max_h": 2800,
        "collision_layer": "floor_underlay",
        "place_on": {"target_instance_id": "floor", "method": "floor"},
    },
    "ceiling_light": {
        "width": 450,
        "height": 450,
        "collision_layer": "ceiling",
        "place_on": {"target_instance_id": "ceiling", "method": "hang_on"},
    },
    "mirror": {
        "width": 800,
        "height": 80,
        "collision_layer": "wall_mounted",
        "mount_region_type": "wall",
        "target_object_types": ["wardrobe", "dresser", "tv_console"],
        "place_on": {"method": "hang_on"},
    },
    "wall_art": {
        "width": 900,
        "height": 70,
        "collision_layer": "wall_mounted",
        "mount_region_type": "wall",
        "place_on": {"target_instance_id": "wall", "method": "hang_on"},
    },
    "curtain": {
        "width_ratio": 0.34,
        "height": 140,
        "min_w": 900,
        "max_w": 2200,
        "collision_layer": "opening_mounted",
        "mount_region_type": "opening",
        "place_on": {"target_instance_id": "opening", "method": "hang_on"},
    },
    "cushion": {
        "width": 420,
        "height": 320,
        "collision_layer": "host_attached",
        "target_object_types": ["bed", "sofa", "chair", "armchair"],
        "place_on": {"method": "on_top"},
    },
    "blanket": {
        "width": 650,
        "height": 500,
        "collision_layer": "host_attached",
        "target_object_types": ["bed", "sofa", "armchair"],
        "place_on": {"method": "on_top"},
    },
}
_REQUEST_NON_FUNCTIONAL_LAYOUT_SPECS.update(all_profile_non_functional_layout_specs())


def ensure_requested_non_functional_contract_objects(
    *,
    absolute_layout: dict[str, Any],
    room_output: dict[str, Any],
    cluster_output: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    objects = [
        row for row in absolute_layout.get("objects") or [] if isinstance(row, dict)
    ]
    missing_items = missing_non_functional_contract_items(
        contract=request_contract_from_payload(cluster_output),
        objects=objects,
    )
    if not missing_items:
        return absolute_layout, {"added_count": 0, "added_object_types": []}

    room_bbox = _room_bbox_for_non_functional_refill(
        room_output=room_output,
        absolute_layout=absolute_layout,
    )
    if room_bbox is None:
        return absolute_layout, {
            "added_count": 0,
            "added_object_types": [],
            "skipped_reason": "missing_room_polygon",
        }

    updated_layout = deepcopy(absolute_layout)
    updated_objects = [
        dict(row)
        for row in updated_layout.get("objects") or []
        if isinstance(row, dict)
    ]
    added_types: list[str] = []
    for item in missing_items:
        object_type = canonical_object_type(str(item.get("object_type") or ""))
        spec = _REQUEST_NON_FUNCTIONAL_LAYOUT_SPECS.get(object_type)
        if not spec:
            continue
        layout_object = _requested_non_functional_layout_object(
            object_type=object_type,
            spec=spec,
            room_bbox=room_bbox,
            existing_objects=updated_objects,
        )
        if layout_object is None:
            continue
        updated_objects.append(layout_object)
        added_types.append(object_type)

    if not added_types:
        return absolute_layout, {"added_count": 0, "added_object_types": []}

    updated_layout["objects"] = updated_objects
    return updated_layout, {
        "added_count": len(added_types),
        "added_object_types": added_types,
        "source": "request_contract_non_functional",
    }


def _room_bbox_for_non_functional_refill(
    *,
    room_output: dict[str, Any],
    absolute_layout: dict[str, Any],
) -> dict[str, int] | None:
    room = room_output.get("room") if isinstance(room_output.get("room"), dict) else {}
    layout_room = (
        absolute_layout.get("room")
        if isinstance(absolute_layout.get("room"), dict)
        else {}
    )
    for candidate in (
        room.get("polygon_ccw"),
        layout_room.get("polygon_ccw"),
        absolute_layout.get("polygon_ccw"),
    ):
        points = _non_functional_refill_points(candidate)
        if len(points) >= 3:
            return _bbox_from_world_points(points)
    return None


def _non_functional_refill_points(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        try:
            points.append((float(row.get("x") or 0.0), float(row.get("y") or 0.0)))
        except (TypeError, ValueError):
            continue
    return points


def _bbox_from_world_points(points: list[tuple[float, float]]) -> dict[str, int]:
    if not points:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "min_x": int(round(min(xs))),
        "min_y": int(round(min(ys))),
        "max_x": int(round(max(xs))),
        "max_y": int(round(max(ys))),
    }


def _requested_non_functional_layout_object(
    *,
    object_type: str,
    spec: dict[str, Any],
    room_bbox: dict[str, int],
    existing_objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    room_width = max(0, int(room_bbox["max_x"]) - int(room_bbox["min_x"]))
    room_height = max(0, int(room_bbox["max_y"]) - int(room_bbox["min_y"]))
    if room_width <= 0 or room_height <= 0:
        return None

    width = _request_refill_dimension(
        spec=spec,
        fixed_key="width",
        ratio_key="width_ratio",
        min_key="min_w",
        max_key="max_w",
        room_span=room_width,
    )
    height = _request_refill_dimension(
        spec=spec,
        fixed_key="height",
        ratio_key="height_ratio",
        min_key="min_h",
        max_key="max_h",
        room_span=room_height,
    )
    if width <= 0 or height <= 0:
        return None

    target_object = _target_existing_layout_object(
        existing_objects,
        spec.get("target_object_types"),
    )
    target_center = _layout_object_center(target_object) if target_object else None
    if target_center is not None:
        center_x, center_y = target_center
    else:
        center_x = int(round((int(room_bbox["min_x"]) + int(room_bbox["max_x"])) / 2.0))
        center_y = int(round((int(room_bbox["min_y"]) + int(room_bbox["max_y"])) / 2.0))
    min_x = _clamp_int(
        center_x - width // 2,
        int(room_bbox["min_x"]),
        int(room_bbox["max_x"]) - width,
    )
    min_y = _clamp_int(
        center_y - height // 2,
        int(room_bbox["min_y"]),
        int(room_bbox["max_y"]) - height,
    )
    max_x = min_x + width
    max_y = min_y + height
    instance_id = _unique_refill_instance_id(object_type, existing_objects)
    place_on = deepcopy(
        spec.get("place_on") if isinstance(spec.get("place_on"), dict) else {}
    )
    if place_on.get("target_instance_id") == "floor":
        target_id = _largest_existing_layout_object_id(existing_objects)
        if target_id:
            place_on["target_instance_id"] = target_id
    elif target_object is not None and not place_on.get("target_instance_id"):
        target_id = target_object.get("instance_id") or target_object.get("object_id")
        if isinstance(target_id, str) and target_id.strip():
            place_on["target_instance_id"] = target_id.strip()

    return {
        "cluster_id": _preferred_existing_cluster_id(existing_objects),
        "object_id": instance_id,
        "instance_id": instance_id,
        "object_type": object_type,
        "category": object_type,
        "source": "request_contract",
        "x": min_x,
        "y": min_y,
        "rot": 0,
        "rotation_ccw": 0,
        "w": width,
        "h": height,
        "rect": [min_x, min_y, width, height],
        "bbox": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
        },
        "center": {
            "x": int(round((min_x + max_x) / 2.0)),
            "y": int(round((min_y + max_y) / 2.0)),
        },
        "polygon_ccw": [
            {"x": min_x, "y": min_y},
            {"x": max_x, "y": min_y},
            {"x": max_x, "y": max_y},
            {"x": min_x, "y": max_y},
        ],
        "place_on": place_on or None,
        "collision_layer": str(spec.get("collision_layer") or "floor_underlay"),
        "priority": "request_non_functional",
        "role": "decor_light",
    }


def _request_refill_dimension(
    *,
    spec: dict[str, Any],
    fixed_key: str,
    ratio_key: str,
    min_key: str,
    max_key: str,
    room_span: int,
) -> int:
    fixed_value = spec.get(fixed_key)
    if isinstance(fixed_value, (int, float)) and fixed_value > 0:
        return min(int(round(float(fixed_value))), max(1, room_span))
    ratio = float(spec.get(ratio_key) or 0.35)
    value = int(round(float(room_span) * ratio))
    lower = min(int(spec.get(min_key) or 1), max(1, room_span))
    upper = min(int(spec.get(max_key) or room_span), max(1, room_span))
    if upper < lower:
        lower = upper
    return _clamp_int(value, lower, upper)


def _target_existing_layout_object(
    objects: list[dict[str, Any]],
    target_object_types: Any,
) -> dict[str, Any] | None:
    if not isinstance(target_object_types, list):
        return None
    target_types = {
        canonical_object_type(str(item or ""))
        for item in target_object_types
        if str(item or "").strip()
    }
    if not target_types:
        return None

    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in objects:
        if not isinstance(row, dict):
            continue
        object_type = canonical_object_type(
            str(row.get("object_type") or row.get("category") or "")
        )
        if object_type not in target_types:
            continue
        area = _layout_object_area(row)
        candidates.append((area, row))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _layout_object_center(row: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(row, dict):
        return None
    center = row.get("center")
    if isinstance(center, dict):
        try:
            return (
                int(round(float(center.get("x") or 0.0))),
                int(round(float(center.get("y") or 0.0))),
            )
        except (TypeError, ValueError):
            pass
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        try:
            return (
                int(
                    round(
                        (
                            float(bbox.get("min_x") or 0.0)
                            + float(bbox.get("max_x") or 0.0)
                        )
                        / 2.0
                    )
                ),
                int(
                    round(
                        (
                            float(bbox.get("min_y") or 0.0)
                            + float(bbox.get("max_y") or 0.0)
                        )
                        / 2.0
                    )
                ),
            )
        except (TypeError, ValueError):
            pass
    try:
        x = float(row.get("x") or 0.0)
        y = float(row.get("y") or 0.0)
        width = float(row.get("w") or 0.0)
        height = float(row.get("h") or 0.0)
    except (TypeError, ValueError):
        return None
    return (int(round(x + width / 2.0)), int(round(y + height / 2.0)))


def _layout_object_area(row: dict[str, Any]) -> float:
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        try:
            width = float(bbox.get("max_x") or 0.0) - float(bbox.get("min_x") or 0.0)
            height = float(bbox.get("max_y") or 0.0) - float(bbox.get("min_y") or 0.0)
            return max(0.0, width * height)
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, float(row.get("w") or 0.0) * float(row.get("h") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _unique_refill_instance_id(
    object_type: str,
    existing_objects: list[dict[str, Any]],
) -> str:
    existing = {
        str(row.get("instance_id") or row.get("object_id") or "")
        for row in existing_objects
        if isinstance(row, dict)
    }
    for index in range(1, 1000):
        candidate = f"{object_type}_{index:03d}"
        if candidate not in existing:
            return candidate
    return f"{object_type}_999"


def _largest_existing_layout_object_id(objects: list[dict[str, Any]]) -> str | None:
    candidates: list[tuple[float, str]] = []
    for row in objects:
        bbox = row.get("bbox")
        if not isinstance(bbox, dict):
            continue
        object_id = row.get("instance_id") or row.get("object_id")
        if not isinstance(object_id, str) or not object_id.strip():
            continue
        try:
            width = float(bbox.get("max_x") or 0.0) - float(bbox.get("min_x") or 0.0)
            height = float(bbox.get("max_y") or 0.0) - float(bbox.get("min_y") or 0.0)
        except (TypeError, ValueError):
            continue
        candidates.append((max(0.0, width * height), object_id.strip()))
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[0])[1]


def _preferred_existing_cluster_id(objects: list[dict[str, Any]]) -> str | None:
    for row in objects:
        cluster_id = row.get("cluster_id")
        if isinstance(cluster_id, str) and cluster_id.strip():
            return cluster_id.strip()
    return None


def _clamp_int(value: int, lower: int, upper: int) -> int:
    if upper < lower:
        return lower
    return max(lower, min(upper, value))
