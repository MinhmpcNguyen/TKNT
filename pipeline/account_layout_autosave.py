from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from db.models import JsonValue

JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class SavedLayoutOptionDraft:
    option_id: str
    option_index: int
    name: str
    floorplan_json: JsonObject
    design_json: JsonObject | None
    styled_result_json: JsonObject | None
    meta: JsonObject


def build_saved_layout_option_drafts(
    *,
    case_id: str,
    case_name: str,
    room_payload: object,
    layout_variants_payload: object,
) -> list[SavedLayoutOptionDraft]:
    variants = _variant_records(layout_variants_payload)
    drafts: list[SavedLayoutOptionDraft] = []
    for index, variant in enumerate(variants, start=1):
        draft = _build_variant_draft(
            case_id=case_id,
            case_name=case_name,
            option_index=index,
            variant=variant,
            room_payload=room_payload,
        )
        if draft is not None:
            drafts.append(draft)
    return drafts


def _build_variant_draft(
    *,
    case_id: str,
    case_name: str,
    option_index: int,
    variant: dict[object, object],
    room_payload: object,
) -> SavedLayoutOptionDraft | None:
    option_id = _clean_string(variant.get("variant_id")) or f"option_{option_index}"
    option_name = _clean_string(variant.get("label")) or f"Option {option_index}"
    absolute_layout = _json_object(variant.get("absolute_layout")) or {}
    styled = _json_object(variant.get("styled_result")) or {}
    styled = _enrich_rotation_ccw(styled, absolute_layout)
    floorplan = _build_floorplan(room_payload, styled, absolute_layout)
    if floorplan is None:
        return None
    styled_floorplan = _apply_styled_room_appearance(floorplan, styled)
    meta: JsonObject = {
        "source": "pipeline_autosave",
        "source_pipeline_case_id": case_id,
        "case_name": case_name,
        "option_id": option_id,
        "option_index": option_index,
        "variant_id": option_id,
        "variant_source": _json_value(variant.get("source")),
        "variant_reason": _json_value(variant.get("reason")),
        "layout_score": _json_value(variant.get("layout_score")),
        "hard_valid": _json_value(variant.get("hard_valid")),
        "complete": _json_value(variant.get("complete")),
        "gallery_eligible": _json_value(variant.get("gallery_eligible")),
        "coverage_ratio": _json_value(variant.get("coverage_ratio")),
        "missing_cluster_ids": _json_value(variant.get("missing_cluster_ids")),
        "notes": _json_value(variant.get("notes")),
        "absolute_layout": absolute_layout,
    }
    return SavedLayoutOptionDraft(
        option_id=option_id,
        option_index=option_index,
        name=option_name,
        floorplan_json=styled_floorplan,
        design_json=_build_design_json(styled, styled_floorplan),
        styled_result_json=styled,
        meta=meta,
    )


def _variant_records(layout_variants_payload: object) -> list[dict[object, object]]:
    payload = _record(layout_variants_payload)
    variants = payload.get("variants")
    if not isinstance(variants, list):
        return []
    return [item for item in variants if isinstance(item, dict)]


def _build_floorplan(
    room_payload: object,
    styled_payload: JsonObject,
    absolute_layout: JsonObject,
) -> JsonObject | None:
    room_source = _record(_record(room_payload).get("room"))
    if not room_source:
        room_source = _record(absolute_layout.get("room"))
    if not room_source:
        room_source = _record(_record(styled_payload.get("room")).get("room"))
    if not room_source:
        room_source = _record(styled_payload.get("room"))

    raw_points = room_source.get("polygon_ccw")
    if not isinstance(raw_points, list):
        raw_points = room_source.get("polygon_mm")
    points = [_point(item) for item in raw_points if isinstance(raw_points, list)]
    points = [point for point in points if point is not None]
    if len(points) < 3:
        return None

    room_id = _clean_string(room_source.get("room_id")) or "room_1"
    wall_thickness = 100
    walls: list[JsonObject] = []
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        walls.append(
            {
                "wall_id": f"wall_{index + 1}",
                "p1_mm": point,
                "p2_mm": next_point,
                "thickness_mm": wall_thickness,
                "room_id": room_id,
            }
        )

    openings = _resolve_openings(room_payload, styled_payload, absolute_layout)
    return {
        "room": {
            "room_id": room_id,
            "name": _clean_string(room_source.get("name")) or "Room",
            "polygon_mm": points,
        },
        "walls": walls,
        "openings": openings,
    }


def _resolve_openings(
    room_payload: object,
    styled_payload: JsonObject,
    absolute_layout: JsonObject,
) -> JsonObject:
    room_record = _record(room_payload)
    openings = _record(room_record.get("openings"))
    if not openings:
        openings = _record(_record(absolute_layout.get("room")).get("openings"))
    if not openings:
        openings = _record(absolute_layout.get("openings"))
    if not openings:
        openings = _record(_record(styled_payload.get("room")).get("openings"))

    return {
        "doors": _opening_rows(openings.get("doors"), id_key="door_id"),
        "windows": _opening_rows(openings.get("windows"), id_key="window_id"),
    }


def _opening_rows(value: object, *, id_key: str) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    rows: list[JsonObject] = []
    for index, item in enumerate(value, start=1):
        record = _record(item)
        segment = record.get("segment_mm")
        if not isinstance(segment, list) or len(segment) < 2:
            continue
        start = _point(segment[0])
        end = _point(segment[1])
        if start is None or end is None:
            continue
        output_id = _clean_string(record.get(id_key)) or _clean_string(record.get("id"))
        if not output_id:
            output_id = f"{id_key.replace('_id', '')}_{index}"
        row: JsonObject = {
            id_key: output_id,
            "wall_id": _json_value(record.get("wall_id")),
            "start_mm": start,
            "end_mm": end,
        }
        rows.append(row)
    return rows


def _apply_styled_room_appearance(
    floorplan: JsonObject,
    styled_payload: JsonObject,
) -> JsonObject:
    output = deepcopy(floorplan)
    room = _record(output.get("room"))
    walls = output.get("walls")
    openings = _record(output.get("openings"))
    styled_room = _record(styled_payload.get("room"))
    surfaces = _record(styled_room.get("surfaces"))
    opening_colors = _record(styled_room.get("opening_colors"))
    floor_color = _clean_string(surfaces.get("floor_color_hex"))
    wall_color = _clean_string(surfaces.get("wall_color_hex"))
    if floor_color:
        room["floor_color_hex"] = floor_color
        output["room"] = _json_object(room) or room
    if wall_color and isinstance(walls, list):
        next_walls: list[JsonValue] = []
        for item in walls:
            wall = _json_object(item) if isinstance(item, dict) else None
            next_walls.append({**wall, "color_hex": wall_color} if wall else item)
        output["walls"] = next_walls
    doors = _apply_opening_colors(openings.get("doors"), opening_colors.get("doors"))
    windows = _apply_opening_colors(
        openings.get("windows"), opening_colors.get("windows")
    )
    output["openings"] = {
        "doors": doors,
        "windows": windows,
    }
    return _json_object(output) or output


def _apply_opening_colors(value: object, color_rows: object) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    openings = [item for item in value if isinstance(item, dict)]
    colors: dict[str, str] = {}
    if isinstance(color_rows, list):
        for color_row in color_rows:
            record = _record(color_row)
            row_id = _clean_string(record.get("id"))
            color = _clean_string(record.get("color_hex"))
            if row_id and color:
                colors[row_id] = color
    output: list[JsonObject] = []
    for item in openings:
        record = _json_object(item) if isinstance(item, dict) else None
        if record is None:
            continue
        row_id = _clean_string(record.get("door_id")) or _clean_string(
            record.get("window_id")
        )
        if row_id in colors:
            record["color_hex"] = colors[row_id]
        output.append(record)
    return output


def _build_design_json(styled_payload: JsonObject, floorplan: JsonObject) -> JsonObject:
    objects = styled_payload.get("objects")
    placements: list[JsonObject] = []
    if isinstance(objects, list):
        for index, item in enumerate(objects):
            record = _record(item)
            if record:
                placements.append(_build_placement(record, index))
    return {
        "floorplan_geometry": floorplan,
        "placements": placements,
    }


def _build_placement(record: dict[object, object], index: int) -> JsonObject:
    category = _clean_string(record.get("object_type")) or "item"
    instance_id = (
        _clean_string(record.get("instance_id"))
        or _clean_string(record.get("id"))
        or _clean_string(record.get("object_id"))
        or f"obj_{index}"
    )
    rotation = _rotation(record)
    bbox = _bbox(record.get("bbox"))
    footprint = _local_footprint(bbox, rotation)
    manual_input = _record(record.get("manual_input"))
    front_world = _front_world(record.get("front_world"))
    front_side_world = _clean_cardinal(record.get("front_side_world")) or _vec_to_side(
        front_world
    )
    if manual_input:
        placement_id = (
            _clean_string(manual_input.get("placement_id"))
            or instance_id
            or f"manual_{index}"
        )
        position = _point(manual_input.get("position_mm")) or {"x": 0, "y": 0}
        manual_footprint = _record(manual_input.get("footprint_mm"))
        footprint = {
            "w": _number(manual_footprint.get("w")),
            "d": _number(manual_footprint.get("d")),
        }
        rotation = _number(manual_input.get("rotation_deg"))
        source = (
            "manual"
            if _clean_string(manual_input.get("source")) == "manual"
            else "generated"
        )
        constraint_mode = _clean_string(manual_input.get("constraint_mode")) or "free"
        asset_id = _normalize_asset_id(manual_input.get("asset_id")) or instance_id
        height = _number(manual_input.get("height_mm"))
        manual_category = _clean_string(manual_input.get("category")) or category
        return {
            "placement_id": placement_id,
            "asset_id": asset_id,
            "category": manual_category,
            "position_mm": position,
            "rotation_deg": rotation,
            "front_world": front_world,
            "front_side_world": front_side_world,
            "front_side_local": None,
            "footprint_mm": footprint,
            "height_mm": height,
            "color_hex": _clean_string(record.get("color_hex")) or "#cfcac2",
            "anchor": "center",
            "source": source,
            "constraint_mode": constraint_mode
            if constraint_mode in {"free", "pinned", "locked"}
            else "free",
            "place_on": _place_on(record.get("place_on")),
            "collision_layer": _clean_string(record.get("collision_layer")),
        }

    return {
        "placement_id": instance_id,
        "asset_id": _normalize_asset_id(record.get("object_id")) or instance_id,
        "category": category,
        "position_mm": _bbox_center(bbox),
        "rotation_deg": rotation,
        "front_world": front_world,
        "front_side_world": front_side_world,
        "front_side_local": None,
        "footprint_mm": footprint,
        "height_mm": 0,
        "color_hex": _clean_string(record.get("color_hex")) or "#cfcac2",
        "anchor": "center",
        "source": "generated",
        "constraint_mode": "free",
        "place_on": _place_on(record.get("place_on")),
        "collision_layer": _clean_string(record.get("collision_layer")),
    }


def _enrich_rotation_ccw(
    styled_payload: JsonObject,
    absolute_layout_payload: JsonObject,
) -> JsonObject:
    absolute_objects = absolute_layout_payload.get("objects")
    if not isinstance(absolute_objects, list):
        return styled_payload
    by_id: dict[str, JsonObject] = {}
    by_bbox: dict[str, JsonObject] = {}
    for item in absolute_objects:
        record = _record(item)
        if not record:
            continue
        orientation: JsonObject = {
            "rotation_ccw": _rotation(record),
            "front_world": _front_world(record.get("front_world")),
            "front_side_world": _clean_cardinal(record.get("front_side_world")),
            "axis_world": _json_value(record.get("axis_world")),
        }
        object_id = _clean_string(record.get("object_id"))
        if object_id:
            by_id[object_id] = orientation
        key = _bbox_key(record.get("bbox"))
        if key:
            by_bbox[key] = orientation

    objects = styled_payload.get("objects")
    if not isinstance(objects, list):
        return styled_payload
    enriched_objects: list[JsonValue] = []
    for item in objects:
        record = _json_object(item) if isinstance(item, dict) else None
        if record is None:
            enriched_objects.append(_json_value(item))
            continue
        instance_id = _clean_string(record.get("instance_id")) or _clean_string(
            record.get("id")
        )
        orientation = (
            by_id.get(instance_id or "") if instance_id else None
        ) or by_bbox.get(_bbox_key(record.get("bbox")) or "")
        if orientation is not None:
            record["rotation_ccw"] = orientation["rotation_ccw"]
            record["front_world"] = orientation["front_world"]
            record["front_side_world"] = orientation["front_side_world"]
            record["axis_world"] = orientation["axis_world"]
        enriched_objects.append(record)
    return {
        **styled_payload,
        "objects": enriched_objects,
    }


def _json_object(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    return {str(key): _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _record(value: object) -> dict[object, object]:
    return value if isinstance(value, dict) else {}


def _point(value: object) -> JsonObject | None:
    record = _record(value)
    if not record:
        return None
    return {
        "x": _number(record.get("x")),
        "y": _number(record.get("y")),
    }


def _bbox(value: object) -> dict[str, float]:
    record = _record(value)
    return {
        "min_x": _number(record.get("min_x")),
        "min_y": _number(record.get("min_y")),
        "max_x": _number(record.get("max_x")),
        "max_y": _number(record.get("max_y")),
    }


def _bbox_center(bbox: dict[str, float]) -> JsonObject:
    return {
        "x": (bbox["min_x"] + bbox["max_x"]) / 2,
        "y": (bbox["min_y"] + bbox["max_y"]) / 2,
    }


def _local_footprint(bbox: dict[str, float], rotation: float) -> JsonObject:
    width = max(0, bbox["max_x"] - bbox["min_x"])
    depth = max(0, bbox["max_y"] - bbox["min_y"])
    if abs(rotation - 90) < 0.5 or abs(rotation - 270) < 0.5:
        return {"w": depth, "d": width}
    return {"w": width, "d": depth}


def _rotation(record: dict[object, object]) -> float:
    return _normalize_rotation(
        record.get("rotation_ccw", record.get("rot", record.get("rotation_deg")))
    )


def _normalize_rotation(value: object) -> float:
    number = _number(value)
    return number % 360


def _number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return number if math.isfinite(number) else 0


def _front_world(value: object) -> JsonObject | None:
    record = _record(value)
    if not record:
        return None
    return {
        "dx": _number(record.get("dx")),
        "dy": _number(record.get("dy")),
    }


def _vec_to_side(value: JsonObject | None) -> str | None:
    if value is None:
        return None
    dx = _number(value.get("dx"))
    dy = _number(value.get("dy"))
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "bottom" if dy >= 0 else "top"


def _clean_cardinal(value: object) -> str | None:
    raw = _clean_string(value)
    return raw if raw in {"top", "bottom", "left", "right"} else None


def _place_on(value: object) -> JsonObject | None:
    record = _record(value)
    method = _clean_string(record.get("method"))
    if not method:
        return None
    return {
        "method": method,
        "target_instance_id": _clean_string(record.get("target_instance_id")),
    }


def _normalize_asset_id(value: object) -> str | None:
    raw = _clean_string(value)
    if not raw:
        return None
    for suffix in ("__reintroduced", "__accessory_refill"):
        marker_index = raw.lower().find(suffix)
        if marker_index > 0:
            return raw[:marker_index]
    return raw


def _bbox_key(value: object) -> str | None:
    record = _record(value)
    if not record:
        return None
    return ":".join(
        str(_number(record.get(key))) for key in ("min_x", "min_y", "max_x", "max_y")
    )


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip()
    return clean or None
