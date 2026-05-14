from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from evaluation.layout_benchmark.schema import (
    LayoutItem,
    NormalizedLayout,
    OpeningSegment,
    Point2D,
    points_from_json,
)

_OBJECT_LIST_KEYS = (
    "objects",
    "furniture",
    "furnitures",
    "placements",
    "layout",
    "items",
    "objects_in_room",
    "furniture_layout",
)


def load_output_payload(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def normalize_payload(
    *,
    normalizer: str,
    payload: object,
    benchmark_case: Mapping[str, object],
    system_id: str,
    source_path: Path | None = None,
) -> NormalizedLayout:
    if normalizer == "tknt":
        return normalize_tknt_payload(
            payload=payload,
            benchmark_case=benchmark_case,
            system_id=system_id,
            source_path=source_path,
        )
    if normalizer == "idesign":
        return normalize_generic_external_payload(
            payload=payload,
            benchmark_case=benchmark_case,
            system_id=system_id,
            source_path=source_path,
        )
    if normalizer == "flairgpt":
        parsed_payload = _parse_flair_text(payload)
        return normalize_generic_external_payload(
            payload=parsed_payload,
            benchmark_case=benchmark_case,
            system_id=system_id,
            source_path=source_path,
        )
    raise ValueError(f"Unknown normalizer: {normalizer}")


def normalize_tknt_payload(
    *,
    payload: object,
    benchmark_case: Mapping[str, object],
    system_id: str,
    source_path: Path | None = None,
) -> NormalizedLayout:
    root = _mapping(payload)
    final_output = _mapping(root.get("final_output")) or root
    room = _mapping(final_output.get("room"))
    room_polygon = _room_polygon(room, benchmark_case)
    openings = _openings(_mapping(room.get("openings")), benchmark_case)

    objects: list[LayoutItem] = []
    for row in _sequence(final_output.get("objects")):
        item = _mapping(row)
        if not item:
            continue
        polygon = _polygon_from_item(item)
        if len(polygon) < 3:
            continue
        object_type = _first_string(
            item,
            "object_type",
            "category",
            "asset_type",
            "object_id",
            "id",
        )
        object_id = _first_string(item, "instance_id", "id", "object_id") or object_type
        objects.append(
            LayoutItem(
                object_id=object_id,
                object_type=object_type,
                polygon=polygon,
                rotation_degrees=_number(item.get("rotation_ccw")) or 0.0,
                raw=dict(item),
            )
        )

    return NormalizedLayout(
        case_id=_case_id(benchmark_case),
        system_id=system_id,
        room_polygon=room_polygon,
        openings=openings,
        objects=tuple(objects),
        runtime_s=_runtime_s(root),
        input_tokens=_token_count(root, "input_tokens"),
        output_tokens=_token_count(root, "output_tokens"),
        source_path=str(source_path) if source_path else None,
    )


def normalize_generic_external_payload(
    *,
    payload: object,
    benchmark_case: Mapping[str, object],
    system_id: str,
    source_path: Path | None = None,
) -> NormalizedLayout:
    root = _mapping(payload)
    room_polygon = _room_polygon(_mapping(root.get("room")), benchmark_case)
    openings = _openings(_mapping(root.get("openings")), benchmark_case)

    objects: list[LayoutItem] = []
    for item in _iter_object_mappings(payload):
        object_type = _first_string(
            item,
            "object_type",
            "category",
            "type",
            "name",
            "label",
            "id",
        )
        if not object_type or _looks_like_room_shell(object_type):
            continue
        polygon = _polygon_from_item(item)
        if len(polygon) < 3:
            polygon = _polygon_from_external_box(item)
        if len(polygon) < 3:
            continue
        object_id = _first_string(item, "instance_id", "id", "object_id", "name")
        objects.append(
            LayoutItem(
                object_id=object_id or f"{object_type}_{len(objects) + 1}",
                object_type=object_type,
                polygon=polygon,
                rotation_degrees=_rotation_degrees(item),
                raw=dict(item),
            )
        )

    return NormalizedLayout(
        case_id=_case_id(benchmark_case),
        system_id=system_id,
        room_polygon=room_polygon,
        openings=openings,
        objects=tuple(objects),
        runtime_s=_runtime_s(root),
        input_tokens=_token_count(root, "input_tokens"),
        output_tokens=_token_count(root, "output_tokens"),
        source_path=str(source_path) if source_path else None,
    )


def _parse_flair_text(payload: object) -> object:
    if not isinstance(payload, str):
        return payload
    try:
        parsed = ast.literal_eval(payload)
        if isinstance(parsed, Mapping | Sequence):
            return parsed
    except (SyntaxError, ValueError):
        pass

    objects: list[dict[str, object]] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        name, raw_value = stripped.split(":", 1)
        parsed_row = _parse_object_assignment(name.strip(), raw_value.strip())
        if parsed_row:
            objects.append(parsed_row)
    return {"objects": objects}


def _parse_object_assignment(name: str, raw_value: str) -> dict[str, object] | None:
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, Mapping):
        row = dict(parsed)
        row.setdefault("name", name)
        return row
    if isinstance(parsed, Sequence) and not isinstance(parsed, str | bytes):
        values = list(parsed)
        if len(values) >= 4:
            return {
                "name": name,
                "x": values[0],
                "y": values[1],
                "width": values[2],
                "depth": values[3],
                "rotation": values[4] if len(values) > 4 else 0,
            }
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw_value)]
    if len(numbers) >= 4:
        return {
            "name": name,
            "x": numbers[0],
            "y": numbers[1],
            "width": numbers[2],
            "depth": numbers[3],
            "rotation": numbers[4] if len(numbers) > 4 else 0,
        }
    return None


def _room_polygon(
    room: Mapping[str, object],
    benchmark_case: Mapping[str, object],
) -> tuple[Point2D, ...]:
    polygon = points_from_json(_sequence(room.get("polygon_ccw")))
    if len(polygon) >= 3:
        return polygon
    case_room = _mapping(benchmark_case.get("room"))
    polygon = points_from_json(_sequence(case_room.get("polygon_mm")))
    if len(polygon) >= 3:
        return polygon
    dimensions = _sequence(case_room.get("dimensions_mm"))
    if len(dimensions) >= 2:
        width = _number(dimensions[0]) or 0.0
        depth = _number(dimensions[1]) or 0.0
        return _axis_aligned_box(0.0, 0.0, width, depth)
    return tuple()


def _openings(
    openings: Mapping[str, object],
    benchmark_case: Mapping[str, object],
) -> tuple[OpeningSegment, ...]:
    rows: list[OpeningSegment] = []
    benchmark_room = _mapping(benchmark_case.get("room"))
    source = (
        _mapping(benchmark_room.get("openings"))
        if "openings" in benchmark_room
        else openings
    )
    for kind in ("door", "window"):
        plural = f"{kind}s"
        for row in _sequence(source.get(plural)):
            item = _mapping(row)
            if not item:
                continue
            segment = _sequence(item.get("segment_mm"))
            if len(segment) < 2:
                segment = _sequence(item.get("segment"))
            start = _point(segment[0]) if len(segment) >= 1 else None
            end = _point(segment[1]) if len(segment) >= 2 else None
            if start is None or end is None:
                continue
            clearance = _number(item.get("clearance_mm"))
            if clearance is None:
                clearance = _number(item.get("swing_radius_mm"))
            if clearance is None:
                clearance = 900.0 if kind == "door" else 450.0
            rows.append(
                OpeningSegment(
                    opening_id=_first_string(item, "id", "door_id", "window_id")
                    or f"{kind}_{len(rows) + 1}",
                    kind=kind,  # type: ignore[arg-type]
                    start=start,
                    end=end,
                    clearance_mm=clearance,
                )
            )
    return tuple(rows)


def _iter_object_mappings(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        for key in _OBJECT_LIST_KEYS:
            rows = payload.get(key)
            if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
                continue
            for row in rows:
                item = _mapping(row)
                if item:
                    yield item
        for value in payload.values():
            if isinstance(value, Mapping):
                yield from _iter_object_mappings(value)
    elif isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        for row in payload:
            item = _mapping(row)
            if item:
                yield item


def _polygon_from_item(item: Mapping[str, object]) -> tuple[Point2D, ...]:
    polygon = points_from_json(_sequence(item.get("polygon_ccw")))
    if len(polygon) >= 3:
        return polygon
    polygon = points_from_json(_sequence(item.get("polygon")))
    if len(polygon) >= 3:
        return polygon
    bbox = _mapping(item.get("bbox"))
    if bbox:
        min_x = _number(bbox.get("min_x"))
        min_y = _number(bbox.get("min_y"))
        max_x = _number(bbox.get("max_x"))
        max_y = _number(bbox.get("max_y"))
        if None not in (min_x, min_y, max_x, max_y):
            return _axis_aligned_box(
                min_x or 0.0,
                min_y or 0.0,
                max_x or 0.0,
                max_y or 0.0,
            )
    return tuple()


def _polygon_from_external_box(item: Mapping[str, object]) -> tuple[Point2D, ...]:
    width = _dimension_mm(_first_number(item, "width", "w", "size_x", "x_size"))
    depth = _dimension_mm(
        _first_number(item, "depth", "length", "height", "h", "size_y", "y_size")
    )
    if width is None or depth is None or width <= 0 or depth <= 0:
        size = _mapping(item.get("size"))
        width = _dimension_mm(_first_number(size, "width", "w", "x"))
        depth = _dimension_mm(_first_number(size, "depth", "length", "h", "y"))
    if width is None or depth is None or width <= 0 or depth <= 0:
        return tuple()

    position = _mapping(item.get("position"))
    x = _coordinate_mm(_first_number(item, "x", "center_x", "left"))
    y = _coordinate_mm(_first_number(item, "y", "center_y", "top"))
    if x is None:
        x = _coordinate_mm(_first_number(position, "x", "center_x"))
    if y is None:
        y = _coordinate_mm(_first_number(position, "y", "center_y", "z"))
    if x is None or y is None:
        return tuple()

    angle_degrees = _rotation_degrees(item)
    return _rotated_box(
        center_x=x, center_y=y, width=width, depth=depth, angle=angle_degrees
    )


def _axis_aligned_box(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> tuple[Point2D, ...]:
    return (
        Point2D(min_x, min_y),
        Point2D(max_x, min_y),
        Point2D(max_x, max_y),
        Point2D(min_x, max_y),
    )


def _rotated_box(
    *,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    angle: float,
) -> tuple[Point2D, ...]:
    half_w = width / 2.0
    half_d = depth / 2.0
    radians = math.radians(angle)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    points: list[Point2D] = []
    for local_x, local_y in (
        (-half_w, -half_d),
        (half_w, -half_d),
        (half_w, half_d),
        (-half_w, half_d),
    ):
        points.append(
            Point2D(
                x=center_x + local_x * cos_a - local_y * sin_a,
                y=center_y + local_x * sin_a + local_y * cos_a,
            )
        )
    return tuple(points)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _point(value: object) -> Point2D | None:
    item = _mapping(value)
    x = _number(item.get("x"))
    y = _number(item.get("y"))
    if x is None or y is None:
        return None
    return Point2D(x=x, y=y)


def _first_string(item: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace(" ", "_")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
    return ""


def _first_number(item: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _number(item.get(key))
        if value is not None:
            return value
    return None


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


def _dimension_mm(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000.0 if abs(value) < 50.0 else value


def _coordinate_mm(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000.0 if abs(value) < 50.0 else value


def _rotation_degrees(item: Mapping[str, object]) -> float:
    value = _first_number(item, "rotation_degrees", "rotation_ccw", "angle_degrees")
    if value is not None:
        return value
    radians = _first_number(item, "rotation", "angle", "theta")
    if radians is None:
        return 0.0
    if abs(radians) <= math.tau:
        return math.degrees(radians)
    return radians


def _runtime_s(root: Mapping[str, object]) -> float | None:
    return _first_number(root, "runtime_s", "elapsed_s", "duration_s")


def _token_count(root: Mapping[str, object], key: str) -> int | None:
    value = _first_number(root, key)
    if value is None:
        return None
    return int(value)


def _case_id(benchmark_case: Mapping[str, object]) -> str:
    value = benchmark_case.get("case_id")
    return str(value).strip() if value is not None else "unknown_case"


def _looks_like_room_shell(object_type: str) -> bool:
    return object_type in {"room", "wall", "walls", "ceiling", "floor", "middle"}
