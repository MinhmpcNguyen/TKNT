from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

JsonDict = dict[str, object]
OpeningKind = Literal["door", "window"]


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class OpeningSegment:
    opening_id: str
    kind: OpeningKind
    start: Point2D
    end: Point2D
    clearance_mm: float


@dataclass(frozen=True)
class LayoutItem:
    object_id: str
    object_type: str
    polygon: tuple[Point2D, ...]
    rotation_degrees: float = 0.0
    raw: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedLayout:
    case_id: str
    system_id: str
    room_polygon: tuple[Point2D, ...]
    openings: tuple[OpeningSegment, ...]
    objects: tuple[LayoutItem, ...]
    runtime_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    source_path: str | None = None


def point_to_json(point: Point2D) -> JsonDict:
    return {"x": point.x, "y": point.y}


def opening_to_json(opening: OpeningSegment) -> JsonDict:
    return {
        "id": opening.opening_id,
        "kind": opening.kind,
        "start": point_to_json(opening.start),
        "end": point_to_json(opening.end),
        "clearance_mm": opening.clearance_mm,
    }


def item_to_json(item: LayoutItem) -> JsonDict:
    return {
        "id": item.object_id,
        "type": item.object_type,
        "polygon": [point_to_json(point) for point in item.polygon],
        "rotation_degrees": item.rotation_degrees,
        "raw": dict(item.raw),
    }


def normalized_layout_to_json(layout: NormalizedLayout) -> JsonDict:
    payload: JsonDict = {
        "case_id": layout.case_id,
        "system_id": layout.system_id,
        "room_polygon": [point_to_json(point) for point in layout.room_polygon],
        "openings": [opening_to_json(opening) for opening in layout.openings],
        "objects": [item_to_json(item) for item in layout.objects],
    }
    if layout.runtime_s is not None:
        payload["runtime_s"] = layout.runtime_s
    if layout.input_tokens is not None:
        payload["input_tokens"] = layout.input_tokens
    if layout.output_tokens is not None:
        payload["output_tokens"] = layout.output_tokens
    if layout.source_path:
        payload["source_path"] = layout.source_path
    return payload


def points_from_json(rows: Sequence[object]) -> tuple[Point2D, ...]:
    points: list[Point2D] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        x = _number(row.get("x"))
        y = _number(row.get("y"))
        if x is None or y is None:
            continue
        points.append(Point2D(x=x, y=y))
    return tuple(points)


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
