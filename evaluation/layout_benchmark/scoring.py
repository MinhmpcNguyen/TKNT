from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evaluation.layout_benchmark.schema import LayoutItem, NormalizedLayout, Point2D

try:
    from shapely.geometry import LineString, Polygon
    from shapely.geometry.base import BaseGeometry
except Exception as exc:  # pragma: no cover - dependency guard only
    Polygon = None  # type: ignore[assignment]
    LineString = None  # type: ignore[assignment]
    BaseGeometry = object  # type: ignore[assignment]
    _SHAPELY_IMPORT_ERROR = exc
else:
    _SHAPELY_IMPORT_ERROR = None

_OVERLAP_AREA_THRESHOLD_MM2 = 10_000.0
_OOB_AREA_THRESHOLD_MM2 = 5_000.0
_ACCESS_PATH_WIDTH_MM = 700.0

_ALIASES: dict[str, tuple[str, ...]] = {
    "bed": ("bed", "queen_bed", "king_bed", "single_bed", "double_bed"),
    "nightstand": ("nightstand", "bedside_table", "side_table"),
    "wardrobe": ("wardrobe", "closet", "armoire"),
    "desk": ("desk", "work_table", "study_table"),
    "chair": ("chair", "desk_chair", "dining_chair", "seat"),
    "sofa": ("sofa", "couch", "loveseat", "sectional"),
    "coffee_table": ("coffee_table", "tea_table", "center_table"),
    "tv_console": ("tv_console", "media_console", "tv_stand"),
    "tv": ("tv", "television"),
    "rug": ("rug", "carpet"),
    "curtain": ("curtain", "drape", "blind"),
    "ceiling_lamp": ("ceiling_lamp", "ceiling_light", "overhead_light"),
    "floor_lamp": ("floor_lamp", "standing_lamp"),
    "table_lamp": ("table_lamp", "bedside_lamp", "desk_lamp", "lamp"),
    "armchair": ("armchair", "lounge_chair", "accent_chair"),
    "bookshelf": ("bookshelf", "bookcase", "shelf"),
}

_NON_BLOCKING_TOKENS = (
    "rug",
    "carpet",
    "curtain",
    "blind",
    "cushion",
    "pillow",
    "throw",
    "wall_art",
    "art",
    "painting",
    "picture",
    "ceiling_lamp",
    "ceiling_light",
    "overhead_light",
    "table_lamp",
    "desk_lamp",
    "bedside_lamp",
)


@dataclass(frozen=True)
class _ShapeItem:
    item: LayoutItem
    canonical_type: str
    blocking: bool
    polygon: BaseGeometry


def score_layout(
    *,
    benchmark_case: Mapping[str, object],
    layout: NormalizedLayout,
) -> dict[str, object]:
    if Polygon is None or LineString is None:
        raise RuntimeError("Install shapely to score benchmark layouts.") from (
            _SHAPELY_IMPORT_ERROR
        )

    room_shape = _polygon_shape(layout.room_polygon)
    if room_shape is None:
        raise ValueError(f"Case {layout.case_id} has no valid room polygon.")

    shaped_items = [
        shaped for item in layout.objects if (shaped := _shape_item(item)) is not None
    ]
    blocking_items = [item for item in shaped_items if item.blocking]
    out_of_bounds = _out_of_bounds(blocking_items, room_shape)
    overlaps = _blocking_overlaps(blocking_items)
    opening_hits = _opening_clearance_hits(blocking_items, layout)
    access_hits = _access_path_hits(blocking_items, layout, room_shape)
    request_checks = _request_checks(
        benchmark_case=benchmark_case,
        layout=layout,
        shaped_items=shaped_items,
    )
    count_score = _object_count_score(
        benchmark_case=benchmark_case,
        valid_blocking_count=_valid_blocking_count(
            blocking_items=blocking_items,
            out_of_bounds_ids={item_id for item_id, _area in out_of_bounds},
            overlap_ids=_overlap_item_ids(overlaps),
        ),
    )

    geometry_score = _clamp_score(
        100.0
        - len(out_of_bounds) * 22.0
        - len(overlaps) * 24.0
        - len(opening_hits) * 16.0
        - len(access_hits) * 12.0
        - min(_total_area(out_of_bounds) / 40_000.0, 18.0)
        - min(_total_overlap_area(overlaps) / 50_000.0, 18.0)
    )
    request_score = _request_score(request_checks)
    arrangement_score = _clamp_score(
        100.0
        - len(overlaps) * 18.0
        - len(out_of_bounds) * 14.0
        - len(opening_hits) * 12.0
        - len(access_hits) * 10.0
    )
    object_count_score = _clamp_score(count_score)
    total_score = _clamp_score(
        geometry_score * 0.30
        + request_score * 0.25
        + arrangement_score * 0.25
        + object_count_score * 0.20
    )

    valid_blocking_count = _valid_blocking_count(
        blocking_items=blocking_items,
        out_of_bounds_ids={item_id for item_id, _area in out_of_bounds},
        overlap_ids=_overlap_item_ids(overlaps),
    )
    return {
        "case_id": layout.case_id,
        "system_id": layout.system_id,
        "overall_score": round(total_score, 3),
        "component_scores": {
            "geometry": round(geometry_score, 3),
            "request_following": round(request_score, 3),
            "arrangement_reasonableness": round(arrangement_score, 3),
            "valid_object_count": round(object_count_score, 3),
        },
        "metrics": {
            "runtime_s": layout.runtime_s,
            "input_tokens": layout.input_tokens,
            "output_tokens": layout.output_tokens,
            "objects_total": len(layout.objects),
            "blocking_objects_total": len(blocking_items),
            "valid_blocking_object_count": valid_blocking_count,
            "objects_out_of_bounds": len(out_of_bounds),
            "out_of_bounds_area_mm2": round(_total_area(out_of_bounds), 3),
            "overlapping_blocking_pairs": len(overlaps),
            "overlap_area_mm2": round(_total_overlap_area(overlaps), 3),
            "opening_clearance_violations": len(opening_hits),
            "access_path_blockers": len(access_hits),
        },
        "violations": {
            "out_of_bounds": [
                {"object_id": item_id, "area_mm2": round(area, 3)}
                for item_id, area in out_of_bounds
            ],
            "overlaps": [
                {
                    "a": first_id,
                    "b": second_id,
                    "area_mm2": round(area, 3),
                }
                for first_id, second_id, area in overlaps
            ],
            "opening_clearance": opening_hits,
            "access_paths": access_hits,
        },
        "request_checks": request_checks,
    }


def _shape_item(item: LayoutItem) -> _ShapeItem | None:
    polygon = _polygon_shape(item.polygon)
    if polygon is None or polygon.area <= 0:
        return None
    canonical_type = canonical_object_type(item.object_type)
    return _ShapeItem(
        item=item,
        canonical_type=canonical_type,
        blocking=_is_blocking(canonical_type) and not _has_non_blocking_placement(item),
        polygon=polygon,
    )


def canonical_object_type(object_type: str) -> str:
    normalized = object_type.strip().lower().replace(" ", "_").replace("-", "_")
    candidates_to_check = _canonical_name_candidates(normalized)
    for canonical, aliases in _ALIASES.items():
        if any(
            candidate == canonical or candidate in aliases
            for candidate in candidates_to_check
        ):
            return canonical
    for canonical, aliases in _ALIASES.items():
        candidates = (canonical, *aliases)
        if any(
            "_" in alias
            and any(
                _object_type_alias_matches(candidate, alias)
                for candidate in candidates_to_check
            )
            for alias in candidates
        ):
            return canonical
    for canonical, aliases in _ALIASES.items():
        candidates = (canonical, *aliases)
        if any(
            "_" not in alias
            and any(
                _object_type_alias_matches(candidate, alias)
                for candidate in candidates_to_check
            )
            for alias in candidates
        ):
            return canonical
    return normalized


def _canonical_name_candidates(normalized: str) -> tuple[str, ...]:
    stripped = normalized.rstrip("0123456789").rstrip("_")
    if stripped and stripped != normalized:
        return (normalized, stripped)
    return (normalized,)


def _object_type_alias_matches(normalized: str, alias: str) -> bool:
    alias = alias.strip().lower().replace(" ", "_").replace("-", "_")
    if not alias:
        return False
    if normalized == alias:
        return True
    if "_" in alias:
        return (
            normalized.startswith(f"{alias}_")
            or normalized.endswith(f"_{alias}")
            or f"_{alias}_" in normalized
        )
    return alias in normalized.split("_")


def _is_blocking(canonical_type: str) -> bool:
    return not any(token in canonical_type for token in _NON_BLOCKING_TOKENS)


def _has_non_blocking_placement(item: LayoutItem) -> bool:
    place_on = _mapping(item.raw.get("place_on")) if isinstance(item.raw, Mapping) else {}
    method = str(place_on.get("method") or "").strip().lower()
    return method in {"on_top", "hang_on", "wall", "ceiling"}


def _polygon_shape(points: Sequence[Point2D]) -> BaseGeometry | None:
    if Polygon is None or len(points) < 3:
        return None
    polygon = Polygon([(point.x, point.y) for point in points])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    return polygon


def _out_of_bounds(
    items: Sequence[_ShapeItem],
    room_shape: BaseGeometry,
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for item in items:
        outside_area = item.polygon.difference(room_shape).area
        if outside_area > _OOB_AREA_THRESHOLD_MM2:
            rows.append((item.item.object_id, float(outside_area)))
    return rows


def _blocking_overlaps(
    items: Sequence[_ShapeItem],
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            area = first.polygon.intersection(second.polygon).area
            if area > _OVERLAP_AREA_THRESHOLD_MM2:
                rows.append((first.item.object_id, second.item.object_id, float(area)))
    return rows


def _opening_clearance_hits(
    items: Sequence[_ShapeItem],
    layout: NormalizedLayout,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if LineString is None:
        return rows
    for opening in layout.openings:
        line = LineString(
            [(opening.start.x, opening.start.y), (opening.end.x, opening.end.y)]
        )
        clearance = max(
            opening.clearance_mm, 900.0 if opening.kind == "door" else 450.0
        )
        zone = line.buffer(clearance, cap_style=2)
        for item in items:
            area = item.polygon.intersection(zone).area
            if area <= _OOB_AREA_THRESHOLD_MM2:
                continue
            rows.append(
                {
                    "object_id": item.item.object_id,
                    "opening_id": opening.opening_id,
                    "opening_kind": opening.kind,
                    "area_mm2": round(float(area), 3),
                }
            )
    return rows


def _access_path_hits(
    items: Sequence[_ShapeItem],
    layout: NormalizedLayout,
    room_shape: BaseGeometry,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if LineString is None:
        return rows
    room_center = room_shape.centroid
    for opening in layout.openings:
        if opening.kind != "door":
            continue
        mid_x = (opening.start.x + opening.end.x) / 2.0
        mid_y = (opening.start.y + opening.end.y) / 2.0
        path = LineString([(mid_x, mid_y), (room_center.x, room_center.y)]).buffer(
            _ACCESS_PATH_WIDTH_MM / 2.0,
            cap_style=2,
        )
        for item in items:
            area = item.polygon.intersection(path).area
            if area <= _OOB_AREA_THRESHOLD_MM2:
                continue
            rows.append(
                {
                    "object_id": item.item.object_id,
                    "opening_id": opening.opening_id,
                    "area_mm2": round(float(area), 3),
                }
            )
    return rows


def _request_checks(
    *,
    benchmark_case: Mapping[str, object],
    layout: NormalizedLayout,
    shaped_items: Sequence[_ShapeItem],
) -> list[dict[str, object]]:
    expected = _mapping(benchmark_case.get("expected"))
    checks: list[dict[str, object]] = []
    present_types = {item.canonical_type for item in shaped_items}
    for object_type in _string_sequence(expected.get("required_objects")):
        canonical = canonical_object_type(object_type)
        passed = canonical in present_types
        checks.append(
            {
                "type": "required_object",
                "target": canonical,
                "passed": passed,
            }
        )

    count_ranges = _mapping(expected.get("count_ranges"))
    counts = _counts_by_type(shaped_items)
    for object_type, raw_range in count_ranges.items():
        range_map = _mapping(raw_range)
        canonical = canonical_object_type(str(object_type))
        count = counts.get(canonical, 0)
        minimum = _int_or_none(range_map.get("min"))
        maximum = _int_or_none(range_map.get("max"))
        passed = (minimum is None or count >= minimum) and (
            maximum is None or count <= maximum
        )
        checks.append(
            {
                "type": "count_range",
                "target": canonical,
                "count": count,
                "min": minimum,
                "max": maximum,
                "passed": passed,
            }
        )

    for relation in _sequence(expected.get("relations")):
        relation_map = _mapping(relation)
        if relation_map.get("type") == "avoid_near_opening":
            checks.append(_avoid_near_opening_check(relation_map, shaped_items, layout))
        elif relation_map.get("type") == "prefer_near_opening":
            checks.append(
                _prefer_near_opening_check(relation_map, shaped_items, layout)
            )
    return checks


def _avoid_near_opening_check(
    relation: Mapping[str, object],
    shaped_items: Sequence[_ShapeItem],
    layout: NormalizedLayout,
) -> dict[str, object]:
    targets = {
        canonical_object_type(value)
        for value in _string_sequence(relation.get("objects"))
    }
    opening_kind = str(relation.get("opening_kind") or "").strip().lower()
    minimum = _float_or_default(relation.get("min_distance_mm"), 900.0)
    offenders: list[str] = []
    for item in shaped_items:
        if item.canonical_type not in targets:
            continue
        if _nearest_opening_distance(item, layout, opening_kind) < minimum:
            offenders.append(item.item.object_id)
    return {
        "type": "avoid_near_opening",
        "target": sorted(targets),
        "opening_kind": opening_kind,
        "min_distance_mm": minimum,
        "offenders": offenders,
        "passed": not offenders,
    }


def _prefer_near_opening_check(
    relation: Mapping[str, object],
    shaped_items: Sequence[_ShapeItem],
    layout: NormalizedLayout,
) -> dict[str, object]:
    targets = {
        canonical_object_type(value)
        for value in _string_sequence(relation.get("objects"))
    }
    opening_kind = str(relation.get("opening_kind") or "").strip().lower()
    maximum = _float_or_default(relation.get("max_distance_mm"), 1500.0)
    matched = [
        item.item.object_id
        for item in shaped_items
        if item.canonical_type in targets
        and _nearest_opening_distance(item, layout, opening_kind) <= maximum
    ]
    return {
        "type": "prefer_near_opening",
        "target": sorted(targets),
        "opening_kind": opening_kind,
        "max_distance_mm": maximum,
        "matched": matched,
        "passed": bool(matched),
    }


def _nearest_opening_distance(
    item: _ShapeItem,
    layout: NormalizedLayout,
    opening_kind: str,
) -> float:
    if LineString is None:
        return float("inf")
    distances: list[float] = []
    for opening in layout.openings:
        if opening_kind and opening.kind != opening_kind:
            continue
        line = LineString(
            [(opening.start.x, opening.start.y), (opening.end.x, opening.end.y)]
        )
        distances.append(float(item.polygon.distance(line)))
    return min(distances) if distances else float("inf")


def _object_count_score(
    *,
    benchmark_case: Mapping[str, object],
    valid_blocking_count: int,
) -> float:
    expected = _mapping(benchmark_case.get("expected"))
    minimum = _int_or_none(expected.get("valid_blocking_count_min"))
    maximum = _int_or_none(expected.get("valid_blocking_count_max"))
    if minimum is None and maximum is None:
        return 100.0
    score = 100.0
    if minimum is not None and valid_blocking_count < minimum:
        score -= (minimum - valid_blocking_count) * 18.0
    if maximum is not None and valid_blocking_count > maximum:
        score -= (valid_blocking_count - maximum) * 10.0
    return score


def _valid_blocking_count(
    *,
    blocking_items: Sequence[_ShapeItem],
    out_of_bounds_ids: set[str],
    overlap_ids: set[str],
) -> int:
    invalid_ids = out_of_bounds_ids | overlap_ids
    return len(
        [item for item in blocking_items if item.item.object_id not in invalid_ids]
    )


def _overlap_item_ids(overlaps: Sequence[tuple[str, str, float]]) -> set[str]:
    ids: set[str] = set()
    for first_id, second_id, _area in overlaps:
        ids.add(first_id)
        ids.add(second_id)
    return ids


def _request_score(checks: Sequence[Mapping[str, object]]) -> float:
    if not checks:
        return 100.0
    passed = len([row for row in checks if row.get("passed") is True])
    return (passed / len(checks)) * 100.0


def _counts_by_type(items: Sequence[_ShapeItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.canonical_type] = counts.get(item.canonical_type, 0) + 1
    return counts


def _total_area(rows: Sequence[tuple[str, float]]) -> float:
    return sum(area for _item_id, area in rows)


def _total_overlap_area(rows: Sequence[tuple[str, str, float]]) -> float:
    return sum(area for _first_id, _second_id, area in rows)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _string_sequence(value: object) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in _sequence(value) if str(item).strip())


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _float_or_default(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))
