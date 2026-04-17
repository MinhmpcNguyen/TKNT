from __future__ import annotations

from copy import deepcopy
from math import cos, hypot, radians, sin
from typing import Any

from layout.orientation_contract import (
    effective_front_rotation as _effective_front_rotation_contract,
    effective_front_side as _effective_front_side_contract,
    normalize_cardinal_side,
    side_to_vec,
)

try:
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - import guard only
    LineString = None  # type: ignore[assignment]
    Polygon = None  # type: ignore[assignment]
    unary_union = None  # type: ignore[assignment]


_WALL_BACK_THRESHOLD_MM = 120.0
_PERIMETER_THRESHOLD_MM = 140.0
_MIN_CLUSTER_ALIGNMENT_DOT = 0.1


def collect_dropped_inventory(
    *,
    previous_tier_output: dict[str, Any],
    next_tier_output: dict[str, Any],
    attempt: int,
) -> dict[str, list[dict[str, Any]]]:
    previous_lookup = _decision_lookup(previous_tier_output)
    next_lookup = _decision_lookup(next_tier_output)

    dropped_by_cluster: dict[str, list[dict[str, Any]]] = {}
    for key, previous_row in previous_lookup.items():
        cluster_id, object_type = key
        previous_quantity = max(0, int(previous_row.get("quantity") or 0))
        next_row = next_lookup.get(key)
        next_quantity = (
            max(0, int(next_row.get("quantity") or 0))
            if isinstance(next_row, dict)
            else 0
        )
        removed_count = max(0, previous_quantity - next_quantity)
        if removed_count <= 0:
            continue

        cluster_items = dropped_by_cluster.setdefault(cluster_id, [])
        for removed_index in range(removed_count):
            cluster_items.append(
                {
                    "cluster_id": cluster_id,
                    "object_type": object_type,
                    "category": str(previous_row.get("category") or object_type),
                    "size_tier": str(previous_row.get("size_tier") or "M"),
                    "priority": str(previous_row.get("priority") or "secondary"),
                    "rep_dims_m": deepcopy(previous_row.get("rep_dims_m") or {}),
                    "rationale": str(previous_row.get("rationale") or ""),
                    "drop_attempt": int(attempt),
                    "drop_sequence": removed_index + 1,
                }
            )
    return dropped_by_cluster


def collect_seed_omitted_inventory(
    *,
    tier_output: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    omitted_by_cluster: dict[str, list[dict[str, Any]]] = {}
    decisions = tier_output.get("decisions")
    if not isinstance(decisions, list):
        return omitted_by_cluster

    for row in decisions:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        object_type = str(row.get("object_type") or row.get("category") or "").strip()
        quantity = max(0, int(row.get("quantity") or 0))
        if not cluster_id or not object_type or quantity > 0:
            continue
        omitted_by_cluster.setdefault(cluster_id, []).append(
            {
                "cluster_id": cluster_id,
                "object_type": object_type,
                "category": str(row.get("category") or object_type),
                "size_tier": str(row.get("size_tier") or "M"),
                "priority": str(row.get("priority") or "secondary"),
                "rep_dims_m": deepcopy(row.get("rep_dims_m") or {}),
                "rationale": str(row.get("rationale") or ""),
                "drop_reason": "seed_omission",
                "drop_attempt": 0,
                "drop_sequence": 1,
            }
        )
    return omitted_by_cluster


def merge_dropped_inventory(
    base_inventory: dict[str, list[dict[str, Any]]] | None,
    additions: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {
        str(cluster_id): [deepcopy(item) for item in items if isinstance(item, dict)]
        for cluster_id, items in (base_inventory or {}).items()
        if isinstance(cluster_id, str) and isinstance(items, list)
    }
    for cluster_id, items in (additions or {}).items():
        if not isinstance(cluster_id, str) or not isinstance(items, list):
            continue
        target = merged.setdefault(cluster_id, [])
        target.extend(deepcopy(item) for item in items if isinstance(item, dict))
    return merged


def dropped_inventory_payload(
    dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    total_items = 0
    for cluster_id in sorted((dropped_inventory_by_cluster or {}).keys()):
        items = dropped_inventory_by_cluster.get(cluster_id)
        if not isinstance(items, list) or not items:
            continue
        clean_items = [deepcopy(item) for item in items if isinstance(item, dict)]
        if not clean_items:
            continue
        total_items += len(clean_items)
        clusters.append({"cluster_id": cluster_id, "items": clean_items})
    return {
        "status": "OK",
        "total_dropped_items": total_items,
        "clusters": clusters,
    }


def reintroduce_dropped_inventory(
    *,
    room_output: dict[str, Any],
    absolute_layout: dict[str, Any],
    cluster_output: dict[str, Any],
    dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]] | None,
    grid_mm: int = 50,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout = deepcopy(absolute_layout)
    if not isinstance(layout.get("objects"), list):
        layout["objects"] = []

    inventory = {
        str(cluster_id): [deepcopy(item) for item in items if isinstance(item, dict)]
        for cluster_id, items in (dropped_inventory_by_cluster or {}).items()
        if isinstance(cluster_id, str) and isinstance(items, list)
    }
    if not inventory:
        return layout, _reintroduction_summary(inventory, {}, {})

    room_polygon = _room_polygon(room_output)
    if room_polygon is None or Polygon is None or unary_union is None:
        return layout, _reintroduction_summary(inventory, {}, inventory)

    cluster_lookup = _cluster_lookup(cluster_output)
    objects = [
        deepcopy(row) for row in layout.get("objects") or [] if isinstance(row, dict)
    ]
    existing_ids = {
        str(row.get("object_id") or "").strip()
        for row in objects
        if isinstance(row.get("object_id"), str) and str(row.get("object_id")).strip()
    }
    remaining = {
        cluster_id: [deepcopy(item) for item in items]
        for cluster_id, items in inventory.items()
        if items
    }
    inserted_by_cluster: dict[str, list[dict[str, Any]]] = {}

    while True:
        active_cluster_ids = [
            cluster_id for cluster_id, items in remaining.items() if items
        ]
        if not active_cluster_ids:
            break

        progress = False
        for cluster_id in active_cluster_ids:
            item = remaining[cluster_id][0]
            cluster_meta = cluster_lookup.get(cluster_id) or {}
            placed_object = _place_reintroduced_item(
                room_polygon=room_polygon,
                room_output=room_output,
                existing_objects=objects,
                existing_ids=existing_ids,
                cluster_id=cluster_id,
                cluster_meta=cluster_meta,
                item=item,
                grid_mm=grid_mm,
            )
            if placed_object is None:
                continue

            objects.append(placed_object)
            existing_ids.add(str(placed_object["object_id"]))
            remaining[cluster_id].pop(0)
            inserted_by_cluster.setdefault(cluster_id, []).append(
                {
                    "object_id": str(placed_object["object_id"]),
                    "object_type": str(item.get("object_type") or ""),
                    "rotation_ccw": int(placed_object.get("rotation_ccw") or 0),
                    "bbox": deepcopy(placed_object.get("bbox") or {}),
                }
            )
            progress = True
        if not progress:
            break

    layout["objects"] = objects
    summary = _reintroduction_summary(inventory, inserted_by_cluster, remaining)
    layout["reintroduced_objects_by_cluster"] = deepcopy(
        summary["reintroduced_objects_by_cluster"]
    )
    layout["remaining_dropped_objects_by_cluster"] = deepcopy(
        summary["remaining_dropped_objects_by_cluster"]
    )
    if summary["inserted_count"] > 0:
        notes = [
            str(item).strip() for item in layout.get("notes") or [] if str(item).strip()
        ]
        note = (
            "Reintroduced "
            f"{summary['inserted_count']} deferred object(s) after judge/placer "
            "using the available free space in the room."
        )
        if note not in notes:
            notes.append(note)
        layout["notes"] = notes
    return layout, summary


def _decision_lookup(
    tier_output: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    decisions = tier_output.get("decisions")
    if not isinstance(decisions, list):
        return lookup
    for row in decisions:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        object_type = str(row.get("object_type") or row.get("category") or "").strip()
        if not cluster_id or not object_type:
            continue
        lookup[(cluster_id, object_type)] = row
    return lookup


def _cluster_lookup(cluster_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    clusters = cluster_output.get("clusters")
    if not isinstance(clusters, list):
        return {}
    return {
        str(cluster.get("cluster_id") or "").strip(): cluster
        for cluster in clusters
        if isinstance(cluster, dict) and str(cluster.get("cluster_id") or "").strip()
    }


def _room_polygon(room_output: dict[str, Any]) -> Any | None:
    if Polygon is None:
        return None
    room = room_output.get("room")
    polygon_rows = room.get("polygon_ccw") if isinstance(room, dict) else None
    if not isinstance(polygon_rows, list):
        return None
    points = [
        (float(row.get("x") or 0.0), float(row.get("y") or 0.0))
        for row in polygon_rows
        if isinstance(row, dict)
    ]
    if len(points) < 3:
        return None
    try:
        return Polygon(points)
    except Exception:
        return None


def _blocked_polygons(
    room_output: dict[str, Any], objects: list[dict[str, Any]]
) -> list[Any]:
    blocked: list[Any] = []
    for row in objects:
        polygon = _polygon_from_rows(row.get("polygon_ccw"))
        if polygon is not None:
            blocked.append(polygon)

    obstacle_ids: set[str] = set()
    for obstacle in _obstacle_rows(room_output):
        if not isinstance(obstacle, dict):
            continue
        obstacle_id = str(obstacle.get("id") or "").strip()
        if obstacle_id:
            obstacle_ids.add(obstacle_id)
        polygon = _polygon_from_rows(obstacle.get("polygon_ccw"))
        if polygon is not None:
            blocked.append(polygon)

    blocked.extend(_opening_guard_polygons(room_output, obstacle_ids))
    return blocked


def _obstacle_rows(room_output: dict[str, Any]) -> list[dict[str, Any]]:
    room = room_output.get("room")
    raw_sources = [
        room_output.get("obstacles"),
        room.get("obstacles") if isinstance(room, dict) else None,
    ]

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            obstacle_id = str(item.get("id") or "").strip()
            if obstacle_id and obstacle_id in seen_ids:
                continue
            if obstacle_id:
                seen_ids.add(obstacle_id)
            rows.append(item)
    return rows


def _opening_guard_polygons(
    room_output: dict[str, Any],
    obstacle_ids: set[str],
) -> list[Any]:
    if LineString is None:
        return []

    openings = room_output.get("openings")
    if not isinstance(openings, dict):
        room = room_output.get("room")
        openings = room.get("openings") if isinstance(room, dict) else None
    if not isinstance(openings, dict):
        return []

    guards: list[Any] = []
    for door in openings.get("doors") or []:
        if not isinstance(door, dict):
            continue
        door_id = str(door.get("id") or "").strip()
        if door_id and f"{door_id}_swing" in obstacle_ids:
            continue
        line = _opening_line(door.get("segment_mm"))
        if line is None:
            continue
        depth = float(door.get("swing_radius_mm") or 700.0)
        if depth <= 0.0:
            depth = 700.0
        guards.append(line.buffer(depth, cap_style=2, join_style=2))

    for window in openings.get("windows") or []:
        if not isinstance(window, dict):
            continue
        window_id = str(window.get("id") or "").strip()
        if window_id and f"{window_id}_clearance" in obstacle_ids:
            continue
        line = _opening_line(window.get("segment_mm"))
        if line is None:
            continue
        depth = float(window.get("clearance_mm") or 100.0)
        if depth <= 0.0:
            depth = 100.0
        guards.append(line.buffer(depth, cap_style=2, join_style=2))
    return guards


def _opening_line(rows: Any) -> Any | None:
    if LineString is None or not isinstance(rows, list):
        return None
    points = [
        (float(row.get("x") or 0.0), float(row.get("y") or 0.0))
        for row in rows
        if isinstance(row, dict)
    ]
    if len(points) < 2:
        return None
    try:
        return LineString(points[:2])
    except Exception:
        return None


def _polygon_from_rows(rows: Any) -> Any | None:
    if Polygon is None or not isinstance(rows, list):
        return None
    points = [
        (float(row.get("x") or 0.0), float(row.get("y") or 0.0))
        for row in rows
        if isinstance(row, dict)
    ]
    if len(points) < 3:
        return None
    try:
        return Polygon(points)
    except Exception:
        return None


def _place_reintroduced_item(
    *,
    room_polygon: Any,
    room_output: dict[str, Any],
    existing_objects: list[dict[str, Any]],
    existing_ids: set[str],
    cluster_id: str,
    cluster_meta: dict[str, Any],
    item: dict[str, Any],
    grid_mm: int,
) -> dict[str, Any] | None:
    dims_mm = _object_dims_mm(item)
    if dims_mm is None:
        return None
    source_width, source_height = dims_mm
    blocked_polygons = _blocked_polygons(room_output, existing_objects)
    free_regions = _free_regions(room_polygon, blocked_polygons)
    cluster_bbox = _cluster_bbox(existing_objects, cluster_id)
    object_type = str(
        item.get("object_type") or item.get("category") or "object"
    ).strip()
    allowed_rotations = _allowed_rotations(cluster_meta, object_type)
    front_side = _cluster_front_side(cluster_meta, object_type)
    room_center = _room_center(room_polygon)
    cluster_center = _cluster_center(cluster_bbox)
    cluster_target = cluster_center or room_center
    best_candidate: dict[str, Any] | None = None
    best_preference: tuple[float, ...] | None = None

    for rotation in allowed_rotations:
        box_width, box_height = (
            (source_width, source_height)
            if rotation % 180 == 0
            else (source_height, source_width)
        )
        candidate_origins = _candidate_origins(
            cluster_bbox=cluster_bbox,
            free_regions=free_regions,
            box_width=box_width,
            box_height=box_height,
            grid_mm=grid_mm,
        )
        for min_x, min_y in candidate_origins:
            polygon = Polygon(
                [
                    (float(min_x), float(min_y)),
                    (float(min_x + box_width), float(min_y)),
                    (float(min_x + box_width), float(min_y + box_height)),
                    (float(min_x), float(min_y + box_height)),
                ]
            )
            if not room_polygon.buffer(1e-6).covers(polygon):
                continue
            if _overlaps_blockers(polygon, blocked_polygons):
                continue
            item_center = (
                float(min_x + (box_width / 2.0)),
                float(min_y + (box_height / 2.0)),
            )
            effective_front_rotation = _effective_front_rotation(
                rotation=rotation,
                front_side=front_side,
            )
            front_side_world = _effective_front_side_contract(
                base_front=front_side,
                rotation=rotation,
            )
            front_world_vec = side_to_vec(front_side_world)
            if not _candidate_satisfies_reintroduce_contract(
                room_polygon=room_polygon,
                polygon=polygon,
                effective_front_rotation=effective_front_rotation,
                item_center=item_center,
                cluster_bbox=cluster_bbox,
                cluster_target=cluster_target,
                box_width=box_width,
                box_height=box_height,
            ):
                continue
            object_id = _next_reintroduced_object_id(existing_ids, object_type)
            candidate = {
                "object_id": object_id,
                "instance_id": object_id,
                "object_type": object_type,
                "cluster_id": cluster_id,
                "rotation_ccw": int(rotation % 360),
                "bbox": {
                    "min_x": int(min_x),
                    "min_y": int(min_y),
                    "max_x": int(min_x + box_width),
                    "max_y": int(min_y + box_height),
                },
                "polygon_ccw": [
                    {"x": int(min_x), "y": int(min_y)},
                    {"x": int(min_x + box_width), "y": int(min_y)},
                    {"x": int(min_x + box_width), "y": int(min_y + box_height)},
                    {"x": int(min_x), "y": int(min_y + box_height)},
                ],
                "source_rect": {
                    "x": 0,
                    "y": 0,
                    "w": int(source_width),
                    "h": int(source_height),
                },
                "front_world": (
                    None
                    if front_world_vec is None
                    else {"dx": int(front_world_vec[0]), "dy": int(front_world_vec[1])}
                ),
                "front_side_world": front_side_world,
            }
            preference = _candidate_preference(
                room_polygon=room_polygon,
                polygon=polygon,
                effective_front_rotation=effective_front_rotation,
                item_center=item_center,
                cluster_center=cluster_center,
                target_center=cluster_target,
            )
            if best_preference is None or preference > best_preference:
                best_preference = preference
                best_candidate = candidate
    return best_candidate


def _object_dims_mm(item: dict[str, Any]) -> tuple[int, int] | None:
    rep_dims = item.get("rep_dims_m")
    if not isinstance(rep_dims, dict):
        return None
    try:
        length_mm = int(round(float(rep_dims.get("L") or 0.0) * 1000.0))
        width_mm = int(round(float(rep_dims.get("W") or 0.0) * 1000.0))
    except Exception:
        return None
    if length_mm <= 0 or width_mm <= 0:
        return None
    return max(length_mm, width_mm), min(length_mm, width_mm)


def _allowed_rotations(cluster_meta: dict[str, Any], object_type: str) -> list[int]:
    cluster_rules = (
        cluster_meta.get("cluster_rules")
        if isinstance(cluster_meta.get("cluster_rules"), dict)
        else {}
    )
    allowed = (
        cluster_rules.get("allowed_rotations")
        if isinstance(cluster_rules, dict)
        else {}
    )
    rotations = allowed.get(object_type) if isinstance(allowed, dict) else None
    clean_rotations = sorted(
        {
            int(rotation) % 360
            for rotation in (rotations or [0, 90, 180, 270])
            if isinstance(rotation, int | float)
        }
    )
    return clean_rotations or [0, 90, 180, 270]


def _cluster_bbox(
    objects: list[dict[str, Any]], cluster_id: str
) -> dict[str, int] | None:
    relevant = [
        row.get("bbox")
        for row in objects
        if isinstance(row, dict)
        and str(row.get("cluster_id") or "").strip() == cluster_id
        and isinstance(row.get("bbox"), dict)
    ]
    if not relevant:
        return None
    return {
        "min_x": min(int(bbox.get("min_x") or 0) for bbox in relevant),
        "min_y": min(int(bbox.get("min_y") or 0) for bbox in relevant),
        "max_x": max(int(bbox.get("max_x") or 0) for bbox in relevant),
        "max_y": max(int(bbox.get("max_y") or 0) for bbox in relevant),
    }


def _free_regions(
    room_polygon: Any, blocked_polygons: list[Any]
) -> list[dict[str, Any]]:
    free_geometry = (
        room_polygon
        if not blocked_polygons
        else room_polygon.difference(unary_union(blocked_polygons))
    )
    geometries = [free_geometry]
    if hasattr(free_geometry, "geoms"):
        geometries = [
            geometry
            for geometry in free_geometry.geoms
            if float(getattr(geometry, "area", 0.0)) > 1.0
        ]

    regions: list[dict[str, Any]] = []
    for geometry in sorted(
        geometries,
        key=lambda row: float(getattr(row, "area", 0.0)),
        reverse=True,
    ):
        min_x, min_y, max_x, max_y = geometry.bounds
        regions.append(
            {
                "area": float(getattr(geometry, "area", 0.0)),
                "bbox": {
                    "min_x": int(round(min_x)),
                    "min_y": int(round(min_y)),
                    "max_x": int(round(max_x)),
                    "max_y": int(round(max_y)),
                },
                "center": {
                    "x": int(round(float(geometry.centroid.x))),
                    "y": int(round(float(geometry.centroid.y))),
                },
            }
        )
    return regions


def _candidate_origins(
    *,
    cluster_bbox: dict[str, int] | None,
    free_regions: list[dict[str, Any]],
    box_width: int,
    box_height: int,
    grid_mm: int,
) -> list[tuple[int, int]]:
    gap = max(50, int(grid_mm) * 2)
    candidates: list[tuple[int, int]] = []
    if isinstance(cluster_bbox, dict):
        center_x = (cluster_bbox["min_x"] + cluster_bbox["max_x"] - box_width) // 2
        center_y = (cluster_bbox["min_y"] + cluster_bbox["max_y"] - box_height) // 2
        candidates.extend(
            [
                (cluster_bbox["min_x"] - box_width - gap, center_y),
                (cluster_bbox["max_x"] + gap, center_y),
                (center_x, cluster_bbox["min_y"] - box_height - gap),
                (center_x, cluster_bbox["max_y"] + gap),
            ]
        )

    for region in free_regions:
        bbox = region.get("bbox")
        center = region.get("center")
        if not isinstance(bbox, dict) or not isinstance(center, dict):
            continue
        candidates.extend(
            [
                (int(bbox["min_x"]) + gap, int(bbox["min_y"]) + gap),
                (int(bbox["max_x"]) - box_width - gap, int(bbox["min_y"]) + gap),
                (int(bbox["min_x"]) + gap, int(bbox["max_y"]) - box_height - gap),
                (
                    int(center["x"]) - box_width // 2,
                    int(bbox["min_y"]) + gap,
                ),
                (
                    int(center["x"]) - box_width // 2,
                    int(bbox["max_y"]) - box_height - gap,
                ),
                (
                    int(bbox["min_x"]) + gap,
                    int(center["y"]) - box_height // 2,
                ),
                (
                    int(bbox["max_x"]) - box_width - gap,
                    int(center["y"]) - box_height // 2,
                ),
                (
                    int(center["x"]) - box_width // 2,
                    int(center["y"]) - box_height // 2,
                ),
            ]
        )
        if isinstance(cluster_bbox, dict):
            cluster_center_x = (
                cluster_bbox["min_x"] + cluster_bbox["max_x"] - box_width
            ) // 2
            cluster_center_y = (
                cluster_bbox["min_y"] + cluster_bbox["max_y"] - box_height
            ) // 2
            candidates.extend(
                [
                    (cluster_center_x, int(bbox["min_y"]) + gap),
                    (cluster_center_x, int(bbox["max_y"]) - box_height - gap),
                    (int(bbox["min_x"]) + gap, cluster_center_y),
                    (int(bbox["max_x"]) - box_width - gap, cluster_center_y),
                ]
            )

    seen: set[tuple[int, int]] = set()
    snapped: list[tuple[int, int]] = []
    for x_pos, y_pos in candidates:
        origin = (_snap_to_grid(x_pos, grid_mm), _snap_to_grid(y_pos, grid_mm))
        if origin in seen:
            continue
        seen.add(origin)
        snapped.append(origin)
    return snapped


def _room_center(room_polygon: Any) -> tuple[float, float] | None:
    try:
        centroid = room_polygon.centroid
        return (float(centroid.x), float(centroid.y))
    except Exception:
        return None


def _rotation_alignment_to_target(
    *,
    effective_front_rotation: int,
    item_center: tuple[float, float],
    target_center: tuple[float, float] | None,
) -> float:
    if target_center is None:
        return 0.0
    dx = float(target_center[0] - item_center[0])
    dy = float(target_center[1] - item_center[1])
    length = hypot(dx, dy)
    if length <= 1e-6:
        return 1.0
    front_x, front_y = _rotation_front_vector(effective_front_rotation)
    return ((front_x * dx) + (front_y * dy)) / length


def _rotation_front_vector(rotation: int) -> tuple[float, float]:
    angle_rad = radians(int(rotation) % 360)
    return (-sin(angle_rad), -cos(angle_rad))


def _rotation_back_vector(rotation: int) -> tuple[float, float]:
    front_x, front_y = _rotation_front_vector(rotation)
    return (-front_x, -front_y)


def _candidate_preference(
    *,
    room_polygon: Any,
    polygon: Any,
    effective_front_rotation: int,
    item_center: tuple[float, float],
    cluster_center: tuple[float, float] | None,
    target_center: tuple[float, float] | None,
) -> tuple[float, ...]:
    back_wall_distance = _back_edge_wall_distance(
        room_polygon=room_polygon,
        polygon=polygon,
        rotation=effective_front_rotation,
    )
    wall_distance = _polygon_wall_distance(
        room_polygon=room_polygon,
        polygon=polygon,
    )
    wall_backed = 1.0 if back_wall_distance <= _WALL_BACK_THRESHOLD_MM else 0.0
    perimeter_hugging = 1.0 if wall_distance <= _PERIMETER_THRESHOLD_MM else 0.0
    cluster_proximity = -_distance_to_point(item_center, cluster_center)
    center_avoidance = _distance_to_point(item_center, target_center)
    front_alignment = _rotation_alignment_to_target(
        effective_front_rotation=effective_front_rotation,
        item_center=item_center,
        target_center=target_center,
    )
    return (
        wall_backed,
        perimeter_hugging,
        -back_wall_distance,
        cluster_proximity,
        front_alignment,
        center_avoidance,
    )


def _cluster_center(cluster_bbox: dict[str, int] | None) -> tuple[float, float] | None:
    if not isinstance(cluster_bbox, dict):
        return None
    return (
        float(cluster_bbox["min_x"] + cluster_bbox["max_x"]) / 2.0,
        float(cluster_bbox["min_y"] + cluster_bbox["max_y"]) / 2.0,
    )


def _cluster_front_side(cluster_meta: dict[str, Any], object_type: str) -> str:
    cluster_rules = (
        cluster_meta.get("cluster_rules")
        if isinstance(cluster_meta.get("cluster_rules"), dict)
        else {}
    )
    facing = cluster_rules.get("facing") if isinstance(cluster_rules, dict) else None
    rule = facing.get(object_type) if isinstance(facing, dict) else None
    front_side = rule.get("front") if isinstance(rule, dict) else None
    return normalize_cardinal_side(front_side, default="top") or "top"


def _effective_front_rotation(*, rotation: int, front_side: str) -> int:
    return _effective_front_rotation_contract(rotation=rotation, front_side=front_side)


def _candidate_satisfies_reintroduce_contract(
    *,
    room_polygon: Any,
    polygon: Any,
    effective_front_rotation: int,
    item_center: tuple[float, float],
    cluster_bbox: dict[str, int] | None,
    cluster_target: tuple[float, float] | None,
    box_width: int,
    box_height: int,
) -> bool:
    back_wall_distance = _back_edge_wall_distance(
        room_polygon=room_polygon,
        polygon=polygon,
        rotation=effective_front_rotation,
    )
    if back_wall_distance > _WALL_BACK_THRESHOLD_MM:
        return False

    wall_distance = _polygon_wall_distance(room_polygon=room_polygon, polygon=polygon)
    if wall_distance > _PERIMETER_THRESHOLD_MM:
        return False

    if cluster_target is not None:
        front_alignment = _rotation_alignment_to_target(
            effective_front_rotation=effective_front_rotation,
            item_center=item_center,
            target_center=cluster_target,
        )
        if front_alignment < _MIN_CLUSTER_ALIGNMENT_DOT:
            return False

    if cluster_bbox is None:
        return True

    cluster_distance = _distance_to_point(item_center, _cluster_center(cluster_bbox))
    max_cluster_distance = _max_cluster_distance_for_reintroduced_item(
        cluster_bbox=cluster_bbox,
        box_width=box_width,
        box_height=box_height,
    )
    return cluster_distance <= max_cluster_distance


def _max_cluster_distance_for_reintroduced_item(
    *,
    cluster_bbox: dict[str, int],
    box_width: int,
    box_height: int,
) -> float:
    cluster_width = max(0, int(cluster_bbox["max_x"]) - int(cluster_bbox["min_x"]))
    cluster_height = max(0, int(cluster_bbox["max_y"]) - int(cluster_bbox["min_y"]))
    cluster_diag = hypot(float(cluster_width), float(cluster_height))
    item_diag = hypot(float(box_width), float(box_height))
    return min(2400.0, max(900.0, cluster_diag + (0.85 * item_diag) + 250.0))


def _distance_to_point(
    point: tuple[float, float],
    target: tuple[float, float] | None,
) -> float:
    if target is None:
        return 0.0
    return hypot(point[0] - target[0], point[1] - target[1])


def _polygon_wall_distance(*, room_polygon: Any, polygon: Any) -> float:
    try:
        return float(room_polygon.boundary.distance(polygon.boundary))
    except Exception:
        return float("inf")


def _back_edge_wall_distance(
    *,
    room_polygon: Any,
    polygon: Any,
    rotation: int,
) -> float:
    if LineString is None:
        return float("inf")
    min_x, min_y, max_x, max_y = polygon.bounds
    back_edge = _back_edge_for_rotation(
        rotation=rotation,
        min_x=float(min_x),
        min_y=float(min_y),
        max_x=float(max_x),
        max_y=float(max_y),
    )
    if back_edge is None:
        return float("inf")
    try:
        return float(room_polygon.boundary.distance(back_edge))
    except Exception:
        return float("inf")


def _back_edge_for_rotation(
    *,
    rotation: int,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> Any | None:
    normalized = int(rotation) % 360
    if normalized == 0:
        points = [(min_x, max_y), (max_x, max_y)]
    elif normalized == 90:
        points = [(max_x, min_y), (max_x, max_y)]
    elif normalized == 180:
        points = [(min_x, min_y), (max_x, min_y)]
    elif normalized == 270:
        points = [(min_x, min_y), (min_x, max_y)]
    else:
        back_x, back_y = _rotation_back_vector(normalized)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        half_width = (max_x - min_x) / 2.0
        half_height = (max_y - min_y) / 2.0
        extent = abs(back_x) * half_width + abs(back_y) * half_height
        points = [
            (center_x + (back_x * extent), center_y + (back_y * extent)),
            (center_x + (back_x * extent), center_y + (back_y * extent)),
        ]
    try:
        return LineString(points)
    except Exception:
        return None


def _overlaps_blockers(polygon: Any, blockers: list[Any]) -> bool:
    for blocker in blockers:
        try:
            if float(polygon.intersection(blocker).area) > 1.0:
                return True
        except Exception:
            continue
    return False


def _next_reintroduced_object_id(existing_ids: set[str], object_type: str) -> str:
    base = f"{object_type}__reintroduced"
    index = 1
    while f"{base}_{index}" in existing_ids:
        index += 1
    return f"{base}_{index}"


def _snap_to_grid(value: int, grid_mm: int) -> int:
    step = max(1, int(grid_mm))
    return int(round(value / step) * step)


def _reintroduction_summary(
    original_inventory: dict[str, list[dict[str, Any]]],
    inserted_by_cluster: dict[str, list[dict[str, Any]]],
    remaining_by_cluster: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    inserted_count = sum(len(items) for items in inserted_by_cluster.values())
    remaining_clean = {
        cluster_id: [deepcopy(item) for item in items if isinstance(item, dict)]
        for cluster_id, items in remaining_by_cluster.items()
        if isinstance(items, list) and items
    }
    total_count = sum(len(items) for items in original_inventory.values())
    return {
        "status": "OK",
        "total_dropped_items": total_count,
        "inserted_count": inserted_count,
        "remaining_count": sum(len(items) for items in remaining_clean.values()),
        "reintroduced_objects_by_cluster": deepcopy(inserted_by_cluster),
        "remaining_dropped_objects_by_cluster": remaining_clean,
    }
