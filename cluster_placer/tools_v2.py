from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from shapely.geometry import LineString, Polygon
except Exception as e:  # pragma: no cover
    raise RuntimeError("phase2_placer_tools_v3 requires shapely") from e


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def _fix_geom(geom: Any) -> Any:
    try:
        if hasattr(geom, "is_valid") and not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        pass
    return geom


def rotate_point_ccw_90s(x: float, y: float, rot: int) -> tuple[float, float]:
    r = int(rot) % 360
    if r == 0:
        return x, y
    if r == 90:
        return -y, x
    if r == 180:
        return -x, -y
    if r == 270:
        return y, -x
    raise ValueError(f"Unsupported rot={rot}")


def rotate_vec_ccw_90s(
    vec: tuple[float, float] | None, rot: int
) -> tuple[float, float] | None:
    if vec is None:
        return None
    return rotate_point_ccw_90s(float(vec[0]), float(vec[1]), int(rot))


def normalize_vec(vec: tuple[float, float] | None) -> tuple[float, float] | None:
    if vec is None:
        return None
    x, y = float(vec[0]), float(vec[1])
    norm = (x * x + y * y) ** 0.5
    if norm <= 1e-9:
        return None
    return (x / norm, y / norm)


def parse_vec2(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        dx = value.get("dx")
        dy = value.get("dy")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        dx, dy = value[0], value[1]
    else:
        return None
    try:
        return normalize_vec((float(dx), float(dy)))
    except Exception:
        return None


def mirror_vec(
    vec: tuple[float, float] | None, axis: str
) -> tuple[float, float] | None:
    v = normalize_vec(vec)
    if v is None:
        return None
    axis = str(axis).lower()
    if axis == "x":
        return (-v[0], v[1])
    if axis == "y":
        return (v[0], -v[1])
    return v


def transform_rect_world(
    rect: dict[str, Any], tx: int, ty: int, rot: int
) -> dict[str, Any]:
    x = float(rect.get("x", 0))
    y = float(rect.get("y", 0))
    w = float(rect.get("w", 0))
    h = float(rect.get("h", 0))
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    world = [rotate_point_ccw_90s(px, py, rot) for px, py in corners]
    world = [(px + tx, py + ty) for px, py in world]
    xs = [p[0] for p in world]
    ys = [p[1] for p in world]
    return {
        "polygon_ccw": [{"x": int(round(px)), "y": int(round(py))} for px, py in world],
        "bbox": {
            "min_x": int(round(min(xs))),
            "min_y": int(round(min(ys))),
            "max_x": int(round(max(xs))),
            "max_y": int(round(max(ys))),
        },
        "world_center": {
            "x": int(round(sum(xs) / 4.0)),
            "y": int(round(sum(ys) / 4.0)),
        },
    }


def _polygon_from_points(points: List[Dict[str, Any]]) -> Any:
    return _fix_geom(Polygon([(float(p["x"]), float(p["y"])) for p in points]))


def _bbox_poly(bbox: Dict[str, Any]) -> Any:
    return _fix_geom(
        Polygon(
            [
                (float(bbox["min_x"]), float(bbox["min_y"])),
                (float(bbox["max_x"]), float(bbox["min_y"])),
                (float(bbox["max_x"]), float(bbox["max_y"])),
                (float(bbox["min_x"]), float(bbox["max_y"])),
            ]
        )
    )


# -----------------------------------------------------------------------------
# Payload helpers
# -----------------------------------------------------------------------------
def _cluster_cards_as_map(cluster_cards: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(cluster_cards, dict):
        return {
            str(k): deepcopy(v)
            for k, v in cluster_cards.items()
            if isinstance(k, str) and isinstance(v, dict)
        }
    if isinstance(cluster_cards, list):
        out: Dict[str, Dict[str, Any]] = {}
        for row in cluster_cards:
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
                out[row["cluster_id"]] = deepcopy(row)
        return out
    return {}


def _seed_transform_map(seed_layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in seed_layout.get("cluster_transforms") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _seed_variant_map(seed_layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in seed_layout.get("selected_variants") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _room_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(((payload.get("room_context") or {}).get("room_model_used") or {}))


def _room_polygon(payload: Dict[str, Any]) -> Any:
    room_model = _room_model(payload)
    pts = ((room_model.get("room") or {}).get("polygon_ccw")) or []
    return _polygon_from_points(pts)


def _hard_obstacles(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for ob in _room_model(payload).get("obstacles") or []:
        if not isinstance(ob, dict) or not ob.get("hard", True):
            continue
        pts = ob.get("polygon_ccw") or []
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        out.append(
            {
                "id": str(ob.get("id") or "obstacle"),
                "type": str(ob.get("type") or "unknown"),
                "poly": _polygon_from_points(pts),
            }
        )
    return out


def _openings(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    openings = _room_model(payload).get("openings") or {}
    out = {"doors": [], "windows": []}
    for key in ("doors", "windows"):
        for item in openings.get(key) or []:
            if not isinstance(item, dict):
                continue
            seg = item.get("segment_mm") or []
            if not isinstance(seg, list) or len(seg) != 2:
                continue
            p1 = (float(seg[0]["x"]), float(seg[0]["y"]))
            p2 = (float(seg[1]["x"]), float(seg[1]["y"]))
            out[key].append(
                {
                    "id": str(item.get("id") or ""),
                    "midpoint": ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0),
                    "line": LineString([p1, p2]),
                }
            )
    return out


def _nearest_midpoint(
    point: Tuple[float, float], items: List[Dict[str, Any]]
) -> Tuple[float, float] | None:
    best = None
    best_d = None
    px, py = point
    for item in items:
        mid = item.get("midpoint")
        if not isinstance(mid, tuple):
            continue
        d = math.hypot(mid[0] - px, mid[1] - py)
        if best_d is None or d < best_d:
            best = mid
            best_d = d
    return best


def _vec_from_to(
    a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[float, float] | None:
    return normalize_vec((float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def _dot(a: Tuple[float, float] | None, b: Tuple[float, float] | None) -> float | None:
    va = normalize_vec(a)
    vb = normalize_vec(b)
    if va is None or vb is None:
        return None
    return max(-1.0, min(1.0, va[0] * vb[0] + va[1] * vb[1]))


# -----------------------------------------------------------------------------
# Unified object state materialization
# -----------------------------------------------------------------------------
def _cluster_local_bbox(card: Dict[str, Any]) -> Dict[str, int]:
    fp = card.get("cluster_footprint") or {}
    bbox = fp.get("local_bbox") or {}
    try:
        min_x = int(round(float(bbox.get("min_x", 0))))
        min_y = int(round(float(bbox.get("min_y", 0))))
        max_x = int(round(float(bbox.get("max_x", 0))))
        max_y = int(round(float(bbox.get("max_y", 0))))
    except Exception:
        min_x = min_y = max_x = max_y = 0
    if max_x <= min_x or max_y <= min_y:
        xs: List[int] = []
        ys: List[int] = []
        for rect in fp.get("rects") or []:
            if not isinstance(rect, dict):
                continue
            x = int(round(float(rect.get("x", 0))))
            y = int(round(float(rect.get("y", 0))))
            w = int(round(float(rect.get("w", 0))))
            h = int(round(float(rect.get("h", 0))))
            xs.extend([x, x + w])
            ys.extend([y, y + h])
        if xs and ys:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
    return {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}


def _rotate_rect_about_center(rect: Dict[str, Any], rot: int) -> Dict[str, Any]:
    rot = int(rot) % 360
    x = float(rect.get("x", 0))
    y = float(rect.get("y", 0))
    w = float(rect.get("w", 0))
    h = float(rect.get("h", 0))
    if rot in {0, 180}:
        return {
            **deepcopy(rect),
            "x": int(round(x)),
            "y": int(round(y)),
            "w": int(round(w)),
            "h": int(round(h)),
        }
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw, nh = h, w
    return {
        **deepcopy(rect),
        "x": int(round(cx - nw / 2.0)),
        "y": int(round(cy - nh / 2.0)),
        "w": int(round(nw)),
        "h": int(round(nh)),
    }


def _mirror_rect_in_bbox(
    rect: Dict[str, Any], bbox: Dict[str, int], axis: str
) -> Dict[str, Any]:
    x = int(round(float(rect.get("x", 0))))
    y = int(round(float(rect.get("y", 0))))
    w = int(round(float(rect.get("w", 0))))
    h = int(round(float(rect.get("h", 0))))
    out = deepcopy(rect)
    axis = str(axis).lower()
    if axis == "x":
        out["x"] = int(bbox["max_x"] - (x + w) + bbox["min_x"])
        out["y"] = y
    else:
        out["x"] = x
        out["y"] = int(bbox["max_y"] - (y + h) + bbox["min_y"])
    out["w"] = w
    out["h"] = h
    return out


def _current_cluster_variant_card(
    payload: Dict[str, Any], cluster_id: str, variant_id: str | None
) -> Dict[str, Any]:
    base = _cluster_cards_as_map(payload.get("cluster_cards") or {}).get(cluster_id)
    if not isinstance(base, dict):
        return {}
    if not variant_id:
        return deepcopy(base)
    variants = base.get("available_variants") or base.get("variants") or []
    for item in variants:
        if (
            isinstance(item, dict)
            and item.get("variant_id") == variant_id
            and isinstance(item.get("card"), dict)
        ):
            out = deepcopy(item["card"])
            out.setdefault("cluster_id", cluster_id)
            return out
    return deepcopy(base)


def materialize_phase2_state(
    payload: Dict[str, Any], repair: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    repair = deepcopy(repair or {})
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)
    vmap = _seed_variant_map(seed_layout)
    for row in repair.get("cluster_transforms") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            tmap[row["cluster_id"]] = deepcopy(row)
    for row in repair.get("selected_variants") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            vmap[row["cluster_id"]] = deepcopy(row)

    repairs_by_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for rep in repair.get("object_repairs") or []:
        if isinstance(rep, dict) and isinstance(rep.get("cluster_id"), str):
            repairs_by_cluster.setdefault(rep["cluster_id"], []).append(deepcopy(rep))

    room_poly = _room_polygon(payload)
    room_center = (float(room_poly.centroid.x), float(room_poly.centroid.y))
    obstacles = _hard_obstacles(payload)

    clusters: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    object_overlap_pairs: List[Dict[str, Any]] = []
    hard_violations: List[Dict[str, Any]] = []

    for cid in sorted(tmap.keys()):
        tf = tmap[cid]
        variant_id = None
        if cid in vmap:
            variant_id = str(vmap[cid].get("variant_id") or "") or None
        card = _current_cluster_variant_card(payload, cid, variant_id)
        if not card:
            hard_violations.append({"code": "MISSING_CLUSTER_CARD", "cluster_id": cid})
            continue
        bbox = _cluster_local_bbox(card)
        rects = []
        for rect in (card.get("cluster_footprint") or {}).get("rects") or []:
            if isinstance(rect, dict) and isinstance(rect.get("id"), str):
                rects.append(deepcopy(rect))
        rect_map = {r["id"]: deepcopy(r) for r in rects}
        important = deepcopy(
            (card.get("orientation_meta") or {}).get("important_objects") or {}
        )
        cluster_front_local = parse_vec2(
            (card.get("orientation_meta") or {}).get("cluster_front_local")
        )
        cluster_axis_local = parse_vec2(
            (card.get("orientation_meta") or {}).get("cluster_axis_local")
        )
        cluster_meta = {"anchor_override": None}

        for rep in repairs_by_cluster.get(cid, []):
            oid = rep.get("object_id")
            op = rep.get("op")
            params = rep.get("params") or {}
            if op == "set_anchor":
                cluster_meta["anchor_override"] = params.get("anchor")
                continue
            if not isinstance(oid, str) or oid not in rect_map:
                continue
            rect = rect_map[oid]
            obj_meta = (
                important.get(oid)
                if isinstance(important, dict) and isinstance(important.get(oid), dict)
                else {}
            )
            front_local = (
                parse_vec2((obj_meta or {}).get("front_local")) or cluster_front_local
            )
            axis_local = parse_vec2((obj_meta or {}).get("axis_local"))

            if op == "rotate_object":
                rr = int(params.get("rot", 0)) % 360
                rect = _rotate_rect_about_center(rect, rr)
                front_local = rotate_vec_ccw_90s(front_local, rr)
                axis_local = rotate_vec_ccw_90s(axis_local, rr)
            elif op == "mirror_object":
                axis = str(params.get("axis") or "x").lower()
                rect = _mirror_rect_in_bbox(rect, bbox, axis)
                front_local = mirror_vec(front_local, axis)
                axis_local = mirror_vec(axis_local, axis)
            elif op == "nudge_object":
                rect = {
                    **rect,
                    "x": int(
                        round(float(rect.get("x", 0)) + float(params.get("dx", 0)))
                    ),
                    "y": int(
                        round(float(rect.get("y", 0)) + float(params.get("dy", 0)))
                    ),
                }
            elif op == "swap_objects":
                other_id = str(params.get("other_object_id") or "")
                other = rect_map.get(other_id)
                if isinstance(other, dict):
                    ox, oy = other.get("x"), other.get("y")
                    other["x"], other["y"] = rect.get("x"), rect.get("y")
                    rect["x"], rect["y"] = ox, oy
                    rect_map[other_id] = other
            elif op == "set_front_override":
                front_local = normalize_vec(
                    (float(params.get("dx", 0.0)), float(params.get("dy", 0.0)))
                )

            rect_map[oid] = rect
            if oid not in important:
                important[oid] = {}
            if front_local is not None:
                important[oid]["front_local"] = {
                    "dx": round(front_local[0], 3),
                    "dy": round(front_local[1], 3),
                }
            if axis_local is not None:
                important[oid]["axis_local"] = {
                    "dx": round(axis_local[0], 3),
                    "dy": round(axis_local[1], 3),
                }

        tx = int(tf.get("x") or 0)
        ty = int(tf.get("y") or 0)
        crot = int(tf.get("rot") or 0) % 360

        cluster_object_indices: List[int] = []
        for oid, rect in rect_map.items():
            geom = transform_rect_world(rect, tx, ty, crot)
            poly = _bbox_poly(geom["bbox"])
            obj_meta = (
                important.get(oid) if isinstance(important.get(oid), dict) else {}
            )
            front_local = (
                parse_vec2((obj_meta or {}).get("front_local")) or cluster_front_local
            )
            axis_local = parse_vec2((obj_meta or {}).get("axis_local"))
            front_world = normalize_vec(rotate_vec_ccw_90s(front_local, crot))
            axis_world = normalize_vec(rotate_vec_ccw_90s(axis_local, crot))
            w = int(round(float(rect.get("w", 0))))
            h = int(round(float(rect.get("h", 0))))
            if front_local is None:
                size_along_access = max(w, h)
            else:
                dx, dy = abs(front_local[0]), abs(front_local[1])
                size_along_access = w if dx >= dy else h
            required_clearance = int(round(0.25 * size_along_access))
            row = {
                "cluster_id": cid,
                "variant_id": variant_id,
                "object_id": oid,
                "rotation_ccw": crot,
                "local_rect": {
                    "x": int(round(float(rect.get("x", 0)))),
                    "y": int(round(float(rect.get("y", 0)))),
                    "w": w,
                    "h": h,
                },
                "polygon_ccw": geom["polygon_ccw"],
                "bbox": geom["bbox"],
                "world_center": geom["world_center"],
                "poly": poly,
                "front_world": None
                if front_world is None
                else {"dx": round(front_world[0], 3), "dy": round(front_world[1], 3)},
                "axis_world": None
                if axis_world is None
                else {"dx": round(axis_world[0], 3), "dy": round(axis_world[1], 3)},
                "required_clearance_mm": required_clearance,
            }
            cluster_object_indices.append(len(objects))
            objects.append(row)

            if not room_poly.buffer(1e-6).covers(poly):
                hard_violations.append(
                    {
                        "code": "OBJECT_OUT_OF_BOUNDS",
                        "cluster_id": cid,
                        "object_id": oid,
                    }
                )
            for ob in obstacles:
                if poly.intersects(ob["poly"]):
                    hard_violations.append(
                        {
                            "code": "OBJECT_HITS_OBSTACLE",
                            "cluster_id": cid,
                            "object_id": oid,
                            "obstacle_id": ob["id"],
                        }
                    )

        # cluster bbox from member objects
        if cluster_object_indices:
            xs1 = [objects[i]["bbox"]["min_x"] for i in cluster_object_indices]
            ys1 = [objects[i]["bbox"]["min_y"] for i in cluster_object_indices]
            xs2 = [objects[i]["bbox"]["max_x"] for i in cluster_object_indices]
            ys2 = [objects[i]["bbox"]["max_y"] for i in cluster_object_indices]
            cluster_bbox = {
                "min_x": min(xs1),
                "min_y": min(ys1),
                "max_x": max(xs2),
                "max_y": max(ys2),
            }
            centroid = (
                (cluster_bbox["min_x"] + cluster_bbox["max_x"]) / 2.0,
                (cluster_bbox["min_y"] + cluster_bbox["max_y"]) / 2.0,
            )
            clusters.append(
                {
                    "cluster_id": cid,
                    "variant_id": variant_id,
                    "bbox": cluster_bbox,
                    "world_center": {
                        "x": int(round(centroid[0])),
                        "y": int(round(centroid[1])),
                    },
                    "front_world": None
                    if cluster_front_local is None
                    else {
                        "dx": round(
                            (
                                rotate_vec_ccw_90s(cluster_front_local, crot)
                                or (0.0, 0.0)
                            )[0],
                            3,
                        ),
                        "dy": round(
                            (
                                rotate_vec_ccw_90s(cluster_front_local, crot)
                                or (0.0, 0.0)
                            )[1],
                            3,
                        ),
                    },
                    "axis_world": None
                    if cluster_axis_local is None
                    else {
                        "dx": round(
                            (
                                rotate_vec_ccw_90s(cluster_axis_local, crot)
                                or (0.0, 0.0)
                            )[0],
                            3,
                        ),
                        "dy": round(
                            (
                                rotate_vec_ccw_90s(cluster_axis_local, crot)
                                or (0.0, 0.0)
                            )[1],
                            3,
                        ),
                    },
                    "anchor_override": cluster_meta.get("anchor_override"),
                }
            )

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]
            b = objects[j]
            inter = float(a["poly"].intersection(b["poly"]).area)
            if inter > 1e-6:
                object_overlap_pairs.append(
                    {
                        "a": {
                            "cluster_id": a["cluster_id"],
                            "object_id": a["object_id"],
                        },
                        "b": {
                            "cluster_id": b["cluster_id"],
                            "object_id": b["object_id"],
                        },
                        "intersection_area_mm2": round(inter, 2),
                    }
                )
                hard_violations.append(
                    {
                        "code": "OBJECT_OVERLAP",
                        "a_cluster_id": a["cluster_id"],
                        "a_object_id": a["object_id"],
                        "b_cluster_id": b["cluster_id"],
                        "b_object_id": b["object_id"],
                        "intersection_area_mm2": round(inter, 2),
                    }
                )

    return {
        "room_polygon": room_poly,
        "room_center": room_center,
        "obstacles": obstacles,
        "clusters": clusters,
        "objects": objects,
        "hard_violations": hard_violations,
        "object_overlap_pairs": object_overlap_pairs,
        "hard_valid": len(hard_violations) == 0,
    }


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def _project_distance_along_ray(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    point: Tuple[float, float],
) -> float | None:
    v = normalize_vec(direction)
    if v is None:
        return None
    ox, oy = float(origin[0]), float(origin[1])
    px, py = float(point[0]), float(point[1])
    t = ((px - ox) * v[0]) + ((py - oy) * v[1])
    if t <= 1e-6:
        return None
    return t


def _collect_intersection_points(geom: Any, out: List[Tuple[float, float]]) -> None:
    if geom is None:
        return
    try:
        if geom.is_empty:
            return
    except Exception:
        return
    gtype = getattr(geom, "geom_type", "")
    if gtype == "Point":
        out.append((float(geom.x), float(geom.y)))
        return
    if gtype == "MultiPoint":
        for sub in geom.geoms:
            _collect_intersection_points(sub, out)
        return
    if gtype in {"LineString", "LinearRing"}:
        coords = list(geom.coords)
        if coords:
            out.append((float(coords[0][0]), float(coords[0][1])))
            out.append((float(coords[-1][0]), float(coords[-1][1])))
        return
    if gtype == "MultiLineString":
        for sub in geom.geoms:
            _collect_intersection_points(sub, out)
        return
    if gtype == "GeometryCollection":
        for sub in geom.geoms:
            _collect_intersection_points(sub, out)
        return
    if gtype == "Polygon":
        coords = list(geom.exterior.coords)
        if coords:
            out.append((float(coords[0][0]), float(coords[0][1])))
            out.append((float(coords[-1][0]), float(coords[-1][1])))
        return
    if gtype == "MultiPolygon":
        for sub in geom.geoms:
            _collect_intersection_points(sub, out)


def _directional_clearance_mm(
    point: Tuple[float, float],
    direction: Tuple[float, float] | None,
    room_poly: Any,
    blockers: List[Any],
    max_distance: float,
) -> float:
    v = normalize_vec(direction)
    if v is None:
        return 0.0
    start = (float(point[0]) + v[0] * 1e-3, float(point[1]) + v[1] * 1e-3)
    end = (float(point[0]) + v[0] * max_distance, float(point[1]) + v[1] * max_distance)
    ray = LineString([start, end])
    best = max_distance

    pts: List[Tuple[float, float]] = []
    _collect_intersection_points(ray.intersection(room_poly.boundary), pts)
    for pt in pts:
        dist = _project_distance_along_ray(point, v, pt)
        if dist is not None and dist < best:
            best = dist

    for geom in blockers:
        try:
            inter = ray.intersection(geom.boundary)
        except Exception:
            try:
                inter = ray.intersection(geom)
            except Exception:
                continue
        pts2: List[Tuple[float, float]] = []
        _collect_intersection_points(inter, pts2)
        for pt in pts2:
            dist = _project_distance_along_ray(point, v, pt)
            if dist is not None and dist < best:
                best = dist

    return max(0.0, float(best))


def _candidate_open_dirs(
    point: Tuple[float, float], room_center: Tuple[float, float]
) -> List[Tuple[float, float]]:
    base = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ]
    to_center = _vec_from_to(point, room_center)
    if to_center is not None:
        base = [to_center, (-to_center[0], -to_center[1])] + base
    out: List[Tuple[float, float]] = []
    seen = set()
    for d in base:
        v = normalize_vec(d)
        if v is None:
            continue
        key = (round(v[0], 3), round(v[1], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _best_open_dir(
    point: Tuple[float, float],
    room_center: Tuple[float, float],
    room_poly: Any,
    blockers: List[Any],
) -> Tuple[Tuple[float, float] | None, float]:
    max_distance = max(1800.0, room_poly.length)
    best_dir = None
    best_clear = -1.0
    for d in _candidate_open_dirs(point, room_center):
        clear = _directional_clearance_mm(point, d, room_poly, blockers, max_distance)
        if clear > best_clear:
            best_dir = d
            best_clear = clear
    return best_dir, max(0.0, best_clear)


def _scoring_object_blockers(state: Dict[str, Any], current_index: int) -> List[Any]:
    blockers = [ob["poly"] for ob in state.get("obstacles") or []]
    for i, row in enumerate(state.get("objects") or []):
        if i == current_index:
            continue
        blockers.append(row["poly"])
    return blockers


def score_phase2_state(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, Any]:
    room_poly = state["room_polygon"]
    room_center = state["room_center"]
    openings = _openings(payload)
    goals = payload.get("goals") or {}
    relation_plan = goals.get("relation_plan_used") or {}
    repair_debug = payload.get("repair_debug") or {}

    orientation_debug: List[Dict[str, Any]] = []
    cluster_penalty: Dict[str, float] = {}
    object_penalty: Dict[Tuple[str, str], float] = {}

    # object-level generic access/open-space score
    for idx, row in enumerate(state.get("objects") or []):
        cid = row["cluster_id"]
        oid = row["object_id"]
        center = (float(row["world_center"]["x"]), float(row["world_center"]["y"]))
        front_world = parse_vec2(row.get("front_world"))
        blockers = _scoring_object_blockers(state, idx)
        max_distance = max(1800.0, room_poly.length)
        front_clear = _directional_clearance_mm(
            center, front_world, room_poly, blockers, max_distance
        )
        best_dir, best_clear = _best_open_dir(center, room_center, room_poly, blockers)
        required = float(row.get("required_clearance_mm") or 0.0)
        dot = _dot(front_world, best_dir)
        shortage = max(0.0, required - front_clear)
        misalign = max(0.0, 1.0 - (dot if dot is not None else -1.0))
        pen = shortage * 2.2 + misalign * 600.0
        object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + pen
        cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + pen
        orientation_debug.append(
            {
                "kind": "object_orientation",
                "cluster_id": cid,
                "object_id": oid,
                "intent": "front_to_open_space",
                "penalty_mm": int(round(pen)),
                "front_clear_mm": int(round(front_clear)),
                "best_clear_mm": int(round(best_clear)),
                "dot": None if dot is None else round(dot, 3),
            }
        )

    # relation-plan specific object intents
    for item in relation_plan.get("object_orientations") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        oid = item.get("object_id")
        intents = item.get("intents") or []
        if not isinstance(cid, str) or not isinstance(oid, str):
            continue
        target = None
        for row in state.get("objects") or []:
            if row["cluster_id"] == cid and row["object_id"] == oid:
                target = row
                break
        if target is None:
            continue
        center = (
            float(target["world_center"]["x"]),
            float(target["world_center"]["y"]),
        )
        front_world = parse_vec2(target.get("front_world"))
        for intent in intents:
            intent = str(intent).lower()
            pen = 0.0
            dot = None
            if intent == "face_window":
                mid = _nearest_midpoint(center, openings.get("windows") or [])
                if mid is not None:
                    dot = _dot(front_world, _vec_from_to(center, mid))
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 500.0
            elif intent == "face_entry":
                mid = _nearest_midpoint(center, openings.get("doors") or [])
                if mid is not None:
                    dot = _dot(front_world, _vec_from_to(center, mid))
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 520.0
            elif intent == "preserve_front_access":
                front_clear = next(
                    (
                        d["front_clear_mm"]
                        for d in orientation_debug
                        if d.get("cluster_id") == cid
                        and d.get("object_id") == oid
                        and d.get("intent") == "front_to_open_space"
                    ),
                    0,
                )
                req = int(target.get("required_clearance_mm") or 0)
                pen = max(0.0, req - front_clear) * 2.0
            if pen > 0:
                object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + pen
                cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + pen
                orientation_debug.append(
                    {
                        "kind": "object_orientation",
                        "cluster_id": cid,
                        "object_id": oid,
                        "intent": intent,
                        "penalty_mm": int(round(pen)),
                        "dot": None if dot is None else round(dot, 3),
                    }
                )

    # cluster-level intents
    for item in relation_plan.get("cluster_orientations") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        intents = item.get("intents") or []
        if not isinstance(cid, str):
            continue
        cluster = next(
            (x for x in state.get("clusters") or [] if x["cluster_id"] == cid), None
        )
        if cluster is None:
            continue
        center = (
            float(cluster["world_center"]["x"]),
            float(cluster["world_center"]["y"]),
        )
        front = parse_vec2(cluster.get("front_world"))
        for intent in intents:
            intent = str(intent).lower()
            pen = 0.0
            dot = None
            if intent == "access_to_open_space":
                blockers = [ob["poly"] for ob in state.get("obstacles") or []]
                for row in state.get("objects") or []:
                    if row["cluster_id"] != cid:
                        blockers.append(row["poly"])
                best_dir, _ = _best_open_dir(center, room_center, room_poly, blockers)
                dot = _dot(front, best_dir)
                pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 700.0
            elif intent == "face_window":
                mid = _nearest_midpoint(center, openings.get("windows") or [])
                if mid is not None:
                    dot = _dot(front, _vec_from_to(center, mid))
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 640.0
            elif intent == "back_to_wall":
                if front is not None:
                    target_vec = _vec_from_to(room_center, center)
                    dot = _dot(front, target_vec)
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 760.0
            if pen > 0:
                cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + pen
                orientation_debug.append(
                    {
                        "kind": "cluster_orientation",
                        "cluster_id": cid,
                        "intent": intent,
                        "penalty_mm": int(round(pen)),
                        "dot": None if dot is None else round(dot, 3),
                    }
                )

    # amplify seeded failing objects/clusters a bit to keep focus aligned with solver
    seed_verify = repair_debug.get("seed_verify") or {}
    quality = seed_verify.get("quality") or {}
    for item in (quality.get("critical_orientation_debug") or [])[:20]:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        oid = item.get("object_id")
        pen = float(item.get("penalty_mm") or 0.0) * 0.10
        if isinstance(cid, str):
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + pen
        if isinstance(cid, str) and isinstance(oid, str):
            object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + pen

    hard_penalty = 1_000_000.0 * len(state.get("hard_violations") or [])
    soft_penalty = sum(cluster_penalty.values())
    total_penalty = hard_penalty + soft_penalty

    prioritized_clusters = [
        {"cluster_id": cid, "score": round(score, 2)}
        for cid, score in sorted(cluster_penalty.items(), key=lambda kv: -kv[1])
    ]
    prioritized_objects = [
        {"cluster_id": cid, "object_id": oid, "score": round(score, 2)}
        for (cid, oid), score in sorted(object_penalty.items(), key=lambda kv: -kv[1])
    ]

    return {
        "hard_valid": bool(state.get("hard_valid")),
        "hard_violation_count": len(state.get("hard_violations") or []),
        "hard_violations": deepcopy(state.get("hard_violations") or []),
        "orientation_debug": sorted(
            orientation_debug, key=lambda x: -int(x.get("penalty_mm") or 0)
        ),
        "prioritized_clusters": prioritized_clusters[:8],
        "prioritized_objects": prioritized_objects[:16],
        "penalties": {
            "hard_penalty": int(round(hard_penalty)),
            "soft_penalty": int(round(soft_penalty)),
            "total_penalty": int(round(total_penalty)),
        },
        "score": -int(round(total_penalty)),
    }


# -----------------------------------------------------------------------------
# Public tool API
# -----------------------------------------------------------------------------
def DiagnosePhase2Seed(*, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = materialize_phase2_state(payload, repair=None)
    score = score_phase2_state(payload, state)
    return {
        "result": "OK",
        "hard_valid": score["hard_valid"],
        "score": score["score"],
        "penalties": score["penalties"],
        "prioritized_clusters": score["prioritized_clusters"],
        "prioritized_objects": score["prioritized_objects"],
        "hard_violations": score["hard_violations"],
        "orientation_debug": score["orientation_debug"][:16],
    }


def EnumeratePhase2RepairMoves(
    *, payload: Dict[str, Any], limit: int = 24
) -> Dict[str, Any]:
    state = materialize_phase2_state(payload, repair=None)
    score = score_phase2_state(payload, state)
    cluster_cards = _cluster_cards_as_map(payload.get("cluster_cards") or {})
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)

    moves: List[Dict[str, Any]] = []
    # Cluster-level first for the worst clusters
    for row in score["prioritized_clusters"][:3]:
        cid = row["cluster_id"]
        tf = tmap.get(cid) or {"cluster_id": cid, "x": 0, "y": 0, "rot": 0}
        for rot in (0, 90, 180, 270):
            if rot == int(tf.get("rot") or 0):
                continue
            moves.append(
                {
                    "kind": "cluster_pose",
                    "cluster_id": cid,
                    "proposal": {
                        "cluster_transforms": [{**deepcopy(tf), "rot": rot}],
                        "selected_variants": [],
                        "object_repairs": [],
                    },
                    "reason": f"Try cluster rotation for failing cluster {cid}",
                }
            )

    # Object-level for top objects
    by_cluster_objects: Dict[str, List[str]] = {}
    for obj in payload.get("objects_world") or []:
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("cluster_id"), str)
            and isinstance(obj.get("object_id"), str)
        ):
            by_cluster_objects.setdefault(obj["cluster_id"], []).append(
                obj["object_id"]
            )

    for row in score["prioritized_objects"][:6]:
        cid = row["cluster_id"]
        oid = row["object_id"]
        for rot in (90, 180, 270):
            moves.append(
                {
                    "kind": "object_pose",
                    "cluster_id": cid,
                    "object_id": oid,
                    "proposal": {
                        "cluster_transforms": [],
                        "selected_variants": [],
                        "object_repairs": [
                            {
                                "cluster_id": cid,
                                "object_id": oid,
                                "op": "rotate_object",
                                "params": {"rot": rot},
                            }
                        ],
                    },
                    "reason": f"Try rotating {cid}.{oid}",
                }
            )
        for dx, dy in ((50, 0), (-50, 0), (0, 50), (0, -50)):
            moves.append(
                {
                    "kind": "object_pose",
                    "cluster_id": cid,
                    "object_id": oid,
                    "proposal": {
                        "cluster_transforms": [],
                        "selected_variants": [],
                        "object_repairs": [
                            {
                                "cluster_id": cid,
                                "object_id": oid,
                                "op": "nudge_object",
                                "params": {"dx": dx, "dy": dy},
                            }
                        ],
                    },
                    "reason": f"Try nudging {cid}.{oid}",
                }
            )
        peers = [x for x in by_cluster_objects.get(cid, []) if x != oid]
        for peer in peers[:2]:
            moves.append(
                {
                    "kind": "object_pose",
                    "cluster_id": cid,
                    "object_id": oid,
                    "proposal": {
                        "cluster_transforms": [],
                        "selected_variants": [],
                        "object_repairs": [
                            {
                                "cluster_id": cid,
                                "object_id": oid,
                                "op": "swap_objects",
                                "params": {"other_object_id": peer},
                            }
                        ],
                    },
                    "reason": f"Try swapping {cid}.{oid} with {peer}",
                }
            )

    return {"result": "OK", "moves": moves[:limit], "baseline_score": score["score"]}


def ScorePhase2Repair(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    baseline_state = materialize_phase2_state(payload, repair=None)
    baseline_score = score_phase2_state(payload, baseline_state)
    cand_state = materialize_phase2_state(payload, repair=repair)
    cand_score = score_phase2_state(payload, cand_state)
    return {
        "result": "OK",
        "baseline": {
            "score": baseline_score["score"],
            "hard_valid": baseline_score["hard_valid"],
        },
        "candidate": {
            "score": cand_score["score"],
            "hard_valid": cand_score["hard_valid"],
            "hard_violations": cand_score["hard_violations"],
            "prioritized_clusters": cand_score["prioritized_clusters"],
            "prioritized_objects": cand_score["prioritized_objects"],
            "orientation_debug": cand_score["orientation_debug"][:16],
        },
        "delta_score": int(cand_score["score"] - baseline_score["score"]),
        "improved": bool(
            cand_score["hard_valid"] and cand_score["score"] > baseline_score["score"]
        ),
    }


def PreviewPhase2Repair(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    state = materialize_phase2_state(payload, repair=repair)
    score = score_phase2_state(payload, state)
    room_model = _room_model(payload)
    room = deepcopy(room_model.get("room") or {})
    if isinstance(room_model.get("openings"), dict):
        room["openings"] = deepcopy(room_model.get("openings") or {})
    if isinstance(room_model.get("obstacles"), list):
        room["obstacles"] = deepcopy(room_model.get("obstacles") or [])
    objects = []
    for row in state.get("objects") or []:
        objects.append(
            {
                "object_id": row["object_id"],
                "cluster_id": row["cluster_id"],
                "polygon_ccw": deepcopy(row["polygon_ccw"]),
                "bbox": deepcopy(row["bbox"]),
                "rotation_ccw": int(row.get("rotation_ccw") or 0),
                "source_rect": deepcopy(row.get("local_rect") or {}),
            }
        )
    return {
        "result": "OK",
        "compiled": {
            "status": "OK" if score["hard_valid"] else "INVALID",
            "room": room,
            "objects": objects,
            "notes": deepcopy(repair.get("notes") or []),
            "missing": deepcopy(score.get("hard_violations") or []),
        },
        "score": score,
    }


TOOL_REGISTRY: Dict[str, Any] = {
    "DiagnosePhase2Seed": DiagnosePhase2Seed,
    "EnumeratePhase2RepairMoves": EnumeratePhase2RepairMoves,
    "ScorePhase2Repair": ScorePhase2Repair,
    "PreviewPhase2Repair": PreviewPhase2Repair,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "DiagnosePhase2Seed",
            "description": "Diagnose the current phase-2 seed using the unified object-pose scorer.",
            "parameters": {
                "type": "object",
                "properties": {"payload": {"type": "object"}},
                "required": ["payload"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "EnumeratePhase2RepairMoves",
            "description": "Enumerate small candidate repair moves for the current phase-2 seed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "limit": {"type": "integer"},
                },
                "required": ["payload"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ScorePhase2Repair",
            "description": "Score a repair proposal using the same unified state as preview and final compile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PreviewPhase2Repair",
            "description": "Preview the compiled final output for a repair proposal using the unified state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
]
