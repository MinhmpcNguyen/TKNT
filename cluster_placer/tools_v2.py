from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import nearest_points, unary_union
except Exception as e:  # pragma: no cover
    raise RuntimeError("phase2_placer_tools_v3 requires shapely") from e

from cluster_composer.constraint_score import (
    build_object_specs_from_cluster,
    score_cluster_constraints,
)
from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM, normalize_layout_grid_mm


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


def _resolved_seed_layout(
    payload: Dict[str, Any], repair: Dict[str, Any] | None = None
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)
    vmap = _seed_variant_map(seed_layout)

    for row in (repair or {}).get("cluster_transforms") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            tmap[row["cluster_id"]] = deepcopy(row)

    for row in (repair or {}).get("selected_variants") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            vmap[row["cluster_id"]] = deepcopy(row)

    cluster_ids = sorted(tmap.keys())
    return (
        [deepcopy(tmap[cid]) for cid in cluster_ids],
        [deepcopy(vmap[cid]) for cid in cluster_ids if cid in vmap],
    )


def _full_seed_repair(
    payload: Dict[str, Any],
    repair: Dict[str, Any] | None = None,
    *,
    status: str = "REPAIRED",
    notes: list[str] | None = None,
) -> Dict[str, Any]:
    cluster_transforms, selected_variants = _resolved_seed_layout(payload, repair)
    merged_notes: list[str] = []
    for source in (((repair or {}).get("notes") or []), notes or []):
        for item in source:
            text = str(item).strip()
            if text and text not in merged_notes:
                merged_notes.append(text)
    return {
        "status": status,
        "cluster_transforms": cluster_transforms,
        "selected_variants": selected_variants,
        "object_repairs": deepcopy((repair or {}).get("object_repairs") or []),
        "notes": merged_notes,
    }


def _room_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(((payload.get("room_context") or {}).get("room_model_used") or {}))


def _room_notes(payload: Dict[str, Any]) -> List[str]:
    notes = _room_model(payload).get("notes") or []
    out: List[str] = []
    for item in notes:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _repair_phase(payload: Dict[str, Any]) -> str:
    phase_control = payload.get("phase_control") or {}
    value = str(phase_control.get("repair_phase") or "").strip().lower()
    if value == "object_refine":
        return "object_refine"
    return "macro_layout"


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
        local_placement_map: Dict[str, Dict[str, Any]] = {}
        for row in card.get("local_placements") or []:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                local_placement_map[row["id"]] = {
                    "id": row["id"],
                    "x": int(round(float(row.get("x", 0)))),
                    "y": int(round(float(row.get("y", 0)))),
                    "rot": int(round(float(row.get("rot", 0)))) % 360,
                }
        for oid, rect in rect_map.items():
            local_placement_map.setdefault(
                oid,
                {
                    "id": oid,
                    "x": int(round(float(rect.get("x", 0)))),
                    "y": int(round(float(rect.get("y", 0)))),
                    "rot": 0,
                },
            )
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
            placement = deepcopy(
                local_placement_map.get(oid)
                or {
                    "id": oid,
                    "x": int(round(float(rect.get("x", 0)))),
                    "y": int(round(float(rect.get("y", 0)))),
                    "rot": 0,
                }
            )
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
                placement["x"] = int(round(float(rect.get("x", 0))))
                placement["y"] = int(round(float(rect.get("y", 0))))
                placement["rot"] = (int(placement.get("rot") or 0) + rr) % 360
                front_local = rotate_vec_ccw_90s(front_local, rr)
                axis_local = rotate_vec_ccw_90s(axis_local, rr)
            elif op == "mirror_object":
                axis = str(params.get("axis") or "x").lower()
                rect = _mirror_rect_in_bbox(rect, bbox, axis)
                placement["x"] = int(round(float(rect.get("x", 0))))
                placement["y"] = int(round(float(rect.get("y", 0))))
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
                placement["x"] = int(round(float(rect.get("x", 0))))
                placement["y"] = int(round(float(rect.get("y", 0))))
            elif op == "swap_objects":
                other_id = str(params.get("other_object_id") or "")
                other = rect_map.get(other_id)
                if isinstance(other, dict):
                    ox, oy = other.get("x"), other.get("y")
                    other["x"], other["y"] = rect.get("x"), rect.get("y")
                    rect["x"], rect["y"] = ox, oy
                    rect_map[other_id] = other
                    other_placement = deepcopy(
                        local_placement_map.get(other_id)
                        or {
                            "id": other_id,
                            "x": int(round(float(other.get("x", 0)))),
                            "y": int(round(float(other.get("y", 0)))),
                            "rot": 0,
                        }
                    )
                    px = int(other_placement.get("x") or 0)
                    py = int(other_placement.get("y") or 0)
                    other_placement["x"], other_placement["y"] = (
                        int(placement.get("x") or 0),
                        int(placement.get("y") or 0),
                    )
                    placement["x"], placement["y"] = px, py
                    local_placement_map[other_id] = other_placement
            elif op == "set_front_override":
                front_local = normalize_vec(
                    (float(params.get("dx", 0.0)), float(params.get("dy", 0.0)))
                )

            rect_map[oid] = rect
            local_placement_map[oid] = placement
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
            cluster_poly = _fix_geom(
                unary_union([objects[i]["poly"] for i in cluster_object_indices])
            )
            if cluster_poly is None or getattr(cluster_poly, "is_empty", True):
                cluster_poly = _bbox_poly(
                    {
                        "min_x": min(xs1),
                        "min_y": min(ys1),
                        "max_x": max(xs2),
                        "max_y": max(ys2),
                    }
                )
            cluster_bbox = {
                "min_x": min(xs1),
                "min_y": min(ys1),
                "max_x": max(xs2),
                "max_y": max(ys2),
            }
            centroid = (float(cluster_poly.centroid.x), float(cluster_poly.centroid.y))
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
                    "poly": cluster_poly,
                    "area_mm2": float(cluster_poly.area),
                    "local_placements": sorted(
                        [
                            deepcopy(local_placement_map[oid])
                            for oid in local_placement_map
                            if isinstance(oid, str)
                        ],
                        key=lambda item: str(item.get("id") or ""),
                    ),
                    "hard_constraints": deepcopy(card.get("hard_constraints") or []),
                    "soft_constraints": deepcopy(card.get("soft_constraints") or []),
                    "cluster_rules": deepcopy(card.get("cluster_rules") or {}),
                    "decisions": deepcopy(card.get("decisions") or []),
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


def _distance_between_points(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _distance_to_room_center(
    center: Tuple[float, float], room_center: Tuple[float, float]
) -> int:
    return int(round(_distance_between_points(center, room_center)))


def _distance_to_nearest_opening(
    center: Tuple[float, float], items: List[Dict[str, Any]]
) -> int | None:
    midpoint = _nearest_midpoint(center, items)
    if midpoint is None:
        return None
    return int(round(_distance_between_points(center, midpoint)))


def _door_window_metrics(
    center: Tuple[float, float], openings: Dict[str, List[Dict[str, Any]]]
) -> tuple[int | None, int | None]:
    door_distance = _distance_to_nearest_opening(center, openings.get("doors") or [])
    window_distance = _distance_to_nearest_opening(
        center, openings.get("windows") or []
    )
    return door_distance, window_distance


def _distance_poly_to_nearest_opening(
    poly: Any, items: List[Dict[str, Any]]
) -> float | None:
    best: float | None = None
    for item in items:
        line = item.get("line")
        if line is None:
            continue
        try:
            distance = float(poly.distance(line))
        except Exception:
            continue
        if best is None or distance < best:
            best = distance
    return best


def _nearest_opening_line_direction(
    point: Tuple[float, float], items: List[Dict[str, Any]]
) -> tuple[float, float] | None:
    best_dir = None
    best_distance = None
    px, py = point
    for item in items:
        line = item.get("line")
        midpoint = item.get("midpoint")
        if line is None or not isinstance(midpoint, tuple):
            continue
        coords = list(getattr(line, "coords", []))
        if len(coords) < 2:
            continue
        direction = normalize_vec(
            (
                float(coords[-1][0]) - float(coords[0][0]),
                float(coords[-1][1]) - float(coords[0][1]),
            )
        )
        if direction is None:
            continue
        distance = math.hypot(float(midpoint[0]) - px, float(midpoint[1]) - py)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_dir = direction
    return best_dir


def _nearest_boundary_point(
    point: Tuple[float, float], room_poly: Any
) -> tuple[float, float] | None:
    try:
        boundary_point, _ = nearest_points(room_poly.boundary, Point(point))
    except Exception:
        return None
    return (float(boundary_point.x), float(boundary_point.y))


def _nearest_wall_inward_dir(
    point: Tuple[float, float], room_poly: Any
) -> tuple[float, float] | None:
    boundary_point = _nearest_boundary_point(point, room_poly)
    if boundary_point is None:
        return None
    return _vec_from_to(boundary_point, point)


def _room_wall_segments(room_poly: Any) -> list[dict[str, Any]]:
    try:
        coords = list(room_poly.exterior.coords)
    except Exception:
        return []
    segments: list[dict[str, Any]] = []
    for idx in range(max(0, len(coords) - 1)):
        p1 = coords[idx]
        p2 = coords[idx + 1]
        line = LineString([p1, p2])
        segments.append(
            {
                "line": line,
                "length": float(line.length),
                "direction": normalize_vec(
                    (
                        float(p2[0]) - float(p1[0]),
                        float(p2[1]) - float(p1[1]),
                    )
                ),
            }
        )
    return segments


def _longest_wall_segment(room_poly: Any) -> dict[str, Any] | None:
    segments = _room_wall_segments(room_poly)
    if not segments:
        return None
    return max(segments, key=lambda item: float(item.get("length") or 0.0))


def _distance_to_longest_wall(
    point: Tuple[float, float], room_poly: Any
) -> float | None:
    longest = _longest_wall_segment(room_poly)
    if not longest:
        return None
    try:
        return float(longest["line"].distance(Point(point)))
    except Exception:
        return None


def _window_clearance_hits(state: Dict[str, Any], cluster_id: str | None = None) -> int:
    window_clearance_polys = [
        ob["poly"]
        for ob in state.get("obstacles") or []
        if str(ob.get("type") or "").lower() == "window_clearance"
    ]
    if not window_clearance_polys:
        return 0

    hits = 0
    for row in state.get("objects") or []:
        if cluster_id is not None and row.get("cluster_id") != cluster_id:
            continue
        poly = row.get("poly")
        if poly is None:
            continue
        for clearance in window_clearance_polys:
            if poly.intersects(clearance):
                hits += 1
                break
    return hits


def _classify_cluster_zone(
    center: Tuple[float, float],
    room_center: Tuple[float, float],
    openings: Dict[str, List[Dict[str, Any]]],
) -> str:
    door_distance, window_distance = _door_window_metrics(center, openings)
    center_distance = _distance_to_room_center(center, room_center)
    if window_distance is not None and window_distance + 200 < min(
        door_distance or 10**9, center_distance
    ):
        return "window_side"
    if door_distance is not None and door_distance + 200 < min(
        window_distance or 10**9, center_distance
    ):
        return "entry_side"
    return "interior"


def _cluster_intents_by_id(payload: Dict[str, Any]) -> Dict[str, set[str]]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    out: Dict[str, set[str]] = {}

    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str):
            continue
        slot = out.setdefault(cid, set())
        for intent in row.get("intents") or []:
            text = str(intent).strip().lower()
            if text:
                slot.add(text)

    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str):
            continue
        slot = out.setdefault(cid, set())
        for intent in row.get("intents") or []:
            text = str(intent).strip().lower()
            if text:
                slot.add(text)

    return out


def _cluster_preference_profile(
    payload: Dict[str, Any],
) -> Dict[str, Dict[str, set[str]]]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    out: Dict[str, Dict[str, set[str]]] = {}

    def _slot(cluster_id: str) -> Dict[str, set[str]]:
        return out.setdefault(
            cluster_id,
            {
                "prefer": set(),
                "avoid": set(),
                "intents": set(),
            },
        )

    for row in relation_plan.get("cluster_affinities") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if not isinstance(cluster_id, str):
            continue
        slot = _slot(cluster_id)
        for tag in row.get("prefer") or []:
            text = str(tag).strip().lower()
            if text:
                slot["prefer"].add(text)
        for tag in row.get("avoid") or []:
            text = str(tag).strip().lower()
            if text:
                slot["avoid"].add(text)

    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if not isinstance(cluster_id, str):
            continue
        slot = _slot(cluster_id)
        for intent in row.get("intents") or []:
            text = str(intent).strip().lower()
            if text:
                slot["intents"].add(text)

    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if not isinstance(cluster_id, str):
            continue
        slot = _slot(cluster_id)
        for intent in row.get("intents") or []:
            text = str(intent).strip().lower()
            if text:
                slot["intents"].add(text)

    return out


def _preferred_zone_for_cluster(intents: set[str]) -> str | None:
    lowered = {str(item).lower() for item in intents}
    if "face_window" in lowered:
        return "window_side"
    if "face_entry" in lowered:
        return "entry_side"
    if lowered & {"back_to_wall", "access_to_open_space"}:
        return "edge_side"
    return None


def _opening_bands(
    payload: Dict[str, Any],
    openings: Dict[str, List[Dict[str, Any]]],
    room_poly: Any,
) -> List[Dict[str, Any]]:
    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    out: List[Dict[str, Any]] = []

    for kind, depth in (
        ("doors", max(500, grid_mm * 8)),
        ("windows", max(300, grid_mm * 5)),
    ):
        opening_type = "door" if kind == "doors" else "window"
        for item in openings.get(kind) or []:
            line = item.get("line")
            if line is None:
                continue
            band = _fix_geom(room_poly.intersection(line.buffer(depth, cap_style=2)))
            try:
                if band.is_empty:
                    continue
            except Exception:
                continue
            out.append(
                {
                    "id": str(item.get("id") or ""),
                    "type": opening_type,
                    "depth_mm": int(depth),
                    "poly": band,
                    "area_mm2": float(band.area),
                }
            )
    return out


def _main_path_corridors(
    payload: Dict[str, Any],
    openings: Dict[str, List[Dict[str, Any]]],
    room_center: Tuple[float, float],
    room_poly: Any,
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    corridors: List[Dict[str, Any]] = []
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    cluster_by_id = {
        str(row.get("cluster_id") or ""): row for row in state.get("clusters") or []
    }

    def _opening_by_id(opening_id: str) -> dict[str, Any] | None:
        for key in ("doors", "windows"):
            for item in openings.get(key) or []:
                if str(item.get("id") or "") == opening_id:
                    return item
        return None

    def _cluster_access_point(cluster: Dict[str, Any]) -> tuple[float, float]:
        center = (
            float((cluster.get("world_center") or {}).get("x") or room_center[0]),
            float((cluster.get("world_center") or {}).get("y") or room_center[1]),
        )
        bbox = cluster.get("bbox") or {}
        span_x = float(bbox.get("max_x", 0.0)) - float(bbox.get("min_x", 0.0))
        span_y = float(bbox.get("max_y", 0.0)) - float(bbox.get("min_y", 0.0))
        front = parse_vec2(cluster.get("front_world"))
        if front is None:
            return center
        offset_mm = max(450.0, (max(span_x, span_y) / 2.0) + 150.0)
        return (
            center[0] + front[0] * offset_mm,
            center[1] + front[1] * offset_mm,
        )

    def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for point in points:
            if not out or point != out[-1]:
                out.append(point)
        return out

    def _best_corridor_line(
        start: tuple[float, float], end: tuple[float, float]
    ) -> Any:
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        cx, cy = float(room_center[0]), float(room_center[1])

        candidates: list[list[tuple[float, float]]] = [
            [(sx, sy), (ex, ey)],
            [(sx, sy), (sx, ey), (ex, ey)],
            [(sx, sy), (ex, sy), (ex, ey)],
            [(sx, sy), (cx, sy), (cx, ey), (ex, ey)],
            [(sx, sy), (sx, cy), (ex, cy), (ex, ey)],
        ]

        best_line = LineString([start, end])
        best_key: tuple[float, int, float] | None = None
        room_cover = room_poly.buffer(1e-6)

        for raw_points in candidates:
            points = _dedupe_points(raw_points)
            if len(points) < 2:
                continue
            line = LineString(points)
            try:
                outside_length = float(line.difference(room_cover).length)
            except Exception:
                outside_length = float(line.length)
            bend_count = max(0, len(points) - 2)
            rank_key = (outside_length, bend_count, float(line.length))
            if best_key is None or rank_key < best_key:
                best_key = rank_key
                best_line = line
        return best_line

    for row in relation_plan.get("circulation_plan", {}).get("main_paths") or []:
        if not isinstance(row, dict):
            continue
        opening_id = str(row.get("from") or "").strip()
        target_cluster_id = str(row.get("to_cluster") or "").strip()
        opening = _opening_by_id(opening_id)
        cluster = cluster_by_id.get(target_cluster_id)
        midpoint = opening.get("midpoint") if isinstance(opening, dict) else None
        if not isinstance(midpoint, tuple):
            continue
        target_point = (
            _cluster_access_point(cluster) if isinstance(cluster, dict) else room_center
        )
        priority = str(row.get("priority") or "medium").strip().lower()
        corridor_width = {
            "high": max(850, grid_mm * 12),
            "medium": max(700, grid_mm * 10),
            "low": max(600, grid_mm * 8),
        }.get(priority, max(700, grid_mm * 10))
        line = _best_corridor_line(midpoint, target_point)
        corridor = _fix_geom(
            room_poly.intersection(line.buffer(corridor_width / 2.0, cap_style=2))
        )
        try:
            if corridor.is_empty:
                continue
        except Exception:
            continue
        corridors.append(
            {
                "id": f"{opening_id}->{target_cluster_id or 'room_center'}",
                "opening_id": opening_id,
                "target_cluster_id": target_cluster_id or None,
                "priority": priority,
                "desired_clearance_mm": int(corridor_width),
                "width_mm": int(corridor_width),
                "line": line,
                "poly": corridor,
                "area_mm2": float(corridor.area),
            }
        )

    if corridors:
        return corridors

    corridor_width = max(700, grid_mm * 10)
    for item in openings.get("doors") or []:
        midpoint = item.get("midpoint")
        if not isinstance(midpoint, tuple):
            continue
        line = _best_corridor_line(midpoint, room_center)
        corridor = _fix_geom(
            room_poly.intersection(line.buffer(corridor_width / 2.0, cap_style=2))
        )
        try:
            if corridor.is_empty:
                continue
        except Exception:
            continue
        corridors.append(
            {
                "id": str(item.get("id") or ""),
                "opening_id": str(item.get("id") or ""),
                "target_cluster_id": None,
                "priority": "medium",
                "desired_clearance_mm": int(corridor_width),
                "width_mm": int(corridor_width),
                "line": line,
                "poly": corridor,
                "area_mm2": float(corridor.area),
            }
        )
    return corridors


def _free_space_polygon(state: Dict[str, Any]) -> Any:
    room_poly = state["room_polygon"]
    blockers = [row["poly"] for row in state.get("objects") or []]
    blockers.extend(ob["poly"] for ob in state.get("obstacles") or [])
    if not blockers:
        return room_poly
    try:
        blocked = unary_union(blockers)
        return _fix_geom(room_poly.difference(blocked))
    except Exception:
        return room_poly


def _corridor_clearance_summary(
    state: Dict[str, Any], corridor: Dict[str, Any]
) -> Dict[str, Any]:
    free_space = _free_space_polygon(state)
    line = corridor.get("line")
    if line is None:
        return {
            "min_clearance_mm": 0,
            "blocked_samples": 0,
            "sample_count": 0,
            "clearance_shortage_mm": int(corridor.get("desired_clearance_mm") or 0),
        }

    try:
        length = float(line.length)
    except Exception:
        length = 0.0
    sample_count = max(5, min(13, int(length / 450.0) + 1))
    blocked_samples = 0
    min_clearance = float(corridor.get("desired_clearance_mm") or 0.0)
    free_space_cover = free_space.buffer(1e-6)

    for idx in range(sample_count):
        fraction = (idx + 1) / float(sample_count + 1)
        sample_point = line.interpolate(fraction, normalized=True)
        if not bool(free_space_cover.covers(sample_point)):
            clearance = 0.0
            blocked_samples += 1
        else:
            clearance = float(free_space.boundary.distance(sample_point))
        min_clearance = min(min_clearance, clearance)

    desired_clearance = float(corridor.get("desired_clearance_mm") or 0.0) / 2.0
    shortage = max(0.0, desired_clearance - min_clearance)
    return {
        "min_clearance_mm": int(round(min_clearance)),
        "blocked_samples": blocked_samples,
        "sample_count": sample_count,
        "clearance_shortage_mm": int(round(shortage)),
    }


def _central_zone(room_poly: Any) -> Any:
    min_x, min_y, max_x, max_y = room_poly.bounds
    max_span = max(max_x - min_x, max_y - min_y)
    radius = max(800.0, max_span * 0.18)
    center = (float(room_poly.centroid.x), float(room_poly.centroid.y))
    return _fix_geom(room_poly.intersection(Point(center).buffer(radius)))


def _cluster_poly(cluster: Dict[str, Any]) -> Any:
    poly = cluster.get("poly")
    if poly is not None:
        return poly
    return _bbox_poly(cluster.get("bbox") or {})


def _score_cluster_internal_fidelity(
    state: Dict[str, Any],
) -> tuple[Dict[str, float], Dict[Tuple[str, str], float], List[Dict[str, Any]]]:
    cluster_penalty: Dict[str, float] = {}
    object_penalty: Dict[Tuple[str, str], float] = {}
    debug_rows: List[Dict[str, Any]] = []

    for cluster in state.get("clusters") or []:
        cid = str(cluster.get("cluster_id") or "")
        local_placements = deepcopy(cluster.get("local_placements") or [])
        if not cid or not local_placements:
            continue

        cluster_payload = {
            "cluster_id": cid,
            "hard_constraints": deepcopy(cluster.get("hard_constraints") or []),
            "soft_constraints": deepcopy(cluster.get("soft_constraints") or []),
            "cluster_rules": deepcopy(cluster.get("cluster_rules") or {}),
            "decisions": deepcopy(cluster.get("decisions") or []),
        }
        cluster_rules = cluster_payload.get("cluster_rules") or {}
        grid_mm = normalize_layout_grid_mm(cluster_rules.get("grid_mm"))
        scoring = score_cluster_constraints(
            hard_constraints=cluster_payload["hard_constraints"],
            soft_constraints=cluster_payload["soft_constraints"],
            objects=build_object_specs_from_cluster(cluster_payload),
            local_placements=local_placements,
            grid_mm=grid_mm,
            cluster_rules=cluster_rules,
            use_clearance=True,
        )

        hard_summary = scoring.get("hard") or {}
        soft_summary = scoring.get("soft") or {}
        cluster_total = (
            float(hard_summary.get("total_weighted_violation") or 0.0)
            + float(soft_summary.get("total_weighted_penalty") or 0.0) * 0.6
        )
        if cluster_total > 0:
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + cluster_total

        for row in (scoring.get("all_evaluations") or [])[:20]:
            if not isinstance(row, dict):
                continue
            weighted = float(row.get("weighted_violation") or 0.0)
            if weighted <= 0:
                continue
            debug_rows.append(
                {
                    "kind": "cluster_internal_constraint",
                    "cluster_id": cid,
                    "constraint_type": str(row.get("type") or ""),
                    "penalty_mm": int(round(weighted)),
                    "hard": bool(row.get("hard")),
                    "subjects": deepcopy(row.get("subjects") or {}),
                }
            )
            subjects = row.get("subjects") or {}
            member_ids: List[str] = []
            for key in ("a", "b", "id"):
                value = subjects.get(key)
                if isinstance(value, str) and value:
                    member_ids.append(value)
            if not member_ids:
                continue
            share = weighted / float(len(member_ids))
            for oid in member_ids:
                object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + share

    debug_rows.sort(
        key=lambda row: (
            -int(row.get("penalty_mm") or 0),
            str(row.get("cluster_id") or ""),
            str(row.get("constraint_type") or ""),
        )
    )
    return cluster_penalty, object_penalty, debug_rows


def _score_global_layout_metrics(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    openings: Dict[str, List[Dict[str, Any]]],
    room_center: Tuple[float, float],
) -> Dict[str, Any]:
    room_poly = state["room_polygon"]
    room_area = max(float(room_poly.area), 1.0)
    room_bounds = room_poly.bounds
    max_span = max(room_bounds[2] - room_bounds[0], room_bounds[3] - room_bounds[1])
    cluster_preferences = _cluster_preference_profile(payload)
    opening_bands = _opening_bands(payload, openings, room_poly)
    path_corridors = _main_path_corridors(
        payload, openings, room_center, room_poly, state
    )
    central_zone = _central_zone(room_poly)
    central_zone_area = max(float(central_zone.area), 1.0)

    cluster_penalty: Dict[str, float] = {}
    preferred_zone_rows: List[Dict[str, Any]] = []
    opening_block_rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    path_object_rows: List[Dict[str, Any]] = []
    central_rows: List[Dict[str, Any]] = []
    edge_fit_rows: List[Dict[str, Any]] = []

    for cluster in state.get("clusters") or []:
        cid = str(cluster.get("cluster_id") or "")
        center = (
            float((cluster.get("world_center") or {}).get("x") or 0.0),
            float((cluster.get("world_center") or {}).get("y") or 0.0),
        )
        poly = _cluster_poly(cluster)
        area_mm2 = max(float(getattr(poly, "area", 0.0) or 0.0), 1.0)
        area_ratio = area_mm2 / room_area
        edge_distance = float(room_poly.boundary.distance(poly))
        actual_zone = _classify_cluster_zone(center, room_center, openings)
        profile = cluster_preferences.get(
            cid, {"prefer": set(), "avoid": set(), "intents": set()}
        )
        prefer_tags = set(profile.get("prefer") or set())
        avoid_tags = set(profile.get("avoid") or set())
        intents = set(profile.get("intents") or set())
        edge_pref = bool(
            prefer_tags & {"recess_or_edge", "long_wall"}
            or intents & {"back_to_wall", "access_to_open_space"}
        )
        window_pref = bool(
            "window_side" in prefer_tags
            or intents & {"face_window", "front_to_window", "axis_parallel_window"}
        )
        entry_pref = bool("entry_side" in prefer_tags or intents & {"face_entry"})

        affinity_rows: list[tuple[str, float, str]] = []
        if window_pref:
            distance = float(
                _distance_poly_to_nearest_opening(poly, openings.get("windows") or [])
                or max_span
            )
            affinity_rows.append(("window_side", distance, "window_side"))
        if entry_pref:
            distance = float(
                _distance_poly_to_nearest_opening(poly, openings.get("doors") or [])
                or max_span
            )
            affinity_rows.append(("entry_side", distance, "entry_side"))
        if "far_from_entry" in prefer_tags:
            distance = float(
                _distance_poly_to_nearest_opening(poly, openings.get("doors") or [])
                or 0.0
            )
            desired_clear = max(1500.0, max_span * 0.26)
            affinity_rows.append(
                (
                    "far_from_entry",
                    max(0.0, desired_clear - distance),
                    "far_from_entry",
                )
            )
        if "recess_or_edge" in prefer_tags:
            affinity_rows.append(("recess_or_edge", edge_distance, "edge_side"))
        if "long_wall" in prefer_tags:
            affinity_rows.append(
                (
                    "long_wall",
                    float(
                        _distance_to_longest_wall(center, room_poly) or edge_distance
                    ),
                    "edge_side",
                )
            )
        if not affinity_rows:
            fallback_zone = _preferred_zone_for_cluster(intents)
            if fallback_zone == "window_side":
                affinity_rows.append(
                    (
                        fallback_zone,
                        float(
                            _distance_poly_to_nearest_opening(
                                poly, openings.get("windows") or []
                            )
                            or max_span
                        ),
                        actual_zone,
                    )
                )
            elif fallback_zone == "entry_side":
                affinity_rows.append(
                    (
                        fallback_zone,
                        float(
                            _distance_poly_to_nearest_opening(
                                poly, openings.get("doors") or []
                            )
                            or max_span
                        ),
                        actual_zone,
                    )
                )
            elif fallback_zone == "edge_side":
                affinity_rows.append((fallback_zone, edge_distance, actual_zone))

        for preferred_zone, affinity_distance, actual_zone_label in affinity_rows:
            if preferred_zone == "window_side":
                zone_penalty = affinity_distance * 0.38
            elif preferred_zone == "entry_side":
                zone_penalty = affinity_distance * 0.32
            elif preferred_zone == "far_from_entry":
                zone_penalty = affinity_distance * 0.72
            elif preferred_zone == "long_wall":
                zone_penalty = affinity_distance * 0.44
            else:
                zone_penalty = affinity_distance * 0.42
            if zone_penalty <= 0:
                continue
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + zone_penalty
            preferred_zone_rows.append(
                {
                    "cluster_id": cid,
                    "preferred_zone": preferred_zone,
                    "actual_zone": actual_zone_label,
                    "distance_mm": int(round(affinity_distance)),
                    "penalty_mm": int(round(zone_penalty)),
                }
            )

        opening_penalty = 0.0
        for band in opening_bands:
            band_poly = band.get("poly")
            if band_poly is None:
                continue
            intersection_area = float(poly.intersection(band_poly).area)
            if intersection_area <= 1e-6:
                continue
            area_ratio_in_band = intersection_area / max(
                float(band.get("area_mm2") or 1.0), 1.0
            )
            weight = 1000.0 if band.get("type") == "door" else 640.0
            if band.get("type") == "window" and (
                "window_blocking" in avoid_tags or window_pref
            ):
                weight *= 1.7
            if band.get("type") == "door" and (
                avoid_tags & {"door_swing", "entry_blocking"}
                or "far_from_entry" in prefer_tags
            ):
                weight *= 1.8
            penalty = area_ratio_in_band * weight
            opening_penalty += penalty
            opening_block_rows.append(
                {
                    "cluster_id": cid,
                    "opening_id": band.get("id"),
                    "opening_type": band.get("type"),
                    "intersection_area_mm2": int(round(intersection_area)),
                    "penalty_mm": int(round(penalty)),
                }
            )
        if opening_penalty > 0:
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + opening_penalty

        central_area = float(poly.intersection(central_zone).area)
        if central_area > 1e-6:
            central_ratio = central_area / central_zone_area
            weight = 950.0
            if edge_pref or window_pref or "far_from_entry" in prefer_tags:
                weight = 1450.0
            penalty = central_ratio * weight
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + penalty
            central_rows.append(
                {
                    "cluster_id": cid,
                    "intersection_area_mm2": int(round(central_area)),
                    "central_ratio": round(central_ratio, 3),
                    "penalty_mm": int(round(penalty)),
                }
            )

        edge_weight = (
            2.8 if edge_pref or window_pref or "far_from_entry" in prefer_tags else 1.2
        )
        edge_fit_penalty = edge_distance * max(area_ratio, 0.08) * edge_weight
        if edge_fit_penalty > 0:
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + edge_fit_penalty
            edge_fit_rows.append(
                {
                    "cluster_id": cid,
                    "area_ratio": round(area_ratio, 4),
                    "edge_distance_mm": int(round(edge_distance)),
                    "penalty_mm": int(round(edge_fit_penalty)),
                }
            )

    for corridor in path_corridors:
        corridor_poly = corridor.get("poly")
        if corridor_poly is None:
            continue
        corridor_area = max(float(corridor.get("area_mm2") or 1.0), 1.0)
        corridor_summary = _corridor_clearance_summary(state, corridor)
        blocked_sample_ratio = float(
            corridor_summary.get("blocked_samples") or 0.0
        ) / max(float(corridor_summary.get("sample_count") or 1.0), 1.0)
        desired_half_clearance = max(
            float(corridor.get("desired_clearance_mm") or 0.0) / 2.0, 1.0
        )
        shortage_ratio = min(
            1.6,
            float(corridor_summary.get("clearance_shortage_mm") or 0.0)
            / desired_half_clearance,
        )
        priority = str(corridor.get("priority") or "medium").strip().lower()
        priority_weight = {"high": 1.3, "medium": 1.0, "low": 0.82}.get(priority, 1.0)

        cluster_blockers: List[Dict[str, Any]] = []
        total_cluster_blocked_area = 0.0
        for cluster in state.get("clusters") or []:
            blocked_area = float(
                _cluster_poly(cluster).intersection(corridor_poly).area
            )
            if blocked_area <= 1e-6:
                continue
            blocked_ratio = blocked_area / corridor_area
            total_cluster_blocked_area += blocked_area
            cluster_blockers.append(
                {
                    "cluster_id": str(cluster.get("cluster_id") or ""),
                    "blocked_area_mm2": blocked_area,
                    "blocked_ratio": blocked_ratio,
                }
            )

        object_blockers: List[Dict[str, Any]] = []
        total_object_blocked_area = 0.0
        for obj in state.get("objects") or []:
            blocked_area = float(obj["poly"].intersection(corridor_poly).area)
            if blocked_area <= 1e-6:
                continue
            blocked_ratio = blocked_area / corridor_area
            total_object_blocked_area += blocked_area
            object_blockers.append(
                {
                    "cluster_id": str(obj.get("cluster_id") or ""),
                    "object_id": str(obj.get("object_id") or ""),
                    "blocked_area_mm2": blocked_area,
                    "blocked_ratio": blocked_ratio,
                }
            )

        total_blocked_ratio = min(
            1.0,
            total_cluster_blocked_area / corridor_area if corridor_area > 0 else 0.0,
        )
        path_penalty_total = priority_weight * (
            (total_blocked_ratio * 1180.0)
            + (blocked_sample_ratio * 1560.0)
            + (shortage_ratio * 1420.0)
        )

        if (
            total_cluster_blocked_area <= 1e-6
            and total_object_blocked_area <= 1e-6
            and float(corridor_summary.get("clearance_shortage_mm") or 0.0) <= 0.0
            and float(corridor_summary.get("blocked_samples") or 0.0) <= 0.0
        ):
            continue

        for blocker in cluster_blockers:
            cluster_id = blocker.get("cluster_id")
            if not isinstance(cluster_id, str) or not cluster_id:
                continue
            if total_cluster_blocked_area > 0:
                share = path_penalty_total * (
                    float(blocker.get("blocked_area_mm2") or 0.0)
                    / total_cluster_blocked_area
                )
            else:
                share = path_penalty_total / max(float(len(cluster_blockers)), 1.0)
            cluster_penalty[cluster_id] = cluster_penalty.get(cluster_id, 0.0) + share

        if not cluster_blockers and path_penalty_total > 0.0:
            target_cluster_id = corridor.get("target_cluster_id")
            if isinstance(target_cluster_id, str) and target_cluster_id:
                cluster_penalty[target_cluster_id] = (
                    cluster_penalty.get(target_cluster_id, 0.0) + path_penalty_total
                )

        for blocker in object_blockers:
            if total_object_blocked_area > 0:
                share = path_penalty_total * (
                    float(blocker.get("blocked_area_mm2") or 0.0)
                    / total_object_blocked_area
                )
            else:
                share = path_penalty_total / max(float(len(object_blockers)), 1.0)
            path_object_rows.append(
                {
                    "path_id": corridor.get("id"),
                    "opening_id": corridor.get("opening_id"),
                    "target_cluster_id": corridor.get("target_cluster_id"),
                    "cluster_id": blocker.get("cluster_id"),
                    "object_id": blocker.get("object_id"),
                    "blocked_area_mm2": int(
                        round(float(blocker.get("blocked_area_mm2") or 0.0))
                    ),
                    "blocked_ratio": round(
                        float(blocker.get("blocked_ratio") or 0.0), 3
                    ),
                    "min_clearance_mm": int(
                        corridor_summary.get("min_clearance_mm") or 0
                    ),
                    "clearance_shortage_mm": int(
                        corridor_summary.get("clearance_shortage_mm") or 0
                    ),
                    "penalty_mm": int(round(share)),
                }
            )

        path_rows.append(
            {
                "path_id": corridor.get("id"),
                "opening_id": corridor.get("opening_id"),
                "target_cluster_id": corridor.get("target_cluster_id"),
                "priority": priority,
                "desired_clearance_mm": int(corridor.get("desired_clearance_mm") or 0),
                "min_clearance_mm": int(corridor_summary.get("min_clearance_mm") or 0),
                "clearance_shortage_mm": int(
                    corridor_summary.get("clearance_shortage_mm") or 0
                ),
                "blocked_samples": int(corridor_summary.get("blocked_samples") or 0),
                "sample_count": int(corridor_summary.get("sample_count") or 0),
                "blocked_ratio": round(total_blocked_ratio, 3),
                "blocking_clusters": [
                    blocker.get("cluster_id")
                    for blocker in sorted(
                        cluster_blockers,
                        key=lambda row: -float(row.get("blocked_area_mm2") or 0.0),
                    )[:4]
                    if isinstance(blocker.get("cluster_id"), str)
                ],
                "blocking_objects": [
                    f"{blocker.get('cluster_id')}.{blocker.get('object_id')}"
                    for blocker in sorted(
                        object_blockers,
                        key=lambda row: -float(row.get("blocked_area_mm2") or 0.0),
                    )[:4]
                    if isinstance(blocker.get("cluster_id"), str)
                    and isinstance(blocker.get("object_id"), str)
                ],
                "penalty_mm": int(round(path_penalty_total)),
            }
        )

    preferred_zone_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    opening_block_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    path_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    path_object_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    central_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    edge_fit_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))

    return {
        "cluster_penalty": cluster_penalty,
        "cluster_affinity_to_preferred_zone": preferred_zone_rows[:12],
        "opening_band_blocking": opening_block_rows[:16],
        "main_path_clearance": {
            "paths": path_rows[:16],
            "blocking_objects": path_object_rows[:20],
            "worst_blocked_ratio": max(
                [float(row.get("blocked_ratio") or 0.0) for row in path_rows] or [0.0]
            ),
            "worst_clearance_shortage_mm": max(
                [int(row.get("clearance_shortage_mm") or 0) for row in path_rows] or [0]
            ),
            "lowest_min_clearance_mm": min(
                [int(row.get("min_clearance_mm") or 0) for row in path_rows] or [0]
            ),
        },
        "central_congestion": central_rows[:12],
        "cluster_edge_vs_center_fit": edge_fit_rows[:12],
    }


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


def _orientation_debug_index(
    score: Dict[str, Any],
) -> tuple[
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    best_object = {}
    front_clearance = {}
    best_cluster = {}

    for item in score.get("orientation_debug") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        if not isinstance(cid, str):
            continue

        pen = int(item.get("penalty_mm") or 0)
        if item.get("kind") == "object_orientation":
            oid = item.get("object_id")
            if isinstance(oid, str):
                key = (cid, oid)
                prev = best_object.get(key)
                if prev is None or pen > int(prev.get("penalty_mm") or 0):
                    best_object[key] = deepcopy(item)
                if str(item.get("intent") or "") == "front_to_open_space":
                    front_clearance[key] = deepcopy(item)
        else:
            prev_cluster = best_cluster.get(cid)
            if prev_cluster is None or pen > int(prev_cluster.get("penalty_mm") or 0):
                best_cluster[cid] = deepcopy(item)

    return best_object, front_clearance, best_cluster


def _build_materialized_layout(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, Any]:
    room_model = _room_model(payload)
    room = deepcopy(room_model.get("room") or {})
    if isinstance(room_model.get("openings"), dict):
        room["openings"] = deepcopy(room_model.get("openings") or {})
    if isinstance(room_model.get("obstacles"), list):
        room["obstacles"] = deepcopy(room_model.get("obstacles") or [])

    clusters = []
    for row in state.get("clusters") or []:
        clusters.append(
            {
                "cluster_id": row.get("cluster_id"),
                "variant_id": row.get("variant_id"),
                "bbox": deepcopy(row.get("bbox") or {}),
                "world_center": deepcopy(row.get("world_center") or {}),
                "front_world": deepcopy(row.get("front_world")),
                "axis_world": deepcopy(row.get("axis_world")),
                "anchor_override": row.get("anchor_override"),
            }
        )

    objects = []
    for row in state.get("objects") or []:
        objects.append(
            {
                "object_id": row.get("object_id"),
                "cluster_id": row.get("cluster_id"),
                "variant_id": row.get("variant_id"),
                "bbox": deepcopy(row.get("bbox") or {}),
                "world_center": deepcopy(row.get("world_center") or {}),
                "front_world": deepcopy(row.get("front_world")),
                "axis_world": deepcopy(row.get("axis_world")),
                "required_clearance_mm": int(row.get("required_clearance_mm") or 0),
            }
        )

    return {"room": room, "clusters": clusters, "objects": objects}


def _build_goal_alignment_summary(
    payload: Dict[str, Any], score: Dict[str, Any]
) -> Dict[str, Any]:
    cluster_intents: List[Dict[str, Any]] = []
    object_intents: List[Dict[str, Any]] = []
    global_issues: List[Dict[str, Any]] = []

    for row in score.get("orientation_debug") or []:
        if not isinstance(row, dict):
            continue
        item = {
            "cluster_id": row.get("cluster_id"),
            "intent": row.get("intent"),
            "penalty_mm": int(row.get("penalty_mm") or 0),
            "dot": row.get("dot"),
        }
        if row.get("kind") == "object_orientation":
            item["object_id"] = row.get("object_id")
            object_intents.append(item)
        else:
            cluster_intents.append(item)

    for row in score.get("global_layout_debug") or []:
        if not isinstance(row, dict):
            continue
        item = {
            "kind": row.get("kind"),
            "cluster_id": row.get("cluster_id"),
            "penalty_mm": int(row.get("penalty_mm") or 0),
        }
        if isinstance(row.get("object_id"), str):
            item["object_id"] = row.get("object_id")
        if isinstance(row.get("constraint_type"), str):
            item["constraint_type"] = row.get("constraint_type")
        global_issues.append(item)

    cluster_intents.sort(
        key=lambda row: (
            -int(row.get("penalty_mm") or 0),
            str(row.get("cluster_id") or ""),
            str(row.get("intent") or ""),
        )
    )
    object_intents.sort(
        key=lambda row: (
            -int(row.get("penalty_mm") or 0),
            str(row.get("cluster_id") or ""),
            str(row.get("object_id") or ""),
            str(row.get("intent") or ""),
        )
    )
    global_issues.sort(
        key=lambda row: (
            -int(row.get("penalty_mm") or 0),
            str(row.get("cluster_id") or ""),
            str(row.get("kind") or ""),
        )
    )

    return {
        "room_notes": _room_notes(payload)[:8],
        "top_cluster_intents": cluster_intents[:8],
        "top_object_intents": object_intents[:10],
        "top_global_layout_issues": global_issues[:10],
    }


def _build_objects_world_from_state(
    state: Dict[str, Any], score: Dict[str, Any]
) -> List[Dict[str, Any]]:
    dominant_by_object, front_clear_by_object, _ = _orientation_debug_index(score)
    out: List[Dict[str, Any]] = []

    for row in state.get("objects") or []:
        cid = str(row.get("cluster_id") or "")
        oid = str(row.get("object_id") or "")
        dominant = dominant_by_object.get((cid, oid), {})
        front_clear = front_clear_by_object.get((cid, oid), {})
        bbox = row.get("bbox") or {}
        out.append(
            {
                "cluster_id": cid,
                "object_id": oid,
                "cluster_rot": int(row.get("rotation_ccw") or 0),
                "local_rect": deepcopy(row.get("local_rect") or {}),
                "polygon_ccw": deepcopy(row.get("polygon_ccw") or []),
                "bbox": deepcopy(bbox),
                "world_center": deepcopy(row.get("world_center") or {}),
                "size_mm": {
                    "w": int((row.get("local_rect") or {}).get("w") or 0),
                    "h": int((row.get("local_rect") or {}).get("h") or 0),
                },
                "front_world": deepcopy(row.get("front_world")),
                "axis_world": deepcopy(row.get("axis_world")),
                "required_clearance_mm": int(row.get("required_clearance_mm") or 0),
                "current_front_clear_mm": front_clear.get("front_clear_mm"),
                "current_back_clear_mm": front_clear.get("back_clear_mm"),
                "best_clear_mm": front_clear.get("best_clear_mm"),
                "distance_to_nearest_wall_mm": None,
                "dominant_debug_intent": dominant.get("intent"),
                "dominant_penalty_mm": dominant.get("penalty_mm"),
                "dominant_dot": dominant.get("dot"),
            }
        )

    out.sort(
        key=lambda item: (
            item.get("cluster_id") or "",
            -int(item.get("dominant_penalty_mm") or 0),
            item.get("object_id") or "",
        )
    )
    return out


def _build_metrics(
    payload: Dict[str, Any], state: Dict[str, Any], score: Dict[str, Any]
) -> Dict[str, Any]:
    openings = _openings(payload)
    room_center = state["room_center"]
    dominant_by_object, front_clear_by_object, dominant_by_cluster = (
        _orientation_debug_index(score)
    )

    object_front_clearance = []
    for row in state.get("objects") or []:
        cid = str(row.get("cluster_id") or "")
        oid = str(row.get("object_id") or "")
        center = (
            float((row.get("world_center") or {}).get("x") or 0.0),
            float((row.get("world_center") or {}).get("y") or 0.0),
        )
        door_distance, window_distance = _door_window_metrics(center, openings)
        front_debug = front_clear_by_object.get((cid, oid), {})
        current_front = front_debug.get("front_clear_mm")
        best_clear = front_debug.get("best_clear_mm")
        required = int(row.get("required_clearance_mm") or 0)
        shortage = None
        if current_front is not None:
            shortage = max(0, required - int(current_front))
        object_front_clearance.append(
            {
                "cluster_id": cid,
                "object_id": oid,
                "required_clearance_mm": required,
                "current_front_clear_mm": current_front,
                "best_open_clear_mm": best_clear,
                "shortage_mm": shortage,
                "dominant_penalty_mm": dominant_by_object.get((cid, oid), {}).get(
                    "penalty_mm"
                ),
                "distance_to_nearest_door_mm": door_distance,
                "distance_to_nearest_window_mm": window_distance,
            }
        )

    cluster_entry_proximity = []
    cluster_window_alignment = []
    cluster_window_band_occupancy = []
    zone_usage = []
    near_entry_clusters = []

    for cluster in state.get("clusters") or []:
        cid = str(cluster.get("cluster_id") or "")
        center = (
            float((cluster.get("world_center") or {}).get("x") or 0.0),
            float((cluster.get("world_center") or {}).get("y") or 0.0),
        )
        door_distance, window_distance = _door_window_metrics(center, openings)
        zone = _classify_cluster_zone(center, room_center, openings)
        entry_row = {
            "cluster_id": cid,
            "distance_to_nearest_door_mm": door_distance,
        }
        cluster_entry_proximity.append(entry_row)
        if door_distance is not None and door_distance < 1200:
            near_entry_clusters.append({**entry_row, "zone": zone})

        dominant_cluster = dominant_by_cluster.get(cid, {})
        cluster_window_alignment.append(
            {
                "cluster_id": cid,
                "distance_to_nearest_window_mm": window_distance,
                "dominant_intent": dominant_cluster.get("intent"),
                "dominant_penalty_mm": dominant_cluster.get("penalty_mm"),
                "dot": dominant_cluster.get("dot"),
            }
        )
        cluster_window_band_occupancy.append(
            {
                "cluster_id": cid,
                "distance_to_nearest_window_mm": window_distance,
                "window_clearance_hits": _window_clearance_hits(state, cluster_id=cid),
            }
        )
        zone_usage.append({"cluster_id": cid, "zone": zone})

    obstructing_objects = []
    for row in object_front_clearance:
        distance = row.get("distance_to_nearest_door_mm")
        if distance is not None and int(distance) < 900:
            obstructing_objects.append(
                {
                    "cluster_id": row["cluster_id"],
                    "object_id": row["object_id"],
                    "distance_to_nearest_door_mm": distance,
                }
            )

    return {
        "score_summary": {
            "score": int(score.get("score") or 0),
            "penalties": deepcopy(score.get("penalties") or {}),
            "hard_violation_count": int(score.get("hard_violation_count") or 0),
        },
        "goal_alignment_summary": _build_goal_alignment_summary(payload, score),
        "cluster_internal_constraint_fidelity": [
            deepcopy(row)
            for row in (score.get("global_layout_debug") or [])
            if isinstance(row, dict)
            and str(row.get("kind") or "") == "cluster_internal_constraint"
        ][:16],
        "cluster_affinity_to_preferred_zone": deepcopy(
            (score.get("global_layout_metrics") or {}).get(
                "cluster_affinity_to_preferred_zone"
            )
            or []
        ),
        "opening_band_blocking": deepcopy(
            (score.get("global_layout_metrics") or {}).get("opening_band_blocking")
            or []
        ),
        "main_path_clearance": deepcopy(
            (score.get("global_layout_metrics") or {}).get("main_path_clearance") or {}
        ),
        "central_congestion": deepcopy(
            (score.get("global_layout_metrics") or {}).get("central_congestion") or []
        ),
        "cluster_edge_vs_center_fit": deepcopy(
            (score.get("global_layout_metrics") or {}).get("cluster_edge_vs_center_fit")
            or []
        ),
        "object_front_clearance": object_front_clearance,
        "cluster_entry_proximity": cluster_entry_proximity,
        "cluster_window_alignment": cluster_window_alignment,
        "cluster_window_band_occupancy": cluster_window_band_occupancy,
        "path_obstruction_summary": {
            "near_entry_clusters": near_entry_clusters,
            "near_entry_objects": obstructing_objects,
            "worst_paths": [
                {
                    "path_id": row.get("path_id"),
                    "target_cluster_id": row.get("target_cluster_id"),
                    "priority": row.get("priority"),
                    "min_clearance_mm": row.get("min_clearance_mm"),
                    "clearance_shortage_mm": row.get("clearance_shortage_mm"),
                    "blocked_samples": row.get("blocked_samples"),
                    "sample_count": row.get("sample_count"),
                    "blocking_clusters": deepcopy(row.get("blocking_clusters") or []),
                    "blocking_objects": deepcopy(row.get("blocking_objects") or []),
                }
                for row in (
                    (
                        (score.get("global_layout_metrics") or {}).get(
                            "main_path_clearance"
                        )
                        or {}
                    ).get("paths")
                    or []
                )[:4]
                if isinstance(row, dict)
            ],
            "main_path_blockers": [
                deepcopy(row)
                for row in (
                    (
                        (score.get("global_layout_metrics") or {}).get(
                            "main_path_clearance"
                        )
                        or {}
                    ).get("blocking_objects")
                    or []
                )[:8]
                if isinstance(row, dict)
            ],
        },
        "zone_usage_summary": zone_usage,
        "orientation_debug": deepcopy(score.get("orientation_debug") or [])[:16],
        "global_layout_debug": deepcopy(score.get("global_layout_debug") or [])[:16],
    }


def _violations_by_cluster(errors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for error in errors:
        if not isinstance(error, dict):
            continue
        cluster_ids = []
        for key in ("cluster_id", "a_cluster_id", "b_cluster_id"):
            value = error.get(key)
            if isinstance(value, str) and value:
                cluster_ids.append(value)
        for cid in cluster_ids:
            slot = out.setdefault(cid, {"error_count": 0, "codes": [], "details": []})
            code = str(error.get("code") or "UNKNOWN")
            slot["error_count"] += 1
            if code not in slot["codes"]:
                slot["codes"].append(code)
            slot["details"].append(deepcopy(error))
    return out


def _build_key_findings(
    score: Dict[str, Any], metrics: Dict[str, Any], errors: List[Dict[str, Any]]
) -> List[str]:
    findings: List[str] = []

    for error in errors[:4]:
        if not isinstance(error, dict):
            continue
        code = str(error.get("code") or "UNKNOWN")
        if code == "OBJECT_OVERLAP":
            findings.append(
                "Hard invalid: overlap between "
                f"{error.get('a_cluster_id')}.{error.get('a_object_id')} and "
                f"{error.get('b_cluster_id')}.{error.get('b_object_id')}."
            )
        elif code == "OBJECT_OUT_OF_BOUNDS":
            findings.append(
                "Hard invalid: "
                f"{error.get('cluster_id')}.{error.get('object_id')} leaves the room."
            )
        elif code == "OBJECT_HITS_OBSTACLE":
            findings.append(
                "Hard invalid: "
                f"{error.get('cluster_id')}.{error.get('object_id')} intersects "
                f"{error.get('obstacle_id')}."
            )

    if findings:
        return findings[:6]

    for row in (score.get("prioritized_clusters") or [])[:3]:
        cid = row.get("cluster_id")
        cluster_score = row.get("score")
        findings.append(
            f"Cluster {cid} carries the largest remaining soft penalty ({cluster_score})."
        )

    for row in (metrics.get("cluster_internal_constraint_fidelity") or [])[:2]:
        if not isinstance(row, dict):
            continue
        findings.append(
            "Cluster "
            f"{row.get('cluster_id')} drifts from local constraint "
            f"{row.get('constraint_type')} (penalty={row.get('penalty_mm')})."
        )
        if len(findings) >= 6:
            return findings[:6]

    for row in (metrics.get("cluster_affinity_to_preferred_zone") or [])[:2]:
        if not isinstance(row, dict):
            continue
        findings.append(
            "Cluster "
            f"{row.get('cluster_id')} sits away from its preferred zone "
            f"{row.get('preferred_zone')} (penalty={row.get('penalty_mm')})."
        )
        if len(findings) >= 6:
            return findings[:6]

    for row in ((metrics.get("main_path_clearance") or {}).get("paths") or [])[:2]:
        if not isinstance(row, dict):
            continue
        shortage = int(row.get("clearance_shortage_mm") or 0)
        blocked_samples = int(row.get("blocked_samples") or 0)
        if shortage <= 0 and blocked_samples <= 0:
            continue
        target = str(row.get("target_cluster_id") or "room target")
        findings.append(
            f"Main path to {target} narrows to {row.get('min_clearance_mm')}mm "
            f"(shortage={shortage}, blocked_samples={blocked_samples})."
        )
        if len(findings) >= 6:
            return findings[:6]

    for row in (metrics.get("object_front_clearance") or [])[:4]:
        current_front = row.get("current_front_clear_mm")
        best_clear = row.get("best_open_clear_mm")
        shortage = row.get("shortage_mm")
        if current_front is None or best_clear is None:
            continue
        findings.append(
            f"Object {row.get('cluster_id')}.{row.get('object_id')} has "
            f"{current_front}mm front clearance vs {best_clear}mm best open space "
            f"(shortage={shortage})."
        )
        if len(findings) >= 6:
            break

    return findings[:6]


def _build_repair_debug_from_score(
    score: Dict[str, Any], candidate_counts: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    orientation_debug = deepcopy(score.get("orientation_debug") or [])
    global_layout_debug = deepcopy(score.get("global_layout_debug") or [])
    critical_orientation_debug = [
        row for row in orientation_debug if int(row.get("penalty_mm") or 0) >= 400
    ]
    return {
        "seed_metrics": {
            "score": int(score.get("score") or 0),
            "penalties": deepcopy(score.get("penalties") or {}),
            "global_layout_metrics": deepcopy(score.get("global_layout_metrics") or {}),
        },
        "seed_verify": {
            "hard_valid": bool(score.get("hard_valid")),
            "quality": {
                "layout_score": int(score.get("score") or 0),
                "orientation_debug": orientation_debug,
                "critical_orientation_debug": critical_orientation_debug,
                "global_layout_debug": global_layout_debug,
            },
            "repair_guidance": {
                "prioritized_clusters": deepcopy(
                    score.get("prioritized_clusters") or []
                ),
                "prioritized_objects": deepcopy(score.get("prioritized_objects") or []),
            },
            "quality_gate": {
                "result": "PASS"
                if bool(score.get("hard_valid")) and not critical_orientation_debug
                else "REVISE"
            },
        },
        "repair_targets": {
            "prioritized_clusters": deepcopy(score.get("prioritized_clusters") or []),
            "prioritized_objects": deepcopy(score.get("prioritized_objects") or []),
        },
        "candidate_counts": deepcopy(candidate_counts or {}),
        "quality_gate": {
            "result": "PASS"
            if bool(score.get("hard_valid")) and not critical_orientation_debug
            else "REVISE"
        },
    }


def PromotePhase2RepairToSeedPayload(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    promoted = deepcopy(payload)
    cluster_transforms, selected_variants = _resolved_seed_layout(payload, repair)
    promoted["seed_layout"] = {
        "cluster_transforms": cluster_transforms,
        "selected_variants": selected_variants,
    }

    variant_by_cluster = {
        str(item.get("cluster_id")): str(item.get("variant_id") or "")
        for item in selected_variants
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str)
    }
    repairs_by_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for row in repair.get("object_repairs") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            repairs_by_cluster.setdefault(row["cluster_id"], []).append(deepcopy(row))

    promoted_cards = []
    for transform in cluster_transforms:
        cid = str(transform.get("cluster_id") or "")
        card = _current_cluster_variant_card(
            payload, cid, variant_by_cluster.get(cid) or None
        )
        if not card:
            continue

        bbox = _cluster_local_bbox(card)
        rects = []
        rect_order = []
        for rect in (card.get("cluster_footprint") or {}).get("rects") or []:
            if isinstance(rect, dict) and isinstance(rect.get("id"), str):
                rects.append(deepcopy(rect))
                rect_order.append(rect["id"])
        rect_map = {row["id"]: row for row in rects}
        placements = []
        for placement in card.get("local_placements") or []:
            if isinstance(placement, dict) and isinstance(placement.get("id"), str):
                placements.append(deepcopy(placement))
        placement_map = {row["id"]: row for row in placements}

        orientation_meta = deepcopy(card.get("orientation_meta") or {})
        important = deepcopy(orientation_meta.get("important_objects") or {})
        cluster_front_local = parse_vec2(orientation_meta.get("cluster_front_local"))

        for rep in repairs_by_cluster.get(cid, []):
            oid = rep.get("object_id")
            op = rep.get("op")
            params = rep.get("params") or {}
            if op == "set_anchor":
                card["anchor_override"] = params.get("anchor")
                continue
            if not isinstance(oid, str) or oid not in rect_map:
                continue

            rect = rect_map[oid]
            placement = placement_map.get(
                oid, {"id": oid, "x": rect.get("x", 0), "y": rect.get("y", 0), "rot": 0}
            )
            obj_meta = (
                important.get(oid) if isinstance(important.get(oid), dict) else {}
            )
            front_local = (
                parse_vec2((obj_meta or {}).get("front_local")) or cluster_front_local
            )
            axis_local = parse_vec2((obj_meta or {}).get("axis_local"))

            if op == "rotate_object":
                rr = int(params.get("rot", 0)) % 360
                rect = _rotate_rect_about_center(rect, rr)
                placement["rot"] = int((placement.get("rot") or 0) + rr) % 360
                front_local = rotate_vec_ccw_90s(front_local, rr)
                axis_local = rotate_vec_ccw_90s(axis_local, rr)
            elif op == "mirror_object":
                axis = str(params.get("axis") or "x").lower()
                rect = _mirror_rect_in_bbox(rect, bbox, axis)
                placement["x"] = rect.get("x", placement.get("x"))
                placement["y"] = rect.get("y", placement.get("y"))
                front_local = mirror_vec(front_local, axis)
                axis_local = mirror_vec(axis_local, axis)
            elif op == "nudge_object":
                dx = int(round(float(params.get("dx", 0))))
                dy = int(round(float(params.get("dy", 0))))
                rect = {
                    **rect,
                    "x": int(round(float(rect.get("x", 0)) + dx)),
                    "y": int(round(float(rect.get("y", 0)) + dy)),
                }
                placement["x"] = int(round(float(placement.get("x", 0)) + dx))
                placement["y"] = int(round(float(placement.get("y", 0)) + dy))
            elif op == "swap_objects":
                other_id = str(params.get("other_object_id") or "")
                other_rect = rect_map.get(other_id)
                if isinstance(other_rect, dict):
                    ox, oy = other_rect.get("x"), other_rect.get("y")
                    other_rect["x"], other_rect["y"] = rect.get("x"), rect.get("y")
                    rect["x"], rect["y"] = ox, oy
                    other_place = placement_map.get(other_id)
                    if isinstance(other_place, dict):
                        px, py = other_place.get("x"), other_place.get("y")
                        other_place["x"], other_place["y"] = (
                            placement.get("x"),
                            placement.get("y"),
                        )
                        placement["x"], placement["y"] = px, py
                        placement_map[other_id] = other_place
                    rect_map[other_id] = other_rect
            elif op == "set_front_override":
                front_local = normalize_vec(
                    (float(params.get("dx", 0.0)), float(params.get("dy", 0.0)))
                )

            placement["x"] = int(round(float(rect.get("x", placement.get("x", 0)))))
            placement["y"] = int(round(float(rect.get("y", placement.get("y", 0)))))
            rect_map[oid] = rect
            placement_map[oid] = placement
            important.setdefault(oid, {})
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

        card.setdefault("cluster_footprint", {})
        card["cluster_footprint"]["rects"] = [
            deepcopy(rect_map[oid]) for oid in rect_order if oid in rect_map
        ]
        card["cluster_footprint"]["local_bbox"] = _cluster_local_bbox(card)
        card["orientation_meta"] = orientation_meta
        card["orientation_meta"]["important_objects"] = important
        ordered_placements = []
        for oid in rect_order:
            placement = placement_map.get(oid)
            if isinstance(placement, dict):
                ordered_placements.append(placement)
        card["local_placements"] = ordered_placements
        promoted_cards.append(card)

    promoted["cluster_cards"] = promoted_cards
    state = materialize_phase2_state(promoted, repair=None)
    score = score_phase2_state(promoted, state)
    promoted["objects_world"] = _build_objects_world_from_state(state, score)
    previous_debug = payload.get("repair_debug") or {}
    promoted["repair_debug"] = _build_repair_debug_from_score(
        score, candidate_counts=previous_debug.get("candidate_counts") or {}
    )
    promoted.pop("tool_context", None)
    return promoted


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
    global_layout_debug: List[Dict[str, Any]] = []
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
            if intent in {"face_window", "front_to_window"}:
                mid = _nearest_midpoint(center, openings.get("windows") or [])
                if mid is not None:
                    dot = _dot(front_world, _vec_from_to(center, mid))
                    weight = 620.0 if intent == "front_to_window" else 520.0
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * weight
            elif intent in {"face_entry", "front_to_entry"}:
                mid = _nearest_midpoint(center, openings.get("doors") or [])
                if mid is not None:
                    dot = _dot(front_world, _vec_from_to(center, mid))
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 520.0
            elif intent == "back_to_wall":
                target_vec = _nearest_wall_inward_dir(center, room_poly)
                dot = _dot(front_world, target_vec)
                pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 720.0
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
        axis = parse_vec2(cluster.get("axis_world"))
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
                pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 860.0
            elif intent == "face_window":
                mid = _nearest_midpoint(center, openings.get("windows") or [])
                if mid is not None:
                    dot = _dot(front, _vec_from_to(center, mid))
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 760.0
            elif intent == "back_to_wall":
                target_vec = _nearest_wall_inward_dir(center, room_poly)
                if front is not None and target_vec is not None:
                    dot = _dot(front, target_vec)
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 900.0
            elif intent == "axis_parallel_window":
                window_dir = _nearest_opening_line_direction(
                    center, openings.get("windows") or []
                )
                if axis is not None and window_dir is not None:
                    parallel = abs(_dot(axis, window_dir) or -1.0)
                    dot = parallel
                    pen = max(0.0, 1.0 - parallel) * 620.0
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

    internal_cluster_penalty, internal_object_penalty, internal_debug = (
        _score_cluster_internal_fidelity(state)
    )
    for cid, penalty in internal_cluster_penalty.items():
        cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + penalty
    for key, penalty in internal_object_penalty.items():
        object_penalty[key] = object_penalty.get(key, 0.0) + penalty
        cluster_penalty[key[0]] = cluster_penalty.get(key[0], 0.0) + penalty * 0.2
    global_layout_debug.extend(internal_debug)

    global_layout_metrics = _score_global_layout_metrics(
        payload, state, openings, room_center
    )
    for cid, penalty in (global_layout_metrics.get("cluster_penalty") or {}).items():
        cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + float(penalty or 0.0)
    for row in (global_layout_metrics.get("main_path_clearance") or {}).get(
        "blocking_objects"
    ) or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        oid = row.get("object_id")
        penalty = float(row.get("penalty_mm") or 0.0)
        if isinstance(cid, str) and isinstance(oid, str) and penalty > 0.0:
            object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + penalty
    for key in (
        "cluster_affinity_to_preferred_zone",
        "opening_band_blocking",
        "central_congestion",
        "cluster_edge_vs_center_fit",
    ):
        for row in global_layout_metrics.get(key) or []:
            if isinstance(row, dict):
                global_layout_debug.append({"kind": key, **deepcopy(row)})
    for row in (global_layout_metrics.get("main_path_clearance") or {}).get(
        "paths"
    ) or []:
        if isinstance(row, dict):
            global_layout_debug.append({"kind": "main_path_clearance", **deepcopy(row)})
    for row in (global_layout_metrics.get("main_path_clearance") or {}).get(
        "blocking_objects"
    ) or []:
        if isinstance(row, dict):
            global_layout_debug.append(
                {"kind": "main_path_object_blocker", **deepcopy(row)}
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
        "global_layout_debug": sorted(
            global_layout_debug, key=lambda x: -int(x.get("penalty_mm") or 0)
        ),
        "prioritized_clusters": prioritized_clusters[:8],
        "prioritized_objects": prioritized_objects[:16],
        "penalties": {
            "hard_penalty": int(round(hard_penalty)),
            "soft_penalty": int(round(soft_penalty)),
            "total_penalty": int(round(total_penalty)),
        },
        "score": -int(round(total_penalty)),
        "global_layout_metrics": {
            key: deepcopy(value)
            for key, value in global_layout_metrics.items()
            if key != "cluster_penalty"
        },
    }


# -----------------------------------------------------------------------------
# Public tool API
# -----------------------------------------------------------------------------
def DiagnosePhase2Seed(*, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = materialize_phase2_state(payload, repair=None)
    score = score_phase2_state(payload, state)
    return {
        "result": "OK",
        "search_phase": _repair_phase(payload),
        "hard_valid": score["hard_valid"],
        "score": score["score"],
        "penalties": score["penalties"],
        "prioritized_clusters": score["prioritized_clusters"],
        "prioritized_objects": score["prioritized_objects"],
        "hard_violations": score["hard_violations"],
        "orientation_debug": score["orientation_debug"][:16],
        "global_layout_debug": score.get("global_layout_debug") or [],
        "global_layout_metrics": deepcopy(score.get("global_layout_metrics") or {}),
    }


def EnumeratePhase2RepairMoves(
    *, payload: Dict[str, Any], limit: int = 24
) -> Dict[str, Any]:
    state = materialize_phase2_state(payload, repair=None)
    score = score_phase2_state(payload, state)
    search_phase = _repair_phase(payload)
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)
    vmap = _seed_variant_map(seed_layout)
    allowed_variants = {}
    for cid, card in _cluster_cards_as_map(payload.get("cluster_cards") or {}).items():
        variant_ids = []
        for item in card.get("available_variants") or card.get("variants") or []:
            if isinstance(item, dict) and isinstance(item.get("variant_id"), str):
                variant_ids.append(item["variant_id"])
            elif isinstance(item, str):
                variant_ids.append(item)
        if variant_ids:
            allowed_variants[cid] = sorted(set(variant_ids))

    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    cluster_preferences = _cluster_preference_profile(payload)
    cluster_state_by_id = {
        str(row.get("cluster_id") or ""): row for row in state.get("clusters") or []
    }
    room_poly = state["room_polygon"]
    openings = _openings(payload)

    macro_moves: List[Dict[str, Any]] = []
    object_moves: List[Dict[str, Any]] = []

    priority_cluster_ids: List[str] = []
    for row in score["prioritized_clusters"][:4]:
        cid = row.get("cluster_id")
        if isinstance(cid, str) and cid not in priority_cluster_ids:
            priority_cluster_ids.append(cid)
    for row in (score.get("global_layout_metrics") or {}).get(
        "cluster_affinity_to_preferred_zone"
    ) or []:
        cid = row.get("cluster_id")
        if isinstance(cid, str) and cid not in priority_cluster_ids:
            priority_cluster_ids.append(cid)
        if len(priority_cluster_ids) >= 4:
            break

    def _offsets_from_vector(
        raw_dx: float, raw_dy: float, *, include_far_steps: bool
    ) -> list[tuple[int, int]]:
        sx = 0 if abs(raw_dx) < (grid_mm * 0.5) else (1 if raw_dx > 0 else -1)
        sy = 0 if abs(raw_dy) < (grid_mm * 0.5) else (1 if raw_dy > 0 else -1)
        if sx == 0 and sy == 0:
            return []
        multipliers = (2, 4, 6) if include_far_steps else (1, 2, 3)
        prefer_x = abs(raw_dx) >= abs(raw_dy)
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for mult in multipliers:
            step = mult * grid_mm
            candidates: list[tuple[int, int]] = []
            if sx and sy:
                candidates.append((sx * step, sy * step))
            if prefer_x:
                if sx:
                    candidates.append((sx * step, 0))
                if sy:
                    candidates.append((0, sy * step))
            else:
                if sy:
                    candidates.append((0, sy * step))
                if sx:
                    candidates.append((sx * step, 0))
            for offset in candidates:
                if offset not in seen:
                    seen.add(offset)
                    out.append(offset)
        return out

    def _targeted_macro_offsets(cluster_id: str) -> list[tuple[tuple[int, int], str]]:
        cluster = cluster_state_by_id.get(cluster_id)
        if not isinstance(cluster, dict):
            return []
        profile = cluster_preferences.get(
            cluster_id, {"prefer": set(), "avoid": set(), "intents": set()}
        )
        prefer_tags = set(profile.get("prefer") or set())
        avoid_tags = set(profile.get("avoid") or set())
        intents = set(profile.get("intents") or set())
        center = (
            float((cluster.get("world_center") or {}).get("x") or 0.0),
            float((cluster.get("world_center") or {}).get("y") or 0.0),
        )
        targets: list[tuple[tuple[float, float], str]] = []
        if "window_side" in prefer_tags or intents & {
            "face_window",
            "front_to_window",
            "axis_parallel_window",
        }:
            midpoint = _nearest_midpoint(center, openings.get("windows") or [])
            if midpoint is not None:
                targets.append(
                    (
                        (
                            float(midpoint[0]) - center[0],
                            float(midpoint[1]) - center[1],
                        ),
                        "toward window-side zone",
                    )
                )
        if "entry_side" in prefer_tags or intents & {"face_entry", "front_to_entry"}:
            midpoint = _nearest_midpoint(center, openings.get("doors") or [])
            if midpoint is not None:
                targets.append(
                    (
                        (
                            float(midpoint[0]) - center[0],
                            float(midpoint[1]) - center[1],
                        ),
                        "toward entry-side zone",
                    )
                )
        if "far_from_entry" in prefer_tags or avoid_tags & {
            "door_swing",
            "entry_blocking",
        }:
            midpoint = _nearest_midpoint(center, openings.get("doors") or [])
            if midpoint is not None:
                targets.append(
                    (
                        (
                            center[0] - float(midpoint[0]),
                            center[1] - float(midpoint[1]),
                        ),
                        "away from the entry zone",
                    )
                )
        if prefer_tags & {"recess_or_edge", "long_wall"} or intents & {
            "back_to_wall",
            "access_to_open_space",
        }:
            edge_point = _nearest_boundary_point(center, room_poly)
            if edge_point is not None:
                targets.append(
                    (
                        (
                            float(edge_point[0]) - center[0],
                            float(edge_point[1]) - center[1],
                        ),
                        "toward the room edge",
                    )
                )
        if "long_wall" in prefer_tags:
            longest = _longest_wall_segment(room_poly)
            if longest:
                try:
                    wall_point, _ = nearest_points(longest["line"], Point(center))
                except Exception:
                    wall_point = None
                if wall_point is not None:
                    targets.append(
                        (
                            (
                                float(wall_point.x) - center[0],
                                float(wall_point.y) - center[1],
                            ),
                            "toward the longest wall",
                        )
                    )

        out: list[tuple[tuple[int, int], str]] = []
        seen: set[tuple[int, int, str]] = set()
        for vector, reason in targets:
            for offset in _offsets_from_vector(
                float(vector[0]), float(vector[1]), include_far_steps=True
            ):
                signature = (offset[0], offset[1], reason)
                if signature in seen or offset == (0, 0):
                    continue
                seen.add(signature)
                out.append((offset, reason))
        return out

    # Macro-first: variants, cluster rotation, then small translations.
    for cid in priority_cluster_ids:
        tf = tmap.get(cid) or {"cluster_id": cid, "x": 0, "y": 0, "rot": 0}
        current_variant = str((vmap.get(cid) or {}).get("variant_id") or "")

        for variant_id in allowed_variants.get(cid, []):
            if variant_id == current_variant:
                continue
            macro_moves.append(
                {
                    "kind": "cluster_variant",
                    "cluster_id": cid,
                    "proposal": {
                        "cluster_transforms": [],
                        "selected_variants": [
                            {"cluster_id": cid, "variant_id": variant_id}
                        ],
                        "object_repairs": [],
                    },
                    "reason": f"Try variant switch for failing cluster {cid}",
                }
            )

        for rot in (0, 90, 180, 270):
            if rot == int(tf.get("rot") or 0):
                continue
            macro_moves.append(
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
        for (dx, dy), move_reason in _targeted_macro_offsets(cid):
            macro_moves.append(
                {
                    "kind": "cluster_pose",
                    "cluster_id": cid,
                    "proposal": {
                        "cluster_transforms": [
                            {
                                **deepcopy(tf),
                                "x": int(tf.get("x") or 0) + dx,
                                "y": int(tf.get("y") or 0) + dy,
                            }
                        ],
                        "selected_variants": [],
                        "object_repairs": [],
                    },
                    "reason": f"Move {cid} {move_reason}",
                }
            )
        for dx, dy in (
            (grid_mm, 0),
            (-grid_mm, 0),
            (0, grid_mm),
            (0, -grid_mm),
        ):
            macro_moves.append(
                {
                    "kind": "cluster_pose",
                    "cluster_id": cid,
                    "proposal": {
                        "cluster_transforms": [
                            {
                                **deepcopy(tf),
                                "x": int(tf.get("x") or 0) + dx,
                                "y": int(tf.get("y") or 0) + dy,
                            }
                        ],
                        "selected_variants": [],
                        "object_repairs": [],
                    },
                    "reason": f"Try a small grid translation for failing cluster {cid}",
                }
            )

    # Object-level only after macro moves.
    for row in score["prioritized_objects"][:6]:
        cid = row["cluster_id"]
        oid = row["object_id"]
        dominant_rows = [
            entry
            for entry in score.get("orientation_debug") or []
            if isinstance(entry, dict)
            and str(entry.get("cluster_id") or "") == cid
            and str(entry.get("object_id") or "") == oid
        ]
        dominant_intents = {
            str(entry.get("intent") or "").strip().lower()
            for entry in dominant_rows
            if int(entry.get("penalty_mm") or 0) >= 200
        }
        if dominant_intents & {
            "front_to_open_space",
            "face_window",
            "front_to_window",
            "face_entry",
            "front_to_entry",
            "back_to_wall",
        }:
            for rot in (90, 180, 270):
                object_moves.append(
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
        for dx, dy in (
            (GLOBAL_LAYOUT_GRID_MM, 0),
            (-GLOBAL_LAYOUT_GRID_MM, 0),
            (0, GLOBAL_LAYOUT_GRID_MM),
            (0, -GLOBAL_LAYOUT_GRID_MM),
        ):
            object_moves.append(
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

    moves = object_moves if search_phase == "object_refine" else macro_moves

    return {
        "result": "OK",
        "search_phase": search_phase,
        "moves": moves[:limit],
        "baseline_score": score["score"],
    }


def EvaluatePhase2Proposal(
    *, payload: Dict[str, Any], repair: Dict[str, Any], move_limit: int = 16
) -> Dict[str, Any]:
    baseline_state = materialize_phase2_state(payload, repair=None)
    baseline_score = score_phase2_state(payload, baseline_state)

    candidate_state = materialize_phase2_state(payload, repair=repair)
    candidate_score = score_phase2_state(payload, candidate_state)
    errors = deepcopy(candidate_score.get("hard_violations") or [])
    metrics = _build_metrics(payload, candidate_state, candidate_score)

    next_payload = PromotePhase2RepairToSeedPayload(payload=payload, repair=repair)
    next_moves = []
    if bool(candidate_score.get("hard_valid")):
        next_moves = (
            EnumeratePhase2RepairMoves(payload=next_payload, limit=move_limit).get(
                "moves"
            )
            or []
        )

    diagnosis = {
        "prioritized_clusters": deepcopy(
            candidate_score.get("prioritized_clusters") or []
        ),
        "prioritized_objects": deepcopy(
            candidate_score.get("prioritized_objects") or []
        ),
        "key_findings": _build_key_findings(candidate_score, metrics, errors),
        "enumerated_moves": next_moves,
    }

    return {
        "result": "OK",
        "hard_valid": bool(candidate_score.get("hard_valid")),
        "errors": errors,
        "violations_by_cluster": _violations_by_cluster(errors),
        "materialized_layout": _build_materialized_layout(payload, candidate_state),
        "metrics": metrics,
        "diagnosis": diagnosis,
        "baseline_comparison": {
            "baseline_score": int(baseline_score.get("score") or 0),
            "baseline_hard_valid": bool(baseline_score.get("hard_valid")),
            "candidate_score": int(candidate_score.get("score") or 0),
            "delta_score": int(
                int(candidate_score.get("score") or 0)
                - int(baseline_score.get("score") or 0)
            ),
            "improved": bool(
                candidate_score.get("hard_valid")
                and int(candidate_score.get("score") or 0)
                > int(baseline_score.get("score") or 0)
            ),
        },
    }


def ResolvePhase2HardViolations(
    *,
    payload: Dict[str, Any],
    repair: Dict[str, Any],
    max_rounds: int = 40,
    max_radius_steps: int = 6,
) -> Dict[str, Any]:
    working_repair = _full_seed_repair(payload, repair)
    best_repair = deepcopy(working_repair)
    best_evaluation = EvaluatePhase2Proposal(payload=payload, repair=working_repair)
    if bool(best_evaluation.get("hard_valid")):
        return {
            "result": "OK",
            "repair": best_repair,
            "evaluation": best_evaluation,
            "attempts": [],
        }

    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    attempted_signatures = {
        json.dumps(
            {
                "cluster_transforms": best_repair.get("cluster_transforms"),
                "selected_variants": best_repair.get("selected_variants"),
                "object_repairs": best_repair.get("object_repairs"),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    }
    attempts: list[dict[str, Any]] = []

    def _repair_rank(evaluation: Dict[str, Any]) -> tuple[int, int, int]:
        return (
            1 if bool(evaluation.get("hard_valid")) else 0,
            -len(evaluation.get("errors") or []),
            int(
                ((evaluation.get("baseline_comparison") or {}).get("candidate_score"))
                or 0
            ),
        )

    def _target_clusters(errors: list[dict[str, Any]]) -> list[str]:
        ordered: list[str] = []
        for error in errors:
            if not isinstance(error, dict):
                continue
            code = str(error.get("code") or "")
            cluster_ids: list[str] = []
            if code == "OBJECT_OVERLAP":
                for key in ("a_cluster_id", "b_cluster_id"):
                    value = error.get(key)
                    if isinstance(value, str) and value:
                        cluster_ids.append(value)
            else:
                value = error.get("cluster_id")
                if isinstance(value, str) and value:
                    cluster_ids.append(value)
            for cluster_id in cluster_ids:
                if cluster_id not in ordered:
                    ordered.append(cluster_id)
        return ordered

    offsets: list[tuple[int, int]] = []
    for radius in range(1, max(1, int(max_radius_steps)) + 1):
        step = radius * grid_mm
        offsets.extend(
            [
                (step, 0),
                (-step, 0),
                (0, step),
                (0, -step),
                (step, step),
                (step, -step),
                (-step, step),
                (-step, -step),
            ]
        )

    for round_idx in range(1, max(1, int(max_rounds)) + 1):
        candidate_clusters = _target_clusters(best_evaluation.get("errors") or [])
        if not candidate_clusters:
            break

        local_best_repair: Dict[str, Any] | None = None
        local_best_evaluation: Dict[str, Any] | None = None
        local_best_cluster: str | None = None
        local_best_offset = (0, 0)

        for cluster_id in candidate_clusters:
            current_transform = next(
                (
                    row
                    for row in working_repair.get("cluster_transforms") or []
                    if isinstance(row, dict) and row.get("cluster_id") == cluster_id
                ),
                None,
            )
            if not isinstance(current_transform, dict):
                continue

            for dx, dy in offsets:
                candidate_repair = deepcopy(working_repair)
                for row in candidate_repair.get("cluster_transforms") or []:
                    if isinstance(row, dict) and row.get("cluster_id") == cluster_id:
                        row["x"] = int(row.get("x") or 0) + dx
                        row["y"] = int(row.get("y") or 0) + dy
                        break
                candidate_repair["notes"] = [
                    *[
                        str(note)
                        for note in candidate_repair.get("notes") or []
                        if str(note).strip()
                    ],
                    f"Hard-fix nudge for {cluster_id}: dx={dx}, dy={dy}.",
                ]
                signature = json.dumps(
                    {
                        "cluster_transforms": candidate_repair.get(
                            "cluster_transforms"
                        ),
                        "selected_variants": candidate_repair.get("selected_variants"),
                        "object_repairs": candidate_repair.get("object_repairs"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                if signature in attempted_signatures:
                    continue
                attempted_signatures.add(signature)

                candidate_evaluation = EvaluatePhase2Proposal(
                    payload=payload,
                    repair=candidate_repair,
                )
                if local_best_evaluation is None or _repair_rank(
                    candidate_evaluation
                ) > _repair_rank(local_best_evaluation):
                    local_best_repair = candidate_repair
                    local_best_evaluation = candidate_evaluation
                    local_best_cluster = cluster_id
                    local_best_offset = (dx, dy)
                if bool(candidate_evaluation.get("hard_valid")):
                    attempts.append(
                        {
                            "round": round_idx,
                            "cluster_id": cluster_id,
                            "dx": dx,
                            "dy": dy,
                            "hard_valid": True,
                            "error_count": len(
                                candidate_evaluation.get("errors") or []
                            ),
                        }
                    )
                    return {
                        "result": "OK",
                        "repair": candidate_repair,
                        "evaluation": candidate_evaluation,
                        "attempts": attempts,
                    }

        if local_best_repair is None or local_best_evaluation is None:
            break

        attempts.append(
            {
                "round": round_idx,
                "cluster_id": local_best_cluster,
                "dx": local_best_offset[0],
                "dy": local_best_offset[1],
                "hard_valid": bool(local_best_evaluation.get("hard_valid")),
                "error_count": len(local_best_evaluation.get("errors") or []),
            }
        )
        if _repair_rank(local_best_evaluation) <= _repair_rank(best_evaluation):
            break

        working_repair = local_best_repair
        best_repair = local_best_repair
        best_evaluation = local_best_evaluation

    return {
        "result": "OK",
        "repair": best_repair,
        "evaluation": best_evaluation,
        "attempts": attempts,
    }


def CompileAcceptedPhase2Proposal(
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
        "status": "OK" if score["hard_valid"] else "INVALID",
        "room": room,
        "objects": objects,
        "notes": deepcopy(repair.get("notes") or []),
        "missing": deepcopy(score.get("hard_violations") or []),
    }


def ScorePhase2Repair(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    evaluation = EvaluatePhase2Proposal(payload=payload, repair=repair)
    baseline = evaluation.get("baseline_comparison") or {}
    candidate = evaluation.get("diagnosis") or {}
    materialized = evaluation.get("materialized_layout") or {}
    return {
        "result": "OK",
        "baseline": {
            "score": int(baseline.get("baseline_score") or 0),
            "hard_valid": bool(baseline.get("baseline_hard_valid")),
        },
        "candidate": {
            "score": int(baseline.get("candidate_score") or 0),
            "hard_valid": bool(evaluation.get("hard_valid")),
            "hard_violations": deepcopy(evaluation.get("errors") or []),
            "prioritized_clusters": deepcopy(
                candidate.get("prioritized_clusters") or []
            ),
            "prioritized_objects": deepcopy(candidate.get("prioritized_objects") or []),
            "orientation_debug": deepcopy(
                (evaluation.get("metrics") or {}).get("orientation_debug") or []
            ),
            "materialized_layout": materialized,
        },
        "delta_score": int(baseline.get("delta_score") or 0),
        "improved": bool(baseline.get("improved")),
    }


def PreviewPhase2Repair(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    compiled = CompileAcceptedPhase2Proposal(payload=payload, repair=repair)
    evaluation = EvaluatePhase2Proposal(payload=payload, repair=repair)
    return {
        "result": "OK",
        "compiled": compiled,
        "score": {
            "hard_valid": bool(evaluation.get("hard_valid")),
            "hard_violations": deepcopy(evaluation.get("errors") or []),
            "orientation_debug": deepcopy(
                (evaluation.get("metrics") or {}).get("orientation_debug") or []
            ),
            "prioritized_clusters": deepcopy(
                (evaluation.get("diagnosis") or {}).get("prioritized_clusters") or []
            ),
            "prioritized_objects": deepcopy(
                (evaluation.get("diagnosis") or {}).get("prioritized_objects") or []
            ),
            "score": int(
                ((evaluation.get("baseline_comparison") or {}).get("candidate_score"))
                or 0
            ),
        },
    }


TOOL_REGISTRY: Dict[str, Any] = {
    "DiagnosePhase2Seed": DiagnosePhase2Seed,
    "EnumeratePhase2RepairMoves": EnumeratePhase2RepairMoves,
    "ScorePhase2Repair": ScorePhase2Repair,
    "PreviewPhase2Repair": PreviewPhase2Repair,
    "EvaluatePhase2Proposal": EvaluatePhase2Proposal,
    "ResolvePhase2HardViolations": ResolvePhase2HardViolations,
    "CompileAcceptedPhase2Proposal": CompileAcceptedPhase2Proposal,
    "PromotePhase2RepairToSeedPayload": PromotePhase2RepairToSeedPayload,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "EvaluatePhase2Proposal",
            "description": "Materialize and evaluate a phase-2 repair proposal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                    "move_limit": {"type": "integer"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "CompileAcceptedPhase2Proposal",
            "description": "Compile an accepted proposal into the absolute layout output shape.",
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
            "name": "ResolvePhase2HardViolations",
            "description": "Apply deterministic grid nudges to reduce overlap and out-of-bounds violations on a chosen candidate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                    "max_rounds": {"type": "integer"},
                    "max_radius_steps": {"type": "integer"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PromotePhase2RepairToSeedPayload",
            "description": "Promote a repair proposal into a new phase-2 seed payload for the next iteration.",
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
