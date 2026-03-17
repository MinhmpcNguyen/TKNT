from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def GlobalClusterVerifier(
    *,
    room_model: Dict[str, Any],
    clusters_outlines: Any,
    cluster_transforms: List[Dict[str, Any]],
    grid_mm: int,
    eps_area: float = 1e-6,
    forbid_obstacle_touch: bool = True,
    generate_hints: bool = True,
    nudge_steps: int = 3,
    max_candidates_per_issue: int = 4,
    margin_mm: int | None = None,
    consider_rotations: bool = True,
    relation_plan: Dict[str, Any] | None = None,
    base_preferred_gap_mm: int | None = None,
    min_preferred_gap_mm: int | None = 180,
    max_preferred_gap_mm: int | None = 450,
    hard_orientation_gate: bool = False,
    orientation_hard_threshold_mm: int = 220,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {"used_shapely": True}
    repair_hints: List[Dict[str, Any]] = []
    pairwise_repair_hints: List[Dict[str, Any]] = []
    improve_hints: List[Dict[str, Any]] = []

    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except Exception:
        return {
            "result": "INVALID",
            "errors": [
                {
                    "code": "NO_SHAPELY",
                    "detail": "Shapely is required for GlobalClusterVerifier.",
                }
            ],
            "debug": debug,
            "quality": _empty_quality(),
        }

    grid_mm = int(grid_mm or 0)
    if margin_mm is None:
        margin_mm = max(grid_mm, 0)
    margin_mm = int(margin_mm)

    min_preferred_gap_mm, max_preferred_gap_mm = _normalize_gap_bounds(
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )
    orientation_hard_threshold_mm = int(orientation_hard_threshold_mm or 0)

    room_model_u = _unwrap_any(room_model)
    clusters_u = _unwrap_any(clusters_outlines)

    room_pts = (room_model_u.get("room") or {}).get("polygon_ccw") or []
    if not isinstance(room_pts, list) or len(room_pts) < 3:
        return {
            "result": "INVALID",
            "errors": [
                {"code": "ROOM_INVALID", "detail": "room.polygon_ccw missing/invalid"}
            ],
            "debug": debug,
            "quality": _empty_quality(),
        }

    room_poly = Polygon([(int(p["x"]), int(p["y"])) for p in room_pts])
    room_poly = _fix_poly(room_poly)
    if room_poly.is_empty or not room_poly.is_valid:
        return {
            "result": "INVALID",
            "errors": [
                {"code": "ROOM_INVALID", "detail": "room polygon invalid after fix"}
            ],
            "debug": debug,
            "quality": _empty_quality(),
        }

    room_prepared = prep(room_poly)
    room_bbox = room_poly.bounds
    room_area_mm2 = float(room_poly.area)
    room_diag_mm = _geom_diag_mm(room_poly)
    room_center = (float(room_poly.centroid.x), float(room_poly.centroid.y))

    debug["room_bbox"] = room_bbox
    debug["room_area_mm2"] = round(room_area_mm2, 2)
    debug["room_diag_mm"] = round(room_diag_mm, 2)

    opening_ctx = _build_opening_context(room_model_u)
    debug["doors_count"] = len(opening_ctx["doors"])
    debug["windows_count"] = len(opening_ctx["windows"])

    obstacles: List[Tuple[str, Any]] = []
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]] = {}

    for ob in room_model_u.get("obstacles") or []:
        if not isinstance(ob, dict):
            continue
        if not ob.get("hard", True):
            continue
        oid = str(ob.get("id", "obstacle"))
        otype = str(ob.get("type", "unknown"))
        pts = ob.get("polygon_ccw") or []
        if isinstance(pts, list) and len(pts) >= 3:
            poly = Polygon([(int(p["x"]), int(p["y"])) for p in pts])
            poly = _fix_poly(poly)
            if not poly.is_empty:
                obstacles.append((oid, poly))
                obstacle_by_type.setdefault(otype, []).append((oid, poly))

    obstacle_by_id: Dict[str, Any] = {oid: poly for oid, poly in obstacles}
    obstacle_union = unary_union([poly for _, poly in obstacles]) if obstacles else None
    free_room_area_mm2 = _compute_free_room_area_mm2(room_poly, obstacle_union)

    debug["obstacles_count"] = len(obstacles)
    debug["free_room_area_mm2"] = round(free_room_area_mm2, 2)

    tf_by_id: Dict[str, Dict[str, Any]] = {}
    for t in cluster_transforms:
        if not isinstance(t, dict):
            continue
        cid = t.get("cluster_id")
        if isinstance(cid, str) and cid:
            tf_by_id[cid] = {
                "cluster_id": cid,
                "x": int(t.get("x", 0) or 0),
                "y": int(t.get("y", 0) or 0),
                "rot": int(t.get("rot", 0) or 0) % 360,
            }

    cluster_entries = _iter_cluster_entries(clusters_u)
    if not cluster_entries:
        return {
            "result": "INVALID",
            "errors": [
                {"code": "CLUSTERS_INVALID", "detail": "No valid cluster entries found"}
            ],
            "debug": debug,
            "quality": _empty_quality(),
        }

    cinfo_by_id: Dict[str, Dict[str, Any]] = {
        cid: cinfo for cid, cinfo in cluster_entries
    }
    object_index = _build_global_object_index(cinfo_by_id)

    cluster_geoms: Dict[str, Any] = {}
    cluster_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    baseline_violations: Dict[str, Dict[str, Any]] = {}

    for cid, cinfo in cluster_entries:
        if cid not in tf_by_id:
            errors.append(
                {
                    "code": "MISSING_TRANSFORM",
                    "cluster_id": cid,
                    "detail": "No transform provided",
                }
            )
            continue

        t = tf_by_id[cid]
        x = int(t.get("x", 0))
        y = int(t.get("y", 0))
        rot = int(t.get("rot", 0)) % 360

        if rot not in (0, 90, 180, 270):
            errors.append(
                {
                    "code": "ROTATION_INVALID",
                    "cluster_id": cid,
                    "detail": f"rot={rot} must be 0/90/180/270",
                }
            )
            continue

        if grid_mm > 0 and (x % grid_mm != 0 or y % grid_mm != 0):
            errors.append(
                {
                    "code": "GRID_VIOLATION",
                    "cluster_id": cid,
                    "detail": f"x,y must be multiples of {grid_mm}",
                }
            )

        polys = _build_cluster_polys(Polygon, cinfo, x, y, rot)
        if not polys:
            errors.append(
                {
                    "code": "OUTLINE_INVALID",
                    "cluster_id": cid,
                    "detail": "No valid polygons built from rects/outline",
                }
            )
            continue

        geom = unary_union(polys)
        geom = _fix_geom(geom)
        if geom.is_empty or (hasattr(geom, "is_valid") and not geom.is_valid):
            errors.append(
                {
                    "code": "GEOM_INVALID",
                    "cluster_id": cid,
                    "detail": "Cluster geometry invalid after fix",
                }
            )
            continue

        cluster_geoms[cid] = geom
        bb = geom.bounds
        cluster_bboxes[cid] = bb
        debug.setdefault("cluster_bbox", {})[cid] = tuple(round(v, 2) for v in bb)

        out_bbox = (
            (bb[0] < room_bbox[0])
            or (bb[1] < room_bbox[1])
            or (bb[2] > room_bbox[2])
            or (bb[3] > room_bbox[3])
        )
        out_poly = False
        if not out_bbox:
            out_poly = not room_prepared.covers(geom)

        if out_bbox or out_poly:
            errors.append(
                {
                    "code": "CLUSTER_OUT_OF_BOUNDS",
                    "cluster_id": cid,
                    "detail": "Cluster not fully inside room",
                }
            )

        hit_ids: List[str] = []
        for oid, ob_poly in obstacles:
            if forbid_obstacle_touch:
                if geom.intersects(ob_poly):
                    hit_ids.append(oid)
            else:
                inter_area = float(geom.intersection(ob_poly).area)
                if inter_area > eps_area:
                    hit_ids.append(oid)

        if hit_ids:
            errors.append(
                {
                    "code": "CLUSTER_INTERSECTS_OBSTACLE",
                    "cluster_id": cid,
                    "obstacles": hit_ids,
                    "detail": f"Hits obstacles: {hit_ids}",
                }
            )

        baseline_violations[cid] = {
            "out_of_bounds": bool(out_bbox or out_poly),
            "hit_obstacles": hit_ids[:],
            "overlaps": [],
            "orientation_penalty_mm": 0,
        }

    cids = sorted(cluster_geoms.keys())
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a_id, b_id = cids[i], cids[j]
            ga, gb = cluster_geoms[a_id], cluster_geoms[b_id]

            a_bb = cluster_bboxes[a_id]
            b_bb = cluster_bboxes[b_id]
            if (
                (a_bb[2] <= b_bb[0])
                or (b_bb[2] <= a_bb[0])
                or (a_bb[3] <= b_bb[1])
                or (b_bb[3] <= a_bb[1])
            ):
                continue

            inter_area = float(ga.intersection(gb).area)
            if inter_area > eps_area:
                errors.append(
                    {
                        "code": "CLUSTER_OVERLAP",
                        "cluster_id": a_id,
                        "with": b_id,
                        "intersection_area_mm2": round(inter_area, 2),
                        "detail": f"Intersection area={inter_area:.2f} mm^2",
                    }
                )
                baseline_violations.setdefault(
                    a_id,
                    {
                        "out_of_bounds": False,
                        "hit_obstacles": [],
                        "overlaps": [],
                        "orientation_penalty_mm": 0,
                    },
                )["overlaps"].append(b_id)
                baseline_violations.setdefault(
                    b_id,
                    {
                        "out_of_bounds": False,
                        "hit_obstacles": [],
                        "overlaps": [],
                        "orientation_penalty_mm": 0,
                    },
                )["overlaps"].append(a_id)

    quality = _evaluate_layout_quality(
        cluster_geoms=cluster_geoms,
        cluster_transforms_by_id=tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        room_poly=room_poly,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        relation_plan=relation_plan,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )
    quality_out = _compact_quality_for_output(quality)

    orientation_hard_errors: List[Dict[str, Any]] = []
    if hard_orientation_gate and isinstance(relation_plan, dict):
        orientation_hard_errors = _build_orientation_hard_errors(
            quality=quality,
            threshold_mm=orientation_hard_threshold_mm,
        )
        errors.extend(orientation_hard_errors)
        if (
            generate_hints
            and orientation_hard_errors
            and isinstance(relation_plan, dict)
        ):
            pairwise_repair_hints = _generate_true_pairwise_orientation_repair_hints(
                orientation_hard_errors=orientation_hard_errors,
                all_errors=errors,
                tf_by_id=tf_by_id,
                cinfo_by_id=cinfo_by_id,
                object_index=object_index,
                cluster_geoms=cluster_geoms,
                room_poly=room_poly,
                room_prepared=room_prepared,
                room_bbox=room_bbox,
                room_center=room_center,
                room_diag_mm=room_diag_mm,
                free_room_area_mm2=free_room_area_mm2,
                forbid_obstacle_touch=forbid_obstacle_touch,
                eps_area=eps_area,
                obstacles=obstacles,
                opening_ctx=opening_ctx,
                obstacle_by_type=obstacle_by_type,
                relation_plan=relation_plan,
                base_preferred_gap_mm=base_preferred_gap_mm,
                min_preferred_gap_mm=min_preferred_gap_mm,
                max_preferred_gap_mm=max_preferred_gap_mm,
                grid_mm=grid_mm,
                consider_rotations=consider_rotations,
                max_candidates_per_issue=max_candidates_per_issue,
            )

        per_cluster_pen = _cluster_orientation_penalties(
            quality.get("orientation_debug")
        )
        for cid, pen in per_cluster_pen.items():
            baseline_violations.setdefault(
                cid,
                {
                    "out_of_bounds": False,
                    "hit_obstacles": [],
                    "overlaps": [],
                    "orientation_penalty_mm": 0,
                },
            )["orientation_penalty_mm"] = int(pen)

    result = "INVALID" if errors else "VALID"

    orientation_penalty_mm = int(quality.get("orientation_penalty_mm", 0) or 0)

    if (
        generate_hints
        and orientation_penalty_mm > 0
        and isinstance(relation_plan, dict)
    ):
        improve_hints = _generate_orientation_improve_hints(
            tf_by_id=tf_by_id,
            cinfo_by_id=cinfo_by_id,
            object_index=object_index,
            cluster_geoms=cluster_geoms,
            room_poly=room_poly,
            room_prepared=room_prepared,
            room_bbox=room_bbox,
            room_center=room_center,
            room_diag_mm=room_diag_mm,
            free_room_area_mm2=free_room_area_mm2,
            forbid_obstacle_touch=forbid_obstacle_touch,
            eps_area=eps_area,
            obstacles=obstacles,
            opening_ctx=opening_ctx,
            obstacle_by_type=obstacle_by_type,
            relation_plan=relation_plan,
            base_preferred_gap_mm=base_preferred_gap_mm,
            min_preferred_gap_mm=min_preferred_gap_mm,
            max_preferred_gap_mm=max_preferred_gap_mm,
            baseline_quality=quality,
            grid_mm=grid_mm,
            consider_rotations=consider_rotations,
        )

    if result == "INVALID" and generate_hints:
        issues = _collect_issues(errors)

        for issue in issues:
            cid = issue["cluster_id"]
            if cid not in tf_by_id or cid not in cinfo_by_id:
                continue

            base_t = tf_by_id[cid]
            base_x = int(base_t.get("x", 0))
            base_y = int(base_t.get("y", 0))
            base_rot = int(base_t.get("rot", 0)) % 360

            trigger = issue["trigger"]
            with_id = issue.get("with_id")
            issue_payload = (
                issue.get("issue_payload")
                if isinstance(issue.get("issue_payload"), dict)
                else None
            )

            if trigger == "ORIENTATION_HARD":
                candidates = _generate_candidates_for_orientation_issue(
                    cid=cid,
                    tf_by_id=tf_by_id,
                    cinfo_by_id=cinfo_by_id,
                    object_index=object_index,
                    base_x=base_x,
                    base_y=base_y,
                    base_rot=base_rot,
                    grid_mm=grid_mm,
                    consider_rotations=consider_rotations,
                    room_poly=room_poly,
                    room_prepared=room_prepared,
                    room_bbox=room_bbox,
                    room_center=room_center,
                    room_diag_mm=room_diag_mm,
                    free_room_area_mm2=free_room_area_mm2,
                    forbid_obstacle_touch=forbid_obstacle_touch,
                    eps_area=eps_area,
                    obstacles=obstacles,
                    cluster_geoms=cluster_geoms,
                    opening_ctx=opening_ctx,
                    obstacle_by_type=obstacle_by_type,
                    relation_plan=relation_plan,
                    base_preferred_gap_mm=base_preferred_gap_mm,
                    min_preferred_gap_mm=min_preferred_gap_mm,
                    max_preferred_gap_mm=max_preferred_gap_mm,
                    issue_payload=issue_payload,
                )
            else:
                candidates = _generate_candidates_for_issue(
                    cid=cid,
                    tf_by_id=tf_by_id,
                    cinfo_by_id=cinfo_by_id,
                    object_index=object_index,
                    trigger=trigger,
                    with_id=with_id,
                    base_x=base_x,
                    base_y=base_y,
                    base_rot=base_rot,
                    grid_mm=grid_mm,
                    nudge_steps=nudge_steps,
                    margin_mm=margin_mm,
                    consider_rotations=consider_rotations,
                    room_poly=room_poly,
                    room_prepared=room_prepared,
                    room_bbox=room_bbox,
                    room_center=room_center,
                    room_diag_mm=room_diag_mm,
                    free_room_area_mm2=free_room_area_mm2,
                    forbid_obstacle_touch=forbid_obstacle_touch,
                    eps_area=eps_area,
                    obstacles=obstacles,
                    obstacle_by_id=obstacle_by_id,
                    cluster_geoms=cluster_geoms,
                    cluster_bboxes=cluster_bboxes,
                    opening_ctx=opening_ctx,
                    obstacle_by_type=obstacle_by_type,
                    relation_plan=relation_plan,
                    base_preferred_gap_mm=base_preferred_gap_mm,
                    min_preferred_gap_mm=min_preferred_gap_mm,
                    max_preferred_gap_mm=max_preferred_gap_mm,
                )

            base_v = baseline_violations.get(
                cid,
                {
                    "out_of_bounds": False,
                    "hit_obstacles": [],
                    "overlaps": [],
                    "orientation_penalty_mm": 0,
                },
            )
            base_score = _violation_score(base_v)

            filtered: List[Dict[str, Any]] = []
            for cand in candidates:
                post = cand.get("postcheck") or {}
                hard_score = int(
                    post.get("hard_violation_score", post.get("violation_score", 10**9))
                    or 10**9
                )
                solves = bool(post.get("solves_trigger"))
                all_hard_clear = bool(post.get("all_hard_clear", False))
                remaining_hard_issues = int(
                    post.get("remaining_hard_issues", 10**9) or 10**9
                )
                cand_cluster_orient = int(
                    post.get("cluster_orientation_penalty_mm", 10**9) or 10**9
                )
                cand_global_orient = int(
                    post.get("global_orientation_penalty_mm", 10**9) or 10**9
                )

                if trigger == "ORIENTATION_HARD":
                    base_cluster_orient = int(
                        base_v.get("orientation_penalty_mm", 10**9) or 10**9
                    )
                    if hard_score == 0 and cand_cluster_orient < base_cluster_orient:
                        filtered.append(cand)
                else:
                    if (
                        all_hard_clear
                        or solves
                        or remaining_hard_issues < _hard_issue_count(base_v)
                        or hard_score < base_score
                    ):
                        filtered.append(cand)

            ranked = filtered if filtered else candidates
            ranked = sorted(
                ranked,
                key=lambda c: (
                    0 if c.get("postcheck", {}).get("all_hard_clear", False) else 1,
                    c.get("postcheck", {}).get("remaining_hard_issues", 10**9),
                    0 if c.get("postcheck", {}).get("solves_trigger") else 1,
                    c.get("postcheck", {}).get(
                        "hard_violation_score",
                        c.get("postcheck", {}).get("violation_score", 10**9),
                    ),
                    c.get("postcheck", {}).get("cluster_orientation_penalty_mm", 10**9),
                    c.get("postcheck", {}).get("global_orientation_penalty_mm", 10**9),
                    c.get("postcheck", {}).get("soft_penalty_mm", 10**9),
                    c.get("cost", 10**9),
                ),
            )[:max_candidates_per_issue]

            repair_hints.append(
                {
                    "cluster_id": cid,
                    "trigger": trigger,
                    "with_id": with_id,
                    "base_transform": {"x": base_x, "y": base_y, "rot": base_rot},
                    "candidates": ranked,
                }
            )

    out = {
        "result": result,
        "errors": errors,
        "debug": debug,
        "quality": quality_out,
    }
    if generate_hints:
        merged_repair_hints = []
        if isinstance(pairwise_repair_hints, list):
            merged_repair_hints.extend(pairwise_repair_hints)
        if isinstance(repair_hints, list):
            merged_repair_hints.extend(repair_hints)

        out["repair_hints"] = merged_repair_hints[:8]
        out["improve_hints"] = improve_hints[:6]
    return out


def _normalize_gap_bounds(
    *,
    min_preferred_gap_mm: int | None,
    max_preferred_gap_mm: int | None,
) -> Tuple[int, int]:
    if min_preferred_gap_mm is None:
        min_preferred_gap_mm = 180
    if max_preferred_gap_mm is None:
        max_preferred_gap_mm = 450

    min_preferred_gap_mm = int(min_preferred_gap_mm)
    max_preferred_gap_mm = int(max_preferred_gap_mm)

    if min_preferred_gap_mm < 0:
        min_preferred_gap_mm = 0
    if max_preferred_gap_mm < min_preferred_gap_mm:
        max_preferred_gap_mm = min_preferred_gap_mm

    return min_preferred_gap_mm, max_preferred_gap_mm


def _build_orientation_hard_errors(
    *,
    quality: Dict[str, Any],
    threshold_mm: int,
) -> List[Dict[str, Any]]:
    orientation_debug = quality.get("orientation_debug")
    if not isinstance(orientation_debug, list):
        return []

    errors: List[Dict[str, Any]] = []

    for item in orientation_debug:
        if not isinstance(item, dict):
            continue
        penalty = int(item.get("penalty_mm", 0) or 0)
        if penalty < threshold_mm:
            continue

        kind = str(item.get("kind") or "")
        if kind == "cluster_orientation":
            cid = item.get("cluster_id")
            if isinstance(cid, str) and cid:
                errors.append(
                    {
                        "code": "ORIENTATION_HARD",
                        "cluster_id": cid,
                        "detail": f"Cluster orientation mismatch too large for {cid}",
                        "kind": kind,
                        "intent": item.get("intent"),
                        "target_cluster_id": item.get("target_cluster_id"),
                        "penalty_mm": penalty,
                    }
                )

        elif kind == "object_orientation":
            cid = item.get("cluster_id")
            if isinstance(cid, str) and cid:
                errors.append(
                    {
                        "code": "ORIENTATION_HARD",
                        "cluster_id": cid,
                        "detail": f"Object orientation mismatch too large inside {cid}",
                        "kind": kind,
                        "object_id": item.get("object_id"),
                        "intent": item.get("intent"),
                        "target_object_id": item.get("target_object_id"),
                        "resolved_target_cluster_id": item.get(
                            "resolved_target_cluster_id"
                        ),
                        "penalty_mm": penalty,
                    }
                )

        elif kind == "cluster_directional_relation":
            a = item.get("a")
            b = item.get("b")
            rel = item.get("relation")
            for cid, other in ((a, b), (b, a)):
                if isinstance(cid, str) and cid:
                    errors.append(
                        {
                            "code": "ORIENTATION_HARD",
                            "cluster_id": cid,
                            "detail": f"Directional relation mismatch too large for {cid}",
                            "kind": kind,
                            "relation": rel,
                            "other_cluster_id": other,
                            "a": a,
                            "b": b,
                            "penalty_mm": penalty,
                        }
                    )

    dedup: List[Dict[str, Any]] = []
    seen = set()
    for e in errors:
        key = (
            e.get("cluster_id"),
            e.get("kind"),
            e.get("intent"),
            e.get("relation"),
            e.get("object_id"),
            e.get("target_object_id"),
            e.get("other_cluster_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    return dedup


def _cluster_orientation_penalties(orientation_debug: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(orientation_debug, list):
        return out

    for item in orientation_debug:
        if not isinstance(item, dict):
            continue
        pen = int(item.get("penalty_mm", 0) or 0)
        if pen <= 0:
            continue

        kind = str(item.get("kind") or "")
        if kind in {"cluster_orientation", "object_orientation"}:
            cid = item.get("cluster_id")
            if isinstance(cid, str) and cid:
                out[cid] = out.get(cid, 0) + pen
        elif kind == "cluster_directional_relation":
            a = item.get("a")
            b = item.get("b")
            if isinstance(a, str) and a:
                out[a] = out.get(a, 0) + pen
            if isinstance(b, str) and b:
                out[b] = out.get(b, 0) + pen
    return out


def _clusters_orientation_penalty_from_quality(
    quality: Dict[str, Any] | None,
    cluster_ids: List[str],
) -> int:
    if not isinstance(quality, dict):
        return 0
    pen_map = _cluster_orientation_penalties(quality.get("orientation_debug"))
    total = 0
    for cid in cluster_ids:
        total += int(pen_map.get(cid, 0) or 0)
    return total


def _canonicalize_patch_clusters(
    patch_clusters: Any,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(patch_clusters, list):
        return out

    seen = set()
    for item in patch_clusters:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        row = {
            "cluster_id": cid,
            "x": int(item.get("x", 0) or 0),
            "y": int(item.get("y", 0) or 0),
            "rot": int(item.get("rot", 0) or 0) % 360,
        }
        key = (row["cluster_id"], row["x"], row["y"], row["rot"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    out.sort(key=lambda r: r["cluster_id"])
    return out


def _limit_orientation_raw_candidates(
    raw: List[Tuple[int, int, int, str]],
    *,
    base_x: int,
    base_y: int,
    base_rot: int,
    max_items: int = 6,
) -> List[Tuple[int, int, int, str]]:
    out: List[Tuple[int, int, int, str]] = []
    seen = set()

    def add(x: int, y: int, rot: int, reason: str) -> None:
        key = (int(x), int(y), int(rot) % 360)
        if key in seen:
            return
        seen.add(key)
        out.append((int(x), int(y), int(rot) % 360, reason))

    add(base_x, base_y, base_rot, "KEEP_BASE")

    for x, y, rot, reason in raw:
        add(x, y, rot, reason)
        if len(out) >= max_items:
            break

    return out


def _collect_pairwise_orientation_issues_from_items(
    items: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []

    pair_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen_reason_keys = set()

    def ensure_pair(a: str, b: str) -> Dict[str, Any]:
        key = tuple(sorted((a, b)))
        if key not in pair_map:
            pair_map[key] = {
                "cluster_ids": [key[0], key[1]],
                "reasons_by_cluster": {key[0]: [], key[1]: []},
                "pair_penalty_mm": 0,
                "has_directional_relation": False,
            }
        return pair_map[key]

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("code") != "ORIENTATION_HARD":
            continue

        kind = str(item.get("kind") or "")
        penalty = int(item.get("penalty_mm", 0) or 0)

        if kind == "cluster_directional_relation":
            a = item.get("a")
            b = item.get("b")
            rel = item.get("relation")
            if isinstance(a, str) and a and isinstance(b, str) and b and a != b:
                reason_key = ("dir", tuple(sorted((a, b))), rel)
                if reason_key in seen_reason_keys:
                    continue
                seen_reason_keys.add(reason_key)

                rec = ensure_pair(a, b)

                ra = {
                    "kind": kind,
                    "relation": rel,
                    "other_cluster_id": b,
                    "a": a,
                    "b": b,
                    "penalty_mm": penalty,
                }
                rb = {
                    "kind": kind,
                    "relation": rel,
                    "other_cluster_id": a,
                    "a": a,
                    "b": b,
                    "penalty_mm": penalty,
                }

                rec["reasons_by_cluster"][a].append(ra)
                rec["reasons_by_cluster"][b].append(rb)
                rec["pair_penalty_mm"] += penalty
                rec["has_directional_relation"] = True

        elif kind == "cluster_orientation":
            cid = item.get("cluster_id")
            other = item.get("target_cluster_id")
            intent = item.get("intent")
            if (
                isinstance(cid, str)
                and cid
                and isinstance(other, str)
                and other
                and cid != other
            ):
                reason_key = ("cluster", cid, other, intent)
                if reason_key in seen_reason_keys:
                    continue
                seen_reason_keys.add(reason_key)

                rec = ensure_pair(cid, other)
                rec["reasons_by_cluster"][cid].append(
                    {
                        "kind": kind,
                        "intent": intent,
                        "target_cluster_id": other,
                        "penalty_mm": penalty,
                    }
                )
                rec["pair_penalty_mm"] += penalty

        elif kind == "object_orientation":
            cid = item.get("cluster_id")
            other = item.get("resolved_target_cluster_id")
            intent = item.get("intent")
            oid = item.get("object_id")
            toid = item.get("target_object_id")
            if (
                isinstance(cid, str)
                and cid
                and isinstance(other, str)
                and other
                and cid != other
            ):
                reason_key = ("object", cid, other, oid, toid, intent)
                if reason_key in seen_reason_keys:
                    continue
                seen_reason_keys.add(reason_key)

                rec = ensure_pair(cid, other)
                rec["reasons_by_cluster"][cid].append(
                    {
                        "kind": kind,
                        "object_id": oid,
                        "intent": intent,
                        "target_object_id": toid,
                        "resolved_target_cluster_id": other,
                        "penalty_mm": penalty,
                    }
                )
                rec["pair_penalty_mm"] += penalty

    out = list(pair_map.values())
    out.sort(
        key=lambda x: (
            0 if x.get("has_directional_relation") else 1,
            -int(x.get("pair_penalty_mm", 0) or 0),
            x["cluster_ids"][0],
            x["cluster_ids"][1],
        )
    )
    return out


def _collect_hard_issues_by_cluster(
    errors: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}

    def push(cid: str, trigger: str, with_id: str | None) -> None:
        if not isinstance(cid, str) or not cid:
            return
        out.setdefault(cid, []).append(
            {
                "cluster_id": cid,
                "trigger": trigger,
                "with_id": with_id,
            }
        )

    for e in errors:
        if not isinstance(e, dict):
            continue

        code = e.get("code")

        if code == "CLUSTER_OUT_OF_BOUNDS":
            cid = e.get("cluster_id")
            if isinstance(cid, str) and cid:
                push(cid, "CLUSTER_OUT_OF_BOUNDS", None)

        elif code == "CLUSTER_INTERSECTS_OBSTACLE":
            cid = e.get("cluster_id")
            obs_ids = e.get("obstacles")
            if not isinstance(obs_ids, list) or not obs_ids:
                obs_ids = [None]
            if isinstance(cid, str) and cid:
                for oid in obs_ids:
                    push(
                        cid,
                        "CLUSTER_INTERSECTS_OBSTACLE",
                        oid if isinstance(oid, str) else None,
                    )

        elif code == "CLUSTER_OVERLAP":
            a = e.get("cluster_id")
            b = e.get("with")
            if isinstance(a, str) and a and isinstance(b, str) and b and a != b:
                push(a, "CLUSTER_OVERLAP", b)
                push(b, "CLUSTER_OVERLAP", a)

    # dedup
    for cid, items in out.items():
        seen = set()
        dedup = []
        for it in items:
            key = (it.get("trigger"), it.get("with_id"))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(it)
        out[cid] = dedup

    return out


def _collect_issues(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    seen = set()

    for e in errors:
        code = e.get("code")
        cid = e.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue

        if code == "CLUSTER_INTERSECTS_OBSTACLE":
            obs_ids = e.get("obstacles")
            if not isinstance(obs_ids, list) or not obs_ids:
                obs_ids = [None]
            for oid in obs_ids:
                key = (cid, code, oid)
                if key in seen:
                    continue
                seen.add(key)
                issues.append(
                    {
                        "cluster_id": cid,
                        "trigger": "CLUSTER_INTERSECTS_OBSTACLE",
                        "with_id": oid,
                    }
                )

        elif code == "CLUSTER_OVERLAP":
            other = e.get("with")
            key = (cid, code, other)
            if key in seen:
                continue
            seen.add(key)
            issues.append(
                {"cluster_id": cid, "trigger": "CLUSTER_OVERLAP", "with_id": other}
            )

        elif code == "CLUSTER_OUT_OF_BOUNDS":
            key = (cid, code, None)
            if key in seen:
                continue
            seen.add(key)
            issues.append(
                {"cluster_id": cid, "trigger": "CLUSTER_OUT_OF_BOUNDS", "with_id": None}
            )

        elif code == "ORIENTATION_HARD":
            other_cluster_id = e.get("other_cluster_id")
            if not isinstance(other_cluster_id, str) or not other_cluster_id:
                a = e.get("a")
                b = e.get("b")
                if cid == a and isinstance(b, str) and b:
                    other_cluster_id = b
                elif cid == b and isinstance(a, str) and a:
                    other_cluster_id = a
                else:
                    other_cluster_id = None

            issue_payload = {
                "kind": e.get("kind"),
                "intent": e.get("intent"),
                "relation": e.get("relation"),
                "object_id": e.get("object_id"),
                "target_object_id": e.get("target_object_id"),
                "resolved_target_cluster_id": e.get("resolved_target_cluster_id"),
                "target_cluster_id": e.get("target_cluster_id"),
                "other_cluster_id": other_cluster_id,
                "a": e.get("a"),
                "b": e.get("b"),
                "penalty_mm": e.get("penalty_mm"),
            }

            key = (
                cid,
                code,
                issue_payload.get("kind"),
                issue_payload.get("intent"),
                issue_payload.get("relation"),
                issue_payload.get("object_id"),
                issue_payload.get("target_object_id"),
                issue_payload.get("other_cluster_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            issues.append(
                {
                    "cluster_id": cid,
                    "trigger": "ORIENTATION_HARD",
                    "with_id": None,
                    "issue_payload": issue_payload,
                }
            )

    return issues


def _build_pairwise_side_pool(
    *,
    cid: str,
    other_cid: str,
    base_state: Tuple[int, int, int],
    reasons: List[Dict[str, Any]],
    hard_issues: List[Dict[str, Any]],
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    cluster_geoms: Dict[str, Any],
    cluster_bboxes: Dict[str, Tuple[float, float, float, float]],
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    obstacles: List[Tuple[str, Any]],
    obstacle_by_id: Dict[str, Any],
    relation_plan: Dict[str, Any] | None,
    forbid_obstacle_touch: bool,
    eps_area: float,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
    grid_mm: int,
    consider_rotations: bool,
    max_items: int = 14,
) -> List[Tuple[int, int, int, str]]:
    base_x, base_y, base_rot = base_state
    seen = set()
    pool: List[Tuple[int, int, int, str]] = []

    def add(x: int, y: int, rot: int, reason: str) -> None:
        if grid_mm > 0:
            x = _snap_grid(int(x), grid_mm)
            y = _snap_grid(int(y), grid_mm)
        rot = int(rot) % 360
        key = (x, y, rot)
        if key in seen:
            return
        seen.add(key)
        pool.append((x, y, rot, reason))

    add(base_x, base_y, base_rot, "KEEP_BASE")

    # A. orientation-local candidates
    raw = _build_orientation_candidates_for_cluster(
        cid=cid,
        tf_by_id=tf_by_id,
        cluster_geoms=cluster_geoms,
        room_poly=room_poly,
        room_center=room_center,
        room_bbox=room_bbox,
        opening_ctx=opening_ctx,
        reasons=reasons,
        base_x=base_x,
        base_y=base_y,
        base_rot=base_rot,
        grid_mm=grid_mm,
        consider_rotations=consider_rotations,
    )
    for x, y, rot, reason in raw:
        add(x, y, rot, f"PAIR_ORIENT::{reason}")

    # B. aggressive directional moves toward/away other + center
    own_geom = cluster_geoms.get(cid)
    other_geom = cluster_geoms.get(other_cid)
    if own_geom is not None and other_geom is not None and grid_mm > 0:
        own_cent = (float(own_geom.centroid.x), float(own_geom.centroid.y))
        other_cent = (float(other_geom.centroid.x), float(other_geom.centroid.y))

        vecs = [
            (_vec_from_to(own_cent, other_cent), f"TOWARD_{other_cid}"),
            (_vec_from_to(other_cent, own_cent), f"AWAY_{other_cid}"),
            (_vec_from_to(own_cent, room_center), "TOWARD_CENTER"),
        ]

        for vec, tag in vecs:
            step = _dominant_step_from_vector(vec, grid_mm)
            sx, sy = step
            if sx == 0 and sy == 0:
                continue
            for k in (1, 2, 4, 8, 12):
                dx = sx * k
                dy = sy * k
                add(base_x + dx, base_y + dy, base_rot, f"{tag}_{k}g")
                if consider_rotations:
                    for rot in (0, 90, 180, 270):
                        add(base_x + dx, base_y + dy, rot, f"{tag}_{k}g_ROT_{rot}")

    # C. geometry repair candidates hút từ overlap/obstacle/OOB hiện tại
    for issue in hard_issues[:4]:
        trigger = issue.get("trigger")
        with_id = issue.get("with_id")

        geom_cands = _generate_candidates_for_issue(
            cid=cid,
            tf_by_id=tf_by_id,
            cinfo_by_id=cinfo_by_id,
            object_index=object_index,
            trigger=trigger,
            with_id=with_id,
            base_x=base_x,
            base_y=base_y,
            base_rot=base_rot,
            grid_mm=grid_mm,
            nudge_steps=3,
            margin_mm=max(grid_mm, 0),
            consider_rotations=consider_rotations,
            room_poly=room_poly,
            room_prepared=room_prepared,
            room_bbox=room_bbox,
            room_center=room_center,
            room_diag_mm=room_diag_mm,
            free_room_area_mm2=free_room_area_mm2,
            forbid_obstacle_touch=forbid_obstacle_touch,
            eps_area=eps_area,
            obstacles=obstacles,
            obstacle_by_id=obstacle_by_id,
            cluster_geoms=cluster_geoms,
            cluster_bboxes=cluster_bboxes,
            opening_ctx=opening_ctx,
            obstacle_by_type=obstacle_by_type,
            relation_plan=relation_plan,
            base_preferred_gap_mm=base_preferred_gap_mm,
            min_preferred_gap_mm=min_preferred_gap_mm,
            max_preferred_gap_mm=max_preferred_gap_mm,
        )

        geom_cands.sort(
            key=lambda c: (
                0 if c.get("postcheck", {}).get("all_hard_clear", False) else 1,
                c.get("postcheck", {}).get("remaining_hard_issues", 10**9),
                c.get("postcheck", {}).get("hard_violation_score", 10**9),
                c.get("cost", 10**9),
            )
        )

        for cand in geom_cands[:4]:
            x = cand.get("x")
            y = cand.get("y")
            rot = cand.get("rot")
            if all(isinstance(v, (int, float)) for v in (x, y, rot)):
                add(
                    int(x),
                    int(y),
                    int(rot) % 360,
                    f"PAIR_HARD::{trigger}::{cand.get('reason')}",
                )

    pool.sort(
        key=lambda t: (
            abs(t[0] - base_x) + abs(t[1] - base_y) + (0 if t[2] == base_rot else 500),
            t[3],
        )
    )
    return pool[:max_items]


def _collect_orientation_targets_with_reasons(
    orientation_debug: Any,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(orientation_debug, list):
        return out

    for item in orientation_debug:
        if not isinstance(item, dict):
            continue

        penalty = int(item.get("penalty_mm", 0) or 0)
        if penalty <= 0:
            continue

        kind = str(item.get("kind") or "")

        if kind == "cluster_orientation":
            cid = item.get("cluster_id")
            if isinstance(cid, str) and cid:
                rec = out.setdefault(cid, {"penalty": 0, "reasons": []})
                rec["penalty"] += penalty
                rec["reasons"].append(
                    {
                        "kind": kind,
                        "intent": item.get("intent"),
                        "target_cluster_id": item.get("target_cluster_id"),
                    }
                )

        elif kind == "object_orientation":
            cid = item.get("cluster_id")
            if isinstance(cid, str) and cid:
                rec = out.setdefault(cid, {"penalty": 0, "reasons": []})
                rec["penalty"] += penalty
                rec["reasons"].append(
                    {
                        "kind": kind,
                        "object_id": item.get("object_id"),
                        "intent": item.get("intent"),
                        "target_object_id": item.get("target_object_id"),
                        "resolved_target_cluster_id": item.get(
                            "resolved_target_cluster_id"
                        ),
                    }
                )

        elif kind == "cluster_directional_relation":
            a = item.get("a")
            b = item.get("b")
            relation = item.get("relation")
            for cid, other in ((a, b), (b, a)):
                if isinstance(cid, str) and cid:
                    rec = out.setdefault(cid, {"penalty": 0, "reasons": []})
                    rec["penalty"] += penalty
                    rec["reasons"].append(
                        {
                            "kind": kind,
                            "relation": relation,
                            "other_cluster_id": other,
                            "a": a,
                            "b": b,
                        }
                    )

    return out


def _generate_candidates_for_issue(
    *,
    cid: str,
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    trigger: str,
    with_id: str | None,
    base_x: int,
    base_y: int,
    base_rot: int,
    grid_mm: int,
    nudge_steps: int,
    margin_mm: int,
    consider_rotations: bool,
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    obstacle_by_id: Dict[str, Any],
    cluster_geoms: Dict[str, Any],
    cluster_bboxes: Dict[str, Tuple[float, float, float, float]],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[int, int, int, str]] = []
    seen = set()

    def add_candidate(x: int, y: int, rot: int, reason: str) -> None:
        if grid_mm > 0:
            x = _snap_grid(x, grid_mm)
            y = _snap_grid(y, grid_mm)
        key = (x, y, rot)
        if key in seen:
            return
        seen.add(key)
        candidates.append((x, y, rot, reason))

    base_bb = cluster_bboxes.get(cid)
    cinfo = cinfo_by_id[cid]

    rotations_to_try = [base_rot]
    if consider_rotations:
        for r in (0, 90, 180, 270):
            if r not in rotations_to_try:
                rotations_to_try.append(r)

    for k in range(1, max(1, int(nudge_steps)) + 1):
        step = k * grid_mm
        add_candidate(base_x + step, base_y, base_rot, f"NUDGE_RIGHT_{k}g")
        add_candidate(base_x - step, base_y, base_rot, f"NUDGE_LEFT_{k}g")
        add_candidate(base_x, base_y + step, base_rot, f"NUDGE_UP_{k}g")
        add_candidate(base_x, base_y - step, base_rot, f"NUDGE_DOWN_{k}g")

    if base_bb is not None:
        blocker_bb = None

        if (
            trigger == "CLUSTER_INTERSECTS_OBSTACLE"
            and with_id
            and with_id in obstacle_by_id
        ):
            blocker_bb = obstacle_by_id[with_id].bounds
        elif trigger == "CLUSTER_OVERLAP" and with_id and with_id in cluster_bboxes:
            blocker_bb = cluster_bboxes[with_id]

        if blocker_bb is not None:
            cx1, cy1, cx2, cy2 = base_bb
            bx1, by1, bx2, by2 = blocker_bb

            dx_right = int((bx2 - cx1) + margin_mm)
            dx_left = int(-((cx2 - bx1) + margin_mm))
            dy_up = int((by2 - cy1) + margin_mm)
            dy_down = int(-((cy2 - by1) + margin_mm))

            if dx_right != 0:
                add_candidate(
                    base_x + _snap_grid(dx_right, grid_mm),
                    base_y,
                    base_rot,
                    "AXIS_CLEAR_RIGHT",
                )
            if dx_left != 0:
                add_candidate(
                    base_x + _snap_grid(dx_left, grid_mm),
                    base_y,
                    base_rot,
                    "AXIS_CLEAR_LEFT",
                )
            if dy_up != 0:
                add_candidate(
                    base_x,
                    base_y + _snap_grid(dy_up, grid_mm),
                    base_rot,
                    "AXIS_CLEAR_UP",
                )
            if dy_down != 0:
                add_candidate(
                    base_x,
                    base_y + _snap_grid(dy_down, grid_mm),
                    base_rot,
                    "AXIS_CLEAR_DOWN",
                )

            for rot in rotations_to_try:
                bb = _cluster_bbox_for_transform(cinfo, base_x, base_y, rot)
                if bb is None:
                    continue
                rx1, ry1, rx2, ry2 = bb

                shift_right = int((bx2 - rx1) + margin_mm)
                shift_left = int(-((rx2 - bx1) + margin_mm))
                shift_up = int((by2 - ry1) + margin_mm)
                shift_down = int(-((ry2 - by1) + margin_mm))

                add_candidate(
                    base_x + _snap_grid(shift_right, grid_mm),
                    base_y,
                    rot,
                    f"BLOCKER_CLEAR_RIGHT_ROT_{rot}",
                )
                add_candidate(
                    base_x + _snap_grid(shift_left, grid_mm),
                    base_y,
                    rot,
                    f"BLOCKER_CLEAR_LEFT_ROT_{rot}",
                )
                add_candidate(
                    base_x,
                    base_y + _snap_grid(shift_up, grid_mm),
                    rot,
                    f"BLOCKER_CLEAR_UP_ROT_{rot}",
                )
                add_candidate(
                    base_x,
                    base_y + _snap_grid(shift_down, grid_mm),
                    rot,
                    f"BLOCKER_CLEAR_DOWN_ROT_{rot}",
                )

        if trigger == "CLUSTER_OUT_OF_BOUNDS":
            cx1, cy1, cx2, cy2 = base_bb
            rx1, ry1, rx2, ry2 = room_bbox

            dx = 0
            dy = 0
            if cx1 < rx1:
                dx = int((rx1 - cx1) + margin_mm)
            elif cx2 > rx2:
                dx = int((rx2 - cx2) - margin_mm)

            if cy1 < ry1:
                dy = int((ry1 - cy1) + margin_mm)
            elif cy2 > ry2:
                dy = int((ry2 - cy2) - margin_mm)

            if dx != 0:
                add_candidate(
                    base_x + _snap_grid(dx, grid_mm), base_y, base_rot, "OOB_CLEAR_X"
                )
            if dy != 0:
                add_candidate(
                    base_x, base_y + _snap_grid(dy, grid_mm), base_rot, "OOB_CLEAR_Y"
                )
            if dx != 0 and dy != 0:
                add_candidate(
                    base_x + _snap_grid(dx, grid_mm),
                    base_y + _snap_grid(dy, grid_mm),
                    base_rot,
                    "OOB_CLEAR_XY",
                )

    if consider_rotations:
        for rot in (0, 90, 180, 270):
            if rot == base_rot:
                continue
            add_candidate(base_x, base_y, rot, f"ROTATE_{rot}")
            add_candidate(base_x + grid_mm, base_y, rot, f"ROTATE_{rot}_NUDGE_RIGHT")
            add_candidate(base_x - grid_mm, base_y, rot, f"ROTATE_{rot}_NUDGE_LEFT")
            add_candidate(base_x, base_y + grid_mm, rot, f"ROTATE_{rot}_NUDGE_UP")
            add_candidate(base_x, base_y - grid_mm, rot, f"ROTATE_{rot}_NUDGE_DOWN")

    rx1, ry1, rx2, ry2 = room_bbox
    corner_margin = max(margin_mm, grid_mm)

    for rot in rotations_to_try:
        bb_rot = _cluster_bbox_for_transform(cinfo, base_x, base_y, rot)
        if bb_rot is None:
            continue
        bw = int(round(bb_rot[2] - bb_rot[0]))
        bh = int(round(bb_rot[3] - bb_rot[1]))
        cur_min_x = int(round(bb_rot[0]))
        cur_min_y = int(round(bb_rot[1]))

        def add_bbox_reloc(target_min_x: int, target_min_y: int, reason: str) -> None:
            dx = target_min_x - cur_min_x
            dy = target_min_y - cur_min_y
            add_candidate(base_x + dx, base_y + dy, rot, reason)

        cx = int(round(room_center[0] - bw / 2))
        cy = int(round(room_center[1] - bh / 2))

        relocate_targets = [
            (
                int(rx1 + corner_margin),
                int(ry1 + corner_margin),
                f"RELOCATE_BOTTOM_LEFT_ROT_{rot}",
            ),
            (
                int(rx2 - bw - corner_margin),
                int(ry1 + corner_margin),
                f"RELOCATE_BOTTOM_RIGHT_ROT_{rot}",
            ),
            (
                int(rx1 + corner_margin),
                int(ry2 - bh - corner_margin),
                f"RELOCATE_TOP_LEFT_ROT_{rot}",
            ),
            (
                int(rx2 - bw - corner_margin),
                int(ry2 - bh - corner_margin),
                f"RELOCATE_TOP_RIGHT_ROT_{rot}",
            ),
            (cx, cy, f"RELOCATE_CENTER_ROT_{rot}"),
            (int(rx1 + corner_margin), cy, f"RELOCATE_LEFT_MID_ROT_{rot}"),
            (int(rx2 - bw - corner_margin), cy, f"RELOCATE_RIGHT_MID_ROT_{rot}"),
            (cx, int(ry1 + corner_margin), f"RELOCATE_BOTTOM_MID_ROT_{rot}"),
            (cx, int(ry2 - bh - corner_margin), f"RELOCATE_TOP_MID_ROT_{rot}"),
        ]
        for tx, ty, reason in relocate_targets:
            add_bbox_reloc(tx, ty, reason)

    evaluated: List[Dict[str, Any]] = []
    for cand_x, cand_y, cand_rot, reason in candidates:
        post = _postcheck_candidate(
            cid=cid,
            tf_by_id=tf_by_id,
            cinfo_by_id=cinfo_by_id,
            object_index=object_index,
            trigger=trigger,
            with_id=with_id,
            cand_x=cand_x,
            cand_y=cand_y,
            cand_rot=cand_rot,
            room_poly=room_poly,
            room_prepared=room_prepared,
            room_bbox=room_bbox,
            room_center=room_center,
            room_diag_mm=room_diag_mm,
            free_room_area_mm2=free_room_area_mm2,
            forbid_obstacle_touch=forbid_obstacle_touch,
            eps_area=eps_area,
            obstacles=obstacles,
            cluster_geoms=cluster_geoms,
            cinfo=cinfo_by_id[cid],
            opening_ctx=opening_ctx,
            obstacle_by_type=obstacle_by_type,
            relation_plan=relation_plan,
            base_preferred_gap_mm=base_preferred_gap_mm,
            min_preferred_gap_mm=min_preferred_gap_mm,
            max_preferred_gap_mm=max_preferred_gap_mm,
        )

        dx = cand_x - base_x
        dy = cand_y - base_y
        cost = abs(dx) + abs(dy) + (0 if cand_rot == base_rot else 500)

        evaluated.append(
            {
                "x": cand_x,
                "y": cand_y,
                "rot": cand_rot,
                "dx": dx,
                "dy": dy,
                "reason": reason,
                "cost": cost,
                "postcheck": _compact_postcheck(post),
            }
        )

    return evaluated


def _generate_candidates_for_orientation_issue(
    *,
    cid: str,
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    base_x: int,
    base_y: int,
    base_rot: int,
    grid_mm: int,
    consider_rotations: bool,
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    cluster_geoms: Dict[str, Any],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
    issue_payload: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    candidates_raw: List[Tuple[int, int, int, str]] = []
    seen = set()

    def add_candidate(x: int, y: int, rot: int, reason: str) -> None:
        if grid_mm > 0:
            x = _snap_grid(x, grid_mm)
            y = _snap_grid(y, grid_mm)
        key = (x, y, rot)
        if key in seen:
            return
        seen.add(key)
        candidates_raw.append((x, y, rot, reason))

    # 1) smart candidates theo offender cụ thể
    smart_reasons: List[Dict[str, Any]] = []
    if isinstance(issue_payload, dict):
        smart_reasons.append(issue_payload)

    if smart_reasons:
        smart = _build_orientation_candidates_for_cluster(
            cid=cid,
            tf_by_id=tf_by_id,
            cluster_geoms=cluster_geoms,
            room_poly=room_poly,
            room_center=room_center,
            room_bbox=room_bbox,
            opening_ctx=opening_ctx,
            reasons=smart_reasons,
            base_x=base_x,
            base_y=base_y,
            base_rot=base_rot,
            grid_mm=grid_mm,
            consider_rotations=consider_rotations,
        )
        for x, y, rot, reason in smart:
            add_candidate(x, y, rot, reason)

    # 2) fallback generic candidates
    if consider_rotations:
        for rot in (0, 90, 180, 270):
            if rot != base_rot:
                add_candidate(base_x, base_y, rot, f"ROTATE_{rot}_FOR_HARD_ORIENTATION")

    for step in (1, 2):
        s = step * grid_mm
        add_candidate(base_x + s, base_y, base_rot, f"HARD_ORIENT_NUDGE_RIGHT_{step}g")
        add_candidate(base_x - s, base_y, base_rot, f"HARD_ORIENT_NUDGE_LEFT_{step}g")
        add_candidate(base_x, base_y + s, base_rot, f"HARD_ORIENT_NUDGE_UP_{step}g")
        add_candidate(base_x, base_y - s, base_rot, f"HARD_ORIENT_NUDGE_DOWN_{step}g")

        if consider_rotations:
            for rot in (0, 90, 180, 270):
                if rot == base_rot:
                    continue
                add_candidate(
                    base_x + s,
                    base_y,
                    rot,
                    f"ROTATE_{rot}_NUDGE_RIGHT_{step}g_FOR_HARD_ORIENTATION",
                )
                add_candidate(
                    base_x - s,
                    base_y,
                    rot,
                    f"ROTATE_{rot}_NUDGE_LEFT_{step}g_FOR_HARD_ORIENTATION",
                )
                add_candidate(
                    base_x,
                    base_y + s,
                    rot,
                    f"ROTATE_{rot}_NUDGE_UP_{step}g_FOR_HARD_ORIENTATION",
                )
                add_candidate(
                    base_x,
                    base_y - s,
                    rot,
                    f"ROTATE_{rot}_NUDGE_DOWN_{step}g_FOR_HARD_ORIENTATION",
                )

    base_quality = _evaluate_layout_quality(
        cluster_geoms=cluster_geoms,
        cluster_transforms_by_id=tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        room_poly=room_poly,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        relation_plan=relation_plan,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )
    base_cluster_orientation_pen = _cluster_orientation_penalty_from_quality(
        base_quality, cid
    )
    base_global_orientation_pen = int(
        base_quality.get("orientation_penalty_mm", 0) or 0
    )

    evaluated: List[Dict[str, Any]] = []
    for cand_x, cand_y, cand_rot, reason in candidates_raw:
        post = _postcheck_candidate(
            cid=cid,
            tf_by_id=tf_by_id,
            cinfo_by_id=cinfo_by_id,
            object_index=object_index,
            trigger="ORIENTATION_HARD",
            with_id=None,
            cand_x=cand_x,
            cand_y=cand_y,
            cand_rot=cand_rot,
            room_poly=room_poly,
            room_prepared=room_prepared,
            room_bbox=room_bbox,
            room_center=room_center,
            room_diag_mm=room_diag_mm,
            free_room_area_mm2=free_room_area_mm2,
            forbid_obstacle_touch=forbid_obstacle_touch,
            eps_area=eps_area,
            obstacles=obstacles,
            cluster_geoms=cluster_geoms,
            cinfo=cinfo_by_id[cid],
            opening_ctx=opening_ctx,
            obstacle_by_type=obstacle_by_type,
            relation_plan=relation_plan,
            base_preferred_gap_mm=base_preferred_gap_mm,
            min_preferred_gap_mm=min_preferred_gap_mm,
            max_preferred_gap_mm=max_preferred_gap_mm,
        )

        hard_score = int(post.get("hard_violation_score", 10**9) or 10**9)
        cand_cluster_orientation_pen = int(
            post.get("cluster_orientation_penalty_mm", 10**9) or 10**9
        )
        cand_global_orientation_pen = int(
            post.get("global_orientation_penalty_mm", 10**9) or 10**9
        )

        if hard_score != 0:
            continue

        improved = cand_cluster_orientation_pen < base_cluster_orientation_pen or (
            cand_cluster_orientation_pen == base_cluster_orientation_pen
            and cand_global_orientation_pen < base_global_orientation_pen
        )
        if not improved:
            continue

        dx = cand_x - base_x
        dy = cand_y - base_y
        cost = abs(dx) + abs(dy) + (0 if cand_rot == base_rot else 500)

        evaluated.append(
            {
                "x": cand_x,
                "y": cand_y,
                "rot": cand_rot,
                "dx": dx,
                "dy": dy,
                "reason": reason,
                "cost": cost,
                "cluster_orientation_penalty_mm": cand_cluster_orientation_pen,
                "global_orientation_penalty_mm": cand_global_orientation_pen,
                "postcheck": _compact_postcheck(post),
            }
        )

    evaluated.sort(
        key=lambda c: (
            c.get("cluster_orientation_penalty_mm", 10**9),
            c.get("global_orientation_penalty_mm", 10**9),
            c.get("postcheck", {}).get("soft_penalty_mm", 10**9),
            c.get("cost", 10**9),
        )
    )
    return evaluated


def _generate_pairwise_orientation_candidates_for_issue(
    *,
    cluster_ids: List[str],
    reasons_by_cluster: Dict[str, List[Dict[str, Any]]],
    hard_issues_by_cluster: Dict[str, List[Dict[str, Any]]],
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    grid_mm: int,
    consider_rotations: bool,
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    cluster_geoms: Dict[str, Any],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
    max_side_candidates: int = 14,
    max_return: int = 8,
) -> List[Dict[str, Any]]:
    if len(cluster_ids) != 2:
        return []

    a, b = cluster_ids[0], cluster_ids[1]
    if a not in tf_by_id or b not in tf_by_id:
        return []

    obstacle_by_id = {oid: poly for oid, poly in obstacles}
    cluster_bboxes = {cid: geom.bounds for cid, geom in cluster_geoms.items()}

    base_quality = _evaluate_layout_quality(
        cluster_geoms=cluster_geoms,
        cluster_transforms_by_id=tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        room_poly=room_poly,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        relation_plan=relation_plan,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )
    base_pair_orientation_pen = _clusters_orientation_penalty_from_quality(
        base_quality, [a, b]
    )
    base_global_orientation_pen = int(
        base_quality.get("orientation_penalty_mm", 0) or 0
    )
    base_soft_pen = int(base_quality.get("total_soft_penalty_mm", 10**9) or 10**9)

    ta = tf_by_id[a]
    tb = tf_by_id[b]
    base_state_a = (
        int(ta.get("x", 0)),
        int(ta.get("y", 0)),
        int(ta.get("rot", 0)) % 360,
    )
    base_state_b = (
        int(tb.get("x", 0)),
        int(tb.get("y", 0)),
        int(tb.get("rot", 0)) % 360,
    )

    base_post = _postcheck_pair_candidate(
        cluster_ids=[a, b],
        patch_clusters=[
            {
                "cluster_id": a,
                "x": base_state_a[0],
                "y": base_state_a[1],
                "rot": base_state_a[2],
            },
            {
                "cluster_id": b,
                "x": base_state_b[0],
                "y": base_state_b[1],
                "rot": base_state_b[2],
            },
        ],
        tf_by_id=tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        room_poly=room_poly,
        room_prepared=room_prepared,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        forbid_obstacle_touch=forbid_obstacle_touch,
        eps_area=eps_area,
        obstacles=obstacles,
        cluster_geoms=cluster_geoms,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        relation_plan=relation_plan,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )
    base_hard_score = int(base_post.get("hard_violation_score", 10**9) or 10**9)

    pool_a = _build_pairwise_side_pool(
        cid=a,
        other_cid=b,
        base_state=base_state_a,
        reasons=reasons_by_cluster.get(a, []),
        hard_issues=hard_issues_by_cluster.get(a, []),
        tf_by_id=tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        cluster_geoms=cluster_geoms,
        cluster_bboxes=cluster_bboxes,
        room_poly=room_poly,
        room_prepared=room_prepared,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        obstacles=obstacles,
        obstacle_by_id=obstacle_by_id,
        relation_plan=relation_plan,
        forbid_obstacle_touch=forbid_obstacle_touch,
        eps_area=eps_area,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
        grid_mm=grid_mm,
        consider_rotations=consider_rotations,
        max_items=max_side_candidates,
    )

    pool_b = _build_pairwise_side_pool(
        cid=b,
        other_cid=a,
        base_state=base_state_b,
        reasons=reasons_by_cluster.get(b, []),
        hard_issues=hard_issues_by_cluster.get(b, []),
        tf_by_id=tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        cluster_geoms=cluster_geoms,
        cluster_bboxes=cluster_bboxes,
        room_poly=room_poly,
        room_prepared=room_prepared,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        obstacles=obstacles,
        obstacle_by_id=obstacle_by_id,
        relation_plan=relation_plan,
        forbid_obstacle_touch=forbid_obstacle_touch,
        eps_area=eps_area,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
        grid_mm=grid_mm,
        consider_rotations=consider_rotations,
        max_items=max_side_candidates,
    )

    strict_improved: List[Dict[str, Any]] = []
    hard_clear_fallback: List[Dict[str, Any]] = []
    hard_reduce_fallback: List[Dict[str, Any]] = []

    for ax, ay, arot, areason in pool_a:
        for bx, by, brot, breason in pool_b:
            if (ax, ay, arot) == base_state_a and (bx, by, brot) == base_state_b:
                continue

            patch_clusters = _canonicalize_patch_clusters(
                [
                    {"cluster_id": a, "x": ax, "y": ay, "rot": arot},
                    {"cluster_id": b, "x": bx, "y": by, "rot": brot},
                ]
            )

            post = _postcheck_pair_candidate(
                cluster_ids=[a, b],
                patch_clusters=patch_clusters,
                tf_by_id=tf_by_id,
                cinfo_by_id=cinfo_by_id,
                object_index=object_index,
                room_poly=room_poly,
                room_prepared=room_prepared,
                room_bbox=room_bbox,
                room_center=room_center,
                room_diag_mm=room_diag_mm,
                free_room_area_mm2=free_room_area_mm2,
                forbid_obstacle_touch=forbid_obstacle_touch,
                eps_area=eps_area,
                obstacles=obstacles,
                cluster_geoms=cluster_geoms,
                opening_ctx=opening_ctx,
                obstacle_by_type=obstacle_by_type,
                relation_plan=relation_plan,
                base_preferred_gap_mm=base_preferred_gap_mm,
                min_preferred_gap_mm=min_preferred_gap_mm,
                max_preferred_gap_mm=max_preferred_gap_mm,
            )

            hard_score = int(post.get("hard_violation_score", 10**9) or 10**9)
            pair_cluster_orientation_pen = int(
                post.get("cluster_orientation_penalty_mm", 10**9) or 10**9
            )
            global_orientation_pen = int(
                post.get("global_orientation_penalty_mm", 10**9) or 10**9
            )
            soft_pen = int(post.get("soft_penalty_mm", 10**9) or 10**9)

            cost = (
                abs(ax - base_state_a[0])
                + abs(ay - base_state_a[1])
                + (0 if arot == base_state_a[2] else 500)
                + abs(bx - base_state_b[0])
                + abs(by - base_state_b[1])
                + (0 if brot == base_state_b[2] else 500)
            )

            item = {
                "patch_clusters": patch_clusters,
                "reason": f"PAIRWISE::{a}:{areason}__{b}:{breason}",
                "cost": int(cost),
                "postcheck": _compact_postcheck(post),
                "cluster_orientation_penalty_mm": int(pair_cluster_orientation_pen),
                "global_orientation_penalty_mm": int(global_orientation_pen),
            }

            improved = (
                pair_cluster_orientation_pen < base_pair_orientation_pen
                or (
                    pair_cluster_orientation_pen == base_pair_orientation_pen
                    and global_orientation_pen < base_global_orientation_pen
                )
                or (
                    pair_cluster_orientation_pen == base_pair_orientation_pen
                    and global_orientation_pen == base_global_orientation_pen
                    and soft_pen < base_soft_pen
                )
            )

            if hard_score == 0 and improved:
                strict_improved.append(item)
            elif hard_score == 0:
                hard_clear_fallback.append(item)
            elif hard_score < base_hard_score:
                hard_reduce_fallback.append(item)

    def sort_key(c: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            c.get("cluster_orientation_penalty_mm", 10**9),
            c.get("global_orientation_penalty_mm", 10**9),
            c.get("postcheck", {}).get("soft_penalty_mm", 10**9),
            c.get("postcheck", {}).get("hard_violation_score", 10**9),
            c.get("cost", 10**9),
            c.get("reason", ""),
        )

    strict_improved.sort(key=sort_key)
    hard_clear_fallback.sort(key=sort_key)
    hard_reduce_fallback.sort(key=sort_key)

    if strict_improved:
        return strict_improved[:max_return]
    if hard_clear_fallback:
        return hard_clear_fallback[:max_return]
    return hard_reduce_fallback[:max_return]


def _generate_true_pairwise_orientation_repair_hints(
    *,
    orientation_hard_errors: List[Dict[str, Any]],
    all_errors: List[Dict[str, Any]],
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    cluster_geoms: Dict[str, Any],
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
    grid_mm: int,
    consider_rotations: bool,
    max_candidates_per_issue: int,
) -> List[Dict[str, Any]]:
    issues = _collect_pairwise_orientation_issues_from_items(orientation_hard_errors)
    hard_issues_by_cluster = _collect_hard_issues_by_cluster(all_errors)

    hints: List[Dict[str, Any]] = []

    for issue in issues[:6]:
        cluster_ids = issue.get("cluster_ids")
        reasons_by_cluster = issue.get("reasons_by_cluster")
        if not isinstance(cluster_ids, list) or len(cluster_ids) != 2:
            continue
        if not isinstance(reasons_by_cluster, dict):
            continue

        a, b = cluster_ids[0], cluster_ids[1]
        if a not in tf_by_id or b not in tf_by_id:
            continue

        candidates = _generate_pairwise_orientation_candidates_for_issue(
            cluster_ids=cluster_ids,
            reasons_by_cluster=reasons_by_cluster,
            hard_issues_by_cluster=hard_issues_by_cluster,
            tf_by_id=tf_by_id,
            cinfo_by_id=cinfo_by_id,
            object_index=object_index,
            grid_mm=grid_mm,
            consider_rotations=consider_rotations,
            room_poly=room_poly,
            room_prepared=room_prepared,
            room_bbox=room_bbox,
            room_center=room_center,
            room_diag_mm=room_diag_mm,
            free_room_area_mm2=free_room_area_mm2,
            forbid_obstacle_touch=forbid_obstacle_touch,
            eps_area=eps_area,
            obstacles=obstacles,
            cluster_geoms=cluster_geoms,
            opening_ctx=opening_ctx,
            obstacle_by_type=obstacle_by_type,
            relation_plan=relation_plan,
            base_preferred_gap_mm=base_preferred_gap_mm,
            min_preferred_gap_mm=min_preferred_gap_mm,
            max_preferred_gap_mm=max_preferred_gap_mm,
            max_return=max_candidates_per_issue,
        )

        if not candidates:
            continue

        hints.append(
            {
                "cluster_id": a,
                "cluster_ids": [a, b],
                "with_id": b,
                "trigger": "PAIRWISE_ORIENTATION_HARD",
                "base_transform": {
                    "patch_clusters": _canonicalize_patch_clusters(
                        [tf_by_id[a], tf_by_id[b]]
                    )
                },
                "candidates": candidates[:max_candidates_per_issue],
            }
        )

    return hints


def _generate_orientation_improve_hints(
    *,
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    cluster_geoms: Dict[str, Any],
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
    baseline_quality: Dict[str, Any],
    grid_mm: int,
    consider_rotations: bool,
) -> List[Dict[str, Any]]:
    baseline_orientation_penalty = int(
        baseline_quality.get("orientation_penalty_mm", 0) or 0
    )
    baseline_soft_penalty = int(
        baseline_quality.get("total_soft_penalty_mm", 10**9) or 10**9
    )

    if baseline_orientation_penalty <= 0:
        return []

    per_cluster = _collect_orientation_targets_with_reasons(
        baseline_quality.get("orientation_debug")
    )
    if not per_cluster:
        return []

    hints_by_cluster: Dict[str, Dict[str, Any]] = {}

    ranked_targets = sorted(
        per_cluster.items(),
        key=lambda x: (-int(x[1].get("penalty", 0) or 0), x[0]),
    )[:6]

    for cid, rec in ranked_targets:
        if cid not in tf_by_id or cid not in cinfo_by_id or cid not in cluster_geoms:
            continue

        base_t = tf_by_id[cid]
        base_x = int(base_t.get("x", 0))
        base_y = int(base_t.get("y", 0))
        base_rot = int(base_t.get("rot", 0)) % 360
        reasons = rec.get("reasons") if isinstance(rec, dict) else []
        reasons = reasons if isinstance(reasons, list) else []

        candidates_raw = _build_orientation_candidates_for_cluster(
            cid=cid,
            tf_by_id=tf_by_id,
            cluster_geoms=cluster_geoms,
            room_poly=room_poly,
            room_center=room_center,
            room_bbox=room_bbox,
            opening_ctx=opening_ctx,
            reasons=reasons,
            base_x=base_x,
            base_y=base_y,
            base_rot=base_rot,
            grid_mm=grid_mm,
            consider_rotations=consider_rotations,
        )

        evaluated = _evaluate_orientation_candidates(
            cid=cid,
            candidates=candidates_raw,
            tf_by_id=tf_by_id,
            cinfo_by_id=cinfo_by_id,
            object_index=object_index,
            cluster_geoms=cluster_geoms,
            room_poly=room_poly,
            room_prepared=room_prepared,
            room_bbox=room_bbox,
            room_center=room_center,
            room_diag_mm=room_diag_mm,
            free_room_area_mm2=free_room_area_mm2,
            forbid_obstacle_touch=forbid_obstacle_touch,
            eps_area=eps_area,
            obstacles=obstacles,
            opening_ctx=opening_ctx,
            obstacle_by_type=obstacle_by_type,
            relation_plan=relation_plan,
            base_preferred_gap_mm=base_preferred_gap_mm,
            min_preferred_gap_mm=min_preferred_gap_mm,
            max_preferred_gap_mm=max_preferred_gap_mm,
            baseline_orientation_penalty=baseline_orientation_penalty,
            baseline_soft_penalty=baseline_soft_penalty,
        )

        if not evaluated:
            continue

        hints_by_cluster[cid] = {
            "cluster_id": cid,
            "trigger": "ORIENTATION_MISMATCH",
            "baseline_cluster_orientation_penalty_mm": int(rec.get("penalty", 0) or 0),
            "baseline_orientation_penalty_mm": int(baseline_orientation_penalty),
            "base_transform": {
                "x": base_x,
                "y": base_y,
                "rot": base_rot,
            },
            "candidates": evaluated[:6],
        }

    pairwise = _generate_pairwise_orientation_improve_hints(
        tf_by_id=tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        cluster_geoms=cluster_geoms,
        room_poly=room_poly,
        room_prepared=room_prepared,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        forbid_obstacle_touch=forbid_obstacle_touch,
        eps_area=eps_area,
        obstacles=obstacles,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        relation_plan=relation_plan,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
        baseline_quality=baseline_quality,
        grid_mm=grid_mm,
        consider_rotations=consider_rotations,
    )

    for hint in pairwise:
        cid = hint.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        if cid not in hints_by_cluster:
            hints_by_cluster[cid] = hint
            continue

        existing = hints_by_cluster[cid]
        candidates = existing.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        extra = hint.get("candidates")
        if isinstance(extra, list):
            candidates.extend(extra)

        candidates = _dedup_improve_candidates(candidates)
        candidates.sort(
            key=lambda c: (
                c.get("cluster_orientation_penalty_mm", 10**9),
                c.get("orientation_penalty_mm", 10**9),
                c.get("soft_penalty_mm", 10**9),
                c.get("cost", 10**9),
                c.get("reason", ""),
            )
        )
        existing["candidates"] = candidates[:6]
        existing["baseline_cluster_orientation_penalty_mm"] = max(
            int(existing.get("baseline_cluster_orientation_penalty_mm", 0) or 0),
            int(hint.get("baseline_cluster_orientation_penalty_mm", 0) or 0),
        )

    hints = list(hints_by_cluster.values())
    hints.sort(
        key=lambda h: (
            -int(h.get("baseline_cluster_orientation_penalty_mm", 0) or 0),
            h.get("cluster_id", ""),
        )
    )
    return hints


def _build_orientation_candidates_for_cluster(
    *,
    cid: str,
    tf_by_id: Dict[str, Dict[str, Any]],
    cluster_geoms: Dict[str, Any],
    room_poly: Any,
    room_center: Tuple[float, float],
    room_bbox: Tuple[float, float, float, float],
    opening_ctx: Dict[str, Any],
    reasons: List[Dict[str, Any]],
    base_x: int,
    base_y: int,
    base_rot: int,
    grid_mm: int,
    consider_rotations: bool,
) -> List[Tuple[int, int, int, str]]:
    candidates: List[Tuple[int, int, int, str]] = []
    seen = set()

    def add_candidate(x: int, y: int, rot: int, reason: str) -> None:
        if grid_mm > 0:
            x = _snap_grid(x, grid_mm)
            y = _snap_grid(y, grid_mm)
        key = (x, y, rot)
        if key in seen:
            return
        seen.add(key)
        candidates.append((x, y, rot, reason))

    if consider_rotations:
        for rot in (0, 90, 180, 270):
            if rot == base_rot:
                continue
            add_candidate(base_x, base_y, rot, f"ROTATE_{rot}_FOR_ORIENTATION")

    for step in (1, 2):
        s = step * grid_mm
        add_candidate(base_x + s, base_y, base_rot, f"ORIENT_NUDGE_RIGHT_{step}g")
        add_candidate(base_x - s, base_y, base_rot, f"ORIENT_NUDGE_LEFT_{step}g")
        add_candidate(base_x, base_y + s, base_rot, f"ORIENT_NUDGE_UP_{step}g")
        add_candidate(base_x, base_y - s, base_rot, f"ORIENT_NUDGE_DOWN_{step}g")

    wants_wall = False
    wants_open = False
    target_cluster_ids: List[str] = []

    for r in reasons:
        if not isinstance(r, dict):
            continue
        kind = str(r.get("kind") or "")
        if kind == "cluster_orientation":
            intent = str(r.get("intent") or "")
            if intent in {
                "back_to_wall",
                "outward_to_wall",
                "axis_parallel_wall",
                "axis_perpendicular_wall",
            }:
                wants_wall = True
            if intent in {
                "access_to_open_space",
                "inward_to_room",
                "face_center",
                "face_entry",
                "face_window",
            }:
                wants_open = True
            tc = r.get("target_cluster_id")
            if isinstance(tc, str) and tc:
                target_cluster_ids.append(tc)
        elif kind == "cluster_directional_relation":
            oc = r.get("other_cluster_id")
            if isinstance(oc, str) and oc:
                target_cluster_ids.append(oc)
        elif kind == "object_orientation":
            tc = r.get("resolved_target_cluster_id")
            if isinstance(tc, str) and tc:
                target_cluster_ids.append(tc)

    base_geom = cluster_geoms.get(cid)
    if base_geom is not None:
        centroid = (float(base_geom.centroid.x), float(base_geom.centroid.y))

        if wants_wall:
            wall = _nearest_wall_info(room_poly, centroid)
            if wall is not None:
                vx, vy = _dominant_step_from_vector(
                    _vec_from_to(centroid, wall["midpoint"]), grid_mm
                )
                if vx != 0 or vy != 0:
                    add_candidate(
                        base_x + vx, base_y + vy, base_rot, "MOVE_TOWARD_NEAREST_WALL"
                    )
                    if consider_rotations:
                        for rot in (0, 90, 180, 270):
                            add_candidate(
                                base_x + vx,
                                base_y + vy,
                                rot,
                                f"ROTATE_{rot}_TOWARD_WALL",
                            )

        if wants_open:
            vx, vy = _dominant_step_from_vector(
                _vec_from_to(centroid, room_center), grid_mm
            )
            if vx != 0 or vy != 0:
                add_candidate(
                    base_x + vx, base_y + vy, base_rot, "MOVE_TOWARD_ROOM_OPEN_SPACE"
                )
                if consider_rotations:
                    for rot in (0, 90, 180, 270):
                        add_candidate(
                            base_x + vx,
                            base_y + vy,
                            rot,
                            f"ROTATE_{rot}_TOWARD_OPEN_SPACE",
                        )

        for target_cid in _uniq_keep_order_str(target_cluster_ids)[:2]:
            target_geom = cluster_geoms.get(target_cid)
            if target_geom is None:
                continue
            target_centroid = (
                float(target_geom.centroid.x),
                float(target_geom.centroid.y),
            )
            vx, vy = _dominant_step_from_vector(
                _vec_from_to(centroid, target_centroid), grid_mm
            )
            if vx != 0 or vy != 0:
                add_candidate(
                    base_x + vx, base_y + vy, base_rot, f"MOVE_TOWARD_{target_cid}"
                )
                if consider_rotations:
                    for rot in (0, 90, 180, 270):
                        add_candidate(
                            base_x + vx,
                            base_y + vy,
                            rot,
                            f"ROTATE_{rot}_TOWARD_{target_cid}",
                        )

    return candidates


def _evaluate_orientation_candidates(
    *,
    cid: str,
    candidates: List[Tuple[int, int, int, str]],
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    cluster_geoms: Dict[str, Any],
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
    baseline_orientation_penalty: int,
    baseline_soft_penalty: int,
) -> List[Dict[str, Any]]:
    base_t = tf_by_id[cid]
    base_x = int(base_t.get("x", 0))
    base_y = int(base_t.get("y", 0))
    base_rot = int(base_t.get("rot", 0)) % 360

    evaluated: List[Dict[str, Any]] = []

    base_cluster_orientation_penalty = _cluster_orientation_penalty_from_quality(
        _evaluate_layout_quality(
            cluster_geoms=cluster_geoms,
            cluster_transforms_by_id=tf_by_id,
            cinfo_by_id=cinfo_by_id,
            object_index=object_index,
            room_poly=room_poly,
            room_bbox=room_bbox,
            room_center=room_center,
            room_diag_mm=room_diag_mm,
            free_room_area_mm2=free_room_area_mm2,
            opening_ctx=opening_ctx,
            obstacle_by_type=obstacle_by_type,
            relation_plan=relation_plan,
            base_preferred_gap_mm=base_preferred_gap_mm,
            min_preferred_gap_mm=min_preferred_gap_mm,
            max_preferred_gap_mm=max_preferred_gap_mm,
        ),
        cid,
    )

    for cand_x, cand_y, cand_rot, reason in candidates:
        post = _postcheck_candidate(
            cid=cid,
            tf_by_id=tf_by_id,
            cinfo_by_id=cinfo_by_id,
            object_index=object_index,
            trigger="ORIENTATION_MISMATCH",
            with_id=None,
            cand_x=cand_x,
            cand_y=cand_y,
            cand_rot=cand_rot,
            room_poly=room_poly,
            room_prepared=room_prepared,
            room_bbox=room_bbox,
            room_center=room_center,
            room_diag_mm=room_diag_mm,
            free_room_area_mm2=free_room_area_mm2,
            forbid_obstacle_touch=forbid_obstacle_touch,
            eps_area=eps_area,
            obstacles=obstacles,
            cluster_geoms=cluster_geoms,
            cinfo=cinfo_by_id[cid],
            opening_ctx=opening_ctx,
            obstacle_by_type=obstacle_by_type,
            relation_plan=relation_plan,
            base_preferred_gap_mm=base_preferred_gap_mm,
            min_preferred_gap_mm=min_preferred_gap_mm,
            max_preferred_gap_mm=max_preferred_gap_mm,
        )

        hard_score = int(post.get("hard_violation_score", 10**9) or 10**9)
        if hard_score != 0:
            continue

        cand_cluster_orientation_penalty = int(
            post.get("cluster_orientation_penalty_mm", 10**9) or 10**9
        )
        cand_global_orientation_penalty = int(
            post.get("global_orientation_penalty_mm", 10**9) or 10**9
        )
        cand_soft_penalty = int(post.get("soft_penalty_mm", 10**9) or 10**9)

        improved = (
            cand_cluster_orientation_penalty < base_cluster_orientation_penalty
            or (
                cand_cluster_orientation_penalty == base_cluster_orientation_penalty
                and cand_global_orientation_penalty < baseline_orientation_penalty
            )
            or (
                cand_cluster_orientation_penalty == base_cluster_orientation_penalty
                and cand_global_orientation_penalty == baseline_orientation_penalty
                and cand_soft_penalty < baseline_soft_penalty
            )
        )
        if not improved:
            continue

        dx = cand_x - base_x
        dy = cand_y - base_y
        cost = abs(dx) + abs(dy) + (0 if cand_rot == base_rot else 500)

        evaluated.append(
            {
                "x": cand_x,
                "y": cand_y,
                "rot": cand_rot,
                "dx": dx,
                "dy": dy,
                "reason": reason,
                "cost": cost,
                "postcheck": _compact_postcheck(post),
                "cluster_orientation_penalty_mm": cand_cluster_orientation_penalty,
                "orientation_penalty_mm": cand_global_orientation_penalty,
                "soft_penalty_mm": cand_soft_penalty,
            }
        )

    evaluated = _dedup_improve_candidates(evaluated)
    evaluated.sort(
        key=lambda c: (
            c.get("cluster_orientation_penalty_mm", 10**9),
            c.get("orientation_penalty_mm", 10**9),
            c.get("soft_penalty_mm", 10**9),
            c.get("cost", 10**9),
            c.get("reason", ""),
        )
    )
    return evaluated


def _generate_pairwise_orientation_improve_hints(
    *,
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    cluster_geoms: Dict[str, Any],
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
    baseline_quality: Dict[str, Any],
    grid_mm: int,
    consider_rotations: bool,
) -> List[Dict[str, Any]]:
    orientation_debug = baseline_quality.get("orientation_debug")
    if not isinstance(orientation_debug, list):
        return []

    baseline_orientation_penalty = int(
        baseline_quality.get("orientation_penalty_mm", 0) or 0
    )
    baseline_soft_penalty = int(
        baseline_quality.get("total_soft_penalty_mm", 10**9) or 10**9
    )

    pair_items: List[Dict[str, Any]] = []
    for item in orientation_debug:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "cluster_directional_relation":
            continue
        penalty = int(item.get("penalty_mm", 0) or 0)
        if penalty <= 0:
            continue
        relation = str(item.get("relation") or "")
        if relation not in {
            "face_each_other",
            "turn_toward",
            "access_faces_other",
            "same_axis",
            "parallel_alignment",
            "perpendicular_alignment",
        }:
            continue
        a = item.get("a")
        b = item.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a not in tf_by_id or b not in tf_by_id:
            continue
        pair_items.append(item)

    hints_by_cluster: Dict[str, Dict[str, Any]] = {}

    pair_items.sort(
        key=lambda x: (
            -int(x.get("penalty_mm", 0) or 0),
            str(x.get("a")),
            str(x.get("b")),
        )
    )
    pair_items = pair_items[:6]

    for item in pair_items:
        a = str(item["a"])
        b = str(item["b"])
        penalty = int(item.get("penalty_mm", 0) or 0)

        for cid, other in ((a, b), (b, a)):
            if (
                cid not in tf_by_id
                or cid not in cinfo_by_id
                or cid not in cluster_geoms
            ):
                continue
            if other not in cluster_geoms:
                continue

            base_t = tf_by_id[cid]
            base_x = int(base_t.get("x", 0))
            base_y = int(base_t.get("y", 0))
            base_rot = int(base_t.get("rot", 0)) % 360
            own_cent = (
                float(cluster_geoms[cid].centroid.x),
                float(cluster_geoms[cid].centroid.y),
            )
            oth_cent = (
                float(cluster_geoms[other].centroid.x),
                float(cluster_geoms[other].centroid.y),
            )

            candidates_raw: List[Tuple[int, int, int, str]] = []
            seen = set()

            def add_candidate(x: int, y: int, rot: int, reason: str) -> None:
                if grid_mm > 0:
                    x = _snap_grid(x, grid_mm)
                    y = _snap_grid(y, grid_mm)
                key = (x, y, rot)
                if key in seen:
                    return
                seen.add(key)
                candidates_raw.append((x, y, rot, reason))

            if consider_rotations:
                for rot in (0, 90, 180, 270):
                    if rot == base_rot:
                        continue
                    add_candidate(
                        base_x, base_y, rot, f"PAIR_ROTATE_{rot}_TOWARD_{other}"
                    )

            vx, vy = _dominant_step_from_vector(
                _vec_from_to(own_cent, oth_cent), grid_mm
            )
            if vx != 0 or vy != 0:
                add_candidate(
                    base_x + vx, base_y + vy, base_rot, f"PAIR_MOVE_TOWARD_{other}"
                )
                if consider_rotations:
                    for rot in (0, 90, 180, 270):
                        add_candidate(
                            base_x + vx,
                            base_y + vy,
                            rot,
                            f"PAIR_ROTATE_{rot}_MOVE_TOWARD_{other}",
                        )

            evaluated = _evaluate_orientation_candidates(
                cid=cid,
                candidates=candidates_raw,
                tf_by_id=tf_by_id,
                cinfo_by_id=cinfo_by_id,
                object_index=object_index,
                cluster_geoms=cluster_geoms,
                room_poly=room_poly,
                room_prepared=room_prepared,
                room_bbox=room_bbox,
                room_center=room_center,
                room_diag_mm=room_diag_mm,
                free_room_area_mm2=free_room_area_mm2,
                forbid_obstacle_touch=forbid_obstacle_touch,
                eps_area=eps_area,
                obstacles=obstacles,
                opening_ctx=opening_ctx,
                obstacle_by_type=obstacle_by_type,
                relation_plan=relation_plan,
                base_preferred_gap_mm=base_preferred_gap_mm,
                min_preferred_gap_mm=min_preferred_gap_mm,
                max_preferred_gap_mm=max_preferred_gap_mm,
                baseline_orientation_penalty=baseline_orientation_penalty,
                baseline_soft_penalty=baseline_soft_penalty,
            )
            if not evaluated:
                continue

            entry = hints_by_cluster.setdefault(
                cid,
                {
                    "cluster_id": cid,
                    "trigger": "PAIRWISE_ORIENTATION_MISMATCH",
                    "baseline_cluster_orientation_penalty_mm": 0,
                    "baseline_orientation_penalty_mm": int(
                        baseline_orientation_penalty
                    ),
                    "base_transform": {
                        "x": base_x,
                        "y": base_y,
                        "rot": base_rot,
                    },
                    "candidates": [],
                },
            )
            entry["baseline_cluster_orientation_penalty_mm"] = max(
                int(entry.get("baseline_cluster_orientation_penalty_mm", 0) or 0),
                penalty,
            )
            existing_candidates = entry.get("candidates")
            if not isinstance(existing_candidates, list):
                existing_candidates = []
            existing_candidates.extend(evaluated[:4])
            entry["candidates"] = existing_candidates

    out = list(hints_by_cluster.values())
    for item in out:
        cands = _dedup_improve_candidates(item.get("candidates"))
        cands.sort(
            key=lambda c: (
                c.get("orientation_penalty_mm", 10**9),
                c.get("soft_penalty_mm", 10**9),
                c.get("cost", 10**9),
                c.get("reason", ""),
            )
        )
        item["candidates"] = cands[:6]

    out.sort(
        key=lambda h: (
            -int(h.get("baseline_cluster_orientation_penalty_mm", 0) or 0),
            h.get("cluster_id", ""),
        )
    )
    return out


def _dedup_improve_candidates(candidates: Any) -> List[Dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for c in candidates:
        if not isinstance(c, dict):
            continue
        key = (
            int(c.get("x", 0)),
            int(c.get("y", 0)),
            int(c.get("rot", 0)) % 360,
            str(c.get("reason", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _dominant_step_from_vector(
    vec: Tuple[float, float] | None,
    grid_mm: int,
) -> Tuple[int, int]:
    if vec is None or grid_mm <= 0:
        return (0, 0)
    dx, dy = vec
    if abs(dx) >= abs(dy):
        return (grid_mm if dx >= 0 else -grid_mm, 0)
    return (0, grid_mm if dy >= 0 else -grid_mm)


def _uniq_keep_order_str(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        if not isinstance(v, str) or not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _postcheck_candidate(
    *,
    cid: str,
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    trigger: str,
    with_id: str | None,
    cand_x: int,
    cand_y: int,
    cand_rot: int,
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    cluster_geoms: Dict[str, Any],
    cinfo: Dict[str, Any],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
) -> Dict[str, Any]:
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except Exception:
        return {
            "valid_geom": False,
            "hard_violation_score": 10**9,
            "violation_score": 10**9,
            "soft_penalty_mm": 10**9,
            "layout_score": -(10**9),
            "solves_trigger": False,
            "remaining_hard_issues": 10**9,
            "all_hard_clear": False,
            "cluster_orientation_penalty_mm": 10**9,
            "global_orientation_penalty_mm": 10**9,
        }

    polys = _build_cluster_polys(Polygon, cinfo, cand_x, cand_y, cand_rot)
    if not polys:
        return {
            "valid_geom": False,
            "hard_violation_score": 10**9,
            "violation_score": 10**9,
            "soft_penalty_mm": 10**9,
            "layout_score": -(10**9),
            "solves_trigger": False,
            "remaining_hard_issues": 10**9,
            "all_hard_clear": False,
            "cluster_orientation_penalty_mm": 10**9,
            "global_orientation_penalty_mm": 10**9,
        }

    geom = unary_union(polys)
    geom = _fix_geom(geom)
    if geom.is_empty or (hasattr(geom, "is_valid") and not geom.is_valid):
        return {
            "valid_geom": False,
            "hard_violation_score": 10**9,
            "violation_score": 10**9,
            "soft_penalty_mm": 10**9,
            "layout_score": -(10**9),
            "solves_trigger": False,
            "remaining_hard_issues": 10**9,
            "all_hard_clear": False,
            "cluster_orientation_penalty_mm": 10**9,
            "global_orientation_penalty_mm": 10**9,
        }

    in_room = room_prepared.covers(geom)

    hit_ids: List[str] = []
    for oid, ob_poly in obstacles:
        if forbid_obstacle_touch:
            if geom.intersects(ob_poly):
                hit_ids.append(oid)
        else:
            if float(geom.intersection(ob_poly).area) > eps_area:
                hit_ids.append(oid)

    overlaps: List[str] = []
    candidate_geoms = dict(cluster_geoms)
    candidate_geoms[cid] = geom

    for other_id, other_geom in cluster_geoms.items():
        if other_id == cid:
            continue
        inter_area = float(geom.intersection(other_geom).area)
        if inter_area > eps_area:
            overlaps.append(other_id)

    candidate_tf_by_id: Dict[str, Dict[str, Any]] = {}
    for k, v in tf_by_id.items():
        candidate_tf_by_id[k] = {
            "cluster_id": k,
            "x": int(v.get("x", 0)),
            "y": int(v.get("y", 0)),
            "rot": int(v.get("rot", 0)) % 360,
        }
    candidate_tf_by_id[cid] = {
        "cluster_id": cid,
        "x": int(cand_x),
        "y": int(cand_y),
        "rot": int(cand_rot) % 360,
    }

    quality = _evaluate_layout_quality(
        cluster_geoms=candidate_geoms,
        cluster_transforms_by_id=candidate_tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        room_poly=room_poly,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        relation_plan=relation_plan,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )

    out_of_bounds = not in_room
    remaining_hard_issues = int(bool(out_of_bounds)) + len(hit_ids) + len(overlaps)
    all_hard_clear = remaining_hard_issues == 0

    hard_violation_score = _violation_score(
        {
            "out_of_bounds": out_of_bounds,
            "hit_obstacles": hit_ids,
            "overlaps": overlaps,
        }
    )

    cluster_orientation_penalty_mm = _cluster_orientation_penalty_from_quality(
        quality, cid
    )
    global_orientation_penalty_mm = int(quality.get("orientation_penalty_mm", 0) or 0)

    solves = False
    if trigger == "CLUSTER_INTERSECTS_OBSTACLE":
        solves = len(hit_ids) == 0 if with_id is None else (with_id not in hit_ids)
    elif trigger == "CLUSTER_OVERLAP":
        solves = len(overlaps) == 0 if with_id is None else (with_id not in overlaps)
    elif trigger == "CLUSTER_OUT_OF_BOUNDS":
        solves = bool(in_room)
    elif trigger in {
        "ORIENTATION_MISMATCH",
        "PAIRWISE_ORIENTATION_MISMATCH",
        "ORIENTATION_HARD",
    }:
        # Orientation có cải thiện hay không sẽ do caller quyết định.
        solves = all_hard_clear

    out = {
        "valid_geom": True,
        "in_room": bool(in_room),
        "hit_obstacles": hit_ids,
        "overlaps": overlaps,
        "solves_trigger": bool(solves),
        "hard_violation_score": int(hard_violation_score),
        # giữ alias này để code cũ không vỡ
        "violation_score": int(hard_violation_score),
        "soft_penalty_mm": int(quality.get("total_soft_penalty_mm", 0) or 0),
        "layout_score": int(quality.get("layout_score", 0) or 0),
        "quality": quality,
        "remaining_hard_issues": remaining_hard_issues,
        "all_hard_clear": all_hard_clear,
        "cluster_orientation_penalty_mm": int(cluster_orientation_penalty_mm),
        "global_orientation_penalty_mm": int(global_orientation_penalty_mm),
    }
    return out


def _postcheck_pair_candidate(
    *,
    cluster_ids: List[str],
    patch_clusters: List[Dict[str, Any]],
    tf_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    room_poly: Any,
    room_prepared: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    forbid_obstacle_touch: bool,
    eps_area: float,
    obstacles: List[Tuple[str, Any]],
    cluster_geoms: Dict[str, Any],
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
) -> Dict[str, Any]:
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except Exception:
        return {
            "valid_geom": False,
            "hard_violation_score": 10**9,
            "violation_score": 10**9,
            "soft_penalty_mm": 10**9,
            "layout_score": -(10**9),
            "solves_trigger": False,
            "remaining_hard_issues": 10**9,
            "all_hard_clear": False,
            "cluster_orientation_penalty_mm": 10**9,
            "global_orientation_penalty_mm": 10**9,
        }

    patch_clusters = _canonicalize_patch_clusters(patch_clusters)
    if not patch_clusters:
        return {
            "valid_geom": False,
            "hard_violation_score": 10**9,
            "violation_score": 10**9,
            "soft_penalty_mm": 10**9,
            "layout_score": -(10**9),
            "solves_trigger": False,
            "remaining_hard_issues": 10**9,
            "all_hard_clear": False,
            "cluster_orientation_penalty_mm": 10**9,
            "global_orientation_penalty_mm": 10**9,
        }

    patch_map = {p["cluster_id"]: p for p in patch_clusters}

    candidate_tf_by_id: Dict[str, Dict[str, Any]] = {}
    for k, v in tf_by_id.items():
        candidate_tf_by_id[k] = {
            "cluster_id": k,
            "x": int(v.get("x", 0)),
            "y": int(v.get("y", 0)),
            "rot": int(v.get("rot", 0)) % 360,
        }
    for p in patch_clusters:
        candidate_tf_by_id[p["cluster_id"]] = dict(p)

    candidate_geoms = dict(cluster_geoms)

    for cid in cluster_ids:
        if cid not in patch_map:
            continue
        cinfo = cinfo_by_id.get(cid)
        if not isinstance(cinfo, dict):
            return {
                "valid_geom": False,
                "hard_violation_score": 10**9,
                "violation_score": 10**9,
                "soft_penalty_mm": 10**9,
                "layout_score": -(10**9),
                "solves_trigger": False,
                "remaining_hard_issues": 10**9,
                "all_hard_clear": False,
                "cluster_orientation_penalty_mm": 10**9,
                "global_orientation_penalty_mm": 10**9,
            }

        p = patch_map[cid]
        polys = _build_cluster_polys(Polygon, cinfo, p["x"], p["y"], p["rot"])
        if not polys:
            return {
                "valid_geom": False,
                "hard_violation_score": 10**9,
                "violation_score": 10**9,
                "soft_penalty_mm": 10**9,
                "layout_score": -(10**9),
                "solves_trigger": False,
                "remaining_hard_issues": 10**9,
                "all_hard_clear": False,
                "cluster_orientation_penalty_mm": 10**9,
                "global_orientation_penalty_mm": 10**9,
            }

        geom = unary_union(polys)
        geom = _fix_geom(geom)
        if geom.is_empty or (hasattr(geom, "is_valid") and not geom.is_valid):
            return {
                "valid_geom": False,
                "hard_violation_score": 10**9,
                "violation_score": 10**9,
                "soft_penalty_mm": 10**9,
                "layout_score": -(10**9),
                "solves_trigger": False,
                "remaining_hard_issues": 10**9,
                "all_hard_clear": False,
                "cluster_orientation_penalty_mm": 10**9,
                "global_orientation_penalty_mm": 10**9,
            }

        candidate_geoms[cid] = geom

    out_of_bounds_clusters: List[str] = []
    hit_ids: List[str] = []
    overlaps: List[str] = []

    for cid in cluster_ids:
        geom = candidate_geoms.get(cid)
        if geom is None:
            continue

        in_room = room_prepared.covers(geom)
        if not in_room:
            out_of_bounds_clusters.append(cid)

        for oid, ob_poly in obstacles:
            if forbid_obstacle_touch:
                if geom.intersects(ob_poly):
                    hit_ids.append(f"{cid}:{oid}")
            else:
                if float(geom.intersection(ob_poly).area) > eps_area:
                    hit_ids.append(f"{cid}:{oid}")

    all_ids = sorted(candidate_geoms.keys())
    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            a_id, b_id = all_ids[i], all_ids[j]
            ga, gb = candidate_geoms[a_id], candidate_geoms[b_id]
            inter_area = float(ga.intersection(gb).area)
            if inter_area > eps_area:
                overlaps.append(f"{a_id}:{b_id}")

    quality = _evaluate_layout_quality(
        cluster_geoms=candidate_geoms,
        cluster_transforms_by_id=candidate_tf_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        room_poly=room_poly,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        free_room_area_mm2=free_room_area_mm2,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        relation_plan=relation_plan,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )

    hard_violation_score = (
        1000 * len(out_of_bounds_clusters) + 200 * len(hit_ids) + 300 * len(overlaps)
    )
    remaining_hard_issues = len(out_of_bounds_clusters) + len(hit_ids) + len(overlaps)
    all_hard_clear = remaining_hard_issues == 0

    pair_cluster_orientation_penalty_mm = _clusters_orientation_penalty_from_quality(
        quality, cluster_ids
    )
    global_orientation_penalty_mm = int(quality.get("orientation_penalty_mm", 0) or 0)

    return {
        "valid_geom": True,
        "in_room": len(out_of_bounds_clusters) == 0,
        "out_of_bounds_clusters": out_of_bounds_clusters,
        "hit_obstacles": hit_ids,
        "overlaps": overlaps,
        "solves_trigger": all_hard_clear,
        "hard_violation_score": int(hard_violation_score),
        "violation_score": int(hard_violation_score),
        "soft_penalty_mm": int(quality.get("total_soft_penalty_mm", 0) or 0),
        "layout_score": int(quality.get("layout_score", 0) or 0),
        "quality": quality,
        "remaining_hard_issues": int(remaining_hard_issues),
        "all_hard_clear": bool(all_hard_clear),
        "cluster_orientation_penalty_mm": int(pair_cluster_orientation_penalty_mm),
        "global_orientation_penalty_mm": int(global_orientation_penalty_mm),
        "patch_cluster_ids": list(cluster_ids),
    }


def _evaluate_layout_quality(
    *,
    cluster_geoms: Dict[str, Any],
    cluster_transforms_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    room_poly: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    free_room_area_mm2: float,
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int,
    max_preferred_gap_mm: int,
) -> Dict[str, Any]:
    cids = sorted(cluster_geoms.keys())
    occupied_area_mm2 = sum(float(g.area) for g in cluster_geoms.values())
    density_ratio = (
        occupied_area_mm2 / free_room_area_mm2 if free_room_area_mm2 > 0 else 1.0
    )

    adaptive_base_gap_mm = _adaptive_base_gap_mm(
        free_room_area_mm2=free_room_area_mm2,
        occupied_area_mm2=occupied_area_mm2,
        base_preferred_gap_mm=base_preferred_gap_mm,
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )

    gaps: List[float] = []
    tight_pairs: List[Dict[str, Any]] = []
    tight_gap_penalty = 0.0

    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a_id, b_id = cids[i], cids[j]
            ga, gb = cluster_geoms[a_id], cluster_geoms[b_id]
            dist = float(ga.distance(gb))
            gaps.append(dist)

            pref_gap = _pair_preferred_gap_mm(
                ga=ga,
                gb=gb,
                adaptive_base_gap_mm=adaptive_base_gap_mm,
                min_preferred_gap_mm=min_preferred_gap_mm,
                max_preferred_gap_mm=max_preferred_gap_mm,
            )
            if dist < pref_gap:
                deficit = pref_gap - dist
                tight_gap_penalty += deficit
                tight_pairs.append(
                    {
                        "a": a_id,
                        "b": b_id,
                        "gap_mm": int(round(dist)),
                        "preferred_gap_mm": int(round(pref_gap)),
                        "deficit_mm": int(round(deficit)),
                    }
                )

    min_gap = min(gaps) if gaps else None
    avg_gap = (sum(gaps) / len(gaps)) if gaps else None

    tight_pairs.sort(key=lambda x: (-x["deficit_mm"], x["a"], x["b"]))
    tight_pairs = tight_pairs[:16]

    spread_penalty_mm, spread_meta = _compute_spread_penalty(
        cluster_geoms=cluster_geoms,
        room_bbox=room_bbox,
    )

    affinity_penalty_mm, affinity_meta = _compute_affinity_penalty(
        cluster_geoms=cluster_geoms,
        room_poly=room_poly,
        room_bbox=room_bbox,
        room_center=room_center,
        room_diag_mm=room_diag_mm,
        opening_ctx=opening_ctx,
        obstacle_by_type=obstacle_by_type,
        relation_plan=relation_plan,
    )

    relation_penalty_mm, relation_meta = _compute_relation_penalty(
        cluster_geoms=cluster_geoms,
        relation_plan=relation_plan,
    )

    circulation_penalty_mm, circulation_meta = _compute_circulation_penalty(
        cluster_geoms=cluster_geoms,
        room_bbox=room_bbox,
        opening_ctx=opening_ctx,
        relation_plan=relation_plan,
    )

    orientation_penalty_mm, orientation_meta = _compute_orientation_penalty(
        cluster_geoms=cluster_geoms,
        cluster_transforms_by_id=cluster_transforms_by_id,
        cinfo_by_id=cinfo_by_id,
        object_index=object_index,
        room_poly=room_poly,
        room_bbox=room_bbox,
        room_center=room_center,
        opening_ctx=opening_ctx,
        relation_plan=relation_plan,
    )

    total_soft_penalty_mm = (
        tight_gap_penalty
        + spread_penalty_mm
        + affinity_penalty_mm
        + relation_penalty_mm
        + circulation_penalty_mm
        + orientation_penalty_mm
    )

    min_gap_term = min_gap if min_gap is not None else adaptive_base_gap_mm
    avg_gap_term = avg_gap if avg_gap is not None else adaptive_base_gap_mm

    span_x_ratio = float(spread_meta.get("span_x_ratio", 0.0))
    span_y_ratio = float(spread_meta.get("span_y_ratio", 0.0))

    layout_score = int(
        round(
            (1.8 * min_gap_term)
            + (0.8 * avg_gap_term)
            + (550.0 * span_x_ratio)
            + (650.0 * span_y_ratio)
            - (1.2 * tight_gap_penalty)
            - (1.0 * spread_penalty_mm)
            - (1.0 * affinity_penalty_mm)
            - (1.0 * relation_penalty_mm)
            - (1.2 * circulation_penalty_mm)
            - (2.8 * orientation_penalty_mm)
        )
    )

    return {
        "density_ratio": _round3(density_ratio),
        "adaptive_base_preferred_gap_mm": int(round(adaptive_base_gap_mm)),
        "min_cluster_gap_mm": None if min_gap is None else int(round(min_gap)),
        "avg_cluster_gap_mm": None if avg_gap is None else int(round(avg_gap)),
        "tight_gap_penalty_mm": int(round(tight_gap_penalty)),
        "tight_gap_pairs": tight_pairs,
        "spread_penalty_mm": int(round(spread_penalty_mm)),
        "affinity_penalty_mm": int(round(affinity_penalty_mm)),
        "relation_penalty_mm": int(round(relation_penalty_mm)),
        "circulation_penalty_mm": int(round(circulation_penalty_mm)),
        "orientation_penalty_mm": int(round(orientation_penalty_mm)),
        "total_soft_penalty_mm": int(round(total_soft_penalty_mm)),
        "spread": spread_meta,
        "affinity_debug": affinity_meta[:8],
        "relation_debug": relation_meta[:8],
        "circulation_debug": circulation_meta[:8],
        "orientation_debug": orientation_meta[:12],
        "layout_score": layout_score,
    }


def _compute_spread_penalty(
    *,
    cluster_geoms: Dict[str, Any],
    room_bbox: Tuple[float, float, float, float],
) -> tuple[float, Dict[str, Any]]:
    cids = sorted(cluster_geoms.keys())
    if len(cids) <= 1:
        return 0.0, {
            "span_x_ratio": 1.0,
            "span_y_ratio": 1.0,
            "centroid_bbox_area_ratio": 1.0,
        }

    centers = []
    for cid in cids:
        geom = cluster_geoms[cid]
        centers.append((float(geom.centroid.x), float(geom.centroid.y)))

    xs = [p[0] for p in centers]
    ys = [p[1] for p in centers]

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    rx1, ry1, rx2, ry2 = room_bbox
    room_w = max(float(rx2 - rx1), 1.0)
    room_h = max(float(ry2 - ry1), 1.0)

    span_x_ratio = span_x / room_w
    span_y_ratio = span_y / room_h
    centroid_bbox_area_ratio = (span_x * span_y) / max(room_w * room_h, 1.0)

    target_x_ratio = 0.32 if len(cids) >= 3 else 0.18
    target_y_ratio = 0.24 if len(cids) >= 3 else 0.15
    target_area_ratio = 0.06 if len(cids) >= 3 else 0.02

    penalty = 0.0
    penalty += max(0.0, target_x_ratio - span_x_ratio) * 1200.0
    penalty += max(0.0, target_y_ratio - span_y_ratio) * 1500.0
    penalty += max(0.0, target_area_ratio - centroid_bbox_area_ratio) * 3000.0

    return penalty, {
        "span_x_ratio": _round3(span_x_ratio),
        "span_y_ratio": _round3(span_y_ratio),
        "centroid_bbox_area_ratio": _round3(centroid_bbox_area_ratio),
    }


def _compute_affinity_penalty(
    *,
    cluster_geoms: Dict[str, Any],
    room_poly: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    room_diag_mm: float,
    opening_ctx: Dict[str, Any],
    obstacle_by_type: Dict[str, List[Tuple[str, Any]]],
    relation_plan: Dict[str, Any] | None,
) -> tuple[float, List[Dict[str, Any]]]:
    if not isinstance(relation_plan, dict):
        return 0.0, []

    affs = relation_plan.get("cluster_affinities")
    if not isinstance(affs, list):
        return 0.0, []

    rx1, ry1, rx2, ry2 = room_bbox
    room_w = max(float(rx2 - rx1), 1.0)
    room_h = max(float(ry2 - ry1), 1.0)
    wall_target_mm = _clamp(0.08 * min(room_w, room_h), 100.0, 450.0)
    far_entry_target = 0.35 * room_diag_mm

    total = 0.0
    meta: List[Dict[str, Any]] = []

    for item in affs:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        if not isinstance(cid, str) or cid not in cluster_geoms:
            continue

        geom = cluster_geoms[cid]
        weight = _priority_weight(item.get("priority"))

        wall_dist = float(room_poly.boundary.distance(geom))
        center_dist = float(geom.centroid.distance(room_poly.centroid))
        door_dist = _min_geom_to_lines_distance(
            geom, opening_ctx.get("door_lines") or []
        )
        window_dist = _min_geom_to_lines_distance(
            geom, opening_ctx.get("window_lines") or []
        )
        door_swing_dist = _min_geom_to_polys_distance(
            geom, [p for _, p in obstacle_by_type.get("door_swing", [])]
        )
        window_clearance_dist = _min_geom_to_polys_distance(
            geom, [p for _, p in obstacle_by_type.get("window_clearance", [])]
        )

        local = 0.0

        for token in item.get("prefer") or []:
            t = str(token)
            if t in {"wall", "long_wall", "recess_or_edge", "short_wall"}:
                local += max(0.0, wall_dist - wall_target_mm) * 0.8
            elif t == "center":
                local += (center_dist / max(room_diag_mm, 1.0)) * 500.0
            elif t == "window_side":
                d = window_dist if window_dist is not None else 1200.0
                local += min(d, 1200.0) * 0.45
            elif t == "entry_side":
                d = door_dist if door_dist is not None else 1200.0
                local += min(d, 1200.0) * 0.45
            elif t == "far_from_entry":
                d = door_dist if door_dist is not None else far_entry_target
                local += max(0.0, far_entry_target - d) * 0.75

        for token in item.get("avoid") or []:
            t = str(token)
            if t == "center":
                local += max(0.0, 0.20 * room_diag_mm - center_dist) * 0.85
            elif t in {"door_swing", "entry_blocking"}:
                d = door_swing_dist
                if d is None:
                    d = door_dist
                d = d if d is not None else 1200.0
                local += max(0.0, 700.0 - d) * 1.1
            elif t in {"window_clearance", "window_blocking"}:
                d = window_clearance_dist
                if d is None:
                    d = window_dist
                d = d if d is not None else 1200.0
                local += max(0.0, 500.0 - d) * 1.0
            elif t == "main_path":
                local += 0.0

        local *= weight
        total += local

        meta.append(
            {
                "cluster_id": cid,
                "penalty_mm": int(round(local)),
                "wall_dist_mm": int(round(wall_dist)),
                "center_dist_mm": int(round(center_dist)),
                "door_dist_mm": None if door_dist is None else int(round(door_dist)),
                "window_dist_mm": None
                if window_dist is None
                else int(round(window_dist)),
            }
        )

    meta.sort(key=lambda x: (-x["penalty_mm"], x["cluster_id"]))
    return total, meta


def _compute_relation_penalty(
    *,
    cluster_geoms: Dict[str, Any],
    relation_plan: Dict[str, Any] | None,
) -> tuple[float, List[Dict[str, Any]]]:
    if not isinstance(relation_plan, dict):
        return 0.0, []

    rels = relation_plan.get("cluster_relations")
    if not isinstance(rels, list):
        return 0.0, []

    total = 0.0
    meta: List[Dict[str, Any]] = []

    for item in rels:
        if not isinstance(item, dict):
            continue
        a = item.get("a")
        b = item.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a not in cluster_geoms or b not in cluster_geoms or a == b:
            continue

        ga = cluster_geoms[a]
        gb = cluster_geoms[b]
        dist = float(ga.centroid.distance(gb.centroid))
        diag_a = _geom_diag_mm(ga)
        diag_b = _geom_diag_mm(gb)
        weight = _priority_weight(item.get("priority"))

        rel = str(item.get("relation") or "")
        local = 0.0

        near_target = _clamp(0.35 * (diag_a + diag_b), 500.0, 2500.0)
        sep_target = _clamp(0.70 * (diag_a + diag_b), 900.0, 4200.0)

        if rel in {"near", "adjacent_if_possible"}:
            local += max(0.0, dist - near_target) * 0.30 * weight
        elif rel in {"separate", "far_if_possible"}:
            local += max(0.0, sep_target - dist) * 0.35 * weight

        total += local
        meta.append(
            {
                "a": a,
                "b": b,
                "relation": rel,
                "distance_mm": int(round(dist)),
                "penalty_mm": int(round(local)),
            }
        )

    meta.sort(key=lambda x: (-x["penalty_mm"], x["a"], x["b"]))
    return total, meta


def _compute_circulation_penalty(
    *,
    cluster_geoms: Dict[str, Any],
    room_bbox: Tuple[float, float, float, float],
    opening_ctx: Dict[str, Any],
    relation_plan: Dict[str, Any] | None,
) -> tuple[float, List[Dict[str, Any]]]:
    if not isinstance(relation_plan, dict):
        return 0.0, []

    try:
        from shapely.geometry import LineString
    except Exception:
        return 0.0, []

    total = 0.0
    meta: List[Dict[str, Any]] = []

    circ = relation_plan.get("circulation_plan")
    if not isinstance(circ, dict):
        return 0.0, []

    rx1, ry1, rx2, ry2 = room_bbox
    room_w = max(float(rx2 - rx1), 1.0)
    room_h = max(float(ry2 - ry1), 1.0)

    main_paths = circ.get("main_paths")
    if isinstance(main_paths, list):
        for item in main_paths:
            if not isinstance(item, dict):
                continue
            door_id = item.get("from")
            target_cid = item.get("to_cluster")
            if not isinstance(door_id, str) or not isinstance(target_cid, str):
                continue
            if target_cid not in cluster_geoms:
                continue

            door_pt = _find_opening_midpoint(opening_ctx.get("doors") or [], door_id)
            if door_pt is None:
                continue

            target_geom = cluster_geoms[target_cid]
            target_pt = (float(target_geom.centroid.x), float(target_geom.centroid.y))
            path = LineString([door_pt, target_pt])
            width = _clamp(0.08 * min(room_w, room_h), 350.0, 650.0)
            corridor = path.buffer(width / 2.0, cap_style=2, join_style=2)

            weight = _priority_weight(item.get("priority"))

            for cid, geom in cluster_geoms.items():
                if cid == target_cid:
                    continue
                if corridor.intersects(geom):
                    inter_area = float(corridor.intersection(geom).area)
                    local = weight * _clamp(inter_area / max(width, 1.0), 120.0, 900.0)
                    total += local
                    meta.append(
                        {
                            "kind": "main_path",
                            "path_from": door_id,
                            "path_to": target_cid,
                            "blocker": cid,
                            "penalty_mm": int(round(local)),
                        }
                    )

    keep_open_regions = circ.get("keep_open_regions")
    if isinstance(keep_open_regions, list):
        for item in keep_open_regions:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type") or "")
            near = item.get("near")
            weight = _priority_weight(item.get("priority"))

            if typ == "entry_buffer" and isinstance(near, str):
                door_line = _find_opening_line(opening_ctx.get("doors") or [], near)
                if door_line is None:
                    continue
                zone = door_line.buffer(700.0, cap_style=2, join_style=2)
                for cid, geom in cluster_geoms.items():
                    if zone.intersects(geom):
                        inter_area = float(zone.intersection(geom).area)
                        local = weight * _clamp(inter_area / 250.0, 100.0, 1200.0)
                        total += local
                        meta.append(
                            {
                                "kind": "entry_buffer",
                                "near": near,
                                "blocker": cid,
                                "penalty_mm": int(round(local)),
                            }
                        )

            elif typ == "window_buffer" and isinstance(near, str):
                window_line = _find_opening_line(opening_ctx.get("windows") or [], near)
                if window_line is None:
                    continue
                zone = window_line.buffer(450.0, cap_style=2, join_style=2)
                for cid, geom in cluster_geoms.items():
                    if zone.intersects(geom):
                        inter_area = float(zone.intersection(geom).area)
                        local = weight * _clamp(inter_area / 250.0, 80.0, 900.0)
                        total += local
                        meta.append(
                            {
                                "kind": "window_buffer",
                                "near": near,
                                "blocker": cid,
                                "penalty_mm": int(round(local)),
                            }
                        )

            elif typ == "center_lane":
                if room_w >= room_h:
                    lane = LineString(
                        [(rx1, (ry1 + ry2) / 2.0), (rx2, (ry1 + ry2) / 2.0)]
                    ).buffer(350.0, cap_style=2, join_style=2)
                else:
                    lane = LineString(
                        [((rx1 + rx2) / 2.0, ry1), ((rx1 + rx2) / 2.0, ry2)]
                    ).buffer(350.0, cap_style=2, join_style=2)

                for cid, geom in cluster_geoms.items():
                    if lane.intersects(geom):
                        inter_area = float(lane.intersection(geom).area)
                        local = weight * _clamp(inter_area / 350.0, 60.0, 650.0)
                        total += local
                        meta.append(
                            {
                                "kind": "center_lane",
                                "near": near,
                                "blocker": cid,
                                "penalty_mm": int(round(local)),
                            }
                        )

            elif typ == "work_lane" and isinstance(near, str) and near in cluster_geoms:
                target_geom = cluster_geoms[near]
                target_pt = (
                    float(target_geom.centroid.x),
                    float(target_geom.centroid.y),
                )
                door_pt = _nearest_opening_midpoint_to_point(
                    opening_ctx.get("doors") or [], target_pt
                )
                if door_pt is None:
                    continue
                path = LineString([door_pt, target_pt]).buffer(
                    300.0, cap_style=2, join_style=2
                )
                for cid, geom in cluster_geoms.items():
                    if cid == near:
                        continue
                    if path.intersects(geom):
                        inter_area = float(path.intersection(geom).area)
                        local = weight * _clamp(inter_area / 300.0, 60.0, 700.0)
                        total += local
                        meta.append(
                            {
                                "kind": "work_lane",
                                "near": near,
                                "blocker": cid,
                                "penalty_mm": int(round(local)),
                            }
                        )

    meta.sort(key=lambda x: (-x["penalty_mm"], x.get("kind", ""), x.get("blocker", "")))
    return total, meta


def _compute_orientation_penalty(
    *,
    cluster_geoms: Dict[str, Any],
    cluster_transforms_by_id: Dict[str, Dict[str, Any]],
    cinfo_by_id: Dict[str, Dict[str, Any]],
    object_index: Dict[str, List[Dict[str, Any]]],
    room_poly: Any,
    room_bbox: Tuple[float, float, float, float],
    room_center: Tuple[float, float],
    opening_ctx: Dict[str, Any],
    relation_plan: Dict[str, Any] | None,
) -> tuple[float, List[Dict[str, Any]]]:
    if not isinstance(relation_plan, dict):
        return 0.0, []

    total = 0.0
    meta: List[Dict[str, Any]] = []

    cluster_orientations = relation_plan.get("cluster_orientations")
    if not isinstance(cluster_orientations, list):
        cluster_orientations = []

    cluster_directional_relations = relation_plan.get("cluster_directional_relations")
    if not isinstance(cluster_directional_relations, list):
        cluster_directional_relations = []

    object_orientations = relation_plan.get("object_orientations")
    if not isinstance(object_orientations, list):
        object_orientations = []

    for item in cluster_orientations:
        if not isinstance(item, dict):
            continue

        cid = item.get("cluster_id")
        if (
            not isinstance(cid, str)
            or cid not in cluster_geoms
            or cid not in cinfo_by_id
        ):
            continue

        tf = cluster_transforms_by_id.get(cid) or {}
        rot = int(tf.get("rot", 0)) % 360
        geom = cluster_geoms[cid]
        cinfo = cinfo_by_id[cid]
        centroid = (float(geom.centroid.x), float(geom.centroid.y))
        front = _cluster_front_vector(cinfo, rot)
        axis = _cluster_axis_vector(cinfo, rot)
        weight = _priority_weight(item.get("priority"))
        target_cluster_id = item.get("target_cluster_id")

        for intent in item.get("intents") or []:
            local = 0.0
            dot = None

            if intent == "face_center" and front is not None:
                local, dot = _penalty_face(
                    front, _vec_from_to(centroid, room_center), 420.0, weight
                )

            elif intent == "face_entry" and front is not None:
                info = _nearest_opening_info(opening_ctx.get("doors") or [], centroid)
                if info is not None:
                    local, dot = _penalty_face(
                        front, _vec_from_to(centroid, info["midpoint"]), 460.0, weight
                    )

            elif intent == "face_window" and front is not None:
                info = _nearest_opening_info(opening_ctx.get("windows") or [], centroid)
                if info is not None:
                    local, dot = _penalty_face(
                        front, _vec_from_to(centroid, info["midpoint"]), 400.0, weight
                    )

            elif intent == "face_cluster" and front is not None:
                if (
                    isinstance(target_cluster_id, str)
                    and target_cluster_id in cluster_geoms
                    and target_cluster_id != cid
                ):
                    tgeom = cluster_geoms[target_cluster_id]
                    tcent = (float(tgeom.centroid.x), float(tgeom.centroid.y))
                    scale = 760.0
                    if _cluster_pair_is_viewing_like(
                        cinfo_by_id.get(cid, {}),
                        cinfo_by_id.get(target_cluster_id, {}),
                    ):
                        scale = 980.0
                    local, dot = _penalty_face(
                        front, _vec_from_to(centroid, tcent), scale, weight
                    )

            elif (
                intent in {"access_to_open_space", "inward_to_room"}
                and front is not None
            ):
                local, dot = _penalty_face(
                    front, _vec_from_to(centroid, room_center), 360.0, weight
                )

            elif intent == "back_to_wall" and front is not None:
                wall = _nearest_wall_info(room_poly, centroid)
                if wall is not None:
                    local, dot = _penalty_face(
                        front,
                        _vec_from_to(wall["midpoint"], centroid),
                        520.0,
                        weight,
                    )

            elif intent == "outward_to_wall" and front is not None:
                wall = _nearest_wall_info(room_poly, centroid)
                if wall is not None:
                    local, dot = _penalty_face(
                        front,
                        _vec_from_to(centroid, wall["midpoint"]),
                        360.0,
                        weight,
                    )

            elif intent == "axis_parallel_wall" and axis is not None:
                wall = _nearest_wall_info(room_poly, centroid)
                if wall is not None:
                    local, dot = _penalty_parallel(axis, wall["dir"], 360.0, weight)

            elif intent == "axis_perpendicular_wall" and axis is not None:
                wall = _nearest_wall_info(room_poly, centroid)
                if wall is not None:
                    local, dot = _penalty_perpendicular(
                        axis, wall["dir"], 360.0, weight
                    )

            elif intent == "axis_parallel_window" and axis is not None:
                info = _nearest_opening_info(opening_ctx.get("windows") or [], centroid)
                if info is not None:
                    local, dot = _penalty_parallel(axis, info["dir"], 300.0, weight)

            elif intent == "axis_perpendicular_window" and axis is not None:
                info = _nearest_opening_info(opening_ctx.get("windows") or [], centroid)
                if info is not None:
                    local, dot = _penalty_perpendicular(
                        axis, info["dir"], 300.0, weight
                    )

            if local > 0.0:
                total += local
                meta.append(
                    {
                        "kind": "cluster_orientation",
                        "cluster_id": cid,
                        "intent": intent,
                        "target_cluster_id": target_cluster_id,
                        "penalty_mm": int(round(local)),
                        "dot": None if dot is None else round(dot, 3),
                    }
                )

    for item in cluster_directional_relations:
        if not isinstance(item, dict):
            continue

        a = item.get("a")
        b = item.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a == b or a not in cluster_geoms or b not in cluster_geoms:
            continue

        tf_a = cluster_transforms_by_id.get(a) or {}
        tf_b = cluster_transforms_by_id.get(b) or {}
        rot_a = int(tf_a.get("rot", 0)) % 360
        rot_b = int(tf_b.get("rot", 0)) % 360

        front_a = _cluster_front_vector(cinfo_by_id.get(a, {}), rot_a)
        front_b = _cluster_front_vector(cinfo_by_id.get(b, {}), rot_b)
        axis_a = _cluster_axis_vector(cinfo_by_id.get(a, {}), rot_a)
        axis_b = _cluster_axis_vector(cinfo_by_id.get(b, {}), rot_b)

        ca = (float(cluster_geoms[a].centroid.x), float(cluster_geoms[a].centroid.y))
        cb = (float(cluster_geoms[b].centroid.x), float(cluster_geoms[b].centroid.y))
        ab = _vec_from_to(ca, cb)
        ba = _vec_from_to(cb, ca)

        rel = str(item.get("relation") or "")
        weight = _priority_weight(item.get("priority"))
        local = 0.0
        dot = None

        viewing_like = _cluster_pair_is_viewing_like(
            cinfo_by_id.get(a, {}),
            cinfo_by_id.get(b, {}),
        )

        if rel == "face_each_other" and front_a is not None and front_b is not None:
            scale = 700.0 if not viewing_like else 1050.0
            p1, d1 = _penalty_face(front_a, ab, scale, weight)
            p2, d2 = _penalty_face(front_b, ba, scale, weight)
            local = p1 + p2
            dot = min(d1, d2)

        elif (
            rel == "avoid_facing_each_other"
            and front_a is not None
            and front_b is not None
        ):
            p1, d1 = _penalty_avoid_face(front_a, ab, 320.0, weight)
            p2, d2 = _penalty_avoid_face(front_b, ba, 320.0, weight)
            local = p1 + p2
            dot = max(d1, d2)

        elif (
            rel in {"same_axis", "parallel_alignment"}
            and axis_a is not None
            and axis_b is not None
        ):
            local, dot = _penalty_parallel(
                axis_a, axis_b, 300.0 if rel == "parallel_alignment" else 360.0, weight
            )

        elif (
            rel == "perpendicular_alignment"
            and axis_a is not None
            and axis_b is not None
        ):
            local, dot = _penalty_perpendicular(axis_a, axis_b, 340.0, weight)

        elif rel in {"access_faces_other", "turn_toward"} and front_a is not None:
            scale = 360.0 if not viewing_like else 520.0
            local, dot = _penalty_face(front_a, ab, scale, weight)

        elif rel == "turn_away" and front_a is not None:
            local, dot = _penalty_away(front_a, ab, 340.0, weight)

        if local > 0.0:
            total += local
            meta.append(
                {
                    "kind": "cluster_directional_relation",
                    "a": a,
                    "b": b,
                    "relation": rel,
                    "penalty_mm": int(round(local)),
                    "dot": None if dot is None else round(dot, 3),
                }
            )

    for item in object_orientations:
        if not isinstance(item, dict):
            continue

        cid = item.get("cluster_id")
        oid = item.get("object_id")
        if not isinstance(cid, str) or not isinstance(oid, str):
            continue
        if cid not in cluster_geoms or cid not in cinfo_by_id:
            continue

        tf = cluster_transforms_by_id.get(cid) or {}
        rot = int(tf.get("rot", 0)) % 360
        cinfo = cinfo_by_id[cid]
        geom = cluster_geoms[cid]
        cluster_centroid = (float(geom.centroid.x), float(geom.centroid.y))
        obj_center = _object_world_center(cinfo, tf, oid)
        if obj_center is None:
            continue

        front = _object_front_vector(cinfo, oid, rot)
        axis = _object_axis_vector(cinfo, oid, rot)
        cluster_axis = _cluster_axis_vector(cinfo, rot)
        weight = _priority_weight(item.get("priority"))

        for intent in item.get("intents") or []:
            local = 0.0
            dot = None
            target_object_id = item.get("target_object_id")
            preferred_target_cluster = item.get("target_object_cluster_id")
            if not isinstance(preferred_target_cluster, str):
                preferred_target_cluster = item.get("target_cluster_id")
            if not isinstance(preferred_target_cluster, str):
                preferred_target_cluster = None
            resolved_target_cluster_id = None

            if (
                intent
                in {
                    "front_to_open_space",
                    "front_to_room_center",
                    "preserve_front_access",
                }
                and front is not None
            ):
                local, dot = _penalty_face(
                    front, _vec_from_to(obj_center, room_center), 320.0, weight
                )

            elif intent == "front_to_cluster_center" and front is not None:
                local, dot = _penalty_face(
                    front, _vec_from_to(obj_center, cluster_centroid), 260.0, weight
                )

            elif intent == "front_to_window" and front is not None:
                info = _nearest_opening_info(
                    opening_ctx.get("windows") or [], obj_center
                )
                if info is not None:
                    local, dot = _penalty_face(
                        front, _vec_from_to(obj_center, info["midpoint"]), 300.0, weight
                    )

            elif intent == "front_to_entry" and front is not None:
                info = _nearest_opening_info(opening_ctx.get("doors") or [], obj_center)
                if info is not None:
                    local, dot = _penalty_face(
                        front, _vec_from_to(obj_center, info["midpoint"]), 340.0, weight
                    )

            elif intent == "back_to_wall" and front is not None:
                wall = _nearest_wall_info(room_poly, obj_center)
                if wall is not None:
                    local, dot = _penalty_face(
                        front, _vec_from_to(wall["midpoint"], obj_center), 360.0, weight
                    )

            elif intent == "long_axis_parallel_wall" and axis is not None:
                wall = _nearest_wall_info(room_poly, obj_center)
                if wall is not None:
                    local, dot = _penalty_parallel(axis, wall["dir"], 260.0, weight)

            elif intent == "long_axis_perpendicular_wall" and axis is not None:
                wall = _nearest_wall_info(room_poly, obj_center)
                if wall is not None:
                    local, dot = _penalty_perpendicular(
                        axis, wall["dir"], 260.0, weight
                    )

            elif (
                intent == "align_with_cluster_axis"
                and axis is not None
                and cluster_axis is not None
            ):
                local, dot = _penalty_parallel(axis, cluster_axis, 240.0, weight)

            elif intent == "face_object" and front is not None:
                if isinstance(target_object_id, str):
                    resolved = _resolve_target_object_world_center(
                        target_object_id=target_object_id,
                        object_index=object_index,
                        cluster_transforms_by_id=cluster_transforms_by_id,
                        preferred_cluster_id=preferred_target_cluster,
                    )
                    if resolved is not None:
                        target_center, resolved_target_cluster_id = resolved
                        if not (
                            resolved_target_cluster_id == cid
                            and target_object_id == oid
                        ):
                            if target_center != obj_center:
                                scale = 600.0
                                if _is_seating_like_object(
                                    oid
                                ) and _is_focal_like_object(target_object_id):
                                    scale = 920.0
                                elif _is_focal_like_object(
                                    oid
                                ) and _is_seating_like_object(target_object_id):
                                    scale = 760.0
                                local, dot = _penalty_face(
                                    front,
                                    _vec_from_to(obj_center, target_center),
                                    scale,
                                    weight,
                                )

            elif intent == "face_away_from_object" and front is not None:
                if isinstance(target_object_id, str):
                    resolved = _resolve_target_object_world_center(
                        target_object_id=target_object_id,
                        object_index=object_index,
                        cluster_transforms_by_id=cluster_transforms_by_id,
                        preferred_cluster_id=preferred_target_cluster,
                    )
                    if resolved is not None:
                        target_center, resolved_target_cluster_id = resolved
                        if not (
                            resolved_target_cluster_id == cid
                            and target_object_id == oid
                        ):
                            if target_center != obj_center:
                                local, dot = _penalty_away(
                                    front,
                                    _vec_from_to(obj_center, target_center),
                                    340.0,
                                    weight,
                                )

            if local > 0.0:
                total += local
                meta.append(
                    {
                        "kind": "object_orientation",
                        "cluster_id": cid,
                        "object_id": oid,
                        "intent": intent,
                        "target_object_id": target_object_id,
                        "resolved_target_cluster_id": resolved_target_cluster_id,
                        "penalty_mm": int(round(local)),
                        "dot": None if dot is None else round(dot, 3),
                    }
                )

    meta.sort(
        key=lambda x: (
            -int(x.get("penalty_mm", 0)),
            str(x.get("kind", "")),
            str(x.get("cluster_id", x.get("a", ""))),
        )
    )
    return total, meta


def _orientation_meta(cinfo: Dict[str, Any]) -> Dict[str, Any]:
    value = cinfo.get("orientation_meta")
    return value if isinstance(value, dict) else {}


def _parse_vec2(value: Any) -> Tuple[float, float] | None:
    if isinstance(value, dict):
        dx = value.get("dx")
        dy = value.get("dy")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        dx, dy = value[0], value[1]
    else:
        return None
    try:
        dx = float(dx)
        dy = float(dy)
    except Exception:
        return None
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return _normalize_vec((dx, dy))


def _normalize_vec(vec: Tuple[float, float]) -> Tuple[float, float] | None:
    dx, dy = float(vec[0]), float(vec[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return (dx / norm, dy / norm)


def _rotate_vec_ccw_90s(
    vec: Tuple[float, float], rot: int
) -> Tuple[float, float] | None:
    v = _normalize_vec(vec)
    if v is None:
        return None
    x, y = v
    r = rot % 360
    if r == 0:
        return (x, y)
    if r == 90:
        return (-y, x)
    if r == 180:
        return (-x, -y)
    if r == 270:
        return (y, -x)
    return (x, y)


def _rotate_point_ccw_90s(x: float, y: float, rot: int) -> Tuple[float, float]:
    r = rot % 360
    if r == 0:
        return (x, y)
    if r == 90:
        return (-y, x)
    if r == 180:
        return (-x, -y)
    if r == 270:
        return (y, -x)
    return (x, y)


def _transform_local_point(
    x: float, y: float, tx: float, ty: float, rot: int
) -> Tuple[float, float]:
    rx, ry = _rotate_point_ccw_90s(x, y, rot)
    return (rx + tx, ry + ty)


def _cluster_front_vector(
    cinfo: Dict[str, Any], rot: int
) -> Tuple[float, float] | None:
    meta = _orientation_meta(cinfo)
    front = _parse_vec2(meta.get("cluster_front_local") or meta.get("front_local"))
    if front is None:
        return None
    return _rotate_vec_ccw_90s(front, rot)


def _cluster_axis_vector(cinfo: Dict[str, Any], rot: int) -> Tuple[float, float] | None:
    meta = _orientation_meta(cinfo)
    axis = _parse_vec2(meta.get("cluster_axis_local") or meta.get("axis_local"))
    if axis is None:
        return None
    return _rotate_vec_ccw_90s(axis, rot)


def _object_meta(cinfo: Dict[str, Any], object_id: str) -> Dict[str, Any]:
    meta = _orientation_meta(cinfo)
    objs = meta.get("important_objects")
    if not isinstance(objs, dict):
        return {}
    item = objs.get(object_id)
    return item if isinstance(item, dict) else {}


def _object_front_vector(
    cinfo: Dict[str, Any], object_id: str, rot: int
) -> Tuple[float, float] | None:
    meta = _object_meta(cinfo, object_id)
    front = _parse_vec2(meta.get("front_local"))
    if front is None:
        front = _parse_vec2(_orientation_meta(cinfo).get("cluster_front_local"))
    if front is None:
        return None
    return _rotate_vec_ccw_90s(front, rot)


def _object_axis_vector(
    cinfo: Dict[str, Any], object_id: str, rot: int
) -> Tuple[float, float] | None:
    meta = _object_meta(cinfo, object_id)
    axis = _parse_vec2(meta.get("axis_local"))
    if axis is None:
        axis = _parse_vec2(_orientation_meta(cinfo).get("cluster_axis_local"))
    if axis is None:
        return None
    return _rotate_vec_ccw_90s(axis, rot)


def _object_local_center(
    cinfo: Dict[str, Any], object_id: str
) -> Tuple[float, float] | None:
    fp = cinfo.get("cluster_footprint") or {}
    rects = fp.get("rects")
    if not isinstance(rects, list):
        return None
    for r in rects:
        if not isinstance(r, dict):
            continue
        if r.get("id") != object_id:
            continue
        try:
            x = float(r.get("x", 0))
            y = float(r.get("y", 0))
            w = float(r.get("w", 0))
            h = float(r.get("h", 0))
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return (x + w / 2.0, y + h / 2.0)
    return None


def _object_world_center(
    cinfo: Dict[str, Any],
    tf: Dict[str, Any],
    object_id: str,
) -> Tuple[float, float] | None:
    local = _object_local_center(cinfo, object_id)
    if local is None:
        return None
    try:
        x = float(tf.get("x", 0))
        y = float(tf.get("y", 0))
        rot = int(tf.get("rot", 0)) % 360
    except Exception:
        return None
    return _transform_local_point(local[0], local[1], x, y, rot)


def _build_global_object_index(
    cinfo_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}

    for cid, cinfo in cinfo_by_id.items():
        fp = cinfo.get("cluster_footprint") or {}
        rects = fp.get("rects")
        if not isinstance(rects, list):
            continue

        seen_in_cluster: set[str] = set()
        for r in rects:
            if not isinstance(r, dict):
                continue
            oid = r.get("id")
            if not isinstance(oid, str) or not oid:
                continue
            if oid in seen_in_cluster:
                continue
            seen_in_cluster.add(oid)
            index.setdefault(oid, []).append(
                {
                    "cluster_id": cid,
                    "cinfo": cinfo,
                }
            )

    return index


def _resolve_target_object_world_center(
    *,
    target_object_id: str,
    object_index: Dict[str, List[Dict[str, Any]]],
    cluster_transforms_by_id: Dict[str, Dict[str, Any]],
    preferred_cluster_id: str | None = None,
) -> Tuple[Tuple[float, float], str] | None:
    candidates = object_index.get(target_object_id) or []
    if not candidates:
        return None

    ordered: List[Dict[str, Any]] = []
    if isinstance(preferred_cluster_id, str) and preferred_cluster_id:
        ordered.extend(
            [x for x in candidates if x.get("cluster_id") == preferred_cluster_id]
        )
        ordered.extend(
            [x for x in candidates if x.get("cluster_id") != preferred_cluster_id]
        )
    else:
        ordered = list(candidates)

    for item in ordered:
        cid = item.get("cluster_id")
        cinfo = item.get("cinfo")
        if not isinstance(cid, str) or not isinstance(cinfo, dict):
            continue
        tf = cluster_transforms_by_id.get(cid)
        if not isinstance(tf, dict):
            continue

        center = _object_world_center(cinfo, tf, target_object_id)
        if center is not None:
            return center, cid

    return None


def _vec_from_to(
    a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[float, float] | None:
    return _normalize_vec((float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def _dot_unit(
    a: Tuple[float, float] | None, b: Tuple[float, float] | None
) -> float | None:
    va = _normalize_vec(a) if a is not None else None
    vb = _normalize_vec(b) if b is not None else None
    if va is None or vb is None:
        return None
    return _clamp((va[0] * vb[0]) + (va[1] * vb[1]), -1.0, 1.0)


def _penalty_face(
    actual_vec: Tuple[float, float] | None,
    target_vec: Tuple[float, float] | None,
    scale: float,
    weight: float,
) -> tuple[float, float | None]:
    dot = _dot_unit(actual_vec, target_vec)
    if dot is None:
        return 0.0, None
    penalty = max(0.0, 1.0 - dot) * scale * weight
    return penalty, dot


def _penalty_away(
    actual_vec: Tuple[float, float] | None,
    target_vec: Tuple[float, float] | None,
    scale: float,
    weight: float,
) -> tuple[float, float | None]:
    dot = _dot_unit(actual_vec, target_vec)
    if dot is None:
        return 0.0, None
    desired_dot = -dot
    penalty = max(0.0, 1.0 - desired_dot) * scale * weight
    return penalty, desired_dot


def _penalty_avoid_face(
    actual_vec: Tuple[float, float] | None,
    target_vec: Tuple[float, float] | None,
    scale: float,
    weight: float,
) -> tuple[float, float | None]:
    dot = _dot_unit(actual_vec, target_vec)
    if dot is None:
        return 0.0, None
    penalty = max(0.0, dot) * scale * weight
    return penalty, dot


def _penalty_parallel(
    a: Tuple[float, float] | None,
    b: Tuple[float, float] | None,
    scale: float,
    weight: float,
) -> tuple[float, float | None]:
    dot = _dot_unit(a, b)
    if dot is None:
        return 0.0, None
    score = abs(dot)
    penalty = max(0.0, 1.0 - score) * scale * weight
    return penalty, score


def _penalty_perpendicular(
    a: Tuple[float, float] | None,
    b: Tuple[float, float] | None,
    scale: float,
    weight: float,
) -> tuple[float, float | None]:
    dot = _dot_unit(a, b)
    if dot is None:
        return 0.0, None
    score = abs(dot)
    penalty = score * scale * weight
    return penalty, score


def _project_point_to_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[Tuple[float, float], float]:
    vx = x2 - x1
    vy = y2 - y1
    denom = (vx * vx) + (vy * vy)
    if denom <= 1e-9:
        qx, qy = x1, y1
        return (qx, qy), math.hypot(px - qx, py - qy)

    t = ((px - x1) * vx + (py - y1) * vy) / denom
    t = _clamp(t, 0.0, 1.0)
    qx = x1 + t * vx
    qy = y1 + t * vy
    return (qx, qy), math.hypot(px - qx, py - qy)


def _nearest_wall_info(
    room_poly: Any, point: Tuple[float, float]
) -> Dict[str, Any] | None:
    try:
        coords = list(room_poly.exterior.coords)
    except Exception:
        return None
    if len(coords) < 2:
        return None

    px, py = float(point[0]), float(point[1])
    best = None
    best_dist = None

    for i in range(len(coords) - 1):
        x1, y1 = float(coords[i][0]), float(coords[i][1])
        x2, y2 = float(coords[i + 1][0]), float(coords[i + 1][1])
        proj, dist = _project_point_to_segment(px, py, x1, y1, x2, y2)
        seg_dir = _normalize_vec((x2 - x1, y2 - y1))
        if seg_dir is None:
            continue
        item = {
            "midpoint": proj,
            "dir": seg_dir,
            "distance_mm": dist,
        }
        if best_dist is None or dist < best_dist:
            best = item
            best_dist = dist

    return best


def _nearest_opening_info(
    items: List[Dict[str, Any]], point: Tuple[float, float]
) -> Dict[str, Any] | None:
    px, py = float(point[0]), float(point[1])
    best = None
    best_dist = None

    for item in items:
        line = item.get("line")
        midpoint = item.get("midpoint")
        if line is None or not isinstance(midpoint, tuple) or len(midpoint) != 2:
            continue
        try:
            coords = list(line.coords)
        except Exception:
            continue
        if len(coords) < 2:
            continue
        x1, y1 = float(coords[0][0]), float(coords[0][1])
        x2, y2 = float(coords[-1][0]), float(coords[-1][1])
        seg_dir = _normalize_vec((x2 - x1, y2 - y1))
        if seg_dir is None:
            continue
        dist = math.hypot(float(midpoint[0]) - px, float(midpoint[1]) - py)
        payload = {
            "id": item.get("id"),
            "midpoint": (float(midpoint[0]), float(midpoint[1])),
            "dir": seg_dir,
            "distance_mm": dist,
        }
        if best_dist is None or dist < best_dist:
            best = payload
            best_dist = dist

    return best


def _cluster_object_ids(cinfo: Dict[str, Any]) -> List[str]:
    ids: List[str] = []

    fp = cinfo.get("cluster_footprint") or {}
    rects = fp.get("rects")
    if isinstance(rects, list):
        for r in rects:
            if not isinstance(r, dict):
                continue
            oid = r.get("id")
            if isinstance(oid, str) and oid and oid not in ids:
                ids.append(oid)

    members = cinfo.get("members")
    if isinstance(members, list):
        for oid in members:
            if isinstance(oid, str) and oid and oid not in ids:
                ids.append(oid)

    local_placements = cinfo.get("local_placements")
    if isinstance(local_placements, list):
        for p in local_placements:
            if not isinstance(p, dict):
                continue
            oid = p.get("id")
            if isinstance(oid, str) and oid and oid not in ids:
                ids.append(oid)

    return ids


def _is_seating_like_object(object_id: str) -> bool:
    key = object_id.lower()
    return any(
        token in key
        for token in [
            "sofa",
            "sectional_sofa",
            "loveseat",
            "armchair",
            "accent_chair",
            "lounge_chair",
            "chair",
            "bench",
            "recliner",
            "main_chair",
        ]
    )


def _is_focal_like_object(object_id: str) -> bool:
    key = object_id.lower()
    return any(
        token in key
        for token in [
            "tv_console",
            "media_console",
            "media_shelf",
            "tv",
            "screen",
            "projector_screen",
            "fireplace",
            "stage_focal",
        ]
    )


def _cluster_pair_is_viewing_like(
    cinfo_a: Dict[str, Any],
    cinfo_b: Dict[str, Any],
) -> bool:
    ids_a = _cluster_object_ids(cinfo_a)
    ids_b = _cluster_object_ids(cinfo_b)

    a_has_seating = any(_is_seating_like_object(x) for x in ids_a)
    a_has_focal = any(_is_focal_like_object(x) for x in ids_a)
    b_has_seating = any(_is_seating_like_object(x) for x in ids_b)
    b_has_focal = any(_is_focal_like_object(x) for x in ids_b)

    return (a_has_seating and b_has_focal) or (b_has_seating and a_has_focal)


def _adaptive_base_gap_mm(
    *,
    free_room_area_mm2: float,
    occupied_area_mm2: float,
    base_preferred_gap_mm: int | None,
    min_preferred_gap_mm: int | None,
    max_preferred_gap_mm: int | None,
) -> float:
    min_preferred_gap_mm, max_preferred_gap_mm = _normalize_gap_bounds(
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )
    density = occupied_area_mm2 / free_room_area_mm2 if free_room_area_mm2 > 0 else 1.0
    seed = float(base_preferred_gap_mm) if base_preferred_gap_mm is not None else 320.0
    density_factor = _clamp(1.25 - density, 0.65, 1.20)
    base_gap = seed * density_factor
    return _clamp(base_gap, float(min_preferred_gap_mm), float(max_preferred_gap_mm))


def _pair_preferred_gap_mm(
    *,
    ga: Any,
    gb: Any,
    adaptive_base_gap_mm: float,
    min_preferred_gap_mm: int | None,
    max_preferred_gap_mm: int | None,
) -> float:
    min_preferred_gap_mm, max_preferred_gap_mm = _normalize_gap_bounds(
        min_preferred_gap_mm=min_preferred_gap_mm,
        max_preferred_gap_mm=max_preferred_gap_mm,
    )

    diag_a = _geom_diag_mm(ga)
    diag_b = _geom_diag_mm(gb)
    small_diag = min(diag_a, diag_b)

    size_bonus = _clamp(0.06 * small_diag, 0.0, 120.0)
    pref = adaptive_base_gap_mm + size_bonus
    return _clamp(pref, float(min_preferred_gap_mm), float(max_preferred_gap_mm))


def _build_opening_context(room_model_u: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from shapely.geometry import LineString
    except Exception:
        return {"doors": [], "windows": [], "door_lines": [], "window_lines": []}

    openings = room_model_u.get("openings")
    if not isinstance(openings, dict):
        openings = room_model_u.get("openings") or {}
    if not isinstance(openings, dict):
        openings = (room_model_u.get("room") or {}).get("openings") or {}
    if not isinstance(openings, dict):
        openings = {}

    out = {"doors": [], "windows": [], "door_lines": [], "window_lines": []}

    for key in ("doors", "windows"):
        items = openings.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            oid = str(item.get("id") or "")
            seg = item.get("segment_mm") or []
            if not oid or not isinstance(seg, list) or len(seg) != 2:
                continue
            try:
                p1 = (float(seg[0]["x"]), float(seg[0]["y"]))
                p2 = (float(seg[1]["x"]), float(seg[1]["y"]))
            except Exception:
                continue
            line = LineString([p1, p2])
            mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            payload = {"id": oid, "line": line, "midpoint": mid}
            out[key].append(payload)
            if key == "doors":
                out["door_lines"].append(line)
            else:
                out["window_lines"].append(line)

    return out


def _find_opening_midpoint(
    items: List[Dict[str, Any]], oid: str
) -> Tuple[float, float] | None:
    for item in items:
        if item.get("id") == oid:
            mid = item.get("midpoint")
            if isinstance(mid, tuple) and len(mid) == 2:
                return (float(mid[0]), float(mid[1]))
    return None


def _find_opening_line(items: List[Dict[str, Any]], oid: str) -> Any | None:
    for item in items:
        if item.get("id") == oid:
            return item.get("line")
    return None


def _nearest_opening_midpoint_to_point(
    items: List[Dict[str, Any]],
    point: Tuple[float, float],
) -> Tuple[float, float] | None:
    best = None
    best_d = None
    px, py = point
    for item in items:
        mid = item.get("midpoint")
        if not isinstance(mid, tuple) or len(mid) != 2:
            continue
        d = math.hypot(float(mid[0]) - px, float(mid[1]) - py)
        if best_d is None or d < best_d:
            best_d = d
            best = (float(mid[0]), float(mid[1]))
    return best


def _min_geom_to_lines_distance(geom: Any, lines: List[Any]) -> float | None:
    if not lines:
        return None
    best = None
    for line in lines:
        try:
            d = float(geom.distance(line))
        except Exception:
            continue
        if best is None or d < best:
            best = d
    return best


def _min_geom_to_polys_distance(geom: Any, polys: List[Any]) -> float | None:
    if not polys:
        return None
    best = None
    for poly in polys:
        try:
            d = float(geom.distance(poly))
        except Exception:
            continue
        if best is None or d < best:
            best = d
    return best


def _priority_weight(value: Any) -> float:
    v = str(value or "").lower()
    if v == "high":
        return 1.5
    if v == "medium":
        return 1.0
    if v == "low":
        return 0.6
    return 1.0


def _empty_quality() -> Dict[str, Any]:
    return {
        "density_ratio": None,
        "adaptive_base_preferred_gap_mm": None,
        "min_cluster_gap_mm": None,
        "avg_cluster_gap_mm": None,
        "tight_gap_penalty_mm": None,
        "tight_gap_pairs": [],
        "spread_penalty_mm": None,
        "affinity_penalty_mm": None,
        "relation_penalty_mm": None,
        "circulation_penalty_mm": None,
        "orientation_penalty_mm": None,
        "total_soft_penalty_mm": None,
        "orientation_debug": [],
        "layout_score": None,
    }


def _compact_quality_for_output(quality: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(quality, dict):
        return _empty_quality()

    def _as_list(value: Any, limit: int) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in value[:limit]:
            if isinstance(item, dict):
                out.append(item)
        return out

    return {
        "density_ratio": quality.get("density_ratio"),
        "adaptive_base_preferred_gap_mm": quality.get("adaptive_base_preferred_gap_mm"),
        "min_cluster_gap_mm": quality.get("min_cluster_gap_mm"),
        "avg_cluster_gap_mm": quality.get("avg_cluster_gap_mm"),
        "tight_gap_penalty_mm": quality.get("tight_gap_penalty_mm"),
        "tight_gap_pairs": _as_list(quality.get("tight_gap_pairs"), 8),
        "spread_penalty_mm": quality.get("spread_penalty_mm"),
        "affinity_penalty_mm": quality.get("affinity_penalty_mm"),
        "relation_penalty_mm": quality.get("relation_penalty_mm"),
        "circulation_penalty_mm": quality.get("circulation_penalty_mm"),
        "orientation_penalty_mm": quality.get("orientation_penalty_mm"),
        "total_soft_penalty_mm": quality.get("total_soft_penalty_mm"),
        "spread": quality.get("spread")
        if isinstance(quality.get("spread"), dict)
        else {},
        "affinity_debug": _as_list(quality.get("affinity_debug"), 8),
        "relation_debug": _as_list(quality.get("relation_debug"), 8),
        "circulation_debug": _as_list(quality.get("circulation_debug"), 8),
        "orientation_debug": _as_list(quality.get("orientation_debug"), 12),
        "layout_score": quality.get("layout_score"),
    }


def _compact_quality_for_candidate(quality: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(quality, dict):
        return {
            "layout_score": None,
            "total_soft_penalty_mm": None,
            "orientation_penalty_mm": None,
            "circulation_penalty_mm": None,
            "affinity_penalty_mm": None,
            "min_cluster_gap_mm": None,
        }

    return {
        "layout_score": quality.get("layout_score"),
        "total_soft_penalty_mm": quality.get("total_soft_penalty_mm"),
        "orientation_penalty_mm": quality.get("orientation_penalty_mm"),
        "circulation_penalty_mm": quality.get("circulation_penalty_mm"),
        "affinity_penalty_mm": quality.get("affinity_penalty_mm"),
        "min_cluster_gap_mm": quality.get("min_cluster_gap_mm"),
    }


def _safe_int_default(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return int(default)


def _compact_postcheck(post: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(post, dict):
        return {}

    q = post.get("quality")
    if not isinstance(q, dict):
        q = {}

    hard_violation_score = _safe_int_default(
        post.get("hard_violation_score"),
        _safe_int_default(post.get("violation_score"), 10**9),
    )
    violation_score = _safe_int_default(
        post.get("violation_score"),
        hard_violation_score,
    )
    remaining_hard_issues = _safe_int_default(
        post.get("remaining_hard_issues"),
        10**9,
    )
    soft_penalty_mm = _safe_int_default(post.get("soft_penalty_mm"), 10**9)
    layout_score = _safe_int_default(post.get("layout_score"), -(10**9))
    cluster_orientation_penalty_mm = _safe_int_default(
        post.get("cluster_orientation_penalty_mm"),
        _safe_int_default(q.get("orientation_penalty_mm"), 10**9),
    )
    global_orientation_penalty_mm = _safe_int_default(
        post.get("global_orientation_penalty_mm"),
        _safe_int_default(q.get("orientation_penalty_mm"), 10**9),
    )

    out = {
        "valid_geom": bool(post.get("valid_geom", False)),
        "in_room": bool(post.get("in_room", False)),
        "hit_obstacles": post.get("hit_obstacles", [])
        if isinstance(post.get("hit_obstacles"), list)
        else [],
        "overlaps": post.get("overlaps", [])
        if isinstance(post.get("overlaps"), list)
        else [],
        "solves_trigger": bool(post.get("solves_trigger", False)),
        "hard_violation_score": hard_violation_score,
        "violation_score": violation_score,
        "soft_penalty_mm": soft_penalty_mm,
        "layout_score": layout_score,
        "remaining_hard_issues": remaining_hard_issues,
        "all_hard_clear": bool(post.get("all_hard_clear", False)),
        "cluster_orientation_penalty_mm": cluster_orientation_penalty_mm,
        "global_orientation_penalty_mm": global_orientation_penalty_mm,
        "quality": _compact_quality_for_candidate(q),
    }

    if isinstance(post.get("patch_cluster_ids"), list):
        out["patch_cluster_ids"] = post.get("patch_cluster_ids")

    return out


def _violation_score(v: Dict[str, Any]) -> int:
    """
    HARD GEOMETRY ONLY.
    Không cộng orientation vào đây nữa.
    """
    score = 0
    if v.get("out_of_bounds"):
        score += 1000
    score += 200 * len(v.get("hit_obstacles") or [])
    score += 300 * len(v.get("overlaps") or [])
    return score


def _cluster_orientation_penalty_from_quality(
    quality: Dict[str, Any] | None,
    cid: str,
) -> int:
    if not isinstance(quality, dict):
        return 0
    return int(
        _cluster_orientation_penalties(quality.get("orientation_debug")).get(cid, 0)
        or 0
    )


def _hard_issue_count(v: Dict[str, Any]) -> int:
    return (
        int(bool(v.get("out_of_bounds")))
        + len(v.get("hit_obstacles") or [])
        + len(v.get("overlaps") or [])
    )


def _geom_diag_mm(geom: Any) -> float:
    x1, y1, x2, y2 = geom.bounds
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    return math.hypot(dx, dy)


def _compute_free_room_area_mm2(room_poly: Any, obstacle_union: Any | None) -> float:
    if obstacle_union is None:
        return max(float(room_poly.area), 1.0)
    try:
        free_geom = room_poly.difference(obstacle_union)
        return max(float(free_geom.area), 1.0)
    except Exception:
        return max(float(room_poly.area), 1.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _round3(x: float) -> float:
    return round(float(x), 3)


def _snap_grid(value: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return int(value)
    q = int(round(value / grid_mm))
    return q * grid_mm


def _build_cluster_polys(
    PolygonCls, cinfo: Dict[str, Any], x: int, y: int, rot: int
) -> List[Any]:
    polys: List[Any] = []
    fp = cinfo.get("cluster_footprint") or {}

    rects = fp.get("rects")
    if isinstance(rects, list) and rects:
        for r in rects:
            if not isinstance(r, dict):
                continue
            rx = int(r.get("x", 0))
            ry = int(r.get("y", 0))
            w = int(r.get("w", 0))
            h = int(r.get("h", 0))
            if w <= 0 or h <= 0:
                continue
            pts = [(rx, ry), (rx + w, ry), (rx + w, ry + h), (rx, ry + h)]
            pts_xy = []
            for px, py in pts:
                tx, ty = _rotate_ccw_90s(px, py, rot)
                pts_xy.append((tx + x, ty + y))
            poly = PolygonCls(pts_xy)
            poly = _fix_poly(poly)
            if not poly.is_empty:
                polys.append(poly)
        if polys:
            return polys

    outlines = fp.get("outline_polygons_ccw") or cinfo.get("outline_polygons_ccw") or []
    if isinstance(outlines, list):
        for poly_pts in outlines:
            if not isinstance(poly_pts, list) or len(poly_pts) < 3:
                continue
            pts_xy = []
            for p in poly_pts:
                px, py = int(p["x"]), int(p["y"])
                tx, ty = _rotate_ccw_90s(px, py, rot)
                pts_xy.append((tx + x, ty + y))
            poly = PolygonCls(pts_xy)
            poly = _fix_poly(poly)
            if not poly.is_empty:
                polys.append(poly)

    return polys


def _cluster_bbox_for_transform(
    cinfo: Dict[str, Any],
    x: int,
    y: int,
    rot: int,
) -> Tuple[float, float, float, float] | None:
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except Exception:
        return None
    polys = _build_cluster_polys(Polygon, cinfo, x, y, rot)
    if not polys:
        return None
    geom = unary_union(polys)
    geom = _fix_geom(geom)
    if geom.is_empty:
        return None
    return geom.bounds


def _rotate_ccw_90s(x: int, y: int, rot: int) -> Tuple[int, int]:
    r = rot % 360
    if r == 0:
        return x, y
    if r == 90:
        return -y, x
    if r == 180:
        return -x, -y
    if r == 270:
        return y, -x
    return x, y


def _unwrap_any(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    for key in ("parsed", "raw"):
        sub = obj.get(key)
        if isinstance(sub, dict):
            if "room" in sub and isinstance(sub.get("room"), dict):
                return sub
            if _looks_like_cluster_map(sub):
                return sub
    return obj


def _looks_like_cluster_map(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    for _, v in d.items():
        if (
            isinstance(v, dict)
            and ("cluster_id" in v)
            and (("cluster_footprint" in v) or ("outline_polygons_ccw" in v))
        ):
            return True
    return False


def _iter_cluster_entries(clusters_outlines: Any) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(clusters_outlines, dict):
        for cid, cinfo in clusters_outlines.items():
            if isinstance(cid, str) and cid and isinstance(cinfo, dict):
                if "cluster_id" in cinfo and (
                    ("cluster_footprint" in cinfo) or ("outline_polygons_ccw" in cinfo)
                ):
                    out.append((cid, cinfo))
        if out:
            return out

        clusters_list = clusters_outlines.get("clusters")
        if isinstance(clusters_list, list):
            for cinfo in clusters_list:
                if not isinstance(cinfo, dict):
                    continue
                cid = cinfo.get("cluster_id")
                if isinstance(cid, str) and cid:
                    out.append((cid, cinfo))
            return out

    if isinstance(clusters_outlines, list):
        for cinfo in clusters_outlines:
            if not isinstance(cinfo, dict):
                continue
            cid = cinfo.get("cluster_id")
            if isinstance(cid, str) and cid:
                out.append((cid, cinfo))
        return out

    return out


def _fix_poly(poly):
    try:
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        pass
    return poly


def _fix_geom(geom):
    try:
        if hasattr(geom, "is_valid") and not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        pass
    return geom


TOOL_REGISTRY: Dict[str, Any] = {
    "GlobalClusterVerifier": GlobalClusterVerifier,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "GlobalClusterVerifier",
            "description": (
                "Global verifier (CCW): checks in-room, obstacle intersection, cluster overlaps, "
                "and optionally treats significant orientation mismatch as HARD INVALID when relation_plan "
                "contains orientation constraints. Uses rects if available, otherwise outline polygons. "
                "Returns adaptive soft spacing quality, repair_hints for INVALID layouts, and improve_hints "
                "for orientation/circulation/layout refinement. Supports cross-cluster target_object_id "
                "resolution in object_orientations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room_model": {"type": "object"},
                    "clusters_outlines": {"type": "object"},
                    "cluster_transforms": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "grid_mm": {"type": "integer"},
                    "eps_area": {"type": "number"},
                    "forbid_obstacle_touch": {"type": "boolean"},
                    "generate_hints": {"type": "boolean"},
                    "nudge_steps": {"type": "integer"},
                    "max_candidates_per_issue": {"type": "integer"},
                    "margin_mm": {"type": ["integer", "null"]},
                    "consider_rotations": {"type": "boolean"},
                    "relation_plan": {"type": ["object", "null"]},
                    "base_preferred_gap_mm": {"type": ["integer", "null"]},
                    "min_preferred_gap_mm": {"type": ["integer", "null"]},
                    "max_preferred_gap_mm": {"type": ["integer", "null"]},
                    "hard_orientation_gate": {"type": "boolean"},
                    "orientation_hard_threshold_mm": {"type": "integer"},
                },
                "required": [
                    "room_model",
                    "clusters_outlines",
                    "cluster_transforms",
                    "grid_mm",
                ],
            },
        },
    }
]
