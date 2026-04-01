from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from agent_schema.cluster_relation_planner_schema import ClusterRelationPlannerOutput
from clients.base_client import ChatMessage
from clients.llm_client import get_llm_client
from config.gemini_config import GeminiConfig
from prompt.cluster_relation_planner import CLUSTER_RELATION_PLANNER_PROMPT
from prompt.system import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

SEATING_ANCHOR_PATTERNS = [
    "sectional_sofa",
    "loveseat",
    "recliner",
    "accent_chair",
    "lounge_chair",
    "main_chair",
    "armchair",
    "sofa",
    "bench",
    "chair",
]

FOCAL_ANCHOR_PATTERNS = [
    "projector_screen",
    "media_console",
    "media_shelf",
    "tv_console",
    "fireplace",
    "stage_focal",
    "screen",
    "tv",
]

ACCESS_SENSITIVE_PATTERNS = [
    "fridge",
    "wardrobe",
    "dresser",
    "cabinet",
    "bookshelf",
    "storage",
    "console_table",
    "console",
    "pantry",
    "dishwasher",
    "washer",
    "dryer",
    "vanity",
    "toilet",
]

MEDIA_CLUSTER_HINTS = [
    "tv",
    "media",
    "screen",
    "projector",
    "fireplace",
]

STORAGE_CLUSTER_HINTS = [
    "storage",
    "bookshelf",
    "cabinet",
    "wardrobe",
    "pantry",
    "dresser",
]


@dataclass(frozen=True)
class ClusterRelationPlanner:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = CLUSTER_RELATION_PLANNER_PROMPT

    def build_messages(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_json: dict[str, Any],
        description: str | None = None,
        special_notes: str | None = None,
    ) -> list[ChatMessage]:
        room_min = _minify_room_model(room_model_json)
        clusters_min = _minify_clusters(clusters_json)

        user_prompt = (
            self.prompt_template.replace("{ROOM_MODEL_JSON}", _json_block(room_min))
            .replace("{CLUSTERS_JSON}", _json_block(clusters_min))
            .replace("{DESCRIPTION}", (description or "").strip())
            .replace("{SPECIAL_NOTES}", (special_notes or "").strip())
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_raw(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_json: dict[str, Any],
        description: str | None = None,
        special_notes: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        t0 = perf_counter()
        messages = self.build_messages(
            room_model_json=room_model_json,
            clusters_json=clusters_json,
            description=description,
            special_notes=special_notes,
        )
        logger.info(
            "ClusterRelationPlanner start room_id=%s clusters=%s",
            _extract_room_id(room_model_json),
            len(_extract_clusters_map(clusters_json)),
        )

        client = get_llm_client(provider="gemini")
        last_raw = ""

        for attempt in range(2):
            logger.info("ClusterRelationPlanner attempt %s/2", attempt + 1)
            response = client.chat_completion(
                messages,
                model_key="primary",
                model_name=GeminiConfig.AGENT_MODELS.get("planner"),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            last_raw = _extract_content(response)
            logger.info(
                "ClusterRelationPlanner response length=%s",
                len(last_raw) if isinstance(last_raw, str) else 0,
            )

            try:
                _parse_json(last_raw)
                logger.info(
                    "ClusterRelationPlanner parsed OK in %.2fs",
                    perf_counter() - t0,
                )
                return last_raw
            except ValueError:
                if attempt == 1:
                    break
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return valid JSON only. "
                            "Do not include markdown fences, explanations, or prose outside the JSON object."
                        ),
                    }
                )

        logger.warning(
            "ClusterRelationPlanner returning unparsed output after %.2fs",
            perf_counter() - t0,
        )
        return last_raw

    def generate(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_json: dict[str, Any],
        description: str | None = None,
        special_notes: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        t0 = perf_counter()
        raw = self.generate_raw(
            room_model_json=room_model_json,
            clusters_json=clusters_json,
            description=description,
            special_notes=special_notes,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        payload = _parse_json(raw)
        payload = _normalize_relation_plan_payload(
            payload,
            room_model_json=room_model_json,
            clusters_json=clusters_json,
        )

        if "room_id" not in payload or not isinstance(payload.get("room_id"), str):
            payload["room_id"] = _extract_room_id(room_model_json)

        result = ClusterRelationPlannerOutput.model_validate(payload).model_dump(
            by_alias=True
        )
        logger.info(
            "ClusterRelationPlanner done in %.2fs status=%s",
            perf_counter() - t0,
            result.get("status"),
        )
        return result


def _minify_room_model(room_model_json: dict[str, Any]) -> dict[str, Any]:
    src = _unwrap_room_model(room_model_json)

    room = src.get("room")
    openings = src.get("openings")
    obstacles = src.get("obstacles")
    meta = src.get("meta")

    room = room if isinstance(room, dict) else {}
    openings = openings if isinstance(openings, dict) else {}
    obstacles = obstacles if isinstance(obstacles, list) else []
    meta = meta if isinstance(meta, dict) else {}

    polygon = room.get("polygon_ccw")
    polygon = polygon if isinstance(polygon, list) else []

    bbox = _bbox_from_points(polygon)
    area_mm2 = _polygon_area_mm2(polygon)

    return {
        "room": {
            "room_id": str(room.get("room_id") or "unknown"),
            "name": room.get("name"),
            "polygon_ccw": polygon,
            "bbox": bbox,
            "area_mm2": area_mm2,
        },
        "openings": {
            "doors": _normalize_doors(openings.get("doors")),
            "windows": _normalize_windows(openings.get("windows")),
        },
        "obstacles": _normalize_obstacles(obstacles),
        "meta": {
            "room_type": meta.get("room_type"),
            "style": meta.get("style"),
            "height_mm": meta.get("height_mm"),
            "window_direction": meta.get("window_direction"),
        },
    }


def _minify_clusters(clusters_json: dict[str, Any]) -> dict[str, Any]:
    clusters = _extract_clusters_map(clusters_json)
    out: dict[str, Any] = {}

    for cluster_id in sorted(clusters.keys()):
        cluster = clusters[cluster_id]
        if not isinstance(cluster, dict):
            continue

        fp = cluster.get("cluster_footprint")
        fp = fp if isinstance(fp, dict) else {}

        rects = fp.get("rects")
        rects = rects if isinstance(rects, list) else []

        bbox = fp.get("local_bbox")
        bbox = bbox if isinstance(bbox, dict) else {}

        object_ids = _extract_cluster_object_ids(cluster)

        members = cluster.get("members")
        members = members if isinstance(members, list) else object_ids

        anchors = cluster.get("anchors")
        anchors = (
            [x for x in anchors if isinstance(x, str)]
            if isinstance(anchors, list)
            else []
        )

        footprint_area_mm2 = _sum_rect_area_mm2(rects)
        bbox_w = int(bbox.get("max_x", 0) or 0) - int(bbox.get("min_x", 0) or 0)
        bbox_h = int(bbox.get("max_y", 0) or 0) - int(bbox.get("min_y", 0) or 0)

        seating_candidates = _pick_object_ids_by_priority(
            object_ids,
            SEATING_ANCHOR_PATTERNS,
        )
        focal_candidates = _pick_object_ids_by_priority(
            object_ids,
            FOCAL_ANCHOR_PATTERNS,
        )
        access_candidates = _collect_access_sensitive_object_ids(cluster)

        out[cluster_id] = {
            "cluster_id": str(cluster.get("cluster_id") or cluster_id),
            "status": cluster.get("status"),
            "members": [m for m in members if isinstance(m, str)],
            "object_ids": object_ids,
            "member_count": len(object_ids),
            "anchors": anchors,
            "likely_role": _classify_cluster_role(object_ids, str(cluster_id)),
            "seating_anchor_candidates": seating_candidates,
            "focal_anchor_candidates": focal_candidates,
            "access_sensitive_object_ids": access_candidates,
            "hard_constraints": _normalize_constraints(cluster.get("hard_constraints")),
            "soft_constraints": _normalize_constraints(cluster.get("soft_constraints")),
            "cluster_rules": _normalize_cluster_rules(cluster.get("cluster_rules")),
            "orientation_meta": cluster.get("orientation_meta")
            if isinstance(cluster.get("orientation_meta"), dict)
            else {},
            "cluster_footprint": {
                "rects": [
                    {
                        "id": r.get("id"),
                        "x": r.get("x"),
                        "y": r.get("y"),
                        "w": r.get("w"),
                        "h": r.get("h"),
                    }
                    for r in rects
                    if isinstance(r, dict)
                ],
                "local_bbox": bbox,
                "bbox_size_mm": {
                    "w": max(0, bbox_w),
                    "h": max(0, bbox_h),
                },
                "footprint_area_mm2": footprint_area_mm2,
            },
            "notes": cluster.get("notes")
            if isinstance(cluster.get("notes"), list)
            else [],
        }

    return out


def _normalize_constraints(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        normalized: dict[str, Any] = {}
        for key in (
            "type",
            "a",
            "b",
            "id",
            "mode",
            "required",
            "priority",
            "reason",
            "target",
        ):
            if key in item:
                normalized[key] = item.get(key)

        if not normalized:
            for k, v in item.items():
                if isinstance(k, str):
                    normalized[k] = v

        out.append(normalized)

    return out


def _normalize_cluster_rules(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    out: dict[str, Any] = {}
    for key in (
        "grid_mm",
        "allowed_rotations",
        "access_clearance_ratio",
        "min_spacing_mm",
        "max_spacing_mm",
        "preferred_wall_attach",
        "facing",
        "access_requirements",
    ):
        if key in value:
            out[key] = value.get(key)
    return out


def _collect_allowed_cluster_and_object_ids(
    clusters_json: dict[str, Any],
) -> tuple[set[str], dict[str, set[str]], set[str], dict[str, str]]:
    clusters = _extract_clusters_map(clusters_json)

    cluster_ids: set[str] = set()
    object_ids_by_cluster: dict[str, set[str]] = {}
    all_object_ids: set[str] = set()
    object_owner_by_id: dict[str, str] = {}

    for cluster_key, cluster in clusters.items():
        if not isinstance(cluster, dict):
            continue

        cid = cluster.get("cluster_id")
        cid = cid.strip() if isinstance(cid, str) else str(cluster_key).strip()
        if not cid:
            continue

        cluster_ids.add(cid)

        ids = set(_extract_cluster_object_ids(cluster))
        object_ids_by_cluster[cid] = ids

        for oid in ids:
            all_object_ids.add(oid)
            object_owner_by_id.setdefault(oid, cid)

    return cluster_ids, object_ids_by_cluster, all_object_ids, object_owner_by_id


def _extract_clusters_map(clusters_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(clusters_json, dict):
        return {}

    direct: dict[str, dict[str, Any]] = {}
    for k, v in clusters_json.items():
        if (
            isinstance(k, str)
            and isinstance(v, dict)
            and (
                "cluster_id" in v or "cluster_footprint" in v or "local_placements" in v
            )
        ):
            direct[k] = v
    if direct:
        return direct

    wrapped = clusters_json.get("clusters")
    if isinstance(wrapped, dict):
        out: dict[str, dict[str, Any]] = {}
        for k, v in wrapped.items():
            if isinstance(k, str) and isinstance(v, dict):
                out[k] = v
        return out

    if isinstance(wrapped, list):
        out = {}
        for item in wrapped:
            if not isinstance(item, dict):
                continue
            cid = item.get("cluster_id")
            if isinstance(cid, str) and cid:
                out[cid] = item
        return out

    return {}


def _extract_cluster_object_ids(cluster: dict[str, Any]) -> list[str]:
    ids: list[str] = []

    members = cluster.get("members")
    if isinstance(members, list):
        for m in members:
            if isinstance(m, str) and m not in ids:
                ids.append(m)

    local_placements = cluster.get("local_placements")
    if isinstance(local_placements, list):
        for p in local_placements:
            if not isinstance(p, dict):
                continue
            pid = p.get("id")
            if isinstance(pid, str) and pid and pid not in ids:
                ids.append(pid)

    rects = (cluster.get("cluster_footprint") or {}).get("rects") or []
    if isinstance(rects, list):
        for r in rects:
            if not isinstance(r, dict):
                continue
            rid = r.get("id")
            if isinstance(rid, str) and rid and rid not in ids:
                ids.append(rid)

    return ids


def _classify_cluster_role(object_ids: list[str], cluster_id: str) -> str:
    names = {x.lower() for x in object_ids}
    cid = cluster_id.lower()

    if "kitchen_island" in names or "island" in cid:
        return "island"

    if {
        "stove",
        "sink",
        "dishwasher",
        "kitchen_base_cabinet",
        "kitchen_wall_cabinet",
    } & names:
        return "cooking_line"

    if {
        "fridge",
        "pantry_cabinet",
        "wine_cabinet",
        "storage_cabinet",
        "wardrobe",
        "dresser",
        "bookshelf",
    } & names:
        return "storage"

    if {"desk", "office_chair", "chair", "monitor", "laptop"} & names:
        return "work"

    if {"bed", "nightstand", "bench"} & names:
        return "sleep"

    if {"sofa", "armchair", "coffee_table", "tv_console"} & names:
        return "lounge"

    if any(pat in cid for pat in MEDIA_CLUSTER_HINTS):
        return "media"

    if "storage" in cid:
        return "storage"
    if "cooking" in cid:
        return "cooking_line"
    if "work" in cid or "office" in cid:
        return "work"
    if "sleep" in cid or "bed" in cid:
        return "sleep"

    return "mixed"


def _sum_rect_area_mm2(rects: list[dict[str, Any]]) -> int:
    total = 0
    for r in rects:
        if not isinstance(r, dict):
            continue
        try:
            w = int(r.get("w", 0) or 0)
            h = int(r.get("h", 0) or 0)
        except Exception:
            continue
        if w > 0 and h > 0:
            total += w * h
    return total


def _unwrap_room_model(room_model_json: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(room_model_json, dict):
        return {}
    if isinstance(room_model_json.get("room"), dict):
        return room_model_json
    for key in ("parsed", "raw"):
        sub = room_model_json.get(key)
        if isinstance(sub, dict) and isinstance(sub.get("room"), dict):
            return sub
    return room_model_json


def _extract_room_id(room_model_json: dict[str, Any]) -> str:
    src = _unwrap_room_model(room_model_json)
    room = src.get("room")
    if isinstance(room, dict):
        rid = room.get("room_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    return "unknown"


def _normalize_doors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": item.get("id"),
                "segment_mm": item.get("segment_mm")
                if isinstance(item.get("segment_mm"), list)
                else [],
                "swing_radius_mm": item.get("swing_radius_mm"),
                "hinge_hint": item.get("hinge_hint"),
            }
        )
    return out


def _normalize_windows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": item.get("id"),
                "segment_mm": item.get("segment_mm")
                if isinstance(item.get("segment_mm"), list)
                else [],
                "clearance_mm": item.get("clearance_mm"),
            }
        )
    return out


def _normalize_obstacles(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": item.get("id"),
                "type": item.get("type"),
                "hard": bool(item.get("hard", True)),
                "polygon_ccw": item.get("polygon_ccw")
                if isinstance(item.get("polygon_ccw"), list)
                else [],
            }
        )
    return out


def _bbox_from_points(points: list[dict[str, Any]]) -> dict[str, int]:
    xs: list[int] = []
    ys: list[int] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        try:
            x = int(p["x"])
            y = int(p["y"])
        except Exception:
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


def _polygon_area_mm2(points: list[dict[str, Any]]) -> int:
    if not isinstance(points, list) or len(points) < 3:
        return 0
    s = 0
    clean: list[tuple[int, int]] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        try:
            clean.append((int(p["x"]), int(p["y"])))
        except Exception:
            return 0
    if len(clean) < 3:
        return 0
    for i in range(len(clean)):
        x1, y1 = clean[i]
        x2, y2 = clean[(i + 1) % len(clean)]
        s += x1 * y2 - x2 * y1
    return int(abs(s) / 2)


def _json_block(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, indent=2)


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("OpenAI response missing message content")


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("ClusterRelationPlanner returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("ClusterRelationPlanner response must be a JSON object")
    return payload


def _coerce_json_text(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    parts = text.split("```")
    for idx in range(1, len(parts), 2):
        candidate = parts[idx].strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip("\n").strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return text


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _normalize_relation_plan_payload(
    payload: dict[str, Any],
    *,
    room_model_json: dict[str, Any] | None = None,
    clusters_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    (
        allowed_cluster_ids,
        object_ids_by_cluster,
        all_object_ids,
        object_owner_by_id,
    ) = _collect_allowed_cluster_and_object_ids(clusters_json or {})

    def normalize_token(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"_+", "_", value)
        return value.strip("_")

    def clean_str(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    def ensure_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                s = clean_str(item)
                if s is not None:
                    out.append(s)
            return out
        s = clean_str(value)
        return [s] if s is not None else []

    def normalize_status(value: Any) -> str:
        key = normalize_token(str(value or "OK"))
        mapping = {
            "ok": "OK",
            "valid": "OK",
            "success": "OK",
            "need_info": "NEED_INFO",
            "needs_info": "NEED_INFO",
            "missing_info": "NEED_INFO",
            "clarification_needed": "NEED_INFO",
            "ambiguous": "NEED_INFO",
            "unsat": "UNSAT",
            "infeasible": "UNSAT",
            "impossible": "UNSAT",
            "invalid": "UNSAT",
            "conflict": "UNSAT",
        }
        return mapping.get(key, "OK")

    def normalize_priority(value: Any) -> str:
        key = normalize_token(str(value or "medium"))
        mapping = {
            "high": "high",
            "medium": "medium",
            "med": "medium",
            "normal": "medium",
            "default": "medium",
            "low": "low",
        }
        return mapping.get(key, "medium")

    def normalize_list(
        values: Any,
        *,
        allowed: set[str],
        mapping: dict[str, str],
    ) -> list[str]:
        out: list[str] = []
        for item in ensure_str_list(values):
            key = normalize_token(item)
            key = mapping.get(key, key)
            if key in allowed and key not in out:
                out.append(key)
        return out

    prefer_allowed = {
        "wall",
        "center",
        "window_side",
        "entry_side",
        "far_from_entry",
        "recess_or_edge",
        "long_wall",
        "short_wall",
    }
    avoid_allowed = {
        "door_swing",
        "window_clearance",
        "entry_blocking",
        "center",
        "bottleneck",
        "window_blocking",
        "main_path",
    }
    cluster_orientation_allowed = {
        "face_center",
        "face_window",
        "face_entry",
        "face_cluster",
        "back_to_wall",
        "access_to_open_space",
        "axis_parallel_wall",
        "axis_perpendicular_wall",
        "axis_parallel_window",
        "axis_perpendicular_window",
        "inward_to_room",
        "outward_to_wall",
    }
    object_orientation_allowed = {
        "front_to_open_space",
        "front_to_cluster_center",
        "front_to_room_center",
        "front_to_window",
        "front_to_entry",
        "back_to_wall",
        "side_to_wall",
        "long_axis_parallel_wall",
        "long_axis_perpendicular_wall",
        "align_with_cluster_axis",
        "face_object",
        "face_away_from_object",
        "preserve_front_access",
    }
    relation_allowed = {
        "near",
        "separate",
        "adjacent_if_possible",
        "far_if_possible",
    }
    directional_relation_allowed = {
        "face_each_other",
        "avoid_facing_each_other",
        "same_axis",
        "parallel_alignment",
        "perpendicular_alignment",
        "access_faces_other",
        "turn_toward",
        "turn_away",
    }
    keep_open_allowed = {
        "entry_buffer",
        "window_buffer",
        "center_lane",
        "work_lane",
    }

    prefer_map = {
        "semi_center": "center",
        "semi-center": "center",
        "near_center": "center",
        "mid_center": "center",
        "edge": "recess_or_edge",
        "recess": "recess_or_edge",
        "edge_or_recess": "recess_or_edge",
        "longwall": "long_wall",
        "shortwall": "short_wall",
        "window": "window_side",
        "entry": "entry_side",
    }
    avoid_map = {
        "door": "door_swing",
        "entry": "entry_blocking",
        "window": "window_clearance",
    }
    cluster_orientation_map = {
        "face_room_center": "face_center",
        "face_center_of_room": "face_center",
        "face_window_side": "face_window",
        "face_entry_side": "face_entry",
        "face_door": "face_entry",
        "toward_cluster": "face_cluster",
        "front_to_open_space": "access_to_open_space",
        "open_to_space": "access_to_open_space",
        "parallel_to_wall": "axis_parallel_wall",
        "perpendicular_to_wall": "axis_perpendicular_wall",
        "parallel_to_window": "axis_parallel_window",
        "perpendicular_to_window": "axis_perpendicular_window",
        "toward_room": "inward_to_room",
        "toward_wall": "outward_to_wall",
    }
    object_orientation_map = {
        "face_open_space": "front_to_open_space",
        "front_to_space": "front_to_open_space",
        "face_cluster_center": "front_to_cluster_center",
        "face_room_center": "front_to_room_center",
        "face_window": "front_to_window",
        "face_entry": "front_to_entry",
        "parallel_to_wall": "long_axis_parallel_wall",
        "perpendicular_to_wall": "long_axis_perpendicular_wall",
        "aligned_with_cluster_axis": "align_with_cluster_axis",
        "access_front_clear": "preserve_front_access",
    }
    relation_map = {
        "close": "near",
        "nearby": "near",
        "adjacent": "adjacent_if_possible",
        "together": "near",
        "apart": "separate",
        "far": "far_if_possible",
    }
    directional_relation_map = {
        "facing_each_other": "face_each_other",
        "face_one_another": "face_each_other",
        "avoid_facing": "avoid_facing_each_other",
        "parallel": "parallel_alignment",
        "perpendicular": "perpendicular_alignment",
        "toward": "turn_toward",
        "away": "turn_away",
        "same_alignment": "same_axis",
    }
    keep_open_map = {
        "entry": "entry_buffer",
        "window": "window_buffer",
        "center": "center_lane",
        "work": "work_lane",
        "walk_lane": "center_lane",
        "circulation_lane": "center_lane",
    }

    def valid_cluster_id(value: Any) -> str | None:
        cid = clean_str(value)
        if cid is None:
            return None
        if allowed_cluster_ids and cid not in allowed_cluster_ids:
            return None
        return cid

    def valid_object_id(cluster_id: str, value: Any) -> str | None:
        oid = clean_str(value)
        if oid is None:
            return None
        allowed = object_ids_by_cluster.get(cluster_id, set())
        if allowed and oid not in allowed:
            return None
        return oid

    def valid_target_object_id(value: Any) -> str | None:
        oid = clean_str(value)
        if oid is None:
            return None
        if all_object_ids and oid not in all_object_ids:
            return None
        return oid

    def normalize_affinities(values: Any) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []

        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        for entry in values:
            if not isinstance(entry, dict):
                continue

            cid = valid_cluster_id(entry.get("cluster_id"))
            if cid is None or cid in seen:
                continue

            prefer = normalize_list(
                entry.get("prefer"),
                allowed=prefer_allowed,
                mapping=prefer_map,
            )
            avoid = normalize_list(
                entry.get("avoid"),
                allowed=avoid_allowed,
                mapping=avoid_map,
            )

            out.append(
                {
                    "cluster_id": cid,
                    "prefer": prefer,
                    "avoid": avoid,
                    "priority": normalize_priority(entry.get("priority")),
                    "reason": clean_str(entry.get("reason")) or "macro affinity",
                }
            )
            seen.add(cid)

        return out

    def normalize_cluster_orientations(values: Any) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []

        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        for entry in values:
            if not isinstance(entry, dict):
                continue

            cid = valid_cluster_id(entry.get("cluster_id"))
            if cid is None or cid in seen:
                continue

            intents = normalize_list(
                entry.get("intents"),
                allowed=cluster_orientation_allowed,
                mapping=cluster_orientation_map,
            )

            target_cluster_id = valid_cluster_id(entry.get("target_cluster_id"))

            if "face_cluster" in intents and target_cluster_id is None:
                intents = [x for x in intents if x != "face_cluster"]

            if "face_cluster" not in intents:
                target_cluster_id = None

            if not intents:
                continue

            out.append(
                {
                    "cluster_id": cid,
                    "intents": intents,
                    "target_cluster_id": target_cluster_id,
                    "priority": normalize_priority(entry.get("priority")),
                    "reason": clean_str(entry.get("reason")) or "cluster orientation",
                }
            )
            seen.add(cid)

        return out

    def normalize_object_orientations(values: Any) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []

        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for entry in values:
            if not isinstance(entry, dict):
                continue

            cid = valid_cluster_id(entry.get("cluster_id"))
            if cid is None:
                continue

            oid = valid_object_id(cid, entry.get("object_id"))
            if oid is None:
                continue

            key = (cid, oid)
            if key in seen:
                continue

            intents = normalize_list(
                entry.get("intents"),
                allowed=object_orientation_allowed,
                mapping=object_orientation_map,
            )

            target_object_id = valid_target_object_id(entry.get("target_object_id"))

            needs_target = any(
                x in intents for x in ("face_object", "face_away_from_object")
            )
            if needs_target and target_object_id is None:
                intents = [
                    x
                    for x in intents
                    if x not in {"face_object", "face_away_from_object"}
                ]

            if not any(x in intents for x in ("face_object", "face_away_from_object")):
                target_object_id = None

            if not intents:
                continue

            out.append(
                {
                    "cluster_id": cid,
                    "object_id": oid,
                    "intents": intents,
                    "target_object_id": target_object_id,
                    "priority": normalize_priority(entry.get("priority")),
                    "reason": clean_str(entry.get("reason")) or "object orientation",
                }
            )
            seen.add(key)

        return out

    def normalize_cluster_relations(values: Any) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []

        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for entry in values:
            if not isinstance(entry, dict):
                continue

            a = valid_cluster_id(entry.get("a"))
            b = valid_cluster_id(entry.get("b"))
            if a is None or b is None or a == b:
                continue

            relation_raw = clean_str(entry.get("relation"))
            if relation_raw is None:
                continue
            relation = relation_map.get(
                normalize_token(relation_raw), normalize_token(relation_raw)
            )
            if relation not in relation_allowed:
                continue

            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue

            out.append(
                {
                    "a": a,
                    "b": b,
                    "relation": relation,
                    "priority": normalize_priority(entry.get("priority")),
                    "reason": clean_str(entry.get("reason")) or "cluster relation",
                }
            )
            seen.add(pair)

        return out

    def normalize_cluster_directional_relations(values: Any) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []

        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for entry in values:
            if not isinstance(entry, dict):
                continue

            a = valid_cluster_id(entry.get("a"))
            b = valid_cluster_id(entry.get("b"))
            if a is None or b is None or a == b:
                continue

            relation_raw = clean_str(entry.get("relation"))
            if relation_raw is None:
                continue
            relation = directional_relation_map.get(
                normalize_token(relation_raw), normalize_token(relation_raw)
            )
            if relation not in directional_relation_allowed:
                continue

            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue

            out.append(
                {
                    "a": a,
                    "b": b,
                    "relation": relation,
                    "priority": normalize_priority(entry.get("priority")),
                    "reason": clean_str(entry.get("reason")) or "directional relation",
                }
            )
            seen.add(pair)

        return out

    def normalize_circulation_plan(value: Any) -> dict[str, Any]:
        value = value if isinstance(value, dict) else {}

        main_paths_raw = value.get("main_paths")
        keep_open_raw = value.get("keep_open_regions")

        main_paths: list[dict[str, Any]] = []
        seen_paths: set[tuple[str, str]] = set()

        if isinstance(main_paths_raw, list):
            for entry in main_paths_raw:
                if not isinstance(entry, dict):
                    continue

                from_value = (
                    clean_str(entry.get("from"))
                    or clean_str(entry.get("from_"))
                    or clean_str(entry.get("source"))
                )
                to_cluster = valid_cluster_id(
                    entry.get("to_cluster")
                    or entry.get("to")
                    or entry.get("target_cluster")
                )

                if from_value is None or to_cluster is None or from_value == to_cluster:
                    continue

                key = (from_value, to_cluster)
                if key in seen_paths:
                    continue

                main_paths.append(
                    {
                        "from": from_value,
                        "to_cluster": to_cluster,
                        "priority": normalize_priority(entry.get("priority")),
                        "reason": clean_str(entry.get("reason")) or "main path",
                    }
                )
                seen_paths.add(key)

        keep_open_regions: list[dict[str, Any]] = []
        seen_regions: set[tuple[str, str]] = set()

        if isinstance(keep_open_raw, list):
            for entry in keep_open_raw:
                if not isinstance(entry, dict):
                    continue

                region_type_raw = clean_str(entry.get("type"))
                near_value = clean_str(entry.get("near"))

                if region_type_raw is None or near_value is None:
                    continue

                region_type = keep_open_map.get(
                    normalize_token(region_type_raw), normalize_token(region_type_raw)
                )
                if region_type not in keep_open_allowed:
                    continue

                key = (region_type, near_value)
                if key in seen_regions:
                    continue

                keep_open_regions.append(
                    {
                        "type": region_type,
                        "near": near_value,
                        "priority": normalize_priority(entry.get("priority")),
                        "reason": clean_str(entry.get("reason")) or "open region",
                    }
                )
                seen_regions.add(key)

        return {
            "main_paths": main_paths,
            "keep_open_regions": keep_open_regions,
        }

    out: dict[str, Any] = dict(payload)

    out["status"] = normalize_status(out.get("status"))
    out["room_id"] = clean_str(out.get("room_id")) or _extract_room_id(
        room_model_json or {}
    )

    out["cluster_affinities"] = normalize_affinities(
        out.get("cluster_affinities") or out.get("affinities")
    )
    out["cluster_orientations"] = normalize_cluster_orientations(
        out.get("cluster_orientations") or out.get("cluster_orientation")
    )
    out["object_orientations"] = normalize_object_orientations(
        out.get("object_orientations") or out.get("object_orientation")
    )
    out["cluster_relations"] = normalize_cluster_relations(
        out.get("cluster_relations") or out.get("relations")
    )
    out["cluster_directional_relations"] = normalize_cluster_directional_relations(
        out.get("cluster_directional_relations")
        or out.get("directional_relations")
        or out.get("cluster_direction_relations")
    )
    out["circulation_plan"] = normalize_circulation_plan(out.get("circulation_plan"))

    out["placement_guidelines"] = _uniq_keep_order(
        ensure_str_list(out.get("placement_guidelines"))
    )
    out["notes"] = _uniq_keep_order(ensure_str_list(out.get("notes")))
    out["missing"] = _uniq_keep_order(ensure_str_list(out.get("missing")))

    _enforce_viewing_orientation_requirements(
        out,
        clusters_json=clusters_json or {},
    )
    _enforce_access_orientation_requirements(
        out,
        clusters_json=clusters_json or {},
    )
    _apply_generic_macro_relation_policy(
        out,
        room_model_json=room_model_json or {},
        clusters_json=clusters_json or {},
    )

    return out


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _priority_rank(value: Any) -> int:
    v = str(value or "").strip().lower()
    if v == "high":
        return 2
    if v == "medium":
        return 1
    if v == "low":
        return 0
    return 1


def _max_priority(a: Any, b: Any) -> str:
    return a if _priority_rank(a) >= _priority_rank(b) else b


def _uniq_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if not isinstance(v, str):
            continue
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _matches_any_pattern(object_id: str, patterns: list[str]) -> bool:
    key = object_id.lower()
    return any(p in key for p in patterns)


def _pick_object_ids_by_priority(
    object_ids: list[str], patterns: list[str]
) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for oid in object_ids:
        key = oid.lower()
        best_idx = None
        for idx, pat in enumerate(patterns):
            if pat in key:
                best_idx = idx
                break
        if best_idx is not None:
            ranked.append((best_idx, oid))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [oid for _, oid in ranked]


def _pick_first_object_id_by_priority(
    object_ids: list[str], patterns: list[str]
) -> str | None:
    ranked = _pick_object_ids_by_priority(object_ids, patterns)
    return ranked[0] if ranked else None


def _collect_access_sensitive_object_ids(cluster: dict[str, Any]) -> list[str]:
    object_ids = _extract_cluster_object_ids(cluster)
    result: list[str] = []

    hard_constraints = cluster.get("hard_constraints")
    if isinstance(hard_constraints, list):
        for c in hard_constraints:
            if not isinstance(c, dict):
                continue
            if (
                c.get("type") == "requires_access"
                and c.get("mode") == "front_clearance"
            ):
                oid = c.get("id")
                if isinstance(oid, str) and oid in object_ids and oid not in result:
                    result.append(oid)

    rules = cluster.get("cluster_rules")
    if isinstance(rules, dict):
        access_reqs = rules.get("access_requirements")
        if isinstance(access_reqs, list):
            for item in access_reqs:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "front_clearance":
                    continue
                if item.get("required") is not True:
                    continue
                oid = item.get("id")
                if isinstance(oid, str) and oid in object_ids and oid not in result:
                    result.append(oid)

    for oid in object_ids:
        if oid not in result and _matches_any_pattern(oid, ACCESS_SENSITIVE_PATTERNS):
            result.append(oid)

    return result


def _ensure_cluster_relation(
    plan: dict[str, Any],
    *,
    a: str,
    b: str,
    relation: str,
    priority: str,
    reason: str,
) -> None:
    arr = plan.setdefault("cluster_relations", [])
    if not isinstance(arr, list):
        arr = []
        plan["cluster_relations"] = arr

    pair = tuple(sorted((a, b)))
    for entry in arr:
        if not isinstance(entry, dict):
            continue
        ea = entry.get("a")
        eb = entry.get("b")
        if not isinstance(ea, str) or not isinstance(eb, str):
            continue
        if tuple(sorted((ea, eb))) != pair:
            continue

        existing_rel = entry.get("relation")
        if relation == "near" and existing_rel in {"separate", "far_if_possible"}:
            entry["relation"] = "near"
        elif relation == "adjacent_if_possible" and existing_rel == "far_if_possible":
            entry["relation"] = "adjacent_if_possible"

        entry["priority"] = _max_priority(entry.get("priority"), priority)
        if not _clean_text(entry.get("reason")):
            entry["reason"] = reason
        return

    arr.append(
        {
            "a": a,
            "b": b,
            "relation": relation,
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_cluster_directional_relation(
    plan: dict[str, Any],
    *,
    a: str,
    b: str,
    relation: str,
    priority: str,
    reason: str,
) -> None:
    arr = plan.setdefault("cluster_directional_relations", [])
    if not isinstance(arr, list):
        arr = []
        plan["cluster_directional_relations"] = arr

    pair = tuple(sorted((a, b)))
    for entry in arr:
        if not isinstance(entry, dict):
            continue
        ea = entry.get("a")
        eb = entry.get("b")
        if not isinstance(ea, str) or not isinstance(eb, str):
            continue
        if tuple(sorted((ea, eb))) != pair:
            continue

        entry["priority"] = _max_priority(entry.get("priority"), priority)
        if not _clean_text(entry.get("reason")):
            entry["reason"] = reason
        if entry.get("relation") not in {
            "face_each_other",
            "turn_toward",
            "access_faces_other",
        }:
            entry["relation"] = relation
        return

    arr.append(
        {
            "a": a,
            "b": b,
            "relation": relation,
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_cluster_orientation(
    plan: dict[str, Any],
    *,
    cluster_id: str,
    intents: list[str],
    target_cluster_id: str | None,
    priority: str,
    reason: str,
) -> None:
    arr = plan.setdefault("cluster_orientations", [])
    if not isinstance(arr, list):
        arr = []
        plan["cluster_orientations"] = arr

    for entry in arr:
        if not isinstance(entry, dict):
            continue
        if entry.get("cluster_id") != cluster_id:
            continue

        old_intents = entry.get("intents")
        if not isinstance(old_intents, list):
            old_intents = []
        entry["intents"] = _uniq_keep_order(
            [x for x in old_intents if isinstance(x, str)] + intents
        )

        entry["priority"] = _max_priority(entry.get("priority"), priority)

        if "face_cluster" in entry["intents"]:
            if (
                _clean_text(entry.get("target_cluster_id")) is None
                and target_cluster_id is not None
            ):
                entry["target_cluster_id"] = target_cluster_id
        else:
            entry["target_cluster_id"] = None

        if not _clean_text(entry.get("reason")):
            entry["reason"] = reason
        return

    arr.append(
        {
            "cluster_id": cluster_id,
            "intents": _uniq_keep_order(intents),
            "target_cluster_id": target_cluster_id
            if "face_cluster" in intents
            else None,
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_object_orientation(
    plan: dict[str, Any],
    *,
    cluster_id: str,
    object_id: str,
    intents: list[str],
    target_object_id: str | None,
    priority: str,
    reason: str,
) -> None:
    arr = plan.setdefault("object_orientations", [])
    if not isinstance(arr, list):
        arr = []
        plan["object_orientations"] = arr

    for entry in arr:
        if not isinstance(entry, dict):
            continue
        if entry.get("cluster_id") != cluster_id or entry.get("object_id") != object_id:
            continue

        old_intents = entry.get("intents")
        if not isinstance(old_intents, list):
            old_intents = []
        entry["intents"] = _uniq_keep_order(
            [x for x in old_intents if isinstance(x, str)] + intents
        )

        entry["priority"] = _max_priority(entry.get("priority"), priority)

        current_target = _clean_text(entry.get("target_object_id"))
        if current_target is None and target_object_id is not None:
            entry["target_object_id"] = target_object_id

        if not _clean_text(entry.get("reason")):
            entry["reason"] = reason
        return

    arr.append(
        {
            "cluster_id": cluster_id,
            "object_id": object_id,
            "intents": _uniq_keep_order(intents),
            "target_object_id": target_object_id,
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_guideline(plan: dict[str, Any], text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        return
    arr = plan.setdefault("placement_guidelines", [])
    if not isinstance(arr, list):
        arr = []
        plan["placement_guidelines"] = arr
    if text not in arr:
        arr.append(text)


def _infer_viewing_cluster_pairs(
    plan: dict[str, Any],
    clusters_json: dict[str, Any],
) -> list[tuple[str, str]]:
    clusters = _extract_clusters_map(clusters_json)
    if not clusters:
        return []

    seating_clusters: dict[str, list[str]] = {}
    focal_clusters: dict[str, list[str]] = {}

    for cid, cluster in clusters.items():
        object_ids = _extract_cluster_object_ids(cluster)
        seating_ids = _pick_object_ids_by_priority(object_ids, SEATING_ANCHOR_PATTERNS)
        focal_ids = _pick_object_ids_by_priority(object_ids, FOCAL_ANCHOR_PATTERNS)

        if seating_ids:
            seating_clusters[cid] = seating_ids
        if focal_ids:
            focal_clusters[cid] = focal_ids

    if not seating_clusters or not focal_clusters:
        return []

    directional_pairs: set[tuple[str, str]] = set()
    for entry in plan.get("cluster_directional_relations", []):
        if not isinstance(entry, dict):
            continue
        a = entry.get("a")
        b = entry.get("b")
        rel = entry.get("relation")
        if not isinstance(a, str) or not isinstance(b, str) or not isinstance(rel, str):
            continue
        if rel in {"face_each_other", "turn_toward", "access_faces_other"}:
            directional_pairs.add(tuple(sorted((a, b))))

    relation_pairs: set[tuple[str, str]] = set()
    for entry in plan.get("cluster_relations", []):
        if not isinstance(entry, dict):
            continue
        a = entry.get("a")
        b = entry.get("b")
        rel = entry.get("relation")
        if not isinstance(a, str) or not isinstance(b, str) or not isinstance(rel, str):
            continue
        if rel in {"near", "adjacent_if_possible"}:
            relation_pairs.add(tuple(sorted((a, b))))

    orientation_pairs: set[tuple[str, str]] = set()
    for entry in plan.get("cluster_orientations", []):
        if not isinstance(entry, dict):
            continue
        cid = entry.get("cluster_id")
        intents = entry.get("intents")
        target = entry.get("target_cluster_id")
        if not isinstance(cid, str) or not isinstance(target, str):
            continue
        if isinstance(intents, list) and "face_cluster" in intents:
            orientation_pairs.add((cid, target))

    candidates: list[tuple[int, str, str]] = []
    for seating_cid in seating_clusters.keys():
        for focal_cid in focal_clusters.keys():
            if seating_cid == focal_cid:
                continue

            score = 0
            pair = tuple(sorted((seating_cid, focal_cid)))

            if pair in directional_pairs:
                score += 5
            if pair in relation_pairs:
                score += 3
            if (seating_cid, focal_cid) in orientation_pairs or (
                focal_cid,
                seating_cid,
            ) in orientation_pairs:
                score += 4
            if len(seating_clusters) == 1 and len(focal_clusters) == 1:
                score += 2
            if "seating" in seating_cid.lower() or "lounge" in seating_cid.lower():
                score += 1
            if any(p in focal_cid.lower() for p in MEDIA_CLUSTER_HINTS):
                score += 1

            if score > 0:
                candidates.append((score, seating_cid, focal_cid))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    chosen: list[tuple[str, str]] = []
    used_seating: set[str] = set()
    for _, seating_cid, focal_cid in candidates:
        if seating_cid in used_seating:
            continue
        chosen.append((seating_cid, focal_cid))
        used_seating.add(seating_cid)

    return chosen


def _enforce_viewing_orientation_requirements(
    plan: dict[str, Any],
    *,
    clusters_json: dict[str, Any],
) -> None:
    if not isinstance(plan, dict):
        return

    clusters = _extract_clusters_map(clusters_json)
    if not clusters:
        return

    viewing_pairs = _infer_viewing_cluster_pairs(plan, clusters_json)
    if not viewing_pairs:
        return

    for seating_cid, focal_cid in viewing_pairs:
        seating_cluster = clusters.get(seating_cid, {})
        focal_cluster = clusters.get(focal_cid, {})

        seating_ids = _extract_cluster_object_ids(seating_cluster)
        focal_ids = _extract_cluster_object_ids(focal_cluster)

        seating_anchors = _pick_object_ids_by_priority(
            seating_ids,
            SEATING_ANCHOR_PATTERNS,
        )
        focal_anchor = _pick_first_object_id_by_priority(
            focal_ids,
            FOCAL_ANCHOR_PATTERNS,
        )

        if not seating_anchors or focal_anchor is None:
            continue

        primary_seat = seating_anchors[0]

        _ensure_cluster_relation(
            plan,
            a=seating_cid,
            b=focal_cid,
            relation="near",
            priority="high",
            reason="viewing relationship benefits from practical proximity",
        )

        _ensure_cluster_directional_relation(
            plan,
            a=seating_cid,
            b=focal_cid,
            relation="face_each_other",
            priority="high",
            reason="viewing relationship between seating and focal/media clusters",
        )

        _ensure_cluster_orientation(
            plan,
            cluster_id=seating_cid,
            intents=["inward_to_room", "face_cluster"],
            target_cluster_id=focal_cid,
            priority="high",
            reason="primary seating cluster should orient toward focal/media cluster",
        )

        _ensure_cluster_orientation(
            plan,
            cluster_id=focal_cid,
            intents=["back_to_wall", "access_to_open_space", "face_cluster"],
            target_cluster_id=seating_cid,
            priority="high",
            reason="focal/media cluster should orient toward seating with open viewing side",
        )

        for oid in seating_anchors[:3]:
            _ensure_object_orientation(
                plan,
                cluster_id=seating_cid,
                object_id=oid,
                intents=["face_object"],
                target_object_id=focal_anchor,
                priority="high",
                reason="important seating anchor should face focal/media anchor",
            )

        _ensure_object_orientation(
            plan,
            cluster_id=focal_cid,
            object_id=focal_anchor,
            intents=["front_to_open_space", "face_object"],
            target_object_id=primary_seat,
            priority="high",
            reason="focal/media anchor should present its usable front toward seating",
        )

        _ensure_guideline(
            plan,
            f"Preserve a clear viewing relationship between {primary_seat} and {focal_anchor}.",
        )
        _ensure_guideline(
            plan,
            f"Do not orient {primary_seat} away from {focal_anchor}.",
        )
        _ensure_guideline(
            plan,
            f"Keep {focal_cid} readable from {seating_cid}.",
        )


def _enforce_access_orientation_requirements(
    plan: dict[str, Any],
    *,
    clusters_json: dict[str, Any],
) -> None:
    if not isinstance(plan, dict):
        return

    clusters = _extract_clusters_map(clusters_json)
    if not clusters:
        return

    for cid, cluster in clusters.items():
        access_ids = _collect_access_sensitive_object_ids(cluster)
        if not access_ids:
            continue

        cluster_role = _classify_cluster_role(
            _extract_cluster_object_ids(cluster),
            str(cid),
        )

        cluster_intents: list[str] = []
        if cluster_role in {"storage", "cooking_line", "work", "media"}:
            cluster_intents = ["access_to_open_space"]
            if cluster_role in {"storage", "cooking_line", "media"}:
                cluster_intents.append("back_to_wall")

        if cluster_intents:
            _ensure_cluster_orientation(
                plan,
                cluster_id=cid,
                intents=cluster_intents,
                target_cluster_id=None,
                priority="medium",
                reason="access-sensitive cluster should expose usable front toward open space",
            )

        for oid in access_ids[:4]:
            intents = ["preserve_front_access"]
            if _matches_any_pattern(oid, ACCESS_SENSITIVE_PATTERNS):
                intents.append("front_to_open_space")

            _ensure_object_orientation(
                plan,
                cluster_id=cid,
                object_id=oid,
                intents=intents,
                target_object_id=None,
                priority="high",
                reason="object has meaningful front usability and should preserve access",
            )

            _ensure_guideline(
                plan,
                f"Preserve front usability for {oid}.",
            )


def _apply_generic_macro_relation_policy(
    plan: dict[str, Any],
    *,
    room_model_json: dict[str, Any],
    clusters_json: dict[str, Any],
) -> None:
    if not isinstance(plan, dict):
        return

    clusters = _extract_clusters_map(clusters_json)
    if not clusters:
        return

    room_type = _extract_room_type(room_model_json)
    roles_by_cluster: dict[str, str] = {}
    footprint_area_by_cluster: dict[str, int] = {}
    for cid, cluster in clusters.items():
        object_ids = _extract_cluster_object_ids(cluster)
        roles_by_cluster[cid] = _classify_cluster_role(object_ids, cid)
        footprint = cluster.get("cluster_footprint")
        rects = (footprint or {}).get("rects") if isinstance(footprint, dict) else None
        footprint_area_by_cluster[cid] = _sum_rect_area_mm2(rects or [])

    primary_cluster_id = _pick_primary_cluster_id(
        room_type=room_type,
        roles_by_cluster=roles_by_cluster,
        footprint_area_by_cluster=footprint_area_by_cluster,
    )

    viewing_pairs = {
        tuple(sorted(pair))
        for pair in _infer_viewing_cluster_pairs(plan, clusters_json)
    }

    _prune_directional_relations_without_focal_support(
        plan,
        viewing_pairs=viewing_pairs,
        roles_by_cluster=roles_by_cluster,
    )
    _cap_center_preference_to_primary_cluster(
        plan,
        primary_cluster_id=primary_cluster_id,
        roles_by_cluster=roles_by_cluster,
    )


def _extract_room_type(room_model_json: dict[str, Any]) -> str:
    src = _unwrap_room_model(room_model_json)
    meta = src.get("meta")
    if not isinstance(meta, dict):
        return ""
    room_type = meta.get("room_type")
    if not isinstance(room_type, str):
        return ""
    return room_type.strip().lower()


def _pick_primary_cluster_id(
    *,
    room_type: str,
    roles_by_cluster: dict[str, str],
    footprint_area_by_cluster: dict[str, int],
) -> str | None:
    room_bias: dict[str, dict[str, int]] = {
        "bedroom": {
            "sleep": 20,
            "storage": 8,
            "work": 6,
            "lounge": 4,
            "media": 2,
        },
        "living_room": {
            "lounge": 20,
            "media": 18,
            "work": 6,
            "storage": 4,
            "sleep": 0,
        },
        "office": {
            "work": 20,
            "storage": 8,
            "lounge": 4,
        },
    }
    generic_role_priority = {
        "sleep": 12,
        "lounge": 11,
        "work": 10,
        "cooking_line": 10,
        "island": 9,
        "media": 8,
        "storage": 6,
        "mixed": 5,
    }

    best_cluster_id: str | None = None
    best_score = -(10**9)
    room_role_bonus = room_bias.get(room_type, {})
    for cid, role in roles_by_cluster.items():
        score = generic_role_priority.get(role, 4) * 1000
        score += room_role_bonus.get(role, 0) * 1000
        score += int(footprint_area_by_cluster.get(cid, 0) // 10000)
        if score > best_score or (
            score == best_score and (best_cluster_id is None or cid < best_cluster_id)
        ):
            best_score = score
            best_cluster_id = cid
    return best_cluster_id


def _prune_directional_relations_without_focal_support(
    plan: dict[str, Any],
    *,
    viewing_pairs: set[tuple[str, str]],
    roles_by_cluster: dict[str, str],
) -> None:
    values = plan.get("cluster_directional_relations")
    if not isinstance(values, list):
        return

    kept: list[dict[str, Any]] = []
    dropped_pairs: list[str] = []

    for entry in values:
        if not isinstance(entry, dict):
            continue
        a = entry.get("a")
        b = entry.get("b")
        relation = entry.get("relation")
        if (
            not isinstance(a, str)
            or not isinstance(b, str)
            or not isinstance(relation, str)
        ):
            continue

        relation_key = relation.strip().lower()
        pair = tuple(sorted((a, b)))
        a_role = roles_by_cluster.get(a, "mixed")
        b_role = roles_by_cluster.get(b, "mixed")
        pair_has_media = "media" in {a_role, b_role}
        keep = True

        if relation_key in {"face_each_other", "turn_toward", "access_faces_other"}:
            keep = pair in viewing_pairs or pair_has_media
        elif relation_key in {"parallel_alignment", "perpendicular_alignment"}:
            keep = pair_has_media or pair in viewing_pairs

        if keep:
            kept.append(entry)
            continue

        dropped_pairs.append(f"{a}:{b}:{relation_key}")

    plan["cluster_directional_relations"] = kept
    if dropped_pairs:
        notes = plan.setdefault("notes", [])
        if isinstance(notes, list):
            notes.append(
                "Dropped over-specific directional relations that were not supported "
                "by a focal/viewing macro relationship."
            )


def _cap_center_preference_to_primary_cluster(
    plan: dict[str, Any],
    *,
    primary_cluster_id: str | None,
    roles_by_cluster: dict[str, str],
) -> None:
    if primary_cluster_id is None:
        return

    affinities = plan.get("cluster_affinities")
    if isinstance(affinities, list):
        for entry in affinities:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("cluster_id")
            prefer = entry.get("prefer")
            if not isinstance(cid, str) or not isinstance(prefer, list):
                continue
            if cid == primary_cluster_id:
                continue
            if roles_by_cluster.get(cid) in {"island"}:
                continue
            entry["prefer"] = [item for item in prefer if item != "center"]

    orientations = plan.get("cluster_orientations")
    if isinstance(orientations, list):
        for entry in orientations:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("cluster_id")
            intents = entry.get("intents")
            if not isinstance(cid, str) or not isinstance(intents, list):
                continue
            if cid == primary_cluster_id:
                continue
            if roles_by_cluster.get(cid) in {"island"}:
                continue
            entry["intents"] = [item for item in intents if item != "face_center"]
