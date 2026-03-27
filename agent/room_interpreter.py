from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Literal

from agent_schema.roomI_schema import RoomInterpreterOutput
from clients.base_client import ChatMessage
from clients.llm_client import get_llm_client
from prompt.room_interpreter import ROOM_INTERPRETER_PROMPT
from prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class RoomInterpreter:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = ROOM_INTERPRETER_PROMPT

    def build_messages(self, input_payload: Mapping[str, object]) -> list[ChatMessage]:
        payload = json.dumps(input_payload, ensure_ascii=True)
        user_prompt = f"{self.prompt_template}\n\nINPUT_JSON:\n{payload}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_raw(
        self,
        input_payload: Mapping[str, object],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        messages = self.build_messages(input_payload)
        response = get_llm_client().chat_completion(
            messages,
            model_key="primary",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_content(response)

    def generate(
        self,
        input_payload: Mapping[str, object],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> RoomInterpreterOutput:
        raw = self.generate_raw(
            input_payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        payload = _parse_json(raw)
        payload = _ensure_room_interpreter_payload(payload, input_payload)
        payload = _apply_default_openings(payload)
        payload = _normalize_room_interpreter_numbers(payload)
        return RoomInterpreterOutput.model_validate(payload)


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("OpenAI response missing message content")


def _parse_json(raw: str) -> dict[str, object]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("RoomInterpreter returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("RoomInterpreter response must be a JSON object")
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


def _ensure_room_interpreter_payload(
    payload: dict[str, object],
    input_payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, dict):
        payload = {}

    room = payload.get("room")
    room = room if isinstance(room, dict) else {}

    openings = payload.get("openings")
    openings = openings if isinstance(openings, dict) else {}

    meta = payload.get("meta")
    meta = meta if isinstance(meta, dict) else {}

    room_polygon = room.get("polygon_ccw")
    if not isinstance(room_polygon, list) or len(room_polygon) < 3:
        room_polygon = _resolve_room_polygon(input_payload) or []
    room_polygon = _ensure_ccw(room_polygon)

    room_id = (
        room.get("room_id")
        if isinstance(room.get("room_id"), str)
        else _resolve_room_id(input_payload)
    )
    name = (
        room.get("name")
        if isinstance(room.get("name"), str)
        else _resolve_room_name(input_payload)
    )
    room = {
        "unit": "mm",
        "room_id": room_id or "room_1",
        "name": name or "Main",
        "polygon_ccw": room_polygon,
    }

    doors = openings.get("doors")
    windows = openings.get("windows")
    if not isinstance(doors, list) or not isinstance(windows, list):
        fp_openings = _resolve_floorplan_openings(input_payload)
        if not isinstance(doors, list):
            doors = fp_openings.get("doors", [])
        if not isinstance(windows, list):
            windows = fp_openings.get("windows", [])

    openings = {
        "doors": doors if isinstance(doors, list) else [],
        "windows": windows if isinstance(windows, list) else [],
    }

    meta = {
        "room_type": meta.get("room_type")
        if isinstance(meta.get("room_type"), str)
        else _resolve_room_type(input_payload),
        "style": meta.get("style")
        if isinstance(meta.get("style"), str)
        else _resolve_style(input_payload),
        "height_mm": meta.get("height_mm")
        if isinstance(meta.get("height_mm"), int)
        else _resolve_height_mm(input_payload),
        "window_direction": (
            meta.get("window_direction")
            if isinstance(meta.get("window_direction"), str)
            else _resolve_window_direction(input_payload)
        ),
        "max_iterations": meta.get("max_iterations")
        if isinstance(meta.get("max_iterations"), int)
        else _resolve_max_iterations(input_payload),
    }

    payload["room"] = room
    payload["openings"] = openings
    payload["meta"] = meta
    if "status" not in payload:
        payload["status"] = "OK" if len(room_polygon) >= 3 else "NEED_INFO"
    if "obstacles" not in payload or not isinstance(payload.get("obstacles"), list):
        payload["obstacles"] = []
    if "notes" not in payload or not isinstance(payload.get("notes"), list):
        payload["notes"] = []
    if "missing" not in payload or not isinstance(payload.get("missing"), list):
        payload["missing"] = []
    if "conflicts" not in payload or not isinstance(payload.get("conflicts"), list):
        payload["conflicts"] = []

    return payload


def _normalize_room_interpreter_numbers(
    payload: dict[str, object],
) -> dict[str, object]:
    normalized = _normalize_nested_room_value(payload)
    if not isinstance(normalized, dict):
        return payload

    room = normalized.get("room")
    if isinstance(room, dict):
        polygon = room.get("polygon_ccw")
        if isinstance(polygon, list):
            room["polygon_ccw"] = _ensure_ccw(_normalize_polygon_points(polygon))
    return normalized


def _normalize_nested_room_value(value: object) -> object:
    if isinstance(value, dict):
        normalized_dict: dict[str, object] = {}
        for key, nested_value in value.items():
            normalized_value = _normalize_nested_room_value(nested_value)
            if key in {
                "x",
                "y",
                "height_mm",
                "max_iterations",
                "swing_radius_mm",
                "clearance_mm",
            }:
                normalized_value = _coerce_int_value(normalized_value)
            normalized_dict[key] = normalized_value
        return normalized_dict
    if isinstance(value, list):
        return [_normalize_nested_room_value(item) for item in value]
    return value


def _coerce_int_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    return value


def _resolve_floorplan_openings(
    input_payload: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        openings = floorplan.get("openings")
        if isinstance(openings, dict):
            doors = openings.get("doors")
            windows = openings.get("windows")
            return {
                "doors": doors if isinstance(doors, list) else [],
                "windows": windows if isinstance(windows, list) else [],
            }
    constraints = input_payload.get("constraints")
    if isinstance(constraints, dict):
        openings = constraints.get("openings")
        if isinstance(openings, dict):
            doors = openings.get("doors")
            windows = openings.get("windows")
            return {
                "doors": doors if isinstance(doors, list) else [],
                "windows": windows if isinstance(windows, list) else [],
            }
    return {"doors": [], "windows": []}


def _resolve_room_polygon(
    input_payload: Mapping[str, object],
) -> list[dict[str, int]] | None:
    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        room = floorplan.get("room")
        if isinstance(room, dict):
            poly = room.get("polygon_mm")
            if isinstance(poly, list) and len(poly) >= 3:
                return _normalize_polygon_points(poly)

    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        poly = user_input.get("shape_points")
        if isinstance(poly, list) and len(poly) >= 3:
            return _normalize_polygon_points(poly)
    return None


def _resolve_room_id(input_payload: Mapping[str, object]) -> str:
    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        room = floorplan.get("room")
        if isinstance(room, dict):
            room_id = room.get("room_id")
            if isinstance(room_id, str) and room_id:
                return room_id
    return "room_1"


def _resolve_room_name(input_payload: Mapping[str, object]) -> str:
    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        room = floorplan.get("room")
        if isinstance(room, dict):
            name = room.get("name")
            if isinstance(name, str) and name:
                return name
    return "Main"


def _resolve_room_type(input_payload: Mapping[str, object]) -> str:
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        room_type = user_input.get("room_type")
        if isinstance(room_type, str) and room_type:
            return room_type
    return "unknown"


def _resolve_style(input_payload: Mapping[str, object]) -> str:
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        style = user_input.get("style")
        if isinstance(style, str) and style:
            return style
    return "unspecified"


def _resolve_height_mm(input_payload: Mapping[str, object]) -> int:
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        for key in ("height_mm", "height"):
            height = user_input.get(key)
            if isinstance(height, (int, float)) and height > 0:
                return int(round(height))
    return 2800


def _resolve_window_direction(input_payload: Mapping[str, object]) -> str:
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        direction = user_input.get("window_direction")
        if isinstance(direction, str):
            return direction
    return ""


def _resolve_max_iterations(input_payload: Mapping[str, object]) -> int:
    value = input_payload.get("max_iterations")
    if isinstance(value, int) and value > 0:
        return value
    return 12


def _normalize_polygon_points(points: list[object]) -> list[dict[str, int]]:
    normalized: list[dict[str, int]] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            normalized.append({"x": int(round(x)), "y": int(round(y))})
    return normalized


def _ensure_ccw(points: list[dict[str, int]]) -> list[dict[str, int]]:
    if len(points) < 3:
        return points
    area = _signed_area(points)
    return list(reversed(points)) if area < 0 else points


def _signed_area(points: list[dict[str, int]]) -> float:
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]["x"], points[i]["y"]
        x2, y2 = points[(i + 1) % n]["x"], points[(i + 1) % n]["y"]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _apply_default_openings(payload: dict[str, object]) -> dict[str, object]:
    room = payload.get("room")
    if not isinstance(room, dict):
        return payload
    room_poly = room.get("polygon_ccw")
    points = _extract_polygon_points(room_poly)
    if len(points) < 3:
        return payload

    openings = payload.get("openings")
    if not isinstance(openings, dict):
        openings = {}
        payload["openings"] = openings

    doors = openings.get("doors")
    if not isinstance(doors, list):
        doors = []
        openings["doors"] = doors
    windows = openings.get("windows")
    if not isinstance(windows, list):
        windows = []
        openings["windows"] = windows

    obstacles = payload.get("obstacles")
    if not isinstance(obstacles, list):
        obstacles = []
        payload["obstacles"] = obstacles

    existing_ids = {str(d.get("id")) for d in doors if isinstance(d, dict)} | {
        str(w.get("id")) for w in windows if isinstance(w, dict)
    }

    edges = _polygon_edges(points)
    if not edges:
        return payload
    edges_sorted = sorted(edges, key=lambda edge: edge["length"], reverse=True)
    door_edge = edges_sorted[0]
    window_edge = edges_sorted[1] if len(edges_sorted) > 1 else edges_sorted[0]

    if not doors:
        door_id = _unique_opening_id("door", existing_ids)
        door = _make_opening_on_edge(
            door_id,
            door_edge,
            length_mm=900,
            clearance_mm=600,
            kind="door",
        )
        if door is not None:
            doors.append(door["opening"])
            obstacles.append(door["obstacle"])
            existing_ids.add(door_id)

    if not windows:
        window_id = _unique_opening_id("window", existing_ids)
        window = _make_opening_on_edge(
            window_id,
            window_edge,
            length_mm=1200,
            clearance_mm=100,
            kind="window",
        )
        if window is not None:
            windows.append(window["opening"])
            obstacles.append(window["obstacle"])
            existing_ids.add(window_id)

    return payload


def _extract_polygon_points(value: object) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            points.append((float(x), float(y)))
    return points


def _polygon_edges(points: list[tuple[float, float]]) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for idx in range(len(points)):
        start = points[idx]
        end = points[(idx + 1) % len(points)]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        edges.append({"start": start, "end": end, "length": length})
    return edges


def _unique_opening_id(prefix: str, existing: set[str]) -> str:
    counter = 1
    while True:
        candidate = f"{prefix}_{counter}"
        if candidate not in existing:
            return candidate
        counter += 1


def _make_opening_on_edge(
    opening_id: str,
    edge: dict[str, object],
    *,
    length_mm: float,
    clearance_mm: float,
    kind: Literal["door", "window"],
) -> dict[str, object] | None:
    start = edge.get("start")
    end = edge.get("end")
    if not isinstance(start, tuple) or not isinstance(end, tuple):
        return None
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    edge_len = math.hypot(dx, dy)
    if edge_len <= 0:
        return None

    usable_len = min(length_mm, edge_len * 0.7)
    if usable_len < 300:
        return None

    dir_x = dx / edge_len
    dir_y = dy / edge_len
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0
    half_len = usable_len / 2.0
    seg_start = (mid_x - dir_x * half_len, mid_y - dir_y * half_len)
    seg_end = (mid_x + dir_x * half_len, mid_y + dir_y * half_len)

    # For CCW polygons, inward normal points left of edge.
    normal_x = -dir_y
    normal_y = dir_x
    offset_x = normal_x * clearance_mm
    offset_y = normal_y * clearance_mm
    p1 = (seg_start[0], seg_start[1])
    p2 = (seg_end[0], seg_end[1])
    p3 = (seg_end[0] + offset_x, seg_end[1] + offset_y)
    p4 = (seg_start[0] + offset_x, seg_start[1] + offset_y)

    opening: dict[str, object]
    obstacle: dict[str, object]
    if kind == "door":
        opening = {
            "id": opening_id,
            "segment_mm": [
                {"x": int(round(p1[0])), "y": int(round(p1[1]))},
                {"x": int(round(p2[0])), "y": int(round(p2[1]))},
            ],
            "swing_radius_mm": int(round(clearance_mm)),
            "hinge_hint": "UNKNOWN",
        }
        obstacle = {
            "id": f"{opening_id}_swing",
            "type": "door_swing",
            "polygon_ccw": [
                {"x": int(round(p1[0])), "y": int(round(p1[1]))},
                {"x": int(round(p2[0])), "y": int(round(p2[1]))},
                {"x": int(round(p3[0])), "y": int(round(p3[1]))},
                {"x": int(round(p4[0])), "y": int(round(p4[1]))},
            ],
            "hard": True,
        }
    else:
        opening = {
            "id": opening_id,
            "segment_mm": [
                {"x": int(round(p1[0])), "y": int(round(p1[1]))},
                {"x": int(round(p2[0])), "y": int(round(p2[1]))},
            ],
            "clearance_mm": int(round(clearance_mm)),
        }
        obstacle = {
            "id": f"{opening_id}_clearance",
            "type": "window_clearance",
            "polygon_ccw": [
                {"x": int(round(p1[0])), "y": int(round(p1[1]))},
                {"x": int(round(p2[0])), "y": int(round(p2[1]))},
                {"x": int(round(p3[0])), "y": int(round(p3[1]))},
                {"x": int(round(p4[0])), "y": int(round(p4[1]))},
            ],
            "hard": True,
        }

    return {"opening": opening, "obstacle": obstacle}
