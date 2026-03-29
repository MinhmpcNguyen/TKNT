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

DEFAULT_DOOR_SWING_RADIUS_MM = 600
DEFAULT_WINDOW_CLEARANCE_MM = 100
DEFAULT_ROOM_HEIGHT_MM = 2800
DEFAULT_MAX_ITERATIONS = 12


@dataclass(frozen=True)
class RoomInterpreter:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = ROOM_INTERPRETER_PROMPT

    def build_messages(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None = None,
        special_notes: str | None = None,
    ) -> list[ChatMessage]:
        payload = json.dumps(
            _build_summary_input(
                input_payload,
                description=description,
                special_notes=special_notes,
            ),
            ensure_ascii=True,
        )
        user_prompt = f"{self.prompt_template}\n\nINPUT_JSON:\n{payload}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_raw(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None = None,
        special_notes: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._build_output_payload(
            input_payload,
            description=description,
            special_notes=special_notes,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def generate(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None = None,
        special_notes: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> RoomInterpreterOutput:
        payload = self._build_output_payload(
            input_payload,
            description=description,
            special_notes=special_notes,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return RoomInterpreterOutput.model_validate(payload)

    def _build_output_payload(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None,
        special_notes: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, object]:
        payload = _build_room_interpreter_payload(input_payload)
        payload = _apply_default_openings(payload)
        payload = _ensure_opening_obstacles(payload)
        payload = _normalize_room_interpreter_numbers(payload)
        payload["notes"] = self._summarize_user_guidance_notes(
            input_payload,
            description=description,
            special_notes=special_notes,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return payload

    def _summarize_user_guidance_notes(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None,
        special_notes: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> list[str]:
        summary_input = _build_summary_input(
            input_payload,
            description=description,
            special_notes=special_notes,
        )
        fallback_notes = _fallback_guidance_notes(summary_input)
        if not _has_guidance(summary_input):
            return fallback_notes

        messages = self.build_messages(
            input_payload,
            description=description,
            special_notes=special_notes,
        )
        try:
            response = get_llm_client().chat_completion(
                messages,
                model_key="helper",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = _extract_content(response)
            notes = _parse_notes_payload(raw)
        except Exception:
            notes = []

        return notes or fallback_notes


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("OpenAI response missing message content")


def _parse_notes_payload(raw: str) -> list[str]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = json.loads(_extract_json_object(text))
    if not isinstance(payload, dict):
        return []
    return _normalize_notes(payload.get("notes"))


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
        return "{}"
    return text[start : end + 1]


def _build_room_interpreter_payload(
    input_payload: Mapping[str, object],
) -> dict[str, object]:
    room_polygon = _resolve_room_polygon(input_payload) or []
    room_polygon = _ensure_ccw(room_polygon)

    status: Literal["OK", "NEED_INFO", "CONFLICT"] = "OK"
    missing: list[str] = []
    conflicts: list[str] = []

    if len(room_polygon) < 3:
        status = "NEED_INFO"
        missing.append("room.polygon_ccw")

    if _room_polygon_conflicts(input_payload):
        conflicts.append(
            "floorplan_geometry.room.polygon_mm conflicts with user_input.shape_points; using floorplan geometry."
        )
        if status == "OK":
            status = "CONFLICT"

    doors = _normalize_doors(_resolve_floorplan_openings(input_payload).get("doors"))
    windows = _normalize_windows(
        _resolve_floorplan_openings(input_payload).get("windows")
    )

    payload: dict[str, object] = {
        "status": status,
        "room": {
            "unit": "mm",
            "room_id": _resolve_room_id(input_payload) or "room_1",
            "name": _resolve_room_name(input_payload) or "Main",
            "polygon_ccw": room_polygon,
        },
        "openings": {
            "doors": doors,
            "windows": windows,
        },
        "obstacles": _resolve_input_obstacles(input_payload),
        "meta": {
            "room_type": _resolve_room_type(input_payload),
            "style": _resolve_style(input_payload),
            "height_mm": _resolve_height_mm(input_payload),
            "window_direction": _resolve_window_direction(input_payload),
            "max_iterations": _resolve_max_iterations(input_payload),
        },
        "notes": [],
        "missing": missing,
        "conflicts": conflicts,
    }
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


def _build_summary_input(
    input_payload: Mapping[str, object],
    *,
    description: str | None,
    special_notes: str | None,
) -> dict[str, object]:
    user_input = input_payload.get("user_input")
    user_input = user_input if isinstance(user_input, dict) else {}

    guidance: dict[str, str] = {}
    if description and description.strip():
        guidance["description"] = description.strip()
    if special_notes and special_notes.strip():
        guidance["special_notes"] = special_notes.strip()

    for key, value in user_input.items():
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        guidance[key] = text

    constraints = input_payload.get("constraints")
    if isinstance(constraints, dict):
        feng_shui = constraints.get("feng_shui")
        if isinstance(feng_shui, str) and feng_shui.strip():
            guidance.setdefault("feng_shui", feng_shui.strip())

    return {
        "room_context": {
            "room_type": _resolve_room_type(input_payload),
            "style": _resolve_style(input_payload),
            "window_direction": _resolve_window_direction(input_payload),
        },
        "user_guidance": guidance,
    }


def _has_guidance(summary_input: Mapping[str, object]) -> bool:
    guidance = summary_input.get("user_guidance")
    return isinstance(guidance, dict) and any(
        isinstance(value, str) and value.strip() for value in guidance.values()
    )


def _fallback_guidance_notes(summary_input: Mapping[str, object]) -> list[str]:
    notes: list[str] = []
    room_context = summary_input.get("room_context")
    guidance = summary_input.get("user_guidance")

    room_type = None
    style = None
    if isinstance(room_context, dict):
        room_type = room_context.get("room_type")
        style = room_context.get("style")

    if isinstance(room_type, str) and room_type and room_type != "unknown":
        notes.append(f"Design target room type: {room_type}.")
    if isinstance(style, str) and style and style != "unspecified":
        notes.append(f"Requested style: {style}.")

    if isinstance(guidance, dict):
        for key in (
            "description",
            "special_description",
            "special_notes",
            "notes",
            "feng_shui",
        ):
            value = guidance.get(key)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text:
                continue
            notes.append(text)

    return _normalize_notes(notes)


def _normalize_notes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    notes: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        notes.append(text)
    return notes[:6]


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


def _room_polygon_conflicts(input_payload: Mapping[str, object]) -> bool:
    floorplan = input_payload.get("floorplan_geometry")
    user_input = input_payload.get("user_input")
    if not isinstance(floorplan, dict) or not isinstance(user_input, dict):
        return False

    room = floorplan.get("room")
    if not isinstance(room, dict):
        return False

    floorplan_polygon = room.get("polygon_mm")
    shape_points = user_input.get("shape_points")
    if not isinstance(floorplan_polygon, list) or not isinstance(shape_points, list):
        return False

    normalized_floorplan = _ensure_ccw(_normalize_polygon_points(floorplan_polygon))
    normalized_shape = _ensure_ccw(_normalize_polygon_points(shape_points))
    if len(normalized_floorplan) < 3 or len(normalized_shape) < 3:
        return False
    return normalized_floorplan != normalized_shape


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
    return DEFAULT_ROOM_HEIGHT_MM


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
    return DEFAULT_MAX_ITERATIONS


def _normalize_polygon_points(points: list[object]) -> list[dict[str, int]]:
    normalized: list[dict[str, int]] = []
    for item in points:
        point = _normalize_point(item)
        if point is not None:
            normalized.append(point)
    return normalized


def _normalize_point(item: object) -> dict[str, int] | None:
    if not isinstance(item, dict):
        return None
    x = item.get("x")
    y = item.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return {"x": int(round(x)), "y": int(round(y))}
    return None


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


def _normalize_doors(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    doors: list[dict[str, object]] = []
    existing_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        door = _normalize_single_door(item, fallback_id=f"door_{index}")
        if door is None:
            continue
        opening_id = _ensure_unique_identifier(str(door["id"]), existing_ids)
        door["id"] = opening_id
        existing_ids.add(opening_id)
        doors.append(door)
    return doors


def _normalize_single_door(
    value: object,
    *,
    fallback_id: str,
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None

    opening_id = _coerce_identifier(
        value.get("id") or value.get("door_id"),
        fallback=fallback_id,
    )
    segment = _normalize_segment(
        value.get("segment_mm"),
        start=value.get("start_mm"),
        end=value.get("end_mm"),
    )
    if segment is None:
        return None

    swing_radius_mm = _coerce_positive_int(
        value.get("swing_radius_mm") or value.get("leaf_width_mm"),
        fallback=DEFAULT_DOOR_SWING_RADIUS_MM,
    )
    hinge_hint = _normalize_hinge_hint(
        value.get("hinge_hint") or value.get("hinge_side")
    )
    return {
        "id": opening_id,
        "segment_mm": segment,
        "swing_radius_mm": swing_radius_mm,
        "hinge_hint": hinge_hint,
    }


def _normalize_windows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    windows: list[dict[str, object]] = []
    existing_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        window = _normalize_single_window(item, fallback_id=f"window_{index}")
        if window is None:
            continue
        opening_id = _ensure_unique_identifier(str(window["id"]), existing_ids)
        window["id"] = opening_id
        existing_ids.add(opening_id)
        windows.append(window)
    return windows


def _normalize_single_window(
    value: object,
    *,
    fallback_id: str,
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None

    opening_id = _coerce_identifier(
        value.get("id") or value.get("window_id"),
        fallback=fallback_id,
    )
    segment = _normalize_segment(
        value.get("segment_mm"),
        start=value.get("start_mm"),
        end=value.get("end_mm"),
    )
    if segment is None:
        return None

    clearance_mm = _coerce_positive_int(
        value.get("clearance_mm"),
        fallback=DEFAULT_WINDOW_CLEARANCE_MM,
    )
    return {
        "id": opening_id,
        "segment_mm": segment,
        "clearance_mm": clearance_mm,
    }


def _normalize_segment(
    value: object,
    *,
    start: object | None = None,
    end: object | None = None,
) -> list[dict[str, int]] | None:
    if isinstance(value, list) and len(value) == 2:
        p1 = _normalize_point(value[0])
        p2 = _normalize_point(value[1])
        if p1 is not None and p2 is not None:
            return [p1, p2]

    p1 = _normalize_point(start)
    p2 = _normalize_point(end)
    if p1 is not None and p2 is not None:
        return [p1, p2]
    return None


def _coerce_identifier(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _coerce_positive_int(value: object, *, fallback: int) -> int:
    if isinstance(value, (int, float)) and float(value) > 0:
        return int(round(float(value)))
    return fallback


def _normalize_hinge_hint(value: object) -> Literal["UNKNOWN", "LEFT", "RIGHT"]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"left", "l"}:
            return "LEFT"
        if normalized in {"right", "r"}:
            return "RIGHT"
    return "UNKNOWN"


def _ensure_unique_identifier(identifier: str, existing_ids: set[str]) -> str:
    if identifier not in existing_ids:
        return identifier
    return _unique_opening_id(identifier, existing_ids)


def _resolve_input_obstacles(
    input_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    candidates: list[object] = []

    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        openings = floorplan.get("openings")
        if isinstance(openings, dict):
            obstacles = openings.get("obstacles")
            if isinstance(obstacles, list):
                candidates.extend(obstacles)

    constraints = input_payload.get("constraints")
    if isinstance(constraints, dict):
        obstacles = constraints.get("obstacles")
        if isinstance(obstacles, list):
            candidates.extend(obstacles)
        no_go_zones = constraints.get("no_go_zones")
        if isinstance(no_go_zones, list):
            candidates.extend(no_go_zones)

    normalized: list[dict[str, object]] = []
    existing_ids: set[str] = set()
    for index, item in enumerate(candidates, start=1):
        obstacle = _normalize_obstacle(item, fallback_id=f"obstacle_{index}")
        if obstacle is None:
            continue
        obstacle_id = _ensure_unique_identifier(str(obstacle["id"]), existing_ids)
        obstacle["id"] = obstacle_id
        existing_ids.add(obstacle_id)
        normalized.append(obstacle)
    return normalized


def _normalize_obstacle(
    value: object,
    *,
    fallback_id: str,
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None

    obstacle_type = _normalize_obstacle_type(value.get("type"))
    if obstacle_type is None:
        obstacle_type = "no_go"

    polygon = value.get("polygon_ccw")
    if not isinstance(polygon, list):
        polygon = value.get("polygon_mm")
    if not isinstance(polygon, list):
        return None

    normalized_polygon = _ensure_ccw(_normalize_polygon_points(polygon))
    if len(normalized_polygon) < 3:
        return None

    return {
        "id": _coerce_identifier(
            value.get("id") or value.get("obstacle_id"),
            fallback=fallback_id,
        ),
        "type": obstacle_type,
        "polygon_ccw": normalized_polygon,
        "hard": bool(value.get("hard", True)),
    }


def _normalize_obstacle_type(
    value: object,
) -> Literal["door_swing", "window_clearance", "no_go"] | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"door_swing", "door"}:
        return "door_swing"
    if normalized in {"window_clearance", "window"}:
        return "window_clearance"
    if normalized in {"no_go", "nogo", "blocked"}:
        return "no_go"
    return None


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
            clearance_mm=DEFAULT_DOOR_SWING_RADIUS_MM,
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
            clearance_mm=DEFAULT_WINDOW_CLEARANCE_MM,
            kind="window",
        )
        if window is not None:
            windows.append(window["opening"])
            obstacles.append(window["obstacle"])
            existing_ids.add(window_id)

    return payload


def _ensure_opening_obstacles(payload: dict[str, object]) -> dict[str, object]:
    room = payload.get("room")
    if not isinstance(room, dict):
        return payload

    room_poly = room.get("polygon_ccw")
    room_points = _extract_polygon_points(room_poly)
    if len(room_points) < 3:
        return payload

    openings = payload.get("openings")
    if not isinstance(openings, dict):
        return payload

    obstacles = payload.get("obstacles")
    if not isinstance(obstacles, list):
        obstacles = []
        payload["obstacles"] = obstacles

    existing_ids = {
        str(obstacle.get("id"))
        for obstacle in obstacles
        if isinstance(obstacle, dict) and isinstance(obstacle.get("id"), str)
    }

    for door in openings.get("doors") or []:
        if not isinstance(door, dict):
            continue
        obstacle = _build_obstacle_from_opening(
            opening=door,
            kind="door",
            room_points=room_points,
        )
        if obstacle is None:
            continue
        obstacle_id = str(obstacle.get("id") or "")
        if obstacle_id and obstacle_id not in existing_ids:
            obstacles.append(obstacle)
            existing_ids.add(obstacle_id)

    for window in openings.get("windows") or []:
        if not isinstance(window, dict):
            continue
        obstacle = _build_obstacle_from_opening(
            opening=window,
            kind="window",
            room_points=room_points,
        )
        if obstacle is None:
            continue
        obstacle_id = str(obstacle.get("id") or "")
        if obstacle_id and obstacle_id not in existing_ids:
            obstacles.append(obstacle)
            existing_ids.add(obstacle_id)

    return payload


def _build_obstacle_from_opening(
    *,
    opening: Mapping[str, object],
    kind: Literal["door", "window"],
    room_points: list[tuple[float, float]],
) -> dict[str, object] | None:
    opening_id = opening.get("id")
    if not isinstance(opening_id, str) or not opening_id:
        return None

    segment = _normalize_segment(opening.get("segment_mm"))
    if segment is None:
        return None

    p1 = (float(segment[0]["x"]), float(segment[0]["y"]))
    p2 = (float(segment[1]["x"]), float(segment[1]["y"]))
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    segment_len = math.hypot(dx, dy)
    if segment_len <= 0:
        return None

    dir_x = dx / segment_len
    dir_y = dy / segment_len
    clearance_mm = (
        _coerce_positive_int(
            opening.get("swing_radius_mm"),
            fallback=DEFAULT_DOOR_SWING_RADIUS_MM,
        )
        if kind == "door"
        else _coerce_positive_int(
            opening.get("clearance_mm"),
            fallback=DEFAULT_WINDOW_CLEARANCE_MM,
        )
    )

    normal_x, normal_y = _select_inward_normal(
        p1=p1,
        p2=p2,
        dir_x=dir_x,
        dir_y=dir_y,
        room_points=room_points,
        clearance_mm=float(clearance_mm),
    )

    offset_x = normal_x * clearance_mm
    offset_y = normal_y * clearance_mm
    p3 = (p2[0] + offset_x, p2[1] + offset_y)
    p4 = (p1[0] + offset_x, p1[1] + offset_y)

    polygon = _ensure_ccw(
        [
            {"x": int(round(p1[0])), "y": int(round(p1[1]))},
            {"x": int(round(p2[0])), "y": int(round(p2[1]))},
            {"x": int(round(p3[0])), "y": int(round(p3[1]))},
            {"x": int(round(p4[0])), "y": int(round(p4[1]))},
        ]
    )

    if kind == "door":
        obstacle_id = f"{opening_id}_swing"
        obstacle_type = "door_swing"
    else:
        obstacle_id = f"{opening_id}_clearance"
        obstacle_type = "window_clearance"

    return {
        "id": obstacle_id,
        "type": obstacle_type,
        "polygon_ccw": polygon,
        "hard": True,
    }


def _select_inward_normal(
    *,
    p1: tuple[float, float],
    p2: tuple[float, float],
    dir_x: float,
    dir_y: float,
    room_points: list[tuple[float, float]],
    clearance_mm: float,
) -> tuple[float, float]:
    half_clearance = max(clearance_mm * 0.5, 10.0)
    mid_x = (p1[0] + p2[0]) / 2.0
    mid_y = (p1[1] + p2[1]) / 2.0
    candidates = [(-dir_y, dir_x), (dir_y, -dir_x)]

    for normal_x, normal_y in candidates:
        probe = (
            mid_x + normal_x * half_clearance,
            mid_y + normal_y * half_clearance,
        )
        if _point_in_polygon(probe, room_points):
            return normal_x, normal_y

    return candidates[0]


def _point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    if count < 3:
        return False

    for idx in range(count):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % count]
        intersects = (y1 > y) != (y2 > y)
        if not intersects:
            continue
        denom = y2 - y1
        if abs(denom) < 1e-9:
            continue
        x_intersection = (x2 - x1) * (y - y1) / denom + x1
        if x < x_intersection:
            inside = not inside
    return inside


def _extract_polygon_points(value: object) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        point = _normalize_point(item)
        if point is None:
            continue
        points.append((float(point["x"]), float(point["y"])))
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
