from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent_schema.stylist_schema import StylistOutput
from clients.openai_client import OpenAIClient
from prompt.stylist import STYLIST_PROMPT
from prompt.system import SYSTEM_PROMPT
from stylist.room_essentials_seed import ROOM_SURFACE_GROUPS
from stylist.tools import TOOL_REGISTRY, TOOL_SCHEMAS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stylist:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = STYLIST_PROMPT

    def generate_raw(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> str:
        normalized_layout = _normalize_layout_json(layout_json)
        merged_user_context = _inject_tenant(user_context_json, tenant_id)
        surface_requirements = _compute_surface_requirements(normalized_layout)

        user_prompt = _build_prompt(
            template=self.prompt_template,
            layout_json=normalized_layout,
            user_context_json=merged_user_context,
            surface_requirements_json=surface_requirements,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return _run_with_tools(
            messages,
            layout_json=normalized_layout,
            user_context_json=merged_user_context,
        )

    def generate(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_layout = _normalize_layout_json(layout_json)
        raw = self.generate_raw(
            layout_json=normalized_layout,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        payload = _parse_json(raw)
        payload = _fix_place_on_targets(payload, layout_json=normalized_layout)
        return StylistOutput.model_validate(payload).model_dump()


def _build_prompt(
    *,
    template: str,
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any] | None,
    surface_requirements_json: dict[str, Any],
) -> str:
    mapping = {
        "LAYOUT_JSON": _json_block(layout_json),
        "USER_CONTEXT_JSON": _json_block(user_context_json),
        "SURFACE_REQUIREMENTS_JSON": _json_block(surface_requirements_json),
    }
    output = template
    for key, value in mapping.items():
        output = output.replace("{" + key + "}", value)
    return output


def _inject_tenant(
    user_context_json: dict[str, Any] | None,
    tenant_id: str | None,
) -> dict[str, Any] | None:
    if tenant_id is None:
        return user_context_json
    context = dict(user_context_json or {})
    context.setdefault("tenant_id", tenant_id)
    return context


def _json_block(obj: Any) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=True, indent=2)


def _normalize_layout_json(layout_json: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(layout_json, dict):
        return {}

    out = dict(layout_json)
    room = out.get("room")
    if not isinstance(room, dict):
        room = {}
    out["room"] = room

    if not isinstance(out.get("openings"), dict):
        openings = room.get("openings")
        if isinstance(openings, dict):
            out["openings"] = deepcopy(openings)

    objects = out.get("objects")
    if not isinstance(objects, list):
        return out

    normalized_objects: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        row = dict(obj)
        obj_type = row.get("object_type")
        if not isinstance(obj_type, str) or not obj_type:
            obj_id = row.get("object_id")
            if isinstance(obj_id, str) and obj_id:
                row["object_type"] = obj_id
        if not isinstance(row.get("instance_id"), str) or not row.get("instance_id"):
            obj_id = row.get("object_id")
            if isinstance(obj_id, str) and obj_id:
                row["instance_id"] = obj_id
        normalized_objects.append(row)
    out["objects"] = normalized_objects
    return out


def _run_with_tools(
    messages: list[dict[str, Any]],
    *,
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any] | None,
    max_steps: int = 10,
) -> str:
    client = OpenAIClient()

    surface_requirements = _compute_surface_requirements(layout_json)
    required_types = surface_requirements.get("required_types") or []
    candidate_types = surface_requirements.get("candidate_types") or []
    logger.info(
        "Stylist surface requirements: room_type=%s required_types=%s candidate_types=%s",
        surface_requirements.get("room_type"),
        required_types,
        candidate_types,
    )

    used_surface_groups = False
    used_inventory = False
    used_knowledge = False
    forced_tools = False

    inventory_items: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_rank: int | None = None
    best_object_count = 0

    for step in range(1, max_steps + 1):
        logger.info("Stylist step %s/%s", step, max_steps)

        try:
            response = client.chat_completion(
                messages,
                model_key="gpt_4o",
                temperature=0.0,
                tools=TOOL_SCHEMAS,
            )
        except Exception as exc:
            if _is_context_length_exceeded(exc):
                logger.warning(
                    "Stylist stopped early due to context length exceeded.",
                    exc_info=exc,
                )
                if best_payload is not None:
                    return json.dumps(best_payload, ensure_ascii=True)
                return json.dumps(
                    {
                        "status": "NEED_INFO",
                        "room": None,
                        "objects": [],
                        "notes": [
                            "Stopped due to context length exceeded before producing a final result."
                        ],
                        "missing": ["context_length_exceeded"],
                    },
                    ensure_ascii=True,
                )
            raise
        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)

        if tool_calls:
            logger.info(
                "Tool calls requested: %s",
                [(c.get("function", {}) or {}).get("name") for c in tool_calls],
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": getattr(message, "content", "") or "",
                    "tool_calls": tool_calls,
                }
            )

            for idx, call in enumerate(tool_calls):
                call_id = call.get("id") or f"tool_{idx}"
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                args_text = fn.get("arguments") or "{}"
                args = _safe_json_loads(args_text)

                args = _coerce_tool_args(
                    name=name,
                    args=args,
                    layout_json=layout_json,
                    user_context_json=user_context_json,
                    surface_requirements=surface_requirements,
                )

                tool_output = _safe_run_tool(name, args)
                logger.info("Tool %s args: %s", name, _short_json(args))
                logger.info("Tool %s output: %s", name, _short_json(tool_output))

                if name == "GetRoomSurfaceGroups":
                    used_surface_groups = True

                if name == "ListInventoryByTypes":
                    used_inventory = True
                    items = tool_output.get("items")
                    if isinstance(items, list):
                        inventory_items = [i for i in items if isinstance(i, dict)]

                if name == "ListDesignKnowledge":
                    used_knowledge = True

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": str(name),
                        "content": json.dumps(tool_output, ensure_ascii=True),
                    }
                )

            continue

        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            messages.append(
                {
                    "role": "user",
                    "content": "Return valid JSON only. Do not return prose or markdown.",
                }
            )
            continue

        logger.info("Assistant content length: %s", len(content))
        logger.info("Assistant content preview: %s", _truncate_text(content, 500))

        if (
            not (used_surface_groups and used_inventory and used_knowledge)
            and not forced_tools
        ):
            forced_tools = True
            logger.info(
                "Forcing required tools. used_surface_groups=%s used_inventory=%s used_knowledge=%s",
                used_surface_groups,
                used_inventory,
                used_knowledge,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You MUST call GetRoomSurfaceGroups(room_type), "
                        "ListInventoryByTypes(tenant_id, types[]) and "
                        "ListDesignKnowledge(tenant_id, tags, category) before final output."
                    ),
                }
            )
            continue

        draft = _try_parse_json_object(content)
        if not isinstance(draft, dict):
            messages.append(
                {
                    "role": "user",
                    "content": "Return valid JSON only. Do not include markdown fences.",
                }
            )
            continue

        status = draft.get("status")
        draft_eval = draft
        if status == "OK" and inventory_items:
            draft_eval = _snap_inventory_sizes(dict(draft), inventory_items)

        if status in {"OK", "NEED_INFO", "UNSAT"}:
            candidate = dict(draft_eval)
            if status == "OK":
                missing_required = _validate_required_types(draft_eval, required_types)
                size_issues = (
                    _validate_inventory_sizes(draft_eval, inventory_items)
                    if inventory_items
                    else []
                )
                if size_issues and inventory_items:
                    forced = _force_snap_inventory_sizes(dict(draft_eval), inventory_items)
                    forced_issues = _validate_inventory_sizes(forced, inventory_items)
                    if not forced_issues:
                        draft_eval = forced
                        candidate = dict(draft_eval)
                        size_issues = []
                if missing_required or size_issues:
                    candidate["status"] = "NEED_INFO"
                    notes = candidate.get("notes")
                    if not isinstance(notes, list):
                        notes = []
                    notes.append(
                        "Partial draft does not satisfy required_types and/or inventory sizing; downgraded to NEED_INFO."
                    )
                    candidate["notes"] = notes

                    missing = candidate.get("missing")
                    if not isinstance(missing, list):
                        missing = []
                    if missing_required:
                        missing.extend([f"required_type:{t}" for t in missing_required])
                    if size_issues:
                        missing.append("inventory_size_mismatch")
                    candidate["missing"] = missing

                    rank = _stylist_status_rank("NEED_INFO")
                else:
                    rank = _stylist_status_rank("OK")
            else:
                rank = _stylist_status_rank(str(status))
            objects = candidate.get("objects")
            obj_count = len(objects) if isinstance(objects, list) else 0
            if (
                best_rank is None
                or rank < best_rank
                or (rank == best_rank and obj_count > best_object_count)
            ):
                best_payload = candidate
                best_rank = rank
                best_object_count = obj_count

        logger.info(
            "Stylist draft status=%s objects=%s missing=%s",
            status,
            len(draft.get("objects") or []),
            draft.get("missing"),
        )
        if status not in {"OK", "NEED_INFO", "UNSAT"}:
            messages.append(
                {
                    "role": "user",
                    "content": "Top-level field status must be one of OK, NEED_INFO, UNSAT.",
                }
            )
            continue

        if status == "OK":
            missing_required = _validate_required_types(draft_eval, required_types)
            if missing_required:
                logger.warning(
                    "Stylist final rejected: missing required types = %s",
                    missing_required,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "SURFACE_REQUIREMENTS_JSON.required_types still has missing types: "
                            f"{missing_required}. "
                            "Add at least one instance for each missing type, or return NEED_INFO/UNSAT if legal placement is impossible."
                        ),
                    }
                )
                continue

            if inventory_items:
                size_issues = _validate_inventory_sizes(draft_eval, inventory_items)
                if size_issues:
                    forced = _force_snap_inventory_sizes(dict(draft_eval), inventory_items)
                    forced_issues = _validate_inventory_sizes(forced, inventory_items)
                    if not forced_issues:
                        draft_eval = forced
                        return json.dumps(draft_eval, ensure_ascii=True)
                    logger.warning(
                        "Stylist final rejected: inventory size issues = %s",
                        size_issues,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Inventory-sourced objects must match the dimensions of at least one inventory item of the same type "
                                "(rotation swap allowed). Fix these objects: "
                                + "; ".join(size_issues)
                            ),
                        }
                    )
                    continue

        if status == "OK":
            return json.dumps(draft_eval, ensure_ascii=True)
        return content

    if best_payload is not None:
        if inventory_items and isinstance(best_payload, dict):
            forced = _force_snap_inventory_sizes(dict(best_payload), inventory_items)
            forced_issues = _validate_inventory_sizes(forced, inventory_items)
            if not forced_issues:
                best_payload = forced
                if isinstance(best_payload.get("missing"), list):
                    best_payload["missing"] = [
                        x
                        for x in best_payload["missing"]
                        if x != "inventory_size_mismatch"
                    ]
                best_payload["status"] = "OK"
        logger.warning(
            "Stylist hit max_steps; returning best partial payload with status=%s",
            best_payload.get("status"),
        )
        return json.dumps(best_payload, ensure_ascii=True)

    logger.warning("Stylist hit max_steps with no usable draft.")
    return json.dumps(
        {
            "status": "NEED_INFO",
            "room": None,
            "objects": [],
            "notes": ["Stopped due to max_steps with no usable draft."],
            "missing": ["max_steps_exceeded"],
        },
        ensure_ascii=True,
    )


def _coerce_tool_args(
    *,
    name: str | None,
    args: dict[str, Any],
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any] | None,
    surface_requirements: dict[str, Any],
) -> dict[str, Any]:
    out = dict(args)

    room_type = _extract_room_type(layout_json)
    tenant_id = _extract_tenant_id(user_context_json, layout_json)
    style_tags = _extract_style_tags(user_context_json, layout_json)

    if name == "GetRoomSurfaceGroups":
        out["room_type"] = room_type

    elif name == "ListInventoryByTypes":
        out["tenant_id"] = tenant_id
        requested_types = out.get("types")
        if not isinstance(requested_types, list) or not requested_types:
            requested_types = surface_requirements.get("candidate_types") or []
        normalized = []
        for t in requested_types:
            if isinstance(t, str) and t and t not in normalized:
                normalized.append(t)
        out["types"] = normalized

    elif name == "ListDesignKnowledge":
        out["tenant_id"] = tenant_id

        # IMPORTANT: DesignKnowledgeFilter uses "tags @> <tags>" (contains all tags).
        # To avoid over-filtering, we only filter by room_type as requested.
        tags: list[str] = ["styling"]
        if room_type and room_type != "unknown":
            tags.append(room_type)

        out["tags"] = tags
        out["category"] = None

    return out


def _compute_surface_requirements(layout_json: dict[str, Any]) -> dict[str, Any]:
    room_type = _extract_room_type(layout_json)
    groups = ROOM_SURFACE_GROUPS.get(room_type, {})

    no_stack = groups.get("no_stack") or []
    can_stack_map = groups.get("can_stack_or_be_stacked_or_hang_or_soft") or {}

    base_present: set[str] = set()
    for obj in layout_json.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("object_type")
        if isinstance(obj_type, str) and obj_type in no_stack:
            base_present.add(obj_type)

    active_anchors: list[str] = []
    for base in sorted(base_present):
        if base in can_stack_map:
            active_anchors.append(base)

    openings = layout_json.get("openings") or {}
    has_openings = bool(
        (openings.get("doors") or []) or (openings.get("windows") or [])
    )
    if has_openings and "__opening__" in can_stack_map:
        active_anchors.append("__opening__")
    if "__wall__" in can_stack_map:
        active_anchors.append("__wall__")
    if "__ceiling__" in can_stack_map:
        active_anchors.append("__ceiling__")
    if "__utility_zone__" in can_stack_map:
        active_anchors.append("__utility_zone__")

    required_types: list[str] = []
    candidate_types: list[str] = []

    for anchor in active_anchors:
        spec = can_stack_map.get(anchor)

        # Backward-compatible mode:
        # - list[str] => candidate only
        # - dict => may contain required / optional / recommended
        if isinstance(spec, list):
            for item in spec:
                if isinstance(item, str):
                    candidate_types.append(item)
            continue

        if isinstance(spec, dict):
            for item in spec.get("required", []) or []:
                if isinstance(item, str):
                    required_types.append(item)
                    candidate_types.append(item)

            for key in ("optional", "recommended", "candidates", "items"):
                for item in spec.get(key, []) or []:
                    if isinstance(item, str):
                        candidate_types.append(item)

    required_types = _stable_unique(required_types)
    candidate_types = _stable_unique(candidate_types)

    return {
        "room_type": room_type,
        "base_present": sorted(base_present),
        "active_anchors": active_anchors,
        "required_types": required_types,
        "candidate_types": candidate_types,
    }


def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _extract_tool_calls(message: object) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []
    output: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            output.append(call)
            continue
        fn = getattr(call, "function", None)
        output.append(
            {
                "id": getattr(call, "id", None),
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": getattr(fn, "name", None) if fn is not None else None,
                    "arguments": getattr(fn, "arguments", None)
                    if fn is not None
                    else None,
                },
            }
        )
    return output


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_run_tool(name: str | None, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return {"error": "unknown_tool", "tool": name, "args": args}
    try:
        return TOOL_REGISTRY[name](**args)
    except Exception as exc:
        return {
            "error": "tool_failed",
            "tool": name,
            "message": str(exc),
            "args": args,
        }


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Stylist returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Stylist response must be a JSON object")
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


def _truncate_text(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _short_json(obj: Any, max_len: int = 1200) -> str:
    try:
        dumped = json.dumps(obj, ensure_ascii=True)
    except Exception:
        dumped = str(obj)
    return _truncate_text(dumped, max_len=max_len)


def _is_context_length_exceeded(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == "context_length_exceeded":
        return True
    text = str(exc)
    return ("context_length_exceeded" in text) or ("maximum context length" in text)


def _stylist_status_rank(status: str) -> int:
    return {"OK": 0, "NEED_INFO": 1, "UNSAT": 2}.get(status, 3)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _extract_room_type(layout_json: dict[str, Any]) -> str:
    if not isinstance(layout_json, dict):
        return "unknown"
    room = layout_json.get("room") or {}
    return str(room.get("room_type") or "unknown")


def _extract_tenant_id(
    user_context_json: dict[str, Any] | None,
    layout_json: dict[str, Any],
) -> str | None:
    if isinstance(user_context_json, dict):
        tenant_id = user_context_json.get("tenant_id")
        if isinstance(tenant_id, str) and tenant_id.strip():
            return tenant_id.strip()

    tenant_id = layout_json.get("tenant_id")
    if isinstance(tenant_id, str) and tenant_id.strip():
        return tenant_id.strip()

    room = layout_json.get("room") or {}
    tenant_id = room.get("tenant_id")
    if isinstance(tenant_id, str) and tenant_id.strip():
        return tenant_id.strip()

    return None


def _extract_style_tags(
    user_context_json: dict[str, Any] | None,
    layout_json: dict[str, Any],
) -> list[str]:
    tags: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value.strip() and value not in tags:
            tags.append(value.strip())

    if isinstance(user_context_json, dict):
        style = user_context_json.get("style")
        _add(style)

        preferred = user_context_json.get("style_tags")
        if isinstance(preferred, list):
            for item in preferred:
                _add(item)

        user_input = user_context_json.get("user_input")
        if isinstance(user_input, dict):
            _add(user_input.get("style"))
            style_tags = user_input.get("style_tags")
            if isinstance(style_tags, list):
                for item in style_tags:
                    _add(item)

    room = layout_json.get("room") or {}
    _add(room.get("style"))

    meta = layout_json.get("meta") or {}
    if isinstance(meta, dict):
        _add(meta.get("style"))
        style_tags = meta.get("style_tags")
        if isinstance(style_tags, list):
            for item in style_tags:
                _add(item)

    return tags


def _validate_required_types(
    draft: dict[str, Any],
    required_types: list[str],
) -> list[str]:
    if not required_types:
        return []

    objects = draft.get("objects") or []
    present: set[str] = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("object_type")
        if isinstance(obj_type, str):
            present.add(obj_type)

    return [t for t in required_types if t not in present]


def _snap_inventory_sizes(
    draft: dict[str, Any],
    inventory_items: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(draft, dict):
        return draft

    objects = draft.get("objects")
    if not isinstance(objects, list):
        return draft

    def _to_int_mm(value: Any) -> int | None:
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        return None

    dims_by_type: dict[str, list[tuple[int, int, int]]] = {}
    for item in inventory_items:
        obj_type = item.get("type")
        if not isinstance(obj_type, str) or not obj_type:
            continue
        dims = item.get("dimensions") or {}
        w = None
        h = None
        hh = None
        if isinstance(dims, dict):
            w = _to_int_mm(dims.get("length_mm"))
            h = _to_int_mm(dims.get("width_mm"))
            hh = _to_int_mm(dims.get("height_mm"))
        if w is None or h is None:
            w = _to_int_mm(item.get("length_mm"))
            h = _to_int_mm(item.get("width_mm"))
        if hh is None:
            hh = _to_int_mm(item.get("height_mm"))
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            dims_by_type.setdefault(obj_type.lower(), []).append((w, h, hh or 0))

    if not dims_by_type:
        return draft

    def _bbox_from_poly(poly: list[dict[str, Any]]) -> tuple[int, int, int, int] | None:
        if not poly:
            return None
        xs, ys = [], []
        for pt in poly:
            try:
                xs.append(int(pt.get("x")))
                ys.append(int(pt.get("y")))
            except Exception:
                return None
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def _rect_poly(min_x: int, min_y: int, max_x: int, max_y: int) -> list[dict[str, int]]:
        return [
            {"x": min_x, "y": min_y},
            {"x": max_x, "y": min_y},
            {"x": max_x, "y": max_y},
            {"x": min_x, "y": max_y},
        ]

    out_objects: list[dict[str, Any]] = []
    adjusted = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        row = dict(obj)
        if row.get("source") != "inventory":
            out_objects.append(row)
            continue

        obj_type = row.get("object_type")
        if not isinstance(obj_type, str):
            out_objects.append(row)
            continue
        dims_list = dims_by_type.get(obj_type.lower()) or []
        if not dims_list:
            out_objects.append(row)
            continue

        bbox = row.get("bbox")
        if not isinstance(bbox, dict):
            poly = row.get("polygon_ccw") or []
            bbox_vals = _bbox_from_poly(poly)
            if bbox_vals is None:
                out_objects.append(row)
                continue
            min_x, min_y, max_x, max_y = bbox_vals
        else:
            try:
                min_x = int(bbox.get("min_x"))
                min_y = int(bbox.get("min_y"))
                max_x = int(bbox.get("max_x"))
                max_y = int(bbox.get("max_y"))
            except Exception:
                out_objects.append(row)
                continue

        cur_w = max_x - min_x
        cur_h = max_y - min_y
        if cur_w <= 0 or cur_h <= 0:
            out_objects.append(row)
            continue

        place_on = row.get("place_on") if isinstance(row.get("place_on"), dict) else {}
        method = str(place_on.get("method") or "floor")

        def _pairs_for_dims(L: int, W: int, H: int, *, method: str) -> list[tuple[int, int]]:
            if method in {"floor", "on_top"}:
                return [(L, W)]
            pairs: list[tuple[int, int]] = []
            if W > 0 and H > 0:
                pairs.append((W, H))
            if L > 0 and W > 0:
                pairs.append((L, W))
            if L > 0 and H > 0:
                pairs.append((L, H))
            return pairs

        best = None
        best_score = None
        for w, h, hh in dims_list:
            for cand_w, cand_h in _pairs_for_dims(w, h, hh, method=method):
                score = abs(cur_w - cand_w) + abs(cur_h - cand_h)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (cand_w, cand_h)

        if best is None:
            out_objects.append(row)
            continue

        new_w, new_h = best
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        nmin_x = int(round(cx - new_w / 2.0))
        nmax_x = int(round(cx + new_w / 2.0))
        nmin_y = int(round(cy - new_h / 2.0))
        nmax_y = int(round(cy + new_h / 2.0))

        row["bbox"] = {
            "min_x": nmin_x,
            "min_y": nmin_y,
            "max_x": nmax_x,
            "max_y": nmax_y,
        }
        row["polygon_ccw"] = _rect_poly(nmin_x, nmin_y, nmax_x, nmax_y)
        if cur_w != new_w or cur_h != new_h:
            adjusted += 1
        out_objects.append(row)

    draft["objects"] = out_objects
    if adjusted:
        logger.info("Stylist snapped inventory sizes for %s object(s).", adjusted)
    return draft


def _force_snap_inventory_sizes(
    draft: dict[str, Any],
    inventory_items: list[dict[str, Any]],
) -> dict[str, Any]:
    draft = _snap_inventory_sizes(draft, inventory_items)

    if not isinstance(draft, dict):
        return draft

    objects = draft.get("objects")
    if not isinstance(objects, list):
        return draft

    dims_by_type: dict[str, list[tuple[int, int, int]]] = {}
    for item in inventory_items:
        obj_type = item.get("type")
        dims = item.get("dimensions") or {}
        if not isinstance(obj_type, str) or not isinstance(dims, dict):
            continue
        L = int(dims.get("length_mm") or 0)
        W = int(dims.get("width_mm") or 0)
        H = int(dims.get("height_mm") or 0)
        if L <= 0 or W <= 0:
            continue
        dims_by_type.setdefault(obj_type, []).append((L, W, H))

    if not dims_by_type:
        return draft

    def pairs_for_dims(L: int, W: int, H: int, *, method: str) -> list[tuple[int, int]]:
        if method in {"floor", "on_top"}:
            return [(L, W)]
        pairs: list[tuple[int, int]] = []
        if W > 0 and H > 0:
            pairs.append((W, H))
        if L > 0 and W > 0:
            pairs.append((L, W))
        if L > 0 and H > 0:
            pairs.append((L, H))
        return pairs

    def _bbox_from_poly(poly: list[dict[str, Any]]) -> tuple[int, int, int, int] | None:
        if not poly:
            return None
        xs, ys = [], []
        for pt in poly:
            try:
                xs.append(int(pt.get("x")))
                ys.append(int(pt.get("y")))
            except Exception:
                return None
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def _rect_poly(min_x: int, min_y: int, max_x: int, max_y: int) -> list[dict[str, int]]:
        return [
            {"x": min_x, "y": min_y},
            {"x": max_x, "y": min_y},
            {"x": max_x, "y": max_y},
            {"x": min_x, "y": max_y},
        ]

    out_objects: list[dict[str, Any]] = []
    forced = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        row = dict(obj)
        if row.get("source") != "inventory":
            out_objects.append(row)
            continue

        obj_type = row.get("object_type")
        if not isinstance(obj_type, str) or obj_type not in dims_by_type:
            out_objects.append(row)
            continue

        place_on = row.get("place_on") if isinstance(row.get("place_on"), dict) else {}
        method = str(place_on.get("method") or "floor")

        dims_list = dims_by_type[obj_type]
        pair = None
        for L, W, H in dims_list:
            pairs = pairs_for_dims(L, W, H, method=method)
            if pairs:
                pair = pairs[0]
                break
        if pair is None:
            out_objects.append(row)
            continue

        bbox = row.get("bbox")
        if not isinstance(bbox, dict):
            poly = row.get("polygon_ccw") or []
            bbox_vals = _bbox_from_poly(poly)
            if bbox_vals is None:
                out_objects.append(row)
                continue
            min_x, min_y, max_x, max_y = bbox_vals
        else:
            try:
                min_x = int(bbox.get("min_x"))
                min_y = int(bbox.get("min_y"))
                max_x = int(bbox.get("max_x"))
                max_y = int(bbox.get("max_y"))
            except Exception:
                out_objects.append(row)
                continue

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        new_w, new_h = pair
        nmin_x = int(round(cx - new_w / 2.0))
        nmax_x = int(round(cx + new_w / 2.0))
        nmin_y = int(round(cy - new_h / 2.0))
        nmax_y = int(round(cy + new_h / 2.0))

        row["bbox"] = {
            "min_x": nmin_x,
            "min_y": nmin_y,
            "max_x": nmax_x,
            "max_y": nmax_y,
        }
        row["polygon_ccw"] = _rect_poly(nmin_x, nmin_y, nmax_x, nmax_y)
        forced += 1
        out_objects.append(row)

    if forced:
        logger.info("Stylist forced inventory sizes for %s object(s).", forced)
    draft["objects"] = out_objects
    return draft


def _validate_inventory_sizes(
    draft: dict[str, Any],
    inventory_items: list[dict[str, Any]],
    tol_mm: int = 5,
) -> list[str]:
    dims_by_type: dict[str, list[tuple[int, int, int]]] = {}
    dims_by_id: dict[str, tuple[str, int, int, int]] = {}

    for item in inventory_items:
        obj_type = item.get("type")
        dims = item.get("dimensions") or {}
        if not isinstance(obj_type, str) or not isinstance(dims, dict):
            continue

        L = int(dims.get("length_mm") or 0)
        W = int(dims.get("width_mm") or 0)
        H = int(dims.get("height_mm") or 0)
        if L <= 0 or W <= 0:
            continue

        dims_by_type.setdefault(obj_type, []).append((L, W, H))
        inv_id = item.get("id")
        if isinstance(inv_id, str) and inv_id:
            dims_by_id[inv_id] = (obj_type, L, W, H)

    if not dims_by_type:
        return []

    def pairs_for_dims(L: int, W: int, H: int, *, method: str) -> list[tuple[int, int]]:
        # For footprint-like placements, we only accept (length,width).
        # For wall/ceiling/lean placements, a 2D representation can use any two axes.
        if method in {"floor", "on_top"}:
            return [(L, W)]
        pairs: list[tuple[int, int]] = []
        if W > 0 and H > 0:
            pairs.append((W, H))
        if L > 0 and W > 0:
            pairs.append((L, W))
        if L > 0 and H > 0:
            pairs.append((L, H))
        return pairs

    def match_pair(a: int, b: int, x: int, y: int) -> bool:
        return (abs(x - a) <= tol_mm and abs(y - b) <= tol_mm) or (
            abs(x - b) <= tol_mm and abs(y - a) <= tol_mm
        )

    issues: list[str] = []

    for obj in draft.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if obj.get("source") != "inventory":
            continue

        obj_type = obj.get("object_type")
        if not isinstance(obj_type, str) or obj_type not in dims_by_type:
            continue

        w, h = _bbox_dims(obj)
        if w is None or h is None:
            issues.append(f"{obj.get('instance_id') or obj_type}(missing_bbox)")
            continue

        place_on = obj.get("place_on") if isinstance(obj.get("place_on"), dict) else {}
        method = str(place_on.get("method") or "floor")
        instance_id = obj.get("instance_id")

        candidates: list[tuple[int, int, int]] = []
        if isinstance(instance_id, str) and instance_id in dims_by_id:
            _, L0, W0, H0 = dims_by_id[instance_id]
            candidates = [(L0, W0, H0)]
        else:
            candidates = dims_by_type.get(obj_type, [])

        ok = False
        sample_pairs: list[tuple[int, int]] = []
        for L, W, H in candidates:
            pairs = pairs_for_dims(L, W, H, method=method)
            if len(sample_pairs) < 3:
                sample_pairs.extend(pairs[: 3 - len(sample_pairs)])
            if any(match_pair(a, b, w, h) for a, b in pairs):
                ok = True
                break

        if not ok:
            name = str(instance_id or obj_type)
            issues.append(
                f"{name}({w}x{h} not matching inventory dims, sample={sample_pairs})"
            )

    return issues


def _bbox_dims(obj: dict[str, Any]) -> tuple[int | None, int | None]:
    bbox = obj.get("bbox")
    if not isinstance(bbox, dict):
        return None, None

    try:
        min_x = int(bbox.get("min_x"))
        min_y = int(bbox.get("min_y"))
        max_x = int(bbox.get("max_x"))
        max_y = int(bbox.get("max_y"))
    except Exception:
        return None, None

    return max_x - min_x, max_y - min_y


def _stable_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(_extract_json_object(_coerce_json_text(text)))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _fix_place_on_targets(payload: dict[str, Any], *, layout_json: dict[str, Any]) -> dict[str, Any]:
    """
    Stylist sometimes returns place_on.target_instance_id=null even though the schema requires a string.
    We repair this deterministically to keep the pipeline stable.
    """
    if not isinstance(payload, dict):
        return payload

    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        return payload

    supports: list[tuple[str, int, int, int, int, int]] = []  # (id, minx, miny, maxx, maxy, area)
    for o in (layout_json.get("objects") or []):
        if not isinstance(o, dict):
            continue
        inst = o.get("instance_id")
        bbox = o.get("bbox")
        if not isinstance(inst, str) or not inst or not isinstance(bbox, dict):
            continue
        try:
            min_x = int(bbox.get("min_x"))
            min_y = int(bbox.get("min_y"))
            max_x = int(bbox.get("max_x"))
            max_y = int(bbox.get("max_y"))
        except Exception:
            continue
        area = max(0, max_x - min_x) * max(0, max_y - min_y)
        supports.append((inst, min_x, min_y, max_x, max_y, area))

    # Prefer smallest container when inferring "on_top" supports.
    supports.sort(key=lambda x: x[5])

    def contains(a: tuple[str, int, int, int, int, int], b: dict[str, Any], *, tol: int = 5) -> bool:
        w, h = _bbox_dims(b)
        bbox = b.get("bbox")
        if w is None or h is None or not isinstance(bbox, dict):
            return False
        try:
            bmin_x = int(bbox.get("min_x"))
            bmin_y = int(bbox.get("min_y"))
            bmax_x = int(bbox.get("max_x"))
            bmax_y = int(bbox.get("max_y"))
        except Exception:
            return False
        _, amin_x, amin_y, amax_x, amax_y, _ = a
        return (
            bmin_x >= amin_x - tol
            and bmin_y >= amin_y - tol
            and bmax_x <= amax_x + tol
            and bmax_y <= amax_y + tol
        )

    def is_ceiling_type(obj_type: str) -> bool:
        return obj_type in {"ceiling_light", "pendant_light", "track_light", "projector"}

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        place_on = obj.get("place_on")
        if not isinstance(place_on, dict):
            continue
        target = place_on.get("target_instance_id")
        if isinstance(target, str) and target:
            continue

        method = str(place_on.get("method") or "")
        obj_type = str(obj.get("object_type") or "")

        if method in {"hang_on", "lean_on"}:
            place_on["target_instance_id"] = "ceiling" if is_ceiling_type(obj_type) else "wall"
            continue

        if method == "floor":
            place_on["target_instance_id"] = "floor"
            continue

        if method == "on_top":
            inferred = None
            for s in supports:
                if contains(s, obj):
                    inferred = s[0]
                    break
            if inferred:
                place_on["target_instance_id"] = inferred
            else:
                # If we can't infer a support, downgrade to a floor object for schema + renderer safety.
                place_on["method"] = "floor"
                place_on["target_instance_id"] = "floor"

    payload["objects"] = objects
    return payload
