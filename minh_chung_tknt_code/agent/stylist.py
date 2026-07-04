from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent_schema.stylist_schema import StylistOutput
from clients.llm_client import get_llm_client
from prompt.stylist import STYLIST_PROMPT
from prompt.system import SYSTEM_PROMPT
from stylist.deterministic_layout import (
    DEFAULT_OBJECT_COLOR,
    DEFAULT_SURFACES,
    build_deterministic_stylist_payload,
)
from stylist.style_policy import (
    compile_style_policy as compile_layout_style_policy,
)
from stylist.style_policy import (
    final_style_plan_from_surface_payload,
)

logger = logging.getLogger(__name__)

try:
    from config.llm_config import TextLLMConfig
except Exception:  # pragma: no cover
    TextLLMConfig = None  # type: ignore[assignment]


def stylist_model_name_for_variant(variant_index: int) -> str:
    _ = variant_index
    strict_model_name = _strict_single_model_name()
    if strict_model_name:
        return strict_model_name
    configured_model_name = _configured_agent_model_name("stylist")
    if configured_model_name:
        return configured_model_name
    if TextLLMConfig is not None:
        return TextLLMConfig.primary_model_name()
    return ""


@dataclass(frozen=True)
class Stylist:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = STYLIST_PROMPT

    def compile_style_policy(
        self,
        *,
        room_type: str,
        brief_text: str,
        room_model_json: dict[str, Any] | None = None,
        semantic_program_rules: dict[str, Any] | None = None,
        use_llm: bool = True,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        return compile_layout_style_policy(
            room_type=room_type,
            brief_text=brief_text,
            room_model_json=room_model_json,
            semantic_program_rules=semantic_program_rules,
            use_llm=use_llm,
            model_name=model_name,
        )

    def generate_style_plan(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        _, payload, _, style_context = self._prepare_style_inputs(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        style_prompt = _build_color_prompt(
            template=self.prompt_template,
            style_context_json=style_context,
            locked_style_payload_json=_build_locked_style_payload(payload),
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": style_prompt},
        ]
        return _generate_style_plan(messages, model_name=model_name)

    def generate_raw(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> str:
        payload = self._build_output_payload(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def generate(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        return self.apply_style_plan(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
            model_name=model_name,
        )

    def apply_style_plan(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        style_plan: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        normalized_layout, payload, _, style_context = self._prepare_style_inputs(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        resolved_style_plan = (
            deepcopy(style_plan)
            if isinstance(style_plan, dict)
            else self.generate_style_plan(
                layout_json=normalized_layout,
                user_context_json=user_context_json,
                tenant_id=tenant_id,
                model_name=model_name,
            )
        )
        styled_payload = _apply_style_plan(
            payload=payload,
            style_plan=resolved_style_plan,
            style_context=style_context,
        )
        styled_payload = _fix_place_on_targets(
            styled_payload,
            layout_json=normalized_layout,
        )
        return StylistOutput.model_validate(styled_payload).model_dump()

    def _build_output_payload(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None,
        tenant_id: str | None,
    ) -> dict[str, Any]:
        normalized_layout, payload, _, style_context = self._prepare_style_inputs(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        style_prompt = _build_color_prompt(
            template=self.prompt_template,
            style_context_json=style_context,
            locked_style_payload_json=_build_locked_style_payload(payload),
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": style_prompt},
        ]
        style_plan = _generate_style_plan(messages)
        styled_payload = _apply_style_plan(
            payload=payload,
            style_plan=style_plan,
            style_context=style_context,
        )
        return styled_payload

    def _prepare_style_inputs(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None,
        tenant_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        normalized_layout = _normalize_layout_json(layout_json)
        merged_user_context = _inject_tenant(user_context_json, tenant_id)
        resolved_tenant_id = _extract_tenant_id(merged_user_context, normalized_layout)
        payload = build_deterministic_stylist_payload(
            normalized_layout,
            tenant_id=resolved_tenant_id,
        )
        style_context = _build_style_context(
            layout_json=normalized_layout,
            user_context_json=merged_user_context,
            payload=payload,
        )
        return normalized_layout, payload, merged_user_context, style_context


def _build_color_prompt(
    *,
    template: str,
    style_context_json: dict[str, Any],
    locked_style_payload_json: dict[str, Any],
) -> str:
    mapping = {
        "STYLE_CONTEXT_JSON": _json_block(style_context_json),
        "LOCKED_STYLE_PAYLOAD_JSON": _json_block(locked_style_payload_json),
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


def _build_style_context(
    *,
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any] | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    room = layout_json.get("room") if isinstance(layout_json.get("room"), dict) else {}
    notes = _extract_guidance_notes(user_context_json)
    payload_notes = payload.get("notes")
    if isinstance(payload_notes, list):
        for note in payload_notes:
            if isinstance(note, str) and note.strip() and note.strip() not in notes:
                notes.append(note.strip())

    objects_summary: list[dict[str, Any]] = []
    for row in payload.get("objects") or []:
        if not isinstance(row, dict):
            continue
        place_on = (
            row.get("place_on") if isinstance(row.get("place_on"), dict) else None
        )
        objects_summary.append(
            {
                "instance_id": row.get("instance_id"),
                "object_type": row.get("object_type"),
                "source": row.get("source"),
                "cluster_id": row.get("cluster_id"),
                "place_on": place_on,
            }
        )

    room_info = payload.get("room") if isinstance(payload.get("room"), dict) else {}
    openings = (
        room_info.get("openings") if isinstance(room_info.get("openings"), dict) else {}
    )
    return {
        "room_type": _extract_room_type(layout_json),
        "requested_style": _extract_requested_style(user_context_json, layout_json),
        "style_tags": _extract_style_tags(user_context_json, layout_json),
        "style_policy": _extract_style_policy(user_context_json, layout_json),
        "notes": notes[:8],
        "room_id": room_info.get("room_id") or room.get("room_id") or "room_1",
        "door_ids": [
            row.get("id")
            for row in openings.get("doors", [])
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        ],
        "window_ids": [
            row.get("id")
            for row in openings.get("windows", [])
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        ],
        "objects": objects_summary,
    }


def _extract_guidance_notes(
    user_context_json: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(user_context_json, dict):
        return []

    notes: list[str] = []

    def _add(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if text and text not in notes:
            notes.append(text)

    for key in ("guidance_text", "notes_text"):
        _add(user_context_json.get(key))

    for key in ("notes", "room_notes"):
        value = user_context_json.get(key)
        if isinstance(value, list):
            for item in value:
                _add(item)

    user_input = user_context_json.get("user_input")
    if isinstance(user_input, dict):
        for key in (
            "description",
            "special_description",
            "special_notes",
            "notes",
            "feng_shui",
        ):
            _add(user_input.get(key))

    return notes


def _extract_requested_style(
    user_context_json: dict[str, Any] | None,
    layout_json: dict[str, Any],
) -> str:
    if isinstance(user_context_json, dict):
        style = user_context_json.get("style")
        if isinstance(style, str) and style.strip():
            return style.strip()
        user_input = user_context_json.get("user_input")
        if isinstance(user_input, dict):
            style = user_input.get("style")
            if isinstance(style, str) and style.strip():
                return style.strip()

    room = layout_json.get("room") if isinstance(layout_json.get("room"), dict) else {}
    style = room.get("style")
    if isinstance(style, str) and style.strip():
        return style.strip()

    meta = layout_json.get("meta")
    if isinstance(meta, dict):
        style = meta.get("style")
        if isinstance(style, str) and style.strip():
            return style.strip()

    return ""


def _extract_style_policy(
    user_context_json: dict[str, Any] | None,
    layout_json: dict[str, Any],
) -> dict[str, Any] | None:
    if isinstance(user_context_json, dict):
        style_policy = user_context_json.get("style_policy")
        if isinstance(style_policy, dict):
            return deepcopy(style_policy)
        user_input = user_context_json.get("user_input")
        if isinstance(user_input, dict):
            style_policy = user_input.get("style_policy")
            if isinstance(style_policy, dict):
                return deepcopy(style_policy)
    style_policy = layout_json.get("style_policy")
    if isinstance(style_policy, dict):
        return deepcopy(style_policy)
    meta = layout_json.get("meta")
    if isinstance(meta, dict):
        style_policy = meta.get("style_policy")
        if isinstance(style_policy, dict):
            return deepcopy(style_policy)
    return None


def _build_locked_style_payload(payload: dict[str, Any]) -> dict[str, Any]:
    room = payload.get("room") if isinstance(payload.get("room"), dict) else {}
    objects_out: list[dict[str, Any]] = []
    for row in payload.get("objects") or []:
        if not isinstance(row, dict):
            continue
        objects_out.append(
            {
                "instance_id": row.get("instance_id"),
                "object_type": row.get("object_type"),
                "source": row.get("source"),
                "cluster_id": row.get("cluster_id"),
                "place_on": row.get("place_on"),
            }
        )

    return {
        "room": {
            "room_id": room.get("room_id"),
            "room_type": room.get("room_type"),
            "opening_ids": {
                "doors": [
                    item.get("id")
                    for item in (room.get("openings") or {}).get("doors", [])
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ],
                "windows": [
                    item.get("id")
                    for item in (room.get("openings") or {}).get("windows", [])
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ],
            },
        },
        "objects": objects_out,
    }


def _generate_style_plan(
    messages: list[dict[str, str]],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    client = get_llm_client()
    attempted_models: list[str] = []
    last_retryable_error: Exception | None = None

    for candidate_model_name in _style_plan_model_attempt_order(model_name):
        attempted_models.append(candidate_model_name)
        candidate_messages = list(messages)
        try:
            for attempt in range(2):
                response = client.chat_completion(
                    candidate_messages,
                    model_key="primary",
                    model_name=candidate_model_name,
                    fallback_model_names=(),
                    temperature=0.35,
                    max_tokens=2200,
                    thinking_level="medium",
                    response_mime_type="application/json",
                )
                content = _extract_content_from_response(response)
                parsed = _try_parse_json_object(content)
                if parsed is not None and _is_style_plan_shape(parsed):
                    return parsed
                last_retryable_error = ValueError(
                    "Stylist returned invalid JSON: "
                    f"{_truncate_text(content, max_len=300)}"
                )
                if attempt == 1:
                    logger.warning(
                        "Stylist model %s returned invalid JSON after repair.",
                        candidate_model_name,
                    )
                    break
                _record_llm_retry(
                    stage="stylist.style_plan",
                    model_name=candidate_model_name,
                    reason="invalid_json_or_schema",
                )
                candidate_messages.append({"role": "assistant", "content": content})
                candidate_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return exactly one valid JSON object matching the style "
                            "plan shape. No markdown. No prose outside JSON."
                        ),
                    }
                )
        except Exception as exc:
            if _is_retryable_model_unavailable_error(exc):
                last_retryable_error = exc
                logger.warning(
                    "Stylist model %s is temporarily unavailable; moving to the next configured model if available. Error: %s",
                    candidate_model_name,
                    exc,
                )
                continue
            logger.warning("Stylist color planner fallback triggered: %s", exc)
            return {}

    if last_retryable_error is not None:
        logger.warning(
            "Stylist color planner fallback triggered after exhausting failover models %s: %s",
            attempted_models,
            last_retryable_error,
        )
    return {}


def _style_plan_model_attempt_order(model_name: str | None) -> list[str]:
    strict_model_name = _strict_single_model_name()
    if strict_model_name:
        return [strict_model_name]
    configured_model_name = _configured_agent_model_name("stylist")
    requested_model_name = str(model_name or configured_model_name or "").strip()
    if not requested_model_name:
        primary_model_name = (
            TextLLMConfig.primary_model_name() if TextLLMConfig is not None else ""
        )
        return [primary_model_name] if primary_model_name else []
    return [requested_model_name]


def _strict_single_model_name() -> str:
    if TextLLMConfig is None:
        return ""
    return str(getattr(TextLLMConfig, "STRICT_SINGLE_TEXT_MODEL", "") or "").strip()


def _configured_agent_model_name(key: str) -> str | None:
    if TextLLMConfig is None:
        return None
    return TextLLMConfig.agent_model(key)


def _record_llm_retry(*, stage: str, model_name: str | None, reason: str) -> None:
    _ = (stage, model_name, reason)


def _is_retryable_model_unavailable_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    if not error_text:
        return False
    return (
        "503 service unavailable" in error_text
        or '"code": 503' in error_text
        or '"status": "unavailable"' in error_text
        or "currently experiencing high demand" in error_text
    )


def _apply_style_plan(
    *,
    payload: dict[str, Any],
    style_plan: dict[str, Any],
    style_context: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(payload)
    palette = _fallback_palette(style_context)

    room = out.get("room")
    if isinstance(room, dict):
        room["surfaces"] = {
            "wall_color_hex": palette["wall"],
            "floor_color_hex": palette["floor"],
            "ceiling_color_hex": palette["ceiling"],
        }
        opening_colors = room.get("opening_colors")
        if not isinstance(opening_colors, dict):
            opening_colors = {"doors": [], "windows": []}
            room["opening_colors"] = opening_colors
        for item in opening_colors.get("doors", []):
            if isinstance(item, dict):
                item["color_hex"] = palette["trim"]
        for item in opening_colors.get("windows", []):
            if isinstance(item, dict):
                item["color_hex"] = palette["glass"]

    for row in out.get("objects") or []:
        if not isinstance(row, dict):
            continue
        object_type = str(row.get("object_type") or "")
        instance_id = str(row.get("instance_id") or object_type)
        row["color_hex"] = _sanitize_object_color(
            object_type=object_type,
            instance_id=instance_id,
            palette=palette,
            color_hex=_fallback_object_color(
                object_type=object_type,
                instance_id=instance_id,
                palette=palette,
            ),
        )
        material = row.get("material")
        if not isinstance(material, str) or not material.strip():
            row["material"] = _fallback_material_name(object_type)

    if isinstance(style_plan, dict):
        room_surfaces = style_plan.get("room_surfaces")
        if isinstance(room_surfaces, dict) and isinstance(room, dict):
            surfaces = room.get("surfaces")
            if not isinstance(surfaces, dict):
                surfaces = dict(DEFAULT_SURFACES)
                room["surfaces"] = surfaces
            for key in ("wall_color_hex", "floor_color_hex", "ceiling_color_hex"):
                value = room_surfaces.get(key)
                if _is_hex_color(value):
                    surfaces[key] = value

        opening_colors_plan = style_plan.get("opening_colors")
        if isinstance(opening_colors_plan, dict) and isinstance(room, dict):
            opening_colors = room.get("opening_colors")
            if not isinstance(opening_colors, dict):
                opening_colors = {"doors": [], "windows": []}
                room["opening_colors"] = opening_colors
            for key in ("doors", "windows"):
                plan_rows = opening_colors_plan.get(key)
                output_rows = opening_colors.get(key)
                if not isinstance(plan_rows, list) or not isinstance(output_rows, list):
                    continue
                by_id = {
                    item.get("id"): item
                    for item in output_rows
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                }
                for plan_row in plan_rows:
                    if not isinstance(plan_row, dict):
                        continue
                    opening_id = plan_row.get("id")
                    color_hex = plan_row.get("color_hex")
                    if (
                        isinstance(opening_id, str)
                        and opening_id in by_id
                        and _is_hex_color(color_hex)
                    ):
                        by_id[opening_id]["color_hex"] = color_hex

        object_styles = style_plan.get("object_styles")
        if isinstance(object_styles, list):
            style_by_id = {
                row.get("instance_id"): row
                for row in object_styles
                if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
            }
            for row in out.get("objects") or []:
                if not isinstance(row, dict):
                    continue
                object_type = str(row.get("object_type") or "")
                instance_id = row.get("instance_id")
                if not isinstance(instance_id, str):
                    continue
                planned = style_by_id.get(instance_id)
                if not isinstance(planned, dict):
                    continue
                color_hex = planned.get("color_hex")
                if _is_hex_color(color_hex):
                    row["color_hex"] = _sanitize_object_color(
                        object_type=object_type,
                        instance_id=instance_id,
                        palette=palette,
                        color_hex=color_hex,
                    )
                material = planned.get("material")
                if material is None:
                    row["material"] = None
                elif isinstance(material, str) and material.strip():
                    row["material"] = material.strip()

        combined_notes = list(out.get("notes") or [])
        style_notes = style_plan.get("notes")
        if not isinstance(style_notes, list):
            style_notes = []
        for note in style_notes:
            if (
                isinstance(note, str)
                and note.strip()
                and note.strip() not in combined_notes
            ):
                combined_notes.append(note.strip())
        out["notes"] = combined_notes[:6]

    style_policy = style_context.get("style_policy")
    out["final_style_plan"] = final_style_plan_from_surface_payload(
        style_policy=style_policy if isinstance(style_policy, dict) else None,
        style_plan=style_plan if isinstance(style_plan, dict) else {},
        styled_payload=out,
    )
    return out


def _fallback_palette(style_context: dict[str, Any]) -> dict[str, str]:
    style_text = " ".join(
        [
            str(style_context.get("requested_style") or ""),
            " ".join(
                item
                for item in style_context.get("style_tags", [])
                if isinstance(item, str)
            ),
            " ".join(
                item for item in style_context.get("notes", []) if isinstance(item, str)
            ),
        ]
    ).lower()

    if "japandi" in style_text:
        return {
            "wall": "#F1E8DC",
            "floor": "#B98A61",
            "ceiling": "#FBF5EC",
            "trim": "#8F7259",
            "glass": "#DDE7F0",
            "large": "#B79A7C",
            "soft": "#DCC9B6",
            "accent": "#A46D3F",
            "accent_alt": "#7E8A70",
            "metal": "#5B5651",
            "dark": "#4B4038",
        }
    if "industrial" in style_text:
        return {
            "wall": "#E1DDD7",
            "floor": "#8B6B4F",
            "ceiling": "#F4F1EC",
            "trim": "#4D4D4D",
            "glass": "#C6D4DE",
            "large": "#6D726E",
            "soft": "#B8ADA0",
            "accent": "#B35C3B",
            "accent_alt": "#6C7A7A",
            "metal": "#4D4D4D",
            "dark": "#2E2E2E",
        }
    if "boho" in style_text:
        return {
            "wall": "#F5E6D4",
            "floor": "#B57E50",
            "ceiling": "#FBF5EE",
            "trim": "#99613D",
            "glass": "#D7E4EA",
            "large": "#B48A68",
            "soft": "#D6B79D",
            "accent": "#C46849",
            "accent_alt": "#7B8F65",
            "metal": "#7D665A",
            "dark": "#5B4336",
        }
    if "minimal" in style_text or "scandinav" in style_text:
        return {
            "wall": "#F6F2EC",
            "floor": "#C6A47A",
            "ceiling": "#FCFAF6",
            "trim": "#8D8C84",
            "glass": "#E1EAF0",
            "large": "#C2B8AA",
            "soft": "#DDD6CB",
            "accent": "#8F9C9A",
            "accent_alt": "#BCA08A",
            "metal": "#76736E",
            "dark": "#55514C",
        }
    return {
        "wall": "#F2ECE4",
        "floor": "#BC926A",
        "ceiling": "#FBF7F2",
        "trim": "#8A715B",
        "glass": "#D9E3EC",
        "large": "#B9A690",
        "soft": "#D7C8B8",
        "accent": "#8F7A6A",
        "accent_alt": "#7F8E86",
        "metal": "#6F6962",
        "dark": "#4F463F",
    }


def _fallback_object_color(
    *,
    object_type: str,
    instance_id: str,
    palette: dict[str, str],
) -> str:
    lowered = object_type.lower()
    accent = _pick_palette_color(
        [palette["accent"], palette["accent_alt"], palette["soft"]],
        seed=instance_id,
    )
    if any(token in lowered for token in ("rug", "curtain", "blanket", "cushion")):
        return accent
    if "plant" in lowered:
        return palette["accent_alt"]
    if any(
        token in lowered
        for token in (
            "lamp",
            "speaker",
            "clock",
            "desktop",
            "printer",
            "monitor",
            "tv",
            "air_conditioner",
            "appliance",
            "cooktop",
            "oven",
            "microwave",
            "toaster",
            "blender",
            "coffee_machine",
        )
    ):
        return palette["metal"]
    if any(token in lowered for token in ("mirror", "window", "screen")):
        return palette["glass"]
    if any(token in lowered for token in ("decor", "vase")):
        return accent
    if any(
        token in lowered
        for token in (
            "chair",
            "armchair",
            "sofa",
            "bed",
            "bench",
            "bean_bag",
            "ottoman",
        )
    ):
        return _pick_palette_color(
            [palette["large"], palette["soft"], accent],
            seed=instance_id,
        )
    if any(
        token in lowered
        for token in (
            "desk",
            "table",
            "dresser",
            "wardrobe",
            "cabinet",
            "bookshelf",
            "console",
            "nightstand",
            "storage",
        )
    ):
        return _pick_palette_color(
            [palette["large"], palette["dark"]],
            seed=instance_id,
        )
    return DEFAULT_OBJECT_COLOR


def _pick_palette_color(options: list[str], *, seed: str) -> str:
    if not options:
        return DEFAULT_OBJECT_COLOR
    index = sum(ord(ch) for ch in seed) % len(options)
    return options[index]


def _sanitize_object_color(
    *,
    object_type: str,
    instance_id: str,
    palette: dict[str, str],
    color_hex: Any,
) -> str:
    if _is_hex_color(color_hex) and not _is_near_black_hex(color_hex):
        return color_hex
    return _fallback_object_color(
        object_type=object_type,
        instance_id=instance_id,
        palette=palette,
    )


def _is_hex_color(value: Any) -> bool:
    return isinstance(value, str) and bool(re.match(r"^#[0-9A-Fa-f]{6}$", value))


def _is_near_black_hex(value: str) -> bool:
    if not _is_hex_color(value):
        return False
    red = int(value[1:3], 16)
    green = int(value[3:5], 16)
    blue = int(value[5:7], 16)
    return max(red, green, blue) <= 36 or (red + green + blue) <= 120


def _extract_content_from_response(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("Stylist color planner response missing message content")


def _fallback_material_name(object_type: str) -> str:
    lowered = object_type.lower()
    if any(
        token in lowered
        for token in (
            "rug",
            "curtain",
            "blanket",
            "cushion",
            "bed",
            "sofa",
            "chair",
            "armchair",
            "bean_bag",
        )
    ):
        return "fabric"
    if any(
        token in lowered
        for token in (
            "lamp",
            "clock",
            "speaker",
            "air_conditioner",
            "monitor",
            "laptop",
            "desktop",
            "printer",
            "appliance",
            "microwave",
            "oven",
            "toaster",
            "blender",
            "hood",
            "cooktop",
        )
    ):
        return "metal"
    if any(token in lowered for token in ("mirror", "window", "glass")):
        return "glass"
    if any(token in lowered for token in ("vase", "plant", "decor")):
        return "ceramic"
    return "wood"


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


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    for candidate in _json_object_parse_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _is_style_plan_shape(payload: dict[str, Any]) -> bool:
    room_surfaces = payload.get("room_surfaces")
    opening_colors = payload.get("opening_colors")
    object_styles = payload.get("object_styles")
    return (
        isinstance(room_surfaces, dict)
        and isinstance(opening_colors, dict)
        and isinstance(object_styles, list)
    )


def _json_object_parse_candidates(text: str) -> list[str]:
    coerced = _coerce_json_text(text).strip()
    if not coerced:
        return []
    raw_candidate = _extract_outer_json_slice(coerced).strip()
    candidates = [
        coerced,
        raw_candidate,
        _strip_trailing_json_commas(raw_candidate),
        _close_unterminated_json(_strip_trailing_json_commas(raw_candidate)),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _extract_outer_json_slice(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            depth += 1
            continue
        if char in "}]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _strip_trailing_json_commas(text: str) -> str:
    previous = text
    while True:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", previous)
        if cleaned == previous:
            return cleaned
        previous = cleaned


def _close_unterminated_json(text: str) -> str:
    if not text:
        return text

    closers: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            closers.append("}")
        elif char == "[":
            closers.append("]")
        elif char in "}]":
            if closers and closers[-1] == char:
                closers.pop()
            else:
                return text
    if in_string:
        return text
    return text + "".join(reversed(closers))


def _fix_place_on_targets(
    payload: dict[str, Any], *, layout_json: dict[str, Any]
) -> dict[str, Any]:
    """
    Stylist sometimes returns place_on.target_instance_id=null even though the schema requires a string.
    We repair this deterministically to keep the pipeline stable.
    """
    if not isinstance(payload, dict):
        return payload

    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        return payload

    supports: list[
        tuple[str, int, int, int, int, int]
    ] = []  # (id, minx, miny, maxx, maxy, area)
    for o in layout_json.get("objects") or []:
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

    def contains(
        a: tuple[str, int, int, int, int, int], b: dict[str, Any], *, tol: int = 5
    ) -> bool:
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
        return obj_type in {
            "ceiling_light",
            "pendant_light",
            "track_light",
            "projector",
        }

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
            place_on["target_instance_id"] = (
                "ceiling" if is_ceiling_type(obj_type) else "wall"
            )
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
