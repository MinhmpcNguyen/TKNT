from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agent_schema.clusterF_schema import ClusterForgeOutput
from clients.base_client import ChatMessage
from clients.llm_client import get_llm_client
from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM, normalize_cluster_rules_grid
from prompt.cluster_forge import CLUSTER_FORGE_PROMPT
from prompt.system import SYSTEM_PROMPT
from stylist.room_essentials_seed import ROOM_SURFACE_GROUPS


@dataclass(frozen=True)
class ClusterForge:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = CLUSTER_FORGE_PROMPT

    def build_input_json(
        self,
        room_type: str,
        room_surface_groups: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    ) -> list[dict[str, object]]:
        groups = (
            ROOM_SURFACE_GROUPS if room_surface_groups is None else room_surface_groups
        )
        room_group = groups.get(room_type, {})
        no_stack = list(room_group.get("no_stack", []))
        objects = [{"object_id": name, "name": name} for name in no_stack]
        return [
            {
                "room_type": room_type,
                "no_stack": no_stack,
                "objects": objects,
            }
        ]

    def build_messages(
        self,
        room_type: str,
        room_surface_groups: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        description: str | None = None,
        special_notes: str | None = None,
    ) -> list[ChatMessage]:
        input_json = self.build_input_json(room_type, room_surface_groups)
        payload = json.dumps(input_json, ensure_ascii=True)
        description_block = ""
        if description:
            description_block = (
                f"USER DESCRIPTION (free text):\n{description.strip()}\n\n"
            )
        special_notes_block = ""
        if special_notes:
            special_notes_block = (
                f"USER SPECIAL NOTES (free text):\n{special_notes.strip()}\n\n"
            )
        user_prompt = (
            self.prompt_template.replace("{INPUT_JSON}", payload)
            .replace("{DESCRIPTION_BLOCK}", description_block)
            .replace("{SPECIAL_NOTES_BLOCK}", special_notes_block)
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_raw(
        self,
        room_type: str,
        room_surface_groups: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        description: str | None = None,
        special_notes: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        messages = self.build_messages(
            room_type,
            room_surface_groups,
            description=description,
            special_notes=special_notes,
        )
        response = get_llm_client().chat_completion(
            messages,
            model_key="primary",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_content(response)

    def generate(
        self,
        room_type: str,
        room_surface_groups: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        description: str | None = None,
        special_notes: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ClusterForgeOutput:
        raw = self.generate_raw(
            room_type,
            room_surface_groups,
            description=description,
            special_notes=special_notes,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        payload = _parse_json(raw)
        payload = _normalize_cluster_forge_payload(payload)
        return ClusterForgeOutput.model_validate(payload)


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
            raise ValueError("ClusterForge returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("ClusterForge response must be a JSON object")
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


def _normalize_cluster_forge_payload(payload: dict[str, object]) -> dict[str, object]:
    if not isinstance(payload, dict):
        return payload
    payload["notes"] = _coerce_str_list(payload.get("notes"))
    payload["missing"] = _coerce_str_list(payload.get("missing"))

    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        payload["clusters"] = []
        return payload

    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        cluster["cluster_rules"] = normalize_cluster_rules_grid(
            cluster.get("cluster_rules")
        )
        cluster["notes"] = _coerce_str_list(cluster.get("notes"))
        cluster["members"] = _coerce_str_list(cluster.get("members"))
        cluster["anchors"] = _coerce_str_list(cluster.get("anchors"))
        hard = cluster.get("hard_constraints")
        if not isinstance(hard, list):
            cluster["hard_constraints"] = []
        else:
            normalized_hard: list[dict[str, object]] = []
            for constraint in hard:
                normalized_constraint = _normalize_hard_constraint(constraint)
                if normalized_constraint is None:
                    _append_payload_note(
                        payload,
                        f"Dropped unsupported hard constraint: {constraint!r}",
                    )
                    continue
                normalized_hard.append(normalized_constraint)
            cluster["hard_constraints"] = normalized_hard
        soft = cluster.get("soft_constraints")
        if not isinstance(soft, list):
            cluster["soft_constraints"] = []
            continue
        normalized_soft: list[dict[str, object]] = []
        for constraint in soft:
            normalized_constraint = _normalize_soft_constraint(constraint)
            if normalized_constraint is None:
                _append_payload_note(
                    payload,
                    f"Dropped unsupported soft constraint: {constraint!r}",
                )
                continue
            normalized_soft.append(normalized_constraint)
        cluster["soft_constraints"] = normalized_soft
        _sanitize_cluster_constraints(
            payload=payload,
            cluster_id=cluster_id if isinstance(cluster_id, str) else "",
            cluster=cluster,
        )
    return payload


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _normalize_soft_constraint(constraint: object) -> dict[str, object] | None:
    if not isinstance(constraint, dict):
        return None

    normalized_constraint = dict(constraint)
    ctype = normalized_constraint.get("type")
    if not isinstance(ctype, str):
        return None

    normalized_type = ctype.strip().lower()
    normalized_constraint["type"] = normalized_type
    if normalized_type == "prefer_facing":
        mode = _normalize_prefer_facing_mode(normalized_constraint.get("mode"))
        if mode is None:
            return None
        normalized_constraint["mode"] = mode
        normalized_constraint["weight"] = _normalize_constraint_weight(
            normalized_constraint.get("weight")
        )
        return normalized_constraint

    if normalized_type == "prefer_near":
        normalized_constraint["weight"] = _normalize_constraint_weight(
            normalized_constraint.get("weight")
        )
        return normalized_constraint

    if normalized_type == "prefer_align_edge":
        edge = _normalize_align_edge_token(normalized_constraint.get("edge"))
        if edge is None:
            return None
        normalized_constraint["edge"] = edge
        normalized_constraint["weight"] = _normalize_constraint_weight(
            normalized_constraint.get("weight")
        )
        return normalized_constraint

    return None


def _normalize_hard_constraint(constraint: object) -> dict[str, object] | None:
    if not isinstance(constraint, dict):
        return None

    normalized_constraint = dict(constraint)
    ctype = normalized_constraint.get("type")
    if not isinstance(ctype, str):
        return None

    normalized_type = ctype.strip().lower()
    normalized_constraint["type"] = normalized_type

    if normalized_type in {"no_overlap", "contain_in"}:
        return normalized_constraint

    if normalized_type == "anchor_side":
        side = _normalize_anchor_side_token(normalized_constraint.get("side"))
        if side is None:
            return None
        gap_min, gap_max = _normalize_gap_range(
            normalized_constraint.get("gap_min"),
            normalized_constraint.get("gap_max"),
        )
        normalized_constraint["side"] = side
        normalized_constraint["gap_min"] = gap_min
        normalized_constraint["gap_max"] = gap_max
        return normalized_constraint

    if normalized_type == "dock_to_edge":
        b_edge = _normalize_dock_edge_token(normalized_constraint.get("b_edge"))
        if b_edge is None:
            return None
        span = _normalize_dock_span_token(normalized_constraint.get("span"))
        gap_min, gap_max = _normalize_gap_range(
            normalized_constraint.get("gap_min"),
            normalized_constraint.get("gap_max"),
        )
        normalized_constraint["b_edge"] = b_edge
        normalized_constraint["span"] = span
        normalized_constraint["gap_min"] = gap_min
        normalized_constraint["gap_max"] = gap_max
        return normalized_constraint

    if normalized_type == "requires_access":
        obj_id = normalized_constraint.get("id")
        if not isinstance(obj_id, str) or not obj_id.strip():
            fallback_id = normalized_constraint.get("a")
            if isinstance(fallback_id, str) and fallback_id.strip():
                normalized_constraint["id"] = fallback_id.strip()
        mode = normalized_constraint.get("mode")
        if not isinstance(mode, str) or not mode.strip():
            normalized_constraint["mode"] = "front_clearance"
        return normalized_constraint

    return None


def _normalize_gap_range(gap_min: object, gap_max: object) -> tuple[int, int]:
    min_value = _coerce_int(gap_min, default=0)
    max_value = _coerce_int(gap_max, default=min_value)
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    return min_value, max_value


def _normalize_constraint_weight(value: object) -> int:
    weight = _coerce_int(value, default=5)
    return max(1, weight)


def _normalize_anchor_side_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    mapping = {
        "front": "head",
        "back": "foot",
        "front_left": "head_left",
        "front_right": "head_right",
        "back_left": "foot_left",
        "back_right": "foot_right",
    }
    token = mapping.get(token, token)
    if token in {
        "head_left",
        "head_right",
        "foot_left",
        "foot_right",
        "head",
        "foot",
        "left",
        "right",
        "top",
        "bottom",
    }:
        return token
    return None


def _normalize_dock_edge_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    mapping = {
        "head": "front",
        "foot": "back",
        "top": "top",
        "bottom": "bottom",
        "front_left": "left",
        "front_right": "right",
        "back_left": "left",
        "back_right": "right",
        "head_left": "left",
        "head_right": "right",
        "foot_left": "left",
        "foot_right": "right",
    }
    token = mapping.get(token, token)
    if token in {"front", "back", "left", "right", "top", "bottom"}:
        return token
    return None


def _normalize_dock_span_token(value: object) -> str:
    if not isinstance(value, str):
        return "any"
    token = value.strip().lower()
    if token in {"any", "center", "left", "right", "short_edge", "long_edge"}:
        return token
    if token in {"middle", "mid"}:
        return "center"
    return "any"


def _normalize_align_edge_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    mapping = {
        "front_left": "left",
        "front_right": "right",
        "back_left": "left",
        "back_right": "right",
        "head_left": "left",
        "head_right": "right",
        "foot_left": "left",
        "foot_right": "right",
    }
    token = mapping.get(token, token)
    if token in {"left", "right", "top", "bottom", "front", "back", "head", "foot"}:
        return token
    return None


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(round(float(value.strip())))
        except ValueError:
            return default
    return default


def _normalize_prefer_facing_mode(value: object) -> str | None:
    if value is None:
        return "face_same_direction"
    if not isinstance(value, str):
        return None

    normalized_value = value.strip().lower()
    if normalized_value in {
        "face_each_other",
        "face_eachother",
        "each_other",
        "towards",
        "face_each",
    }:
        return "face_each_other"
    if normalized_value in {
        "face_same_direction",
        "same_direction",
        "same_dir",
        "aligned",
        "same",
    }:
        return "face_same_direction"
    return None


def _append_payload_note(payload: dict[str, object], note: str) -> None:
    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
        payload["notes"] = notes
    if note not in notes:
        notes.append(note)


def _sanitize_cluster_constraints(
    *,
    payload: dict[str, object],
    cluster_id: str,
    cluster: dict[str, object],
) -> None:
    members = {
        item
        for item in _coerce_str_list(cluster.get("members"))
        if isinstance(item, str) and item
    }
    cluster_rules = cluster.get("cluster_rules")
    grid_mm = GLOBAL_LAYOUT_GRID_MM
    if isinstance(cluster_rules, Mapping):
        grid_mm = _coerce_int(
            cluster_rules.get("grid_mm"),
            default=GLOBAL_LAYOUT_GRID_MM,
        )

    hard_constraints = cluster.get("hard_constraints")
    if isinstance(hard_constraints, list):
        sanitized_hard: list[dict[str, object]] = []
        for constraint in hard_constraints:
            if not isinstance(constraint, dict):
                continue
            if not _constraint_subjects_are_known(constraint, members):
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: dropped hard constraint with unknown subjects: {constraint!r}",
                )
                continue
            if _should_drop_object_containment(constraint):
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: dropped contain_in between floor objects: {constraint!r}",
                )
                continue
            normalized = dict(constraint)
            if str(normalized.get("type") or "").lower() in {
                "anchor_side",
                "dock_to_edge",
            }:
                gap_min, gap_max = _normalize_gap_range_for_grid(
                    normalized.get("gap_min"),
                    normalized.get("gap_max"),
                    grid_mm=grid_mm,
                )
                if gap_min != normalized.get("gap_min") or gap_max != normalized.get(
                    "gap_max"
                ):
                    _append_payload_note(
                        payload,
                        "Cluster "
                        f"{cluster_id}: widened {normalized.get('type')} gap window "
                        f"to fit grid {grid_mm}mm.",
                    )
                normalized["gap_min"] = gap_min
                normalized["gap_max"] = gap_max
            sanitized_hard.append(normalized)
        cluster["hard_constraints"] = sanitized_hard

    soft_constraints = cluster.get("soft_constraints")
    if isinstance(soft_constraints, list):
        sanitized_soft: list[dict[str, object]] = []
        for constraint in soft_constraints:
            if not isinstance(constraint, dict):
                continue
            if not _constraint_subjects_are_known(constraint, members):
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: dropped soft constraint with unknown subjects: {constraint!r}",
                )
                continue
            sanitized_soft.append(dict(constraint))
        cluster["soft_constraints"] = sanitized_soft


def _constraint_subjects_are_known(
    constraint: Mapping[str, object],
    members: set[str],
) -> bool:
    ctype = str(constraint.get("type") or "").lower()
    if ctype == "requires_access":
        obj_id = constraint.get("id")
        return isinstance(obj_id, str) and obj_id in members

    a = constraint.get("a")
    b = constraint.get("b")
    if not isinstance(a, str) or a not in members:
        return False
    if ctype == "contain_in" and b == "room":
        return True
    return isinstance(b, str) and b in members


def _should_drop_object_containment(constraint: Mapping[str, object]) -> bool:
    return (
        str(constraint.get("type") or "").lower() == "contain_in"
        and constraint.get("b") != "room"
    )


def _normalize_gap_range_for_grid(
    gap_min: object,
    gap_max: object,
    *,
    grid_mm: int,
) -> tuple[int, int]:
    min_value, max_value = _normalize_gap_range(gap_min, gap_max)
    snapped_grid = max(1, _coerce_int(grid_mm, default=GLOBAL_LAYOUT_GRID_MM))
    if max_value <= 0:
        return min_value, max_value
    if max_value <= snapped_grid:
        return 0, snapped_grid
    return min_value, max_value
