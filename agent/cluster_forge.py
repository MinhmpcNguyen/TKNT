from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agent_schema.clusterF_schema import ClusterForgeOutput
from clients.base_client import ChatMessage
from clients.llm_client import get_llm_client
from config.gemini_config import GeminiConfig
from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM, normalize_cluster_rules_grid
from layout.semantic_roles import (
    is_bed_like,
    is_bedside_support_like,
    is_bench_like,
    is_lounge_anchor_like,
    is_seat_like,
    is_surface_side_accessory_like,
    is_work_surface_like,
)
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
        response = get_llm_client(provider="gemini").chat_completion(
            messages,
            model_key="primary",
            model_name=GeminiConfig.AGENT_MODELS.get("forge"),
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
        else:
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


def _normalize_facing_front_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    token = value.strip().lower()
    mapping = {
        "front": "top",
        "back": "bottom",
        "head": "top",
        "foot": "bottom",
        "up": "top",
        "down": "bottom",
        "north": "top",
        "south": "bottom",
        "west": "left",
        "east": "right",
    }
    token = mapping.get(token, token)
    if token in {"top", "bottom", "left", "right"}:
        return token
    return None


def _object_prefers_explicit_facing(object_id: str) -> bool:
    key = object_id.lower()
    if (
        is_seat_like(object_id)
        or is_work_surface_like(object_id)
        or is_lounge_anchor_like(object_id)
    ):
        return True
    return any(
        pattern in key
        for pattern in (
            "wardrobe",
            "closet",
            "dresser",
            "cabinet",
            "console",
            "tv",
            "vanity",
            "storage",
        )
    )


def _default_facing_front(object_id: str) -> str:
    _ = object_id
    return "top"


def _normalize_facing_rules(
    *,
    payload: dict[str, object],
    cluster_id: str,
    members: set[str],
    facing: object,
) -> dict[str, dict[str, str]]:
    if not isinstance(facing, Mapping):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for obj_id, raw_rule in facing.items():
        if not isinstance(obj_id, str) or obj_id not in members:
            _append_payload_note(
                payload,
                f"Cluster {cluster_id}: dropped facing rule for unknown object {obj_id!r}.",
            )
            continue

        rule = raw_rule if isinstance(raw_rule, Mapping) else {}
        front = _normalize_facing_front_token(rule.get("front"))
        if front is None:
            if _object_prefers_explicit_facing(obj_id):
                front = _default_facing_front(obj_id)
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: defaulted ambiguous facing.front for {obj_id} to {front}.",
                )
            else:
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: dropped facing rule for {obj_id} because it has no stable front side.",
                )
                continue

        normalized_rule: dict[str, str] = {"front": front}
        notes = rule.get("notes")
        if isinstance(notes, str):
            note_text = notes.strip()
            if note_text:
                normalized_rule["notes"] = note_text
        normalized[obj_id] = normalized_rule
    return normalized


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
    member_list = [
        item
        for item in _coerce_str_list(cluster.get("members"))
        if isinstance(item, str) and item
    ]
    cluster["members"] = member_list
    members = set(member_list)
    cluster["anchors"] = _sanitize_cluster_anchors(
        payload=payload,
        cluster_id=cluster_id,
        cluster_tag=str(cluster.get("tag") or "").lower(),
        members=member_list,
        anchors=cluster.get("anchors"),
    )
    cluster_rules = cluster.get("cluster_rules")
    grid_mm = GLOBAL_LAYOUT_GRID_MM
    if isinstance(cluster_rules, Mapping):
        grid_mm = _coerce_int(
            cluster_rules.get("grid_mm"),
            default=GLOBAL_LAYOUT_GRID_MM,
        )
    if isinstance(cluster_rules, dict):
        allowed_rotations = cluster_rules.get("allowed_rotations")
        if isinstance(allowed_rotations, Mapping):
            cluster_rules["allowed_rotations"] = {
                obj_id: rotations
                for obj_id, rotations in allowed_rotations.items()
                if isinstance(obj_id, str) and obj_id in members
            }
        cluster_rules["facing"] = _normalize_facing_rules(
            payload=payload,
            cluster_id=cluster_id,
            members=members,
            facing=cluster_rules.get("facing"),
        )
        access_requirements = cluster_rules.get("access_requirements")
        if isinstance(access_requirements, list):
            sanitized_access_requirements: list[dict[str, object]] = []
            for requirement in access_requirements:
                if not isinstance(requirement, Mapping):
                    continue
                obj_id = requirement.get("id")
                if not isinstance(obj_id, str) or obj_id not in members:
                    _append_payload_note(
                        payload,
                        f"Cluster {cluster_id}: dropped access requirement for unknown object {obj_id!r}.",
                    )
                    continue
                sanitized_access_requirements.append(dict(requirement))
            cluster_rules["access_requirements"] = sanitized_access_requirements
        else:
            cluster_rules["access_requirements"] = []

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
    _apply_semantic_layout_intents(
        payload=payload,
        cluster_id=cluster_id,
        cluster=cluster,
        members=members,
    )


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


def _sanitize_cluster_anchors(
    *,
    payload: dict[str, object],
    cluster_id: str,
    cluster_tag: str,
    members: Sequence[str],
    anchors: object,
) -> list[str]:
    member_set = {member for member in members if isinstance(member, str) and member}
    raw_anchors = _coerce_str_list(anchors)
    kept = [anchor for anchor in raw_anchors if anchor in member_set]
    dropped = [anchor for anchor in raw_anchors if anchor not in member_set]

    if dropped:
        _append_payload_note(
            payload,
            f"Cluster {cluster_id}: dropped anchors not present in members: {sorted(set(dropped))}.",
        )

    if kept:
        return kept

    fallback = _choose_fallback_anchor(cluster_tag=cluster_tag, members=members)
    if fallback is not None:
        if raw_anchors:
            note = (
                f"Cluster {cluster_id}: defaulted anchors to {fallback!r} because the original anchors "
                "were invalid for this cluster."
            )
        else:
            note = f"Cluster {cluster_id}: defaulted anchors to {fallback!r} because none were provided."
        _append_payload_note(payload, note)
        return [fallback]

    return []


def _choose_fallback_anchor(
    *,
    cluster_tag: str,
    members: Sequence[str],
) -> str | None:
    if not members:
        return None

    best_member: str | None = None
    best_score = -1
    for member in members:
        if not isinstance(member, str) or not member:
            continue
        score = _anchor_priority_score(cluster_tag=cluster_tag, member=member)
        if score > best_score:
            best_member = member
            best_score = score

    return best_member


def _anchor_priority_score(*, cluster_tag: str, member: str) -> int:
    key = member.lower()
    score = 0

    if cluster_tag == "sleep":
        if is_bed_like(member):
            score += 1_000
        elif is_bedside_support_like(member):
            score += 700
        elif is_bench_like(member):
            score += 500
    elif cluster_tag == "work":
        if is_work_surface_like(member):
            score += 1_000
        elif is_seat_like(member):
            score += 700
    elif cluster_tag == "living":
        if "sectional" in key:
            score += 1_050
        elif "sofa" in key or "loveseat" in key:
            score += 1_000
        elif is_lounge_anchor_like(member):
            score += 850
        elif is_seat_like(member):
            score += 700
    elif cluster_tag == "dining":
        if "dining_table" in key or "table" in key:
            score += 1_000
        elif is_seat_like(member):
            score += 700
    elif cluster_tag == "storage":
        if "wardrobe" in key or "closet" in key:
            score += 1_120
        elif "storage_cabinet" in key:
            score += 1_100
        elif "dresser" in key or "sideboard" in key or "buffet" in key:
            score += 1_080
        elif "media_shelf" in key or "bookshelf" in key:
            score += 1_060
        elif "tv_console" in key:
            score += 1_040
        elif "console_table" in key:
            score += 1_000
        elif "coffee_table" in key or "ottoman" in key:
            score += 300
    else:
        if is_bed_like(member):
            score += 950
        if is_work_surface_like(member):
            score += 900
        if "sofa" in key or "sectional" in key:
            score += 900
        if is_lounge_anchor_like(member):
            score += 850
        if any(token in key for token in ("wardrobe", "storage_cabinet", "dresser")):
            score += 800

    return score


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


def _apply_semantic_layout_intents(
    *,
    payload: dict[str, object],
    cluster_id: str,
    cluster: dict[str, object],
    members: set[str],
) -> None:
    rules = cluster.get("cluster_rules")
    if not isinstance(rules, dict):
        rules = {}
        cluster["cluster_rules"] = rules

    semantic_placements = _normalize_semantic_placements(
        rules.get("semantic_placements"),
        members=members,
    )

    hard_constraints = cluster.get("hard_constraints")
    if not isinstance(hard_constraints, list):
        hard_constraints = []
        cluster["hard_constraints"] = hard_constraints

    # Strengthen common serving-surface relationships so the composer can place
    # dependents in a semantically meaningful region, not merely any valid edge.
    for constraint in hard_constraints:
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("type") or "").lower() != "dock_to_edge":
            continue

        a = constraint.get("a")
        b = constraint.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            continue

        if is_seat_like(a) and is_work_surface_like(b):
            if constraint.get("span") in {"any", "short_edge", "long_edge"}:
                constraint["span"] = "center"
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: refined {a}->{b} dock span to center for a serving-surface seating relationship.",
                )
            _upsert_semantic_placement(
                semantic_placements,
                {
                    "id": a,
                    "relative_to": b,
                    "kind": "dock_to_edge",
                    "b_edge": str(constraint.get("b_edge") or "front"),
                    "span": str(constraint.get("span") or "center"),
                    "gap_min": _coerce_int(constraint.get("gap_min"), default=0),
                    "gap_max": _coerce_int(constraint.get("gap_max"), default=0),
                    "selection": "best_fit",
                    "orientation": "face_base",
                },
            )
            _ensure_soft_constraint(
                cluster,
                {
                    "type": "prefer_facing",
                    "a": a,
                    "b": b,
                    "mode": "face_each_other",
                    "weight": 8,
                },
            )
            continue

        if is_bench_like(a) and is_bed_like(b):
            if constraint.get("span") in {"any", "short_edge", "long_edge"}:
                constraint["span"] = "center"
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: refined {a}->{b} dock span to center for a foot-of-bed seating relationship.",
                )
            _upsert_semantic_placement(
                semantic_placements,
                {
                    "id": a,
                    "relative_to": b,
                    "kind": "dock_to_edge",
                    "b_edge": str(constraint.get("b_edge") or "back"),
                    "span": str(constraint.get("span") or "center"),
                    "gap_min": _coerce_int(constraint.get("gap_min"), default=0),
                    "gap_max": _coerce_int(constraint.get("gap_max"), default=0),
                    "selection": "best_fit",
                    "orientation": "face_base",
                },
            )

    bedside_relations: list[dict[str, object]] = []
    for constraint in hard_constraints:
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("type") or "").lower() != "anchor_side":
            continue
        a = constraint.get("a")
        b = constraint.get("b")
        side = constraint.get("side")
        if (
            isinstance(a, str)
            and isinstance(b, str)
            and isinstance(side, str)
            and is_bedside_support_like(a)
            and is_bed_like(b)
            and side in {"head", "foot"}
        ):
            bedside_relations.append(constraint)

    if len(bedside_relations) >= 2:
        ordered = sorted(
            bedside_relations,
            key=lambda item: str(item.get("a") or ""),
        )
        side_tokens = ["head_left", "head_right"]
        if str(ordered[0].get("side") or "") == "foot":
            side_tokens = ["foot_left", "foot_right"]
        for constraint, side_token in zip(ordered[:2], side_tokens, strict=False):
            if constraint.get("side") != side_token:
                constraint["side"] = side_token
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: assigned {constraint.get('a')} to {side_token} for clearer bedside semantics.",
                )

    for constraint in bedside_relations:
        a = str(constraint.get("a") or "")
        b = str(constraint.get("b") or "")
        side = str(constraint.get("side") or "")
        if side in {"head", "foot"}:
            option_prefix = "head" if side == "head" else "foot"
            _upsert_semantic_placement(
                semantic_placements,
                {
                    "id": a,
                    "relative_to": b,
                    "kind": "anchor_side",
                    "side_options": [
                        f"{option_prefix}_left",
                        f"{option_prefix}_right",
                    ],
                    "gap_min": _coerce_int(constraint.get("gap_min"), default=0),
                    "gap_max": _coerce_int(constraint.get("gap_max"), default=0),
                    "selection": "best_fit",
                    "orientation": "same_direction",
                },
            )

    soft_constraints = cluster.get("soft_constraints")
    if isinstance(soft_constraints, list):
        for constraint in soft_constraints:
            if not isinstance(constraint, dict):
                continue
            if str(constraint.get("type") or "").lower() != "prefer_near":
                continue
            a = constraint.get("a")
            b = constraint.get("b")
            if not isinstance(a, str) or not isinstance(b, str):
                continue

            if is_surface_side_accessory_like(a) and (
                is_lounge_anchor_like(b) or is_seat_like(b)
            ):
                _upsert_semantic_placement(
                    semantic_placements,
                    {
                        "id": a,
                        "relative_to": b,
                        "kind": "anchor_side",
                        "side_options": ["left", "right"],
                        "gap_min": 50,
                        "gap_max": 200,
                        "selection": "best_fit",
                        "orientation": "same_direction",
                    },
                )

    rules["semantic_placements"] = semantic_placements


def _normalize_semantic_placements(
    value: object,
    *,
    members: set[str],
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    out: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        obj_id = item.get("id")
        relative_to = item.get("relative_to")
        kind = item.get("kind")
        if not (
            isinstance(obj_id, str)
            and obj_id in members
            and isinstance(relative_to, str)
            and relative_to in members
            and isinstance(kind, str)
        ):
            continue
        record: dict[str, object] = {
            "id": obj_id,
            "relative_to": relative_to,
            "kind": kind.strip().lower(),
            "selection": _normalize_semantic_selection(item.get("selection")),
        }
        orientation = _normalize_semantic_orientation(item.get("orientation"))
        if orientation is not None:
            record["orientation"] = orientation
        if "gap_min" in item or "gap_max" in item:
            gap_min, gap_max = _normalize_gap_range(
                item.get("gap_min"),
                item.get("gap_max"),
            )
            record["gap_min"] = gap_min
            record["gap_max"] = gap_max

        if record["kind"] == "dock_to_edge":
            b_edge = _normalize_dock_edge_token(item.get("b_edge"))
            if b_edge is not None:
                record["b_edge"] = b_edge
            record["span"] = _normalize_dock_span_token(item.get("span"))
        elif record["kind"] == "anchor_side":
            side_options = item.get("side_options")
            if isinstance(side_options, list):
                normalized_options = [
                    side
                    for raw_side in side_options
                    if (side := _normalize_anchor_side_token(raw_side)) is not None
                ]
                if normalized_options:
                    record["side_options"] = normalized_options
        out.append(record)
    return out


def _normalize_semantic_selection(value: object) -> str:
    if not isinstance(value, str):
        return "best_fit"
    token = value.strip().lower()
    if token in {"best_fit", "first_fit"}:
        return token
    return "best_fit"


def _normalize_semantic_orientation(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if token in {"face_base", "same_direction"}:
        return token
    return None


def _upsert_semantic_placement(
    semantic_placements: list[dict[str, object]],
    record: dict[str, object],
) -> None:
    obj_id = record.get("id")
    relative_to = record.get("relative_to")
    kind = record.get("kind")
    for existing in semantic_placements:
        if not isinstance(existing, dict):
            continue
        if (
            existing.get("id") == obj_id
            and existing.get("relative_to") == relative_to
            and existing.get("kind") == kind
        ):
            existing.update(record)
            return
    semantic_placements.append(record)


def _ensure_soft_constraint(
    cluster: dict[str, object], record: dict[str, object]
) -> None:
    soft_constraints = cluster.get("soft_constraints")
    if not isinstance(soft_constraints, list):
        soft_constraints = []
        cluster["soft_constraints"] = soft_constraints

    for existing in soft_constraints:
        if not isinstance(existing, dict):
            continue
        if (
            existing.get("type") == record.get("type")
            and existing.get("a") == record.get("a")
            and existing.get("b") == record.get("b")
        ):
            return
    soft_constraints.append(record)
