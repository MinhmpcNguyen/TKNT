from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from agent_schema.clusterF_schema import ClusterForgeOutput
from clients.base_client import ChatMessage
from clients.llm_client import get_llm_client
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
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return payload

    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        soft = cluster.get("soft_constraints")
        if not isinstance(soft, list):
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
    return payload


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
        return normalized_constraint

    if normalized_type in {"prefer_near", "prefer_align_edge"}:
        return normalized_constraint

    return None


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
