from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from agent_schema.roomI_schema import RoomInterpreterOutput
from clients.openai_client import ChatMessage, OpenAIClient
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
        response = OpenAIClient().chat_completion(
            messages,
            model_key="gpt_4o",
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
