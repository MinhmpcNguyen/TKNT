from __future__ import annotations

import json
import unittest
from collections.abc import Mapping, Sequence
from unittest.mock import patch

from stylist import style_policy

ChatMessage = dict[str, str]
LLMModelKey = str


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    def __init__(self, outcomes: Sequence[str | Mapping[str, object]]) -> None:
        self._outcomes = list(outcomes)
        self.model_names: list[str | None] = []

    def get_model_name(self, key: LLMModelKey) -> str:
        _ = key
        return "gpt-5.4-mini"

    def chat_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        model_key: LLMModelKey = "primary",
        model_name: str | None = None,
        fallback_model_names: Sequence[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_mime_type: str | None = None,
        response_schema: Mapping[str, object] | None = None,
        tools: Sequence[dict[str, object]] | None = None,
        tool_choice: dict[str, object] | str | None = None,
    ) -> _FakeResponse:
        _ = (
            messages,
            model_key,
            fallback_model_names,
            temperature,
            top_p,
            max_tokens,
            response_mime_type,
            response_schema,
            tools,
            tool_choice,
        )
        self.model_names.append(model_name)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, str):
            return _FakeResponse(outcome)
        return _FakeResponse(json.dumps(dict(outcome)))


def _style_seed(style_name: str) -> dict[str, object]:
    return {
        "style_name": style_name,
        "target_density": "balanced",
        "center_openness_bias": "medium",
        "wall_loading_bias": "balanced",
        "symmetry_bias": "medium",
        "cluster_spacing_bias": "balanced",
        "clutter_tolerance": "medium",
        "decor_tolerance": "medium",
        "daylight_bias": "medium",
        "floating_cluster_tolerance": "medium",
        "material_weight_bias": "balanced",
        "visual_balance_bias": "balanced",
    }


class StylePolicyArchitectureTest(unittest.TestCase):
    def test_strict_llm_style_policy_repairs_with_openai_model_after_unterminated_json(
        self,
    ) -> None:
        broken_json = '{\n  "style_name": "minimal",\n  "target_density": "balanced'
        fake_client = _FakeClient(
            [
                broken_json,
                _style_seed("japandi"),
            ]
        )

        with (
            patch.object(style_policy.TextLLMConfig, "STRICT_SINGLE_TEXT_MODEL", ""),
            patch.object(style_policy, "get_llm_client", return_value=fake_client),
        ):
            policy = style_policy.compile_style_policy(
                room_type="bedroom",
                brief_text="quiet japandi bedroom",
            )

        self.assertEqual(policy["style_name"], "japandi")
        self.assertTrue(policy["controller"]["llm_seed_used"])
        self.assertEqual(
            fake_client.model_names,
            [
                "gpt-5.4-mini",
                "gpt-5.4-mini",
            ],
        )


if __name__ == "__main__":
    raise SystemExit(unittest.main())
