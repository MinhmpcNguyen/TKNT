from __future__ import annotations

import unittest

from clients.openai_client import OpenAIClient


class _FakeResponse:
    usage = None


class _FakeCompletions:
    def __init__(self) -> None:
        self.params: dict[str, object] | None = None

    def create(self, **params: object) -> _FakeResponse:
        self.params = params
        return _FakeResponse()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeSdkClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


class OpenAIClientArchitectureTest(unittest.TestCase):
    def test_gpt5_chat_completion_uses_max_completion_tokens_and_json_schema(
        self,
    ) -> None:
        fake_sdk_client = _FakeSdkClient()
        client = object.__new__(OpenAIClient)
        client._client = fake_sdk_client

        response = client.chat_completion(
            [{"role": "user", "content": "Return JSON."}],
            model_name="gpt-5.4-nano",
            max_tokens=2000,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "style_name": {"type": "STRING", "enum": ["balanced"]},
                    "relation": {
                        "type": "OBJECT",
                        "properties": {"type": {"type": "STRING"}},
                    },
                },
                "required": ["style_name"],
            },
        )

        self.assertIsInstance(response, _FakeResponse)
        params = fake_sdk_client.chat.completions.params
        self.assertIsNotNone(params)
        assert params is not None
        self.assertNotIn("max_tokens", params)
        self.assertEqual(params["max_completion_tokens"], 2000)
        response_format = params["response_format"]
        self.assertIsInstance(response_format, dict)
        json_schema = response_format["json_schema"]
        self.assertIsInstance(json_schema, dict)
        schema = json_schema["schema"]
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "object")
        properties = schema["properties"]
        self.assertIsInstance(properties, dict)
        style_name = properties["style_name"]
        self.assertIsInstance(style_name, dict)
        self.assertEqual(style_name["type"], "string")
        relation = properties["relation"]
        self.assertIsInstance(relation, dict)
        relation_properties = relation["properties"]
        self.assertIsInstance(relation_properties, dict)
        relation_type = relation_properties["type"]
        self.assertIsInstance(relation_type, dict)
        self.assertEqual(relation_type["type"], "string")

    def test_non_gpt5_chat_completion_keeps_legacy_max_tokens(self) -> None:
        fake_sdk_client = _FakeSdkClient()
        client = object.__new__(OpenAIClient)
        client._client = fake_sdk_client

        client.chat_completion(
            [{"role": "user", "content": "Hello."}],
            model_name="gpt-4o-mini",
            max_tokens=256,
        )

        params = fake_sdk_client.chat.completions.params
        self.assertIsNotNone(params)
        assert params is not None
        self.assertEqual(params["max_tokens"], 256)
        self.assertNotIn("max_completion_tokens", params)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
