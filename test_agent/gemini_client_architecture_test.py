from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import TypedDict
from unittest import mock

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


config_models_module = _load_module(
    "test_config_models_for_gemini",
    "config/models.py",
)
Service = config_models_module.Service

fake_clients_package = types.ModuleType("clients")
fake_clients_package.__path__ = []  # type: ignore[attr-defined]
sys.modules["clients"] = fake_clients_package

fake_base_client_module = types.ModuleType("clients.base_client")


class _FakeChatMessage(TypedDict, total=False):
    role: str
    content: str


fake_base_client_module.ChatMessage = _FakeChatMessage
fake_base_client_module.LLMClientProtocol = object
fake_base_client_module.LLMModelKey = str
fake_base_client_module.ThinkingLevel = str
sys.modules["clients.base_client"] = fake_base_client_module

fake_config_package = types.ModuleType("config")
fake_config_package.__path__ = []  # type: ignore[attr-defined]
sys.modules["config"] = fake_config_package

fake_config_models_module = types.ModuleType("config.models")
fake_config_models_module.GeminiModelConfig = config_models_module.GeminiModelConfig
sys.modules["config.models"] = fake_config_models_module

fake_gemini_config_module = types.ModuleType("config.gemini_config")


class _FakeGeminiConfig:
    ENABLED = False
    API_KEY = "gemini-key"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    MODELS = None
    AGENT_MODELS: dict[str, str] = {}


fake_gemini_config_module.GeminiConfig = _FakeGeminiConfig
sys.modules["config.gemini_config"] = fake_gemini_config_module

fake_mistral_config_module = types.ModuleType("config.mistral_config")


class _FakeMistralConfig:
    ENABLED = False


fake_mistral_config_module.MistralConfig = _FakeMistralConfig
sys.modules["config.mistral_config"] = fake_mistral_config_module

fake_ollama_config_module = types.ModuleType("config.ollama_config")


class _FakeOllamaConfig:
    ENABLED = False


fake_ollama_config_module.OllamaConfig = _FakeOllamaConfig
sys.modules["config.ollama_config"] = fake_ollama_config_module

gemini_client_module = _load_module("clients.gemini_client", "clients/gemini_client.py")
GeminiClient = gemini_client_module.GeminiClient
llm_client_module = _load_module("clients.llm_client", "clients/llm_client.py")


class GeminiClientArchitectureTest(unittest.TestCase):
    def test_service_accepts_gemini_agent_models(self) -> None:
        service = Service(
            openai={
                "azure": False,
                "api_version": "2024-08-01-preview",
                "api_key": "openai-key",
                "models": {
                    "gpt_4o": {"name": "gpt-4o"},
                    "gpt_4o_mini": {"name": "gpt-4o-mini"},
                    "text_embedding_3_small": {"name": "text-embedding-3-small"},
                },
            },
            gemini={
                "enabled": True,
                "api_key": "gemini-key",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "models": {
                    "primary": {"name": "gemini-3-flash"},
                    "helper": {"name": "gemini-3.1-flash-lite"},
                    "embedding": {"name": "gemini-embedding-001"},
                },
                "agent_models": {
                    "forge": {"name": "gemma-3-27b-it"},
                    "judge": {"name": "gemini-3.1-flash-lite"},
                },
            },
        )

        self.assertEqual(
            service.gemini.agent_models["forge"].name,
            "gemma-3-27b-it",
        )

    def test_service_rejects_simultaneous_ollama_and_mistral(self) -> None:
        with self.assertRaises(ValueError):
            Service(
                openai={
                    "azure": False,
                    "api_version": "2024-08-01-preview",
                    "api_key": "openai-key",
                    "models": {
                        "gpt_4o": {"name": "gpt-4o"},
                        "gpt_4o_mini": {"name": "gpt-4o-mini"},
                        "text_embedding_3_small": {"name": "text-embedding-3-small"},
                    },
                },
                ollama={
                    "enabled": True,
                    "api_key": "ollama",
                    "base_url": "http://localhost:11434/v1",
                    "models": {
                        "primary": {"name": "qwen3:8b"},
                        "helper": {"name": "qwen3:8b"},
                        "embedding": {"name": "nomic-embed-text"},
                    },
                },
                mistral={
                    "enabled": True,
                    "api_key": "mistral",
                    "base_url": "http://localhost:11434/v1",
                    "models": {
                        "primary": {"name": "mistral:instruct"},
                        "helper": {"name": "mistral:instruct"},
                        "embedding": {"name": "nomic-embed-text"},
                    },
                },
            )

    def test_raw_response_parses_manual_tool_calls(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        response = client._to_openai_style_raw_response(
            {
                "modelVersion": "gemini-2.5-flash",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        "[TOOL_CALLS] "
                                        '[{"name":"estimate_budget","arguments":{"cluster_id":"work"}}]'
                                    )
                                }
                            ]
                        }
                    }
                ],
            }
        )

        message = response.choices[0].message
        self.assertEqual(message.content, "")
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].function.name, "estimate_budget")
        self.assertEqual(
            message.tool_calls[0].function.arguments,
            '{"cluster_id": "work"}',
        )

    def test_embedding_response_matches_openai_shape(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        response = client._to_openai_style_embedding_response(
            [[0.1, 0.2], [0.3, 0.4]],
            {"modelVersion": "gemini-embedding-001"},
        )

        self.assertEqual(len(response.data), 2)
        self.assertEqual(response.data[0].embedding, [0.1, 0.2])
        self.assertEqual(response.data[1].embedding, [0.3, 0.4])

    def test_build_chat_payload_includes_thinking_level(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        payload = client._build_chat_payload(
            [{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=256,
            thinking_level="medium",
            response_mime_type=None,
        )

        self.assertEqual(
            ((payload.get("generationConfig") or {}).get("thinkingConfig") or {}).get(
                "thinkingLevel"
            ),
            "medium",
        )

    def test_build_chat_payload_includes_response_mime_type(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        payload = client._build_chat_payload(
            [{"role": "user", "content": "Hello"}],
            temperature=0.0,
            max_tokens=128,
            thinking_level=None,
            response_mime_type="application/json",
        )

        self.assertEqual(
            ((payload.get("generationConfig") or {}).get("responseMimeType")),
            "application/json",
        )

    def test_openai_style_response_carries_finish_reason(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        response = GeminiClient._to_openai_style_response(
            client,
            {
                "modelVersion": "gemini-3-flash-preview",
                "candidates": [
                    {
                        "finishReason": "MAX_TOKENS",
                        "content": {"parts": [{"text": '{"status":"OK"}'}]},
                    }
                ],
            },
        )

        self.assertEqual(response.choices[0].finish_reason, "MAX_TOKENS")

    def test_chat_completion_skips_thinking_for_known_unsupported_model(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        captured: dict[str, object] = {}

        def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
            captured["url"] = url
            captured["payload"] = payload
            return {"modelVersion": "gemma-4-31b-it", "candidates": []}

        client._post_json = _fake_post_json  # type: ignore[method-assign]
        client._to_openai_style_response = lambda data: data  # type: ignore[assignment]
        client._base_url = "https://generativelanguage.googleapis.com/v1beta"

        response = client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model_key="primary",
            model_name="gemma-4-31b-it",
            thinking_level="medium",
        )

        self.assertEqual(
            captured["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent",
        )
        self.assertNotIn("generationConfig", captured["payload"])
        self.assertEqual(response["modelVersion"], "gemma-4-31b-it")

    def test_chat_completion_retries_without_thinking_when_model_rejects_it(
        self,
    ) -> None:
        client = GeminiClient.__new__(GeminiClient)
        original_models = set(GeminiClient._thinking_level_unsupported_models)
        self.addCleanup(
            lambda: (
                GeminiClient._thinking_level_unsupported_models.clear(),
                GeminiClient._thinking_level_unsupported_models.update(original_models),
            )
        )

        captured_payloads: list[dict[str, object]] = []

        def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
            captured_payloads.append(payload)
            if len(captured_payloads) == 1:
                request = httpx.Request("POST", url)
                response = httpx.Response(
                    400,
                    request=request,
                    text=(
                        '{"error":{"message":"Thinking level is not supported for this model."}}'
                    ),
                )
                raise httpx.HTTPStatusError(
                    "Thinking level is not supported for this model.",
                    request=request,
                    response=response,
                )
            return {"modelVersion": "future-model", "candidates": []}

        client._post_json = _fake_post_json  # type: ignore[method-assign]
        client._to_openai_style_response = lambda data: data  # type: ignore[assignment]
        client._base_url = "https://generativelanguage.googleapis.com/v1beta"

        response = client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model_key="primary",
            model_name="future-model",
            thinking_level="medium",
        )

        self.assertEqual(len(captured_payloads), 2)
        self.assertEqual(
            captured_payloads[0]["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "medium",
        )
        self.assertNotIn("generationConfig", captured_payloads[1])
        self.assertIn("future-model", GeminiClient._thinking_level_unsupported_models)
        self.assertEqual(response["modelVersion"], "future-model")

    def test_chat_completion_uses_model_name_override(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        captured: dict[str, object] = {}

        def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
            captured["url"] = url
            captured["payload"] = payload
            return {"modelVersion": "gemma-3-27b-it", "candidates": []}

        client._post_json = _fake_post_json  # type: ignore[method-assign]
        client._build_chat_payload = lambda *args, **kwargs: {"contents": []}  # type: ignore[assignment]
        client._to_openai_style_response = lambda data: data  # type: ignore[assignment]
        client._base_url = "https://generativelanguage.googleapis.com/v1beta"

        response = client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model_key="primary",
            model_name="gemma-3-27b-it",
        )

        self.assertEqual(
            captured["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent",
        )
        self.assertEqual(response["modelVersion"], "gemma-3-27b-it")

    def test_chat_completion_falls_back_after_retryable_503(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        attempted_urls: list[str] = []

        def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
            attempted_urls.append(url)
            if "gemini-3.1-flash-lite-preview" in url:
                request = httpx.Request("POST", url)
                response = httpx.Response(
                    503,
                    request=request,
                    text=(
                        '{"error":{"code":503,"message":"This model is currently '
                        'experiencing high demand.","status":"UNAVAILABLE"}}'
                    ),
                )
                raise httpx.HTTPStatusError(
                    "503 high demand",
                    request=request,
                    response=response,
                )
            return {"modelVersion": "gemini-3-flash-preview", "candidates": []}

        client._post_json = _fake_post_json  # type: ignore[method-assign]
        client._to_openai_style_response = lambda data: data  # type: ignore[assignment]
        client._base_url = "https://generativelanguage.googleapis.com/v1beta"

        response = client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model_name="gemini-3.1-flash-lite-preview",
        )

        self.assertEqual(
            attempted_urls,
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
            ],
        )
        self.assertEqual(response["modelVersion"], "gemini-3-flash-preview")

    def test_chat_completion_tries_requested_model_before_fallback_pool(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        attempted_urls: list[str] = []

        def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
            attempted_urls.append(url)
            if "custom-model" in url:
                request = httpx.Request("POST", url)
                response = httpx.Response(
                    503,
                    request=request,
                    text=(
                        '{"error":{"code":503,"message":"This model is currently '
                        'experiencing high demand.","status":"UNAVAILABLE"}}'
                    ),
                )
                raise httpx.HTTPStatusError(
                    "503 high demand",
                    request=request,
                    response=response,
                )
            return {"modelVersion": "gemini-3.1-flash-lite-preview", "candidates": []}

        client._post_json = _fake_post_json  # type: ignore[method-assign]
        client._to_openai_style_response = lambda data: data  # type: ignore[assignment]
        client._base_url = "https://generativelanguage.googleapis.com/v1beta"

        response = client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model_name="custom-model",
        )

        self.assertEqual(
            attempted_urls[:2],
            [
                "https://generativelanguage.googleapis.com/v1beta/models/custom-model:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent",
            ],
        )
        self.assertEqual(response["modelVersion"], "gemini-3.1-flash-lite-preview")

    def test_chat_completion_honors_explicit_fallback_sequence(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        attempted_urls: list[str] = []

        def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
            attempted_urls.append(url)
            if "gemini-3.1-flash-lite-preview" in url:
                request = httpx.Request("POST", url)
                response = httpx.Response(
                    503,
                    request=request,
                    text=(
                        '{"error":{"code":503,"message":"This model is currently '
                        'experiencing high demand.","status":"UNAVAILABLE"}}'
                    ),
                )
                raise httpx.HTTPStatusError(
                    "503 high demand",
                    request=request,
                    response=response,
                )
            return {"modelVersion": "gemini-2.5-flash-lite", "candidates": []}

        client._post_json = _fake_post_json  # type: ignore[method-assign]
        client._to_openai_style_response = lambda data: data  # type: ignore[assignment]
        client._base_url = "https://generativelanguage.googleapis.com/v1beta"

        response = client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model_name="gemini-3.1-flash-lite-preview",
            fallback_model_names=[
                "gemini-2.5-flash-lite",
                "gemini-3-flash-preview",
            ],
        )

        self.assertEqual(
            attempted_urls,
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
            ],
        )
        self.assertEqual(response["modelVersion"], "gemini-2.5-flash-lite")

    def test_tool_chat_completion_falls_back_after_retryable_503(self) -> None:
        client = GeminiClient.__new__(GeminiClient)
        attempted_urls: list[str] = []

        def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
            attempted_urls.append(url)
            if "gemini-3.1-flash-lite-preview" in url:
                request = httpx.Request("POST", url)
                response = httpx.Response(
                    503,
                    request=request,
                    text=(
                        '{"error":{"code":503,"message":"This model is currently '
                        'experiencing high demand.","status":"UNAVAILABLE"}}'
                    ),
                )
                raise httpx.HTTPStatusError(
                    "503 high demand",
                    request=request,
                    response=response,
                )
            return {"modelVersion": "gemini-3-flash-preview", "candidates": []}

        client._post_json = _fake_post_json  # type: ignore[method-assign]
        client._to_openai_style_raw_response = lambda data: data  # type: ignore[assignment]
        client._base_url = "https://generativelanguage.googleapis.com/v1beta"

        response = client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model_name="gemini-3.1-flash-lite-preview",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup something",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        self.assertEqual(
            attempted_urls,
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
            ],
        )
        self.assertEqual(response["modelVersion"], "gemini-3-flash-preview")

    def test_factory_returns_explicit_gemini_provider(self) -> None:
        fake_module = types.ModuleType("clients.gemini_client")

        class _FakeGeminiClient:
            pass

        fake_module.GeminiClient = _FakeGeminiClient

        with (
            mock.patch.object(llm_client_module.GeminiConfig, "ENABLED", True),
            mock.patch.object(llm_client_module.MistralConfig, "ENABLED", False),
            mock.patch.object(llm_client_module.OllamaConfig, "ENABLED", True),
            mock.patch.dict(sys.modules, {"clients.gemini_client": fake_module}),
        ):
            client = llm_client_module.get_llm_client(provider="gemini")

        self.assertIsInstance(client, _FakeGeminiClient)


if __name__ == "__main__":
    unittest.main()
