from __future__ import annotations

import logging
from collections.abc import Sequence
from threading import Lock
from typing import ClassVar
from urllib.parse import urlparse

from openai import AzureOpenAI, OpenAI

from clients.base_client import ChatMessage, LLMModelKey, ThinkingLevel
from config.openai_config import OpenAIConfig
from config.models import OpenAIModelConfig

logger = logging.getLogger(__name__)


class OpenAIClient:
    _instance: ClassVar["OpenAIClient | None"] = None
    _lock: ClassVar[Lock] = Lock()

    _initialized: bool = False
    _client: OpenAI | AzureOpenAI
    _model_map: dict[LLMModelKey, OpenAIModelConfig]
    _azure_endpoint: str | None
    _api_key: str

    def __new__(cls) -> "OpenAIClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._api_key = self._require_api_key()
        self._model_map = self._build_model_map()
        self._azure_endpoint = self._resolve_azure_endpoint()
        self._client = self._build_client()
        self._initialized = True

    @property
    def client(self) -> OpenAI | AzureOpenAI:
        return self._client

    def get_model_name(self, key: LLMModelKey) -> str:
        return self._model_map[key].name

    def get_model_endpoint(self, key: LLMModelKey) -> str | None:
        return self._model_map[key].azure_endpoint

    def chat_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        model_key: LLMModelKey = "primary",
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        thinking_level: ThinkingLevel | None = None,
        response_mime_type: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
        tool_choice: dict[str, object] | str | None = None,
    ) -> object:
        resolved_model_name = model_name or self.get_model_name(model_key)
        message_list = list(messages)
        _ = thinking_level
        _ = response_mime_type

        params: dict[str, object] = {
            "model": resolved_model_name,
            "messages": message_list,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools is not None:
            params["tools"] = list(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        return self._client.chat.completions.create(**params)

    def embeddings(
        self,
        inputs: str | Sequence[str],
        *,
        model_key: LLMModelKey = "embedding",
    ) -> object:
        model_name = self.get_model_name(model_key)
        normalized_inputs = self._normalize_embedding_input(inputs)
        return self._client.embeddings.create(
            model=model_name,
            input=normalized_inputs,
        )

    def _build_client(self) -> OpenAI | AzureOpenAI:
        if OpenAIConfig.IS_OPENAI_AZURE:
            if self._azure_endpoint is None:
                raise ValueError("Azure endpoint is required when azure=True.")
            return AzureOpenAI(
                api_key=self._api_key,
                api_version=OpenAIConfig.API_VERSION,
                azure_endpoint=self._azure_endpoint,
            )
        return OpenAI(api_key=self._api_key)

    def _build_model_map(self) -> dict[LLMModelKey, OpenAIModelConfig]:
        models = OpenAIConfig.MODELS
        return {
            "primary": models.gpt_4o,
            "helper": models.gpt_4o_mini,
            "embedding": models.text_embedding_3_small,
        }

    def _resolve_azure_endpoint(self) -> str | None:
        if not OpenAIConfig.IS_OPENAI_AZURE:
            return None

        endpoints: set[str] = set()
        for model_config in self._model_map.values():
            endpoint = model_config.azure_endpoint
            if endpoint is None:
                continue
            normalized = self._normalize_azure_endpoint(endpoint)
            endpoints.add(normalized)

        if not endpoints:
            return None

        if len(endpoints) > 1:
            logger.warning(
                "Multiple Azure endpoints detected; using the first one: %s",
                sorted(endpoints)[0],
            )

        return sorted(endpoints)[0]

    def _normalize_azure_endpoint(self, endpoint: str) -> str:
        parsed = urlparse(endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid Azure endpoint: {endpoint}")
        return f"{parsed.scheme}://{parsed.netloc}"

    def _normalize_embedding_input(
        self, inputs: str | Sequence[str]
    ) -> str | list[str]:
        if isinstance(inputs, str):
            return inputs
        return list(inputs)

    def _require_api_key(self) -> str:
        api_key = OpenAIConfig.OPENAI_API_KEY
        if not api_key:
            raise ValueError("Missing OpenAI API key. Please configure it first.")
        return api_key
