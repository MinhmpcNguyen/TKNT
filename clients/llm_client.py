from __future__ import annotations

from typing import Literal

from clients.base_client import LLMClientProtocol

try:
    from config.openai_config import OpenAIConfig
except Exception:  # pragma: no cover

    class OpenAIConfig:
        ENABLED = False


LLMProvider = Literal["auto", "openai"]


def get_llm_client(provider: LLMProvider = "auto") -> LLMClientProtocol:
    if provider == "openai" or (provider == "auto" and OpenAIConfig.ENABLED):
        from clients.openai_client import OpenAIClient

        return OpenAIClient()
    from clients.openai_client import OpenAIClient

    return OpenAIClient()
