from __future__ import annotations

from clients.base_client import LLMClientProtocol
from config.mistral_config import MistralConfig
from config.ollama_config import OllamaConfig


def get_llm_client() -> LLMClientProtocol:
    if MistralConfig.ENABLED:
        from clients.mistral_client import MistralClient

        return MistralClient()
    if OllamaConfig.ENABLED:
        from clients.ollama_client import OllamaClient

        return OllamaClient()
    from clients.openai_client import OpenAIClient

    return OpenAIClient()
