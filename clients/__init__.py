from __future__ import annotations

from typing import TYPE_CHECKING

from clients.base_client import ChatMessage, LLMClientProtocol, LLMModelKey

if TYPE_CHECKING:
    from clients.gemini_client import GeminiClient
    from clients.llm_client import get_llm_client
    from clients.mistral_client import MistralClient
    from clients.ollama_client import OllamaClient
    from clients.openai_client import OpenAIClient


def __getattr__(name: str) -> object:
    if name == "GeminiClient":
        from clients.gemini_client import GeminiClient

        return GeminiClient
    if name == "MistralClient":
        from clients.mistral_client import MistralClient

        return MistralClient
    if name == "OllamaClient":
        from clients.ollama_client import OllamaClient

        return OllamaClient
    if name == "OpenAIClient":
        from clients.openai_client import OpenAIClient

        return OpenAIClient
    if name == "get_llm_client":
        from clients.llm_client import get_llm_client

        return get_llm_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ChatMessage",
    "GeminiClient",
    "LLMClientProtocol",
    "LLMModelKey",
    "MistralClient",
    "OllamaClient",
    "OpenAIClient",
    "get_llm_client",
]
