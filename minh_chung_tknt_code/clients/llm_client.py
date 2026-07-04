from __future__ import annotations

from clients.base_client import LLMClientProtocol


def get_llm_client() -> LLMClientProtocol:
    from clients.openai_client import OpenAIClient

    return OpenAIClient()
