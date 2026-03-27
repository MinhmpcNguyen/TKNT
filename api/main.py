from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import pipeline, inventory
from config import root_config
from config.logging_config import setup_logging

app = FastAPI(title="TKNT Pipeline API", version="0.2.0")
logger = logging.getLogger(__name__)


@app.on_event("startup")
def _startup_logging() -> None:
    # Ensure our internal logs (agents/RAG/search) show up when running via uvicorn.
    setup_logging()
    _log_runtime_profile()


def _log_runtime_profile() -> None:
    mistral_config = root_config.services.mistral
    ollama_config = root_config.services.ollama
    openai_config = root_config.services.openai
    semantic_enabled = root_config.services.semantic_search.enabled

    if mistral_config.enabled and mistral_config.models is not None:
        logger.info(
            "LLM provider: Mistral | base_url=%s | primary=%s | helper=%s | embedding=%s",
            mistral_config.base_url,
            mistral_config.models.primary.name,
            mistral_config.models.helper.name,
            mistral_config.models.embedding.name,
        )
    elif ollama_config.enabled and ollama_config.models is not None:
        logger.info(
            "LLM provider: Ollama | base_url=%s | primary=%s | helper=%s | embedding=%s",
            ollama_config.base_url,
            ollama_config.models.primary.name,
            ollama_config.models.helper.name,
            ollama_config.models.embedding.name,
        )
    elif openai_config.azure:
        logger.info(
            "LLM provider: Azure OpenAI | primary=%s | helper=%s | embedding=%s",
            openai_config.models.gpt_4o.name,
            openai_config.models.gpt_4o_mini.name,
            openai_config.models.text_embedding_3_small.name,
        )
    else:
        logger.info(
            "LLM provider: OpenAI-compatible API | primary=%s | helper=%s | embedding=%s",
            openai_config.models.gpt_4o.name,
            openai_config.models.gpt_4o_mini.name,
            openai_config.models.text_embedding_3_small.name,
        )

    if semantic_enabled:
        logger.info("Knowledge retrieval: semantic search enabled.")
    else:
        logger.info(
            "Knowledge retrieval: semantic search disabled; lexical fallback search enabled."
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(inventory.router)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}
