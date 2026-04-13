from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import account, auth, inventory, object_3d, pipeline
from config import root_config
from config.logging_config import setup_logging
from db.runtime_init import ensure_runtime_schema

app = FastAPI(title="TKNT Pipeline API", version="0.2.0")
logger = logging.getLogger(__name__)


@app.on_event("startup")
def _startup_logging() -> None:
    # Ensure our internal logs (agents/RAG/search) show up when running via uvicorn.
    setup_logging()
    _initialize_runtime_schema()
    _log_runtime_profile()


def _initialize_runtime_schema() -> None:
    try:
        ensure_runtime_schema()
    except Exception as exc:
        logger.exception("Runtime schema initialization skipped: %s", exc)


def _log_runtime_profile() -> None:
    gemini_config = root_config.services.gemini
    mistral_config = root_config.services.mistral
    ollama_config = root_config.services.ollama
    openai_config = root_config.services.openai
    semantic_enabled = root_config.services.semantic_search.enabled
    logged_provider = False

    if mistral_config.enabled and mistral_config.models is not None:
        logger.info(
            "LLM provider: Mistral | base_url=%s | primary=%s | helper=%s | embedding=%s",
            mistral_config.base_url,
            mistral_config.models.primary.name,
            mistral_config.models.helper.name,
            mistral_config.models.embedding.name,
        )
        logged_provider = True
    if ollama_config.enabled and ollama_config.models is not None:
        logger.info(
            "LLM provider: Ollama | base_url=%s | primary=%s | helper=%s | embedding=%s",
            ollama_config.base_url,
            ollama_config.models.primary.name,
            ollama_config.models.helper.name,
            ollama_config.models.embedding.name,
        )
        logged_provider = True
    if gemini_config.enabled and gemini_config.models is not None:
        logger.info(
            "LLM provider: Gemini | base_url=%s | primary=%s | helper=%s | embedding=%s",
            gemini_config.base_url,
            gemini_config.models.primary.name,
            gemini_config.models.helper.name,
            gemini_config.models.embedding.name,
        )
        logged_provider = True
    if not logged_provider and openai_config.azure:
        logger.info(
            "LLM provider: Azure OpenAI | primary=%s | helper=%s | embedding=%s",
            openai_config.models.gpt_4o.name,
            openai_config.models.gpt_4o_mini.name,
            openai_config.models.text_embedding_3_small.name,
        )
    elif not logged_provider:
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
app.include_router(auth.router)
app.include_router(account.router)
app.include_router(inventory.router)
app.include_router(object_3d.router)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}
