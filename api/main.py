from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import (
    account,
    admin,
    auth,
    billing,
    inventory,
    pipeline,
    snapshot_images,
)
from config import root_config
from config.logging_config import setup_logging
from config.models import OpenAIModelGroupConfig
from db.runtime_init import ensure_runtime_schema
from db.shared_inventory_bootstrap import ensure_shared_inventory_loaded

app = FastAPI(title="TKNT Pipeline API", version="0.2.0")
logger = logging.getLogger(__name__)


@app.on_event("startup")
def _startup_logging() -> None:
    # Ensure internal logs are configured when running via uvicorn.
    setup_logging()
    _initialize_runtime_schema()
    _log_runtime_profile()


def _initialize_runtime_schema() -> None:
    ensure_runtime_schema()
    loaded_count = ensure_shared_inventory_loaded()
    if loaded_count:
        logger.info("Loaded %d bundled shared inventory assets.", loaded_count)


def _log_runtime_profile() -> None:
    openai_config = root_config.services.openai
    semantic_enabled = root_config.services.semantic_search.enabled
    provider_label = "Azure OpenAI" if openai_config.azure else "OpenAI"
    primary_model, helper_model, embedding_model = _openai_model_names(
        openai_config.models
    )
    logger.info(
        "LLM provider: %s | primary=%s | helper=%s | embedding=%s",
        provider_label,
        primary_model,
        helper_model,
        embedding_model,
    )

    if semantic_enabled:
        logger.info("Knowledge retrieval: semantic search enabled.")
    else:
        logger.info("Knowledge retrieval: semantic search disabled.")


def _openai_model_names(models: OpenAIModelGroupConfig) -> tuple[str, str, str]:
    resolved_models = models.resolved_models()
    return (
        resolved_models["primary"].name,
        resolved_models["helper"].name,
        resolved_models["embedding"].name,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(snapshot_images.router)
app.include_router(auth.router)
app.include_router(billing.router)
app.include_router(admin.router)
app.include_router(account.router)
app.include_router(inventory.router)

assets_gen_dir = Path(__file__).resolve().parents[1] / "assets_gen"
if assets_gen_dir.exists():
    app.mount(
        "/assets_gen",
        StaticFiles(directory=assets_gen_dir),
        name="assets_gen",
    )


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}
