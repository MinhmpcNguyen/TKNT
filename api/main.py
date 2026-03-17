from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import pipeline, inventory
from config.logging_config import setup_logging

app = FastAPI(title="TKNT Pipeline API", version="0.2.0")


@app.on_event("startup")
def _startup_logging() -> None:
    # Ensure our internal logs (agents/RAG/search) show up when running via uvicorn.
    setup_logging()


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
