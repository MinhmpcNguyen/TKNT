from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from queue import Queue
from threading import Thread
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import get_optional_current_user
from api.routes.snapshot_image_support import (
    SnapshotImagePresetOptionResponse,
    SnapshotImagePresetsResponse,
    SnapshotImageRenderRequest,
    SnapshotImageRenderResponse,
    _raise_snapshot_render_http_exception,
    _render_snapshot_image_for_request,
    _resolve_render_layout_context_or_raise,
    _save_snapshot_render_for_request,
)
from config import config_file, load_config
from db.models import UserAccount
from pipeline.image_flow_logging import log_image_flow_event
from services.user_content_service import UserContentService

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)


def get_user_content_service() -> UserContentService:
    return UserContentService()


@router.post("/render-snapshot-image", response_model=SnapshotImageRenderResponse)
def render_snapshot_image_route(
    req: SnapshotImageRenderRequest,
    current_user: UserAccount | None = Depends(get_optional_current_user),
    user_content_service: UserContentService = Depends(get_user_content_service),
) -> SnapshotImageRenderResponse:
    trace_id = uuid4().hex
    log_image_flow_event(
        "snapshot_image.api_input",
        {
            "trace_id": trace_id,
            "route": "/pipeline/render-snapshot-image",
            "authenticated": current_user is not None,
            "request": req.model_dump(),
        },
    )
    render_layout_context = _resolve_render_layout_context_or_raise(
        req=req,
        current_user=current_user,
        user_content_service=user_content_service,
        trace_id=trace_id,
    )
    try:
        render = _render_snapshot_image_for_request(
            req,
            trace_id=trace_id,
            streamed=False,
        )
    except HTTPException:
        raise
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _raise_snapshot_render_http_exception(trace_id, exc)

    saved_render = _save_snapshot_render_for_request(
        render=render,
        current_user=current_user,
        render_layout_context=render_layout_context,
        user_content_service=user_content_service,
        trace_id=trace_id,
    )
    return SnapshotImageRenderResponse(render=render, saved_render=saved_render)


def _encode_sse_event(event_name: str, payload: Mapping[str, Any]) -> str:
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
    )


@router.post("/render-snapshot-image/stream")
def render_snapshot_image_stream_route(
    req: SnapshotImageRenderRequest,
    current_user: UserAccount | None = Depends(get_optional_current_user),
    user_content_service: UserContentService = Depends(get_user_content_service),
) -> StreamingResponse:
    trace_id = uuid4().hex
    log_image_flow_event(
        "snapshot_image.api_input",
        {
            "trace_id": trace_id,
            "route": "/pipeline/render-snapshot-image/stream",
            "authenticated": current_user is not None,
            "request": req.model_dump(),
        },
    )
    event_queue: Queue[dict[str, Any] | None] = Queue()

    def enqueue(event_name: str, payload: Mapping[str, Any]) -> None:
        event_queue.put({"event": event_name, "data": dict(payload)})

    def forward_renderer_event(event: Mapping[str, object]) -> None:
        event_name = str(event.get("type") or "status")
        payload = {key: value for key, value in event.items() if key != "type"}
        enqueue(event_name, payload)

    def worker() -> None:
        try:
            enqueue("status", {"message": "Preparing image render.", "attempt": None})
            render_layout_context = _resolve_render_layout_context_or_raise(
                req=req,
                current_user=current_user,
                user_content_service=user_content_service,
                trace_id=trace_id,
            )
            render = _render_snapshot_image_for_request(
                req,
                trace_id=trace_id,
                streamed=True,
                stream_event_handler=forward_renderer_event,
            )
            saved_render = _save_snapshot_render_for_request(
                render=render,
                current_user=current_user,
                render_layout_context=render_layout_context,
                user_content_service=user_content_service,
                trace_id=trace_id,
            )
            enqueue(
                "completed",
                SnapshotImageRenderResponse(
                    render=render,
                    saved_render=saved_render,
                ).model_dump(),
            )
        except HTTPException as exc:
            enqueue("error", {"message": str(exc.detail)})
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            log_image_flow_event(
                "snapshot_image.api_error",
                {
                    "trace_id": trace_id,
                    "route": "/pipeline/render-snapshot-image/stream",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            enqueue("error", {"message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Snapshot image stream failed: %s", exc)
            log_image_flow_event(
                "snapshot_image.api_error",
                {
                    "trace_id": trace_id,
                    "route": "/pipeline/render-snapshot-image/stream",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            enqueue("error", {"message": str(exc)})
        finally:
            event_queue.put(None)

    def event_stream():
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield _encode_sse_event(
                str(item["event"]),
                item["data"] if isinstance(item["data"], dict) else {},
            )

    Thread(target=worker, daemon=True).start()
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/render-presets", response_model=SnapshotImagePresetsResponse)
def get_render_presets() -> SnapshotImagePresetsResponse:
    presets = load_config(config_file).presets
    return SnapshotImagePresetsResponse(
        styles={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.styles.items()
        },
        lights={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.lights.items()
        },
        sceneries={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.sceneries.items()
        },
    )
