from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.deps import get_current_user
from db.models import JsonValue, UserAccount
from services.user_content_service import UserContentService

router = APIRouter(prefix="/account", tags=["account"])


class SavedLayoutRecord(BaseModel):
    id: str
    name: str
    floorplan_json: dict[str, Any]
    design_json: dict[str, Any] | None = None
    styled_result_json: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class SavedLayoutCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    floorplan_json: dict[str, Any]
    design_json: dict[str, Any] | None = None
    styled_result_json: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class GeneratedRenderRecord(BaseModel):
    id: str
    source: str
    model_name: str
    prompt: str
    negative_prompt: str | None = None
    mime_type: str
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    file_url: str


def get_user_content_service() -> UserContentService:
    return UserContentService()


@router.get("/layouts", response_model=list[SavedLayoutRecord])
def list_layouts(
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> list[SavedLayoutRecord]:
    return [
        SavedLayoutRecord(**service.serialize_saved_layout(layout))
        for layout in service.list_layouts(user=user)
    ]


@router.post("/layouts", response_model=SavedLayoutRecord)
def create_layout(
    request: SavedLayoutCreateRequest,
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> SavedLayoutRecord:
    layout = service.save_layout(
        user=user,
        name=request.name,
        floorplan_json=_coerce_json_dict(request.floorplan_json),
        design_json=_coerce_json_dict(request.design_json)
        if request.design_json is not None
        else None,
        styled_result_json=_coerce_json_dict(request.styled_result_json)
        if request.styled_result_json is not None
        else None,
        meta=_coerce_json_dict(request.meta),
    )
    return SavedLayoutRecord(**service.serialize_saved_layout(layout))


@router.get("/layouts/{layout_id}", response_model=SavedLayoutRecord)
def get_layout(
    layout_id: str,
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> SavedLayoutRecord:
    layout = service.get_layout(user=user, layout_id=layout_id)
    if layout is None:
        raise HTTPException(status_code=404, detail="Layout not found.")
    return SavedLayoutRecord(**service.serialize_saved_layout(layout))


@router.get("/renders", response_model=list[GeneratedRenderRecord])
def list_renders(
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> list[GeneratedRenderRecord]:
    return [
        GeneratedRenderRecord(
            **service.serialize_generated_render(
                render,
                file_url=f"/account/renders/{render.id}/file",
            )
        )
        for render in service.list_generated_renders(user=user)
    ]


@router.get("/renders/{render_id}/file")
def get_render_file(
    render_id: str,
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> FileResponse:
    render = service.get_generated_render(user=user, render_id=render_id)
    if render is None:
        raise HTTPException(status_code=404, detail="Render not found.")
    return FileResponse(
        render.storage_path,
        media_type=render.mime_type,
        filename=f"{render.id}",
    )


def _coerce_json_dict(value: dict[str, Any]) -> dict[str, JsonValue]:
    return {str(key): item for key, item in value.items()}
