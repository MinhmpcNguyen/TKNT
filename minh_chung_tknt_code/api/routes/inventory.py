from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from db.models import AssetFile
from db.pg_assets import PostgresAssetRepository
from services.auth_service import get_shared_inventory_tenant_id
from services.user_content_service import UserContentService

router = APIRouter(prefix="/inventory", tags=["inventory"])


class InventoryItem(BaseModel):
    id: str
    name: str
    type: str
    style_tags: list[str] = Field(default_factory=list)
    material: str | None = None
    brand: str | None = None
    dimensions: dict[str, float | None] | None = None
    price: dict[str, object] | None = None
    attributes: dict[str, object] = Field(default_factory=dict)


class InventoryListResponse(BaseModel):
    tenant_id: str
    items: list[InventoryItem]


def get_user_content_service() -> UserContentService:
    return UserContentService()


def get_asset_repository() -> PostgresAssetRepository:
    return PostgresAssetRepository()


@router.get("/items", response_model=InventoryListResponse)
def list_items(
    types: list[str] | None = Query(default=None),
    style_tags: list[str] | None = Query(default=None),
    service: UserContentService = Depends(get_user_content_service),
) -> InventoryListResponse:
    shared_tenant = get_shared_inventory_tenant_id()
    items = _load_inventory_items(
        types=types,
        style_tags=style_tags,
        service=service,
    )
    return InventoryListResponse(tenant_id=shared_tenant, items=items)


@router.get("/files/{asset_file_id}")
def get_inventory_file(
    asset_file_id: str,
    service: UserContentService = Depends(get_user_content_service),
    asset_repository: PostgresAssetRepository = Depends(get_asset_repository),
) -> FileResponse:
    asset_file = asset_repository.get_asset_file(asset_file_id)
    if asset_file is None:
        raise HTTPException(status_code=404, detail="Asset file not found.")
    asset = asset_repository.get_asset(asset_file.asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    if not service.can_access_asset(asset=asset):
        raise HTTPException(status_code=403, detail="Access denied.")

    file_path = Path(asset_file.storage_key).expanduser().resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Asset file is missing on disk.")

    media_type, _ = mimetypes.guess_type(file_path.name)
    if file_path.suffix.lower() == ".glb":
        media_type = "model/gltf-binary"
    return FileResponse(
        file_path,
        media_type=media_type or "application/octet-stream",
        filename=file_path.name,
    )


def _load_inventory_items(
    *,
    types: list[str] | None = None,
    style_tags: list[str] | None = None,
    service: UserContentService,
) -> list[InventoryItem]:
    persisted_assets = service.list_shared_inventory_assets(
        style_tags=style_tags,
    )
    type_set = {value for value in (types or []) if isinstance(value, str) and value}
    results: list[InventoryItem] = []
    for persisted in persisted_assets:
        payload = service.serialize_inventory_asset(
            persisted=persisted,
            file_url_builder=_build_file_url,
        )
        category = str(payload.get("type") or payload.get("name") or "")
        if type_set and category not in type_set:
            continue
        results.append(InventoryItem(**payload))
    return results


def _build_file_url(asset_file: AssetFile) -> str:
    return f"/inventory/files/{asset_file.id}"
