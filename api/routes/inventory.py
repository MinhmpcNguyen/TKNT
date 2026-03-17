from __future__ import annotations

from typing import Any
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/inventory", tags=["inventory"])


class InventoryItem(BaseModel):
    id: str
    name: str
    type: str
    style_tags: list[str] = Field(default_factory=list)
    material: str | None = None
    brand: str | None = None
    dimensions: dict[str, float | None] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class InventoryListResponse(BaseModel):
    tenant_id: str
    items: list[InventoryItem]


class InventoryTypesResponse(BaseModel):
    tenant_id: str
    types: list[str]


class InventorySearchResponse(BaseModel):
    tenant_id: str
    query: str
    items: list[InventoryItem]


@router.get("/items", response_model=InventoryListResponse)
def list_items(
    tenant_id: str | None = Query(default=None),
    types: list[str] | None = Query(default=None),
    style_tags: list[str] | None = Query(default=None),
) -> InventoryListResponse:
    resolved_tenant = tenant_id or "demo_tenant"
    items = _load_inventory_items(
        tenant_id=resolved_tenant,
        types=types,
        style_tags=style_tags,
    )
    return InventoryListResponse(tenant_id=resolved_tenant, items=items)


@router.get("/types", response_model=InventoryTypesResponse)
def list_types(tenant_id: str | None = Query(default=None)) -> InventoryTypesResponse:
    resolved_tenant = tenant_id or "demo_tenant"
    items = _load_inventory_items(tenant_id=resolved_tenant)
    types_set = {item.type for item in items}
    return InventoryTypesResponse(tenant_id=resolved_tenant, types=sorted(types_set))


@router.get("/search", response_model=InventorySearchResponse)
def search_items(
    q: str = Query(..., min_length=1),
    tenant_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
) -> InventorySearchResponse:
    resolved_tenant = tenant_id or "demo_tenant"
    items = _load_inventory_items(tenant_id=resolved_tenant)
    query = q.lower()
    results: list[InventoryItem] = []
    for item in items:
        hay = " ".join(
            [
                item.name,
                item.type,
                item.brand or "",
                item.material or "",
                " ".join(item.style_tags or []),
            ]
        ).lower()
        if query in hay:
            results.append(item)
        if len(results) >= limit:
            break
    return InventorySearchResponse(tenant_id=resolved_tenant, query=q, items=results)


def _load_inventory_items(
    *,
    tenant_id: str,
    types: list[str] | None = None,
    style_tags: list[str] | None = None,
) -> list[InventoryItem]:
    try:
        from db.pg_repository import PostgresAssetRepository
        from db.models import AssetFilter, TenantId

        repo = PostgresAssetRepository()
        assets = repo.list_assets(
            AssetFilter(
                tenant_id=TenantId(tenant_id),
                style_tags=list(style_tags or []),
            )
        )
        type_set = {t for t in (types or []) if isinstance(t, str)}
        results: list[InventoryItem] = []
        for asset in assets:
            category = str((asset.attributes or {}).get("category") or asset.name)
            if type_set and category not in type_set:
                continue
            results.append(
                InventoryItem(
                    id=str(asset.id),
                    name=asset.name,
                    type=category,
                    style_tags=list(asset.style_tags or []),
                    material=asset.material,
                    brand=asset.brand,
                    dimensions=asset.dimensions.model_dump() if asset.dimensions else None,
                    attributes=dict(asset.attributes or {}),
                )
            )
        return results
    except Exception:
        # Fallback to local JSON (synthetic_data/inventory.json)
        base = Path(__file__).resolve().parents[2]
        path = base / "synthetic_data" / "inventory.json"
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        type_set = {t for t in (types or []) if isinstance(t, str)}
        style_set = {t for t in (style_tags or []) if isinstance(t, str)}
        results: list[InventoryItem] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            if item.get("tenant_id") not in (tenant_id, None, "demo_tenant"):
                continue
            category = str(item.get("type") or item.get("name") or "")
            if type_set and category not in type_set:
                continue
            item_styles = set(item.get("style_tags") or [])
            if style_set and not (style_set & item_styles):
                continue
            dimensions = {
                "length_mm": item.get("length_mm"),
                "width_mm": item.get("width_mm"),
                "height_mm": item.get("height_mm"),
            }
            results.append(
                InventoryItem(
                    id=str(item.get("id") or ""),
                    name=str(item.get("name") or category),
                    type=category,
                    style_tags=list(item.get("style_tags") or []),
                    material=item.get("material"),
                    brand=item.get("brand"),
                    dimensions=dimensions,
                    attributes=dict(item.get("attributes") or {}),
                )
            )
        return results
