from __future__ import annotations

from typing import Any, Dict, List

from db.models import AssetFilter, DesignKnowledgeFilter, TenantId
from db.pg_repository import PostgresAssetRepository, PostgresDesignKnowledgeRepository
from stylist.room_essentials_seed import ROOM_SURFACE_GROUPS


def _stable_unique_strs(values: List[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        v = value.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _normalize_lookup_set(values: List[str] | None) -> set[str]:
    return {
        v.strip().lower() for v in (values or []) if isinstance(v, str) and v.strip()
    }


def _serialize_dimensions(asset: Any) -> dict[str, Any] | None:
    dims = getattr(asset, "dimensions", None)
    if dims is None:
        return None

    length_mm = getattr(dims, "length_mm", None)
    width_mm = getattr(dims, "width_mm", None)
    height_mm = getattr(dims, "height_mm", None)

    return {
        "length_mm": int(length_mm) if isinstance(length_mm, (int, float)) else None,
        "width_mm": int(width_mm) if isinstance(width_mm, (int, float)) else None,
        "height_mm": int(height_mm) if isinstance(height_mm, (int, float)) else None,
    }


def GetRoomSurfaceGroups(*, room_type: str) -> Dict[str, Any]:
    groups = ROOM_SURFACE_GROUPS.get(room_type)
    if groups is None:
        return {
            "error": "room_type_not_found",
            "room_type": room_type,
            "groups": {},
        }
    return {
        "room_type": room_type,
        "groups": groups,
    }


def ListInventoryByTypes(
    *,
    tenant_id: str | None,
    types: List[str],
    limit: int | None = 200,
) -> Dict[str, Any]:
    resolved_tenant = tenant_id or "demo_tenant"
    repo = PostgresAssetRepository()
    assets = list(repo.list_assets(AssetFilter(tenant_id=TenantId(resolved_tenant))))

    normalized = _normalize_lookup_set(types)
    results: list[dict[str, Any]] = []

    for asset in assets:
        asset_type = str(getattr(asset, "type", "") or "").strip()
        asset_name = str(getattr(asset, "name", "") or "").strip()
        attrs = dict(getattr(asset, "attributes", {}) or {})
        category = str(attrs.get("category") or asset_type).strip()

        if normalized:
            candidates = {
                asset_type.lower(),
                category.lower(),
                asset_name.lower(),
            }
            if candidates.isdisjoint(normalized):
                continue

        dims = _serialize_dimensions(asset)

        results.append(
            {
                "id": str(asset.id),
                "inventory_id": str(asset.id),
                "name": asset_name,
                "type": category,
                "asset_type": asset_type,
                "style_tags": list(getattr(asset, "style_tags", []) or []),
                "material": getattr(asset, "material", None),
                "brand": getattr(asset, "brand", None),
                "length_mm": dims["length_mm"] if isinstance(dims, dict) else None,
                "width_mm": dims["width_mm"] if isinstance(dims, dict) else None,
                "height_mm": dims["height_mm"] if isinstance(dims, dict) else None,
                "dimensions": dims,
                "attributes": attrs,
            }
        )

    results.sort(
        key=lambda x: (
            str(x.get("type") or ""),
            str(x.get("name") or ""),
            str(x.get("inventory_id") or ""),
        )
    )

    if isinstance(limit, int) and limit > 0:
        results = results[:limit]

    return {
        "tenant_id": resolved_tenant,
        "requested_types": _stable_unique_strs(types),
        "count": len(results),
        "items": results,
    }


def ListDesignKnowledge(
    *,
    tenant_id: str | None,
    tags: List[str] | None = None,
    category: str | None = None,
    limit: int | None = 20,
) -> Dict[str, Any]:
    repo = PostgresDesignKnowledgeRepository()
    normalized_tags = _stable_unique_strs(tags)

    # Ưu tiên tenant-specific trước, sau đó fallback global.
    filters: list[DesignKnowledgeFilter] = []
    if tenant_id:
        filters.append(
            DesignKnowledgeFilter(
                tenant_id=TenantId(tenant_id),
                tags=normalized_tags,
                category=category,
            )
        )

    filters.append(
        DesignKnowledgeFilter(
            tenant_id=None,
            tags=normalized_tags,
            category=category,
        )
    )

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for filter_obj in filters:
        items = list(repo.list_knowledge(filter_obj))
        items.sort(
            key=lambda item: (
                str(getattr(item, "category", "") or ""),
                str(getattr(item, "title", "") or ""),
                str(getattr(item, "id", "") or ""),
            )
        )

        for item in items:
            item_id = str(item.id)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            results.append(
                {
                    "id": item_id,
                    "title": item.title,
                    "content": item.content,
                    "category": item.category,
                    "tags": list(item.tags or []),
                    "source": item.source,
                    "meta": dict(item.meta or {}),
                }
            )

            if isinstance(limit, int) and limit > 0 and len(results) >= limit:
                return {
                    "tenant_id": tenant_id,
                    "tags": normalized_tags,
                    "category": category,
                    "count": len(results),
                    "items": results,
                }

    return {
        "tenant_id": tenant_id,
        "tags": normalized_tags,
        "category": category,
        "count": len(results),
        "items": results,
    }


TOOL_REGISTRY: Dict[str, Any] = {
    "GetRoomSurfaceGroups": GetRoomSurfaceGroups,
    "ListInventoryByTypes": ListInventoryByTypes,
    "ListDesignKnowledge": ListDesignKnowledge,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "GetRoomSurfaceGroups",
            "description": "Return surface grouping rules for a room type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {"type": "string"},
                },
                "required": ["room_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ListInventoryByTypes",
            "description": (
                "List inventory items filtered by eligible object types or categories for a tenant. "
                "Returns flattened dimensions (length_mm, width_mm, height_mm) and inventory_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tenant_id": {"type": ["string", "null"]},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": ["integer", "null"]},
                },
                "required": ["types"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ListDesignKnowledge",
            "description": (
                "List design knowledge entries for a tenant with tenant-specific priority and global fallback. "
                "Supports tags/category filtering and a result limit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tenant_id": {"type": ["string", "null"]},
                    "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                    "category": {"type": ["string", "null"]},
                    "limit": {"type": ["integer", "null"]},
                },
            },
        },
    },
]
