from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

from clients.base_client import LLMClientProtocol, LLMModelKey
from clients.llm_client import get_llm_client
from config import root_config
from db import PostgresAssetRepository, PostgresDesignKnowledgeRepository
from db.models import (
    Asset,
    AssetDimensions,
    AssetEmbedding,
    AssetFilter,
    AssetPrice,
    DesignKnowledge,
    DesignKnowledgeEmbedding,
    DesignKnowledgeFilter,
    TenantId,
)
from db.postgres import PostgresConfig, create_connection
from db.schema import get_schema_statements
from stylist.room_essentials_loader import load_room_essentials as load_room_essentials_kb


def ensure_schema(connection_factory) -> None:
    with connection_factory() as conn:
        with conn.cursor() as cur:
            for statement in get_schema_statements():
                cur.execute(statement)


def load_inventory(path: Path, connection_factory) -> int:
    items = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    repo = PostgresAssetRepository(connection_factory=connection_factory)
    for item in items:
        attributes = dict(item.get("attributes") or {})
        attributes.setdefault("category", item.get("type"))
        attributes.setdefault("source", "synthetic_data")
        if "size_tier" in item:
            attributes["size_tier"] = item.get("size_tier")
        if "footprint_area_m2" in item:
            attributes["footprint_area_m2"] = item.get("footprint_area_m2")
        if "length_mm" in item:
            attributes["length_mm"] = item.get("length_mm")
        if "width_mm" in item:
            attributes["width_mm"] = item.get("width_mm")
        if "height_mm" in item:
            attributes["height_mm"] = item.get("height_mm")

        length_mm = item.get("length_mm")
        width_mm = item.get("width_mm")
        height_mm = item.get("height_mm")
        dimensions = AssetDimensions(
            length_mm=length_mm,
            width_mm=width_mm,
            height_mm=height_mm,
        )
        price_amount = item.get("price_amount")
        price_currency = item.get("price_currency")
        price = (
            AssetPrice(amount=float(price_amount), currency=price_currency or "VND")
            if price_amount is not None
            else None
        )

        asset = Asset(
            id=item["id"],
            tenant_id=item.get("tenant_id", "demo_tenant"),
            type="FURNITURE",
            name=item.get("name", item["id"]),
            style_tags=item.get("style_tags", []),
            material=item.get("material"),
            brand=item.get("brand"),
            dimensions=dimensions,
            price=price,
            attributes=attributes,
        )
        repo.upsert_asset(asset)
        count += 1
    return count


def load_templates(path: Path, connection_factory) -> int:
    templates = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    repo = PostgresAssetRepository(connection_factory=connection_factory)
    for template in templates:
        attributes = {
            "room_type": template.get("room_type"),
            "supported_shapes": template.get("supported_shapes", ["rect"]),
            "required_zones": template.get("zones", []),
            "required_types": template.get("required_types", []),
            "recommended_types": template.get("recommended_types", []),
            "layout_rules": template.get("layout_rules", []),
            "source": "synthetic_data",
        }
        asset = Asset(
            id=template["template_id"],
            tenant_id=template.get("tenant_id", "demo_tenant"),
            type="TEMPLATE",
            name=template.get("name", template["template_id"]),
            style_tags=template.get("style_tags", []),
            attributes=attributes,
        )
        repo.upsert_asset(asset)
        count += 1
    return count


def load_design_kb(path: Path, connection_factory) -> int:
    count = 0
    repo = PostgresDesignKnowledgeRepository(connection_factory=connection_factory)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        tags = doc.get("tags", [])
        category = tags[0] if tags else None
        knowledge = DesignKnowledge(
            id=doc["doc_id"],
            tenant_id=None,
            title=doc.get("title", doc["doc_id"]),
            content=doc.get("text", ""),
            category=category,
            tags=tags,
            source="synthetic_data",
            meta={"raw": doc},
        )
        repo.upsert_knowledge(knowledge)
        count += 1
    return count


def _asset_text(asset: Asset) -> str:
    parts: list[str] = [asset.name]
    category = asset.attributes.get("category")
    if category:
        parts.append(f"category: {category}")
    if asset.style_tags:
        parts.append(f"style_tags: {', '.join(asset.style_tags)}")
    if asset.material:
        parts.append(f"material: {asset.material}")
    if asset.brand:
        parts.append(f"brand: {asset.brand}")
    dims = asset.dimensions
    if dims:
        parts.append(
            f"dims_mm: {dims.length_mm or ''}x{dims.width_mm or ''}x{dims.height_mm or ''}"
        )
    return " | ".join([part for part in parts if part])


def _knowledge_text(item: DesignKnowledge) -> str:
    parts = [item.title, item.content]
    return "\n".join([part for part in parts if part])


def _embed_batch(
    client: LLMClientProtocol,
    texts: list[str],
    model_key: LLMModelKey,
) -> list[list[float]]:
    response = client.embeddings(inputs=texts, model_key=model_key)
    data = getattr(response, "data", None)
    if not isinstance(data, list):
        return []
    vectors: list[list[float]] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if isinstance(embedding, list):
            vectors.append([float(val) for val in embedding])
    return vectors


def _backfill_embeddings(batch_size: int = 32) -> tuple[int, int]:
    if not root_config.services.semantic_search.enabled:
        return 0, 0

    client = get_llm_client()
    asset_repo = PostgresAssetRepository()
    knowledge_repo = PostgresDesignKnowledgeRepository()
    model_key: LLMModelKey = "embedding"

    tenant_id = TenantId("demo_tenant")
    assets = list(asset_repo.list_assets(AssetFilter(tenant_id=tenant_id)))
    asset_texts = [_asset_text(asset) for asset in assets]
    asset_count = 0
    for idx in range(0, len(asset_texts), batch_size):
        chunk = asset_texts[idx : idx + batch_size]
        vectors = _embed_batch(client, chunk, model_key)
        for asset, vector in zip(assets[idx : idx + batch_size], vectors, strict=False):
            embedding = AssetEmbedding(
                asset_id=asset.id,
                content=_asset_text(asset),
                vector=vector,
                model=client.get_model_name(model_key),
                meta={"source": "auto_embed_load"},
            )
            asset_repo.upsert_embedding(embedding)
            asset_count += 1

    knowledge_items = list(knowledge_repo.list_knowledge(DesignKnowledgeFilter()))
    knowledge_texts = [_knowledge_text(item) for item in knowledge_items]
    knowledge_count = 0
    for idx in range(0, len(knowledge_texts), batch_size):
        chunk = knowledge_texts[idx : idx + batch_size]
        vectors = _embed_batch(client, chunk, model_key)
        for item, vector in zip(
            knowledge_items[idx : idx + batch_size], vectors, strict=False
        ):
            embedding = DesignKnowledgeEmbedding(
                knowledge_id=item.id,
                content=_knowledge_text(item),
                vector=vector,
                model=client.get_model_name(model_key),
                meta={"source": "auto_embed_load"},
            )
            knowledge_repo.upsert_knowledge_embedding(embedding)
            knowledge_count += 1

    return asset_count, knowledge_count


def _build_schema_connection_factory():
    admin_config = _load_admin_target_config()
    if admin_config is None:
        return create_connection
    return lambda: create_connection(admin_config)


def _load_admin_target_config() -> PostgresConfig | None:
    admin_user = os.getenv("PG_ADMIN_USER")
    admin_host = os.getenv("PG_ADMIN_HOST")
    admin_port = os.getenv("PG_ADMIN_PORT")
    target_db = os.getenv("POSTGRES_DB")
    if not admin_user or not admin_host or not admin_port or not target_db:
        return None
    admin_password = os.getenv("PG_ADMIN_PASSWORD", "")
    return PostgresConfig(
        host=admin_host,
        port=int(admin_port),
        database=target_db,
        user=admin_user,
        password=admin_password,
    )


def main() -> None:
    load_dotenv()
    schema_connection_factory = _build_schema_connection_factory()
    ensure_schema(schema_connection_factory)

    data_connection_factory = create_connection
    inventory_count = load_inventory(BASE_DIR / "inventory.json", data_connection_factory)
    template_count = load_templates(BASE_DIR / "templates.json", data_connection_factory)
    kb_count = load_design_kb(BASE_DIR / "design_kb.jsonl", data_connection_factory)
    room_essentials_count = load_room_essentials_kb(
        PostgresDesignKnowledgeRepository(connection_factory=data_connection_factory)
    )

    print(
        f"Loaded inventory={inventory_count}, templates={template_count}, "
        f"knowledge={kb_count}, room_essentials={room_essentials_count}"
    )

    if os.getenv("AUTO_EMBED_ON_LOAD", "1").lower() in {"1", "true", "yes", "on"}:
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
        asset_count, knowledge_embed_count = _backfill_embeddings(batch_size=batch_size)
        print(
            f"Embeddings created: assets={asset_count}, knowledge={knowledge_embed_count}"
        )


if __name__ == "__main__":
    main()
