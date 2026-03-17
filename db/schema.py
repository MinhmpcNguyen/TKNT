from __future__ import annotations

from config.settings import settings

VECTOR_DIMENSIONS = settings.VECTOR_DIMS


def get_schema_statements() -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        """
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            style_tags TEXT[] NOT NULL DEFAULT '{}',
            material TEXT,
            brand TEXT,
            length_mm DOUBLE PRECISION,
            width_mm DOUBLE PRECISION,
            height_mm DOUBLE PRECISION,
            price_amount DOUBLE PRECISION,
            price_currency TEXT,
            attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        "ALTER TABLE assets ADD COLUMN IF NOT EXISTS length_mm DOUBLE PRECISION;",
        "ALTER TABLE assets ADD COLUMN IF NOT EXISTS width_mm DOUBLE PRECISION;",
        "ALTER TABLE assets ADD COLUMN IF NOT EXISTS height_mm DOUBLE PRECISION;",
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'assets' AND column_name = 'length_cm'
            ) THEN
                UPDATE assets
                SET length_mm = length_cm * 10
                WHERE length_mm IS NULL AND length_cm IS NOT NULL;
            END IF;
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'assets' AND column_name = 'width_cm'
            ) THEN
                UPDATE assets
                SET width_mm = width_cm * 10
                WHERE width_mm IS NULL AND width_cm IS NOT NULL;
            END IF;
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'assets' AND column_name = 'height_cm'
            ) THEN
                UPDATE assets
                SET height_mm = height_cm * 10
                WHERE height_mm IS NULL AND height_cm IS NOT NULL;
            END IF;
        END $$;
        """,
        "ALTER TABLE assets DROP COLUMN IF EXISTS length_cm;",
        "ALTER TABLE assets DROP COLUMN IF EXISTS width_cm;",
        "ALTER TABLE assets DROP COLUMN IF EXISTS height_cm;",
        """
        CREATE TABLE IF NOT EXISTS asset_files (
            id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            file_kind TEXT NOT NULL,
            provider TEXT NOT NULL,
            storage_key TEXT NOT NULL,
            mime TEXT,
            width_px INTEGER,
            height_px INTEGER,
            bytes_size BIGINT,
            role TEXT,
            meta JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS asset_embeddings (
            asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            model TEXT NOT NULL,
            meta JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            vector VECTOR({VECTOR_DIMENSIONS}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (asset_id, model)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS design_knowledge (
            id TEXT PRIMARY KEY,
            tenant_id TEXT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT,
            tags TEXT[] NOT NULL DEFAULT '{}',
            source TEXT,
            meta JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS design_knowledge_embeddings (
            knowledge_id TEXT NOT NULL REFERENCES design_knowledge(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            model TEXT NOT NULL,
            meta JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            vector VECTOR({VECTOR_DIMENSIONS}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (knowledge_id, model)
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_assets_tenant ON assets (tenant_id);",
        "CREATE INDEX IF NOT EXISTS idx_assets_style_tags ON assets USING GIN (style_tags);",
        "CREATE INDEX IF NOT EXISTS idx_assets_attributes ON assets USING GIN (attributes);",
        "CREATE INDEX IF NOT EXISTS idx_asset_files_asset_id ON asset_files (asset_id);",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON asset_embeddings USING HNSW (vector vector_l2_ops);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_tenant ON design_knowledge (tenant_id);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON design_knowledge USING GIN (tags);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_category ON design_knowledge (category);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_vector ON design_knowledge_embeddings USING HNSW (vector vector_l2_ops);",
    ]


def get_schema_notes() -> list[str]:
    return [
        "Change VECTOR_DIMENSIONS to match your embedding model.",
        "Use TEXT ids to keep the schema tool-agnostic; the app should provide ids.",
        "If you prefer UUIDs, switch id fields to UUID and add a UUID extension.",
        "design_knowledge.tenant_id is optional to support shared/global knowledge.",
    ]
