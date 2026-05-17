from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

LOGGER = logging.getLogger(__name__)
DEFAULT_TENANT_ID = "demo_tenant"
DEFAULT_ASSET_TYPE = "FURNITURE"

MISSING_GLB_ASSETS_SQL = """
WITH scoped_assets AS (
    SELECT *
    FROM assets
    WHERE tenant_id = %(tenant_id)s
      AND type = %(asset_type)s
),
flags AS (
    SELECT
        a.id,
        a.name,
        COALESCE(a.attributes->>'category', a.type) AS category,
        EXISTS (
            SELECT 1
            FROM asset_files af
            WHERE af.asset_id = a.id
              AND (
                  LOWER(af.storage_key) LIKE '%%.glb%%'
                  OR LOWER(COALESCE(af.mime, '')) = 'model/gltf-binary'
              )
        ) AS has_asset_file_glb,
        (
            LOWER(COALESCE(a.attributes#>>'{model_3d,url}', '')) LIKE '%%.glb%%'
            OR LOWER(COALESCE(a.attributes#>>'{model_3d,local_path}', '')) LIKE '%%.glb%%'
            OR LOWER(COALESCE(a.attributes#>>'{model_3d,storage_key}', '')) LIKE '%%.glb%%'
            OR LOWER(COALESCE(a.attributes->>'model_url', '')) LIKE '%%.glb%%'
        ) AS has_model_attr_glb,
        EXISTS (
            SELECT 1
            FROM jsonb_each(
                CASE
                    WHEN jsonb_typeof(a.attributes->'model_variants') = 'object'
                    THEN a.attributes->'model_variants'
                    ELSE '{}'::jsonb
                END
            ) AS mv(style_key, variant)
            WHERE LOWER(COALESCE(variant->>'url', '')) LIKE '%%.glb%%'
               OR LOWER(COALESCE(variant->>'local_path', '')) LIKE '%%.glb%%'
               OR LOWER(COALESCE(variant->>'storage_key', '')) LIKE '%%.glb%%'
        ) AS has_variant_glb
    FROM scoped_assets a
)
SELECT
    id,
    name,
    category,
    has_asset_file_glb,
    has_model_attr_glb,
    has_variant_glb
FROM flags
WHERE NOT (has_asset_file_glb OR has_model_attr_glb OR has_variant_glb)
ORDER BY category, id
"""

DELETE_ASSETS_SQL = """
DELETE FROM assets
WHERE tenant_id = %(tenant_id)s
  AND type = %(asset_type)s
  AND id = ANY(%(asset_ids)s::text[])
RETURNING id
"""

SIZE_RULES_EXISTS_SQL = """
SELECT id
FROM design_knowledge
WHERE id = %(knowledge_id)s
LIMIT 1
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete tenant furniture assets that do not have any GLB reference in "
            "asset_files, attributes.model_3d/model_url, or attributes.model_variants."
        )
    )
    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT_ID,
        help="Tenant whose assets should be pruned.",
    )
    parser.add_argument(
        "--asset-type",
        default=DEFAULT_ASSET_TYPE,
        help="Asset type scope. Defaults to FURNITURE.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete candidates. Without this flag, the script is a dry run.",
    )
    parser.add_argument(
        "--allow-without-size-rules",
        action="store_true",
        help="Allow deletion even if size_rules:<tenant> has not been frozen.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def default_report_path(tenant_id: str, *, delete: bool) -> Path:
    safe_tenant = tenant_id.replace("/", "_").replace(" ", "_")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    mode = "deleted" if delete else "dry_run"
    return (
        BASE_DIR / "outputs" / f"assets_without_glb_{safe_tenant}_{mode}_{stamp}.json"
    )


def size_rules_exist(conn: Any, tenant_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(SIZE_RULES_EXISTS_SQL, {"knowledge_id": f"size_rules:{tenant_id}"})
        return cur.fetchone() is not None


def list_assets_without_glb(
    conn: Any,
    *,
    tenant_id: str,
    asset_type: str,
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            MISSING_GLB_ASSETS_SQL,
            {"tenant_id": tenant_id, "asset_type": asset_type},
        )
        return [dict(row) for row in cur.fetchall()]


def delete_assets(
    conn: Any,
    *,
    tenant_id: str,
    asset_type: str,
    asset_ids: list[str],
) -> list[str]:
    if not asset_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            DELETE_ASSETS_SQL,
            {
                "tenant_id": tenant_id,
                "asset_type": asset_type,
                "asset_ids": asset_ids,
            },
        )
        return [str(row["id"]) for row in cur.fetchall()]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    tenant_id = str(args.tenant_id).strip()
    asset_type = str(args.asset_type).strip()
    if not tenant_id:
        raise ValueError("--tenant-id must not be empty.")
    if not asset_type:
        raise ValueError("--asset-type must not be empty.")

    from db.postgres import create_connection

    deleted_ids: list[str] = []
    with create_connection() as conn:
        frozen = size_rules_exist(conn, tenant_id)
        if not frozen and not args.allow_without_size_rules:
            raise RuntimeError(
                f"Refusing to prune because size_rules:{tenant_id} is missing. "
                "Run scripts/freeze_size_rules.py first, or pass "
                "--allow-without-size-rules."
            )

        candidates = list_assets_without_glb(
            conn,
            tenant_id=tenant_id,
            asset_type=asset_type,
        )
        candidate_ids = [str(row["id"]) for row in candidates]

        if args.delete:
            deleted_ids = delete_assets(
                conn,
                tenant_id=tenant_id,
                asset_type=asset_type,
                asset_ids=candidate_ids,
            )
        else:
            conn.rollback()

    payload = {
        "tenant_id": tenant_id,
        "asset_type": asset_type,
        "mode": "delete" if args.delete else "dry_run",
        "size_rules_guard": {
            "required": not args.allow_without_size_rules,
            "knowledge_id": f"size_rules:{tenant_id}",
            "found": frozen,
        },
        "candidate_count": len(candidates),
        "deleted_count": len(deleted_ids),
        "category_counts": category_counts(candidates),
        "candidates": candidates,
        "deleted_ids": deleted_ids,
        "ran_at_utc": datetime.now(UTC).isoformat(),
    }

    report_path = args.output or default_report_path(tenant_id, delete=args.delete)
    write_report(report_path, payload)
    LOGGER.info("Found %d assets without GLB references.", len(candidates))
    LOGGER.info("Deleted %d assets.", len(deleted_ids))
    LOGGER.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()
