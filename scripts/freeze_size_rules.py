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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze tier-count S/M/L size profiles from the current inventory DB "
            "into design_knowledge so later inventory pruning does not shift percentiles."
        )
    )
    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT_ID,
        help="Tenant whose current furniture inventory should be frozen.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to outputs/size_rules_<tenant>.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the JSON snapshot but do not upsert design_knowledge.",
    )
    return parser.parse_args()


def default_output_path(tenant_id: str) -> Path:
    safe_tenant = tenant_id.replace("/", "_").replace(" ", "_")
    return BASE_DIR / "outputs" / f"size_rules_{safe_tenant}.json"


def size_rule_payload(
    *,
    tenant_id: str,
    profiles: dict[str, Any],
    inventory_count: int,
    frozen_at_utc: str,
) -> dict[str, Any]:
    return {
        "size_profiles_by_category": profiles,
        "tenant_id": tenant_id,
        "frozen_at_utc": frozen_at_utc,
        "source": "scripts/freeze_size_rules.py",
        "builder": {
            "module": "tier_count.tools",
            "function": "build_size_profiles_from_inventory",
            "metric": "footprint_area_m2",
            "percentiles": {"p_low": 30, "p_high": 70},
            "min_n_for_category_thresholds": 30,
            "outlier_filter": "iqr_log_area_k_1_5",
        },
        "counts": {
            "inventory_items_with_dimensions": inventory_count,
            "categories": len([key for key in profiles if not key.startswith("__")]),
            "has_generic_profile": "__generic__" in profiles,
        },
    }


def upsert_size_rules(
    *,
    tenant_id: str,
    payload: dict[str, Any],
    frozen_at_utc: str,
) -> None:
    from db import PostgresDesignKnowledgeRepository
    from db.models import DesignKnowledge, TenantId
    from db.postgres import create_connection

    repo = PostgresDesignKnowledgeRepository(connection_factory=create_connection)
    profiles = payload.get("size_profiles_by_category")
    category_count = (
        len([key for key in profiles if not str(key).startswith("__")])
        if isinstance(profiles, dict)
        else 0
    )
    knowledge = DesignKnowledge(
        id=f"size_rules:{tenant_id}",
        tenant_id=TenantId(tenant_id),
        title=f"Frozen tier-count size rules for {tenant_id}",
        content=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        category="size_rules",
        tags=["size_rules", "tier_count", "frozen"],
        source="scripts/freeze_size_rules.py",
        meta={
            "tenant_id": tenant_id,
            "frozen_at_utc": frozen_at_utc,
            "count_categories": category_count,
            "count_inventory_items": payload["counts"][
                "inventory_items_with_dimensions"
            ],
        },
    )
    repo.upsert_knowledge(knowledge)


def write_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    tenant_id = str(args.tenant_id).strip()
    if not tenant_id:
        raise ValueError("--tenant-id must not be empty.")

    from tier_count.tools import (
        build_size_profiles_from_inventory,
        load_size_profile_inventory_from_db,
    )

    inventory = load_size_profile_inventory_from_db(tenant_id)
    profiles = build_size_profiles_from_inventory(inventory)
    if not profiles:
        raise RuntimeError(
            f"No size profiles could be built for tenant_id={tenant_id}."
        )

    frozen_at_utc = datetime.now(UTC).isoformat()
    payload = size_rule_payload(
        tenant_id=tenant_id,
        profiles=profiles,
        inventory_count=len(inventory),
        frozen_at_utc=frozen_at_utc,
    )

    output_path = args.output or default_output_path(tenant_id)
    write_snapshot(output_path, payload)
    LOGGER.info("Wrote frozen size rules snapshot to %s", output_path)

    if args.dry_run:
        LOGGER.info("Dry run: skipped design_knowledge upsert.")
        return

    upsert_size_rules(
        tenant_id=tenant_id,
        payload=payload,
        frozen_at_utc=frozen_at_utc,
    )
    LOGGER.info("Upserted design_knowledge id=size_rules:%s", tenant_id)


if __name__ == "__main__":
    main()
