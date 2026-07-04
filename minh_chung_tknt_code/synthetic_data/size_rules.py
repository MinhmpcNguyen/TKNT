from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

from db import PostgresAssetRepository, PostgresDesignKnowledgeRepository
from db.models import AssetFilter, DesignKnowledge, TenantId
from db.postgres import create_connection


# ============================================================
# Offline builder: S/M/L definitions per category (data-driven)
# - Metric: footprint area A = L * W (m^2), canonical L >= W
# - Outlier filter: IQR on log(A)
# - Thresholds: P30 / P70 of filtered A (per category; fallback to global)
# - Representative sizes: medoid per tier (robust z-score w/ MAD on L,W,H)
# ============================================================


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        raise ValueError("percentile() on empty list")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def median(xs: List[float]) -> float:
    if not xs:
        raise ValueError("median() on empty list")
    mid = len(xs) // 2
    ys = sorted(xs)
    return ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2


def mad(xs: List[float]) -> float:
    if not xs:
        raise ValueError("mad() on empty list")
    m = median(xs)
    return median([abs(x - m) for x in xs])


def safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))


@dataclass(frozen=True)
class NormItem:
    inventory_id: str
    category: str
    L: float  # meters, canonical L >= W
    W: float  # meters
    H: float  # meters
    A: float  # footprint area (m^2)
    R: float  # aspect ratio L/W
    raw: Dict[str, Any]


def get_category(item: Dict[str, Any]) -> str:
    attrs = item.get("attributes") or {}
    return str(attrs.get("category") or item.get("type") or "unknown")


def normalize_item(item: Dict[str, Any]) -> Optional[NormItem]:
    try:
        inv_id = str(item.get("id", "")).strip()
        if not inv_id:
            return None

        cat = get_category(item)

        lm = float(item["length_mm"])
        wm = float(item["width_mm"])
        hm = float(item["height_mm"])
        if lm <= 0 or wm <= 0 or hm <= 0:
            return None

        L = lm / 1000.0
        W = wm / 1000.0
        H = hm / 1000.0

        if W > L:
            L, W = W, L

        A = L * W
        R = L / W if W > 0 else float("inf")

        return NormItem(
            inventory_id=inv_id,
            category=cat,
            L=L,
            W=W,
            H=H,
            A=A,
            R=R,
            raw=item,
        )
    except Exception:
        return None


def filter_outliers_iqr_logA(items: List[NormItem], k: float = 1.5) -> List[NormItem]:
    if len(items) < 8:
        return items[:]

    log_as = [safe_log(it.A) for it in items]
    q1 = percentile(log_as, 25)
    q3 = percentile(log_as, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr

    kept = [it for it in items if lo <= safe_log(it.A) <= hi]
    return kept if kept else items[:]


def compute_thresholds(
    items: List[NormItem], p_low: float = 30, p_high: float = 70
) -> Tuple[float, float]:
    areas = [it.A for it in items]
    a_low = percentile(areas, p_low)
    a_high = percentile(areas, p_high)
    if a_high < a_low:
        a_low, a_high = a_high, a_low
    return a_low, a_high


def tier_of_area(area: float, a_low: float, a_high: float) -> str:
    if area <= a_low:
        return "S"
    if area <= a_high:
        return "M"
    return "L"


def medoid(items: List[NormItem]) -> Optional[NormItem]:
    if not items:
        return None
    if len(items) == 1:
        return items[0]

    ls = [it.L for it in items]
    ws = [it.W for it in items]
    hs = [it.H for it in items]
    m_l, m_w, m_h = median(ls), median(ws), median(hs)
    mad_l, mad_w, mad_h = mad(ls), mad(ws), mad(hs)

    eps = 1e-9
    mad_l = mad_l if mad_l > 0 else eps
    mad_w = mad_w if mad_w > 0 else eps
    mad_h = mad_h if mad_h > 0 else eps

    best: Optional[NormItem] = None
    best_score = float("inf")
    for it in items:
        z_l = (it.L - m_l) / mad_l
        z_w = (it.W - m_w) / mad_w
        z_h = (it.H - m_h) / mad_h
        score = z_l * z_l + z_w * z_w + z_h * z_h
        if score < best_score:
            best_score = score
            best = it
    return best


def build_size_profiles(
    inventory: List[Dict[str, Any]],
    *,
    p_low: float = 30,
    p_high: float = 70,
    outlier_k: float = 1.5,
    min_n_for_cat_thresholds: int = 30,
) -> Dict[str, Any]:
    norm = [n for it in inventory if (n := normalize_item(it)) is not None]
    if not norm:
        return {}

    by_cat: Dict[str, List[NormItem]] = {}
    for it in norm:
        by_cat.setdefault(it.category, []).append(it)

    filtered_global = filter_outliers_iqr_logA(norm, k=outlier_k)
    filtered_cat = {c: filter_outliers_iqr_logA(v, k=outlier_k) for c, v in by_cat.items()}

    profiles: Dict[str, Any] = {}
    for cat, items_cat in by_cat.items():
        base = filtered_cat[cat]
        source = "category"
        if len(base) < min_n_for_cat_thresholds:
            base = filtered_global
            source = "global"

        a_low, a_high = compute_thresholds(base, p_low=p_low, p_high=p_high)

        tiers: Dict[str, List[NormItem]] = {"S": [], "M": [], "L": []}
        for it in items_cat:
            tiers[tier_of_area(it.A, a_low, a_high)].append(it)

        rep_dims = {}
        for t in ("S", "M", "L"):
            rep = medoid(tiers[t])
            rep_dims[t] = None if rep is None else {
                "L": rep.L,
                "W": rep.W,
                "H": rep.H,
                "A": rep.A,
                "R": rep.R,
                "source_id": rep.inventory_id,
            }

        profiles[cat] = {
            "metric": "footprint_area_m2",
            "percentiles": {"p_low": p_low, "p_high": p_high},
            "thresholds": {"A_low": a_low, "A_high": a_high},
            "rep_dims_m": rep_dims,
            "counts": {
                "n_category": len(items_cat),
                "n_threshold_base": len(base),
                "threshold_source": source,
                "n_by_tier": {k: len(v) for k, v in tiers.items()},
            },
        }

    return profiles


def _load_inventory_from_db(tenant_id: str) -> List[Dict[str, Any]]:
    repo = PostgresAssetRepository(connection_factory=create_connection)
    assets = list(
        repo.list_assets(
            AssetFilter(tenant_id=TenantId(tenant_id), type="FURNITURE")
        )
    )

    items: List[Dict[str, Any]] = []
    for asset in assets:
        dims = asset.dimensions
        if dims is None:
            continue
        if dims.length_mm is None or dims.width_mm is None or dims.height_mm is None:
            continue
        item = {
            "id": str(asset.id),
            "type": str(asset.type),
            "attributes": dict(asset.attributes),
            "length_mm": float(dims.length_mm),
            "width_mm": float(dims.width_mm),
            "height_mm": float(dims.height_mm),
        }
        items.append(item)
    return items


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert_size_rules_to_db(
    *,
    tenant_id: str | None,
    profiles: Dict[str, Any],
) -> None:
    repo = PostgresDesignKnowledgeRepository(connection_factory=create_connection)
    knowledge_id = f"size_rules:{tenant_id or 'global'}"
    knowledge = DesignKnowledge(
        id=knowledge_id,
        tenant_id=TenantId(tenant_id) if tenant_id else None,
        title="Size rules: S/M/L by category",
        content=json.dumps(profiles, ensure_ascii=True),
        category="size_rules",
        tags=["size_rules"],
        source="size_rules_builder",
        meta={"tenant_id": tenant_id, "count_categories": len(profiles)},
    )
    repo.upsert_knowledge(knowledge)


def main() -> None:
    load_dotenv()
    tenant_id = os.getenv("TENANT_ID", "demo_tenant")
    output_path = BASE_DIR / "size_rules.json"

    inventory = _load_inventory_from_db(tenant_id)
    profiles = build_size_profiles(inventory)

    _write_json(output_path, profiles)
    _upsert_size_rules_to_db(tenant_id=tenant_id, profiles=profiles)

    print(f"Wrote size rules to {output_path}")
    print(f"Upserted size rules for tenant_id={tenant_id}")


if __name__ == "__main__":
    main()
