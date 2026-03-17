from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from db import PostgresAssetRepository, PostgresDesignKnowledgeRepository
from db.models import AssetFilter, DesignKnowledgeFilter, TenantId
from db.postgres import create_connection

# ============================================================
# Generic helpers
# ============================================================


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_key(value: str | None) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


# ============================================================
# Size profile builder (S/M/L) with robust stats + backfill
# ============================================================


def _percentile(xs: list[float], p: float) -> float:
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


def _median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median() on empty list")
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2 == 1:
        return ys[mid]
    return (ys[mid - 1] + ys[mid]) / 2


def _mad(xs: list[float]) -> float:
    if not xs:
        raise ValueError("mad() on empty list")
    m = _median(xs)
    return _median([abs(x - m) for x in xs])


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))


@dataclass(frozen=True)
class _NormItem:
    inventory_id: str
    category: str
    L: float
    W: float
    H: float
    A: float
    R: float
    raw: dict[str, Any]


def _get_category(item: dict[str, Any]) -> str:
    attrs = item.get("attributes") or {}
    return str(attrs.get("category") or item.get("type") or "unknown")


def _normalize_item(item: dict[str, Any]) -> _NormItem | None:
    try:
        inv_id = str(item.get("id", "")).strip()
        if not inv_id:
            return None

        cat = _get_category(item)

        length_mm = float(item["length_mm"])
        width_mm = float(item["width_mm"])
        height_mm = float(item["height_mm"])
        if length_mm <= 0 or width_mm <= 0 or height_mm <= 0:
            return None

        L = length_mm / 1000.0
        W = width_mm / 1000.0
        H = height_mm / 1000.0
        if W > L:
            L, W = W, L  # ensure L >= W

        A = L * W
        R = L / W if W > 0 else float("inf")

        return _NormItem(
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


def _filter_outliers_iqr_logA(
    items: list[_NormItem], k: float = 1.5
) -> list[_NormItem]:
    if len(items) < 8:
        return items[:]
    log_as = [_safe_log(it.A) for it in items]
    q1 = _percentile(log_as, 25)
    q3 = _percentile(log_as, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    kept = [it for it in items if lo <= _safe_log(it.A) <= hi]
    return kept if kept else items[:]


def _compute_thresholds(
    items: list[_NormItem], p_low: float, p_high: float
) -> tuple[float, float]:
    areas = [it.A for it in items]
    a_low = _percentile(areas, p_low)
    a_high = _percentile(areas, p_high)
    if a_high < a_low:
        a_low, a_high = a_high, a_low
    return a_low, a_high


def _tier_of_area(area: float, a_low: float, a_high: float) -> str:
    if area <= a_low:
        return "S"
    if area <= a_high:
        return "M"
    return "L"


def _medoid(items: list[_NormItem]) -> _NormItem | None:
    if not items:
        return None
    if len(items) == 1:
        return items[0]

    ls = [it.L for it in items]
    ws = [it.W for it in items]
    hs = [it.H for it in items]
    m_l, m_w, m_h = _median(ls), _median(ws), _median(hs)
    mad_l, mad_w, mad_h = _mad(ls), _mad(ws), _mad(hs)

    eps = 1e-9
    mad_l = mad_l if mad_l > 0 else eps
    mad_w = mad_w if mad_w > 0 else eps
    mad_h = mad_h if mad_h > 0 else eps

    best: _NormItem | None = None
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


def _rep_from_items_for_tier(items_cat: list[_NormItem], tier: str) -> _NormItem | None:
    """
    Backfill representative when a tier has 0 samples.
    - S: smallest footprint in category
    - L: largest footprint in category
    - M: medoid of entire category
    """
    if not items_cat:
        return None
    if tier == "S":
        return min(items_cat, key=lambda it: it.A)
    if tier == "L":
        return max(items_cat, key=lambda it: it.A)
    return _medoid(items_cat)


def _rep_dict(rep: _NormItem | None) -> dict[str, Any] | None:
    if rep is None:
        return None
    return {
        "L": rep.L,
        "W": rep.W,
        "H": rep.H,
        "A": rep.A,
        "R": rep.R,
        "source_id": rep.inventory_id,
    }


def _build_size_profiles(
    inventory: list[dict[str, Any]],
    *,
    p_low: float = 30,
    p_high: float = 70,
    outlier_k: float = 1.5,
    min_n_for_cat_thresholds: int = 30,
) -> dict[str, Any]:
    norm = [n for item in inventory if (n := _normalize_item(item)) is not None]
    if not norm:
        return {}

    by_cat: dict[str, list[_NormItem]] = {}
    for it in norm:
        by_cat.setdefault(it.category, []).append(it)

    filtered_global = _filter_outliers_iqr_logA(norm, k=outlier_k)
    filtered_cat = {
        c: _filter_outliers_iqr_logA(v, k=outlier_k) for c, v in by_cat.items()
    }

    profiles: dict[str, Any] = {}
    for cat, items_cat in by_cat.items():
        base = filtered_cat[cat]
        source = "category"
        if len(base) < min_n_for_cat_thresholds:
            base = filtered_global
            source = "global"

        a_low, a_high = _compute_thresholds(base, p_low, p_high)

        tiers: dict[str, list[_NormItem]] = {"S": [], "M": [], "L": []}
        for it in items_cat:
            tiers[_tier_of_area(it.A, a_low, a_high)].append(it)

        rep_dims: dict[str, Any] = {}
        backfilled: dict[str, bool] = {"S": False, "M": False, "L": False}

        for t in ("S", "M", "L"):
            rep = _medoid(tiers[t])
            if rep is None:
                rep = _rep_from_items_for_tier(items_cat, t)
                backfilled[t] = True
            rep_dims[t] = _rep_dict(rep)

        profiles[cat] = {
            "metric": "footprint_area_m2",
            "percentiles": {"p_low": p_low, "p_high": p_high},
            "thresholds": {"A_low": a_low, "A_high": a_high},
            "rep_dims_m": rep_dims,
            "rep_dims_backfilled": backfilled,
            "counts": {
                "n_category": len(items_cat),
                "n_threshold_base": len(base),
                "threshold_source": source,
                "n_by_tier": {k: len(v) for k, v in tiers.items()},
            },
        }

    # Generic fallback profile, useful when a category is missing at inference time.
    generic_s = _medoid(
        [it for items_cat in by_cat.values() for it in items_cat[:]]
    )  # rough medoid for all
    generic_base = filtered_global if filtered_global else norm
    a_low_g, a_high_g = _compute_thresholds(generic_base, p_low, p_high)

    tiers_g: dict[str, list[_NormItem]] = {"S": [], "M": [], "L": []}
    for it in norm:
        tiers_g[_tier_of_area(it.A, a_low_g, a_high_g)].append(it)

    generic_rep_dims: dict[str, Any] = {}
    generic_backfilled = {"S": False, "M": False, "L": False}
    for t in ("S", "M", "L"):
        rep = _medoid(tiers_g[t])
        if rep is None:
            rep = generic_s
            generic_backfilled[t] = True
        generic_rep_dims[t] = _rep_dict(rep)

    profiles["__generic__"] = {
        "metric": "footprint_area_m2",
        "percentiles": {"p_low": p_low, "p_high": p_high},
        "thresholds": {"A_low": a_low_g, "A_high": a_high_g},
        "rep_dims_m": generic_rep_dims,
        "rep_dims_backfilled": generic_backfilled,
        "counts": {
            "n_category": len(norm),
            "n_threshold_base": len(generic_base),
            "threshold_source": "global",
            "n_by_tier": {k: len(v) for k, v in tiers_g.items()},
        },
    }

    return profiles


# ============================================================
# Tool: get_size_profiles
# ============================================================


def _load_inventory_from_db(tenant_id: str) -> list[dict[str, Any]]:
    repo = PostgresAssetRepository(connection_factory=create_connection)
    assets = list(
        repo.list_assets(AssetFilter(tenant_id=TenantId(tenant_id), type="FURNITURE"))
    )

    items: list[dict[str, Any]] = []
    for asset in assets:
        dims = asset.dimensions
        if dims is None:
            continue
        if dims.length_mm is None or dims.width_mm is None or dims.height_mm is None:
            continue
        items.append(
            {
                "id": str(asset.id),
                "type": str(asset.type),
                "attributes": dict(asset.attributes),
                "length_mm": float(dims.length_mm),
                "width_mm": float(dims.width_mm),
                "height_mm": float(dims.height_mm),
            }
        )
    return items


def _parse_profiles_payload(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    if isinstance(parsed.get("size_profiles_by_category"), dict):
        return parsed["size_profiles_by_category"]

    return parsed


def _load_profiles_from_db(tenant_id: str | None) -> dict[str, Any] | None:
    repo = PostgresDesignKnowledgeRepository(connection_factory=create_connection)

    if tenant_id:
        knowledge = repo.get_knowledge(f"size_rules:{tenant_id}")
        if knowledge is not None:
            payload = _parse_profiles_payload(knowledge.content)
            if payload is not None:
                return payload

    knowledge_global = repo.get_knowledge("size_rules:global")
    if knowledge_global is not None:
        payload = _parse_profiles_payload(knowledge_global.content)
        if payload is not None:
            return payload

    candidates = list(repo.list_knowledge(DesignKnowledgeFilter(category="size_rules")))
    if candidates:
        payload = _parse_profiles_payload(candidates[0].content)
        if payload is not None:
            return payload

    return None


def _rep_has_positive_dims(rep: Any) -> bool:
    if not isinstance(rep, dict):
        return False
    try:
        return (
            float(rep.get("L", 0)) > 0
            and float(rep.get("W", 0)) > 0
            and float(rep.get("H", 0)) > 0
        )
    except Exception:
        return False


def _sanitize_profiles_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Defensive sanitize when profiles are loaded from DB and may have rep_dims_m[tier]=null.
    We backfill missing tiers using any available tier within the same category (M -> L -> S).
    """
    out = payload if isinstance(payload, dict) else {}
    for cat, prof in list(out.items()):
        if not isinstance(prof, dict):
            continue
        rep_dims = prof.get("rep_dims_m")
        if not isinstance(rep_dims, dict):
            continue

        fallbacks = []
        for t in ("M", "L", "S"):
            rep = rep_dims.get(t)
            if _rep_has_positive_dims(rep):
                fallbacks.append(rep)

        for t in ("S", "M", "L"):
            rep = rep_dims.get(t)
            if _rep_has_positive_dims(rep):
                continue
            if fallbacks:
                rep_dims[t] = fallbacks[0]
    return out


def _build_generic_profile_from_profiles(
    profiles_by_category: dict[str, Any],
) -> dict[str, Any] | None:
    reps_by_tier: dict[str, list[dict[str, float]]] = {"S": [], "M": [], "L": []}

    for cat, prof in profiles_by_category.items():
        if str(cat).startswith("__"):
            continue
        if not isinstance(prof, dict):
            continue
        rep_dims = prof.get("rep_dims_m") or {}
        if not isinstance(rep_dims, dict):
            continue
        for tier in ("S", "M", "L"):
            rep = rep_dims.get(tier)
            if _rep_has_positive_dims(rep):
                reps_by_tier[tier].append(
                    {
                        "L": float(rep["L"]),
                        "W": float(rep["W"]),
                        "H": float(rep["H"]),
                        "A": float(rep.get("A", float(rep["L"]) * float(rep["W"]))),
                        "R": float(
                            rep.get("R", float(rep["L"]) / max(float(rep["W"]), 1e-9))
                        ),
                        "source_id": str(rep.get("source_id", "__generic__")),
                    }
                )

    if not any(reps_by_tier.values()):
        return None

    def _median_rep(reps: list[dict[str, float]]) -> dict[str, Any] | None:
        if not reps:
            return None
        return {
            "L": _median([r["L"] for r in reps]),
            "W": _median([r["W"] for r in reps]),
            "H": _median([r["H"] for r in reps]),
            "A": _median([r["A"] for r in reps]),
            "R": _median([r["R"] for r in reps]),
            "source_id": "__generic__",
        }

    rep_s = (
        _median_rep(reps_by_tier["S"])
        or _median_rep(reps_by_tier["M"])
        or _median_rep(reps_by_tier["L"])
    )
    rep_m = (
        _median_rep(reps_by_tier["M"])
        or _median_rep(reps_by_tier["S"])
        or _median_rep(reps_by_tier["L"])
    )
    rep_l = (
        _median_rep(reps_by_tier["L"])
        or _median_rep(reps_by_tier["M"])
        or _median_rep(reps_by_tier["S"])
    )

    if not rep_s or not rep_m or not rep_l:
        return None

    return {
        "metric": "footprint_area_m2",
        "percentiles": {"p_low": 30, "p_high": 70},
        "thresholds": {"A_low": float(rep_s["A"]), "A_high": float(rep_l["A"])},
        "rep_dims_m": {"S": rep_s, "M": rep_m, "L": rep_l},
        "rep_dims_backfilled": {"S": False, "M": False, "L": False},
        "counts": {
            "n_category": sum(len(v) for v in reps_by_tier.values()),
            "n_threshold_base": sum(len(v) for v in reps_by_tier.values()),
            "threshold_source": "derived_from_existing_profiles",
            "n_by_tier": {k: len(v) for k, v in reps_by_tier.items()},
        },
    }


def get_size_profiles(
    *, categories: list[str], tenant_id: str | None = None
) -> dict[str, Any]:
    profiles_payload = _load_profiles_from_db(tenant_id)
    if isinstance(profiles_payload, dict):
        profiles_payload = _sanitize_profiles_payload(profiles_payload)

    if not isinstance(profiles_payload, dict) or not profiles_payload:
        inventory = _load_inventory_from_db(tenant_id or "demo_tenant")
        profiles_payload = _build_size_profiles(inventory)

    profiles_by_category = (
        profiles_payload if isinstance(profiles_payload, dict) else {}
    )

    if "__generic__" not in profiles_by_category:
        generic = _build_generic_profile_from_profiles(profiles_by_category)
        if generic is not None:
            profiles_by_category["__generic__"] = generic

    requested: dict[str, Any] = {}
    missing: list[str] = []
    normalized_lookup = {_norm_key(k): v for k, v in profiles_by_category.items()}

    for category in categories:
        if category in profiles_by_category:
            requested[category] = profiles_by_category[category]
            continue

        nk = _norm_key(category)
        if nk in normalized_lookup:
            requested[category] = normalized_lookup[nk]
        else:
            missing.append(category)

    if "__generic__" in profiles_by_category:
        requested["__generic__"] = profiles_by_category["__generic__"]

    return {"size_profiles_by_category": requested, "missing_categories": missing}


# ============================================================
# Tool: estimate_budget
# ============================================================

FIXED_ACCESS_CLEARANCE_RATIO = 0.25


def _polygon_area_m2(points_ccw_mm: list[dict[str, int]]) -> float:
    if len(points_ccw_mm) < 3:
        return 0.0
    s = 0
    n = len(points_ccw_mm)
    for i in range(n):
        x1, y1 = points_ccw_mm[i]["x"], points_ccw_mm[i]["y"]
        x2, y2 = points_ccw_mm[(i + 1) % n]["x"], points_ccw_mm[(i + 1) % n]["y"]
        s += x1 * y2 - x2 * y1
    area_mm2 = abs(s) / 2.0
    return area_mm2 / 1_000_000.0


def _polygon_perimeter_m(points_mm: list[dict[str, int]]) -> float:
    if len(points_mm) < 2:
        return 0.0
    per_mm = 0.0
    n = len(points_mm)
    for i in range(n):
        x1, y1 = points_mm[i]["x"], points_mm[i]["y"]
        x2, y2 = points_mm[(i + 1) % n]["x"], points_mm[(i + 1) % n]["y"]
        dx, dy = x2 - x1, y2 - y1
        per_mm += (dx * dx + dy * dy) ** 0.5
    return per_mm / 1000.0


def _compactness(room_pts_mm: list[dict[str, int]]) -> float:
    area = _polygon_area_m2(room_pts_mm)
    perimeter = _polygon_perimeter_m(room_pts_mm)
    if area <= 0 or perimeter <= 0:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter * perimeter)


def _available_area_m2(room_model: dict[str, Any]) -> float:
    room_pts = (room_model.get("room") or {}).get("polygon_ccw") or []
    room_area = _polygon_area_m2(room_pts)
    obstacles = room_model.get("obstacles", []) or []

    try:
        from shapely.geometry import Polygon  # type: ignore
        from shapely.ops import unary_union  # type: ignore

        room_poly = Polygon([(p["x"], p["y"]) for p in room_pts])
        obs_polys = []
        for ob in obstacles:
            if not ob.get("hard", True):
                continue
            pts = ob.get("polygon_ccw") or []
            if len(pts) >= 3:
                obs_polys.append(Polygon([(p["x"], p["y"]) for p in pts]))
        if not obs_polys:
            return float(room_poly.area) / 1_000_000.0
        free = room_poly.difference(unary_union(obs_polys))
        return float(free.area) / 1_000_000.0
    except Exception:
        obs_area = 0.0
        for ob in obstacles:
            if not ob.get("hard", True):
                continue
            pts = ob.get("polygon_ccw") or []
            obs_area += _polygon_area_m2(pts) if pts else 0.0
        return max(0.0, room_area - obs_area)


@dataclass(frozen=True)
class _AccessRequirements:
    object_ids: frozenset[str]
    categories: frozenset[str]


def _is_access_required(entry: dict[str, Any]) -> bool:
    return entry.get("required", True) is not False


def _extract_access_requirements(clusters_json: Any) -> _AccessRequirements:
    object_ids: set[str] = set()
    categories: set[str] = set()

    if not isinstance(clusters_json, dict):
        return _AccessRequirements(frozenset(), frozenset())

    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return _AccessRequirements(frozenset(), frozenset())

    def _consume_entry(entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            return
        if not _is_access_required(entry):
            return

        oid = entry.get("id") or entry.get("object_id") or entry.get("name")
        if isinstance(oid, str) and oid.strip():
            object_ids.add(_norm_key(oid))

        cat = entry.get("category")
        if isinstance(cat, str) and cat.strip():
            categories.add(_norm_key(cat))

        cats = entry.get("categories")
        if isinstance(cats, list):
            for c in cats:
                if isinstance(c, str) and c.strip():
                    categories.add(_norm_key(c))

    for c in clusters:
        if not isinstance(c, dict):
            continue

        rules = c.get("cluster_rules") or {}
        if isinstance(rules, dict):
            access_reqs = rules.get("access_requirements") or []
            if isinstance(access_reqs, list):
                for ar in access_reqs:
                    if not isinstance(ar, dict):
                        continue
                    if ar.get("type") != "front_clearance":
                        continue
                    _consume_entry(ar)

        hard = c.get("hard_constraints") or []
        if isinstance(hard, list):
            for hc in hard:
                if not isinstance(hc, dict):
                    continue
                if hc.get("type") != "requires_access":
                    continue
                if hc.get("mode") != "front_clearance":
                    continue
                _consume_entry(hc)

    return _AccessRequirements(frozenset(object_ids), frozenset(categories))


def _clearance_policy_for_category(category: str) -> tuple[float, float, float]:
    """
    Returns (min_depth_m, max_depth_m, occupancy_multiplier)
    - depth is front-clearance depth in meters
    - occupancy_multiplier softens categories needing partial/occasional access
    """
    k = _norm_key(category)

    seating_tokens = ("sofa", "sectional", "armchair", "recliner", "chair", "bench")
    bed_tokens = ("bed",)
    work_tokens = (
        "desk",
        "dining_table",
        "table",
        "kitchen_island",
        "island",
        "worktop",
        "counter",
    )
    storage_tokens = (
        "wardrobe",
        "closet",
        "cabinet",
        "storage",
        "bookshelf",
        "shelf",
        "dresser",
    )
    media_tokens = ("tv_console", "tv_stand", "media_console", "media", "console")

    if any(t in k for t in seating_tokens):
        return (0.45, 0.90, 1.00)
    if any(t in k for t in bed_tokens):
        return (0.45, 0.80, 0.90)
    if any(t in k for t in work_tokens):
        return (0.40, 0.80, 0.95)
    if any(t in k for t in storage_tokens):
        return (0.35, 0.65, 0.65)
    if any(t in k for t in media_tokens):
        return (0.35, 0.60, 0.55)

    return (0.30, 0.75, 0.80)


def _effective_footprint_m2_from_profile(
    size_profile: dict[str, Any],
    tier: str,
    *,
    category: str,
    needs_front_clearance: bool,
) -> tuple[float, float, float]:
    """
    Effective footprint:
      base = L * W

    If needs front clearance:
      clearance_depth = clamp(W * FIXED_ACCESS_CLEARANCE_RATIO, min_depth, max_depth)
      extra = L * clearance_depth * occupancy_multiplier
      effective = base + extra

    Returns:
      (base_m2, effective_m2, clearance_depth_m)
    """
    rep = (size_profile.get("rep_dims_m") or {}).get(tier.upper())
    if not isinstance(rep, dict):
        raise ValueError(f"Missing rep_dims_m for tier={tier}")

    L = float(rep.get("L", 0) or 0)
    W = float(rep.get("W", 0) or 0)
    if L <= 0 or W <= 0:
        raise ValueError(f"Invalid rep_dims_m: L/W non-positive for tier={tier}")

    base = L * W
    if not needs_front_clearance:
        return base, base, 0.0

    min_depth, max_depth, occupancy_multiplier = _clearance_policy_for_category(
        category
    )
    clearance_depth = _clamp(W * FIXED_ACCESS_CLEARANCE_RATIO, min_depth, max_depth)
    extra = L * clearance_depth * occupancy_multiplier
    effective = base + extra
    return base, effective, clearance_depth


def _prepare_profile_lookup(
    size_profiles_by_category: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    normalized: dict[str, dict[str, Any]] = {}
    for k, v in size_profiles_by_category.items():
        if isinstance(v, dict):
            normalized[_norm_key(k)] = v

    generic = size_profiles_by_category.get("__generic__")
    if not isinstance(generic, dict):
        generic = _build_generic_profile_from_profiles(size_profiles_by_category)

    return normalized, generic


def _resolve_profile_for_category(
    category: str,
    normalized_profiles: dict[str, dict[str, Any]],
    generic_profile: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    nk = _norm_key(category)

    if nk in normalized_profiles:
        return normalized_profiles[nk], "exact_or_normalized"

    if generic_profile is not None:
        return generic_profile, "generic_fallback"

    raise ValueError(f"Missing size profile for category={category}")


def _decision_identity_keys(decision: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("id", "object_id", "furniture_id", "asset_id", "name"):
        value = decision.get(field)
        if isinstance(value, str) and value.strip():
            keys.add(_norm_key(value))
    return keys


def _decision_needs_access(
    decision: dict[str, Any],
    access_requirements: _AccessRequirements,
) -> bool:
    category = _norm_key(str(decision.get("category", "")))
    decision_keys = _decision_identity_keys(decision)

    if category and category in access_requirements.categories:
        return True
    if decision_keys and any(
        k in access_requirements.object_ids for k in decision_keys
    ):
        return True

    # backward compatibility with old data
    if category and category in access_requirements.object_ids:
        return True

    return False


def _estimate_decision_footprint_detail(
    decision: dict[str, Any],
    normalized_profiles: dict[str, dict[str, Any]],
    generic_profile: dict[str, Any] | None,
    *,
    access_requirements: _AccessRequirements,
) -> dict[str, Any]:
    category = str(decision["category"])
    tier = str(decision["size_tier"]).upper()
    qty = int(decision.get("quantity", 1))
    qty = max(qty, 0)

    profile, profile_source = _resolve_profile_for_category(
        category,
        normalized_profiles,
        generic_profile,
    )

    needs_access = _decision_needs_access(decision, access_requirements)
    base_fp_per_item, effective_fp_per_item, clearance_depth_m = (
        _effective_footprint_m2_from_profile(
            profile,
            tier,
            category=category,
            needs_front_clearance=needs_access,
        )
    )

    return {
        "decision_id": next(
            (
                str(decision.get(k))
                for k in ("id", "object_id", "furniture_id", "asset_id", "name")
                if decision.get(k) is not None
            ),
            "",
        ),
        "cluster_id": str(decision.get("cluster_id", "unknown_cluster")),
        "category": category,
        "size_tier": tier,
        "quantity": qty,
        "priority": str(decision.get("priority", "secondary")),
        "needs_front_clearance": needs_access,
        "profile_source": profile_source,
        "clearance_depth_m": clearance_depth_m,
        "base_footprint_m2_per_item": base_fp_per_item,
        "effective_footprint_m2_per_item": effective_fp_per_item,
        "base_footprint_m2_total": base_fp_per_item * qty,
        "effective_footprint_m2_total": effective_fp_per_item * qty,
    }


def _sum_footprint_by_cluster_m2(
    decision_details: list[dict[str, Any]],
    *,
    field: str = "effective_footprint_m2_total",
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for d in decision_details:
        cluster_id = str(d["cluster_id"])
        totals.setdefault(cluster_id, 0.0)
        totals[cluster_id] += float(d[field])
    return totals


def _sum_footprint_breakdown(
    decision_details: list[dict[str, Any]],
    *,
    field: str = "effective_footprint_m2_total",
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for d in decision_details:
        cluster_id = str(d["cluster_id"])
        priority = str(d.get("priority", "secondary"))
        out.setdefault(
            cluster_id,
            {"anchor": 0.0, "primary": 0.0, "secondary": 0.0, "optional": 0.0},
        )

        if priority not in out[cluster_id]:
            priority = "secondary"
        out[cluster_id][priority] += float(d[field])
    return out


def _choose_budget_ratio(style: str, user_notes: str) -> float:
    style_l = (style or "").lower()
    notes_l = (user_notes or "").lower()

    ratio = 0.50
    if "japandi" in style_l or "minimal" in style_l or "scandi" in style_l:
        ratio = 0.45
    if "cozy" in notes_l or "ấm cúng" in notes_l:
        ratio = 0.60
    if (
        "airy" in notes_l
        or "spacious" in notes_l
        or "thoáng" in notes_l
        or "tối giản" in notes_l
    ):
        ratio = min(ratio, 0.45)

    return ratio


def _cluster_ratio_multiplier(cluster_id: str) -> float:
    k = _norm_key(cluster_id)

    if any(t in k for t in ("misc", "decor", "accent", "plant", "accessory")):
        return 0.75
    if any(t in k for t in ("storage", "wardrobe", "closet", "tv", "media", "console")):
        return 0.85
    if any(
        t in k
        for t in (
            "seating",
            "living",
            "lounge",
            "bed",
            "sleep",
            "kitchen",
            "cook",
            "prep",
            "dining",
            "study",
            "work",
        )
    ):
        return 1.10
    return 1.00


def _extract_zone_areas_m2(room_model: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    zones = room_model.get("zones", []) or []
    for zone in zones:
        zone_id = str(zone.get("id", "unknown_zone"))
        poly = zone.get("polygon_ccw", [])
        out[zone_id] = _polygon_area_m2(poly) if poly else 0.0
    return out


def _priority_trim_rank(priority: str) -> int:
    p = _norm_key(priority)
    order = {
        "optional": 0,
        "secondary": 1,
        "primary": 2,
        "anchor": 3,
    }
    return order.get(p, 1)


def _is_trimmable_priority(priority: str) -> bool:
    return _norm_key(priority) in {"optional", "secondary"}


def _check_budget_limits(
    *,
    limits_m2: dict[str, float],
    footprints_m2: dict[str, float],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for cid, fp in footprints_m2.items():
        limit = float(limits_m2.get(cid, 0.0))
        if fp > limit:
            violations.append(
                {
                    "cluster_id": cid,
                    "limit_m2": limit,
                    "footprint_m2": fp,
                    "over_by_m2": fp - limit,
                }
            )
    return violations


def _derive_cluster_budget_limits(
    *,
    cluster_ids: list[str],
    raw_cluster_footprints_m2: dict[str, float],
    room_area_m2: float,
    base_ratio: float,
    shape_factor: float,
    cluster_area_mode: str,
    room_model: dict[str, Any],
    adaptive_cluster_ratios: bool,
    cluster_ratio_overrides: dict[str, float] | None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """
    Returns:
      global_budget_m2,
      cluster_ratios,
      cluster_budget_limits_m2

    Main fix:
    - no longer gives every cluster the whole room
    - if zones are missing, budgets are distributed from ONE global room budget
    """
    global_ratio = _clamp(base_ratio * shape_factor, 0.20, 0.75)
    global_budget_m2 = room_area_m2 * global_ratio

    cluster_ratios: dict[str, float] = {}
    for cid in cluster_ids:
        if cluster_ratio_overrides and cid in cluster_ratio_overrides:
            r = float(cluster_ratio_overrides[cid])
        else:
            mult = _cluster_ratio_multiplier(cid) if adaptive_cluster_ratios else 1.0
            r = _clamp(base_ratio * shape_factor * mult, 0.20, 0.75)
        cluster_ratios[cid] = r

    limits: dict[str, float] = {cid: 0.0 for cid in cluster_ids}

    zone_areas = (
        _extract_zone_areas_m2(room_model) if cluster_area_mode == "zones" else {}
    )
    zone_based_total = 0.0
    zone_backed_ids: set[str] = set()

    for cid in cluster_ids:
        zone_area = float(zone_areas.get(cid, 0.0))
        if zone_area > 0:
            limits[cid] = zone_area * cluster_ratios[cid]
            zone_based_total += limits[cid]
            zone_backed_ids.add(cid)

    remaining_ids = [cid for cid in cluster_ids if cid not in zone_backed_ids]
    remaining_budget = max(0.0, global_budget_m2 - zone_based_total)

    if remaining_ids:
        weights: dict[str, float] = {}
        for cid in remaining_ids:
            base_fp = float(raw_cluster_footprints_m2.get(cid, 0.0))
            base_fp = max(base_fp, 0.25)
            weights[cid] = base_fp * _cluster_ratio_multiplier(cid)

        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            share = remaining_budget / max(len(remaining_ids), 1)
            for cid in remaining_ids:
                limits[cid] = share
        else:
            for cid in remaining_ids:
                limits[cid] = remaining_budget * (weights[cid] / weight_sum)

    total_limits = sum(limits.values())
    if total_limits > global_budget_m2 and total_limits > 0:
        scale = global_budget_m2 / total_limits
        for cid in list(limits.keys()):
            limits[cid] *= scale

    return global_budget_m2, cluster_ratios, limits


def _fit_decisions_to_budget(
    decision_details: list[dict[str, Any]],
    *,
    cluster_budget_limits_m2: dict[str, float],
    global_budget_m2: float,
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    """
    Greedy fitter:
    - anchor / primary: keep
    - optional trimmed first
    - then secondary
    - trims one item at a time until cluster/global budgets fit or cannot trim more
    """
    fitted: list[dict[str, Any]] = []
    for d in decision_details:
        q = int(d.get("quantity", 0))
        q = max(q, 0)
        row = dict(d)
        row["recommended_quantity"] = q
        row["dropped_quantity"] = 0
        row["recommended_effective_footprint_m2_total"] = (
            float(d["effective_footprint_m2_per_item"]) * q
        )
        row["recommended_base_footprint_m2_total"] = (
            float(d["base_footprint_m2_per_item"]) * q
        )
        fitted.append(row)

    def _cluster_totals() -> dict[str, float]:
        out: dict[str, float] = {}
        for row in fitted:
            cid = str(row["cluster_id"])
            out.setdefault(cid, 0.0)
            out[cid] += float(row["recommended_effective_footprint_m2_total"])
        return out

    def _global_total(cluster_totals: dict[str, float]) -> float:
        return sum(cluster_totals.values())

    def _trim_one_from_cluster(cluster_id: str) -> bool:
        candidates = [
            row
            for row in fitted
            if str(row["cluster_id"]) == cluster_id
            and int(row["recommended_quantity"]) > 0
            and _is_trimmable_priority(str(row.get("priority", "secondary")))
        ]
        if not candidates:
            return False

        candidates.sort(
            key=lambda row: (
                _priority_trim_rank(str(row.get("priority", "secondary"))),
                -float(row["effective_footprint_m2_per_item"]),
                str(row.get("category", "")),
            )
        )
        row = candidates[0]
        row["recommended_quantity"] = int(row["recommended_quantity"]) - 1
        row["dropped_quantity"] = int(row["dropped_quantity"]) + 1
        row["recommended_effective_footprint_m2_total"] = float(
            row["effective_footprint_m2_per_item"]
        ) * int(row["recommended_quantity"])
        row["recommended_base_footprint_m2_total"] = float(
            row["base_footprint_m2_per_item"]
        ) * int(row["recommended_quantity"])
        return True

    def _trim_one_global(cluster_overruns: dict[str, float]) -> bool:
        candidates = [
            row
            for row in fitted
            if int(row["recommended_quantity"]) > 0
            and _is_trimmable_priority(str(row.get("priority", "secondary")))
        ]
        if not candidates:
            return False

        def _score(row: dict[str, Any]) -> tuple[float, float, float, str]:
            cid = str(row["cluster_id"])
            over = float(cluster_overruns.get(cid, 0.0))
            return (
                1.0 if over > 0 else 0.0,
                -float(_priority_trim_rank(str(row.get("priority", "secondary")))),
                float(row["effective_footprint_m2_per_item"]),
                cid,
            )

        candidates.sort(key=_score, reverse=True)
        row = candidates[0]
        row["recommended_quantity"] = int(row["recommended_quantity"]) - 1
        row["dropped_quantity"] = int(row["dropped_quantity"]) + 1
        row["recommended_effective_footprint_m2_total"] = float(
            row["effective_footprint_m2_per_item"]
        ) * int(row["recommended_quantity"])
        row["recommended_base_footprint_m2_total"] = float(
            row["base_footprint_m2_per_item"]
        ) * int(row["recommended_quantity"])
        return True

    # Phase 1: trim per-cluster overruns
    while True:
        cluster_totals = _cluster_totals()
        overruns = {
            cid: cluster_totals.get(cid, 0.0)
            - float(cluster_budget_limits_m2.get(cid, 0.0))
            for cid in cluster_budget_limits_m2.keys()
        }
        active = sorted(
            [cid for cid, over in overruns.items() if over > 1e-9],
            key=lambda cid: overruns[cid],
            reverse=True,
        )
        if not active:
            break

        changed = False
        for cid in active:
            if _trim_one_from_cluster(cid):
                changed = True
                break
        if not changed:
            break

    # Phase 2: trim global overrun if still needed
    while True:
        cluster_totals = _cluster_totals()
        total = _global_total(cluster_totals)
        if total <= global_budget_m2 + 1e-9:
            break

        overruns = {
            cid: cluster_totals.get(cid, 0.0)
            - float(cluster_budget_limits_m2.get(cid, 0.0))
            for cid in cluster_budget_limits_m2.keys()
        }
        changed = _trim_one_global(overruns)
        if not changed:
            break

    final_cluster_totals = _cluster_totals()
    final_global_total = _global_total(final_cluster_totals)
    return fitted, final_cluster_totals, final_global_total


def estimate_budget(
    *,
    room_model: dict[str, Any],
    decisions: list[dict[str, Any]],
    size_profiles_by_category: dict[str, dict[str, Any]],
    style: str,
    user_notes: str,
    ratio_override: float | None = None,
    cluster_area_mode: str = "room",
    clusters_json: dict[str, Any] | None = None,
    adaptive_cluster_ratios: bool = True,
    cluster_ratio_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Improved budget estimator + quantity fitter.

    Key changes:
    - access_clearance_ratio is FIXED to 0.25
    - budgets are derived from ONE global room budget, not duplicated per cluster
    - secondary/optional quantities are greedily trimmed to fit cluster/global limits
    """
    base_ratio = (
        float(ratio_override)
        if ratio_override is not None
        else _choose_budget_ratio(style, user_notes)
    )

    access_requirements = (
        _extract_access_requirements(clusters_json)
        if clusters_json
        else _AccessRequirements(frozenset(), frozenset())
    )

    normalized_profiles, generic_profile = _prepare_profile_lookup(
        size_profiles_by_category
    )

    decision_details: list[dict[str, Any]] = []
    missing_profile_categories: set[str] = set()

    for decision in decisions:
        try:
            detail = _estimate_decision_footprint_detail(
                decision,
                normalized_profiles,
                generic_profile,
                access_requirements=access_requirements,
            )
            if detail["profile_source"] == "generic_fallback":
                missing_profile_categories.add(str(decision.get("category", "unknown")))
            decision_details.append(detail)
        except ValueError as e:
            missing_profile_categories.add(str(decision.get("category", "unknown")))
            raise ValueError(f"{e}. decision={decision}") from e

    room_poly = (room_model.get("room") or {}).get("polygon_ccw") or []
    room_area = _available_area_m2(room_model) if room_poly else 0.0
    compactness = _compactness(room_poly) if room_poly else 0.0
    shape_factor = max(0.7, min(1.0, compactness / 0.6)) if compactness > 0 else 1.0
    ratio = _clamp(base_ratio * shape_factor, 0.20, 0.75)

    raw_cluster_footprints = _sum_footprint_by_cluster_m2(
        decision_details,
        field="effective_footprint_m2_total",
    )
    raw_breakdown = _sum_footprint_breakdown(
        decision_details,
        field="effective_footprint_m2_total",
    )

    cluster_ids = sorted(raw_cluster_footprints.keys())
    global_budget_m2, cluster_ratios, cluster_budget_limits_m2 = (
        _derive_cluster_budget_limits(
            cluster_ids=cluster_ids,
            raw_cluster_footprints_m2=raw_cluster_footprints,
            room_area_m2=room_area,
            base_ratio=base_ratio,
            shape_factor=shape_factor,
            cluster_area_mode=cluster_area_mode,
            room_model=room_model,
            adaptive_cluster_ratios=adaptive_cluster_ratios,
            cluster_ratio_overrides=cluster_ratio_overrides,
        )
    )

    violations_before_fit = _check_budget_limits(
        limits_m2=cluster_budget_limits_m2,
        footprints_m2=raw_cluster_footprints,
    )
    global_violation_before_fit = max(
        0.0, sum(raw_cluster_footprints.values()) - global_budget_m2
    )

    fitted_details, fitted_cluster_footprints, fitted_global_footprint = (
        _fit_decisions_to_budget(
            decision_details,
            cluster_budget_limits_m2=cluster_budget_limits_m2,
            global_budget_m2=global_budget_m2,
        )
    )

    fitted_breakdown = _sum_footprint_breakdown(
        fitted_details,
        field="recommended_effective_footprint_m2_total",
    )

    violations_after_fit = _check_budget_limits(
        limits_m2=cluster_budget_limits_m2,
        footprints_m2=fitted_cluster_footprints,
    )
    global_violation_after_fit = max(0.0, fitted_global_footprint - global_budget_m2)

    return {
        "ratio": ratio,
        "base_ratio": base_ratio,
        "shape_compactness": compactness,
        "shape_factor": shape_factor,
        "cluster_area_mode": cluster_area_mode,
        "room_area_m2": room_area,
        "global_budget_m2": global_budget_m2,
        "global_requested_footprint_m2": sum(raw_cluster_footprints.values()),
        "global_recommended_footprint_m2": fitted_global_footprint,
        "cluster_ratios": cluster_ratios,
        "cluster_budget_limits_m2": cluster_budget_limits_m2,
        "cluster_requested_footprints_m2": raw_cluster_footprints,
        "cluster_recommended_footprints_m2": fitted_cluster_footprints,
        "cluster_requested_footprints_by_priority_m2": raw_breakdown,
        "cluster_recommended_footprints_by_priority_m2": fitted_breakdown,
        "violations_before_fit": violations_before_fit,
        "violations_after_fit": violations_after_fit,
        "global_violation_before_fit_m2": global_violation_before_fit,
        "global_violation_after_fit_m2": global_violation_after_fit,
        "decision_footprint_details": fitted_details,
        "missing_profile_categories": sorted(missing_profile_categories),
        "access_required_object_ids": sorted(access_requirements.object_ids),
        "access_required_categories": sorted(access_requirements.categories),
        "access_clearance_ratio_fixed": FIXED_ACCESS_CLEARANCE_RATIO,
        "access_clearance_rule": (
            "effective = L*W + L*clamp(W*(1/3), min_depth, max_depth)*occupancy_multiplier "
            "for decisions requiring front_clearance"
        ),
        "adaptive_cluster_ratios": adaptive_cluster_ratios,
        "cluster_ratio_overrides": cluster_ratio_overrides or {},
    }


# ============================================================
# Tool registry + schema
# ============================================================

TOOL_REGISTRY: dict[str, Any] = {
    "get_size_profiles": get_size_profiles,
    "estimate_budget": estimate_budget,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_size_profiles",
            "description": "Return representative dimensions (S/M/L) and thresholds for each category. Guarantees rep_dims_m[S/M/L] are non-null when category has inventory, and includes __generic__ fallback when possible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "tenant_id": {"type": ["string", "null"]},
                },
                "required": ["categories"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_budget",
            "description": (
                "Estimate cluster footprints (m^2), derive one global room budget, "
                "allocate per-cluster budget limits, and greedily trim optional/secondary quantities "
                "to fit space. Uses FIXED front-clearance ratio = 1/3."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room_model": {"type": "object"},
                    "decisions": {"type": "array", "items": {"type": "object"}},
                    "size_profiles_by_category": {"type": "object"},
                    "style": {"type": "string"},
                    "user_notes": {"type": "string"},
                    "ratio_override": {"type": ["number", "null"]},
                    "cluster_area_mode": {"type": "string", "enum": ["room", "zones"]},
                    "clusters_json": {"type": ["object", "null"]},
                    "adaptive_cluster_ratios": {"type": "boolean"},
                    "cluster_ratio_overrides": {
                        "type": ["object", "null"],
                        "additionalProperties": {"type": "number"},
                    },
                },
                "required": [
                    "room_model",
                    "decisions",
                    "size_profiles_by_category",
                    "style",
                    "user_notes",
                ],
            },
        },
    },
]
